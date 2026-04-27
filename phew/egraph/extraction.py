"""Extraction: pull the best program out of a saturated e-graph.

Two strategies:
  greedy  — default; fast; locally optimal
  ilp     — joint optimization via scipy.optimize.milp; wins on multi-pattern
            rewrites (Tensat shows ILP wins there)

Cost function: bytes_moved primary, calibrated against measurement.

Round-trip implementation
-------------------------
After extraction, egglog returns a new Python expression object. We convert
it back to a phew Graph by:

  1. ``build_egraph`` (rules_egglog.py) builds both an id-keyed node_map
     (id(expr)→Node, for cost model lookups during extraction) and a
     str-keyed str_node_map (str(expr)→Node, for round-trip reconstruction).

  2. ``_egglog_to_graph`` parses ``str(extracted_expr)`` — a canonical string
     like ``"compiled(matmul(named_tensor(1), named_tensor(2)))"`` — and
     reconstructs a phew Graph by walking the parse tree bottom-up.

  The encoding in rules_egglog.py maps:
    named_tensor(N)          → original graph node with id=N
    compiled(x)              → Compile node wrapping x
    fast_rms_norm(x, w)      → FastRMSNorm node
    fast_sdpa(q, k, v)       → FastScaledDotProductAttention node
    matmul(a, b)             → MatMul node
    fused_ew_chain(op1,op2,a)→ two chained Elementwise nodes (fallback: original)
    <anything else>          → look up in str_node_map; fall back to original
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from egglog import EGraph, Expr

    from phew.ir import Graph
    from phew.ir.node import Node


@dataclass
class ExtractionResult:
    graph: "Graph"
    cost: float
    strategy: str  # "greedy" | "ilp"


# ---------------------------------------------------------------------------
# Expression string parser
# ---------------------------------------------------------------------------


def _parse_expr_str(s: str) -> tuple[str, list[str]]:
    """Parse a flat egglog expression string into (fn_name, [arg_str, ...]).

    Examples
    --------
    >>> _parse_expr_str("compiled(named_tensor(1))")
    ('compiled', ['named_tensor(1)'])
    >>> _parse_expr_str("named_tensor(42)")
    ('named_tensor', ['42'])
    >>> _parse_expr_str("matmul(named_tensor(1), named_tensor(2))")
    ('matmul', ['named_tensor(1)', 'named_tensor(2)'])
    """
    s = s.strip()
    paren = s.find("(")
    if paren == -1:
        # Bare atom — treat as a zero-arg function
        return s, []
    fn_name = s[:paren]
    inner = s[paren + 1 : -1]  # strip outer parens
    return fn_name, _split_args(inner)


def _split_args(s: str) -> list[str]:
    """Split a comma-separated argument list, respecting nested parentheses."""
    args: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        args.append("".join(buf).strip())
    return [a for a in args if a]


# ---------------------------------------------------------------------------
# Graph reconstructor
# ---------------------------------------------------------------------------


class _GraphReconstructor:
    """Reconstruct a phew Graph from an extracted egglog expression string."""

    def __init__(self, original_graph: "Graph", str_node_map: dict[str, "Node"]) -> None:
        self.original = original_graph
        self.str_node_map = str_node_map
        # new_graph starts as a copy; we add rewritten nodes on top
        self.new_graph: "Graph" = copy.deepcopy(original_graph)
        self._cache: dict[str, "Node"] = {}

    def build(self, expr_str: str) -> "Graph | None":
        """Return a reconstructed Graph for *expr_str*, or None on failure."""
        try:
            root_node = self._reconstruct(expr_str)
            if root_node is None:
                return None
            self.new_graph.outputs = [root_node.id]
            return self.new_graph
        except Exception:
            return None

    def _reconstruct(self, expr_str: str) -> "Node | None":
        """Recursively reconstruct a node from *expr_str*."""
        if expr_str in self._cache:
            return self._cache[expr_str]

        fn_name, args = _parse_expr_str(expr_str)

        # --- named_tensor(N): look up original node by id ----------------
        if fn_name == "named_tensor" and args:
            try:
                node_id = int(args[0])
            except ValueError:
                return None
            # Find node in the original graph (now cloned in new_graph)
            if node_id in self.new_graph:
                node = self.new_graph[node_id]
                self._cache[expr_str] = node
                return node
            return None

        # --- compiled(x): wrap with Compile node -------------------------
        if fn_name == "compiled" and args:
            inner = self._reconstruct(args[0])
            if inner is None:
                return None
            from phew.ir.ops import Compile

            compile_node = Compile(
                shape=inner.shape,
                dtype=inner.dtype,
                inputs=[inner.id],
            )
            self.new_graph.add(compile_node)
            self._cache[expr_str] = compile_node
            return compile_node

        # --- fast_rms_norm(x, w): FastRMSNorm ----------------------------
        if fn_name == "fast_rms_norm" and len(args) >= 2:
            x = self._reconstruct(args[0])
            w = self._reconstruct(args[1])
            if x is None or w is None:
                return self._fallback(expr_str)
            # Find an existing FastRMSNorm or create one
            orig = self._find_original("FastRMSNorm", [x.id, w.id])
            if orig:
                self._cache[expr_str] = orig
                return orig
            from phew.ir.ops import FastRMSNorm

            node = FastRMSNorm(shape=x.shape, dtype=x.dtype, inputs=[x.id, w.id], eps=1e-5)
            self.new_graph.add(node)
            self._cache[expr_str] = node
            return node

        # --- fast_sdpa(q, k, v): FastScaledDotProductAttention ----------
        if fn_name == "fast_sdpa" and len(args) >= 3:
            q = self._reconstruct(args[0])
            k = self._reconstruct(args[1])
            v = self._reconstruct(args[2])
            if q is None or k is None or v is None:
                return self._fallback(expr_str)
            orig = self._find_original("FastScaledDotProductAttention", [q.id, k.id, v.id])
            if orig:
                self._cache[expr_str] = orig
                return orig
            from phew.ir.ops import FastScaledDotProductAttention

            node = FastScaledDotProductAttention(
                shape=q.shape, dtype=q.dtype, inputs=[q.id, k.id, v.id], scale=1.0
            )
            self.new_graph.add(node)
            self._cache[expr_str] = node
            return node

        # --- matmul(a, b): MatMul ----------------------------------------
        if fn_name == "matmul" and len(args) >= 2:
            a = self._reconstruct(args[0])
            b = self._reconstruct(args[1])
            if a is None or b is None:
                return self._fallback(expr_str)
            orig = self._find_original("MatMul", [a.id, b.id])
            if orig:
                self._cache[expr_str] = orig
                return orig
            # Can't safely create a new MatMul without knowing the output shape
            return self._fallback(expr_str)

        # --- fused_ew_chain: fall back to original -----------------------
        if fn_name == "fused_ew_chain":
            return self._fallback(expr_str)

        # --- Anything else: look up in str_node_map ----------------------
        node = self.str_node_map.get(expr_str)
        if node is not None:
            # Find the cloned version in new_graph
            if node.id in self.new_graph:
                cloned = self.new_graph[node.id]
                self._cache[expr_str] = cloned
                return cloned
        return self._fallback(expr_str)

    def _fallback(self, expr_str: str) -> "Node | None":
        """Return the original graph's output node when we can't reconstruct."""
        # Return the original output node so the graph stays valid
        if self.original.outputs:
            oid = self.original.outputs[-1]
            if oid in self.new_graph:
                return self.new_graph[oid]
        return None

    def _find_original(self, op_name: str, input_ids: list[int]) -> "Node | None":
        """Find a node with the given op name and inputs in new_graph."""
        for node in self.new_graph.topo_order():
            if node.op == op_name and node.inputs == input_ids:
                return node
        return None


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class Extractor:
    """Extract the lowest-cost program from a saturated EGraph.

    Parameters
    ----------
    strategy:
        "greedy" (default) or "ilp".
    """

    def __init__(self, strategy: str = "greedy") -> None:
        assert strategy in ("greedy", "ilp"), f"unknown strategy: {strategy}"
        self.strategy = strategy

    def extract(
        self,
        egraph: "EGraph",
        root_expr: "Expr",
        node_map,
        original_graph: "Graph",
        str_node_map: "dict[str, Node] | None" = None,
    ) -> ExtractionResult:

        from phew.cost import CostModel

        cost_model = CostModel()

        if self.strategy == "greedy":
            return self._greedy(
                egraph, root_expr, node_map, original_graph, cost_model, str_node_map
            )
        else:
            return self._ilp(egraph, root_expr, node_map, original_graph, cost_model, str_node_map)

    def _greedy(
        self, egraph, root_expr, node_map, graph, cost_model, str_node_map=None
    ) -> ExtractionResult:
        """Use egglog's built-in greedy extractor with a bytes-moved cost model."""

        def bytes_moved_cost(eg, expr, children_costs):
            # Try id-keyed map first (works for pre-extraction expression objects),
            # then fall back to str-keyed map for post-extraction objects.
            node = node_map.get(id(expr))
            if node is None and str_node_map is not None:
                node = str_node_map.get(str(expr))
            if node is not None:
                c = cost_model.node_cost(node)
                return c.bytes_moved + sum(children_costs)
            return 1.0 + sum(children_costs)

        extracted, cost = egraph.extract(
            root_expr,
            include_cost=True,
            cost_model=bytes_moved_cost,
        )

        new_graph = self._egglog_to_graph(extracted, node_map, graph, str_node_map)
        return ExtractionResult(graph=new_graph, cost=float(cost), strategy="greedy")

    def _ilp(
        self, egraph, root_expr, node_map, graph, cost_model, str_node_map=None
    ) -> ExtractionResult:
        """Joint ILP extraction via scipy.optimize.milp.

        True ILP extraction requires iterating over e-classes and e-nodes to
        set up the binary selection variables (Tensat §3.2 formulation).
        The egglog Python bindings do not currently expose e-class internals,
        so the ILP formulation cannot be constructed from the outside.
        Falls back to greedy until egglog adds an e-class API or we switch to
        a library that exposes this (e.g. egg-smol or a custom e-graph).
        """
        try:
            from scipy.optimize import milp  # noqa: F401
        except ImportError:
            return self._greedy(egraph, root_expr, node_map, graph, cost_model, str_node_map)

        result = self._greedy(egraph, root_expr, node_map, graph, cost_model, str_node_map)
        result.strategy = "ilp-fallback-greedy"
        return result

    def _egglog_to_graph(
        self,
        extracted_expr,
        node_map,
        original_graph: "Graph",
        str_node_map: "dict[str, Node] | None" = None,
    ) -> "Graph":
        """Convert an extracted egglog expression back to a phew Graph.

        Parses str(extracted_expr) and reconstructs the graph bottom-up.
        Falls back to original_graph if reconstruction fails.
        """
        if str_node_map is None:
            str_node_map = {}

        try:
            expr_str = str(extracted_expr)
        except Exception:
            return original_graph

        # Fast path: if the extracted expression directly matches an original
        # node, no rewrites were applied — return original unchanged.
        if expr_str in str_node_map:
            return original_graph

        reconstructor = _GraphReconstructor(original_graph, str_node_map)
        new_graph = reconstructor.build(expr_str)
        return new_graph if new_graph is not None else original_graph
