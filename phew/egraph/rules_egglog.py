"""Egglog expression types and rewrite rules for each rule class.

Each op in the μGraph IR maps to an egglog Expr subclass. Rules are organized
into named "rule sets" that can be selectively enabled based on the profiler's
bottleneck classification.

Rule sets:
  algebraic          — assoc/commut, const fold, cast elim, DCE
  fusion             — elementwise+elementwise, elem+reduce, matmul+epilogue, softmax
  layout             — axis perm, contiguity, transpose-through-matmul
  precision          — fp32→fp16/bf16 (opt-in)
  primitive_subst    — fast.* pattern matching
  quantization       — matmul → quantized_matmul (opt-in)
  compile_boundaries — mx.compile, async_eval, vmap
  tensorops          — M5 TensorOps substitution (hardware-gated)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from phew.ir import Graph, Node
    from phew.trace import BottleneckClass

# ---------------------------------------------------------------------------
# Egglog type and function definitions (module-level so get_type_hints works)
# ---------------------------------------------------------------------------

from egglog import Expr, eq, function, i64, rewrite, rule, set_, union, vars_


class Tensor(Expr):
    """An abstract tensor value in the e-graph."""

    def __add__(self, other: Tensor) -> Tensor: ...
    def __mul__(self, other: Tensor) -> Tensor: ...
    def __sub__(self, other: Tensor) -> Tensor: ...
    def __truediv__(self, other: Tensor) -> Tensor: ...


class TensorList(Expr):
    """Ordered list of tensors (for multi-output ops)."""


@function
def named_tensor(n: i64) -> Tensor: ...


# Shape facts for the last two matrix dimensions.
# Populated by build_egraph for every node; derived by propagation rules for
# new matmul expressions created during saturation.
@function
def mat_rows(t: Tensor) -> i64: ...


@function
def mat_cols(t: Tensor) -> i64: ...


@function
def matmul(a: Tensor, b: Tensor) -> Tensor: ...


@function
def matmul_t(a: Tensor, b: Tensor, trans_a: i64, trans_b: i64) -> Tensor: ...


@function
def reduce(x: Tensor, op: i64, axis: i64, keepdims: i64) -> Tensor: ...


@function
def cast(x: Tensor, dtype: i64) -> Tensor: ...


@function
def transpose(x: Tensor, axes: i64) -> Tensor: ...


@function
def reshape(x: Tensor, shape: i64) -> Tensor: ...


@function
def elementwise(op: i64, a: Tensor) -> Tensor: ...


@function
def elementwise2(op: i64, a: Tensor, b: Tensor) -> Tensor: ...


@function(cost=1)
def fast_rms_norm(x: Tensor, weight: Tensor) -> Tensor: ...


@function(cost=1)
def fast_layer_norm(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor: ...


@function(cost=1)
def fast_rope(x: Tensor, offset: i64) -> Tensor: ...


@function(cost=1)
def fast_sdpa(q: Tensor, k: Tensor, v: Tensor) -> Tensor: ...


@function(cost=2)
def quantized_matmul(x: Tensor, w: Tensor, bits: i64) -> Tensor: ...


@function(cost=1)
def compiled(x: Tensor) -> Tensor: ...


@function(cost=1)
def fused_ew_chain(op1: i64, op2: i64, a: Tensor) -> Tensor: ...


# ---------------------------------------------------------------------------
# Build egraph from a phew Graph
# ---------------------------------------------------------------------------


def build_egraph(egraph, graph: "Graph") -> tuple:
    """Encode a phew Graph into egglog expressions.

    Returns (root_exprs, node_map, str_node_map) where:
      root_exprs   — list of egglog Exprs, one per graph output (single-output
                     graphs still return a one-element list for uniformity)
      node_map     — dict mapping id(egglog_expr) → phew Node  (for cost model)
      str_node_map — dict mapping str(egglog_expr) → phew Node  (for round-trip)
    """
    from egglog import i64 as ei64

    from phew.ir import (
        Cast,
        Compile,
        Constant,
        Elementwise,
        FastLayerNorm,
        FastRMSNorm,
        FastRoPE,
        FastScaledDotProductAttention,
        Input,
        MatMul,
        QuantizedMatMul,
        Reduce,
        Reshape,
        Transpose,
    )

    node_map: dict[int, "Node"] = {}
    str_node_map: dict[str, "Node"] = {}
    expr_cache: dict[int, object] = {}  # node_id → egglog expr

    def encode(node_id: int):
        if node_id in expr_cache:
            return expr_cache[node_id]
        node = graph[node_id]
        ins = [encode(i) for i in node.inputs]

        if isinstance(node, (Input, Constant)):
            expr = named_tensor(ei64(node.id))
        elif isinstance(node, MatMul):
            expr = matmul(ins[0], ins[1])
        elif isinstance(node, Reduce):
            expr = reduce(ins[0], ei64(0), ei64(0), ei64(int(node.keepdims)))
        elif isinstance(node, Elementwise):
            if len(ins) == 1:
                expr = elementwise(ei64(0), ins[0])
            else:
                expr = elementwise2(ei64(0), ins[0], ins[1])
        elif isinstance(node, Cast):
            expr = cast(ins[0], ei64(0))
        elif isinstance(node, (Transpose, Reshape)):
            expr = transpose(ins[0], ei64(0))
        elif isinstance(node, FastRMSNorm):
            expr = fast_rms_norm(ins[0], ins[1] if len(ins) > 1 else ins[0])
        elif isinstance(node, FastLayerNorm):
            expr = fast_layer_norm(
                ins[0], ins[1] if len(ins) > 1 else ins[0], ins[2] if len(ins) > 2 else ins[0]
            )
        elif isinstance(node, FastRoPE):
            expr = fast_rope(ins[0], ei64(node.offset))
        elif isinstance(node, FastScaledDotProductAttention):
            expr = fast_sdpa(ins[0], ins[1], ins[2])
        elif isinstance(node, QuantizedMatMul):
            expr = quantized_matmul(ins[0], ins[1], ei64(node.bits))
        elif isinstance(node, Compile):
            expr = compiled(ins[0])
        else:
            expr = named_tensor(ei64(node.id))

        egraph.register(expr)
        expr_cache[node_id] = expr
        # id-keyed for cost model lookups during extraction
        node_map[id(expr)] = node
        # str-keyed for round-trip reconstruction after extraction
        try:
            str_node_map[str(expr)] = node
        except Exception:
            pass

        # Register shape facts so conditional matmul rules can fire correctly.
        # mat_rows / mat_cols track the last two dimensions of the node's shape.
        # i64 is a primitive sort — use set_() not union().
        sh = node.shape
        if sh and len(sh) >= 2:
            egraph.register(
                set_(mat_rows(expr)).to(ei64(sh[-2])),
                set_(mat_cols(expr)).to(ei64(sh[-1])),
            )
        elif sh and len(sh) == 1:
            egraph.register(
                set_(mat_rows(expr)).to(ei64(1)),
                set_(mat_cols(expr)).to(ei64(sh[-1])),
            )

        return expr

    if graph.outputs:
        # Encode ALL outputs so the e-graph sees the full computation.
        # Multi-output functions (e.g. returning (y, state)) previously only
        # encoded the last output, causing the extractor to drop all but the
        # last return value and produce a shape-mismatched candidate.
        root_exprs = [encode(oid) for oid in graph.outputs]
    else:
        for node in graph.topo_order():
            encode(node.id)
        root_exprs = [list(expr_cache.values())[-1]] if expr_cache else []

    return root_exprs, node_map, str_node_map


# ---------------------------------------------------------------------------
# Rule set functions — use vars_() + egraph.register(rewrite(...)) style
# to avoid get_type_hints issues with the generator/decorator pattern.
# ---------------------------------------------------------------------------


def _register_algebraic_rules(egraph) -> None:
    a, b, c = vars_("a b c", Tensor)
    egraph.register(
        rewrite(a + b).to(b + a),
        rewrite((a + b) + c).to(a + (b + c)),
        rewrite(a * b).to(b * a),
    )

    # Shape-propagation rule for matmul: when matmul(a, b) already exists and we
    # know the shapes of a and b, derive the shape of the result.
    # Guard on `eq(ab_sp).to(matmul(a, b))` so we MATCH an existing matmul rather
    # than CREATE a new one for every shape-compatible pair of tensors.
    (ab_sp,) = vars_("ab_sp", Tensor)
    m, k, n = vars_("m k n", i64)
    egraph.register(
        rule(
            eq(ab_sp).to(matmul(a, b)),
            eq(mat_rows(a)).to(m),
            eq(mat_cols(a)).to(k),
            eq(mat_rows(b)).to(k),
            eq(mat_cols(b)).to(n),
        ).then(
            set_(mat_rows(ab_sp)).to(m),
            set_(mat_cols(ab_sp)).to(n),
        )
    )

    # Matmul associativity: (A @ B) @ C  ↔  A @ (B @ C)
    # Both directions require the input expression to already exist so the rule
    # expands an existing matmul rather than generating new ones for every
    # shape-compatible triple, which would cause unbounded e-graph growth.
    ab_v, abc_l = vars_("ab_v abc_l", Tensor)
    bc_v, abc_r = vars_("bc_v abc_r", Tensor)
    m2, k2, n2, p2 = vars_("m2 k2 n2 p2", i64)

    # (a @ b) @ c  →  a @ (b @ c)
    egraph.register(
        rule(
            eq(ab_v).to(matmul(a, b)),
            eq(abc_l).to(matmul(ab_v, c)),
            eq(mat_rows(a)).to(m2),
            eq(mat_cols(a)).to(k2),
            eq(mat_rows(b)).to(k2),
            eq(mat_cols(b)).to(n2),
            eq(mat_rows(c)).to(n2),
            eq(mat_cols(c)).to(p2),
        ).then(
            union(abc_l).with_(matmul(a, matmul(b, c))),
        )
    )

    # a @ (b @ c)  →  (a @ b) @ c
    egraph.register(
        rule(
            eq(bc_v).to(matmul(b, c)),
            eq(abc_r).to(matmul(a, bc_v)),
            eq(mat_rows(a)).to(m2),
            eq(mat_cols(a)).to(k2),
            eq(mat_rows(b)).to(k2),
            eq(mat_cols(b)).to(n2),
            eq(mat_rows(c)).to(n2),
            eq(mat_cols(c)).to(p2),
        ).then(
            union(abc_r).with_(matmul(matmul(a, b), c)),
        )
    )


def _register_fusion_rules(egraph) -> None:
    # Fuse consecutive unary elementwise ops into a single kernel pass.
    # The round-trip back to Graph (extraction.py) is not yet implemented,
    # so these rules inform cost estimation only; emitted code is unaffected.
    (a,) = vars_("a", Tensor)
    op1, op2 = vars_("op1 op2", i64)
    egraph.register(
        rewrite(elementwise(op2, elementwise(op1, a))).to(fused_ew_chain(op1, op2, a)),
    )


def _register_layout_rules(egraph) -> None:
    from egglog import i64 as ei64

    a, b = vars_("a b", Tensor)
    egraph.register(
        rewrite(transpose(matmul(a, b), ei64(0))).to(
            matmul(transpose(b, ei64(0)), transpose(a, ei64(0)))
        ),
    )


def _register_precision_rules(egraph) -> None:
    from egglog import i64 as ei64

    (a,) = vars_("a", Tensor)
    DTYPE_FP32 = ei64(0)
    egraph.register(
        rewrite(cast(cast(a, DTYPE_FP32), DTYPE_FP32)).to(a),
    )


def _register_primitive_subst_rules(egraph) -> None:
    # Primitive substitution (rms_norm, layer_norm, rope, sdpa) runs as a
    # structural graph pass before egraph encoding (rules/primitive_subst.py).
    # It cannot be expressed as egglog rewrites without shape-aware types,
    # because the current Tensor encoding carries no shape or op-id information.
    # Fast.* nodes inserted by the graph pass are encoded directly by
    # build_egraph() with cost=1 and will be preferred by the extractor.
    pass


def _register_quantization_rules(egraph) -> None:
    from egglog import i64 as ei64

    x, w = vars_("x w", Tensor)
    egraph.register(
        rewrite(matmul(x, w)).to(quantized_matmul(x, w, ei64(4))),
    )


def _register_compile_boundary_rules(egraph) -> None:
    a, b = vars_("a b", Tensor)
    egraph.register(
        rewrite(matmul(a, b)).to(compiled(matmul(a, b))),
    )


def _register_tensorops_rules(egraph) -> None:
    # TensorOps patterns are detected at the Graph level in rules/tensorops.py.
    pass


# ---------------------------------------------------------------------------
# Rule set registry and bottleneck routing
# ---------------------------------------------------------------------------

RULE_SETS: dict[str, Callable] = {
    "algebraic": _register_algebraic_rules,
    "fusion": _register_fusion_rules,
    "layout": _register_layout_rules,
    "precision": _register_precision_rules,
    "primitive_subst": _register_primitive_subst_rules,
    "quantization": _register_quantization_rules,
    "compile_boundaries": _register_compile_boundary_rules,
    "tensorops": _register_tensorops_rules,
}

# Which rule sets each bottleneck class activates (search prior from spec)
_BOTTLENECK_RULES = {
    "memory_bound": [
        "algebraic",
        "fusion",
        "layout",
        "precision",
        "primitive_subst",
        "compile_boundaries",
    ],
    "compute_bound": ["algebraic", "fusion", "quantization", "tensorops", "compile_boundaries"],
    "occupancy_limited": ["algebraic", "compile_boundaries"],
    "launch_overhead": ["compile_boundaries", "fusion"],
    None: list(RULE_SETS.keys()),  # no trace → try all
}


def bottleneck_rule_sets(
    bottleneck: "BottleneckClass | None",
    enabled_subst_classes: set,
) -> list[str]:
    from phew.verify import SubstitutionClass

    key = bottleneck.value if bottleneck is not None else None
    rules = list(_BOTTLENECK_RULES.get(key, list(RULE_SETS.keys())))

    if (
        SubstitutionClass.fp32_to_fp16 not in enabled_subst_classes
        and SubstitutionClass.fp32_to_bf16 not in enabled_subst_classes
    ):
        rules = [r for r in rules if r != "precision"]

    if SubstitutionClass.quantized_4bit not in enabled_subst_classes:
        rules = [r for r in rules if r != "quantization"]

    return rules
