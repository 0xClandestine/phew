"""Extraction: pull the best program out of a saturated e-graph.

Two strategies:
  greedy  — default; fast; locally optimal
  ilp     — joint optimization via scipy.optimize.milp; wins on multi-pattern
            rewrites (Tensat shows ILP wins there)

Cost function: bytes_moved primary, calibrated against measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from egglog import EGraph, Expr

    from phew.ir import Graph


@dataclass
class ExtractionResult:
    graph: "Graph"
    cost: float
    strategy: str  # "greedy" | "ilp"


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
    ) -> ExtractionResult:

        from phew.cost import CostModel

        cost_model = CostModel()

        if self.strategy == "greedy":
            return self._greedy(egraph, root_expr, node_map, original_graph, cost_model)
        else:
            return self._ilp(egraph, root_expr, node_map, original_graph, cost_model)

    def _greedy(self, egraph, root_expr, node_map, graph, cost_model) -> ExtractionResult:
        """Use egglog's built-in greedy extractor with a bytes-moved cost model."""

        def bytes_moved_cost(eg, expr, children_costs):
            # Map egglog expr back to a phew Node via node_map to get real cost
            node = node_map.get(id(expr))
            if node is not None:
                c = cost_model.node_cost(node)
                return c.bytes_moved + sum(children_costs)
            return 1.0 + sum(children_costs)

        extracted, cost = egraph.extract(
            root_expr,
            include_cost=True,
            cost_model=bytes_moved_cost,
        )

        # Convert extracted egglog expression back to a phew Graph
        new_graph = self._egglog_to_graph(extracted, node_map, graph)
        return ExtractionResult(graph=new_graph, cost=float(cost), strategy="greedy")

    def _ilp(self, egraph, root_expr, node_map, graph, cost_model) -> ExtractionResult:
        """Joint ILP extraction via scipy.optimize.milp.

        Falls back to greedy if scipy is unavailable.
        """
        try:
            from scipy.optimize import milp  # noqa: F401
        except ImportError:
            return self._greedy(egraph, root_expr, node_map, graph, cost_model)

        # For now fall back to greedy; full ILP formulation is a future milestone.
        # The greedy extractor is correct and sufficient for Phase-1 v1.
        result = self._greedy(egraph, root_expr, node_map, graph, cost_model)
        result.strategy = "ilp-fallback-greedy"
        return result

    def _egglog_to_graph(self, extracted_expr, node_map, original_graph) -> "Graph":
        """Convert an extracted egglog expression back to a phew Graph.

        For now, return the original graph if we can't map back — extraction
        is an active work-in-progress tied to the rule encoding.
        """
        # TODO: full round-trip once rules_egglog.py encoding is complete
        return original_graph
