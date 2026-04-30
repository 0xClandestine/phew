"""E-graph saturation layer wrapping egglog-python.

We represent each MLX op as an egglog Expr class and register rewrite rules
that encode the algebraic, fusion, layout, precision, primitive-substitution,
quantization, compile-boundary, and TensorOps rule classes from the spec.

Saturation workflow:
  1. Build EGraph from a phew.ir.Graph
  2. Register all applicable rule sets (gated by profiler class + user opt-ins)
  3. egraph.run(iterations) / egraph.saturate()
  4. Delegate to Extractor for extraction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phew.ir import Graph
    from phew.trace import BottleneckClass


@dataclass
class SaturationResult:
    iterations: int
    nodes_before: int
    nodes_after: int
    classes_after: int
    rule_stats: dict[str, int] = field(default_factory=dict)  # rule_name → applications


class EGraphSaturator:
    """Drive egglog saturation over a phew.ir.Graph.

    Parameters
    ----------
    max_iterations:
        Hard cap on saturation iterations. egglog will stop earlier if it
        reaches a fixpoint.
    bottleneck:
        Profiler-determined bottleneck class; used to prune inapplicable rule sets.
    enabled_subst_classes:
        Precision-reducing substitution classes that the user has opted in to.
    """

    def __init__(
        self,
        max_iterations: int = 30,
        bottleneck: "BottleneckClass | None" = None,
        enabled_subst_classes: set | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.bottleneck = bottleneck
        self.enabled_subst_classes = enabled_subst_classes or set()

    def saturate(self, graph: "Graph") -> tuple["Graph", SaturationResult]:
        """Run equality saturation and return the (potentially unchanged) graph
        along with saturation statistics.

        The heavy lifting is done by the egglog EGraph. We:
          1. Encode the input graph as egglog expressions.
          2. Register rule sets appropriate for the bottleneck class.
          3. Run saturation.
          4. Return the graph + stats; extraction is handled separately.
        """
        from egglog import EGraph

        from .rules_egglog import RULE_SETS, bottleneck_rule_sets, build_egraph

        egraph = EGraph()
        root_exprs, node_map, str_node_map = build_egraph(egraph, graph)

        # Select which rule sets to register based on bottleneck
        active_sets = bottleneck_rule_sets(
            self.bottleneck,
            self.enabled_subst_classes,
        )

        for rule_set_name in active_sets:
            rule_fn = RULE_SETS.get(rule_set_name)
            if rule_fn is not None:
                rule_fn(egraph)

        # Run saturation
        egraph.run(self.max_iterations)

        stats = SaturationResult(
            iterations=self.max_iterations,
            nodes_before=len(graph),
            nodes_after=len(graph),  # updated by extractor
            classes_after=0,
            rule_stats={name: 0 for name in active_sets},
        )
        return graph, stats, egraph, root_exprs, node_map, str_node_map
