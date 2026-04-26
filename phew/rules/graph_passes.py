"""Run all graph-level passes in priority order.

Search prior (highest-yield first):
  1. mx.compile whole function (CompileBoundaryPass)
  2. Primitive substitution (PrimitiveSubstPass) — fast.* matches end search
  3. TensorOps (TensorOpsPass) — hardware-gated

Egglog rules (algebraic/fusion/layout/precision/quant) run inside the
EGraphSaturator after these graph passes complete.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phew.ir import Graph


def run_all_passes(
    graph: "Graph",
    *,
    enable_compile: bool = True,
    enable_primitive_subst: bool = True,
    enable_tensorops: bool = True,
    enable_fusion: bool = False,
) -> tuple["Graph", list[str]]:
    """Run graph-level passes. Return (graph, list_of_applied_pass_names)."""
    from .compile_boundaries import CompileBoundaryPass
    from .fusion import ElementwiseFusionPass
    from .primitive_subst import PrimitiveSubstPass
    from .tensorops import TensorOpsPass

    applied: list[str] = []

    if enable_fusion:
        if ElementwiseFusionPass().run(graph):
            applied.append("elementwise_fusion")

    if enable_compile:
        if CompileBoundaryPass().run(graph):
            applied.append("compile_boundary")

    if enable_primitive_subst:
        if PrimitiveSubstPass().run(graph):
            applied.append("primitive_subst")
            # Primitive subst ends Phase-1 for matched subgraphs — no eggsat needed
            return graph, applied

    if enable_tensorops:
        if TensorOpsPass().run(graph):
            applied.append("tensorops")

    return graph, applied
