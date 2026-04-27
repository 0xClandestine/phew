"""Run all graph-level passes in priority order.

Search prior (highest-yield first):
  1. mx.compile whole function (CompileBoundaryPass)
  2. Primitive substitution (PrimitiveSubstPass) — fast.* matches end search
  3. TensorOps (TensorOpsPass) — hardware-gated

Opt-in passes (require explicit enable_*=True):
  - PrecisionPass: fp32 → bf16 (halves bandwidth, ~2× for memory-bound graphs)
  - QuantizationPass: MatMul → QuantizedMatMul (4-bit weights)

Egglog rules (algebraic/fusion/layout) run inside the EGraphSaturator after
these graph passes complete.
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
    enable_algebraic: bool = True,
    enable_precision: bool = False,
    enable_quantization: bool = False,
    precision_dtype=None,
    quantization_bits: int = 4,
    quantization_group_size: int = 64,
) -> tuple["Graph", list[str]]:
    """Run graph-level passes. Return (graph, list_of_applied_pass_names).

    Opt-in parameters
    -----------------
    enable_precision:
        Insert fp32→bf16 casts at inputs and bf16→fp32 casts at outputs.
        Halves memory bandwidth for activation tensors.
    enable_quantization:
        Replace MatMul(x, W) with QuantizedMatMul when W is a parameter/constant.
    precision_dtype:
        Target dtype for PrecisionPass. Defaults to Dtype.bfloat16.
    quantization_bits:
        Bit-width for QuantizationPass (4 or 8). Default 4.
    quantization_group_size:
        Group size for group-wise quantization. Default 64.
    """
    from .algebraic import AlgebraicPass
    from .compile_boundaries import CompileBoundaryPass
    from .fusion import ElementwiseFusionPass
    from .primitive_subst import PrimitiveSubstPass
    from .quantization import QuantizationPass
    from .tensorops import TensorOpsPass

    applied: list[str] = []

    # Algebraic simplification runs first: eliminates redundant casts/transposes/reshapes
    # before other passes run, reducing graph size and improving pattern matching.
    if enable_algebraic:
        before = len(graph)
        AlgebraicPass().run(graph)
        if len(graph) != before:
            applied.append("algebraic")

    # Opt-in: precision reduction (fp32 → bf16) — do this first so subsequent
    # passes see the reduced-precision graph and can make better decisions.
    if enable_precision:
        from phew.ir.dtype import Dtype  # noqa: PLC0415

        from .precision import PrecisionPass  # noqa: PLC0415

        target = precision_dtype or Dtype.bfloat16
        before = len(graph)
        PrecisionPass(target_dtype=target).run(graph)
        if len(graph) != before:
            applied.append("precision")

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

    # Opt-in: 4-bit quantization — run after primitive_subst so we don't
    # quantize matmuls that were already replaced with fast.* ops.
    if enable_quantization:
        before = len(graph)
        QuantizationPass(bits=quantization_bits, group_size=quantization_group_size).run(graph)
        if len(graph) != before:
            applied.append("quantization")

    if enable_tensorops:
        if TensorOpsPass().run(graph):
            applied.append("tensorops")

    return graph, applied
