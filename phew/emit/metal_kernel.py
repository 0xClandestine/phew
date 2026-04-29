"""Phase-2 kernel parameter search.

Triggered when: single kernel > 30% GPU time AND not replaceable by fast.*.

Searchable knobs (from the spec):
  Threadgroup 1D  : {64, 128, 256, 512, 1024}
  Threadgroup 2D  : {16×16, 32×8, 8×32}
  Tile (matmul)   : 8×8, 16×16, 32×32
  simdgroup prims : simd_sum, simd_max, simd_prefix_inclusive_sum
  Memory placement: threadgroup-mem vs device-mem
  Address space   : constant vs device (read-only)
  Vector width    : scalar / float2 / float4
  Loop unroll     : {1, 2, 4, 8}
  TensorOps tile  : M, N as compile-time constants (M5+ only)

Search strategy:
  1. Cost-model prune before measurement
  2. Benchmark surviving candidates
  3. Verify equivalence
  4. Keep best by on-device measurement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from phew.ir import MetalKernel
    from phew.trace import ProfileData


THREADGROUP_1D = [64, 128, 256, 512, 1024]
THREADGROUP_2D = [(16, 16), (32, 8), (8, 32)]
TILE_SIZES = [(8, 8), (16, 16), (32, 32)]
VECTOR_WIDTHS = [1, 2, 4]
LOOP_UNROLLS = [1, 2, 4, 8]


@dataclass
class KernelCandidate:
    """A single parameterization of a Metal kernel."""

    threadgroup: tuple[int, ...]
    tile: tuple[int, int] | None
    vector_width: int
    loop_unroll: int
    use_threadgroup_mem: bool
    use_constant_address: bool
    template_params: list[tuple[str, object]] = field(default_factory=list)
    source: str = ""
    measured_ms: float | None = None
    verified: bool = False
    pruned: bool = False
    prune_reason: str = ""


class KernelParamSearch:
    """Phase-2 parameter search for a hot kernel.

    Parameters
    ----------
    baseline_fn:
        The original kernel as a callable.
    max_candidates:
        Maximum candidates to benchmark (after cost-model pruning).
    """

    def __init__(
        self,
        baseline_fn: Callable | None = None,
        max_candidates: int = 50,
    ) -> None:
        self.baseline_fn = baseline_fn
        self.max_candidates = max_candidates

    def search(
        self,
        kernel_node: "MetalKernel",
        input_factory: Callable[[str, int], tuple[list, dict]],
        profile_data: "ProfileData | None" = None,
    ) -> list[KernelCandidate]:
        """Generate, prune, benchmark, and verify candidates.

        Returns candidates sorted by measured_ms (best first).
        """
        from phew.bench import benchmark
        from phew.cost import CostModel, should_prune
        from phew.verify import EquivalenceChecker

        cost_model = CostModel()
        baseline_cost = cost_model.node_cost(kernel_node)

        # Enumerate candidates
        all_candidates = list(self._enumerate(kernel_node))

        # Prune
        surviving: list[KernelCandidate] = []
        for cand in all_candidates:
            tg_size = cand.threadgroup[0] * (
                cand.threadgroup[1] if len(cand.threadgroup) > 1 else 1
            )
            prune, reason = should_prune(
                baseline_cost,
                baseline_cost,
                threadgroup_size=tg_size,
            )
            if prune:
                cand.pruned = True
                cand.prune_reason = reason
            else:
                surviving.append(cand)

        surviving = surviving[: self.max_candidates]

        # Benchmark survivors
        best_ms = float("inf")
        for cand in surviving:
            try:
                fn = self._build_kernel_fn(kernel_node, cand)
                args, kwargs = input_factory("typical", 0)
                result = benchmark(fn, *args, n_warmup=3, n_bench=10, **kwargs)
                cand.measured_ms = result.mean_ms
                if result.mean_ms < best_ms:
                    best_ms = result.mean_ms
            except Exception as exc:
                cand.prune_reason = str(exc)
                cand.pruned = True

        # Verify top-N survivors against baseline
        checker = EquivalenceChecker()
        measured = [c for c in surviving if c.measured_ms is not None]
        measured.sort(key=lambda c: c.measured_ms)

        for cand in measured[:5]:
            if self.baseline_fn is None:
                cand.verified = True
                continue
            fn = self._build_kernel_fn(kernel_node, cand)
            result = checker.check(self.baseline_fn, fn, input_factory)
            cand.verified = result.passed

        return [c for c in measured if c.verified]

    def _enumerate(self, node: "MetalKernel"):
        """Yield all valid parameterizations."""
        for tg in THREADGROUP_1D:
            for vw in VECTOR_WIDTHS:
                for unroll in LOOP_UNROLLS:
                    for use_tg_mem in (False, True):
                        yield KernelCandidate(
                            threadgroup=(tg,),
                            tile=None,
                            vector_width=vw,
                            loop_unroll=unroll,
                            use_threadgroup_mem=use_tg_mem,
                            use_constant_address=False,
                            template_params=[
                                ("THREADGROUP", tg),
                                ("VW", vw),
                                ("UNROLL", unroll),
                            ],
                        )

    def _build_kernel_fn(self, node: "MetalKernel", cand: KernelCandidate) -> Callable:
        """Build a callable that runs the kernel with the given parameters.

        Threadgroup size is varied via the ``threadgroup=`` call parameter,
        which always takes effect.  VW and UNROLL are injected as MSL template
        constants only when the kernel source actually references those names —
        otherwise they compile away silently and the search result would be
        meaningless.
        """
        import mlx.core as mx

        source = node.source or cand.source
        header = node.header
        input_names = node.input_names
        output_names = node.output_names
        output_shapes = node.output_shapes
        output_dtypes = node.output_dtypes

        # Only pass template constants that the source actually references.
        template = [(k, v) for k, v in cand.template_params if k in source]
        tg = cand.threadgroup

        # Import here to avoid issues when MLX is not installed
        kernel = mx.fast.metal_kernel(
            name=f"search_kernel_{id(cand)}",
            input_names=input_names,
            output_names=output_names,
            source=source,
            header=header,
        )

        node_grid = node.grid

        def fn(*inputs):
            return kernel(
                inputs=list(inputs),
                output_shapes=output_shapes,
                output_dtypes=[getattr(mx, d.to_mlx()) for d in output_dtypes],
                grid=node_grid
                if node_grid is not None
                else (inputs[0].size if inputs else 1, 1, 1),
                threadgroup=tg if len(tg) == 3 else (tg[0], 1, 1),
                template=template,
            )

        return fn
