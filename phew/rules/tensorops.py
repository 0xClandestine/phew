"""TensorOps substitution pass (M5/A19 hardware).

Substitutes eligible matmul subgraphs with Metal Performance Primitives
TensorOps kernels when mx.device_info() reports tensor-op support.

From the spec: "when mx.device_info() reports tensor-op support,
substitute matmul subgraphs with TensorOps via Metal Performance Primitives.
Up to 4× prefill on M5 vs M4."

Only certain M/N tile combinations work (Zakharyo 2025: too small/large →
poor perf or compile error). We conservatively restrict to known-good tiles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phew.ir import Graph


# Known-good TensorOps tile dimensions (M, N)
VALID_TENSOROPS_TILES = {
    (8, 8),
    (16, 16),
    (32, 32),
    (8, 16),
    (16, 8),
}

# Minimum problem size to benefit from TensorOps
MIN_MATMUL_SIZE = 64  # M * N >= 64


def _arch_generation(arch: str) -> int:
    """Extract numeric GPU generation from architecture string.

    MLX device_info()["architecture"] format: "applegpu_gNNx"
      g13 = M1/A15, g14 = M2/A16, g15 = M3/A17, g16 = M4/A18, g17 = M5/A19
    """
    try:
        after_g = arch.split("_g")[-1]  # e.g. "16p" or "15s"
        return int("".join(c for c in after_g if c.isdigit()))
    except (IndexError, ValueError):
        return 0


def has_tensorops_support() -> bool:
    """Return True if the current device supports TensorOps (M5/A19+, gen ≥ 17)."""
    try:
        import mlx.core as mx

        if not mx.metal.is_available():
            return False
        info = mx.device_info()
        arch = info.get("architecture", "")
        # M5/A19 is generation g17; TensorOps via Metal Performance Primitives
        # are documented as available from that generation onward.
        return _arch_generation(arch) >= 17
    except Exception:
        return False


class TensorOpsPass:
    """Replace eligible MatMul nodes with TensorOps MetalKernel nodes."""

    def run(self, graph: "Graph") -> bool:
        if not has_tensorops_support():
            return False

        from phew.ir import MatMul, MetalKernel

        changed = False

        for node in list(graph.topo_order()):
            if not isinstance(node, MatMul):
                continue
            if len(node.shape) < 2:
                continue

            M, N = node.shape[-2], node.shape[-1]
            if M * N < MIN_MATMUL_SIZE:
                continue

            # Find best tile
            tile = self._best_tile(M, N)
            if tile is None:
                continue

            tile_m, tile_n = tile
            metal_node = MetalKernel(
                shape=node.shape,
                dtype=node.dtype,
                inputs=list(node.inputs),
                source=self._gen_tensorops_source(tile_m, tile_n, node.dtype),
                header=self._tensorops_header(),
                template_params=[
                    ("T", node.dtype.to_mlx()),
                    ("TILE_M", tile_m),
                    ("TILE_N", tile_n),
                ],
                threadgroup=(32, 1, 1),
                input_names=["a", "b"],
                output_names=["out"],
                output_shapes=[node.shape],
                output_dtypes=[node.dtype],
            )
            graph.replace(node.id, metal_node)
            changed = True

        return changed

    def _best_tile(self, M: int, N: int) -> tuple[int, int] | None:
        for tm, tn in sorted(VALID_TENSOROPS_TILES, key=lambda t: t[0] * t[1], reverse=True):
            if M % tm == 0 and N % tn == 0:
                return tm, tn
        return None

    def _tensorops_header(self) -> str:
        return "#include <metal_stdlib>\nusing namespace metal;\n"

    def _gen_tensorops_source(self, tile_m: int, tile_n: int, dtype) -> str:
        # Placeholder: real TensorOps source uses simdgroup_matrix or
        # cooperative_tensor API from Metal Performance Primitives.
        return f"""
// TensorOps matmul: TILE_M={tile_m} TILE_N={tile_n}
// TODO: replace with MPP cooperative_tensor API
uint row = thread_position_in_grid.y;
uint col = thread_position_in_grid.x;
T acc = 0;
for (uint k = 0; k < K; k++) {{
    acc += a[row * K + k] * b[k * N + col];
}}
out[row * N + col] = acc;
"""
