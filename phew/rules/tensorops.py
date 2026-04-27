"""TensorOps substitution pass — simdgroup_matrix GEMM (M2+, gen ≥ 14).

Substitutes eligible matmul subgraphs with a simdgroup_matrix GEMM kernel
when the device supports it (Apple Silicon M2+ / A16+, generation ≥ 14).

`simdgroup_matrix<T, Rows, Cols>` is the Metal 3 API for 8×8 cooperative
matrix-multiply operations within a simdgroup (32 threads).  Loading and
storing is done cooperatively; multiply-accumulate uses
`simdgroup_multiply_accumulate(D, A, B, C)` → D = A·B + C.

The kernel tiles A[M,K] @ B[K,N] → C[M,N] in 8×8 simdgroup tiles:
  - Grid: (⌈N/8⌉, ⌈M/8⌉, batch)
  - Threadgroup: (32, 1, 1) — one simdgroup per dispatch
  - Inner loop over K in steps of 8

All arithmetic is done in float32 regardless of T; results are cast to T
on store.  This avoids bfloat16 precision loss in the reduction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phew.ir import Graph


# Known-good TensorOps tile dimensions (M, N) for the outer threadgroup tiling.
# Each tile is decomposed into 8×8 simdgroup sub-tiles.
VALID_TENSOROPS_TILES = {
    (8, 8),
    (16, 16),
    (32, 32),
    (8, 16),
    (16, 8),
}

# Minimum problem size to benefit from simdgroup_matrix (avoid overhead for tiny matmuls)
MIN_MATMUL_SIZE = 64  # M * N >= 64

# simdgroup_matrix is available from M2/A16 (applegpu_g14) onward
MIN_GENERATION = 14


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
    """Return True if the device supports simdgroup_matrix GEMM (M2+, gen ≥ 14)."""
    try:
        import mlx.core as mx

        if not mx.metal.is_available():
            return False
        info = mx.device_info()
        arch = info.get("architecture", "")
        return _arch_generation(arch) >= MIN_GENERATION
    except Exception:
        return False


# ---------------------------------------------------------------------------
# MSL kernel source
# ---------------------------------------------------------------------------

_TENSOROPS_SOURCE = """\
// simdgroup_matrix GEMM: C[batch, M, N] = A[batch, M, K] @ B[K, N]
// Grid:        (ceil(N/8), ceil(M/8), batch)
// Threadgroup: (32, 1, 1) — one simdgroup per dispatch
// Accumulates in float32; stores as T.
//
// simdgroup_matrix<float,8,8> API (Metal 3, M2+):
//   simdgroup_load(mat, ptr, stride, offset, transpose=false)
//   simdgroup_store(mat, ptr, stride, offset)
//   simdgroup_multiply_accumulate(D, A, B, C) → D = A*B + C

kernel void simdgroup_gemm(
    device const T*  a    [[buffer(0)]],
    device const T*  b    [[buffer(1)]],
    device       T*  out  [[buffer(2)]],
    constant  uint&  M    [[buffer(3)]],
    constant  uint&  N    [[buffer(4)]],
    constant  uint&  K    [[buffer(5)]],
    uint3  tg_pos   [[threadgroup_position_in_grid]],
    uint   lane_id  [[thread_index_in_simdgroup]]
) {
    // Each threadgroup covers one 8x8 output tile.
    const uint row0  = tg_pos.y * 8u;
    const uint col0  = tg_pos.x * 8u;
    const uint batch = tg_pos.z;

    device const T* A = a + batch * M * K;
    device const T* B = b;
    device       T* C = out + batch * M * N;

    // Shared buffers for type conversion T → float for simdgroup_load.
    // 64 elements = one 8×8 tile.
    threadgroup float a_shared[64];
    threadgroup float b_shared[64];
    threadgroup float c_shared[64];

    // Accumulator in float for numerical precision.
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < K; k0 += 8u) {
        // 32 lanes fill 64 floats using 2 passes.
        for (uint pass = 0u; pass < 2u; ++pass) {
            uint li   = pass * 32u + lane_id;
            uint lr   = li / 8u;   // local row 0..7
            uint lc   = li % 8u;   // local col 0..7

            uint a_row = row0 + lr;
            uint a_col = k0   + lc;
            a_shared[li] = (a_row < M && a_col < K)
                ? float(A[a_row * K + a_col]) : 0.0f;

            uint b_row = k0   + lr;
            uint b_col = col0 + lc;
            b_shared[li] = (b_row < K && b_col < N)
                ? float(B[b_row * N + b_col]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 a_sg, b_sg;
        simdgroup_load(a_sg, a_shared, 8u, ulong2(0, 0));
        simdgroup_load(b_sg, b_shared, 8u, ulong2(0, 0));
        simdgroup_multiply_accumulate(acc, a_sg, b_sg, acc);
    }

    // Store: convert float → T via threadgroup buffer.
    simdgroup_store(acc, c_shared, 8u, ulong2(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pass = 0u; pass < 2u; ++pass) {
        uint li    = pass * 32u + lane_id;
        uint lr    = li / 8u;
        uint lc    = li % 8u;
        uint c_row = row0 + lr;
        uint c_col = col0 + lc;
        if (c_row < M && c_col < N) {
            C[c_row * N + c_col] = T(c_shared[li]);
        }
    }
}
"""

_TENSOROPS_HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"


# ---------------------------------------------------------------------------
# Pass
# ---------------------------------------------------------------------------


class TensorOpsPass:
    """Replace eligible MatMul nodes with simdgroup_matrix GEMM MetalKernel nodes."""

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

            M, N = int(node.shape[-2]), int(node.shape[-1])
            if M * N < MIN_MATMUL_SIZE:
                continue

            # Grid: one threadgroup per 8×8 output tile.
            # Batch = product of all dims except the last two.
            batch = 1
            for d in node.shape[:-2]:
                batch *= int(d)
            grid = (
                (N + 7) // 8,
                (M + 7) // 8,
                batch,
            )

            metal_node = MetalKernel(
                shape=node.shape,
                dtype=node.dtype,
                inputs=list(node.inputs),
                source=_TENSOROPS_SOURCE,
                header=_TENSOROPS_HEADER,
                template_params=[("T", node.dtype.to_mlx())],
                threadgroup=(32, 1, 1),
                grid=grid,
                input_names=["a", "b"],
                output_names=["out"],
                output_shapes=[node.shape],
                output_dtypes=[node.dtype],
            )
            graph.replace(node.id, metal_node)
            changed = True

        return changed
