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
// mx.fast.metal_kernel body — no function signature needed; MLX generates it.
// Template constants: T (dtype), M (rows), N (cols), K (inner dim)
// Grid:        (ceil(N/8), ceil(M/8), batch)   threadgroup: (32, 1, 1)
// Inputs:  a[batch*M*K], b[K*N]   Output: out[batch*M*N]

const uint row8     = threadgroup_position_in_grid.y * 8u;
const uint col8     = threadgroup_position_in_grid.x * 8u;
const uint batch_i  = threadgroup_position_in_grid.z;
const uint lane     = thread_index_in_simdgroup;

const uint a_off    = batch_i * (uint)M * (uint)K;
const uint out_off  = batch_i * (uint)M * (uint)N;

threadgroup float a_shared[64];
threadgroup float b_shared[64];
threadgroup float c_shared[64];

simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

for (uint k0 = 0u; k0 < (uint)K; k0 += 8u) {
    for (uint pass = 0u; pass < 2u; ++pass) {
        uint li  = pass * 32u + lane;
        uint lr  = li / 8u;
        uint lc  = li % 8u;
        uint ar  = row8 + lr, ac = k0 + lc;
        a_shared[li] = (ar < (uint)M && ac < (uint)K)
                       ? float(a[a_off + ar * (uint)K + ac]) : 0.0f;
        uint br  = k0 + lr, bc = col8 + lc;
        b_shared[li] = (br < (uint)K && bc < (uint)N)
                       ? float(b[br * (uint)N + bc]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 a_sg, b_sg;
    simdgroup_load(a_sg, a_shared, 8u, ulong2(0, 0));
    simdgroup_load(b_sg, b_shared, 8u, ulong2(0, 0));
    simdgroup_multiply_accumulate(acc, a_sg, b_sg, acc);
}

simdgroup_store(acc, c_shared, 8u, ulong2(0, 0));
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint pass = 0u; pass < 2u; ++pass) {
    uint li  = pass * 32u + lane;
    uint lr  = li / 8u;
    uint lc  = li % 8u;
    uint cr  = row8 + lr, cc = col8 + lc;
    if (cr < (uint)M && cc < (uint)N) {
        out[out_off + cr * (uint)N + cc] = T(c_shared[li]);
    }
}
"""

_TENSOROPS_HEADER = ""


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

            # K from first input's last dimension.
            a_node = graph[node.inputs[0]] if node.inputs and node.inputs[0] in graph else None
            K = int(a_node.shape[-1]) if a_node and a_node.shape else 0
            if K == 0:
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

            # Input shapes: a is [batch*M, K] flat, b is [K, N] flat.
            a_shape = (batch * M, K)
            b_shape = (K, N)

            metal_node = MetalKernel(
                shape=node.shape,
                dtype=node.dtype,
                inputs=list(node.inputs),
                source=_TENSOROPS_SOURCE,
                header=_TENSOROPS_HEADER,
                template_params=[("T", node.dtype.to_mlx()), ("M", M), ("N", N), ("K", K)],
                threadgroup=(32, 1, 1),
                grid=grid,
                input_names=["a", "b"],
                output_names=["out"],
                output_shapes=[node.shape],
                output_dtypes=[node.dtype],
                input_shapes=[a_shape, b_shape],
            )
            graph.replace(node.id, metal_node)
            changed = True

        return changed
