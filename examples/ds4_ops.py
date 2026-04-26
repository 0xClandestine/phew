"""DeepSeek V4 hot-path ops — PHEW benchmark.

One HyperConnection cycle (the compute that runs per-layer):
  _hc_mixes          — fused RMS-rsqrt + matmul → mix weights
  hc_split_sinkhorn  — HC sinkhorn (inline Metal kernel on Apple GPU)
  _hc_collapse_op    — weighted-sum collapse (B,L,HC,D) → (B,L,D)
  _limited_swiglu    — clamped SwiGLU activation
  _hc_expand_op      — expand + comb matmul   (B,L,D) → (B,L,HC,D)

Run:
    phew run   examples/ds4_ops.py
    phew bench examples/ds4_ops.py
"""

import mlx.core as mx
import numpy as np

# DS4 default config constants
HC = 4  # hc_mult
D = 4096  # hidden_size
MIX = (2 + HC) * HC  # 24
ITERS = 20  # hc_sinkhorn_iters
NORM_EPS = 1e-6  # rms_norm_eps
LIMIT = 10.0  # swiglu_limit


# ---------------------------------------------------------------------------
# HC sinkhorn Metal kernel
# PHEW patches mx.fast.metal_kernel at load time → _MetalKernelWrapper.
# During tracing the call is recorded as a MetalKernel IR node; at runtime
# it delegates to the real kernel.
# ---------------------------------------------------------------------------


def _make_hc_split_sinkhorn_kernel():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint idx = thread_position_in_grid.x;
        constexpr int MIX  = (2 + HC) * HC;
        constexpr int BASE = 2 * HC;

        const device float* mix = (const device float*)mixes + idx * MIX;
        device float* pre_out   = (device float*)pre  + idx * HC;
        device float* post_out  = (device float*)post + idx * HC;
        device float* comb_out  = (device float*)comb + idx * HC * HC;

        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];
        const float epsv       = eps[0];

        // Pre-sigmoid
        {
            float4 z = *(const device float4*)mix * pre_scale
                     + *(const device float4*)base;
            *(device float4*)pre_out = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
        }

        // Post-sigmoid
        {
            float4 z = *(const device float4*)(mix + HC) * post_scale
                     + *(const device float4*)(base + HC);
            *(device float4*)post_out = 2.0f * 1.0f / (1.0f + metal::fast::exp(-z));
        }

        // Comb: four float4 loads — all independent, GPU issues in parallel
        float4 v0 = *(const device float4*)(mix  + BASE     ) * comb_scale + *(const device float4*)(base + BASE     );
        float4 v1 = *(const device float4*)(mix  + BASE +  4) * comb_scale + *(const device float4*)(base + BASE +  4);
        float4 v2 = *(const device float4*)(mix  + BASE +  8) * comb_scale + *(const device float4*)(base + BASE +  8);
        float4 v3 = *(const device float4*)(mix  + BASE + 12) * comb_scale + *(const device float4*)(base + BASE + 12);

        // Per-row stable softmax
        float m0 = metal::max(metal::max(v0.x, v0.y), metal::max(v0.z, v0.w));
        float m1 = metal::max(metal::max(v1.x, v1.y), metal::max(v1.z, v1.w));
        float m2 = metal::max(metal::max(v2.x, v2.y), metal::max(v2.z, v2.w));
        float m3 = metal::max(metal::max(v3.x, v3.y), metal::max(v3.z, v3.w));

        float4 e0 = metal::fast::exp(v0 - m0);
        float4 e1 = metal::fast::exp(v1 - m1);
        float4 e2 = metal::fast::exp(v2 - m2);
        float4 e3 = metal::fast::exp(v3 - m3);

        float4 r0 = e0 * 1.0f / (e0.x + e0.y + e0.z + e0.w) + epsv;
        float4 r1 = e1 * 1.0f / (e1.x + e1.y + e1.z + e1.w) + epsv;
        float4 r2 = e2 * 1.0f / (e2.x + e2.y + e2.z + e2.w) + epsv;
        float4 r3 = e3 * 1.0f / (e3.x + e3.y + e3.z + e3.w) + epsv;

        // Initial column normalization
        float4 col = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col; r1 *= col; r2 *= col; r3 *= col;

        // Sinkhorn iterations
        for (int iter = 1; iter < ITERS; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);
            col = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col; r1 *= col; r2 *= col; r3 *= col;
        }

        *(device float4*)(comb_out)      = r0;
        *(device float4*)(comb_out +  4) = r1;
        *(device float4*)(comb_out +  8) = r2;
        *(device float4*)(comb_out + 12) = r3;
    """

    return mx.fast.metal_kernel(
        name="deepseek_v4_hc_split_sinkhorn",
        input_names=["mixes", "scale", "base", "eps"],
        output_names=["pre", "post", "comb"],
        source=source,
    )


_KERNEL = _make_hc_split_sinkhorn_kernel()


# ---------------------------------------------------------------------------
# Ops — no @mx.compile so PHEW can trace through them and see the full graph
# ---------------------------------------------------------------------------


def _hc_mixes(flat, fn_T):
    """Fused RMS-rsqrt + matmul: (B,L,HC*D) @ (HC*D,MIX) → (B,L,MIX)."""
    rsqrt = mx.rsqrt((flat * flat).mean(axis=-1, keepdims=True) + NORM_EPS)
    return (flat @ fn_T) * rsqrt


def hc_split_sinkhorn(mixes, scale, base, eps_arr):
    """HC sinkhorn: uses inline Metal kernel when available, MLX fallback otherwise."""
    if _KERNEL is not None:
        n_rows = mixes.size // MIX
        return _KERNEL(
            inputs=[mixes, scale, base, eps_arr],
            template=[("HC", HC), ("ITERS", ITERS)],
            grid=(n_rows, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[
                (*mixes.shape[:-1], HC),
                (*mixes.shape[:-1], HC),
                (*mixes.shape[:-1], HC, HC),
            ],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )

    # MLX fallback
    mixes = mixes.astype(mx.float32)
    scale = scale.astype(mx.float32)
    base = base.astype(mx.float32)
    eps = eps_arr[0]
    pre_scale, post_scale, comb_scale = scale[0], scale[1], scale[2]
    pre = mx.sigmoid(mixes[..., :HC] * pre_scale + base[:HC]) + eps
    post = 2 * mx.sigmoid(mixes[..., HC : 2 * HC] * post_scale + base[HC : 2 * HC])
    comb = mixes[..., 2 * HC :].reshape(*mixes.shape[:-1], HC, HC) * comb_scale + base[
        2 * HC :
    ].reshape(HC, HC)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(ITERS - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _hc_collapse_op(pre, x):
    """Weighted-sum collapse: (B,L,HC,D) → (B,L,D)."""
    return (pre[..., None] * x.astype(mx.float32)).sum(axis=2).astype(x.dtype)


def _limited_swiglu(gate, up):
    """Clamped SwiGLU: silu(clamp(gate)) * clamp(up)."""
    gate = mx.minimum(gate, LIMIT)
    up = mx.clip(up, -LIMIT, LIMIT)
    return gate * mx.sigmoid(gate) * up


def _hc_expand_op(post, block_out, comb, residual):
    """HC expand: post-weighted block_out + comb-mixed residual → (B,L,HC,D)."""
    y = post[..., None] * block_out[:, :, None, :].astype(mx.float32)
    y = y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))
    return y.astype(block_out.dtype)


# ---------------------------------------------------------------------------
# Combined fn: one HyperConnection cycle
#   x        (B, L, HC, D)  — HC-expanded hidden states
#   fn_T     (HC*D, MIX)    — HyperConnection mixing matrix (fn.T)
#   hc_scale (3,)           — pre/post/comb scale factors
#   hc_base  (MIX,)         — mixing bias
#   eps_arr  (1,)           — sinkhorn epsilon
#   gate     (B, L, D)      — gate input for SwiGLU (block output gate)
#   up       (B, L, D)      — up input for SwiGLU
# ---------------------------------------------------------------------------


def fn(x, fn_T, hc_scale, hc_base, eps_arr, gate, up):
    """HC collapse → SwiGLU → HC expand (one transformer sub-block)."""
    B, L, H, Dv = x.shape
    flat = x.reshape(B, L, H * Dv).astype(mx.float32)
    mixes = _hc_mixes(flat, fn_T)  # (B, L, MIX)
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, eps_arr)
    _collapsed = _hc_collapse_op(pre, x)  # (B, L, D)
    block_out = _limited_swiglu(gate, up)  # (B, L, D)
    return _hc_expand_op(post, block_out, comb, x)  # (B, L, HC, D)


fn_name = "ds4_hc_cycle_opt"

_SIZES = {
    "small": (1, 1),  # single token, batch=1
    "typical": (1, 8),  # short generation context
    "large": (2, 64),  # batched / longer prompt
}


def input_factory(size_label: str, seed: int):
    rng = np.random.default_rng(seed)
    B, L = _SIZES[size_label]
    x = mx.array(rng.standard_normal((B, L, HC, D)).astype(np.float32))
    fn_T = mx.array(rng.standard_normal((HC * D, MIX)).astype(np.float32))
    hc_scale = mx.array(np.ones(3, dtype=np.float32))
    hc_base = mx.array(np.zeros(MIX, dtype=np.float32))
    eps_arr = mx.array(np.array([1e-6], dtype=np.float32))
    gate = mx.array(rng.standard_normal((B, L, D)).astype(np.float32))
    up = mx.array(rng.standard_normal((B, L, D)).astype(np.float32))
    return [x, fn_T, hc_scale, hc_base, eps_arr, gate, up], {}
