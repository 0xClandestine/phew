"""Example: _hc_split_sinkhorn_ops — MLX fallback Sinkhorn (DeepSeek V4).

The MLX reference path (used during training or when Metal kernel is unavailable).
Three outputs: pre (sigmoid), post (2*sigmoid), comb (doubly-stochastic HC×HC).

The sinkhorn loop (ITERS iterations) is the expensive part — each iteration
is two column/row normalizations over a small HC×HC matrix, but dispatched
for every token in the batch.

Run:
    phew run examples/hc_sinkhorn.py
"""

import mlx.core as mx
import numpy as np

HC = 4
MIX = (2 + HC) * HC  # 24
ITERS = 20


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

        // Per-row stable softmax: compute all maxes before any exp
        float m0 = metal::max(metal::max(v0.x, v0.y), metal::max(v0.z, v0.w));
        float m1 = metal::max(metal::max(v1.x, v1.y), metal::max(v1.z, v1.w));
        float m2 = metal::max(metal::max(v2.x, v2.y), metal::max(v2.z, v2.w));
        float m3 = metal::max(metal::max(v3.x, v3.y), metal::max(v3.z, v3.w));

        float4 e0 = metal::fast::exp(v0 - m0);
        float4 e1 = metal::fast::exp(v1 - m1);
        float4 e2 = metal::fast::exp(v2 - m2);
        float4 e3 = metal::fast::exp(v3 - m3);

        // Explicit adds instead of dot(e, 1) — avoids unnecessary fmul
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

        // Write comb output (four aligned 128-bit stores)
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


def fn(mixes, scale, base, eps_arr):
    """Inner body of _hc_split_sinkhorn_ops — single eps as array scalar."""
    if _KERNEL is not None:
        B = mixes.shape[0]
        pre, post, comb = _KERNEL(
            inputs=[mixes, scale, base, eps_arr],
            output_shapes=[(B, HC), (B, HC), (B, HC, HC)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
            grid=(B, 1, 1),
            threadgroup=(min(256, B), 1, 1),
            template=[("HC", HC), ("ITERS", ITERS)],
        )
        return pre, post, comb

    eps = eps_arr[0]  # keep as traced array (shape=())
    mixes = mixes.astype(mx.float32)
    scale = scale.astype(mx.float32)
    base = base.astype(mx.float32)
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


fn_name = "hc_sinkhorn_opt"

_SIZES = {
    "small": 64,
    "typical": 512,
    "large": 2048,
}


def input_factory(size_label: str, seed: int):
    rng = np.random.default_rng(seed)
    B = _SIZES[size_label]
    mixes = mx.array(rng.standard_normal((B, MIX)).astype(np.float32))
    scale = mx.array(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    base = mx.array(np.zeros(MIX, dtype=np.float32))
    eps_arr = mx.array(np.array([1e-6], dtype=np.float32))
    return [mixes, scale, base, eps_arr], {}
