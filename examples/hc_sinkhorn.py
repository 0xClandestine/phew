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


def fn(mixes, scale, base, eps_arr):
    """Inner body of _hc_split_sinkhorn_ops — single eps as array scalar."""
    eps = eps_arr[0]  # keep as traced array (shape=())
    mixes = mixes.astype(mx.float32)
    scale = scale.astype(mx.float32)
    base = base.astype(mx.float32)
    pre_scale, post_scale, comb_scale = scale[0], scale[1], scale[2]

    pre = mx.sigmoid(mixes[..., :HC] * pre_scale + base[:HC]) + eps
    post = 2 * mx.sigmoid(mixes[..., HC : 2 * HC] * post_scale + base[HC : 2 * HC])
    comb = (
        mixes[..., 2 * HC :].reshape(*mixes.shape[:-1], HC, HC) * comb_scale
        + base[2 * HC :].reshape(HC, HC)
    )
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
