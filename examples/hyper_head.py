"""Example: HyperHead _hyper_head_op — RMS + matmul + sigmoid + weighted sum (DeepSeek V4).

Called once per forward pass after the final norm. Fuses:
  RMS-rsqrt → matmul(fn.T) → sigmoid(mixes*scale+base) → weighted reduce over HC dim.

Run:
    phew run examples/hyper_head.py
"""

import mlx.core as mx
import numpy as np

HC = 4
HIDDEN = 4096


def fn(x, fn_w, scale, base, norm_eps=1e-6, hc_eps=1e-6):
    """Inner body of _hyper_head_op without @mx.compile."""
    B, L, H, D = x.shape
    flat = x.reshape(B, L, H * D).astype(mx.float32)
    rsqrt = mx.rsqrt((flat * flat).mean(axis=-1, keepdims=True) + norm_eps)
    mixes = (flat @ fn_w.T) * rsqrt
    pre = mx.sigmoid(mixes * scale[0] + base) + hc_eps
    return (pre[..., None] * x.astype(mx.float32)).sum(axis=2).astype(x.dtype)


fn_name = "hyper_head_opt"

_SIZES = {
    "small": (2, 32),
    "typical": (2, 256),
    "large": (4, 1024),
}


def input_factory(size_label: str, seed: int):
    rng = np.random.default_rng(seed)
    B, L = _SIZES[size_label]
    x = mx.array(rng.standard_normal((B, L, HC, HIDDEN)).astype(np.float32))
    fn_w = mx.array(rng.standard_normal((HC, HC * HIDDEN)).astype(np.float32))
    scale = mx.array(rng.random((1,)).astype(np.float32) + 0.5)
    base = mx.array(rng.standard_normal((HC,)).astype(np.float32))
    return [x, fn_w, scale, base], {}
