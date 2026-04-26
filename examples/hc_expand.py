"""Example: HyperConnection _hc_expand_op — matmul + elementwise add (DeepSeek V4).

Called after every attention and FFN block. Two ops:
  1. post[...,None] * block_out  (HC×D elementwise scale)
  2. matmul(comb.T, residual)    (HC×HC matmul, tiny but dispatched separately)
  3. sum → cast

Run:
    phew run examples/hc_expand.py
"""

import mlx.core as mx
import numpy as np

HC = 4
HIDDEN = 4096


def fn(post, block_out, comb, residual):
    """Inner body of _hc_expand_op, without @mx.compile."""
    y = post[..., None] * block_out[:, :, None, :].astype(mx.float32)
    y = y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))
    return y.astype(block_out.dtype)


fn_name = "hc_expand_opt"

_SIZES = {
    "small": (2, 32),
    "typical": (2, 256),
    "large": (4, 1024),
}


def input_factory(size_label: str, seed: int):
    rng = np.random.default_rng(seed)
    B, L = _SIZES[size_label]
    post = mx.array(rng.random((B, L, HC)).astype(np.float32))
    block_out = mx.array(rng.standard_normal((B, L, HIDDEN)).astype(np.float16))
    comb = mx.array(
        (rng.random((B, L, HC, HC)) / HC).astype(np.float32)
    )  # doubly-stochastic
    residual = mx.array(rng.standard_normal((B, L, HC, HIDDEN)).astype(np.float16))
    return [post, block_out, comb, residual], {}
