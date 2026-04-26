"""Example: multi-head attention optimization.

Run:
    phew run examples/attention.py -o examples/attention_optimized.py

Or from Python:
    from phew import Optimizer
    from examples.attention import fn, input_factory
    result = Optimizer(fn, input_factory).run()
    print(result.speedup)
"""

import mlx.core as mx
import numpy as np


def fn(q, k, v, scale: float = 1.0):
    """Naive scaled dot-product attention in pure MLX."""
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    weights = mx.softmax(scores, axis=-1)
    return weights @ v


# Name for the emitted function
fn_name = "fast_attention"

B, H, T, D = 2, 8, 256, 64
SCALE = D**-0.5

_SIZES = {
    "small": (2, 4, 64, 32),
    "typical": (2, 8, 256, 64),
    "large": (4, 16, 512, 64),
}


def input_factory(size_label: str, seed: int):
    rng = np.random.default_rng(seed)
    b, h, t, d = _SIZES[size_label]
    q = mx.array(rng.standard_normal((b, h, t, d)).astype(np.float32))
    k = mx.array(rng.standard_normal((b, h, t, d)).astype(np.float32))
    v = mx.array(rng.standard_normal((b, h, t, d)).astype(np.float32))
    return [q, k, v], {"scale": d**-0.5}
