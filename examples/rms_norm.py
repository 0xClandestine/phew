"""Example: RMS norm — should match fast.rms_norm via primitive substitution.

Run:
    phew run examples/rms_norm.py
"""

import mlx.core as mx
import numpy as np


def fn(x, weight, eps: float = 1e-5):
    """Naive RMS norm."""
    ms = mx.mean(x * x, axis=-1, keepdims=True)
    x_norm = x / mx.sqrt(ms + eps)
    return x_norm * weight


fn_name = "fast_rms_norm"

_SIZES = {
    "small": (32, 512),
    "typical": (128, 2048),
    "large": (512, 4096),
}


def input_factory(size_label: str, seed: int):
    rng = np.random.default_rng(seed)
    batch, hidden = _SIZES[size_label]
    x = mx.array(rng.standard_normal((batch, hidden)).astype(np.float32))
    w = mx.array(rng.standard_normal((hidden,)).astype(np.float32))
    return [x, w], {}
