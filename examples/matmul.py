"""Example: plain matmul — exercises Phase-2 parameter search.

Run:
    phew run examples/matmul.py
"""

import mlx.core as mx
import numpy as np


def fn(a, b):
    return a @ b


fn_name = "fast_matmul"

_SIZES = {
    "small": (64, 64, 64),
    "typical": (512, 512, 512),
    "large": (2048, 2048, 2048),
}


def input_factory(size_label: str, seed: int):
    rng = np.random.default_rng(seed)
    M, K, N = _SIZES[size_label]
    a = mx.array(rng.standard_normal((M, K)).astype(np.float32))
    b = mx.array(rng.standard_normal((K, N)).astype(np.float32))
    return [a, b], {}
