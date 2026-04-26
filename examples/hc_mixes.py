"""Example: HyperConnection _hc_mixes — RMS-rsqrt + matmul (DeepSeek V4).

The hot path in HyperConnection.compute_weights(), called every token
for all 43 layers × 2 HC modules (attn_hc + ffn_hc).

The fn matrix is [MIX, HC*HIDDEN] = [24, 16384] at fp32 — memory-bandwidth-bound.
Key question: does 4-bit quantization of fn pass equivalence verification at
fp32→fp32 tolerance?  (Expect not at 1e-5; test with fp32→bf16 opt-in.)

Run:
    phew run examples/hc_mixes.py
"""

import mlx.core as mx
import numpy as np

HC = 4
HIDDEN = 4096
MIX = (2 + HC) * HC  # 24


def fn(flat, fn_T, norm_eps=1e-6):
    """Fused RMS-rsqrt + matmul (inner body of _hc_mixes, without @mx.compile)."""
    rsqrt = mx.rsqrt((flat * flat).mean(axis=-1, keepdims=True) + norm_eps)
    return (flat @ fn_T) * rsqrt


fn_name = "hc_mixes_opt"

_SIZES = {
    "small": 64,
    "typical": 512,
    "large": 2048,
}


def input_factory(size_label: str, seed: int):
    rng = np.random.default_rng(seed)
    B = _SIZES[size_label]
    flat = mx.array(rng.standard_normal((B, HC * HIDDEN)).astype(np.float32))
    fn_T = mx.array(rng.standard_normal((HC * HIDDEN, MIX)).astype(np.float32))
    return [flat, fn_T], {}
