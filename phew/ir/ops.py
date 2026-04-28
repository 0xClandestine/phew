"""Concrete node types for the μGraph IR.

Two levels:
  Kernel level  — MLX graph ops (what the user writes)
  Thread level  — threadgroup/simdgroup ops (inside metal_kernel)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .deps import MemDep
from .dtype import Dtype
from .node import Node

# ---------------------------------------------------------------------------
# Kernel-level nodes
# ---------------------------------------------------------------------------


@dataclass
class Input(Node):
    """Placeholder for a function input tensor."""

    name: str = ""


@dataclass
class Constant(Node):
    """A compile-time constant scalar or tensor."""

    value: object = None


@dataclass
class MatMul(Node):
    """Matrix multiplication: (…, M, K) × (…, K, N) → (…, M, N)."""

    transpose_a: bool = False
    transpose_b: bool = False


@dataclass
class Reduce(Node):
    """Reduction along one or more axes.

    op: "sum" | "max" | "min" | "mean" | "prod"
    """

    op: str = "sum"
    axes: tuple[int, ...] = field(default_factory=tuple)
    keepdims: bool = False


@dataclass
class Elementwise(Node):
    """Point-wise unary or binary op.

    op: "add" | "sub" | "mul" | "div" | "exp" | "log" | "sqrt" |
        "relu" | "gelu" | "silu" | "sigmoid" | "tanh" | "neg" |
        "abs" | "square" | "maximum" | "minimum" | ...
    """

    op: str = "add"


@dataclass
class Transpose(Node):
    """Permute axes."""

    axes: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class Reshape(Node):
    """Reshape to a new shape (same numel)."""

    new_shape: tuple[int, ...] = field(default_factory=tuple)
    input_shape: tuple[int, ...] = field(default_factory=tuple)  # shape before reshape


@dataclass
class Cast(Node):
    """Type cast; may change precision."""

    target_dtype: Dtype = Dtype.float32


@dataclass
class Slice(Node):
    """Array slicing / indexing."""

    slices: tuple = field(default_factory=tuple)


@dataclass
class Concat(Node):
    """Concatenate along an axis."""

    axis: int = 0


@dataclass
class Repeat(Node):
    """Repeat elements along an axis (mx.repeat semantics)."""

    repeats: int = 1
    axis: int = 0


@dataclass
class Split(Node):
    """Split along an axis."""

    axis: int = 0
    indices: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class Broadcast(Node):
    """Broadcast to a target shape."""

    target_shape: tuple[int, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Compile boundary nodes
# ---------------------------------------------------------------------------


@dataclass
class Compile(Node):
    """Wraps a subgraph in mx.compile."""

    shapeless: bool = False


@dataclass
class AsyncEval(Node):
    """Inserts mx.async_eval to overlap CPU work."""


@dataclass
class VMap(Node):
    """Vectorized map over a Python loop axis."""

    in_axes: tuple[int | None, ...] = field(default_factory=tuple)
    out_axes: int = 0


# ---------------------------------------------------------------------------
# MLX fast.* primitives (subgraph matches — these end Phase-1 search)
# ---------------------------------------------------------------------------


@dataclass
class FastRMSNorm(Node):
    """mlx.core.fast.rms_norm"""

    eps: float = 1e-5
    axis: int = -1


@dataclass
class FastLayerNorm(Node):
    """mlx.core.fast.layer_norm"""

    eps: float = 1e-5


@dataclass
class FastRoPE(Node):
    """mlx.core.fast.rope"""

    dims: int = 0
    traditional: bool = False
    base: float = 10000.0
    scale: float = 1.0
    offset: int = 0


@dataclass
class FastScaledDotProductAttention(Node):
    """mlx.core.fast.scaled_dot_product_attention"""

    scale: float = 1.0
    mask: str = "none"  # "none" | "causal" | "additive"


@dataclass
class FastQuantizedScaledDotProductAttention(Node):
    """mlx.core.fast.quantized_scaled_dot_product_attention

    inputs: [queries, keys, values, scale_k, biases_k, scale_v, biases_v]
    Output shape == queries.shape.
    """

    scale: float = 1.0
    bits: int = 4
    group_size: int = 64


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


@dataclass
class QuantizedMatMul(Node):
    """quantized_matmul with group-wise quantization.

    Replaces MatMul when weight is quantized.
    bits: 4 | 8
    group_size: e.g. 64
    """

    bits: int = 4
    group_size: int = 64


# ---------------------------------------------------------------------------
# Custom Metal kernel (Phase-2 result)
# ---------------------------------------------------------------------------


@dataclass
class MetalKernelSelect(Node):
    """Extracts output ``output_idx`` from a multi-output MetalKernel node.

    Used when a kernel produces more than one output array.  The single
    MetalKernel node represents the dispatch; one MetalKernelSelect node per
    output array carries the per-output shape and dtype.
    """

    output_idx: int = 0


@dataclass
class MetalKernel(Node):
    """A custom kernel produced by Phase-2 search.

    source:           MSL shader source
    header:           extra MSL header code
    template_params:  list of (name, value) compile-time constants
    threadgroup:      (x, y, z) threadgroup dimensions
    grid:             (x, y, z) grid dimensions (or None = inferred)
    """

    source: str = ""
    header: str = ""
    template_params: list[tuple[str, object]] = field(default_factory=list)
    threadgroup: tuple[int, int, int] = (256, 1, 1)
    grid: tuple[int, int, int] | None = None
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = field(default_factory=list)
    output_dtypes: list[Dtype] = field(default_factory=list)
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    deps: MemDep = field(default=MemDep.device_mem | MemDep.threadgroup_mem)
