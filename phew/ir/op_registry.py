"""Registry of mlx.core and mlx.core.fast ops and their coverage status.

This is the single source of truth for which ops the phew tracer handles.
Every callable in ``mlx.core`` (except IO / system / meta ops) must appear in
exactly one of the four sets below.  ``tests/test_op_coverage.py`` validates
this invariant automatically on every test run.

Categories
----------
IMPLEMENTED
    The op has a method on ``_TracingContext`` in ``phew/ir/importer.py`` and
    a corresponding emit path in ``phew/emit/codegen.py``.

NOT_COMPUTATIONAL
    System, IO, meta, or testing ops that will never appear inside a
    compute graph being optimised.  The tracer does not intercept them.

REQUIRES_NEW_NODE
    The op needs a new dedicated IR node type (e.g. a struct with non-trivial
    spatial kernel / gather / mask semantics).  Not yet implemented.

TODO
    Computational ops that could be added with straightforward tracer +
    codegen work but have not been added yet.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# mlx.core
# ---------------------------------------------------------------------------

#: Ops fully implemented in the tracer and codegen.
IMPLEMENTED: frozenset[str] = frozenset(
    [
        # --- elementwise unary ---
        "abs",
        "arccos",
        "arccosh",
        "arcsin",
        "arcsinh",
        "arctan",
        "arctanh",
        "bitwise_invert",
        "ceil",
        "conj",
        "conjugate",
        "contiguous",
        "cos",
        "cosh",
        "degrees",
        "erf",
        "erfinv",
        "exp",
        "expm1",
        "floor",
        "hadamard_transform",
        "imag",
        "isfinite",
        "isinf",
        "isnan",
        "isneginf",
        "isposinf",
        "log",
        "log10",
        "log1p",
        "log2",
        "logical_not",
        "nan_to_num",
        "negative",
        "radians",
        "real",
        "reciprocal",
        "round",
        "rsqrt",
        "sigmoid",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "square",
        "tan",
        "tanh",
        # --- elementwise binary ---
        "add",
        "arctan2",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "divide",
        "equal",
        "floor_divide",
        "greater",
        "greater_equal",
        "left_shift",
        "less",
        "less_equal",
        "logaddexp",
        "logical_and",
        "logical_or",
        "maximum",
        "minimum",
        "multiply",
        "not_equal",
        "power",
        "remainder",
        "right_shift",
        "subtract",
        # --- elementwise ternary / misc ---
        "clip",
        "where",
        # --- elementwise with attrs ---
        "argmax",
        "argmin",
        "argpartition",
        "argsort",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
        "logcumsumexp",
        "roll",
        "sort",
        "take_along_axis",
        "tril",
        "triu",
        # --- reductions ---
        "all",
        "any",
        "logsumexp",
        "max",
        "mean",
        "median",
        "min",
        "prod",
        "softmax",
        "std",
        "sum",
        "var",
        # --- matmul / linear algebra ---
        "matmul",
        "quantized_matmul",
        # --- shape / layout ---
        "broadcast_to",
        "concat",
        "concatenate",
        "expand_dims",
        "flatten",
        "moveaxis",
        "pad",
        "permute_dims",
        "repeat",
        "reshape",
        "split",
        "squeeze",
        "stack",
        "swapaxes",
        "take",
        "transpose",
        "unflatten",
        # --- array constructors ---
        "arange",
        "asarray",
        "eye",
        "full",
        "linspace",
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
        # --- passthrough (intercepted to no-op during tracing) ---
        "eval",
        "partition",
        "synchronize",
        "topk",
    ]
)

#: System, IO, meta, and testing ops — never inside an optimisable graph.
NOT_COMPUTATIONAL: frozenset[str] = frozenset(
    [
        # IO / persistence
        "load",
        "save",
        "save_gguf",
        "save_safetensors",
        "savez",
        "savez_compressed",
        "import_function",
        "export_function",
        "export_to_dot",
        "exporter",
        # System / device management
        "async_eval",
        "checkpoint",
        "clear_cache",
        "clear_streams",
        "compile",
        "default_device",
        "default_stream",
        "depends",
        "device_count",
        "device_info",
        "disable_compile",
        "enable_compile",
        "get_active_memory",
        "get_cache_memory",
        "get_peak_memory",
        "get_printoptions",
        "is_available",
        "new_stream",
        "new_thread_local_stream",
        "printoptions",
        "reset_peak_memory",
        "set_cache_limit",
        "set_default_device",
        "set_default_stream",
        "set_memory_limit",
        "set_printoptions",
        "set_wired_limit",
        "stream",
        # Autodiff / transforms
        "grad",
        "jvp",
        "stop_gradient",
        "value_and_grad",
        "vjp",
        "vmap",
        # Quantisation helpers (not a direct compute op)
        "dequantize",
        "from_fp8",
        "quantize",
        "to_fp8",
        # Utilities returning scalars / metadata, not compute arrays
        "allclose",
        "array_equal",
        "broadcast_arrays",
        "broadcast_shapes",
        "einsum_path",
        "isclose",
        "issubdtype",
    ]
)

#: Ops requiring a new dedicated IR node type before they can be traced.
REQUIRES_NEW_NODE: frozenset[str] = frozenset(
    [
        # Convolutions — need ConvNode with kernel/stride/padding/dilation attrs
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_general",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "convolve",
        # Specialised matmul variants — need dedicated node with gather/mask args
        "addmm",
        "block_masked_mm",
        "gather_mm",
        "gather_qmm",
        "inner",
        "kron",
        "outer",
        "qqmm",
        "segmented_mm",
        "tensordot",
    ]
)

#: Computational ops not yet implemented — tractable additions.
TODO: frozenset[str] = frozenset(
    [
        # Shape / view
        "as_strided",
        "atleast_1d",
        "slice",  # handled via __getitem__; direct mx.slice call not traced
        "slice_update",  # handled via __setitem__; direct call not traced
        "atleast_2d",
        "atleast_3d",
        "diag",
        "diagonal",
        "identity",
        "meshgrid",
        "put_along_axis",
        "tile",
        "trace",
        "tri",
        "view",
        # Math
        "divmod",
        "einsum",
        # Signal processing window functions — rarely used in ML inference graphs
        "bartlett",
        "blackman",
        "hamming",
        "hanning",
    ]
)

# ---------------------------------------------------------------------------
# mlx.core.fast
# ---------------------------------------------------------------------------

#: fast ops fully implemented in the tracer and codegen via _TracingContext methods.
FAST_IMPLEMENTED: frozenset[str] = frozenset(
    [
        "layer_norm",
        "rms_norm",
        "rope",
        "scaled_dot_product_attention",
        "quantized_scaled_dot_product_attention",
    ]
)

#: fast ops handled through a class-based wrapper rather than a _TracingContext method.
FAST_SPECIAL: frozenset[str] = frozenset(
    [
        "metal_kernel",  # intercepted via MetalKernelWrapper, not _TracingContext
    ]
)

#: fast ops not applicable to Metal/MLX optimisation.
FAST_NOT_NEEDED: frozenset[str] = frozenset(
    [
        "cuda_kernel",  # CUDA-only, not Metal
        "precompiled_cuda_kernel",  # CUDA-only, not Metal
    ]
)
