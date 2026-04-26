#!/usr/bin/env python3
"""Compare MLX public API against ops traced by phew.

Run:
    python scripts/check_op_coverage.py

Prints ops that exist in mlx.core but are not yet handled by the tracer,
grouped by rough category. Ops that are intentionally skipped (infra,
non-array utilities, submodules) are excluded from the report.
"""

import inspect

import mlx.core as mx

# ---------------------------------------------------------------------------
# What we currently monkey-patch in phew/ir/importer.py
# ---------------------------------------------------------------------------
PATCHED = {
    # original
    "sum",
    "mean",
    "max",
    "min",
    "softmax",
    "sqrt",
    "rsqrt",
    "exp",
    "log",
    "sigmoid",
    "expand_dims",
    "minimum",
    "maximum",
    "clip",
    "logaddexp",
    "transpose",
    "reshape",
    "matmul",
    "array",
    "zeros",
    "ones",
    "cos",
    "sin",
    "stack",
    "argpartition",
    "take_along_axis",
    "eval",
    "synchronize",
    # unary
    "abs",
    "negative",
    "ceil",
    "floor",
    "round",
    "sign",
    "square",
    "reciprocal",
    "logical_not",
    "erf",
    "erfinv",
    "expm1",
    "log1p",
    "log2",
    "log10",
    "tanh",
    "cosh",
    "sinh",
    "tan",
    # binary
    "add",
    "subtract",
    "multiply",
    "divide",
    "floor_divide",
    "remainder",
    "power",
    "equal",
    "not_equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "logical_and",
    "logical_or",
    "arctan2",
    "where",
    # reductions
    "all",
    "any",
    "prod",
    "std",
    "var",
    "logsumexp",
    "argmax",
    "argmin",
    "sort",
    "argsort",
    "topk",
    "partition",
    "cumsum",
    "cumprod",
    "logcumsumexp",
    # shape / indexing
    "concat",
    "concatenate",
    "split",
    "squeeze",
    "flatten",
    "swapaxes",
    "moveaxis",
    "broadcast_to",
    "take",
    "roll",
    "pad",
    "unflatten",
    # construction
    "zeros_like",
    "ones_like",
    "full",
    "arange",
    "linspace",
    "asarray",
    "eye",
    # trig / predicates
    "arctan",
    "arcsin",
    "arccos",
    "arctanh",
    "arcsinh",
    "arccosh",
    "degrees",
    "radians",
    "isfinite",
    "isinf",
    "isnan",
    "nan_to_num",
    "real",
    "imag",
}

# ---------------------------------------------------------------------------
# Intentionally out-of-scope: infra, non-array utils, submodules
# ---------------------------------------------------------------------------
SKIP = {
    # Device / stream management
    "set_default_device",
    "default_device",
    "get_default_device",
    "set_default_stream",
    "default_stream",
    "new_stream",
    "new_thread_local_stream",
    "stream",
    "clear_streams",
    "synchronize",
    "eval",
    "no_grad",
    "depends",
    # Dtypes
    "float32",
    "float16",
    "bfloat16",
    "int32",
    "int16",
    "int8",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int64",
    "bool_",
    "complex64",
    # Autograd / compilation
    "grad",
    "value_and_grad",
    "vjp",
    "jvp",
    "compile",
    "disable_compile",
    "enable_compile",
    "custom_function",
    "checkpoint",
    "async_eval",
    "vmap",
    "stop_gradient",
    "export_function",
    "import_function",
    "export_to_dot",
    "exporter",
    # Schedules (non-array)
    "linear_schedule",
    "cosine_decay",
    # Tree utils (non-array)
    "tree_flatten",
    "tree_unflatten",
    "tree_map",
    "tree_map_with_path",
    "tree_reduce",
    # Memory management
    "get_cache_memory",
    "get_peak_memory",
    "reset_peak_memory",
    "set_cache_limit",
    "set_memory_limit",
    "set_wired_limit",
    "get_active_memory",
    "clear_cache",
    # Printing / options
    "set_printoptions",
    "printoptions",
    "get_printoptions",
    # Device info (returns dict, not array)
    "device_count",
    "device_info",
    "is_available",
    # Submodules
    "random",
    "fft",
    "linalg",
    "fast",
    "distributed",
    "io",
    "metal",
    "nn",
    # Quantization (handled as IR node, not elemental patch)
    "quantize",
    "dequantize",
    "quantized_matmul",
    # I/O
    "load",
    "save",
    "savez",
    "savez_compressed",
    "save_gguf",
    "save_safetensors",
    # Misc utilities unlikely to appear in inference graphs
    "issubdtype",
    "result_type",
    "promote_types",
    "broadcast_shapes",
    "einsum_path",
    "export_to_dot",
    "exporter",
    "depends",
}

# ---------------------------------------------------------------------------
# Rough categories for the report
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Elementwise unary": [
        "abs",
        "neg",
        "negative",
        "ceil",
        "floor",
        "round",
        "sign",
        "square",
        "reciprocal",
        "logical_not",
        "bitwise_invert",
        "erf",
        "erfinv",
        "expm1",
        "log1p",
        "log2",
        "log10",
        "cosh",
        "sinh",
        "tan",
        "tanh",
        "arccos",
        "arccosh",
        "arcsin",
        "arcsinh",
        "arctan",
        "arctanh",
        "degrees",
        "radians",
        "real",
        "imag",
        "conj",
        "conjugate",
        "isfinite",
        "isinf",
        "isnan",
        "isneginf",
        "isposinf",
        "nan_to_num",
        "hadamard_transform",
    ],
    "Elementwise binary": [
        "add",
        "subtract",
        "multiply",
        "divide",
        "floor_divide",
        "remainder",
        "power",
        "equal",
        "not_equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "left_shift",
        "right_shift",
        "logical_and",
        "logical_or",
        "arctan2",
        "divmod",
        "allclose",
        "isclose",
        "array_equal",
    ],
    "Reduction": [
        "all",
        "any",
        "prod",
        "std",
        "var",
        "median",
        "logsumexp",
        "logcumsumexp",
        "cumsum",
        "cumprod",
        "cummax",
        "cummin",
        "argmax",
        "argmin",
        "argsort",
        "sort",
        "partition",
        "topk",
        "trace",
    ],
    "Shape / indexing": [
        "concat",
        "concatenate",
        "split",
        "stack",
        "repeat",
        "tile",
        "pad",
        "flatten",
        "unflatten",
        "squeeze",
        "broadcast_to",
        "broadcast_arrays",
        "swapaxes",
        "moveaxis",
        "permute_dims",
        "roll",
        "atleast_1d",
        "atleast_2d",
        "atleast_3d",
        "diag",
        "diagonal",
        "tril",
        "triu",
        "tri",
        "take",
        "take_along_axis",
        "put_along_axis",
        "slice",
        "slice_update",
        "as_strided",
        "meshgrid",
        "contiguous",
        "view",
    ],
    "Linear algebra": [
        "matmul",
        "inner",
        "outer",
        "tensordot",
        "einsum",
        "addmm",
        "kron",
        "block_masked_mm",
        "gather_mm",
        "segmented_mm",
        "qqmm",
        "gather_qmm",
    ],
    "Construction": [
        "arange",
        "linspace",
        "full",
        "eye",
        "identity",
        "zeros_like",
        "ones_like",
        "asarray",
        "bartlett",
        "blackman",
        "hamming",
        "hanning",
        "from_fp8",
        "to_fp8",
    ],
    "Where / ternary": ["where"],
}


def main():
    all_mx = {
        name
        for name in dir(mx)
        if not name.startswith("_")
        and callable(getattr(mx, name))
        and not inspect.isclass(getattr(mx, name))
    }

    unpatched = all_mx - PATCHED - SKIP

    print(f"MLX public callable ops : {len(all_mx)}")
    print(f"Patched by tracer       : {len(PATCHED)}")
    print(f"Intentionally skipped   : {len(SKIP)}")
    print(f"Unpatched (gap)         : {len(unpatched)}")
    print()

    categorised: set[str] = set()
    for cat, ops in CATEGORIES.items():
        missing = [op for op in ops if op in unpatched]
        if missing:
            print(f"  {cat}:")
            for op in missing:
                print(f"    {op}")
            categorised.update(missing)

    uncategorised = unpatched - categorised
    if uncategorised:
        print("  Other:")
        for op in sorted(uncategorised):
            print(f"    {op}")


if __name__ == "__main__":
    main()
