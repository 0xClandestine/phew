"""MLX Python code generator.

Takes an optimized phew.ir.Graph and emits a self-contained Python function
with the same signature as the original, using MLX operations.

The emitted code is designed to be:
  - Copy-pasteable / importable by the user
  - Human-readable (not minified)
  - Correct by construction (modulo verification)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phew.ir import Graph


class MLXCodegen:
    """Generate MLX Python source from a phew Graph."""

    def __init__(self, indent: int = 4) -> None:
        self._indent = " " * indent

    def emit(self, graph: "Graph", fn_name: str = "optimized") -> str:
        """Return Python source string for the optimized function."""
        from phew.ir import Compile

        lines: list[str] = []
        lines.append("import mlx.core as mx")
        lines.append("import mlx.core.fast as fast")
        lines.append("")

        # Detect whether any output node is a Compile node — if so, wrap
        # the whole function with @mx.compile rather than emitting a broken lambda.
        has_compile = any(isinstance(graph[o], Compile) for o in graph.outputs if o in graph._nodes)

        # Use graph.inputs() (insertion order) for the signature so that the
        # emitted function's positional parameters match how the caller passes args.
        inputs = graph.inputs()
        args = ", ".join(n.name or f"x{i}" for i, n in enumerate(inputs))

        if has_compile:
            lines.append("@mx.compile")

        lines.append(f"def {fn_name}({args}):")

        body = self._emit_body(graph)
        for line in body:
            lines.append(self._indent + line)

        return "\n".join(lines) + "\n"

    def _emit_body(self, graph: "Graph") -> list[str]:
        from phew.ir import (
            AsyncEval,
            Cast,
            Compile,
            Concat,
            Constant,
            Elementwise,
            FastLayerNorm,
            FastRMSNorm,
            FastRoPE,
            FastScaledDotProductAttention,
            Input,
            MatMul,
            MetalKernel,
            MetalKernelSelect,
            QuantizedMatMul,
            Reduce,
            Repeat,
            Reshape,
            Slice,
            Transpose,
        )

        lines: list[str] = []
        name_map: dict[int, str] = {}
        # Maps MetalKernel node.id → the "_out" variable name (the tuple returned by the call)
        kernel_out_map: dict[int, str] = {}
        var_counter = [0]

        def fresh(prefix: str = "t") -> str:
            var_counter[0] += 1
            return f"{prefix}{var_counter[0]}"

        for node in graph.topo_order():
            if isinstance(node, Input):
                name_map[node.id] = node.name or f"x{node.id}"
                continue

            if isinstance(node, Constant):
                vname = fresh("c")
                name_map[node.id] = vname
                dtype_str = node.dtype.to_mlx() if node.dtype is not None else None
                if dtype_str and dtype_str != "float32":
                    lines.append(f"{vname} = mx.array({node.value!r}, dtype=mx.{dtype_str})")
                else:
                    lines.append(f"{vname} = mx.array({node.value!r})")
                continue

            ins = [name_map.get(i, f"_missing_{i}") for i in node.inputs]

            if isinstance(node, MatMul):
                vname = fresh()
                ta = ".T" if node.transpose_a else ""
                tb = ".T" if node.transpose_b else ""
                lines.append(f"{vname} = {ins[0]}{ta} @ {ins[1]}{tb}")

            elif isinstance(node, Reduce):
                vname = fresh()
                axes = list(node.axes) if node.axes else None
                kd = repr(node.keepdims)
                if node.op == "sum":
                    lines.append(f"{vname} = mx.sum({ins[0]}, axis={axes}, keepdims={kd})")
                elif node.op == "mean":
                    lines.append(f"{vname} = mx.mean({ins[0]}, axis={axes}, keepdims={kd})")
                elif node.op == "max":
                    lines.append(f"{vname} = mx.max({ins[0]}, axis={axes}, keepdims={kd})")
                elif node.op == "min":
                    lines.append(f"{vname} = mx.min({ins[0]}, axis={axes}, keepdims={kd})")
                elif node.op == "softmax":
                    # softmax takes axis as a scalar, not a list
                    ax = axes[0] if axes and len(axes) == 1 else axes
                    lines.append(f"{vname} = mx.softmax({ins[0]}, axis={ax})")
                else:
                    lines.append(f"{vname} = mx.{node.op}({ins[0]}, axis={axes}, keepdims={kd})")

            elif isinstance(node, Concat):
                vname = fresh()
                lines.append(f"{vname} = mx.concatenate([{', '.join(ins)}], axis={node.axis})")

            elif isinstance(node, Repeat):
                vname = fresh()
                lines.append(f"{vname} = mx.repeat({ins[0]}, {node.repeats}, axis={node.axis})")

            elif isinstance(node, Elementwise):
                vname = fresh()
                op = node.op
                if op == "add" and len(ins) == 2:
                    lines.append(f"{vname} = {ins[0]} + {ins[1]}")
                elif op == "mul" and len(ins) == 2:
                    lines.append(f"{vname} = {ins[0]} * {ins[1]}")
                elif op == "sub" and len(ins) == 2:
                    lines.append(f"{vname} = {ins[0]} - {ins[1]}")
                elif op == "div" and len(ins) == 2:
                    lines.append(f"{vname} = {ins[0]} / {ins[1]}")
                elif len(ins) == 1:
                    lines.append(f"{vname} = mx.{op}({ins[0]})")
                else:
                    args = ", ".join(ins)
                    lines.append(f"{vname} = mx.{op}({args})")

            elif isinstance(node, Transpose):
                vname = fresh()
                if node.axes:
                    lines.append(f"{vname} = mx.transpose({ins[0]}, {list(node.axes)})")
                else:
                    lines.append(f"{vname} = {ins[0]}.T")

            elif isinstance(node, Slice):
                vname = fresh()
                idx_parts = []
                for s in node.slices:
                    if s is None:
                        idx_parts.append("None")
                    elif isinstance(s, int):
                        idx_parts.append(str(s))
                    elif isinstance(s, slice):
                        parts = [
                            "" if s.start is None else str(s.start),
                            "" if s.stop is None else str(s.stop),
                        ]
                        if s.step is not None and s.step != 1:
                            parts.append(str(s.step))
                        idx_parts.append(":".join(parts))
                    else:
                        idx_parts.append("...")
                lines.append(f"{vname} = {ins[0]}[{', '.join(idx_parts)}]")

            elif isinstance(node, Reshape):
                vname = fresh()
                in_sh = node.input_shape
                out_sh = node.new_shape
                # Detect expand_dims: one more output dim than input, new dim is exactly 1
                expand_axis = None
                if in_sh and len(out_sh) == len(in_sh) + 1:
                    for i in range(len(out_sh)):
                        if (
                            out_sh[i] == 1
                            and out_sh[:i] == in_sh[:i]
                            and out_sh[i + 1 :] == in_sh[i:]
                        ):
                            expand_axis = i
                            break
                if expand_axis is not None:
                    lines.append(f"{vname} = mx.expand_dims({ins[0]}, {expand_axis})")
                else:
                    # Find longest matching prefix between input and output shapes
                    prefix = 0
                    if in_sh:
                        for i in range(min(len(in_sh), len(out_sh))):
                            if in_sh[i] == out_sh[i]:
                                prefix = i + 1
                            else:
                                break
                    suffix_out = out_sh[prefix:]
                    suffix_in = in_sh[prefix:] if in_sh else ()
                    suffix_in_prod = 1
                    for s in suffix_in:
                        suffix_in_prod *= s
                    suffix_out_prod = 1
                    for s in suffix_out:
                        suffix_out_prod *= s

                    if in_sh and prefix > 0 and suffix_in_prod == suffix_out_prod and suffix_out:
                        # Dynamic prefix + static suffix (split or fold of trailing dims)
                        shape_parts = ", ".join(f"{ins[0]}.shape[{i}]" for i in range(prefix))
                        if len(suffix_out) == 1:
                            lines.append(f"{vname} = mx.reshape({ins[0]}, [{shape_parts}, -1])")
                        else:
                            static_suffix = ", ".join(str(s) for s in suffix_out)
                            lines.append(
                                f"{vname} = mx.reshape({ins[0]}, [{shape_parts}, {static_suffix}])"
                            )
                    else:
                        lines.append(f"{vname} = mx.reshape({ins[0]}, {list(node.new_shape)})")

            elif isinstance(node, Cast):
                vname = fresh()
                dt = node.target_dtype.to_mlx()
                lines.append(f"{vname} = {ins[0]}.astype(mx.{dt})")

            elif isinstance(node, Compile):
                # Handled at function level via @mx.compile decorator; pass through
                vname = ins[0] if ins else fresh("compiled")
                name_map[node.id] = vname
                continue

            elif isinstance(node, AsyncEval):
                # mx.async_eval schedules evaluation without blocking, overlapping
                # CPU work with GPU execution. Pass-through: the output array
                # reference is unchanged; the async_eval call is a side effect.
                vname = ins[0] if ins else fresh("async")
                lines.append(f"mx.async_eval({', '.join(ins)})")
                name_map[node.id] = vname
                continue

            elif isinstance(node, FastRMSNorm):
                vname = fresh()
                lines.append(f"{vname} = mx.fast.rms_norm({ins[0]}, {ins[1]}, eps={node.eps})")

            elif isinstance(node, FastLayerNorm):
                vname = fresh()
                bias = ins[2] if len(ins) > 2 else "None"
                lines.append(
                    f"{vname} = mx.fast.layer_norm({ins[0]}, {ins[1]}, {bias}, eps={node.eps})"
                )

            elif isinstance(node, FastRoPE):
                vname = fresh()
                lines.append(
                    f"{vname} = mx.fast.rope({ins[0]}, {node.dims}, "
                    f"traditional={node.traditional}, base={node.base}, "
                    f"scale={node.scale}, offset={node.offset})"
                )

            elif isinstance(node, FastScaledDotProductAttention):
                vname = fresh()
                lines.append(
                    f"{vname} = mx.fast.scaled_dot_product_attention("
                    f"{ins[0]}, {ins[1]}, {ins[2]}, scale={node.scale})"
                )

            elif isinstance(node, QuantizedMatMul):
                vname = fresh()
                lines.append(
                    f"{vname} = mx.quantized_matmul({ins[0]}, {ins[1]}, "
                    f"bits={node.bits}, group_size={node.group_size})"
                )

            elif isinstance(node, MetalKernel):
                vname = fresh("kernel")
                lines.extend(self._emit_metal_kernel(node, ins, vname, kernel_out_map))

            elif isinstance(node, MetalKernelSelect):
                vname = fresh()
                kernel_out_var = kernel_out_map.get(
                    node.inputs[0], f"_missing_{node.inputs[0]}_out"
                )
                lines.append(f"{vname} = {kernel_out_var}[{node.output_idx}]")

            else:
                vname = fresh("unknown")
                lines.append(f"# TODO: unhandled op {node.op}")
                lines.append(f"{vname} = {ins[0] if ins else 'None'}")

            name_map[node.id] = vname

        # Return outputs
        out_names = [name_map.get(o, f"_out_{o}") for o in graph.outputs]
        if len(out_names) == 1:
            lines.append(f"return {out_names[0]}")
        else:
            lines.append(f"return {', '.join(out_names)}")

        return lines

    def _emit_metal_kernel(
        self, node, ins: list[str], vname: str, kernel_out_map: dict
    ) -> list[str]:
        lines = []
        kname = f"_kernel_{node.id}"
        # Emit kernel definition
        in_names_str = str(node.input_names)
        out_names_str = str(node.output_names)
        source_escaped = node.source.replace('"""', r"\"\"\"")
        header_escaped = node.header.replace('"""', r"\"\"\"")
        lines.append(f"{kname} = mx.fast.metal_kernel(")
        lines.append(f'    name="kernel_{node.id}",')
        lines.append(f"    input_names={in_names_str},")
        lines.append(f"    output_names={out_names_str},")
        lines.append(f'    source="""{source_escaped}""",')
        if node.header:
            lines.append(f'    header="""{header_escaped}""",')
        lines.append(")")

        inputs_str = f"[{', '.join(ins)}]"
        dtypes_str = "[" + ", ".join(f"mx.{d.to_mlx()}" for d in node.output_dtypes) + "]"
        tg = node.threadgroup
        # Build template list with dtype values as bare `mx.*` expressions (not quoted strings).
        tmpl_parts = []
        for k, v in node.template_params:
            if isinstance(v, str) and not v.startswith("mx."):
                tmpl_parts.append(f'("{k}", mx.{v})')
            else:
                tmpl_parts.append(f'("{k}", {v!r})')
        tmpl = "[" + ", ".join(tmpl_parts) + "]"

        n_outputs = len(node.output_shapes)

        if n_outputs == 1:
            # Single output: use matching input for dynamic shape / grid.
            target_numel = 1
            for s in node.output_shapes[0]:
                target_numel *= s
            matching_inp = None
            if node.input_shapes:
                for var_name, in_sh in zip(ins, node.input_shapes):
                    n = 1
                    for s in in_sh:
                        n *= s
                    if n == target_numel:
                        matching_inp = var_name
                        break
                if matching_inp is None:
                    best_n = 0
                    for var_name, in_sh in zip(ins, node.input_shapes):
                        n = 1
                        for s in in_sh:
                            n *= s
                        if n > best_n:
                            best_n = n
                            matching_inp = var_name
            if matching_inp is None and ins:
                matching_inp = ins[0]
            shapes_expr = f"[{matching_inp}.shape]" if matching_inp else str(node.output_shapes)
            grid_expr = f"({matching_inp}.size, 1, 1)" if matching_inp else "(1, 1, 1)"
        else:
            # Multi-output: find an input where product(shape[:-1]) == grid[0].
            # This covers both (B, MIX) -> grid=(B,1,1) and (B,L,MIX) -> grid=(B*L,1,1).
            batch_val = node.grid[0] if node.grid else 0
            batch_inp = None
            n_batch_dims = 0

            if node.input_shapes and batch_val:
                for var_name, in_sh in zip(ins, node.input_shapes):
                    if len(in_sh) < 2:
                        continue
                    prod = 1
                    for s in in_sh[:-1]:
                        prod *= s
                    if prod == batch_val:
                        batch_inp = var_name
                        n_batch_dims = len(in_sh) - 1
                        break

            if batch_inp is not None:
                # Emit one shape variable per dynamic (non-last) input dim.
                dim_vars = []
                for i in range(n_batch_dims):
                    dvar = f"_D{i}_{node.id}"
                    lines.append(f"{dvar} = {batch_inp}.shape[{i}]")
                    dim_vars.append(dvar)

                if n_batch_dims == 1:
                    grid_expr = f"({dim_vars[0]}, 1, 1)"
                else:
                    grid_expr = f"({' * '.join(dim_vars)}, 1, 1)"

                # Output shapes: first n_batch_dims are dynamic, rest are static.
                def _out_shape_expr(out_sh):
                    dims = [
                        dim_vars[i] if i < n_batch_dims else str(d) for i, d in enumerate(out_sh)
                    ]
                    return "(" + ", ".join(dims) + ("," if len(dims) == 1 else "") + ")"

                shapes_expr = (
                    "[" + ", ".join(_out_shape_expr(sh) for sh in node.output_shapes) + "]"
                )
            else:
                shapes_expr = str(node.output_shapes)
                grid_expr = str(node.grid) if node.grid else "(1, 1, 1)"

        out_var = f"{vname}_out"
        lines.append(f"{out_var} = {kname}(")
        lines.append(f"    inputs={inputs_str},")
        lines.append(f"    output_shapes={shapes_expr},")
        lines.append(f"    output_dtypes={dtypes_str},")
        lines.append(f"    grid={grid_expr},")
        lines.append(f"    threadgroup={tg},")
        lines.append(f"    template={tmpl},")
        lines.append(")")

        kernel_out_map[node.id] = out_var

        if n_outputs == 1:
            # Single-output: extract immediately; no MetalKernelSelect nodes expected.
            lines.append(f"{vname} = {out_var}[0]")

        return lines
