"""MSL (Metal Shading Language) source generator for elementwise subgraphs.

Takes a topologically-ordered list of fuseable IR nodes and emits the body
of an mx.fast.metal_kernel kernel — one thread per output element.

Supported node types:
  Elementwise  — all unary/binary ops
  Cast         — type coercions
  Constant     — inlined as MSL constexpr literals

Limitations (initial version):
  - Elementwise / same-shape ops only (no reductions inside the kernel)
  - All inputs must share the same numel (broadcasting already resolved by tracer)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phew.ir.node import Node

# Dtype → MSL type name
_DTYPE_MSL: dict[str, str] = {
    "float32": "float",
    "float16": "half",
    "bfloat16": "bfloat16",
    "int32": "int",
    "int16": "short",
    "int8": "char",
    "uint8": "uchar",
    "bool": "bool",
}

# Binary ops with infix syntax
_BINARY_OPS: dict[str, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}


def _msl(dtype) -> str:
    return _DTYPE_MSL.get(dtype.value, "float")


def _lit(value: object, msl_type: str) -> str:
    """Format a Python scalar as an MSL literal."""
    if isinstance(value, float):
        s = f"{value:.8g}"
        # Ensure the literal has a decimal point or exponent so that appending
        # 'f' produces valid Metal syntax (e.g. '10f' is illegal; '10.0f' is not)
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        s += "f"
        return s if msl_type == "float" else f"({msl_type}){s}"
    if isinstance(value, int):
        return str(value)
    return f"({msl_type}){value}"


def _broadcast_index(elem_var: str, out_shape: tuple, inp_shape: tuple) -> str:
    """Generate an MSL flat-index expression to access a broadcast input.

    Pads ``inp_shape`` on the left with 1s to match ``len(out_shape)``, then
    emits the strided index expression — skipping (size-1) broadcast dims.
    """
    ndim = len(out_shape)
    padded = (1,) * (ndim - len(inp_shape)) + tuple(inp_shape)

    # Strides in the output (row-major)
    out_strides: list[int] = []
    s = 1
    for d in reversed(out_shape):
        out_strides.insert(0, s)
        s *= d

    # Strides in the (left-padded) input (row-major)
    inp_strides: list[int] = []
    s = 1
    for d in reversed(padded):
        inp_strides.insert(0, s)
        s *= d

    terms: list[str] = []
    for i in range(ndim):
        if padded[i] == 1:
            continue  # broadcast dimension — always index 0
        # Coordinate along output dimension i
        if out_strides[i] == 1:
            dim_idx = f"({elem_var} % {out_shape[i]}u)"
        else:
            dim_idx = f"(({elem_var} / {out_strides[i]}u) % {out_shape[i]}u)"
        if inp_strides[i] == 1:
            terms.append(dim_idx)
        else:
            terms.append(f"{dim_idx} * {inp_strides[i]}u")

    return " + ".join(terms) if terms else "0"


class SubgraphMSLCodegen:
    """Emit an MSL kernel body for a subgraph of fuseable nodes.

    Parameters
    ----------
    subgraph_nodes:
        Topologically ordered fuseable nodes (Elementwise, Cast, Constant).
    external_inputs:
        Nodes from outside the subgraph whose values are consumed inside it.
        These become buffer parameters ``inp0, inp1, …``.
    external_outputs:
        Nodes inside the subgraph whose values are consumed outside it (or are
        graph outputs).  These become buffer parameters ``out0, out1, …``.
    """

    def emit(
        self,
        subgraph_nodes: list["Node"],
        external_inputs: list["Node"],
        external_outputs: list["Node"],
    ) -> tuple[str, list[str], list[str]]:
        """Return ``(source, input_names, output_names)``.

        ``source`` is the MSL kernel *body* string suitable for passing to
        ``mx.fast.metal_kernel(source=...)``.
        """
        from phew.ir.ops import Cast, Constant, Elementwise

        inp_names = [f"inp{i}" for i in range(len(external_inputs))]
        out_names = [f"out{i}" for i in range(len(external_outputs))]

        # Map from node id → MSL variable name
        var: dict[int, str] = {}
        _ctr = [0]

        def fresh() -> str:
            _ctr[0] += 1
            return f"v{_ctr[0]}"

        lines: list[str] = []

        # Bounds check — use numel of first output
        external_outputs[0].numel if external_outputs else 1

        # No bounds check — MLX dispatches exactly numel threads via
        # dispatchThreads, so thread_position_in_grid is always in range.
        lines.append("uint elem = thread_position_in_grid.x;")
        lines.append("")

        # Load external inputs.  All inputs are guaranteed to have the same
        # numel as the output (enforced by the fusion pass), so `elem` is a
        # valid flat index.  0-d arrays (numel==1, ndim==0) are passed as
        # plain scalars by MLX — no subscript needed.
        for node, name in zip(external_inputs, inp_names):
            t = _msl(node.dtype)
            vname = fresh()
            var[node.id] = vname
            if len(node.shape) == 0:
                lines.append(f"{t} {vname} = {name};")
            else:
                lines.append(f"{t} {vname} = {name}[elem];")

        if external_inputs:
            lines.append("")

        # Walk subgraph
        for node in subgraph_nodes:
            t = _msl(node.dtype)
            vname = fresh()
            var[node.id] = vname
            ins = [var.get(i, f"/* missing:{i} */") for i in node.inputs]

            if isinstance(node, Constant):
                lit = _lit(node.value, t)
                lines.append(f"constexpr {t} {vname} = {lit};")

            elif isinstance(node, Cast):
                ct = _msl(node.target_dtype)
                var[node.id] = vname  # reassign with correct type in var map
                # Re-use vname but correct type
                lines.append(f"{ct} {vname} = ({ct}){ins[0]};")

            elif isinstance(node, Elementwise):
                op = node.op
                if op in _BINARY_OPS and len(ins) == 2:
                    sym = _BINARY_OPS[op]
                    lines.append(f"{t} {vname} = {ins[0]} {sym} {ins[1]};")
                elif op == "neg":
                    lines.append(f"{t} {vname} = -{ins[0]};")
                elif op == "abs":
                    lines.append(f"{t} {vname} = metal::abs({ins[0]});")
                elif op == "exp":
                    lines.append(f"{t} {vname} = metal::exp({ins[0]});")
                elif op == "log":
                    lines.append(f"{t} {vname} = metal::log({ins[0]});")
                elif op == "sqrt":
                    lines.append(f"{t} {vname} = metal::sqrt({ins[0]});")
                elif op == "rsqrt":
                    lines.append(f"{t} {vname} = metal::rsqrt({ins[0]});")
                elif op == "sigmoid":
                    lines.append(f"{t} {vname} = 1.0f / (1.0f + metal::exp(-{ins[0]}));")
                elif op == "tanh":
                    lines.append(f"{t} {vname} = metal::tanh({ins[0]});")
                elif op == "relu":
                    lines.append(f"{t} {vname} = metal::max({ins[0]}, ({t})0);")
                elif op == "gelu":
                    lines.append(
                        f"{t} {vname} = {ins[0]} * 0.5f"
                        f" * (1.0f + metal::erf({ins[0]} * 0.7071067811865476f));"
                    )
                elif op == "silu":
                    lines.append(f"{t} {vname} = {ins[0]} / (1.0f + metal::exp(-{ins[0]}));")
                elif op == "cos":
                    lines.append(f"{t} {vname} = metal::cos({ins[0]});")
                elif op == "sin":
                    lines.append(f"{t} {vname} = metal::sin({ins[0]});")
                elif op == "arctan":
                    lines.append(f"{t} {vname} = metal::atan({ins[0]});")
                elif op == "arcsin":
                    lines.append(f"{t} {vname} = metal::asin({ins[0]});")
                elif op == "arccos":
                    lines.append(f"{t} {vname} = metal::acos({ins[0]});")
                elif op == "arctanh":
                    lines.append(f"{t} {vname} = metal::atanh({ins[0]});")
                elif op == "arcsinh":
                    lines.append(f"{t} {vname} = metal::asinh({ins[0]});")
                elif op == "arccosh":
                    lines.append(f"{t} {vname} = metal::acosh({ins[0]});")
                elif op == "degrees":
                    lines.append(f"{t} {vname} = {ins[0]} * 57.29577951308232f;")
                elif op == "radians":
                    lines.append(f"{t} {vname} = {ins[0]} * 0.017453292519943295f;")
                elif op == "square":
                    lines.append(f"{t} {vname} = {ins[0]} * {ins[0]};")
                elif op == "maximum" and len(ins) == 2:
                    lines.append(f"{t} {vname} = metal::max({ins[0]}, {ins[1]});")
                elif op == "minimum" and len(ins) == 2:
                    lines.append(f"{t} {vname} = metal::min({ins[0]}, {ins[1]});")
                elif op == "logaddexp" and len(ins) == 2:
                    lines.append(
                        f"{t} {vname} = metal::log(metal::exp({ins[0]}) + metal::exp({ins[1]}));"
                    )
                elif op == "ceil":
                    lines.append(f"{t} {vname} = metal::ceil({ins[0]});")
                elif op == "floor":
                    lines.append(f"{t} {vname} = metal::floor({ins[0]});")
                elif op == "round":
                    lines.append(f"{t} {vname} = metal::round({ins[0]});")
                elif op == "sign":
                    lines.append(f"{t} {vname} = ({t})(({ins[0]} > ({t})0) - ({ins[0]} < ({t})0));")
                elif op == "reciprocal":
                    lines.append(f"{t} {vname} = 1.0f / {ins[0]};")
                elif op == "logical_not":
                    lines.append(f"{t} {vname} = !{ins[0]};")
                elif op == "erf":
                    lines.append(f"{t} {vname} = metal::erf({ins[0]});")
                elif op == "erfinv":
                    lines.append(f"{t} {vname} = metal::erfinv({ins[0]});")
                elif op == "expm1":
                    lines.append(f"{t} {vname} = metal::exp({ins[0]}) - 1.0f;")
                elif op == "log1p":
                    lines.append(f"{t} {vname} = metal::log(1.0f + {ins[0]});")
                elif op == "log2":
                    lines.append(f"{t} {vname} = metal::log2({ins[0]});")
                elif op == "log10":
                    lines.append(f"{t} {vname} = metal::log10({ins[0]});")
                elif op == "cosh":
                    lines.append(f"{t} {vname} = metal::cosh({ins[0]});")
                elif op == "sinh":
                    lines.append(f"{t} {vname} = metal::sinh({ins[0]});")
                elif op == "tan":
                    lines.append(f"{t} {vname} = metal::tan({ins[0]});")
                elif op == "floor_divide":
                    lines.append(
                        f"{t} {vname} = ({t})metal::floor((float){ins[0]} / (float){ins[1]});"
                    )
                elif op == "remainder":
                    lines.append(f"{t} {vname} = metal::fmod({ins[0]}, {ins[1]});")
                elif op == "power":
                    lines.append(f"{t} {vname} = metal::pow({ins[0]}, {ins[1]});")
                elif op == "eq":
                    lines.append(f"{t} {vname} = ({t})({ins[0]} == {ins[1]});")
                elif op == "ne":
                    lines.append(f"{t} {vname} = ({t})({ins[0]} != {ins[1]});")
                elif op == "gt":
                    lines.append(f"{t} {vname} = ({t})({ins[0]} > {ins[1]});")
                elif op == "ge":
                    lines.append(f"{t} {vname} = ({t})({ins[0]} >= {ins[1]});")
                elif op == "lt":
                    lines.append(f"{t} {vname} = ({t})({ins[0]} < {ins[1]});")
                elif op == "le":
                    lines.append(f"{t} {vname} = ({t})({ins[0]} <= {ins[1]});")
                elif op == "logical_and":
                    lines.append(f"{t} {vname} = ({t})({ins[0]} && {ins[1]});")
                elif op == "logical_or":
                    lines.append(f"{t} {vname} = ({t})({ins[0]} || {ins[1]});")
                elif op == "arctan2":
                    lines.append(f"{t} {vname} = metal::atan2({ins[0]}, {ins[1]});")
                elif op == "where" and len(ins) == 3:
                    lines.append(f"{t} {vname} = {ins[0]} ? {ins[1]} : {ins[2]};")
                else:
                    lines.append(f"// unhandled op: {op}")
                    lines.append(f"{t} {vname} = {ins[0] if ins else '0'};")

        # Store outputs
        if external_outputs:
            lines.append("")
        for node, name in zip(external_outputs, out_names):
            t = _msl(node.dtype)
            vname = var.get(node.id, "/* missing */")
            lines.append(f"{name}[elem] = ({t}){vname};")

        source = "\n".join(f"    {line}" for line in lines)
        return source, inp_names, out_names
