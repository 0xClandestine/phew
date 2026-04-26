"""MLX → μGraph importer.

Traces a Python function by wrapping MLX array operations with a proxy
that records each op into a phew.ir.Graph.

Limitations:
  - Static shapes only (shapes are fixed at trace time)
  - No dynamic control flow
  - Python-level loops are unrolled at trace time

Usage:
    graph = trace_to_graph(fn, args, kwargs)
"""

from __future__ import annotations

from typing import Any, Callable

from .dtype import Dtype
from .graph import Graph
from .node import Node
from .ops import (
    Cast,
    Constant,
    Elementwise,
    Input,
    MatMul,
    Reduce,
    Reshape,
    Slice,
    Transpose,
)

# ---------------------------------------------------------------------------
# Metal kernel tracing
# ---------------------------------------------------------------------------

# Set to the active Graph while trace_to_graph is running so wrappers can
# record MetalKernel IR nodes without being explicitly passed the graph.
_TRACING_GRAPH: "Graph | None" = None


class _MetalKernelWrapper:
    """Drop-in proxy for an mx.fast.metal_kernel callable.

    Created at module-load time when the user module calls
    ``mx.fast.metal_kernel(...)``.  During tracing (when ``_TRACING_GRAPH``
    is set) and inputs contain ``_TracedArray`` objects, records a
    ``MetalKernel`` + ``MetalKernelSelect`` subgraph into the graph instead
    of executing the kernel.  Otherwise delegates to the real kernel.
    """

    def __init__(
        self,
        real_kernel,
        name: str,
        input_names: list,
        output_names: list,
        source: str,
        header: str,
    ) -> None:
        self._real = real_kernel
        self._name = name
        self._input_names = list(input_names)
        self._output_names = list(output_names)
        self._source = source
        self._header = header

    def __call__(
        self,
        inputs,
        output_shapes,
        output_dtypes,
        grid,
        threadgroup,
        template=None,
        **kw,
    ):
        global _TRACING_GRAPH

        template = template or []

        if _TRACING_GRAPH is None or not any(isinstance(x, _TracedArray) for x in inputs):
            return self._real(
                inputs=inputs,
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
                grid=grid,
                threadgroup=threadgroup,
                template=template,
                **kw,
            )

        from .deps import MemDep
        from .ops import MetalKernel, MetalKernelSelect

        graph = _TRACING_GRAPH

        inp_node_ids = [x._node.id for x in inputs if isinstance(x, _TracedArray)]
        inp_shapes = [x._node.shape for x in inputs if isinstance(x, _TracedArray)]
        out_dtypes_ir = [d if isinstance(d, Dtype) else Dtype.from_mlx(d) for d in output_dtypes]
        out_shapes = [tuple(s) for s in output_shapes]
        tg = threadgroup if isinstance(threadgroup, tuple) else (int(threadgroup), 1, 1)
        g = grid if isinstance(grid, tuple) else (int(grid), 1, 1)

        kernel_node = MetalKernel(
            shape=out_shapes[0],
            dtype=out_dtypes_ir[0],
            inputs=inp_node_ids,
            source=self._source,
            header=self._header,
            input_names=self._input_names,
            output_names=self._output_names,
            output_shapes=out_shapes,
            output_dtypes=out_dtypes_ir,
            input_shapes=inp_shapes,
            threadgroup=tg,
            grid=g,
            template_params=list(template),
            deps=MemDep.device_mem,
        )
        graph.add(kernel_node)

        if len(out_shapes) == 1:
            return [_TracedArray(kernel_node, graph)]

        # Multi-output: one MetalKernelSelect per output.
        results = []
        for i, (shape, dtype) in enumerate(zip(out_shapes, out_dtypes_ir)):
            sel = MetalKernelSelect(
                shape=shape,
                dtype=dtype,
                inputs=[kernel_node.id],
                output_idx=i,
            )
            graph.add(sel)
            results.append(_TracedArray(sel, graph))
        return results


class _TracedArray:
    """Proxy that records MLX operations into a Graph."""

    def __init__(self, node: Node, graph: Graph) -> None:
        self._node = node
        self._graph = graph

    @property
    def shape(self):
        return self._node.shape

    @property
    def dtype(self):
        return self._node.dtype

    @property
    def size(self):
        n = 1
        for s in self._node.shape:
            n *= s
        return n

    def _binop(self, other: "_TracedArray | Any", op: str) -> "_TracedArray":
        if not isinstance(other, _TracedArray):
            # Scalar constant
            const = Constant(shape=(), dtype=self._node.dtype, value=other)
            self._graph.add(const)
            other = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(self._node.shape, other._node.shape)
        node = Elementwise(
            shape=result_shape,
            dtype=self._node.dtype,
            inputs=[self._node.id, other._node.id],
            op=op,
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def __add__(self, other):
        return self._binop(other, "add")

    def __radd__(self, other):
        return self._binop(other, "add")

    def __sub__(self, other):
        return self._binop(other, "sub")

    def __rsub__(self, other):
        return self._binop(other, "sub")

    def __mul__(self, other):
        return self._binop(other, "mul")

    def __rmul__(self, other):
        return self._binop(other, "mul")

    def __truediv__(self, other):
        return self._binop(other, "div")

    def __rtruediv__(self, other):
        return self._binop(other, "div")

    def __neg__(self):
        node = Elementwise(
            shape=self._node.shape, dtype=self._node.dtype, inputs=[self._node.id], op="neg"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def __matmul__(self, other: "_TracedArray") -> "_TracedArray":
        if not self._node.shape or not other._node.shape:
            return self
        M = self._node.shape[-2] if len(self._node.shape) >= 2 else 1
        N = other._node.shape[-1]
        batch = self._node.shape[:-2]
        node = MatMul(
            shape=(*batch, M, N), dtype=self._node.dtype, inputs=[self._node.id, other._node.id]
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    @property
    def T(self) -> "_TracedArray":
        axes = tuple(range(len(self._node.shape) - 1, -1, -1))
        new_shape = tuple(reversed(self._node.shape))
        node = Transpose(shape=new_shape, dtype=self._node.dtype, inputs=[self._node.id], axes=axes)
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def astype(self, dtype) -> "_TracedArray":
        d = dtype if isinstance(dtype, Dtype) else Dtype.from_mlx(dtype)
        node = Cast(shape=self._node.shape, dtype=d, inputs=[self._node.id], target_dtype=d)
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def reshape(self, *shape) -> "_TracedArray":
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        node = Reshape(
            shape=shape,
            dtype=self._node.dtype,
            inputs=[self._node.id],
            new_shape=shape,
            input_shape=self._node.shape,
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def _reduce_method(self, op: str, axis=None, keepdims=False, **_) -> "_TracedArray":
        ndim = len(self._node.shape)
        if isinstance(axis, int):
            axes = (axis % ndim,)
        elif axis is not None:
            axes = tuple(a % ndim for a in axis)
        else:
            axes = tuple(range(ndim))
        out_shape = _reduce_shape(self._node.shape, axes, bool(keepdims))
        node = Reduce(
            shape=out_shape,
            dtype=self._node.dtype,
            inputs=[self._node.id],
            op=op,
            axes=axes,
            keepdims=bool(keepdims),
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def mean(self, axis=None, keepdims=False, **_):
        return self._reduce_method("mean", axis, keepdims)

    def sum(self, axis=None, keepdims=False, **_):
        return self._reduce_method("sum", axis, keepdims)

    def max(self, axis=None, keepdims=False, **_):
        return self._reduce_method("max", axis, keepdims)

    def min(self, axis=None, keepdims=False, **_):
        return self._reduce_method("min", axis, keepdims)

    def swapaxes(self, a: int, b: int) -> "_TracedArray":
        ndim = len(self._node.shape)
        a, b = a % ndim, b % ndim
        axes = list(range(ndim))
        axes[a], axes[b] = axes[b], axes[a]
        new_shape = tuple(self._node.shape[i] for i in axes)
        node = Transpose(
            shape=new_shape,
            dtype=self._node.dtype,
            inputs=[self._node.id],
            axes=tuple(axes),
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def squeeze(self, axis=None) -> "_TracedArray":
        if axis is None:
            new_shape = tuple(s for s in self._node.shape if s != 1)
        else:
            axis = axis % len(self._node.shape)
            new_shape = tuple(s for i, s in enumerate(self._node.shape) if i != axis)
        node = Reshape(
            shape=new_shape,
            dtype=self._node.dtype,
            inputs=[self._node.id],
            new_shape=new_shape,
            input_shape=self._node.shape,
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def flatten(self, start_dim=0, end_dim=-1) -> "_TracedArray":
        ndim = len(self._node.shape)
        start = start_dim % ndim
        end = end_dim % ndim
        flat = 1
        for i in range(start, end + 1):
            flat *= self._node.shape[i]
        new_shape = self._node.shape[:start] + (flat,) + self._node.shape[end + 1 :]
        node = Reshape(
            shape=new_shape,
            dtype=self._node.dtype,
            inputs=[self._node.id],
            new_shape=new_shape,
            input_shape=self._node.shape,
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def __getitem__(self, idx) -> "_TracedArray":
        shape = self._node.shape
        ndim = len(shape)
        if not isinstance(idx, tuple):
            idx = (idx,)
        # Count real (non-None, non-Ellipsis) indices for Ellipsis expansion
        n_real = sum(1 for i in idx if i is not None and i is not Ellipsis)
        # Expand Ellipsis → run of slice(None)
        expanded: list = []
        for i in idx:
            if i is Ellipsis:
                expanded.extend([slice(None)] * (ndim - n_real))
            else:
                expanded.append(i)
        # Build output shape, detect if we need a real Slice vs. just Reshape
        out_shape: list = []
        needs_slice = False
        inp_dim = 0
        for i in expanded:
            if i is None:
                out_shape.append(1)
            elif isinstance(i, int):
                needs_slice = True
                inp_dim += 1  # dimension consumed, not appended
            elif isinstance(i, slice):
                dim_size = shape[inp_dim] if inp_dim < ndim else 1
                start, stop, step = i.indices(dim_size)
                length = len(range(start, stop, step))
                out_shape.append(length)
                if not (start == 0 and stop == dim_size and step == 1):
                    needs_slice = True
                inp_dim += 1
            else:
                out_shape.append(shape[inp_dim] if inp_dim < ndim else 1)
                inp_dim += 1
        # Remaining unconsumed dims pass through
        while inp_dim < ndim:
            out_shape.append(shape[inp_dim])
            inp_dim += 1
        out_shape = tuple(out_shape)
        if not needs_slice:
            node = Reshape(
                shape=out_shape,
                dtype=self._node.dtype,
                inputs=[self._node.id],
                new_shape=out_shape,
                input_shape=self._node.shape,
            )
        else:
            node = Slice(
                shape=out_shape,
                dtype=self._node.dtype,
                inputs=[self._node.id],
                slices=tuple(expanded),
            )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def __repr__(self):
        return f"_TracedArray(shape={self.shape}, dtype={self.dtype})"


def _broadcast_shape(a: tuple, b: tuple) -> tuple:
    if not a:
        return b
    if not b:
        return a
    result = []
    for x, y in zip(reversed(a), reversed(b)):
        result.append(max(x, y))
    prefix = a[: -len(b)] if len(a) > len(b) else b[: -len(a)]
    return tuple(prefix) + tuple(reversed(result))


class _TracingContext:
    """Context that intercepts mlx.core calls to build a Graph."""

    def __init__(self, graph: Graph) -> None:
        self._graph = graph

    def _wrap(self, arr: Any) -> _TracedArray:
        return arr if isinstance(arr, _TracedArray) else arr

    def _unwrap_inputs(self, *args) -> list[int]:
        ids = []
        for a in args:
            if isinstance(a, _TracedArray):
                ids.append(a._node.id)
        return ids

    def _reduce(self, x, op, axis, keepdims) -> _TracedArray:
        if not isinstance(x, _TracedArray):
            return x
        ndim = len(x.shape)
        if isinstance(axis, int):
            axes = (axis % ndim,)
        elif axis is not None:
            axes = tuple(a % ndim for a in axis)
        else:
            axes = tuple(range(ndim))
        out_shape = _reduce_shape(x.shape, axes, keepdims)
        node = Reduce(
            shape=out_shape,
            dtype=x.dtype,
            inputs=[x._node.id],
            op=op,
            axes=axes,
            keepdims=bool(keepdims),
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def sum(self, x, axis=None, keepdims=False, **_):
        return self._reduce(x, "sum", axis, keepdims)

    def mean(self, x, axis=None, keepdims=False, **_):
        return self._reduce(x, "mean", axis, keepdims)

    def max(self, x, axis=None, keepdims=False, **_):
        return self._reduce(x, "max", axis, keepdims)

    def min(self, x, axis=None, keepdims=False, **_):
        return self._reduce(x, "min", axis, keepdims)

    def softmax(self, x, axis=-1, **_):
        if not isinstance(x, _TracedArray):
            return x
        # softmax = exp(x - max(x)) / sum(exp(...))
        m = self._reduce(x, "max", axis, True)
        shifted = x - m
        exp_node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[shifted._node.id], op="exp")
        self._graph.add(exp_node)
        exp_t = _TracedArray(exp_node, self._graph)
        s = self._reduce(exp_t, "sum", axis, True)
        return exp_t / s

    def sqrt(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math

            return math.sqrt(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="sqrt")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def exp(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math

            return math.exp(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="exp")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def log(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math

            return math.log(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="log")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def rsqrt(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="rsqrt")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def sigmoid(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="sigmoid")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def expand_dims(self, x, axis, **_):
        if not isinstance(x, _TracedArray):
            return x
        ndim = len(x.shape) + 1
        axis = axis % ndim
        new_shape = x.shape[:axis] + (1,) + x.shape[axis:]
        node = Reshape(
            shape=new_shape,
            dtype=x.dtype,
            inputs=[x._node.id],
            new_shape=new_shape,
            input_shape=x.shape,
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def minimum(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape,
            dtype=a._node.dtype,
            inputs=[a._node.id, b._node.id],
            op="minimum",
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def maximum(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape,
            dtype=a._node.dtype,
            inputs=[a._node.id, b._node.id],
            op="maximum",
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def clip(self, x, a_min, a_max, **_):
        if not isinstance(x, _TracedArray):
            return x
        # clip(x, lo, hi) = maximum(minimum(x, hi), lo)
        hi = Constant(shape=(), dtype=x._node.dtype, value=a_max)
        self._graph.add(hi)
        lo = Constant(shape=(), dtype=x._node.dtype, value=a_min)
        self._graph.add(lo)
        clipped = self.minimum(x, _TracedArray(hi, self._graph))
        return self.maximum(clipped, _TracedArray(lo, self._graph))

    def logaddexp(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape,
            dtype=a._node.dtype,
            inputs=[a._node.id, b._node.id],
            op="logaddexp",
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def transpose(self, x, axes=None, **_):
        if not isinstance(x, _TracedArray):
            return x
        if axes is None:
            return x.T
        new_shape = tuple(x.shape[a] for a in axes)
        node = Transpose(shape=new_shape, dtype=x.dtype, inputs=[x._node.id], axes=tuple(axes))
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def reshape(self, x, shape, **_):
        if not isinstance(x, _TracedArray):
            return x
        return x.reshape(shape)

    def matmul(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        return a @ b

    def array(self, value, **_):
        import numpy as np

        arr = np.asarray(value)
        node = Constant(shape=tuple(arr.shape), dtype=Dtype.float32, value=value)
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def zeros(self, shape, dtype=None, **_):
        node = Constant(shape=tuple(shape), dtype=Dtype.float32, value=0.0)
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def ones(self, shape, dtype=None, **_):
        node = Constant(shape=tuple(shape), dtype=Dtype.float32, value=1.0)
        self._graph.add(node)
        return _TracedArray(node, self._graph)


    def cos(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math
            return math.cos(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="cos")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def sin(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math
            return math.sin(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="sin")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def stack(self, arrays, axis=0, **_):
        traced = [a for a in arrays if isinstance(a, _TracedArray)]
        if not traced:
            import mlx.core as _mx
            return _mx.stack(arrays, axis=axis)
        ref = traced[0]
        new_shape = ref.shape[:axis] + (len(arrays),) + ref.shape[axis:]
        node = Elementwise(
            shape=new_shape,
            dtype=ref.dtype,
            inputs=[a._node.id for a in arrays if isinstance(a, _TracedArray)],
            op="stack",
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def argpartition(self, x, kth, axis=-1, **_):
        if not isinstance(x, _TracedArray):
            import mlx.core as _mx
            return _mx.argpartition(x, kth, axis=axis)
        # Result shape same as input, dtype int32
        from phew.ir.dtype import Dtype as _Dtype
        node = Elementwise(
            shape=x.shape, dtype=_Dtype.int32, inputs=[x._node.id], op="argpartition"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def take_along_axis(self, x, indices, axis, **_):
        if not isinstance(x, _TracedArray):
            import mlx.core as _mx
            return _mx.take_along_axis(x, indices, axis)
        idx = indices if isinstance(indices, _TracedArray) else indices
        idx_node_id = idx._node.id if isinstance(idx, _TracedArray) else x._node.id
        node = Elementwise(
            shape=x.shape, dtype=x.dtype, inputs=[x._node.id, idx_node_id], op="take_along_axis"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def eval(self, *args, **_):
        pass  # no-op in tracing

    def synchronize(self, **_):
        pass


def _reduce_shape(shape, axes, keepdims) -> tuple:
    if not axes:
        return () if not keepdims else tuple(1 for _ in shape)
    result = []
    for i, s in enumerate(shape):
        if i in axes:
            if keepdims:
                result.append(1)
        else:
            result.append(s)
    return tuple(result)


def trace_to_graph(
    fn: Callable,
    args: list,
    kwargs: dict,
) -> Graph:
    """Trace fn(*args, **kwargs) and return a phew Graph.

    MLX arrays in args are replaced with _TracedArray proxies.
    The function is called with the proxies; all MLX ops become IR nodes.
    """
    import mlx.core as mx

    graph = Graph()
    ctx = _TracingContext(graph)

    def to_proxy(arr: Any, name: str = "x") -> _TracedArray:
        if isinstance(arr, mx.array):
            dtype = Dtype.from_mlx(arr.dtype)
            node = Input(shape=tuple(arr.shape), dtype=dtype, name=name)
            graph.add(node)
            return _TracedArray(node, graph)
        return arr

    # Convert args
    proxy_args = [to_proxy(a, f"x{i}") for i, a in enumerate(args)]
    proxy_kwargs = {k: to_proxy(v, k) for k, v in kwargs.items()}

    # Monkey-patch mlx.core for the duration of the trace
    # This is a minimal patch; a full implementation would use
    # __array_function__ protocol or a custom dispatch mechanism.
    _orig = {}
    for name in [
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
    ]:
        _orig[name] = getattr(mx, name, None)
        setattr(mx, name, getattr(ctx, name))

    global _TRACING_GRAPH
    _TRACING_GRAPH = graph
    try:
        result = fn(*proxy_args, **proxy_kwargs)
    finally:
        _TRACING_GRAPH = None
        for name, orig in _orig.items():
            if orig is not None:
                setattr(mx, name, orig)
            else:
                try:
                    delattr(mx, name)
                except AttributeError:
                    pass

    # Record outputs
    if isinstance(result, _TracedArray):
        graph.outputs = [result._node.id]
    elif isinstance(result, (list, tuple)):
        graph.outputs = [r._node.id for r in result if isinstance(r, _TracedArray)]

    return graph
