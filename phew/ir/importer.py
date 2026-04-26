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

    def zeros_like(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Constant(shape=x.shape, dtype=x.dtype, value=0.0)
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def ones_like(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Constant(shape=x.shape, dtype=x.dtype, value=1.0)
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def full(self, shape, fill_value, dtype=None, **_):
        node = Constant(shape=tuple(shape), dtype=Dtype.float32, value=fill_value)
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def arange(self, start, stop=None, step=1, dtype=None, **_):
        import math

        if stop is None:
            n = max(0, int(start))
        else:
            n = max(0, math.ceil((stop - start) / step))
        node = Constant(shape=(n,), dtype=Dtype.float32, value=0.0)
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def linspace(self, start, stop, num=50, dtype=None, **_):
        node = Constant(shape=(num,), dtype=Dtype.float32, value=0.0)
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def asarray(self, x, dtype=None, **_):
        if isinstance(x, _TracedArray):
            return x
        return self.array(x)

    def eye(self, n, m=None, k=0, dtype=None, **_):
        node = Constant(shape=(n, m if m is not None else n), dtype=Dtype.float32, value=1.0)
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def arctan(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="arctan")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def arcsin(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="arcsin")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def arccos(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="arccos")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def arctanh(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="arctanh")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def arcsinh(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="arcsinh")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def arccosh(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="arccosh")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def degrees(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="degrees")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def radians(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="radians")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def isfinite(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        from phew.ir.dtype import Dtype as _Dtype

        node = Elementwise(shape=x.shape, dtype=_Dtype.bool_, inputs=[x._node.id], op="isfinite")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def isinf(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        from phew.ir.dtype import Dtype as _Dtype

        node = Elementwise(shape=x.shape, dtype=_Dtype.bool_, inputs=[x._node.id], op="isinf")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def isnan(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        from phew.ir.dtype import Dtype as _Dtype

        node = Elementwise(shape=x.shape, dtype=_Dtype.bool_, inputs=[x._node.id], op="isnan")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def nan_to_num(self, x, nan=0, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="nan_to_num")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def real(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="real")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def imag(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="imag")
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

    def concat(self, arrays, axis=0, **_):
        traced = [a for a in arrays if isinstance(a, _TracedArray)]
        if not traced:
            return arrays[0]
        ref = traced[0]
        axis = axis % len(ref.shape)
        new_dim = sum(a.shape[axis] if isinstance(a, _TracedArray) else 0 for a in arrays)
        new_shape = ref.shape[:axis] + (new_dim,) + ref.shape[axis + 1 :]
        node = Elementwise(
            shape=new_shape,
            dtype=ref.dtype,
            inputs=[a._node.id for a in arrays if isinstance(a, _TracedArray)],
            op="concat",
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def concatenate(self, arrays, axis=0, **_):
        return self.concat(arrays, axis=axis)

    def split(self, x, indices_or_sections, axis=0, **_):
        if not isinstance(x, _TracedArray):
            return [x]
        axis = axis % len(x.shape)
        if isinstance(indices_or_sections, int):
            n = indices_or_sections
            chunk = x.shape[axis] // n
            out_shape = x.shape[:axis] + (chunk,) + x.shape[axis + 1 :]
        else:
            n = len(indices_or_sections) + 1
            chunk = x.shape[axis] // n if n > 0 else x.shape[axis]
            out_shape = x.shape[:axis] + (chunk,) + x.shape[axis + 1 :]
        results = []
        for _ in range(n):
            node = Elementwise(shape=out_shape, dtype=x.dtype, inputs=[x._node.id], op="split")
            self._graph.add(node)
            results.append(_TracedArray(node, self._graph))
        return results

    def squeeze(self, x, axis=None, **_):
        if not isinstance(x, _TracedArray):
            return x
        if axis is None:
            new_shape = tuple(s for s in x.shape if s != 1)
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            axes = tuple(a % len(x.shape) for a in axes)
            new_shape = tuple(s for i, s in enumerate(x.shape) if i not in axes)
        node = Reshape(
            shape=new_shape,
            dtype=x.dtype,
            inputs=[x._node.id],
            new_shape=new_shape,
            input_shape=x.shape,
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def flatten(self, x, start_axis=0, end_axis=-1, **_):
        if not isinstance(x, _TracedArray):
            return x
        ndim = len(x.shape)
        start = start_axis % ndim
        end = end_axis % ndim
        flat_size = 1
        for i in range(start, end + 1):
            flat_size *= x.shape[i]
        new_shape = x.shape[:start] + (flat_size,) + x.shape[end + 1 :]
        node = Reshape(
            shape=new_shape,
            dtype=x.dtype,
            inputs=[x._node.id],
            new_shape=new_shape,
            input_shape=x.shape,
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def swapaxes(self, x, axis1, axis2, **_):
        if not isinstance(x, _TracedArray):
            return x
        ndim = len(x.shape)
        axes = list(range(ndim))
        a1, a2 = axis1 % ndim, axis2 % ndim
        axes[a1], axes[a2] = axes[a2], axes[a1]
        new_shape = tuple(x.shape[a] for a in axes)
        node = Transpose(shape=new_shape, dtype=x.dtype, inputs=[x._node.id], axes=tuple(axes))
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def moveaxis(self, x, source, destination, **_):
        if not isinstance(x, _TracedArray):
            return x
        ndim = len(x.shape)
        src = source % ndim
        dst = destination % ndim
        order = [i for i in range(ndim) if i != src]
        order.insert(dst, src)
        new_shape = tuple(x.shape[i] for i in order)
        node = Transpose(shape=new_shape, dtype=x.dtype, inputs=[x._node.id], axes=tuple(order))
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def broadcast_to(self, x, shape, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Reshape(
            shape=tuple(shape),
            dtype=x.dtype,
            inputs=[x._node.id],
            new_shape=tuple(shape),
            input_shape=x.shape,
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def take(self, x, indices, axis=None, **_):
        if not isinstance(x, _TracedArray):
            return x
        idx_node_id = indices._node.id if isinstance(indices, _TracedArray) else x._node.id
        out_shape = x.shape  # simplified
        node = Elementwise(
            shape=out_shape,
            dtype=x.dtype,
            inputs=[x._node.id, idx_node_id],
            op="take",
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def roll(self, x, shift, axis=None, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="roll")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def pad(self, x, pad_width, mode="constant", **_):
        if not isinstance(x, _TracedArray):
            return x
        if isinstance(pad_width, int):
            new_shape = tuple(s + 2 * pad_width for s in x.shape)
        else:
            pw = (
                pad_width if isinstance(pad_width[0], (list, tuple)) else [pad_width] * len(x.shape)
            )
            new_shape = tuple(s + p[0] + p[1] for s, p in zip(x.shape, pw))
        node = Reshape(
            shape=new_shape,
            dtype=x.dtype,
            inputs=[x._node.id],
            new_shape=new_shape,
            input_shape=x.shape,
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def unflatten(self, x, axis, shape, **_):
        if not isinstance(x, _TracedArray):
            return x
        axis = axis % len(x.shape)
        new_shape = x.shape[:axis] + tuple(shape) + x.shape[axis + 1 :]
        node = Reshape(
            shape=new_shape,
            dtype=x.dtype,
            inputs=[x._node.id],
            new_shape=new_shape,
            input_shape=x.shape,
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def abs(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math

            return math.fabs(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="abs")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def negative(self, x, **_):
        if not isinstance(x, _TracedArray):
            return -x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="neg")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def ceil(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math

            return math.ceil(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="ceil")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def floor(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math

            return math.floor(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="floor")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def round(self, x, decimals=0, **_):
        if not isinstance(x, _TracedArray):
            import builtins

            return builtins.round(x, decimals)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="round")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def sign(self, x, **_):
        if not isinstance(x, _TracedArray):
            return (x > 0) - (x < 0)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="sign")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def square(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x * x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="square")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def reciprocal(self, x, **_):
        if not isinstance(x, _TracedArray):
            return 1.0 / x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="reciprocal")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def logical_not(self, x, **_):
        if not isinstance(x, _TracedArray):
            return not x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="logical_not")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def erf(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math

            return math.erf(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="erf")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def erfinv(self, x, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="erfinv")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def expm1(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math

            return math.expm1(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="expm1")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def log1p(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math

            return math.log1p(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="log1p")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def log2(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math

            return math.log2(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="log2")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def log10(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math

            return math.log10(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="log10")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def tanh(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math

            return math.tanh(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="tanh")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def cosh(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math

            return math.cosh(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="cosh")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def sinh(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math

            return math.sinh(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="sinh")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def tan(self, x, **_):
        if not isinstance(x, _TracedArray):
            import math

            return math.tan(x)
        node = Elementwise(shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="tan")
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def add(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape, dtype=a._node.dtype, inputs=[a._node.id, b._node.id], op="add"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def subtract(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape, dtype=a._node.dtype, inputs=[a._node.id, b._node.id], op="sub"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def multiply(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape, dtype=a._node.dtype, inputs=[a._node.id, b._node.id], op="mul"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def divide(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape, dtype=a._node.dtype, inputs=[a._node.id, b._node.id], op="div"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def floor_divide(self, a, b, **_):
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
            op="floor_divide",
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def remainder(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape, dtype=a._node.dtype, inputs=[a._node.id, b._node.id], op="remainder"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def power(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape, dtype=a._node.dtype, inputs=[a._node.id, b._node.id], op="power"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def equal(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape, dtype=Dtype.bool_, inputs=[a._node.id, b._node.id], op="eq"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def not_equal(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape, dtype=Dtype.bool_, inputs=[a._node.id, b._node.id], op="ne"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def greater(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape, dtype=Dtype.bool_, inputs=[a._node.id, b._node.id], op="gt"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def greater_equal(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape, dtype=Dtype.bool_, inputs=[a._node.id, b._node.id], op="ge"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def less(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape, dtype=Dtype.bool_, inputs=[a._node.id, b._node.id], op="lt"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def less_equal(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape, dtype=Dtype.bool_, inputs=[a._node.id, b._node.id], op="le"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def logical_and(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape, dtype=Dtype.bool_, inputs=[a._node.id, b._node.id], op="logical_and"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def logical_or(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape, dtype=Dtype.bool_, inputs=[a._node.id, b._node.id], op="logical_or"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def arctan2(self, a, b, **_):
        if not isinstance(a, _TracedArray):
            return a
        if not isinstance(b, _TracedArray):
            const = Constant(shape=(), dtype=a._node.dtype, value=b)
            self._graph.add(const)
            b = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(a._node.shape, b._node.shape)
        node = Elementwise(
            shape=result_shape, dtype=a._node.dtype, inputs=[a._node.id, b._node.id], op="arctan2"
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def where(self, condition, x, y, **_):
        if (
            not isinstance(condition, _TracedArray)
            and not isinstance(x, _TracedArray)
            and not isinstance(y, _TracedArray)
        ):
            return condition
        if not isinstance(condition, _TracedArray):
            const = Constant(shape=(), dtype=Dtype.bool_, value=condition)
            self._graph.add(const)
            condition = _TracedArray(const, self._graph)
        if not isinstance(x, _TracedArray):
            ref_dtype = y._node.dtype if isinstance(y, _TracedArray) else condition._node.dtype
            const = Constant(shape=(), dtype=ref_dtype, value=x)
            self._graph.add(const)
            x = _TracedArray(const, self._graph)
        if not isinstance(y, _TracedArray):
            const = Constant(shape=(), dtype=x._node.dtype, value=y)
            self._graph.add(const)
            y = _TracedArray(const, self._graph)
        result_shape = _broadcast_shape(
            _broadcast_shape(condition._node.shape, x._node.shape), y._node.shape
        )
        node = Elementwise(
            shape=result_shape,
            dtype=x._node.dtype,
            inputs=[condition._node.id, x._node.id, y._node.id],
            op="where",
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def all(self, x, axis=None, keepdims=False, **_):
        if not isinstance(x, _TracedArray):
            return x
        return x._reduce_method("all", axis=axis, keepdims=keepdims)

    def any(self, x, axis=None, keepdims=False, **_):
        if not isinstance(x, _TracedArray):
            return x
        return x._reduce_method("any", axis=axis, keepdims=keepdims)

    def prod(self, x, axis=None, keepdims=False, **_):
        if not isinstance(x, _TracedArray):
            return x
        return x._reduce_method("prod", axis=axis, keepdims=keepdims)

    def std(self, x, axis=None, keepdims=False, ddof=0, **_):
        if not isinstance(x, _TracedArray):
            return x
        return x._reduce_method("std", axis=axis, keepdims=keepdims)

    def var(self, x, axis=None, keepdims=False, ddof=0, **_):
        if not isinstance(x, _TracedArray):
            return x
        return x._reduce_method("var", axis=axis, keepdims=keepdims)

    def logsumexp(self, x, axis=None, keepdims=False, **_):
        if not isinstance(x, _TracedArray):
            return x
        return x._reduce_method("logsumexp", axis=axis, keepdims=keepdims)

    def argmax(self, x, axis=None, keepdims=False, **_):
        if not isinstance(x, _TracedArray):
            return x
        from phew.ir.dtype import Dtype as _Dtype

        ndim = len(x.shape)
        axes = (axis % ndim,) if isinstance(axis, int) else tuple(range(ndim))
        out_shape = _reduce_shape(x.shape, axes, bool(keepdims))
        node = Reduce(
            shape=out_shape,
            dtype=_Dtype.int32,
            inputs=[x._node.id],
            op="argmax",
            axes=axes,
            keepdims=bool(keepdims),
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def argmin(self, x, axis=None, keepdims=False, **_):
        if not isinstance(x, _TracedArray):
            return x
        from phew.ir.dtype import Dtype as _Dtype

        ndim = len(x.shape)
        axes = (axis % ndim,) if isinstance(axis, int) else tuple(range(ndim))
        out_shape = _reduce_shape(x.shape, axes, bool(keepdims))
        node = Reduce(
            shape=out_shape,
            dtype=_Dtype.int32,
            inputs=[x._node.id],
            op="argmin",
            axes=axes,
            keepdims=bool(keepdims),
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def sort(self, x, axis=-1, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Reduce(
            shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="sort", axes=(), keepdims=True
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def argsort(self, x, axis=-1, **_):
        if not isinstance(x, _TracedArray):
            return x
        from phew.ir.dtype import Dtype as _Dtype

        node = Reduce(
            shape=x.shape,
            dtype=_Dtype.int32,
            inputs=[x._node.id],
            op="argsort",
            axes=(),
            keepdims=True,
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def topk(self, x, k, axis=-1, **_):
        if not isinstance(x, _TracedArray):
            return x
        ndim = len(x.shape)
        ax = axis % ndim
        out_shape = x.shape[:ax] + (k,) + x.shape[ax + 1 :]
        node = Reduce(
            shape=out_shape,
            dtype=x.dtype,
            inputs=[x._node.id],
            op="topk",
            axes=(ax,),
            keepdims=False,
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def partition(self, x, kth, axis=-1, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Reduce(
            shape=x.shape,
            dtype=x.dtype,
            inputs=[x._node.id],
            op="partition",
            axes=(),
            keepdims=True,
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def cumsum(self, x, axis=None, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Reduce(
            shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="cumsum", axes=(), keepdims=True
        )
        self._graph.add(node)
        return _TracedArray(node, self._graph)

    def cumprod(self, x, axis=None, **_):
        if not isinstance(x, _TracedArray):
            return x
        node = Reduce(
            shape=x.shape, dtype=x.dtype, inputs=[x._node.id], op="cumprod", axes=(), keepdims=True
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
        "zeros_like",
        "ones_like",
        "full",
        "arange",
        "linspace",
        "asarray",
        "eye",
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
