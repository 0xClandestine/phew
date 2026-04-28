"""Algebraic simplification pass — operates directly on phew.ir.Graph.

Rewrites applied (in order, iterated until fixed point):
  1. No-op cast removal:       cast(x, x.dtype)              → x
  2. Double-cast elimination:  cast(cast(x, A), B)            → cast(x, B)
  3. Transpose cancellation:   transpose(transpose(x,p), q)   → x  if p∘q = id
  4. Reshape of reshape:       reshape(reshape(x, s1), s2)    → reshape(x, s2)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phew.ir.graph import Graph
    from phew.ir.node import Node, NodeId


def _compose_axes(outer: tuple[int, ...], inner: tuple[int, ...]) -> tuple[int, ...]:
    """Compose two permutations: result[i] = inner[outer[i]]."""
    return tuple(inner[outer[i]] for i in range(len(outer)))


def _is_identity(axes: tuple[int, ...]) -> bool:
    return axes == tuple(range(len(axes)))


class AlgebraicPass:
    """Algebraic simplification pass.

    Rewrites:
    - No-op cast removal: cast(x, x.dtype) → x
    - Double-cast elimination: cast(cast(x, A), B) → cast(x, B)
    - Transpose cancellation: transpose(transpose(x, p), q) → x when p∘q = identity
    - Reshape of reshape: reshape(reshape(x, s1), s2) → reshape(x, s2)
    """

    def run(self, graph: "Graph") -> "Graph":
        """Apply rewrites to *graph* until fixed point. Returns graph."""
        changed = True
        while changed:
            changed = self._one_pass(graph)
        return graph

    def _one_pass(self, graph: "Graph") -> bool:
        from phew.ir.ops import Cast, Reshape, Transpose

        for node in graph.topo_order():
            if isinstance(node, Cast) and self._rewrite_cast(node, graph):
                return True
            if isinstance(node, Transpose) and self._rewrite_transpose(node, graph):
                return True
            if isinstance(node, Reshape) and self._rewrite_reshape(node, graph):
                return True
        return False

    # ------------------------------------------------------------------
    # Cast
    # ------------------------------------------------------------------

    def _rewrite_cast(self, node: "Node", graph: "Graph") -> bool:
        from phew.ir.ops import Cast

        if not node.inputs:
            return False
        inp_id: "NodeId" = node.inputs[0]
        if inp_id not in graph:
            return False
        inp = graph[inp_id]

        # Rule 1: no-op cast
        if node.target_dtype == inp.dtype:
            self._redirect(node.id, inp_id, graph)
            graph.remove(node.id)
            return True

        # Rule 2: double-cast — cast(cast(x, A), B) → cast(x, B)
        if isinstance(inp, Cast) and inp.inputs:
            inner_inp_id = inp.inputs[0]
            if inner_inp_id not in graph:
                return False
            node.inputs = [inner_inp_id]
            if not graph.successors(inp.id):
                graph.remove(inp.id)
            return True

        return False

    # ------------------------------------------------------------------
    # Transpose
    # ------------------------------------------------------------------

    def _rewrite_transpose(self, node: "Node", graph: "Graph") -> bool:
        from phew.ir.ops import Transpose

        if not node.inputs:
            return False
        inp_id = node.inputs[0]
        if inp_id not in graph:
            return False
        inp = graph[inp_id]

        if not isinstance(inp, Transpose) or not inp.inputs:
            return False

        outer = node.axes
        inner = inp.axes
        if not outer or not inner or len(outer) != len(inner):
            return False

        composed = _compose_axes(outer, inner)
        if _is_identity(composed):
            inner_inp_id = inp.inputs[0]
            self._redirect(node.id, inner_inp_id, graph)
            graph.remove(node.id)
            if not graph.successors(inp.id):
                graph.remove(inp.id)
            return True

        return False

    # ------------------------------------------------------------------
    # Reshape
    # ------------------------------------------------------------------

    def _rewrite_reshape(self, node: "Node", graph: "Graph") -> bool:
        from phew.ir.ops import Reshape

        if not node.inputs:
            return False
        inp_id = node.inputs[0]
        if inp_id not in graph:
            return False
        inp = graph[inp_id]

        if not isinstance(inp, Reshape) or not inp.inputs:
            return False

        inner_inp_id = inp.inputs[0]
        if inner_inp_id not in graph:
            return False
        inner_inp = graph[inner_inp_id]

        # Only collapse when both operations are true reshapes: the element
        # count must be identical at every stage.  broadcast_to() is also
        # encoded as a Reshape node but *changes* the element count, so
        # merging it with a preceding reshape would produce an invalid op.
        def _numel(shape):
            n = 1
            for s in shape:
                n *= s
            return n

        if _numel(inner_inp.shape) != _numel(inp.new_shape):
            return False
        if _numel(inp.new_shape) != _numel(node.new_shape):
            return False

        node.inputs = [inner_inp_id]
        node.input_shape = inner_inp.shape
        if not graph.successors(inp.id):
            graph.remove(inp.id)
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _redirect(self, old_id: "NodeId", new_id: "NodeId", graph: "Graph") -> None:
        for n in graph.nodes():
            n.inputs = [new_id if i == old_id else i for i in n.inputs]
        graph.outputs = [new_id if o == old_id else o for o in graph.outputs]
