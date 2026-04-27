"""QuantizationPass: opt-in 4-bit weight quantization.

Replaces MatMul(x, W) with QuantizedMatMul(x, W, bits=4, group_size=64)
when the weight operand is a parameter (Input with is_parameter=True in attrs)
or a Constant node. Requires explicit opt-in.

From the search prior (CLAUDE.md): #4 highest-yield — nn.quantize(bits=4)
for matmul-heavy graphs, after mx.compile and fast.* primitives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phew.ir.graph import Graph


class QuantizationPass:
    """Opt-in 4-bit weight quantization.

    Replaces MatMul(x, W) with QuantizedMatMul(x, W, bits=bits, group_size=group_size)
    when the weight (second operand) is a parameter or constant.

    Parameters
    ----------
    bits:
        Quantization bit-width. 4 (default) or 8.
    group_size:
        Group size for group-wise quantization. Default 64.
    """

    def __init__(self, bits: int = 4, group_size: int = 64) -> None:
        self.bits = bits
        self.group_size = group_size

    def run(self, graph: "Graph") -> "Graph":
        """Mutate *graph* in place. Returns graph."""
        from phew.ir.ops import Constant, Input, MatMul, QuantizedMatMul

        for node in list(graph.topo_order()):
            if not isinstance(node, MatMul):
                continue
            if len(node.inputs) < 2:
                continue

            weight_id = node.inputs[1]
            if weight_id not in graph:
                continue
            weight_node = graph[weight_id]

            # Only quantize when weight is a parameter or constant —
            # activations change every forward pass and can't be pre-quantized.
            is_param = (
                isinstance(weight_node, Input)
                and bool(weight_node.attrs.get("is_parameter", False))
            ) or isinstance(weight_node, Constant)

            if not is_param:
                continue

            qmatmul = QuantizedMatMul(
                shape=node.shape,
                dtype=node.dtype,
                inputs=list(node.inputs),
                bits=self.bits,
                group_size=self.group_size,
            )
            graph.replace(node.id, qmatmul)

        return graph
