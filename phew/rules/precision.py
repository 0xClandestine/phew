"""PrecisionPass: opt-in fp32 → bf16/fp16 promotion.

Inserts Cast nodes at graph inputs and a Cast back to fp32 at graph outputs
when the user opts into a reduced-precision substitution class.

This halves memory bandwidth for activation tensors and is the #2 highest-yield
optimization after mx.compile (from the search prior in CLAUDE.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phew.ir.graph import Graph

from phew.ir.dtype import Dtype


class PrecisionPass:
    """Opt-in fp32 → bf16 (or fp16) precision reduction.

    Inserts Cast(x, target_dtype) after each fp32 Input node and a Cast back
    to float32 before each graph output that was originally fp32, so the
    function signature is unchanged but internal activations run at lower
    precision.

    Parameters
    ----------
    target_dtype:
        Target reduced-precision dtype. Defaults to bfloat16 (recommended for
        Apple Silicon; better dynamic range than fp16).
    """

    def __init__(self, target_dtype: Dtype = Dtype.bfloat16) -> None:
        self.target_dtype = target_dtype

    def run(self, graph: "Graph") -> "Graph":
        """Mutate *graph* in place. Returns graph."""
        from phew.ir.ops import Cast, Input

        # Step 1: insert downcast after each fp32 Input
        for node in list(graph.topo_order()):
            if not isinstance(node, Input):
                continue
            if node.dtype != Dtype.float32:
                continue

            cast_down = Cast(
                shape=node.shape,
                dtype=self.target_dtype,
                inputs=[node.id],
                target_dtype=self.target_dtype,
            )
            graph.add(cast_down)

            # Redirect all consumers of this Input (except cast_down itself)
            for n in graph.nodes():
                if n.id == cast_down.id:
                    continue
                n.inputs = [cast_down.id if i == node.id else i for i in n.inputs]

            # Update graph outputs that pointed directly at the Input
            graph.outputs = [cast_down.id if o == node.id else o for o in graph.outputs]

        # Step 2: insert upcast back to fp32 at each graph output that is now
        # reduced-precision, to preserve the original function's output dtype.
        new_outputs = []
        for out_id in graph.outputs:
            if out_id not in graph:
                new_outputs.append(out_id)
                continue
            out_node = graph[out_id]
            if out_node.dtype == self.target_dtype:
                cast_up = Cast(
                    shape=out_node.shape,
                    dtype=Dtype.float32,
                    inputs=[out_id],
                    target_dtype=Dtype.float32,
                )
                graph.add(cast_up)
                new_outputs.append(cast_up.id)
            else:
                new_outputs.append(out_id)
        graph.outputs = new_outputs

        return graph
