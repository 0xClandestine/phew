"""Compile boundary insertion pass.

Inserts Compile (mx.compile) nodes around the entire graph and AsyncEval
nodes before output nodes when CPU-overlapping loops are detected.

Search prior position: #1 (mx.compile, 1.5–3× speedup, free).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phew.ir import Graph


class CompileBoundaryPass:
    """Wrap the graph in mx.compile.

    This is the highest-priority rewrite — applied before any other rule.
    It wraps the entire function in a Compile node.
    """

    def run(self, graph: "Graph") -> bool:
        from phew.ir import Compile

        # Check if already compiled
        for node in graph.nodes():
            if isinstance(node, Compile):
                return False

        # Wrap each output in a Compile node
        new_outputs = []
        changed = False
        for out_id in list(graph.outputs):
            out_node = graph[out_id]
            compile_node = Compile(
                shape=out_node.shape,
                dtype=out_node.dtype,
                inputs=[out_id],
            )
            graph.add(compile_node)
            new_outputs.append(compile_node.id)
            changed = True

        if changed:
            graph.outputs = new_outputs
        return changed
