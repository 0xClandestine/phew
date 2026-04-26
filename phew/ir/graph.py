"""μGraph: directed acyclic graph of Node objects."""

from __future__ import annotations

from typing import Iterator

from .node import Node, NodeId


class Graph:
    """Container for a μGraph IR.

    Nodes are stored in insertion order. Edges are encoded by
    `node.inputs` (list of NodeIds of predecessor nodes).

    Usage
    -----
    g = Graph()
    a = g.add(Input(shape=(4, 4), name="x"))
    b = g.add(Input(shape=(4, 4), name="w"))
    c = g.add(MatMul(shape=(4, 4), inputs=[a.id, b.id]))
    g.outputs = [c.id]
    """

    def __init__(self) -> None:
        self._nodes: dict[NodeId, Node] = {}
        self._order: list[NodeId] = []
        self.outputs: list[NodeId] = []
        self.name: str = ""

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, node: Node) -> Node:
        self._nodes[node.id] = node
        self._order.append(node.id)
        return node

    def remove(self, node_id: NodeId) -> None:
        self._nodes.pop(node_id, None)
        self._order.remove(node_id)

    def replace(self, old_id: NodeId, new_node: Node) -> Node:
        """Replace a node and update all references to it."""
        self.add(new_node)
        for node in self._nodes.values():
            node.inputs = [new_node.id if i == old_id else i for i in node.inputs]
        self.outputs = [new_node.id if o == old_id else o for o in self.outputs]
        self.remove(old_id)
        return new_node

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __getitem__(self, node_id: NodeId) -> Node:
        return self._nodes[node_id]

    def __contains__(self, node_id: NodeId) -> bool:
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def nodes(self) -> list[Node]:
        return [self._nodes[nid] for nid in self._order]

    def topo_order(self) -> list[Node]:
        """Return nodes in topological order (sources first)."""
        visited: set[NodeId] = set()
        result: list[Node] = []

        def visit(nid: NodeId) -> None:
            if nid in visited:
                return
            visited.add(nid)
            node = self._nodes[nid]
            for inp in node.inputs:
                if inp in self._nodes:
                    visit(inp)
            result.append(node)

        for nid in self._order:
            visit(nid)
        return result

    def successors(self, node_id: NodeId) -> list[Node]:
        """Nodes that directly consume the given node's output."""
        return [n for n in self._nodes.values() if node_id in n.inputs]

    def predecessors(self, node_id: NodeId) -> list[Node]:
        return [self._nodes[i] for i in self._nodes[node_id].inputs if i in self._nodes]

    def output_nodes(self) -> list[Node]:
        return [self._nodes[o] for o in self.outputs if o in self._nodes]

    def inputs(self) -> list[Node]:
        """Return all Input nodes (sources with no inputs)."""
        from .ops import Input

        return [n for n in self._nodes.values() if isinstance(n, Input)]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def copy(self) -> "Graph":
        """Shallow copy — nodes are shared."""
        g = Graph()
        g._nodes = dict(self._nodes)
        g._order = list(self._order)
        g.outputs = list(self.outputs)
        g.name = self.name
        return g

    def __repr__(self) -> str:
        lines = [f"Graph({self.name!r}, {len(self)} nodes)"]
        for node in self.topo_order():
            lines.append(f"  {node}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Node]:
        return iter(self.topo_order())
