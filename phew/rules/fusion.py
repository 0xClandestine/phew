"""Elementwise fusion pass — replaces chains of Elementwise/Cast nodes with
a single MetalKernel node backed by an MSL kernel.

Strategy
--------
1. Walk the graph in topological order, collecting maximal "fuseable" chains:
   - Node types: Elementwise, Cast, Constant
   - A node is safe to fuse if every path to an external consumer goes through
     a single exit point (no internal fan-out to multiple non-fused consumers).
2. Chains with fewer than MIN_CHAIN_ELEMS Elementwise nodes are skipped (kernel
   launch overhead isn't worth it).
3. Each surviving chain is replaced by a MetalKernel node.  External inputs
   (non-chain predecessors) become kernel buffer inputs.  External outputs
   (chain nodes consumed outside the chain, or graph outputs) become kernel
   buffer outputs.

Limitations (initial version)
------------------------------
- Elementwise / same-numel ops only — no Reduce inside the kernel.
- All external inputs must have the same numel as the kernel output (no
  implicit broadcasting from different shapes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phew.ir import Graph
    from phew.ir.node import Node

# Minimum number of Elementwise (non-Constant, non-Cast) nodes to fuse.
MIN_CHAIN_ELEMS = 3


def _is_fuseable(node: "Node") -> bool:
    from phew.ir.ops import Cast, Constant, Elementwise

    return isinstance(node, (Elementwise, Cast, Constant))


def _chain_numel(nodes: "list[Node]") -> int:
    """Return the numel shared by all non-Constant nodes in the chain."""
    from phew.ir.ops import Constant

    for n in nodes:
        if not isinstance(n, Constant):
            return n.numel
    return 0


def _any_ext_input_descends_from_chain(
    ext_inputs: "list[Node]",
    chain_ids: "set[int]",
    graph: "Graph",
) -> bool:
    """Return True if any external input has a chain node as a transitive ancestor.

    This detects cases where fusing would create a data-flow cycle through an
    external node (e.g. exp→Reduce→div where both exp and div are in the chain
    but the Reduce is outside).
    """

    def _has_chain_ancestor(start_id: int) -> bool:
        from collections import deque

        queue = deque([start_id])
        seen: set[int] = set()
        while queue:
            nid = queue.popleft()
            if nid in seen:
                continue
            seen.add(nid)
            if nid in chain_ids:
                return True
            if nid not in graph:
                continue
            for inp_id in graph[nid].inputs:
                if inp_id not in seen:
                    queue.append(inp_id)
        return False

    for node in ext_inputs:
        # Check the node's own inputs (not the node itself — it's external)
        for inp_id in node.inputs:
            if _has_chain_ancestor(inp_id):
                return True
    return False


class ElementwiseFusionPass:
    """Replace elementwise chains with MetalKernel nodes.

    Parameters
    ----------
    min_chain_elems:
        Minimum number of Elementwise ops (excluding Constants and Casts) to
        trigger fusion.
    """

    def __init__(self, min_chain_elems: int = MIN_CHAIN_ELEMS) -> None:
        self.min_chain_elems = min_chain_elems

    def run(self, graph: "Graph") -> bool:
        """Mutate graph in-place.  Return True if any fusion was applied."""
        chains = self._find_chains(graph)
        if not chains:
            return False

        changed = False
        for chain in chains:
            if self._fuse_chain(chain, graph):
                changed = True
        return changed

    # ------------------------------------------------------------------
    # Chain discovery
    # ------------------------------------------------------------------

    def _find_chains(self, graph: "Graph") -> list[list["Node"]]:
        """Return a list of fuseable chains, longest first."""
        from phew.ir.ops import Elementwise

        topo = graph.topo_order()
        in_chain: set[int] = set()
        chains: list[list[Node]] = []

        for start in topo:
            if start.id in in_chain or not _is_fuseable(start):
                continue
            # Grow a chain starting from `start`
            chain = self._grow_chain(start, graph, in_chain)
            n_elems = sum(1 for n in chain if isinstance(n, Elementwise))
            if n_elems >= self.min_chain_elems:
                for n in chain:
                    in_chain.add(n.id)
                chains.append(chain)

        return chains

    def _grow_chain(
        self,
        start: "Node",
        graph: "Graph",
        already_claimed: set[int],
    ) -> list["Node"]:
        """Collect the maximal fuseable subgraph reachable from *start*.

        BFS over both downstream (consumers) and upstream (producers) edges,
        staying within fuseable nodes.  This correctly handles diamond-shaped
        subgraphs where two branches fan out from a shared input and then
        merge into a single computation node.
        """
        subgraph_ids: set[int] = set()
        queue: list[Node] = [start]

        while queue:
            node = queue.pop(0)
            if node.id in subgraph_ids or node.id in already_claimed:
                continue
            if not _is_fuseable(node):
                continue
            subgraph_ids.add(node.id)

            # Extend downstream through fuseable consumers
            for succ in graph.successors(node.id):
                if (
                    succ.id not in subgraph_ids
                    and _is_fuseable(succ)
                    and succ.id not in already_claimed
                ):
                    queue.append(succ)

            # Extend upstream through fuseable producers
            for inp_id in node.inputs:
                if inp_id not in graph:
                    continue
                pred = graph[inp_id]
                if (
                    pred.id not in subgraph_ids
                    and _is_fuseable(pred)
                    and pred.id not in already_claimed
                ):
                    queue.append(pred)

        # Return in stable topological order
        return [n for n in graph.topo_order() if n.id in subgraph_ids]

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _fuse_chain(self, chain: "list[Node]", graph: "Graph") -> bool:
        """Replace a chain with a MetalKernel node. Return True on success."""
        from phew.emit.msl_codegen import SubgraphMSLCodegen
        from phew.ir.deps import MemDep
        from phew.ir.ops import Constant, MetalKernel

        chain_ids = {n.id for n in chain}

        # External inputs: predecessors of chain nodes that are outside the chain
        # (excluding Constants, which are inlined in MSL)
        ext_inputs: list[Node] = []
        ext_input_ids: set[int] = set()
        for node in chain:
            if isinstance(node, Constant):
                continue
            for inp_id in node.inputs:
                if inp_id in chain_ids or inp_id in ext_input_ids:
                    continue
                if inp_id in graph:
                    ext_inputs.append(graph[inp_id])
                    ext_input_ids.add(inp_id)

        # External outputs: chain nodes consumed by nodes outside the chain,
        # or that are graph outputs
        graph_output_ids = set(graph.outputs)
        ext_outputs: list[Node] = []
        ext_output_ids: set[int] = set()
        for node in chain:
            if isinstance(node, Constant):
                continue
            is_graph_out = node.id in graph_output_ids
            has_ext_consumer = any(s.id not in chain_ids for s in graph.successors(node.id))
            if (is_graph_out or has_ext_consumer) and node.id not in ext_output_ids:
                ext_outputs.append(node)
                ext_output_ids.add(node.id)

        if not ext_outputs:
            return False

        # Validate: chain has a well-defined output numel
        target_numel = _chain_numel(chain)
        if target_numel == 0:
            return False

        # Guard: all external inputs must have the same numel as the output so
        # the MSL body can use a simple identity index `elem`.  Inputs with
        # different shapes require broadcast indexing with trace-time strides,
        # which breaks when the verifier runs on a different problem size.
        for inp in ext_inputs:
            if len(inp.shape) > 0 and inp.numel != target_numel:
                return False

        # Guard against cycles-through-external-nodes: if any external input
        # has a chain node as a transitive ancestor, fusing would create a
        # data-flow cycle (e.g., exp→Reduce(sum)→div where exp and div are both
        # inside the kernel but Reduce is outside).
        if _any_ext_input_descends_from_chain(ext_inputs, chain_ids, graph):
            return False

        # Generate MSL
        try:
            codegen = SubgraphMSLCodegen()
            source, inp_names, out_names = codegen.emit(chain, ext_inputs, ext_outputs)
        except Exception:
            return False

        # Output dtype / shape from the first external output
        out_dtypes = [n.dtype for n in ext_outputs]
        out_shapes = [n.shape for n in ext_outputs]

        # Grid: one thread per element (1D)
        numel = ext_outputs[0].numel
        tg = min(256, numel)
        import math

        grid_x = math.ceil(numel / tg) * tg

        kernel_node = MetalKernel(
            shape=ext_outputs[0].shape,
            dtype=ext_outputs[0].dtype,
            inputs=[n.id for n in ext_inputs],
            source=source,
            header="",
            input_names=inp_names,
            output_names=out_names,
            output_shapes=out_shapes,
            output_dtypes=out_dtypes,
            input_shapes=[n.shape for n in ext_inputs],
            threadgroup=(tg, 1, 1),
            grid=(grid_x, 1, 1),
            deps=MemDep.device_mem,
            attrs={"fusion_generated": True},
        )
        graph.add(kernel_node)

        # Redirect all external consumers of each ext_output to the kernel node.
        if len(ext_outputs) == 1:
            old_id = ext_outputs[0].id
            for node in graph.nodes():
                if node.id == kernel_node.id:
                    continue
                node.inputs = [kernel_node.id if i == old_id else i for i in node.inputs]
            graph.outputs = [kernel_node.id if o == old_id else o for o in graph.outputs]
        else:
            from phew.ir.ops import MetalKernelSelect

            for i, ext_out in enumerate(ext_outputs):
                sel = MetalKernelSelect(
                    shape=ext_out.shape,
                    dtype=ext_out.dtype,
                    inputs=[kernel_node.id],
                    output_idx=i,
                )
                graph.add(sel)
                old_id = ext_out.id
                for node in graph.nodes():
                    if node.id in (kernel_node.id, sel.id):
                        continue
                    node.inputs = [sel.id if i_id == old_id else i_id for i_id in node.inputs]
                graph.outputs = [sel.id if o == old_id else o for o in graph.outputs]

        # Remove fused nodes from graph
        for node in chain:
            if node.id in graph:
                graph.remove(node.id)

        return True
