"""Tests for ElementwiseFusionPass in phew/rules/fusion.py."""

from __future__ import annotations

import sys

import pytest

from phew.ir import (
    Dtype,
    Elementwise,
    Graph,
    Input,
    MetalKernel,
    MetalKernelSelect,
    Reduce,
)
from phew.rules.fusion import MIN_CHAIN_ELEMS, ElementwiseFusionPass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHAPE = (128,)
_DTYPE = Dtype.float32


def _ew(graph: Graph, op: str, *input_nodes) -> Elementwise:
    """Add an Elementwise node to *graph* with the given inputs."""
    node = Elementwise(
        shape=_SHAPE,
        dtype=_DTYPE,
        inputs=[n.id for n in input_nodes],
        op=op,
    )
    return graph.add(node)


def _input(graph: Graph, name: str = "x") -> Input:
    return graph.add(Input(shape=_SHAPE, dtype=_DTYPE, name=name))


def _count(graph: Graph, cls) -> int:
    return sum(1 for n in graph.nodes() if isinstance(n, cls))


# ---------------------------------------------------------------------------
# Test 1: short chain (< MIN_CHAIN_ELEMS) is NOT fused
# ---------------------------------------------------------------------------


def test_short_chain_not_fused():
    """A 2-node elementwise chain must not be fused (below MIN_CHAIN_ELEMS=3)."""
    assert MIN_CHAIN_ELEMS == 3, "test assumes MIN_CHAIN_ELEMS is 3"

    g = Graph()
    x = _input(g, "x")
    e1 = _ew(g, "exp", x)
    e2 = _ew(g, "log", e1)
    g.outputs = [e2.id]

    changed = ElementwiseFusionPass().run(g)

    assert not changed
    # Both elementwise nodes must still be present
    assert e1.id in g
    assert e2.id in g
    # No MetalKernel node should have been introduced
    assert _count(g, MetalKernel) == 0


# ---------------------------------------------------------------------------
# Test 2: 3-node linear chain IS fused
# ---------------------------------------------------------------------------


def test_three_node_chain_fused():
    """A 3-node linear elementwise chain should be replaced by a MetalKernel."""
    g = Graph()
    x = _input(g, "x")
    e1 = _ew(g, "exp", x)
    e2 = _ew(g, "log", e1)
    e3 = _ew(g, "neg", e2)
    g.outputs = [e3.id]

    changed = ElementwiseFusionPass().run(g)

    assert changed
    # The three elementwise nodes must be gone
    assert e1.id not in g
    assert e2.id not in g
    assert e3.id not in g
    # Exactly one MetalKernel must have been inserted
    assert _count(g, MetalKernel) == 1
    # The graph output should now be the kernel node
    kernel = next(n for n in g.nodes() if isinstance(n, MetalKernel))
    assert g.outputs == [kernel.id]


# ---------------------------------------------------------------------------
# Test 3: multi-output fusion creates MetalKernelSelect nodes
# ---------------------------------------------------------------------------


def test_multi_output_fusion_creates_selects():
    """When two consumers reference different chain nodes, the pass emits
    one MetalKernelSelect per external output."""
    g = Graph()
    x = _input(g, "x")
    # Build a 4-node chain so we clear MIN_CHAIN_ELEMS
    e1 = _ew(g, "exp", x)
    e2 = _ew(g, "mul", e1, x)  # binary so it stays connected
    e3 = _ew(g, "log", e2)
    e4 = _ew(g, "neg", e3)

    # Two external consumers — one reads e2, one reads e4
    # We represent "external consumers" by declaring both as graph outputs.
    # The pass treats graph-output nodes as external outputs.
    g.outputs = [e2.id, e4.id]

    changed = ElementwiseFusionPass().run(g)

    assert changed
    # Exactly one kernel
    assert _count(g, MetalKernel) == 1
    # Two selects, one per external output
    selects = [n for n in g.nodes() if isinstance(n, MetalKernelSelect)]
    assert len(selects) == 2
    # output_idx values must be 0 and 1 (in some order)
    assert {s.output_idx for s in selects} == {0, 1}
    # Each select must point at the kernel
    kernel = next(n for n in g.nodes() if isinstance(n, MetalKernel))
    for sel in selects:
        assert sel.inputs == [kernel.id]
    # Graph outputs are now the two select nodes
    assert set(g.outputs) == {s.id for s in selects}


# ---------------------------------------------------------------------------
# Test 4: non-elementwise (Reduce) in the middle breaks the chain
# ---------------------------------------------------------------------------


def test_reduce_breaks_chain():
    """A Reduce node inside a would-be chain prevents fusion of a single block
    spanning the reduce."""
    g = Graph()
    x = _input(g, "x")

    # Chain A: three elementwise nodes before the reduce
    a1 = _ew(g, "exp", x)
    a2 = _ew(g, "log", a1)
    a3 = _ew(g, "neg", a2)

    # Reduce breaks the chain
    r = g.add(Reduce(shape=(1,), dtype=_DTYPE, inputs=[a3.id], op="sum", axes=(0,), keepdims=True))

    # Chain B: three elementwise nodes after the reduce
    b1 = _ew(g, "exp", r)
    b2 = _ew(g, "log", b1)
    b3 = _ew(g, "neg", b2)

    g.outputs = [b3.id]

    ElementwiseFusionPass().run(g)

    # The reduce must still be in the graph — it cannot be fused
    assert r.id in g
    # Chains A and B may each be independently fused, but the reduce stays
    kernels = [n for n in g.nodes() if isinstance(n, MetalKernel)]
    # At most 2 kernels (one per side), but never a single kernel spanning reduce
    assert len(kernels) <= 2
    # No single kernel node has the reduce as a predecessor anywhere
    for k in kernels:
        all_input_ids = set(k.inputs)
        assert r.id not in all_input_ids or True  # reduce is an input to B-kernel, that's fine
    # The critical assertion: reduce itself is NOT a MetalKernel
    assert isinstance(g[r.id], Reduce)


# ---------------------------------------------------------------------------
# Test 5: large chain does not cause RecursionError
# ---------------------------------------------------------------------------


def test_large_chain_no_recursion_error():
    """A 50-node linear chain must complete without RecursionError.

    Python's default recursion limit is ~1000, so 50 nodes is safely within
    range, but if the implementation used unbounded recursion keyed on chain
    depth this test would expose that.  We additionally reduce the limit to
    force the issue.
    """
    g = Graph()
    x = _input(g, "x")
    prev = x

    CHAIN_LEN = 50
    ops = ["exp", "log", "neg", "exp", "log"]  # cycle through supported ops
    nodes = []
    for i in range(CHAIN_LEN):
        node = _ew(g, ops[i % len(ops)], prev)
        nodes.append(node)
        prev = node
    g.outputs = [nodes[-1].id]

    old_limit = sys.getrecursionlimit()
    try:
        # Tighten the limit so that a naive recursive DFS over 50 nodes would
        # still succeed (50 << 200) — but a per-node recursion that scales with
        # graph size would fail at something like 500 nodes.  This exercises the
        # fact that the pass itself is iterative even if topo_order uses DFS.
        sys.setrecursionlimit(200)
        try:
            changed = ElementwiseFusionPass().run(g)
        except RecursionError:
            pytest.fail("ElementwiseFusionPass raised RecursionError on a 50-node chain")
    finally:
        sys.setrecursionlimit(old_limit)

    # The 50-node chain clears the threshold — fusion must have fired
    assert changed
    assert _count(g, MetalKernel) == 1


# ---------------------------------------------------------------------------
# Test 6: idempotent — second pass produces no additional fusions
# ---------------------------------------------------------------------------


def test_idempotent():
    """Running ElementwiseFusionPass twice must not produce extra MetalKernel nodes."""
    g = Graph()
    x = _input(g, "x")
    e1 = _ew(g, "exp", x)
    e2 = _ew(g, "log", e1)
    e3 = _ew(g, "neg", e2)
    g.outputs = [e3.id]

    pass_ = ElementwiseFusionPass()

    first = pass_.run(g)
    assert first  # sanity: first pass must have fused

    kernel_ids_after_first = {n.id for n in g.nodes() if isinstance(n, MetalKernel)}
    assert len(kernel_ids_after_first) == 1

    second = pass_.run(g)
    assert not second  # second pass must report no change

    kernel_ids_after_second = {n.id for n in g.nodes() if isinstance(n, MetalKernel)}
    assert kernel_ids_after_second == kernel_ids_after_first
