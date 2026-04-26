"""Unit tests for the cost model and pruner."""

from phew.cost import CostModel, NodeCost, should_prune
from phew.ir import Dtype, Elementwise, Graph, Input, MatMul


def _make_matmul(M=128, K=128, N=128):
    g = Graph()
    a = g.add(Input(shape=(M, K), dtype=Dtype.float32, name="a"))
    b = g.add(Input(shape=(K, N), dtype=Dtype.float32, name="b"))
    c = g.add(MatMul(shape=(M, N), inputs=[a.id, b.id]))
    g.outputs = [c.id]
    return g, c


def test_matmul_cost_positive():
    _, node = _make_matmul()
    cm = CostModel()
    cost = cm.node_cost(node)
    assert cost.flops > 0
    assert cost.bytes_moved > 0


def test_elementwise_cost():
    g = Graph()
    x = g.add(Input(shape=(1024,), dtype=Dtype.float32, name="x"))
    e = g.add(Elementwise(shape=(1024,), inputs=[x.id], op="exp"))
    cm = CostModel()
    cost = cm.node_cost(e)
    assert cost.flops == 1024.0
    assert cost.bytes_moved > 0


def test_pruner_threadgroup_too_large():
    pruned, reason = should_prune(
        NodeCost(),
        NodeCost(),
        threadgroup_size=2048,
    )
    assert pruned
    assert "1024" in reason


def test_pruner_passes_ok():
    best = NodeCost(bytes_moved=1000.0, flops=5000.0)
    cand = NodeCost(bytes_moved=800.0, flops=4000.0)
    pruned, reason = should_prune(cand, best, threadgroup_size=256)
    assert not pruned


def test_pruner_bytes_exceed():
    best = NodeCost(bytes_moved=1000.0)
    cand = NodeCost(bytes_moved=2000.0)
    pruned, reason = should_prune(cand, best, threadgroup_size=256)
    assert pruned


def test_occupancy_model():
    c = NodeCost(registers_per_thread=32)
    assert c.occupancy == 1.0
    c2 = NodeCost(registers_per_thread=200)
    assert 0 < c2.occupancy < 1.0
