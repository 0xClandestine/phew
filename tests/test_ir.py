"""Unit tests for the μGraph IR."""

from phew.ir import (
    Dtype,
    Elementwise,
    Graph,
    Input,
    MatMul,
    MemDep,
)


def test_node_ids_unique():
    g = Graph()
    a = g.add(Input(shape=(4, 4), dtype=Dtype.float32, name="a"))
    b = g.add(Input(shape=(4, 4), dtype=Dtype.float32, name="b"))
    assert a.id != b.id


def test_graph_topo_order():
    g = Graph()
    a = g.add(Input(shape=(4, 4), name="a"))
    b = g.add(Input(shape=(4, 4), name="b"))
    c = g.add(MatMul(shape=(4, 4), inputs=[a.id, b.id]))
    g.outputs = [c.id]

    order = g.topo_order()
    ids = [n.id for n in order]
    assert ids.index(a.id) < ids.index(c.id)
    assert ids.index(b.id) < ids.index(c.id)


def test_graph_replace():
    g = Graph()
    a = g.add(Input(shape=(4,), name="x"))
    old = g.add(Elementwise(shape=(4,), inputs=[a.id], op="exp"))
    g.outputs = [old.id]

    new = Elementwise(shape=(4,), inputs=[a.id], op="sqrt")
    g.replace(old.id, new)

    assert new.id in g
    assert old.id not in g
    assert g.outputs == [new.id]


def test_dtype_itemsize():
    assert Dtype.float32.itemsize == 4
    assert Dtype.float16.itemsize == 2
    assert Dtype.int8.itemsize == 1


def test_memdep_flags():
    dep = MemDep.device_mem | MemDep.threadgroup_mem
    assert MemDep.device_mem in dep
    assert MemDep.register not in dep


def test_graph_successors():
    g = Graph()
    a = g.add(Input(shape=(4,), name="a"))
    b = g.add(Elementwise(shape=(4,), inputs=[a.id], op="exp"))
    c = g.add(Elementwise(shape=(4,), inputs=[a.id], op="sqrt"))

    succs = g.successors(a.id)
    succ_ids = {n.id for n in succs}
    assert b.id in succ_ids
    assert c.id in succ_ids


def test_graph_nbytes():
    g = Graph()
    a = g.add(Input(shape=(128, 256), dtype=Dtype.float32, name="x"))
    assert a.nbytes == 128 * 256 * 4
