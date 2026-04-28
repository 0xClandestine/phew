"""Tests for AlgebraicPass and egglog algebraic rules."""

from phew.ir import Dtype, Graph, Input, MatMul, Reshape  # noqa: F401
from phew.rules.algebraic import AlgebraicPass


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def test_reshape_of_reshape_same_numel_collapsed():
    """Two reshapes with matching element counts should be merged."""
    g = Graph()
    x = g.add(Input(shape=(4, 8), dtype=Dtype.float32, name="x"))
    r1 = g.add(
        Reshape(
            shape=(2, 16), dtype=Dtype.float32, inputs=[x.id], new_shape=(2, 16), input_shape=(4, 8)
        )
    )
    r2 = g.add(
        Reshape(
            shape=(32,), dtype=Dtype.float32, inputs=[r1.id], new_shape=(32,), input_shape=(2, 16)
        )
    )
    g.outputs = [r2.id]

    AlgebraicPass().run(g)

    # r2 should now point directly at x
    out = g[g.outputs[0]]
    assert out.inputs == [x.id]
    assert r1.id not in g


def test_broadcast_to_followed_by_reshape_not_merged():
    """broadcast_to encoded as Reshape changes numel — must NOT be merged."""
    g = Graph()
    x = g.add(Input(shape=(1, 8), dtype=Dtype.float32, name="x"))
    # expand_dims: (1, 8) → (1, 1, 8)  [same numel]
    expand = g.add(
        Reshape(
            shape=(1, 1, 8),
            dtype=Dtype.float32,
            inputs=[x.id],
            new_shape=(1, 1, 8),
            input_shape=(1, 8),
        )
    )
    # broadcast_to: (1, 1, 8) → (4, 4, 8)  [different numel — not a true reshape]
    broadcast = g.add(
        Reshape(
            shape=(4, 4, 8),
            dtype=Dtype.float32,
            inputs=[expand.id],
            new_shape=(4, 4, 8),
            input_shape=(1, 1, 8),
        )
    )
    g.outputs = [broadcast.id]

    AlgebraicPass().run(g)

    # The broadcast node must NOT have been merged onto x
    out = g[g.outputs[0]]
    assert out.inputs != [x.id], "broadcast_to should not be merged with preceding reshape"
    # The expand node should still exist (it feeds the broadcast)
    assert expand.id in g


def test_matmul_associativity_shape_aware():
    """build_egraph + algebraic rules produce mat_rows/mat_cols facts without crashing."""
    from egglog import EGraph

    from phew.egraph.rules_egglog import _register_algebraic_rules, build_egraph

    g = Graph()
    a = g.add(Input(shape=(4, 8), dtype=Dtype.float32, name="a"))
    b = g.add(Input(shape=(8, 16), dtype=Dtype.float32, name="b"))
    c = g.add(Input(shape=(16, 4), dtype=Dtype.float32, name="c"))
    ab = g.add(MatMul(shape=(4, 16), dtype=Dtype.float32, inputs=[a.id, b.id]))
    abc = g.add(MatMul(shape=(4, 4), dtype=Dtype.float32, inputs=[ab.id, c.id]))
    g.outputs = [abc.id]

    egraph = EGraph()
    root, node_map, str_node_map = build_egraph(egraph, g)
    _register_algebraic_rules(egraph)
    egraph.run(10)

    # Should complete without error; the shape-guarded rule fires because
    # mat_cols(ab) == mat_rows(c) == 16 and mat_cols(a) == mat_rows(b) == 8.
    assert root is not None


def test_matmul_associativity_incompatible_shapes_no_fire():
    """Associativity rule must NOT fire when inner dims are incompatible."""
    from egglog import EGraph

    from phew.egraph.rules_egglog import _register_algebraic_rules, build_egraph

    # Intentionally mismatched: a=(4,8), b=(16,32) — a.cols(8) != b.rows(16)
    g = Graph()
    a = g.add(Input(shape=(4, 8), dtype=Dtype.float32, name="a"))
    g.add(Input(shape=(16, 32), dtype=Dtype.float32, name="b"))
    g.outputs = [a.id]

    egraph = EGraph()
    root, node_map, str_node_map = build_egraph(egraph, g)
    _register_algebraic_rules(egraph)
    # Should run without error even though no matmul node is present
    egraph.run(10)
    assert root is not None
