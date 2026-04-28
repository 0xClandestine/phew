"""Tests for phew/ir/importer.py tracer fixes."""

import mlx.core.fast as _fast_module
import pytest

_has_qsdpa = hasattr(_fast_module, "quantized_scaled_dot_product_attention")


def test_expand_dims_list_axis():
    """expand_dims with a list of axes should create multi-dim expansion."""
    import mlx.core as mx

    from phew.ir.importer import trace_to_graph

    def fn(x):
        return mx.expand_dims(x, [0, 2])

    x = mx.zeros((8, 4))
    g = trace_to_graph(fn, [x], {})
    # (8, 4) → expand at output positions 0 and 2 → (1, 8, 1, 4)
    assert len(g.outputs) == 1
    out_node = g[g.outputs[0]]
    assert out_node.shape == (1, 8, 1, 4)


def test_expand_dims_tuple_axis():
    """expand_dims with a tuple of axes mirrors list behaviour."""
    import mlx.core as mx

    from phew.ir.importer import trace_to_graph

    def fn(x):
        return mx.expand_dims(x, (0,))

    x = mx.zeros((6,))
    g = trace_to_graph(fn, [x], {})
    out_node = g[g.outputs[0]]
    assert out_node.shape == (1, 6)


def test_roll_stores_attrs():
    """roll tracer stores shift and axis in Elementwise.attrs."""
    import mlx.core as mx

    from phew.ir import Elementwise
    from phew.ir.importer import trace_to_graph

    def fn(x):
        return mx.roll(x, 3, axis=1)

    x = mx.zeros((4, 8))
    g = trace_to_graph(fn, [x], {})
    out_node = g[g.outputs[0]]
    assert isinstance(out_node, Elementwise)
    assert out_node.op == "roll"
    assert out_node.attrs["shift"] == 3
    assert out_node.attrs["axis"] == 1


def test_fast_rms_norm_traced():
    """mx.fast.rms_norm is intercepted during tracing."""
    import mlx.core as mx
    import mlx.core.fast as fast

    from phew.ir.importer import trace_to_graph
    from phew.ir.ops import FastRMSNorm

    def fn(x, w):
        return fast.rms_norm(x, w)

    x = mx.zeros((4, 8))
    w = mx.ones((8,))
    g = trace_to_graph(fn, [x, w], {})
    out_node = g[g.outputs[0]]
    assert isinstance(out_node, FastRMSNorm)
    assert out_node.shape == (4, 8)


def test_fast_rope_traced():
    """mx.fast.rope is intercepted during tracing."""
    import mlx.core as mx
    import mlx.core.fast as fast

    from phew.ir.importer import trace_to_graph
    from phew.ir.ops import FastRoPE

    def fn(x):
        return fast.rope(x, dims=8)

    x = mx.zeros((1, 1, 4, 16))
    g = trace_to_graph(fn, [x], {})
    out_node = g[g.outputs[0]]
    assert isinstance(out_node, FastRoPE)
    assert out_node.dims == 8


def test_fast_sdpa_traced():
    """mx.fast.scaled_dot_product_attention is intercepted during tracing."""
    import mlx.core as mx
    import mlx.core.fast as fast

    from phew.ir.importer import trace_to_graph
    from phew.ir.ops import FastScaledDotProductAttention

    def fn(q, k, v):
        return fast.scaled_dot_product_attention(q, k, v, scale=0.125)

    q = mx.zeros((1, 4, 8, 16))
    k = mx.zeros((1, 4, 8, 16))
    v = mx.zeros((1, 4, 8, 16))
    g = trace_to_graph(fn, [q, k, v], {})
    out_node = g[g.outputs[0]]
    assert isinstance(out_node, FastScaledDotProductAttention)
    assert out_node.scale == 0.125


def test_rms_norm_rule_scalar_weight_uses_none():
    """_match_rms_norm: 0-dim constant weight → FastRMSNorm with no weight input."""
    from phew.ir import Constant, Dtype, Elementwise, Graph, Input, Reduce
    from phew.ir.ops import FastRMSNorm
    from phew.rules.primitive_subst import PrimitiveSubstPass

    # Build the graph manually: rsqrt(mean(x*x) + eps) * x * scalar_weight
    g = Graph()
    x = g.add(Input(shape=(4, 8), dtype=Dtype.float32, name="x"))
    sq = g.add(Elementwise(shape=(4, 8), dtype=Dtype.float32, inputs=[x.id, x.id], op="mul"))
    mn = g.add(
        Reduce(
            shape=(4, 1), dtype=Dtype.float32, inputs=[sq.id], op="mean", axes=(1,), keepdims=True
        )
    )
    eps = g.add(Constant(shape=(), dtype=Dtype.float32, value=1e-5))
    add = g.add(Elementwise(shape=(4, 1), dtype=Dtype.float32, inputs=[mn.id, eps.id], op="add"))
    inv = g.add(Elementwise(shape=(4, 1), dtype=Dtype.float32, inputs=[add.id], op="rsqrt"))
    inner = g.add(Elementwise(shape=(4, 8), dtype=Dtype.float32, inputs=[x.id, inv.id], op="mul"))
    # scalar weight (0-dim constant)
    w_scalar = g.add(Constant(shape=(), dtype=Dtype.float32, value=1.0))
    out = g.add(
        Elementwise(shape=(4, 8), dtype=Dtype.float32, inputs=[inner.id, w_scalar.id], op="mul")
    )
    g.outputs = [out.id]

    PrimitiveSubstPass().run(g)

    # The scalar weight node should not appear in FastRMSNorm inputs
    found = [n for n in g._nodes.values() if isinstance(n, FastRMSNorm)]
    assert found, "FastRMSNorm should have been created"
    assert len(found[0].inputs) == 1, "Scalar weight should be excluded; weight=None"


@pytest.mark.skipif(
    not _has_qsdpa, reason="MLX version lacks quantized_scaled_dot_product_attention"
)
def test_fast_quantized_sdpa_traced():
    """mx.fast.quantized_scaled_dot_product_attention is intercepted during tracing."""
    import mlx.core as mx
    import mlx.core.fast as fast

    from phew.ir.importer import trace_to_graph
    from phew.ir.ops import FastQuantizedScaledDotProductAttention

    B, H, S, D = 1, 4, 8, 16
    groups = D // 64 if D >= 64 else 1
    packed_cols = D * 4 // 8  # bits=4

    def fn(q, k, v, sk, bk, sv, bv):
        return fast.quantized_scaled_dot_product_attention(
            q, k, v, sk, bk, sv, bv, scale=0.25, bits=4, group_size=D
        )

    q = mx.zeros((B, H, S, D))
    k = mx.zeros((B, H, S, packed_cols), dtype=mx.uint32)
    v = mx.zeros((B, H, S, packed_cols), dtype=mx.uint32)
    sk = mx.ones((B, H, S, groups))
    bk = mx.zeros((B, H, S, groups))
    sv = mx.ones((B, H, S, groups))
    bv = mx.zeros((B, H, S, groups))

    g = trace_to_graph(fn, [q, k, v, sk, bk, sv, bv], {})
    out_node = g[g.outputs[0]]
    assert isinstance(out_node, FastQuantizedScaledDotProductAttention)
    assert out_node.scale == 0.25
    assert out_node.bits == 4
