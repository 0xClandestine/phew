"""Tests for phew/emit/msl_codegen.py — SubgraphMSLCodegen.

Constructs IR nodes directly (no tracer) and checks emitted MSL source.
"""

from __future__ import annotations

import pytest

from phew.emit.msl_codegen import SubgraphMSLCodegen
from phew.ir.dtype import Dtype
from phew.ir.ops import Elementwise, Input

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _input(shape=(4,), dtype=Dtype.float32) -> Input:
    return Input(shape=shape, dtype=dtype)


def _ew(op: str, inputs: list, shape=(4,), dtype=Dtype.float32) -> Elementwise:
    node = Elementwise(op=op, shape=shape, dtype=dtype)
    node.inputs = [n.id for n in inputs]
    return node


def emit(subgraph, ext_in, ext_out):
    """Thin wrapper so tests don't repeat the call signature."""
    return SubgraphMSLCodegen().emit(subgraph, ext_in, ext_out)


# ---------------------------------------------------------------------------
# 1. Unary op emission — exp
# ---------------------------------------------------------------------------


def test_unary_exp_contains_metal_exp():
    inp = _input()
    node = _ew("exp", [inp])

    source, inp_names, out_names = emit([node], [inp], [node])

    assert "metal::exp" in source
    assert inp_names == ["inp0"]
    assert out_names == ["out0"]


def test_unary_exp_input_name_in_source():
    inp = _input()
    node = _ew("exp", [inp])

    source, _, _ = emit([node], [inp], [node])

    # The external input is loaded from inp0[elem]
    assert "inp0[elem]" in source


def test_unary_exp_output_store_in_source():
    inp = _input()
    node = _ew("exp", [inp])

    source, _, _ = emit([node], [inp], [node])

    # The output is stored to out0[elem]
    assert "out0[elem]" in source


# ---------------------------------------------------------------------------
# 2. Binary op emission — add
# ---------------------------------------------------------------------------


def test_binary_add_contains_plus():
    a = _input()
    b = _input()
    node = _ew("add", [a, b])

    source, inp_names, out_names = emit([node], [a, b], [node])

    assert "+" in source
    assert inp_names == ["inp0", "inp1"]
    assert out_names == ["out0"]


def test_binary_add_two_inp_loads():
    a = _input()
    b = _input()
    node = _ew("add", [a, b])

    source, _, _ = emit([node], [a, b], [node])

    assert "inp0[elem]" in source
    assert "inp1[elem]" in source


def test_binary_sub_contains_minus():
    a = _input()
    b = _input()
    node = _ew("sub", [a, b])

    source, _, _ = emit([node], [a, b], [node])

    assert "-" in source


def test_binary_mul_contains_star():
    a = _input()
    b = _input()
    node = _ew("mul", [a, b])

    source, _, _ = emit([node], [a, b], [node])

    assert "*" in source


def test_binary_div_contains_slash():
    a = _input()
    b = _input()
    node = _ew("div", [a, b])

    source, _, _ = emit([node], [a, b], [node])

    assert "/" in source


# ---------------------------------------------------------------------------
# 3. `negative` op — was previously broken (op == "neg" check only)
# ---------------------------------------------------------------------------


def test_negative_op_emits_unary_minus():
    inp = _input()
    node = _ew("negative", [inp])

    source, _, _ = emit([node], [inp], [node])

    # Must contain unary negation, not the string "unhandled"
    assert "-" in source
    assert "unhandled" not in source


def test_negative_op_does_not_raise():
    inp = _input()
    node = _ew("negative", [inp])

    # Should not raise ValueError
    source, _, _ = emit([node], [inp], [node])
    assert source  # non-empty


def test_neg_op_also_works():
    """The legacy alias 'neg' must still emit unary negation."""
    inp = _input()
    node = _ew("neg", [inp])

    source, _, _ = emit([node], [inp], [node])

    assert "-" in source
    assert "unhandled" not in source


# ---------------------------------------------------------------------------
# 4. dtype mapping completeness
# ---------------------------------------------------------------------------

# Map from Dtype member to the expected MSL type name.
# Dtypes whose MSL name we can predict precisely.
_EXPECTED_MSL_TYPE = {
    Dtype.float32: "float",
    Dtype.float16: "half",
    Dtype.bfloat16: "bfloat16",
    Dtype.int32: "int",
    Dtype.int16: "short",
    Dtype.int8: "char",
    Dtype.uint8: "uchar",
    Dtype.uint16: "ushort",
    Dtype.uint32: "uint",
    Dtype.int64: "long",
    Dtype.uint64: "ulong",
    Dtype.float64: "double",
    Dtype.complex64: "float2",
    Dtype.bool_: "bool",
}


@pytest.mark.parametrize("dtype", list(Dtype))
def test_dtype_emit_no_error(dtype):
    """Emitting a simple abs node for every dtype must not raise and must
    return a non-empty MSL string.  If the codegen raises NotImplementedError
    for a dtype (genuinely unsupported), the test is skipped."""
    inp = _input(dtype=dtype)
    node = _ew("abs", [inp], dtype=dtype)

    try:
        source, _, _ = emit([node], [inp], [node])
    except NotImplementedError:
        pytest.skip(f"dtype {dtype} not implemented in msl_codegen")

    assert source, f"Empty source for dtype {dtype}"


@pytest.mark.parametrize("dtype,msl_type", list(_EXPECTED_MSL_TYPE.items()))
def test_dtype_msl_type_name(dtype, msl_type):
    """The variable declaration in the emitted MSL must use the expected MSL
    type keyword for each dtype."""
    inp = _input(dtype=dtype)
    node = _ew("abs", [inp], dtype=dtype)

    source, _, _ = emit([node], [inp], [node])

    assert msl_type in source, (
        f"Expected MSL type {msl_type!r} not found in source for {dtype}:\n{source}"
    )


# ---------------------------------------------------------------------------
# 5. Unknown op raises ValueError
# ---------------------------------------------------------------------------


def test_unknown_op_raises_value_error():
    inp = _input()
    node = _ew("totally_unknown_op_xyz", [inp])

    with pytest.raises(ValueError, match="totally_unknown_op_xyz"):
        emit([node], [inp], [node])


def test_unknown_op_does_not_silently_emit():
    """Ensure the old '// unhandled' pattern is gone — must raise instead."""
    inp = _input()
    node = _ew("definitely_not_a_real_op", [inp])

    with pytest.raises(ValueError):
        emit([node], [inp], [node])


# ---------------------------------------------------------------------------
# 6. Multi-node chain: exp → tanh → sigmoid
# ---------------------------------------------------------------------------


def test_chain_exp_tanh_sigmoid_all_present():
    inp = _input()
    exp_node = _ew("exp", [inp])
    tanh_node = _ew("tanh", [exp_node])
    sigmoid_node = _ew("sigmoid", [tanh_node])

    source, _, _ = emit(
        [exp_node, tanh_node, sigmoid_node],
        [inp],
        [sigmoid_node],
    )

    assert "metal::exp" in source
    assert "metal::tanh" in source
    # sigmoid is emitted as: 1.0f / (1.0f + metal::exp(-...))
    assert "metal::exp" in source  # used by both exp and sigmoid
    assert "1.0f" in source  # sigmoid formula


def test_chain_order_exp_before_tanh_before_sigmoid():
    """Verify that the operations appear in the correct order in the source."""
    inp = _input()
    exp_node = _ew("exp", [inp])
    tanh_node = _ew("tanh", [exp_node])
    sigmoid_node = _ew("sigmoid", [tanh_node])

    source, _, _ = emit(
        [exp_node, tanh_node, sigmoid_node],
        [inp],
        [sigmoid_node],
    )

    pos_exp = source.index("metal::exp")
    pos_tanh = source.index("metal::tanh")
    # sigmoid also uses metal::exp — find the second occurrence for sigmoid
    pos_sigmoid_exp = source.index("metal::exp", pos_tanh)

    assert pos_exp < pos_tanh < pos_sigmoid_exp


def test_chain_intermediate_variables_wired():
    """Each node in the chain must consume the variable produced by its
    predecessor — i.e., v2 feeds v3, v3 feeds v4 (variable names are
    deterministic: v1=input load, v2=exp, v3=tanh, v4=sigmoid)."""
    inp = _input()
    exp_node = _ew("exp", [inp])
    tanh_node = _ew("tanh", [exp_node])
    sigmoid_node = _ew("sigmoid", [tanh_node])

    source, _, _ = emit(
        [exp_node, tanh_node, sigmoid_node],
        [inp],
        [sigmoid_node],
    )

    # v1 is the loaded input, v2 = exp(v1), v3 = tanh(v2), v4 = sigmoid(v3)
    assert "metal::exp(v1)" in source
    assert "metal::tanh(v2)" in source
    assert "v3" in source  # sigmoid result references v3


def test_chain_single_input_one_output():
    inp = _input()
    exp_node = _ew("exp", [inp])
    tanh_node = _ew("tanh", [exp_node])
    sigmoid_node = _ew("sigmoid", [tanh_node])

    _, inp_names, out_names = emit(
        [exp_node, tanh_node, sigmoid_node],
        [inp],
        [sigmoid_node],
    )

    assert inp_names == ["inp0"]
    assert out_names == ["out0"]


# ---------------------------------------------------------------------------
# Bonus: return type structure
# ---------------------------------------------------------------------------


def test_emit_returns_three_tuple():
    inp = _input()
    node = _ew("relu", [inp])

    result = emit([node], [inp], [node])

    assert isinstance(result, tuple)
    assert len(result) == 3
    source, inp_names, out_names = result
    assert isinstance(source, str)
    assert isinstance(inp_names, list)
    assert isinstance(out_names, list)


def test_emit_source_is_indented():
    """emit() must return an indented source (4-space indent per line)."""
    inp = _input()
    node = _ew("exp", [inp])

    source, _, _ = emit([node], [inp], [node])

    non_empty_lines = [line for line in source.splitlines() if line.strip()]
    assert all(line.startswith("    ") for line in non_empty_lines), (
        "Expected all non-empty source lines to be indented with 4 spaces"
    )
