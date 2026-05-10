"""Tests for Metal-specific lint rules and the MSL parser."""

from __future__ import annotations

from phew.metal.lint_rules import (
    ALL_METAL_RULES,
    FastExpRule,
    IntegerDivPow2Rule,
    LoopInvariantLoadRule,
    MaxThreadsRule,
    SinCosSplitRule,
    SlowPowRule,
    SlowTrigRule,
    ThreadgroupBankConflictRule,
    UnrolledInnerLoopRule,
)
from phew.metal.parser import parse_kernels
from phew.metal.wrapper import generate_wrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAST_EXP_RULE = FastExpRule()
_LOOP_INV_RULE = LoopInvariantLoadRule()
_MAX_THREADS_RULE = MaxThreadsRule()
_SLOW_TRIG_RULE = SlowTrigRule()
_SINCOS_SPLIT_RULE = SinCosSplitRule()
_SLOW_POW_RULE = SlowPowRule()
_UNROLL_RULE = UnrolledInnerLoopRule()
_INT_DIV_RULE = IntegerDivPow2Rule()
_TG_BANK_RULE = ThreadgroupBankConflictRule()


def _parse_one(src: str):
    kernels = parse_kernels(src)
    assert kernels, "expected at least one kernel"
    return kernels[0]


def _issues(rule, src: str):
    sig = _parse_one(src)
    return rule.check(sig, "<test>")


# ---------------------------------------------------------------------------
# Parser: inline comment stripping
# ---------------------------------------------------------------------------

_KERNEL_WITH_INLINE_COMMENTS = """\
kernel void my_kernel(
    device const half* x  [[buffer(0)]],  // input tensor [batch, dim]
    device half*       y  [[buffer(1)]],  // output tensor [batch, dim]
    constant uint& dim    [[buffer(2)]],  // number of elements
    uint tid [[thread_position_in_grid]]
) {
    y[tid] = x[tid];
}
"""


def test_parser_strips_inline_comments():
    """Inline // comments must not bleed into the next arg's type field."""
    sig = _parse_one(_KERNEL_WITH_INLINE_COMMENTS)
    assert sig.name == "my_kernel"
    assert len(sig.args) == 4  # x, y, dim, tid

    x_arg = next(a for a in sig.args if a.name == "x")
    y_arg = next(a for a in sig.args if a.name == "y")
    dim_arg = next(a for a in sig.args if a.name == "dim")

    # None of the type strings should contain comment text
    assert "input" not in x_arg.type
    assert "output" not in y_arg.type
    assert "number" not in dim_arg.type

    # Address spaces must be correctly detected (comments don't interfere)
    assert x_arg.address_space == "device"
    assert x_arg.is_const is True
    assert y_arg.address_space == "device"
    assert y_arg.is_output is True
    assert dim_arg.address_space == "constant"


def test_parser_comment_with_brackets_does_not_confuse_splitter():
    """A // comment containing [ ] must not unbalance the arg splitter."""
    src = """\
kernel void kern(
    device const half* q [[buffer(0)]],  // shape [n_q, head_dim*2]
    device const half* k [[buffer(1)]],  // shape [n_kv, head_dim]
    device half*       o [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) { o[tid] = q[tid] + k[tid]; }
"""
    sig = _parse_one(src)
    assert len(sig.args) == 4  # q, k, o, tid
    q_arg = next(a for a in sig.args if a.name == "q")
    k_arg = next(a for a in sig.args if a.name == "k")
    assert "shape" not in q_arg.type
    assert "shape" not in k_arg.type
    assert q_arg.address_space == "device"
    assert k_arg.address_space == "device"


# ---------------------------------------------------------------------------
# FastExpRule
# ---------------------------------------------------------------------------

_SIGMOID_KERNEL = """\
kernel void sigmoid_gate(
    device const half* x [[buffer(0)]],
    device half*       y [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    float v = float(x[tid]);
    y[tid] = half(1.0f / (1.0f + exp(-v)));
}
"""

_SILU_FAST_KERNEL = """\
kernel void swiglu(
    device const half* gate [[buffer(0)]],
    device half*       out  [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    float g = float(gate[tid]);
    float s = 1.0f / (1.0f + metal::fast::exp(-g));
    out[tid] = half(g * s);
}
"""

_NO_EXP_KERNEL = """\
kernel void add_inplace(
    device half*       x [[buffer(0)]],
    device const half* y [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    x[tid] = half(float(x[tid]) + float(y[tid]));
}
"""


def test_fast_exp_flags_bare_exp():
    issues = _issues(_FAST_EXP_RULE, _SIGMOID_KERNEL)
    assert len(issues) == 1
    assert "fast::exp" in issues[0].message
    assert issues[0].rule == "fast_exp"
    assert issues[0].kernel == "sigmoid_gate"


def test_fast_exp_passes_when_already_fast():
    issues = _issues(_FAST_EXP_RULE, _SILU_FAST_KERNEL)
    assert issues == []


def test_fast_exp_passes_when_no_exp():
    issues = _issues(_FAST_EXP_RULE, _NO_EXP_KERNEL)
    assert issues == []


def test_fast_exp_ignores_namespaced_exp():
    """metal::precise::exp( should not be flagged by this rule."""
    src = """\
kernel void precise_kernel(
    device const half* x [[buffer(0)]],
    device half* y       [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    y[tid] = half(metal::precise::exp(float(x[tid])));
}
"""
    # metal::precise::exp contains "::exp" — our regex excludes `::exp`
    issues = _issues(_FAST_EXP_RULE, src)
    assert issues == []


# ---------------------------------------------------------------------------
# LoopInvariantLoadRule
# ---------------------------------------------------------------------------

_LOOP_INVARIANT_KERNEL = """\
kernel void gqa_attn_step(
    device const half* v  [[buffer(0)]],
    device half*       out [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    float acc = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        // v is indexed by `k` (outer loop var), not `d` — loop invariant
        acc += float(v[k]) * float(out[d]);
    }
    out[tid] = half(acc);
}
"""

_NO_INVARIANT_KERNEL = """\
kernel void plain_dot(
    device const half* a  [[buffer(0)]],
    device const half* b  [[buffer(1)]],
    device half*       c  [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    float acc = 0.0f;
    for (uint d = 0; d < n; d++) {
        acc += float(a[d]) * float(b[d]);
    }
    c[tid] = half(acc);
}
"""


def test_loop_invariant_load_flagged():
    issues = _issues(_LOOP_INV_RULE, _LOOP_INVARIANT_KERNEL)
    assert len(issues) >= 1
    assert any("v[k]" in i.message for i in issues)
    assert all(i.rule == "loop_invariant_load" for i in issues)


def test_loop_invariant_load_passes_when_index_matches_loop_var():
    """a[d] and b[d] inside loop over d should NOT be flagged."""
    issues = _issues(_LOOP_INV_RULE, _NO_INVARIANT_KERNEL)
    assert issues == []


def test_loop_invariant_skips_non_device_args():
    """threadgroup or constant args are not device loads — skip them."""
    src = """\
kernel void tg_kernel(
    threadgroup float* sh [[threadgroup(0)]],
    device half* out      [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    float acc = 0.0f;
    for (uint d = 0; d < 32u; d++) {
        acc += sh[k];  // threadgroup, not device — rule should ignore
    }
    out[tid] = half(acc);
}
"""
    issues = _issues(_LOOP_INV_RULE, src)
    assert issues == []


# ---------------------------------------------------------------------------
# Wrapper: absolute source path
# ---------------------------------------------------------------------------

_SIMPLE_KERNEL_SRC = """\
[[max_total_threads_per_threadgroup(256)]]
kernel void simple(
    device const half* x [[buffer(0)]],
    device half* y       [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    y[tid] = x[tid];
}
"""


def test_wrapper_uses_absolute_path(tmp_path):
    metal_file = tmp_path / "kernels" / "my_model.metal"
    metal_file.parent.mkdir()
    metal_file.write_text(_SIMPLE_KERNEL_SRC)

    sig = _parse_one(_SIMPLE_KERNEL_SRC)
    wrapper = generate_wrapper(sig, metal_file)

    # Must contain the full absolute path, not just the filename
    assert str(metal_file.resolve()) in wrapper
    assert 'open("my_model.metal")' not in wrapper


def test_wrapper_generated_code_is_valid_python(tmp_path):
    """The generated wrapper must parse as valid Python."""
    import ast

    metal_file = tmp_path / "simple.metal"
    metal_file.write_text(_SIMPLE_KERNEL_SRC)

    sig = _parse_one(_SIMPLE_KERNEL_SRC)
    wrapper = "import mlx.core as mx\n\n" + generate_wrapper(sig, metal_file)

    # Should not raise SyntaxError
    ast.parse(wrapper)


def test_wrapper_with_inline_comment_args_is_valid_python(tmp_path):
    """Wrapper generated from a kernel with // inline arg comments must be valid Python."""
    import ast

    metal_file = tmp_path / "commented.metal"
    metal_file.write_text(_KERNEL_WITH_INLINE_COMMENTS)

    sig = _parse_one(_KERNEL_WITH_INLINE_COMMENTS)
    wrapper = "import mlx.core as mx\n\n" + generate_wrapper(sig, metal_file)

    # This would fail before the parser comment-stripping fix because
    # const_names would contain newlines, breaking the # NOTE: comment line.
    ast.parse(wrapper)


# ---------------------------------------------------------------------------
# SlowTrigRule
# ---------------------------------------------------------------------------

_ROPE_KERNEL = """\
[[max_total_threads_per_threadgroup(256)]]
kernel void rope_partial(
    device half* x            [[buffer(0)]],
    constant uint& rope_dim   [[buffer(1)]],
    constant uint& pos        [[buffer(2)]],
    constant float& theta     [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    float freq = pow(theta, -float(tid * 2) / float(rope_dim));
    float angle = float(pos) * freq;
    float c = cos(angle), s = sin(angle);
    float a = float(x[tid * 2]), b = float(x[tid * 2 + 1]);
    x[tid * 2]     = half(a * c - b * s);
    x[tid * 2 + 1] = half(a * s + b * c);
}
"""

_ROPE_FAST_TRIG_KERNEL = """\
kernel void rope_partial_fast(
    device half* x            [[buffer(0)]],
    constant uint& rope_dim   [[buffer(1)]],
    constant float& theta     [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    float angle = 0.5f;
    float c = metal::fast::cos(angle), s = metal::fast::sin(angle);
    x[tid] = half(float(x[tid]) * c);
}
"""


def test_slow_trig_flags_bare_sin_cos():
    issues = _issues(_SLOW_TRIG_RULE, _ROPE_KERNEL)
    assert len(issues) == 1
    assert "cos" in issues[0].message or "sin" in issues[0].message
    assert "fast::" in issues[0].message
    assert issues[0].rule == "slow_trig"


def test_slow_trig_passes_when_already_fast():
    issues = _issues(_SLOW_TRIG_RULE, _ROPE_FAST_TRIG_KERNEL)
    assert issues == []


def test_slow_trig_does_not_match_asin_acos():
    """asin/acos have no fast:: variant and must not be flagged."""
    src = """\
kernel void inv_trig(
    device const half* x [[buffer(0)]],
    device half* y       [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    y[tid] = half(asin(float(x[tid])));
}
"""
    issues = _issues(_SLOW_TRIG_RULE, src)
    assert issues == []


# ---------------------------------------------------------------------------
# SinCosSplitRule
# ---------------------------------------------------------------------------


def test_sincos_split_flagged():
    issues = _issues(_SINCOS_SPLIT_RULE, _ROPE_KERNEL)
    assert len(issues) == 1
    assert "sincos" in issues[0].message
    assert "angle" in issues[0].message
    assert issues[0].rule == "sincos_split"


def test_sincos_split_passes_when_different_args():
    src = """\
kernel void no_split(
    device half* x [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    float c = cos(1.0f);
    float s = sin(2.0f);
    x[tid] = half(c + s);
}
"""
    # No simple-identifier args shared between sin and cos
    issues = _issues(_SINCOS_SPLIT_RULE, src)
    assert issues == []


def test_sincos_split_also_fires_with_fast_variants():
    """Even when fast:: is used, if both sin and cos share an arg, sincos is better."""
    src = """\
kernel void still_splittable(
    device half* x [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    float angle = 0.5f;
    float c = metal::fast::cos(angle);
    float s = metal::fast::sin(angle);
    x[tid] = half(c + s);
}
"""
    issues = _issues(_SINCOS_SPLIT_RULE, src)
    assert len(issues) == 1
    assert "angle" in issues[0].message


# ---------------------------------------------------------------------------
# SlowPowRule
# ---------------------------------------------------------------------------


def test_slow_pow_flagged():
    issues = _issues(_SLOW_POW_RULE, _ROPE_KERNEL)
    assert len(issues) == 1
    assert "pow()" in issues[0].message
    assert "fast::exp" in issues[0].message
    assert issues[0].rule == "slow_pow"


def test_slow_pow_passes_when_no_pow():
    issues = _issues(_SLOW_POW_RULE, _SIGMOID_KERNEL)
    assert issues == []


# ---------------------------------------------------------------------------
# UnrolledInnerLoopRule
# ---------------------------------------------------------------------------

_SHIFT_BOUND_KERNEL = """\
[[max_total_threads_per_threadgroup(256)]]
kernel void rmsnorm(
    device const half* x  [[buffer(0)]],
    device half* out      [[buffer(1)]],
    constant uint& dim    [[buffer(2)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tg   [[threads_per_threadgroup]],
    threadgroup float* sh [[threadgroup(0)]]
) {
    float ssq = 0.0f;
    for (uint i = tid; i < dim; i += tg) ssq += float(x[i]) * float(x[i]);
    ssq = simd_sum(ssq);
    uint simd_lane = tid & 31u, simd_gid = tid >> 5u;
    if (simd_lane == 0) sh[simd_gid] = ssq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint g = 0; g < (tg >> 5u); g++) total += sh[g];
    out[tid] = half(x[tid] * rsqrt(total / float(dim) + 1e-5f));
}
"""

_ALREADY_UNROLLED_KERNEL = """\
kernel void already_unrolled(
    threadgroup float* sh [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]]
) {
    float total = 0.0f;
    [[unroll]] for (uint g = 0; g < (tg >> 5u); g++) total += sh[g];
    sh[tid] = total;
}
"""


def test_unrolled_inner_loop_flagged_shift_bound():
    issues = _issues(_UNROLL_RULE, _SHIFT_BOUND_KERNEL)
    assert len(issues) == 1
    assert "[[unroll]]" in issues[0].message
    assert issues[0].rule == "unrolled_inner_loop"


def test_unrolled_inner_loop_passes_when_already_unrolled():
    issues = _issues(_UNROLL_RULE, _ALREADY_UNROLLED_KERNEL)
    assert issues == []


def test_unvectorized_loop_suppressed_when_half4_in_loop_body():
    """A strided outer loop that broadcasts a scalar while the inner loop uses
    half4 must NOT be flagged — the scalar load is intentional (broadcast pattern)."""
    src = """\
kernel void deltanet_outer(
    device const half* k  [[buffer(0)]],
    device const half* v  [[buffer(1)]],
    device float* S       [[buffer(2)]],
    device half* out      [[buffer(3)]],
    constant uint& d_k    [[buffer(4)]],
    constant uint& d_v    [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg  [[threads_per_threadgroup]]
) {
    for (uint r = tid; r < d_k; r += tg) {
        float kr = float(k[r]);   // scalar broadcast — outer loop var
        for (uint i = 0; i < d_v; i += 4) {
            float4 vv4 = float4(*(device const half4*)(v + i));  // half4
            S[r * d_v + i] += kr * vv4.x;
        }
    }
}
"""
    from phew.metal.lint_rules import VectorizationRule

    rule = VectorizationRule()
    sig = _parse_one(src)
    assert rule.check(sig, "<test>") == []


def test_unrolled_inner_loop_flagged_literal_bound():
    src = """\
kernel void literal_loop(
    threadgroup float* sh [[threadgroup(0)]],
    device half* out      [[buffer(0)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    float total = 0.0f;
    for (uint g = 0; g < 8; g++) total += sh[g];
    out[tid] = half(total);
}
"""
    issues = _issues(_UNROLL_RULE, src)
    assert len(issues) == 1


# ---------------------------------------------------------------------------
# IntegerDivPow2Rule
# ---------------------------------------------------------------------------

_INT_DIV_KERNEL = """\
kernel void vectorized(
    device const half* x [[buffer(0)]],
    device half* out     [[buffer(1)]],
    constant uint& dim   [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint dim4 = dim / 4;
    uint half_dim = dim / 2;
    uint mod_val = tid % 32;
    device const half4* x4 = (device const half4*)x;
    out[tid] = x4[dim4][0];
}
"""


def test_int_div_pow2_flagged():
    issues = _issues(_INT_DIV_RULE, _INT_DIV_KERNEL)
    rules_fired = {i.message for i in issues}
    assert any("dim / 4" in m and ">> 2" in m for m in rules_fired)
    assert any("dim / 2" in m and ">> 1" in m for m in rules_fired)
    assert any("tid % 32" in m and "& 31" in m for m in rules_fired)
    assert all(i.rule == "integer_div_pow2" for i in issues)


def test_int_div_pow2_ignores_non_power_of_two():
    src = """\
kernel void non_pow2(
    device half* out [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    uint x = n / 3;
    uint y = tid % 7;
    out[tid] = half(x + y);
}
"""
    issues = _issues(_INT_DIV_RULE, src)
    assert issues == []


def test_int_div_pow2_skips_comments():
    src = """\
kernel void commented(
    device half* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    // uint x = n / 4;  -- this should not fire
    out[tid] = half(0.0f);
}
"""
    issues = _issues(_INT_DIV_RULE, src)
    assert issues == []


# ---------------------------------------------------------------------------
# ThreadgroupBankConflictRule
# ---------------------------------------------------------------------------

_BANK_CONFLICT_KERNEL = """\
kernel void strided_tg(
    device const float* x  [[buffer(0)]],
    device float* out      [[buffer(1)]],
    constant uint& stride  [[buffer(2)]],
    uint tid  [[thread_position_in_threadgroup]],
    threadgroup float* sh  [[threadgroup(0)]]
) {
    // stride-4 write to threadgroup memory — 4-way bank conflict
    sh[tid * 4] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[tid] = sh[tid * 4];
}
"""

_NO_BANK_CONFLICT_KERNEL = """\
kernel void stride1_tg(
    device const float* x  [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint tid  [[thread_position_in_threadgroup]],
    threadgroup float* sh  [[threadgroup(0)]]
) {
    sh[tid] = x[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[tid] = sh[tid];
}
"""


def test_threadgroup_bank_conflict_flagged():
    issues = _issues(_TG_BANK_RULE, _BANK_CONFLICT_KERNEL)
    assert len(issues) == 1
    assert "stride 4" in issues[0].message
    assert "sh" in issues[0].message
    assert issues[0].rule == "threadgroup_bank_conflict"


def test_threadgroup_bank_conflict_passes_stride_one():
    issues = _issues(_TG_BANK_RULE, _NO_BANK_CONFLICT_KERNEL)
    assert issues == []


def test_threadgroup_bank_conflict_no_tg_args():
    """Kernels with no threadgroup args should not be checked."""
    issues = _issues(_TG_BANK_RULE, _SIGMOID_KERNEL)
    assert issues == []


# ---------------------------------------------------------------------------
# Registry smoke test
# ---------------------------------------------------------------------------


def test_all_metal_rules_have_ids():
    ids = [r.id for r in ALL_METAL_RULES]
    assert "max_threads" in ids
    assert "half_accumulator" in ids
    assert "missing_simd_reduce" in ids
    assert "unvectorized_loop" in ids
    assert "fast_exp" in ids
    assert "loop_invariant_load" in ids
    assert "slow_trig" in ids
    assert "sincos_split" in ids
    assert "slow_pow" in ids
    assert "unrolled_inner_loop" in ids
    assert "integer_div_pow2" in ids
    assert "threadgroup_bank_conflict" in ids
