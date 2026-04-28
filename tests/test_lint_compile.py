"""Tests for the compile lint rule — compile-ability analysis."""

from __future__ import annotations

import ast

from phew.lint.compile import CompileRule

rule = CompileRule()


def _issues(src: str):
    tree = ast.parse(src)
    return rule.check(tree, "<test>")


# ---------------------------------------------------------------------------
# Basic detection
# ---------------------------------------------------------------------------


def test_simple_mx_function_flagged():
    src = """
def forward(x):
    return mx.softmax(x)
"""
    assert len(_issues(src)) == 1
    assert "forward" in _issues(src)[0].message
    assert "@mx.compile" in _issues(src)[0].message


def test_already_decorated_skipped():
    src = """
@mx.compile
def forward(x):
    return mx.softmax(x)
"""
    assert _issues(src) == []


def test_partial_mx_compile_skipped():
    """@partial(mx.compile, shapeless=True) must not trigger the rule."""
    src = """
@partial(mx.compile, shapeless=True)
def swiglu(gate, x):
    return nn.silu(gate) * x
"""
    assert _issues(src) == []


def test_partial_compile_with_random_state_skipped():
    """@partial(mx.compile, inputs=mx.random.state, ...) must not trigger the rule."""
    src = """
@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def sample(logits):
    return mx.random.categorical(logits)
"""
    assert _issues(src) == []


def test_private_function_skipped():
    src = """
def _forward(x):
    return mx.softmax(x)
"""
    assert _issues(src) == []


def test_class_method_skipped():
    src = """
class Model:
    def forward(self, x):
        return mx.softmax(x)
"""
    assert _issues(src) == []


def test_no_mx_ops_skipped():
    src = """
def add(x, y):
    return x + y
"""
    assert _issues(src) == []


# ---------------------------------------------------------------------------
# Non-compilable arg suppression
# ---------------------------------------------------------------------------


def test_module_annotation_suppressed():
    src = """
def generate_step(tokens, model: nn.Module):
    return model(tokens)
"""
    assert _issues(src) == []


def test_callable_annotation_suppressed():
    src = """
def run(x, sampler: Callable):
    return mx.softmax(sampler(x))
"""
    assert _issues(src) == []


def test_tokenizer_annotation_suppressed():
    src = """
def tokenize(text, tokenizer: Tokenizer):
    return mx.array(tokenizer.encode(text))
"""
    assert _issues(src) == []


def test_model_arg_name_heuristic_suppressed():
    src = """
def generate(tokens, model):
    return model(mx.softmax(tokens))
"""
    assert _issues(src) == []


def test_sampler_arg_name_heuristic_suppressed():
    src = """
def sample(logits, sampler):
    return mx.softmax(sampler(logits))
"""
    assert _issues(src) == []


def test_no_annotation_compilable_arg_flagged():
    """A function with only array-ish args and no annotation should still be flagged."""
    src = """
def forward(x, weights):
    return mx.matmul(x, weights)
"""
    assert len(_issues(src)) == 1


# ---------------------------------------------------------------------------
# Random state suggestion
# ---------------------------------------------------------------------------


def test_mx_random_suggests_partial():
    src = """
def sample(logits):
    return mx.random.categorical(logits)
"""
    issues = _issues(src)
    assert len(issues) == 1
    assert "mx.random.state" in issues[0].message
    assert "partial" in issues[0].message


def test_mx_random_with_noncompilable_arg_suppressed():
    """Even if random is used, non-compilable arg should still suppress the warning."""
    src = """
def sample(logits, sampler: Callable):
    return mx.random.categorical(sampler(logits))
"""
    assert _issues(src) == []
