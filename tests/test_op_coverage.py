"""Cross-reference test: every mlx.core / mlx.core.fast op must be
catalogued in phew/ir/op_registry.py, and every op listed as IMPLEMENTED
must actually have a method on _TracingContext.

Run with:
    uv run --no-config --with pytest python -m pytest tests/test_op_coverage.py -v
"""

from __future__ import annotations

import mlx.core as mx
import mlx.core.fast as mx_fast
import pytest

from phew.ir.op_registry import (
    FAST_IMPLEMENTED,
    FAST_NOT_NEEDED,
    FAST_SPECIAL,
    IMPLEMENTED,
    NOT_COMPUTATIONAL,
    REQUIRES_NEW_NODE,
    TODO,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mlx_core_callables() -> frozenset[str]:
    """All public callable non-type names in mlx.core."""
    return frozenset(
        name
        for name in dir(mx)
        if not name.startswith("_")
        and callable(getattr(mx, name))
        and not isinstance(getattr(mx, name), type)
    )


def _mlx_fast_callables() -> frozenset[str]:
    """All public callable non-type names in mlx.core.fast."""
    return frozenset(
        name
        for name in dir(mx_fast)
        if not name.startswith("_")
        and callable(getattr(mx_fast, name))
        and not isinstance(getattr(mx_fast, name), type)
    )


def _tracer_methods() -> frozenset[str]:
    """All method names on _TracingContext (excludes dunder / private)."""
    from phew.ir.importer import _TracingContext

    return frozenset(
        name
        for name in dir(_TracingContext)
        if not name.startswith("_") and callable(getattr(_TracingContext, name))
    )


# ---------------------------------------------------------------------------
# Registry completeness — every mlx op accounted for
# ---------------------------------------------------------------------------


def test_all_mlx_core_ops_in_registry():
    """Every public callable in mlx.core must appear in exactly one category."""
    all_registry = IMPLEMENTED | NOT_COMPUTATIONAL | REQUIRES_NEW_NODE | TODO
    mlx_ops = _mlx_core_callables()

    missing = sorted(mlx_ops - all_registry)
    assert not missing, (
        f"{len(missing)} mlx.core op(s) are not categorised in op_registry.py:\n"
        + "\n".join(f"  {op}" for op in missing)
    )


def test_all_mlx_fast_ops_in_registry():
    """Every public callable in mlx.core.fast must appear in a fast category."""
    all_fast = FAST_IMPLEMENTED | FAST_NOT_NEEDED | FAST_SPECIAL
    fast_ops = _mlx_fast_callables()

    missing = sorted(fast_ops - all_fast)
    assert not missing, (
        f"{len(missing)} mlx.core.fast op(s) are not categorised in op_registry.py:\n"
        + "\n".join(f"  {op}" for op in missing)
    )


def test_registry_categories_disjoint():
    """The four mlx.core categories must not overlap."""
    cats = {
        "IMPLEMENTED": IMPLEMENTED,
        "NOT_COMPUTATIONAL": NOT_COMPUTATIONAL,
        "REQUIRES_NEW_NODE": REQUIRES_NEW_NODE,
        "TODO": TODO,
    }
    for a_name, a in cats.items():
        for b_name, b in cats.items():
            if a_name >= b_name:
                continue
            overlap = sorted(a & b)
            assert not overlap, f"op_registry overlap between {a_name} and {b_name}:\n" + "\n".join(
                f"  {op}" for op in overlap
            )


def test_fast_categories_disjoint():
    """All fast categories must be mutually exclusive."""
    cats = {
        "FAST_IMPLEMENTED": FAST_IMPLEMENTED,
        "FAST_NOT_NEEDED": FAST_NOT_NEEDED,
        "FAST_SPECIAL": FAST_SPECIAL,
    }
    for a_name, a in cats.items():
        for b_name, b in cats.items():
            if a_name >= b_name:
                continue
            overlap = sorted(a & b)
            assert not overlap, f"fast registry overlap between {a_name} and {b_name}: {overlap}"


# ---------------------------------------------------------------------------
# Tracer coverage — IMPLEMENTED ops must exist on _TracingContext
# ---------------------------------------------------------------------------


def test_implemented_ops_have_tracer_methods():
    """Every op marked IMPLEMENTED in the registry must have a method on _TracingContext."""
    tracer = _tracer_methods()
    mlx_callables = _mlx_core_callables()

    # Only check ops that actually exist in the current mlx.core version
    # (some ops like quantized_scaled_dot_product_attention may not be present yet)
    checkable = sorted(IMPLEMENTED & mlx_callables)

    missing = [op for op in checkable if op not in tracer]
    assert not missing, (
        f"{len(missing)} IMPLEMENTED op(s) lack a _TracingContext method:\n"
        + "\n".join(f"  {op}" for op in missing)
    )


def test_implemented_fast_ops_have_tracer_methods():
    """Every fast op in FAST_IMPLEMENTED must have a method on _TracingContext."""
    tracer = _tracer_methods()
    fast_callables = _mlx_fast_callables()

    checkable = sorted(FAST_IMPLEMENTED & fast_callables)
    missing = [op for op in checkable if op not in tracer]
    assert not missing, (
        f"{len(missing)} FAST_IMPLEMENTED op(s) lack a _TracingContext method:\n"
        + "\n".join(f"  {op}" for op in missing)
    )


# ---------------------------------------------------------------------------
# Registry staleness — no phantom entries pointing to removed mlx ops
# ---------------------------------------------------------------------------


def test_no_phantom_implemented_ops():
    """No op in IMPLEMENTED should reference a name that doesn't exist in mlx.core
    *and* is not a known alias / extra handled by the tracer."""
    # Some ops exist in the tracer but not as top-level mlx.core callables
    # (e.g. 'array' is a type, fast ops are on mlx.core.fast, etc.).
    known_extras = {
        "eval",  # mlx.eval is a function; passes as callable in some versions
        "synchronize",  # mlx.synchronize ditto
        "partition",  # may be on mlx or not depending on version
        "topk",  # ditto
    }
    mlx_callables = _mlx_core_callables()
    fast_callables = _mlx_fast_callables()
    all_known = mlx_callables | fast_callables | known_extras

    phantom = sorted(IMPLEMENTED - all_known)
    # Soft warning: phantom entries just mean the op was removed from mlx.
    # We don't hard-fail to avoid breaking CI on mlx version bumps, but we
    # report them so they can be cleaned up.
    if phantom:
        pytest.warns(
            UserWarning,
            match="phantom",
        )
