"""Unit tests for the benchmark harness (no MLX needed for basic tests)."""

from phew.bench.convergence import check_convergence
from phew.bench.harness import BenchResult, compare


def _make_result(mean, std, converged=True):
    return BenchResult(
        mean_ms=mean,
        std_ms=std,
        cv=std / mean if mean else 0,
        n_bench=20,
        converged=converged,
        samples_ms=[mean],
    )


def test_compare_speedup():
    b = _make_result(10.0, 0.1)
    o = _make_result(5.0, 0.1)
    cmp = compare(b, o)
    assert abs(cmp["speedup"] - 2.0) < 0.01
    assert cmp["is_significant"]


def test_compare_not_significant():
    b = _make_result(10.0, 0.1)
    o = _make_result(9.9, 0.1)  # < 3% improvement
    cmp = compare(b, o)
    assert not cmp["is_significant"]


def test_convergence_beats():
    assert check_convergence(10.0, 9.0, problem_sizes=1, problems_beaten=1)


def test_convergence_no_beat():
    assert not check_convergence(10.0, 9.9, problem_sizes=1, problems_beaten=1)


def test_convergence_must_beat_all_sizes():
    # Only beaten 1 of 3 problem sizes → not a win
    assert not check_convergence(10.0, 5.0, problem_sizes=3, problems_beaten=1)
