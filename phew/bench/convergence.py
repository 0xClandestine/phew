"""Convergence detection for the search loop.

Stop when no candidate beats the current best by >3% across all problem sizes.
"""

from __future__ import annotations

IMPROVEMENT_THRESHOLD = 0.03  # 3% noise band


def check_convergence(
    current_best_ms: float,
    candidate_ms: float,
    problem_sizes: int = 1,
    problems_beaten: int = 0,
) -> bool:
    """Return True if the candidate meaningfully beats the current best.

    A candidate wins only if it is >3% faster across ALL problem sizes tested.
    A one-problem-size winner is not a winner.
    """
    if current_best_ms <= 0:
        return False
    speedup = current_best_ms / candidate_ms
    return speedup > (1.0 + IMPROVEMENT_THRESHOLD) and problems_beaten >= problem_sizes
