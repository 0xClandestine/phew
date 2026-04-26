"""Benchmark harness.

Rules:
  - Never .item() in timed path (forces sync)
  - mx.synchronize() brackets timed loop
  - Increase n_bench until σ/μ < 5%, surface if it won't converge
  - Speedup inside ±3% noise band is not a speedup
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class BenchResult:
    mean_ms: float
    std_ms: float
    cv: float  # coefficient of variation = std/mean
    n_bench: int
    converged: bool  # σ/μ < 5%
    samples_ms: list[float]


def benchmark(
    fn: Callable[..., Any],
    *args: Any,
    n_warmup: int = 5,
    n_bench: int = 20,
    max_n_bench: int = 100,
    cv_target: float = 0.05,
    **kwargs: Any,
) -> BenchResult:
    """Measure fn(*args, **kwargs) mean latency in milliseconds.

    Automatically increases n_bench up to max_n_bench until σ/μ < cv_target.
    """
    import mlx.core as mx

    # Warmup
    for _ in range(n_warmup):
        mx.eval(fn(*args, **kwargs))
    mx.synchronize()

    samples: list[float] = []
    converged = False

    while True:
        t0 = time.perf_counter()
        for _ in range(n_bench):
            mx.eval(fn(*args, **kwargs))
        mx.synchronize()
        elapsed = time.perf_counter() - t0
        samples.append(elapsed / n_bench * 1000.0)

        mean = sum(samples) / len(samples)
        if len(samples) > 1:
            variance = sum((s - mean) ** 2 for s in samples) / (len(samples) - 1)
            std = math.sqrt(variance)
            cv = std / mean if mean > 0 else 0.0
            if cv < cv_target or n_bench * len(samples) >= max_n_bench:
                converged = cv < cv_target
                # Expand per-run samples to per-iteration estimates
                per_iter = []
                for s in samples:
                    per_iter.append(s)
                return BenchResult(
                    mean_ms=mean,
                    std_ms=std,
                    cv=cv,
                    n_bench=n_bench * len(samples),
                    converged=converged,
                    samples_ms=samples,
                )
        else:
            std = 0.0
            cv = 0.0

        if n_bench * len(samples) >= max_n_bench:
            return BenchResult(
                mean_ms=mean,
                std_ms=std,
                cv=cv,
                n_bench=n_bench * len(samples),
                converged=False,
                samples_ms=samples,
            )


def compare(baseline: BenchResult, optimized: BenchResult) -> dict[str, float]:
    """Return speedup and significance.

    Speedup is only reported as real if it exceeds the ±3% noise band.
    """
    speedup = baseline.mean_ms / optimized.mean_ms if optimized.mean_ms > 0 else float("inf")
    # Propagated uncertainty: if either CV > 3% the result is in the noise band
    noise_band = 0.03
    is_significant = abs(speedup - 1.0) > noise_band

    return {
        "speedup": speedup,
        "baseline_ms": baseline.mean_ms,
        "baseline_std_ms": baseline.std_ms,
        "optimized_ms": optimized.mean_ms,
        "optimized_std_ms": optimized.std_ms,
        "is_significant": float(is_significant),
        "baseline_converged": float(baseline.converged),
        "optimized_converged": float(optimized.converged),
    }
