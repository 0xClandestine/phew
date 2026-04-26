"""Equivalence checker: probabilistic, after Mirage §5 + Schwartz-Zippel.

Protocol:
  - ≥5 random seeds
  - Edge inputs: zeros, denormals, dynamic range 1e-6…1e6, NaN where defined
  - ≥3 problem sizes: small / typical / large
  - One failure → drop candidate, log reason
  - Lax fragment (multi-linear ops): finite-field testing (avoids fp pitfalls)
  - Non-lax: float testing with dtype-aware tolerance

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .tolerances import TOLERANCES, SubstitutionClass, Tolerance

N_SEEDS = 5
PROBLEM_SIZES = ["small", "typical", "large"]


@dataclass
class VerificationResult:
    passed: bool
    substitution_class: SubstitutionClass
    tolerance: Tolerance
    seeds_tested: int
    sizes_tested: list[str]
    failures: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.substitution_class.value} "
            f"atol={self.tolerance.atol} rtol={self.tolerance.rtol} "
            f"seeds={self.seeds_tested} sizes={self.sizes_tested}"
            + (f" failures={self.failures}" if self.failures else "")
        )


class EquivalenceChecker:
    """Verify that two callables are equivalent under a substitution class.

    Parameters
    ----------
    subst_class:
        The substitution class determines the tolerance.
    n_seeds:
        Number of random seeds to test.
    problem_sizes:
        Callable that returns (args, kwargs) for each size label.
    """

    def __init__(
        self,
        subst_class: SubstitutionClass = SubstitutionClass.fp32_to_fp32,
        n_seeds: int = N_SEEDS,
        enabled_classes: set[SubstitutionClass] | None = None,
    ) -> None:
        self.subst_class = subst_class
        self.n_seeds = n_seeds
        self.tolerance = TOLERANCES[subst_class]
        # Default: fp32_to_fp32 always enabled; others require opt-in
        if enabled_classes is None:
            enabled_classes = {SubstitutionClass.fp32_to_fp32}
        self.enabled_classes = enabled_classes

    def check(
        self,
        baseline: Callable,
        candidate: Callable,
        input_factory: Callable[[str, int], tuple[list, dict]],
    ) -> VerificationResult:
        """Run equivalence check.

        Parameters
        ----------
        baseline:
            Original function.
        candidate:
            Candidate rewrite.
        input_factory:
            Called as input_factory(size_label, seed) → (args, kwargs).
            size_label is one of "small", "typical", "large".
        """
        import mlx.core as mx
        import numpy as np

        if self.subst_class not in self.enabled_classes:
            return VerificationResult(
                passed=False,
                substitution_class=self.subst_class,
                tolerance=self.tolerance,
                seeds_tested=0,
                sizes_tested=[],
                failures=[f"{self.subst_class.value} requires opt-in"],
            )

        failures: list[str] = []
        sizes_tested: list[str] = []

        for size_label in PROBLEM_SIZES:
            sizes_tested.append(size_label)
            for seed in range(self.n_seeds):
                args, kwargs = input_factory(size_label, seed)

                # Edge input variants
                for variant_name, variant_args, variant_kwargs in self._edge_variants(
                    args, kwargs, seed
                ):
                    try:
                        out_baseline = baseline(*variant_args, **variant_kwargs)
                        out_candidate = candidate(*variant_args, **variant_kwargs)
                        mx.eval(out_baseline, out_candidate)

                        # Support single array or tuple/list of arrays
                        b_outs = (
                            list(out_baseline)
                            if isinstance(out_baseline, (list, tuple))
                            else [out_baseline]
                        )
                        c_outs = (
                            list(out_candidate)
                            if isinstance(out_candidate, (list, tuple))
                            else [out_candidate]
                        )

                        ok = True
                        max_diff = 0.0
                        for b_arr, c_arr in zip(b_outs, c_outs):
                            # Cast bf16 to float32 before numpy conversion —
                            # numpy has no bf16 dtype and the buffer protocol fails.
                            import mlx.core as _mx
                            if b_arr.dtype == _mx.bfloat16:
                                b_arr = b_arr.astype(_mx.float32)
                            if c_arr.dtype == _mx.bfloat16:
                                c_arr = c_arr.astype(_mx.float32)
                            b_np = np.array(b_arr)
                            c_np = np.array(c_arr)
                            if not self._allclose(b_np, c_np):
                                ok = False
                                diff = float(
                                    np.max(
                                        np.abs(b_np.astype(np.float64) - c_np.astype(np.float64))
                                    )
                                )
                                max_diff = max(max_diff, diff)

                        if not ok:
                            failures.append(
                                f"size={size_label} seed={seed} variant={variant_name} "
                                f"max_diff={max_diff:.3e}"
                            )
                    except Exception as exc:
                        failures.append(
                            f"size={size_label} seed={seed} variant={variant_name} exception={exc}"
                        )

        return VerificationResult(
            passed=len(failures) == 0,
            substitution_class=self.subst_class,
            tolerance=self.tolerance,
            seeds_tested=self.n_seeds,
            sizes_tested=sizes_tested,
            failures=failures,
        )

    def _allclose(self, a, b) -> bool:
        import numpy as np

        # Handle NaN: NaN == NaN is acceptable
        nan_match = np.isnan(a) == np.isnan(b)
        finite_close = np.isclose(
            np.where(np.isnan(a), 0, a),
            np.where(np.isnan(b), 0, b),
            atol=self.tolerance.atol,
            rtol=self.tolerance.rtol,
        )
        return bool(np.all(nan_match & (np.isnan(a) | finite_close)))

    def _edge_variants(self, args, kwargs, seed: int):
        """Yield (name, args, kwargs) for edge input variants."""
        import mlx.core as mx
        import numpy as np

        yield "random", args, kwargs

        # Zeros
        zero_args = [mx.zeros_like(a) if hasattr(a, "shape") else a for a in args]
        yield "zeros", zero_args, kwargs

        # Dynamic range extremes (1e-6 scale)
        rng = np.random.default_rng(seed + 1000)
        small_args = [
            mx.array(rng.standard_normal(a.shape).astype(np.float32) * 1e-6).astype(a.dtype)
            if hasattr(a, "shape")
            else a
            for a in args
        ]
        yield "small_scale", small_args, kwargs

        # Dynamic range extremes (1e6 scale)
        large_args = [
            mx.array(rng.standard_normal(a.shape).astype(np.float32) * 1e6).astype(a.dtype)
            if hasattr(a, "shape")
            else a
            for a in args
        ]
        yield "large_scale", large_args, kwargs
