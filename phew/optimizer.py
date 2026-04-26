"""Top-level Optimizer: orchestrates the full PHEW pipeline.

Pipeline
--------
1. Profile (capture baseline Metal trace + classify bottleneck)
2. Phase 1 — graph-level equality saturation
   a. Graph passes (compile boundary, primitive subst, TensorOps)
   b. egglog saturation
   c. Extraction
3. Verify equivalence
4. Benchmark (compare baseline vs candidate on ≥3 problem sizes)
5. Phase 2 (if hottest kernel > 30% GPU time and not fast.*-replaced)
6. Report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .ir import Graph
from .verify import SubstitutionClass


@dataclass
class OptimizationResult:
    baseline_ms: float
    optimized_ms: float
    speedup: float
    is_significant: bool  # speedup > 3% noise band
    converged: bool
    applied_rules: list[str]
    verification_passed: bool
    verification_details: list[str]
    search_trace: list[str]
    output_source: str  # emitted Python code
    hardware_info: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class Optimizer:
    """Full PHEW optimization pipeline.

    Parameters
    ----------
    fn:
        The Python function to optimize (must use MLX ops).
    input_factory:
        Callable(size_label, seed) → (args, kwargs).
        Used for benchmarking and verification.
        size_label: "small" | "typical" | "large"
    enabled_subst_classes:
        Precision-reducing substitution classes the user has opted in to.
    max_eqsat_iters:
        E-graph saturation iteration cap.
    extraction_strategy:
        "greedy" or "ilp".
    fn_name:
        Name for the emitted function.
    """

    def __init__(
        self,
        fn: Callable,
        input_factory: Callable[[str, int], tuple[list, dict]],
        enabled_subst_classes: set[SubstitutionClass] | None = None,
        max_eqsat_iters: int = 30,
        extraction_strategy: str = "greedy",
        fn_name: str = "optimized",
        enable_fusion: bool = False,
    ) -> None:
        self.fn = fn
        self.input_factory = input_factory
        self.enabled_subst_classes = enabled_subst_classes or {SubstitutionClass.fp32_to_fp32}
        self.max_eqsat_iters = max_eqsat_iters
        self.extraction_strategy = extraction_strategy
        self.fn_name = fn_name
        self.enable_fusion = enable_fusion

    def run(
        self,
        graph: Graph | None = None,
        trace_path: str | None = None,
    ) -> OptimizationResult:
        """Run the full optimization pipeline."""
        import mlx.core as mx

        from .bench import benchmark, compare
        from .egraph import EGraphSaturator, Extractor
        from .emit import MLXCodegen
        from .rules import run_all_passes
        from .trace import BottleneckClassifier, ProfileData
        from .verify import EquivalenceChecker

        search_trace: list[str] = []
        warnings: list[str] = []

        # ----------------------------------------------------------------
        # Hardware info
        # ----------------------------------------------------------------
        hw = {}
        if mx.metal.is_available():
            hw = dict(mx.device_info())

        # ----------------------------------------------------------------
        # Step 1 — Baseline benchmark
        # ----------------------------------------------------------------
        search_trace.append("Step 1: Baseline benchmark")
        args_typical, kwargs_typical = self.input_factory("typical", 0)
        baseline_result = benchmark(self.fn, *args_typical, **kwargs_typical)
        search_trace.append(
            f"  baseline: {baseline_result.mean_ms:.3f} ± {baseline_result.std_ms:.3f} ms"
        )
        if not baseline_result.converged:
            warnings.append("Baseline benchmark did not converge (σ/μ > 5%)")

        # ----------------------------------------------------------------
        # Step 2 — Profiling / bottleneck classification
        # ----------------------------------------------------------------
        search_trace.append("Step 2: Bottleneck classification")
        profile_data = ProfileData(device_info=hw)
        bottleneck = None

        if trace_path:
            try:
                from .trace.classifier import BottleneckClassifier

                # In a real run, trace_path points to a parsed trace;
                # for now we classify based on hw info alone.
                bottleneck = BottleneckClassifier().classify(profile_data)
                search_trace.append(f"  bottleneck: {bottleneck.value}")
            except Exception as exc:
                warnings.append(f"Bottleneck classification failed: {exc}")
        else:
            search_trace.append("  no trace — using full rule set")

        # ----------------------------------------------------------------
        # Step 3 — Phase-1 graph passes
        # ----------------------------------------------------------------
        search_trace.append("Step 3: Graph-level passes")
        if graph is None:
            graph = self._graph_from_fn(args_typical, kwargs_typical)

        graph, applied = run_all_passes(
            graph,
            enable_compile=True,
            enable_primitive_subst=True,
            enable_tensorops=True,
            enable_fusion=self.enable_fusion,
        )
        search_trace.append(f"  applied: {applied or 'none'}")

        # ----------------------------------------------------------------
        # Step 4 — E-graph saturation
        # ----------------------------------------------------------------
        search_trace.append("Step 4: E-graph saturation")
        try:
            saturator = EGraphSaturator(
                max_iterations=self.max_eqsat_iters,
                bottleneck=bottleneck,
                enabled_subst_classes=self.enabled_subst_classes,
            )
            graph, sat_stats, egraph, root_expr, node_map = saturator.saturate(graph)
            search_trace.append(
                f"  iterations: {sat_stats.iterations}, nodes_before: {sat_stats.nodes_before}"
            )

            extractor = Extractor(strategy=self.extraction_strategy)
            extraction = extractor.extract(egraph, root_expr, node_map, graph)
            graph = extraction.graph
            search_trace.append(f"  extraction cost: {extraction.cost:.2e} ({extraction.strategy})")
        except ImportError:
            warnings.append("egglog not installed — skipping saturation")
            search_trace.append("  SKIPPED (egglog not installed)")

        # ----------------------------------------------------------------
        # Step 5 — Emit optimized code
        # ----------------------------------------------------------------
        codegen = MLXCodegen()
        output_source = codegen.emit(graph, fn_name=self.fn_name)
        search_trace.append("Step 5: Code emitted")

        # ----------------------------------------------------------------
        # Step 6 — Build optimized callable and benchmark
        # ----------------------------------------------------------------
        search_trace.append("Step 6: Benchmark candidate")
        try:
            optimized_fn = self._build_fn_from_source(output_source, self.fn_name)
        except Exception as exc:
            warnings.append(f"Failed to build optimized fn: {exc}")
            optimized_fn = self.fn

        opt_result = benchmark(optimized_fn, *args_typical, **kwargs_typical)
        search_trace.append(f"  optimized: {opt_result.mean_ms:.3f} ± {opt_result.std_ms:.3f} ms")

        # Multi-size validation (one-problem-size winners are not winners)
        all_speedups: list[float] = []
        for size_label in ["small", "typical", "large"]:
            try:
                args_s, kwargs_s = self.input_factory(size_label, 0)
                b = benchmark(self.fn, *args_s, **kwargs_s)
                o = benchmark(optimized_fn, *args_s, **kwargs_s)
                sp = b.mean_ms / o.mean_ms if o.mean_ms > 0 else 1.0
                all_speedups.append(sp)
                search_trace.append(f"  {size_label}: {sp:.3f}×")
            except Exception:
                pass

        # ----------------------------------------------------------------
        # Step 7 — Verify equivalence
        # ----------------------------------------------------------------
        search_trace.append("Step 7: Equivalence verification")
        checker = EquivalenceChecker(
            enabled_classes=self.enabled_subst_classes,
        )
        verify_result = checker.check(self.fn, optimized_fn, self.input_factory)
        search_trace.append(f"  {verify_result}")
        if not verify_result.passed:
            warnings.append("Verification FAILED — candidate dropped")
            optimized_fn = self.fn
            output_source = "# Verification failed — no optimization applied\n"

        # ----------------------------------------------------------------
        # Summarize
        # ----------------------------------------------------------------
        cmp = compare(baseline_result, opt_result)
        speedup = cmp["speedup"]
        # Require all sizes to beat by >3%
        is_significant = verify_result.passed and all(s > 1.03 for s in all_speedups)

        if speedup < 1.05 and verify_result.passed:
            warnings.append(f"Final speedup {speedup:.3f}× < 1.05× — input may be near-optimal.")

        return OptimizationResult(
            baseline_ms=baseline_result.mean_ms,
            optimized_ms=opt_result.mean_ms,
            speedup=speedup,
            is_significant=is_significant,
            converged=baseline_result.converged and opt_result.converged,
            applied_rules=applied,
            verification_passed=verify_result.passed,
            verification_details=verify_result.failures,
            search_trace=search_trace,
            output_source=output_source,
            hardware_info=hw,
            warnings=warnings,
        )

    def _graph_from_fn(self, args, kwargs) -> Graph:
        """Build a phew Graph by tracing the function with MLX arrays.

        This is a best-effort structural trace; the IR importer lives in
        phew/ir/importer.py and handles the common MLX op patterns.
        """
        from .ir.importer import trace_to_graph

        return trace_to_graph(self.fn, args, kwargs)

    def _build_fn_from_source(self, source: str, fn_name: str) -> Callable:
        """Compile emitted Python source and return the function."""
        ns: dict = {}
        exec(compile(source, "<phew_emitted>", "exec"), ns)
        return ns[fn_name]
