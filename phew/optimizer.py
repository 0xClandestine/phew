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
    verify_fusion:
        When True and elementwise fusion is enabled, verify the fused graph
        against the original function before proceeding.  If the check fails
        the pipeline re-runs without fusion and logs a warning.
    """

    def __init__(
        self,
        fn: Callable,
        input_factory: Callable[[str, int], tuple[list, dict]],
        enabled_subst_classes: set[SubstitutionClass] | None = None,
        max_eqsat_iters: int = 30,
        extraction_strategy: str = "greedy",
        fn_name: str = "optimized",
        enable_elementwise_fusion: bool = False,
        enable_phase2_search: bool = False,
        enable_tensorops: bool = False,
        verify_fusion: bool = False,
    ) -> None:
        self.fn = fn
        self.input_factory = input_factory
        self.enabled_subst_classes = enabled_subst_classes or {SubstitutionClass.fp32_to_fp32}
        self.max_eqsat_iters = max_eqsat_iters
        self.extraction_strategy = extraction_strategy
        self.fn_name = fn_name
        self.enable_elementwise_fusion = enable_elementwise_fusion
        self.enable_phase2_search = enable_phase2_search
        self.enable_tensorops = enable_tensorops
        self.verify_fusion = verify_fusion

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
            enable_tensorops=self.enable_tensorops,
            enable_fusion=self.enable_elementwise_fusion,
            enabled_subst_classes=self.enabled_subst_classes,
        )
        search_trace.append(f"  applied: {applied or 'none'}")

        # ----------------------------------------------------------------
        # Step 3b — Verify fusion-generated kernels (optional)
        # When verify_fusion=True and fusion was enabled, emit the fused
        # graph to a callable and check it against the original function.
        # If verification fails, re-run passes without fusion so downstream
        # steps get a safe graph.
        # ----------------------------------------------------------------
        if self.enable_elementwise_fusion and self.verify_fusion:
            from .ir.ops import MetalKernel

            has_fusion_nodes = any(
                isinstance(n, MetalKernel) and n.attrs.get("fusion_generated")
                for n in graph.topo_order()
            )
            if has_fusion_nodes:
                search_trace.append("Step 3b: Verify fusion-generated kernels")
                try:
                    fused_source = MLXCodegen().emit(graph, fn_name="_phew_fused_check")
                    fused_fn = self._build_fn_from_source(fused_source, "_phew_fused_check")
                    fusion_checker = EquivalenceChecker(
                        enabled_classes=self.enabled_subst_classes,
                    )
                    fusion_result = fusion_checker.check(self.fn, fused_fn, self.input_factory)
                    if fusion_result.passed:
                        search_trace.append("  fusion verification: PASS")
                    else:
                        search_trace.append("  fusion verification: FAIL — disabling fusion")
                        warnings.append(
                            "Fusion verification FAILED — re-running passes without fusion. "
                            f"Failures: {fusion_result.failures}"
                        )
                        # Re-trace and re-run without fusion
                        graph = self._graph_from_fn(args_typical, kwargs_typical)
                        graph, applied = run_all_passes(
                            graph,
                            enable_compile=True,
                            enable_primitive_subst=True,
                            enable_tensorops=self.enable_tensorops,
                            enable_fusion=False,
                            enabled_subst_classes=self.enabled_subst_classes,
                        )
                        search_trace.append(f"  re-applied (no fusion): {applied or 'none'}")
                except Exception as exc:
                    warnings.append(f"Fusion verification error — disabling fusion: {exc}")
                    search_trace.append(f"  fusion verification: ERROR ({exc}) — disabling fusion")
                    graph = self._graph_from_fn(args_typical, kwargs_typical)
                    graph, applied = run_all_passes(
                        graph,
                        enable_compile=True,
                        enable_primitive_subst=True,
                        enable_tensorops=self.enable_tensorops,
                        enable_fusion=False,
                        enabled_subst_classes=self.enabled_subst_classes,
                    )
                    search_trace.append(f"  re-applied (no fusion): {applied or 'none'}")

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
            graph, sat_stats, egraph, root_expr, node_map, str_node_map = saturator.saturate(graph)
            search_trace.append(
                f"  iterations: {sat_stats.iterations}, nodes_before: {sat_stats.nodes_before}"
            )

            extractor = Extractor(strategy=self.extraction_strategy)
            extraction = extractor.extract(egraph, root_expr, node_map, graph, str_node_map)
            graph = extraction.graph
            search_trace.append(f"  extraction cost: {extraction.cost:.2e} ({extraction.strategy})")
        except ImportError:
            warnings.append("egglog not installed — skipping saturation")
            search_trace.append("  SKIPPED (egglog not installed)")

        # ----------------------------------------------------------------
        # Step 5a — Phase-2 kernel parameter search (enable_phase2_search flag)
        # Triggered when MetalKernel nodes appear in the graph, meaning the
        # input used mx.fast.metal_kernel and Phase-1 didn't replace it.
        # Fusion-generated kernels are excluded (tagged with fusion_generated=True)
        # because their valid parameter search spaces differ from user/TensorOps kernels.
        # ----------------------------------------------------------------
        if self.enable_phase2_search:
            from .emit import KernelParamSearch
            from .ir import MetalKernel

            metal_nodes = [
                n
                for n in graph.topo_order()
                if isinstance(n, MetalKernel) and not n.attrs.get("fusion_generated")
            ]
            if metal_nodes:
                search_trace.append("Step 5a: Phase-2 kernel parameter search")
                for knode in metal_nodes:
                    searcher = KernelParamSearch(
                        baseline_fn=self.fn,
                        max_candidates=50,
                        bottleneck=getattr(self, "bottleneck_class", bottleneck),
                    )
                    try:
                        candidates = searcher.search(knode, self.input_factory, profile_data)
                        if candidates:
                            best = candidates[0]
                            search_trace.append(
                                f"  kernel: best {best.measured_ms:.3f} ms "
                                f"(tg={best.threadgroup}, vw={best.vector_width})"
                            )
                        else:
                            search_trace.append("  Phase-2: no verified candidates found")
                    except Exception as exc:
                        warnings.append(f"Phase-2 search failed: {exc}")
            else:
                search_trace.append("Step 5a: Phase-2 skipped (no MetalKernel nodes in graph)")

        # ----------------------------------------------------------------
        # Step 5b — Emit optimized code
        # ----------------------------------------------------------------
        codegen = MLXCodegen()
        output_source = codegen.emit(graph, fn_name=self.fn_name)
        search_trace.append("Step 5b: Code emitted")

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
