"""Tests for phew/emit/metal_kernel.py (KernelParamSearch / KernelCandidate)."""

from unittest.mock import MagicMock, patch

import pytest

from phew.emit.metal_kernel import (
    THREADGROUP_1D,
    KernelCandidate,
    KernelParamSearch,
)
from phew.ir import Dtype, Graph, Input, MetalKernel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_SOURCE = "uint elem = thread_position_in_grid.x; out[elem] = inp[elem];"


def _make_metal_kernel_node(
    grid=(1024, 1, 1),
    threadgroup=(256, 1, 1),
    source=_MINIMAL_SOURCE,
):
    """Return a MetalKernel node wired into a minimal Graph."""
    g = Graph()
    inp = g.add(Input(shape=(1024,), dtype=Dtype.float32, name="inp"))
    mk = g.add(
        MetalKernel(
            shape=(1024,),
            dtype=Dtype.float32,
            inputs=[inp.id],
            source=source,
            header="",
            template_params=[],
            threadgroup=threadgroup,
            grid=grid,
            input_names=["inp"],
            output_names=["out"],
            output_shapes=[(1024,)],
            output_dtypes=[Dtype.float32],
            input_shapes=[(1024,)],
        )
    )
    g.outputs = [mk.id]
    return mk


# ---------------------------------------------------------------------------
# Test 1: _build_kernel_fn uses node.grid, not a hardcoded 1-D fallback
# ---------------------------------------------------------------------------


class TestBuildKernelFnUsesNodeGrid:
    """_build_kernel_fn must forward node.grid to the kernel call, not derive it."""

    def test_explicit_grid_is_forwarded(self):
        """When node.grid is set, the closure captures that exact grid."""
        expected_grid = (2048, 4, 1)
        mk = _make_metal_kernel_node(grid=expected_grid)

        # Patch mx.fast.metal_kernel so no real MLX JIT happens
        fake_kernel = MagicMock()
        fake_kernel.return_value = [MagicMock()]

        searcher = KernelParamSearch()
        cand = KernelCandidate(
            threadgroup=(256,),
            tile=None,
            vector_width=1,
            loop_unroll=1,
            use_threadgroup_mem=False,
            use_constant_address=False,
            template_params=[],
            source=_MINIMAL_SOURCE,
        )

        with patch("mlx.core.fast.metal_kernel", return_value=fake_kernel):
            fn = searcher._build_kernel_fn(mk, cand)

        # Call the returned function with a dummy input
        dummy_input = MagicMock()
        dummy_input.size = 2048
        fn(dummy_input)

        # Verify the grid passed to the kernel call is node.grid
        call_kwargs = fake_kernel.call_args[1]
        assert call_kwargs["grid"] == expected_grid, (
            f"Expected grid {expected_grid}, got {call_kwargs['grid']}"
        )

    def test_none_grid_falls_back_to_input_size(self):
        """When node.grid is None, the closure falls back to (inputs[0].size, 1, 1)."""
        mk = _make_metal_kernel_node(grid=None)
        # Manually clear grid (default factory may already set it)
        mk.grid = None

        fake_kernel = MagicMock()
        fake_kernel.return_value = [MagicMock()]

        searcher = KernelParamSearch()
        cand = KernelCandidate(
            threadgroup=(128,),
            tile=None,
            vector_width=1,
            loop_unroll=1,
            use_threadgroup_mem=False,
            use_constant_address=False,
            template_params=[],
        )

        with patch("mlx.core.fast.metal_kernel", return_value=fake_kernel):
            fn = searcher._build_kernel_fn(mk, cand)

        dummy_input = MagicMock()
        dummy_input.size = 512
        fn(dummy_input)

        call_kwargs = fake_kernel.call_args[1]
        assert call_kwargs["grid"] == (
            512,
            1,
            1,
        ), f"Expected fallback grid (512, 1, 1), got {call_kwargs['grid']}"

    def test_threadgroup_is_expanded_to_3d(self):
        """1-D threadgroup tuple must be broadcast to (tg, 1, 1) for the call."""
        mk = _make_metal_kernel_node(grid=(256, 1, 1))

        fake_kernel = MagicMock()
        fake_kernel.return_value = [MagicMock()]

        searcher = KernelParamSearch()
        cand = KernelCandidate(
            threadgroup=(64,),
            tile=None,
            vector_width=1,
            loop_unroll=1,
            use_threadgroup_mem=False,
            use_constant_address=False,
            template_params=[],
        )

        with patch("mlx.core.fast.metal_kernel", return_value=fake_kernel):
            fn = searcher._build_kernel_fn(mk, cand)

        dummy_input = MagicMock()
        dummy_input.size = 256
        fn(dummy_input)

        call_kwargs = fake_kernel.call_args[1]
        assert call_kwargs["threadgroup"] == (64, 1, 1)


# ---------------------------------------------------------------------------
# Test 2: _enumerate produces multiple candidates with different threadgroup sizes
# ---------------------------------------------------------------------------


class TestCandidateGeneration:
    """_enumerate must yield multiple candidates covering all THREADGROUP_1D sizes."""

    def test_returns_multiple_candidates(self):
        mk = _make_metal_kernel_node()
        searcher = KernelParamSearch()
        candidates = list(searcher._enumerate(mk))
        assert len(candidates) > 1, "Should generate more than one candidate"

    def test_covers_all_threadgroup_sizes(self):
        mk = _make_metal_kernel_node()
        searcher = KernelParamSearch()
        candidates = list(searcher._enumerate(mk))

        seen_tg_sizes = {cand.threadgroup[0] for cand in candidates}
        for tg in THREADGROUP_1D:
            assert tg in seen_tg_sizes, f"Threadgroup size {tg} not found among candidates"

    def test_candidates_are_kernel_candidate_instances(self):
        mk = _make_metal_kernel_node()
        searcher = KernelParamSearch()
        for cand in searcher._enumerate(mk):
            assert isinstance(cand, KernelCandidate)

    def test_candidates_have_template_params(self):
        mk = _make_metal_kernel_node()
        searcher = KernelParamSearch()
        for cand in searcher._enumerate(mk):
            # Each candidate should carry at least THREADGROUP in template_params
            names = {k for k, _ in cand.template_params}
            assert "THREADGROUP" in names, (
                f"Candidate missing THREADGROUP in template_params: {cand.template_params}"
            )

    def test_candidates_vary_vector_width_and_unroll(self):
        mk = _make_metal_kernel_node()
        searcher = KernelParamSearch()
        candidates = list(searcher._enumerate(mk))

        vws = {cand.vector_width for cand in candidates}
        unrolls = {cand.loop_unroll for cand in candidates}

        assert len(vws) > 1, "Should generate candidates with varying vector widths"
        assert len(unrolls) > 1, "Should generate candidates with varying unroll factors"


# ---------------------------------------------------------------------------
# Test 3: all candidates pruned — search returns empty list (no uncaught IndexError)
# ---------------------------------------------------------------------------


class TestNoCandidatesCrash:
    """If all candidates are pruned, search() must return [] — not crash."""

    def test_all_pruned_returns_empty_list(self):
        mk = _make_metal_kernel_node()
        searcher = KernelParamSearch(baseline_fn=None, max_candidates=50)

        # Patch should_prune to always prune
        def always_prune(baseline_cost, candidate_cost, **kwargs):
            return True, "forced prune"

        # Patch cost model so node_cost doesn't need real MLX arrays
        fake_cost_model = MagicMock()
        fake_cost_model.node_cost.return_value = 1.0

        with (
            patch(
                "phew.emit.metal_kernel.KernelParamSearch._enumerate", return_value=[]
            ) as _mock_enum,
            patch("phew.bench.benchmark") as _mock_bench,
        ):
            # _enumerate returns nothing → surviving is empty → no benchmark calls
            result = searcher.search(
                mk,
                input_factory=lambda kind, seed: ([], {}),
            )

        assert result == [], f"Expected empty list, got {result}"

    def test_all_candidates_fail_benchmark_returns_empty(self):
        """Candidates that raise during benchmark are marked pruned; result is []."""
        mk = _make_metal_kernel_node()
        searcher = KernelParamSearch(baseline_fn=None, max_candidates=50)

        # Make _enumerate yield a single candidate
        single_cand = KernelCandidate(
            threadgroup=(256,),
            tile=None,
            vector_width=1,
            loop_unroll=1,
            use_threadgroup_mem=False,
            use_constant_address=False,
            template_params=[("THREADGROUP", 256), ("VW", 1), ("UNROLL", 1)],
        )

        fake_kernel_factory = MagicMock()
        fake_kernel_factory.return_value = MagicMock(side_effect=RuntimeError("JIT fail"))

        fake_cost_model = MagicMock()
        fake_cost_model.node_cost.return_value = 1.0

        # Patch should_prune to not prune (so the candidate survives pruning stage)
        # but _build_kernel_fn raises so benchmark fails
        with (
            patch(
                "phew.emit.metal_kernel.KernelParamSearch._enumerate",
                return_value=iter([single_cand]),
            ),
            patch("phew.cost.should_prune", return_value=(False, "")),
            patch("phew.cost.CostModel") as MockCostModel,
            patch(
                "phew.emit.metal_kernel.KernelParamSearch._build_kernel_fn",
                side_effect=RuntimeError("JIT fail"),
            ),
            patch("phew.verify.EquivalenceChecker"),
        ):
            MockCostModel.return_value.node_cost.return_value = 1.0
            result = searcher.search(
                mk,
                input_factory=lambda kind, seed: ([], {}),
            )

        # No verified candidates → empty list, no IndexError
        assert result == []

    def test_search_does_not_raise_index_error_on_empty(self):
        """Directly verify that an empty surviving list never triggers IndexError."""
        mk = _make_metal_kernel_node()
        searcher = KernelParamSearch(baseline_fn=None, max_candidates=0)

        with (
            patch("phew.emit.metal_kernel.KernelParamSearch._enumerate", return_value=iter([])),
            patch("phew.cost.CostModel") as MockCostModel,
            patch("phew.verify.EquivalenceChecker"),
        ):
            MockCostModel.return_value.node_cost.return_value = 1.0
            try:
                result = searcher.search(
                    mk,
                    input_factory=lambda kind, seed: ([], {}),
                )
            except IndexError as exc:
                pytest.fail(f"search() raised IndexError on empty candidates: {exc}")

        assert isinstance(result, list)
