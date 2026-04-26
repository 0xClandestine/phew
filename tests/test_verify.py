"""Unit tests for the equivalence checker (no MLX needed for tolerance tests)."""

from phew.verify import TOLERANCES, SubstitutionClass


def test_fp32_not_opt_in():
    t = TOLERANCES[SubstitutionClass.fp32_to_fp32]
    assert not t.opt_in


def test_precision_classes_opt_in():
    for cls in [
        SubstitutionClass.fp32_to_fp16,
        SubstitutionClass.fp32_to_bf16,
        SubstitutionClass.quantized_4bit,
    ]:
        assert TOLERANCES[cls].opt_in


def test_fp16_tolerances_looser_than_fp32():
    fp32 = TOLERANCES[SubstitutionClass.fp32_to_fp32]
    fp16 = TOLERANCES[SubstitutionClass.fp32_to_fp16]
    assert fp16.atol > fp32.atol
    assert fp16.rtol > fp32.rtol


def test_checker_rejects_disabled_class():
    from phew.verify import EquivalenceChecker

    checker = EquivalenceChecker(
        subst_class=SubstitutionClass.fp32_to_fp16,
        enabled_classes={SubstitutionClass.fp32_to_fp32},  # fp16 not enabled
    )

    def dummy_input_factory(size, seed):
        return [], {}

    result = checker.check(lambda: None, lambda: None, dummy_input_factory)
    assert not result.passed
    assert "opt-in" in result.failures[0].lower()


def test_trace_classifier_memory_bound():
    from phew.trace.classifier import BottleneckClass, BottleneckClassifier, KernelStat, ProfileData

    data = ProfileData(
        kernels=[
            KernelStat("k1", 10.0, alu_utilization=0.2, occupancy=0.9),
            KernelStat("k2", 5.0, alu_utilization=0.3, occupancy=0.9),
        ],
        total_ms=15.0,
    )
    cls = BottleneckClassifier().classify(data)
    assert cls == BottleneckClass.memory_bound


def test_trace_classifier_compute_bound():
    from phew.trace.classifier import BottleneckClass, BottleneckClassifier, KernelStat, ProfileData

    data = ProfileData(
        kernels=[KernelStat("k1", 10.0, alu_utilization=0.85, occupancy=0.9)],
        total_ms=10.0,
    )
    cls = BottleneckClassifier().classify(data)
    assert cls == BottleneckClass.compute_bound


def test_trace_classifier_launch_overhead():
    from phew.trace.classifier import BottleneckClass, BottleneckClassifier, KernelStat, ProfileData

    kernels = [KernelStat(f"k{i}", 0.05, alu_utilization=0.5, is_small=True) for i in range(20)]
    data = ProfileData(kernels=kernels, total_ms=1.0)
    cls = BottleneckClassifier().classify(data)
    assert cls == BottleneckClass.launch_overhead
