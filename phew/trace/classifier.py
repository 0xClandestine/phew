"""Bottleneck classifier: map profiling data to bottleneck class.

Bottleneck classes determine which rule sets are activated in Phase-1 search.

| ALU util | BW   | Occupancy | Small kernels | Class             | Search bias              |
|----------|------|-----------|---------------|-------------------|--------------------------|
| <40%     | high | —         | —             | memory_bound      | fusion, fp16/bf16, fast.* |
| >70%     | —    | —         | —             | compute_bound     | quantize, tile, TensorOps |
| —        | —    | <50%      | —             | occupancy_limited | threadgroup, reg pressure |
| —        | —    | —         | yes           | launch_overhead   | mx.compile, async_eval    |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BottleneckClass(Enum):
    memory_bound = "memory_bound"
    compute_bound = "compute_bound"
    occupancy_limited = "occupancy_limited"
    launch_overhead = "launch_overhead"
    unknown = "unknown"


@dataclass
class KernelStat:
    """Per-kernel profiling statistics parsed from a Metal trace."""

    name: str
    duration_ms: float
    alu_utilization: float = 0.0  # fraction [0, 1]
    memory_bandwidth_gb: float = 0.0
    occupancy: float = 1.0  # fraction [0, 1]
    is_small: bool = False  # heuristic: duration < 0.1ms


@dataclass
class ProfileData:
    """Aggregated data from a Metal trace."""

    kernels: list[KernelStat] = field(default_factory=list)
    total_ms: float = 0.0
    device_info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_kernels(self) -> int:
        return len(self.kernels)

    @property
    def small_kernel_fraction(self) -> float:
        if not self.kernels:
            return 0.0
        return sum(1 for k in self.kernels if k.is_small) / len(self.kernels)

    @property
    def mean_alu_util(self) -> float:
        if not self.kernels:
            return 0.0
        return sum(k.alu_utilization for k in self.kernels) / len(self.kernels)

    @property
    def mean_occupancy(self) -> float:
        if not self.kernels:
            return 1.0
        return sum(k.occupancy for k in self.kernels) / len(self.kernels)

    @property
    def hottest_kernel(self) -> KernelStat | None:
        if not self.kernels:
            return None
        return max(self.kernels, key=lambda k: k.duration_ms)

    @property
    def hottest_kernel_fraction(self) -> float:
        if not self.kernels or self.total_ms <= 0:
            return 0.0
        k = self.hottest_kernel
        return k.duration_ms / self.total_ms if k else 0.0


class BottleneckClassifier:
    """Classify a workload based on ProfileData.

    Thresholds from the spec:
      ALU util < 40% → memory_bound
      ALU util > 70% → compute_bound
      occupancy < 50% → occupancy_limited
      small_kernel_fraction > 50% → launch_overhead
    """

    ALU_LOW = 0.40
    ALU_HIGH = 0.70
    OCC_LOW = 0.50
    SMALL_FRAC = 0.50
    # Phase-2 trigger: hottest kernel > 30% of total time
    PHASE2_TRIGGER = 0.30

    def classify(self, data: ProfileData) -> BottleneckClass:
        if data.small_kernel_fraction > self.SMALL_FRAC:
            return BottleneckClass.launch_overhead
        if data.mean_occupancy < self.OCC_LOW:
            return BottleneckClass.occupancy_limited
        if data.mean_alu_util > self.ALU_HIGH:
            return BottleneckClass.compute_bound
        if data.mean_alu_util < self.ALU_LOW:
            return BottleneckClass.memory_bound
        return BottleneckClass.unknown

    def needs_phase2(self, data: ProfileData) -> bool:
        """Return True if the hottest kernel warrants Phase-2 kernel search."""
        return data.hottest_kernel_fraction > self.PHASE2_TRIGGER

    @staticmethod
    def from_device_info() -> "ProfileData":
        """Collect device_info from the live Metal device."""
        import mlx.core as mx

        info = mx.device_info() if mx.metal.is_available() else {}
        return ProfileData(device_info=info)
