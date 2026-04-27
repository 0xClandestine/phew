"""Metal-specific lint rules operating on parsed KernelSig objects."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .parser import KernelSig


@dataclass
class MetalLintIssue:
    file: str
    line: int
    kernel: str
    rule: str
    message: str


class MetalRule(ABC):
    id: str
    description: str

    @abstractmethod
    def check(self, sig: KernelSig, filename: str) -> list[MetalLintIssue]: ...


# ---------------------------------------------------------------------------
# Rule: missing [[max_total_threads_per_threadgroup]]
# ---------------------------------------------------------------------------


class MaxThreadsRule(MetalRule):
    id = "max_threads"
    description = (
        "kernel missing [[max_total_threads_per_threadgroup(N)]]"
        "  →  hint lets the compiler optimize register allocation"
    )

    def check(self, sig: KernelSig, filename: str) -> list[MetalLintIssue]:
        if sig.has_max_threads_attr:
            return []
        return [
            MetalLintIssue(
                file=filename,
                line=sig.line,
                kernel=sig.name,
                rule=self.id,
                message=(
                    f"kernel {sig.name}() has no [[max_total_threads_per_threadgroup(N)]]"
                    "  →  add to let the compiler optimise register allocation"
                ),
            )
        ]


# ---------------------------------------------------------------------------
# Rule: half-typed accumulator variable
# ---------------------------------------------------------------------------

# Matches `half varname` or `half varname =` but not `device const half *` etc.
_RE_HALF_ACC = re.compile(
    r"(?<![*&])\bhalf\s+(\w+)\s*(?:=|;|\[)",
)
# Exclude pointer/reference declarations (those are fine — it's the scalar accumulators)
_RE_PTR_OR_REF = re.compile(r"\b(device|constant|threadgroup)\b.*\bhalf\b")


class HalfAccumulatorRule(MetalRule):
    id = "half_accumulator"
    description = (
        "scalar `half` local variable used as accumulator"
        "  →  use `float` to avoid precision loss and slow fp16 ALU"
    )

    def check(self, sig: KernelSig, filename: str) -> list[MetalLintIssue]:
        issues: list[MetalLintIssue] = []
        body_lines = sig.body.splitlines()
        body_start_line = sig.line  # body lines are offset from kernel declaration

        for offset, line in enumerate(body_lines):
            stripped = line.strip()
            # Skip commented lines
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            # Skip address-space declarations (those are params or already-typed vars)
            if _RE_PTR_OR_REF.search(line):
                continue
            for m in _RE_HALF_ACC.finditer(line):
                varname = m.group(1)
                issues.append(
                    MetalLintIssue(
                        file=filename,
                        line=body_start_line + offset,
                        kernel=sig.name,
                        rule=self.id,
                        message=(
                            f"scalar `half {varname}` in {sig.name}()"
                            "  →  use `float` for accumulation; convert to half only on store"
                        ),
                    )
                )
        return issues


# ---------------------------------------------------------------------------
# Rule: threadgroup barrier reduction without simd_sum first pass
# ---------------------------------------------------------------------------

_RE_BARRIER_REDUCTION = re.compile(
    r"for\s*\([^)]*\)\s*\{[^}]*threadgroup_barrier\s*\([^)]*\)[^}]*\}",
    re.DOTALL,
)
_RE_SIMD_SUM = re.compile(r"\bsimd_sum\s*\(")


class MissingSimdReduceRule(MetalRule):
    id = "missing_simd_reduce"
    description = (
        "threadgroup barrier reduction without prior simd_sum"
        "  →  add simd_sum() first pass to cut barrier count by 5×"
    )

    def check(self, sig: KernelSig, filename: str) -> list[MetalLintIssue]:
        body = sig.body

        # Look for the classic pattern:
        #   sh[tid] = acc;
        #   threadgroup_barrier(...)
        #   for (s = tg/2; s > 0; ...) { if (tid < s) sh[tid] += ...; threadgroup_barrier }
        # where simd_sum() is NOT called before writing to sh[tid]

        has_tg_barrier_reduction = bool(
            re.search(
                r"threadgroup_barrier\s*\(\s*mem_flags::mem_threadgroup\s*\)",
                body,
            )
        )
        if not has_tg_barrier_reduction:
            return []

        # If simd_sum is already present, the kernel is already doing it right
        if _RE_SIMD_SUM.search(body):
            return []

        # Check for the store-then-reduce pattern: sh[tid] = X followed by barrier loop
        has_sh_store = bool(re.search(r"\bsh\s*\[\s*\w+\s*\]\s*=", body))
        if not has_sh_store:
            return []

        return [
            MetalLintIssue(
                file=filename,
                line=sig.line,
                kernel=sig.name,
                rule=self.id,
                message=(
                    f"{sig.name}(): threadgroup reduction with no simd_sum() first pass"
                    "  →  acc = simd_sum(acc); then only 1 barrier needed across SIMD groups"
                    "  (see gemv_f16 pattern in same file)"
                ),
            )
        ]


# ---------------------------------------------------------------------------
# Rule: strided scalar loop over half* — vectorize to half4
# ---------------------------------------------------------------------------

_RE_STRIDED_HALF_LOOP = re.compile(
    # for (uint i = tid; i < N; i += tg)
    r"for\s*\(\s*\w+\s+(\w+)\s*=\s*\w+\s*;\s*\w+\s*<\s*\w+\s*;\s*\w+\s*\+=\s*\w+\s*\)"
    r".*?"  # loop body preamble
    r"(?:float\s*\(\s*\w+\s*\[\s*\1\s*\]\s*\)|half\s*\(\s*\w+\s*\[\s*\1\s*\]\s*\))",
    re.DOTALL,
)


class VectorizationRule(MetalRule):
    id = "unvectorized_loop"
    description = (
        "strided loop over half* reads one element at a time"
        "  →  use half4/float4 loads to quadruple memory throughput"
    )

    def check(self, sig: KernelSig, filename: str) -> list[MetalLintIssue]:
        # Check if any input is a half pointer
        has_half_input = any("half" in a.type and a.address_space == "device" for a in sig.args)
        if not has_half_input:
            return []

        if not _RE_STRIDED_HALF_LOOP.search(sig.body):
            return []

        return [
            MetalLintIssue(
                file=filename,
                line=sig.line,
                kernel=sig.name,
                rule=self.id,
                message=(
                    f"{sig.name}(): strided loop reads half* scalarly"
                    "  →  load as half4 (4× bandwidth); ensure dim divisible by 4 or add tail"
                ),
            )
        ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_METAL_RULES: list[MetalRule] = [
    MaxThreadsRule(),
    HalfAccumulatorRule(),
    MissingSimdReduceRule(),
    VectorizationRule(),
]
