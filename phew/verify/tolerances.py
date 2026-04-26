"""Tolerance tables and substitution class definitions.

All precision-reducing substitution classes require explicit user opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SubstitutionClass(Enum):
    fp32_to_fp32 = "fp32_to_fp32"  # structural rewrites only — always on
    fp32_to_fp16 = "fp32_to_fp16"  # opt-in
    fp32_to_bf16 = "fp32_to_bf16"  # opt-in
    quantized_4bit = "quantized_4bit"  # opt-in


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float
    opt_in: bool


TOLERANCES: dict[SubstitutionClass, Tolerance] = {
    SubstitutionClass.fp32_to_fp32: Tolerance(atol=1e-5, rtol=1e-5, opt_in=False),
    SubstitutionClass.fp32_to_fp16: Tolerance(atol=1e-3, rtol=1e-2, opt_in=True),
    SubstitutionClass.fp32_to_bf16: Tolerance(atol=1e-2, rtol=1e-2, opt_in=True),
    SubstitutionClass.quantized_4bit: Tolerance(atol=1e-2, rtol=5e-2, opt_in=True),
}
