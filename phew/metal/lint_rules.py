"""Metal-specific lint rules operating on parsed KernelSig objects."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .parser import KernelSig

# ---------------------------------------------------------------------------
# Shared helpers: brace-matched for-loop body extraction
# ---------------------------------------------------------------------------


def _extract_brace_body(text: str, open_pos: int) -> str | None:
    """Return the text between balanced { } starting at open_pos, or None."""
    if open_pos >= len(text) or text[open_pos] != "{":
        return None
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_pos + 1 : i]
    return None


def _iter_for_bodies(body: str):
    """
    Yield (header, loop_var, loop_body) for each for-loop with a braced body.

    header   — the full ``for (...)`` text
    loop_var — the loop control variable (first identifier after ``for (``)
    loop_body — the text between the outermost ``{ }``
    """
    for start_m in re.finditer(r"\bfor\s*\(", body):
        start = start_m.start()
        # Walk to the matching ) of the for(...) header
        depth = 0
        i = start_m.end() - 1  # position of the opening (
        while i < len(body):
            if body[i] == "(":
                depth += 1
            elif body[i] == ")":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        header = body[start:i]

        # Extract loop variable from the init clause
        var_m = re.search(r"\bfor\s*\(\s*\w+\s+(\w+)\s*=", header)
        if not var_m:
            continue
        loop_var = var_m.group(1)

        # Skip whitespace to the opening {
        j = i
        while j < len(body) and body[j] in " \t\n":
            j += 1
        if j >= len(body) or body[j] != "{":
            continue  # single-statement body — skip

        loop_body = _extract_brace_body(body, j)
        if loop_body is not None:
            yield header, loop_var, loop_body


# Header pattern for a *strided* for loop: the increment uses += <variable>
# (not ++ or += <literal>), meaning each thread steps through different indices.
_RE_STRIDED_HEADER = re.compile(
    r"\bfor\s*\(\s*\w+\s+\w+\s*=\s*\w+\s*;\s*[^;]+;\s*\w+\s*\+=\s*(?!\d)\w+"
)


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
                    f"{sig.name}() missing [[max_total_threads_per_threadgroup(N)]]"
                    "  →  add attribute to hint compiler register allocation"
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
                            f"`half {varname}` in {sig.name}()  →  use float; cast to half on store"
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
                    f"{sig.name}() threadgroup reduction without simd_sum"
                    "  →  add simd_sum(acc) before writing to shared memory"
                ),
            )
        ]


# ---------------------------------------------------------------------------
# Rule: strided scalar loop over half* — vectorize to half4
# ---------------------------------------------------------------------------


class VectorizationRule(MetalRule):
    id = "unvectorized_loop"
    description = (
        "strided loop over half* reads one element at a time"
        "  →  use half4/float4 loads to quadruple memory throughput"
    )

    def check(self, sig: KernelSig, filename: str) -> list[MetalLintIssue]:
        has_half_input = any("half" in a.type and a.address_space == "device" for a in sig.args)
        if not has_half_input:
            return []

        for header, loop_var, loop_body in _iter_for_bodies(sig.body):
            # Only consider strided loops (i += tg, not ++ or += literal)
            if not _RE_STRIDED_HEADER.search(header):
                continue

            # Scalar half/float cast of the loop variable present in the body
            scalar_pat = re.compile(
                rf"(?:float|half)\s*\(\s*\w+\s*\[\s*{re.escape(loop_var)}\s*\]\s*\)"
            )
            if not scalar_pat.search(loop_body):
                continue

            # Suppress when the loop body already contains half4 — that means
            # the outer loop broadcasts a scalar while an inner loop vectorizes.
            # Flagging such cases is a false positive.
            if "half4" in loop_body:
                continue

            return [
                MetalLintIssue(
                    file=filename,
                    line=sig.line,
                    kernel=sig.name,
                    rule=self.id,
                    message=(
                        f"{sig.name}() strided half* loop  →  use half4 loads for 4× bandwidth"
                    ),
                )
            ]

        return []


# ---------------------------------------------------------------------------
# Rule: slow transcendentals — sin/cos/tan without fast::
# ---------------------------------------------------------------------------

_RE_BARE_TRIG = re.compile(r"(?<!:)\b(sin|cos|tan)\s*\(")
_RE_FAST_TRIG = re.compile(r"fast::(sin|cos|tan)\s*\(")


class SlowTrigRule(MetalRule):
    id = "slow_trig"
    description = (
        "uses stdlib sin/cos/tan instead of metal::fast equivalents"
        "  →  fast:: variants are ~4× faster and sufficient for inference"
    )

    def check(self, sig: KernelSig, filename: str) -> list[MetalLintIssue]:
        body = sig.body
        if not _RE_BARE_TRIG.search(body):
            return []
        bare_fns = {m.group(1) for m in _RE_BARE_TRIG.finditer(body)}
        fast_fns = {m.group(1) for m in _RE_FAST_TRIG.finditer(body)}
        slow = bare_fns - fast_fns
        if not slow:
            return []
        names = ", ".join(sorted(slow))
        return [
            MetalLintIssue(
                file=filename,
                line=sig.line,
                kernel=sig.name,
                rule=self.id,
                message=(
                    f"{sig.name}() calls {names}()"
                    f"  →  use metal::fast::{{{names}}}() for ~4× speedup"
                ),
            )
        ]


# ---------------------------------------------------------------------------
# Rule: sin(X) and cos(X) called separately — use sincos()
# ---------------------------------------------------------------------------

# Only capture simple identifiers as arguments (not expressions)
_RE_COS_SIMPLE = re.compile(r"\bcos\s*\(\s*(\w+)\s*\)")
_RE_SIN_SIMPLE = re.compile(r"\bsin\s*\(\s*(\w+)\s*\)")


class SinCosSplitRule(MetalRule):
    id = "sincos_split"
    description = (
        "sin(X) and cos(X) called separately with the same argument"
        "  →  use sincos(X, &s, &c) to compute both in one instruction"
    )

    def check(self, sig: KernelSig, filename: str) -> list[MetalLintIssue]:
        body = sig.body
        cos_args = {m.group(1) for m in _RE_COS_SIMPLE.finditer(body)}
        sin_args = {m.group(1) for m in _RE_SIN_SIMPLE.finditer(body)}
        shared = cos_args & sin_args
        if not shared:
            return []
        arg = next(iter(sorted(shared)))
        return [
            MetalLintIssue(
                file=filename,
                line=sig.line,
                kernel=sig.name,
                rule=self.id,
                message=(
                    f"{sig.name}() calls sin({arg}) and cos({arg}) separately"
                    f"  →  use sincos({arg}, &s, &c) for one instruction"
                ),
            )
        ]


# ---------------------------------------------------------------------------
# Rule: slow pow() — use exp(log(base) * exponent) instead
# ---------------------------------------------------------------------------

_RE_BARE_POW = re.compile(r"(?<!:)\bpow\s*\(")


class SlowPowRule(MetalRule):
    id = "slow_pow"
    description = (
        "uses stdlib pow() instead of fast::exp(fast::log(base) * exponent)"
        "  →  pow() maps to a slow general path; exp+log is 3-5× faster"
    )

    def check(self, sig: KernelSig, filename: str) -> list[MetalLintIssue]:
        if not _RE_BARE_POW.search(sig.body):
            return []
        return [
            MetalLintIssue(
                file=filename,
                line=sig.line,
                kernel=sig.name,
                rule=self.id,
                message=(
                    f"{sig.name}() calls pow()"
                    "  →  replace with metal::fast::exp(metal::fast::log(base) * exp)"
                    " for 3-5× speedup"
                ),
            )
        ]


# ---------------------------------------------------------------------------
# Rule: small fixed-bound loop without [[unroll]]
# ---------------------------------------------------------------------------

# Bounds given by a right-shift: tg >> 5u, threads_per_threadgroup >> 4, etc.
_RE_SHIFT_BOUND = re.compile(r"\bfor\s*\([^;]*;\s*\w+\s*<\s*\(?[^;)]*>>\s*\d+[^;)]*\)?\s*[;)]")
# Literal numeric bound in [1, 16]
_RE_LITERAL_BOUND = re.compile(r"\bfor\s*\([^;]*;\s*\w+\s*<\s*([1-9]|1[0-6])\s*[;)]")
_RE_UNROLL_ATTR = re.compile(r"\[\[unroll\]\]")


class UnrolledInnerLoopRule(MetalRule):
    id = "unrolled_inner_loop"
    description = (
        "small fixed-bound loop without [[unroll]]"
        "  →  add [[unroll]] to eliminate branch overhead (~1 cycle/iter saved)"
    )

    def check(self, sig: KernelSig, filename: str) -> list[MetalLintIssue]:
        body = sig.body
        if _RE_UNROLL_ATTR.search(body):
            return []
        if _RE_SHIFT_BOUND.search(body) or _RE_LITERAL_BOUND.search(body):
            return [
                MetalLintIssue(
                    file=filename,
                    line=sig.line,
                    kernel=sig.name,
                    rule=self.id,
                    message=(
                        f"{sig.name}() has a small fixed-bound loop without [[unroll]]"
                        "  →  prefix with [[unroll]] to eliminate branch overhead"
                    ),
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Rule: integer division/modulo by a power-of-two constant
# ---------------------------------------------------------------------------

# Match: identifier / literal  (not preceded by another slash, i.e. not //)
_RE_INT_DIV = re.compile(r"\b(\w+)\s*/\s*(\d+)[uU]?\b")
_RE_INT_MOD = re.compile(r"\b(\w+)\s*%\s*(\d+)[uU]?\b")


def _exact_log2(n: int) -> int | None:
    """Return log2(n) if n is a positive power of two, else None."""
    if n > 1 and (n & (n - 1)) == 0:
        return n.bit_length() - 1
    return None


class IntegerDivPow2Rule(MetalRule):
    id = "integer_div_pow2"
    description = (
        "integer division or modulo by a power-of-two literal"
        "  →  use >> or & instead (compiler may not always optimize these)"
    )

    def check(self, sig: KernelSig, filename: str) -> list[MetalLintIssue]:
        issues: list[MetalLintIssue] = []
        for line_offset, line in enumerate(sig.body.splitlines()):
            if line.strip().startswith("//"):
                continue
            code = re.sub(r"//.*", "", line)

            for m in _RE_INT_DIV.finditer(code):
                # Exclude //  (double-slash comment remnant)
                if code[: m.start()].rstrip().endswith("/"):
                    continue
                shift = _exact_log2(int(m.group(2)))
                if shift is not None:
                    issues.append(
                        MetalLintIssue(
                            file=filename,
                            line=sig.line + line_offset,
                            kernel=sig.name,
                            rule=self.id,
                            message=(
                                f"{sig.name}(): `{m.group(1)} / {m.group(2)}`"
                                f"  →  use `{m.group(1)} >> {shift}`"
                            ),
                        )
                    )

            for m in _RE_INT_MOD.finditer(code):
                n = int(m.group(2))
                shift = _exact_log2(n)
                if shift is not None:
                    issues.append(
                        MetalLintIssue(
                            file=filename,
                            line=sig.line + line_offset,
                            kernel=sig.name,
                            rule=self.id,
                            message=(
                                f"{sig.name}(): `{m.group(1)} % {m.group(2)}`"
                                f"  →  use `{m.group(1)} & {n - 1}`"
                            ),
                        )
                    )
        return issues


# ---------------------------------------------------------------------------
# Rule: threadgroup memory accessed with stride > 1 (bank conflict)
# ---------------------------------------------------------------------------


class ThreadgroupBankConflictRule(MetalRule):
    id = "threadgroup_bank_conflict"
    description = (
        "threadgroup array accessed with a stride-N multiplier"
        "  →  causes N-way bank conflicts; prefer stride-1 layout or add padding"
    )

    def check(self, sig: KernelSig, filename: str) -> list[MetalLintIssue]:
        tg_names = [a.name for a in sig.threadgroup_args]
        if not tg_names:
            return []

        issues: list[MetalLintIssue] = []
        for name in tg_names:
            pat = re.compile(
                rf"\b{re.escape(name)}\s*\["
                r"\s*(?:(\w+)\s*\*\s*(\d+)|(\d+)\s*\*\s*(\w+))\s*\]"
            )
            for m in pat.finditer(sig.body):
                k = int(m.group(2) if m.group(2) else m.group(3))
                if k > 1:
                    issues.append(
                        MetalLintIssue(
                            file=filename,
                            line=sig.line,
                            kernel=sig.name,
                            rule=self.id,
                            message=(
                                f"{sig.name}() accesses threadgroup `{name}` with"
                                f" stride {k}  →  {k}-way bank conflict;"
                                " pad the array or reorder access"
                            ),
                        )
                    )
                    break
        return issues


# ---------------------------------------------------------------------------
# Rule: exp() without metal::fast::exp
# ---------------------------------------------------------------------------

# Matches bare exp( but not fast::exp( or any other namespaced variant
_RE_BARE_EXP = re.compile(r"(?<!:)\bexp\s*\(")
_RE_FAST_EXP = re.compile(r"fast::exp\s*\(")


class FastExpRule(MetalRule):
    id = "fast_exp"
    description = (
        "uses stdlib exp() instead of metal::fast::exp()"
        "  →  fast::exp is ~4× faster and sufficient for inference"
    )

    def check(self, sig: KernelSig, filename: str) -> list[MetalLintIssue]:
        body = sig.body
        if not _RE_BARE_EXP.search(body):
            return []
        # If the kernel already uses fast::exp it's fine
        if _RE_FAST_EXP.search(body):
            return []
        return [
            MetalLintIssue(
                file=filename,
                line=sig.line,
                kernel=sig.name,
                rule=self.id,
                message=(
                    f"{sig.name}() calls exp()  →  replace with metal::fast::exp() for ~4× speedup"
                ),
            )
        ]


# ---------------------------------------------------------------------------
# Rule: loop-invariant device buffer load
# ---------------------------------------------------------------------------


class LoopInvariantLoadRule(MetalRule):
    id = "loop_invariant_load"
    description = (
        "device buffer indexed by a variable that does not change with the loop counter"
        "  →  hoist the load before the loop to avoid redundant memory traffic"
    )

    def check(self, sig: KernelSig, filename: str) -> list[MetalLintIssue]:
        ptr_args = {
            a.name
            for a in sig.args
            if a.address_space == "device"
            and a.is_const
            and ("half" in a.type or "float" in a.type)
        }
        if not ptr_args:
            return []

        issues: list[MetalLintIssue] = []
        seen: set[tuple[str, str, str]] = set()

        for _header, loop_var, loop_body in _iter_for_bodies(sig.body):
            for ptr_name in ptr_args:
                for access_m in re.finditer(rf"\b{re.escape(ptr_name)}\[(\w+)\]", loop_body):
                    idx = access_m.group(1)
                    if idx and idx.isidentifier() and idx != loop_var:
                        key = (ptr_name, idx, loop_var)
                        if key in seen:
                            continue
                        seen.add(key)
                        issues.append(
                            MetalLintIssue(
                                file=filename,
                                line=sig.line,
                                kernel=sig.name,
                                rule=self.id,
                                message=(
                                    f"{sig.name}() loads {ptr_name}[{idx}] inside"
                                    f" loop over `{loop_var}`  →  hoist before loop"
                                ),
                            )
                        )
        return issues


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_METAL_RULES: list[MetalRule] = [
    MaxThreadsRule(),
    HalfAccumulatorRule(),
    MissingSimdReduceRule(),
    VectorizationRule(),
    FastExpRule(),
    LoopInvariantLoadRule(),
    SlowTrigRule(),
    SinCosSplitRule(),
    SlowPowRule(),
    UnrolledInnerLoopRule(),
    IntegerDivPow2Rule(),
    ThreadgroupBankConflictRule(),
]
