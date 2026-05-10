"""Parse [[kernel]] function signatures and bodies from MSL source."""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class KernelArg:
    name: str
    type: str  # e.g. "half", "float", "uint"
    address_space: str  # "device", "constant", "threadgroup", "builtin"
    is_const: bool
    is_output: bool  # device non-const → output buffer
    buffer_idx: int | None  # [[buffer(N)]] index, or None for builtins
    tg_idx: int | None  # [[threadgroup(N)]] index, or None
    raw: str  # original text of this argument


@dataclass
class KernelSig:
    name: str
    line: int  # 1-based line number of `kernel void name(`
    args: list[KernelArg]
    body: str  # source between the outermost { }
    has_max_threads_attr: bool  # [[max_total_threads_per_threadgroup(...)]]

    @property
    def input_args(self) -> list[KernelArg]:
        return [a for a in self.args if a.address_space == "device" and not a.is_output]

    @property
    def output_args(self) -> list[KernelArg]:
        return [a for a in self.args if a.is_output]

    @property
    def constant_args(self) -> list[KernelArg]:
        return [a for a in self.args if a.address_space == "constant"]

    @property
    def builtin_args(self) -> list[KernelArg]:
        return [a for a in self.args if a.address_space == "builtin"]

    @property
    def threadgroup_args(self) -> list[KernelArg]:
        return [a for a in self.args if a.address_space == "threadgroup"]


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches `[[max_total_threads_per_threadgroup(...)]]` anywhere before `kernel void`
_RE_MAX_THREADS = re.compile(r"\[\[max_total_threads_per_threadgroup\s*\(\s*(\d+)\s*\)\]\]")

# Matches `kernel void name(` — the start of a kernel declaration
_RE_KERNEL_START = re.compile(r"\bkernel\s+void\s+(\w+)\s*\(")

# Matches a single kernel argument (split on commas after balancing parens/brackets)
_RE_BUFFER_ATTR = re.compile(r"\[\[buffer\s*\(\s*(\d+)\s*\)\]\]")
_RE_TG_ATTR = re.compile(r"\[\[threadgroup\s*\(\s*(\d+)\s*\)\]\]")

# Built-in thread-position argument names
_BUILTINS = {
    "thread_position_in_grid",
    "thread_position_in_threadgroup",
    "threadgroup_position_in_grid",
    "threads_per_threadgroup",
    "threads_per_grid",
    "thread_index_in_simdgroup",
    "simdgroup_index_in_threadgroup",
    "thread_index_in_quadgroup",
    "quadgroup_index_in_threadgroup",
    "simd_lane_id",
    "simd_group_id",
}

# address-space keywords
_ADDR_SPACES = ("device", "constant", "threadgroup", "thread")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_body(source: str, open_brace_pos: int) -> tuple[str, int]:
    """Return (body_text, end_pos) where body_text is between { } at open_brace_pos."""
    depth = 0
    i = open_brace_pos
    start = None
    while i < len(source):
        ch = source[i]
        if ch == "{":
            depth += 1
            if start is None:
                start = i
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : i], i
        i += 1
    return "", len(source)


def _split_args(args_str: str) -> list[str]:
    """Split a comma-separated argument list respecting < > and () nesting."""
    args: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in args_str:
        if ch in "(<[":
            depth += 1
            current.append(ch)
        elif ch in ")>]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append("".join(current).strip())
    return [a for a in args if a]


def _parse_arg(raw: str) -> KernelArg:
    """Parse a single kernel argument declaration."""
    # Check for [[buffer(N)]]
    buf_m = _RE_BUFFER_ATTR.search(raw)
    tg_m = _RE_TG_ATTR.search(raw)
    buffer_idx = int(buf_m.group(1)) if buf_m else None
    tg_idx = int(tg_m.group(1)) if tg_m else None

    # Strip attributes [[...]] for type parsing
    clean = re.sub(r"\[\[.*?\]\]", "", raw).strip()

    # Detect address space
    address_space = "builtin"
    for space in _ADDR_SPACES:
        if re.search(rf"\b{space}\b", clean):
            address_space = space
            break

    is_const = bool(re.search(r"\bconst\b", clean))

    # Extract name (last identifier before any [[...]])
    tokens = re.findall(r"\b\w+\b", clean)
    name = tokens[-1] if tokens else "unknown"

    # Detect builtin by attribute
    builtin_attrs = {
        "thread_position_in_grid",
        "thread_position_in_threadgroup",
        "threadgroup_position_in_grid",
        "threads_per_threadgroup",
        "threads_per_grid",
        "thread_index_in_simdgroup",
        "simdgroup_index_in_threadgroup",
        "thread_index_in_quadgroup",
        "quadgroup_index_in_threadgroup",
        "simd_lane_id",
        "simd_group_id",
    }
    builtin_attr_pattern = r"\[\[(" + "|".join(re.escape(b) for b in builtin_attrs) + r")\]\]"
    if re.search(builtin_attr_pattern, raw):
        address_space = "builtin"

    # Determine type string (everything before the name)
    type_str = clean
    for space in _ADDR_SPACES:
        type_str = re.sub(rf"\b{space}\b", "", type_str)
    type_str = re.sub(r"\bconst\b", "", type_str)
    type_str = type_str.replace("*", "").replace("&", "").strip()
    # Remove trailing name
    type_str = re.sub(rf"\b{re.escape(name)}\b$", "", type_str).strip()

    is_output = address_space == "device" and not is_const and buffer_idx is not None

    return KernelArg(
        name=name,
        type=type_str,
        address_space=address_space,
        is_const=is_const,
        is_output=is_output,
        buffer_idx=buffer_idx,
        tg_idx=tg_idx,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_kernels(source: str) -> list[KernelSig]:
    """Extract all [[kernel]] function signatures and bodies from MSL source."""
    lines = source.splitlines(keepends=True)
    # Build a line-start offset table for line-number lookup
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    def offset_to_line(offset: int) -> int:
        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1  # 1-based

    kernels: list[KernelSig] = []

    for m in _RE_KERNEL_START.finditer(source):
        name = m.group(1)
        line_no = offset_to_line(m.start())

        # Check for [[max_total_threads_per_threadgroup]] in the 3 lines before
        line_idx = line_no - 1  # 0-based
        preceding = "".join(lines[max(0, line_idx - 3) : line_idx])
        has_max = bool(_RE_MAX_THREADS.search(preceding))

        # Find the closing ) of the argument list
        paren_start = source.index("(", m.start())
        depth = 0
        i = paren_start
        while i < len(source):
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        args_str = source[paren_start + 1 : i]
        # Strip inline // comments before splitting — comment text would otherwise
        # bleed into the next argument's `raw` field via the bracket-depth tracker.
        args_str = re.sub(r"//[^\n]*", "", args_str)
        raw_args = _split_args(args_str)

        # Find the opening { of the body
        brace_pos = source.find("{", i)
        if brace_pos == -1:
            continue
        body, _ = _find_body(source, brace_pos)

        args = [_parse_arg(a) for a in raw_args if a.strip()]

        kernels.append(
            KernelSig(
                name=name,
                line=line_no,
                args=args,
                body=body,
                has_max_threads_attr=has_max,
            )
        )

    return kernels
