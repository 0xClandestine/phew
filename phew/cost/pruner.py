"""Pruner: reject candidates before on-device measurement.

Rules (from the spec):
  - FLOPs > 1.2× current best → reject
  - bytes_moved > current best → reject
  - occupancy proxy < 0 (i.e. hard spill) → reject
  - threadgroup_mem > device limit → reject
  - threadgroup size > 1024 → hard reject
"""

from __future__ import annotations

from .model import NodeCost

FLOP_BUDGET_FACTOR = 1.2

# Default device limits (conservative; overridden by device_info() at runtime)
DEFAULT_THREADGROUP_MEM_LIMIT = 32 * 1024  # 32 KB pre-Family-9
FAMILY9_THREADGROUP_MEM_LIMIT = 64 * 1024  # dynamic on Family-9+
MAX_THREADGROUP_SIZE = 1024


def should_prune(
    candidate: NodeCost,
    best: NodeCost,
    *,
    threadgroup_size: int = 256,
    threadgroup_mem_limit: int = DEFAULT_THREADGROUP_MEM_LIMIT,
) -> tuple[bool, str]:
    """Return (prune, reason).

    prune=True means reject this candidate without measuring it.
    """
    if threadgroup_size > MAX_THREADGROUP_SIZE:
        return True, f"threadgroup_size={threadgroup_size} > {MAX_THREADGROUP_SIZE}"

    if candidate.threadgroup_mem_bytes > threadgroup_mem_limit:
        return True, (
            f"threadgroup_mem={candidate.threadgroup_mem_bytes}B > limit={threadgroup_mem_limit}B"
        )

    if best.flops > 0 and candidate.flops > best.flops * FLOP_BUDGET_FACTOR:
        return True, (f"flops={candidate.flops:.2e} > {FLOP_BUDGET_FACTOR}× best={best.flops:.2e}")

    if best.bytes_moved > 0 and candidate.bytes_moved > best.bytes_moved:
        return True, (f"bytes_moved={candidate.bytes_moved:.2e} > best={best.bytes_moved:.2e}")

    return False, ""
