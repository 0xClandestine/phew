from __future__ import annotations

from enum import Flag, auto


class MemDep(Flag):
    """BALLS R/W dependency tracking — determines legal commutations."""

    NONE = 0
    device_mem = auto()
    threadgroup_mem = auto()
    register = auto()
    control_flow = auto()

    @classmethod
    def graph_level(cls) -> "MemDep":
        """Typical R/W deps for graph-level (MLX) ops."""
        return cls.device_mem

    @classmethod
    def threadgroup_level(cls) -> "MemDep":
        """Typical R/W deps for threadgroup/simdgroup ops."""
        return cls.threadgroup_mem | cls.register
