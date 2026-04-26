from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .deps import MemDep
from .dtype import Dtype

NodeId = int

_next_id: int = 0


def _new_id() -> NodeId:
    global _next_id
    _next_id += 1
    return _next_id


@dataclass
class Node:
    """Base class for μGraph nodes.

    Every node tracks:
      - A unique id within its graph
      - Output shape and dtype
      - R/W memory dependencies (BALLS discipline)
      - Input node ids (positional)
      - Op-specific attributes stored in `attrs`
    """

    id: NodeId = field(default_factory=_new_id, init=False)
    shape: tuple[int, ...] = field(default=())
    dtype: Dtype = field(default=Dtype.float32)
    deps: MemDep = field(default=MemDep.device_mem)
    inputs: list[NodeId] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def op(self) -> str:
        return type(self).__name__

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        result = 1
        for s in self.shape:
            result *= s
        return result

    @property
    def nbytes(self) -> int:
        return self.numel * self.dtype.itemsize

    def __repr__(self) -> str:
        return (
            f"{self.op}(id={self.id}, shape={self.shape}, "
            f"dtype={self.dtype.value}, inputs={self.inputs})"
        )
