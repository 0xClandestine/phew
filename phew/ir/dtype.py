from __future__ import annotations

from enum import Enum


class Dtype(Enum):
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"
    int32 = "int32"
    int16 = "int16"
    int8 = "int8"
    uint8 = "uint8"
    bool_ = "bool"

    @property
    def itemsize(self) -> int:
        _sizes = {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
            "int32": 4,
            "int16": 2,
            "int8": 1,
            "uint8": 1,
            "bool": 1,
        }
        return _sizes[self.value]

    @property
    def is_floating(self) -> bool:
        return self in (Dtype.float32, Dtype.float16, Dtype.bfloat16)

    @property
    def is_reduced_precision(self) -> bool:
        return self in (Dtype.float16, Dtype.bfloat16)

    def to_mlx(self) -> str:
        """Return the mlx.core dtype attribute name."""
        _map = {
            "float32": "float32",
            "float16": "float16",
            "bfloat16": "bfloat16",
            "int32": "int32",
            "int16": "int16",
            "int8": "int8",
            "uint8": "uint8",
            "bool": "bool_",
        }
        return _map[self.value]

    @staticmethod
    def from_mlx(mlx_dtype) -> "Dtype":
        import mlx.core as mx

        _map = {
            mx.float32: Dtype.float32,
            mx.float16: Dtype.float16,
            mx.bfloat16: Dtype.bfloat16,
            mx.int32: Dtype.int32,
            mx.int16: Dtype.int16,
            mx.int8: Dtype.int8,
            mx.uint8: Dtype.uint8,
            mx.bool_: Dtype.bool_,
        }
        return _map[mlx_dtype]
