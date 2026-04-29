from __future__ import annotations

from enum import Enum


class Dtype(Enum):
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"
    float64 = "float64"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"
    complex64 = "complex64"
    bool_ = "bool"

    @property
    def itemsize(self) -> int:
        _sizes = {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
            "float64": 8,
            "int8": 1,
            "int16": 2,
            "int32": 4,
            "int64": 8,
            "uint8": 1,
            "uint16": 2,
            "uint32": 4,
            "uint64": 8,
            "complex64": 8,
            "bool": 1,
        }
        return _sizes[self.value]

    @property
    def is_floating(self) -> bool:
        return self in (Dtype.float32, Dtype.float16, Dtype.bfloat16, Dtype.float64)

    @property
    def is_reduced_precision(self) -> bool:
        return self in (Dtype.float16, Dtype.bfloat16)

    def to_mlx(self) -> str:
        """Return the mlx.core dtype attribute name."""
        _map = {
            "float32": "float32",
            "float16": "float16",
            "bfloat16": "bfloat16",
            "float64": "float64",
            "int8": "int8",
            "int16": "int16",
            "int32": "int32",
            "int64": "int64",
            "uint8": "uint8",
            "uint16": "uint16",
            "uint32": "uint32",
            "uint64": "uint64",
            "complex64": "complex64",
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
            mx.float64: Dtype.float64,
            mx.int8: Dtype.int8,
            mx.int16: Dtype.int16,
            mx.int32: Dtype.int32,
            mx.int64: Dtype.int64,
            mx.uint8: Dtype.uint8,
            mx.uint16: Dtype.uint16,
            mx.uint32: Dtype.uint32,
            mx.uint64: Dtype.uint64,
            mx.complex64: Dtype.complex64,
            mx.bool_: Dtype.bool_,
        }
        return _map[mlx_dtype]
