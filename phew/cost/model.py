"""Static cost model for pruning candidates before measurement.

IMPORTANT: The cost model only prunes — it never decides the final ranking.
Final ranking is always on-device measurement.

Cost terms (primary: bytes_moved):
  - flops: FP multiply-adds
  - bytes_moved: weighted by cache level
    register=1, threadgroup=4, L1=8, device=32  (relative weights)
  - occupancy_proxy: based on register count per thread (Rosenzweig model)
  - threadgroup_mem_bytes: for limit checking

Pruning thresholds (relative to current best):
  - FLOPs > 1.2× best → reject
  - bytes_moved > best → reject
  - threadgroup > 1024 → hard reject
  - threadgroup_mem > device_limit → reject
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phew.ir import Graph, Node


# Relative cache-level weights (lower = cheaper)
CACHE_WEIGHT = {
    "register": 1,
    "threadgroup": 4,
    "l1": 8,
    "device": 32,
}

# Occupancy model (Rosenzweig)
FULL_OCCUPANCY_REG_THRESHOLD = 112  # regs per thread
SPILL_THRESHOLD = 256
THREADS_PER_SIMD = 32


@dataclass
class NodeCost:
    flops: float = 0.0
    bytes_moved: float = 0.0  # weighted bytes
    raw_bytes: float = 0.0  # unweighted (for reporting)
    threadgroup_mem_bytes: int = 0
    registers_per_thread: int = 32  # estimate
    threadgroup_size: int = 256

    @property
    def occupancy(self) -> float:
        """Estimated occupancy fraction [0, 1]."""
        r = self.registers_per_thread
        if r <= FULL_OCCUPANCY_REG_THRESHOLD:
            return 1.0
        if r >= SPILL_THRESHOLD:
            return 0.1  # heavy spilling
        # Linear falloff in steps of 64 threads
        steps = (r - FULL_OCCUPANCY_REG_THRESHOLD) // 64
        return max(0.1, 1.0 - steps * 0.1)

    @property
    def total(self) -> float:
        """Composite cost: bytes_moved is primary metric."""
        return self.bytes_moved


class CostModel:
    """Estimate the cost of a graph or individual node."""

    def node_cost(self, node: "Node") -> NodeCost:
        from phew.ir import (
            AsyncEval,
            Cast,
            Compile,
            Constant,
            Elementwise,
            FastLayerNorm,
            FastRMSNorm,
            FastRoPE,
            FastScaledDotProductAttention,
            Input,
            MatMul,
            MetalKernel,
            QuantizedMatMul,
            Reduce,
            Reshape,
            Transpose,
            VMap,
        )

        if isinstance(node, MatMul):
            return self._matmul_cost(node)
        if isinstance(node, Reduce):
            return self._reduce_cost(node)
        if isinstance(node, Elementwise):
            return self._elementwise_cost(node)
        if isinstance(node, Cast):
            return self._cast_cost(node)
        if isinstance(node, (Transpose, Reshape)):
            return self._reshape_cost(node)
        if isinstance(node, QuantizedMatMul):
            return self._quantized_matmul_cost(node)
        if isinstance(node, (FastRMSNorm, FastLayerNorm, FastRoPE)):
            return self._fast_primitive_cost(node)
        if isinstance(node, FastScaledDotProductAttention):
            return self._sdpa_cost(node)
        if isinstance(node, MetalKernel):
            return self._metal_kernel_cost(node)
        if isinstance(node, (Input, Constant, Compile, AsyncEval, VMap)):
            return NodeCost()
        return NodeCost()

    def graph_cost(self, graph: "Graph") -> NodeCost:
        total = NodeCost()
        for node in graph.topo_order():
            c = self.node_cost(node)
            total.flops += c.flops
            total.bytes_moved += c.bytes_moved
            total.raw_bytes += c.raw_bytes
            total.threadgroup_mem_bytes = max(total.threadgroup_mem_bytes, c.threadgroup_mem_bytes)
        return total

    # ------------------------------------------------------------------
    # Per-op cost estimators
    # ------------------------------------------------------------------

    def _matmul_cost(self, node) -> NodeCost:
        # shape = (..., M, N), inputs provide (..., M, K) and (..., K, N)
        if len(node.shape) < 2:
            return NodeCost()
        M, N = node.shape[-2], node.shape[-1]
        batch = math.prod(node.shape[:-2]) if len(node.shape) > 2 else 1
        # Infer K from output shape; not available without input nodes — use N as proxy
        K = N
        flops = batch * 2 * M * N * K
        raw = (batch * M * K + batch * K * N + batch * M * N) * node.dtype.itemsize
        bw = raw * CACHE_WEIGHT["device"]
        return NodeCost(flops=flops, bytes_moved=bw, raw_bytes=raw)

    def _reduce_cost(self, node) -> NodeCost:
        numel = node.numel
        raw = numel * node.dtype.itemsize
        bw = raw * CACHE_WEIGHT["device"]
        return NodeCost(flops=float(numel), bytes_moved=bw, raw_bytes=raw)

    def _elementwise_cost(self, node) -> NodeCost:
        numel = node.numel
        raw = numel * node.dtype.itemsize * 2  # one read, one write
        bw = raw * CACHE_WEIGHT["device"]
        return NodeCost(flops=float(numel), bytes_moved=bw, raw_bytes=raw)

    def _cast_cost(self, node) -> NodeCost:
        # Cast halves bytes if going to fp16
        in_bytes = node.numel * node.dtype.itemsize
        out_bytes = node.numel * node.target_dtype.itemsize
        raw = in_bytes + out_bytes
        bw = raw * CACHE_WEIGHT["device"]
        return NodeCost(flops=0.0, bytes_moved=bw, raw_bytes=raw)

    def _reshape_cost(self, node) -> NodeCost:
        # Reshape/transpose may be free (view) or require a copy
        raw = node.nbytes
        bw = raw * CACHE_WEIGHT["device"]
        return NodeCost(flops=0.0, bytes_moved=bw, raw_bytes=raw)

    def _quantized_matmul_cost(self, node) -> NodeCost:
        if len(node.shape) < 2:
            return NodeCost()
        M, N = node.shape[-2], node.shape[-1]
        batch = math.prod(node.shape[:-2]) if len(node.shape) > 2 else 1
        K = N
        flops = batch * 2 * M * N * K
        # Weight bytes compressed by bits/8
        weight_bytes = batch * K * N * (node.bits / 8)
        activation_bytes = batch * M * K * node.dtype.itemsize
        output_bytes = batch * M * N * node.dtype.itemsize
        raw = weight_bytes + activation_bytes + output_bytes
        bw = raw * CACHE_WEIGHT["device"]
        return NodeCost(flops=flops, bytes_moved=bw, raw_bytes=raw)

    def _fast_primitive_cost(self, node) -> NodeCost:
        # Fast primitives are hand-tuned; cost ~= elementwise pass
        raw = node.nbytes * 2
        bw = raw * CACHE_WEIGHT["device"]
        return NodeCost(flops=float(node.numel), bytes_moved=bw, raw_bytes=raw)

    def _sdpa_cost(self, node) -> NodeCost:
        # Approximation: 2 matmuls + softmax
        raw = node.nbytes * 6
        bw = raw * CACHE_WEIGHT["device"]
        flops = node.numel * 4.0
        return NodeCost(flops=flops, bytes_moved=bw, raw_bytes=raw)

    def _metal_kernel_cost(self, node) -> NodeCost:
        raw = node.nbytes * 2
        bw = raw * CACHE_WEIGHT["device"]
        tg_mem = node.attrs.get("threadgroup_mem_bytes", 0)
        regs = node.attrs.get("registers_per_thread", 32)
        tg_size = node.threadgroup[0] * node.threadgroup[1] * node.threadgroup[2]
        return NodeCost(
            flops=float(node.numel),
            bytes_moved=bw,
            raw_bytes=raw,
            threadgroup_mem_bytes=tg_mem,
            registers_per_thread=regs,
            threadgroup_size=tg_size,
        )
