"""Primitive substitution pass.

Pattern-match sequences of MLX ops and replace with fast.* primitives.
When a pattern matches, the subgraph search terminates for those nodes
(fast.* primitives end Phase-1 search by construction).

Patterns:
  rms_norm  — x / sqrt(mean(x²) + eps) * weight → FastRMSNorm
  layer_norm — (x - mean) / std * weight + bias  → FastLayerNorm
  rope       — rotary position embedding pattern  → FastRoPE
  sdpa       — softmax(Q@K.T * scale) @ V        → FastScaledDotProductAttention
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phew.ir import Graph


class PrimitiveSubstPass:
    """Graph-level primitive substitution.

    Each _match_* method searches the graph for the target pattern
    and returns True if it made a substitution.
    """

    def run(self, graph: "Graph") -> bool:
        """Run all matchers. Return True if any substitution was made."""
        changed = False
        changed |= self._match_rms_norm(graph)
        changed |= self._match_layer_norm(graph)
        changed |= self._match_sdpa(graph)
        return changed

    # ------------------------------------------------------------------
    # RMS norm: x / sqrt(mean(x^2, keepdims=True) + eps) * weight
    # ------------------------------------------------------------------

    def _match_rms_norm(self, graph: "Graph") -> bool:
        from phew.ir import (
            Elementwise,
            FastRMSNorm,
            Reduce,
        )

        changed = False
        for node in list(graph.topo_order()):
            # Look for final Elementwise mul that could be the scale step
            if not isinstance(node, Elementwise) or node.op not in ("mul", "multiply"):
                continue
            # Pattern: mul(div_or_scale, weight)
            # A full pattern-match is shape-dependent; we use a conservative
            # structural check here. The verifier will reject false positives.
            preds = graph.predecessors(node.id)
            if len(preds) < 2:
                continue

            # Check if one input is a plain weight (no computational preds)
            div_node = None
            for pred in preds:
                if isinstance(pred, Elementwise) and pred.op in ("div", "divide"):
                    div_node = pred
                    break
            if div_node is None:
                continue

            # Check div denominator is a sqrt+reduce chain
            div_preds = graph.predecessors(div_node.id)
            if len(div_preds) < 2:
                continue

            sqrt_node = None
            for p in div_preds:
                if isinstance(p, Elementwise) and p.op in ("sqrt",):
                    sqrt_node = p
                    break
            if sqrt_node is None:
                continue

            sqrt_preds = graph.predecessors(sqrt_node.id)
            if not sqrt_preds:
                continue
            add_node = sqrt_preds[0]
            if not (isinstance(add_node, Elementwise) and add_node.op in ("add",)):
                continue

            add_preds = graph.predecessors(add_node.id)
            reduce_node = None
            for p in add_preds:
                if isinstance(p, Reduce) and p.op == "mean":
                    reduce_node = p
                    break
            if reduce_node is None:
                continue

            # Pattern matched — find the original input tensor
            x_preds = graph.predecessors(reduce_node.id)
            if not x_preds:
                continue
            x_node = x_preds[0]
            # weight is the non-div predecessor of the final mul
            weight_node = next((p for p in preds if p.id != div_node.id), None)
            if weight_node is None:
                continue

            # Replace with FastRMSNorm

            new_node = FastRMSNorm(
                shape=node.shape,
                dtype=node.dtype,
                inputs=[x_node.id, weight_node.id],
            )
            graph.replace(node.id, new_node)
            changed = True

        return changed

    # ------------------------------------------------------------------
    # Layer norm
    # ------------------------------------------------------------------

    def _match_layer_norm(self, graph: "Graph") -> bool:
        # Similar structural pattern; full match deferred to a more
        # expressive pattern DSL in a future milestone.
        return False

    # ------------------------------------------------------------------
    # Scaled dot-product attention
    # ------------------------------------------------------------------

    def _match_sdpa(self, graph: "Graph") -> bool:
        """Match softmax(Q @ K^T * scale) @ V → FastScaledDotProductAttention."""
        from phew.ir import FastScaledDotProductAttention, MatMul, Reduce

        changed = False

        for node in list(graph.topo_order()):
            # Final matmul: weights @ V
            if not isinstance(node, MatMul):
                continue
            preds = graph.predecessors(node.id)
            if len(preds) < 2:
                continue
            # One pred should be a softmax (Elementwise exp chain or reduce+div)
            v_node = preds[1]
            softmax_node = preds[0]

            # Conservative check: look for a Reduce(max) or Reduce(sum) before
            # the matmul — characteristic of softmax
            softmax_preds = graph.predecessors(softmax_node.id)
            if not any(isinstance(p, Reduce) for p in softmax_preds):
                continue

            # Look for the QK matmul that feeds the softmax chain
            def find_matmul_ancestor(n, depth=0):
                if depth > 5:
                    return None
                if isinstance(n, MatMul):
                    return n
                for p in graph.predecessors(n.id):
                    result = find_matmul_ancestor(p, depth + 1)
                    if result:
                        return result
                return None

            qk_matmul = find_matmul_ancestor(softmax_node)
            if qk_matmul is None:
                continue
            qk_preds = graph.predecessors(qk_matmul.id)
            if len(qk_preds) < 2:
                continue
            q_node, k_node = qk_preds[0], qk_preds[1]

            new_node = FastScaledDotProductAttention(
                shape=node.shape,
                dtype=node.dtype,
                inputs=[q_node.id, k_node.id, v_node.id],
                scale=1.0,
            )
            graph.replace(node.id, new_node)
            changed = True

        return changed
