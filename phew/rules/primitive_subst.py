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
        changed |= self._match_rope(graph)
        changed |= self._match_sdpa(graph)
        return changed

    # ------------------------------------------------------------------
    # RMS norm — two structural forms:
    #   Form A (div+sqrt):  mul( div(x, sqrt(add(mean(x²), eps))), weight )
    #   Form B (mul+rsqrt): mul( mul(x, rsqrt(add(mean(x²), eps))), weight )
    # ------------------------------------------------------------------

    def _match_rms_norm(self, graph: "Graph") -> bool:
        from phew.ir import Elementwise, FastRMSNorm, Reduce

        def _find_mean_add_chain(inv_node):
            """Check inv_node is sqrt/rsqrt(add(mean(...), eps)).
            Return the add_node predecessor or None."""
            inv_preds = graph.predecessors(inv_node.id)
            if not inv_preds:
                return None
            add_node = inv_preds[0]
            if not (isinstance(add_node, Elementwise) and add_node.op == "add"):
                return None
            add_preds = graph.predecessors(add_node.id)
            has_mean = any(isinstance(p, Reduce) and p.op == "mean" for p in add_preds)
            return add_node if has_mean else None

        changed = False
        for node in list(graph.topo_order()):
            if not isinstance(node, Elementwise) or node.op not in ("mul", "multiply"):
                continue
            preds = graph.predecessors(node.id)
            if len(preds) < 2:
                continue

            x_node = None
            weight_node = None

            for i, pred in enumerate(preds):
                other = preds[1 - i]

                # Form A: div(x, sqrt(add(mean(x²), eps)))
                if isinstance(pred, Elementwise) and pred.op in ("div", "divide"):
                    div_preds = graph.predecessors(pred.id)
                    if len(div_preds) < 2:
                        continue
                    sqrt_n = next(
                        (p for p in div_preds if isinstance(p, Elementwise) and p.op == "sqrt"),
                        None,
                    )
                    if sqrt_n is not None and _find_mean_add_chain(sqrt_n) is not None:
                        # x is the non-sqrt input to div
                        x_node = next(p for p in div_preds if p.id != sqrt_n.id)
                        weight_node = other
                        break

                # Form B: mul(x, rsqrt(add(mean(x²), eps)))
                elif isinstance(pred, Elementwise) and pred.op in ("mul", "multiply"):
                    inner_preds = graph.predecessors(pred.id)
                    if len(inner_preds) < 2:
                        continue
                    rsqrt_n = next(
                        (p for p in inner_preds if isinstance(p, Elementwise) and p.op == "rsqrt"),
                        None,
                    )
                    if rsqrt_n is not None and _find_mean_add_chain(rsqrt_n) is not None:
                        # x is the non-rsqrt input to the inner mul
                        x_node = next(p for p in inner_preds if p.id != rsqrt_n.id)
                        weight_node = other
                        break

            if x_node is None or weight_node is None:
                continue

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
        """Match (x - mean(x)) / sqrt(var(x) + eps) * weight + bias → FastLayerNorm."""
        from phew.ir import Elementwise, FastLayerNorm, Reduce

        changed = False
        for node in list(graph.topo_order()):
            # Top-level: add(scaled_norm, bias)
            if not isinstance(node, Elementwise) or node.op not in ("add",):
                continue
            preds = graph.predecessors(node.id)
            if len(preds) < 2:
                continue

            # One pred is mul(normalized, weight); the other is bias
            mul_node = next(
                (p for p in preds if isinstance(p, Elementwise) and p.op in ("mul", "multiply")),
                None,
            )
            if mul_node is None:
                continue
            bias_node = next((p for p in preds if p.id != mul_node.id), None)

            mul_preds = graph.predecessors(mul_node.id)
            if len(mul_preds) < 2:
                continue

            # One pred is div(sub_x, std); the other is weight
            div_node = next(
                (p for p in mul_preds if isinstance(p, Elementwise) and p.op in ("div", "divide")),
                None,
            )
            if div_node is None:
                continue
            weight_node = next((p for p in mul_preds if p.id != div_node.id), None)

            div_preds = graph.predecessors(div_node.id)
            if len(div_preds) < 2:
                continue

            # Numerator: sub(x, mean(x)); denominator: sqrt(var(x) + eps)
            sub_node = next(
                (
                    p
                    for p in div_preds
                    if isinstance(p, Elementwise) and p.op in ("sub", "subtract")
                ),
                None,
            )
            sqrt_node = next(
                (p for p in div_preds if isinstance(p, Elementwise) and p.op == "sqrt"),
                None,
            )
            if sub_node is None or sqrt_node is None:
                continue

            # sqrt(add(reduce_var, eps))
            sqrt_preds = graph.predecessors(sqrt_node.id)
            if not sqrt_preds:
                continue
            var_add_node = sqrt_preds[0]
            if not (isinstance(var_add_node, Elementwise) and var_add_node.op == "add"):
                continue
            var_add_preds = graph.predecessors(var_add_node.id)
            var_node = next(
                (p for p in var_add_preds if isinstance(p, Reduce) and p.op in ("var", "variance")),
                None,
            )
            if var_node is None:
                continue

            # x from sub(x, mean_x)
            sub_preds = graph.predecessors(sub_node.id)
            if not sub_preds:
                continue
            x_node = sub_preds[0]

            if weight_node is None or bias_node is None:
                continue

            new_node = FastLayerNorm(
                shape=node.shape,
                dtype=node.dtype,
                inputs=[x_node.id, weight_node.id, bias_node.id],
            )
            graph.replace(node.id, new_node)
            changed = True

        return changed

    # ------------------------------------------------------------------
    # Rotary position embedding
    # ------------------------------------------------------------------

    def _match_rope(self, graph: "Graph") -> bool:
        """Match rotary embedding pattern → FastRoPE.

        Structural signature: Concat node whose predecessor subtrees both
        contain cos and sin elementwise ops sharing a common ancestor
        (the angle tensor).
        """
        from phew.ir import Concat, Elementwise, FastRoPE

        def _has_op(node, target_op, graph, depth=0):
            if depth > 6:
                return False
            if isinstance(node, Elementwise) and node.op == target_op:
                return True
            return any(_has_op(p, target_op, graph, depth + 1) for p in graph.predecessors(node.id))

        def _find_input(node, graph, depth=0):
            """Walk back past elementwise/reduce ops to find an Input-like root."""
            from phew.ir import Elementwise, Input

            if depth > 8:
                return None
            if isinstance(node, Input):
                return node
            preds = graph.predecessors(node.id)
            if not preds:
                return None
            # Prefer predecessors that aren't trig ops
            for p in preds:
                if isinstance(p, (Input,)):
                    return p
            for p in preds:
                if not (isinstance(p, Elementwise) and p.op in ("cos", "sin")):
                    result = _find_input(p, graph, depth + 1)
                    if result:
                        return result
            return None

        changed = False
        for node in list(graph.topo_order()):
            if not isinstance(node, Concat):
                continue
            preds = graph.predecessors(node.id)
            if len(preds) < 2:
                continue

            # Both halves must involve cos and sin
            all_preds_subtree = preds
            has_cos = any(_has_op(p, "cos", graph) for p in all_preds_subtree)
            has_sin = any(_has_op(p, "sin", graph) for p in all_preds_subtree)
            if not (has_cos and has_sin):
                continue

            x_node = _find_input(preds[0], graph)
            if x_node is None:
                continue

            dims = x_node.shape[-1] if x_node.shape else 0
            new_node = FastRoPE(
                shape=node.shape,
                dtype=node.dtype,
                inputs=[x_node.id],
                dims=dims,
                offset=0,
            )
            graph.replace(node.id, new_node)
            changed = True

        return changed

    # ------------------------------------------------------------------
    # Scaled dot-product attention
    # ------------------------------------------------------------------

    def _match_sdpa(self, graph: "Graph") -> bool:
        """Match softmax(Q @ K^T * scale) @ V → FastScaledDotProductAttention."""
        from phew.ir import (
            Cast,
            Constant,
            Elementwise,
            FastScaledDotProductAttention,
            MatMul,
            Reduce,
            Transpose,
        )

        changed = False

        for node in list(graph.topo_order()):
            # Final matmul: attn_weights @ V
            if not isinstance(node, MatMul):
                continue
            preds = graph.predecessors(node.id)
            if len(preds) < 2:
                continue
            v_node = preds[1]
            softmax_node = preds[0]

            # Walk past Cast nodes (e.g. bfloat16 cast after softmax)
            while isinstance(softmax_node, Cast):
                cast_preds = graph.predecessors(softmax_node.id)
                if not cast_preds:
                    break
                softmax_node = cast_preds[0]

            # Must be a softmax reduce or have one as a direct predecessor
            if isinstance(softmax_node, Reduce) and softmax_node.op == "softmax":
                pass  # direct softmax
            else:
                softmax_preds = graph.predecessors(softmax_node.id)
                if not any(isinstance(p, Reduce) and p.op == "softmax" for p in softmax_preds):
                    # Fall back: any Reduce predecessor (sum/max characteristic of manual softmax)
                    if not any(isinstance(p, Reduce) for p in softmax_preds):
                        continue

            # Walk back from softmax to find QK matmul and extract scale factor
            scale = 1.0

            def find_matmul_and_scale(n, depth=0):
                nonlocal scale
                if depth > 8:
                    return None
                if isinstance(n, MatMul):
                    return n
                for p in graph.predecessors(n.id):
                    # If this node is a mul with a Constant, extract the scale
                    if isinstance(n, Elementwise) and n.op in ("mul", "multiply"):
                        for pp in graph.predecessors(n.id):
                            if isinstance(pp, Constant) and isinstance(pp.value, (int, float)):
                                scale = float(pp.value)
                    result = find_matmul_and_scale(p, depth + 1)
                    if result:
                        return result
                return None

            qk_matmul = find_matmul_and_scale(softmax_node)
            if qk_matmul is None:
                continue
            qk_preds = graph.predecessors(qk_matmul.id)
            if len(qk_preds) < 2:
                continue
            q_node, k_node = qk_preds[0], qk_preds[1]

            # mx.fast.scaled_dot_product_attention expects keys in (B, H, S, D).
            # If the QK matmul used K.T (a Transpose node), unwrap it so we pass
            # the pre-transposed keys.
            if isinstance(k_node, Transpose):
                k_preds = graph.predecessors(k_node.id)
                if k_preds:
                    k_node = k_preds[0]

            new_node = FastScaledDotProductAttention(
                shape=node.shape,
                dtype=node.dtype,
                inputs=[q_node.id, k_node.id, v_node.id],
                scale=scale,
            )
            graph.replace(node.id, new_node)
            changed = True

        return changed
