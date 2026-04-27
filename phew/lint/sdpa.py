"""Rule: manual softmax attention → mx.fast.scaled_dot_product_attention."""

from __future__ import annotations

import ast

from ._utils import contains, is_call
from .rule import LintIssue, Rule


class SDPARule(Rule):
    id = "sdpa"
    description = (
        "softmax(Q @ K.T * s) @ V  →  mx.fast.scaled_dot_product_attention(Q, K, V, scale=s)"
    )

    def check(self, tree: ast.AST, filename: str) -> list[LintIssue]:
        visitor = _Visitor(filename)
        visitor.visit(tree)
        return visitor.issues


def _is_softmax(node: ast.expr) -> bool:
    return is_call(node, "softmax")


def _is_matmul(node: ast.expr) -> bool:
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult)


class _Visitor(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.issues: list[LintIssue] = []

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if (
            isinstance(node.op, ast.MatMult)
            and contains(node.left, _is_softmax)
            and contains(node.left, _is_matmul)
        ):
            self.issues.append(
                LintIssue(
                    file=self.filename,
                    line=node.lineno,
                    rule=SDPARule.id,
                    message=(
                        "manual SDPA  →  mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale)"
                    ),
                )
            )
        self.generic_visit(node)
