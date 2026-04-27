"""Rule: (x @ W) * rms_scalar → mx.fast.rms_norm(x, None, eps) @ W."""

from __future__ import annotations

import ast

from ._utils import is_rms_scalar
from .rule import LintIssue, Rule


class NormedMatmulRule(Rule):
    id = "normed_matmul"
    description = (
        "(x @ W) * rsqrt(mean(x²)+eps)  →  mx.fast.rms_norm(x, None, eps=eps) @ W"
        "  [opt-in: SubstitutionClass.normed_matmul]"
    )

    def check(self, tree: ast.AST, filename: str) -> list[LintIssue]:
        visitor = _Visitor(filename)
        visitor.visit(tree)
        return visitor.issues


class _Visitor(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.issues: list[LintIssue] = []

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Mult):
            for side, other in ((node.left, node.right), (node.right, node.left)):
                if (
                    is_rms_scalar(side)
                    and isinstance(other, ast.BinOp)
                    and isinstance(other.op, ast.MatMult)
                ):
                    self.issues.append(
                        LintIssue(
                            file=self.filename,
                            line=node.lineno,
                            rule=NormedMatmulRule.id,
                            message=(
                                "(x @ W) * rsqrt(mean(x²)+eps)"
                                "  →  mx.fast.rms_norm(x, None, eps=eps) @ W"
                            ),
                        )
                    )
                    break
        self.generic_visit(node)
