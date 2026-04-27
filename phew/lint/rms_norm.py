"""Rule: manual RMSNorm → mx.fast.rms_norm."""

from __future__ import annotations

import ast

from ._utils import is_rms_scalar
from .rule import LintIssue, Rule


class RMSNormRule(Rule):
    id = "rms_norm"
    description = "x * rsqrt(mean(x²)+eps) * w  →  mx.fast.rms_norm(x, w, eps=eps)"

    def check(self, tree: ast.AST, filename: str) -> list[LintIssue]:
        visitor = _Visitor(filename)
        visitor.visit(tree)
        return visitor.issues


class _Visitor(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.issues: list[LintIssue] = []

    def _add(self, line: int) -> None:
        if not any(i.line == line for i in self.issues):
            self.issues.append(
                LintIssue(
                    file=self.filename,
                    line=line,
                    rule=RMSNormRule.id,
                    message="manual rms_norm  →  mx.fast.rms_norm(x, weight, eps=eps)",
                )
            )

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Mult):
            for side in (node.left, node.right):
                if is_rms_scalar(side):
                    self._add(node.lineno)
                    break
                # nested: (x * rsqrt_scalar) * w
                if isinstance(side, ast.BinOp) and isinstance(side.op, ast.Mult):
                    if any(is_rms_scalar(s) for s in (side.left, side.right)):
                        self._add(node.lineno)
                        break
        self.generic_visit(node)
