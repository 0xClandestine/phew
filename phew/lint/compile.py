"""Rule: standalone mx-op functions missing @mx.compile."""

from __future__ import annotations

import ast

from .rule import LintIssue, Rule
from ._utils import has_compile_decorator, node_uses_mx


class CompileRule(Rule):
    id = "compile"
    description = "standalone mx-op function missing @mx.compile  →  add @mx.compile"

    def check(self, tree: ast.AST, filename: str) -> list[LintIssue]:
        visitor = _Visitor(filename)
        visitor.visit(tree)
        return visitor.issues


class _Visitor(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.issues: list[LintIssue] = []

    def _check(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if has_compile_decorator(node):
            return
        # Skip dunders, private helpers, and class methods (self/cls first arg).
        # @mx.compile targets standalone pure functions, not nn.Module methods.
        if node.name.startswith("_"):
            return
        args = node.args.args
        if args and args[0].arg in ("self", "cls"):
            return
        if node_uses_mx(node):
            self.issues.append(LintIssue(
                file=self.filename,
                line=node.lineno,
                rule=CompileRule.id,
                message=(
                    f"def {node.name}() uses mx ops but has no @mx.compile"
                    "  →  add @mx.compile for ~1.5–3× free speedup"
                ),
            ))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check(node)
        self.generic_visit(node)
