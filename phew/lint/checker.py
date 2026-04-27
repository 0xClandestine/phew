"""LintChecker: runs all registered rules over a file or directory."""

from __future__ import annotations

import ast
from pathlib import Path

from .compile import CompileRule
from .normed_matmul import NormedMatmulRule
from .rms_norm import RMSNormRule
from .rule import LintIssue, Rule
from .sdpa import SDPARule

#: Canonical rule registry — add new rules here.
ALL_RULES: list[Rule] = [
    RMSNormRule(),
    NormedMatmulRule(),
    SDPARule(),
    CompileRule(),
]


class LintChecker:
    """Run a subset of rules over parsed source.

    Parameters
    ----------
    rules:
        Rules to run. Defaults to :data:`ALL_RULES`.
    """

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules = rules if rules is not None else ALL_RULES

    def check(self, tree: ast.AST, filename: str) -> list[LintIssue]:
        issues: list[LintIssue] = []
        for rule in self.rules:
            issues.extend(rule.check(tree, filename))
        issues.sort(key=lambda i: i.line)
        return issues


def lint_file(path: str | Path, rules: list[Rule] | None = None) -> list[LintIssue]:
    """Parse and lint a single Python file."""
    path = Path(path)
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return []
    return LintChecker(rules).check(tree, str(path))


def lint_path(
    path: str | Path,
    glob: str = "**/*.py",
    rules: list[Rule] | None = None,
) -> list[LintIssue]:
    """Recursively lint all Python files under *path*."""
    path = Path(path)
    if path.is_file():
        return lint_file(path, rules)
    issues: list[LintIssue] = []
    for p in sorted(path.glob(glob)):
        issues.extend(lint_file(p, rules))
    return issues
