"""Lint a .metal file: parse kernels, run all Metal rules, return LintIssue list."""

from __future__ import annotations

from pathlib import Path

from phew.lint.rule import LintIssue

from .lint_rules import ALL_METAL_RULES, MetalLintIssue, MetalRule
from .parser import parse_kernels


def _to_lint_issue(m: MetalLintIssue) -> LintIssue:
    return LintIssue(file=m.file, line=m.line, rule=m.rule, message=m.message)


def lint_metal_file(
    path: str | Path,
    rules: list[MetalRule] | None = None,
) -> list[LintIssue]:
    """Parse and lint a single .metal file, returning standard LintIssue objects."""
    path = Path(path)
    source = path.read_text(errors="replace")
    kernels = parse_kernels(source)

    active_rules = rules if rules is not None else ALL_METAL_RULES
    issues: list[LintIssue] = []

    for sig in kernels:
        for rule in active_rules:
            for m in rule.check(sig, str(path)):
                issues.append(_to_lint_issue(m))

    issues.sort(key=lambda i: i.line)
    return issues
