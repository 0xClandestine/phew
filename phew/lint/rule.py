"""Rule ABC and LintIssue."""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LintIssue:
    file: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}  [{self.rule}]  {self.message}"


class Rule(ABC):
    """Base class for a lint rule.

    Subclasses implement :meth:`check`, which walks an already-parsed AST and
    returns all :class:`LintIssue` objects found.  The ``id`` and ``description``
    class attributes are used for documentation and ``--rule`` filtering.
    """

    #: Short machine-readable name, e.g. ``"rms_norm"``
    id: str
    #: One-line human description shown in ``phew lint --help``
    description: str

    @abstractmethod
    def check(self, tree: ast.AST, filename: str) -> list[LintIssue]:
        """Return all issues found in *tree*."""
