"""Static AST linter for MLX inefficiency patterns.

Walks Python source files looking for known suboptimal patterns and
reports the line number and suggested replacement. No execution needed.

Rules
-----
rms_norm        x * rsqrt(mean(x²) + eps) * w
                → mx.fast.rms_norm(x, w, eps=eps)

normed_matmul   (x @ W) * rsqrt(mean(x²) + eps)
                → mx.fast.rms_norm(x, None, eps=eps) @ W
                (opt-in: SubstitutionClass.normed_matmul — ~1.2× on large K)

sdpa            softmax(Q @ K.T * scale) @ V
                → mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale)

compile         pure function using mx.* ops with no @mx.compile decorator
                → add @mx.compile
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LintIssue:
    file: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}  [{self.rule}]  {self.message}"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _fn_name(node: ast.expr) -> str | None:
    """Return the bare function/method name from a Call.func node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_call(node: ast.expr, *names: str) -> bool:
    return isinstance(node, ast.Call) and _fn_name(node.func) in names


def _is_sq(node: ast.expr) -> bool:
    """True if node looks like x*x or x**2."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        # Approximate: both sides are simple references (names, subscripts, attrs)
        return isinstance(node.left, (ast.Name, ast.Attribute, ast.Subscript)) and isinstance(
            node.right, (ast.Name, ast.Attribute, ast.Subscript)
        )
    return False


def _is_mean_of_sq(node: ast.expr) -> bool:
    """True if node is mean(x*x, ...) or (x*x).mean(...)."""
    if _is_call(node, "mean"):
        return bool(node.args) and _is_sq(node.args[0])  # type: ignore[union-attr]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mean"
    ):
        return _is_sq(node.func.value)
    return False


def _is_rms_scalar(node: ast.expr) -> bool:
    """True if node is rsqrt(mean(x²) + eps) or rsqrt((x*x).mean(...) + eps)."""
    if not _is_call(node, "rsqrt"):
        return False
    args = node.args  # type: ignore[union-attr]
    if not args:
        return False
    arg = args[0]
    if not isinstance(arg, ast.BinOp) or not isinstance(arg.op, ast.Add):
        return False
    return _is_mean_of_sq(arg.left) or _is_mean_of_sq(arg.right)


def _contains_softmax(node: ast.expr, depth: int = 0) -> bool:
    """True if node or any descendant is a softmax call."""
    if depth > 8:
        return False
    if _is_call(node, "softmax"):
        return True
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr) and _contains_softmax(child, depth + 1):
            return True
    return False


def _contains_matmul(node: ast.expr, depth: int = 0) -> bool:
    """True if node or any descendant is a @ BinOp."""
    if depth > 8:
        return False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        return True
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr) and _contains_matmul(child, depth + 1):
            return True
    return False


_MX_OPS = frozenset(
    [
        "matmul", "linear", "conv2d", "softmax", "sigmoid", "relu", "silu",
        "gelu", "tanh", "exp", "log", "sqrt", "rsqrt", "mean", "sum", "max",
        "argmax", "argmin", "sort", "scatter", "gather", "concatenate",
        "reshape", "transpose", "broadcast_to", "pad", "slice",
        "random", "zeros", "ones", "eye", "arange",
    ]
)


def _node_uses_mx(node: ast.AST) -> bool:
    """True if node contains any recognisable mx.* call."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            fname = _fn_name(n.func)
            if fname in _MX_OPS:
                return True
            # Explicit mx.something
            if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Attribute):
                if isinstance(n.func.value.value, ast.Name) and n.func.value.value.id == "mx":
                    return True
            if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
                if n.func.value.id in ("mx", "mlx"):
                    return True
    return False


def _has_compile_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        # @mx.compile or @partial(mx.compile, ...)
        if isinstance(dec, ast.Attribute) and dec.attr == "compile":
            return True
        if isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Attribute) and dec.func.attr == "compile":
                return True
            if _is_call(dec, "compile"):
                return True
    return False


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class LintChecker(ast.NodeVisitor):
    """Walk an AST and collect LintIssue objects."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.issues: list[LintIssue] = []

    def _add(self, rule: str, line: int, message: str) -> None:
        # Deduplicate: one issue per (rule, line)
        if any(i.rule == rule and i.line == line for i in self.issues):
            return
        self.issues.append(LintIssue(file=self.filename, line=line, rule=rule, message=message))

    # ------------------------------------------------------------------
    # BinOp: rms_norm and normed_matmul
    # ------------------------------------------------------------------

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Mult):
            self._check_mult(node)
        self.generic_visit(node)

    def _check_mult(self, node: ast.BinOp) -> None:
        for side, other in [(node.left, node.right), (node.right, node.left)]:
            if _is_rms_scalar(side):
                if isinstance(other, ast.BinOp) and isinstance(other.op, ast.MatMult):
                    self._add(
                        "normed_matmul",
                        node.lineno,
                        "(x @ W) * rsqrt(mean(x²)+eps)  →  mx.fast.rms_norm(x, None, eps=eps) @ W"
                        "  [opt-in: SubstitutionClass.normed_matmul]",
                    )
                else:
                    self._add(
                        "rms_norm",
                        node.lineno,
                        "manual rms_norm  →  mx.fast.rms_norm(x, weight, eps=eps)",
                    )
                return

            # Nested: (x * rsqrt_scalar) * w  — rsqrt at the inner level
            if isinstance(side, ast.BinOp) and isinstance(side.op, ast.Mult):
                for inner in (side.left, side.right):
                    if _is_rms_scalar(inner):
                        self._add(
                            "rms_norm",
                            node.lineno,
                            "manual rms_norm  →  mx.fast.rms_norm(x, weight, eps=eps)",
                        )
                        return

    # ------------------------------------------------------------------
    # MatMult: SDPA — softmax(Q @ K.T * s) @ V
    # ------------------------------------------------------------------

    def visit_BinOp_matmul(self, node: ast.BinOp) -> None:
        # Handled inside visit_BinOp to avoid double-dispatch confusion
        pass

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: F811
        if isinstance(node.op, ast.Mult):
            self._check_mult(node)
        elif isinstance(node.op, ast.MatMult):
            self._check_sdpa(node)
        self.generic_visit(node)

    def _check_sdpa(self, node: ast.BinOp) -> None:
        # pattern: softmax_expr @ V  where softmax_expr contains Q @ K
        if _contains_softmax(node.left) and _contains_matmul(node.left):
            self._add(
                "sdpa",
                node.lineno,
                "manual SDPA  →  mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale)",
            )

    # ------------------------------------------------------------------
    # FunctionDef: missing @mx.compile
    # ------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_compile(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_compile(node)
        self.generic_visit(node)

    def _check_compile(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if _has_compile_decorator(node):
            return
        # Skip constructors, dunders, and class methods (self/cls first arg).
        # @mx.compile belongs on standalone pure functions, not nn.Module methods.
        if node.name.startswith("_"):
            return
        args = node.args.args
        if args and args[0].arg in ("self", "cls"):
            return
        if _node_uses_mx(node):
            self._add(
                "compile",
                node.lineno,
                f"def {node.name}() uses mx ops but has no @mx.compile"
                "  →  add @mx.compile for ~1.5–3× free speedup",
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def lint_file(path: str | Path) -> list[LintIssue]:
    """Parse and lint a single Python file. Returns list of issues."""
    path = Path(path)
    source = path.read_text()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    checker = LintChecker(str(path))
    checker.visit(tree)
    return checker.issues


def lint_path(path: str | Path, glob: str = "**/*.py") -> list[LintIssue]:
    """Recursively lint all Python files under path."""
    path = Path(path)
    if path.is_file():
        return lint_file(path)
    issues: list[LintIssue] = []
    for p in sorted(path.glob(glob)):
        issues.extend(lint_file(p))
    return issues
