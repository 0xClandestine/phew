"""Shared AST helpers used across lint rules."""

from __future__ import annotations

import ast


def fn_name(node: ast.expr) -> str | None:
    """Return the bare function/method name from a Call.func, ignoring module prefix."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def is_call(node: ast.expr, *names: str) -> bool:
    """True if *node* is a Call whose function name (ignoring module) is in *names*."""
    return isinstance(node, ast.Call) and fn_name(node.func) in names


def is_sq(node: ast.expr) -> bool:
    """True if *node* looks like ``x * x`` or ``x ** 2``."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return isinstance(node.left, (ast.Name, ast.Attribute, ast.Subscript)) and isinstance(
            node.right, (ast.Name, ast.Attribute, ast.Subscript)
        )
    return False


def is_mean_of_sq(node: ast.expr) -> bool:
    """True if *node* is ``mean(x*x, ...)`` or ``(x*x).mean(...)``."""
    if is_call(node, "mean"):
        return bool(node.args) and is_sq(node.args[0])  # type: ignore[union-attr]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mean"
    ):
        return is_sq(node.func.value)
    return False


def is_rms_scalar(node: ast.expr) -> bool:
    """True if *node* is ``rsqrt(mean(x²) + eps)``."""
    if not is_call(node, "rsqrt"):
        return False
    args = node.args  # type: ignore[union-attr]
    if not args:
        return False
    arg = args[0]
    if not isinstance(arg, ast.BinOp) or not isinstance(arg.op, ast.Add):
        return False
    return is_mean_of_sq(arg.left) or is_mean_of_sq(arg.right)


def contains(node: ast.expr, predicate, depth: int = 0, max_depth: int = 8) -> bool:
    """True if *node* or any descendant satisfies *predicate*."""
    if depth > max_depth:
        return False
    if predicate(node):
        return True
    return any(
        isinstance(child, ast.expr) and contains(child, predicate, depth + 1, max_depth)
        for child in ast.iter_child_nodes(node)
    )


def has_compile_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Attribute) and dec.attr == "compile":
            return True
        if isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Attribute) and dec.func.attr == "compile":
                return True
            if is_call(dec, "compile"):
                return True
    return False


_MX_OPS = frozenset(
    [
        "matmul",
        "linear",
        "conv2d",
        "softmax",
        "sigmoid",
        "relu",
        "silu",
        "gelu",
        "tanh",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "mean",
        "sum",
        "max",
        "argmax",
        "argmin",
        "sort",
        "scatter",
        "gather",
        "concatenate",
        "reshape",
        "transpose",
        "broadcast_to",
        "pad",
        "slice",
        "random",
        "zeros",
        "ones",
        "eye",
        "arange",
    ]
)


def node_uses_mx(node: ast.AST) -> bool:
    """True if *node* contains any recognisable ``mx.*`` call."""
    for n in ast.walk(node):
        if not isinstance(n, ast.Call):
            continue
        name = fn_name(n.func)
        if name in _MX_OPS:
            return True
        if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
            if n.func.value.id in ("mx", "mlx"):
                return True
    return False
