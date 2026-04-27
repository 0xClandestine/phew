"""PHEW CLI.

Commands:
  phew run    <input.py>   — optimize and emit
  phew bench  <input.py>   — baseline benchmark only
  phew trace  <input.py>   — capture Metal GPU trace
  phew verify <input.py> <optimized.py>  — verify equivalence
  phew lint   <path>       — static scan for inefficiency patterns
  phew metal  list|wrap    — analyse .metal files
  phew skill               — print Claude Code skill guide
  phew upgrade             — upgrade to latest version
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import click
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_module(path: str):
    """Load a Python file as a module and return it.

    Patches ``mx.fast.metal_kernel`` before exec so that any
    ``mx.fast.metal_kernel(...)`` calls at module level produce
    ``_MetalKernelWrapper`` objects that PHEW's tracer can intercept.
    """
    from phew.ir.importer import _MetalKernelWrapper

    try:
        import mlx.core.fast as _mx_fast

        _orig_mk = getattr(_mx_fast, "metal_kernel", None)
    except ImportError:
        _orig_mk = None

    if _orig_mk is not None:

        def _patched_mk(name, input_names, output_names, source, header="", **kw):
            real = _orig_mk(
                name=name,
                input_names=input_names,
                output_names=output_names,
                source=source,
                header=header,
                **kw,
            )
            return _MetalKernelWrapper(real, name, input_names, output_names, source, header)

        _mx_fast.metal_kernel = _patched_mk

    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("_phew_target", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if _orig_mk is not None:
        _mx_fast.metal_kernel = _orig_mk  # restore; wrappers remain in module globals

    return mod


def _get_fn_and_factory(mod) -> tuple[Callable, Callable]:
    """Extract (fn, input_factory) from a loaded module.

    The module must define:
      - `fn`: the function to optimize
      - `input_factory(size_label, seed) -> (args, kwargs)`: input generator
    """
    fn = getattr(mod, "fn", None)
    input_factory = getattr(mod, "input_factory", None)
    if fn is None:
        raise click.ClickException("Module must define a function named `fn` to optimize.")
    if input_factory is None:
        raise click.ClickException(
            "Module must define `input_factory(size_label, seed) -> (args, kwargs)`."
        )
    return fn, input_factory


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="phew-mlx")
def cli():
    """PHEW — MLX/Metal optimizer for Apple Silicon."""


# ---------------------------------------------------------------------------
# phew run
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output .py file (default: stdout)")
@click.option("--trace", default=None, help="Path to existing .gputrace file")
@click.option(
    "--allow-fp16",
    is_flag=True,
    default=False,
    help="Opt in to fp32→fp16 precision substitution",
)
@click.option(
    "--allow-bf16",
    is_flag=True,
    default=False,
    help="Opt in to fp32→bf16 precision substitution",
)
@click.option(
    "--allow-quant",
    is_flag=True,
    default=False,
    help="Opt in to 4-bit quantization of matmuls",
)
@click.option("--eqsat-iters", default=30, show_default=True)
@click.option(
    "--strategy",
    default="greedy",
    show_default=True,
    type=click.Choice(["greedy", "ilp"]),
    help="E-graph extraction strategy",
)
@click.option(
    "--fusion",
    is_flag=True,
    default=False,
    help="Enable Phase-2 elementwise fusion into Metal kernels",
)
@click.option(
    "--diff",
    is_flag=True,
    default=False,
    help="Print a unified diff of input vs optimized source to stdout",
)
@click.option(
    "--diff-output",
    default=None,
    metavar="FILE",
    help="Write the unified diff to FILE (implies --diff)",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit result as JSON (suppresses all other output)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress progress output and search trace",
)
def run(
    input_file,
    output,
    trace,
    allow_fp16,
    allow_bf16,
    allow_quant,
    eqsat_iters,
    strategy,
    fusion,
    diff,
    diff_output,
    as_json,
    quiet,
):
    """Optimize INPUT_FILE and emit faster equivalent code.

    Exits 0 if a significant, verified speedup was found.
    Exits 1 if optimization failed, verification failed, or no speedup.
    """
    from phew import Optimizer
    from phew.verify import SubstitutionClass

    _check_metal()

    mod = _load_module(input_file)
    fn, input_factory = _get_fn_and_factory(mod)

    enabled = {SubstitutionClass.fp32_to_fp32}
    if allow_fp16:
        enabled.add(SubstitutionClass.fp32_to_fp16)
        if not as_json and not quiet:
            console.print("[yellow]Opt-in: fp32→fp16 precision substitution[/yellow]")
    if allow_bf16:
        enabled.add(SubstitutionClass.fp32_to_bf16)
        if not as_json and not quiet:
            console.print("[yellow]Opt-in: fp32→bf16 precision substitution[/yellow]")
    if allow_quant:
        enabled.add(SubstitutionClass.quantized_4bit)
        if not as_json and not quiet:
            console.print("[yellow]Opt-in: 4-bit quantization[/yellow]")

    opt = Optimizer(
        fn=fn,
        input_factory=input_factory,
        enabled_subst_classes=enabled,
        max_eqsat_iters=eqsat_iters,
        extraction_strategy=strategy,
        fn_name=getattr(mod, "fn_name", "optimized"),
        enable_fusion=fusion,
    )

    if not as_json and not quiet:
        console.print("[bold]Running PHEW optimizer...[/bold]")

    result = opt.run(trace_path=trace)

    if as_json:
        print(
            json.dumps(
                {
                    "baseline_ms": result.baseline_ms,
                    "optimized_ms": result.optimized_ms,
                    "speedup": result.speedup,
                    "significant": result.is_significant,
                    "verified": result.verification_passed,
                    "applied_rules": result.applied_rules,
                    "hardware": result.hardware_info,
                    "warnings": result.warnings,
                    "search_trace": result.search_trace,
                }
            )
        )
    else:
        _print_result(result, quiet=quiet)

    if diff or diff_output:
        import difflib

        original = Path(input_file).read_text().splitlines(keepends=True)
        optimized = result.output_source.splitlines(keepends=True)
        diff_text = "".join(
            difflib.unified_diff(
                original,
                optimized,
                fromfile=input_file,
                tofile=output or input_file + " [optimized]",
            )
        )

        if diff_output:
            Path(diff_output).write_text(diff_text)
            if not as_json and not quiet:
                console.print(f"\n[green]Diff written to {diff_output}[/green]")

        if diff and not as_json:
            if diff_text:
                console.print("\n[bold]Diff:[/bold]")
                for line in diff_text.splitlines():
                    esc = escape(line)
                    if line.startswith("+++") or line.startswith("---"):
                        console.print(f"[bold]{esc}[/bold]")
                    elif line.startswith("+"):
                        console.print(f"[green]{esc}[/green]")
                    elif line.startswith("-"):
                        console.print(f"[red]{esc}[/red]")
                    elif line.startswith("@@"):
                        console.print(f"[cyan]{esc}[/cyan]")
                    else:
                        console.print(esc)
            else:
                console.print("\n[dim]No changes — optimized source is identical to input.[/dim]")

    if output:
        Path(output).write_text(result.output_source)
        if not as_json and not quiet:
            console.print(f"\n[green]Optimized code written to {output}[/green]")
    elif not diff and not diff_output and not as_json:
        console.print("\n[bold]Optimized source:[/bold]")
        console.print(result.output_source)

    # Exit 1 if no meaningful result for the agent
    if not result.is_significant or not result.verification_passed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# phew bench
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--n-warmup", default=5, show_default=True)
@click.option("--n-bench", default=20, show_default=True)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit results as JSON",
)
def bench(input_file, n_warmup, n_bench, as_json):
    """Benchmark the baseline function in INPUT_FILE."""
    from phew.bench import benchmark

    _check_metal()

    mod = _load_module(input_file)
    fn, input_factory = _get_fn_and_factory(mod)

    rows = []
    for size_label in ["small", "typical", "large"]:
        try:
            args, kwargs = input_factory(size_label, 0)
            result = benchmark(fn, *args, n_warmup=n_warmup, n_bench=n_bench, **kwargs)
            rows.append(
                {
                    "size": size_label,
                    "mean_ms": result.mean_ms,
                    "std_ms": result.std_ms,
                    "cv": result.cv,
                    "converged": result.converged,
                }
            )
        except Exception as exc:
            rows.append({"size": size_label, "error": str(exc)})

    if as_json:
        print(json.dumps(rows))
        return

    table = Table(title="Benchmark Results", box=box.ROUNDED)
    table.add_column("Size", style="cyan")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("Std (ms)", justify="right")
    table.add_column("CV", justify="right")
    table.add_column("Converged", justify="center")

    for row in rows:
        if "error" in row:
            table.add_row(row["size"], "ERROR", row["error"], "-", "-")
        else:
            table.add_row(
                row["size"],
                f"{row['mean_ms']:.3f}",
                f"{row['std_ms']:.3f}",
                f"{row['cv']:.3f}",
                "[green]yes[/green]" if row["converged"] else "[red]no[/red]",
            )

    console.print(table)


# ---------------------------------------------------------------------------
# phew trace
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    default="phew_trace.gputrace",
    show_default=True,
    help="Output .gputrace file path",
)
@click.option("--n-iters", default=10, show_default=True)
def trace(input_file, output, n_iters):
    """Capture a Metal GPU trace for INPUT_FILE.

    \b
    Run as:
        MTL_CAPTURE_ENABLED=1 phew trace input.py
    """
    from phew.trace import TraceContext

    _check_metal()

    if not TraceContext.check_env():
        console.print(
            "[bold red]ERROR:[/bold red] MTL_CAPTURE_ENABLED=1 must be set "
            "before starting the process.\n"
            "Re-run as:  MTL_CAPTURE_ENABLED=1 phew trace " + input_file
        )
        sys.exit(1)

    out_path = Path(output)
    if out_path.exists():
        console.print(
            f"[bold red]ERROR:[/bold red] Output path {output} already exists. "
            "Remove it first (Metal silently fails if path exists)."
        )
        sys.exit(1)

    mod = _load_module(input_file)
    fn, input_factory = _get_fn_and_factory(mod)
    args, kwargs = input_factory("typical", 0)

    import mlx.core as mx

    console.print(f"[bold]Capturing {n_iters} iterations to {output}...[/bold]")
    with TraceContext(out_path):
        for _ in range(n_iters):
            mx.eval(fn(*args, **kwargs))

    console.print(f"[green]Trace saved to {output}[/green]")
    console.print("Open in Xcode → Instruments to inspect GPU counters.")


# ---------------------------------------------------------------------------
# phew verify
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("baseline_file", type=click.Path(exists=True))
@click.argument("optimized_file", type=click.Path(exists=True))
@click.option(
    "--allow-fp16",
    is_flag=True,
    default=False,
    help="Enable fp16 tolerance for verification",
)
@click.option("--allow-bf16", is_flag=True, default=False)
@click.option("--allow-quant", is_flag=True, default=False)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit result as JSON",
)
def verify(baseline_file, optimized_file, allow_fp16, allow_bf16, allow_quant, as_json):
    """Verify that OPTIMIZED_FILE is equivalent to BASELINE_FILE.

    Exits 0 on PASS, 1 on FAIL.
    """
    from phew.verify import EquivalenceChecker, SubstitutionClass

    baseline_mod = _load_module(baseline_file)
    optimized_mod = _load_module(optimized_file)

    baseline_fn, input_factory = _get_fn_and_factory(baseline_mod)
    optimized_fn = getattr(optimized_mod, "fn", getattr(optimized_mod, "optimized", None))
    if optimized_fn is None:
        raise click.ClickException("Optimized module must define `fn` or `optimized`.")

    enabled = {SubstitutionClass.fp32_to_fp32}
    if allow_fp16:
        enabled.add(SubstitutionClass.fp32_to_fp16)
    if allow_bf16:
        enabled.add(SubstitutionClass.fp32_to_bf16)
    if allow_quant:
        enabled.add(SubstitutionClass.quantized_4bit)

    checker = EquivalenceChecker(enabled_classes=enabled)
    result = checker.check(baseline_fn, optimized_fn, input_factory)

    if as_json:
        print(
            json.dumps(
                {
                    "passed": result.passed,
                    "failures": result.failures if hasattr(result, "failures") else [],
                }
            )
        )
    elif result.passed:
        console.print(f"[bold green]PASS[/bold green] {result}")
    else:
        console.print(f"[bold red]FAIL[/bold red] {result}")
        for f in result.failures:
            console.print(f"  [red]{f}[/red]")

    if not result.passed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_metal() -> None:
    try:
        import mlx.core as mx

        if not mx.metal.is_available():
            console.print(
                "[bold yellow]WARNING:[/bold yellow] Metal is not available. Running on CPU only."
            )
    except ImportError:
        console.print("[bold red]ERROR:[/bold red] mlx is not installed.")
        sys.exit(1)


def _print_result(result, quiet: bool = False) -> None:
    """Pretty-print an OptimizationResult."""
    table = Table(title="PHEW Optimization Result", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Baseline", f"{result.baseline_ms:.3f} ms")
    table.add_row("Optimized", f"{result.optimized_ms:.3f} ms")
    speedup_color = "green" if result.speedup > 1.03 else "yellow"
    table.add_row("Speedup", f"[{speedup_color}]{result.speedup:.3f}×[/{speedup_color}]")
    table.add_row("Significant", "[green]yes[/green]" if result.is_significant else "[red]no[/red]")
    table.add_row(
        "Verified",
        "[green]PASS[/green]" if result.verification_passed else "[red]FAIL[/red]",
    )
    table.add_row("Applied rules", ", ".join(result.applied_rules) or "none")

    if result.hardware_info:
        arch = result.hardware_info.get("architecture", "unknown")
        table.add_row("Hardware", str(arch))

    console.print(table)

    if result.warnings:
        for w in result.warnings:
            console.print(f"[yellow]WARNING:[/yellow] {w}")

    if not quiet:
        console.print("\n[dim]Search trace:[/dim]")
        for line in result.search_trace:
            console.print(f"  [dim]{line}[/dim]")


# ---------------------------------------------------------------------------
# phew upgrade
# ---------------------------------------------------------------------------


@cli.command()
def upgrade():
    """Upgrade phew to the latest version."""
    import importlib.metadata
    import shutil
    import subprocess

    try:
        before = importlib.metadata.version("phew-mlx")
    except importlib.metadata.PackageNotFoundError:
        before = None

    if shutil.which("uv"):
        subprocess.run(["uv", "tool", "upgrade", "phew-mlx"], check=False)
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "phew-mlx"], check=False
        )

    try:
        # Reload metadata after upgrade
        import importlib

        importlib.invalidate_caches()
        after = importlib.metadata.version("phew-mlx")
    except importlib.metadata.PackageNotFoundError:
        after = None

    if before and after:
        if before == after:
            console.print(f"[dim]Already at latest ({after})[/dim]")
        else:
            console.print(f"[green]{before} → {after}[/green]")


# ---------------------------------------------------------------------------
# phew skill
# ---------------------------------------------------------------------------


@cli.command()
def skill():
    """Print the phew skill guide for Claude Code."""
    import importlib.metadata
    import importlib.resources

    try:
        version = f"v{importlib.metadata.version('phew-mlx')}"
    except importlib.metadata.PackageNotFoundError:
        version = "(dev)"

    text = importlib.resources.files("phew").joinpath("SKILL.md").read_text()
    print(text.replace("{version}", version))


# ---------------------------------------------------------------------------
# phew lint
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--rule", "-r", multiple=True, help="Filter to rule(s): rms_norm, normed_matmul, sdpa, compile"
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit issues as JSON array",
)
def lint(path, rule, as_json):
    """Scan PATH for MLX and Metal inefficiency patterns.

    PATH may be a .py file, .metal file, or directory (searched recursively).

    \b
    Python rules:
      rms_norm       x*rsqrt(mean(x²)+eps)*w  →  mx.fast.rms_norm
      normed_matmul  (x@W)*rsqrt(mean(x²)+eps)  →  mx.fast.rms_norm(x,None)@W
      sdpa           softmax(Q@K.T*s)@V  →  mx.fast.scaled_dot_product_attention
      compile        mx-op function missing @mx.compile

    \b
    Metal rules (.metal files):
      max_threads         missing [[max_total_threads_per_threadgroup(N)]]
      missing_simd_reduce threadgroup reduction without simd_sum first pass
      half_accumulator    scalar half local used as accumulator
      unvectorized_loop   strided loop over half* reading scalarly
    """
    from phew.lint import lint_path
    from phew.metal.checker import lint_metal_file

    p = Path(path)
    if p.suffix == ".metal":
        issues = lint_metal_file(p)
    elif p.is_dir():
        issues = lint_path(p)
        for metal_file in sorted(p.rglob("*.metal")):
            issues.extend(lint_metal_file(metal_file))
    else:
        issues = lint_path(p)

    if rule:
        issues = [i for i in issues if i.rule in rule]

    if as_json:
        print(
            json.dumps(
                [
                    {"file": i.file, "line": i.line, "rule": i.rule, "message": i.message}
                    for i in sorted(issues, key=lambda i: (i.file, i.line))
                ]
            )
        )
        return

    if not issues:
        console.print("[green]No issues found.[/green]")
        return

    issues.sort(key=lambda i: (i.file, i.line))

    rule_colors = {
        "rms_norm": "cyan",
        "normed_matmul": "yellow",
        "sdpa": "magenta",
        "compile": "blue",
        "max_threads": "red",
        "half_accumulator": "yellow",
        "missing_simd_reduce": "magenta",
        "unvectorized_loop": "cyan",
    }

    for issue in issues:
        color = rule_colors.get(issue.rule, "white")
        loc = f"{issue.file}:{issue.line}"
        rule_col = f"[{color}]{issue.rule:<20}[/{color}]"
        # Split on  →  to color the suggestion green
        parts = issue.message.split("  →  ", 1)
        if len(parts) == 2:
            what, fix = escape(parts[0]), escape(parts[1])
            msg = f"{what}  [dim]→[/dim]  [green]{fix}[/green]"
        else:
            msg = escape(issue.message)
        console.print(f"[dim]{loc}[/dim]  {rule_col}  {msg}")

    total = len(issues)
    rule_counts: dict[str, int] = {}
    for i in issues:
        rule_counts[i.rule] = rule_counts.get(i.rule, 0) + 1
    summary = "  ".join(f"{r}: {c}" for r, c in sorted(rule_counts.items()))
    console.print(f"\n[bold]{total} issue{'s' if total != 1 else ''}[/bold]  ({summary})")


# ---------------------------------------------------------------------------
# phew metal
# ---------------------------------------------------------------------------


@cli.group()
def metal():
    """Analyse and wrap .metal kernel files."""


@metal.command("list")
@click.argument("metal_file", type=click.Path(exists=True))
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit kernel list as JSON",
)
def metal_list(metal_file, as_json):
    """List all [[kernel]] functions found in METAL_FILE."""
    from phew.metal.parser import parse_kernels

    source = Path(metal_file).read_text(errors="replace")
    kernels = parse_kernels(source)

    if as_json:
        print(
            json.dumps(
                [
                    {
                        "name": sig.name,
                        "line": sig.line,
                        "inputs": len(sig.input_args),
                        "outputs": len(sig.output_args),
                        "constants": len(sig.constant_args),
                        "has_max_threads_attr": sig.has_max_threads_attr,
                    }
                    for sig in kernels
                ]
            )
        )
        return

    if not kernels:
        console.print("[yellow]No [[kernel]] functions found.[/yellow]")
        return

    table = Table(title=f"Kernels in {Path(metal_file).name}", box=box.ROUNDED)
    table.add_column("Line", style="dim", justify="right")
    table.add_column("Name", style="cyan")
    table.add_column("Inputs", justify="right")
    table.add_column("Outputs", justify="right")
    table.add_column("Constants", justify="right")
    table.add_column("max_threads", justify="center")

    for sig in kernels:
        table.add_row(
            str(sig.line),
            sig.name,
            str(len(sig.input_args)),
            str(len(sig.output_args)),
            str(len(sig.constant_args)),
            "[green]yes[/green]" if sig.has_max_threads_attr else "[red]no[/red]",
        )

    console.print(table)


@metal.command("wrap")
@click.argument("metal_file", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output .py file (default: stdout)")
@click.option(
    "--kernel",
    "-k",
    multiple=True,
    help="Only wrap specific kernel(s) by name (default: all)",
)
def metal_wrap(metal_file, output, kernel):
    """Generate mx.fast.metal_kernel Python wrappers for kernels in METAL_FILE."""
    from phew.metal.parser import parse_kernels
    from phew.metal.wrapper import generate_file

    source = Path(metal_file).read_text(errors="replace")
    kernels = parse_kernels(source)

    if not kernels:
        raise click.ClickException("No [[kernel]] functions found.")

    include = list(kernel) if kernel else None
    text = generate_file(kernels, metal_file, include=include)

    if output:
        Path(output).write_text(text)
        console.print(f"[green]Wrapper written to {output}[/green]")
    else:
        print(text)
