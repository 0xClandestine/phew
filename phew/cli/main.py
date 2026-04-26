"""PHEW CLI.

Commands:
  phew run    <input.py>   — optimize and emit
  phew bench  <input.py>   — baseline benchmark only
  phew trace  <input.py>   — capture Metal GPU trace
  phew verify <input.py> <optimized.py>  — verify equivalence
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import click
from rich import box
from rich.console import Console
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
@click.version_option(package_name="phew")
def cli():
    """PHEW — MLX/Metal superoptimizer for Apple Silicon."""


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
    help="Show a unified diff of input vs optimized source instead of the full output",
)
def run(
    input_file, output, trace, allow_fp16, allow_bf16, allow_quant, eqsat_iters, strategy, fusion, diff
):
    """Optimize INPUT_FILE and emit faster equivalent code."""
    from phew import Optimizer
    from phew.verify import SubstitutionClass

    _check_metal()

    mod = _load_module(input_file)
    fn, input_factory = _get_fn_and_factory(mod)

    enabled = {SubstitutionClass.fp32_to_fp32}
    if allow_fp16:
        enabled.add(SubstitutionClass.fp32_to_fp16)
        console.print("[yellow]Opt-in: fp32→fp16 precision substitution[/yellow]")
    if allow_bf16:
        enabled.add(SubstitutionClass.fp32_to_bf16)
        console.print("[yellow]Opt-in: fp32→bf16 precision substitution[/yellow]")
    if allow_quant:
        enabled.add(SubstitutionClass.quantized_4bit)
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

    console.print("[bold]Running PHEW optimizer...[/bold]")
    result = opt.run(trace_path=trace)

    _print_result(result)

    if diff:
        import difflib

        original = Path(input_file).read_text().splitlines(keepends=True)
        optimized = result.output_source.splitlines(keepends=True)
        delta = difflib.unified_diff(
            original,
            optimized,
            fromfile=input_file,
            tofile=output or input_file + " [optimized]",
        )
        diff_text = "".join(delta)
        if diff_text:
            console.print("\n[bold]Diff:[/bold]")
            for line in diff_text.splitlines():
                if line.startswith("+++") or line.startswith("---"):
                    console.print(f"[bold]{line}[/bold]", markup=False)
                elif line.startswith("+"):
                    console.print(f"[green]{line}[/green]", markup=False)
                elif line.startswith("-"):
                    console.print(f"[red]{line}[/red]", markup=False)
                elif line.startswith("@@"):
                    console.print(f"[cyan]{line}[/cyan]", markup=False)
                else:
                    console.print(line, markup=False)
        else:
            console.print("\n[dim]No changes — optimized source is identical to input.[/dim]")

    if output:
        Path(output).write_text(result.output_source)
        console.print(f"\n[green]Optimized code written to {output}[/green]")
    elif not diff:
        console.print("\n[bold]Optimized source:[/bold]")
        console.print(result.output_source)


# ---------------------------------------------------------------------------
# phew bench
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--n-warmup", default=5, show_default=True)
@click.option("--n-bench", default=20, show_default=True)
def bench(input_file, n_warmup, n_bench):
    """Benchmark the baseline function in INPUT_FILE."""
    from phew.bench import benchmark

    _check_metal()

    mod = _load_module(input_file)
    fn, input_factory = _get_fn_and_factory(mod)

    table = Table(title="Benchmark Results", box=box.ROUNDED)
    table.add_column("Size", style="cyan")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("Std (ms)", justify="right")
    table.add_column("CV", justify="right")
    table.add_column("Converged", justify="center")

    for size_label in ["small", "typical", "large"]:
        try:
            args, kwargs = input_factory(size_label, 0)
            result = benchmark(fn, *args, n_warmup=n_warmup, n_bench=n_bench, **kwargs)
            table.add_row(
                size_label,
                f"{result.mean_ms:.3f}",
                f"{result.std_ms:.3f}",
                f"{result.cv:.3f}",
                "[green]yes[/green]" if result.converged else "[red]no[/red]",
            )
        except Exception as exc:
            table.add_row(size_label, "ERROR", str(exc), "-", "-")

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
def verify(baseline_file, optimized_file, allow_fp16, allow_bf16, allow_quant):
    """Verify that OPTIMIZED_FILE is equivalent to BASELINE_FILE."""
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

    if result.passed:
        console.print(f"[bold green]PASS[/bold green] {result}")
    else:
        console.print(f"[bold red]FAIL[/bold red] {result}")
        for f in result.failures:
            console.print(f"  [red]{f}[/red]")
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


def _print_result(result) -> None:
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

    console.print("\n[dim]Search trace:[/dim]")
    for line in result.search_trace:
        console.print(f"  [dim]{line}[/dim]")
