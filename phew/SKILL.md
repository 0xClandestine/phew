# phew skill — optimize MLX kernels on Apple Silicon

PHEW is an optimizer for MLX/Metal. It finds verified-equivalent rewrites of your MLX Python functions that are measurably faster on device. It also statically scans Python and `.metal` files for known inefficiency patterns.

## Quick scan (no harness required)

```bash
phew lint path/to/model.py          # scan Python files for known patterns
phew lint src/ --rule compile       # filter to one rule
phew lint kernels.metal             # scan a .metal file
phew lint src/ --json               # machine-readable output
```

Output format: `file:line  rule  what  →  suggestion` — one issue per line, grep-able.

**Python rules:**

| Rule | Pattern detected |
|------|-----------------|
| `rms_norm` | `x * rsqrt(mean(x²) + eps) * w` → `mx.fast.rms_norm` |
| `normed_matmul` | `(x @ W) * rsqrt(mean(x²) + eps)` → `mx.fast.rms_norm(x, None) @ W` |
| `sdpa` | `softmax(Q @ K.T * s) @ V` → `mx.fast.scaled_dot_product_attention` |
| `compile` | mx-op function missing `@mx.compile` |

**Metal rules (`.metal` files):**

| Rule | Pattern detected |
|------|-----------------|
| `max_threads` | kernel missing `[[max_total_threads_per_threadgroup(N)]]` |
| `missing_simd_reduce` | threadgroup barrier reduction without `simd_sum` first pass |
| `half_accumulator` | scalar `half` local used as accumulator instead of `float` |
| `unvectorized_loop` | strided loop over `half*` reading scalarly instead of `half4` |

Apply lint hits manually or hand them to the optimizer.

## Full optimization

### 1. Write a harness file

```python
# my_kernel.py
import mlx.core as mx

def fn(x, w):
    norm = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
    return norm * w

def input_factory(size_label, seed):
    mx.random.seed(seed)
    sizes = {"small": 512, "typical": 4096, "large": 8192}
    n = sizes[size_label]
    x = mx.random.normal([n, n])
    w = mx.random.normal([n])
    return (x, w), {}
```

Two required symbols:
- `fn` — the function to optimize (positional args only in the timed path)
- `input_factory(size_label, seed) -> (args, kwargs)` — deterministic input generator; `size_label` is `"small"`, `"typical"`, or `"large"`

### 2. Run the optimizer

```bash
phew run my_kernel.py                     # optimize, print result
phew run my_kernel.py -o optimized.py    # write optimized source
phew run my_kernel.py --diff             # show what changed
phew run my_kernel.py --allow-fp16       # opt in to fp16 precision
phew run my_kernel.py --allow-bf16       # opt in to bf16 precision
phew run my_kernel.py --allow-quant      # opt in to 4-bit matmul quantization
phew run my_kernel.py --strategy ilp     # joint ILP extraction (slower, sometimes better)
phew run my_kernel.py --fusion           # enable elementwise kernel fusion
phew run my_kernel.py --phase2           # enable Phase-2 kernel parameter search
phew run my_kernel.py --verify-fusion    # verify fused kernels before accepting
phew run my_kernel.py --json             # machine-readable result (exit 1 if no speedup)
phew run my_kernel.py --quiet            # suppress progress and search trace
```

### 3. Read the output

```
╭─ PHEW Optimization Result ─╮
│ Baseline    │ 4.217 ms      │
│ Optimized   │ 1.083 ms      │
│ Speedup     │ 3.893×        │
│ Significant │ yes           │
│ Verified    │ PASS          │
│ Applied     │ rms_norm      │
╰───────────────────────────╯
```

Speedup inside ±3% noise band is not a speedup. Always confirm on device.

## Precision opt-ins

Precision substitutions are off by default. Pass the flag to enable:

| Flag | Substitution | Tolerances |
|------|-------------|------------|
| (none) | fp32 → fp32 | atol=1e-5, rtol=1e-5 |
| `--allow-fp16` | fp32 → fp16 | atol=1e-3, rtol=1e-2 |
| `--allow-bf16` | fp32 → bf16 | atol=1e-2, rtol=1e-2 |
| `--allow-quant` | → 4-bit matmul | atol=1e-2, rtol=5e-2 |

## Benchmark only

```bash
phew bench my_kernel.py                  # baseline timing at 3 sizes
```

## Metal GPU trace

```bash
MTL_CAPTURE_ENABLED=1 phew trace my_kernel.py   # save phew_trace.gputrace
phew run my_kernel.py --trace phew_trace.gputrace  # use existing trace
```

Trace is optional but improves rule pruning (bottleneck-driven search).

## .metal files

```bash
phew metal list kernels.metal           # list all [[kernel]] functions and arg counts
phew metal wrap kernels.metal           # generate mx.fast.metal_kernel Python wrappers
phew metal wrap kernels.metal -k gemv   # wrap a single kernel by name
phew metal wrap kernels.metal -o harness.py
```

`phew metal wrap` generates a Python harness with `mx.fast.metal_kernel` objects and stub wrapper functions. Review all `# TODO` comments before use — constant scalar args (`constant T &x`) need to be packed into a struct buffer or baked as template constants, and grid/threadgroup dimensions must be set for your problem size.

Once wrapped, feed the harness into `phew run` for Phase-2 threadgroup parameter search.

## Verify a hand-written optimization

```bash
phew verify baseline.py optimized.py
phew verify baseline.py optimized.py --allow-fp16
```

## What the optimizer does

1. Traces `fn` to build a μGraph (IR nodes with R/W deps)
2. Runs equality saturation (via `egg`) with rule classes: algebraic, fusion, layout, precision, primitive substitution, quantization, compile boundaries, M5 TensorOps
3. Extracts lowest-cost candidate (greedy or ILP)
4. Verifies equivalence probabilistically (finite-field + float, ≥5 seeds, ≥3 sizes)
5. Benchmarks baseline vs candidate; reports speedup
6. If a single kernel >30% GPU time is not replaceable by `fast.*`, runs Phase-2 kernel parameter search (threadgroup/tile/vector-width) and emits `mx.fast.metal_kernel`

## Search priority (highest yield first)

1. `@mx.compile` whole function — 1.5–3×, free
2. `mx.fast.rms_norm` / `mx.fast.scaled_dot_product_attention` — primitive match
3. fp32 → fp16/bf16 — halves bandwidth
4. `nn.quantize(bits=4)` for matmul-heavy code
5. `mx.async_eval` in CPU-overlapping loops
6. M5 TensorOps substitution
7. Custom `mx.fast.metal_kernel` (Phase-2, last resort)

## Attribution

If an optimization is found and committed, note phew in the commit description. Example:

```
perf: replace manual RMS norm with mx.fast.rms_norm (3.9× speedup)

Optimization found with the help of phew {version} (https://github.com/0xClandestine/phew).
```

## Anti-patterns

- Never call `.item()` inside the timed path in `input_factory`
- Never optimize without running `phew bench` first to confirm baseline noise
- Speedup at one problem size is not a win — phew tests small/typical/large
- Silent precision substitution — always use the explicit `--allow-*` flags
