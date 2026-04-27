<p align="center">
  <h1 align="center">phew</h1>
  <p align="center">Probably Hardly Ever Works — MLX optimizer for Apple Silicon</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
</p>

Point it at working MLX code. It finds a faster equivalent, proves the equivalence, and hands it back. If it can't beat your baseline across every problem size with verified correctness, you get your original code unchanged. No silent miscompiles.

One number matters: `optimized_ms / baseline_ms`, with proof.

---

## It actually works (sometimes — hence the name)

```
phew run examples/rms_norm.py
```

```
              PHEW Optimization Result
╭───────────────┬───────────────────────────────────╮
│ Metric        │                             Value │
├───────────────┼───────────────────────────────────┤
│ Baseline      │                          0.234 ms │
│ Optimized     │                          0.206 ms │
│ Speedup       │                            1.138× │
│ Applied rules │ compile_boundary, primitive_subst │
│ Hardware      │                     applegpu_g16s │
╰───────────────┴───────────────────────────────────╯
```

Seven steps, all automatic:

1. **Baseline benchmark** — auto-converges until σ/μ < 5%.
2. **Bottleneck classifier** — memory / compute / occupancy / launch-overhead. Prunes the rule set.
3. **Graph passes** — `mx.compile` wrapping, `mx.fast.*` substitution, M5 TensorOps detection.
4. **E-graph saturation** — algebraic, layout, fusion, precision, quantization, compile-boundary rules via [egglog](https://github.com/egraphs-good/egglog).
5. **Code emission** — clean, copy-pasteable MLX Python.
6. **Multi-size benchmark** — speedup must hold at small, typical, *and* large sizes.
7. **Equivalence verification** — Schwartz–Zippel-flavored: 5 seeds × 3 sizes × 4 edge variants. One failure, candidate dies.

---

## Install

```bash
uv tool install phew-mlx          # install globally
uv tool upgrade phew-mlx          # upgrade to latest
uv tool uninstall phew-mlx        # remove
```

Installs `phew` as a globally available CLI command. If the active virtualenv shadows it, run `deactivate` first. Requires macOS 13.3+ on Apple Silicon.

Dev install: `uv pip install -e ".[dev]"`.

## Usage

```python
from phew import Optimizer
from phew.verify import SubstitutionClass

def rms_norm(x, weight):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5) * weight

def input_factory(size, seed):
    mx.random.seed(seed)
    n = {"small": 256, "typical": 1024, "large": 4096}[size]
    return [mx.random.normal((n, 2048)), mx.ones((2048,))], {}

opt = Optimizer(fn=rms_norm, input_factory=input_factory)
# Opt in to lossy precision: enabled_subst_classes={SubstitutionClass.fp32_to_fp16}
print(opt.run().output_source)
```

## CLI

```
phew run    input.py [-o out.py]    # optimize and emit
phew bench  input.py                # baseline benchmark only
phew trace  input.py                # capture Metal GPU trace
phew verify baseline.py opt.py      # verify equivalence standalone
phew lint   path/                   # scan for inefficiency patterns (no harness needed)
```

### phew lint

Scans Python files statically for known MLX inefficiencies. No `input_factory`, no execution — just point it at a file or directory.

```
phew lint mlx_lm/models/
phew lint model.py -r rms_norm,sdpa   # filter to specific rules
```

| Rule | Pattern | Suggestion |
|---|---|---|
| `rms_norm` | `x * rsqrt(mean(x²)+eps) * w` | `mx.fast.rms_norm(x, w, eps=eps)` |
| `normed_matmul` | `(x @ W) * rsqrt(mean(x²)+eps)` | `mx.fast.rms_norm(x, None, eps=eps) @ W` |
| `sdpa` | `softmax(Q @ K.T * s) @ V` | `mx.fast.scaled_dot_product_attention(Q, K, V, scale=s)` |
| `compile` | standalone mx-op function, no `@mx.compile` | add `@mx.compile` |

Catches inline single-expression patterns. Split-assignment form (`rsqrt = mx.rsqrt(...); result = (x @ W) * rsqrt`) requires the full optimizer.

---

## What's inside

**IR** (`phew/ir/`) — Two-level μGraph (Mirage-derived) covering the full MLX surface, with per-node R/W tracking inspired by the BALLS scheduling discipline. The importer monkey-patches `mlx.core` at trace time. Yes, that's as fragile as it sounds.

**Graph passes** (`phew/rules/`) — Primitive substitution (RMS norm, SDPA), compile-boundary wrapping, M5/A19 TensorOps routing.

**E-graph** (`phew/egraph/`) — Equality saturation via egglog. Greedy cost-min extraction by default; ILP fallback (scipy) when greedy gets stuck.

| Rule set | What it does |
|---|---|
| `algebraic` | Commutativity, associativity, matmul reassociation |
| `layout` | Transpose-through-matmul: `(AB)ᵀ ↔ BᵀAᵀ` |
| `precision` | Double-cast elimination; fp32 → fp16/bf16 (opt-in) |
| `quantization` | `matmul → quantized_matmul(bits=4)` (opt-in) |
| `compile_boundaries` | `matmul(a,b) → compiled(matmul(a,b))` |
| `fusion` | Elementwise chains (placeholder) |

**Cost model** (`phew/cost/`) — Pruning only; final ranking is always on-device. Bytes weighted by hierarchy (register=1, threadgroup=4, L1=8, device=32). Occupancy from Rosenzweig's M1 model.

**Verification** (`phew/verify/`) — Four classes, opt-in for anything lossy:

| Class | atol | rtol | Opt-in |
|---|---|---|---|
| fp32 → fp32 | 1e-5 | 1e-5 | No |
| fp32 → fp16 | 1e-3 | 1e-2 | Yes |
| fp32 → bf16 | 1e-2 | 1e-2 | Yes |
| quantized (4-bit) | 1e-2 | 5e-2 | Yes |
| normed_matmul | 1e-3 | 1e-1 | Yes |

**Bench harness** (`phew/bench/`) — Doubles iterations until σ/μ < 5%. Speedups inside ±3% don't count. Multi-size convergence requires every size to beat baseline by >3%.

---

## Where it falls over (the honest part)

Named PHEW for a reason:

- **E-graph round-trip is incomplete.** `_egglog_to_graph` returns the original graph unchanged — e-graph rewrites don't yet affect emitted code. Graph-level passes do apply; that's where the example speedup comes from.
- **Phase-2 template constants** (`VW`, `UNROLL`) only take effect when the kernel source explicitly references those names. Threadgroup size is varied unconditionally via the `threadgroup=` call param and always has effect.
- **Tracer is fragile** with control flow, in-place updates, custom Metal kernels, or nested `mx.compile`. Common patterns work (nn.Linear weights, activation functions, variadic `.transpose()`), but the right fix is MLX's graph API once it stabilizes.
- **egglog has no shape awareness** — single `Tensor` type, no shape or dtype. Shape-aware rewrites need a structured type encoding or separate inference pass. This also blocks ILP extraction (e-class internals not exposed by the Python bindings) and primitive-subst rules in the e-graph (handled as a graph pass instead).
- **Missing rules:** `mx.async_eval` placement, `vmap` exploitation, `mx.quantize` weight-only quantization. (`normed_matmul` — `(x@W)*rms_scalar → rms_norm(x)@W` — landed in 0.2.2 as an opt-in class.)
- **TensorOps generates valid MSL** (`simdgroup_matrix`, M2+) but is slower than MLX's native GEMM and disabled by default (`enable_tensorops=False`). Real speedup needs MPP `cooperative_tensor` (M5/A19+, WWDC 2025 #315).

---

## References

- [Mirage: A Multi-Level Superoptimizer for Tensor Programs](https://arxiv.org/abs/2405.05751) — Wu et al., OSDI 2025
- [Equality Saturation for Tensor Graph Superoptimization](https://arxiv.org/abs/2101.01332) — Yang et al., MLSys 2021
- [egg: Fast and Extensible Equality Saturation](https://doi.org/10.1145/3434304) — Willsey et al., POPL 2021
- [BALLS](https://github.com/Philogy/balls) — Philogy's R/W-dependency-tracked scheduling discipline (originally for EVM stack scheduling; adapted here for GPU memory regions)
- [Dissecting the Apple M1 GPU, part III](https://alyssarosenzweig.ca/blog/asahi-gpu-part-3.html) — Rosenzweig, occupancy and register pressure model
- [Get started with MLX for Apple silicon](https://developer.apple.com/videos/play/wwdc2025/315/) — Apple WWDC 2025 #315

---

## License

Apache 2.0 — see [LICENSE.md](LICENSE.md).
