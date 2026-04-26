# PHEW

**Probably Hardly Ever Works**

A search-based superoptimizer for MLX on Apple Silicon. Point it at working MLX code; it finds a measurably faster equivalent and proves the equivalence before handing it back.

Single success metric: `optimized_ms / baseline_ms` with proven equivalence. If it can't beat the baseline across all problem sizes with verified correctness, it returns your original code unchanged.

---

## What's built

### The full pipeline runs end-to-end

```
phew run examples/rms_norm.py
```

Produces:

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

Seven steps run automatically:

1. **Baseline benchmark** — auto-converges (increases iterations until σ/μ < 5%)
2. **Bottleneck classification** — memory-bound / compute-bound / occupancy-limited / launch-overhead; prunes the rule set
3. **Graph-level passes** — `mx.compile` wrapping, `mx.fast.*` primitive substitution, M5 TensorOps detection
4. **E-graph saturation** — algebraic, layout, fusion, precision, quantization, and compile-boundary rules via [egglog](https://github.com/egraphs-good/egglog)
5. **Code emission** — generates clean, copy-pasteable MLX Python
6. **Multi-size benchmark** — validates the speedup holds across small/typical/large problem sizes, not just one
7. **Equivalence verification** — probabilistic testing (Schwartz–Zippel-inspired): 5 seeds × 3 sizes × 4 edge variants (random, zeros, small-scale, large-scale). Candidate is dropped if any check fails.

### IR (`phew/ir/`)

A two-level μGraph (Mirage-derived). Kernel-level ops cover the full MLX surface: matmul, reduce, elementwise, transpose, reshape, cast, concat, split, broadcast, `mx.compile`, `mx.async_eval`, vmap, and all `mx.fast.*` primitives (rms_norm, layer_norm, rope, scaled_dot_product_attention, metal_kernel). Per-node R/W dependency tracking using the BALLS discipline (`device_mem`, `threadgroup_mem`, `register`, `control_flow`).

The importer (`phew/ir/importer.py`) builds a graph by monkey-patching `mlx.core` ops during a trace pass.

### Rules (`phew/rules/`)

Three graph-level passes that run before e-graph saturation:

- **Primitive substitution** — structural pattern matching for RMS norm → `mx.fast.rms_norm`, SDPA → `mx.fast.scaled_dot_product_attention`
- **Compile boundaries** — wraps eligible subgraphs in `mx.compile`
- **TensorOps** — detects M5/A19 hardware (`applegpu_g17+`) and substitutes eligible matmuls with Metal Performance Primitives kernels

### E-graph saturation (`phew/egraph/`)

Equality saturation via egglog (Python bindings). Rule sets:

| Rule set | What it does |
|---|---|
| `algebraic` | Commutativity, associativity, matmul reassociation |
| `layout` | Transpose-through-matmul: `(AB)ᵀ ↔ BᵀAᵀ` |
| `precision` | Double-cast elimination; fp32→fp16/bf16 (opt-in) |
| `quantization` | `matmul(x,w) → quantized_matmul(x,w,bits=4)` (opt-in) |
| `compile_boundaries` | `matmul(a,b) → compiled(matmul(a,b))` |
| `fusion` | Elementwise chains (placeholder; graph-level pass handles most cases) |

Extraction uses greedy cost minimization by default; an ILP fallback (scipy) handles joint multi-pattern optimization.

### Cost model (`phew/cost/`)

Static cost model used for pruning only — final ranking is always on-device measurement. Bytes moved weighted by memory hierarchy level (register=1, threadgroup=4, L1=8, device=32). Occupancy proxy from Rosenzweig's M1 GPU model: 0–112 registers = full occupancy, 112–256 = linear falloff in steps of 64 threads, >256 = spill. Hard-rejects: threadgroup > 1024, threadgroup memory > device limit.

### Verification (`phew/verify/`)

Four substitution classes with explicit tolerances:

| Class | atol | rtol | Opt-in required |
|---|---|---|---|
| fp32 → fp32 | 1e-5 | 1e-5 | No |
| fp32 → fp16 | 1e-3 | 1e-2 | Yes |
| fp32 → bf16 | 1e-2 | 1e-2 | Yes |
| quantized (4-bit) | 1e-2 | 5e-2 | Yes |

### Profiling (`phew/trace/`)

`TraceCapture` wraps `mx.metal.start_capture` / `stop_capture`. The bottleneck classifier maps aggregated kernel stats to one of four classes that gate which rule sets are activated.

### Benchmark harness (`phew/bench/`)

Auto-converging benchmark: doubles iteration count until σ/μ < 5%. Speedup inside ±3% noise band is not reported as significant. Multi-size convergence check requires all problem sizes to beat baseline by >3%.

### CLI (`phew/cli/`)

```
phew run    input.py [-o out.py]    # optimize and emit
phew bench  input.py                # baseline benchmark only
phew trace  input.py                # capture Metal GPU trace
phew verify baseline.py opt.py      # verify equivalence standalone
```

### Code emission (`phew/emit/`)

`MLXCodegen` generates clean MLX Python from the optimized graph. `KernelParamSearch` enumerates threadgroup/tile/vector-width/unroll combinations for Phase-2 kernel search.

---

## Install

```bash
# Requires macOS 13.3+ on Apple Silicon, Python 3.12+
uv pip install --no-config -e ".[dev]"
```

To install `phew` globally so it's available in your terminal:

```bash
uv tool install --no-config -e .
```

Run the tests:

```bash
uv run --no-config pytest tests/ -v
```

---

## Usage

```python
from phew import Optimizer
from phew.verify import SubstitutionClass

def rms_norm(x, weight):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5) * weight

def input_factory(size, seed):
    mx.random.seed(seed)
    sizes = {"small": 256, "typical": 1024, "large": 4096}
    n = sizes[size]
    return [mx.random.normal((n, 2048)), mx.ones((2048,))], {}

opt = Optimizer(
    fn=rms_norm,
    input_factory=input_factory,
    # Enable precision reduction if acceptable for your use case:
    # enabled_subst_classes={SubstitutionClass.fp32_to_fp16},
)
result = opt.run()
print(result.output_source)
```

---

## Roadmap

### Correctness gaps

- **IR round-trip from egglog** — `_egglog_to_graph` in `egraph/extraction.py` currently returns the original graph unchanged. The full round-trip (egglog expression → phew Graph) isn't implemented, so e-graph rewrites don't yet affect the emitted code. Graph-level passes (compile, primitive subst, TensorOps) do apply correctly.
- **Primitive substitution correctness** — The RMS norm pattern matcher in `rules/primitive_subst.py` is structural only and currently produces incorrect results for some input shapes. Verification catches and drops these, but the pattern needs to be fixed.
- **Layer norm pattern** — `rules/primitive_subst.py` has a stub; the layer norm matcher always returns False.

### Phase-2 kernel search

The `KernelParamSearch` in `emit/metal_kernel.py` enumerates parameter combinations (threadgroup size, tile dimensions, SIMD width, vector width, unroll factor) but the kernel source template doesn't yet wire in the `THREADGROUP`, `VW`, and `UNROLL` template variables. Phase-2 runs only when the hottest kernel is >30% of GPU time and not replaceable by a `fast.*` primitive.

**External kernel parameter sweep** — currently Phase-2 only applies to kernels PHEW emits itself. A natural extension is to accept a user-supplied Metal kernel source and sweep its compile-time constants (e.g., `ITERS`, `HC` in a Sinkhorn kernel), using PHEW's benchmark harness to find the best configuration.

### Tracer fidelity

`ir/importer.py` monkey-patches `mlx.core` ops at trace time. This is fragile for functions with:
- Control flow (conditionals, loops over tensors)
- In-place updates
- Custom Metal kernels (`mx.fast.metal_kernel`)
- Nested `mx.compile` scopes

A more robust approach is to use MLX's computation graph API directly once it stabilises.

### egglog shape awareness

The egglog type encoding in `egraph/rules_egglog.py` uses a single `Tensor` type without shape or dtype. Shape-aware rewrites (layout rules, broadcast elimination, reshape-through-matmul) need shape information propagated into the e-graph, which requires a more structured type encoding or a separate shape-inference pass.

### Rules not yet implemented

- Elementwise fusion chains (the `fusion` rule set is a placeholder)
- `mx.async_eval` placement for CPU-overlapping loops
- `vmap` exploitation
- `mx.quantize` integration for weight-only quantization

### TensorOps

The Metal kernel source in `rules/tensorops.py` is a placeholder. The real implementation needs the MPP `cooperative_tensor` API from Metal Performance Primitives (available on M5/A19+, WWDC 2025 #315).

### M5 Hardware

TensorOps detection uses architecture string parsing (`applegpu_g17+`). The current test environment is `applegpu_g16s` (M4). Full TensorOps testing requires M5/A19 hardware.

---

## Design references

```bibtex
@inproceedings{mirage2025,
  title     = {Mirage: A Multi-Level Superoptimizer for Tensor Programs},
  author    = {Wu, Mengdi and Yao, Zhen and Chen, Jian and Liu, Zhijian},
  booktitle = {19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 2025)},
  year      = {2025},
  note      = {arXiv:2405.05751}
}

@inproceedings{tensat2021,
  title     = {Equality Saturation for Tensor Graph Superoptimization},
  author    = {Yang, Yichen and Phothilimthana, Phitchaya Mangpo and Hong, Yisu Remy and Murthy, Madhura and Moon, Shishir G. and Steinhardt, Jacob},
  booktitle = {Proceedings of Machine Learning and Systems (MLSys 2021)},
  year      = {2021},
  note      = {arXiv:2101.01332}
}

@inproceedings{egg2021,
  title     = {egg: Fast and Extensible Equality Saturation},
  author    = {Willsey, Max and Nandi, Chandrakana and Wang, Yisu Remy and Flatt, Oliver and Tatlock, Zachary and Panchekha, Pavel},
  booktitle = {Proceedings of the ACM on Programming Languages (POPL 2021)},
  year      = {2021}
}
```

- BALLS (Philogy) — R/W-dependency-tracked scheduling discipline
- Rosenzweig, *Dissecting the Apple M1 GPU* — occupancy and register pressure model
- Zakharyo 2025 — M5 TensorOps tile constraints
- Apple WWDC 2025 #315 — MLX optimization guidance
