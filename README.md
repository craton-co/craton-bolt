# Craton Bolt

[![crates.io](https://img.shields.io/crates/v/craton-bolt.svg)](https://crates.io/crates/craton-bolt) [![docs.rs](https://docs.rs/craton-bolt/badge.svg)](https://docs.rs/craton-bolt) [![CI](https://github.com/craton-co/craton-bolt/actions/workflows/ci.yml/badge.svg)](https://github.com/craton-co/craton-bolt/actions/workflows/ci.yml) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![MSRV: 1.74](https://img.shields.io/badge/MSRV-1.74-orange.svg)](Cargo.toml)

> JIT-compiled GPU SQL engine. SQL strings go in, NVIDIA PTX comes out at runtime, the GPU does the rest.

Craton Bolt is a SQL execution engine written in Rust that compiles each query into a fresh NVIDIA PTX kernel at runtime, loads it via the CUDA driver, and runs it on the GPU. There is no C++ shim, no precompiled kernel library, and no FFI to a third-party query engine. The full pipeline — parse → plan → codegen → launch — is pure Rust on top of the raw CUDA driver API.

The project's two distinguishing ideas:

1. **Kernel fusion via runtime PTX.** Most GPU dataframe engines (RAPIDS / cuDF) chain precompiled kernels and bounce intermediates through global memory. Craton Bolt emits a single PTX kernel per query, keeping the entire fused expression tree in registers. Comparable in spirit to what Polars / DataFusion do for the CPU via codegen and Arrow-native vectorisation, but targeting the GPU.
2. **Borrow-checked GPU memory ("CUDA-Oxide").** GPU allocations are typed handles (`GpuVec<T>`), borrowed as `GpuView<'a, T>` for read-only access and `GpuViewMut<'a, T>` (a `!Sync`, `!Copy` exclusive handle) for write access. Kernel launches require those borrows, so use-after-free, double-free, and mutable / shared aliasing across kernel boundaries are rejected at compile time. The host-side type system makes the same guarantees Rust already makes for CPU memory.

## Status

**Active development — v0.7.0.** The crate compiles clean on Windows MSVC and Linux against a CUDA Toolkit ≥ 12. It targets `sm_70` (Volta) and newer. End-to-end pipelines for projection, filter, scalar aggregate, GROUP BY (multi-tier shared-memory + hash-partitioned), joins (`INNER` / `LEFT [OUTER]` / `RIGHT [OUTER]` / `FULL [OUTER]` on GPU when the shape qualifies, host-side hash join otherwise; `CROSS` on GPU or host; plus small-cardinality non-equi joins via a host nested-loop fallback), `DISTINCT`, `ORDER BY` (GPU bitonic sort integrated, plus an env-gated GPU radix path; host `lexsort` fallback), `LIMIT`, `HAVING`, `UNION [ALL]`, `EXCEPT [ALL]`, and `INTERSECT [ALL]` are implemented. The frontend also accepts CTEs (`WITH`, including `WITH RECURSIVE` — linear, non-linear, and mutual), derived tables and `LATERAL` subqueries in `FROM`, uncorrelated subqueries plus a single correlated `WHERE` subquery (`EXISTS` / `NOT EXISTS` / scalar), `VALUES` as a row source, the `generate_series` table-valued function, `DISTINCT ON`, host-side window functions with named `WINDOW` clauses and `QUALIFY`, super-aggregates (`ROLLUP` / `CUBE` / `GROUPING SETS`), and query-clause sugar (`FETCH` / `TOP` → `LIMIT`, `FOR UPDATE` no-op, `PREWHERE` → `WHERE`). The scalar surface includes `IN`, `BETWEEN`, `CASE`, `CAST`, `COALESCE` / `NULLIF`, and `LIKE` (numeric/Bool results lower to GPU). `Decimal128` has full GPU arithmetic (`+`, `-`, `*`, `/`) and comparisons, with scalar **and** grouped GPU `SUM` / `MIN` / `MAX`; `Date32` / `Timestamp` arithmetic (Date−Date, Timestamp−Timestamp, Day-INTERVAL) lowers to GPU. String predicates (`=`, `!=`, `IN` over dictionary-encoded literals) plus GPU `UPPER` / `LOWER` / `LENGTH` / `LIKE` and host-side `CONCAT` / `SUBSTRING` / `TRIM` are available. [`docs/SQL_REFERENCE.md`](docs/SQL_REFERENCE.md) is the authoritative list of the supported SQL surface. Production use is **not** recommended — the public API is unstable pre-1.0.

> **CI runs no GPU code.** The CI pipeline builds, tests, lints, and runs
> `cargo deny` using the `cuda-stub` feature only — it exercises **0 GPU
> code paths** because no GPU runner exists. The `#[ignore]`-gated CUDA
> integration tests are dark in CI; GPU correctness is validated separately
> on developer/maintainer hardware (see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
> for the verification harness). Treat CI green as "host logic + codegen
> shape are sound," not "GPU execution is verified."

> **Limitations / not yet production-ready.** See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)
> for the consolidated list of requirements, pre-1.0 caveats, and known
> semantic gaps before you depend on Craton Bolt.

See [`docs/SQL_REFERENCE.md`](docs/SQL_REFERENCE.md) for the exact supported subset.

## What's in the box

| Layer            | What it does                                                                  |
|------------------|-------------------------------------------------------------------------------|
| `src/cuda/`      | Raw CUDA driver FFI, Arrow-aligned device buffers, borrow-checked `GpuVec`, host-side dictionary encoders (i32 and i64 indices). |
| `src/plan/`      | Logical plan AST, lazy `DataFrame` builder, SQL frontend (sqlparser), physical-plan lowering with SSA-shaped IR, string-literal predicate rewriting. |
| `src/jit/`       | PTX codegen — projection kernels, predicate-only kernels, scalar reductions, GROUP BY hash kernels (sentinel-based and valid-flag), float-atomic MIN/MAX via CAS loop, single-pass and multi-pass prefix scan, gather, hash-join build/probe kernels, and bitonic + radix sort kernels. The NVRTC-equivalent driver path (`cuModuleLoadData`) and the `KernelSpec`-keyed module cache are also here. |
| `src/exec/`      | Top-level engine; per-shape executors (scalar / GROUP BY / pre-projection / pre+GROUP BY / wide keys / sentinel-free); GPU and host hash-join executors; GPU and host ORDER BY; GPU and host filter compaction; dictionary registry; host-side aggregate fallbacks for Bool / Utf8. |

## Quick start

### Requirements

- Rust 1.74 or newer (2021 edition).
- An NVIDIA CUDA Toolkit ≥ 12, with `cuda.lib` (Windows) / `libcuda.so` (Linux) on the linker path.
- An NVIDIA GPU with compute capability ≥ 7.0 (Volta or newer) and a driver matching the toolkit.

`cargo check` and `cargo build --lib` work on a host without CUDA installed (everything type-checks). `cargo test` and `cargo bench` require the linker to find `cuda.lib`; the ignored integration tests further require an actual GPU.

### Platform support

- **Linux (x86_64):** supported.
- **Windows (x86_64 MSVC):** supported.
- **macOS (any arch):** NOT supported — Apple ended CUDA support in 2019. `cargo check --features cuda-stub` works for type-checking only.
- **ARM (aarch64-linux):** in theory supported by Jetson; not tested.

### Build

```bash
git clone https://github.com/craton-co/craton-bolt
cd craton-bolt
cargo build --release
```

Hosts without a CUDA toolkit can type-check the crate with `cargo build --no-default-features --features cuda-stub` — useful for CI and `docs.rs` builds.

### Run a query

```rust
use std::sync::Arc;
use arrow_array::{Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use craton_bolt::Engine;

let mut engine = Engine::new()?;

// Register a table.
let region: Int32Array = (0..1_000_000_i32).map(|i| i % 4).collect();
let price:  Float64Array = (0..1_000_000_u64).map(|i| i as f64).collect();
let tax:    Float64Array = (0..1_000_000_u64).map(|_| 0.0825_f64).collect();
let schema = Arc::new(Schema::new(vec![
    Field::new("region_id", DataType::Int32,   false),
    Field::new("price",     DataType::Float64, false),
    Field::new("tax",       DataType::Float64, false),
]));
let batch = RecordBatch::try_new(schema, vec![Arc::new(region), Arc::new(price), Arc::new(tax)])?;
engine.register_table("sales", batch)?;

// Execute.
let handle = engine.sql("SELECT price * tax FROM sales WHERE region_id = 1")?;
println!("got {} rows", handle.num_rows());
```

Behind the scenes for that single line: the SQL is parsed; column references and string literals are rewritten as needed; the logical plan is lowered to a `KernelSpec` of SSA-shaped ops; the codegen emits a fresh PTX module; the CUDA driver assembles it to SASS; the kernel launches one thread per row with predicate gating; a GPU-side prefix-scan + gather compacts the output; the surviving rows download into an Arrow `RecordBatch`.

## Architecture overview

```
                ┌────────── SQL string ──────────┐
                │                                │
                ▼                                ▼
        sqlparser (3rd-party)            DataFrame builder
                │                                │
                └─────────────┬──────────────────┘
                              ▼
                       LogicalPlan AST
                              │
                              │  string-literal rewrite
                              │  (col = 'X' → __idx_col = i32(idx))
                              ▼
                       LogicalPlan
                              │
                              │  physical-plan lowering
                              │  (resolve columns to ordinals, expressions to Op IR)
                              ▼
                       PhysicalPlan
                              │
                              ├── Projection { KernelSpec, ... }
                              └── Aggregate  { pre?, AggregateSpec }
                              │
                              │  per-shape executor selection
                              ▼
                ┌──────────────────────────────────────────────┐
                │  PTX codegen (per kernel)                     │
                │   - projection kernel                         │
                │   - predicate-only kernel                     │
                │   - per-block reduction (SUM / MIN / MAX / …) │
                │   - GROUP BY hash insert + per-aggregate      │
                │   - float MIN/MAX via atom.cas                │
                │   - prefix scan + gather                      │
                └──────────────────────────────────────────────┘
                              │
                              │  CudaModule::from_ptx (calls cuModuleLoadData)
                              ▼
                       cuLaunchKernel
                              │
                              │  download outputs → arrow_array
                              ▼
                        RecordBatch
```

For the long form, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/JIT_PIPELINE.md`](docs/JIT_PIPELINE.md).

## Performance

All GPU numbers below were measured on an **NVIDIA GeForce RTX 2060**,
CUDA 12.6, verified end-to-end equivalent against Polars 0.42 and DuckDB 1.2
before timing. Full methodology and per-bench breakdown: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

**CPU-side overhead** (plan + lower + codegen, no GPU needed) is **sub-25 µs** per
query regardless of dataset size — JIT-compiling every query has negligible cost.

The tables below are illustrative; see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
for the canonical numbers and methodology.

### Heavy-arithmetic OLAP (50 M rows, fused multi-operator, RTX 2060)

| Query | Polars (CPU MT) | Craton Bolt (GPU) | Speedup |
|---|---|---|---|
| 11-op arithmetic chain (50 M rows) | 4.05 s | **124.8 ms** | **32.4×** |
| Filter + 4-op arithmetic (50 M rows) | 369 ms | **41.8 ms** | **8.8×** |

### h2o.ai db-benchmark GROUP BY subset (10 M rows, RTX 2060)

| Query | DuckDB | Polars | Craton Bolt | Notes |
|---|---|---|---|---|
| q1 low-card SUM (100 groups) | 6.9 ms | 19.0 ms | **51.4 ms** | DuckDB wins |
| q2 med-card 2-SUM (10 K groups) | 46.4 ms | 99.4 ms | 384 ms | DuckDB wins |
| q3 two-key SUM (~1 M groups) | 498 ms | 385 ms | **219 ms** ⭐ | Craton Bolt fastest |
| q4 low-card 3-AVG (100 groups) | 12.9 ms | 97.0 ms | **70.5 ms** | DuckDB wins |
| q5 high-card SUM (1 M groups) | 623 ms | 358 ms | **237 ms** ⭐ | Craton Bolt fastest |

Craton Bolt wins outright on the two highest-cardinality workloads (q3, q5) where
GPU-parallel hash-partitioning outpaces CPU per-core hash tables. CPU-native engines
win at low cardinality (q1, q4) where their per-thread L1-resident tables beat GPU
atomic contention. See [`docs/BENCHMARKS.md §honest read`](docs/BENCHMARKS.md#the-honest-read)
for the full analysis.

```bash
cargo bench                              # CPU-only (plan, codegen, CPU ref, Polars)
BOLT_BENCH_GPU=1 cargo bench            # add the GPU engine path
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). All non-trivial changes should come with tests; the build machine doesn't have a GPU, so PTX-shape assertions (the "compile and search the emitted string") are an acceptable substitute for the JIT layer, and `#[ignore]`-gated tests are the convention for live-GPU integration. See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for the full workflow.

## Project layout

```
craton-bolt/
├── Cargo.toml
├── README.md
├── CONTRIBUTING.md
├── RELEASING.md              # maintainer release checklist
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CHANGELOG.md
├── ROADMAP.md
├── docs/
│   ├── INSTALL.md            # prerequisites, build configs, troubleshooting
│   ├── USER_GUIDE.md         # end-to-end usage walkthrough
│   ├── ARCHITECTURE.md       # the layer cake and module map
│   ├── JIT_PIPELINE.md       # SQL → PTX, step by step
│   ├── SQL_REFERENCE.md      # what works, what doesn't
│   ├── API_SURFACE.md        # public API reference
│   ├── ENV_VARS.md           # environment variables and tuning knobs
│   ├── DEVELOPMENT.md        # building, testing, benchmarking
│   ├── FAQ.md                # frequently asked questions
│   ├── BENCHMARKS.md         # measured numbers and methodology
│   ├── COMPETITIVE_BENCHMARKING.md  # how to run fair comparisons
│   ├── GROUPBY_PERF.md       # GROUP BY kernel design and analysis
│   ├── LIMITATIONS.md        # requirements, pre-1.0 caveats, known gaps
│   ├── MIGRATION_GUIDE.md    # upgrading across breaking changes
│   └── PATH_TO_1.0.md        # detailed 1.0 milestone plan
├── src/
│   ├── lib.rs                # crate root, public re-exports
│   ├── error.rs              # BoltError + BoltResult
│   ├── cuda/                 # driver FFI, GpuVec, dictionaries
│   ├── plan/                 # AST, DataFrame, SQL frontend, physical IR
│   ├── jit/                  # PTX codegen + module loader
│   └── exec/                 # per-shape executors + top-level Engine
├── tests/                   # integration tests: parser, optimizer, aggregates,
│                            #   joins, sorts, GROUP BY paths, string fns, casts,
│                            #   datetime/decimal types, set ops, PTX golden
│                            #   snapshots, proptest fuzzing, DuckDB cross-checks
└── benches/
    ├── query_benchmarks.rs   # criterion + Polars + CPU-ref (small dataset)
    └── olap_benchmarks.rs    # h2o.ai groupby vs Polars vs DuckDB
```

## Security

Security issues should be reported privately per the policy in [SECURITY.md](SECURITY.md). Do not file public GitHub issues for vulnerabilities.

## Releases

Version history and per-release notes live in [`CHANGELOG.md`](CHANGELOG.md). Craton Bolt follows [Semantic Versioning](https://semver.org/); pre-1.0 the public API is unstable and minor bumps may break it.

## Acknowledgements

Craton Bolt stands on the shoulders of [`arrow-rs`](https://github.com/apache/arrow-rs) (columnar memory format), [`sqlparser-rs`](https://github.com/apache/datafusion-sqlparser-rs) (SQL frontend), and NVIDIA's CUDA driver (everything below `cuModuleLoadData`).

*Craton* and *Bolt* are trademarks of Craton Software Company.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

By contributing to Craton Bolt, you agree that your contributions will be
licensed under the same Apache-2.0 license. See [`CONTRIBUTING.md`](CONTRIBUTING.md)
for details.
