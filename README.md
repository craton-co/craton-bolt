# Craton Bolt

[![crates.io](https://img.shields.io/crates/v/craton-bolt.svg)](https://crates.io/crates/craton-bolt) [![docs.rs](https://docs.rs/craton-bolt/badge.svg)](https://docs.rs/craton-bolt) [![CI](https://github.com/craton-co/craton-bolt/actions/workflows/ci.yml/badge.svg)](https://github.com/craton-co/craton-bolt/actions/workflows/ci.yml) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![MSRV: 1.74](https://img.shields.io/badge/MSRV-1.74-orange.svg)](Cargo.toml)

> JIT-compiled GPU SQL engine. SQL strings go in, NVIDIA PTX comes out at runtime, the GPU does the rest.

Craton Bolt is a SQL execution engine written in Rust that compiles each query into a fresh NVIDIA PTX kernel at runtime, loads it via the CUDA driver, and runs it on the GPU. There is no C++ shim, no precompiled kernel library, and no FFI to a third-party query engine. The full pipeline — parse → plan → codegen → launch — is pure Rust on top of the raw CUDA driver API.

The project's two distinguishing ideas:

1. **Kernel fusion via runtime PTX.** Most GPU dataframe engines (RAPIDS / cuDF) chain precompiled kernels and bounce intermediates through global memory. Craton Bolt emits a single PTX kernel per query, keeping the entire fused expression tree in registers. Comparable in spirit to what Polars / DataFusion do for the CPU via codegen and Arrow-native vectorisation, but targeting the GPU.
2. **Borrow-checked GPU memory ("CUDA-Oxide").** GPU allocations are typed handles (`GpuVec<T>`), borrowed as `GpuView<'a, T>` for read-only access and `GpuViewMut<'a, T>` (a `!Sync`, `!Copy` exclusive handle) for write access. Kernel launches require those borrows, so use-after-free, double-free, and mutable / shared aliasing across kernel boundaries are rejected at compile time. The host-side type system makes the same guarantees Rust already makes for CPU memory.

## Status

**Active development.** The crate compiles clean on Windows MSVC and Linux against a CUDA Toolkit ≥ 12. It targets `sm_70` (Volta) and newer. End-to-end pipelines for projection, filter, scalar aggregate, GROUP BY (incl. multi-column, float keys, sentinel-collision-safe), and join-free SQL are implemented, along with string predicates (=, !=, IN over dictionary-encoded literals) and a small set of host-callable string operations (`UPPER`, `LOWER`, `LENGTH`, `CONCAT`) reachable only via the Rust `string_ops` API, not yet via SQL. Production use is **not** recommended — the public API is unstable and large swaths still want hardening, benchmarking, and battle-testing.

See [`docs/SQL_REFERENCE.md`](docs/SQL_REFERENCE.md) for the exact supported subset.

## What's in the box

| Layer            | What it does                                                                  |
|------------------|-------------------------------------------------------------------------------|
| `src/cuda/`      | Raw CUDA driver FFI, Arrow-aligned device buffers, borrow-checked `GpuVec`, host-side dictionary encoders (i32 and i64 indices). |
| `src/plan/`      | Logical plan AST, lazy `DataFrame` builder, SQL frontend (sqlparser), physical-plan lowering with SSA-shaped IR, string-literal predicate rewriting. |
| `src/jit/`       | PTX codegen — projection kernels, predicate-only kernels, scalar reductions, GROUP BY hash kernels (sentinel-based and valid-flag), float-atomic MIN/MAX via CAS loop, single-pass and multi-pass prefix scan, gather. The NVRTC-equivalent driver path (`cuModuleLoadData`) is also here. |
| `src/exec/`      | Top-level engine; per-shape executors (scalar / GROUP BY / pre-projection / pre+GROUP BY / wide keys / sentinel-free); GPU and host filter compaction; dictionary registry; host-side aggregate fallbacks for Bool / Utf8. |

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
git clone <repo-url> craton-bolt
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

First measured run on a CPU-only host (no GPU available; `engine_execute`
was skipped). All numbers are criterion's middle estimate over 100 samples
of three queries against a 1,000,000-row synthetic dataset:

| Stage              | proj        | arith       | filtered    |
|--------------------|-------------|-------------|-------------|
| `plan`             |   13.7 µs   |    9.4 µs   |   11.9 µs   |
| `lower`            |  429.5 ns   |  563.2 ns   |  621.0 ns   |
| `ptx_gen`          |    7.0 µs   |    9.5 µs   |   11.2 µs   |
| **CPU overhead**   | **~21 µs**  | **~19 µs**  | **~23 µs**  |
| `polars` (multi-threaded) | 6.5 µs † | 2.94 ms | 1.60 ms |
| `cpu_reference` (single-threaded baseline) | — | 3.51 ms | — |
| `engine_execute` (GPU) | skipped | skipped | skipped |

† Polars's `proj` is essentially a metadata clone of the existing column —
not a fair compute comparison.

**Takeaways:**

- Parse + plan + codegen is **sub-25-µs** per query. JIT-compiling at every
  query is not a meaningful overhead.
- The CPU reference single-threaded loop on 1M float multiplies takes
  ~3.5 ms; that's the floor a GPU implementation needs to beat at this
  problem size. We expect GPU break-even somewhere around 100k rows for
  arithmetic queries; below that the launch + h2d/d2h round-trip dominates.
- Polars on `filtered` (1.6 ms, multi-threaded) is the CPU number to beat.

The bench file is `benches/query_benchmarks.rs`. Run with:

```bash
cargo bench                              # CPU-only (plan, codegen, CPU ref, Polars)
BOLT_BENCH_GPU=1 cargo bench          # add the GPU engine path
```

For the methodology and a full per-bench breakdown, see
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). All non-trivial changes should come with tests; the build machine doesn't have a GPU, so PTX-shape assertions (the "compile and search the emitted string") are an acceptable substitute for the JIT layer, and `#[ignore]`-gated tests are the convention for live-GPU integration. See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for the full workflow.

## Project layout

```
craton-bolt/
├── Cargo.toml
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CHANGELOG.md
├── docs/
│   ├── ARCHITECTURE.md       # the layer cake and module map
│   ├── JIT_PIPELINE.md       # SQL → PTX, step by step
│   ├── SQL_REFERENCE.md      # what works, what doesn't
│   └── DEVELOPMENT.md        # building, testing, benchmarking
├── src/
│   ├── lib.rs                # crate root, public re-exports
│   ├── error.rs              # BoltError + BoltResult
│   ├── cuda/                 # driver FFI, GpuVec, dictionaries
│   ├── plan/                 # AST, DataFrame, SQL frontend, physical IR
│   ├── jit/                  # PTX codegen + module loader
│   └── exec/                 # per-shape executors + top-level Engine
├── tests/
│   ├── memory_tests.rs       # CUDA-Oxide compile-fail proofs
│   └── e2e_tests.rs          # parser/plan/PTX-shape + ignored live-GPU
└── benches/
    └── query_benchmarks.rs   # criterion + Polars head-to-head
```

## Security

Security issues should be reported privately per the policy in [SECURITY.md](SECURITY.md). Do not file public GitHub issues for vulnerabilities.

## Releases

Version history and per-release notes live in [`CHANGELOG.md`](CHANGELOG.md). Craton Bolt follows [Semantic Versioning](https://semver.org/); pre-1.0 the public API is unstable and minor bumps may break it.

## Acknowledgements

Craton Bolt stands on the shoulders of [`arrow-rs`](https://github.com/apache/arrow-rs) (columnar memory format), [`sqlparser-rs`](https://github.com/apache/datafusion-sqlparser-rs) (SQL frontend), and NVIDIA's CUDA driver (everything below `cuModuleLoadData`).

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

By contributing to Craton Bolt, you agree that your contributions will be
licensed under the same Apache-2.0 license. See [`CONTRIBUTING.md`](CONTRIBUTING.md)
for details.
