# Craton Bolt — Project Presentation

---

## Slide 1 · What Is Craton Bolt?

> A JIT-compiled GPU SQL engine in Rust. SQL strings go in, NVIDIA PTX comes out at runtime, the GPU does the rest.

- **What**: A SQL execution engine that compiles *each* query into a fresh NVIDIA PTX kernel at runtime, loads it via the CUDA driver, and runs it on the GPU.
- **Language**: Rust (2021 edition, MSRV 1.74) on top of the raw CUDA driver API — no C++ shim, no precompiled kernel library, no third-party query-engine FFI.
- **Pipeline**: parse → plan → codegen → launch, end to end in pure Rust.
- **Status**: `0.7.0`, active development, pre-1.0 (public API unstable). Compiles clean on Windows MSVC and Linux against CUDA Toolkit ≥ 12; targets `sm_70` (Volta) and newer.

---

## Slide 2 · The Two Distinguishing Ideas

**1 · Kernel fusion via runtime PTX.**
Most GPU dataframe engines (RAPIDS / cuDF) chain *precompiled* kernels and bounce
intermediates through global memory between each one. Craton Bolt emits a **single
PTX kernel per query**, keeping the entire fused expression tree in registers —
comparable in spirit to what Polars / DataFusion do for the CPU via codegen and
Arrow-native vectorisation, but targeting the GPU.

**2 · Borrow-checked GPU memory ("CUDA-Oxide").**
GPU allocations are typed handles (`GpuVec<T>`), borrowed as `GpuView<'a, T>` for
read-only access and `GpuViewMut<'a, T>` (a `!Sync`, `!Copy` exclusive handle) for
writes. Kernel launches require those borrows, so **use-after-free, double-free, and
mutable/shared aliasing across kernel boundaries are rejected at compile time**. The
host-side type system makes the same guarantees Rust already makes for CPU memory.

---

## Slide 3 · The JIT Pipeline

```
                ┌────────── SQL string ──────────┐
                ▼                                 ▼
        sqlparser (3rd-party)            DataFrame builder
                └───────────────┬─────────────────┘
                                ▼
                         LogicalPlan AST
                                │  string-literal rewrite
                                │  (col = 'X' → __idx_col = i32(idx))
                                ▼
                         PhysicalPlan  (columns → ordinals, exprs → Op IR)
                                │  per-shape executor selection
                                ▼
                ┌──────────────────────────────────────────────┐
                │  PTX codegen (per kernel)                      │
                │   projection · predicate-only · reductions     │
                │   GROUP BY hash · float MIN/MAX via atom.cas   │
                │   prefix scan + gather · hash-join · sort      │
                └──────────────────────────────────────────────┘
                                │  CudaModule::from_ptx → cuModuleLoadData
                                ▼
                          cuLaunchKernel  →  download → Arrow RecordBatch
```

A `KernelSpec`-keyed LRU module cache (128-bit key) plus a self-invalidating on-disk
cache means repeated query *shapes* skip recompilation entirely.

---

## Slide 4 · What's In the Box

| Layer       | What it does                                                                 |
|-------------|------------------------------------------------------------------------------|
| `src/cuda/` | Raw CUDA driver FFI, Arrow-aligned device buffers, borrow-checked `GpuVec`, host-side dictionary encoders (i32 / i64 indices). |
| `src/plan/` | Logical plan AST, lazy `DataFrame` builder, SQL frontend (sqlparser), physical-plan lowering with SSA-shaped IR, string-literal predicate rewriting. |
| `src/jit/`  | PTX codegen — projection, predicate-only, scalar reductions, GROUP BY hash kernels, float-atomic MIN/MAX via CAS, prefix scan, gather, hash-join build/probe, bitonic + radix sort. Plus the `cuModuleLoadData` driver path and the module cache. |
| `src/exec/` | Top-level engine; per-shape executors (scalar / GROUP BY / pre-projection / wide keys / sentinel-free); GPU & host hash-join; GPU & host ORDER BY; filter compaction; dictionary registry; host aggregate fallbacks. |

---

## Slide 5 · SQL Surface

End-to-end GPU pipelines exist for **projection, filter, scalar aggregates, GROUP BY**
(multi-tier shared-memory + hash-partitioned), **joins** (`INNER` / `LEFT` / `RIGHT` /
`FULL [OUTER]` on the GPU when the shape qualifies, host hash-join otherwise; `CROSS`
on GPU or host; small-cardinality non-equi joins via host nested-loop), **`DISTINCT`**,
**`ORDER BY`** (GPU bitonic sort + env-gated GPU radix path; host `lexsort` fallback),
**`LIMIT`**, **`HAVING`**, and **`UNION` / `EXCEPT` / `INTERSECT [ALL]`**.

The frontend also accepts:

- CTEs (`WITH`, incl. `WITH RECURSIVE` — linear, non-linear, mutual)
- Derived tables and `LATERAL` subqueries in `FROM`
- Uncorrelated subqueries + one correlated `WHERE` subquery (`EXISTS` / scalar)
- `VALUES` row sources, the `generate_series` TVF, `DISTINCT ON`
- Host-side **window functions** with named `WINDOW` clauses and `QUALIFY`
- Super-aggregates: `ROLLUP` / `CUBE` / `GROUPING SETS`
- Clause sugar: `FETCH` / `TOP` → `LIMIT`, `PREWHERE` → `WHERE`, `FOR UPDATE` no-op

Scalar surface: `IN`, `BETWEEN`, `CASE`, `CAST`, `COALESCE` / `NULLIF`, `LIKE` / `ILIKE`.
`Decimal128` has full GPU arithmetic + comparisons and scalar **and** grouped GPU
`SUM` / `MIN` / `MAX`; `Date32` / `Timestamp` arithmetic lowers to the GPU. Equality and
`LIKE` over dictionary-encoded `Utf8` fold to **pure integer index-membership predicates**
on the GPU. (`docs/SQL_REFERENCE.md` is authoritative.)

---

## Slide 6 · Performance — Heavy-Arithmetic OLAP

Measured on an **NVIDIA GeForce RTX 2060**, CUDA 12.6, verified bit-equivalent against
Polars 0.42 and DuckDB 1.2 before timing. 50 M rows, fused multi-operator arithmetic —
the regime where GPU compute density pays off.

| Query                                | Polars (CPU MT) | Craton Bolt (GPU) | Speedup |
|--------------------------------------|-----------------|-------------------|---------|
| 11-op arithmetic chain (50 M rows)   | 4.05 s          | **124.8 ms**      | **32.4×** |
| Filter + 4-op arithmetic (50 M rows) | 369 ms          | **41.8 ms**       | **8.8×**  |

Against an honest single-thread CPU reference loop (1.06 s) the arithmetic chain is still
**8.5× faster** — the 32.4× figure is partly amplified by Polars' eager-binary
materialisation under chained `lit() * col()` expressions.

**CPU-side overhead** — plan + lower + codegen, no GPU needed — is **under 25 µs per
query** regardless of dataset size. JIT-compiling every single query is a viable
execution model: the codegen budget is negligible next to any real GPU launch + H2D/D2H
round trip.

---

## Slide 7 · Performance — h2o.ai GROUP BY (10 M rows)

Protocol-disciplined OLAP comparison; schema, query shapes and cardinalities match the
h2o.ai db-benchmark spec.

| Query                          | DuckDB | Polars | Craton Bolt | Notes                |
|--------------------------------|--------|--------|-------------|----------------------|
| q1 low-card SUM (100 groups)   | 6.9 ms | 19.0 ms| 51.4 ms     | DuckDB wins          |
| q2 med-card 2-SUM (10 K groups)| 46.4 ms| 99.4 ms| 384 ms      | DuckDB wins          |
| q3 two-key SUM (≈ 1 M groups)  | 498 ms | 385 ms | **219 ms** ⭐ | Craton Bolt fastest |
| q4 low-card 3-AVG (100 groups) | 12.9 ms| 97.0 ms| 70.5 ms     | DuckDB wins          |
| q5 high-card SUM (1 M groups)  | 623 ms | 358 ms | **237 ms** ⭐ | Craton Bolt fastest |

Craton Bolt wins outright on the two highest-cardinality workloads (q3, q5), where
GPU-parallel hash-partitioning outpaces CPU per-core hash tables. CPU-native engines win
at low cardinality (q1, q4) where their per-thread L1-resident tables beat GPU atomic
contention. This is the honest read — Craton Bolt is a *complement* to CPU engines, not a
universal replacement.

---

## Slide 8 · The Honest Caveats

This is a research-grade engine, **not** production-ready (pre-1.0, unstable API).

- **CI runs zero GPU code.** The pipeline builds, tests, lints, and runs `cargo deny`
  using the `cuda-stub` feature only — no GPU runner exists. The `#[ignore]`-gated CUDA
  integration tests are dark in CI; GPU correctness is validated on developer/maintainer
  hardware. Treat CI green as "host logic + codegen shape are sound," not "GPU execution
  is verified."
- **Non-dictionary device string path is host-validated only.** The `LIKE` matcher and
  the two-pass `UPPER` / `LOWER` / `CONCAT` / `SUBSTRING` / `TRIM` producers are byte-validated
  against the host path but **not enabled by default** — they sit behind the opt-in
  `BOLT_GPU_STRING` env var. The host path is the default correctness path.
- **Platform**: Linux x86_64 and Windows x86_64 MSVC supported; macOS not supported
  (Apple ended CUDA support in 2019); aarch64/Jetson untested.
- See `docs/LIMITATIONS.md` for the consolidated list of requirements, pre-1.0 caveats,
  and known semantic gaps.

---

## Slide 9 · Quick Start

```bash
git clone https://github.com/craton-co/craton-bolt
cd craton-bolt
cargo build --release
```

```rust
use craton_bolt::Engine;

let mut engine = Engine::new()?;
engine.register_table("sales", batch)?;            // an Arrow RecordBatch

let handle = engine.sql("SELECT price * tax FROM sales WHERE region_id = 1")?;
println!("got {} rows", handle.num_rows());
```

Behind that one line: the SQL is parsed, string literals rewritten, the logical plan
lowered to a `KernelSpec` of SSA-shaped ops, a fresh PTX module emitted, the CUDA driver
assembles it to SASS, the kernel launches one thread per row with predicate gating, a
GPU prefix-scan + gather compacts the output, and the surviving rows download into an
Arrow `RecordBatch`.

Hosts without a CUDA toolkit can type-check the crate:
`cargo build --no-default-features --features cuda-stub` (used for CI and docs.rs).

---

## Slide 10 · Testing & Verification Discipline

The build machine has no GPU, so the test strategy is layered:

- **PTX-shape assertions** — compile a query and search the emitted PTX string for the
  expected instructions. This is an accepted substitute for the JIT layer in CI.
- **`#[ignore]`-gated live-GPU integration tests** — run on real hardware only.
- **PTX golden snapshots**, **proptest fuzzing**, and **DuckDB cross-checks** assert that
  the host correctness path matches a reference engine bit-for-bit.

Coverage spans the parser, optimizer, aggregates, joins, sorts, GROUP BY paths, string
functions, casts, datetime/decimal types, and set ops.

---

## Slide 11 · Roadmap to 1.0

The detailed milestone plan lives in `docs/PATH_TO_1.0.md` and `ROADMAP.md`. The broad
strokes before a stable 1.0:

- Enable and default-on the GPU non-dictionary string path (currently `BOLT_GPU_STRING`-gated).
- Broaden GPU coverage for the correlated-subquery and window-function cases that
  currently fall back to the host.
- Stabilise the public API surface (`docs/API_SURFACE.md`) and freeze the `KernelSpec`
  / cache key formats.
- Stand up a real GPU CI runner so GPU execution — not just host logic and codegen
  shape — is verified on every change.

---

## Summary

| | |
|--|--|
| **What** | JIT-compiled GPU SQL engine in Rust |
| **How** | SQL → logical plan → SSA IR → fresh PTX kernel per query → CUDA driver → Arrow |
| **Distinguishing idea 1** | Kernel fusion: one PTX kernel per query, fused tree in registers |
| **Distinguishing idea 2** | Borrow-checked GPU memory — UAF / double-free / aliasing are compile errors |
| **Arithmetic win** | 32.4× vs Polars on a 50 M-row 11-op chain (8.5× vs honest CPU ref) |
| **GROUP BY win** | Beats DuckDB & Polars on the two highest-cardinality h2o.ai queries |
| **Codegen cost** | < 25 µs plan + lower + codegen per query, any dataset size |
| **Stack** | Pure Rust on the raw CUDA driver API — no C++ shim, no FFI to a 3rd-party engine |
| **Targets** | CUDA 12+, NVIDIA `sm_70`+ (Volta); Linux & Windows MSVC; `cuda-stub` type-checks anywhere |
| **Status** | v0.7.0, active development, pre-1.0 — API unstable, not production-ready |
| **License** | Apache-2.0 |

---

Copyright 2026 Craton Software Company. Licensed under Apache-2.0.
