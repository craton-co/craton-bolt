# Javelin Benchmarks

This document captures the first measured numbers for the Javelin GPU SQL engine and explains how to reproduce them. The numbers below were captured on **2026-05-23** on a CPU-only host (no CUDA device available), so the full `engine_execute` GPU pipeline group was skipped. Everything that does not require a GPU — SQL parsing, logical planning, physical lowering, PTX codegen, the hand-written CPU reference loop, and the Polars head-to-head — was measured end-to-end.

The benchmark suite lives in [`benches/query_benchmarks.rs`](../benches/query_benchmarks.rs) and is driven by Criterion 0.5. It runs against a fixed synthetic 1,000,000-row dataset and exercises six bench groups (three of which are CPU-only stages, one CPU reference, one Polars comparison, one GPU end-to-end).

## TL;DR

All times are the median of the criterion `[low, mid, high]` triplet. Throughput is whatever criterion reported for the group; "—" means criterion did not measure throughput for that stage.

| Group               | proj           | arith          | filtered       | Notes                                              |
| ------------------- | -------------- | -------------- | -------------- | -------------------------------------------------- |
| `plan/*`            | 13.713 µs      | 9.4081 µs      | 11.850 µs      | SQL string → `LogicalPlan`                         |
| `lower/*`           | 429.53 ns      | 563.19 ns      | 620.96 ns      | `LogicalPlan` → `PhysicalPlan`                     |
| `ptx_gen/*`         | 6.9632 µs      | 9.4604 µs      | 11.221 µs      | `KernelSpec` → PTX string                          |
| `cpu_reference`     | —              | 3.5080 ms (285.07 Melem/s) | — | Single-threaded Rust `for` loop, `a[i] * b[i]`     |
| `polars/*`          | 6.5173 µs (153.44 Gelem/s) | 2.9405 ms (340.08 Melem/s) | 1.5970 ms (626.16 Melem/s) | Multi-threaded Polars LazyFrame baseline |
| `engine_execute/*`  | SKIPPED        | SKIPPED        | SKIPPED        | Needs `JAVELIN_BENCH_GPU=1` and a CUDA device      |

These numbers reflect the **0.1.0 baseline**, captured before wave 5's
codegen and runtime perf changes (PTX cache, `.ptr .global .restrict`,
`cuMemsetD8`-based `GpuBuffer::zeros`, `#[inline]` on smart-pointer
accessors). The CPU-side stages were already cheap; the codegen tweaks
mainly help the `engine_execute/*` path, which still needs a re-run on a
CUDA host. Treat the table above as a regression floor — anything
slower in a future run is a regression.

## The benchmark queries

Three SQL queries are exercised end-to-end through every applicable stage:

- **`proj`** — `SELECT price FROM sales`
  Pure column projection. Exercises the simplest possible plan, lowering, and codegen path; on the GPU it should be near a straight `cudaMemcpy`.
- **`arith`** — `SELECT price * tax FROM sales`
  Scalar arithmetic over two `Float64` columns. The interesting kernel: tests fused multiply-add codegen on the GPU and is the apples-to-apples target for the CPU reference loop.
- **`filtered`** — `SELECT price FROM sales WHERE region_id = 1`
  Projection with a selection. Exercises predicate codegen and (on the GPU) stream compaction.

The dataset is generated synthetically in the bench file:

| Column      | Type      | Generator                |
| ----------- | --------- | ------------------------ |
| `region_id` | `Int32`   | `i % 4`                  |
| `price`     | `Float64` | `(i + 1) as f64`         |
| `tax`       | `Float64` | constant `0.0825`        |

With `region_id = i % 4`, exactly ~25% of the 1,000,000 rows pass the `WHERE region_id = 1` predicate.

## CPU pipeline overhead

`plan + lower + ptx_gen` is everything that happens on the CPU before a kernel can be launched on the GPU. Summed per query (median estimates):

| Query      | plan      | lower     | ptx_gen   | **Total**     |
| ---------- | --------- | --------- | --------- | ------------- |
| `proj`     | 13.713 µs | 429.53 ns | 6.9632 µs | **21.106 µs** |
| `arith`    | 9.4081 µs | 563.19 ns | 9.4604 µs | **19.432 µs** |
| `filtered` | 11.850 µs | 620.96 ns | 11.221 µs | **23.692 µs** |

Even the worst-case query is well under 50 µs of pre-launch overhead. **JIT-compiling-on-every-query is a viable execution model** — the planning + codegen budget is negligible compared to any non-trivial GPU launch + h2d/d2h round trip, which is typically in the high tens to hundreds of microseconds. There is no need for a kernel cache from a correctness or even a performance standpoint at this scale; one would only become interesting at sub-millisecond query rates.

Note that `plan/arith` is *faster* than `plan/proj` — the SQL parser apparently has slightly more overhead for the bare `SELECT price FROM sales` path than for the arithmetic one. This is a small absolute number and not worth chasing.

## CPU reference baseline

`cpu_reference/price_times_tax` is the simplest possible hand-written single-threaded baseline for the `arith` workload:

```text
let mut out = Vec::with_capacity(n);
for i in 0..n {
    out.push(price[i] * tax[i]);
}
```

`Vec::with_capacity` is used so no reallocations happen inside the timed region. The result: **3.5080 ms** for 1,000,000 multiplies, throughput **285.07 Melem/s**.

This is the floor we want any "real" SQL engine to beat on this workload — it is what a competent C programmer would write by hand and represents close to the per-core memory-bandwidth ceiling for paired `Float64` reads on commodity hardware.

## Polars head-to-head

[Polars](https://www.pola.rs/) is the obvious head-to-head benchmark: it is an in-memory columnar engine that compiles a LazyFrame plan and executes it on a multi-threaded CPU backend (Rayon's default thread pool). Each measurement constructs a fresh LazyFrame, calls `select` / `filter`, and `collect`s.

| Query      | Time      | Throughput     | Comment                                                                                          |
| ---------- | --------- | -------------- | ------------------------------------------------------------------------------------------------ |
| `proj`     | 6.5173 µs | 153.44 Gelem/s | Suspiciously fast — see below                                                                    |
| `arith`    | 2.9405 ms | 340.08 Melem/s | The real CPU compute baseline                                                                    |
| `filtered` | 1.5970 ms | 626.16 Melem/s | Fast because predicate is cheap and output is ~250k rows                                         |

### `polars/proj` is not a real compute number

153 Gelem/s on a CPU is nonsense — main memory cannot deliver `Float64`s at that rate. What is actually happening: Polars's LazyFrame `select` of a single already-materialised column is a metadata-only operation. It bumps a reference count on the underlying `ChunkedArray` and returns. No data is read, no data is written. The 6.5 µs is plan construction + the metadata clone.

This is not a Polars defect; it is a correct implementation of "select a column that already exists." It just means the number cannot be compared to anything that actually moves bytes. Treat `polars/proj` as a "what is the LazyFrame overhead" measurement, not a throughput measurement.

### `polars/arith` is the number to beat

2.94 ms for `price * tax` over 1M rows is the realistic apples-to-apples CPU number for the `arith` query. Polars uses SIMD and the full thread pool, so it is roughly 1.2× the single-threaded reference loop in wall time — most of the cost here is memory bandwidth, not compute, and Polars cannot parallelise its way out of that.

### `polars/filtered` benefits from cheap predicate + small output

1.6 ms for `SELECT price WHERE region_id = 1` is fast because (a) the comparison `region_id == 1` is a single integer compare per row, (b) only ~25% of rows survive, so the gathered output is ~250k `Float64`s, and (c) Polars's filter implementation is well-tuned. This is the CPU number our GPU pipeline needs to beat for the `filtered` shape on this dataset size.

## GPU pipeline

The `engine_execute/{proj,arith,filtered}` group runs the full Javelin GPU pipeline: parse → plan → lower → codegen → PTX compile → kernel launch → device-to-host copy. **It was SKIPPED on this run** because the bench host did not have a CUDA-capable GPU available.

To enable it on a CUDA-equipped host:

```bash
JAVELIN_BENCH_GPU=1 cargo bench
```

When enabled, criterion will report three additional rows in the form:

```text
engine_execute/proj         <time>   throughput <Gelem/s or Melem/s>
engine_execute/arith        <time>   throughput <Gelem/s or Melem/s>
engine_execute/filtered     <time>   throughput <Gelem/s or Melem/s>
```

These numbers are the headline GPU performance story and need to be captured on a real device before this document is complete. See [What's missing](#whats-missing) below.

## What we learn

- **Pre-launch overhead is essentially free.** Parse + plan + codegen all-in is under 25 µs per query. There is no JIT cache, no kernel pool, no plan cache — and at this scale, none of those are needed. Anything that runs once per query and finishes in tens of microseconds is invisible next to a real GPU launch.
- **The CPU reference is ~3.5 ms for 1M `Float64` multiplies, single-threaded.** This is the floor we'd want the GPU to crush. A Volta-class V100 has roughly 7 TFLOPS of `Float64` throughput and ~900 GB/s of HBM bandwidth, so on a pure-throughput basis the GPU should be at least an order of magnitude faster than the CPU here. The question is **whether we can amortise the launch + h2d/d2h transfer cost**. Back-of-the-envelope: PCIe Gen3 x16 is ~12 GB/s usable; transferring 1M `Float64` columns (16 MB total) is ~1.3 ms each way. So on a cold cache the data movement alone dominates the CPU compute time, and the GPU only wins if the data is already resident or the kernel does enough work per byte. **Anticipated break-even for arithmetic queries: ~100k rows** for hot-cache, much higher for cold-cache.
- **Polars `filtered` at 1.6 ms is the number to beat on this hardware class.** It is multi-threaded and well-tuned; beating it on the GPU at 1M rows requires either a resident dataset or a heavier per-row predicate than `region_id == 1`.

## Methodology

- **Harness**: Criterion 0.5 with default settings — 100 samples per measurement, 3-second warmup, default outlier filter.
- **Concurrency**: criterion serialises individual sample iterations, so each measurement is effectively single-threaded *from the harness's perspective*. Polars itself uses Rayon's default thread pool inside the timed region.
- **Polars setup**: each Polars sample constructs a fresh LazyFrame from the in-memory `DataFrame`, applies `select` or `filter().select()`, and calls `collect()`. Plan caching is *not* in scope; the LazyFrame is built per iteration.
- **CPU reference**: a plain `for i in 0..n { out.push(a[i] * b[i]) }` loop with `Vec::with_capacity(n)` preallocated. No SIMD, no parallelism. The `out` `Vec` is dropped between iterations.
- **Dataset size**: 1,000,000 rows. This is intentionally on the small side — large enough that per-row costs dominate, small enough that everything fits comfortably in L2/L3 on most hardware. The constant `BENCH_ROWS` in `benches/query_benchmarks.rs` can be increased to sweep larger sizes.
- **GPU gating**: the `engine_execute` group is gated on the `JAVELIN_BENCH_GPU=1` environment variable so the suite is runnable on machines without CUDA installed.

## Reproducing

CPU-only run (everything except `engine_execute`):

```bash
cargo bench
```

Full run with the GPU pipeline (requires CUDA toolkit and a CUDA-capable device):

```bash
JAVELIN_BENCH_GPU=1 cargo bench
```

Single bench group only (criterion filters by group name):

```bash
cargo bench --bench query_benchmarks -- polars
```

After a run, the full HTML reports (with per-sample distributions, regression plots, and outlier callouts) land in:

```text
target/criterion/
```

Open `target/criterion/report/index.html` for the top-level summary.

## System context

The numbers above were captured on a CPU-only bench host whose exact CPU model and RAM configuration are not recorded here. **The absolute numbers may not transfer to your machine** — a faster or slower CPU will shift everything proportionally, and the Polars numbers will scale with core count.

What *should* hold across hardware is the **relative ordering**:

```text
lower  ≪  ptx_gen  ≈  plan  ≪  polars/proj  ≪  polars/filtered  <  cpu_reference  ≈  polars/arith
```

The structural conclusion — that pre-launch overhead is negligible relative to any compute, and that Polars's `arith` time tracks the single-threaded reference loop closely — should be reproducible anywhere.

## What's missing

- **Live GPU numbers.** The headline `engine_execute/*` row is empty. The benchmark needs to be re-run on a CUDA-equipped host (ideally something like a V100, an A100, or a consumer RTX-class card for comparison) and the numbers added here.
- **Per-shape `engine_execute` breakdown.** Right now `engine_execute` is a single end-to-end measurement. A useful extension is to break it into the constituent costs: kernel launch latency, h2d transfer, compute, d2h transfer, and (for `filtered`) compaction or gather overhead.
- **Long-tail / large-N sweep.** Everything is fixed at 1M rows. The interesting curve — and the one that determines GPU break-even — is throughput vs row count from ~1k up to ~100M. A `cargo bench --bench query_benchmarks_sweep` would be the natural follow-up.
- **Memory-pressure tests.** No measurements anywhere near VRAM limits. A 100M-row dataset (~3.2 GB at the current schema) starts to stress consumer GPUs and is the regime where memory-management decisions show up.

## See also

- [`docs/JIT_PIPELINE.md`](./JIT_PIPELINE.md) — what `plan`, `lower`, and `ptx_gen` actually do under the hood, and why each stage exists as a separate measurement.
- [`docs/SQL_REFERENCE.md`](./SQL_REFERENCE.md) — which SQL features are supported today, which constrains what we can meaningfully benchmark.
- [`docs/ARCHITECTURE.md`](./ARCHITECTURE.md) — overall system architecture.
- [`benches/query_benchmarks.rs`](../benches/query_benchmarks.rs) — the source of truth for the bench definitions.
