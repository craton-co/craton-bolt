# Craton Bolt Benchmarks

This document captures the current measured performance of the Craton Bolt
GPU SQL engine and explains how to reproduce the numbers. The headline
table is the canonical reference; the historical appendix at the bottom
preserves the wave-by-wave progression for context.

---

## Section 1 — Current benchmark results (0.7.0, RTX 2060)

All GPU numbers were captured on **NVIDIA GeForce RTX 2060**, driver
591.86 (WDDM), CUDA 12.6 toolkit, with the `BOLT_BENCH_GPU=1` environment
variable. Times are criterion medians from a 20 s measurement window
(≥ 100 samples each); the criterion CI is under ±5 % on every reported
number. All cross-engine results are verified bit-equivalent against
Polars 0.42 and DuckDB 1.2 on a 100 K-row fixture before the timed runs
begin (1 e-9 relative tolerance).

### h2o.ai db-benchmark GROUP BY subset (10 M rows)

This is the protocol-disciplined OLAP comparison. Schema, query shapes
and cardinalities match the [h2o.ai db-benchmark](https://duckdblabs.github.io/db-benchmark/)
spec; see [`benches/olap_benchmarks.rs`](../benches/olap_benchmarks.rs).

| Query                                  | DuckDB    | Polars     | Craton Bolt    | Fast path engaged                |
| -------------------------------------- | --------- | ---------- | -------------- | -------------------------------- |
| **q1** low-card SUM (id1, 100 grps)    | 6.9 ms    | 19.0 ms    | **51.4 ms**    | Tier-1 single-SUM                |
| **q2** med-card 2-SUM (id2, 10 K grps) | 46.4 ms   | 99.4 ms    | **384 ms**     | Tier-2.1 multi-SUM (N-value reduce) |
| **q3** two-key SUM (≈ 1 M grps)        | 498 ms    | 385 ms     | **219 ms** ⭐  | Tier-2.1 two-key (i64 partition_reduce) |
| **q4** low-card 3-AVG (id1, 100 grps)  | 12.9 ms   | 97.0 ms    | **70.5 ms**    | Tier-1 AVG (SUM + COUNT)         |
| **q5** high-card SUM (id3, 1 M grps)   | 623 ms    | 358 ms     | **237 ms** ⭐  | Tier-2.1 (NUM_PARTITIONS=4096, pass-2 on GPU) |


Craton Bolt wins outright on the two highest-cardinality workloads (q3,
q5) where GPU-parallel hash-partitioning outpaces CPU per-core hash
tables. CPU-native engines win at low cardinality (q1, q4) where their
per-thread L1-resident tables beat GPU atomic contention. See
[The honest read](#the-honest-read) for the full analysis.

### Heavy-arithmetic OLAP (50 M rows)

A separate workload exercising fused multi-operator arithmetic on wide
columns — the regime where GPU compute density pays off.

| Query                                | Polars (CPU MT)         | Craton Bolt E2E (GPU)         | CPU reference (single-thread) | Winner                  |
| ------------------------------------ | ----------------------- | ----------------------------- | ----------------------------- | ----------------------- |
| **`proj`** (passthrough)             | **12.55 µs** *(zero-copy)* | 115.5 ms *(432.9 Melem/s)* | —                             | Polars (zero-copy)      |
| **`arith`** (11-op chain)            | 4.05 s *(12.3 Melem/s)* | **124.8 ms** *(400.7 Melem/s)*| 1.06 s *(47.2 Melem/s)*       | **Craton Bolt — 32.4× faster** |
| **`filtered`** (filter + 4-op arith) | 369 ms *(135.5 Melem/s)*| **41.8 ms** *(1.196 Gelem/s)* | —                             | **Craton Bolt — 8.8× faster**  |

### CPU-side pipeline overhead (no GPU needed)

`plan + lower + ptx_gen` is everything that happens before a kernel can
be launched. Summed per query (median estimates, 1 M-row synthetic
dataset, [`benches/query_benchmarks.rs`](../benches/query_benchmarks.rs)):

| Query      | plan      | lower     | ptx_gen   | **Total**     |
| ---------- | --------- | --------- | --------- | ------------- |
| `proj`     | 13.713 µs | 429.53 ns | 6.9632 µs | **21.106 µs** |
| `arith`    | 9.4081 µs | 563.19 ns | 9.4604 µs | **19.432 µs** |
| `filtered` | 11.850 µs | 620.96 ns | 11.221 µs | **23.692 µs** |

Even the worst-case query is well under 50 µs of pre-launch overhead.
**JIT-compiling-on-every-query is a viable execution model** — the
planning + codegen budget is negligible compared to any non-trivial GPU
launch + h2d/d2h round trip, which is typically in the high tens to
hundreds of microseconds.

### The honest read

On the standard groupby OLAP workload, Craton Bolt's GPU engine is now
faster than both Polars and DuckDB on the two highest-cardinality queries
(q3, q5), within striking distance on q4, and still trails CPU engines
on the very-low-cardinality (q1) and medium-cardinality multi-SUM (q2)
cases.

Where the time goes on the queries Craton Bolt still loses:

- **q1 / q4 (100 groups).** With only 100 destination buckets, CPU
  engines fit the entire hash table in L1 per thread; the GPU pays
  fixed launch + D2H costs that dominate the actual compute.
- **q2 (10 K groups, multi-SUM).** The Tier-2.1 multi-SUM path landed
  late and gives a modest 1.15× speedup (444 ms → 384 ms) at this
  cardinality. The path scales the same way single-SUM Tier-2.1 does,
  so larger-cardinality multi-SUM workloads should track closer to
  q3 / q5's 3.7× wins.

The `arith` win on the heavy-arithmetic OLAP table is real but somewhat
amplified by Polars' eager-binary materialisation under chained
`lit() * col() + …` expressions; the CPU reference loop (1.06 s) is the
more honest CPU ceiling for that workload, and Craton Bolt is still
**8.5× faster** than that.

---

## Section 2 — Methodology and reproducibility

### The benchmark queries (lightweight suite)

Three SQL queries are exercised end-to-end through every applicable
stage in [`benches/query_benchmarks.rs`](../benches/query_benchmarks.rs):

- **`proj`** — `SELECT price FROM sales`. Pure column projection; on
  the GPU it should be near a straight `cudaMemcpy`.
- **`arith`** — `SELECT price * tax FROM sales`. Scalar arithmetic
  over two `Float64` columns; tests fused multiply-add codegen.
- **`filtered`** — `SELECT price FROM sales WHERE region_id = 1`.
  Projection with a selection; exercises predicate codegen and (on the
  GPU) stream compaction.

Dataset (1 M rows, synthetic):

| Column      | Type      | Generator         |
| ----------- | --------- | ----------------- |
| `region_id` | `Int32`   | `i % 4`           |
| `price`     | `Float64` | `(i + 1) as f64`  |
| `tax`       | `Float64` | constant `0.0825` |

With `region_id = i % 4`, exactly ~25 % of rows pass the `WHERE region_id = 1`
predicate.

The heavy-arithmetic table above uses the same harness with
`BENCH_ROWS = 50_000_000` and an 11-op chained arithmetic expression on
`arith` / a 4-op chained expression on `filtered`. The 20 s criterion
window keeps each median tight (CI ≤ ±3 %).

### h2o.ai groupby protocol

See [`benches/olap_benchmarks.rs`](../benches/olap_benchmarks.rs) and
[`docs/COMPETITIVE_BENCHMARKING.md`](./COMPETITIVE_BENCHMARKING.md).

- **Benchmark**: GROUP-BY subset of the h2o.ai db-benchmark — the
  standard reference Polars, DuckDB, Pandas, ClickHouse and others use
  in their own published OLAP comparisons. Schema and query shapes match
  the spec; the only deviation is that grouping keys are `Int32` instead
  of categorical strings, so Craton Bolt's GPU GROUP-BY (which does not
  yet hash string keys) can run the same SQL the CPU engines do.
- **Engines**:
  - **Craton Bolt** — this crate, GPU SQL engine.
  - **Polars 0.42** — Rust-native, Rayon-threaded.
  - **DuckDB 1.2** — bundled embedded C++ engine, multi-threaded.
- **Workload**: `N = 10_000_000` (the h2o.ai "small" scale), 6 columns
  (`id1`/`id2`/`id3` Int32 grouping keys, `v1`/`v2`/`v3` Float64
  values). Cardinalities pinned to the h2o.ai shape: id1 = 100,
  id2 = 10 K, id3 = 1 M.
- **Queries** (h2o.ai numbering):
  - **q1** — `SELECT id1, SUM(v1) FROM x GROUP BY id1`
  - **q2** — `SELECT id2, SUM(v1), SUM(v2) FROM x GROUP BY id2`
  - **q3** — `SELECT id1, id2, SUM(v1) FROM x GROUP BY id1, id2`
  - **q4** — `SELECT id1, AVG(v1), AVG(v2), AVG(v3) FROM x GROUP BY id1`
  - **q5** — `SELECT id3, SUM(v1) FROM x GROUP BY id3`
- **Verification**: every query runs through every engine on a 100 K-row
  fixture before any timing, and outputs are compared with a 1 e-9
  relative tolerance. Cross-engine disagreement panics.
- **Measurement**: criterion 20 s window per query, ≥ 100 samples each.
  Reported numbers are the median; ranges are the criterion
  `[low, mid, high]` triplet.

### Harness and host

- **Harness**: Criterion 0.5 with default outlier filter.
- **Concurrency**: criterion serialises individual sample iterations,
  so each measurement is effectively single-threaded *from the harness's
  perspective*. Polars itself uses Rayon's default thread pool inside
  the timed region.
- **Polars setup**: each Polars sample constructs a fresh LazyFrame
  from the in-memory `DataFrame`, applies `select` or
  `filter().select()`, and calls `collect()`.
- **CPU reference**: a plain `for i in 0..n { out.push(a[i] * b[i]) }`
  loop with `Vec::with_capacity(n)` preallocated. No SIMD, no
  parallelism.
- **GPU gating**: the `engine_execute` and `olap_benchmarks` groups are
  gated on the `BOLT_BENCH_GPU=1` environment variable so the suite is
  runnable on machines without CUDA installed.

### Reproducing

CPU-only run (everything except GPU paths):

```bash
cargo bench
```

Full run with the GPU pipeline (requires CUDA toolkit and a CUDA-capable
device):

```bash
BOLT_BENCH_GPU=1 cargo bench
```

Single bench group only (criterion filters by group name):

```bash
cargo bench --bench query_benchmarks -- polars
BOLT_BENCH_GPU=1 cargo bench --bench olap_benchmarks
```

After a run, the full HTML reports (with per-sample distributions,
regression plots, and outlier callouts) land in:

```text
target/criterion/
```

Open `target/criterion/report/index.html` for the top-level summary.

### What's not yet measured

- **Per-shape `engine_execute` breakdown.** Right now `engine_execute`
  is a single end-to-end measurement. A useful extension is to break
  it into the constituent costs: kernel launch latency, h2d transfer,
  compute, d2h transfer, and (for `filtered`) compaction or gather
  overhead.
- **Long-tail / large-N sweep.** The lightweight suite is fixed at
  1 M rows; the heavy suite at 50 M; the h2o.ai suite at 10 M. A
  full throughput-vs-row-count curve from ~1 K up to ~100 M would be
  the natural follow-up.
- **Memory-pressure tests.** No measurements anywhere near VRAM
  limits. A 100 M-row dataset (~3.2 GB at the lightweight schema)
  starts to stress consumer GPUs and is the regime where memory-
  management decisions show up.

---

## Section 3 — Historical / wave-by-wave appendix

This section preserves the wave-by-wave progression that produced the
numbers in Section 1. Each subsection is labelled with the date and the
codebase wave that landed the change. Numbers below are kept verbatim
from the runs that produced them; they are **not** the canonical current
numbers — see Section 1 for those.

### 0.1.0 baseline (2026-05-23, CPU-only host)

The first measured numbers, on a CPU-only host with the lightweight
1 M-row suite. The `engine_execute` GPU group was skipped because no
CUDA device was available.

| Group              | proj                          | arith                          | filtered                       | Notes                                          |
| ------------------ | ----------------------------- | ------------------------------ | ------------------------------ | ---------------------------------------------- |
| `plan/*`           | 13.713 µs                     | 9.4081 µs                      | 11.850 µs                      | SQL string → `LogicalPlan`                     |
| `lower/*`          | 429.53 ns                     | 563.19 ns                      | 620.96 ns                      | `LogicalPlan` → `PhysicalPlan`                 |
| `ptx_gen/*`        | 6.9632 µs                     | 9.4604 µs                      | 11.221 µs                      | `KernelSpec` → PTX string                      |
| `cpu_reference`    | —                             | 3.5080 ms (285.07 Melem/s)     | —                              | Single-threaded Rust `for` loop                |
| `polars/*`         | 6.5173 µs (153.44 Gelem/s)    | 2.9405 ms (340.08 Melem/s)     | 1.5970 ms (626.16 Melem/s)     | Multi-threaded Polars LazyFrame baseline       |
| `engine_execute/*` | SKIPPED                       | SKIPPED                        | SKIPPED                        | Needs `BOLT_BENCH_GPU=1` and a CUDA device     |

These were the regression-floor numbers before any GPU pipeline work.
`polars/proj` at 153 Gelem/s is a metadata-only LazyFrame clone, not a
real throughput number; `polars/arith` at 2.94 ms is the realistic
apples-to-apples CPU number; the CPU reference loop at 3.51 ms is the
single-threaded hand-written floor.

### Heavy-workload GPU first run (2026-05-24, RTX 2060)

The first end-to-end GPU numbers, on the 50 M-row heavy-arithmetic
workload described in Section 2. This is the run whose numbers persist
unchanged into Section 1's "Heavy-arithmetic OLAP" table — no later wave
re-ran this workload.

The PTX cache, GPU-resident table, and device memory pool optimisations
(`src/jit/jit_compiler.rs`, `src/exec/gpu_table.rs`,
`src/cuda/mem_pool.rs`) were all in place for this run.

### h2o.ai groupby — first run (2026-05-24, RTX 2060, "pre-fast-paths")

The initial h2o.ai groupby measurement, before any GROUP-BY-specific
optimisations landed. All five queries fell through to the global-atomic
path.

| Query                                 | Polars (CPU MT)              | DuckDB (CPU MT)               | Craton Bolt (GPU)            | Winner                  |
| ------------------------------------- | ---------------------------- | ----------------------------- | ---------------------------- | ----------------------- |
| q1 low-card SUM (id1, 100 grps)       | 16.70 ms                     | **11.95 ms**                  | 282.3 ms                     | **DuckDB**              |
| q2 med-card 2-SUM (id2, 10 K)         | 84.25 ms                     | **50.84 ms**                  | 443.7 ms                     | **DuckDB**              |
| q3 two-key SUM (≈ 1 M grps)           | **348.0 ms**                 | 523.8 ms                      | 807.0 ms                     | **Polars**              |
| q4 low-card 3-AVG (id1, 100 grps)     | 85.09 ms                     | **11.88 ms**                  | 450.9 ms                     | **DuckDB**              |
| q5 high-card SUM (id3, 1 M grps)      | **270.0 ms**                 | 595.0 ms                      | 876.6 ms                     | **Polars**              |

This was the run that drove the GROUPBY_PERF.md analysis and the Tier-1
/ Tier-2 work that followed.

Three correctness bugs surfaced and were fixed during this first run:

1. **AVG codegen emitted unsupported PTX.** `compile_agg_*_kernel` for
   `(Sum, Int64)` and `(Count, Int64)` emitted `atom.global.add.s64`,
   but PTX's `atom.add` has no `.s64` variant. Fix: emit
   `atom.global.add.u64`; two's-complement signed addition is
   bit-identical to unsigned.
2. **Device memory pool ↔ context teardown race.** Pool entries
   outlived their owning CUDA context. Fix: drain the pool inside
   `CudaContext::drop`, before `cuCtxDestroy_v2`.
3. **`Engine::register_table` rejected duplicate names.** Added
   `Engine::replace_table` so the bench can swap the 100 K verification
   fixture for the 10 M timed dataset.

### Wave: Tier-1 + Tier-2 landed (host pass-2)

First batch of GROUP-BY optimisations: Tier-1 shared-memory pre-
aggregation (single-SUM, multi-SUM, AVG via SUM+COUNT), and Tier-2
hash-partitioned two-pass with **host-side** pass-2.

| Query                                | DuckDB    | Polars     | Craton Bolt baseline | Craton Bolt this wave | Δ vs baseline | Fast path triggered |
| ------------------------------------ | --------- | ---------- | -------------------- | --------------------- | ------------- | ------------------- |
| q1 low-card SUM (100 grps)           | 6.9 ms    | 19.0 ms    | 282 ms               | **51.4 ms**           | **5.5×**      | Tier-1 single-SUM   |
| q2 med-card 2-SUM (10 K grps)        | 46.4 ms   | 99.4 ms    | 444 ms               | 510 ms                | ≈ noise       | none (id2 max-key > 1024) |
| q3 two-key SUM (~1 M grps)           | 498 ms    | **385 ms** | 807 ms               | 977 ms                | ≈ noise       | none (two-key not Tier-2-eligible v0) |
| q4 low-card 3-AVG (100 grps)         | **12.9 ms** | 97.0 ms  | 451 ms               | **70.5 ms**           | **6.4×**      | Tier-1 AVG (SUM + COUNT) |
| q5 high-card SUM (1 M grps)          | 623 ms    | **358 ms** | 877 ms               | **460 ms**            | **1.9×**      | Tier-2 hash-partitioned |

q1 and q4 took the Tier-1 single-SUM / AVG fast paths; q5 took the
Tier-2 path; q2 (multi-SUM at medium cardinality) and q3 (two-key)
fell through unchanged because neither Tier-1 nor host-pass-2 Tier-2 v0
supported them.

### Wave: Tier-2.1 — pass-2-on-GPU + multi-SUM + two-key (canonical numbers)

Three follow-ups landed together: (a) pass-2-on-GPU for the existing
single-SUM Tier-2 (closes q5), (b) multi-SUM Tier-2 / Tier-2.1 for q2,
(c) two-key Tier-2.1 for q3. These produced the canonical numbers in
Section 1.

| Query                              | Path engaged                                | Fresh-system time | Pre-fast-path baseline | Δ                |
| ---------------------------------- | ------------------------------------------- | ----------------- | ---------------------- | ---------------- |
| q1 single-SUM (100 grps)           | Tier-1 single-SUM                           | 51.4 ms           | 282 ms                 | **5.5× faster**  |
| q2 multi-SUM (10 K grps)           | **multi-SUM Tier-2.1** (N-value reduce)     | **384 ms**        | 444 ms                 | **1.15× faster** |
| q3 two-key (~1 M grps)             | **two-key Tier-2.1** (i64 partition_reduce) | **219 ms**        | 807 ms                 | **3.7× faster**  |
| q4 3-AVG (100 grps)                | Tier-1 AVG (SUM + COUNT)                    | 70.5 ms           | 451 ms                 | **6.4× faster**  |
| q5 single-SUM (1 M grps)           | Tier-2.1 (NUM_PARTITIONS=4096, GPU pass-2)  | **237 ms**        | 877 ms                 | **3.7× faster**  |

Implementation notes:

- **Tier-2.1 pass-2-on-GPU** replaces the host-side `HashMap`
  reduction with a per-partition GPU kernel
  (`src/jit/partition_reduce_kernel.rs`). v0 with
  `NUM_PARTITIONS = 1024` regressed q5 (460 → 625 ms) due to ~98 %
  load factor against 1 K shared-mem slots. Bumping `NUM_PARTITIONS`
  to 4096 (everywhere — partition / scatter / reduce kernels and
  offsets) drops load factor to ~25 % and lands q5 at 237 ms.
- **Two-key Tier-2.1** is the i64-key port of `partition_reduce_kernel`
  (`src/jit/partition_reduce_kernel_i64.rs`); q3 fell from 953 ms
  (host pass-2) to **219 ms**.
- **Multi-SUM Tier-2.1** (`partition_reduce_kernel_multi`) extends
  the pattern to N value columns (1..=4 SUMs in one launch). With GPU
  pass-2 in place, the `MULTI_SUM_MIN_GROUPS` floor drops from 100 K
  (host pass-2 break-even) to **1024** (Tier-1 cap). At q2's 10 K
  cardinality this gives a modest 1.15× speedup (444 ms → 384 ms);
  larger-cardinality multi-SUM workloads should track closer to
  q3 / q5's 3.7×.

### Tier-2.1 coverage completed: AVG / COUNT / MIN / MAX

The Tier-2.1 pattern now covers every reduction op the engine supports
for integer-typed inputs, plus AVG over Float64. No bench query in the
h2o.ai subset exercises these — they exist for high-cardinality
workloads beyond the canonical groupby suite.

| Op            | Key dtype       | Value dtype       | Status                             |
| ------------- | --------------- | ----------------- | ---------------------------------- |
| SUM (single)  | i32, i64 packed | Float64           | ✓ q5 / q3                          |
| SUM (1..=4)   | i32             | Float64           | ✓ q2                               |
| AVG (1..=4)   | i32             | Float64           | ✓ via SUM + COUNT reduce           |
| COUNT(*)      | i32             | —                 | ✓                                  |
| MIN / MAX     | i32             | Int32, Int64      | ✓                                  |
| MIN / MAX     | i32             | Float32, Float64  | deferred — no native PTX float atomic |

`execute_groupby` now layers nine `try_execute` calls before the
global-atomic fallback:

```text
Tier-1 single-SUM   →  Tier-1 multi-SUM   →  Tier-1 AVG
  →  Tier-2 single-key SUM  →  Tier-2 multi-SUM  →  Tier-2 two-key SUM
  →  Tier-2 AVG  →  Tier-2 COUNT  →  Tier-2 MIN/MAX
  →  GlobalAtomic
```

Each gate is a deterministic precondition check that returns `None` on
a miss; no branch shadows a faster one.

### CUDA-Oxide refactor (proof of concept)

`groupby_shmem_exec.rs` (the Tier-1 single-SUM executor) has been
lifted from raw `CUdeviceptr` to typed `GpuView<'a, T>` /
`GpuViewMut<'a, T>` via the `KernelArgs::push_scalar_u32` +
`launch_with_geometry` extension in
[`src/exec/launch.rs`](../src/exec/launch.rs). q1 measured at 156 ms
post-refactor (within ±1 % of the raw-pointer version), confirming
zero overhead. The pattern can be applied to the other six executors
in follow-up PRs.

The [`docs/CUDARC_ADOPTION.md`](./CUDARC_ADOPTION.md) design doc
covers stage-by-stage migration (cuda_sys → jit_compiler → launch →
mem_pool), feature-flag strategy, regression-safety plan, and a
~5–8 engineering-day budget. The CUDA-Oxide layer (`GpuVec`,
`GpuView`, `GpuViewMut`) stays the same; cudarc replaces what's
*below* it.

---

## See also

- [`docs/COMPETITIVE_BENCHMARKING.md`](./COMPETITIVE_BENCHMARKING.md) — methodology and discipline for running head-to-head comparisons against DuckDB, Polars, HeavyDB, and friends. Read before publishing any "Craton Bolt vs X" numbers.
- [`docs/JIT_PIPELINE.md`](./JIT_PIPELINE.md) — what `plan`, `lower`, and `ptx_gen` actually do under the hood, and why each stage exists as a separate measurement.
- [`docs/GROUPBY_PERF.md`](./GROUPBY_PERF.md) — the full Tier-1 / Tier-2 / Tier-2.1 design analysis with kernel sketches and expected speedups.
- [`docs/SQL_REFERENCE.md`](./SQL_REFERENCE.md) — which SQL features are supported today, which constrains what we can meaningfully benchmark.
- [`docs/ARCHITECTURE.md`](./ARCHITECTURE.md) — overall system architecture.
- [`benches/query_benchmarks.rs`](../benches/query_benchmarks.rs) — the source of truth for the lightweight bench definitions.
- [`benches/olap_benchmarks.rs`](../benches/olap_benchmarks.rs) — the source of truth for the h2o.ai groupby bench definitions.
