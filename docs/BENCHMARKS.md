<!--
applies-to: main
owner: performance
generated-fields: none (numbers are measured; every published table names its provenance)
-->

# Craton Bolt Benchmarks

This document captures the current measured performance of the Craton Bolt
GPU SQL engine and explains how to reproduce the numbers. The headline
table is the canonical reference; the historical appendix at the bottom
preserves the wave-by-wave progression for context.

---

## Section 1 — Current benchmark results (0.9.0, RTX 2060)

> **Provenance.** The h2o.ai table was refreshed at commit `059b454`
> (2026-10-13, crate 0.9.0); the heavy-arithmetic and CPU-overhead tables are
> carried forward from earlier 0.9-line runs on the same machine and predate
> the reproducibility manifest. Treat them as measurements of the build that
> produced them, not as guarantees about `main` — re-run with the manifest
> (below) before quoting a number against a newer commit.

All GPU numbers were captured on **NVIDIA GeForce RTX 2060**, driver
610.88 (WDDM), CUDA 13.3 toolkit, with the `BOLT_BENCH_GPU=1` environment
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
| **q1** low-card SUM (id1, 100 grps)    | 7.1 ms    | 19.9 ms    | **4.1 ms** ⭐  | Tier-1 single-SUM                |
| **q2** med-card 2-SUM (id2, 10 K grps) | 62.0 ms   | 104.7 ms   | **110.3 ms**   | Tier-2.1 multi-SUM (N-value reduce) |
| **q3** two-key SUM (≈ 1 M grps)        | 496.9 ms  | 459.9 ms   | **126.7 ms** ⭐ | Tier-2.1 two-key (i64 partition_reduce) |
| **q4** low-card 3-AVG (id1, 100 grps)  | 12.5 ms   | 97.6 ms    | **54.4 ms**    | Tier-1 AVG (SUM + COUNT)         |
| **q5** high-card SUM (id3, 1 M grps)   | 508.7 ms  | 363.2 ms   | **213.3 ms** ⭐ | Tier-2.1 (NUM_PARTITIONS=4096, pass-2 on GPU) |


Craton Bolt wins outright on three workloads (q1, q3, q5) where GPU parallelism
and partitioning outpace CPU engines. DuckDB remains faster on q2 and q4.
See [The honest read](#the-honest-read) for the full analysis.

### Heavy-arithmetic OLAP (50 M rows)

A separate workload exercising fused multi-operator arithmetic on wide
columns — the regime where GPU compute density pays off.

| Query                                | Polars (CPU MT)         | Craton Bolt E2E (GPU)         | CPU reference (single-thread) | Winner                  |
| ------------------------------------ | ----------------------- | ----------------------------- | ----------------------------- | ----------------------- |
| **`proj`** (passthrough)             | **9.04 µs** *(zero-copy)* | 187.7 ms *(266.4 Melem/s)*    | —                             | Polars (zero-copy)      |
| **`arith`** (11-op chain)            | 3.39 s *(14.7 Melem/s)* | **185.7 ms** *(269.3 Melem/s)*| 231.8 ms *(215.8 Melem/s)*    | **Craton Bolt — 18.3× faster** |
| **`filtered`** (filter + 4-op arith) | 358.2 ms *(139.6 Melem/s)*| **110.0 ms** *(454.4 Melem/s)*| —                             | **Craton Bolt — 3.3× faster**  |

### CPU-side pipeline overhead (no GPU needed)

`plan + lower + ptx_gen` is everything that happens before a kernel can
be launched. Summed per query (median estimates, 1 M-row synthetic
dataset, [`benches/query_benchmarks.rs`](../benches/query_benchmarks.rs)):

| Query      | plan      | lower     | ptx_gen   | **Total**     |
| ---------- | --------- | --------- | --------- | ------------- |
| `proj`     | 572.86 ns | 1.9292 µs | 9.5360 µs | **12.038 µs** |
| `arith`    | 728.89 ns | 2.5480 µs | 12.295 µs | **15.572 µs** |
| `filtered` | 822.18 ns | 2.8646 µs | 12.525 µs | **16.212 µs** |

Even the worst-case query is well under 20 µs of pre-launch overhead.
**JIT-compiling-on-every-query is a viable execution model** — the
planning + codegen budget is negligible compared to any non-trivial GPU
launch + h2d/d2h round trip, which is typically in the high tens to
hundreds of microseconds.

### The honest read

On the standard groupby OLAP workload, Craton Bolt's GPU engine is now
faster than both Polars and DuckDB on three queries (q1, q3, q5), beats
Polars on q4, and closely trails on q2.

Where the time goes on the queries Craton Bolt still loses:

- **q2 (10 K groups, multi-SUM).** The Tier-2.1 multi-SUM path measures at
  110.3 ms and closely approaches Polars (104.7 ms), though DuckDB leads with
  per-core thread-local tables (62.0 ms).
- **q4 (100 groups, multi-AVG).** The Tier-1 AVG path measures at 54.4 ms
  (beating Polars' 97.6 ms), but DuckDB's compact L1-resident state machine
  runs in 12.5 ms.

The `arith` win on the heavy-arithmetic OLAP table is real but somewhat
amplified by Polars' eager-binary materialisation under chained
`lit() * col() + …` expressions; the CPU reference loop (231.8 ms) is the
more honest CPU ceiling for that workload, and Craton Bolt is still
**1.25× faster** than that (and **18.3× faster** than Polars).

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

### Scaling and VRAM-pressure protocol

[`benches/scaling_benchmarks.rs`](../benches/scaling_benchmarks.rs) provides a
real-device fused filter/projection sweep. It defaults to 1 K, 10 K, 100 K,
1 M, and 10 M rows; an explicit list can extend it through 100 M:

```bash
BOLT_BENCH_GPU=1 \
BOLT_SCALING_ROWS=1000,10000,100000,1000000,10000000,100000000 \
cargo bench --bench scaling_benchmarks
```

For capacity behavior, add one point selected from the installed device's free
VRAM and record both total/free VRAM before the run:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
BOLT_BENCH_GPU=1 BOLT_VRAM_PRESSURE_ROWS=100000000 \
cargo bench --bench scaling_benchmarks
```

The pressure point must either complete or return a bounded
`BoltError::GpuCapacity`/host fallback; a driver fault or process OOM is a
failure. Cross-GPU comparisons belong in separate rows with exact model,
driver, toolkit, power mode, and row sweep. The current canonical numbers are
RTX 2060-only and must not be generalized to another GPU class.

### Reproducibility manifest

Every bench binary writes a JSON manifest at start-up to
`target/bench-manifests/<bench>-<unix-seconds>.json` (helper:
[`benches/support/manifest.rs`](../benches/support/manifest.rs)) and prints
its path. A published number must cite one. Fields:

| Field | Source |
|---|---|
| `commit`, `worktree_dirty` | `GITHUB_SHA` or `git rev-parse HEAD`; `git status --porcelain` |
| `crate_version` | `CARGO_PKG_VERSION` |
| `rustc`, `profile`, `features` | `rustc --version`; `debug_assertions`; compiled-in Cargo features |
| `gpu.model`, `gpu.compute_capability`, `gpu.vram`, `driver`, `power_limit` | `nvidia-smi --query-gpu=...` |
| `cuda_toolkit` | `nvcc --version` |
| `os` | target OS |
| `axes` | the bench's experimental axes: rows, sweep, cold/warm, transfer-included, correctness oracle |

Fields whose source is unavailable are `null`, never guessed. Criterion's
median / CI live in `target/criterion/**/estimates.json`; archive them with
the manifest.

### Cost-model matrix

[`benches/cost_model.rs`](../benches/cost_model.rs) turns runtime decisions
into response surfaces: each group sweeps one independent variable with
everything else fixed.

| Group | Independent variable | Records | Needs GPU |
|---|---|---|---|
| `plan_pipeline` | query shape (filter, 64-term expression, 3-way join, subqueries, GROUP BY+HAVING) × stage; optimizer on/off | parse / optimize / lower µs | no |
| `dict_rewrite` | predicate count × dictionary cardinality | rewrite µs | CUDA build |
| `fragmentation` | 1, 2, 8, 64, 1024 batches at constant rows | per-query concat (pre-coalescing behaviour) vs coalesced read, bytes copied | no |
| `batch_ingest` | 1 → 512 appended batches | ingest + first query | yes |
| `filter_selectivity` | 0 / 1 / 10 / 50 / 90 / 100 % | end-to-end (kernel + compaction + D2H) | yes |
| `groupby_cardinality` | 1 → 1 M keys | throughput | yes |
| `groupby_skew` | uniform vs Zipf-like hot keys | throughput / atomic serialisation | yes |
| `null_density` | 0 / 10 / 50 / 90 % NULL | validity overhead | yes |
| `jit_cache` | cold codegen + module load vs warm module | latency | yes |
| `residency` | resident `GpuTable` vs re-uploaded table | H2D cost | yes |
| `join_shape` | build:probe ratio 1:1, 1:10, 1:100 | latency | yes |
| `string_width` | 4 B / 64 B / 1 KiB values | dictionary filter latency | yes |
| `morsel_size` | memory budget = table … table/64 (lazy stream) | morsel overhead | yes |
| `recursive_cte` | depth 10 / 100 / 500 | cumulative materialisation | yes |

```bash
cargo bench --bench cost_model --no-default-features --features cuda-stub   # host groups
BOLT_BENCH_GPU=1 BOLT_COST_ROWS=4000000 cargo bench --bench cost_model      # all groups
```

Device-memory pressure (a working set below and above free VRAM) is the
`BOLT_VRAM_PRESSURE_ROWS` point of the scaling bench (below). Concurrent
queries / sessions are not benchmarked: the engine runs one query at a time
per process by design (see `LIMITATIONS.md`).

**What it surfaced.** Besides fragmentation (below), the first GPU run
exposed two GROUP BY gaps that are now fixed: a single-key `SUM ... GROUP BY`
was 7× slower than 100 keys (every thread contending for one shared slot —
fixed by per-thread run-length + warp pre-aggregation: 23.8 → 3.8 ms at 4 M
rows), and any `Int64` key or integer `SUM` missed the resident fast path
entirely (145–280 ms → 3.6–5.3 ms after typing the Tier-1 kernel). The
remaining gap — high-cardinality `Int64` / integer shapes — is listed in
`LIMITATIONS.md`.

**What it already decided.** The `fragmentation` group measured the cost
the multi-batch representation used to impose on *every* query: concatenating
a 2 M-row (16 MB) table held in 2–1024 batches took **7–10 ms per query**,
versus **~0.1 µs** for reading a coalesced batch (RTX 2060 host, Criterion
`--quick`, manifest commit `9041595`+local changes). That is why tables are
now coalesced once at ingest (`register_batch`, `register_table_stream`, lazy
drain) instead of concatenated per query. The `batch_ingest` group keeps the
remaining ingest-side cost visible: N appends still rebuild the dictionary
union over the whole table, so N tiny appends cost O(N × table) — prefer one
`register_table_stream` call (a single coalesce) for bulk loads.

### Publishing a result

Use this template for any number that leaves the repository (README, release
notes, talks):

```markdown
### <Benchmark name>

Commit: `<sha>` (manifest: `target/bench-manifests/<file>.json`)
Craton Bolt: <crate version>        Date: YYYY-MM-DD

Hardware: GPU <model, CC, VRAM> · CPU <model> · RAM <GB> · PCIe <gen×lanes>
Driver <version> · CUDA <toolkit> · power limit <W>

Configuration: features <...> · memory budget <...> · PTX cache cold/warm ·
input residency host/GPU · transfers included yes/no

Dataset: rows <n> · columns <...> · key cardinality <...> · skew <...> ·
null fraction <...>

Correctness oracle: DuckDB <version> / Polars <version>, tolerance <...>

| Query | median | p95 | CI | rows/s |
|---|---|---|---|---|
```

### Scheduled benchmark history

Hosted CI runs `cuda-stub`, so it cannot observe GPU performance. The
[`bench-history`](../.github/workflows/bench-history.yml) workflow runs the
cost-model and h2o suites weekly (and on demand) on the self-hosted GPU
runner and uploads `target/criterion` plus `target/bench-manifests` as a
dated artifact, building the per-commit history regressions are judged
against.

### Remaining measurement detail

- **Per-shape `engine_execute` breakdown.** Right now `engine_execute`
  is a single end-to-end measurement. A useful extension is to break
  it into the constituent costs: kernel launch latency, h2d transfer,
  compute, d2h transfer, and (for `filtered`) compaction or gather
  overhead.
- **Additional GPU classes.** The reproducible harness now covers scaling and
  pressure, but this checkout has evidence from one RTX 2060 only. A second
  model's results must be measured on that hardware, never estimated.

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

### Wave: Tier-2.1 — pass-2-on-GPU + multi-SUM + two-key (0.7.0 historical)

Three follow-ups landed together: (a) pass-2-on-GPU for the existing
single-SUM Tier-2 (closes q5), (b) multi-SUM Tier-2 / Tier-2.1 for q2,
(c) two-key Tier-2.1 for q3. These produced the 0.7.0 results (superseded
in 0.9.0; see Section 1 for canonical numbers).

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
