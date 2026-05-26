# Competitive Benchmarking Guide

How to compare Craton Bolt against other SQL engines without lying to yourself or
your readers. This document is for engineers running an evaluation, contributors
adding head-to-head numbers, and anyone publishing a "Craton Bolt vs X" post.

It is **not** the place where Craton Bolt's own measured numbers live — those are in
[`BENCHMARKS.md`](./BENCHMARKS.md). This is the methodology and discipline doc
that any future numbers should follow.

## TL;DR

- Craton Bolt 0.1.x is a single-node, single-batch, in-memory GPU SQL engine. Pick
  competitors that occupy the same niche; treat cross-niche comparisons as
  illustrative, not damning.
- Use **ClickBench** as your primary suite: it's single-table, aggregation-heavy,
  and lands squarely inside what Craton Bolt can run today.
- Always run cold and warm separately. Always verify result equivalence. Always
  publish the queries you ran, not just the summary.
- Report **geometric mean** for the headline number, **per-query** for the
  detail. Never report only arithmetic mean.
- If you can't beat the CPU baseline on a workload, say so. Cherry-picked GPU
  wins are how the GPU-database category lost credibility the first time.

---

## 1. What category is Craton Bolt in?

Be specific about the contest before you measure. Craton Bolt 0.1.x is:

- **Single-node** (one process, one GPU).
- **Single-batch in-memory** (no streaming, no larger-than-VRAM tables, no
  spill).
- **JIT-compiled per-query** (no plan cache, no kernel cache beyond PTX).
- **Embedded-style API** (Rust crate; no server, no wire protocol).
- **SQL surface**: projection, filter, scalar arithmetic, `GROUP BY`,
  `DISTINCT`, `ORDER BY`, `LIMIT`, `HAVING`, `UNION`, `INNER JOIN ... ON
  <equi>`. See [`SQL_REFERENCE.md`](./SQL_REFERENCE.md) for the exact list.

These constraints determine which comparisons are *fair*. A comparison is fair
when Craton Bolt and the competitor can both execute the workload natively, with
neither system stretched outside its design center.

## 2. The competitor matrix

| System         | Niche overlap | Use for...                                                                 |
| -------------- | ------------- | -------------------------------------------------------------------------- |
| **DuckDB**     | High          | The default CPU baseline. Embedded, in-memory, columnar, very well-tuned. |
| **Polars**     | High          | The Rust-native CPU baseline. Already in `benches/query_benchmarks.rs`.    |
| **DataFusion** | High          | Rust CPU. Closest architecturally (plan / lower / execute).                |
| **ClickHouse** (`clickhouse-local`) | Medium | Single-node columnar; useful as a "fully optimised CPU" upper bound. |
| **HeavyDB**    | High          | The other production-GPU SQL engine. Closest direct competitor.            |
| **Spark RAPIDS** | Low (today) | Useful as a "distributed GPU" reference, but apples-to-oranges on a single node. |
| **Theseus** (Voltron Data) | Medium | Commercial; benchmark only if you have access.                  |
| **cuDF**       | Medium        | Library, not SQL — only fair if you write an equivalent imperative pipeline.|
| **BlazingSQL** | None          | Archived. Skip.                                                            |
| **Trino / Presto** | None on single node | Distributed-only design; meaningless single-node comparison. |
| **PostgreSQL / MySQL** | None  | Different category (transactional row store). Don't bother.                |

The systems in the "High" niche-overlap rows are the ones you must include
before claiming "Craton Bolt vs the world." DuckDB in particular is the rigorous
modern benchmark; if Craton Bolt doesn't beat DuckDB on a workload, the GPU isn't
buying anything *on that workload*, full stop.

## 3. Choosing the workload

### 3.1 Standard suites, ranked by fit

| Suite | Tables | Queries | Fits Craton Bolt 0.1.x?       | Comment |
| ----- | ------ | ------- | ------------------------- | ------- |
| **ClickBench**     | 1 (`hits`)   | 43 | Yes (most queries)     | Aggregation-heavy single-table. Ideal. |
| **SSB** (denormalised flat) | 1 | 13 | Yes              | Star Schema Benchmark; "denorm" form fits 0.1.x. |
| **SSB** (joined)   | 5            | 13 | Partial (needs INNER) | The 0.1.x INNER hash join handles SSB joins. |
| **TPC-H** SF1      | 8            | 22 | Partial (~6 queries)  | Many queries need LEFT JOIN, subqueries, CTEs. |
| **TPC-H** SF10+    | 8            | 22 | No                     | Multi-batch / streaming needed. |
| **TPC-DS**         | 24           | 99 | No                     | Out of scope until 0.3+.               |
| **YCSB**           | 1            | KV  | No                     | Wrong category (transactional).        |

**Default recommendation: ClickBench**. It's the modern standard, it's
single-table, it's aggregation-heavy, and the queries map cleanly onto Craton Bolt's
supported SQL. If you can run all 43 (or document which fail and why), you have
a credible result.

### 3.2 Microbenchmarks

Use these to *explain* a ClickBench result, not in place of one. The existing
`benches/query_benchmarks.rs` already covers projection, scalar arithmetic, and
filter; extend it for:

- **Scan throughput**: pure `SELECT col FROM t` at increasing row counts. The
  curve tells you where data-transfer cost stops dominating.
- **Per-operator break-even** (rows): the N below which the CPU baseline wins.
- **Group-by cardinality sweep**: 8, 256, 8K, 256K distinct keys. The hash
  kernel's behaviour changes dramatically across this range.
- **JIT compilation cost**: cold vs warm of the same query. Establishes the
  amortisation point.

### 3.3 Anti-patterns

Don't benchmark:

- **Queries no one writes.** `SELECT 1` tells you nothing.
- **One-row tables.** Launch overhead dominates everything.
- **Queries Craton Bolt doesn't support.** Use a different suite or reduce the
  query; do not paper over a parse error.
- **Result rendering or pretty-printing.** Time the engine's "result available"
  point, not stdout flush.

## 4. Methodology

### 4.1 Apples-to-apples

Same machine. Same data file. Same query text (modulo dialect differences,
which must be documented). Same row count. Same column types. Same NULL
density. If you change anything for one engine, change it for all engines or
disclose the asymmetry in the writeup.

### 4.2 Cold vs warm

Always report both. Define them up front:

- **Cold** = engine process starts fresh, data loaded from disk, all caches
  empty. Includes JIT / compilation. Includes any first-query setup. Runs once
  per measurement to avoid the OS page cache muddying the next iteration.
- **Warm** = data resident in engine memory, plan / kernel caches populated.
  Run **≥ 5 iterations after a discard-1 warmup**. Report p50 and p95.

Craton Bolt's warm path is where the GPU should win. Its cold path is where it
pays the worst tax (PTX assembly + module load + first H2D transfer). Both
matter; don't conflate them.

### 4.3 Statistics

- **Iterations**: 1 cold + 10 warm minimum. 1 cold + 30 warm preferred.
- **Headline**: geometric mean across the query set (arithmetic mean
  over-weights slow queries).
- **Per-query**: report median and p95. Add max if the distribution is bimodal.
- **Never** report only mean. Modes matter.
- **Confidence intervals**: bootstrap the geometric mean. If your CI overlaps
  the competitor's, you don't have a result, you have a tie.

### 4.4 Correctness verification

Run every query through every engine and **compare results**. A faster wrong
answer is not a result. Acceptable verification techniques:

- **Hash check**: sort the result rows and hash them. Compare hashes across
  engines. Cheap and catches most disagreements.
- **Full-row diff** for small results (< 10k rows).
- **Aggregate-of-aggregate**: `SUM(col) FROM (result)` matches across engines.
  Catches sign and overflow bugs.

Be ready to find disagreements. Float arithmetic isn't bit-exact across
engines; integer overflow may be promoted differently; NULL semantics in GROUP
BY can vary. Document the tolerance you used.

## 5. System control

Pin and disclose every one of these:

| Axis           | What to pin                                                       |
| -------------- | ----------------------------------------------------------------- |
| CPU            | Model, socket count, cores, base/boost clock, microcode rev       |
| RAM            | Capacity, channels, speed (e.g. DDR5-5600, 4ch)                   |
| GPU            | Model, VRAM, compute capability, driver version, CUDA version     |
| PCIe           | Gen and lane count (Gen3 x16 vs Gen4 x16 ≈ 2× transfer ceiling)   |
| Storage        | NVMe vs SATA vs spinning; relevant for cold-load timings          |
| OS             | Distro and kernel; matters for io_uring / huge pages              |
| Power state    | Performance governor, no thermal throttling, GPU persistence mode |
| Thread count   | Set explicit per-engine thread count; don't accept defaults blindly |
| NUMA           | Single-socket pin if applicable; otherwise document NUMA topology |

A common failure mode is comparing a tuned competitor to default Craton Bolt or
vice-versa. Set every engine's thread count and memory budget explicitly; cite
the values in your writeup.

## 6. What Craton Bolt will likely win and lose at

Setting reader expectations honestly is part of the deliverable. As of 0.1.x:

**Likely wins**
- Warm scalar arithmetic at ≥ 10M rows (GPU FLOPs + bandwidth ratio).
- Warm GROUP BY with moderate (8-bit to 16-bit) key cardinality.
- Aggregation-heavy single-table queries where data is resident (ClickBench
  warm path is the bullseye).

**Likely ties or losses**
- Cold-start queries: parse + JIT + first H2D often exceeds DuckDB's entire
  warm run on small data.
- Sub-million-row workloads: launch + transfer overhead dominates.
- Queries dominated by string operations: 0.1.x only does dictionary equality;
  `LIKE` / `SUBSTRING` / `CONCAT` aren't routed through SQL.
- Multi-table joins beyond the single equi-INNER pattern: 0.1.x rejects them
  at the parser.
- Anything that requires `IS NULL`, `IN`, `BETWEEN`, `CASE`, `NULLIF`,
  `COALESCE`, or `CAST` — these are unimplemented and will fail.

**Likely losses (by design, until 0.2+)**
- Streaming / multi-batch tables.
- Larger-than-VRAM datasets.
- Distributed queries.
- Anything from TPC-DS.

If you find Craton Bolt winning on a workload in the "Likely losses" bucket, that's
suspicious — recheck your correctness assertions before publishing.

## 7. Running each competitor

Concrete invocations to make competitor numbers reproducible. **Always** pin
thread counts and disable result caches.

### DuckDB

```bash
# Repl invocation; emits per-query timings.
duckdb -c "
  PRAGMA threads=$(nproc);
  PRAGMA enable_object_cache=true;
  PRAGMA enable_progress_bar=false;
  -- Disable result cache so warm-cache numbers reflect re-execution.
  .timer on
  -- Load data into a persistent in-memory table.
  CREATE TABLE hits AS SELECT * FROM read_parquet('hits.parquet');
  -- Then your query, run 11+ times; discard the first.
  SELECT count(*) FROM hits WHERE ...
"
```

### Polars (Rust)

```rust
let df = LazyFrame::scan_parquet("hits.parquet", Default::default())?
    .with_streaming(false)        // match Craton Bolt's in-memory model
    .collect()?;                  // materialise once before timing.

// Per warm iteration:
let t0 = Instant::now();
let _ = df.clone().lazy().filter(...).select(...).collect()?;
let dt = t0.elapsed();
```

Confirm Polars's thread pool is sized explicitly:
`POLARS_MAX_THREADS=$(nproc) cargo bench`.

### DataFusion

```rust
let ctx = SessionContext::new();
ctx.register_parquet("hits", "hits.parquet", Default::default()).await?;
// Warmup:
ctx.sql("SELECT 1").await?.collect().await?;
// Timed:
let t0 = Instant::now();
let _ = ctx.sql("SELECT ...").await?.collect().await?;
```

### ClickHouse (`clickhouse-local`)

```bash
clickhouse-local --query="
  CREATE TABLE hits ENGINE = Memory AS SELECT * FROM file('hits.parquet');
  SELECT ...
" --time
```

### HeavyDB

```bash
heavysql -p HyperInteractive -u admin
> \timing
> COPY hits FROM 'hits.csv' WITH (header='true');
> SELECT ...
```

Document `EXPLAIN` output. HeavyDB will reveal its plan, which lets readers
verify it's running the same shape as Craton Bolt.

### Craton Bolt

Use the existing harness shape:

```bash
BOLT_BENCH_GPU=1 cargo bench --bench query_benchmarks
```

For non-criterion runs (e.g. ClickBench), call `Engine::sql_and_execute` in a
loop with `Instant::now()` bracketing — see
`benches/query_benchmarks.rs` for the established pattern.

## 8. Recording and publishing results

A credible benchmark writeup includes, at minimum:

1. **System table** — every row from §5 filled in.
2. **Software table** — engine versions, build flags (release, LTO, native
   march), CUDA toolkit version, OS kernel.
3. **Workload definition** — exact suite, exact data file (with SHA-256),
   exact query text. Link to the queries; don't paraphrase them.
4. **Cold timings** — one number per (engine, query). Note that cold is a
   single sample by construction.
5. **Warm timings** — `[engine][query] = {p50, p95, max, n}`.
6. **Geometric mean** per engine, with bootstrap CI.
7. **Failures table** — per (engine, query): timeout, parse error, wrong
   answer, OOM. Don't hide them.
8. **Verification method** — which check from §4.4, what tolerance.
9. **Disclosures** — anything where you deviated from defaults: thread counts,
   memory limits, compiler flags, vendor-specific tuning.

Publish the harness code. If a reader can't run your benchmark, you have an
ad, not a benchmark.

## 9. Pitfalls

The single most common ways benchmarks lie. Avoid all of them.

- **Mixed cold and warm.** Reporting one engine's cold next to another's
  warm. Always state which, always match.
- **Forgotten result cache.** DuckDB, ClickHouse, and HeavyDB cache results
  by default. Disable explicitly.
- **OS page cache leaking.** A second engine "loads from disk" but the file is
  already in the page cache from the first. Drop caches (`sync; echo 3 >
  /proc/sys/vm/drop_caches`) between cold runs.
- **GPU warm-up not counted.** The first kernel after driver init pays a
  one-time tax. Either bracket it into "cold" or run a separate warm-up kernel
  before "cold" timing starts. Document either way.
- **Asymmetric data placement.** Craton Bolt holds data in VRAM after the first
  query; DuckDB holds it in OS memory after the first query. The "warm" point
  is engine-specific.
- **Different row counts.** ClickBench's `hits` table at 100M rows is *not*
  the same workload as `hits` at 1M. Pin row count and disclose it.
- **Compiler flag leak.** `cargo build` vs `cargo build --release` is a 5-50×
  difference. Always release. Always document.
- **Single-iteration warm.** One measurement is not a benchmark.
- **Cherry-picked queries.** Pre-commit to the query set before you start
  measuring.
- **Ignored failures.** "We skipped queries 17 and 22" is a failure, not a
  win. Report it.
- **Result hash skipped.** Faster + wrong is not faster. Always verify.
- **Old competitor version.** Pin to a release within ~6 months. Document.

## 10. See also

- [`BENCHMARKS.md`](./BENCHMARKS.md) — Craton Bolt's own measured numbers and the
  internal bench-suite description.
- [`SQL_REFERENCE.md`](./SQL_REFERENCE.md) — which SQL Craton Bolt actually
  supports today; constrains what's benchable.
- [`ROADMAP.md`](../ROADMAP.md) — known limitations, by design.
- [`benches/query_benchmarks.rs`](../benches/query_benchmarks.rs) — the
  existing micro-bench harness.
- [ClickBench](https://github.com/ClickHouse/ClickBench) — the recommended
  primary suite.
- [Star Schema Benchmark](https://www.cs.umb.edu/~poneil/StarSchemaB.pdf) —
  for join coverage as it lands.
