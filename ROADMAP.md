# Craton Bolt Roadmap

This document tracks intentional gaps in the current release and the milestones
planned beyond it. For day-to-day progress, see `CHANGELOG.md`. For supported
SQL today, see `docs/SQL_REFERENCE.md`. For the full 1.0 plan, see
`docs/PATH_TO_1.0.md`. To install and build, see `docs/INSTALL.md`.

## 0.9.0 (current — pre-production, API stabilising)

0.9 makes SQL semantics the default and hardens the runtime contract (see
`CHANGELOG.md` for the full list): SQL NULL semantics for integer / decimal
division, NULL grouped `AVG` over empty groups, three-valued literal `IN`; a
process-wide permit rejecting a second live `Engine`; MSRV 1.85; `cudarc` as
the supported CUDA adapter; planner-selected (not env-gated) GPU paths; the
public `ExecutionTier` contract; and a committed public-API snapshot.

The capability summary below was assembled across the 0.6 – 0.7 milestones
and is kept for reference:

- **`Decimal128` GPU arithmetic** (`+`, `-`, `*`) and comparisons (`=`, `!=`,
  `<`, `>`, `<=`, `>=`, reachable from `WHERE`); `SUM(Decimal128)` via
  host-side reduction.
- **`Date32` / `Timestamp` arithmetic** (Date−Date, Timestamp−Timestamp,
  Day-INTERVAL only) lowered to GPU.
- **Grouped `STDDEV` / `VAR`** under `GROUP BY` via per-group host-side
  Welford.
- **GPU radix sort integrated** into `src/exec/sort.rs` — single-key
  `Int32` / `Int64` ASC plus multi-key and `DESC`. Still opt-in via
  `BOLT_GPU_SORT=1` (see `docs/ENV_VARS.md`); not yet planner-selected by
  default.
- **`KernelSpec` module cache wired into call sites** — scalar aggregate,
  hash-join, radix-sort, and compaction kernels now hit the cache (skipping
  both codegen and PTXAS); async memcpy rolled out to the remaining
  `GROUP BY` variants and the `WHERE` filter D2H path.
- **WHERE-predicate type-checking** during SQL lowering.
- **SQL surface expansion** (set ops, windows, CTEs, subqueries, joins):
  `EXCEPT [ALL]` / `INTERSECT [ALL]` (host-side), non-recursive CTEs
  (`WITH`), host-side window functions (`ROW_NUMBER` / `RANK` /
  `DENSE_RANK` / `SUM` / `AVG` / `MIN` / `MAX` / `COUNT` `OVER`, default
  frame only), uncorrelated scalar and `[NOT] IN` subqueries,
  `JOIN ... USING` / `NATURAL JOIN`, and `COUNT(DISTINCT col)` (sole
  SELECT item). GPU `LIKE` (dict + non-dict `Utf8`) and GPU
  `UPPER` / `LOWER` / `LENGTH`; host-side `SUBSTRING` / `TRIM` / `CONCAT`.
- **SQL surface expansion (later 0.7 feature waves)**: `LATERAL` derived
  tables and plain derived tables (subquery in `FROM`); `WITH RECURSIVE`
  (linear, non-linear, and mutual recursion, optional column-list alias);
  a single correlated `WHERE` subquery (scalar comparison / `EXISTS` /
  `NOT EXISTS`); `VALUES` as a row source (bare and in `FROM`); the
  `generate_series(start, stop[, step])` table-valued function;
  `DISTINCT ON (...)`; named `WINDOW` clause + `QUALIFY`; super-aggregates
  (`GROUP BY ROLLUP` / `CUBE` / `GROUPING SETS` / `ALL`, `WITH TOTALS` /
  `ROLLUP` / `CUBE`, and `GROUPING()` / `GROUPING_ID()`); and query-clause
  sugar (`FETCH` / T-SQL `TOP` → `LIMIT`, `FOR UPDATE` / `FOR SHARE`
  no-op, `PREWHERE` → `WHERE`). The correlated-`WHERE` and `LATERAL`
  nested-loop apply paths are bounded by `CRATON_MAX_APPLY_ROWS`.
- **Grouped `Decimal128` GPU aggregation** — `SUM` / `MIN` / `MAX` over a
  `Decimal128` column under `GROUP BY` lowered on-device, plus
  `Decimal128` division added to the GPU arithmetic set (complementing the
  scalar `Decimal128` arithmetic and aggregation above).

### What works (carried forward from 0.5)

- SQL → PTX → execution end-to-end for projection, filter, scalar
  aggregate, and GROUP BY (single/multi-column, packed and wide keys).
- `DISTINCT`, `LIMIT [OFFSET]`, `ORDER BY [ASC|DESC]`, `HAVING`,
  `UNION [ALL]`, `EXCEPT [ALL]`, and `INTERSECT [ALL]` (host-side
  executors for the non-GROUP-BY paths). Non-recursive CTEs (`WITH`).
- `INNER`, `LEFT [OUTER]`, `RIGHT [OUTER]`, `FULL [OUTER]`, and `CROSS`
  joins (GPU fast path + host hash-join fallback), with `ON` /
  `USING (...)` / `NATURAL` constraints. Multiple joins per `SELECT` are
  permitted.
- Host-side window functions (`OVER`) and uncorrelated scalar / `[NOT] IN`
  subqueries.
- Borrow-checked GPU memory primitives (`GpuVec` / `GpuView` /
  `GpuViewMut`) — use-after-free, double-free, and mutable/shared
  aliasing across kernel boundaries are compile-time errors.
- The full v0.5 SQL scalar surface (`NOT`, `IN`, `BETWEEN`, `CASE`,
  `CAST`, `COALESCE` / `NULLIF`, `LIKE`, `||`, `STDDEV` / `VAR`,
  scalar string fns) — parsed and type-checked. v0.7 landed GPU
  execution for grouped `STDDEV` / `VAR`, `Decimal128` arithmetic and
  comparisons, and `Date` / `Timestamp` arithmetic; the remaining items
  (e.g. `CASE` / `CAST` / scalar string funcs on the GPU, `LIKE` with
  `ESCAPE`, `||` in `WHERE`) still reject cleanly at physical lowering.
- Dictionary-encoded Utf8, float GROUP BY with sentinel-free fallback,
  GPU-side filter compaction, process-wide PTX module cache,
  `--features cuda-stub` for CI / `docs.rs`.

### New in 0.6.0 — M1 (Foundation)

- `Engine::register_table_stream(name, schema, iter)` — eager
  implementation in v0.6, signature future-compatible with the lazy
  streaming path scheduled for v0.7.
- Async memcpy + pinned host buffers piloted in the scalar aggregate
  executor (`upload_primitive_values_async`).
- `KernelSpec`-keyed module cache built and unit-tested in
  `src/exec/module_cache.rs` (skips both codegen and PTXAS on a hit).

### New in 0.6.0 — M3 (Join + Sort)

- GPU radix-sort kernel scaffold for `Int32` / `Int64` in
  `src/jit/sort_kernel_radix.rs`. Env-gated via `BOLT_GPU_SORT=1`.
- Non-equi join via nested-loop in
  `src/exec/join.rs::execute_nested_loop_join` (INNER only; cap
  `MAX_NESTED_LOOP_INNER_ROWS = 1024`).

### New in 0.6.0 — M4 (Types)

- `DataType::Decimal128(p, s)` plumbed end-to-end through plan + Arrow
  round-trip; `CAST(int AS DECIMAL(p, s))` parses.
- `DataType::Date32` and `DataType::Timestamp(TimeUnit, Option<&'static str>)`
  with a `TimeUnit` enum. `DATE '...'` and `TIMESTAMP '...'` literals
  parse. Timezones interned via `intern_timezone` so `DataType` stays
  `Copy`.

### New in 0.6.0 — M5 (Observability + ergonomics)

- `tracing` dependency; spans on parse / plan / lower / codegen /
  ptx_load / launch / transfer / materialize. Off by default; opt-in
  via the consumer's `tracing_subscriber`. Catalogue in
  `src/observability.rs`.
- `BoltError` is now `#[non_exhaustive]` and gains a
  `SqlWithSpan { msg, span: Range<usize> }` variant plus a
  `BoltError::span()` accessor. sqlparser parse errors wrapped via
  `parse_error_to_bolt_error`.
- Did-you-mean suggestions in `Schema::index_of`,
  `NameResolver::resolve_compound`, and `try_aggregate`. Shared helper
  in `src/plan/suggest.rs` (Levenshtein capped at 2).

### New in 0.6.0 — M6 (Performance)

- Disk-backed PTX cache in `src/jit/disk_cache.rs`. Opt-in via the
  `BOLT_PTX_CACHE_DIR=/path` env var; writes are atomic.
- Criterion regression scaffold in `benches/regression.rs`, alongside
  `query_benchmarks.rs`, `olap_benchmarks.rs`, and `compute_benchmarks.rs`.

### New in 0.6.0 — M7 (API stabilization)

- `Engine::Builder` (`EngineBuilder`) with `device`, `memory_budget`,
  `persistent_cache`, `enable_tracing` knobs. `Engine::new` /
  `Engine::new_with_device` preserved as thin wrappers. `Engine` is
  now `#[non_exhaustive]`.
- `DataFrame::collect(self, &mut Engine)` materializes through the new
  `Engine::run_logical_plan`. The 0.1-era `#[doc(hidden)]` tombstone
  is gone.
- `PlanRewrite` trait in `src/plan/rewrite.rs`. `Engine` stores
  `rewrites: Vec<Box<dyn PlanRewrite>>` and threads them through
  `Engine::sql` immediately before `lower_physical`.
  `Engine::with_rewrite(self, r)` registers a rewrite.
- `docs/API_SURFACE.md` enumerates the public surface by stability
  tier.

### New in 0.6.0 — M8 (Freeze prep) + Docs

- `docs/MIGRATION_GUIDE.md` covers the 0.3 → 0.5 → 0.6 upgrade path.
- `docs/USER_GUIDE.md` ships as a 10-minute tutorial.

### Known limitations (not bugs) — as of 0.9.0

v0.9 closed most of the v0.8 carry-overs and security/correctness findings. What remains:

- String and relational shapes span `Gpu`, `Host`, and `Hybrid`; callers can
  inspect `QueryHandle::planned_execution_tier()` instead of inferring a tier.
- The planner selects the GPU radix sort by default for supported key shapes;
  `BOLT_GPU_SORT` is a tri-state diagnostic override.
- The disk PTX cache honors both `BOLT_PTX_CACHE_DIR` and
  `EngineBuilder::persistent_cache`, with the builder taking precedence.
- `Engine::register_table_stream` drains eagerly (one validated, coalesced
  install); `register_table_stream_lazy` defers the producer to the first
  query that references the table. Both end host-resident (see
  `docs/LIMITATIONS.md`).

### On `main` since 0.9.0 (see CHANGELOG `[Unreleased]`)

- One compile pipeline (`QueryCompiler`) for `sql`, `explain_sql`,
  `run_logical_plan` and host-orchestrated subplans; SQL classified once.
- Lazy streams drain only referenced tables and enforce the declared schema.
- Multi-batch tables coalesced at ingest (no per-query concatenation).
- Aggregates over joins / unions / host-only filters (`AggregateOverInput`);
  string-literal aggregates and `Utf8` projections under a predicate fixed.
- Cost-model benchmark suite, reproducibility manifests, scheduled GPU
  benchmark history; generated module inventory and CI matrix docs.
- Storage layer: Parquet tables and a `StorageTable` contract with zone-map
  pruning and statistics; SQL DDL / DML / `COPY` / `ANALYZE`; a persistent,
  snapshot-versioned catalog; materialized views.
- Memory governor (host / device budgets, eviction) and out-of-core
  `GROUP BY` / `DISTINCT` / grace hash join / external sort through spill
  files; a shared runtime (`Database`) serving concurrent `Session`s.
- `DELETE` / `UPDATE` / `ALTER TABLE` and `BEGIN` / `COMMIT` / `ROLLBACK`
  (one catalog snapshot per transaction, session-owned through a
  `Database`); skew repartitioning, chunked over-budget scans and scalar
  aggregates, `Engine::sql_stream`; an opt-in cost-based CPU / GPU tier
  choice (`CostModel`, calibration, `EXPLAIN` estimates); SQL NULL semantics
  in the fused GPU kernels.

## Beyond 0.9 — toward 1.0 (next)

With the v0.8 execution carry-overs largely landed in v0.9, the
remaining pre-1.0 work is the last GPU-lowering gaps, planner-driven
dispatch, and the freeze checklist:

### Goals

- **Broaden GPU-native coverage** for host-only string, window, set-op, and
  dedup shapes while preserving the explicit execution-tier contract.
- **Planner cost refinement** beyond the current dtype/cardinality radix-sort
  heuristic.
- **Security audit prep (M8 from `docs/PATH_TO_1.0.md`)** — dependency
  audit, public-surface review, and the freeze checklist needed
  before the 1.0 stabilisation window opens.

### Stretch goals

- Wider GPU join coverage beyond the current gated fast paths.
- GPU lowering for `||` in `WHERE` predicates.

## 1.0 — public API freeze

See [`docs/PATH_TO_1.0.md`](./docs/PATH_TO_1.0.md) for the detailed
milestone-by-milestone plan, acceptance criteria, open decisions, and
explicit exclusions. Headlines:

- All `#[doc(hidden)]` IR types (`PhysicalPlan`, `KernelSpec`,
  `AggregateSpec`, `Op`, `Reg`, `Value`, `ColumnIO`) either stabilised
  or replaced with a public builder surface.
- `DataFrame::collect()` becomes a real materialising terminal.
  (Landed in v0.6 — kept here for the 1.0 acceptance checklist.)
- Stable `Engine::sql` contract; `cuda-stub` feature documented as a
  permanent CI helper rather than an experiment.
- Multi-platform: `aarch64-linux` (Jetson) tested in CI.
- Regression-CI green; ClickBench numbers published per release.
