# Craton Bolt Roadmap

This document tracks intentional gaps in the 0.6.0 release and the milestones
planned for 0.7+. For day-to-day progress, see `CHANGELOG.md`. For supported
SQL today, see `docs/SQL_REFERENCE.md`. For the full 1.0 plan, see
`docs/PATH_TO_1.0.md`.

## 0.6.0 (current — pre-production, API stabilising)

This release covers milestones M1 (foundation), M3 (join + sort), M4 (types),
M5 (observability + ergonomics), M6 (performance), M7 (API stabilization), and
M8 (freeze prep) from `docs/PATH_TO_1.0.md`. Where v0.5 brought the SQL surface
up to "table stakes", v0.6 turns to the execution-layer plumbing, the type
system, and the public-API shape that 1.0 will freeze.

### What works (carried forward from 0.5)

- SQL → PTX → execution end-to-end for projection, filter, scalar
  aggregate, and GROUP BY (single/multi-column, packed and wide keys).
- `DISTINCT`, `LIMIT [OFFSET]`, `ORDER BY [ASC|DESC]`, `HAVING`, and
  `UNION [ALL]` (host-side executors for the non-GROUP-BY paths).
- `INNER`, `LEFT [OUTER]`, `RIGHT [OUTER]`, `FULL [OUTER]`, and `CROSS`
  joins (host-side hash join). Multiple joins per `SELECT` are
  permitted.
- Borrow-checked GPU memory primitives (`GpuVec` / `GpuView` /
  `GpuViewMut`) — use-after-free, double-free, and mutable/shared
  aliasing across kernel boundaries are compile-time errors.
- The full v0.5 SQL scalar surface (`NOT`, `IN`, `BETWEEN`, `CASE`,
  `CAST`, `COALESCE` / `NULLIF`, `LIKE`, `||`, `STDDEV` / `VAR`,
  scalar string fns) — parsed and type-checked; physical lowering
  rejects cleanly for the items still pending a GPU runtime path.
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
- Criterion regression bench scaffold in `benches/regression.rs`
  covering scalar agg / GROUP BY / filter at parse / lower / ptx_gen.

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

### Known limitations (not bugs)

- The v0.5 SQL scalar items that parse but reject at the physical
  layer still do so in v0.6 (carried over to v0.7 — see below).
- `Decimal128`, `Date32`, and `Timestamp` parse and type-check, but
  arithmetic and the GPU lowering paths reject cleanly.
- The async-memcpy and `KernelSpec`-cache improvements ship as the
  pilot + library only; the broader executor wiring is a v0.7 task.
- The GPU radix-sort kernel is gated behind `BOLT_GPU_SORT=1` and not
  yet selected by the planner.
- The disk PTX cache honours `BOLT_PTX_CACHE_DIR`; the
  `EngineBuilder::persistent_cache` knob is wired through the builder
  surface but does not yet drive `EngineBuilder::build`.

## 0.7+ — execution catch-up (next)

Most of the v0.6 work was either type-system, observability, or
infrastructure (cache, builder, rewrite trait, regression bench). The
v0.7 cycle is about turning the new surfaces into running GPU code:

### Goals

- **GPU lowering for the deferred scalar items**: `CASE WHEN ... END`
  (predicated select), `CAST` over documented primitive pairs, scalar
  string functions (`UPPER` / `LOWER` / `LENGTH` / `SUBSTRING`),
  `Decimal128` arithmetic (`+` `-` `*`), and `Date` / `Timestamp`
  arithmetic.
- **`STDDEV` / `VAR` under `GROUP BY`** via per-group Welford state.
- **Per-shape async memcpy wiring** — extend the scalar-aggregate
  pilot to filter, GROUP BY, and join executors on explicit streams +
  pinned host buffers.
- **`KernelSpec` cache integration** at the call sites that still
  rebuild PTX on every query.
- **GPU radix sort integration** in `src/exec/sort.rs` — promote the
  v0.6 scaffold from env-var gated to planner-driven dispatch.
- **`EngineBuilder::persistent_cache` wiring** through
  `EngineBuilder::build` (today the env-var path is the only
  honoured surface).
- **Security audit prep (M8 from `docs/PATH_TO_1.0.md`)** — dependency
  audit, public-surface review, and the freeze checklist needed
  before the 1.0 stabilisation window opens.

### Stretch goals

- GPU hash join (the existing executor is host-side; a GPU-resident
  probe path is the natural next step).
- GPU lowering for `LIKE` with `ESCAPE` and `||` in `WHERE`
  predicates.

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
