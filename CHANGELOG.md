# Changelog

All notable changes to this project will be documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project tries to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it leaves `0.x`.

## Note on version 0.2.0

There is no `0.2.0` release. The project jumped from `0.1.0` (2026-05-23) directly to `0.3.0` (2026-05-26) — a three-day span in which the scope grew well past what a single minor bump could honestly carry (multi-batch tables, INNER JOIN, DISTINCT / LIMIT / ORDER BY / HAVING / UNION, real `cuda-stub`, PTX cache, CI). Tagging an intermediate `0.2.0` would have been a paper milestone, so the version number was reserved and skipped.

## [Unreleased]

### Performance
- **Tier-2 group-by host slot-walk** (`exec::groupby_tier2_common::collect_populated_slots_sorted`):
  the post-reduce collection over the fixed `NUM_PARTITIONS × BLOCK_GROUPS`
  (~4.2M-entry) slot buffer now uses a fused single-pass serial scan (pre-sized
  output `Vec`, hoisted bounds) and, above a 256K-slot threshold, a
  `std::thread::scope` parallel scan (no new dependency). Output is byte-identical
  to the previous serial implementation (ordered chunk concatenation followed by the
  same stable sort), proven by a 1M-slot parity test; the collector's generic bound
  widened to `T: Copy + Send + Sync`.

### Changed
- **De-duplicated `StreamSet`** — `cuda::async_copy` now consumes the canonical
  `cuda::buffer::StreamSet` instead of carrying its own identical copy (~46 LOC
  removed). Stream tracking, `Drop` fencing, and the event-based deferred-free path
  are unchanged; `buffer::StreamSet` gained only additive `pub(crate)` accessors.

### Internal
- Documented the GPU-only performance items deliberately deferred pending on-hardware
  benchmarking (AVG sum+count reduce fusion, device-side compaction before the 52 MiB
  group-by D2H, adaptive spin back-off, pinned-memory pool) — see
  `reviews/PERF_BACKLOG.md`. These change device behavior or emitted PTX and cannot be
  validated under the `cuda-stub` + host-oracle CI used on this branch.

## [0.7.0] - 2026-05-29

v0.7 turns the v0.6 carry-overs into live code paths. The themes are
the same three the v0.6 closing notes scheduled for "v0.7+": wiring the
`KernelSpec` module cache into real call sites, lighting up the
`Decimal128` / `Date` / `Timestamp` GPU lowering boundaries, and landing
the GPU radix sort dispatch in the executor. It also widens the SQL
surface itself — set operations (`EXCEPT` / `INTERSECT`), host-side
window functions, non-recursive CTEs, uncorrelated subqueries,
`JOIN ... USING` / `NATURAL`, `COUNT(DISTINCT)`, and GPU `LIKE` /
`UPPER` / `LOWER` / `LENGTH`. Items are grouped to mirror the v0.6
milestone headings.

### Added — Types (Decimal128 / Date / Timestamp)
- **`Decimal128` GPU arithmetic** — dual-register IR (`Op::*128` +
  `RegAlloc::assign_pair`), `GpuColumnData` ingest, and `Codegen` wiring
  so `+`, `-`, `*` are reachable from lowering.
- **`Decimal128` comparisons** (`=`, `!=`, `<`, `>`, `<=`, `>=`) lowered
  to GPU and reachable from `WHERE` predicates.
- **`SUM(Decimal128)`** via host-side reduction.
- **`Date32` / `Timestamp` arithmetic** (Date−Date and
  Timestamp−Timestamp; Day-INTERVAL only) lowered to GPU.

### Added — Aggregates
- **Grouped `STDDEV` / `VAR`** under `GROUP BY` via per-group host-side
  Welford.

### Added — Join + Sort
- **GPU radix sort integration** in `src/exec/sort.rs` — single-key
  `Int32` / `Int64` ASC, plus multi-key and `DESC` support. Fixes the
  `DESC` pre-transform to use `!(val ^ MIN)` rather than a bare `!val`.

### Changed — KernelSpec module cache wiring
- **`KernelSpec` extended** to model aggregate / join / sort /
  compaction kernel kinds (sibling spec types).
- **Cache wired into call sites**: `ScalarAggSpec` (scalar reduction),
  `HashJoinKernelSpec` (10 `gpu_join` call sites), `RadixSortKernelSpec`
  (4 `gpu_sort` call sites), and `CompactionKernelSpec` (6 compaction
  kernels: prefix-scan + gather). Cache hits skip both codegen and
  PTXAS.
- **Async memcpy** rolled out to the remaining `GROUP BY` variants
  (tier2 / shmem / wide / valid) and async D2H for
  `compact::download_mask` (the `WHERE` filter path).

### Changed — Lowering / validation
- **WHERE predicate type-checking** during SQL lowering (fixes `LIKE` on
  a non-`Utf8` column).

### Added — SQL surface (set ops, windows, CTEs, subqueries, joins)
- **`EXCEPT [ALL]` / `INTERSECT [ALL]`** — lowered to a binary
  `LogicalPlan::SetOp` node and executed **host-side** by
  `src/exec/setops.rs`. Set forms return distinct left rows; multiset
  (`ALL`) forms follow the SQL-standard `max(0, lc - rc)` /
  `min(lc, rc)` multiplicities. Row equality reuses the `DISTINCT`
  executor's row-key machinery (NULLs not distinct; `±0.0`
  canonicalised). `UNION` / `EXCEPT` / `INTERSECT BY NAME` rejected.
- **Window functions (`OVER`)** — host-side executor
  (`src/exec/window.rs`) for `ROW_NUMBER`, `RANK`, `DENSE_RANK`, and
  `SUM` / `AVG` / `MIN` / `MAX` / `COUNT` aggregate windows. Default
  `RANGE UNBOUNDED PRECEDING AND CURRENT ROW` frame only; window
  functions must be top-level SELECT items. Explicit / non-default
  frames, named `WINDOW`, `QUALIFY`, and `COUNT(DISTINCT) OVER` rejected.
- **Non-recursive CTEs (`WITH name AS (...)`)** — lowered against the
  left-to-right CTE scope and type-checked at the definition site.
  `WITH RECURSIVE`, CTE column-list aliases, and the materialization
  hint rejected.
- **Uncorrelated subqueries** — scalar `(SELECT ...)` and
  `[NOT] IN (SELECT ...)` in `SELECT` / `WHERE`, resolved to constants
  before physical lowering (`src/exec/subquery_resolve.rs`). Scalar
  `>1` row errors; `IN` folds to an `OR`/`AND` chain. Correlated
  subqueries, `EXISTS`, and derived tables in `FROM` rejected.
- **`JOIN ... USING (...)` / `NATURAL JOIN`** — desugared to equi
  `left.col = right.col` pairs and run through the existing join paths.
  Missing / ambiguous / duplicate `USING` columns and a `NATURAL` join
  with no common column are rejected.
- **`COUNT(DISTINCT col)`** — supported as the sole SELECT item (no
  `GROUP BY` / `HAVING` / `SELECT DISTINCT`), lowered to
  `COUNT(*) ∘ Distinct ∘ Project([col]) ∘ Filter(col IS NOT NULL)` and
  executed via the new `PhysicalPlan::CountRows` node.

### Added — GPU string functions
- **GPU `LIKE` / `NOT LIKE`** over `Utf8` columns — dictionary columns
  via dictionary-precompute → index membership; non-dictionary `Utf8`
  via the new `PhysicalPlan::StringLikeFilter` device matcher
  (`compile_like_match_kernel`, EXACT / PREFIX / SUFFIX / CONTAINS).
  Host `host_like` fallback retained.
- **GPU `UPPER` / `LOWER`** — two-pass variable-width device output via
  `PhysicalPlan::StringProject`.
- **GPU `LENGTH`** — `PhysicalPlan::StringLength` (dictionary-gather,
  `Int64` output).
- **Host-side `SUBSTRING` / `TRIM`** (`TRIM BOTH` / `LEADING` /
  `TRAILING`) executed end-to-end through a host projection.
- **GPU `NOT`** in a predicate lowered via `Op::Not`.

### Internal
- **Schema-converter consolidation** — the plan↔Arrow schema converters
  are unified into `exec::schema_convert`.
- **`ScalarAggSpec` dedup** (collision between two sibling-spec
  additions); field references updated (`dtype` → `input_dtype`).
- Radix dispatch gate tests serialized via an override hook to remove
  env-var test contention; dead single-key wrapper / warning cleanup.

## [0.6.0] - 2026-05-28

This release covers milestones M1 (foundation), M3 (join + sort), M4
(types), M5 (observability + ergonomics), M6 (performance), M7 (API
stabilization), and M8 (freeze prep) from `docs/PATH_TO_1.0.md`. v0.5
brought the SQL surface up to "table stakes"; v0.6 turns to the
execution-layer plumbing, the type system, and the public-API shape
that 1.0 will freeze. Many of the new code paths are present but
intentionally not yet wired into the default execution hot path — see
the closing paragraph for the explicit carry-overs.

### Added — M1 (Foundation)
- **`Engine::register_table_stream(name, schema, iter)`** in
  `src/exec/engine.rs`. v0.6 ships an eager implementation that drains
  the iterator into the existing in-memory table representation; the
  signature is future-compatible with a truly-lazy streaming path so
  callers won't need to rewrite their code when the lazy executor
  lands.
- **Async memcpy + pinned host buffers** piloted in the scalar
  aggregate executor (`src/exec/aggregate.rs::upload_primitive_values_async`).
  Per-shape rollout to the other executors is deferred to v0.7.
- **`KernelSpec`-keyed module cache** in `src/exec/module_cache.rs`,
  built and unit-tested. The cache skips both codegen and PTXAS on a
  hit. Call-site wiring is deferred to v0.7.

### Added — M3 (Join + Sort)
- **GPU radix-sort kernel scaffold** for `Int32` and `Int64` in
  `src/jit/sort_kernel_radix.rs`. Env-gated via `BOLT_GPU_SORT=1`; not
  integrated into `src/exec/sort.rs` yet (that wiring is a v0.7 task).
- **Non-equi join via nested-loop** in
  `src/exec/join.rs::execute_nested_loop_join`. INNER only, capped at
  `MAX_NESTED_LOOP_INNER_ROWS = 1024`. Closes the long-standing
  non-equi gap for small-cardinality cases.

### Added — M4 (Types)
- **`DataType::Decimal128(p, s)`** plumbed end-to-end through the
  logical plan + Arrow round-trip. `Literal::Decimal128` carried
  through the parser and type-checker. `CAST(int AS DECIMAL(p, s))`
  parses; GPU codegen rejects cleanly with `"Decimal128 not yet
  lowered to GPU"` until the runtime path lands.
- **`DataType::Date32`** and **`DataType::Timestamp(TimeUnit,
  Option<&'static str>)`** with a `TimeUnit` enum. `Literal::Date32(i32)`
  and `Literal::Timestamp(i64, unit, tz)`. `DATE '...'` and
  `TIMESTAMP '...'` literals parse. Timezones are interned via
  `crate::plan::logical_plan::intern_timezone` so `DataType` stays
  `Copy`.

### Added — M5 (Observability + ergonomics)
- **`tracing` crate dependency** with spans on the full
  parse / plan / lower / codegen / ptx_load / launch / transfer /
  materialize pipeline. Span names catalogued in
  `src/observability.rs`. Off by default; opt-in via the consumer's
  `tracing_subscriber`.
- **`BoltError` is now `#[non_exhaustive]`** and gains a
  `SqlWithSpan { msg, span: Range<usize> }` variant plus a
  `BoltError::span()` accessor. sqlparser parse errors are wrapped
  via `parse_error_to_bolt_error` in `src/plan/sql_frontend.rs`.
- **Did-you-mean suggestions** in `Schema::index_of`,
  `NameResolver::resolve_compound`, and `try_aggregate`. Backed by a
  shared Levenshtein helper in `src/plan/suggest.rs` (edit distance
  capped at 2).

### Added — M6 (Performance)
- **Disk-backed PTX cache** in `src/jit/disk_cache.rs`. Opt-in via the
  `BOLT_PTX_CACHE_DIR=/path` env var or a builder hook. Writes are
  atomic (`tempfile` + rename) so a partially-written cache entry
  can't poison subsequent runs.
- **Criterion regression bench scaffold** in `benches/regression.rs`.
  Three queries (scalar aggregate, GROUP BY, filter) measured at
  parse / lower / ptx_gen. cuda-stub invocation is documented; a >5%
  slowdown convention is established for the regression workflow.

### Added — M7 (API stabilization)
- **`Engine::Builder` (`EngineBuilder`)** with knobs for `device`,
  `memory_budget`, `persistent_cache`, and `enable_tracing`.
  `Engine::new` and `Engine::new_with_device` are preserved as thin
  wrappers over the builder. `Engine` is now `#[non_exhaustive]` so
  future fields don't break downstream destructuring.
- **`DataFrame::collect(self, engine: &mut Engine) -> BoltResult<RecordBatch>`** —
  the `#[doc(hidden)]` tombstone is gone; `collect` now materializes
  through the new `Engine::run_logical_plan` entry point.
- **`PlanRewrite` trait** in `src/plan/rewrite.rs`. `Engine` stores
  `rewrites: Vec<Box<dyn PlanRewrite>>` and threads them through
  `Engine::sql` immediately before `lower_physical`. Builder /
  fluent hook: `Engine::with_rewrite(self, r) -> Self`.
- **`docs/API_SURFACE.md`** enumerates the public surface by
  stability tier, distinguishing the items 1.0 will freeze from the
  ones still subject to change.

### Added — M8 (Freeze prep)
- **`docs/MIGRATION_GUIDE.md`** — covers `0.3 → 0.5 → 0.6` upgrade
  paths.

### Added — Docs
- **`docs/USER_GUIDE.md`** — 10-minute-tutorial structure aimed at
  first-time users.

### Notes — intentionally NOT in v0.6 (carry-overs for v0.7+)

The following items parse and type-check in v0.6 but reject at the
GPU lowering boundary; the runtime paths are scheduled for v0.7+:

- GPU lowering for `CASE`, `CAST`, scalar string funcs, `LIKE` with
  `ESCAPE`, `||` in `WHERE` predicates, grouped `STDDEV` / `VAR`,
  `Decimal128` arithmetic, and `Date` / `Timestamp` arithmetic.
- Per-executor async-memcpy wiring beyond the scalar aggregate pilot.
- `KernelSpec` cache integration into call sites (the cache is built
  and unit-tested; wiring is deferred).
- GPU radix sort integration in `src/exec/sort.rs` (the kernel
  scaffold exists; the dispatch is gated behind an env var and not
  yet selected by the planner).
- Disk PTX cache wiring through `EngineBuilder::build` — the env-var
  path works today, but the builder knob is not yet honored.

## [0.5.0] - 2026-05-28

This release covers the M2 milestone from `docs/PATH_TO_1.0.md`: SQL scalar
completeness. Version 0.4 is skipped — the M1 foundation work (streaming
tables, async Stage 2, KernelSpec cache) is deferred to a later release;
this cut focuses on bringing the SQL surface up to "table stakes" while
keeping the existing in-memory execution model.

### Added — SQL scalar surface
- **`NOT <bool-expr>`** — new `UnaryOp::Not` variant routed through the
  host-side filter path (GPU lowering is a follow-up).
- **`<expr> [NOT] IN (v1, v2, …)`** — desugared to an OR/AND chain of
  element-wise comparisons. Capped at 64 values; a large-list hash probe
  is a follow-up.
- **`<expr> [NOT] BETWEEN low AND high`** — desugared to
  `(expr >= low) AND (expr <= high)` (or the DeMorgan inverse).
- **`CASE WHEN cond THEN val [WHEN…] [ELSE val] END`** — both plain and
  simple (with-operand) forms. Type-check unifies numeric arms via
  `unify_numeric` and requires exact match for non-numeric. Physical
  lowering rejects cleanly with "CASE not yet lowered to GPU".
- **`CAST(expr AS type)`** — primitive numeric and boolean pairs only.
  Physical lowering rejects cleanly until the runtime conversion lands.
- **`COALESCE(a, b, …)`** and **`NULLIF(a, b)`** — desugared to `CASE`.
- **`<expr> [NOT] LIKE 'pattern'`** — constant-pattern LIKE with `%` and
  `_` wildcards. Routes through the host-side `host_like` evaluator;
  fast paths for prefix / suffix / contains / exact shapes.
- **String concat `a || b`** — new `BinaryOp::Concat` operator, lowered
  through the host-side `PhysicalPlan::Project` executor for SELECT
  positions. WHERE-clause concat is rejected with a clear message.
- **`STDDEV_POP`, `STDDEV_SAMP`, `STDDEV`** aggregates (Welford on host).
  Scalar-aggregate only; GROUP BY support is a follow-up.
- **`VAR_POP`, `VAR_SAMP`, `VARIANCE`** aggregates (shared Welford state).
- **`UPPER`, `LOWER`, `LENGTH`, `SUBSTRING`, `CONCAT`** scalar functions
  surfaced via `Expr::ScalarFn`. Parser + type-check only; physical
  lowering rejects each with a "follow-up" message.

### Added — SQL ergonomics
- **Aggregate aliasing** (`SELECT SUM(x) AS total`) — the alias carries
  through the post-Aggregate Project and is visible to HAVING / ORDER BY.
- **Qualified column references** (`t.col`, `alias.col`) — resolved
  against the FROM-tree, including JOIN aliases. Schema-qualified
  three-part names are rejected with a dedicated message.
- **Post-aggregate scalar expressions** (`SUM(x) + 1`, `AVG(qty) * 2`,
  `(SUM(a) + SUM(b)) / 2`) — extracted as aggregate feeds + rewritten
  surface expression in a post-Aggregate Project.
- **Case-insensitive identifiers** — unquoted SQL idents fold to
  lowercase at parse time; schema lookup falls back to case-insensitive
  match when the lookup name is all-ASCII-lowercase. Quoted
  (`"MyCol"`) identifiers preserve case and match verbatim.

### Added — M1 foundation
- **Validity propagation through primitive scalar aggregates** —
  `COUNT(col)` now excludes NULLs via the bitmap; `SUM`/`MIN`/`MAX`/`AVG`
  host-strip NULL positions before the GPU reduction. The zero-null
  fast path (`null_count == 0`) remains a zero-copy `primitive_to_gpu`
  upload.

### Notes
- The execution surface remains conservative: many of the items above
  parse and type-check, but the physical layer rejects them with a clear
  "not yet lowered to GPU" message until the corresponding kernel /
  host-side runtime path lands in a follow-up. The intent is to unblock
  third-party tooling (which can now generate the SQL it would naturally
  write) without claiming false execution coverage.
- This release also skips the original 0.4 milestone (streaming /
  async-memcpy Stage 2 / KernelSpec cache) for the same reason 0.2.0 was
  skipped: scope grew past what a single minor bump could carry, and
  scalar completeness was the more user-visible delta.

## [0.3.0] - 2026-05-26

### Added
- **`INNER JOIN ... ON <equi predicate>`** — host-side hash join.
  Recursively executes both sides, builds a
  `HashMap<JoinKey, Vec<row_idx>>` on the smaller input, probes the
  larger, and materialises matches via `arrow::compute::take`. One join
  per `SELECT`; LEFT / RIGHT / FULL / CROSS and non-equi predicates are
  rejected at the parser. NULL keys never match (SQL
  `NULL = NULL → UNKNOWN`).
- **`DISTINCT`, `LIMIT [OFFSET]`, `ORDER BY [ASC|DESC]`, `HAVING`,
  `UNION [ALL]`** — full plan + parser + standalone executors
  (`src/exec/{distinct,sort,limit}.rs`). HAVING desugars to a `Filter`
  over the `Aggregate`; plain `UNION` lowers to `Distinct(Union(..))`,
  `UNION ALL` stays a flat `Union`. Executors are host-side for 0.3.x.
- Multi-batch tables: the engine accepts more than one `RecordBatch` per
  registered table and threads them through the new operators (was:
  single-batch only).
- Validity propagation through `compact` / `gpu_compact`: filter
  selection masks now carry per-row validity for downstream consumers.
- Warp-shuffle reduction path in `agg_kernels.rs` for the last 5 strides
  of the agg-kernel tree (replaces the all-stride `__syncthreads` +
  shared-memory reduction the TODO marker called out).
- 13 new offline e2e tests in `tests/e2e_tests.rs` covering the new
  operators: 9 for DISTINCT / LIMIT / ORDER BY / HAVING / UNION (plan
  shapes, ASC/DESC defaults, `LIMIT -1` parse rejection) and 4 for
  INNER JOIN (single-key, multi-key, schema disambiguation, combined
  physical output schema).
- `Engine::new_with_device(idx)` for selecting a specific GPU on
  multi-GPU hosts. `Engine::new()` delegates to it with device 0.
- `cuda-stub` feature is now real: the `#[link(name = "cuda")]` block is
  gated and every FFI entry has a stub returning `CUDA_ERROR_STUB`, so
  `cargo check --no-default-features --features cuda-stub` works without
  the CUDA toolkit. `[package.metadata.docs.rs]` requests the feature so
  `docs.rs` builds the crate.
- Process-wide PTX cache in `jit_compiler` — FIFO at 256 entries, hashes
  the emitted PTX text and reuses the loaded `CudaModule` on a hit,
  skipping `cuModuleLoadDataEx` / PTXAS re-assembly.
- `BoolNullable` variant in the device-column enums propagates Arrow
  validity bitmaps for `BooleanArray` columns; the projection round-trip
  reconstructs a nullable `BooleanArray` on download. Filter / aggregate
  kernels still consume the values buffer only (TODO marker in
  `engine.rs`).
- New FFI bindings and safe wrappers for `cuMemAllocHost_v2`,
  `cuMemFreeHost`, `cuMemcpyHtoDAsync_v2`, `cuMemcpyDtoHAsync_v2`,
  `cuMemsetD8_v2`, `cuMemsetD8Async`.
- New CI workflow `.github/workflows/ci.yml` (Ubuntu + Windows × stable
  + 1.74) gated on `cuda-stub`, plus `dependabot.yml`, issue / PR
  templates, `CODEOWNERS`, and `SECURITY.md`.
- `tests/ptx_golden_tests.rs`: golden-snapshot smoke tests for emitted
  PTX (substring assertions on `.target sm_70`, `atom.*`, predicate
  gate, `.restrict`, sign-extension before atomic add, etc.).
- `tests/parser_tests.rs`: 17 negative parser tests covering DISTINCT,
  ORDER BY, LIMIT, HAVING, UNION, subqueries, JOIN, CTE, qualified
  column refs, integer-literal overflow, plus one positive
  bare-bool-predicate control.
- 10 offline aggregate / GROUP BY tests in `tests/e2e_tests.rs` covering
  SUM widening, COUNT(*), AVG, alias preservation, SELECT-order
  preservation, and `i64::MIN` literal handling.
- Host-only unit tests on `src/cuda/buffer.rs`, `dictionary.rs`,
  `dictionary_any.rs`, `smart_ptrs.rs` (via test-only
  `new_host_only` constructors). `dictionary_any` regains four
  previously `#[ignore]`'d dispatch tests via host-only execution.
- `DCO` file at repo root and DCO sign-off section in `CONTRIBUTING.md`.
- `ROADMAP.md` and `docs/FAQ.md`.

### Changed
- `LogicalPlan::Join::schema()` and `PhysicalPlan::Join::output_schema()`
  now return the *combined* (left + right) schema with collision-safe
  naming: any right-side field whose name clashes with a left-side name
  is prefixed `right.<col>`, with a `__2`, `__3`, … suffix as a final
  uniqueness guard. Both methods share a single `join_combined_schema`
  helper so they can't drift. Previously the logical version
  concatenated without disambiguation (duplicate names) and the physical
  version returned only the left input.
- `SUM(Int32) -> Int64` widening end-to-end (plan output dtype, scalar
  reducer, GROUP BY accumulator, kernel emits `atom.global.add.s64` with
  `cvt.s64.s32` sign extension). SUM(Int64), SUM(Float*) unchanged.
- Float-MIN/MAX GROUP BY launch in `groupby_valid` now passes 7 params
  (kernel ABI) instead of 11; integer / float-SUM variants keep all 11.
  `debug_assert_eq!` on arg count at each launch site.
- `pub fn craton_bolt::sql()` convenience deleted (it constructed an Engine
  with no tables — unusable).
- `pub struct Reg(pub u32)` IR type: field demoted to `pub(crate)` with
  a new `Reg::id() -> u32` accessor.
- `BoltError::Cuda` is now a tuple variant `Cuda(String)` (was a
  struct variant). Internal-only ergonomic; not part of the stable API.
- `GpuBuffer::zeros` uses `cuMemsetD8` (no host alloc + memcpy).
- IR types (`PhysicalPlan`, `KernelSpec`, `AggregateSpec`, `Op`, `Reg`,
  `Value`, `ColumnIO`) and internal re-exports under `exec::*` / `jit::*`
  are marked `#[doc(hidden)]` for 0.3.x.
- `Cargo.toml` gains `authors`, `repository`, `homepage`,
  `documentation`, `readme`, `keywords`, `categories`, `rust-version`,
  `[package.metadata.docs.rs]`. `log = "0.4"` added as a runtime dep.
- `LICENSE` and `NOTICE` updated to "Copyright 2026 Craton Software
  Company"; `NOTICE` lists `arrow-array`, `arrow-buffer`, `arrow-schema`,
  and `log` explicitly.
- README gains badges (crates.io / docs.rs / CI / license / MSRV), a
  Platform support subsection, and Security / Releases sections. The
  string-subset claim is tightened to flag `UPPER`/`LOWER`/`LENGTH`/
  `CONCAT` as host-only Rust API, not SQL.
- `docs/SQL_REFERENCE.md`: explicit "Not yet supported (planned)"
  section; documented `SUM(Int32) -> Int64` widening and all-NULL group
  semantics. `docs/JIT_PIPELINE.md`: predicate-gate snippet now matches
  the emitter byte-for-byte; per-instruction CC table.
  `docs/ARCHITECTURE.md`: `GpuView` corrected to `Send`-only / `!Sync`
  with rationale; IR-types stability disclaimer.
- `build.rs`: skips CUDA discovery under `cuda-stub`; picks the
  highest-version CUDA install on Windows; also searches
  `lib64/stubs/` on Linux for driverless hosts (NVIDIA's CI shim).
- `#[inline]` on leaf accessors of `GpuVec`/`GpuView`/`GpuViewMut`.
- `.ptr .global .restrict .align 16` on emitted kernel column-pointer
  params (enables PTXAS alias optimizations).

### Fixed
- **Aggregate output column order was silently rearranged**:
  `SELECT SUM(x), key FROM t GROUP BY key` previously returned
  `[key, sum_x]` because the `selected_keys` projection was built but
  never wrapped around the `Aggregate`. SELECT order is now preserved
  via a top-level `Project`; aliases on group keys are honored.
- **Windows linkage**: dropped `kind = "static"` on the
  `#[link(name = "cuda")]` attribute. `cuda.lib` is an import library
  for `nvcuda.dll`, not a static archive.
- **Soundness**: `GpuView` is now `!Sync` (was unsoundly `Sync`). A
  concurrent writer kernel launched through `GpuViewMut` against the
  parent `GpuVec` would have raced a `GpuView` reader.
- **Soundness**: `static mut INIT_RESULT` in `cuda_sys` replaced with
  `OnceLock<CUresult>` — the previous pattern was a data race and a hard
  error under Rust 2024.
- **32-bit hosts**: pointer-truncation bug in `GpuBuffer::with_capacity`
  alignment check; the `idx`-to-`usize` narrowing in
  `DictionaryColumnI64::to_string_array`.
- **`n_rows as u32` silent truncation** across every executor launch
  site, via a new `n_rows_to_u32(n_rows) -> BoltResult<u32>` helper.
- **`pack_keys` UB shift**: bare `<<` replaced with `wrapping_shl` plus
  `debug_assert!(shift + bit_width <= 64, ...)`.
- **`BooleanArray` null/false conflation** — upload now distinguishes
  null from false via the `BoolNullable` variant (round-trip works for
  projection; filter / agg kernels still see values only).
- **`__idx_<col>` device→host→device bounce** removed; the engine
  borrows the dictionary's existing `GpuVec` directly.
- **Integer literal overflow** in `parse_number`: a positive literal
  whose magnitude exceeds `i64::MAX` is now rejected with a clear error
  rather than silently demoted to `Float64`. The `i64::MIN`-magnitude
  literal `-9223372036854775808` is preserved as `Literal::Int64(i64::MIN)`.
- **AVG over all-NULL group** in `groupby_valid` now returns SQL `NULL`
  instead of `0.0` (matches `SQL_REFERENCE.md`).
- **Test memory_tests**: `shared_view_is_send_but_not_sync` assertion
  updated to match the new `GpuView: !Sync` contract.
- **DataFrame builder**: `select`/`filter`/`group_by`/`agg` now validate
  column references at builder time, deferring the first error via a
  `String`-typed `first_error` field surfaced through
  `DataFrame::validation_error()` and `schema()`.
- **`physical_plan::lower`** now folds arbitrary `Scan / Filter / Project`
  chain shapes (was: only `Scan` or `Filter(Scan)`). DataFrame chains
  like `scan().select().filter().select()` no longer produce
  unlowerable plans.
- **String literal rewriter**: peels `Alias` wrappers on either side of
  `BinaryOp::Eq`; `LiteralResolver::index_dtype` lets i64-indexed dicts
  emit `Int64` index columns rather than the hardcoded `Int32`.
- **`hash_kernels` classic keys kernel**: bounded probe loop with
  `MAX_PROBE_FACTOR = 2`; previously could spin forever on a full table.
- **`jit_compiler::from_ptx`**: uses `cuModuleLoadDataEx` with PTXAS
  info / error log buffers; failures now surface line numbers.
- **build.rs Windows fallback** picks the highest CUDA version on disk
  (was: first NTFS-ordered entry).

### Removed
- `BoltError::Nvrtc` variant (Craton Bolt uses `cuModuleLoadDataEx`, not
  NVRTC). The 4 jit_compiler.rs call sites migrated to `Cuda`.
- `pub fn craton_bolt::sql(query)` (broken — see Fixed).

### Deprecated
- `DataFrame::collect()` (use `into_plan()`; tombstone retained for 0.1
  call-site compatibility).

### Security
- (none yet — see `SECURITY.md` for the disclosure address.)

## [0.1.0] - 2026-05-23

### Added

#### CUDA layer (`src/cuda/`)
- Raw CUDA driver FFI (`cuda_sys.rs`) — context init, device discovery, memory alloc / free / memcpy, module load, stream create / destroy / sync, `cuLaunchKernel`.
- `GpuBuffer<T>` (`buffer.rs`) — owned device allocation with Arrow's 64-byte alignment.
- `GpuVec<T>` / `GpuView<'a, T>` / `GpuViewMut<'a, T>` (`smart_ptrs.rs`) — borrow-checked GPU memory. Kernel launches require borrows; use-after-free, double-free, and shared/mutable aliasing across kernel boundaries are rejected at compile time.
- `DictionaryColumn` (`dictionary.rs`) — i32-indexed string dictionary with NULL at slot 0.
- `DictionaryColumnI64` (`dictionary_i64.rs`) — i64-indexed dictionary for columns with > i32::MAX unique strings.
- `DictionaryColumnAny` (`dictionary_any.rs`) — unified enum picking i32/i64 by cardinality at construction.

#### Plan layer (`src/plan/`)
- `LogicalPlan` AST (`logical_plan.rs`) — Scan / Filter / Project / Aggregate. `Expr` covers Column / Literal / Binary / Alias. Numeric type promotion follows the standard SQL rules.
- `DataFrame` builder (`dataframe.rs`) — Polars-style lazy API.
- SQL frontend (`sql_frontend.rs`) — sqlparser-based; supports SELECT with WHERE, GROUP BY, scalar aggregates.
- `PhysicalPlan` lowering (`physical_plan.rs`) — produces fused `KernelSpec` with SSA-shaped op IR.
- `StringPredicateRewriter` (`string_literal_rewrite.rs`) — rewrites `col = 'literal'` to `__idx_col = i32/i64(idx)` against registered dictionaries.

#### JIT layer (`src/jit/`)
- PTX codegen for projection (`ptx_gen.rs`) — targets `sm_70` / `.version 7.5` / 64-bit addressing.
- PTX codegen for predicate-only kernels (`scan_kernel.rs`) — materialises u8 keep-masks.
- Scalar reduction kernels (`agg_kernels.rs`) — SUM / MIN / MAX / COUNT / AVG with per-block reduction + host-side cross-block finish.
- Hash GROUP BY kernels (`hash_kernels.rs`) — single-pass open-addressing with `atom.cas.b64` on the keys table.
- Float MIN/MAX via CAS loop (`float_atomics.rs`) — closes the sm_70 gap for `atom.global.{min,max}.f{32,64}`.
- Sentinel-free GROUP BY kernels (`valid_flag_kernels.rs`) — parallel `slot_valid: u32[]` table eliminates `i64::MIN` collision risk (notably Float64 `-0.0`).
- Sentinel-free float MIN/MAX (`valid_flag_float.rs`) — combines the CAS loop with the valid-flag probe.
- Parallel prefix-scan + gather (`prefix_scan.rs`) — Hillis-Steele per-block scan, host-side block-base reduction, per-dtype gather.
- Multi-pass prefix-scan (`prefix_scan_multipass.rs`) — recursive scan over block_sums; unbounded row counts.
- CUDA module loader (`jit_compiler.rs`) — `cuModuleLoadData` wrapper; PTX-to-cubin assembly happens inside the driver.

#### Execution layer (`src/exec/`)
- `Engine` (`engine.rs`) — top-level entry point. Holds the CUDA context, registered tables, dictionary registry. `sql(query)` returns a `QueryHandle` wrapping an Arrow `RecordBatch`.
- Scalar aggregate executor (`aggregate.rs`) — primitive SUM / MIN / MAX / COUNT / AVG.
- Aggregate with pre kernel (`agg_with_pre.rs`) — handles aggregates over expressions / filtered inputs.
- GROUP BY executor (`groupby.rs`) — packed-i64-key path with composite-tuple decode.
- GROUP BY + pre (`groupby_with_pre.rs`) — fused pre kernel + GROUP BY.
- Wide-key GROUP BY fallback (`groupby_wide.rs`) — host-side reduction for > 64-bit composite keys.
- Sentinel-free GROUP BY (`groupby_valid.rs`) — float-key safe path with bounded spin + spill.
- Stream + kernel launcher (`launch.rs`) — `CudaStream`, `KernelArgs`, 1D launch helper.
- Host-side filter compaction (`compact.rs`) — downloads mask, applies via `arrow::compute::filter`.
- GPU-side filter compaction (`gpu_compact.rs`) — prefix-scan + gather, end-to-end on the GPU.
- GPU compaction multi-pass driver (`gpu_compact_multipass.rs`).
- Dictionary registry (`dict_registry.rs`) — per-table dictionaries, drives the predicate rewrite at `Engine::sql` time.
- Bool / Utf8 aggregate executor (`extended_agg.rs`) — host-side SUM(bool) / MIN(utf8) / etc.
- Host-side expression evaluator (`expr_agg.rs`) — fallback when an aggregate input isn't a bare column ref.
- Dictionary-aware string ops (`string_ops.rs`) — UPPER / LOWER / LENGTH / input_eq_literal.
- Variable-width-free CONCAT / SUBSTRING (`string_ops_extended.rs`).
- Bool / Utf8 device columns (`string_col.rs`).

#### Tests & benches
- Memory-safety tests (`tests/memory_tests.rs`) — type-level proofs, compile-fail doctests, ignored live-GPU round-trips.
- End-to-end tests (`tests/e2e_tests.rs`) — parser → plan → PTX-shape assertions; ignored live-GPU query verification.
- Criterion benchmarks (`benches/query_benchmarks.rs`) — plan / lower / ptx_gen, CPU reference, Polars head-to-head, GPU engine path (gated behind `BOLT_BENCH_GPU=1`).

### Build status

Compiles clean on Windows MSVC / Linux with CUDA Toolkit ≥ 12. `cargo check --lib --tests --benches` works on hosts without CUDA. `cargo test` requires `cuda.lib` on the linker path; `cargo test -- --ignored` requires an NVIDIA GPU with compute capability ≥ 7.0.

### Known limitations

- No JOIN support. Single-table queries only.
- No NULL-aware GPU aggregates yet — COUNT counts every row, not just non-null. The host-side `extended_agg` path does honour nulls for Bool/Utf8.
- Variable-width string outputs (CONCAT producing genuinely new strings) work via host-side dictionary cross-product, not on the GPU.
- Polars head-to-head numbers are not yet published.

[0.7.0]: https://github.com/craton-co/craton-bolt/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/craton-co/craton-bolt/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/craton-co/craton-bolt/compare/v0.3.0...v0.5.0
[0.3.0]: https://github.com/craton-co/craton-bolt/compare/v0.1.0...v0.3.0
[0.1.0]: https://github.com/craton-co/craton-bolt/releases/tag/v0.1.0
