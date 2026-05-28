# Craton Bolt Roadmap

This document tracks intentional gaps in the 0.5.0 release and the milestones
planned for 0.6+. For day-to-day progress, see `CHANGELOG.md`. For supported
SQL today, see `docs/SQL_REFERENCE.md`.

## 0.5.0 (current — pre-production, API unstable)

This release covers the M2 milestone from `docs/PATH_TO_1.0.md`: SQL scalar
completeness. Version 0.4 was skipped — the M1 foundation items (streaming
tables, async-memcpy Stage 2, KernelSpec cache) are deferred to a later
release.

### What works

- SQL → PTX → execution end-to-end for projection, filter, scalar
  aggregate, and GROUP BY (single/multi-column, packed and wide keys).
- `DISTINCT`, `LIMIT [OFFSET]`, `ORDER BY [ASC|DESC]`, `HAVING`, and
  `UNION [ALL]` (host-side executors for the non-GROUP-BY paths).
- `INNER`, `LEFT [OUTER]`, `RIGHT [OUTER]`, `FULL [OUTER]`, and `CROSS`
  joins (host-side hash join; CROSS is host-side cartesian capped at the
  `arrow::compute::take` u32 row limit). Multiple joins per `SELECT` are
  permitted. Non-equi predicates remain out of scope.
- Borrow-checked GPU memory primitives (`GpuVec` / `GpuView` /
  `GpuViewMut`) — use-after-free, double-free, and mutable/shared
  aliasing across kernel boundaries are compile-time errors.
- Per-shape executor dispatch (scalar agg, GROUP BY, pre+agg,
  pre+GROUP BY, wide keys, sentinel-free) selected by `PhysicalPlan`
  shape.
- Dictionary-encoded Utf8 (i32 and i64 indices, cardinality-driven) for
  `=` / `!=` / `IN`-shaped string predicates, with the literal rewriter
  folding `WHERE col = 'X'` to integer equality at plan time.
- Float GROUP BY with sentinel-free fallback for keys that collide with
  `i64::MIN` (notably `-0.0`).
- Float MIN/MAX via `atom.cas` loop on the bit pattern.
- GPU-side filter compaction (Hillis-Steele prefix scan + per-dtype
  gather), with a host-side fallback for Utf8 outputs and a multi-pass
  scan driver for `n_rows > 16.8M`.
- Process-wide PTX module cache keyed on the emitted PTX hash.
- `--features cuda-stub` build path for CI and `docs.rs`.

### New in 0.5.0 — SQL scalar surface

- `IS NULL` / `IS NOT NULL` — GPU-lowered via the validity bitmap on
  bare-column operands; compound operands route through the host filter.
- `NOT <bool-expr>` — host-side filter path.
- `<expr> [NOT] IN (v1, …, vN)` — desugared to OR/AND chain, capped at 64.
- `<expr> [NOT] BETWEEN low AND high` — desugared to `>=` AND `<=`.
- `CASE WHEN cond THEN val [WHEN…] [ELSE val] END` — parser + type-check
  only; physical lowering rejects cleanly until a follow-up.
- `CAST(expr AS type)` — primitive numeric / bool pairs at the type-check
  layer; physical lowering rejects until the runtime conversion lands.
- `COALESCE` / `NULLIF` — desugared to CASE.
- `<expr> [NOT] LIKE 'pattern'` — host-side evaluator with prefix /
  suffix / contains / exact fast paths plus a generic backtracking
  matcher.
- `<expr> || <expr>` — Utf8 concat, host-side projection. WHERE concat
  rejected with a clear message.
- `STDDEV_POP` / `STDDEV_SAMP` / `STDDEV` and `VAR_POP` / `VAR_SAMP` /
  `VARIANCE` via host-side Welford. Scalar-aggregate only.
- `UPPER` / `LOWER` / `LENGTH` / `SUBSTRING` / `CONCAT` parsed and
  type-checked via `Expr::ScalarFn`; physical lowering rejects until the
  runtime path lands.

### New in 0.5.0 — SQL ergonomics

- Aggregate aliasing (`SELECT SUM(x) AS total`) carries through the
  post-Aggregate Project; visible to HAVING / ORDER BY.
- Qualified column references (`t.col`, `alias.col`) resolve against the
  FROM-tree; three-part `schema.table.col` rejected with a clear message.
- Post-aggregate scalar expressions (`SUM(x) + 1`, `(SUM(a) + SUM(b)) / 2`)
  via aggregate-feed extraction + rewritten projection.
- Case-insensitive identifiers: unquoted idents fold to lowercase at
  parse time; schema lookups fall back to case-insensitive match.
  Quoted (`"MyCol"`) identifiers preserve case.

### New in 0.5.0 — M1 foundation

- Validity propagation through primitive scalar aggregates:
  `COUNT(col)` excludes NULLs via the bitmap, and `SUM` / `MIN` / `MAX` /
  `AVG` host-strip NULL positions before the GPU reduction. The
  zero-null fast path (`null_count == 0`) remains a zero-copy upload.

### Known limitations (not bugs)

- Many of the new SQL scalar items above parse and type-check but reject
  cleanly at the physical-plan boundary with "not yet lowered to GPU"
  until the corresponding runtime path lands. Specifically: `CASE`,
  `CAST`, `STDDEV` / `VAR` under GROUP BY, scalar string functions,
  `LIKE` with ESCAPE, `||` in WHERE predicates.
- Multi-batch tables are supported but no streaming
  (`register_table_stream`) or larger-than-VRAM tables yet.
- One CUDA context, one device per `Engine`.
- JOIN executor is host-side (build map + probe); no GPU-resident path.
- No CTE, subqueries, or window functions.
- Date / time / timestamp / decimal / list / struct / map types are
  unimplemented.
- Async memcpy: FFI + Stage 1 safe wrappers have landed; Stage 2 (wiring
  executors onto explicit streams + pinned host buffers) is deferred.

## 0.6 — execution catch-up (next)

Most of the v0.5 work added SQL surface that the planner accepts and the
physical layer rejects. 0.6 is about closing those gaps:

### Goals

- GPU lowering for `CASE WHEN ... END` (predicated select).
- GPU lowering for `CAST` over the documented primitive pairs.
- Host-side runtime for the string scalar functions surfaced in 0.5
  (`UPPER`, `LOWER`, `LENGTH`, `SUBSTRING`); `CONCAT` is already wired
  through the host Project executor.
- `STDDEV` / `VAR` under GROUP BY (per-group Welford state).
- Streaming / multi-batch tables behind a stable `register_table_stream`
  API.
- Async memcpy Stage 2 — wire per-shape executors onto explicit streams
  and pinned host buffers.
- `KernelSpec`-keyed cache that skips codegen as well as PTXAS.

### Stretch goals

- GPU hash join (the existing executor is host-side; a GPU-resident
  probe path is the natural next step).
- Non-equi predicates via nested-loop.
- GPU sort kernel to back `ORDER BY` and the dedup step of
  `UNION` / `DISTINCT` without round-tripping through host.

## 1.0 — public API freeze

See [`docs/PATH_TO_1.0.md`](./docs/PATH_TO_1.0.md) for the detailed
milestone-by-milestone plan, acceptance criteria, open decisions, and
explicit exclusions. Headlines:

- All `#[doc(hidden)]` IR types (`PhysicalPlan`, `KernelSpec`,
  `AggregateSpec`, `Op`, `Reg`, `Value`, `ColumnIO`) either stabilised
  or replaced with a public builder surface.
- `DataFrame::collect()` becomes a real materialising terminal.
- Stable `Engine::sql` contract; `cuda-stub` feature documented as a
  permanent CI helper rather than an experiment.
- Multi-platform: `aarch64-linux` (Jetson) tested in CI.
- Regression-CI green; ClickBench numbers published per release.
