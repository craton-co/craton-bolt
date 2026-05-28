# Craton Bolt Roadmap

This document tracks intentional gaps in the 0.3.0 release and the milestones
planned for 0.4+. For day-to-day progress, see `CHANGELOG.md`. For supported
SQL today, see `docs/SQL_REFERENCE.md`.

## 0.3.0 (current — pre-production, API unstable)

### What works

- SQL → PTX → execution end-to-end for projection, filter, scalar
  aggregate, and GROUP BY (single/multi-column, packed and wide keys).
- `DISTINCT`, `LIMIT [OFFSET]`, `ORDER BY [ASC|DESC]`, `HAVING`, and
  `UNION [ALL]` (host-side executors for the non-GROUP-BY paths).
- `INNER`, `LEFT [OUTER]`, `RIGHT [OUTER]`, `FULL [OUTER]`, and `CROSS`
  joins (host-side hash join: build one side into a `HashMap`, probe the
  other; CROSS is a host-side cartesian product capped at the
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
- Float MIN/MAX via `atom.cas` loop on the bit pattern (no native
  `atom.global.{min,max}.f*` through sm_90).
- GPU-side filter compaction (Hillis-Steele prefix scan + per-dtype
  gather), with a host-side fallback for Utf8 outputs and a multi-pass
  scan driver for `n_rows > 16.8M`.
- Process-wide PTX module cache keyed on the emitted PTX hash, skipping
  PTXAS reassembly on hit.
- `--features cuda-stub` build path for CI and `docs.rs` (type-checks
  without a CUDA toolkit on the host).

### Known limitations (not bugs)

- Multi-batch tables are supported, but no streaming (`register_table_stream`) or
  larger-than-VRAM tables yet.
- One CUDA context, one device per `Engine`. `Engine::new_with_device`
  exists, but multi-GPU means one engine per device.
- JOIN: INNER / LEFT / RIGHT / FULL / CROSS supported, equi predicates
  (or no predicate for CROSS) only; non-equi predicates remain rejected
  at the parser. The executor is host-side (build map + probe; cartesian
  product for CROSS), not GPU-backed.
- No CTE, subqueries, window functions.
- No `IS NULL` / `IS NOT NULL`, `LIKE`, `IN`, `BETWEEN`, `CASE`,
- `NULLIF`, `COALESCE`, `CAST`, or string concat (`||`).
- No `NOT` (would need a unary op in the AST).
- Identifiers are case-sensitive; no folding.
- Qualified column references (`t.col`) are rejected even when
  unambiguous.
- Validity propagation through compact / gpu_compact selection masks is implemented,
  but primitive aggregate kernels still do not propagate validity bitmaps.
  `COUNT(expr)` over a primitive column counts every row; only the Bool/Utf8
  `extended_agg` path honours nulls.
- Post-aggregate expressions (`SUM(price) + 1`) are not yet supported.
- String functions (`UPPER`, `LOWER`, `LENGTH`, `CONCAT`, `SUBSTRING`)
  are reachable only via `src/exec/string_ops*`, not via SQL.
- Date / time / timestamp / decimal / list / struct / map types are
  unimplemented.
- Async memcpy: FFI is bound and Stage 1 safe wrappers
  (`memcpy_h2d_async`, `memcpy_d2h_async`, `memset_d8_async`,
  `PinnedHostBuffer<T>`, `GpuBuffer::copy_{from,to}_async`) have landed,
  but executors still call the synchronous `from_slice` / `to_vec`
  paths. Stage 2 (wiring executors onto explicit streams + pinned host
  buffers) is the 0.4 task.

## 0.4 — production-readiness target

### Goals

- Streaming / multi-batch tables behind a stable `register_table_stream`
  API.
- Validity propagation through primitive aggregate kernels.
- Async memcpy + pinned host buffers. **Stage 1** (safe wrappers +
  `PinnedHostBuffer<T>` + `GpuBuffer::copy_{from,to}_async`) has landed
  on top of the 0.3.0 FFI bindings; **Stage 2** wires the per-shape
  executors onto explicit streams and pinned host buffers and is the
  remaining 0.4 task.
- `KernelSpec`-keyed cache that skips codegen as well as PTXAS (the
  current cache only skips PTXAS reassembly).

### Stretch goals

- GPU hash join (the 0.3.x INNER / LEFT / RIGHT / FULL / CROSS equi-join
  executor is host-side; a GPU-resident probe path is the natural next
  step).
- Non-equi predicates via nested-loop.
- GPU sort kernel to back `ORDER BY` and the dedup step of
  `UNION` / `DISTINCT` without round-tripping through host.
- SQL functions surfaced through the parser (`UPPER`, `LOWER`, `LENGTH`,
  `CONCAT`).

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
