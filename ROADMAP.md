# Javelin Roadmap

This document tracks intentional gaps in the 0.1 release and the milestones
planned for 0.2+. For day-to-day progress, see `CHANGELOG.md`. For supported
SQL today, see `docs/SQL_REFERENCE.md`.

## 0.1.x (current — pre-production, API unstable)

### What works

- SQL → PTX → execution end-to-end for projection, filter, scalar
  aggregate, and GROUP BY (single/multi-column, packed and wide keys).
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

- Single batch per registered table. No streaming, no larger-than-VRAM
  tables.
- One CUDA context, one device per `Engine`. `Engine::new_with_device`
  exists, but multi-GPU means one engine per device.
- No JOIN of any kind.
- No DISTINCT, ORDER BY, LIMIT, OFFSET, HAVING, UNION, CTE, subqueries,
  window functions.
- No `IS NULL` / `IS NOT NULL`, `LIKE`, `IN`, `BETWEEN`, `CASE`,
  `NULLIF`, `COALESCE`, `CAST`, or string concat (`||`).
- No `NOT` (would need a unary op in the AST).
- Identifiers are case-sensitive; no folding.
- Qualified column references (`t.col`) are rejected even when
  unambiguous.
- Validity bitmaps are not propagated through filter or primitive
  aggregate kernels. `COUNT(expr)` over a primitive column counts every
  row; only the Bool/Utf8 `extended_agg` path honours nulls.
- Aggregate aliasing (`SUM(price) AS total`) is rejected by the SQL
  frontend — aggregates carry plan-assigned names.
- Post-aggregate expressions (`SUM(price) + 1`) are not yet supported.
- String functions (`UPPER`, `LOWER`, `LENGTH`, `CONCAT`, `SUBSTRING`)
  are reachable only via `src/exec/string_ops*`, not via SQL.
- Date / time / timestamp / decimal / list / struct / map types are
  unimplemented.
- Async memcpy: FFI is bound, integration is a 0.2 task; today every
  H2D / D2H is synchronous.

## 0.2 — production-readiness target

### Goals

- Streaming / multi-batch tables behind a stable `register_table_stream`
  API.
- Validity propagation through filter and primitive aggregate kernels
  (currently only the projection round-trip honours `BoolArray` nulls).
- Async memcpy + pinned host buffers (FFI is bound in 0.1.x; integration
  in 0.2).
- Warp-shuffle reduction for the last 5 strides of the agg-kernel tree
  (a TODO marker already exists in `src/jit/agg_kernels.rs`).
- `KernelSpec`-keyed cache that skips codegen as well as PTXAS (the
  current cache only skips PTXAS reassembly).
- Standardised `cargo bench` baselines published per release.

### Stretch goals

- JOINs (hash join for equi-joins, nested-loop for inequality).
- HAVING / ORDER BY (with a GPU sort).
- SQL functions surfaced through the parser (`UPPER`, `LOWER`, `LENGTH`,
  `CONCAT`).

## 1.0 — public API freeze

- All `#[doc(hidden)]` IR types (`PhysicalPlan`, `KernelSpec`,
  `AggregateSpec`, `Op`, `Reg`, `Value`, `ColumnIO`) either stabilised
  or replaced with a public builder surface.
- `DataFrame::collect()` becomes a real materialising terminal.
- Stable `Engine::sql` contract; `cuda-stub` feature documented as a
  permanent CI helper rather than an experiment.
- Multi-platform: `aarch64-linux` (Jetson) tested in CI.
