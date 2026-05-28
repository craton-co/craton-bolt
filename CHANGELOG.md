# Changelog

All notable changes to this project will be documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project tries to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it leaves `0.x`.

## Note on version 0.2.0

There is no `0.2.0` release. The project jumped from `0.1.0` (2026-05-23) directly to `0.3.0` (2026-05-26) — a three-day span in which the scope grew well past what a single minor bump could honestly carry (multi-batch tables, INNER JOIN, DISTINCT / LIMIT / ORDER BY / HAVING / UNION, real `cuda-stub`, PTX cache, CI). Tagging an intermediate `0.2.0` would have been a paper milestone, so the version number was reserved and skipped.

## [Unreleased]

### Added

### Changed

### Fixed

### Removed

### Deprecated

### Security

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

[0.3.0]: https://github.com/craton-co/craton-bolt/compare/v0.1.0...v0.3.0
[0.1.0]: https://github.com/craton-co/craton-bolt/releases/tag/v0.1.0
