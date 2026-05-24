# Changelog

All notable changes to this project will be documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project tries to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it leaves `0.1.x`.

## [Unreleased]

### Changed
- License is now **Apache-2.0** only (was previously `MIT OR Apache-2.0`).
  See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). All source files now
  carry an `// SPDX-License-Identifier: Apache-2.0` header.

### Added
- First criterion benchmark run captured. CPU-side numbers (plan, lower,
  ptx_gen, cpu_reference, polars) measured on a 1M-row dataset across
  three queries (`proj`, `arith`, `filtered`). See
  [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). The GPU `engine_execute`
  group still awaits a CUDA-equipped host.
- `LICENSE` (full Apache-2.0 text) and `NOTICE` (third-party attribution
  for arrow-rs, sqlparser-rs, polars dev-dep, CUDA driver).
- `docs/BENCHMARKS.md` — full benchmark methodology, results, and
  reproduction instructions.
- New `## License` section in `README.md`.

## [0.1.0] — initial release

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
- Criterion benchmarks (`benches/query_benchmarks.rs`) — plan / lower / ptx_gen, CPU reference, Polars head-to-head, GPU engine path (gated behind `JAVELIN_BENCH_GPU=1`).

### Build status

Compiles clean on Windows MSVC / Linux with CUDA Toolkit ≥ 12. `cargo check --lib --tests --benches` works on hosts without CUDA. `cargo test` requires `cuda.lib` on the linker path; `cargo test -- --ignored` requires an NVIDIA GPU with compute capability ≥ 7.0.

### Known limitations

- No JOIN support. Single-table queries only.
- No NULL-aware GPU aggregates yet — COUNT counts every row, not just non-null. The host-side `extended_agg` path does honour nulls for Bool/Utf8.
- Variable-width string outputs (CONCAT producing genuinely new strings) work via host-side dictionary cross-product, not on the GPU.
- Polars head-to-head numbers are not yet published.
