# Fixes Applied (2026-05-30)

Build: `cargo build --features cuda-stub` clean. Host tests: 1759 lib pass / 0 fail
(+31 new), all host integration suites pass, golden-PTX suites pass.

GPU-gated paths (exec/groupby/strings/cuda kernels) cannot be runtime-verified
here (no CUDA device); they are compile-verified and reasoned, runnable via
`BOLT_BENCH_GPU=1 cargo test -- --ignored` on a GPU host.

## Critical
- **Projection pruning join collision** (`plan/optimizer/projection_pruning.rs`): rebuilt
  the combined schema and map parent-required columns back to each child by position,
  undoing the `right.<name>` rename so the needed right column is no longer pruned. Probe
  test now asserts; added a collision regression test.
- **JIT integer DIV/0 + INT_MIN/-1 UB** (`jit/ptx_gen.rs` `emit_int_div_guarded`): guarded
  PTX — divisor 0 → defined 0 (no UB), INT_MIN/-1 → INT_MIN. (Limitation: full DuckDB
  div-by-zero→NULL not delivered because `emit_binary` lacks per-output validity ptrs;
  input-validity AND-fold still NULLs the row; documented.)
- **JIT cache key omits arch** (`jit/ptx_gen.rs` `codegen_salt`/`arch_salt_token`,
  `jit/disk_cache.rs`): target arch + ISA folded into the salt feeding both the in-process
  module key and the disk key.

## High
- **Grouped float MIN/MAX NaN** (`jit/float_atomics.rs`): CAS loop now honors the scalar
  NaN-as-largest total order via `testp.notanumber`. (Tier-2 partition-reduce path still
  defers NaN — comment corrected to the true reason; that kernel is out of the touched set.)
- **Non-stable radix sort** (`jit/sort_kernel_radix.rs`): replaced racing per-element
  global atomic with a block-stable histogram+reservation prologue; histogram privatized in
  shared memory. (Cross-block stability still deferred — documented; path is `BOLT_GPU_SORT`-gated, default off.)
- **Host dict-key join** (`exec/join.rs`): dictionary keys now decode to string values before
  hashing/compare (removed `DictIdx` variant), so independent dictionaries compare correctly.
- **Coarse JIT NULL-prop** (`jit/ptx_gen.rs`): wired `output_input_dependencies` so each
  output ANDs only the inputs it depends on; single-output path byte-identical.
- **NULL group keys dropped** (`exec/groupby.rs` + `plan/logical_plan.rs` key field now
  nullable): single-column GROUP BY emits a NULL group (sentinel `i64::MAX`, collision guard).
  Multi-key NULL-tuple grouping deferred (documented).
- **ILIKE `_` expanding-fold desync** (`exec/like.rs`): boundary-aware folded matcher so `_`
  consumes exactly one original codepoint.
- **kernels/ crate compile** (`kernels/*`): removed obsolete `feature(register_attr)`,
  modernized `cuda_std` to a pinned git rev consistent with the toolchain. Best-effort,
  NOT compile-verified (no rust-cuda toolchain); host `cuda_builder` version mismatch flagged.

## Medium
- **Disk JIT cache eviction** (`jit/disk_cache.rs`): LRU-by-mtime, `CRATON_BOLT_PTX_CACHE_MAX_BYTES`
  (64 MiB) / `_MAX_ENTRIES` (4096), 0 disables.
- **Unbounded set-ops map** (`exec/setops.rs`): cap mirroring DISTINCT, `CRATON_SETOP_HOST_MAX_ROWS`.
- **Sync per-column D2H** (`exec/gpu_compact.rs` `download_columns` + wired in `engine.rs`):
  batched async pinned copies on one stream, single sync.
- **Dead metrics** (`exec/engine.rs`): wired `Phase::Plan`, `Phase::Materialize`,
  `Counter::GpuLaunchesTotal`. (BytesUploaded/Downloaded, Codegen/PtxLoad/Transfer/Launch
  histograms remain unwired — need deeper instrumentation across cuda/* upload/download/load
  sites; partial.)

## Docs / OSS / CI
- Docs: created `docs/CUDARC_ADOPTION.md`; fixed `ENV_VARS.md` (the two real vars); reconciled
  UPPER/LOWER/LENGTH/LIKE host-vs-GPU + char-vs-byte across ARCHITECTURE/COMPETITIVE/LIMITATIONS/
  USER_GUIDE/SQL_REFERENCE; completed `API_SURFACE.md` (streaming, metrics, register_table_stream_lazy).
- OSS: `LICENSE` Appendix → standard placeholder; `deny.toml` TODO resolved (dated comment +
  commented placeholder RUSTSEC id — needs confirmation); removed `bolt_continue_prompt.md` +
  fixed Cargo.toml exclude.
- CI: `git mv .github/.wf → .github/workflows`; enabled `gpu-integration` lane (removed `if:false`,
  self-hosted GPU runner + fork guard, `BOLT_BENCH_GPU=1 ... -- --ignored`).

## NOT done — Section 5 "Directions" (large features, not bug fixes)
Streaming-to-device, cost-based optimizer, GPU DISTINCT/set-ops/window, collations + regex,
full persistent cubin/fatbin cache. These are multi-week subsystems; the disk-cache
eviction + arch-key work is a foundation for the cubin cache. Recommend separate tracked tasks.

## Follow-ups needing a GPU / maintainer
- Run `BOLT_BENCH_GPU=1 cargo test -- --ignored` on a GPU host to validate the grouped-NaN,
  dict-join, NULL-group, radix-sort, div-guard, and async-D2H paths.
- Confirm the `deny.toml` RUSTSEC id; register the self-hosted GPU runner; build kernels/ with
  the rust-cuda toolchain and reconcile host `cuda_builder` to the same git rev.
