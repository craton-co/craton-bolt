// SPDX-License-Identifier: Apache-2.0

pub mod engine;
#[doc(hidden)]
pub mod launch;
// Pure-reorg split of the former monolithic `engine.rs`: self-contained items
// moved into sibling modules. `engine` re-imports the names it still uses.
mod engine_cache_key;
mod engine_device_col;
mod engine_provider;
mod engine_support;
// Pure-reorg split (engine-split): coherent `impl Engine` method clusters
// lifted out of `engine.rs` into sibling modules. These define additional
// `impl Engine { .. }` blocks (no new types); `engine.rs` keeps the struct,
// builder, and top-level dispatch. `engine_query` / `engine_join_impl` are
// reserved placeholders for follow-up extraction (see their module docs).
pub mod aggregate;
pub mod compact;
mod engine_agg_impl;
mod engine_join_impl;
mod engine_query;
mod engine_streaming_impl;
mod engine_string_impl;
/// Shared async-memcpy / pinned-host-buffer helpers for executors.
/// Lifted out of `exec::aggregate` (v0.6 pilot) so filter, GROUP BY, joins
/// and friends can adopt the same `(slice, &stream) -> GpuVec<T>` shape
/// without each rolling their own `cfg(feature = "cuda-stub")` branch.
#[doc(hidden)]
pub(crate) mod gpu_upload;
pub mod groupby;
/// Process-wide JIT-module cache shared across executors. Lifted out of
/// per-file `static MODULE_CACHE` declarations to avoid the multi-GPU
/// hazard described in the module docs.
#[doc(hidden)]
pub mod module_cache;
/// Single-source-of-truth plan<->Arrow `DataType`/`Schema` converters.
/// Replaces ~25 copy-pasted `plan_dtype_to_arrow` / `arrow_dtype_to_plan` /
/// `plan_schema_to_arrow_schema` definitions across the executors.
pub(crate) mod schema_convert;
pub mod string_col;
/// NULL / validity propagation audit matrix + shared validity helpers
/// (Arrow-LE packed-bit bitmap construction). See the module docs for the
/// per-executor propagation matrix.
pub mod validity_audit;
// dedup (groupby_common): single home for the host-side key-packing +
// pinned-D2H helpers shared by groupby / groupby_valid / groupby_with_pre.
pub mod agg_with_pre;
pub mod dict_registry;
pub mod gpu_compact;
pub(crate) mod groupby_common;
/// Fully-GPU `SELECT LENGTH(<utf8_col>)` executor: a per-row gather of a
/// precomputed per-dictionary-entry byte-length table (see
/// [`crate::jit::string_kernel::compile_length_gather_kernel`]), with a clean
/// host-side fallback for non-dict / null-bearing inputs.
pub mod string_length;
/// Executor for the GPU per-row `LIKE` matcher over variable-width (non-dict)
/// `Utf8` columns
/// ([`crate::plan::physical_plan::PhysicalPlan::StringLikeFilter`]). UNVALIDATED
/// device path — see the module docs; correctness is guaranteed by a host
/// mirror + a clean host fallback for any unsupported shape / layout.
pub mod string_like;
pub mod string_ops;
/// Executor for the GPU variable-width string projection
/// ([`crate::plan::physical_plan::PhysicalPlan::StringProject`]): `UPPER` /
/// `LOWER` over a Utf8 column, produced on the device via the two-pass
/// length/scan/write kernels in
/// [`crate::jit::string_kernel`], with a clean host-side fallback.
pub mod string_project;
// Pre-lowering pass that resolves uncorrelated scalar / IN subqueries to
// constants before physical lowering. See the module docs.
pub mod expr_agg;
pub mod extended_agg;
pub mod gpu_compact_multipass;
pub mod gpu_table;
pub mod groupby_shmem_avg_exec;
pub mod groupby_shmem_dispatch;
pub mod groupby_shmem_exec;
pub mod groupby_shmem_launch;
pub mod groupby_shmem_multi_exec;
pub mod groupby_tier2_dispatch;
pub mod groupby_valid;
pub mod groupby_wide;
pub mod groupby_with_pre;
/// Morsel / chunk streaming + larger-than-VRAM budget orchestration. Pure
/// host-side scaffolding (no CUDA on the morsel iterator or budget hooks);
/// device-pinned intermediates are a `cuda`-feature follow-up.
pub mod streaming;
pub mod string_ops_extended;
pub mod subquery_resolve;
/// Welford's online algorithm for variance, shared by the `VAR_POP` /
/// `VAR_SAMP` scalar-aggregate path. The GROUP BY path is intentionally
/// rejected by the executors below in v0.5.
/// Welford's one-pass algorithm for numerically-stable mean / variance.
/// Shared between STDDEV_* (this crate's v0.5 surface) and the upcoming
/// VAR_* aggregates.
pub mod welford;
// dedup (tier2/shmem): single home for the genuinely-identical host-side
// key-range scan shared by the Tier-1 (shmem) and Tier-2 single-key
// executors. See the module docs for why only this loop — and not the
// per-variant launch/dispatch boilerplate — is safe to share.
pub mod groupby_shmem_count_exec;
pub mod groupby_shmem_minmax_exec;
pub mod groupby_tier2_avg_exec;
pub(crate) mod groupby_tier2_common;
pub mod groupby_tier2_count_exec;
pub mod groupby_tier2_exec;
pub mod groupby_tier2_merge;
pub mod groupby_tier2_minmax_exec;
pub mod groupby_tier2_minmax_float_exec;
pub mod groupby_tier2_multi_exec;
pub mod groupby_tier2_multi_merge;
pub mod groupby_tier2_multi_orchestrator;
pub mod groupby_tier2_orchestrator;
pub mod groupby_tier2_twokey_avg_exec;
pub mod groupby_tier2_twokey_count_exec;
pub mod groupby_tier2_twokey_exec;
pub mod groupby_tier2_twokey_merge;
pub mod groupby_tier2_twokey_minmax_exec;
pub mod groupby_tier2_twokey_minmax_float_exec;
pub mod groupby_tier2_twokey_multi_exec;
pub mod groupby_tier2_twokey_orchestrator;
pub mod partition_offsets;
// Wave-7 executor scaffolds — owned by agents 3-6.
// Marked #[doc(hidden)] to match the wave-3 sweep: these are internal dispatch
// surfaces, not part of the public 0.2 API.
#[doc(hidden)]
pub mod distinct;
#[doc(hidden)]
pub mod filter;
#[doc(hidden)]
pub(crate) mod gpu_join;
#[doc(hidden)]
pub(crate) mod gpu_sort;
#[doc(hidden)]
pub mod join;
#[doc(hidden)]
pub mod limit;
/// Host-side `EXCEPT` / `INTERSECT` (with optional `ALL`) executor. Lowered
/// from `LogicalPlan::SetOp` / `PhysicalPlan::SetOp`; reuses the DISTINCT
/// executor's row-key / NULL canonicalisation. Host-only for now.
#[doc(hidden)]
pub mod setops;
#[doc(hidden)]
pub mod sort;
/// Host-side window-function executor (`func(...) OVER (...)`). Lowered from
/// `LogicalPlan::Window` / `PhysicalPlan::Window`. Host-only for now.
#[doc(hidden)]
pub mod window;
// v0.5: host-side SQL `LIKE` evaluator (`Expr::Like` lowering target).
// Exposed `pub` so the host filter executor and the expression evaluator
// both reach the `PatternMatcher` API through one source-of-truth module.
pub mod like;

pub use engine::{Engine, EngineBuilder, QueryHandle};
#[doc(hidden)]
pub use launch::{launch_1d, CudaStream, KernelArgs};
pub use streaming::{BatchProducer, BatchStream, MorselPlan, PinnedBudget, TableSource};

/// Convert a host-side row count to the `u32` shape CUDA kernel launches require,
/// returning a structured error if the count exceeds `u32::MAX`.
///
/// CUDA's `cuLaunchKernel` shape parameters and most of the kernels in this
/// crate take row counts as `u32`. Truncating a `usize` (or any wider integer
/// width) with `as u32` would silently wrap on a > 4 GiB-row input and launch
/// the wrong grid; this helper surfaces that overflow as a `BoltError::Other`
/// instead. Every executor that crosses the host/device boundary with a row
/// count should funnel through this helper rather than rolling its own cast.
pub(crate) fn n_rows_to_u32(n_rows: usize) -> crate::error::BoltResult<u32> {
    u32::try_from(n_rows).map_err(|_| {
        crate::error::BoltError::Other(format!(
            "row count {} exceeds the u32 launch-shape limit ({})",
            n_rows,
            u32::MAX
        ))
    })
}
