// SPDX-License-Identifier: Apache-2.0

#[doc(hidden)]
pub mod launch;
pub mod engine;
/// Process-wide JIT-module cache shared across executors. Lifted out of
/// per-file `static MODULE_CACHE` declarations to avoid the multi-GPU
/// hazard described in the module docs.
#[doc(hidden)]
pub mod module_cache;
pub mod aggregate;
pub mod compact;
pub mod string_col;
pub mod groupby;
pub mod agg_with_pre;
pub mod gpu_compact;
pub mod string_ops;
pub mod dict_registry;
pub mod groupby_with_pre;
pub mod groupby_wide;
pub mod gpu_compact_multipass;
pub mod string_ops_extended;
pub mod extended_agg;
pub mod expr_agg;
/// Welford's one-pass algorithm for numerically-stable mean / variance.
/// Shared between STDDEV_* (this crate's v0.5 surface) and the upcoming
/// VAR_* aggregates.
pub mod welford;
pub mod groupby_valid;
pub mod gpu_table;
pub mod groupby_shmem_dispatch;
pub mod groupby_shmem_launch;
pub mod groupby_shmem_exec;
pub mod groupby_shmem_multi_exec;
pub mod groupby_shmem_avg_exec;
pub mod groupby_tier2_dispatch;
pub mod partition_offsets;
pub mod groupby_tier2_merge;
pub mod groupby_tier2_orchestrator;
pub mod groupby_tier2_exec;
pub mod groupby_tier2_avg_exec;
pub mod groupby_tier2_count_exec;
pub mod groupby_tier2_minmax_exec;
pub mod groupby_shmem_count_exec;
pub mod groupby_shmem_minmax_exec;
pub mod groupby_tier2_twokey_multi_exec;
pub mod groupby_tier2_minmax_float_exec;
pub mod groupby_tier2_multi_orchestrator;
pub mod groupby_tier2_multi_merge;
pub mod groupby_tier2_multi_exec;
pub mod groupby_tier2_twokey_orchestrator;
pub mod groupby_tier2_twokey_merge;
pub mod groupby_tier2_twokey_exec;
pub mod groupby_tier2_twokey_count_exec;
pub mod groupby_tier2_twokey_avg_exec;
pub mod groupby_tier2_twokey_minmax_exec;
pub mod groupby_tier2_twokey_minmax_float_exec;
// Wave-7 executor scaffolds — owned by agents 3-6.
// Marked #[doc(hidden)] to match the wave-3 sweep: these are internal dispatch
// surfaces, not part of the public 0.2 API.
#[doc(hidden)]
pub mod distinct;
#[doc(hidden)]
pub mod sort;
#[doc(hidden)]
pub(crate) mod gpu_sort;
#[doc(hidden)]
pub mod limit;
#[doc(hidden)]
pub mod join;
#[doc(hidden)]
pub(crate) mod gpu_join;
#[doc(hidden)]
pub mod filter;

#[doc(hidden)]
pub use launch::{launch_1d, CudaStream, KernelArgs};
pub use engine::{Engine, QueryHandle};

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
