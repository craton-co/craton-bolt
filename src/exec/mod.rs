// SPDX-License-Identifier: Apache-2.0

#[doc(hidden)]
pub mod launch;
pub mod engine;
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
pub mod groupby_valid;
// Wave-7 executor scaffolds — owned by agents 3-6.
// Marked #[doc(hidden)] to match the wave-3 sweep: these are internal dispatch
// surfaces, not part of the public 0.2 API.
#[doc(hidden)]
pub mod distinct;
#[doc(hidden)]
pub mod sort;
#[doc(hidden)]
pub mod limit;
#[doc(hidden)]
pub mod join;

#[doc(hidden)]
pub use launch::{launch_1d, CudaStream, KernelArgs};
pub use engine::{Engine, QueryHandle};

/// Convert a host-side row count to the `u32` shape CUDA kernel launches require,
/// returning a structured error if the count exceeds `u32::MAX`.
///
/// CUDA's `cuLaunchKernel` shape parameters and most of the kernels in this
/// crate take row counts as `u32`. Truncating a `usize` (or any wider integer
/// width) with `as u32` would silently wrap on a > 4 GiB-row input and launch
/// the wrong grid; this helper surfaces that overflow as a `JavelinError::Other`
/// instead. Every executor that crosses the host/device boundary with a row
/// count should funnel through this helper rather than rolling its own cast.
pub(crate) fn n_rows_to_u32(n_rows: usize) -> crate::error::JavelinResult<u32> {
    u32::try_from(n_rows).map_err(|_| {
        crate::error::JavelinError::Other(format!(
            "row count {} exceeds the u32 launch-shape limit ({})",
            n_rows,
            u32::MAX
        ))
    })
}
