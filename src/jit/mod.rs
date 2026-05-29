// SPDX-License-Identifier: Apache-2.0

pub mod ptx_gen;
pub mod jit_compiler;
/// Optional disk-backed PTX cache (v0.6 / M6). Opt-in via the
/// `BOLT_PTX_CACHE_DIR` env var or `Engine::Builder::persistent_cache`.
/// See [`disk_cache`] module docs for the path-resolution rules.
pub mod disk_cache;
pub mod agg_kernels;
/// PTX codegen for the `SUM(Decimal128)` reduction kernel (atomic-free
/// two-stage block reduce over `i128` hi/lo halves). See [`decimal_agg`].
pub mod decimal_agg;
/// PTX codegen + host reference math for the date/time scalar functions
/// `EXTRACT(field FROM ts)` and `DATE_TRUNC(unit, ts)`, lowered to integer
/// arithmetic on `Date32`/`Timestamp` storage. See [`date_scalar`].
pub mod date_scalar;
pub mod scan_kernel;
pub mod hash_kernels;
pub mod prefix_scan;
pub mod float_atomics;
pub mod prefix_scan_multipass;
pub mod valid_flag_kernels;
pub mod valid_flag_float;
pub mod shmem_sum_kernel;
pub mod shmem_multi_sum_kernel;
pub mod shmem_count_kernel;
pub mod partition_kernel;
pub mod scatter_kernel;
pub mod scatter_with_dest_idx_kernel;
pub mod scatter_values_by_dest_idx_kernel;
pub mod partition_reduce_kernel;
pub mod partition_reduce_kernel_i64;
pub(crate) mod partition_reduce_kernel_spill_common;
pub mod partition_reduce_kernel_multi;
pub mod partition_reduce_kernel_count;
pub mod partition_reduce_kernel_minmax;
pub mod shmem_minmax_kernel;
pub mod partition_reduce_kernel_multi_i64;
pub mod partition_reduce_kernel_minmax_float;
pub mod partition_reduce_kernel_count_i64;
pub mod partition_reduce_kernel_minmax_i64;
pub mod partition_reduce_kernel_minmax_float_i64;
pub mod partition_kernel_i64;
pub mod scatter_kernel_i64;
pub mod sort_kernel;
pub mod sort_kernel_radix;
pub mod hash_join_kernel;

#[doc(hidden)]
pub use ptx_gen::compile as compile_ptx;
#[doc(hidden)]
pub use jit_compiler::{compile_and_load, CudaFunction, CudaModule};

/// Public observability hook: snapshot of the process-wide PTX module
/// cache counters. Returns `(hits, misses, evictions)` — see
/// [`jit_compiler::ptx_cache_stats`] for the full contract. Exposed at
/// the `jit` module path (not crate root) to keep `lib.rs` focused on
/// top-level engine surface.
pub use jit_compiler::ptx_cache_stats;

/// Public re-export of the disk-PTX-cache builder hook. Engine builders
/// (or the test harness) call [`disk_cache::set_override_dir`] to point
/// the process-wide disk cache at a specific directory, overriding the
/// `BOLT_PTX_CACHE_DIR` env var. Pass `None` to clear the override and
/// re-fall-back to env-var resolution.
pub use disk_cache::set_override_dir as set_disk_ptx_cache_dir;
