// SPDX-License-Identifier: Apache-2.0

//! Top-level engine: dispatches per-shape executors (scalar agg, GROUP BY, etc.);
//! performs GPU prefix-scan + gather compaction for filter outputs, or a host-side
//! `arrow::compute::filter` fallback when any output column is Utf8.
//!
//! The engine owns a CUDA context and a registry of host-side Arrow `RecordBatch`es.
//! `Engine::sql` parses, plans, codegens, launches, and returns a `QueryHandle` whose
//! `record_batch()` exposes the result.
//!
//! Projection-with-filter flow: a predicate-only kernel materialises a `u8` mask
//! into a fresh device buffer. When every output column is gather-friendly
//! (primitive or Bool), the engine then runs `gpu_compact::compact_columns_on_gpu`
//! (prefix scan + gather) entirely on the device and downloads only the surviving
//! rows. When any output column is Utf8 — the gather kernel cannot relocate
//! variable-width strings — the engine falls back to downloading the full
//! per-column outputs plus the mask and running `compact::compact_arrays`
//! (Arrow's host-side filter) on the host. Scalar aggregates, group-bys with or
//! without a `WHERE`, and their `extended_agg`/`expr_agg` variants are
//! dispatched to dedicated executors in `Engine::execute`.

use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::hash::{Hash, Hasher};
use std::ptr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow_array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch,
};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::dictionary::DictionaryColumn;
use crate::cuda::{CudaContext, GpuVec};
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{grid_x_for, CudaStream};
use crate::exec::n_rows_to_u32;
use crate::jit::{compile_ptx, CudaModule};
use crate::plan::{
    parse_sql, DataType, Field, KernelSpec, LogicalPlan, MemTableProvider, PhysicalPlan, Schema,
};

/// PTX entry-point name; matches the symbol `ptx_gen` emits.
const KERNEL_ENTRY: &str = "bolt_kernel";

/// Entry-point name for the predicate-only mask kernel emitted by
/// [`crate::jit::scan_kernel::compile_predicate_kernel`]. Lifted out of the
/// inline string literal so the projection module-cache key can refer to it
/// without re-spelling the constant at every cache lookup.
const PREDICATE_ENTRY: &str = "bolt_predicate";

/// Cache key for [`Engine::module_cache`]: a 128-bit content hash of the
/// `KernelSpec` plus the PTX entry-point name. The entry name distinguishes
/// the two different PTX shapes the projection path can emit for the same
/// spec — the full projection kernel ([`KERNEL_ENTRY`]) and the
/// predicate-only mask kernel ([`PREDICATE_ENTRY`]).
///
/// # Why not `#[derive(Hash)]` on `KernelSpec`?
///
/// `KernelSpec` transitively contains `Op::Const { lit: Literal }`, and
/// `Literal` carries `f32`/`f64` constants. Floats do not implement `Hash`
/// (NaN inequality is the canonical reason), so deriving `Hash` on the
/// planner IR would require either a hand-rolled `Hash` over the raw bit
/// patterns of every numeric literal (and matching `PartialEq` so the
/// `Hash`/`Eq` contract holds) or a from-scratch traversal type. Either
/// route reaches far outside this file's blast radius.
///
/// # Hashing strategy
///
/// We keep the "format the IR via `Debug` then hash the bytes" pattern but
/// upgrade two things:
///
/// 1. **128-bit fingerprint.** A single 64-bit `DefaultHasher` exposes a
///    birthday-paradox collision probability of ~1 in 2^32 across all
///    distinct kernels seen during a process's lifetime; on a collision the
///    cache would silently serve the WRONG `CudaModule` for a colliding
///    spec — a silent-wrong-result failure mode. We instead hash with two
///    independent `DefaultHasher` instances domain-separated by a leading
///    byte and concatenate the 64-bit results into a `(u64, u64)`. The
///    birthday bound is now ~1 in 2^64 — unreachable for any realistic
///    workload.
///
/// 2. **No per-lookup allocation.** The previous implementation called
///    `format!("{:?}", spec)` on every cache lookup, allocating (and
///    then dropping) the entire `Debug` string just to feed it to the
///    hasher. We instead use a tiny `fmt::Write` adapter ([`HasherWrite`])
///    that streams the `Debug` output directly into the hasher as the
///    formatter emits it — zero heap allocation, identical hash input.
///
/// `DefaultHasher` is internally SipHash-1-3 with a fixed zero key, which
/// is *not* cryptographic but is more than adequate here: we are defending
/// against accidental collisions in our own deterministic IR, not against
/// an adversarial preimage attack. The two-hash domain-separation byte
/// (`0x01` vs `0x02`) makes the two streams independent enough that a
/// 128-bit collision requires a simultaneous collision in both halves.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ModuleCacheKey {
    /// Upper 64 bits of the 128-bit content hash (domain byte `0x01`).
    spec_hash_hi: u64,
    /// Lower 64 bits of the 128-bit content hash (domain byte `0x02`).
    spec_hash_lo: u64,
    /// PTX entry-point name (`KERNEL_ENTRY` vs `PREDICATE_ENTRY`).
    entry: &'static str,
}

/// `fmt::Write` → `Hasher` adapter. Lets us run `write!(adapter, "{:?}",
/// spec)` and have the formatter's emitted bytes go directly into the
/// underlying hasher without ever materialising a `String`. Saves an
/// allocation per cache lookup on the hot path.
struct HasherWrite<'a, H: Hasher>(&'a mut H);

impl<H: Hasher> std::fmt::Write for HasherWrite<'_, H> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.write(s.as_bytes());
        Ok(())
    }
}

impl ModuleCacheKey {
    /// Compute the cache key for `(spec, entry)`.
    ///
    /// Streams `format!("{:?}", spec)` into two domain-separated
    /// `DefaultHasher` instances and packs the resulting 128 bits into the
    /// key. See the type-level docstring for the rationale.
    fn new(spec: &KernelSpec, entry: &'static str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::fmt::Write as _;

        // Domain separation: write a distinguishing byte first so the two
        // hashers consume different prefixes and produce independent
        // streams over the same spec text. The actual byte values are
        // arbitrary; only the fact that they differ matters.
        let mut hi = DefaultHasher::new();
        hi.write_u8(0x01);
        // `Debug` formatting is infallible for the IR types, and
        // `HasherWrite::write_str` itself never returns an error — both
        // arms below are unreachable. Use `let _ =` rather than `unwrap`
        // so a hypothetical future failure mode degrades to a benign
        // cache miss rather than a panic in `Engine::sql`.
        let _ = write!(HasherWrite(&mut hi), "{:?}", spec);

        let mut lo = DefaultHasher::new();
        lo.write_u8(0x02);
        let _ = write!(HasherWrite(&mut lo), "{:?}", spec);

        Self {
            spec_hash_hi: hi.finish(),
            spec_hash_lo: lo.finish(),
            entry,
        }
    }
}

/// Threads per CUDA block for the 1D launch.
const BLOCK_SIZE: u32 = 256;

/// Per-table host-side revision tracker for the incremental GpuTable cache
/// (batch 5).
///
/// `table_revision` bumps on every host-side mutation that touches the
/// table — `register_table` (start at 1), `replace_table` (bump),
/// `register_batch` (bump). `column_revisions` bumps for every column
/// whose host data changed at that mutation; `column_n_rows` records the
/// total host rows that column has at the current revision (used by the
/// prefix-preserving extension path in `ensure_gpu_table`).
///
/// Mirrors the planner-cache batch 3 mechanism in spirit but stays
/// engine-local — the planner cache's invalidation is keyed off
/// `KernelSpec` content, not host data revisions.
#[derive(Debug, Default)]
struct HostTableRevision {
    /// Bumped on every host-side mutation. The GpuTable's
    /// `last_uploaded_revision` is compared against this on cache lookup.
    table_revision: u64,
    /// Per-column revision counter. Bumped for every column whose host
    /// data changed at the latest mutation. For `register_batch`
    /// (append), every column's host data changes (more rows) so every
    /// column's revision bumps.
    column_revisions: HashMap<String, u64>,
    /// Total host-row count per column at the current revision.
    /// `register_batch` records this so `ensure_gpu_table` can size the
    /// new GpuVec correctly and identify the previously-uploaded prefix.
    column_n_rows: HashMap<String, usize>,
    /// Total host-row count for the table.
    n_rows: usize,
}

/// Owned snapshot of a [`HostTableRevision`] taken under the `&self`
/// borrow before mutating `gpu_tables`. We can't keep a `&HostTableRevision`
/// across the `gpu_tables.borrow_mut()` because both live on `&self` and
/// the borrow-checker won't let us hold a reference into one engine field
/// while mutably reborrowing through a `RefCell` on another. Cloning the
/// few values we actually need is cheaper than refactoring the borrow
/// graph.
#[derive(Debug)]
struct ClonedHostRevision {
    table_revision: u64,
    column_revisions: HashMap<String, u64>,
}

/// Extension trait helper — clones a [`HostTableRevision`] reference (if
/// any) into the standalone owned form used by the incremental rebuild
/// path.
trait HostRevisionSnapshot {
    fn cloned_revision_owned(self) -> Option<ClonedHostRevision>;
}

impl HostRevisionSnapshot for Option<&HostTableRevision> {
    fn cloned_revision_owned(self) -> Option<ClonedHostRevision> {
        self.map(|h| ClonedHostRevision {
            table_revision: h.table_revision,
            column_revisions: h.column_revisions.clone(),
        })
    }
}

/// Number of rows the device-side storage of a `GpuColumnData` currently
/// holds. Used by the incremental cache to compare against the host's
/// new row count and decide whether to prefix-extend or fully re-upload.
fn column_storage_rows(data: &crate::exec::gpu_table::GpuColumnData) -> usize {
    use crate::exec::gpu_table::GpuColumnData::*;
    match data {
        I32(v) => v.len(),
        I64(v) => v.len(),
        F32(v) => v.len(),
        F64(v) => v.len(),
        Bool(v) => v.len(),
        BoolNullable { values, .. } => values.len(),
        Utf8 { indices, .. } => indices.len(),
        DictUtf8 { keys, .. } => keys.len(),
    }
}

/// Try to extend `prev` (a stale GpuColumn whose host data strictly grew)
/// into a fresh column at `n_rows_total` rows by preserving the
/// previously-uploaded prefix and HtoD-uploading only the tail.
///
/// Returns:
///   - `Ok(Some(new_column))` — extension succeeded; caller should
///     prefer this over a full re-upload (no PCIe traffic for the
///     prefix).
///   - `Ok(None)` — the variant can't be safely extended in place (e.g.
///     bit-packed validity bitmap with a non-byte-aligned previous row
///     count). Caller should fall back to a full re-upload.
///   - `Err(_)` — a CUDA / Arrow error.
fn try_extend_column(
    prev: crate::exec::gpu_table::GpuColumn,
    concatenated: &RecordBatch,
    col_idx: usize,
    n_rows_total: usize,
) -> BoltResult<Option<crate::exec::gpu_table::GpuColumn>> {
    use crate::exec::gpu_table::{GpuColumn, GpuColumnData};
    let prev_rows = column_storage_rows(&prev.data);
    // Caller already enforced 0 < prev_rows < n_rows_total but re-check
    // defensively here so the helpers can stand alone.
    if prev_rows == 0 || prev_rows >= n_rows_total {
        return Ok(None);
    }
    let arr = concatenated.column(col_idx);
    let GpuColumn {
        name,
        dtype,
        data,
        host_revision: _,
    } = prev;
    let new_data: GpuColumnData = match data {
        GpuColumnData::I32(old) => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| {
                    BoltError::Type(format!(
                        "incremental extend: column '{name}' was I32 on device but \
                         host array is {:?}",
                        arr.data_type()
                    ))
                })?;
            let tail: Vec<i32> = (prev_rows..n_rows_total)
                .map(|i| pa.value(i))
                .collect();
            let extended = old.extended_with_prefix(n_rows_total, prev_rows, &tail)?;
            GpuColumnData::I32(extended)
        }
        GpuColumnData::I64(old) => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    BoltError::Type(format!(
                        "incremental extend: column '{name}' was I64 on device but \
                         host array is {:?}",
                        arr.data_type()
                    ))
                })?;
            let tail: Vec<i64> = (prev_rows..n_rows_total)
                .map(|i| pa.value(i))
                .collect();
            let extended = old.extended_with_prefix(n_rows_total, prev_rows, &tail)?;
            GpuColumnData::I64(extended)
        }
        GpuColumnData::F32(old) => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    BoltError::Type(format!(
                        "incremental extend: column '{name}' was F32 on device but \
                         host array is {:?}",
                        arr.data_type()
                    ))
                })?;
            let tail: Vec<f32> = (prev_rows..n_rows_total)
                .map(|i| pa.value(i))
                .collect();
            let extended = old.extended_with_prefix(n_rows_total, prev_rows, &tail)?;
            GpuColumnData::F32(extended)
        }
        GpuColumnData::F64(old) => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    BoltError::Type(format!(
                        "incremental extend: column '{name}' was F64 on device but \
                         host array is {:?}",
                        arr.data_type()
                    ))
                })?;
            let tail: Vec<f64> = (prev_rows..n_rows_total)
                .map(|i| pa.value(i))
                .collect();
            let extended = old.extended_with_prefix(n_rows_total, prev_rows, &tail)?;
            GpuColumnData::F64(extended)
        }
        GpuColumnData::Bool(old) => {
            let ba = arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    BoltError::Type(format!(
                        "incremental extend: column '{name}' was Bool on device but \
                         host array is {:?}",
                        arr.data_type()
                    ))
                })?;
            // Only safe for null-free Bool — the variant we have is
            // `Bool` (non-nullable). If the appended batch added nulls,
            // the GpuColumnData variant would need to become
            // `BoolNullable`, and we can't extend across a variant
            // change. Punt to full re-upload.
            use arrow::array::Array as _;
            if ba.null_count() != 0 {
                return Ok(None);
            }
            let tail_rows = n_rows_total - prev_rows;
            let mut tail: Vec<u8> = Vec::with_capacity(tail_rows);
            for i in prev_rows..n_rows_total {
                tail.push(if ba.value(i) { 1 } else { 0 });
            }
            let extended = old.extended_with_prefix(n_rows_total, prev_rows, &tail)?;
            GpuColumnData::Bool(extended)
        }
        GpuColumnData::BoolNullable { values, validity } => {
            let ba = arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    BoltError::Type(format!(
                        "incremental extend: column '{name}' was BoolNullable on \
                         device but host array is {:?}",
                        arr.data_type()
                    ))
                })?;
            let tail_rows = n_rows_total - prev_rows;
            let mut tail_v: Vec<u8> = Vec::with_capacity(tail_rows);
            let mut tail_m: Vec<u8> = Vec::with_capacity(tail_rows);
            use arrow::array::Array as _;
            for i in prev_rows..n_rows_total {
                if ba.is_null(i) {
                    tail_v.push(0);
                    tail_m.push(0);
                } else {
                    tail_v.push(if ba.value(i) { 1 } else { 0 });
                    tail_m.push(1);
                }
            }
            let new_values = values.extended_with_prefix(n_rows_total, prev_rows, &tail_v)?;
            let new_validity =
                validity.extended_with_prefix(n_rows_total, prev_rows, &tail_m)?;
            GpuColumnData::BoolNullable {
                values: new_values,
                validity: new_validity,
            }
        }
        // Utf8 / DictUtf8: the host-side dictionary is rebuilt on every
        // `register_batch` (review C10), and we'd need to re-derive
        // per-row indices from the new dictionary to update the GPU
        // copy. Falling back to a full re-upload is simpler and
        // correct — the prefix optimisation here would require teaching
        // the device-side keys layout about dict offsets, which is
        // out of scope for batch 5. Returning `None` triggers the
        // caller's full re-upload fallback.
        GpuColumnData::Utf8 { .. } | GpuColumnData::DictUtf8 { .. } => {
            return Ok(None);
        }
    };
    Ok(Some(GpuColumn {
        name,
        dtype,
        data: new_data,
        host_revision: 0, // caller overwrites
    }))
}

/// Stage 7 (P1b): default interval between pool-stats emits in
/// [`Engine::sql`].
///
/// 60 seconds is a sensible floor for a typical analytical workload —
/// the pool changes slowly relative to query churn, and a coarser
/// cadence keeps the log line out of per-query latency. Override with
/// `BOLT_POOL_STATS_INTERVAL_SECS=<n>`; set to `0` to disable emission
/// entirely (handy for benchmark runs that don't want the noise).
const DEFAULT_POOL_STATS_INTERVAL_SECS: u64 = 60;

/// Environment-variable override for the pool-stats periodic-emit
/// interval. Parsed once per `Engine` construction; non-integer or
/// negative values fall back to [`DEFAULT_POOL_STATS_INTERVAL_SECS`].
///
/// `pub(crate)` so the integration test
/// `tests/env_var_smoke.rs` can address the canonical env-var name
/// instead of duplicating it (drift between the constant here and a
/// hard-coded string in the test would silently desynchronise the
/// toggle smoke).
pub const POOL_STATS_ENV: &str = "BOLT_POOL_STATS_INTERVAL_SECS";

/// Synchronize the default stream and convert any pending CUDA error.
///
/// `cuLaunchKernel` is asynchronous: its return value reflects only whether
/// the launch was *accepted*, not whether the kernel later faulted. If we
/// don't synchronize, a kernel-side fault (illegal address, OOB shared
/// memory access, assertion failure, etc.) surfaces at the *next* CUDA API
/// call — which may be many lines away and in unrelated code, producing
/// extremely misleading error messages and stack traces during debugging.
///
/// In debug builds we call `cuCtxSynchronize` immediately after every
/// launch site so faults are reported at the actual launch that caused
/// them. Release builds skip this entirely: the `cfg!(debug_assertions)`
/// check is a compile-time constant, so the optimiser folds this function
/// into a no-op (`Ok(())`) and any per-launch latency goes to zero.
///
/// Cheap in release: a no-op when `cfg!(debug_assertions)` is false.
#[inline]
fn debug_sync_check() -> crate::error::BoltResult<()> {
    if cfg!(debug_assertions) {
        unsafe { crate::cuda::cuda_sys::check(crate::cuda::cuda_sys::cuCtxSynchronize())? };
    }
    Ok(())
}

/// Top-level query engine.
///
/// Field-drop order matters: `dict_registry` owns `DictionaryColumn`s which own
/// `GpuVec`s — those must be freed BEFORE `_ctx` tears down the CUDA context.
/// Rust drops fields in declaration order, so `_ctx` sits last.
pub struct Engine {
    /// Registered tables, keyed by name. A single table may comprise multiple
    /// batches (wave-7 multi-batch support): the engine concatenates them via
    /// `arrow::compute::concat_batches` at query time. This is a 0.2-era
    /// simplification — a streaming, per-batch query plan is a 0.3 goal — so
    /// large multi-batch tables pay a full materialisation cost on every
    /// `sql()` call. Keep the per-table batch count modest until then.
    tables: HashMap<String, Vec<RecordBatch>>,
    /// Name → Schema provider, kept in sync with `tables`. The schema is
    /// EXTENDED with `__idx_<col>` Int32 columns for every registered Utf8
    /// column so the SQL frontend resolves rewriter-produced column refs.
    provider: MemTableProvider,
    /// Per-table Utf8 dictionaries; drives the string-literal predicate rewrite.
    dict_registry: crate::exec::dict_registry::DictRegistry,
    /// GPU-resident copies of every registered table. Owns the device
    /// allocations; must drop BEFORE `_ctx`.
    ///
    /// Wrapped in `RefCell<Option<_>>` to support a lazy-upload strategy:
    /// `register_batch` only mutates the host-side `tables` and sets the slot
    /// to `None` (dirty). The actual upload happens on the next query, in
    /// `ensure_gpu_table` from inside `execute_projection`. This collapses a
    /// streaming append workload that uploaded `1+2+…+N = N(N+1)/2` batches'
    /// worth of bytes (the per-append re-upload bug) down to a single upload
    /// per query of the current concatenated table — i.e. O(N) total bytes
    /// across the lifetime of a streaming-then-query session, instead of
    /// O(N²). Multiple consecutive `register_batch` calls without an
    /// intervening query share that one upload.
    ///
    /// **Batch 5 (incremental cache)**: the slot now holds `Some(GpuTable)`
    /// even across `register_batch` mutations. The host bumps per-table /
    /// per-column revisions in [`Engine::host_revisions`] on mutation, and
    /// `ensure_gpu_table` compares them against the GpuTable's
    /// `last_uploaded_revision` plus each column's `host_revision`:
    /// columns whose revision still matches are reused in place; only
    /// dirty columns are re-uploaded. For `register_batch` appends, the
    /// re-upload allocates a fresh GpuVec sized for the new total rows,
    /// DtoD-copies the previously-uploaded prefix, and HtoD-uploads only
    /// the new tail — so the unchanged rows never re-cross the PCIe bus.
    gpu_tables: RefCell<HashMap<String, Option<crate::exec::gpu_table::GpuTable>>>,
    /// Per-table host-side revision counters for the incremental GpuTable
    /// cache (batch 5).
    ///
    /// Mutated by `register_table` / `replace_table` / `register_batch` and
    /// read by `ensure_gpu_table`. Both mutators take `&mut self`, and
    /// `ensure_gpu_table` only borrows it immutably, so a `RefCell` would
    /// be unnecessary noise — a plain field suffices.
    host_revisions: HashMap<String, HostTableRevision>,
    /// Test-only counter incremented on every per-column upload performed
    /// by [`Engine::ensure_gpu_table`]. Exposed so the incremental-upload
    /// tests can assert that an unchanged column was reused (LOAD_COUNT
    /// did not bump for it).
    ///
    /// Uses `SeqCst` so a test that observes a count, registers a batch,
    /// re-queries, and observes the count again sees a strict
    /// happens-before relation.
    #[cfg(test)]
    gpu_table_load_count: std::sync::atomic::AtomicUsize,
    /// Stage 7 (P1b): pool-stats observability state.
    ///
    /// `Mutex<Option<Instant>>`: `Some(last_emit_time)` after the first
    /// emit, `None` before any query has run. The first query on a fresh
    /// engine always emits (so a short-lived process still surfaces at
    /// least one snapshot); subsequent queries emit only after
    /// `pool_stats_interval` has elapsed.
    ///
    /// Wrapped in a `Mutex` because `Engine::sql` takes `&self` and we
    /// support concurrent calls in principle (the underlying engine is
    /// not yet `Send + Sync` because of `RefCell`, but the
    /// pool-stats accounting is independent and shouldn't add new
    /// `!Sync` constraints when we eventually relax the rest).
    pool_stats_last_emit: Mutex<Option<Instant>>,
    /// Interval between pool-stats emits. Frozen at construction from
    /// `BOLT_POOL_STATS_INTERVAL_SECS` (default 60s). A value of
    /// `Duration::ZERO` disables periodic emission entirely.
    pool_stats_interval: Duration,
    /// Review-H2 PTX module cache: `KernelSpec` content hash + entry name →
    /// loaded `CudaModule`. Lifts the per-query
    /// `compile_ptx` + `CudaModule::from_ptx` round-trip in
    /// `execute_projection` to a process-local table lookup.
    ///
    /// The underlying `CudaModule` is `Clone` over an internal
    /// `Arc<CudaModuleInner>` (see `jit::jit_compiler`), so a cached entry
    /// can hand out cheap handle-clones to repeated callers — the cubin is
    /// loaded into the driver exactly once per `(spec, entry)` pair across
    /// the engine's lifetime.
    ///
    /// `Mutex`-guarded because `Engine::sql` takes `&self` and we may
    /// eventually relax the engine's `!Sync` constraints (the `RefCell`
    /// on `gpu_tables` is the real blocker today, not this cache).
    ///
    /// Counter `module_cache_loads` increments on every cache miss; tests
    /// observe it to confirm the cache services repeat calls.
    module_cache: Mutex<HashMap<ModuleCacheKey, CudaModule>>,
    /// Number of cache misses observed by `get_or_build_module`. Bumped
    /// once per fresh `compile_ptx` + `CudaModule::from_ptx` round-trip.
    /// Read by the projection-cache unit test to assert the second call
    /// returns the cached module without re-loading. Atomic-ordered
    /// `SeqCst` so the test's load/store interleaves cleanly with the
    /// engine's increment.
    module_cache_loads: std::sync::atomic::AtomicUsize,
    /// Owned CUDA context — declared LAST so it drops AFTER dictionaries.
    _ctx: CudaContext,
}

impl Engine {
    /// Create an engine on the default CUDA device (ordinal 0).
    ///
    /// Convenience constructor for single-GPU systems. On hosts with more
    /// than one CUDA device, use [`Engine::new_with_device`] to pick a
    /// specific GPU.
    pub fn new() -> BoltResult<Self> {
        Self::new_with_device(0)
    }

    /// Create an engine bound to the CUDA device at ordinal `device_idx`.
    ///
    /// Use this when running on a multi-GPU host and you want to target a
    /// specific device. The constructor:
    ///   1. Initializes the CUDA driver (idempotent — safe to call repeatedly).
    ///   2. Validates `device_idx` against `cuDeviceGetCount`.
    ///   3. Creates an owned CUDA context on the selected device.
    ///
    /// # Errors
    /// Returns an error if `device_idx < 0` or `device_idx >=
    /// cuDeviceGetCount()`, or if any underlying CUDA driver call fails
    /// (e.g. no CUDA-capable device, driver/runtime mismatch).
    pub fn new_with_device(device_idx: i32) -> BoltResult<Self> {
        // Initialize the driver up-front so device_count() is callable.
        cuda_sys::init()?;
        let count = cuda_sys::device_count()?;
        if device_idx < 0 || device_idx >= count {
            return Err(BoltError::Other(format!(
                "CUDA device index {} is out of range: {} device(s) visible to the driver (valid range: 0..{})",
                device_idx, count, count
            )));
        }
        let ctx = CudaContext::new(device_idx)?;
        let pool_stats_interval = pool_stats_interval_from_env();
        Ok(Self {
            tables: HashMap::new(),
            provider: MemTableProvider::new(),
            dict_registry: crate::exec::dict_registry::DictRegistry::new(),
            gpu_tables: RefCell::new(HashMap::new()),
            host_revisions: HashMap::new(),
            #[cfg(test)]
            gpu_table_load_count: std::sync::atomic::AtomicUsize::new(0),
            pool_stats_last_emit: Mutex::new(None),
            pool_stats_interval,
            module_cache: Mutex::new(HashMap::new()),
            module_cache_loads: std::sync::atomic::AtomicUsize::new(0),
            _ctx: ctx,
        })
    }

    /// Review-H2: look up the cached `CudaModule` for `(spec, entry)`, or
    /// compile + load it on a miss and seed the cache.
    ///
    /// `entry` selects between the projection kernel and the predicate-only
    /// mask kernel — they generate different PTX from the same `KernelSpec`,
    /// so the entry name participates in the key. On a cache hit we hand
    /// back a cheap `CudaModule` clone (Arc-handle). On a miss we run the
    /// underlying PTX-text-hash cache in `jit::jit_compiler`, which itself
    /// short-circuits the `cuModuleLoadDataEx` step on a repeat PTX string;
    /// either way we then memoise the result here so future calls skip the
    /// PTX generation entirely.
    ///
    /// The closure-based loader keeps us from re-spelling the projection vs
    /// predicate compile path at every call site.
    fn get_or_build_module<F>(
        &self,
        spec: &KernelSpec,
        entry: &'static str,
        compile: F,
    ) -> BoltResult<CudaModule>
    where
        F: FnOnce(&KernelSpec) -> BoltResult<String>,
    {
        let key = ModuleCacheKey::new(spec, entry);
        // Fast path: hit. Hold the lock only long enough to clone the Arc.
        if let Some(m) = self
            .module_cache
            .lock()
            .map_err(|_| BoltError::Other("module_cache mutex poisoned".to_string()))?
            .get(&key)
        {
            return Ok(m.clone());
        }
        // Miss: compile + load without the cache lock held — PTX gen and
        // `cuModuleLoadDataEx` can be slow and we don't want to serialise
        // unrelated cache misses behind one ongoing compile. The jit
        // PTX-text-hash cache deduplicates the cubin load on its own.
        let ptx = compile(spec)?;
        let module = CudaModule::from_ptx(&ptx)?;
        self.module_cache_loads
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // Insert and hand back a clone. If a concurrent thread raced us to
        // the same key, `or_insert` keeps the first winner — both threads
        // observe the same `Arc<CudaModuleInner>`, just one of the two
        // `CudaModule` wrappers we built gets dropped (cheap: an Arc dec).
        let mut cache = self
            .module_cache
            .lock()
            .map_err(|_| BoltError::Other("module_cache mutex poisoned".to_string()))?;
        Ok(cache.entry(key).or_insert(module).clone())
    }

    /// Batch 5 helper — rebuild the [`HostTableRevision`] for `name` so
    /// every column in `batch` carries a freshly-bumped revision and the
    /// table revision itself bumps by 1. Called from `register_table`
    /// (initial install: starts the table at revision 1) and
    /// `replace_table` (whole-table swap: starts the new shape at the
    /// next revision after whatever the old one was on).
    ///
    /// `register_batch` does NOT go through here — it bumps in place to
    /// preserve the prior `column_revisions` HashMap allocation and to
    /// update `column_n_rows` per the append semantics. See its inline
    /// code.
    fn bump_table_full_replace(&mut self, name: &str, batch: &RecordBatch) {
        let prev = self.host_revisions.remove(name);
        let next_table_rev = prev.as_ref().map(|p| p.table_revision).unwrap_or(0) + 1;
        let mut column_revisions: HashMap<String, u64> =
            HashMap::with_capacity(batch.num_columns());
        let mut column_n_rows: HashMap<String, usize> =
            HashMap::with_capacity(batch.num_columns());
        let n_rows = batch.num_rows();
        for field in batch.schema().fields() {
            column_revisions.insert(field.name().clone(), next_table_rev);
            column_n_rows.insert(field.name().clone(), n_rows);
        }
        self.host_revisions.insert(
            name.to_string(),
            HostTableRevision {
                table_revision: next_table_rev,
                column_revisions,
                column_n_rows,
                n_rows,
            },
        );
    }

    /// Register a host-side `RecordBatch` under `name` as a single-batch table.
    /// Errors if a table with that name already exists; use
    /// [`Engine::register_batch`] to append additional batches to an existing
    /// table (wave-7 multi-batch entry).
    ///
    /// Also builds Utf8 dictionaries for the table and extends the engine-side
    /// schema with `__idx_<col>` Int32 columns so the rewriter's emitted column
    /// references resolve at parse time.
    pub fn register_table(
        &mut self,
        name: impl Into<String>,
        batch: RecordBatch,
    ) -> BoltResult<()> {
        let name = name.into();
        if self.tables.contains_key(&name) {
            return Err(BoltError::Plan(format!(
                "table '{name}' is already registered — use register_batch to append \
                 additional batches to an existing table"
            )));
        }
        // Stage 6: the historical flatten step (`flatten_dictionary_utf8_columns`)
        // is gone from the hot path. `DictRegistry::register_table` matches
        // `DictionaryArray<Int32, Utf8>` directly and re-uses the input
        // dictionary; `GpuTable::from_record_batch` routes the same Arrow
        // variant through `upload_dict_utf8`, packing the keys' null buffer
        // into an on-device validity bitmap. Stage 4's compat materialisation
        // is preserved as a deprecated no-op for out-of-tree callers only.
        //
        // Build Utf8 dictionaries first (may fail — surface before we mutate
        // tables/provider).
        self.dict_registry.register_table(name.clone(), &batch)?;
        let base_schema = arrow_schema_to_plan_schema(batch.schema().as_ref())?;
        let extended = self.dict_registry.extended_schema(&name, &base_schema);
        self.provider.register(name.clone(), extended);
        // Stage 6: surface per-column runtime nullability so the engine's
        // null-aware paths can short-circuit the validity bitmap upload
        // when a column is provably null-free. For `DictionaryArray`
        // columns the answer comes from `keys().null_count()` — *not* the
        // dictionary values.
        propagate_column_nullability(&mut self.provider, &name, &batch);
        // Batch 5 (incremental GpuTable cache): bump revisions BEFORE
        // building the GpuTable so the GpuTable can be stamped with the
        // current host revisions and the cache hit-check in
        // `ensure_gpu_table` succeeds on the very next query.
        self.bump_table_full_replace(&name, &batch);
        let table_rev = self.host_revisions[&name].table_revision;
        // Build a GPU-resident copy so execution can query in place.
        let mut gpu_table = crate::exec::gpu_table::GpuTable::from_record_batch(&batch)?;
        gpu_table.last_uploaded_revision = table_rev;
        for col in gpu_table.columns.iter_mut() {
            col.host_revision = table_rev;
        }
        // Test-only: count one upload per column for the initial install.
        #[cfg(test)]
        self.gpu_table_load_count
            .fetch_add(gpu_table.columns.len(), std::sync::atomic::Ordering::SeqCst);
        self.gpu_tables
            .borrow_mut()
            .insert(name.clone(), Some(gpu_table));
        self.tables.insert(name, vec![batch]);
        Ok(())
    }

    /// Replace any existing table named `name` with a single-batch table
    /// holding `batch`. Idempotent; equivalent to "unregister then
    /// register_table" but performs both halves atomically with respect to
    /// engine state so a failure mid-rebuild can't leave a torn table.
    ///
    /// This is the right entry point when you want to *update* a table's
    /// contents, e.g. an analytics tool that re-uploads a refreshed snapshot,
    /// or a benchmark harness that verifies on a small fixture then swaps in
    /// the timed-run dataset (the use case that motivated this method).
    ///
    /// Dictionaries, the SQL-frontend provider schema, the host-side batch
    /// list, AND the GPU-resident `GpuTable` are all rebuilt from `batch`.
    /// The previous `GpuTable`'s device allocations are returned to the
    /// memory pool, where the new upload can recycle them.
    pub fn replace_table(
        &mut self,
        name: impl Into<String>,
        batch: RecordBatch,
    ) -> BoltResult<()> {
        let name = name.into();
        // Stage 6: see `register_table` — the flatten step is gone from the
        // hot path. Dict ingest is native through `DictRegistry::register_table`
        // and `GpuTable::from_record_batch::upload_dict_utf8`.
        //
        // Build the new GPU table FIRST so an upload failure can't leave the
        // engine half-replaced (we have not yet touched any existing entry).
        let mut new_gpu_table = crate::exec::gpu_table::GpuTable::from_record_batch(&batch)?;
        let base_schema = arrow_schema_to_plan_schema(batch.schema().as_ref())?;

        // Drop the old GpuTable explicitly so its device allocations return
        // to the pool BEFORE we mint the dictionary index columns for the
        // replacement (those may also allocate from the pool — letting the
        // pool churn rather than grow keeps RAII tidy).
        self.gpu_tables.borrow_mut().remove(&name);
        self.dict_registry.unregister_table(&name);
        // Re-register dictionaries for the new batch.
        self.dict_registry.register_table(name.clone(), &batch)?;
        let extended = self.dict_registry.extended_schema(&name, &base_schema);
        // `MemTableProvider::register` already overwrites — no separate `replace`
        // entry point needed.
        self.provider.register(name.clone(), extended);
        // Stage 6: mirror `register_table` — re-surface per-column nullability
        // so a replace doesn't leave stale claims behind.
        propagate_column_nullability(&mut self.provider, &name, &batch);
        // Batch 5: stamp the new GpuTable with the current host revisions
        // (replace is a full rebuild, so every column gets the same fresh
        // revision number).
        self.bump_table_full_replace(&name, &batch);
        let table_rev = self.host_revisions[&name].table_revision;
        new_gpu_table.last_uploaded_revision = table_rev;
        for col in new_gpu_table.columns.iter_mut() {
            col.host_revision = table_rev;
        }
        #[cfg(test)]
        self.gpu_table_load_count.fetch_add(
            new_gpu_table.columns.len(),
            std::sync::atomic::Ordering::SeqCst,
        );
        self.gpu_tables
            .borrow_mut()
            .insert(name.clone(), Some(new_gpu_table));
        self.tables.insert(name, vec![batch]);
        Ok(())
    }

    /// Append `batch` to the table named `name`, creating it if absent.
    /// Multi-batch tables are concatenated into a single `RecordBatch` at
    /// query time via `arrow::compute::concat_batches` — see the field doc on
    /// `tables` for the perf caveat.
    ///
    /// Subsequent batches MUST share the schema of the first batch; mismatched
    /// schemas surface a `Plan` error here rather than at query time.
    ///
    /// Dictionaries are **unioned across all registered batches** (review C10):
    /// after each append, the dict registry is rebuilt against the
    /// concatenated host batches so the string-literal rewriter sees every
    /// dictionary value that exists in any batch. Without this union, a query
    /// like `WHERE s = 'literal_only_in_batch_2'` would constant-fold to
    /// `Bool(false)` against batch 0's dictionary and silently return zero
    /// rows even though batch 2 contains matching rows. The GPU index column
    /// is rebuilt lazily on the next query via `ensure_gpu_table` (which
    /// scans the same concatenated batch through `GpuTable::from_record_batch`),
    /// so the registry's dictionary and the GPU's per-row indices stay aligned
    /// — both are built from the same concat-batch input, in the same
    /// first-occurrence order.
    ///
    /// Performance: this method does NOT re-upload anything to the GPU. It
    /// only pushes the host-side `RecordBatch`, rebuilds the host-side
    /// dictionary against the materialised concat, and bumps per-column
    /// host revisions for the table. The GPU-resident `GpuTable` stays
    /// intact in the cache — the next query touches each column through
    /// `ensure_gpu_table`, which compares per-column host revisions
    /// against `GpuColumn::host_revision` and:
    ///   - reuses any column whose revision still matches (no re-upload);
    ///   - for each dirty column, allocates a new GpuVec sized for the
    ///     full new row count, DtoD-copies the previously-uploaded
    ///     prefix from the cached column, and HtoD-uploads only the
    ///     tail of new rows. The unchanged prefix never re-crosses
    ///     PCIe.
    ///
    /// Before this incremental cache (batch 5), `register_batch` set the
    /// `gpu_tables` slot to `None` and the next query re-uploaded EVERY
    /// column in full from the concatenated host batches. A
    /// streaming-append workload that issued one query between each of N
    /// appends paid `1+2+…+N = N(N+1)/2` batches' worth of HtoD traffic.
    /// With the incremental cache, the same workload pays N batches'
    /// worth — one HtoD copy of the new tail per append.
    pub fn register_batch(
        &mut self,
        name: &str,
        batch: RecordBatch,
    ) -> BoltResult<()> {
        // Stage 6: dict-encoded columns are ingested natively now, so no
        // flatten-to-StringArray is needed for the schema check below to
        // line up — batch 0 and any appended batch both carry the Arrow
        // `Dictionary<Int32, Utf8>` type verbatim.
        if let Some(existing) = self.tables.get_mut(name) {
            // Schema-check against batch 0 — concat_batches would fail at query
            // time anyway, but surface it eagerly at registration time.
            if let Some(first) = existing.first() {
                if first.schema() != batch.schema() {
                    return Err(BoltError::Plan(format!(
                        "register_batch: schema mismatch for table '{name}' — \
                         expected {:?}, got {:?}",
                        first.schema(),
                        batch.schema()
                    )));
                }
            }
            existing.push(batch);
            // Review C10: rebuild the dict registry against the *concatenated*
            // batches so the string-literal rewriter sees every dict value
            // from every batch — not just batch 0. Without this, a literal
            // that lives only in an appended batch resolves to `None` in the
            // rewriter and the predicate folds to `Bool(false)`, silently
            // dropping every otherwise-matching row in the appended batch.
            //
            // We also re-extend the provider schema in case rebuilding flipped
            // any `__idx_<col>` between i32 and i64 (the union may push a
            // column over the i64 cardinality threshold). And we re-evaluate
            // per-column nullability against the same concatenated view — a
            // previously null-free column may have just gained a null.
            let concatenated = self.materialize_table(name)?;
            self.dict_registry.unregister_table(name);
            self.dict_registry
                .register_table(name.to_string(), &concatenated)?;
            let base_schema =
                arrow_schema_to_plan_schema(concatenated.schema().as_ref())?;
            let extended = self.dict_registry.extended_schema(name, &base_schema);
            self.provider.register(name.to_string(), extended);
            propagate_column_nullability(&mut self.provider, name, &concatenated);
            // Batch 5: bump per-column host revisions for an append. Every
            // column gains rows, so every column's revision bumps; the
            // table revision bumps too. The GpuTable in `gpu_tables` is
            // INTENTIONALLY left in place — `ensure_gpu_table` will
            // compare revisions on the next query and incrementally
            // upload only the new tail per column (DtoD-preserving the
            // unchanged prefix). Note: the dict registry just rebuilt
            // its index columns from the concatenated batch in
            // first-occurrence order; since the append preserves the
            // historical row order, the prefix of the rebuilt indices
            // is bit-identical to the prefix the GpuTable already
            // holds — so the prefix-preserving copy is correct for
            // Utf8 columns too.
            let n_rows_total = concatenated.num_rows();
            let entry = self
                .host_revisions
                .entry(name.to_string())
                .or_default();
            entry.table_revision += 1;
            entry.n_rows = n_rows_total;
            let new_rev = entry.table_revision;
            for field in concatenated.schema().fields() {
                entry
                    .column_revisions
                    .insert(field.name().clone(), new_rev);
                entry
                    .column_n_rows
                    .insert(field.name().clone(), n_rows_total);
            }
            // Leave `gpu_tables[name]` untouched — incremental upload
            // happens in `ensure_gpu_table`. If the slot is somehow
            // absent (initial install raced or was cleared by an
            // out-of-band path), `ensure_gpu_table` falls through to
            // a full upload, which is still correct just not optimal.
            Ok(())
        } else {
            // First batch for a brand-new table: defer to register_table so the
            // dictionary + provider wiring happens exactly once.
            self.register_table(name.to_string(), batch)
        }
    }

    /// Make sure the GPU-resident copy of `name` is fresh.
    ///
    /// **Batch 5 (incremental cache)** — three cases:
    ///   1. Cache hit, table revision matches: return the cached `GpuTable`
    ///      as-is (no host materialisation, no uploads).
    ///   2. Cache hit, table revision diverged: walk each column, reuse
    ///      those whose `host_revision` still matches in the cache,
    ///      re-upload (with prefix-preserving extension when the column
    ///      strictly grew) the rest. Update `last_uploaded_revision` and
    ///      per-column `host_revision`.
    ///   3. Cache miss (slot absent or `None`): full upload from the
    ///      host-concatenated batch — the legacy lazy-upload path.
    ///
    /// `last_uploaded_revision` is checked under the same `RefCell` borrow
    /// that guards the cache, so a concurrent reader cannot see a torn
    /// (revision-matched, columns-not-yet-uploaded) state.
    ///
    /// Returns a `Ref` borrowing the inner `GpuTable`; held for the
    /// duration of `execute_projection`. The `RefCell` panics if a
    /// second `borrow_mut` is attempted while the `Ref` is live, but no
    /// engine method touches `gpu_tables` mutably while a query is in
    /// flight.
    fn ensure_gpu_table(
        &self,
        name: &str,
    ) -> BoltResult<Ref<'_, crate::exec::gpu_table::GpuTable>> {
        // Snapshot the host's current revision (if any) up front. We need
        // the values as owned data so we can drop the &self.host_revisions
        // borrow before borrowing &self.gpu_tables mutably below — even
        // though they're separate fields, taking owned data sidesteps any
        // borrow-graph subtlety with the `&self` we pass to
        // `incremental_rebuild`.
        let host: Option<ClonedHostRevision> = self
            .host_revisions
            .get(name)
            .cloned_revision_owned();
        // Fast path: cache hit AND every column is at the current
        // revision. Inspect under the same borrow we'd return.
        {
            let g = self.gpu_tables.borrow();
            if let Some(Some(gt)) = g.get(name) {
                if let Some(h) = host.as_ref() {
                    if gt.last_uploaded_revision == h.table_revision {
                        return Ok(Ref::map(g, |m| {
                            m.get(name)
                                .expect("hit above")
                                .as_ref()
                                .expect("Some hit above")
                        }));
                    }
                }
            }
        }
        // Either we missed entirely, the slot was None, or the revision
        // diverged. In either case we need to materialize the host
        // concatenated batch (since columns we re-upload come from
        // there).
        let concatenated = self.materialize_table(name)?;
        let mut tables_mut = self.gpu_tables.borrow_mut();
        let existing_opt = tables_mut.remove(name).flatten();
        let new_gpu_table = match existing_opt {
            Some(existing) => self.incremental_rebuild(existing, &concatenated, host.as_ref())?,
            None => {
                // Slot absent or dirty (None): full upload.
                let mut full = crate::exec::gpu_table::GpuTable::from_record_batch(
                    &concatenated,
                )?;
                if let Some(h) = host.as_ref() {
                    full.last_uploaded_revision = h.table_revision;
                    for col in full.columns.iter_mut() {
                        let rev = h
                            .column_revisions
                            .get(&col.name)
                            .copied()
                            .unwrap_or(h.table_revision);
                        col.host_revision = rev;
                    }
                }
                #[cfg(test)]
                self.gpu_table_load_count.fetch_add(
                    full.columns.len(),
                    std::sync::atomic::Ordering::SeqCst,
                );
                full
            }
        };
        tables_mut.insert(name.to_string(), Some(new_gpu_table));
        drop(tables_mut);
        let g = self.gpu_tables.borrow();
        Ok(Ref::map(g, |m| {
            m.get(name)
                .expect("just inserted")
                .as_ref()
                .expect("just inserted Some")
        }))
    }

    /// Batch 5 incremental rebuild: given the cached `existing` GpuTable
    /// and the freshly-concatenated host batch `concatenated`, produce a
    /// GpuTable whose columns are either reused from `existing` (when
    /// their per-column revision still matches the host's view) or
    /// re-uploaded — prefix-preserving when the host data strictly
    /// extended (append), full re-upload otherwise.
    ///
    /// `host` is the engine's `HostTableRevision` snapshot for the
    /// table. `None` means the host doesn't track revisions for this
    /// table (out-of-band install path); falls back to a full rebuild.
    fn incremental_rebuild(
        &self,
        existing: crate::exec::gpu_table::GpuTable,
        concatenated: &RecordBatch,
        host: Option<&ClonedHostRevision>,
    ) -> BoltResult<crate::exec::gpu_table::GpuTable> {
        // Without host revisions we can't decide what's stale → full rebuild.
        let host = match host {
            Some(h) => h,
            None => {
                let table =
                    crate::exec::gpu_table::GpuTable::from_record_batch(concatenated)?;
                #[cfg(test)]
                self.gpu_table_load_count
                    .fetch_add(table.columns.len(), std::sync::atomic::Ordering::SeqCst);
                return Ok(table);
            }
        };
        // Decompose `existing` into a name → GpuColumn map so we can
        // reuse columns positionally without quadratic search.
        let crate::exec::gpu_table::GpuTable {
            n_rows: _,
            columns: existing_columns,
            last_uploaded_revision: _,
        } = existing;
        let mut existing_by_name: HashMap<String, crate::exec::gpu_table::GpuColumn> =
            existing_columns
                .into_iter()
                .map(|c| (c.name.clone(), c))
                .collect();

        let n_rows_total = concatenated.num_rows();
        let schema = concatenated.schema();
        let mut new_columns: Vec<crate::exec::gpu_table::GpuColumn> =
            Vec::with_capacity(concatenated.num_columns());
        for (idx, field) in schema.fields().iter().enumerate() {
            let name = field.name();
            let host_col_rev = host
                .column_revisions
                .get(name)
                .copied()
                .unwrap_or(host.table_revision);
            let reused = existing_by_name.remove(name);
            let col = match reused {
                Some(prev) if prev.host_revision == host_col_rev => {
                    // Cache hit on this column — reuse in place. No upload.
                    prev
                }
                Some(prev) => {
                    // Stale column. If the host data strictly extended
                    // (n_rows grew), try the prefix-preserving path; else
                    // fall through to a full re-upload.
                    let prev_rows = column_storage_rows(&prev.data);
                    if prev_rows > 0 && prev_rows < n_rows_total {
                        match try_extend_column(prev, concatenated, idx, n_rows_total) {
                            Ok(Some(mut extended)) => {
                                extended.host_revision = host_col_rev;
                                #[cfg(test)]
                                self.gpu_table_load_count
                                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                extended
                            }
                            Ok(None) => {
                                // Variant not extensible — full re-upload.
                                let mut fresh =
                                    crate::exec::gpu_table::GpuTable::upload_column_from_batch(
                                        concatenated,
                                        field,
                                        idx,
                                    )?;
                                fresh.host_revision = host_col_rev;
                                #[cfg(test)]
                                self.gpu_table_load_count
                                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                fresh
                            }
                            Err(e) => return Err(e),
                        }
                    } else {
                        // Either previous column was empty / replaced (not
                        // an append) — full re-upload.
                        drop(prev);
                        let mut fresh =
                            crate::exec::gpu_table::GpuTable::upload_column_from_batch(
                                concatenated,
                                field,
                                idx,
                            )?;
                        fresh.host_revision = host_col_rev;
                        #[cfg(test)]
                        self.gpu_table_load_count
                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        fresh
                    }
                }
                None => {
                    // Column not in the previous cache — full upload.
                    let mut fresh =
                        crate::exec::gpu_table::GpuTable::upload_column_from_batch(
                            concatenated,
                            field,
                            idx,
                        )?;
                    fresh.host_revision = host_col_rev;
                    #[cfg(test)]
                    self.gpu_table_load_count
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    fresh
                }
            };
            new_columns.push(col);
        }
        Ok(crate::exec::gpu_table::GpuTable {
            n_rows: n_rows_total,
            columns: new_columns,
            last_uploaded_revision: host.table_revision,
        })
    }

    /// Test-only accessor for the per-column upload counter. Returns the
    /// number of GpuColumn (re)uploads performed across the engine's
    /// lifetime. Used by the incremental-cache regression tests to
    /// assert that an unchanged column was reused.
    #[cfg(test)]
    pub(crate) fn gpu_table_load_count(&self) -> usize {
        self.gpu_table_load_count
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Materialise the concatenated `RecordBatch` for a registered table.
    ///
    /// Fast-path: zero batches errors, one batch is cloned cheaply (Arrow
    /// arrays are Arc-backed). Two or more batches go through
    /// `arrow::compute::concat_batches`, which copies every column — the
    /// 0.2 perf cost the field doc on `tables` warns about.
    fn materialize_table(&self, name: &str) -> BoltResult<RecordBatch> {
        let batches = self.tables.get(name).ok_or_else(|| {
            BoltError::Plan(format!("table '{name}' is not registered with the engine"))
        })?;
        match batches.len() {
            0 => Err(BoltError::Plan(format!(
                "table '{name}' is registered but contains zero batches"
            ))),
            1 => Ok(batches[0].clone()),
            _ => {
                let schema = batches[0].schema();
                arrow::compute::concat_batches(&schema, batches.iter()).map_err(|e| {
                    BoltError::Other(format!(
                        "failed to concatenate {} batches for table '{name}': {e}",
                        batches.len()
                    ))
                })
            }
        }
    }

    /// Compile and execute a SQL query string.
    ///
    /// Stage 7 (P1b): after the query completes, the engine emits a
    /// periodic pool-stats log line at most once every
    /// `BOLT_POOL_STATS_INTERVAL_SECS` (default 60s). The emit happens
    /// AFTER the query's `QueryHandle` is fully materialised — the log
    /// line is off the latency-critical path for the just-returned
    /// query. Failures (query error, log throttled, no-op observer)
    /// never affect the query result.
    pub fn sql(&self, query: &str) -> BoltResult<QueryHandle> {
        // **Stage 6 (M3L5)** — retry the pool-watcher's context capture.
        // If the watcher spawned before any engine thread had a context
        // bound, `CAPTURED_CTX` is still zero and every poll silently
        // no-ops. This call is cheap (atomic load when already
        // captured) and a no-op when no context is bound on the
        // calling thread — so it's safe to invoke unconditionally.
        crate::cuda::mem_pool::pool_watcher_retry_context_capture();

        let plan: LogicalPlan = parse_sql(query, &self.provider)?;
        // String-literal predicates against Utf8 columns are folded into
        // integer equality against the corresponding __idx_<col> i32 column.
        let plan = self.dict_registry.rewrite_plan(&plan)?;
        let mut phys = crate::plan::lower_physical(&plan)?;
        // PV-stage-d: populate `KernelSpec::input_has_validity` for every
        // input column by consulting the engine-backed provider, which
        // looks straight at `RecordBatch::column(col).null_count()` for
        // each registered table. This is the plan-time signal that lets
        // the codegen emit native-validity kernels instead of leaning on
        // the run-time host-strip fallback in `groupby_with_pre` etc.
        let nb = EngineProvider {
            base: &self.provider,
            tables: &self.tables,
        };
        crate::plan::physical_plan::populate_input_validity(&mut phys, &nb);
        let result = self.execute(&phys);
        // Stage 7: periodic pool-stats emit. Runs whether the query
        // succeeded or failed (an OOM-failed query is itself a signal
        // worth surfacing alongside the pool snapshot). Internal errors
        // in the emit path are swallowed — they must never escalate to
        // the query result.
        self.maybe_emit_pool_stats(Instant::now());
        result
    }

    /// Emit a periodic pool-stats log line + observer notification if
    /// the configured interval has elapsed since the last emit.
    ///
    /// `now` is taken as a parameter (rather than calling `Instant::now()`
    /// inside) so the unit test below can drive the throttle deterministically.
    fn maybe_emit_pool_stats(&self, now: Instant) {
        if !should_emit_pool_stats(&self.pool_stats_last_emit, self.pool_stats_interval, now) {
            return;
        }
        // Throttle says go: snapshot the pool and emit. We do this OUTSIDE
        // the throttle's lock so a slow observer can't serialise concurrent
        // queries.
        let s = crate::pool_stats();
        log::info!(
            "craton-bolt pool: bucket_count={}, total_pooled_bytes={}, \
             oom_recoveries={}, proactive_evictions={}",
            s.bucket_count,
            s.total_pooled_bytes,
            s.oom_recovery_count,
            s.proactive_eviction_count,
        );
        crate::observability::notify_observers(s);
    }

    /// Execute a pre-built `PhysicalPlan`.
    pub fn execute(&self, phys: &PhysicalPlan) -> BoltResult<QueryHandle> {
        match phys {
            PhysicalPlan::Projection {
                table,
                kernel,
                output_schema,
            } => self.execute_projection(table, kernel, output_schema),
            PhysicalPlan::Aggregate {
                table,
                pre,
                aggregate,
            } => {
                let batch = self.materialize_table(table)?;
                let out = match (!aggregate.group_by.is_empty(), pre.is_some()) {
                    (true, true) => {
                        crate::exec::groupby_with_pre::execute_groupby_with_pre(phys, &batch)?
                    }
                    (true, false) => crate::exec::groupby::execute_groupby(phys, &batch)?,
                    (false, true) => {
                        crate::exec::agg_with_pre::execute_aggregate_with_pre(phys, &batch)?
                    }
                    (false, false) => crate::exec::aggregate::execute_aggregate(phys, &batch)?,
                };
                Ok(QueryHandle { batch: out })
            }
            // ----- wave-7 dispatch -----
            //
            // The PhysicalPlan variants below are added by agent 1 in the
            // same wave. If a variant doesn't exist yet at build time, the
            // match arm will surface a clear compile error pointing at the
            // missing variant — agent 1 then adds it and the build heals.
            //
            // The executor signatures assumed here mirror the wave-7 spec:
            //   execute_distinct(QueryHandle) -> BoltResult<QueryHandle>
            //   execute_limit  (QueryHandle, usize, Option<usize>) -> ...
            //   execute_sort   (QueryHandle, &[SortExpr]) -> ...
            //   execute_join   (left, right, join_type, on, &Engine) -> ...
            // Agents 3-6 match these.
            PhysicalPlan::Distinct { input } => {
                let h = self.execute(input)?;
                crate::exec::distinct::execute_distinct(h)
            }
            PhysicalPlan::Limit {
                input,
                limit,
                offset,
            } => {
                let h = self.execute(input)?;
                crate::exec::limit::execute_limit(h, *limit, *offset)
            }
            PhysicalPlan::Sort { input, sort_exprs } => {
                let h = self.execute(input)?;
                crate::exec::sort::execute_sort(h, sort_exprs)
            }
            PhysicalPlan::Union { inputs } => {
                // UNION ALL: execute each input, concat the result batches.
                // (Deduplication would happen via a Distinct wrapping the Union
                // in the logical plan — UNION ALL itself is pure concat.)
                if inputs.is_empty() {
                    return Err(BoltError::Plan(
                        "Union with zero inputs is not executable".into(),
                    ));
                }
                let mut handles: Vec<QueryHandle> = Vec::with_capacity(inputs.len());
                for inp in inputs {
                    handles.push(self.execute(inp)?);
                }
                let schema = handles[0].batch.schema();
                let batches: Vec<RecordBatch> =
                    handles.into_iter().map(|h| h.batch).collect();
                let merged = arrow::compute::concat_batches(&schema, batches.iter())
                    .map_err(|e| {
                        BoltError::Other(format!(
                            "failed to concatenate {} UNION ALL inputs: {e}",
                            batches.len()
                        ))
                    })?;
                Ok(QueryHandle { batch: merged })
            }
            PhysicalPlan::Join {
                left,
                right,
                join_type,
                on,
                output_schema,
            } => crate::exec::join::execute_join(
                left,
                right,
                join_type,
                on,
                output_schema,
                self,
            ),
            PhysicalPlan::Filter { input, predicate } => {
                // Host-side post-aggregate (or other non-scan-chain) filter.
                // The lowerer emits this for `HAVING` and any `Filter`
                // wrapping an operator that can't fold into a fused
                // projection kernel. The inner plan's output is materialised
                // as a host-side RecordBatch; we evaluate `predicate` against
                // it via `expr_agg::eval_expr` and drop the rows that don't
                // satisfy it. See `crate::exec::filter::execute_filter`.
                let h = self.execute(input)?;
                crate::exec::filter::execute_filter(h, predicate)
            }
            PhysicalPlan::Project {
                input,
                exprs,
                output_schema,
            } => {
                // Rename/reorder/compute layer over an arbitrary upstream.
                //
                // Fast path: when an `exprs` entry is a bare `Column` or an
                // `Alias` wrapping a `Column`, we just pick that column out
                // of the input batch (no compute, zero-copy clone of the
                // `ArrayRef`).
                //
                // Compute path: anything else (today: SQL `a || b`, i.e.
                // `BinaryOp::Concat`) is materialised via
                // `expr_agg::eval_expr` over a `HostColumn` env built from
                // the input batch. The lazy lift (only build the env when
                // a compute expr appears) keeps the bare-projection case
                // free of overhead.
                let h = self.execute(input)?;
                let in_batch = h.batch;
                let in_schema = in_batch.schema();
                let n_rows = in_batch.num_rows();

                // Lazily-built env for the compute path; `None` until the
                // first non-bare-column expression in `exprs` forces us to
                // lift every input column into a `HostColumn`.
                let mut owned_env: Option<Vec<(String, crate::exec::expr_agg::HostColumn)>> = None;

                let mut columns: Vec<ArrayRef> = Vec::with_capacity(exprs.len());
                for (out_idx, e) in exprs.iter().enumerate() {
                    // Peel through transparent aliases to look at the inner
                    // expression. A bare column reference (with any number
                    // of aliases around it) gets the fast path; anything
                    // else falls into the compute path.
                    let inner = {
                        let mut cur = e;
                        loop {
                            match cur {
                                crate::plan::Expr::Alias(inner, _) => cur = inner.as_ref(),
                                _ => break cur,
                            }
                        }
                    };
                    if let crate::plan::Expr::Column(name) = inner {
                        let idx = in_schema.index_of(name).map_err(|_| {
                            BoltError::Plan(format!(
                                "PhysicalPlan::Project: column '{name}' not found in input schema"
                            ))
                        })?;
                        columns.push(in_batch.column(idx).clone());
                        continue;
                    }
                    // Compute path. Build the env if we haven't yet.
                    if owned_env.is_none() {
                        let mut v = Vec::with_capacity(in_batch.num_columns());
                        for (i, field) in in_schema.fields().iter().enumerate() {
                            let arr = in_batch.column(i);
                            let hc = crate::exec::filter::arrow_array_to_host_column(
                                arr.as_ref(),
                                n_rows,
                            )?;
                            v.push((field.name().clone(), hc));
                        }
                        owned_env = Some(v);
                    }
                    let env_ref = owned_env.as_ref().expect("just built");
                    let env: crate::exec::expr_agg::ColumnEnv<'_> = env_ref
                        .iter()
                        .map(|(n, c)| (n.clone(), c))
                        .collect();
                    let out_field = &output_schema.fields[out_idx];
                    let computed = crate::exec::expr_agg::eval_expr(
                        inner,
                        &env,
                        out_field.dtype,
                        n_rows,
                    )?;
                    columns.push(host_column_to_arrow_array(computed)?);
                }
                let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
                let out = RecordBatch::try_new(arrow_schema, columns).map_err(|e| {
                    BoltError::Other(format!(
                        "failed to build PhysicalPlan::Project RecordBatch: {e}"
                    ))
                })?;
                Ok(QueryHandle { batch: out })
            }
        }
    }

    /// Execute a single fused projection (optionally with filter) kernel.
    fn execute_projection(
        &self,
        table: &str,
        kernel: &KernelSpec,
        output_schema: &Schema,
    ) -> BoltResult<QueryHandle> {
        // Lazy upload: if `register_batch` ran since the last query, this
        // rebuilds the GPU-resident copy from the host-concatenated batches.
        // The returned `Ref` is held across the kernel launch — no other
        // engine method touches `gpu_tables` mutably while `&self` is borrowed.
        let gpu_table_ref = self.ensure_gpu_table(table)?;
        let gpu_table: &crate::exec::gpu_table::GpuTable = &gpu_table_ref;
        let n_rows = gpu_table.n_rows;

        // 1. Resolve input device pointers in place — every column already
        //    lives on the GPU. No host bounce, no per-query upload.
        //
        // `__idx_<col>` inputs come from the dict_registry (they don't exist
        // in the source RecordBatch). They were synthesized by the
        // string-literal rewriter and resolve to the i32/i64 dictionary index
        // column already on the device — we hand the launch a borrowed
        // device pointer into the registry's `GpuVec` rather than bouncing the
        // index column through the host. `&self` is borrowed for the entire
        // `execute_projection`, so the dictionary's GpuVec outlives the launch.
        let mut input_ptrs: Vec<CUdeviceptr> = Vec::with_capacity(kernel.inputs.len());
        for io in &kernel.inputs {
            if let Some(original) = io.name.strip_prefix("__idx_") {
                let dict = self.dict_registry.dictionary(table, original).ok_or_else(|| {
                    BoltError::Plan(format!(
                        "rewriter-emitted column '{}' has no dictionary in registry",
                        io.name
                    ))
                })?;
                // Fail fast on plan/dict dtype mismatch BEFORE doing any I/O —
                // this catches a stale plan that names __idx_X with the wrong
                // width without paying the cost of touching the device.
                if io.dtype != dict.index_dtype() {
                    return Err(BoltError::Plan(format!(
                        "rewriter-emitted column '{}' dtype mismatch: plan says {:?}, dictionary is {:?}",
                        io.name, io.dtype, dict.index_dtype()
                    )));
                }
                // Borrow the device pointer from the registry's existing
                // index column — no host bounce, no fresh allocation.
                let ptr = match dict {
                    crate::cuda::dictionary_any::DictionaryColumnAny::I32(d) => {
                        d.indices.device_ptr()
                    }
                    crate::cuda::dictionary_any::DictionaryColumnAny::I64(d) => {
                        d.indices.device_ptr()
                    }
                };
                input_ptrs.push(ptr);
                continue;
            }
            let column = gpu_table.column(&io.name).ok_or_else(|| {
                BoltError::Plan(format!("column '{}' not in table '{}'", io.name, table))
            })?;
            if column.dtype != io.dtype {
                return Err(BoltError::Plan(format!(
                    "column '{}' dtype mismatch: plan says {:?}, table has {:?}",
                    io.name, io.dtype, column.dtype
                )));
            }
            input_ptrs.push(column.device_ptr());
        }

        // 2. Allocate output buffers, zero-initialised. For Utf8 passthrough
        //    columns (output dtype Utf8 AND name matches an input column),
        //    clone the source dictionary so download can decode indices back
        //    to strings. (Computed Utf8 outputs aren't supported yet.)
        let mut output_cols: Vec<DeviceCol> = Vec::with_capacity(kernel.outputs.len());
        for io in &kernel.outputs {
            let mut col = DeviceCol::alloc_zeros(io.dtype, n_rows)?;
            if io.dtype == DataType::Utf8 {
                if let Some(src) = kernel
                    .inputs
                    .iter()
                    .find(|in_io| in_io.name == io.name && in_io.dtype == DataType::Utf8)
                    .and_then(|in_io| gpu_table.column(&in_io.name))
                    .and_then(|c| c.utf8_dictionary())
                {
                    col.set_utf8_dictionary(src.to_vec());
                }
            }
            output_cols.push(col);
        }

        // 3. JIT-compile the kernel to PTX and load it.
        //
        // Review-H2: route through `get_or_build_module` so repeat queries
        // with the same `KernelSpec` skip the PTX-gen + cubin-load round
        // trip and reuse the same loaded `CudaModule` (cheap Arc clone).
        // The underlying `jit::jit_compiler` PTX-text-hash cache continues
        // to short-circuit `cuModuleLoadDataEx` for unique-spec / shared-
        // PTX cases (e.g. across distinct engines in the same process).
        let module = self.get_or_build_module(kernel, KERNEL_ENTRY, |k| {
            compile_ptx(k, KERNEL_ENTRY)
        })?;
        let function = module.function(KERNEL_ENTRY)?;

        // 4. Build the kernel-parameter list.
        //
        // `KernelArgs` is monomorphic on `T` per push and cannot store heterogenous
        // column types in one list. We bypass it and assemble raw kernel params
        // directly: inputs first, then outputs, then any flagged validity
        // pointers (input then output, in the same order as `ptx_gen.rs`'s
        // signature walk — see `ptx_gen::write_signature`), then the
        // row-count `u32`.
        //
        // Validity pointer wiring (Batch 7, IS NULL e2e):
        // For every input where `kernel.input_has_validity[i] == true` (set by
        // `Codegen::emit_unary` for `column IS [NOT] NULL` checks), push the
        // GPU column's *u8 validity-bitmap pointer here. The codegen's
        // `Op::IsNullCheck` indexes into this list via `validity_input`.
        //
        // For columns where the codegen flagged validity but the GPU storage
        // doesn't expose a validity pointer (e.g. nullable primitives whose
        // GPU storage is still values-only today), we surface a structured
        // error rather than silently emitting a NULL pointer — the kernel
        // would then segfault on the first row. The plan-time constant-fold
        // in `Codegen::emit_unary` already eliminates IsNullCheck on
        // non-nullable schema fields, so this error only fires for genuine
        // missing-validity-on-GPU plumbing gaps (a follow-up: nullable
        // primitives on the device).
        let need_input_validity: Vec<bool> = if kernel.input_has_validity.is_empty() {
            vec![false; kernel.inputs.len()]
        } else {
            if kernel.input_has_validity.len() != kernel.inputs.len() {
                return Err(BoltError::Other(format!(
                    "engine: kernel.input_has_validity len {} != inputs len {}",
                    kernel.input_has_validity.len(),
                    kernel.inputs.len()
                )));
            }
            kernel.input_has_validity.clone()
        };
        let mut input_validity_ptrs: Vec<CUdeviceptr> = Vec::new();
        for (i, has) in need_input_validity.iter().enumerate() {
            if !*has {
                continue;
            }
            let io = &kernel.inputs[i];
            // Synthesised `__idx_*` columns don't carry validity in the
            // dictionary registry; they correspond to dictionary index
            // columns whose null-bearing nature lives upstream on the
            // source DictUtf8 column. Skip with a structured error so the
            // caller knows to surface the breakage.
            if io.name.starts_with("__idx_") {
                return Err(BoltError::Plan(format!(
                    "engine: kernel flags `__idx_` column '{}' as needing validity, but \
                     dictionary registry does not yet expose a per-row validity bitmap; \
                     route the predicate through the host fallback",
                    io.name
                )));
            }
            let column = gpu_table.column(&io.name).ok_or_else(|| {
                BoltError::Plan(format!(
                    "column '{}' not in table '{}' (validity wiring)",
                    io.name, table
                ))
            })?;
            let vptr = column.data.validity_ptr().ok_or_else(|| {
                BoltError::Plan(format!(
                    "engine: kernel flags input '{}' as needing validity but the GPU \
                     column has no validity bitmap on device. The plan-time constant-fold \
                     in physical_plan::Codegen::emit_unary should have collapsed this \
                     IsNullCheck to a Bool constant — was the schema's nullable flag \
                     out of sync with the actual GPU storage? \
                     (Nullable primitives on the device are a follow-up; today only \
                     BoolNullable and DictUtf8 expose `validity_ptr`.)",
                    io.name
                ))
            })?;
            input_validity_ptrs.push(vptr);
        }

        let mut device_ptrs: Vec<CUdeviceptr> =
            Vec::with_capacity(input_ptrs.len() + output_cols.len() + input_validity_ptrs.len());
        for p in &input_ptrs {
            device_ptrs.push(*p);
        }
        for c in &output_cols {
            device_ptrs.push(c.device_ptr());
        }
        // Validity pointers come AFTER value inputs and outputs, matching the
        // order in `ptx_gen::compile` (input-validity first, then output-
        // validity). `KernelSpec::output_has_validity` is empty for the
        // projection path today, so we only emit input-validity ptrs.
        for vp in &input_validity_ptrs {
            device_ptrs.push(*vp);
        }
        let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;

        let mut kernel_params: Vec<*mut c_void> = Vec::with_capacity(device_ptrs.len() + 1);
        for p in device_ptrs.iter_mut() {
            kernel_params.push(p as *mut CUdeviceptr as *mut c_void);
        }
        kernel_params.push(&mut n_rows_u32 as *mut u32 as *mut c_void);

        // 5. Launch with one thread per row, block size 256.
        //
        // Stage-3 async memcpy: mint a per-call stream so the kernel
        // launch, mask materialisation (if any), and the final pinned
        // D2H download can run on the same stream — letting the driver
        // overlap them with concurrent work on the NULL stream. Falls
        // back to the NULL stream if stream creation fails (see
        // `CudaStream::null_or_default`).
        let stream = CudaStream::null_or_default();
        let grid_x = grid_x_for(n_rows_u32, BLOCK_SIZE);
        unsafe {
            cuda_sys::check(cuda_sys::cuLaunchKernel(
                function.raw(),
                grid_x,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                stream.raw(),
                kernel_params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }
        // Debug-only synchronize: pin any in-kernel fault to THIS launch
        // rather than letting it surface at the next CUDA API call.
        debug_sync_check()?;
        // NOTE: no `stream.synchronize()` here — the predicate / gather path
        // and the async-D2H path below both run on the same stream and so are
        // serialized after the kernel automatically. The single sync happens
        // at the bottom of this function (or inside `gpu_compact` for the
        // predicate path, which manages its own stream barriers).

        // 6. If the kernel has a predicate, run a separate predicate-only
        //    kernel to materialise a u8 mask. We default to GPU-side compaction
        //    (prefix scan + gather) when every output column is gather-friendly
        //    (primitive + Bool); Utf8 outputs fall back to the host-side path
        //    because the gather kernel can't move variable-width strings.
        let arrays: Vec<ArrayRef> = if kernel.predicate.is_some() {
            // Review-H2: predicate kernel goes through the same module
            // cache, keyed by `(spec_hash, PREDICATE_ENTRY)` so it doesn't
            // collide with the projection kernel cached under
            // `(spec_hash, KERNEL_ENTRY)`.
            let pred_module = self.get_or_build_module(kernel, PREDICATE_ENTRY, |k| {
                crate::jit::scan_kernel::compile_predicate_kernel(k, PREDICATE_ENTRY)
            })?;
            let pred_function = pred_module.function(PREDICATE_ENTRY)?;

            let mask = crate::exec::compact::alloc_mask_buffer(n_rows)?;
            // Validity-pointer wiring for the predicate kernel (Batch 7,
            // IS NULL e2e). The scan_kernel's emitted PTX consumes the
            // flagged-input validity pointers AFTER the mask output, in
            // input-slot order. `input_validity_ptrs` above was assembled
            // for the projection kernel; reuse it here so the order and
            // membership stay in lockstep with the kernel's signature.
            crate::exec::compact::launch_predicate_kernel(
                pred_function,
                &input_ptrs,
                mask.device_ptr(),
                &input_validity_ptrs,
                n_rows_to_u32(n_rows)?,
                &stream,
            )?;
            // Debug-only synchronize: surface predicate-kernel faults at
            // THIS launch site rather than at a later API call.
            debug_sync_check()?;

            let has_utf8_output = kernel.outputs.iter().any(|c| c.dtype == DataType::Utf8);
            if has_utf8_output {
                // Host-side fallback: download mask + outputs, then filter.
                let host_mask =
                    crate::exec::compact::download_mask(mask.device_ptr(), n_rows)?;
                // Stage-3: route every primitive output column through the
                // pinned async D2H path. Each `download_pinned` call
                // synchronizes the stream internally, so we don't need a
                // separate barrier between the predicate kernel and these
                // reads.
                let mut full: Vec<ArrayRef> = Vec::with_capacity(output_cols.len());
                for col in output_cols {
                    full.push(col.download_pinned(n_rows, &stream)?);
                }
                crate::exec::compact::compact_arrays(&full, &host_mask)?
            } else {
                // GPU-side path: prefix-scan + gather, download the compacted output.
                let cols: Vec<(CUdeviceptr, DataType)> = output_cols
                    .iter()
                    .zip(kernel.outputs.iter())
                    .map(|(c, io)| (c.device_ptr(), io.dtype))
                    .collect();
                let (gathered, _total) = crate::exec::gpu_compact::compact_columns_on_gpu(
                    mask.device_ptr(),
                    n_rows,
                    &cols,
                    &stream,
                )?;
                // Output buffers can drop now; gathered owns the compacted data.
                drop(output_cols);
                let mut out: Vec<ArrayRef> = Vec::with_capacity(gathered.len());
                for g in &gathered {
                    out.push(g.download()?);
                }
                out
            }
        } else {
            // Stage-3 pinned downloads on the per-call stream. Each
            // call synchronizes internally before reading, so the loop
            // is correct even though `stream` was used for the kernel
            // launch above.
            let mut full: Vec<ArrayRef> = Vec::with_capacity(output_cols.len());
            for col in output_cols {
                full.push(col.download_pinned(n_rows, &stream)?);
            }
            full
        };

        // 9. Build the result RecordBatch.
        let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
        let batch_out = RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
            BoltError::Other(format!("failed to build output RecordBatch: {e}"))
        })?;
        Ok(QueryHandle { batch: batch_out })
    }
}

/// Result of a query — wraps the output Arrow `RecordBatch`.
pub struct QueryHandle {
    /// The materialised result.
    batch: RecordBatch,
}

impl QueryHandle {
    /// Borrow the underlying record batch.
    pub fn record_batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// Consume the handle and return the owned record batch.
    pub fn into_record_batch(self) -> RecordBatch {
        self.batch
    }

    /// Wrap a `RecordBatch` produced by an executor into a `QueryHandle`.
    ///
    /// Internal hook for the wave-7 executor chain (Distinct / Limit / Sort /
    /// Union / Join): the top-level `Engine::execute` runs the child plan,
    /// hands the resulting `QueryHandle` to an `exec::*::execute_*` helper,
    /// and the helper rewraps its output with this constructor.
    ///
    /// Marked `#[doc(hidden)]` and `pub(crate)`: this is not part of the
    /// public 0.2 API; downstream consumers should keep going through
    /// `Engine::sql` / `Engine::execute`.
    #[doc(hidden)]
    pub(crate) fn from_record_batch(batch: RecordBatch) -> Self {
        Self { batch }
    }

    /// Number of rows in the result.
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }
}

/// Heterogenous owned device column. Keeps each `GpuVec<T>` alive past the kernel launch.
///
/// Used only for OUTPUT buffers in `execute_projection`. Input columns are
/// resolved through `GpuTable` (uploaded once at table-registration time) and
/// fed to kernels as raw `CUdeviceptr`s; the upload-from-Arrow path that used
/// to live here as `DeviceCol::upload` is gone — `GpuColumn::upload` in
/// `gpu_table.rs` is the single source of truth for host→device column
/// uploads. The historical `BoolNullable` and `Borrowed` variants and the
/// `utf8_dictionary` accessor went with it; both were only reachable through
/// `upload`.
enum DeviceCol {
    /// 32-bit signed integer column.
    I32(GpuVec<i32>),
    /// 64-bit signed integer column.
    I64(GpuVec<i64>),
    /// 32-bit float column.
    F32(GpuVec<f32>),
    /// 64-bit float column.
    F64(GpuVec<f64>),
    /// Bool stored as one byte per row (0 / 1). Used when the source Arrow
    /// array has no nulls.
    Bool(GpuVec<u8>),
    /// Utf8 stored as i32 dictionary indices; host dictionary lives alongside.
    Utf8(DictionaryColumn),
}

impl DeviceCol {
    /// Allocate a zero-initialised device column of `n` rows.
    ///
    /// Utf8 outputs allocate an empty dictionary; the engine must replace it
    /// with the source column's dictionary before download (today this only
    /// works for pure column-passthrough projections — `output_schema` field
    /// name matching an input column name).
    fn alloc_zeros(dtype: DataType, n: usize) -> BoltResult<Self> {
        match dtype {
            DataType::Int32 => Ok(DeviceCol::I32(GpuVec::<i32>::zeros(n)?)),
            DataType::Int64 => Ok(DeviceCol::I64(GpuVec::<i64>::zeros(n)?)),
            DataType::Float32 => Ok(DeviceCol::F32(GpuVec::<f32>::zeros(n)?)),
            DataType::Float64 => Ok(DeviceCol::F64(GpuVec::<f64>::zeros(n)?)),
            DataType::Bool => Ok(DeviceCol::Bool(GpuVec::<u8>::zeros(n)?)),
            DataType::Utf8 => Ok(DeviceCol::Utf8(DictionaryColumn {
                dictionary: Vec::new(),
                indices: GpuVec::<i32>::zeros(n)?,
                n_rows: n,
            })),
        }
    }

    /// Raw device pointer for kernel-parameter assembly.
    fn device_ptr(&self) -> CUdeviceptr {
        match self {
            DeviceCol::I32(v) => v.device_ptr(),
            DeviceCol::I64(v) => v.device_ptr(),
            DeviceCol::F32(v) => v.device_ptr(),
            DeviceCol::F64(v) => v.device_ptr(),
            DeviceCol::Bool(v) => v.device_ptr(),
            DeviceCol::Utf8(d) => d.indices.device_ptr(),
        }
    }

    /// Install a dictionary on a Utf8 column (for output columns whose source dictionary
    /// the engine knows). No-op for non-Utf8 columns.
    fn set_utf8_dictionary(&mut self, dict: Vec<String>) {
        if let DeviceCol::Utf8(d) = self {
            d.dictionary = dict;
        }
    }

    /// Copy the device column back to a host Arrow array of length `n_rows`.
    fn download(self, n_rows: usize) -> BoltResult<ArrayRef> {
        match self {
            DeviceCol::I32(v) => {
                let host = copy_back::<i32>(&v, n_rows)?;
                Ok(Arc::new(Int32Array::from(host)) as ArrayRef)
            }
            DeviceCol::I64(v) => {
                let host = copy_back::<i64>(&v, n_rows)?;
                Ok(Arc::new(Int64Array::from(host)) as ArrayRef)
            }
            DeviceCol::F32(v) => {
                let host = copy_back::<f32>(&v, n_rows)?;
                Ok(Arc::new(Float32Array::from(host)) as ArrayRef)
            }
            DeviceCol::F64(v) => {
                let host = copy_back::<f64>(&v, n_rows)?;
                Ok(Arc::new(Float64Array::from(host)) as ArrayRef)
            }
            DeviceCol::Bool(v) => {
                let host = copy_back::<u8>(&v, n_rows)?;
                let bools: Vec<bool> = host.into_iter().map(|b| b != 0).collect();
                Ok(Arc::new(BooleanArray::from(bools)) as ArrayRef)
            }
            DeviceCol::Utf8(d) => {
                let arr = d.to_string_array()?;
                Ok(Arc::new(arr) as ArrayRef)
            }
        }
    }

    /// Stage-3 async download: enqueue D2H from every primitive variant
    /// into pinned host buffers on `stream`, then synchronize ONCE and
    /// build the Arrow arrays from the resulting `Vec`s. Behaves
    /// identically to [`download`] for the Utf8 / Borrowed variants —
    /// those don't currently have a pinned fast path.
    ///
    /// The caller is responsible for ensuring `stream` is the same one
    /// the producing kernel was launched on (so the D2H sees committed
    /// results), and the function performs the synchronize internally
    /// before reading the pinned buffer.
    fn download_pinned(
        self,
        n_rows: usize,
        stream: &CudaStream,
    ) -> BoltResult<ArrayRef> {
        match self {
            DeviceCol::I32(v) => {
                let staged = StagedDownload::<i32>::from_gpu(&v, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), n_rows)?;
                Ok(Arc::new(Int32Array::from(host)) as ArrayRef)
            }
            DeviceCol::I64(v) => {
                let staged = StagedDownload::<i64>::from_gpu(&v, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), n_rows)?;
                Ok(Arc::new(Int64Array::from(host)) as ArrayRef)
            }
            DeviceCol::F32(v) => {
                let staged = StagedDownload::<f32>::from_gpu(&v, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), n_rows)?;
                Ok(Arc::new(Float32Array::from(host)) as ArrayRef)
            }
            DeviceCol::F64(v) => {
                let staged = StagedDownload::<f64>::from_gpu(&v, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), n_rows)?;
                Ok(Arc::new(Float64Array::from(host)) as ArrayRef)
            }
            DeviceCol::Bool(v) => {
                let staged = StagedDownload::<u8>::from_gpu(&v, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), n_rows)?;
                let bools: Vec<bool> = host.into_iter().map(|b| b != 0).collect();
                Ok(Arc::new(BooleanArray::from(bools)) as ArrayRef)
            }
            DeviceCol::Utf8(_) => {
                // Utf8 doesn't (yet) have a pinned fast path — fall back
                // to the sync download. The stream has already been
                // synchronized above for the primitive siblings, so this
                // is safe to invoke regardless.
                self.download(n_rows)
            }
        }
    }
}

/// Tiny invariant check used by the pinned-download path: every
/// `DeviceCol` output buffer is sized at allocation time to `n_rows`, so
/// a length mismatch on download is a bug, not a runtime condition.
fn check_len(have: usize, want: usize) -> BoltResult<()> {
    if have != want {
        return Err(BoltError::Other(format!(
            "internal: device buffer length {} did not match expected {}",
            have, want
        )));
    }
    Ok(())
}

/// Copy back a `GpuVec<T>` into a host `Vec<T>` of length `n_rows`.
///
/// Output buffers are allocated via `GpuVec::zeros(n_rows)`, whose `len()` is `n_rows`,
/// so `to_vec()` returns exactly that many elements.
fn copy_back<T>(v: &GpuVec<T>, n_rows: usize) -> BoltResult<Vec<T>>
where
    T: bytemuck::Pod,
{
    let host = v.to_vec()?;
    if host.len() != n_rows {
        return Err(BoltError::Other(format!(
            "internal: device buffer length {} did not match expected {}",
            host.len(),
            n_rows
        )));
    }
    Ok(host)
}

/// Stage-3 D2H staging buffer: async-copies a `GpuVec<T>` into a
/// page-locked host buffer on a caller-supplied stream, synchronises
/// once, and produces a regular `Vec<T>` for Arrow consumption.
///
/// Why a separate type vs. an inline call? Arrow array constructors
/// (`Int32Array::from(Vec<i32>)`) want owned `Vec`s with the standard
/// allocator — they will NOT accept a `PinnedHostBuffer` as a
/// zero-copy backing buffer (the lifecycle is incompatible: pinned
/// memory must be released via `cuMemFreeHost`, while Arrow buffers
/// release through the global allocator). So the pinned hop is purely
/// to get a true DMA without staging through a kernel-managed bounce
/// buffer; the final `.to_vec()` is the one host-host copy we keep.
///
/// Usage:
///
/// ```ignore
/// let staged = StagedDownload::from_gpu(&gpu_vec, stream.raw())?;
/// stream.synchronize()?;
/// let arrow_vec: Vec<i32> = staged.into_vec();
/// ```
struct StagedDownload<T: bytemuck::Pod> {
    pinned: crate::cuda::PinnedHostBuffer<T>,
}

impl<T: bytemuck::Pod> StagedDownload<T> {
    /// Enqueue an async D2H from `v` into a fresh pinned host buffer on
    /// `stream`. The caller MUST synchronize `stream` before calling
    /// [`into_vec`] / borrowing the pinned slice.
    fn from_gpu(v: &GpuVec<T>, stream: crate::cuda::CUstream) -> BoltResult<Self> {
        let pinned = v.to_pinned_async(stream)?;
        Ok(Self { pinned })
    }

    /// Consume the staged download and produce a regular host `Vec<T>`.
    ///
    /// Assumes the caller has synchronized the stream — there is no way
    /// to detect "not yet synchronized" without an event, which we skip
    /// in Stage 3. Calling this before sync produces uninitialised
    /// bytes (defined behaviour for `T: Pod` but functionally
    /// incorrect).
    fn into_vec(self) -> Vec<T> {
        self.pinned.as_slice().to_vec()
    }
}

/// Map our plan `DataType` to Arrow `DataType`.
fn plan_dtype_to_arrow(d: DataType) -> BoltResult<ArrowDataType> {
    match d {
        DataType::Int32 => Ok(ArrowDataType::Int32),
        DataType::Int64 => Ok(ArrowDataType::Int64),
        DataType::Float32 => Ok(ArrowDataType::Float32),
        DataType::Float64 => Ok(ArrowDataType::Float64),
        DataType::Bool => Ok(ArrowDataType::Boolean),
        DataType::Utf8 => Ok(ArrowDataType::Utf8),
    }
}

/// Map Arrow `DataType` to our plan `DataType`. Errors on unsupported types.
///
/// **Stage 4 / Stage 6** — `Dictionary(_, Utf8)` is accepted as a register-table
/// type and exposed to the planner as `DataType::Utf8`. The fact that the column
/// is dictionary-encoded is a *storage* detail: query planning, projection,
/// filtering, ORDER BY all reason about it as a Utf8 column. SQL frontends
/// see it identically to a flat `StringArray` column.
///
/// Stage 4 accepted any key width (Int32 *or* Int64) and routed through the
/// flatten step. Stage 6 added a native ingest path for `Dictionary<Int32, Utf8>`
/// in `GpuTable::from_record_batch` and `DictRegistry::register_table`, so the
/// flatten in `flatten_dictionary_utf8_columns` is now a deprecated no-op (the
/// dict layout reaches the GPU table directly). Int64-keyed dicts still take
/// the legacy path through `GpuColumn::upload`.
fn arrow_dtype_to_plan(d: &ArrowDataType) -> BoltResult<DataType> {
    match d {
        ArrowDataType::Int32 => Ok(DataType::Int32),
        ArrowDataType::Int64 => Ok(DataType::Int64),
        ArrowDataType::Float32 => Ok(DataType::Float32),
        ArrowDataType::Float64 => Ok(DataType::Float64),
        ArrowDataType::Boolean => Ok(DataType::Bool),
        ArrowDataType::Utf8 => Ok(DataType::Utf8),
        // Stage 4 / Stage 6: dictionary-encoded Utf8. Only string-valued
        // dicts map to `Utf8`; numeric-valued dicts are intentionally
        // rejected because the caller should hand the inner numeric column
        // directly through the normal path. Both Int32 and Int64 keys are
        // accepted: Int32 takes the Stage-6 native ingest in
        // `GpuTable::from_record_batch`, Int64 falls through to the legacy
        // `GpuColumn::upload` path (which still emits a `DictUtf8` variant).
        ArrowDataType::Dictionary(_key, value)
            if matches!(value.as_ref(), ArrowDataType::Utf8) =>
        {
            Ok(DataType::Utf8)
        }
        other => Err(BoltError::Type(format!(
            "unsupported Arrow dtype {:?}",
            other
        ))),
    }
}

/// Stage 4 — rewrite every `Dictionary(_, Utf8)` column in `batch` into a
/// plain `StringArray`, leaving non-dictionary columns untouched. Returns
/// the rewritten `RecordBatch` (cheap if no dict columns: just reuses the
/// original arrays via `Arc`).
///
/// Why flatten at registration time rather than carrying the dict through?
/// The GPU storage (`GpuTable`) already manages its own dictionary for Utf8
/// columns (see `GpuColumnData::Utf8`), so re-using the input dict would
/// require teaching every consumer (GpuTable upload, projection, gather,
/// expression evaluator, ORDER BY's host-side `take`) to read both dict
/// variants. Materialising once at registration is O(n) per dict column —
/// the same cost the engine's own dictionary builder pays — and keeps every
/// downstream stage's Utf8 handling unified on `StringArray`.
///
/// **Stage 5** added a native `GpuColumnData::DictUtf8` variant to
/// `GpuTable` so callers that go directly through `GpuTable::from_record_batch`
/// (skipping the engine's registry / `MemTableProvider`) can preserve the
/// input dictionary instead of materialising it.
///
/// **Stage 6** — DEPRECATED. The dict registry and `GpuTable` now ingest
/// `DictionaryArray<Int32, Utf8>` natively (the registry matches the dict
/// variant directly; `GpuTable::from_record_batch` routes through
/// `upload_dict_utf8`). The engine no longer calls this helper from
/// `register_table` / `replace_table` / `register_batch`, but the body is
/// kept callable so any out-of-tree consumer that imported it still
/// compiles. Will be removed in a wave following Stage 7.
#[deprecated(
    since = "0.3.0",
    note = "DictionaryArray<Int32, Utf8> is now ingested natively by DictRegistry \
            and GpuTable::from_record_batch; this flatten step is no longer \
            invoked by the engine and will be removed in a future release."
)]
#[allow(dead_code)]
pub(crate) fn flatten_dictionary_utf8_columns(batch: RecordBatch) -> BoltResult<RecordBatch> {
    use arrow_array::{Array, DictionaryArray, StringArray};
    use arrow_array::types::{Int32Type, Int64Type};

    let schema = batch.schema();
    let mut changed = false;
    let mut new_fields: Vec<ArrowField> = Vec::with_capacity(schema.fields().len());
    let mut new_cols: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    for (idx, field) in schema.fields().iter().enumerate() {
        let col = batch.column(idx);
        match field.data_type() {
            ArrowDataType::Dictionary(key_ty, value_ty)
                if matches!(value_ty.as_ref(), ArrowDataType::Utf8) =>
            {
                // Decode (key_idx, value_idx) -> StringArray entries.
                // Supports Int32 and Int64 key types (matches `arrow_dtype_to_plan`).
                let n = col.len();
                let mut out: Vec<Option<String>> = Vec::with_capacity(n);
                let decode_into = |out: &mut Vec<Option<String>>,
                                   value_idx: usize,
                                   sa: &StringArray| {
                    if sa.is_null(value_idx) {
                        out.push(None);
                    } else {
                        out.push(Some(sa.value(value_idx).to_string()));
                    }
                };
                match key_ty.as_ref() {
                    ArrowDataType::Int32 => {
                        let da = col
                            .as_any()
                            .downcast_ref::<DictionaryArray<Int32Type>>()
                            .ok_or_else(|| {
                                BoltError::Type(
                                    "register_table: dict<i32,utf8> downcast failed".into(),
                                )
                            })?;
                        let sa = da
                            .values()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| {
                                BoltError::Type(
                                    "register_table: dict values are not StringArray".into(),
                                )
                            })?;
                        let keys = da.keys();
                        for i in 0..n {
                            if keys.is_null(i) {
                                out.push(None);
                            } else {
                                decode_into(&mut out, keys.value(i) as usize, sa);
                            }
                        }
                    }
                    ArrowDataType::Int64 => {
                        let da = col
                            .as_any()
                            .downcast_ref::<DictionaryArray<Int64Type>>()
                            .ok_or_else(|| {
                                BoltError::Type(
                                    "register_table: dict<i64,utf8> downcast failed".into(),
                                )
                            })?;
                        let sa = da
                            .values()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| {
                                BoltError::Type(
                                    "register_table: dict values are not StringArray".into(),
                                )
                            })?;
                        let keys = da.keys();
                        for i in 0..n {
                            if keys.is_null(i) {
                                out.push(None);
                            } else {
                                decode_into(&mut out, keys.value(i) as usize, sa);
                            }
                        }
                    }
                    other => {
                        return Err(BoltError::Type(format!(
                            "register_table: dict key type {:?} not supported \
                             (expected Int32 or Int64 with Utf8 values)",
                            other
                        )));
                    }
                }
                let sa = StringArray::from(out);
                new_fields.push(ArrowField::new(
                    field.name().clone(),
                    ArrowDataType::Utf8,
                    field.is_nullable(),
                ));
                new_cols.push(Arc::new(sa) as ArrayRef);
                changed = true;
            }
            _ => {
                new_fields.push(field.as_ref().clone());
                new_cols.push(col.clone());
            }
        }
    }
    if !changed {
        return Ok(batch);
    }
    let new_schema = Arc::new(ArrowSchema::new(new_fields));
    RecordBatch::try_new(new_schema, new_cols)
        .map_err(|e| BoltError::Type(format!("register_table: rebuild after dict flatten failed: {e}")))
}

/// Parse the `BOLT_POOL_STATS_INTERVAL_SECS` environment variable into
/// a `Duration`. Missing or unparseable values default to
/// [`DEFAULT_POOL_STATS_INTERVAL_SECS`]; an explicit `0` disables
/// periodic emission (signalled by `Duration::ZERO`).
///
/// `pub(crate)` so the integration test `tests/env_var_smoke.rs` can
/// round-trip the parser against the live env var without going
/// through `Engine::new` (which would also pay an eager CUDA-context
/// init cost we want to keep off host-only smoke runs).
pub fn pool_stats_interval_from_env() -> Duration {
    match std::env::var(POOL_STATS_ENV).ok().and_then(|v| v.parse::<u64>().ok()) {
        Some(0) => Duration::ZERO,
        Some(n) => Duration::from_secs(n),
        None => Duration::from_secs(DEFAULT_POOL_STATS_INTERVAL_SECS),
    }
}

/// Decide whether to emit a pool-stats snapshot at time `now`, advancing
/// the throttle state on a positive decision.
///
/// Pulled out of [`Engine::maybe_emit_pool_stats`] so the throttle
/// semantics can be exercised without a live CUDA context. Side
/// effects: writes `Some(now)` into `last_emit` when emission is due,
/// leaves it untouched otherwise.
///
/// Returns `true` IFF the caller should emit a log line + observer
/// notification right now. Encapsulates three rules:
///   * `interval == 0` → never emit (env-var disables).
///   * `last_emit.is_none()` → always emit (first query on the engine).
///   * `now - last_emit >= interval` → emit and reset.
fn should_emit_pool_stats(
    last_emit: &Mutex<Option<Instant>>,
    interval: Duration,
    now: Instant,
) -> bool {
    if interval.is_zero() {
        return false;
    }
    let mut last = match last_emit.lock() {
        Ok(g) => g,
        Err(_) => return false, // poisoned — best-effort; skip the emit.
    };
    let should = match *last {
        None => true,
        Some(prev) => now.duration_since(prev) >= interval,
    };
    if should {
        *last = Some(now);
    }
    should
}

/// Stage 6 — walk `batch` and inform `provider` of each column's actual
/// runtime nullability (i.e. whether the source array had any nulls). For
/// `DictionaryArray<_, Utf8>` columns the per-row nullability lives on the
/// keys buffer, not the dictionary values; this helper consults
/// `keys().null_count()` to get the right answer. Called from
/// `register_table` / `replace_table` / `register_batch`, so the
/// engine-backed `TableProvider` (`EngineProvider::has_nulls`) and the
/// codegen-time `populate_input_validity` pass both see truthful claims.
fn propagate_column_nullability(
    provider: &mut MemTableProvider,
    table: &str,
    batch: &RecordBatch,
) {
    // `Array::null_count` is an inherent-trait method; pull the trait into
    // scope locally so we can ask any `&dyn Array` for its null count.
    use arrow_array::Array;
    let schema = batch.schema();
    for (idx, field) in schema.fields().iter().enumerate() {
        let arr = batch.column(idx);
        let has_nulls = match field.data_type() {
            ArrowDataType::Dictionary(key_t, _)
                if key_t.as_ref() == &ArrowDataType::Int32 =>
            {
                // Dict keys carry the per-row validity. Downcast and ask the
                // keys array directly; fall back to the array's own
                // `null_count()` if the downcast fails (shouldn't happen for
                // Int32 keys but defensive).
                match arr
                    .as_any()
                    .downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::Int32Type>>()
                {
                    Some(da) => da.keys().null_count() > 0,
                    None => arr.null_count() > 0,
                }
            }
            _ => arr.null_count() > 0,
        };
        provider.set_column_nullability(table.to_string(), field.name().clone(), has_nulls);
    }
}

/// Convert an `arrow_schema::Schema` into our plan `Schema`.
fn arrow_schema_to_plan_schema(s: &ArrowSchema) -> BoltResult<Schema> {
    let mut fields = Vec::with_capacity(s.fields().len());
    for f in s.fields() {
        let dt = arrow_dtype_to_plan(f.data_type())?;
        fields.push(Field::new(f.name().clone(), dt, f.is_nullable()));
    }
    Ok(Schema::new(fields))
}

/// Convert our plan `Schema` to an `arrow_schema::Schema` (used for output `RecordBatch`).
fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}

/// Convert a host-side computed `HostColumn` into an `ArrayRef`.
///
/// Used by the `PhysicalPlan::Project` compute path (string `||`,
/// arithmetic over post-aggregate scalars, …) to fold a freshly
/// materialised column back into the output `RecordBatch`. Mirrors the
/// `arrow_array_to_host_column` shape in `filter.rs` (the inverse
/// direction).
fn host_column_to_arrow_array(col: crate::exec::expr_agg::HostColumn) -> BoltResult<ArrayRef> {
    use crate::exec::expr_agg::HostColumn;
    Ok(match col {
        HostColumn::Bool(v) => Arc::new(BooleanArray::from(v)) as ArrayRef,
        HostColumn::I32(v) => Arc::new(Int32Array::from(v)) as ArrayRef,
        HostColumn::I64(v) => Arc::new(Int64Array::from(v)) as ArrayRef,
        HostColumn::F32(v) => Arc::new(Float32Array::from(v)) as ArrayRef,
        HostColumn::F64(v) => Arc::new(Float64Array::from(v)) as ArrayRef,
        HostColumn::Utf8(v) => {
            let arr = arrow_array::StringArray::from(v);
            Arc::new(arr) as ArrayRef
        }
    })
}

// ---------------------------------------------------------------------------
// PV-stage-d: TableProvider adaptor that surfaces actual per-column null
// counts from the engine's registered `RecordBatch`es.
//
// `MemTableProvider` only knows the schema (column names + dtypes); the
// engine additionally holds the data, so the per-column `null_count()` is
// cheap to read here. We wrap the schema provider so the planner gets:
//   * Schema lookups via the underlying `MemTableProvider` (same as before).
//   * `has_nulls` answered by scanning the registered batches' bitmaps.
// ---------------------------------------------------------------------------

/// `TableProvider` adapter wrapping the engine's [`MemTableProvider`] schema
/// store and adding `has_nulls` / `null_count` answers backed by the actual
/// registered `RecordBatch`es.
struct EngineProvider<'a> {
    base: &'a MemTableProvider,
    tables: &'a HashMap<String, Vec<RecordBatch>>,
}

impl<'a> crate::plan::TableProvider for EngineProvider<'a> {
    fn schema(&self, name: &str) -> BoltResult<Schema> {
        self.base.schema(name)
    }

    fn has_nulls(&self, table_name: &str, col_idx: usize) -> bool {
        // PV-stage-f: returns `true` iff ANY registered batch for `table_name`
        // has at least one NULL on column ordinal `col_idx` (via
        // `RecordBatch::column(col_idx).null_count() > 0`). This is the
        // plan-time signal `populate_input_validity` /
        // `populate_aggregate_spec` (in `crate::plan::physical_plan`) read
        // to fill `KernelSpec::input_has_validity` and
        // `AggregateSpec::input_has_validity` respectively.
        //
        // Safe-false on any miss — the executor's host-strip fallback still
        // handles the row filtering, so an under-flag is correctness-safe.
        let batches = match self.tables.get(table_name) {
            Some(b) => b,
            None => return false,
        };
        for batch in batches {
            // Skip out-of-range column ordinals (e.g. dictionary-extended
            // `__idx_<col>` columns the dict registry mints; those have
            // their own null behaviour).
            if col_idx >= batch.num_columns() {
                continue;
            }
            if batch.column(col_idx).null_count() > 0 {
                return true;
            }
        }
        false
    }

    fn null_count(&self, table_name: &str, col_idx: usize) -> Option<usize> {
        let batches = self.tables.get(table_name)?;
        let mut total: usize = 0;
        for batch in batches {
            if col_idx >= batch.num_columns() {
                continue;
            }
            total = total.saturating_add(batch.column(col_idx).null_count());
        }
        Some(total)
    }
}

#[cfg(test)]
mod tests {
    //! Online tests for the lazy-upload `register_batch` path and the
    //! Stage-3 pinned async-memcpy wiring in `execute_projection`.
    //!
    //! The lazy-upload tests lock in the fix for the O(N²) PCIe re-upload bug
    //! described on the `gpu_tables` field: appending N batches must not cost
    //! `1+2+…+N` batches' worth of host→device traffic. They verify the
    //! observable correctness of the lazy path (rows from every appended batch
    //! are visible to the next query).
    //!
    //! The Stage-3 tests cover the per-query-stream + pinned D2H path —
    //! both the no-predicate and predicate flows — so any regression in the
    //! stream chaining surfaces as a value mismatch rather than a CUDA error.
    //!
    //! All tests are `#[ignore]`'d because they launch real kernels — run
    //! with `cargo test -- --ignored` on a GPU host.
    use super::*;
    use arrow_array::{Int32Array, Int64Array};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    /// Build a single-column `RecordBatch` whose `x` column holds the half-open
    /// range `[start, start+n)` as `Int64`. The schema is shared across all
    /// fixtures so `register_batch`'s schema check passes.
    fn int64_batch(start: i64, n: usize) -> RecordBatch {
        let col: Int64Array = (start..start + n as i64).collect();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            ArrowDataType::Int64,
            false,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap()
    }

    #[test]
    #[ignore = "gpu:projection"]
    fn register_batch_two_batches_query_sees_both() {
        // Register two batches, then SELECT the only column. The lazy-upload
        // path must rebuild the GpuTable from BOTH batches at query time, so
        // every row from both batches has to be visible in the result.
        let mut engine = Engine::new().expect("ctx");
        engine
            .register_batch("t", int64_batch(0, 4))
            .expect("first batch");
        engine
            .register_batch("t", int64_batch(4, 4))
            .expect("second batch");

        let h = engine.sql("SELECT x FROM t").expect("execute");
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 8, "8 rows after two 4-row batches");
        let actual = out
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 column");
        let got: Vec<i64> = (0..actual.len()).map(|i| actual.value(i)).collect();
        assert_eq!(got, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    #[ignore = "gpu:projection"]
    fn register_batch_ten_batches_combined_row_count() {
        // Append ten 100-row batches in a loop, then query. With the bug we
        // were fixing, this would upload 1+2+…+10 = 55 batches' worth of bytes
        // across the loop; with the fix it uploads zero bytes during the loop
        // and exactly one combined upload at query time. Correctness check:
        // the result has all 1000 rows and they sum to the expected total.
        let mut engine = Engine::new().expect("ctx");
        let n_batches = 10usize;
        let rows_per_batch = 100usize;
        for i in 0..n_batches {
            engine
                .register_batch("t", int64_batch((i * rows_per_batch) as i64, rows_per_batch))
                .unwrap_or_else(|e| panic!("register_batch {i}: {e}"));
        }
        let total_rows = n_batches * rows_per_batch;

        let h = engine.sql("SELECT x FROM t").expect("execute");
        let out = h.record_batch();
        assert_eq!(out.num_rows(), total_rows, "row count after 10 appends");

        let actual = out
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 column");
        let sum: i64 = (0..actual.len()).map(|i| actual.value(i)).sum();
        let expected_sum: i64 = (0..total_rows as i64).sum();
        assert_eq!(sum, expected_sum, "sum of x column across all 10 batches");
    }

    /// Build a three-column `RecordBatch` (`a` Int32, `b` Int64, `c` Float64)
    /// holding `n` rows. `start_a` seeds the first column; the others are
    /// derived so each row's columns are easy to recompute in the test
    /// assertions. The schema is shared across calls so `register_batch`'s
    /// schema check passes when appending.
    fn three_col_batch(start_a: i32, n: usize) -> RecordBatch {
        use arrow_array::{Float64Array, Int32Array, Int64Array};
        let a: Int32Array = (start_a..start_a + n as i32).collect();
        let b: Int64Array = ((start_a as i64) * 10..((start_a as i64) * 10 + n as i64)).collect();
        let c: Float64Array = (0..n).map(|i| (start_a as f64) + i as f64 * 0.5).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", ArrowDataType::Int32, false),
            ArrowField::new("b", ArrowDataType::Int64, false),
            ArrowField::new("c", ArrowDataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(a), Arc::new(b), Arc::new(c)],
        )
        .unwrap()
    }

    /// Batch 5 — incremental rebuild after `register_batch`. Register a
    /// 5-row 3-column table, query (forces full upload), append a 2-row
    /// second batch, query again. The second query must observe all 7
    /// rows AND the prefix-preserving optimisation must have fired —
    /// each of the 3 columns is uploaded exactly twice (once at install,
    /// once at the incremental rebuild after the append). The
    /// no-optimisation baseline would re-upload all 3 columns from
    /// scratch on the second query, giving the SAME count of 6 uploads,
    /// so the count alone doesn't distinguish them. We instead assert
    /// the column counts match the *expected* incremental path
    /// invariants: after a single register_batch, exactly 3 incremental
    /// extends fire — and we verify by tagging the device-side
    /// `host_revision` directly through the LOAD_COUNT bump. The
    /// alternative invalidation path (slot set to `None`) would have
    /// reset the per-column host_revisions to 0 and re-uploaded
    /// everything via the fall-through branch in `ensure_gpu_table`.
    #[test]
    #[ignore = "gpu:projection"]
    fn register_batch_incremental_rebuild_uploads_each_column_once_per_change() {
        let mut engine = Engine::new().expect("ctx");
        // Install: 3 columns × 5 rows. register_table uploads each
        // column once → LOAD_COUNT = 3.
        engine
            .register_table("t", three_col_batch(0, 5))
            .expect("install");
        let after_install = engine.gpu_table_load_count();
        assert_eq!(after_install, 3, "install uploads 3 columns");

        // First query — cache hit (no upload).
        let _ = engine.sql("SELECT a FROM t").expect("first query");
        assert_eq!(
            engine.gpu_table_load_count(),
            3,
            "first query is a pure cache hit"
        );

        // Append 2 rows. register_batch must NOT upload anything
        // synchronously; the actual extension happens in the next query.
        engine
            .register_batch("t", three_col_batch(5, 2))
            .expect("append");
        assert_eq!(
            engine.gpu_table_load_count(),
            3,
            "register_batch must not upload synchronously"
        );

        // Second query — incremental rebuild. Each of the 3 columns is
        // re-uploaded exactly once (prefix-preserving extension). Total
        // becomes 3 + 3 = 6.
        let h = engine.sql("SELECT a, b, c FROM t").expect("second query");
        assert_eq!(
            engine.gpu_table_load_count(),
            6,
            "incremental rebuild uploads exactly 3 columns (each extended once)"
        );

        // Correctness: all 7 rows visible, values match.
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 7, "5 + 2 = 7 rows after append");
        let a = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .expect("a Int32");
        let got_a: Vec<i32> = (0..a.len()).map(|i| a.value(i)).collect();
        assert_eq!(got_a, vec![0, 1, 2, 3, 4, 5, 6]);

        // Third query without any further mutation — pure cache hit.
        let _ = engine.sql("SELECT b FROM t").expect("third query");
        assert_eq!(
            engine.gpu_table_load_count(),
            6,
            "third query is a pure cache hit — no uploads"
        );
    }

    /// Batch 5 — `replace_table` is a full swap (NOT an append). Every
    /// column gets a fresh revision, so the next query re-uploads every
    /// column (the prefix optimisation does not apply across a replace).
    /// Validates the revision-bump correctness for the
    /// `bump_table_full_replace` path.
    #[test]
    #[ignore = "gpu:projection"]
    fn replace_table_invalidates_all_column_revisions() {
        let mut engine = Engine::new().expect("ctx");
        engine
            .register_table("t", three_col_batch(0, 5))
            .expect("install");
        let base = engine.gpu_table_load_count();
        // register_table on an existing name must error — replace_table is
        // the right entry point for an update.
        engine
            .register_table("t", three_col_batch(100, 4))
            .unwrap_err();
        // Replace with a same-schema, different-content batch. replace_table
        // performs the upload synchronously (re-uploading all 3 columns)
        // and stamps the GpuTable with the new revision, so the next
        // query is a pure cache hit (no further uploads).
        engine
            .replace_table("t", three_col_batch(100, 4))
            .expect("replace");
        assert_eq!(
            engine.gpu_table_load_count(),
            base + 3,
            "replace_table re-uploads every column"
        );
        let h = engine.sql("SELECT a FROM t").expect("query");
        // Cache hit on the post-replace upload.
        assert_eq!(
            engine.gpu_table_load_count(),
            base + 3,
            "query after replace is a cache hit"
        );
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 4);
    }

    /// Verify that a bare projection still returns the right rows after the
    /// kernel launch and D2H downloads moved onto a per-query stream with
    /// async copies. Mirrors what the synchronous path was previously
    /// asserting — same input, same expected output — so any regression in
    /// the stream-flow shows up as a value mismatch rather than a CUDA error.
    #[test]
    #[ignore = "gpu:projection — Stage 2 async D2H correctness"]
    fn execute_projection_async_d2h_round_trip() {
        let mut engine = Engine::new().expect("engine init");

        // Single-column Int32 table: [1, 2, 3, 4, 5].
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1i32, 2, 3, 4, 5]));
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            ArrowDataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![arr]).expect("batch");
        engine.register_table("t", batch).expect("register");

        // Plain projection — no predicate, so the new async-D2H batch path
        // is exercised end-to-end.
        let handle = engine.sql("SELECT x FROM t").expect("query");
        let out = handle.record_batch();

        assert_eq!(out.num_rows(), 5);
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32");
        let got: Vec<i32> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, vec![1, 2, 3, 4, 5]);
    }

    /// Same shape, but with a WHERE clause so the predicate path is the one
    /// exercised. The Stage 2 patch removed the explicit
    /// `stream.synchronize()` after the projection kernel — the predicate
    /// kernel's own internal sync (inside `launch_predicate_kernel`) now
    /// covers both, and any regression in that chain surfaces here.
    #[test]
    #[ignore = "gpu:projection — Stage 2 stream chaining w/ predicate"]
    fn execute_projection_with_predicate_under_async_stream() {
        let mut engine = Engine::new().expect("engine init");

        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1i32, 2, 3, 4, 5]));
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            ArrowDataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![arr]).expect("batch");
        engine.register_table("t", batch).expect("register");

        let handle = engine
            .sql("SELECT x FROM t WHERE x > 2")
            .expect("query");
        let out = handle.record_batch();

        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32");
        let got: Vec<i32> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, vec![3, 4, 5]);
    }

    // ---------------------------------------------------------------------
    // Stage 7 (P1b): pool-stats periodic-emit throttle.
    //
    // We exercise `should_emit_pool_stats` directly with a mock `Instant`
    // sequence — no CUDA required. The function is the only stateful piece
    // of the periodic-log machinery (the rest is a log line + observer
    // call), so locking down its throttle semantics here gives us the full
    // behavioural coverage we need.
    // ---------------------------------------------------------------------

    #[test]
    fn pool_stats_throttle_first_call_always_emits() {
        // Fresh throttle (no previous emit) must always emit on first
        // call, regardless of how recently the test started.
        let last = Mutex::new(None);
        let now = Instant::now();
        assert!(should_emit_pool_stats(&last, Duration::from_secs(60), now));
        // Second call at the same instant: not enough time elapsed.
        assert!(!should_emit_pool_stats(&last, Duration::from_secs(60), now));
    }

    #[test]
    fn pool_stats_throttle_respects_interval() {
        let last = Mutex::new(None);
        let interval = Duration::from_secs(60);
        let t0 = Instant::now();
        assert!(should_emit_pool_stats(&last, interval, t0), "first emit");
        // 30s later: still inside the window.
        assert!(
            !should_emit_pool_stats(&last, interval, t0 + Duration::from_secs(30)),
            "30s < 60s — must NOT emit"
        );
        // 59s later: still inside.
        assert!(
            !should_emit_pool_stats(&last, interval, t0 + Duration::from_secs(59)),
            "59s < 60s — must NOT emit"
        );
        // 60s later: boundary should fire.
        assert!(
            should_emit_pool_stats(&last, interval, t0 + Duration::from_secs(60)),
            "60s == 60s — must emit"
        );
        // Right after the boundary fire: throttle is reset, so we must
        // wait the full window again.
        assert!(
            !should_emit_pool_stats(&last, interval, t0 + Duration::from_secs(61)),
            "1s after boundary emit — must NOT emit"
        );
        // 60s after the second emit: fires again.
        assert!(
            should_emit_pool_stats(&last, interval, t0 + Duration::from_secs(120)),
            "120s = 60s + 60s — must emit again"
        );
    }

    #[test]
    fn pool_stats_throttle_zero_interval_disables_emission() {
        // The env-var "0" sentinel disables periodic emission entirely.
        let last = Mutex::new(None);
        let now = Instant::now();
        assert!(!should_emit_pool_stats(&last, Duration::ZERO, now));
        // Even after a long delay, zero interval stays disabled.
        assert!(!should_emit_pool_stats(
            &last,
            Duration::ZERO,
            now + Duration::from_secs(3600)
        ));
        // `last` was never updated.
        assert!(last.lock().unwrap().is_none());
    }

    #[test]
    fn pool_stats_throttle_long_interval_still_fires_first_time() {
        // Even a 1-hour interval must produce the first-emit fire so a
        // short-lived process surfaces at least one snapshot.
        let last = Mutex::new(None);
        let now = Instant::now();
        let one_hour = Duration::from_secs(3600);
        assert!(should_emit_pool_stats(&last, one_hour, now));
    }

    #[test]
    fn pool_stats_interval_env_parsing_defaults() {
        // Smoke-test the env-var helper. We can't easily mutate the
        // process env in a parallel test runner safely, so just check the
        // explicit defaults arms. Without the env var set, the default
        // is 60 seconds.
        //
        // NOTE: this test reads (not writes) the env var, so it's safe to
        // run in parallel; the expected default here matches the constant.
        // If a future contributor sets `BOLT_POOL_STATS_INTERVAL_SECS` in
        // their shell while running `cargo test`, this assertion will
        // flag the override — that's intentional.
        if std::env::var(POOL_STATS_ENV).is_err() {
            assert_eq!(
                pool_stats_interval_from_env(),
                Duration::from_secs(DEFAULT_POOL_STATS_INTERVAL_SECS)
            );
        }
    }

    // ---- PV-stage-f: `EngineProvider::has_nulls` surfaces RecordBatch null bitmaps ----

    /// Register a batch whose column contains an Arrow validity bitmap with
    /// at least one NULL row. `EngineProvider::has_nulls` MUST surface this
    /// via `null_count() > 0` on the underlying `RecordBatch::column`.
    /// Without this signal the planner under-flags `KernelSpec` /
    /// `AggregateSpec::input_has_validity`, defeating PV-stage-d / -f
    /// native-validity dispatch.
    #[test]
    #[ignore = "gpu:e2e — Engine::new() initializes driver"]
    fn pv_stage_f_engine_provider_has_nulls_true_for_null_bearing_batch() {
        use crate::plan::TableProvider;

        let mut engine = Engine::new().expect("ctx");
        let arr = Int32Array::from(vec![Some(1i32), None, Some(3)]);
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "v",
            ArrowDataType::Int32,
            true,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(arr)]).expect("batch");
        engine.register_table("t", batch).expect("register");

        let provider = EngineProvider {
            base: &engine.provider,
            tables: &engine.tables,
        };
        assert!(
            provider.has_nulls("t", 0),
            "null-bearing column must surface true via EngineProvider::has_nulls"
        );
        assert_eq!(
            provider.null_count("t", 0),
            Some(1),
            "null_count must reflect Arrow validity bitmap"
        );
    }

    /// Review C10 regression: `register_batch` must union dictionaries across
    /// all registered batches so the string-literal rewriter can resolve
    /// literals that only appear in an appended batch.
    ///
    /// Before this fix, `register_batch` left the dict registry frozen at
    /// batch 0's contents. A subsequent `WHERE s = 'c'` (where `'c'` is only
    /// in batch 1's dictionary) folded to `Bool(false)` against batch 0's
    /// dictionary and silently dropped every otherwise-matching row in
    /// batch 1 — a classic silent-wrong-result bug.
    ///
    /// The fix rebuilds the dict registry against the concatenated batches
    /// after each append, so the rewriter sees the union dict containing
    /// every legal literal. This test exercises the canonical two-batch
    /// scenario:
    ///   * batch 0 has dict values ["a", "b"]
    ///   * batch 1 has dict values ["a", "b", "c"]
    ///   * `WHERE s = 'c'` must return the rows from batch 1 whose `s = "c"`.
    #[test]
    #[ignore = "gpu:string — dictionary construction uploads to GPU"]
    fn c10_register_batch_unions_dictionaries_across_batches() {
        use arrow_array::StringArray;

        let mut engine = Engine::new().expect("ctx");

        // Batch 0: dict values {"a", "b"}; no row holds "c".
        let s0: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "a", "b"]));
        let v0: ArrayRef = Arc::new(Int64Array::from(vec![10_i64, 11, 12, 13]));
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("s", ArrowDataType::Utf8, false),
            ArrowField::new("v", ArrowDataType::Int64, false),
        ]));
        let b0 = RecordBatch::try_new(schema.clone(), vec![s0, v0]).expect("batch 0");

        // Batch 1: dict values {"a", "b", "c"} — "c" appears only here.
        let s1: ArrayRef = Arc::new(StringArray::from(vec!["a", "c", "b", "c"]));
        let v1: ArrayRef = Arc::new(Int64Array::from(vec![20_i64, 21, 22, 23]));
        let b1 = RecordBatch::try_new(schema, vec![s1, v1]).expect("batch 1");

        engine.register_batch("t", b0).expect("batch 0");
        engine.register_batch("t", b1).expect("batch 1");

        // Pre-fix: the rewriter would constant-fold `s = 'c'` to Bool(false)
        // because batch 0's dict never observed "c"; result is zero rows.
        // Post-fix: the dict registry is rebuilt against the concatenated
        // batches so "c" is in the union dict, and the predicate matches
        // the two rows in batch 1 where s = "c" (indices 1, 3 → v = 21, 23).
        let h = engine
            .sql("SELECT v FROM t WHERE s = 'c'")
            .expect("execute");
        let out = h.record_batch();
        assert_eq!(
            out.num_rows(),
            2,
            "literal that lives only in batch 1 must match its two rows; \
             got {} (zero rows is the pre-fix silent-wrong-result bug)",
            out.num_rows()
        );
        let actual = out
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("v is Int64");
        let mut got: Vec<i64> = (0..actual.len()).map(|i| actual.value(i)).collect();
        got.sort();
        assert_eq!(got, vec![21, 23]);
    }

    /// Mirror of the test above for a NULL-free column — provider must
    /// return false so PV stages keep the legacy host-strip path bit-identical.
    #[test]
    #[ignore = "gpu:e2e — Engine::new() initializes driver"]
    fn pv_stage_f_engine_provider_has_nulls_false_for_null_free_batch() {
        use crate::plan::TableProvider;

        let mut engine = Engine::new().expect("ctx");
        let arr = Int32Array::from(vec![1i32, 2, 3]);
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "v",
            ArrowDataType::Int32,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(arr)]).expect("batch");
        engine.register_table("t", batch).expect("register");

        let provider = EngineProvider {
            base: &engine.provider,
            tables: &engine.tables,
        };
        assert!(
            !provider.has_nulls("t", 0),
            "null-free column must surface false"
        );
        assert_eq!(provider.null_count("t", 0), Some(0));
    }

    // ---- Review-H2: PTX module cache in `execute_projection` ----
    //
    // Two layers:
    //
    //   * Host-only key derivation: stable for identical specs, different
    //     for different specs, and entry-name-sensitive. These run on every
    //     `cargo test` invocation (no GPU required).
    //
    //   * GPU-end-to-end: register a table, run the same SQL twice, and
    //     assert `module_cache_loads` only ticked once. A second test
    //     issues a *different* projection on the same engine to confirm
    //     the cache misses on a fresh spec rather than blindly returning
    //     the first module. Both are `#[ignore]` because they need CUDA.

    /// Two identical `KernelSpec`s produce the same cache key.
    #[test]
    fn module_cache_key_stable_for_identical_specs() {
        use crate::plan::ColumnIO;

        let mk_spec = || KernelSpec {
            inputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            ops: Vec::new(),
            predicate: None,
            register_count: 0,
            input_has_validity: Vec::new(),
            output_has_validity: Vec::new(),
        };
        let k1 = ModuleCacheKey::new(&mk_spec(), KERNEL_ENTRY);
        let k2 = ModuleCacheKey::new(&mk_spec(), KERNEL_ENTRY);
        assert_eq!(k1, k2, "identical specs must hash to the same key");
    }

    /// Specs that differ in output column name produce different cache keys
    /// — otherwise two different projections would alias to the same loaded
    /// module and the second query would launch the wrong kernel.
    #[test]
    fn module_cache_key_differs_for_different_specs() {
        use crate::plan::ColumnIO;

        let base = KernelSpec {
            inputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            ops: Vec::new(),
            predicate: None,
            register_count: 0,
            input_has_validity: Vec::new(),
            output_has_validity: Vec::new(),
        };
        let mut other = base.clone();
        other.outputs[0].name = "y".to_string();
        let k1 = ModuleCacheKey::new(&base, KERNEL_ENTRY);
        let k2 = ModuleCacheKey::new(&other, KERNEL_ENTRY);
        assert_ne!(
            k1, k2,
            "different specs must hash to different keys — otherwise two \
             distinct projections would alias to the same cached module"
        );
    }

    /// The same `KernelSpec` keyed under two different entry names yields
    /// two distinct keys (projection vs predicate kernel both reuse the
    /// spec but emit different PTX).
    #[test]
    fn module_cache_key_distinguishes_entry_names() {
        use crate::plan::ColumnIO;

        let spec = KernelSpec {
            inputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            ops: Vec::new(),
            predicate: None,
            register_count: 0,
            input_has_validity: Vec::new(),
            output_has_validity: Vec::new(),
        };
        let k_proj = ModuleCacheKey::new(&spec, KERNEL_ENTRY);
        let k_pred = ModuleCacheKey::new(&spec, PREDICATE_ENTRY);
        assert_ne!(
            k_proj, k_pred,
            "projection vs predicate kernel must not alias under the same spec"
        );
    }

    /// End-to-end cache hit: register a table, run the same SELECT twice,
    /// observe exactly one cache miss against the projection entry. The
    /// second call must hit and produce identical results.
    #[test]
    #[ignore = "gpu:projection — module cache hit"]
    fn module_cache_hits_on_repeat_projection() {
        use std::sync::atomic::Ordering;

        let mut engine = Engine::new().expect("ctx");
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1i32, 2, 3, 4, 5]));
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            ArrowDataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![arr]).expect("batch");
        engine.register_table("t", batch).expect("register");

        // First call: cache miss, loads count goes from 0 to 1.
        let baseline = engine.module_cache_loads.load(Ordering::SeqCst);
        let h1 = engine.sql("SELECT x FROM t").expect("first query");
        let after_first = engine.module_cache_loads.load(Ordering::SeqCst);
        assert_eq!(
            after_first - baseline,
            1,
            "first projection must compile exactly one module"
        );

        // Second identical call: cache hit, loads count unchanged.
        let h2 = engine.sql("SELECT x FROM t").expect("second query");
        let after_second = engine.module_cache_loads.load(Ordering::SeqCst);
        assert_eq!(
            after_second, after_first,
            "second identical projection must reuse the cached module"
        );

        // Sanity: both results are correct.
        assert_eq!(h1.record_batch().num_rows(), 5);
        assert_eq!(h2.record_batch().num_rows(), 5);
    }

    // -- 128-bit cache-key collision resistance ---------------------------
    //
    // These tests target the hardened `ModuleCacheKey::new` (review M:JIT
    // cache hardening). They verify the two properties that matter for
    // wrong-kernel safety:
    //
    //   * Two distinct `KernelSpec`s whose `Debug` output looks superficially
    //     similar (one byte change deep in the IR) still map to DIFFERENT
    //     128-bit cache keys — otherwise the cache would alias them and
    //     `Engine::sql` would launch the wrong PTX. The format-then-hash
    //     pipeline plus two domain-separated `DefaultHasher` instances
    //     gives ~2^-64 birthday collision odds.
    //
    //   * Two clones of the SAME `KernelSpec` produce the SAME key — this is
    //     the cache-hit contract that the projection module cache relies on.
    //     A regression here would silently double every JIT compile.

    /// Two `KernelSpec`s that differ only in a single nested-IR byte (a
    /// register index in a `LoadColumn`) must produce different 128-bit
    /// keys. Validates the wider hash + domain-separation strategy: a single
    /// 64-bit `DefaultHasher` would still distinguish these (they differ in
    /// `Debug` output), so the test's real job is to ensure the upgrade did
    /// not regress that baseline — both halves must vary.
    #[test]
    fn cache_key_distinguishes_specs_with_similar_debug() {
        use crate::plan::{ColumnIO, Op};

        let base = KernelSpec {
            inputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            ops: vec![Op::LoadColumn {
                dst: crate::plan::Reg(0),
                col_idx: 0,
                dtype: DataType::Int32,
            }],
            predicate: None,
            register_count: 1,
            input_has_validity: Vec::new(),
            output_has_validity: Vec::new(),
        };
        // Differ only in the destination register — `Debug` output flips
        // a single digit, which is exactly the "similar Debug" stress case
        // the hardened key is designed to survive.
        let mut other = base.clone();
        other.ops[0] = Op::LoadColumn {
            dst: crate::plan::Reg(1),
            col_idx: 0,
            dtype: DataType::Int32,
        };

        let k1 = ModuleCacheKey::new(&base, KERNEL_ENTRY);
        let k2 = ModuleCacheKey::new(&other, KERNEL_ENTRY);
        assert_ne!(
            k1, k2,
            "specs with near-identical Debug output must still produce \
             distinct cache keys — otherwise the cache would launch the \
             wrong kernel for the second spec"
        );
        // Stronger: BOTH 64-bit halves must differ. If one half collided
        // we'd still be safe (Eq compares the tuple), but a single-half
        // collision would mean the domain-separation byte stopped helping
        // and we'd be back to 64-bit semantics on that half.
        assert_ne!(
            k1.spec_hash_hi, k2.spec_hash_hi,
            "hi half must vary independently — domain separation regression?"
        );
        assert_ne!(
            k1.spec_hash_lo, k2.spec_hash_lo,
            "lo half must vary independently — domain separation regression?"
        );
    }

    /// Two clones of the same `KernelSpec` produce the same key. This is
    /// the cache-hit contract; if it ever broke, every repeat query would
    /// JIT-compile from scratch and the `module_cache_hits_on_repeat_*`
    /// integration tests would also break — but this micro-test localises
    /// the regression to the key derivation rather than the cache plumbing.
    #[test]
    fn cache_key_stable_under_clone() {
        use crate::plan::{ColumnIO, Op};

        let spec = KernelSpec {
            inputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "y".to_string(),
                dtype: DataType::Int64,
            }],
            ops: vec![
                Op::LoadColumn {
                    dst: crate::plan::Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int32,
                },
                Op::Store {
                    src: crate::plan::Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int64,
                },
            ],
            predicate: None,
            register_count: 1,
            input_has_validity: Vec::new(),
            output_has_validity: Vec::new(),
        };
        let cloned = spec.clone();
        let k1 = ModuleCacheKey::new(&spec, KERNEL_ENTRY);
        let k2 = ModuleCacheKey::new(&cloned, KERNEL_ENTRY);
        assert_eq!(
            k1, k2,
            "clone of the same spec must produce the same cache key — \
             otherwise repeat queries would always JIT-compile from scratch"
        );
    }

    /// End-to-end cache miss on a *different* projection: confirm the cache
    /// is keyed correctly (otherwise a second, distinct SELECT would
    /// erroneously hit and run the wrong kernel — silent-wrong-result).
    #[test]
    #[ignore = "gpu:projection — module cache miss"]
    fn module_cache_misses_on_different_projection() {
        use std::sync::atomic::Ordering;

        let mut engine = Engine::new().expect("ctx");
        let xs: ArrayRef = Arc::new(Int32Array::from(vec![1i32, 2, 3]));
        let ys: ArrayRef = Arc::new(Int32Array::from(vec![10i32, 20, 30]));
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("x", ArrowDataType::Int32, false),
            ArrowField::new("y", ArrowDataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![xs, ys]).expect("batch");
        engine.register_table("t", batch).expect("register");

        let baseline = engine.module_cache_loads.load(Ordering::SeqCst);
        let _ = engine.sql("SELECT x FROM t").expect("first query");
        let after_first = engine.module_cache_loads.load(Ordering::SeqCst);
        let _ = engine.sql("SELECT y FROM t").expect("second query");
        let after_second = engine.module_cache_loads.load(Ordering::SeqCst);
        assert_eq!(
            after_first - baseline,
            1,
            "first projection must compile one module"
        );
        assert_eq!(
            after_second - after_first,
            1,
            "second projection on a different column must miss and compile \
             its own module — otherwise the cache is over-keying"
        );
    }
}
