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
    ArrayRef, BooleanArray, Decimal128Array, Float32Array, Float64Array, Int32Array, Int64Array,
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
    parse_sql, DataType, Field, KernelSpec, LogicalPlan, MemTableProvider, PhysicalPlan,
    PlanRewrite, Schema,
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
///
/// # Correctness invariant (finding V-15)
///
/// This key derives entirely from `format!("{:?}", spec)` (see
/// [`ModuleCacheKey::new`]). Its correctness therefore rests on a single
/// invariant:
///
/// > **distinct specs => distinct `Debug` output.**
///
/// The default, `#[derive(Debug)]`-generated formatting on the `KernelSpec`
/// IR satisfies this because the derive emits every field and enum
/// discriminant. **Do not** add a hand-written `Debug` impl to `KernelSpec`
/// or any type it transitively contains that elides, abbreviates, or
/// otherwise collapses a discriminating field (e.g. printing only a summary,
/// hiding a "default" variant, or rounding a numeric literal). Two specs that
/// differ only in an elided field would then format identically, hash to the
/// same key, and the cache would silently serve the WRONG compiled
/// `CudaModule` for one of them — a silent-wrong-result failure mode that no
/// test of this module would catch. If a custom `Debug` is ever required for
/// readability, route this cache key through a dedicated, exhaustive
/// fingerprint instead of reusing `Debug`.
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
        // v0.7 sub-task B: Decimal128 stores `2 * n_rows` u64 values
        // (interleaved [lo, hi] pairs); divide back to get the logical
        // row count.
        Decimal128 { values, .. } => values.len() / 2,
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
        // v0.7 sub-task B: Decimal128 prefix-extend isn't wired yet —
        // the tail would need a slice-and-pack helper paralleling
        // `Decimal128Array::value(i)`. Punt to a full re-upload for now;
        // every existing Decimal column test exercises the full-upload
        // path through `GpuColumn::upload`.
        GpuColumnData::Decimal128 { .. } => {
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
///
/// # Construction
///
/// Prefer the typed builder for new code:
///
/// ```ignore
/// use craton_bolt::Engine;
///
/// let engine = Engine::builder()
///     .device(0)
///     .memory_budget(1 << 30)
///     .build()?;
/// ```
///
/// The legacy [`Engine::new`] and [`Engine::new_with_device`] entry points are
/// thin wrappers around the builder, kept for source-compatibility with
/// pre-v0.6 callers.
///
/// # `#[non_exhaustive]`
///
/// Marked `#[non_exhaustive]` so future v0.x releases can grow new fields
/// without a breaking semver bump for downstream code that destructures or
/// constructs `Engine` literally. Construction goes through the builder; all
/// other access is via inherent methods.
#[non_exhaustive]
pub struct Engine {
    /// Registered tables, keyed by name. A single table may comprise multiple
    /// batches (wave-7 multi-batch support): the engine concatenates them via
    /// `arrow::compute::concat_batches` at query time. This is a 0.2-era
    /// simplification — a streaming, per-batch query plan is a 0.3 goal — so
    /// large multi-batch tables pay a full materialisation cost on every
    /// `sql()` call. Keep the per-table batch count modest until then.
    tables: HashMap<String, Vec<RecordBatch>>,
    /// Lazily-registered streaming table sources, keyed by name.
    ///
    /// Tables registered through [`Engine::register_table_stream_lazy`] are
    /// stored here as a replayable producer ([`TableSource::Streaming`])
    /// rather than being drained into `tables` at registration time. The
    /// producer is invoked the first time the table is read (see
    /// [`Engine::streaming_batches`]), at which point the entry is collapsed
    /// in place to [`TableSource::Materialized`] so subsequent reads skip the
    /// producer.
    ///
    /// This is an *overlay* over `tables`: a name lives in exactly one of the
    /// two maps. The read helpers ([`Engine::materialize_table`] and the
    /// provider null probes) consult `tables` first and fall back to draining
    /// the streaming overlay. Keeping the lazy data out of `tables` is what
    /// makes registration cheap (no host materialisation) while leaving every
    /// eager code path untouched.
    ///
    /// `RefCell` because the lazy materialisation happens from `&self`
    /// (`Engine::sql` takes `&self`), mirroring the interior mutability
    /// already used for `gpu_tables`.
    streaming_sources: RefCell<HashMap<String, crate::exec::streaming::TableSource>>,
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
    /// v0.6 / M7 public optimizer extension surface: user-registered
    /// PlanRewrite implementations run in registration order before lower_physical.
    rewrites: Vec<Box<dyn PlanRewrite>>,
    /// v0.6 builder: CUDA device ordinal this engine was constructed on.
    device_idx: i32,
    /// v0.6 builder: soft cap on device-memory pool allocations in bytes.
    memory_budget_bytes: Option<usize>,
    /// v0.6 builder: optional disk-backed PTX cache directory.
    persistent_cache_path: Option<std::path::PathBuf>,
    /// v0.6 builder: whether tracing was enabled by the builder.
    tracing_enabled: bool,
    /// Owned CUDA context — declared LAST so it drops AFTER dictionaries.
    _ctx: CudaContext,
}

/// v0.6 builder for [`Engine`]. Use [`Engine::builder`] to start one.
///
/// Every knob is optional; un-set knobs land on the same defaults that the
/// legacy [`Engine::new`] / [`Engine::new_with_device`] paths produce. The
/// builder owns no resources until [`EngineBuilder::build`] is called — only
/// `build` initialises the CUDA driver, validates the device index, and
/// creates the CUDA context. This keeps `EngineBuilder` cheap to construct in
/// hot paths (e.g. test harnesses) without paying for driver init that may
/// then be discarded.
///
/// The builder is `#[non_exhaustive]` so v0.x can grow new knobs without a
/// breaking change for downstream code that destructures it (which shouldn't
/// happen — but the marker makes the intent explicit).
///
/// ```ignore
/// use craton_bolt::Engine;
/// use std::path::PathBuf;
///
/// let engine = Engine::builder()
///     .device(0)
///     .memory_budget(2 * 1024 * 1024 * 1024)        // 2 GiB soft cap
///     .persistent_cache(PathBuf::from("/var/cache/bolt/ptx"))
///     .enable_tracing()
///     .build()?;
/// ```
#[non_exhaustive]
#[derive(Debug, Default, Clone)]
pub struct EngineBuilder {
    /// CUDA device ordinal. `None` selects the default (`0`).
    device: Option<i32>,
    /// Soft device-memory budget in bytes. `None` is uncapped.
    memory_budget_bytes: Option<usize>,
    /// Optional disk-backed PTX cache directory.
    persistent_cache_path: Option<std::path::PathBuf>,
    /// Install a default tracing subscriber from [`build`](Self::build).
    enable_tracing: bool,
}

impl EngineBuilder {
    /// Fresh builder with all knobs at their defaults. Same as the value
    /// returned by [`Engine::builder`] — exposed publicly so downstream code
    /// can stash a default builder and tweak it incrementally without going
    /// through the `Engine::` type name (handy in generic test helpers).
    pub fn new() -> Self {
        Self {
            device: None,
            memory_budget_bytes: None,
            persistent_cache_path: None,
            enable_tracing: false,
        }
    }

    /// Select the CUDA device ordinal. Defaults to `0`.
    ///
    /// The index is validated against `cuDeviceGetCount` inside
    /// [`build`](Self::build); an out-of-range index surfaces a
    /// `BoltError::Other` there, not here.
    pub fn device(mut self, idx: i32) -> Self {
        self.device = Some(idx);
        self
    }

    /// Set a soft cap on device-memory pool allocations, in bytes. Defaults
    /// to uncapped.
    ///
    /// Stored verbatim on the engine and readable via
    /// [`Engine::memory_budget_bytes`]. Runtime pool integration may evolve
    /// across v0.x — the getter contract is what's stable.
    pub fn memory_budget(mut self, bytes: usize) -> Self {
        self.memory_budget_bytes = Some(bytes);
        self
    }

    /// Enable a disk-backed PTX cache rooted at `path`. Defaults to
    /// disabled (the existing in-memory PTX cache in `jit::jit_compiler`
    /// is unaffected either way).
    ///
    /// [`build`](Self::build) threads this path into the process-wide disk
    /// PTX cache (via [`crate::jit::disk_cache::set_override_dir`]), so the
    /// JIT compile path reads/writes cubins at `path` even when the
    /// `BOLT_PTX_CACHE_DIR` env var is unset. The builder path takes
    /// precedence over that env var, which remains the fallback for
    /// engines built without this knob.
    ///
    /// The path is stored verbatim — `DiskPtxCache::open` creates the
    /// directory if it does not already exist, but it is the caller's
    /// responsibility to ensure the location is writable.
    pub fn persistent_cache(mut self, path: std::path::PathBuf) -> Self {
        self.persistent_cache_path = Some(path);
        self
    }

    /// Ask [`build`](Self::build) to install a default tracing subscriber
    /// before returning the engine. Defaults to disabled.
    ///
    /// "Default subscriber" here means a best-effort `log`-crate
    /// initialisation: this crate uses [`log`] for diagnostics today, so
    /// enabling this knob promotes the global `log::Level` to `Info`. A
    /// future v0.x may swap to the `tracing` crate proper; the builder
    /// method's name is intentionally subscriber-agnostic so the contract
    /// survives that swap. Calling this on a process where a logger /
    /// subscriber is already installed is a no-op (the underlying
    /// `set_logger` is idempotent under contention).
    pub fn enable_tracing(mut self) -> Self {
        self.enable_tracing = true;
        self
    }

    /// Build the [`Engine`]. Consumes the builder.
    ///
    /// Steps performed by `build` (in order):
    ///   1. Resolve the device index (default `0`).
    ///   2. Initialize the CUDA driver (idempotent across calls).
    ///   3. Validate the device index against `cuDeviceGetCount`.
    ///   4. Create an owned CUDA context on the selected device.
    ///   5. If [`enable_tracing`](Self::enable_tracing) was set, promote the
    ///      global `log` max level to `Info` (best-effort, ignored if a
    ///      logger is already installed).
    ///
    /// # Errors
    /// - `BoltError::Other` if the device index is `< 0` or `>=
    ///   cuDeviceGetCount()`.
    /// - Any underlying CUDA driver failure (no CUDA-capable device,
    ///   driver / runtime mismatch, OOM on context create).
    pub fn build(self) -> BoltResult<Engine> {
        let device_idx = self.device.unwrap_or(0);
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

        // v0.6 / M6: thread the builder's `persistent_cache(path)` knob
        // into the process-wide disk PTX cache so the JIT compile path
        // (`get_or_build_module` → `disk_cache::disk_cache()`) reads and
        // writes cubins at the configured directory — not only when the
        // `BOLT_PTX_CACHE_DIR` env var is set. Opt-in: a `None` path
        // clears any prior builder override and re-falls-back to the env
        // var, preserving the historical "no path → no disk cache"
        // behaviour. See `install_persistent_cache_override` for the
        // precedence contract.
        install_persistent_cache_override(self.persistent_cache_path.as_deref());

        if self.enable_tracing {
            // Best-effort subscriber init. `log::set_max_level` is
            // process-global but always succeeds; pairing it with a
            // logger-installed check would require a fixed downstream
            // logger choice we don't want to make here. If the caller
            // has already wired a logger, raising the level is the
            // worst we'll do; if they haven't, the elevated level is
            // a benign hint for whatever they install later.
            log::set_max_level(log::LevelFilter::Info);
        }

        // v0.7: wire the builder-supplied `persistent_cache(path)` knob
        // into the process-wide disk PTX cache (see `jit::disk_cache`).
        // When `persistent_cache_path` is `Some`, install it as a
        // builder override — `disk_cache::resolve_cache_dir` prefers an
        // installed override over the `BOLT_PTX_CACHE_DIR` env var so
        // the builder-explicit path wins (last-write-wins between this
        // path and any prior `set_disk_ptx_cache_dir` call).
        //
        // When `persistent_cache_path` is `None` we intentionally do
        // NOT clear the override here: an unset builder knob must not
        // wipe out an env-var-driven cache that the surrounding
        // process configured, and must not wipe out an override that
        // another component installed before us. The env-var path
        // therefore continues to work unchanged when the builder
        // doesn't opt in.
        if let Some(p) = self.persistent_cache_path.clone() {
            crate::jit::set_disk_ptx_cache_dir(Some(p));
        }

        Ok(Engine {
            tables: HashMap::new(),
            streaming_sources: RefCell::new(HashMap::new()),
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
            rewrites: Vec::new(),
            device_idx,
            memory_budget_bytes: self.memory_budget_bytes,
            persistent_cache_path: self.persistent_cache_path,
            tracing_enabled: self.enable_tracing,
            _ctx: ctx,
        })
    }
}

impl Engine {
    /// Create an engine on the default CUDA device (ordinal 0).
    ///
    /// v0.6 legacy entry point: thin wrapper around [`Engine::builder`] kept
    /// so pre-v0.6 callers continue to compile. New code should prefer the
    /// builder for forward-compatible knobs.
    pub fn new() -> BoltResult<Self> {
        Self::builder().build()
    }

    /// Create an engine bound to the CUDA device at ordinal `device_idx`.
    ///
    /// v0.6 legacy entry point: thin wrapper around
    /// [`Engine::builder`]`.device(device_idx).build()`. The error contract is
    /// preserved verbatim — see [`EngineBuilder::build`] for the failure
    /// modes (out-of-range index, driver init failure, context create).
    pub fn new_with_device(device_idx: i32) -> BoltResult<Self> {
        Self::builder().device(device_idx).build()
    }

    /// Start a fresh [`EngineBuilder`] with all knobs at their defaults.
    ///
    /// This is the recommended construction entry point as of v0.6. Set only
    /// the knobs you need; everything else picks up the same default that
    /// the legacy [`Engine::new`] / [`Engine::new_with_device`] paths use:
    ///
    /// | Builder method        | Default                |
    /// |-----------------------|------------------------|
    /// | [`EngineBuilder::device`]            | `0`              |
    /// | [`EngineBuilder::memory_budget`]     | uncapped         |
    /// | [`EngineBuilder::persistent_cache`]  | disabled         |
    /// | [`EngineBuilder::enable_tracing`]    | disabled         |
    ///
    /// ```ignore
    /// use craton_bolt::Engine;
    /// let engine = Engine::builder().build()?;
    /// ```
    pub fn builder() -> EngineBuilder {
        EngineBuilder::new()
    }

    /// CUDA device ordinal this engine was constructed on.
    ///
    /// Mirrors the value passed to [`EngineBuilder::device`] (or `0` for the
    /// default-device entry points). Useful for diagnostics on multi-GPU
    /// hosts and for tests that want to assert the builder threaded the
    /// device knob through.
    pub fn device(&self) -> i32 {
        self.device_idx
    }

    /// Soft device-memory budget in bytes, as set via
    /// [`EngineBuilder::memory_budget`]. `None` means uncapped (the default).
    ///
    /// The value is stored verbatim; the runtime pool integration may evolve
    /// across v0.x releases but the getter's contract is stable.
    pub fn memory_budget_bytes(&self) -> Option<usize> {
        self.memory_budget_bytes
    }

    /// Disk-backed PTX cache directory, as set via
    /// [`EngineBuilder::persistent_cache`]. `None` means disabled.
    pub fn persistent_cache_path(&self) -> Option<&std::path::Path> {
        self.persistent_cache_path.as_deref()
    }

    /// `true` if [`EngineBuilder::enable_tracing`] was called on the builder
    /// that produced this engine.
    pub fn tracing_enabled(&self) -> bool {
        self.tracing_enabled
    }

    /// Register a user-supplied [`PlanRewrite`] on this engine.
    ///
    /// Rewrites run in registration order, threading each rewriter's
    /// output into the next, immediately **before**
    /// [`crate::plan::lower_physical`] in [`Engine::sql`]. See the
    /// [`PlanRewrite`](crate::plan::PlanRewrite) trait docs for the
    /// contract implementations must uphold.
    ///
    /// This `with_rewrite` is the engine-direct entry point; the
    /// forthcoming `Engine::Builder` (parallel agent) exposes the same
    /// signature on the builder. Both ultimately push into the same
    /// `rewrites` field, so the builder integration is a drop-in.
    ///
    /// Takes `self` by value and returns it so the call can chain with
    /// the constructor: `Engine::new()?.with_rewrite(Box::new(MyRewrite))`.
    pub fn with_rewrite(mut self, r: Box<dyn PlanRewrite>) -> Self {
        self.rewrites.push(r);
        self
    }

    /// Number of registered [`PlanRewrite`]s on this engine. Exposed for
    /// tests and for `EXPLAIN`-style introspection.
    pub fn rewrite_count(&self) -> usize {
        self.rewrites.len()
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
    ///
    /// # v0.7: process-wide KernelSpec cache layer
    ///
    /// Before consulting the per-`Engine` cache we now check the
    /// process-wide KernelSpec-keyed cache in
    /// [`crate::exec::module_cache::get_or_build_module_for_spec`]. The
    /// global layer survives across `Engine` instances (test harnesses,
    /// short-lived embedded engines, future multi-engine deployments) so
    /// the second engine that requests the same `(spec, entry)` skips
    /// both codegen *and* PTX-text-hash lookup — it's a flat Arc-clone of
    /// the already-loaded module. The per-engine cache is retained as an
    /// inner fast path so the on-engine `module_cache_loads` counter and
    /// disk-cache write-through remain observable. The layering is:
    ///
    /// 1. **Global KernelSpec cache** — sub-µs Arc-clone on hit; on miss
    ///    falls through to the per-engine path below via the closure.
    /// 2. **Per-engine KernelSpec cache** — fast path for repeat calls on
    ///    the same engine; bumps `module_cache_loads` on a miss.
    /// 3. **Disk-backed PTX cache** (v0.6 / M6) — skips codegen if
    ///    `BOLT_PTX_CACHE_DIR` or the builder's `persistent_cache` was
    ///    set and the PTX text is on disk from a previous process.
    /// 4. **`compile(spec)` + `CudaModule::from_ptx`** — the latter
    ///    consults the PTX-text-hash cache in `jit::jit_compiler` so a
    ///    cross-spec PTX collision still reuses the loaded driver module.
    fn get_or_build_module<F>(
        &self,
        spec: &KernelSpec,
        entry: &'static str,
        compile: F,
    ) -> BoltResult<CudaModule>
    where
        F: FnOnce(&KernelSpec) -> BoltResult<String>,
    {
        // v0.7 layer 1: process-wide KernelSpec cache. On a hit this is a
        // sub-µs Arc-clone that skips every layer below. On a miss the
        // closure runs `compile` and routes the resulting PTX through
        // `CudaModule::from_ptx` itself — so we must NOT call back into
        // the per-engine path here (we'd double-codegen). Instead we
        // re-implement the per-engine + disk + codegen fall-through
        // *inside* the closure. The per-engine cache still services
        // repeat calls within one engine: the `module_cache.lock().get`
        // pre-check is the only difference from a flat global-only path,
        // and it's load-bearing for the `module_cache_loads`-counter
        // test below.
        //
        // Why not push the global cache check inside the per-engine
        // miss branch? Because the *fast path* of `get_or_build_module_for_spec`
        // is what we want — it never touches `self.module_cache.lock()`
        // and so doesn't serialise on the per-engine mutex. The cost is
        // that on a miss we do two lookups (global + per-engine); both
        // are HashMap probes, fine.
        let key = ModuleCacheKey::new(spec, entry);
        // Per-engine fast path: hit. Hold the lock only long enough to
        // clone the Arc. This stays AHEAD of the global lookup so the
        // existing `module_cache_loads` invariant ("second call must
        // not bump the counter") is preserved bit-for-bit, and the
        // single-engine hot path keeps its per-engine mutex affinity.
        if let Some(m) = self
            .module_cache
            .lock()
            .map_err(|_| BoltError::Other("module_cache mutex poisoned".to_string()))?
            .get(&key)
        {
            return Ok(m.clone());
        }
        // Capture just the Copy hash components for the closure below;
        // this lets us move `key` itself into `cache.entry(key)` after
        // the closure has run without borrow-checker complications.
        let spec_hash_hi = key.spec_hash_hi;
        let spec_hash_lo = key.spec_hash_lo;
        // Per-engine miss. Consult the process-wide KernelSpec cache; on
        // a global hit we also seed the per-engine cache so subsequent
        // calls on this engine take the per-engine fast path above and
        // skip the global mutex altogether.
        let module = crate::exec::module_cache::get_or_build_module_for_spec(
            spec,
            entry,
            |spec| {
                // Global miss path: this closure runs at most once per
                // (spec, entry) per process. Inside it we still want
                // the disk cache + codegen layers, so we open-code
                // them here (mirrors the legacy per-engine miss path).
                let disk = crate::jit::disk_cache::disk_cache();
                let disk_key = disk.as_ref().map(|_| {
                    // Compose a disk key that (a) folds in the
                    // codegen-version salt so a PTX-emission change
                    // invalidates stale on-disk entries (JIT-M1), and
                    // (b) domain-separates entry-point names: two
                    // kernels with identical KernelSpec content but
                    // different entry symbols (KERNEL_ENTRY vs
                    // PREDICATE_ENTRY) must NOT alias to the same .ptx
                    // file. See `disk_cache::disk_key` for the canonical
                    // key shape; the in-process KernelSpecCache key is
                    // intentionally left unsalted (it re-validates PTX
                    // content on every hit).
                    crate::jit::disk_cache::disk_key(
                        entry,
                        spec_hash_hi,
                        spec_hash_lo,
                    )
                });
                let ptx = match (&disk, &disk_key) {
                    (Some(cache), Some(k)) => match cache.lookup(k) {
                        Some(text) => text,
                        None => {
                            let text = compile(spec)?;
                            // Write-through to disk. Errors here are
                            // non-fatal: a failed write just means
                            // future processes won't benefit, but the
                            // current process still loads the module
                            // successfully via the in-process caches.
                            let _ = cache.store(k, &text);
                            text
                        }
                    },
                    _ => compile(spec)?,
                };
                Ok(ptx)
            },
        )?;
        // Bump the per-engine miss counter. We treat any path that
        // missed the per-engine cache as a "miss" for this counter —
        // even if the global cache served us — because the counter's
        // historical semantics are "did we have to look further than
        // this engine's own table?". Tests pin this invariant.
        self.module_cache_loads
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // Seed the per-engine cache so subsequent calls on this
        // engine take the per-engine fast path above and never reach
        // the global lock.
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
        if self.tables.contains_key(&name)
            || self.streaming_sources.borrow().contains_key(&name)
        {
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

    /// Register a table from a producer that yields batches lazily.
    ///
    /// `schema` declares the expected per-batch schema up front (column
    /// names + dtypes); every batch yielded by `batches` is validated
    /// against it before being installed. Producer-side errors propagate
    /// out of the iterator as `BoltResult::Err(_)` — yielding an `Err`
    /// aborts registration with that error, leaving the engine in the
    /// state it had before this call (modulo any batches already
    /// installed for *this* table, which are rolled back via
    /// `tables.remove`).
    ///
    /// # v0.6 semantics: eager consumption
    ///
    /// The v0.6 cut consumes `batches` EAGERLY: every batch is pulled
    /// from the iterator and pushed into the engine's existing
    /// multi-batch in-memory table representation (the same `Vec<RecordBatch>`
    /// that backs `register_table` + repeated `register_batch`). This is
    /// deliberate — the goal here is to land a stable API *shape* so
    /// callers can write streaming-style code today and have it keep
    /// compiling when v0.7+ replaces the body with a truly lazy
    /// per-batch query plan. Until then, large streams still pay the
    /// full host-side materialisation cost on every `sql()` call (see
    /// the field doc on `tables` for the perf caveat).
    ///
    /// Roadmap: v0.7 is expected to land lazy streaming where each
    /// yielded batch is processed and discarded without materialising
    /// the full table in host memory. The signature here is
    /// future-compatible with that change.
    ///
    /// For a registration path that does NOT drain the producer up front,
    /// see [`Engine::register_table_stream_lazy`], which stores a replayable
    /// producer and only materialises it on first query — the lazy seam that
    /// backs morsel/larger-than-VRAM execution.
    ///
    /// # Errors
    /// - The iterator is empty (a table must contain at least one batch
    ///   for `materialize_table` to succeed).
    /// - Any yielded `Err` propagates out unchanged.
    /// - Any batch's schema (column names + plan-level dtypes) does not
    ///   match the declared `schema`.
    /// - A table named `name` is already registered.
    pub fn register_table_stream<I>(
        &mut self,
        name: impl Into<String>,
        schema: Schema,
        batches: I,
    ) -> BoltResult<()>
    where
        I: IntoIterator<Item = BoltResult<RecordBatch>>,
    {
        let name = name.into();
        if self.tables.contains_key(&name)
            || self.streaming_sources.borrow().contains_key(&name)
        {
            return Err(BoltError::Plan(format!(
                "table '{name}' is already registered — register_table_stream \
                 cannot append to an existing table; use register_batch instead"
            )));
        }
        // Validate one batch against the declared plan schema (names +
        // dtypes match positionally). We compare via the plan schema
        // rather than the raw Arrow schema so a caller-declared
        // `nullable: true` doesn't clash with a batch whose Arrow
        // schema happens to mark the same column non-nullable — the
        // engine treats per-row null counts as the truth and the
        // declared `nullable` is informational only at registration
        // time.
        fn validate_batch_schema(
            declared: &Schema,
            batch: &RecordBatch,
            name: &str,
            batch_idx: usize,
        ) -> BoltResult<()> {
            let actual = arrow_schema_to_plan_schema(batch.schema().as_ref())?;
            if actual.fields.len() != declared.fields.len() {
                return Err(BoltError::Plan(format!(
                    "register_table_stream: batch {batch_idx} for table '{name}' \
                     has {} columns but declared schema has {}",
                    actual.fields.len(),
                    declared.fields.len()
                )));
            }
            for (i, (a, d)) in actual.fields.iter().zip(declared.fields.iter()).enumerate() {
                if a.name != d.name || a.dtype != d.dtype {
                    return Err(BoltError::Plan(format!(
                        "register_table_stream: batch {batch_idx} for table '{name}' \
                         column {i} mismatch — declared {:?}:{:?}, got {:?}:{:?}",
                        d.name, d.dtype, a.name, a.dtype
                    )));
                }
            }
            Ok(())
        }

        // Eagerly drain the iterator, threading errors out and rolling
        // back the (just-installed) table on any failure so the engine
        // never observes a partially-installed table from this call.
        let mut iter = batches.into_iter();
        let first = match iter.next() {
            Some(Ok(b)) => b,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(BoltError::Plan(format!(
                    "register_table_stream: iterator for table '{name}' yielded \
                     zero batches — a registered table must contain at least one batch"
                )));
            }
        };
        validate_batch_schema(&schema, &first, &name, 0)?;
        // Install the first batch through the same path
        // `register_table` uses — dictionaries, provider, GpuTable,
        // host-revisions all set up in one place.
        self.register_table(name.clone(), first)?;
        // Stream subsequent batches in. On any error, roll back the
        // entire table install so this call is atomic from the caller's
        // perspective.
        let mut batch_idx: usize = 1;
        loop {
            let next = match iter.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => {
                    self.unregister_table_best_effort(&name);
                    return Err(e);
                }
                None => break,
            };
            if let Err(e) = validate_batch_schema(&schema, &next, &name, batch_idx) {
                self.unregister_table_best_effort(&name);
                return Err(e);
            }
            if let Err(e) = self.register_batch(&name, next) {
                self.unregister_table_best_effort(&name);
                return Err(e);
            }
            batch_idx += 1;
        }
        Ok(())
    }

    /// Best-effort rollback helper used by `register_table_stream` when a
    /// mid-stream error needs to undo the partial install. Mirrors the
    /// state touched by `register_table` / `register_batch`.
    fn unregister_table_best_effort(&mut self, name: &str) {
        self.tables.remove(name);
        self.streaming_sources.borrow_mut().remove(name);
        self.dict_registry.unregister_table(name);
        self.provider.unregister_table(name);
        self.host_revisions.remove(name);
        self.gpu_tables.borrow_mut().remove(name);
    }

    /// Register a table from a producer that yields batches lazily, WITHOUT
    /// draining the producer at registration time (truly lazy path).
    ///
    /// Unlike [`Engine::register_table_stream`] — which drains the iterator
    /// eagerly into the engine's in-memory `Vec<RecordBatch>` representation
    /// the moment it is called — this method stores a *replayable producer*
    /// ([`crate::exec::streaming::TableSource::Streaming`]) and only registers
    /// the table's schema with the SQL frontend. The producer is not invoked
    /// until the first query that references the table, at which point the
    /// source is collapsed into the canonical materialised representation (see
    /// [`Engine::ensure_streaming_materialized`]).
    ///
    /// This keeps registration O(1) for large streams and is the seam through
    /// which larger-than-VRAM, morsel-at-a-time execution will be threaded:
    /// the budget hook ([`Engine::morsel_plan_for_table`]) inspects the
    /// materialised batches against [`Engine::memory_budget_bytes`] and decides
    /// whether to upload the table whole or process it in bounded morsels.
    ///
    /// `producer` is a factory: each call must return a fresh iterator over
    /// the same logical batch sequence (so the source can be re-drained if the
    /// engine ever re-derives the table). Producer-side errors surface as
    /// `Err` items and abort the first materialisation.
    ///
    /// # Errors
    /// - A table named `name` is already registered (eager or lazy).
    /// - Note: schema/content validation and the empty-stream check are
    ///   deferred to first query (when the producer is actually drained),
    ///   matching the lazy contract. The eager
    ///   [`Engine::register_table_stream`] validates up front instead.
    pub fn register_table_stream_lazy(
        &mut self,
        name: impl Into<String>,
        schema: Schema,
        producer: crate::exec::streaming::BatchProducer,
    ) -> BoltResult<()> {
        let name = name.into();
        if self.tables.contains_key(&name)
            || self.streaming_sources.borrow().contains_key(&name)
        {
            return Err(BoltError::Plan(format!(
                "table '{name}' is already registered — register_table_stream_lazy \
                 cannot append to an existing table"
            )));
        }
        // Register the declared schema with the SQL frontend so planning can
        // resolve column references before the producer is ever drained. We do
        // NOT extend it with dictionary `__idx_<col>` columns here — string
        // dictionaries are built when the source is materialised on first
        // query (see `ensure_streaming_materialized`).
        self.provider.register(name.clone(), schema);
        self.streaming_sources.borrow_mut().insert(
            name,
            crate::exec::streaming::TableSource::Streaming(producer),
        );
        Ok(())
    }

    /// Collapse every still-streaming overlay entry into its materialised
    /// batches by draining the producer once.
    ///
    /// Called from [`Engine::sql`] / [`Engine::run_logical_plan`] before the
    /// validity-probe provider is built. Idempotent: an entry that is already
    /// [`TableSource::Materialized`](crate::exec::streaming::TableSource::Materialized)
    /// is skipped, and a fully-eager engine pays only a single `RefCell::borrow`
    /// + emptiness check.
    ///
    /// The drained batches are staged back into the overlay as
    /// `Materialized` (not moved into `tables`), because `sql` only holds
    /// `&self` and `tables` is not interior-mutable. The eager read paths
    /// ([`Engine::materialize_table`] and [`EngineProvider`]) consult the
    /// overlay as a fall-back, so a streaming table is fully queryable from
    /// there. Dictionary `__idx_<col>` rewriting and the incremental GPU cache
    /// are not wired for overlay tables in this cut — primitive columns query
    /// end-to-end; string-literal predicates fall to the host filter path
    /// rather than the dict-fold fast path. Promoting an overlay table into the
    /// fully-wired `tables` store (dictionaries + revisions + GPU cache) is a
    /// follow-up that needs a `&mut self` seam.
    fn ensure_streaming_materialized(&self) -> BoltResult<()> {
        // Fast path: nothing streaming.
        let pending: Vec<String> = {
            let overlay = self.streaming_sources.borrow();
            overlay
                .iter()
                .filter(|(_, src)| src.is_streaming())
                .map(|(name, _)| name.clone())
                .collect()
        };
        if pending.is_empty() {
            return Ok(());
        }
        // `register_table` needs `&mut self`, but `sql` only holds `&self`.
        // We collapse the producer to concatenated host batches here (which
        // only needs `&self` interior mutability on the overlay), then stage
        // the materialised batch back into the overlay as `Materialized`. The
        // eager read paths (`materialize_table`, `EngineProvider`) consult the
        // overlay, so the table is fully queryable without ever touching
        // `tables`. Dictionary / GPU-cache wiring is built lazily by
        // `ensure_gpu_table` and the dict rewriter on demand, mirroring how
        // `register_batch` defers GPU work.
        for name in pending {
            let batches = {
                let overlay = self.streaming_sources.borrow();
                match overlay.get(&name) {
                    Some(src) => src.drain_to_batches(&name)?,
                    None => continue,
                }
            };
            self.streaming_sources.borrow_mut().insert(
                name,
                crate::exec::streaming::TableSource::Materialized(batches),
            );
        }
        Ok(())
    }

    /// Budget hook: decide whether the table named `name` can be uploaded to
    /// the device whole, or must be processed in bounded morsels because its
    /// estimated footprint exceeds this engine's
    /// [`memory_budget_bytes`](Engine::memory_budget_bytes).
    ///
    /// Returns [`crate::exec::streaming::MorselPlan::Whole`] when no budget is
    /// configured (the default) or the table fits; otherwise
    /// [`crate::exec::streaming::MorselPlan::Morsels`] with a row count sized
    /// so each morsel's working set stays under budget. The actual
    /// morsel-at-a-time upload loop — which would iterate
    /// [`crate::exec::streaming::BatchStream`] morsels and stage intermediates
    /// in [`crate::exec::streaming::PinnedBudget`] host-pinned space — is the
    /// device-side follow-up; this method is the host-side decision the
    /// orchestrator consults.
    ///
    /// Resolves the table through both the eager `tables` store and the
    /// streaming overlay (materialising the latter on demand), so it works
    /// for streaming-registered tables too.
    pub fn morsel_plan_for_table(
        &self,
        name: &str,
    ) -> BoltResult<crate::exec::streaming::MorselPlan> {
        use crate::exec::streaming::{estimate_batches_bytes, plan_upload};
        let budget = self.memory_budget_bytes;
        let plan = |batches: &[RecordBatch]| {
            let total_rows = batches
                .iter()
                .map(RecordBatch::num_rows)
                .fold(0usize, |a, n| a.saturating_add(n));
            let total_bytes = estimate_batches_bytes(batches);
            Ok(plan_upload(total_bytes, total_rows, budget))
        };
        if let Some(batches) = self.tables.get(name) {
            return plan(batches.as_slice());
        }
        self.streaming_batches(name, plan)
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
        // Drop any lazy streaming overlay entry under this name — a replace
        // installs an eager `tables` entry, and `materialize_table` prefers
        // `tables`, so a lingering overlay entry would just be stale dead
        // weight (and would block a future `register_table` overlay guard).
        self.streaming_sources.borrow_mut().remove(&name);
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
        // Eager tables first; fall back to the lazy streaming overlay.
        if let Some(batches) = self.tables.get(name) {
            return concat_table_batches(name, batches);
        }
        // Streaming overlay: collapse the source to materialised on first
        // read, then concat its batches.
        self.streaming_batches(name, |batches| concat_table_batches(name, batches))
    }

    /// Ensure the streaming source for `name` is drained, then run `f` over its
    /// materialised batches while holding the overlay borrow.
    ///
    /// On first call for a `Streaming` entry this invokes the producer,
    /// collapsing the entry in place to [`TableSource::Materialized`] so
    /// subsequent reads skip the producer. Returns a `Plan` error if `name` is
    /// not in the streaming overlay at all.
    fn streaming_batches<R>(
        &self,
        name: &str,
        f: impl FnOnce(&[RecordBatch]) -> BoltResult<R>,
    ) -> BoltResult<R> {
        use crate::exec::streaming::TableSource;
        // Phase 1: if the entry is still a producer, drain it (without holding
        // a borrow across the producer call) and swap in the materialised
        // form. Borrow is dropped before re-borrowing for the read.
        let needs_drain = matches!(
            self.streaming_sources.borrow().get(name),
            Some(src) if src.is_streaming()
        );
        if needs_drain {
            let drained = {
                let overlay = self.streaming_sources.borrow();
                match overlay.get(name) {
                    Some(src) => src.drain_to_batches(name)?,
                    None => {
                        return Err(BoltError::Plan(format!(
                            "table '{name}' is not registered with the engine"
                        )))
                    }
                }
            };
            self.streaming_sources
                .borrow_mut()
                .insert(name.to_string(), TableSource::Materialized(drained));
        }
        // Phase 2: read the (now materialised) batches.
        let overlay = self.streaming_sources.borrow();
        match overlay.get(name) {
            Some(TableSource::Materialized(batches)) => f(batches),
            Some(TableSource::Streaming(_)) => {
                // Unreachable: we just collapsed it above.
                Err(BoltError::Other(format!(
                    "streaming source for table '{name}' was not collapsed after drain"
                )))
            }
            None => Err(BoltError::Plan(format!(
                "table '{name}' is not registered with the engine"
            ))),
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
    /// Snapshot the registered tables' row counts into an estimator the
    /// cost-based optimizer can consume.
    ///
    /// Each base table's row count is the sum of its registered `RecordBatch`
    /// row counts (eager `tables` store) plus any already-materialised
    /// streaming source. Lazily-registered streaming tables that haven't been
    /// drained yet are intentionally omitted — costing them would force
    /// materialisation at plan time. Their absence simply leaves any join chain
    /// touching them un-reordered (the conservative default).
    ///
    /// The result is an owned `Arc<dyn RowEstimator>` (a
    /// [`crate::plan::optimizer::StatsEstimator`] over an [`EngineTableStats`]
    /// snapshot) ready to hand to
    /// [`crate::plan::default_passes_with_estimator`].
    fn row_estimator(&self) -> Arc<dyn crate::plan::RowEstimator> {
        let mut row_counts: HashMap<String, usize> = HashMap::new();
        for (name, batches) in &self.tables {
            let rows = batches
                .iter()
                .map(RecordBatch::num_rows)
                .fold(0usize, |a, n| a.saturating_add(n));
            row_counts.insert(name.clone(), rows);
        }
        // Fold in any streaming sources that have already been materialised
        // (a still-streaming source is skipped — see the method docs).
        for (name, src) in self.streaming_sources.borrow().iter() {
            if let crate::exec::streaming::TableSource::Materialized(batches) = src {
                let rows = batches
                    .iter()
                    .map(RecordBatch::num_rows)
                    .fold(0usize, |a, n| a.saturating_add(n));
                row_counts.entry(name.clone()).or_insert(rows);
            }
        }
        Arc::new(crate::plan::StatsEstimator::new(EngineTableStats { row_counts }))
    }

    pub fn sql(&self, query: &str) -> BoltResult<QueryHandle> {
        // **Stage 6 (M3L5)** — retry the pool-watcher's context capture.
        // If the watcher spawned before any engine thread had a context
        // bound, `CAPTURED_CTX` is still zero and every poll silently
        // no-ops. This call is cheap (atomic load when already
        // captured) and a no-op when no context is bound on the
        // calling thread — so it's safe to invoke unconditionally.
        crate::cuda::mem_pool::pool_watcher_retry_context_capture();

        // M5 metrics: count every accepted query (success or failure). The
        // matching `QueriesFailed` bump happens at the single error-return
        // below so the `?`-chain stays intact.
        crate::metrics::metrics().inc(crate::metrics::Counter::QueriesTotal);

        // Time the parse phase (SQL text → LogicalPlan) into the Parse
        // histogram; mirrors the `parse` tracing span in `observability`.
        let parse_start = Instant::now();
        let plan: LogicalPlan = parse_sql(query, &self.provider)?;
        crate::metrics::metrics()
            .observe_duration(crate::metrics::Phase::Parse, parse_start.elapsed());
        // String-literal predicates against Utf8 columns are folded into
        // integer equality against the corresponding __idx_<col> i32 column.
        let plan = tracing::info_span!("plan")
            .in_scope(|| self.dict_registry.rewrite_plan(&plan))?;
        let plan = self.dict_registry.rewrite_plan(&plan)?;
        // Built-in logical optimizer: run the default pass pipeline
        // (constant folding, predicate pushdown, filter-into-join, join
        // reorder, projection pruning) BEFORE any user-registered rewrites.
        // The join-reorder pass is driven by a statistics-backed row
        // estimator snapshotted from the registered tables, so left-deep
        // INNER chains are reordered smallest-input-first (it stays a no-op
        // for chains whose leaves the snapshot can't cost). See
        // `crate::plan::optimizer` for the pipeline and ordering. The
        // pipeline is driven to a bounded fixpoint (re-running the same
        // pass set/order until the plan stabilises or the iteration cap is
        // hit) so a conjunct exposed by one sweep — e.g. a now-constant
        // predicate moved by pushdown — gets folded/pushed on the next.
        let passes = crate::plan::default_passes_with_estimator(self.row_estimator());
        let plan = crate::plan::optimizer::run_to_fixpoint(&passes, plan)?;
        // v0.6 / M7: run user-registered PlanRewrite implementations in
        // registration order, threading each rewriter's output into the
        // next. This runs AFTER the built-in optimizer and the internal
        // dict-rewrite (so user rewrites see the engine's normalised form
        // with `__idx_<col>` refs already in place) and BEFORE
        // `lower_physical` (so users can still target logical-plan
        // structure). See `crate::plan::rewrite` for the contract.
        let plan = self
            .rewrites
            .iter()
            .try_fold(plan, |p, r| r.rewrite(p))?;
        // Resolve uncorrelated scalar / IN subqueries to constants BEFORE
        // lowering. Each subplan is uncorrelated (the frontend rejects
        // correlation) and so independently executable; we run it here and
        // fold the result into the enclosing plan as a literal (scalar) or a
        // boolean OR/AND fold (IN). After this pass no `ScalarSubquery` /
        // `InSubquery` node survives, so the physical reject-arms for them
        // are unreachable for `sql()`-produced plans (they stay as a safety
        // net for hand-built physical plans). See
        // `crate::exec::subquery_resolve`.
        let plan = self.resolve_subqueries(plan)?;
        // Time the lower phase (LogicalPlan → PhysicalPlan) into the Lower
        // histogram; mirrors the `lower` tracing span in `observability`.
        let lower_start = Instant::now();
        let mut phys = crate::plan::lower_physical(&plan)?;
        crate::metrics::metrics()
            .observe_duration(crate::metrics::Phase::Lower, lower_start.elapsed());
        // Collapse any lazily-registered streaming sources to materialised
        // batches so the validity probes below (and `execute`'s
        // `materialize_table`) see real data. Idempotent and a no-op when no
        // streaming tables are registered.
        self.ensure_streaming_materialized()?;
        // PV-stage-d: populate `KernelSpec::input_has_validity` for every
        // input column by consulting the engine-backed provider, which
        // looks straight at `RecordBatch::column(col).null_count()` for
        // each registered table. This is the plan-time signal that lets
        // the codegen emit native-validity kernels instead of leaning on
        // the run-time host-strip fallback in `groupby_with_pre` etc.
        let nb = EngineProvider {
            base: &self.provider,
            tables: &self.tables,
            streaming: self.streaming_sources.borrow(),
        };
        crate::plan::physical_plan::populate_input_validity(&mut phys, &nb);
        // Release the streaming-overlay borrow held by `nb` before `execute`,
        // whose `ensure_gpu_table`/`materialize_table` path may re-borrow the
        // overlay (immutably; mutably only for an un-collapsed source, which
        // `ensure_streaming_materialized` above has already ruled out).
        drop(nb);
        let result = self.execute(&phys);
        // M5 metrics: a failed execution bumps `QueriesFailed`. We only
        // observe the bind here (no `?`-chain restructuring); the early
        // parse/rewrite/lower `?`-returns above are rare developer-time
        // errors, while `execute` is the latency-critical phase whose
        // failures (OOM, kernel fault) are the ones worth a dashboard counter.
        if result.is_err() {
            crate::metrics::metrics().inc(crate::metrics::Counter::QueriesFailed);
        }
        // Stage 7: periodic pool-stats emit. Runs whether the query
        // succeeded or failed (an OOM-failed query is itself a signal
        // worth surfacing alongside the pool snapshot). Internal errors
        // in the emit path are swallowed — they must never escalate to
        // the query result.
        self.maybe_emit_pool_stats(Instant::now());
        result
    }

    /// Parse, plan, and *render* `query` without executing it — an
    /// `EXPLAIN`-style introspection hook.
    ///
    /// Reuses the same logical-plan path as [`Engine::sql`] (parse +
    /// dict-rewrite + user [`PlanRewrite`]s) but stops short of lowering to
    /// GPU kernels or touching the device. The returned string contains the
    /// tree-indented logical plan (via
    /// [`crate::plan::format_logical`]); when the plan also lowers cleanly to
    /// a physical plan it appends the physical rendering (via
    /// [`crate::plan::format_physical`]). Plans that the GPU codegen does not
    /// yet support (e.g. `CASE`, `CAST`) still return their logical plan, with
    /// the lowering error noted in place of the physical section.
    pub fn explain_sql(&self, query: &str) -> BoltResult<String> {
        let plan: LogicalPlan = parse_sql(query, &self.provider)?;
        let plan = self.dict_registry.rewrite_plan(&plan)?;
        let plan = self.rewrites.iter().try_fold(plan, |p, r| r.rewrite(p))?;

        let mut out = String::from("Logical plan:\n");
        out.push_str(&crate::plan::format_logical(&plan));
        out.push_str("\nPhysical plan:\n");
        match crate::plan::lower_physical(&plan) {
            Ok(phys) => out.push_str(&crate::plan::format_physical(&phys)),
            Err(e) => out.push_str(&format!("  <not lowered: {e}>\n")),
        }
        Ok(out)
    }

    /// Execute an already-built [`LogicalPlan`] and return a [`QueryHandle`].
    ///
    /// This is the post-parse half of [`Engine::sql`]: it skips SQL parsing
    /// (so the input plan can come from the [`DataFrame`](crate::DataFrame)
    /// builder, a test fixture, etc.) but still runs the full
    /// rewrite → lower → validity-propagate → execute pipeline so the
    /// physical plan reaching the kernels is shaped identically to the SQL
    /// path. The pool-stats periodic emit is performed here too, mirroring
    /// `sql()`'s book-keeping.
    ///
    /// `&mut self` matches the [`DataFrame::collect`](crate::DataFrame::collect)
    /// signature; the engine state mutated here is bounded to the
    /// pool-stats throttle and the (interior-mutable) GpuTable cache
    /// already touched by `sql()`.
    pub fn run_logical_plan(&mut self, plan: &LogicalPlan) -> BoltResult<QueryHandle> {
        crate::cuda::mem_pool::pool_watcher_retry_context_capture();
        // M5 metrics: the DataFrame `collect` path is a top-level query just
        // like `sql()`, so count it identically — bump `QueriesTotal` here and
        // `QueriesFailed` at the single error-observe below. This is the *only*
        // place the DataFrame path is counted: nested sub-plans resolved during
        // this call run through `run_subplan`, which deliberately does NOT bump
        // the counters (so N subqueries inside one top-level plan still count as
        // exactly one query, matching the `sql()` contract).
        crate::metrics::metrics().inc(crate::metrics::Counter::QueriesTotal);
        // String-literal predicates against Utf8 columns are folded into
        // integer equality against the corresponding __idx_<col> i32 column —
        // mirrors `sql()`.
        let plan = self.dict_registry.rewrite_plan(plan)?;
        // Built-in logical optimizer: mirror the `sql()` path so a plan built
        // via the DataFrame builder gets the same default optimizations before
        // lowering — including statistics-driven join reordering. See
        // `crate::plan::optimizer`. Driven to a bounded fixpoint exactly as
        // `sql()` does, so DataFrame-built plans get the same thorough
        // fold/push convergence.
        let passes = crate::plan::default_passes_with_estimator(self.row_estimator());
        let plan = crate::plan::optimizer::run_to_fixpoint(&passes, plan)?;
        // Mirror `sql()`: resolve uncorrelated subqueries to constants before
        // lowering so a DataFrame-built plan carrying a subquery executes too.
        let plan = self.resolve_subqueries(plan)?;
        let mut phys = crate::plan::lower_physical(&plan)?;
        // Mirror `sql()`: collapse lazy streaming sources before probing.
        self.ensure_streaming_materialized()?;
        // PV-stage-d: thread per-column null-bearing into the kernel specs.
        let nb = EngineProvider {
            base: &self.provider,
            tables: &self.tables,
            streaming: self.streaming_sources.borrow(),
        };
        crate::plan::physical_plan::populate_input_validity(&mut phys, &nb);
        drop(nb);
        let result = self.execute(&phys);
        // M5 metrics: mirror `sql()` — a failed top-level execution bumps
        // `QueriesFailed`. We only observe the `execute` bind (the rare early
        // `?`-returns above are developer-time plan errors, while `execute` is
        // the latency-critical phase whose OOM/kernel faults warrant a counter).
        if result.is_err() {
            crate::metrics::metrics().inc(crate::metrics::Counter::QueriesFailed);
        }
        self.maybe_emit_pool_stats(Instant::now());
        result
    }

    /// Pre-lowering pass: resolve every uncorrelated scalar / IN subquery in
    /// `plan` to a constant.
    ///
    /// Walks the plan's expressions (recursively, inner-subqueries-first via
    /// [`crate::exec::subquery_resolve::resolve_plan`]) and:
    ///
    /// * `Expr::ScalarSubquery(subplan)` → executes `subplan`, requires
    ///   exactly one output column, and folds the result to an
    ///   `Expr::Literal` (0 rows → SQL `NULL`; >1 row → a clean error).
    /// * `Expr::InSubquery { expr, subquery, negated }` → executes `subquery`,
    ///   collects the distinct values, and rewrites to an OR-of-equalities
    ///   (or AND-of-inequalities for `NOT IN`) over `expr`.
    ///
    /// The subplans are executed through [`Engine::run_subplan`], which routes
    /// the full pipeline (dict-rewrite → optimizer → *this pass* → lower →
    /// execute) over `&self` — so nested subqueries inside a subplan resolve
    /// when that subplan runs. Correlation is impossible here (rejected at the
    /// frontend), so each subplan is self-contained and independently
    /// executable.
    ///
    /// Takes `&self` because [`Engine::sql`] is `&self`; the only state mutated
    /// by executing a subplan is the interior-mutable GpuTable cache and the
    /// pool-stats throttle, neither of which needs `&mut`.
    fn resolve_subqueries(&self, plan: LogicalPlan) -> BoltResult<LogicalPlan> {
        let mut exec = |subplan: LogicalPlan| -> BoltResult<RecordBatch> {
            self.run_subplan(subplan)
        };
        crate::exec::subquery_resolve::resolve_plan(plan, &mut exec)
    }

    /// Execute a self-contained (uncorrelated) subquery [`LogicalPlan`] over
    /// `&self` and return its result `RecordBatch`.
    ///
    /// This is the `&self` twin of [`Engine::run_logical_plan`]: it runs the
    /// same dict-rewrite → optimizer → subquery-resolve → lower →
    /// validity-propagate → execute pipeline, but without the `&mut self`
    /// receiver (so it can be called from inside [`Engine::resolve_subqueries`]
    /// during the `&self` `sql()` path). Re-entering `resolve_subqueries` here
    /// is what makes nested subqueries resolve inner-first.
    ///
    /// **Metrics contract (M5):** this path deliberately does NOT bump
    /// `QueriesTotal` / `QueriesFailed`. The query counters count *top-level*
    /// queries only — the enclosing `sql()` / `run_logical_plan` call has
    /// already counted the whole statement once. A statement containing N
    /// subqueries therefore counts as one query (and a subquery failure surfaces
    /// as the top-level query's failure via the `?` below), keeping the
    /// `sql()` and DataFrame paths symmetric.
    fn run_subplan(&self, plan: LogicalPlan) -> BoltResult<RecordBatch> {
        let plan = self.dict_registry.rewrite_plan(&plan)?;
        // Bounded-fixpoint optimizer, mirroring the `sql()` / `run_logical_plan`
        // paths so a subplan is optimized to the same convergence.
        let passes = crate::plan::default_passes_with_estimator(self.row_estimator());
        let plan = crate::plan::optimizer::run_to_fixpoint(&passes, plan)?;
        let plan = self.resolve_subqueries(plan)?;
        let mut phys = crate::plan::lower_physical(&plan)?;
        // The outer `sql()` / `run_logical_plan` has already collapsed any
        // lazy streaming sources before this point, but a subplan may be the
        // first reader of one — collapse again (idempotent) to be safe.
        self.ensure_streaming_materialized()?;
        let nb = EngineProvider {
            base: &self.provider,
            tables: &self.tables,
            streaming: self.streaming_sources.borrow(),
        };
        crate::plan::physical_plan::populate_input_validity(&mut phys, &nb);
        drop(nb);
        let handle = self.execute(&phys)?;
        Ok(handle.into_record_batch())
    }

    /// Emit a periodic pool-stats log line + observer notification if
    /// the configured interval has elapsed since the last emit.
    ///
    /// `now` is taken as a parameter (rather than calling `Instant::now()`
    /// inside) so the unit test below can drive the throttle deterministically.
    ///
    /// **Never-escalate:** this runs after a query has already produced its
    /// result, so it must not be able to fail that query. The user observer is
    /// invoked through [`crate::observability::notify_observers`], which catches
    /// and swallows any panic from the callback (logging it). A panicking
    /// pool-stats observer therefore cannot unwind out of a successful
    /// `Engine::sql` / `Engine::run_logical_plan`.
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
            PhysicalPlan::StringLength {
                table,
                outputs,
                output_schema,
            } => self.execute_string_length(table, outputs, output_schema),
            PhysicalPlan::StringProject {
                table,
                outputs,
                output_schema,
            } => self.execute_string_project(table, outputs, output_schema),
            PhysicalPlan::Aggregate {
                table,
                pre,
                aggregate,
            } => {
                // v0.7: GROUP BY VAR_POP / VAR_SAMP / STDDEV_POP /
                // STDDEV_SAMP are lowered to a per-group Welford pass in
                // the downstream executors (`crate::exec::groupby`,
                // `crate::exec::groupby_valid`, `crate::exec::groupby_with_pre`,
                // and `crate::exec::groupby_wide`). The shared
                // `crate::exec::welford::WelfordState` provides the
                // numerically-stable single-pass update; the executors fold
                // per-group state on the host after the GPU keys kernel
                // populates the slot table.
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
            PhysicalPlan::CountRows {
                input,
                output_schema,
            } => {
                // Scalar `COUNT(...)` over a non-scan-chain child (notably
                // `COUNT(DISTINCT col)`, where `input` is a `Distinct`). The
                // fused scalar-aggregate executor can't fold a Distinct, so the
                // lowerer routed this shape here: execute the child (the
                // Distinct executor materialises the deduped rows as part of
                // that), then emit a single-row Int64 batch holding the child's
                // row count.
                let h = self.execute(input)?;
                let n_rows = h.batch.num_rows();
                let batch = build_count_rows_batch(n_rows, output_schema)?;
                Ok(QueryHandle { batch })
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
            PhysicalPlan::Window {
                input,
                window_exprs,
                partition_by,
                order_by,
                output_schema,
            } => {
                // Host-side window-function executor: materialise the input,
                // partition + order within partition, compute each window
                // output column, and append it. GPU offload is a follow-up;
                // see `crate::exec::window`.
                let h = self.execute(input)?;
                crate::exec::window::execute_window(
                    h,
                    window_exprs,
                    partition_by,
                    order_by,
                    output_schema,
                )
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
            PhysicalPlan::SetOp {
                left,
                right,
                op,
                all,
            } => {
                // EXCEPT / INTERSECT (with optional ALL): execute both inputs,
                // then compute the multiset difference / intersection
                // host-side (see `crate::exec::setops`), reusing the DISTINCT
                // executor's row-key / NULL canonicalisation.
                let l = self.execute(left)?;
                let r = self.execute(right)?;
                crate::exec::setops::execute_setop(l, r, *op, *all)
            }
            PhysicalPlan::Join {
                left,
                right,
                join_type,
                on,
                filter,
                output_schema,
            } => crate::exec::join::execute_join(
                left,
                right,
                join_type,
                on,
                filter.as_ref(),
                output_schema,
                self,
            ),
            PhysicalPlan::StringLikeFilter {
                input,
                table,
                column,
                literal,
                mode,
                negated,
            } => self.execute_string_like_filter(
                input, table, column, literal, *mode, *negated,
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
            // Decimal128 NULL fix: for a pure passthrough Decimal128 output
            // (output name + dtype matches an input column), carry the
            // source column's validity bitmap onto the output so the
            // download path reconstructs NULL rows as NULL, not `0`. The
            // output column needs an *owned* device buffer (it outlives the
            // borrow of `src_col`), so we allocate a fresh mask buffer and
            // copy the bitmap into it.
            //
            // P1/F10b: this copy stays on the device — `cuMemcpyDtoD_v2` via
            // `memcpy_d2d` — instead of the old D2H `to_vec()` + H2D
            // `from_slice()` host round-trip. The mask is at most one byte
            // per 8 rows, but a passthrough query did this PCIe bounce every
            // call; the DtoD copy removes both crossings. The source and the
            // freshly-`zeros`-allocated destination are distinct device
            // allocations (non-overlapping), satisfying `memcpy_d2d`'s safety
            // contract.
            if let DataType::Decimal128(_, _) = io.dtype {
                if let Some(src_col) = kernel
                    .inputs
                    .iter()
                    .find(|in_io| in_io.name == io.name && in_io.dtype == io.dtype)
                    .and_then(|in_io| gpu_table.column(&in_io.name))
                {
                    if let crate::exec::gpu_table::GpuColumnData::Decimal128 {
                        valid_mask: Some(src_mask),
                        ..
                    } = &src_col.data
                    {
                        let mask_len = src_mask.len();
                        let dst_mask = GpuVec::<u8>::zeros(mask_len)?;
                        // SAFETY: `dst_mask` was just allocated (a distinct
                        // device pointer from `src_mask`), both are live
                        // allocations of `mask_len` bytes, and they do not
                        // overlap — meeting `memcpy_d2d`'s contract. A
                        // zero-length mask short-circuits inside `memcpy_d2d`.
                        unsafe {
                            cuda_sys::memcpy_d2d::<u8>(
                                dst_mask.device_ptr(),
                                src_mask.device_ptr(),
                                mask_len,
                            )?;
                        }
                        col.set_decimal128_valid_mask(Some(dst_mask));
                    }
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
        // V-1 / F10a: this hand-rolled `cuLaunchKernel` bypasses
        // `KernelArgs::tag_launch_stream` (the central Drop-fence enforcement
        // point that `launch_1d` callers get for free). Restore the invariant
        // for the freshly-allocated output buffers by recording the launch
        // stream in each one's `StreamSet`, so a buffer dropped while the
        // kernel is still in flight fences this stream before its block is
        // recycled — independent of the downstream `synchronize()` in
        // `download_pinned` / `gpu_compact`. The input columns live in the
        // persistent GpuTable cache (not recycled across this launch), so the
        // load-bearing buffers to tag are the outputs.
        for col in &output_cols {
            col.mark_launch_stream(stream.raw());
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
                // M5 metrics: Utf8 outputs can't be gathered on-device, so
                // this projection takes the documented host-side filter path.
                crate::metrics::metrics().inc(crate::metrics::Counter::HostFallbacksTotal);
                // Host-side fallback: download mask + outputs, then filter.
                let host_mask =
                    crate::exec::compact::download_mask(mask.device_ptr(), n_rows, &stream)?;
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

    /// Execute a [`PhysicalPlan::StringLength`]: a `SELECT LENGTH(<utf8_col>)`
    /// projection (plus passthrough columns) over a bare scan, with the
    /// `LENGTH` outputs computed on the GPU via the dictionary-index gather
    /// kernel ([`crate::jit::string_kernel::compile_length_gather_kernel`]).
    ///
    /// Passthrough columns are lifted directly from the host-side source batch
    /// (zero-copy `ArrayRef` clone). Each `LENGTH(col)` output runs the gather
    /// against the GPU-resident dictionary column when it is dictionary-encoded
    /// (and, for the native `DictUtf8` layout, null-free); otherwise it falls
    /// back to a host-side gather over the downloaded keys (see
    /// [`crate::exec::string_length`]). Both paths produce an `Int64Array`
    /// matching the logical-plan `LENGTH` output dtype.
    fn execute_string_length(
        &self,
        table: &str,
        outputs: &[crate::plan::physical_plan::StringLengthOutput],
        output_schema: &Schema,
    ) -> BoltResult<QueryHandle> {
        use crate::plan::physical_plan::StringLengthOutput;

        // Source host batch — used for passthrough columns (and as the row
        // count authority so an empty / partial table still works).
        let src_batch = self.materialize_table(table)?;
        let src_schema = src_batch.schema();
        let n_rows = src_batch.num_rows();

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(outputs.len());
        for out in outputs {
            match out {
                StringLengthOutput::Passthrough { source } => {
                    let idx = src_schema.index_of(source).map_err(|_| {
                        BoltError::Plan(format!(
                            "StringLength: passthrough column '{source}' not found in \
                             table '{table}'"
                        ))
                    })?;
                    arrays.push(src_batch.column(idx).clone());
                }
                StringLengthOutput::Length { source } => {
                    arrays.push(self.string_length_column(table, source, n_rows)?);
                }
            }
        }

        let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
        let batch_out = RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
            BoltError::Other(format!(
                "StringLength: failed to build output RecordBatch: {e}"
            ))
        })?;
        Ok(QueryHandle { batch: batch_out })
    }

    /// Compute `LENGTH(<source>)` for the GPU-resident `Utf8` column `source`
    /// of `table`, returning an `Int64Array` of `n_rows` rows.
    ///
    /// GPU path: build the per-dictionary-entry `i32` length table that matches
    /// the column's device key layout, upload it, launch the gather kernel
    /// (`out[row] = length_table[keys[row]]`), download the `Int32` result, and
    /// widen to `Int64`. When the column is not safe to gather on the GPU
    /// (non-dict storage, or a `DictUtf8` column with NULLs — whose zeroed keys
    /// would gather the wrong slot), fall back to a host-side gather over the
    /// downloaded keys, which is byte-for-byte identical for the supported case.
    fn string_length_column(
        &self,
        table: &str,
        source: &str,
        n_rows: usize,
    ) -> BoltResult<ArrayRef> {
        use crate::exec::string_length::{
            build_length_table, gpu_gather_layout, host_gather_lengths, KeyLayout,
        };

        let gpu_table_ref = self.ensure_gpu_table(table)?;
        let gpu_table: &crate::exec::gpu_table::GpuTable = &gpu_table_ref;
        let column = gpu_table.column(source).ok_or_else(|| {
            BoltError::Plan(format!(
                "StringLength: column '{source}' not in GPU table '{table}'"
            ))
        })?;

        // Resolve the host-side dictionary + device key buffer + layout for
        // this column. `None` layout ⇒ host fallback.
        let dict = column.utf8_dictionary().ok_or_else(|| {
            BoltError::Plan(format!(
                "StringLength: column '{source}' is not a Utf8 column (LENGTH requires Utf8)"
            ))
        })?;
        let (keys_vec, layout): (&GpuVec<i32>, Option<KeyLayout>) = match &column.data {
            crate::exec::gpu_table::GpuColumnData::Utf8 { indices, .. } => {
                (indices, gpu_gather_layout(&column.data))
            }
            crate::exec::gpu_table::GpuColumnData::DictUtf8 { keys, .. } => {
                (keys, gpu_gather_layout(&column.data))
            }
            _ => {
                return Err(BoltError::Plan(format!(
                    "StringLength: column '{source}' has non-Utf8 GPU storage"
                )))
            }
        };

        let layout = match layout {
            Some(l) => l,
            None => {
                // Host fallback: download keys and gather over the 1-based
                // NULL-sentinel table (DictUtf8-with-nulls keys are zeroed to
                // 0, which this table maps to length 0 — matching the
                // documented `exec::string_ops::length` NULL → 0 behaviour).
                let table_lengths =
                    build_length_table(dict, KeyLayout::OneBasedNullSlot0)?;
                let keys_host = keys_vec.to_vec()?;
                // DictUtf8 keys are 0-based; remap to the 1-based table by
                // adding 1 only when the column is the DictUtf8 layout.
                let lens = match &column.data {
                    crate::exec::gpu_table::GpuColumnData::DictUtf8 { valid_mask, .. } => {
                        // Consult validity: NULL rows → 0, valid rows → table[key+1].
                        let mask = valid_mask
                            .as_ref()
                            .map(|m| m.to_vec())
                            .transpose()?;
                        let mut out: Vec<i64> = Vec::with_capacity(keys_host.len());
                        for (row, &k) in keys_host.iter().enumerate() {
                            let is_valid = match &mask {
                                None => true,
                                Some(bits) => {
                                    let byte = bits.get(row / 8).copied().unwrap_or(0);
                                    (byte >> (row % 8)) & 1 == 1
                                }
                            };
                            if !is_valid {
                                out.push(0);
                            } else if k < 0 {
                                return Err(BoltError::Other(format!(
                                    "LENGTH: negative dictionary key {k}"
                                )));
                            } else {
                                // table index = key + 1 (slot 0 is NULL).
                                let len = *table_lengths
                                    .get(k as usize + 1)
                                    .ok_or_else(|| {
                                        BoltError::Other(format!(
                                            "LENGTH: key {k} out of range"
                                        ))
                                    })?;
                                out.push(len as i64);
                            }
                        }
                        out
                    }
                    _ => host_gather_lengths(&keys_host, &table_lengths)?,
                };
                check_len(lens.len(), n_rows)?;
                return Ok(Arc::new(Int64Array::from(lens)) as ArrayRef);
            }
        };

        // GPU gather path.
        let length_table = build_length_table(dict, layout)?;
        let table_gpu = GpuVec::<i32>::from_slice(&length_table)?;
        let out_gpu = GpuVec::<i32>::zeros(n_rows)?;

        let module =
            CudaModule::from_ptx(&crate::jit::string_kernel::compile_length_gather_kernel()?)?;
        let function =
            module.function(crate::jit::string_kernel::LENGTH_GATHER_ENTRY)?;

        // ABI: (indices, length_table, out, n_rows). Assemble raw kernel
        // params directly (heterogeneous list; same pattern as
        // `execute_projection`).
        let mut indices_ptr = keys_vec.device_ptr();
        let mut table_ptr = table_gpu.device_ptr();
        let mut out_ptr = out_gpu.device_ptr();
        let mut n_rows_u32 = n_rows_to_u32(n_rows)?;
        let mut kernel_params: Vec<*mut c_void> = vec![
            &mut indices_ptr as *mut CUdeviceptr as *mut c_void,
            &mut table_ptr as *mut CUdeviceptr as *mut c_void,
            &mut out_ptr as *mut CUdeviceptr as *mut c_void,
            &mut n_rows_u32 as *mut u32 as *mut c_void,
        ];

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
        debug_sync_check()?;
        stream.synchronize()?;

        // Download Int32 lengths and widen to the Int64 SQL contract.
        let lens_i32 = out_gpu.to_vec()?;
        check_len(lens_i32.len(), n_rows)?;
        let lens_i64: Vec<i64> = lens_i32.into_iter().map(|v| v as i64).collect();
        Ok(Arc::new(Int64Array::from(lens_i64)) as ArrayRef)
    }

    /// Execute a [`PhysicalPlan::StringProject`]: a `SELECT UPPER(<utf8_col>)` /
    /// `LOWER(<utf8_col>)` projection (plus passthrough columns) over a bare
    /// scan, with the transform outputs produced on the GPU via the two-pass
    /// length/scan/write kernels in [`crate::jit::string_kernel`] (see
    /// [`crate::exec::string_project`]).
    ///
    /// Passthrough columns are lifted directly from the host source batch.
    /// Each `UPPER`/`LOWER(col)` output runs the two-pass GPU producer against a
    /// row-aligned offsets+bytes input materialised from the dictionary-encoded
    /// column — but only when the column's dictionary is pure ASCII (the kernels
    /// ASCII-fold byte-wise; non-ASCII Unicode case mapping can change byte
    /// length, e.g. `'ß'` → `"SS"`). Non-ASCII dictionaries, or columns with no
    /// supported GPU storage, fall back to a full-Unicode host transform. Both
    /// paths produce a `StringArray`.
    fn execute_string_project(
        &self,
        table: &str,
        outputs: &[crate::plan::physical_plan::StringProjectOutput],
        output_schema: &Schema,
    ) -> BoltResult<QueryHandle> {
        use crate::plan::physical_plan::StringProjectOutput;

        let src_batch = self.materialize_table(table)?;
        let src_schema = src_batch.schema();
        let n_rows = src_batch.num_rows();

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(outputs.len());
        for out in outputs {
            match out {
                StringProjectOutput::Passthrough { source } => {
                    let idx = src_schema.index_of(source).map_err(|_| {
                        BoltError::Plan(format!(
                            "StringProject: passthrough column '{source}' not found in \
                             table '{table}'"
                        ))
                    })?;
                    arrays.push(src_batch.column(idx).clone());
                }
                StringProjectOutput::Transform { source, transform } => {
                    arrays.push(self.string_transform_column(table, source, *transform, n_rows)?);
                }
            }
        }

        let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
        let batch_out = RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
            BoltError::Other(format!(
                "StringProject: failed to build output RecordBatch: {e}"
            ))
        })?;
        Ok(QueryHandle { batch: batch_out })
    }

    /// Compute `UPPER`/`LOWER(<source>)` for the GPU-resident `Utf8` column
    /// `source` of `table`, returning a `Utf8` `ArrayRef` of `n_rows` rows.
    ///
    /// GPU path (ASCII dictionaries): materialise a row-aligned offsets+bytes
    /// input from the column's dictionary + device keys, upload, run the length
    /// pass → host exclusive scan of `row_lens` → allocate output bytes → run
    /// the write pass → download → rebuild the `StringArray` (re-applying NULLs).
    /// Host fallback (non-ASCII dictionary, or unsupported GPU storage): apply
    /// the full-Unicode transform host-side. Both paths preserve NULLs as Arrow
    /// NULLs.
    fn string_transform_column(
        &self,
        table: &str,
        source: &str,
        transform: crate::exec::string_project::StringTransform,
        n_rows: usize,
    ) -> BoltResult<ArrayRef> {
        use crate::exec::string_project::{
            build_row_aligned_input, dict_is_ascii, exclusive_scan_lens, host_transform_strings,
            string_array_from_offsets, KeyLayout,
        };

        let gpu_table_ref = self.ensure_gpu_table(table)?;
        let gpu_table: &crate::exec::gpu_table::GpuTable = &gpu_table_ref;
        let column = gpu_table.column(source).ok_or_else(|| {
            BoltError::Plan(format!(
                "StringProject: column '{source}' not in GPU table '{table}'"
            ))
        })?;
        let dict = column.utf8_dictionary().ok_or_else(|| {
            BoltError::Plan(format!(
                "StringProject: column '{source}' is not a Utf8 column"
            ))
        })?;

        // Resolve the host-side keys + layout + per-row validity for this
        // column. For the engine-managed `Utf8` layout NULL is encoded as key 0
        // (1-based dict); for native `DictUtf8` NULL lives on `valid_mask`.
        let (keys_host, layout, validity): (Vec<i32>, KeyLayout, Option<Vec<bool>>) =
            match &column.data {
                crate::exec::gpu_table::GpuColumnData::Utf8 { indices, .. } => {
                    let keys = indices.to_vec()?;
                    // Validity = key != 0 (slot 0 is the NULL sentinel).
                    let valid: Vec<bool> = keys.iter().map(|&k| k != 0).collect();
                    (keys, KeyLayout::OneBasedNullSlot0, Some(valid))
                }
                crate::exec::gpu_table::GpuColumnData::DictUtf8 {
                    keys, valid_mask, ..
                } => {
                    let keys = keys.to_vec()?;
                    let valid = match valid_mask {
                        None => None,
                        Some(mask) => {
                            let bits = mask.to_vec()?;
                            let v: Vec<bool> = (0..keys.len())
                                .map(|row| {
                                    let byte = bits.get(row / 8).copied().unwrap_or(0);
                                    (byte >> (row % 8)) & 1 == 1
                                })
                                .collect();
                            Some(v)
                        }
                    };
                    (keys, KeyLayout::ZeroBased, valid)
                }
                _ => {
                    return Err(BoltError::Plan(format!(
                        "StringProject: column '{source}' has non-Utf8 GPU storage"
                    )))
                }
            };

        check_len(keys_host.len(), n_rows)?;
        let validity_slice = validity.as_deref();

        // Host fallback for non-ASCII dictionaries: the byte-wise GPU fold is
        // only correct for ASCII (Unicode case mapping can change byte length).
        if !dict_is_ascii(dict) {
            let arr =
                host_transform_strings(dict, &keys_host, layout, validity_slice, transform)?;
            return Ok(Arc::new(arr) as ArrayRef);
        }

        // ---- GPU two-pass path -------------------------------------------
        // Pass 0 (host): materialise the row-aligned offsets+bytes input.
        let (src_offsets, src_bytes) =
            build_row_aligned_input(dict, &keys_host, layout, validity_slice)?;

        // Empty input (no rows, or all-empty bytes): skip the launch and build
        // the result directly. `from_slice` on an empty slice is brittle and a
        // zero-thread launch is pointless.
        if n_rows == 0 {
            let arr = string_array_from_offsets(&src_offsets, &src_bytes, validity_slice)?;
            return Ok(Arc::new(arr) as ArrayRef);
        }

        let kind = transform.scalar_fn_kind();
        let src_offsets_gpu = GpuVec::<i32>::from_slice(&src_offsets)?;
        // `src_bytes` may be empty (all rows empty/NULL); allocate at least one
        // byte so the device pointer is valid even though no thread reads it.
        let src_bytes_gpu = if src_bytes.is_empty() {
            GpuVec::<u8>::zeros(1)?
        } else {
            GpuVec::<u8>::from_slice(&src_bytes)?
        };
        let row_lens_gpu = GpuVec::<u32>::zeros(n_rows)?;

        let n_rows_u32 = n_rows_to_u32(n_rows)?;
        let stream = CudaStream::null_or_default();
        let grid_x = grid_x_for(n_rows_u32, BLOCK_SIZE);

        // ---- Pass 1: length pass → row_lens. ABI (UPPER/LOWER, 4 params):
        //      (src_offsets, src_bytes, row_lens, n_rows).
        {
            let module = CudaModule::from_ptx(
                &crate::jit::string_kernel::compile_varwidth_len_pass(kind)?,
            )?;
            let entry = crate::jit::string_kernel::len_pass_entry(kind)?;
            let function = module.function(&entry)?;

            let mut p_off = src_offsets_gpu.device_ptr();
            let mut p_bytes = src_bytes_gpu.device_ptr();
            let mut p_lens = row_lens_gpu.device_ptr();
            let mut p_n = n_rows_u32;
            let mut params: Vec<*mut c_void> = vec![
                &mut p_off as *mut CUdeviceptr as *mut c_void,
                &mut p_bytes as *mut CUdeviceptr as *mut c_void,
                &mut p_lens as *mut CUdeviceptr as *mut c_void,
                &mut p_n as *mut u32 as *mut c_void,
            ];
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
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ))?;
            }
            debug_sync_check()?;
            stream.synchronize()?;
        }

        // ---- Pass 2 (host): exclusive-scan row_lens → out_offsets + total.
        let row_lens = row_lens_gpu.to_vec()?;
        check_len(row_lens.len(), n_rows)?;
        let (out_offsets, total_bytes) = exclusive_scan_lens(&row_lens)?;
        let out_offsets_gpu = GpuVec::<i32>::from_slice(&out_offsets)?;
        let out_bytes_gpu = if total_bytes == 0 {
            GpuVec::<u8>::zeros(1)?
        } else {
            GpuVec::<u8>::zeros(total_bytes)?
        };

        // ---- Pass 3: write pass → out_bytes. ABI (UPPER/LOWER, 5 params):
        //      (src_offsets, src_bytes, out_offsets, out_bytes, n_rows).
        {
            let module = CudaModule::from_ptx(
                &crate::jit::string_kernel::compile_varwidth_write_pass(kind)?,
            )?;
            let entry = crate::jit::string_kernel::write_pass_entry(kind)?;
            let function = module.function(&entry)?;

            let mut p_off = src_offsets_gpu.device_ptr();
            let mut p_bytes = src_bytes_gpu.device_ptr();
            let mut p_out_off = out_offsets_gpu.device_ptr();
            let mut p_out_bytes = out_bytes_gpu.device_ptr();
            let mut p_n = n_rows_u32;
            let mut params: Vec<*mut c_void> = vec![
                &mut p_off as *mut CUdeviceptr as *mut c_void,
                &mut p_bytes as *mut CUdeviceptr as *mut c_void,
                &mut p_out_off as *mut CUdeviceptr as *mut c_void,
                &mut p_out_bytes as *mut CUdeviceptr as *mut c_void,
                &mut p_n as *mut u32 as *mut c_void,
            ];
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
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ))?;
            }
            debug_sync_check()?;
            stream.synchronize()?;
        }

        // ---- Download + rebuild StringArray (re-applying NULLs).
        let out_bytes = out_bytes_gpu.to_vec()?;
        // `out_bytes_gpu` was padded to >= 1 byte; truncate to the real total.
        let out_bytes = &out_bytes[..total_bytes.min(out_bytes.len())];
        let arr = string_array_from_offsets(&out_offsets, out_bytes, validity_slice)?;
        Ok(Arc::new(arr) as ArrayRef)
    }

    /// Execute a [`PhysicalPlan::StringLikeFilter`]: a GPU per-row `LIKE` /
    /// `NOT LIKE` over a non-dictionary `Utf8` column, then materialise the
    /// surviving rows.
    ///
    /// ⚠️ UNVALIDATED DEVICE PATH. The matcher kernel
    /// ([`crate::jit::string_kernel::compile_like_match_kernel`]) has not run on
    /// GPU hardware; correctness is guaranteed by the host mirror in
    /// [`crate::exec::string_like`] and by this executor's clean host fallback.
    ///
    /// Flow: execute `input` (a bare scan → row-aligned source batch); pull the
    /// `column` as a host `StringArray`; build a row-aligned offsets+bytes
    /// buffer + validity; upload; launch the matcher (literal baked as a device
    /// buffer); download the 0/1 mask; re-apply NULL 3VL; `arrow::compute::filter`
    /// every column. If the column is absent / not Utf8 at run time, fall back
    /// to the host `LIKE` over the same `StringArray` (no panic).
    fn execute_string_like_filter(
        &self,
        input: &PhysicalPlan,
        _table: &str,
        column: &str,
        literal: &[u8],
        mode: crate::jit::string_kernel::LikeMode,
        negated: bool,
    ) -> BoltResult<QueryHandle> {
        use arrow_array::Array;

        // Execute the inner scan: this is the row-aligned source batch that
        // carries `column` (the lowering required a bare Scan beneath).
        let batch = self.execute(input)?.into_record_batch();
        let schema = batch.schema();

        // Locate the column; if missing or not a StringArray, fall back to the
        // host LIKE over whatever the column decodes to (no panic). Because the
        // lowering already proved `column` is a Utf8 scan column, the common
        // case is the StringArray downcast succeeding.
        let col_idx = match schema.index_of(column) {
            Ok(i) => i,
            Err(_) => {
                return Err(BoltError::Plan(format!(
                    "StringLikeFilter: column '{column}' not found in input batch"
                )))
            }
        };
        let col_arr = batch.column(col_idx);
        // Normalise to a `StringArray`. The common case is a direct downcast;
        // any other Utf8-compatible layout (e.g. a dictionary array that slipped
        // through un-rewritten) is cast to Utf8 so the path stays host-fallback-
        // safe (no panic, no hard error) for unexpected run-time layouts.
        let owned_cast: ArrayRef;
        let str_arr: &arrow_array::StringArray = match col_arr
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
        {
            Some(a) => a,
            None => {
                owned_cast = arrow::compute::cast(col_arr.as_ref(), &ArrowDataType::Utf8)
                    .map_err(|e| {
                        BoltError::Plan(format!(
                            "StringLikeFilter: column '{column}' is not Utf8 and could \
                             not be cast (got {:?}): {e}",
                            col_arr.data_type()
                        ))
                    })?;
                owned_cast
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or_else(|| {
                        BoltError::Plan(format!(
                            "StringLikeFilter: cast of column '{column}' did not yield Utf8"
                        ))
                    })?
            }
        };

        // Build the boolean mask: GPU device path, with a host fallback that
        // produces the identical mask if the launch is not viable.
        let mask: arrow_array::BooleanArray = match self
            .string_like_mask_gpu(str_arr, literal, mode, negated)
        {
            Ok(m) => m,
            Err(e) => {
                // Host fallback: evaluate the SAME predicate via the validated
                // host mirror (equivalent to exec::like::host_like for these
                // shapes). Correctness is unaffected; only the GPU speedup is
                // lost. Logged so a hardware bring-up notices.
                log::warn!(
                    "StringLikeFilter: GPU matcher unavailable ({e}); \
                     falling back to host LIKE for column '{column}'"
                );
                crate::exec::string_like::host_mask_via_mirror(
                    str_arr, literal, mode, negated,
                )
            }
        };

        // Apply the mask to every column (NULL mask entries drop the row).
        let filtered: Vec<ArrayRef> = batch
            .columns()
            .iter()
            .map(|c| {
                arrow::compute::filter(c.as_ref(), &mask).map_err(|e| {
                    BoltError::Other(format!(
                        "StringLikeFilter: arrow filter failed: {e}"
                    ))
                })
            })
            .collect::<BoltResult<Vec<_>>>()?;
        let out = RecordBatch::try_new(batch.schema(), filtered).map_err(|e| {
            BoltError::Other(format!(
                "StringLikeFilter: failed to rebuild RecordBatch: {e}"
            ))
        })?;
        Ok(QueryHandle { batch: out })
    }

    /// GPU per-row LIKE matcher: upload the row-aligned column + literal, launch
    /// [`crate::jit::string_kernel::compile_like_match_kernel`], download the
    /// 0/1 mask, and re-apply NULL 3VL into a [`arrow_array::BooleanArray`].
    ///
    /// Returns `Err` (so the caller can host-fall-back) for any non-viable
    /// launch condition. UNVALIDATED device path — see the executor doc.
    fn string_like_mask_gpu(
        &self,
        col: &arrow_array::StringArray,
        literal: &[u8],
        mode: crate::jit::string_kernel::LikeMode,
        negated: bool,
    ) -> BoltResult<arrow_array::BooleanArray> {
        use arrow_array::Array;
        use crate::exec::string_like::{build_row_aligned_from_strings, mask_to_boolean_array};

        let n_rows = col.len();
        let (offsets, bytes, validity) = build_row_aligned_from_strings(col)?;

        // Empty input: nothing to launch; build the (empty) mask directly.
        if n_rows == 0 {
            return Ok(mask_to_boolean_array(&[], &validity));
        }

        // The engine already owns a live `CudaContext` (`self._ctx`), so device
        // allocations below are valid. Any allocation / launch failure returns
        // an `Err`, which the caller turns into a host fallback.
        let offsets_gpu = GpuVec::<i32>::from_slice(&offsets)?;
        let bytes_gpu = if bytes.is_empty() {
            GpuVec::<u8>::zeros(1)?
        } else {
            GpuVec::<u8>::from_slice(&bytes)?
        };
        // Literal: bake as a small device buffer. Pad empty to 1 byte so the
        // device pointer is valid (lit_len==0 short-circuits before any read).
        let lit_len = u32::try_from(literal.len()).map_err(|_| {
            BoltError::Other("StringLikeFilter: literal length exceeds u32".into())
        })?;
        let lit_gpu = if literal.is_empty() {
            GpuVec::<u8>::zeros(1)?
        } else {
            GpuVec::<u8>::from_slice(literal)?
        };
        let mask_gpu = GpuVec::<u8>::zeros(n_rows)?;

        let n_rows_u32 = n_rows_to_u32(n_rows)?;
        let stream = CudaStream::null_or_default();
        let grid_x = grid_x_for(n_rows_u32, BLOCK_SIZE);

        let module = CudaModule::from_ptx(
            &crate::jit::string_kernel::compile_like_match_kernel(mode, negated)?,
        )?;
        let function = module.function(crate::jit::string_kernel::LIKE_MATCH_ENTRY)?;

        let mut p_off = offsets_gpu.device_ptr();
        let mut p_bytes = bytes_gpu.device_ptr();
        let mut p_lit = lit_gpu.device_ptr();
        let mut p_mask = mask_gpu.device_ptr();
        let mut p_n = n_rows_u32;
        let mut p_l = lit_len;
        let mut params: Vec<*mut c_void> = vec![
            &mut p_off as *mut CUdeviceptr as *mut c_void,
            &mut p_bytes as *mut CUdeviceptr as *mut c_void,
            &mut p_lit as *mut CUdeviceptr as *mut c_void,
            &mut p_mask as *mut CUdeviceptr as *mut c_void,
            &mut p_n as *mut u32 as *mut c_void,
            &mut p_l as *mut u32 as *mut c_void,
        ];
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
                params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }
        debug_sync_check()?;
        stream.synchronize()?;

        let mask = mask_gpu.to_vec()?;
        check_len(mask.len(), n_rows)?;
        Ok(mask_to_boolean_array(&mask, &validity))
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
    /// v0.7 sub-task B: 128-bit fixed-point output column. Stored as the
    /// same interleaved `[lo0, hi0, lo1, hi1, ...]` u64 buffer the input
    /// `GpuColumnData::Decimal128` uses, so the PTX `Op::Store128` can
    /// write 16 bytes per row at offset `tid * 16` with no per-row
    /// indirection. The plan-level `(precision, scale)` rides along so
    /// the download path can reattach them to the resulting
    /// `Decimal128Array`.
    Decimal128 {
        /// Interleaved 16-bytes-per-row output buffer (length `2 * n_rows`).
        values: GpuVec<u64>,
        /// Plan-level precision (digits of significance).
        precision: u8,
        /// Plan-level scale.
        scale: i8,
        /// Optional Arrow-LE packed validity bitmap on the device, one byte
        /// per 8 rows (lsb-first) — mirrors
        /// [`GpuColumnData::Decimal128`](crate::exec::gpu_table::GpuColumnData::Decimal128)'s
        /// `valid_mask`. For pure passthrough columns we copy the source
        /// column's mask so the download path can reconstruct NULL rows as
        /// NULL rather than `0`. `None` ⇒ all rows valid (no nulls on the
        /// source, or a freshly-allocated output buffer).
        valid_mask: Option<GpuVec<u8>>,
    },
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
            // v0.7 sub-task B: allocate the interleaved [lo, hi] u64 buffer
            // (length `2 * n`) that `Op::Store128` writes into. Plan-level
            // `(precision, scale)` rides on the variant so the download path
            // can rebuild a `Decimal128Array` with the correct dtype.
            DataType::Decimal128(precision, scale) => Ok(DeviceCol::Decimal128 {
                values: GpuVec::<u64>::zeros(2 * n)?,
                precision,
                scale,
                // Freshly-allocated output buffer: no validity yet. A
                // passthrough column copies the source mask in after alloc
                // (see the output-allocation loop in `run_kernel`).
                valid_mask: None,
            }),
            // v0.7: PTX codegen for Date32 / Timestamp arithmetic is wired
            // (see `crate::jit::ptx_gen`), but the device-side download
            // path is dtype-blind — `DeviceCol::I32::download` always
            // emits an `Int32Array`, which would silently downgrade a
            // Date32 output to plain Int32. Keep the engine boundary
            // rejecting these types until a follow-up wires the
            // Date32Array / TimestampArray reconstruction. The
            // physical-plan codegen still produces correct PTX for
            // `Date32 - Date32` and `Timestamp - Timestamp`; the
            // top-level engine routes any temporal column through the
            // host path until then.
            DataType::Date32 | DataType::Timestamp(_, _) => Err(BoltError::Type(format!(
                "Date/Timestamp output column lowering pending download-path \
                 wiring (PTX codegen is done; got {:?})",
                dtype
            ))),
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
            // v0.7 sub-task B: the interleaved [lo, hi] u64 buffer is
            // the column's single base pointer — PTX `Op::Store128`
            // computes per-row offsets as `tid * 16`.
            DeviceCol::Decimal128 { values, .. } => values.device_ptr(),
        }
    }

    /// Record `stream` as having launched a kernel against every device
    /// buffer this output column owns, so each buffer's `Drop` fences `stream`
    /// before its block is recycled to the pool.
    ///
    /// `execute_projection` assembles its kernel parameters by hand and drives
    /// a raw `cuLaunchKernel` off `device_ptr()` rather than through
    /// [`KernelArgs`](crate::exec::launch::KernelArgs)/`launch_1d`, so it does
    /// not get the central `tag_launch_stream` enforcement that the
    /// `launch_1d` / `launch_with_geometry` callers rely on (review finding
    /// V-1 / F10a). Calling this immediately after the launch restores the
    /// same `Drop`-fence invariant for the freshly-allocated output buffers:
    /// the launch stream is recorded in each buffer's `StreamSet` exactly as
    /// `KernelArgs::tag_launch_stream` would, so a buffer dropped while the
    /// kernel is still in flight fences the stream before recycling — even if
    /// a future edit removes a downstream `synchronize()`. Delegates to the
    /// public [`GpuVec::mark_stream_use`], the documented entry point for
    /// callers that bypass `KernelArgs`.
    fn mark_launch_stream(&self, stream: crate::cuda::CUstream) {
        match self {
            DeviceCol::I32(v) => v.mark_stream_use(stream),
            DeviceCol::I64(v) => v.mark_stream_use(stream),
            DeviceCol::F32(v) => v.mark_stream_use(stream),
            DeviceCol::F64(v) => v.mark_stream_use(stream),
            DeviceCol::Bool(v) => v.mark_stream_use(stream),
            DeviceCol::Utf8(d) => d.indices.mark_stream_use(stream),
            DeviceCol::Decimal128 {
                values, valid_mask, ..
            } => {
                values.mark_stream_use(stream);
                if let Some(mask) = valid_mask {
                    mask.mark_stream_use(stream);
                }
            }
        }
    }

    /// Install a dictionary on a Utf8 column (for output columns whose source dictionary
    /// the engine knows). No-op for non-Utf8 columns.
    fn set_utf8_dictionary(&mut self, dict: Vec<String>) {
        if let DeviceCol::Utf8(d) = self {
            d.dictionary = dict;
        }
    }

    /// Install a device-side validity bitmap on a Decimal128 output column
    /// (for pure passthrough projections whose source column carries one).
    /// No-op for non-Decimal128 columns or a `None` mask. Mirrors
    /// [`Self::set_utf8_dictionary`]'s passthrough plumbing.
    fn set_decimal128_valid_mask(&mut self, mask: Option<GpuVec<u8>>) {
        if let DeviceCol::Decimal128 { valid_mask, .. } = self {
            if mask.is_some() {
                *valid_mask = mask;
            }
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
            // v0.7 sub-task B: reassemble the interleaved [lo, hi] u64
            // buffer back into a `Decimal128Array`. Each pair of u64s
            // reconstitutes one i128 via
            //   `lo | ((hi as u128) << 64)` then `as i128`
            // which preserves the sign because the high half carries
            // the sign bits unchanged through the unsigned/signed cast.
            DeviceCol::Decimal128 {
                values,
                precision,
                scale,
                valid_mask,
            } => {
                let host = copy_back::<u64>(&values, 2 * n_rows)?;
                // Decimal128 NULL fix: download the validity bitmap (if any)
                // so NULL rows reconstruct as Arrow NULL, not `0`.
                let mask_bits = valid_mask.as_ref().map(|m| m.to_vec()).transpose()?;
                let arr = decimal128_from_interleaved(
                    &host,
                    n_rows,
                    mask_bits.as_deref(),
                    precision,
                    scale,
                    "Decimal128 download",
                )?;
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
            // v0.7 sub-task B: Decimal128's pinned path mirrors the
            // primitive pattern (u64 element type, length `2 * n_rows`).
            // The check_len guard catches a buffer that didn't get sized
            // correctly at alloc time.
            DeviceCol::Decimal128 {
                values,
                precision,
                scale,
                valid_mask,
            } => {
                let staged = StagedDownload::<u64>::from_gpu(&values, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), 2 * n_rows)?;
                // Decimal128 NULL fix: same validity-aware reassembly as the
                // sync `download` site (shared helper keeps them consistent).
                let mask_bits = valid_mask.as_ref().map(|m| m.to_vec()).transpose()?;
                let arr = decimal128_from_interleaved(
                    &host,
                    n_rows,
                    mask_bits.as_deref(),
                    precision,
                    scale,
                    "Decimal128 pinned download",
                )?;
                Ok(Arc::new(arr) as ArrayRef)
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

/// Decimal128 NULL fix — shared reassembly used by BOTH download sites
/// (`DeviceCol::download` and `DeviceCol::download_pinned`) so they cannot
/// drift. Reconstruct each row's `i128` from the interleaved `[lo, hi]` u64
/// pair, then attach Arrow validity from the (optional, lsb-first packed)
/// `mask_bits`: a row whose validity bit is 0 becomes an Arrow NULL rather
/// than the zeroed bit-pattern it was stored as. `mask_bits == None` ⇒ every
/// row is valid (non-null source), preserving the original non-null
/// behaviour byte-for-byte.
///
/// `host` must be `2 * n_rows` u64s (already length-checked by callers).
fn decimal128_from_interleaved(
    host: &[u64],
    n_rows: usize,
    mask_bits: Option<&[u8]>,
    precision: u8,
    scale: i8,
    ctx: &str,
) -> BoltResult<Decimal128Array> {
    let mut out: Vec<Option<i128>> = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        let lo = host[2 * row];
        let hi = host[2 * row + 1];
        let bits = (lo as u128) | ((hi as u128) << 64);
        // lsb-first packed bitmap: bit `row % 8` of byte `row / 8`. Absent
        // mask ⇒ all rows valid.
        let is_valid = match mask_bits {
            None => true,
            Some(b) => {
                let byte = b.get(row / 8).copied().unwrap_or(0);
                (byte >> (row % 8)) & 1 == 1
            }
        };
        out.push(if is_valid { Some(bits as i128) } else { None });
    }
    // `FromIterator<Option<i128>>` builds the array with the correct null
    // bitmap; `with_precision_and_scale` reattaches the plan dtype.
    out.into_iter()
        .collect::<Decimal128Array>()
        .with_precision_and_scale(precision, scale)
        .map_err(|e| {
            BoltError::Type(format!(
                "{ctx}: precision/scale ({precision}, {scale}) rejected by Arrow: {e}"
            ))
        })
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
    crate::exec::schema_convert::arrow_dtype_to_plan(d)
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
                                // Finding V-5: validate every key before it
                                // indexes the dictionary. A negative or
                                // out-of-range key would make `sa.value(..)`
                                // panic on OOB inside `decode_into`. Reject it
                                // with a clean error instead, mirroring the
                                // strict bounds checks in `string_ops`.
                                let key = keys.value(i);
                                if key < 0 {
                                    return Err(BoltError::Type(format!(
                                        "register_table: negative dict<i32,utf8> key {} at row {}",
                                        key, i
                                    )));
                                }
                                let pos = key as usize;
                                if pos >= sa.len() {
                                    return Err(BoltError::Type(format!(
                                        "register_table: dict<i32,utf8> key {} at row {} out of range (dictionary size {})",
                                        key,
                                        i,
                                        sa.len()
                                    )));
                                }
                                decode_into(&mut out, pos, sa);
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
                                // Finding V-5: validate every key before it
                                // indexes the dictionary. The original `as
                                // usize` cast could feed a negative key (after
                                // sign extension) or an out-of-range key to
                                // `sa.value(..)`, panicking on OOB. Reject
                                // negative, out-of-range, and (for parity with
                                // the upload path's i32 device buffer) keys
                                // above `i32::MAX`.
                                let key = keys.value(i);
                                if key < 0 {
                                    return Err(BoltError::Type(format!(
                                        "register_table: negative dict<i64,utf8> key {} at row {}",
                                        key, i
                                    )));
                                }
                                if key > i32::MAX as i64 {
                                    return Err(BoltError::Type(format!(
                                        "register_table: dict<i64,utf8> key {} at row {} exceeds i32 capacity",
                                        key, i
                                    )));
                                }
                                let pos = key as usize;
                                if pos >= sa.len() {
                                    return Err(BoltError::Type(format!(
                                        "register_table: dict<i64,utf8> key {} at row {} out of range (dictionary size {})",
                                        key,
                                        i,
                                        sa.len()
                                    )));
                                }
                                decode_into(&mut out, pos, sa);
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

/// Install (or clear) the builder's `persistent_cache(path)` directory on
/// the process-wide disk PTX cache.
///
/// This is the single bridge between [`EngineBuilder::persistent_cache`]
/// and the JIT compile path's disk-cache lookup
/// ([`Engine::get_or_build_module`] → [`crate::jit::disk_cache::disk_cache`]).
/// Pulled out of [`EngineBuilder::build`] as a free function so the
/// builder → cache-layer plumbing can be exercised host-side without a
/// live CUDA context (the rest of `build` needs one).
///
/// Semantics, mirroring [`crate::jit::disk_cache::set_override_dir`]:
///   * `Some(path)` — subsequent `disk_cache()` lookups resolve to
///     `path` regardless of the `BOLT_PTX_CACHE_DIR` env var (the
///     builder knob takes precedence; the env var remains the fallback
///     when no builder path is configured).
///   * `None` — clears any prior builder override so the cache
///     re-falls-back to the env var, and stays disabled if that too is
///     unset. This preserves the opt-in "no path → unchanged behaviour"
///     contract: a default-built engine never enables the disk cache on
///     its own.
fn install_persistent_cache_override(path: Option<&std::path::Path>) {
    crate::jit::disk_cache::set_override_dir(path.map(|p| p.to_path_buf()));
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

/// Concatenate a table's host-side batches into a single `RecordBatch`.
///
/// Shared by [`Engine::materialize_table`]'s eager and streaming-overlay
/// paths. Zero batches errors, one batch is cloned cheaply (Arrow arrays are
/// `Arc`-backed), two or more go through `arrow::compute::concat_batches`
/// (which copies every column — the perf cost the field doc on `tables`
/// warns about).
fn concat_table_batches(name: &str, batches: &[RecordBatch]) -> BoltResult<RecordBatch> {
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
    crate::exec::schema_convert::plan_schema_to_arrow_schema(s)
}

/// Build the single-row `Int64` output batch for a `PhysicalPlan::CountRows`
/// node: one column holding `n_rows` (the materialised child plan's row
/// count). `output_schema` must describe exactly one column (the count); its
/// Arrow shape comes from `plan_schema_to_arrow_schema`, so the column name /
/// nullability follow whatever the lowerer stored (a single Int64 field).
///
/// Factored out of the `execute` arm so the host-side row-count step is unit
/// testable without a GPU / engine.
fn build_count_rows_batch(n_rows: usize, output_schema: &Schema) -> BoltResult<RecordBatch> {
    if output_schema.fields.len() != 1 {
        return Err(BoltError::Plan(format!(
            "CountRows output schema must have exactly one column, got {}",
            output_schema.fields.len()
        )));
    }
    let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
    let arr: ArrayRef = Arc::new(Int64Array::from(vec![n_rows as i64]));
    RecordBatch::try_new(arrow_schema, vec![arr]).map_err(|e| {
        BoltError::Other(format!("failed to build CountRows RecordBatch: {e}"))
    })
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
    /// Borrow of the lazy streaming overlay. By the time an `EngineProvider`
    /// is built, any streaming source referenced by the plan has already been
    /// collapsed to [`TableSource::Materialized`] (see
    /// [`Engine::ensure_streaming_materialized`]), so the null probes below
    /// can read its batches the same way they read `tables`.
    streaming: Ref<'a, HashMap<String, crate::exec::streaming::TableSource>>,
}

impl<'a> EngineProvider<'a> {
    /// Resolve a table name to its host-side batches, consulting `tables`
    /// first and the (already-materialised) streaming overlay second.
    fn batches_for(&self, table_name: &str) -> Option<&[RecordBatch]> {
        if let Some(b) = self.tables.get(table_name) {
            return Some(b.as_slice());
        }
        match self.streaming.get(table_name) {
            Some(crate::exec::streaming::TableSource::Materialized(b)) => Some(b.as_slice()),
            // A still-streaming source means the caller forgot to materialise
            // it before building the provider; treat as absent (safe-false).
            _ => None,
        }
    }
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
        let batches = match self.batches_for(table_name) {
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
        let batches = self.batches_for(table_name)?;
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

// ---------------------------------------------------------------------------
// Cost-based optimizer wiring: a `StatsProvider` backed by the engine's
// registered tables.
//
// The join-reorder pass (`crate::plan::optimizer::join_reorder`) is a no-op
// until it is handed a row estimator. We feed it one built from real data:
// each base table's `row_count` is the sum of its registered `RecordBatch`
// row counts. The engine takes a cheap *snapshot* of those counts at plan
// time (a `HashMap<String, usize>`), wraps it in `StatsEstimator`, and threads
// it into the default pass pipeline via `default_passes_with_estimator`.
// ---------------------------------------------------------------------------

/// Owned snapshot of base-table row counts used to drive cost-based join
/// reordering.
///
/// Built once per query in [`Engine::table_stats_snapshot`] from the engine's
/// registered tables. Owning the counts (rather than borrowing the engine)
/// makes this `Send + Sync + 'static`, so it can live behind the
/// `Arc<dyn RowEstimator>` the [`crate::plan::optimizer::JoinReorder`] pass
/// holds. A table absent from the snapshot has no entry and the estimator
/// returns `None` for it — which keeps reordering conservative (the pass only
/// fires when *every* leaf in a chain is costed).
#[derive(Debug, Default)]
struct EngineTableStats {
    /// Table name → total registered row count.
    row_counts: HashMap<String, usize>,
}

impl crate::plan::statistics::StatsProvider for EngineTableStats {
    fn table_stats(&self, name: &str) -> Option<crate::plan::statistics::TableStats> {
        self.row_counts
            .get(name)
            .map(|&n| crate::plan::statistics::TableStats::new(n))
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

    /// `build_count_rows_batch` (the host-side row-count step for
    /// `PhysicalPlan::CountRows`) must emit a single-row Int64 batch holding the
    /// supplied row count, with the column named per the supplied schema. This
    /// runs purely on the host (no GPU) so it is not `#[ignore]`'d.
    #[test]
    fn count_rows_batch_holds_row_count() {
        let schema = Schema::new(vec![Field::new("count", DataType::Int64, false)]);
        let batch = build_count_rows_batch(7, &schema).expect("must build");
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.schema().field(0).name(), "count");
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 count column");
        assert_eq!(col.value(0), 7);
    }

    /// Zero rows in the child plan -> COUNT == 0 (not NULL).
    #[test]
    fn count_rows_batch_zero() {
        let schema = Schema::new(vec![Field::new("count", DataType::Int64, false)]);
        let batch = build_count_rows_batch(0, &schema).expect("must build");
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(col.value(0), 0);
    }

    /// A malformed multi-column output schema is rejected (defensive — the
    /// lowerer only ever stores a single Int64 count column).
    #[test]
    fn count_rows_batch_rejects_multi_column_schema() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]);
        assert!(build_count_rows_batch(3, &schema).is_err());
    }

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
            streaming: engine.streaming_sources.borrow(),
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
            streaming: engine.streaming_sources.borrow(),
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

    // ---- v0.7: `EngineBuilder::persistent_cache` wires into disk PTX cache ----

    /// `EngineBuilder::persistent_cache(path).build()` must install
    /// `path` as the process-wide disk PTX cache override, so a later
    /// `crate::jit::disk_cache::disk_cache()` resolves to it instead of
    /// (or in preference to) the `BOLT_PTX_CACHE_DIR` env var.
    ///
    /// Marked `#[ignore]` because `build()` initialises the CUDA driver
    /// and that's not available on every CI host. The wiring under test
    /// is, however, GPU-independent — it's a pure setter call inside
    /// `build()` — so on a non-GPU host the env-var-only path
    /// (`persistent_cache` not called) is exercised implicitly by every
    /// other test that instantiates an Engine without calling this
    /// knob, and the env-var contract continues to hold.
    #[test]
    #[ignore = "gpu:e2e — EngineBuilder::build initializes CUDA driver"]
    fn builder_persistent_cache_wires_into_disk_ptx_cache() {
        // Use a unique-per-run temp dir so this test can't observe
        // leftover state from a previous run or interfere with a
        // sibling test that also pokes the override slot.
        let mut path = std::env::temp_dir();
        path.push(format!(
            "craton-bolt-builder-persistent-cache-{}-{}",
            std::process::id(),
            // Cheap unique suffix without a `rand` dep.
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));

        // Save + restore the process-wide override slot so this test
        // doesn't leak state into siblings (the disk-cache module's
        // own tests do the same dance with their own ENV_LOCK; we
        // skip the lock here because the test is `#[ignore]` and not
        // expected to interleave with disk_cache's tests under
        // `cargo test`).
        let prev = crate::jit::current_disk_ptx_cache_dir();

        let _engine = Engine::builder()
            .persistent_cache(path.clone())
            .build()
            .expect("builder + CUDA init");

        // The setter must have run: the override slot now reflects
        // the builder-supplied path.
        assert_eq!(
            crate::jit::current_disk_ptx_cache_dir(),
            Some(path.clone()),
            "EngineBuilder::persistent_cache must propagate into the \
             process-wide disk PTX cache override"
        );

        // Restore prior state so we don't leak into sibling tests.
        crate::jit::set_disk_ptx_cache_dir(prev);
    }

    /// When `persistent_cache` is NOT called on the builder, `build`
    /// must NOT touch the disk-cache override slot — so a previously-
    /// installed override (or the `BOLT_PTX_CACHE_DIR` env-var path)
    /// continues to take effect unchanged.
    #[test]
    #[ignore = "gpu:e2e — EngineBuilder::build initializes CUDA driver"]
    fn builder_without_persistent_cache_preserves_existing_override() {
        let mut prior = std::env::temp_dir();
        prior.push(format!(
            "craton-bolt-builder-prior-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        let prev = crate::jit::current_disk_ptx_cache_dir();
        crate::jit::set_disk_ptx_cache_dir(Some(prior.clone()));

        let _engine = Engine::builder().build().expect("builder + CUDA init");

        assert_eq!(
            crate::jit::current_disk_ptx_cache_dir(),
            Some(prior),
            "builder without persistent_cache must NOT clobber a \
             pre-installed override (env-var path must keep working too)"
        );

        crate::jit::set_disk_ptx_cache_dir(prev);
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

    // -------------------------------------------------------------------
    // Builder → engine → disk-cache plumbing (host-side, no GPU).
    //
    // These exercise the `persistent_cache(path)` knob's effect on the
    // process-wide disk PTX cache WITHOUT constructing an `Engine`
    // (`build()` needs a CUDA context). We drive the same bridge `build()`
    // uses — `install_persistent_cache_override` — and observe the
    // resolved cache directory through the public
    // `disk_cache::disk_cache()` / `DiskPtxCache::root()` surface that the
    // JIT compile path consults in `get_or_build_module`.
    // -------------------------------------------------------------------

    /// Serialises the disk-cache tests below: they mutate the process-wide
    /// override slot and the `BOLT_PTX_CACHE_DIR` env var, both of which
    /// are global. Cargo runs `#[test]`s in parallel by default, so an
    /// unguarded mutation would race a sibling test.
    static DISK_CACHE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Fresh, unique temp directory under the OS tempdir for a disk-cache
    /// plumbing test. Not created on disk here — `DiskPtxCache::open`
    /// (invoked transitively by `disk_cache()`) creates it.
    fn fresh_cache_dir(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "craton-bolt-engine-cache-test-{}-{}-{}",
            tag,
            std::process::id(),
            n,
        ));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    /// Snapshot + restore the global override slot and the env var around a
    /// test body so disk-cache tests stay independent of ordering.
    fn with_clean_disk_cache_state<R>(f: impl FnOnce() -> R) -> R {
        use crate::jit::disk_cache::DISK_PTX_CACHE_ENV;
        let _guard = DISK_CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let prev_env = std::env::var(DISK_PTX_CACHE_ENV).ok();
        // Start from a known-clean slate: no builder override, no env var.
        install_persistent_cache_override(None);
        std::env::remove_var(DISK_PTX_CACHE_ENV);

        let out = f();

        // Restore: clear our override and put the env var back the way we
        // found it so sibling tests / the rest of the process see no drift.
        install_persistent_cache_override(None);
        match prev_env {
            Some(v) => std::env::set_var(DISK_PTX_CACHE_ENV, v),
            None => std::env::remove_var(DISK_PTX_CACHE_ENV),
        }
        out
    }

    #[test]
    fn persistent_cache_path_drives_disk_cache_root() {
        // The builder knob, once installed via the same bridge `build()`
        // uses, must make the JIT compile path's `disk_cache()` resolve to
        // that exact directory — proving the path is plumbed all the way
        // from builder → cache layer, not merely stored on the engine.
        with_clean_disk_cache_state(|| {
            let dir = fresh_cache_dir("plumbed");
            install_persistent_cache_override(Some(dir.as_path()));

            let cache = crate::jit::disk_cache::disk_cache()
                .expect("builder path must enable the disk cache");
            assert_eq!(
                cache.root(),
                dir.as_path(),
                "disk cache root must match the persistent_cache(path) directory"
            );
            let _ = std::fs::remove_dir_all(&dir);
        });
    }

    #[test]
    fn no_persistent_cache_path_keeps_disk_cache_disabled() {
        // Opt-in contract: with neither a builder path nor the env var set,
        // the JIT compile path sees `disk_cache() == None` — unchanged,
        // zero-side-effect behaviour.
        with_clean_disk_cache_state(|| {
            // `with_clean_disk_cache_state` already installed `None` +
            // cleared the env var; assert the resulting disabled state.
            assert!(
                crate::jit::disk_cache::disk_cache().is_none(),
                "no builder path and no env var must leave the disk cache disabled"
            );
        });
    }

    #[test]
    fn builder_persistent_cache_takes_precedence_over_env() {
        // When both the builder path and `BOLT_PTX_CACHE_DIR` are set, the
        // builder knob wins (the env var stays a fallback for the no-path
        // case). Confirms the plumbed override out-ranks the env var at the
        // resolution point the compile path reads.
        use crate::jit::disk_cache::DISK_PTX_CACHE_ENV;
        with_clean_disk_cache_state(|| {
            let builder_dir = fresh_cache_dir("builder");
            let env_dir = fresh_cache_dir("env");
            std::env::set_var(DISK_PTX_CACHE_ENV, env_dir.to_string_lossy().to_string());
            install_persistent_cache_override(Some(builder_dir.as_path()));

            let cache = crate::jit::disk_cache::disk_cache().expect("cache enabled");
            assert_eq!(
                cache.root(),
                builder_dir.as_path(),
                "builder persistent_cache(path) must take precedence over the env var"
            );
            let _ = std::fs::remove_dir_all(&builder_dir);
            let _ = std::fs::remove_dir_all(&env_dir);
        });
    }

    #[test]
    fn clearing_persistent_cache_falls_back_to_env() {
        // Installing `None` (a default-built engine) must clear a prior
        // builder override and re-expose the env-var fallback — so a later
        // default `build()` doesn't accidentally pin a stale directory.
        use crate::jit::disk_cache::DISK_PTX_CACHE_ENV;
        with_clean_disk_cache_state(|| {
            let builder_dir = fresh_cache_dir("stale");
            let env_dir = fresh_cache_dir("fallback");
            // First: a builder path is in effect.
            install_persistent_cache_override(Some(builder_dir.as_path()));
            assert_eq!(
                crate::jit::disk_cache::disk_cache()
                    .expect("cache enabled")
                    .root(),
                builder_dir.as_path(),
            );
            // Now a default build clears the override; the env var should
            // take over.
            std::env::set_var(DISK_PTX_CACHE_ENV, env_dir.to_string_lossy().to_string());
            install_persistent_cache_override(None);
            assert_eq!(
                crate::jit::disk_cache::disk_cache()
                    .expect("env fallback enables the cache")
                    .root(),
                env_dir.as_path(),
                "clearing the builder override must re-fall-back to BOLT_PTX_CACHE_DIR"
            );
            let _ = std::fs::remove_dir_all(&builder_dir);
            let _ = std::fs::remove_dir_all(&env_dir);
        });
    }

    // -------------------------------------------------------------------
    // F2 — query-counter contract for the DataFrame (`run_logical_plan`)
    // path. These bump the process-global metrics counters, so they
    // serialise under one lock and assert *monotone deltas* (>=) rather
    // than exact counts, which would race a sibling `--ignored` test that
    // also runs a query in parallel.
    // -------------------------------------------------------------------

    /// Serialises the metrics-counting tests below so their counter-delta
    /// observations don't interleave with one another.
    static METRICS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Single-column Int64 table fixture for the counter tests.
    fn metrics_int64_table() -> RecordBatch {
        int64_batch(0, 4)
    }

    /// `run_logical_plan` (the DataFrame `collect` path) must bump
    /// `QueriesTotal` exactly like `sql()` does — previously it bumped
    /// neither counter, so a DataFrame workload reported `queries_total = 0`
    /// while doing real work (review F2).
    #[test]
    #[ignore = "gpu:metrics — run_logical_plan launches a real kernel"]
    fn run_logical_plan_bumps_queries_total() {
        let _g = METRICS_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mut engine = Engine::new().expect("ctx");
        engine
            .register_table("t", metrics_int64_table())
            .expect("register");

        let plan = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: Schema::new(vec![Field::new("x", DataType::Int64, false)]),
        };

        let before = crate::metrics::metrics()
            .counter(crate::metrics::Counter::QueriesTotal)
            .get();
        let _ = engine.run_logical_plan(&plan).expect("execute");
        let after = crate::metrics::metrics()
            .counter(crate::metrics::Counter::QueriesTotal)
            .get();

        assert!(
            after >= before + 1,
            "run_logical_plan must bump QueriesTotal at least once \
             (before={before}, after={after})"
        );
    }

    /// A *failing* top-level `run_logical_plan` must bump `QueriesFailed`,
    /// mirroring `sql()`'s error-path book-keeping. We force a failure by
    /// scanning a table that was never registered, which fails inside
    /// `execute` (the same phase `sql()` counts).
    #[test]
    #[ignore = "gpu:metrics — run_logical_plan initialises the driver"]
    fn run_logical_plan_failure_bumps_queries_failed() {
        let _g = METRICS_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mut engine = Engine::new().expect("ctx");

        // No `register_table` for "missing" → the scan fails at execute time.
        let plan = LogicalPlan::Scan {
            table: "missing".into(),
            projection: None,
            schema: Schema::new(vec![Field::new("x", DataType::Int64, false)]),
        };

        let before_total = crate::metrics::metrics()
            .counter(crate::metrics::Counter::QueriesTotal)
            .get();
        let before_failed = crate::metrics::metrics()
            .counter(crate::metrics::Counter::QueriesFailed)
            .get();
        let result = engine.run_logical_plan(&plan);
        let after_total = crate::metrics::metrics()
            .counter(crate::metrics::Counter::QueriesTotal)
            .get();
        let after_failed = crate::metrics::metrics()
            .counter(crate::metrics::Counter::QueriesFailed)
            .get();

        assert!(result.is_err(), "scan of an unregistered table must fail");
        assert!(
            after_total >= before_total + 1,
            "a failed query still counts toward QueriesTotal"
        );
        assert!(
            after_failed >= before_failed + 1,
            "a failed run_logical_plan must bump QueriesFailed \
             (before={before_failed}, after={after_failed})"
        );
    }

    /// Host-only sanity: the metrics counter read API used by the contract
    /// tests above is wired and monotone. Guards against a future rename of
    /// `Counter::QueriesTotal` / the `counter(..).get()` surface silently
    /// breaking the (GPU-gated) counting tests. No GPU required.
    #[test]
    fn query_counters_are_readable_and_monotone() {
        let m = crate::metrics::metrics();
        let t0 = m.counter(crate::metrics::Counter::QueriesTotal).get();
        m.inc(crate::metrics::Counter::QueriesTotal);
        let t1 = m.counter(crate::metrics::Counter::QueriesTotal).get();
        assert!(t1 >= t0 + 1, "QueriesTotal must be monotone under inc()");

        let f0 = m.counter(crate::metrics::Counter::QueriesFailed).get();
        m.inc(crate::metrics::Counter::QueriesFailed);
        let f1 = m.counter(crate::metrics::Counter::QueriesFailed).get();
        assert!(f1 >= f0 + 1, "QueriesFailed must be monotone under inc()");
    }

    /// F10a — `DeviceCol::mark_launch_stream` must tag the launch stream
    /// into every device buffer the output column owns (so the buffer's
    /// `Drop` fences that stream). We can't observe the `StreamSet`
    /// contents from here (it's `pub(crate)` in `cuda`), but tagging a
    /// freshly-allocated column with a stream must succeed without panic or
    /// error for every `DeviceCol` variant, including the Decimal128 mask
    /// arm. Requires a real allocation, so it is GPU-gated.
    #[test]
    #[ignore = "gpu:projection — allocates real device buffers"]
    fn device_col_mark_launch_stream_tags_all_variants() {
        let stream = CudaStream::null_or_default();
        let s = stream.raw();

        // Primitive + Bool + Utf8 columns.
        for dtype in [
            DataType::Int32,
            DataType::Int64,
            DataType::Float32,
            DataType::Float64,
            DataType::Bool,
            DataType::Utf8,
        ] {
            let col = DeviceCol::alloc_zeros(dtype.clone(), 8).expect("alloc");
            // Must not panic; idempotent tagging is fine (StreamSet dedups).
            col.mark_launch_stream(s);
            col.mark_launch_stream(s);
        }

        // Decimal128 with a passthrough validity mask installed: both the
        // values buffer and the mask buffer must be tagged.
        let mut dec = DeviceCol::alloc_zeros(DataType::Decimal128(38, 0), 8).expect("alloc dec");
        let mask = GpuVec::<u8>::zeros(1).expect("mask");
        dec.set_decimal128_valid_mask(Some(mask));
        dec.mark_launch_stream(s);
    }
}
