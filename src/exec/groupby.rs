// SPDX-License-Identifier: Apache-2.0

//! GROUP BY aggregate execution.
//!
//! Single-pass open-addressing GPU hash table:
//!
//!   1. Host inspects the group-by key column to estimate `K` (the hash-table
//!      size, rounded up to a power of two), and validates the key dtype +
//!      no key collides with the `EMPTY_KEY = i64::MIN` sentinel.
//!   2. Allocate keys table (`length K`, initialised to `EMPTY_KEY`) and one
//!      accumulator table per aggregate (`length K`, initialised to that
//!      aggregate's identity).
//!   3. Upload the key column to the GPU as `i64` (Int32 columns are upcast
//!      host-side).
//!   4. Launch the keys kernel (`bolt_groupby_keys`) — one thread per row.
//!      Each row hashes its key and inserts it into the open-addressing table
//!      via an `atom.cas` linear probe.
//!   5. For each aggregate, upload its input column, JIT + launch
//!      `bolt_groupby_agg` against the already-populated keys table and
//!      that aggregate's accumulator table.
//!   6. Download the keys table and each accumulator table; walk slots,
//!      filtering out empty ones, sort by key for deterministic ordering, and
//!      build the output `RecordBatch` matching `aggregate.output_schema`.
//!
//! Scope (v1):
//!   - GROUP BY keys are encoded host-side into i64 before upload. Supported
//!     packings (all LOSSLESS — distinct tuples yield distinct i64 keys, so
//!     the existing single-i64-key kernel needs no changes):
//!       * 1 col Int32   → upcast to i64 (sign-extended).
//!       * 1 col Int64   → as i64.
//!       * 1 col Float32 → `f32::to_bits() as u32 as i64`.
//!       * 1 col Float64 → `f64::to_bits() as i64`.
//!       * 2 cols (Int32, Int32)     → `(a as u64 << 32) | (b as u32 as u64)`.
//!       * 2 cols (Int32, Float32)   → same packing on the bit patterns.
//!       * 2 cols (Float32, Float32) → same packing on the bit patterns.
//!     Anything wider than 64 bits of key material (e.g. 2× Int64, 2× Float64,
//!     3+ columns) returns a "not yet supported" error. The general fallback
//!     (composite hash + host-side per-slot tuple verification) is deferred.
//!   - Float keys: bitwise-equal floats group together. NaN bit patterns are
//!     distinct keys (acceptable for v1; SQL standard NaN-grouping is
//!     implementation-defined). `-0.0` and `+0.0` group SEPARATELY because
//!     their bit patterns differ — documented v1 limitation.
//!   - Aggregates: `SUM`, `MIN`, `MAX`, `COUNT(*)`, `AVG`. `MIN`/`MAX` over
//!     float inputs are rejected (would need float-CAS loops on sm_70).
//!   - Aggregate inputs must be bare column references (mirrors
//!     `aggregate.rs`'s scalar path); the host fetches them straight from the
//!     input `RecordBatch`.
//!   - The `pre` kernel (filter / projection feeding the aggregate) is not
//!     yet supported here; the caller should run the scalar path or extend
//!     this module to materialise its outputs first.
//!   - EMPTY_KEY sentinel: a single-Int64 key column whose value equals
//!     `i64::MIN` is rejected. For multi-column packed keys, a packed value
//!     of `0x8000_0000_0000_0000` is unlikely to clash with a real composite
//!     tuple but CAN — accepted risk for v1; the same validation rejects it.

use std::collections::HashSet;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
};
use bytemuck::Pod;

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{grid_x_for, CudaStream};
use crate::exec::n_rows_to_u32;
use crate::jit::agg_kernels::ReduceOp;
// Stage C: the `_with_validity` variants in `crate::jit::hash_kernels`
// (`compile_groupby_agg_kernel_with_validity`,
// `compile_groupby_keys_kernel_with_validity`) are wired up and available;
// today this executor keeps the H1 host-strip-at-call-site pattern, which
// is correct for NULL keys (via `key_valid` from `groupby_valid::pack_keys`)
// and for NULL values (the source path here reads a `RecordBatch` and the
// classic groupby rejects EMPTY_KEY collisions early). Switching to the
// native GPU validity path is a performance follow-up — the host-strip
// remains correct.
use crate::jit::hash_kernels::{
    compile_groupby_agg_kernel, compile_groupby_keys_kernel, groupby_block_size,
    AGG_KERNEL_ENTRY, KEYS_KERNEL_ENTRY,
};
use crate::jit::CudaModule;
use crate::plan::logical_plan::{
    sum_output_dtype, AggregateExpr, DataType, Expr, Field, Schema,
};
use crate::plan::physical_plan::{AggregateSpec, ColumnIO, PhysicalPlan};

/// Empty-slot sentinel; mirrors the literal baked into the keys kernel.
const EMPTY_KEY: i64 = i64::MIN;

// ---------------------------------------------------------------------------
// Stage-3 pinned D2H helpers.
//
// Each accumulator-download site previously called `gpu_vec.to_vec()?`,
// which uses a synchronous `cuMemcpyDtoH_v2`. These helpers swap that for
// the pinned `to_pinned_async` + `stream.synchronize()` + host-host copy
// sequence, matching the Stage-3 pattern in `aggregate.rs::reduce_gpu_vec`.
//
// They are monomorphised on the element type so they can land directly
// in an `AccDownload` variant without further casts. The trailing
// `as_slice().to_vec()` is the one unavoidable host-host copy — Arrow
// arrays cannot be built directly on top of a `PinnedHostBuffer`.
// ---------------------------------------------------------------------------

fn download_pinned_i32(v: &GpuVec<i32>, stream: &CudaStream) -> BoltResult<Vec<i32>> {
    let pinned = v.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    Ok(pinned.as_slice().to_vec())
}

fn download_pinned_i64(v: &GpuVec<i64>, stream: &CudaStream) -> BoltResult<Vec<i64>> {
    let pinned = v.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    Ok(pinned.as_slice().to_vec())
}

fn download_pinned_f32(v: &GpuVec<f32>, stream: &CudaStream) -> BoltResult<Vec<f32>> {
    let pinned = v.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    Ok(pinned.as_slice().to_vec())
}

fn download_pinned_f64(v: &GpuVec<f64>, stream: &CudaStream) -> BoltResult<Vec<f64>> {
    let pinned = v.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    Ok(pinned.as_slice().to_vec())
}

/// Execute a GROUP BY aggregate plan against a host-side `RecordBatch`.
///
/// `plan` must be `PhysicalPlan::Aggregate` with non-empty `group_by`.
/// Supports single-column (Int32/Int64/Float32/Float64) and a limited set of
/// 2-column packings whose combined width fits in 64 bits; see module docs.
pub fn execute_groupby(
    plan: &PhysicalPlan,
    table_batch: &RecordBatch,
) -> BoltResult<RecordBatch> {
    // Layered fast-paths. Each `try_execute` returns `Some(_)` only if the
    // query's shape matches that path's preconditions; misses fall through
    // to the next. See docs/GROUPBY_PERF.md for the policy + cardinality
    // breakdown.
    //
    //   Tier-1 single-SUM      — single Int32 key, ONE SUM(Float64), small n_groups
    //   Tier-1 multi-SUM       — single Int32 key, 1..=4 SUMs(Float64), small n_groups
    //   Tier-1 AVG             — single Int32 key, 1..=4 AVGs(Float64), small n_groups
    //   Tier-2 hash-partitioned — single Int32 key, ONE SUM(Float64), large n_groups
    //   GlobalAtomic (below)   — everything else
    if let Some(result) =
        crate::exec::groupby_shmem_exec::try_execute(plan, table_batch)
    {
        return result;
    }
    if let Some(result) =
        crate::exec::groupby_shmem_multi_exec::try_execute(plan, table_batch)
    {
        return result;
    }
    if let Some(result) =
        crate::exec::groupby_shmem_avg_exec::try_execute(plan, table_batch)
    {
        return result;
    }
    if let Some(result) =
        crate::exec::groupby_tier2_exec::try_execute(plan, table_batch)
    {
        return result;
    }
    // Multi-SUM Tier-2: enabled with `MULTI_SUM_MIN_GROUPS = 100_000` floor
    // in the executor itself. Below 100K groups the global-atomic baseline
    // wins (q2 / 10K groups regressed 444 ms → 1.05 s when this path was
    // unconditional); the gate now lets q2 fall through cleanly while
    // capturing future workloads with more groups.
    if let Some(result) =
        crate::exec::groupby_tier2_multi_exec::try_execute(plan, table_batch)
    {
        return result;
    }
    // Two-key Tier-2: enabled now that `partition_reduce_kernel_i64`
    // replaces the host-HashMap pass-2 (Tier 2.1 for two-key).
    if let Some(result) =
        crate::exec::groupby_tier2_twokey_exec::try_execute(plan, table_batch)
    {
        return result;
    }
    // Two-key MULTI-aggregate Tier-2.1: `SELECT a, b, SUM(v1), SUM(v2)
    // FROM x GROUP BY a, b` — combines i64 partitioning with
    // multi-value reduce. Two-key single-SUM falls through to the line
    // above first.
    if let Some(result) =
        crate::exec::groupby_tier2_twokey_multi_exec::try_execute(plan, table_batch)
    {
        return result;
    }
    // AVG-at-Tier-2.1: SUM (via multi-SUM reduce) + COUNT (via count
    // reduce) → divide host-side. High-cardinality AVG over Float64.
    if let Some(result) =
        crate::exec::groupby_tier2_avg_exec::try_execute(plan, table_batch)
    {
        return result;
    }
    // Two-key multi-AVG Tier-2.1: `SELECT a, b, AVG(v1), AVG(v2), ...
    // FROM x GROUP BY a, b`. Same shape as the single-key AVG path but
    // with i64-packed (Int32, Int32) keys.
    if let Some(result) =
        crate::exec::groupby_tier2_twokey_avg_exec::try_execute(plan, table_batch)
    {
        return result;
    }
    // COUNT(*) at Tier-2.1: high-cardinality `SELECT k, COUNT(*) FROM x
    // GROUP BY k`. Reuses partition + scatter; one COUNT reduce launch.
    if let Some(result) =
        crate::exec::groupby_tier2_count_exec::try_execute(plan, table_batch)
    {
        return result;
    }
    // Two-key COUNT(*) Tier-2.1: `SELECT a, b, COUNT(*) FROM x GROUP BY
    // a, b`. Same shape as the single-key COUNT path but with i64-packed
    // (Int32, Int32) keys.
    if let Some(result) =
        crate::exec::groupby_tier2_twokey_count_exec::try_execute(plan, table_batch)
    {
        return result;
    }
    // Two-key integer MIN/MAX at Tier-2.1: `SELECT a, b, {MIN,MAX}(v)
    // FROM x GROUP BY a, b` with Int32 / Int64 value column. Routes
    // through partition_reduce_kernel_minmax_i64. Must come before the
    // single-key minmax path so the two-key shape isn't mishandled.
    if let Some(result) =
        crate::exec::groupby_tier2_twokey_minmax_exec::try_execute(plan, table_batch)
    {
        return result;
    }
    // Two-key float MIN/MAX at Tier-2.1: same shape, Float64 value
    // column, CAS-loop kernel (partition_reduce_kernel_minmax_float_i64).
    if let Some(result) =
        crate::exec::groupby_tier2_twokey_minmax_float_exec::try_execute(plan, table_batch)
    {
        return result;
    }
    // MIN/MAX at Tier-2.1: high-cardinality integer MIN/MAX. Float
    // MIN/MAX is deferred — needs a CAS-loop kernel and no workload
    // demands it yet.
    if let Some(result) =
        crate::exec::groupby_tier2_minmax_exec::try_execute(plan, table_batch)
    {
        return result;
    }
    // Float MIN/MAX: routes through partition_reduce_kernel_minmax_float
    // (CAS-loop kernel). Integer MIN/MAX above catches first; this
    // handles the float-value-column path.
    if let Some(result) =
        crate::exec::groupby_tier2_minmax_float_exec::try_execute(plan, table_batch)
    {
        return result;
    }
    // Tier-1 COUNT(*): low-cardinality COUNT GROUP BY.
    if let Some(result) =
        crate::exec::groupby_shmem_count_exec::try_execute(plan, table_batch)
    {
        return result;
    }
    // Tier-1 MIN/MAX: low-cardinality integer MIN/MAX. Float MIN/MAX
    // is deferred — needs a CAS-loop kernel.
    if let Some(result) =
        crate::exec::groupby_shmem_minmax_exec::try_execute(plan, table_batch)
    {
        return result;
    }

    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        other => {
            return Err(BoltError::Other(format!(
                "execute_groupby: expected Aggregate plan, got {:?}",
                std::mem::discriminant(other)
            )))
        }
    };

    if aggregate.group_by.is_empty() {
        return Err(BoltError::Other(
            "execute_groupby: aggregate has no GROUP BY columns; use execute_aggregate".into(),
        ));
    }
    if pre.is_some() {
        return Err(BoltError::Other(
            "GROUP BY with projection/filter pre-kernel not yet implemented".into(),
        ));
    }

    // Encode all group-by columns into i64 keys (host-side packing). If the
    // key is too wide to pack losslessly into i64, delegate to the wide-key
    // host-side fallback in `crate::exec::groupby_wide`.
    let packed = match pack_keys(aggregate, table_batch) {
        Ok(p) => p,
        Err(BoltError::Other(msg))
            if msg.contains("> 64 bits") || msg.contains("not yet supported") =>
        {
            return crate::exec::groupby_wide::execute_groupby_wide(plan, table_batch);
        }
        Err(e) => return Err(e),
    };
    let key_components = packed.components;
    let key_valid = packed.key_valid;

    // NULL keys: SQL standard semantics are implementation-defined for whether
    // a NULL key forms its own group. For v1 we drop rows whose key is NULL
    // (matching the simplest behaviour: NULL keys are not a group). The
    // alternative — synthesise an explicit "NULL group" — would require
    // either a reserved sentinel collision check (we already use i64::MIN
    // for that purpose) or a separate code path; the dropped-rows approach
    // is the conservative-and-correct first cut.
    let host_keys: Vec<i64> = match &key_valid {
        Some(mask) => packed
            .keys_i64
            .iter()
            .zip(mask.iter())
            .filter_map(|(k, &keep)| if keep { Some(*k) } else { None })
            .collect(),
        None => packed.keys_i64,
    };

    // Check for EMPTY_KEY sentinel collision. If any encoded key equals
    // i64::MIN (most commonly: a Float64 column containing -0.0), the
    // sentinel-based classic kernel can't tell that row from an empty slot.
    // Fall back to the sentinel-free valid-flag variant.
    if host_keys.iter().any(|&k| k == EMPTY_KEY) {
        return crate::exec::groupby_valid::execute_groupby_valid(plan, table_batch);
    }

    let n_rows = host_keys.len();

    // Estimate K from the unique-key count (host scan).
    let n_unique = unique_count(&host_keys);
    let k = next_pow2((n_unique.saturating_mul(2)).saturating_add(16)).max(64);
    let k_u32 = u32::try_from(k).map_err(|_| {
        BoltError::Other(format!(
            "GROUP BY hash table size {} exceeds u32::MAX",
            k
        ))
    })?;

    // Stage-3: mint a per-call stream up front so the host->device
    // uploads, kernel launches, and the final D2Hs share a single
    // ordering domain — the driver can then overlap kernel work with
    // any unrelated activity on the NULL stream. Falls back to NULL if
    // stream creation fails (functionally identical, just no overlap).
    let stream = CudaStream::null_or_default();

    // Build the keys table on the host (filled with EMPTY_KEY) and upload it.
    //
    // Stage-3: H2D upload is async on `stream`. The keys-kernel launch
    // (queued on the same stream below) depends on this copy, so the
    // kernel is automatically ordered after the upload. We do NOT
    // pinned-source these uploads — they're one-shot at executor entry
    // and the pinned pool would only see ~k * 8 bytes of churn, which
    // isn't worth the host-side allocator pressure for a single query.
    let host_keys_init: Vec<i64> = vec![EMPTY_KEY; k];
    let mut keys_table = GpuVec::<i64>::from_slice_async(&host_keys_init, stream.raw())?;
    let key_col_gpu = GpuVec::<i64>::from_slice_async(&host_keys, stream.raw())?;

    // Launch the keys-only kernel.
    launch_keys_kernel(&key_col_gpu, &mut keys_table, n_rows, k_u32, &stream)?;

    // For each aggregate, prepare its accumulator and launch the agg kernel.
    // We collect (input_dtype_for_acc, downloaded acc vector as a typed enum)
    // per aggregate so that the host-side post-processing knows what to read.
    let mut acc_results: Vec<AccDownload> = Vec::with_capacity(aggregate.aggregates.len());
    for agg in &aggregate.aggregates {
        let acc = run_one_aggregate(
            agg,
            &aggregate.inputs,
            &key_col_gpu,
            &keys_table,
            table_batch,
            n_rows,
            k,
            k_u32,
            &stream,
            key_valid.as_deref(),
        )?;
        acc_results.push(acc);
    }

    // Stage-3: download the keys table through a pinned host buffer
    // so the driver can DMA directly. Sync once on this stream, then
    // hand the data off to host-side group assembly.
    let host_keys_table: Vec<i64> = {
        let pinned = keys_table.to_pinned_async(stream.raw())?;
        stream.synchronize()?;
        pinned.as_slice().to_vec()
    };
    drop(keys_table);
    drop(key_col_gpu);

    // Walk the keys table: every non-empty slot is a group. Build a list of
    // `(key, slot)` and sort by key for deterministic output ordering.
    let mut groups: Vec<(i64, usize)> = host_keys_table
        .iter()
        .enumerate()
        .filter_map(|(slot, &k)| if k == EMPTY_KEY { None } else { Some((k, slot)) })
        .collect();
    groups.sort_unstable_by_key(|(k, _)| *k);

    // Assemble the output RecordBatch.
    let n_groups = groups.len();
    let m_keys = key_components.len();
    let mut arrays: Vec<ArrayRef> =
        Vec::with_capacity(m_keys + aggregate.aggregates.len());

    // Columns 0..M: one per group-by column, decoded from the packed i64 key.
    let key_arrays = build_key_arrays(&groups, &key_components)?;
    for arr in key_arrays {
        arrays.push(arr);
    }

    // Columns M..M+N: one per aggregate, taken from the corresponding
    // accumulator.
    for (i, agg) in aggregate.aggregates.iter().enumerate() {
        let out_field =
            aggregate.output_schema.fields.get(m_keys + i).ok_or_else(|| {
                BoltError::Other(format!(
                    "execute_groupby: output_schema missing field for aggregate index {}",
                    i
                ))
            })?;
        let arr = build_agg_array(agg, out_field, &acc_results[i], &groups, n_groups)?;
        arrays.push(arr);
    }

    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
        BoltError::Other(format!("failed to build GROUP BY RecordBatch: {e}"))
    })
}

// ---------------------------------------------------------------------------
// Key column extraction, packing, and decoding.
// ---------------------------------------------------------------------------

/// Number of bits a column's encoded value occupies inside a packed i64 key.
/// Anything wider than 64 bits total is rejected (composite-hash fallback is
/// not implemented in v1).
fn key_bit_width(dtype: DataType) -> BoltResult<u32> {
    match dtype {
        DataType::Int32 | DataType::Float32 => Ok(32),
        DataType::Int64 | DataType::Float64 => Ok(64),
        DataType::Bool | DataType::Utf8 => Err(BoltError::Type(format!(
            "GROUP BY key dtype {:?} not supported in v1",
            dtype
        ))),
    }
}

/// Per-group-by-column metadata: the original dtype and the bit offset at
/// which this column's value is packed inside the i64 key (low = 0).
#[derive(Debug, Clone)]
struct KeyComponent {
    /// Column name (used for error messages and field naming).
    // used by: pack_keys_two_int32 test (asserts component identity)
    #[allow(dead_code)]
    name: String,
    /// Original dtype as declared by the plan (drives encode/decode).
    original_dtype: DataType,
    /// Bit position within the i64 key — low = 0, high = 32 for the
    /// two-column pack. Single-column keys always use offset 0.
    bit_offset: u32,
}

/// Result of `pack_keys`: the per-row encoded i64 key column plus enough
/// metadata to decode each unique slot back into its constituent columns.
#[derive(Debug)]
struct PackedKeys {
    /// i64-encoded keys ready to upload, one entry per input row. NULL-key
    /// rows (per `key_valid`) carry an undefined bit pattern at their slot
    /// because the per-column loader propagates garbage values; callers MUST
    /// filter via `key_valid` before forwarding to the GPU.
    keys_i64: Vec<i64>,
    /// Per-column dtype + original ordinal in `aggregate.inputs`, in pack order.
    components: Vec<KeyComponent>,
    /// Per-row keep mask: `true` iff every GROUP BY key column is non-null at
    /// that row. `None` means "all key columns have zero nulls" — the fast
    /// path where no filtering is required.
    key_valid: Option<Vec<bool>>,
}

/// Read the per-row value of one group-by column out of `batch` and return a
/// `Vec<u64>` of its low-significance bit pattern, zero-extended into u64,
/// plus an optional per-row validity mask (`Some(false)` at NULL positions).
///
/// For Int32 this is the unsigned 32-bit representation (i.e. sign bits are
/// dropped); for Int64 it's a straight bitcast; for floats it's `to_bits`.
/// The result is ready to be OR'd into a packed key at the right shift.
///
/// When the input array has no NULLs the returned mask is `None` (saves a
/// per-row allocation in the common path). Callers must treat `None` as
/// "all valid". For NULL positions the raw u64 bit pattern is whatever
/// happens to live in the values buffer — callers MUST drop those rows
/// before they reach the GPU keys kernel, otherwise garbage bytes from the
/// NULL slots will form a fake group.
fn load_key_column_bits(
    key_io: &ColumnIO,
    batch: &RecordBatch,
) -> BoltResult<(Vec<u64>, Option<Vec<bool>>)> {
    let idx = batch.schema().index_of(&key_io.name).map_err(|e| {
        BoltError::Plan(format!(
            "GROUP BY key '{}' not present in table batch: {}",
            key_io.name, e
        ))
    })?;
    let arr = batch.column(idx);

    // Sanity-check the dtype matches what the plan promised.
    let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
    if arr_dtype != key_io.dtype {
        return Err(BoltError::Type(format!(
            "GROUP BY key '{}' dtype mismatch: plan says {:?}, batch has {:?}",
            key_io.name, key_io.dtype, arr_dtype
        )));
    }

    let null_mask: Option<Vec<bool>> = if arr.null_count() == 0 {
        None
    } else {
        Some((0..arr.len()).map(|i| !arr.is_null(i)).collect())
    };

    let bits = match key_io.dtype {
        DataType::Int32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(&key_io.name, "Int32"))?;
            pa.values().iter().map(|&v| v as u32 as u64).collect()
        }
        DataType::Int64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&key_io.name, "Int64"))?;
            pa.values().iter().map(|&v| v as u64).collect()
        }
        DataType::Float32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&key_io.name, "Float32"))?;
            pa.values().iter().map(|&v| v.to_bits() as u64).collect()
        }
        DataType::Float64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&key_io.name, "Float64"))?;
            pa.values().iter().map(|&v| v.to_bits()).collect()
        }
        DataType::Bool | DataType::Utf8 => {
            return Err(BoltError::Type(format!(
                "GROUP BY key dtype {:?} not supported in v1",
                key_io.dtype
            )))
        }
    };
    Ok((bits, null_mask))
}

/// Encode each group-by column (in `aggregate.group_by` order) into a single
/// i64 key per row. Returns the encoded keys plus per-column metadata for the
/// reverse trip in `build_key_arrays`.
///
/// Supported packings — all LOSSLESS, so distinct tuples produce distinct
/// i64 keys and the existing single-i64-key kernel needs no changes:
///
/// | columns                      | encoding                                   |
/// |------------------------------|--------------------------------------------|
/// | (Int32,)                     | `v as u32 as u64 as i64` (sign-extended)   |
/// | (Int64,)                     | `v as i64` (identity)                      |
/// | (Float32,)                   | `f.to_bits() as u64 as i64`                |
/// | (Float64,)                   | `f.to_bits() as i64`                       |
/// | (Int32, Int32)               | `(a << 32) \| b`, b is u32 zero-extended    |
/// | (Int32, Float32)             | same, using bit patterns                   |
/// | (Float32, Float32)           | same, using bit patterns                   |
///
/// Anything wider than 64 bits of combined key material (e.g. two Int64s,
/// two Float64s, or three or more columns) returns a "not yet supported"
/// error — the composite-hash + host-side verification fallback is deferred.
fn pack_keys(
    aggregate: &AggregateSpec,
    batch: &RecordBatch,
) -> BoltResult<PackedKeys> {
    if aggregate.group_by.is_empty() {
        return Err(BoltError::Other(
            "pack_keys: aggregate has no GROUP BY columns".into(),
        ));
    }

    // Resolve every group-by ordinal to a ColumnIO and validate widths.
    let mut col_ios: Vec<&ColumnIO> = Vec::with_capacity(aggregate.group_by.len());
    let mut total_bits: u32 = 0;
    for &ord in &aggregate.group_by {
        let io = aggregate.inputs.get(ord).ok_or_else(|| {
            BoltError::Plan(format!(
                "pack_keys: group_by ordinal {} out of range (only {} inputs)",
                ord,
                aggregate.inputs.len()
            ))
        })?;
        let bits = key_bit_width(io.dtype)?;
        total_bits = total_bits.saturating_add(bits);
        col_ios.push(io);
    }

    if aggregate.group_by.len() > 2 {
        return Err(BoltError::Other(format!(
            "multi-column GROUP BY with > 64 bits of key width not yet supported \
             ({} columns requested)",
            aggregate.group_by.len()
        )));
    }
    if total_bits > 64 {
        return Err(BoltError::Other(format!(
            "multi-column GROUP BY with > 64 bits of key width not yet supported \
             (requested columns total {} bits)",
            total_bits
        )));
    }

    // Build the per-column bit streams + the component metadata.
    //
    // Layout convention (matches the doc comment above):
    //   - For one column: the value occupies the low 32 (or 64) bits.
    //     bit_offset = 0.
    //   - For two columns: the FIRST group-by column occupies the HIGH 32
    //     bits (bit_offset = 32), the second occupies the LOW 32 bits
    //     (bit_offset = 0).
    let mut components: Vec<KeyComponent> =
        Vec::with_capacity(aggregate.group_by.len());
    let mut bit_streams: Vec<Vec<u64>> =
        Vec::with_capacity(aggregate.group_by.len());

    // Accumulate per-key-column null masks into a single combined `key_valid`
    // vector. If every key column has zero nulls we keep `combined_mask` as
    // `None` (the common fast path).
    let mut combined_mask: Option<Vec<bool>> = None;

    if aggregate.group_by.len() == 1 {
        let io = col_ios[0];
        let (bits, mask) = load_key_column_bits(io, batch)?;
        components.push(KeyComponent {
            name: io.name.clone(),
            original_dtype: io.dtype,
            bit_offset: 0,
        });
        if let Some(m) = mask {
            combined_mask = Some(m);
        }
        bit_streams.push(bits);
    } else {
        // len == 2, total_bits <= 64. Both individual widths are <= 32 (else
        // total_bits > 64), so the high column gets bit_offset=32 and the
        // low column gets bit_offset=0.
        let io_hi = col_ios[0];
        let io_lo = col_ios[1];
        let (bits_hi, mask_hi) = load_key_column_bits(io_hi, batch)?;
        let (bits_lo, mask_lo) = load_key_column_bits(io_lo, batch)?;
        if bits_hi.len() != bits_lo.len() {
            return Err(BoltError::Other(format!(
                "pack_keys: group-by columns '{}' and '{}' have different row \
                 counts ({} vs {})",
                io_hi.name,
                io_lo.name,
                bits_hi.len(),
                bits_lo.len()
            )));
        }
        components.push(KeyComponent {
            name: io_hi.name.clone(),
            original_dtype: io_hi.dtype,
            bit_offset: 32,
        });
        components.push(KeyComponent {
            name: io_lo.name.clone(),
            original_dtype: io_lo.dtype,
            bit_offset: 0,
        });
        bit_streams.push(bits_hi);
        bit_streams.push(bits_lo);
        // Combine: a row is valid iff every key column is non-null at that row.
        combined_mask = and_masks(mask_hi, mask_lo);
    }

    // OR each column's bit stream into the final i64 column at the right
    // shift. The first stream determines the row count; we already verified
    // lengths match above.
    let n_rows = bit_streams[0].len();
    let mut keys_i64: Vec<i64> = vec![0i64; n_rows];
    for (comp, stream) in components.iter().zip(bit_streams.iter()) {
        // For single-column keys the per-column width may be 64, in which
        // case we MUST avoid a shift by 64 — only the bit_offset == 0 case
        // is relevant for 64-bit widths and the shift is a no-op.
        let shift = comp.bit_offset;
        let bit_width = key_bit_width(comp.original_dtype)?;
        debug_assert!(
            shift + bit_width <= 64,
            "pack_keys: shift+width must fit i64"
        );
        // Use `wrapping_shl` so that a future regression where `shift == 64`
        // (e.g. a 32-bit dtype appended after a 64-bit field) produces a
        // deterministic 0 instead of triggering UB on a bare `<<` overshift.
        // For the supported case (`shift` in 0..=32 with 32-bit widths, or
        // `shift == 0` with a 64-bit width) this is identical to `raw << shift`.
        for (i, &raw) in stream.iter().enumerate() {
            let packed = raw.wrapping_shl(shift) as i64;
            // Bitwise OR (preserving any bits already set by earlier streams).
            keys_i64[i] = ((keys_i64[i] as u64) | (packed as u64)) as i64;
        }
    }

    Ok(PackedKeys {
        keys_i64,
        components,
        key_valid: combined_mask,
    })
}

/// Logical AND of two optional masks. `None` denotes "all-true" (no nulls
/// from that side). Returns `None` only when both inputs are `None`.
fn and_masks(a: Option<Vec<bool>>, b: Option<Vec<bool>>) -> Option<Vec<bool>> {
    match (a, b) {
        (None, None) => None,
        (Some(m), None) | (None, Some(m)) => Some(m),
        (Some(ma), Some(mb)) => {
            debug_assert_eq!(ma.len(), mb.len(), "and_masks: length mismatch");
            Some(ma.iter().zip(mb.iter()).map(|(x, y)| *x && *y).collect())
        }
    }
}

/// Reverse of `pack_keys` for a single packed i64 key: extract each
/// component value as a typed `KeyValue` in the same order the components
/// were packed (i.e. matching `aggregate.group_by` order).
fn decode_key(packed: i64, components: &[KeyComponent]) -> Vec<KeyValue> {
    let mut out: Vec<KeyValue> = Vec::with_capacity(components.len());
    let u = packed as u64;
    for comp in components {
        let raw = match comp.original_dtype {
            DataType::Int32 | DataType::Float32 => {
                // 32-bit field at `bit_offset`. Mask to u32.
                ((u >> comp.bit_offset) & 0xFFFF_FFFFu64) as u64
            }
            DataType::Int64 | DataType::Float64 => {
                // 64-bit field: there can be at most one component and
                // bit_offset is always 0.
                u
            }
            DataType::Bool | DataType::Utf8 => {
                // pack_keys would have already rejected these dtypes — we
                // can't reach this branch from `execute_groupby`, but keep
                // the match exhaustive and return a 0 placeholder.
                0
            }
        };
        let val = match comp.original_dtype {
            DataType::Int32 => KeyValue::I32(raw as u32 as i32),
            DataType::Int64 => KeyValue::I64(raw as i64),
            DataType::Float32 => KeyValue::F32(f32::from_bits(raw as u32)),
            DataType::Float64 => KeyValue::F64(f64::from_bits(raw)),
            DataType::Bool | DataType::Utf8 => KeyValue::I64(0),
        };
        out.push(val);
    }
    out
}

/// Typed per-column key value as decoded from a packed i64.
#[derive(Debug, Clone, Copy, PartialEq)]
enum KeyValue {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

/// Count of distinct values in `keys`. Linear, O(n) using a HashSet.
fn unique_count(keys: &[i64]) -> usize {
    let mut set: HashSet<i64> = HashSet::with_capacity(keys.len().min(1 << 20));
    for &k in keys {
        set.insert(k);
    }
    set.len()
}

/// Smallest power of two >= `n` (saturating to `usize::MAX`'s previous pow-2
/// on overflow).
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p: usize = 1;
    while p < n {
        match p.checked_mul(2) {
            Some(v) => p = v,
            None => return p, // already as large as we can express
        }
    }
    p
}

// ---------------------------------------------------------------------------
// Kernel launches.
// ---------------------------------------------------------------------------

/// Launch the keys-only kernel.
fn launch_keys_kernel(
    group_col: &GpuVec<i64>,
    keys_table: &mut GpuVec<i64>,
    n_rows: usize,
    k_u32: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    if n_rows == 0 {
        // Nothing to insert; the empty keys table is already correct.
        return Ok(());
    }

    let ptx = compile_groupby_keys_kernel()?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(KEYS_KERNEL_ENTRY)?;

    let mut group_ptr: CUdeviceptr = group_col.device_ptr();
    let mut keys_ptr: CUdeviceptr = keys_table.device_ptr();
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let mut k_param: u32 = k_u32;

    let mut params: [*mut c_void; 4] = [
        &mut group_ptr as *mut CUdeviceptr as *mut c_void,
        &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
        &mut k_param as *mut u32 as *mut c_void,
    ];

    let block = groupby_block_size();
    let grid_x = grid_x_for(n_rows_u32, block);

    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x,
            1,
            1,
            block,
            1,
            1,
            0,
            stream.raw(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }
    stream.synchronize()?;
    let _ = (group_ptr, keys_ptr);
    Ok(())
}

/// Launch one aggregate-update kernel for a (typed) input column and a
/// (typed) accumulator buffer.
fn launch_agg_kernel<T: Pod>(
    op: ReduceOp,
    input_dtype: DataType,
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    input_col: &GpuVec<T>,
    acc_table: &mut GpuVec<T>,
    n_rows: usize,
    k_u32: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    if n_rows == 0 {
        return Ok(());
    }

    // MIN/MAX over floats need a CAS loop (no native atom.min.fXX on sm_70);
    // route those to the float_atomics codegen. Everything else uses the
    // standard integer-atomic agg kernel.
    let ptx = match (op, input_dtype) {
        (ReduceOp::Min, DataType::Float32)
        | (ReduceOp::Max, DataType::Float32)
        | (ReduceOp::Min, DataType::Float64)
        | (ReduceOp::Max, DataType::Float64) => {
            crate::jit::float_atomics::compile_groupby_float_atomic_kernel(op, input_dtype)?
        }
        _ => compile_groupby_agg_kernel(op, input_dtype)?,
    };
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(AGG_KERNEL_ENTRY)?;

    let mut group_ptr: CUdeviceptr = group_col.device_ptr();
    let mut keys_ptr: CUdeviceptr = keys_table.device_ptr();
    let mut input_ptr: CUdeviceptr = input_col.device_ptr();
    let mut acc_ptr: CUdeviceptr = acc_table.device_ptr();
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let mut k_param: u32 = k_u32;

    let mut params: [*mut c_void; 6] = [
        &mut group_ptr as *mut CUdeviceptr as *mut c_void,
        &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut input_ptr as *mut CUdeviceptr as *mut c_void,
        &mut acc_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
        &mut k_param as *mut u32 as *mut c_void,
    ];

    let block = groupby_block_size();
    let grid_x = grid_x_for(n_rows_u32, block);

    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x,
            1,
            1,
            block,
            1,
            1,
            0,
            stream.raw(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }
    stream.synchronize()?;
    let _ = (group_ptr, keys_ptr, input_ptr, acc_ptr);
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-aggregate plumbing: prepare buffer, launch kernel(s), download result.
// ---------------------------------------------------------------------------

/// Downloaded accumulator table for a single aggregate. The dtype variant
/// tells the host-side assembler how to read each slot.
enum AccDownload {
    /// Int32 SUM/MIN/MAX result column (length `k`).
    I32(Vec<i32>),
    /// Int64 SUM/MIN/MAX/COUNT result column (length `k`).
    I64(Vec<i64>),
    /// Float32 SUM result column (length `k`).
    F32(Vec<f32>),
    /// Float64 SUM result column (length `k`).
    F64(Vec<f64>),
    /// AVG: a SUM accumulator (downloaded as f64) and a COUNT accumulator
    /// (downloaded as i64), both length `k`.
    Avg { sum: Vec<f64>, count: Vec<i64> },
}

/// Compile + launch one aggregate kernel (or, for `Avg`, two), download its
/// accumulator table(s), and return the result.
///
/// `key_valid` is the per-row keep mask produced by `pack_keys` over the
/// ORIGINAL (pre-filter) row indices — `None` means no key column has nulls.
/// When the value column itself has NULLs we logically AND the masks and
/// upload a fresh, per-aggregate filtered key column so that the GPU sees
/// only (non-NULL key, non-NULL value) pairs. `n_rows` is the post-key-filter
/// row count: it equals `group_col`'s length when we end up reusing the
/// shared key column.
#[allow(clippy::too_many_arguments)]
fn run_one_aggregate(
    agg: &AggregateExpr,
    inputs: &[ColumnIO],
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    batch: &RecordBatch,
    n_rows: usize,
    k: usize,
    k_u32: u32,
    stream: &CudaStream,
    key_valid: Option<&[bool]>,
) -> BoltResult<AccDownload> {
    match agg {
        AggregateExpr::Sum(expr)
        | AggregateExpr::Min(expr)
        | AggregateExpr::Max(expr) => {
            let op = ReduceOp::from_agg(agg)?;
            let col_name = bare_column_name(expr)?;
            let col_io = resolve_input(inputs, col_name)?;
            run_typed_agg(
                op, col_io, group_col, keys_table, batch, n_rows, k, k_u32, stream,
                key_valid,
            )
        }

        AggregateExpr::Count(expr) => {
            // COUNT(col) excludes NULL inputs; COUNT(*) (an expression that
            // doesn't resolve to a column) counts surviving (post-key-filter)
            // rows. Either way we synthesise an all-ones column over the
            // post-filter rows; the only difference is whether the value-NULL
            // mask is applied. Stage-3: async H2D + pinned D2H.
            let value_valid: Option<Vec<bool>> = match bare_column_name(expr)
                .ok()
                .and_then(|name| resolve_input(inputs, name).ok())
            {
                Some(col_io) => column_null_mask(col_io, batch)?,
                None => None,
            };

            let filtered = prepare_filtered_keys(group_col, n_rows, key_valid, value_valid.as_deref())?;
            let count_n_rows = filtered.n_rows();

            let ones: Vec<i64> = vec![1i64; count_n_rows];
            let input_gpu = GpuVec::<i64>::from_slice_async(&ones, stream.raw())?;
            let identity_init: Vec<i64> = vec![0i64; k];
            let mut acc_table = GpuVec::<i64>::from_slice_async(&identity_init, stream.raw())?;
            launch_agg_kernel::<i64>(
                ReduceOp::Sum,
                DataType::Int64,
                filtered.col(),
                keys_table,
                &input_gpu,
                &mut acc_table,
                count_n_rows,
                k_u32,
                stream,
            )?;
            Ok(AccDownload::I64(download_pinned_i64(&acc_table, stream)?))
        }

        AggregateExpr::Avg(expr) => {
            // AVG = SUM(expr) / COUNT(expr), where COUNT is the non-NULL row
            // count of the value column within each group. SUM in f64 (so we
            // don't worry about int-overflow during accumulation), COUNT in
            // i64. Both kernels share the (key_valid ∧ value_valid) filter so
            // every contribution to the SUM increments the matching COUNT.
            //
            // Stage-3: async H2D for the input + identity tables; pinned
            // D2H for both final accumulators.
            let col_name = bare_column_name(expr)?;
            let col_io = resolve_input(inputs, col_name)?;

            let value_valid = column_null_mask(col_io, batch)?;
            let filtered = prepare_filtered_keys(group_col, n_rows, key_valid, value_valid.as_deref())?;
            let avg_n_rows = filtered.n_rows();

            // --- SUM(expr) cast to f64. We upcast the input host-side and
            //     drop NULL positions in the same step. ---
            let sum_input: Vec<f64> = load_input_column_as_f64_filtered(
                col_io,
                batch,
                key_valid,
                value_valid.as_deref(),
            )?;
            debug_assert_eq!(sum_input.len(), avg_n_rows);
            let input_gpu = GpuVec::<f64>::from_slice_async(&sum_input, stream.raw())?;
            let sum_init: Vec<f64> = vec![0.0f64; k];
            let mut sum_acc = GpuVec::<f64>::from_slice_async(&sum_init, stream.raw())?;
            launch_agg_kernel::<f64>(
                ReduceOp::Sum,
                DataType::Float64,
                filtered.col(),
                keys_table,
                &input_gpu,
                &mut sum_acc,
                avg_n_rows,
                k_u32,
                stream,
            )?;
            let sum_host = download_pinned_f64(&sum_acc, stream)?;

            // --- COUNT(non-null) per group. ---
            let ones: Vec<i64> = vec![1i64; avg_n_rows];
            let count_input = GpuVec::<i64>::from_slice_async(&ones, stream.raw())?;
            let count_init: Vec<i64> = vec![0i64; k];
            let mut count_acc = GpuVec::<i64>::from_slice_async(&count_init, stream.raw())?;
            launch_agg_kernel::<i64>(
                ReduceOp::Sum,
                DataType::Int64,
                filtered.col(),
                keys_table,
                &count_input,
                &mut count_acc,
                avg_n_rows,
                k_u32,
                stream,
            )?;
            let count_host = download_pinned_i64(&count_acc, stream)?;

            Ok(AccDownload::Avg {
                sum: sum_host,
                count: count_host,
            })
        }
    }
}

/// Return the per-row validity mask for `col_io` in `batch`, or `None` if
/// the column has no nulls (saves the per-row allocation in the hot path).
fn column_null_mask(
    col_io: &ColumnIO,
    batch: &RecordBatch,
) -> BoltResult<Option<Vec<bool>>> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        BoltError::Plan(format!(
            "aggregate input '{}' not present in table batch: {}",
            col_io.name, e
        ))
    })?;
    let arr = batch.column(idx);
    if arr.null_count() == 0 {
        Ok(None)
    } else {
        Ok(Some((0..arr.len()).map(|i| !arr.is_null(i)).collect()))
    }
}

/// A filtered key column for a single aggregate launch: either a borrowed
/// view of the shared `group_col` (the common fast path: no value-NULLs
/// shrunk the row set further) or a freshly-uploaded smaller column for the
/// (key_valid AND value_valid) joint mask. The variants paper over Rust's
/// borrow-checker constraints around returning `&GpuVec<i64>` whose lifetime
/// might come from local storage.
enum FilteredKeys<'a> {
    /// Reuse the shared post-key-filter key column.
    Borrowed { group_col: &'a GpuVec<i64>, n_rows: usize },
    /// Freshly-uploaded smaller column applying a value-NULL filter on top
    /// of `key_valid`. The owned vec must live across the kernel launch.
    Owned { group_col: GpuVec<i64>, n_rows: usize },
}

impl<'a> FilteredKeys<'a> {
    fn col(&self) -> &GpuVec<i64> {
        match self {
            FilteredKeys::Borrowed { group_col, .. } => *group_col,
            FilteredKeys::Owned { group_col, .. } => group_col,
        }
    }
    fn n_rows(&self) -> usize {
        match self {
            FilteredKeys::Borrowed { n_rows, .. } | FilteredKeys::Owned { n_rows, .. } => *n_rows,
        }
    }
}

/// Decide whether to reuse the shared `group_col` (when no value-NULL filter
/// shrinks the row set further) or upload a freshly-filtered key column for
/// this aggregate. The shared `group_col` was built from a `host_keys` that
/// already had the `key_valid` rows kept; if `value_valid` is `None` we can
/// reuse it directly. Otherwise we re-download once and refilter against
/// the joint mask, then upload a fresh i64 column.
fn prepare_filtered_keys<'a>(
    group_col: &'a GpuVec<i64>,
    n_rows: usize,
    key_valid: Option<&[bool]>,
    value_valid: Option<&[bool]>,
) -> BoltResult<FilteredKeys<'a>> {
    if value_valid.is_none() {
        return Ok(FilteredKeys::Borrowed { group_col, n_rows });
    }

    // Project `value_valid` (indexed by ORIGINAL row position) through the
    // key filter to align with the post-key-filter `host_keys` list.
    let value_valid_unwrapped = value_valid.expect("checked above");
    let value_valid_filtered: Vec<bool> = match key_valid {
        Some(kv) => kv
            .iter()
            .zip(value_valid_unwrapped.iter())
            .filter_map(|(&k, &v)| if k { Some(v) } else { None })
            .collect(),
        None => value_valid_unwrapped.to_vec(),
    };
    debug_assert_eq!(value_valid_filtered.len(), n_rows);

    let host_keys: Vec<i64> = group_col.to_vec()?;
    debug_assert_eq!(host_keys.len(), n_rows);
    let filtered: Vec<i64> = host_keys
        .iter()
        .zip(value_valid_filtered.iter())
        .filter_map(|(&k, &v)| if v { Some(k) } else { None })
        .collect();
    let filtered_n = filtered.len();
    let owned = GpuVec::<i64>::from_slice(&filtered)?;
    Ok(FilteredKeys::Owned { group_col: owned, n_rows: filtered_n })
}

/// Common path for SUM/MIN/MAX. Uploads the typed input column, allocates a
/// typed accumulator initialised to the op's identity, launches the agg
/// kernel, and downloads the accumulator.
///
/// `key_valid` (from `pack_keys`) and the value column's own validity mask
/// are AND'd together: rows where EITHER is NULL are dropped before upload.
/// This matches the standard SQL semantics — NULL inputs are skipped by
/// SUM/MIN/MAX rather than coerced to 0 / dtype-min / dtype-max (which is
/// what reading the raw `.values()` buffer at NULL positions would do).
#[allow(clippy::too_many_arguments)]
fn run_typed_agg(
    op: ReduceOp,
    col_io: &ColumnIO,
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    batch: &RecordBatch,
    n_rows: usize,
    k: usize,
    k_u32: u32,
    stream: &CudaStream,
    key_valid: Option<&[bool]>,
) -> BoltResult<AccDownload> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        BoltError::Plan(format!(
            "aggregate input '{}' not present in table batch: {}",
            col_io.name, e
        ))
    })?;
    let arr = batch.column(idx);
    let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
    if arr_dtype != col_io.dtype {
        return Err(BoltError::Type(format!(
            "aggregate input '{}' dtype mismatch: plan says {:?}, batch has {:?}",
            col_io.name, col_io.dtype, arr_dtype
        )));
    }

    let value_valid = column_null_mask(col_io, batch)?;
    let filtered = prepare_filtered_keys(group_col, n_rows, key_valid, value_valid.as_deref())?;
    let n = filtered.n_rows();

    match col_io.dtype {
        DataType::Int32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int32"))?;

            // SUM(Int32) widens to Int64 per the single-source-of-truth
            // `crate::plan::logical_plan::sum_output_dtype`: silent i32
            // overflow inside the GPU atomic was the prior bug. The
            // groupby agg kernel (`compile_groupby_agg_kernel`) does not
            // itself sign-extend at load time, so we widen host-side by
            // upcasting each i32 to i64 before upload, allocate an i64
            // accumulator, and request the i64-typed kernel — which then
            // emits `atom.global.add.u64` (PTX has no `.s64` variant of
            // atom.add — `.u64` is bit-identical for two's-complement signed
            // addition). MIN/MAX preserve the input
            // dtype and stay on the i32 path.
            let widened_dtype = sum_output_dtype(DataType::Int32);
            let widen_to_i64 = matches!(op, ReduceOp::Sum) && widened_dtype == DataType::Int64;
            if widen_to_i64 {
                let host: Vec<i64> = collect_filtered_primitive(pa, key_valid, value_valid.as_deref())
                    .into_iter()
                    .map(|v| v as i64)
                    .collect();
                debug_assert_eq!(host.len(), n);
                let input_gpu = GpuVec::<i64>::from_slice_async(&host, stream.raw())?;
                let init: Vec<i64> = vec![identity_i64(op); k];
                let mut acc = GpuVec::<i64>::from_slice_async(&init, stream.raw())?;
                launch_agg_kernel::<i64>(
                    op,
                    DataType::Int64,
                    filtered.col(),
                    keys_table,
                    &input_gpu,
                    &mut acc,
                    n,
                    k_u32,
                    stream,
                )?;
                Ok(AccDownload::I64(download_pinned_i64(&acc, stream)?))
            } else {
                let host: Vec<i32> = collect_filtered_primitive(pa, key_valid, value_valid.as_deref());
                debug_assert_eq!(host.len(), n);
                let input_gpu = GpuVec::<i32>::from_slice_async(&host, stream.raw())?;
                let init: Vec<i32> = vec![identity_i32(op); k];
                let mut acc = GpuVec::<i32>::from_slice_async(&init, stream.raw())?;
                launch_agg_kernel::<i32>(
                    op,
                    DataType::Int32,
                    filtered.col(),
                    keys_table,
                    &input_gpu,
                    &mut acc,
                    n,
                    k_u32,
                    stream,
                )?;
                Ok(AccDownload::I32(download_pinned_i32(&acc, stream)?))
            }
        }
        DataType::Int64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int64"))?;
            let host: Vec<i64> = collect_filtered_primitive(pa, key_valid, value_valid.as_deref());
            debug_assert_eq!(host.len(), n);
            let input_gpu = GpuVec::<i64>::from_slice_async(&host, stream.raw())?;
            let init: Vec<i64> = vec![identity_i64(op); k];
            let mut acc = GpuVec::<i64>::from_slice_async(&init, stream.raw())?;
            launch_agg_kernel::<i64>(
                op,
                DataType::Int64,
                filtered.col(),
                keys_table,
                &input_gpu,
                &mut acc,
                n,
                k_u32,
                stream,
            )?;
            Ok(AccDownload::I64(download_pinned_i64(&acc, stream)?))
        }
        DataType::Float32 => {
            // MIN/MAX over floats are routed to the float-atomic CAS kernel
            // by launch_agg_kernel; no early rejection needed.
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float32"))?;
            let host: Vec<f32> = collect_filtered_primitive(pa, key_valid, value_valid.as_deref());
            debug_assert_eq!(host.len(), n);
            let input_gpu = GpuVec::<f32>::from_slice_async(&host, stream.raw())?;
            let init: Vec<f32> = vec![identity_f32(op); k];
            let mut acc = GpuVec::<f32>::from_slice_async(&init, stream.raw())?;
            launch_agg_kernel::<f32>(
                op,
                DataType::Float32,
                filtered.col(),
                keys_table,
                &input_gpu,
                &mut acc,
                n,
                k_u32,
                stream,
            )?;
            Ok(AccDownload::F32(download_pinned_f32(&acc, stream)?))
        }
        DataType::Float64 => {
            // MIN/MAX over floats are routed to the float-atomic CAS kernel
            // by launch_agg_kernel; no early rejection needed.
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float64"))?;
            let host: Vec<f64> = collect_filtered_primitive(pa, key_valid, value_valid.as_deref());
            debug_assert_eq!(host.len(), n);
            let input_gpu = GpuVec::<f64>::from_slice_async(&host, stream.raw())?;
            let init: Vec<f64> = vec![identity_f64(op); k];
            let mut acc = GpuVec::<f64>::from_slice_async(&init, stream.raw())?;
            launch_agg_kernel::<f64>(
                op,
                DataType::Float64,
                filtered.col(),
                keys_table,
                &input_gpu,
                &mut acc,
                n,
                k_u32,
                stream,
            )?;
            Ok(AccDownload::F64(download_pinned_f64(&acc, stream)?))
        }
        DataType::Bool | DataType::Utf8 => Err(BoltError::Type(format!(
            "aggregate input dtype {:?} not supported (column '{}')",
            col_io.dtype, col_io.name
        ))),
    }
}

/// Collect a primitive Arrow array's values into a fresh `Vec`, filtering
/// out positions where (key_valid AND value_valid) is false. Either mask
/// being `None` means "all true" for that side. The output length equals the
/// post-filter row count.
fn collect_filtered_primitive<P>(
    pa: &arrow_array::PrimitiveArray<P>,
    key_valid: Option<&[bool]>,
    value_valid: Option<&[bool]>,
) -> Vec<P::Native>
where
    P: arrow_array::types::ArrowPrimitiveType,
    P::Native: Copy,
{
    let n = pa.len();
    let vals = pa.values();
    let mut out: Vec<P::Native> = Vec::with_capacity(n);
    for i in 0..n {
        let kv = key_valid.map(|m| m[i]).unwrap_or(true);
        let vv = value_valid.map(|m| m[i]).unwrap_or(true);
        if kv && vv {
            out.push(vals[i]);
        }
    }
    out
}

/// Pull a numeric input column out of `batch`, upcast each element to f64,
/// and drop positions where (key_valid AND value_valid) is false. Either
/// mask being `None` means "all true" for that side. Used by AVG so the
/// numerator and denominator stay aligned with the (key-NULL, value-NULL)
/// filter applied to the rest of the launch.
fn load_input_column_as_f64_filtered(
    col_io: &ColumnIO,
    batch: &RecordBatch,
    key_valid: Option<&[bool]>,
    value_valid: Option<&[bool]>,
) -> BoltResult<Vec<f64>> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        BoltError::Plan(format!(
            "AVG input '{}' not present in table batch: {}",
            col_io.name, e
        ))
    })?;
    let arr = batch.column(idx);
    match col_io.dtype {
        DataType::Int32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int32"))?;
            Ok(filter_iter_to_f64(pa, key_valid, value_valid, |v| v as f64))
        }
        DataType::Int64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int64"))?;
            Ok(filter_iter_to_f64(pa, key_valid, value_valid, |v| v as f64))
        }
        DataType::Float32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float32"))?;
            Ok(filter_iter_to_f64(pa, key_valid, value_valid, |v| v as f64))
        }
        DataType::Float64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float64"))?;
            Ok(filter_iter_to_f64(pa, key_valid, value_valid, |v| v))
        }
        DataType::Bool | DataType::Utf8 => Err(BoltError::Type(format!(
            "AVG input dtype {:?} not supported (column '{}')",
            col_io.dtype, col_io.name
        ))),
    }
}

/// Helper for `load_input_column_as_f64_filtered`: walks a primitive Arrow
/// array, applies the joint key/value validity mask, and casts each surviving
/// element via `f`.
fn filter_iter_to_f64<P, F>(
    pa: &arrow_array::PrimitiveArray<P>,
    key_valid: Option<&[bool]>,
    value_valid: Option<&[bool]>,
    f: F,
) -> Vec<f64>
where
    P: arrow_array::types::ArrowPrimitiveType,
    P::Native: Copy,
    F: Fn(P::Native) -> f64,
{
    let n = pa.len();
    let vals = pa.values();
    let mut out: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let kv = key_valid.map(|m| m[i]).unwrap_or(true);
        let vv = value_valid.map(|m| m[i]).unwrap_or(true);
        if kv && vv {
            out.push(f(vals[i]));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Output assembly.
// ---------------------------------------------------------------------------

/// Build one Arrow array per group-by column by decoding each packed i64 key
/// back through `decode_key`. Returns arrays in the order the columns appear
/// in `aggregate.group_by` (which matches `components`).
fn build_key_arrays(
    groups: &[(i64, usize)],
    components: &[KeyComponent],
) -> BoltResult<Vec<ArrayRef>> {
    let m = components.len();
    let n = groups.len();

    // Per-column typed buffers; we allocate the right one based on the
    // component dtype and push exactly one value per group.
    enum ColBuf {
        I32(Vec<i32>),
        I64(Vec<i64>),
        F32(Vec<f32>),
        F64(Vec<f64>),
    }

    let mut buffers: Vec<ColBuf> = Vec::with_capacity(m);
    for comp in components {
        match comp.original_dtype {
            DataType::Int32 => buffers.push(ColBuf::I32(Vec::with_capacity(n))),
            DataType::Int64 => buffers.push(ColBuf::I64(Vec::with_capacity(n))),
            DataType::Float32 => buffers.push(ColBuf::F32(Vec::with_capacity(n))),
            DataType::Float64 => buffers.push(ColBuf::F64(Vec::with_capacity(n))),
            DataType::Bool | DataType::Utf8 => {
                return Err(BoltError::Type(format!(
                    "GROUP BY key dtype {:?} not supported on output",
                    comp.original_dtype
                )))
            }
        }
    }

    for (k, _) in groups {
        let decoded = decode_key(*k, components);
        for (buf, val) in buffers.iter_mut().zip(decoded.iter()) {
            match (buf, val) {
                (ColBuf::I32(v), KeyValue::I32(x)) => v.push(*x),
                (ColBuf::I64(v), KeyValue::I64(x)) => v.push(*x),
                (ColBuf::F32(v), KeyValue::F32(x)) => v.push(*x),
                (ColBuf::F64(v), KeyValue::F64(x)) => v.push(*x),
                _ => {
                    return Err(BoltError::Other(
                        "internal: decode_key produced a KeyValue variant \
                         that disagrees with its KeyComponent dtype"
                            .into(),
                    ))
                }
            }
        }
    }

    let mut out: Vec<ArrayRef> = Vec::with_capacity(m);
    for buf in buffers {
        match buf {
            ColBuf::I32(v) => out.push(Arc::new(Int32Array::from(v)) as ArrayRef),
            ColBuf::I64(v) => out.push(Arc::new(Int64Array::from(v)) as ArrayRef),
            ColBuf::F32(v) => out.push(Arc::new(Float32Array::from(v)) as ArrayRef),
            ColBuf::F64(v) => out.push(Arc::new(Float64Array::from(v)) as ArrayRef),
        }
    }
    Ok(out)
}

/// Build the output Arrow array for one aggregate, indexing the downloaded
/// accumulator by each group's slot.
///
/// For SUM(Int32) the accumulator was widened to i64 host-side per
/// `crate::plan::logical_plan::sum_output_dtype`, so the `AccDownload` arrives
/// as `I64` and `out_field.dtype` is `Int64` — `pack_array` consumes those
/// directly. SUM(Int64), SUM(Float32), SUM(Float64), and all MIN/MAX paths
/// preserve their input dtype unchanged.
fn build_agg_array(
    agg: &AggregateExpr,
    out_field: &Field,
    acc: &AccDownload,
    groups: &[(i64, usize)],
    n_groups: usize,
) -> BoltResult<ArrayRef> {
    match (agg, acc) {
        (AggregateExpr::Count(_), AccDownload::I64(host)) => {
            let mut out: Vec<i64> = Vec::with_capacity(n_groups);
            for (_, slot) in groups {
                out.push(host[*slot]);
            }
            pack_array(out_field.dtype, Scalars::I64(out))
        }
        (AggregateExpr::Avg(_), AccDownload::Avg { sum, count }) => {
            let mut out: Vec<f64> = Vec::with_capacity(n_groups);
            for (_, slot) in groups {
                let s = sum[*slot];
                let c = count[*slot];
                let v = if c == 0 { 0.0 } else { s / (c as f64) };
                out.push(v);
            }
            pack_array(out_field.dtype, Scalars::F64(out))
        }
        (AggregateExpr::Sum(_) | AggregateExpr::Min(_) | AggregateExpr::Max(_), other) => {
            let scalars = match other {
                AccDownload::I32(host) => Scalars::I32(
                    groups.iter().map(|(_, slot)| host[*slot]).collect(),
                ),
                AccDownload::I64(host) => Scalars::I64(
                    groups.iter().map(|(_, slot)| host[*slot]).collect(),
                ),
                AccDownload::F32(host) => Scalars::F32(
                    groups.iter().map(|(_, slot)| host[*slot]).collect(),
                ),
                AccDownload::F64(host) => Scalars::F64(
                    groups.iter().map(|(_, slot)| host[*slot]).collect(),
                ),
                AccDownload::Avg { .. } => {
                    return Err(BoltError::Other(
                        "internal: AVG accumulator passed to non-AVG aggregate"
                            .into(),
                    ))
                }
            };
            pack_array(out_field.dtype, scalars)
        }
        (_, _) => Err(BoltError::Other(
            "internal: aggregate / accumulator-variant mismatch".into(),
        )),
    }
}

/// Typed batch of per-group scalar values, prior to dtype-casting into Arrow.
enum Scalars {
    /// Int32 column.
    I32(Vec<i32>),
    /// Int64 column.
    I64(Vec<i64>),
    /// Float32 column.
    F32(Vec<f32>),
    /// Float64 column.
    F64(Vec<f64>),
}

/// Cast a `Scalars` batch into an Arrow array of `out_dtype`. Mirrors the
/// cross-dtype matrix in `aggregate.rs::scalar_to_array`.
fn pack_array(out_dtype: DataType, scalars: Scalars) -> BoltResult<ArrayRef> {
    match (scalars, out_dtype) {
        (Scalars::I32(v), DataType::Int32) => Ok(Arc::new(Int32Array::from(v)) as ArrayRef),
        (Scalars::I64(v), DataType::Int64) => Ok(Arc::new(Int64Array::from(v)) as ArrayRef),
        (Scalars::F32(v), DataType::Float32) => {
            Ok(Arc::new(Float32Array::from(v)) as ArrayRef)
        }
        (Scalars::F64(v), DataType::Float64) => {
            Ok(Arc::new(Float64Array::from(v)) as ArrayRef)
        }

        // Cross-dtype paths the scalar reducer also accepts.
        (Scalars::I32(v), DataType::Int64) => Ok(Arc::new(Int64Array::from(
            v.into_iter().map(|x| x as i64).collect::<Vec<_>>(),
        )) as ArrayRef),
        (Scalars::I32(v), DataType::Float32) => Ok(Arc::new(Float32Array::from(
            v.into_iter().map(|x| x as f32).collect::<Vec<_>>(),
        )) as ArrayRef),
        (Scalars::I32(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(
            v.into_iter().map(|x| x as f64).collect::<Vec<_>>(),
        )) as ArrayRef),
        (Scalars::I64(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(
            v.into_iter().map(|x| x as f64).collect::<Vec<_>>(),
        )) as ArrayRef),
        (Scalars::F32(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(
            v.into_iter().map(|x| x as f64).collect::<Vec<_>>(),
        )) as ArrayRef),

        (_, dt) => Err(BoltError::Type(format!(
            "GROUP BY: cannot pack scalars into output dtype {:?}",
            dt
        ))),
    }
}

// ---------------------------------------------------------------------------
// Identities for the accumulator initialiser.
// ---------------------------------------------------------------------------

fn identity_i32(op: ReduceOp) -> i32 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => 0,
        ReduceOp::Min => i32::MAX,
        ReduceOp::Max => i32::MIN,
    }
}

fn identity_i64(op: ReduceOp) -> i64 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => 0,
        ReduceOp::Min => i64::MAX,
        ReduceOp::Max => i64::MIN,
    }
}

fn identity_f32(op: ReduceOp) -> f32 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => 0.0,
        ReduceOp::Min => f32::INFINITY,
        ReduceOp::Max => f32::NEG_INFINITY,
    }
}

fn identity_f64(op: ReduceOp) -> f64 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => 0.0,
        ReduceOp::Min => f64::INFINITY,
        ReduceOp::Max => f64::NEG_INFINITY,
    }
}

// ---------------------------------------------------------------------------
// Misc helpers (mirror of the private helpers in aggregate.rs).
// ---------------------------------------------------------------------------

/// Resolve `name` to its `ColumnIO` within `inputs`.
fn resolve_input<'a>(inputs: &'a [ColumnIO], name: &str) -> BoltResult<&'a ColumnIO> {
    inputs.iter().find(|c| c.name == name).ok_or_else(|| {
        BoltError::Plan(format!(
            "aggregate input column '{}' not found in plan inputs",
            name
        ))
    })
}

/// Extract the column name from a bare-column-ref expression. The v1 path
/// requires every aggregate input to be a bare column ref.
fn bare_column_name(expr: &Expr) -> BoltResult<&str> {
    match expr {
        Expr::Column(name) => Ok(name.as_str()),
        Expr::Alias(inner, _) => bare_column_name(inner),
        _ => Err(BoltError::Other(
            "GROUP BY: aggregate input must be a bare column reference in v1".into(),
        )),
    }
}

/// `Type` error for a failed Arrow downcast on column `name`.
fn downcast_err(name: &str, expected: &str) -> BoltError {
    BoltError::Type(format!(
        "GROUP BY input column '{}' could not be downcast to {}",
        name, expected
    ))
}

/// Map Arrow `DataType` to our plan `DataType` (mirrors `aggregate.rs`).
fn arrow_dtype_to_plan(d: &ArrowDataType) -> BoltResult<DataType> {
    match d {
        ArrowDataType::Int32 => Ok(DataType::Int32),
        ArrowDataType::Int64 => Ok(DataType::Int64),
        ArrowDataType::Float32 => Ok(DataType::Float32),
        ArrowDataType::Float64 => Ok(DataType::Float64),
        ArrowDataType::Boolean => Ok(DataType::Bool),
        ArrowDataType::Utf8 => Ok(DataType::Utf8),
        other => Err(BoltError::Type(format!(
            "unsupported Arrow dtype {:?}",
            other
        ))),
    }
}

/// Map our plan `DataType` to Arrow `DataType` (mirrors `aggregate.rs`).
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

/// Build an Arrow `Schema` from our plan `Schema` for the output `RecordBatch`.
fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}

// ---------------------------------------------------------------------------
// Host-only tests for `pack_keys` / `decode_key` (no GPU required).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal `AggregateSpec` for the pack_keys tests. We only need
    /// `inputs` + `group_by`; the `aggregates` and `output_schema` aren't
    /// touched by `pack_keys`.
    fn spec(inputs: Vec<(&str, DataType)>, group_by: Vec<usize>) -> AggregateSpec {
        AggregateSpec {
            inputs: inputs
                .into_iter()
                .map(|(n, d)| ColumnIO {
                    name: n.to_string(),
                    dtype: d,
                })
                .collect(),
            group_by,
            aggregates: vec![],
            output_schema: Schema::new(vec![]),
        }
    }

    /// Build a single-column RecordBatch from a typed Arrow array. The field
    /// is marked nullable so callers can pass arrays with NULL validity
    /// bitmaps (used by the H1 NULL tests below).
    fn one_col_batch(name: &str, arr: ArrayRef) -> RecordBatch {
        let dt = arr.data_type().clone();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            name, dt, true,
        )]));
        RecordBatch::try_new(schema, vec![arr]).expect("one-col batch")
    }

    fn two_col_batch(
        n1: &str,
        a1: ArrayRef,
        n2: &str,
        a2: ArrayRef,
    ) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(n1, a1.data_type().clone(), true),
            ArrowField::new(n2, a2.data_type().clone(), true),
        ]));
        RecordBatch::try_new(schema, vec![a1, a2]).expect("two-col batch")
    }

    /// 1. Single Int32 key column packs to i64 via sign-extension.
    #[test]
    fn pack_keys_single_int32() {
        let agg = spec(vec![("k", DataType::Int32)], vec![0]);
        let batch = one_col_batch(
            "k",
            Arc::new(Int32Array::from(vec![7i32, -3, 0, i32::MIN, i32::MAX]))
                as ArrayRef,
        );
        let packed = pack_keys(&agg, &batch).expect("pack ok");
        assert_eq!(packed.components.len(), 1);
        assert_eq!(packed.components[0].original_dtype, DataType::Int32);
        assert_eq!(packed.components[0].bit_offset, 0);

        // Sign-extension: -3 as u32 = 0xFFFF_FFFD; that zero-extended into
        // u64 is 0x0000_0000_FFFF_FFFD, which `as i64` is 4294967293.
        let expected: Vec<i64> = vec![
            7i64,
            ((-3i32) as u32 as u64) as i64,
            0i64,
            (i32::MIN as u32 as u64) as i64,
            (i32::MAX as u32 as u64) as i64,
        ];
        assert_eq!(packed.keys_i64, expected);

        // Round-trip via decode_key for the -3 entry.
        let dec = decode_key(packed.keys_i64[1], &packed.components);
        assert_eq!(dec, vec![KeyValue::I32(-3)]);

        // i32::MIN round-trip.
        let dec_min = decode_key(packed.keys_i64[3], &packed.components);
        assert_eq!(dec_min, vec![KeyValue::I32(i32::MIN)]);
    }

    /// 2. (Int32, Int32) packs to `((a as u64) << 32) | (b as u32 as u64)`,
    ///    then `as i64`. Verified on the worked example a=7, b=-3.
    #[test]
    fn pack_keys_two_int32() {
        let agg = spec(
            vec![("a", DataType::Int32), ("b", DataType::Int32)],
            vec![0, 1],
        );
        let batch = two_col_batch(
            "a",
            Arc::new(Int32Array::from(vec![7i32, 1, -1])) as ArrayRef,
            "b",
            Arc::new(Int32Array::from(vec![-3i32, 2, -1])) as ArrayRef,
        );
        let packed = pack_keys(&agg, &batch).expect("pack ok");
        assert_eq!(packed.components.len(), 2);
        // First group-by column lives in the high 32 bits.
        assert_eq!(packed.components[0].name, "a");
        assert_eq!(packed.components[0].bit_offset, 32);
        assert_eq!(packed.components[1].name, "b");
        assert_eq!(packed.components[1].bit_offset, 0);

        // a=7, b=-3 → ((7u64 << 32) | (0xFFFF_FFFDu32 as u64)) as i64
        //          → 0x0000_0007_FFFF_FFFD as i64
        let expected_0 = ((7u64 << 32) | (-3i32 as u32 as u64)) as i64;
        assert_eq!(packed.keys_i64[0], expected_0);

        // Decode round-trip.
        let dec = decode_key(packed.keys_i64[0], &packed.components);
        assert_eq!(dec, vec![KeyValue::I32(7), KeyValue::I32(-3)]);

        // a=-1, b=-1 should NOT collide with a=7, b=-3 (the whole point of
        // a lossless pack).
        assert_ne!(packed.keys_i64[0], packed.keys_i64[2]);
        let dec_neg = decode_key(packed.keys_i64[2], &packed.components);
        assert_eq!(dec_neg, vec![KeyValue::I32(-1), KeyValue::I32(-1)]);
    }

    /// 3. Single Float32 key packs to `f32::to_bits() as i64` and round-trips
    ///    via `decode_key` back to the same f32 bit pattern.
    #[test]
    fn pack_keys_float32() {
        let agg = spec(vec![("f", DataType::Float32)], vec![0]);
        let vals = vec![1.5f32, -2.25, 0.0, -0.0, f32::INFINITY];
        let batch = one_col_batch(
            "f",
            Arc::new(Float32Array::from(vals.clone())) as ArrayRef,
        );
        let packed = pack_keys(&agg, &batch).expect("pack ok");
        assert_eq!(packed.components.len(), 1);
        assert_eq!(packed.components[0].original_dtype, DataType::Float32);
        assert_eq!(packed.components[0].bit_offset, 0);

        for (i, v) in vals.iter().enumerate() {
            let expected = (v.to_bits() as u64) as i64;
            assert_eq!(packed.keys_i64[i], expected, "row {i}");
            let dec = decode_key(packed.keys_i64[i], &packed.components);
            assert_eq!(dec.len(), 1);
            match dec[0] {
                KeyValue::F32(f) => assert_eq!(f.to_bits(), v.to_bits()),
                other => panic!("expected F32, got {:?}", other),
            }
        }

        // Documented limitation: -0.0 and +0.0 have different bit patterns
        // and therefore pack to different i64 keys (they would group
        // separately). The test pins this behaviour.
        assert_ne!(packed.keys_i64[2], packed.keys_i64[3]);
    }

    /// 4. Three columns exceeds the v1 lossless packing budget.
    #[test]
    fn pack_keys_unsupported() {
        let agg = spec(
            vec![
                ("a", DataType::Int32),
                ("b", DataType::Int32),
                ("c", DataType::Int32),
            ],
            vec![0, 1, 2],
        );
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                ArrowField::new("a", ArrowDataType::Int32, false),
                ArrowField::new("b", ArrowDataType::Int32, false),
                ArrowField::new("c", ArrowDataType::Int32, false),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![1])) as ArrayRef,
                Arc::new(Int32Array::from(vec![2])) as ArrayRef,
                Arc::new(Int32Array::from(vec![3])) as ArrayRef,
            ],
        )
        .expect("batch");

        let err = pack_keys(&agg, &batch).expect_err("should error");
        let msg = format!("{err}");
        assert!(
            msg.contains("> 64 bits") || msg.contains("not yet supported"),
            "unexpected error message: {msg}"
        );
    }

    /// 5. (Int64, Int64) is 128 bits of key material and is rejected for v1.
    #[test]
    fn pack_keys_int64_pair() {
        let agg = spec(
            vec![("a", DataType::Int64), ("b", DataType::Int64)],
            vec![0, 1],
        );
        let batch = two_col_batch(
            "a",
            Arc::new(Int64Array::from(vec![1i64])) as ArrayRef,
            "b",
            Arc::new(Int64Array::from(vec![2i64])) as ArrayRef,
        );
        let err = pack_keys(&agg, &batch).expect_err("should error");
        let msg = format!("{err}");
        assert!(
            msg.contains("> 64 bits") || msg.contains("not yet supported"),
            "unexpected error message: {msg}"
        );
    }

    // -----------------------------------------------------------------
    // NULL-handling tests (H1 fix).
    //
    // The pre-fix `pack_keys` read `.values()` straight off the Arrow
    // array, picking up garbage bytes at NULL positions and forming a
    // fake group. The post-fix `pack_keys` surfaces a `key_valid` mask
    // and the executor drops NULL-key rows before they reach the GPU.
    // -----------------------------------------------------------------

    /// pack_keys surfaces a `key_valid = None` when the key column has no nulls.
    #[test]
    fn pack_keys_no_nulls_omits_mask() {
        let agg = spec(vec![("k", DataType::Int32)], vec![0]);
        let batch = one_col_batch(
            "k",
            Arc::new(Int32Array::from(vec![1i32, 2, 3])) as ArrayRef,
        );
        let packed = pack_keys(&agg, &batch).expect("pack ok");
        assert!(
            packed.key_valid.is_none(),
            "expected None mask for null-free input"
        );
    }

    /// pack_keys surfaces a `key_valid` mask reflecting per-row nullness
    /// when the key column has NULLs. Downstream the executor drops those
    /// rows; here we just pin the mask shape.
    #[test]
    fn pack_keys_int32_with_nulls_surfaces_mask() {
        let agg = spec(vec![("k", DataType::Int32)], vec![0]);
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![
            Some(1i32),
            None,
            Some(3),
            None,
            Some(5),
        ]));
        let batch = one_col_batch("k", arr);
        let packed = pack_keys(&agg, &batch).expect("pack ok");
        let mask = packed.key_valid.expect("expected mask");
        assert_eq!(mask, vec![true, false, true, false, true]);
    }

    /// Two-column GROUP BY merges per-column null masks via logical AND:
    /// any column NULL at row i drops that row from the keep set.
    #[test]
    fn pack_keys_two_col_null_mask_is_and() {
        let agg = spec(
            vec![("a", DataType::Int32), ("b", DataType::Int32)],
            vec![0, 1],
        );
        let a: ArrayRef =
            Arc::new(Int32Array::from(vec![Some(1), Some(2), None, Some(4)]));
        let b: ArrayRef =
            Arc::new(Int32Array::from(vec![Some(10), None, Some(30), Some(40)]));
        let batch = two_col_batch("a", a, "b", b);
        let packed = pack_keys(&agg, &batch).expect("pack ok");
        let mask = packed.key_valid.expect("expected mask");
        // Row 0: keep (a=1, b=10), row 1: drop (b NULL), row 2: drop (a NULL),
        // row 3: keep (a=4, b=40).
        assert_eq!(mask, vec![true, false, false, true]);
    }

    /// `and_masks` honours `None` as all-true.
    #[test]
    fn and_masks_combinators() {
        assert_eq!(and_masks(None, None), None);
        assert_eq!(and_masks(Some(vec![true, false]), None), Some(vec![true, false]));
        assert_eq!(and_masks(None, Some(vec![false, true])), Some(vec![false, true]));
        assert_eq!(
            and_masks(Some(vec![true, true, false]), Some(vec![true, false, true])),
            Some(vec![true, false, false])
        );
    }

    /// `column_null_mask` returns `None` when the column has no nulls and
    /// a per-row mask when it does. Drives the SUM/MIN/MAX value-NULL
    /// filtering in `run_typed_agg`.
    #[test]
    fn column_null_mask_basic() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2, 3]));
        let batch = one_col_batch("v", arr);
        let io = ColumnIO {
            name: "v".to_string(),
            dtype: DataType::Int64,
        };
        assert!(column_null_mask(&io, &batch).unwrap().is_none());

        let arr2: ArrayRef =
            Arc::new(Int64Array::from(vec![Some(1i64), None, Some(3)]));
        let batch2 = one_col_batch("v", arr2);
        let mask = column_null_mask(&io, &batch2).unwrap().expect("mask");
        assert_eq!(mask, vec![true, false, true]);
    }

    /// `collect_filtered_primitive` drops positions where EITHER mask is
    /// false; this is exactly the SUM/MIN/MAX value-upload path. The
    /// garbage-at-NULL bytes that would otherwise corrupt the reduction
    /// stay in the source buffer and never reach the kernel.
    #[test]
    fn collect_filtered_primitive_drops_null_rows() {
        // Underlying values buffer at NULL positions could be anything;
        // here we use a large value (1000) that would visibly skew SUM.
        let arr = Int32Array::from(vec![Some(1i32), None, Some(3), Some(4)]);
        // value_valid derived from arrow: [T, F, T, T]
        let vv: Vec<bool> = (0..arr.len()).map(|i| !arr.is_null(i)).collect();
        let out = collect_filtered_primitive::<arrow_array::types::Int32Type>(
            &arr,
            None,
            Some(&vv),
        );
        assert_eq!(out, vec![1, 3, 4]);

        // With a key_valid that ALSO drops row 0, only rows 2 and 3 survive.
        let kv = vec![false, true, true, true];
        let out2 = collect_filtered_primitive::<arrow_array::types::Int32Type>(
            &arr,
            Some(&kv),
            Some(&vv),
        );
        assert_eq!(out2, vec![3, 4]);
    }

    /// `filter_iter_to_f64` (the AVG-input helper) drops NULL positions
    /// from both the key and the value side and upcasts in one pass.
    #[test]
    fn filter_iter_to_f64_drops_and_casts() {
        let arr = Int32Array::from(vec![Some(2i32), None, Some(4), Some(6)]);
        let vv: Vec<bool> = (0..arr.len()).map(|i| !arr.is_null(i)).collect();
        let out = filter_iter_to_f64::<arrow_array::types::Int32Type, _>(
            &arr,
            None,
            Some(&vv),
            |v| v as f64,
        );
        assert_eq!(out, vec![2.0f64, 4.0, 6.0]);
    }

    // -------- Stage-3 async round-trip (requires GPU) ----------------

    /// Single-key GROUP BY through the engine: confirms that the
    /// Stage-3 async memcpy + pinned D2H plumbing produces the same
    /// per-group sums as a host-side check.
    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn async_groupby_int32_sum_round_trip() {
        use crate::Engine;
        use arrow_array::Int64Array;

        let mut engine = Engine::new().expect("ctx");
        // 12 rows, key in {0, 1, 2}; expected SUMs derived from the
        // closed form 0..12 grouped by key % 3.
        let keys: Vec<i32> = (0..12i32).map(|i| i % 3).collect();
        let vals: Vec<i32> = (0..12i32).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(keys)) as ArrayRef,
                Arc::new(Int32Array::from(vals)) as ArrayRef,
            ],
        )
        .unwrap();
        engine.register_table("t", batch).unwrap();

        let h = engine
            .sql("SELECT k, SUM(v) FROM t GROUP BY k")
            .expect("execute");
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 3);
        let ks = out
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let ss = out
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        // Build a host-side expected map and compare.
        let mut expected = std::collections::HashMap::<i32, i64>::new();
        for v in 0..12i64 {
            *expected.entry((v as i32) % 3).or_default() += v;
        }
        for i in 0..3 {
            let k = ks.value(i);
            let s = ss.value(i);
            assert_eq!(
                Some(&s),
                expected.get(&k).map(|x| x),
                "key={} sum={}",
                k,
                s
            );
        }
    }
}
