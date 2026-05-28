// SPDX-License-Identifier: Apache-2.0

//! GROUP BY execution using the sentinel-free (slot-valid-flag) protocol.
//!
//! This module mirrors [`crate::exec::groupby::execute_groupby`] but swaps
//! the keys + agg kernels for the variants in
//! [`crate::jit::valid_flag_kernels`], which carry a parallel `slot_valid:
//! GpuVec<u32>` occupancy table rather than reserving `i64::MIN` as an
//! "empty slot" sentinel.
//!
//! The orchestrator in `groupby.rs` dispatches here for queries whose key
//! encoding could collide with the classic sentinel. Historically the
//! flagship case was Float64 `-0.0` (whose `f64::to_bits()` is exactly
//! `0x8000_0000_0000_0000 == i64::MIN`); review C12 added a host-side
//! `-0.0 → +0.0` canonicalisation in `load_key_column_bits` so that case
//! no longer reaches the kernel as `i64::MIN`. The sentinel-free path
//! still earns its keep for any other key shape whose packed bits land
//! on `i64::MIN` (e.g. a future two-column packing that happens to
//! produce `0x8000_0000_0000_0000`), and stays the explicit opt-in for
//! callers who want to avoid the sentinel pre-validation altogether.
//!
//! # Slot-state machine
//!
//! Each entry of `slot_valid: GpuVec<u32>` cycles through:
//!
//! | value | meaning                                                      |
//! |-------|--------------------------------------------------------------|
//! | `0`   | empty — no thread has claimed this slot                      |
//! | `1`   | claimed — a winner is about to write the key                 |
//! | `2`   | committed — `keys_table[slot]` is safe to read               |
//!
//! Host post-processing treats `slot_valid[i] != 0` as "this slot is a
//! group". After a synchronous kernel launch every claimed slot will be in
//! state `2`, so the looser `!= 0` predicate and the strict `== 2`
//! predicate agree in practice; we use `!= 0` so a partially-committed
//! download (should we ever go async) at worst yields a slot whose key is
//! the right one but hasn't yet been observed by the agg kernel — which
//! the executor never permits anyway thanks to `stream.synchronize()`
//! between launches.
//!
//! # Memory overhead
//!
//! The classic variant allocates an `i64[k]` keys table (= 8k bytes). This
//! variant allocates the same keys table PLUS a `u32[k]` slot_valid table
//! (= 4k bytes extra). For a typical `k` of, say, 65 536 slots, that is an
//! extra 256 KiB of device memory — small compared to the keys table
//! itself.
//!
//! Each kernel launch ALSO allocates a small "spill" buffer used by the
//! deadlock-hardening path documented in
//! [`crate::jit::valid_flag_kernels`]: an `i64[SPILL_CAPACITY]` keys
//! buffer, a `T[SPILL_CAPACITY]` values buffer (agg kernels only), and a
//! single-element `u32` counter. At the default [`SPILL_CAPACITY`] of
//! 4096 rows this is ~32 KiB per agg kernel + 32 KiB for the keys
//! kernel — negligible. The kernels write to the spill buffer only when
//! the bounded PROBE or SPIN loop overflows; under the executor-enforced
//! load factor of < 0.5 we expect zero spills in practice.
//!
//! # PV-stage-d: native-validity dispatch
//!
//! This executor is the natural home for the
//! [`crate::jit::valid_flag_kernels::compile_keys_valid_kernel_with_validity`]
//! /
//! [`crate::jit::valid_flag_kernels::compile_agg_valid_kernel_with_validity`]
//! companions. Once the dispatch logic upstream looks at
//! `KernelSpec::input_has_validity` (or the plan-level signal from
//! [`crate::plan::sql_frontend::TableProvider::has_nulls`]) we will:
//!
//! 1. Inspect the source `RecordBatch` for an Arrow null buffer on the
//!    group-by column AND each aggregate input.
//! 2. If any column has nulls, build the packed-bit validity vector via
//!    [`crate::jit::valid_flag_kernels::pack_validity_bits`], upload as
//!    a `GpuVec<u8>`, and dispatch to the `_with_validity` variant.
//! 3. Otherwise fall through to the existing keys + agg launch (the
//!    current code below).
//!
//! Until then, the host-side `null_count` check inside `pack_keys` (and
//! its sibling helpers) implicitly rejects null-bearing inputs by
//! treating them as "no batch found" — the safety net documented in
//! `crate::exec::groupby`'s module header.

use std::collections::HashMap;
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
use crate::exec::module_cache;
use crate::exec::n_rows_to_u32;
use crate::jit::agg_kernels::ReduceOp;
use crate::jit::valid_flag_kernels::{
    compile_agg_valid_kernel, compile_agg_valid_kernel_with_validity,
    compile_keys_valid_kernel, pack_validity_bits, valid_block_size,
    VALID_AGG_KERNEL_ENTRY, VALID_KEYS_KERNEL_ENTRY,
};
// PV-stage-f: the sentinel-free validity-aware emitters are now wired in
// at the launcher boundary. `compile_agg_valid_kernel_with_validity`
// covers integer SUM/MIN/MAX + float SUM; the float MIN/MAX path routes
// through `valid_flag_float::compile_agg_valid_float_kernel_with_validity`.
// The host-strip path remains as the correctness fallback for any
// (op, dtype) outside that coverage.
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Field, Schema};
use crate::plan::physical_plan::{AggregateSpec, ColumnIO, PhysicalPlan};

/// Capacity (in rows) of each per-kernel spill buffer used by the
/// deadlock-hardening path in [`crate::jit::valid_flag_kernels`]. At a
/// load factor < 0.5 — which the executor enforces — we expect zero
/// spills in practice; this value is "very generous" headroom against
/// pathological cases (warp-scheduler unfairness, hash-collision storms).
///
/// Memory cost is ~32 KiB per kernel launch (8 bytes × 4096 for keys,
/// plus up to 8 bytes × 4096 for values on the agg kernels). If a kernel
/// ever exceeds this we error out on the host side rather than silently
/// drop rows — see [`SpillError`] equivalents at the call sites.
const SPILL_CAPACITY: usize = 4096;

/// Sentinel value used by the host to detect "uninitialised" entries in
/// the spill_keys buffer. The kernel only writes entries with index
/// `< max_spill`, so on the host we can rely on the counter to know how
/// many real entries there are; this placeholder is purely defensive.
const SPILL_EMPTY_KEY: i64 = i64::MIN;

/// PV-stage-e: observability counter — increments once per agg launch this
/// sentinel-free executor routes through the native `_with_validity`
/// kernel path (i.e.
/// [`crate::jit::valid_flag_kernels::compile_agg_valid_kernel_with_validity`]
/// or
/// [`crate::jit::valid_flag_float::compile_agg_valid_float_kernel_with_validity`]).
///
/// `execute_groupby_valid` does not have a `pre` KernelSpec, so the
/// planner-time `input_has_validity` signal is unavailable; the runtime
/// gate is the source `RecordBatch`'s per-column `null_count()` — see
/// [`column_should_use_native_validity`]. Stage F will plumb the planner
/// signal end-to-end so this executor's dispatch matches the
/// `groupby_with_pre` plan-time path exactly.
///
/// Used by inline `#[cfg(test)]` tests; production code does not read it.
#[doc(hidden)]
pub static NATIVE_VALIDITY_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// PV-stage-f: runtime predicate — should this column's agg launch route
/// through the native `_with_validity` kernel (instead of host-stripping
/// before upload)?
///
/// `compile_agg_valid_kernel_with_validity` (integer SUM/MIN/MAX + float
/// SUM) and `compile_agg_valid_float_kernel_with_validity` (float
/// MIN/MAX, finished in Stage E) together cover the full agg matrix for
/// the sentinel-free path — so unlike `groupby.rs`, all (op, dtype)
/// combinations with at least one NULL row are eligible here. The
/// host-strip remains the correctness fallback for shapes outside this
/// set (Utf8, Bool, etc., which the kernels reject anyway).
///
/// Stage F now consumes this predicate in `run_typed_agg` and dispatches
/// to [`launch_agg_kernel`] with a `validity_ptr` set; the launcher
/// increments [`NATIVE_VALIDITY_LAUNCHES`] when it actually routes
/// through the `_with_validity` PTX entry.
fn column_should_use_native_validity(
    arr: &dyn arrow_array::Array,
    op: ReduceOp,
    dtype: DataType,
) -> bool {
    if arr.null_count() == 0 {
        return false;
    }
    // Full coverage now that `valid_flag_float::compile_agg_valid_float_kernel_with_validity`
    // is implemented (PV-stage-e). The sentinel-free integer kernel
    // (`valid_flag_kernels::compile_agg_valid_kernel_with_validity`)
    // covers SUM/MIN/MAX over Int32/Int64 and SUM over Float32/Float64;
    // the float kernel covers MIN/MAX over Float32/Float64. Bool/Utf8
    // are rejected by the kernels, so they don't dispatch here.
    matches!(
        (op, dtype),
        (ReduceOp::Sum, DataType::Int32)
            | (ReduceOp::Sum, DataType::Int64)
            | (ReduceOp::Sum, DataType::Float32)
            | (ReduceOp::Sum, DataType::Float64)
            | (ReduceOp::Min, DataType::Int32)
            | (ReduceOp::Max, DataType::Int32)
            | (ReduceOp::Min, DataType::Int64)
            | (ReduceOp::Max, DataType::Int64)
            | (ReduceOp::Min, DataType::Float32)
            | (ReduceOp::Max, DataType::Float32)
            | (ReduceOp::Min, DataType::Float64)
            | (ReduceOp::Max, DataType::Float64)
    )
}

// ---------------------------------------------------------------------------
// Stage-3 pinned D2H helpers for the accumulator-download path.
// Mirrors the helpers in `groupby.rs`. See that module's doc for the
// rationale.
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

/// Execute a GROUP BY plan using the sentinel-free (slot-valid-flag)
/// protocol.
///
/// Mirrors `groupby::execute_groupby`'s public surface; the orchestrator
/// dispatches to this path when the key encoding could collide with the
/// classic kernel's `i64::MIN` sentinel, or when the user explicitly
/// opts in. Note: review C12 canonicalises `-0.0` to `+0.0` before
/// upload, so the historical `-0.0 → i64::MIN` clash that originally
/// motivated this module no longer arises for that case; the sentinel-
/// free path is still the right home for any other shape whose packed
/// bits happen to land on `i64::MIN`.
pub fn execute_groupby_valid(
    plan: &PhysicalPlan,
    table_batch: &RecordBatch,
) -> BoltResult<RecordBatch> {
    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        other => {
            return Err(BoltError::Other(format!(
                "execute_groupby_valid: expected Aggregate plan, got {:?}",
                std::mem::discriminant(other)
            )))
        }
    };

    if aggregate.group_by.is_empty() {
        return Err(BoltError::Other(
            "execute_groupby_valid: aggregate has no GROUP BY columns; use execute_aggregate".into(),
        ));
    }
    if pre.is_some() {
        return Err(BoltError::Other(
            "GROUP BY (valid-flag) with projection/filter pre-kernel not yet implemented".into(),
        ));
    }

    // Encode all group-by columns into i64 keys (host-side packing). Reuses
    // the same encoding contract as the classic path: distinct tuples map
    // to distinct i64s. The KEY DIFFERENCE versus `execute_groupby` is that
    // we do NOT validate-out the `i64::MIN` value — any i64 is legal here.
    let packed = pack_keys(aggregate, table_batch)?;
    let key_components = packed.components;
    let key_valid = packed.key_valid;

    // NULL keys: SQL standard semantics are implementation-defined for whether
    // a NULL key forms its own group. For v1 we drop rows whose key is NULL,
    // matching the classic path's behaviour (see `groupby::execute_groupby`).
    // This is also what prevents the garbage bit pattern in the NULL slots of
    // the values buffer from reaching the GPU keys kernel and forming a fake
    // group (H1 fix).
    let host_keys: Vec<i64> = match &key_valid {
        Some(mask) => packed
            .keys_i64
            .iter()
            .zip(mask.iter())
            .filter_map(|(k, &keep)| if keep { Some(*k) } else { None })
            .collect(),
        None => packed.keys_i64,
    };
    let n_rows = host_keys.len();

    // Estimate K from the unique-key count (host scan). Same heuristic as
    // the classic path: load factor < 0.5 so spin contention is rare.
    let n_unique = unique_count(&host_keys);
    let k = next_pow2((n_unique.saturating_mul(2)).saturating_add(16)).max(64);
    let k_u32 = u32::try_from(k).map_err(|_| {
        BoltError::Other(format!(
            "GROUP BY hash table size {} exceeds u32::MAX",
            k
        ))
    })?;

    // Stage-3: per-call stream up front so all subsequent H2D / kernels /
    // D2H land on the same ordering domain.
    let stream = CudaStream::null_or_default();

    // Allocate keys table (arbitrary contents — slot_valid drives occupancy)
    // and the slot_valid table (zero-initialised = all slots empty).
    //
    // Stage-3: async H2D for the keys + key-column uploads. slot_valid is
    // a pure zero buffer, so we use `zeros_async` (skips the host
    // allocation entirely).
    let keys_init: Vec<i64> = vec![0i64; k];
    let mut keys_table = GpuVec::<i64>::from_slice_async(&keys_init, stream.raw())?;
    let mut slot_valid = GpuVec::<u32>::zeros_async(k, stream.raw())?;
    let key_col_gpu = GpuVec::<i64>::from_slice_async(&host_keys, stream.raw())?;

    // Allocate the keys-kernel spill buffers. The kernel writes to these
    // only when its bounded probe / spin loops overflow (extremely rare
    // under load factor < 0.5).
    //
    // Spill init uses async H2D for the keys buffer and a `zeros_async`
    // for the single-element counter (cheap and dependency-ordered).
    let spill_init: Vec<i64> = vec![SPILL_EMPTY_KEY; SPILL_CAPACITY];
    let mut keys_spill_keys =
        GpuVec::<i64>::from_slice_async(&spill_init, stream.raw())?;
    let mut keys_spill_counter = GpuVec::<u32>::zeros_async(1, stream.raw())?;
    let max_spill_u32 = u32::try_from(SPILL_CAPACITY).map_err(|_| {
        BoltError::Other(format!(
            "SPILL_CAPACITY {} exceeds u32::MAX",
            SPILL_CAPACITY
        ))
    })?;

    // Launch the keys-only kernel.
    launch_keys_kernel(
        &key_col_gpu,
        &mut keys_table,
        &mut slot_valid,
        &mut keys_spill_keys,
        &mut keys_spill_counter,
        max_spill_u32,
        n_rows,
        k_u32,
        &stream,
    )?;

    // For each aggregate, prepare its accumulator and launch the agg kernel.
    // Each agg launch produces its own (potentially-empty) spill that we
    // fold back into the result after the main keys-table download.
    //
    // PV-stage-f: read the planner's `AggregateSpec::input_has_validity`
    // signal — when any input was flagged, the per-aggregate dispatch
    // below uses this together with the runtime per-column null check
    // (`column_should_use_native_validity`) to decide between the native
    // `_with_validity` kernel and the legacy host-strip path.
    let any_input_has_validity: bool =
        aggregate.input_has_validity.iter().any(|&v| v);
    let mut acc_results: Vec<AccDownload> = Vec::with_capacity(aggregate.aggregates.len());
    for agg in &aggregate.aggregates {
        let acc = run_one_aggregate(
            agg,
            &aggregate.inputs,
            &key_col_gpu,
            &keys_table,
            &slot_valid,
            table_batch,
            n_rows,
            k,
            k_u32,
            max_spill_u32,
            &stream,
            key_valid.as_deref(),
            any_input_has_validity,
        )?;
        acc_results.push(acc);
    }

    // Stage-3: download keys table + slot_valid through pinned host
    // buffers. The keys kernel synchronized the stream before
    // returning, so the underlying device memory is already settled;
    // these D2Hs just want the directly-DMAable pinned path.
    let host_keys_table: Vec<i64> = {
        let pinned = keys_table.to_pinned_async(stream.raw())?;
        stream.synchronize()?;
        pinned.as_slice().to_vec()
    };
    let host_slot_valid: Vec<u32> = {
        let pinned = slot_valid.to_pinned_async(stream.raw())?;
        stream.synchronize()?;
        pinned.as_slice().to_vec()
    };
    drop(keys_table);
    drop(slot_valid);
    drop(key_col_gpu);

    // Download the keys-kernel spill. Any entries here represent rows
    // whose key never made it into the committed `keys_table` — we must
    // add them as new groups on the host.
    //
    // Counter download stays sync (single u32, not worth the pinned
    // hop). Spill keys go through the pinned path when the counter
    // indicates non-zero entries.
    let keys_spill_count_raw = keys_spill_counter.to_vec()?[0] as usize;
    if keys_spill_count_raw > SPILL_CAPACITY {
        return Err(BoltError::Other(format!(
            "GROUP BY spill overflow ({} rows lost from keys kernel); \
             increase load factor or SPILL_CAPACITY",
            keys_spill_count_raw - SPILL_CAPACITY
        )));
    }
    let keys_spill_count = keys_spill_count_raw.min(SPILL_CAPACITY);
    let host_keys_spill: Vec<i64> = if keys_spill_count > 0 {
        let pinned = keys_spill_keys.to_pinned_async(stream.raw())?;
        stream.synchronize()?;
        pinned.as_slice()[..keys_spill_count].to_vec()
    } else {
        Vec::new()
    };
    drop(keys_spill_keys);
    drop(keys_spill_counter);

    // Walk slot_valid: every non-zero entry is a group. Then fold in any
    // keys-kernel-spilled keys as additional groups (deduplicating against
    // both each other and the committed table).
    //
    // `groups` is a list of distinct (encoded_key, source) entries; source
    // is either `Slot(idx)` for keys placed in the committed table by the
    // keys kernel, or `Spilled` for keys that overflowed the bounded probe.
    // The agg-kernel accumulators we already downloaded are indexed by
    // GPU slot; spilled groups start at the per-op identity and are
    // populated by folding the agg-kernel spills below.
    let mut groups: Vec<GroupEntry> = Vec::new();
    let mut key_to_group: HashMap<i64, usize> = HashMap::new();
    for (slot, &v) in host_slot_valid.iter().enumerate() {
        if v != 0 {
            let key = host_keys_table[slot];
            // The keys kernel guarantees each committed key appears in
            // exactly one slot, but defensively dedupe — a hypothetical
            // future race could otherwise produce a doubled group.
            if let std::collections::hash_map::Entry::Vacant(e) =
                key_to_group.entry(key)
            {
                e.insert(groups.len());
                groups.push(GroupEntry {
                    key,
                    source: GroupSource::Slot(slot),
                });
            }
        }
    }
    for &spilled_key in &host_keys_spill {
        if let std::collections::hash_map::Entry::Vacant(e) =
            key_to_group.entry(spilled_key)
        {
            e.insert(groups.len());
            groups.push(GroupEntry {
                key: spilled_key,
                source: GroupSource::Spilled,
            });
        }
    }

    // Sort by encoded key for deterministic output ordering, mirroring the
    // classic path. We must rebuild `key_to_group` after sorting because
    // group indices change.
    groups.sort_unstable_by_key(|g| g.key);
    key_to_group.clear();
    for (idx, g) in groups.iter().enumerate() {
        key_to_group.insert(g.key, idx);
    }

    // Materialise per-group accumulator vectors from the GPU download,
    // initialise spilled groups with the op's identity, then fold the
    // per-agg spill buffers in. The result is a flat Vec<T> of length
    // n_groups, indexed by group position.
    let n_groups = groups.len();
    let mut per_group_accs: Vec<AccDownload> =
        Vec::with_capacity(aggregate.aggregates.len());
    for (acc, agg) in acc_results.into_iter().zip(aggregate.aggregates.iter()) {
        let op = aggregate_to_op(agg);
        let folded = fold_acc_with_spill(acc, op, &groups, &key_to_group)?;
        per_group_accs.push(folded);
    }

    // Assemble the output RecordBatch.
    let m_keys = key_components.len();
    let mut arrays: Vec<ArrayRef> =
        Vec::with_capacity(m_keys + aggregate.aggregates.len());

    let key_arrays = build_key_arrays_from_entries(&groups, &key_components)?;
    for arr in key_arrays {
        arrays.push(arr);
    }

    for (i, agg) in aggregate.aggregates.iter().enumerate() {
        let out_field =
            aggregate.output_schema.fields.get(m_keys + i).ok_or_else(|| {
                BoltError::Other(format!(
                    "execute_groupby_valid: output_schema missing field for aggregate index {}",
                    i
                ))
            })?;
        let arr =
            build_agg_array_from_per_group(agg, out_field, &per_group_accs[i], n_groups)?;
        arrays.push(arr);
    }

    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
        BoltError::Other(format!(
            "failed to build GROUP BY (valid-flag) RecordBatch: {e}"
        ))
    })
}

/// Where a group came from in the GPU pipeline.
#[derive(Debug, Clone, Copy)]
enum GroupSource {
    /// Key was committed to `keys_table[slot]` by the keys kernel.
    Slot(usize),
    /// Key only appears in the keys-kernel spill buffer; no GPU slot.
    Spilled,
}

/// One row of the final groups list. Combines GPU-committed and spilled
/// keys into a single uniform representation.
#[derive(Debug, Clone, Copy)]
struct GroupEntry {
    /// The packed i64 key — same encoding as `pack_keys` produces.
    key: i64,
    /// Where the host should pull / initialise this group's accumulator.
    source: GroupSource,
}

// ---------------------------------------------------------------------------
// Key column extraction, packing, and decoding.
//
// Duplicated from `crate::exec::groupby` because the helpers there are
// private to that module. Keeping a local copy avoids cross-module
// surgery; the two implementations are intentionally kept in lockstep.
// ---------------------------------------------------------------------------

/// Number of bits a column's encoded value occupies inside a packed i64 key.
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

/// Per-group-by-column metadata: dtype + bit offset inside the i64 key.
#[derive(Debug, Clone)]
struct KeyComponent {
    /// Original dtype as declared by the plan (drives encode/decode).
    original_dtype: DataType,
    /// Bit position within the i64 key — low = 0, high = 32 for the
    /// two-column pack. Single-column keys always use offset 0.
    bit_offset: u32,
}

/// Output of [`pack_keys`]: encoded i64 column + per-component metadata.
#[derive(Debug)]
struct PackedKeys {
    /// i64-encoded keys ready to upload, one entry per input row. NULL-key
    /// rows (per `key_valid`) carry an undefined bit pattern at their slot
    /// because the per-column loader propagates garbage values from the
    /// Arrow values buffer at NULL positions; callers MUST filter via
    /// `key_valid` before forwarding to the GPU.
    keys_i64: Vec<i64>,
    /// Per-column dtype + position, in pack order.
    components: Vec<KeyComponent>,
    /// Per-row keep mask: `true` iff every GROUP BY key column is non-null
    /// at that row. `None` means "all key columns have zero nulls" — the
    /// fast path where no filtering is required. See `load_key_column_bits`.
    key_valid: Option<Vec<bool>>,
}

/// Load one group-by column's bit pattern (zero-extended into u64) from
/// `batch`, plus an optional per-row validity mask (`Some(false)` at NULL
/// positions).
///
/// H1 NULL-mask fix mirrored from `crate::exec::groupby::load_key_column_bits`.
/// When the input array has no NULLs the returned mask is `None` (saves a
/// per-row allocation in the common path). Callers must treat `None` as
/// "all valid". For NULL positions the raw u64 bit pattern is whatever
/// happens to live in the Arrow values buffer — callers MUST drop those
/// rows before they reach the GPU keys kernel, otherwise garbage bytes
/// from the NULL slots will form a fake group. This is doubly important
/// for the sentinel-free path: the orchestrator dispatches here precisely
/// when keys may collide with `i64::MIN`, so we cannot rely on the classic
/// path's "sentinel-equals-empty" reject either.
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
            // Review C12: canonicalise -0.0 -> +0.0 so signed-zero pairs
            // hash into one group. NaN bit patterns preserved.
            pa.values()
                .iter()
                .map(|&v| canonicalise_f32(v).to_bits() as u64)
                .collect()
        }
        DataType::Float64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&key_io.name, "Float64"))?;
            // Review C12: same signed-zero canonicalisation as Float32.
            pa.values()
                .iter()
                .map(|&v| canonicalise_f64(v).to_bits())
                .collect()
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

/// Canonicalise `-0.0` to `+0.0` so signed-zero pairs key the same
/// group. Preserves every other bit pattern (NaN payloads stay verbatim
/// — `x == 0.0` is false for any NaN). Mirrors the same expression in
/// `distinct.rs` and `groupby.rs` so DISTINCT, GROUP BY, and JOIN share
/// one float-equality relation (review C12).
#[inline]
fn canonicalise_f64(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x }
}

/// `f32` analogue of [`canonicalise_f64`].
#[inline]
fn canonicalise_f32(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { x }
}

/// Encode each group-by column into a single i64 key per row. Mirrors the
/// supported packings of `groupby::pack_keys`.
fn pack_keys(
    aggregate: &AggregateSpec,
    batch: &RecordBatch,
) -> BoltResult<PackedKeys> {
    if aggregate.group_by.is_empty() {
        return Err(BoltError::Other(
            "pack_keys: aggregate has no GROUP BY columns".into(),
        ));
    }

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
            original_dtype: io.dtype,
            bit_offset: 0,
        });
        if let Some(m) = mask {
            combined_mask = Some(m);
        }
        bit_streams.push(bits);
    } else {
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
            original_dtype: io_hi.dtype,
            bit_offset: 32,
        });
        components.push(KeyComponent {
            original_dtype: io_lo.dtype,
            bit_offset: 0,
        });
        bit_streams.push(bits_hi);
        bit_streams.push(bits_lo);
        // Combine: a row is valid iff every key column is non-null at that row.
        combined_mask = and_masks(mask_hi, mask_lo);
    }

    let n_rows = bit_streams[0].len();
    let mut keys_i64: Vec<i64> = vec![0i64; n_rows];
    for (comp, stream) in components.iter().zip(bit_streams.iter()) {
        let shift = comp.bit_offset;
        for (i, &raw) in stream.iter().enumerate() {
            let packed = (raw << shift) as i64;
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

/// Reverse of `pack_keys` for a single packed i64 key.
fn decode_key(packed: i64, components: &[KeyComponent]) -> Vec<KeyValue> {
    let mut out: Vec<KeyValue> = Vec::with_capacity(components.len());
    let u = packed as u64;
    for comp in components {
        let raw = match comp.original_dtype {
            DataType::Int32 | DataType::Float32 => {
                ((u >> comp.bit_offset) & 0xFFFF_FFFFu64) as u64
            }
            DataType::Int64 | DataType::Float64 => u,
            DataType::Bool | DataType::Utf8 => 0,
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

/// Count of distinct values in `keys`. O(n) via HashSet.
fn unique_count(keys: &[i64]) -> usize {
    let mut set: HashSet<i64> = HashSet::with_capacity(keys.len().min(1 << 20));
    for &k in keys {
        set.insert(k);
    }
    set.len()
}

/// Smallest power of two >= `n`.
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p: usize = 1;
    while p < n {
        match p.checked_mul(2) {
            Some(v) => p = v,
            None => return p,
        }
    }
    p
}

// ---------------------------------------------------------------------------
// Kernel launches.
// ---------------------------------------------------------------------------

/// Launch the sentinel-free keys-only kernel. Writes to both `keys_table`
/// and `slot_valid`, and (on bounded-probe overflow) to `spill_keys` +
/// `spill_counter`. See [`crate::jit::valid_flag_kernels::compile_keys_valid_kernel`]
/// for the full ABI.
#[allow(clippy::too_many_arguments)]
fn launch_keys_kernel(
    group_col: &GpuVec<i64>,
    keys_table: &mut GpuVec<i64>,
    slot_valid: &mut GpuVec<u32>,
    spill_keys: &mut GpuVec<i64>,
    spill_counter: &mut GpuVec<u32>,
    max_spill: u32,
    n_rows: usize,
    k_u32: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    if n_rows == 0 {
        return Ok(());
    }

    let module = module_cache::get_or_build_module(
        module_path!(),
        "keys_valid".to_string(),
        None,
        || compile_keys_valid_kernel(),
    )?;
    let function = module.function(VALID_KEYS_KERNEL_ENTRY)?;

    let mut group_ptr: CUdeviceptr = group_col.device_ptr();
    let mut keys_ptr: CUdeviceptr = keys_table.device_ptr();
    let mut valid_ptr: CUdeviceptr = slot_valid.device_ptr();
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let mut k_param: u32 = k_u32;
    let mut spill_keys_ptr: CUdeviceptr = spill_keys.device_ptr();
    let mut spill_counter_ptr: CUdeviceptr = spill_counter.device_ptr();
    let mut max_spill_param: u32 = max_spill;

    let mut params: [*mut c_void; 8] = [
        &mut group_ptr as *mut CUdeviceptr as *mut c_void,
        &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut valid_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
        &mut k_param as *mut u32 as *mut c_void,
        &mut spill_keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut spill_counter_ptr as *mut CUdeviceptr as *mut c_void,
        &mut max_spill_param as *mut u32 as *mut c_void,
    ];

    let block = valid_block_size();
    let grid_x = grid_x_for(n_rows_to_u32(n_rows)?, block);

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
    let _ = (group_ptr, keys_ptr, valid_ptr, spill_keys_ptr, spill_counter_ptr);
    Ok(())
}

/// Launch one aggregate-update kernel for a (typed) input column.
///
/// ABI matrix:
///
/// | path                                | params | entry symbol                                        |
/// |-------------------------------------|--------|-----------------------------------------------------|
/// | integer / float-SUM, no validity    | 11     | [`VALID_AGG_KERNEL_ENTRY`]                          |
/// | float MIN/MAX, no validity          | 7      | [`crate::jit::valid_flag_float::VALID_AGG_FLOAT_KERNEL_ENTRY`] |
/// | integer / float-SUM, with validity  | 12     | [`crate::jit::valid_flag_kernels::VALID_AGG_KERNEL_WITH_VALIDITY_ENTRY`] |
/// | float MIN/MAX, with validity        | 12     | [`crate::jit::valid_flag_float::VALID_AGG_FLOAT_WITH_VALIDITY_ENTRY`] |
///
/// The float MIN/MAX kernel cannot spill (sm_70 has no native
/// `atom.global.{min,max}.f*` so it uses a CAS retry loop that resolves
/// in-place); its non-validity ABI is therefore 7 params, the validity
/// variant 12 (matching the integer with-validity ABI but with the
/// spill block and validity_ptr reshuffled — see the kernel doc
/// comments for the exact ordering).
///
/// PV-stage-f: when `validity_ptr` is `Some(_)`, the
/// `_with_validity` PTX entry is used and [`NATIVE_VALIDITY_LAUNCHES`]
/// is incremented for inline-test observability. The host-strip
/// fallback remains correct for the `None` path.
#[allow(clippy::too_many_arguments)]
fn launch_agg_kernel<T: Pod>(
    op: ReduceOp,
    input_dtype: DataType,
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    slot_valid: &GpuVec<u32>,
    input_col: &GpuVec<T>,
    acc_table: &mut GpuVec<T>,
    spill_keys: &mut GpuVec<i64>,
    spill_values: &mut GpuVec<T>,
    spill_counter: &mut GpuVec<u32>,
    max_spill: u32,
    n_rows: usize,
    k_u32: u32,
    stream: &CudaStream,
    validity_ptr: Option<CUdeviceptr>,
) -> BoltResult<()> {
    if n_rows == 0 {
        return Ok(());
    }

    // Dispatch: native PTX atomics handle integer SUM/MIN/MAX and float SUM
    // via `compile_agg_valid_kernel`. sm_70 has no `atom.global.{min,max}.f*`,
    // so float MIN/MAX go to the CAS-loop kernel in `valid_flag_float`. Both
    // kernels expose the same `VALID_AGG_KERNEL_ENTRY` symbol so the rest of
    // the launch path is identical.
    let is_float_min_max = matches!(
        (op, input_dtype),
        (ReduceOp::Min, DataType::Float32)
            | (ReduceOp::Max, DataType::Float32)
            | (ReduceOp::Min, DataType::Float64)
            | (ReduceOp::Max, DataType::Float64)
    );
    let use_validity = validity_ptr.is_some();
    let entry_symbol = match (is_float_min_max, use_validity) {
        (true, true) => crate::jit::valid_flag_float::VALID_AGG_FLOAT_WITH_VALIDITY_ENTRY,
        (true, false) => VALID_AGG_KERNEL_ENTRY,
        (false, true) => crate::jit::valid_flag_kernels::VALID_AGG_KERNEL_WITH_VALIDITY_ENTRY,
        (false, false) => VALID_AGG_KERNEL_ENTRY,
    };
    let module = module_cache::get_or_build_module(
        module_path!(),
        format!(
            "agg_valid:{:?}:{:?}:float_min_max={}:validity={}",
            op, input_dtype, is_float_min_max, use_validity
        ),
        None,
        || match (is_float_min_max, use_validity) {
            (true, true) => {
                crate::jit::valid_flag_float::compile_agg_valid_float_kernel_with_validity(
                    op,
                    input_dtype,
                )
            }
            (true, false) => {
                crate::jit::valid_flag_float::compile_agg_valid_float_kernel(op, input_dtype)
            }
            (false, true) => compile_agg_valid_kernel_with_validity(op, input_dtype),
            (false, false) => compile_agg_valid_kernel(op, input_dtype),
        },
    )?;
    let function = module.function(entry_symbol)?;

    let mut group_ptr: CUdeviceptr = group_col.device_ptr();
    let mut keys_ptr: CUdeviceptr = keys_table.device_ptr();
    let mut valid_ptr: CUdeviceptr = slot_valid.device_ptr();
    let mut input_ptr: CUdeviceptr = input_col.device_ptr();
    let mut acc_ptr: CUdeviceptr = acc_table.device_ptr();
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let mut k_param: u32 = k_u32;
    let mut spill_keys_ptr: CUdeviceptr = spill_keys.device_ptr();
    let mut spill_values_ptr: CUdeviceptr = spill_values.device_ptr();
    let mut spill_counter_ptr: CUdeviceptr = spill_counter.device_ptr();
    let mut max_spill_param: u32 = max_spill;

    // Assemble the per-variant param vector. Four shapes (see ABI matrix
    // in the doc comment): no-validity {7 or 11} and with-validity
    // {12}. The kernel doc comments enumerate the per-shape param order.
    let mut vptr: CUdeviceptr = validity_ptr.unwrap_or(0);
    let block = valid_block_size();
    let grid_x = grid_x_for(n_rows_to_u32(n_rows)?, block);

    if use_validity {
        // Account the native-dispatch launch for test observability.
        NATIVE_VALIDITY_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if is_float_min_max {
            // Float MIN/MAX with-validity ABI (12 params):
            //   group, keys, slot_valid, input, acc,
            //   spill_keys, spill_values, spill_counter, n_rows, k,
            //   max_spill, validity_ptr
            let mut params: [*mut c_void; 12] = [
                &mut group_ptr as *mut CUdeviceptr as *mut c_void,
                &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
                &mut valid_ptr as *mut CUdeviceptr as *mut c_void,
                &mut input_ptr as *mut CUdeviceptr as *mut c_void,
                &mut acc_ptr as *mut CUdeviceptr as *mut c_void,
                &mut spill_keys_ptr as *mut CUdeviceptr as *mut c_void,
                &mut spill_values_ptr as *mut CUdeviceptr as *mut c_void,
                &mut spill_counter_ptr as *mut CUdeviceptr as *mut c_void,
                &mut n_rows_u32 as *mut u32 as *mut c_void,
                &mut k_param as *mut u32 as *mut c_void,
                &mut max_spill_param as *mut u32 as *mut c_void,
                &mut vptr as *mut CUdeviceptr as *mut c_void,
            ];
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
        } else {
            // Integer / float-SUM with-validity ABI (12 params):
            //   group, keys, slot_valid, input, acc, n_rows, k,
            //   validity_ptr, spill_keys, spill_values, spill_counter,
            //   max_spill
            let mut params: [*mut c_void; 12] = [
                &mut group_ptr as *mut CUdeviceptr as *mut c_void,
                &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
                &mut valid_ptr as *mut CUdeviceptr as *mut c_void,
                &mut input_ptr as *mut CUdeviceptr as *mut c_void,
                &mut acc_ptr as *mut CUdeviceptr as *mut c_void,
                &mut n_rows_u32 as *mut u32 as *mut c_void,
                &mut k_param as *mut u32 as *mut c_void,
                &mut vptr as *mut CUdeviceptr as *mut c_void,
                &mut spill_keys_ptr as *mut CUdeviceptr as *mut c_void,
                &mut spill_values_ptr as *mut CUdeviceptr as *mut c_void,
                &mut spill_counter_ptr as *mut CUdeviceptr as *mut c_void,
                &mut max_spill_param as *mut u32 as *mut c_void,
            ];
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
        }
    } else if is_float_min_max {
        let mut params: [*mut c_void; 7] = [
            &mut group_ptr as *mut CUdeviceptr as *mut c_void,
            &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
            &mut valid_ptr as *mut CUdeviceptr as *mut c_void,
            &mut input_ptr as *mut CUdeviceptr as *mut c_void,
            &mut acc_ptr as *mut CUdeviceptr as *mut c_void,
            &mut n_rows_u32 as *mut u32 as *mut c_void,
            &mut k_param as *mut u32 as *mut c_void,
        ];
        // Defense in depth: the float-MIN/MAX kernel ABI is exactly 7
        // params; if a future refactor desyncs this we want a loud panic
        // in debug builds rather than the CUDA driver reading garbage
        // into trailing slots.
        debug_assert_eq!(
            params.len(),
            7,
            "float-MIN/MAX kernel ABI requires exactly 7 params"
        );
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
        // Touch the spill-related host pointers post-launch so the borrow
        // checker doesn't complain about unused `mut` for variables the
        // float path intentionally drops.
        let _ = (spill_keys_ptr, spill_values_ptr, spill_counter_ptr, max_spill_param);
    } else {
        let mut params: [*mut c_void; 11] = [
            &mut group_ptr as *mut CUdeviceptr as *mut c_void,
            &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
            &mut valid_ptr as *mut CUdeviceptr as *mut c_void,
            &mut input_ptr as *mut CUdeviceptr as *mut c_void,
            &mut acc_ptr as *mut CUdeviceptr as *mut c_void,
            &mut n_rows_u32 as *mut u32 as *mut c_void,
            &mut k_param as *mut u32 as *mut c_void,
            &mut spill_keys_ptr as *mut CUdeviceptr as *mut c_void,
            &mut spill_values_ptr as *mut CUdeviceptr as *mut c_void,
            &mut spill_counter_ptr as *mut CUdeviceptr as *mut c_void,
            &mut max_spill_param as *mut u32 as *mut c_void,
        ];
        // Defense in depth: the integer / float-SUM agg kernel ABI is
        // exactly 11 params.
        debug_assert_eq!(
            params.len(),
            11,
            "integer/float-SUM kernel ABI requires exactly 11 params"
        );
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
    }
    stream.synchronize()?;
    let _ = (
        group_ptr,
        keys_ptr,
        valid_ptr,
        input_ptr,
        acc_ptr,
        spill_keys_ptr,
        spill_values_ptr,
        spill_counter_ptr,
        vptr,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-aggregate plumbing.
// ---------------------------------------------------------------------------

/// Downloaded accumulator table for a single aggregate, plus any rows
/// that the kernel had to spill instead of folding directly into the
/// table. The host applies `spill` row-by-row to the per-group result
/// after the keys-table walk is done; see [`fold_acc_with_spill`].
enum AccDownload {
    I32 {
        gpu_acc: Vec<i32>,
        spill: Vec<(i64, i32)>,
    },
    I64 {
        gpu_acc: Vec<i64>,
        spill: Vec<(i64, i64)>,
    },
    F32 {
        gpu_acc: Vec<f32>,
        spill: Vec<(i64, f32)>,
    },
    F64 {
        gpu_acc: Vec<f64>,
        spill: Vec<(i64, f64)>,
    },
    Avg {
        sum: Vec<f64>,
        count: Vec<i64>,
        sum_spill: Vec<(i64, f64)>,
        count_spill: Vec<(i64, i64)>,
    },
}

/// Allocate the (keys, values, counter) spill triple for a typed agg
/// kernel launch. `T` matches the input column dtype.
fn alloc_agg_spill<T: Pod + Default>(
) -> BoltResult<(GpuVec<i64>, GpuVec<T>, GpuVec<u32>)> {
    let keys = GpuVec::<i64>::from_slice(&vec![SPILL_EMPTY_KEY; SPILL_CAPACITY])?;
    // `Pod: Copy` gives us Clone for free; the explicit vec! lets the
    // value type fall back to Default at construction time.
    let values_init: Vec<T> = vec![T::default(); SPILL_CAPACITY];
    let values = GpuVec::<T>::from_slice(&values_init)?;
    let counter = GpuVec::<u32>::from_slice(&[0u32])?;
    Ok((keys, values, counter))
}

/// Download a kernel's spill buffer, returning the `(key, value)` pairs
/// the kernel actually wrote. Errors out if the counter exceeded the
/// capacity (which would mean rows were silently dropped on the device).
fn download_agg_spill<T: Pod>(
    spill_keys: GpuVec<i64>,
    spill_values: GpuVec<T>,
    spill_counter: GpuVec<u32>,
    label: &str,
) -> BoltResult<Vec<(i64, T)>> {
    let count_raw = spill_counter.to_vec()?[0] as usize;
    if count_raw > SPILL_CAPACITY {
        return Err(BoltError::Other(format!(
            "GROUP BY spill overflow ({} rows lost from {}); \
             increase load factor or SPILL_CAPACITY",
            count_raw - SPILL_CAPACITY,
            label
        )));
    }
    let count = count_raw.min(SPILL_CAPACITY);
    if count == 0 {
        return Ok(Vec::new());
    }
    let keys = spill_keys.to_vec()?;
    let values = spill_values.to_vec()?;
    let mut out: Vec<(i64, T)> = Vec::with_capacity(count);
    for i in 0..count {
        out.push((keys[i], values[i]));
    }
    Ok(out)
}

/// Compile + launch one aggregate kernel (or two, for AVG), download its
/// accumulator table(s), and return the result alongside any spilled
/// rows the kernel had to defer to host-side folding.
///
/// `key_valid` is the per-row keep mask produced by `pack_keys` over the
/// ORIGINAL (pre-filter) row indices — `None` means no key column has
/// nulls. When the value column itself has NULLs we logically AND the
/// masks and upload a fresh, per-aggregate filtered key column so that
/// the GPU sees only (non-NULL key, non-NULL value) pairs. `n_rows` is
/// the post-key-filter row count: it equals `group_col`'s length when we
/// reuse the shared key column.
#[allow(clippy::too_many_arguments)]
fn run_one_aggregate(
    agg: &AggregateExpr,
    inputs: &[ColumnIO],
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    slot_valid: &GpuVec<u32>,
    batch: &RecordBatch,
    n_rows: usize,
    k: usize,
    k_u32: u32,
    max_spill: u32,
    stream: &CudaStream,
    key_valid: Option<&[bool]>,
    any_input_has_validity: bool,
) -> BoltResult<AccDownload> {
    match agg {
        AggregateExpr::Sum(expr)
        | AggregateExpr::Min(expr)
        | AggregateExpr::Max(expr) => {
            let op = ReduceOp::from_agg(agg)?;
            let col_name = bare_column_name(expr)?;
            let col_io = resolve_input(inputs, col_name)?;
            run_typed_agg(
                op, col_io, group_col, keys_table, slot_valid, batch, n_rows, k,
                k_u32, max_spill, stream, key_valid, any_input_has_validity,
            )
        }

        AggregateExpr::Count(expr) => {
            // COUNT(col) excludes NULL inputs; COUNT(*) (an expression that
            // doesn't resolve to a column) counts surviving (post-key-filter)
            // rows. We synthesise an all-ones column over the (key AND value)
            // filtered rows; the only difference is whether the value-NULL
            // mask is applied. Stage-3: async H2D + pinned D2H.
            let value_valid: Option<Vec<bool>> = match bare_column_name(expr)
                .ok()
                .and_then(|name| resolve_input(inputs, name).ok())
            {
                Some(col_io) => column_null_mask(col_io, batch)?,
                None => None,
            };

            let filtered =
                prepare_filtered_keys(group_col, n_rows, key_valid, value_valid.as_deref())?;
            let count_n_rows = filtered.n_rows();

            let ones: Vec<i64> = vec![1i64; count_n_rows];
            let input_gpu = GpuVec::<i64>::from_slice_async(&ones, stream.raw())?;
            let identity_init: Vec<i64> = vec![0i64; k];
            let mut acc_table = GpuVec::<i64>::from_slice_async(&identity_init, stream.raw())?;
            let (mut spill_keys, mut spill_values, mut spill_counter) =
                alloc_agg_spill::<i64>()?;
            launch_agg_kernel::<i64>(
                ReduceOp::Sum,
                DataType::Int64,
                filtered.col(),
                keys_table,
                slot_valid,
                &input_gpu,
                &mut acc_table,
                &mut spill_keys,
                &mut spill_values,
                &mut spill_counter,
                max_spill,
                count_n_rows,
                k_u32,
                stream,
                None,
            )?;
            let gpu_acc = download_pinned_i64(&acc_table, stream)?;
            let spill = download_agg_spill(
                spill_keys,
                spill_values,
                spill_counter,
                "COUNT agg kernel",
            )?;
            Ok(AccDownload::I64 { gpu_acc, spill })
        }

        AggregateExpr::Avg(expr) => {
            // AVG = SUM(expr) / COUNT(expr), where COUNT is the non-NULL row
            // count of the value column within each group. SUM in f64; COUNT
            // in i64. Both kernels share the (key_valid ∧ value_valid) filter
            // so every contribution to the SUM increments the matching COUNT.
            let col_name = bare_column_name(expr)?;
            let col_io = resolve_input(inputs, col_name)?;

            let value_valid = column_null_mask(col_io, batch)?;
            let filtered =
                prepare_filtered_keys(group_col, n_rows, key_valid, value_valid.as_deref())?;
            let avg_n_rows = filtered.n_rows();

            // --- SUM(expr) cast to f64. Upcast host-side, drop NULL positions
            //     in the same step. Stage-3: async H2D + pinned D2H. ---
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
            let (mut sum_spill_keys, mut sum_spill_values, mut sum_spill_counter) =
                alloc_agg_spill::<f64>()?;
            launch_agg_kernel::<f64>(
                ReduceOp::Sum,
                DataType::Float64,
                filtered.col(),
                keys_table,
                slot_valid,
                &input_gpu,
                &mut sum_acc,
                &mut sum_spill_keys,
                &mut sum_spill_values,
                &mut sum_spill_counter,
                max_spill,
                avg_n_rows,
                k_u32,
                stream,
                None,
            )?;
            let sum_host = download_pinned_f64(&sum_acc, stream)?;
            let sum_spill = download_agg_spill(
                sum_spill_keys,
                sum_spill_values,
                sum_spill_counter,
                "AVG.SUM agg kernel",
            )?;

            // --- COUNT(non-null) per group. ---
            let ones: Vec<i64> = vec![1i64; avg_n_rows];
            let count_input = GpuVec::<i64>::from_slice_async(&ones, stream.raw())?;
            let count_init: Vec<i64> = vec![0i64; k];
            let mut count_acc = GpuVec::<i64>::from_slice_async(&count_init, stream.raw())?;
            let (mut cnt_spill_keys, mut cnt_spill_values, mut cnt_spill_counter) =
                alloc_agg_spill::<i64>()?;
            launch_agg_kernel::<i64>(
                ReduceOp::Sum,
                DataType::Int64,
                filtered.col(),
                keys_table,
                slot_valid,
                &count_input,
                &mut count_acc,
                &mut cnt_spill_keys,
                &mut cnt_spill_values,
                &mut cnt_spill_counter,
                max_spill,
                avg_n_rows,
                k_u32,
                stream,
                None,
            )?;
            let count_host = download_pinned_i64(&count_acc, stream)?;
            let count_spill = download_agg_spill(
                cnt_spill_keys,
                cnt_spill_values,
                cnt_spill_counter,
                "AVG.COUNT agg kernel",
            )?;

            Ok(AccDownload::Avg {
                sum: sum_host,
                count: count_host,
                sum_spill,
                count_spill,
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
/// shrunk the row set further) or a freshly-uploaded smaller column for
/// the (key_valid AND value_valid) joint mask. The variants paper over
/// Rust's borrow-checker constraints around returning `&GpuVec<i64>`
/// whose lifetime might come from local storage.
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
            FilteredKeys::Borrowed { n_rows, .. }
            | FilteredKeys::Owned { n_rows, .. } => *n_rows,
        }
    }
}

/// Decide whether to reuse the shared `group_col` (when no value-NULL
/// filter shrinks the row set further) or upload a freshly-filtered key
/// column for this aggregate. The shared `group_col` was built from a
/// `host_keys` that already had the `key_valid` rows kept; if
/// `value_valid` is `None` we can reuse it directly. Otherwise we
/// re-download once and refilter against the joint mask, then upload a
/// fresh i64 column.
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
    Ok(FilteredKeys::Owned {
        group_col: owned,
        n_rows: filtered_n,
    })
}

/// Common path for SUM/MIN/MAX.
///
/// `key_valid` (from `pack_keys`) and the value column's own validity
/// mask are AND'd together: rows where EITHER is NULL are dropped before
/// upload. This matches standard SQL semantics — NULL inputs are skipped
/// by SUM/MIN/MAX rather than coerced to 0 / dtype-min / dtype-max (which
/// is what reading the raw `.values()` buffer at NULL positions would do).
#[allow(clippy::too_many_arguments)]
fn run_typed_agg(
    op: ReduceOp,
    col_io: &ColumnIO,
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    slot_valid: &GpuVec<u32>,
    batch: &RecordBatch,
    n_rows: usize,
    k: usize,
    k_u32: u32,
    max_spill: u32,
    stream: &CudaStream,
    key_valid: Option<&[bool]>,
    any_input_has_validity: bool,
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

    // PV-stage-f: native-validity dispatch. When the planner flagged this
    // input AND the column actually carries nulls AND the (op, dtype)
    // combination has a `_with_validity` emitter (integer SUM/MIN/MAX +
    // float SUM/MIN/MAX), upload the FULL value column + the packed
    // bitmap and skip the host strip. Falls through to host-strip
    // otherwise. Requires `key_valid.is_none()` so the kernel's parallel
    // (group, value, validity) row alignment is preserved.
    if any_input_has_validity
        && key_valid.is_none()
        && column_should_use_native_validity(arr.as_ref(), op, col_io.dtype)
    {
        let vv = value_valid.as_deref().expect(
            "column_should_use_native_validity guarantees arr.null_count() > 0",
        );
        return run_typed_agg_native_validity(
            op, col_io, arr.as_ref(), vv, group_col, keys_table, slot_valid,
            n_rows, k, k_u32, max_spill, stream,
        );
    }

    let filtered =
        prepare_filtered_keys(group_col, n_rows, key_valid, value_valid.as_deref())?;
    let n = filtered.n_rows();

    match col_io.dtype {
        DataType::Int32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int32"))?;
            // SUM(Int32) widens to Int64 per
            // `crate::plan::logical_plan::sum_output_dtype` /
            // `crate::jit::agg_kernels::reduction_output_dtype`. The
            // valid-flag agg kernel emits its atomic at the same dtype it
            // loads at, so to get `atom.global.add.u64` we sign-extend the
            // input column to i64 on the host and pass `DataType::Int64`
            // to the kernel compiler. The accumulator buffer, spill values
            // buffer, and host-side combine path (`combine_i64` via
            // `AccDownload::I64`) all agree at i64. MIN/MAX preserve the
            // input dtype and keep the narrow i32 path.
            if matches!(op, ReduceOp::Sum) {
                let host: Vec<i64> = collect_filtered_primitive(
                    pa,
                    key_valid,
                    value_valid.as_deref(),
                )
                .into_iter()
                .map(|v| v as i64)
                .collect();
                debug_assert_eq!(host.len(), n);
                let input_gpu = GpuVec::<i64>::from_slice_async(&host, stream.raw())?;
                let init: Vec<i64> = vec![identity_i64(op); k];
                let mut acc = GpuVec::<i64>::from_slice_async(&init, stream.raw())?;
                let (mut spill_keys, mut spill_values, mut spill_counter) =
                    alloc_agg_spill::<i64>()?;
                launch_agg_kernel::<i64>(
                    op,
                    DataType::Int64,
                    filtered.col(),
                    keys_table,
                    slot_valid,
                    &input_gpu,
                    &mut acc,
                    &mut spill_keys,
                    &mut spill_values,
                    &mut spill_counter,
                    max_spill,
                    n,
                    k_u32,
                    stream,
                    None,
                )?;
                let gpu_acc = download_pinned_i64(&acc, stream)?;
                let spill = download_agg_spill(
                    spill_keys,
                    spill_values,
                    spill_counter,
                    "i64 agg kernel (widened from SUM(Int32))",
                )?;
                Ok(AccDownload::I64 { gpu_acc, spill })
            } else {
                let host: Vec<i32> = collect_filtered_primitive(
                    pa,
                    key_valid,
                    value_valid.as_deref(),
                );
                debug_assert_eq!(host.len(), n);
                let input_gpu = GpuVec::<i32>::from_slice_async(&host, stream.raw())?;
                let init: Vec<i32> = vec![identity_i32(op); k];
                let mut acc = GpuVec::<i32>::from_slice_async(&init, stream.raw())?;
                let (mut spill_keys, mut spill_values, mut spill_counter) =
                    alloc_agg_spill::<i32>()?;
                launch_agg_kernel::<i32>(
                    op,
                    DataType::Int32,
                    filtered.col(),
                    keys_table,
                    slot_valid,
                    &input_gpu,
                    &mut acc,
                    &mut spill_keys,
                    &mut spill_values,
                    &mut spill_counter,
                    max_spill,
                    n,
                    k_u32,
                    stream,
                    None,
                )?;
                let gpu_acc = download_pinned_i32(&acc, stream)?;
                let spill = download_agg_spill(
                    spill_keys,
                    spill_values,
                    spill_counter,
                    "i32 agg kernel",
                )?;
                Ok(AccDownload::I32 { gpu_acc, spill })
            }
        }
        DataType::Int64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int64"))?;
            let host: Vec<i64> =
                collect_filtered_primitive(pa, key_valid, value_valid.as_deref());
            debug_assert_eq!(host.len(), n);
            let input_gpu = GpuVec::<i64>::from_slice_async(&host, stream.raw())?;
            let init: Vec<i64> = vec![identity_i64(op); k];
            let mut acc = GpuVec::<i64>::from_slice_async(&init, stream.raw())?;
            let (mut spill_keys, mut spill_values, mut spill_counter) =
                alloc_agg_spill::<i64>()?;
            launch_agg_kernel::<i64>(
                op,
                DataType::Int64,
                filtered.col(),
                keys_table,
                slot_valid,
                &input_gpu,
                &mut acc,
                &mut spill_keys,
                &mut spill_values,
                &mut spill_counter,
                max_spill,
                n,
                k_u32,
                stream,
                None,
            )?;
            let gpu_acc = download_pinned_i64(&acc, stream)?;
            let spill = download_agg_spill(
                spill_keys,
                spill_values,
                spill_counter,
                "i64 agg kernel",
            )?;
            Ok(AccDownload::I64 { gpu_acc, spill })
        }
        DataType::Float32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float32"))?;
            let host: Vec<f32> =
                collect_filtered_primitive(pa, key_valid, value_valid.as_deref());
            debug_assert_eq!(host.len(), n);
            let input_gpu = GpuVec::<f32>::from_slice_async(&host, stream.raw())?;
            let init: Vec<f32> = vec![identity_f32(op); k];
            let mut acc = GpuVec::<f32>::from_slice_async(&init, stream.raw())?;
            let (mut spill_keys, mut spill_values, mut spill_counter) =
                alloc_agg_spill::<f32>()?;
            launch_agg_kernel::<f32>(
                op,
                DataType::Float32,
                filtered.col(),
                keys_table,
                slot_valid,
                &input_gpu,
                &mut acc,
                &mut spill_keys,
                &mut spill_values,
                &mut spill_counter,
                max_spill,
                n,
                k_u32,
                stream,
                None,
            )?;
            let gpu_acc = download_pinned_f32(&acc, stream)?;
            let spill = download_agg_spill(
                spill_keys,
                spill_values,
                spill_counter,
                "f32 agg kernel",
            )?;
            Ok(AccDownload::F32 { gpu_acc, spill })
        }
        DataType::Float64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float64"))?;
            let host: Vec<f64> =
                collect_filtered_primitive(pa, key_valid, value_valid.as_deref());
            debug_assert_eq!(host.len(), n);
            let input_gpu = GpuVec::<f64>::from_slice_async(&host, stream.raw())?;
            let init: Vec<f64> = vec![identity_f64(op); k];
            let mut acc = GpuVec::<f64>::from_slice_async(&init, stream.raw())?;
            let (mut spill_keys, mut spill_values, mut spill_counter) =
                alloc_agg_spill::<f64>()?;
            launch_agg_kernel::<f64>(
                op,
                DataType::Float64,
                filtered.col(),
                keys_table,
                slot_valid,
                &input_gpu,
                &mut acc,
                &mut spill_keys,
                &mut spill_values,
                &mut spill_counter,
                max_spill,
                n,
                k_u32,
                stream,
                None,
            )?;
            let gpu_acc = download_pinned_f64(&acc, stream)?;
            let spill = download_agg_spill(
                spill_keys,
                spill_values,
                spill_counter,
                "f64 agg kernel",
            )?;
            Ok(AccDownload::F64 { gpu_acc, spill })
        }
        DataType::Bool | DataType::Utf8 => Err(BoltError::Type(format!(
            "aggregate input dtype {:?} not supported (column '{}')",
            col_io.dtype, col_io.name
        ))),
    }
}

/// PV-stage-f: sentinel-free native-validity dispatch. Upload the FULL
/// value column verbatim + the packed Arrow LE bitmap and let the GPU
/// kernel skip NULL rows on the device. Mirror of `groupby.rs`'s helper
/// adapted to the sentinel-free agg kernel ABI (which carries the
/// spill triple alongside the validity pointer).
///
/// Float MIN/MAX routes through the float `_with_validity` companion;
/// integer + float SUM routes through `compile_agg_valid_kernel_with_validity`.
/// Both are wired in `launch_agg_kernel` via the four-way ABI dispatch.
///
/// Preconditions (enforced at the call site in `run_typed_agg`):
/// 1. `arr.null_count() > 0`.
/// 2. The key column has no NULL rows (`key_valid == None`), so the
///    upload is parallel-by-row to the batch.
/// 3. `(op, dtype)` is in the coverage of
///    [`column_should_use_native_validity`].
#[allow(clippy::too_many_arguments)]
fn run_typed_agg_native_validity(
    op: ReduceOp,
    col_io: &ColumnIO,
    arr: &dyn Array,
    value_valid: &[bool],
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    slot_valid: &GpuVec<u32>,
    n_rows: usize,
    k: usize,
    k_u32: u32,
    max_spill: u32,
    stream: &CudaStream,
) -> BoltResult<AccDownload> {
    debug_assert_eq!(value_valid.len(), n_rows);

    // Pack to Arrow LE bytes — `valid_flag_kernels` consumes `Vec<u8>`.
    let packed_bytes = pack_validity_bits(value_valid);
    let validity_gpu = GpuVec::<u8>::from_slice_async(&packed_bytes, stream.raw())?;
    let validity_ptr = Some(validity_gpu.device_ptr());

    match col_io.dtype {
        DataType::Int32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int32"))?;
            // SUM(Int32) widens host-side to i64 (matches sentinel-based path
            // and `sum_output_dtype`).
            if matches!(op, ReduceOp::Sum) {
                let widened: Vec<i64> = pa.values().iter().map(|&v| v as i64).collect();
                let input_gpu = GpuVec::<i64>::from_slice_async(&widened, stream.raw())?;
                let init: Vec<i64> = vec![identity_i64(op); k];
                let mut acc = GpuVec::<i64>::from_slice_async(&init, stream.raw())?;
                let (mut sk, mut sv, mut sc) = alloc_agg_spill::<i64>()?;
                launch_agg_kernel::<i64>(
                    op,
                    DataType::Int64,
                    group_col,
                    keys_table,
                    slot_valid,
                    &input_gpu,
                    &mut acc,
                    &mut sk,
                    &mut sv,
                    &mut sc,
                    max_spill,
                    n_rows,
                    k_u32,
                    stream,
                    validity_ptr,
                )?;
                let _ = validity_gpu;
                let gpu_acc = download_pinned_i64(&acc, stream)?;
                let spill = download_agg_spill(sk, sv, sc, "i64 agg kernel (widened SUM(Int32), validity)")?;
                return Ok(AccDownload::I64 { gpu_acc, spill });
            }
            let host: Vec<i32> = pa.values().to_vec();
            let input_gpu = GpuVec::<i32>::from_slice_async(&host, stream.raw())?;
            let init: Vec<i32> = vec![identity_i32(op); k];
            let mut acc = GpuVec::<i32>::from_slice_async(&init, stream.raw())?;
            let (mut sk, mut sv, mut sc) = alloc_agg_spill::<i32>()?;
            launch_agg_kernel::<i32>(
                op,
                DataType::Int32,
                group_col,
                keys_table,
                slot_valid,
                &input_gpu,
                &mut acc,
                &mut sk,
                &mut sv,
                &mut sc,
                max_spill,
                n_rows,
                k_u32,
                stream,
                validity_ptr,
            )?;
            let _ = validity_gpu;
            let gpu_acc = download_pinned_i32(&acc, stream)?;
            let spill = download_agg_spill(sk, sv, sc, "i32 agg kernel (validity)")?;
            Ok(AccDownload::I32 { gpu_acc, spill })
        }
        DataType::Int64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int64"))?;
            let host: Vec<i64> = pa.values().to_vec();
            let input_gpu = GpuVec::<i64>::from_slice_async(&host, stream.raw())?;
            let init: Vec<i64> = vec![identity_i64(op); k];
            let mut acc = GpuVec::<i64>::from_slice_async(&init, stream.raw())?;
            let (mut sk, mut sv, mut sc) = alloc_agg_spill::<i64>()?;
            launch_agg_kernel::<i64>(
                op,
                DataType::Int64,
                group_col,
                keys_table,
                slot_valid,
                &input_gpu,
                &mut acc,
                &mut sk,
                &mut sv,
                &mut sc,
                max_spill,
                n_rows,
                k_u32,
                stream,
                validity_ptr,
            )?;
            let _ = validity_gpu;
            let gpu_acc = download_pinned_i64(&acc, stream)?;
            let spill = download_agg_spill(sk, sv, sc, "i64 agg kernel (validity)")?;
            Ok(AccDownload::I64 { gpu_acc, spill })
        }
        DataType::Float32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float32"))?;
            let host: Vec<f32> = pa.values().to_vec();
            let input_gpu = GpuVec::<f32>::from_slice_async(&host, stream.raw())?;
            let init: Vec<f32> = vec![identity_f32(op); k];
            let mut acc = GpuVec::<f32>::from_slice_async(&init, stream.raw())?;
            let (mut sk, mut sv, mut sc) = alloc_agg_spill::<f32>()?;
            launch_agg_kernel::<f32>(
                op,
                DataType::Float32,
                group_col,
                keys_table,
                slot_valid,
                &input_gpu,
                &mut acc,
                &mut sk,
                &mut sv,
                &mut sc,
                max_spill,
                n_rows,
                k_u32,
                stream,
                validity_ptr,
            )?;
            let _ = validity_gpu;
            let gpu_acc = download_pinned_f32(&acc, stream)?;
            let spill = download_agg_spill(sk, sv, sc, "f32 agg kernel (validity)")?;
            Ok(AccDownload::F32 { gpu_acc, spill })
        }
        DataType::Float64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float64"))?;
            let host: Vec<f64> = pa.values().to_vec();
            let input_gpu = GpuVec::<f64>::from_slice_async(&host, stream.raw())?;
            let init: Vec<f64> = vec![identity_f64(op); k];
            let mut acc = GpuVec::<f64>::from_slice_async(&init, stream.raw())?;
            let (mut sk, mut sv, mut sc) = alloc_agg_spill::<f64>()?;
            launch_agg_kernel::<f64>(
                op,
                DataType::Float64,
                group_col,
                keys_table,
                slot_valid,
                &input_gpu,
                &mut acc,
                &mut sk,
                &mut sv,
                &mut sc,
                max_spill,
                n_rows,
                k_u32,
                stream,
                validity_ptr,
            )?;
            let _ = validity_gpu;
            let gpu_acc = download_pinned_f64(&acc, stream)?;
            let spill = download_agg_spill(sk, sv, sc, "f64 agg kernel (validity)")?;
            Ok(AccDownload::F64 { gpu_acc, spill })
        }
        DataType::Bool | DataType::Utf8 => Err(BoltError::Type(format!(
            "native-validity dispatch reached unsupported dtype {:?} (column '{}')",
            col_io.dtype, col_io.name
        ))),
    }
}

/// Collect a primitive Arrow array's values into a fresh `Vec`, filtering
/// out positions where (key_valid AND value_valid) is false. Either mask
/// being `None` means "all true" for that side. The output length equals
/// the post-filter row count.
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

/// Pull a numeric input column out of `batch`, upcast each element to
/// f64, and drop positions where (key_valid AND value_valid) is false.
/// Either mask being `None` means "all true" for that side. Used by AVG
/// so the numerator and denominator stay aligned with the (key-NULL,
/// value-NULL) filter applied to the rest of the launch.
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

/// Helper for `load_input_column_as_f64_filtered`: walks a primitive
/// Arrow array, applies the joint key/value validity mask, and casts each
/// surviving element via `f`.
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

/// Build one Arrow array per group-by column by decoding each packed i64
/// key back through `decode_key`. Works against the post-spill `GroupEntry`
/// list, which mixes GPU-committed and host-only (spilled) groups.
fn build_key_arrays_from_entries(
    groups: &[GroupEntry],
    components: &[KeyComponent],
) -> BoltResult<Vec<ArrayRef>> {
    let m = components.len();
    let n = groups.len();

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

    for g in groups {
        let decoded = decode_key(g.key, components);
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

/// Build the output Arrow array for one aggregate from its already-folded
/// per-group accumulator vector (length `n_groups`, indexed by position
/// in the final `GroupEntry` list).
fn build_agg_array_from_per_group(
    agg: &AggregateExpr,
    out_field: &Field,
    acc: &AccDownload,
    n_groups: usize,
) -> BoltResult<ArrayRef> {
    match (agg, acc) {
        (AggregateExpr::Count(_), AccDownload::I64 { gpu_acc, .. }) => {
            debug_assert_eq!(gpu_acc.len(), n_groups);
            pack_array(out_field.dtype, Scalars::I64(gpu_acc.clone()))
        }
        (
            AggregateExpr::Avg(_),
            AccDownload::Avg {
                sum,
                count,
                ..
            },
        ) => {
            debug_assert_eq!(sum.len(), n_groups);
            debug_assert_eq!(count.len(), n_groups);
            // SQL spec (see docs/SQL_REFERENCE.md): aggregate functions over
            // an all-NULL group return NULL. For AVG specifically, a group
            // whose count is 0 means every input row in that group was NULL,
            // so the result must be NULL rather than 0.0 / NaN.
            //
            // We therefore bypass `pack_array` here (which only emits
            // non-nullable arrays) and build a nullable Float64Array directly
            // via `from_iter(Option<f64>)`. The output schema field already
            // has `nullable = true` for AVG (the schema-construction path in
            // logical_plan marks AVG outputs nullable), so this surface
            // change is transparent to engine.rs, which simply pulls the
            // ArrayRef out of the RecordBatch.
            //
            // Only handle DataType::Float64 here — AVG's output dtype is
            // always Float64 per the planner contract. If a future plan ever
            // emits a non-Float64 AVG output we'd need to widen this.
            match out_field.dtype {
                DataType::Float64 => {
                    let iter = (0..n_groups).map(|i| {
                        let c = count[i];
                        if c == 0 {
                            None
                        } else {
                            Some(sum[i] / (c as f64))
                        }
                    });
                    Ok(Arc::new(Float64Array::from_iter(iter)) as ArrayRef)
                }
                other => Err(BoltError::Type(format!(
                    "GROUP BY (valid-flag): AVG output dtype must be Float64, got {:?}",
                    other
                ))),
            }
        }
        (AggregateExpr::Sum(_) | AggregateExpr::Min(_) | AggregateExpr::Max(_), other) => {
            // NOTE on SQL NULL-group semantics: per docs/SQL_REFERENCE.md,
            // SUM/MIN/MAX over an all-NULL group should return NULL. In this
            // GROUP BY path a group only exists because at least one input
            // row carried that key (groups are formed FROM the input rows),
            // so a structurally-empty group cannot occur here — the
            // initialiser identity is never observed in the output. The
            // separate scalar (no-GROUP-BY) aggregate path in
            // `extended_agg.rs` handles the all-NULL whole-table case and
            // owns that NULL-projection logic. If a future change ever
            // introduces a pre-filter that can zero-out a group's rows,
            // those branches must be revisited to emit a nullable array
            // (analogous to the AVG branch above).
            //
            // MIN(bool)/MAX(bool) are also a NULL concern per the spec, but
            // this file rejects Bool aggregate inputs at run_typed_agg
            // (search for "aggregate input dtype" in this file) — the bool
            // aggregate path lives in extended_agg.rs, which is out of scope
            // for this module.
            let scalars = match other {
                AccDownload::I32 { gpu_acc, .. } => Scalars::I32(gpu_acc.clone()),
                AccDownload::I64 { gpu_acc, .. } => Scalars::I64(gpu_acc.clone()),
                AccDownload::F32 { gpu_acc, .. } => Scalars::F32(gpu_acc.clone()),
                AccDownload::F64 { gpu_acc, .. } => Scalars::F64(gpu_acc.clone()),
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

/// Map a logical AggregateExpr to its reduce op. AVG decomposes into
/// SUM + COUNT internally; this returns the op of the "primary" reduction
/// for spill-folding purposes (the secondary COUNT spill is folded with
/// `ReduceOp::Sum`).
fn aggregate_to_op(agg: &AggregateExpr) -> ReduceOp {
    match agg {
        AggregateExpr::Sum(_) => ReduceOp::Sum,
        AggregateExpr::Min(_) => ReduceOp::Min,
        AggregateExpr::Max(_) => ReduceOp::Max,
        AggregateExpr::Count(_) => ReduceOp::Sum, // count = sum of ones
        AggregateExpr::Avg(_) => ReduceOp::Sum,   // AVG sum-side; count uses Sum too
    }
}

/// Re-shape an `AccDownload` from "indexed by GPU slot" to "indexed by
/// post-sort group position", folding any spilled rows in as we go.
///
/// For each spilled `(key, value)`:
///   * find the group's position via `key_to_group` (the keys-kernel spill
///     fold already added a `Spilled` entry for any key the GPU never
///     committed, so every spilled key MUST resolve here);
///   * apply the op to the per-group accumulator slot.
fn fold_acc_with_spill(
    acc: AccDownload,
    op: ReduceOp,
    groups: &[GroupEntry],
    key_to_group: &HashMap<i64, usize>,
) -> BoltResult<AccDownload> {
    match acc {
        AccDownload::I32 { gpu_acc, spill } => {
            let per_group = reindex_i32(&gpu_acc, groups, op);
            let folded = apply_spill_i32(per_group, &spill, op, key_to_group)?;
            Ok(AccDownload::I32 {
                gpu_acc: folded,
                spill: Vec::new(),
            })
        }
        AccDownload::I64 { gpu_acc, spill } => {
            let per_group = reindex_i64(&gpu_acc, groups, op);
            let folded = apply_spill_i64(per_group, &spill, op, key_to_group)?;
            Ok(AccDownload::I64 {
                gpu_acc: folded,
                spill: Vec::new(),
            })
        }
        AccDownload::F32 { gpu_acc, spill } => {
            let per_group = reindex_f32(&gpu_acc, groups, op);
            let folded = apply_spill_f32(per_group, &spill, op, key_to_group)?;
            Ok(AccDownload::F32 {
                gpu_acc: folded,
                spill: Vec::new(),
            })
        }
        AccDownload::F64 { gpu_acc, spill } => {
            let per_group = reindex_f64(&gpu_acc, groups, op);
            let folded = apply_spill_f64(per_group, &spill, op, key_to_group)?;
            Ok(AccDownload::F64 {
                gpu_acc: folded,
                spill: Vec::new(),
            })
        }
        AccDownload::Avg {
            sum,
            count,
            sum_spill,
            count_spill,
        } => {
            // AVG = SUM(value) / COUNT(*). Both sub-accumulators use SUM
            // semantics (the count-side adds 1 per row).
            let per_group_sum = reindex_f64(&sum, groups, ReduceOp::Sum);
            let per_group_count = reindex_i64(&count, groups, ReduceOp::Sum);
            let folded_sum =
                apply_spill_f64(per_group_sum, &sum_spill, ReduceOp::Sum, key_to_group)?;
            let folded_count = apply_spill_i64(
                per_group_count,
                &count_spill,
                ReduceOp::Sum,
                key_to_group,
            )?;
            Ok(AccDownload::Avg {
                sum: folded_sum,
                count: folded_count,
                sum_spill: Vec::new(),
                count_spill: Vec::new(),
            })
        }
    }
}

// Re-index helpers: produce a Vec<T> of length groups.len() where index `i`
// is the per-group accumulator for `groups[i]`. For `Slot(idx)` groups we
// copy the GPU value at `gpu_acc[idx]`; for `Spilled` groups we start at
// the op's identity.
fn reindex_i32(gpu_acc: &[i32], groups: &[GroupEntry], op: ReduceOp) -> Vec<i32> {
    let mut out = Vec::with_capacity(groups.len());
    let id = identity_i32(op);
    for g in groups {
        out.push(match g.source {
            GroupSource::Slot(idx) => gpu_acc[idx],
            GroupSource::Spilled => id,
        });
    }
    out
}
fn reindex_i64(gpu_acc: &[i64], groups: &[GroupEntry], op: ReduceOp) -> Vec<i64> {
    let mut out = Vec::with_capacity(groups.len());
    let id = identity_i64(op);
    for g in groups {
        out.push(match g.source {
            GroupSource::Slot(idx) => gpu_acc[idx],
            GroupSource::Spilled => id,
        });
    }
    out
}
fn reindex_f32(gpu_acc: &[f32], groups: &[GroupEntry], op: ReduceOp) -> Vec<f32> {
    let mut out = Vec::with_capacity(groups.len());
    let id = identity_f32(op);
    for g in groups {
        out.push(match g.source {
            GroupSource::Slot(idx) => gpu_acc[idx],
            GroupSource::Spilled => id,
        });
    }
    out
}
fn reindex_f64(gpu_acc: &[f64], groups: &[GroupEntry], op: ReduceOp) -> Vec<f64> {
    let mut out = Vec::with_capacity(groups.len());
    let id = identity_f64(op);
    for g in groups {
        out.push(match g.source {
            GroupSource::Slot(idx) => gpu_acc[idx],
            GroupSource::Spilled => id,
        });
    }
    out
}

// Spill-apply helpers: for each (key, value), look up the group index and
// apply the op. A spilled key that doesn't resolve to a group is a bug —
// the keys-kernel spill fold should have created an entry for it.
fn apply_spill_i32(
    mut acc: Vec<i32>,
    spill: &[(i64, i32)],
    op: ReduceOp,
    key_to_group: &HashMap<i64, usize>,
) -> BoltResult<Vec<i32>> {
    for &(key, val) in spill {
        let idx = key_to_group.get(&key).copied().ok_or_else(|| spill_lookup_err(key))?;
        acc[idx] = combine_i32(op, acc[idx], val);
    }
    Ok(acc)
}
fn apply_spill_i64(
    mut acc: Vec<i64>,
    spill: &[(i64, i64)],
    op: ReduceOp,
    key_to_group: &HashMap<i64, usize>,
) -> BoltResult<Vec<i64>> {
    for &(key, val) in spill {
        let idx = key_to_group.get(&key).copied().ok_or_else(|| spill_lookup_err(key))?;
        acc[idx] = combine_i64(op, acc[idx], val);
    }
    Ok(acc)
}
fn apply_spill_f32(
    mut acc: Vec<f32>,
    spill: &[(i64, f32)],
    op: ReduceOp,
    key_to_group: &HashMap<i64, usize>,
) -> BoltResult<Vec<f32>> {
    for &(key, val) in spill {
        let idx = key_to_group.get(&key).copied().ok_or_else(|| spill_lookup_err(key))?;
        acc[idx] = combine_f32(op, acc[idx], val);
    }
    Ok(acc)
}
fn apply_spill_f64(
    mut acc: Vec<f64>,
    spill: &[(i64, f64)],
    op: ReduceOp,
    key_to_group: &HashMap<i64, usize>,
) -> BoltResult<Vec<f64>> {
    for &(key, val) in spill {
        let idx = key_to_group.get(&key).copied().ok_or_else(|| spill_lookup_err(key))?;
        acc[idx] = combine_f64(op, acc[idx], val);
    }
    Ok(acc)
}

fn combine_i32(op: ReduceOp, a: i32, b: i32) -> i32 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => a.wrapping_add(b),
        ReduceOp::Min => a.min(b),
        ReduceOp::Max => a.max(b),
    }
}
fn combine_i64(op: ReduceOp, a: i64, b: i64) -> i64 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => a.wrapping_add(b),
        ReduceOp::Min => a.min(b),
        ReduceOp::Max => a.max(b),
    }
}
fn combine_f32(op: ReduceOp, a: f32, b: f32) -> f32 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => a + b,
        // NaN-handling: match the GPU kernels' "ignore NaN candidate"
        // semantics. NaN compared via < / > is always false, so for NaN
        // candidates we keep `a`.
        ReduceOp::Min => {
            if b.is_nan() {
                a
            } else if a.is_nan() || b < a {
                b
            } else {
                a
            }
        }
        ReduceOp::Max => {
            if b.is_nan() {
                a
            } else if a.is_nan() || b > a {
                b
            } else {
                a
            }
        }
    }
}
fn combine_f64(op: ReduceOp, a: f64, b: f64) -> f64 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => a + b,
        ReduceOp::Min => {
            if b.is_nan() {
                a
            } else if a.is_nan() || b < a {
                b
            } else {
                a
            }
        }
        ReduceOp::Max => {
            if b.is_nan() {
                a
            } else if a.is_nan() || b > a {
                b
            } else {
                a
            }
        }
    }
}

fn spill_lookup_err(key: i64) -> BoltError {
    BoltError::Other(format!(
        "internal: spilled key {} not present in any group — keys-kernel \
         spill should have created an entry",
        key
    ))
}

/// Typed batch of per-group scalar values, prior to dtype-casting into Arrow.
enum Scalars {
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// Cast a `Scalars` batch into an Arrow array of `out_dtype`.
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
            "GROUP BY (valid-flag): cannot pack scalars into output dtype {:?}",
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
// Misc helpers (mirror of `groupby.rs`'s private helpers).
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

/// Extract the column name from a bare-column-ref expression.
fn bare_column_name(expr: &Expr) -> BoltResult<&str> {
    match expr {
        Expr::Column(name) => Ok(name.as_str()),
        Expr::Alias(inner, _) => bare_column_name(inner),
        _ => Err(BoltError::Other(
            "GROUP BY (valid-flag): aggregate input must be a bare column reference in v1".into(),
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

/// Map Arrow `DataType` to our plan `DataType`.
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
// Tests (H1 NULL-mask propagation in the sentinel-free GROUP BY path).
//
// These mirror the host-side tests in `crate::exec::groupby::tests` — they
// verify that:
//   * `load_key_column_bits` returns `(bits, Some(mask))` when the key
//     column has nulls and `(bits, None)` when it doesn't.
//   * `pack_keys` propagates the combined mask through `PackedKeys`.
//   * The host-side filtering helpers used by SUM/MIN/MAX/COUNT/AVG drop
//     positions in lockstep on `key_valid ∧ value_valid`.
//
// GPU end-to-end behaviour for the sentinel-free path needs a live CUDA
// device and is out of scope for this unit-test module (the host-only
// helpers below cover the data-shape contract end-to-end).
//
// Sentinel-collision coverage: the `-0.0` Float64 case — whose
// `to_bits()` is `i64::MIN` — is the entire reason this module exists.
// The `pack_keys_float64_neg_zero_with_null_surfaces_mask` test below
// checks that a NULL row in the same batch as a `-0.0` row produces the
// correct (post-mask) key vector with the `-0.0` row preserved.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal `AggregateSpec` for the pack_keys tests. We only
    /// need `inputs` + `group_by`; the `aggregates` and `output_schema`
    /// aren't touched by `pack_keys`.
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
            input_has_validity: Vec::new(),
        }
    }

    /// Build a single-column RecordBatch from a typed Arrow array. The
    /// field is marked nullable so callers can pass arrays with NULL
    /// validity bitmaps.
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

    // -----------------------------------------------------------------
    // pack_keys / load_key_column_bits — surface the NULL mask.
    // -----------------------------------------------------------------

    /// `pack_keys` surfaces `key_valid = None` when no key column has nulls.
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

    /// `pack_keys` surfaces a per-row `key_valid` mask when the key column
    /// has NULLs. Downstream the executor drops those rows.
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

    /// Two-column GROUP BY merges per-column null masks via logical AND.
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
        assert_eq!(mask, vec![true, false, false, true]);
    }

    /// `and_masks` honours `None` as all-true.
    #[test]
    fn and_masks_combinators() {
        assert_eq!(and_masks(None, None), None);
        assert_eq!(
            and_masks(Some(vec![true, false]), None),
            Some(vec![true, false])
        );
        assert_eq!(
            and_masks(None, Some(vec![false, true])),
            Some(vec![false, true])
        );
        assert_eq!(
            and_masks(
                Some(vec![true, true, false]),
                Some(vec![true, false, true])
            ),
            Some(vec![true, false, false])
        );
    }

    /// Sentinel-collision row coverage: the whole reason this module
    /// exists. A Float64 `-0.0` packs to `i64::MIN`. With a NULL row in
    /// the same batch, the mask MUST flag the NULL row and the `-0.0`
    /// row must survive the post-mask filter — proving the NULL fix
    /// doesn't accidentally drop a sentinel-colliding value.
    #[test]
    fn pack_keys_float64_neg_zero_with_null_surfaces_mask() {
        let agg = spec(vec![("k", DataType::Float64)], vec![0]);
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(-0.0f64),
            None,
            Some(1.5),
            Some(-0.0f64),
        ]));
        let batch = one_col_batch("k", arr);
        let packed = pack_keys(&agg, &batch).expect("pack ok");
        let mask = packed.key_valid.expect("expected mask");
        assert_eq!(mask, vec![true, false, true, true]);

        // Apply the mask just like `execute_groupby_valid` does: the two
        // -0.0 rows survive (both equal to i64::MIN as i64), the NULL row
        // is dropped, the 1.5 row survives.
        let filtered: Vec<i64> = packed
            .keys_i64
            .iter()
            .zip(mask.iter())
            .filter_map(|(k, &keep)| if keep { Some(*k) } else { None })
            .collect();
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0], i64::MIN);
        assert_eq!(filtered[2], i64::MIN);
        // 1.5_f64.to_bits() is non-i64::MIN.
        assert_ne!(filtered[1], i64::MIN);
        assert_eq!(filtered[1], 1.5f64.to_bits() as i64);
    }

    // -----------------------------------------------------------------
    // column_null_mask + collect_filtered_primitive + filter_iter_to_f64
    //
    // These are the host-side helpers the agg launches call to drop
    // (key, value) tuples in lockstep on the joint mask.
    // -----------------------------------------------------------------

    /// `column_null_mask` returns `None` when the column has no nulls and
    /// a per-row mask when it does.
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
    /// false. SUM/MIN/MAX/COUNT(col) all use this to skip value-NULL rows.
    ///
    /// Value-NULL only: MIN/MAX/AVG/COUNT(col) ignores garbage at NULL.
    #[test]
    fn collect_filtered_primitive_drops_value_null_rows() {
        let arr = Int32Array::from(vec![Some(1i32), None, Some(3), Some(4)]);
        let vv: Vec<bool> = (0..arr.len()).map(|i| !arr.is_null(i)).collect();
        let out = collect_filtered_primitive::<arrow_array::types::Int32Type>(
            &arr,
            None,
            Some(&vv),
        );
        assert_eq!(out, vec![1, 3, 4]);
    }

    /// Key-NULL only: MIN/MAX/AVG/COUNT(col) on a key column with NULLs.
    /// Rows where the key is NULL must be dropped from the value stream.
    #[test]
    fn collect_filtered_primitive_drops_key_null_rows() {
        let arr = Int32Array::from(vec![10i32, 20, 30, 40]);
        // Simulate "row 0 and row 2 had a NULL key" — all values are
        // present in the value column, but key-NULLs disqualify those rows.
        let kv = vec![false, true, false, true];
        let out = collect_filtered_primitive::<arrow_array::types::Int32Type>(
            &arr,
            Some(&kv),
            None,
        );
        assert_eq!(out, vec![20, 40]);
    }

    /// Joint NULL — row dropped from BOTH input arrays when EITHER mask
    /// is false. This is the contract the agg paths rely on so that the
    /// filtered key column and the filtered value column have the same
    /// length and stay aligned row-for-row.
    #[test]
    fn collect_filtered_primitive_joint_mask() {
        let arr = Int32Array::from(vec![Some(1i32), None, Some(3), Some(4)]);
        let vv: Vec<bool> = (0..arr.len()).map(|i| !arr.is_null(i)).collect();
        // key_valid: drop row 0 too.
        let kv = vec![false, true, true, true];
        let out = collect_filtered_primitive::<arrow_array::types::Int32Type>(
            &arr,
            Some(&kv),
            Some(&vv),
        );
        // Survivors: row 2 (kv=T, vv=T) and row 3 (kv=T, vv=T).
        // Row 0: kv=F. Row 1: vv=F. Both dropped.
        assert_eq!(out, vec![3, 4]);
    }

    /// `filter_iter_to_f64` (AVG helper) drops NULLs on both sides and
    /// upcasts in one pass. This is what keeps SUM and COUNT aligned for
    /// AVG: every contribution to the f64 numerator increments the i64
    /// denominator exactly once.
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

        // Joint mask: also drop row 2 via key_valid. Survivors: rows 0
        // (kv=T, vv=T → 2.0) and 3 (kv=T, vv=T → 6.0). Row 1: vv=F.
        // Row 2: kv=F.
        let kv = vec![true, true, false, true];
        let out2 = filter_iter_to_f64::<arrow_array::types::Int32Type, _>(
            &arr,
            Some(&kv),
            Some(&vv),
            |v| v as f64,
        );
        assert_eq!(out2, vec![2.0f64, 6.0]);
    }

    // ---- PV-stage-e: runtime native-validity dispatch decision ----

    /// Columns with no NULLs must keep the classic kernel — the sentinel-
    /// free path's classic kernel already handles every row correctly and
    /// the `_with_validity` variant costs an extra bitmap upload.
    #[test]
    fn pv_stage_e_no_nulls_skips_native_validity() {
        let arr = Int64Array::from(vec![1i64, 2, 3, 4, 5]);
        assert_eq!(arr.null_count(), 0);
        assert!(
            !column_should_use_native_validity(&arr, ReduceOp::Sum, DataType::Int64),
            "no NULLs in column -> classic kernel"
        );
    }

    /// Integer SUM with NULLs is the canonical case: the
    /// `valid_flag_kernels::compile_agg_valid_kernel_with_validity`
    /// emitter covers it natively.
    #[test]
    fn pv_stage_e_int64_sum_with_nulls_dispatches_native() {
        let arr = Int64Array::from(vec![Some(1i64), None, Some(3), None, Some(5)]);
        assert!(arr.null_count() > 0);
        assert!(
            column_should_use_native_validity(&arr, ReduceOp::Sum, DataType::Int64),
            "Int64 SUM with NULLs should dispatch native validity"
        );
    }

    /// PV-stage-e completed the float MIN/MAX `_with_validity` emitter
    /// (`compile_agg_valid_float_kernel_with_validity`), so this executor
    /// dispatches natively for float MIN/MAX too — unlike
    /// `crate::exec::groupby::column_should_use_native_validity`, which
    /// still falls through to host-strip for that case (no companion in
    /// `float_atomics`).
    #[test]
    fn pv_stage_e_float_minmax_with_nulls_dispatches_native() {
        let arr = Float64Array::from(vec![Some(1.0f64), None, Some(3.0)]);
        assert!(arr.null_count() > 0);
        for op in [ReduceOp::Min, ReduceOp::Max] {
            assert!(
                column_should_use_native_validity(&arr, op, DataType::Float64),
                "Float64 {:?} now has a _with_validity emitter; expected native dispatch",
                op
            );
        }
    }

    /// Float SUM with NULLs follows the same coverage rule — the integer
    /// agg kernel handles `atom.global.add.f{32,64}` natively.
    #[test]
    fn pv_stage_e_float_sum_with_nulls_dispatches_native() {
        let arr = Float32Array::from(vec![Some(1.0f32), None, Some(2.5)]);
        assert!(arr.null_count() > 0);
        assert!(
            column_should_use_native_validity(&arr, ReduceOp::Sum, DataType::Float32),
            "Float32 SUM with NULLs should dispatch native validity"
        );
    }
}
