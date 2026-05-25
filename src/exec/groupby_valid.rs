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
//! encoding could collide with the classic sentinel — most notably Float64
//! group-by columns containing `-0.0`, whose `f64::to_bits()` is exactly
//! `0x8000_0000_0000_0000 == i64::MIN`.
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

use std::collections::HashMap;
use std::collections::HashSet;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
};
use bytemuck::Pod;

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{PatinaError, PatinaResult};
use crate::exec::launch::CudaStream;
use crate::exec::n_rows_to_u32;
use crate::jit::agg_kernels::ReduceOp;
use crate::jit::valid_flag_kernels::{
    compile_agg_valid_kernel, compile_keys_valid_kernel, valid_block_size,
    VALID_AGG_KERNEL_ENTRY, VALID_KEYS_KERNEL_ENTRY,
};
use crate::jit::CudaModule;
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

/// Execute a GROUP BY plan using the sentinel-free (slot-valid-flag)
/// protocol.
///
/// Mirrors `groupby::execute_groupby`'s public surface; the orchestrator
/// dispatches to this path when the key encoding could collide with the
/// classic kernel's `i64::MIN` sentinel (notably Float64 keys including
/// `-0.0`) or when the user explicitly opts in.
pub fn execute_groupby_valid(
    plan: &PhysicalPlan,
    table_batch: &RecordBatch,
) -> PatinaResult<RecordBatch> {
    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        other => {
            return Err(PatinaError::Other(format!(
                "execute_groupby_valid: expected Aggregate plan, got {:?}",
                std::mem::discriminant(other)
            )))
        }
    };

    if aggregate.group_by.is_empty() {
        return Err(PatinaError::Other(
            "execute_groupby_valid: aggregate has no GROUP BY columns; use execute_aggregate".into(),
        ));
    }
    if pre.is_some() {
        return Err(PatinaError::Other(
            "GROUP BY (valid-flag) with projection/filter pre-kernel not yet implemented".into(),
        ));
    }

    // Encode all group-by columns into i64 keys (host-side packing). Reuses
    // the same encoding contract as the classic path: distinct tuples map
    // to distinct i64s. The KEY DIFFERENCE versus `execute_groupby` is that
    // we do NOT validate-out the `i64::MIN` value — any i64 is legal here.
    let packed = pack_keys(aggregate, table_batch)?;
    let host_keys = packed.keys_i64;
    let key_components = packed.components;
    let n_rows = host_keys.len();

    // Estimate K from the unique-key count (host scan). Same heuristic as
    // the classic path: load factor < 0.5 so spin contention is rare.
    let n_unique = unique_count(&host_keys);
    let k = next_pow2((n_unique.saturating_mul(2)).saturating_add(16)).max(64);
    let k_u32 = u32::try_from(k).map_err(|_| {
        PatinaError::Other(format!(
            "GROUP BY hash table size {} exceeds u32::MAX",
            k
        ))
    })?;

    // Allocate keys table (arbitrary contents — slot_valid drives occupancy)
    // and the slot_valid table (zero-initialised = all slots empty).
    let keys_init: Vec<i64> = vec![0i64; k];
    let mut keys_table = GpuVec::<i64>::from_slice(&keys_init)?;
    let slot_valid_init: Vec<u32> = vec![0u32; k];
    let mut slot_valid = GpuVec::<u32>::from_slice(&slot_valid_init)?;
    let key_col_gpu = GpuVec::<i64>::from_slice(&host_keys)?;

    // Allocate the keys-kernel spill buffers. The kernel writes to these
    // only when its bounded probe / spin loops overflow (extremely rare
    // under load factor < 0.5).
    let mut keys_spill_keys =
        GpuVec::<i64>::from_slice(&vec![SPILL_EMPTY_KEY; SPILL_CAPACITY])?;
    let mut keys_spill_counter = GpuVec::<u32>::from_slice(&[0u32])?;
    let max_spill_u32 = u32::try_from(SPILL_CAPACITY).map_err(|_| {
        PatinaError::Other(format!(
            "SPILL_CAPACITY {} exceeds u32::MAX",
            SPILL_CAPACITY
        ))
    })?;

    // Launch the keys-only kernel.
    let stream = CudaStream::null();
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
        )?;
        acc_results.push(acc);
    }

    // Download the keys + slot_valid tables; the latter drives "which slots
    // are real groups" without relying on a sentinel.
    let host_keys_table: Vec<i64> = keys_table.to_vec()?;
    let host_slot_valid: Vec<u32> = slot_valid.to_vec()?;
    drop(keys_table);
    drop(slot_valid);
    drop(key_col_gpu);

    // Download the keys-kernel spill. Any entries here represent rows
    // whose key never made it into the committed `keys_table` — we must
    // add them as new groups on the host.
    let keys_spill_count_raw = keys_spill_counter.to_vec()?[0] as usize;
    if keys_spill_count_raw > SPILL_CAPACITY {
        return Err(PatinaError::Other(format!(
            "GROUP BY spill overflow ({} rows lost from keys kernel); \
             increase load factor or SPILL_CAPACITY",
            keys_spill_count_raw - SPILL_CAPACITY
        )));
    }
    let keys_spill_count = keys_spill_count_raw.min(SPILL_CAPACITY);
    let host_keys_spill: Vec<i64> = if keys_spill_count > 0 {
        let v = keys_spill_keys.to_vec()?;
        v[..keys_spill_count].to_vec()
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
                PatinaError::Other(format!(
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
        PatinaError::Other(format!(
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
fn key_bit_width(dtype: DataType) -> PatinaResult<u32> {
    match dtype {
        DataType::Int32 | DataType::Float32 => Ok(32),
        DataType::Int64 | DataType::Float64 => Ok(64),
        DataType::Bool | DataType::Utf8 => Err(PatinaError::Type(format!(
            "GROUP BY key dtype {:?} not supported in v1",
            dtype
        ))),
    }
}

/// Per-group-by-column metadata: dtype + bit offset inside the i64 key.
#[derive(Debug, Clone)]
struct KeyComponent {
    /// Column name (used for error messages and field naming).
    name: String,
    /// Original dtype as declared by the plan (drives encode/decode).
    original_dtype: DataType,
    /// Bit position within the i64 key — low = 0, high = 32 for the
    /// two-column pack. Single-column keys always use offset 0.
    bit_offset: u32,
}

/// Output of [`pack_keys`]: encoded i64 column + per-component metadata.
#[derive(Debug)]
struct PackedKeys {
    /// i64-encoded keys ready to upload, one entry per input row.
    keys_i64: Vec<i64>,
    /// Per-column dtype + position, in pack order.
    components: Vec<KeyComponent>,
}

/// Load one group-by column's bit pattern (zero-extended into u64) from
/// `batch`.
fn load_key_column_bits(
    key_io: &ColumnIO,
    batch: &RecordBatch,
) -> PatinaResult<Vec<u64>> {
    let idx = batch.schema().index_of(&key_io.name).map_err(|e| {
        PatinaError::Plan(format!(
            "GROUP BY key '{}' not present in table batch: {}",
            key_io.name, e
        ))
    })?;
    let arr = batch.column(idx);

    let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
    if arr_dtype != key_io.dtype {
        return Err(PatinaError::Type(format!(
            "GROUP BY key '{}' dtype mismatch: plan says {:?}, batch has {:?}",
            key_io.name, key_io.dtype, arr_dtype
        )));
    }

    match key_io.dtype {
        DataType::Int32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(&key_io.name, "Int32"))?;
            Ok(pa.values().iter().map(|&v| v as u32 as u64).collect())
        }
        DataType::Int64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&key_io.name, "Int64"))?;
            Ok(pa.values().iter().map(|&v| v as u64).collect())
        }
        DataType::Float32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&key_io.name, "Float32"))?;
            Ok(pa.values().iter().map(|&v| v.to_bits() as u64).collect())
        }
        DataType::Float64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&key_io.name, "Float64"))?;
            Ok(pa.values().iter().map(|&v| v.to_bits()).collect())
        }
        DataType::Bool | DataType::Utf8 => Err(PatinaError::Type(format!(
            "GROUP BY key dtype {:?} not supported in v1",
            key_io.dtype
        ))),
    }
}

/// Encode each group-by column into a single i64 key per row. Mirrors the
/// supported packings of `groupby::pack_keys`.
fn pack_keys(
    aggregate: &AggregateSpec,
    batch: &RecordBatch,
) -> PatinaResult<PackedKeys> {
    if aggregate.group_by.is_empty() {
        return Err(PatinaError::Other(
            "pack_keys: aggregate has no GROUP BY columns".into(),
        ));
    }

    let mut col_ios: Vec<&ColumnIO> = Vec::with_capacity(aggregate.group_by.len());
    let mut total_bits: u32 = 0;
    for &ord in &aggregate.group_by {
        let io = aggregate.inputs.get(ord).ok_or_else(|| {
            PatinaError::Plan(format!(
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
        return Err(PatinaError::Other(format!(
            "multi-column GROUP BY with > 64 bits of key width not yet supported \
             ({} columns requested)",
            aggregate.group_by.len()
        )));
    }
    if total_bits > 64 {
        return Err(PatinaError::Other(format!(
            "multi-column GROUP BY with > 64 bits of key width not yet supported \
             (requested columns total {} bits)",
            total_bits
        )));
    }

    let mut components: Vec<KeyComponent> =
        Vec::with_capacity(aggregate.group_by.len());
    let mut bit_streams: Vec<Vec<u64>> =
        Vec::with_capacity(aggregate.group_by.len());

    if aggregate.group_by.len() == 1 {
        let io = col_ios[0];
        let bits = load_key_column_bits(io, batch)?;
        components.push(KeyComponent {
            name: io.name.clone(),
            original_dtype: io.dtype,
            bit_offset: 0,
        });
        bit_streams.push(bits);
    } else {
        let io_hi = col_ios[0];
        let io_lo = col_ios[1];
        let bits_hi = load_key_column_bits(io_hi, batch)?;
        let bits_lo = load_key_column_bits(io_lo, batch)?;
        if bits_hi.len() != bits_lo.len() {
            return Err(PatinaError::Other(format!(
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
    })
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
) -> PatinaResult<()> {
    if n_rows == 0 {
        return Ok(());
    }

    let ptx = compile_keys_valid_kernel()?;
    let module = CudaModule::from_ptx(&ptx)?;
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
    let grid_x = ((n_rows_to_u32(n_rows)? + block - 1) / block).max(1);

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
/// Float-MIN/MAX kernel has a 7-param ABI (no spill); other variants have
/// an 11-param ABI. The integer agg kernel (from `valid_flag_kernels`)
/// declares the 11-parameter ABI that includes the spill buffers +
/// counter + capacity at positions 7..=10. The float-MIN/MAX kernel
/// (from `valid_flag_float`) only takes the 7 non-spill params — it
/// cannot spill because sm_70 has no native `atom.global.{min,max}.f*`
/// and the CAS-loop variant resolves in-place. Passing 11 params to its
/// 7-param ABI would have CUDA read garbage into the trailing slots:
/// the driver doesn't validate, but the assertion below makes the
/// mismatch a hard error rather than fragile silence.
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
) -> PatinaResult<()> {
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
    let ptx = if is_float_min_max {
        crate::jit::valid_flag_float::compile_agg_valid_float_kernel(op, input_dtype)?
    } else {
        compile_agg_valid_kernel(op, input_dtype)?
    };
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(VALID_AGG_KERNEL_ENTRY)?;

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

    // Assemble the per-variant param vector. The float-MIN/MAX kernel's
    // 7-param ABI is the integer kernel's 11-param ABI with the four
    // spill-related trailing params dropped — same prefix.
    let block = valid_block_size();
    let grid_x = ((n_rows_to_u32(n_rows)? + block - 1) / block).max(1);

    if is_float_min_max {
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
) -> PatinaResult<(GpuVec<i64>, GpuVec<T>, GpuVec<u32>)> {
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
) -> PatinaResult<Vec<(i64, T)>> {
    let count_raw = spill_counter.to_vec()?[0] as usize;
    if count_raw > SPILL_CAPACITY {
        return Err(PatinaError::Other(format!(
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
) -> PatinaResult<AccDownload> {
    match agg {
        AggregateExpr::Sum(expr)
        | AggregateExpr::Min(expr)
        | AggregateExpr::Max(expr) => {
            let op = ReduceOp::from_agg(agg)?;
            let col_name = bare_column_name(expr)?;
            let col_io = resolve_input(inputs, col_name)?;
            run_typed_agg(
                op, col_io, group_col, keys_table, slot_valid, batch, n_rows, k,
                k_u32, max_spill, stream,
            )
        }

        AggregateExpr::Count(_) => {
            // COUNT(*) over groups: synthesize an all-ones i64 input column.
            let ones: Vec<i64> = vec![1i64; n_rows];
            let input_gpu = GpuVec::<i64>::from_slice(&ones)?;
            let identity_init: Vec<i64> = vec![0i64; k];
            let mut acc_table = GpuVec::<i64>::from_slice(&identity_init)?;
            let (mut spill_keys, mut spill_values, mut spill_counter) =
                alloc_agg_spill::<i64>()?;
            launch_agg_kernel::<i64>(
                ReduceOp::Sum,
                DataType::Int64,
                group_col,
                keys_table,
                slot_valid,
                &input_gpu,
                &mut acc_table,
                &mut spill_keys,
                &mut spill_values,
                &mut spill_counter,
                max_spill,
                n_rows,
                k_u32,
                stream,
            )?;
            let gpu_acc = acc_table.to_vec()?;
            let spill = download_agg_spill(
                spill_keys,
                spill_values,
                spill_counter,
                "COUNT agg kernel",
            )?;
            Ok(AccDownload::I64 { gpu_acc, spill })
        }

        AggregateExpr::Avg(expr) => {
            let col_name = bare_column_name(expr)?;
            let col_io = resolve_input(inputs, col_name)?;

            // SUM(expr) cast to f64.
            let sum_input: Vec<f64> = load_input_column_as_f64(col_io, batch)?;
            let input_gpu = GpuVec::<f64>::from_slice(&sum_input)?;
            let sum_init: Vec<f64> = vec![0.0f64; k];
            let mut sum_acc = GpuVec::<f64>::from_slice(&sum_init)?;
            let (mut sum_spill_keys, mut sum_spill_values, mut sum_spill_counter) =
                alloc_agg_spill::<f64>()?;
            launch_agg_kernel::<f64>(
                ReduceOp::Sum,
                DataType::Float64,
                group_col,
                keys_table,
                slot_valid,
                &input_gpu,
                &mut sum_acc,
                &mut sum_spill_keys,
                &mut sum_spill_values,
                &mut sum_spill_counter,
                max_spill,
                n_rows,
                k_u32,
                stream,
            )?;
            let sum_host = sum_acc.to_vec()?;
            let sum_spill = download_agg_spill(
                sum_spill_keys,
                sum_spill_values,
                sum_spill_counter,
                "AVG.SUM agg kernel",
            )?;

            // COUNT(*) per group.
            let ones: Vec<i64> = vec![1i64; n_rows];
            let count_input = GpuVec::<i64>::from_slice(&ones)?;
            let count_init: Vec<i64> = vec![0i64; k];
            let mut count_acc = GpuVec::<i64>::from_slice(&count_init)?;
            let (mut cnt_spill_keys, mut cnt_spill_values, mut cnt_spill_counter) =
                alloc_agg_spill::<i64>()?;
            launch_agg_kernel::<i64>(
                ReduceOp::Sum,
                DataType::Int64,
                group_col,
                keys_table,
                slot_valid,
                &count_input,
                &mut count_acc,
                &mut cnt_spill_keys,
                &mut cnt_spill_values,
                &mut cnt_spill_counter,
                max_spill,
                n_rows,
                k_u32,
                stream,
            )?;
            let count_host = count_acc.to_vec()?;
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

/// Common path for SUM/MIN/MAX.
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
) -> PatinaResult<AccDownload> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        PatinaError::Plan(format!(
            "aggregate input '{}' not present in table batch: {}",
            col_io.name, e
        ))
    })?;
    let arr = batch.column(idx);
    let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
    if arr_dtype != col_io.dtype {
        return Err(PatinaError::Type(format!(
            "aggregate input '{}' dtype mismatch: plan says {:?}, batch has {:?}",
            col_io.name, col_io.dtype, arr_dtype
        )));
    }

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
                let host: Vec<i64> = pa.values().iter().map(|&v| v as i64).collect();
                let input_gpu = GpuVec::<i64>::from_slice(&host)?;
                let init: Vec<i64> = vec![identity_i64(op); k];
                let mut acc = GpuVec::<i64>::from_slice(&init)?;
                let (mut spill_keys, mut spill_values, mut spill_counter) =
                    alloc_agg_spill::<i64>()?;
                launch_agg_kernel::<i64>(
                    op,
                    DataType::Int64,
                    group_col,
                    keys_table,
                    slot_valid,
                    &input_gpu,
                    &mut acc,
                    &mut spill_keys,
                    &mut spill_values,
                    &mut spill_counter,
                    max_spill,
                    n_rows,
                    k_u32,
                    stream,
                )?;
                let gpu_acc = acc.to_vec()?;
                let spill = download_agg_spill(
                    spill_keys,
                    spill_values,
                    spill_counter,
                    "i64 agg kernel (widened from SUM(Int32))",
                )?;
                Ok(AccDownload::I64 { gpu_acc, spill })
            } else {
                let host: Vec<i32> = pa.values().to_vec();
                let input_gpu = GpuVec::<i32>::from_slice(&host)?;
                let init: Vec<i32> = vec![identity_i32(op); k];
                let mut acc = GpuVec::<i32>::from_slice(&init)?;
                let (mut spill_keys, mut spill_values, mut spill_counter) =
                    alloc_agg_spill::<i32>()?;
                launch_agg_kernel::<i32>(
                    op,
                    DataType::Int32,
                    group_col,
                    keys_table,
                    slot_valid,
                    &input_gpu,
                    &mut acc,
                    &mut spill_keys,
                    &mut spill_values,
                    &mut spill_counter,
                    max_spill,
                    n_rows,
                    k_u32,
                    stream,
                )?;
                let gpu_acc = acc.to_vec()?;
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
            let host: Vec<i64> = pa.values().to_vec();
            let input_gpu = GpuVec::<i64>::from_slice(&host)?;
            let init: Vec<i64> = vec![identity_i64(op); k];
            let mut acc = GpuVec::<i64>::from_slice(&init)?;
            let (mut spill_keys, mut spill_values, mut spill_counter) =
                alloc_agg_spill::<i64>()?;
            launch_agg_kernel::<i64>(
                op,
                DataType::Int64,
                group_col,
                keys_table,
                slot_valid,
                &input_gpu,
                &mut acc,
                &mut spill_keys,
                &mut spill_values,
                &mut spill_counter,
                max_spill,
                n_rows,
                k_u32,
                stream,
            )?;
            let gpu_acc = acc.to_vec()?;
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
            let host: Vec<f32> = pa.values().to_vec();
            let input_gpu = GpuVec::<f32>::from_slice(&host)?;
            let init: Vec<f32> = vec![identity_f32(op); k];
            let mut acc = GpuVec::<f32>::from_slice(&init)?;
            let (mut spill_keys, mut spill_values, mut spill_counter) =
                alloc_agg_spill::<f32>()?;
            launch_agg_kernel::<f32>(
                op,
                DataType::Float32,
                group_col,
                keys_table,
                slot_valid,
                &input_gpu,
                &mut acc,
                &mut spill_keys,
                &mut spill_values,
                &mut spill_counter,
                max_spill,
                n_rows,
                k_u32,
                stream,
            )?;
            let gpu_acc = acc.to_vec()?;
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
            let host: Vec<f64> = pa.values().to_vec();
            let input_gpu = GpuVec::<f64>::from_slice(&host)?;
            let init: Vec<f64> = vec![identity_f64(op); k];
            let mut acc = GpuVec::<f64>::from_slice(&init)?;
            let (mut spill_keys, mut spill_values, mut spill_counter) =
                alloc_agg_spill::<f64>()?;
            launch_agg_kernel::<f64>(
                op,
                DataType::Float64,
                group_col,
                keys_table,
                slot_valid,
                &input_gpu,
                &mut acc,
                &mut spill_keys,
                &mut spill_values,
                &mut spill_counter,
                max_spill,
                n_rows,
                k_u32,
                stream,
            )?;
            let gpu_acc = acc.to_vec()?;
            let spill = download_agg_spill(
                spill_keys,
                spill_values,
                spill_counter,
                "f64 agg kernel",
            )?;
            Ok(AccDownload::F64 { gpu_acc, spill })
        }
        DataType::Bool | DataType::Utf8 => Err(PatinaError::Type(format!(
            "aggregate input dtype {:?} not supported (column '{}')",
            col_io.dtype, col_io.name
        ))),
    }
}

/// Pull a numeric input column out of `batch` and upcast to f64 (for AVG).
fn load_input_column_as_f64(
    col_io: &ColumnIO,
    batch: &RecordBatch,
) -> PatinaResult<Vec<f64>> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        PatinaError::Plan(format!(
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
            Ok(pa.values().iter().map(|&v| v as f64).collect())
        }
        DataType::Int64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int64"))?;
            Ok(pa.values().iter().map(|&v| v as f64).collect())
        }
        DataType::Float32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float32"))?;
            Ok(pa.values().iter().map(|&v| v as f64).collect())
        }
        DataType::Float64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float64"))?;
            Ok(pa.values().to_vec())
        }
        DataType::Bool | DataType::Utf8 => Err(PatinaError::Type(format!(
            "AVG input dtype {:?} not supported (column '{}')",
            col_io.dtype, col_io.name
        ))),
    }
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
) -> PatinaResult<Vec<ArrayRef>> {
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
                return Err(PatinaError::Type(format!(
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
                    return Err(PatinaError::Other(
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
) -> PatinaResult<ArrayRef> {
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
                other => Err(PatinaError::Type(format!(
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
                    return Err(PatinaError::Other(
                        "internal: AVG accumulator passed to non-AVG aggregate"
                            .into(),
                    ))
                }
            };
            pack_array(out_field.dtype, scalars)
        }
        (_, _) => Err(PatinaError::Other(
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
) -> PatinaResult<AccDownload> {
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
) -> PatinaResult<Vec<i32>> {
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
) -> PatinaResult<Vec<i64>> {
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
) -> PatinaResult<Vec<f32>> {
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
) -> PatinaResult<Vec<f64>> {
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

fn spill_lookup_err(key: i64) -> PatinaError {
    PatinaError::Other(format!(
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
fn pack_array(out_dtype: DataType, scalars: Scalars) -> PatinaResult<ArrayRef> {
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

        (_, dt) => Err(PatinaError::Type(format!(
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
fn resolve_input<'a>(inputs: &'a [ColumnIO], name: &str) -> PatinaResult<&'a ColumnIO> {
    inputs.iter().find(|c| c.name == name).ok_or_else(|| {
        PatinaError::Plan(format!(
            "aggregate input column '{}' not found in plan inputs",
            name
        ))
    })
}

/// Extract the column name from a bare-column-ref expression.
fn bare_column_name(expr: &Expr) -> PatinaResult<&str> {
    match expr {
        Expr::Column(name) => Ok(name.as_str()),
        Expr::Alias(inner, _) => bare_column_name(inner),
        _ => Err(PatinaError::Other(
            "GROUP BY (valid-flag): aggregate input must be a bare column reference in v1".into(),
        )),
    }
}

/// `Type` error for a failed Arrow downcast on column `name`.
fn downcast_err(name: &str, expected: &str) -> PatinaError {
    PatinaError::Type(format!(
        "GROUP BY input column '{}' could not be downcast to {}",
        name, expected
    ))
}

/// Map Arrow `DataType` to our plan `DataType`.
fn arrow_dtype_to_plan(d: &ArrowDataType) -> PatinaResult<DataType> {
    match d {
        ArrowDataType::Int32 => Ok(DataType::Int32),
        ArrowDataType::Int64 => Ok(DataType::Int64),
        ArrowDataType::Float32 => Ok(DataType::Float32),
        ArrowDataType::Float64 => Ok(DataType::Float64),
        ArrowDataType::Boolean => Ok(DataType::Bool),
        ArrowDataType::Utf8 => Ok(DataType::Utf8),
        other => Err(PatinaError::Type(format!(
            "unsupported Arrow dtype {:?}",
            other
        ))),
    }
}

/// Map our plan `DataType` to Arrow `DataType`.
fn plan_dtype_to_arrow(d: DataType) -> PatinaResult<ArrowDataType> {
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
fn plan_schema_to_arrow_schema(s: &Schema) -> PatinaResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}
