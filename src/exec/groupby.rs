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
//!   4. Launch the keys kernel (`javelin_groupby_keys`) — one thread per row.
//!      Each row hashes its key and inserts it into the open-addressing table
//!      via an `atom.cas` linear probe.
//!   5. For each aggregate, upload its input column, JIT + launch
//!      `javelin_groupby_agg` against the already-populated keys table and
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
    ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
};
use bytemuck::Pod;

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{JavelinError, JavelinResult};
use crate::exec::launch::CudaStream;
use crate::exec::n_rows_to_u32;
use crate::jit::agg_kernels::ReduceOp;
use crate::jit::hash_kernels::{
    compile_groupby_agg_kernel, compile_groupby_keys_kernel, groupby_block_size,
    AGG_KERNEL_ENTRY, KEYS_KERNEL_ENTRY,
};
use crate::jit::CudaModule;
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Field, Schema};
use crate::plan::physical_plan::{AggregateSpec, ColumnIO, PhysicalPlan};

/// Empty-slot sentinel; mirrors the literal baked into the keys kernel.
const EMPTY_KEY: i64 = i64::MIN;

/// Execute a GROUP BY aggregate plan against a host-side `RecordBatch`.
///
/// `plan` must be `PhysicalPlan::Aggregate` with non-empty `group_by`.
/// Supports single-column (Int32/Int64/Float32/Float64) and a limited set of
/// 2-column packings whose combined width fits in 64 bits; see module docs.
pub fn execute_groupby(
    plan: &PhysicalPlan,
    table_batch: &RecordBatch,
) -> JavelinResult<RecordBatch> {
    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        PhysicalPlan::Projection { .. } => {
            return Err(JavelinError::Other(
                "execute_groupby: expected Aggregate plan, got Projection".into(),
            ))
        }
    };

    if aggregate.group_by.is_empty() {
        return Err(JavelinError::Other(
            "execute_groupby: aggregate has no GROUP BY columns; use execute_aggregate".into(),
        ));
    }
    if pre.is_some() {
        return Err(JavelinError::Other(
            "GROUP BY with projection/filter pre-kernel not yet implemented".into(),
        ));
    }

    // Encode all group-by columns into i64 keys (host-side packing). If the
    // key is too wide to pack losslessly into i64, delegate to the wide-key
    // host-side fallback in `crate::exec::groupby_wide`.
    let packed = match pack_keys(aggregate, table_batch) {
        Ok(p) => p,
        Err(JavelinError::Other(msg))
            if msg.contains("> 64 bits") || msg.contains("not yet supported") =>
        {
            return crate::exec::groupby_wide::execute_groupby_wide(plan, table_batch);
        }
        Err(e) => return Err(e),
    };
    let host_keys = packed.keys_i64;
    let key_components = packed.components;

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
        JavelinError::Other(format!(
            "GROUP BY hash table size {} exceeds u32::MAX",
            k
        ))
    })?;

    // Build the keys table on the host (filled with EMPTY_KEY) and upload it.
    let host_keys_init: Vec<i64> = vec![EMPTY_KEY; k];
    let mut keys_table = GpuVec::<i64>::from_slice(&host_keys_init)?;
    let key_col_gpu = GpuVec::<i64>::from_slice(&host_keys)?;

    // Launch the keys-only kernel.
    let stream = CudaStream::null();
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
        )?;
        acc_results.push(acc);
    }

    // Download the keys table to drive the host-side output assembly.
    let host_keys_table: Vec<i64> = keys_table.to_vec()?;
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
                JavelinError::Other(format!(
                    "execute_groupby: output_schema missing field for aggregate index {}",
                    i
                ))
            })?;
        let arr = build_agg_array(agg, out_field, &acc_results[i], &groups, n_groups)?;
        arrays.push(arr);
    }

    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
        JavelinError::Other(format!("failed to build GROUP BY RecordBatch: {e}"))
    })
}

// ---------------------------------------------------------------------------
// Key column extraction, packing, and decoding.
// ---------------------------------------------------------------------------

/// Number of bits a column's encoded value occupies inside a packed i64 key.
/// Anything wider than 64 bits total is rejected (composite-hash fallback is
/// not implemented in v1).
fn key_bit_width(dtype: DataType) -> JavelinResult<u32> {
    match dtype {
        DataType::Int32 | DataType::Float32 => Ok(32),
        DataType::Int64 | DataType::Float64 => Ok(64),
        DataType::Bool | DataType::Utf8 => Err(JavelinError::Type(format!(
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
    /// i64-encoded keys ready to upload, one entry per input row.
    keys_i64: Vec<i64>,
    /// Per-column dtype + original ordinal in `aggregate.inputs`, in pack order.
    components: Vec<KeyComponent>,
}

/// Read the per-row value of one group-by column out of `batch` and return a
/// `Vec<u64>` of its low-significance bit pattern, zero-extended into u64.
///
/// For Int32 this is the unsigned 32-bit representation (i.e. sign bits are
/// dropped); for Int64 it's a straight bitcast; for floats it's `to_bits`.
/// The result is ready to be OR'd into a packed key at the right shift.
fn load_key_column_bits(
    key_io: &ColumnIO,
    batch: &RecordBatch,
) -> JavelinResult<Vec<u64>> {
    let idx = batch.schema().index_of(&key_io.name).map_err(|e| {
        JavelinError::Plan(format!(
            "GROUP BY key '{}' not present in table batch: {}",
            key_io.name, e
        ))
    })?;
    let arr = batch.column(idx);

    // Sanity-check the dtype matches what the plan promised.
    let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
    if arr_dtype != key_io.dtype {
        return Err(JavelinError::Type(format!(
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
        DataType::Bool | DataType::Utf8 => Err(JavelinError::Type(format!(
            "GROUP BY key dtype {:?} not supported in v1",
            key_io.dtype
        ))),
    }
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
) -> JavelinResult<PackedKeys> {
    if aggregate.group_by.is_empty() {
        return Err(JavelinError::Other(
            "pack_keys: aggregate has no GROUP BY columns".into(),
        ));
    }

    // Resolve every group-by ordinal to a ColumnIO and validate widths.
    let mut col_ios: Vec<&ColumnIO> = Vec::with_capacity(aggregate.group_by.len());
    let mut total_bits: u32 = 0;
    for &ord in &aggregate.group_by {
        let io = aggregate.inputs.get(ord).ok_or_else(|| {
            JavelinError::Plan(format!(
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
        return Err(JavelinError::Other(format!(
            "multi-column GROUP BY with > 64 bits of key width not yet supported \
             ({} columns requested)",
            aggregate.group_by.len()
        )));
    }
    if total_bits > 64 {
        return Err(JavelinError::Other(format!(
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
        // len == 2, total_bits <= 64. Both individual widths are <= 32 (else
        // total_bits > 64), so the high column gets bit_offset=32 and the
        // low column gets bit_offset=0.
        let io_hi = col_ios[0];
        let io_lo = col_ios[1];
        let bits_hi = load_key_column_bits(io_hi, batch)?;
        let bits_lo = load_key_column_bits(io_lo, batch)?;
        if bits_hi.len() != bits_lo.len() {
            return Err(JavelinError::Other(format!(
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
    })
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
) -> JavelinResult<()> {
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
    let grid_x = ((n_rows_u32 + block - 1) / block).max(1);

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
) -> JavelinResult<()> {
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
    let grid_x = ((n_rows_u32 + block - 1) / block).max(1);

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
) -> JavelinResult<AccDownload> {
    match agg {
        AggregateExpr::Sum(expr)
        | AggregateExpr::Min(expr)
        | AggregateExpr::Max(expr) => {
            let op = ReduceOp::from_agg(agg)?;
            let col_name = bare_column_name(expr)?;
            let col_io = resolve_input(inputs, col_name)?;
            run_typed_agg(
                op, col_io, group_col, keys_table, batch, n_rows, k, k_u32, stream,
            )
        }

        AggregateExpr::Count(_) => {
            // COUNT(*) over groups: synthesize an all-ones i64 input column.
            let ones: Vec<i64> = vec![1i64; n_rows];
            let input_gpu = GpuVec::<i64>::from_slice(&ones)?;
            let identity_init: Vec<i64> = vec![0i64; k];
            let mut acc_table = GpuVec::<i64>::from_slice(&identity_init)?;
            launch_agg_kernel::<i64>(
                ReduceOp::Sum,
                DataType::Int64,
                group_col,
                keys_table,
                &input_gpu,
                &mut acc_table,
                n_rows,
                k_u32,
                stream,
            )?;
            Ok(AccDownload::I64(acc_table.to_vec()?))
        }

        AggregateExpr::Avg(expr) => {
            // AVG = SUM / COUNT, both grouped. SUM in f64 (so we don't worry
            // about int-overflow during accumulation), COUNT in i64.
            let col_name = bare_column_name(expr)?;
            let col_io = resolve_input(inputs, col_name)?;

            // --- SUM(expr) cast to f64. We upcast the input host-side. ---
            let sum_input: Vec<f64> = load_input_column_as_f64(col_io, batch)?;
            let input_gpu = GpuVec::<f64>::from_slice(&sum_input)?;
            let sum_init: Vec<f64> = vec![0.0f64; k];
            let mut sum_acc = GpuVec::<f64>::from_slice(&sum_init)?;
            launch_agg_kernel::<f64>(
                ReduceOp::Sum,
                DataType::Float64,
                group_col,
                keys_table,
                &input_gpu,
                &mut sum_acc,
                n_rows,
                k_u32,
                stream,
            )?;
            let sum_host = sum_acc.to_vec()?;

            // --- COUNT(*) per group. ---
            let ones: Vec<i64> = vec![1i64; n_rows];
            let count_input = GpuVec::<i64>::from_slice(&ones)?;
            let count_init: Vec<i64> = vec![0i64; k];
            let mut count_acc = GpuVec::<i64>::from_slice(&count_init)?;
            launch_agg_kernel::<i64>(
                ReduceOp::Sum,
                DataType::Int64,
                group_col,
                keys_table,
                &count_input,
                &mut count_acc,
                n_rows,
                k_u32,
                stream,
            )?;
            let count_host = count_acc.to_vec()?;

            Ok(AccDownload::Avg {
                sum: sum_host,
                count: count_host,
            })
        }
    }
}

/// Common path for SUM/MIN/MAX. Uploads the typed input column, allocates a
/// typed accumulator initialised to the op's identity, launches the agg
/// kernel, and downloads the accumulator.
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
) -> JavelinResult<AccDownload> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        JavelinError::Plan(format!(
            "aggregate input '{}' not present in table batch: {}",
            col_io.name, e
        ))
    })?;
    let arr = batch.column(idx);
    let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
    if arr_dtype != col_io.dtype {
        return Err(JavelinError::Type(format!(
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
            let host: Vec<i32> = pa.values().to_vec();
            let input_gpu = GpuVec::<i32>::from_slice(&host)?;
            let init: Vec<i32> = vec![identity_i32(op); k];
            let mut acc = GpuVec::<i32>::from_slice(&init)?;
            launch_agg_kernel::<i32>(
                op,
                DataType::Int32,
                group_col,
                keys_table,
                &input_gpu,
                &mut acc,
                n_rows,
                k_u32,
                stream,
            )?;
            Ok(AccDownload::I32(acc.to_vec()?))
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
            launch_agg_kernel::<i64>(
                op,
                DataType::Int64,
                group_col,
                keys_table,
                &input_gpu,
                &mut acc,
                n_rows,
                k_u32,
                stream,
            )?;
            Ok(AccDownload::I64(acc.to_vec()?))
        }
        DataType::Float32 => {
            // MIN/MAX over floats are routed to the float-atomic CAS kernel
            // by launch_agg_kernel; no early rejection needed.
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float32"))?;
            let host: Vec<f32> = pa.values().to_vec();
            let input_gpu = GpuVec::<f32>::from_slice(&host)?;
            let init: Vec<f32> = vec![identity_f32(op); k];
            let mut acc = GpuVec::<f32>::from_slice(&init)?;
            launch_agg_kernel::<f32>(
                op,
                DataType::Float32,
                group_col,
                keys_table,
                &input_gpu,
                &mut acc,
                n_rows,
                k_u32,
                stream,
            )?;
            Ok(AccDownload::F32(acc.to_vec()?))
        }
        DataType::Float64 => {
            // MIN/MAX over floats are routed to the float-atomic CAS kernel
            // by launch_agg_kernel; no early rejection needed.
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float64"))?;
            let host: Vec<f64> = pa.values().to_vec();
            let input_gpu = GpuVec::<f64>::from_slice(&host)?;
            let init: Vec<f64> = vec![identity_f64(op); k];
            let mut acc = GpuVec::<f64>::from_slice(&init)?;
            launch_agg_kernel::<f64>(
                op,
                DataType::Float64,
                group_col,
                keys_table,
                &input_gpu,
                &mut acc,
                n_rows,
                k_u32,
                stream,
            )?;
            Ok(AccDownload::F64(acc.to_vec()?))
        }
        DataType::Bool | DataType::Utf8 => Err(JavelinError::Type(format!(
            "aggregate input dtype {:?} not supported (column '{}')",
            col_io.dtype, col_io.name
        ))),
    }
}

/// Pull a numeric input column out of `batch` and upcast to f64 (used by AVG).
fn load_input_column_as_f64(
    col_io: &ColumnIO,
    batch: &RecordBatch,
) -> JavelinResult<Vec<f64>> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        JavelinError::Plan(format!(
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
        DataType::Bool | DataType::Utf8 => Err(JavelinError::Type(format!(
            "AVG input dtype {:?} not supported (column '{}')",
            col_io.dtype, col_io.name
        ))),
    }
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
) -> JavelinResult<Vec<ArrayRef>> {
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
                return Err(JavelinError::Type(format!(
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
                    return Err(JavelinError::Other(
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
fn build_agg_array(
    agg: &AggregateExpr,
    out_field: &Field,
    acc: &AccDownload,
    groups: &[(i64, usize)],
    n_groups: usize,
) -> JavelinResult<ArrayRef> {
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
                    return Err(JavelinError::Other(
                        "internal: AVG accumulator passed to non-AVG aggregate"
                            .into(),
                    ))
                }
            };
            pack_array(out_field.dtype, scalars)
        }
        (_, _) => Err(JavelinError::Other(
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
fn pack_array(out_dtype: DataType, scalars: Scalars) -> JavelinResult<ArrayRef> {
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

        (_, dt) => Err(JavelinError::Type(format!(
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
fn resolve_input<'a>(inputs: &'a [ColumnIO], name: &str) -> JavelinResult<&'a ColumnIO> {
    inputs.iter().find(|c| c.name == name).ok_or_else(|| {
        JavelinError::Plan(format!(
            "aggregate input column '{}' not found in plan inputs",
            name
        ))
    })
}

/// Extract the column name from a bare-column-ref expression. The v1 path
/// requires every aggregate input to be a bare column ref.
fn bare_column_name(expr: &Expr) -> JavelinResult<&str> {
    match expr {
        Expr::Column(name) => Ok(name.as_str()),
        Expr::Alias(inner, _) => bare_column_name(inner),
        _ => Err(JavelinError::Other(
            "GROUP BY: aggregate input must be a bare column reference in v1".into(),
        )),
    }
}

/// `Type` error for a failed Arrow downcast on column `name`.
fn downcast_err(name: &str, expected: &str) -> JavelinError {
    JavelinError::Type(format!(
        "GROUP BY input column '{}' could not be downcast to {}",
        name, expected
    ))
}

/// Map Arrow `DataType` to our plan `DataType` (mirrors `aggregate.rs`).
fn arrow_dtype_to_plan(d: &ArrowDataType) -> JavelinResult<DataType> {
    match d {
        ArrowDataType::Int32 => Ok(DataType::Int32),
        ArrowDataType::Int64 => Ok(DataType::Int64),
        ArrowDataType::Float32 => Ok(DataType::Float32),
        ArrowDataType::Float64 => Ok(DataType::Float64),
        ArrowDataType::Boolean => Ok(DataType::Bool),
        ArrowDataType::Utf8 => Ok(DataType::Utf8),
        other => Err(JavelinError::Type(format!(
            "unsupported Arrow dtype {:?}",
            other
        ))),
    }
}

/// Map our plan `DataType` to Arrow `DataType` (mirrors `aggregate.rs`).
fn plan_dtype_to_arrow(d: DataType) -> JavelinResult<ArrowDataType> {
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
fn plan_schema_to_arrow_schema(s: &Schema) -> JavelinResult<Arc<ArrowSchema>> {
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

    /// Build a single-column RecordBatch from a typed Arrow array.
    fn one_col_batch(name: &str, arr: ArrayRef) -> RecordBatch {
        let dt = arr.data_type().clone();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            name, dt, false,
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
            ArrowField::new(n1, a1.data_type().clone(), false),
            ArrowField::new(n2, a2.data_type().clone(), false),
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
}
