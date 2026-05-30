// SPDX-License-Identifier: Apache-2.0

//! Shared host-side helpers for the GROUP BY execution family.
//!
//! ## Why this module exists (dedup — groupby_common)
//!
//! The audit's top maintainability finding was that a cluster of host-side
//! helper functions and key-packing types were COPY-PASTED across the three
//! sibling GROUP BY executors —
//! [`crate::exec::groupby`] (classic sentinel kernel),
//! [`crate::exec::groupby_valid`] (sentinel-free slot-valid-flag kernel), and
//! [`crate::exec::groupby_with_pre`] (fused pre-projection + GROUP BY).
//!
//! The copies had already DRIFTED: the V-17 hardening of `pack_keys`
//! (switching the bare `<<` shift to `wrapping_shl` so a future `shift == 64`
//! regression produces a deterministic `0` instead of UB / a debug panic) had
//! to be applied to `groupby.rs` and then *separately* re-applied to
//! `groupby_valid.rs`. That class of "fix-it-twice" bug is exactly what this
//! module removes: there is now ONE canonical copy of each truly-identical
//! helper, with `pub(crate)` visibility, and the three executors import them.
//!
//! ## What lives here vs. what stayed local
//!
//! Only helpers whose copies were *byte-equivalent in behavior* (after
//! adopting the most-hardened variant where copies had drifted) were
//! consolidated. The key-packing types and functions
//! ([`KeyComponent`], [`KeyValue`], [`PackedKeys`], [`pack_keys`],
//! [`decode_key`], [`load_key_column_bits`]) lived in two of the three
//! executors with one subtle difference — the classic `groupby.rs` copy
//! carried a `KeyComponent::name` field (read by its `pack_keys` unit tests
//! and surfaced in two-column error messages) while the `groupby_valid.rs`
//! copy did not. The canonical version here keeps the SUPERSET (the `name`
//! field is present); `groupby_valid`'s production path never reads it, so
//! adopting the superset is behavior-preserving there.
//!
//! Helpers that are NOT shared (and intentionally stay local):
//!   * `key_array_contains_sentinel` (only `groupby.rs` — classic-kernel
//!     sentinel pre-flight).
//!   * `launch_keys_kernel` / `launch_agg_kernel` (each executor has a
//!     different kernel ABI — classic 4/6-param, valid-flag 8/11/12-param,
//!     with-pre 4/5/6/7-param — so they are genuinely different code, not
//!     copies).
//!   * `column_should_use_native_validity` / `dispatch_native_validity`
//!     (different per-executor kernel coverage matrices).

use std::collections::HashSet;

use arrow_array::{
    Array, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::CudaStream;
use crate::plan::logical_plan::DataType;
use crate::plan::physical_plan::{AggregateSpec, ColumnIO};

// ---------------------------------------------------------------------------
// Stage-3 pinned D2H helpers.
//
// dedup (groupby_common): consolidated from the byte-identical copies in
// `groupby.rs`, `groupby_valid.rs`, and `groupby_with_pre.rs`. Each
// accumulator-download site previously called `gpu_vec.to_vec()?`, which uses
// a synchronous `cuMemcpyDtoH_v2`. These helpers swap that for the pinned
// `to_pinned_async` + `stream.synchronize()` + host-host copy sequence,
// matching the Stage-3 pattern in `aggregate.rs::reduce_gpu_vec`.
//
// They are monomorphised on the element type so they can land directly in an
// `AccDownload` variant without further casts. The trailing
// `as_slice().to_vec()` is the one unavoidable host-host copy — Arrow arrays
// cannot be built directly on top of a `PinnedHostBuffer`.
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn download_pinned_i32(v: &GpuVec<i32>, stream: &CudaStream) -> BoltResult<Vec<i32>> {
    let pinned = v.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    Ok(pinned.as_slice().to_vec())
}

#[inline]
pub(crate) fn download_pinned_i64(v: &GpuVec<i64>, stream: &CudaStream) -> BoltResult<Vec<i64>> {
    let pinned = v.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    Ok(pinned.as_slice().to_vec())
}

#[inline]
pub(crate) fn download_pinned_f32(v: &GpuVec<f32>, stream: &CudaStream) -> BoltResult<Vec<f32>> {
    let pinned = v.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    Ok(pinned.as_slice().to_vec())
}

#[inline]
pub(crate) fn download_pinned_f64(v: &GpuVec<f64>, stream: &CudaStream) -> BoltResult<Vec<f64>> {
    let pinned = v.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    Ok(pinned.as_slice().to_vec())
}

// ---------------------------------------------------------------------------
// Key column extraction, packing, and decoding.
//
// dedup (groupby_common): consolidated from `groupby.rs` and
// `groupby_valid.rs`. The canonical copy is the classic `groupby.rs` variant
// (the more-hardened superset): it keeps the `KeyComponent::name` field and
// the V-17 `wrapping_shl` in `pack_keys`. `groupby_valid.rs` previously had a
// `name`-less `KeyComponent`; adopting the superset is behavior-preserving
// there because its production path never reads `name`.
// ---------------------------------------------------------------------------

/// Number of bits a column's encoded value occupies inside a packed i64 key.
/// Anything wider than 64 bits total is rejected (composite-hash fallback is
/// not implemented in v1).
///
/// dedup (groupby_common): consolidated from `groupby.rs` / `groupby_valid.rs`
/// (the two copies were byte-identical).
pub(crate) fn key_bit_width(dtype: DataType) -> BoltResult<u32> {
    match dtype {
        DataType::Int32 | DataType::Float32 => Ok(32),
        DataType::Int64 | DataType::Float64 => Ok(64),
        DataType::Bool | DataType::Utf8 | DataType::Decimal128(_, _) | DataType::Date32 | DataType::Timestamp(_, _) => Err(BoltError::Type(format!(
            "GROUP BY key dtype {:?} not supported in v1",
            dtype
        ))),
    }
}

/// Per-group-by-column metadata: the original dtype and the bit offset at
/// which this column's value is packed inside the i64 key (low = 0).
///
/// dedup (groupby_common): the canonical (superset) variant. The `name` field
/// is read by `groupby.rs`'s `pack_keys` unit tests and surfaced in the
/// two-column row-count-mismatch error message; `groupby_valid.rs` does not
/// read it but carrying it is inert there.
#[derive(Debug, Clone)]
pub(crate) struct KeyComponent {
    /// Column name (used for error messages and field naming).
    // used by: pack_keys_two_int32 test (asserts component identity)
    #[allow(dead_code)]
    pub(crate) name: String,
    /// Original dtype as declared by the plan (drives encode/decode).
    pub(crate) original_dtype: DataType,
    /// Bit position within the i64 key — low = 0, high = 32 for the
    /// two-column pack. Single-column keys always use offset 0.
    pub(crate) bit_offset: u32,
}

/// Result of [`pack_keys`]: the per-row encoded i64 key column plus enough
/// metadata to decode each unique slot back into its constituent columns.
///
/// dedup (groupby_common): consolidated from `groupby.rs` / `groupby_valid.rs`.
#[derive(Debug)]
pub(crate) struct PackedKeys {
    /// i64-encoded keys ready to upload, one entry per input row. NULL-key
    /// rows (per `key_valid`) carry an undefined bit pattern at their slot
    /// because the per-column loader propagates garbage values; callers MUST
    /// filter via `key_valid` before forwarding to the GPU.
    pub(crate) keys_i64: Vec<i64>,
    /// Per-column dtype + original ordinal in `aggregate.inputs`, in pack order.
    pub(crate) components: Vec<KeyComponent>,
    /// Per-row keep mask: `true` iff every GROUP BY key column is non-null at
    /// that row. `None` means "all key columns have zero nulls" — the fast
    /// path where no filtering is required.
    pub(crate) key_valid: Option<Vec<bool>>,
}

/// Typed per-column key value as decoded from a packed i64.
///
/// dedup (groupby_common): consolidated from `groupby.rs` / `groupby_valid.rs`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum KeyValue {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
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
///
/// Review C12: `-0.0` is canonicalised to `+0.0` (see [`canonicalise_f32`] /
/// [`canonicalise_f64`]) so signed-zero pairs hash into one group. F3: every
/// NaN bit pattern is also folded to a single canonical quiet NaN so all NaN
/// keys collapse into one group, matching DuckDB.
///
/// dedup (groupby_common): consolidated from `groupby.rs` / `groupby_valid.rs`
/// (the two copies were byte-identical apart from doc wording).
pub(crate) fn load_key_column_bits(
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
            // Review C12: canonicalise -0.0 -> +0.0 so signed-zero pairs hash
            // into one group. F3: all NaN payloads fold to one canonical quiet
            // NaN so NaN keys collapse into a single group (DuckDB semantics).
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
        DataType::Bool | DataType::Utf8 | DataType::Decimal128(_, _) | DataType::Date32 | DataType::Timestamp(_, _) => {
            return Err(BoltError::Type(format!(
                "GROUP BY key dtype {:?} not supported in v1",
                key_io.dtype
            )))
        }
    };
    Ok((bits, null_mask))
}

/// Canonicalise a float GROUP BY / DISTINCT key so that values which SQL
/// (DuckDB) treats as equal hash into one group:
///
/// * `-0.0` is folded to `+0.0` so signed-zero pairs key the same group
///   (review C12).
/// * Every NaN bit pattern (any payload, quiet or signalling, either sign)
///   is folded to the single canonical quiet NaN (`f64::NAN`). This matches
///   DuckDB, which treats all NaN as one GROUP BY / DISTINCT key (F3). Without
///   this, two NaNs with different payloads — or even the negative-NaN half —
///   would form distinct groups, diverging from the reference engine.
///
/// All float key operators (GROUP BY, DISTINCT, JOIN) must agree on this
/// equivalence relation. `join::extract_key` carries its own copy; the
/// `distinct::` / `groupby_wide::` copies are kept in lock-step.
///
/// dedup (groupby_common): consolidated from `groupby.rs` / `groupby_valid.rs`
/// (the two copies were byte-identical).
#[inline]
pub(crate) fn canonicalise_f64(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

/// `f32` analogue of [`canonicalise_f64`].
///
/// dedup (groupby_common): consolidated from `groupby.rs` / `groupby_valid.rs`.
#[inline]
pub(crate) fn canonicalise_f32(x: f32) -> f32 {
    if x.is_nan() {
        f32::NAN
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

/// Encode each group-by column (in `aggregate.group_by` order) into a single
/// i64 key per row. Returns the encoded keys plus per-column metadata for the
/// reverse trip in `build_key_arrays` / `build_key_arrays_from_entries`.
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
///
/// dedup (groupby_common): consolidated from `groupby.rs` / `groupby_valid.rs`.
/// The canonical body is the V-17-hardened one — it uses `wrapping_shl` so a
/// future regression where `shift == 64` produces a deterministic `0` instead
/// of triggering UB / a debug panic on a bare `<<` overshift. This is the
/// single source of truth that previously had to be fixed twice.
pub(crate) fn pack_keys(
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
        // V-17: use `wrapping_shl` so that a future regression where
        // `shift == 64` (e.g. a 32-bit dtype appended after a 64-bit field)
        // produces a deterministic 0 instead of triggering UB / a debug panic
        // on a bare `<<` overshift. For the supported case (`shift` in 0..=32
        // with 32-bit widths, or `shift == 0` with a 64-bit width) this is
        // identical to `raw << shift`. Consolidating here (dedup —
        // groupby_common) is what removes the "fix-it-twice" drift class: the
        // hardening previously had to be applied to `groupby.rs` and then
        // re-applied to `groupby_valid.rs` separately.
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
///
/// dedup (groupby_common): consolidated from `groupby.rs` / `groupby_valid.rs`
/// (the two copies were byte-identical).
pub(crate) fn and_masks(a: Option<Vec<bool>>, b: Option<Vec<bool>>) -> Option<Vec<bool>> {
    match (a, b) {
        (None, None) => None,
        (Some(m), None) | (None, Some(m)) => Some(m),
        (Some(ma), Some(mb)) => {
            debug_assert_eq!(ma.len(), mb.len(), "and_masks: length mismatch");
            Some(ma.iter().zip(mb.iter()).map(|(x, y)| *x && *y).collect())
        }
    }
}

/// Reverse of [`pack_keys`] for a single packed i64 key: extract each
/// component value as a typed [`KeyValue`] in the same order the components
/// were packed (i.e. matching `aggregate.group_by` order).
///
/// dedup (groupby_common): consolidated from `groupby.rs` / `groupby_valid.rs`.
/// All copies agreed on the bare `>>` here (no overshift risk — the shifts are
/// only ever 0 or 32 for 32-bit fields, and the 64-bit field uses the whole
/// word), so the bare shift is kept verbatim.
pub(crate) fn decode_key(packed: i64, components: &[KeyComponent]) -> Vec<KeyValue> {
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
            DataType::Bool | DataType::Utf8 | DataType::Decimal128(_, _) | DataType::Date32 | DataType::Timestamp(_, _) => {
                // pack_keys would have already rejected these dtypes — we
                // can't reach this branch from the executors, but keep the
                // match exhaustive and return a 0 placeholder.
                0
            }
        };
        let val = match comp.original_dtype {
            DataType::Int32 => KeyValue::I32(raw as u32 as i32),
            DataType::Int64 => KeyValue::I64(raw as i64),
            DataType::Float32 => KeyValue::F32(f32::from_bits(raw as u32)),
            DataType::Float64 => KeyValue::F64(f64::from_bits(raw)),
            DataType::Bool | DataType::Utf8 | DataType::Decimal128(_, _) | DataType::Date32 | DataType::Timestamp(_, _) => KeyValue::I64(0),
        };
        out.push(val);
    }
    out
}

/// Count of distinct values in `keys`. Linear, O(n) using a HashSet.
///
/// dedup (groupby_common): consolidated from the byte-identical copies in
/// `groupby.rs`, `groupby_valid.rs`, and `groupby_with_pre.rs`.
pub(crate) fn unique_count(keys: &[i64]) -> usize {
    let mut set: HashSet<i64> = HashSet::with_capacity(keys.len().min(1 << 20));
    for &k in keys {
        set.insert(k);
    }
    set.len()
}

/// Smallest power of two >= `n` (saturating to `usize::MAX`'s previous pow-2
/// on overflow).
///
/// dedup (groupby_common): consolidated from the byte-identical copies in
/// `groupby.rs`, `groupby_valid.rs`, and `groupby_with_pre.rs`.
pub(crate) fn next_pow2(n: usize) -> usize {
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

/// `Type` error for a failed Arrow downcast on column `name`.
///
/// dedup (groupby_common): local copy used by [`load_key_column_bits`]; the
/// executors keep their own identically-worded `downcast_err` for the other
/// (non-shared) downcast sites.
fn downcast_err(name: &str, expected: &str) -> BoltError {
    BoltError::Type(format!(
        "GROUP BY input column '{}' could not be downcast to {}",
        name, expected
    ))
}

/// Map Arrow `DataType` to our plan `DataType` (mirrors `aggregate.rs`).
///
/// dedup (groupby_common): used by [`load_key_column_bits`]; delegates to the
/// shared `schema_convert` helper exactly as the executors' local
/// `arrow_dtype_to_plan` aliases did.
fn arrow_dtype_to_plan(
    d: &arrow_schema::DataType,
) -> BoltResult<DataType> {
    crate::exec::schema_convert::arrow_dtype_to_plan_basic(d, "")
}

// ---------------------------------------------------------------------------
// Tests for the consolidated key-packing helpers.
//
// dedup (groupby_common): centralised here from the formerly-duplicated test
// modules in `groupby.rs` and `groupby_valid.rs`. Coverage is the UNION of
// both sites (no coverage was dropped): single/two-column Int32, Float32,
// Int64, the >64-bit reject paths, the V-17 round-trip pins, the NULL-mask
// surfacing tests, and the `and_masks` combinator table. The executors keep
// their own `#[cfg(test)]` modules for the helpers that stayed local.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, RecordBatch};
    use arrow_schema::{
        DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
    };

    use crate::plan::logical_plan::Schema;
    use crate::plan::physical_plan::AggregateSpec;

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
            input_has_validity: Vec::new(),
        }
    }

    /// Build a single-column RecordBatch from a typed Arrow array. The field
    /// is marked nullable so callers can pass arrays with NULL validity
    /// bitmaps.
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
    ///    then `as i64`. Verified on the worked example a=7, b=-3. Pins the
    ///    `KeyComponent::name` field too (only the classic `groupby.rs` copy
    ///    carried it; now it's the canonical superset).
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

    /// 3. Single Float32 key packs to `canonicalise_f32(f).to_bits() as i64`
    ///    and round-trips via `decode_key` back to the canonicalised f32 bit
    ///    pattern. Review C12: `-0.0` packs to the same key as `+0.0`.
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
            // The packed key must equal the CANONICALISED bit pattern,
            // not the raw `v.to_bits()` — -0.0 collapses to +0.0.
            let canon = if *v == 0.0f32 { 0.0f32 } else { *v };
            let expected = (canon.to_bits() as u64) as i64;
            assert_eq!(packed.keys_i64[i], expected, "row {i}");
            let dec = decode_key(packed.keys_i64[i], &packed.components);
            assert_eq!(dec.len(), 1);
            match dec[0] {
                KeyValue::F32(f) => assert_eq!(f.to_bits(), canon.to_bits()),
                other => panic!("expected F32, got {:?}", other),
            }
        }

        // Review C12: -0.0 and +0.0 are canonicalised to the same key so
        // they group together (matches SQL/IEEE and DuckDB). The test
        // pins this convention.
        assert_eq!(packed.keys_i64[2], packed.keys_i64[3]);
    }

    /// F3: every NaN bit pattern (distinct payloads, both signs, quiet and
    /// signalling) is folded by `canonicalise_f32`/`f64` to a single canonical
    /// quiet NaN, so all NaN GROUP BY keys pack to the SAME `keys_i64` value
    /// and therefore collapse into one group — matching DuckDB. A finite value
    /// must remain a distinct key. This pins the GROUP-BY half of the F3
    /// decision (the DISTINCT half is pinned in `distinct.rs`).
    #[test]
    fn pack_keys_nan_collapse_f32() {
        let agg = spec(vec![("f", DataType::Float32)], vec![0]);
        let payload_nan = f32::from_bits(0x7fc0_0001); // quiet NaN, payload 1
        let signalling_nan = f32::from_bits(0x7f80_0001); // signalling NaN
        let neg_nan = f32::from_bits(0xffc0_0000); // negative NaN
        let plain_nan = f32::NAN;
        let finite = 1.5f32;
        let vals = vec![payload_nan, signalling_nan, neg_nan, plain_nan, finite];
        let batch = one_col_batch(
            "f",
            Arc::new(Float32Array::from(vals)) as ArrayRef,
        );
        let packed = pack_keys(&agg, &batch).expect("pack ok");
        // All four NaN rows pack to one key.
        let nan_key = packed.keys_i64[0];
        assert_eq!(packed.keys_i64[1], nan_key, "signalling NaN must collapse");
        assert_eq!(packed.keys_i64[2], nan_key, "negative NaN must collapse");
        assert_eq!(packed.keys_i64[3], nan_key, "plain NaN must collapse");
        // The canonical key is exactly `f32::NAN.to_bits()`.
        assert_eq!(nan_key, (f32::NAN.to_bits() as u64) as i64);
        // A finite value is a distinct group.
        assert_ne!(packed.keys_i64[4], nan_key, "finite value must NOT join NaN group");
    }

    /// F3 (f64 lane): same NaN-collapse contract as `pack_keys_nan_collapse_f32`.
    #[test]
    fn pack_keys_nan_collapse_f64() {
        let agg = spec(vec![("f", DataType::Float64)], vec![0]);
        let payload_nan = f64::from_bits(0x7ff8_0000_0000_0001);
        let signalling_nan = f64::from_bits(0x7ff0_0000_0000_0001);
        let neg_nan = f64::from_bits(0xfff8_0000_0000_0000);
        let plain_nan = f64::NAN;
        let finite = -2.25f64;
        let vals = vec![payload_nan, signalling_nan, neg_nan, plain_nan, finite];
        let batch = one_col_batch(
            "f",
            Arc::new(Float64Array::from(vals)) as ArrayRef,
        );
        let packed = pack_keys(&agg, &batch).expect("pack ok");
        let nan_key = packed.keys_i64[0];
        assert_eq!(packed.keys_i64[1], nan_key, "signalling NaN must collapse");
        assert_eq!(packed.keys_i64[2], nan_key, "negative NaN must collapse");
        assert_eq!(packed.keys_i64[3], nan_key, "plain NaN must collapse");
        assert_eq!(nan_key, f64::NAN.to_bits() as i64);
        assert_ne!(packed.keys_i64[4], nan_key, "finite value must NOT join NaN group");
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

    /// V-17: two 32-bit keys pack into a single i64 (low at offset 0, high at
    /// offset 32) and `decode_key` recovers the originals byte-for-byte. This
    /// pins the packing semantics for the supported shifts (0 and 32) so the
    /// `wrapping_shl` hardening can never silently change the encoding.
    #[test]
    fn pack_keys_two_i32_round_trips() {
        let agg = spec(
            vec![("hi", DataType::Int32), ("lo", DataType::Int32)],
            vec![0, 1],
        );
        // `pack_keys` treats the first group-by column as the high half
        // (offset 32) and the second as the low half (offset 0).
        let hi: ArrayRef = Arc::new(Int32Array::from(vec![7i32, -3, 0]));
        let lo: ArrayRef = Arc::new(Int32Array::from(vec![11i32, 42, -1]));
        let batch = two_col_batch("hi", hi, "lo", lo);
        let packed = pack_keys(&agg, &batch).expect("pack ok");
        assert!(packed.key_valid.is_none(), "null-free input");
        let expected = [(7i32, 11i32), (-3, 42), (0, -1)];
        for (row, &(eh, el)) in expected.iter().enumerate() {
            let decoded = decode_key(packed.keys_i64[row], &packed.components);
            assert_eq!(decoded.len(), 2, "two components expected");
            assert_eq!(decoded[0], KeyValue::I32(eh), "high half round-trip");
            assert_eq!(decoded[1], KeyValue::I32(el), "low half round-trip");
        }
    }

    /// V-17: a single 64-bit key uses shift 0 (no overshift) and round-trips.
    #[test]
    fn pack_keys_single_i64_round_trips() {
        let agg = spec(vec![("k", DataType::Int64)], vec![0]);
        let arr: ArrayRef =
            Arc::new(Int64Array::from(vec![0i64, -1, i64::MAX, i64::MIN]));
        let batch = one_col_batch("k", arr);
        let packed = pack_keys(&agg, &batch).expect("pack ok");
        for (row, &v) in [0i64, -1, i64::MAX, i64::MIN].iter().enumerate() {
            let decoded = decode_key(packed.keys_i64[row], &packed.components);
            assert_eq!(decoded, vec![KeyValue::I64(v)], "i64 round-trip");
        }
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

    /// Sentinel-collision row coverage (from `groupby_valid.rs`): a Float64
    /// `-0.0` canonicalises to `+0.0` (bit pattern 0, not i64::MIN). With a
    /// NULL row in the same batch, the mask MUST flag the NULL row and the
    /// `-0.0` rows must survive the post-mask filter.
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
        // -0.0 rows survive (canonicalised to +0.0), the NULL row is
        // dropped, the 1.5 row survives.
        let filtered: Vec<i64> = packed
            .keys_i64
            .iter()
            .zip(mask.iter())
            .filter_map(|(k, &keep)| if keep { Some(*k) } else { None })
            .collect();
        assert_eq!(filtered.len(), 3);
        // -0.0 is canonicalised to +0.0 (see distinct::canonicalise_f64) so
        // its bit pattern is 0, not i64::MIN. Same equivalence rule applies
        // here so GROUP BY, DISTINCT, and JOIN agree on float identity.
        assert_eq!(filtered[0], 0);
        assert_eq!(filtered[2], 0);
        assert_eq!(filtered[1], 1.5f64.to_bits() as i64);
    }

    /// `next_pow2` rounds up to the next power of two and saturates at 1 for
    /// small inputs.
    #[test]
    fn next_pow2_basic() {
        assert_eq!(next_pow2(0), 1);
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(17), 32);
    }

    /// `unique_count` counts distinct values.
    #[test]
    fn unique_count_basic() {
        assert_eq!(unique_count(&[]), 0);
        assert_eq!(unique_count(&[1, 1, 1]), 1);
        assert_eq!(unique_count(&[1, 2, 3, 2, 1]), 3);
    }
}
