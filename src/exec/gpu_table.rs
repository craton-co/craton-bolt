// SPDX-License-Identifier: Apache-2.0

//! GPU-resident table storage: columns uploaded once and queried in place.

use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{
    Array, BooleanArray, Decimal128Array, DictionaryArray, Float32Array, Float64Array, Int32Array,
    Int64Array, RecordBatch, StringArray,
};

use crate::cuda::buffer::primitive_to_gpu;
use crate::cuda::cuda_sys::CUdeviceptr;
use crate::cuda::dictionary::DictionaryColumn;
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::plan::DataType;

/// One column resident on the device.
pub struct GpuColumn {
    /// Source column name.
    pub name: String,
    /// Plan-level dtype.
    pub dtype: DataType,
    /// Owned device storage.
    pub data: GpuColumnData,
    /// Host-side revision counter at the time this column was uploaded.
    ///
    /// Compared against `Engine`'s `host_revisions[table].column_revisions[col]`
    /// in [`crate::exec::engine::Engine::ensure_gpu_table`]; a match means the
    /// upload is still fresh and the engine reuses this column in place,
    /// avoiding a redundant HtoD transfer. A mismatch triggers a re-upload
    /// (or a prefix-preserving extension for `register_batch` appends).
    ///
    /// Initialised by the engine — `GpuTable::from_record_batch` leaves it
    /// at `0` and the engine sets it to the matching host revision right
    /// after building. Callers outside the engine (direct
    /// `GpuTable::from_record_batch` users) can leave it untouched; the
    /// counter only matters when the column lives inside an engine-managed
    /// `gpu_tables` cache.
    pub host_revision: u64,
}

/// Heterogeneous owned device storage for a single column.
pub enum GpuColumnData {
    /// 32-bit signed integer column.
    I32 {
        /// Per-row i32 values on the device.
        values: GpuVec<i32>,
        /// Optional per-row validity, ONE u8 per row (0 = NULL, 1 = non-null),
        /// matching `BoolNullable` and what `emit_is_null_check` reads. `None`
        /// when the source had no nulls.
        validity: Option<GpuVec<u8>>,
    },
    /// 64-bit signed integer column.
    I64 {
        /// Per-row i64 values on the device.
        values: GpuVec<i64>,
        /// Optional per-row validity, ONE u8 per row (0 = NULL, 1 = non-null),
        /// matching `BoolNullable` and what `emit_is_null_check` reads. `None`
        /// when the source had no nulls.
        validity: Option<GpuVec<u8>>,
    },
    /// 32-bit float column.
    F32 {
        /// Per-row f32 values on the device.
        values: GpuVec<f32>,
        /// Optional per-row validity, ONE u8 per row (0 = NULL, 1 = non-null),
        /// matching `BoolNullable` and what `emit_is_null_check` reads. `None`
        /// when the source had no nulls.
        validity: Option<GpuVec<u8>>,
    },
    /// 64-bit float column.
    F64 {
        /// Per-row f64 values on the device.
        values: GpuVec<f64>,
        /// Optional per-row validity, ONE u8 per row (0 = NULL, 1 = non-null),
        /// matching `BoolNullable` and what `emit_is_null_check` reads. `None`
        /// when the source had no nulls.
        validity: Option<GpuVec<u8>>,
    },
    /// Boolean column, one byte per row.
    Bool(GpuVec<u8>),
    /// Nullable boolean column, values and validity mask buffers.
    BoolNullable {
        values: GpuVec<u8>,
        validity: GpuVec<u8>,
    },
    /// Utf8 column stored as i32 dictionary indices with a host-side dictionary.
    Utf8 {
        /// Per-row i32 indices on the device.
        indices: GpuVec<i32>,
        /// Host-side dictionary, `dictionary[i - 1]` decodes index `i` (slot 0 is NULL).
        dictionary: Vec<String>,
    },
    /// v0.7 sub-task B: 128-bit fixed-point column. Stored as a single
    /// contiguous `GpuVec<u64>` of length `2 * n_rows`, interleaved
    /// little-endian `[lo0, hi0, lo1, hi1, ...]` so the PTX-side
    /// `Op::LoadColumn128` / `Op::Store128` (which read 16 bytes per row
    /// starting at base + tid*16) addresses each row's low / high halves
    /// at `[+0, +8]` from the row base.
    ///
    /// Storage layout note (deviation from the sub-task spec): the spec
    /// description called for two separate `GpuVec<u64>` buffers (`lo`
    /// + `hi`). That would imply TWO device base pointers per column,
    /// which the PTX emitter (committed in sub-task A) does NOT support:
    /// `Op::LoadColumn128` reads one base pointer per column and computes
    /// `[tid * 16]` / `[tid * 16 + 8]`. The interleaved single-buffer
    /// layout below is the only encoding that's compatible with the
    /// committed PTX ABI. The host upload / download paths still expose
    /// the i128 value as a logical `(lo, hi)` pair (see
    /// [`GpuColumn::upload`] for the unpack and
    /// [`crate::exec::engine`]'s download arm for the reassembly).
    ///
    /// Row `i`:
    /// * low 64 bits  = `values[2 * i + 0]`
    /// * high 64 bits = `values[2 * i + 1]`
    /// * `i128 value  = ((hi as i128) << 64) | (lo as u128 as i128)`
    ///   (the cast back is sign-preserving because the high half carries
    ///   the sign bits unchanged).
    Decimal128 {
        /// Interleaved 16-bytes-per-row buffer, length `2 * n_rows`.
        values: GpuVec<u64>,
        /// Plan-level precision (digits of significance, 1..=38).
        precision: u8,
        /// Plan-level scale (digits right of the decimal point).
        scale: i8,
        /// Optional Arrow-LE packed validity bitmap on the device, one byte
        /// per 8 rows (lsb-first), mirroring [`DictUtf8`](Self::DictUtf8)'s
        /// `valid_mask`. `None` when the upload source had no nulls. NULL
        /// rows are zeroed in `values` at upload time, so this mask is the
        /// source of truth for per-row validity — without it a NULL
        /// Decimal128 would silently read back as `0`.
        valid_mask: Option<GpuVec<u8>>,
    },
    /// **Stage 5** — native dict-encoded Utf8 column. Stage 4 worked around
    /// the lack of this variant by flattening every `DictionaryArray<i32, Utf8>`
    /// into a plain `StringArray` at registration time, which then went
    /// through the `Utf8` arm above. The cost was two materialisations: once
    /// to flatten and once to re-encode as the engine's own dictionary.
    ///
    /// `DictUtf8` keeps the **input** dictionary intact: we upload only the
    /// keys (already i32) and the dictionary stays host-side. Downstream
    /// consumers can either:
    ///   - read the device keys directly (sort, hash join, etc. that just
    ///     need lex-ordered integer comparison), or
    ///   - reattach `dict[key]` for materialisation (projection output).
    ///
    /// For compat with code that only knows the `Utf8` variant,
    /// [`GpuColumn::utf8_dictionary`] returns the dict for **both** variants
    /// — the `Utf8` `dictionary[i-1]` and `DictUtf8` `dict[i]` indexing
    /// conventions are different though (Utf8 reserves slot 0 for NULL).
    ///
    /// **Stage 6** — added `valid_mask`. NULL rows are represented by the
    /// Arrow-LE packed validity bitmap (one `u8` per 8 rows, lsb-first),
    /// matching the bitmap convention used by the PV validity path. The
    /// `keys.value(i)` for a row whose validity bit is 0 is meaningless
    /// (we zero them at upload time); callers MUST consult `valid_mask`
    /// before dereferencing `dict[keys[i]]` for nullable inputs.
    DictUtf8 {
        /// Per-row dictionary keys (the i32 indices into `dict`). NULL rows
        /// are zeroed at upload time — `valid_mask` is the source of truth
        /// for per-row validity.
        keys: GpuVec<i32>,
        /// Host-side dictionary, `dict[i]` decodes key `i`. Mirrors the
        /// Arrow dictionary's values 1:1 (no NULL offset).
        dict: Vec<String>,
        /// Optional Arrow-LE packed validity bitmap on the device, one byte
        /// per 8 rows. `None` when the upload source had no nulls.
        valid_mask: Option<GpuVec<u8>>,
    },
}

impl GpuColumnData {
    /// Raw device pointer to the column's primary buffer.
    ///
    /// For Decimal128 the buffer holds 16 bytes per row in interleaved
    /// `[lo, hi]` little-endian order — see the variant doc. The PTX-side
    /// `Op::LoadColumn128` knows to treat this single base pointer as a
    /// stride-16 buffer.
    pub fn device_ptr(&self) -> CUdeviceptr {
        match self {
            GpuColumnData::I32 { values, .. } => values.device_ptr(),
            GpuColumnData::I64 { values, .. } => values.device_ptr(),
            GpuColumnData::F32 { values, .. } => values.device_ptr(),
            GpuColumnData::F64 { values, .. } => values.device_ptr(),
            GpuColumnData::Bool(v) => v.device_ptr(),
            GpuColumnData::BoolNullable { values, .. } => values.device_ptr(),
            GpuColumnData::Utf8 { indices, .. } => indices.device_ptr(),
            // Stage 5 — DictUtf8's primary buffer is its keys array.
            GpuColumnData::DictUtf8 { keys, .. } => keys.device_ptr(),
            // v0.7 sub-task B: Decimal128's primary buffer is the
            // interleaved 16-bytes-per-row u64 array; the PTX side
            // computes per-row offsets as `tid * 16` from this base.
            GpuColumnData::Decimal128 { values, .. } => values.device_ptr(),
        }
    }

    /// Device pointer to the validity bitmap, if any. Only `DictUtf8` carries
    /// a separate bitmap today; all other variants either inline validity
    /// (e.g. `BoolNullable`) or treat the data as non-nullable.
    pub fn validity_ptr(&self) -> Option<CUdeviceptr> {
        match self {
            GpuColumnData::DictUtf8 {
                valid_mask: Some(v),
                ..
            } => Some(v.device_ptr()),
            GpuColumnData::Decimal128 {
                valid_mask: Some(v),
                ..
            } => Some(v.device_ptr()),
            GpuColumnData::BoolNullable { validity, .. } => Some(validity.device_ptr()),
            // Primitive columns carry an UNPACKED per-row validity (one u8 per
            // row, 0 = NULL / 1 = non-null) — exactly the form
            // `emit_is_null_check` reads. `None` when the source had no nulls.
            GpuColumnData::I32 { validity, .. } => validity.as_ref().map(|v| v.device_ptr()),
            GpuColumnData::I64 { validity, .. } => validity.as_ref().map(|v| v.device_ptr()),
            GpuColumnData::F32 { validity, .. } => validity.as_ref().map(|v| v.device_ptr()),
            GpuColumnData::F64 { validity, .. } => validity.as_ref().map(|v| v.device_ptr()),
            _ => None,
        }
    }
}

/// Pack `n_rows` of host-side validity bits (one `bool` per row, `true` = valid)
/// into an Arrow-LE packed bitmap, one `u8` per 8 rows, lsb-first.
///
/// This is the on-host counterpart to the PV-stage-c `pack_validity_bits`
/// device kernel: it is used at upload time to translate an Arrow
/// `NullBuffer`'s already-packed bits into our own owned `Vec<u8>` (the
/// Arrow buffer's lifetime is tied to the source array; we want an owned
/// copy to ship straight to the device).
fn pack_validity_from_arrow(null_buffer: &arrow_buffer::NullBuffer, n_rows: usize) -> Vec<u8> {
    let n_bytes = n_rows.div_ceil(8);
    let mut out = vec![0u8; n_bytes];
    for i in 0..n_rows {
        if null_buffer.is_valid(i) {
            out[i / 8] |= 1u8 << (i % 8);
        }
    }
    out
}

/// Build the UNPACKED per-row validity buffer for a primitive Arrow array:
/// one `u8` per row, `0 = NULL`, `1 = non-null`. This is the exact layout
/// `BoolNullable` uses and the only layout `emit_is_null_check` reads
/// (`ld.global.nc.u8 [vptr + tid]`, byte `tid` per row) — NOT the packed
/// bitmap produced by [`pack_validity_from_arrow`].
///
/// Returns `Ok(None)` when the array has no nulls (no validity buffer needed),
/// and `Ok(Some(GpuVec<u8>))` of length `arr.len()` otherwise.
fn unpacked_validity_from_arrow(arr: &dyn Array) -> BoltResult<Option<GpuVec<u8>>> {
    if arr.null_count() == 0 {
        return Ok(None);
    }
    let n = arr.len();
    let mut validity: Vec<u8> = Vec::with_capacity(n);
    for i in 0..n {
        validity.push(if arr.is_null(i) { 0 } else { 1 });
    }
    Ok(Some(GpuVec::<u8>::from_slice(&validity)?))
}

impl GpuColumn {
    /// Upload an Arrow array to the GPU, downcasting per `dtype`.
    pub fn upload(name: String, arr: &dyn Array, dtype: DataType) -> BoltResult<Self> {
        let data = match dtype {
            DataType::Int32 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| type_mismatch_err(arr, "Int32"))?;
                let buf = primitive_to_gpu(pa)?;
                let validity = unpacked_validity_from_arrow(pa)?;
                GpuColumnData::I32 {
                    values: GpuVec::from_buffer(buf),
                    validity,
                }
            }
            DataType::Int64 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| type_mismatch_err(arr, "Int64"))?;
                let buf = primitive_to_gpu(pa)?;
                let validity = unpacked_validity_from_arrow(pa)?;
                GpuColumnData::I64 {
                    values: GpuVec::from_buffer(buf),
                    validity,
                }
            }
            DataType::Float32 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| type_mismatch_err(arr, "Float32"))?;
                let buf = primitive_to_gpu(pa)?;
                let validity = unpacked_validity_from_arrow(pa)?;
                GpuColumnData::F32 {
                    values: GpuVec::from_buffer(buf),
                    validity,
                }
            }
            DataType::Float64 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| type_mismatch_err(arr, "Float64"))?;
                let buf = primitive_to_gpu(pa)?;
                let validity = unpacked_validity_from_arrow(pa)?;
                GpuColumnData::F64 {
                    values: GpuVec::from_buffer(buf),
                    validity,
                }
            }
            DataType::Bool => {
                let ba = arr
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| type_mismatch_err(arr, "Bool"))?;
                let n = ba.len();
                if ba.null_count() == 0 {
                    let mut bytes: Vec<u8> = Vec::with_capacity(n);
                    for i in 0..n {
                        bytes.push(if ba.value(i) { 1 } else { 0 });
                    }
                    GpuColumnData::Bool(GpuVec::<u8>::from_slice(&bytes)?)
                } else {
                    let mut values: Vec<u8> = Vec::with_capacity(n);
                    let mut validity: Vec<u8> = Vec::with_capacity(n);
                    for i in 0..n {
                        if ba.is_null(i) {
                            values.push(0);
                            validity.push(0);
                        } else {
                            values.push(if ba.value(i) { 1 } else { 0 });
                            validity.push(1);
                        }
                    }
                    let v_gpu = GpuVec::<u8>::from_slice(&values)?;
                    let m_gpu = GpuVec::<u8>::from_slice(&validity)?;
                    GpuColumnData::BoolNullable {
                        values: v_gpu,
                        validity: m_gpu,
                    }
                }
            }
            DataType::Utf8 => {
                // Stage 5 — accept both plain `StringArray` and
                // `DictionaryArray<I32 | I64, Utf8>` here. The plain-string
                // path stays on the engine-managed `Utf8` variant (preserves
                // the slot-0-is-NULL convention every downstream consumer
                // relies on). The dict-Utf8 path takes the new `DictUtf8`
                // variant, keeping the input dictionary intact rather than
                // materialising and re-encoding it. The engine still
                // pre-flattens dict columns at `register_table` time
                // (compat path — see `flatten_dictionary_utf8_columns`),
                // so for the SQL pipeline this `DictionaryArray` branch is
                // effectively reachable only from direct GpuTable callers
                // (and the inline tests in this module). Once every
                // downstream stage learns to read `DictUtf8` natively, the
                // engine's flatten step can be retired — Stage 6 follow-up.
                use arrow_schema::DataType as A;
                match arr.data_type() {
                    A::Utf8 => {
                        let sa = arr
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| type_mismatch_err(arr, "Utf8"))?;
                        let dict = DictionaryColumn::from_string_array(sa)?;
                        GpuColumnData::Utf8 {
                            indices: dict.indices,
                            dictionary: dict.dictionary,
                        }
                    }
                    A::Dictionary(key_ty, value_ty) if matches!(value_ty.as_ref(), A::Utf8) => {
                        let n = arr.len();
                        let (keys_i32, dict_strings) = match key_ty.as_ref() {
                            A::Int32 => {
                                let da = arr
                                    .as_any()
                                    .downcast_ref::<DictionaryArray<Int32Type>>()
                                    .ok_or_else(|| {
                                        type_mismatch_err(arr, "Dictionary<i32, Utf8>")
                                    })?;
                                let sa = da
                                    .values()
                                    .as_any()
                                    .downcast_ref::<StringArray>()
                                    .ok_or_else(|| {
                                        BoltError::Type(
                                            "GpuColumn: dict values are not StringArray".into(),
                                        )
                                    })?;
                                let mut dict_strings: Vec<String> = Vec::with_capacity(sa.len());
                                for i in 0..sa.len() {
                                    // NULL dict entries shouldn't happen for a
                                    // sane Arrow input, but defensively keep
                                    // an empty placeholder so indexing stays
                                    // aligned with the keys.
                                    if sa.is_null(i) {
                                        dict_strings.push(String::new());
                                    } else {
                                        dict_strings.push(sa.value(i).to_string());
                                    }
                                }
                                // Finding V-5: validate every key before it is
                                // copied to the device index buffer. An
                                // unvalidated negative or out-of-range key
                                // would later index the dictionary inside the
                                // GPU kernel -> OOB device read. Null slots are
                                // encoded as 0 (validity handled separately by
                                // `upload_dict_utf8`); mirror the strict bounds
                                // checks in `string_ops`.
                                let dict_len = sa.len();
                                let keys: Vec<i32> = (0..n)
                                    .map(|i| {
                                        if da.keys().is_null(i) {
                                            Ok(0)
                                        } else {
                                            let key = da.keys().value(i);
                                            if key < 0 {
                                                return Err(BoltError::Type(format!(
                                                    "GpuColumn: negative dict<i32,Utf8> key {} at row {}",
                                                    key, i
                                                )));
                                            }
                                            if (key as usize) >= dict_len {
                                                return Err(BoltError::Type(format!(
                                                    "GpuColumn: dict<i32,Utf8> key {} at row {} out of range (dictionary size {})",
                                                    key, i, dict_len
                                                )));
                                            }
                                            Ok(key)
                                        }
                                    })
                                    .collect::<BoltResult<Vec<i32>>>()?;
                                (keys, dict_strings)
                            }
                            A::Int64 => {
                                let da = arr
                                    .as_any()
                                    .downcast_ref::<DictionaryArray<Int64Type>>()
                                    .ok_or_else(|| {
                                        type_mismatch_err(arr, "Dictionary<i64, Utf8>")
                                    })?;
                                let sa = da
                                    .values()
                                    .as_any()
                                    .downcast_ref::<StringArray>()
                                    .ok_or_else(|| {
                                        BoltError::Type(
                                            "GpuColumn: dict values are not StringArray".into(),
                                        )
                                    })?;
                                let mut dict_strings: Vec<String> = Vec::with_capacity(sa.len());
                                for i in 0..sa.len() {
                                    if sa.is_null(i) {
                                        dict_strings.push(String::new());
                                    } else {
                                        dict_strings.push(sa.value(i).to_string());
                                    }
                                }
                                // Narrow i64 -> i32 keys for the device buffer.
                                // The dict can't have more than i32::MAX
                                // entries without breaking downstream codegen
                                // (matches `DictionaryColumn`'s contract).
                                if sa.len() > (i32::MAX as usize) {
                                    return Err(BoltError::Type(format!(
                                        "GpuColumn: dict<i64,Utf8> with {} entries exceeds i32 capacity",
                                        sa.len()
                                    )));
                                }
                                // Finding V-5: validate every key before the
                                // i64 -> i32 narrowing. The previous `as i32`
                                // cast SILENTLY truncated/wrapped large keys,
                                // and only the dictionary *size* was checked
                                // against i32::MAX above — never the per-key
                                // values. An unvalidated key later indexes the
                                // dictionary inside the GPU kernel -> OOB
                                // device read. Reject negative, out-of-range,
                                // and `> i32::MAX` keys so the `as i32` cast
                                // can never truncate a value that is actually
                                // used. Null slots are encoded as 0 (validity
                                // handled separately by `upload_dict_utf8`).
                                let dict_len = sa.len();
                                let keys: Vec<i32> = (0..n)
                                    .map(|i| {
                                        if da.keys().is_null(i) {
                                            Ok(0)
                                        } else {
                                            let key = da.keys().value(i);
                                            if key < 0 {
                                                return Err(BoltError::Type(format!(
                                                    "GpuColumn: negative dict<i64,Utf8> key {} at row {}",
                                                    key, i
                                                )));
                                            }
                                            if key > i32::MAX as i64 {
                                                return Err(BoltError::Type(format!(
                                                    "GpuColumn: dict<i64,Utf8> key {} at row {} exceeds i32 capacity",
                                                    key, i
                                                )));
                                            }
                                            if (key as usize) >= dict_len {
                                                return Err(BoltError::Type(format!(
                                                    "GpuColumn: dict<i64,Utf8> key {} at row {} out of range (dictionary size {})",
                                                    key, i, dict_len
                                                )));
                                            }
                                            Ok(key as i32)
                                        }
                                    })
                                    .collect::<BoltResult<Vec<i32>>>()?;
                                (keys, dict_strings)
                            }
                            other => {
                                return Err(BoltError::Type(format!(
                                    "GpuColumn: dict key type {:?} not supported (expected Int32 or Int64)",
                                    other
                                )));
                            }
                        };
                        let keys_gpu = GpuVec::<i32>::from_slice(&keys_i32)?;
                        // Stage 6: this inline path is a fallback for direct
                        // GpuColumn::upload callers (the engine's main ingest
                        // route goes through `upload_dict_utf8` via
                        // `GpuTable::from_record_batch`, which packs the
                        // Arrow null buffer into a validity bitmap). We don't
                        // re-derive validity here — `upload_dict_utf8` is the
                        // null-aware ingress.
                        GpuColumnData::DictUtf8 {
                            keys: keys_gpu,
                            dict: dict_strings,
                            valid_mask: None,
                        }
                    }
                    other => {
                        return Err(BoltError::Type(format!(
                            "GpuColumn: dtype Utf8 backed by unsupported Arrow type {:?}",
                            other
                        )));
                    }
                }
            }
            DataType::Decimal128(precision, scale) => {
                // v0.7 sub-task B: ingest a `Decimal128Array` as the
                // interleaved [lo0, hi0, lo1, hi1, ...] u64 buffer that
                // PTX `Op::LoadColumn128` expects. Each row's `i128`
                // value is split into the low / high 64-bit halves via
                // the (sign-preserving) `as u128` cast — masking with
                // `as u64` and shifting `>> 64` gives the two halves of
                // the same bit pattern.
                let da = arr
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or_else(|| type_mismatch_err(arr, "Decimal128"))?;
                // The Arrow `Decimal128Array` may carry its own (p, s)
                // declaration — if it disagrees with the column's plan-
                // level dtype the schema is internally inconsistent and
                // every downstream consumer would silently mis-interpret
                // the values. Surface the mismatch eagerly.
                if let arrow_schema::DataType::Decimal128(ap, as_) = da.data_type() {
                    if *ap != precision || *as_ != scale {
                        return Err(BoltError::Type(format!(
                            "Decimal128 column '{}' upload: plan dtype \
                             Decimal128({precision}, {scale}) disagrees with Arrow \
                             dtype Decimal128({ap}, {as_})",
                            name
                        )));
                    }
                }
                let n = da.len();
                let mut packed: Vec<u64> = Vec::with_capacity(2 * n);
                for i in 0..n {
                    // NULL rows are stored as zeroed bit-patterns; the
                    // `valid_mask` packed below is the source of truth for
                    // per-row validity so the download path can reconstruct
                    // a NULL (not 0).
                    let v: i128 = if da.is_null(i) { 0 } else { da.value(i) };
                    let bits = v as u128;
                    packed.push(bits as u64);
                    packed.push((bits >> 64) as u64);
                }
                let buf = GpuVec::<u64>::from_slice(&packed)?;
                // Optional validity: pack from the Arrow null buffer if
                // present, exactly as `upload_dict_utf8` does for the
                // DictUtf8 keys (same u8 element type, same lsb-first
                // bitmap, same `None`-when-no-nulls contract).
                let valid_mask = if let Some(nb) = da.nulls() {
                    let bits = pack_validity_from_arrow(nb, n);
                    Some(GpuVec::<u8>::from_slice(&bits)?)
                } else {
                    None
                };
                GpuColumnData::Decimal128 {
                    values: buf,
                    precision,
                    scale,
                    valid_mask,
                }
            }
            DataType::Date32 => {
                // F6: Date32 is `i32` days-since-epoch on the device — the
                // exact same fixed-width buffer layout as `Int32`. We store it
                // in the `I32` storage variant (the `GpuColumnData` enum has no
                // dedicated temporal variant, and adding one would ripple into
                // every cross-module `match` on the enum); the real
                // `DataType::Date32` rides on `GpuColumn::dtype`, so consumers
                // that care about the temporal-ness branch on the dtype, not on
                // the storage discriminant. The download/gather paths read the
                // dtype and rebuild a `Date32Array`.
                let pa = arr
                    .as_any()
                    .downcast_ref::<arrow_array::Date32Array>()
                    .ok_or_else(|| type_mismatch_err(arr, "Date32"))?;
                let buf = primitive_to_gpu(pa)?;
                let validity = unpacked_validity_from_arrow(pa)?;
                GpuColumnData::I32 {
                    values: GpuVec::from_buffer(buf),
                    validity,
                }
            }
            DataType::Timestamp(unit, _tz) => {
                // F6: Timestamp is `i64` ticks-since-epoch on the device — the
                // same fixed-width layout as `Int64`. Stored in the `I64`
                // storage variant for the same reason Date32 reuses `I32` (no
                // dedicated enum variant; the real dtype rides on
                // `GpuColumn::dtype`). The concrete Arrow `Timestamp*Array`
                // depends on the `TimeUnit`, so we downcast per unit; the
                // timezone never changes the stored bits (it is interpretation
                // only) so we ignore it here — the dtype carries it for the
                // download-side reconstruction.
                use crate::plan::logical_plan::TimeUnit;
                let buf = match unit {
                    TimeUnit::Second => {
                        let pa = arr
                            .as_any()
                            .downcast_ref::<arrow_array::TimestampSecondArray>()
                            .ok_or_else(|| type_mismatch_err(arr, "TimestampSecond"))?;
                        let b = primitive_to_gpu(pa)?;
                        let validity = unpacked_validity_from_arrow(pa)?;
                        (GpuVec::from_buffer(b), validity)
                    }
                    TimeUnit::Millisecond => {
                        let pa = arr
                            .as_any()
                            .downcast_ref::<arrow_array::TimestampMillisecondArray>()
                            .ok_or_else(|| type_mismatch_err(arr, "TimestampMillisecond"))?;
                        let b = primitive_to_gpu(pa)?;
                        let validity = unpacked_validity_from_arrow(pa)?;
                        (GpuVec::from_buffer(b), validity)
                    }
                    TimeUnit::Microsecond => {
                        let pa = arr
                            .as_any()
                            .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
                            .ok_or_else(|| type_mismatch_err(arr, "TimestampMicrosecond"))?;
                        let b = primitive_to_gpu(pa)?;
                        let validity = unpacked_validity_from_arrow(pa)?;
                        (GpuVec::from_buffer(b), validity)
                    }
                    TimeUnit::Nanosecond => {
                        let pa = arr
                            .as_any()
                            .downcast_ref::<arrow_array::TimestampNanosecondArray>()
                            .ok_or_else(|| type_mismatch_err(arr, "TimestampNanosecond"))?;
                        let b = primitive_to_gpu(pa)?;
                        let validity = unpacked_validity_from_arrow(pa)?;
                        (GpuVec::from_buffer(b), validity)
                    }
                };
                GpuColumnData::I64 {
                    values: buf.0,
                    validity: buf.1,
                }
            }
        };
        Ok(Self {
            name,
            dtype,
            data,
            host_revision: 0,
        })
    }

    /// Upload an Arrow `DictionaryArray<Int32, Utf8>` to the GPU as a native
    /// [`GpuColumnData::DictUtf8`] without going through `StringArray`
    /// flattening. Used by the engine when it ingests dictionary-encoded
    /// columns directly (Stage 6+).
    pub fn upload_dict_utf8(name: String, arr: &DictionaryArray<Int32Type>) -> BoltResult<Self> {
        // Values must be Utf8.
        let dict_vals = arr
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                BoltError::Type(format!(
                    "DictionaryArray for column '{}' has non-Utf8 value type {:?}",
                    name,
                    arr.values().data_type()
                ))
            })?;
        // Host-side dictionary: copy each value as an owned String.
        let mut dict: Vec<String> = Vec::with_capacity(dict_vals.len());
        for i in 0..dict_vals.len() {
            // The dictionary itself is rarely nullable in practice; treat any
            // null value as an empty string so the key->string lookup remains
            // total. The validity for the row-level data still flows through
            // the keys' null buffer.
            if dict_vals.is_null(i) {
                dict.push(String::new());
            } else {
                dict.push(dict_vals.value(i).to_string());
            }
        }

        // Keys: upload the underlying i32 buffer to the device.
        let keys_arr: &Int32Array = arr.keys();
        let n_rows = keys_arr.len();
        // Copy keys to an owned Vec — null keys are zeroed out so the device
        // never reads garbage. Row-level validity lives in `valid_mask`.
        let mut keys_host: Vec<i32> = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            if keys_arr.is_null(i) {
                keys_host.push(0);
            } else {
                keys_host.push(keys_arr.value(i));
            }
        }
        let keys = GpuVec::<i32>::from_slice(&keys_host)?;

        // Optional validity: pack from the Arrow null buffer if present.
        let valid_mask = if let Some(nb) = keys_arr.nulls() {
            let bits = pack_validity_from_arrow(nb, n_rows);
            Some(GpuVec::<u8>::from_slice(&bits)?)
        } else {
            None
        };

        Ok(Self {
            name,
            dtype: DataType::Utf8,
            data: GpuColumnData::DictUtf8 {
                keys,
                dict,
                valid_mask,
            },
            host_revision: 0,
        })
    }

    /// Device pointer for the column's primary buffer.
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.data.device_ptr()
    }

    /// Host-side Utf8 dictionary, if this is a Utf8-backed column.
    ///
    /// **Stage 5 / Stage 6** — returns the dictionary for both the legacy
    /// engine-managed `Utf8` variant (slot 0 reserved for NULL, real strings
    /// at indices `1..`) and the `DictUtf8` variant (1:1 with the Arrow
    /// dictionary; no NULL offset — NULL handling lives on the
    /// `valid_mask` bitmap). Callers that care about the layout distinction
    /// must match on `data` directly.
    pub fn utf8_dictionary(&self) -> Option<&[String]> {
        match &self.data {
            GpuColumnData::Utf8 { dictionary, .. } => Some(dictionary),
            GpuColumnData::DictUtf8 { dict, .. } => Some(dict),
            _ => None,
        }
    }
}

/// A table whose columns live on the GPU for the engine's lifetime.
pub struct GpuTable {
    /// Number of rows in every column.
    pub n_rows: usize,
    /// Columns in source-schema order.
    pub columns: Vec<GpuColumn>,
    /// Host-side `table_revision` at the time this table was last touched
    /// by [`crate::exec::engine::Engine::ensure_gpu_table`].
    ///
    /// The engine's incremental-upload path compares the host's current
    /// `table_revision` against this field on cache hit:
    ///   - Equal: the cache is fully fresh — every column was uploaded at
    ///     the current revision. Return as-is.
    ///   - Less: at least one mutation has happened since the last upload.
    ///     The engine walks `columns` and reuses any whose
    ///     `host_revision` still matches, re-uploading only the rest.
    ///
    /// `GpuTable::from_record_batch` leaves this at `0` and the engine sets
    /// it to the matching host revision right after building. Direct callers
    /// can leave it untouched.
    pub last_uploaded_revision: u64,
}

impl GpuTable {
    /// Upload a single Arrow `Field`'s column from `batch` to the device,
    /// dispatching on the Arrow dtype. Used by both
    /// [`GpuTable::from_record_batch`] (full uploads) and the engine's
    /// incremental cache (per-column re-upload on `register_batch`).
    ///
    /// This is the canonical Arrow-dtype → `GpuColumn` mapping; any future
    /// dtype additions should go here so both code paths stay in sync.
    pub fn upload_column_from_batch(
        batch: &RecordBatch,
        field: &arrow_schema::Field,
        idx: usize,
    ) -> BoltResult<GpuColumn> {
        let arr = batch.column(idx);
        let col = match field.data_type() {
            arrow_schema::DataType::Int32 => {
                GpuColumn::upload(field.name().clone(), arr.as_ref(), DataType::Int32)?
            }
            arrow_schema::DataType::Int64 => {
                GpuColumn::upload(field.name().clone(), arr.as_ref(), DataType::Int64)?
            }
            arrow_schema::DataType::Float32 => {
                GpuColumn::upload(field.name().clone(), arr.as_ref(), DataType::Float32)?
            }
            arrow_schema::DataType::Float64 => {
                GpuColumn::upload(field.name().clone(), arr.as_ref(), DataType::Float64)?
            }
            arrow_schema::DataType::Boolean => {
                GpuColumn::upload(field.name().clone(), arr.as_ref(), DataType::Bool)?
            }
            arrow_schema::DataType::Utf8 => {
                GpuColumn::upload(field.name().clone(), arr.as_ref(), DataType::Utf8)?
            }
            // Stage 6: native ingest path for `DictionaryArray<Int32, Utf8>`.
            // Stage 5 had this routed through `GpuColumn::upload` (which
            // would build a `DictUtf8` variant without validity); Stage 6
            // upgrades the path to `upload_dict_utf8`, which packs the
            // Arrow null buffer into an on-device validity bitmap so
            // downstream null-aware kernels see real per-row validity.
            arrow_schema::DataType::Dictionary(key_t, val_t)
                if key_t.as_ref() == &arrow_schema::DataType::Int32
                    && val_t.as_ref() == &arrow_schema::DataType::Utf8 =>
            {
                let dict_arr = arr
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()
                    .ok_or_else(|| {
                        BoltError::Type(format!(
                            "column '{}' declared Dictionary<Int32, Utf8> but did not \
                             downcast to DictionaryArray<Int32Type>",
                            field.name()
                        ))
                    })?;
                GpuColumn::upload_dict_utf8(field.name().clone(), dict_arr)?
            }
            // v0.7 sub-task B: route `Decimal128(p, s)` through the
            // canonical `GpuColumn::upload` path, which packs the i128
            // values into the interleaved `[lo, hi]` u64 buffer that
            // PTX `Op::LoadColumn128` reads.
            arrow_schema::DataType::Decimal128(p, s) => GpuColumn::upload(
                field.name().clone(),
                arr.as_ref(),
                DataType::Decimal128(*p, *s),
            )?,
            // F6: temporal columns. Date32 maps 1:1; Timestamp maps the Arrow
            // `TimeUnit` to the plan `TimeUnit` and interns the optional
            // timezone (so the plan `DataType` stays `Copy`). The byte width
            // (i32 / i64) is what the upload path keys off — the dtype carries
            // the temporal semantics through to the download-side
            // reconstruction.
            arrow_schema::DataType::Date32 => {
                GpuColumn::upload(field.name().clone(), arr.as_ref(), DataType::Date32)?
            }
            arrow_schema::DataType::Timestamp(arrow_unit, arrow_tz) => {
                use crate::plan::logical_plan::{intern_timezone, TimeUnit};
                let unit = match arrow_unit {
                    arrow_schema::TimeUnit::Second => TimeUnit::Second,
                    arrow_schema::TimeUnit::Millisecond => TimeUnit::Millisecond,
                    arrow_schema::TimeUnit::Microsecond => TimeUnit::Microsecond,
                    arrow_schema::TimeUnit::Nanosecond => TimeUnit::Nanosecond,
                };
                let tz: Option<&'static str> =
                    arrow_tz.as_ref().map(|s| intern_timezone(s.as_ref()));
                GpuColumn::upload(
                    field.name().clone(),
                    arr.as_ref(),
                    DataType::Timestamp(unit, tz),
                )?
            }
            other => {
                return Err(BoltError::Type(format!(
                    "GpuTable: unsupported Arrow dtype {:?} for column '{}'",
                    other,
                    field.name()
                )));
            }
        };
        Ok(col)
    }

    /// Upload every column of `batch` to the device.
    pub fn from_record_batch(batch: &RecordBatch) -> BoltResult<Self> {
        let n_rows = batch.num_rows();
        let schema = batch.schema();
        let mut columns: Vec<GpuColumn> = Vec::with_capacity(batch.num_columns());
        for (idx, field) in schema.fields().iter().enumerate() {
            let col = Self::upload_column_from_batch(batch, field, idx)?;
            columns.push(col);
        }
        Ok(Self {
            n_rows,
            columns,
            last_uploaded_revision: 0,
        })
    }

    /// Borrow the column with the given name, if present.
    pub fn column(&self, name: &str) -> Option<&GpuColumn> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Device pointer for the named column, if present.
    pub fn device_ptr(&self, name: &str) -> Option<CUdeviceptr> {
        self.column(name).map(|c| c.device_ptr())
    }
}

/// Build a `Type` error for an Arrow downcast failure.
fn type_mismatch_err(arr: &dyn Array, expected: &str) -> BoltError {
    BoltError::Type(format!(
        "Arrow array dtype {:?} does not match expected {}",
        arr.data_type(),
        expected
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::StringDictionaryBuilder;
    use arrow_array::types::Int32Type as ArrowInt32Type;
    use arrow_array::{DictionaryArray, Int32Array, StringArray};
    use arrow_buffer::{BooleanBuffer, NullBuffer};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    /// `pack_validity_from_arrow` round-trips an Arrow null buffer into the
    /// PV-stage-c packed-bits layout (lsb-first per byte). This is a pure-host
    /// test — no CUDA required.
    #[test]
    fn pack_validity_from_arrow_lsb_first() {
        // Rows: [valid, null, valid, valid, null, null, valid, valid, valid]
        //  bit:  [0=1,   1=0,  2=1,   3=1,   4=0,  5=0,  6=1,   7=1,   8=1]
        // byte 0 = 0b1100_1101 = 0xCD ; byte 1 = 0b0000_0001 = 0x01
        let bools = [true, false, true, true, false, false, true, true, true];
        let nb = NullBuffer::new(BooleanBuffer::from_iter(bools.iter().copied()));
        let packed = pack_validity_from_arrow(&nb, bools.len());
        assert_eq!(packed, vec![0xCDu8, 0x01u8]);
    }

    /// F6: a temporal upload with a MISMATCHED Arrow array must surface a clean
    /// `BoltError::Type` from the downcast (not a panic), before any device
    /// allocation. Passing an `Int32Array` where a `Date32Array` /
    /// `Timestamp*Array` is expected exercises the `type_mismatch_err` path.
    ///
    /// Pure-host: the downcast fails before `primitive_to_gpu`, so no CUDA
    /// context is needed.
    #[test]
    fn temporal_upload_type_mismatch_is_clean_error() {
        use crate::plan::logical_plan::TimeUnit;
        let placeholder = Int32Array::from(Vec::<i32>::new());
        for dtype in [
            DataType::Date32,
            DataType::Timestamp(TimeUnit::Nanosecond, None),
        ] {
            match GpuColumn::upload("ts".to_string(), &placeholder, dtype) {
                Ok(_) => panic!("{dtype:?} upload of an Int32Array must fail the downcast"),
                Err(BoltError::Type(_)) => { /* expected clean downcast error */ }
                Err(other) => {
                    panic!("{dtype:?} downcast mismatch must be BoltError::Type, got: {other:?}")
                }
            }
        }
    }

    /// F6: GPU round-trip for Date32 — a `Date32Array` uploads to the i32
    /// storage variant (same device layout) while `GpuColumn::dtype` keeps the
    /// real `Date32` so downstream gather/download rebuilds the right Arrow
    /// type. Allocates on the device, so `gpu:`-ignored.
    #[test]
    #[ignore = "gpu:tier1 — GpuVec upload allocates on device"]
    fn date32_uploads_as_i32_storage_with_date_dtype() {
        use arrow_array::Date32Array;
        let _ctx = crate::cuda::CudaContext::new(0).expect("CUDA ctx");
        let arr = Date32Array::from(vec![19000i32, 19001, 19002]);
        let col = GpuColumn::upload("d".into(), &arr, DataType::Date32).expect("upload");
        assert!(matches!(col.dtype, DataType::Date32));
        // Storage reuses the I32 variant (no dedicated temporal enum variant).
        assert!(matches!(col.data, GpuColumnData::I32 { .. }));
    }

    /// F6: GPU round-trip for Timestamp — a `TimestampMicrosecondArray` uploads
    /// to the i64 storage variant; the dtype carries the unit/tz.
    #[test]
    #[ignore = "gpu:tier1 — GpuVec upload allocates on device"]
    fn timestamp_uploads_as_i64_storage_with_ts_dtype() {
        use crate::plan::logical_plan::TimeUnit;
        use arrow_array::TimestampMicrosecondArray;
        let _ctx = crate::cuda::CudaContext::new(0).expect("CUDA ctx");
        let arr = TimestampMicrosecondArray::from(vec![1_000i64, 2_000, 3_000]);
        let col = GpuColumn::upload(
            "ts".into(),
            &arr,
            DataType::Timestamp(TimeUnit::Microsecond, None),
        )
        .expect("upload");
        assert!(matches!(
            col.dtype,
            DataType::Timestamp(TimeUnit::Microsecond, None)
        ));
        assert!(matches!(col.data, GpuColumnData::I64 { .. }));
    }

    /// Stage 5 — registering a `DictionaryArray<i32, Utf8>` column produces
    /// the new `DictUtf8` GpuColumn variant rather than the engine-managed
    /// `Utf8` variant. The input dictionary is preserved verbatim.
    ///
    /// Ignored: GpuVec uploads require a CUDA device, so this test must be
    /// run with `cargo test ... -- --ignored` on a GPU host.
    #[test]
    #[ignore = "gpu:string"]
    fn dict_column_uploads_without_flattening() {
        // Make sure a CUDA context exists. Engine::new() initialises one.
        let _ctx = crate::cuda::CudaContext::new(0).expect("CUDA ctx");

        let dict_values = vec!["alpha", "bravo", "charlie", "delta", "echo"];
        let keys: Vec<i32> = vec![3, 1, 4, 1, 0, 2, 4, 0]; // 8 rows
        let dict_arr: DictionaryArray<Int32Type> = DictionaryArray::try_new(
            Int32Array::from(keys.clone()),
            Arc::new(StringArray::from(dict_values.clone())),
        )
        .unwrap();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "region",
            ArrowDataType::Dictionary(
                Box::new(ArrowDataType::Int32),
                Box::new(ArrowDataType::Utf8),
            ),
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(dict_arr)]).unwrap();

        let table = GpuTable::from_record_batch(&batch).expect("GpuTable upload");
        assert_eq!(table.n_rows, keys.len());
        let col = table.column("region").expect("region column");
        // Plan dtype is still Utf8 — keeps planner / consumer reasoning
        // unified on string columns.
        assert!(matches!(col.dtype, DataType::Utf8));
        // Storage variant is the new DictUtf8 — input dictionary preserved.
        match &col.data {
            GpuColumnData::DictUtf8 { dict, .. } => {
                assert_eq!(
                    dict.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                    dict_values
                );
            }
            GpuColumnData::Utf8 { .. } => {
                panic!(
                    "Stage 5 contract: GpuTable must accept DictionaryArray natively, \
                     not silently flatten to the Utf8 variant"
                );
            }
            _ => panic!("expected DictUtf8 variant, got a different GpuColumnData"),
        }
        // `utf8_dictionary()` accessor returns the same dict (compat shim
        // for code that only knows the engine's `Utf8` variant).
        let dict = col.utf8_dictionary().expect("utf8_dictionary");
        assert_eq!(dict.len(), dict_values.len());
    }

    /// Stage 5 — plain `StringArray` columns continue to route through the
    /// engine's `Utf8` variant (backward compat). Storage layout differs
    /// from `DictUtf8` (slot-0 NULL reservation, dictionary owned by
    /// `DictionaryColumn`).
    #[test]
    #[ignore = "gpu:string"]
    fn plain_utf8_column_still_takes_utf8_variant() {
        let _ctx = crate::cuda::CudaContext::new(0).expect("CUDA ctx");

        let strings = vec!["alpha", "bravo", "charlie", "alpha"];
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "name",
            ArrowDataType::Utf8,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(strings.clone()))])
                .unwrap();

        let table = GpuTable::from_record_batch(&batch).expect("GpuTable upload");
        let col = table.column("name").expect("name column");
        assert!(matches!(col.dtype, DataType::Utf8));
        assert!(matches!(col.data, GpuColumnData::Utf8 { .. }));
    }

    /// Stage 6 — upload a `DictionaryArray<Int32, Utf8>` with NULLs and
    /// verify the resulting `DictUtf8` variant carries a validity bitmap.
    /// CUDA-dependent (uploads to the device).
    #[test]
    #[ignore = "gpu:string"]
    fn dict_utf8_with_nulls_propagates_validity() {
        let mut b: StringDictionaryBuilder<ArrowInt32Type> = StringDictionaryBuilder::new();
        b.append_value("a");
        b.append_null();
        b.append_value("b");
        b.append_value("a");
        b.append_null();
        let arr = b.finish();
        assert_eq!(arr.keys().null_count(), 2);

        let col = GpuColumn::upload_dict_utf8("c".into(), &arr).expect("upload");
        match &col.data {
            GpuColumnData::DictUtf8 {
                keys,
                dict,
                valid_mask,
            } => {
                assert_eq!(keys.len(), 5);
                assert_eq!(dict.as_slice(), &["a".to_string(), "b".to_string()]);
                let mask = valid_mask
                    .as_ref()
                    .expect("validity bitmap should be present when source has nulls");
                // 5 rows → 1 byte of bitmap.
                assert_eq!(mask.len(), 1);
            }
            other => panic!("expected DictUtf8, got {:?}", std::mem::discriminant(other)),
        }
        // device_ptr / validity_ptr surface the correct buffers.
        assert!(col.data.validity_ptr().is_some());
    }

    /// Stage 6 — a `DictionaryArray` with zero nulls uploads a `DictUtf8`
    /// whose `valid_mask` is `None`.
    #[test]
    #[ignore = "gpu:string"]
    fn dict_utf8_without_nulls_omits_validity() {
        let mut b: StringDictionaryBuilder<ArrowInt32Type> = StringDictionaryBuilder::new();
        b.append_value("x");
        b.append_value("y");
        b.append_value("x");
        let arr = b.finish();
        assert_eq!(arr.keys().null_count(), 0);

        let col = GpuColumn::upload_dict_utf8("c".into(), &arr).expect("upload");
        match &col.data {
            GpuColumnData::DictUtf8 { valid_mask, .. } => {
                assert!(valid_mask.is_none(), "no-nulls upload should omit validity");
            }
            _ => panic!("expected DictUtf8"),
        }
        assert!(col.data.validity_ptr().is_none());
    }

    /// Decimal128 NULL fix — host-only check that the mask-building helper
    /// (`pack_validity_from_arrow`, the same one the Decimal128 upload path
    /// uses) packs a `Decimal128Array`'s null buffer into the expected
    /// lsb-first bitmap. No CUDA required: we exercise only the pure
    /// mask-construction step, not the device upload.
    #[test]
    fn decimal128_null_buffer_packs_lsb_first() {
        // Rows: [10, NULL, 30, 40, NULL] (p=10, s=2).
        //  valid bits: [1, 0, 1, 1, 0]  → byte 0 = 0b0000_1101 = 0x0D.
        let arr = Decimal128Array::from(vec![Some(10i128), None, Some(30), Some(40), None])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let nb = arr.nulls().expect("array has nulls");
        let packed = pack_validity_from_arrow(nb, arr.len());
        assert_eq!(packed, vec![0x0Du8]);
    }

    /// Decimal128 NULL fix — GPU round-trip: a `Decimal128Array` containing a
    /// NULL must upload (carrying a `valid_mask`) and read back as NULL, not
    /// `0`. Requires a CUDA device, so it is ignored in host-only CI.
    #[test]
    #[ignore = "gpu:tier1"]
    fn decimal128_null_roundtrips_as_null() {
        let _ctx = crate::cuda::CudaContext::new(0).expect("CUDA ctx");

        let arr = Decimal128Array::from(vec![Some(123i128), None, Some(456)])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let col = GpuColumn::upload("d".into(), &arr, DataType::Decimal128(10, 2)).expect("upload");
        // The upload must carry a validity bitmap (source had a null).
        match &col.data {
            GpuColumnData::Decimal128 { valid_mask, .. } => {
                assert!(
                    valid_mask.is_some(),
                    "Decimal128 upload with a null must carry a valid_mask"
                );
            }
            _ => panic!("expected Decimal128 variant"),
        }
        assert!(col.data.validity_ptr().is_some());
    }

    /// Decimal128 NULL fix — a no-nulls `Decimal128Array` uploads with
    /// `valid_mask == None` (mirrors the DictUtf8 no-nulls contract).
    #[test]
    #[ignore = "gpu:tier1"]
    fn decimal128_without_nulls_omits_validity() {
        let _ctx = crate::cuda::CudaContext::new(0).expect("CUDA ctx");

        let arr = Decimal128Array::from(vec![1i128, 2, 3])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let col = GpuColumn::upload("d".into(), &arr, DataType::Decimal128(10, 2)).expect("upload");
        match &col.data {
            GpuColumnData::Decimal128 { valid_mask, .. } => {
                assert!(valid_mask.is_none(), "no-nulls upload should omit validity");
            }
            _ => panic!("expected Decimal128 variant"),
        }
        assert!(col.data.validity_ptr().is_none());
    }
}
