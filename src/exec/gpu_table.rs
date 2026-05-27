// SPDX-License-Identifier: Apache-2.0

//! GPU-resident table storage: columns uploaded once and queried in place.

use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{
    Array, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray,
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
}

/// Heterogeneous owned device storage for a single column.
pub enum GpuColumnData {
    /// 32-bit signed integer column.
    I32(GpuVec<i32>),
    /// 64-bit signed integer column.
    I64(GpuVec<i64>),
    /// 32-bit float column.
    F32(GpuVec<f32>),
    /// 64-bit float column.
    F64(GpuVec<f64>),
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
    DictUtf8 {
        /// Per-row dictionary keys (the i32 indices into `dict`).
        keys: GpuVec<i32>,
        /// Host-side dictionary, `dict[i]` decodes key `i`. NULL rows are
        /// represented by the Arrow keys array's null bit, not by a slot-0
        /// convention — `keys.value(i)` is meaningless if the original
        /// `keys.is_null(i)`. (Validity is currently not uploaded; callers
        /// that need null handling must build a separate bitmap. Stage 6
        /// follow-up.)
        dict: Vec<String>,
    },
}

impl GpuColumnData {
    /// Raw device pointer to the column's primary buffer.
    pub fn device_ptr(&self) -> CUdeviceptr {
        match self {
            GpuColumnData::I32(v) => v.device_ptr(),
            GpuColumnData::I64(v) => v.device_ptr(),
            GpuColumnData::F32(v) => v.device_ptr(),
            GpuColumnData::F64(v) => v.device_ptr(),
            GpuColumnData::Bool(v) => v.device_ptr(),
            GpuColumnData::BoolNullable { values, .. } => values.device_ptr(),
            GpuColumnData::Utf8 { indices, .. } => indices.device_ptr(),
            // Stage 5 — DictUtf8's primary buffer is its keys array.
            GpuColumnData::DictUtf8 { keys, .. } => keys.device_ptr(),
        }
    }
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
                GpuColumnData::I32(GpuVec::from_buffer(buf))
            }
            DataType::Int64 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| type_mismatch_err(arr, "Int64"))?;
                let buf = primitive_to_gpu(pa)?;
                GpuColumnData::I64(GpuVec::from_buffer(buf))
            }
            DataType::Float32 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| type_mismatch_err(arr, "Float32"))?;
                let buf = primitive_to_gpu(pa)?;
                GpuColumnData::F32(GpuVec::from_buffer(buf))
            }
            DataType::Float64 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| type_mismatch_err(arr, "Float64"))?;
                let buf = primitive_to_gpu(pa)?;
                GpuColumnData::F64(GpuVec::from_buffer(buf))
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
                    A::Dictionary(key_ty, value_ty)
                        if matches!(value_ty.as_ref(), A::Utf8) =>
                    {
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
                                let mut dict_strings: Vec<String> =
                                    Vec::with_capacity(sa.len());
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
                                let keys: Vec<i32> = (0..n)
                                    .map(|i| {
                                        if da.keys().is_null(i) {
                                            0
                                        } else {
                                            da.keys().value(i)
                                        }
                                    })
                                    .collect();
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
                                let mut dict_strings: Vec<String> =
                                    Vec::with_capacity(sa.len());
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
                                let keys: Vec<i32> = (0..n)
                                    .map(|i| {
                                        if da.keys().is_null(i) {
                                            0
                                        } else {
                                            da.keys().value(i) as i32
                                        }
                                    })
                                    .collect();
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
                        GpuColumnData::DictUtf8 {
                            keys: keys_gpu,
                            dict: dict_strings,
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
        };
        Ok(Self { name, dtype, data })
    }

    /// Device pointer for the column's primary buffer.
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.data.device_ptr()
    }

    /// Host-side Utf8 dictionary, if this is a Utf8 column.
    ///
    /// **Stage 5** — returns the dict for both the engine-managed `Utf8`
    /// variant (with the slot-0-is-NULL convention) and the new `DictUtf8`
    /// variant (input dict, no NULL reservation). Callers that need to know
    /// the indexing convention should match on `data` directly.
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
}

impl GpuTable {
    /// Upload every column of `batch` to the device.
    pub fn from_record_batch(batch: &RecordBatch) -> BoltResult<Self> {
        let n_rows = batch.num_rows();
        let schema = batch.schema();
        let mut columns: Vec<GpuColumn> = Vec::with_capacity(batch.num_columns());
        for (idx, field) in schema.fields().iter().enumerate() {
            let arr = batch.column(idx);
            let dtype = match field.data_type() {
                arrow_schema::DataType::Int32 => DataType::Int32,
                arrow_schema::DataType::Int64 => DataType::Int64,
                arrow_schema::DataType::Float32 => DataType::Float32,
                arrow_schema::DataType::Float64 => DataType::Float64,
                arrow_schema::DataType::Boolean => DataType::Bool,
                arrow_schema::DataType::Utf8 => DataType::Utf8,
                // Stage 5 — dict-encoded Utf8 maps to the engine's Utf8 dtype
                // so the planner / consumers continue to reason about it as
                // a string column. `GpuColumn::upload` then dispatches on
                // the runtime Arrow type to pick the storage variant
                // (`Utf8` for plain StringArray, `DictUtf8` for the dict
                // input).
                arrow_schema::DataType::Dictionary(_, value_ty)
                    if matches!(value_ty.as_ref(), arrow_schema::DataType::Utf8) =>
                {
                    DataType::Utf8
                }
                other => {
                    return Err(BoltError::Type(format!(
                        "GpuTable: unsupported Arrow dtype {:?} for column '{}'",
                        other,
                        field.name()
                    )));
                }
            };
            let col = GpuColumn::upload(field.name().clone(), arr.as_ref(), dtype)?;
            columns.push(col);
        }
        Ok(Self { n_rows, columns })
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
    use arrow_array::{DictionaryArray, Int32Array, StringArray};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    /// Stage 5 — registering a `DictionaryArray<i32, Utf8>` column produces
    /// the new `DictUtf8` GpuColumn variant rather than the engine-managed
    /// `Utf8` variant. The input dictionary is preserved verbatim.
    ///
    /// Ignored: GpuVec uploads require a CUDA device, so this test must be
    /// run with `cargo test ... -- --ignored` on a GPU host.
    #[test]
    #[ignore = "requires CUDA device"]
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
    #[ignore = "requires CUDA device"]
    fn plain_utf8_column_still_takes_utf8_variant() {
        let _ctx = crate::cuda::CudaContext::new(0).expect("CUDA ctx");

        let strings = vec!["alpha", "bravo", "charlie", "alpha"];
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "name",
            ArrowDataType::Utf8,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(strings.clone()))],
        )
        .unwrap();

        let table = GpuTable::from_record_batch(&batch).expect("GpuTable upload");
        let col = table.column("name").expect("name column");
        assert!(matches!(col.dtype, DataType::Utf8));
        assert!(matches!(col.data, GpuColumnData::Utf8 { .. }));
    }
}
