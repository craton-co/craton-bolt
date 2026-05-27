// SPDX-License-Identifier: Apache-2.0

//! GPU-resident table storage: columns uploaded once and queried in place.

use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray,
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
        };
        Ok(Self { name, dtype, data })
    }

    /// Device pointer for the column's primary buffer.
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.data.device_ptr()
    }

    /// Host-side Utf8 dictionary, if this is a Utf8 column.
    pub fn utf8_dictionary(&self) -> Option<&[String]> {
        match &self.data {
            GpuColumnData::Utf8 { dictionary, .. } => Some(dictionary),
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
