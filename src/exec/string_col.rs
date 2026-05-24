// SPDX-License-Identifier: Apache-2.0

//! Device columns for non-primitive dtypes: `Bool` and `Utf8`.
//!
//! `engine.rs::DeviceCol` covers the fixed-width numeric types. This module
//! provides parallel upload/alloc/download paths for `Bool` (one byte per row
//! on the GPU) and `Utf8` (dictionary-encoded i32 indices). The orchestrator
//! merges `ExtendedDeviceCol` into `DeviceCol` once both halves are wired.
//!
//! Layout decisions:
//!
//! * **Bool** — Arrow stores booleans as a packed bitmap (1 bit per row); the
//!   GPU codegen path prefers byte-wide loads. We expand to `u8` per row at
//!   upload time (`0`/`1`) and re-pack to a bitmap on download. For inputs
//!   with no nulls we use the simpler `Bool` variant; for inputs with nulls
//!   we use `BoolNullable`, which carries a parallel validity bitmap (one
//!   byte per row) preserving the null/false distinction. The value byte for
//!   a null row is conservatively `0` so existing kernels that read only the
//!   value buffer still produce a defined result.
//!
//! * **Utf8** — see `crate::cuda::dictionary::DictionaryColumn`. Index `0`
//!   means NULL; real strings start at index `1`.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, StringArray};

use crate::cuda::dictionary::DictionaryColumn;
use crate::cuda::GpuVec;
use crate::error::JavelinResult;

/// Extension of `engine.rs`'s `DeviceCol` enum for non-primitive dtypes.
///
/// The orchestrator will merge these variants into the main `DeviceCol`. Until
/// then this enum stands alone so the upload/download halves can be developed
/// and tested in isolation.
pub enum ExtendedDeviceCol {
    /// Boolean column, no nulls. One `u8` per row on the device: `0` for
    /// false, `1` for true. Use this variant when the source array has
    /// `null_count() == 0`; downstream kernels can read the buffer with no
    /// validity check.
    Bool(GpuVec<u8>),
    /// Boolean column with nulls. Two parallel byte-per-row device buffers
    /// of identical length:
    ///
    /// * `values[i] == 1` iff row `i` is `true`. `values[i] == 0` if row `i`
    ///   is `false` OR null — i.e. existing value-only kernels still see a
    ///   defined byte at every offset and never UB-read uninitialised data.
    /// * `validity[i] == 1` iff row `i` is non-null, `0` iff row `i` is null.
    ///   This is the byte-wide expansion of Arrow's packed validity bitmap;
    ///   the kernel-side ABI is `u8` to match `values`.
    ///
    /// Both buffers must have the same length as the logical row count.
    BoolNullable {
        values: GpuVec<u8>,
        validity: GpuVec<u8>,
    },
    /// UTF-8 string column. Stored as i32 dictionary indices on the device;
    /// the host-side dictionary lives inside the `DictionaryColumn`.
    Utf8(DictionaryColumn),
}

impl ExtendedDeviceCol {
    /// Upload a `BooleanArray` as one or two `u8`-per-row device buffers.
    ///
    /// * If `arr.null_count() == 0` returns [`ExtendedDeviceCol::Bool`] with
    ///   a single value buffer.
    /// * Otherwise returns [`ExtendedDeviceCol::BoolNullable`] with a value
    ///   buffer (`1`=true, `0`=false-or-null) AND a parallel validity buffer
    ///   (`1`=non-null, `0`=null). See the variant docs for the contract.
    pub fn upload_bool(arr: &BooleanArray) -> JavelinResult<Self> {
        let n = arr.len();
        if arr.null_count() == 0 {
            let mut bytes: Vec<u8> = Vec::with_capacity(n);
            for i in 0..n {
                bytes.push(if arr.value(i) { 1 } else { 0 });
            }
            let gpu = GpuVec::<u8>::from_slice(&bytes)?;
            return Ok(ExtendedDeviceCol::Bool(gpu));
        }
        let mut values: Vec<u8> = Vec::with_capacity(n);
        let mut validity: Vec<u8> = Vec::with_capacity(n);
        for i in 0..n {
            // `is_null` checks the validity bitmap; `value` is well-defined
            // even for null slots (it reads the underlying bit) but we
            // conservatively force the value byte to `0` for nulls so
            // value-only kernels see a defined byte at every offset.
            if arr.is_null(i) {
                values.push(0);
                validity.push(0);
            } else {
                values.push(if arr.value(i) { 1 } else { 0 });
                validity.push(1);
            }
        }
        let gpu_values = GpuVec::<u8>::from_slice(&values)?;
        let gpu_validity = GpuVec::<u8>::from_slice(&validity)?;
        Ok(ExtendedDeviceCol::BoolNullable {
            values: gpu_values,
            validity: gpu_validity,
        })
    }

    /// Upload a `StringArray` as a dictionary column.
    pub fn upload_utf8(arr: &StringArray) -> JavelinResult<Self> {
        let dict = DictionaryColumn::from_string_array(arr)?;
        Ok(ExtendedDeviceCol::Utf8(dict))
    }

    /// Allocate a zero-initialised Bool column of `len` rows (all bytes 0 →
    /// all rows decode to `false`).
    pub fn alloc_bool(len: usize) -> JavelinResult<Self> {
        let gpu = GpuVec::<u8>::zeros(len)?;
        Ok(ExtendedDeviceCol::Bool(gpu))
    }

    /// Allocate a zero-initialised Utf8 dictionary column.
    ///
    /// `dict` is the host-side dictionary the eventual decode will consult.
    /// For a pure Utf8 projection the caller should pass a clone of the source
    /// `DictionaryColumn::dictionary` — the output indices the kernel writes
    /// are values in the source dictionary's index space.
    ///
    /// If `dict` is `None`, an empty dictionary is installed and the column
    /// can be uploaded/used as a kernel target, but `download()` will only
    /// succeed if every output index is `0` (NULL). Use `None` only when the
    /// column will be overwritten before any decode.
    pub fn alloc_utf8(len: usize, dict: Option<Vec<String>>) -> JavelinResult<Self> {
        let indices = GpuVec::<i32>::zeros(len)?;
        Ok(ExtendedDeviceCol::Utf8(DictionaryColumn {
            dictionary: dict.unwrap_or_default(),
            indices,
            n_rows: len,
        }))
    }

    /// Raw device pointer to the *value* buffer (for `cuLaunchKernel` args).
    ///
    /// For `BoolNullable`, this is the value buffer only — the validity
    /// buffer is reachable via [`Self::validity_device_ptr`]. Kernels that
    /// don't consume validity continue to work unchanged.
    pub fn device_ptr(&self) -> u64 {
        match self {
            ExtendedDeviceCol::Bool(v) => v.device_ptr(),
            ExtendedDeviceCol::BoolNullable { values, .. } => values.device_ptr(),
            ExtendedDeviceCol::Utf8(d) => d.indices.device_ptr(),
        }
    }

    /// Raw device pointer to the validity buffer, if this column carries
    /// one. Only `BoolNullable` has a validity buffer today; all other
    /// variants return `None`.
    pub fn validity_device_ptr(&self) -> Option<u64> {
        match self {
            ExtendedDeviceCol::BoolNullable { validity, .. } => Some(validity.device_ptr()),
            _ => None,
        }
    }

    /// Download back to an Arrow `ArrayRef`. `BoolNullable` materialises a
    /// nullable `BooleanArray` by zipping values with the validity buffer.
    pub fn download(&self) -> JavelinResult<ArrayRef> {
        match self {
            ExtendedDeviceCol::Bool(v) => {
                let host: Vec<u8> = v.to_vec()?;
                let bools: Vec<bool> = host.into_iter().map(|b| b != 0).collect();
                Ok(Arc::new(BooleanArray::from(bools)) as ArrayRef)
            }
            ExtendedDeviceCol::BoolNullable { values, validity } => {
                let host_values: Vec<u8> = values.to_vec()?;
                let host_validity: Vec<u8> = validity.to_vec()?;
                let arr: BooleanArray = host_values
                    .into_iter()
                    .zip(host_validity.into_iter())
                    .map(|(v, m)| if m == 1 { Some(v == 1) } else { None })
                    .collect();
                Ok(Arc::new(arr) as ArrayRef)
            }
            ExtendedDeviceCol::Utf8(d) => {
                let arr = d.to_string_array()?;
                Ok(Arc::new(arr) as ArrayRef)
            }
        }
    }
}
