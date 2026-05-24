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
//!   upload time (`0`/`1`) and re-pack to a bitmap on download. Nulls are
//!   conservatively materialised as `0` (a documented choice, NOT semantic
//!   three-valued logic — that's the planner's problem). Round-tripping a
//!   nullable `BooleanArray` through this path is therefore lossy w.r.t.
//!   null/false distinction.
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
    /// Boolean column. One `u8` per row on the device: `0` for false/null,
    /// `1` for true. See module docs for the null caveat.
    Bool(GpuVec<u8>),
    /// UTF-8 string column. Stored as i32 dictionary indices on the device;
    /// the host-side dictionary lives inside the `DictionaryColumn`.
    Utf8(DictionaryColumn),
}

impl ExtendedDeviceCol {
    /// Upload a `BooleanArray` as a `u8`-per-row column.
    ///
    /// Nulls are written as `0`. Callers that need to preserve null/false
    /// distinction must layer their own validity bitmap on top.
    pub fn upload_bool(arr: &BooleanArray) -> JavelinResult<Self> {
        let n = arr.len();
        let mut bytes: Vec<u8> = Vec::with_capacity(n);
        for i in 0..n {
            // `is_null` checks the validity bitmap; `value` is well-defined
            // even for null slots (it reads the underlying bit) so guard.
            if arr.is_null(i) {
                bytes.push(0);
            } else {
                bytes.push(if arr.value(i) { 1 } else { 0 });
            }
        }
        let gpu = GpuVec::<u8>::from_slice(&bytes)?;
        Ok(ExtendedDeviceCol::Bool(gpu))
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

    /// Raw device pointer (for `cuLaunchKernel` args).
    pub fn device_ptr(&self) -> u64 {
        match self {
            ExtendedDeviceCol::Bool(v) => v.device_ptr(),
            ExtendedDeviceCol::Utf8(d) => d.indices.device_ptr(),
        }
    }

    /// Download back to an Arrow `ArrayRef`.
    pub fn download(&self) -> JavelinResult<ArrayRef> {
        match self {
            ExtendedDeviceCol::Bool(v) => {
                let host: Vec<u8> = v.to_vec()?;
                let bools: Vec<bool> = host.into_iter().map(|b| b != 0).collect();
                Ok(Arc::new(BooleanArray::from(bools)) as ArrayRef)
            }
            ExtendedDeviceCol::Utf8(d) => {
                let arr = d.to_string_array()?;
                Ok(Arc::new(arr) as ArrayRef)
            }
        }
    }
}
