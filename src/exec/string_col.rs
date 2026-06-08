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
use crate::error::BoltResult;

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
    pub fn upload_bool(arr: &BooleanArray) -> BoltResult<Self> {
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
    pub fn upload_utf8(arr: &StringArray) -> BoltResult<Self> {
        let dict = DictionaryColumn::from_string_array(arr)?;
        Ok(ExtendedDeviceCol::Utf8(dict))
    }

    /// Allocate a zero-initialised Bool column of `len` rows (all bytes 0 →
    /// all rows decode to `false`).
    pub fn alloc_bool(len: usize) -> BoltResult<Self> {
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
    pub fn alloc_utf8(len: usize, dict: Option<Vec<String>>) -> BoltResult<Self> {
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
    pub fn download(&self) -> BoltResult<ArrayRef> {
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

// ---------------------------------------------------------------------------
// Tests. Every entry point in this module ultimately touches the CUDA driver
// (via `GpuVec::from_slice`/`zeros` or `DictionaryColumn::from_string_array`),
// so the end-to-end round-trip cases are gated behind `#[ignore]`. The pure
// host-side tests in this module exercise the *Arrow-input contract* the
// dispatch logic depends on — `len()`, `null_count()`, `is_null(i)`,
// `value(i)` semantics, and the empty/very-long/unicode/null-mix shapes —
// so a future change to those preconditions would surface here rather than
// silently mis-dispatching on a GPU host.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{types::Int32Type, DictionaryArray, LargeStringArray, StringArray};

    // -------- Pure host-side fixtures: lock in the Arrow input contract ----

    #[test]
    fn empty_string_array_has_zero_len_and_zero_nulls() {
        // upload_utf8 / upload_bool see n = arr.len() = 0; the upload loops
        // run zero iterations and the dispatch decision (Bool vs BoolNullable
        // for the bool variant, or empty-dict Utf8) depends on these values
        // being correct for the empty case.
        let arr = StringArray::from(Vec::<Option<&str>>::new());
        assert_eq!(arr.len(), 0);
        assert_eq!(arr.null_count(), 0);
    }

    #[test]
    fn single_row_string_array_round_trip_shape() {
        // The simplest non-empty input: one short non-null string. This is
        // the single-iteration path through `upload_utf8`'s inner loop and
        // the trivial 1-row case for download.
        let arr = StringArray::from(vec!["hello"]);
        assert_eq!(arr.len(), 1);
        assert_eq!(arr.null_count(), 0);
        assert!(!arr.is_null(0));
        assert_eq!(arr.value(0), "hello");
    }

    #[test]
    fn multi_row_string_array_varied_lengths() {
        // Multi-row, no nulls, mixed lengths. Verifies the contract
        // `upload_utf8` relies on when iterating `0..arr.len()` and
        // dereferencing `arr.value(i)` for each row.
        let arr = StringArray::from(vec!["a", "bb", "ccc", "dddd"]);
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.null_count(), 0);
        let collected: Vec<&str> = (0..arr.len()).map(|i| arr.value(i)).collect();
        assert_eq!(collected, vec!["a", "bb", "ccc", "dddd"]);
    }

    #[test]
    fn string_array_with_nulls_reports_null_count() {
        // upload_utf8 delegates to DictionaryColumn::from_string_array, which
        // branches on arr.is_null(i) per row. Make sure the Arrow null mask
        // is shaped the way that branch expects.
        let arr = StringArray::from(vec![Some("a"), None, Some("b"), None, Some("a")]);
        assert_eq!(arr.len(), 5);
        assert_eq!(arr.null_count(), 2);
        assert!(arr.is_null(1));
        assert!(arr.is_null(3));
        assert!(!arr.is_null(0));
        assert!(!arr.is_null(2));
        assert!(!arr.is_null(4));
        assert_eq!(arr.value(0), "a");
        assert_eq!(arr.value(2), "b");
        assert_eq!(arr.value(4), "a");
    }

    #[test]
    fn string_array_with_empty_string_is_distinct_from_null() {
        // An empty string is a valid dictionary entry distinct from NULL —
        // index 0 means NULL, an empty string takes a real index.
        let arr = StringArray::from(vec![Some(""), None, Some("")]);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 1);
        assert!(!arr.is_null(0));
        assert!(arr.is_null(1));
        assert!(!arr.is_null(2));
        assert_eq!(arr.value(0), "");
        assert_eq!(arr.value(2), "");
    }

    #[test]
    fn string_array_with_very_long_string_preserves_length() {
        // 10 KiB single string — exercise the wide-row path. The dictionary
        // builder allocates exactly one String of this size on the host.
        let long = "x".repeat(10 * 1024);
        let arr = StringArray::from(vec![long.as_str()]);
        assert_eq!(arr.len(), 1);
        assert_eq!(arr.value(0).len(), 10 * 1024);
        assert_eq!(arr.value(0), long.as_str());
    }

    #[test]
    fn string_array_with_unicode_preserves_bytes() {
        // Various unicode: CJK, Latin-with-diacritic, control + BMP boundary.
        // Arrow's StringArray is byte-addressed UTF-8, so `value(i)` must
        // round-trip the exact bytes (no normalisation).
        let nul_and_max = "\u{0}\u{FFFF}".to_string();
        let arr = StringArray::from(vec!["日本語", "naïve", nul_and_max.as_str()]);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.value(0), "日本語");
        assert_eq!(arr.value(1), "naïve");
        assert_eq!(arr.value(2), nul_and_max);
        // The unicode strings each have multi-byte chars: byte length differs
        // from char length, which is the documented contract.
        assert_eq!(arr.value(0).len(), 9); // 3 CJK chars * 3 bytes
        assert_eq!(arr.value(1).len(), 6); // n a ï(2) v e
    }

    #[test]
    fn dictionary_encoded_input_decodes_to_string_array_for_upload() {
        // `upload_utf8` consumes a *materialised* `StringArray`. Callers that
        // hold a `DictionaryArray<Int32, Utf8>` must `cast` to Utf8 first;
        // verify the path that produces the StringArray we'd hand to
        // `upload_utf8` round-trips the values correctly.
        let dict: DictionaryArray<Int32Type> = vec!["a", "b", "a", "c", "b"].into_iter().collect();
        // arrow_cast is not in scope; use the array's downcast + values API
        // to materialise the equivalent StringArray.
        let keys = dict.keys();
        let values = dict
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("dict values are Utf8");
        let materialised: StringArray = (0..dict.len())
            .map(|i| {
                if keys.is_null(i) {
                    None
                } else {
                    Some(values.value(keys.value(i) as usize))
                }
            })
            .collect();
        assert_eq!(materialised.len(), 5);
        assert_eq!(materialised.value(0), "a");
        assert_eq!(materialised.value(1), "b");
        assert_eq!(materialised.value(2), "a");
        assert_eq!(materialised.value(3), "c");
        assert_eq!(materialised.value(4), "b");
        assert_eq!(materialised.null_count(), 0);
    }

    #[test]
    fn large_utf8_input_has_the_same_arrow_contract() {
        // `upload_utf8` only accepts `StringArray` (Utf8, i32 offsets), not
        // `LargeStringArray` (LargeUtf8, i64 offsets). Document that contract
        // by showing the two types are *not* interchangeable at the Arrow
        // level, while their value semantics line up — callers wishing to
        // upload LargeUtf8 must downcast offsets first.
        let large: LargeStringArray = LargeStringArray::from(vec![Some("a"), None, Some("bb")]);
        assert_eq!(large.len(), 3);
        assert_eq!(large.null_count(), 1);
        assert_eq!(large.value(0), "a");
        assert_eq!(large.value(2), "bb");
        // The matching StringArray would carry identical observable values.
        let narrow = StringArray::from(vec![Some("a"), None, Some("bb")]);
        assert_eq!(narrow.len(), large.len());
        assert_eq!(narrow.null_count(), large.null_count());
        for i in 0..narrow.len() {
            assert_eq!(narrow.is_null(i), large.is_null(i));
            if !narrow.is_null(i) {
                assert_eq!(narrow.value(i), large.value(i));
            }
        }
    }

    #[test]
    fn boolean_array_with_no_nulls_takes_bool_branch() {
        // `upload_bool` reads `arr.null_count() == 0` to decide between the
        // Bool and BoolNullable variants. Lock in that the no-null shape
        // really reports zero.
        let arr = BooleanArray::from(vec![true, false, true]);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 0);
        assert!(arr.value(0));
        assert!(!arr.value(1));
        assert!(arr.value(2));
    }

    #[test]
    fn boolean_array_with_nulls_takes_nullable_branch() {
        // The other side of the dispatch: at least one null, so the upload
        // path produces a BoolNullable with a parallel validity buffer.
        let arr = BooleanArray::from(vec![Some(true), None, Some(false), None]);
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.null_count(), 2);
        assert!(!arr.is_null(0));
        assert!(arr.is_null(1));
        assert!(!arr.is_null(2));
        assert!(arr.is_null(3));
        assert!(arr.value(0));
        assert!(!arr.value(2));
    }

    #[test]
    fn empty_boolean_array_is_valid_no_null_input() {
        // Zero-length input must still report `null_count() == 0` so the
        // upload path takes the Bool (not BoolNullable) branch with an empty
        // value buffer.
        let arr = BooleanArray::from(Vec::<bool>::new());
        assert_eq!(arr.len(), 0);
        assert_eq!(arr.null_count(), 0);
    }

    // -------- GPU-needing round-trip tests ----------------------------------
    // These exercise the actual entry points (upload_bool, upload_utf8,
    // alloc_*, download). They allocate device memory via GpuVec and so are
    // gated behind `gpu:string_col`.

    #[test]
    #[ignore = "gpu:string_col"]
    fn gpu_upload_bool_no_nulls_round_trips() {
        let arr = BooleanArray::from(vec![true, false, true, true, false]);
        let dev = ExtendedDeviceCol::upload_bool(&arr).expect("upload");
        assert!(matches!(dev, ExtendedDeviceCol::Bool(_)));
        assert!(dev.validity_device_ptr().is_none());
        let back = dev.download().expect("download");
        let got = back
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("BooleanArray");
        assert_eq!(got.len(), 5);
        assert_eq!(got.null_count(), 0);
        let collected: Vec<bool> = (0..got.len()).map(|i| got.value(i)).collect();
        assert_eq!(collected, vec![true, false, true, true, false]);
    }

    #[test]
    #[ignore = "gpu:string_col"]
    fn gpu_upload_bool_with_nulls_preserves_validity() {
        let arr = BooleanArray::from(vec![Some(true), None, Some(false), None, Some(true)]);
        let dev = ExtendedDeviceCol::upload_bool(&arr).expect("upload");
        assert!(matches!(dev, ExtendedDeviceCol::BoolNullable { .. }));
        assert!(dev.validity_device_ptr().is_some());
        let back = dev.download().expect("download");
        let got = back
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("BooleanArray");
        assert_eq!(got.len(), 5);
        assert_eq!(got.null_count(), 2);
        assert!(got.value(0));
        assert!(got.is_null(1));
        assert!(!got.value(2));
        assert!(got.is_null(3));
        assert!(got.value(4));
    }

    #[test]
    #[ignore = "gpu:string_col"]
    fn gpu_upload_bool_empty_round_trips() {
        let arr = BooleanArray::from(Vec::<bool>::new());
        let dev = ExtendedDeviceCol::upload_bool(&arr).expect("upload");
        assert!(matches!(dev, ExtendedDeviceCol::Bool(_)));
        let back = dev.download().expect("download");
        let got = back
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("BooleanArray");
        assert_eq!(got.len(), 0);
    }

    #[test]
    #[ignore = "gpu:string_col"]
    fn gpu_upload_utf8_round_trip_with_nulls_and_dedupe() {
        // Mix of repeated values, nulls, empty string, and unicode — the
        // dictionary path must dedupe distinct non-nulls, encode NULL as 0,
        // and preserve every byte on download.
        let long = "x".repeat(10 * 1024);
        let arr = StringArray::from(vec![
            Some("a"),
            None,
            Some(""),
            Some("a"),
            Some("日本語"),
            Some("naïve"),
            Some(long.as_str()),
        ]);
        let dev = ExtendedDeviceCol::upload_utf8(&arr).expect("upload");
        assert!(matches!(dev, ExtendedDeviceCol::Utf8(_)));
        assert!(dev.validity_device_ptr().is_none());
        let back = dev.download().expect("download");
        let got = back
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("StringArray");
        assert_eq!(got.len(), 7);
        assert_eq!(got.null_count(), 1);
        assert_eq!(got.value(0), "a");
        assert!(got.is_null(1));
        assert_eq!(got.value(2), "");
        assert_eq!(got.value(3), "a");
        assert_eq!(got.value(4), "日本語");
        assert_eq!(got.value(5), "naïve");
        assert_eq!(got.value(6), long.as_str());
    }

    #[test]
    #[ignore = "gpu:string_col"]
    fn gpu_upload_utf8_empty_round_trips() {
        let arr = StringArray::from(Vec::<Option<&str>>::new());
        let dev = ExtendedDeviceCol::upload_utf8(&arr).expect("upload");
        let back = dev.download().expect("download");
        let got = back
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("StringArray");
        assert_eq!(got.len(), 0);
    }

    #[test]
    #[ignore = "gpu:string_col"]
    fn gpu_alloc_bool_is_all_false() {
        let dev = ExtendedDeviceCol::alloc_bool(4).expect("alloc");
        assert!(matches!(dev, ExtendedDeviceCol::Bool(_)));
        let back = dev.download().expect("download");
        let got = back
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("BooleanArray");
        assert_eq!(got.len(), 4);
        for i in 0..got.len() {
            assert!(!got.value(i), "row {i} must default to false");
        }
    }

    #[test]
    #[ignore = "gpu:string_col"]
    fn gpu_alloc_utf8_with_dict_decodes_to_nulls() {
        // alloc_utf8 zeroes the index buffer, so every row decodes to NULL
        // (index 0 reserved for NULL). Passing a non-empty dictionary must
        // not change that — the indices, not the dict, determine the rows.
        let dict = Some(vec!["a".to_string(), "b".to_string()]);
        let dev = ExtendedDeviceCol::alloc_utf8(3, dict).expect("alloc");
        let back = dev.download().expect("download");
        let got = back
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("StringArray");
        assert_eq!(got.len(), 3);
        assert_eq!(got.null_count(), 3);
    }
}
