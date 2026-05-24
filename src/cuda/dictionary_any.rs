// SPDX-License-Identifier: Apache-2.0

//! Unified wrapper over the i32- and i64-indexed dictionary variants.
//!
//! Two flavours of dictionary live in this crate:
//!   * [`DictionaryColumn`] — i32 indices, cheaper, the default.
//!   * [`DictionaryColumnI64`] — i64 indices, used when a column's distinct
//!     string count crowds the i32 range.
//!
//! Callers that don't care which variant is in play — the dictionary registry,
//! the engine's upload path, the literal resolver — want a single handle that
//! abstracts the difference. [`DictionaryColumnAny`] is that handle: a thin
//! enum over the two underlying types with helpers that erase the index-width
//! distinction wherever it doesn't matter (NULL is still slot 0, real strings
//! still start at slot 1, lookups still widen losslessly to `i64`).
//!
//! The variant choice is made once, at [`Self::from_string_array`] time, based
//! on [`crate::cuda::dictionary_i64::estimate_distinct_count`] compared against
//! [`I32_INDEX_THRESHOLD`]. The decision is sticky: once a column is encoded
//! into a particular width, callers see only that width through this wrapper.

use arrow_array::StringArray;

use crate::cuda::cuda_sys::CUdeviceptr;
use crate::cuda::dictionary::DictionaryColumn;
use crate::cuda::dictionary_i64::{
    estimate_distinct_count, DictionaryColumnI64, I32_INDEX_THRESHOLD,
};
use crate::error::JavelinResult;
use crate::plan::logical_plan::DataType;

/// Unified wrapper over the two dictionary variants.
///
/// Owns the underlying [`DictionaryColumn`] / [`DictionaryColumnI64`] — and
/// therefore the GPU allocation backing it. Drop the wrapper, drop the device
/// memory.
pub enum DictionaryColumnAny {
    /// i32-indexed variant; preferred when the column's cardinality fits
    /// comfortably below `i32::MAX`.
    I32(DictionaryColumn),
    /// i64-indexed variant; used when the column's cardinality estimate
    /// crowds the i32 range (see [`I32_INDEX_THRESHOLD`]).
    I64(DictionaryColumnI64),
}

impl DictionaryColumnAny {
    /// Choose the index width based on cardinality, then encode `arr`.
    ///
    /// The decision rule is exact, not heuristic: if
    /// [`estimate_distinct_count`] reports `>= I32_INDEX_THRESHOLD` distinct
    /// non-null strings, the column is encoded as [`Self::I64`]; otherwise as
    /// [`Self::I32`]. The threshold sits a small margin below `i32::MAX` so
    /// callers never have to reason about the absolute edge of the i32 range.
    ///
    /// Both branches forward to the underlying type's `from_string_array`,
    /// which in turn uploads the index column to the device.
    pub fn from_string_array(arr: &StringArray) -> JavelinResult<Self> {
        let distinct = estimate_distinct_count(arr);
        if distinct >= I32_INDEX_THRESHOLD {
            // Wide path: i64 indices, headroom up to i64::MAX.
            let inner = DictionaryColumnI64::from_string_array(arr)?;
            Ok(Self::I64(inner))
        } else {
            // Narrow path: i32 indices, the common case.
            let inner = DictionaryColumn::from_string_array(arr)?;
            Ok(Self::I32(inner))
        }
    }

    /// Borrow the host-side dictionary (slot 0 = NULL, real strings at 1..).
    ///
    /// Layout is identical across variants — only the index integer width
    /// differs — so callers can decode literals against this slice without
    /// caring which variant they hold.
    pub fn dictionary(&self) -> &[String] {
        match self {
            Self::I32(d) => &d.dictionary,
            Self::I64(d) => &d.dictionary,
        }
    }

    /// Number of source rows the column was built from.
    pub fn n_rows(&self) -> usize {
        match self {
            Self::I32(d) => d.n_rows,
            Self::I64(d) => d.n_rows,
        }
    }

    /// Device pointer to the index column.
    ///
    /// The pointer's element width depends on the variant: an `i32*` for
    /// [`Self::I32`], an `i64*` for [`Self::I64`]. Use [`Self::index_dtype`]
    /// to recover that width when dispatching kernel arguments.
    pub fn indices_device_ptr(&self) -> CUdeviceptr {
        match self {
            Self::I32(d) => d.indices.device_ptr(),
            Self::I64(d) => d.indices.device_ptr(),
        }
    }

    /// Lookup the index of a literal string. Returns `None` if `s` was not in
    /// the dictionary at construction time.
    ///
    /// The result is always widened to `i64` — an `i32` index loses no
    /// information when widened to `i64`, and a single return type spares
    /// callers from branching on the variant just to forward a literal index
    /// to the planner. `None` is not an error: an unknown literal trivially
    /// matches no rows.
    pub fn index_of_any(&self, s: &str) -> Option<i64> {
        match self {
            // i32 → i64 widening is lossless; `as i64` is the canonical path.
            Self::I32(d) => d.index_of(s).map(|i| i as i64),
            Self::I64(d) => d.index_of(s),
        }
    }

    /// Plan dtype of the index column. Drives the engine's `__idx_<col>`
    /// schema injection and any per-column kernel-arg dispatch.
    pub fn index_dtype(&self) -> DataType {
        match self {
            Self::I32(_) => DataType::Int32,
            Self::I64(_) => DataType::Int64,
        }
    }

    /// True if the underlying variant uses i32 indices.
    pub fn is_i32(&self) -> bool {
        matches!(self, Self::I32(_))
    }

    /// Borrow as an i32 dictionary if that's the variant; otherwise `None`.
    pub fn as_i32(&self) -> Option<&DictionaryColumn> {
        match self {
            Self::I32(d) => Some(d),
            Self::I64(_) => None,
        }
    }

    /// Borrow as an i64 dictionary if that's the variant; otherwise `None`.
    pub fn as_i64(&self) -> Option<&DictionaryColumnI64> {
        match self {
            Self::I64(d) => Some(d),
            Self::I32(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity-check the dispatch rule on a small synthetic input: a 100-row,
    /// 100-distinct StringArray sits far below the threshold, so the picker
    /// must choose i32.
    ///
    /// Marked `#[ignore]` because `from_string_array` uploads the index
    /// vector to the device — the build machine has no CUDA toolkit.
    #[test]
    #[ignore = "requires CUDA toolkit at runtime (GpuVec upload)"]
    fn cardinality_below_threshold_picks_i32() {
        let strings: Vec<String> = (0..100).map(|i| format!("s{i}")).collect();
        let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();
        let arr = StringArray::from(refs);

        let any = DictionaryColumnAny::from_string_array(&arr)
            .expect("encode should succeed on a small input");
        assert!(any.is_i32(), "100 distinct strings must land on the i32 path");
        assert!(any.as_i64().is_none());
        assert_eq!(any.n_rows(), 100);
        assert_eq!(any.dictionary().len(), 100);
        assert_eq!(any.index_dtype(), DataType::Int32);
    }

    /// The complementary case — cardinality at or above the threshold lands
    /// on the i64 path — is impractical to synthesize: it would require
    /// allocating `>= I32_INDEX_THRESHOLD` distinct strings (~2 GiB at
    /// minimum just for the pointers). The dispatch rule is exercised
    /// indirectly by `estimate_distinct_count`'s tests and the structure of
    /// [`DictionaryColumnAny::from_string_array`]; ignore the integration
    /// here.
    #[test]
    #[ignore = "would require allocating > 2 billion distinct strings"]
    fn cardinality_above_threshold_picks_i64() {
        // Intentionally left empty — see the comment above.
    }

    /// `index_of_any` on an i32-indexed dictionary must return the same value
    /// as the underlying `index_of`, widened to `i64`.
    ///
    /// Marked `#[ignore]` because building the dictionary uploads indices to
    /// the device. The widening rule itself (`i as i64`) is trivially
    /// correct; the test exists to pin the contract.
    #[test]
    #[ignore = "requires CUDA toolkit at runtime (GpuVec upload)"]
    fn index_of_any_widens_i32_to_i64() {
        let arr = StringArray::from(vec!["X", "Y", "X", "Z"]);
        let any = DictionaryColumnAny::from_string_array(&arr)
            .expect("encode should succeed");
        assert!(any.is_i32(), "tiny input must take the i32 path");

        let i32_dict = any.as_i32().expect("variant is i32");
        // The widened result must match the underlying i32 lookup, slot-for-slot.
        assert_eq!(
            any.index_of_any("X"),
            i32_dict.index_of("X").map(|i| i as i64),
        );
        assert_eq!(
            any.index_of_any("Z"),
            i32_dict.index_of("Z").map(|i| i as i64),
        );
        // Unknown literal still surfaces as None.
        assert_eq!(any.index_of_any("missing"), None);
    }
}
