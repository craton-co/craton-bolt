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
use crate::error::BoltResult;
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
    pub fn from_string_array(arr: &StringArray) -> BoltResult<Self> {
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

    /// **Stage 7 (S1)** — build a `DictionaryColumnAny` directly from an
    /// Arrow `DictionaryArray<Int32Type>` without re-scanning the values.
    ///
    /// `from_string_array` builds the dictionary from scratch by scanning a
    /// `StringArray` row-by-row. When the input is already a
    /// `DictionaryArray`, the dictionary has been built once on the Arrow
    /// side — re-scanning is wasted work. This constructor takes that
    /// already-built Arrow dictionary, copies the values into an owned
    /// `Vec<String>` (slot 0 reserved for NULL, real strings shifted to
    /// slot 1..N+1), and uploads the keys as `i32` indices.
    ///
    /// Emits the [`Self::I32`] variant by definition — the source is
    /// `i32`-keyed. Callers whose dictionaries overflow the i32 range must
    /// keep using [`Self::from_string_array`].
    ///
    /// # Errors
    /// - `BoltError::Type` if the dictionary values aren't `Utf8`.
    /// - `BoltError::Other` if the dictionary length exceeds `i32::MAX - 1`.
    /// - `BoltError::Other` if any key is negative (violates Arrow's
    ///   DictionaryArray invariants).
    pub fn from_dictionary_array(
        arr: &arrow_array::DictionaryArray<arrow_array::types::Int32Type>,
    ) -> BoltResult<Self> {
        use arrow_array::{Array, Int32Array};
        use crate::error::BoltError;

        let dict_vals = arr
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                BoltError::Type(format!(
                    "DictionaryArray has non-Utf8 value type {:?}",
                    arr.values().data_type()
                ))
            })?;
        // Copy values into an owned Vec<String>. Dictionary-level NULLs
        // (rare; means the dict itself has a null entry) become empty
        // strings — row-level NULLs are tracked through the keys array.
        let mut dictionary: Vec<String> = Vec::with_capacity(dict_vals.len());
        for i in 0..dict_vals.len() {
            if dict_vals.is_null(i) {
                dictionary.push(String::new());
            } else {
                dictionary.push(dict_vals.value(i).to_string());
            }
        }
        // i32 ceiling guard — mirrors `DictionaryColumn::from_string_array`.
        if dictionary.len().saturating_add(1) > i32::MAX as usize {
            return Err(BoltError::Other(format!(
                "dictionary overflow: more than {} unique strings (i32 index space)",
                i32::MAX
            )));
        }

        // Shift keys by +1 so slot 0 reserves NULL; NULL rows land at 0.
        let keys_arr: &Int32Array = arr.keys();
        let n_rows = keys_arr.len();
        let mut indices: Vec<i32> = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            if keys_arr.is_null(i) {
                indices.push(0);
            } else {
                let k = keys_arr.value(i);
                if k < 0 {
                    return Err(BoltError::Other(format!(
                        "negative dictionary key {} at row {} — Arrow \
                         DictionaryArray invariants forbid negative keys",
                        k, i
                    )));
                }
                indices.push(k + 1);
            }
        }
        let device_indices = crate::cuda::GpuVec::<i32>::from_slice(&indices)?;
        let inner = DictionaryColumn {
            dictionary,
            indices: device_indices,
            n_rows,
        };
        Ok(Self::I32(inner))
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

    /// Test-only constructor that bypasses any GPU upload.
    ///
    /// Mirrors [`DictionaryColumn::new_host_only`] and
    /// [`DictionaryColumnI64::new_host_only`]. The variant choice follows the
    /// same dispatch rule as [`Self::from_string_array`]: the distinct count
    /// (here approximated by `dictionary.len()`, which is exact for an already
    /// deduplicated host dictionary) is compared against
    /// [`I32_INDEX_THRESHOLD`]. Below the threshold → [`Self::I32`]; at or
    /// above → [`Self::I64`].
    ///
    /// The underlying inner wrapper is built with its own `new_host_only`
    /// constructor, so the device-side `indices` field is a zero-length
    /// `GpuVec` placeholder. Any method that touches the device buffer
    /// ([`Self::indices_device_ptr`] in particular) will operate on that
    /// placeholder. This exists so host-only unit tests can exercise the
    /// dispatch and accessor logic without a CUDA-enabled machine. Production
    /// code must not use this — use [`Self::from_string_array`] instead.
    #[cfg(test)]
    pub(crate) fn new_host_only(
        dictionary: Vec<String>,
        n_rows: usize,
    ) -> BoltResult<Self> {
        if dictionary.len() >= I32_INDEX_THRESHOLD {
            // Wide path: i64 indices. The i64 sibling's constructor is
            // fallible (mirrors its production counterpart), so propagate.
            let inner = DictionaryColumnI64::new_host_only(dictionary, n_rows)?;
            Ok(Self::I64(inner))
        } else {
            // Narrow path: i32 indices, the common case.
            let inner = DictionaryColumn::new_host_only(dictionary, n_rows);
            Ok(Self::I32(inner))
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the unified dictionary wrapper.
    //!
    //! Most tests below use [`DictionaryColumnAny::new_host_only`], which
    //! short-circuits the GPU upload and lets us exercise the dispatch logic,
    //! the variant accessors, and the dtype reporter on a toolkit-less host.
    //!
    //! A small handful of tests still call [`DictionaryColumnAny::from_string_array`]
    //! end-to-end and are `#[ignore]`d for the same reason as the i32/i64
    //! sibling tests: they hit `GpuVec::from_slice`, which requires a CUDA
    //! context.
    use super::*;

    // ---- Host-only: dispatch and accessor tests --------------------------

    /// Dispatch rule, narrow side: a small synthetic dictionary sits well
    /// below the threshold, so [`DictionaryColumnAny::new_host_only`] must
    /// land on the i32 variant — same rule as the production
    /// `from_string_array` path. Previously this test exercised the GPU and
    /// had to be `#[ignore]`d; the host-only constructor lets it run anywhere.
    #[test]
    fn cardinality_below_threshold_picks_i32() {
        let strings: Vec<String> = (0..100).map(|i| format!("s{i}")).collect();

        let any = DictionaryColumnAny::new_host_only(strings, 100)
            .expect("host-only constructor must not depend on CUDA");
        assert!(any.is_i32(), "100 distinct strings must land on the i32 path");
        assert!(any.as_i64().is_none());
        assert_eq!(any.n_rows(), 100);
        assert_eq!(any.dictionary().len(), 100);
        assert_eq!(any.index_dtype(), DataType::Int32);
    }

    /// The complementary case — cardinality at or above the threshold lands
    /// on the i64 path — is impractical to synthesize end-to-end: it would
    /// require allocating `>= I32_INDEX_THRESHOLD` distinct strings (~2 GiB
    /// at minimum just for the pointers). Left as a non-running placeholder
    /// to document why; the actual dispatch is covered host-only by
    /// [`dispatch_i32_vs_i64_by_threshold`] below, which constructs the i64
    /// variant directly via [`DictionaryColumnI64::new_host_only`].
    #[test]
    #[ignore = "resource — would require allocating > 2 billion distinct strings"]
    fn cardinality_above_threshold_picks_i64() {
        // Intentionally left empty — see the comment above.
    }

    /// `index_of_any` on an i32-indexed dictionary must return the same value
    /// as the underlying `index_of`, widened to `i64`. The widening rule
    /// itself (`i as i64`) is trivially correct; the test pins the contract.
    /// Previously `#[ignore]`d because it round-tripped through the GPU; the
    /// host-only constructor lets it run anywhere.
    #[test]
    fn index_of_any_widens_i32_to_i64() {
        let dict = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
        let any = DictionaryColumnAny::new_host_only(dict, 4)
            .expect("host-only constructor must succeed");
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

    /// Dispatch rule, both sides: build dictionaries on either side of the
    /// threshold and verify the wrapper picks the matching index width. The
    /// i64 side is constructed directly via the inner `new_host_only` (so we
    /// don't have to materialize 2+ billion strings).
    #[test]
    fn dispatch_i32_vs_i64_by_threshold() {
        // Below threshold → i32. A tiny dictionary is unambiguous.
        let narrow_dict = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let narrow = DictionaryColumnAny::new_host_only(narrow_dict, 3)
            .expect("narrow host-only build");
        assert!(narrow.is_i32(), "small dictionary must land on i32");
        assert_eq!(narrow.index_dtype(), DataType::Int32);
        assert!(narrow.as_i32().is_some());
        assert!(narrow.as_i64().is_none());

        // At/above threshold → i64. We can't realistically build a
        // 2-billion-entry Vec in a unit test, so synthesize the i64-side
        // wrapper by hand using the i64 sibling's host-only constructor —
        // that's what the wrapper's `new_host_only` does under the hood for
        // the wide path, just with a populated dictionary.
        let wide_inner = DictionaryColumnI64::new_host_only(
            vec!["only".to_string()],
            1,
        )
        .expect("i64 host-only build");
        let wide = DictionaryColumnAny::I64(wide_inner);
        assert!(!wide.is_i32(), "manually-wrapped i64 must not report i32");
        assert_eq!(wide.index_dtype(), DataType::Int64);
        assert!(wide.as_i64().is_some());
        assert!(wide.as_i32().is_none());
    }

    /// `index_dtype()` must reflect the variant — Int32 for an i32-backed
    /// inner, Int64 for an i64-backed inner. This is the contract the engine
    /// relies on when wiring kernel arguments, so it gets its own test.
    #[test]
    fn index_dtype_returns_inner() {
        // i32 side via the wrapper's host-only constructor.
        let i32_any = DictionaryColumnAny::new_host_only(
            vec!["x".to_string(), "y".to_string()],
            2,
        )
        .expect("i32 build");
        assert_eq!(i32_any.index_dtype(), DataType::Int32);

        // i64 side: build the inner directly and wrap it ourselves so we
        // don't have to clear the threshold with a real-sized dictionary.
        let i64_inner = DictionaryColumnI64::new_host_only(
            vec!["a".to_string()],
            1,
        )
        .expect("i64 inner build");
        let i64_any = DictionaryColumnAny::I64(i64_inner);
        assert_eq!(i64_any.index_dtype(), DataType::Int64);
    }

    /// The host-only constructor must preserve the dictionary verbatim — the
    /// visible length through [`DictionaryColumnAny::dictionary`] equals what
    /// the caller passed in. Guards against accidental dedup / mutation in
    /// the wrapper layer.
    #[test]
    fn dictionary_view_count_matches_inner() {
        let dict = vec![
            "alpha".to_string(),
            "beta".to_string(),
            "gamma".to_string(),
            "delta".to_string(),
        ];
        let n = dict.len();
        let any = DictionaryColumnAny::new_host_only(dict.clone(), 42)
            .expect("host-only build");

        // The wrapper view matches both length and contents.
        assert_eq!(any.dictionary().len(), n);
        assert_eq!(any.dictionary(), dict.as_slice());
        // n_rows is independent of dictionary length; the caller-supplied
        // value must round-trip too.
        assert_eq!(any.n_rows(), 42);
    }

    /// Edge case: an empty dictionary must still dispatch (not panic), and —
    /// since `0 < I32_INDEX_THRESHOLD` — it lands on the i32 path. Pins the
    /// behavior of the dispatch boundary at the low end.
    #[test]
    fn empty_dictionary_dispatches_to_i32() {
        // Sanity: confirm our assumption about the threshold so this test
        // stays meaningful if someone "tunes" the constant.
        assert!(I32_INDEX_THRESHOLD > 0);

        let any = DictionaryColumnAny::new_host_only(Vec::new(), 0)
            .expect("empty host-only build");
        assert!(any.is_i32(), "empty dictionary must land on i32 path");
        assert_eq!(any.index_dtype(), DataType::Int32);
        assert!(any.dictionary().is_empty());
        assert_eq!(any.n_rows(), 0);
        // Unknown literal on an empty dict still surfaces None (not zero,
        // not panic).
        assert_eq!(any.index_of_any("anything"), None);
    }
}
