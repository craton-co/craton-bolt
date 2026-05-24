// SPDX-License-Identifier: Apache-2.0

//! Host-side string dictionary paired with on-device **i64** indices.
//!
//! This is the wide-index sibling of [`crate::cuda::dictionary::DictionaryColumn`].
//! The layout convention is identical — index `0` is reserved for SQL `NULL`,
//! real strings start at index `1`, and `dictionary[k - 1]` decodes GPU index
//! `k` — only the index width changes from `i32` to `i64`.
//!
//! The engine picks between the i32 and i64 variants at register-table time
//! based on a distinct-string estimate. The threshold lives in
//! [`I32_INDEX_THRESHOLD`]: any column whose unique-string count is at or above
//! the threshold uses [`DictionaryColumnI64`]. The cap on this type is
//! `i64::MAX`, which is effectively unbounded for any input that fits in host
//! memory.
//!
//! Deferred: downstream PTX kernels still consume i32 indices. Wiring the
//! codegen path to accept i64 indices belongs to the orchestrator and is out
//! of scope for this file.

use std::collections::{HashMap, HashSet};

use arrow_array::{Array, StringArray};

use crate::cuda::GpuVec;
use crate::error::{JavelinError, JavelinResult};

/// Threshold for picking i64 over i32 indices.
///
/// Set to `i32::MAX - 1024` — a small safety margin so the engine never has to
/// reason about the absolute edge of the i32 range. A column whose
/// distinct-string estimate meets or exceeds this value should be encoded with
/// [`DictionaryColumnI64`]; everything else stays on the cheaper i32 path.
pub const I32_INDEX_THRESHOLD: usize = (i32::MAX as usize) - 1024;

/// i64-indexed dictionary, for columns with > `i32::MAX` unique strings.
///
/// Layout convention (mirrors [`crate::cuda::dictionary::DictionaryColumn`]):
///   * Index `0` is reserved for SQL `NULL`. Never appears in `dictionary[]`.
///   * Real strings start at index `1`; the i-th unique non-null string
///     occupies `dictionary[i - 1]`.
///   * Indices are `i64`. The cap is `i64::MAX` unique strings (effectively
///     unlimited for any host that fits the input column in memory).
pub struct DictionaryColumnI64 {
    /// Host-side dictionary: position `i` → the `(i + 1)`-th index's string.
    pub dictionary: Vec<String>,
    /// GPU-side indices: one `i64` per source row. `0` means NULL.
    pub indices: GpuVec<i64>,
    /// Number of source rows.
    pub n_rows: usize,
}

impl DictionaryColumnI64 {
    /// Encode an Arrow `StringArray` as an i64-indexed dictionary.
    ///
    /// Nulls in `arr` map to index `0`. Distinct non-null strings are
    /// deduplicated and assigned sequential indices starting at `1`, in
    /// first-occurrence order. Returns `JavelinError::Other` if the unique
    /// count would overflow `i64::MAX` (defensive — unreachable in practice).
    pub fn from_string_array(arr: &StringArray) -> JavelinResult<Self> {
        let n_rows = arr.len();
        let mut dictionary: Vec<String> = Vec::new();
        let mut lookup: HashMap<String, i64> = HashMap::new();
        let mut indices: Vec<i64> = Vec::with_capacity(n_rows);

        for i in 0..n_rows {
            if arr.is_null(i) {
                indices.push(0);
                continue;
            }
            let s = arr.value(i);
            if let Some(&idx) = lookup.get(s) {
                indices.push(idx);
            } else {
                // Next index = current dictionary length + 1 (slot 0 reserved for NULL).
                let next_len = dictionary.len().checked_add(1).ok_or_else(|| {
                    JavelinError::Other(
                        "dictionary overflow: more than usize::MAX unique strings".into(),
                    )
                })?;
                if (next_len as u128) > (i64::MAX as u128) {
                    return Err(JavelinError::Other(format!(
                        "dictionary overflow: more than {} unique strings (i64 index space)",
                        i64::MAX
                    )));
                }
                let idx = next_len as i64;
                let owned = s.to_string();
                dictionary.push(owned.clone());
                lookup.insert(owned, idx);
                indices.push(idx);
            }
        }

        let device_indices = GpuVec::<i64>::from_slice(&indices)?;
        Ok(Self {
            dictionary,
            indices: device_indices,
            n_rows,
        })
    }

    /// Lookup the index of a literal string in the dictionary.
    ///
    /// Returns `Some(index)` if `s` was seen during construction, or `None`
    /// otherwise. Linear scan matches the i32 sibling: literal lookups happen
    /// once per query, so per-query cost is dominated by the per-row scan, not
    /// dictionary size.
    pub fn index_of(&self, s: &str) -> Option<i64> {
        self.dictionary
            .iter()
            .position(|d| d == s)
            // position is 0-based; real indices start at 1.
            .map(|p| (p as i64) + 1)
    }

    /// Download indices and reconstruct a `StringArray`.
    ///
    /// Index `0` becomes a SQL `NULL`. Negative indices or indices outside
    /// `1..=dictionary.len()` surface as `JavelinError::Other` — that would
    /// indicate a kernel wrote something the host dictionary cannot decode.
    pub fn to_string_array(&self) -> JavelinResult<StringArray> {
        let host_indices: Vec<i64> = self.indices.to_vec()?;
        let mut out: Vec<Option<&str>> = Vec::with_capacity(host_indices.len());
        for &idx in &host_indices {
            if idx == 0 {
                out.push(None);
            } else if idx < 0 {
                return Err(JavelinError::Other(format!(
                    "dictionary decode: negative index {} (NULL is encoded as 0)",
                    idx
                )));
            } else {
                let pos = (idx as u64) as usize - 1;
                let s = self.dictionary.get(pos).ok_or_else(|| {
                    JavelinError::Other(format!(
                        "dictionary decode: index {} out of range (dictionary size {})",
                        idx,
                        self.dictionary.len()
                    ))
                })?;
                out.push(Some(s.as_str()));
            }
        }
        Ok(StringArray::from(out))
    }

    /// Construct from an existing `DictionaryColumn` (i32-indexed).
    ///
    /// Downloads the i32 indices, widens each to `i64`, and re-uploads. The
    /// host dictionary is cloned verbatim — the same strings get the same
    /// numeric slot, just in a wider integer. Cheap relative to a re-encode
    /// from the source `StringArray`: no hashing, no string allocation, no
    /// duplicate scan — just one device→host copy, an `as i64` widen, and one
    /// host→device copy.
    pub fn from_i32(input: &crate::cuda::dictionary::DictionaryColumn) -> JavelinResult<Self> {
        let narrow: Vec<i32> = input.indices.to_vec()?;
        let wide: Vec<i64> = narrow.into_iter().map(|x| x as i64).collect();
        let device_indices = GpuVec::<i64>::from_slice(&wide)?;
        Ok(Self {
            dictionary: input.dictionary.clone(),
            indices: device_indices,
            n_rows: input.n_rows,
        })
    }

    /// Try to narrow back to a `DictionaryColumn` (i32-indexed) if every index
    /// fits in `i32`.
    ///
    /// Downloads the indices, checks each against `[0, i32::MAX]`, narrows to
    /// `Vec<i32>`, and uploads via `GpuVec::<i32>::from_slice`. Returns
    /// `JavelinError::Other` on the first index that falls outside the i32
    /// range (including negatives — those were always invalid).
    pub fn try_into_i32(self) -> JavelinResult<crate::cuda::dictionary::DictionaryColumn> {
        let wide: Vec<i64> = self.indices.to_vec()?;
        let mut narrow: Vec<i32> = Vec::with_capacity(wide.len());
        for &idx in &wide {
            if idx < 0 {
                return Err(JavelinError::Other(format!(
                    "narrow to i32: negative index {} (invalid; NULL is 0)",
                    idx
                )));
            }
            if idx > i32::MAX as i64 {
                return Err(JavelinError::Other(format!(
                    "narrow to i32: index {} exceeds i32::MAX ({})",
                    idx,
                    i32::MAX
                )));
            }
            narrow.push(idx as i32);
        }
        let device_indices = GpuVec::<i32>::from_slice(&narrow)?;
        Ok(crate::cuda::dictionary::DictionaryColumn {
            dictionary: self.dictionary,
            indices: device_indices,
            n_rows: self.n_rows,
        })
    }

    /// Test-only constructor that bypasses any GPU upload.
    ///
    /// The `indices` field is initialized to a zero-length `GpuVec`
    /// placeholder; callers may only exercise the host-side `dictionary`
    /// field (e.g. via [`Self::index_of`]). Any method that touches the
    /// device buffer ([`Self::to_string_array`], [`Self::try_into_i32`])
    /// will operate on an empty index vector.
    ///
    /// This exists so host-only unit tests can construct a populated
    /// dictionary without requiring a CUDA-enabled machine. Production code
    /// must not use this — use [`Self::from_string_array`] or
    /// [`Self::from_i32`] instead.
    #[cfg(test)]
    pub(crate) fn new_host_only(
        dictionary: Vec<String>,
        n_rows: usize,
    ) -> JavelinResult<Self> {
        Ok(Self {
            dictionary,
            indices: GpuVec::<i64>::empty(),
            n_rows,
        })
    }
}

/// Estimate the unique-string count of a `StringArray`.
///
/// Walks the array building a `HashSet<&str>` over non-null values and returns
/// its size. The result is exact, not approximate — the function is named
/// "estimate" because the engine consumes it as a planning hint, not a
/// guarantee.
///
/// Cost: O(n) time, O(distinct) extra memory. Nulls are excluded. Acceptable
/// at register-table time; not intended for query-hot paths.
pub fn estimate_distinct_count(arr: &StringArray) -> usize {
    let mut seen: HashSet<&str> = HashSet::new();
    for i in 0..arr.len() {
        if arr.is_null(i) {
            continue;
        }
        seen.insert(arr.value(i));
    }
    seen.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_distinct_basic() {
        let arr = StringArray::from(vec!["a", "b", "a", "c"]);
        assert_eq!(estimate_distinct_count(&arr), 3);
    }

    #[test]
    fn estimate_distinct_with_nulls() {
        // Nulls do not count toward the distinct estimate.
        let arr = StringArray::from(vec![Some("a"), None, Some("a")]);
        assert_eq!(estimate_distinct_count(&arr), 1);
    }

    #[test]
    fn index_of_pure_via_dictionary_field() {
        // Uses the test-only constructor to avoid touching the GPU.
        let dict = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let col = DictionaryColumnI64::new_host_only(dict, 0)
            .expect("host-only constructor must not depend on CUDA");

        // Slot 0 is reserved for NULL; real strings start at 1.
        assert_eq!(col.index_of("alpha"), Some(1));
        assert_eq!(col.index_of("beta"), Some(2));
        assert_eq!(col.index_of("gamma"), Some(3));
        // Absent literal → None (predicate trivially matches no rows).
        assert_eq!(col.index_of("delta"), None);
    }

    #[test]
    fn threshold_value_is_just_under_i32_max() {
        // Sanity-check that the threshold sits a small, fixed margin below
        // i32::MAX. If someone "tunes" this without thinking, the engine's
        // dispatch breaks; this test pins the contract.
        assert_eq!(I32_INDEX_THRESHOLD, (i32::MAX as usize) - 1024);
        assert!(I32_INDEX_THRESHOLD < i32::MAX as usize);
    }

    // The following tests require an actual CUDA context for GpuVec upload /
    // download. They are marked `#[ignore]` so `cargo check` and CI on a
    // toolkit-less build machine pass.

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn from_string_array_round_trip() {
        let input = StringArray::from(vec![Some("us"), None, Some("uk"), Some("us")]);
        let col = DictionaryColumnI64::from_string_array(&input)
            .expect("encode should succeed");
        let decoded = col.to_string_array().expect("decode should succeed");

        assert_eq!(decoded.len(), input.len());
        for i in 0..input.len() {
            assert_eq!(input.is_null(i), decoded.is_null(i));
            if !input.is_null(i) {
                assert_eq!(input.value(i), decoded.value(i));
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn try_into_i32_succeeds_for_small_dict() {
        let input = StringArray::from(vec![Some("us"), Some("uk"), Some("us")]);
        let wide = DictionaryColumnI64::from_string_array(&input).expect("encode");
        let narrow = wide.try_into_i32().expect("all indices fit in i32");
        let decoded = narrow.to_string_array().expect("decode narrow");

        assert_eq!(decoded.value(0), "us");
        assert_eq!(decoded.value(1), "uk");
        assert_eq!(decoded.value(2), "us");
    }

    #[test]
    #[ignore = "requires CUDA toolkit at runtime — narrow path uploads the oversized vec"]
    fn try_into_i32_fails_for_oversized_index() {
        // We can't realistically build a dictionary with > i32::MAX entries in
        // a unit test, so synthesize the failure by hand-uploading a single
        // oversized index. Marked `#[ignore]` because GpuVec::from_slice
        // requires a CUDA context.
        let oversized: i64 = i32::MAX as i64 + 1;
        let indices = GpuVec::<i64>::from_slice(&[oversized]).expect("upload");
        let col = DictionaryColumnI64 {
            dictionary: vec!["only".to_string()],
            indices,
            n_rows: 1,
        };
        // `expect_err` would require `DictionaryColumn: Debug`, which we
        // can't derive (its inner GpuVec doesn't impl Debug). Match instead.
        match col.try_into_i32() {
            Ok(_) => panic!("oversized index must reject narrowing"),
            Err(JavelinError::Other(msg)) => assert!(msg.contains("exceeds i32::MAX")),
            Err(other) => panic!("expected JavelinError::Other, got {:?}", other),
        }
    }
}
