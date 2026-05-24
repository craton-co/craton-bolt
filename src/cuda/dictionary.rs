// SPDX-License-Identifier: Apache-2.0

//! Host-side string dictionary paired with on-device i32 indices.
//!
//! Variable-width strings are a poor fit for a fused-codegen GPU kernel: a
//! dynamic offset/byte layout forces every comparison to dereference, which
//! defeats coalesced loads. Instead we dictionary-encode strings on the host
//! and ship only fixed-width 32-bit indices to the device. Predicates like
//! `region = 'US'` reduce to integer equality against the index of `'US'` —
//! work the codegen path already does well.
//!
//! Layout convention:
//!   * Index `0` is reserved for SQL `NULL`. It NEVER appears in
//!     `dictionary[]` and is never returned by `index_of`.
//!   * Real strings start at index `1`. The i-th unique non-null string is
//!     stored at `dictionary[i - 1]`.
//!   * Indices are `i32`. Allowing > `i32::MAX` distinct strings would break
//!     downstream codegen; we surface that as a `JavelinError::Other`.

use std::collections::HashMap;

use arrow_array::{Array, StringArray};

use crate::cuda::GpuVec;
use crate::error::{JavelinError, JavelinResult};

/// On-host string dictionary + on-device i32 indices.
///
/// Strings are encoded as `i32` indices on the GPU; the host holds the
/// bidirectional mapping for decoding. Index `0` is reserved for NULL; real
/// strings occupy indices `1..=dictionary.len()`. The implementation never
/// emits a negative index.
pub struct DictionaryColumn {
    /// Host-side dictionary: position `i` → the `(i + 1)`-th index's string.
    /// Equivalently, `dictionary[k - 1]` is the string for GPU index `k`.
    pub dictionary: Vec<String>,
    /// GPU-side indices: one `i32` per source row. `0` means NULL.
    pub indices: GpuVec<i32>,
    /// Number of source rows.
    pub n_rows: usize,
}

impl DictionaryColumn {
    /// Encode an Arrow `StringArray` as a dictionary and upload the indices.
    ///
    /// Nulls in `arr` map to index `0`. Distinct non-null strings are
    /// deduplicated and assigned sequential indices starting at `1`, in
    /// first-occurrence order.
    pub fn from_string_array(arr: &StringArray) -> JavelinResult<Self> {
        let n_rows = arr.len();
        let mut dictionary: Vec<String> = Vec::new();
        let mut lookup: HashMap<String, i32> = HashMap::new();
        let mut indices: Vec<i32> = Vec::with_capacity(n_rows);

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
                if next_len > i32::MAX as usize {
                    return Err(JavelinError::Other(format!(
                        "dictionary overflow: more than {} unique strings (i32 index space)",
                        i32::MAX
                    )));
                }
                let idx = next_len as i32;
                let owned = s.to_string();
                dictionary.push(owned.clone());
                lookup.insert(owned, idx);
                indices.push(idx);
            }
        }

        let device_indices = GpuVec::<i32>::from_slice(&indices)?;
        Ok(Self {
            dictionary,
            indices: device_indices,
            n_rows,
        })
    }

    /// Lookup the index of a literal string in the dictionary.
    ///
    /// Returns `Some(index)` if `s` was seen during construction, or `None`
    /// otherwise. A `None` here is not an error: a predicate against an
    /// unknown literal trivially matches no rows.
    pub fn index_of(&self, s: &str) -> Option<i32> {
        // Linear scan keeps `index_of` O(dict) but avoids carrying the
        // construction-time HashMap. Literal lookups happen once per query, so
        // the asymptotic cost is dominated by row count, not dictionary size.
        self.dictionary
            .iter()
            .position(|d| d == s)
            // position is 0-based; real indices start at 1.
            .map(|p| (p as i32) + 1)
    }

    /// Test-only constructor that bypasses any GPU upload.
    ///
    /// Mirrors `DictionaryColumnI64::new_host_only`. The `indices` field is
    /// initialized to an empty `GpuVec` placeholder; callers may only
    /// exercise the host-side `dictionary` field (e.g. via
    /// [`Self::index_of`]). Any method that touches the device buffer
    /// ([`Self::to_string_array`]) will operate on an empty index vector.
    ///
    /// This exists so host-only unit tests can construct a populated
    /// dictionary without requiring a CUDA-enabled machine. Production code
    /// must not use this — use [`Self::from_string_array`] instead.
    #[cfg(test)]
    pub(crate) fn new_host_only(dictionary: Vec<String>, n_rows: usize) -> Self {
        Self {
            dictionary,
            indices: GpuVec::<i32>::empty(),
            n_rows,
        }
    }

    /// Download indices and reconstruct a `StringArray`.
    ///
    /// Index `0` becomes a SQL `NULL`. Indices outside `1..=dictionary.len()`
    /// surface as `JavelinError::Other` — that would indicate a kernel wrote
    /// something the host dictionary cannot decode.
    pub fn to_string_array(&self) -> JavelinResult<StringArray> {
        let host_indices: Vec<i32> = self.indices.to_vec()?;
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
                let pos = (idx as usize) - 1;
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
}

#[cfg(test)]
mod tests {
    //! Tests for the i32-indexed string dictionary.
    //!
    //! The host-only tests use [`DictionaryColumn::new_host_only`] so they
    //! pass on machines without a CUDA toolkit (including docs.rs). The
    //! GPU-touching tests (anything that calls [`DictionaryColumn::from_string_array`]
    //! or [`DictionaryColumn::to_string_array`]) are marked `#[ignore]` and
    //! only run when explicitly requested with `--ignored`.
    //!
    //! Layout reminder: slot 0 is reserved for NULL, real strings start at
    //! index 1. The i64 sibling uses the same convention.
    use super::*;

    // ---- Host-only: index_of + new_host_only -----------------------------

    #[test]
    fn index_of_returns_one_based_position_for_known_strings() {
        // Dictionary is `["a", "b", "c"]`; slot 0 is NULL, so the strings
        // live at indices 1, 2, 3. `index_of` must reflect that.
        let dict = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let col = DictionaryColumn::new_host_only(dict, 0);

        assert_eq!(col.index_of("a"), Some(1));
        assert_eq!(col.index_of("b"), Some(2));
        assert_eq!(col.index_of("c"), Some(3));
    }

    #[test]
    fn index_of_returns_none_for_unknown_string() {
        // Predicates against literals that never appeared in the column
        // must return `None` — not zero (which would mean NULL). The docs
        // are explicit: "a predicate against an unknown literal trivially
        // matches no rows".
        let dict = vec!["us".to_string(), "uk".to_string()];
        let col = DictionaryColumn::new_host_only(dict, 0);

        assert_eq!(col.index_of("fr"), None);
        // Empty string isn't in the dictionary either; must also be None.
        assert_eq!(col.index_of(""), None);
    }

    #[test]
    fn index_of_on_empty_dictionary_is_none() {
        let col = DictionaryColumn::new_host_only(Vec::new(), 0);
        assert_eq!(col.index_of("anything"), None);
    }

    #[test]
    fn host_only_constructor_preserves_dictionary_and_row_count() {
        // Sanity-check the test helper itself: the dictionary and n_rows
        // round-trip verbatim, and `indices` is the zero-length placeholder.
        let dict = vec!["alpha".to_string(), "beta".to_string()];
        let col = DictionaryColumn::new_host_only(dict.clone(), 5);

        assert_eq!(col.dictionary, dict);
        assert_eq!(col.n_rows, 5);
        // The placeholder indices vec must have zero length on the host
        // side — it's a stand-in, not real data.
        assert_eq!(col.indices.len(), 0);
    }

    // ---- GPU-required tests ----------------------------------------------
    //
    // These call `from_string_array` (which uploads to the device) or
    // `to_string_array` (which downloads). They cannot run on a host without
    // a CUDA toolkit, so they are `#[ignore]`d the same way the i64 sibling
    // tests are.

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn dict_basic_encoding() {
        // ["a", "b", "a", "c", "b"] => dictionary ["a", "b", "c"], indices
        // [1, 2, 1, 3, 2]. First-occurrence order, slot 0 reserved for NULL.
        let input = StringArray::from(vec!["a", "b", "a", "c", "b"]);
        let col = DictionaryColumn::from_string_array(&input).expect("encode");

        assert_eq!(col.n_rows, 5);
        assert_eq!(
            col.dictionary,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        let indices = col.indices.to_vec().expect("download indices");
        assert_eq!(indices, vec![1, 2, 1, 3, 2]);
    }

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn dict_with_nulls() {
        // Nulls collapse to index 0 and never enter the dictionary.
        let input = StringArray::from(vec![Some("a"), None, Some("b"), None, Some("a")]);
        let col = DictionaryColumn::from_string_array(&input).expect("encode");

        assert_eq!(col.n_rows, 5);
        assert_eq!(col.dictionary, vec!["a".to_string(), "b".to_string()]);
        let indices = col.indices.to_vec().expect("download indices");
        assert_eq!(indices, vec![1, 0, 2, 0, 1]);
    }

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn dict_empty_input() {
        // Edge case: zero rows. Dictionary and indices must both be empty,
        // and n_rows must agree.
        let input = StringArray::from(Vec::<&str>::new());
        let col = DictionaryColumn::from_string_array(&input).expect("encode");

        assert_eq!(col.n_rows, 0);
        assert!(col.dictionary.is_empty());
        let indices = col.indices.to_vec().expect("download indices");
        assert!(indices.is_empty());
    }

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn dict_all_null() {
        // Every row is NULL: the dictionary stays empty (no non-null
        // strings to deduplicate), and every index is 0.
        let input = StringArray::from(vec![None::<&str>, None, None, None]);
        let col = DictionaryColumn::from_string_array(&input).expect("encode");

        assert_eq!(col.n_rows, 4);
        assert!(col.dictionary.is_empty());
        let indices = col.indices.to_vec().expect("download indices");
        assert_eq!(indices, vec![0, 0, 0, 0]);
    }

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn dict_index_of_lookup() {
        // After a real encode, `index_of` must report the same slots that
        // the indices vec already uses for those strings.
        let input = StringArray::from(vec!["red", "green", "blue", "green"]);
        let col = DictionaryColumn::from_string_array(&input).expect("encode");

        assert_eq!(col.index_of("red"), Some(1));
        assert_eq!(col.index_of("green"), Some(2));
        assert_eq!(col.index_of("blue"), Some(3));
        // Literal never seen during construction => None, not 0.
        assert_eq!(col.index_of("yellow"), None);
    }

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn dict_to_string_array_roundtrip() {
        // encode -> decode -> assert byte-equality (with NULL preservation).
        let input = StringArray::from(vec![
            Some("us"),
            None,
            Some("uk"),
            Some("us"),
            None,
            Some("fr"),
        ]);
        let col = DictionaryColumn::from_string_array(&input).expect("encode");
        let decoded = col.to_string_array().expect("decode");

        assert_eq!(decoded.len(), input.len());
        for i in 0..input.len() {
            assert_eq!(
                input.is_null(i),
                decoded.is_null(i),
                "null bit mismatch at row {}",
                i
            );
            if !input.is_null(i) {
                assert_eq!(
                    input.value(i),
                    decoded.value(i),
                    "value mismatch at row {}",
                    i
                );
            }
        }
    }
}
