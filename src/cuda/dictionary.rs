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
//!     downstream codegen; we surface that as a `BoltError::Other`.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use arrow_array::{Array, StringArray};

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};

/// Hash a string with `DefaultHasher` for the construction-time lookup index.
///
/// Used only inside [`DictionaryColumn::from_string_array`] (and the i64
/// sibling) to dedupe strings without paying an extra owned `String` per
/// distinct entry. The full string still lives once in `dictionary[]`;
/// collisions are resolved by an explicit equality check on the candidate
/// (see the lookup logic). `DefaultHasher` (SipHash-1-3) is fine here — this
/// is a host-side dedupe path, not a security boundary, and collisions are
/// astronomically rare on real data.
#[inline]
fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

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
    pub fn from_string_array(arr: &StringArray) -> BoltResult<Self> {
        let n_rows = arr.len();
        let mut dictionary: Vec<String> = Vec::new();
        // Construction-time dedupe index. Keying the map on a 64-bit string
        // digest (rather than an owned `String`) means each distinct string is
        // allocated exactly once — in `dictionary[]`. The bucket value is a
        // `Vec<i32>` of candidate dictionary indices (1-based, matching the
        // GPU encoding) that hashed to the same digest; on lookup we tiebreak
        // by comparing the candidate strings via the dictionary itself. In
        // the common case the bucket holds a single entry, so per-row work is
        // one hash + one compare. SipHash collisions on real text are
        // astronomically rare; the tiebreak is defensive, not hot.
        //
        // The previous implementation kept a `HashMap<String, i32>` alongside
        // the `Vec<String>` — that double-allocated each distinct string
        // (once as the map key, once as the vec entry). For a 100M-row column
        // with millions of distinct strings, that wasted ~half the host
        // memory used by the dictionary. The digest map keeps the same
        // amortised dedupe cost without the second `String` per distinct
        // value.
        let mut lookup: HashMap<u64, Vec<i32>> = HashMap::new();
        let mut indices: Vec<i32> = Vec::with_capacity(n_rows);

        for i in 0..n_rows {
            if arr.is_null(i) {
                indices.push(0);
                continue;
            }
            let s = arr.value(i);
            let digest = hash_str(s);
            // Probe the digest bucket. The bucket's i32s are 1-based
            // dictionary indices; `dictionary[idx - 1]` is the candidate
            // string. Iterate every candidate (typically one) and accept the
            // first byte-equal match.
            let existing = lookup.get(&digest).and_then(|bucket| {
                bucket
                    .iter()
                    .find(|&&idx| dictionary[(idx as usize) - 1] == s)
                    .copied()
            });
            if let Some(idx) = existing {
                indices.push(idx);
            } else {
                // Next index = current dictionary length + 1 (slot 0 reserved for NULL).
                let next_len = dictionary.len().checked_add(1).ok_or_else(|| {
                    BoltError::Other(
                        "dictionary overflow: more than usize::MAX unique strings".into(),
                    )
                })?;
                if next_len > i32::MAX as usize {
                    return Err(BoltError::Other(format!(
                        "dictionary overflow: more than {} unique strings (i32 index space)",
                        i32::MAX
                    )));
                }
                let idx = next_len as i32;
                // Single owned-string allocation: the dictionary takes the
                // only copy. The lookup map gets just the digest -> index
                // mapping.
                dictionary.push(s.to_string());
                lookup.entry(digest).or_default().push(idx);
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
        // For multi-literal predicates (e.g. `IN ('a', 'b', 'c', ...)`),
        // prefer [`Self::index_of_many`] which amortizes the scan cost by
        // building the lookup map once.
        self.dictionary
            .iter()
            .position(|d| d == s)
            // position is 0-based; real indices start at 1.
            .map(|p| (p as i32) + 1)
    }

    /// Batched variant of [`Self::index_of`].
    ///
    /// Builds a temporary `HashMap` once and resolves every query against it,
    /// turning an `O(N * dict_len)` sequence of `index_of` calls into
    /// `O(dict_len + N)`. Returns `None` in any slot whose literal is not in
    /// the dictionary, matching the single-lookup convention. Useful for
    /// `IN`-list predicates or any path that wants several literal indices
    /// at once.
    pub fn index_of_many(&self, queries: &[&str]) -> Vec<Option<i32>> {
        // Build the reverse map lazily — callers that hit this path already
        // know they have many queries, so the up-front cost pays for itself.
        let lookup: HashMap<&str, i32> = self
            .dictionary
            .iter()
            .enumerate()
            // position is 0-based; real indices start at 1 (slot 0 = NULL).
            .map(|(i, s)| (s.as_str(), (i as i32) + 1))
            .collect();
        queries.iter().map(|q| lookup.get(*q).copied()).collect()
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

    /// Host-only dedupe helper that mirrors the loop body of
    /// [`Self::from_string_array`] without the device upload.
    ///
    /// Returns `(dictionary, indices)` exactly as the real path would
    /// produce them, so a test can assert dedupe behaviour on a multi-million
    /// row input without needing a CUDA toolkit. Production code must not use
    /// this — use [`Self::from_string_array`].
    #[cfg(test)]
    pub(crate) fn dedupe_for_test<'a, I>(
        rows: I,
    ) -> BoltResult<(Vec<String>, Vec<i32>)>
    where
        I: IntoIterator<Item = Option<&'a str>>,
    {
        let iter = rows.into_iter();
        let (lo, _hi) = iter.size_hint();
        let mut dictionary: Vec<String> = Vec::new();
        let mut lookup: HashMap<u64, Vec<i32>> = HashMap::new();
        let mut indices: Vec<i32> = Vec::with_capacity(lo);

        for row in iter {
            let Some(s) = row else {
                indices.push(0);
                continue;
            };
            let digest = hash_str(s);
            let existing = lookup.get(&digest).and_then(|bucket| {
                bucket
                    .iter()
                    .find(|&&idx| dictionary[(idx as usize) - 1] == s)
                    .copied()
            });
            if let Some(idx) = existing {
                indices.push(idx);
            } else {
                let next_len = dictionary.len().checked_add(1).ok_or_else(|| {
                    BoltError::Other(
                        "dictionary overflow: more than usize::MAX unique strings".into(),
                    )
                })?;
                if next_len > i32::MAX as usize {
                    return Err(BoltError::Other(format!(
                        "dictionary overflow: more than {} unique strings (i32 index space)",
                        i32::MAX
                    )));
                }
                let idx = next_len as i32;
                dictionary.push(s.to_string());
                lookup.entry(digest).or_default().push(idx);
                indices.push(idx);
            }
        }
        Ok((dictionary, indices))
    }

    /// Download indices and reconstruct a `StringArray`.
    ///
    /// Index `0` becomes a SQL `NULL`. Indices outside `1..=dictionary.len()`
    /// surface as `BoltError::Other` — that would indicate a kernel wrote
    /// something the host dictionary cannot decode.
    pub fn to_string_array(&self) -> BoltResult<StringArray> {
        let host_indices: Vec<i32> = self.indices.to_vec()?;
        let mut out: Vec<Option<&str>> = Vec::with_capacity(host_indices.len());
        for &idx in &host_indices {
            if idx == 0 {
                out.push(None);
            } else if idx < 0 {
                return Err(BoltError::Other(format!(
                    "dictionary decode: negative index {} (NULL is encoded as 0)",
                    idx
                )));
            } else {
                let pos = (idx as usize) - 1;
                let s = self.dictionary.get(pos).ok_or_else(|| {
                    BoltError::Other(format!(
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
    fn index_of_many_matches_single_lookup_semantics() {
        // The batched lookup must agree with N calls to `index_of`, including
        // the `None` slots for unknown literals. This guards against the
        // common refactor bug of returning 0 (NULL) for misses.
        let dict = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let col = DictionaryColumn::new_host_only(dict, 0);

        let got = col.index_of_many(&["a", "missing", "c", "b", ""]);
        assert_eq!(got, vec![Some(1), None, Some(3), Some(2), None]);
    }

    #[test]
    fn index_of_many_on_empty_dictionary_is_all_none() {
        let col = DictionaryColumn::new_host_only(Vec::new(), 0);
        let got = col.index_of_many(&["x", "y"]);
        assert_eq!(got, vec![None, None]);
    }

    #[test]
    fn index_of_many_with_empty_query_is_empty() {
        let dict = vec!["a".to_string()];
        let col = DictionaryColumn::new_host_only(dict, 0);
        let got = col.index_of_many(&[]);
        assert!(got.is_empty());
    }

    // ---- Construction-time dedupe ----------------------------------------

    #[test]
    fn dedupe_large_redundant_input_yields_only_distinct_strings() {
        // High-redundancy regression: 1M rows over 100 distinct strings.
        // Verifies the digest-keyed dedupe map collapses the input to the
        // expected distinct count and that each row's emitted index is
        // consistent with `index_of` on the resulting dictionary. This is
        // also the load-bearing test for the memory-allocation fix: the
        // previous `HashMap<String, i32>` implementation would have
        // allocated 100 redundant `String`s for the map keys, on top of the
        // 100 `String`s in the dictionary vec. The new digest map allocates
        // each distinct string exactly once.
        const ROWS: usize = 1_000_000;
        const DISTINCT: usize = 100;

        // Pre-materialise the 100 distinct strings so we can borrow them as
        // `&str` for the dedupe iterator without re-allocating per row.
        let pool: Vec<String> = (0..DISTINCT).map(|i| format!("val_{i}")).collect();
        let rows = (0..ROWS).map(|r| Some(pool[r % DISTINCT].as_str()));

        let (dictionary, indices) =
            DictionaryColumn::dedupe_for_test(rows).expect("dedupe");

        // Distinct count must equal the input cardinality, in first-
        // occurrence order.
        assert_eq!(dictionary.len(), DISTINCT);
        for i in 0..DISTINCT {
            assert_eq!(dictionary[i], pool[i]);
        }
        // Every row must have a positive index (slot 0 reserved for NULL).
        assert_eq!(indices.len(), ROWS);
        for (r, &idx) in indices.iter().enumerate() {
            let expected = ((r % DISTINCT) as i32) + 1;
            assert_eq!(
                idx, expected,
                "row {r} expected index {expected}, got {idx}"
            );
        }
        // `index_of` on the resulting dictionary must agree with the emitted
        // indices for every distinct string.
        let col = DictionaryColumn::new_host_only(dictionary, ROWS);
        for (i, s) in pool.iter().enumerate() {
            assert_eq!(col.index_of(s), Some((i as i32) + 1));
        }
        // Sanity: a string never seen during construction returns None.
        assert_eq!(col.index_of("missing-literal"), None);
    }

    #[test]
    fn dedupe_handles_interleaved_nulls() {
        // Nulls go to index 0 and do not enter the dictionary. Verifies the
        // digest-map dedupe doesn't accidentally treat None as a value.
        let rows = vec![
            Some("a"),
            None,
            Some("b"),
            None,
            Some("a"),
            Some("c"),
            None,
        ];
        let (dictionary, indices) =
            DictionaryColumn::dedupe_for_test(rows).expect("dedupe");

        assert_eq!(
            dictionary,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(indices, vec![1, 0, 2, 0, 1, 3, 0]);
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
