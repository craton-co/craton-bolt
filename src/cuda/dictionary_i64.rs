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

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use arrow_array::{Array, StringArray};

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};

/// Hash a string with `DefaultHasher` for the construction-time lookup index.
///
/// Mirrors the i32 sibling's `hash_str` — kept private here to avoid an
/// inter-module dependency just for a one-liner. See the i32 module for the
/// rationale (host-side dedupe, collision-tolerant via tiebreak compare).
#[inline]
fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

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
    /// first-occurrence order. Returns `BoltError::Other` if the unique
    /// count would overflow `i64::MAX` (defensive — unreachable in practice).
    pub fn from_string_array(arr: &StringArray) -> BoltResult<Self> {
        let n_rows = arr.len();
        let mut dictionary: Vec<String> = Vec::new();
        // Construction-time dedupe by 64-bit digest. See the i32 sibling for
        // the full rationale: the previous `HashMap<String, i64>` design
        // allocated each distinct string twice (once as map key, once as vec
        // entry), wasting ~half the host memory used by the dictionary on
        // wide-cardinality columns. Keying by digest with a per-bucket
        // candidate list keeps the same dedupe cost without the second
        // allocation.
        let mut lookup: HashMap<u64, Vec<i64>> = HashMap::new();
        let mut indices: Vec<i64> = Vec::with_capacity(n_rows);

        for i in 0..n_rows {
            if arr.is_null(i) {
                indices.push(0);
                continue;
            }
            let s = arr.value(i);
            let digest = hash_str(s);
            // Tiebreak by candidate-string compare against the dictionary.
            // Buckets typically hold a single entry; the linear scan is over
            // collision candidates only.
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
                if (next_len as u128) > (i64::MAX as u128) {
                    return Err(BoltError::Other(format!(
                        "dictionary overflow: more than {} unique strings (i64 index space)",
                        i64::MAX
                    )));
                }
                let idx = next_len as i64;
                dictionary.push(s.to_string());
                lookup.entry(digest).or_default().push(idx);
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

    /// Batched variant of [`Self::index_of`].
    ///
    /// Mirrors [`crate::cuda::dictionary::DictionaryColumn::index_of_many`]:
    /// builds a temporary `HashMap` once and resolves every query against it,
    /// turning an `O(N * dict_len)` sequence of `index_of` calls into
    /// `O(dict_len + N)`. Returns `None` in any slot whose literal is not in
    /// the dictionary, matching the single-lookup convention. Useful for
    /// `IN`-list predicates or any path that wants several literal indices at
    /// once. The result is `Option<i64>` to match the underlying index width.
    pub fn index_of_many(&self, queries: &[&str]) -> Vec<Option<i64>> {
        // Lazy reverse map; callers on this path already know they have many
        // queries, so the up-front cost pays for itself.
        let lookup: HashMap<&str, i64> = self
            .dictionary
            .iter()
            .enumerate()
            // position is 0-based; real indices start at 1 (slot 0 = NULL).
            .map(|(i, s)| (s.as_str(), (i as i64) + 1))
            .collect();
        queries.iter().map(|q| lookup.get(*q).copied()).collect()
    }

    /// Download indices and reconstruct a `StringArray`.
    ///
    /// Index `0` becomes a SQL `NULL`. Negative indices or indices outside
    /// `1..=dictionary.len()` surface as `BoltError::Other` — that would
    /// indicate a kernel wrote something the host dictionary cannot decode.
    pub fn to_string_array(&self) -> BoltResult<StringArray> {
        let host_indices: Vec<i64> = self.indices.to_vec()?;
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
                // Width-safe decode: validate the 1-based offset against the
                // dictionary length in u64 before narrowing to usize. On a
                // 32-bit host a direct `as usize` cast would truncate before
                // the bounds check, letting `idx > u32::MAX` accidentally hit
                // a valid slot.
                let pos_u64 = (idx as u64) - 1;
                if pos_u64 >= self.dictionary.len() as u64 {
                    return Err(BoltError::Other(format!(
                        "dictionary decode: index {} out of bounds (dictionary size {})",
                        idx,
                        self.dictionary.len()
                    )));
                }
                let pos = pos_u64 as usize;
                let s = &self.dictionary[pos];
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
    pub fn from_i32(input: &crate::cuda::dictionary::DictionaryColumn) -> BoltResult<Self> {
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
    /// `BoltError::Other` on the first index that falls outside the i32
    /// range (including negatives — those were always invalid).
    pub fn try_into_i32(self) -> BoltResult<crate::cuda::dictionary::DictionaryColumn> {
        let wide: Vec<i64> = self.indices.to_vec()?;
        let mut narrow: Vec<i32> = Vec::with_capacity(wide.len());
        for &idx in &wide {
            if idx < 0 {
                return Err(BoltError::Other(format!(
                    "narrow to i32: negative index {} (invalid; NULL is 0)",
                    idx
                )));
            }
            if idx > i32::MAX as i64 {
                return Err(BoltError::Other(format!(
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
    ) -> BoltResult<Self> {
        Ok(Self {
            dictionary,
            indices: GpuVec::<i64>::empty(),
            n_rows,
        })
    }

    /// Host-only dedupe helper that mirrors the loop body of
    /// [`Self::from_string_array`] without the device upload. See the i32
    /// sibling's `dedupe_for_test` for rationale.
    #[cfg(test)]
    pub(crate) fn dedupe_for_test<'a, I>(
        rows: I,
    ) -> BoltResult<(Vec<String>, Vec<i64>)>
    where
        I: IntoIterator<Item = Option<&'a str>>,
    {
        let iter = rows.into_iter();
        let (lo, _hi) = iter.size_hint();
        let mut dictionary: Vec<String> = Vec::new();
        let mut lookup: HashMap<u64, Vec<i64>> = HashMap::new();
        let mut indices: Vec<i64> = Vec::with_capacity(lo);

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
                if (next_len as u128) > (i64::MAX as u128) {
                    return Err(BoltError::Other(format!(
                        "dictionary overflow: more than {} unique strings (i64 index space)",
                        i64::MAX
                    )));
                }
                let idx = next_len as i64;
                dictionary.push(s.to_string());
                lookup.entry(digest).or_default().push(idx);
                indices.push(idx);
            }
        }
        Ok((dictionary, indices))
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

    // ---- index_of_many symmetry with the i32 sibling ---------------------

    #[test]
    fn index_of_many_matches_single_lookup_semantics() {
        // Mirrors the i32 sibling's test of the same name: the batched
        // lookup must agree with N calls to `index_of`, including the
        // `None` slots for unknown literals. Guards against the common
        // refactor bug of returning 0 (NULL) for misses.
        let dict = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let col = DictionaryColumnI64::new_host_only(dict, 0)
            .expect("host-only constructor must not depend on CUDA");

        let got = col.index_of_many(&["a", "missing", "c", "b", ""]);
        assert_eq!(got, vec![Some(1), None, Some(3), Some(2), None]);
    }

    #[test]
    fn index_of_many_agrees_with_i32_sibling_on_parallel_fixture() {
        // Symmetry test: build the i32 and i64 wrappers from the *same*
        // dictionary contents, run the *same* queries through their
        // `index_of_many`, and assert the results agree slot-for-slot
        // (modulo the index integer width). This pins the contract that the
        // i64 sibling is a drop-in widening of the i32 path, not a
        // separately-evolving implementation.
        use crate::cuda::dictionary::DictionaryColumn;

        let dict = vec![
            "us".to_string(),
            "uk".to_string(),
            "fr".to_string(),
            "jp".to_string(),
        ];
        let i32_col = DictionaryColumn::new_host_only(dict.clone(), 0);
        let i64_col = DictionaryColumnI64::new_host_only(dict, 0)
            .expect("host-only constructor");

        let queries = ["jp", "missing", "us", "fr", "", "uk"];
        let got_i32 = i32_col.index_of_many(&queries);
        let got_i64 = i64_col.index_of_many(&queries);

        // Same length, same None/Some pattern, same numeric values (widened).
        assert_eq!(got_i32.len(), got_i64.len());
        for (q_idx, (a, b)) in got_i32.iter().zip(got_i64.iter()).enumerate() {
            match (a, b) {
                (Some(a32), Some(b64)) => assert_eq!(
                    *a32 as i64, *b64,
                    "query {q_idx} ('{}'): i32={} but i64={}",
                    queries[q_idx], a32, b64
                ),
                (None, None) => {}
                _ => panic!(
                    "query {q_idx} ('{}'): None/Some mismatch i32={:?} i64={:?}",
                    queries[q_idx], a, b
                ),
            }
        }
    }

    #[test]
    fn index_of_many_on_empty_dictionary_is_all_none() {
        let col = DictionaryColumnI64::new_host_only(Vec::new(), 0)
            .expect("host-only constructor");
        let got = col.index_of_many(&["x", "y"]);
        assert_eq!(got, vec![None, None]);
    }

    #[test]
    fn index_of_many_with_empty_query_is_empty() {
        let dict = vec!["a".to_string()];
        let col = DictionaryColumnI64::new_host_only(dict, 0)
            .expect("host-only constructor");
        let got = col.index_of_many(&[]);
        assert!(got.is_empty());
    }

    // ---- Construction-time dedupe ----------------------------------------

    #[test]
    fn dedupe_large_redundant_input_yields_only_distinct_strings() {
        // Same memory-fix regression as the i32 sibling: 1M rows over 100
        // distinct strings, asserting the digest dedupe collapses to the
        // expected cardinality with the right per-row indices.
        const ROWS: usize = 1_000_000;
        const DISTINCT: usize = 100;

        let pool: Vec<String> = (0..DISTINCT).map(|i| format!("v_{i}")).collect();
        let rows = (0..ROWS).map(|r| Some(pool[r % DISTINCT].as_str()));

        let (dictionary, indices) =
            DictionaryColumnI64::dedupe_for_test(rows).expect("dedupe");

        assert_eq!(dictionary.len(), DISTINCT);
        for i in 0..DISTINCT {
            assert_eq!(dictionary[i], pool[i]);
        }
        assert_eq!(indices.len(), ROWS);
        for (r, &idx) in indices.iter().enumerate() {
            let expected = ((r % DISTINCT) as i64) + 1;
            assert_eq!(idx, expected);
        }

        let col = DictionaryColumnI64::new_host_only(dictionary, ROWS)
            .expect("host-only constructor");
        for (i, s) in pool.iter().enumerate() {
            assert_eq!(col.index_of(s), Some((i as i64) + 1));
        }
        assert_eq!(col.index_of("missing-literal"), None);
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
            Err(BoltError::Other(msg)) => assert!(msg.contains("exceeds i32::MAX")),
            Err(other) => panic!("expected BoltError::Other, got {:?}", other),
        }
    }
}
