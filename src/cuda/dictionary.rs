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

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use arrow_array::{Array, StringArray};

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};

// 32-bit host pointer-width note.
//
// `to_string_array` below performs `(idx as usize) - 1` after the `idx <= ...`
// upper-bound check. On a 64-bit `usize` this is unambiguously lossless. On a
// 32-bit `usize`, the cast is still lossless *because* we validate
// `idx <= i32::MAX` (and reject `idx < 0`) before the cast, so the value fits
// in `u32` and therefore in a 32-bit `usize`. The crate does not advertise
// 32-bit support, but this const-context note flags the dependency between
// the i32 bound and the pointer-width assumption: if the bound is ever
// raised, this site needs revisiting before 32-bit builds remain safe.
#[cfg(target_pointer_width = "32")]
const _: () = {
    // The dictionary code does `(idx as usize) - 1` after validating
    // `idx <= i32::MAX`, which is lossless on 32-bit usize. Keep this
    // const-context note so future bumps of the bound are flagged.
};

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

/// Re-order a first-seen `(dictionary, indices)` pair so the dictionary is in
/// byte-lexicographic order and every index points at its value's new
/// (lex-ranked) slot (R1, lex-rank dictionaries at ingest).
///
/// On entry `dictionary[p]` is the string for first-seen GPU index `p + 1` and
/// `indices[r]` is the 1-based GPU index of row `r` (or `0` for NULL). On exit
/// the dictionary is sorted byte-lexicographically (`str`'s `Ord`, i.e. raw
/// UTF-8 bytes — NOT locale-aware, consistent with [`collation_ranks_of`]) and
/// each non-NULL index is rewritten to `lex_rank + 1`, so the integer code
/// order matches lexicographic string order. NULL indices (`0`) are untouched.
///
/// This is the source-of-truth lex ordering: once codes equal lex rank, every
/// downstream consumer that compares dict indices (ORDER BY, range predicates,
/// dict sorts/joins/group-bys) gets the right order for free, and
/// [`collation_ranks_of`] becomes the identity permutation.
///
/// Decode-correctness is preserved: the remap is a bijection on the non-NULL
/// codes, so decoding `lex_rank + 1 -> dictionary[lex_rank]` yields the same
/// string the row originally encoded.
///
/// Pure host function. Cost: `O(N log N)` to sort the distinct values plus
/// `O(R)` to rewrite the row indices; intended for register-table time, not a
/// per-row hot path. Generic over the index integer type so the i32 and i64
/// builders share one implementation.
pub(crate) fn lex_sort_dictionary<I>(dictionary: &mut Vec<String>, indices: &mut [I])
where
    I: Copy + TryFrom<usize> + TryInto<usize> + PartialEq + Default,
{
    let n = dictionary.len();
    if n == 0 {
        return;
    }
    // `order[new_slot]` is the OLD 0-based slot whose string sorts into
    // `new_slot`. Sorting slot ids by their string value gives this directly.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| dictionary[a].cmp(&dictionary[b]));

    // `remap[old_slot] = new GPU index (1-based)` for every old slot. Built by
    // inverting `order`: the value at new slot `r` came from old slot
    // `order[r]`, so old slot `order[r]` now lives at GPU index `r + 1`.
    let mut remap: Vec<usize> = vec![0; n];
    for (new_slot, &old_slot) in order.iter().enumerate() {
        remap[old_slot] = new_slot + 1;
    }

    // Rewrite the dictionary into sorted order.
    let mut sorted: Vec<String> = Vec::with_capacity(n);
    for &old_slot in &order {
        // Move each owned String into its sorted position via take + replace
        // to avoid cloning. `std::mem::take` leaves an empty placeholder we
        // never read again (the old vector is dropped below).
        sorted.push(std::mem::take(&mut dictionary[old_slot]));
    }
    *dictionary = sorted;

    // Rewrite every non-NULL row index to its value's new GPU index. NULL (0)
    // stays 0. The `TryInto`/`TryFrom` round-trip is infallible here: every
    // non-zero index was a valid 1-based slot (`1..=n`), and `remap[..]` values
    // are also in `1..=n`, both of which fit the original index width.
    let zero = I::default();
    for slot in indices.iter_mut() {
        if *slot == zero {
            continue;
        }
        let old_idx: usize = (*slot)
            .try_into()
            .unwrap_or_else(|_| unreachable!("dictionary index must fit usize"));
        let new_idx = remap[old_idx - 1];
        *slot = I::try_from(new_idx)
            .unwrap_or_else(|_| unreachable!("remapped index must fit original width"));
    }
}

/// Compute the byte-lexicographic collation rank permutation of a dictionary
/// slice (finding F10).
///
/// `dictionary[p]` is the string at insertion slot `p` (GPU index `p + 1`).
/// The returned `ranks[p]` is the 0-based position of that string in the
/// byte-sorted order of all distinct dictionary entries. Ordering is binary
/// (raw UTF-8 bytes via `str`'s `Ord`), NOT locale-aware — see
/// [`DictionaryColumn::collation_ranks`].
///
/// Shared by the i32 and i64 dictionary variants (and the `Any` wrapper) so
/// the collation definition lives in exactly one place.
pub(crate) fn collation_ranks_of(dictionary: &[String]) -> Vec<usize> {
    // Sort the insertion slots by their string value; the position of slot `p`
    // within that sorted order is its rank.
    let mut order: Vec<usize> = (0..dictionary.len()).collect();
    order.sort_by(|&a, &b| dictionary[a].cmp(&dictionary[b]));
    let mut ranks = vec![0usize; dictionary.len()];
    for (rank, &slot) in order.iter().enumerate() {
        ranks[slot] = rank;
    }
    ranks
}

/// GPU indices (1-based) of every entry in `dictionary` satisfying
/// `entry OP probe` under byte-lexicographic collation (finding F10).
///
/// `op` must be one of the four ordering comparisons; any other op returns an
/// empty vector (the rewriter only ever calls this for ordering ops). Slot 0
/// (NULL) is never represented in `dictionary` and so is never returned. The
/// result is in ascending GPU-index order. Shared by both dictionary variants.
pub(crate) fn indices_satisfying_in(
    dictionary: &[String],
    op: crate::plan::logical_plan::BinaryOp,
    probe: &str,
) -> Vec<usize> {
    use crate::plan::logical_plan::BinaryOp;
    let keep = |s: &str| -> bool {
        match op {
            BinaryOp::Lt => s < probe,
            BinaryOp::LtEq => s <= probe,
            BinaryOp::Gt => s > probe,
            BinaryOp::GtEq => s >= probe,
            // Non-ordering ops are not this helper's responsibility.
            _ => false,
        }
    };
    dictionary
        .iter()
        .enumerate()
        // GPU index of insertion slot `p` is `p + 1` (slot 0 = NULL).
        .filter(|(_, s)| keep(s.as_str()))
        .map(|(p, _)| p + 1)
        .collect()
}

/// Sentinel rank assigned to the NULL slot (GPU index 0) in a per-row rank
/// column (finding F12, column-vs-column Utf8 ordering).
///
/// A NULL string has no position in any byte-collation ordering — SQL compares
/// it as NULL, never satisfying an ordering predicate. When a rank column is
/// materialised so two dict-encoded Utf8 columns can be compared as integer
/// ranks (`rank(a) OP rank(b)`), slot 0 must therefore map to a value that the
/// integer comparison machinery treats as "this row never passes". `-1` is that
/// value: every real rank is `>= 0` (see [`rank_for_index_in`]), so a `-1` on
/// either side makes the comparison's outcome irrelevant to correctness *as
/// long as the executor applies SQL 3VL* (a row with a NULL operand is dropped
/// by the filter, exactly as it would be for any other NULL comparison).
///
/// IMPORTANT (deferred-exec contract): the `-1` sentinel alone does NOT encode
/// 3VL. A plain `rank_a < rank_b` integer compare with `rank_a = -1` would, for
/// example, report `true` for `-1 < 0`, which is wrong (NULL `<` anything must
/// be NULL/false). The executor hook that materialises these rank columns MUST
/// also propagate the source columns' validity (NULL when *either* index is 0)
/// onto the comparison's output, so a NULL-operand row is excluded. See the
/// exec-hook note in [`crate::plan::string_literal_rewrite`]. The sentinel
/// exists so the rank buffer is total (defined for every GPU index, including
/// slot 0) and so a debugging/host oracle can detect the NULL rows.
pub(crate) const NULL_RANK_SENTINEL: i64 = -1;

/// Build a per-GPU-index rank lookup for `dictionary` against a shared,
/// byte-sorted universe of strings (finding F12).
///
/// `universe` must be the byte-lexicographically sorted, de-duplicated set of
/// strings that defines the common ordering space — typically the sorted union
/// of two columns' dictionaries (see [`unified_rank_maps_of`]). The returned
/// vector `out` has length `dictionary.len() + 1` and is indexed by **GPU
/// index** (the same 1-based convention used everywhere in this module):
///   * `out[0]` is [`NULL_RANK_SENTINEL`] — the NULL slot has no rank.
///   * `out[k]` (for `k` in `1..=dictionary.len()`) is the 0-based position of
///     `dictionary[k - 1]` within `universe`, i.e. its rank in the shared
///     ordering.
///
/// Because both columns' ranks are computed against the *same* `universe`,
/// `rank_a(i) OP rank_b(j)` is order-equivalent to `string_a(i) OP string_b(j)`
/// under byte collation — which is exactly what column-vs-column Utf8 ordering
/// requires, even when the two columns have entirely different dictionaries.
///
/// Every entry of `dictionary` is guaranteed to appear in `universe` when the
/// universe was built as a superset (the union); a defensive `binary_search`
/// miss falls back to the half-open insertion point, which still yields a
/// consistent total order for that string relative to the universe.
///
/// This is **binary collation** (raw UTF-8 bytes via `str`'s `Ord`), NOT
/// locale-aware / ICU collation — consistent with [`collation_ranks_of`].
pub(crate) fn rank_for_index_in(dictionary: &[String], universe: &[String]) -> Vec<i64> {
    let mut out = Vec::with_capacity(dictionary.len() + 1);
    // Slot 0 = NULL: no rank.
    out.push(NULL_RANK_SENTINEL);
    for s in dictionary {
        // `universe` is sorted + de-duped, so its byte-order position is the
        // rank. `binary_search` returns `Ok(pos)` for a present string;
        // `Err(insert_pos)` (defensive) still gives a consistent position.
        let rank = match universe.binary_search(s) {
            Ok(pos) => pos,
            Err(pos) => pos,
        };
        out.push(rank as i64);
    }
    out
}

/// Build the shared, byte-sorted, de-duplicated universe of two dictionaries
/// and the per-GPU-index rank lookup for each (finding F12, column-vs-column
/// Utf8 ordering).
///
/// Returns `(rank_a, rank_b)` where `rank_a` is the rank lookup for
/// `dict_a` and `rank_b` for `dict_b`, both computed against the *same* sorted
/// union of the two dictionaries (see [`rank_for_index_in`] for the per-column
/// layout, including the slot-0 NULL sentinel).
///
/// This is the cross-dictionary-correct primitive: `dict_a` and `dict_b` may be
/// completely different dictionaries (different strings, different insertion
/// order, even different lengths). Comparing each column's *own* collation rank
/// would be WRONG — rank 2 in one dictionary and rank 2 in another name
/// unrelated strings. Building one common universe and ranking both columns
/// against it makes `rank_a(i) OP rank_b(j)` exactly reproduce
/// `string_a(i) OP string_b(j)` under byte collation. When the two dictionaries
/// are identical the union collapses to a single copy and the two rank lookups
/// are identical, which is the degenerate same-dictionary case.
///
/// Cost: `O((Na + Nb) log(Na + Nb))` to sort the union, then `O(Na log U +
/// Nb log U)` for the two rank lookups. Intended for query-plan time, not a
/// per-row hot path.
pub(crate) fn unified_rank_maps_of(dict_a: &[String], dict_b: &[String]) -> (Vec<i64>, Vec<i64>) {
    // Sorted, de-duplicated union under byte collation.
    let mut universe: Vec<String> = Vec::with_capacity(dict_a.len() + dict_b.len());
    universe.extend(dict_a.iter().cloned());
    universe.extend(dict_b.iter().cloned());
    universe.sort();
    universe.dedup();

    let rank_a = rank_for_index_in(dict_a, &universe);
    let rank_b = rank_for_index_in(dict_b, &universe);
    (rank_a, rank_b)
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
    /// **byte-lexicographic order** of the string values (R1): code `k`
    /// (GPU index `k`) names the `k`-th smallest distinct string under `str`'s
    /// `Ord` (raw UTF-8 bytes, NOT locale-aware). The integer code order
    /// therefore matches lexicographic string order, so downstream dict-index
    /// ORDER BY / range predicates / sorts / joins / group-bys are correct
    /// without a decode. Decode still yields the original strings.
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

        // R1: re-order codes into byte-lexicographic rank and remap the row
        // indices so the integer codes match string order before upload.
        lex_sort_dictionary(&mut dictionary, &mut indices);

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

    /// Collation rank array (finding F10).
    ///
    /// Returns a vector `ranks` of length `dictionary.len()` where `ranks[p]`
    /// is the 0-based position of `dictionary[p]` in the byte-lexicographically
    /// sorted order of all distinct dictionary entries. In other words, it is
    /// the permutation that maps each entry's *insertion* slot (`p`, i.e. GPU
    /// index `p + 1`) to its *sort* rank.
    ///
    /// This is **binary collation**: entries are ordered by raw UTF-8 byte
    /// sequence (`str`'s `Ord`, which is `[u8]` lexicographic), NOT a
    /// locale-aware / ICU collation. `'Z' < 'a'` and combining sequences are
    /// compared bytewise. Locale collation is explicitly out of scope.
    ///
    /// The NULL slot (GPU index 0) is not represented here — it has no string
    /// to rank. Callers that need a NULL-aware mapping treat slot 0 separately
    /// (a NULL value compares as SQL NULL, never satisfying an ordering
    /// predicate; see [`crate::plan::string_literal_rewrite`]).
    ///
    /// Cost: `O(N log N)` over the dictionary. Intended for query-plan time,
    /// not a per-row hot path.
    pub fn collation_ranks(&self) -> Vec<usize> {
        collation_ranks_of(&self.dictionary)
    }

    /// Byte-lexicographic insertion rank of a probe literal (finding F10).
    ///
    /// Returns the number of distinct dictionary entries that sort strictly
    /// before `probe` under binary (UTF-8 byte) collation — equivalently, the
    /// half-open insertion point of `probe` in the sorted distinct-value
    /// sequence. The result is in `0..=dictionary.len()` and is well defined
    /// whether or not `probe` is itself present in the dictionary.
    ///
    /// This is the value an ordering rewrite compares ranks against: for a
    /// `col < probe` predicate the matching entries are exactly those whose
    /// sort rank is `< insertion_rank(probe)`; for `col <= probe` the bound is
    /// the count of entries that sort `<= probe`, which is
    /// `insertion_rank(probe)` plus one when `probe` is present. See
    /// [`Self::indices_satisfying`].
    pub fn insertion_rank(&self, probe: &str) -> usize {
        self.dictionary
            .iter()
            .filter(|d| d.as_str() < probe)
            .count()
    }

    /// GPU indices of every dictionary entry that satisfies `entry OP probe`
    /// under byte-lexicographic collation (finding F10).
    ///
    /// `op` is one of the four ordering comparisons (`<`, `<=`, `>`, `>=`),
    /// expressed as a [`crate::plan::logical_plan::BinaryOp`]. The returned
    /// indices are 1-based GPU indices (slot 0 / NULL is never included — a
    /// NULL row must yield SQL NULL, not a match), in ascending index order so
    /// the caller can lower them to a deterministic OR-of-equalities.
    ///
    /// This is the dictionary-precompute that lets `col OP 'lit'` over a
    /// dict-encoded Utf8 column be evaluated by the existing integer-index GPU
    /// machinery: the host computes the matching index set once and the GPU
    /// only ever does integer compares. The literal need not be present in the
    /// dictionary — the comparison is against the actual entry strings, so an
    /// absent literal still partitions the entries correctly (half-open
    /// insertion semantics fall out of the per-entry `<` / `<=` test).
    pub fn indices_satisfying(
        &self,
        op: crate::plan::logical_plan::BinaryOp,
        probe: &str,
    ) -> Vec<i32> {
        indices_satisfying_in(&self.dictionary, op, probe)
            .into_iter()
            .map(|p| p as i32)
            .collect()
    }

    /// Per-GPU-index rank lookup against a shared sorted `universe`
    /// (finding F12, column-vs-column Utf8 ordering).
    ///
    /// Thin wrapper over [`rank_for_index_in`]: see that function for the
    /// layout (length `dictionary.len() + 1`, indexed by GPU index, slot 0 =
    /// [`NULL_RANK_SENTINEL`]). Used by the column-vs-column ordering lowering
    /// to materialise a per-row rank column comparable across two different
    /// dictionaries.
    pub fn rank_for_index(&self, universe: &[String]) -> Vec<i64> {
        rank_for_index_in(&self.dictionary, universe)
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
    pub(crate) fn dedupe_for_test<'a, I>(rows: I) -> BoltResult<(Vec<String>, Vec<i32>)>
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
        // R1 parity with `from_string_array`: lex-sort and remap.
        lex_sort_dictionary(&mut dictionary, &mut indices);
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
            out.push(Self::decode_index(&self.dictionary, idx)?);
        }
        Ok(StringArray::from(out))
    }

    /// Decode a single 1-based dictionary index into the string it names.
    ///
    /// `0` decodes to `None` (SQL `NULL`). Negative indices, or indices
    /// outside `1..=dictionary.len()`, surface as `BoltError::Other`.
    ///
    /// V-14 (defense-in-depth, parity with the i64 sibling in
    /// `dictionary_i64.rs::to_string_array`): width-safe decode. The 1-based
    /// offset is validated against the dictionary length in `u64` *before*
    /// narrowing to `usize`. On a 32-bit host a direct `as usize` cast would
    /// truncate before the bounds check, letting an index that does not fit a
    /// 32-bit `usize` accidentally hit a valid slot. `.get()` keeps the final
    /// lookup memory-safe regardless of host width.
    ///
    /// Pulled out of `to_string_array` so the guard is exercisable host-only
    /// (the loop there downloads from a `GpuVec`, which a CUDA-less test host
    /// cannot populate).
    fn decode_index(dictionary: &[String], idx: i32) -> BoltResult<Option<&str>> {
        if idx == 0 {
            return Ok(None);
        }
        if idx < 0 {
            return Err(BoltError::Other(format!(
                "dictionary decode: negative index {} (NULL is encoded as 0)",
                idx
            )));
        }
        let pos_u64 = (idx as u64) - 1;
        if pos_u64 >= dictionary.len() as u64 {
            return Err(BoltError::Other(format!(
                "dictionary decode: index {} out of range (dictionary size {})",
                idx,
                dictionary.len()
            )));
        }
        let pos = pos_u64 as usize;
        let s = dictionary.get(pos).ok_or_else(|| {
            BoltError::Other(format!(
                "dictionary decode: index {} out of range (dictionary size {})",
                idx,
                dictionary.len()
            ))
        })?;
        Ok(Some(s.as_str()))
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

    // ---- F10: byte-lexicographic collation (host-only) -------------------

    #[test]
    fn collation_ranks_is_byte_lexicographic_permutation() {
        // Insertion order is deliberately unsorted. Byte order puts the
        // uppercase 'Z' (0x5A) before any lowercase letter (>= 0x61) — the
        // binary-collation signature, distinct from locale collation.
        // slots:            0        1        2        3
        let dict = vec![
            "delta".to_string(),
            "apple".to_string(),
            "Zebra".to_string(),
            "mango".to_string(),
        ];
        let col = DictionaryColumn::new_host_only(dict, 0);
        // sorted: Zebra(2) < apple(1) < delta(0) < mango(3)
        assert_eq!(col.collation_ranks(), vec![2, 1, 0, 3]);
    }

    #[test]
    fn collation_ranks_empty_dictionary_is_empty() {
        let col = DictionaryColumn::new_host_only(Vec::new(), 0);
        assert!(col.collation_ranks().is_empty());
    }

    #[test]
    fn insertion_rank_present_and_absent() {
        let dict = vec![
            "apple".to_string(),
            "delta".to_string(),
            "mango".to_string(),
        ];
        let col = DictionaryColumn::new_host_only(dict, 0);
        // present
        assert_eq!(col.insertion_rank("apple"), 0);
        assert_eq!(col.insertion_rank("mango"), 2);
        // absent: half-open insertion point
        assert_eq!(col.insertion_rank("aardvark"), 0);
        assert_eq!(col.insertion_rank("cat"), 1);
        assert_eq!(col.insertion_rank("zzz"), 3);
    }

    #[test]
    fn indices_satisfying_matches_direct_byte_comparison() {
        use crate::plan::logical_plan::BinaryOp;
        let entries = ["delta", "apple", "Zebra", "mango"];
        let dict: Vec<String> = entries.iter().map(|s| s.to_string()).collect();
        let col = DictionaryColumn::new_host_only(dict, 0);

        let oracle = |op: BinaryOp, lit: &str| -> Vec<i32> {
            entries
                .iter()
                .enumerate()
                .filter(|(_, s)| match op {
                    BinaryOp::Lt => **s < lit,
                    BinaryOp::LtEq => **s <= lit,
                    BinaryOp::Gt => **s > lit,
                    BinaryOp::GtEq => **s >= lit,
                    _ => unreachable!(),
                })
                .map(|(p, _)| (p as i32) + 1)
                .collect()
        };

        for &lit in &["mango", "cat", "Zebra", "zzz", "AAA"] {
            for op in [BinaryOp::Lt, BinaryOp::LtEq, BinaryOp::Gt, BinaryOp::GtEq] {
                let mut got = col.indices_satisfying(op, lit);
                got.sort_unstable();
                let mut want = oracle(op, lit);
                want.sort_unstable();
                assert_eq!(got, want, "op {op:?} lit {lit:?}");
            }
            // The NULL slot 0 must never appear in any match set.
            for op in [BinaryOp::Lt, BinaryOp::LtEq, BinaryOp::Gt, BinaryOp::GtEq] {
                assert!(!col.indices_satisfying(op, lit).contains(&0));
            }
        }
    }

    // ---- F12: cross-dictionary unified collation ranks (host-only) -------

    /// `rank_for_index` against a shared universe maps each GPU index to the
    /// byte-sorted position of its string in that universe, with slot 0 (NULL)
    /// holding the sentinel. Insertion order is deliberately unsorted.
    #[test]
    fn rank_for_index_against_universe() {
        // dict slots:        1        2        3
        let dict = vec![
            "mango".to_string(),
            "apple".to_string(),
            "delta".to_string(),
        ];
        let col = DictionaryColumn::new_host_only(dict, 0);
        // Universe (sorted, deduped) — here equal to the dict's own sorted set.
        let universe = vec![
            "apple".to_string(),
            "delta".to_string(),
            "mango".to_string(),
        ];
        let ranks = col.rank_for_index(&universe);
        // index0=NULL sentinel; idx1="mango"->2, idx2="apple"->0, idx3="delta"->1
        assert_eq!(ranks, vec![NULL_RANK_SENTINEL, 2, 0, 1]);
    }

    /// `unified_rank_maps_of` over TWO DIFFERENT dictionaries must rank both
    /// against one shared universe, so `rank_a OP rank_b` reproduces the direct
    /// byte-string comparison for every row pairing and every ordering op. This
    /// is the load-bearing cross-dictionary correctness test.
    #[test]
    fn unified_rank_maps_cross_dictionary_oracle() {
        use crate::plan::logical_plan::BinaryOp;
        // Two genuinely different dictionaries (different strings + order).
        let dict_a = vec![
            "delta".to_string(), // idx1
            "apple".to_string(), // idx2
            "mango".to_string(), // idx3
        ];
        let dict_b = vec![
            "cherry".to_string(), // idx1
            "Zebra".to_string(),  // idx2
            "apple".to_string(),  // idx3 (shared string with dict_a)
        ];
        let (rank_a, rank_b) = unified_rank_maps_of(&dict_a, &dict_b);
        // Slot 0 is the NULL sentinel on both sides.
        assert_eq!(rank_a[0], NULL_RANK_SENTINEL);
        assert_eq!(rank_b[0], NULL_RANK_SENTINEL);

        let cmp = |op: BinaryOp, x: i64, y: i64| -> bool {
            match op {
                BinaryOp::Lt => x < y,
                BinaryOp::LtEq => x <= y,
                BinaryOp::Gt => x > y,
                BinaryOp::GtEq => x >= y,
                _ => unreachable!(),
            }
        };
        // For every (a_string, b_string) pairing, the rank comparison must
        // match the direct byte-string comparison, for all four ops.
        for (ai, a_s) in dict_a.iter().enumerate() {
            for (bi, b_s) in dict_b.iter().enumerate() {
                let ra = rank_a[ai + 1]; // +1: skip NULL slot
                let rb = rank_b[bi + 1];
                for op in [BinaryOp::Lt, BinaryOp::LtEq, BinaryOp::Gt, BinaryOp::GtEq] {
                    let by_rank = cmp(op, ra, rb);
                    let by_string = match op {
                        BinaryOp::Lt => a_s < b_s,
                        BinaryOp::LtEq => a_s <= b_s,
                        BinaryOp::Gt => a_s > b_s,
                        BinaryOp::GtEq => a_s >= b_s,
                        _ => unreachable!(),
                    };
                    assert_eq!(
                        by_rank, by_string,
                        "op {op:?}: rank({a_s})={ra} vs rank({b_s})={rb} disagrees with strings"
                    );
                }
            }
        }
        // Shared string "apple" must get the SAME rank in both tables (it is the
        // same position in the shared universe): dict_a idx2, dict_b idx3.
        assert_eq!(rank_a[2], rank_b[3], "shared string must share a rank");
    }

    /// Same-dictionary degenerate case: the two rank tables coincide and equal
    /// the column's own `collation_ranks` (offset by the NULL slot).
    #[test]
    fn unified_rank_maps_same_dictionary_collapses() {
        let dict = vec![
            "delta".to_string(),
            "apple".to_string(),
            "Zebra".to_string(),
        ];
        let (rank_a, rank_b) = unified_rank_maps_of(&dict, &dict);
        assert_eq!(rank_a, rank_b, "identical dicts → identical rank tables");
        // Drop the NULL sentinel and compare to collation_ranks (which is
        // indexed by 0-based slot, no NULL entry).
        let col = DictionaryColumn::new_host_only(dict, 0);
        let collation: Vec<i64> = col.collation_ranks().iter().map(|&r| r as i64).collect();
        assert_eq!(&rank_a[1..], collation.as_slice());
    }

    /// A string present in one dictionary but absent from the universe is
    /// impossible when the universe is the union (defensive); but a column
    /// ranked against a SUPERSET universe still gets a consistent position.
    #[test]
    fn rank_for_index_superset_universe_is_consistent() {
        let dict = vec!["banana".to_string(), "apple".to_string()];
        let col = DictionaryColumn::new_host_only(dict, 0);
        // Universe is a strict superset (adds "aardvark","cherry").
        let universe = vec![
            "aardvark".to_string(),
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ];
        let ranks = col.rank_for_index(&universe);
        // idx1="banana"->2, idx2="apple"->1
        assert_eq!(ranks, vec![NULL_RANK_SENTINEL, 2, 1]);
    }

    #[test]
    fn indices_satisfying_non_ordering_op_is_empty() {
        use crate::plan::logical_plan::BinaryOp;
        let dict = vec!["a".to_string(), "b".to_string()];
        let col = DictionaryColumn::new_host_only(dict, 0);
        // Eq/NotEq are not this helper's responsibility — empty result.
        assert!(col.indices_satisfying(BinaryOp::Eq, "a").is_empty());
        assert!(col.indices_satisfying(BinaryOp::NotEq, "a").is_empty());
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
        //
        // R1: codes are now byte-lexicographic, so the dictionary comes out
        // *sorted* (NOT first-seen order). `format!("val_{i}")` sorts
        // lexicographically as val_0, val_1, val_10, val_11, …, val_19, val_2,
        // … — so we verify against the sorted pool, not the numeric pool.
        const ROWS: usize = 1_000_000;
        const DISTINCT: usize = 100;

        // Pre-materialise the 100 distinct strings so we can borrow them as
        // `&str` for the dedupe iterator without re-allocating per row.
        let pool: Vec<String> = (0..DISTINCT).map(|i| format!("val_{i}")).collect();
        let rows = (0..ROWS).map(|r| Some(pool[r % DISTINCT].as_str()));

        let (dictionary, indices) = DictionaryColumn::dedupe_for_test(rows).expect("dedupe");

        // Distinct count must equal the input cardinality, now in byte-lex
        // order.
        assert_eq!(dictionary.len(), DISTINCT);
        let mut sorted_pool = pool.clone();
        sorted_pool.sort();
        assert_eq!(
            dictionary, sorted_pool,
            "dictionary must be byte-lex sorted"
        );
        // The dictionary is strictly increasing (sorted + distinct).
        for w in dictionary.windows(2) {
            assert!(w[0] < w[1], "dictionary must be strictly lex-increasing");
        }
        // Every row must have a positive index (slot 0 reserved for NULL) that
        // decodes back to the original string for that row — the load-bearing
        // round-trip / remap-consistency check.
        assert_eq!(indices.len(), ROWS);
        let col = DictionaryColumn::new_host_only(dictionary, ROWS);
        for (r, &idx) in indices.iter().enumerate() {
            assert!(idx >= 1, "row {r} must have a non-NULL index, got {idx}");
            let decoded = DictionaryColumn::decode_index(&col.dictionary, idx).expect("decode");
            assert_eq!(
                decoded,
                Some(pool[r % DISTINCT].as_str()),
                "row {r} index {idx} must decode to its original string"
            );
        }
        // `index_of` on the resulting dictionary must equal each value's lex
        // rank (+1 for the NULL slot).
        for (lex_rank, s) in sorted_pool.iter().enumerate() {
            assert_eq!(col.index_of(s), Some((lex_rank as i32) + 1));
        }
        // Sanity: a string never seen during construction returns None.
        assert_eq!(col.index_of("missing-literal"), None);
    }

    #[test]
    fn dedupe_handles_interleaved_nulls() {
        // Nulls go to index 0 and do not enter the dictionary. Verifies the
        // digest-map dedupe doesn't accidentally treat None as a value.
        let rows = vec![Some("a"), None, Some("b"), None, Some("a"), Some("c"), None];
        let (dictionary, indices) = DictionaryColumn::dedupe_for_test(rows).expect("dedupe");

        assert_eq!(
            dictionary,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(indices, vec![1, 0, 2, 0, 1, 3, 0]);
    }

    // ---- R1: lex-rank dictionaries at ingest -----------------------------

    /// `lex_sort_dictionary` re-orders a first-seen dictionary into byte-lex
    /// order and remaps the row indices so each value keeps pointing at its
    /// string. Pure host — exercises the shared primitive directly.
    #[test]
    fn lex_sort_dictionary_reorders_and_remaps() {
        // First-seen dictionary (slots 1,2,3) and a row index stream that uses
        // every code plus a NULL.
        //                          idx1     idx2     idx3
        let mut dict = vec![
            "delta".to_string(),
            "apple".to_string(),
            "mango".to_string(),
        ];
        //               delta apple NULL mango apple delta
        let mut indices: Vec<i32> = vec![1, 2, 0, 3, 2, 1];

        lex_sort_dictionary(&mut dict, &mut indices);

        // Sorted: apple(1) < delta(2) < mango(3).
        assert_eq!(
            dict,
            vec![
                "apple".to_string(),
                "delta".to_string(),
                "mango".to_string()
            ]
        );
        // Each non-NULL index now names its value's lex rank; NULL stays 0.
        // delta→2, apple→1, NULL→0, mango→3, apple→1, delta→2.
        assert_eq!(indices, vec![2, 1, 0, 3, 1, 2]);
    }

    /// `lex_sort_dictionary` on an empty dictionary is a no-op (no panic).
    #[test]
    fn lex_sort_dictionary_empty_is_noop() {
        let mut dict: Vec<String> = Vec::new();
        let mut indices: Vec<i32> = vec![0, 0];
        lex_sort_dictionary(&mut dict, &mut indices);
        assert!(dict.is_empty());
        assert_eq!(indices, vec![0, 0]);
    }

    /// R1 (distinct → lex codes): after dedupe the dictionary is strictly
    /// byte-lex increasing and `index_of` returns each value's 1-based lex rank,
    /// regardless of the (deliberately unsorted, repeated, out-of-order) input.
    #[test]
    fn dedupe_assigns_codes_in_lex_order() {
        // Repeated / out-of-order inserts: "mango" first, "Apple" (uppercase),
        // "delta", repeats, plus a NULL.
        let rows = vec![
            Some("mango"),
            Some("delta"),
            Some("Apple"),
            Some("mango"),
            None,
            Some("delta"),
            Some("Apple"),
        ];
        let (dictionary, indices) =
            DictionaryColumn::dedupe_for_test(rows.clone()).expect("dedupe");

        // Byte collation: 'A' (0x41) sorts before lowercase letters.
        assert_eq!(
            dictionary,
            vec![
                "Apple".to_string(),
                "delta".to_string(),
                "mango".to_string()
            ]
        );
        let col = DictionaryColumn::new_host_only(dictionary, rows.len());
        assert_eq!(col.index_of("Apple"), Some(1));
        assert_eq!(col.index_of("delta"), Some(2));
        assert_eq!(col.index_of("mango"), Some(3));

        // R1 (encode → decode round-trips): every row index decodes to the
        // exact original string it was built from (NULL preserved).
        assert_eq!(indices.len(), rows.len());
        for (r, &idx) in indices.iter().enumerate() {
            let decoded = DictionaryColumn::decode_index(&col.dictionary, idx).expect("decode");
            assert_eq!(decoded, rows[r], "row {r} must round-trip");
        }
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

    // ---- V-14: width-safe index decode (host-only) -----------------------
    //
    // `decode_index` is the pure host half of `to_string_array`; testing it
    // directly avoids needing a CUDA device to populate `indices`. Mirrors the
    // i64 sibling's width-safe guard in `dictionary_i64.rs::to_string_array`.

    #[test]
    fn decode_index_in_range_returns_correct_slot() {
        // 1-based indices: slot 0 is NULL, "a"/"b"/"c" live at 1/2/3.
        let dict = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(DictionaryColumn::decode_index(&dict, 0).unwrap(), None);
        assert_eq!(DictionaryColumn::decode_index(&dict, 1).unwrap(), Some("a"));
        assert_eq!(DictionaryColumn::decode_index(&dict, 2).unwrap(), Some("b"));
        assert_eq!(DictionaryColumn::decode_index(&dict, 3).unwrap(), Some("c"));
    }

    #[test]
    fn decode_index_out_of_range_errors_without_panic_or_wrong_slot() {
        // Just past the end, the largest positive index, and a negative
        // sentinel must all surface a graceful error — never a panic and never
        // a stray slot. The V-14 guard compares in u64 before narrowing, so
        // even an index that wouldn't fit a 32-bit usize is rejected.
        let dict = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert!(DictionaryColumn::decode_index(&dict, 4).is_err());
        assert!(DictionaryColumn::decode_index(&dict, i32::MAX).is_err());
        assert!(DictionaryColumn::decode_index(&dict, -1).is_err());

        // Empty dictionary: any positive index is out of range, 0 is NULL.
        let empty: Vec<String> = Vec::new();
        assert_eq!(DictionaryColumn::decode_index(&empty, 0).unwrap(), None);
        assert!(DictionaryColumn::decode_index(&empty, 1).is_err());
    }

    // ---- GPU-required tests ----------------------------------------------
    //
    // These call `from_string_array` (which uploads to the device) or
    // `to_string_array` (which downloads). They cannot run on a host without
    // a CUDA toolkit, so they are `#[ignore]`d the same way the i64 sibling
    // tests are.

    #[test]
    #[ignore = "gpu:string"]
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
    #[ignore = "gpu:string"]
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
    #[ignore = "gpu:string"]
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
    #[ignore = "gpu:string"]
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
    #[ignore = "gpu:string"]
    fn dict_index_of_lookup() {
        // After a real encode, `index_of` must report the same slots that
        // the indices vec already uses for those strings. R1: codes are in
        // byte-lexicographic order, so blue < green < red ⇒ 1, 2, 3.
        let input = StringArray::from(vec!["red", "green", "blue", "green"]);
        let col = DictionaryColumn::from_string_array(&input).expect("encode");

        assert_eq!(
            col.dictionary,
            vec!["blue".to_string(), "green".to_string(), "red".to_string()]
        );
        assert_eq!(col.index_of("blue"), Some(1));
        assert_eq!(col.index_of("green"), Some(2));
        assert_eq!(col.index_of("red"), Some(3));
        // Literal never seen during construction => None, not 0.
        assert_eq!(col.index_of("yellow"), None);
    }

    #[test]
    #[ignore = "gpu:string"]
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
