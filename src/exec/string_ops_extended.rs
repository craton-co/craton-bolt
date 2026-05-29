// SPDX-License-Identifier: Apache-2.0

//! Dictionary-aware extended string operations: `CONCAT`, `SUBSTRING`,
//! `CONCAT_WS`.
//!
//! ## Why this lives on the host
//!
//! Same rationale as `string_ops`: variable-width string writes on the GPU are
//! painful, so we operate purely on the host-side `DictionaryColumn`. The
//! distinction from `UPPER` / `LOWER` is that:
//!
//! * **CONCAT / CONCAT_WS** consume *two or more* dictionaries and can produce
//!   strings that are in NEITHER input dictionary. We therefore build the
//!   output dictionary row-by-row from the concatenated values rather than via
//!   a per-entry remap. Worst case the output dictionary is
//!   `|d_1| * |d_2| * ...` entries; in practice repeated rows dedup heavily.
//!
//! * **SUBSTRING** is a per-entry transform (so its shape mirrors `upper` /
//!   `lower` exactly) but the transformed strings may collapse new duplicates
//!   ("abc" and "abd" both substring to "ab"), so we still reuse the
//!   dedup-with-remap pattern.
//!
//! ## NULL semantics
//!
//! * `CONCAT(a, b)`: if EITHER side is NULL the result is NULL (standard SQL).
//! * `SUBSTRING(NULL, ...)`: result is NULL.
//! * `CONCAT_WS(sep, ...)`: NULLs are SKIPPED (standard CONCAT_WS semantic).
//!   If every argument is NULL the result is the empty string, which is
//!   distinct from NULL (lives at index 1, not 0).
//!
//! ## v1 caveats
//!
//! * Byte-indexed `SUBSTRING` (matches the byte-length convention in
//!   `string_ops::length`). If the requested byte boundary falls inside a
//!   multi-byte UTF-8 codepoint we round DOWN to the previous valid char
//!   boundary; the slice never panics on non-ASCII input. A future
//!   `SUBSTRING_CHAR` would walk codepoints / graphemes.
//! * Two-argument `SUBSTRING(col, start)` ("to the end") is not exposed as a
//!   separate entry point; pass `length = i32::MAX` instead.
//! * `CONCAT` is binary here. The variadic case is covered by `CONCAT_WS`
//!   with `separator = ""`; a true variadic `CONCAT` could be added but is
//!   not required by the current planner.
//! * No regex / `REPLACE` / `TRIM` yet.
//! * Output dictionary size is not capped beyond the usual `i32::MAX` index
//!   bound. A pathological `CONCAT` of two 1M-entry dictionaries with no row
//!   coincidences could theoretically produce 1M unique outputs; that is the
//!   caller's responsibility, mirroring how the existing `from_string_array`
//!   path also makes no a priori cap.

use std::collections::HashMap;

use crate::cuda::dictionary::DictionaryColumn;
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};

// ---------------------------------------------------------------------------
// Internal: shared dictionary-overflow checked insert.
// ---------------------------------------------------------------------------

/// Insert `s` into (`dict`, `lookup`) if absent and return its 1-based index.
/// Surfaces dictionary overflow as `BoltError::Other`, matching the
/// existing `string_ops` / `from_string_array` conventions.
fn intern(
    dict: &mut Vec<String>,
    lookup: &mut HashMap<String, i32>,
    s: String,
) -> BoltResult<i32> {
    if let Some(&idx) = lookup.get(&s) {
        return Ok(idx);
    }
    let next_len = dict.len().checked_add(1).ok_or_else(|| {
        BoltError::Other("dictionary overflow: more than usize::MAX unique strings".into())
    })?;
    if next_len > i32::MAX as usize {
        return Err(BoltError::Other(format!(
            "dictionary overflow: more than {} unique strings (i32 index space)",
            i32::MAX
        )));
    }
    let idx = next_len as i32;
    dict.push(s.clone());
    lookup.insert(s, idx);
    Ok(idx)
}

/// Re-deduplicate a transformed-per-entry dictionary and produce the index
/// remap (slot 0 = NULL passthrough, slot `i+1` maps old index `i+1` to new).
///
/// This is the same shape as `string_ops::dedup_transformed`; duplicating it
/// here rather than reaching across modules avoids coupling these two
/// independently-evolving feature sets.
fn dedup_transformed(transformed: Vec<String>) -> BoltResult<(Vec<String>, Vec<i32>)> {
    let n_old = transformed.len();
    let mut new_dict: Vec<String> = Vec::new();
    let mut lookup: HashMap<String, i32> = HashMap::new();
    // remap[0] = 0 (NULL passthrough); remainder filled in below.
    let mut remap: Vec<i32> = vec![0; n_old + 1];

    for (i, s) in transformed.into_iter().enumerate() {
        let idx = intern(&mut new_dict, &mut lookup, s)?;
        remap[i + 1] = idx;
    }

    Ok((new_dict, remap))
}

// ---------------------------------------------------------------------------
// SUBSTRING byte-slice helper.
// ---------------------------------------------------------------------------

/// Compute `SUBSTRING(s, start_1based, length)` over BYTES, per the v1
/// byte-semantics caveat documented at the top of the module.
///
/// Semantics:
///
/// * `start < 1` clamps to `1` (ANSI SQL).
/// * `length < 0` is treated as `0` (empty result).
/// * `length` saturates: `byte_end = min(byte_start + length, s.len())`.
/// * If `byte_start >= s.len()`, the result is `""`.
/// * If `byte_start` or `byte_end` falls inside a multi-byte UTF-8 codepoint,
///   we round DOWN to the nearest preceding `is_char_boundary`. This means
///   `SUBSTRING("héllo", 1, 2)` returns `"h"` rather than panicking: byte 2
///   lands inside the two-byte `é`, so we round the end back to byte 1.
///
/// The result is always a valid UTF-8 substring of `s`.
fn sql_substring(s: &str, start_1based: i32, length: i32) -> String {
    // ANSI: start < 1 clamps to 1.
    let start = start_1based.max(1);
    // ANSI: negative length is 0. We use an i64 intermediate so length =
    // i32::MAX ("to end") doesn't wrap when added to byte_start.
    let length = length.max(0);
    // Convert to usize-domain math. `start` is now >= 1, so `start - 1 >= 0`.
    let byte_start_raw = (start - 1) as usize;
    let byte_end_raw = byte_start_raw.saturating_add(length as usize);

    let s_len = s.len();
    // Fast exit when the requested slice starts past the end of the string.
    if byte_start_raw >= s_len {
        return String::new();
    }
    // Clamp end to string length.
    let byte_end_clamped = byte_end_raw.min(s_len);

    // Round both endpoints DOWN to char boundaries so the slice is valid UTF-8.
    // `s_len` is always a valid boundary, but we still apply this generically
    // for the interior case. `is_char_boundary(0)` is true so the loop
    // terminates immediately for byte_start = 0.
    let byte_start = round_down_to_char_boundary(s, byte_start_raw);
    let byte_end = round_down_to_char_boundary(s, byte_end_clamped);

    // After rounding, byte_end could end up below byte_start (e.g. start
    // rounded up and end rounded down). Guard against that: return empty
    // rather than panicking on a reversed slice.
    if byte_end <= byte_start {
        return String::new();
    }
    s[byte_start..byte_end].to_string()
}

/// Per-string `SUBSTRING(s, start, length)` over BYTES — public wrapper around
/// the internal `sql_substring` helper so the host expression evaluator
/// (`expr_agg::eval_expr`) can apply SUBSTRING directly to a `Vec<Option<String>>`
/// column without round-tripping through a `DictionaryColumn`. See module docs
/// for the byte/UTF-8 boundary semantics.
///
/// TODO(string-fn-gpu): the GPU two-pass SUBSTRING producer exists in
/// `jit::string_kernel` but is not wired into the executor; this host path is
/// the supported one.
pub fn substring_str(s: &str, start_1based: i32, length: i32) -> String {
    sql_substring(s, start_1based, length)
}

/// Which end(s) a TRIM operation strips from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimSide {
    /// Strip from BOTH ends (default for `TRIM`).
    Both,
    /// Strip from the START only (`TRIM(LEADING ...)`).
    Leading,
    /// Strip from the END only (`TRIM(TRAILING ...)`).
    Trailing,
}

/// Pure-host `TRIM`. Strips `side` end(s) of `s`.
///
/// * `chars = None`: strip ASCII/Unicode whitespace (Rust's `char::is_whitespace`,
///   matching the conventional SQL default).
/// * `chars = Some(set)`: strip any leading/trailing character that appears in
///   `set` (the set is a bag of individual characters, per SQL `TRIM('xy' FROM s)`
///   which removes any of `x`/`y`, NOT the literal substring `"xy"`).
///
/// Operates on `char`s so multi-byte trim characters and multi-byte input both
/// behave correctly. An empty `chars` set (e.g. `TRIM('' FROM s)`) strips
/// nothing and returns `s` unchanged.
///
/// TODO(string-fn-gpu): no GPU kernel yet; this host path is the supported one.
pub fn trim_str(s: &str, side: TrimSide, chars: Option<&str>) -> String {
    match chars {
        None => match side {
            TrimSide::Both => s.trim().to_string(),
            TrimSide::Leading => s.trim_start().to_string(),
            TrimSide::Trailing => s.trim_end().to_string(),
        },
        Some(set) => {
            // Empty set: nothing to strip.
            if set.is_empty() {
                return s.to_string();
            }
            let in_set = |c: char| set.chars().any(|t| t == c);
            match side {
                TrimSide::Both => s.trim_matches(in_set).to_string(),
                TrimSide::Leading => s.trim_start_matches(in_set).to_string(),
                TrimSide::Trailing => s.trim_end_matches(in_set).to_string(),
            }
        }
    }
}

/// Round `byte_idx` down to the nearest `is_char_boundary`. `byte_idx` must be
/// `<= s.len()`; this is the case at every caller.
fn round_down_to_char_boundary(s: &str, mut byte_idx: usize) -> usize {
    // s.len() and 0 are always char boundaries, so the loop terminates.
    while !s.is_char_boundary(byte_idx) {
        byte_idx -= 1;
    }
    byte_idx
}

// ---------------------------------------------------------------------------
// Pure helpers (no GPU). The transformation logic lives here so it can be
// exhaustively unit tested without a CUDA runtime.
// ---------------------------------------------------------------------------

/// Pure-host implementation of binary `CONCAT`. Walks each row, materialises
/// the concatenated string, and interns it in a freshly-built dictionary.
///
/// Returns `(new_dict, new_indices)`. `new_indices.len() == left_indices.len()
/// == right_indices.len()`. NULL on either side propagates as index `0`.
///
/// Errors:
/// * `BoltError::Other` if the input index vectors have different lengths.
/// * `BoltError::Other` if either index falls outside `0..=dict.len()`.
/// * `BoltError::Other` on dictionary overflow (>= i32::MAX outputs).
fn concat_pure(
    left_dict: &[String],
    left_indices: &[i32],
    right_dict: &[String],
    right_indices: &[i32],
) -> BoltResult<(Vec<String>, Vec<i32>)> {
    if left_indices.len() != right_indices.len() {
        return Err(BoltError::Other(format!(
            "CONCAT: n_rows mismatch (left = {}, right = {})",
            left_indices.len(),
            right_indices.len()
        )));
    }

    let n = left_indices.len();
    let mut new_dict: Vec<String> = Vec::new();
    let mut lookup: HashMap<String, i32> = HashMap::new();
    let mut out: Vec<i32> = Vec::with_capacity(n);

    for i in 0..n {
        let li = left_indices[i];
        let ri = right_indices[i];

        // Strict bounds: a negative index would indicate a kernel-side bug
        // we'd rather surface than mask. Same posture as `string_ops`.
        if li < 0 {
            return Err(BoltError::Other(format!(
                "CONCAT: negative left index {} at row {} (NULL is encoded as 0)",
                li, i
            )));
        }
        if ri < 0 {
            return Err(BoltError::Other(format!(
                "CONCAT: negative right index {} at row {} (NULL is encoded as 0)",
                ri, i
            )));
        }

        // SQL: NULL on either side -> NULL output.
        if li == 0 || ri == 0 {
            out.push(0);
            continue;
        }

        let lpos = (li as usize) - 1;
        let rpos = (ri as usize) - 1;
        let lstr = left_dict.get(lpos).ok_or_else(|| {
            BoltError::Other(format!(
                "CONCAT: left index {} out of range (dictionary size {}) at row {}",
                li,
                left_dict.len(),
                i
            ))
        })?;
        let rstr = right_dict.get(rpos).ok_or_else(|| {
            BoltError::Other(format!(
                "CONCAT: right index {} out of range (dictionary size {}) at row {}",
                ri,
                right_dict.len(),
                i
            ))
        })?;

        // `format!` over two known strings; no allocator surprises.
        let combined = format!("{}{}", lstr, rstr);
        let idx = intern(&mut new_dict, &mut lookup, combined)?;
        out.push(idx);
    }

    Ok((new_dict, out))
}

/// Pure-host implementation of `SUBSTRING`. Applies `sql_substring` to every
/// dictionary entry then dedups via the standard remap pattern. The remap
/// table has length `input_dict.len() + 1` with slot 0 = NULL passthrough.
fn substring_pure(
    input_dict: &[String],
    start: i32,
    length: i32,
) -> BoltResult<(Vec<String>, Vec<i32>)> {
    let transformed: Vec<String> = input_dict
        .iter()
        .map(|s| sql_substring(s, start, length))
        .collect();
    dedup_transformed(transformed)
}

/// Pure-host implementation of `CONCAT_WS`. For each row, collects every
/// non-NULL value from `inputs` (in order), joins them with `separator`, and
/// interns the result. NULL handling is the standard CONCAT_WS semantic:
/// NULLs are SKIPPED, not propagated. An all-NULL row therefore produces the
/// empty string (a real, non-NULL dictionary entry), not SQL NULL.
///
/// Errors:
/// * `BoltError::Other` if `inputs` is empty (no columns to join — the
///   planner should never emit this, but we surface it rather than panic).
/// * `BoltError::Other` if column row counts disagree.
/// * `BoltError::Other` on any negative or out-of-range index.
/// * `BoltError::Other` on dictionary overflow.
fn concat_ws_pure(
    separator: &str,
    inputs: &[(&[String], &[i32])],
) -> BoltResult<(Vec<String>, Vec<i32>)> {
    if inputs.is_empty() {
        return Err(BoltError::Other(
            "CONCAT_WS: at least one input column is required".into(),
        ));
    }

    let n = inputs[0].1.len();
    for (k, (_, idx)) in inputs.iter().enumerate().skip(1) {
        if idx.len() != n {
            return Err(BoltError::Other(format!(
                "CONCAT_WS: n_rows mismatch (column 0 = {}, column {} = {})",
                n,
                k,
                idx.len()
            )));
        }
    }

    let mut new_dict: Vec<String> = Vec::new();
    let mut lookup: HashMap<String, i32> = HashMap::new();
    let mut out: Vec<i32> = Vec::with_capacity(n);
    // Reused per-row scratch so we don't reallocate on every iteration.
    let mut pieces: Vec<&str> = Vec::with_capacity(inputs.len());

    for i in 0..n {
        pieces.clear();
        for (col_k, (dict, idx_vec)) in inputs.iter().enumerate() {
            let idx = idx_vec[i];
            if idx < 0 {
                return Err(BoltError::Other(format!(
                    "CONCAT_WS: negative index {} at row {}, column {} (NULL is encoded as 0)",
                    idx, i, col_k
                )));
            }
            if idx == 0 {
                // NULL: skipped per CONCAT_WS semantics.
                continue;
            }
            let pos = (idx as usize) - 1;
            let s = dict.get(pos).ok_or_else(|| {
                BoltError::Other(format!(
                    "CONCAT_WS: index {} out of range (column {} dictionary size {}) at row {}",
                    idx,
                    col_k,
                    dict.len(),
                    i
                ))
            })?;
            pieces.push(s.as_str());
        }
        // `Vec::<&str>::join` is the cheapest way to glue without a per-row
        // String allocator dance — it pre-sizes the output buffer.
        let joined = pieces.join(separator);
        let idx = intern(&mut new_dict, &mut lookup, joined)?;
        out.push(idx);
    }

    Ok((new_dict, out))
}

// ---------------------------------------------------------------------------
// Public API.
// ---------------------------------------------------------------------------

/// Concatenate two dictionary columns row-by-row.
///
/// Both inputs must have the same `n_rows`; the output preserves it. The
/// output dictionary is built from the cross-product of actually-occurring
/// concatenations, deduplicated. NULL on EITHER side at row `i` yields NULL
/// at row `i` (standard SQL).
///
/// Cost: O(n_rows) host work + O(|new_dict|) for dedup. The output dictionary
/// is at most `|left.dictionary| * |right.dictionary|` entries in the worst
/// case; in practice many concatenations coincide and the dictionary stays
/// small. The caller is responsible for not blowing up the dictionary on
/// pathological cross-products.
pub fn concat(
    left: &DictionaryColumn,
    right: &DictionaryColumn,
) -> BoltResult<DictionaryColumn> {
    if left.n_rows != right.n_rows {
        return Err(BoltError::Other(format!(
            "CONCAT: n_rows mismatch (left = {}, right = {})",
            left.n_rows, right.n_rows
        )));
    }

    let li: Vec<i32> = left.indices.to_vec()?;
    let ri: Vec<i32> = right.indices.to_vec()?;
    let (new_dict, new_indices) =
        concat_pure(&left.dictionary, &li, &right.dictionary, &ri)?;

    let device_indices = GpuVec::<i32>::from_slice(&new_indices)?;
    Ok(DictionaryColumn {
        dictionary: new_dict,
        indices: device_indices,
        n_rows: left.n_rows,
    })
}

/// `SUBSTRING(col, start, length)` over byte indices.
///
/// `start` is 1-indexed (SQL semantics); `start < 1` clamps to `1` (ANSI).
/// `length` is required (no two-arg variant); pass `i32::MAX` for "to the
/// end". Negative `length` is treated as `0`.
///
/// Returns a new dictionary column with `SUBSTRING` applied to each unique
/// dictionary entry, then re-deduplicated — substrings of different originals
/// may coincide ("abc" and "abd" both become "ab"). NULL (index 0) is
/// preserved as NULL. See module docs for UTF-8 boundary behaviour.
pub fn substring(
    input: &DictionaryColumn,
    start: i32,
    length: i32,
) -> BoltResult<DictionaryColumn> {
    let (new_dict, remap) = substring_pure(&input.dictionary, start, length)?;
    let old: Vec<i32> = input.indices.to_vec()?;

    let mut new_indices: Vec<i32> = Vec::with_capacity(old.len());
    for &idx in &old {
        if idx < 0 {
            return Err(BoltError::Other(format!(
                "SUBSTRING: negative dictionary index {} (NULL is encoded as 0)",
                idx
            )));
        }
        let pos = idx as usize;
        let mapped = *remap.get(pos).ok_or_else(|| {
            BoltError::Other(format!(
                "SUBSTRING: index {} out of range (old dictionary size {})",
                idx,
                input.dictionary.len()
            ))
        })?;
        new_indices.push(mapped);
    }

    let device_indices = GpuVec::<i32>::from_slice(&new_indices)?;
    Ok(DictionaryColumn {
        dictionary: new_dict,
        indices: device_indices,
        n_rows: input.n_rows,
    })
}

/// `CONCAT_WS(separator, [col1, col2, ...])` — variadic over a slice of
/// columns. NULLs are SKIPPED (standard CONCAT_WS), not propagated, so an
/// all-NULL row produces the empty string `""` (a real dictionary entry, NOT
/// NULL).
///
/// All input columns must share `n_rows`; the output preserves it. At least
/// one input column is required.
pub fn concat_ws(
    separator: &str,
    columns: &[&DictionaryColumn],
) -> BoltResult<DictionaryColumn> {
    if columns.is_empty() {
        return Err(BoltError::Other(
            "CONCAT_WS: at least one input column is required".into(),
        ));
    }
    let n_rows = columns[0].n_rows;
    for (k, c) in columns.iter().enumerate().skip(1) {
        if c.n_rows != n_rows {
            return Err(BoltError::Other(format!(
                "CONCAT_WS: n_rows mismatch (column 0 = {}, column {} = {})",
                n_rows, k, c.n_rows
            )));
        }
    }

    // Download every index vector once. We keep them alive for the duration
    // of `concat_ws_pure` by binding to a Vec; the `inputs` slice below
    // borrows from these Vecs.
    let downloaded: Vec<Vec<i32>> = columns
        .iter()
        .map(|c| c.indices.to_vec())
        .collect::<BoltResult<_>>()?;
    let inputs: Vec<(&[String], &[i32])> = columns
        .iter()
        .zip(downloaded.iter())
        .map(|(c, d)| (c.dictionary.as_slice(), d.as_slice()))
        .collect();

    let (new_dict, new_indices) = concat_ws_pure(separator, &inputs)?;
    let device_indices = GpuVec::<i32>::from_slice(&new_indices)?;
    Ok(DictionaryColumn {
        dictionary: new_dict,
        indices: device_indices,
        n_rows,
    })
}

// ---------------------------------------------------------------------------
// Tests. As in `string_ops`, the GPU upload/download paths talk to the driver
// and are exercised by the integration suite under `#[ignore]`. Here we test
// the pure helpers exhaustively, since they contain all the actual logic.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn owned(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| s.to_string()).collect()
    }

    // ----- CONCAT --------------------------------------------------------

    #[test]
    fn concat_basic() {
        // Worked example from the spec:
        //   left  = ["us", "eu"]    indices = [1, 2, 1]
        //   right = ["NY", "BERLIN"] indices = [1, 2, 2]
        // Row 0: "us" + "NY"     = "usNY"     -> new idx 1
        // Row 1: "eu" + "BERLIN" = "euBERLIN" -> new idx 2
        // Row 2: "us" + "BERLIN" = "usBERLIN" -> new idx 3
        let l_dict = owned(&["us", "eu"]);
        let l_idx = vec![1, 2, 1];
        let r_dict = owned(&["NY", "BERLIN"]);
        let r_idx = vec![1, 2, 2];

        let (new_dict, new_idx) =
            concat_pure(&l_dict, &l_idx, &r_dict, &r_idx).unwrap();
        assert_eq!(new_dict, vec!["usNY", "euBERLIN", "usBERLIN"]);
        assert_eq!(new_idx, vec![1, 2, 3]);
    }

    #[test]
    fn concat_with_nulls() {
        // NULL on either side -> NULL output. Mixed-NULL rows must NOT pull
        // anything into the output dictionary.
        //   left  dict = ["a"]      indices = [0, 1, 1, 0]
        //   right dict = ["x", "y"] indices = [1, 0, 2, 0]
        // Rows: NULL, NULL, "ay", NULL
        // Expected new_dict = ["ay"]; indices = [0, 0, 1, 0].
        let l_dict = owned(&["a"]);
        let l_idx = vec![0, 1, 1, 0];
        let r_dict = owned(&["x", "y"]);
        let r_idx = vec![1, 0, 2, 0];

        let (new_dict, new_idx) =
            concat_pure(&l_dict, &l_idx, &r_dict, &r_idx).unwrap();
        assert_eq!(new_dict, vec!["ay"]);
        assert_eq!(new_idx, vec![0, 0, 1, 0]);
    }

    #[test]
    fn concat_n_rows_mismatch_errors() {
        let l_dict = owned(&["a"]);
        let r_dict = owned(&["b"]);
        let err = concat_pure(&l_dict, &[1, 1], &r_dict, &[1]).unwrap_err();
        match err {
            BoltError::Other(msg) => assert!(msg.contains("n_rows mismatch")),
            _ => panic!("expected Other(n_rows mismatch), got {:?}", err),
        }
    }

    #[test]
    fn concat_empty_inputs() {
        // 0 rows: empty dictionary, empty indices.
        let l_dict = owned(&["a"]);
        let r_dict = owned(&["b"]);
        let (new_dict, new_idx) =
            concat_pure(&l_dict, &[], &r_dict, &[]).unwrap();
        assert!(new_dict.is_empty());
        assert!(new_idx.is_empty());
    }

    #[test]
    fn concat_all_null_rows_empty_dict() {
        // Every row is NULL on the left -> dictionary stays empty.
        let l_dict = owned(&["a", "b"]);
        let r_dict = owned(&["x"]);
        let (new_dict, new_idx) =
            concat_pure(&l_dict, &[0, 0, 0], &r_dict, &[1, 1, 1]).unwrap();
        assert!(new_dict.is_empty());
        assert_eq!(new_idx, vec![0, 0, 0]);
    }

    #[test]
    fn concat_repeated_rows_collapse_dictionary() {
        // Same row repeated -> single dictionary entry, indices all point at 1.
        let l_dict = owned(&["foo"]);
        let r_dict = owned(&["bar"]);
        let (new_dict, new_idx) =
            concat_pure(&l_dict, &[1, 1, 1, 1], &r_dict, &[1, 1, 1, 1]).unwrap();
        assert_eq!(new_dict, vec!["foobar"]);
        assert_eq!(new_idx, vec![1, 1, 1, 1]);
    }

    #[test]
    fn concat_out_of_range_index_errors() {
        let l_dict = owned(&["a"]);
        let r_dict = owned(&["b"]);
        // left index 2 is out of range for a 1-entry dict.
        let err = concat_pure(&l_dict, &[2], &r_dict, &[1]).unwrap_err();
        match err {
            BoltError::Other(msg) => assert!(msg.contains("out of range")),
            _ => panic!("expected Other(out of range), got {:?}", err),
        }
    }

    #[test]
    fn concat_negative_index_errors() {
        let l_dict = owned(&["a"]);
        let r_dict = owned(&["b"]);
        let err = concat_pure(&l_dict, &[-1], &r_dict, &[1]).unwrap_err();
        match err {
            BoltError::Other(msg) => assert!(msg.contains("negative")),
            _ => panic!("expected Other(negative), got {:?}", err),
        }
    }

    // ----- SUBSTRING -----------------------------------------------------

    #[test]
    fn substring_basic() {
        // input dict = ["hello","world"] indices [1,2,1,2], SUBSTRING(1,3):
        //   "hello"[0..3] = "hel"
        //   "world"[0..3] = "wor"
        // No collapse -> new dict = ["hel","wor"], remap = [0,1,2].
        let dict = owned(&["hello", "world"]);
        let (new_dict, remap) = substring_pure(&dict, 1, 3).unwrap();
        assert_eq!(new_dict, vec!["hel", "wor"]);
        assert_eq!(remap, vec![0, 1, 2]);
    }

    #[test]
    fn substring_collapses_duplicates() {
        // input = ["abc","abd"], SUBSTRING(1,2) -> both "ab".
        let dict = owned(&["abc", "abd"]);
        let (new_dict, remap) = substring_pure(&dict, 1, 2).unwrap();
        assert_eq!(new_dict, vec!["ab"]);
        // Both old entries map to new index 1.
        assert_eq!(remap, vec![0, 1, 1]);
    }

    #[test]
    fn substring_unicode_boundary() {
        // "héllo" is bytes: 'h' | 0xC3 0xA9 ('é') | 'l' 'l' 'o' -> 6 bytes.
        // SUBSTRING(1, 2) requests bytes [0..2]. Byte 2 lands INSIDE the
        // two-byte 'é' codepoint, which is not a char boundary. We round
        // DOWN to byte 1, so the result is "h" — never panics, never produces
        // invalid UTF-8.
        let dict = owned(&["héllo"]);
        let (new_dict, _remap) = substring_pure(&dict, 1, 2).unwrap();
        assert_eq!(new_dict, vec!["h"]);

        // SUBSTRING(1, 3) requests bytes [0..3], which IS a char boundary
        // (right after 'é'), so we get "hé".
        let (new_dict2, _) = substring_pure(&dict, 1, 3).unwrap();
        assert_eq!(new_dict2, vec!["hé"]);
    }

    #[test]
    fn substring_start_clamps_to_one() {
        // ANSI: start < 1 clamps to 1, so SUBSTRING("abc", 0, 2) = "ab",
        // SUBSTRING("abc", -5, 2) = "ab".
        let dict = owned(&["abc"]);
        let (new_dict_0, _) = substring_pure(&dict, 0, 2).unwrap();
        assert_eq!(new_dict_0, vec!["ab"]);
        let (new_dict_neg, _) = substring_pure(&dict, -5, 2).unwrap();
        assert_eq!(new_dict_neg, vec!["ab"]);
    }

    #[test]
    fn substring_zero_length_collapses_to_single_empty() {
        // length = 0 -> every non-NULL row becomes "" (one dictionary entry).
        let dict = owned(&["abc", "xyz", "pqr"]);
        let (new_dict, remap) = substring_pure(&dict, 1, 0).unwrap();
        assert_eq!(new_dict, vec![""]);
        assert_eq!(remap, vec![0, 1, 1, 1]);
    }

    #[test]
    fn substring_negative_length_is_empty() {
        let dict = owned(&["abc"]);
        let (new_dict, remap) = substring_pure(&dict, 1, -7).unwrap();
        assert_eq!(new_dict, vec![""]);
        assert_eq!(remap, vec![0, 1]);
    }

    #[test]
    fn substring_start_past_end_is_empty() {
        // start past string length -> "".
        let dict = owned(&["abc"]);
        let (new_dict, _) = substring_pure(&dict, 10, 3).unwrap();
        assert_eq!(new_dict, vec![""]);
    }

    #[test]
    fn substring_length_saturates_to_end() {
        // Length larger than remaining bytes is clamped to s.len().
        let dict = owned(&["abc"]);
        let (new_dict, _) = substring_pure(&dict, 2, 100).unwrap();
        assert_eq!(new_dict, vec!["bc"]);

        // The "to end" idiom: pass i32::MAX without overflowing.
        let (new_dict_end, _) = substring_pure(&dict, 1, i32::MAX).unwrap();
        assert_eq!(new_dict_end, vec!["abc"]);
    }

    #[test]
    fn substring_empty_dictionary() {
        let (new_dict, remap) = substring_pure(&[], 1, 3).unwrap();
        assert!(new_dict.is_empty());
        assert_eq!(remap, vec![0]); // just the NULL passthrough
    }

    #[test]
    fn substring_empty_string_in_dictionary() {
        // An empty input string substrings to an empty output string and
        // stays at the same dictionary position.
        let dict = owned(&["", "abc"]);
        let (new_dict, remap) = substring_pure(&dict, 1, 2).unwrap();
        assert_eq!(new_dict, vec!["", "ab"]);
        assert_eq!(remap, vec![0, 1, 2]);
    }

    // ----- CONCAT_WS -----------------------------------------------------

    #[test]
    fn concat_ws_skips_nulls() {
        // a has NULL at row 0 ("x" otherwise); b is "x" at row 0.
        //   a dict = ["foo"] idx = [0, 1]
        //   b dict = ["x"]   idx = [1, 1]
        // Row 0: a NULL, b "x" -> "x" (separator suppressed when only one side
        //        contributes — that's the standard CONCAT_WS behaviour).
        // Row 1: a "foo", b "x" -> "foo-x".
        let a = owned(&["foo"]);
        let b = owned(&["x"]);
        let (new_dict, new_idx) = concat_ws_pure(
            "-",
            &[(&a, &[0, 1]), (&b, &[1, 1])],
        )
        .unwrap();
        assert_eq!(new_dict, vec!["x", "foo-x"]);
        assert_eq!(new_idx, vec![1, 2]);
    }

    #[test]
    fn concat_ws_all_null_row_is_empty_string() {
        // All-NULL row -> empty string, which is a real (non-NULL) dictionary
        // entry. This is the documented CONCAT_WS divergence from CONCAT.
        let a = owned(&["foo"]);
        let b = owned(&["bar"]);
        let (new_dict, new_idx) = concat_ws_pure(
            ",",
            &[(&a, &[0, 1]), (&b, &[0, 1])],
        )
        .unwrap();
        // Row 0: both NULL -> "" -> new idx 1.
        // Row 1: "foo,bar" -> new idx 2.
        assert_eq!(new_dict, vec!["", "foo,bar"]);
        assert_eq!(new_idx, vec![1, 2]);
    }

    #[test]
    fn concat_ws_three_columns() {
        let a = owned(&["a"]);
        let b = owned(&["b"]);
        let c = owned(&["c"]);
        let (new_dict, new_idx) = concat_ws_pure(
            "/",
            &[(&a, &[1, 1]), (&b, &[1, 0]), (&c, &[1, 1])],
        )
        .unwrap();
        // Row 0: "a/b/c"; Row 1: skip NULL b -> "a/c".
        assert_eq!(new_dict, vec!["a/b/c", "a/c"]);
        assert_eq!(new_idx, vec![1, 2]);
    }

    #[test]
    fn concat_ws_empty_separator_acts_like_concat() {
        let a = owned(&["us"]);
        let b = owned(&["NY"]);
        let (new_dict, new_idx) =
            concat_ws_pure("", &[(&a, &[1]), (&b, &[1])]).unwrap();
        assert_eq!(new_dict, vec!["usNY"]);
        assert_eq!(new_idx, vec![1]);
    }

    #[test]
    fn concat_ws_zero_rows() {
        let a = owned(&["a"]);
        let b = owned(&["b"]);
        let (new_dict, new_idx) =
            concat_ws_pure("-", &[(&a, &[]), (&b, &[])]).unwrap();
        assert!(new_dict.is_empty());
        assert!(new_idx.is_empty());
    }

    #[test]
    fn concat_ws_n_rows_mismatch_errors() {
        let a = owned(&["a"]);
        let b = owned(&["b"]);
        let err = concat_ws_pure("-", &[(&a, &[1, 1]), (&b, &[1])]).unwrap_err();
        match err {
            BoltError::Other(msg) => assert!(msg.contains("n_rows mismatch")),
            _ => panic!("expected Other(n_rows mismatch), got {:?}", err),
        }
    }

    #[test]
    fn concat_ws_empty_input_list_errors() {
        let err = concat_ws_pure("-", &[]).unwrap_err();
        match err {
            BoltError::Other(msg) => {
                assert!(msg.contains("at least one input column"))
            }
            _ => panic!("expected Other(at least one input), got {:?}", err),
        }
    }

    // ----- sql_substring helper directly --------------------------------

    #[test]
    fn sql_substring_basic_ascii() {
        assert_eq!(sql_substring("hello", 1, 3), "hel");
        assert_eq!(sql_substring("hello", 2, 3), "ell");
        assert_eq!(sql_substring("hello", 5, 1), "o");
    }

    #[test]
    fn sql_substring_unicode_round_down_at_start() {
        // SUBSTRING("héllo", 2, 4): byte_start = 1 (valid: 'h' boundary),
        // byte_end = 5 (boundary right after second 'l'). 'é' is two bytes
        // starting at byte 1, so the slice yields bytes [1..5] = "éll".
        assert_eq!(sql_substring("héllo", 2, 4), "éll");

        // SUBSTRING("héllo", 3, 2): byte_start_raw = 2 — INSIDE the two-byte
        // 'é'. We round down to byte 1. byte_end_raw = 4, which IS a char
        // boundary (right before the second 'l'). After rounding, start = 1,
        // end = 4 -> bytes [é, l] -> "él". (Documented behaviour: rounding
        // down can recover bytes left of the requested start.)
        assert_eq!(sql_substring("héllo", 3, 2), "él");
    }

    #[test]
    fn sql_substring_handles_i32_max_length() {
        // i32::MAX as usize is huge but never wraps because we add it via
        // saturating_add and clamp to s.len().
        assert_eq!(sql_substring("abc", 1, i32::MAX), "abc");
    }

    #[test]
    fn substring_str_public_wrapper_matches_internal() {
        assert_eq!(substring_str("hello", 2, 3), "ell");
        assert_eq!(substring_str("héllo", 1, 3), "hé");
    }

    // ----- TRIM ----------------------------------------------------------

    #[test]
    fn trim_default_whitespace_both() {
        assert_eq!(trim_str("  hi  ", TrimSide::Both, None), "hi");
        assert_eq!(trim_str("\t\nhi \n", TrimSide::Both, None), "hi");
        assert_eq!(trim_str("nospace", TrimSide::Both, None), "nospace");
    }

    #[test]
    fn trim_leading_and_trailing_whitespace() {
        assert_eq!(trim_str("  hi  ", TrimSide::Leading, None), "hi  ");
        assert_eq!(trim_str("  hi  ", TrimSide::Trailing, None), "  hi");
    }

    #[test]
    fn trim_custom_chars_is_a_char_set_not_substring() {
        // TRIM('xy' FROM ...) strips any leading/trailing 'x' or 'y'.
        assert_eq!(trim_str("xyxabcyx", TrimSide::Both, Some("xy")), "abc");
        assert_eq!(trim_str("xyxabcyx", TrimSide::Leading, Some("xy")), "abcyx");
        assert_eq!(trim_str("xyxabcyx", TrimSide::Trailing, Some("xy")), "xyxabc");
    }

    #[test]
    fn trim_custom_chars_single_char() {
        assert_eq!(trim_str("---val---", TrimSide::Both, Some("-")), "val");
    }

    #[test]
    fn trim_empty_char_set_strips_nothing() {
        assert_eq!(trim_str("  hi  ", TrimSide::Both, Some("")), "  hi  ");
    }

    #[test]
    fn trim_all_chars_collapses_to_empty() {
        assert_eq!(trim_str("aaaa", TrimSide::Both, Some("a")), "");
        assert_eq!(trim_str("   ", TrimSide::Both, None), "");
    }

    #[test]
    fn trim_unicode_chars() {
        // Multi-byte trim character.
        assert_eq!(trim_str("→→go→", TrimSide::Both, Some("→")), "go");
        // Multi-byte content preserved.
        assert_eq!(trim_str("  héllo  ", TrimSide::Both, None), "héllo");
    }
}
