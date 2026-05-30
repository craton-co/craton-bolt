// SPDX-License-Identifier: Apache-2.0

//! Dictionary-aware string operations: `UPPER`, `LOWER`, `LENGTH`, and
//! literal-equality on `DictionaryColumn`.
//!
//! ## Why this lives on the host
//!
//! Variable-width string writes on a GPU kernel are painful: producing
//! `"hello"` → `"HELLO"` for each row would force per-row offset bookkeeping
//! and broken coalesced writes. The dictionary-encoded layout sidesteps the
//! whole problem:
//!
//! * **UPPER / LOWER** are a pure transform of the *dictionary*. The device
//!   indices stay aligned to (post-remap) entries of the new dictionary; we
//!   download i32 indices, remap them on the host, and re-upload. No kernel
//!   launch, no variable-width device writes.
//!
//! * **LENGTH** is a per-index lookup over a tiny `lengths_table`. Same
//!   download → host map pattern, returned as an `Int64Array` (the SQL
//!   `LENGTH → Int64` contract) whose validity bitmap carries SQL `NULL` for
//!   NULL rows.
//!
//! * **input_eq_literal** lifts a `col = 'literal'` predicate that the
//!   codegen path was unable to fuse: one O(dict) lookup followed by
//!   pointwise i32-equality on the downloaded indices.
//!
//! ## Caveats and v1 scope
//!
//! * No `CONCAT` — would require variable-width device writes (or a host-side
//!   materialise that defeats the whole dictionary trick when the two inputs
//!   live on the device).
//! * No `SUBSTRING` — same reason. Could be added as a pure dictionary
//!   transform later (substring is closed under dictionary remap) but is out
//!   of scope here.
//! * No regex.
//! * `LENGTH(NULL)` returns SQL `NULL` (carried on the result `Int64Array`'s
//!   validity bitmap), distinct from `LENGTH('') = 0`.
//! * `LENGTH` counts CHARACTERS (Unicode codepoints), matching SQL
//!   `LENGTH` / `CHAR_LENGTH`: `LENGTH('héllo') = 5`. Byte length (Arrow
//!   `Utf8` semantics) is available as `OCTET_LENGTH` via
//!   `octet_lengths_table_pure`.

use std::collections::HashMap;

use arrow_array::{Array, BooleanArray, Int64Array, StringArray};

use crate::cuda::dictionary::DictionaryColumn;
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};

// ---------------------------------------------------------------------------
// Pure helpers (no GPU). These exist so the transformation logic can be unit
// tested without a CUDA runtime. The public `upper` / `lower` / `length`
// functions wrap them with the download → remap → upload plumbing.
// ---------------------------------------------------------------------------

/// Re-deduplicate a transformed dictionary and produce the index remap.
///
/// Given `old_dict[i]` already transformed (e.g. uppercased) into `t[i]`,
/// returns `(new_dict, remap)` where:
///
/// * `new_dict` is `t` with duplicates collapsed in first-occurrence order.
/// * `remap` is sized `old_dict.len() + 1`. Slot `0` is `0` (NULL passes
///   through unchanged). Slot `i + 1` maps the old i32 index `i + 1`
///   (i.e. the index that pointed at `old_dict[i]`) to the new i32 index
///   pointing at the same string in `new_dict`.
///
/// Errors if the deduplicated dictionary would exceed `i32::MAX` entries —
/// impossible in practice (input dictionary was already i32-bounded and
/// the transform can only shrink the unique count), but we surface it
/// rather than silently truncating.
fn dedup_transformed(transformed: Vec<String>) -> BoltResult<(Vec<String>, Vec<i32>)> {
    let n_old = transformed.len();
    let mut new_dict: Vec<String> = Vec::new();
    // Borrow-friendly map: key is owned so we can clone-on-insert and avoid
    // tangling with `new_dict`'s growing borrow.
    let mut lookup: HashMap<String, i32> = HashMap::new();
    // `remap[0]` is NULL → NULL; rest filled in below.
    let mut remap: Vec<i32> = vec![0; n_old + 1];

    for (i, s) in transformed.into_iter().enumerate() {
        let new_idx = if let Some(&idx) = lookup.get(&s) {
            idx
        } else {
            // Reserve slot before pushing so we surface overflow before the
            // dictionary grows past the i32 index space.
            let next_len = new_dict.len().checked_add(1).ok_or_else(|| {
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
            new_dict.push(s.clone());
            lookup.insert(s, idx);
            idx
        };
        remap[i + 1] = new_idx;
    }

    Ok((new_dict, remap))
}

/// Pure-host implementation of `UPPER` over a dictionary slice. Returns the
/// `(new_dict, remap_table)` pair. See [`dedup_transformed`] for layout.
fn upper_dict_pure(old_dict: &[String]) -> BoltResult<(Vec<String>, Vec<i32>)> {
    let transformed: Vec<String> = old_dict.iter().map(|s| s.to_uppercase()).collect();
    dedup_transformed(transformed)
}

/// Pure-host implementation of `LOWER` over a dictionary slice.
fn lower_dict_pure(old_dict: &[String]) -> BoltResult<(Vec<String>, Vec<i32>)> {
    let transformed: Vec<String> = old_dict.iter().map(|s| s.to_lowercase()).collect();
    dedup_transformed(transformed)
}

/// Build the CHARACTER-length lookup table for SQL `LENGTH` / `CHAR_LENGTH`.
///
/// SQL `LENGTH` counts characters (Unicode codepoints), not bytes — so
/// `LENGTH('héllo') = 5`, not `6`. Byte length is available separately via
/// [`octet_lengths_table_pure`] (`OCTET_LENGTH`).
///
/// `out[0] = 0` (NULL sentinel slot — see [`length`] for how the NULL row is
/// surfaced as SQL `NULL` rather than `0`); `out[k] = char_count(old_dict[k-1])`
/// for `k` in `1..=old_dict.len()`.
///
/// Errors if any individual string's character count exceeds `i32::MAX`.
fn lengths_table_pure(old_dict: &[String]) -> BoltResult<Vec<i32>> {
    let mut out: Vec<i32> = Vec::with_capacity(old_dict.len() + 1);
    out.push(0); // NULL slot
    for s in old_dict {
        let len = s.chars().count();
        if len > i32::MAX as usize {
            return Err(BoltError::Other(format!(
                "LENGTH: string of {} characters exceeds i32::MAX",
                len
            )));
        }
        out.push(len as i32);
    }
    Ok(out)
}

/// Build the BYTE-length lookup table for `OCTET_LENGTH`.
///
/// Same layout as [`lengths_table_pure`] (slot 0 = NULL sentinel) but each slot
/// holds the UTF-8 byte length (`s.len()`), not the character count. Kept
/// available for any path that needs Arrow `Utf8` byte length; SQL `LENGTH`
/// itself uses the character-count table.
///
/// Errors if any individual string's byte length exceeds `i32::MAX` — an absurd
/// 2 GiB single value.
fn octet_lengths_table_pure(old_dict: &[String]) -> BoltResult<Vec<i32>> {
    let mut out: Vec<i32> = Vec::with_capacity(old_dict.len() + 1);
    out.push(0); // NULL slot
    for s in old_dict {
        let len = s.len();
        if len > i32::MAX as usize {
            return Err(BoltError::Other(format!(
                "OCTET_LENGTH: string of {} bytes exceeds i32::MAX",
                len
            )));
        }
        out.push(len as i32);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API.
// ---------------------------------------------------------------------------

/// Apply `UPPER` to a dictionary column.
///
/// The output column has a *re-deduplicated* dictionary — uppercasing can
/// collapse distinct inputs (e.g. `"us"` and `"US"` both → `"US"`) — so the
/// device indices are rewritten host-side and re-uploaded as a fresh
/// `DictionaryColumn`. `n_rows` is preserved; NULL (index `0`) is preserved.
///
/// Cost: one device→host copy of `n_rows` i32s, one host→device copy of the
/// same. No kernel launch.
pub fn upper(input: &DictionaryColumn) -> BoltResult<DictionaryColumn> {
    let (new_dict, remap) = upper_dict_pure(&input.dictionary)?;
    remap_and_upload(input, new_dict, &remap)
}

/// Apply `LOWER` to a dictionary column. See [`upper`] — same shape, just
/// `to_lowercase` instead of `to_uppercase`.
pub fn lower(input: &DictionaryColumn) -> BoltResult<DictionaryColumn> {
    let (new_dict, remap) = lower_dict_pure(&input.dictionary)?;
    remap_and_upload(input, new_dict, &remap)
}

/// Compute the SQL `LENGTH` (character count) of each row's string as `Int64`.
///
/// Returns an [`Int64Array`] (the SQL `LENGTH → Int64` type contract). NULL
/// rows (dictionary index `0`) yield SQL `NULL` — carried on the returned
/// array's validity bitmap — and are therefore *distinct* from `LENGTH('') = 0`
/// (a non-NULL empty string at a non-zero index). `LENGTH` counts characters,
/// not bytes: `LENGTH('héllo') = 5`.
///
/// Cost: one device→host copy of `n_rows` i32s + an O(dict) table build.
pub fn length(input: &DictionaryColumn) -> BoltResult<Int64Array> {
    let table = lengths_table_pure(&input.dictionary)?;
    let indices: Vec<i32> = input.indices.to_vec()?;
    length_from_indices(&indices, &table, input.dictionary.len())
}

/// Pure-host core of [`length`]: map dictionary `indices` through the
/// (character-)length `table` into a validity-carrying [`Int64Array`].
///
/// Index `0` → SQL `NULL` (validity bit cleared), distinct from `LENGTH('')=0`.
/// Lives as a free function so the NULL / Int64 / 3VL behaviour is unit-testable
/// without a CUDA device (constructing a `DictionaryColumn` would upload to the
/// driver). `dict_len` is only used to phrase the out-of-range error.
fn length_from_indices(
    indices: &[i32],
    table: &[i32],
    dict_len: usize,
) -> BoltResult<Int64Array> {
    // `Option<i64>`: index 0 → NULL (SQL `LENGTH(NULL) = NULL`); otherwise the
    // character count from the table. Building from `Vec<Option<i64>>` gives the
    // result array a validity bitmap so a downstream `IS NULL` / `> 0` sees the
    // correct 3VL.
    let mut lens: Vec<Option<i64>> = Vec::with_capacity(indices.len());
    for &idx in indices {
        // A negative or out-of-range index would mean a kernel wrote
        // something the dictionary cannot decode. Mirror the strictness of
        // `DictionaryColumn::to_string_array`.
        if idx < 0 {
            return Err(BoltError::Other(format!(
                "LENGTH: negative dictionary index {} (NULL is encoded as 0)",
                idx
            )));
        }
        if idx == 0 {
            // NULL row → SQL NULL (not 0).
            lens.push(None);
            continue;
        }
        let pos = idx as usize;
        let len = *table.get(pos).ok_or_else(|| {
            BoltError::Other(format!(
                "LENGTH: index {} out of range (dictionary size {})",
                idx, dict_len
            ))
        })?;
        lens.push(Some(len as i64));
    }

    Ok(Int64Array::from(lens))
}

/// Predicate: `col = literal` evaluated host-side on a dictionary column.
///
/// Used by the predicate-rewrite path when the equality could not be pushed
/// into the fused codegen kernel. Walks the dictionary once for the literal
/// lookup, then translates indices to booleans on the host.
///
/// SQL three-valued logic: a NULL row (dictionary index `0`) evaluates to SQL
/// `NULL`, NOT `false` — so the returned [`BooleanArray`] carries a validity
/// bitmap with NULL at those rows (mirroring [`crate::exec::like::host_like`]).
/// This keeps `NOT (NULL = 'x')`, `OR`, and projected-boolean compositions
/// 3VL-correct rather than collapsing NULL to `false`. A non-NULL row is
/// `true` iff its index equals the literal's, else `false`.
///
/// If the literal is absent from the dictionary it matches no non-NULL row, but
/// NULL rows still evaluate to NULL (so the result is a mix of `false`/`NULL`,
/// not all-`false`). Returns an `n_rows`-length array even when the input is
/// empty.
pub fn input_eq_literal(
    input: &DictionaryColumn,
    literal: &str,
) -> BoltResult<BooleanArray> {
    let indices: Vec<i32> = input.indices.to_vec()?;
    // `target` is `None` when the literal is not in the dictionary: no non-NULL
    // row can equal it, but NULL rows still propagate NULL under 3VL.
    let target = input.index_of(literal);
    Ok(eq_literal_from_indices(&indices, target))
}

/// Pure-host core of [`input_eq_literal`]: map dictionary `indices` to a
/// 3VL-correct [`BooleanArray`] against the literal's dictionary index `target`
/// (`None` ⇒ literal absent from the dictionary).
///
/// Index `0` (NULL row) → SQL `NULL` (validity cleared), NEVER `false`. A
/// non-NULL row is `true` iff its index equals `target`. Free function so the
/// 3VL behaviour is unit-testable without a CUDA device.
fn eq_literal_from_indices(indices: &[i32], target: Option<i32>) -> BooleanArray {
    let vals: Vec<Option<bool>> = indices
        .iter()
        .map(|&i| {
            if i == 0 {
                // NULL row → SQL NULL (3VL), never false.
                None
            } else {
                Some(Some(i) == target)
            }
        })
        .collect();
    BooleanArray::from(vals)
}

// ---------------------------------------------------------------------------
// Internal: download → remap → upload, shared by `upper` and `lower`.
// ---------------------------------------------------------------------------

/// Apply `remap` to `input.indices` host-side and upload the result as a new
/// `DictionaryColumn` paired with `new_dict`.
fn remap_and_upload(
    input: &DictionaryColumn,
    new_dict: Vec<String>,
    remap: &[i32],
) -> BoltResult<DictionaryColumn> {
    let old: Vec<i32> = input.indices.to_vec()?;

    let mut new_indices: Vec<i32> = Vec::with_capacity(old.len());
    for &idx in &old {
        // Same strict bounds check as decode: negative or out-of-range is a
        // kernel-side bug we'd rather surface than mask.
        if idx < 0 {
            return Err(BoltError::Other(format!(
                "string_ops: negative dictionary index {} (NULL is encoded as 0)",
                idx
            )));
        }
        let pos = idx as usize;
        let mapped = *remap.get(pos).ok_or_else(|| {
            BoltError::Other(format!(
                "string_ops: index {} out of range (old dictionary size {})",
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

// ---------------------------------------------------------------------------
// SQL `||` (StringConcat) — host-side helper over Arrow `StringArray`s.
//
// `string_ops_extended::concat` operates on `DictionaryColumn`s (device-side
// inputs / outputs) and is the right tool when the engine is already holding
// dictionary-encoded columns. The v0.5 SQL frontend lowers `a || b` to
// `BinaryOp::Concat`, which the executor's host-side projection path
// evaluates over `StringArray` cells lifted from the input batch — see
// `exec::expr_agg::eval_binary` / `exec::engine.rs::execute` Project arm.
//
// Semantics: NULL on either side propagates as NULL (standard SQL).
// ---------------------------------------------------------------------------

/// Concatenate two Arrow `StringArray`s row-by-row.
///
/// Returns a new `StringArray` with the same number of rows as both inputs.
/// NULL on either side at row `i` yields NULL at row `i` in the output
/// (standard SQL). Errors if the two inputs have different lengths.
pub fn host_concat_strings(left: &StringArray, right: &StringArray) -> BoltResult<StringArray> {
    if left.len() != right.len() {
        return Err(BoltError::Other(format!(
            "CONCAT (||): row count mismatch (left = {}, right = {})",
            left.len(),
            right.len()
        )));
    }
    let n = left.len();
    let mut out: Vec<Option<String>> = Vec::with_capacity(n);
    for i in 0..n {
        if left.is_null(i) || right.is_null(i) {
            out.push(None);
        } else {
            let mut s = String::with_capacity(left.value(i).len() + right.value(i).len());
            s.push_str(left.value(i));
            s.push_str(right.value(i));
            out.push(Some(s));
        }
    }
    Ok(StringArray::from(out))
}

/// Pure-host helper used by `crate::exec::expr_agg::eval_binary` to fold
/// `BinaryOp::Concat` over two `Vec<Option<String>>` columns. NULL on either
/// side propagates as NULL.
pub fn host_concat_option_strings(
    left: &[Option<String>],
    right: &[Option<String>],
) -> BoltResult<Vec<Option<String>>> {
    if left.len() != right.len() {
        return Err(BoltError::Other(format!(
            "CONCAT (||): row count mismatch (left = {}, right = {})",
            left.len(),
            right.len()
        )));
    }
    let out = left
        .iter()
        .zip(right.iter())
        .map(|(l, r)| match (l, r) {
            (Some(l), Some(r)) => {
                let mut s = String::with_capacity(l.len() + r.len());
                s.push_str(l);
                s.push_str(r);
                Some(s)
            }
            _ => None,
        })
        .collect();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests. Live round-trip tests would need a CUDA runtime (the GpuVec
// upload/download paths talk to the driver), so we instead exhaustively test
// the pure helpers that contain all the actual logic. Anything that *does*
// need the device is exercised by the integration suite under `#[ignore]`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn owned(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn upper_collapses_duplicates() {
        // The worked example from the spec: ["us", "EU", "US", "ca"]
        //   uppercased: ["US", "EU", "US", "CA"]
        //   dedup:      new_dict = ["US", "EU", "CA"]
        //   remap (index 0 = NULL passthrough):
        //     old 1 ("us") -> new 1 ("US")
        //     old 2 ("EU") -> new 2 ("EU")
        //     old 3 ("US") -> new 1 ("US")  <- collapse
        //     old 4 ("ca") -> new 3 ("CA")
        let old = owned(&["us", "EU", "US", "ca"]);
        let (new_dict, remap) = upper_dict_pure(&old).unwrap();
        assert_eq!(new_dict, vec!["US", "EU", "CA"]);
        assert_eq!(remap, vec![0, 1, 2, 1, 3]);
    }

    #[test]
    fn lower_collapses_duplicates() {
        let old = owned(&["US", "eu", "us", "CA"]);
        let (new_dict, remap) = lower_dict_pure(&old).unwrap();
        assert_eq!(new_dict, vec!["us", "eu", "ca"]);
        assert_eq!(remap, vec![0, 1, 2, 1, 3]);
    }

    #[test]
    fn upper_no_collapse_is_identity_remap() {
        // No casing collisions → 1:1 remap, same length dictionary.
        let old = owned(&["alpha", "beta", "gamma"]);
        let (new_dict, remap) = upper_dict_pure(&old).unwrap();
        assert_eq!(new_dict, vec!["ALPHA", "BETA", "GAMMA"]);
        assert_eq!(remap, vec![0, 1, 2, 3]);
    }

    #[test]
    fn upper_empty_dictionary() {
        let (new_dict, remap) = upper_dict_pure(&[]).unwrap();
        assert!(new_dict.is_empty());
        // Only the NULL slot.
        assert_eq!(remap, vec![0]);
    }

    #[test]
    fn upper_preserves_first_occurrence_order() {
        // "z" appears before "A" → its uppercase "Z" must come first.
        let old = owned(&["z", "A", "Z", "a"]);
        let (new_dict, remap) = upper_dict_pure(&old).unwrap();
        assert_eq!(new_dict, vec!["Z", "A"]);
        // old 1 "z" -> "Z" (new 1)
        // old 2 "A" -> "A" (new 2)
        // old 3 "Z" -> "Z" (new 1, collapse)
        // old 4 "a" -> "A" (new 2, collapse)
        assert_eq!(remap, vec![0, 1, 2, 1, 2]);
    }

    #[test]
    fn upper_unicode_lowercase_collapses() {
        // Greek sigma has two lowercase forms but one uppercase. Verify the
        // transform deduplicates them under UPPER.
        let old = owned(&["σ", "ς"]); // medial + final lowercase sigma
        let (new_dict, remap) = upper_dict_pure(&old).unwrap();
        assert_eq!(new_dict, vec!["Σ"]);
        assert_eq!(remap, vec![0, 1, 1]);
    }

    #[test]
    fn lengths_table_basic() {
        let old = owned(&["a", "bb", "ccc"]);
        let table = lengths_table_pure(&old).unwrap();
        // [NULL=0, "a"=1, "bb"=2, "ccc"=3]
        assert_eq!(table, vec![0, 1, 2, 3]);
    }

    #[test]
    fn lengths_table_counts_characters_not_bytes() {
        // SQL LENGTH counts characters: "é" is one character (two UTF-8 bytes),
        // "héllo" is five characters (six bytes), "日本語" is three characters
        // (nine bytes). The character table must report 1 / 5 / 3.
        let old = owned(&["é", "héllo", "日本語"]);
        let table = lengths_table_pure(&old).unwrap();
        assert_eq!(table, vec![0, 1, 5, 3]);
    }

    #[test]
    fn octet_lengths_table_counts_bytes() {
        // OCTET_LENGTH keeps the byte semantics: "é"=2, "héllo"=6, "日本語"=9.
        let old = owned(&["é", "héllo", "日本語"]);
        let table = octet_lengths_table_pure(&old).unwrap();
        assert_eq!(table, vec![0, 2, 6, 9]);
    }

    #[test]
    fn lengths_table_empty_dictionary() {
        let table = lengths_table_pure(&[]).unwrap();
        assert_eq!(table, vec![0]); // just the NULL slot
    }

    #[test]
    fn lengths_table_includes_empty_string() {
        // An empty string is a perfectly valid dictionary entry distinct from
        // NULL: it has length 0 but lives at index 1, not 0.
        let old = owned(&["", "x"]);
        let table = lengths_table_pure(&old).unwrap();
        assert_eq!(table, vec![0, 0, 1]);
    }

    #[test]
    fn dedup_transformed_all_same_collapses_to_one() {
        let t = owned(&["X", "X", "X", "X"]);
        let (new_dict, remap) = dedup_transformed(t).unwrap();
        assert_eq!(new_dict, vec!["X"]);
        assert_eq!(remap, vec![0, 1, 1, 1, 1]);
    }

    #[test]
    fn host_concat_strings_basic() {
        let l = StringArray::from(vec!["a", "b", "c"]);
        let r = StringArray::from(vec!["1", "2", "3"]);
        let out = host_concat_strings(&l, &r).unwrap();
        let got: Vec<&str> = (0..out.len()).map(|i| out.value(i)).collect();
        assert_eq!(got, vec!["a1", "b2", "c3"]);
    }

    #[test]
    fn host_concat_strings_propagates_nulls() {
        let l = StringArray::from(vec![Some("a"), None, Some("c")]);
        let r = StringArray::from(vec![Some("x"), Some("y"), None]);
        let out = host_concat_strings(&l, &r).unwrap();
        assert!(!out.is_null(0));
        assert_eq!(out.value(0), "ax");
        assert!(out.is_null(1));
        assert!(out.is_null(2));
    }

    #[test]
    fn host_concat_strings_length_mismatch_errors() {
        let l = StringArray::from(vec!["a", "b"]);
        let r = StringArray::from(vec!["x"]);
        let err = host_concat_strings(&l, &r).unwrap_err();
        match err {
            BoltError::Other(msg) => assert!(msg.contains("row count mismatch")),
            _ => panic!("expected Other(row count mismatch), got {:?}", err),
        }
    }

    // ----- LENGTH: NULL / Int64 / character semantics --------------------

    #[test]
    fn length_returns_int64_array() {
        // F-8: SQL LENGTH is Int64. The result array's dtype must be Int64.
        let table = lengths_table_pure(&owned(&["a", "bb"])).unwrap();
        let out = length_from_indices(&[1, 2, 1], &table, 2).unwrap();
        assert_eq!(out.data_type(), &arrow_schema::DataType::Int64);
        let got: Vec<i64> = (0..out.len()).map(|i| out.value(i)).collect();
        assert_eq!(got, vec![1, 2, 1]);
    }

    #[test]
    fn length_of_null_is_sql_null_not_zero() {
        // F-3: LENGTH(NULL) must be SQL NULL, distinct from LENGTH('') = 0.
        // dict = ["", "x"]; index 0 = NULL, index 1 = "", index 2 = "x".
        let table = lengths_table_pure(&owned(&["", "x"])).unwrap();
        // rows: NULL, "", "x".
        let out = length_from_indices(&[0, 1, 2], &table, 2).unwrap();
        assert!(out.is_null(0), "LENGTH(NULL) must be NULL");
        // The empty string is non-NULL with length 0 — distinguishable from NULL.
        assert!(!out.is_null(1));
        assert_eq!(out.value(1), 0);
        assert!(!out.is_null(2));
        assert_eq!(out.value(2), 1);
        assert_eq!(out.null_count(), 1);
    }

    #[test]
    fn length_counts_characters() {
        // F-2: LENGTH('héllo') = 5 characters (not 6 bytes).
        let table = lengths_table_pure(&owned(&["héllo"])).unwrap();
        let out = length_from_indices(&[1], &table, 1).unwrap();
        assert_eq!(out.value(0), 5);
    }

    #[test]
    fn length_rejects_negative_index() {
        let table = lengths_table_pure(&owned(&["a"])).unwrap();
        let err = length_from_indices(&[1, -1], &table, 1).unwrap_err();
        match err {
            BoltError::Other(msg) => assert!(msg.contains("negative")),
            _ => panic!("expected Other(negative), got {:?}", err),
        }
    }

    // ----- input_eq_literal: 3VL ----------------------------------------

    #[test]
    fn eq_literal_null_row_is_sql_null() {
        // F-5: a NULL row (index 0) must evaluate to SQL NULL under `= 'lit'`,
        // not false. dict literal lives at index 1; rows: NULL, "lit", other.
        let arr = eq_literal_from_indices(&[0, 1, 2], Some(1));
        assert!(arr.is_null(0), "NULL = 'lit' must be NULL, not false");
        assert!(!arr.is_null(1));
        assert!(arr.value(1), "match → true");
        assert!(!arr.is_null(2));
        assert!(!arr.value(2), "non-match → false");
    }

    #[test]
    fn eq_literal_absent_literal_still_nulls_null_rows() {
        // Literal not in the dictionary (target None): no non-NULL row matches,
        // but NULL rows must still be NULL (not collapsed to all-false).
        let arr = eq_literal_from_indices(&[0, 1, 2], None);
        assert!(arr.is_null(0));
        assert!(!arr.is_null(1));
        assert!(!arr.value(1));
        assert!(!arr.is_null(2));
        assert!(!arr.value(2));
    }

    #[test]
    fn host_concat_option_strings_basic() {
        let l: Vec<Option<String>> =
            vec![Some("a".into()), None, Some("c".into())];
        let r: Vec<Option<String>> =
            vec![Some("x".into()), Some("y".into()), None];
        let out = host_concat_option_strings(&l, &r).unwrap();
        assert_eq!(out[0], Some("ax".into()));
        assert_eq!(out[1], None);
        assert_eq!(out[2], None);
    }
}
