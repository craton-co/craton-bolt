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
//!   download → host map → upload-as-`Int32Array` pattern.
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
//! * `LENGTH(NULL)` returns `0`, NOT SQL `NULL`. Real SQL semantics demand
//!   `NULL`; surfacing that requires a validity bitmap the surrounding
//!   pipeline does not yet plumb through. Documented; revisit when null
//!   tracking lands.
//! * `LENGTH` is byte length (Arrow `Utf8` semantics), not character count.
//!   A future `CHAR_LENGTH` would walk graphemes per dictionary entry.

use std::collections::HashMap;

use arrow_array::{BooleanArray, Int32Array};

use crate::cuda::dictionary::DictionaryColumn;
use crate::cuda::GpuVec;
use crate::error::{PatinaError, PatinaResult};

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
fn dedup_transformed(transformed: Vec<String>) -> PatinaResult<(Vec<String>, Vec<i32>)> {
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
                PatinaError::Other(
                    "dictionary overflow: more than usize::MAX unique strings".into(),
                )
            })?;
            if next_len > i32::MAX as usize {
                return Err(PatinaError::Other(format!(
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
fn upper_dict_pure(old_dict: &[String]) -> PatinaResult<(Vec<String>, Vec<i32>)> {
    let transformed: Vec<String> = old_dict.iter().map(|s| s.to_uppercase()).collect();
    dedup_transformed(transformed)
}

/// Pure-host implementation of `LOWER` over a dictionary slice.
fn lower_dict_pure(old_dict: &[String]) -> PatinaResult<(Vec<String>, Vec<i32>)> {
    let transformed: Vec<String> = old_dict.iter().map(|s| s.to_lowercase()).collect();
    dedup_transformed(transformed)
}

/// Build the byte-length lookup table for `LENGTH`.
///
/// `out[0] = 0` (NULL → 0, per the v1 caveat in the module docs);
/// `out[k] = old_dict[k - 1].len() as i32` for `k` in `1..=old_dict.len()`.
///
/// Errors if any individual string's byte length exceeds `i32::MAX` — which
/// would also be an absurd 2 GiB single value.
fn lengths_table_pure(old_dict: &[String]) -> PatinaResult<Vec<i32>> {
    let mut out: Vec<i32> = Vec::with_capacity(old_dict.len() + 1);
    out.push(0); // NULL slot
    for s in old_dict {
        let len = s.len();
        if len > i32::MAX as usize {
            return Err(PatinaError::Other(format!(
                "LENGTH: string of {} bytes exceeds i32::MAX",
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
pub fn upper(input: &DictionaryColumn) -> PatinaResult<DictionaryColumn> {
    let (new_dict, remap) = upper_dict_pure(&input.dictionary)?;
    remap_and_upload(input, new_dict, &remap)
}

/// Apply `LOWER` to a dictionary column. See [`upper`] — same shape, just
/// `to_lowercase` instead of `to_uppercase`.
pub fn lower(input: &DictionaryColumn) -> PatinaResult<DictionaryColumn> {
    let (new_dict, remap) = lower_dict_pure(&input.dictionary)?;
    remap_and_upload(input, new_dict, &remap)
}

/// Compute the byte length of each row's string as `Int32`.
///
/// NULL rows yield `0` (NOT SQL `NULL`); see the module-level caveat. The
/// returned `Int32Array` has no validity bitmap.
///
/// Cost: one device→host copy of `n_rows` i32s + an O(dict) table build.
pub fn length(input: &DictionaryColumn) -> PatinaResult<Int32Array> {
    let table = lengths_table_pure(&input.dictionary)?;
    let indices: Vec<i32> = input.indices.to_vec()?;

    let mut lens: Vec<i32> = Vec::with_capacity(indices.len());
    for &idx in &indices {
        // A negative or out-of-range index would mean a kernel wrote
        // something the dictionary cannot decode. Mirror the strictness of
        // `DictionaryColumn::to_string_array`.
        if idx < 0 {
            return Err(PatinaError::Other(format!(
                "LENGTH: negative dictionary index {} (NULL is encoded as 0)",
                idx
            )));
        }
        let pos = idx as usize;
        let len = *table.get(pos).ok_or_else(|| {
            PatinaError::Other(format!(
                "LENGTH: index {} out of range (dictionary size {})",
                idx,
                input.dictionary.len()
            ))
        })?;
        lens.push(len);
    }

    Ok(Int32Array::from(lens))
}

/// Predicate: `col = literal` evaluated host-side on a dictionary column.
///
/// Used by the predicate-rewrite path when the equality could not be pushed
/// into the fused codegen kernel. Walks the dictionary once for the literal
/// lookup, then translates indices to bools on the host.
///
/// Returns an all-`false` array if the literal is absent from the dictionary
/// (which trivially matches no rows) and an all-`false` array of `n_rows`
/// length even when the input is empty.
pub fn input_eq_literal(
    input: &DictionaryColumn,
    literal: &str,
) -> PatinaResult<BooleanArray> {
    let n = input.n_rows;
    match input.index_of(literal) {
        None => Ok(BooleanArray::from(vec![false; n])),
        Some(target) => {
            let indices: Vec<i32> = input.indices.to_vec()?;
            let bools: Vec<bool> = indices.iter().map(|&i| i == target).collect();
            Ok(BooleanArray::from(bools))
        }
    }
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
) -> PatinaResult<DictionaryColumn> {
    let old: Vec<i32> = input.indices.to_vec()?;

    let mut new_indices: Vec<i32> = Vec::with_capacity(old.len());
    for &idx in &old {
        // Same strict bounds check as decode: negative or out-of-range is a
        // kernel-side bug we'd rather surface than mask.
        if idx < 0 {
            return Err(PatinaError::Other(format!(
                "string_ops: negative dictionary index {} (NULL is encoded as 0)",
                idx
            )));
        }
        let pos = idx as usize;
        let mapped = *remap.get(pos).ok_or_else(|| {
            PatinaError::Other(format!(
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
    fn lengths_table_byte_not_char() {
        // "é" is two bytes in UTF-8 but one character. We document byte
        // semantics, so the length is 2.
        let old = owned(&["é"]);
        let table = lengths_table_pure(&old).unwrap();
        assert_eq!(table, vec![0, 2]);
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
}
