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
//! * Character-indexed `SUBSTRING` (ANSI SQL / DuckDB semantics): positions are
//!   1-based Unicode codepoints, never bytes. A multibyte codepoint is never
//!   split, and characters at positions before the requested `start` are never
//!   leaked into the result. Matches the character-count convention of
//!   `string_ops::length` (`LENGTH` counts characters; byte length is
//!   `OCTET_LENGTH`).
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
// SUBSTRING character-slice helper.
// ---------------------------------------------------------------------------

/// Compute `SUBSTRING(s, start_1based, length)` over CHARACTERS (Unicode
/// codepoints), per ANSI SQL / DuckDB semantics.
///
/// `SUBSTRING` is defined over characters, not bytes. The window is the set of
/// 1-based character positions `[start, start + length)`. We never split a
/// multibyte codepoint and never leak bytes that precede the requested `start`.
///
/// Semantics (matching DuckDB / Postgres):
///
/// * Positions are 1-based characters; `start = 1` is the first character.
/// * `start < 1` is honoured as a real (out-of-range) position: the window
///   still ends at character `start + length - 1`, so characters that fall at
///   positions `< 1` simply don't exist and are not emitted. e.g.
///   `SUBSTRING('abc', 0, 2)` covers char positions `[0, 2)` → only position
///   `1` exists → `"a"`; `SUBSTRING('abc', -1, 3)` covers `[-1, 2)` → only
///   position `1` exists → `"a"`.
/// * `length < 0` is treated as `0` (empty result).
/// * The window saturates against the string's character count.
///
/// The result is always a valid UTF-8 substring of `s` and never contains any
/// character whose 1-based position is below `start`.
fn sql_substring(s: &str, start_1based: i32, length: i32) -> String {
    // ANSI: negative length is empty.
    if length <= 0 {
        return String::new();
    }
    // The requested character window is the 1-based position range
    // [start, start + length). Compute it in i64 so `length = i32::MAX`
    // ("to end") plus a large start cannot overflow.
    let win_start = start_1based as i64; // first 1-based position included
    let win_end = win_start.saturating_add(length as i64); // one past the last

    // Clamp the window's lower edge to the first real character position (1):
    // characters at positions < 1 do not exist and must never be emitted.
    let first_pos = win_start.max(1);
    if win_end <= first_pos {
        // Window collapses to empty once positions < 1 are excluded.
        return String::new();
    }

    // Convert 1-based character positions into 0-based char skip / take counts.
    // `first_pos >= 1`, so `skip` is non-negative.
    let skip = (first_pos - 1) as usize;
    // Number of characters to take = win_end - first_pos (both >= 1-based,
    // win_end > first_pos guaranteed above).
    let take = (win_end - first_pos) as usize;

    s.chars().skip(skip).take(take).collect()
}

/// Per-string `SUBSTRING(s, start, length)` over CHARACTERS — public wrapper
/// around the internal `sql_substring` helper so the host expression evaluator
/// (`expr_agg::eval_expr`) can apply SUBSTRING directly to a
/// `Vec<Option<String>>` column without round-tripping through a
/// `DictionaryColumn`. See module docs for the character/UTF-8 semantics.
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
/// GPU status: a two-pass GPU producer for the *ASCII-whitespace-default* case
/// (`chars = None`, BOTH/LEADING/TRAILING) now exists in
/// [`crate::jit::string_kernel`] — see `compile_varwidth_len_pass` /
/// `compile_varwidth_write_pass` for `ScalarFnKind::Trim*` and
/// `string_kernel::emit_trim_bounds`. It is restricted to ASCII whitespace
/// (HT/LF/VT/FF/CR/SPACE), which is UTF-8-safe because none of those bytes can
/// appear inside a multi-byte codepoint. This host path remains the supported
/// fallback and the ONLY path for: (a) custom trim-character sets
/// (`chars = Some(..)`), and (b) Unicode (non-ASCII) whitespace, which the GPU
/// scan deliberately does not strip. Whatever routes TRIM to the GPU must keep
/// this host fallback for those cases and on any kernel/launch error, so
/// results never change.
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

// ---------------------------------------------------------------------------
// Per-string scalar helpers (host evaluator path).
//
// These mirror `substring_str` / `trim_str`: pure `&str -> String` (or
// `-> i64`) transforms the host expression evaluator (`expr_agg::eval_scalar_fn`)
// applies directly to a `Vec<Option<String>>` column. They are character-based
// (Unicode codepoints), NULL-propagation is handled by the caller (a NULL input
// cell never reaches these — see `expr_agg`). No GPU producer is wired for any
// of them; the host path is the supported one.
// ---------------------------------------------------------------------------

/// `CHAR_LENGTH(s)` / `CHARACTER_LENGTH(s)` — character (Unicode codepoint)
/// count, returned as `i64` to match the SQL `LENGTH -> Int64` contract.
/// Synonym for `LENGTH`; `CHAR_LENGTH('héllo') = 5`.
pub fn char_length_str(s: &str) -> i64 {
    s.chars().count() as i64
}

/// `OCTET_LENGTH(s)` — UTF-8 byte length, returned as `i64`.
/// `OCTET_LENGTH('héllo') = 6` (the 'é' is two bytes). Byte counterpart of
/// [`char_length_str`].
pub fn octet_length_str(s: &str) -> i64 {
    s.len() as i64
}

/// `POSITION(substr IN s)` / `STRPOS(s, substr)` — 1-based CHARACTER index of
/// the first occurrence of `substr` in `s`, or `0` if `substr` is not present.
///
/// Per ANSI SQL the empty substring is found at position `1`. The index is a
/// character (codepoint) position, not a byte offset: searching `"héllo"` for
/// `"llo"` returns `3`, not `4`.
pub fn position_str(s: &str, substr: &str) -> i64 {
    if substr.is_empty() {
        return 1;
    }
    // `find` gives a byte offset; convert it to a 1-based character index by
    // counting codepoints in the prefix before the match.
    match s.find(substr) {
        Some(byte_off) => s[..byte_off].chars().count() as i64 + 1,
        None => 0,
    }
}

/// `REPLACE(s, from, to)` — replace every non-overlapping occurrence of `from`
/// in `s` with `to`. When `from` is empty, `s` is returned unchanged (matching
/// PostgreSQL / DuckDB, which never inject `to` into an empty-needle search).
pub fn replace_str(s: &str, from: &str, to: &str) -> String {
    if from.is_empty() {
        return s.to_string();
    }
    s.replace(from, to)
}

/// `LEFT(s, n)` — the first `n` CHARACTERS of `s` (ANSI / PostgreSQL).
///
/// * `n >= len`  → the whole string.
/// * `n == 0`    → `""`.
/// * `n < 0`     → all but the last `|n|` characters (PostgreSQL semantics):
///   `LEFT('abcde', -2) = "abc"`. If `|n| >= len` the result is `""`.
pub fn left_str(s: &str, n: i64) -> String {
    let char_count = s.chars().count() as i64;
    let take = if n >= 0 {
        n.min(char_count)
    } else {
        // Drop the last |n| chars: keep char_count - |n|, floored at 0.
        (char_count + n).max(0)
    };
    s.chars().take(take as usize).collect()
}

/// `RIGHT(s, n)` — the last `n` CHARACTERS of `s` (ANSI / PostgreSQL).
///
/// * `n >= len`  → the whole string.
/// * `n == 0`    → `""`.
/// * `n < 0`     → all but the first `|n|` characters (PostgreSQL semantics):
///   `RIGHT('abcde', -2) = "cde"`. If `|n| >= len` the result is `""`.
pub fn right_str(s: &str, n: i64) -> String {
    let char_count = s.chars().count() as i64;
    let skip = if n >= 0 {
        (char_count - n).max(0)
    } else {
        // Drop the first |n| chars: skip |n|, capped at char_count.
        (-n).min(char_count)
    };
    s.chars().skip(skip as usize).collect()
}

/// Which side [`pad_str`] pads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadSide {
    /// `LPAD` — pad/truncate on the LEFT.
    Left,
    /// `RPAD` — pad/truncate on the RIGHT.
    Right,
}

/// `LPAD(s, len, pad)` / `RPAD(s, len, pad)` — pad or truncate `s` to exactly
/// `len` CHARACTERS using `pad` as the fill (ANSI / PostgreSQL).
///
/// * If `s` is longer than `len`, it is TRUNCATED to the first `len` characters
///   (both LPAD and RPAD truncate from the right, keeping the prefix — matching
///   PostgreSQL).
/// * If `s` is shorter, `pad` is repeated (and cut mid-`pad` if needed) to fill
///   the gap on the requested side.
/// * `len <= 0` → `""`.
/// * An empty `pad` cannot fill, so `s` is only ever truncated, never padded
///   (PostgreSQL returns the truncated/verbatim string in that case).
///
/// `len` is taken as `i64` and saturated into the character domain; absurd
/// values just clamp to the string length.
pub fn pad_str(s: &str, len: i64, pad: &str, side: PadSide) -> String {
    if len <= 0 {
        return String::new();
    }
    let target = len as usize;
    let src: Vec<char> = s.chars().collect();
    if src.len() >= target {
        // Truncate to the first `target` characters (prefix kept on both sides).
        return src.into_iter().take(target).collect();
    }
    let gap = target - src.len();
    let pad_chars: Vec<char> = pad.chars().collect();
    if pad_chars.is_empty() {
        // Nothing to pad with: return the (shorter) source verbatim.
        return src.into_iter().collect();
    }
    // Build the fill by cycling `pad` until `gap` characters are produced.
    let fill: String = pad_chars.iter().cycle().take(gap).collect();
    match side {
        PadSide::Left => {
            let mut out = fill;
            out.extend(src.iter());
            out
        }
        PadSide::Right => {
            let mut out: String = src.into_iter().collect();
            out.push_str(&fill);
            out
        }
    }
}

/// `REVERSE(s)` — reverse the CHARACTERS of `s` (codepoint order, so multibyte
/// characters are preserved intact). `REVERSE('héllo') = "olléh"`.
pub fn reverse_str(s: &str) -> String {
    s.chars().rev().collect()
}

/// `INITCAP(s)` — upper-case the first character of each word and lower-case the
/// rest (PostgreSQL semantics). A "word" is a maximal run of alphanumeric
/// characters; any non-alphanumeric character is a separator and resets the
/// "start of word" state. `INITCAP('hi tHERE-bob') = "Hi There-Bob"`.
///
/// Case folding uses Unicode default case mapping (locale-invariant), matching
/// `UPPER` / `LOWER`.
pub fn initcap_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut at_word_start = true;
    for c in s.chars() {
        if c.is_alphanumeric() {
            if at_word_start {
                out.extend(c.to_uppercase());
            } else {
                out.extend(c.to_lowercase());
            }
            at_word_start = false;
        } else {
            out.push(c);
            at_word_start = true;
        }
    }
    out
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

/// `SUBSTRING(col, start, length)` over character (codepoint) indices.
///
/// `start` is 1-indexed over characters (SQL semantics). `length` is required
/// (no two-arg variant); pass `i32::MAX` for "to the end". Negative `length`
/// is treated as `0`. Characters at positions before `start` are never leaked.
///
/// Returns a new dictionary column with `SUBSTRING` applied to each unique
/// dictionary entry, then re-deduplicated — substrings of different originals
/// may coincide ("abc" and "abd" both become "ab"). NULL (index 0) is
/// preserved as NULL. See module docs for the character/UTF-8 semantics.
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
        // "héllo" is 5 CHARACTERS: 'h' 'é' 'l' 'l' 'o'. SUBSTRING is character-
        // indexed (ANSI / DuckDB), so positions count codepoints, not bytes.
        let dict = owned(&["héllo"]);

        // SUBSTRING(1, 2) → characters 1..=2 → "hé".
        let (new_dict, _remap) = substring_pure(&dict, 1, 2).unwrap();
        assert_eq!(new_dict, vec!["hé"]);

        // SUBSTRING(1, 3) → characters 1..=3 → "hél".
        let (new_dict2, _) = substring_pure(&dict, 1, 3).unwrap();
        assert_eq!(new_dict2, vec!["hél"]);
    }

    #[test]
    fn substring_multibyte_three_byte_chars() {
        // "日本語" is three 3-byte characters. Character-indexed SUBSTRING
        // must treat each codepoint as a single position (byte semantics would
        // have produced "" here).
        let dict = owned(&["日本語"]);
        // SUBSTRING(1, 2) → "日本".
        let (d1, _) = substring_pure(&dict, 1, 2).unwrap();
        assert_eq!(d1, vec!["日本"]);
        // SUBSTRING(2, 2) → "本語".
        let (d2, _) = substring_pure(&dict, 2, 2).unwrap();
        assert_eq!(d2, vec!["本語"]);
        // SUBSTRING(3, 5) saturates to the last character → "語".
        let (d3, _) = substring_pure(&dict, 3, 5).unwrap();
        assert_eq!(d3, vec!["語"]);
    }

    #[test]
    fn substring_emoji_four_byte_chars() {
        // 4-byte codepoints must also count as one position each.
        let dict = owned(&["a😀b😀c"]);
        // characters: 'a' '😀' 'b' '😀' 'c'. SUBSTRING(2, 3) → "😀b😀".
        let (d, _) = substring_pure(&dict, 2, 3).unwrap();
        assert_eq!(d, vec!["😀b😀"]);
    }

    #[test]
    fn substring_start_below_one_window_semantics() {
        // DuckDB / ANSI: the window is [start, start+len) over 1-based char
        // positions; positions < 1 simply don't exist (they are not emitted,
        // and they DO consume window length).
        let dict = owned(&["abc"]);
        // SUBSTRING("abc", 0, 2): positions [0,2) → only 1 → "a".
        let (new_dict_0, _) = substring_pure(&dict, 0, 2).unwrap();
        assert_eq!(new_dict_0, vec!["a"]);
        // SUBSTRING("abc", -5, 2): positions [-5,-3) → none → "".
        let (new_dict_neg, _) = substring_pure(&dict, -5, 2).unwrap();
        assert_eq!(new_dict_neg, vec![""]);
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
    fn sql_substring_unicode_character_indexed() {
        // SUBSTRING("héllo", 2, 4): characters 2..=5 → 'é' 'l' 'l' 'o' → "éllo".
        assert_eq!(sql_substring("héllo", 2, 4), "éllo");

        // SUBSTRING("héllo", 3, 2): characters 3..=4 → 'l' 'l' → "ll".
        // The character at position 2 ('é') must NEVER appear — a character at
        // a position left of `start` may not leak into the result.
        assert_eq!(sql_substring("héllo", 3, 2), "ll");
    }

    #[test]
    fn sql_substring_no_left_of_start_leak() {
        // Regression for F-1: the byte-rounding model could pull a character
        // that begins before the requested start into the result. With
        // character indexing, the result must never contain any character at a
        // 1-based position below `start`.
        //
        // "héllo": positions 1='h', 2='é', 3='l', 4='l', 5='o'.
        // For every (start, len), the result chars must equal the slice of the
        // full char vector at positions [start, start+len) intersected with the
        // valid 1..=5 range — in particular nothing left of `start`.
        let chars: Vec<char> = "héllo".chars().collect();
        for start in 1..=6i32 {
            for len in 0..=6i32 {
                let got = sql_substring("héllo", start, len);
                let got_chars: Vec<char> = got.chars().collect();
                // Build the expected window directly from char positions.
                let mut expected: Vec<char> = Vec::new();
                if len > 0 {
                    let lo = start.max(1);
                    let hi = (start as i64 + len as i64).min(chars.len() as i64 + 1);
                    for pos in lo as i64..hi {
                        if pos >= 1 && (pos as usize) <= chars.len() {
                            expected.push(chars[(pos - 1) as usize]);
                        }
                    }
                }
                assert_eq!(
                    got_chars, expected,
                    "SUBSTRING(\"héllo\", {start}, {len}) leaked or dropped chars"
                );
                // Explicitly: 'é' (position 2) must not appear when start > 2.
                if start > 2 {
                    assert!(
                        !got.contains('é'),
                        "left-of-start leak: 'é' in SUBSTRING(\"héllo\", {start}, {len})"
                    );
                }
            }
        }
    }

    #[test]
    fn sql_substring_handles_i32_max_length() {
        // i32::MAX is huge but never overflows because the window math is done
        // in i64 and char-take saturates at the string's character count.
        assert_eq!(sql_substring("abc", 1, i32::MAX), "abc");
        // "to end" idiom over multibyte input.
        assert_eq!(sql_substring("héllo", 2, i32::MAX), "éllo");
    }

    #[test]
    fn sql_substring_start_below_one_excludes_phantom_positions() {
        // DuckDB: SUBSTRING('abc', 0, 2) covers positions [0,2) → only 1 exists.
        assert_eq!(sql_substring("abc", 0, 2), "a");
        // SUBSTRING('abc', -1, 3) covers positions [-1,2) → only 1 exists.
        assert_eq!(sql_substring("abc", -1, 3), "a");
        // SUBSTRING('abc', -1, 5) covers positions [-1,4) → 1,2,3 → "abc".
        assert_eq!(sql_substring("abc", -1, 5), "abc");
        // A window entirely left of position 1 yields "".
        assert_eq!(sql_substring("abc", -5, 3), "");
    }

    #[test]
    fn substring_str_public_wrapper_matches_internal() {
        assert_eq!(substring_str("hello", 2, 3), "ell");
        // Character-indexed: characters 1..=3 of "héllo" are 'h' 'é' 'l'.
        assert_eq!(substring_str("héllo", 1, 3), "hél");
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

    // ----- CHAR_LENGTH / OCTET_LENGTH ------------------------------------

    #[test]
    fn char_length_counts_characters() {
        assert_eq!(char_length_str("héllo"), 5);
        assert_eq!(char_length_str("日本語"), 3);
        assert_eq!(char_length_str(""), 0);
        assert_eq!(char_length_str("abc"), 3);
    }

    #[test]
    fn octet_length_counts_bytes() {
        assert_eq!(octet_length_str("héllo"), 6); // 'é' is 2 bytes
        assert_eq!(octet_length_str("日本語"), 9); // each char is 3 bytes
        assert_eq!(octet_length_str(""), 0);
        assert_eq!(octet_length_str("abc"), 3);
    }

    // ----- POSITION / STRPOS ---------------------------------------------

    #[test]
    fn position_basic() {
        assert_eq!(position_str("hello", "ll"), 3);
        assert_eq!(position_str("hello", "h"), 1);
        assert_eq!(position_str("hello", "o"), 5);
        assert_eq!(position_str("hello", "z"), 0);
    }

    #[test]
    fn position_empty_substring_is_one() {
        // ANSI: empty needle is found at position 1.
        assert_eq!(position_str("hello", ""), 1);
        assert_eq!(position_str("", ""), 1);
    }

    #[test]
    fn position_is_character_indexed_not_byte() {
        // "héllo": 'h'(1) 'é'(2) 'l'(3) 'l'(4) 'o'(5). "llo" begins at CHAR 3
        // even though it begins at BYTE 4 ('é' is two bytes).
        assert_eq!(position_str("héllo", "llo"), 3);
        assert_eq!(position_str("héllo", "é"), 2);
        assert_eq!(position_str("日本語", "語"), 3);
    }

    #[test]
    fn position_first_occurrence() {
        assert_eq!(position_str("abcabc", "bc"), 2);
    }

    // ----- REPLACE -------------------------------------------------------

    #[test]
    fn replace_basic() {
        assert_eq!(replace_str("hello world", "o", "0"), "hell0 w0rld");
        assert_eq!(replace_str("aaa", "a", "bb"), "bbbbbb");
        assert_eq!(replace_str("abc", "x", "y"), "abc");
    }

    #[test]
    fn replace_substring_not_char_set() {
        // REPLACE replaces the whole "from" substring, not a char set.
        assert_eq!(replace_str("a.b.c", ".", "-"), "a-b-c");
        assert_eq!(replace_str("foofoo", "foo", ""), "");
    }

    #[test]
    fn replace_empty_from_returns_unchanged() {
        // Empty needle: leave the string untouched (Postgres/DuckDB).
        assert_eq!(replace_str("abc", "", "X"), "abc");
    }

    #[test]
    fn replace_unicode() {
        assert_eq!(replace_str("héllo", "é", "e"), "hello");
        assert_eq!(replace_str("日本語", "本", "X"), "日X語");
    }

    // ----- LEFT / RIGHT --------------------------------------------------

    #[test]
    fn left_basic() {
        assert_eq!(left_str("abcde", 3), "abc");
        assert_eq!(left_str("abcde", 0), "");
        assert_eq!(left_str("abcde", 10), "abcde");
    }

    #[test]
    fn left_negative_drops_from_end() {
        // PostgreSQL: LEFT('abcde', -2) keeps all but the last 2 -> "abc".
        assert_eq!(left_str("abcde", -2), "abc");
        assert_eq!(left_str("abcde", -5), "");
        assert_eq!(left_str("abcde", -10), "");
    }

    #[test]
    fn left_character_indexed() {
        assert_eq!(left_str("héllo", 2), "hé");
        assert_eq!(left_str("日本語", 2), "日本");
        assert_eq!(left_str("héllo", -2), "hél");
    }

    #[test]
    fn right_basic() {
        assert_eq!(right_str("abcde", 3), "cde");
        assert_eq!(right_str("abcde", 0), "");
        assert_eq!(right_str("abcde", 10), "abcde");
    }

    #[test]
    fn right_negative_drops_from_front() {
        // PostgreSQL: RIGHT('abcde', -2) keeps all but the first 2 -> "cde".
        assert_eq!(right_str("abcde", -2), "cde");
        assert_eq!(right_str("abcde", -5), "");
        assert_eq!(right_str("abcde", -10), "");
    }

    #[test]
    fn right_character_indexed() {
        assert_eq!(right_str("héllo", 3), "llo");
        assert_eq!(right_str("日本語", 2), "本語");
        assert_eq!(right_str("héllo", -1), "éllo");
    }

    // ----- LPAD / RPAD ---------------------------------------------------

    #[test]
    fn lpad_pads_on_left() {
        assert_eq!(pad_str("5", 3, "0", PadSide::Left), "005");
        assert_eq!(pad_str("abc", 5, "xy", PadSide::Left), "xyabc");
        assert_eq!(pad_str("abc", 6, "xy", PadSide::Left), "xyxabc");
    }

    #[test]
    fn rpad_pads_on_right() {
        assert_eq!(pad_str("5", 3, "0", PadSide::Right), "500");
        assert_eq!(pad_str("abc", 5, "xy", PadSide::Right), "abcxy");
        assert_eq!(pad_str("abc", 6, "xy", PadSide::Right), "abcxyx");
    }

    #[test]
    fn pad_truncates_when_source_too_long() {
        // Both LPAD and RPAD truncate to the first `len` chars when source is
        // longer than the target (Postgres).
        assert_eq!(pad_str("abcdef", 3, "x", PadSide::Left), "abc");
        assert_eq!(pad_str("abcdef", 3, "x", PadSide::Right), "abc");
    }

    #[test]
    fn pad_zero_or_negative_len_is_empty() {
        assert_eq!(pad_str("abc", 0, "x", PadSide::Left), "");
        assert_eq!(pad_str("abc", -3, "x", PadSide::Right), "");
    }

    #[test]
    fn pad_empty_pad_string_only_truncates() {
        // No fill available: shorter source returned verbatim, longer truncated.
        assert_eq!(pad_str("ab", 5, "", PadSide::Left), "ab");
        assert_eq!(pad_str("abcdef", 3, "", PadSide::Right), "abc");
    }

    #[test]
    fn pad_character_indexed() {
        // Multibyte pad and source count by characters, not bytes.
        assert_eq!(pad_str("x", 3, "→", PadSide::Left), "→→x");
        assert_eq!(pad_str("héllo", 7, "*", PadSide::Right), "héllo**");
    }

    // ----- REVERSE -------------------------------------------------------

    #[test]
    fn reverse_basic() {
        assert_eq!(reverse_str("abc"), "cba");
        assert_eq!(reverse_str(""), "");
        assert_eq!(reverse_str("a"), "a");
    }

    #[test]
    fn reverse_preserves_multibyte_chars() {
        assert_eq!(reverse_str("héllo"), "olléh");
        assert_eq!(reverse_str("日本語"), "語本日");
        assert_eq!(reverse_str("a😀b"), "b😀a");
    }

    // ----- INITCAP -------------------------------------------------------

    #[test]
    fn initcap_basic() {
        assert_eq!(initcap_str("hello world"), "Hello World");
        assert_eq!(initcap_str("HELLO WORLD"), "Hello World");
        assert_eq!(initcap_str(""), "");
    }

    #[test]
    fn initcap_word_boundaries_on_non_alphanumeric() {
        // Hyphen and other punctuation reset the word boundary.
        assert_eq!(initcap_str("hi tHERE-bob"), "Hi There-Bob");
        assert_eq!(initcap_str("a.b.c"), "A.B.C");
        // Digits are alphanumeric: a digit starts a word, the following letter
        // is mid-word and lower-cased.
        assert_eq!(initcap_str("1ab 2CD"), "1ab 2cd");
    }

    #[test]
    fn initcap_unicode() {
        // Accented letters are alphanumeric and case-fold via Unicode default.
        assert_eq!(initcap_str("éric dupont"), "Éric Dupont");
    }

    // ----- GPU TRIM byte-rule mirror -------------------------------------
    //
    // The GPU two-pass TRIM kernel (jit::string_kernel) strips only the SIX
    // ASCII whitespace bytes HT/LF/VT/FF/CR (0x09..=0x0D) and SPACE (0x20).
    // This host mirror replicates that exact byte rule so we can assert the
    // restricted GPU semantics agree with `trim_str` on ASCII-whitespace
    // input (the only input the executor is allowed to route to the GPU).

    /// Byte-for-byte mirror of the GPU kernel's ASCII-whitespace trim.
    fn gpu_ascii_trim(s: &str, side: TrimSide) -> String {
        let is_ws = |b: u8| (0x09..=0x0D).contains(&b) || b == 0x20;
        let bytes = s.as_bytes();
        let mut begin = 0usize;
        let mut end = bytes.len();
        if matches!(side, TrimSide::Both | TrimSide::Leading) {
            while begin < end && is_ws(bytes[begin]) {
                begin += 1;
            }
        }
        if matches!(side, TrimSide::Both | TrimSide::Trailing) {
            while end > begin && is_ws(bytes[end - 1]) {
                end -= 1;
            }
        }
        // The kept window is always valid UTF-8: only single-byte ASCII
        // whitespace was removed, never a continuation/lead byte.
        std::str::from_utf8(&bytes[begin..end]).unwrap().to_string()
    }

    #[test]
    fn gpu_ascii_trim_matches_host_on_ascii_whitespace() {
        // For ASCII-whitespace-delimited input the GPU's restricted byte rule
        // must produce the SAME result as the host `trim_str` default path.
        let cases = [
            "  hi  ",
            "\t\nhi \r",
            "nospace",
            "   ",
            "",
            " a b c ",
            "\x0b\x0cmid\x0c\x0b",
        ];
        for s in cases {
            for side in [TrimSide::Both, TrimSide::Leading, TrimSide::Trailing] {
                assert_eq!(
                    gpu_ascii_trim(s, side),
                    trim_str(s, side, None),
                    "GPU/host TRIM divergence for {s:?} side {side:?}"
                );
            }
        }
    }

    #[test]
    fn gpu_ascii_trim_preserves_multibyte_content() {
        // Multi-byte content with ASCII whitespace padding: the GPU rule trims
        // only the ASCII spaces and leaves the codepoints intact.
        assert_eq!(gpu_ascii_trim("  héllo  ", TrimSide::Both), "héllo");
        // A non-ASCII whitespace (NBSP, U+00A0 = 0xC2 0xA0) is NOT stripped by
        // the GPU rule — confirming why such input stays on the host path.
        let nbsp = "\u{00A0}x\u{00A0}";
        assert_eq!(gpu_ascii_trim(nbsp, TrimSide::Both), nbsp);
    }
}
