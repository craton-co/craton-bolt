// SPDX-License-Identifier: Apache-2.0

//! Executor helpers for the GPU per-row `LIKE` matcher over **variable-width
//! (non-dictionary) `Utf8`** columns
//! ([`crate::plan::physical_plan::PhysicalPlan::StringLikeFilter`]).
//!
//! ## ⚠️ UNVALIDATED DEVICE PATH ⚠️
//!
//! The device kernel this path drives
//! ([`crate::jit::string_kernel::compile_like_match_kernel`]) has **not** been
//! executed on GPU hardware — this engine builds and tests with no CUDA device
//! in CI. Correctness of the device path is established ONLY by:
//!
//!   * the **host mirror** [`like_match_row`] in this module, which replicates
//!     the exact per-row byte logic the PTX emits and is unit-tested to equal
//!     [`crate::exec::like::PatternMatcher`] over a sample set for every
//!     supported shape, and
//!   * the **PTX-shape tests** in [`crate::jit::string_kernel`].
//!
//! The executor is therefore deliberately **host-fallback-safe**: any column
//! layout / pattern shape it cannot drive on the device at run time evaluates
//! the identical predicate on the host via [`crate::exec::like::host_like`].
//! A latent device bug can only cost performance, never correctness — until a
//! GPU hardware test pass validates the kernel.
//!
//! ## Scope (what fires the GPU path)
//!
//! The lowering only routes a `col LIKE 'pattern'` / `col NOT LIKE 'pattern'`
//! to this path when ALL hold:
//!
//!   * the pattern is a **constant** string with **no `ESCAPE`** clause and
//!     **no `_` wildcard**, and reduces (via [`decompose_like_pattern`]) to a
//!     **single literal segment** with optional leading / trailing `%`:
//!     `'lit'` (EXACT), `'lit%'` (PREFIX), `'%lit'` (SUFFIX), `'%lit%'`
//!     (CONTAINS). Any interior `%` (e.g. `'a%b'`), any `_`, or any `ESCAPE`
//!     → `None` → host fallback.
//!   * `col` is a **non-dictionary `Utf8`** column (the engine's variable-width
//!     layout). Dict-encoded `Utf8` keeps its existing GPU LIKE rewrite (see
//!     [`crate::plan::string_literal_rewrite`]) — that path is untouched.
//!
//! Everything else stays on the pre-existing, correct host `Expr::Like`
//! fallback.

use arrow_array::{Array, BooleanArray, StringArray};

use crate::error::BoltResult;
use crate::jit::string_kernel::LikeMode;

/// Decompose a SQL `LIKE` pattern into a `(mode, literal_bytes)` pair the GPU
/// matcher can drive, or `None` to signal "host fallback".
///
/// Returns `Some((mode, lit))` ONLY for the supported single-literal-segment
/// shapes with optional leading / trailing `%`:
///
///   | pattern  | mode                  | literal |
///   |----------|-----------------------|---------|
///   | `lit`    | [`LikeMode::Exact`]   | `lit`   |
///   | `lit%`   | [`LikeMode::Prefix`]  | `lit`   |
///   | `%lit`   | [`LikeMode::Suffix`]  | `lit`   |
///   | `%lit%`  | [`LikeMode::Contains`]| `lit`   |
///
/// Returns `None` (→ host fallback) for any of:
///   * an `ESCAPE` clause (`escape.is_some()`),
///   * a `_` wildcard anywhere,
///   * an interior `%` (e.g. `'a%b'`, `'%a%b%'`) — i.e. more than the leading /
///     trailing wildcard the four shapes allow,
///   * a literal segment that itself contains `%` after stripping the at-most-
///     one leading and at-most-one trailing wildcard.
///
/// The literal bytes are the UTF-8 bytes of the segment — the GPU matcher
/// compares raw bytes, which is correct because the leading/trailing `%`
/// shapes never need codepoint-aware matching (only `_` does, and `_` is
/// rejected). This mirrors [`crate::exec::like::PatternMatcher`]'s fast-path
/// classification for the SAME shapes, so the device result equals the host
/// `PatternMatcher` result by construction (verified in tests).
pub fn decompose_like_pattern(pattern: &str, escape: Option<char>) -> Option<(LikeMode, Vec<u8>)> {
    // NOTE on ILIKE: the GPU matcher compares raw bytes and has no
    // case-folding, so case-insensitive `ILIKE` is never routed here. The
    // physical-plan lowering (`try_lower_string_like_filter`) rejects an
    // `Expr::Like { case_insensitive: true, .. }` before this function is
    // reached, sending it to the host `exec::like::host_like` /
    // `PatternMatcher::compile_ci` path. Everything in this module is
    // therefore the case-sensitive `LIKE` path only.
    //
    // ESCAPE is out of scope — keep it on the host path.
    if escape.is_some() {
        return None;
    }
    // `_` forces codepoint-aware matching → host.
    if pattern.contains('_') {
        return None;
    }

    let leading = pattern.starts_with('%');
    let trailing = pattern.ends_with('%');

    // Strip at most one leading and one trailing `%`. Guard the degenerate
    // single-`%` (and `%%`) case: a lone `%` is `starts==ends`, so naive
    // double-strip on the same char would underflow / double-count.
    let body = match (leading, trailing) {
        (false, false) => pattern,
        (true, false) => &pattern[1..],
        (false, true) => &pattern[..pattern.len() - 1],
        (true, true) => {
            // Need at least two chars to strip both ends. `"%"` (len 1) and
            // `"%%"` (len 2) collapse to an empty body that matches everything.
            if pattern.len() <= 2 {
                ""
            } else {
                &pattern[1..pattern.len() - 1]
            }
        }
    };

    // After removing the allowed leading/trailing wildcard(s), the body is the
    // single literal segment. Any remaining `%` is an interior wildcard the
    // four shapes can't express → host fallback.
    if body.contains('%') {
        return None;
    }

    let mode = match (leading, trailing) {
        (false, false) => LikeMode::Exact,
        (false, true) => LikeMode::Prefix,
        (true, false) => LikeMode::Suffix,
        (true, true) => LikeMode::Contains,
    };
    Some((mode, body.as_bytes().to_vec()))
}

/// Host mirror of the per-row device matcher
/// ([`crate::jit::string_kernel::compile_like_match_kernel`]), byte-for-byte.
///
/// `row` is the row's raw UTF-8 bytes, `lit` the literal segment's bytes,
/// `mode` the comparison shape, and `negated` inverts the result (`NOT LIKE`).
/// This is the reference the device path is validated against (it must agree
/// with [`crate::exec::like::PatternMatcher`] for the supported shapes — see
/// the tests).
///
/// Empty `lit`: PREFIX / SUFFIX / CONTAINS match every row; EXACT matches iff
/// `row` is also empty — identical to the kernel's `L == 0` short-circuit.
pub fn like_match_row(row: &[u8], lit: &[u8], mode: LikeMode, negated: bool) -> bool {
    let n = row.len();
    let l = lit.len();
    let raw = if l == 0 {
        match mode {
            LikeMode::Exact => n == 0,
            LikeMode::Prefix | LikeMode::Suffix | LikeMode::Contains => true,
        }
    } else {
        match mode {
            LikeMode::Exact => n == l && &row[..] == lit,
            LikeMode::Prefix => n >= l && &row[..l] == lit,
            LikeMode::Suffix => n >= l && &row[n - l..] == lit,
            LikeMode::Contains => {
                if n < l {
                    false
                } else {
                    // Naive scan over the n-l+1 candidate start offsets,
                    // mirroring the kernel's double loop.
                    (0..=(n - l)).any(|s| &row[s..s + l] == lit)
                }
            }
        }
    };
    if negated {
        !raw
    } else {
        raw
    }
}

/// Build the Arrow-`Utf8`-shaped row-aligned `(offsets, bytes)` the GPU matcher
/// consumes, plus a per-row validity vector, from a host [`StringArray`].
///
/// NULL rows decode to an empty slice (the matcher has no validity channel);
/// the returned `validity[r]` records the true nullness so the caller can
/// re-apply SQL 3VL to the downloaded mask. `offsets` has `n_rows + 1` `i32`
/// entries.
///
/// Errors if the concatenated byte length would exceed `i32::MAX` (Arrow
/// `Utf8`, not `LargeUtf8`).
pub fn build_row_aligned_from_strings(
    col: &StringArray,
) -> BoltResult<(Vec<i32>, Vec<u8>, Vec<bool>)> {
    let n = col.len();
    let mut offsets: Vec<i32> = Vec::with_capacity(n + 1);
    let mut bytes: Vec<u8> = Vec::new();
    let mut validity: Vec<bool> = Vec::with_capacity(n);
    offsets.push(0);
    for i in 0..n {
        if col.is_null(i) {
            validity.push(false);
        } else {
            validity.push(true);
            bytes.extend_from_slice(col.value(i).as_bytes());
        }
        if bytes.len() > i32::MAX as usize {
            return Err(crate::error::BoltError::Other(format!(
                "StringLikeFilter: total bytes {} exceeds i32::MAX (Utf8)",
                bytes.len()
            )));
        }
        offsets.push(bytes.len() as i32);
    }
    Ok((offsets, bytes, validity))
}

/// Turn a downloaded `u8` (0/1) device mask into a [`BooleanArray`] with SQL
/// 3VL nullness re-applied: a NULL input row surfaces as a NULL mask entry
/// (which the filter drops), matching [`crate::exec::like::host_like`].
///
/// `mask[r]` is the device matcher's 0/1 output for row `r`; `validity[r]`
/// gates NULL rows. The `negated` flag was ALREADY applied inside the kernel,
/// so it is not re-applied here — only NULL re-masking happens.
pub fn mask_to_boolean_array(mask: &[u8], validity: &[bool]) -> BooleanArray {
    let pairs: Vec<Option<bool>> = (0..validity.len())
        .map(|r| {
            if !validity[r] {
                None
            } else {
                Some(mask.get(r).copied().unwrap_or(0) != 0)
            }
        })
        .collect();
    BooleanArray::from(pairs)
}

/// Host evaluation of the whole predicate as a [`BooleanArray`], for the
/// host-fallback path. Identical semantics to the GPU path by construction:
/// it composes [`like_match_row`] (the device mirror) with the 3VL NULL
/// re-masking, so a fallback produces the same boolean mask the validated
/// device path would.
///
/// (The engine's run-time fallback may instead call
/// [`crate::exec::like::host_like`] directly with the original pattern; this
/// helper exists so tests can assert the decomposed-shape host result equals
/// the `PatternMatcher` result.)
pub fn host_mask_via_mirror(
    col: &StringArray,
    lit: &[u8],
    mode: LikeMode,
    negated: bool,
) -> BooleanArray {
    let n = col.len();
    let pairs: Vec<Option<bool>> = (0..n)
        .map(|i| {
            if col.is_null(i) {
                None
            } else {
                Some(like_match_row(col.value(i).as_bytes(), lit, mode, negated))
            }
        })
        .collect();
    BooleanArray::from(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::like::PatternMatcher;

    fn dec(p: &str) -> Option<(LikeMode, String)> {
        decompose_like_pattern(p, None)
            .map(|(m, b)| (m, String::from_utf8(b).unwrap()))
    }

    // ---- decomposer: accepted shapes -------------------------------------

    #[test]
    fn decompose_exact() {
        assert_eq!(dec("foo"), Some((LikeMode::Exact, "foo".into())));
        // Empty exact pattern → matches only the empty string.
        assert_eq!(dec(""), Some((LikeMode::Exact, "".into())));
    }

    #[test]
    fn decompose_prefix() {
        assert_eq!(dec("foo%"), Some((LikeMode::Prefix, "foo".into())));
    }

    #[test]
    fn decompose_suffix() {
        assert_eq!(dec("%foo"), Some((LikeMode::Suffix, "foo".into())));
    }

    #[test]
    fn decompose_contains() {
        assert_eq!(dec("%foo%"), Some((LikeMode::Contains, "foo".into())));
    }

    #[test]
    fn decompose_bare_percent_variants() {
        // `%` and `%%` reduce to a Contains over the empty literal → match all.
        assert_eq!(dec("%"), Some((LikeMode::Contains, "".into())));
        assert_eq!(dec("%%"), Some((LikeMode::Contains, "".into())));
    }

    // ---- decomposer: rejected shapes (→ host fallback / None) -------------

    #[test]
    fn decompose_rejects_underscore() {
        assert_eq!(dec("f_o"), None);
        assert_eq!(dec("foo_%"), None);
        assert_eq!(dec("_"), None);
    }

    #[test]
    fn decompose_rejects_escape() {
        // Any ESCAPE clause → host fallback regardless of shape.
        assert_eq!(decompose_like_pattern("foo%", Some('\\')), None);
        assert_eq!(decompose_like_pattern("foo", Some('!')), None);
    }

    #[test]
    fn decompose_rejects_interior_percent() {
        assert_eq!(dec("a%b"), None);
        assert_eq!(dec("a%b%"), None);
        assert_eq!(dec("%a%b"), None);
        assert_eq!(dec("%a%b%"), None);
        // Three+ percents always have an interior one.
        assert_eq!(dec("%a%b%c%"), None);
    }

    // ---- host mirror equals PatternMatcher for supported shapes ----------

    #[test]
    fn mirror_equals_pattern_matcher_on_samples() {
        // For every accepted pattern + a spread of inputs, the device mirror
        // (decompose → like_match_row) must equal the host PatternMatcher.
        let patterns = ["foo", "foo%", "%foo", "%foo%", "", "%", "%%"];
        let inputs = [
            "foo", "foobar", "barfoo", "abcfoodef", "bar", "", "f", "fo",
            "FOO", "foofoo", "xfooy",
        ];
        for p in patterns {
            let (mode, lit) = decompose_like_pattern(p, None)
                .unwrap_or_else(|| panic!("pattern {p:?} should decompose"));
            let pm = PatternMatcher::compile(p, None).unwrap();
            for s in inputs {
                let mirror = like_match_row(s.as_bytes(), &lit, mode, false);
                assert_eq!(
                    mirror,
                    pm.matches(s),
                    "LIKE mismatch: pattern={p:?} input={s:?} mode={mode:?}"
                );
                // NOT LIKE is the strict inversion of LIKE for non-NULL rows.
                let mirror_neg = like_match_row(s.as_bytes(), &lit, mode, true);
                assert_eq!(mirror_neg, !pm.matches(s), "NOT LIKE: {p:?} {s:?}");
            }
        }
    }

    #[test]
    fn mirror_empty_literal_rules() {
        // EXACT "" matches only "".
        assert!(like_match_row(b"", b"", LikeMode::Exact, false));
        assert!(!like_match_row(b"x", b"", LikeMode::Exact, false));
        // PREFIX/SUFFIX/CONTAINS "" match everything.
        for mode in [LikeMode::Prefix, LikeMode::Suffix, LikeMode::Contains] {
            assert!(like_match_row(b"anything", b"", mode, false));
            assert!(like_match_row(b"", b"", mode, false));
        }
    }

    #[test]
    fn mirror_contains_overlapping() {
        // Substring scan must find the needle at any start offset.
        assert!(like_match_row(b"aXbcfoo", b"foo", LikeMode::Contains, false));
        assert!(like_match_row(b"fooooo", b"foo", LikeMode::Contains, false));
        assert!(!like_match_row(b"fo", b"foo", LikeMode::Contains, false));
    }

    // ---- row-aligned builder + mask re-masking ---------------------------

    #[test]
    fn row_aligned_with_nulls() {
        let col = StringArray::from(vec![Some("ab"), None, Some(""), Some("cde")]);
        let (offsets, bytes, validity) = build_row_aligned_from_strings(&col).unwrap();
        assert_eq!(offsets, vec![0, 2, 2, 2, 5]);
        assert_eq!(&bytes, b"abcde");
        assert_eq!(validity, vec![true, false, true, true]);
    }

    #[test]
    fn mask_reapplies_nulls() {
        // Device mask 0/1, with row 1 NULL → BooleanArray [false?, NULL, true].
        let mask = [1u8, 1, 0];
        let validity = [true, false, true];
        let arr = mask_to_boolean_array(&mask, &validity);
        assert_eq!(arr.value(0), true);
        assert!(arr.is_null(1), "NULL row stays NULL");
        assert_eq!(arr.value(2), false);
    }

    #[test]
    fn host_mask_matches_host_like_3vl() {
        // host_mask_via_mirror must agree with exec::like::host_like (the
        // canonical host path) including NULL 3VL.
        let col = StringArray::from(vec![Some("foo"), None, Some("foobar"), Some("bar")]);
        for (p, mode, lit, neg) in [
            ("foo%", LikeMode::Prefix, "foo", false),
            ("foo%", LikeMode::Prefix, "foo", true),
            ("%bar", LikeMode::Suffix, "bar", false),
            ("foo", LikeMode::Exact, "foo", false),
        ] {
            let mine = host_mask_via_mirror(&col, lit.as_bytes(), mode, neg);
            let canon = crate::exec::like::host_like(&col, p, None, neg).unwrap();
            assert_eq!(mine.len(), canon.len());
            for i in 0..mine.len() {
                assert_eq!(mine.is_null(i), canon.is_null(i), "null@{i} p={p}");
                if !mine.is_null(i) {
                    assert_eq!(mine.value(i), canon.value(i), "val@{i} p={p}");
                }
            }
        }
    }
}
