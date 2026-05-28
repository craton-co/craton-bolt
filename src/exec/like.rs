// SPDX-License-Identifier: Apache-2.0

//! Host-side SQL `LIKE` evaluator.
//!
//! v0.5 (M2 SQL scalar completeness) ships LIKE with a **constant pattern**
//! only: the SQL frontend captures `expr LIKE 'pattern'` as
//! [`crate::plan::logical_plan::Expr::Like`] with `pattern: String`. The
//! physical-plan lowerer routes every LIKE predicate through the host-side
//! [`crate::plan::physical_plan::PhysicalPlan::Filter`] path because the
//! GPU codegen has no Utf8 column access yet. This module owns the
//! actual matcher.
//!
//! ## Wildcards
//!
//!   * `%` — matches zero-or-more arbitrary characters.
//!   * `_` — matches exactly one arbitrary character.
//!
//! ## Pattern shapes recognised
//!
//! Four common shapes get a fast-path implementation; the rest fall back
//! to a generic byte-by-byte matcher with `%` backtracking. The fast-path
//! split is invisible to callers — [`PatternMatcher::compile`] picks the
//! variant and [`PatternMatcher::matches`] dispatches on it.
//!
//!   | Pattern  | Semantics            | Variant            |
//!   |----------|----------------------|--------------------|
//!   | `foo`    | exact match          | [`Shape::Exact`]   |
//!   | `foo%`   | starts-with prefix   | [`Shape::Prefix`]  |
//!   | `%foo`   | ends-with suffix     | [`Shape::Suffix`]  |
//!   | `%foo%`  | substring contains   | [`Shape::Contains`]|
//!   | other    | generic char-class   | [`Shape::Generic`] |
//!
//! The exact-match shape is canonicalised — the SQL frontend leaves the
//! pattern in the AST exactly as the user wrote it; if the engine wants
//! to rewrite `LIKE 'foo'` (no wildcards) into `= 'foo'` to pick up the
//! dictionary-eq fast path, it can do so at the planner before this
//! module is reached. We still implement [`Shape::Exact`] here so the
//! direct host path stays correct in isolation.
//!
//! ## ESCAPE
//!
//! `escape` is reserved on [`crate::plan::logical_plan::Expr::Like`] but
//! the SQL frontend rejects `ESCAPE '<char>'` for v0.5 — implementing the
//! escape rule (a `\%` becomes a literal `%`, etc.) is a follow-up. The
//! `compile` API still takes an `Option<char>` so the eventual change
//! doesn't have to reshape callers.

use arrow_array::{Array, BooleanArray, StringArray};

use crate::error::{BoltError, BoltResult};

/// Compiled LIKE pattern, ready for fast-path evaluation per row.
///
/// Constructed via [`PatternMatcher::compile`] (validates the pattern and
/// picks a [`Shape`]); evaluated per cell via [`PatternMatcher::matches`].
#[derive(Debug, Clone)]
pub struct PatternMatcher {
    /// Detected shape — drives the dispatch in `matches`.
    shape: Shape,
}

/// Recognised pattern shapes, picked at compile time so per-row matching
/// avoids re-scanning the pattern.
#[derive(Debug, Clone)]
enum Shape {
    /// `'foo'` — exact string equality (no wildcards in the pattern).
    Exact(String),
    /// `'foo%'` — `s.starts_with("foo")`. The wildcard tail has no other
    /// characters past the final `%`.
    Prefix(String),
    /// `'%foo'` — `s.ends_with("foo")`. The wildcard head sits at index 0
    /// and the rest is literal.
    Suffix(String),
    /// `'%foo%'` — `s.contains("foo")`. Wildcard at both ends, no internal
    /// wildcards.
    Contains(String),
    /// Generic char-class fallback. Stored as the original byte-segments:
    /// a `Vec<Token>` produced by [`tokenise`], replayed against the input
    /// per row via [`generic_match`].
    Generic(Vec<Token>),
}

/// Compiled token inside a [`Shape::Generic`] pattern.
#[derive(Debug, Clone)]
enum Token {
    /// Literal byte run (no `%` or `_` inside).
    Literal(String),
    /// `_` — match exactly one character.
    One,
    /// `%` — match zero-or-more characters.
    Any,
}

impl PatternMatcher {
    /// Compile a SQL LIKE pattern.
    ///
    /// `escape` is currently always `None` (the SQL frontend rejects the
    /// `ESCAPE '<char>'` clause for v0.5). Passing `Some(_)` here returns
    /// a clear "not yet supported" error so a hand-built logical plan that
    /// tried to use escape fails loudly rather than silently producing
    /// wrong results.
    pub fn compile(pattern: &str, escape: Option<char>) -> BoltResult<Self> {
        if escape.is_some() {
            return Err(BoltError::Plan(
                "LIKE ... ESCAPE '<char>' is not yet implemented (v0.5 follow-up)"
                    .into(),
            ));
        }
        let shape = classify(pattern);
        Ok(Self { shape })
    }

    /// True if `s` matches this compiled pattern.
    pub fn matches(&self, s: &str) -> bool {
        match &self.shape {
            Shape::Exact(p) => s == p,
            Shape::Prefix(p) => s.starts_with(p.as_str()),
            Shape::Suffix(p) => s.ends_with(p.as_str()),
            Shape::Contains(p) => {
                // `str::contains` over a `&str` performs the same substring
                // search the SQL semantics demand. Empty needle means the
                // pattern was `%%` (or just `%`), which matches everything;
                // `"".contains("")` is `true` in Rust, so this is correct.
                s.contains(p.as_str())
            }
            Shape::Generic(tokens) => generic_match(s, tokens),
        }
    }
}

/// Classify the pattern into a fast-path [`Shape`] when possible, falling
/// back to [`Shape::Generic`] for anything else.
///
/// The fast-path recognisers all require the pattern to contain *no* `_`
/// wildcards (those force per-character matching, so they go to Generic),
/// and the `%` placements must match one of the four canonical shapes.
fn classify(pattern: &str) -> Shape {
    // Bail out to Generic immediately if there's any `_` — fast paths only
    // handle `%`-style shapes.
    if pattern.contains('_') {
        return Shape::Generic(tokenise(pattern));
    }
    let n_pct = pattern.chars().filter(|c| *c == '%').count();
    if n_pct == 0 {
        // No wildcards at all → exact match.
        return Shape::Exact(pattern.to_string());
    }
    if n_pct == 1 {
        if let Some(rest) = pattern.strip_suffix('%') {
            // `foo%` — prefix. (Pattern `%` alone strips to empty rest →
            // prefix "" matches everything, which is the correct semantic.)
            return Shape::Prefix(rest.to_string());
        }
        if let Some(rest) = pattern.strip_prefix('%') {
            // `%foo` — suffix.
            return Shape::Suffix(rest.to_string());
        }
        // `%` was internal (`fo%o`) — not one of the fast-path shapes.
        return Shape::Generic(tokenise(pattern));
    }
    if n_pct == 2 {
        // Look for the canonical `%foo%` shape: starts with `%`, ends with
        // `%`, no other `%` in between.
        if let Some(mid) = pattern
            .strip_prefix('%')
            .and_then(|s| s.strip_suffix('%'))
        {
            // The middle slice contains no `%` (we already accounted for
            // exactly two in the original) and no `_` (we bailed out above
            // for that), so `Contains` is correct.
            return Shape::Contains(mid.to_string());
        }
    }
    // Anything else (multiple internal `%`s, mix of `_` already handled
    // above) → generic matcher.
    Shape::Generic(tokenise(pattern))
}

/// Split `pattern` into a `Vec<Token>` for the generic matcher.
///
/// Runs of literal characters become one [`Token::Literal`]; each `_` and
/// `%` becomes its own token. The matcher then walks tokens linearly,
/// backtracking on `%` when a subsequent literal/`_` fails to match.
fn tokenise(pattern: &str) -> Vec<Token> {
    let mut out: Vec<Token> = Vec::new();
    let mut buf = String::new();
    for ch in pattern.chars() {
        match ch {
            '%' => {
                if !buf.is_empty() {
                    out.push(Token::Literal(std::mem::take(&mut buf)));
                }
                // Collapse runs of `%%` into a single `Any` — they're
                // semantically equivalent and the matcher does less work
                // with one token.
                if !matches!(out.last(), Some(Token::Any)) {
                    out.push(Token::Any);
                }
            }
            '_' => {
                if !buf.is_empty() {
                    out.push(Token::Literal(std::mem::take(&mut buf)));
                }
                out.push(Token::One);
            }
            c => buf.push(c),
        }
    }
    if !buf.is_empty() {
        out.push(Token::Literal(buf));
    }
    out
}

/// Generic LIKE matcher: char-level backtracking over `s` against
/// `tokens`. Returns `true` iff the whole string matches the whole
/// pattern.
///
/// Walks `tokens` in order, consuming characters from `s`:
///
///   * `Literal(t)` — must match `t` exactly at the current position
///     (case-sensitive — SQL `LIKE` is case-sensitive by default; the
///     case-insensitive `ILIKE` variant is a separate code path).
///   * `One`        — consume exactly one char from `s` (fails if `s`
///     is exhausted).
///   * `Any`        — consume zero-or-more chars; the matcher tries
///     successive split points and recurses on the remainder. The classic
///     LIKE algorithm; runs in O(|s| * |pattern|) worst case which is
///     plenty for the dashboard-sized inputs the host fallback targets.
///
/// We work in Rust `char` units (Unicode scalar values) so `_` matches
/// exactly one Unicode character — same as SQL standard, and the obvious
/// expectation for users supplying patterns over Utf8 columns.
fn generic_match(s: &str, tokens: &[Token]) -> bool {
    // Operate on `Vec<char>` slices for O(1) length checks and indexing.
    // Strings in this engine are dashboard-sized, so the allocation cost
    // is dwarfed by the per-row Arrow access; if we ever profile this
    // we can revisit a byte-level matcher.
    let chars: Vec<char> = s.chars().collect();
    match_chars(&chars, tokens)
}

/// Recursive helper for [`generic_match`]. Splits on `Token::Any` to do
/// the LIKE backtracking; for `Literal`/`One` runs we consume linearly.
fn match_chars(s: &[char], tokens: &[Token]) -> bool {
    let mut s_idx = 0usize;
    for (i, tok) in tokens.iter().enumerate() {
        match tok {
            Token::Literal(lit) => {
                // Compare `lit`'s chars against `s[s_idx..]`.
                let lit_chars: Vec<char> = lit.chars().collect();
                if s_idx + lit_chars.len() > s.len() {
                    return false;
                }
                for (j, lc) in lit_chars.iter().enumerate() {
                    if s[s_idx + j] != *lc {
                        return false;
                    }
                }
                s_idx += lit_chars.len();
            }
            Token::One => {
                if s_idx >= s.len() {
                    return false;
                }
                s_idx += 1;
            }
            Token::Any => {
                // Try every split point of `s[s_idx..]` against the tail of
                // `tokens`. Trailing `%` after the last meaningful token
                // (no remaining tokens) just consumes the rest.
                let rest = &tokens[i + 1..];
                if rest.is_empty() {
                    return true;
                }
                let mut k = s_idx;
                loop {
                    if match_chars(&s[k..], rest) {
                        return true;
                    }
                    if k >= s.len() {
                        return false;
                    }
                    k += 1;
                }
            }
        }
    }
    // All tokens consumed; we match iff the whole string was consumed too.
    s_idx == s.len()
}

/// Apply a SQL `LIKE` / `NOT LIKE` predicate to a Utf8 column, producing
/// a `BooleanArray` whose null bitmap mirrors the input's: `NULL LIKE
/// 'pat'` is `NULL` (SQL 3VL — never `false`).
///
/// Non-null rows produce `true` / `false` per the [`PatternMatcher`]
/// rules. `negated` inverts the per-row boolean but preserves the
/// validity bitmap: `NULL NOT LIKE 'pat'` is still `NULL`.
///
/// This is the entry point the host-side filter executor and the
/// expression evaluator both call into — see
/// [`crate::exec::filter::execute_filter`] and
/// [`crate::exec::expr_agg::eval_expr`].
pub fn host_like(col: &StringArray, pattern: &str, negated: bool) -> BoltResult<BooleanArray> {
    let matcher = PatternMatcher::compile(pattern, None)?;
    let n = col.len();
    let mut pairs: Vec<Option<bool>> = Vec::with_capacity(n);
    for i in 0..n {
        if col.is_null(i) {
            // SQL 3VL: `NULL LIKE 'pat'` is NULL, regardless of `negated`.
            pairs.push(None);
        } else {
            let m = matcher.matches(col.value(i));
            pairs.push(Some(if negated { !m } else { m }));
        }
    }
    // BooleanArray::from(Vec<Option<bool>>) preserves validity bits.
    Ok(BooleanArray::from(pairs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;

    /// Helper: compile + match in one call.
    fn m(pattern: &str, s: &str) -> bool {
        PatternMatcher::compile(pattern, None)
            .unwrap()
            .matches(s)
    }

    #[test]
    fn exact_match_no_wildcards() {
        assert!(m("foo", "foo"));
        assert!(!m("foo", "bar"));
        assert!(!m("foo", "foobar"));
        assert!(!m("foo", "fo"));
    }

    #[test]
    fn prefix_pattern() {
        assert!(m("foo%", "foo"));
        assert!(m("foo%", "foobar"));
        assert!(!m("foo%", "bar"));
        assert!(!m("foo%", "fo"));
        // Empty prefix → matches everything.
        assert!(m("%", ""));
        assert!(m("%", "anything"));
    }

    #[test]
    fn suffix_pattern() {
        assert!(m("%foo", "foo"));
        assert!(m("%foo", "barfoo"));
        assert!(!m("%foo", "foobar"));
        assert!(!m("%foo", "bar"));
    }

    #[test]
    fn contains_pattern() {
        assert!(m("%foo%", "foo"));
        assert!(m("%foo%", "abcfoodef"));
        assert!(m("%foo%", "foobar"));
        assert!(m("%foo%", "barfoo"));
        assert!(!m("%foo%", "bar"));
        // Empty middle (`%%`) matches everything.
        assert!(m("%%", ""));
        assert!(m("%%", "hi"));
    }

    #[test]
    fn underscore_matches_exactly_one() {
        assert!(m("f_o", "foo"));
        assert!(m("f_o", "fbo"));
        assert!(!m("f_o", "fo"));
        assert!(!m("f_o", "fooo"));
    }

    #[test]
    fn generic_mixed_pattern() {
        // `%foo_bar%` — contains "foo<any>bar" anywhere.
        assert!(m("%foo_bar%", "foo_bar"));
        assert!(m("%foo_bar%", "fooXbar"));
        assert!(m("%foo_bar%", "abcfoo!barxyz"));
        assert!(!m("%foo_bar%", "foobar"));
        assert!(!m("%foo_bar%", "fooo"));
    }

    #[test]
    fn unicode_underscore_is_one_codepoint() {
        // `_` matches exactly one Unicode scalar.
        assert!(m("h_llo", "héllo"));
        assert!(m("h_llo", "hxllo"));
        // The full emoji is one scalar (single codepoint U+1F600).
        assert!(m("a_b", "a\u{1F600}b"));
    }

    #[test]
    fn host_like_handles_nulls_3vl() {
        let arr = StringArray::from(vec![Some("foo"), None, Some("bar"), None, Some("fool")]);
        let out = host_like(&arr, "foo%", false).expect("ok");
        assert_eq!(out.len(), 5);
        // foo, NULL, bar, NULL, fool  → t, NULL, f, NULL, t
        assert_eq!(out.value(0), true);
        assert!(out.is_null(1));
        assert_eq!(out.value(2), false);
        assert!(out.is_null(3));
        assert_eq!(out.value(4), true);
    }

    #[test]
    fn host_like_negated_preserves_nulls() {
        // NOT LIKE: `NULL NOT LIKE 'pat'` is still NULL, not TRUE.
        let arr = StringArray::from(vec![Some("foo"), None, Some("bar")]);
        let out = host_like(&arr, "foo", true).expect("ok");
        assert_eq!(out.value(0), false);
        assert!(out.is_null(1), "NULL NOT LIKE 'pat' must be NULL");
        assert_eq!(out.value(2), true);
    }

    #[test]
    fn escape_clause_rejected_for_v05() {
        // The SQL frontend rejects ESCAPE at parse time, but
        // PatternMatcher::compile also rejects so hand-built plans
        // surface the same error.
        let err = PatternMatcher::compile("foo", Some('\\')).expect_err("must reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("ESCAPE"),
            "expected ESCAPE-related error, got: {msg}"
        );
    }

    /// `%%` (multiple `%`) collapses cleanly — and matches everything.
    #[test]
    fn double_percent_matches_all() {
        assert!(m("%%", ""));
        assert!(m("%%", "anything"));
        assert!(m("%%%", "anything"));
    }

    /// Pattern with internal `%` (not one of the fast paths) routes
    /// through the generic matcher and still works.
    #[test]
    fn internal_percent_uses_generic_matcher() {
        // `fo%o` — starts with "fo", ends with "o", anything in between.
        assert!(m("fo%o", "foo"));
        assert!(m("fo%o", "fooo"));
        assert!(m("fo%o", "fobaro"));
        assert!(!m("fo%o", "bar"));
        assert!(!m("fo%o", "foob"));
    }
}
