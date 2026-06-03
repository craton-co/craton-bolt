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
//! `expr LIKE 'pat' ESCAPE '<c>'` is supported as of v0.7. The escape
//! character is a single-codepoint literal known at compile time (the SQL
//! frontend enforces single-char). When the matcher encounters the escape
//! character in the pattern, the *following* character is interpreted
//! literally (no wildcard semantics): `'\%'` matches a literal `%`,
//! `'\_'` matches a literal `_`, and `'\\'` matches a literal `\` when
//! `ESCAPE '\'` is in effect. An escape character with no following char
//! (trailing escape at end of pattern) is a malformed pattern and surfaces
//! as a clear error from [`PatternMatcher::compile`]. Choosing an escape
//! character equal to `%` or `_` is rejected for the same reason — it
//! would make the wildcard unreachable.
//!
//! The fast-path shape classifier ([`classify`]) deliberately routes any
//! pattern carrying an escape character straight to [`Shape::Generic`] —
//! the prefix/suffix/contains fast paths reason about raw `%` positions
//! and cannot account for escaped occurrences without a second scan.
//! The generic matcher handles escape semantics uniformly via the
//! tokeniser.

use arrow_array::{Array, BooleanArray, StringArray};

use crate::error::{BoltError, BoltResult};

/// Case-insensitive single-character equality used by the `ILIKE` path.
///
/// Two `char`s compare equal when they are the same scalar, or when their
/// Unicode `to_lowercase` expansions are equal scalar-for-scalar (this also
/// covers ASCII case differences). Crucially this is a *per-character*
/// comparison: it never collapses or expands the input relative to the
/// pattern, so one input `char` always maps to exactly one pattern position.
///
/// ## Length-fold hazard (the bug this fixes)
///
/// The previous `ILIKE` implementation pre-folded *both* the whole pattern
/// and the whole input with [`str::to_lowercase`] and reused the
/// case-sensitive matcher. But `to_lowercase` is **not length-preserving**:
/// some scalars lowercase to several chars (e.g. `'İ'` U+0130 →
/// `"i\u{0307}"`, two chars). Folding a flat string can therefore change the
/// number of chars on one side of a `_`/literal boundary but not the other,
/// so positional `_` ("exactly one char"), prefix/suffix/contains matching
/// silently produced wrong results — e.g. `'a_b' ILIKE 'aİb'` returned
/// `false` where Postgres/DuckDB return `true`. Comparing char-by-char and
/// consuming exactly one input char per pattern position removes the hazard
/// entirely.
fn chars_eq_ci(a: char, b: char) -> bool {
    a == b || a.to_lowercase().eq(b.to_lowercase())
}

/// Compiled LIKE pattern, ready for fast-path evaluation per row.
///
/// Constructed via [`PatternMatcher::compile`] (validates the pattern and
/// picks a [`Shape`]); evaluated per cell via [`PatternMatcher::matches`].
#[derive(Debug, Clone)]
pub struct PatternMatcher {
    /// Detected shape — drives the dispatch in `matches`.
    shape: Shape,
    /// When `true` (compiled for `ILIKE`), the pattern was Unicode
    /// case-folded (`to_lowercase`) at compile time and the input is
    /// case-folded the same way before matching, yielding case-insensitive
    /// comparison. When `false` (plain `LIKE`), matching is byte-for-byte
    /// case-sensitive — the original, unchanged behaviour.
    case_insensitive: bool,
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
    /// When `escape` is `Some(c)`, the pattern is interpreted with `c` as
    /// the escape character: a `c` in the pattern causes the *next*
    /// character to be treated as a literal (no wildcard semantics). The
    /// escape character itself does not appear in the matched input.
    ///
    /// Errors on:
    /// * trailing escape character (e.g. pattern `"abc\"` with
    ///   `ESCAPE '\'`) — no character follows the escape;
    /// * escape character equal to `%` or `_` — would render the wildcard
    ///   unreachable, which is almost certainly a user mistake.
    pub fn compile(pattern: &str, escape: Option<char>) -> BoltResult<Self> {
        Self::compile_ci(pattern, escape, false)
    }

    /// Compile a SQL LIKE pattern with optional case-insensitivity.
    ///
    /// When `case_insensitive` is `false` this is exactly [`compile`] — the
    /// pattern, escape rules, and matching are byte-for-byte case-sensitive
    /// (the plain `LIKE` path, unchanged).
    ///
    /// When `case_insensitive` is `true` (the `ILIKE` path) the pattern is
    /// stored *unfolded* and matched with a per-character case-insensitive
    /// comparison ([`chars_eq_ci`]) in [`matches`]. We deliberately do NOT
    /// pre-fold the pattern (or input) to a flat lowercased string: that is
    /// not length-preserving and breaks the one-char-per-position invariant
    /// the `_`/prefix/suffix/contains matchers rely on (see the detailed
    /// hazard note on [`chars_eq_ci`]). For the same reason the ILIKE path
    /// always compiles to [`Shape::Generic`], whose char-level matcher is the
    /// one place we can compare per character; the byte-oriented fast-path
    /// shapes (`starts_with`/`ends_with`/`contains`/`==`) cannot do
    /// case-insensitive comparison without re-introducing the fold hazard.
    /// Wildcards (`%`, `_`) and the escape char are unaffected by case, so
    /// tokenisation and escape semantics are identical to the case-sensitive
    /// path.
    ///
    /// [`compile`]: PatternMatcher::compile
    /// [`matches`]: PatternMatcher::matches
    pub fn compile_ci(
        pattern: &str,
        escape: Option<char>,
        case_insensitive: bool,
    ) -> BoltResult<Self> {
        if let Some(c) = escape {
            if c == '%' || c == '_' {
                return Err(BoltError::Plan(format!(
                    "LIKE ESCAPE character must not be a wildcard (got {c:?})"
                )));
            }
        }
        let shape = if case_insensitive {
            // Force the char-level generic matcher: it is the only shape that
            // can compare case-insensitively per character without folding a
            // whole string (which is not length-preserving — see
            // [`chars_eq_ci`]). `tokenise` is case-neutral, so escape / `%` /
            // `_` semantics are preserved exactly.
            Shape::Generic(tokenise(pattern, escape)?)
        } else {
            classify(pattern, escape)?
        };
        Ok(Self {
            shape,
            case_insensitive,
        })
    }

    /// True if `s` matches this compiled pattern.
    ///
    /// The input is never pre-folded: for the `ILIKE` path the pattern is a
    /// [`Shape::Generic`] and comparison happens per character inside
    /// [`generic_match`] (case-insensitively when `self.case_insensitive`).
    /// This avoids the length-changing-fold hazard described on
    /// [`chars_eq_ci`]. Case-sensitive `LIKE` dispatches exactly as before.
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
            // `ILIKE` always lands here (see `compile_ci`); the case-sensitive
            // generic path passes `case_insensitive = false`.
            Shape::Generic(tokens) => generic_match(s, tokens, self.case_insensitive),
        }
    }
}

/// Classify the pattern into a fast-path [`Shape`] when possible, falling
/// back to [`Shape::Generic`] for anything else.
///
/// The fast-path recognisers all require the pattern to contain *no* `_`
/// wildcards (those force per-character matching, so they go to Generic),
/// and the `%` placements must match one of the four canonical shapes.
///
/// When an escape character is in effect, any pattern that *contains* the
/// escape character routes straight to [`Shape::Generic`] — the fast paths
/// can't reason about escaped vs. unescaped `%` without re-scanning, and
/// the generic matcher already handles escape semantics via [`tokenise`].
/// The escape character is also validated here: a trailing escape (no
/// following char) is rejected as a malformed pattern.
fn classify(pattern: &str, escape: Option<char>) -> BoltResult<Shape> {
    // Any pattern that carries the escape character cannot use the prefix /
    // suffix / contains / exact fast paths (those look at raw `%` / `_`
    // positions, ignoring escapes). Route to Generic so [`tokenise`] can
    // apply escape semantics uniformly.
    if let Some(c) = escape {
        if pattern.contains(c) {
            return Ok(Shape::Generic(tokenise(pattern, Some(c))?));
        }
    }
    // Bail out to Generic immediately if there's any `_` — fast paths only
    // handle `%`-style shapes.
    if pattern.contains('_') {
        return Ok(Shape::Generic(tokenise(pattern, escape)?));
    }
    let n_pct = pattern.chars().filter(|c| *c == '%').count();
    if n_pct == 0 {
        // No wildcards at all → exact match.
        return Ok(Shape::Exact(pattern.to_string()));
    }
    if n_pct == 1 {
        if let Some(rest) = pattern.strip_suffix('%') {
            // `foo%` — prefix. (Pattern `%` alone strips to empty rest →
            // prefix "" matches everything, which is the correct semantic.)
            return Ok(Shape::Prefix(rest.to_string()));
        }
        if let Some(rest) = pattern.strip_prefix('%') {
            // `%foo` — suffix.
            return Ok(Shape::Suffix(rest.to_string()));
        }
        // `%` was internal (`fo%o`) — not one of the fast-path shapes.
        return Ok(Shape::Generic(tokenise(pattern, escape)?));
    }
    if n_pct == 2 {
        // Look for the canonical `%foo%` shape: starts with `%`, ends with
        // `%`, no other `%` in between.
        if let Some(mid) = pattern.strip_prefix('%').and_then(|s| s.strip_suffix('%')) {
            // The middle slice contains no `%` (we already accounted for
            // exactly two in the original) and no `_` (we bailed out above
            // for that), so `Contains` is correct.
            return Ok(Shape::Contains(mid.to_string()));
        }
    }
    // Anything else (multiple internal `%`s, mix of `_` already handled
    // above) → generic matcher.
    Ok(Shape::Generic(tokenise(pattern, escape)?))
}

/// Split `pattern` into a `Vec<Token>` for the generic matcher.
///
/// Runs of literal characters become one [`Token::Literal`]; each `_` and
/// `%` becomes its own token. The matcher then walks tokens linearly,
/// backtracking on `%` when a subsequent literal/`_` fails to match.
///
/// When `escape` is `Some(c)`, a `c` in the pattern marks the next
/// character as a literal: `c%` emits a literal `%`, `c_` emits a literal
/// `_`, and `cc` emits a literal `c`. The escape character itself does
/// not appear in the emitted tokens. A trailing escape (`...c` with no
/// following char) is rejected — it's a malformed pattern.
fn tokenise(pattern: &str, escape: Option<char>) -> BoltResult<Vec<Token>> {
    let mut out: Vec<Token> = Vec::new();
    let mut buf = String::new();
    // Hold the iterator by-ref so the loop body can also call `next()` to
    // consume the character *after* an escape — that two-character peek is
    // the whole point of escape handling.
    let mut chars = pattern.chars();
    #[allow(clippy::while_let_on_iterator)]
    while let Some(ch) = chars.next() {
        // Handle escape *before* wildcard interpretation so `\%` and `\_`
        // become literal `%` / `_` rather than the wildcards they would
        // otherwise be.
        if let Some(esc) = escape {
            if ch == esc {
                match chars.next() {
                    Some(next) => {
                        // Any character following the escape is a literal —
                        // wildcards (`%`, `_`), the escape char itself
                        // (`\\`), or any other character (`\a` → literal
                        // `a`, which is harmless / standard SQL behaviour).
                        buf.push(next);
                        continue;
                    }
                    None => {
                        return Err(BoltError::Plan(format!(
                            "LIKE pattern ends with a dangling escape character {esc:?}"
                        )));
                    }
                }
            }
        }
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
    Ok(out)
}

/// Generic LIKE matcher: non-recursive char-level matching over `s`
/// against `tokens`. Returns `true` iff the whole string matches the whole
/// pattern.
///
/// Token semantics (unchanged from the previous implementation):
///
///   * `Literal(t)` — must match `t` exactly at the current position
///     (case-sensitive — SQL `LIKE` is case-sensitive by default; the
///     case-insensitive `ILIKE` variant is a separate code path).
///   * `One`        — consume exactly one char from `s` (fails if `s`
///     is exhausted).
///   * `Any` (`%`)  — match zero-or-more chars.
///
/// ## Complexity (V-6)
///
/// The previous implementation recursed on every `%` split point, which is
/// **exponential** for multi-`%` patterns (e.g. `'%a%a%a%a%a%b'` over a
/// long non-matching string — a ReDoS / catastrophic-backtracking DoS,
/// since LIKE patterns are user-supplied). The old doc-comment's
/// `O(|s| * |pattern|)` claim was therefore false.
///
/// We now use the standard **non-recursive greedy two-pointer** LIKE
/// algorithm with a single backtrack pointer. It walks `s` and `tokens`
/// in lockstep; on a `%` it records a resume point (`star_tok` / `star_s`)
/// and greedily assumes the `%` matches the empty string. On any later
/// literal/`_` mismatch it backtracks: advance `star_s` by one char,
/// resume matching the tokens right after the recorded `%`. Because each
/// `%` is recorded only once and `star_s` only ever advances, the total
/// work is bounded by `O(|s| * |pattern|)` — now a genuinely true bound,
/// with no recursion and no exponential blowup.
///
/// We work in Rust `char` units (Unicode scalar values) so `_` matches
/// exactly one Unicode character — same as SQL standard, and the obvious
/// expectation for users supplying patterns over Utf8 columns.
///
/// When `case_insensitive` is `true` (the `ILIKE` path), literal pattern
/// chars are compared to input chars with [`chars_eq_ci`] — a *per-character*
/// case-insensitive comparison that still consumes exactly one input `char`
/// per literal pattern char and one per `_`. This preserves the
/// one-char-per-position invariant; see the length-fold hazard note on
/// [`chars_eq_ci`].
fn generic_match(s: &str, tokens: &[Token], case_insensitive: bool) -> bool {
    // Collect the input once into `Vec<char>` for O(1) indexing / length;
    // literal tokens are compared char-by-char directly off their `&str`
    // (no per-token `Vec<char>` allocation — V-6 cleanup).
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();

    // Two pointers: position in the input (`s_idx`) and position in the
    // token stream (`t_idx`). Backtrack state: `star_tok` is the index of
    // the token *after* the most recent `Any`, and `star_s` is the input
    // position to resume from if the greedy assumption fails.
    let mut s_idx = 0usize;
    let mut t_idx = 0usize;
    let mut star_tok: Option<usize> = None;
    let mut star_s = 0usize;

    loop {
        if t_idx < tokens.len() {
            match &tokens[t_idx] {
                Token::Any => {
                    // Record a backtrack point: assume `%` matches empty
                    // for now, and remember where to resume if that fails.
                    star_tok = Some(t_idx + 1);
                    star_s = s_idx;
                    t_idx += 1;
                    continue;
                }
                Token::One => {
                    if s_idx < n {
                        s_idx += 1;
                        t_idx += 1;
                        continue;
                    }
                    // else fall through to backtrack / fail below.
                }
                Token::Literal(lit) => {
                    // Match `lit`'s chars against `chars[s_idx..]` without
                    // allocating a `Vec<char>` for the literal.
                    let mut k = s_idx;
                    let mut ok = true;
                    for lc in lit.chars() {
                        // Case-insensitive (ILIKE) compares per character via
                        // `chars_eq_ci`; case-sensitive uses plain `==`. Both
                        // consume exactly one input char per literal char, so
                        // positions stay aligned regardless of case folding.
                        let hit = k < n
                            && if case_insensitive {
                                chars_eq_ci(chars[k], lc)
                            } else {
                                chars[k] == lc
                            };
                        if hit {
                            k += 1;
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        s_idx = k;
                        t_idx += 1;
                        continue;
                    }
                    // else fall through to backtrack / fail below.
                }
            }
        } else if s_idx == n {
            // All tokens consumed and the whole string consumed → match.
            return true;
        }
        // Reached here because either: a literal/`_` failed to match, or
        // tokens ran out before the input did. Backtrack to the last `%`
        // if there was one — advance the resume position by one char and
        // retry the post-`%` tokens. Otherwise no match.
        match star_tok {
            Some(tok) if star_s < n => {
                star_s += 1;
                s_idx = star_s;
                t_idx = tok;
            }
            _ => return false,
        }
    }
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
pub fn host_like(
    col: &StringArray,
    pattern: &str,
    escape: Option<char>,
    negated: bool,
) -> BoltResult<BooleanArray> {
    let matcher = PatternMatcher::compile(pattern, escape)?;
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

    /// Helper: compile + match in one call (no ESCAPE clause).
    fn m(pattern: &str, s: &str) -> bool {
        PatternMatcher::compile(pattern, None).unwrap().matches(s)
    }

    /// Helper: compile + match with an explicit ESCAPE character.
    fn me(pattern: &str, escape: char, s: &str) -> bool {
        PatternMatcher::compile(pattern, Some(escape))
            .unwrap()
            .matches(s)
    }

    /// Helper: case-insensitive (ILIKE) compile + match, no ESCAPE.
    fn mi(pattern: &str, s: &str) -> bool {
        PatternMatcher::compile_ci(pattern, None, true)
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
        let out = host_like(&arr, "foo%", None, false).expect("ok");
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
        let out = host_like(&arr, "foo", None, true).expect("ok");
        assert_eq!(out.value(0), false);
        assert!(out.is_null(1), "NULL NOT LIKE 'pat' must be NULL");
        assert_eq!(out.value(2), true);
    }

    // ─── ESCAPE clause (v0.7) ────────────────────────────────────────────

    /// Canonical task example: `'a\%b' ESCAPE '\'` matches literal
    /// `'a%b'` and does NOT match `'a_b'` (the escaped `%` no longer acts
    /// as a wildcard).
    #[test]
    fn escape_literalises_percent() {
        assert!(me(r"a\%b", '\\', "a%b"));
        assert!(!me(r"a\%b", '\\', "a_b"));
        // Also doesn't match arbitrary strings the way unescaped `%`
        // would.
        assert!(!me(r"a\%b", '\\', "axb"));
        assert!(!me(r"a\%b", '\\', "ab"));
        assert!(!me(r"a\%b", '\\', "aXYZb"));
    }

    /// Escape of `_` produces a literal `_` (no single-char wildcard).
    #[test]
    fn escape_literalises_underscore() {
        assert!(me(r"a\_b", '\\', "a_b"));
        assert!(!me(r"a\_b", '\\', "aXb"));
        assert!(!me(r"a\_b", '\\', "ab"));
    }

    /// Escape of the escape character itself — `'\\'` matches a literal
    /// backslash when `ESCAPE '\'`.
    #[test]
    fn escape_of_escape_yields_literal_escape_char() {
        assert!(me(r"a\\b", '\\', r"a\b"));
        assert!(!me(r"a\\b", '\\', "ab"));
        assert!(!me(r"a\\b", '\\', r"a\\b"));
    }

    /// Standard `%` and `_` wildcards still work when ESCAPE is set — the
    /// escape only changes the next char, not the wildcard semantics
    /// elsewhere in the pattern.
    #[test]
    fn unescaped_wildcards_still_work_when_escape_set() {
        // Unescaped `%` is still "zero-or-more".
        assert!(me("a%b", '\\', "ab"));
        assert!(me("a%b", '\\', "aXYZb"));
        assert!(!me("a%b", '\\', "a"));
        // Unescaped `_` is still "exactly one".
        assert!(me("a_b", '\\', "aXb"));
        assert!(!me("a_b", '\\', "ab"));
        // Mix: literal `%` then wildcard `_`.
        assert!(me(r"a\%_b", '\\', "a%Xb"));
        assert!(!me(r"a\%_b", '\\', "aXXb"));
    }

    /// Custom non-backslash escape character (`!`) — same rules apply.
    #[test]
    fn custom_escape_character() {
        assert!(me("a!%b", '!', "a%b"));
        assert!(!me("a!%b", '!', "aXb"));
        // Unescaped `%` in the same pattern still wildcards.
        assert!(me("a%!_b", '!', "axyz_b"));
        assert!(!me("a%!_b", '!', "axyzb"));
    }

    /// Trailing escape character (no char follows) is a malformed
    /// pattern — surface a clean error.
    #[test]
    fn dangling_escape_rejected() {
        let err = PatternMatcher::compile(r"abc\", Some('\\')).expect_err("must reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("dangling escape"),
            "expected dangling-escape error, got: {msg}"
        );
    }

    /// Choosing the escape character equal to a wildcard would render the
    /// wildcard unreachable — reject so users get a clear error rather
    /// than silently broken matching.
    #[test]
    fn escape_equal_to_wildcard_rejected() {
        let err = PatternMatcher::compile("foo", Some('%')).expect_err("must reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("wildcard"),
            "expected wildcard-conflict error, got: {msg}"
        );
        let err = PatternMatcher::compile("foo", Some('_')).expect_err("must reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("wildcard"),
            "expected wildcard-conflict error, got: {msg}"
        );
    }

    /// `host_like` honors the escape clause end-to-end with 3VL NULL
    /// semantics preserved.
    #[test]
    fn host_like_with_escape() {
        let arr = StringArray::from(vec![
            Some("a%b"), // matches literal `%`
            Some("a_b"), // does not match literal `%`
            None,        // NULL stays NULL
            Some("axb"), // unescaped `%` would match, escaped does not
        ]);
        let out = host_like(&arr, r"a\%b", Some('\\'), false).expect("ok");
        assert_eq!(out.value(0), true);
        assert_eq!(out.value(1), false);
        assert!(out.is_null(2));
        assert_eq!(out.value(3), false);
    }

    /// Patterns whose escape sequences happen to leave only literal
    /// characters still classify (via Generic) and match exactly.
    #[test]
    fn escape_pattern_routes_through_generic_shape() {
        // No wildcards survive after escaping — Generic shape with a
        // single Literal token matches the literal string only.
        assert!(me(r"\%\%", '\\', "%%"));
        assert!(!me(r"\%\%", '\\', "anything"));
        assert!(!me(r"\%\%", '\\', "%"));
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

    // ─── V-6: linear generic matcher (no catastrophic backtracking) ──────

    /// Regression test for finding V-6 (ReDoS / catastrophic backtracking).
    ///
    /// The pathological multi-`%` pattern `%a%a%a%a%a%a%a%a%b` against a
    /// long all-`a` string forced the old *recursive* matcher into
    /// exponential blowup. With the non-recursive two-pointer matcher this
    /// resolves in linear time. The string ends in `a` (not `b`), so the
    /// final literal `b` can never match → correct result is `false`.
    ///
    /// If this test ever hangs (rather than failing), the exponential
    /// matcher has regressed back in.
    #[test]
    fn v6_pathological_multi_percent_is_linear() {
        let pat = "%a%a%a%a%a%a%a%a%b";
        let s: String = std::iter::repeat('a').take(10_000).collect();
        assert!(!m(pat, &s), "no 'b' in the input → must not match");

        // Same pattern, but the input *does* end in `b` → must match,
        // also resolved quickly.
        let mut s2: String = std::iter::repeat('a').take(10_000).collect();
        s2.push('b');
        assert!(m(pat, &s2), "input ends in 'b' → must match");
    }

    /// `%` at start, end, and middle in a single generic pattern, plus the
    /// degenerate empty cases — exercise the two-pointer backtrack pointer.
    #[test]
    fn v6_percent_positions_and_empties() {
        // Leading `%`.
        assert!(m("%bar", "foobar"));
        assert!(!m("%bar", "barfoo"));
        // Trailing `%`.
        assert!(m("foo%", "foobar"));
        assert!(!m("foo%", "xfoo"));
        // Middle `%`.
        assert!(m("a%z", "az"));
        assert!(m("a%z", "abcz"));
        assert!(!m("a%z", "abc"));
        // Multiple internal `%` (generic path).
        assert!(m("a%b%c", "aXbYc"));
        assert!(m("a%b%c", "abc"));
        assert!(!m("a%b%c", "acb"));
        // Empty pattern matches only the empty string.
        assert!(m("", ""));
        assert!(!m("", "x"));
        // `%` alone matches everything including empty.
        assert!(m("%", ""));
        assert!(m("%", "anything"));
        // Empty string against a non-trivial generic pattern.
        assert!(!m("a%b", ""));
        // Generic pattern that reduces to all-`%`/`_` over empty input.
        assert!(!m("_", ""));
    }

    /// `_` (single-char wildcard) interleaved with `%` in the generic
    /// matcher, including trailing `_` after a `%`.
    #[test]
    fn v6_underscore_with_percent_generic() {
        assert!(m("a%_c", "abXc")); // `%`=>"b", `_`=>"X"
        assert!(m("a%_c", "aXc")); // `%`=>"", `_`=>"X"
        assert!(!m("a%_c", "ac")); // `_` needs one char between a and c
        assert!(m("%_", "x")); // at least one char
        assert!(!m("%_", "")); // empty fails the `_`
        assert!(m("_%", "x"));
        assert!(m("__", "ab"));
        assert!(!m("__", "a"));
    }

    /// Cross-check: the new matcher agrees with the documented LIKE
    /// semantics (and thus the previous implementation) on a handful of
    /// mixed cases routed through the generic matcher.
    #[test]
    fn v6_semantics_match_previous_on_samples() {
        // (pattern, input, expected) — all chosen to hit Shape::Generic
        // (internal `%`, or `_` present) so they exercise generic_match.
        let cases: &[(&str, &str, bool)] = &[
            ("a%a%b", "aaab", true),
            ("a%a%b", "aaa", false),
            ("%a%a%", "xaya", true),
            ("%a%a%", "a", false),
            ("f_o%", "foobar", true),
            ("f_o%", "fo", false),
            ("%_%", "x", true),
            ("%_%", "", false),
            ("a_%_b", "aXYb", true),
            ("a_%_b", "aXb", false),
            ("h_l%o", "héllxo", true),
            ("h_l%o", "hello", true),
        ];
        for (pat, s, want) in cases {
            assert_eq!(m(pat, s), *want, "pattern={pat:?} input={s:?}");
        }
    }

    // ─── ILIKE (case-insensitive) ────────────────────────────────────────

    /// Case-insensitive matching across every fast-path shape: pattern and
    /// input differ only in case yet still match.
    #[test]
    fn ilike_matches_across_case_all_shapes() {
        // Exact.
        assert!(mi("FOO", "foo"));
        assert!(mi("foo", "FOO"));
        assert!(mi("FoO", "fOo"));
        // Prefix.
        assert!(mi("Foo%", "FOOBAR"));
        assert!(mi("foo%", "FOObar"));
        // Suffix.
        assert!(mi("%Bar", "FOOBAR"));
        assert!(mi("%bar", "fooBAR"));
        // Contains.
        assert!(mi("%OoB%", "fOObar"));
        // Underscore (generic) is case-insensitive on the literal parts.
        assert!(mi("F_O", "foo"));
        assert!(mi("f_O", "FXO"));
        // Generic multi-`%`.
        assert!(mi("a%B%c", "AxbYC"));
    }

    /// Case-insensitive matching still respects the actual characters — a
    /// genuine mismatch is not masked by case folding.
    #[test]
    fn ilike_rejects_genuine_mismatch() {
        assert!(!mi("FOO", "bar"));
        assert!(!mi("foo%", "BAR"));
        assert!(!mi("%bar", "BARFOO"));
        assert!(!mi("f_o", "fooo"));
    }

    /// Unicode case folding: ILIKE folds non-ASCII letters too.
    #[test]
    fn ilike_unicode_case_fold() {
        assert!(mi("ÉCOLE", "école"));
        assert!(mi("école", "ÉCOLE"));
        assert!(mi("café%", "CAFÉTERIA"));
        assert!(!mi("café%", "TEACUP"));
    }

    /// Plain (case-sensitive) LIKE is UNCHANGED by the new flag: a case
    /// difference must NOT match on the `compile`/`m` path.
    #[test]
    fn plain_like_stays_case_sensitive() {
        assert!(!m("FOO", "foo"));
        assert!(!m("foo", "FOO"));
        assert!(!m("Foo%", "fooBAR"));
        assert!(!m("%Bar", "foobar"));
        // Same-case still matches (regression guard).
        assert!(m("foo", "foo"));
        assert!(m("foo%", "foobar"));
    }

    /// `host_like` (case-sensitive) keeps NOT-matching across case — verifies
    /// the default 4-arg entry point is the case-sensitive path.
    #[test]
    fn host_like_default_is_case_sensitive() {
        let arr = StringArray::from(vec![Some("FOO"), Some("foo")]);
        let out = host_like(&arr, "foo", None, false).expect("ok");
        assert_eq!(
            out.value(0),
            false,
            "FOO must NOT match case-sensitive 'foo'"
        );
        assert_eq!(out.value(1), true);
    }

    /// ILIKE preserves SQL 3VL NULL propagation: `NULL ILIKE 'x'` is NULL,
    /// not false — exercised here at the matcher level by confirming the
    /// matcher only sees non-NULL rows (NULL handling lives in the caller,
    /// `exec::expr_agg::eval_like` / `host_like`), and that a non-NULL row
    /// folds correctly. The end-to-end 3VL behaviour for ILIKE is identical
    /// to LIKE because both route through the same NULL-masking caller.
    #[test]
    fn ilike_compile_ci_matches_folded_only() {
        let mr = PatternMatcher::compile_ci("AB%", None, true).expect("ok");
        assert!(mr.matches("abc"));
        assert!(mr.matches("ABC"));
        assert!(!mr.matches("xyz"));
    }

    /// ILIKE with NOT semantics is applied by the caller; confirm the raw
    /// case-insensitive match result that the caller will invert.
    #[test]
    fn ilike_raw_match_for_not_ilike() {
        // `s NOT ILIKE 'foo'`: matcher says FOO matches 'foo' (true), the
        // caller inverts to false; "bar" does not match (false) → caller
        // inverts to true.
        assert!(mi("foo", "FOO"));
        assert!(!mi("foo", "bar"));
    }

    /// Escape semantics compose with case-insensitivity: an escaped literal
    /// still matches case-insensitively.
    #[test]
    fn ilike_with_escape_is_case_insensitive() {
        let mr = PatternMatcher::compile_ci(r"A\%B", Some('\\'), true).expect("ok");
        assert!(mr.matches("a%b"));
        assert!(mr.matches("A%B"));
        assert!(!mr.matches("axb"));
    }

    /// Regression: a pattern char whose `to_lowercase()` is NOT
    /// length-preserving must still count as exactly one position so that a
    /// `_` on the other side stays aligned.
    ///
    /// `'İ'` (U+0130 LATIN CAPITAL LETTER I WITH DOT ABOVE) lowercases to the
    /// TWO chars `"i\u{0307}"`. The old ILIKE implementation pre-folded the
    /// whole pattern (`'aİb'` → `"ai\u{0307}b"`, 4 chars) and the whole input
    /// (`'a_b'` matched against the folded pattern), which moved the boundary
    /// and made `'a_b' ILIKE 'aİb'` return `false`. Postgres/DuckDB return
    /// `true` because `_` matches the single input char `İ`. The per-char
    /// comparison fixes this without changing length on either side.
    #[test]
    fn ilike_pattern_char_with_length_changing_lowercase() {
        // `_` in the pattern must match the single input scalar `İ`.
        assert!(mi("a_b", "a\u{0130}b"));
        // And a literal `İ` in the pattern matches itself case-insensitively
        // (same scalar) without expanding the position count.
        assert!(mi("a\u{0130}b", "a\u{0130}b"));
        // A `_` either side of the cased char also stays aligned.
        assert!(mi("_\u{0130}_", "x\u{0130}y"));
        assert!(mi("\u{0130}_", "\u{0130}z"));
        // Genuine mismatch is still rejected (length fold doesn't mask it).
        assert!(!mi("a_b", "a\u{0130}bc"));
        assert!(!mi("a\u{0130}", "a\u{0130}b"));
    }

    /// Prefix/suffix/contains-shaped ILIKE patterns around a
    /// length-changing-lowercase char also stay correct — they compile to the
    /// generic char matcher under ILIKE, so the boundary never drifts.
    #[test]
    fn ilike_length_fold_in_anchored_shapes() {
        // Prefix `aİ%` over input whose `İ` differs only conceptually.
        assert!(mi("a\u{0130}%", "a\u{0130}xyz"));
        // Suffix `%İb`.
        assert!(mi("%\u{0130}b", "xyz\u{0130}b"));
        // Contains `%İ%`.
        assert!(mi("%\u{0130}%", "xy\u{0130}zw"));
        // Non-match still rejected.
        assert!(!mi("a\u{0130}%", "ab\u{0130}xyz"));
    }
}
