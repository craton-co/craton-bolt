// SPDX-License-Identifier: Apache-2.0

//! v0.5 / M2 — SQL-standard case folding for identifiers.
//!
//! Pin the behaviour that:
//!   * Unquoted identifiers fold to ASCII lowercase at parse time, so a
//!     SQL query can use any casing and still resolve against a
//!     verbatim-cased registered schema.
//!   * Quoted identifiers (`"MyCol"`) keep their case verbatim, and so do
//!     NOT match a lowercase schema field. This is the standard rule.
//!
//! These tests go through the public `parse_sql` + `lower_physical` API,
//! which is the same path applications use, so a future refactor that
//! breaks the resolver fallback or the parse-time fold rule fails here
//! loudly rather than silently regressing user queries.

use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::{lower_physical, parse_sql, DataType, Field, MemTableProvider, Schema};

/// `users(name, age)` with verbatim lowercase column names — used by the
/// "SQL uppercase finds lowercase schema field" tests.
fn lower_users_provider() -> MemTableProvider {
    let schema = Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]);
    MemTableProvider::new().with_table("users", schema)
}

/// `USERS(NAME, age)` — table registered uppercase, with one uppercase and
/// one lowercase column. Lets the tests exercise the "registered uppercase,
/// SQL lowercase" direction symmetrically.
fn upper_users_provider() -> MemTableProvider {
    let schema = Schema::new(vec![
        Field::new("NAME", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]);
    MemTableProvider::new().with_table("USERS", schema)
}

/// `MyTable(MyCol)` with mixed-case names — used by the quoted-identifier
/// case-preservation tests.
fn mixed_case_provider() -> MemTableProvider {
    let schema = Schema::new(vec![Field::new("MyCol", DataType::Int32, false)]);
    MemTableProvider::new().with_table("MyTable", schema)
}

/// Smoke: parse + lower must succeed. Panics with the failure message if
/// either step errors. Used by the positive tests below.
fn parse_and_lower_ok(sql: &str, provider: &MemTableProvider) {
    let plan = parse_sql(sql, provider).unwrap_or_else(|e| panic!("parse failed for {sql:?}: {e}"));
    lower_physical(&plan).unwrap_or_else(|e| panic!("lower failed for {sql:?}: {e}"));
}

// ---------------------------------------------------------------------------
// 1. SELECT NAME FROM users — column reference uses uppercase, schema has
//    lowercase `name`. The folded `Expr::Column("name")` (lowercase) hits the
//    verbatim schema field directly; the fallback isn't even needed.
// ---------------------------------------------------------------------------

#[test]
fn select_uppercase_column_against_lowercase_schema() {
    let provider = lower_users_provider();
    parse_and_lower_ok("SELECT NAME FROM users", &provider);
}

#[test]
fn select_mixedcase_column_against_lowercase_schema() {
    let provider = lower_users_provider();
    parse_and_lower_ok("SELECT Name FROM Users", &provider);
}

// ---------------------------------------------------------------------------
// 2. SELECT name FROM USERS — table reference uses uppercase, schema is
//    registered as `USERS`. The folded `users` misses the verbatim table
//    map entry and resolves via the case-insensitive fallback in
//    `MemTableProvider::schema` / `NameResolver::resolve_compound`.
// ---------------------------------------------------------------------------

#[test]
fn select_lowercase_table_against_uppercase_registration() {
    let provider = upper_users_provider();
    // `name` (lowercase, folded) misses the verbatim `NAME` field and
    // resolves via `Schema::index_of`'s case-insensitive fallback; `users`
    // misses `USERS` and resolves via the provider's fallback.
    parse_and_lower_ok("SELECT name FROM users", &provider);
}

#[test]
fn where_clause_folds_against_uppercase_registration() {
    let provider = upper_users_provider();
    // Both `NAME` (folded → `name`, fallback to schema field `NAME`) and
    // the bare `USERS` table must resolve.
    parse_and_lower_ok("SELECT name FROM USERS WHERE age > 18", &provider);
}

// ---------------------------------------------------------------------------
// 3. Quoted identifiers preserve case verbatim. `"Name"` (quoted) does NOT
//    match the lowercase `name` field — that is the SQL-standard behaviour.
// ---------------------------------------------------------------------------

#[test]
fn quoted_identifier_is_case_sensitive_and_misses_lowercase_schema() {
    let provider = lower_users_provider();
    // The schema only has `name` (lowercase). A quoted `"Name"` keeps
    // its case verbatim. The lookup-key carries an uppercase letter, so
    // `Schema::index_of` suppresses its case-insensitive fallback and
    // the miss is final — `"Name"` does NOT resolve to `name`. This is
    // the SQL-standard rule for quoted identifiers.
    let res = parse_sql(r#"SELECT "Name" FROM users"#, &provider)
        .and_then(|p| lower_physical(&p).map(|_| ()));
    assert!(
        res.is_err(),
        r#"quoted "Name" must not match lowercase `name` (case-sensitive standard)"#
    );
}

#[test]
fn quoted_identifier_matches_verbatim() {
    // Symmetric to the miss case above: a quoted identifier with the
    // exact registered casing resolves cleanly. We verify both column
    // and table forms in `mixed_case_provider` below
    // (`quoted_table_name_preserves_case`); this test pins the simple
    // column-only path so a regression to "quoted always misses" is
    // caught here too.
    let provider = upper_users_provider();
    parse_and_lower_ok(r#"SELECT "NAME" FROM USERS"#, &provider);
}

#[test]
fn quoted_table_name_preserves_case() {
    let provider = mixed_case_provider();
    // Both quoted and unquoted forms must resolve `MyTable` / `MyCol`:
    //   * `"MyTable"."MyCol"` is verbatim — direct hit.
    //   * `MyTable.MyCol` is folded to `mytable.mycol`, which hits the
    //     case-insensitive fallback in the resolver and schema.
    parse_and_lower_ok(r#"SELECT "MyCol" FROM "MyTable""#, &provider);
    parse_and_lower_ok("SELECT MyCol FROM MyTable", &provider);
}

#[test]
fn quoted_mismatch_with_different_letters_errors() {
    // Verifies that *quoted* identifiers are still case-sensitive in the
    // sense that they pass through to the lookup verbatim — a typo in a
    // quoted identifier surfaces an unknown-column error rather than
    // silently resolving to a same-letters-but-different-case schema
    // field.
    let provider = mixed_case_provider();
    let res = parse_sql(r#"SELECT "NoSuchCol" FROM "MyTable""#, &provider)
        .and_then(|p| lower_physical(&p).map(|_| ()));
    assert!(
        res.is_err(),
        r#"quoted "NoSuchCol" must not match `MyCol` (different letters)"#
    );
}

// ---------------------------------------------------------------------------
// 4. Aliases and compound identifiers fold consistently.
// ---------------------------------------------------------------------------

#[test]
fn unquoted_column_alias_is_folded() {
    let provider = lower_users_provider();
    // `AS Total` should be visible to downstream stages as `total`. We
    // only smoke-test that the query parses and lowers — observing the
    // alias case in the lowered plan would require the LogicalPlan
    // pattern-matching scaffolding from the in-crate plan tests.
    parse_and_lower_ok("SELECT age AS Total FROM users", &provider);
}

#[test]
fn compound_identifier_folds_qualifier_and_column() {
    // No JOINs in our public API surface, but a single-table compound
    // identifier `users.name` is accepted by the resolver and should fold
    // both segments. The plan must lower successfully against the
    // lowercase schema.
    let provider = lower_users_provider();
    parse_and_lower_ok("SELECT users.NAME FROM users", &provider);
    parse_and_lower_ok("SELECT USERS.name FROM USERS", &provider);
}

// ===========================================================================
// 5. ILIKE (case-insensitive) case folding.
//
// `case_folding_test.rs` previously only exercised *identifier* folding
// (UPPER/LOWER casing of column/table names) via parse_sql + lower_physical.
// This is the designated home for case-folding coverage, so the ILIKE
// value-folding tests live here too.
//
// CONTEXT: `str::to_lowercase()` is not length-preserving — e.g. `'İ'`
// (U+0130 LATIN CAPITAL LETTER I WITH DOT ABOVE) lowercases to TWO chars
// (`"i\u{0307}"`). The old ILIKE path pre-folded both pattern and input into
// flat lowercased strings and reused the case-sensitive matcher, so a fold
// that changed length on one side of a `_`/literal boundary (but not the
// other) silently produced wrong results: `'a_b' ILIKE 'aİb'` returned
// `false` where Postgres/DuckDB return `true`. The fix folds *per character*
// (one input char → exactly one pattern position).
//
// Like the execution tests in `tests/string_ops_e2e.rs`, these run the full
// `Engine::sql` pipeline and therefore require a CUDA device — they carry
// `#[ignore = "gpu:string"]` so they compile in CI but only run under
// `cargo test -- --ignored --filter gpu:string` on a GPU host. Each is
// paired with a non-gated parse/lower guard so the SQL-frontend surface for
// ILIKE is pinned even without a device.
//
// The engine does not compact filtered output: masked (non-matching) rows
// keep their zero-init value. We therefore SELECT a companion Int64 `v`
// column alongside the `s ILIKE '...'` predicate and collect the non-zero
// `v` values — exactly the pattern `string_ops_e2e.rs` uses — so a row
// "matches" iff its `v` survives. Every fixture uses 1-based, all-distinct,
// non-zero `v` values so zero-init masking never collides with a real row.
// ===========================================================================

/// Two-column fixture: a Utf8 `s` (the ILIKE target) and an Int64 `v` (a
/// row tag used to read back which rows matched). Mirrors the
/// `string_ops_e2e.rs::sv_batch` shape so the engine takes the same path.
fn sv_batch(strings: &[&str], values: &[i64]) -> RecordBatch {
    let s_arr: StringArray = StringArray::from(strings.to_vec());
    let v_arr: Int64Array = Int64Array::from(values.to_vec());
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("s", ArrowDataType::Utf8, false),
        ArrowField::new("v", ArrowDataType::Int64, false),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(s_arr), Arc::new(v_arr)]).unwrap()
}

/// Companion `MemTableProvider` for the parse/lower guards (no GPU). Same
/// `(s Utf8, v Int64)` shape as `sv_batch`.
fn sv_provider() -> MemTableProvider {
    let schema = Schema::new(vec![
        Field::new("s", DataType::Utf8, false),
        Field::new("v", DataType::Int64, false),
    ]);
    MemTableProvider::new().with_table("t", schema)
}

/// Run `query` against a freshly-registered `t = sv_batch(strings, values)`
/// and return the sorted set of non-zero `v` values that survived the
/// filter — i.e. the tags of the rows that matched. Requires a CUDA device.
fn ilike_matching_tags(strings: &[&str], values: &[i64], query: &str) -> Vec<i64> {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    engine
        .register_table("t", sv_batch(strings, values))
        .expect("register");
    let h = engine
        .sql(query)
        .unwrap_or_else(|e| panic!("execute {query:?}: {e}"));
    let out = h.record_batch();
    let v = out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("v is Int64");
    let mut got: Vec<i64> = (0..v.len())
        .map(|i| v.value(i))
        .filter(|x| *x != 0)
        .collect();
    got.sort();
    got
}

// ---------------------------------------------------------------------------
// Parse/lower guards (no GPU) — pin that `... ILIKE '...'` parses and lowers
// for the patterns the execution tests below assert truth values for.
// ---------------------------------------------------------------------------

#[test]
fn ilike_patterns_parse_and_lower() {
    let provider = sv_provider();
    // The dotted-I core regression pattern, the sharp-S pair, the three
    // anchored fast-path shapes, and the ASCII sanity pattern must all
    // parse and lower cleanly through the public frontend.
    for sql in [
        "SELECT v FROM t WHERE s ILIKE 'a\u{0130}b'", // 'aİb'
        "SELECT v FROM t WHERE s ILIKE '\u{00DF}'",   // 'ß'
        "SELECT v FROM t WHERE s ILIKE '\u{1E9E}'",   // 'ẞ' (capital sharp S)
        "SELECT v FROM t WHERE s ILIKE '\u{0130}%'",  // prefix 'İ%'
        "SELECT v FROM t WHERE s ILIKE '%\u{0130}'",  // suffix '%İ'
        "SELECT v FROM t WHERE s ILIKE '%\u{0130}%'", // contains '%İ%'
        "SELECT v FROM t WHERE s ILIKE 'hello'",      // ASCII
    ] {
        let plan =
            parse_sql(sql, &provider).unwrap_or_else(|e| panic!("parse failed for {sql}: {e}"));
        lower_physical(&plan).unwrap_or_else(|e| panic!("lower failed for {sql}: {e}"));
    }
}

// ---------------------------------------------------------------------------
// Execution tests (require a CUDA device) — assert the actual truth values
// match Postgres/DuckDB ILIKE semantics.
// ---------------------------------------------------------------------------

/// Core regression. SQL `value ILIKE pattern` puts the wildcard in the
/// pattern, so the prompt's `'a_b' ILIKE 'aİb'` is exercised here as
/// `'aİb' ILIKE 'a_b'` — value `aİb`, pattern `a_b` — which must MATCH.
/// `İ` (U+0130) lowercases to two chars, but the pattern's `_` matches the
/// single input scalar `İ`, so per-char folding keeps the boundary aligned
/// and the row matches, as Postgres/DuckDB return `true`. A genuine
/// non-match (`'axyb'`, two middle chars where `_` allows exactly one) must
/// NOT match.
#[test]
#[ignore = "gpu:string"]
fn ilike_length_changing_fold_dotted_i() {
    // Rows (value `s` matched against pattern `a_b`): tag 1 = "a\u{0130}b"
    // — the dotted-I is one input scalar, so `_` matches it; tag 2 = "aXb"
    // (single middle char) also matches `_`; tag 3 = "axyb" (two middle
    // chars) must NOT match a single `_`.
    let tags = ilike_matching_tags(
        &["a\u{0130}b", "aXb", "axyb"],
        &[1, 2, 3],
        // `_` is the single-char wildcard; the input scalar `İ` is one char.
        "SELECT v FROM t WHERE s ILIKE 'a_b'",
    );
    assert_eq!(
        tags,
        vec![1, 2],
        "`a_b` (single `_`) must match the 1-scalar middle rows (İ, X) \
         and NOT the 2-char middle row (xy)"
    );

    // The literal dotted-I form: `'aİb' ILIKE 'aİb'` matches itself, and a
    // genuine different string does not.
    let tags2 = ilike_matching_tags(
        &["a\u{0130}b", "aZb"],
        &[10, 20],
        "SELECT v FROM t WHERE s ILIKE 'a\u{0130}b'",
    );
    assert_eq!(
        tags2,
        vec![10],
        "literal `aİb` must match itself case-insensitively and not `aZb`"
    );
}

/// Sharp-S fold: `ß` (U+00DF) and its uppercase form `ẞ` (U+1E9E) under
/// ILIKE. Both casings of a single-`ß` string match either-cased single-`ß`
/// pattern; a plain ASCII `ss` does NOT fold to `ß` (Unicode simple
/// case-folding maps `ß`↔`ẞ` but not `ß`→`ss`), matching DuckDB/Postgres
/// default (non-full-fold) ILIKE.
#[test]
#[ignore = "gpu:string"]
fn ilike_sharp_s_fold() {
    // Pattern `ß` (lowercase sharp s). Rows: tag 1 = "ß", tag 2 = "ẞ"
    // (capital sharp s) — both must match; tag 3 = "ss" must NOT match
    // (no full-fold expansion); tag 4 = "x" must NOT match.
    let tags = ilike_matching_tags(
        &["\u{00DF}", "\u{1E9E}", "ss", "x"],
        &[1, 2, 3, 4],
        "SELECT v FROM t WHERE s ILIKE '\u{00DF}'",
    );
    assert_eq!(
        tags,
        vec![1, 2],
        "ILIKE 'ß' must match both 'ß' and 'ẞ', but not 'ss' or 'x'"
    );

    // Symmetric: pattern `ẞ` (capital sharp s) matches both casings too.
    let tags_upper = ilike_matching_tags(
        &["\u{00DF}", "\u{1E9E}", "ss"],
        &[10, 20, 30],
        "SELECT v FROM t WHERE s ILIKE '\u{1E9E}'",
    );
    assert_eq!(
        tags_upper,
        vec![10, 20],
        "ILIKE 'ẞ' must match both 'ß' and 'ẞ', but not 'ss'"
    );
}

/// Prefix (`İ%`), suffix (`%İ`), and contains (`%İ%`) ILIKE patterns around
/// the dotted-I cover the length-skew on the fast-path shapes. Under ILIKE
/// these compile to the per-char generic matcher, so the anchor boundary
/// never drifts even though `İ` lowercases to two chars.
#[test]
#[ignore = "gpu:string"]
fn ilike_prefix_suffix_contains_with_boundary_char() {
    let i = "\u{0130}"; // İ

    // Prefix `İ%`: matches strings that start with `İ` (any case), not ones
    // where `İ` is interior. (Build owned Strings first so the row slice is a
    // uniform `&[&str]`.)
    let p_lead = format!("{i}xyz"); // tag 1: starts with İ → match
    let p_mid = format!("ab{i}z"); // tag 2: İ interior → no match
    let prefix = ilike_matching_tags(
        &[p_lead.as_str(), p_mid.as_str(), "qrs"],
        &[1, 2, 3],
        "SELECT v FROM t WHERE s ILIKE '\u{0130}%'",
    );
    assert_eq!(
        prefix,
        vec![1],
        "prefix `İ%` must match only the leading-İ row"
    );

    // Suffix `%İb`: matches strings that end with `İb`.
    let s_end = format!("xyz{i}b"); // tag 1: ends with İb → match
    let s_start = format!("{i}bxyz"); // tag 2: İb at the start → no match
    let s_other = format!("xyz{i}c"); // tag 3: ends with İc → no match
    let suffix = ilike_matching_tags(
        &[s_end.as_str(), s_start.as_str(), s_other.as_str()],
        &[1, 2, 3],
        "SELECT v FROM t WHERE s ILIKE '%\u{0130}b'",
    );
    assert_eq!(
        suffix,
        vec![1],
        "suffix `%İb` must match only the trailing-İb row"
    );

    // Contains `%İ%`: matches strings with an `İ` anywhere.
    let c_lead = format!("{i}xyz"); // tag 1: leading İ → match
    let c_mid = format!("ab{i}z"); // tag 2: interior İ → match
    let c_trail = format!("xyz{i}"); // tag 3: trailing İ → match
    let contains = ilike_matching_tags(
        &[c_lead.as_str(), c_mid.as_str(), c_trail.as_str(), "qrs"],
        &[1, 2, 3, 4],
        "SELECT v FROM t WHERE s ILIKE '%\u{0130}%'",
    );
    assert_eq!(
        contains,
        vec![1, 2, 3],
        "contains `%İ%` must match every row holding an İ, regardless of position"
    );
}

/// ASCII ILIKE sanity: `'HELLO' ILIKE 'hello'` is a case-insensitive match;
/// a different word does not match.
#[test]
#[ignore = "gpu:string"]
fn ilike_ascii_case_insensitive_sanity() {
    let tags = ilike_matching_tags(
        &["HELLO", "hello", "HeLLo", "world"],
        &[1, 2, 3, 4],
        "SELECT v FROM t WHERE s ILIKE 'hello'",
    );
    assert_eq!(
        tags,
        vec![1, 2, 3],
        "ILIKE 'hello' must match every casing of 'hello' but not 'world'"
    );
}
