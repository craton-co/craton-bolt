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
    let schema = Schema::new(vec![
        Field::new("MyCol", DataType::Int32, false),
    ]);
    MemTableProvider::new().with_table("MyTable", schema)
}

/// Smoke: parse + lower must succeed. Panics with the failure message if
/// either step errors. Used by the positive tests below.
fn parse_and_lower_ok(sql: &str, provider: &MemTableProvider) {
    let plan = parse_sql(sql, provider)
        .unwrap_or_else(|e| panic!("parse failed for {sql:?}: {e}"));
    lower_physical(&plan)
        .unwrap_or_else(|e| panic!("lower failed for {sql:?}: {e}"));
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
