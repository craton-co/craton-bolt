// SPDX-License-Identifier: Apache-2.0

//! E2E pin for the "did you mean...?" suggestions surfaced by the SQL
//! frontend when an identifier (column, table qualifier, or aggregate
//! function name) doesn't resolve.
//!
//! These tests assert the exact human-facing message shape so future
//! refactors of the error-formatting code (in `Schema::index_of`,
//! `NameResolver::resolve_compound`, and `try_aggregate`) don't silently
//! regress the v0.6 ergonomics work.

use craton_bolt::plan::{lower_physical, parse_sql, DataType, Field, MemTableProvider, Schema};

/// Parse `sql` and run the physical-plan lowerer; return the first error
/// rendered to a string. Matches the helper used in `tests/parser_tests.rs`:
/// some identifier-resolution errors only surface during lowering (the
/// frontend builds an unvalidated logical plan and the post-lower physical
/// boundary type-checks it).
fn try_plan(sql: &str, provider: &MemTableProvider) -> String {
    match parse_sql(sql, provider) {
        Err(e) => e.to_string(),
        Ok(plan) => match lower_physical(&plan) {
            Ok(_) => panic!("expected error, got Ok for: {sql}"),
            Err(e) => e.to_string(),
        },
    }
}

/// Two-table fixture with simple, distinct column names so typos in the
/// tests are unambiguous against the candidate set.
fn fixture() -> MemTableProvider {
    let users = Schema::new(vec![
        Field {
            name: "id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "name".into(),
            dtype: DataType::Utf8,
            nullable: false,
        },
        Field {
            name: "age".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
    let orders = Schema::new(vec![
        Field {
            name: "id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "user_id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "qty".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
    MemTableProvider::new()
        .with_table("users", users)
        .with_table("orders", orders)
}

/// Bare unknown column on a single-table SELECT must surface a hint that
/// matches the closest schema field name. This exercises the hook in
/// `Schema::index_of` (logical_plan.rs).
#[test]
fn bare_unknown_column_suggests_close_name() {
    let msg = try_plan("SELECT naem FROM users", &fixture());
    assert!(
        msg.contains("column 'naem' not found"),
        "original message preserved: {msg}",
    );
    assert!(
        msg.contains("did you mean 'name'?"),
        "suggestion attached: {msg}",
    );
}

/// Qualified `tbl.col` with an unknown column part must hit
/// `NameResolver::resolve_compound`'s column lookup and surface a hint.
#[test]
fn qualified_unknown_column_suggests_close_name() {
    let msg = try_plan("SELECT users.naem FROM users", &fixture());
    assert!(
        msg.contains("unknown column 'naem' in table 'users'"),
        "original message preserved: {msg}",
    );
    assert!(
        msg.contains("did you mean 'name'?"),
        "suggestion attached: {msg}",
    );
}

/// Unknown table qualifier on a compound identifier must surface a hint
/// for the closest in-scope table name. This is the other half of
/// `NameResolver::resolve_compound`.
#[test]
fn unknown_table_qualifier_suggests_close_name() {
    let msg = try_plan("SELECT usres.id FROM users", &fixture());
    assert!(
        msg.contains("unknown table qualifier 'usres'"),
        "original message preserved: {msg}",
    );
    assert!(
        msg.contains("did you mean 'users'?"),
        "suggestion attached: {msg}",
    );
}

/// Unknown aggregate function name close to a real aggregate must surface
/// a "did you mean...?" hint from `try_aggregate`.
#[test]
fn unknown_aggregate_function_suggests_close_name() {
    let msg = try_plan("SELECT COUTN(*) FROM users", &fixture());
    assert!(
        msg.contains("unknown function 'COUTN'"),
        "original message identifies the function: {msg}",
    );
    assert!(
        msg.contains("did you mean 'COUNT'?"),
        "suggestion attached: {msg}",
    );
}

/// When the typo is too far from any known name, the original error
/// message stays unchanged — no spurious suggestion is appended.
#[test]
fn no_suggestion_for_distant_typo() {
    let msg = try_plan("SELECT zzzzzz FROM users", &fixture());
    assert!(
        msg.contains("column 'zzzzzz' not found"),
        "original message preserved: {msg}",
    );
    assert!(
        !msg.contains("did you mean"),
        "no suggestion for distant typo: {msg}",
    );
}
