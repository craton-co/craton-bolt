// SPDX-License-Identifier: Apache-2.0

//! Negative parser/lower tests for the SQL frontend.
//!
//! The frontend in `src/plan/sql_frontend.rs` has ~40 explicit `unsupported(...)`
//! branches; this file pins their error paths so future refactors don't silently
//! drop them. Each test parses (and sometimes lowers) a malformed or unsupported
//! query and asserts that an error is returned, with a substring check on the
//! message where the message is load-bearing.

use javelin::plan::{
    lower_physical, parse_sql, DataType, Field, MemTableProvider, Schema,
};

// ---- Fixture ----------------------------------------------------------------

/// Mirrors the e2e fixture but adds a `qty32` int column and a `Bool` `active`
/// column so we can exercise boolean-predicate paths and SUM widening.
///
/// Rust integration tests are separate compilation units, so this helper is
/// duplicated rather than imported from `tests/e2e_tests.rs`.
fn fixture_table() -> MemTableProvider {
    let schema = Schema::new(vec![
        Field {
            name: "region_id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "sub_region_id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "qty".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "qty32".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "price".into(),
            dtype: DataType::Float64,
            nullable: false,
        },
        Field {
            name: "active".into(),
            dtype: DataType::Bool,
            nullable: false,
        },
    ]);
    MemTableProvider::new().with_table("sales", schema)
}

/// Same `sales` shape plus a sibling `sales2` table for JOIN tests.
fn fixture_with_sales2() -> MemTableProvider {
    let mut provider = fixture_table();
    let sales2 = Schema::new(vec![
        Field {
            name: "id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
    provider.register("sales2", sales2);
    let sales_id = Schema::new(vec![
        Field {
            name: "id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
    // Overwrite `sales` with an id-bearing variant so the JOIN predicate has columns to bind to.
    provider.register("sales", sales_id);
    provider
}

/// Convenience: parse and, if that succeeds, attempt to lower. Returns the
/// first error encountered. Many "unsupported" features error at parse time,
/// but some (e.g. unknown columns) only surface during lowering.
fn try_plan(sql: &str, provider: &MemTableProvider) -> Result<(), String> {
    match parse_sql(sql, provider) {
        Err(e) => Err(format!("{e}")),
        Ok(plan) => match lower_physical(&plan) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("{e}")),
        },
    }
}

/// Assert that `result` is `Err` and its rendered message contains `needle`
/// (case-insensitive). Prints the actual message on failure to aid debugging.
fn assert_err_contains(result: Result<(), String>, needle: &str) {
    match result {
        Ok(()) => panic!("expected error containing {needle:?}, got Ok"),
        Err(msg) => assert!(
            msg.to_ascii_lowercase().contains(&needle.to_ascii_lowercase()),
            "expected error to contain {needle:?}, got: {msg}"
        ),
    }
}

/// Looser variant: assert that `result` is `Err` regardless of message content,
/// while still printing the message so test logs show *why* it failed.
fn assert_err(result: Result<(), String>, context: &str) {
    match result {
        Ok(()) => panic!("{context}: expected Err, got Ok"),
        Err(msg) => eprintln!("{context}: errored as expected with: {msg}"),
    }
}

// ---- Tests ------------------------------------------------------------------

#[test]
fn empty_sql_errors() {
    let provider = fixture_table();
    let res = try_plan("", &provider);
    assert_err(res, "empty SQL");
}

#[test]
fn select_without_from_errors() {
    let provider = fixture_table();
    let res = try_plan("SELECT 1, 2, 3", &provider);
    // Expected: the frontend rejects this because it requires exactly one FROM table.
    assert_err_contains(res, "from");
}

#[test]
fn unknown_table_errors() {
    let provider = fixture_table();
    let res = try_plan("SELECT * FROM nope", &provider);
    assert_err_contains(res, "nope");
}

#[test]
fn unknown_column_errors() {
    let provider = fixture_table();
    let res = try_plan("SELECT no_such_col FROM sales", &provider);
    assert_err(res, "unknown column");
}

#[test]
fn select_with_distinct_unsupported() {
    let provider = fixture_table();
    let res = try_plan("SELECT DISTINCT region_id FROM sales", &provider);
    assert_err_contains(res, "distinct");
}

#[test]
fn select_with_order_by_unsupported() {
    let provider = fixture_table();
    let res = try_plan("SELECT region_id FROM sales ORDER BY region_id", &provider);
    assert_err_contains(res, "order by");
}

#[test]
fn select_with_limit_unsupported() {
    let provider = fixture_table();
    let res = try_plan("SELECT * FROM sales LIMIT 10", &provider);
    assert_err_contains(res, "limit");
}

#[test]
fn select_with_having_unsupported() {
    let provider = fixture_table();
    let sql =
        "SELECT region_id, COUNT(*) FROM sales GROUP BY region_id HAVING COUNT(*) > 1";
    let res = try_plan(sql, &provider);
    assert_err_contains(res, "having");
}

#[test]
fn union_unsupported() {
    let provider = fixture_table();
    let res = try_plan("SELECT * FROM sales UNION SELECT * FROM sales", &provider);
    // Frontend message says "UNION/EXCEPT/INTERSECT".
    assert_err_contains(res, "union");
}

#[test]
fn subquery_unsupported() {
    let provider = fixture_table();
    let res = try_plan("SELECT * FROM (SELECT * FROM sales)", &provider);
    // Subqueries in FROM hit the "only bare table references" arm.
    assert_err(res, "subquery in FROM");
}

#[test]
fn join_unsupported() {
    let provider = fixture_with_sales2();
    let res = try_plan(
        "SELECT * FROM sales JOIN sales2 ON sales.id = sales2.id",
        &provider,
    );
    assert_err_contains(res, "join");
}

#[test]
fn cte_unsupported() {
    let provider = fixture_table();
    let res = try_plan("WITH t AS (SELECT * FROM sales) SELECT * FROM t", &provider);
    // Frontend says "unsupported: WITH / CTEs".
    assert_err_contains(res, "with");
}

#[test]
fn qualified_column_unsupported() {
    let provider = fixture_table();
    let res = try_plan("SELECT sales.region_id FROM sales", &provider);
    // Frontend says "qualified column references (no table aliases yet)".
    assert_err_contains(res, "qualified");
}

#[test]
fn integer_overflow_errors() {
    let provider = fixture_table();
    let res = try_plan("SELECT * FROM sales WHERE qty = 99999999999999999999", &provider);
    assert_err_contains(res, "i64");
}

#[test]
fn bare_boolean_predicate_works() {
    // Positive control: a bare Bool column should be a valid WHERE predicate.
    // If the frontend regresses and rejects this, we surface it explicitly.
    let provider = fixture_table();
    let res = try_plan("SELECT * FROM sales WHERE active", &provider);
    if let Err(msg) = &res {
        panic!(
            "bare Bool column should be a valid WHERE predicate but got error: {msg}"
        );
    }
}

#[test]
fn bool_arithmetic_rejected() {
    // The WHERE predicate must be Bool, not Int. `WHERE 1` should be rejected
    // by the type-checker even though the literal parses cleanly.
    let provider = fixture_table();
    let res = try_plan("SELECT * FROM sales WHERE 1", &provider);
    assert_err(res, "non-Bool WHERE predicate");
}

#[test]
fn aggregate_alias_rejected() {
    // Per the frontend doc: "unsupported: alias on aggregate expression".
    let provider = fixture_table();
    let res = try_plan("SELECT SUM(qty) AS total FROM sales", &provider);
    assert_err_contains(res, "alias");
}
