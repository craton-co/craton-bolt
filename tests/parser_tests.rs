// SPDX-License-Identifier: Apache-2.0

//! Negative parser/lower tests for the SQL frontend.
//!
//! The frontend in `src/plan/sql_frontend.rs` has ~40 explicit `unsupported(...)`
//! branches; this file pins their error paths so future refactors don't silently
//! drop them. Each test parses (and sometimes lowers) a malformed or unsupported
//! query and asserts that an error is returned, with a substring check on the
//! message where the message is load-bearing.

use craton_bolt::plan::{
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

// The following six tests were originally negative ("unsupported") assertions
// written before waves 7 and 8 landed DISTINCT / ORDER BY / LIMIT / HAVING /
// UNION / INNER JOIN. Now that those features are implemented they are kept
// as positive parse-and-lower smoke tests; representative *still*-unsupported
// variants (LEFT/CROSS JOIN, non-equi JOIN) are covered by the negative
// tests further down.

#[test]
fn select_with_distinct_parses() {
    let provider = fixture_table();
    let res = try_plan("SELECT DISTINCT region_id FROM sales", &provider);
    assert!(res.is_ok(), "DISTINCT should parse and lower: {res:?}");
}

#[test]
fn select_with_order_by_parses() {
    let provider = fixture_table();
    let res = try_plan("SELECT region_id FROM sales ORDER BY region_id", &provider);
    assert!(res.is_ok(), "ORDER BY should parse and lower: {res:?}");
}

#[test]
fn select_with_limit_parses() {
    let provider = fixture_table();
    let res = try_plan("SELECT * FROM sales LIMIT 10", &provider);
    assert!(res.is_ok(), "LIMIT should parse and lower: {res:?}");
}

#[test]
fn select_with_having_parses() {
    let provider = fixture_table();
    let sql =
        "SELECT region_id, COUNT(*) FROM sales GROUP BY region_id HAVING COUNT(*) > 1";
    let res = try_plan(sql, &provider);
    assert!(res.is_ok(), "HAVING with aggregate should parse and lower: {res:?}");
}

#[test]
fn union_parses() {
    let provider = fixture_table();
    let res = try_plan("SELECT * FROM sales UNION SELECT * FROM sales", &provider);
    assert!(res.is_ok(), "UNION should parse and lower: {res:?}");
}

#[test]
fn subquery_unsupported() {
    let provider = fixture_table();
    let res = try_plan("SELECT * FROM (SELECT * FROM sales)", &provider);
    // Subqueries in FROM hit the "only bare table references" arm.
    assert_err(res, "subquery in FROM");
}

#[test]
fn inner_join_parses() {
    let provider = fixture_with_sales2();
    let res = try_plan(
        "SELECT * FROM sales INNER JOIN sales2 ON sales.id = sales2.id",
        &provider,
    );
    assert!(res.is_ok(), "INNER JOIN with equi-predicate should parse and lower: {res:?}");
}

#[test]
fn left_join_unsupported() {
    let provider = fixture_with_sales2();
    let res = try_plan(
        "SELECT * FROM sales LEFT JOIN sales2 ON sales.id = sales2.id",
        &provider,
    );
    // 0.1.x supports INNER only; LEFT/RIGHT/FULL/CROSS are deferred.
    assert_err(res, "non-INNER join");
}

#[test]
fn non_equi_join_unsupported() {
    let provider = fixture_with_sales2();
    let res = try_plan(
        "SELECT * FROM sales INNER JOIN sales2 ON sales.id > sales2.id",
        &provider,
    );
    // 0.1.x supports equi predicates only.
    assert_err(res, "non-equi join predicate");
}

#[test]
fn cte_unsupported() {
    let provider = fixture_table();
    let res = try_plan("WITH t AS (SELECT * FROM sales) SELECT * FROM t", &provider);
    // Frontend says "unsupported: WITH / CTEs".
    assert_err_contains(res, "with");
}

#[test]
fn qualified_column_resolves() {
    // `table.col` references are now supported in the SELECT list (and WHERE,
    // GROUP BY, HAVING) for both single-table queries and JOINs. This used to
    // be rejected with "qualified column references (no table aliases yet)";
    // the negative coverage moved to `qualified_column_unknown_table` and
    // `qualified_column_unknown_field` below.
    let provider = fixture_table();
    let res = try_plan("SELECT sales.region_id FROM sales", &provider);
    assert!(res.is_ok(), "qualified column should resolve: {res:?}");
}

#[test]
fn qualified_column_unknown_table_errors() {
    // Qualifier must match the (only) FROM table. A stray qualifier produces a
    // clean "unknown table qualifier" message rather than a downstream
    // "unknown column" surprise.
    let provider = fixture_table();
    let res = try_plan("SELECT t3.region_id FROM sales", &provider);
    assert_err_contains(res, "unknown table qualifier");
}

#[test]
fn qualified_column_unknown_field_errors() {
    // Qualifier is valid; column name isn't part of that table's schema.
    let provider = fixture_table();
    let res = try_plan("SELECT sales.nope FROM sales", &provider);
    assert_err_contains(res, "unknown column 'nope'");
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

// ---- INNER JOIN qualified-column resolution --------------------------------
//
// These cover the `SELECT t.col FROM t1 JOIN t2 ON ...` usability path that
// was rejected before: the wildcard expansion emits `right.a` for the
// colliding right-side column, but users couldn't *type* either qualifier.
// The fixture `join_provider` deliberately uses different shapes so each
// test exercises a distinct branch of `NameResolver`.

/// Two tables with one shared join key `k` and one non-shared payload column
/// each (`a` on `t1`, `b` on `t2`). Useful for the "no collision in SELECT
/// list" path.
fn join_provider() -> MemTableProvider {
    let t1 = Schema::new(vec![
        Field {
            name: "k".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "a".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
    let t2 = Schema::new(vec![
        Field {
            name: "k".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "b".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
    MemTableProvider::new()
        .with_table("t1", t1)
        .with_table("t2", t2)
}

/// Two tables that both have an `a` column (besides the join key) — exercises
/// the rename-on-collision path: `a` resolves to t1's column, `t2.a` must
/// resolve to the renamed `right.a` output column.
fn join_provider_collision() -> MemTableProvider {
    let t1 = Schema::new(vec![
        Field {
            name: "k".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "a".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
    let t2 = Schema::new(vec![
        Field {
            name: "k".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "a".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
    MemTableProvider::new()
        .with_table("t1", t1)
        .with_table("t2", t2)
}

#[test]
fn join_select_left_qualified() {
    // `t1.a` resolves through the base-table scope: column `a` is unique to
    // t1, so the resolver maps it to plain `a` (its output name in the join
    // schema). This worked before the fix in the ON path, but not in SELECT.
    let provider = join_provider();
    let res = try_plan(
        "SELECT t1.a FROM t1 INNER JOIN t2 ON t1.k = t2.k",
        &provider,
    );
    assert!(res.is_ok(), "SELECT t1.a should work: {res:?}");
}

#[test]
fn join_select_right_qualified_no_collision() {
    // `t2.b` resolves through the joined-table scope. `b` is unique to t2
    // so it keeps its bare name in the join schema. Pre-fix the frontend
    // rejected this with "unsupported: qualified column references".
    let provider = join_provider();
    let res = try_plan(
        "SELECT t2.b FROM t1 INNER JOIN t2 ON t1.k = t2.k",
        &provider,
    );
    assert!(res.is_ok(), "SELECT t2.b should work: {res:?}");
}

#[test]
fn join_select_both_sides_colliding_name() {
    // Both tables expose an `a`. The SELECT list names both — t1.a maps to
    // `a` (left wins the bare name), t2.a maps to `right.a` (the rename
    // applied by `join_combined_schema`). The Project must produce two
    // distinct columns, not collapse them.
    let provider = join_provider_collision();
    let res = try_plan(
        "SELECT t1.a, t2.a FROM t1 INNER JOIN t2 ON t1.k = t2.k",
        &provider,
    );
    assert!(
        res.is_ok(),
        "SELECT t1.a, t2.a on colliding-name JOIN should work: {res:?}"
    );
}

#[test]
fn join_select_bare_colliding_name_picks_left() {
    // A bare `a` reference is unambiguous *post-rename*: only t1's column
    // keeps the name `a` (t2's `a` was renamed to `right.a`). So bare `a`
    // silently picks the left side — same as the pre-fix behaviour for
    // `SELECT *` followed by `SELECT a`. Documented here as a positive test
    // so a future "make bare collision an error" change has to consciously
    // update this expectation.
    let provider = join_provider_collision();
    let res = try_plan(
        "SELECT a FROM t1 INNER JOIN t2 ON t1.k = t2.k",
        &provider,
    );
    assert!(res.is_ok(), "bare `a` should resolve to left side: {res:?}");
}

#[test]
fn join_select_unknown_qualifier_errors() {
    // `t3` is not in the FROM list at all — produce a clean "unknown table
    // qualifier" error rather than a downstream "unknown column" surprise.
    let provider = join_provider();
    let res = try_plan(
        "SELECT t3.x FROM t1 INNER JOIN t2 ON t1.k = t2.k",
        &provider,
    );
    assert_err_contains(res, "unknown table qualifier");
}

#[test]
fn join_select_unknown_column_in_known_table_errors() {
    // Qualifier matches, but the column isn't in that table's schema. The
    // message must name the column and the table so users can spot the
    // typo immediately.
    let provider = join_provider();
    let res = try_plan(
        "SELECT t2.zzz FROM t1 INNER JOIN t2 ON t1.k = t2.k",
        &provider,
    );
    assert_err_contains(res, "unknown column 'zzz'");
}

#[test]
fn join_where_uses_qualified_column() {
    // Qualified refs in WHERE go through the same resolver as SELECT items.
    // Pre-fix this would error at the WHERE clause.
    let provider = join_provider();
    let res = try_plan(
        "SELECT t1.a FROM t1 INNER JOIN t2 ON t1.k = t2.k WHERE t2.b > 0",
        &provider,
    );
    assert!(res.is_ok(), "WHERE with qualified column should work: {res:?}");
}
