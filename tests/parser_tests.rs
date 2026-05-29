// SPDX-License-Identifier: Apache-2.0

//! Negative parser/lower tests for the SQL frontend.
//!
//! The frontend in `src/plan/sql_frontend.rs` has ~40 explicit `unsupported(...)`
//! branches; this file pins their error paths so future refactors don't silently
//! drop them. Each test parses (and sometimes lowers) a malformed or unsupported
//! query and asserts that an error is returned, with a substring check on the
//! message where the message is load-bearing.

use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Expr, Field, LogicalPlan, MemTableProvider, PhysicalPlan,
    Schema, UnaryOp,
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

// The following tests were originally negative ("unsupported") assertions
// written before waves 7 and 8 landed DISTINCT / ORDER BY / LIMIT / HAVING /
// UNION / INNER JOIN. Now that those features (and LEFT / RIGHT / FULL /
// CROSS joins as of 0.3.1) are implemented they are kept as positive
// parse-and-lower smoke tests; representative *still*-unsupported
// variants (non-equi JOIN) are covered by the negative tests further down.

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
fn left_join_parses() {
    let provider = fixture_with_sales2();
    let res = try_plan(
        "SELECT * FROM sales LEFT JOIN sales2 ON sales.id = sales2.id",
        &provider,
    );
    assert!(res.is_ok(), "LEFT JOIN should parse and lower: {res:?}");
}

#[test]
fn right_join_parses() {
    let provider = fixture_with_sales2();
    let res = try_plan(
        "SELECT * FROM sales RIGHT JOIN sales2 ON sales.id = sales2.id",
        &provider,
    );
    assert!(res.is_ok(), "RIGHT JOIN should parse and lower: {res:?}");
}

#[test]
fn full_outer_join_parses() {
    let provider = fixture_with_sales2();
    let res = try_plan(
        "SELECT * FROM sales FULL OUTER JOIN sales2 ON sales.id = sales2.id",
        &provider,
    );
    assert!(res.is_ok(), "FULL OUTER JOIN should parse and lower: {res:?}");
}

#[test]
fn cross_join_parses() {
    let provider = fixture_with_sales2();
    let res = try_plan("SELECT * FROM sales CROSS JOIN sales2", &provider);
    assert!(res.is_ok(), "CROSS JOIN should parse and lower: {res:?}");
}

#[test]
fn non_equi_join_now_supported() {
    // v0.6: non-equi JOIN predicates are no longer rejected at parse /
    // physical-plan time. The SQL frontend extracts the residual non-equi
    // expression into the `filter` slot on `LogicalPlan::Join` and the
    // executor switches to the nested-loop fallback. Earlier versions
    // emitted "non-equi join predicate" here; the new contract is that
    // both parse and lowering must succeed.
    let provider = fixture_with_sales2();
    let res = try_plan(
        "SELECT * FROM sales INNER JOIN sales2 ON sales.id > sales2.id",
        &provider,
    );
    assert!(
        res.is_ok(),
        "v0.6: non-equi JOIN must parse and lower without error; got: {res:?}"
    );
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
fn aggregate_alias_accepted() {
    // v0.5: aggregate aliasing (`SUM(x) AS total`) is now supported.
    // The SELECT-list lowerer attaches the alias to the post-Aggregate Project
    // node, so the aggregate's plan-assigned name (e.g. `sum_qty`) is renamed
    // to the user alias before downstream stages see it.
    let provider = fixture_table();
    let res = try_plan("SELECT SUM(qty) AS total FROM sales", &provider);
    assert!(
        res.is_ok(),
        "aggregate aliasing must succeed in v0.5; got: {res:?}"
    );
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

// ---- NOT unary operator (v0.5) ---------------------------------------------

/// Walk a logical plan and return the first `Filter` predicate found.
/// Used by the NOT tests below to inspect the lowered `WHERE` predicate
/// without depending on physical-plan shape.
fn find_filter_predicate(plan: &LogicalPlan) -> Option<&Expr> {
    match plan {
        LogicalPlan::Filter { predicate, .. } => Some(predicate),
        LogicalPlan::Project { input, .. }
        | LogicalPlan::Distinct { input }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Sort { input, .. } => find_filter_predicate(input),
        LogicalPlan::Aggregate { input, .. } => find_filter_predicate(input),
        _ => None,
    }
}

/// `WHERE NOT (qty > 5)` must parse, type-check (operand is Bool), and
/// lower to an `Expr::Unary { op: UnaryOp::Not, .. }` node sitting on the
/// `Filter` predicate. End-to-end coverage of the SQL-frontend wiring
/// added in v0.5.
#[test]
fn not_unary_parses_and_lowers_to_unary_not() {
    let provider = fixture_table();
    let sql = "SELECT * FROM sales WHERE NOT (qty > 5)";
    let plan = parse_sql(sql, &provider).expect("parse NOT (qty > 5)");
    // Physical lowering must succeed — `NOT` routes to the host-side
    // filter path (see `physical_plan::predicate_contains_unary`).
    lower_physical(&plan).expect("lower NOT (qty > 5) to physical plan");

    let pred = find_filter_predicate(&plan).expect("Filter on the plan spine");
    match pred {
        Expr::Unary { op: UnaryOp::Not, operand } => match operand.as_ref() {
            Expr::Binary { .. } => {} // `qty > 5` inside the NOT — good.
            other => panic!(
                "expected `qty > 5` (Binary) under NOT, got {other:?}"
            ),
        },
        other => panic!(
            "expected Expr::Unary(Not, _) at Filter predicate, got {other:?}"
        ),
    }
}

/// `WHERE NOT active` (over a bare Bool column) is the other shape we
/// expect — `NOT` over a column reference, no parenthesised inner. The
/// type-check enforces the operand is Bool, so `NOT region_id` (an Int32)
/// must fail at lowering time.
#[test]
fn not_unary_on_bare_bool_column_works() {
    let provider = fixture_table();
    let plan =
        parse_sql("SELECT * FROM sales WHERE NOT active", &provider).expect("parse NOT active");
    lower_physical(&plan).expect("lower NOT active");

    let pred = find_filter_predicate(&plan).expect("Filter on the plan spine");
    assert!(
        matches!(pred, Expr::Unary { op: UnaryOp::Not, .. }),
        "expected NOT at Filter predicate, got {pred:?}"
    );
}

/// `NOT` over a non-Bool operand must surface a `Type` error from the
/// logical-plan layer. Pin the type-check rule so a future refactor
/// can't silently drop it.
#[test]
fn not_unary_on_non_bool_operand_errors() {
    let provider = fixture_table();
    let res = try_plan("SELECT * FROM sales WHERE NOT region_id", &provider);
    assert_err_contains(res, "NOT requires a Bool operand");
}

// ---- Uncorrelated subqueries + CTEs ----------------------------------------
//
// These pin the new frontend surface added for non-recursive WITH/CTEs and
// uncorrelated scalar / IN subqueries. Physical execution is a follow-up, so
// the positive cases assert only that parsing + logical lowering (and
// type-checking via `plan.schema()`) succeed; the negative cases pin the
// rejection messages.

/// `sales` (region_id, qty, price, ...) plus a sibling `other` table with an
/// `id`/`val` shape so subqueries have a second relation to read from.
fn subquery_provider() -> MemTableProvider {
    let mut provider = fixture_table();
    let other = Schema::new(vec![
        Field {
            name: "id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "val".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
    provider.register("other", other);
    provider
}

/// Parse + lower (logical only) helper for the positive subquery/CTE cases.
/// Returns the type-checked `LogicalPlan`, panicking with context on error.
fn plan_ok(sql: &str, provider: &MemTableProvider) -> LogicalPlan {
    let plan = parse_sql(sql, provider).unwrap_or_else(|e| panic!("parse {sql:?}: {e}"));
    plan.schema()
        .unwrap_or_else(|e| panic!("type-check {sql:?}: {e}"));
    plan
}

#[test]
fn simple_with_cte_parses_and_resolves() {
    let provider = subquery_provider();
    // The CTE `r` selects from `sales`; the main query reads the CTE.
    let plan = plan_ok(
        "WITH r AS (SELECT region_id, qty FROM sales) SELECT region_id FROM r",
        &provider,
    );
    // The CTE is inlined: there must be no dangling reference to a table
    // named `r` (the provider has no such table), which `plan.schema()`
    // already proved by type-checking. Sanity-check the output schema shape.
    let schema = plan.schema().expect("type-check");
    assert_eq!(schema.fields.len(), 1);
    assert_eq!(schema.fields[0].name, "region_id");
}

#[test]
fn chained_with_ctes_reference_earlier_cte() {
    let provider = subquery_provider();
    // `b` references the earlier CTE `a` — standard left-to-right visibility.
    let plan = plan_ok(
        "WITH a AS (SELECT region_id, qty FROM sales), \
              b AS (SELECT region_id FROM a) \
         SELECT region_id FROM b",
        &provider,
    );
    let schema = plan.schema().expect("type-check");
    assert_eq!(schema.fields.len(), 1);
    assert_eq!(schema.fields[0].name, "region_id");
}

#[test]
fn cte_shadows_base_table() {
    let provider = subquery_provider();
    // A CTE named `other` shadows the registered base table `other`; the main
    // query's `FROM other` resolves to the CTE (one column), not the base
    // table (two columns).
    let plan = plan_ok(
        "WITH other AS (SELECT region_id FROM sales) SELECT region_id FROM other",
        &provider,
    );
    let schema = plan.schema().expect("type-check");
    assert_eq!(schema.fields.len(), 1, "should see the CTE's single column");
    assert_eq!(schema.fields[0].name, "region_id");
}

#[test]
fn scalar_subquery_in_where_parses() {
    let provider = subquery_provider();
    // `WHERE qty > (SELECT MAX(val) FROM other)` — uncorrelated scalar
    // subquery on the RHS of a comparison.
    let plan = plan_ok(
        "SELECT region_id FROM sales WHERE qty > (SELECT MAX(val) FROM other)",
        &provider,
    );
    let pred = find_filter_predicate(&plan).expect("Filter on the plan spine");
    // The predicate is `qty > <scalar subquery>`.
    match pred {
        Expr::Binary { right, .. } => assert!(
            matches!(right.as_ref(), Expr::ScalarSubquery(_)),
            "expected ScalarSubquery on RHS of comparison, got {right:?}"
        ),
        other => panic!("expected Binary comparison at Filter, got {other:?}"),
    }
}

#[test]
fn in_subquery_in_where_parses() {
    let provider = subquery_provider();
    let plan = plan_ok(
        "SELECT region_id FROM sales WHERE region_id IN (SELECT id FROM other)",
        &provider,
    );
    let pred = find_filter_predicate(&plan).expect("Filter on the plan spine");
    match pred {
        Expr::InSubquery { negated, .. } => assert!(!negated, "IN must not be negated"),
        other => panic!("expected InSubquery at Filter predicate, got {other:?}"),
    }
}

#[test]
fn not_in_subquery_in_where_parses() {
    let provider = subquery_provider();
    let plan = plan_ok(
        "SELECT region_id FROM sales WHERE region_id NOT IN (SELECT id FROM other)",
        &provider,
    );
    let pred = find_filter_predicate(&plan).expect("Filter on the plan spine");
    match pred {
        Expr::InSubquery { negated, .. } => assert!(*negated, "NOT IN must be negated"),
        other => panic!("expected InSubquery at Filter predicate, got {other:?}"),
    }
}

#[test]
fn correlated_scalar_subquery_rejected() {
    let provider = subquery_provider();
    // The subquery references `sales.region_id` (an outer table) — correlated.
    let res = try_plan(
        "SELECT region_id FROM sales \
         WHERE qty > (SELECT MAX(val) FROM other WHERE sales.region_id = other.id)",
        &provider,
    );
    assert_err_contains(res, "correlated subquery");
}

#[test]
fn correlated_in_subquery_bare_column_rejected() {
    let provider = subquery_provider();
    // The subquery's WHERE references the bare outer column `qty` (a `sales`
    // column not present in `other`) — correlation via a bare name.
    let res = try_plan(
        "SELECT region_id FROM sales \
         WHERE region_id IN (SELECT id FROM other WHERE qty > 0)",
        &provider,
    );
    assert_err_contains(res, "correlated subquery");
}

#[test]
fn with_recursive_rejected() {
    let provider = subquery_provider();
    let res = try_plan(
        "WITH RECURSIVE r AS (SELECT region_id FROM sales) SELECT region_id FROM r",
        &provider,
    );
    assert_err_contains(res, "recursive");
}

#[test]
fn scalar_subquery_multi_column_rejected() {
    let provider = subquery_provider();
    // A scalar subquery must return exactly one column. The arity error is a
    // *logical* type-check failure, so assert it via `plan.schema()` — the
    // physical layer would otherwise short-circuit with its own
    // "subqueries not yet lowered" rejection before the arity check runs.
    let plan = parse_sql(
        "SELECT region_id FROM sales WHERE qty > (SELECT id, val FROM other)",
        &provider,
    )
    .expect("parse should succeed (arity is a type-check concern)");
    let err = plan
        .schema()
        .expect_err("multi-column scalar subquery must fail type-check");
    assert!(
        format!("{err}").to_ascii_lowercase().contains("exactly one column"),
        "expected arity error, got: {err}"
    );
}

#[test]
fn exists_subquery_rejected() {
    let provider = subquery_provider();
    let res = try_plan(
        "SELECT region_id FROM sales WHERE EXISTS (SELECT id FROM other)",
        &provider,
    );
    assert_err_contains(res, "EXISTS");
}

#[test]
fn duplicate_cte_name_rejected() {
    let provider = subquery_provider();
    let res = try_plan(
        "WITH a AS (SELECT region_id FROM sales), a AS (SELECT id FROM other) \
         SELECT region_id FROM a",
        &provider,
    );
    assert_err_contains(res, "duplicate CTE");
}

#[test]
fn subquery_physical_lowering_rejected_cleanly() {
    let provider = subquery_provider();
    // Physical execution is deferred: the logical plan is well-typed, but
    // `lower_physical` must reject it with a clear message rather than
    // mis-lowering.
    let plan = parse_sql(
        "SELECT region_id FROM sales WHERE region_id IN (SELECT id FROM other)",
        &provider,
    )
    .expect("parse + lower logical");
    let res = lower_physical(&plan);
    assert!(
        res.is_err(),
        "physical lowering of a subquery plan should be rejected, got Ok"
    );
}

// ---- COUNT(DISTINCT col) ----------------------------------------------------

#[test]
fn count_distinct_lowers_to_distinct_over_project() {
    let provider = fixture_table();
    // Sole SELECT item, no GROUP BY: lowers to
    //   Project(count) <- Aggregate(COUNT(*)) <- Distinct <- Project([col]) <- Filter(IS NOT NULL)
    let plan = plan_ok("SELECT COUNT(DISTINCT region_id) FROM sales", &provider);
    // Top is the SELECT-order Project producing a single `count` column.
    let LogicalPlan::Project { input, exprs } = &plan else {
        panic!("expected top-level Project, got {plan:?}");
    };
    assert_eq!(exprs.len(), 1);
    // Below it: Aggregate(COUNT) over a Distinct.
    let LogicalPlan::Aggregate { input, aggregates, group_by } = input.as_ref() else {
        panic!("expected Aggregate under Project, got {input:?}");
    };
    assert!(group_by.is_empty(), "COUNT(DISTINCT) must have no group keys");
    assert_eq!(aggregates.len(), 1);
    let LogicalPlan::Distinct { input } = input.as_ref() else {
        panic!("expected Distinct under Aggregate, got {input:?}");
    };
    // Distinct's child narrows to the single counted column.
    let LogicalPlan::Project { input, exprs } = input.as_ref() else {
        panic!("expected Project under Distinct, got {input:?}");
    };
    assert_eq!(exprs.len(), 1, "distinct project must be the single column");
    // And below that a NULL-excluding Filter (SQL DISTINCT excludes NULLs).
    assert!(
        matches!(input.as_ref(), LogicalPlan::Filter { .. }),
        "expected NULL-excluding Filter under the distinct Project, got {input:?}"
    );
    // Output column is named `count`.
    let schema = plan.schema().expect("type-check");
    assert_eq!(schema.fields.len(), 1);
    assert_eq!(schema.fields[0].name, "count");
}

#[test]
fn count_distinct_with_alias_renames_output() {
    let provider = fixture_table();
    let plan = plan_ok(
        "SELECT COUNT(DISTINCT region_id) AS n FROM sales",
        &provider,
    );
    let schema = plan.schema().expect("type-check");
    assert_eq!(schema.fields.len(), 1);
    assert_eq!(schema.fields[0].name, "n");
}

#[test]
fn count_distinct_star_rejected() {
    let provider = fixture_table();
    let res = try_plan("SELECT COUNT(DISTINCT *) FROM sales", &provider);
    assert_err_contains(res, "COUNT(DISTINCT *)");
}

#[test]
fn distinct_in_non_count_aggregate_rejected() {
    let provider = fixture_table();
    let res = try_plan("SELECT SUM(DISTINCT qty) FROM sales", &provider);
    assert_err_contains(res, "DISTINCT inside SUM");
}

#[test]
fn count_distinct_not_sole_item_rejected() {
    let provider = fixture_table();
    let res = try_plan(
        "SELECT COUNT(DISTINCT region_id), region_id FROM sales",
        &provider,
    );
    assert_err_contains(res, "sole SELECT item");
}

#[test]
fn count_distinct_with_group_by_rejected() {
    let provider = fixture_table();
    let res = try_plan(
        "SELECT COUNT(DISTINCT region_id) FROM sales GROUP BY qty",
        &provider,
    );
    // Either the "not sole item"/"GROUP BY" guard fires; both are clear.
    assert_err(res, "COUNT(DISTINCT) with GROUP BY");
}

// ---- EXCEPT / INTERSECT (+ ALL) ---------------------------------------------

use craton_bolt::plan::SetOpKind;

#[test]
fn except_lowers_to_setop() {
    let provider = fixture_table();
    let plan = plan_ok(
        "SELECT region_id FROM sales EXCEPT SELECT region_id FROM sales",
        &provider,
    );
    let LogicalPlan::SetOp { op, all, .. } = &plan else {
        panic!("expected SetOp at root, got {plan:?}");
    };
    assert_eq!(*op, SetOpKind::Except);
    assert!(!*all, "plain EXCEPT is the set (non-ALL) variant");
    // Result schema is the left input's.
    let schema = plan.schema().expect("type-check");
    assert_eq!(schema.fields.len(), 1);
    assert_eq!(schema.fields[0].name, "region_id");
}

#[test]
fn except_all_sets_multiset_flag() {
    let provider = fixture_table();
    let plan = plan_ok(
        "SELECT region_id FROM sales EXCEPT ALL SELECT region_id FROM sales",
        &provider,
    );
    let LogicalPlan::SetOp { op, all, .. } = &plan else {
        panic!("expected SetOp, got {plan:?}");
    };
    assert_eq!(*op, SetOpKind::Except);
    assert!(*all, "EXCEPT ALL is the multiset variant");
}

#[test]
fn intersect_lowers_to_setop() {
    let provider = fixture_table();
    let plan = plan_ok(
        "SELECT region_id FROM sales INTERSECT SELECT region_id FROM sales",
        &provider,
    );
    let LogicalPlan::SetOp { op, all, .. } = &plan else {
        panic!("expected SetOp, got {plan:?}");
    };
    assert_eq!(*op, SetOpKind::Intersect);
    assert!(!*all);
}

#[test]
fn intersect_all_sets_multiset_flag() {
    let provider = fixture_table();
    let plan = plan_ok(
        "SELECT region_id FROM sales INTERSECT ALL SELECT region_id FROM sales",
        &provider,
    );
    let LogicalPlan::SetOp { op, all, .. } = &plan else {
        panic!("expected SetOp, got {plan:?}");
    };
    assert_eq!(*op, SetOpKind::Intersect);
    assert!(*all);
}

#[test]
fn except_left_associative_chain() {
    // `a EXCEPT b EXCEPT c` is left-associative: SetOp(SetOp(a, b), c).
    let provider = fixture_table();
    let plan = plan_ok(
        "SELECT region_id FROM sales \
         EXCEPT SELECT region_id FROM sales \
         EXCEPT SELECT region_id FROM sales",
        &provider,
    );
    let LogicalPlan::SetOp { left, op, .. } = &plan else {
        panic!("expected SetOp at root, got {plan:?}");
    };
    assert_eq!(*op, SetOpKind::Except);
    assert!(
        matches!(left.as_ref(), LogicalPlan::SetOp { .. }),
        "left of an EXCEPT chain should itself be a SetOp, got {left:?}"
    );
}

#[test]
fn except_lowers_to_physical_setop() {
    let provider = fixture_table();
    let plan = parse_sql(
        "SELECT region_id FROM sales EXCEPT SELECT region_id FROM sales",
        &provider,
    )
    .expect("parse + lower logical");
    let phys = lower_physical(&plan).expect("physical lowering of EXCEPT");
    assert!(
        matches!(phys, PhysicalPlan::SetOp { .. }),
        "expected PhysicalPlan::SetOp, got {phys:?}"
    );
}

#[test]
fn except_incompatible_schemas_rejected() {
    let provider = fixture_table();
    // Left has one column, right has two — incompatible schemas.
    let res = try_plan(
        "SELECT region_id FROM sales EXCEPT SELECT region_id, qty FROM sales",
        &provider,
    );
    assert_err_contains(res, "EXCEPT");
}

#[test]
fn except_by_name_rejected() {
    let provider = fixture_table();
    let res = try_plan(
        "SELECT region_id FROM sales EXCEPT BY NAME SELECT region_id FROM sales",
        &provider,
    );
    // The BY NAME variant is rejected for every set operator.
    assert_err(res, "EXCEPT BY NAME");
}

// ---- JOIN ... USING / NATURAL JOIN ------------------------------------------

/// `emp(id, dept_id, name)`, `dept(dept_id, dname)` (share `dept_id`), and
/// `widget(wid, color)` (disjoint from `emp`) for the USING / NATURAL tests.
fn join_fixture() -> MemTableProvider {
    let i32f = |n: &str| Field { name: n.into(), dtype: DataType::Int32, nullable: false };
    let emp = Schema::new(vec![i32f("id"), i32f("dept_id"), i32f("name")]);
    let dept = Schema::new(vec![i32f("dept_id"), i32f("dname")]);
    let widget = Schema::new(vec![i32f("wid"), i32f("color")]);
    MemTableProvider::new()
        .with_table("emp", emp)
        .with_table("dept", dept)
        .with_table("widget", widget)
}

/// Pull the equi-pairs out of a top-level `Join` node.
fn join_on_pairs(plan: &LogicalPlan) -> &[(Expr, Expr)] {
    match plan {
        LogicalPlan::Join { on, .. } => on,
        other => panic!("expected Join at root, got {other:?}"),
    }
}

#[test]
fn join_using_single_column_desugars_to_equi_pair() {
    let provider = join_fixture();
    let plan = plan_ok(
        "SELECT id FROM emp JOIN dept USING (dept_id)",
        &provider,
    );
    let on = join_on_pairs(&plan);
    assert_eq!(on.len(), 1, "one USING column -> one equi pair");
    // Both sides reference the (un-renamed) `dept_id` column.
    assert_eq!(on[0].0, Expr::Column("dept_id".into()));
    assert_eq!(on[0].1, Expr::Column("dept_id".into()));
    plan.schema().expect("type-check");
}

#[test]
fn join_using_lowers_physically() {
    let provider = join_fixture();
    let res = try_plan("SELECT id FROM emp JOIN dept USING (dept_id)", &provider);
    assert!(res.is_ok(), "JOIN ... USING should parse and lower: {res:?}");
}

#[test]
fn natural_join_desugars_to_common_columns() {
    let provider = join_fixture();
    // emp and dept share exactly `dept_id`.
    let plan = plan_ok("SELECT id FROM emp NATURAL JOIN dept", &provider);
    let on = join_on_pairs(&plan);
    assert_eq!(on.len(), 1, "one common column -> one equi pair");
    assert_eq!(on[0].0, Expr::Column("dept_id".into()));
    assert_eq!(on[0].1, Expr::Column("dept_id".into()));
}

#[test]
fn natural_left_join_uses_common_column() {
    let provider = join_fixture();
    // NATURAL applies to outer joins too: `NATURAL LEFT JOIN` desugars the
    // common column into the equi-pair while keeping the LEFT join type.
    let plan = plan_ok("SELECT id FROM emp NATURAL LEFT JOIN dept", &provider);
    let on = join_on_pairs(&plan);
    assert_eq!(on.len(), 1, "one common column -> one equi pair");
    assert_eq!(on[0].0, Expr::Column("dept_id".into()));
}

#[test]
fn join_using_unknown_column_rejected() {
    let provider = join_fixture();
    let res = try_plan("SELECT id FROM emp JOIN dept USING (nope)", &provider);
    assert_err(res, "USING column 'nope'");
}

#[test]
fn natural_join_no_common_column_rejected() {
    let provider = join_fixture();
    // emp(id, dept_id, name) vs widget(wid, color) share nothing.
    let res = try_plan("SELECT id FROM emp NATURAL JOIN widget", &provider);
    assert_err_contains(res, "no common column");
}

#[test]
fn join_using_ambiguous_left_column_rejected() {
    let provider = join_fixture();
    // After `emp JOIN dept USING(dept_id)`, the accumulated left side has
    // `dept_id` in BOTH emp and dept; a second join USING(dept_id) is then
    // ambiguous on the left.
    let res = try_plan(
        "SELECT id FROM emp \
         JOIN dept USING (dept_id) \
         JOIN dept AS d2 USING (dept_id)",
        &provider,
    );
    assert_err_contains(res, "ambiguous");
}
