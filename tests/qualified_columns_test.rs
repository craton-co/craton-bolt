// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for v0.5 qualified column references (`t.col`).
//!
//! Covers:
//!   * single-table FROM with a `table.col` prefix
//!   * single-table FROM with an `AS alias` and `alias.col` references
//!   * schema-qualified FROM (`FROM schema.table`) and column refs
//!     (`SELECT schema.table.col`) — the single-catalog engine accepts and
//!     drops the leading qualifier, resolving by the trailing table name
//!   * JOIN with `table.col` prefixes in WHERE / SELECT (disambiguation)
//!   * JOIN with aliases on both sides (`FROM t1 AS a JOIN t2 AS b ON ...`)
//!   * unknown table prefix → clear "unknown table qualifier" error
//!   * 4-part identifier `cat.schema.table.col` → "deeply qualified"
//!
//! The single-catalog rule (accept ≤3-part `schema.table.col` by dropping the
//! leading schema, reject 4+ as "deeply qualified") is applied CONSISTENTLY in
//! both the SELECT/WHERE path (`lower_expr`) and the JOIN ON path
//! (`lower_join_side`) — see the two `three_part_*` tests below.
//!
//! These tests only assert that parse + lower succeed (or fail with the
//! expected message); the executor-side correctness of the resolved column
//! is exercised by the broader e2e suite.

use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Expr, Field, LogicalPlan, MemTableProvider, Schema,
};

// ---- Fixtures ---------------------------------------------------------------

/// Two-table fixture where each table has a uniquely-named column plus a
/// shared `id` column. The shared `id` lets the JOIN tests exercise the
/// "ambiguous bare reference disambiguated by a qualifier" path.
fn two_tables() -> MemTableProvider {
    let orders = Schema::new(vec![
        Field {
            name: "id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "customer_id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "total".into(),
            dtype: DataType::Int64,
            nullable: false,
        },
    ]);
    // `name` is Int32 (not Utf8) so the physical-plan lowerer doesn't have
    // to materialise a string projection — these tests only care about
    // qualifier resolution, not about which value types make it through
    // the rest of the pipeline.
    let customers = Schema::new(vec![
        Field {
            name: "id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "name".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
    MemTableProvider::new()
        .with_table("orders", orders)
        .with_table("customers", customers)
}

/// Single-table fixture for the alias-only tests.
fn one_table() -> MemTableProvider {
    let mytable = Schema::new(vec![
        Field {
            name: "x".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "y".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
    MemTableProvider::new().with_table("mytable", mytable)
}

fn try_plan(sql: &str, provider: &MemTableProvider) -> Result<(), String> {
    match parse_sql(sql, provider) {
        Err(e) => Err(format!("{e}")),
        Ok(plan) => match lower_physical(&plan) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("{e}")),
        },
    }
}

fn assert_err_contains(result: Result<(), String>, needle: &str) {
    match result {
        Ok(()) => panic!("expected error containing {needle:?}, got Ok"),
        Err(msg) => assert!(
            msg.to_ascii_lowercase().contains(&needle.to_ascii_lowercase()),
            "expected error to contain {needle:?}, got: {msg}"
        ),
    }
}

// ---- Single-table: bare table prefix ---------------------------------------

#[test]
fn single_table_with_table_prefix() {
    // `orders.id` resolves against the only FROM table.
    let provider = two_tables();
    let res = try_plan("SELECT orders.id FROM orders", &provider);
    assert!(res.is_ok(), "qualified column should resolve: {res:?}");
}

#[test]
fn single_table_with_table_prefix_in_where() {
    let provider = two_tables();
    let res = try_plan(
        "SELECT id FROM orders WHERE orders.total > 100",
        &provider,
    );
    assert!(res.is_ok(), "qualified WHERE column should resolve: {res:?}");
}

// ---- Single-table: alias ----------------------------------------------------

#[test]
fn single_table_alias_with_as_resolves() {
    // `SELECT t.x FROM mytable AS t` — the alias shadows the table name.
    let provider = one_table();
    let res = try_plan("SELECT t.x FROM mytable AS t", &provider);
    assert!(res.is_ok(), "aliased column should resolve: {res:?}");
}

#[test]
fn single_table_alias_without_as_resolves() {
    // The `AS` keyword is optional in SQL: `FROM mytable t` is equivalent.
    let provider = one_table();
    let res = try_plan("SELECT t.x FROM mytable t", &provider);
    assert!(res.is_ok(), "implicit-AS alias should resolve: {res:?}");
}

#[test]
fn aliased_table_original_name_is_shadowed() {
    // Once an alias is bound, the underlying table name is no longer in
    // scope as a qualifier — that matches standard SQL semantics.
    let provider = one_table();
    let res = try_plan("SELECT mytable.x FROM mytable AS t", &provider);
    assert_err_contains(res, "unknown table qualifier");
}

#[test]
fn aliased_table_unknown_column_errors() {
    // Qualifier is the alias, column doesn't exist in the underlying table.
    let provider = one_table();
    let res = try_plan("SELECT t.nope FROM mytable AS t", &provider);
    assert_err_contains(res, "unknown column 'nope'");
}

// ---- JOIN: table prefixes ---------------------------------------------------

#[test]
fn join_with_table_prefixes() {
    // Both sides have an `id` column; the join's ON clause uses table
    // prefixes to disambiguate. The SELECT pulls `customers.name` (only on
    // one side) and `orders.total` (only on the other) by their qualified
    // names — exercising the FROM-side resolver across both scopes.
    let provider = two_tables();
    let sql = "SELECT customers.name, orders.total \
               FROM orders INNER JOIN customers \
               ON orders.customer_id = customers.id";
    let res = try_plan(sql, &provider);
    assert!(res.is_ok(), "JOIN with qualified columns should plan: {res:?}");
}

#[test]
fn join_with_aliases_on_both_sides() {
    // `o.id` and `c.id` reference the same logical column on different
    // sides. The aliases are the only qualifier the SELECT may use; the
    // underlying table names are shadowed.
    let provider = two_tables();
    let sql = "SELECT c.name, o.total \
               FROM orders AS o INNER JOIN customers AS c \
               ON o.customer_id = c.id";
    let res = try_plan(sql, &provider);
    assert!(res.is_ok(), "JOIN with both-side aliases should plan: {res:?}");
}

#[test]
fn join_aliased_left_unaliased_right() {
    // Mixed: only the left side has an alias.
    let provider = two_tables();
    let sql = "SELECT o.total, customers.name \
               FROM orders AS o INNER JOIN customers \
               ON o.customer_id = customers.id";
    let res = try_plan(sql, &provider);
    assert!(res.is_ok(), "mixed-alias JOIN should plan: {res:?}");
}

#[test]
fn join_with_qualified_where_predicate() {
    // Qualified column in WHERE after a JOIN: the resolver must look up
    // the qualifier across both joined scopes.
    let provider = two_tables();
    let sql = "SELECT customers.name FROM orders INNER JOIN customers \
               ON orders.customer_id = customers.id \
               WHERE orders.total > 100";
    let res = try_plan(sql, &provider);
    assert!(res.is_ok(), "qualified WHERE after JOIN should plan: {res:?}");
}

// ---- Error paths ------------------------------------------------------------

#[test]
fn unknown_table_prefix_errors() {
    // `t3` isn't in FROM at all → clear "unknown table qualifier" error.
    let provider = two_tables();
    let res = try_plan(
        "SELECT t3.id FROM orders INNER JOIN customers \
         ON orders.customer_id = customers.id",
        &provider,
    );
    assert_err_contains(res, "unknown table qualifier");
}

#[test]
fn unknown_table_prefix_lists_candidate_tables() {
    // The error mentions the in-scope qualifiers so users can spot typos.
    let provider = two_tables();
    let res = try_plan(
        "SELECT bogus.name FROM orders INNER JOIN customers \
         ON orders.customer_id = customers.id",
        &provider,
    );
    let msg = match res {
        Err(m) => m,
        Ok(()) => panic!("expected error, got Ok"),
    };
    assert!(
        msg.contains("orders") && msg.contains("customers"),
        "expected candidate tables in error, got: {msg}"
    );
}

// ---- Schema-qualified names (single-catalog: qualifier accepted & dropped) --

#[test]
fn schema_qualified_table_in_from_resolves() {
    // `FROM main.orders` names the same table as `FROM orders`: the leading
    // schema/catalog qualifier is accepted and dropped.
    let provider = two_tables();
    let res = try_plan("SELECT id FROM main.orders", &provider);
    assert!(res.is_ok(), "schema-qualified FROM should resolve: {res:?}");
}

#[test]
fn schema_qualified_unknown_base_table_errors() {
    // The qualifier is dropped, but the *base* table must still exist —
    // `main.nope` resolves to base `nope`, which isn't registered.
    let provider = two_tables();
    let res = try_plan("SELECT id FROM main.nope", &provider);
    assert_err_contains(res, "unknown table 'nope'");
}

#[test]
fn three_part_column_ref_resolves_by_table_and_column() {
    // `SELECT main.orders.id` — drop the leading `main`, resolve the trailing
    // `orders.id` pair exactly as the two-part `orders.id` form would.
    let provider = two_tables();
    let res = try_plan("SELECT main.orders.id FROM main.orders", &provider);
    assert!(
        res.is_ok(),
        "schema-qualified column ref should resolve: {res:?}"
    );
}

#[test]
fn three_part_column_ref_unknown_table_errors() {
    // The middle (table) slot must be in scope. `main.bogus.id` resolves the
    // trailing `bogus.id`, and `bogus` is not a FROM table → standard
    // "unknown table qualifier" error (the leading `main` is irrelevant).
    let provider = two_tables();
    let res = try_plan("SELECT main.bogus.id FROM orders", &provider);
    assert_err_contains(res, "unknown table qualifier");
}

#[test]
fn three_part_column_ref_respects_alias_precedence() {
    // When the table is aliased, the *alias* is the only valid middle-slot
    // qualifier (the underlying table name is shadowed), and the leading
    // schema part is still dropped. `any_schema.t.x` resolves via alias `t`;
    // the underlying `mytable` does not.
    let provider = one_table();
    assert!(
        try_plan("SELECT any_schema.t.x FROM mytable AS t", &provider).is_ok(),
        "schema-qualified ref via alias should resolve"
    );
    let shadowed =
        try_plan("SELECT any_schema.mytable.x FROM mytable AS t", &provider);
    assert_err_contains(shadowed, "unknown table qualifier");
}

#[test]
fn four_part_identifier_rejected_as_deeply_qualified() {
    // Beyond 3 parts the message falls back to "deeply qualified".
    let provider = two_tables();
    let res = try_plan(
        "SELECT db.public.orders.id FROM orders",
        &provider,
    );
    assert_err_contains(res, "deeply qualified");
}

// DECISION (consistent across both CompoundIdentifier paths): a single-catalog
// engine accepts 3-part `schema.table.col` by dropping the leading schema
// segment and resolving the trailing `table.col`, and rejects 4+ parts as
// "deeply qualified". This holds for SELECT/WHERE (`lower_expr`) AND JOIN ON
// (`lower_join_side`) — see `three_part_column_ref_resolves_by_table_and_column`
// (SELECT) and `four_part_identifier_rejected_as_deeply_qualified` (4-part).
// This test pins the JOIN-ON side and proves it resolves to the CORRECT bare
// columns (not a silent mis-resolution), mirroring the SELECT behavior.
#[test]
fn three_part_in_join_on_resolves_consistently_with_select() {
    let provider = two_tables();
    // 3-part `schema.table.col` on BOTH sides of the equi-join must resolve
    // (consistent with the accepted 3-part SELECT form), dropping the leading
    // `public` schema and keying on the trailing `table.col`.
    let plan = parse_sql(
        "SELECT * FROM orders INNER JOIN customers \
         ON public.orders.customer_id = public.customers.id",
        &provider,
    )
    .expect("3-part schema-qualified JOIN ON must resolve (consistent with SELECT)");

    // `SELECT *` may wrap the Join in a Project; unwrap if so.
    let join_plan = match &plan {
        LogicalPlan::Project { input, .. } => input.as_ref(),
        other => other,
    };
    match join_plan {
        LogicalPlan::Join { on, .. } => {
            assert_eq!(on.len(), 1, "expected exactly one equi pair, got {on:?}");
            match &on[0] {
                (Expr::Column(l), Expr::Column(r)) => {
                    // Leading `public` dropped; resolved by the trailing column.
                    assert_eq!(l, "customer_id", "left ON key resolves to the bare column");
                    assert_eq!(r, "id", "right ON key resolves to the bare column");
                }
                other => panic!("expected two Column refs in the ON pair, got {other:?}"),
            }
        }
        other => panic!("expected a Join under the plan root, got {other:?}"),
    }

    // And it must lower cleanly end-to-end, exactly like the 2-part form.
    lower_physical(&plan).expect("3-part JOIN ON plan must lower");
}

/// SELECT-side mirror: the same 3-part `schema.table.col` form in a SELECT
/// list resolves to the CORRECT column in the produced LogicalPlan (the
/// leading schema is dropped; the trailing `table.col` is resolved), proving
/// the SELECT and JOIN-ON paths agree.
#[test]
fn three_part_select_resolves_to_correct_column_in_plan() {
    let provider = two_tables();
    let plan = parse_sql("SELECT main.orders.customer_id FROM main.orders", &provider)
        .expect("3-part schema-qualified SELECT column must resolve");
    match &plan {
        LogicalPlan::Project { exprs, .. } => {
            assert_eq!(exprs.len(), 1, "expected one projected expr, got {exprs:?}");
            match &exprs[0] {
                Expr::Column(name) => assert_eq!(
                    name, "customer_id",
                    "3-part SELECT ref must resolve to the bare trailing column"
                ),
                other => panic!("expected a Column ref, got {other:?}"),
            }
        }
        other => panic!("expected a Project at the plan root, got {other:?}"),
    }
}

#[test]
fn alias_with_column_list_rejected() {
    // `AS t (c1, c2)` would also rename the table's columns. The v0.5
    // frontend doesn't implement that, so it's rejected with a clear
    // message; the bare `AS t` form continues to work (see other tests).
    let provider = one_table();
    let res = try_plan("SELECT t.c1 FROM mytable AS t (c1, c2)", &provider);
    assert_err_contains(res, "table alias with column list");
}

// ---- t.col uniquely disambiguates a join-collision -------------------------

#[test]
fn qualified_picks_left_side_id_in_join() {
    // Both `orders` and `customers` have an `id` column. The join's output
    // schema renames the right side's `id` to `right.id` (per
    // `join_combined_schema`'s leftmost-wins rule). A user-typed
    // `orders.id` resolves to the unrenamed left `id`, and a user-typed
    // `customers.id` resolves to the renamed `right.id`. Either form
    // should plan cleanly.
    let provider = two_tables();
    let sql_left = "SELECT orders.id FROM orders INNER JOIN customers \
                    ON orders.customer_id = customers.id";
    assert!(
        try_plan(sql_left, &provider).is_ok(),
        "left-side qualified id should resolve"
    );
    let sql_right = "SELECT customers.id FROM orders INNER JOIN customers \
                     ON orders.customer_id = customers.id";
    assert!(
        try_plan(sql_right, &provider).is_ok(),
        "right-side qualified id should resolve"
    );
}
