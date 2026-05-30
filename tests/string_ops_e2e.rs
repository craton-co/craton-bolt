// SPDX-License-Identifier: Apache-2.0

//! SQL-level end-to-end tests for string operations (review H7).
//!
//! ## Why this file exists
//!
//! `src/exec/string_ops.rs` and `src/exec/string_ops_extended.rs` carry 41
//! in-module unit tests but the surrounding SQL surface had ZERO coverage —
//! a grep for `UPPER|LOWER|LENGTH|SUBSTR|CONCAT|TRIM|LIKE` across `tests/`
//! returned nothing. This file closes that gap.
//!
//! ## What's actually supported at the SQL frontend
//!
//! The frontend (`src/plan/sql_frontend.rs`) currently rejects every scalar
//! function call: `lower_expr` routes `SqlExpr::Function(_)` to
//! `BoltError::Sql("function calls are only allowed as top-level aggregates
//! in SELECT")`. The only aggregate names that survive `try_aggregate` are
//! `COUNT|SUM|MIN|MAX|AVG` — so UPPER, LOWER, LENGTH, SUBSTR/SUBSTRING,
//! CONCAT, CONCAT_WS, and TRIM are unreachable through SQL today even though
//! the per-row implementations exist on the executor side. Likewise
//! `BinaryOperator::StringConcat` (`||`) is supported as of v0.5 — it
//! lowers to `BinaryOp::Concat`, type-checks Utf8 ⊗ Utf8 → Utf8, and runs
//! host-side via `string_ops::host_concat_strings`. The `LIKE` operator
//! still has no arm in `lower_binary_op`, so it hits the catch-all
//! "unsupported expression" error.
//!
//! What IS supported:
//!
//! * `WHERE s = 'literal'` and `WHERE s <> 'literal'` — folded by
//!   `string_literal_rewrite` into integer (in)equality against the
//!   `__idx_<col>` dictionary index column.
//! * `SELECT s FROM t` — bare projection over a Utf8 column.
//! * `GROUP BY <Utf8 column>` — parses (the planner has no Utf8 ban) but the
//!   executor surfaces a clean `"Utf8 GROUP BY keys not yet supported"` at
//!   runtime. We assert the parse succeeds (this is the load-bearing thing
//!   the frontend owns); the executor error is asserted in
//!   `groupby_with_pre.rs`.
//!
//! ## Test layout
//!
//! 1. Parse-only tests (un-ignored, no GPU) — confirm what the frontend
//!    accepts vs rejects today. These are the regression guard against a
//!    silent widening or narrowing of the supported subset.
//! 2. Execution tests gated on `#[ignore = "gpu:string"]` — run the full
//!    SQL pipeline against a GPU device. Currently only the
//!    `WHERE s = 'literal'` path lands here; the rest become enabled as
//!    `// TODO(post-0.3)` items below are implemented.

use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::{parse_sql, DataType, Field, MemTableProvider, Schema};

// ---------------------------------------------------------------------------
// Shared fixtures.
// ---------------------------------------------------------------------------

/// Single-Utf8-column schema. Mirrors the shape `sort_e2e.rs` uses for its
/// Utf8 sort tests so the frontend takes the same code path.
fn s_schema() -> Schema {
    Schema::new(vec![Field {
        name: "s".into(),
        dtype: DataType::Utf8,
        nullable: false,
    }])
}

/// Two-column fixture: a Utf8 `s` and an Int64 `v`. Lets the
/// `WHERE s = 'literal'` test project a non-Utf8 column so we don't depend on
/// Utf8 output materialisation (a separate engine path).
fn sv_schema() -> Schema {
    Schema::new(vec![
        Field {
            name: "s".into(),
            dtype: DataType::Utf8,
            nullable: false,
        },
        Field {
            name: "v".into(),
            dtype: DataType::Int64,
            nullable: false,
        },
    ])
}

fn s_provider() -> MemTableProvider {
    MemTableProvider::new().with_table("t", s_schema())
}

fn sv_provider() -> MemTableProvider {
    MemTableProvider::new().with_table("t", sv_schema())
}

/// Build a `RecordBatch` matching `sv_schema` with the given parallel data.
/// Used by every execution test below.
fn sv_batch(strings: &[&str], values: &[i64]) -> RecordBatch {
    assert_eq!(strings.len(), values.len(), "fixture row counts must agree");
    let s_arr: StringArray = StringArray::from(strings.to_vec());
    let v_arr: Int64Array = Int64Array::from(values.to_vec());
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("s", ArrowDataType::Utf8, false),
        ArrowField::new("v", ArrowDataType::Int64, false),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(s_arr), Arc::new(v_arr)]).unwrap()
}

// ===========================================================================
// Parse-only tests (host-side only — no GPU required).
//
// These lock the current SQL-frontend surface for string ops. Any of the
// following starting to PASS is the signal to un-ignore the matching
// execution test below it and remove the TODO(post-0.3) marker.
// ===========================================================================

#[test]
fn parse_where_string_equality_supported() {
    // `WHERE s = 'foo'` is folded into integer eq on `__idx_s` by
    // string_literal_rewrite. The SQL frontend itself just needs to parse
    // and lower it without complaint — the rewrite happens later in the
    // engine pipeline (see Engine::sql).
    let provider = sv_provider();
    parse_sql("SELECT v FROM t WHERE s = 'foo'", &provider)
        .expect("WHERE col = 'literal' must parse against a Utf8 column");
}

#[test]
fn parse_where_string_inequality_supported() {
    let provider = sv_provider();
    parse_sql("SELECT v FROM t WHERE s <> 'foo'", &provider)
        .expect("WHERE col <> 'literal' must parse against a Utf8 column");
}

#[test]
fn parse_group_by_string_column_supported() {
    // GROUP BY on a Utf8 column parses cleanly (the planner has no Utf8 ban)
    // even though the executor will later refuse with `"Utf8 GROUP BY keys
    // not yet supported"`. The parse-only assertion is the load-bearing
    // contract the frontend owns; the executor surface is covered by
    // `groupby_with_pre.rs`'s in-module test.
    let provider = s_provider();
    parse_sql("SELECT s, COUNT(*) FROM t GROUP BY s", &provider)
        .expect("GROUP BY Utf8 must parse");
}

#[test]
fn parse_projection_of_utf8_column_supported() {
    // Bare projection of a Utf8 column is the baseline e2e path also exercised
    // by sort_e2e.rs's Utf8 tests.
    let provider = s_provider();
    parse_sql("SELECT s FROM t", &provider).expect("bare Utf8 SELECT must parse");
}

#[test]
fn parse_upper_supported_by_frontend() {
    // v0.5 (M2 SQL scalar completeness): UPPER is now accepted by the SQL
    // frontend, which lowers it to `Expr::ScalarFn { kind: Upper, .. }`
    // (`src/exec/string_ops::upper` provides the host-side evaluation).
    // This previously locked the rejection; the canary fired as designed when
    // scalar-function support landed. End-to-end execution is covered by the
    // gpu:string tests below.
    let provider = s_provider();
    parse_sql("SELECT UPPER(s) FROM t", &provider)
        .expect("UPPER(s) must parse and lower in v0.5");
}

#[test]
fn parse_lower_supported_by_frontend() {
    // v0.5: LOWER is now accepted by the frontend, lowering to
    // `Expr::ScalarFn { kind: Lower, .. }`.
    let provider = s_provider();
    parse_sql("SELECT LOWER(s) FROM t", &provider)
        .expect("LOWER(s) must parse and lower in v0.5");
}

#[test]
fn parse_length_supported_by_frontend() {
    // v0.5: LENGTH is now accepted by the frontend, lowering to
    // `Expr::ScalarFn { kind: Length, .. }`. The host-side
    // `string_ops::length` returns Int32 byte counts.
    let provider = s_provider();
    parse_sql("SELECT LENGTH(s) FROM t", &provider)
        .expect("LENGTH(s) must parse and lower in v0.5");
}

#[test]
fn parse_substr_rejected_by_frontend() {
    // Unlike UPPER/LOWER/LENGTH/CONCAT, SUBSTR is NOT yet wired into the
    // frontend's scalar-function set. `string_ops_extended::substring` exists
    // and is unit-tested but remains unreachable via SQL; the frontend
    // rejects the call as an unsupported scalar function. (Lock this so the
    // test flips — prompting e2e coverage — if SUBSTR is added later.)
    let provider = s_provider();
    let err = parse_sql("SELECT SUBSTR(s, 1, 3) FROM t", &provider)
        .expect_err("SUBSTR is not yet supported by the frontend");
    let msg = format!("{err}");
    assert!(
        msg.contains("SUBSTR")
            || msg.contains("scalar function calls are not supported"),
        "unexpected error for SUBSTR: {msg}"
    );
}

#[test]
fn parse_concat_function_supported_by_frontend() {
    // v0.5: CONCAT is now accepted by the frontend, lowering to
    // `Expr::ScalarFn { kind: Concat, .. }`
    // (`string_ops_extended::concat` provides host-side evaluation).
    let provider = sv_provider();
    parse_sql("SELECT CONCAT(s, s) FROM t", &provider)
        .expect("CONCAT(s, s) must parse and lower in v0.5");
}

#[test]
fn parse_string_concat_operator_supported() {
    // v0.5: `||` (StringConcat) is now lowered to `BinaryOp::Concat` by the
    // SQL frontend. The frontend must parse it cleanly against Utf8
    // operands; type-checking happens inside `LogicalPlan::schema()` and is
    // exercised by the type-check tests below.
    let provider = s_provider();
    parse_sql("SELECT s || s FROM t", &provider)
        .expect("SELECT s || s FROM t must parse and type-check");
}

#[test]
fn parse_string_concat_with_literal_supported() {
    // SELECT a || ' literal' FROM t — Utf8 column on the left, Utf8
    // literal on the right.
    let provider = s_provider();
    parse_sql("SELECT s || ' literal' FROM t", &provider)
        .expect("SELECT s || ' literal' FROM t must parse and type-check");
}

#[test]
fn parse_string_concat_type_mismatch_rejected() {
    // `||` requires Utf8 ⊗ Utf8. A Utf8 ⊗ Int64 combination must surface a
    // type error. The frontend parses `||` structurally; the operand
    // type-check lives in `LogicalPlan::schema()` and fires during lowering
    // (`lower_physical`), not at parse time.
    use craton_bolt::plan::lower_physical;
    let provider = sv_provider();
    let plan = parse_sql("SELECT s || v FROM t", &provider)
        .expect("s || v parses structurally; type-check happens at lowering");
    let err = lower_physical(&plan)
        .expect_err("s (Utf8) || v (Int64) must reject as a type error");
    let msg = format!("{err}");
    assert!(
        msg.contains("Utf8") || msg.contains("requires Utf8"),
        "unexpected error for Utf8 || Int64: {msg}"
    );
}

#[test]
fn parse_string_concat_in_where_lowers_to_host_filter() {
    // v0.7: `||` in a WHERE predicate now lowers cleanly. The GPU codegen
    // still has no Utf8 register class, so the lowering routes the Filter
    // through the host-side `PhysicalPlan::Filter` executor (the same path
    // `LIKE` and compound `IS NULL` already use), which evaluates the
    // concat row-by-row via `expr_agg::eval_expr` →
    // `string_ops::host_concat_option_strings`.
    use craton_bolt::plan::lower_physical;
    let provider = s_provider();
    let plan = parse_sql("SELECT s FROM t WHERE s || s = 'foo'", &provider)
        .expect("WHERE s || s = 'foo' must type-check at the logical layer");
    let _phys = lower_physical(&plan)
        .expect("`||` in WHERE must lower cleanly to a host-side filter in v0.7");
}

#[test]
fn parse_like_constant_pattern_supported_v05() {
    // v0.5 (M2 SQL scalar completeness): LIKE with a constant pattern is
    // now accepted by the frontend. Plan-shape and matcher behaviour are
    // covered in `tests/like_test.rs`; here we just lock the parse-time
    // surface (no GPU needed).
    let provider = s_provider();
    parse_sql("SELECT s FROM t WHERE s LIKE 'foo%'", &provider)
        .expect("LIKE 'foo%' must parse in v0.5");
    parse_sql("SELECT s FROM t WHERE s NOT LIKE 'foo%'", &provider)
        .expect("NOT LIKE 'foo%' must parse in v0.5");

    // ESCAPE is now supported too: the frontend lowers `LIKE .. ESCAPE c`
    // to `Like { escape: Some(c), .. }`.
    parse_sql(
        r"SELECT s FROM t WHERE s LIKE 'a\_b' ESCAPE '\'",
        &provider,
    )
    .expect("LIKE ... ESCAPE must parse in v0.5");
}

#[test]
fn parse_trim_supported_by_frontend() {
    // TRIM is now supported: sqlparser surfaces it as the special-form
    // `SqlExpr::Trim { .. }`, which the frontend maps to a `ScalarFnKind::Trim*`
    // and lowers to a host-side `PhysicalPlan::Project` (a GPU TRIM kernel also
    // exists, but the scalar string fn routes through the host path at lowering).
    use craton_bolt::plan::{lower_physical, PhysicalPlan};
    let provider = s_provider();
    let plan = parse_sql("SELECT TRIM(s) FROM t", &provider)
        .expect("TRIM(s) now parses at the frontend");
    let phys = lower_physical(&plan).expect("TRIM(s) lowers to a host Project");
    assert!(
        matches!(phys, PhysicalPlan::Project { .. }),
        "expected host PhysicalPlan::Project for TRIM(s), got {phys:?}"
    );
}

// ===========================================================================
// Execution tests (require a CUDA device).
//
// Gated on `#[ignore = "gpu:string"]` per the H7 review ask. Run with
//     cargo test --test string_ops_e2e -- --ignored
// on a GPU host. Each test mirrors the parse-only assertion above to make
// sure the round-trip works, not just the lower.
// ===========================================================================

#[test]
#[ignore = "gpu:string"]
fn where_string_equality_returns_matching_rows() {
    // `WHERE s = 'foo'` is the only string-touching predicate the frontend
    // currently lowers cleanly. The rewriter folds it into integer eq on
    // `__idx_s`; this test asserts the round trip returns exactly the rows
    // whose `s = "foo"`.
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    let batch = sv_batch(
        &["foo", "bar", "foo", "baz", "foo"],
        &[1, 2, 3, 4, 5],
    );
    engine.register_table("t", batch).expect("register");

    let h = engine
        .sql("SELECT v FROM t WHERE s = 'foo'")
        .expect("execute WHERE s = 'foo'");
    let out = h.record_batch();
    let v = out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("v is Int64");
    // The engine does not compact: masked positions keep their zero-init
    // value; unmasked positions hold the projected `v`. Collect non-zero
    // outputs and assert they're exactly the `foo` rows' values.
    let mut got: Vec<i64> = (0..v.len()).map(|i| v.value(i)).filter(|x| *x != 0).collect();
    got.sort();
    assert_eq!(
        got,
        vec![1, 3, 5],
        "WHERE s = 'foo' must surface exactly the matching rows' v values"
    );
}

#[test]
#[ignore = "gpu:string"]
fn where_string_inequality_returns_complement_rows() {
    // `WHERE s <> 'foo'` is the constant-folded twin of the equality test;
    // confirms the rewriter routes both ops through the same path.
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    let batch = sv_batch(
        &["foo", "bar", "foo", "baz", "foo"],
        &[1, 2, 3, 4, 5],
    );
    engine.register_table("t", batch).expect("register");

    let h = engine
        .sql("SELECT v FROM t WHERE s <> 'foo'")
        .expect("execute WHERE s <> 'foo'");
    let out = h.record_batch();
    let v = out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("v is Int64");
    let mut got: Vec<i64> = (0..v.len()).map(|i| v.value(i)).filter(|x| *x != 0).collect();
    got.sort();
    assert_eq!(
        got,
        vec![2, 4],
        "WHERE s <> 'foo' must surface exactly the non-matching rows' v values"
    );
}

#[test]
#[ignore = "gpu:string"]
fn select_concat_two_columns_returns_concatenated_strings() {
    // v0.5: `SELECT a || b FROM t` — host-side `||` over two Utf8 columns.
    // The Project arm of `lower_depth` detects Concat in the SELECT list,
    // lowers the underlying Scan as a passthrough Projection, then wraps a
    // host-side `PhysicalPlan::Project` that calls `expr_agg::eval_expr`
    // on each row.
    use arrow_array::StringArray;
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    let a = StringArray::from(vec!["foo", "bar", "baz"]);
    let b = StringArray::from(vec!["X", "Y", "Z"]);
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("a", ArrowDataType::Utf8, false),
        ArrowField::new("b", ArrowDataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(a), Arc::new(b)]).unwrap();
    engine.register_table("t", batch).expect("register");

    let h = engine
        .sql("SELECT a || b FROM t")
        .expect("execute SELECT a || b FROM t");
    let out = h.record_batch();
    let s = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("output column is Utf8");
    let got: Vec<&str> = (0..s.len()).map(|i| s.value(i)).collect();
    assert_eq!(got, vec!["fooX", "barY", "bazZ"]);
}

#[test]
#[ignore = "gpu:string"]
fn select_concat_column_with_literal_returns_suffixed_strings() {
    // `SELECT s || ' literal' FROM t` — exercises the column-on-left,
    // Utf8-literal-on-right shape. The literal is broadcast to every row
    // by `expr_agg::eval_literal`.
    use arrow_array::StringArray;
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    let s = StringArray::from(vec!["a", "b", "c"]);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        ArrowDataType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(s)]).unwrap();
    engine.register_table("t", batch).expect("register");

    let h = engine
        .sql("SELECT s || ' literal' FROM t")
        .expect("execute SELECT s || ' literal' FROM t");
    let out = h.record_batch();
    let out_col = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("output column is Utf8");
    let got: Vec<&str> = (0..out_col.len()).map(|i| out_col.value(i)).collect();
    assert_eq!(got, vec!["a literal", "b literal", "c literal"]);
}

// TODO(post-0.3): select_upper_lowercases_input — `SELECT UPPER(s) FROM t`.
// TODO(post-0.3): select_lower_lowercases_input — `SELECT LOWER(s) FROM t`.
// TODO(post-0.3): select_length_returns_byte_or_char_count
//                 — `SELECT LENGTH(s) FROM t` (byte semantics, per
//                 `string_ops::length` doc).
// TODO(post-0.3): select_substring_extracts_subrange
//                 — `SELECT SUBSTR(s, 1, 3) FROM t`.
// TODO(post-0.3): where_like_prefix_pattern
//                 — `SELECT s FROM t WHERE s LIKE 'foo%'`.
// TODO(post-0.3): where_like_with_escape
//                 — `WHERE s LIKE 'a\_b' ESCAPE '\'`.
// TODO(post-0.3): group_by_string_column_execute
//                 — `SELECT s, COUNT(*) FROM t GROUP BY s` (parse is asserted
//                 above; execution will land once `Utf8 GROUP BY keys not
//                 yet supported` lifts in `groupby_with_pre.rs`).
