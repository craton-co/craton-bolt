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
//! `BinaryOperator::StringConcat` (`||`) and the `LIKE` operator have no arm
//! in `lower_binary_op`, so they hit the catch-all "unsupported expression"
//! /"unsupported binary operator" error.
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
//! 2. Execution tests gated on `#[ignore = "gpu:string_ops"]` — run the full
//!    SQL pipeline against a GPU device. Currently only the
//!    `WHERE s = 'literal'` path lands here; the rest become enabled as
//!    `// TODO(post-0.3)` items below are implemented.

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringArray};
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
fn parse_upper_rejected_by_frontend() {
    // TODO(post-0.3): UPPER not yet supported by frontend (review H7).
    // `src/exec/string_ops::upper` exists and is unit-tested, but the SQL
    // frontend currently routes every non-aggregate function call to
    // `BoltError::Sql("function calls are only allowed as top-level
    // aggregates in SELECT")`. Lock that rejection so a later widening
    // (e.g. adding UPPER to a scalar-function whitelist) flips this test
    // and prompts the e2e author to add execution coverage at the same time.
    let provider = s_provider();
    let err = parse_sql("SELECT UPPER(s) FROM t", &provider)
        .expect_err("UPPER must reject at the frontend until scalar fns land");
    let msg = format!("{err}");
    assert!(
        msg.contains("function calls are only allowed as top-level aggregates"),
        "unexpected error for UPPER: {msg}"
    );
}

#[test]
fn parse_lower_rejected_by_frontend() {
    // TODO(post-0.3): LOWER not yet supported by frontend (review H7).
    let provider = s_provider();
    let err = parse_sql("SELECT LOWER(s) FROM t", &provider)
        .expect_err("LOWER must reject at the frontend until scalar fns land");
    let msg = format!("{err}");
    assert!(
        msg.contains("function calls are only allowed as top-level aggregates"),
        "unexpected error for LOWER: {msg}"
    );
}

#[test]
fn parse_length_rejected_by_frontend() {
    // TODO(post-0.3): LENGTH not yet supported by frontend (review H7).
    // The host-side `string_ops::length` returns Int32 byte counts — but
    // it's only reachable through the executor's internal API today, not via
    // SQL. Lock the rejection.
    let provider = s_provider();
    let err = parse_sql("SELECT LENGTH(s) FROM t", &provider)
        .expect_err("LENGTH must reject at the frontend until scalar fns land");
    let msg = format!("{err}");
    assert!(
        msg.contains("function calls are only allowed as top-level aggregates"),
        "unexpected error for LENGTH: {msg}"
    );
}

#[test]
fn parse_substring_rejected_by_frontend() {
    // TODO(post-0.3): SUBSTR/SUBSTRING not yet supported by frontend
    // (review H7). `string_ops_extended::substring` is implemented and unit
    // tested but unreachable through SQL. SUBSTR lowers as a regular
    // function call → rejected by the same arm as UPPER. SUBSTRING is a
    // SQL-standard special form that sqlparser surfaces as
    // `SqlExpr::Substring { .. }` (not a function call), so it hits the
    // catch-all "unsupported expression" arm — both rejections lock the
    // current surface.
    let provider = s_provider();
    let err_substr = parse_sql("SELECT SUBSTR(s, 1, 3) FROM t", &provider)
        .expect_err("SUBSTR must reject at the frontend until scalar fns land");
    let msg = format!("{err_substr}");
    assert!(
        msg.contains("function calls are only allowed as top-level aggregates"),
        "unexpected error for SUBSTR: {msg}"
    );
    let err_substring = parse_sql("SELECT SUBSTRING(s, 1, 3) FROM t", &provider)
        .expect_err("SUBSTRING must reject at the frontend until scalar fns land");
    let msg = format!("{err_substring}");
    assert!(
        // Accept either the function-call or the special-form rejection,
        // since the SUBSTRING(... FROM ... FOR ...) syntax is its own
        // sqlparser variant and our error path may differ between versions.
        msg.contains("function calls are only allowed as top-level aggregates")
            || msg.contains("unsupported expression")
            || msg.contains("unsupported"),
        "unexpected error for SUBSTRING: {msg}"
    );
}

#[test]
fn parse_concat_function_rejected_by_frontend() {
    // TODO(post-0.3): CONCAT not yet supported by frontend (review H7).
    // `string_ops_extended::concat` is implemented for two-column inputs.
    let provider = sv_provider();
    let err = parse_sql("SELECT CONCAT(s, s) FROM t", &provider)
        .expect_err("CONCAT must reject at the frontend until scalar fns land");
    let msg = format!("{err}");
    assert!(
        msg.contains("function calls are only allowed as top-level aggregates"),
        "unexpected error for CONCAT: {msg}"
    );
}

#[test]
fn parse_string_concat_operator_rejected_by_frontend() {
    // TODO(post-0.3): `||` (StringConcat) not yet supported by frontend
    // (review H7). `lower_binary_op` has no arm for
    // `BinaryOperator::StringConcat`, so `a || b` falls into the
    // "unsupported binary operator" rejection.
    let provider = s_provider();
    let err = parse_sql("SELECT s || s FROM t", &provider)
        .expect_err("s || s must reject until StringConcat is wired up");
    let msg = format!("{err}");
    assert!(
        msg.contains("unsupported binary operator"),
        "unexpected error for ||: {msg}"
    );
}

#[test]
fn parse_like_rejected_by_frontend() {
    // TODO(post-0.3): LIKE not yet supported by frontend (review H7).
    // sqlparser surfaces LIKE as a dedicated `SqlExpr::Like { .. }` node, not
    // a `BinaryOp`, so it falls into `lower_expr`'s catch-all "unsupported
    // expression" arm. This covers both the prefix-pattern case and the
    // backslash-escape variant listed in the H7 checklist.
    let provider = s_provider();

    let err = parse_sql("SELECT s FROM t WHERE s LIKE 'foo%'", &provider)
        .expect_err("LIKE 'foo%' must reject until pattern matching lands");
    let msg = format!("{err}");
    assert!(
        msg.contains("unsupported"),
        "unexpected error for LIKE prefix: {msg}"
    );

    // TODO(post-0.3): where_like_with_escape — `WHERE s LIKE 'a\_b' ESCAPE '\'`.
    // Same rejection path; locked here so re-enabling either form flips both
    // tests together.
    let err_esc = parse_sql(
        r"SELECT s FROM t WHERE s LIKE 'a\_b' ESCAPE '\'",
        &provider,
    )
    .expect_err("LIKE ... ESCAPE must reject until pattern matching lands");
    let msg = format!("{err_esc}");
    assert!(
        msg.contains("unsupported"),
        "unexpected error for LIKE escape: {msg}"
    );
}

#[test]
fn parse_trim_rejected_by_frontend() {
    // TODO(post-0.3): TRIM not yet supported (review H7). No host
    // implementation exists yet either — listed for symmetry with the
    // review's checklist. sqlparser parses TRIM as a special-form
    // `SqlExpr::Trim { .. }`, not a function call, so the catch-all
    // "unsupported expression" arm fires.
    let provider = s_provider();
    let err = parse_sql("SELECT TRIM(s) FROM t", &provider)
        .expect_err("TRIM must reject at the frontend until trim ops land");
    let msg = format!("{err}");
    assert!(
        msg.contains("unsupported") || msg.contains("function"),
        "unexpected error for TRIM: {msg}"
    );
}

// ===========================================================================
// Execution tests (require a CUDA device).
//
// Gated on `#[ignore = "gpu:string_ops"]` per the H7 review ask. Run with
//     cargo test --test string_ops_e2e -- --ignored
// on a GPU host. Each test mirrors the parse-only assertion above to make
// sure the round-trip works, not just the lower.
// ===========================================================================

#[test]
#[ignore = "gpu:string_ops"]
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
#[ignore = "gpu:string_ops"]
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

// TODO(post-0.3): select_upper_lowercases_input — `SELECT UPPER(s) FROM t`.
// TODO(post-0.3): select_lower_lowercases_input — `SELECT LOWER(s) FROM t`.
// TODO(post-0.3): select_length_returns_byte_or_char_count
//                 — `SELECT LENGTH(s) FROM t` (byte semantics, per
//                 `string_ops::length` doc).
// TODO(post-0.3): select_substring_extracts_subrange
//                 — `SELECT SUBSTR(s, 1, 3) FROM t`.
// TODO(post-0.3): select_concat_two_columns
//                 — `SELECT CONCAT(a, b) FROM t` or `a || b`.
// TODO(post-0.3): where_like_prefix_pattern
//                 — `SELECT s FROM t WHERE s LIKE 'foo%'`.
// TODO(post-0.3): where_like_with_escape
//                 — `WHERE s LIKE 'a\_b' ESCAPE '\'`.
// TODO(post-0.3): group_by_string_column_execute
//                 — `SELECT s, COUNT(*) FROM t GROUP BY s` (parse is asserted
//                 above; execution will land once `Utf8 GROUP BY keys not
//                 yet supported` lifts in `groupby_with_pre.rs`).
