// SPDX-License-Identifier: Apache-2.0

//! Focused end-to-end + rejection tests for uncorrelated subquery resolution.
//!
//! The resolution logic lives in `src/exec/subquery_resolve.rs` (folds an
//! executed subquery's result into an IN-predicate / scalar literal) and the
//! correlation guard in `src/plan/subquery.rs` (rejects correlated subqueries
//! at parse/lower time). The unit tests inside those modules already cover the
//! pure host helpers (`build_in_predicate`, `scalar_value_from_batch`, …) and
//! `tests/parser_tests.rs` pins the parse-shape; this file adds:
//!
//!   * GPU-gated END-TO-END execution of the supported forms through a real
//!     `Engine` (uncorrelated `IN`, `NOT IN`, scalar subquery in a predicate),
//!     comparing against an equivalent explicit value list, and exercising the
//!     dedup/build path with a duplicate-returning subquery.
//!   * A NON-gated rejection test asserting a CORRELATED subquery is rejected
//!     with a clean `BoltError` (`is_err()`, no panic) — runs in CI since it
//!     only parses/lowers (no device needed).
//!
//! ## Gating
//!
//! Every test that registers a table needs a CUDA context (`Engine::new`
//! opens a device), so those are `#[ignore = "gpu:e2e"]`'d per the project
//! convention (see `tests/common/mod.rs`). They compile + link under the
//! `cuda-stub` CI build but only execute on a GPU host:
//!
//! ```text
//! cargo test --test subquery_e2e_test -- --ignored
//! ```
//!
//! The correlated-rejection test uses only `parse_sql` + `MemTableProvider`
//! (no `Engine`, no device), so it is NOT gated and runs in normal CI.

use std::sync::Arc;

use arrow_array::{Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::{parse_sql, DataType, Field, MemTableProvider, Schema};
use craton_bolt::Engine;

// ---------------------------------------------------------------------------
// Fixtures / decoding helpers (each integration binary is its own crate).
// ---------------------------------------------------------------------------

/// Decode column `c` of `batch` as an `Int32Array`.
#[allow(dead_code)]
fn col_int32(batch: &RecordBatch, c: usize) -> &Int32Array {
    batch
        .column(c)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("column is Int32")
}

/// Collect the non-null Int32 values of column `c` into a `Vec`.
#[allow(dead_code)]
fn collect_int32(batch: &RecordBatch, c: usize) -> Vec<i32> {
    let a = col_int32(batch, c);
    (0..a.len())
        .filter(|&i| !a.is_null(i))
        .map(|i| a.value(i))
        .collect()
}

/// Register a probe table `t(k)` and a subquery-source table `other(id)`,
/// both single-column nullable Int32. Mirrors the shape `semantics_e2e.rs`
/// uses for its NOT-IN suite so the two files stay legible side by side.
#[allow(dead_code)]
fn engine_with_probe_and_set(probe: Vec<Option<i32>>, set: Vec<Option<i32>>) -> Engine {
    let mut engine = Engine::new().expect("CUDA ctx");

    let t_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "k",
        ArrowDataType::Int32,
        true,
    )]));
    let t =
        RecordBatch::try_new(t_schema, vec![Arc::new(Int32Array::from(probe))]).expect("t batch");
    engine.register_table("t", t).expect("register t");

    let o_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "id",
        ArrowDataType::Int32,
        true,
    )]));
    let o =
        RecordBatch::try_new(o_schema, vec![Arc::new(Int32Array::from(set))]).expect("other batch");
    engine.register_table("other", o).expect("register other");
    engine
}

// ===========================================================================
// Supported form 1 — uncorrelated `WHERE k IN (SELECT id FROM other)`.
// ===========================================================================

/// The IN-subquery form must return exactly the same filtered rows as the
/// equivalent explicit value list. `subquery_resolve.rs` executes the subplan,
/// collects its distinct values, and rewrites `k IN (subquery)` into the same
/// OR-of-equalities the explicit `IN (1, 4)` list lowers to — so the two
/// queries are required to agree row-for-row.
#[test]
#[ignore = "gpu:e2e"]
fn in_subquery_matches_equivalent_value_list() {
    // probe rows {1,2,3,4,5}; subquery set {1,4} → only {1,4} survive.
    let engine = engine_with_probe_and_set(
        vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
        vec![Some(1), Some(4)],
    );

    let via_sub = engine
        .sql("SELECT k FROM t WHERE k IN (SELECT id FROM other) ORDER BY k")
        .expect("IN (subquery)");
    let via_list = engine
        .sql("SELECT k FROM t WHERE k IN (1, 4) ORDER BY k")
        .expect("IN (explicit list)");

    let got_sub = collect_int32(via_sub.record_batch(), 0);
    let got_list = collect_int32(via_list.record_batch(), 0);

    assert_eq!(got_sub, vec![1, 4], "IN (subquery) filtered rows");
    assert_eq!(
        got_sub, got_list,
        "IN (subquery) must agree with the equivalent explicit IN list",
    );
}

/// Dedup / build-path coverage: a subquery that returns DUPLICATE values
/// (`other.id` = {2, 2, 5, 5, 5}) must collapse to the distinct set {2, 5}
/// inside `in_set_from_batch` and still produce the correct membership result.
/// The answer must equal the explicit `IN (2, 5)` list — duplicates in the
/// source change nothing about which probe rows pass.
#[test]
#[ignore = "gpu:e2e"]
fn in_subquery_with_duplicate_set_dedups_and_matches() {
    let engine = engine_with_probe_and_set(
        vec![Some(1), Some(2), Some(3), Some(5), Some(8)],
        vec![Some(2), Some(2), Some(5), Some(5), Some(5)],
    );

    let via_sub = engine
        .sql("SELECT k FROM t WHERE k IN (SELECT id FROM other) ORDER BY k")
        .expect("IN (subquery with dups)");
    let via_list = engine
        .sql("SELECT k FROM t WHERE k IN (2, 5) ORDER BY k")
        .expect("IN (2, 5)");

    let got_sub = collect_int32(via_sub.record_batch(), 0);
    assert_eq!(
        got_sub,
        vec![2, 5],
        "duplicate-returning subquery still selects {{2,5}}"
    );
    assert_eq!(
        got_sub,
        collect_int32(via_list.record_batch(), 0),
        "deduped subquery set must equal the explicit distinct list",
    );
}

// ===========================================================================
// Supported form 2 — `WHERE k NOT IN (SELECT id FROM other)` semantics,
// including the NULL-in-NOT-IN three-valued-logic case.
// ===========================================================================

/// NULL-free `NOT IN` is the ordinary complement: probe {1,2,3,4,5} minus the
/// set {1,4} → {2,3,5}. Pinned against the equivalent explicit `NOT IN (1, 4)`.
#[test]
#[ignore = "gpu:e2e"]
fn not_in_subquery_without_null_is_complement() {
    let engine = engine_with_probe_and_set(
        vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
        vec![Some(1), Some(4)],
    );

    let via_sub = engine
        .sql("SELECT k FROM t WHERE k NOT IN (SELECT id FROM other) ORDER BY k")
        .expect("NOT IN (subquery)");
    let via_list = engine
        .sql("SELECT k FROM t WHERE k NOT IN (1, 4) ORDER BY k")
        .expect("NOT IN (explicit list)");

    let got_sub = collect_int32(via_sub.record_batch(), 0);
    assert_eq!(got_sub, vec![2, 3, 5], "NULL-free NOT IN complement");
    assert_eq!(
        got_sub,
        collect_int32(via_list.record_batch(), 0),
        "NOT IN (subquery) must match the equivalent explicit NOT IN list",
    );
}

/// PROMINENT case — `k NOT IN (SELECT id FROM other)` where the subquery set
/// contains a NULL.
///
/// This is the documented SQL three-valued-logic foot-gun (see
/// `docs/LIMITATIONS.md` "NOT IN / IN with NULL" and the F-6 fix described in
/// `src/exec/subquery_resolve.rs::build_in_predicate`). Strict SQL says: with
/// any NULL in the set, `x NOT IN (…)` is UNKNOWN for every row (a match →
/// FALSE, a non-match → NULL), so ZERO rows pass under `WHERE`.
///
/// The engine implements EXACTLY this strict behavior: `build_in_predicate`
/// folds the negated predicate straight to `Bool(false)` when the set has a
/// NULL. So the assertion here is **zero rows** — it MATCHES standard SQL (the
/// LIMITATIONS note flags the historical foot-gun and the divergence of naive
/// engines, but craton-bolt's current code handles the negated form correctly;
/// the residual `IN`-vs-strict-SQL divergence the doc mentions is only for the
/// non-negated form under `WHERE`, where FALSE and NULL are both filtered).
#[test]
#[ignore = "gpu:e2e"]
fn not_in_subquery_with_null_in_set_returns_zero_rows() {
    // set {1, 2, NULL}: a NULL anywhere makes every probe row's predicate
    // UNKNOWN, so NO row passes — NOT the naive complement {3, 4, 5}.
    let engine = engine_with_probe_and_set(
        vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
        vec![Some(1), Some(2), None],
    );

    let h = engine
        .sql("SELECT k FROM t WHERE k NOT IN (SELECT id FROM other)")
        .expect("NOT IN with NULL in set");
    let out = h.record_batch();
    assert_eq!(
        out.num_rows(),
        0,
        "strict SQL 3VL: a NULL in the NOT IN set excludes ALL rows (F-6); \
         a naive engine that dropped the NULL would wrongly return {{3,4,5}}",
    );
}

// ===========================================================================
// Supported form 3 — scalar subquery in a predicate.
// ===========================================================================

/// `WHERE k > (SELECT MAX(id) FROM other)` — an uncorrelated scalar subquery
/// in a comparison predicate. `subquery_resolve.rs` executes the aggregate
/// subplan, reduces the one-row result to a literal, and the predicate becomes
/// `k > <max>`. With `other.id` = {2, 7, 4}, MAX is 7, so only probe rows
/// strictly greater than 7 survive: {8, 9}.
#[test]
#[ignore = "gpu:e2e"]
fn scalar_subquery_max_in_predicate() {
    let engine = engine_with_probe_and_set(
        vec![Some(3), Some(7), Some(8), Some(9)],
        vec![Some(2), Some(7), Some(4)],
    );

    let via_sub = engine
        .sql("SELECT k FROM t WHERE k > (SELECT MAX(id) FROM other) ORDER BY k")
        .expect("scalar subquery in predicate");
    let via_const = engine
        .sql("SELECT k FROM t WHERE k > 7 ORDER BY k")
        .expect("equivalent constant predicate");

    let got_sub = collect_int32(via_sub.record_batch(), 0);
    assert_eq!(got_sub, vec![8, 9], "k > MAX(other.id)=7");
    assert_eq!(
        got_sub,
        collect_int32(via_const.record_batch(), 0),
        "scalar subquery must fold to the same constant comparison (k > 7)",
    );
}

// ===========================================================================
// Rejection — CORRELATED subquery is rejected cleanly (NON-gated / CI).
// ===========================================================================

/// `sales(region_id, qty)` plus a sibling `other(id, val)` so a subquery has a
/// second relation. Built purely on the host (`MemTableProvider`) — no device.
fn rejection_provider() -> MemTableProvider {
    let sales = Schema::new(vec![
        Field {
            name: "region_id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "qty".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
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
    MemTableProvider::new()
        .with_table("sales", sales)
        .with_table("other", other)
}

/// A CORRELATED subquery (the inner `WHERE` references the OUTER `sales.region_id`)
/// must be REJECTED with a clean `BoltError` at parse/lower time — never a
/// panic and never a silently-wrong plan. The engine has no correlated-execution
/// path (see `src/plan/subquery.rs` header and `docs/LIMITATIONS.md` "Rejected
/// SQL constructs"). This runs in CI: it only parses/lowers, no device needed.
#[test]
fn correlated_in_subquery_is_rejected_cleanly() {
    let provider = rejection_provider();

    // The subquery's WHERE references `sales.region_id`, an outer-scope column
    // not in the subquery's own (`other`) FROM scope → correlated.
    let result = std::panic::catch_unwind(|| {
        parse_sql(
            "SELECT region_id FROM sales \
             WHERE qty IN (SELECT id FROM other WHERE other.val = sales.region_id)",
            &rejection_provider(),
        )
        .is_err()
    });

    // It must NOT panic, and the wrapped parse result must be an error.
    match result {
        Ok(is_err) => assert!(
            is_err,
            "correlated IN-subquery must return Err(BoltError), got Ok(plan)",
        ),
        Err(_) => panic!("correlated subquery rejection must not panic"),
    }

    // Belt-and-braces: assert the message names the correlation contract so a
    // future refactor can't quietly turn this into some unrelated error.
    let err = parse_sql(
        "SELECT region_id FROM sales \
         WHERE qty IN (SELECT id FROM other WHERE other.val = sales.region_id)",
        &provider,
    )
    .expect_err("correlated subquery must error");
    let msg = format!("{err}").to_ascii_lowercase();
    assert!(
        msg.contains("correlated"),
        "rejection message should mention 'correlated', got: {err}",
    );
}
