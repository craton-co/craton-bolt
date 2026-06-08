// SPDX-License-Identifier: Apache-2.0

//! End-to-end correctness tests for SQL set operations
//! (`UNION` / `UNION ALL` / `EXCEPT` / `EXCEPT ALL` / `INTERSECT` /
//! `INTERSECT ALL`), driven through the public `Engine::sql` surface.
//!
//! The dedicated host-side executor lives in `src/exec/setops.rs` (EXCEPT /
//! INTERSECT); `UNION ALL` lowers to a `Union { inputs }` concatenation and
//! plain `UNION` to `Distinct(Union { ... })`. That module has unit tests for
//! the executor in isolation, but there was no integration file exercising the
//! operators through the full SQL pipeline — this file fills that gap with a
//! focus on the subtle parts: multiset (`ALL`) multiplicities and the
//! "two NULLs compare equal" set-membership rule.
//!
//! ## Semantics pinned here (per `src/exec/setops.rs`)
//!   * `UNION ALL`    — concatenation, duplicates kept.
//!   * `UNION`        — concatenation then DISTINCT (dedup).
//!   * `EXCEPT ALL`   — `max(0, lc - rc)` copies of each left row.
//!   * `EXCEPT`       — distinct left rows whose key is absent from the right.
//!   * `INTERSECT ALL`— `min(lc, rc)` copies of each common row.
//!   * `INTERSECT`    — distinct common rows, each at most once.
//!   * NULL-equality  — two NULLs in the same column position compare EQUAL
//!     (the engine-wide "NULLs are not distinct" rule shared with DISTINCT),
//!     so a NULL row is removable by EXCEPT and matchable by INTERSECT.
//!
//! ## Gating
//!
//! Every test registers a table, and `Engine::new()` opens a CUDA device, so
//! each test is `#[ignore = "gpu:e2e"]` per the project convention documented
//! in `tests/common/mod.rs`. The file COMPILES (and links) under the cuda-stub
//! feature; it only RUNS on a GPU host:
//!
//! ```text
//! cargo test --test setops_e2e_test -- --ignored
//! ```
//!
//! Set operations do not guarantee output ordering, so every assertion
//! compares the result ROW MULTISET in an order-insensitive way: the decoded
//! rows are sorted before comparing against the SQL-standard expected multiset.

use std::sync::Arc;

use arrow_array::{Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::Engine;

mod common;

// ---------------------------------------------------------------------------
// Fixtures + decoding helpers (each integration binary is its own crate, so
// these carry `#[allow(dead_code)]` like the sibling e2e files).
// ---------------------------------------------------------------------------

/// Register a single-column nullable `Int32` table `name(a)` and return the
/// engine handle. Nullable so the NULL-equality fixtures can inject `None`.
#[allow(dead_code)]
fn engine_with_int32(name: &str, vals: Vec<Option<i32>>) -> Engine {
    let mut engine = Engine::new().expect("CUDA ctx");
    register_int32(&mut engine, name, vals);
    engine
}

/// Register an additional single-column nullable `Int32` table on an existing
/// engine (set ops need two registered tables).
#[allow(dead_code)]
fn register_int32(engine: &mut Engine, name: &str, vals: Vec<Option<i32>>) {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "a",
        ArrowDataType::Int32,
        true,
    )]));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vals))]).expect("int32 batch");
    engine.register_table(name, batch).expect("register");
}

/// Decode the first (Int32) column of a result batch into a sorted multiset of
/// `Option<i32>` (NULLs sort first). Order-insensitive: set ops do not pin
/// output order, so the oracle and the result are both canonicalised this way.
#[allow(dead_code)]
fn sorted_int32_rows(batch: &RecordBatch) -> Vec<Option<i32>> {
    let arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 column");
    let mut rows: Vec<Option<i32>> = (0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i))
            }
        })
        .collect();
    // None < Some(_) under the derived Ord, giving a stable canonical multiset.
    rows.sort();
    rows
}

/// Canonical sorted multiset for the expected oracle (mirrors
/// [`sorted_int32_rows`] so a literal expectation sorts identically).
#[allow(dead_code)]
fn sorted(mut expected: Vec<Option<i32>>) -> Vec<Option<i32>> {
    expected.sort();
    expected
}

/// Decode a two-column `(Utf8, Int32)` result batch into a sorted multiset of
/// `(Option<String>, Option<i32>)` rows. Used by the multi-column /
/// schema-unification cases.
#[allow(dead_code)]
fn sorted_str_int_rows(batch: &RecordBatch) -> Vec<(Option<String>, Option<i32>)> {
    let s = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Utf8 column 0");
    let n = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 column 1");
    let mut rows: Vec<(Option<String>, Option<i32>)> = (0..batch.num_rows())
        .map(|i| {
            let sv = if s.is_null(i) {
                None
            } else {
                Some(s.value(i).to_string())
            };
            let nv = if n.is_null(i) { None } else { Some(n.value(i)) };
            (sv, nv)
        })
        .collect();
    rows.sort();
    rows
}

// ===========================================================================
// UNION (dedup) vs UNION ALL (keeps duplicates)
// ===========================================================================

/// `UNION ALL` is pure concatenation: every row from both branches is kept,
/// including duplicates. left {1,2,2,3} ++ right {2,3,4} → the 7-row multiset
/// {1,2,2,2,3,3,4}.
#[test]
#[ignore = "gpu:e2e"]
fn union_all_keeps_all_duplicates() {
    let mut engine = engine_with_int32("l", vec![Some(1), Some(2), Some(2), Some(3)]);
    register_int32(&mut engine, "r", vec![Some(2), Some(3), Some(4)]);

    let h = engine
        .sql("SELECT a FROM l UNION ALL SELECT a FROM r")
        .expect("UNION ALL");
    let out = h.record_batch();
    assert_eq!(
        sorted_int32_rows(out),
        sorted(vec![
            Some(1),
            Some(2),
            Some(2),
            Some(2),
            Some(3),
            Some(3),
            Some(4),
        ]),
        "UNION ALL must concatenate without dedup",
    );
}

/// Plain `UNION` deduplicates the concatenation: the SAME inputs as the
/// `UNION ALL` case collapse to the DISTINCT set {1,2,3,4}.
#[test]
#[ignore = "gpu:e2e"]
fn union_dedups_overlapping_rows() {
    let mut engine = engine_with_int32("l", vec![Some(1), Some(2), Some(2), Some(3)]);
    register_int32(&mut engine, "r", vec![Some(2), Some(3), Some(4)]);

    let h = engine
        .sql("SELECT a FROM l UNION SELECT a FROM r")
        .expect("UNION");
    let out = h.record_batch();
    assert_eq!(
        sorted_int32_rows(out),
        sorted(vec![Some(1), Some(2), Some(3), Some(4)]),
        "UNION must dedup the concatenated multiset",
    );
}

// ===========================================================================
// EXCEPT (set) vs EXCEPT ALL (multiset difference, max(0, lc - rc))
// ===========================================================================

/// `EXCEPT ALL` with duplicate counts: left has x (=2) three times, right
/// twice → `max(0, 3 - 2) = 1` copy of x survives. The distinct value y (left
/// once, absent right) survives once. Expected multiset {2, 7}.
#[test]
#[ignore = "gpu:e2e"]
fn except_all_subtracts_multiplicities() {
    let mut engine = engine_with_int32("l", vec![Some(2), Some(2), Some(2), Some(7)]);
    register_int32(&mut engine, "r", vec![Some(2), Some(2)]);

    let h = engine
        .sql("SELECT a FROM l EXCEPT ALL SELECT a FROM r")
        .expect("EXCEPT ALL");
    let out = h.record_batch();
    assert_eq!(
        sorted_int32_rows(out),
        sorted(vec![Some(2), Some(7)]),
        "EXCEPT ALL keeps max(0, lc - rc) copies of each row",
    );
}

/// Plain `EXCEPT` is the set difference: any value PRESENT in the right is
/// removed entirely regardless of multiplicity, and the survivors are
/// distinct. Same inputs as the ALL case: x (=2) is present in right so it is
/// dropped completely; only the distinct value 7 (absent from right) remains.
#[test]
#[ignore = "gpu:e2e"]
fn except_set_drops_any_present_right_key() {
    let mut engine = engine_with_int32("l", vec![Some(2), Some(2), Some(2), Some(7)]);
    register_int32(&mut engine, "r", vec![Some(2), Some(2)]);

    let h = engine
        .sql("SELECT a FROM l EXCEPT SELECT a FROM r")
        .expect("EXCEPT");
    let out = h.record_batch();
    assert_eq!(
        sorted_int32_rows(out),
        sorted(vec![Some(7)]),
        "EXCEPT drops every left row whose key appears in right, and dedups",
    );
}

// ===========================================================================
// INTERSECT (set) vs INTERSECT ALL (multiset intersection, min(lc, rc))
// ===========================================================================

/// `INTERSECT ALL` keeps `min(lc, rc)` copies of each common row. left has 5
/// (=2) three times and 8 once; right has 5 twice and 8 twice →
/// `min(3, 2) = 2` copies of 5, and `min(1, 2) = 1` copy of 8. The left-only
/// value 1 and the right-only value 9 contribute nothing. Expected {5, 5, 8}.
#[test]
#[ignore = "gpu:e2e"]
fn intersect_all_keeps_min_multiplicity() {
    let mut engine = engine_with_int32("l", vec![Some(1), Some(5), Some(5), Some(5), Some(8)]);
    register_int32(
        &mut engine,
        "r",
        vec![Some(5), Some(5), Some(8), Some(8), Some(9)],
    );

    let h = engine
        .sql("SELECT a FROM l INTERSECT ALL SELECT a FROM r")
        .expect("INTERSECT ALL");
    let out = h.record_batch();
    assert_eq!(
        sorted_int32_rows(out),
        sorted(vec![Some(5), Some(5), Some(8)]),
        "INTERSECT ALL keeps min(lc, rc) copies of each common row",
    );
}

/// Plain `INTERSECT` is the set intersection: each common value appears once
/// regardless of multiplicity. Same inputs as the ALL case → the distinct
/// common values {5, 8}.
#[test]
#[ignore = "gpu:e2e"]
fn intersect_set_keeps_common_keys_once() {
    let mut engine = engine_with_int32("l", vec![Some(1), Some(5), Some(5), Some(5), Some(8)]);
    register_int32(
        &mut engine,
        "r",
        vec![Some(5), Some(5), Some(8), Some(8), Some(9)],
    );

    let h = engine
        .sql("SELECT a FROM l INTERSECT SELECT a FROM r")
        .expect("INTERSECT");
    let out = h.record_batch();
    assert_eq!(
        sorted_int32_rows(out),
        sorted(vec![Some(5), Some(8)]),
        "INTERSECT keeps each common key exactly once",
    );
}

// ===========================================================================
// NULL-equality: two NULLs in the same column compare EQUAL for set membership
// (the engine-wide "NULLs are not distinct" rule reused from DISTINCT —
// see src/exec/setops.rs RowKey handling). This is DIFFERENT from SQL `=`,
// where NULL = NULL is UNKNOWN, but is the standard rule for the set
// operators and DISTINCT.
// ===========================================================================

/// A NULL row present on BOTH sides is REMOVED by `EXCEPT`: the left NULL key
/// matches the right NULL key. left {NULL, 1, NULL}, right {NULL} → distinct
/// left {NULL, 1} minus {NULL} = {1}.
#[test]
#[ignore = "gpu:e2e"]
fn except_removes_null_row_matched_by_null() {
    let mut engine = engine_with_int32("l", vec![None, Some(1), None]);
    register_int32(&mut engine, "r", vec![None]);

    let h = engine
        .sql("SELECT a FROM l EXCEPT SELECT a FROM r")
        .expect("EXCEPT with NULL");
    let out = h.record_batch();
    assert_eq!(
        sorted_int32_rows(out),
        sorted(vec![Some(1)]),
        "EXCEPT: a NULL on the right removes the NULL on the left (NULLs compare equal)",
    );
}

/// A NULL row present on BOTH sides is KEPT by `INTERSECT`: the NULL keys match
/// so NULL is a common row. left {NULL, 2, NULL}, right {NULL, 3} → distinct
/// common {NULL}.
#[test]
#[ignore = "gpu:e2e"]
fn intersect_keeps_null_row_present_in_both() {
    let mut engine = engine_with_int32("l", vec![None, Some(2), None]);
    register_int32(&mut engine, "r", vec![None, Some(3)]);

    let h = engine
        .sql("SELECT a FROM l INTERSECT SELECT a FROM r")
        .expect("INTERSECT with NULL");
    let out = h.record_batch();
    assert_eq!(
        sorted_int32_rows(out),
        sorted(vec![None]),
        "INTERSECT: a NULL present on both sides is a common row (NULLs compare equal)",
    );
}

/// `INTERSECT ALL` honours NULL multiplicities just like any other key:
/// left has NULL twice, right has NULL three times → `min(2, 3) = 2` NULL
/// copies survive. The non-matching non-null values drop out.
#[test]
#[ignore = "gpu:e2e"]
fn intersect_all_counts_null_multiplicity() {
    let mut engine = engine_with_int32("l", vec![None, Some(4), None]);
    register_int32(&mut engine, "r", vec![None, None, None]);

    let h = engine
        .sql("SELECT a FROM l INTERSECT ALL SELECT a FROM r")
        .expect("INTERSECT ALL with NULL");
    let out = h.record_batch();
    assert_eq!(
        sorted_int32_rows(out),
        sorted(vec![None, None]),
        "INTERSECT ALL keeps min(lc, rc) NULL copies (NULL is an ordinary key)",
    );
}

// ===========================================================================
// Column-count / type compatibility: a valid same-arity set op produces the
// unified schema, and multi-column row keys match the WHOLE row.
// ===========================================================================

/// Register a two-column `(s Utf8, n Int32)` table.
#[allow(dead_code)]
fn register_str_int(engine: &mut Engine, name: &str, rows: Vec<(&str, i32)>) {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("s", ArrowDataType::Utf8, false),
        ArrowField::new("n", ArrowDataType::Int32, false),
    ]));
    let s: Vec<&str> = rows.iter().map(|(s, _)| *s).collect();
    let n: Vec<i32> = rows.iter().map(|(_, n)| *n).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(s)),
            Arc::new(Int32Array::from(n)),
        ],
    )
    .expect("str/int batch");
    engine.register_table(name, batch).expect("register");
}

/// A same-arity two-column `EXCEPT` produces the unified `(s, n)` schema and
/// keys on the WHOLE row: ("a",2) is NOT in the right even though "a" and 2
/// each appear there separately, so it survives. left {(a,1),(b,2),(a,2)}
/// minus right {(a,1),(b,2)} → {(a,2)}.
#[test]
#[ignore = "gpu:e2e"]
fn multi_column_except_keys_whole_row_and_unifies_schema() {
    let mut engine = Engine::new().expect("CUDA ctx");
    register_str_int(&mut engine, "l", vec![("a", 1), ("b", 2), ("a", 2)]);
    register_str_int(&mut engine, "r", vec![("a", 1), ("b", 2)]);

    let h = engine
        .sql("SELECT s, n FROM l EXCEPT SELECT s, n FROM r")
        .expect("two-column EXCEPT");
    let out = h.record_batch();

    // Unified schema: two fields, (Utf8, Int32), names from the left input.
    let schema = out.schema();
    assert_eq!(schema.fields().len(), 2, "unified schema has two columns");
    assert_eq!(schema.field(0).data_type(), &ArrowDataType::Utf8);
    assert_eq!(schema.field(1).data_type(), &ArrowDataType::Int32);

    assert_eq!(
        sorted_str_int_rows(out),
        vec![(Some("a".to_string()), Some(2))],
        "multi-column EXCEPT must key on the whole (s, n) row",
    );
}

/// Two-column `UNION` (dedup) over overlapping rows unifies the schema and
/// dedups whole rows. left {(a,1),(b,2)} UNION right {(b,2),(c,3)} →
/// {(a,1),(b,2),(c,3)}.
#[test]
#[ignore = "gpu:e2e"]
fn multi_column_union_dedups_whole_rows() {
    let mut engine = Engine::new().expect("CUDA ctx");
    register_str_int(&mut engine, "l", vec![("a", 1), ("b", 2)]);
    register_str_int(&mut engine, "r", vec![("b", 2), ("c", 3)]);

    let h = engine
        .sql("SELECT s, n FROM l UNION SELECT s, n FROM r")
        .expect("two-column UNION");
    let out = h.record_batch();
    assert_eq!(out.schema().fields().len(), 2, "unified two-column schema");
    assert_eq!(
        sorted_str_int_rows(out),
        vec![
            (Some("a".to_string()), Some(1)),
            (Some("b".to_string()), Some(2)),
            (Some("c".to_string()), Some(3)),
        ],
        "two-column UNION dedups identical whole rows",
    );
}
