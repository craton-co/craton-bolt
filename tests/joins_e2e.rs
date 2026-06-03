// SPDX-License-Identifier: Apache-2.0

//! End-to-end JOIN tests for INNER / LEFT / RIGHT / FULL / CROSS.
//!
//! Plan-shape sanity assertions run offline (no CUDA). Full-execution
//! tests are `#[ignore]`-gated because they need a CUDA device and table
//! registration through `Engine`. Run them with
//! `cargo test --test joins_e2e -- --ignored`.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Field, LogicalPlan, MemTableProvider, PhysicalPlan, Schema,
};

// ---- Fixture ----------------------------------------------------------------

/// Two-table fixture (all Int32 to avoid the dict-registry virtual
/// `__idx_<col>` columns that complicate SELECT * over JOIN today):
///   `t1` (left):  id Int32 (1, 2, 3, 4),  v Int32 (10, 20, 30, 40)
///   `t2` (right): id Int32 (2, 3, 5),     w Int32 (200, 300, 500)
///
/// Joining `t1.id = t2.id` gives:
///   INNER:    2 matches  -> (2, 200), (3, 300)
///   LEFT:     4 rows     -> (1,NULL), (2,200), (3,300), (4,NULL)
///   RIGHT:    3 rows     -> (NULL,500), (2,200), (3,300)
///   FULL:     5 rows     -> LEFT result ∪ RIGHT-only rows
///   CROSS:    12 rows    -> 4 × 3
fn provider_and_batches() -> (MemTableProvider, RecordBatch, RecordBatch) {
    let t1_schema = Schema::new(vec![
        Field {
            name: "id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "v".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
    let t2_schema = Schema::new(vec![
        Field {
            name: "id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "w".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
    let provider = MemTableProvider::new()
        .with_table("t1", t1_schema)
        .with_table("t2", t2_schema);

    let t1_arrow = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Int32, false),
    ]));
    let t1_batch = RecordBatch::try_new(
        t1_arrow,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])) as ArrayRef,
            Arc::new(Int32Array::from(vec![10, 20, 30, 40])) as ArrayRef,
        ],
    )
    .unwrap();

    let t2_arrow = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int32, false),
        ArrowField::new("w", ArrowDataType::Int32, false),
    ]));
    let t2_batch = RecordBatch::try_new(
        t2_arrow,
        vec![
            Arc::new(Int32Array::from(vec![2, 3, 5])) as ArrayRef,
            Arc::new(Int32Array::from(vec![200, 300, 500])) as ArrayRef,
        ],
    )
    .unwrap();

    (provider, t1_batch, t2_batch)
}

// ---- Offline: plan-shape sanity -------------------------------------------

#[test]
fn left_join_lowers_to_physical_join() {
    let (provider, _, _) = provider_and_batches();
    let plan =
        parse_sql("SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id", &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    fn find_phys_join(p: &PhysicalPlan) -> Option<&PhysicalPlan> {
        match p {
            PhysicalPlan::Join { .. } => Some(p),
            PhysicalPlan::Project { input, .. }
            | PhysicalPlan::Distinct { input }
            | PhysicalPlan::Limit { input, .. }
            | PhysicalPlan::Sort { input, .. } => find_phys_join(input),
            _ => None,
        }
    }
    assert!(
        find_phys_join(&phys).is_some(),
        "LEFT JOIN must lower to a PhysicalPlan::Join, got {phys:?}"
    );
}

#[test]
fn cross_join_lowers_with_empty_on() {
    let (provider, _, _) = provider_and_batches();
    let plan = parse_sql("SELECT * FROM t1 CROSS JOIN t2", &provider).expect("parse");
    // Find the Join under the wildcard Project and assert its `on` is empty.
    fn find_join(p: &LogicalPlan) -> &LogicalPlan {
        match p {
            LogicalPlan::Join { .. } => p,
            LogicalPlan::Project { input, .. }
            | LogicalPlan::Filter { input, .. }
            | LogicalPlan::Distinct { input }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Sort { input, .. } => find_join(input),
            other => panic!("expected to find a Join under {other:?}"),
        }
    }
    match find_join(&plan) {
        LogicalPlan::Join { on, .. } => {
            assert!(on.is_empty(), "CROSS JOIN has no ON predicate");
        }
        other => panic!("expected Join, got {other:?}"),
    }
    // And lowers without error.
    let _ = lower_physical(&plan).expect("lower");
}

// ---- Online (require CUDA device) ------------------------------------------
//
// These tests register two tables, run a SQL JOIN through the engine, and
// validate the result row-by-row. Run with:
//   cargo test --test joins_e2e -- --ignored

#[test]
#[ignore = "gpu:join"]
fn e2e_inner_join_basic() {
    use craton_bolt::Engine;

    let (_, t1, t2) = provider_and_batches();
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    let h = engine
        .sql("SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.id")
        .expect("execute");
    let out = h.record_batch();
    // INNER matches: id ∈ {2, 3} -> 2 rows.
    assert_eq!(out.num_rows(), 2, "INNER expects 2 rows");
}

#[test]
#[ignore = "gpu:join"]
fn e2e_left_join_unmatched_rows_get_null_right() {
    use craton_bolt::Engine;

    let (_, t1, t2) = provider_and_batches();
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    let h = engine
        .sql("SELECT * FROM t1 LEFT JOIN t2 ON t1.id = t2.id")
        .expect("execute");
    let out = h.record_batch();
    // LEFT preserves every t1 row: 4 rows.
    assert_eq!(out.num_rows(), 4, "LEFT JOIN: every left row emits");
    // Two of those rows are unmatched -> right columns NULL. Look up the
    // right-side `label` column by name (the schema may have extra
    // dictionary-index columns appended for Utf8 inputs, so the column
    // ordinal isn't load-bearing).
    let w_idx = out
        .schema()
        .index_of("w")
        .expect("'w' column in output schema");
    let right_w = out.column(w_idx);
    let nulls: usize = (0..out.num_rows()).filter(|&i| right_w.is_null(i)).count();
    assert_eq!(nulls, 2, "LEFT JOIN: two unmatched left rows -> NULL w");
}

#[test]
#[ignore = "gpu:join"]
fn e2e_right_join_unmatched_rows_get_null_left() {
    use craton_bolt::Engine;

    let (_, t1, t2) = provider_and_batches();
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    let h = engine
        .sql("SELECT * FROM t1 RIGHT JOIN t2 ON t1.id = t2.id")
        .expect("execute");
    let out = h.record_batch();
    // RIGHT preserves every t2 row: 3 rows.
    assert_eq!(out.num_rows(), 3, "RIGHT JOIN: every right row emits");
    // One of those rows is unmatched (id=5) -> left columns NULL.
    // Look up the left `v` column by name to side-step any extra
    // dictionary-index columns.
    let v_idx = out
        .schema()
        .index_of("v")
        .expect("'v' column in output schema");
    let left_v = out.column(v_idx);
    let nulls: usize = (0..out.num_rows()).filter(|&i| left_v.is_null(i)).count();
    assert_eq!(nulls, 1, "RIGHT JOIN: one unmatched right row -> NULL left");
}

#[test]
#[ignore = "gpu:join"]
fn e2e_full_outer_join_emits_both_sides() {
    use craton_bolt::Engine;

    let (_, t1, t2) = provider_and_batches();
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    let h = engine
        .sql("SELECT * FROM t1 FULL OUTER JOIN t2 ON t1.id = t2.id")
        .expect("execute");
    let out = h.record_batch();
    // FULL: 2 matches + 2 left-only (id=1, 4) + 1 right-only (id=5) = 5 rows.
    assert_eq!(out.num_rows(), 5, "FULL OUTER expects 5 rows");
    let v_idx = out
        .schema()
        .index_of("v")
        .expect("'v' column in output schema");
    let w_idx = out
        .schema()
        .index_of("w")
        .expect("'w' column in output schema");
    let left_v = out.column(v_idx);
    let right_w = out.column(w_idx);
    // Left NULL count = unmatched right rows = 1.
    let left_nulls: usize = (0..out.num_rows()).filter(|&i| left_v.is_null(i)).count();
    assert_eq!(left_nulls, 1, "FULL: one right-only row gets NULL left");
    // Right NULL count = unmatched left rows = 2.
    let right_nulls: usize = (0..out.num_rows()).filter(|&i| right_w.is_null(i)).count();
    assert_eq!(right_nulls, 2, "FULL: two left-only rows get NULL right");
}

#[test]
#[ignore = "gpu:join"]
fn e2e_cross_join_row_count_is_product() {
    use craton_bolt::Engine;

    let (_, t1, t2) = provider_and_batches();
    let n_left = t1.num_rows();
    let n_right = t2.num_rows();
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    let h = engine
        .sql("SELECT * FROM t1 CROSS JOIN t2")
        .expect("execute");
    let out = h.record_batch();
    assert_eq!(
        out.num_rows(),
        n_left * n_right,
        "CROSS JOIN: |left| × |right| rows"
    );
}
