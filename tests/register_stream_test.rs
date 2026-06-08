// SPDX-License-Identifier: Apache-2.0

//! End-to-end test for `Engine::register_table_stream`.
//!
//! The v0.6 cut consumes the iterator eagerly into the engine's existing
//! multi-batch in-memory table representation; v0.7+ is expected to land
//! lazy per-batch streaming behind the same API surface. This test pins
//! the API shape and the eager-consumption behaviour: a producer that
//! yields `BoltResult<RecordBatch>` items can register a multi-batch
//! table in one call, and a follow-up `SELECT` sees every row.

use std::sync::Arc;

use arrow_array::{Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::{DataType, Field, Schema};
use craton_bolt::{BoltError, BoltResult};

/// Build a batch of `(k, v)` Int32 rows with the given row values.
fn make_batch(ks: Vec<i32>, vs: Vec<i32>) -> RecordBatch {
    assert_eq!(ks.len(), vs.len(), "ks and vs must agree on length");
    let k = Int32Array::from(ks);
    let v = Int32Array::from(vs);
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Int32, false),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(k), Arc::new(v)]).unwrap()
}

/// Declared plan schema matching the batches built by `make_batch`.
fn declared_schema() -> Schema {
    Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int32, false),
    ])
}

/// E2E: register a 3-batch table via the streaming API and run a SELECT
/// that touches every batch. The summed `v` total proves all three
/// batches were installed (and concatenated) into the same table.
#[test]
#[ignore = "gpu:tier1"]
fn e2e_register_table_stream_basic_select() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");

    // Producer-style iterator: each item is a BoltResult<RecordBatch>.
    // Total v across all three batches: 1+2+3 + 10+20+30 + 100+200+300 = 666.
    let batches: Vec<BoltResult<RecordBatch>> = vec![
        Ok(make_batch(vec![1, 1, 1], vec![1, 2, 3])),
        Ok(make_batch(vec![2, 2, 2], vec![10, 20, 30])),
        Ok(make_batch(vec![3, 3, 3], vec![100, 200, 300])),
    ];

    engine
        .register_table_stream("t", declared_schema(), batches)
        .expect("register_table_stream");

    let h = engine
        .sql("SELECT SUM(v) FROM t")
        .expect("execute SELECT SUM(v)");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 1, "scalar aggregate returns one row");
    // SUM widens Int32 to Int64.
    let sum = out
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("SUM(Int32) -> Int64");
    assert_eq!(sum.value(0), 666, "every batch must reach the engine");

    // Also count rows to confirm the multi-batch concat is intact.
    let h2 = engine
        .sql("SELECT COUNT(*) FROM t")
        .expect("execute COUNT(*)");
    let out2 = h2.record_batch();
    let cnt = out2
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("COUNT(*) -> Int64");
    assert_eq!(cnt.value(0), 9, "9 rows across 3 batches of 3");
}

/// The API must NOT require GPU access to validate its contract: a
/// producer-side `Err` aborts registration and leaves the engine in a
/// state where the table is not registered. We can't run the full
/// `Engine` constructor without CUDA, so this is a structural test
/// gated the same way as the e2e above.
#[test]
#[ignore = "gpu:tier1"]
fn e2e_register_table_stream_producer_error_aborts() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");

    let batches: Vec<BoltResult<RecordBatch>> = vec![
        Ok(make_batch(vec![1], vec![1])),
        Err(BoltError::Other("simulated producer failure".to_string())),
        // Third item never reached.
        Ok(make_batch(vec![3], vec![3])),
    ];

    let err = engine
        .register_table_stream("t_err", declared_schema(), batches)
        .expect_err("producer error must propagate");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("simulated producer failure"),
        "error must carry the producer's message; got {msg}"
    );

    // Table must NOT be registered after the failed install — the
    // next call with the same name should succeed.
    let good: Vec<BoltResult<RecordBatch>> = vec![Ok(make_batch(vec![7, 8], vec![70, 80]))];
    engine
        .register_table_stream("t_err", declared_schema(), good)
        .expect("post-rollback re-register");

    let h = engine.sql("SELECT SUM(v) FROM t_err").expect("execute");
    let sum = h
        .record_batch()
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("SUM(Int32) -> Int64")
        .value(0);
    assert_eq!(sum, 150, "post-rollback table must have only the new rows");
}
