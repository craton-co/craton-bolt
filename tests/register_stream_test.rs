// SPDX-License-Identifier: AGPL-3.0-only

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

/// craton-bolt-008: `register_batch` extends the Utf8 dictionary with each
/// appended batch's values only and keeps the table as chunks. Queries
/// interleaved with appends must still see every row under both tier
/// policies: equality on a literal that only a later batch holds, values
/// that sort before every earlier one (the existing codes shift), LIKE,
/// GROUP BY over the Utf8 column, NULLs and COUNT.
#[test]
#[ignore = "gpu:tier1"]
fn e2e_register_batch_appends_keep_utf8_queries_correct() {
    use std::collections::HashMap;

    use arrow_array::{Int64Array, StringArray};
    use craton_bolt::{Engine, TierPolicy};

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int64, false),
        ArrowField::new("s", ArrowDataType::Utf8, true),
    ]));
    let value = |b: i64, i: i64| -> Option<String> {
        match i % 13 {
            0 => None,
            1 => Some(format!("a{b:02}")),
            2 => Some(format!("new{b:02}")),
            _ => Some(format!("v{:02}", i % 9)),
        }
    };
    let rows_per = 64i64;
    let make = |b: i64| {
        let ids: Vec<i64> = (b * rows_per..(b + 1) * rows_per).collect();
        let s: Vec<Option<String>> = ids.iter().map(|&i| value(b, i)).collect();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(s)),
            ],
        )
        .unwrap()
    };
    let scalar = |e: &Engine, q: &str| -> i64 {
        let out = e.sql(q).unwrap_or_else(|err| panic!("{q}: {err}"));
        let b = out.record_batch();
        let c = arrow::compute::cast(b.column(0), &ArrowDataType::Int64).unwrap();
        let c = c.as_any().downcast_ref::<Int64Array>().unwrap();
        if c.is_null(0) {
            0
        } else {
            c.value(0)
        }
    };

    for policy in [TierPolicy::GpuFirst, TierPolicy::CostBased] {
        let mut engine = Engine::new().expect("ctx");
        engine.set_tier_policy(policy);
        let mut all: Vec<(i64, Option<String>)> = Vec::new();
        for b in 0..12i64 {
            let batch = make(b);
            for i in b * rows_per..(b + 1) * rows_per {
                all.push((i, value(b, i)));
            }
            if b == 0 {
                engine.register_table("t", batch).unwrap();
            } else {
                engine.register_batch("t", batch).unwrap();
            }
            if b % 3 != 1 && b != 0 {
                continue;
            }
            let ctx = format!("{policy:?} after batch {b}");
            let count_eq = |lit: &str| all.iter().filter(|r| r.1.as_deref() == Some(lit)).count();
            assert_eq!(
                scalar(&engine, "SELECT COUNT(*) FROM t"),
                all.len() as i64,
                "{ctx}"
            );
            let lit = format!("a{b:02}");
            assert_eq!(
                scalar(
                    &engine,
                    &format!("SELECT COUNT(*) FROM t WHERE s = '{lit}'")
                ),
                count_eq(&lit) as i64,
                "{ctx}: literal only in the newest batch"
            );
            assert_eq!(
                scalar(&engine, "SELECT SUM(id) FROM t WHERE s = 'v04'"),
                all.iter()
                    .filter(|r| r.1.as_deref() == Some("v04"))
                    .map(|r| r.0)
                    .sum::<i64>(),
                "{ctx}: equality on a value every batch holds"
            );
            assert_eq!(
                scalar(&engine, "SELECT COUNT(*) FROM t WHERE s LIKE 'new%'"),
                all.iter()
                    .filter(|r| r.1.as_deref().is_some_and(|s| s.starts_with("new")))
                    .count() as i64,
                "{ctx}: LIKE"
            );
            assert_eq!(
                scalar(&engine, "SELECT COUNT(s) FROM t"),
                all.iter().filter(|r| r.1.is_some()).count() as i64,
                "{ctx}: NULLs"
            );
            let out = engine
                .sql("SELECT s, COUNT(*) AS n FROM t GROUP BY s")
                .unwrap_or_else(|e| panic!("{ctx}: GROUP BY: {e}"));
            let rb = out.record_batch();
            let keys = arrow::compute::cast(rb.column(0), &ArrowDataType::Utf8).unwrap();
            let keys = keys.as_any().downcast_ref::<StringArray>().unwrap();
            let ns = arrow::compute::cast(rb.column(1), &ArrowDataType::Int64).unwrap();
            let ns = ns.as_any().downcast_ref::<Int64Array>().unwrap();
            let got: HashMap<Option<String>, i64> = (0..rb.num_rows())
                .map(|r| {
                    let k = (!keys.is_null(r)).then(|| keys.value(r).to_string());
                    (k, ns.value(r))
                })
                .collect();
            let mut want: HashMap<Option<String>, i64> = HashMap::new();
            for r in &all {
                *want.entry(r.1.clone()).or_default() += 1;
            }
            assert_eq!(got, want, "{ctx}: GROUP BY s");
        }
    }
}
