// SPDX-License-Identifier: AGPL-3.0-only

//! Per-table unregistration (`Engine::deregister_table` / `drop_table` /
//! `DROP TABLE`): dropping one table releases its host, dictionary and
//! device state without touching any other table or rebuilding the engine.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use craton_bolt::Engine;

fn ints(name: &str, n: i64, offset: i64) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from((offset..offset + n).collect::<Vec<_>>())) as ArrayRef],
    )
    .unwrap()
}

fn strings(values: &[&str]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(StringArray::from(values.to_vec())) as ArrayRef],
    )
    .unwrap()
}

fn scalar_i64(engine: &mut Engine, sql: &str) -> i64 {
    let out = engine.sql(sql).unwrap().into_record_batch();
    let col = arrow::compute::cast(out.column(0), &DataType::Int64).unwrap();
    col.as_any().downcast_ref::<Int64Array>().unwrap().value(0)
}

#[test]
#[ignore = "gpu:e2e"]
fn dropping_one_table_leaves_the_others_and_forgets_it() {
    let mut e = Engine::new().expect("CUDA context");
    e.register_table("a", ints("x", 1_000, 0)).unwrap();
    e.register_table("b", ints("y", 10, 0)).unwrap();
    assert_eq!(scalar_i64(&mut e, "SELECT SUM(x) FROM a"), 499_500);
    assert_eq!(scalar_i64(&mut e, "SELECT COUNT(*) FROM b"), 10);

    e.deregister_table("a").unwrap();
    assert_eq!(e.table_names(), vec!["b".to_string()]);
    let err = e.sql("SELECT SUM(x) FROM a").unwrap_err().to_string();
    assert!(err.contains('a'), "{err}");
    assert!(e.explain_sql("SELECT SUM(x) FROM a").is_err());
    assert!(
        e.deregister_table("a").is_err(),
        "a second drop reports the missing table"
    );
    assert_eq!(scalar_i64(&mut e, "SELECT COUNT(*) FROM b"), 10);

    // The name is free again, with a different schema: no cached plan,
    // statistic or resident column of the old table is reused.
    e.register_table("a", ints("z", 5, 100)).unwrap();
    assert_eq!(scalar_i64(&mut e, "SELECT SUM(z) FROM a"), 510);
    assert!(e.sql("SELECT SUM(x) FROM a").is_err());
}

#[test]
#[ignore = "gpu:e2e"]
fn dropped_dictionaries_are_not_reused() {
    let mut e = Engine::new().expect("CUDA context");
    e.register_table("t", strings(&["red", "green", "blue", "red"]))
        .unwrap();
    assert_eq!(
        scalar_i64(&mut e, "SELECT COUNT(*) FROM t WHERE s = 'red'"),
        2
    );
    e.execute_sql("DROP TABLE t").unwrap();
    e.execute_sql("DROP TABLE IF EXISTS t").unwrap();
    e.register_table("t", strings(&["blue", "amber", "red"]))
        .unwrap();
    assert_eq!(
        scalar_i64(&mut e, "SELECT COUNT(*) FROM t WHERE s = 'red'"),
        1
    );
    assert_eq!(
        scalar_i64(&mut e, "SELECT COUNT(*) FROM t WHERE s = 'amber'"),
        1
    );
    assert_eq!(
        scalar_i64(&mut e, "SELECT COUNT(*) FROM t WHERE s = 'green'"),
        0
    );
}
