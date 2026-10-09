// SPDX-License-Identifier: Apache-2.0

//! SQL shapes that failed before the analytical-database round and were
//! found by the DuckDB differential lane: ORDER BY a column the SELECT list
//! does not output, GROUP BY over a join of a dictionary-encoded table under
//! an ORDER BY, and string functions next to computed columns.
//!
//! `cargo test --test sql_regressions_test -- --ignored --test-threads=1`

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{Float64Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::Engine;

/// `(k = i % 4, v = i, s = ["a", "bb", "ccc"][i % 3])` for i in 0..12.
fn t() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Float64, false),
        ArrowField::new("s", ArrowDataType::Utf8, false),
    ]));
    let k: Int32Array = (0..12).map(|i| i % 4).collect();
    let v: Float64Array = (0..12).map(|i| i as f64).collect();
    let s: StringArray = (0..12).map(|i| Some(["a", "bb", "ccc"][i % 3])).collect();
    RecordBatch::try_new(schema, vec![Arc::new(k), Arc::new(v), Arc::new(s)]).unwrap()
}

fn engine() -> Engine {
    let mut e = Engine::new().expect("ctx");
    e.register_table("t", t()).unwrap();
    e
}

#[test]
#[ignore = "gpu:e2e — ORDER BY a column outside the SELECT list"]
fn order_by_hidden_column() {
    let e = engine();
    let out = e
        .sql("SELECT v FROM t ORDER BY k DESC, v")
        .unwrap()
        .into_record_batch();
    assert_eq!(out.num_columns(), 1, "the sort key stays hidden");
    let got: Vec<f64> = out
        .column(0)
        .as_primitive::<Float64Type>()
        .values()
        .to_vec();
    assert_eq!(
        got,
        vec![3.0, 7.0, 11.0, 2.0, 6.0, 10.0, 1.0, 5.0, 9.0, 0.0, 4.0, 8.0]
    );
    let top = e
        .sql("SELECT v, LENGTH(s) AS n FROM t ORDER BY k, v LIMIT 2")
        .unwrap()
        .into_record_batch();
    assert_eq!(
        top.column(0)
            .as_primitive::<Float64Type>()
            .values()
            .to_vec(),
        vec![0.0, 4.0]
    );
    assert_eq!(
        top.column(1).as_primitive::<Int64Type>().values().to_vec(),
        vec![1, 2]
    );
}

#[test]
#[ignore = "gpu:e2e — GROUP BY over a join with ORDER BY on a dictionary table"]
fn grouped_join_with_order_by() {
    let mut e = engine();
    let u = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "uk",
            ArrowDataType::Int32,
            false,
        )])),
        vec![Arc::new(Int32Array::from(vec![0, 1]))],
    )
    .unwrap();
    e.register_table("u", u).unwrap();
    let out = e
        .sql("SELECT k, COUNT(*) AS n FROM t JOIN u ON t.k = u.uk GROUP BY k ORDER BY k")
        .unwrap()
        .into_record_batch();
    assert_eq!(out.num_rows(), 2);
    assert_eq!(
        out.column(1).as_primitive::<Int64Type>().values().to_vec(),
        vec![3, 3]
    );
}

#[test]
#[ignore = "gpu:e2e — string functions next to computed columns"]
fn string_functions_next_to_computed_columns() {
    let e = engine();
    let out = e
        .sql("SELECT UPPER(s) AS u, LENGTH(s) AS n, v * 2 AS w FROM t WHERE k = 1")
        .unwrap()
        .into_record_batch();
    assert_eq!(out.num_rows(), 3);
    let u = out.column(0).as_string::<i32>();
    let mut ups: Vec<&str> = (0..3).map(|i| u.value(i)).collect();
    ups.sort();
    assert_eq!(ups, vec!["A", "BB", "CCC"]);
    let n: i64 = out
        .column(1)
        .as_primitive::<Int64Type>()
        .values()
        .iter()
        .sum();
    assert_eq!(n, 6);
}
