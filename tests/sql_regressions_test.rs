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
use arrow_array::{Array, Float64Array, Int32Array, RecordBatch, StringArray};
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

#[test]
#[ignore = "gpu:e2e — APPROX_COUNT_DISTINCT / APPROX_DISTINCT"]
fn approx_distinct_counts() {
    let e = engine();
    let out = e
        .sql("SELECT APPROX_COUNT_DISTINCT(s) FROM t")
        .unwrap()
        .into_record_batch();
    assert_eq!(out.column(0).as_primitive::<Int64Type>().value(0), 3);
    let g = e
        .sql("SELECT k, APPROX_DISTINCT(s) AS d FROM t GROUP BY k")
        .unwrap()
        .into_record_batch();
    assert_eq!(g.num_rows(), 4);
    assert!(g
        .column(1)
        .as_primitive::<Int64Type>()
        .values()
        .iter()
        .all(|&d| d == 3));
}

/// Aggregates filtered by `IS NULL` / `IS NOT NULL` read garbage validity
/// pointers in the fused pre-stage predicate kernel (CUDA "invalid argument"
/// or an access violation).
#[test]
#[ignore = "gpu:e2e — aggregates under an IS NULL filter"]
fn aggregates_filtered_by_is_null() {
    let mut e = Engine::new().expect("ctx");
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("a", ArrowDataType::Int64, false),
        ArrowField::new("g", ArrowDataType::Int32, false),
        ArrowField::new("n", ArrowDataType::Float64, true),
        ArrowField::new("full", ArrowDataType::Float64, true),
    ]));
    let b = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int64Array::from(vec![1, 2, 3, 4, 5, 6])),
            Arc::new(Int32Array::from(vec![0, 0, 1, 1, 2, 2])),
            Arc::new(Float64Array::from(vec![
                Some(1.0),
                None,
                Some(2.0),
                None,
                None,
                Some(3.0),
            ])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        ],
    )
    .unwrap();
    e.register_table("t", b).unwrap();
    let one = |sql: &str| -> i64 {
        e.sql(sql)
            .unwrap()
            .into_record_batch()
            .column(0)
            .as_primitive::<Int64Type>()
            .value(0)
    };
    assert_eq!(one("SELECT COUNT(*) FROM t WHERE n IS NULL"), 3);
    assert_eq!(one("SELECT COUNT(a) FROM t WHERE n IS NOT NULL"), 3);
    assert_eq!(one("SELECT SUM(a) FROM t WHERE n IS NULL"), 2 + 4 + 5);
    // A nullable column without NULLs.
    assert_eq!(one("SELECT COUNT(*) FROM t WHERE full IS NULL"), 0);
    assert_eq!(one("SELECT SUM(a) FROM t WHERE full IS NOT NULL"), 21);
    let g = e
        .sql("SELECT g, SUM(a) AS s FROM t WHERE n IS NULL GROUP BY g ORDER BY g")
        .unwrap()
        .into_record_batch();
    let s: Vec<i64> = g.column(1).as_primitive::<Int64Type>().values().to_vec();
    assert_eq!(s, vec![2, 4, 5]);
}

/// Two GPU NULL bugs over nullable primitive columns: comparisons treated a
/// NULL as 0 (`WHERE i < 5` kept NULL rows, computed outputs lost their
/// NULLs), and `SUM(Int32)` through the aggregate pre-stage wrote its i64
/// block partials into an i32 buffer (only the first 256 rows were summed;
/// the rest was an out-of-bounds device write). `COUNT(col)` there also
/// counted NULLs.
#[test]
#[ignore = "gpu:e2e — NULLs in GPU predicates, projections and aggregates"]
fn gpu_nulls_in_predicates_projections_and_sums() {
    let mut e = Engine::new().expect("ctx");
    let n = 1000i64;
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int64, false),
        ArrowField::new("i", ArrowDataType::Int32, true),
    ]));
    let i: Vec<Option<i32>> = (0..n)
        .map(|x| (x % 7 != 3).then_some((x % 50) as i32 - 20))
        .collect();
    let b = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int64Array::from((0..n).collect::<Vec<_>>())),
            Arc::new(Int32Array::from(i.clone())),
        ],
    )
    .unwrap();
    e.register_table("t", b).unwrap();
    let one = |sql: &str| -> i64 {
        e.sql(sql)
            .unwrap()
            .into_record_batch()
            .column(0)
            .as_primitive::<Int64Type>()
            .value(0)
    };
    let valid: Vec<i64> = i.iter().flatten().map(|&v| v as i64).collect();
    assert_eq!(
        one("SELECT COUNT(*) FROM t WHERE i < 5"),
        valid.iter().filter(|&&v| v < 5).count() as i64
    );
    assert_eq!(one("SELECT SUM(i) FROM t"), valid.iter().sum::<i64>());
    let s = e
        .sql("SELECT COUNT(*) AS n, COUNT(i) AS c, SUM(i) AS s FROM t")
        .unwrap()
        .into_record_batch();
    assert_eq!(s.column(0).as_primitive::<Int64Type>().value(0), n);
    assert_eq!(
        s.column(1).as_primitive::<Int64Type>().value(0),
        valid.len() as i64
    );
    assert_eq!(
        s.column(2).as_primitive::<Int64Type>().value(0),
        valid.iter().sum::<i64>()
    );
    assert_eq!(
        one("SELECT COUNT(i) FROM t WHERE k > 3"),
        (4..n).filter(|x| x % 7 != 3).count() as i64
    );
    let p = e
        .sql("SELECT k, i + 1 AS j FROM t WHERE k < 10")
        .unwrap()
        .into_record_batch();
    let j = arrow::compute::cast(p.column(1), &ArrowDataType::Int64).unwrap();
    let j = j.as_primitive::<Int64Type>();
    assert!(j.is_null(3), "i is NULL at k = 3");
    assert_eq!(j.value(4), 4 - 20 + 1);
}

/// Several aggregates of one column under a predicate (`SUM(v), MIN(v),
/// MAX(v) ... WHERE ...`) failed: their pre-stage feeds shared the output
/// name `v` ("duplicate pre output name").
#[test]
#[ignore = "gpu:e2e — several aggregates of one column under a predicate"]
fn repeated_aggregate_column_under_predicate() {
    let e = engine();
    let out = e
        .sql("SELECT k, SUM(v) AS s, MIN(v) AS lo, MAX(v) AS hi FROM t WHERE v > 2 GROUP BY k ORDER BY k")
        .unwrap()
        .into_record_batch();
    let col = |c: usize| {
        out.column(c)
            .as_primitive::<Float64Type>()
            .values()
            .to_vec()
    };
    // k = i % 4, v = i, rows with v > 2: i in 3..12.
    let mut s = [0.0f64; 4];
    let mut lo = [f64::MAX; 4];
    let mut hi = [f64::MIN; 4];
    for i in 3..12 {
        let k = i % 4;
        s[k] += i as f64;
        lo[k] = lo[k].min(i as f64);
        hi[k] = hi[k].max(i as f64);
    }
    assert_eq!(col(1), s.to_vec());
    assert_eq!(col(2), lo.to_vec());
    assert_eq!(col(3), hi.to_vec());
}

/// A grouped `SUM` / `MIN` / `MAX` over a group whose values are all NULL
/// returned the accumulator identity (0, `i64::MAX`, `inf`, NaN) instead of
/// SQL NULL; float `MIN` / `MAX` under a predicate failed ("MIN/MAX over
/// float not yet supported").
#[test]
#[ignore = "gpu:e2e — grouped aggregates of an all-NULL group are NULL"]
fn grouped_aggregates_of_all_null_groups_are_null() {
    let mut e = Engine::new().expect("ctx");
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int64, false),
        ArrowField::new("g", ArrowDataType::Int64, false),
        ArrowField::new("f", ArrowDataType::Float64, true),
        ArrowField::new("i", ArrowDataType::Int64, true),
    ]));
    let b = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int64Array::from(vec![1, 2, 3, 4, 5, 6])),
            Arc::new(arrow_array::Int64Array::from(vec![1, 1, 2, 2, 3, 3])),
            Arc::new(Float64Array::from(vec![
                Some(1.5),
                Some(-2.0),
                None,
                None,
                Some(7.0),
                None,
            ])),
            Arc::new(arrow_array::Int64Array::from(vec![
                Some(1),
                Some(-2),
                None,
                None,
                Some(7),
                None,
            ])),
        ],
    )
    .unwrap();
    e.register_table("t", b).unwrap();
    let f64s = |b: &RecordBatch, c: usize| -> Vec<Option<f64>> {
        let a = b.column(c).as_primitive::<Float64Type>();
        (0..a.len())
            .map(|r| (!a.is_null(r)).then(|| a.value(r)))
            .collect()
    };
    let i64s = |b: &RecordBatch, c: usize| -> Vec<Option<i64>> {
        let a = b.column(c).as_primitive::<Int64Type>();
        (0..a.len())
            .map(|r| (!a.is_null(r)).then(|| a.value(r)))
            .collect()
    };
    let o = e
        .sql("SELECT g, MIN(f) AS a, MAX(f) AS b FROM t WHERE k > 0 GROUP BY g ORDER BY g")
        .unwrap()
        .into_record_batch();
    assert_eq!(f64s(&o, 1), vec![Some(-2.0), None, Some(7.0)]);
    assert_eq!(f64s(&o, 2), vec![Some(1.5), None, Some(7.0)]);
    let o = e
        .sql("SELECT g, MIN(i) AS a, MAX(i) AS b, SUM(i) AS s, COUNT(i) AS n FROM t GROUP BY g ORDER BY g")
        .unwrap()
        .into_record_batch();
    assert_eq!(i64s(&o, 1), vec![Some(-2), None, Some(7)]);
    assert_eq!(i64s(&o, 2), vec![Some(1), None, Some(7)]);
    assert_eq!(i64s(&o, 3), vec![Some(-1), None, Some(7)]);
    assert_eq!(i64s(&o, 4), vec![Some(2), Some(0), Some(1)]);
    let o = e
        .sql("SELECT g, MIN(f) AS a FROM t GROUP BY g ORDER BY g")
        .unwrap()
        .into_record_batch();
    assert_eq!(f64s(&o, 1), vec![Some(-2.0), None, Some(7.0)]);
}
