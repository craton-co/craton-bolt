// SPDX-License-Identifier: Apache-2.0

//! Aggregates filtered by a string-equality predicate.
//!
//! The dictionary rewrite folds `s = 'lit'` into an integer comparison on the
//! hidden `__idx_s` column; the aggregate pre-stage must then read that column
//! from the registered table, which only holds the user-visible `s`. These
//! tests pin the end-to-end result for plain `Utf8` and dictionary-encoded
//! columns, across low/high cardinality and literal-present/absent cases.
//!
//! GPU-gated: run with `cargo test --test string_eq_aggregate_test -- --ignored`.

use std::sync::Arc;

use arrow_array::types::Int32Type;
use arrow_array::{
    Array, ArrayRef, DictionaryArray, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::{DataType, Field, Schema};
use craton_bolt::Engine;

fn utf8_table(values: &[&str]) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("s", ArrowDataType::Utf8, false),
        ArrowField::new("v", ArrowDataType::Int64, false),
    ]));
    let s: StringArray = values.iter().map(|v| Some(*v)).collect();
    let v: Int64Array = (0..values.len() as i64).collect();
    RecordBatch::try_new(schema, vec![Arc::new(s) as ArrayRef, Arc::new(v)]).unwrap()
}

fn scalar_i64(engine: &Engine, sql: &str) -> Option<i64> {
    let out = engine
        .sql(sql)
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .into_record_batch();
    assert_eq!(out.num_rows(), 1, "{sql}");
    let col = out.column(0);
    if col.is_null(0) {
        return None;
    }
    Some(
        col.as_any()
            .downcast_ref::<Int64Array>()
            .unwrap_or_else(|| panic!("{sql}: expected Int64, got {:?}", col.data_type()))
            .value(0),
    )
}

#[test]
#[ignore = "gpu:e2e — COUNT/SUM filtered by string equality"]
fn count_and_sum_filtered_by_string_equality() {
    let values = ["a", "b", "a", "c", "a", "b"];
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", utf8_table(&values)).unwrap();

    assert_eq!(
        scalar_i64(&engine, "SELECT COUNT(*) FROM t WHERE s = 'a'"),
        Some(3)
    );
    assert_eq!(
        scalar_i64(&engine, "SELECT COUNT(*) FROM t WHERE s = 'zzz'"),
        Some(0)
    );
    assert_eq!(
        scalar_i64(&engine, "SELECT COUNT(*) FROM t WHERE s <> 'a'"),
        Some(3)
    );
    assert_eq!(
        scalar_i64(&engine, "SELECT SUM(v) FROM t WHERE s = 'b'"),
        Some(1 + 5)
    );
    assert_eq!(
        scalar_i64(&engine, "SELECT COUNT(v) FROM t WHERE s = 'c'"),
        Some(1)
    );
}

#[test]
#[ignore = "gpu:e2e — single-valued string column"]
fn count_filtered_by_string_equality_single_distinct_value() {
    let values = ["same"; 64];
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", utf8_table(&values)).unwrap();
    assert_eq!(
        scalar_i64(&engine, "SELECT COUNT(*) FROM t WHERE s = 'same'"),
        Some(64)
    );
    assert_eq!(
        scalar_i64(&engine, "SELECT COUNT(*) FROM t WHERE s = 'other'"),
        Some(0)
    );
}

#[test]
#[ignore = "gpu:e2e — high-cardinality string column"]
fn count_filtered_by_string_equality_high_cardinality() {
    let owned: Vec<String> = (0..10_000).map(|i| format!("key-{}", i % 2_500)).collect();
    let values: Vec<&str> = owned.iter().map(String::as_str).collect();
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", utf8_table(&values)).unwrap();
    assert_eq!(
        scalar_i64(&engine, "SELECT COUNT(*) FROM t WHERE s = 'key-17'"),
        Some(4)
    );
}

#[test]
#[ignore = "gpu:e2e — lazy (dictionary-less) table"]
fn lazy_table_aggregate_filtered_by_string_equality() {
    let mut engine = Engine::new().expect("ctx");
    let decl = Schema::new(vec![
        Field::new("s", DataType::Utf8, false),
        Field::new("v", DataType::Int64, false),
    ]);
    engine
        .register_table_stream_lazy(
            "lz",
            decl,
            Box::new(|| Box::new(std::iter::once(Ok(utf8_table(&["a", "b", "a", "c"]))))),
        )
        .unwrap();
    assert_eq!(
        scalar_i64(&engine, "SELECT COUNT(*) FROM lz WHERE s = 'a'"),
        Some(2)
    );
    assert_eq!(
        scalar_i64(&engine, "SELECT SUM(v) FROM lz WHERE s <> 'a'"),
        Some(1 + 3)
    );
}

#[test]
#[ignore = "gpu:e2e — Utf8 group key under a predicate"]
fn group_by_utf8_key_with_predicate() {
    let mut engine = Engine::new().expect("ctx");
    engine
        .register_table("t", utf8_table(&["a", "b", "a", "c", "a", "b"]))
        .unwrap();
    let out = engine
        .sql("SELECT s, COUNT(*) AS n FROM t WHERE v > 0 GROUP BY s ORDER BY s")
        .unwrap()
        .into_record_batch();
    let keys = arrow::compute::cast(out.column(0), &arrow_schema::DataType::Utf8).unwrap();
    let keys = keys.as_any().downcast_ref::<StringArray>().unwrap();
    let n = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    let got: Vec<(String, i64)> = (0..out.num_rows())
        .map(|i| (keys.value(i).to_string(), n.value(i)))
        .collect();
    // v > 0 drops row 0 ('a').
    assert_eq!(got, vec![("a".into(), 2), ("b".into(), 2), ("c".into(), 1)]);
}

#[test]
#[ignore = "gpu:e2e — GROUP BY over JOIN"]
fn group_by_over_join() {
    let mut engine = Engine::new().expect("ctx");
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int64, false),
        ArrowField::new("w", ArrowDataType::Int64, false),
    ]));
    let dim = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10, 20, 30])),
        ],
    )
    .unwrap();
    let fact = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2, 3, 3, 3, 4])) as ArrayRef,
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7])),
        ],
    )
    .unwrap();
    engine.register_table("dim", dim).unwrap();
    engine.register_table("fact", fact).unwrap();
    let out = engine
        .sql(
            "SELECT dim.w AS g, SUM(fact.w) AS total, COUNT(*) AS n FROM fact              JOIN dim ON fact.k = dim.k GROUP BY dim.w ORDER BY g",
        )
        .unwrap()
        .into_record_batch();
    let col = |i: usize| -> Vec<i64> {
        let a = out.column(i).as_any().downcast_ref::<Int64Array>().unwrap();
        (0..a.len()).map(|r| a.value(r)).collect()
    };
    assert_eq!(col(0), vec![10, 20, 30]);
    assert_eq!(col(1), vec![1 + 2, 3, 4 + 5 + 6]);
    assert_eq!(col(2), vec![2, 1, 3]);
    // Scalar aggregate over a join too.
    assert_eq!(
        scalar_i64(
            &engine,
            "SELECT SUM(fact.w) FROM fact JOIN dim ON fact.k = dim.k"
        ),
        Some(21)
    );
}

/// Arrow allows duplicate dictionary values; every row whose value is 'x'
/// must match `s = 'x'`, whichever slot it is keyed through.
#[test]
#[ignore = "gpu:e2e — duplicate Arrow dictionary values"]
fn dictionary_with_duplicate_values_matches_every_row() {
    let values = StringArray::from(vec!["x", "y", "x"]);
    let keys = Int32Array::from(vec![0, 1, 2, 2, 0]);
    let dict = DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        ArrowDataType::Dictionary(
            Box::new(ArrowDataType::Int32),
            Box::new(ArrowDataType::Utf8),
        ),
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(dict) as ArrayRef]).unwrap();
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", batch).unwrap();
    assert_eq!(
        scalar_i64(&engine, "SELECT COUNT(*) FROM t WHERE s = 'x'"),
        Some(4)
    );
    assert_eq!(
        scalar_i64(&engine, "SELECT COUNT(*) FROM t WHERE s = 'y'"),
        Some(1)
    );
}

#[test]
#[ignore = "gpu:e2e — Utf8 projection under a numeric / folded predicate"]
fn utf8_projection_with_non_string_predicate() {
    let mut engine = Engine::new().expect("ctx");
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("s", ArrowDataType::Utf8, false),
        ArrowField::new("u", ArrowDataType::Utf8, false),
        ArrowField::new("v", ArrowDataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "c", "d"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["x", "y", "x", "y"])),
            Arc::new(Int64Array::from(vec![0, 1, 2, 3])),
        ],
    )
    .unwrap();
    engine.register_table("t", batch).unwrap();
    let strings = |sql: &str| -> Vec<String> {
        let out = engine
            .sql(sql)
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .into_record_batch();
        let col = arrow::compute::cast(out.column(0), &ArrowDataType::Utf8).unwrap();
        let col = col.as_any().downcast_ref::<StringArray>().unwrap();
        (0..col.len()).map(|i| col.value(i).to_string()).collect()
    };
    assert_eq!(strings("SELECT s, v FROM t WHERE v > 1"), vec!["c", "d"]);
    // `u` is not projected, so its predicate folds to `__idx_u = k`.
    assert_eq!(strings("SELECT s FROM t WHERE u = 'x'"), vec!["a", "c"]);
    assert_eq!(
        strings("SELECT s FROM t WHERE u = 'x' AND v > 0"),
        vec!["c"]
    );
}

/// A lazily-registered table gets string dictionaries when it is drained, so
/// its string predicates fold to the dictionary index exactly like an eager
/// table's (visible in EXPLAIN as `__idx_s`) — and results match.
#[test]
#[ignore = "gpu:e2e — lazy tables get dictionaries"]
fn lazy_table_string_predicates_use_dictionaries() {
    let decl = Schema::new(vec![
        Field::new("s", DataType::Utf8, false),
        Field::new("v", DataType::Int64, false),
    ]);
    let mut engine = Engine::new().expect("ctx");
    engine
        .register_table_stream_lazy(
            "lz",
            decl,
            Box::new(|| Box::new(std::iter::once(Ok(utf8_table(&["a", "b", "a", "c"]))))),
        )
        .unwrap();
    // First query drains the table (and builds its dictionaries) before the
    // dictionary rewrite runs.
    assert_eq!(
        scalar_i64(&engine, "SELECT COUNT(*) FROM lz WHERE s = 'a'"),
        Some(2)
    );
    let explain = engine
        .explain_sql("SELECT COUNT(*) FROM lz WHERE s = 'a'")
        .unwrap();
    assert!(explain.contains("__idx_s"), "{explain}");
    assert_eq!(
        scalar_i64(&engine, "SELECT SUM(v) FROM lz WHERE s <> 'a'"),
        Some(1 + 3)
    );
    assert_eq!(
        scalar_i64(&engine, "SELECT COUNT(*) FROM lz WHERE s = 'zz'"),
        Some(0)
    );
    let out = engine
        .sql("SELECT s FROM lz WHERE s = 'c'")
        .unwrap()
        .into_record_batch();
    assert_eq!(out.num_rows(), 1);
}

/// `(s Utf8 [nullable?], v Int64 [nullable?])` with a NULL `s` in row 5.
fn mixed_table(nullable: bool) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("s", ArrowDataType::Utf8, true),
        ArrowField::new("v", ArrowDataType::Int64, nullable),
    ]));
    let s: StringArray = (0..12)
        .map(|i| {
            if i == 5 {
                None
            } else {
                Some(["a", "b", "c"][i % 3])
            }
        })
        .collect();
    let v: Int64Array = (0..12i64).collect();
    RecordBatch::try_new(schema, vec![Arc::new(s) as ArrayRef, Arc::new(v)]).unwrap()
}

fn group_counts(engine: &Engine, sql: &str) -> Vec<(Option<String>, i64)> {
    let out = engine
        .sql(sql)
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .into_record_batch();
    let s = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Utf8 key");
    let n = out
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count");
    let mut rows: Vec<(Option<String>, i64)> = (0..out.num_rows())
        .map(|i| (s.is_valid(i).then(|| s.value(i).to_string()), n.value(i)))
        .collect();
    rows.sort();
    rows
}

/// `COUNT(*)` grouped by a string key counts every row of each group,
/// including the NULL-key group, whether or not the table has a
/// non-nullable column to count through.
#[test]
#[ignore = "gpu:e2e — COUNT(*) GROUP BY a Utf8 key"]
fn count_star_grouped_by_string_key() {
    let want = vec![
        (None, 1),
        (Some("a".to_string()), 4),
        (Some("b".to_string()), 4),
        (Some("c".to_string()), 3),
    ];
    let mut e = Engine::new().expect("ctx");
    e.register_table("dense", mixed_table(false)).unwrap();
    e.register_table("sparse", mixed_table(true)).unwrap();
    for t in ["dense", "sparse"] {
        assert_eq!(
            group_counts(&e, &format!("SELECT s, COUNT(*) AS n FROM {t} GROUP BY s")),
            want,
            "{t}"
        );
        // With a predicate, and next to another aggregate.
        let filtered = group_counts(
            &e,
            &format!("SELECT s, COUNT(*) AS n FROM {t} WHERE v >= 6 GROUP BY s"),
        );
        assert_eq!(filtered.iter().map(|r| r.1).sum::<i64>(), 6, "{t}");
    }
}

/// A Utf8 column projected next to computed columns.
#[test]
#[ignore = "gpu:e2e — Utf8 passthrough next to computed columns"]
fn utf8_column_next_to_computed_columns() {
    let mut e = Engine::new().expect("ctx");
    e.register_table("t", mixed_table(false)).unwrap();
    let out = e
        .sql("SELECT s, v * 2 AS w, 7 AS seven FROM t")
        .unwrap()
        .into_record_batch();
    assert_eq!(out.num_rows(), 12);
    let s = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let w = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    let seven = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(s.value(4), "b");
    assert!(s.is_null(5));
    assert_eq!(w.value(11), 22);
    assert!((0..12).all(|i| seven.value(i) == 7));
}
