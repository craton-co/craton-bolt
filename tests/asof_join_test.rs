// SPDX-License-Identifier: Apache-2.0

//! `ASOF JOIN ... MATCH_CONDITION (...) [ON ...]`: each left row matched
//! with the closest right row by time, within equal keys.
//!
//! `cargo test --test asof_join_test -- --ignored --test-threads=1`

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{
    Array, Float64Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema, TimeUnit,
};

use craton_bolt::Engine;

fn engine() -> Engine {
    let mut e = Engine::new().expect("ctx");
    // Quotes: (sym, qt, price).
    let qs = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("sym", ArrowDataType::Utf8, false),
        ArrowField::new("qt", ArrowDataType::Int64, false),
        ArrowField::new("price", ArrowDataType::Float64, false),
    ]));
    let quotes = RecordBatch::try_new(
        qs,
        vec![
            Arc::new(StringArray::from(vec!["a", "a", "a", "b", "b"])),
            Arc::new(Int64Array::from(vec![10, 20, 30, 15, 25])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 10.0, 20.0])),
        ],
    )
    .unwrap();
    // Trades: (sym, tt, qty), deliberately unsorted.
    let ts = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("sym", ArrowDataType::Utf8, false),
        ArrowField::new("tt", ArrowDataType::Int64, false),
        ArrowField::new("qty", ArrowDataType::Int64, false),
    ]));
    let trades = RecordBatch::try_new(
        ts,
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "a", "a", "c", "b", "a"])),
            Arc::new(Int64Array::from(vec![25, 30, 5, 10, 12, 15, 99])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7])),
        ],
    )
    .unwrap();
    e.register_table("quotes", quotes).unwrap();
    e.register_table("trades", trades).unwrap();
    e
}

/// `(tt, price)` of `SELECT t.tt, q.price ... ORDER BY t.sym, t.tt`.
fn prices(e: &Engine, sql: &str) -> Vec<(i64, Option<f64>)> {
    let b = e.sql(sql).unwrap().into_record_batch();
    let tt = b.column(0).as_primitive::<Int64Type>();
    let p = b.column(1).as_primitive::<Float64Type>();
    (0..b.num_rows())
        .map(|i| (tt.value(i), (!p.is_null(i)).then(|| p.value(i))))
        .collect()
}

#[test]
#[ignore = "gpu:e2e — ASOF JOIN picks the closest earlier / later right row"]
fn asof_join_matches_closest_rows() {
    let e = engine();
    // Latest quote at or before each trade.
    let ge = prices(
        &e,
        "SELECT t.tt, q.price FROM trades t ASOF JOIN quotes q \
         MATCH_CONDITION (t.tt >= q.qt) ON t.sym = q.sym ORDER BY t.sym, t.tt",
    );
    assert_eq!(
        ge,
        vec![
            (5, None),        // a: before every quote
            (10, Some(1.0)),  // a: exact time
            (25, Some(2.0)),  // a
            (99, Some(3.0)),  // a
            (15, Some(10.0)), // b: exact
            (30, Some(20.0)), // b
            (12, None),       // c: no quotes
        ]
    );
    // Strictly before.
    let gt = prices(
        &e,
        "SELECT t.tt, q.price FROM trades t ASOF JOIN quotes q \
         MATCH_CONDITION (t.tt > q.qt) ON t.sym = q.sym ORDER BY t.sym, t.tt",
    );
    assert_eq!(gt[1], (10, None));
    assert_eq!(gt[4], (15, None));
    // The earliest quote at or after (written right-to-left).
    let le = prices(
        &e,
        "SELECT t.tt, q.price FROM trades t ASOF JOIN quotes q \
         MATCH_CONDITION (q.qt >= t.tt) ON t.sym = q.sym ORDER BY t.sym, t.tt",
    );
    assert_eq!(
        le,
        vec![
            (5, Some(1.0)),
            (10, Some(1.0)),
            (25, Some(3.0)),
            (99, None),
            (15, Some(10.0)),
            (30, None),
            (12, None),
        ]
    );
}

#[test]
#[ignore = "gpu:e2e — ASOF JOIN under filters, aggregates and without keys"]
fn asof_join_composes_with_the_rest_of_the_query() {
    let e = engine();
    // Notional per symbol of trades with a matched quote.
    let b = e
        .sql(
            "SELECT t.sym, SUM(q.price * t.qty) AS notional, COUNT(q.price) AS matched \
             FROM trades t ASOF JOIN quotes q MATCH_CONDITION (t.tt >= q.qt) \
             ON t.sym = q.sym WHERE t.qty > 1 GROUP BY t.sym ORDER BY t.sym",
        )
        .unwrap()
        .into_record_batch();
    let syms: Vec<&str> = (0..b.num_rows())
        .map(|i| b.column(0).as_string::<i32>().value(i))
        .collect();
    assert_eq!(syms, vec!["a", "b", "c"]);
    let notional = arrow::compute::cast(b.column(1), &ArrowDataType::Float64).unwrap();
    let notional = notional.as_primitive::<Float64Type>();
    // a: qty 3 (no quote), 4 * 1.0, 7 * 3.0; b: 2 * 20.0, 6 * 10.0; c: none.
    assert_eq!(notional.value(0), 4.0 + 21.0);
    assert_eq!(notional.value(1), 40.0 + 60.0);
    assert!(notional.is_null(2));
    let matched = arrow::compute::cast(b.column(2), &ArrowDataType::Int64).unwrap();
    assert_eq!(
        matched.as_primitive::<Int64Type>().values().to_vec(),
        vec![2, 2, 0]
    );
    // No equality keys: the closest quote of any symbol.
    let any = prices(
        &e,
        "SELECT t.tt, q.price FROM trades t ASOF JOIN quotes q \
         MATCH_CONDITION (t.tt >= q.qt) ORDER BY t.tt",
    );
    assert_eq!(any[0], (5, None));
    assert_eq!(any[1], (10, Some(1.0)));
    assert_eq!(any[2], (12, Some(1.0)));
    assert_eq!(any.last(), Some(&(99, Some(3.0))));
    let plan = e
        .explain_sql(
            "SELECT t.tt, q.price FROM trades t ASOF JOIN quotes q \
             MATCH_CONDITION (t.tt >= q.qt) ON t.sym = q.sym",
        )
        .unwrap();
    assert!(plan.contains("AsOfJoin: left.tt >= right.qt"), "{plan}");
    // Malformed shapes are rejected.
    assert!(e
        .sql("SELECT * FROM trades t ASOF JOIN quotes q MATCH_CONDITION (t.tt = q.qt)")
        .is_err());
}

#[test]
#[ignore = "gpu:e2e — ASOF JOIN on timestamps"]
fn asof_join_on_timestamps() {
    let mut e = Engine::new().expect("ctx");
    let ts = |v: Vec<i64>| Arc::new(TimestampMicrosecondArray::from(v)) as arrow_array::ArrayRef;
    let schema = |t: &str, v: &str| {
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new(
                t,
                ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            ArrowField::new(v, ArrowDataType::Int64, false),
        ]))
    };
    let events = RecordBatch::try_new(
        schema("at", "id"),
        vec![
            ts(vec![1_000, 5_000, 9_000]),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();
    let levels = RecordBatch::try_new(
        schema("since", "level"),
        vec![
            ts(vec![0, 4_000, 8_000]),
            Arc::new(Int64Array::from(vec![10, 20, 30])),
        ],
    )
    .unwrap();
    e.register_table("events", events).unwrap();
    e.register_table("levels", levels).unwrap();
    let b = e
        .sql(
            "SELECT e.id, l.level FROM events e ASOF JOIN levels l \
             MATCH_CONDITION (e.at >= l.since) ORDER BY e.id",
        )
        .unwrap()
        .into_record_batch();
    assert_eq!(
        b.column(1).as_primitive::<Int64Type>().values().to_vec(),
        vec![10, 20, 30]
    );
}
