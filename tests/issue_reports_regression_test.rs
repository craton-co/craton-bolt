// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the embedder-filed engine reports
//! (`craton-bolt-003` .. `craton-bolt-010`, and the extension pages that
//! reference them). Each case is a reproduction from a report, run under
//! both tier policies; one run reports every mismatch, not only the first.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Decimal128Array, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use craton_bolt::{Engine, TierPolicy};

/// What a reproduction must produce.
#[derive(Debug, Clone, Copy)]
enum Want {
    /// This many result rows.
    Rows(usize),
    /// One row whose first column is NULL.
    Null,
    /// One row whose first column, as `f64`, equals this.
    Value(f64),
    /// An error whose message contains this text.
    Error(&'static str),
}

fn outcome(engine: &mut Engine, q: &str) -> Result<RecordBatch, String> {
    engine
        .sql(q)
        .map(|o| o.into_record_batch())
        .map_err(|e| e.to_string())
}

fn first_as_f64(batch: &RecordBatch) -> Option<f64> {
    let col = arrow::compute::cast(batch.column(0), &DataType::Float64).ok()?;
    let col = col.as_any().downcast_ref::<Float64Array>()?;
    (!col.is_null(0)).then(|| col.value(0))
}

fn mismatch(engine: &mut Engine, q: &str, want: Want) -> Option<String> {
    let got = outcome(engine, q);
    let ok = match (&got, want) {
        (Ok(b), Want::Rows(n)) => b.num_rows() == n,
        (Ok(b), Want::Null) => b.num_rows() == 1 && b.column(0).is_null(0),
        (Ok(b), Want::Value(v)) => b.num_rows() == 1 && first_as_f64(b) == Some(v),
        (Err(e), Want::Error(text)) => e.contains(text),
        _ => false,
    };
    (!ok).then(|| {
        let got = match got {
            Ok(b) => format!(
                "{} rows, first = {:?}",
                b.num_rows(),
                (b.num_rows() > 0).then(|| first_as_f64(&b))
            ),
            Err(e) => format!("error: {e}"),
        };
        format!("{q}\n    want {want:?}\n    got  {got}")
    })
}

fn check(batch: RecordBatch, cases: &[(&str, Want)]) {
    let mut failures = Vec::new();
    for policy in [TierPolicy::GpuFirst, TierPolicy::CostBased] {
        let mut engine = Engine::builder()
            .tier_policy(policy)
            .build()
            .expect("CUDA context");
        engine.register_table("t", batch.clone()).unwrap();
        for (q, want) in cases {
            if let Some(m) = mismatch(&mut engine, q, *want) {
                failures.push(format!("[{policy:?}] {m}"));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// 100 rows: id = i, d / v NULL when i % 7 == 0 (craton-bolt-003).
fn empty_input_table() -> RecordBatch {
    let n = 100;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("d", DataType::Float64, true),
        Field::new("v", DataType::Int64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from((0..n).collect::<Vec<i32>>())) as ArrayRef,
            Arc::new(Float64Array::from(
                (0..n)
                    .map(|i| (i % 7 != 0).then_some(i as f64))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                (0..n)
                    .map(|i| (i % 7 != 0).then_some(i as i64))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

#[test]
#[ignore = "gpu:e2e"]
fn report_003_aggregates_over_empty_filtered_input_are_null() {
    check(
        empty_input_table(),
        &[
            ("SELECT SUM(d) FROM t WHERE d IS NULL", Want::Null),
            ("SELECT SUM(v) FROM t WHERE id < 0", Want::Null),
            ("SELECT MIN(v) FROM t WHERE id < 0", Want::Null),
            ("SELECT MAX(d) FROM t WHERE id < 0", Want::Null),
            ("SELECT AVG(d) FROM t WHERE id < 0", Want::Null),
            ("SELECT STDDEV_POP(d) FROM t WHERE id < 0", Want::Null),
            ("SELECT VAR_POP(d) FROM t WHERE id < 0", Want::Null),
            ("SELECT STDDEV_SAMP(d) FROM t WHERE id = 1", Want::Null),
            ("SELECT SUM(d) FROM t WHERE id < 0 AND d > 0", Want::Null),
            ("SELECT SUM(d + 1) FROM t WHERE id < 0", Want::Null),
            ("SELECT COUNT(d) FROM t WHERE id < 0", Want::Value(0.0)),
            ("SELECT SUM(v) FROM t WHERE d IS NOT NULL", Want::Value(4215.0)),
            ("SELECT MIN(v) FROM t WHERE id > 90", Want::Value(92.0)),
        ],
    );
}

/// 100 rows: id = i, dec = i * 1.25 (NULL when i % 7 == 0), s = 'v' || i % 6.
fn decimal_table() -> RecordBatch {
    let n = 100;
    let dec = Decimal128Array::from(
        (0..n)
            .map(|i| (i % 7 != 0).then_some(i as i128 * 125))
            .collect::<Vec<_>>(),
    )
    .with_precision_and_scale(10, 2)
    .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("dec", DataType::Decimal128(10, 2), true),
        Field::new("s", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from((0..n).collect::<Vec<i32>>())) as ArrayRef,
            Arc::new(dec),
            Arc::new(StringArray::from(
                (0..n).map(|i| format!("v{}", i % 6)).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

#[test]
#[ignore = "gpu:e2e"]
fn report_004_decimal128_predicates() {
    check(
        decimal_table(),
        &[
            ("SELECT id FROM t WHERE dec IS NULL", Want::Rows(15)),
            ("SELECT id FROM t WHERE dec IS NOT NULL", Want::Rows(85)),
            ("SELECT id FROM t WHERE dec >= 2.50", Want::Rows(84)),
            ("SELECT id FROM t WHERE dec >= 2.5", Want::Rows(84)),
            ("SELECT id FROM t WHERE dec > 2.501", Want::Rows(83)),
            ("SELECT id FROM t WHERE dec = 2.501", Want::Rows(0)),
            ("SELECT id FROM t WHERE dec = 2.5", Want::Rows(1)),
            ("SELECT id FROM t WHERE dec < 3", Want::Rows(2)),
            ("SELECT id FROM t WHERE 2.50 <= dec", Want::Rows(84)),
            (
                "SELECT id FROM t WHERE dec >= CAST(2.50 AS DECIMAL(10,2))",
                Want::Rows(84),
            ),
            ("SELECT id FROM t WHERE CAST(dec AS DOUBLE) >= 2.5", Want::Rows(84)),
            ("SELECT s FROM t WHERE id > 50", Want::Rows(49)),
            ("SELECT id FROM t WHERE s IN ('v1', 'v2')", Want::Rows(34)),
            ("SELECT CAST(dec AS DOUBLE) FROM t WHERE id = 1", Want::Value(1.25)),
            (
                "SELECT CAST(dec - 3 AS DOUBLE) FROM t WHERE id = 2",
                Want::Value(-0.5),
            ),
            ("SELECT COUNT(*) FROM t WHERE dec IS NULL", Want::Value(15.0)),
        ],
    );
}

/// 10 rows, one per day from 2024-01-01 UTC, in four timestamp flavours
/// plus a Date32 (craton-bolt-005).
fn timestamp_table() -> RecordBatch {
    let n = 10i64;
    let day_us = 86_400_000_000i64;
    let base_us = 1_704_067_200_000_000i64;
    let us: Vec<i64> = (0..n).map(|i| base_us + i * day_us).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("ts_ns", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("ts_us", DataType::Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("ts_ms", DataType::Timestamp(TimeUnit::Millisecond, None), false),
        Field::new(
            "ts_tz",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from((0..n as i32).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(TimestampNanosecondArray::from(
                us.iter().map(|v| v * 1000).collect::<Vec<_>>(),
            )),
            Arc::new(TimestampMicrosecondArray::from(us.clone())),
            Arc::new(TimestampMillisecondArray::from(
                us.iter().map(|v| v / 1000).collect::<Vec<_>>(),
            )),
            Arc::new(TimestampMicrosecondArray::from(us.clone()).with_timezone("UTC")),
        ],
    )
    .unwrap()
}

#[test]
#[ignore = "gpu:e2e"]
fn report_005_timestamp_literals_meet_every_unit() {
    let mut cases = Vec::new();
    for col in ["ts_ns", "ts_us", "ts_ms", "ts_tz"] {
        cases.push((
            format!("SELECT id FROM t WHERE {col} >= TIMESTAMP '2024-01-05 00:00:00'"),
            Want::Rows(6),
        ));
        cases.push((
            format!("SELECT id FROM t WHERE TIMESTAMP '2024-01-05 00:00:00' < {col}"),
            Want::Rows(5),
        ));
        cases.push((
            format!("SELECT id FROM t WHERE {col} = TIMESTAMP '2024-01-05'"),
            Want::Rows(1),
        ));
        // Sub-unit digits: no ms column value equals it; > rounds correctly.
        cases.push((
            format!("SELECT id FROM t WHERE {col} > TIMESTAMP '2024-01-04 23:59:59.9999995'"),
            Want::Rows(6),
        ));
        cases.push((
            format!("SELECT id FROM t WHERE {col} >= TIMESTAMPTZ '2024-01-05 02:00:00+02'"),
            Want::Rows(6),
        ));
        cases.push((
            format!(
                "SELECT id FROM t WHERE {col} >= CAST('2024-01-05 00:00:00' AS TIMESTAMP)"
            ),
            Want::Rows(6),
        ));
    }
    cases.push((
        "SELECT id FROM t WHERE ts_ms = TIMESTAMP '2024-01-05 00:00:00.0005'".into(),
        Want::Rows(0),
    ));
    let cases: Vec<(&str, Want)> = cases.iter().map(|(q, w)| (q.as_str(), *w)).collect();
    check(timestamp_table(), &cases);
}

/// 100 rows: id = i, s = 'v' || i % 6, NULL when i % 7 == 0 (craton-bolt-006).
fn utf8_table() -> RecordBatch {
    let n = 100;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("s", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from((0..n).collect::<Vec<i32>>())) as ArrayRef,
            Arc::new(StringArray::from(
                (0..n)
                    .map(|i| (i % 7 != 0).then(|| format!("v{}", i % 6)))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

#[test]
#[ignore = "gpu:e2e"]
fn report_006_utf8_null_in_and_not_equal() {
    let batch = utf8_table();
    let cases = [
        ("SELECT id FROM t WHERE s IS NOT NULL", Want::Rows(85)),
        ("SELECT id FROM t WHERE s IS NULL", Want::Rows(15)),
        ("SELECT id FROM t WHERE s IN ('v1', 'v2')", Want::Rows(28)),
        ("SELECT id FROM t WHERE s NOT IN ('v1', 'v2')", Want::Rows(57)),
        ("SELECT id, s FROM t WHERE id > 3", Want::Rows(96)),
        ("SELECT id FROM t WHERE s != 'v1'", Want::Rows(71)),
        ("SELECT id FROM t WHERE NOT (s = 'v1')", Want::Rows(71)),
        ("SELECT count(*) FROM t WHERE s = 'v1'", Want::Value(14.0)),
        ("SELECT count(*) FROM t WHERE s IS NULL", Want::Value(15.0)),
        ("SELECT id FROM t WHERE s IS NULL OR id = 1", Want::Rows(16)),
    ];
    check(batch.clone(), &cases);
    // EXPLAIN accepts exactly the plans `sql` runs.
    let mut engine = Engine::new().expect("CUDA context");
    engine.register_table("t", batch).unwrap();
    for (q, _) in cases {
        let explain = engine.explain_sql(q).is_ok();
        let run = engine.sql(q).is_ok();
        assert_eq!(explain, run, "{q}: explain ok = {explain}, sql ok = {run}");
    }
}

#[test]
#[ignore = "gpu:e2e"]
fn report_007_deep_predicates_do_not_overflow_a_2mib_stack() {
    let n = 5_000;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from((0..n).collect::<Vec<i32>>())) as ArrayRef,
            Arc::new(StringArray::from(
                (0..n).map(|i| format!("name_{i}")).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    let below = (0..n)
        .filter(|i| format!("name_{i}").as_str() <= "name_4")
        .count();
    let or_chain = |k: usize| {
        (0..k)
            .map(|i| format!("id = {i}"))
            .collect::<Vec<_>>()
            .join(" OR ")
    };
    let and_chain = (0..2_000)
        .map(|i| format!("id <> {}", i + 10_000))
        .collect::<Vec<_>>()
        .join(" AND ");
    let in_list = (0..5_000)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let queries: Vec<(String, usize)> = vec![
        ("SELECT id FROM t WHERE s LIKE 'name_1%'".into(), 1_111),
        ("SELECT id FROM t WHERE s LIKE '%'".into(), n as usize),
        ("SELECT id FROM t WHERE s > 'name_4'".into(), n as usize - below),
        (format!("SELECT id FROM t WHERE {}", or_chain(300)), 300),
        (format!("SELECT id FROM t WHERE {}", or_chain(3_000)), 3_000),
        (format!("SELECT id FROM t WHERE {and_chain}"), n as usize),
        (format!("SELECT id FROM t WHERE id IN ({in_list})"), n as usize),
    ];
    let handle = std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(move || {
            let mut failures = Vec::new();
            let mut engine = Engine::new().expect("CUDA context");
            engine.register_table("t", batch).unwrap();
            for (q, want) in &queries {
                let head = &q[..q.len().min(60)];
                if let Err(e) = engine.explain_sql(q) {
                    failures.push(format!("explain {head}: {e}"));
                }
                match engine.sql(q) {
                    Ok(out) if out.record_batch().num_rows() == *want => {}
                    Ok(out) => failures.push(format!(
                        "{head}: {} rows, want {want}",
                        out.record_batch().num_rows()
                    )),
                    Err(e) => failures.push(format!("{head}: {e}")),
                }
            }
            failures
        })
        .unwrap();
    let failures = handle.join().expect("worker thread overflowed or panicked");
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
#[ignore = "gpu:e2e"]
fn report_008_register_batch_is_proportional_to_the_batch() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    let rows = 2_048i64;
    let make = |b: i64| {
        let start = b * rows;
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from((start..start + rows).collect::<Vec<_>>())) as ArrayRef,
                Arc::new(StringArray::from(
                    (start..start + rows)
                        .map(|i| format!("s{}", i % 1000))
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    };
    let mut engine = Engine::new().expect("CUDA context");
    engine.register_table("t", make(0)).unwrap();
    let time = |engine: &mut Engine, from: i64, to: i64| {
        let t = std::time::Instant::now();
        for b in from..to {
            engine.register_batch("t", make(b)).unwrap();
        }
        t.elapsed()
    };
    let early = time(&mut engine, 1, 21);
    time(&mut engine, 21, 127);
    let late = time(&mut engine, 127, 147);
    // O(batch) appends: the last 20 cost about what the first 20 did.
    assert!(
        late < early * 4 + std::time::Duration::from_millis(50),
        "early {early:?} late {late:?}"
    );
    let out = engine.sql("SELECT COUNT(*) FROM t").unwrap();
    assert_eq!(first_as_f64(out.record_batch()), Some((147 * rows) as f64));
    let out = engine
        .sql("SELECT COUNT(*) FROM t WHERE s = 's7'")
        .unwrap();
    assert_eq!(first_as_f64(out.record_batch()), Some(301.0));
}

/// 100 rows whose NULL slots hold non-zero garbage, as DuckDB-exported
/// arrays do (W2-A-001): id = i, g = i % 5, v / d NULL when i % 7 == 0 and
/// for every row of group g = 4 when i >= 50 ... (see the counts below).
fn garbage_null_table() -> RecordBatch {
    use arrow_buffer::{NullBuffer, ScalarBuffer};
    let n = 100usize;
    // NULL when i % 7 == 0, and the whole group g = 4 is NULL.
    let valid: Vec<bool> = (0..n).map(|i| i % 7 != 0 && i % 5 != 4).collect();
    let nulls = NullBuffer::from(valid.clone());
    let v = Int32Array::new(
        ScalarBuffer::from((0..n).map(|i| if valid[i] { i as i32 } else { 1_000 }).collect::<Vec<_>>()),
        Some(nulls.clone()),
    );
    let d = Float64Array::new(
        ScalarBuffer::from(
            (0..n)
                .map(|i| if valid[i] { i as f64 - 50.0 } else { 777.0 })
                .collect::<Vec<_>>(),
        ),
        Some(nulls),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("g", DataType::Int32, false),
        Field::new("v", DataType::Int32, true),
        Field::new("d", DataType::Float64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from((0..n as i32).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(Int32Array::from((0..n as i32).map(|i| i % 5).collect::<Vec<_>>())),
            Arc::new(v),
            Arc::new(d),
        ],
    )
    .unwrap()
}

#[test]
#[ignore = "gpu:e2e"]
fn residual_validity_is_honoured_everywhere() {
    let b = garbage_null_table();
    let valid = |i: usize| i % 7 != 0 && i % 5 != 4;
    let nn = (0..100).filter(|&i| valid(i)).count();
    let pos = (0..100).filter(|&i| valid(i) && i > 50).count();
    let id_gt_20_nulls = (21..100).filter(|&i| !valid(i)).count();
    let id_gt_20_nulls = id_gt_20_nulls as f64;
    let cases: Vec<(String, Want)> = vec![
        ("SELECT d FROM t WHERE d > 0".into(), Want::Rows(pos)),
        ("SELECT COUNT(*) FROM t WHERE d > 0".into(), Want::Value(pos as f64)),
        (
            "SELECT COUNT(*) - COUNT(d) FROM t WHERE id > 20".into(),
            Want::Value(id_gt_20_nulls),
        ),
        ("SELECT d FROM t WHERE d IS NULL".into(), Want::Rows(100 - nn)),
        ("SELECT COUNT(d + 1) FROM t".into(), Want::Value(nn as f64)),
        (
            "SELECT SUM(CASE WHEN d > 0 THEN 1 ELSE 0 END) FROM t".into(),
            Want::Value(pos as f64),
        ),
        // The all-NULL group g = 4 aggregates to NULL.
        ("SELECT SUM(v) FROM t WHERE g = 4".into(), Want::Null),
        ("SELECT s FROM (SELECT g, SUM(v) AS s FROM t GROUP BY g) q WHERE g = 4".into(), Want::Null),
        ("SELECT m FROM (SELECT g, MIN(v) AS m FROM t GROUP BY g) q WHERE g = 4".into(), Want::Null),
        ("SELECT m FROM (SELECT g, MAX(d) AS m FROM t GROUP BY g) q WHERE g = 4".into(), Want::Null),
        // An aggregate below a sort / limit / distinct keeps its validity (W9-A-003).
        ("SELECT COUNT(v) AS c FROM t ORDER BY c".into(), Want::Value(nn as f64)),
        ("SELECT COUNT(v) FROM t LIMIT 1".into(), Want::Value(nn as f64)),
        ("SELECT DISTINCT COUNT(v) FROM t".into(), Want::Value(nn as f64)),
        // count(*) next to an INTEGER SUM (W5-A-001).
        (
            "SELECT COUNT(*) FROM (SELECT COUNT(*) AS c, SUM(v) AS s FROM t) q".into(),
            Want::Value(1.0),
        ),
        ("SELECT c FROM (SELECT COUNT(*) AS c, SUM(v) AS s FROM t) q".into(), Want::Value(100.0)),
        // An ascending sort puts NULL first (documented default) and keeps it NULL.
        ("SELECT v FROM t ORDER BY v LIMIT 1".into(), Want::Null),
        ("SELECT v FROM t ORDER BY v NULLS LAST LIMIT 1".into(), Want::Value(1.0)),
    ];
    let cases: Vec<(&str, Want)> = cases.iter().map(|(q, w)| (q.as_str(), *w)).collect();
    check(b, &cases);
}
