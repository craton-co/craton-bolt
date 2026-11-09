// SPDX-License-Identifier: Apache-2.0

//! Extended column types (W1-D-007): tables holding `Time32/Time64`,
//! `Duration`, `Binary` / `LargeBinary` / `FixedSizeBinary`, `Interval`,
//! `Decimal256`, `Null` and nested (`List`, `Struct`, `Map`) columns register,
//! store and come back unchanged, and the operations each type supports give
//! the same answer on the GPU path and the cost-based host tier.
//!
//! `cargo test --test extended_types_test -- --ignored --test-threads=1`

use std::sync::Arc;

use arrow::compute::{concat_batches, take};
use arrow_array::builder::{Int32Builder, MapBuilder, StringBuilder};
use arrow_array::{
    Array, ArrayRef, BinaryArray, Decimal256Array, DurationMicrosecondArray, FixedSizeBinaryArray,
    Int32Array, Int64Array, IntervalMonthDayNanoArray, LargeBinaryArray, LargeStringArray,
    ListArray, NullArray, RecordBatch, StringArray, StructArray, Time32MillisecondArray,
    Time64MicrosecondArray, TimestampMicrosecondArray, UInt32Array,
};
use arrow_buffer::{i256, IntervalMonthDayNano, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema, TimeUnit,
};

use craton_bolt::metrics::{metrics, Counter};
use craton_bolt::plan::{DataType, Field, Schema};
use craton_bolt::{Engine, TierPolicy};

const N: usize = 100;
const POLICIES: [TierPolicy; 2] = [TierPolicy::GpuFirst, TierPolicy::CostBased];

fn nulls(valid: impl Fn(usize) -> bool, n: usize) -> NullBuffer {
    NullBuffer::from((0..n).map(valid).collect::<Vec<bool>>())
}

fn ts32_ms(i: usize) -> i32 {
    (i as i32 * 61_000) % 86_400_000
}

fn tm_us(i: usize) -> i64 {
    ts32_ms(i) as i64 * 1000 + (i % 2) as i64
}

fn tm_valid(i: usize) -> bool {
    i % 7 != 3
}
fn ts32_valid(i: usize) -> bool {
    i % 5 != 2
}
fn d_valid(i: usize) -> bool {
    i % 4 != 1
}
fn b_valid(i: usize) -> bool {
    i % 6 != 4
}
fn b_bytes(i: usize) -> Vec<u8> {
    vec![(i % 5) as u8, 0xff, (i % 3) as u8]
}

/// Rows `lo..hi` of the extended-type fixture. Every NULL slot of a
/// fixed-width column holds a non-zero garbage value, and the NULL list slots
/// own child elements, so a reader that ignores validity shows up.
fn fixture(lo: usize, hi: usize) -> RecordBatch {
    let n = hi - lo;
    let rows = || lo..hi;
    let k = Int64Array::from(rows().map(|i| i as i64).collect::<Vec<_>>());
    let tm = Time64MicrosecondArray::new(
        ScalarBuffer::from(
            rows()
                .map(|i| if tm_valid(i) { tm_us(i) } else { 999 })
                .collect::<Vec<_>>(),
        ),
        Some(nulls(|j| tm_valid(lo + j), n)),
    );
    let ts32 = Time32MillisecondArray::new(
        ScalarBuffer::from(
            rows()
                .map(|i| if ts32_valid(i) { ts32_ms(i) } else { 777 })
                .collect::<Vec<_>>(),
        ),
        Some(nulls(|j| ts32_valid(lo + j), n)),
    );
    let d = DurationMicrosecondArray::new(
        ScalarBuffer::from(
            rows()
                .map(|i| {
                    if d_valid(i) {
                        i as i64 * 1_000_000 - 30_000_000
                    } else {
                        123_456
                    }
                })
                .collect::<Vec<_>>(),
        ),
        Some(nulls(|j| d_valid(lo + j), n)),
    );
    let b = BinaryArray::from_iter(rows().map(|i| b_valid(i).then(|| b_bytes(i))));
    let lb = LargeBinaryArray::from_iter(rows().map(|i| b_valid(i).then(|| b_bytes(i))));
    let fb = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
        rows().map(|i| (i % 9 != 0).then(|| [(i * 7 % 256) as u8; 16])),
        16,
    )
    .unwrap();
    let lu =
        LargeStringArray::from_iter(rows().map(|i| (i % 10 != 0).then(|| format!("s{}", i % 4))));
    let iv = IntervalMonthDayNanoArray::from_iter(rows().map(|i| {
        (i % 3 != 0).then(|| IntervalMonthDayNano::new(i as i32, 2 * i as i32, 1000 * i as i64))
    }));
    let dec = Decimal256Array::from_iter(
        rows().map(|i| (i % 8 != 7).then(|| i256::from_i128(i as i128 * 12_345 - 99_999))),
    )
    .with_precision_and_scale(40, 2)
    .unwrap();
    let nul = NullArray::new(n);
    // List<Int32>: NULL slots still own one child element (garbage).
    let mut offsets = vec![0i32];
    let mut values = Vec::new();
    for i in rows() {
        let len = if i % 5 == 0 { 1 } else { i % 3 };
        for x in 0..len {
            values.push((i * 10 + x) as i32);
        }
        offsets.push(values.len() as i32);
    }
    let item = Arc::new(ArrowField::new("item", ArrowDataType::Int32, true));
    let l = ListArray::try_new(
        Arc::clone(&item),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        Arc::new(Int32Array::from(values)),
        Some(nulls(|j| (lo + j) % 5 != 0, n)),
    )
    .unwrap();
    let st = StructArray::try_new(
        vec![
            ArrowField::new("a", ArrowDataType::Int32, true),
            ArrowField::new("s", ArrowDataType::Utf8, true),
        ]
        .into(),
        vec![
            Arc::new(Int32Array::from(
                rows().map(|i| i as i32).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows().map(|i| format!("x{i}")).collect::<Vec<_>>(),
            )),
        ],
        Some(nulls(|j| (lo + j) % 7 != 6, n)),
    )
    .unwrap();
    let mut mb = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    for i in rows() {
        if i % 4 == 0 {
            mb.append(false).unwrap();
        } else {
            mb.keys().append_value("k");
            mb.values().append_value(i as i32);
            mb.append(true).unwrap();
        }
    }
    let m = mb.finish();
    let cols: Vec<ArrayRef> = vec![
        Arc::new(k),
        Arc::new(tm),
        Arc::new(ts32),
        Arc::new(d),
        Arc::new(b),
        Arc::new(lb),
        Arc::new(fb),
        Arc::new(lu),
        Arc::new(iv),
        Arc::new(dec),
        Arc::new(nul),
        Arc::new(l),
        Arc::new(st),
        Arc::new(m),
    ];
    let names = [
        "k", "tm", "ts32", "d", "b", "lb", "fb", "lu", "iv", "dec", "nul", "l", "st", "m",
    ];
    let fields: Vec<ArrowField> = names
        .iter()
        .zip(&cols)
        .map(|(n, c)| ArrowField::new(*n, c.data_type().clone(), true))
        .collect();
    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), cols).unwrap()
}

/// The fixture as the engine returns it: `LargeUtf8` is stored as `Utf8`.
fn normalized(batch: &RecordBatch) -> RecordBatch {
    let cols: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|c| match c.data_type() {
            ArrowDataType::LargeUtf8 => arrow::compute::cast(c, &ArrowDataType::Utf8).unwrap(),
            _ => Arc::clone(c),
        })
        .collect();
    let fields: Vec<ArrowField> = batch
        .schema()
        .fields()
        .iter()
        .zip(&cols)
        .map(|(f, c)| ArrowField::new(f.name(), c.data_type().clone(), true))
        .collect();
    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), cols).unwrap()
}

/// `cols` of `batch` at `rows`, in that order.
fn select(batch: &RecordBatch, cols: &[&str], rows: &[usize]) -> RecordBatch {
    let idx = UInt32Array::from(rows.iter().map(|&r| r as u32).collect::<Vec<_>>());
    let schema = batch.schema();
    let mut fields = Vec::new();
    let mut arrays = Vec::new();
    for c in cols {
        let i = schema.index_of(c).unwrap();
        fields.push(ArrowField::new(
            *c,
            schema.field(i).data_type().clone(),
            true,
        ));
        arrays.push(take(batch.column(i).as_ref(), &idx, None).unwrap());
    }
    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), arrays).unwrap()
}

fn assert_same(got: &RecordBatch, want: &RecordBatch, what: &str) {
    assert_eq!(
        got.num_columns(),
        want.num_columns(),
        "{what}: column count"
    );
    assert_eq!(got.num_rows(), want.num_rows(), "{what}: row count");
    for c in 0..want.num_columns() {
        let name = want.schema().field(c).name().clone();
        assert_eq!(
            got.column(c).data_type(),
            want.column(c).data_type(),
            "{what}: type of column {name}"
        );
        assert_eq!(
            got.column(c).as_ref(),
            want.column(c).as_ref(),
            "{what}: values of column {name}"
        );
    }
}

fn engine_with(batch: RecordBatch) -> Engine {
    let mut e = Engine::new().expect("CUDA context");
    e.register_table("t", batch).unwrap();
    let u = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("uk", ArrowDataType::Int64, true),
            ArrowField::new("v", ArrowDataType::Int32, true),
        ])),
        vec![
            Arc::new(Int64Array::from(
                (0..N as i64).step_by(3).collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                (0..N as i32).step_by(3).map(|x| x * 2).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    e.register_table("u", u).unwrap();
    e
}

fn run(e: &mut Engine, policy: TierPolicy, q: &str) -> RecordBatch {
    e.set_tier_policy(policy);
    e.sql(q)
        .unwrap_or_else(|err| panic!("{policy:?}: {q}: {err}"))
        .into_record_batch()
}

fn rows(pred: impl Fn(usize) -> bool) -> Vec<usize> {
    (0..N).filter(|&i| pred(i)).collect()
}

const OPAQUE: &[&str] = &["iv", "dec", "nul", "l", "st", "m"];
const ALL: &[&str] = &[
    "k", "tm", "ts32", "d", "b", "lb", "fb", "lu", "iv", "dec", "nul", "l", "st", "m",
];

#[test]
#[ignore = "gpu:e2e — extended types pass through projections and filters"]
fn extended_columns_round_trip_through_scans_and_filters() {
    let input = fixture(0, N);
    let want = normalized(&input);
    let mut e = engine_with(input);
    for p in POLICIES {
        let all = run(&mut e, p, "SELECT * FROM t");
        assert_same(&all, &want, &format!("{p:?} SELECT *"));
        let got = run(
            &mut e,
            p,
            "SELECT k, iv, dec, nul, l, st, m, b, fb, tm, d FROM t WHERE k > 10 AND k < 40",
        );
        let cols = [
            "k", "iv", "dec", "nul", "l", "st", "m", "b", "fb", "tm", "d",
        ];
        assert_same(
            &got,
            &select(&want, &cols, &rows(|i| i > 10 && i < 40)),
            &format!("{p:?} filter"),
        );
        let host_before = metrics().counter(Counter::HostTierQueries).get();
        let got = run(&mut e, p, "SELECT l, k, st FROM t ORDER BY k DESC LIMIT 5");
        if p == TierPolicy::CostBased {
            assert!(
                metrics().counter(Counter::HostTierQueries).get() > host_before,
                "the cost model should run the 100-row sort on the host tier"
            );
        }
        let r: Vec<usize> = (N - 5..N).rev().collect();
        assert_same(
            &got,
            &select(&want, &["l", "k", "st"], &r),
            &format!("{p:?} order"),
        );
    }
}

#[test]
#[ignore = "gpu:e2e — extended types pass through UNION ALL and joins"]
fn extended_columns_pass_through_union_all_and_joins() {
    let want = normalized(&fixture(0, N));
    let mut e = engine_with(fixture(0, N));
    for p in POLICIES {
        let got = run(
            &mut e,
            p,
            "SELECT k, l, m FROM (SELECT k, l, m FROM t WHERE k < 3 \
             UNION ALL SELECT k, l, m FROM t WHERE k > 95) x ORDER BY k",
        );
        let r: Vec<usize> = rows(|i| !(3..=95).contains(&i));
        assert_same(
            &got,
            &select(&want, &["k", "l", "m"], &r),
            &format!("{p:?} union"),
        );
        let got = run(
            &mut e,
            p,
            "SELECT t.k, t.st, t.dec, u.v FROM t JOIN u ON t.k = u.uk WHERE t.k < 50 ORDER BY k",
        );
        let r = rows(|i| i % 3 == 0 && i < 50);
        let part = select(&want, &["k", "st", "dec"], &r);
        assert_eq!(got.num_rows(), r.len(), "{p:?} join rows");
        for c in 0..3 {
            assert_eq!(
                got.column(c).as_ref(),
                part.column(c).as_ref(),
                "{p:?} join col {c}"
            );
        }
        let v = Int32Array::from(r.iter().map(|&i| i as i32 * 2).collect::<Vec<_>>());
        assert_eq!(got.column(3).as_ref(), &v as &dyn Array, "{p:?} join v");
    }
}

#[test]
#[ignore = "gpu:e2e — IS NULL and COUNT on every extended type"]
fn extended_columns_support_is_null_and_count() {
    let want = normalized(&fixture(0, N));
    let mut e = engine_with(fixture(0, N));
    for p in POLICIES {
        for c in ALL {
            let col = want.column_by_name(c).unwrap();
            let nulls = col.logical_nulls().map_or(0, |n| n.null_count()) as i64;
            let got = run(&mut e, p, &format!("SELECT COUNT({c}) AS n FROM t"));
            let n = got.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(n.value(0), N as i64 - nulls, "{p:?} COUNT({c})");
            let got = run(
                &mut e,
                p,
                &format!("SELECT k FROM t WHERE {c} IS NULL ORDER BY k"),
            );
            assert_eq!(got.num_rows() as i64, nulls, "{p:?} {c} IS NULL");
            let got = run(&mut e, p, &format!("SELECT k FROM t WHERE {c} IS NOT NULL"));
            assert_eq!(
                got.num_rows() as i64,
                N as i64 - nulls,
                "{p:?} {c} IS NOT NULL"
            );
        }
    }
}

#[test]
#[ignore = "gpu:e2e — TIME semantics"]
fn time_columns_compare_order_group_and_aggregate() {
    let want = normalized(&fixture(0, N));
    let mut e = engine_with(fixture(0, N));
    for p in POLICIES {
        let got = run(
            &mut e,
            p,
            "SELECT k FROM t WHERE tm > TIME '00:50:00' ORDER BY k",
        );
        let r = rows(|i| tm_valid(i) && tm_us(i) > 3_000_000_000);
        assert_same(
            &got,
            &select(&want, &["k"], &r),
            &format!("{p:?} tm > literal"),
        );
        let got = run(
            &mut e,
            p,
            "SELECT k FROM t WHERE ts32 <= TIME '00:20:20.5' ORDER BY k",
        );
        let r = rows(|i| ts32_valid(i) && ts32_ms(i) <= 1_220_500);
        assert_same(
            &got,
            &select(&want, &["k"], &r),
            &format!("{p:?} ts32 <= literal"),
        );
        // Cross-unit comparison: Time64(us) against Time32(ms).
        let got = run(&mut e, p, "SELECT k FROM t WHERE tm = ts32 ORDER BY k");
        let r = rows(|i| tm_valid(i) && ts32_valid(i) && i % 2 == 0);
        assert_same(
            &got,
            &select(&want, &["k"], &r),
            &format!("{p:?} tm = ts32"),
        );
        let got = run(
            &mut e,
            p,
            "SELECT MIN(tm) AS lo, MAX(tm) AS hi, COUNT(tm) AS c FROM t",
        );
        let valid: Vec<i64> = (0..N).filter(|&i| tm_valid(i)).map(tm_us).collect();
        let lo = got
            .column(0)
            .as_any()
            .downcast_ref::<Time64MicrosecondArray>()
            .unwrap();
        let hi = got
            .column(1)
            .as_any()
            .downcast_ref::<Time64MicrosecondArray>()
            .unwrap();
        assert_eq!(lo.value(0), *valid.iter().min().unwrap(), "{p:?} MIN(tm)");
        assert_eq!(hi.value(0), *valid.iter().max().unwrap(), "{p:?} MAX(tm)");
        let got = run(
            &mut e,
            p,
            "SELECT tm, k FROM t ORDER BY tm DESC NULLS FIRST, k LIMIT 10",
        );
        let mut r: Vec<usize> = rows(tm_valid);
        r.sort_by_key(|&i| std::cmp::Reverse(tm_us(i)));
        let mut nulls_first: Vec<usize> = rows(|i| !tm_valid(i));
        nulls_first.extend(r);
        nulls_first.truncate(10);
        assert_same(
            &got,
            &select(&want, &["tm", "k"], &nulls_first),
            &format!("{p:?} ORDER BY tm"),
        );
        let got = run(
            &mut e,
            p,
            "SELECT ts32, COUNT(*) AS c FROM t WHERE k < 30 GROUP BY ts32 ORDER BY ts32",
        );
        assert_eq!(
            got.column(0).data_type(),
            &ArrowDataType::Time32(TimeUnit::Millisecond)
        );
        // 30 distinct non-null times plus one NULL group.
        let r = rows(|i| i < 30 && ts32_valid(i)).len() + 1;
        assert_eq!(got.num_rows(), r, "{p:?} GROUP BY ts32");
        let got = run(&mut e, p, "SELECT MAX(d) AS m, MIN(d) AS n FROM t");
        let d = got
            .column(0)
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .unwrap();
        assert_eq!(d.value(0), 99 * 1_000_000 - 30_000_000, "{p:?} MAX(d)");
        let got = run(
            &mut e,
            p,
            "SELECT k FROM t WHERE d > INTERVAL '1 minute' ORDER BY k",
        );
        let r = rows(|i| d_valid(i) && i as i64 * 1_000_000 - 30_000_000 > 60_000_000);
        assert_same(
            &got,
            &select(&want, &["k"], &r),
            &format!("{p:?} d > INTERVAL"),
        );
    }
}

#[test]
#[ignore = "gpu:e2e — BINARY semantics"]
fn binary_columns_compare_group_and_order_bytewise() {
    let want = normalized(&fixture(0, N));
    let mut e = engine_with(fixture(0, N));
    for p in POLICIES {
        let got = run(&mut e, p, "SELECT k FROM t WHERE b = X'02FF01' ORDER BY k");
        let r = rows(|i| b_valid(i) && b_bytes(i) == [2, 0xff, 1]);
        assert!(!r.is_empty());
        assert_same(
            &got,
            &select(&want, &["k"], &r),
            &format!("{p:?} b = literal"),
        );
        let got = run(&mut e, p, "SELECT k FROM t WHERE b <> lb ORDER BY k");
        assert_eq!(got.num_rows(), 0, "{p:?} b <> lb");
        let got = run(&mut e, p, "SELECT k FROM t WHERE b = lb ORDER BY k");
        assert_same(
            &got,
            &select(&want, &["k"], &rows(b_valid)),
            &format!("{p:?} b = lb"),
        );
        let got = run(
            &mut e,
            p,
            "SELECT b, COUNT(*) AS c FROM t GROUP BY b ORDER BY b NULLS FIRST",
        );
        let mut keys: Vec<Vec<u8>> = rows(b_valid).into_iter().map(b_bytes).collect();
        keys.sort();
        keys.dedup();
        assert_eq!(got.num_rows(), keys.len() + 1, "{p:?} GROUP BY b");
        let gb = got
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        assert!(gb.is_null(0));
        for (j, key) in keys.iter().enumerate() {
            assert_eq!(gb.value(j + 1), key.as_slice(), "{p:?} GROUP BY b key {j}");
        }
        let got = run(
            &mut e,
            p,
            "SELECT fb, k FROM t WHERE fb IS NOT NULL ORDER BY fb, k LIMIT 8",
        );
        let mut r = rows(|i| i % 9 != 0);
        r.sort_by_key(|&i| (i * 7 % 256, i));
        r.truncate(8);
        assert_same(
            &got,
            &select(&want, &["fb", "k"], &r),
            &format!("{p:?} ORDER BY fb"),
        );
    }
}

#[test]
#[ignore = "gpu:e2e — multi-batch appends keep passthrough rows aligned"]
fn register_batch_appends_and_replace_table() {
    let mut e = Engine::new().expect("CUDA context");
    e.register_batch("t", fixture(0, 60)).unwrap();
    e.register_batch("t", fixture(60, 80)).unwrap();
    e.register_batch("t", fixture(80, N)).unwrap();
    let want = normalized(&concat_batches(&fixture(0, 1).schema(), &[fixture(0, N)]).unwrap());
    for p in POLICIES {
        let got = run(&mut e, p, "SELECT * FROM t ORDER BY k");
        assert_same(&got, &want, &format!("{p:?} appended SELECT *"));
        let got = run(
            &mut e,
            p,
            "SELECT k, l, iv, b FROM t WHERE k >= 55 AND k < 85 ORDER BY k",
        );
        let r = rows(|i| (55..85).contains(&i));
        assert_same(
            &got,
            &select(&want, &["k", "l", "iv", "b"], &r),
            &format!("{p:?} span"),
        );
    }
    e.replace_table("t", fixture(10, 20)).unwrap();
    let want = normalized(&fixture(10, 20));
    for p in POLICIES {
        let got = run(&mut e, p, "SELECT * FROM t ORDER BY k");
        assert_same(&got, &want, &format!("{p:?} replaced"));
    }
    let stream = vec![Ok(fixture(0, 50)), Ok(fixture(50, N))];
    drop(e);
    let mut e2 = Engine::new().expect("CUDA context");
    let plan_schema = Schema::new(
        fixture(0, 1)
            .schema()
            .fields()
            .iter()
            .map(|f| Field::new(f.name(), DataType::from_arrow(f.data_type()).unwrap(), true))
            .collect(),
    );
    e2.register_table_stream("s", plan_schema, stream).unwrap();
    let want = normalized(&fixture(0, N));
    for p in POLICIES {
        let got = run(
            &mut e2,
            p,
            "SELECT * FROM s WHERE k >= 40 AND k < 60 ORDER BY k",
        );
        assert_same(
            &got,
            &select(&want, ALL, &rows(|i| (40..60).contains(&i))),
            &format!("{p:?} stream"),
        );
    }
}

#[test]
#[ignore = "gpu:e2e — TIMESTAMP - TIMESTAMP is a Duration"]
fn timestamp_difference_is_a_duration() {
    let mut e = Engine::new().expect("CUDA context");
    let a: Vec<i64> = (0..N as i64)
        .map(|i| 1_700_000_000_000_000 + i * 3_600_000_000)
        .collect();
    let b: Vec<i64> = (0..N as i64)
        .map(|i| 1_700_000_000_000_000 + i * 60_000_000)
        .collect();
    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int64, true),
            ArrowField::new(
                "a",
                ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
                true,
            ),
            ArrowField::new(
                "b",
                ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
                true,
            ),
        ])),
        vec![
            Arc::new(Int64Array::from((0..N as i64).collect::<Vec<_>>())),
            Arc::new(TimestampMicrosecondArray::from(a.clone())),
            Arc::new(TimestampMicrosecondArray::from(
                (0..N)
                    .map(|i| (i % 10 != 4).then_some(b[i]))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    e.register_table("w", batch).unwrap();
    for p in POLICIES {
        let got = run(&mut e, p, "SELECT a - b AS dd FROM w ORDER BY k");
        let want = DurationMicrosecondArray::from(
            (0..N)
                .map(|i| (i % 10 != 4).then_some(a[i] - b[i]))
                .collect::<Vec<_>>(),
        );
        assert_eq!(got.column(0).as_ref(), &want as &dyn Array, "{p:?} a - b");
        let got = run(
            &mut e,
            p,
            "SELECT k FROM w WHERE a - b > INTERVAL '2 hours' ORDER BY k",
        );
        let r: Vec<i64> = (0..N)
            .filter(|&i| i % 10 != 4 && a[i] - b[i] > 7_200_000_000)
            .map(|i| i as i64)
            .collect();
        assert_eq!(
            got.column(0).as_ref(),
            &Int64Array::from(r) as &dyn Array,
            "{p:?} filter"
        );
        let got = run(&mut e, p, "SELECT b + (a - b) AS back FROM w ORDER BY k");
        let want = TimestampMicrosecondArray::from(
            (0..N)
                .map(|i| (i % 10 != 4).then_some(a[i]))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            got.column(0).as_ref(),
            &want as &dyn Array,
            "{p:?} b + (a - b)"
        );
        let got = run(
            &mut e,
            p,
            "SELECT a - INTERVAL '1 hour' AS h FROM w ORDER BY k LIMIT 3",
        );
        let want = TimestampMicrosecondArray::from(
            (0..3).map(|i| a[i] - 3_600_000_000).collect::<Vec<_>>(),
        );
        assert_eq!(
            got.column(0).as_ref(),
            &want as &dyn Array,
            "{p:?} a - INTERVAL"
        );
    }
}

#[test]
#[ignore = "gpu:e2e — unsupported operations fail at plan time naming the type"]
fn unsupported_operations_fail_with_a_clear_error() {
    let mut e = engine_with(fixture(0, N));
    let cases: &[(&str, &str)] = &[
        ("SELECT l + 1 FROM t", "List"),
        ("SELECT k FROM t ORDER BY st", "Struct"),
        ("SELECT m, COUNT(*) FROM t GROUP BY m", "Map"),
        ("SELECT DISTINCT iv FROM t", "Interval"),
        ("SELECT MIN(dec) FROM t", "Decimal256"),
        ("SELECT k FROM t WHERE l = l", "List"),
        ("SELECT UPPER(b) FROM t", "Binary"),
        ("SELECT SUM(tm) FROM t", "Time64"),
        ("SELECT tm + tm FROM t", "Time64"),
        ("SELECT k FROM t WHERE b LIKE 'a%'", "Binary"),
        ("SELECT t.k FROM t JOIN t AS t2 ON t.nul = t2.nul", "Null"),
        ("SELECT l FROM t UNION SELECT l FROM t", "List"),
    ];
    for p in POLICIES {
        e.set_tier_policy(p);
        for (q, ty) in cases {
            let err = match e.sql(q) {
                Ok(_) => panic!("{p:?}: {q} should fail"),
                Err(err) => err.to_string(),
            };
            assert!(
                err.contains(ty),
                "{p:?}: {q}: error should name {ty}: {err}"
            );
            let explain = e.explain_sql(q);
            assert!(explain.is_err(), "{p:?}: explain_sql({q}) should fail too");
        }
    }
    // A query that never touches the extended columns is unaffected.
    let got = run(
        &mut e,
        TierPolicy::GpuFirst,
        "SELECT SUM(k) AS s FROM t WHERE k < 10",
    );
    let s = got.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(s.value(0), 45);
    let _ = OPAQUE;
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "bolt-extended-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The fixture without the columns the Parquet writer cannot store
/// (`Interval(MonthDayNano)`, `Duration`).
fn parquet_fixture(lo: usize, hi: usize) -> RecordBatch {
    let b = normalized(&fixture(lo, hi));
    let keep: Vec<usize> = (0..b.num_columns())
        .filter(|&i| !matches!(b.schema().field(i).name().as_str(), "iv" | "d"))
        .collect();
    b.project(&keep).unwrap()
}

#[test]
#[ignore = "gpu:e2e — extended columns in Parquet storage and catalog tables"]
fn parquet_and_catalog_tables_store_extended_columns() {
    use craton_bolt::storage::{write_parquet, ParquetWriteOptions};
    let dir = temp_dir("parquet");
    let path = dir.join("t.parquet");
    let opts = ParquetWriteOptions {
        row_group_rows: 25,
        ..Default::default()
    };
    write_parquet(&path, &[parquet_fixture(0, N)], &opts).unwrap();
    let want = parquet_fixture(0, N);
    let cols: Vec<String> = want
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let cols: Vec<&str> = cols.iter().map(String::as_str).collect();
    {
        let mut e = Engine::new().expect("CUDA context");
        e.register_parquet("p", &path).unwrap();
        for p in POLICIES {
            let got = run(&mut e, p, "SELECT * FROM p ORDER BY k");
            assert_same(&got, &want, &format!("{p:?} parquet SELECT *"));
            let got = run(
                &mut e,
                p,
                "SELECT k, l, b FROM p WHERE k >= 60 AND k < 70 ORDER BY k",
            );
            let r = rows(|i| (60..70).contains(&i));
            assert_same(
                &got,
                &select(&want, &["k", "l", "b"], &r),
                &format!("{p:?} pruned"),
            );
            let got = run(&mut e, p, "SELECT k FROM p WHERE b = X'02FF01' ORDER BY k");
            let r = rows(|i| b_valid(i) && b_bytes(i) == [2, 0xff, 1]);
            assert_same(
                &got,
                &select(&want, &["k"], &r),
                &format!("{p:?} parquet b = literal"),
            );
            let got = run(
                &mut e,
                p,
                "SELECT k FROM p WHERE tm = TIME '00:00:00' ORDER BY k",
            );
            assert_same(
                &got,
                &select(&want, &["k"], &[0]),
                &format!("{p:?} parquet tm = literal"),
            );
        }
    }
    let db = dir.join("db");
    let mut e = Engine::builder()
        .database(&db)
        .build()
        .expect("CUDA context");
    e.register_table("t", parquet_fixture(0, N)).unwrap();
    e.execute_sql("CREATE TABLE c AS SELECT * FROM t").unwrap();
    for p in POLICIES {
        let got = run(&mut e, p, "SELECT * FROM c ORDER BY k");
        assert_same(&got, &want, &format!("{p:?} catalog SELECT *"));
        let got = run(&mut e, p, "SELECT st, k FROM c WHERE k < 5 ORDER BY k");
        assert_same(
            &got,
            &select(&want, &["st", "k"], &rows(|i| i < 5)),
            &format!("{p:?} catalog filter"),
        );
    }
    let err = e
        .execute_sql("DELETE FROM c WHERE k < 3")
        .unwrap_err()
        .to_string();
    assert!(err.contains("extended type"), "{err}");
    let _ = cols;
    drop(e);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[ignore = "gpu:e2e — DISTINCT, set operations, IN / BETWEEN, windows and SUM(duration)"]
fn extended_columns_in_distinct_windows_and_set_operations() {
    let want = normalized(&fixture(0, N));
    let mut e = engine_with(fixture(0, N));
    let mut distinct_b: Vec<Vec<u8>> = rows(b_valid).into_iter().map(b_bytes).collect();
    distinct_b.sort();
    distinct_b.dedup();
    for p in POLICIES {
        let got = run(&mut e, p, "SELECT DISTINCT b FROM t");
        assert_eq!(got.num_rows(), distinct_b.len() + 1, "{p:?} DISTINCT b");
        let got = run(
            &mut e,
            p,
            "SELECT tm FROM t INTERSECT SELECT tm FROM t WHERE k < 50",
        );
        let r = rows(|i| i < 50 && tm_valid(i)).len() + 1;
        assert_eq!(got.num_rows(), r, "{p:?} INTERSECT tm");
        let got = run(
            &mut e,
            p,
            "SELECT k FROM t WHERE tm BETWEEN TIME '00:00:00' AND TIME '00:10:00' ORDER BY k",
        );
        let r = rows(|i| tm_valid(i) && tm_us(i) <= 600_000_000);
        assert_same(&got, &select(&want, &["k"], &r), &format!("{p:?} BETWEEN"));
        let got = run(
            &mut e,
            p,
            "SELECT k FROM t WHERE b IN (X'00FF00', X'01FF01') ORDER BY k",
        );
        let r =
            rows(|i| b_valid(i) && matches!(b_bytes(i).as_slice(), [0, 0xff, 0] | [1, 0xff, 1]));
        assert_same(&got, &select(&want, &["k"], &r), &format!("{p:?} IN"));
        let got = run(
            &mut e,
            p,
            "SELECT k, l, ROW_NUMBER() OVER (PARTITION BY b ORDER BY tm) AS r FROM t              ORDER BY k LIMIT 7",
        );
        let part = got.project(&[0, 1]).unwrap();
        assert_same(
            &part,
            &select(&want, &["k", "l"], &(0..7).collect::<Vec<_>>()),
            &format!("{p:?} window"),
        );
        let got = run(&mut e, p, "SELECT SUM(d) AS s FROM t");
        let s: i64 = rows(d_valid)
            .iter()
            .map(|&i| i as i64 * 1_000_000 - 30_000_000)
            .sum();
        let d = got
            .column(0)
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .unwrap();
        assert_eq!(d.value(0), s, "{p:?} SUM(d)");
    }
    // Query shapes that combine component results on the host refuse an
    // extended output column.
    for q in [
        "SELECT DISTINCT ON (k) k, l FROM t ORDER BY k",
        "SELECT k, COUNT(DISTINCT b) AS c FROM t GROUP BY k",
    ] {
        let err = e.sql(q).map(|_| ()).unwrap_err().to_string();
        assert!(err.contains("List") || err.contains("Binary"), "{q}: {err}");
    }
}

/// `k` (non-nullable or not), `g = k % 4`, `s` Utf8 (NULL every 4th row),
/// `b` Binary (NULL every 3rd row).
fn count_table(k_nullable: bool) -> RecordBatch {
    let n = N as i64;
    RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int64, k_nullable),
            ArrowField::new("g", ArrowDataType::Int64, true),
            ArrowField::new("s", ArrowDataType::Utf8, true),
            ArrowField::new("b", ArrowDataType::Binary, true),
        ])),
        vec![
            Arc::new(Int64Array::from((0..n).collect::<Vec<_>>())),
            Arc::new(Int64Array::from((0..n).map(|k| k % 4).collect::<Vec<_>>())),
            Arc::new(StringArray::from(
                (0..n)
                    .map(|k| (k % 4 != 1).then(|| format!("s{}", k % 5)))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(BinaryArray::from_iter(
                (0..n).map(|k| (k % 3 != 2).then(|| vec![k as u8, 0xff])),
            )),
        ],
    )
    .unwrap()
}

fn i64_col(b: &RecordBatch, c: usize) -> Vec<i64> {
    let a = b.column(c).as_any().downcast_ref::<Int64Array>().unwrap();
    (0..a.len()).map(|i| a.value(i)).collect()
}

#[test]
#[ignore = "gpu:e2e — COUNT(string / binary column) next to COUNT(*)"]
fn count_of_a_string_or_binary_column_next_to_count_star() {
    for k_nullable in [false, true] {
        let mut e = Engine::new().expect("CUDA context");
        e.register_table("t", count_table(k_nullable)).unwrap();
        let s_ok = |k: i64| k % 4 != 1;
        let b_ok = |k: i64| k % 3 != 2;
        for p in POLICIES {
            let what = format!("{p:?} k_nullable={k_nullable}");
            for (filter, keep) in [
                ("", (|_| true) as fn(i64) -> bool),
                (" WHERE k >= 10", |k| k >= 10),
            ] {
                let rows: Vec<i64> = (0..N as i64).filter(|&k| keep(k)).collect();
                let count = |f: &dyn Fn(i64) -> bool| rows.iter().filter(|&&k| f(k)).count() as i64;
                let q = format!(
                    "SELECT COUNT(s) AS a, COUNT(*) AS c, COUNT(b) AS d, SUM(k) AS e FROM t{filter}"
                );
                let got = run(&mut e, p, &q);
                let want = vec![count(&s_ok), rows.len() as i64, count(&b_ok)];
                assert_eq!(
                    (0..3).map(|c| i64_col(&got, c)[0]).collect::<Vec<_>>(),
                    want,
                    "{what}: {q}"
                );
                assert_eq!(i64_col(&got, 3)[0], rows.iter().sum::<i64>(), "{what}: {q}");
                let q = format!("SELECT COUNT(s) AS a FROM t{filter}");
                assert_eq!(
                    i64_col(&run(&mut e, p, &q), 0)[0],
                    count(&s_ok),
                    "{what}: {q}"
                );
                let q = format!(
                    "SELECT g, COUNT(s) AS a, COUNT(*) AS c, COUNT(b) AS d FROM t{filter}                      GROUP BY g ORDER BY g"
                );
                let got = run(&mut e, p, &q);
                assert_eq!(i64_col(&got, 0), vec![0, 1, 2, 3], "{what}: {q}");
                for (c, f) in [
                    (1, &s_ok as &dyn Fn(i64) -> bool),
                    (2, &|_| true),
                    (3, &b_ok),
                ] {
                    let want: Vec<i64> = (0..4)
                        .map(|g| rows.iter().filter(|&&k| k % 4 == g && f(k)).count() as i64)
                        .collect();
                    assert_eq!(i64_col(&got, c), want, "{what}: {q} column {c}");
                }
            }
        }
    }
}
