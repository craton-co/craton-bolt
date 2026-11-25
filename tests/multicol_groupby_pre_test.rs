// SPDX-License-Identifier: AGPL-3.0-only

//! Regression and end-to-end CUDA GPU tests for multi-column `GROUP BY` with
//! pre-aggregation (`agg pre+groupby`), wider key/aggregate dtypes, sentinel
//! collisions, NULL propagation, and integer `SUM` overflow.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
    Int32Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema, TimeUnit,
};
use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Field, MemTableProvider, PhysicalPlan, Schema,
};
use craton_bolt::BoltError;

// ---------------------------------------------------------------------------
// Host-always planner / lowering checks (run under both `cuda-stub` and CUDA).
// ---------------------------------------------------------------------------

#[test]
fn q2_region_category_rev_lowers_to_aggregate_with_pre_and_two_group_keys() {
    let mut provider = MemTableProvider::new();
    provider.register(
        "sales",
        Schema::new(vec![
            Field::new("region_id", DataType::Int32, false),
            Field::new("category_id", DataType::Int32, false),
            Field::new("order_day", DataType::Int32, false),
            Field::new("quantity", DataType::Int32, false),
            Field::new("price", DataType::Float64, false),
            Field::new("discount", DataType::Float64, false),
        ]),
    );

    let sql = "SELECT region_id, category_id, \
                      SUM(quantity * price * (1.0 - discount / 100.0)) AS gross_revenue, \
                      AVG(price) AS avg_price, \
                      COUNT(*) AS order_lines \
               FROM sales \
               WHERE order_day >= 365 AND order_day < 1460 \
               GROUP BY region_id, category_id";
    let logical = parse_sql(sql, &provider).expect("parse + plan");
    let phys = lower_physical(&logical).expect("lower");

    // Top node may be a rename Project over Aggregate; unwrap if present.
    let agg_node = match &phys {
        PhysicalPlan::Project { input, .. } => input.as_ref(),
        other => other,
    };
    match agg_node {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => {
            assert!(pre.is_some(), "q2 must lower with pre = Some(_)");
            assert_eq!(aggregate.group_by.len(), 2, "must carry 2 GROUP BY keys");
            assert_eq!(aggregate.aggregates.len(), 3);
        }
        other => panic!("expected PhysicalPlan::Aggregate, got {other:?}"),
    }
}

#[test]
fn utf8_key_with_computed_sum_lowers_to_aggregate_over_input() {
    let mut provider = MemTableProvider::new();
    provider.register(
        "sales",
        Schema::new(vec![
            Field::new("region", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
            Field::new("qty", DataType::Int32, false),
        ]),
    );

    let logical = parse_sql(
        "SELECT region, SUM(price * qty) AS rev FROM sales GROUP BY region",
        &provider,
    )
    .expect("parse + plan");
    let phys = lower_physical(&logical).expect("lower");
    let inner = match &phys {
        PhysicalPlan::Project { input, .. } => input.as_ref(),
        other => other,
    };
    assert!(
        matches!(inner, PhysicalPlan::AggregateOverInput { .. }),
        "Utf8 key alongside computed SUM must materialise feeds rather than put Utf8 in pre.outputs: {inner:?}"
    );
}

// ---------------------------------------------------------------------------
// End-to-end CUDA GPU tests (`--ignored --test-threads=1`).
// ---------------------------------------------------------------------------

fn build_q2_sales_batch(n: usize) -> RecordBatch {
    let mut region_id = Vec::with_capacity(n);
    let mut category_id = Vec::with_capacity(n);
    let mut order_day = Vec::with_capacity(n);
    let mut quantity = Vec::with_capacity(n);
    let mut price = Vec::with_capacity(n);
    let mut discount = Vec::with_capacity(n);

    for i in 0..n {
        region_id.push((i % 5) as i32);
        category_id.push(((i / 3) % 7) as i32);
        order_day.push(((i * 37) % 1800) as i32);
        quantity.push(((i % 9) + 1) as i32);
        price.push(10.0 + ((i % 23) as f64) * 1.25);
        discount.push(((i * 5) % 30) as f64);
    }

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("region_id", ArrowDataType::Int32, false),
        ArrowField::new("category_id", ArrowDataType::Int32, false),
        ArrowField::new("order_day", ArrowDataType::Int32, false),
        ArrowField::new("quantity", ArrowDataType::Int32, false),
        ArrowField::new("price", ArrowDataType::Float64, false),
        ArrowField::new("discount", ArrowDataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(region_id)) as ArrayRef,
            Arc::new(Int32Array::from(category_id)) as ArrayRef,
            Arc::new(Int32Array::from(order_day)) as ArrayRef,
            Arc::new(Int32Array::from(quantity)) as ArrayRef,
            Arc::new(Float64Array::from(price)) as ArrayRef,
            Arc::new(Float64Array::from(discount)) as ArrayRef,
        ],
    )
    .expect("q2 sales batch")
}

#[test]
#[ignore = "gpu:e2e"]
fn e2e_q2_region_category_rev_exact_benchmark_query() {
    use craton_bolt::Engine;

    let batch = build_q2_sales_batch(4_096);
    let r_col = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let c_col = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let d_col = batch
        .column(2)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let q_col = batch
        .column(3)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let p_col = batch
        .column(4)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let disc_col = batch
        .column(5)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    let mut expected: HashMap<(i32, i32), (f64, f64, i64)> = HashMap::new();
    for i in 0..batch.num_rows() {
        let day = d_col.value(i);
        if (365..1460).contains(&day) {
            let key = (r_col.value(i), c_col.value(i));
            let rev = (q_col.value(i) as f64) * p_col.value(i) * (1.0 - disc_col.value(i) / 100.0);
            let entry = expected.entry(key).or_insert((0.0, 0.0, 0));
            entry.0 += rev;
            entry.1 += p_col.value(i);
            entry.2 += 1;
        }
    }
    let mut want_keys: Vec<(i32, i32)> = expected.keys().copied().collect();
    want_keys.sort_unstable();

    let mut engine = Engine::new().expect("ctx");
    engine.register_table("sales", batch).unwrap();

    let h = engine
        .sql(
            "SELECT region_id, category_id, \
                    SUM(quantity * price * (1.0 - discount / 100.0)) AS gross_revenue, \
                    AVG(price) AS avg_price, \
                    COUNT(*) AS order_lines \
             FROM sales \
             WHERE order_day >= 365 AND order_day < 1460 \
             GROUP BY region_id, category_id \
             ORDER BY region_id, category_id",
        )
        .expect("q2 execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), want_keys.len());

    let out_r = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let out_c = out.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
    let out_rev = out
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let out_avg = out
        .column(3)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let out_cnt = out.column(4).as_any().downcast_ref::<Int64Array>().unwrap();

    for (row, &key) in want_keys.iter().enumerate() {
        let (sum_rev, sum_price, cnt) = expected[&key];
        assert_eq!((out_r.value(row), out_c.value(row)), key);
        assert!(
            (out_rev.value(row) - sum_rev).abs() < 1e-6,
            "key {key:?}: rev got {}, want {}",
            out_rev.value(row),
            sum_rev
        );
        assert!(
            (out_avg.value(row) - (sum_price / cnt as f64)).abs() < 1e-9,
            "key {key:?}: avg got {}, want {}",
            out_avg.value(row),
            sum_price / cnt as f64
        );
        assert_eq!(out_cnt.value(row), cnt);
    }
}

#[test]
#[ignore = "gpu:e2e"]
fn e2e_multicol_count_star_without_where_and_pg_bolt_int32_float64_key() {
    use craton_bolt::Engine;

    // Covers both:
    // 1. Bare `SELECT a, b, COUNT(*) FROM t GROUP BY a, b` (still hits pre = Some(_))
    // 2. pg-bolt `q2_multi_col`: `(Int32, Float64)` key (96 bits > 64 + float key)
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("region_id", ArrowDataType::Int32, false),
        ArrowField::new("tax", ArrowDataType::Float64, false),
        ArrowField::new("price", ArrowDataType::Float64, false),
        ArrowField::new("active", ArrowDataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 1, 2, 2, 2])) as ArrayRef,
            Arc::new(Float64Array::from(vec![0.05, 0.05, 0.10, 0.05, 0.10, -0.0])) as ArrayRef,
            Arc::new(Float64Array::from(vec![
                100.0, 200.0, 50.0, 80.0, 120.0, 40.0,
            ])) as ArrayRef,
            Arc::new(Int32Array::from(vec![1, 1, 0, 1, 1, 1])) as ArrayRef,
        ],
    )
    .unwrap();

    let mut engine = Engine::new().expect("ctx");
    engine.register_table("sales", batch).unwrap();

    let bare = engine
        .sql("SELECT region_id, active, COUNT(*) AS n FROM sales GROUP BY region_id, active ORDER BY region_id, active")
        .expect("bare multicol COUNT(*)");
    let b_out = bare.record_batch();
    assert_eq!(b_out.num_rows(), 3);
    let cnts = b_out
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(cnts.values(), &[1, 2, 3]);

    let wide_float = engine
        .sql(
            "SELECT region_id, tax, SUM(price * (1.0 - tax)) AS net, COUNT(*) AS n \
             FROM sales WHERE active = 1 GROUP BY region_id, tax ORDER BY region_id, tax",
        )
        .expect("pg-bolt (Int32, Float64) key with pre");
    let wf_out = wide_float.record_batch();
    // Surviving rows for active=1: (1, 0.05)x2, (2, 0.0), (2, 0.05), (2, 0.10) -> 4 groups.
    assert_eq!(wf_out.num_rows(), 4);
    let wf_r = wf_out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let wf_tax = wf_out
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let wf_net = wf_out
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let wf_n = wf_out
        .column(3)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(wf_r.values(), &[1, 2, 2, 2]);
    assert!((wf_tax.value(0) - 0.05).abs() < 1e-12);
    assert!((wf_net.value(0) - 285.0).abs() < 1e-9);
    assert_eq!(wf_n.values(), &[2, 1, 1, 1]);
}

#[test]
#[ignore = "gpu:e2e"]
fn e2e_mixed_keys_three_cols_high_card_and_sentinel_collision() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");

    // 1. Mixed (Int32, Int64), (Int64, Utf8), and 3-col (Int32, Int32, Int32).
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("a", ArrowDataType::Int32, false),
        ArrowField::new("b64", ArrowDataType::Int64, false),
        ArrowField::new("c", ArrowDataType::Int32, false),
        ArrowField::new("label", ArrowDataType::Utf8, false),
        ArrowField::new("v", ArrowDataType::Float64, false),
        ArrowField::new("keep", ArrowDataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            // Include `(i32::MIN, 0)` on `a, c` to exercise packed `i64::MIN` sentinel collision!
            Arc::new(Int32Array::from(vec![i32::MIN, i32::MIN, 1, 1, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![100, 100, 200, 300, 200])) as ArrayRef,
            Arc::new(Int32Array::from(vec![0, 0, 5, 5, 6])) as ArrayRef,
            Arc::new(StringArray::from(vec![
                "east", "east", "west", "west", "east",
            ])) as ArrayRef,
            Arc::new(Float64Array::from(vec![10.0, 15.0, 20.0, 99.0, 30.0])) as ArrayRef,
            Arc::new(Int32Array::from(vec![1, 1, 1, 0, 1])) as ArrayRef,
        ],
    )
    .unwrap();
    engine.register_table("mix", batch).unwrap();

    // Sentinel collision: `(i32::MIN, 0)` packs to `0x8000_0000_0000_0000 == i64::MIN`!
    let sent = engine
        .sql("SELECT a, c, SUM(v * 2.0) AS s, COUNT(*) AS n FROM mix WHERE keep = 1 GROUP BY a, c ORDER BY a, c")
        .expect("sentinel-colliding 2-col packed key with pre");
    let s_out = sent.record_batch();
    assert_eq!(s_out.num_rows(), 3);
    assert_eq!(
        s_out
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values(),
        &[i32::MIN, 1, 2]
    );
    assert_eq!(
        s_out
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values(),
        &[0, 5, 6]
    );
    let s_sums = s_out
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert!((s_sums.value(0) - 50.0).abs() < 1e-9);
    assert!((s_sums.value(1) - 40.0).abs() < 1e-9);
    assert!((s_sums.value(2) - 60.0).abs() < 1e-9);

    // Mixed (Int64, Utf8) with WHERE + computed SUM.
    let i64_utf8 = engine
        .sql("SELECT b64, label, SUM(v * 2.0) AS s, COUNT(*) AS n FROM mix WHERE keep = 1 GROUP BY b64, label ORDER BY b64, label")
        .expect("(Int64, Utf8) with pre");
    let u_out = i64_utf8.record_batch();
    assert_eq!(u_out.num_rows(), 3);
    let u_b64 = u_out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let u_lbl = u_out
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let u_sum = u_out
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(u_b64.values(), &[100, 200, 200]);
    assert_eq!(u_lbl.value(0), "east");
    assert_eq!(u_lbl.value(1), "east");
    assert_eq!(u_lbl.value(2), "west");
    assert!((u_sum.value(0) - 50.0).abs() < 1e-9);
    assert!((u_sum.value(1) - 60.0).abs() < 1e-9);
    assert!((u_sum.value(2) - 40.0).abs() < 1e-9);

    // 3-column GROUP BY with WHERE + computed SUM.
    let three = engine
        .sql("SELECT a, b64, c, SUM(v + 1.0) AS s FROM mix WHERE keep = 1 GROUP BY a, b64, c ORDER BY a, b64, c")
        .expect("3-col GROUP BY with pre");
    assert_eq!(three.record_batch().num_rows(), 3);

    // Empty post-filter result: 0 rows survive -> 0 groups (empty RecordBatch).
    let empty = engine
        .sql("SELECT a, c, SUM(v * 2.0), COUNT(*) FROM mix WHERE keep = 99 GROUP BY a, c")
        .expect("empty post-filter GROUP BY");
    assert_eq!(empty.record_batch().num_rows(), 0);
    assert_eq!(empty.record_batch().num_columns(), 4);

    // High-cardinality 2-key Tier-2 with pre: 12,000 rows over 3,000 (k1, k2) groups.
    let n = 12_000usize;
    let k1: Vec<i32> = (0..n).map(|i| ((i * 7) % 60) as i32).collect();
    let k2: Vec<i32> = (0..n).map(|i| ((i * 13) % 50) as i32).collect();
    let val: Vec<f64> = (0..n).map(|i| (i % 17) as f64 + 0.5).collect();
    let flt: Vec<i32> = (0..n).map(|i| (i % 4) as i32).collect();
    let mut hc_expected: HashMap<(i32, i32), f64> = HashMap::new();
    for i in 0..n {
        if flt[i] > 0 {
            *hc_expected.entry((k1[i], k2[i])).or_insert(0.0) += val[i] * 3.0;
        }
    }
    let hc_batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k1", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
            ArrowField::new("val", ArrowDataType::Float64, false),
            ArrowField::new("flt", ArrowDataType::Int32, false),
        ])),
        vec![
            Arc::new(Int32Array::from(k1)) as ArrayRef,
            Arc::new(Int32Array::from(k2)) as ArrayRef,
            Arc::new(Float64Array::from(val)) as ArrayRef,
            Arc::new(Int32Array::from(flt)) as ArrayRef,
        ],
    )
    .unwrap();
    engine.register_table("hc", hc_batch).unwrap();

    let hc_out = engine
        .sql("SELECT k1, k2, SUM(val * 3.0) FROM hc WHERE flt > 0 GROUP BY k1, k2")
        .expect("high-card 2-key Tier-2 with pre")
        .into_record_batch();
    assert_eq!(hc_out.num_rows(), hc_expected.len());
    let ok1 = hc_out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let ok2 = hc_out
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let osum = hc_out
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for i in 0..hc_out.num_rows() {
        let want = hc_expected[&(ok1.value(i), ok2.value(i))];
        assert!((osum.value(i) - want).abs() < 1e-6);
    }
}

#[test]
#[ignore = "gpu:e2e"]
fn e2e_temporal_bool_decimal_groupby_and_scalar_pre() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");

    let dec_arr = Decimal128Array::from(vec![Some(1050i128), Some(2025), Some(500), Some(3000)])
        .with_precision_and_scale(12, 2)
        .unwrap();
    let ts_arr = TimestampMicrosecondArray::from(vec![
        1_700_000_000_000_000i64,
        1_700_000_000_000_000,
        1_700_086_400_000_000,
        1_700_086_400_000_000,
    ])
    .with_timezone("UTC");

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("region", ArrowDataType::Int32, false),
        ArrowField::new("d", ArrowDataType::Date32, false),
        ArrowField::new(
            "ts",
            ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        ArrowField::new("flag", ArrowDataType::Boolean, false),
        ArrowField::new("amt", ArrowDataType::Decimal128(12, 2), true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 1, 2])) as ArrayRef,
            Arc::new(Date32Array::from(vec![19_723, 19_723, 19_724, 19_724])) as ArrayRef,
            Arc::new(ts_arr) as ArrayRef,
            Arc::new(BooleanArray::from(vec![true, true, false, true])) as ArrayRef,
            Arc::new(dec_arr) as ArrayRef,
        ],
    )
    .unwrap();
    engine.register_table("events", batch).unwrap();

    // (Int32, Date32) packed 2-col key + Decimal128 SUM + Bool WHERE filter!
    let by_date = engine
        .sql("SELECT region, d, SUM(amt * 2), COUNT(*) FROM events WHERE flag = TRUE GROUP BY region, d ORDER BY region, d")
        .expect("(Int32, Date32) + Decimal128 pre");
    let bd = by_date.record_batch();
    assert_eq!(bd.num_rows(), 2);
    let bd_d = bd.column(1).as_any().downcast_ref::<Date32Array>().unwrap();
    let bd_s = bd
        .column(2)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap();
    assert_eq!(bd_d.values(), &[19_723, 19_724]);
    // (1, 19723): (10.50 + 20.25) * 2 = 61.50 -> 6150; (2, 19724): 30.00 * 2 = 60.00 -> 6000.
    assert_eq!(bd_s.value(0), 6150);
    assert_eq!(bd_s.value(1), 6000);

    // Decimal128 GROUP BY key + Timestamp MIN/MAX (preserves UTC tz) with WHERE!
    let by_ts = engine
        .sql("SELECT ts, SUM(amt) FROM events WHERE region >= 1 GROUP BY ts ORDER BY ts")
        .expect("Timestamp key with pre");
    let bt = by_ts.record_batch();
    assert_eq!(bt.num_rows(), 2);
    assert_eq!(
        bt.schema().field(0).data_type(),
        &ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
    );

    // Scalar pre-agg (`agg_with_pre`) over Bool filter + Decimal128 expression + Date32 MIN/MAX!
    let scalar = engine
        .sql("SELECT SUM(amt * 2), MIN(d), MAX(d), COUNT(*) FROM events WHERE flag = TRUE")
        .expect("scalar agg_with_pre over Decimal128 + Date32 + Bool");
    let sc = scalar.record_batch();
    assert_eq!(sc.num_rows(), 1);
    assert_eq!(
        sc.column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap()
            .value(0),
        12150
    );
    assert_eq!(
        sc.column(1)
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap()
            .value(0),
        19_723
    );
    assert_eq!(
        sc.column(2)
            .as_any()
            .downcast_ref::<Date32Array>()
            .unwrap()
            .value(0),
        19_724
    );
    assert_eq!(
        sc.column(3)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        3
    );
}

#[test]
#[ignore = "gpu:e2e"]
fn e2e_null_keys_all_null_group_rollup_and_sum_overflow_with_pre() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");

    // 1. Multi-column SQL-standard NULL keys + all-NULL value group with pre (`null_empty_groups`).
    // Group (1, 10): v = [10.0, NULL] -> sum=20.0, avg=10.0, count(v)=1, count(*)=2
    // Group (1, 20): v = [NULL, NULL] -> sum=NULL, avg=NULL, count(v)=0, count(*)=2
    // Group (1, NULL): v = [25.0, 75.0] -> sum=200.0, avg=50.0, count(v)=2, count(*)=2
    // Group (NULL, 10): v = [5.0] -> sum=10.0, avg=5.0, count(v)=1, count(*)=1
    // Group (NULL, NULL): v = [NULL] -> sum=NULL, avg=NULL, count(v)=0, count(*)=1
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k1", ArrowDataType::Int32, true),
        ArrowField::new("k2", ArrowDataType::Int32, true),
        ArrowField::new("v", ArrowDataType::Float64, true),
        ArrowField::new("keep", ArrowDataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                None,
                None,
            ])) as ArrayRef,
            Arc::new(Int32Array::from(vec![
                Some(10),
                Some(10),
                Some(20),
                Some(20),
                None,
                None,
                Some(10),
                None,
            ])) as ArrayRef,
            Arc::new(Float64Array::from(vec![
                Some(10.0),
                None,
                None,
                None,
                Some(25.0),
                Some(75.0),
                Some(5.0),
                None,
            ])) as ArrayRef,
            Arc::new(Int32Array::from(vec![1; 8])) as ArrayRef,
        ],
    )
    .unwrap();
    engine.register_table("nulls", batch).unwrap();

    let out = engine
        .sql(
            "SELECT k1, k2, SUM(v * 2.0) AS s, AVG(v) AS a, COUNT(v) AS cv, COUNT(*) AS cstar \
             FROM nulls WHERE keep = 1 GROUP BY k1, k2 ORDER BY k1 NULLS LAST, k2 NULLS LAST",
        )
        .expect("nulls multicol with pre")
        .into_record_batch();
    assert_eq!(out.num_rows(), 5);
    let k1_col = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let k2_col = out.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
    let s_col = out
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let a_col = out
        .column(3)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let cv_col = out.column(4).as_any().downcast_ref::<Int64Array>().unwrap();
    let cs_col = out.column(5).as_any().downcast_ref::<Int64Array>().unwrap();
    let keys: Vec<(Option<i32>, Option<i32>)> = (0..5)
        .map(|r| {
            (
                (!k1_col.is_null(r)).then(|| k1_col.value(r)),
                (!k2_col.is_null(r)).then(|| k2_col.value(r)),
            )
        })
        .collect();
    assert_eq!(
        keys,
        vec![
            (Some(1), Some(10)),
            (Some(1), Some(20)),
            (Some(1), None),
            (None, Some(10)),
            (None, None),
        ]
    );
    assert!(!s_col.is_null(0) && (s_col.value(0) - 20.0).abs() < 1e-9);
    assert!(
        s_col.is_null(1),
        "all-NULL group must surface SQL NULL for SUM"
    );
    assert!(!s_col.is_null(2) && (s_col.value(2) - 200.0).abs() < 1e-9);
    assert!(!s_col.is_null(3) && (s_col.value(3) - 10.0).abs() < 1e-9);
    assert!(s_col.is_null(4));
    assert!(!a_col.is_null(0) && (a_col.value(0) - 10.0).abs() < 1e-9);
    assert!(
        a_col.is_null(1),
        "all-NULL group must surface SQL NULL for AVG"
    );
    assert!(!a_col.is_null(2) && (a_col.value(2) - 50.0).abs() < 1e-9);
    assert!(!a_col.is_null(3) && (a_col.value(3) - 5.0).abs() < 1e-9);
    assert!(a_col.is_null(4));
    assert_eq!(cv_col.values(), &[1, 0, 2, 1, 0]);
    assert_eq!(cs_col.values(), &[2, 2, 2, 1, 1]);

    // Sentinel collisions (`i64::MIN` and `i64::MAX`) combined with NULL group keys.
    let sent_batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int64, true),
            ArrowField::new("v", ArrowDataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![
                Some(i64::MIN),
                Some(i64::MAX),
                None,
                None,
                Some(i64::MIN),
            ])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10, 20, 3, 4, 5])) as ArrayRef,
        ],
    )
    .unwrap();
    engine.register_table("sent_null", sent_batch).unwrap();
    let sent_out = engine
        .sql("SELECT k, SUM(v) AS s, COUNT(*) AS n FROM sent_null GROUP BY k ORDER BY k NULLS LAST")
        .expect("sentinel + NULL key groupby")
        .into_record_batch();
    assert_eq!(sent_out.num_rows(), 3);
    let sk = sent_out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let ss = sent_out
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let sn = sent_out
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(sk.value(0), i64::MIN);
    assert_eq!(sk.value(1), i64::MAX);
    assert!(sk.is_null(2));
    assert_eq!(ss.values(), &[15, 20, 7]);
    assert_eq!(sn.values(), &[2, 1, 2]);

    // 2. ROLLUP over multi-column GROUP BY with WHERE + computed SUM.
    //    Surviving rows with `k2 IS NOT NULL` have keys (1, 10), (1, 20), and (NULL, 10):
    //      level (k1, k2): (1, 10), (1, 20), (NULL, 10)  -> 3 rows
    //      level (k1):     (1, NULL), (NULL, NULL)       -> 2 rows
    //      level ():       (NULL, NULL)                  -> 1 row
    //    Total SQL-standard ROLLUP rows = 6.
    let rollup = engine
        .sql("SELECT k1, k2, SUM(v * 2.0) FROM nulls WHERE keep = 1 AND k2 IS NOT NULL GROUP BY ROLLUP(k1, k2)")
        .expect("ROLLUP with pre")
        .into_record_batch();
    assert_eq!(rollup.num_rows(), 6);

    // 3. SUM(Int64) overflow with pre must raise BoltError::Type across BOTH
    //    packed 2-key (`groupby.rs`) AND wide 2-key (`groupby_wide.rs`).
    let ov_batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("g1", ArrowDataType::Int32, false),
            ArrowField::new("g2", ArrowDataType::Int32, false),
            ArrowField::new("w1", ArrowDataType::Int64, false),
            ArrowField::new("w2", ArrowDataType::Int64, false),
            ArrowField::new("big", ArrowDataType::Int64, false),
            ArrowField::new("keep", ArrowDataType::Int32, false),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![1, 1])) as ArrayRef,
            Arc::new(Int32Array::from(vec![2, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10, 10])) as ArrayRef,
            Arc::new(Int64Array::from(vec![20, 20])) as ArrayRef,
            Arc::new(Int64Array::from(vec![i64::MAX, 10])) as ArrayRef,
            Arc::new(Int32Array::from(vec![1, 1])) as ArrayRef,
        ],
    )
    .unwrap();
    engine.register_table("ov", ov_batch).unwrap();

    for sql in [
        "SELECT g1, g2, SUM(big) FROM ov WHERE keep = 1 GROUP BY g1, g2",
        "SELECT w1, w2, SUM(big) FROM ov WHERE keep = 1 GROUP BY w1, w2",
    ] {
        match engine.sql(sql) {
            Err(BoltError::Type(msg)) => {
                assert!(
                    msg.contains("SUM(integer) overflow"),
                    "unexpected overflow message for `{sql}`: {msg}"
                );
            }
            Ok(h) => panic!(
                "expected SUM(integer) overflow for `{sql}`, got {:?}",
                h.record_batch()
            ),
            Err(other) => panic!("expected BoltError::Type for `{sql}`, got {other:?}"),
        }
    }
}

#[test]
#[ignore = "requires CUDA runtime"]
fn e2e_resident_groupby_with_pre_matches_host_and_increments_device_pipelines() {
    use craton_bolt::metrics::{metrics, Counter};

    let mut engine = craton_bolt::Engine::new().expect("Engine::new failed");

    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("region_id", ArrowDataType::Int32, false),
            ArrowField::new("category_id", ArrowDataType::Int32, false),
            ArrowField::new("order_day", ArrowDataType::Int32, false),
            ArrowField::new("qty", ArrowDataType::Int32, false),
            ArrowField::new("price", ArrowDataType::Float64, false),
            ArrowField::new("weight_f32", ArrowDataType::Float32, false),
            ArrowField::new("flag", ArrowDataType::Boolean, false),
            ArrowField::new("f_key", ArrowDataType::Float64, false),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 1, 2, 2, 2])) as ArrayRef,
            Arc::new(Int32Array::from(vec![10, 10, 20, 10, 10, 20])) as ArrayRef,
            Arc::new(Int32Array::from(vec![200, 150, 250, 300, 310, 100])) as ArrayRef,
            Arc::new(Int32Array::from(vec![2, 99, 3, 4, 5, 99])) as ArrayRef,
            Arc::new(Float64Array::from(vec![10.0, 99.0, 20.0, 5.0, 15.0, 88.0])) as ArrayRef,
            Arc::new(Float32Array::from(vec![1.5, 9.0, 2.5, 3.0, 4.0, 8.0])) as ArrayRef,
            Arc::new(BooleanArray::from(vec![
                true, true, false, true, false, true,
            ])) as ArrayRef,
            Arc::new(Float64Array::from(vec![
                0.0,
                -0.0,
                -0.0,
                f64::from_bits(0x7ff8_0000_0000_0001),
                f64::from_bits(0x7ff8_0000_0000_0099),
                1.0,
            ])) as ArrayRef,
        ],
    )
    .unwrap();
    engine.register_table("res_sales", batch).unwrap();

    // 1) 2-col GROUP BY + WHERE + fused arithmetic pre + COUNT(*), SUM(Int32->Int64),
    //    SUM(Float32), AVG, MIN, MAX — must stay 100% VRAM-resident and increment DevicePipelines.
    let before = metrics().counter(Counter::DevicePipelines).get();
    let b = engine
        .sql(
            "SELECT region_id, category_id, \
                    SUM(qty * price) AS rev, \
                    SUM(qty) AS total_qty, \
                    SUM(weight_f32) AS total_w, \
                    AVG(price) AS avg_price, \
                    MIN(qty) AS min_q, \
                    MAX(qty) AS max_q, \
                    COUNT(*) AS cnt \
             FROM res_sales \
             WHERE order_day >= 180 \
             GROUP BY region_id, category_id \
             ORDER BY region_id, category_id",
        )
        .unwrap()
        .into_record_batch();
    let after = metrics().counter(Counter::DevicePipelines).get();
    assert!(
        after > before,
        "expected VRAM-resident groupby_with_pre to increment Counter::DevicePipelines ({before} -> {after})"
    );
    // Surviving rows (order_day >= 180):
    //   row 0: (1, 10), qty=2, price=10, w=1.5 -> rev=20, qty=2, w=1.5, avg=10, min=2, max=2, cnt=1
    //   row 2: (1, 20), qty=3, price=20, w=2.5 -> rev=60, qty=3, w=2.5, avg=20, min=3, max=3, cnt=1
    //   rows 3,4: (2, 10), qty=4,5, price=5,15, w=3,4 -> rev=20+75=95, qty=9, w=7.0, avg=10, min=4, max=5, cnt=2
    assert_eq!(b.num_rows(), 3);
    let r = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let c = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
    let rev = b.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
    let tq = b.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
    let tw = b.column(4).as_any().downcast_ref::<Float32Array>().unwrap();
    let ap = b.column(5).as_any().downcast_ref::<Float64Array>().unwrap();
    let min_q = b.column(6).as_any().downcast_ref::<Int32Array>().unwrap();
    let max_q = b.column(7).as_any().downcast_ref::<Int32Array>().unwrap();
    let cnt = b.column(8).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(r.values().as_ref(), &[1, 1, 2]);
    assert_eq!(c.values().as_ref(), &[10, 20, 10]);
    assert_eq!(rev.values().as_ref(), &[20.0, 60.0, 95.0]);
    assert_eq!(tq.values().as_ref(), &[2, 3, 9]);
    assert_eq!(tw.values().as_ref(), &[1.5, 2.5, 7.0]);
    assert_eq!(ap.values().as_ref(), &[10.0, 20.0, 10.0]);
    assert_eq!(min_q.values().as_ref(), &[2, 3, 4]);
    assert_eq!(max_q.values().as_ref(), &[2, 3, 5]);
    assert_eq!(cnt.values().as_ref(), &[1, 1, 2]);

    // 2) 1-col Float64 key with -0.0 / +0.0 and distinct NaN payloads + AVG(Bool) on VRAM:
    let before2 = metrics().counter(Counter::DevicePipelines).get();
    let b2 = engine
        .sql(
            "SELECT f_key, COUNT(*) AS cnt, AVG(flag) AS flag_rate \
             FROM res_sales \
             WHERE order_day >= 180 \
             GROUP BY f_key",
        )
        .unwrap()
        .into_record_batch();
    let after2 = metrics().counter(Counter::DevicePipelines).get();
    assert!(
        after2 > before2,
        "expected 1-col Float64 resident groupby_with_pre to increment DevicePipelines"
    );
    // Surviving rows: row 0 (+0.0, true), row 2 (-0.0, false) -> merged +0.0 group (cnt=2, avg=0.5);
    //                 rows 3,4 (two NaN payloads, true & false) -> merged NaN group (cnt=2, avg=0.5).
    assert_eq!(b2.num_rows(), 2);
    let fk = b2
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let cnt2 = b2.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    let fr = b2
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(fk.value(0).to_bits(), 0.0_f64.to_bits());
    assert!(fk.value(1).is_nan());
    assert_eq!(cnt2.values().as_ref(), &[2, 2]);
    assert_eq!(fr.values().as_ref(), &[0.5, 0.5]);

    // 3) Empty post-WHERE survival on resident fast path returns 0 rows cleanly:
    let before3 = metrics().counter(Counter::DevicePipelines).get();
    let b3 = engine
        .sql(
            "SELECT region_id, category_id, COUNT(*) AS cnt \
             FROM res_sales \
             WHERE order_day > 9999 \
             GROUP BY region_id, category_id",
        )
        .unwrap()
        .into_record_batch();
    let after3 = metrics().counter(Counter::DevicePipelines).get();
    assert!(after3 > before3);
    assert_eq!(b3.num_rows(), 0);
}
