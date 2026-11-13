// SPDX-License-Identifier: AGPL-3.0-only

//! `SUM` / `AVG` / `VAR_*` / `STDDEV_*` over a `FLOAT` (Float32) column
//! accumulate in double precision (W8-B-007). `SUM(Float32)` still returns
//! Float32, so the result must equal the f64 sum of the inputs rounded once
//! to f32 (within one ulp); the Float64 outputs must match an f64 reference
//! to a tight relative tolerance. Every case runs under both tier policies,
//! scalar and grouped (Tier-1 shared-memory and Tier-2 hash cardinalities),
//! with and without a WHERE filter, and one run reports every mismatch.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use craton_bolt::{Engine, TierPolicy};

const N: usize = 1_000_000;

/// Deterministic values in `[0, 1000)` with a fractional part.
fn y_value(i: usize) -> f32 {
    let h = (i as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((h >> 40) as f32) / ((1u64 << 24) as f32) * 1000.0
}

/// Large and small magnitudes: blocks of 1e7 alternate with blocks of 0.25.
fn z_value(i: usize) -> f32 {
    if (i / 8000) % 2 == 0 {
        1.0e7
    } else {
        0.25
    }
}

struct Data {
    k8: Vec<i32>,
    k2k: Vec<i32>,
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    /// `y` with every fifth row NULL.
    w: Vec<Option<f32>>,
}

fn data() -> Data {
    Data {
        k8: (0..N).map(|i| (i % 8) as i32).collect(),
        k2k: (0..N).map(|i| (i % 2000) as i32).collect(),
        x: vec![0.1_f32; N],
        y: (0..N).map(y_value).collect(),
        z: (0..N).map(z_value).collect(),
        w: (0..N).map(|i| (i % 5 != 0).then(|| y_value(i))).collect(),
    }
}

fn batch(d: &Data) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k8", DataType::Int32, false),
        Field::new("k2k", DataType::Int32, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
        Field::new("z", DataType::Float32, false),
        Field::new("w", DataType::Float32, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(d.k8.clone())) as ArrayRef,
            Arc::new(Int32Array::from(d.k2k.clone())),
            Arc::new(Float32Array::from(d.x.clone())),
            Arc::new(Float32Array::from(d.y.clone())),
            Arc::new(Float32Array::from(d.z.clone())),
            Arc::new(Float32Array::from(d.w.clone())),
        ],
    )
    .unwrap()
}

#[derive(Clone, Copy, Debug)]
enum Agg {
    Sum,
    Avg,
    VarPop,
    StddevSamp,
}

/// f64 reference over the selected non-NULL values.
fn reference(agg: Agg, vals: &[f64]) -> f64 {
    let n = vals.len() as f64;
    let sum: f64 = vals.iter().sum();
    let mean = sum / n;
    let m2: f64 = vals.iter().map(|v| (v - mean) * (v - mean)).sum();
    match agg {
        Agg::Sum => sum,
        Agg::Avg => mean,
        Agg::VarPop => m2 / n,
        Agg::StddevSamp => (m2 / (n - 1.0)).sqrt(),
    }
}

fn ulps_apart(a: f32, b: f32) -> u32 {
    let key = |v: f32| {
        let bits = v.to_bits() as i32;
        if bits < 0 {
            i32::MIN.wrapping_sub(bits)
        } else {
            bits
        }
    };
    key(a).wrapping_sub(key(b)).unsigned_abs()
}

/// Checks one result value; `None` when it is accurate enough.
fn judge(agg: Agg, col: &dyn Array, row: usize, want: f64) -> Option<String> {
    if col.is_null(row) {
        return Some("NULL".into());
    }
    match agg {
        Agg::Sum => {
            let Some(c) = col.as_any().downcast_ref::<Float32Array>() else {
                return Some(format!("output dtype {:?}, want Float32", col.data_type()));
            };
            let got = c.value(row);
            let want32 = want as f32;
            (ulps_apart(got, want32) > 1)
                .then(|| format!("got {got} want {want32} ({} ulps)", ulps_apart(got, want32)))
        }
        _ => {
            let Some(c) = col.as_any().downcast_ref::<Float64Array>() else {
                return Some(format!("output dtype {:?}, want Float64", col.data_type()));
            };
            let got = c.value(row);
            let rel = ((got - want) / want).abs();
            (rel > 1e-9).then(|| format!("got {got} want {want} (rel {rel:e})"))
        }
    }
}

struct Case {
    sql: String,
    agg: Agg,
    col: fn(&Data, usize) -> Option<f32>,
    filter: fn(&Data, usize) -> bool,
    key: Option<fn(&Data, usize) -> i32>,
}

fn col_x(d: &Data, i: usize) -> Option<f32> {
    Some(d.x[i])
}
fn col_y(d: &Data, i: usize) -> Option<f32> {
    Some(d.y[i])
}
fn col_z(d: &Data, i: usize) -> Option<f32> {
    Some(d.z[i])
}
fn col_w(d: &Data, i: usize) -> Option<f32> {
    d.w[i]
}
fn all(_: &Data, _: usize) -> bool {
    true
}
fn y_gt_500(d: &Data, i: usize) -> bool {
    d.y[i] > 500.0
}
fn key_k8(d: &Data, i: usize) -> i32 {
    d.k8[i]
}
fn key_k2k(d: &Data, i: usize) -> i32 {
    d.k2k[i]
}

fn agg_sql(agg: Agg, col: &str) -> String {
    match agg {
        Agg::Sum => format!("SUM({col})"),
        Agg::Avg => format!("AVG({col})"),
        Agg::VarPop => format!("VAR_POP({col})"),
        Agg::StddevSamp => format!("STDDEV_SAMP({col})"),
    }
}

fn cases() -> Vec<Case> {
    type ColFn = fn(&Data, usize) -> Option<f32>;
    type FilterFn = fn(&Data, usize) -> bool;
    let cols: [(&str, ColFn); 4] = [("x", col_x), ("y", col_y), ("z", col_z), ("w", col_w)];
    let filters: [(&str, FilterFn); 2] = [("", all), (" WHERE y > 500", y_gt_500)];
    let mut out = Vec::new();
    for agg in [Agg::Sum, Agg::Avg, Agg::VarPop, Agg::StddevSamp] {
        for (cname, col) in cols {
            for (fsql, filter) in filters {
                let a = agg_sql(agg, cname);
                out.push(Case {
                    sql: format!("SELECT {a} FROM t{fsql}"),
                    agg,
                    col,
                    filter,
                    key: None,
                });
                for (kname, key) in [("k8", key_k8 as fn(&Data, usize) -> i32), ("k2k", key_k2k)] {
                    out.push(Case {
                        sql: format!(
                            "SELECT {kname}, {a} FROM t{fsql} GROUP BY {kname} ORDER BY {kname}"
                        ),
                        agg,
                        col,
                        filter,
                        key: Some(key),
                    });
                }
            }
        }
    }
    out
}

fn key_at(col: &dyn Array, row: usize) -> i64 {
    let c = arrow::compute::cast(&col.slice(row, 1), &DataType::Int64).unwrap();
    c.as_any().downcast_ref::<Int64Array>().unwrap().value(0)
}

fn run_case(engine: &mut Engine, d: &Data, case: &Case) -> Option<String> {
    let out = match engine.sql(&case.sql) {
        Ok(o) => o.into_record_batch(),
        Err(e) => return Some(format!("error: {e}")),
    };
    match case.key {
        None => {
            let vals: Vec<f64> = (0..N)
                .filter(|&i| (case.filter)(d, i))
                .filter_map(|i| (case.col)(d, i).map(f64::from))
                .collect();
            if out.num_rows() != 1 {
                return Some(format!("{} rows", out.num_rows()));
            }
            judge(
                case.agg,
                out.column(0).as_ref(),
                0,
                reference(case.agg, &vals),
            )
        }
        Some(key) => {
            let mut groups: std::collections::BTreeMap<i64, Vec<f64>> = Default::default();
            for i in (0..N).filter(|&i| (case.filter)(d, i)) {
                // A group whose values are all NULL still appears (with a
                // NULL aggregate), so register the key before the value.
                let vals = groups.entry(key(d, i) as i64).or_default();
                if let Some(v) = (case.col)(d, i) {
                    vals.push(f64::from(v));
                }
            }
            if out.num_rows() != groups.len() {
                return Some(format!("{} rows, want {}", out.num_rows(), groups.len()));
            }
            let mut bad = Vec::new();
            for row in 0..out.num_rows() {
                let k = key_at(out.column(0).as_ref(), row);
                let Some(vals) = groups.get(&k) else {
                    return Some(format!("unexpected key {k}"));
                };
                if vals.is_empty() {
                    if !out.column(1).is_null(row) {
                        bad.push(format!("key {k}: want NULL for an all-NULL group"));
                    }
                    continue;
                }
                if let Some(m) = judge(
                    case.agg,
                    out.column(1).as_ref(),
                    row,
                    reference(case.agg, vals),
                ) {
                    bad.push(format!("key {k}: {m}"));
                }
            }
            (!bad.is_empty()).then(|| {
                let n = bad.len();
                bad.truncate(3);
                format!("{n} groups off, e.g. {}", bad.join("; "))
            })
        }
    }
}

#[test]
#[ignore = "gpu:e2e"]
fn float32_aggregates_accumulate_in_double_precision() {
    let d = data();
    let b = batch(&d);
    let mut failures = Vec::new();
    for policy in [TierPolicy::GpuFirst, TierPolicy::CostBased] {
        let mut engine = Engine::builder()
            .tier_policy(policy)
            .build()
            .expect("CUDA context");
        engine.register_table("t", b.clone()).unwrap();
        for case in cases() {
            if let Some(m) = run_case(&mut engine, &d, &case) {
                failures.push(format!("[{policy:?}] {}\n    {m}", case.sql));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
