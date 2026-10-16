// SPDX-License-Identifier: Apache-2.0

//! Cost-based tier selection: statements the cost model sends to the host
//! tier return exactly what the GPU path returns.
//!
//! `cargo test --test cost_tier_test -- --ignored --test-threads=1`

use std::sync::Arc;

use arrow_array::{Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::metrics::{metrics, Counter};
use craton_bolt::{Engine, TierPolicy};

/// 500 rows with NULLs in every nullable column.
fn table() -> RecordBatch {
    let n = 500;
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int64, false),
        ArrowField::new("i", ArrowDataType::Int32, true),
        ArrowField::new("f", ArrowDataType::Float64, true),
        ArrowField::new("s", ArrowDataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from((0..n).collect::<Vec<i64>>())),
            Arc::new(Int32Array::from(
                (0..n)
                    .map(|x| (x % 7 != 3).then_some((x % 50) as i32 - 20))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                (0..n)
                    .map(|x| (x % 11 != 5).then_some(x as f64 * 0.25 - 30.0))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..n)
                    .map(|x| (x % 13 != 0).then(|| format!("v{}", x % 9)))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

const QUERIES: &[&str] = &[
    "SELECT k, i, f FROM t WHERE i > 0",
    "SELECT k, s FROM t WHERE s = 'v3' AND k < 300",
    "SELECT k, i * 2 + 1 AS j, f - 1.5 AS g FROM t WHERE f IS NOT NULL OR i IS NULL",
    "SELECT k, CASE WHEN i > 10 THEN 1 WHEN i < -10 THEN -1 ELSE 0 END AS c FROM t",
    "SELECT k FROM t WHERE NOT (i < 5) ORDER BY k DESC LIMIT 7",
    "SELECT k, i FROM t ORDER BY i, k LIMIT 20 OFFSET 3",
    "SELECT COUNT(*) AS n, COUNT(i) AS ci, SUM(i) AS si, MIN(f) AS lo, MAX(f) AS hi FROM t",
    "SELECT AVG(f) AS a, SUM(k) AS sk FROM t WHERE i >= 0",
    "SELECT CAST(i AS BIGINT) + k AS z FROM t WHERE k BETWEEN 10 AND 40",
];

#[test]
#[ignore = "gpu:e2e — host-tier results match the GPU path"]
fn host_tier_matches_gpu() {
    let mut e = Engine::new().expect("ctx");
    e.register_table("t", table()).unwrap();
    for q in QUERIES {
        e.set_tier_policy(TierPolicy::GpuFirst);
        let gpu = e.sql(q).unwrap().into_record_batch();
        e.set_tier_policy(TierPolicy::CostBased);
        let before = metrics().counter(Counter::HostTierQueries).get();
        let host = e.sql(q).unwrap().into_record_batch();
        assert_eq!(
            metrics().counter(Counter::HostTierQueries).get(),
            before + 1,
            "the cost model should pick the host for 500 rows: {q}"
        );
        assert_eq!(gpu.num_rows(), host.num_rows(), "{q}");
        assert_eq!(gpu.num_columns(), host.num_columns(), "{q}");
        for c in 0..gpu.num_columns() {
            let (g, h) = (gpu.column(c), host.column(c));
            let g = arrow::compute::cast(g, h.data_type()).unwrap();
            if let (Some(g), Some(h)) = (
                g.as_any().downcast_ref::<Float64Array>(),
                h.as_any().downcast_ref::<Float64Array>(),
            ) {
                for r in 0..g.len() {
                    assert_eq!(g.is_null(r), h.is_null(r), "{q} col {c} row {r}");
                    if !g.is_null(r) {
                        assert!(
                            (g.value(r) - h.value(r)).abs() < 1e-9,
                            "{q} col {c} row {r}"
                        );
                    }
                }
            } else {
                assert_eq!(g.as_ref(), h.as_ref(), "{q} col {c}");
            }
        }
        assert_eq!(
            gpu.schema()
                .fields()
                .iter()
                .map(|f| f.name())
                .collect::<Vec<_>>(),
            host.schema()
                .fields()
                .iter()
                .map(|f| f.name())
                .collect::<Vec<_>>(),
            "{q}"
        );
    }
}

#[test]
#[ignore = "gpu:e2e — EXPLAIN shows the estimate; edge cases stay on the GPU"]
fn explain_and_declines() {
    let mut e = Engine::builder()
        .tier_policy(TierPolicy::CostBased)
        .build()
        .expect("ctx");
    e.register_table("t", table()).unwrap();
    let plan = e.explain_sql("SELECT k FROM t WHERE i > 0").unwrap();
    assert!(plan.contains("Tier: host"), "{plan}");
    assert!(plan.contains("Memory: streaming"), "{plan}");
    let plan = e
        .explain_sql("SELECT i, COUNT(*) FROM t GROUP BY i")
        .unwrap();
    assert!(plan.contains("Memory: partitionable"), "{plan}");
    // Empty input: the GPU path's semantics decide.
    let before = metrics().counter(Counter::HostTierQueries).get();
    let z = e
        .sql("SELECT SUM(i) FROM t WHERE k < 0")
        .unwrap()
        .into_record_batch();
    assert_eq!(z.num_rows(), 1);
    assert_eq!(metrics().counter(Counter::HostTierQueries).get(), before);
    // GROUP BY is GPU-only.
    let g = e
        .explain_sql("SELECT i, COUNT(*) FROM t GROUP BY i")
        .unwrap();
    assert!(!g.contains("Tier: host"), "{g}");

    // A large resident table prefers the GPU.
    let n = 4_000_000i64;
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "x",
        ArrowDataType::Int64,
        false,
    )]));
    let big = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from((0..n).collect::<Vec<_>>()))],
    )
    .unwrap();
    e.register_table("big", big).unwrap();
    // Warm it: resident on the device.
    e.sql("SELECT SUM(x) FROM big WHERE x > 5").unwrap();
    let plan = e.explain_sql("SELECT SUM(x) FROM big WHERE x > 5").unwrap();
    assert!(plan.contains("Tier: GPU"), "{plan}");

    // Calibration measures the device and keeps the decisions sensible.
    let m = e.calibrate_cost_model().unwrap();
    assert!(m.h2d_bytes_per_sec > 1e8, "{m:?}");
    assert!(m.cpu_ns_per_row > 0.0 && m.gpu_ns_per_row > 0.0, "{m:?}");
    let plan = e
        .explain_sql("SELECT k FROM t WHERE i > 0 LIMIT 5")
        .unwrap();
    assert!(plan.contains("Tier: host"), "{plan}");
    assert!(!e
        .table_names()
        .iter()
        .any(|t| t.starts_with("__bolt_calibrate")));
}

/// SQL NULL semantics of the fused GPU kernels, checked against the host
/// evaluator: comparisons with a NULL operand, Kleene AND / OR / NOT, CASE
/// (with and without ELSE), NULL propagation through arithmetic, and
/// aggregates over nullable feeds. Each of these returned wrong rows or
/// values on the GPU before (NULLs read as 0).
#[test]
#[ignore = "gpu:e2e — GPU NULL semantics match the host evaluator"]
fn gpu_null_semantics_match_host() {
    let mut e = Engine::new().expect("ctx");
    e.register_table("t", table()).unwrap();
    let qs = [
        "SELECT k, f FROM t WHERE k > 20",
        "SELECT k, f FROM t",
        "SELECT k, i FROM t",
        "SELECT k, i + 1 AS j FROM t",
        "SELECT k, i + 1 AS j FROM t WHERE k > 3",
        "SELECT k FROM t WHERE i < 5",
        "SELECT k FROM t WHERE NOT (i < 5)",
        "SELECT k FROM t WHERE i IS NULL OR f > 0",
        "SELECT k FROM t WHERE i < 5 OR k > 400",
        "SELECT k, CASE WHEN i > 10 THEN 1 ELSE 0 END AS c FROM t",
        "SELECT k, i IS NULL AS n FROM t",
        "SELECT COUNT(*) FROM t WHERE i < 5",
        "SELECT SUM(k) FROM t WHERE i < 5 OR f > 0",
        "SELECT k, CASE WHEN i < 10 THEN 1 ELSE 0 END AS c FROM t",
        "SELECT k, CASE WHEN i < 10 THEN f END AS c FROM t WHERE k < 100",
        "SELECT SUM(CASE WHEN i < 10 THEN 1 ELSE 0 END) AS s FROM t",
        "SELECT COUNT(CASE WHEN i > 10 THEN 1 END) AS s FROM t",
        "SELECT AVG(CASE WHEN f > 0 THEN f ELSE 0 END) AS s FROM t WHERE k > 7",
        "SELECT k FROM t WHERE i < f",
        "SELECT k FROM t WHERE (i < 5 AND f > 0) OR NOT (k < 450)",
        "SELECT k, (i < 5) OR (f > 0) AS b FROM t",
        "SELECT k, NOT (i < 5) AS b FROM t",
        "SELECT k, i + f AS z FROM t WHERE i IS NOT NULL",
        "SELECT MIN(i) AS lo, MAX(f) AS hi, COUNT(f) AS cf FROM t WHERE i < 0",
        "SELECT k FROM t WHERE CASE WHEN i > 0 THEN f > 0 ELSE k > 300 END",
        "SELECT COUNT(f) AS cf FROM t",
        "SELECT COUNT(f) AS cf FROM t WHERE k > 3",
        "SELECT COUNT(f) AS cf, SUM(f) AS sf FROM t WHERE k > 3",
        "SELECT COUNT(i) AS ci FROM t WHERE k > 3",
        "SELECT MIN(i) AS lo FROM t WHERE i < 0",
        "SELECT COUNT(f) AS cf FROM t WHERE i < 0",
    ];
    for q in qs {
        e.set_tier_policy(TierPolicy::GpuFirst);
        let g = e
            .sql(q)
            .unwrap_or_else(|err| panic!("{q}: {err}"))
            .into_record_batch();
        e.set_tier_policy(TierPolicy::CostBased);
        let h = e
            .sql(q)
            .unwrap_or_else(|err| panic!("{q}: {err}"))
            .into_record_batch();
        assert_eq!(g.num_rows(), h.num_rows(), "{q}");
        for c in 0..g.num_columns() {
            let gc = arrow::compute::cast(g.column(c), h.column(c).data_type()).unwrap();
            assert_eq!(gc.as_ref(), h.column(c).as_ref(), "{q} column {c}");
        }
    }
}
