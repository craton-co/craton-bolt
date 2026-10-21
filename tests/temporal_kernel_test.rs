// SPDX-License-Identifier: Apache-2.0

//! `DATE` / `TIMESTAMP` values computed by fused GPU projection kernels:
//! passthrough columns under a predicate, `CASE` choosing between temporal
//! columns, temporal literals, `DATE ± integer`, over nullable and
//! non-nullable columns. Each result must carry the logical type (not the
//! `i32` / `i64` device storage), match the host evaluator, and come from
//! the kernel (`EXPLAIN` shows a fused `Projection`).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, TimestampMicrosecondType};
use arrow_array::{Array, Date32Array, Int64Array, RecordBatch, TimestampMicrosecondArray};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema, TimeUnit,
};

use craton_bolt::{Engine, TierPolicy};

const DAY_US: i64 = 86_400_000_000;

fn table() -> RecordBatch {
    let n = 300i64;
    let ts = ArrowDataType::Timestamp(TimeUnit::Microsecond, None);
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int64, false),
        ArrowField::new("d", ArrowDataType::Date32, false),
        ArrowField::new("nd", ArrowDataType::Date32, true),
        ArrowField::new("t", ts.clone(), false),
        ArrowField::new("nt", ts, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from((0..n).collect::<Vec<_>>())),
            Arc::new(Date32Array::from(
                (0..n).map(|k| 19_000 + k as i32).collect::<Vec<_>>(),
            )),
            Arc::new(Date32Array::from(
                (0..n)
                    .map(|k| (k % 4 != 1).then_some(20_000 - k as i32))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(TimestampMicrosecondArray::from(
                (0..n)
                    .map(|k| 1_700_000_000_000_000 + k * 3_600_000_000)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(TimestampMicrosecondArray::from(
                (0..n)
                    .map(|k| (k % 5 != 2).then_some(1_600_000_000_000_000 + k * DAY_US))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn engine() -> Engine {
    let mut e = Engine::new().expect("ctx");
    e.register_table("t", table()).unwrap();
    e
}

/// GPU result of `q`, after checking it runs as a fused kernel and agrees
/// with the host evaluator.
fn gpu(e: &mut Engine, q: &str) -> RecordBatch {
    let plan = e.explain_sql(q).unwrap();
    assert!(
        plan.contains("Projection") && !plan.contains("StringProject"),
        "{q}: not a fused projection\n{plan}"
    );
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
    e.set_tier_policy(TierPolicy::GpuFirst);
    assert_eq!(g.schema().fields().len(), h.schema().fields().len(), "{q}");
    for c in 0..g.num_columns() {
        assert_eq!(
            g.column(c).data_type(),
            h.column(c).data_type(),
            "{q} column {c}"
        );
        assert_eq!(g.column(c).as_ref(), h.column(c).as_ref(), "{q} column {c}");
    }
    g
}

fn dates(b: &RecordBatch, c: usize) -> Vec<Option<i32>> {
    let a = b.column(c).as_primitive::<Date32Type>();
    (0..a.len())
        .map(|i| (!a.is_null(i)).then(|| a.value(i)))
        .collect()
}

fn times(b: &RecordBatch, c: usize) -> Vec<Option<i64>> {
    let a = b.column(c).as_primitive::<TimestampMicrosecondType>();
    (0..a.len())
        .map(|i| (!a.is_null(i)).then(|| a.value(i)))
        .collect()
}

#[test]
#[ignore = "gpu:e2e — temporal passthrough columns leave the kernel typed"]
fn temporal_passthrough_under_predicates() {
    let mut e = engine();
    let b = gpu(&mut e, "SELECT k, d, nd, t, nt FROM t WHERE k < 10");
    assert_eq!(b.num_rows(), 10);
    assert_eq!(b.column(1).data_type(), &ArrowDataType::Date32);
    assert_eq!(
        b.column(3).data_type(),
        &ArrowDataType::Timestamp(TimeUnit::Microsecond, None)
    );
    assert_eq!(dates(&b, 1)[3], Some(19_003));
    assert_eq!(dates(&b, 2)[1], None);
    assert_eq!(dates(&b, 2)[2], Some(19_998));
    assert_eq!(times(&b, 4)[2], None);
    assert_eq!(times(&b, 4)[3], Some(1_600_000_000_000_000 + 3 * DAY_US));
    // No predicate, nullable columns.
    let b = gpu(&mut e, "SELECT nd, nt, k + 1 AS j FROM t");
    assert_eq!(b.num_rows(), 300);
    assert_eq!(dates(&b, 0)[5], None);
    // Predicates over temporal columns.
    let b = gpu(&mut e, "SELECT d, k FROM t WHERE nd > DATE '2024-08-01'");
    assert!(b.num_rows() > 0 && b.num_rows() < 300);
}

#[test]
#[ignore = "gpu:e2e — CASE / literals / DATE arithmetic compute temporal outputs"]
fn computed_temporal_outputs() {
    let mut e = engine();
    let b = gpu(
        &mut e,
        "SELECT k, CASE WHEN k % 2 = 0 THEN d ELSE nd END AS c FROM t WHERE k < 8",
    );
    assert_eq!(
        dates(&b, 1),
        vec![
            Some(19_000),
            None,
            Some(19_002),
            Some(19_997),
            Some(19_004),
            None,
            Some(19_006),
            Some(19_993)
        ]
    );
    let b = gpu(
        &mut e,
        "SELECT CASE WHEN k > 1 THEN nt END AS c, DATE '2024-01-02' AS lit FROM t WHERE k < 4",
    );
    assert_eq!(
        times(&b, 0),
        vec![None, None, None, Some(1_600_000_000_000_000 + 3 * DAY_US)]
    );
    assert_eq!(dates(&b, 1), vec![Some(19_724); 4]);
    // DATE ± integer days.
    let b = gpu(
        &mut e,
        "SELECT d + 7 AS a, nd - 1 AS b, 30 + d AS c, d - CAST(k AS INT) AS z FROM t WHERE k < 3",
    );
    assert_eq!(dates(&b, 0), vec![Some(19_007), Some(19_008), Some(19_009)]);
    assert_eq!(dates(&b, 1), vec![Some(19_999), None, Some(19_997)]);
    assert_eq!(dates(&b, 2), vec![Some(19_030), Some(19_031), Some(19_032)]);
    assert_eq!(dates(&b, 3), vec![Some(19_000); 3]);
    // Date differences stay integers.
    let b = gpu(&mut e, "SELECT d - nd AS diff FROM t WHERE k < 3");
    assert_eq!(b.column(0).data_type(), &ArrowDataType::Int32);
}
