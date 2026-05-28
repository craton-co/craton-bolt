// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the v0.6 / M4 Date32 + Timestamp type plumbing.
//!
//! These tests pin the **planner-side** contract: the new dtypes survive a
//! register-table round trip, schemas reflect them correctly, the SQL
//! `DATE 'YYYY-MM-DD'` / `TIMESTAMP 'YYYY-MM-DD HH:MM:SS'` literal surfaces
//! parse into the right `Literal` variants, and the GPU codegen accepts a
//! bare projection of Date32 / Timestamp columns (since v0.7 they lower to
//! integer arithmetic on the underlying days / ticks).
//!
//! Nothing here touches the GPU — the goal is purely to confirm the dtype
//! plumbing is in place so a follow-up wave can wire up host execution.
//! The companion changes live in `src/plan/logical_plan.rs` (variant
//! definitions), `src/plan/sql_frontend.rs` (literal parsing), and
//! `src/plan/physical_plan.rs` (codegen accept) and `src/exec/engine.rs`
//! (Arrow conversion).

use std::sync::Arc;

use arrow_array::{
    Array, Date32Array, Int32Array, RecordBatch, TimestampNanosecondArray,
};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
    TimeUnit as ArrowTimeUnit,
};

use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Expr, Field, Literal, LogicalPlan, MemTableProvider,
    Schema, TimeUnit,
};

/// Build a fixture schema with one Date32 and one Timestamp(Nanosecond, None)
/// column alongside an i32 id key. Mirrors the Arrow side built by
/// [`datetime_record_batch`] below.
fn datetime_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("d", DataType::Date32, true),
        Field::new("ts", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
    ])
}

/// Arrow `RecordBatch` mirror of [`datetime_schema`]. Three rows; one
/// NULL date and one NULL timestamp so the validity bits get exercised
/// when (eventually) the executor reads these columns.
fn datetime_record_batch() -> RecordBatch {
    let id = Int32Array::from(vec![1, 2, 3]);
    // 2024-01-01, NULL, 2024-12-31 — values are days since 1970-01-01.
    // Exact values not checked here; we only care that the schema survives.
    let d = Date32Array::from(vec![Some(19_723), None, Some(20_088)]);
    let ts = TimestampNanosecondArray::from(vec![
        Some(1_704_067_200_000_000_000), // 2024-01-01T00:00:00Z
        None,
        Some(1_735_603_200_000_000_000), // 2024-12-31T00:00:00Z
    ]);
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int32, false),
        ArrowField::new("d", ArrowDataType::Date32, true),
        ArrowField::new(
            "ts",
            ArrowDataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
            true,
        ),
    ]));
    RecordBatch::try_new(
        arrow_schema,
        vec![Arc::new(id), Arc::new(d), Arc::new(ts)],
    )
    .unwrap()
}

/// `byte_width()` must declare 4 for Date32 (i32 days) and 8 for any
/// Timestamp variant (i64 ticks), independent of the unit or timezone.
#[test]
fn byte_width_for_date32_and_timestamp() {
    assert_eq!(DataType::Date32.byte_width(), Some(4));
    for unit in [
        TimeUnit::Second,
        TimeUnit::Millisecond,
        TimeUnit::Microsecond,
        TimeUnit::Nanosecond,
    ] {
        assert_eq!(
            DataType::Timestamp(unit, None).byte_width(),
            Some(8),
            "Timestamp({unit:?}, None) byte_width must be 8 (i64 ticks)"
        );
        // Carrying a timezone does not change the storage width.
        assert_eq!(
            DataType::Timestamp(unit, Some("UTC")).byte_width(),
            Some(8),
            "Timestamp({unit:?}, Some(UTC)) byte_width must be 8 (i64 ticks)"
        );
    }
}

/// `Literal::Date32` and `Literal::Timestamp` resolve to the matching
/// `DataType` variant via the `Literal::dtype()` accessor.
#[test]
fn literal_dtypes_for_date_and_timestamp() {
    assert_eq!(
        Literal::Date32(19_723).dtype(),
        Some(DataType::Date32),
        "Literal::Date32 must resolve to DataType::Date32"
    );
    let ts = Literal::Timestamp(1_704_067_200_000_000_000, TimeUnit::Nanosecond, None);
    assert_eq!(
        ts.dtype(),
        Some(DataType::Timestamp(TimeUnit::Nanosecond, None))
    );

    // The owned-String constructor routes through the timezone interner so
    // the resulting dtype carries a `&'static str`.
    let tz_lit = Literal::timestamp_with_tz(
        1_704_067_200_000_000_000,
        TimeUnit::Nanosecond,
        Some("UTC".to_string()),
    );
    match tz_lit.dtype().unwrap() {
        DataType::Timestamp(unit, Some(s)) => {
            assert_eq!(unit, TimeUnit::Nanosecond);
            assert_eq!(s, "UTC");
        }
        other => panic!("expected Timestamp(_, Some(_)), got {other:?}"),
    }
}

/// Register a table with Date32 and Timestamp columns, plan a SELECT that
/// passes them through unchanged, and assert the output schema round-trips
/// the dtypes (including the Timestamp's TimeUnit + tz=None).
#[test]
fn select_round_trips_date32_and_timestamp_schema() {
    let provider = MemTableProvider::new().with_table("events", datetime_schema());
    let plan = parse_sql("SELECT id, d, ts FROM events", &provider).expect("parse");
    let out_schema = plan.schema().expect("schema");
    assert_eq!(out_schema.fields.len(), 3, "expected 3 output fields");
    assert_eq!(out_schema.fields[0].name, "id");
    assert_eq!(out_schema.fields[0].dtype, DataType::Int32);
    assert_eq!(out_schema.fields[1].name, "d");
    assert_eq!(out_schema.fields[1].dtype, DataType::Date32);
    assert_eq!(out_schema.fields[2].name, "ts");
    assert_eq!(
        out_schema.fields[2].dtype,
        DataType::Timestamp(TimeUnit::Nanosecond, None)
    );

    // Sanity-check that a matching Arrow batch is constructible with the
    // same shape. We're not running the executor — just confirming the
    // engine-facing Arrow types we documented in the plan agree with what
    // `arrow_array` produces for the same conceptual values.
    let batch = datetime_record_batch();
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 3);
    assert_eq!(batch.schema().field(1).data_type(), &ArrowDataType::Date32);
    assert_eq!(
        batch.schema().field(2).data_type(),
        &ArrowDataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
    );
}

/// Filtering on an Int32 key keeps the Date32 / Timestamp columns in the
/// output schema, with the same dtypes. The projection-pass-through path
/// is the primary surface the new variants must survive.
#[test]
fn filter_then_select_preserves_datetime_dtypes() {
    let provider = MemTableProvider::new().with_table("events", datetime_schema());
    let plan = parse_sql("SELECT id, d, ts FROM events WHERE id > 0", &provider)
        .expect("parse");
    let s = plan.schema().expect("schema");
    let dtypes: Vec<&DataType> = s.fields.iter().map(|f| &f.dtype).collect();
    assert_eq!(
        dtypes,
        vec![
            &DataType::Int32,
            &DataType::Date32,
            &DataType::Timestamp(TimeUnit::Nanosecond, None),
        ],
        "datetime dtypes must survive a filter+project plan"
    );
}

/// `DATE 'YYYY-MM-DD'` literals parse into `Literal::Date32(days)`.
/// 1970-01-01 is day 0; 2024-01-01 is 19_723 days later.
#[test]
fn sql_date_literal_parses_to_date32() {
    let provider = MemTableProvider::new().with_table("events", datetime_schema());
    let plan = parse_sql(
        "SELECT DATE '1970-01-01' AS epoch, DATE '2024-01-01' AS newyear FROM events",
        &provider,
    )
    .expect("parse DATE literal");

    // Walk to the Project node so we can inspect the projected expressions.
    let exprs = match &plan {
        LogicalPlan::Project { exprs, .. } => exprs,
        other => panic!("expected Project at root, got {other:?}"),
    };

    // The two SELECT items are aliased; peel through `Alias`.
    let epoch_lit = unwrap_alias_literal(&exprs[0]);
    let newyear_lit = unwrap_alias_literal(&exprs[1]);
    assert_eq!(epoch_lit, &Literal::Date32(0), "DATE '1970-01-01' must be day 0");
    assert_eq!(
        newyear_lit,
        &Literal::Date32(19_723),
        "DATE '2024-01-01' must be day 19723 since the Unix epoch"
    );
}

/// `TIMESTAMP 'YYYY-MM-DD HH:MM:SS'` literals parse into a nanosecond
/// `Literal::Timestamp` with `tz = None`. The epoch literal is exactly 0
/// ticks; one second past the epoch is 10^9 ticks.
#[test]
fn sql_timestamp_literal_parses_to_nanosecond_timestamp() {
    let provider = MemTableProvider::new().with_table("events", datetime_schema());
    let plan = parse_sql(
        "SELECT TIMESTAMP '1970-01-01 00:00:00' AS t0, \
                TIMESTAMP '1970-01-01 00:00:01' AS t1 \
         FROM events",
        &provider,
    )
    .expect("parse TIMESTAMP literal");

    let exprs = match &plan {
        LogicalPlan::Project { exprs, .. } => exprs,
        other => panic!("expected Project at root, got {other:?}"),
    };

    let t0 = unwrap_alias_literal(&exprs[0]);
    let t1 = unwrap_alias_literal(&exprs[1]);
    assert_eq!(
        t0,
        &Literal::Timestamp(0, TimeUnit::Nanosecond, None),
        "TIMESTAMP '1970-01-01 00:00:00' must be 0 ticks since the Unix epoch"
    );
    assert_eq!(
        t1,
        &Literal::Timestamp(1_000_000_000, TimeUnit::Nanosecond, None),
        "TIMESTAMP '1970-01-01 00:00:01' must be 10^9 ns since the Unix epoch"
    );
}

/// v0.7: Date32 and Timestamp columns now lower to GPU as their underlying
/// integer types (i32 days / i64 ticks). A bare projection that just passes
/// these columns through must succeed at `lower_physical` — the prior
/// "Date/Timestamp not yet lowered to GPU" rejection was retired by the
/// GPU lowering wiring that treats them as Int32 / Int64 at the PTX layer
/// while preserving the temporal dtype on the IR.
#[test]
fn gpu_codegen_accepts_date32_column_projection() {
    let provider = MemTableProvider::new().with_table("events", datetime_schema());
    let plan = parse_sql("SELECT d FROM events", &provider).expect("parse");
    let _ = lower_physical(&plan).expect("Date32 column projection must lower in v0.7");
}

#[test]
fn gpu_codegen_accepts_timestamp_column_projection() {
    let provider = MemTableProvider::new().with_table("events", datetime_schema());
    let plan = parse_sql("SELECT ts FROM events", &provider).expect("parse");
    let _ = lower_physical(&plan).expect("Timestamp column projection must lower in v0.7");
}

// ----- helpers --------------------------------------------------------------

/// Unwrap `Alias(Literal(l), _)` and return a reference to the inner
/// `Literal`. Panics with the offending expression on any other shape so
/// the test failure points at the planner output that surprised us.
fn unwrap_alias_literal(e: &Expr) -> &Literal {
    match e {
        Expr::Alias(inner, _) => match inner.as_ref() {
            Expr::Literal(l) => l,
            other => panic!("expected Alias(Literal(..), ..), inner was {other:?}"),
        },
        Expr::Literal(l) => l,
        other => panic!("expected Alias or Literal, got {other:?}"),
    }
}
