// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the v0.6 / M4 `DataType::Decimal128(precision, scale)`
//! plumbing. Plan-level only — the GPU codegen rejects every Decimal column
//! with a clear "Decimal128 not yet lowered to GPU; coming in a follow-up"
//! message until the Decimal128 codegen lands.
//!
//! Three scenarios:
//!
//! 1. A schema with a `Decimal128(18, 2)` column registers cleanly through
//!    `MemTableProvider`. This pins the `Field` constructor / Arrow
//!    round-trip plumbing.
//! 2. `SELECT decimal_col FROM t` parses and lowers logically; the
//!    physical-plan boundary rejects it with the documented error.
//! 3. `CAST(int_col AS DECIMAL(18, 2))` parses (sqlparser accepts the
//!    syntax) but the SQL lowering step surfaces a clear `BoltError::Plan`
//!    naming the Decimal128 follow-up.

use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Field, MemTableProvider, Schema,
};
use craton_bolt::BoltError;

/// Build a `Schema` with a `Decimal128(18, 2)` value column alongside an
/// `Int32` key column. Reused by every test below.
fn schema_with_decimal() -> Schema {
    Schema::new(vec![
        Field {
            name: "id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "amount".into(),
            dtype: DataType::Decimal128(18, 2),
            nullable: true,
        },
    ])
}

fn provider_with_decimal() -> MemTableProvider {
    MemTableProvider::new().with_table("t", schema_with_decimal())
}

// ---- 1. Schema registration -------------------------------------------------

/// A schema with a `Decimal128(18, 2)` column registers cleanly through
/// `MemTableProvider` and round-trips through `TableProvider::schema`. The
/// `is_decimal()` helper recognises the type; `byte_width()` reports 16
/// (i128 storage).
#[test]
fn decimal128_schema_registers_and_roundtrips() {
    use craton_bolt::plan::TableProvider;
    let p = provider_with_decimal();
    let s = p
        .schema("t")
        .expect("registered schema must be retrievable");
    assert_eq!(s.fields.len(), 2);
    let amount = &s.fields[1];
    assert_eq!(amount.name, "amount");
    assert_eq!(amount.dtype, DataType::Decimal128(18, 2));
    assert!(amount.nullable);

    assert!(amount.dtype.is_decimal(), "Decimal128 must report is_decimal");
    assert_eq!(
        amount.dtype.byte_width(),
        Some(16),
        "Decimal128 occupies an i128 (16 bytes) on device"
    );

    // The non-decimal column is unaffected by the new helper.
    assert!(!s.fields[0].dtype.is_decimal());
    assert_eq!(s.fields[0].dtype.byte_width(), Some(4));
}

// ---- 2. SELECT over a Decimal column ---------------------------------------

/// `SELECT amount FROM t` parses successfully (the planner is happy to
/// resolve the column and type-check the projection). The physical
/// lowerer then refuses to lower the Decimal column with the documented
/// "Decimal128 not yet lowered to GPU; coming in a follow-up" message.
#[test]
fn select_decimal_column_parses_then_rejects_at_lower() {
    let provider = provider_with_decimal();
    let plan = parse_sql("SELECT amount FROM t", &provider)
        .expect("SELECT over a Decimal column must parse and type-check");
    // The logical schema must already carry Decimal128 through.
    let schema = plan.schema().expect("logical schema resolution");
    assert_eq!(schema.fields.len(), 1);
    assert_eq!(schema.fields[0].dtype, DataType::Decimal128(18, 2));

    // The physical lowerer is the documented rejection boundary.
    let err = lower_physical(&plan)
        .expect_err("physical lowering must reject Decimal128 columns");
    match err {
        BoltError::Plan(msg) => {
            assert!(
                msg.contains("Decimal128 not yet lowered to GPU"),
                "physical lowering should surface the canonical Decimal128 \
                 follow-up message; got: {msg}",
            );
            assert!(
                msg.contains("follow-up"),
                "rejection message should name the follow-up; got: {msg}",
            );
        }
        other => panic!(
            "expected BoltError::Plan with the Decimal128 follow-up message, \
             got {other:?}"
        ),
    }
}

/// Negative companion: a SELECT that touches only the non-Decimal column
/// must still lower cleanly. Guards against the rejection arm
/// over-reaching and breaking previously-supported queries.
#[test]
fn select_non_decimal_column_still_lowers() {
    let provider = provider_with_decimal();
    let plan = parse_sql("SELECT id FROM t", &provider)
        .expect("SELECT of an Int32 column must parse");
    let _ = lower_physical(&plan)
        .expect("physical lowering of a non-Decimal SELECT must succeed");
}

// ---- 3. CAST(int AS DECIMAL(18, 2)) ----------------------------------------

/// `CAST(id AS DECIMAL(18, 2))` parses through sqlparser without an error
/// (the syntax is well-formed SQL). The SQL frontend then surfaces a
/// `BoltError::Plan` at the lowering boundary, mirroring how the
/// dedicated rejection arm for Decimal column references behaves — and
/// matching the documented v0.6 / M4 contract.
#[test]
fn cast_int_to_decimal_parses_then_rejects_at_lower() {
    let provider = provider_with_decimal();
    let result = parse_sql("SELECT CAST(id AS DECIMAL(18, 2)) FROM t", &provider);
    let err = result.expect_err("CAST to DECIMAL must be rejected at lowering");
    match err {
        BoltError::Plan(msg) => {
            assert!(
                msg.contains("CAST") || msg.contains("Decimal128"),
                "CAST rejection should name CAST or Decimal128; got: {msg}",
            );
        }
        // The frontend may also surface this as a Type error if a future
        // refactor moves the rejection earlier — accept that too, but
        // require the message still mentions CAST / DECIMAL so the user
        // can find the source of the error.
        BoltError::Type(msg) => {
            assert!(
                msg.contains("CAST")
                    || msg.contains("DECIMAL")
                    || msg.contains("Decimal128"),
                "Type-based CAST rejection should still mention CAST/DECIMAL; \
                 got: {msg}",
            );
        }
        other => panic!(
            "expected BoltError::Plan (or Type) naming CAST/Decimal128, got {other:?}"
        ),
    }
}

/// Companion: CAST to a primitive non-Decimal type also routes through the
/// same rejection arm for v0.6 / M4 — every CAST surfaces the same
/// "Decimal128 follow-up" plan error. This guards the test above against
/// a future relaxation that accidentally accepts CAST(int AS BIGINT) but
/// not CAST(int AS DECIMAL).
#[test]
fn cast_to_primitive_also_rejects_for_now() {
    let provider = provider_with_decimal();
    let err = parse_sql("SELECT CAST(id AS BIGINT) FROM t", &provider)
        .expect_err("CAST is unsupported at the lowering boundary in v0.6 / M4");
    assert!(
        matches!(err, BoltError::Plan(_) | BoltError::Type(_)),
        "CAST rejection must be a Plan or Type error; got {err:?}",
    );
}
