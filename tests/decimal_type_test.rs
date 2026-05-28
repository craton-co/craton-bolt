// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the v0.6/v0.7 `DataType::Decimal128(precision, scale)`
//! plumbing.
//!
//! v0.7 sub-task B (Decimal128 ingest + Codegen wiring) flipped the
//! formerly-rejected SELECT-of-Decimal-column case to *accept* at physical
//! lowering: a bare column reference now emits the dual-register
//! `Op::LoadColumn128` and a paired `Op::Store128` (see
//! `physical_plan::Codegen::emit_column` and the PTX `Op::LoadColumn128`
//! / `Op::Store128` definitions). Decimal128 `+`/`-`/`*` between two
//! matching Decimal columns lowers too, with SQL-convention result
//! dtype rules.
//!
//! Still rejected (host-fallback / follow-up sub-tasks):
//!   * Division and any non-arithmetic op on Decimal128 (comparisons,
//!     logical, etc.).
//!   * CAST involving Decimal128 (source OR target).
//!   * SUM / MIN / MAX / AVG over Decimal128 (no atomic / per-group
//!     accumulator wired yet).
//!
//! Scenarios below:
//!
//! 1. A schema with a `Decimal128(18, 2)` column registers cleanly through
//!    `MemTableProvider`. This pins the `Field` constructor / Arrow
//!    round-trip plumbing.
//! 2. `SELECT decimal_col FROM t` lowers successfully through the v0.7
//!    physical-plan codegen.
//! 3. `CAST(int_col AS DECIMAL(18, 2))` still parses (sqlparser accepts
//!    the syntax) and is rejected at lowering with the documented
//!    "CAST to/from Decimal128 not yet lowered" message.

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

/// v0.7: `SELECT amount FROM t` now lowers cleanly through the physical
/// codegen. The codegen emits `Op::LoadColumn128` + `Op::Store128`,
/// flowing the i128 value through a pair of u64 registers — see
/// `physical_plan::Codegen::emit_column` for the dual-register pair
/// allocation. The plan-level schema still carries the
/// `Decimal128(18, 2)` dtype so downstream consumers see the right
/// precision / scale.
#[test]
fn select_decimal_column_lowers_in_v07() {
    let provider = provider_with_decimal();
    let plan = parse_sql("SELECT amount FROM t", &provider)
        .expect("SELECT over a Decimal column must parse and type-check");
    // The logical schema must carry Decimal128 through unchanged.
    let schema = plan.schema().expect("logical schema resolution");
    assert_eq!(schema.fields.len(), 1);
    assert_eq!(schema.fields[0].dtype, DataType::Decimal128(18, 2));

    // v0.7 sub-task B: the physical lowerer accepts the bare-column
    // SELECT and produces a Projection plan whose output schema preserves
    // the source dtype.
    let phys = lower_physical(&plan)
        .expect("physical lowering of a Decimal SELECT must succeed in v0.7");
    let out = phys.output_schema();
    assert_eq!(out.fields.len(), 1);
    assert_eq!(out.fields[0].dtype, DataType::Decimal128(18, 2));
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

/// Companion: CAST to a primitive non-Decimal target lowers cleanly in
/// v0.7 — the numeric ↔ numeric path is wired through PTX `cvt.*`. This
/// guards the Decimal-rejection test above against silently degrading
/// into "every CAST is rejected" again if the target-side guard ever
/// over-reaches. CAST(Int32 -> Int64) must keep working even when the
/// surrounding schema also contains a Decimal column.
#[test]
fn cast_to_primitive_lowers_in_v07() {
    let provider = provider_with_decimal();
    let plan = parse_sql("SELECT CAST(id AS BIGINT) FROM t", &provider)
        .expect("CAST(Int32 AS BIGINT) must parse + type-check");
    lower_physical(&plan)
        .expect("CAST(Int32 AS BIGINT) must lower cleanly in v0.7 GPU codegen");
}

// ---- 4. Decimal128 arithmetic (v0.7 sub-task B) ----------------------------

/// Two `Decimal128(p, s)` columns sharing the same scale, used to exercise
/// the v0.7 Add/Sub/Mul lowering path.
fn provider_with_two_decimals() -> MemTableProvider {
    let schema = Schema::new(vec![
        Field {
            name: "x".into(),
            dtype: DataType::Decimal128(10, 2),
            nullable: true,
        },
        Field {
            name: "y".into(),
            dtype: DataType::Decimal128(12, 2),
            nullable: true,
        },
    ]);
    MemTableProvider::new().with_table("d", schema)
}

/// `x + y` between two `Decimal128(p, s)` columns sharing the same scale
/// lowers and resolves to `Decimal128(max(p1, p2) + 1, s)` per SQL
/// convention.
#[test]
fn decimal128_add_two_columns_lowers_in_v07() {
    let provider = provider_with_two_decimals();
    let plan = parse_sql("SELECT x + y FROM d", &provider)
        .expect("Decimal128 + Decimal128 must parse + type-check");
    let phys = lower_physical(&plan)
        .expect("Decimal128 + Decimal128 must lower in v0.7");
    let out = phys.output_schema();
    assert_eq!(out.fields.len(), 1);
    // max(10, 12) + 1 = 13, scale unchanged.
    assert_eq!(out.fields[0].dtype, DataType::Decimal128(13, 2));
}

/// Subtraction follows the same widening rule as addition.
#[test]
fn decimal128_sub_two_columns_lowers_in_v07() {
    let provider = provider_with_two_decimals();
    let plan = parse_sql("SELECT x - y FROM d", &provider)
        .expect("Decimal128 - Decimal128 must parse + type-check");
    let phys = lower_physical(&plan).expect("Decimal128 - Decimal128 must lower");
    assert_eq!(
        phys.output_schema().fields[0].dtype,
        DataType::Decimal128(13, 2)
    );
}

/// Multiplication widens precision and scale by addition:
/// `Decimal128(10, 2) * Decimal128(12, 2)` → `Decimal128(22, 4)`.
#[test]
fn decimal128_mul_two_columns_lowers_in_v07() {
    let provider = provider_with_two_decimals();
    let plan = parse_sql("SELECT x * y FROM d", &provider)
        .expect("Decimal128 * Decimal128 must parse + type-check");
    let phys = lower_physical(&plan).expect("Decimal128 * Decimal128 must lower");
    // p1 + p2 = 22, s1 + s2 = 4.
    assert_eq!(
        phys.output_schema().fields[0].dtype,
        DataType::Decimal128(22, 4)
    );
}

/// Division on Decimal128 stays rejected — the kernel has no wide-divide
/// support yet, so the host fallback is the only sound path.
#[test]
fn decimal128_div_still_rejected() {
    let provider = provider_with_two_decimals();
    // The rejection may surface from the logical type-check or the
    // physical lowering; either layer must mention Decimal128 + Div so
    // the user can find the cause.
    let r = parse_sql("SELECT x / y FROM d", &provider);
    if let Ok(plan) = r {
        let err = lower_physical(&plan).expect_err("Decimal128 Div must reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("Decimal128") && (msg.contains("Div") || msg.contains("/")),
            "Decimal128 Div rejection should mention Decimal128 + Div; got: {msg}"
        );
    } else {
        let msg = format!("{}", r.err().unwrap());
        assert!(
            msg.contains("Decimal128") && (msg.contains("Div") || msg.contains("/")),
            "Decimal128 Div rejection should mention Decimal128 + Div; got: {msg}"
        );
    }
}

/// Add between two Decimal128s with mismatched scale is rejected — the
/// SQL convention requires explicit rescale before the user can add.
#[test]
fn decimal128_add_mismatched_scale_rejected() {
    let schema = Schema::new(vec![
        Field {
            name: "a".into(),
            dtype: DataType::Decimal128(10, 2),
            nullable: true,
        },
        Field {
            name: "b".into(),
            dtype: DataType::Decimal128(10, 3),
            nullable: true,
        },
    ]);
    let provider = MemTableProvider::new().with_table("d", schema);
    // The logical or physical layer must surface a clear scale-mismatch
    // message.
    let r = parse_sql("SELECT a + b FROM d", &provider);
    let msg = match r {
        Ok(plan) => format!(
            "{}",
            lower_physical(&plan).expect_err("scale mismatch must reject")
        ),
        Err(e) => format!("{e}"),
    };
    assert!(
        msg.contains("scale") || msg.contains("Decimal128"),
        "scale-mismatch rejection should mention scale / Decimal128; got: {msg}"
    );
}
