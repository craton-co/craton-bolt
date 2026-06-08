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
//!   * MIN / MAX / AVG over Decimal128 (no per-group accumulator wired
//!     yet). `SUM(Decimal128)` is accepted as a host-side reduction in
//!     v0.7 — see the `sum_decimal128_*` tests below.
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

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Decimal128Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use craton_bolt::plan::{lower_physical, parse_sql, DataType, Field, MemTableProvider, Schema};
use craton_bolt::{BoltError, Engine};

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

    assert!(
        amount.dtype.is_decimal(),
        "Decimal128 must report is_decimal"
    );
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
    let phys =
        lower_physical(&plan).expect("physical lowering of a Decimal SELECT must succeed in v0.7");
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
    let plan =
        parse_sql("SELECT id FROM t", &provider).expect("SELECT of an Int32 column must parse");
    let _ = lower_physical(&plan).expect("physical lowering of a non-Decimal SELECT must succeed");
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
        // The frontend may also surface this earlier as a Type or Sql error
        // (the SQL parses, but the CAST target type is unsupported). The
        // unsupported-target arm in `sql_frontend::convert_sql_type` rejects
        // at the frontend with `BoltError::Sql("CAST target type not
        // supported: DECIMAL(..)")`. Accept any of these variants, but
        // require the message still mentions CAST / DECIMAL so the user can
        // find the source of the error.
        BoltError::Type(msg) | BoltError::Sql(msg) => {
            assert!(
                msg.contains("CAST") || msg.contains("DECIMAL") || msg.contains("Decimal128"),
                "CAST rejection should still mention CAST/DECIMAL; got: {msg}",
            );
        }
        other => panic!("expected BoltError::Plan/Type/Sql naming CAST/Decimal128, got {other:?}"),
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
    lower_physical(&plan).expect("CAST(Int32 AS BIGINT) must lower cleanly in v0.7 GPU codegen");
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
    let phys = lower_physical(&plan).expect("Decimal128 + Decimal128 must lower in v0.7");
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

/// Division on Decimal128 lowers to a Div128 kernel instruction.
/// `Decimal128(10, 2) / Decimal128(12, 2)` → `Decimal128(10, 6)`.
#[test]
fn decimal128_div_lowers_in_v07() {
    let provider = provider_with_two_decimals();
    let plan = parse_sql("SELECT x / y FROM d", &provider)
        .expect("Decimal128 / Decimal128 must parse + type-check");
    let phys = lower_physical(&plan).expect("Decimal128 / Decimal128 must lower in v0.7");
    assert_eq!(
        phys.output_schema().fields[0].dtype,
        DataType::Decimal128(10, 6)
    );
}

/// Add between two Decimal128s with mismatched scales is handled by
/// automatically rescaling the lower-scale operand before adding.
/// `Decimal128(10, 2) + Decimal128(10, 3)` → `Decimal128(11, 3)`.
#[test]
fn decimal128_add_mismatched_scale_rescales() {
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
    let plan = parse_sql("SELECT a + b FROM d", &provider)
        .expect("Decimal128(10,2) + Decimal128(10,3) must parse + type-check");
    let phys = lower_physical(&plan)
        .expect("scale-mismatched Decimal128 Add must lower via auto-rescale in v0.7");
    assert_eq!(
        phys.output_schema().fields[0].dtype,
        DataType::Decimal128(11, 3)
    );
}

// ---- 5. SUM(Decimal128) host-side reduction (v0.7) -------------------------
//
// SUM(Decimal128) is the first aggregate over Decimal128 we support. The
// path is host-side (no GPU launch): the executor walks the
// already-host-resident `Decimal128Array`, sums non-NULL `i128` values
// into a checked accumulator, and packs the result as a single-row
// `Decimal128(38, s)` array (per the SQL widening convention in
// `sum_output_dtype`). MIN / MAX / AVG stay rejected as separate scope.
//
// Below tests run end-to-end through `Engine::sql` and therefore require
// the SUM(Decimal128) code path to be reachable from the SQL frontend
// through to the host-side fold. They do not launch a GPU kernel for the
// reduction itself, but the engine plumbing (table registration,
// planning, lowering) is plan-only and works under `cuda-stub`. We mark
// them `#[ignore = "gpu:tier1"]` to match the rest of the integration
// suite's gating convention for tests that hit `Engine::sql`.

/// Build a one-column `RecordBatch` named `d` from a `Decimal128Array`.
/// Uses the Arrow-declared `(precision, scale)` on the array as the field
/// dtype, so the SUM result widens to `Decimal128(38, scale)` per the
/// engine's `sum_output_dtype` rule.
fn decimal_batch(values: Decimal128Array) -> RecordBatch {
    let dt = values.data_type().clone();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new("d", dt, true)]));
    RecordBatch::try_new(schema, vec![Arc::new(values) as ArrayRef])
        .expect("single-column Decimal batch")
}

/// Basic SUM(Decimal128) shape: a handful of small fixed-point values
/// with no NULLs. The result's `(precision, scale)` widens to
/// `(38, 2)`.
#[test]
#[ignore = "gpu:tier1"]
fn sum_decimal128_basic() {
    let arr = Decimal128Array::from(vec![
        // 1.23, -4.56, 7.89, 0.01, 100.00 (raw i128 values at scale 2)
        123_i128,
        -456_i128,
        789_i128,
        1_i128,
        10_000_i128,
    ])
    .with_precision_and_scale(18, 2)
    .expect("valid (p, s)");
    let expected: i128 = 123 - 456 + 789 + 1 + 10_000;

    let mut engine = Engine::new().expect("engine");
    engine
        .register_table("t", decimal_batch(arr))
        .expect("register");
    let handle = engine
        .sql("SELECT SUM(d) FROM t")
        .expect("SUM(Decimal) must execute");
    let out = handle.record_batch();
    assert_eq!(out.num_rows(), 1, "scalar SUM is a single-row result");

    // Schema check: SUM(Decimal128(18, 2)) widens precision to 38.
    let f = out.schema().field(0).clone();
    assert!(
        matches!(f.data_type(), ArrowDataType::Decimal128(38, 2)),
        "SUM(Decimal128) output schema must be Decimal128(38, 2); got {:?}",
        f.data_type()
    );

    let arr = out
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("Decimal128 output");
    assert!(!arr.is_null(0), "SUM over non-empty input must be non-null");
    assert_eq!(arr.value(0), expected);
}

/// SUM(Decimal128) on a column with NULL entries: NULLs are skipped.
#[test]
#[ignore = "gpu:tier1"]
fn sum_decimal128_skips_nulls() {
    let arr = Decimal128Array::from(vec![
        Some(100_i128),
        None,
        Some(-25_i128),
        None,
        Some(75_i128),
    ])
    .with_precision_and_scale(10, 2)
    .expect("valid (p, s)");
    let expected: i128 = 100 - 25 + 75;

    let mut engine = Engine::new().expect("engine");
    engine
        .register_table("t", decimal_batch(arr))
        .expect("register");
    let handle = engine
        .sql("SELECT SUM(d) FROM t")
        .expect("SUM(Decimal) must execute");
    let out = handle.record_batch();

    let arr = out
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("Decimal128 output");
    assert_eq!(arr.value(0), expected);
}

/// SUM(Decimal128) on values whose `i128` accumulator would overflow
/// must surface a clear "precision overflow" error rather than wrapping
/// silently. We use the largest representable value at precision 38
/// (`10^38 - 1`) twice — `2 * (10^38 - 1)` exceeds `i128::MAX`
/// (≈ `1.7 * 10^38`), so the second `checked_add` returns `None` and
/// the executor surfaces the overflow as a `BoltError::Type`.
#[test]
#[ignore = "gpu:tier1"]
fn sum_decimal128_overflow_errors() {
    // 10^38 - 1 — the largest value Arrow accepts at precision 38.
    // Two of them sum to `~2 * 10^38`, which exceeds `i128::MAX`
    // (≈ 1.7 * 10^38) and overflows the host accumulator.
    let big: i128 = 10_i128.pow(38) - 1;
    let arr = Decimal128Array::from(vec![big, big])
        .with_precision_and_scale(38, 0)
        .expect("10^38 - 1 fits at precision 38");

    let mut engine = Engine::new().expect("engine");
    engine
        .register_table("t", decimal_batch(arr))
        .expect("register");
    let err = engine
        .sql("SELECT SUM(d) FROM t")
        .err()
        .expect("SUM(Decimal128) on overflowing input must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("overflow") && msg.contains("Decimal128"),
        "SUM(Decimal128) overflow should name overflow + Decimal128; got: {msg}"
    );
}

// ---- 6. Decimal128 comparisons (v0.7 GPU lowering) -------------------------

/// Build a provider whose schema carries two Decimal128 columns with
/// *matching* precision + scale (so the comparison lowerer accepts) and
/// one BIGINT companion column used by the mixed-type rejection tests.
fn provider_with_matching_decimals_and_bigint() -> MemTableProvider {
    let schema = Schema::new(vec![
        Field {
            name: "d1".into(),
            dtype: DataType::Decimal128(10, 2),
            nullable: true,
        },
        Field {
            name: "d2".into(),
            dtype: DataType::Decimal128(10, 2),
            nullable: true,
        },
        Field {
            name: "n".into(),
            dtype: DataType::Int64,
            nullable: true,
        },
    ]);
    MemTableProvider::new().with_table("t", schema)
}

/// `WHERE d1 = d2` between two matching-precision matching-scale
/// Decimal128 columns lowers cleanly through the v0.7 physical-plan
/// codegen. The comparison emits `Op::Cmp128 { op: Eq, ... }` which the
/// PTX layer lowers to the split-register `setp.eq.u64` / `setp.eq.s64`
/// + `and.pred` + `selp.s32` pattern (see `ptx_gen::emit_cmp_128`).
#[test]
fn decimal128_eq_two_matching_columns_lowers_in_v07() {
    let provider = provider_with_matching_decimals_and_bigint();
    let plan = parse_sql("SELECT d1 FROM t WHERE d1 = d2", &provider)
        .expect("Decimal128 = Decimal128 must parse + type-check");
    lower_physical(&plan).expect("WHERE d1 = d2 must lower in v0.7 (Op::Cmp128 wired)");
}

/// `WHERE d1 < d2` lowers cleanly. The PTX layer emits the
/// signed-high / unsigned-low pattern (`setp.lt.s64` on the high half,
/// `setp.eq.s64` for the tie-break, `setp.lt.u64` on the low half,
/// combined with `and.pred` + `or.pred`).
#[test]
fn decimal128_lt_two_matching_columns_lowers_in_v07() {
    let provider = provider_with_matching_decimals_and_bigint();
    let plan = parse_sql("SELECT d1 FROM t WHERE d1 < d2", &provider)
        .expect("Decimal128 < Decimal128 must parse + type-check");
    lower_physical(&plan).expect("WHERE d1 < d2 must lower in v0.7 (Op::Cmp128 wired)");
}

/// Every comparison operator (`=`, `!=`, `<`, `>`, `<=`, `>=`) lowers
/// through the v0.7 codegen path. Iterates so a regression in one
/// operator surfaces cleanly without forcing six separate tests.
#[test]
fn decimal128_all_comparison_ops_lower_in_v07() {
    let provider = provider_with_matching_decimals_and_bigint();
    for op in &["=", "<>", "<", ">", "<=", ">="] {
        let sql = format!("SELECT d1 FROM t WHERE d1 {op} d2");
        let plan = parse_sql(&sql, &provider)
            .unwrap_or_else(|e| panic!("`{sql}` must parse + type-check, got: {e}"));
        lower_physical(&plan)
            .unwrap_or_else(|e| panic!("`{sql}` must lower in v0.7 (Op::Cmp128 wired), got: {e}"));
    }
}

/// Mixed Decimal128 / BIGINT comparison stays rejected: the lowering
/// has no auto-rescale path (comparing 1.00 vs `1` raises the question
/// of what scale to coerce to), and the v0.7 envelope requires the user
/// to wire an explicit CAST first. The rejection message must mention
/// Decimal128 and the matching-scale contract.
#[test]
fn decimal128_eq_bigint_rejects_in_v07() {
    let provider = provider_with_matching_decimals_and_bigint();
    let r = parse_sql("SELECT d1 FROM t WHERE d1 = n", &provider);
    let msg = match r {
        Ok(plan) => format!(
            "{}",
            lower_physical(&plan)
                .expect_err("Decimal128 = BIGINT must reject (no auto-coerce path)")
        ),
        Err(e) => format!("{e}"),
    };
    assert!(
        msg.contains("Decimal128"),
        "Decimal128 / non-Decimal comparison rejection should mention Decimal128; \
         got: {msg}"
    );
    assert!(
        msg.contains("scale") || msg.contains("CAST") || msg.contains("Decimal128"),
        "rejection should mention the matching-scale contract or CAST; got: {msg}"
    );
}

/// Decimal128 / Decimal128 with mismatched scale stays rejected. Same
/// rationale as the BIGINT case: comparing values of different scales
/// (1.00 vs 1.000) would compare different raw i128 bit-patterns.
#[test]
fn decimal128_eq_mismatched_scale_rejected_in_v07() {
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
    let r = parse_sql("SELECT a FROM d WHERE a = b", &provider);
    let msg = match r {
        Ok(plan) => format!(
            "{}",
            lower_physical(&plan).expect_err("Decimal128 cmp with mismatched scale must reject")
        ),
        Err(e) => format!("{e}"),
    };
    assert!(
        msg.contains("scale") || msg.contains("Decimal128"),
        "scale-mismatch comparison rejection should mention scale / Decimal128; \
         got: {msg}"
    );
}
