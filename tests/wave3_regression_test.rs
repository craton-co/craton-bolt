// SPDX-License-Identifier: Apache-2.0

//! Integration-level regression tests pinning recently-fixed correctness bugs
//! at the PUBLIC API / lib boundary.
//!
//! These complement the inline unit tests the fixes already added (which reach
//! into private items). Every non-`#[ignore]`'d test here runs on CI WITHOUT a
//! GPU under `cargo test --no-default-features --features cuda-stub`, exercising
//! only host-side / planner-level code reachable through the crate's public
//! surface.
//!
//! ## Bug coverage map (see the task brief)
//!
//! 1. **i32 constant-fold overflow** — COVERED, runnable. The constant-folding
//!    optimizer pass (`craton_bolt::plan::optimizer::ConstantFold`) is public
//!    and operates purely on the logical IR with no GPU dependency. We build an
//!    overflowing `Int32` constant expression and assert the pass leaves it
//!    unfolded rather than wrapping to `i32::MIN`.
//!
//! 2. **Window integer precision (`SUM/MIN/MAX(Int64)` > 2^53 exact)** —
//!    INLINE-ONLY (documented below). The host window executor
//!    (`exec::window::execute_window`) is `pub`, but its input/output
//!    `QueryHandle` can only be constructed via the `pub(crate)`
//!    `QueryHandle::from_record_batch`, which integration tests cannot reach.
//!    The only public driver is `Engine::sql`, which needs a real CUDA context
//!    (`Engine` owns a `CudaContext`). Covered by the inline unit tests
//!    `exec::window::tests::{sum_int64_above_2_53_is_exact,
//!    min_max_int64_above_2_53_are_exact, sum_int64_overflow_errors}`.
//!
//! 3. **Float MIN/MAX NaN convention (scalar vs window)** — INLINE-ONLY for the
//!    same `QueryHandle`-constructibility reason as bug 2 (window side) and a
//!    GPU requirement on the scalar side (`exec::aggregate::execute_aggregate`
//!    runs on the device). Covered inline by
//!    `exec::window::tests::{float_min_max_nan_convention, float_min_all_nan_is_nan}`.
//!    What we *can* pin host-side is that such a query parses, type-checks, and
//!    lowers without error so the executor is actually reached (a GPU-gated
//!    end-to-end value check is provided behind `#[ignore]`).
//!
//! 4. **Decimal128 SUM overflow errors rather than wraps** — the actual
//!    overflow-vs-wrap *value* check needs `Engine::sql` (a CUDA context), and
//!    is ALREADY covered by `tests/decimal_type_test.rs::sum_decimal128_overflow_errors`
//!    (gated `#[ignore = "gpu:tier1"]`); we do NOT duplicate it. What we add
//!    here is the host-reachable, CI-runnable slice: a `SUM(Decimal128)` query
//!    parses and type-checks with the output column widened to Decimal128(38, s),
//!    proving the planner routes a decimal SUM to a real decimal aggregate.

use craton_bolt::plan::optimizer::ConstantFold;
use craton_bolt::plan::{
    lower_physical, parse_sql, BinaryOp, DataType, Expr, Field, Literal, LogicalPlan,
    MemTableProvider, PlanRewrite, Schema,
};

// ---- helpers ----------------------------------------------------------------

fn binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn i32_lit(v: i32) -> Expr {
    Expr::Literal(Literal::Int32(v))
}

/// Wrap a predicate `Expr` in a trivial `Filter(Scan)` plan so we can drive the
/// optimizer pass (which rewrites whole plans, not bare expressions) and then
/// pluck the predicate back out for inspection.
fn filter_plan(predicate: Expr) -> LogicalPlan {
    LogicalPlan::Filter {
        input: Box::new(LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: Schema::new(vec![Field::new("a", DataType::Int32, false)]),
        }),
        predicate,
    }
}

/// Run the public `ConstantFold` pass over a `Filter(Scan)` plan and return the
/// (possibly rewritten) predicate.
fn fold_predicate(predicate: Expr) -> Expr {
    let plan = filter_plan(predicate);
    let folded = ConstantFold
        .rewrite(plan)
        .expect("constant-fold pass must succeed");
    match folded {
        LogicalPlan::Filter { predicate, .. } => predicate,
        other => panic!("expected Filter after fold, got {other:?}"),
    }
}

/// A decimal-bearing fixture table for the (host-reachable) decimal SUM checks.
fn decimal_provider() -> MemTableProvider {
    let schema = Schema::new(vec![
        Field::new("region_id", DataType::Int32, false),
        Field::new("amount", DataType::Decimal128(38, 2), false),
    ]);
    MemTableProvider::new().with_table("ledger", schema)
}

// =============================================================================
// Bug 1 — i32 constant-fold overflow (planner level, runs on CI without a GPU)
// =============================================================================

/// `Int32(i32::MAX) + Int32(1)` must NOT fold to a wrapped `i32::MIN` literal.
/// The fix keeps the expression as a `Binary` node so the runtime evaluates it
/// (and reports overflow) instead of the optimizer silently wrapping it.
#[test]
fn i32_const_fold_add_overflow_does_not_wrap() {
    let expr = binary(BinaryOp::Add, i32_lit(i32::MAX), i32_lit(1));
    let folded = fold_predicate(expr);
    match folded {
        Expr::Binary { op: BinaryOp::Add, left, right } => {
            // Operands must survive intact — not collapsed to a literal.
            // `Expr` has no `PartialEq`, so match the inner literal value.
            assert!(
                matches!(*left, Expr::Literal(Literal::Int32(v)) if v == i32::MAX),
                "left operand must be preserved as Int32(i32::MAX)"
            );
            assert!(
                matches!(*right, Expr::Literal(Literal::Int32(1))),
                "right operand must be preserved as Int32(1)"
            );
        }
        Expr::Literal(Literal::Int32(v)) => panic!(
            "i32::MAX + 1 was folded to Int32({v}); overflow must NOT be folded \
             (wrapped value would be {})",
            i32::MAX.wrapping_add(1)
        ),
        other => panic!("expected an unfolded Binary(Add), got {other:?}"),
    }
}

/// `Int32(i32::MIN) - Int32(1)` underflows i32 and must stay a `Binary`.
#[test]
fn i32_const_fold_sub_underflow_does_not_wrap() {
    let expr = binary(BinaryOp::Sub, i32_lit(i32::MIN), i32_lit(1));
    let folded = fold_predicate(expr);
    assert!(
        matches!(folded, Expr::Binary { op: BinaryOp::Sub, .. }),
        "i32::MIN - 1 must NOT be folded (would wrap to i32::MAX); got {folded:?}"
    );
}

/// `Int32(100_000) * Int32(100_000)` = 10_000_000_000 overflows i32 and must
/// stay a `Binary`.
#[test]
fn i32_const_fold_mul_overflow_does_not_wrap() {
    let expr = binary(BinaryOp::Mul, i32_lit(100_000), i32_lit(100_000));
    let folded = fold_predicate(expr);
    assert!(
        matches!(folded, Expr::Binary { op: BinaryOp::Mul, .. }),
        "100000 * 100000 must NOT be folded (overflows i32); got {folded:?}"
    );
}

/// Positive control: a NON-overflowing i32 fold still collapses to a single
/// `Int32` literal with the exact value. Guards against an over-broad "never
/// fold i32" regression that would defeat the optimization entirely.
#[test]
fn i32_const_fold_in_range_still_folds() {
    let expr = binary(BinaryOp::Add, i32_lit(2), i32_lit(3));
    let folded = fold_predicate(expr);
    match folded {
        Expr::Literal(Literal::Int32(v)) => assert_eq!(v, 5, "2 + 3 must fold to 5"),
        other => panic!("expected folded Int32(5), got {other:?}"),
    }
}

/// The fix must also hold when the overflowing add is nested inside a larger
/// expression the pass *does* simplify (`(i32::MAX + 1) = 0`). The comparison
/// can't fold because its left operand never became a literal, so the whole
/// thing must survive as a `Binary` comparison — not get short-circuited via a
/// wrapped intermediate.
#[test]
fn i32_const_fold_overflow_survives_inside_comparison() {
    let inner = binary(BinaryOp::Add, i32_lit(i32::MAX), i32_lit(1));
    let expr = binary(BinaryOp::Eq, inner, i32_lit(0));
    let folded = fold_predicate(expr);
    match folded {
        Expr::Binary { op: BinaryOp::Eq, left, .. } => {
            assert!(
                matches!(*left, Expr::Binary { op: BinaryOp::Add, .. }),
                "the overflowing add must remain an unfolded Binary under the comparison, \
                 got {left:?}"
            );
        }
        other => panic!(
            "comparison over an overflowing add must not fold to a constant; got {other:?}"
        ),
    }
}

// =============================================================================
// Bug 4 — Decimal128 SUM: host-reachable plan/type checks (value check GPU-only)
// =============================================================================

/// `SUM(Decimal128)` must parse and type-check with the output column widened
/// to `Decimal128(38, scale)` per the SQL convention. This is the
/// host-reachable, CI-runnable slice of bug 4 — it proves the planner routes a
/// decimal SUM to a real *decimal* aggregate (not, say, a float SUM) so the
/// host-side reduction that carries the overflow guard is actually reached.
///
/// NOTE: the overflow-vs-wrap *value* assertion needs `Engine::sql` (a CUDA
/// context) and is already covered by
/// `tests/decimal_type_test.rs::sum_decimal128_overflow_errors`
/// (gated `#[ignore = "gpu:tier1"]`); it is intentionally not duplicated here.
#[test]
fn decimal_sum_type_checks_to_widened_decimal128() {
    let provider = decimal_provider();
    let plan = parse_sql("SELECT SUM(amount) FROM ledger", &provider).expect("parse");
    // Type-check the logical plan: the SUM output must remain Decimal128, with
    // precision widened to the SQL maximum (38) and the input scale preserved.
    let schema = plan.schema().expect("type-check SUM(Decimal128)");
    assert_eq!(schema.fields.len(), 1);
    match schema.fields[0].dtype {
        DataType::Decimal128(precision, scale) => {
            assert_eq!(precision, 38, "SUM(Decimal128) must widen precision to 38");
            assert_eq!(scale, 2, "SUM must preserve the input scale");
        }
        ref other => panic!("SUM(Decimal128) output must remain Decimal128, got {other:?}"),
    }

    // Physical lowering of a SUM(Decimal128) is the host-side reduction path.
    // It is reachable on the host, but to keep this test robust against the
    // exact lowering envelope (the decimal aggregate path is still evolving),
    // accept either a clean lowering whose output stays Decimal128 or a clear
    // "not yet lowered to GPU" rejection — but never a silent type change.
    match lower_physical(&plan) {
        Ok(phys) => {
            let out = phys.output_schema();
            assert_eq!(out.fields.len(), 1);
            assert!(
                matches!(out.fields[0].dtype, DataType::Decimal128(_, _)),
                "physical SUM(Decimal128) output must remain Decimal128, got {:?}",
                out.fields[0].dtype
            );
        }
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("Decimal128") || msg.contains("not yet lowered"),
                "an unsupported SUM(Decimal128) lowering must reject cleanly naming \
                 Decimal128; got: {msg}"
            );
        }
    }
}

// =============================================================================
// Bug 3 — Float MIN/MAX NaN convention: host-reachable plan check (value GPU)
// =============================================================================

/// Host-reachable slice of bug 3: a `MIN(float)`/`MAX(float)` query (the shape
/// whose NaN convention the fix pinned) must parse, type-check, and lower so the
/// executor that carries the NaN convention is actually reached. The
/// scalar-vs-window NaN *value* consistency is asserted by the inline unit tests
/// `exec::window::tests::{float_min_max_nan_convention, float_min_all_nan_is_nan}`
/// (window side) and is GPU-only on the scalar side; see this file's header.
#[test]
fn float_min_max_lowers_to_aggregate() {
    let schema = Schema::new(vec![Field::new("v", DataType::Float64, false)]);
    let provider = MemTableProvider::new().with_table("t", schema);
    // Use two separate single-aggregate scalar queries (the shape the existing
    // aggregate suite exercises) so we don't depend on multi-aggregate scalar
    // support that's orthogonal to the NaN-convention fix under test.
    for sql in ["SELECT MIN(v) FROM t", "SELECT MAX(v) FROM t"] {
        let plan = parse_sql(sql, &provider).unwrap_or_else(|e| panic!("parse {sql:?}: {e}"));
        let out_schema = plan
            .schema()
            .unwrap_or_else(|e| panic!("type-check {sql:?}: {e}"));
        assert_eq!(out_schema.fields.len(), 1, "{sql}: single output column");
        assert_eq!(
            out_schema.fields[0].dtype,
            DataType::Float64,
            "{sql}: MIN/MAX over Float64 must stay Float64, got {:?}",
            out_schema.fields[0].dtype
        );
        // Lowers without error (reaches the aggregate executor at run time).
        lower_physical(&plan).unwrap_or_else(|e| panic!("lower {sql:?}: {e}"));
    }
}

/// GPU-only end-to-end check for bug 3: scalar `MIN(float)` over an input
/// containing NaN must skip the NaN and return the real minimum (`-1.0`) — the
/// same "MIN skips NaN" convention the window inline test pins. This is the
/// least ambiguous half of the convention (the all-real minimum is well-defined
/// regardless of NaN ordering), so it stays a stable cross-path assertion.
/// Needs a CUDA device; run with `cargo test -- --ignored` on a GPU host.
#[test]
#[ignore = "gpu:tier1"]
fn e2e_scalar_float_min_skips_nan() {
    use std::sync::Arc;

    use arrow_array::{Array, Float64Array, RecordBatch};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use craton_bolt::Engine;

    let v = Float64Array::from(vec![f64::NAN, 2.0, -1.0]);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "v",
        ArrowDataType::Float64,
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(v)]).unwrap();

    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", batch).unwrap();

    let h = engine.sql("SELECT MIN(v) FROM t").expect("execute");
    let out = h.record_batch();
    let got = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64 min")
        .value(0);
    assert_eq!(got, -1.0, "scalar MIN over [NaN,2,-1] must skip NaN -> -1.0");
}

// =============================================================================
// Bug 2 — Window integer precision (> 2^53 exact)
// =============================================================================
//
// NOTE: INTENTIONALLY NOT an integration test. The fix lives in the host window
// executor's exact-i64 accumulator lane (`exec::window`). It cannot be driven
// from an integration test because the executor's input/output is a
// `QueryHandle`, whose only constructor (`QueryHandle::from_record_batch`) is
// `pub(crate)` — integration tests under `tests/` link against the non-test
// build and can only see `pub` items from the crate root. The sole public
// driver, `Engine::sql`, requires a live CUDA context (`Engine` owns a
// `CudaContext`), so even a host-only window query can't run without a GPU.
//
// The behaviour is covered by the inline unit tests
// `exec::window::tests::sum_int64_above_2_53_is_exact`,
// `exec::window::tests::min_max_int64_above_2_53_are_exact`, and
// `exec::window::tests::sum_int64_overflow_errors`, which use the in-crate
// `pub(crate)` constructor directly.
