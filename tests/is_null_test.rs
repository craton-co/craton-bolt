// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `IS NULL` / `IS NOT NULL`.
//!
//! Two waves of work landed here:
//!
//! 1. **Batch 4** (host-side fallback). The planner gained `Expr::Unary
//!    { op: UnaryOp::{IsNull, IsNotNull}, operand }`, the SQL frontend
//!    lowered SQL `expr IS [NOT] NULL` into it, and the physical planner
//!    routed every such predicate through `PhysicalPlan::Filter` so the
//!    host-side `execute_filter` → `expr_agg::eval_unary` evaluator
//!    handled it. The GPU codegen still rejected Unary outright.
//!
//! 2. **Batch 5** (GPU codegen). `Op::IsNullCheck` joins the IR; the
//!    planner now lowers `column IS [NOT] NULL` (and aliased bare-column
//!    forms) into a fused projection kernel. Only compound operands
//!    (e.g. `(x + y) IS NULL`) still take the host detour.
//!
//! These tests pin the plan-level shape (no GPU needed) and the host-side
//! runtime behaviour. The GPU end-to-end paths are gated behind
//! `#[ignore]` markers so CPU-only CI stays green:
//!
//!   - `gpu:e2e`         — host-fallback compound-unary path (Batch 4).
//!   - `gpu:projection`  — fused projection-kernel path (Batch 5).

use std::sync::Arc;

use arrow_array::{Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Expr, Field, KernelSpec, MemTableProvider, Op,
    PhysicalPlan, Schema,
};

// The e2e (`#[ignore = "gpu:e2e"]`) test below uses `Int32Array::value` on
// the engine's output to verify the surviving rows; the offline tests
// build a `RecordBatch` only for the engine fixture so the type imports
// above are still needed across the file.

// ---- Fixtures --------------------------------------------------------------

fn t_schema() -> Schema {
    Schema::new(vec![
        Field {
            name: "id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "x".into(),
            dtype: DataType::Int32,
            // Provider-declared nullability. The planner uses this to
            // typecheck `x IS NULL`; the actual NULL bits live in the
            // RecordBatch validity bitmap and are inspected at runtime.
            nullable: true,
        },
    ])
}

fn t_provider() -> MemTableProvider {
    MemTableProvider::new().with_table("t", t_schema())
}

/// Small batch with NULLs only in column `x`:
///   id=1, x=10
///   id=2, x=NULL
///   id=3, x=30
///   id=4, x=NULL
///   id=5, x=50
fn t_batch() -> RecordBatch {
    let id = Int32Array::from(vec![1, 2, 3, 4, 5]);
    let x = Int32Array::from(vec![Some(10), None, Some(30), None, Some(50)]);
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int32, false),
        ArrowField::new("x", ArrowDataType::Int32, true),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(id), Arc::new(x)]).unwrap()
}

// ---- Offline: SQL -> logical -> physical shape -----------------------------

/// Batch 5: `SELECT id FROM t WHERE x IS NULL` must now lower to a
/// fused `PhysicalPlan::Projection` whose kernel carries an
/// `Op::IsNullCheck` against the nullable input column `x`. The
/// projection's `KernelSpec::input_has_validity` must flag the `x` input
/// so the launch wires the validity pointer through.
///
/// (Pre-Batch-5 this lowered to a host-side `PhysicalPlan::Filter` —
/// the comment block at the top of this file walks through the history.)
#[test]
fn is_null_lowers_to_fused_projection_with_op_is_null_check() {
    let sql = "SELECT id FROM t WHERE x IS NULL";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");

    let (kernel, table) = unwrap_projection(&phys);
    assert_eq!(table, "t", "expected scan against table `t`");

    // 1. The kernel's ops list contains exactly one IsNullCheck with
    //    want_null=true. The IsNullCheck must point at an input slot
    //    matching column `x`.
    let n_is_null_checks = kernel
        .ops
        .iter()
        .filter(|op| matches!(op, Op::IsNullCheck { want_null: true, .. }))
        .count();
    assert_eq!(
        n_is_null_checks, 1,
        "expected exactly one Op::IsNullCheck {{ want_null: true, .. }}, got {n_is_null_checks}: \
         ops = {:?}",
        kernel.ops
    );

    let validity_input = kernel
        .ops
        .iter()
        .find_map(|op| match op {
            Op::IsNullCheck {
                validity_input,
                want_null: true,
                ..
            } => Some(*validity_input),
            _ => None,
        })
        .expect("IsNullCheck present");

    let x_slot = kernel
        .inputs
        .iter()
        .position(|io| io.name == "x")
        .expect("kernel must have input slot for column x");
    assert_eq!(
        validity_input, x_slot,
        "IsNullCheck.validity_input ({validity_input}) must point at input slot for `x` ({x_slot})"
    );

    // 2. The kernel must flag the `x` input as needing validity, so the
    //    launch wires the *u8 validity pointer through.
    assert!(
        !kernel.input_has_validity.is_empty(),
        "expected input_has_validity to be populated for an IsNullCheck kernel, was empty"
    );
    assert!(
        kernel.input_has_validity[x_slot],
        "input_has_validity[{x_slot}] (x) must be true, got {:?}",
        kernel.input_has_validity
    );
}

/// Same as above but with `IS NOT NULL`: `want_null = false`.
#[test]
fn is_not_null_lowers_to_fused_projection_with_op_is_null_check() {
    let sql = "SELECT id FROM t WHERE x IS NOT NULL";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");

    let (kernel, _table) = unwrap_projection(&phys);
    let n_is_not_null_checks = kernel
        .ops
        .iter()
        .filter(|op| matches!(op, Op::IsNullCheck { want_null: false, .. }))
        .count();
    assert_eq!(
        n_is_not_null_checks, 1,
        "expected exactly one Op::IsNullCheck {{ want_null: false, .. }}, got {n_is_not_null_checks}: \
         ops = {:?}",
        kernel.ops
    );
}

/// AND-of-(IS NULL, other) is now also pushed onto the GPU: `x IS NULL`
/// is a bare-column unary and `id > 1` is a plain comparison, so the
/// whole predicate folds into the projection kernel. This pins the new
/// `predicate_contains_unary` semantics — bare-column unary does NOT
/// trigger the host fallback.
#[test]
fn is_null_anded_with_other_lowers_to_fused_projection() {
    let sql = "SELECT id FROM t WHERE x IS NULL AND id > 1";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");

    let (kernel, _table) = unwrap_projection(&phys);
    // One IsNullCheck for `x IS NULL`, one Binary{Gt} for `id > 1`,
    // one Binary{And} fusing them.
    assert!(
        kernel
            .ops
            .iter()
            .any(|op| matches!(op, Op::IsNullCheck { want_null: true, .. })),
        "expected Op::IsNullCheck in fused predicate, got {:?}",
        kernel.ops
    );
    assert!(
        kernel.predicate.is_some(),
        "expected predicate reg on fused projection, got None"
    );
}

/// Compound unary operands (e.g. `(x + 1) IS NULL`) cannot be lowered to
/// the GPU yet — `Op::IsNullCheck` only reads validity bitmaps for
/// input columns, not for arbitrary subexpressions. These predicates
/// must still take the host fallback via `PhysicalPlan::Filter`.
#[test]
fn compound_unary_operand_takes_host_fallback() {
    let sql = "SELECT id FROM t WHERE (x + 1) IS NULL";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    assert!(
        find_unary_filter_predicate(&phys).is_some(),
        "compound Unary operand must surface as a host-side PhysicalPlan::Filter, got {phys:?}"
    );
}

/// Helper: descend past any `PhysicalPlan::Project` rename layer and
/// return the underlying `Projection`'s `(KernelSpec, table)`. Panics if
/// the plan is not a (Project over) Projection.
fn unwrap_projection(plan: &PhysicalPlan) -> (&KernelSpec, &str) {
    let mut cur = plan;
    loop {
        match cur {
            PhysicalPlan::Projection { kernel, table, .. } => return (kernel, table.as_str()),
            PhysicalPlan::Project { input, .. } => cur = input.as_ref(),
            other => panic!(
                "expected PhysicalPlan::Projection (optionally under a Project rename layer), got {other:?}"
            ),
        }
    }
}

/// Walk a lowered plan looking for the first `PhysicalPlan::Filter` whose
/// predicate contains an `Expr::Unary` node, and return a clone of that
/// predicate. Returns `None` if no such Filter exists in the spine.
///
/// Used by the IS-NULL plan-shape tests above: lower() may sit a thin
/// `PhysicalPlan::Project` (SELECT-list rename / column pickout) on top
/// of the host Filter, so the test must peel that layer off before
/// reaching the Filter we care about.
fn find_unary_filter_predicate(plan: &PhysicalPlan) -> Option<Expr> {
    match plan {
        PhysicalPlan::Filter { predicate, input } => {
            if contains_unary(predicate) {
                Some(predicate.clone())
            } else {
                find_unary_filter_predicate(input)
            }
        }
        PhysicalPlan::Project { input, .. } => find_unary_filter_predicate(input),
        _ => None,
    }
}

/// Local copy of `physical_plan::predicate_contains_unary` (not exported)
/// — used by `find_unary_filter_predicate` above. Keep in sync with the
/// in-crate implementation; that one is the source of truth.
fn contains_unary(expr: &Expr) -> bool {
    match expr {
        Expr::Unary { .. } => true,
        Expr::Binary { left, right, .. } => contains_unary(left) || contains_unary(right),
        Expr::Alias(inner, _) => contains_unary(inner),
        Expr::Column(_) | Expr::Literal(_) => false,
    }
}

// ---- Host-side runtime ------------------------------------------------------
//
// The runtime correctness of `execute_filter` against an `Expr::Unary`
// predicate is pinned by unit tests inside
// `src/exec/filter.rs` (`filter_is_null_keeps_only_null_rows`,
// `filter_is_not_null_drops_only_null_rows`) — they have access to the
// `pub(crate)` `QueryHandle::from_record_batch` constructor, which the
// integration tests do not. The plan-shape tests above are sufficient to
// guard the lowering contract from this side of the crate boundary.

// ---- Online: full engine path (requires CUDA) ------------------------------

/// Batch 4 path (`gpu:e2e`): compound-unary predicates still go through
/// the host fallback. `(x + 1) IS NULL` cannot fuse into the projection
/// kernel because `Op::IsNullCheck` only reads validity for input
/// columns, not for arbitrary subexpressions.
#[test]
#[ignore = "gpu:e2e"]
fn e2e_compound_is_null_filters_through_host_path() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", t_batch()).unwrap();

    // `(x + 1) IS NULL` propagates NULL through the addition and so picks
    // the same rows as `x IS NULL` — but it forces the host detour.
    let h = engine
        .sql("SELECT id FROM t WHERE (x + 1) IS NULL")
        .expect("execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 2);
    let id_arr = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 id");
    let ids: Vec<i32> = (0..id_arr.len()).map(|i| id_arr.value(i)).collect();
    assert_eq!(ids, vec![2, 4]);
}

/// Batch 5 path (`gpu:projection`): `SELECT * FROM t WHERE x IS NULL`
/// over a nullable `Int32` column folds the `IS NULL` predicate into the
/// fused projection kernel via `Op::IsNullCheck` and returns the correct
/// surviving rows.
///
/// Marked `#[ignore]` under the `gpu:projection` bucket because it
/// requires both an available CUDA device AND the engine-side validity-
/// pointer wiring in `execute_projection` (see
/// `KernelSpec::input_has_validity`). With both in place, the host
/// filter detour is not exercised at all.
#[test]
#[ignore = "gpu:projection"]
fn e2e_is_null_filters_through_gpu_projection() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", t_batch()).unwrap();

    // Plan-shape pre-check: confirm the lowered plan is the
    // fused-projection form before exercising the device path. If the
    // routing ever regresses (e.g. someone re-introduces the host
    // detour for bare-column unary), this assert fires before the GPU
    // launch and makes the failure easy to diagnose.
    let plan = parse_sql(
        "SELECT id, x FROM t WHERE x IS NULL",
        &t_provider(),
    )
    .expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let (kernel, _) = unwrap_projection(&phys);
    assert!(
        kernel
            .ops
            .iter()
            .any(|op| matches!(op, Op::IsNullCheck { .. })),
        "expected Op::IsNullCheck in the fused projection kernel, got {:?}",
        kernel.ops
    );

    let h = engine
        .sql("SELECT id, x FROM t WHERE x IS NULL")
        .expect("execute");
    let out = h.record_batch();
    // The fused projection drops every non-NULL row of `x`.
    assert_eq!(out.num_rows(), 2);
    let id_arr = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 id");
    let ids: Vec<i32> = (0..id_arr.len()).map(|i| id_arr.value(i)).collect();
    assert_eq!(ids, vec![2, 4]);
}
