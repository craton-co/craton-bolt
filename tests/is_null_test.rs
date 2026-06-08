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
    lower_physical, parse_sql, DataType, Expr, Field, KernelSpec, Literal, MemTableProvider, Op,
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
        .filter(|op| {
            matches!(
                op,
                Op::IsNullCheck {
                    want_null: true,
                    ..
                }
            )
        })
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
        .filter(|op| {
            matches!(
                op,
                Op::IsNullCheck {
                    want_null: false,
                    ..
                }
            )
        })
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
        kernel.ops.iter().any(|op| matches!(
            op,
            Op::IsNullCheck {
                want_null: true,
                ..
            }
        )),
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

/// `WHERE id IS NULL` over a non-nullable Int32 column constant-folds at
/// codegen time: `Codegen::emit_unary` collapses to `Op::Const { Bool(false) }`
/// (and `IS NOT NULL` to `Bool(true)`), avoiding the need to wire a validity
/// pointer for a column whose GPU storage doesn't carry one. The plan is
/// still a fused projection — no host fallback, no `Op::IsNullCheck` in the
/// kernel ops.
///
/// Pins the second branch in `physical_plan::Codegen::emit_unary`. A
/// regression that emitted an unconditional `Op::IsNullCheck` would either
/// fail to launch (no validity pointer for the column) or silently return
/// wrong results when the validity slot was read past-end-of-allocation.
#[test]
fn is_null_on_non_nullable_column_folds_to_const_false() {
    // `id` is the non-nullable Int32 column in `t_schema()`.
    let sql = "SELECT id FROM t WHERE id IS NULL";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");

    let (kernel, _table) = unwrap_projection(&phys);

    // No IsNullCheck must survive — the codegen folded to Const(Bool(false)).
    let n_is_null_checks = kernel
        .ops
        .iter()
        .filter(|op| matches!(op, Op::IsNullCheck { .. }))
        .count();
    assert_eq!(
        n_is_null_checks, 0,
        "expected zero Op::IsNullCheck on a non-nullable column (constant-fold path), got {n_is_null_checks}: \
         ops = {:?}",
        kernel.ops
    );

    // Exactly one Const(Bool(false)) must show up in the ops list — the
    // predicate is now a literal false.
    let has_const_false = kernel.ops.iter().any(|op| {
        matches!(
            op,
            Op::Const {
                lit: Literal::Bool(false),
                ..
            }
        )
    });
    assert!(
        has_const_false,
        "expected a Const(Bool(false)) op from the IS NULL fold, got ops = {:?}",
        kernel.ops
    );
}

/// Symmetric: `IS NOT NULL` on a non-nullable column folds to
/// `Const(Bool(true))`. Pinned alongside `is_null_on_non_nullable_column_folds_to_const_false`
/// because the fold uses `!want_null` and a regression that flipped the
/// polarity would only show up here.
#[test]
fn is_not_null_on_non_nullable_column_folds_to_const_true() {
    let sql = "SELECT id FROM t WHERE id IS NOT NULL";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");

    let (kernel, _table) = unwrap_projection(&phys);
    let has_const_true = kernel.ops.iter().any(|op| {
        matches!(
            op,
            Op::Const {
                lit: Literal::Bool(true),
                ..
            }
        )
    });
    assert!(
        has_const_true,
        "expected a Const(Bool(true)) op from the IS NOT NULL fold, got ops = {:?}",
        kernel.ops
    );
}

/// PTX-shape coverage for the projection-kernel `Op::IsNullCheck` path
/// (Batch 5+7). The fused projection-kernel PTX for `WHERE x IS NULL` over
/// a nullable column must:
///
/// * include an `ld.global.nc.u8` of the validity byte
/// * include `setp.eq.u32` (IS NULL polarity), since `want_null=true`
/// * materialise the 0/1 Bool with `selp.s32`
/// * declare an extra `.u64` param after the value-output params (the
///   validity pointer). The param-naming convention in `ptx_gen.rs` is
///   positional (`_param_<N>`), so we assert the COUNT of `.param .u64`
///   declarations equals `inputs + outputs + 1` rather than matching the
///   specific param name.
#[test]
fn projection_kernel_ptx_is_null_shape() {
    use craton_bolt::jit::compile_ptx;

    let sql = "SELECT id FROM t WHERE x IS NULL";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let (kernel, _) = unwrap_projection(&phys);

    let ptx = compile_ptx(kernel, "bolt_kernel").expect("PTX codegen");

    assert!(
        ptx.contains("ld.global.nc.u8"),
        "expected validity-byte read `ld.global.nc.u8` in PTX, got:\n{ptx}"
    );
    assert!(
        ptx.contains("setp.eq.u32"),
        "expected IS NULL polarity `setp.eq.u32` in PTX, got:\n{ptx}"
    );
    assert!(
        ptx.contains("selp.s32"),
        "expected `selp.s32` Bool materialisation in PTX, got:\n{ptx}"
    );

    // One `.param .u64 ...` line per value-input, per value-output, plus
    // one for each flagged input-validity pointer (>= 1 here because `x`
    // is the IsNullCheck source).
    let n_u64_params = ptx.matches(".param .u64").count();
    let expected_min = kernel.inputs.len() + kernel.outputs.len() + 1;
    assert!(
        n_u64_params >= expected_min,
        "expected at least {expected_min} `.param .u64` lines (inputs {} + outputs {} + >=1 validity), \
         got {n_u64_params} in:\n{ptx}",
        kernel.inputs.len(),
        kernel.outputs.len()
    );
}

/// PTX-shape coverage for `IS NOT NULL`: same `selp.s32` and validity-byte
/// load, but the polarity must be `setp.ne.u32` so the predicate fires
/// when the byte is non-zero. Pinned separately because `want_null=false`
/// flips the comparator and a regression that swapped polarities would
/// pass the IS NULL test above but fail here.
#[test]
fn projection_kernel_ptx_is_not_null_shape() {
    use craton_bolt::jit::compile_ptx;

    let sql = "SELECT id FROM t WHERE x IS NOT NULL";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let (kernel, _) = unwrap_projection(&phys);

    let ptx = compile_ptx(kernel, "bolt_kernel").expect("PTX codegen");
    assert!(
        ptx.contains("ld.global.nc.u8"),
        "expected validity-byte read in PTX, got:\n{ptx}"
    );
    assert!(
        ptx.contains("setp.ne.u32"),
        "expected IS NOT NULL polarity `setp.ne.u32` in PTX, got:\n{ptx}"
    );
    assert!(
        ptx.contains("selp.s32"),
        "expected `selp.s32` in PTX, got:\n{ptx}"
    );
}

/// `compile_predicate_kernel` (the scan-kernel-form predicate-only PTX)
/// also lowers `Op::IsNullCheck`. Pins the matching wire shape for the
/// `scan_kernel` path so a regression that taught `ptx_gen.rs` but not
/// `scan_kernel.rs` would surface here.
#[test]
fn scan_kernel_ptx_is_null_shape() {
    use craton_bolt::jit::scan_kernel::compile_predicate_kernel;

    let sql = "SELECT id FROM t WHERE x IS NULL";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let (kernel, _) = unwrap_projection(&phys);

    // The scan kernel only makes sense when the predicate path is active.
    assert!(
        kernel.predicate.is_some(),
        "fixture must have a predicate; got ops = {:?}",
        kernel.ops
    );

    let ptx = compile_predicate_kernel(kernel, "bolt_predicate").expect("scan_kernel PTX codegen");

    assert!(
        ptx.contains("ld.global.nc.u8"),
        "expected validity-byte read in scan-kernel PTX, got:\n{ptx}"
    );
    assert!(
        ptx.contains("setp.eq.u32"),
        "expected IS NULL polarity `setp.eq.u32` in scan-kernel PTX, got:\n{ptx}"
    );
    assert!(
        ptx.contains("selp.s32"),
        "expected `selp.s32` Bool materialisation in scan-kernel PTX, got:\n{ptx}"
    );

    // Validity param count: scan_kernel adds N input pointers + 1 mask
    // output + K validity pointers. With `x IS NULL` (one flagged input),
    // K == 1.
    let n_u64_params = ptx.matches(".param .u64").count();
    let expected = kernel.inputs.len() + 1 + 1;
    assert_eq!(
        n_u64_params, expected,
        "expected exactly {expected} `.param .u64` lines (inputs {} + mask 1 + validity 1), got {n_u64_params} in:\n{ptx}",
        kernel.inputs.len()
    );
}

/// `WHERE NOT (id > 1)` now lowers `NOT` to the GPU: the comparison
/// `id > 1` emits a Bool register and `Op::Not` negates it. The whole
/// predicate folds into the fused projection kernel — no host fallback.
/// Pins the new `Codegen::emit_unary` NOT arm + `predicate_contains_unary`
/// recursion (a regression that re-forced the host path for NOT would
/// fail to find a `Projection` here).
#[test]
fn not_comparison_lowers_to_fused_projection_with_op_not() {
    let sql = "SELECT id FROM t WHERE NOT (id > 1)";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");

    let (kernel, _table) = unwrap_projection(&phys);

    let n_not = kernel
        .ops
        .iter()
        .filter(|op| matches!(op, Op::Not { .. }))
        .count();
    assert_eq!(
        n_not, 1,
        "expected exactly one Op::Not for `NOT (id > 1)`, got {n_not}: ops = {:?}",
        kernel.ops
    );
    assert!(
        kernel.predicate.is_some(),
        "expected predicate reg on fused projection, got None"
    );
}

/// `NOT ((x + 1) IS NULL)` must still take the host fallback: the operand
/// `(x + 1) IS NULL` is a compound-unary the GPU can't lower, so
/// `predicate_contains_unary` recurses into it and reports `true`.
/// Pins that the NOT recursion does NOT blindly accept every operand.
#[test]
fn not_over_compound_unary_takes_host_fallback() {
    let sql = "SELECT id FROM t WHERE NOT ((x + 1) IS NULL)";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    assert!(
        find_unary_filter_predicate(&phys).is_some(),
        "NOT over a compound-unary operand must surface as a host-side \
         PhysicalPlan::Filter, got {phys:?}"
    );
}

/// PTX-shape coverage for the projection-kernel `Op::Not` path. The fused
/// projection PTX for `WHERE NOT (id > 1)` over a non-nullable column must
/// contain the inner comparison (`setp.gt.s64`, since `1` widens to Int64)
/// and the negation (`xor.b32`).
#[test]
fn projection_kernel_ptx_not_shape() {
    use craton_bolt::jit::compile_ptx;

    let sql = "SELECT id FROM t WHERE NOT (id > 1)";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let (kernel, _) = unwrap_projection(&phys);

    let ptx = compile_ptx(kernel, "bolt_kernel").expect("PTX codegen");
    assert!(
        ptx.contains("setp.gt.s64"),
        "expected inner `id > 1` comparison `setp.gt.s64` in PTX, got:\n{ptx}"
    );
    assert!(
        ptx.contains("xor.b32"),
        "expected `xor.b32` from the NOT negation in PTX, got:\n{ptx}"
    );
}

/// Matching scan-kernel (predicate-only) PTX path for `Op::Not`. A
/// regression that taught `ptx_gen.rs` but not `scan_kernel.rs` to emit
/// the negation would surface here.
#[test]
fn scan_kernel_ptx_not_shape() {
    use craton_bolt::jit::scan_kernel::compile_predicate_kernel;

    let sql = "SELECT id FROM t WHERE NOT (id > 1)";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let (kernel, _) = unwrap_projection(&phys);

    assert!(
        kernel.predicate.is_some(),
        "fixture must have a predicate; got ops = {:?}",
        kernel.ops
    );

    let ptx = compile_predicate_kernel(kernel, "bolt_predicate").expect("scan_kernel PTX codegen");
    assert!(
        ptx.contains("setp.gt.s64"),
        "expected inner comparison `setp.gt.s64` in scan-kernel PTX, got:\n{ptx}"
    );
    assert!(
        ptx.contains("xor.b32"),
        "expected `xor.b32` from the NOT negation in scan-kernel PTX, got:\n{ptx}"
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
        Expr::Case {
            branches,
            else_branch,
        } => {
            branches
                .iter()
                .any(|(w, t)| contains_unary(w) || contains_unary(t))
                || else_branch.as_deref().is_some_and(contains_unary)
        }
        Expr::Cast { expr, .. } | Expr::CastFormat { expr, .. } => contains_unary(expr),
        Expr::ScalarFn { args, .. } => args.iter().any(contains_unary),
        // LIKE's operand can itself contain a unary; recurse into it (the
        // pattern is a literal). Mirrors the wrapper-recursion of the arms
        // above for the purpose of unary detection.
        Expr::Like { expr, .. } => contains_unary(expr),
        // Date scalar fns wrap an inner operand but carry no unary themselves;
        // they have no GPU unary-over-non-column shape, so they don't force the
        // host path on their own. Mirror `predicate_contains_unary` (false).
        Expr::Extract { .. } | Expr::DateTrunc { .. } => false,
        // Subqueries have no GPU path; the `InSubquery` probe lives in this
        // query's namespace, so recurse into it (matches the in-crate helper).
        Expr::ScalarSubquery(_) => false,
        Expr::InSubquery { expr, .. } => contains_unary(expr),
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
    let plan = parse_sql("SELECT id, x FROM t WHERE x IS NULL", &t_provider()).expect("parse");
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
