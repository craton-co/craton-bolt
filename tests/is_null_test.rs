// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the host-side `IS NULL` / `IS NOT NULL` path.
//!
//! Before the IS-NULL evaluator landed, the planner produced an
//! `Expr::Binary` shape with no Unary support, and the physical-plan
//! boundary rejected unary predicates with
//! `BoltError::Plan("IS NULL not yet supported by GPU executor; use host
//! fallback")`. The fix:
//!
//!   - Adds `Expr::Unary { op: UnaryOp::{IsNull, IsNotNull}, operand }`.
//!   - Lowers SQL `expr IS [NOT] NULL` into it from
//!     `crate::plan::sql_frontend::lower_expr`.
//!   - Routes Filter predicates that contain `Expr::Unary` through the
//!     host-side `PhysicalPlan::Filter` executor, which calls
//!     `crate::exec::filter::execute_filter` →
//!     `crate::exec::expr_agg::eval_unary`.
//!
//! These tests pin the plan-level shape (no GPU needed) and the host-side
//! runtime behaviour. A `gpu:e2e`-gated test guards the engine end-to-end
//! shape so CPU-only CI stays green.

use std::sync::Arc;

use arrow_array::{Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Expr, Field, MemTableProvider, PhysicalPlan, Schema,
    UnaryOp,
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

/// `SELECT id FROM t WHERE x IS NULL` must lower so the IS-NULL predicate
/// survives as a host-side `PhysicalPlan::Filter`. The SQL adds a
/// SELECT-list projection (`SELECT id` only projects column `id`), so the
/// final physical-plan top is either `PhysicalPlan::Project` over the
/// host Filter, or the bare host Filter if the project is identity.
/// In either case there must be a `PhysicalPlan::Filter` with an
/// `Expr::Unary { IsNull, .. }` predicate somewhere in the spine.
#[test]
fn is_null_lowers_to_host_side_filter() {
    let sql = "SELECT id FROM t WHERE x IS NULL";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");

    // Walk PhysicalPlan::Project layers to find the underlying Filter.
    let predicate = find_unary_filter_predicate(&phys).unwrap_or_else(|| {
        panic!(
            "expected a PhysicalPlan::Filter with Expr::Unary predicate \
             in the lowered plan spine, got {phys:?}"
        )
    });

    match predicate {
        Expr::Unary {
            op: UnaryOp::IsNull,
            operand,
        } => {
            assert!(
                matches!(operand.as_ref(), Expr::Column(n) if n == "x"),
                "expected IS NULL operand to be column `x`, got {operand:?}"
            );
        }
        other => panic!("expected predicate Expr::Unary {{ IsNull, .. }}, got {other:?}"),
    }
}

/// Same as above but with `IS NOT NULL`.
#[test]
fn is_not_null_lowers_to_host_side_filter() {
    let sql = "SELECT id FROM t WHERE x IS NOT NULL";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");

    let predicate = find_unary_filter_predicate(&phys).unwrap_or_else(|| {
        panic!("expected PhysicalPlan::Filter with Unary predicate in spine, got {phys:?}")
    });

    assert!(
        matches!(
            &predicate,
            Expr::Unary {
                op: UnaryOp::IsNotNull,
                operand,
            } if matches!(operand.as_ref(), Expr::Column(n) if n == "x"),
        ),
        "expected predicate `x IS NOT NULL`, got {predicate:?}"
    );
}

/// AND-of-(IS NULL, other) must still take the host detour: the presence
/// of `Expr::Unary` anywhere in the predicate is enough to force the
/// fallback path. This pins `predicate_contains_unary`'s recursive walk.
#[test]
fn is_null_anded_with_other_takes_host_fallback() {
    let sql = "SELECT id FROM t WHERE x IS NULL AND id > 1";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    assert!(
        find_unary_filter_predicate(&phys).is_some(),
        "AND'd Unary predicate must surface as a host-side PhysicalPlan::Filter, got {phys:?}"
    );
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

#[test]
#[ignore = "gpu:e2e"]
fn e2e_is_null_filters_through_engine() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", t_batch()).unwrap();

    let h = engine
        .sql("SELECT id FROM t WHERE x IS NULL")
        .expect("execute");
    let out = h.record_batch();
    // The host filter drops everything except the NULL rows.
    assert_eq!(out.num_rows(), 2);
    let id_arr = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 id");
    let ids: Vec<i32> = (0..id_arr.len()).map(|i| id_arr.value(i)).collect();
    assert_eq!(ids, vec![2, 4]);
}
