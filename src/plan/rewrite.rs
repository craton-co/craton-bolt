// SPDX-License-Identifier: Apache-2.0

//! Public optimizer extension surface: the [`PlanRewrite`] trait.
//!
//! A small extension point that lets out-of-tree code plug additional
//! logical-plan rewrites into the engine without forking it. The motivating
//! use case is predicate pushdown into a custom
//! [`TableProvider`](crate::plan::TableProvider): a user-provided rewriter
//! can rewrite `Filter(Scan(custom_table), pred)` into a `Scan` whose
//! `projection`/`filters` carry the pushed-down predicate, leaving the
//! built-in lowering pipeline unchanged.
//!
//! ## Where rewrites run
//!
//! Rewrites are registered with
//! [`Engine::with_rewrite`](crate::exec::Engine::with_rewrite) and run in
//! registration order, each receiving the [`LogicalPlan`] produced by the
//! previous one — a single linear pass with no fixpoint loop and no
//! per-node visitor scaffolding (implementations that need either build it
//! inside [`PlanRewrite::rewrite`]).
//!
//! Every plan the engine compiles goes through ONE pipeline (see
//! `exec::engine_query`), and the hooks sit at the same point in it for
//! every entry point:
//!
//! ```text
//! dictionary rewrite → built-in optimizer (fixpoint) → PlanRewrite hooks
//!     → subquery resolution → lower_physical → execute
//! ```
//!
//! | Entry point | Hooks see |
//! |---|---|
//! | [`Engine::sql`](crate::exec::Engine::sql), ordinary query | the whole optimized plan, once |
//! | [`Engine::run_logical_plan`](crate::exec::Engine::run_logical_plan) / [`DataFrame::collect`](crate::DataFrame::collect) | the whole optimized plan, once |
//! | [`Engine::explain_sql`](crate::exec::Engine::explain_sql) | the whole optimized plan, once (the rendered plan includes the hooks' effect) |
//! | Host-orchestrated SQL shapes (recursive CTE, LATERAL, correlated `WHERE`, `VALUES`, `generate_series`, `DISTINCT ON`, `COUNT(DISTINCT)` + `GROUP BY`) | each component subplan, every time it is compiled — e.g. a recursive term once per iteration, a LATERAL subquery once per left row |
//! | An uncorrelated scalar / `IN` subquery | only as part of its enclosing plan — the extracted subplan is NOT re-submitted, so no hook is applied to it twice |
//!
//! The engine's compiled-plan cache (which reuses prepared physical plans for
//! repeated SQL) is bypassed whenever hooks are registered, so hooks see every
//! compilation exactly as described above.
//!
//! ## Contract
//!
//! Implementations MUST preserve the plan's external semantics — the engine
//! does not re-typecheck or re-validate the rewritten plan before lowering,
//! so a buggy rewrite that produces an ill-typed plan will surface as a
//! lowering error or, worse, a wrong result. In particular, the output
//! [`Schema`](crate::plan::Schema) (field count, dtypes, and order) of the
//! returned plan should match the input's output schema for the parent
//! pipeline to keep type-checking. Returning the input unchanged is always
//! safe.
//!
//! ## Identity-rewrite example
//!
//! ```
//! use craton_bolt::plan::{PlanRewrite, LogicalPlan};
//! use craton_bolt::BoltResult;
//!
//! struct Noop;
//! impl PlanRewrite for Noop {
//!     fn name(&self) -> &str { "noop" }
//!     fn rewrite(&self, plan: LogicalPlan) -> BoltResult<LogicalPlan> {
//!         Ok(plan)
//!     }
//! }
//! # let _ = Noop.name();
//! ```

use crate::error::BoltResult;
use crate::plan::LogicalPlan;

/// Public optimizer extension point.
///
/// A `PlanRewrite` is a single-pass transformation over a [`LogicalPlan`].
/// The trait carries two methods:
///
/// * [`name`](PlanRewrite::name) — a stable identifier used in diagnostics.
///   Should be short and `kebab-case`.
/// * [`rewrite`](PlanRewrite::rewrite) — the transformation itself. Takes
///   ownership of the plan so implementations can move sub-nodes out
///   without cloning.
///
/// # Stability
///
/// Experimental until 1.0. The trait shape is small on purpose; the set of
/// [`LogicalPlan`] / [`Expr`](crate::plan::Expr) variants a hook can observe
/// grows between minor releases.
///
/// # Execution semantics
///
/// Runs after the dictionary rewrite and the built-in logical optimizer, and
/// before subquery resolution and physical lowering — identically for
/// `sql`, `run_logical_plan` / `DataFrame::collect`, `explain_sql`, and the
/// component subplans of host-orchestrated SQL shapes. See the module docs
/// for exactly which plans a hook is handed on each path.
///
/// # Threading
///
/// Called synchronously on the thread running the query (the `Engine` is
/// `!Sync`, so at most one query per engine is compiling at a time). The
/// `Send + Sync` bound lets a rewrite be shared across engines; per-rewrite
/// mutable state needs interior mutability (`rewrite` takes `&self`).
///
/// # Errors
///
/// Returning `Err` aborts planning of the current statement; the error is
/// returned from the entry point unchanged, and for `sql` /
/// `run_logical_plan` the statement counts toward both `QueriesTotal` and
/// `QueriesFailed`.
///
/// # Compatibility
///
/// Implementations must tolerate plan nodes and expressions they do not
/// recognise — pass them through unchanged rather than erroring — so a new
/// engine release that introduces a variant (or normalises a plan
/// differently in the optimizer) does not break the hook.
///
/// ## Object-safety
///
/// `PlanRewrite` is intentionally object-safe (`dyn PlanRewrite`) so it can
/// be stored alongside the engine and registered from anywhere.
///
/// ## Composition
///
/// Multiple rewrites compose by registering them in the order they should
/// run. The engine does not iterate to a fixpoint; rewrites that need a
/// fixpoint should loop internally inside `rewrite`.
pub trait PlanRewrite: Send + Sync {
    /// Stable identifier for this rewrite, used in diagnostics and future
    /// `EXPLAIN`-style output. Convention: lowercase `kebab-case`,
    /// e.g. `"predicate-pushdown"`.
    fn name(&self) -> &str;

    /// Transform `plan` into the rewritten plan. Takes ownership so
    /// implementations can move sub-nodes out without an extra clone.
    ///
    /// MUST preserve the plan's output schema (field count, dtypes, and
    /// order) and external semantics. The engine does not re-validate the
    /// returned plan before lowering — type errors here surface as
    /// lowering failures downstream.
    ///
    /// Returning `Ok(plan)` unchanged is always safe.
    fn rewrite(&self, plan: LogicalPlan) -> BoltResult<LogicalPlan>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{col, DataType, Field, LogicalPlan, Schema};

    /// Identity rewrite that swaps a `Project` for an equivalent `Project`
    /// (same input, same expression list, cloned). Exercises the trait
    /// plumbing: object-safety (`Box<dyn PlanRewrite>`), `name()` retrieval,
    /// and a `rewrite()` round-trip that returns a structurally identical
    /// plan. The test asserts on the rewritten plan's output schema rather
    /// than on `Debug` equality (the inner `Box<LogicalPlan>` allocates
    /// fresh, so pointer identity is not preserved).
    struct IdentityProjectRewrite;

    impl PlanRewrite for IdentityProjectRewrite {
        fn name(&self) -> &str {
            "identity-project"
        }

        fn rewrite(&self, plan: LogicalPlan) -> BoltResult<LogicalPlan> {
            match plan {
                LogicalPlan::Project { input, exprs } => Ok(LogicalPlan::Project {
                    input,
                    // Clone the expr list to materialise a "new" Project
                    // node — proves the rewriter actually produces a
                    // distinct value rather than just moving the input
                    // through.
                    exprs: exprs.clone(),
                }),
                other => Ok(other),
            }
        }
    }

    fn fixture_scan_project() -> LogicalPlan {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int64, false),
        ]);
        let scan = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema,
        };
        LogicalPlan::Project {
            input: Box::new(scan),
            exprs: vec![col("a"), col("b")],
        }
    }

    #[test]
    fn identity_rewrite_preserves_output_schema() {
        let r: Box<dyn PlanRewrite> = Box::new(IdentityProjectRewrite);
        assert_eq!(r.name(), "identity-project");
        let plan = fixture_scan_project();
        let before = plan.schema().expect("input plan must type-check");
        let after_plan = r.rewrite(plan).expect("identity rewrite must succeed");
        let after = after_plan.schema().expect("rewritten plan must type-check");
        assert_eq!(after.fields.len(), before.fields.len());
        for (b, a) in before.fields.iter().zip(after.fields.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.dtype, b.dtype);
        }
        // And specifically: still a Project with the same expression count.
        match after_plan {
            LogicalPlan::Project { exprs, .. } => {
                assert_eq!(exprs.len(), 2, "Project should keep both exprs");
            }
            other => panic!("expected Project, got {:?}", other),
        }
    }

    #[test]
    fn non_project_plans_pass_through_unchanged() {
        // Sanity: the identity rewrite is defined to fall through for
        // non-Project inputs, exercising the `_ => Ok(other)` arm.
        let scan = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: Schema::new(vec![Field::new("a", DataType::Int32, false)]),
        };
        let r = IdentityProjectRewrite;
        let out = r.rewrite(scan).expect("scan passes through");
        assert!(matches!(out, LogicalPlan::Scan { .. }));
    }

    /// Bonus: also confirm the trait composes via a Vec<Box<dyn ...>> — the
    /// shape Engine stores. Two identity rewrites chained still produce a
    /// well-typed plan.
    #[test]
    fn chained_rewrites_compose() {
        let chain: Vec<Box<dyn PlanRewrite>> = vec![
            Box::new(IdentityProjectRewrite),
            Box::new(IdentityProjectRewrite),
        ];
        let mut plan = fixture_scan_project();
        for r in &chain {
            plan = r.rewrite(plan).expect("each rewrite must succeed");
        }
        let s = plan.schema().expect("final plan must type-check");
        assert_eq!(s.fields.len(), 2);
    }
}
