// SPDX-License-Identifier: Apache-2.0

//! Projection pruning: stop reading columns no operator upstream needs.
//!
//! The pass walks the plan top-down threading the *set of column names the
//! parent requires*. At a [`LogicalPlan::Scan`] it narrows the scan's
//! `projection` to the intersection of that required set with the table's
//! columns — so the executor only materialises live columns. At every other
//! node it computes the columns *that node* needs from its input (its own
//! expression references plus whatever its parent still needs that the node
//! passes through) and recurses.
//!
//! ## Correctness
//!
//! Narrowing a scan's `projection` changes the scan's output schema, so it is
//! only sound when every dropped column is provably unreferenced above. The
//! required-set is derived bottom-of-parent-up from actual `Expr` column
//! references, so a column is only dropped when nothing upstream names it. The
//! overall plan output schema is therefore unchanged.
//!
//! The pass is deliberately conservative:
//!
//! * The root is seeded with *all* of its own output columns (the query
//!   result must keep every output column), so a top-level `SELECT *`-style
//!   plan prunes nothing.
//! * Nodes whose row identity depends on *all* columns — [`LogicalPlan::Distinct`]
//!   and [`LogicalPlan::Union`] — require their full input schema (dropping a
//!   column there would change which rows are considered duplicates / how
//!   branches line up), so the pass passes the full set through them.
//! * A scan is only rewritten when the pruned projection is a strict, non-empty
//!   subset of what it already reads; an empty required-set leaves the scan
//!   untouched (some executor paths need at least one column to infer row
//!   count).

use std::collections::BTreeSet;

use crate::error::BoltResult;
use crate::plan::logical_plan::{
    join_combined_schema, AggregateExpr, Expr, LogicalPlan, Schema,
};
use crate::plan::rewrite::PlanRewrite;

use super::expr_util::{collect_agg_columns, collect_columns};

/// Projection-pruning pass. See module docs.
#[derive(Debug, Default)]
pub struct ProjectionPruning;

impl PlanRewrite for ProjectionPruning {
    fn name(&self) -> &str {
        "projection-pruning"
    }

    fn rewrite(&self, plan: LogicalPlan) -> BoltResult<LogicalPlan> {
        // Seed the required set with the plan's full output schema: every
        // top-level output column must survive.
        let required: BTreeSet<String> = match plan.schema() {
            Ok(s) => s.fields.into_iter().map(|f| f.name).collect(),
            // If the plan doesn't type-check we can't safely prune; return it
            // untouched and let the normal lowering surface the error.
            Err(_) => return Ok(plan),
        };
        Ok(prune(plan, &required))
    }
}

/// Rewrite `plan` so scans read only columns in `required` (plus whatever the
/// plan's own operators consume internally). `required` is the set of output
/// column names the parent still needs from `plan`'s output.
fn prune(plan: LogicalPlan, required: &BTreeSet<String>) -> LogicalPlan {
    match plan {
        LogicalPlan::Window { input, window_exprs, partition_by, order_by } => {
            let mut child_req = required.clone();
            for e in &partition_by { add_expr_cols(e, &mut child_req); }
            for se in &order_by { add_expr_cols(&se.expr, &mut child_req); }
            for we in &window_exprs { if let Some(a) = we.func.arg() { add_expr_cols(a, &mut child_req); } }
            LogicalPlan::Window {
                input: Box::new(prune(*input, &child_req)),
                window_exprs,
                partition_by,
                order_by,
            }
        }
        LogicalPlan::Scan {
            table,
            projection,
            schema,
        } => prune_scan(table, projection, schema, required),

        LogicalPlan::Filter { input, predicate } => {
            // The filter needs its parent's columns plus its predicate's.
            let mut child_req = required.clone();
            add_expr_cols(&predicate, &mut child_req);
            LogicalPlan::Filter {
                input: Box::new(prune(*input, &child_req)),
                predicate,
            }
        }

        LogicalPlan::Project { input, exprs } => {
            // The project's output columns are its expr output names; the
            // parent's `required` is over *those* names. The input only needs
            // the columns referenced by the surviving expressions. (We keep
            // every expr — pruning unused projection *expressions* is a
            // separate transformation; here we only prune what the input
            // must supply.)
            let mut child_req = BTreeSet::new();
            for e in &exprs {
                add_expr_cols(e, &mut child_req);
            }
            LogicalPlan::Project {
                input: Box::new(prune(*input, &child_req)),
                exprs,
            }
        }

        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
        } => {
            // The aggregate reads exactly the columns named by its group keys
            // and aggregate inputs, regardless of what the parent needs (the
            // parent sees aggregate *output* names, which don't exist below).
            let mut child_req = BTreeSet::new();
            for g in &group_by {
                add_expr_cols(g, &mut child_req);
            }
            for a in &aggregates {
                add_agg_cols(a, &mut child_req);
            }
            LogicalPlan::Aggregate {
                input: Box::new(prune(*input, &child_req)),
                group_by,
                aggregates,
            }
        }

        // Row-identity-sensitive: must keep the full input schema.
        LogicalPlan::Distinct { input } => {
            let full = full_schema_cols(&input);
            LogicalPlan::Distinct {
                input: Box::new(prune(*input, &full)),
            }
        }

        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => {
            // Limit is a pass-through; propagate the parent's required set.
            LogicalPlan::Limit {
                input: Box::new(prune(*input, required)),
                limit,
                offset,
            }
        }

        LogicalPlan::Sort { input, sort_exprs } => {
            // Sort needs the parent's columns plus its sort keys.
            let mut child_req = required.clone();
            for se in &sort_exprs {
                add_expr_cols(&se.expr, &mut child_req);
            }
            LogicalPlan::Sort {
                input: Box::new(prune(*input, &child_req)),
                sort_exprs,
            }
        }

        LogicalPlan::Union { inputs } => {
            // Each branch must keep its full schema so branch alignment and
            // the result schema (taken from branch 0) are preserved.
            LogicalPlan::Union {
                inputs: inputs
                    .into_iter()
                    .map(|b| {
                        let full = full_schema_cols(&b);
                        prune(b, &full)
                    })
                    .collect(),
            }
        }

        LogicalPlan::SetOp { left, right, op, all } => {
            // EXCEPT / INTERSECT compare *whole rows* across both inputs, so
            // neither side may have columns pruned (dropping a column would
            // change the row-equality relation and the result schema). Keep
            // both sides' full schemas, mirroring the UNION rule above.
            let left_full = full_schema_cols(&left);
            let right_full = full_schema_cols(&right);
            LogicalPlan::SetOp {
                left: Box::new(prune(*left, &left_full)),
                right: Box::new(prune(*right, &right_full)),
                op,
                all,
            }
        }

        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => {
            // The join needs: the parent's required columns that resolve on a
            // side, plus every column its own `on` pairs and residual `filter`
            // reference. Attribute each required name to the side that owns it.
            //
            // Subtlety: the parent's `required` set is expressed in the join's
            // *combined* output schema, where a right-side column whose name
            // collides with a left-side name has been renamed (e.g. `a` ->
            // `right.a`) by `join_combined_schema` / `join_rename`. The child
            // schemas, however, still use the bare names. So we cannot simply
            // intersect `required` against each child schema: a renamed right
            // column (`right.a`) would match neither child and get dropped.
            //
            // Instead we rebuild the combined schema and use it as the
            // authoritative map from combined-name -> originating child. The
            // first `lschema.fields.len()` combined fields are the left child's
            // columns (always bare); the remaining ones correspond positionally
            // to the right child's fields, so combined field at index
            // `nleft + i` maps back to `rschema.fields[i].name`. Columns the
            // join itself consumes (`on` / `filter`) are named in the *child*
            // schemas (bare) and attributed directly.
            let lschema = left.schema().ok();
            let rschema = right.schema().ok();
            let (Some(lschema), Some(rschema)) = (lschema, rschema) else {
                // Can't attribute columns safely — keep both sides whole.
                return LogicalPlan::Join {
                    left,
                    right,
                    join_type,
                    on,
                    filter,
                };
            };

            let combined = join_combined_schema(&lschema, &rschema, join_type);
            let nleft = lschema.fields.len();

            let mut left_req = BTreeSet::new();
            let mut right_req = BTreeSet::new();

            // Attribute each parent-required combined-schema name back to the
            // child column it originated from.
            for (idx, field) in combined.fields.iter().enumerate() {
                if !required.contains(&field.name) {
                    continue;
                }
                if idx < nleft {
                    // Left columns are passed through with their bare name.
                    left_req.insert(field.name.clone());
                } else {
                    // Right column: recover the child's bare name positionally,
                    // undoing any `right.`/`__N` rename the combined schema
                    // applied.
                    right_req.insert(rschema.fields[idx - nleft].name.clone());
                }
            }

            // Columns the join itself consumes are referenced in the *child*
            // (bare-named) schemas via `on` / `filter`. Attribute each to
            // whichever side actually owns it.
            let mut join_cols = BTreeSet::new();
            for (l, r) in &on {
                add_expr_cols(l, &mut join_cols);
                add_expr_cols(r, &mut join_cols);
            }
            if let Some(f) = &filter {
                add_expr_cols(f, &mut join_cols);
            }
            for c in &join_cols {
                if lschema.fields.iter().any(|f| &f.name == c) {
                    left_req.insert(c.clone());
                }
                if rschema.fields.iter().any(|f| &f.name == c) {
                    right_req.insert(c.clone());
                }
            }

            // Output-schema stability under child pruning. A right-side column
            // whose bare name collides with a left-side name is RENAMED to
            // `right.<name>` by `join_combined_schema`/`join_rename`, and the
            // parent references it by that renamed name. If we pruned the
            // colliding LEFT column away, the collision — and hence the rename —
            // would vanish: the right column would re-emerge as bare `<name>`
            // and the parent's `right.<name>` reference would dangle (the exact
            // Critical bug this pass had). So retain every left column that
            // anchors a required right column's rename, even if it is otherwise
            // unused on the left. (Limitation: duplicate column names *within*
            // one side can still shift `__N` suffixes; that pathological shape
            // is not handled here.)
            let left_names: BTreeSet<String> =
                lschema.fields.iter().map(|f| f.name.clone()).collect();
            let anchors: Vec<String> = right_req
                .iter()
                .filter(|rname| left_names.contains(*rname))
                .cloned()
                .collect();
            for a in anchors {
                left_req.insert(a);
            }

            LogicalPlan::Join {
                left: Box::new(prune(*left, &left_req)),
                right: Box::new(prune(*right, &right_req)),
                join_type,
                on,
                filter,
            }
        }
    }
}

/// Narrow a scan's `projection` to the columns in `required` that exist in the
/// table, preserving the table's natural column order. Leaves the scan
/// untouched when `required` is empty or already a superset of the table's
/// columns (nothing to prune).
fn prune_scan(
    table: String,
    projection: Option<Vec<String>>,
    schema: Schema,
    required: &BTreeSet<String>,
) -> LogicalPlan {
    // The set of columns currently produced by the scan (its projection, or
    // the full schema when unprojected).
    let current: Vec<String> = match &projection {
        Some(p) => p.clone(),
        None => schema.fields.iter().map(|f| f.name.clone()).collect(),
    };
    // Keep current-order columns that are still required.
    let pruned: Vec<String> = current
        .iter()
        .filter(|c| required.contains(*c))
        .cloned()
        .collect();

    // Only rewrite when we actually drop at least one column and keep at least
    // one (an empty projection can confuse row-count inference downstream).
    if !pruned.is_empty() && pruned.len() < current.len() {
        LogicalPlan::Scan {
            table,
            projection: Some(pruned),
            schema,
        }
    } else {
        LogicalPlan::Scan {
            table,
            projection,
            schema,
        }
    }
}

/// Full output-column-name set of `plan` (used where the input schema must be
/// preserved wholesale). Empty on a type-check failure, which makes the caller
/// fall through to a no-op prune.
fn full_schema_cols(plan: &LogicalPlan) -> BTreeSet<String> {
    match plan.schema() {
        Ok(s) => s.fields.into_iter().map(|f| f.name).collect(),
        Err(_) => BTreeSet::new(),
    }
}

/// Add every column referenced by `expr` into `set`.
fn add_expr_cols(expr: &Expr, set: &mut BTreeSet<String>) {
    let mut cols = Vec::new();
    collect_columns(expr, &mut cols);
    set.extend(cols);
}

/// Add every column referenced by `agg` into `set`.
fn add_agg_cols(agg: &AggregateExpr, set: &mut BTreeSet<String>) {
    let mut cols = Vec::new();
    collect_agg_columns(agg, &mut cols);
    set.extend(cols);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{DataType, Field};
    use crate::plan::col;

    fn wide_scan() -> LogicalPlan {
        LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("a", DataType::Int64, false),
                Field::new("b", DataType::Int64, false),
                Field::new("c", DataType::Int64, false),
            ]),
        }
    }

    #[test]
    fn prunes_scan_to_referenced_columns() {
        // SELECT a FROM t  =>  scan reads only [a]
        let plan = LogicalPlan::Project {
            input: Box::new(wide_scan()),
            exprs: vec![col("a")],
        };
        let before = plan.schema().expect("typecheck");
        let out = ProjectionPruning.rewrite(plan).expect("prune");
        let after = out.schema().expect("typecheck after");
        assert_eq!(before.fields.len(), after.fields.len());
        match out {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::Scan { projection, .. } => {
                    assert_eq!(projection, Some(vec!["a".to_string()]));
                }
                other => panic!("expected Scan, got {other:?}"),
            },
            other => panic!("expected Project, got {other:?}"),
        }
    }

    #[test]
    fn keeps_filter_columns_alive() {
        // SELECT a FROM t WHERE b > 0  =>  scan reads [a, b]
        let plan = LogicalPlan::Project {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(wide_scan()),
                predicate: col("b").gt(crate::plan::lit(0_i64)),
            }),
            exprs: vec![col("a")],
        };
        let out = ProjectionPruning.rewrite(plan).expect("prune");
        let scan = match out {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::Filter { input, .. } => *input,
                other => panic!("expected Filter, got {other:?}"),
            },
            other => panic!("expected Project, got {other:?}"),
        };
        match scan {
            LogicalPlan::Scan { projection, .. } => {
                let p = projection.expect("should be pruned");
                assert_eq!(p, vec!["a".to_string(), "b".to_string()]);
                assert!(!p.contains(&"c".to_string()), "c is unused and should be dropped");
            }
            other => panic!("expected Scan, got {other:?}"),
        }
    }

    #[test]
    fn aggregate_prunes_to_keys_and_inputs() {
        // SELECT a, SUM(b) FROM t GROUP BY a  =>  scan reads [a, b]
        let plan = LogicalPlan::Aggregate {
            input: Box::new(wide_scan()),
            group_by: vec![col("a")],
            aggregates: vec![AggregateExpr::Sum(col("b"))],
        };
        let out = ProjectionPruning.rewrite(plan).expect("prune");
        match out {
            LogicalPlan::Aggregate { input, .. } => match *input {
                LogicalPlan::Scan { projection, .. } => {
                    assert_eq!(projection, Some(vec!["a".to_string(), "b".to_string()]));
                }
                other => panic!("expected Scan, got {other:?}"),
            },
            other => panic!("expected Aggregate, got {other:?}"),
        }
    }

    #[test]
    fn distinct_keeps_full_schema() {
        let plan = LogicalPlan::Distinct {
            input: Box::new(wide_scan()),
        };
        let out = ProjectionPruning.rewrite(plan).expect("prune");
        match out {
            LogicalPlan::Distinct { input } => match *input {
                // No pruning: projection stays None (full schema).
                LogicalPlan::Scan { projection, .. } => assert!(projection.is_none()),
                other => panic!("expected Scan, got {other:?}"),
            },
            other => panic!("expected Distinct, got {other:?}"),
        }
    }

    #[test]
    fn does_not_change_output_schema() {
        let plan = LogicalPlan::Project {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(wide_scan()),
                predicate: col("c").gt(crate::plan::lit(0_i64)),
            }),
            exprs: vec![col("a"), col("b")],
        };
        let before = plan.schema().expect("typecheck");
        let out = ProjectionPruning.rewrite(plan).expect("prune");
        let after = out.schema().expect("typecheck after");
        let bnames: Vec<_> = before.fields.iter().map(|f| &f.name).collect();
        let anames: Vec<_> = after.fields.iter().map(|f| &f.name).collect();
        assert_eq!(bnames, anames);
    }

    /// Build the `t1(k,a) JOIN t2(k,a)` plan used by the collision tests,
    /// selecting only the renamed right-side column `right.a`.
    fn join_collision_select_right_a() -> LogicalPlan {
        use crate::plan::logical_plan::JoinType;
        // t1(k, a), t2(k, a). SELECT right.a FROM t1 JOIN t2 ON t1.k=t2.k
        // Each table carries an unused `x` so pruning has something safe to
        // drop (proving the pass still narrows scans) while `k` (join key) and
        // `a` (collision anchor / required) must be kept.
        let left = LogicalPlan::Scan {
            table: "t1".into(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("a", DataType::Int64, false),
                Field::new("x", DataType::Int64, false),
            ]),
        };
        let right = LogicalPlan::Scan {
            table: "t2".into(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("a", DataType::Int64, false),
                Field::new("x", DataType::Int64, false),
            ]),
        };
        let join = LogicalPlan::Join {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
            // on uses child (bare) names
            on: vec![(col("k"), col("k"))],
            filter: None,
        };
        // combined schema is [k, a, right.k, right.a]; select right.a
        LogicalPlan::Project {
            input: Box::new(join),
            exprs: vec![col("right.a")],
        }
    }

    // REVIEW PROBE: join with colliding column names, select only the
    // renamed right-side column. Asserts pruning keeps the needed right
    // column (`a` in the t2 scan) and the plan still type-checks with the
    // same output schema. With the bug present, the right scan would be
    // pruned to `[k]` (dropping `a`) and the plan would fail to type-check.
    #[test]
    fn review_probe_join_collision_rename() {
        let plan = join_collision_select_right_a();
        let before = plan.schema().expect("typecheck");
        let out = ProjectionPruning.rewrite(plan).expect("prune");

        // The right (t2) scan must still produce `a` (the originating bare
        // name of the renamed combined column `right.a`).
        let right_proj = match &out {
            LogicalPlan::Project { input, .. } => match input.as_ref() {
                LogicalPlan::Join { right, .. } => match right.as_ref() {
                    LogicalPlan::Scan { projection, .. } => projection.clone(),
                    other => panic!("expected right Scan, got {other:?}"),
                },
                other => panic!("expected Join, got {other:?}"),
            },
            other => panic!("expected Project, got {other:?}"),
        };
        if let Some(p) = &right_proj {
            assert!(
                p.contains(&"a".to_string()),
                "right scan must still read `a`; got {right_proj:?}",
            );
        }

        // And the pruned plan must still type-check to the same output schema.
        let after = out.schema().expect("pruned plan must still type-check");
        let bnames: Vec<_> = before.fields.iter().map(|f| f.name.clone()).collect();
        let anames: Vec<_> = after.fields.iter().map(|f| f.name.clone()).collect();
        assert_eq!(bnames, anames, "output schema must be preserved");
    }

    #[test]
    fn join_collision_keeps_renamed_right_column() {
        // SELECT right.a FROM t1 JOIN t2 (both have column `a`): the right
        // scan must keep `a`, and the left scan must NOT keep `a` (it is
        // unreferenced on the left — only `k` is needed there for the join).
        let plan = join_collision_select_right_a();
        let out = ProjectionPruning.rewrite(plan).expect("prune");

        let (left_proj, right_proj) = match &out {
            LogicalPlan::Project { input, .. } => match input.as_ref() {
                LogicalPlan::Join { left, right, .. } => {
                    let lp = match left.as_ref() {
                        LogicalPlan::Scan { projection, .. } => projection.clone(),
                        other => panic!("expected left Scan, got {other:?}"),
                    };
                    let rp = match right.as_ref() {
                        LogicalPlan::Scan { projection, .. } => projection.clone(),
                        other => panic!("expected right Scan, got {other:?}"),
                    };
                    (lp, rp)
                }
                other => panic!("expected Join, got {other:?}"),
            },
            other => panic!("expected Project, got {other:?}"),
        };

        // Right side keeps `a` (required as `right.a`) and `k` (join key), and
        // drops the unused `x`.
        let rp = right_proj.expect("right scan should be pruned to [k, a]");
        assert!(rp.contains(&"a".to_string()), "right scan must keep `a`; got {rp:?}");
        assert!(rp.contains(&"k".to_string()), "right scan must keep `k`; got {rp:?}");
        assert!(!rp.contains(&"x".to_string()), "right scan must drop unused `x`; got {rp:?}");

        // Left side keeps `k` (join key) and `a` (the collision ANCHOR — its
        // presence is what makes the right `a` render as `right.a`; dropping it
        // would dangle the parent's `right.a` reference). It still drops the
        // genuinely-unused `x`, proving the pass narrows scans without breaking
        // the renamed output schema.
        let lp = left_proj.expect("left scan should be pruned to [k, a]");
        assert!(lp.contains(&"k".to_string()), "left scan must keep `k`; got {lp:?}");
        assert!(
            lp.contains(&"a".to_string()),
            "left scan must keep collision anchor `a`; got {lp:?}",
        );
        assert!(!lp.contains(&"x".to_string()), "left scan must drop unused `x`; got {lp:?}");
    }
}
