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
use crate::plan::logical_plan::{AggregateExpr, Expr, LogicalPlan, Schema};
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

            // Columns the join itself consumes.
            let mut join_cols = BTreeSet::new();
            for (l, r) in &on {
                add_expr_cols(l, &mut join_cols);
                add_expr_cols(r, &mut join_cols);
            }
            if let Some(f) = &filter {
                add_expr_cols(f, &mut join_cols);
            }

            let mut wanted = required.clone();
            wanted.extend(join_cols);

            let left_req = restrict_to_schema(&wanted, &lschema);
            let right_req = restrict_to_schema(&wanted, &rschema);

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

/// Subset of `wanted` whose names exist in `schema`.
fn restrict_to_schema(wanted: &BTreeSet<String>, schema: &Schema) -> BTreeSet<String> {
    wanted
        .iter()
        .filter(|c| schema.fields.iter().any(|f| &f.name == *c))
        .cloned()
        .collect()
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
}
