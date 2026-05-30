// SPDX-License-Identifier: Apache-2.0

//! Predicate pushdown: move `WHERE` conjuncts as close to the data source as
//! correctness allows.
//!
//! The pass splits each [`LogicalPlan::Filter`] predicate into its top-level
//! `AND` conjuncts and pushes each one independently below the filter's input
//! when it is provably safe:
//!
//! * **through `Project`** — only conjuncts whose every referenced column is a
//!   *pass-through* output column (a bare `Column` or `Alias(Column, _)` in the
//!   projection list, i.e. a column that exists unchanged below the project).
//!   Such a conjunct is rewritten to reference the underlying input column and
//!   pushed below the project. Conjuncts referencing computed columns stay put.
//! * **through `Sort`, `Distinct`** — always safe (these are row-preserving and
//!   value-preserving; a row passes the filter iff it would after sort/dedup).
//! * **through `Union` / `SetOp`** — the whole predicate is replicated into
//!   every branch: `Filter(p, Union(A, B))` => `Union(Filter(p', A),
//!   Filter(p', B))`, and likewise for `EXCEPT` / `INTERSECT`. Because the
//!   predicate is deterministic and evaluated row-wise, filtering the rows of
//!   each branch *before* the set/multiset op produces exactly the rows that
//!   survive filtering *after* it (see the soundness note on [`push_into_union`]
//!   / [`push_into_setop`]). The filter's column references are by the op's
//!   *output* schema (which UNION/SetOp take from their first / left branch),
//!   while a non-leading branch may use *different* column names for the same
//!   positions; the predicate is therefore remapped to each branch's schema
//!   **by position** before being attached — see [`remap_to_branch`].
//! * **into `Join`** — a conjunct referencing only the left side becomes a
//!   `Filter` on the left input; only the right side, a `Filter` on the right
//!   input. Conjuncts referencing *both* sides remain above the join (the
//!   separate [`crate::plan::optimizer::filter_into_join`] pass folds those
//!   into the join residual). For OUTER joins, pushing a filter into the
//!   *non-preserved* side would change NULL-padding semantics, so it is only
//!   pushed into the preserved side(s) — see [`can_push_into_join_side`].
//!
//! `Limit` and `Aggregate` are *not* descended through for pushdown: a filter
//! below a `Limit` changes which rows survive the limit, and a filter
//! referencing aggregate outputs cannot move below the aggregate. In those
//! cases the conjunct is kept above the node.
//!
//! Every conjunct that cannot be pushed is re-attached in a single `Filter`
//! immediately above the (possibly rewritten) input, so the output schema and
//! semantics are unchanged.

use crate::error::BoltResult;
use crate::plan::logical_plan::{Expr, JoinType, LogicalPlan, Schema, SetOpKind};
use crate::plan::rewrite::PlanRewrite;

use super::expr_util::{collect_columns, combine_conjuncts, split_conjuncts};

/// Predicate-pushdown pass. See module docs.
#[derive(Debug, Default)]
pub struct PredicatePushdown;

impl PlanRewrite for PredicatePushdown {
    fn name(&self) -> &str {
        "predicate-pushdown"
    }

    fn rewrite(&self, plan: LogicalPlan) -> BoltResult<LogicalPlan> {
        Ok(push_plan(plan))
    }
}

/// Recursively rewrite `plan`, pushing predicates down where safe.
fn push_plan(plan: LogicalPlan) -> LogicalPlan {
    // First recurse into children so a nested filter is normalised before its
    // parent considers it.
    let plan = recurse_children(plan);
    match plan {
        LogicalPlan::Filter { input, predicate } => push_filter(*input, predicate),
        other => other,
    }
}

/// Rebuild a plan node with each child rewritten by [`push_plan`], without
/// otherwise changing the node. `Filter` is handled by the caller, so this
/// only needs to cover the remaining variants' children.
fn recurse_children(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        LogicalPlan::Window { input, window_exprs, partition_by, order_by } => LogicalPlan::Window {
            input: Box::new(push_plan(*input)),
            window_exprs,
            partition_by,
            order_by,
        },
        LogicalPlan::Scan { .. } => plan,
        LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
            input: Box::new(push_plan(*input)),
            predicate,
        },
        LogicalPlan::Project { input, exprs } => LogicalPlan::Project {
            input: Box::new(push_plan(*input)),
            exprs,
        },
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
        } => LogicalPlan::Aggregate {
            input: Box::new(push_plan(*input)),
            group_by,
            aggregates,
        },
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
            input: Box::new(push_plan(*input)),
        },
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => LogicalPlan::Limit {
            input: Box::new(push_plan(*input)),
            limit,
            offset,
        },
        LogicalPlan::Sort { input, sort_exprs } => LogicalPlan::Sort {
            input: Box::new(push_plan(*input)),
            sort_exprs,
        },
        LogicalPlan::Union { inputs } => LogicalPlan::Union {
            inputs: inputs.into_iter().map(push_plan).collect(),
        },
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => LogicalPlan::Join {
            left: Box::new(push_plan(*left)),
            right: Box::new(push_plan(*right)),
            join_type,
            on,
            filter,
        },
        // EXCEPT / INTERSECT: no own predicate to push; recurse into inputs.
        LogicalPlan::SetOp { left, right, op, all } => LogicalPlan::SetOp {
            left: Box::new(push_plan(*left)),
            right: Box::new(push_plan(*right)),
            op,
            all,
        },
    }
}

/// Push the conjuncts of `predicate` below `input` where possible, returning
/// the rewritten subtree (which may or may not still have a `Filter` on top).
fn push_filter(input: LogicalPlan, predicate: Expr) -> LogicalPlan {
    let mut conjuncts = Vec::new();
    split_conjuncts(predicate, &mut conjuncts);
    push_conjuncts(input, conjuncts)
}

/// Attempt to push every conjunct in `conjuncts` below `input`. Conjuncts that
/// cannot move are re-wrapped in a `Filter` above the rewritten input.
fn push_conjuncts(input: LogicalPlan, conjuncts: Vec<Expr>) -> LogicalPlan {
    match input {
        // Filter below filter: merge the conjunct lists and retry against the
        // grandchild — this flattens `Filter(Filter(x))` so pushdown reaches
        // through both layers.
        LogicalPlan::Filter {
            input: inner,
            predicate,
        } => {
            let mut merged = conjuncts;
            split_conjuncts(predicate, &mut merged);
            push_conjuncts(*inner, merged)
        }

        // Through a projection: split conjuncts into ones referencing only
        // pass-through columns (rewritten + pushed) and the rest (kept above).
        LogicalPlan::Project { input: proj_in, exprs } => {
            let passthrough = passthrough_map(&exprs);
            let mut pushable = Vec::new();
            let mut kept = Vec::new();
            for c in conjuncts {
                match rewrite_through_project(&c, &passthrough) {
                    Some(rewritten) => pushable.push(rewritten),
                    None => kept.push(c),
                }
            }
            let new_input = if pushable.is_empty() {
                *proj_in
            } else {
                push_conjuncts(*proj_in, pushable)
            };
            let project = LogicalPlan::Project {
                input: Box::new(new_input),
                exprs,
            };
            wrap_filter(project, kept)
        }

        // Through value/row-preserving wrappers: push everything below.
        LogicalPlan::Sort { input: inner, sort_exprs } => {
            let pushed = push_conjuncts(*inner, conjuncts);
            LogicalPlan::Sort {
                input: Box::new(pushed),
                sort_exprs,
            }
        }
        LogicalPlan::Distinct { input: inner } => {
            let pushed = push_conjuncts(*inner, conjuncts);
            LogicalPlan::Distinct {
                input: Box::new(pushed),
            }
        }

        // Through a UNION ALL: replicate the (per-branch remapped) predicate
        // into every branch.
        LogicalPlan::Union { inputs } => push_into_union(inputs, conjuncts),

        // Through EXCEPT / INTERSECT (incl. their ALL variants): replicate the
        // (per-branch remapped) predicate into both inputs.
        LogicalPlan::SetOp { left, right, op, all } => {
            push_into_setop(left, right, op, all, conjuncts)
        }

        // Into a join: route single-side conjuncts to the owning input.
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => push_into_join(left, right, join_type, on, filter, conjuncts),

        // Everything else: cannot push; keep a filter above.
        other => wrap_filter(other, conjuncts),
    }
}

/// Push conjuncts into a join. Single-side conjuncts that are safe given the
/// join type sink into the owning input; the rest stay above the join.
fn push_into_join(
    left: Box<LogicalPlan>,
    right: Box<LogicalPlan>,
    join_type: JoinType,
    on: Vec<(Expr, Expr)>,
    filter: Option<Expr>,
    conjuncts: Vec<Expr>,
) -> LogicalPlan {
    // Resolve child schemas to attribute each conjunct to a side. If either
    // side fails to type-check (shouldn't happen on a valid plan) we keep all
    // conjuncts above to stay safe.
    let (lschema, rschema) = match (left.schema(), right.schema()) {
        (Ok(l), Ok(r)) => (l, r),
        _ => {
            let join = LogicalPlan::Join {
                left,
                right,
                join_type,
                on,
                filter,
            };
            return wrap_filter(join, conjuncts);
        }
    };

    let mut to_left = Vec::new();
    let mut to_right = Vec::new();
    let mut kept = Vec::new();
    for c in conjuncts {
        let side = classify_side(&c, &lschema, &rschema);
        match side {
            Side::Left if can_push_into_join_side(join_type, JoinSide::Left) => to_left.push(c),
            Side::Right if can_push_into_join_side(join_type, JoinSide::Right) => to_right.push(c),
            _ => kept.push(c),
        }
    }

    let new_left = if to_left.is_empty() {
        left
    } else {
        Box::new(push_conjuncts(*left, to_left))
    };
    let new_right = if to_right.is_empty() {
        right
    } else {
        Box::new(push_conjuncts(*right, to_right))
    };
    let join = LogicalPlan::Join {
        left: new_left,
        right: new_right,
        join_type,
        on,
        filter,
    };
    wrap_filter(join, kept)
}

/// Push `conjuncts` into every branch of a `UNION ALL`.
///
/// **Soundness.** A `UNION ALL` is a row-wise concatenation: the bag of output
/// rows is the multiset sum of the branch bags. A deterministic, row-wise
/// predicate `p` partitions rows independently of which branch they came from,
/// so `σ_p(A ⊎ B) = σ_p(A) ⊎ σ_p(B)`. Replicating `p` into each branch is
/// therefore exact (no rows gained or lost), with the only subtlety being
/// column-name alignment, handled by [`remap_to_branch`].
///
/// If the predicate cannot be remapped onto a branch (a malformed plan whose
/// branch schemas don't line up, which schema validation would already reject),
/// we leave the whole filter above the union rather than emit an invalid plan.
fn push_into_union(inputs: Vec<LogicalPlan>, conjuncts: Vec<Expr>) -> LogicalPlan {
    // The union's output schema is its first branch's schema; the filter's
    // column references are resolved against that. Each branch is remapped from
    // those output names to the branch's own names by position.
    let out_schema = match inputs.first().map(LogicalPlan::schema) {
        Some(Ok(s)) => s,
        // Empty union (rejected by schema validation) or an un-typecheckable
        // first branch: stay safe and keep the filter above.
        _ => return wrap_filter(LogicalPlan::Union { inputs }, conjuncts),
    };

    // Pre-flight: every branch must accept the remap. If any branch can't be
    // remapped, abandon the pushdown wholesale and keep the filter above.
    let mut remapped: Vec<Vec<Expr>> = Vec::with_capacity(inputs.len());
    for branch in &inputs {
        match remap_to_branch(&conjuncts, &out_schema, branch) {
            Some(cs) => remapped.push(cs),
            None => return wrap_filter(LogicalPlan::Union { inputs }, conjuncts),
        }
    }

    let new_inputs = inputs
        .into_iter()
        .zip(remapped)
        .map(|(branch, cs)| push_conjuncts(branch, cs))
        .collect();
    LogicalPlan::Union { inputs: new_inputs }
}

/// Push `conjuncts` into both inputs of an `EXCEPT` / `INTERSECT` node.
///
/// **Soundness, per variant** (let `lc(r)` / `rc(r)` be a row's multiplicity in
/// the left / right input, and `p` the deterministic row-wise predicate):
///
/// * `INTERSECT` (set) — output is the set of rows with `lc > 0 ∧ rc > 0`.
///   Filtering removes rows failing `p` from both inputs identically, and a row
///   passing `p` keeps its presence on each side, so the surviving-on-both set
///   is unchanged. Pushing `p` into both sides is exact.
/// * `INTERSECT ALL` (multiset) — output multiplicity is `min(lc, rc)`. For a
///   row failing `p`, both `lc` and `rc` drop to 0 → `min` is 0 either way; for
///   a row passing `p`, both counts are preserved → `min` unchanged. Exact.
/// * `EXCEPT` (set) — output is rows with `lc > 0 ∧ rc == 0`. `p` is applied to
///   both sides, so a row's presence/absence on each side is preserved for rows
///   passing `p`, and rows failing `p` vanish from both (and from the output).
///   Exact.
/// * `EXCEPT ALL` (multiset) — output multiplicity is `max(0, lc - rc)`. A row
///   failing `p` has `lc = rc = 0` → `max(0, 0) = 0`; a row passing `p` keeps
///   both counts → `max(0, lc - rc)` unchanged. Exact.
///
/// In every case the predicate is deterministic and evaluated per row, so it
/// commutes with the multiset arithmetic above — pushing into both branches is
/// sound for all four (`EXCEPT`, `EXCEPT ALL`, `INTERSECT`, `INTERSECT ALL`).
/// Column-name alignment is handled by [`remap_to_branch`]; if either branch
/// can't be remapped we keep the filter above the node.
fn push_into_setop(
    left: Box<LogicalPlan>,
    right: Box<LogicalPlan>,
    op: SetOpKind,
    all: bool,
    conjuncts: Vec<Expr>,
) -> LogicalPlan {
    // The result schema is the left input's; the right branch may name the same
    // positions differently and so is remapped by position too.
    let out_schema = match left.schema() {
        Ok(s) => s,
        Err(_) => {
            let set_op = LogicalPlan::SetOp { left, right, op, all };
            return wrap_filter(set_op, conjuncts);
        }
    };

    let left_conjuncts = remap_to_branch(&conjuncts, &out_schema, &left);
    let right_conjuncts = remap_to_branch(&conjuncts, &out_schema, &right);
    let (left_conjuncts, right_conjuncts) = match (left_conjuncts, right_conjuncts) {
        (Some(l), Some(r)) => (l, r),
        // A branch we can't remap (malformed plan): keep the filter above.
        _ => {
            let set_op = LogicalPlan::SetOp { left, right, op, all };
            return wrap_filter(set_op, conjuncts);
        }
    };

    let new_left = Box::new(push_conjuncts(*left, left_conjuncts));
    let new_right = Box::new(push_conjuncts(*right, right_conjuncts));
    LogicalPlan::SetOp {
        left: new_left,
        right: new_right,
        op,
        all,
    }
}

/// Rewrite each conjunct in `conjuncts` from the set/union *output* schema
/// (`out_schema`) onto `branch`'s own schema **by position**, returning the
/// remapped conjuncts.
///
/// UNION / SetOp align their inputs positionally and take the output column
/// *names* from the leading (first / left) branch, so a non-leading branch may
/// use a different name for the same column position. The filter's references
/// are by the output names, so we map each output name to the branch name that
/// occupies the same index and rewrite the predicate accordingly (reusing
/// [`rename_columns`]).
///
/// Returns `None` (caller leaves the filter above the node) if the branch can't
/// be type-checked or has fewer fields than the output schema — i.e. when a
/// referenced output position has no counterpart in the branch. Such a plan is
/// already rejected by schema validation; the guard is defensive so we never
/// synthesise a column reference the branch can't resolve.
fn remap_to_branch(
    conjuncts: &[Expr],
    out_schema: &Schema,
    branch: &LogicalPlan,
) -> Option<Vec<Expr>> {
    let branch_schema = branch.schema().ok()?;
    // A well-formed union/set-op guarantees equal field counts; bail out
    // defensively if the branch is short so we never index past its fields.
    if branch_schema.fields.len() < out_schema.fields.len() {
        return None;
    }

    // Output-name -> branch-name, position by position. When the names already
    // match (the common case, and always so for the leading branch) the entry
    // is an identity and `rename_columns` leaves the reference untouched.
    let mut map = std::collections::HashMap::new();
    for (out_field, branch_field) in out_schema.fields.iter().zip(branch_schema.fields.iter()) {
        map.insert(out_field.name.clone(), branch_field.name.clone());
    }

    Some(conjuncts.iter().map(|c| rename_columns(c, &map)).collect())
}

/// Which join input a conjunct's columns belong to.
enum Side {
    Left,
    Right,
    /// References both sides, neither, or an unresolvable column.
    Both,
}

/// Side of a join, for [`can_push_into_join_side`].
#[derive(Clone, Copy)]
enum JoinSide {
    Left,
    Right,
}

/// Attribute `expr` to a join side by its referenced columns. A conjunct that
/// references columns from exactly one side is attributed to that side; one
/// referencing both (or columns resolvable in neither) is [`Side::Both`].
///
/// Column attribution uses the *child* schemas (pre-rename), so a column that
/// the join's combined schema would have renamed `right.x` is matched against
/// the right child's bare `x`. We only push a conjunct when it is *unambiguous*
/// — present on one side and absent from the other — to avoid mis-routing a
/// reference to a renamed/duplicated name.
fn classify_side(expr: &Expr, left: &Schema, right: &Schema) -> Side {
    let mut cols = Vec::new();
    collect_columns(expr, &mut cols);
    if cols.is_empty() {
        // A constant conjunct (e.g. a folded literal) is safe to push to
        // either side; route it left arbitrarily.
        return Side::Left;
    }
    let in_left = |c: &String| left.fields.iter().any(|f| &f.name == c);
    let in_right = |c: &String| right.fields.iter().any(|f| &f.name == c);
    let all_left = cols.iter().all(|c| in_left(c) && !in_right(c));
    let all_right = cols.iter().all(|c| in_right(c) && !in_left(c));
    if all_left {
        Side::Left
    } else if all_right {
        Side::Right
    } else {
        Side::Both
    }
}

/// Whether a single-side conjunct can be pushed into the given side of a join
/// of `join_type` without changing semantics.
///
/// Pushing a filter into the *preserved* side of an outer join is always safe
/// (those rows are emitted regardless of match, so filtering them before the
/// join yields the same result). Pushing into the *non-preserved* side is
/// **not** safe: it would drop rows that should have been NULL-padded, turning
/// e.g. a LEFT join into something closer to an INNER join on that predicate.
/// INNER and CROSS preserve neither side's NULL-padding semantics, so both
/// sides may receive pushed filters.
fn can_push_into_join_side(join_type: JoinType, side: JoinSide) -> bool {
    match join_type {
        // INNER / CROSS: filtering either input before the join is equivalent.
        JoinType::Inner | JoinType::Cross => true,
        // LEFT preserves the left side; a right-side filter must stay above.
        JoinType::LeftOuter => matches!(side, JoinSide::Left),
        // RIGHT preserves the right side; a left-side filter must stay above.
        JoinType::RightOuter => matches!(side, JoinSide::Right),
        // FULL preserves both — neither side may have a filter pushed in.
        JoinType::FullOuter => false,
    }
}

/// Build a map from a projection's *output* column name to the underlying
/// input column name, for pass-through projections only. A pass-through entry
/// is a bare `Column(c)` (output name `c` -> input `c`) or
/// `Alias(Column(c), out)` (output name `out` -> input `c`). Computed
/// expressions are not entered, so a conjunct referencing them won't push.
fn passthrough_map(exprs: &[Expr]) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    for e in exprs {
        match e {
            Expr::Column(c) => {
                map.insert(c.clone(), c.clone());
            }
            Expr::Alias(inner, out) => {
                if let Expr::Column(c) = inner.as_ref() {
                    map.insert(out.clone(), c.clone());
                }
            }
            _ => {}
        }
    }
    map
}

/// If every column in `expr` is a pass-through projection output, return a
/// copy of `expr` with those references rewritten to the underlying input
/// column names (so it can be pushed below the project). Otherwise `None`.
fn rewrite_through_project(
    expr: &Expr,
    passthrough: &std::collections::HashMap<String, String>,
) -> Option<Expr> {
    let mut cols = Vec::new();
    collect_columns(expr, &mut cols);
    if !cols.iter().all(|c| passthrough.contains_key(c)) {
        return None;
    }
    Some(rename_columns(expr, passthrough))
}

/// Deep-copy `expr`, renaming any `Column(c)` to `Column(map[c])` when present.
fn rename_columns(
    expr: &Expr,
    map: &std::collections::HashMap<String, String>,
) -> Expr {
    match expr {
        Expr::Extract { field, expr } => Expr::Extract { field: *field, expr: Box::new(rename_columns(expr, map)) },
        Expr::DateTrunc { unit, expr } => Expr::DateTrunc { unit: *unit, expr: Box::new(rename_columns(expr, map)) },
        Expr::InSubquery { expr, subquery, negated } => Expr::InSubquery { expr: Box::new(rename_columns(expr, map)), subquery: subquery.clone(), negated: *negated },
        Expr::ScalarSubquery(_) => expr.clone(),
        Expr::Column(c) => Expr::Column(map.get(c).cloned().unwrap_or_else(|| c.clone())),
        Expr::Literal(_) => expr.clone(),
        Expr::Binary { op, left, right } => Expr::Binary {
            op: *op,
            left: Box::new(rename_columns(left, map)),
            right: Box::new(rename_columns(right, map)),
        },
        Expr::Unary { op, operand } => Expr::Unary {
            op: *op,
            operand: Box::new(rename_columns(operand, map)),
        },
        Expr::Case {
            branches,
            else_branch,
        } => Expr::Case {
            branches: branches
                .iter()
                .map(|(w, t)| (rename_columns(w, map), rename_columns(t, map)))
                .collect(),
            else_branch: else_branch
                .as_ref()
                .map(|e| Box::new(rename_columns(e, map))),
        },
        Expr::Like {
            expr,
            pattern,
            escape,
            negated,
            case_insensitive,
        } => Expr::Like {
            expr: Box::new(rename_columns(expr, map)),
            pattern: pattern.clone(),
            escape: *escape,
            negated: *negated,
            case_insensitive: *case_insensitive,
        },
        Expr::Cast { expr, target } => Expr::Cast {
            expr: Box::new(rename_columns(expr, map)),
            target: *target,
        },
        Expr::ScalarFn { kind, args } => Expr::ScalarFn {
            kind: *kind,
            args: args.iter().map(|a| rename_columns(a, map)).collect(),
        },
        Expr::Alias(inner, name) => {
            Expr::Alias(Box::new(rename_columns(inner, map)), name.clone())
        }
    }
}

/// Re-attach `conjuncts` as a single `Filter` above `plan`, or return `plan`
/// unchanged when there are no conjuncts to keep.
fn wrap_filter(plan: LogicalPlan, conjuncts: Vec<Expr>) -> LogicalPlan {
    match combine_conjuncts(conjuncts) {
        Some(predicate) => LogicalPlan::Filter {
            input: Box::new(plan),
            predicate,
        },
        None => plan,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{BinaryOp, DataType, Field};
    use crate::plan::{col, lit};

    fn scan(name: &str, fields: Vec<Field>) -> LogicalPlan {
        LogicalPlan::Scan {
            table: name.into(),
            projection: None,
            schema: Schema::new(fields),
        }
    }

    fn t() -> LogicalPlan {
        scan(
            "t",
            vec![
                Field::new("a", DataType::Int64, false),
                Field::new("b", DataType::Int64, false),
            ],
        )
    }

    #[test]
    fn pushes_filter_below_passthrough_project() {
        // Filter(Project([a, b], scan), a > 0) => Project([a, b], Filter(scan, a > 0))
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Project {
                input: Box::new(t()),
                exprs: vec![col("a"), col("b")],
            }),
            predicate: col("a").gt(lit(0_i64)),
        };
        let before = plan.schema().expect("typecheck");
        let out = PredicatePushdown.rewrite(plan).expect("push");
        let after = out.schema().expect("typecheck after");
        assert_eq!(before.fields.len(), after.fields.len());
        match out {
            LogicalPlan::Project { input, .. } => {
                assert!(matches!(*input, LogicalPlan::Filter { .. }),
                    "filter should now sit below the project");
            }
            other => panic!("expected Project on top, got {other:?}"),
        }
    }

    #[test]
    fn keeps_filter_above_computed_projection() {
        // The filtered column is computed (a + b aliased as c) so it can't be
        // pushed below the project.
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Project {
                input: Box::new(t()),
                exprs: vec![col("a").add(col("b")).alias("c")],
            }),
            predicate: col("c").gt(lit(0_i64)),
        };
        let out = PredicatePushdown.rewrite(plan).expect("push");
        match out {
            LogicalPlan::Filter { input, .. } => {
                assert!(matches!(*input, LogicalPlan::Project { .. }));
            }
            other => panic!("expected Filter to stay on top, got {other:?}"),
        }
    }

    #[test]
    fn splits_and_pushes_single_side_conjuncts_into_join() {
        let left = scan("l", vec![Field::new("a", DataType::Int64, false)]);
        let right = scan("r", vec![Field::new("b", DataType::Int64, false)]);
        let join = LogicalPlan::Join {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
            on: vec![(col("a"), col("b"))],
            filter: None,
        };
        // WHERE a > 0 AND b < 10  — each conjunct goes to its own side.
        let plan = LogicalPlan::Filter {
            input: Box::new(join),
            predicate: col("a").gt(lit(0_i64)).and(col("b").lt(lit(10_i64))),
        };
        let out = PredicatePushdown.rewrite(plan).expect("push");
        // Top node is the join (no residual filter left above it).
        match out {
            LogicalPlan::Join { left, right, .. } => {
                assert!(matches!(*left, LogicalPlan::Filter { .. }),
                    "a > 0 should land on the left input");
                assert!(matches!(*right, LogicalPlan::Filter { .. }),
                    "b < 10 should land on the right input");
            }
            other => panic!("expected Join on top, got {other:?}"),
        }
    }

    #[test]
    fn keeps_both_side_conjunct_above_join() {
        let left = scan("l", vec![Field::new("a", DataType::Int64, false)]);
        let right = scan("r", vec![Field::new("b", DataType::Int64, false)]);
        let join = LogicalPlan::Join {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
            on: vec![(col("a"), col("b"))],
            filter: None,
        };
        // a > b references both sides; must stay above the join.
        let plan = LogicalPlan::Filter {
            input: Box::new(join),
            predicate: Expr::Binary {
                op: BinaryOp::Gt,
                left: Box::new(col("a")),
                right: Box::new(col("b")),
            },
        };
        let out = PredicatePushdown.rewrite(plan).expect("push");
        assert!(matches!(out, LogicalPlan::Filter { .. }),
            "cross-side conjunct stays above the join");
    }

    #[test]
    fn does_not_push_into_non_preserved_outer_side() {
        // LEFT join: a right-side filter must stay above the join.
        let left = scan("l", vec![Field::new("a", DataType::Int64, false)]);
        let right = scan("r", vec![Field::new("b", DataType::Int64, false)]);
        let join = LogicalPlan::Join {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::LeftOuter,
            on: vec![(col("a"), col("b"))],
            filter: None,
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(join),
            predicate: col("b").gt(lit(0_i64)),
        };
        let out = PredicatePushdown.rewrite(plan).expect("push");
        assert!(matches!(out, LogicalPlan::Filter { .. }),
            "right-side filter must stay above a LEFT join");
    }

    #[test]
    fn pushes_through_sort() {
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Sort {
                input: Box::new(t()),
                sort_exprs: vec![crate::plan::logical_plan::SortExpr {
                    expr: col("a"),
                    descending: false,
                    nulls_first: false,
                }],
            }),
            predicate: col("a").gt(lit(0_i64)),
        };
        let out = PredicatePushdown.rewrite(plan).expect("push");
        match out {
            LogicalPlan::Sort { input, .. } => {
                assert!(matches!(*input, LogicalPlan::Filter { .. }));
            }
            other => panic!("expected Sort on top, got {other:?}"),
        }
    }

    /// Collect every column name referenced in `e` (test helper, since `Expr`
    /// has no `PartialEq`).
    fn cols_of(e: &Expr) -> Vec<String> {
        let mut out = Vec::new();
        collect_columns(e, &mut out);
        out
    }

    #[test]
    fn pushes_filter_into_both_union_branches() {
        // Filter(Union(scan a/b, scan a/b), a > 0)
        //   => Union(Filter(scan, a > 0), Filter(scan, a > 0))
        // Both branches gain a Filter; the outer Filter disappears.
        let union = LogicalPlan::Union {
            inputs: vec![t(), t()],
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(union),
            predicate: col("a").gt(lit(0_i64)),
        };
        let before = plan.schema().expect("typecheck");
        let out = PredicatePushdown.rewrite(plan).expect("push");
        let after = out.schema().expect("typecheck after");
        assert_eq!(before.fields.len(), after.fields.len());
        match out {
            LogicalPlan::Union { inputs } => {
                assert_eq!(inputs.len(), 2);
                for branch in &inputs {
                    assert!(matches!(branch, LogicalPlan::Filter { .. }),
                        "each union branch should now carry the pushed Filter");
                }
            }
            other => panic!("expected Union on top with no residual Filter, got {other:?}"),
        }
    }

    #[test]
    fn remaps_predicate_to_branch_names_by_position() {
        // Branch 0 outputs column `a`; branch 1 renames its column to `x` via a
        // projection. The union's output schema takes names from branch 0, so a
        // filter on `a` must be remapped to `x` when pushed into branch 1.
        let branch0 = scan("t0", vec![Field::new("a", DataType::Int64, false)]);
        let branch1 = LogicalPlan::Project {
            input: Box::new(scan("t1", vec![Field::new("x", DataType::Int64, false)])),
            exprs: vec![col("x")],
        };
        let union = LogicalPlan::Union {
            inputs: vec![branch0, branch1],
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(union),
            predicate: col("a").gt(lit(0_i64)),
        };
        // Sanity: the union output column is `a` (from branch 0).
        assert_eq!(plan.schema().expect("typecheck").fields[0].name, "a");

        let out = PredicatePushdown.rewrite(plan).expect("push");
        match out {
            LogicalPlan::Union { inputs } => {
                // Branch 0: filter references `a` (identity remap).
                match &inputs[0] {
                    LogicalPlan::Filter { predicate, .. } => {
                        assert_eq!(cols_of(predicate), vec!["a".to_string()]);
                    }
                    other => panic!("branch 0 should be Filter, got {other:?}"),
                }
                // Branch 1: the pushdown descends through the project, so the
                // surviving Filter (below the project) references the branch's
                // own column name `x`, not the union output name `a`.
                let mut found_x = false;
                let mut node = &inputs[1];
                loop {
                    match node {
                        LogicalPlan::Filter { predicate, input } => {
                            assert_eq!(cols_of(predicate), vec!["x".to_string()],
                                "branch 1 predicate must be remapped to `x` by position");
                            found_x = true;
                            node = input.as_ref();
                        }
                        LogicalPlan::Project { input, .. } => node = input.as_ref(),
                        _ => break,
                    }
                }
                assert!(found_x, "expected a remapped Filter in branch 1");
            }
            other => panic!("expected Union on top, got {other:?}"),
        }
    }

    #[test]
    fn pushes_filter_into_both_except_branches() {
        // EXCEPT: the filter is replicated into both inputs.
        let left = scan("l", vec![Field::new("a", DataType::Int64, false)]);
        let right = scan("r", vec![Field::new("a", DataType::Int64, false)]);
        let set_op = LogicalPlan::SetOp {
            left: Box::new(left),
            right: Box::new(right),
            op: SetOpKind::Except,
            all: false,
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(set_op),
            predicate: col("a").gt(lit(0_i64)),
        };
        let out = PredicatePushdown.rewrite(plan).expect("push");
        match out {
            LogicalPlan::SetOp { left, right, op, .. } => {
                assert_eq!(op, SetOpKind::Except);
                assert!(matches!(*left, LogicalPlan::Filter { .. }),
                    "EXCEPT left input should carry the pushed Filter");
                assert!(matches!(*right, LogicalPlan::Filter { .. }),
                    "EXCEPT right input should carry the pushed Filter");
            }
            other => panic!("expected SetOp on top with no residual Filter, got {other:?}"),
        }
    }

    #[test]
    fn pushes_filter_into_both_intersect_all_branches() {
        // INTERSECT ALL: same replication into both inputs.
        let left = scan("l", vec![Field::new("a", DataType::Int64, false)]);
        let right = scan("r", vec![Field::new("a", DataType::Int64, false)]);
        let set_op = LogicalPlan::SetOp {
            left: Box::new(left),
            right: Box::new(right),
            op: SetOpKind::Intersect,
            all: true,
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(set_op),
            predicate: col("a").lt(lit(10_i64)),
        };
        let out = PredicatePushdown.rewrite(plan).expect("push");
        match out {
            LogicalPlan::SetOp { left, right, op, all } => {
                assert_eq!(op, SetOpKind::Intersect);
                assert!(all, "ALL flag must be preserved");
                assert!(matches!(*left, LogicalPlan::Filter { .. }));
                assert!(matches!(*right, LogicalPlan::Filter { .. }));
            }
            other => panic!("expected SetOp on top, got {other:?}"),
        }
    }

    #[test]
    fn does_not_push_below_limit() {
        let plan = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Limit {
                input: Box::new(t()),
                limit: 5,
                offset: 0,
            }),
            predicate: col("a").gt(lit(0_i64)),
        };
        let out = PredicatePushdown.rewrite(plan).expect("push");
        assert!(matches!(out, LogicalPlan::Filter { .. }),
            "filter must remain above a Limit");
    }
}
