// SPDX-License-Identifier: Apache-2.0

//! Lazy DataFrame builder over [`LogicalPlan`].
//!
//! `DataFrame` is a thin, immutable, lazy builder: each combinator
//! (`.select(...)`, `.filter(...)`, `.group_by(...).agg(...)`) wraps the
//! current `LogicalPlan` in a new node and returns a fresh `DataFrame`. No
//! execution happens until the plan is handed to the engine.
//!
//! # Composing select / filter chains
//!
//! Composed pipelines such as `.select(...).filter(...).select(...)` are
//! supported. Wave 1's physical-plan normalization (see
//! `crate::plan::physical_plan`) is responsible for flattening arbitrary
//! `Scan / Filter / Project` chain shapes into a single fused pipeline, so
//! callers do not need to pre-coalesce projections or push filters down
//! manually — write the chain in the natural order and lowering handles it.
//!
//! # Handing the plan to the engine
//!
//! Call [`DataFrame::into_plan`] to consume the builder and obtain the
//! underlying [`LogicalPlan`]; pass that to `Engine::sql` (or the lower-level
//! planner entry points) to execute.
//!
//! ## `collect()` is a 0.1 tombstone
//!
//! [`DataFrame::collect`] is a `#[doc(hidden)]` deprecated alias for
//! `into_plan()`, kept only so older internal call sites compile. The name
//! `collect` is reserved for a future materializing API (Polars-style) in
//! 1.0; today it does not materialize anything. New code should call
//! `into_plan()` directly.
//!
//! # Builder-time validation
//!
//! The combinators perform best-effort, builder-time validation of column
//! references against the current plan's schema so that obvious user errors
//! (typos in column names) surface as close to the offending call as
//! possible rather than from deep inside lowering. See
//! [`DataFrame::validation_error`] for how those errors are exposed under
//! the current 0.1 signature constraints (terminal `into_plan()` returns a
//! bare `LogicalPlan`, so propagation is necessarily limited).

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{AggregateExpr, Expr, LogicalPlan, Schema};

/// Lazy DataFrame — wraps a `LogicalPlan` and offers a builder API.
#[derive(Debug, Clone)]
pub struct DataFrame {
    plan: LogicalPlan,
    /// First builder-time validation error encountered, if any. Stored so
    /// that combinators can keep their infallible `Self`-returning
    /// signatures (a hard requirement for 0.1) while still letting callers
    /// query for errors via [`DataFrame::validation_error`].
    //
    // TODO(post-0.1): once we are free to break the public builder API,
    // change the combinators to return `BoltResult<Self>` and drop this
    // field — propagating errors through a side channel is a 0.1 workaround.
    //
    // Stored as `Option<String>` (not `Option<BoltError>`) because
    // `BoltError` is intentionally not `Clone` (it owns a `std::io::Error`
    // for the `Io` variant); a string snapshot is sufficient for surfacing the
    // first builder-time error via `validation_error()` / `schema()`.
    first_error: Option<String>,
}

impl DataFrame {
    /// Start a query against a registered table.
    pub fn scan(table: impl Into<String>, schema: Schema) -> Self {
        Self {
            plan: LogicalPlan::Scan {
                table: table.into(),
                projection: None,
                schema,
            },
            first_error: None,
        }
    }

    /// Wrap an already-built `LogicalPlan` as a `DataFrame`.
    ///
    /// Performs a builder-time structural check that rejects any
    /// `LogicalPlan::Union { inputs: vec![] }` anywhere in the tree. A UNION
    /// with zero branches has no well-defined schema, and several downstream
    /// accessors (notably `PhysicalPlan::output_schema`) assume at least one
    /// branch is present. Rather than letting that degenerate shape escape
    /// through the public entry point and trip an internal `expect()` later,
    /// the error is recorded here and surfaced via
    /// [`DataFrame::validation_error`] / [`DataFrame::schema`], matching the
    /// 0.1 pattern used by the other combinators (we cannot change the
    /// signature to `BoltResult<Self>` without breaking the public builder
    /// API — see the `first_error` field doc).
    pub fn from_plan(plan: LogicalPlan) -> Self {
        let first_error = check_no_empty_union(&plan)
            .err()
            .map(|e| e.to_string());
        Self { plan, first_error }
    }

    /// SELECT — replace the projection list.
    ///
    /// Each `Expr::Column(name)` in `exprs` is validated against the current
    /// schema; an unknown column is recorded as a builder-time error and
    /// surfaced later via [`DataFrame::validation_error`] (the signature is
    /// pinned to `Self` for 0.1 — see the module docs).
    pub fn select<I: IntoIterator<Item = Expr>>(self, exprs: I) -> Self {
        let exprs: Vec<Expr> = exprs.into_iter().collect();
        let next_error = self.first_error.clone().or_else(|| {
            validate_exprs_against_plan(&self.plan, &exprs, "select")
                .err()
                .map(|e| e.to_string())
        });
        Self {
            plan: LogicalPlan::Project {
                input: Box::new(self.plan),
                exprs,
            },
            first_error: next_error,
        }
    }

    /// WHERE — narrow rows by a boolean predicate.
    ///
    /// The predicate's column references are validated against the current
    /// schema; an unknown column is recorded and surfaced via
    /// [`DataFrame::validation_error`].
    pub fn filter(self, predicate: Expr) -> Self {
        let next_error = self.first_error.clone().or_else(|| {
            validate_exprs_against_plan(&self.plan, std::slice::from_ref(&predicate), "filter")
                .err()
                .map(|e| e.to_string())
        });
        Self {
            plan: LogicalPlan::Filter {
                input: Box::new(self.plan),
                predicate,
            },
            first_error: next_error,
        }
    }

    /// GROUP BY — returns a `GroupedDataFrame` awaiting `.agg(...)`.
    ///
    /// Grouping keys are validated against the current schema; any error is
    /// carried forward and surfaced after the subsequent `.agg(...)` via
    /// [`DataFrame::validation_error`].
    pub fn group_by<I: IntoIterator<Item = Expr>>(self, keys: I) -> GroupedDataFrame {
        let keys: Vec<Expr> = keys.into_iter().collect();
        let next_error = self.first_error.clone().or_else(|| {
            validate_exprs_against_plan(&self.plan, &keys, "group_by")
                .err()
                .map(|e| e.to_string())
        });
        GroupedDataFrame {
            plan: self.plan,
            keys,
            first_error: next_error,
        }
    }

    /// Inspect the current plan.
    pub fn logical_plan(&self) -> &LogicalPlan {
        &self.plan
    }

    /// Type-check the plan and return its output schema.
    pub fn schema(&self) -> BoltResult<Schema> {
        if let Some(e) = &self.first_error {
            // Mirror the stored builder-time error so `schema()` does not
            // silently succeed when an upstream combinator was invalid.
            return Err(BoltError::Plan(e.clone()));
        }
        self.plan.schema()
    }

    /// Returns the first builder-time validation error encountered while
    /// constructing this `DataFrame`, if any.
    ///
    /// Combinators (`select`, `filter`, `group_by`, `agg`) are pinned to
    /// infallible `Self`-returning signatures for 0.1, so column-resolution
    /// errors detected at builder time are accumulated here instead of
    /// being returned directly. Callers who want fail-fast validation can
    /// check this before [`DataFrame::into_plan`].
    pub fn validation_error(&self) -> Option<&str> {
        self.first_error.as_deref()
    }

    /// Hand the plan off to the engine.
    ///
    /// Note: if a builder-time validation error was recorded (see
    /// [`DataFrame::validation_error`]) it is *not* returned here — the 0.1
    /// signature returns a bare `LogicalPlan`. The error will resurface
    /// when the plan is type-checked downstream (e.g. during lowering or
    /// via [`DataFrame::schema`]). Callers wanting fail-fast behavior
    /// should check `validation_error()` first.
    // TODO(1.0): introduce a real `collect()` that materializes the plan to
    // a `RecordBatch` via `Engine`. The current `collect` alias below is a
    // doc-hidden tombstone kept only so older internal call sites compile;
    // it should be removed once that materializing API lands.
    // TODO(post-0.1): change this to `BoltResult<LogicalPlan>` so the
    // stored `first_error` can be surfaced here instead of via the separate
    // `validation_error()` accessor.
    pub fn into_plan(self) -> LogicalPlan {
        self.plan
    }

    /// Deprecated alias for [`DataFrame::into_plan`]. Hidden from rustdoc
    /// because the name `collect` is reserved for a future materializing
    /// API (Polars-style) in 1.0; today this is a no-op rename kept so
    /// older internal call sites continue to compile.
    #[doc(hidden)]
    #[deprecated(since = "0.1.0", note = "use into_plan() instead")]
    pub fn collect(self) -> LogicalPlan {
        self.into_plan()
    }
}

/// Intermediate produced by `DataFrame::group_by`.
#[derive(Debug, Clone)]
pub struct GroupedDataFrame {
    plan: LogicalPlan,
    keys: Vec<Expr>,
    /// Carried forward from the originating `DataFrame::group_by` call so
    /// that the subsequent `.agg(...)` can fold its own validation into a
    /// single first-error chain.
    first_error: Option<String>,
}

impl GroupedDataFrame {
    /// Attach aggregate expressions and return a `DataFrame`.
    ///
    /// The expression inside each aggregate is validated against the
    /// pre-aggregation input schema; an unknown column is recorded as a
    /// builder-time error and surfaced via [`DataFrame::validation_error`].
    pub fn agg<I: IntoIterator<Item = AggregateExpr>>(self, aggs: I) -> DataFrame {
        let aggs: Vec<AggregateExpr> = aggs.into_iter().collect();
        let agg_exprs: Vec<Expr> = aggs.iter().map(agg_inner_expr).cloned().collect();
        let next_error = self.first_error.clone().or_else(|| {
            validate_exprs_against_plan(&self.plan, &agg_exprs, "agg")
                .err()
                .map(|e| e.to_string())
        });
        DataFrame {
            plan: LogicalPlan::Aggregate {
                input: Box::new(self.plan),
                group_by: self.keys,
                aggregates: aggs,
            },
            first_error: next_error,
        }
    }
}

/// Return a reference to the inner `Expr` of any `AggregateExpr` variant.
fn agg_inner_expr(a: &AggregateExpr) -> &Expr {
    match a {
        AggregateExpr::Count(e)
        | AggregateExpr::Sum(e)
        | AggregateExpr::Min(e)
        | AggregateExpr::Max(e)
        | AggregateExpr::Avg(e) => e,
        AggregateExpr::VarPop(e) | AggregateExpr::VarSamp(e) => e,
    }
}

/// Walk `expr` and collect every `Expr::Column` name it references.
fn collect_column_refs<'a>(expr: &'a Expr, out: &mut Vec<&'a str>) {
    match expr {
        Expr::Column(name) => out.push(name.as_str()),
        Expr::Literal(_) => {}
        Expr::Binary { left, right, .. } => {
            collect_column_refs(left, out);
            collect_column_refs(right, out);
        }
        Expr::Unary { operand, .. } => collect_column_refs(operand, out),
        Expr::Alias(inner, _) => collect_column_refs(inner, out),
    }
}

/// Walk `plan` and reject any `LogicalPlan::Union { inputs: vec![] }`.
///
/// A UNION with no branches has no schema and would later panic in
/// `PhysicalPlan::output_schema` (which calls `inputs.first().expect(..)`).
/// This is the single public entry point for hand-built logical plans, so
/// gating here closes the only way that degenerate shape can reach the
/// physical planner from outside this crate. The check is `O(nodes)` and
/// runs once per `from_plan` call.
fn check_no_empty_union(plan: &LogicalPlan) -> BoltResult<()> {
    match plan {
        LogicalPlan::Scan { .. } => Ok(()),
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Project { input, .. }
        | LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Distinct { input }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Sort { input, .. } => check_no_empty_union(input),
        LogicalPlan::Union { inputs } => {
            if inputs.is_empty() {
                return Err(BoltError::Plan(
                    "Union with zero inputs is invalid: UNION requires at \
                     least one branch"
                        .into(),
                ));
            }
            for branch in inputs {
                check_no_empty_union(branch)?;
            }
            Ok(())
        }
        LogicalPlan::Join { left, right, .. } => {
            check_no_empty_union(left)?;
            check_no_empty_union(right)
        }
    }
}

/// Resolve `plan`'s output schema and verify every column referenced in
/// `exprs` exists in it. Returns the first unresolved column as a
/// [`BoltError::Plan`]; `op` is included in the message for context.
fn validate_exprs_against_plan(
    plan: &LogicalPlan,
    exprs: &[Expr],
    op: &str,
) -> BoltResult<()> {
    // If the plan itself is already malformed enough that we can't compute a
    // schema, defer to the downstream type-checker rather than emitting a
    // misleading "column not found" — the real error is upstream.
    let schema = match plan.schema() {
        Ok(s) => s,
        Err(_) => return Ok(()),
    };
    let mut refs = Vec::new();
    for e in exprs {
        collect_column_refs(e, &mut refs);
    }
    for name in refs {
        if schema.index_of(name).is_err() {
            return Err(BoltError::Plan(format!(
                "{op}: column '{name}' not found in input schema"
            )));
        }
    }
    Ok(())
}

/// `COUNT(expr)` aggregate.
pub fn count(e: Expr) -> AggregateExpr {
    AggregateExpr::Count(e)
}

/// `SUM(expr)` aggregate.
pub fn sum(e: Expr) -> AggregateExpr {
    AggregateExpr::Sum(e)
}

/// `MIN(expr)` aggregate.
pub fn min(e: Expr) -> AggregateExpr {
    AggregateExpr::Min(e)
}

/// `MAX(expr)` aggregate.
pub fn max(e: Expr) -> AggregateExpr {
    AggregateExpr::Max(e)
}

/// `AVG(expr)` aggregate.
pub fn avg(e: Expr) -> AggregateExpr {
    AggregateExpr::Avg(e)
}

/// `VAR_POP(expr)` — population variance. Output dtype `Float64`. The
/// GROUP BY path is rejected with a clear error in v0.5; the scalar
/// (no GROUP BY) path is host-side Welford in `f64`.
pub fn var_pop(e: Expr) -> AggregateExpr {
    AggregateExpr::VarPop(Box::new(e))
}

/// `VAR_SAMP(expr)` — sample variance (`VARIANCE` per SQL standard).
/// Output dtype `Float64`. Returns NULL when fewer than 2 observations
/// were aggregated.
pub fn var_samp(e: Expr) -> AggregateExpr {
    AggregateExpr::VarSamp(Box::new(e))
}
