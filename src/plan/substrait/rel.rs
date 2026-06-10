// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "substrait")]

//! Substrait `Rel` → engine [`LogicalPlan`] conversion (feature `substrait`).
//!
//! This module is the *relational* half of the Substrait ingestion path. It
//! walks a `substrait::proto::Rel` tree and produces the engine's
//! [`LogicalPlan`] AST defined in [`crate::plan::logical_plan`]. The *scalar*
//! half — converting `substrait::proto::Expression` into [`Expr`] — lives in
//! the sibling [`super::expr`] module and is called here for every predicate,
//! projection, grouping key, aggregate argument, sort key and join condition.
//!
//! # Contract with the converter context ([`super::ConvertCtx`], owned by mod.rs)
//!
//! Relation conversion needs three things from the shared converter context:
//!
//! 1. **Table-schema resolution.** [`ReadRel`] carries a `NamedTable` whose
//!    `names` identify a registered table; we resolve its [`Schema`] via
//!    [`ConvertCtx::resolve_table`] (which wraps the engine's `TableProvider`).
//! 2. **The "current input schema".** Substrait expressions reference columns
//!    *positionally* (`FieldReference` → struct field index), not by name. The
//!    sibling `convert_expr` therefore needs to know the schema of the relation
//!    it is being evaluated against so it can turn an ordinal into a
//!    [`Expr::Column`] with the right name. We narrow the context to that
//!    schema with [`ConvertCtx::with_input_schema`] before converting the
//!    expressions that reference it.
//! 3. **Extension-function resolution.** Substrait aggregate / scalar functions
//!    are referenced by a numeric `function_reference` anchor declared in the
//!    plan's extensions. The context resolves an anchor to its compound
//!    function name via [`ConvertCtx::function_name`].
//!
//! `ConvertCtx` is `Copy` (it holds only references), so it is threaded *by
//! value* and narrowed allocation-free via `with_input_schema`.

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{AggregateExpr, BinaryOp, Expr, JoinType, LogicalPlan, SortExpr};

use super::expr::convert_expr;
use super::ConvertCtx;

use substrait::proto::{
    aggregate_function::AggregationInvocation,
    aggregate_rel::Measure,
    rel::RelType,
    rel_common::EmitKind,
    sort_field::{SortDirection, SortKind},
    AggregateFunction, AggregateRel, FetchRel, FilterRel, JoinRel, ProjectRel, ReadRel, RelCommon,
    Rel, SortField, SortRel,
};

/// Convert a Substrait [`Rel`] node into the engine's [`LogicalPlan`].
///
/// Recurses on relational inputs and delegates every scalar sub-expression to
/// the sibling [`convert_expr`]. Unsupported relation kinds surface a
/// [`BoltError::Plan`] naming the operator so the caller gets a precise
/// diagnostic rather than a generic failure.
pub(crate) fn convert_rel(rel: &Rel, ctx: &ConvertCtx) -> BoltResult<LogicalPlan> {
    let rel_type = rel
        .rel_type
        .as_ref()
        .ok_or_else(|| BoltError::Plan("Substrait Rel has no rel_type set".into()))?;

    // Convert the operator, then honour its `RelCommon.emit` column remap (if
    // any). Emit is applied uniformly here so every relation kind — not just
    // ProjectRel — reorders / selects its output columns as the producer asked;
    // downstream positional field references resolve against the remapped shape.
    let (plan, common) = match rel_type {
        RelType::Read(read) => (convert_read(read, ctx)?, read.common.as_ref()),
        RelType::Filter(filter) => (convert_filter(filter, ctx)?, filter.common.as_ref()),
        RelType::Project(project) => (convert_project(project, ctx)?, project.common.as_ref()),
        RelType::Aggregate(agg) => (convert_aggregate(agg, ctx)?, agg.common.as_ref()),
        RelType::Sort(sort) => (convert_sort(sort, ctx)?, sort.common.as_ref()),
        RelType::Join(join) => (convert_join(join, ctx)?, join.common.as_ref()),
        RelType::Fetch(fetch) => (convert_fetch(fetch, ctx)?, fetch.common.as_ref()),
        // Relation kinds outside the current ingestion envelope. Each carries
        // a distinct message so the user knows exactly which operator tripped.
        other => {
            return Err(BoltError::Plan(format!(
                "Substrait relation kind {} is not supported by the engine yet",
                rel_type_name(other)
            )))
        }
    };

    apply_emit(plan, common)
}

/// A short human-readable label for an unsupported [`RelType`], for errors.
fn rel_type_name(rt: &RelType) -> &'static str {
    match rt {
        RelType::Read(_) => "ReadRel",
        RelType::Filter(_) => "FilterRel",
        RelType::Fetch(_) => "FetchRel",
        RelType::Aggregate(_) => "AggregateRel",
        RelType::Sort(_) => "SortRel",
        RelType::Join(_) => "JoinRel",
        RelType::Project(_) => "ProjectRel",
        RelType::Set(_) => "SetRel",
        RelType::ExtensionSingle(_) => "ExtensionSingleRel",
        RelType::ExtensionMulti(_) => "ExtensionMultiRel",
        RelType::ExtensionLeaf(_) => "ExtensionLeafRel",
        RelType::Cross(_) => "CrossRel",
        // `substrait` >= recent versions add more arms; the catch-all keeps
        // this exhaustive-by-intent without breaking on crate upgrades.
        _ => "unknown Rel",
    }
}

/// Honour a relation's [`RelCommon::emit_kind`] by reordering / selecting the
/// converted plan's output columns to match the producer's requested output.
///
/// Every Substrait relation carries an optional `RelCommon` whose `emit_kind`
/// oneof is either `Direct` (output the operator's columns unchanged) or
/// `Emit { output_mapping }` — a list of 0-based ordinals into the operator's
/// output that selects *and reorders* the visible columns. This is **not**
/// advisory: when a producer sets an explicit `Emit`, downstream relations
/// reference columns *positionally* against the emitted (remapped) shape, so
/// ignoring it silently mis-resolves every later field reference. We therefore
/// honour it here by wrapping the converted plan in a
/// [`LogicalPlan::Project`] of the selected columns in mapping order.
///
/// `Direct` (and an unset `emit_kind` / unset `common`) are pass-through and
/// return `plan` unchanged. An out-of-range mapping ordinal is rejected.
fn apply_emit(plan: LogicalPlan, common: Option<&RelCommon>) -> BoltResult<LogicalPlan> {
    let emit = match common.and_then(|c| c.emit_kind.as_ref()) {
        // Explicit column remap: honour it.
        Some(EmitKind::Emit(e)) => e,
        // Direct / unset: the operator's natural output is the visible output.
        Some(EmitKind::Direct(_)) | None => return Ok(plan),
    };

    if emit.output_mapping.is_empty() {
        // An empty emit selects no columns, which the engine's relational
        // model cannot represent (a zero-column relation). Reject rather than
        // silently producing the full passthrough.
        return Err(BoltError::Unsupported(
            "substrait: RelCommon.emit with an empty output_mapping (zero output \
             columns) is not supported"
                .into(),
        ));
    }

    let schema = plan.schema()?;
    let mut exprs = Vec::with_capacity(emit.output_mapping.len());
    for &ordinal in &emit.output_mapping {
        if ordinal < 0 {
            return Err(BoltError::Plan(format!(
                "substrait: RelCommon.emit output_mapping has a negative ordinal {ordinal}"
            )));
        }
        let idx = ordinal as usize;
        let field = schema.fields.get(idx).ok_or_else(|| {
            BoltError::Plan(format!(
                "substrait: RelCommon.emit output_mapping ordinal {idx} out of range \
                 (relation has {} output columns)",
                schema.fields.len()
            ))
        })?;
        exprs.push(Expr::Column(field.name.clone()));
    }

    Ok(LogicalPlan::Project {
        input: Box::new(plan),
        exprs,
    })
}

/// `ReadRel` → [`LogicalPlan::Scan`].
///
/// Only the `NamedTable` read-type is supported (the common case for
/// engine-to-engine interchange). `VirtualTable`, `LocalFiles` and
/// `ExtensionTable` sources are rejected. The table name is resolved against
/// the converter context's provider to recover the base [`Schema`]; we trust
/// the registered schema over the (optional) `base_schema` carried in the
/// message so downstream type-checking sees exactly the columns the engine
/// knows about.
fn convert_read(read: &ReadRel, ctx: &ConvertCtx) -> BoltResult<LogicalPlan> {
    use substrait::proto::read_rel::ReadType;

    let read_type = read
        .read_type
        .as_ref()
        .ok_or_else(|| BoltError::Plan("Substrait ReadRel has no read_type".into()))?;

    let names = match read_type {
        ReadType::NamedTable(nt) => &nt.names,
        ReadType::VirtualTable(_) => {
            return Err(BoltError::Plan(
                "Substrait ReadRel: VirtualTable source is not supported".into(),
            ))
        }
        ReadType::LocalFiles(_) => {
            return Err(BoltError::Plan(
                "Substrait ReadRel: LocalFiles source is not supported".into(),
            ))
        }
        ReadType::ExtensionTable(_) => {
            return Err(BoltError::Plan(
                "Substrait ReadRel: ExtensionTable source is not supported".into(),
            ))
        }
        // substrait 0.55 added further read-type variants (e.g. IcebergTable);
        // only NamedTable is supported by this engine.
        other => {
            return Err(BoltError::Plan(format!(
                "Substrait ReadRel: unsupported read_type {other:?}; only NamedTable is supported"
            )))
        }
    };

    if names.is_empty() {
        return Err(BoltError::Plan(
            "Substrait ReadRel NamedTable has no name parts".into(),
        ));
    }

    // A `ReadRel.projection` is an `Expression.MaskExpression` (a nested
    // struct-select mask), not a flat ordinal list. The engine's `Scan`
    // projection is a flat column-name subset and cannot faithfully represent
    // an arbitrary mask, so reject a present projection loudly rather than
    // silently reading every column.
    if read.projection.is_some() {
        return Err(BoltError::Unsupported(
            "substrait: ReadRel.projection (MaskExpression) is not supported; \
             a projection mask cannot be faithfully represented as a flat Scan \
             projection"
                .into(),
        ));
    }

    let (table, schema) = ctx.resolve_table(names)?;

    let scan = LogicalPlan::Scan {
        table,
        projection: None,
        schema,
    };

    // `ReadRel.filter` is a mandatory post-read predicate and
    // `best_effort_filter` is a predicate the source may have applied
    // opportunistically; both are real boolean predicates over the scanned
    // rows, so honouring either as a `Filter` above the `Scan` is correct
    // (applying `best_effort_filter` as a hard filter never changes the result
    // set, since it is a genuine predicate on the data). When both are present
    // they are ANDed. The predicates reference the scan's columns positionally.
    let scan_schema = scan.schema()?;
    let c2 = ctx.with_input_schema(&scan_schema);

    let mut predicate: Option<Expr> = None;
    if let Some(filter) = read.filter.as_deref() {
        predicate = Some(convert_expr(filter, &c2)?);
    }
    if let Some(best_effort) = read.best_effort_filter.as_deref() {
        let extra = convert_expr(best_effort, &c2)?;
        predicate = Some(match predicate {
            Some(existing) => Expr::Binary {
                op: BinaryOp::And,
                left: Box::new(existing),
                right: Box::new(extra),
            },
            None => extra,
        });
    }

    match predicate {
        None => Ok(scan),
        Some(predicate) => Ok(LogicalPlan::Filter {
            input: Box::new(scan),
            predicate,
        }),
    }
}

/// `FilterRel` → [`LogicalPlan::Filter`].
///
/// Recurses on the input, then converts the boolean `condition` against the
/// input's schema (pushed onto the context so positional field references
/// resolve correctly).
fn convert_filter(filter: &FilterRel, ctx: &ConvertCtx) -> BoltResult<LogicalPlan> {
    let input = convert_boxed_input(filter.input.as_deref(), ctx, "FilterRel")?;
    let input_schema = input.schema()?;

    let condition = filter
        .condition
        .as_deref()
        .ok_or_else(|| BoltError::Plan("Substrait FilterRel has no condition".into()))?;

    let c2 = ctx.with_input_schema(&input_schema);
    let predicate = convert_expr(condition, &c2)?;

    Ok(LogicalPlan::Filter {
        input: Box::new(input),
        predicate,
    })
}

/// `ProjectRel` → [`LogicalPlan::Project`].
///
/// Substrait's `ProjectRel` *appends* its computed `expressions` to the input's
/// columns; the operator's natural output is therefore `input_columns ++
/// expressions`. The producer then uses `RelCommon.emit` to choose the final
/// visible set / order, which [`convert_rel`] applies uniformly via
/// [`apply_emit`] after this converter returns. We model the append here by
/// emitting the input's pass-through columns followed by the computed
/// expressions so that emit ordinals (and downstream positional field
/// references) resolve against the correct combined shape.
fn convert_project(project: &ProjectRel, ctx: &ConvertCtx) -> BoltResult<LogicalPlan> {
    let input = convert_boxed_input(project.input.as_deref(), ctx, "ProjectRel")?;
    let input_schema = input.schema()?;

    let c2 = ctx.with_input_schema(&input_schema);
    let computed = project
        .expressions
        .iter()
        .map(|e| convert_expr(e, &c2))
        .collect::<BoltResult<Vec<_>>>()?;

    if computed.is_empty() {
        return Err(BoltError::Plan(
            "Substrait ProjectRel has no expressions".into(),
        ));
    }

    // Output = pass-through input columns ++ computed expressions. Without an
    // explicit `emit`, this is the full visible output (matching Substrait's
    // "Direct" emit). With an `emit`, `apply_emit` selects/reorders from this
    // combined column list, so the pass-through columns must be present here
    // for those ordinals to resolve.
    let mut exprs = Vec::with_capacity(input_schema.fields.len() + computed.len());
    for field in &input_schema.fields {
        exprs.push(Expr::Column(field.name.clone()));
    }
    exprs.extend(computed);

    Ok(LogicalPlan::Project {
        input: Box::new(input),
        exprs,
    })
}

/// `AggregateRel` → [`LogicalPlan::Aggregate`].
///
/// Substrait supports multiple `groupings` (for GROUPING SETS / ROLLUP); the
/// engine's [`LogicalPlan::Aggregate`] models a single flat grouping key list,
/// so we accept either zero groupings (scalar aggregate) or exactly one and
/// reject the multi-grouping case with a clear message. Each `Measure` carries
/// an [`AggregateFunction`] whose `function_reference` anchor is resolved to a
/// canonical name (`sum` / `min` / `max` / `count` / `avg`) via the context.
fn convert_aggregate(agg: &AggregateRel, ctx: &ConvertCtx) -> BoltResult<LogicalPlan> {
    let input = convert_boxed_input(agg.input.as_deref(), ctx, "AggregateRel")?;
    let input_schema = input.schema()?;

    if agg.groupings.len() > 1 {
        return Err(BoltError::Plan(format!(
            "Substrait AggregateRel with {} grouping sets is not supported \
             (only a single flat GROUP BY is modelled)",
            agg.groupings.len()
        )));
    }

    let c2 = ctx.with_input_schema(&input_schema);
    let mut group_by = Vec::new();
    if let Some(grouping) = agg.groupings.first() {
        for e in &grouping.grouping_expressions {
            group_by.push(convert_expr(e, &c2)?);
        }
    }

    let mut aggregates = Vec::with_capacity(agg.measures.len());
    for measure in &agg.measures {
        aggregates.push(convert_measure(measure, &c2)?);
    }

    if aggregates.is_empty() {
        return Err(BoltError::Plan(
            "Substrait AggregateRel has no measures".into(),
        ));
    }

    Ok(LogicalPlan::Aggregate {
        input: Box::new(input),
        group_by,
        aggregates,
    })
}

/// Convert one Substrait aggregate [`Measure`] into an [`AggregateExpr`].
fn convert_measure(measure: &Measure, ctx: &ConvertCtx) -> BoltResult<AggregateExpr> {
    // A per-measure `filter` is SQL `SUM(x) FILTER (WHERE p)`. The engine's
    // [`AggregateExpr`] has no filtered-aggregate variant, so we cannot
    // faithfully represent it; reject loudly rather than silently computing the
    // unfiltered aggregate (which would be a wrong result).
    if measure.filter.is_some() {
        return Err(BoltError::Unsupported(
            "substrait: per-measure Measure.filter (aggregate FILTER (WHERE ..)) \
             is not supported"
                .into(),
        ));
    }

    let func: &AggregateFunction = measure
        .measure
        .as_ref()
        .ok_or_else(|| BoltError::Plan("Substrait Measure has no aggregate function".into()))?;

    // The arguments are `FunctionArgument`s; for the supported aggregates each
    // takes a single scalar value argument. COUNT(*) is represented as a zero-
    // argument aggregate.
    let mut args = Vec::with_capacity(func.arguments.len());
    for arg in &func.arguments {
        args.push(convert_function_arg(arg, ctx)?);
    }

    // DISTINCT aggregates (`AggregationInvocation::Distinct`) change the
    // semantics and are not modelled by `AggregateExpr`; reject them rather
    // than silently computing the non-distinct result.
    if func.invocation == AggregationInvocation::Distinct as i32 {
        return Err(BoltError::Plan(
            "Substrait DISTINCT aggregate is not supported".into(),
        ));
    }

    let name = ctx.function_name(func.function_reference)?;
    let canonical = canonical_agg_name(&name);

    match canonical.as_str() {
        "count" => {
            // COUNT(*) has no argument; map it to COUNT over a synthetic
            // constant-true so the engine counts rows. COUNT(expr) takes one.
            let inner = match args.into_iter().next() {
                Some(e) => e,
                None => Expr::Literal(crate::plan::logical_plan::Literal::Int64(1)),
            };
            Ok(AggregateExpr::Count(inner))
        }
        "sum" => Ok(AggregateExpr::Sum(take_one_arg(args, "SUM")?)),
        "min" => Ok(AggregateExpr::Min(take_one_arg(args, "MIN")?)),
        "max" => Ok(AggregateExpr::Max(take_one_arg(args, "MAX")?)),
        "avg" | "mean" => Ok(AggregateExpr::Avg(take_one_arg(args, "AVG")?)),
        other => Err(BoltError::Plan(format!(
            "Substrait aggregate function '{other}' is not supported \
             (expected one of sum/min/max/count/avg)"
        ))),
    }
}

/// Extract exactly one argument for a unary aggregate, erroring otherwise.
fn take_one_arg(mut args: Vec<Expr>, agg: &str) -> BoltResult<Expr> {
    if args.len() != 1 {
        return Err(BoltError::Plan(format!(
            "Substrait {agg} expects exactly one argument, got {}",
            args.len()
        )));
    }
    Ok(args.pop().expect("len checked == 1"))
}

/// Convert a Substrait `FunctionArgument` that carries a scalar value into an
/// [`Expr`]. Type / enum arguments are rejected (the supported aggregates take
/// only value arguments).
fn convert_function_arg(
    arg: &substrait::proto::FunctionArgument,
    ctx: &ConvertCtx,
) -> BoltResult<Expr> {
    use substrait::proto::function_argument::ArgType;
    match arg.arg_type.as_ref() {
        Some(ArgType::Value(expr)) => convert_expr(expr, ctx),
        Some(ArgType::Type(_)) => Err(BoltError::Plan(
            "Substrait function type-argument is not supported".into(),
        )),
        Some(ArgType::Enum(_)) => Err(BoltError::Plan(
            "Substrait function enum-argument is not supported".into(),
        )),
        None => Err(BoltError::Plan(
            "Substrait function argument has no arg_type".into(),
        )),
    }
}

/// Normalise a resolved extension-function name to its bare canonical form.
///
/// Substrait function names are often decorated with their argument-type
/// signature, e.g. `sum:i64` or `count:opt`. We strip everything from the
/// first `:` and lowercase the head so the match in [`convert_measure`] is
/// signature-agnostic.
fn canonical_agg_name(name: &str) -> String {
    let head = name.split(':').next().unwrap_or(name);
    head.trim().to_ascii_lowercase()
}

/// `SortRel` → [`LogicalPlan::Sort`].
fn convert_sort(sort: &SortRel, ctx: &ConvertCtx) -> BoltResult<LogicalPlan> {
    let input = convert_boxed_input(sort.input.as_deref(), ctx, "SortRel")?;
    let input_schema = input.schema()?;

    if sort.sorts.is_empty() {
        return Err(BoltError::Plan(
            "Substrait SortRel has no sort fields".into(),
        ));
    }

    let c2 = ctx.with_input_schema(&input_schema);
    let sort_exprs = sort
        .sorts
        .iter()
        .map(|sf| convert_sort_field(sf, &c2))
        .collect::<BoltResult<Vec<_>>>()?;

    Ok(LogicalPlan::Sort {
        input: Box::new(input),
        sort_exprs,
    })
}

/// Convert one Substrait [`SortField`] into a [`SortExpr`].
///
/// Substrait's `SortDirection` packs direction *and* NULL placement into a
/// single enum. We map the four directional variants; `Clustered` (group-but-
/// don't-order) and any custom `ComparisonFunctionReference` are rejected.
fn convert_sort_field(sf: &SortField, ctx: &ConvertCtx) -> BoltResult<SortExpr> {
    let expr_proto = sf
        .expr
        .as_ref()
        .ok_or_else(|| BoltError::Plan("Substrait SortField has no expression".into()))?;
    let expr = convert_expr(expr_proto, ctx)?;

    let kind = sf
        .sort_kind
        .as_ref()
        .ok_or_else(|| BoltError::Plan("Substrait SortField has no sort_kind".into()))?;

    let (descending, nulls_first) = match kind {
        SortKind::Direction(d) => {
            // `SortDirection::from_i32` is not stable across crate versions;
            // match the raw discriminant via the generated enum constants.
            match SortDirection::try_from(*d).unwrap_or(SortDirection::Unspecified) {
                SortDirection::AscNullsFirst => (false, true),
                SortDirection::AscNullsLast => (false, false),
                SortDirection::DescNullsFirst => (true, true),
                SortDirection::DescNullsLast => (true, false),
                SortDirection::Clustered => {
                    return Err(BoltError::Plan(
                        "Substrait SORT_DIRECTION_CLUSTERED is not supported".into(),
                    ))
                }
                SortDirection::Unspecified => {
                    return Err(BoltError::Plan(
                        "Substrait SortField has an unspecified sort direction".into(),
                    ))
                }
            }
        }
        SortKind::ComparisonFunctionReference(_) => {
            return Err(BoltError::Plan(
                "Substrait SortField with a custom comparison function is not supported".into(),
            ))
        }
    };

    Ok(SortExpr {
        expr,
        descending,
        nulls_first,
    })
}

/// `JoinRel` → [`LogicalPlan::Join`].
///
/// Maps INNER / LEFT / RIGHT (and FULL / CROSS where representable) join types
/// and converts the single join `expression` against the *combined* schema of
/// the two inputs (so positional field references on the right side resolve
/// past the left side's width). The combined expression is placed in `filter`
/// (the engine's residual-predicate slot); the equi-pair fast path (`on`) is a
/// later optimisation. TODO: pattern-match a top-level conjunction of
/// `left.col = right.col` equalities and lift them into `on` for the hash-join
/// fast path.
///
/// `JoinRel.post_join_filter` — a predicate applied to the join's *output*
/// rows — is honoured by wrapping the converted [`LogicalPlan::Join`] in a
/// [`LogicalPlan::Filter`]. It also references the combined left ++ right
/// schema. (It is kept distinct from the join's own `expression`: the join
/// condition governs which rows match — and, for outer joins, which rows are
/// NULL-extended — whereas `post_join_filter` filters the produced rows after
/// that, so the two are NOT equivalent for outer joins and must not be merged.)
fn convert_join(join: &JoinRel, ctx: &ConvertCtx) -> BoltResult<LogicalPlan> {
    use substrait::proto::join_rel::JoinType as SJoinType;

    let left = convert_boxed_input(join.left.as_deref(), ctx, "JoinRel (left)")?;
    let right = convert_boxed_input(join.right.as_deref(), ctx, "JoinRel (right)")?;

    let join_type = match SJoinType::try_from(join.r#type).unwrap_or(SJoinType::Unspecified) {
        SJoinType::Inner => JoinType::Inner,
        SJoinType::Left => JoinType::LeftOuter,
        SJoinType::Right => JoinType::RightOuter,
        SJoinType::Outer => JoinType::FullOuter,
        other => {
            return Err(BoltError::Plan(format!(
                "Substrait join type {other:?} is not supported \
                 (expected INNER/LEFT/RIGHT/OUTER)"
            )))
        }
    };

    // The join predicate references columns from both inputs by a single
    // positional space (left fields then right fields). Build the combined
    // schema the engine itself uses for the residual filter so `convert_expr`
    // resolves right-side ordinals to their post-rename names.
    let combined = crate::plan::logical_plan::join_combined_schema(
        &left.schema()?,
        &right.schema()?,
        join_type,
    );

    let c2 = ctx.with_input_schema(&combined);

    let filter = match join.expression.as_deref() {
        Some(expr) => Some(convert_expr(expr, &c2)?),
        None => None,
    };

    // Convert the post-join filter (if any) against the same combined schema
    // before consuming `left`/`right` into the Join node.
    let post_join_filter = match join.post_join_filter.as_deref() {
        Some(expr) => Some(convert_expr(expr, &c2)?),
        None => None,
    };

    let join_plan = LogicalPlan::Join {
        left: Box::new(left),
        right: Box::new(right),
        join_type,
        on: Vec::new(),
        filter,
    };

    // `post_join_filter` is a predicate over the join's output rows; honour it
    // as a `Filter` above the `Join` rather than folding it into the join
    // condition (the two differ for outer joins — see the doc comment).
    match post_join_filter {
        None => Ok(join_plan),
        Some(predicate) => Ok(LogicalPlan::Filter {
            input: Box::new(join_plan),
            predicate,
        }),
    }
}

/// `FetchRel` → [`LogicalPlan::Limit`].
///
/// Substrait 0.55 models the row window with two `oneof`s rather than plain
/// scalar fields:
///
/// * `offset_mode` — either the deprecated literal `offset` (`i64`) or an
///   `offset_expr` (an arbitrary [`Expression`]). Unset means offset 0.
/// * `count_mode` — either the deprecated literal `count` (`i64`, `-1` =
///   "all rows") or a `count_expr` ([`Expression`]). Unset means "all rows".
///
/// The engine's [`LogicalPlan::Limit`] takes a concrete `usize` limit/offset,
/// so we only support the literal forms here. An expression-typed offset/count
/// would need constant-folding the [`Expression`] to an integer, which is out
/// of scope for the ingestion core. An unset/`-1` ("all rows") count maps to
/// `usize::MAX` (an effectively unbounded limit) so a bare `OFFSET` still works.
///
/// TODO: constant-fold `offset_expr` / `count_expr` (a literal-or-cast-of-
/// literal `i64`) so producers emitting the non-deprecated expression form are
/// also accepted.
fn convert_fetch(fetch: &FetchRel, ctx: &ConvertCtx) -> BoltResult<LogicalPlan> {
    use substrait::proto::fetch_rel::{CountMode, OffsetMode};

    let input = convert_boxed_input(fetch.input.as_deref(), ctx, "FetchRel")?;

    // OFFSET: unset → 0; literal i64 → that value (negative is invalid);
    // expression form is not yet constant-folded.
    let offset = match &fetch.offset_mode {
        None => 0usize,
        Some(OffsetMode::Offset(n)) => {
            if *n < 0 {
                return Err(BoltError::Plan(
                    "Substrait FetchRel has a negative offset".into(),
                ));
            }
            *n as usize
        }
        Some(OffsetMode::OffsetExpr(_)) => {
            return Err(BoltError::Plan(
                "Substrait FetchRel offset_expr (expression-typed offset) is not \
                 yet supported (only a literal offset)"
                    .into(),
            ))
        }
    };

    // COUNT: unset → "all rows" (usize::MAX); literal -1 → "all rows";
    // literal >= 0 → that value; expression form is not yet constant-folded.
    let limit = match &fetch.count_mode {
        None => usize::MAX,
        Some(CountMode::Count(n)) => {
            if *n < 0 {
                // -1 is the Substrait sentinel for "return ALL records".
                usize::MAX
            } else {
                *n as usize
            }
        }
        Some(CountMode::CountExpr(_)) => {
            return Err(BoltError::Plan(
                "Substrait FetchRel count_expr (expression-typed count) is not \
                 yet supported (only a literal count)"
                    .into(),
            ))
        }
    };

    Ok(LogicalPlan::Limit {
        input: Box::new(input),
        limit,
        offset,
    })
}

/// Shared helper: recurse into a boxed `Option<Rel>` input, mapping a missing
/// input to a precise [`BoltError::Plan`] naming the parent relation.
fn convert_boxed_input(
    input: Option<&Rel>,
    ctx: &ConvertCtx,
    parent: &str,
) -> BoltResult<LogicalPlan> {
    let rel = input.ok_or_else(|| BoltError::Plan(format!("Substrait {parent} has no input")))?;
    convert_rel(rel, ctx)
}

#[cfg(test)]
mod tests {
    //! Host-side unit coverage for relation conversion. These tests build
    //! Substrait proto messages by hand (no network / no real plan producer)
    //! and assert the resulting [`LogicalPlan`] shape.
    //!
    //! The tests build a [`ConvertCtx`] over a [`MemTableProvider`] seeded
    //! with one table `t` plus a small function registry (see [`fixtures`]).

    use super::*;
    use crate::plan::logical_plan::{DataType, Field, Schema};
    use crate::plan::sql_frontend::MemTableProvider;
    use std::collections::HashMap;
    use substrait::proto::{
        expression::{
            field_reference::{ReferenceType, RootReference, RootType},
            reference_segment, FieldReference, MaskExpression, ReferenceSegment, RexType,
            ScalarFunction,
        },
        read_rel::{NamedTable, ReadType},
        rel_common::{Emit, EmitKind},
        Expression, FilterRel, FunctionArgument, RelCommon, ReadRel, Rel,
    };

    /// Build a `t(a Int64, b Int64)` schema for the fixture table.
    fn fixture_schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ])
    }

    /// The converter context's backing state: a provider seeded with one table
    /// `t` and a function registry with anchor `0 -> "equal"`. The owned values
    /// live in the caller's stack frame; [`ctx`] borrows them.
    fn fixtures() -> (MemTableProvider, HashMap<u32, String>) {
        let provider = MemTableProvider::new().with_table("t", fixture_schema());
        let mut functions = HashMap::new();
        functions.insert(0u32, "equal".to_string());
        (provider, functions)
    }

    /// Build a [`ConvertCtx`] borrowing `provider` / `functions` produced by
    /// [`fixtures`].
    fn ctx<'a>(
        provider: &'a MemTableProvider,
        functions: &'a HashMap<u32, String>,
    ) -> ConvertCtx<'a> {
        ConvertCtx::new(provider, functions)
    }

    /// A bare positional column reference `#idx` (Substrait field reference
    /// with a single struct-field selection rooted at the input row).
    fn field_ref(idx: i32) -> Expression {
        Expression {
            rex_type: Some(RexType::Selection(Box::new(FieldReference {
                reference_type: Some(ReferenceType::DirectReference(ReferenceSegment {
                    reference_type: Some(reference_segment::ReferenceType::StructField(Box::new(
                        reference_segment::StructField {
                            field: idx,
                            child: None,
                        },
                    ))),
                })),
                root_type: Some(RootType::RootReference(RootReference {})),
            }))),
        }
    }

    fn read_rel() -> Rel {
        Rel {
            rel_type: Some(RelType::Read(Box::new(ReadRel {
                read_type: Some(ReadType::NamedTable(NamedTable {
                    names: vec!["t".to_string()],
                    advanced_extension: None,
                })),
                ..Default::default()
            }))),
        }
    }

    /// A bare `ReadRel` (not wrapped in a `Rel`) over the fixture table `t`,
    /// so tests can set its `filter` / `projection` / `common` fields before
    /// dispatching through `convert_rel`.
    fn read_rel_inner() -> ReadRel {
        ReadRel {
            read_type: Some(ReadType::NamedTable(NamedTable {
                names: vec!["t".to_string()],
                advanced_extension: None,
            })),
            ..Default::default()
        }
    }

    /// The predicate `#0 = #1` (a = b), built via the test ctx's `equal`
    /// (anchor 0) scalar function so it exercises the real expr.rs path.
    fn eq_a_b() -> Expression {
        Expression {
            rex_type: Some(RexType::ScalarFunction(ScalarFunction {
                function_reference: 0, // resolves to "equal" in test ctx
                arguments: vec![
                    FunctionArgument {
                        arg_type: Some(substrait::proto::function_argument::ArgType::Value(
                            field_ref(0),
                        )),
                    },
                    FunctionArgument {
                        arg_type: Some(substrait::proto::function_argument::ArgType::Value(
                            field_ref(1),
                        )),
                    },
                ],
                ..Default::default()
            })),
        }
    }

    #[test]
    fn read_rel_becomes_scan() {
        let (provider, functions) = fixtures();
        let ctx = ctx(&provider, &functions);
        let plan = convert_rel(&read_rel(), &ctx).expect("read converts");
        match plan {
            LogicalPlan::Scan { table, schema, .. } => {
                assert_eq!(table, "t");
                assert_eq!(schema.fields.len(), 2);
                assert_eq!(schema.fields[0].name, "a");
            }
            other => panic!("expected Scan, got {other:?}"),
        }
    }

    #[test]
    fn filter_over_read_becomes_filter_scan() {
        let (provider, functions) = fixtures();
        let ctx = ctx(&provider, &functions);
        // condition: `#0 = #1` (a = b) — asserts the *relational* shape
        // (Filter wrapping Scan).
        let filter = Rel {
            rel_type: Some(RelType::Filter(Box::new(FilterRel {
                input: Some(Box::new(read_rel())),
                condition: Some(Box::new(eq_a_b())),
                ..Default::default()
            }))),
        };

        let plan = convert_rel(&filter, &ctx).expect("filter converts");
        match plan {
            LogicalPlan::Filter { input, .. } => {
                assert!(
                    matches!(*input, LogicalPlan::Scan { .. }),
                    "filter input should be the Scan"
                );
            }
            other => panic!("expected Filter, got {other:?}"),
        }
    }

    #[test]
    fn missing_rel_type_errors() {
        let (provider, functions) = fixtures();
        let ctx = ctx(&provider, &functions);
        let rel = Rel { rel_type: None };
        assert!(convert_rel(&rel, &ctx).is_err());
    }

    /// `ReadRel.filter` is honoured as a `Filter` wrapping the `Scan` rather
    /// than silently dropped.
    #[test]
    fn read_rel_filter_becomes_filter_over_scan() {
        let (provider, functions) = fixtures();
        let ctx = ctx(&provider, &functions);
        let mut read = read_rel_inner();
        read.filter = Some(Box::new(eq_a_b()));
        let rel = Rel {
            rel_type: Some(RelType::Read(Box::new(read))),
        };
        let plan = convert_rel(&rel, &ctx).expect("read with filter converts");
        match plan {
            LogicalPlan::Filter { input, .. } => assert!(
                matches!(*input, LogicalPlan::Scan { .. }),
                "ReadRel.filter should wrap the Scan in a Filter"
            ),
            other => panic!("expected Filter over Scan, got {other:?}"),
        }
    }

    /// `ReadRel.best_effort_filter` is likewise honoured as a `Filter`.
    #[test]
    fn read_rel_best_effort_filter_honoured() {
        let (provider, functions) = fixtures();
        let ctx = ctx(&provider, &functions);
        let mut read = read_rel_inner();
        read.best_effort_filter = Some(Box::new(eq_a_b()));
        let rel = Rel {
            rel_type: Some(RelType::Read(Box::new(read))),
        };
        let plan = convert_rel(&rel, &ctx).expect("read with best_effort_filter converts");
        assert!(
            matches!(plan, LogicalPlan::Filter { .. }),
            "ReadRel.best_effort_filter should be honoured as a Filter, got {plan:?}"
        );
    }

    /// `ReadRel.projection` (a MaskExpression) cannot be faithfully represented
    /// and is rejected loudly rather than silently ignored.
    #[test]
    fn read_rel_projection_mask_rejected() {
        let (provider, functions) = fixtures();
        let ctx = ctx(&provider, &functions);
        let mut read = read_rel_inner();
        read.projection = Some(MaskExpression::default());
        let rel = Rel {
            rel_type: Some(RelType::Read(Box::new(read))),
        };
        let err = convert_rel(&rel, &ctx).expect_err("projection mask must be rejected");
        assert!(matches!(err, BoltError::Unsupported(_)), "got {err:?}");
    }

    /// `RelCommon.emit` reorders/selects the output columns: an emit of `[1]`
    /// over `t(a, b)` yields a single-column `b` projection above the Scan.
    #[test]
    fn rel_common_emit_reorders_columns() {
        let (provider, functions) = fixtures();
        let ctx = ctx(&provider, &functions);
        let mut read = read_rel_inner();
        read.common = Some(RelCommon {
            emit_kind: Some(EmitKind::Emit(Emit {
                output_mapping: vec![1], // select only column `b`
            })),
            ..Default::default()
        });
        let rel = Rel {
            rel_type: Some(RelType::Read(Box::new(read))),
        };
        let plan = convert_rel(&rel, &ctx).expect("read with emit converts");
        match plan {
            LogicalPlan::Project { input, exprs } => {
                assert!(matches!(*input, LogicalPlan::Scan { .. }));
                assert_eq!(exprs.len(), 1, "emit [1] selects exactly one column");
                assert!(
                    matches!(&exprs[0], Expr::Column(n) if n == "b"),
                    "emit [1] over t(a,b) should select column b, got {:?}",
                    exprs[0]
                );
            }
            other => panic!("expected Project over Scan, got {other:?}"),
        }
    }

    /// An out-of-range emit ordinal is a hard error.
    #[test]
    fn rel_common_emit_out_of_range_rejected() {
        let (provider, functions) = fixtures();
        let ctx = ctx(&provider, &functions);
        let mut read = read_rel_inner();
        read.common = Some(RelCommon {
            emit_kind: Some(EmitKind::Emit(Emit {
                output_mapping: vec![5], // t has only 2 columns
            })),
            ..Default::default()
        });
        let rel = Rel {
            rel_type: Some(RelType::Read(Box::new(read))),
        };
        assert!(convert_rel(&rel, &ctx).is_err());
    }

    /// A `Direct` emit is pass-through: the Scan is returned unchanged.
    #[test]
    fn rel_common_direct_emit_is_passthrough() {
        let (provider, functions) = fixtures();
        let ctx = ctx(&provider, &functions);
        let mut read = read_rel_inner();
        read.common = Some(RelCommon {
            emit_kind: Some(EmitKind::Direct(Default::default())),
            ..Default::default()
        });
        let rel = Rel {
            rel_type: Some(RelType::Read(Box::new(read))),
        };
        let plan = convert_rel(&rel, &ctx).expect("read with direct emit converts");
        assert!(
            matches!(plan, LogicalPlan::Scan { .. }),
            "Direct emit should pass through unchanged, got {plan:?}"
        );
    }

    /// A per-measure `Measure.filter` has no faithful representation and is
    /// rejected.
    #[test]
    fn measure_filter_rejected() {
        use substrait::proto::{
            aggregate_rel::{Grouping, Measure},
            AggregateFunction, AggregateRel,
        };
        let (provider, functions) = fixtures();
        // Add a `sum` aggregate anchor for the measure.
        let mut functions = functions;
        functions.insert(7u32, "sum:i64".to_string());
        let ctx = ctx(&provider, &functions);

        let measure = Measure {
            measure: Some(AggregateFunction {
                function_reference: 7,
                arguments: vec![FunctionArgument {
                    arg_type: Some(substrait::proto::function_argument::ArgType::Value(field_ref(
                        0,
                    ))),
                }],
                ..Default::default()
            }),
            filter: Some(eq_a_b()),
        };
        let agg = Rel {
            rel_type: Some(RelType::Aggregate(Box::new(AggregateRel {
                input: Some(Box::new(read_rel())),
                groupings: vec![Grouping::default()],
                measures: vec![measure],
                ..Default::default()
            }))),
        };
        let err = convert_rel(&agg, &ctx).expect_err("Measure.filter must be rejected");
        assert!(matches!(err, BoltError::Unsupported(_)), "got {err:?}");
    }

    /// `JoinRel.post_join_filter` is honoured as a `Filter` above the `Join`.
    #[test]
    fn join_post_join_filter_becomes_filter_over_join() {
        use substrait::proto::{join_rel::JoinType as SJoinType, JoinRel};
        let (provider, functions) = fixtures();
        let ctx = ctx(&provider, &functions);

        // post_join_filter `#0 = #1` references the combined (left ++ right)
        // schema; with t(a,b) on both sides, #0 = left.a, #1 = left.b.
        let join = Rel {
            rel_type: Some(RelType::Join(Box::new(JoinRel {
                left: Some(Box::new(read_rel())),
                right: Some(Box::new(read_rel())),
                r#type: SJoinType::Inner as i32,
                post_join_filter: Some(Box::new(eq_a_b())),
                ..Default::default()
            }))),
        };
        let plan = convert_rel(&join, &ctx).expect("join with post_join_filter converts");
        match plan {
            LogicalPlan::Filter { input, .. } => assert!(
                matches!(*input, LogicalPlan::Join { .. }),
                "post_join_filter should wrap the Join in a Filter"
            ),
            other => panic!("expected Filter over Join, got {other:?}"),
        }
    }
}
