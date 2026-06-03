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
//! # Contract with the converter context ([`super::Ctx`], owned by B2a/mod.rs)
//!
//! Relation conversion needs three things from the shared converter context:
//!
//! 1. **Table-schema resolution.** [`ReadRel`] carries a `NamedTable` whose
//!    `names` identify a registered table; we resolve its [`Schema`] via the
//!    context (which wraps the engine's [`TableProvider`]).
//! 2. **The "current input schema".** Substrait expressions reference columns
//!    *positionally* (`FieldReference` → struct field index), not by name. The
//!    sibling `convert_expr` therefore needs to know the schema of the relation
//!    it is being evaluated against so it can turn an ordinal into a
//!    [`Expr::Column`] with the right name. We push/pop that schema around each
//!    recursive descent through [`Ctx::with_input_schema`].
//! 3. **Extension-function resolution.** Substrait aggregate / scalar functions
//!    are referenced by a numeric `function_reference` anchor that indexes into
//!    the plan's extension declarations. The context resolves an anchor to a
//!    canonical function name via [`Ctx::function_name`].
//!
//! The concrete `Ctx` type is owned by the `mod.rs` author (task B2a). This
//! module is written against the following **minimal required surface** (see
//! the `integration_notes` in the task summary):
//!
//! ```ignore
//! impl Ctx {
//!     /// Resolve a NamedTable's fully-qualified name parts to a base schema.
//!     pub(crate) fn resolve_table(&self, names: &[String])
//!         -> BoltResult<(String, Schema)>;
//!     /// The schema of the relation currently being converted, if any.
//!     pub(crate) fn input_schema(&self) -> Option<&Schema>;
//!     /// Run `f` with `schema` installed as the current input schema,
//!     /// restoring the previous one afterwards.
//!     pub(crate) fn with_input_schema<T>(
//!         &mut self,
//!         schema: Schema,
//!         f: impl FnOnce(&mut Ctx) -> BoltResult<T>,
//!     ) -> BoltResult<T>;
//!     /// Canonical function name for an extension function anchor.
//!     pub(crate) fn function_name(&self, anchor: u32) -> BoltResult<String>;
//! }
//! ```
//!
//! and against this **sibling expr.rs surface**:
//!
//! ```ignore
//! pub(crate) fn convert_expr(
//!     expr: &substrait::proto::Expression,
//!     ctx: &mut Ctx,
//! ) -> BoltResult<Expr>;
//! ```
//!
//! If B2a's final `Ctx` differs, only the thin adapter calls in this file need
//! to change — the conversion logic stays intact.

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{
    AggregateExpr, Expr, JoinType, LogicalPlan, Schema, SortExpr,
};

use super::expr::convert_expr;
use super::Ctx;

use substrait::proto::{
    aggregate_function::AggregationInvocation,
    aggregate_rel::Measure,
    rel::RelType,
    sort_field::{SortDirection, SortKind},
    AggregateFunction, AggregateRel, FetchRel, FilterRel, JoinRel, ProjectRel, ReadRel, Rel,
    SortField, SortRel,
};

/// Convert a Substrait [`Rel`] node into the engine's [`LogicalPlan`].
///
/// Recurses on relational inputs and delegates every scalar sub-expression to
/// the sibling [`convert_expr`]. Unsupported relation kinds surface a
/// [`BoltError::Plan`] naming the operator so the caller gets a precise
/// diagnostic rather than a generic failure.
pub(crate) fn convert_rel(rel: &Rel, ctx: &mut Ctx) -> BoltResult<LogicalPlan> {
    let rel_type = rel
        .rel_type
        .as_ref()
        .ok_or_else(|| BoltError::Plan("Substrait Rel has no rel_type set".into()))?;

    match rel_type {
        RelType::Read(read) => convert_read(read, ctx),
        RelType::Filter(filter) => convert_filter(filter, ctx),
        RelType::Project(project) => convert_project(project, ctx),
        RelType::Aggregate(agg) => convert_aggregate(agg, ctx),
        RelType::Sort(sort) => convert_sort(sort, ctx),
        RelType::Join(join) => convert_join(join, ctx),
        RelType::Fetch(fetch) => convert_fetch(fetch, ctx),
        // Relation kinds outside the current ingestion envelope. Each carries
        // a distinct message so the user knows exactly which operator tripped.
        other => Err(BoltError::Plan(format!(
            "Substrait relation kind {} is not supported by the engine yet",
            rel_type_name(other)
        ))),
    }
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

/// `ReadRel` → [`LogicalPlan::Scan`].
///
/// Only the `NamedTable` read-type is supported (the common case for
/// engine-to-engine interchange). `VirtualTable`, `LocalFiles` and
/// `ExtensionTable` sources are rejected. The table name is resolved against
/// the converter context's provider to recover the base [`Schema`]; we trust
/// the registered schema over the (optional) `base_schema` carried in the
/// message so downstream type-checking sees exactly the columns the engine
/// knows about.
fn convert_read(read: &ReadRel, ctx: &mut Ctx) -> BoltResult<LogicalPlan> {
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
    };

    if names.is_empty() {
        return Err(BoltError::Plan(
            "Substrait ReadRel NamedTable has no name parts".into(),
        ));
    }

    let (table, schema) = ctx.resolve_table(names)?;

    Ok(LogicalPlan::Scan {
        table,
        // A `ReadRel.projection` mask exists in Substrait but is optional and
        // expressed as a field-select emit list; we leave projection pushdown
        // to a later pass and read all columns here. TODO: honour
        // `read.projection` (an emit mask) by mapping the selected ordinals to
        // their column names.
        projection: None,
        schema,
    })
}

/// `FilterRel` → [`LogicalPlan::Filter`].
///
/// Recurses on the input, then converts the boolean `condition` against the
/// input's schema (pushed onto the context so positional field references
/// resolve correctly).
fn convert_filter(filter: &FilterRel, ctx: &mut Ctx) -> BoltResult<LogicalPlan> {
    let input = convert_boxed_input(filter.input.as_deref(), ctx, "FilterRel")?;
    let input_schema = input.schema()?;

    let condition = filter
        .condition
        .as_deref()
        .ok_or_else(|| BoltError::Plan("Substrait FilterRel has no condition".into()))?;

    let predicate = ctx.with_input_schema(input_schema, |ctx| convert_expr(condition, ctx))?;

    Ok(LogicalPlan::Filter {
        input: Box::new(input),
        predicate,
    })
}

/// `ProjectRel` → [`LogicalPlan::Project`].
///
/// Substrait's `ProjectRel` *appends* its computed expressions to the input's
/// columns (the `emit` list then chooses the final visible set). For the core
/// slice we treat `expressions` as the full projection list — the common shape
/// emitted by producers that set an explicit emit. TODO: honour
/// `RelCommon.emit` to drop pass-through input columns when the producer asked
/// for a strict subset.
fn convert_project(project: &ProjectRel, ctx: &mut Ctx) -> BoltResult<LogicalPlan> {
    let input = convert_boxed_input(project.input.as_deref(), ctx, "ProjectRel")?;
    let input_schema = input.schema()?;

    let exprs = ctx.with_input_schema(input_schema, |ctx| {
        project
            .expressions
            .iter()
            .map(|e| convert_expr(e, ctx))
            .collect::<BoltResult<Vec<_>>>()
    })?;

    if exprs.is_empty() {
        return Err(BoltError::Plan(
            "Substrait ProjectRel has no expressions".into(),
        ));
    }

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
fn convert_aggregate(agg: &AggregateRel, ctx: &mut Ctx) -> BoltResult<LogicalPlan> {
    let input = convert_boxed_input(agg.input.as_deref(), ctx, "AggregateRel")?;
    let input_schema = input.schema()?;

    if agg.groupings.len() > 1 {
        return Err(BoltError::Plan(format!(
            "Substrait AggregateRel with {} grouping sets is not supported \
             (only a single flat GROUP BY is modelled)",
            agg.groupings.len()
        )));
    }

    let (group_by, aggregates) = ctx.with_input_schema(input_schema, |ctx| {
        let mut group_by = Vec::new();
        if let Some(grouping) = agg.groupings.first() {
            for e in &grouping.grouping_expressions {
                group_by.push(convert_expr(e, ctx)?);
            }
        }

        let mut aggregates = Vec::with_capacity(agg.measures.len());
        for measure in &agg.measures {
            aggregates.push(convert_measure(measure, ctx)?);
        }
        Ok((group_by, aggregates))
    })?;

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
fn convert_measure(measure: &Measure, ctx: &mut Ctx) -> BoltResult<AggregateExpr> {
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
    ctx: &mut Ctx,
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
fn convert_sort(sort: &SortRel, ctx: &mut Ctx) -> BoltResult<LogicalPlan> {
    let input = convert_boxed_input(sort.input.as_deref(), ctx, "SortRel")?;
    let input_schema = input.schema()?;

    if sort.sorts.is_empty() {
        return Err(BoltError::Plan(
            "Substrait SortRel has no sort fields".into(),
        ));
    }

    let sort_exprs = ctx.with_input_schema(input_schema, |ctx| {
        sort.sorts
            .iter()
            .map(|sf| convert_sort_field(sf, ctx))
            .collect::<BoltResult<Vec<_>>>()
    })?;

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
fn convert_sort_field(sf: &SortField, ctx: &mut Ctx) -> BoltResult<SortExpr> {
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
fn convert_join(join: &JoinRel, ctx: &mut Ctx) -> BoltResult<LogicalPlan> {
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

    let filter = match join.expression.as_deref() {
        Some(expr) => {
            let converted = ctx.with_input_schema(combined, |ctx| convert_expr(expr, ctx))?;
            Some(converted)
        }
        None => None,
    };

    Ok(LogicalPlan::Join {
        left: Box::new(left),
        right: Box::new(right),
        join_type,
        on: Vec::new(),
        filter,
    })
}

/// `FetchRel` → [`LogicalPlan::Limit`].
///
/// Substrait's `offset` / `count` are `i64`; a negative `count` means
/// "unbounded" in newer Substrait revisions, which the engine's `usize`-typed
/// limit cannot represent, so we reject it. A negative `offset` is invalid.
fn convert_fetch(fetch: &FetchRel, ctx: &mut Ctx) -> BoltResult<LogicalPlan> {
    let input = convert_boxed_input(fetch.input.as_deref(), ctx, "FetchRel")?;

    if fetch.count < 0 {
        return Err(BoltError::Plan(
            "Substrait FetchRel with an unbounded/negative count is not supported".into(),
        ));
    }
    if fetch.offset < 0 {
        return Err(BoltError::Plan(
            "Substrait FetchRel has a negative offset".into(),
        ));
    }

    Ok(LogicalPlan::Limit {
        input: Box::new(input),
        limit: fetch.count as usize,
        offset: fetch.offset as usize,
    })
}

/// Shared helper: recurse into a boxed `Option<Rel>` input, mapping a missing
/// input to a precise [`BoltError::Plan`] naming the parent relation.
fn convert_boxed_input(
    input: Option<&Rel>,
    ctx: &mut Ctx,
    parent: &str,
) -> BoltResult<LogicalPlan> {
    let rel =
        input.ok_or_else(|| BoltError::Plan(format!("Substrait {parent} has no input")))?;
    convert_rel(rel, ctx)
}

#[cfg(test)]
mod tests {
    //! Host-side unit coverage for relation conversion. These tests build
    //! Substrait proto messages by hand (no network / no real plan producer)
    //! and assert the resulting [`LogicalPlan`] shape.
    //!
    //! NOTE: the tests below depend on the `Ctx` constructor / helpers owned
    //! by `mod.rs` (task B2a). They are written against the documented
    //! minimal `Ctx` surface; if B2a's constructor differs, only the
    //! `test_ctx()` helper needs adjusting.

    use super::*;
    use crate::plan::logical_plan::{DataType, Field};
    use substrait::proto::{
        expression::{
            field_reference::{ReferenceType, RootReference},
            reference_segment, FieldReference, ReferenceSegment, RexType,
        },
        read_rel::{NamedTable, ReadType},
        Expression, FilterRel, ReadRel, Rel,
    };

    /// Build a `t(a Int64, b Int64)` schema for the fixture table.
    fn fixture_schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ])
    }

    /// A converter context seeded with one table `t`. Constructed via the
    /// test-only helper B2a's mod.rs is expected to expose
    /// (`Ctx::for_test(table_name, schema)`); see integration_notes.
    fn test_ctx() -> Ctx {
        Ctx::for_test("t", fixture_schema())
    }

    /// A bare positional column reference `#idx` (Substrait field reference
    /// with a single struct-field selection rooted at the input row).
    fn field_ref(idx: i32) -> Expression {
        Expression {
            rex_type: Some(RexType::Selection(Box::new(FieldReference {
                reference_type: Some(ReferenceType::DirectReference(ReferenceSegment {
                    reference_type: Some(reference_segment::ReferenceType::StructField(
                        Box::new(reference_segment::StructField {
                            field: idx,
                            child: None,
                        }),
                    )),
                })),
                root_type: Some(RootReference::RootReference(
                    substrait::proto::expression::field_reference::RootReference {},
                )),
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

    #[test]
    fn read_rel_becomes_scan() {
        let mut ctx = test_ctx();
        let plan = convert_rel(&read_rel(), &mut ctx).expect("read converts");
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
        let mut ctx = test_ctx();
        // condition: `#0 = #1`  (a = b) — build via the equal scalar fn so we
        // exercise the expr.rs path; if expr.rs maps a binary eq differently
        // this still asserts the *relational* shape (Filter wrapping Scan).
        let condition = Expression {
            rex_type: Some(RexType::ScalarFunction(
                substrait::proto::expression::ScalarFunction {
                    function_reference: 0, // resolves to "equal" in test ctx
                    arguments: vec![
                        substrait::proto::FunctionArgument {
                            arg_type: Some(
                                substrait::proto::function_argument::ArgType::Value(
                                    field_ref(0),
                                ),
                            ),
                        },
                        substrait::proto::FunctionArgument {
                            arg_type: Some(
                                substrait::proto::function_argument::ArgType::Value(
                                    field_ref(1),
                                ),
                            ),
                        },
                    ],
                    ..Default::default()
                },
            )),
        };

        let filter = Rel {
            rel_type: Some(RelType::Filter(Box::new(FilterRel {
                input: Some(Box::new(read_rel())),
                condition: Some(Box::new(condition)),
                ..Default::default()
            }))),
        };

        let plan = convert_rel(&filter, &mut ctx).expect("filter converts");
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
        let mut ctx = test_ctx();
        let rel = Rel { rel_type: None };
        assert!(convert_rel(&rel, &mut ctx).is_err());
    }
}
