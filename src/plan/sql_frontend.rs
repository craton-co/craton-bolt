// SPDX-License-Identifier: Apache-2.0

//! SQL frontend: parses a SQL string into a `LogicalPlan` against a `TableProvider`.

use std::collections::HashMap;

use sqlparser::ast::{
    BinaryOperator, Distinct, Expr as SqlExpr, FunctionArg, FunctionArgExpr, FunctionArguments,
    GroupByExpr, Ident, JoinConstraint, JoinOperator, ObjectName, Offset, OrderByExpr, Query,
    Select, SelectItem, SetExpr, SetOperator, SetQuantifier, Statement, TableFactor,
    UnaryOperator, Value,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::error::{PatinaError, PatinaResult};
use crate::plan::logical_plan::{
    AggregateExpr, BinaryOp, Expr, JoinType, Literal, LogicalPlan, Schema, SortExpr,
};

/// Resolves table names to their schemas; the SQL frontend cannot know table shapes otherwise.
pub trait TableProvider {
    /// Return the schema for `name`, or a `Plan` error if the table is unknown.
    fn schema(&self, name: &str) -> PatinaResult<Schema>;
}

/// In-memory `name → Schema` provider; useful in tests and as a default.
#[derive(Debug, Default, Clone)]
pub struct MemTableProvider {
    /// Registered tables, keyed by name.
    tables: HashMap<String, Schema>,
}

impl MemTableProvider {
    /// Empty provider.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style: register a table and return `self`.
    pub fn with_table(mut self, name: impl Into<String>, schema: Schema) -> Self {
        self.register(name, schema);
        self
    }

    /// Mutating register; overwrites any existing entry with the same name.
    pub fn register(&mut self, name: impl Into<String>, schema: Schema) {
        self.tables.insert(name.into(), schema);
    }
}

impl TableProvider for MemTableProvider {
    fn schema(&self, name: &str) -> PatinaResult<Schema> {
        self.tables
            .get(name)
            .cloned()
            .ok_or_else(|| PatinaError::Plan(format!("unknown table '{name}'")))
    }
}

/// Parse a SQL string into a single `LogicalPlan` using the given provider.
pub fn parse(sql: &str, provider: &dyn TableProvider) -> PatinaResult<LogicalPlan> {
    let dialect = GenericDialect {};
    let mut stmts = Parser::parse_sql(&dialect, sql).map_err(|e| PatinaError::Sql(e.to_string()))?;

    if stmts.len() != 1 {
        return Err(PatinaError::Sql(format!(
            "expected exactly one statement, got {}",
            stmts.len()
        )));
    }
    let stmt = stmts.remove(0);
    let query = match stmt {
        Statement::Query(q) => q,
        other => {
            return Err(PatinaError::Sql(format!(
                "only SELECT queries are supported, got: {other}"
            )));
        }
    };
    plan_query(&query, provider)
}

/// Lower a top-level `Query`. Supports SELECT, UNION [ALL], ORDER BY, LIMIT,
/// and OFFSET. Rejects CTEs, FETCH, locks, EXCEPT/INTERSECT, and dialect
/// extensions outside our subset.
fn plan_query(query: &Query, provider: &dyn TableProvider) -> PatinaResult<LogicalPlan> {
    if query.with.is_some() {
        return Err(PatinaError::Sql("unsupported: WITH / CTEs".into()));
    }
    if !query.limit_by.is_empty() {
        return Err(PatinaError::Sql("unsupported: LIMIT BY".into()));
    }
    if query.fetch.is_some() {
        return Err(PatinaError::Sql("unsupported: FETCH".into()));
    }
    if !query.locks.is_empty() {
        return Err(PatinaError::Sql("unsupported: FOR UPDATE/SHARE".into()));
    }
    if query.for_clause.is_some() {
        return Err(PatinaError::Sql("unsupported: FOR clause".into()));
    }
    if query.settings.is_some() {
        return Err(PatinaError::Sql("unsupported: SETTINGS clause".into()));
    }
    if query.format_clause.is_some() {
        return Err(PatinaError::Sql("unsupported: FORMAT clause".into()));
    }

    // Lower the body into a base plan; UNION/UNION ALL builds a `Union` (and
    // optionally a `Distinct` wrapper) here, so the ORDER BY / LIMIT layers
    // below apply to the *combined* result, matching SQL semantics.
    let mut plan = lower_set_expr(query.body.as_ref(), provider)?;

    // ORDER BY: appended *outside* the body so it sees the final schema.
    if let Some(order_by) = &query.order_by {
        let sort_exprs = lower_order_by(&order_by.exprs)?;
        if !sort_exprs.is_empty() {
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                sort_exprs,
            };
        }
    }

    // LIMIT [OFFSET]: fold both into a single `Limit` node so a downstream
    // executor can implement the offset as a skip without needing a separate
    // operator. Either clause alone is legal; OFFSET without LIMIT is
    // represented as `Limit { limit: usize::MAX, offset }`.
    let limit_value = match &query.limit {
        Some(e) => Some(usize_from_literal(e, "LIMIT")?),
        None => None,
    };
    let offset_value = match &query.offset {
        Some(Offset { value, .. }) => Some(usize_from_literal(value, "OFFSET")?),
        None => None,
    };
    if limit_value.is_some() || offset_value.is_some() {
        plan = LogicalPlan::Limit {
            input: Box::new(plan),
            limit: limit_value.unwrap_or(usize::MAX),
            offset: offset_value.unwrap_or(0),
        };
    }

    Ok(plan)
}

/// Lower a `SetExpr` (SELECT body or UNION/EXCEPT/INTERSECT node) into a
/// `LogicalPlan`. UNION ALL becomes `Union { inputs }`; plain UNION becomes
/// `Distinct(Union { inputs })`. EXCEPT/INTERSECT are rejected.
fn lower_set_expr(expr: &SetExpr, provider: &dyn TableProvider) -> PatinaResult<LogicalPlan> {
    match expr {
        SetExpr::Select(s) => plan_select(s.as_ref(), provider),
        SetExpr::Query(q) => plan_query(q.as_ref(), provider),
        SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } => {
            if *op != SetOperator::Union {
                return Err(PatinaError::Sql(format!(
                    "unsupported set operator: {op}; only UNION / UNION ALL"
                )));
            }
            // Reject the BY NAME variants (non-standard, schema-rewriting).
            let dedup = match set_quantifier {
                SetQuantifier::All => false,
                SetQuantifier::Distinct | SetQuantifier::None => true,
                SetQuantifier::ByName
                | SetQuantifier::AllByName
                | SetQuantifier::DistinctByName => {
                    return Err(PatinaError::Sql(
                        "unsupported: UNION BY NAME".into(),
                    ));
                }
            };
            // Flatten left-recursive UNION chains into a single Union node so
            // `q1 UNION ALL q2 UNION ALL q3` becomes one 3-input Union rather
            // than a nested binary tree. UNION (dedup) does NOT flatten across
            // UNION ALL boundaries: their semantics differ.
            let mut inputs: Vec<LogicalPlan> = Vec::new();
            collect_union_branches(left, provider, dedup, &mut inputs)?;
            collect_union_branches(right, provider, dedup, &mut inputs)?;
            let union = LogicalPlan::Union { inputs };
            Ok(if dedup {
                LogicalPlan::Distinct {
                    input: Box::new(union),
                }
            } else {
                union
            })
        }
        SetExpr::Values(_) => Err(PatinaError::Sql("unsupported: VALUES".into())),
        SetExpr::Insert(_) | SetExpr::Update(_) => Err(PatinaError::Sql(
            "unsupported: write statement in query body".into(),
        )),
        SetExpr::Table(_) => Err(PatinaError::Sql("unsupported: TABLE statement".into())),
    }
}

/// Helper for `lower_set_expr`: if `expr` is itself a same-quantifier UNION,
/// recurse to collect its operands directly into `out`; otherwise lower it
/// as a single branch. `parent_dedup` indicates whether the enclosing UNION
/// is a dedup variant (so we only flatten matching-quantifier children).
fn collect_union_branches(
    expr: &SetExpr,
    provider: &dyn TableProvider,
    parent_dedup: bool,
    out: &mut Vec<LogicalPlan>,
) -> PatinaResult<()> {
    if let SetExpr::SetOperation {
        op: SetOperator::Union,
        set_quantifier,
        left,
        right,
    } = expr
    {
        let child_dedup = match set_quantifier {
            SetQuantifier::All => false,
            SetQuantifier::Distinct | SetQuantifier::None => true,
            // Non-flattening cases — fall through to a non-flat lower.
            SetQuantifier::ByName
            | SetQuantifier::AllByName
            | SetQuantifier::DistinctByName => {
                out.push(lower_set_expr(expr, provider)?);
                return Ok(());
            }
        };
        if child_dedup == parent_dedup {
            collect_union_branches(left, provider, parent_dedup, out)?;
            collect_union_branches(right, provider, parent_dedup, out)?;
            return Ok(());
        }
    }
    out.push(lower_set_expr(expr, provider)?);
    Ok(())
}

/// Lower a list of `OrderByExpr` into our `SortExpr`s. The default sort
/// direction is ASC; the default NULL placement follows SQL convention
/// (NULLS FIRST for ASC, NULLS LAST for DESC) when the user omits it.
fn lower_order_by(exprs: &[OrderByExpr]) -> PatinaResult<Vec<SortExpr>> {
    let mut out = Vec::with_capacity(exprs.len());
    for OrderByExpr {
        expr,
        asc,
        nulls_first,
        with_fill,
    } in exprs
    {
        if with_fill.is_some() {
            return Err(PatinaError::Sql(
                "unsupported: ORDER BY ... WITH FILL".into(),
            ));
        }
        let descending = match asc {
            Some(true) | None => false,
            Some(false) => true,
        };
        // Default NULL placement: NULLS FIRST for ASC, NULLS LAST for DESC.
        let nulls_first = match nulls_first {
            Some(b) => *b,
            None => !descending,
        };
        out.push(SortExpr {
            expr: lower_expr(expr)?,
            descending,
            nulls_first,
        });
    }
    Ok(out)
}

/// Parse a SQL `LIMIT` / `OFFSET` clause value into a `usize`. The clause
/// must be a non-negative integer literal; anything else is rejected (no
/// dynamic LIMITs, no expressions). `kind` is used for error messages.
fn usize_from_literal(e: &SqlExpr, kind: &str) -> PatinaResult<usize> {
    let value = match e {
        SqlExpr::Value(Value::Number(n, _)) => n,
        other => {
            return Err(PatinaError::Sql(format!(
                "{kind} must be an integer literal, got: {other}"
            )));
        }
    };
    let parsed: i64 = value.parse().map_err(|_| {
        PatinaError::Sql(format!("{kind} value '{value}' is not a valid integer"))
    })?;
    if parsed < 0 {
        return Err(PatinaError::Sql(format!(
            "{kind} value must be non-negative, got {parsed}"
        )));
    }
    usize::try_from(parsed)
        .map_err(|_| PatinaError::Sql(format!("{kind} value {parsed} exceeds usize range")))
}

/// Lower a `Select` into Scan [→ Filter] → (Project | Aggregate), optionally
/// wrapped in `Filter` (for HAVING) and/or `Distinct` (for SELECT DISTINCT).
/// Supports a single INNER JOIN in FROM.
fn plan_select(select: &Select, provider: &dyn TableProvider) -> PatinaResult<LogicalPlan> {
    reject_unsupported_select(select)?;

    // FROM: exactly one base table reference. JOINs hang off `twj.joins`.
    if select.from.len() != 1 {
        return Err(PatinaError::Sql(format!(
            "expected exactly one FROM table, got {}",
            select.from.len()
        )));
    }
    let twj = &select.from[0];

    // Build the base Scan from the first table reference.
    let (table_name, scan_schema) = lower_table_factor(&twj.relation, provider)?;
    let schema = scan_schema.clone();
    let mut plan = LogicalPlan::Scan {
        table: table_name,
        projection: None,
        schema,
    };

    // JOIN handling. We support a single INNER JOIN with an equi-conjunction
    // ON predicate; the join's right side must itself be a bare table.
    // The wave-7 executor scaffold rejects anything more elaborate.
    for join in &twj.joins {
        if join.global {
            return Err(PatinaError::Sql(
                "unsupported: GLOBAL JOIN (ClickHouse extension)".into(),
            ));
        }
        let on_expr = match &join.join_operator {
            JoinOperator::Inner(JoinConstraint::On(e)) => e,
            JoinOperator::Inner(JoinConstraint::Using(_)) => {
                return Err(PatinaError::Sql(
                    "unsupported: JOIN ... USING; rewrite as ON".into(),
                ));
            }
            JoinOperator::Inner(JoinConstraint::Natural) => {
                return Err(PatinaError::Sql("unsupported: NATURAL JOIN".into()));
            }
            JoinOperator::Inner(JoinConstraint::None) => {
                return Err(PatinaError::Sql(
                    "INNER JOIN requires an ON clause".into(),
                ));
            }
            JoinOperator::CrossJoin => {
                return Err(PatinaError::Sql(
                    "unsupported: CROSS JOIN; rewrite with explicit ON".into(),
                ));
            }
            other => {
                return Err(PatinaError::Sql(format!(
                    "unsupported join kind: {other:?}; only INNER JOIN is supported"
                )));
            }
        };
        let (rhs_table, rhs_schema) = lower_table_factor(&join.relation, provider)?;
        let right_plan = LogicalPlan::Scan {
            table: rhs_table,
            projection: None,
            schema: rhs_schema,
        };
        let on_pairs = lower_join_on(on_expr)?;
        plan = LogicalPlan::Join {
            left: Box::new(plan),
            right: Box::new(right_plan),
            join_type: JoinType::Inner,
            on: on_pairs,
        };
    }
    // After a JOIN, the namespace for WHERE / SELECT items widens to the
    // join's output schema. The scan_schema below is still used for wildcard
    // expansion when no JOIN is present; when a JOIN *is* present, wildcard
    // expansion uses the join's full schema. We compute `scan_schema_for_wildcard`
    // for the wildcard-expansion branch below.
    let scan_schema_for_wildcard: Schema = if twj.joins.is_empty() {
        scan_schema.clone()
    } else {
        plan.schema()?
    };

    // WHERE
    if let Some(filter_sql) = &select.selection {
        let predicate = lower_expr(filter_sql)?;
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate,
        };
    }

    // GROUP BY (must precede projection decision)
    let group_by_sql: Vec<&SqlExpr> = match &select.group_by {
        GroupByExpr::All(_) => {
            return Err(PatinaError::Sql("unsupported: GROUP BY ALL".into()));
        }
        GroupByExpr::Expressions(exprs, modifiers) => {
            if !modifiers.is_empty() {
                return Err(PatinaError::Sql(
                    "unsupported: GROUP BY modifiers (ROLLUP/CUBE/TOTALS)".into(),
                ));
            }
            exprs.iter().collect()
        }
    };

    // Expand SELECT items into (expr, optional alias). Wildcards expand to columns
    // of the scan's full schema.
    let mut items: Vec<(SqlExpr, Option<String>)> = Vec::new();
    for item in &select.projection {
        match item {
            SelectItem::UnnamedExpr(e) => items.push((e.clone(), None)),
            SelectItem::ExprWithAlias { expr, alias } => {
                items.push((expr.clone(), Some(alias.value.clone())))
            }
            SelectItem::Wildcard(_) => {
                for f in &scan_schema_for_wildcard.fields {
                    items.push((SqlExpr::Identifier(Ident::new(f.name.clone())), None));
                }
            }
            SelectItem::QualifiedWildcard(_, _) => {
                return Err(PatinaError::Sql("unsupported: qualified wildcard".into()));
            }
        }
    }

    let has_agg_in_select = items
        .iter()
        .map(|(e, _)| try_aggregate(e))
        .collect::<PatinaResult<Vec<_>>>()?
        .iter()
        .any(|o| o.is_some());

    if has_agg_in_select || !group_by_sql.is_empty() {
        // Aggregate mode. Simplification: every selected item is either a bare
        // aggregate call or a bare group key already listed in GROUP BY. Mixed
        // post-aggregate scalar work (e.g. `SUM(a) + 1`) is rejected up front.
        let group_by: Vec<Expr> = group_by_sql
            .iter()
            .map(|e| lower_expr(e))
            .collect::<PatinaResult<_>>()?;

        let mut aggregates: Vec<AggregateExpr> = Vec::new();
        // For each SELECT item, remember how to pull it back out of the Aggregate
        // node's schema (group keys first, aggregates second per `Aggregate::schema()`).
        // Each entry is the *output* column name produced by the Aggregate, plus an
        // optional SELECT alias to rename it to in the final projection.
        enum SelectSource {
            /// SELECT references a group key; pull by the key's name in the Aggregate schema.
            GroupKey { key_name: String, alias: Option<String> },
            /// SELECT references the Nth aggregate in `aggregates`.
            Aggregate { index: usize },
        }
        let mut select_sources: Vec<SelectSource> = Vec::new();

        for (sql_expr, alias) in &items {
            if let Some(agg) = try_aggregate(sql_expr)? {
                if alias.is_some() {
                    return Err(PatinaError::Sql(
                        "unsupported: alias on aggregate expression".into(),
                    ));
                }
                let idx = aggregates.len();
                aggregates.push(agg);
                select_sources.push(SelectSource::Aggregate { index: idx });
                continue;
            }
            // Non-aggregate: must contain no nested aggregate (no post-aggregate exprs).
            if contains_aggregate(sql_expr)? {
                return Err(PatinaError::Sql(
                    "post-aggregate expressions not yet supported".into(),
                ));
            }
            let lowered = lower_expr(sql_expr)?;
            // Must match some declared GROUP BY key by structural equality of the lowered form.
            if !group_by.iter().any(|g| expr_eq(g, &lowered)) {
                return Err(PatinaError::Sql(
                    "non-aggregate SELECT expression must appear in GROUP BY".into(),
                ));
            }
            // Determine the *output* column name this key receives inside the Aggregate's
            // schema. Must mirror the naming rule in `LogicalPlan::schema()` for the
            // Aggregate arm: bare Column => its name, Alias => its name, else `__group_{i}`.
            // The aggregate plan's group_by list is the GROUP BY clause itself (not the
            // SELECT list), so we look up the matching key there to compute its position
            // and apply the same naming convention.
            let key_pos = group_by
                .iter()
                .position(|g| expr_eq(g, &lowered))
                .expect("matched above");
            let key_name = group_key_output_name(&group_by[key_pos], key_pos);
            select_sources.push(SelectSource::GroupKey {
                key_name,
                alias: alias.clone(),
            });
        }

        // The plan's `group_by` is the SQL GROUP BY list (not the SELECT keys);
        // this matches LogicalPlan::Aggregate's contract and types-checks the keys
        // even if SELECT names only a subset of them.
        let aggregate_plan = LogicalPlan::Aggregate {
            input: Box::new(plan),
            group_by,
            aggregates,
        };

        // Re-project the aggregate's output to honour SELECT-list column order
        // (Aggregate::schema places keys first, aggregates second — independent
        // of the user's SELECT order, which would silently swap columns).
        //
        // Aggregate output names follow `AggregateExpr::output_name()` in
        // `logical_plan.rs` (e.g. SUM(x) -> "sum_x", COUNT(*) -> "count"). That
        // method is private, so we mirror the same convention here in
        // `aggregate_output_name`. Group-key names mirror the rule in
        // `LogicalPlan::schema()` for the Aggregate arm.
        let aggregates_out: &[AggregateExpr] = match &aggregate_plan {
            LogicalPlan::Aggregate { aggregates, .. } => aggregates,
            _ => unreachable!("just constructed an Aggregate"),
        };
        let mut proj_exprs: Vec<Expr> = Vec::with_capacity(select_sources.len());
        for src in &select_sources {
            match src {
                SelectSource::GroupKey { key_name, alias } => {
                    let col = Expr::Column(key_name.clone());
                    proj_exprs.push(match alias {
                        Some(a) => col.alias(a.clone()),
                        None => col,
                    });
                }
                SelectSource::Aggregate { index } => {
                    let name = aggregate_output_name(&aggregates_out[*index]);
                    proj_exprs.push(Expr::Column(name));
                }
            }
        }

        plan = LogicalPlan::Project {
            input: Box::new(aggregate_plan),
            exprs: proj_exprs,
        };
    } else {
        // Scalar projection mode.
        if select.having.is_some() {
            return Err(PatinaError::Sql(
                "HAVING requires GROUP BY or aggregate functions in SELECT".into(),
            ));
        }
        let mut exprs = Vec::with_capacity(items.len());
        for (sql_expr, alias) in items {
            let lowered = lower_expr(&sql_expr)?;
            let lowered = match alias {
                Some(name) => lowered.alias(name),
                None => lowered,
            };
            exprs.push(lowered);
        }
        plan = LogicalPlan::Project {
            input: Box::new(plan),
            exprs,
        };
    }

    // HAVING: wrap the (SELECT-ordered) projection with a Filter. The
    // predicate references aggregate output column names by the names
    // generated in `AggregateExpr::output_name` (mirrored in
    // `aggregate_output_name` above), or group-key column names.
    //
    // SQL allows aggregate function *calls* inside HAVING (e.g.
    // `HAVING SUM(price) > 100`) — the GROUP BY has already established an
    // aggregation context. We rewrite each such call into a `Column`
    // reference using the same name the SELECT-order Project produced for
    // it. Non-aggregate sub-expressions go through the regular
    // `lower_expr`, which also handles bare group-key columns.
    if let Some(having_sql) = &select.having {
        let predicate = lower_expr_in_having(having_sql)?;
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate,
        };
    }

    // SELECT DISTINCT: dedup the *output* rows (after projection, HAVING).
    if matches!(select.distinct, Some(Distinct::Distinct)) {
        plan = LogicalPlan::Distinct {
            input: Box::new(plan),
        };
    }

    Ok(plan)
}

/// Lower a single `TableFactor` into `(table_name, schema)`. Only bare
/// table references are accepted (no aliases, TVFs, version, hints, etc.).
fn lower_table_factor(
    tf: &TableFactor,
    provider: &dyn TableProvider,
) -> PatinaResult<(String, Schema)> {
    match tf {
        TableFactor::Table {
            name,
            alias,
            args,
            with_hints,
            version,
            with_ordinality,
            partitions,
        } => {
            if alias.is_some() {
                return Err(PatinaError::Sql("unsupported: table alias".into()));
            }
            if args.is_some() {
                return Err(PatinaError::Sql("unsupported: table-valued function".into()));
            }
            if !with_hints.is_empty() {
                return Err(PatinaError::Sql("unsupported: WITH hints".into()));
            }
            if version.is_some() {
                return Err(PatinaError::Sql("unsupported: table version".into()));
            }
            if *with_ordinality {
                return Err(PatinaError::Sql("unsupported: WITH ORDINALITY".into()));
            }
            if !partitions.is_empty() {
                return Err(PatinaError::Sql("unsupported: PARTITION".into()));
            }
            let table_name = single_ident_from_object_name(name)?;
            let schema = provider.schema(&table_name)?;
            Ok((table_name, schema))
        }
        _ => Err(PatinaError::Sql(
            "unsupported: only bare table references are allowed in FROM".into(),
        )),
    }
}

/// Look up a join predicate expression as a conjunction of `left.col = right.col`
/// equalities. Reject non-equi joins and non-conjunctive forms with a clear
/// message; the executor scaffold only handles equi joins.
fn lower_join_on(e: &SqlExpr) -> PatinaResult<Vec<(Expr, Expr)>> {
    let mut out = Vec::new();
    collect_join_eq(e, &mut out)?;
    if out.is_empty() {
        return Err(PatinaError::Sql(
            "JOIN ON clause must contain at least one equality predicate".into(),
        ));
    }
    Ok(out)
}

/// Walk `e` flattening `AND` nodes; each leaf must be `<expr> = <expr>`.
fn collect_join_eq(e: &SqlExpr, out: &mut Vec<(Expr, Expr)>) -> PatinaResult<()> {
    match e {
        SqlExpr::Nested(inner) => collect_join_eq(inner, out),
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            collect_join_eq(left, out)?;
            collect_join_eq(right, out)
        }
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } => {
            let l = lower_join_side(left)?;
            let r = lower_join_side(right)?;
            out.push((l, r));
            Ok(())
        }
        other => Err(PatinaError::Sql(format!(
            "non-equi JOIN not yet supported (ON clause must be a conjunction of `a = b` predicates; got {other})"
        ))),
    }
}

/// Lower one side of an equi-join predicate. We accept either a bare
/// identifier or a `table.column` qualified identifier so users can
/// disambiguate same-named columns; both lower to a plain `Column` ref
/// (qualified column lookups beyond bare-name matching aren't supported
/// in 0.1.x but the parser accepts them so error messages stay friendly).
fn lower_join_side(e: &SqlExpr) -> PatinaResult<Expr> {
    match e {
        SqlExpr::Identifier(ident) => Ok(Expr::Column(ident.value.clone())),
        SqlExpr::CompoundIdentifier(parts) => {
            // `table.col` — keep only the trailing column name. Cross-side
            // matching is the executor's job.
            let last = parts
                .last()
                .ok_or_else(|| PatinaError::Sql("empty compound identifier in JOIN ON".into()))?;
            Ok(Expr::Column(last.value.clone()))
        }
        other => Err(PatinaError::Sql(format!(
            "non-equi JOIN not yet supported (JOIN ON sides must be column references; got {other})"
        ))),
    }
}

/// Reject SELECT-level features outside our supported subset. `DISTINCT` and
/// `HAVING` are *not* rejected here — both are recognised by `plan_select`
/// and lowered into the plan.
fn reject_unsupported_select(select: &Select) -> PatinaResult<()> {
    // DISTINCT ON (...) is a Postgres extension we don't support; plain
    // SELECT DISTINCT is handled by `plan_select`.
    if let Some(Distinct::On(_)) = &select.distinct {
        return Err(PatinaError::Sql("unsupported: DISTINCT ON".into()));
    }
    if select.top.is_some() {
        return Err(PatinaError::Sql("unsupported: TOP".into()));
    }
    if select.into.is_some() {
        return Err(PatinaError::Sql("unsupported: SELECT INTO".into()));
    }
    if !select.lateral_views.is_empty() {
        return Err(PatinaError::Sql("unsupported: LATERAL VIEW".into()));
    }
    if select.prewhere.is_some() {
        return Err(PatinaError::Sql("unsupported: PREWHERE".into()));
    }
    if !select.cluster_by.is_empty() {
        return Err(PatinaError::Sql("unsupported: CLUSTER BY".into()));
    }
    if !select.distribute_by.is_empty() {
        return Err(PatinaError::Sql("unsupported: DISTRIBUTE BY".into()));
    }
    if !select.sort_by.is_empty() {
        return Err(PatinaError::Sql("unsupported: SORT BY".into()));
    }
    if !select.named_window.is_empty() {
        return Err(PatinaError::Sql("unsupported: WINDOW".into()));
    }
    if select.qualify.is_some() {
        return Err(PatinaError::Sql("unsupported: QUALIFY".into()));
    }
    if select.value_table_mode.is_some() {
        return Err(PatinaError::Sql("unsupported: SELECT AS STRUCT/VALUE".into()));
    }
    if select.connect_by.is_some() {
        return Err(PatinaError::Sql("unsupported: CONNECT BY".into()));
    }
    Ok(())
}

/// Pull a single-part identifier out of an `ObjectName`, rejecting schema-qualified names.
fn single_ident_from_object_name(name: &ObjectName) -> PatinaResult<String> {
    if name.0.len() != 1 {
        return Err(PatinaError::Sql(format!(
            "qualified table names not supported: {name}"
        )));
    }
    Ok(name.0[0].value.clone())
}

/// Recognize a top-level aggregate function call. Returns `Ok(None)` for non-aggregates.
fn try_aggregate(e: &SqlExpr) -> PatinaResult<Option<AggregateExpr>> {
    let func = match e {
        SqlExpr::Function(f) => f,
        _ => return Ok(None),
    };
    if func.name.0.len() != 1 {
        return Ok(None);
    }
    let fname = func.name.0[0].value.to_ascii_uppercase();
    let kind = match fname.as_str() {
        "COUNT" | "SUM" | "MIN" | "MAX" | "AVG" => fname,
        _ => return Ok(None),
    };

    // Disallow OVER (window), FILTER, ORDER BY, WITHIN GROUP, parameters.
    if func.over.is_some() {
        return Err(PatinaError::Sql(
            "unsupported: window functions (OVER)".into(),
        ));
    }
    if func.filter.is_some() {
        return Err(PatinaError::Sql("unsupported: FILTER on aggregate".into()));
    }
    if func.null_treatment.is_some() {
        return Err(PatinaError::Sql(
            "unsupported: IGNORE/RESPECT NULLS on aggregate".into(),
        ));
    }
    if !func.within_group.is_empty() {
        return Err(PatinaError::Sql(
            "unsupported: WITHIN GROUP on aggregate".into(),
        ));
    }
    if !matches!(func.parameters, FunctionArguments::None) {
        return Err(PatinaError::Sql(
            "unsupported: parametric aggregate function".into(),
        ));
    }

    let arg_list = match &func.args {
        FunctionArguments::List(list) => list,
        FunctionArguments::None => {
            return Err(PatinaError::Sql(format!("{kind} requires arguments")));
        }
        FunctionArguments::Subquery(_) => {
            return Err(PatinaError::Sql(format!(
                "unsupported: subquery argument to {kind}"
            )));
        }
    };
    if arg_list.duplicate_treatment.is_some() {
        return Err(PatinaError::Sql(format!(
            "unsupported: DISTINCT/ALL inside {kind}"
        )));
    }
    if !arg_list.clauses.is_empty() {
        return Err(PatinaError::Sql(format!(
            "unsupported: argument clauses on {kind}"
        )));
    }
    if arg_list.args.len() != 1 {
        return Err(PatinaError::Sql(format!(
            "{kind} expects exactly one argument, got {}",
            arg_list.args.len()
        )));
    }

    let arg_expr = match &arg_list.args[0] {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => Some(e),
        FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => None,
        FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(_)) => {
            return Err(PatinaError::Sql(format!(
                "unsupported: qualified wildcard in {kind}"
            )));
        }
        FunctionArg::Named { .. } => {
            return Err(PatinaError::Sql(format!(
                "unsupported: named argument to {kind}"
            )));
        }
    };

    let inner = match arg_expr {
        Some(e) => lower_expr(e)?,
        None => {
            if kind != "COUNT" {
                return Err(PatinaError::Sql(format!("{kind}(*) is not supported")));
            }
            // COUNT(*) sentinel: a literal 1; counted rows are independent of value.
            Expr::Literal(Literal::Int64(1))
        }
    };

    Ok(Some(match kind.as_str() {
        "COUNT" => AggregateExpr::Count(inner),
        "SUM" => AggregateExpr::Sum(inner),
        "MIN" => AggregateExpr::Min(inner),
        "MAX" => AggregateExpr::Max(inner),
        "AVG" => AggregateExpr::Avg(inner),
        _ => unreachable!("kind already filtered above"),
    }))
}

/// True if `e` contains any aggregate function call (anywhere in the tree).
fn contains_aggregate(e: &SqlExpr) -> PatinaResult<bool> {
    if try_aggregate(e)?.is_some() {
        return Ok(true);
    }
    match e {
        SqlExpr::BinaryOp { left, right, .. } => {
            Ok(contains_aggregate(left)? || contains_aggregate(right)?)
        }
        SqlExpr::UnaryOp { expr, .. } => contains_aggregate(expr),
        SqlExpr::Nested(inner) => contains_aggregate(inner),
        _ => Ok(false),
    }
}

/// Variant of `lower_expr` used inside a HAVING clause. Aggregate function
/// calls (anywhere in the tree) are rewritten into a bare `Column(name)`
/// where `name` is the column the post-aggregate Project produces for that
/// aggregate (per `aggregate_output_name`). Everything else delegates to
/// `lower_expr`, which keeps the usual rules — bare columns become column
/// refs, non-aggregate function calls are still rejected, etc.
fn lower_expr_in_having(e: &SqlExpr) -> PatinaResult<Expr> {
    if let Some(agg) = try_aggregate(e)? {
        return Ok(Expr::Column(aggregate_output_name(&agg)));
    }
    match e {
        SqlExpr::Nested(inner) => lower_expr_in_having(inner),
        SqlExpr::BinaryOp { left, op, right } => {
            let lop = lower_binary_op(op)?;
            let l = lower_expr_in_having(left)?;
            let r = lower_expr_in_having(right)?;
            Ok(Expr::Binary {
                op: lop,
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        SqlExpr::UnaryOp { op, expr } => match op {
            UnaryOperator::Plus => lower_expr_in_having(expr),
            UnaryOperator::Minus => {
                // Re-use the aggregate-aware lowerer for the operand, then
                // negate by hand (we can't fall through to `negate_expr`
                // because it would route through `lower_expr` and reject
                // any aggregate call nested under the unary minus).
                let inner = lower_expr_in_having(expr)?;
                Ok(Expr::Binary {
                    op: BinaryOp::Sub,
                    left: Box::new(Expr::Literal(Literal::Int64(0))),
                    right: Box::new(inner),
                })
            }
            other => Err(PatinaError::Sql(format!(
                "unsupported unary operator: {other:?}"
            ))),
        },
        // Anything else is identical to a scalar HAVING fragment; defer to
        // the normal lowerer (which handles Identifier, Value, etc., and
        // still rejects bare non-aggregate Function calls).
        _ => lower_expr(e),
    }
}

/// Lower a scalar SQL expression into our `Expr`. Aggregates are rejected here —
/// callers must split them off via `try_aggregate` first.
fn lower_expr(e: &SqlExpr) -> PatinaResult<Expr> {
    match e {
        SqlExpr::Identifier(ident) => Ok(Expr::Column(ident.value.clone())),
        SqlExpr::CompoundIdentifier(_) => Err(PatinaError::Sql(
            "unsupported: qualified column references (no table aliases yet)".into(),
        )),
        SqlExpr::Value(v) => lower_value(v),
        SqlExpr::Nested(inner) => lower_expr(inner),
        SqlExpr::BinaryOp { left, op, right } => {
            let lop = lower_binary_op(op)?;
            let l = lower_expr(left)?;
            let r = lower_expr(right)?;
            Ok(Expr::Binary {
                op: lop,
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        SqlExpr::UnaryOp { op, expr } => match op {
            UnaryOperator::Plus => lower_expr(expr),
            UnaryOperator::Minus => negate_expr(expr),
            other => Err(PatinaError::Sql(format!(
                "unsupported unary operator: {other:?}"
            ))),
        },
        SqlExpr::Function(_) => Err(PatinaError::Sql(
            "function calls are only allowed as top-level aggregates in SELECT".into(),
        )),
        other => Err(PatinaError::Sql(format!(
            "unsupported expression: {other}"
        ))),
    }
}

/// Translate a SQL literal `Value` into our `Literal` expression.
fn lower_value(v: &Value) -> PatinaResult<Expr> {
    match v {
        Value::Number(n, _long) => parse_number(n),
        Value::SingleQuotedString(s) => Ok(Expr::Literal(Literal::Utf8(s.clone()))),
        Value::Boolean(b) => Ok(Expr::Literal(Literal::Bool(*b))),
        Value::Null => Ok(Expr::Literal(Literal::Null)),
        other => Err(PatinaError::Sql(format!("unsupported literal: {other}"))),
    }
}

/// True if `s` is written as a pure integer (no decimal point, no exponent).
/// Used to distinguish "user meant an integer that overflows" from
/// "user wrote a float that happens to round-trip through f64".
fn looks_like_pure_integer(s: &str) -> bool {
    !s.contains('.') && !s.contains('e') && !s.contains('E')
}

/// Parse a numeric literal string into `Int64` if it fits, otherwise `Float64`.
/// Integer-looking literals that overflow `i64` are *rejected* rather than silently
/// demoted to `Float64` (which would lose precision past 2^53).
fn parse_number(n: &str) -> PatinaResult<Expr> {
    if let Ok(i) = n.parse::<i64>() {
        return Ok(Expr::Literal(Literal::Int64(i)));
    }
    if looks_like_pure_integer(n) {
        return Err(PatinaError::Sql(format!(
            "integer literal {n} out of i64 range; use scientific notation or an explicit fractional part for Float64"
        )));
    }
    match n.parse::<f64>() {
        Ok(f) => Ok(Expr::Literal(Literal::Float64(f))),
        Err(_) => Err(PatinaError::Sql(format!("invalid number literal '{n}'"))),
    }
}

/// Fold `-<number-literal>` into a single signed literal; otherwise lower as `0 - expr`.
/// The asymmetric `i64` range (`MIN = -2^63`, `MAX = 2^63 - 1`) is handled by
/// trying `i64::from_str` on the *negated* string, which succeeds at `i64::MIN`
/// even though `2^63` does not fit in a positive `i64`.
fn negate_expr(e: &SqlExpr) -> PatinaResult<Expr> {
    if let SqlExpr::Value(Value::Number(n, _)) = e {
        // Common case: positive literal fits in i64; just negate.
        if let Ok(i) = n.parse::<i64>() {
            return Ok(Expr::Literal(Literal::Int64(-i)));
        }
        // Edge case: -i64::MIN. The positive form "9223372036854775808" overflows
        // i64, but the negated literal "-9223372036854775808" parses cleanly.
        let negated = format!("-{n}");
        if let Ok(i) = negated.parse::<i64>() {
            return Ok(Expr::Literal(Literal::Int64(i)));
        }
        // Integer-looking but still out of range (e.g. -10^20): reject, do not
        // silently demote to Float64.
        if looks_like_pure_integer(n) {
            return Err(PatinaError::Sql(format!(
                "integer literal -{n} out of i64 range; use scientific notation or an explicit fractional part for Float64"
            )));
        }
        if let Ok(f) = n.parse::<f64>() {
            return Ok(Expr::Literal(Literal::Float64(-f)));
        }
        return Err(PatinaError::Sql(format!("invalid number literal '{n}'")));
    }
    let inner = lower_expr(e)?;
    Ok(Expr::Binary {
        op: BinaryOp::Sub,
        left: Box::new(Expr::Literal(Literal::Int64(0))),
        right: Box::new(inner),
    })
}

/// Mirror of the (private) `AggregateExpr::output_name` rule in
/// `logical_plan.rs`. Kept in sync by inspection — if that rule changes, this
/// must change with it. Used to re-project aggregate results in SELECT order.
fn aggregate_output_name(agg: &AggregateExpr) -> String {
    fn suffix(e: &Expr) -> String {
        match e {
            Expr::Column(n) => format!("_{n}"),
            Expr::Alias(_, n) => format!("_{n}"),
            _ => String::new(),
        }
    }
    match agg {
        AggregateExpr::Count(e) => format!("count{}", suffix(e)),
        AggregateExpr::Sum(e) => format!("sum{}", suffix(e)),
        AggregateExpr::Min(e) => format!("min{}", suffix(e)),
        AggregateExpr::Max(e) => format!("max{}", suffix(e)),
        AggregateExpr::Avg(e) => format!("avg{}", suffix(e)),
    }
}

/// Mirror of the group-key naming rule inside `LogicalPlan::schema()` for the
/// `Aggregate` arm in `logical_plan.rs`. Kept in sync by inspection.
fn group_key_output_name(key: &Expr, idx: usize) -> String {
    match key {
        Expr::Column(n) => n.clone(),
        Expr::Alias(_, n) => n.clone(),
        _ => format!("__group_{idx}"),
    }
}

/// Map a `sqlparser` `BinaryOperator` onto our small `BinaryOp` set; reject anything else.
fn lower_binary_op(op: &BinaryOperator) -> PatinaResult<BinaryOp> {
    Ok(match op {
        BinaryOperator::Plus => BinaryOp::Add,
        BinaryOperator::Minus => BinaryOp::Sub,
        BinaryOperator::Multiply => BinaryOp::Mul,
        BinaryOperator::Divide => BinaryOp::Div,
        BinaryOperator::Eq => BinaryOp::Eq,
        BinaryOperator::NotEq => BinaryOp::NotEq,
        BinaryOperator::Lt => BinaryOp::Lt,
        BinaryOperator::LtEq => BinaryOp::LtEq,
        BinaryOperator::Gt => BinaryOp::Gt,
        BinaryOperator::GtEq => BinaryOp::GtEq,
        BinaryOperator::And => BinaryOp::And,
        BinaryOperator::Or => BinaryOp::Or,
        other => {
            return Err(PatinaError::Sql(format!(
                "unsupported binary operator: {other}"
            )));
        }
    })
}

/// Structural equality of two lowered `Expr` trees (ignoring aliases at the root).
fn expr_eq(a: &Expr, b: &Expr) -> bool {
    let a = strip_alias(a);
    let b = strip_alias(b);
    match (a, b) {
        (Expr::Column(x), Expr::Column(y)) => x == y,
        (Expr::Literal(x), Expr::Literal(y)) => x == y,
        (
            Expr::Binary {
                op: ao,
                left: al,
                right: ar,
            },
            Expr::Binary {
                op: bo,
                left: bl,
                right: br,
            },
        ) => ao == bo && expr_eq(al, bl) && expr_eq(ar, br),
        _ => false,
    }
}

/// Peel one or more `Alias` wrappers off the root.
fn strip_alias(e: &Expr) -> &Expr {
    let mut cur = e;
    while let Expr::Alias(inner, _) = cur {
        cur = inner;
    }
    cur
}

#[cfg(test)]
mod wave7_tests {
    //! Parse-and-lower smoke tests for wave 7 features (DISTINCT, LIMIT,
    //! ORDER BY, HAVING, UNION [ALL], INNER JOIN). These tests check only
    //! the logical / physical plan *shape* — actual execution is covered by
    //! the e2e suite and is out of scope here.
    use super::*;
    use crate::plan::logical_plan::DataType;
    use crate::plan::physical_plan::{lower, PhysicalPlan};

    /// Minimal two-table fixture with stable column dtypes for plan tests.
    fn provider() -> MemTableProvider {
        use crate::plan::logical_plan::Field;
        let t1 = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int64, false),
        ]);
        let t2 = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("c", DataType::Float64, false),
        ]);
        MemTableProvider::new()
            .with_table("t1", t1)
            .with_table("t2", t2)
    }

    fn lp(sql: &str) -> LogicalPlan {
        parse(sql, &provider()).unwrap_or_else(|e| panic!("parse failed for {sql:?}: {e}"))
    }

    fn pp(sql: &str) -> PhysicalPlan {
        let logical = lp(sql);
        lower(&logical).unwrap_or_else(|e| panic!("lower failed for {sql:?}: {e}"))
    }

    #[test]
    fn select_distinct_wraps_in_distinct() {
        let plan = lp("SELECT DISTINCT a FROM t1");
        assert!(
            matches!(plan, LogicalPlan::Distinct { .. }),
            "expected Distinct at top, got {plan:?}"
        );
        let phys = lower(&plan).unwrap();
        assert!(matches!(phys, PhysicalPlan::Distinct { .. }));
    }

    #[test]
    fn limit_offset_parses() {
        let phys = pp("SELECT a FROM t1 LIMIT 10 OFFSET 5");
        match phys {
            PhysicalPlan::Limit {
                limit,
                offset,
                ref input,
                ..
            } => {
                assert_eq!(limit, 10);
                assert_eq!(offset, 5);
                assert!(matches!(**input, PhysicalPlan::Projection { .. }));
            }
            other => panic!("expected Limit, got {other:?}"),
        }
    }

    #[test]
    fn offset_without_limit_uses_usize_max() {
        let phys = pp("SELECT a FROM t1 OFFSET 3");
        match phys {
            PhysicalPlan::Limit { limit, offset, .. } => {
                assert_eq!(limit, usize::MAX);
                assert_eq!(offset, 3);
            }
            other => panic!("expected Limit, got {other:?}"),
        }
    }

    #[test]
    fn order_by_default_direction_and_nulls() {
        let plan = lp("SELECT a FROM t1 ORDER BY a");
        match plan {
            LogicalPlan::Sort { sort_exprs, .. } => {
                assert_eq!(sort_exprs.len(), 1);
                assert!(!sort_exprs[0].descending);
                assert!(sort_exprs[0].nulls_first, "ASC defaults to NULLS FIRST");
            }
            other => panic!("expected Sort, got {other:?}"),
        }
    }

    #[test]
    fn order_by_desc_defaults_to_nulls_last() {
        let plan = lp("SELECT a FROM t1 ORDER BY a DESC");
        match plan {
            LogicalPlan::Sort { sort_exprs, .. } => {
                assert!(sort_exprs[0].descending);
                assert!(!sort_exprs[0].nulls_first, "DESC defaults to NULLS LAST");
            }
            other => panic!("expected Sort, got {other:?}"),
        }
    }

    #[test]
    fn order_by_with_explicit_nulls_first() {
        let plan = lp("SELECT a FROM t1 ORDER BY a DESC NULLS FIRST");
        match plan {
            LogicalPlan::Sort { sort_exprs, .. } => {
                assert!(sort_exprs[0].descending);
                assert!(sort_exprs[0].nulls_first);
            }
            other => panic!("expected Sort, got {other:?}"),
        }
    }

    #[test]
    fn order_by_then_limit_layering() {
        // ORDER BY must sit *below* LIMIT in the tree (SQL semantics: sort
        // first, then truncate). The lowered physical plan mirrors this.
        let phys = pp("SELECT a FROM t1 ORDER BY a DESC LIMIT 5");
        match phys {
            PhysicalPlan::Limit { input, .. } => {
                assert!(matches!(*input, PhysicalPlan::Sort { .. }));
            }
            other => panic!("expected Limit, got {other:?}"),
        }
    }

    #[test]
    fn having_wraps_aggregate_in_filter() {
        let plan = lp("SELECT a, COUNT(b) FROM t1 GROUP BY a HAVING COUNT(b) > 1");
        // After the wave-1 SELECT-order Project on the aggregate, HAVING
        // appears as the outermost Filter.
        match plan {
            LogicalPlan::Filter { input, .. } => {
                // Below the Filter is the Project that fixes SELECT column order.
                assert!(
                    matches!(*input, LogicalPlan::Project { .. }),
                    "expected Project under HAVING Filter, got {input:?}"
                );
            }
            other => panic!("expected Filter (HAVING) at top, got {other:?}"),
        }
    }

    #[test]
    fn having_rejected_without_group_by_or_aggregate() {
        let err = parse("SELECT a FROM t1 HAVING a > 1", &provider())
            .expect_err("HAVING without aggregate must error");
        let msg = format!("{err}");
        assert!(
            msg.to_lowercase().contains("having"),
            "error message should mention HAVING, got: {msg}"
        );
    }

    #[test]
    fn union_all_builds_union() {
        let plan = lp("SELECT a FROM t1 UNION ALL SELECT a FROM t2");
        match plan {
            LogicalPlan::Union { inputs } => {
                assert_eq!(inputs.len(), 2);
            }
            other => panic!("expected Union, got {other:?}"),
        }
    }

    #[test]
    fn union_dedup_wraps_union_in_distinct() {
        let plan = lp("SELECT a FROM t1 UNION SELECT a FROM t2");
        match plan {
            LogicalPlan::Distinct { input } => {
                assert!(
                    matches!(*input, LogicalPlan::Union { .. }),
                    "expected Distinct(Union), got Distinct({input:?})"
                );
            }
            other => panic!("expected Distinct, got {other:?}"),
        }
    }

    #[test]
    fn union_all_is_flattened() {
        // Three-way UNION ALL should land as a single 3-input Union, not a
        // nested 2-tree, so executors can stream branches without recursion.
        let plan = lp("SELECT a FROM t1 UNION ALL SELECT a FROM t1 UNION ALL SELECT a FROM t2");
        match plan {
            LogicalPlan::Union { inputs } => {
                assert_eq!(inputs.len(), 3, "expected flattened 3-input Union");
            }
            other => panic!("expected Union, got {other:?}"),
        }
    }

    #[test]
    fn inner_join_parses_to_join_node() {
        let plan = lp("SELECT * FROM t1 INNER JOIN t2 ON t1.a = t2.a");
        match plan {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::Join {
                    join_type,
                    on,
                    ..
                } => {
                    assert_eq!(join_type, JoinType::Inner);
                    assert_eq!(on.len(), 1);
                }
                other => panic!("expected Join under Project, got {other:?}"),
            },
            other => panic!("expected Project, got {other:?}"),
        }
    }

    #[test]
    fn inner_join_conjunctive_on_collects_multiple_pairs() {
        let plan = lp("SELECT * FROM t1 INNER JOIN t2 ON t1.a = t2.a AND t1.b = t2.c");
        let join = match plan {
            LogicalPlan::Project { input, .. } => *input,
            other => panic!("expected Project, got {other:?}"),
        };
        match join {
            LogicalPlan::Join { on, .. } => assert_eq!(on.len(), 2),
            other => panic!("expected Join, got {other:?}"),
        }
    }

    #[test]
    fn non_equi_join_rejected_with_clear_message() {
        let err = parse(
            "SELECT * FROM t1 INNER JOIN t2 ON t1.a > t2.a",
            &provider(),
        )
        .expect_err("non-equi JOIN must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("non-equi JOIN not yet supported"),
            "error message should mention non-equi JOIN, got: {msg}"
        );
    }

    #[test]
    fn join_lowers_to_physical_join() {
        // `SELECT * FROM t1 INNER JOIN t2 ON ...` parses to
        // `Project { input: Join }`; our lowerer detects that the source
        // chain isn't a Scan-only chain and falls through to lowering the
        // inner Join directly. The wave-7 executor surfaces the actual
        // "JOIN not yet implemented" error at run time; the planner just
        // needs to produce a PhysicalPlan::Join here.
        let phys = pp("SELECT * FROM t1 INNER JOIN t2 ON t1.a = t2.a");
        assert!(
            matches!(phys, PhysicalPlan::Join { .. }),
            "expected PhysicalPlan::Join, got {phys:?}"
        );
    }
}
