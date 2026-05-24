// SPDX-License-Identifier: Apache-2.0

//! SQL frontend: parses a SQL string into a `LogicalPlan` against a `TableProvider`.

use std::collections::HashMap;

use sqlparser::ast::{
    BinaryOperator, Distinct, Expr as SqlExpr, FunctionArg, FunctionArgExpr, FunctionArguments,
    GroupByExpr, Ident, ObjectName, Query, Select, SelectItem, SetExpr, Statement, TableFactor,
    UnaryOperator, Value,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::error::{JavelinError, JavelinResult};
use crate::plan::logical_plan::{AggregateExpr, BinaryOp, Expr, Literal, LogicalPlan, Schema};

/// Resolves table names to their schemas; the SQL frontend cannot know table shapes otherwise.
pub trait TableProvider {
    /// Return the schema for `name`, or a `Plan` error if the table is unknown.
    fn schema(&self, name: &str) -> JavelinResult<Schema>;
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
    fn schema(&self, name: &str) -> JavelinResult<Schema> {
        self.tables
            .get(name)
            .cloned()
            .ok_or_else(|| JavelinError::Plan(format!("unknown table '{name}'")))
    }
}

/// Parse a SQL string into a single `LogicalPlan` using the given provider.
pub fn parse(sql: &str, provider: &dyn TableProvider) -> JavelinResult<LogicalPlan> {
    let dialect = GenericDialect {};
    let mut stmts = Parser::parse_sql(&dialect, sql).map_err(|e| JavelinError::Sql(e.to_string()))?;

    if stmts.len() != 1 {
        return Err(JavelinError::Sql(format!(
            "expected exactly one statement, got {}",
            stmts.len()
        )));
    }
    let stmt = stmts.remove(0);
    let query = match stmt {
        Statement::Query(q) => q,
        other => {
            return Err(JavelinError::Sql(format!(
                "only SELECT queries are supported, got: {other}"
            )));
        }
    };
    plan_query(&query, provider)
}

/// Lower a top-level `Query` (rejecting CTEs, ORDER BY, LIMIT, OFFSET, FETCH, locks, etc.).
fn plan_query(query: &Query, provider: &dyn TableProvider) -> JavelinResult<LogicalPlan> {
    if query.with.is_some() {
        return Err(JavelinError::Sql("unsupported: WITH / CTEs".into()));
    }
    if query.order_by.is_some() {
        return Err(JavelinError::Sql("unsupported: ORDER BY".into()));
    }
    if query.limit.is_some() {
        return Err(JavelinError::Sql("unsupported: LIMIT".into()));
    }
    if !query.limit_by.is_empty() {
        return Err(JavelinError::Sql("unsupported: LIMIT BY".into()));
    }
    if query.offset.is_some() {
        return Err(JavelinError::Sql("unsupported: OFFSET".into()));
    }
    if query.fetch.is_some() {
        return Err(JavelinError::Sql("unsupported: FETCH".into()));
    }
    if !query.locks.is_empty() {
        return Err(JavelinError::Sql("unsupported: FOR UPDATE/SHARE".into()));
    }
    if query.for_clause.is_some() {
        return Err(JavelinError::Sql("unsupported: FOR clause".into()));
    }
    if query.settings.is_some() {
        return Err(JavelinError::Sql("unsupported: SETTINGS clause".into()));
    }
    if query.format_clause.is_some() {
        return Err(JavelinError::Sql("unsupported: FORMAT clause".into()));
    }

    let select = match query.body.as_ref() {
        SetExpr::Select(s) => s.as_ref(),
        SetExpr::Query(_) => {
            return Err(JavelinError::Sql("unsupported: nested query body".into()));
        }
        SetExpr::SetOperation { .. } => {
            return Err(JavelinError::Sql("unsupported: UNION/EXCEPT/INTERSECT".into()));
        }
        SetExpr::Values(_) => {
            return Err(JavelinError::Sql("unsupported: VALUES".into()));
        }
        SetExpr::Insert(_) | SetExpr::Update(_) => {
            return Err(JavelinError::Sql("unsupported: write statement in query body".into()));
        }
        SetExpr::Table(_) => {
            return Err(JavelinError::Sql("unsupported: TABLE statement".into()));
        }
    };

    plan_select(select, provider)
}

/// Lower a `Select` into Scan [→ Filter] → (Project | Aggregate).
fn plan_select(select: &Select, provider: &dyn TableProvider) -> JavelinResult<LogicalPlan> {
    reject_unsupported_select(select)?;

    // FROM: exactly one table, no joins, no alias, no TVF args.
    if select.from.len() != 1 {
        return Err(JavelinError::Sql(format!(
            "expected exactly one FROM table, got {}",
            select.from.len()
        )));
    }
    let twj = &select.from[0];
    if !twj.joins.is_empty() {
        return Err(JavelinError::Sql("unsupported: JOIN".into()));
    }
    let table_name = match &twj.relation {
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
                return Err(JavelinError::Sql("unsupported: table alias".into()));
            }
            if args.is_some() {
                return Err(JavelinError::Sql("unsupported: table-valued function".into()));
            }
            if !with_hints.is_empty() {
                return Err(JavelinError::Sql("unsupported: WITH hints".into()));
            }
            if version.is_some() {
                return Err(JavelinError::Sql("unsupported: table version".into()));
            }
            if *with_ordinality {
                return Err(JavelinError::Sql("unsupported: WITH ORDINALITY".into()));
            }
            if !partitions.is_empty() {
                return Err(JavelinError::Sql("unsupported: PARTITION".into()));
            }
            single_ident_from_object_name(name)?
        }
        _ => {
            return Err(JavelinError::Sql(
                "unsupported: only bare table references are allowed in FROM".into(),
            ));
        }
    };
    let schema = provider.schema(&table_name)?;
    let scan_schema = schema.clone();
    let mut plan = LogicalPlan::Scan {
        table: table_name,
        projection: None,
        schema,
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
            return Err(JavelinError::Sql("unsupported: GROUP BY ALL".into()));
        }
        GroupByExpr::Expressions(exprs, modifiers) => {
            if !modifiers.is_empty() {
                return Err(JavelinError::Sql(
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
                for f in &scan_schema.fields {
                    items.push((SqlExpr::Identifier(Ident::new(f.name.clone())), None));
                }
            }
            SelectItem::QualifiedWildcard(_, _) => {
                return Err(JavelinError::Sql("unsupported: qualified wildcard".into()));
            }
        }
    }

    let has_agg_in_select = items
        .iter()
        .map(|(e, _)| try_aggregate(e))
        .collect::<JavelinResult<Vec<_>>>()?
        .iter()
        .any(|o| o.is_some());

    if has_agg_in_select || !group_by_sql.is_empty() {
        // Aggregate mode. Simplification: every selected item is either a bare
        // aggregate call or a bare group key already listed in GROUP BY. Mixed
        // post-aggregate scalar work (e.g. `SUM(a) + 1`) is rejected up front.
        let group_by: Vec<Expr> = group_by_sql
            .iter()
            .map(|e| lower_expr(e))
            .collect::<JavelinResult<_>>()?;

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
                    return Err(JavelinError::Sql(
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
                return Err(JavelinError::Sql(
                    "post-aggregate expressions not yet supported".into(),
                ));
            }
            let lowered = lower_expr(sql_expr)?;
            // Must match some declared GROUP BY key by structural equality of the lowered form.
            if !group_by.iter().any(|g| expr_eq(g, &lowered)) {
                return Err(JavelinError::Sql(
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

    Ok(plan)
}

/// Reject SELECT-level features outside our supported subset.
fn reject_unsupported_select(select: &Select) -> JavelinResult<()> {
    match &select.distinct {
        None => {}
        Some(Distinct::Distinct) => return Err(JavelinError::Sql("unsupported: SELECT DISTINCT".into())),
        Some(Distinct::On(_)) => return Err(JavelinError::Sql("unsupported: DISTINCT ON".into())),
    }
    if select.top.is_some() {
        return Err(JavelinError::Sql("unsupported: TOP".into()));
    }
    if select.into.is_some() {
        return Err(JavelinError::Sql("unsupported: SELECT INTO".into()));
    }
    if !select.lateral_views.is_empty() {
        return Err(JavelinError::Sql("unsupported: LATERAL VIEW".into()));
    }
    if select.prewhere.is_some() {
        return Err(JavelinError::Sql("unsupported: PREWHERE".into()));
    }
    if !select.cluster_by.is_empty() {
        return Err(JavelinError::Sql("unsupported: CLUSTER BY".into()));
    }
    if !select.distribute_by.is_empty() {
        return Err(JavelinError::Sql("unsupported: DISTRIBUTE BY".into()));
    }
    if !select.sort_by.is_empty() {
        return Err(JavelinError::Sql("unsupported: SORT BY".into()));
    }
    if select.having.is_some() {
        return Err(JavelinError::Sql("unsupported: HAVING".into()));
    }
    if !select.named_window.is_empty() {
        return Err(JavelinError::Sql("unsupported: WINDOW".into()));
    }
    if select.qualify.is_some() {
        return Err(JavelinError::Sql("unsupported: QUALIFY".into()));
    }
    if select.value_table_mode.is_some() {
        return Err(JavelinError::Sql("unsupported: SELECT AS STRUCT/VALUE".into()));
    }
    if select.connect_by.is_some() {
        return Err(JavelinError::Sql("unsupported: CONNECT BY".into()));
    }
    Ok(())
}

/// Pull a single-part identifier out of an `ObjectName`, rejecting schema-qualified names.
fn single_ident_from_object_name(name: &ObjectName) -> JavelinResult<String> {
    if name.0.len() != 1 {
        return Err(JavelinError::Sql(format!(
            "qualified table names not supported: {name}"
        )));
    }
    Ok(name.0[0].value.clone())
}

/// Recognize a top-level aggregate function call. Returns `Ok(None)` for non-aggregates.
fn try_aggregate(e: &SqlExpr) -> JavelinResult<Option<AggregateExpr>> {
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
        return Err(JavelinError::Sql(
            "unsupported: window functions (OVER)".into(),
        ));
    }
    if func.filter.is_some() {
        return Err(JavelinError::Sql("unsupported: FILTER on aggregate".into()));
    }
    if func.null_treatment.is_some() {
        return Err(JavelinError::Sql(
            "unsupported: IGNORE/RESPECT NULLS on aggregate".into(),
        ));
    }
    if !func.within_group.is_empty() {
        return Err(JavelinError::Sql(
            "unsupported: WITHIN GROUP on aggregate".into(),
        ));
    }
    if !matches!(func.parameters, FunctionArguments::None) {
        return Err(JavelinError::Sql(
            "unsupported: parametric aggregate function".into(),
        ));
    }

    let arg_list = match &func.args {
        FunctionArguments::List(list) => list,
        FunctionArguments::None => {
            return Err(JavelinError::Sql(format!("{kind} requires arguments")));
        }
        FunctionArguments::Subquery(_) => {
            return Err(JavelinError::Sql(format!(
                "unsupported: subquery argument to {kind}"
            )));
        }
    };
    if arg_list.duplicate_treatment.is_some() {
        return Err(JavelinError::Sql(format!(
            "unsupported: DISTINCT/ALL inside {kind}"
        )));
    }
    if !arg_list.clauses.is_empty() {
        return Err(JavelinError::Sql(format!(
            "unsupported: argument clauses on {kind}"
        )));
    }
    if arg_list.args.len() != 1 {
        return Err(JavelinError::Sql(format!(
            "{kind} expects exactly one argument, got {}",
            arg_list.args.len()
        )));
    }

    let arg_expr = match &arg_list.args[0] {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => Some(e),
        FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => None,
        FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(_)) => {
            return Err(JavelinError::Sql(format!(
                "unsupported: qualified wildcard in {kind}"
            )));
        }
        FunctionArg::Named { .. } => {
            return Err(JavelinError::Sql(format!(
                "unsupported: named argument to {kind}"
            )));
        }
    };

    let inner = match arg_expr {
        Some(e) => lower_expr(e)?,
        None => {
            if kind != "COUNT" {
                return Err(JavelinError::Sql(format!("{kind}(*) is not supported")));
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
fn contains_aggregate(e: &SqlExpr) -> JavelinResult<bool> {
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

/// Lower a scalar SQL expression into our `Expr`. Aggregates are rejected here —
/// callers must split them off via `try_aggregate` first.
fn lower_expr(e: &SqlExpr) -> JavelinResult<Expr> {
    match e {
        SqlExpr::Identifier(ident) => Ok(Expr::Column(ident.value.clone())),
        SqlExpr::CompoundIdentifier(_) => Err(JavelinError::Sql(
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
            other => Err(JavelinError::Sql(format!(
                "unsupported unary operator: {other:?}"
            ))),
        },
        SqlExpr::Function(_) => Err(JavelinError::Sql(
            "function calls are only allowed as top-level aggregates in SELECT".into(),
        )),
        other => Err(JavelinError::Sql(format!(
            "unsupported expression: {other}"
        ))),
    }
}

/// Translate a SQL literal `Value` into our `Literal` expression.
fn lower_value(v: &Value) -> JavelinResult<Expr> {
    match v {
        Value::Number(n, _long) => parse_number(n),
        Value::SingleQuotedString(s) => Ok(Expr::Literal(Literal::Utf8(s.clone()))),
        Value::Boolean(b) => Ok(Expr::Literal(Literal::Bool(*b))),
        Value::Null => Ok(Expr::Literal(Literal::Null)),
        other => Err(JavelinError::Sql(format!("unsupported literal: {other}"))),
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
fn parse_number(n: &str) -> JavelinResult<Expr> {
    if let Ok(i) = n.parse::<i64>() {
        return Ok(Expr::Literal(Literal::Int64(i)));
    }
    if looks_like_pure_integer(n) {
        return Err(JavelinError::Sql(format!(
            "integer literal {n} out of i64 range; use scientific notation or an explicit fractional part for Float64"
        )));
    }
    match n.parse::<f64>() {
        Ok(f) => Ok(Expr::Literal(Literal::Float64(f))),
        Err(_) => Err(JavelinError::Sql(format!("invalid number literal '{n}'"))),
    }
}

/// Fold `-<number-literal>` into a single signed literal; otherwise lower as `0 - expr`.
/// The asymmetric `i64` range (`MIN = -2^63`, `MAX = 2^63 - 1`) is handled by
/// trying `i64::from_str` on the *negated* string, which succeeds at `i64::MIN`
/// even though `2^63` does not fit in a positive `i64`.
fn negate_expr(e: &SqlExpr) -> JavelinResult<Expr> {
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
            return Err(JavelinError::Sql(format!(
                "integer literal -{n} out of i64 range; use scientific notation or an explicit fractional part for Float64"
            )));
        }
        if let Ok(f) = n.parse::<f64>() {
            return Ok(Expr::Literal(Literal::Float64(-f)));
        }
        return Err(JavelinError::Sql(format!("invalid number literal '{n}'")));
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
fn lower_binary_op(op: &BinaryOperator) -> JavelinResult<BinaryOp> {
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
            return Err(JavelinError::Sql(format!(
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
