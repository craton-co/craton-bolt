// SPDX-License-Identifier: Apache-2.0

//! Uncorrelated-subquery support for the SQL frontend.
//!
//! Craton Bolt lowers only **uncorrelated** subqueries — a subquery that
//! references no columns from the enclosing (outer) query. This module owns
//! the correlation detector that the frontend consults before it lowers a
//! `(SELECT ...)` scalar subquery or an `IN (SELECT ...)` membership test.
//! Correlated subqueries are rejected with a clear [`BoltError::Sql`] rather
//! than being silently mis-planned (the engine has no correlated-execution
//! path).
//!
//! ## What "correlated" means here
//!
//! A subquery is correlated iff some column reference inside it resolves
//! against an **outer** table/alias that is *not* in the subquery's own FROM
//! scope. We detect this purely from the sqlparser AST + the set of column
//! names the subquery's own FROM tree introduces:
//!
//! * A qualified reference `t.c` is correlation iff the qualifier `t` is not
//!   one of the subquery's own table names/aliases.
//! * A bare reference `c` is correlation iff `c` is not a column of any of the
//!   subquery's own tables. (A bare name that matches a subquery column is
//!   resolved locally per standard SQL inside-out name resolution, so it is
//!   never treated as correlation even if an outer table also has a column
//!   `c`.)
//!
//! The detector is intentionally conservative: when it cannot positively
//! prove a reference is local (e.g. it names no qualifier and matches no
//! local column), it flags correlation. That keeps a genuinely correlated
//! query from slipping through as "uncorrelated" and producing wrong results.

use std::collections::HashSet;

use sqlparser::ast::{
    Expr as SqlExpr, FunctionArg, FunctionArgExpr, FunctionArguments, Query, Select, SelectItem,
    SetExpr, TableFactor,
};

use crate::error::{BoltError, BoltResult};
use crate::plan::sql_frontend::MAX_RECURSION_DEPTH;

/// The names a subquery's own FROM tree binds: the table/alias qualifiers it
/// introduces and the (lower-cased) column names available across them.
///
/// Both sets are stored ASCII-lowercased so membership checks match the SQL
/// frontend's identifier-folding convention (unquoted identifiers fold to
/// lowercase). Quoted-identifier corner cases are handled conservatively:
/// the case-folded comparison can only ever make a reference look *more*
/// local, and a false "local" verdict for a quoted mixed-case name would
/// surface later as a normal column-resolution error during lowering — never
/// as a silently-wrong correlated plan.
#[derive(Debug, Default)]
struct LocalScope {
    /// Table names + aliases the subquery's FROM introduces (lower-cased).
    qualifiers: HashSet<String>,
    /// All column names visible in the subquery's own scope (lower-cased).
    columns: HashSet<String>,
}

impl LocalScope {
    fn has_qualifier(&self, q: &str) -> bool {
        self.qualifiers.contains(&q.to_ascii_lowercase())
    }
    fn has_column(&self, c: &str) -> bool {
        self.columns.contains(&c.to_ascii_lowercase())
    }
}

/// Reject `query` if it is a correlated subquery (references any column from
/// an outer scope). `outer_columns` is the set of column names (lower-cased)
/// visible in the enclosing query — used only to produce a precise error
/// message naming the offending outer reference. Returns `Ok(())` when the
/// subquery is uncorrelated.
///
/// `provider` resolves the subquery's own FROM tables so the detector knows
/// which column names are local.
pub(crate) fn reject_if_correlated(
    query: &Query,
    outer_columns: &HashSet<String>,
    provider: &dyn crate::plan::sql_frontend::TableProvider,
) -> BoltResult<()> {
    let mut scope = LocalScope::default();
    collect_query_scope(query, provider, &mut scope, 0)?;
    check_query_correlation(query, &scope, outer_columns, 0)
}

/// Walk `query`'s FROM trees (recursively into subqueries' own bodies is not
/// needed — each nested subquery validates its own scope) collecting the
/// local table qualifiers and column names.
fn collect_query_scope(
    query: &Query,
    provider: &dyn crate::plan::sql_frontend::TableProvider,
    scope: &mut LocalScope,
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "subquery nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    collect_setexpr_scope(query.body.as_ref(), provider, scope, depth + 1)
}

fn collect_setexpr_scope(
    set: &SetExpr,
    provider: &dyn crate::plan::sql_frontend::TableProvider,
    scope: &mut LocalScope,
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "subquery nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    match set {
        SetExpr::Select(s) => collect_select_scope(s, provider, scope, depth + 1),
        SetExpr::Query(q) => collect_query_scope(q, provider, scope, depth + 1),
        SetExpr::SetOperation { left, right, .. } => {
            collect_setexpr_scope(left, provider, scope, depth + 1)?;
            collect_setexpr_scope(right, provider, scope, depth + 1)
        }
        // VALUES / INSERT / UPDATE / TABLE bodies introduce no resolvable
        // column scope the detector cares about; the frontend rejects these
        // body shapes elsewhere, so nothing to collect here.
        _ => Ok(()),
    }
}

fn collect_select_scope(
    select: &Select,
    provider: &dyn crate::plan::sql_frontend::TableProvider,
    scope: &mut LocalScope,
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "subquery nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    for twj in &select.from {
        collect_table_factor_scope(&twj.relation, provider, scope)?;
        for join in &twj.joins {
            collect_table_factor_scope(&join.relation, provider, scope)?;
        }
    }
    Ok(())
}

fn collect_table_factor_scope(
    tf: &TableFactor,
    provider: &dyn crate::plan::sql_frontend::TableProvider,
    scope: &mut LocalScope,
) -> BoltResult<()> {
    match tf {
        TableFactor::Table { name, alias, .. } => {
            // Underlying table name (last path segment) and its alias both
            // become valid local qualifiers.
            if let Some(last) = name.0.last() {
                scope.qualifiers.insert(last.value.to_ascii_lowercase());
            }
            let table_name = name
                .0
                .last()
                .map(|i| i.value.clone())
                .unwrap_or_default();
            if let Some(a) = alias {
                scope.qualifiers.insert(a.name.value.to_ascii_lowercase());
            }
            // Pull the table's columns from the provider so bare references can
            // be classified as local. A lookup miss is tolerated — the
            // subquery's own lowering will surface the unknown-table error with
            // full context; here we just can't add its columns (the detector
            // then conservatively treats unmatched bare names as correlation,
            // which is the safe side).
            if let Ok(s) = provider.schema(&table_name) {
                for f in &s.fields {
                    scope.columns.insert(f.name.to_ascii_lowercase());
                }
            }
        }
        // F12: derived tables (`(SELECT ...) AS t`) are now accepted in FROM,
        // so a subquery whose own FROM contains one must contribute that
        // derived table's scope here — otherwise a bare reference to one of its
        // columns would be conservatively (and wrongly) flagged as a
        // correlation. Register the alias as a qualifier and recurse into the
        // derived subquery's FROM so its base-table columns become local. This
        // is conservative: any column we fail to register (e.g. a computed
        // projection with no matching base column) simply falls back to the
        // "treat unresolved bare name as correlation" safe side, which yields a
        // clean error rather than a silently-wrong plan.
        TableFactor::Derived {
            subquery, alias, ..
        } => {
            if let Some(a) = alias {
                scope.qualifiers.insert(a.name.value.to_ascii_lowercase());
            }
            collect_query_scope(subquery, provider, scope, 0)?;
        }
        // TVFs / nested joins / other factors in FROM are rejected by the main
        // frontend before we get here, so no other arms need scope collection.
        _ => {}
    }
    Ok(())
}

/// Walk every expression position of `query` and flag the first column
/// reference that is not local to `scope`.
fn check_query_correlation(
    query: &Query,
    scope: &LocalScope,
    outer_columns: &HashSet<String>,
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "subquery nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    check_setexpr_correlation(query.body.as_ref(), scope, outer_columns, depth + 1)
}

fn check_setexpr_correlation(
    set: &SetExpr,
    scope: &LocalScope,
    outer_columns: &HashSet<String>,
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "subquery nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    match set {
        SetExpr::Select(s) => check_select_correlation(s, scope, outer_columns, depth + 1),
        SetExpr::Query(q) => check_query_correlation(q, scope, outer_columns, depth + 1),
        SetExpr::SetOperation { left, right, .. } => {
            check_setexpr_correlation(left, scope, outer_columns, depth + 1)?;
            check_setexpr_correlation(right, scope, outer_columns, depth + 1)
        }
        _ => Ok(()),
    }
}

fn check_select_correlation(
    select: &Select,
    scope: &LocalScope,
    outer_columns: &HashSet<String>,
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "subquery nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    // Projection list.
    for item in &select.projection {
        match item {
            SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => {
                check_expr_correlation(e, scope, outer_columns, depth + 1)?;
            }
            SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(_, _) => {}
        }
    }
    // WHERE / HAVING / GROUP BY all see the same FROM scope.
    if let Some(w) = &select.selection {
        check_expr_correlation(w, scope, outer_columns, depth + 1)?;
    }
    if let Some(h) = &select.having {
        check_expr_correlation(h, scope, outer_columns, depth + 1)?;
    }
    if let sqlparser::ast::GroupByExpr::Expressions(exprs, _) = &select.group_by {
        for e in exprs {
            check_expr_correlation(e, scope, outer_columns, depth + 1)?;
        }
    }
    // JOIN ON predicates.
    for twj in &select.from {
        for join in &twj.joins {
            if let Some(on) = join_on_expr(&join.join_operator) {
                check_expr_correlation(on, scope, outer_columns, depth + 1)?;
            }
        }
    }
    Ok(())
}

/// Pull the ON expression out of a `JoinOperator` if it carries one.
fn join_on_expr(op: &sqlparser::ast::JoinOperator) -> Option<&SqlExpr> {
    use sqlparser::ast::{JoinConstraint, JoinOperator};
    let constraint = match op {
        JoinOperator::Inner(c)
        | JoinOperator::LeftOuter(c)
        | JoinOperator::RightOuter(c)
        | JoinOperator::FullOuter(c) => c,
        _ => return None,
    };
    match constraint {
        JoinConstraint::On(e) => Some(e),
        _ => None,
    }
}

/// Recursively flag the first non-local column reference in `e`.
///
/// Nested subqueries inside `e` are *not* descended into here: each nested
/// subquery is validated against its own scope when the frontend lowers it.
/// Descending would incorrectly attribute an inner subquery's local columns
/// to this scope.
fn check_expr_correlation(
    e: &SqlExpr,
    scope: &LocalScope,
    outer_columns: &HashSet<String>,
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "subquery nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    match e {
        SqlExpr::Identifier(ident) => {
            let name = &ident.value;
            if !scope.has_column(name) {
                // Bare name not resolvable locally. If it is an outer column,
                // this is a correlation; otherwise it is a plain unknown
                // column that the subquery's own lowering will report. We only
                // hard-reject the provably-correlated case so genuine typos
                // keep their existing error path.
                if outer_columns.contains(&name.to_ascii_lowercase()) {
                    return Err(correlated_err(&format!("'{name}'")));
                }
            }
            Ok(())
        }
        SqlExpr::CompoundIdentifier(parts) => {
            if parts.len() >= 2 {
                let qualifier = &parts[0].value;
                if !scope.has_qualifier(qualifier) {
                    return Err(correlated_err(&format!(
                        "'{}.{}'",
                        qualifier,
                        parts[1].value
                    )));
                }
            }
            Ok(())
        }
        // Subqueries embedded in this expression are validated independently;
        // do not descend.
        SqlExpr::Subquery(_) | SqlExpr::InSubquery { .. } | SqlExpr::Exists { .. } => Ok(()),
        SqlExpr::Nested(inner) => check_expr_correlation(inner, scope, outer_columns, depth + 1),
        SqlExpr::BinaryOp { left, right, .. } => {
            check_expr_correlation(left, scope, outer_columns, depth + 1)?;
            check_expr_correlation(right, scope, outer_columns, depth + 1)
        }
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr)
        | SqlExpr::Cast { expr, .. } => {
            check_expr_correlation(expr, scope, outer_columns, depth + 1)
        }
        SqlExpr::Between {
            expr, low, high, ..
        } => {
            check_expr_correlation(expr, scope, outer_columns, depth + 1)?;
            check_expr_correlation(low, scope, outer_columns, depth + 1)?;
            check_expr_correlation(high, scope, outer_columns, depth + 1)
        }
        SqlExpr::InList { expr, list, .. } => {
            check_expr_correlation(expr, scope, outer_columns, depth + 1)?;
            for v in list {
                check_expr_correlation(v, scope, outer_columns, depth + 1)?;
            }
            Ok(())
        }
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
            check_expr_correlation(expr, scope, outer_columns, depth + 1)?;
            check_expr_correlation(pattern, scope, outer_columns, depth + 1)
        }
        SqlExpr::Case {
            operand,
            conditions,
            results,
            else_result,
        } => {
            if let Some(op) = operand {
                check_expr_correlation(op, scope, outer_columns, depth + 1)?;
            }
            for c in conditions {
                check_expr_correlation(c, scope, outer_columns, depth + 1)?;
            }
            for r in results {
                check_expr_correlation(r, scope, outer_columns, depth + 1)?;
            }
            if let Some(er) = else_result {
                check_expr_correlation(er, scope, outer_columns, depth + 1)?;
            }
            Ok(())
        }
        SqlExpr::Function(func) => {
            if let FunctionArguments::List(list) = &func.args {
                for arg in &list.args {
                    if let FunctionArg::Unnamed(FunctionArgExpr::Expr(ae))
                    | FunctionArg::Named {
                        arg: FunctionArgExpr::Expr(ae),
                        ..
                    } = arg
                    {
                        check_expr_correlation(ae, scope, outer_columns, depth + 1)?;
                    }
                }
            }
            Ok(())
        }
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            check_expr_correlation(expr, scope, outer_columns, depth + 1)?;
            if let Some(f) = substring_from {
                check_expr_correlation(f, scope, outer_columns, depth + 1)?;
            }
            if let Some(f) = substring_for {
                check_expr_correlation(f, scope, outer_columns, depth + 1)?;
            }
            Ok(())
        }
        // Literals, typed strings, wildcards, etc. carry no column reference.
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Correlation *collection* for LATERAL apply (feature F3 — LATERAL).
// ---------------------------------------------------------------------------
//
// The detector above *rejects* the first correlated reference it finds. The
// LATERAL apply path instead needs the *set* of outer references a subquery
// makes, so the host nested-loop apply (see
// [`crate::exec::engine::Engine::execute_lateral_apply`]) can substitute each
// one with the current outer row's value. The walk below reuses the same
// scope-classification rule as the detector (a reference is "outer" iff it is
// not resolvable inside the subquery's own FROM scope) but accumulates rather
// than rejecting.

/// One outer (correlated) reference collected from a LATERAL subquery: the
/// optional qualifier (lower-cased, e.g. the `t` of `t.c`) and the column name
/// as written (lower-cased for unquoted idents, verbatim for quoted ones,
/// matching [`super::sql_frontend`]'s folding via the AST `Ident`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct CorrRef {
    /// Qualifier of a `qual.col` reference, lower-cased; `None` for a bare
    /// `col` reference.
    pub qualifier: Option<String>,
    /// Column name (as it should be matched against an outer column).
    pub column: String,
}

/// Collect the **distinct** outer (correlated) references made by `query`
/// against the enclosing scope.
///
/// `outer_columns` is the set of (lower-cased) column names visible in the
/// enclosing (LEFT) scope. A bare reference is treated as a correlation only
/// when it is both *not* local to the subquery AND present in `outer_columns`
/// (so a genuine typo keeps its ordinary unknown-column error path rather than
/// being silently absorbed as a correlation). A qualified `q.col` reference is
/// a correlation iff `q` is not one of the subquery's own table/alias
/// qualifiers.
///
/// `provider` resolves the subquery's own FROM tables so the collector knows
/// which names are local (identical to [`reject_if_correlated`]).
pub(crate) fn collect_correlations(
    query: &Query,
    outer_columns: &HashSet<String>,
    provider: &dyn crate::plan::sql_frontend::TableProvider,
) -> BoltResult<Vec<CorrRef>> {
    let mut scope = LocalScope::default();
    collect_query_scope(query, provider, &mut scope, 0)?;
    let mut out: Vec<CorrRef> = Vec::new();
    collect_query_correlations(query, &scope, outer_columns, &mut out, 0)?;
    Ok(out)
}

fn push_unique(out: &mut Vec<CorrRef>, r: CorrRef) {
    if !out.contains(&r) {
        out.push(r);
    }
}

fn collect_query_correlations(
    query: &Query,
    scope: &LocalScope,
    outer_columns: &HashSet<String>,
    out: &mut Vec<CorrRef>,
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "subquery nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    collect_setexpr_correlations(query.body.as_ref(), scope, outer_columns, out, depth + 1)
}

fn collect_setexpr_correlations(
    set: &SetExpr,
    scope: &LocalScope,
    outer_columns: &HashSet<String>,
    out: &mut Vec<CorrRef>,
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "subquery nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    match set {
        SetExpr::Select(s) => collect_select_correlations(s, scope, outer_columns, out, depth + 1),
        SetExpr::Query(q) => collect_query_correlations(q, scope, outer_columns, out, depth + 1),
        SetExpr::SetOperation { left, right, .. } => {
            collect_setexpr_correlations(left, scope, outer_columns, out, depth + 1)?;
            collect_setexpr_correlations(right, scope, outer_columns, out, depth + 1)
        }
        _ => Ok(()),
    }
}

fn collect_select_correlations(
    select: &Select,
    scope: &LocalScope,
    outer_columns: &HashSet<String>,
    out: &mut Vec<CorrRef>,
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "subquery nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    for item in &select.projection {
        match item {
            SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => {
                collect_expr_correlations(e, scope, outer_columns, out, depth + 1)?;
            }
            SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(_, _) => {}
        }
    }
    if let Some(w) = &select.selection {
        collect_expr_correlations(w, scope, outer_columns, out, depth + 1)?;
    }
    if let Some(h) = &select.having {
        collect_expr_correlations(h, scope, outer_columns, out, depth + 1)?;
    }
    if let sqlparser::ast::GroupByExpr::Expressions(exprs, _) = &select.group_by {
        for e in exprs {
            collect_expr_correlations(e, scope, outer_columns, out, depth + 1)?;
        }
    }
    for twj in &select.from {
        for join in &twj.joins {
            if let Some(on) = join_on_expr(&join.join_operator) {
                collect_expr_correlations(on, scope, outer_columns, out, depth + 1)?;
            }
        }
    }
    Ok(())
}

/// Recursively collect every outer column reference in `e`. Mirrors
/// [`check_expr_correlation`] exactly (same arms, same "do not descend into a
/// nested subquery" rule) but accumulates into `out` instead of erroring.
fn collect_expr_correlations(
    e: &SqlExpr,
    scope: &LocalScope,
    outer_columns: &HashSet<String>,
    out: &mut Vec<CorrRef>,
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "subquery nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    match e {
        SqlExpr::Identifier(ident) => {
            let name = &ident.value;
            if !scope.has_column(name) && outer_columns.contains(&name.to_ascii_lowercase()) {
                push_unique(
                    out,
                    CorrRef {
                        qualifier: None,
                        column: name.to_ascii_lowercase(),
                    },
                );
            }
            Ok(())
        }
        SqlExpr::CompoundIdentifier(parts) => {
            if parts.len() >= 2 {
                let qualifier = &parts[0].value;
                if !scope.has_qualifier(qualifier) {
                    push_unique(
                        out,
                        CorrRef {
                            qualifier: Some(qualifier.to_ascii_lowercase()),
                            column: parts[1].value.to_ascii_lowercase(),
                        },
                    );
                }
            }
            Ok(())
        }
        SqlExpr::Subquery(_) | SqlExpr::InSubquery { .. } | SqlExpr::Exists { .. } => Ok(()),
        SqlExpr::Nested(inner) => {
            collect_expr_correlations(inner, scope, outer_columns, out, depth + 1)
        }
        SqlExpr::BinaryOp { left, right, .. } => {
            collect_expr_correlations(left, scope, outer_columns, out, depth + 1)?;
            collect_expr_correlations(right, scope, outer_columns, out, depth + 1)
        }
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr)
        | SqlExpr::Cast { expr, .. } => {
            collect_expr_correlations(expr, scope, outer_columns, out, depth + 1)
        }
        SqlExpr::Between {
            expr, low, high, ..
        } => {
            collect_expr_correlations(expr, scope, outer_columns, out, depth + 1)?;
            collect_expr_correlations(low, scope, outer_columns, out, depth + 1)?;
            collect_expr_correlations(high, scope, outer_columns, out, depth + 1)
        }
        SqlExpr::InList { expr, list, .. } => {
            collect_expr_correlations(expr, scope, outer_columns, out, depth + 1)?;
            for v in list {
                collect_expr_correlations(v, scope, outer_columns, out, depth + 1)?;
            }
            Ok(())
        }
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
            collect_expr_correlations(expr, scope, outer_columns, out, depth + 1)?;
            collect_expr_correlations(pattern, scope, outer_columns, out, depth + 1)
        }
        SqlExpr::Case {
            operand,
            conditions,
            results,
            else_result,
        } => {
            if let Some(op) = operand {
                collect_expr_correlations(op, scope, outer_columns, out, depth + 1)?;
            }
            for c in conditions {
                collect_expr_correlations(c, scope, outer_columns, out, depth + 1)?;
            }
            for r in results {
                collect_expr_correlations(r, scope, outer_columns, out, depth + 1)?;
            }
            if let Some(er) = else_result {
                collect_expr_correlations(er, scope, outer_columns, out, depth + 1)?;
            }
            Ok(())
        }
        SqlExpr::Function(func) => {
            if let FunctionArguments::List(list) = &func.args {
                for arg in &list.args {
                    if let FunctionArg::Unnamed(FunctionArgExpr::Expr(ae))
                    | FunctionArg::Named {
                        arg: FunctionArgExpr::Expr(ae),
                        ..
                    } = arg
                    {
                        collect_expr_correlations(ae, scope, outer_columns, out, depth + 1)?;
                    }
                }
            }
            Ok(())
        }
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            collect_expr_correlations(expr, scope, outer_columns, out, depth + 1)?;
            if let Some(f) = substring_from {
                collect_expr_correlations(f, scope, outer_columns, out, depth + 1)?;
            }
            if let Some(f) = substring_for {
                collect_expr_correlations(f, scope, outer_columns, out, depth + 1)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Build the canonical "correlated subquery rejected" error naming the
/// offending reference.
fn correlated_err(reference: &str) -> BoltError {
    BoltError::Sql(format!(
        "unsupported: correlated subquery (references outer column {reference}); \
         only uncorrelated subqueries are supported"
    ))
}

// ---------------------------------------------------------------------------
// Correlated WHERE subquery detection (feature F4 — correlated EXISTS /
// NOT EXISTS / scalar subquery in WHERE).
// ---------------------------------------------------------------------------
//
// The detector below recognises a top-level `SELECT` whose WHERE contains a
// *correlated* subquery and classifies it as an EXISTS semi-join, a NOT EXISTS
// anti-join, or a correlated scalar-compare. The engine then executes it as a
// per-outer-row Apply (see `Engine::execute_correlated_where`), reusing the
// LATERAL substitution machinery (rewrite each outer reference to a
// `(SELECT __corr_<i> FROM __lateral_outer)` scalar subquery the engine folds
// per row). Only a SINGLE correlated subquery in the WHERE is in scope; the
// rest of the WHERE (ordinary, uncorrelated conjuncts) is applied as a normal
// `Filter` on the outer plan.

/// Returns `true` iff `query` (as a subquery body) references at least one
/// column from the enclosing scope described by `outer_columns` — i.e. it is
/// correlated. A thin wrapper over [`collect_correlations`] used by the
/// correlated-WHERE detector to tell a correlated subquery (handled via the
/// per-row Apply) apart from an uncorrelated one (folded to a constant).
pub(crate) fn subquery_is_correlated(
    query: &Query,
    outer_columns: &HashSet<String>,
    provider: &dyn crate::plan::sql_frontend::TableProvider,
) -> BoltResult<bool> {
    Ok(!collect_correlations(query, outer_columns, provider)?.is_empty())
}

/// Split a WHERE expression into its top-level `AND` conjuncts (left-to-right
/// order). A non-`AND` expression yields a single-element vector. `Nested`
/// (parenthesised) wrappers around an `AND` are flattened so
/// `a AND (b AND c)` splits into `[a, b, c]`.
pub(crate) fn split_and_conjuncts(e: &SqlExpr) -> Vec<&SqlExpr> {
    fn walk<'a>(e: &'a SqlExpr, out: &mut Vec<&'a SqlExpr>) {
        match e {
            SqlExpr::BinaryOp {
                left,
                op: sqlparser::ast::BinaryOperator::And,
                right,
            } => {
                walk(left, out);
                walk(right, out);
            }
            SqlExpr::Nested(inner) => walk(inner, out),
            other => out.push(other),
        }
    }
    let mut out = Vec::new();
    walk(e, &mut out);
    out
}

/// The classification of a correlated WHERE conjunct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrWhereKind {
    /// `EXISTS (correlated subquery)` — semi-join (keep outer row iff the
    /// subquery returns >= 1 row).
    Exists,
    /// `NOT EXISTS (correlated subquery)` — anti-join (keep outer row iff the
    /// subquery returns 0 rows).
    NotExists,
    /// A scalar comparison conjunct containing exactly one correlated scalar
    /// subquery (e.g. `outer.a > (SELECT max(b) FROM ... WHERE k = outer.k)`).
    /// The whole conjunct is evaluated per outer row as a boolean test.
    Scalar,
}

/// Count how many subquery nodes (`(SELECT ...)`, `EXISTS (...)`,
/// `x IN (SELECT ...)`) appear *directly* in `e` (not descending into a
/// subquery's own body). Used to keep the scalar-correlated path to the
/// single-subquery case it can soundly evaluate.
pub(crate) fn count_direct_subqueries(e: &SqlExpr) -> usize {
    fn walk(e: &SqlExpr, n: &mut usize) {
        match e {
            SqlExpr::Subquery(_) | SqlExpr::Exists { .. } | SqlExpr::InSubquery { .. } => {
                *n += 1;
                // Do not descend into the subquery body — a nested subquery
                // there is validated independently.
            }
            SqlExpr::Nested(inner)
            | SqlExpr::UnaryOp { expr: inner, .. }
            | SqlExpr::IsNull(inner)
            | SqlExpr::IsNotNull(inner)
            | SqlExpr::Cast { expr: inner, .. } => walk(inner, n),
            SqlExpr::BinaryOp { left, right, .. } => {
                walk(left, n);
                walk(right, n);
            }
            SqlExpr::Between {
                expr, low, high, ..
            } => {
                walk(expr, n);
                walk(low, n);
                walk(high, n);
            }
            SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
                walk(expr, n);
                walk(pattern, n);
            }
            _ => {}
        }
    }
    let mut n = 0;
    walk(e, &mut n);
    n
}

/// Collect the distinct outer (correlated) references made by a *scalar WHERE
/// conjunct* `e` against the enclosing scope `outer_columns`.
///
/// Unlike [`collect_correlations`] (which works on a subquery `Query` and stops
/// at the conjunct's own FROM scope), this walks an expression that has NO own
/// FROM scope — so every outer-column reference at the conjunct level is a
/// correlation — AND descends into each direct scalar `(SELECT ...)` subquery,
/// collecting *its* correlations against the same outer scope (using the
/// subquery's own FROM as that subquery's local scope, via
/// [`collect_correlations`]). The union, de-duplicated, is returned in
/// first-seen order. `EXISTS` / `IN (SELECT ...)` bodies are not descended into
/// here (the correlated-WHERE detector handles EXISTS as its own conjunct kind
/// and does not mix it with a scalar conjunct).
pub(crate) fn collect_conjunct_correlations(
    e: &SqlExpr,
    outer_columns: &HashSet<String>,
    provider: &dyn crate::plan::sql_frontend::TableProvider,
) -> BoltResult<Vec<CorrRef>> {
    // The conjunct itself has no FROM scope — an empty `LocalScope` means every
    // outer-column reference at this level is a correlation.
    let empty = LocalScope::default();
    let mut out: Vec<CorrRef> = Vec::new();
    collect_expr_correlations(e, &empty, outer_columns, &mut out, 0)?;
    // Descend into each direct scalar subquery and union its correlations.
    collect_scalar_subquery_correlations(e, outer_columns, provider, &mut out, 0)?;
    Ok(out)
}

/// Walk `e`, and for each direct scalar `(SELECT ...)` subquery accumulate its
/// correlations (via [`collect_correlations`]) into `out`. Does not descend
/// into `EXISTS` / `IN` subquery bodies.
fn collect_scalar_subquery_correlations(
    e: &SqlExpr,
    outer_columns: &HashSet<String>,
    provider: &dyn crate::plan::sql_frontend::TableProvider,
    out: &mut Vec<CorrRef>,
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    match e {
        SqlExpr::Subquery(q) => {
            for r in collect_correlations(q, outer_columns, provider)? {
                push_unique(out, r);
            }
            Ok(())
        }
        SqlExpr::Exists { .. } | SqlExpr::InSubquery { .. } => Ok(()),
        SqlExpr::Nested(inner)
        | SqlExpr::UnaryOp { expr: inner, .. }
        | SqlExpr::IsNull(inner)
        | SqlExpr::IsNotNull(inner)
        | SqlExpr::Cast { expr: inner, .. } => {
            collect_scalar_subquery_correlations(inner, outer_columns, provider, out, depth + 1)
        }
        SqlExpr::BinaryOp { left, right, .. } => {
            collect_scalar_subquery_correlations(left, outer_columns, provider, out, depth + 1)?;
            collect_scalar_subquery_correlations(right, outer_columns, provider, out, depth + 1)
        }
        SqlExpr::Between {
            expr, low, high, ..
        } => {
            collect_scalar_subquery_correlations(expr, outer_columns, provider, out, depth + 1)?;
            collect_scalar_subquery_correlations(low, outer_columns, provider, out, depth + 1)?;
            collect_scalar_subquery_correlations(high, outer_columns, provider, out, depth + 1)
        }
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
            collect_scalar_subquery_correlations(expr, outer_columns, provider, out, depth + 1)?;
            collect_scalar_subquery_correlations(pattern, outer_columns, provider, out, depth + 1)
        }
        _ => Ok(()),
    }
}

/// Extract the `Query` body of an `EXISTS (subquery)` / `NOT EXISTS (subquery)`
/// conjunct, returning `(query, negated)`. `None` for any other expression
/// shape.
pub(crate) fn as_exists(e: &SqlExpr) -> Option<(&Query, bool)> {
    match e {
        SqlExpr::Exists { subquery, negated } => Some((subquery.as_ref(), *negated)),
        SqlExpr::Nested(inner) => as_exists(inner),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{DataType, Field, Schema};
    use crate::plan::sql_frontend::MemTableProvider;
    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;

    /// Standalone provider (no Engine / no GPU): `o(a Int64, k Int64)` outer,
    /// `s(sk Int64, v Int64)` for the subqueries.
    fn provider() -> MemTableProvider {
        MemTableProvider::new()
            .with_table(
                "o",
                Schema::new(vec![
                    Field::new("a", DataType::Int64, false),
                    Field::new("k", DataType::Int64, false),
                ]),
            )
            .with_table(
                "s",
                Schema::new(vec![
                    Field::new("sk", DataType::Int64, false),
                    Field::new("v", DataType::Int64, false),
                ]),
            )
    }

    fn outer_cols() -> HashSet<String> {
        ["a", "k"].iter().map(|s| s.to_string()).collect()
    }

    /// Parse the sole WHERE expression out of a single-SELECT statement.
    fn where_expr(sql: &str) -> SqlExpr {
        let dialect = GenericDialect {};
        let mut stmts = Parser::parse_sql(&dialect, sql).expect("parse");
        let stmt = stmts.remove(0);
        let query = match stmt {
            sqlparser::ast::Statement::Query(q) => q,
            _ => panic!("not a query"),
        };
        match *query.body {
            SetExpr::Select(s) => (*s).selection.expect("a WHERE"),
            _ => panic!("not a SELECT"),
        }
    }

    #[test]
    fn split_and_conjuncts_flattens() {
        let w = where_expr("SELECT 1 FROM o WHERE a > 1 AND (k < 2 AND a < 9)");
        assert_eq!(split_and_conjuncts(&w).len(), 3);
    }

    #[test]
    fn split_and_conjuncts_single() {
        let w = where_expr("SELECT 1 FROM o WHERE a > 1");
        assert_eq!(split_and_conjuncts(&w).len(), 1);
    }

    #[test]
    fn as_exists_matches_exists_and_not_exists() {
        let w = where_expr("SELECT 1 FROM o WHERE EXISTS (SELECT 1 FROM s WHERE sk = o.k)");
        let (q, neg) = as_exists(&w).expect("EXISTS");
        assert!(!neg);
        assert!(subquery_is_correlated(q, &outer_cols(), &provider()).unwrap());

        let w2 = where_expr("SELECT 1 FROM o WHERE NOT EXISTS (SELECT 1 FROM s WHERE sk = o.k)");
        let (_, neg2) = as_exists(&w2).expect("NOT EXISTS");
        assert!(neg2);
    }

    #[test]
    fn uncorrelated_exists_is_not_correlated() {
        let w = where_expr("SELECT 1 FROM o WHERE EXISTS (SELECT 1 FROM s WHERE sk = 5)");
        let (q, _) = as_exists(&w).unwrap();
        assert!(!subquery_is_correlated(q, &outer_cols(), &provider()).unwrap());
    }

    #[test]
    fn conjunct_correlations_union_outer_and_inner() {
        // `o.a > (SELECT max(v) FROM s WHERE sk = o.k)`: the conjunct-level
        // `o.a` AND the subquery-internal `o.k` are both collected.
        let w = where_expr(
            "SELECT 1 FROM o WHERE o.a > (SELECT MAX(v) FROM s WHERE sk = o.k)",
        );
        let corrs = collect_conjunct_correlations(&w, &outer_cols(), &provider()).unwrap();
        let cols: HashSet<&str> = corrs.iter().map(|c| c.column.as_str()).collect();
        assert!(cols.contains("a"), "conjunct-level o.a is a correlation");
        assert!(cols.contains("k"), "subquery-internal o.k is a correlation");
    }
}
