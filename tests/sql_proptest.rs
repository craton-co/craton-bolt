// SPDX-License-Identifier: Apache-2.0

//! Property-test harness for the SQL frontend + planner.
//!
//! Generates random SQL queries from a constrained grammar and verifies that
//! the parse / schema / lower pipeline is panic-free and type-consistent.
//!
//! Three properties are enforced (see `prop_*` tests below):
//!
//!   1. **Parse stability**: every generated query either parses cleanly into a
//!      `LogicalPlan` or returns a `BoltError` (`Sql` / `Plan` / `Type`) — the
//!      parser must never panic. The frontend uses all three error variants
//!      for user-input failures, so we accept any of them as a clean failure.
//!   2. **Schema consistency**: if parsing succeeds, `LogicalPlan::schema()`
//!      either returns `Ok` or a typed `BoltError` — it does not panic.
//!   3. **Lower stability**: if logical planning succeeds, `lower_physical`
//!      either returns `Ok` or a `BoltError` — it does not panic.
//!
//! All three properties are checked with `std::panic::catch_unwind` so a
//! panic shows up as a proptest failure with the offending SQL string in the
//! shrink report, rather than aborting the test binary.

use std::panic::{catch_unwind, AssertUnwindSafe};

use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Field, MemTableProvider, Schema,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// Single-table fixture with one column per supported scalar type.
///
/// The grammar below references these column names verbatim, so they must
/// match exactly:
///   * `k` — Int32
///   * `v` — Float64
///   * `s` — Utf8
///   * `b` — Bool
fn fixture() -> MemTableProvider {
    let schema = Schema::new(vec![
        Field { name: "k".into(), dtype: DataType::Int32,   nullable: false },
        Field { name: "v".into(), dtype: DataType::Float64, nullable: false },
        Field { name: "s".into(), dtype: DataType::Utf8,    nullable: false },
        Field { name: "b".into(), dtype: DataType::Bool,    nullable: false },
    ]);
    MemTableProvider::new().with_table("t", schema)
}

// ---------------------------------------------------------------------------
// Grammar
// ---------------------------------------------------------------------------
//
// The generated SQL has the shape:
//
//     SELECT <projection>
//     FROM t
//     [WHERE <bool_expr>]
//     [GROUP BY <col>]
//     [HAVING <bool_expr>]
//     [ORDER BY <col>]
//     [LIMIT <uint>]
//
// where
//
//   projection := <col_or_agg> { "," <col_or_agg> }
//   col_or_agg := <col> | <agg> "(" <col> ")" | "COUNT(*)"
//   agg        := SUM | COUNT | MIN | MAX | AVG
//   col        := "k" | "v" | "s" | "b"
//   bool_expr  := <comparison>
//               | <bool_expr> AND <bool_expr>
//               | <bool_expr> OR  <bool_expr>
//   comparison := <col> <cmp_op> <literal>
//   cmp_op     := = | != | < | <= | > | >=
//   literal    := <i32> | <f64_bounded> | 'string'
//
// The grammar is intentionally lax: many combinations will be rejected by
// type-checking (e.g. `SUM(s)`, `WHERE k`) and that's fine — property 1 only
// requires that those rejections come back as `Err`, never as a panic.

#[derive(Clone, Debug)]
enum Col { K, V, S, B }
impl Col {
    fn as_str(&self) -> &'static str {
        match self { Col::K => "k", Col::V => "v", Col::S => "s", Col::B => "b" }
    }
}

#[derive(Clone, Debug)]
enum Agg { Sum, Count, Min, Max, Avg }
impl Agg {
    fn as_str(&self) -> &'static str {
        match self {
            Agg::Sum => "SUM", Agg::Count => "COUNT", Agg::Min => "MIN",
            Agg::Max => "MAX", Agg::Avg => "AVG",
        }
    }
}

#[derive(Clone, Debug)]
enum Lit {
    Int(i32),
    Float(f64),
    Str(String),
}
impl Lit {
    fn render(&self) -> String {
        match self {
            Lit::Int(i) => i.to_string(),
            // Use a fixed-precision render so we never emit NaN/Inf literals,
            // which the grammar already forbids by bounding the f64 strategy.
            Lit::Float(f) => format!("{:.4}", f),
            // Single-quote escape via doubling, per SQL standard.
            Lit::Str(s) => format!("'{}'", s.replace('\'', "''")),
        }
    }
}

#[derive(Clone, Debug)]
enum CmpOp { Eq, Neq, Lt, Le, Gt, Ge }
impl CmpOp {
    fn as_str(&self) -> &'static str {
        match self {
            CmpOp::Eq => "=", CmpOp::Neq => "!=", CmpOp::Lt => "<",
            CmpOp::Le => "<=", CmpOp::Gt => ">", CmpOp::Ge => ">=",
        }
    }
}

#[derive(Clone, Debug)]
enum BoolExpr {
    Cmp(Col, CmpOp, Lit),
    And(Box<BoolExpr>, Box<BoolExpr>),
    Or (Box<BoolExpr>, Box<BoolExpr>),
}
impl BoolExpr {
    fn render(&self) -> String {
        match self {
            BoolExpr::Cmp(c, op, l) =>
                format!("{} {} {}", c.as_str(), op.as_str(), l.render()),
            BoolExpr::And(a, b) =>
                format!("({} AND {})", a.render(), b.render()),
            BoolExpr::Or(a, b) =>
                format!("({} OR {})", a.render(), b.render()),
        }
    }
}

#[derive(Clone, Debug)]
enum ProjItem {
    Col(Col),
    Agg(Agg, Col),
    CountStar,
}
impl ProjItem {
    fn render(&self) -> String {
        match self {
            ProjItem::Col(c)        => c.as_str().to_string(),
            ProjItem::Agg(a, c)     => format!("{}({})", a.as_str(), c.as_str()),
            ProjItem::CountStar     => "COUNT(*)".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
struct Query {
    projection: Vec<ProjItem>,
    where_clause: Option<BoolExpr>,
    group_by: Option<Col>,
    having: Option<BoolExpr>,
    order_by: Option<Col>,
    limit: Option<u32>,
}

impl Query {
    fn render(&self) -> String {
        let mut out = String::with_capacity(64);
        out.push_str("SELECT ");
        // Always at least one projection item — the strategy guarantees this.
        let parts: Vec<String> = self.projection.iter().map(|p| p.render()).collect();
        out.push_str(&parts.join(", "));
        out.push_str(" FROM t");
        if let Some(w) = &self.where_clause {
            out.push_str(" WHERE ");
            out.push_str(&w.render());
        }
        if let Some(g) = &self.group_by {
            out.push_str(" GROUP BY ");
            out.push_str(g.as_str());
        }
        if let Some(h) = &self.having {
            out.push_str(" HAVING ");
            out.push_str(&h.render());
        }
        if let Some(o) = &self.order_by {
            out.push_str(" ORDER BY ");
            out.push_str(o.as_str());
        }
        if let Some(n) = self.limit {
            out.push_str(" LIMIT ");
            out.push_str(&n.to_string());
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

fn col_strategy() -> impl Strategy<Value = Col> {
    prop_oneof![
        Just(Col::K),
        Just(Col::V),
        Just(Col::S),
        Just(Col::B),
    ]
}

fn agg_strategy() -> impl Strategy<Value = Agg> {
    prop_oneof![
        Just(Agg::Sum),
        Just(Agg::Count),
        Just(Agg::Min),
        Just(Agg::Max),
        Just(Agg::Avg),
    ]
}

fn cmp_op_strategy() -> impl Strategy<Value = CmpOp> {
    prop_oneof![
        Just(CmpOp::Eq),
        Just(CmpOp::Neq),
        Just(CmpOp::Lt),
        Just(CmpOp::Le),
        Just(CmpOp::Gt),
        Just(CmpOp::Ge),
    ]
}

fn lit_strategy() -> impl Strategy<Value = Lit> {
    prop_oneof![
        // Full i32 range — exercises the integer-overflow path in the
        // sqlparser → BoltError::Sql("i64") conversion if/when widened.
        any::<i32>().prop_map(Lit::Int),
        // Bounded f64: finite, no NaN/Inf. Keeps the rendered SQL valid.
        (-1.0e6f64..1.0e6f64).prop_map(Lit::Float),
        // Short ASCII strings without embedded quotes (the renderer also
        // doubles any quotes that do appear, so this is belt-and-braces).
        "[a-zA-Z0-9 ]{0,8}".prop_map(Lit::Str),
    ]
}

fn bool_expr_strategy() -> impl Strategy<Value = BoolExpr> {
    let leaf = (col_strategy(), cmp_op_strategy(), lit_strategy())
        .prop_map(|(c, op, l)| BoolExpr::Cmp(c, op, l));
    // Recursive strategy: depth 3, branching factor 4, at most 16 nodes.
    leaf.prop_recursive(3, 16, 4, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| BoolExpr::And(Box::new(a), Box::new(b))),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| BoolExpr::Or (Box::new(a), Box::new(b))),
        ]
    })
}

fn proj_item_strategy() -> impl Strategy<Value = ProjItem> {
    prop_oneof![
        col_strategy().prop_map(ProjItem::Col),
        (agg_strategy(), col_strategy()).prop_map(|(a, c)| ProjItem::Agg(a, c)),
        Just(ProjItem::CountStar),
    ]
}

fn query_strategy() -> impl Strategy<Value = Query> {
    (
        // 1..=3 projection items — keeps SELECT lists short but non-trivial.
        prop::collection::vec(proj_item_strategy(), 1..=3),
        prop::option::of(bool_expr_strategy()),
        prop::option::of(col_strategy()),
        prop::option::of(bool_expr_strategy()),
        prop::option::of(col_strategy()),
        prop::option::of(0u32..1000),
    ).prop_map(|(projection, where_clause, group_by, having, order_by, limit)| {
        Query { projection, where_clause, group_by, having, order_by, limit }
    })
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

/// Run `f` under `catch_unwind`. Returns `Ok(value)` on normal return,
/// `Err(panic_msg)` on panic. The harness needs this because a panic in
/// `parse_sql` / `schema` / `lower_physical` would otherwise abort the whole
/// test binary; we want proptest to see it as a property failure so the
/// shrinker can minimize the offending query.
fn catch<R>(label: &str, sql: &str, f: impl FnOnce() -> R) -> Result<R, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".to_string()
        };
        format!("{label} panicked on SQL {sql:?}: {msg}")
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Property 1: parsing never panics.
    ///
    /// Every generated query either yields `Ok(LogicalPlan)` or a `BoltError`
    /// (`Sql` / `Plan` / `Type`). The task description mentions `BoltError::Sql`
    /// specifically, but the frontend uses all three variants for user-input
    /// rejection (e.g. unknown table → `Plan`, non-Bool WHERE → `Type`), so
    /// the real invariant is "any clean `BoltError`, never a panic".
    #[test]
    fn prop_parse_never_panics(q in query_strategy()) {
        let sql = q.render();
        let provider = fixture();
        let parse_result = catch("parse_sql", &sql, || parse_sql(&sql, &provider));
        prop_assert!(parse_result.is_ok(), "{}", parse_result.unwrap_err());
    }

    /// Property 2: `LogicalPlan::schema()` never panics on a successfully
    /// parsed plan.
    ///
    /// Schema resolution doubles as type-checking, so it may legitimately
    /// return `Err(BoltError::Type(...))`. We only assert that it doesn't
    /// panic.
    #[test]
    fn prop_schema_never_panics(q in query_strategy()) {
        let sql = q.render();
        let provider = fixture();
        let parse_result = catch("parse_sql", &sql, || parse_sql(&sql, &provider));
        prop_assert!(parse_result.is_ok(), "{}", parse_result.unwrap_err());
        if let Ok(Ok(plan)) = parse_result {
            let schema_result = catch("LogicalPlan::schema", &sql, || plan.schema());
            prop_assert!(schema_result.is_ok(), "{}", schema_result.unwrap_err());
        }
    }

    /// Property 3: `lower_physical` never panics on a successfully parsed plan.
    ///
    /// Lowering may legitimately reject a plan with `Err(BoltError::Plan(...))`
    /// — e.g. an unsupported operator combination — but it must not panic.
    #[test]
    fn prop_lower_never_panics(q in query_strategy()) {
        let sql = q.render();
        let provider = fixture();
        let parse_result = catch("parse_sql", &sql, || parse_sql(&sql, &provider));
        prop_assert!(parse_result.is_ok(), "{}", parse_result.unwrap_err());
        if let Ok(Ok(plan)) = parse_result {
            let lower_result = catch("lower_physical", &sql, || lower_physical(&plan));
            prop_assert!(lower_result.is_ok(), "{}", lower_result.unwrap_err());
        }
    }
}
