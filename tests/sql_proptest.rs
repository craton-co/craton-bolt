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

mod common;

// ---------------------------------------------------------------------------
// Differential helpers (semantic property, gated behind `gpu:proptest-semantic`)
// ---------------------------------------------------------------------------
//
// Rust integration tests are separate crates, so we can't `use` the helpers
// from `tests/diff_duckdb.rs` directly — but we *can* share constants and
// PRNGs through `tests/common/mod.rs` (review L2). The cell-equality types
// below are still a small copy from `tests/diff_duckdb.rs`; what they need
// from the shared module is just the `REL_TOL` tolerance.
mod semantic_diff {
    use std::sync::Arc;

    use arrow_array::{
        Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
        StringArray,
    };
    use arrow_schema::{
        DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
    };

    use super::common::REL_TOL;

    #[derive(Debug, Clone, PartialEq)]
    pub enum Cell {
        Null,
        Bool(bool),
        Int(i64),
        Float(f64),
        Str(String),
    }

    impl Cell {
        pub fn approx_eq(&self, other: &Cell) -> bool {
            match (self, other) {
                (Cell::Null, Cell::Null) => true,
                (Cell::Bool(a), Cell::Bool(b)) => a == b,
                (Cell::Int(a), Cell::Int(b)) => a == b,
                (Cell::Int(a), Cell::Float(b)) | (Cell::Float(b), Cell::Int(a)) => {
                    close_enough(*a as f64, *b)
                }
                (Cell::Float(a), Cell::Float(b)) => close_enough(*a, *b),
                (Cell::Str(a), Cell::Str(b)) => a == b,
                _ => false,
            }
        }
    }

    impl std::fmt::Display for Cell {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Cell::Null => write!(f, "NULL"),
                Cell::Bool(b) => write!(f, "{b}"),
                Cell::Int(i) => write!(f, "{i}"),
                Cell::Float(x) => write!(f, "{x:.12}"),
                Cell::Str(s) => write!(f, "{s:?}"),
            }
        }
    }

    fn close_enough(a: f64, b: f64) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        let diff = (a - b).abs();
        let mag = a.abs().max(b.abs()).max(1.0);
        diff / mag <= REL_TOL
    }

    #[derive(Debug, Clone)]
    pub struct ResultSet {
        pub columns: Vec<String>,
        pub rows: Vec<Vec<Cell>>,
    }

    impl ResultSet {
        /// Order-agnostic comparison: stringify each row, sort, then diff.
        /// Generated queries may or may not contain `ORDER BY`, so we always
        /// canonicalise (this is conservative — we lose the order check on
        /// queries that did specify one, but a row-set mismatch still trips).
        pub fn canonicalise(mut self) -> Self {
            self.rows.sort_by(|a, b| {
                for (x, y) in a.iter().zip(b.iter()) {
                    let xs = x.to_string();
                    let ys = y.to_string();
                    match xs.cmp(&ys) {
                        std::cmp::Ordering::Equal => continue,
                        other => return other,
                    }
                }
                a.len().cmp(&b.len())
            });
            self
        }

        /// `approx_equal` in the style of `diff_duckdb.rs::assert_results_equal`,
        /// but returns a bool instead of panicking so the proptest harness can
        /// `prop_assert!` and surface the offending SQL through the shrinker.
        pub fn approx_equal(&self, other: &ResultSet) -> bool {
            if self.rows.len() != other.rows.len() {
                return false;
            }
            for (a, b) in self.rows.iter().zip(other.rows.iter()) {
                if a.len() != b.len() {
                    return false;
                }
                for (x, y) in a.iter().zip(b.iter()) {
                    if !x.approx_eq(y) {
                        return false;
                    }
                }
            }
            true
        }
    }

    fn duckdb_value_to_cell(v: duckdb::types::ValueRef<'_>) -> Cell {
        use duckdb::types::ValueRef as V;
        match v {
            V::Null => Cell::Null,
            V::Boolean(b) => Cell::Bool(b),
            V::TinyInt(n) => Cell::Int(n as i64),
            V::SmallInt(n) => Cell::Int(n as i64),
            V::Int(n) => Cell::Int(n as i64),
            V::BigInt(n) => Cell::Int(n),
            V::HugeInt(n) => Cell::Int(n as i64),
            V::UTinyInt(n) => Cell::Int(n as i64),
            V::USmallInt(n) => Cell::Int(n as i64),
            V::UInt(n) => Cell::Int(n as i64),
            V::UBigInt(n) => Cell::Int(n as i64),
            V::Float(f) => Cell::Float(f as f64),
            V::Double(f) => Cell::Float(f),
            V::Text(bytes) => Cell::Str(String::from_utf8_lossy(bytes).into_owned()),
            // Any unsupported type means a future grammar extension reached
            // this helper without updating the decoder; surface it loudly.
            other => panic!("semantic_diff: unsupported DuckDB value type {other:?}"),
        }
    }

    fn arrow_cell(arr: &dyn Array, idx: usize) -> Cell {
        if arr.is_null(idx) {
            return Cell::Null;
        }
        if let Some(a) = arr.as_any().downcast_ref::<BooleanArray>() {
            return Cell::Bool(a.value(idx));
        }
        if let Some(a) = arr.as_any().downcast_ref::<Int32Array>() {
            return Cell::Int(a.value(idx) as i64);
        }
        if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
            return Cell::Int(a.value(idx));
        }
        if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
            return Cell::Float(a.value(idx) as f64);
        }
        if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
            return Cell::Float(a.value(idx));
        }
        if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
            return Cell::Str(a.value(idx).to_string());
        }
        panic!(
            "semantic_diff: unsupported Arrow dtype {:?}",
            arr.data_type()
        );
    }

    fn arrow_batch_to_resultset(batch: &RecordBatch) -> ResultSet {
        let n_rows = batch.num_rows();
        let n_cols = batch.num_columns();
        let columns: Vec<String> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let mut rows: Vec<Vec<Cell>> =
            (0..n_rows).map(|_| Vec::with_capacity(n_cols)).collect();
        for c in 0..n_cols {
            let col = batch.column(c);
            for (r, row) in rows.iter_mut().enumerate().take(n_rows) {
                row.push(arrow_cell(col.as_ref(), r));
            }
        }
        ResultSet { columns, rows }
    }

    /// Build the proptest fixture as an Arrow batch.
    ///
    /// Must match the schema of `fixture()` in the parent module so a query
    /// generated against that schema works against both engines: same column
    /// names, same dtypes, all non-nullable.
    pub fn fixture_batch() -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Float64, false),
            ArrowField::new("s", ArrowDataType::Utf8, false),
            ArrowField::new("b", ArrowDataType::Boolean, false),
        ]));
        // Small deterministic fixture: 64 rows, low cardinality on `k` (8)
        // so GROUP BY queries have multiple groups; `s` cycles through three
        // short strings; `v` is finite-but-varied.
        let n = 64usize;
        let k: Int32Array = (0..n as i32).map(|i| i % 8).collect();
        let v: Float64Array = (0..n).map(|i| (i as f64) * 0.5 - 8.0).collect();
        let s: StringArray = (0..n)
            .map(|i| match i % 3 {
                0 => Some("a"),
                1 => Some("bb"),
                _ => Some("ccc"),
            })
            .collect();
        let b: BooleanArray = (0..n).map(|i| Some(i % 2 == 0)).collect();
        RecordBatch::try_new(
            schema,
            vec![Arc::new(k), Arc::new(v), Arc::new(s), Arc::new(b)],
        )
        .expect("semantic_diff fixture batch")
    }

    /// Set up DuckDB + Bolt engines and load the fixture into both. Returns
    /// `None` if `Engine::new` fails (no CUDA device) — caller treats that as
    /// "skip this property invocation" rather than an outright test failure,
    /// because the harness is `#[ignore]`'d already; we only reach this code
    /// path via `--ignored`, so an unconfigured GPU host is a real bug.
    pub fn setup_engines() -> (duckdb::Connection, craton_bolt::Engine) {
        let mut engine = craton_bolt::Engine::new()
            .expect("CUDA init failed (semantic_diff_against_duckdb is gpu-only)");
        let conn = duckdb::Connection::open_in_memory().expect("duckdb open");
        // DDL mirrors the Arrow schema in `fixture_batch`.
        conn.execute_batch(
            "CREATE TABLE t (\
                k INTEGER NOT NULL, \
                v DOUBLE NOT NULL, \
                s VARCHAR NOT NULL, \
                b BOOLEAN NOT NULL\
            );",
        )
        .expect("duckdb create table");
        let batch = fixture_batch();
        {
            use duckdb::types::ToSql;
            let mut app = conn.appender("t").expect("duckdb appender");
            let n = batch.num_rows();
            let cols: Vec<arrow_array::ArrayRef> =
                (0..batch.num_columns()).map(|c| batch.column(c).clone()).collect();
            for i in 0..n {
                let cells: Vec<Cell> =
                    cols.iter().map(|c| arrow_cell(c.as_ref(), i)).collect();
                let boxed: Vec<Box<dyn ToSql>> = cells
                    .into_iter()
                    .map(|c| -> Box<dyn ToSql> {
                        match c {
                            Cell::Null => Box::new(Option::<i64>::None),
                            Cell::Bool(b) => Box::new(b),
                            Cell::Int(i) => Box::new(i),
                            Cell::Float(f) => Box::new(f),
                            Cell::Str(s) => Box::new(s),
                        }
                    })
                    .collect();
                let refs: Vec<&dyn ToSql> = boxed.iter().map(|b| &**b).collect();
                app.append_row(duckdb::appender_params_from_iter(refs.iter().copied()))
                    .expect("duckdb appender append_row");
            }
            app.flush().expect("duckdb appender flush");
        }
        engine
            .register_table("t", batch)
            .expect("bolt register_table");
        (conn, engine)
    }

    /// Run the same SQL on both engines and normalise to `ResultSet`s.
    /// Errors propagate as `Err(String)` so the proptest harness can treat a
    /// rejected-by-one-engine query as a SKIP (see `semantic_diff_against_duckdb`).
    pub fn run_both(
        conn: &duckdb::Connection,
        engine: &craton_bolt::Engine,
        sql: &str,
    ) -> Result<(ResultSet, ResultSet), String> {
        let duck = run_duckdb(conn, sql).map_err(|e| format!("duckdb: {e}"))?;
        let bolt = run_bolt(engine, sql).map_err(|e| format!("bolt: {e}"))?;
        Ok((duck, bolt))
    }

    fn run_duckdb(conn: &duckdb::Connection, sql: &str) -> Result<ResultSet, String> {
        let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;
        let column_names: Vec<String> = stmt.column_names();
        let ncols = column_names.len();
        let mut rows_iter = stmt.query([]).map_err(|e| e.to_string())?;
        let mut rows: Vec<Vec<Cell>> = Vec::new();
        while let Some(row) = rows_iter.next().map_err(|e| e.to_string())? {
            let mut out = Vec::with_capacity(ncols);
            for i in 0..ncols {
                let v = row.get_ref(i).map_err(|e| e.to_string())?;
                out.push(duckdb_value_to_cell(v));
            }
            rows.push(out);
        }
        Ok(ResultSet {
            columns: column_names,
            rows,
        })
    }

    fn run_bolt(engine: &craton_bolt::Engine, sql: &str) -> Result<ResultSet, String> {
        let handle = engine.sql(sql).map_err(|e| e.to_string())?;
        Ok(arrow_batch_to_resultset(handle.record_batch()))
    }
}

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
    #![proptest_config(ProptestConfig::with_cases(256))]

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

// ---------------------------------------------------------------------------
// Semantic property: differential vs DuckDB
// ---------------------------------------------------------------------------
//
// Stronger than the panic-only properties above: for every generated query
// we render to SQL, run it through BOTH engines via the full execution path,
// and (when both succeed) require the row-set to match approximately. This
// is the moral equivalent of `tests/diff_duckdb.rs` extended over the proptest
// grammar, so we reuse the same `ResultSet::approx_equal` semantics (tight
// float tolerance, exact otherwise).
//
// Gated `#[ignore]` because every invocation pays a real CUDA-init cost and
// a real DuckDB connection setup. Run with:
//
//     cargo test --test sql_proptest semantic_diff_against_duckdb -- --ignored
//
// Smaller case count (32) than the panic properties (256) because each case
// is two full SQL pipelines instead of three parse/plan/lower probes.

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Property 4 (semantic): when both Bolt and DuckDB accept the same
    /// generated query, their result sets must agree row-by-row under
    /// `ResultSet::approx_equal`. A rejection on either side is a SKIP —
    /// the panic-only properties above already cover "rejection is clean".
    #[test]
    #[ignore = "gpu:proptest-semantic"]
    fn semantic_diff_against_duckdb(q in query_strategy()) {
        let sql = q.render();
        // One engine pair per case. The CUDA + DuckDB init cost is real, but
        // sharing state across proptest iterations risks cross-contamination
        // (e.g. DuckDB's `t` accumulating temp objects); a fresh pair is the
        // safer default until the harness is hardened.
        let (conn, engine) = semantic_diff::setup_engines();
        match semantic_diff::run_both(&conn, &engine, &sql) {
            Ok((duck, bolt)) => {
                let duck = duck.canonicalise();
                let bolt = bolt.canonicalise();
                prop_assert!(
                    duck.approx_equal(&bolt),
                    "semantic divergence on SQL {sql:?}\nDuckDB columns: {:?}\nBolt   columns: {:?}\nDuckDB rows: {:?}\nBolt   rows:   {:?}",
                    duck.columns, bolt.columns, duck.rows, bolt.rows,
                );
            }
            // One side rejected — fine. The grammar emits many queries that
            // either engine will refuse on type grounds (e.g. `SUM(s)`);
            // those don't tell us anything about semantic agreement.
            Err(_) => {}
        }
    }
}
