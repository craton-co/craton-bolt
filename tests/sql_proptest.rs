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
use std::sync::atomic::{AtomicU64, Ordering};

use craton_bolt::plan::{lower_physical, parse_sql, DataType, Field, MemTableProvider, Schema};
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
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

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
        let mut rows: Vec<Vec<Cell>> = (0..n_rows).map(|_| Vec::with_capacity(n_cols)).collect();
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

    /// Build the second proptest fixture (`u`) as an Arrow batch.
    ///
    /// Must match the schema of the `u` table in the parent module's
    /// `fixture()`: columns `uk` (Int32 join key, overlapping `t.k`'s 0..8
    /// domain) and `uw` (Float64 payload). Kept small (8 rows, one per `uk`
    /// value) so an `INNER JOIN ... ON t.k = u.uk` stays bounded — each `t`
    /// row matches exactly one `u` row, so the join cardinality equals
    /// `t`'s row count rather than blowing up.
    pub fn fixture_batch_u() -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("uk", ArrowDataType::Int32, false),
            ArrowField::new("uw", ArrowDataType::Float64, false),
        ]));
        let n = 8usize;
        let uk: Int32Array = (0..n as i32).collect();
        let uw: Float64Array = (0..n).map(|i| (i as f64) * 1.25 + 0.5).collect();
        RecordBatch::try_new(schema, vec![Arc::new(uk), Arc::new(uw)])
            .expect("semantic_diff fixture batch u")
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
        // DDL mirrors the Arrow schema in `fixture_batch` / `fixture_batch_u`.
        conn.execute_batch(
            "CREATE TABLE t (\
                k INTEGER NOT NULL, \
                v DOUBLE NOT NULL, \
                s VARCHAR NOT NULL, \
                b BOOLEAN NOT NULL\
            ); \
            CREATE TABLE u (\
                uk INTEGER NOT NULL, \
                uw DOUBLE NOT NULL\
            );",
        )
        .expect("duckdb create table");
        load_into_duckdb(&conn, "t", &fixture_batch());
        load_into_duckdb(&conn, "u", &fixture_batch_u());
        engine
            .register_table("t", fixture_batch())
            .expect("bolt register_table t");
        engine
            .register_table("u", fixture_batch_u())
            .expect("bolt register_table u");
        (conn, engine)
    }

    /// Append every row of `batch` into the already-created DuckDB table
    /// `name`. Shared by both fixtures so the appender boilerplate lives in
    /// one place.
    fn load_into_duckdb(conn: &duckdb::Connection, name: &str, batch: &RecordBatch) {
        use duckdb::types::ToSql;
        let mut app = conn.appender(name).expect("duckdb appender");
        let n = batch.num_rows();
        let cols: Vec<arrow_array::ArrayRef> = (0..batch.num_columns())
            .map(|c| batch.column(c).clone())
            .collect();
        for i in 0..n {
            let cells: Vec<Cell> = cols.iter().map(|c| arrow_cell(c.as_ref(), i)).collect();
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
        // duckdb-1.2.2 only populates the statement's Arrow schema on execute,
        // so `column_names()` (which unwraps that schema) must be called AFTER
        // `query([])` — pre-query it panics at `raw_statement.rs:213`. Read the
        // names off the executed statement via `Rows::as_ref()`.
        let mut rows_iter = stmt.query([]).map_err(|e| e.to_string())?;
        let column_names: Vec<String> = rows_iter
            .as_ref()
            .ok_or_else(|| "duckdb executed statement unavailable".to_string())?
            .column_names();
        let ncols = column_names.len();
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

/// Two-table fixture covering one column per supported scalar type plus a
/// small join partner.
///
/// The grammar below references these column names verbatim, so they must
/// match exactly:
///   * table `t`: `k` (Int32), `v` (Float64), `s` (Utf8), `b` (Bool)
///   * table `u`: `uk` (Int32 join key), `uw` (Float64 payload)
///
/// `u` exists so the grammar can emit a bounded `INNER JOIN t ON t.k = u.uk`.
/// Its schema mirrors `semantic_diff::fixture_batch_u` so the same generated
/// SQL is valid against both the planner (here) and the runtime (DuckDB +
/// Bolt) in the semantic property.
fn fixture() -> MemTableProvider {
    let t = Schema::new(vec![
        Field {
            name: "k".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "v".into(),
            dtype: DataType::Float64,
            nullable: false,
        },
        Field {
            name: "s".into(),
            dtype: DataType::Utf8,
            nullable: false,
        },
        Field {
            name: "b".into(),
            dtype: DataType::Bool,
            nullable: false,
        },
    ]);
    let u = Schema::new(vec![
        Field {
            name: "uk".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "uw".into(),
            dtype: DataType::Float64,
            nullable: false,
        },
    ]);
    MemTableProvider::new()
        .with_table("t", t)
        .with_table("u", u)
}

// ---------------------------------------------------------------------------
// Grammar
// ---------------------------------------------------------------------------
//
// The top-level generated statement (`Stmt`) is either a single query or a
// set operation over two single-column queries:
//
//   stmt       := <query>
//               | <set_select> <set_op> [ALL] <set_select>
//   set_op     := UNION | EXCEPT | INTERSECT
//   set_select := SELECT <col> FROM t [WHERE <bool_expr>]
//
// A `query` has the shape:
//
//     SELECT <projection>
//     FROM t [INNER JOIN u ON t.k = u.uk]
//     [WHERE <bool_expr>]
//     [GROUP BY <col>]
//     [HAVING <bool_expr>]
//     [ORDER BY <col>]
//     [LIMIT <uint>]
//
// where
//
//   projection := <proj_item> { "," <proj_item> }
//   proj_item  := <col> | <agg> "(" <col> ")" | "COUNT(*)"
//               | <str_fn> | <case>
//   agg        := SUM | COUNT | MIN | MAX | AVG
//   str_fn     := UPPER(s) | LOWER(s) | LENGTH(s)
//               | SUBSTRING(s FROM <small> FOR <small>)
//   case       := CASE WHEN <comparison> THEN <literal> ELSE <literal> END
//   col        := "k" | "v" | "s" | "b"
//   bool_expr  := <comparison>
//               | <col> IN ( SELECT uk FROM u [WHERE uk <cmp_op> <i32>] )
//               | <bool_expr> AND <bool_expr>
//               | <bool_expr> OR  <bool_expr>
//   comparison := <col> <cmp_op> <literal>
//   cmp_op     := = | != | < | <= | > | >=
//   literal    := <i32> | <f64_bounded> | 'string'
//
// The grammar is intentionally lax: many combinations will be rejected by
// type-checking (e.g. `SUM(s)`, `WHERE k`, a `CASE` whose THEN/ELSE arms have
// different literal types) and that's fine — property 1 only requires that
// those rejections come back as `Err`, never as a panic, and the semantic
// property (4) treats a rejection on either engine as a SKIP.
//
// The JOIN, IN-subquery, set-op, CASE, and string-function productions were
// added to push the differential oracle past the trivial SELECT/WHERE subset
// into the historically bug-prone features. The second table `u` is a bounded
// join partner (one row per `t.k` value) so `INNER JOIN ... ON t.k = u.uk`
// does not blow up the row count.

#[derive(Clone, Debug)]
enum Col {
    K,
    V,
    S,
    B,
}
impl Col {
    fn as_str(&self) -> &'static str {
        match self {
            Col::K => "k",
            Col::V => "v",
            Col::S => "s",
            Col::B => "b",
        }
    }
}

#[derive(Clone, Debug)]
enum Agg {
    Sum,
    Count,
    Min,
    Max,
    Avg,
}
impl Agg {
    fn as_str(&self) -> &'static str {
        match self {
            Agg::Sum => "SUM",
            Agg::Count => "COUNT",
            Agg::Min => "MIN",
            Agg::Max => "MAX",
            Agg::Avg => "AVG",
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
enum CmpOp {
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
}
impl CmpOp {
    fn as_str(&self) -> &'static str {
        match self {
            CmpOp::Eq => "=",
            CmpOp::Neq => "!=",
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
        }
    }
}

/// One of the integer-typed columns usable as an `IN (subquery)` probe.
/// Restricted to integer columns so the subquery's projected column type
/// matches the probe type on both engines.
#[derive(Clone, Debug)]
enum IntCol {
    K,
}
impl IntCol {
    fn as_str(&self) -> &'static str {
        match self {
            IntCol::K => "k",
        }
    }
}

#[derive(Clone, Debug)]
enum BoolExpr {
    Cmp(Col, CmpOp, Lit),
    And(Box<BoolExpr>, Box<BoolExpr>),
    Or(Box<BoolExpr>, Box<BoolExpr>),
    /// `<int_col> IN (SELECT uk FROM u [WHERE uk <cmp> <int>])` — a correlated-
    /// free scalar/IN subquery against the join-partner table. The subquery's
    /// projected column (`uk`, Int32) matches the probe column's type so both
    /// engines accept it.
    InSubquery(IntCol, Option<(CmpOp, i32)>),
}
impl BoolExpr {
    fn render(&self) -> String {
        match self {
            BoolExpr::Cmp(c, op, l) => format!("{} {} {}", c.as_str(), op.as_str(), l.render()),
            BoolExpr::And(a, b) => format!("({} AND {})", a.render(), b.render()),
            BoolExpr::Or(a, b) => format!("({} OR {})", a.render(), b.render()),
            BoolExpr::InSubquery(c, filter) => {
                let mut sub = String::from("SELECT uk FROM u");
                if let Some((op, n)) = filter {
                    sub.push_str(&format!(" WHERE uk {} {}", op.as_str(), n));
                }
                format!("{} IN ({})", c.as_str(), sub)
            }
        }
    }
}

/// String scalar functions that the frontend supports (see
/// `sql_frontend.rs::scalar_fn_kind`). All take the Utf8 column `s` so the
/// argument type is always valid; `Substring` adds standard `FROM/FOR`
/// position arguments bounded to the fixture's short strings.
#[derive(Clone, Debug)]
enum StrFn {
    Upper,
    Lower,
    Length,
    /// `SUBSTRING(s FROM <start> FOR <len>)` — 1-based start, both small.
    Substring(u8, u8),
}
impl StrFn {
    fn render(&self) -> String {
        match self {
            StrFn::Upper => "UPPER(s)".to_string(),
            StrFn::Lower => "LOWER(s)".to_string(),
            StrFn::Length => "LENGTH(s)".to_string(),
            StrFn::Substring(start, len) => format!("SUBSTRING(s FROM {} FOR {})", start, len),
        }
    }
}

#[derive(Clone, Debug)]
enum ProjItem {
    Col(Col),
    Agg(Agg, Col),
    CountStar,
    /// A string-function call over `s` — exercises the scalar string ops.
    StrFn(StrFn),
    /// `CASE WHEN <bool_expr> THEN <lit> ELSE <lit> END` — a searched CASE.
    /// Both arms use the same literal kind so the result type is unambiguous
    /// across engines.
    Case(Box<BoolExpr>, Lit, Lit),
}
impl ProjItem {
    fn render(&self) -> String {
        match self {
            ProjItem::Col(c) => c.as_str().to_string(),
            ProjItem::Agg(a, c) => format!("{}({})", a.as_str(), c.as_str()),
            ProjItem::CountStar => "COUNT(*)".to_string(),
            ProjItem::StrFn(f) => f.render(),
            ProjItem::Case(w, t, e) => format!(
                "CASE WHEN {} THEN {} ELSE {} END",
                w.render(),
                t.render(),
                e.render()
            ),
        }
    }
}

#[derive(Clone, Debug)]
struct Query {
    projection: Vec<ProjItem>,
    /// When `true`, emit `FROM t INNER JOIN u ON t.k = u.uk`. The projection /
    /// clauses only ever reference `t`'s columns (`k`/`v`/`s`/`b`) which stay
    /// unambiguous because `u`'s columns are named `uk`/`uw`; the join is a
    /// bounded fan-out (one `u` row per `t.k`).
    join: bool,
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
        if self.join {
            out.push_str(" FROM t INNER JOIN u ON t.k = u.uk");
        } else {
            out.push_str(" FROM t");
        }
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

/// Top-level generated statement: either a single `Query` or a set operation
/// (`UNION` / `EXCEPT` / `INTERSECT`, optionally `ALL`) over two
/// single-column scalar selects.
///
/// Set-op branches are constrained to a single scalar projected column drawn
/// from `t` so both sides have matching arity and a column type both engines
/// accept; the column is the same on both sides to keep the result type
/// unambiguous.
#[derive(Clone, Debug)]
enum Stmt {
    Plain(Query),
    SetOp {
        kind: SetOpKind,
        all: bool,
        /// Shared projected column for both branches.
        col: Col,
        /// Optional per-branch WHERE filter.
        left_where: Option<BoolExpr>,
        right_where: Option<BoolExpr>,
    },
}

#[derive(Clone, Debug)]
enum SetOpKind {
    Union,
    Except,
    Intersect,
}
impl SetOpKind {
    fn as_str(&self) -> &'static str {
        match self {
            SetOpKind::Union => "UNION",
            SetOpKind::Except => "EXCEPT",
            SetOpKind::Intersect => "INTERSECT",
        }
    }
}

impl Stmt {
    fn render(&self) -> String {
        match self {
            Stmt::Plain(q) => q.render(),
            Stmt::SetOp {
                kind,
                all,
                col,
                left_where,
                right_where,
            } => {
                let branch = |w: &Option<BoolExpr>| {
                    let mut s = format!("SELECT {} FROM t", col.as_str());
                    if let Some(w) = w {
                        s.push_str(" WHERE ");
                        s.push_str(&w.render());
                    }
                    s
                };
                // `EXCEPT`/`INTERSECT ALL` are accepted by both engines; keep
                // the optional ALL only for UNION to stay on the well-trodden
                // path (set semantics otherwise).
                let all_kw = if *all && matches!(kind, SetOpKind::Union) {
                    " ALL"
                } else {
                    ""
                };
                format!(
                    "{} {}{} {}",
                    branch(left_where),
                    kind.as_str(),
                    all_kw,
                    branch(right_where)
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

fn col_strategy() -> impl Strategy<Value = Col> {
    prop_oneof![Just(Col::K), Just(Col::V), Just(Col::S), Just(Col::B),]
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

fn int_col_strategy() -> impl Strategy<Value = IntCol> {
    Just(IntCol::K)
}

fn str_fn_strategy() -> impl Strategy<Value = StrFn> {
    prop_oneof![
        Just(StrFn::Upper),
        Just(StrFn::Lower),
        Just(StrFn::Length),
        // 1-based start in 1..=3, length in 1..=3 — both bounded so the
        // SUBSTRING args stay within the fixture's 1..=3 char strings.
        (1u8..=3, 1u8..=3).prop_map(|(s, l)| StrFn::Substring(s, l)),
    ]
}

fn bool_expr_strategy() -> impl Strategy<Value = BoolExpr> {
    let leaf = prop_oneof![
        // Weight the plain comparison heavily so the recursive grammar still
        // dominates; the IN-subquery leaf is the new bug-prone feature.
        4 => (col_strategy(), cmp_op_strategy(), lit_strategy())
            .prop_map(|(c, op, l)| BoolExpr::Cmp(c, op, l)),
        1 => (int_col_strategy(), prop::option::of((cmp_op_strategy(), -2i32..10)))
            .prop_map(|(c, f)| BoolExpr::InSubquery(c, f)),
    ];
    // Recursive strategy: depth 3, branching factor 4, at most 16 nodes.
    leaf.prop_recursive(3, 16, 4, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| BoolExpr::And(Box::new(a), Box::new(b))),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| BoolExpr::Or(Box::new(a), Box::new(b))),
        ]
    })
}

fn proj_item_strategy() -> impl Strategy<Value = ProjItem> {
    prop_oneof![
        col_strategy().prop_map(ProjItem::Col),
        (agg_strategy(), col_strategy()).prop_map(|(a, c)| ProjItem::Agg(a, c)),
        Just(ProjItem::CountStar),
        str_fn_strategy().prop_map(ProjItem::StrFn),
        // CASE arm: a leaf comparison guard with two same-kind literals.
        (
            (col_strategy(), cmp_op_strategy(), lit_strategy())
                .prop_map(|(c, op, l)| BoolExpr::Cmp(c, op, l)),
            lit_strategy(),
            lit_strategy(),
        )
            .prop_map(|(w, t, e)| ProjItem::Case(Box::new(w), t, e)),
    ]
}

fn query_strategy() -> impl Strategy<Value = Query> {
    (
        // 1..=3 projection items — keeps SELECT lists short but non-trivial.
        prop::collection::vec(proj_item_strategy(), 1..=3),
        // ~1-in-4 queries add the INNER JOIN.
        prop::bool::weighted(0.25),
        prop::option::of(bool_expr_strategy()),
        prop::option::of(col_strategy()),
        prop::option::of(bool_expr_strategy()),
        prop::option::of(col_strategy()),
        prop::option::of(0u32..1000),
    )
        .prop_map(
            |(projection, join, where_clause, group_by, having, order_by, limit)| Query {
                projection,
                join,
                where_clause,
                group_by,
                having,
                order_by,
                limit,
            },
        )
}

fn set_op_kind_strategy() -> impl Strategy<Value = SetOpKind> {
    prop_oneof![
        Just(SetOpKind::Union),
        Just(SetOpKind::Except),
        Just(SetOpKind::Intersect),
    ]
}

/// Top-level statement strategy: mostly plain queries, with a minority of
/// set-op statements so the grammar covers `UNION`/`EXCEPT`/`INTERSECT`.
fn stmt_strategy() -> impl Strategy<Value = Stmt> {
    prop_oneof![
        // Plain queries dominate (they carry the JOIN / subquery / CASE /
        // string-fn coverage); set-ops are a smaller dedicated slice.
        4 => query_strategy().prop_map(Stmt::Plain),
        1 => (
            set_op_kind_strategy(),
            any::<bool>(),
            col_strategy(),
            prop::option::of(bool_expr_strategy()),
            prop::option::of(bool_expr_strategy()),
        )
            .prop_map(|(kind, all, col, left_where, right_where)| Stmt::SetOp {
                kind,
                all,
                col,
                left_where,
                right_where,
            }),
    ]
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
    fn prop_parse_never_panics(q in stmt_strategy()) {
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
    fn prop_schema_never_panics(q in stmt_strategy()) {
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
    fn prop_lower_never_panics(q in stmt_strategy()) {
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

/// Number of cases the semantic property runs (mirrors the
/// `ProptestConfig::with_cases` below). Used to know when the *last* case has
/// run so the executed-fraction floor can be enforced exactly once per run.
const SEMANTIC_CASES: u64 = 32;

/// Minimum fraction of generated queries that must reach a real DuckDB-vs-Bolt
/// row comparison for the differential property to be considered meaningful.
///
/// Without a floor, `semantic_diff_against_duckdb` silently no-ops: a
/// regression that rejects 100% of generated queries on one engine still
/// "passes" with zero comparisons. 25% is a deliberately loose bar — the
/// grammar intentionally emits many type-invalid queries (e.g. `SUM(s)`,
/// `WHERE k`) — but it is high enough that a wholesale rejection regression
/// trips the assert instead of hiding.
const MIN_EXECUTED_FRACTION: f64 = 0.25;

// Process-wide counters for the executed-fraction floor. The semantic
// property runs its cases sequentially within one process, so plain atomics
// suffice (no cross-test contention — only `semantic_diff_against_duckdb`
// touches these). `generated` counts every case that entered the body;
// `compared` counts those that produced a real two-engine row comparison.
static SEMANTIC_GENERATED: AtomicU64 = AtomicU64::new(0);
static SEMANTIC_COMPARED: AtomicU64 = AtomicU64::new(0);

proptest! {
    #![proptest_config(ProptestConfig::with_cases(SEMANTIC_CASES as u32))]

    /// Property 4 (semantic): when both Bolt and DuckDB accept the same
    /// generated query, their result sets must agree row-by-row under
    /// `ResultSet::approx_equal`. A rejection on either side is a SKIP —
    /// the panic-only properties above already cover "rejection is clean".
    ///
    /// To stop the property from silently no-op'ing (every case rejected →
    /// zero comparisons → vacuous pass), it tracks the generated-vs-compared
    /// counts process-wide and, on the final case of the run, asserts the
    /// executed fraction is at least [`MIN_EXECUTED_FRACTION`].
    #[test]
    #[ignore = "gpu:proptest-semantic"]
    fn semantic_diff_against_duckdb(q in stmt_strategy()) {
        let sql = q.render();
        // `Relaxed` is fine: these are independent counters with no ordering
        // dependency on other memory, and the floor check only reads them
        // after the run-terminating increment.
        SEMANTIC_GENERATED.fetch_add(1, Ordering::Relaxed);
        // One engine pair per case. The CUDA + DuckDB init cost is real, but
        // sharing state across proptest iterations risks cross-contamination
        // (e.g. DuckDB's `t` accumulating temp objects); a fresh pair is the
        // safer default until the harness is hardened.
        let (conn, engine) = semantic_diff::setup_engines();
        match semantic_diff::run_both(&conn, &engine, &sql) {
            Ok((duck, bolt)) => {
                SEMANTIC_COMPARED.fetch_add(1, Ordering::Relaxed);
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

        // Executed-fraction FLOOR. Enforced once, on the case that brings the
        // generated count up to the configured total (proptest runs cases
        // sequentially, so this is the last case). If proptest re-seeds and
        // runs extra cases, the `>=` still fires on the first qualifying one;
        // subsequent cases skip the check (`generated` keeps climbing).
        let generated = SEMANTIC_GENERATED.load(Ordering::Relaxed);
        if generated == SEMANTIC_CASES {
            let compared = SEMANTIC_COMPARED.load(Ordering::Relaxed);
            let fraction = compared as f64 / generated as f64;
            prop_assert!(
                fraction >= MIN_EXECUTED_FRACTION,
                "semantic_diff_against_duckdb executed too few cases: \
                 {compared}/{generated} = {fraction:.3} compared, \
                 floor is {MIN_EXECUTED_FRACTION:.3}. Either both engines are \
                 rejecting nearly everything (a real regression) or the grammar \
                 drifted to emit mostly-invalid SQL.",
            );
        }
    }
}
