// SPDX-License-Identifier: Apache-2.0

//! Differential test harness: **Craton Bolt vs DuckDB** on identical small
//! fixtures.
//!
//! DuckDB is the canonical embedded-OLAP oracle (see
//! `docs/COMPETITIVE_BENCHMARKING.md` and the existing `benches/olap_benchmarks.rs`
//! cross-check). For every query in the curated set below we:
//!
//!   1. Build a deterministic Arrow `RecordBatch` fixture.
//!   2. Load it into BOTH engines (DuckDB via `appender`, Bolt via
//!      [`craton_bolt::Engine::register_table`]).
//!   3. Run the same SQL on both, decode each result row-by-row into a
//!      normalised `Cell` representation, canonicalise the row order, and
//!      assert numerical agreement (`common::REL_TOL = 1e-9` for floats;
//!      exact for ints / bools / strings; null on one side must be null on
//!      the other).
//!   4. On mismatch, print BOTH result sets in full so the diff is obvious.
//!
//! The eight curated cases are chosen to instantly catch the three critical
//! bugs that motivated this harness (see comments next to each test):
//!
//! - **C1** "HAVING dropped" — caught by `diff_groupby_having_sum`.
//! - **C2** "NULL coerced to 0" — caught by `diff_agg_nulls_min_max_avg_count`.
//! - **C3** "DISTINCT hash collision" — caught by `diff_distinct_high_volume`.
//! - **C5** "multi-SUM column-alignment" — caught by `diff_groupby_multi_sum`.
//!
//! ### Running
//!
//! The harness compiles without CUDA (we still link against the engine's host
//! façade), but every test is `#[ignore]`'d because `Engine::new()` needs a
//! live CUDA device. On a GPU host:
//!
//! ```text
//! cargo test --test diff_duckdb -- --ignored
//! ```
//!
//! Mirrors the convention in `tests/e2e_tests.rs` and `tests/memory_tests.rs`.

use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

mod common;
use common::REL_TOL;

// ---------------------------------------------------------------------------
// Tolerances + cell representation
// ---------------------------------------------------------------------------
//
// `REL_TOL` is the shared test-suite-wide constant from `tests/common/mod.rs`
// — SUMs reordered by a parallel engine accumulate up to ~10 ULPs of
// rounding noise, so 1e-9 relative is comfortably above that floor while
// still tight enough to catch a genuine arithmetic bug. The bench crate
// uses `craton_bolt::REL_TOL_TEST` (same value, different binary).

/// Normalised representation of a single result cell from either engine.
///
/// We project every result into this enum before comparing so that the
/// equality check in [`Cell::approx_eq`] is the single source of truth
/// for "do these two cells agree?". Float comparison uses [`REL_TOL`];
/// every other type compares exactly. `Null` is its own variant — a null
/// on one side and `0`/`""` on the other does NOT compare equal (this is
/// the heart of the C2 regression check).
#[derive(Debug, Clone, PartialEq)]
enum Cell {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl Cell {
    fn approx_eq(&self, other: &Cell) -> bool {
        match (self, other) {
            (Cell::Null, Cell::Null) => true,
            (Cell::Bool(a), Cell::Bool(b)) => a == b,
            (Cell::Int(a), Cell::Int(b)) => a == b,
            // Cross-type numeric: DuckDB may return Int64 where Bolt returns
            // Float64 (e.g. COUNT()), or vice versa. Promote to f64 with the
            // standard tolerance — keeps us robust to harmless dtype skew.
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

/// A row is a vector of cells; a result is a vector of rows plus the column
/// names (used only for prettier mismatch reports).
#[derive(Debug, Clone)]
struct ResultSet {
    columns: Vec<String>,
    rows: Vec<Vec<Cell>>,
}

impl ResultSet {
    /// Sort rows by the lexicographic ordering of their stringified cells.
    /// Neither engine guarantees a particular order without `ORDER BY`, so
    /// we canonicalise before comparing. We deliberately AVOID sorting
    /// queries that contain `ORDER BY` / `LIMIT` (see [`diff_query_ordered`]).
    fn canonicalise(mut self) -> Self {
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
}

// ---------------------------------------------------------------------------
// Decoders: DuckDB rows -> ResultSet, Arrow RecordBatch -> ResultSet
// ---------------------------------------------------------------------------

fn duckdb_run(conn: &duckdb::Connection, sql: &str) -> ResultSet {
    let mut stmt = conn.prepare(sql).expect("duckdb prepare");
    // `column_names` / `column_count` read the schema cached by `prepare`
    // (see duckdb-1.2.2 `raw_statement.rs` where `prepare` populates
    // `self.schema`). The docstring's "panics if not yet executed" warning
    // is stale from the rusqlite ancestor — in DuckDB's Rust binding the
    // schema is populated up front and these accessors are safe pre-query.
    let column_names: Vec<String> = stmt.column_names();
    let ncols = column_names.len();

    let mut rows_iter = stmt.query([]).expect("duckdb query");
    let mut rows: Vec<Vec<Cell>> = Vec::new();
    while let Some(row) = rows_iter.next().expect("duckdb row") {
        let mut out = Vec::with_capacity(ncols);
        for i in 0..ncols {
            // `get_ref` exposes the cell's runtime type so we can map any
            // numeric / bool / string / null result column into our `Cell`
            // enum without pre-declaring the schema.
            let v = row.get_ref(i).expect("duckdb cell");
            out.push(duckdb_value_to_cell(v));
        }
        rows.push(out);
    }
    ResultSet {
        columns: column_names,
        rows,
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
        // Anything else (decimal, timestamp, blob, list, struct, …) is not
        // exercised by the curated query set; surface unsupported types as
        // a panic so a future test author updates this decoder rather than
        // silently mis-comparing.
        other => panic!("diff_duckdb: unsupported DuckDB value type {other:?}"),
    }
}

fn bolt_run(engine: &craton_bolt::Engine, sql: &str) -> ResultSet {
    let handle = engine.sql(sql).expect("craton-bolt sql");
    arrow_batch_to_resultset(handle.record_batch())
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
        "diff_duckdb: unsupported Arrow dtype {:?} (extend arrow_cell)",
        arr.data_type()
    );
}

// ---------------------------------------------------------------------------
// Comparison + pretty-printed mismatch report
// ---------------------------------------------------------------------------

fn assert_results_equal(label: &str, duck: &ResultSet, bolt: &ResultSet) {
    let mut errs: Vec<String> = Vec::new();
    if duck.rows.len() != bolt.rows.len() {
        errs.push(format!(
            "row count mismatch: DuckDB={} Bolt={}",
            duck.rows.len(),
            bolt.rows.len()
        ));
    }
    let n_rows = duck.rows.len().min(bolt.rows.len());
    for i in 0..n_rows {
        let dr = &duck.rows[i];
        let br = &bolt.rows[i];
        if dr.len() != br.len() {
            errs.push(format!(
                "row {i}: column count mismatch DuckDB={} Bolt={}",
                dr.len(),
                br.len()
            ));
            continue;
        }
        for c in 0..dr.len() {
            if !dr[c].approx_eq(&br[c]) {
                errs.push(format!(
                    "row {i}, col {c}: DuckDB={} Bolt={}",
                    dr[c], br[c]
                ));
            }
        }
    }
    if !errs.is_empty() {
        let mut msg = format!("\n=== [{label}] differential check FAILED ===\n");
        msg.push_str(&format!("DuckDB columns: {:?}\n", duck.columns));
        msg.push_str(&format!("Bolt   columns: {:?}\n", bolt.columns));
        msg.push_str("DuckDB rows:\n");
        for (i, r) in duck.rows.iter().enumerate() {
            msg.push_str(&format!("  {i:>3}: "));
            for c in r {
                msg.push_str(&format!("{c}  "));
            }
            msg.push('\n');
        }
        msg.push_str("Bolt rows:\n");
        for (i, r) in bolt.rows.iter().enumerate() {
            msg.push_str(&format!("  {i:>3}: "));
            for c in r {
                msg.push_str(&format!("{c}  "));
            }
            msg.push('\n');
        }
        msg.push_str("Differences:\n");
        for e in &errs {
            msg.push_str(&format!("  - {e}\n"));
        }
        panic!("{msg}");
    }
}

// ---------------------------------------------------------------------------
// Drivers: single-table and two-table diff entry points
// ---------------------------------------------------------------------------

/// Spec describing one logical table for [`diff_query`] / [`diff_query_ordered`].
struct TableSpec<'a> {
    name: &'a str,
    /// DuckDB `CREATE TABLE` body (the column list, no parens).
    ddl_cols: &'a str,
    /// The Arrow batch fed to Bolt AND streamed into DuckDB via `appender`.
    batch: RecordBatch,
}

/// Initialise both engines and load every `TableSpec` into them. Returns
/// the live pair so the caller can run SQL through each. Panics on CUDA
/// init failure — every test is `#[ignore]`'d, so the panic only fires
/// on the GPU host we explicitly opted in to.
fn setup(
    label: &str,
    tables: &[TableSpec<'_>],
) -> (duckdb::Connection, craton_bolt::Engine) {
    let mut engine = craton_bolt::Engine::new().unwrap_or_else(|e| {
        panic!("[{label}] CUDA init failed (need a GPU to run --ignored tests): {e}")
    });
    let conn = duckdb::Connection::open_in_memory().expect("duckdb open");
    for t in tables {
        load_table(&conn, &mut engine, t);
    }
    (conn, engine)
}

/// Differential check for an *order-independent* query: results from both
/// engines are sorted into a canonical order before comparison.
fn diff_query(label: &str, sql: &str, tables: &[TableSpec<'_>]) {
    let (conn, engine) = setup(label, tables);
    let duck = duckdb_run(&conn, sql).canonicalise();
    let bolt = bolt_run(&engine, sql).canonicalise();
    assert_results_equal(label, &duck, &bolt);
}

/// Differential check for a query whose result order is meaningful
/// (`ORDER BY` / `LIMIT`): DOES NOT canonicalise — row positions matter.
fn diff_query_ordered(label: &str, sql: &str, tables: &[TableSpec<'_>]) {
    let (conn, engine) = setup(label, tables);
    let duck = duckdb_run(&conn, sql);
    let bolt = bolt_run(&engine, sql);
    assert_results_equal(label, &duck, &bolt);
}

/// Create a DuckDB table from `ddl_cols`, then stream `batch` into both
/// DuckDB (via `appender`) and Bolt (via `register_table`).
fn load_table(conn: &duckdb::Connection, engine: &mut craton_bolt::Engine, t: &TableSpec<'_>) {
    let ddl = format!("CREATE TABLE {} ({});", t.name, t.ddl_cols);
    conn.execute_batch(&ddl).expect("duckdb create table");
    {
        let mut app = conn.appender(t.name).expect("duckdb appender");
        let n = t.batch.num_rows();
        let cols = (0..t.batch.num_columns())
            .map(|c| t.batch.column(c).clone())
            .collect::<Vec<_>>();
        for i in 0..n {
            append_row(&mut app, &cols, i);
        }
        app.flush().expect("duckdb appender flush");
    }
    engine
        .register_table(t.name, t.batch.clone())
        .expect("bolt register_table");
}

/// Append row `i` from a list of Arrow columns into a DuckDB appender.
///
/// We build a `Vec<Box<dyn ToSql>>` so heterogeneous columns can share one
/// `append_row` call. Each cell is read as a `Cell`, then handed to DuckDB
/// via the appropriate `ToSql` impl (`Option<T>` carries null-ness).
fn append_row(app: &mut duckdb::Appender<'_>, cols: &[arrow_array::ArrayRef], i: usize) {
    use duckdb::types::ToSql;
    let mut boxed: Vec<Box<dyn ToSql>> = Vec::with_capacity(cols.len());
    for c in cols {
        let cell = arrow_cell(c.as_ref(), i);
        boxed.push(cell_to_tosql(cell));
    }
    let refs: Vec<&dyn ToSql> = boxed.iter().map(|b| &**b).collect();
    app.append_row(duckdb::appender_params_from_iter(refs.iter().copied()))
        .expect("duckdb appender append_row");
}

fn cell_to_tosql(c: Cell) -> Box<dyn duckdb::types::ToSql> {
    match c {
        Cell::Null => Box::new(Option::<i64>::None),
        Cell::Bool(b) => Box::new(b),
        Cell::Int(i) => Box::new(i),
        Cell::Float(f) => Box::new(f),
        Cell::Str(s) => Box::new(s),
    }
}

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

/// Single-table fixture `t(k: i32, x: i32, v: f64, a: f64, b: f64, c: f64)`
/// with `n` rows. Deterministic so DuckDB and Bolt see byte-identical input
/// across runs.
fn t_no_nulls(n: usize) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("x", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Float64, false),
        ArrowField::new("a", ArrowDataType::Float64, false),
        ArrowField::new("b", ArrowDataType::Float64, false),
        ArrowField::new("c", ArrowDataType::Float64, false),
    ]));
    let k: Int32Array = (0..n as i32).map(|i| i % 8).collect();
    let x: Int32Array = (0..n as i32).map(|i| (i * 7) % 17).collect();
    let v: Float64Array = (0..n).map(|i| (i as f64).sin() * 10.0).collect();
    let a: Float64Array = (0..n).map(|i| (i % 13) as f64).collect();
    let b: Float64Array = (0..n).map(|i| (i % 5) as f64 * 0.5).collect();
    let c: Float64Array = (0..n).map(|i| (i % 11) as f64 - 4.0).collect();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(k),
            Arc::new(x),
            Arc::new(v),
            Arc::new(a),
            Arc::new(b),
            Arc::new(c),
        ],
    )
    .expect("t_no_nulls batch")
}

/// Variant with frequent NULLs in `v`. Used by the C2 (NULL-coerced-to-0)
/// regression case — if a min/max/avg/count path silently treats nulls as
/// zero, the DuckDB oracle (which discards nulls per the SQL spec) will
/// diverge from Bolt and the test trips.
fn t_with_nulls(n: usize) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Float64, true),
    ]));
    let k: Int32Array = (0..n as i32).map(|i| i % 4).collect();
    // Every 3rd row is NULL; the remaining values span a wide range so
    // MIN/MAX/AVG diverge from "treat null as 0" by a large margin.
    let v: Float64Array = (0..n)
        .map(|i| if i % 3 == 0 { None } else { Some((i as f64) - 50.0) })
        .collect();
    RecordBatch::try_new(schema, vec![Arc::new(k), Arc::new(v)]).expect("t_with_nulls batch")
}

/// High-volume fixture for the DISTINCT regression. We pack two i32 columns
/// so that:
///   - the cardinality of `(a, b)` is moderate (~256),
///   - many rows collide on naive hashes of either column alone, forcing
///     a correctly-implemented DISTINCT to use BOTH columns when keying.
/// A bug like C3 (hash collision treated as equality) shows up as Bolt
/// returning fewer unique rows than DuckDB.
fn t_distinct(n: usize) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("a", ArrowDataType::Int32, false),
        ArrowField::new("b", ArrowDataType::Int32, false),
    ]));
    let a: Int32Array = (0..n as i32).map(|i| i % 16).collect();
    let b: Int32Array = (0..n as i32).map(|i| (i / 16) % 16).collect();
    RecordBatch::try_new(schema, vec![Arc::new(a), Arc::new(b)]).expect("t_distinct batch")
}

/// Single-table fixture `t(s: Utf8, v: i64)` for the Utf8/dictionary diff
/// cases. The `s` column draws from a small vocabulary so the engine's
/// dictionary registry sees a dense dict and the string-literal rewriter
/// has the values it needs to fold `WHERE s = 'foo'` into integer
/// equality on `__idx_s`. `v` carries enough numeric variety to exercise
/// the compound-predicate test (`AND v > 10`).
fn t_strings(n: usize) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("s", ArrowDataType::Utf8, false),
        ArrowField::new("v", ArrowDataType::Int64, false),
    ]));
    // 4-way vocabulary, cycled deterministically. Includes "foo" so the
    // value-present case has hits; "qux" is reserved for the absent test.
    let vocab = ["foo", "bar", "baz", "abc"];
    let s_vals: Vec<&str> = (0..n).map(|i| vocab[i % vocab.len()]).collect();
    let s = StringArray::from(s_vals);
    let v: Int64Array = (0..n as i64).map(|i| (i * 3) % 31).collect();
    RecordBatch::try_new(schema, vec![Arc::new(s), Arc::new(v)])
        .expect("t_strings batch")
}

/// Two-table Utf8 join fixture: `t1(s)` and `t2(s)` with overlapping
/// vocabularies. The intersection (`foo`, `bar`) drives the inner-join
/// hit rate; `only1` / `only2` ensure each side has rows that should
/// NOT survive the join, so a degenerate "return everything" bug shows
/// up as a row-count mismatch against DuckDB.
fn t_strings_join_fixture(n: usize) -> (RecordBatch, RecordBatch) {
    let s1_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        ArrowDataType::Utf8,
        false,
    )]));
    let s2_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        ArrowDataType::Utf8,
        false,
    )]));
    let vocab1 = ["foo", "bar", "only1", "foo"];
    let vocab2 = ["foo", "only2", "bar", "bar"];
    let s1_vals: Vec<&str> = (0..n).map(|i| vocab1[i % vocab1.len()]).collect();
    let s2_vals: Vec<&str> = (0..n).map(|i| vocab2[i % vocab2.len()]).collect();
    let s1 = StringArray::from(s1_vals);
    let s2 = StringArray::from(s2_vals);
    let t1 = RecordBatch::try_new(s1_schema, vec![Arc::new(s1)]).expect("t1 strings batch");
    let t2 = RecordBatch::try_new(s2_schema, vec![Arc::new(s2)]).expect("t2 strings batch");
    (t1, t2)
}

/// Build-the-join-fixture: two tables `t1(k, v)` and `t2(k, w)` with a
/// medium-overlap key space.
fn join_fixture(n: usize) -> (RecordBatch, RecordBatch) {
    let s1 = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Float64, false),
    ]));
    let s2 = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("w", ArrowDataType::Float64, false),
    ]));
    let k1: Int32Array = (0..n as i32).map(|i| i % 16).collect();
    let v: Float64Array = (0..n).map(|i| (i as f64) * 1.5).collect();
    let k2: Int32Array = (0..n as i32).map(|i| (i * 3) % 16).collect();
    let w: Float64Array = (0..n).map(|i| (i as f64) - 2.5).collect();
    let t1 = RecordBatch::try_new(s1, vec![Arc::new(k1), Arc::new(v)]).expect("t1 batch");
    let t2 = RecordBatch::try_new(s2, vec![Arc::new(k2), Arc::new(w)]).expect("t2 batch");
    (t1, t2)
}

// ---------------------------------------------------------------------------
// The eight curated diff cases
// ---------------------------------------------------------------------------
//
// Every test is `#[ignore]`'d because `Engine::new` opens a CUDA context.
// Run on a GPU host with:
//
//     cargo test --test diff_duckdb -- --ignored

/// Case 1 — basic filter pass-through. No aggregation, no grouping; sanity
/// check that projection + WHERE agree across engines.
#[test]
#[ignore = "gpu:e2e"]
fn diff_filter_basic() {
    let batch = t_no_nulls(1024);
    diff_query(
        "diff_filter_basic",
        "SELECT k, x, v FROM t WHERE x > 5",
        &[TableSpec {
            name: "t",
            ddl_cols:
                "k INTEGER NOT NULL, x INTEGER NOT NULL, v DOUBLE NOT NULL, \
                 a DOUBLE NOT NULL, b DOUBLE NOT NULL, c DOUBLE NOT NULL",
            batch,
        }],
    );
}

/// Case 2 — single-key SUM. The most common groupby shape; if this fails,
/// nothing harder will pass.
#[test]
#[ignore = "gpu:e2e"]
fn diff_groupby_basic_sum() {
    let batch = t_no_nulls(1024);
    diff_query(
        "diff_groupby_basic_sum",
        "SELECT k, SUM(v) FROM t GROUP BY k",
        &[TableSpec {
            name: "t",
            ddl_cols:
                "k INTEGER NOT NULL, x INTEGER NOT NULL, v DOUBLE NOT NULL, \
                 a DOUBLE NOT NULL, b DOUBLE NOT NULL, c DOUBLE NOT NULL",
            batch,
        }],
    );
}

/// Case 3 — **C1 regression**: GROUP BY with HAVING. A bug that drops the
/// HAVING filter (a real Bolt regression) makes the engine return ALL
/// groups; DuckDB returns only those whose `SUM(v) > 100`. The differential
/// check trips on the row-count mismatch.
#[test]
#[ignore = "gpu:e2e"]
fn diff_groupby_having_sum() {
    // Use an all-positive value column so HAVING > 100 has a meaningful
    // signal (otherwise sin() can sum near zero per group and the filter
    // would drop everything regardless).
    let n = 2048;
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Float64, false),
    ]));
    let k: Int32Array = (0..n as i32).map(|i| i % 8).collect();
    let v: Float64Array = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(k), Arc::new(v)]).expect("having batch");
    diff_query(
        "diff_groupby_having_sum",
        "SELECT k, SUM(v) FROM t GROUP BY k HAVING SUM(v) > 100",
        &[TableSpec {
            name: "t",
            ddl_cols: "k INTEGER NOT NULL, v DOUBLE NOT NULL",
            batch,
        }],
    );
}

/// Case 4 — **C2 regression**: scalar aggregates over a column with NULLs.
/// SQL semantics: nulls are excluded from MIN/MAX/AVG/COUNT(v). A bug that
/// coerces NULL→0 inflates COUNT, drags MIN toward 0, drags MAX from
/// negative-side, and shifts AVG. DuckDB applies the spec correctly; any
/// drift from Bolt fires the assert with a clean per-cell diff.
#[test]
#[ignore = "gpu:e2e"]
fn diff_agg_nulls_min_max_avg_count() {
    let batch = t_with_nulls(1024);
    // Wrap in a single-row scalar-agg shape: no GROUP BY, so the result
    // is exactly one row of (min, max, avg, count).
    diff_query(
        "diff_agg_nulls_min_max_avg_count",
        "SELECT MIN(v), MAX(v), AVG(v), COUNT(v) FROM t",
        &[TableSpec {
            name: "t",
            // `v DOUBLE` (nullable) — no `NOT NULL` constraint.
            ddl_cols: "k INTEGER NOT NULL, v DOUBLE",
            batch,
        }],
    );
}

/// Case 5 — **C3 regression**: DISTINCT over a high-volume two-column input.
/// A naive hash on a single column would let columns collide; the fixture
/// is constructed so that BOTH columns must contribute to the key for the
/// correct cardinality (~256 distinct pairs across 16384 rows). A
/// hash-collision bug shrinks Bolt's row count below DuckDB's.
///
/// Compromise: we don't have a knob to inject *specific* hash collisions
/// without internal-API access, so we substitute high-volume DISTINCT
/// with deliberately overlapping per-column value ranges. That still
/// exercises the multi-column key path and would catch C3.
#[test]
#[ignore = "gpu:e2e"]
fn diff_distinct_high_volume() {
    let batch = t_distinct(16_384);
    diff_query(
        "diff_distinct_high_volume",
        "SELECT DISTINCT a, b FROM t",
        &[TableSpec {
            name: "t",
            ddl_cols: "a INTEGER NOT NULL, b INTEGER NOT NULL",
            batch,
        }],
    );
}

/// Case 6 — GROUP BY ... ORDER BY ... LIMIT. Order-sensitive: we use
/// [`diff_query_ordered`] so row positions are checked. Validates that
/// post-aggregation pipelines (sort + limit) are wired to the same
/// rows + values DuckDB produces.
#[test]
#[ignore = "gpu:e2e"]
fn diff_groupby_order_limit() {
    let batch = t_no_nulls(2048);
    diff_query_ordered(
        "diff_groupby_order_limit",
        "SELECT k, SUM(v) FROM t GROUP BY k ORDER BY k LIMIT 10",
        &[TableSpec {
            name: "t",
            ddl_cols:
                "k INTEGER NOT NULL, x INTEGER NOT NULL, v DOUBLE NOT NULL, \
                 a DOUBLE NOT NULL, b DOUBLE NOT NULL, c DOUBLE NOT NULL",
            batch,
        }],
    );
}

/// Case 7 — INNER JOIN across two registered tables. Exercises the
/// two-table fixture loader and validates the join-output schema matches
/// DuckDB column-for-column.
#[test]
#[ignore = "gpu:e2e"]
fn diff_inner_join() {
    let (t1, t2) = join_fixture(512);
    diff_query(
        "diff_inner_join",
        "SELECT t1.k, t1.v, t2.w FROM t1 INNER JOIN t2 ON t1.k = t2.k",
        &[
            TableSpec {
                name: "t1",
                ddl_cols: "k INTEGER NOT NULL, v DOUBLE NOT NULL",
                batch: t1,
            },
            TableSpec {
                name: "t2",
                ddl_cols: "k INTEGER NOT NULL, w DOUBLE NOT NULL",
                batch: t2,
            },
        ],
    );
}

/// Case 8 — **C5 regression**: GROUP BY with multiple SUMs. The multi-SUM
/// orchestrator's column-alignment bug (see `tests/tier2_multi_sum_e2e.rs`)
/// would scatter aggregates into the wrong output column; DuckDB pins the
/// canonical mapping. Any per-cell drift in a SUM(a) / SUM(b) / SUM(c)
/// triplet trips this case.
#[test]
#[ignore = "gpu:e2e"]
fn diff_groupby_multi_sum() {
    let batch = t_no_nulls(2048);
    diff_query(
        "diff_groupby_multi_sum",
        "SELECT k, SUM(a), SUM(b), SUM(c) FROM t GROUP BY k",
        &[TableSpec {
            name: "t",
            ddl_cols:
                "k INTEGER NOT NULL, x INTEGER NOT NULL, v DOUBLE NOT NULL, \
                 a DOUBLE NOT NULL, b DOUBLE NOT NULL, c DOUBLE NOT NULL",
            batch,
        }],
    );
}

// ---------------------------------------------------------------------------
// Utf8 / dictionary differential cases (review H8)
// ---------------------------------------------------------------------------
//
// The eight cases above cover only numeric + bool dtypes. Bolt's string
// path — dictionary registry (`src/exec/dict_registry.rs`), Utf8 GPU
// dictionaries (`src/cuda/dictionary*.rs`), and the string-literal
// rewriter (`src/plan/string_literal_rewrite.rs`) — had ZERO end-to-end
// coverage against DuckDB. The cases below close that gap for every
// Utf8/dict SQL surface that's currently wired end-to-end in Bolt:
//
//   * `WHERE s = 'lit'` and the constant-folded "literal not in dict"
//     short-circuit (the rewriter emits `Bool(false)`).
//   * Inner JOIN on a Utf8 key (gpu_join's `SingleUtf8` shape).
//   * DISTINCT on a Utf8 column (host-side dedup with `RowKeyValue::Utf8`).
//   * Compound predicate that mixes string equality with a numeric range
//     — exercises the rewriter recursing through `AND`.
//
// NOT covered here: `GROUP BY <Utf8 column>`. The executor currently
// rejects with `"Utf8 GROUP BY keys not yet supported"` (see
// `src/exec/groupby_with_pre.rs` and `src/exec/groupby.rs`), so a diff
// test would panic on the Bolt side rather than exercise the comparison
// path. The H8 review's "GROUP BY string" slot is substituted with the
// `s <> 'lit'` (inequality) case below — same rewriter, opposite
// constant-fold branch — so the dictionary path still gets two-shape
// coverage. Promote the GROUP BY case here the moment that error lifts.
//
// Gating: `#[ignore = "gpu:string"]` per the H8 review ask. Run on
// a GPU host with:
//
//     cargo test --test diff_duckdb -- --ignored

/// Case 9 — `WHERE s = 'lit'` against a literal that IS in the dictionary.
/// Round-trips the rewriter's hot path: the predicate folds to
/// `__idx_s = <dict_idx_of_foo>`, the GPU runs integer eq, and the host
/// gathers exactly the `s = 'foo'` rows. Any drift from DuckDB here
/// indicates either a dictionary-index bug (wrong index for "foo") or a
/// gather-compaction bug.
#[test]
#[ignore = "gpu:string"]
fn diff_string_equality_value_present() {
    let batch = t_strings(1024);
    diff_query(
        "diff_string_equality_value_present",
        "SELECT s FROM t WHERE s = 'foo'",
        &[TableSpec {
            name: "t",
            ddl_cols: "s VARCHAR NOT NULL, v BIGINT NOT NULL",
            batch,
        }],
    );
}

/// Case 10 — `WHERE s = 'lit'` against a literal that is NOT in the
/// dictionary. `string_literal_rewrite` constant-folds this to
/// `Bool(false)`, so Bolt should return zero rows — matching DuckDB,
/// which evaluates the equality and gets no hits. Catches a regression
/// where the fold either fires too eagerly (returns 0 when matches
/// exist) or fails to fire (returns garbage indices).
#[test]
#[ignore = "gpu:string"]
fn diff_string_equality_value_absent() {
    // "qux" is not present in `t_strings`'s 4-word vocabulary, so the
    // rewriter must fold the predicate to Bool(false) and surface zero
    // rows on both engines.
    let batch = t_strings(1024);
    diff_query(
        "diff_string_equality_value_absent",
        "SELECT s FROM t WHERE s = 'qux'",
        &[TableSpec {
            name: "t",
            ddl_cols: "s VARCHAR NOT NULL, v BIGINT NOT NULL",
            batch,
        }],
    );
}

/// Case 10b — `WHERE s <> 'lit'` round-trip. The rewriter folds NotEq
/// the same way it folds Eq: when the literal is in the dictionary it
/// rewrites to integer `<>` on `__idx_s`; when it isn't, it
/// constant-folds to `Bool(true)` (the complement of the absent-case
/// fold above). This case picks the in-dict literal so the rewriter
/// emits a real GPU predicate — the constant-fold absent twin is
/// covered by `diff_string_equality_value_absent`'s `Bool(false)`
/// branch, which is the mirror image. Catches a bug that asymmetric-
/// ally handles the two ops (e.g. swaps Eq for NotEq when folding).
#[test]
#[ignore = "gpu:string"]
fn diff_string_inequality_value_present() {
    let batch = t_strings(1024);
    diff_query(
        "diff_string_inequality_value_present",
        "SELECT s FROM t WHERE s <> 'foo'",
        &[TableSpec {
            name: "t",
            ddl_cols: "s VARCHAR NOT NULL, v BIGINT NOT NULL",
            batch,
        }],
    );
}

/// Case 11 — INNER JOIN on a Utf8 key. Exercises `KeyShape::SingleUtf8`
/// in `src/exec/gpu_join.rs`: both sides string-intern their keys to
/// i32 dictionary indices, the GPU joins on the indices, and the host
/// re-materialises the original strings on the output side. A bug in
/// the interner (e.g. independent dictionaries for the two sides) would
/// produce false negatives — DuckDB's row count gives the canonical
/// answer.
#[test]
#[ignore = "gpu:string"]
fn diff_string_join_inner() {
    let (t1, t2) = t_strings_join_fixture(256);
    diff_query(
        "diff_string_join_inner",
        "SELECT t1.s FROM t1 INNER JOIN t2 ON t1.s = t2.s",
        &[
            TableSpec {
                name: "t1",
                ddl_cols: "s VARCHAR NOT NULL",
                batch: t1,
            },
            TableSpec {
                name: "t2",
                ddl_cols: "s VARCHAR NOT NULL",
                batch: t2,
            },
        ],
    );
}

/// Case 12 — `SELECT DISTINCT s FROM t ORDER BY s`. Exercises the
/// host-side DISTINCT path's `RowKeyValue::Utf8` arm
/// (`src/exec/distinct.rs`) plus the Utf8 ORDER BY (which goes through
/// the GPU sort's inline-dictionary builder, see `src/exec/gpu_sort.rs`).
/// Order is meaningful here — we route through `diff_query_ordered` so
/// row positions are checked.
#[test]
#[ignore = "gpu:string"]
fn diff_string_distinct() {
    let batch = t_strings(1024);
    diff_query_ordered(
        "diff_string_distinct",
        "SELECT DISTINCT s FROM t ORDER BY s",
        &[TableSpec {
            name: "t",
            ddl_cols: "s VARCHAR NOT NULL, v BIGINT NOT NULL",
            batch,
        }],
    );
}

/// Case 13 — compound predicate mixing string equality and a numeric
/// range. The rewriter walks post-order through the `AND` node and
/// rewrites only the string-eq leg; the numeric leg passes through to
/// the generic scan kernel. A bug that drops one half of the conjunct
/// (e.g. the rewriter stomping over the other side) shows up as a row
/// count or per-row value mismatch.
#[test]
#[ignore = "gpu:string"]
fn diff_string_compound_predicate() {
    let batch = t_strings(1024);
    diff_query(
        "diff_string_compound_predicate",
        "SELECT s, v FROM t WHERE s = 'foo' AND v > 10",
        &[TableSpec {
            name: "t",
            ddl_cols: "s VARCHAR NOT NULL, v BIGINT NOT NULL",
            batch,
        }],
    );
}
