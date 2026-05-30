// SPDX-License-Identifier: Apache-2.0

//! Differential harness **extension**: Craton Bolt vs DuckDB for the
//! semantics gaps called out in `reviews/tests.md` (items 1–4):
//!
//!   1. String semantics — `LENGTH` (characters), `OCTET_LENGTH` (bytes),
//!      multibyte `SUBSTRING`, `LENGTH(NULL) = NULL`.
//!   2. `NOT IN (subquery)` with / without a NULL in the set.
//!   3. Two-key `COUNT(col)` over a column with NULLs.
//!   4. Grouped float `MIN`/`MAX` including NaN (DuckDB orders NaN as the
//!      largest float).
//!
//! This is a sibling of `tests/diff_duckdb.rs` and follows the same design:
//! load identical Arrow fixtures into BOTH engines, run identical SQL, decode
//! into a normalised `Cell` representation, canonicalise row order (unless the
//! query is order-sensitive), and assert agreement with a null-aware,
//! tolerance-based comparison. On mismatch we dump BOTH result sets.
//!
//! Each integration test under `tests/` is its own crate, so the minimal
//! harness machinery (`Cell`, decoders, drivers) is re-declared locally rather
//! than shared from `diff_duckdb.rs`.
//!
//! Gating mirrors `diff_duckdb.rs`: every test is `#[ignore]`'d because
//! `Engine::new()` opens a CUDA context. Run on a GPU host with:
//!
//! ```text
//! cargo test --test diff_duckdb_semantics -- --ignored
//! ```

use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

mod common;
use common::REL_TOL;

// ---------------------------------------------------------------------------
// Normalised cell representation + null-aware tolerance comparison
// (mirrors tests/diff_duckdb.rs).
// ---------------------------------------------------------------------------

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
    // NaN == NaN here so a grouped MIN/MAX that legitimately returns NaN
    // (DuckDB orders NaN as the largest float) compares equal across engines.
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    let mag = a.abs().max(b.abs()).max(1.0);
    diff / mag <= REL_TOL
}

#[derive(Debug, Clone)]
struct ResultSet {
    columns: Vec<String>,
    rows: Vec<Vec<Cell>>,
}

impl ResultSet {
    fn canonicalise(mut self) -> Self {
        self.rows.sort_by(|a, b| {
            for (x, y) in a.iter().zip(b.iter()) {
                match x.to_string().cmp(&y.to_string()) {
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
    let column_names: Vec<String> = stmt.column_names();
    let ncols = column_names.len();
    let mut rows_iter = stmt.query([]).expect("duckdb query");
    let mut rows: Vec<Vec<Cell>> = Vec::new();
    while let Some(row) = rows_iter.next().expect("duckdb row") {
        let mut out = Vec::with_capacity(ncols);
        for i in 0..ncols {
            out.push(duckdb_value_to_cell(row.get_ref(i).expect("duckdb cell")));
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
        other => panic!("diff_duckdb_semantics: unsupported DuckDB value type {other:?}"),
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
        "diff_duckdb_semantics: unsupported Arrow dtype {:?} (extend arrow_cell)",
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
                errs.push(format!("row {i}, col {c}: DuckDB={} Bolt={}", dr[c], br[c]));
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
// Drivers + table loader (mirrors diff_duckdb.rs)
// ---------------------------------------------------------------------------

struct TableSpec<'a> {
    name: &'a str,
    ddl_cols: &'a str,
    batch: RecordBatch,
}

fn setup(label: &str, tables: &[TableSpec<'_>]) -> (duckdb::Connection, craton_bolt::Engine) {
    let mut engine = craton_bolt::Engine::new().unwrap_or_else(|e| {
        panic!("[{label}] CUDA init failed (need a GPU to run --ignored tests): {e}")
    });
    let conn = duckdb::Connection::open_in_memory().expect("duckdb open");
    for t in tables {
        load_table(&conn, &mut engine, t);
    }
    (conn, engine)
}

fn diff_query(label: &str, sql: &str, tables: &[TableSpec<'_>]) {
    let (conn, engine) = setup(label, tables);
    let duck = duckdb_run(&conn, sql).canonicalise();
    let bolt = bolt_run(&engine, sql).canonicalise();
    assert_results_equal(label, &duck, &bolt);
}

#[allow(dead_code)]
fn diff_query_ordered(label: &str, sql: &str, tables: &[TableSpec<'_>]) {
    let (conn, engine) = setup(label, tables);
    let duck = duckdb_run(&conn, sql);
    let bolt = bolt_run(&engine, sql);
    assert_results_equal(label, &duck, &bolt);
}

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

fn append_row(app: &mut duckdb::Appender<'_>, cols: &[arrow_array::ArrayRef], i: usize) {
    use duckdb::types::ToSql;
    let mut boxed: Vec<Box<dyn ToSql>> = Vec::with_capacity(cols.len());
    for c in cols {
        boxed.push(cell_to_tosql(arrow_cell(c.as_ref(), i)));
    }
    let refs: Vec<&dyn ToSql> = boxed.iter().map(|b| &**b).collect();
    app.append_row(duckdb::appender_params_from_iter(refs.iter().copied()))
        .expect("duckdb appender append_row");
}

fn cell_to_tosql(c: Cell) -> Box<dyn duckdb::types::ToSql> {
    match c {
        // NULL must be typed so the appender binds the right column type.
        // We only ever null the f64 / Utf8 / i32 columns in these fixtures;
        // Option::<i64>::None binds a typed SQL NULL that DuckDB coerces to
        // the destination column type, matching diff_duckdb.rs.
        Cell::Null => Box::new(Option::<i64>::None),
        Cell::Bool(b) => Box::new(b),
        Cell::Int(i) => Box::new(i),
        Cell::Float(f) => Box::new(f),
        Cell::Str(s) => Box::new(s),
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// `t(s VARCHAR, id INTEGER)` with deterministic multibyte strings. `id` lets
/// us pin row identity for the SUBSTRING per-row checks without relying on
/// scan order.
fn t_unicode_strings() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("s", ArrowDataType::Utf8, true),
        ArrowField::new("id", ArrowDataType::Int32, false),
    ]));
    let s = StringArray::from(vec![
        Some("héllo"),
        Some("世界x"),
        Some("ascii"),
        Some(""),
        None,
    ]);
    let id = Int32Array::from(vec![0, 1, 2, 3, 4]);
    RecordBatch::try_new(schema, vec![Arc::new(s), Arc::new(id)]).expect("t_unicode_strings")
}

/// `t(k)` probe + `other(id)` set, both nullable Int32, for the NOT-IN cases.
fn not_in_tables(probe: Vec<Option<i32>>, set: Vec<Option<i32>>) -> Vec<TableSpec<'static>> {
    let t_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "k",
        ArrowDataType::Int32,
        true,
    )]));
    let t = RecordBatch::try_new(t_schema, vec![Arc::new(Int32Array::from(probe))])
        .expect("t batch");
    let o_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "id",
        ArrowDataType::Int32,
        true,
    )]));
    let o = RecordBatch::try_new(o_schema, vec![Arc::new(Int32Array::from(set))])
        .expect("other batch");
    vec![
        TableSpec {
            name: "t",
            ddl_cols: "k INTEGER",
            batch: t,
        },
        TableSpec {
            name: "other",
            ddl_cols: "id INTEGER",
            batch: o,
        },
    ]
}

// ===========================================================================
// Item 1 — string semantics oracle cases
// ===========================================================================

/// `LENGTH` (character count) and `OCTET_LENGTH` (byte count) over multibyte
/// input, including the empty string and a NULL row. DuckDB is the oracle for
/// both the char/byte distinction and `LENGTH(NULL) = NULL`.
#[test]
#[ignore = "gpu:string"]
fn diff_length_and_octet_length_unicode() {
    diff_query(
        "diff_length_and_octet_length_unicode",
        "SELECT id, LENGTH(s), OCTET_LENGTH(s) FROM t",
        &[TableSpec {
            name: "t",
            ddl_cols: "s VARCHAR, id INTEGER NOT NULL",
            batch: t_unicode_strings(),
        }],
    );
}

/// Multibyte `SUBSTRING` is character-indexed and must not leak the byte to the
/// left of the start position. We pin specific rows by `id` so the oracle and
/// Bolt compare the same logical strings.
#[test]
#[ignore = "gpu:string"]
fn diff_substring_unicode_character_indexed() {
    diff_query(
        "diff_substring_unicode_character_indexed",
        // id 0 = "héllo": SUBSTRING(.,1,3) → "hél"; id 1 = "世界x":
        // SUBSTRING(.,2,2) → "界x". Both exercise no-left-of-start leak.
        "SELECT id, SUBSTRING(s, 1, 3), SUBSTRING(s, 2, 2) FROM t WHERE id IN (0, 1)",
        &[TableSpec {
            name: "t",
            ddl_cols: "s VARCHAR, id INTEGER NOT NULL",
            batch: t_unicode_strings(),
        }],
    );
}

// ===========================================================================
// Item 2 — NOT IN (subquery) with NULL in the set oracle cases
// ===========================================================================

/// `k NOT IN (SELECT id FROM other)` where the set has a NULL → zero rows on
/// both engines (strict SQL 3VL). DuckDB is the canonical oracle.
#[test]
#[ignore = "gpu:e2e"]
fn diff_not_in_subquery_null_in_set() {
    diff_query(
        "diff_not_in_subquery_null_in_set",
        "SELECT k FROM t WHERE k NOT IN (SELECT id FROM other)",
        &not_in_tables(
            vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
            vec![Some(1), Some(2), None],
        ),
    );
}

/// Control: NULL-free set → the normal complement {3, 4, 5}. Same harness,
/// opposite branch of the F-6 fold, oracle-checked against DuckDB.
#[test]
#[ignore = "gpu:e2e"]
fn diff_not_in_subquery_no_null() {
    diff_query(
        "diff_not_in_subquery_no_null",
        "SELECT k FROM t WHERE k NOT IN (SELECT id FROM other)",
        &not_in_tables(
            vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
            vec![Some(1), Some(2)],
        ),
    );
}

// ===========================================================================
// Item 3 — two-key COUNT(col) with NULLs oracle case
// ===========================================================================

/// `SELECT k1, k2, COUNT(v) FROM t GROUP BY k1, k2` where `v` has NULLs.
/// COUNT(v) excludes NULLs; the multi-key grouping result is compared against
/// DuckDB (order-independent → canonicalised).
#[test]
#[ignore = "gpu:tier1"]
fn diff_two_key_count_with_nulls() {
    let n = 8usize;
    let k1: Int32Array = (0..n as i32).map(|i| i / 4).collect();
    let k2: Int32Array = (0..n as i32).map(|i| (i / 2) % 2).collect();
    let v: Int64Array = (0..n as i64)
        .map(|i| if i % 2 == 1 { None } else { Some(i) })
        .collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k1", ArrowDataType::Int32, false),
        ArrowField::new("k2", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Int64, true),
    ]));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(k1), Arc::new(k2), Arc::new(v)])
            .expect("two-key count batch");
    diff_query(
        "diff_two_key_count_with_nulls",
        "SELECT k1, k2, COUNT(v) FROM t GROUP BY k1, k2",
        &[TableSpec {
            name: "t",
            ddl_cols: "k1 INTEGER NOT NULL, k2 INTEGER NOT NULL, v BIGINT",
            batch,
        }],
    );
}

// ===========================================================================
// Item 4 — grouped float MIN/MAX including NaN oracle case
// ===========================================================================

/// Grouped `MIN(v)` / `MAX(v)` where one group contains a NaN. DuckDB orders
/// NaN as the LARGEST float, so the NaN group's MAX is NaN and its MIN ignores
/// the NaN. `close_enough` treats NaN == NaN, so the per-cell compare lines up
/// exactly with DuckDB.
#[test]
#[ignore = "gpu:tier1"]
fn diff_grouped_min_max_with_nan() {
    // group 0: {1.0, NaN, 2.0}; group 1: {-3.0, 0.5}.
    let k: Int32Array = Int32Array::from(vec![0, 0, 0, 1, 1]);
    let v: Float64Array = Float64Array::from(vec![1.0, f64::NAN, 2.0, -3.0, 0.5]);
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Float64, false),
    ]));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(k), Arc::new(v)]).expect("nan batch");
    diff_query(
        "diff_grouped_min_max_with_nan",
        "SELECT k, MIN(v), MAX(v) FROM t GROUP BY k",
        &[TableSpec {
            name: "t",
            ddl_cols: "k INTEGER NOT NULL, v DOUBLE NOT NULL",
            batch,
        }],
    );
}
