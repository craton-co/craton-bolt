// SPDX-License-Identifier: Apache-2.0

//! DuckDB-diff regression tests — lock in semantic conventions that
//! Craton Bolt must match against the reference engine.
//!
//! Today's coverage (review C12):
//!   * Float canonicalisation: `+0.0` and `-0.0` belong to the same
//!     equivalence class for `SELECT DISTINCT` and for equi-`JOIN`. The
//!     same convention applies to GROUP BY (covered by the unit-level
//!     tests in `src/exec/groupby*.rs`); the e2e diff here pins it
//!     across the SQL frontend down to the executor.
//!   * NaN behaviour: NaN bit patterns are LEFT AS-IS by the host-side
//!     canonicalisation. We do not pin a single NaN-vs-NaN convention
//!     for DISTINCT here because DuckDB's stance has shifted between
//!     versions; the diff itself is what we want to lock in. If a
//!     future bump regresses, the assertion below will surface it.
//!
//! All tests in this file are gated `#[ignore = "requires CUDA"]`. Run
//! with `cargo test --test diff_duckdb -- --ignored`.

use std::sync::Arc;

use arrow_array::{Array, Float64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::Engine;

// ---- Helpers ---------------------------------------------------------------

/// Build a one-column Float64 RecordBatch named `x` from `vals`.
fn float64_batch(vals: Vec<f64>) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "x",
        ArrowDataType::Float64,
        false,
    )]));
    let arr: Arc<dyn Array> = Arc::new(Float64Array::from(vals));
    RecordBatch::try_new(schema, vec![arr]).expect("one-col float64 batch")
}

/// Sort an `f64` slice with NaNs treated as the largest element, so two
/// runs containing NaN compare equal under exact-bit equality.
fn sort_f64_total(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| {
        if a.is_nan() && b.is_nan() {
            std::cmp::Ordering::Equal
        } else if a.is_nan() {
            std::cmp::Ordering::Greater
        } else if b.is_nan() {
            std::cmp::Ordering::Less
        } else {
            a.partial_cmp(b).unwrap()
        }
    });
    v
}

/// Bit-equal comparison of two `f64` runs after total-ordering. Treats
/// every NaN as equal to every other NaN (sufficient for the diff —
/// DuckDB and Bolt agree NaNs survive DISTINCT; we don't care about the
/// specific bit pattern).
fn assert_f64_equal_modulo_nan(actual: &[f64], expected: &[f64], ctx: &str) {
    let a = sort_f64_total(actual.to_vec());
    let e = sort_f64_total(expected.to_vec());
    assert_eq!(
        a.len(),
        e.len(),
        "{ctx}: row counts differ — Bolt={a:?}, DuckDB={e:?}"
    );
    for (i, (ai, ei)) in a.iter().zip(e.iter()).enumerate() {
        let eq = (ai.is_nan() && ei.is_nan()) || ai.to_bits() == ei.to_bits();
        assert!(
            eq,
            "{ctx}: row {i} differs — Bolt={ai:?} (bits={:#x}), DuckDB={ei:?} (bits={:#x})",
            ai.to_bits(),
            ei.to_bits()
        );
    }
}

/// Run the same SQL against DuckDB on a single-column Float64 table `t`
/// (or `t1`/`t2` for the join shape) and return the result column.
fn duckdb_run_single_col_f64(sql: &str, tables: &[(&str, &[f64])]) -> Vec<f64> {
    let conn = duckdb::Connection::open_in_memory().expect("duckdb open");
    for (name, vals) in tables {
        let create = format!("CREATE TABLE {name} (x DOUBLE);");
        conn.execute_batch(&create).expect("create");
        let mut app = conn.appender(name).expect("appender");
        for v in *vals {
            app.append_row(duckdb::params![*v]).expect("append");
        }
        app.flush().expect("flush");
    }
    let mut stmt = conn.prepare(sql).expect("prep");
    let mut rows = stmt.query([]).expect("query");
    let mut out = Vec::new();
    while let Some(r) = rows.next().expect("row") {
        let v: f64 = r.get(0).expect("col 0");
        out.push(v);
    }
    out
}

// ---- Tests -----------------------------------------------------------------

/// Review C12: `SELECT DISTINCT x` over `[+0.0, -0.0, 1.5, NaN]` must
/// agree with DuckDB. The contract enforced here is:
///
///   * `+0.0` and `-0.0` collapse to ONE row (per SQL/IEEE; DuckDB does
///     the same).
///   * The remaining `1.5` and `NaN` survive.
///
/// We compare bag-equal (modulo NaN bit pattern, which we do not pin).
#[test]
#[ignore = "requires CUDA"]
fn distinct_signed_zero_matches_duckdb() {
    let vals: Vec<f64> = vec![0.0, -0.0, 1.5, f64::NAN];

    // --- Craton Bolt
    let mut engine = Engine::new().expect("engine");
    engine
        .register_table("t", float64_batch(vals.clone()))
        .expect("register");
    let h = engine.sql("SELECT DISTINCT x FROM t").expect("sql distinct");
    let out = h.record_batch();
    let arr = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64 output");
    let bolt_rows: Vec<f64> = (0..arr.len()).map(|i| arr.value(i)).collect();

    // --- DuckDB reference
    let duck_rows = duckdb_run_single_col_f64("SELECT DISTINCT x FROM t", &[("t", &vals)]);

    // Sanity: Bolt must report `+0.0` and `-0.0` as a single row. This
    // is the headline assertion; if it fails, the canonicalisation
    // regression is exactly what the test is catching.
    let bolt_zeros = bolt_rows.iter().filter(|&&v| v == 0.0).count();
    assert_eq!(
        bolt_zeros, 1,
        "Bolt should collapse +0.0/-0.0 to one row; got {bolt_rows:?}"
    );

    assert_f64_equal_modulo_nan(&bolt_rows, &duck_rows, "DISTINCT");
}

/// Review C12: equi-join on a Float64 key must match `+0.0` against
/// `-0.0` across sides. Pin the expectation against DuckDB.
///
/// Setup: t1 = [+0.0, 1.5], t2 = [-0.0, 1.5]. Both sides have one row
/// per group, so the join produces two rows. We project `t1.x` so the
/// result column has the build-side bit pattern; DuckDB is free to
/// project either side (semantically equal under SQL float equality).
#[test]
#[ignore = "requires CUDA"]
fn join_signed_zero_matches_duckdb() {
    let lhs: Vec<f64> = vec![0.0, 1.5];
    let rhs: Vec<f64> = vec![-0.0, 1.5];

    // --- Craton Bolt
    let mut engine = Engine::new().expect("engine");
    engine
        .register_table("t1", float64_batch(lhs.clone()))
        .expect("register t1");
    engine
        .register_table("t2", float64_batch(rhs.clone()))
        .expect("register t2");
    let h = engine
        .sql("SELECT t1.x FROM t1 JOIN t2 ON t1.x = t2.x")
        .expect("sql join");
    let out = h.record_batch();
    let arr = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64 output");
    let bolt_rows: Vec<f64> = (0..arr.len()).map(|i| arr.value(i)).collect();

    // The +0.0 / -0.0 pair must match: we expect TWO output rows
    // (the +0.0/-0.0 match and the 1.5 match), not just one.
    assert_eq!(
        bolt_rows.len(),
        2,
        "join should match +0.0 against -0.0; got rows = {bolt_rows:?}"
    );

    // --- DuckDB reference
    let duck_rows = duckdb_run_single_col_f64(
        "SELECT t1.x FROM t1 JOIN t2 ON t1.x = t2.x",
        &[("t1", &lhs), ("t2", &rhs)],
    );

    // Float equality across sides means engines are free to surface
    // either operand. Compare bag-equal after total ordering; we treat
    // +0.0 and -0.0 as equal here because the SQL contract under
    // equi-join makes them so.
    let mut a: Vec<f64> = bolt_rows.iter().map(|&v| if v == 0.0 { 0.0 } else { v }).collect();
    let mut e: Vec<f64> = duck_rows.iter().map(|&v| if v == 0.0 { 0.0 } else { v }).collect();
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    e.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert_eq!(a, e, "join output diverges from DuckDB");
}

/// Review C12 (GROUP BY): the third leg of the equivalence-class
/// agreement — `SELECT x, COUNT(*) FROM t GROUP BY x` must collapse
/// `+0.0` and `-0.0` to one group. We assert against DuckDB.
#[test]
#[ignore = "requires CUDA"]
fn groupby_signed_zero_matches_duckdb() {
    let vals: Vec<f64> = vec![0.0, -0.0, -0.0, 1.5];

    // --- Craton Bolt
    let mut engine = Engine::new().expect("engine");
    engine
        .register_table("t", float64_batch(vals.clone()))
        .expect("register");
    let h = engine
        .sql("SELECT x, COUNT(*) AS c FROM t GROUP BY x")
        .expect("sql groupby");
    let out = h.record_batch();
    assert_eq!(
        out.num_rows(),
        2,
        "GROUP BY should produce 2 groups (one for ±0.0 with count=3, one for 1.5 with count=1); \
         got {} rows",
        out.num_rows()
    );

    // --- DuckDB reference: just sanity-check that DuckDB also reports
    // two groups (the headline diff). We don't reach for an apples-to-
    // apples count comparison because DuckDB's COUNT output dtype
    // (UBIGINT) needs its own decode path; the row-count diff is the
    // semantic claim under test.
    let duck_groups = duckdb_run_single_col_f64(
        "SELECT x FROM (SELECT x, COUNT(*) FROM t GROUP BY x)",
        &[("t", &vals)],
    );
    assert_eq!(
        duck_groups.len(),
        2,
        "DuckDB reference disagrees on group count: got {duck_groups:?}"
    );
}
