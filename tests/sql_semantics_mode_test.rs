// SPDX-License-Identifier: AGPL-3.0-only

//! `EngineBuilder::sql_semantics`: the default [`SqlSemantics::Ieee`] keeps
//! IEEE float comparisons, wrapping integer arithmetic and `NULLS FIRST` for
//! `ASC`; [`SqlSemantics::Standard`] matches DuckDB / PostgreSQL (NaN equal to
//! itself and greater than every other value, integer overflow is an error,
//! `NULLS LAST`). Reproduces report craton-bolt-010 on both tier policies.
//!
//! `cargo test --test sql_semantics_mode_test -- --ignored --test-threads=1`

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::{Engine, SqlSemantics, TierPolicy};

const POLICIES: [TierPolicy; 2] = [TierPolicy::GpuFirst, TierPolicy::CostBased];

/// The report's table: `d` is NaN when `i % 4 == 0` (3 NaNs), else `i + 0.4`;
/// `big` is `i32::MAX` on every row; `k` is `i`.
fn report_table() -> RecordBatch {
    let d = Float64Array::from(
        (0..10)
            .map(|i| if i % 4 == 0 { f64::NAN } else { i as f64 + 0.4 })
            .collect::<Vec<_>>(),
    );
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("d", ArrowDataType::Float64, false),
        ArrowField::new("big", ArrowDataType::Int32, false),
        ArrowField::new("k", ArrowDataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(d) as ArrayRef,
            Arc::new(Int32Array::from(vec![i32::MAX; 10])),
            Arc::new(Int64Array::from((0..10).collect::<Vec<i64>>())),
        ],
    )
    .unwrap()
}

/// Nullable floats with NaN (both signs) and NULL on both comparison sides,
/// plus nullable integers near the limits.
fn nullable_table() -> RecordBatch {
    let a: Vec<Option<f64>> = vec![
        Some(f64::NAN),
        Some(1.0),
        None,
        Some(-f64::NAN),
        Some(5.0),
        Some(f64::NAN),
        None,
        Some(2.0),
    ];
    let b: Vec<Option<f64>> = vec![
        Some(f64::NAN),
        Some(f64::NAN),
        Some(f64::NAN),
        Some(1.0),
        None,
        Some(3.0),
        None,
        Some(2.0),
    ];
    let x: Vec<Option<i64>> = vec![
        Some(i64::MAX),
        Some(1),
        None,
        Some(i64::MIN),
        Some(-7),
        Some(0),
        None,
        Some(3_037_000_499),
    ];
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int32, false),
        ArrowField::new("a", ArrowDataType::Float64, true),
        ArrowField::new("b", ArrowDataType::Float64, true),
        ArrowField::new("x", ArrowDataType::Int64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from((0..8).collect::<Vec<i32>>())) as ArrayRef,
            Arc::new(Float64Array::from(a)),
            Arc::new(Float64Array::from(b)),
            Arc::new(Int64Array::from(x)),
        ],
    )
    .unwrap()
}

fn engine(semantics: Option<SqlSemantics>, policy: TierPolicy) -> Engine {
    let mut builder = Engine::builder().tier_policy(policy);
    if let Some(s) = semantics {
        builder = builder.sql_semantics(s);
    }
    let mut e = builder.build().expect("ctx");
    e.register_table("p", report_table()).unwrap();
    e.register_table("n", nullable_table()).unwrap();
    e
}

fn rows(e: &Engine, q: &str) -> usize {
    e.sql(q)
        .unwrap_or_else(|err| panic!("{q}: {err}"))
        .into_record_batch()
        .num_rows()
}

fn ids(e: &Engine, q: &str) -> Vec<i32> {
    let b = e
        .sql(q)
        .unwrap_or_else(|err| panic!("{q}: {err}"))
        .into_record_batch();
    let c = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    c.iter().map(|v| v.unwrap()).collect()
}

fn f64s(e: &Engine, q: &str) -> Vec<Option<f64>> {
    let b = e
        .sql(q)
        .unwrap_or_else(|err| panic!("{q}: {err}"))
        .into_record_batch();
    let c = b.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
    c.iter().collect()
}

fn overflow_error(e: &Engine, q: &str) -> String {
    match e.sql(q) {
        Ok(h) => panic!(
            "{q}: expected an out-of-range error, got {:?}",
            h.into_record_batch().column(0)
        ),
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("out of range"),
                "{q}: error should say `out of range`: {msg}"
            );
            msg
        }
    }
}

/// The report's reproduction table, in the SQL-compatible mode.
#[test]
#[ignore = "gpu:e2e — Standard semantics match DuckDB on the report's table"]
fn standard_mode_matches_duckdb_on_the_report_table() {
    for policy in POLICIES {
        let e = engine(Some(SqlSemantics::Standard), policy);
        assert_eq!(e.sql_semantics(), SqlSemantics::Standard);
        for (q, want) in [
            ("SELECT d FROM p WHERE d > 3", 8),
            ("SELECT d FROM p WHERE NOT (d > 3)", 2),
            ("SELECT d FROM p WHERE d = d", 10),
            ("SELECT d FROM p WHERE d != 3.4", 9),
            ("SELECT d FROM p WHERE d != d", 0),
            ("SELECT d FROM p WHERE d >= (0.0 / 0.0)", 3),
            ("SELECT d FROM p WHERE d < 3", 2),
            ("SELECT d FROM p WHERE d <= 3", 2),
            ("SELECT d FROM p WHERE d BETWEEN 3 AND 6", 2),
        ] {
            assert_eq!(rows(&e, q), want, "{policy:?}: {q}");
        }
        // The report's `big + 1`: an integer literal is BIGINT here, so the
        // sum is BIGINT and exact in both modes. INTEGER + INTEGER overflows.
        assert_eq!(big_plus_one(&e), 2_147_483_648, "{policy:?}");
        overflow_error(&e, "SELECT big + big FROM p");
    }
}

fn big_plus_one(e: &Engine) -> i64 {
    let b = e.sql("SELECT big + 1 FROM p").unwrap().into_record_batch();
    let c = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    c.value(0)
}

/// The default keeps today's IEEE / wrapping results (patch release).
#[test]
#[ignore = "gpu:e2e — the default mode keeps IEEE semantics"]
fn default_mode_keeps_ieee_semantics() {
    for policy in POLICIES {
        let e = engine(None, policy);
        assert_eq!(e.sql_semantics(), SqlSemantics::Ieee);
        for (q, want) in [
            ("SELECT d FROM p WHERE d > 3", 5),
            ("SELECT d FROM p WHERE NOT (d > 3)", 5),
            ("SELECT d FROM p WHERE d = d", 7),
            ("SELECT d FROM p WHERE d != 3.4", 6),
        ] {
            assert_eq!(rows(&e, q), want, "{policy:?}: {q}");
        }
        assert_eq!(big_plus_one(&e), 2_147_483_648, "{policy:?}");
        // MIN / MAX order NaN as the largest value in both modes.
        let mx = f64s(&e, "SELECT MAX(d) FROM p");
        assert!(mx[0].unwrap().is_nan(), "{policy:?}: {mx:?}");
        assert_eq!(
            f64s(&e, "SELECT MIN(d) FROM p"),
            vec![Some(1.4)],
            "{policy:?}"
        );
        let b = e
            .sql("SELECT big + big FROM p")
            .unwrap()
            .into_record_batch();
        let c = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(c.value(0), -2, "{policy:?}: wraps");
        // Narrowing CAST keeps the low 32 bits; Int64 arithmetic wraps.
        let b = e
            .sql("SELECT CAST(x AS INT), x * x FROM n WHERE id = 0")
            .unwrap()
            .into_record_batch();
        let c = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(c.value(0), -1, "{policy:?}: CAST wraps");
        let c = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(c.value(0), 1, "{policy:?}: MAX * MAX wraps");
    }
}

/// NULL stays UNKNOWN through the total-order rewrite (Kleene logic), for
/// column-vs-column comparisons with NaN and NULL on either side.
#[test]
#[ignore = "gpu:e2e — Standard comparisons keep NULL semantics"]
fn standard_comparisons_keep_null_semantics() {
    // id: a, b
    // 0: NaN, NaN   1: 1, NaN   2: NULL, NaN   3: -NaN, 1
    // 4: 5, NULL    5: NaN, 3   6: NULL, NULL  7: 2, 2
    for policy in POLICIES {
        let e = engine(Some(SqlSemantics::Standard), policy);
        for (q, want) in [
            ("SELECT id FROM n WHERE a = b ORDER BY id", vec![0, 7]),
            ("SELECT id FROM n WHERE a != b ORDER BY id", vec![1, 3, 5]),
            (
                "SELECT id FROM n WHERE NOT (a = b) ORDER BY id",
                vec![1, 3, 5],
            ),
            ("SELECT id FROM n WHERE a < b ORDER BY id", vec![1]),
            ("SELECT id FROM n WHERE a <= b ORDER BY id", vec![0, 1, 7]),
            ("SELECT id FROM n WHERE a > b ORDER BY id", vec![3, 5]),
            (
                "SELECT id FROM n WHERE a >= b ORDER BY id",
                vec![0, 3, 5, 7],
            ),
            ("SELECT id FROM n WHERE NOT (a >= b) ORDER BY id", vec![1]),
            (
                "SELECT id FROM n WHERE NOT (a < b) ORDER BY id",
                vec![0, 3, 5, 7],
            ),
            ("SELECT id FROM n WHERE a > 4 ORDER BY id", vec![0, 3, 4, 5]),
            ("SELECT id FROM n WHERE NOT (a > 4) ORDER BY id", vec![1, 7]),
            (
                "SELECT id FROM n WHERE a IN (2, (0.0 / 0.0)) ORDER BY id",
                vec![0, 3, 5, 7],
            ),
            (
                "SELECT id FROM n WHERE (a > b) IS NULL ORDER BY id",
                vec![2, 4, 6],
            ),
        ] {
            assert_eq!(ids(&e, q), want, "{policy:?}: {q}");
        }
    }
}

/// Sort, MIN / MAX and DISTINCT treat every NaN as one value,
/// greater than all others; `ASC` puts NULLs last.
#[test]
#[ignore = "gpu:e2e — Standard NaN ordering in sort / aggregates"]
fn standard_nan_ordering_in_sort_and_aggregates() {
    for policy in POLICIES {
        let e = engine(Some(SqlSemantics::Standard), policy);
        // ASC: 1, 2, 5, NaN, NaN, NaN, NULL, NULL.
        let got = f64s(&e, "SELECT a FROM n ORDER BY a");
        assert_eq!(got.len(), 8, "{policy:?}");
        assert_eq!(
            &got[..3],
            &[Some(1.0), Some(2.0), Some(5.0)],
            "{policy:?}: {got:?}"
        );
        assert!(
            got[3..6].iter().all(|v| v.unwrap().is_nan()),
            "{policy:?}: {got:?}"
        );
        assert_eq!(&got[6..], &[None, None], "{policy:?}: ASC is NULLS LAST");
        // DESC: NaN, NaN, NaN, 5, 2, 1, NULL, NULL.
        let got = f64s(&e, "SELECT a FROM n ORDER BY a DESC");
        assert!(
            got[..3].iter().all(|v| v.unwrap().is_nan()),
            "{policy:?}: {got:?}"
        );
        assert_eq!(
            &got[3..6],
            &[Some(5.0), Some(2.0), Some(1.0)],
            "{policy:?}: {got:?}"
        );
        assert_eq!(&got[6..], &[None, None], "{policy:?}: DESC is NULLS LAST");
        // Explicit NULLS FIRST still wins.
        let got = f64s(&e, "SELECT a FROM n ORDER BY a NULLS FIRST");
        assert_eq!(&got[..2], &[None, None], "{policy:?}");

        let mx = f64s(&e, "SELECT MAX(a) FROM n");
        assert!(mx[0].unwrap().is_nan(), "{policy:?}: MAX is NaN: {mx:?}");
        assert_eq!(
            f64s(&e, "SELECT MIN(a) FROM n"),
            vec![Some(1.0)],
            "{policy:?}"
        );
        // One NaN value (both signs) and one NULL among the distinct values.
        assert_eq!(rows(&e, "SELECT DISTINCT a FROM n"), 5, "{policy:?}");
    }
}

/// The default mode keeps `NULLS FIRST` for `ASC`.
#[test]
#[ignore = "gpu:e2e — the default mode keeps ASC NULLS FIRST"]
fn default_mode_keeps_nulls_first_for_asc() {
    let e = engine(None, TierPolicy::GpuFirst);
    let got = f64s(&e, "SELECT b FROM n ORDER BY b");
    assert_eq!(&got[..2], &[None, None], "{got:?}");
}

/// Large inputs take the GPU sort path; NaN of either sign sorts greatest.
#[test]
#[ignore = "gpu:e2e — Standard NaN ordering on the GPU sort path"]
fn standard_nan_ordering_on_the_gpu_sort_path() {
    let n = 20_000usize;
    let v: Vec<f64> = (0..n)
        .map(|i| match i % 10 {
            0 => f64::NAN,
            5 => -f64::NAN,
            _ => (i as f64) - 10_000.0,
        })
        .collect();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "v",
        ArrowDataType::Float64,
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(v))]).unwrap();
    for policy in POLICIES {
        let mut e = Engine::builder()
            .tier_policy(policy)
            .sql_semantics(SqlSemantics::Standard)
            .build()
            .expect("ctx");
        e.register_table("big", batch.clone()).unwrap();
        let got = f64s(&e, "SELECT v FROM big ORDER BY v");
        let nans = n / 5;
        assert!(
            got[n - nans..].iter().all(|v| v.unwrap().is_nan()),
            "{policy:?}: NaNs last"
        );
        assert!(
            got[..n - nans]
                .windows(2)
                .all(|w| w[0].unwrap() <= w[1].unwrap()),
            "{policy:?}: ascending, no NaN in the prefix"
        );
        assert_eq!(
            rows(&e, "SELECT v FROM big WHERE v > 0"),
            8_000 + nans,
            "{policy:?}"
        );
        assert_eq!(rows(&e, "SELECT v FROM big WHERE v = v"), n, "{policy:?}");
    }
}

/// Checked integer arithmetic: `+ - *` and narrowing `CAST` raise an
/// out-of-range error only for rows that are actually evaluated.
#[test]
#[ignore = "gpu:e2e — Standard integer overflow is an error"]
fn standard_integer_overflow_is_an_error() {
    for policy in POLICIES {
        let e = engine(Some(SqlSemantics::Standard), policy);
        // x: MAX, 1, NULL, MIN, -7, 0, NULL, 3037000499
        for q in [
            "SELECT big + big FROM p",
            "SELECT big * big FROM p",
            "SELECT big + big - big FROM p",
            "SELECT k FROM p WHERE big + big > 0",
            "SELECT SUM(big + big) FROM p",
            "SELECT k, SUM(big + big) FROM p GROUP BY k",
            "SELECT k + 9223372036854775800 FROM p",
            "SELECT x + 1 FROM n",
            "SELECT x - 1 FROM n",
            "SELECT x * x FROM n",
            "SELECT x * -1 FROM n",
            "SELECT CAST(x AS INT) FROM n",
        ] {
            overflow_error(&e, q);
        }
        // Rows that are filtered out, NULL, or in an untaken CASE branch do
        // not raise; in-range arithmetic is exact.
        for (q, want) in [
            ("SELECT x + 1 FROM n WHERE x < 100 AND x > -100", vec![2, -6, 1]),
            ("SELECT x * 2 FROM n WHERE id IN (1, 4, 5)", vec![2, -14, 0]),
            (
                "SELECT CASE WHEN x < 100 AND x > -100 THEN x * 1000 ELSE 0 END FROM n WHERE x IS NOT NULL",
                vec![0, 1000, 0, -7000, 0, 0],
            ),
            ("SELECT x - 1 FROM n WHERE id = 7", vec![3_037_000_498]),
            ("SELECT x * x FROM n WHERE id = 7", vec![9_223_372_030_926_249_001]),
        ] {
            let b = e
                .sql(q)
                .unwrap_or_else(|err| panic!("{policy:?}: {q}: {err}"))
                .into_record_batch();
            let c = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let got: Vec<i64> = c.iter().map(|v| v.unwrap()).collect();
            assert_eq!(got, want, "{policy:?}: {q}");
        }
        assert_eq!(
            rows(&e, "SELECT x + 1 FROM n WHERE x IS NULL"),
            2,
            "{policy:?}"
        );
    }
}

/// `g` group key, `v` float: group 0 all NaN (both signs), group 1 {1, NaN},
/// group 2 {NULL, NaN}, group 3 {NULL}; `n` rows repeating the pattern so
/// large inputs reach the GPU grouped paths.
fn nan_groups(n: usize) -> RecordBatch {
    let pat: [(i32, Option<f64>); 7] = [
        (0, Some(f64::NAN)),
        (0, Some(-f64::NAN)),
        (1, Some(1.0)),
        (1, Some(f64::NAN)),
        (2, None),
        (2, Some(f64::NAN)),
        (3, None),
    ];
    let rows: Vec<_> = (0..n).map(|i| pat[i % pat.len()]).collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("g", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Float64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(
                rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.1).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// MIN is NaN only when every non-NULL value is NaN (NaN is the largest
/// value), never the +inf identity: scalar and grouped, both modes.
#[test]
#[ignore = "gpu:e2e — MIN over all-NaN input is NaN"]
fn min_over_all_nan_input_is_nan() {
    for n in [7usize, 70_000] {
        for policy in POLICIES {
            for sem in [None, Some(SqlSemantics::Standard)] {
                let mut e = engine(sem, policy);
                e.register_table("g", nan_groups(n)).unwrap();
                let tag = format!("{n} {policy:?} {sem:?}");
                for q in [
                    "SELECT MIN(v) FROM g WHERE g = 0",
                    "SELECT MIN(v) FROM g WHERE g = 2",
                    "SELECT MAX(v) FROM g WHERE g = 0",
                ] {
                    let got = f64s(&e, q);
                    assert!(got[0].is_some_and(f64::is_nan), "{tag}: {q}: {got:?}");
                }
                assert_eq!(
                    f64s(&e, "SELECT MIN(v) FROM g WHERE g = 1"),
                    vec![Some(1.0)],
                    "{tag}"
                );
                let b = e
                    .sql("SELECT g, MIN(v) AS lo, MAX(v) AS hi FROM g GROUP BY g ORDER BY g")
                    .unwrap_or_else(|err| panic!("{tag}: {err}"))
                    .into_record_batch();
                let lo = b.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
                let hi = b.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
                let lo: Vec<Option<f64>> = lo.iter().collect();
                let hi: Vec<Option<f64>> = hi.iter().collect();
                assert!(lo[0].is_some_and(f64::is_nan), "{tag}: grouped MIN {lo:?}");
                assert_eq!(lo[1], Some(1.0), "{tag}: grouped MIN {lo:?}");
                assert!(lo[2].is_some_and(f64::is_nan), "{tag}: grouped MIN {lo:?}");
                assert_eq!(lo[3], None, "{tag}: grouped MIN {lo:?}");
                assert!(
                    hi[..3].iter().all(|v| v.is_some_and(f64::is_nan)),
                    "{tag}: {hi:?}"
                );
                assert_eq!(hi[3], None, "{tag}");
            }
        }
    }
}

fn float_keys(name: &str, vals: Vec<Option<f64>>) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(format!("{name}id"), ArrowDataType::Int32, false),
        ArrowField::new(format!("{name}k"), ArrowDataType::Float64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from((0..vals.len() as i32).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(Float64Array::from(vals)),
        ],
    )
    .unwrap()
}

fn join_engine(semantics: Option<SqlSemantics>, policy: TierPolicy) -> Engine {
    let mut e = engine(semantics, policy);
    // l: 0 NaN, 1 1.0, 2 NULL, 3 -NaN ; r: 0 -NaN, 1 1.0, 2 NULL, 3 2.0
    e.register_table(
        "l",
        float_keys("l", vec![Some(f64::NAN), Some(1.0), None, Some(-f64::NAN)]),
    )
    .unwrap();
    e.register_table(
        "r",
        float_keys("r", vec![Some(-f64::NAN), Some(1.0), None, Some(2.0)]),
    )
    .unwrap();
    e
}

fn id_pairs(e: &Engine, q: &str) -> Vec<(i32, i32)> {
    let b = e
        .sql(q)
        .unwrap_or_else(|err| panic!("{q}: {err}"))
        .into_record_batch();
    let a = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let c = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
    let mut v: Vec<_> = a
        .iter()
        .zip(c.iter())
        .map(|(x, y)| (x.unwrap(), y.unwrap()))
        .collect();
    v.sort();
    v
}

/// Standard: NaN = NaN in equi-join keys and IN membership (list and
/// subquery); NULL never matches.
#[test]
#[ignore = "gpu:e2e — Standard NaN equality in joins and IN"]
fn standard_nan_keys_match_in_joins_and_in() {
    for policy in POLICIES {
        let e = join_engine(Some(SqlSemantics::Standard), policy);
        assert_eq!(
            id_pairs(&e, "SELECT lid, rid FROM l JOIN r ON lk = rk"),
            vec![(0, 0), (1, 1), (3, 0)],
            "{policy:?}"
        );
        for (q, want) in [
            ("SELECT lid FROM l WHERE lk IN (SELECT rk FROM r) ORDER BY lid", vec![0, 1, 3]),
            (
                "SELECT lid FROM l WHERE lk NOT IN (SELECT rk FROM r WHERE rk IS NOT NULL) ORDER BY lid",
                vec![],
            ),
            ("SELECT lid FROM l WHERE lk IN (0.0 / 0.0) ORDER BY lid", vec![0, 3]),
            ("SELECT lid FROM l WHERE lk NOT IN (1, 0.0 / 0.0) ORDER BY lid", vec![]),
        ] {
            assert_eq!(ids(&e, q), want, "{policy:?}: {q}");
        }
    }
}

/// Standard: a float-to-integer CAST of NaN, +-inf or an out-of-range value
/// is an out-of-range error; in-range values convert as before.
#[test]
#[ignore = "gpu:e2e — Standard float-to-int CAST overflow is an error"]
fn standard_float_to_int_cast_overflow_is_an_error() {
    for policy in POLICIES {
        let mut e = engine(Some(SqlSemantics::Standard), policy);
        e.register_table(
            "f",
            float_keys(
                "f",
                vec![
                    Some(2.25),
                    Some(-2.25),
                    Some(f64::NAN),
                    Some(f64::INFINITY),
                    Some(3e9),
                    Some(1e19),
                    None,
                    Some(f64::NEG_INFINITY),
                ],
            ),
        )
        .unwrap();
        for q in [
            "SELECT CAST(fk AS INT) FROM f WHERE fid = 2",
            "SELECT CAST(fk AS BIGINT) FROM f WHERE fid = 2",
            "SELECT CAST(fk AS INT) FROM f WHERE fid = 3",
            "SELECT CAST(fk AS BIGINT) FROM f WHERE fid = 7",
            "SELECT CAST(fk AS INT) FROM f WHERE fid = 4",
            "SELECT CAST(fk AS BIGINT) FROM f WHERE fid = 5",
        ] {
            overflow_error(&e, q);
        }
        let b = e
            .sql("SELECT CAST(fk AS INT) FROM f WHERE fid IN (0, 1, 6)")
            .unwrap_or_else(|err| panic!("{policy:?}: {err}"))
            .into_record_batch();
        let c = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let got: Vec<Option<i32>> = c.iter().collect();
        assert_eq!(got, vec![Some(2), Some(-2), None], "{policy:?}");
        assert_eq!(
            rows(&e, "SELECT CAST(fk AS BIGINT) FROM f WHERE fid = 4"),
            1,
            "{policy:?}: 3e9 fits BIGINT"
        );
    }
}

/// The default mode keeps today's float-key behaviour: join keys compare
/// bit patterns (so only identical NaN payloads match) and IN membership uses
/// IEEE `=` (a NaN never matches).
#[test]
#[ignore = "gpu:e2e — default mode keeps bitwise NaN join keys"]
fn default_mode_keeps_float_key_equality() {
    for policy in POLICIES {
        let e = join_engine(None, policy);
        assert_eq!(
            id_pairs(&e, "SELECT lid, rid FROM l JOIN r ON lk = rk"),
            vec![(1, 1), (3, 0)],
            "{policy:?}"
        );
        for (q, want) in [
            (
                "SELECT lid FROM l WHERE lk IN (SELECT rk FROM r) ORDER BY lid",
                vec![1],
            ),
            (
                "SELECT lid FROM l WHERE lk IN (0.0 / 0.0) ORDER BY lid",
                vec![],
            ),
        ] {
            assert_eq!(ids(&e, q), want, "{policy:?}: {q}");
        }
    }
}
