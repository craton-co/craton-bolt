// SPDX-License-Identifier: Apache-2.0

//! Correctness of the Tier-1 shared-memory GROUP BY SUM across key / value
//! dtypes (`Int32` / `Int64` keys × `Float64` / `Int64` / `Int32` values) and
//! the low-cardinality run-length pre-aggregation, against a host fold.
//!
//! Also pins the two guard rails of the integer instances: a `SUM(Int64)`
//! whose magnitude bound could overflow `i64` must take the guarded general
//! path (which reports `SUM(integer) overflow` when it really overflows), and
//! negative keys must decline to the general path rather than index shared
//! memory.
//!
//! GPU-gated: `cargo test --test groupby_typed_shmem_test -- --ignored --test-threads=1`.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{Field, Schema};

use craton_bolt::Engine;

/// Rows above the Tier-1 minimum (64 Ki) so the shared-memory path is taken.
const ROWS: usize = 200_000;

#[derive(Clone, Copy, Debug)]
enum K {
    I32,
    I64,
}

#[derive(Clone, Copy, Debug)]
enum V {
    F64,
    I64,
    I32,
}

fn table(k: K, v: V, key_of: impl Fn(usize) -> i64, val_of: impl Fn(usize) -> i64) -> RecordBatch {
    let keys: ArrayRef = match k {
        K::I32 => Arc::new((0..ROWS).map(|i| key_of(i) as i32).collect::<Int32Array>()),
        K::I64 => Arc::new((0..ROWS).map(&key_of).collect::<Int64Array>()),
    };
    let vals: ArrayRef = match v {
        V::F64 => Arc::new(
            (0..ROWS)
                .map(|i| val_of(i) as f64)
                .collect::<Float64Array>(),
        ),
        V::I64 => Arc::new((0..ROWS).map(&val_of).collect::<Int64Array>()),
        V::I32 => Arc::new((0..ROWS).map(|i| val_of(i) as i32).collect::<Int32Array>()),
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", keys.data_type().clone(), false),
        Field::new("v", vals.data_type().clone(), false),
    ]));
    RecordBatch::try_new(schema, vec![keys, vals]).unwrap()
}

fn as_i64(a: &ArrayRef, i: usize) -> i64 {
    if let Some(x) = a.as_any().downcast_ref::<Int64Array>() {
        return x.value(i);
    }
    if let Some(x) = a.as_any().downcast_ref::<Int32Array>() {
        return i64::from(x.value(i));
    }
    if let Some(x) = a.as_any().downcast_ref::<Float64Array>() {
        return x.value(i) as i64;
    }
    panic!("unexpected dtype {:?}", a.data_type())
}

fn run(engine: &Engine) -> BTreeMap<i64, i64> {
    let out = engine
        .sql("SELECT k, SUM(v) FROM t GROUP BY k")
        .unwrap()
        .into_record_batch();
    (0..out.num_rows())
        .map(|i| (as_i64(out.column(0), i), as_i64(out.column(1), i)))
        .collect()
}

fn expected(key_of: impl Fn(usize) -> i64, val_of: impl Fn(usize) -> i64) -> BTreeMap<i64, i64> {
    let mut m = BTreeMap::new();
    for i in 0..ROWS {
        *m.entry(key_of(i)).or_insert(0i64) += val_of(i);
    }
    m
}

#[test]
#[ignore = "gpu:groupby — typed Tier-1 SUM matches the host fold"]
fn typed_shmem_sum_matches_host_fold() {
    // Sparse keys (gaps must not appear), clustered runs, and a single key.
    type KeyShape = (&'static str, fn(usize) -> i64);
    let key_shapes: [KeyShape; 4] = [
        ("single", |_| 7),
        ("mod100", |i| (i % 100) as i64),
        ("sparse", |i| ((i % 50) * 13) as i64),
        ("runs", |i| (i / 1000 % 900) as i64),
    ];
    let val_of = |i: usize| ((i * 7919) % 2001) as i64 - 1000;
    for k in [K::I32, K::I64] {
        for v in [V::F64, V::I64, V::I32] {
            for (name, key_of) in key_shapes {
                let mut engine = Engine::new().expect("ctx");
                engine
                    .register_table("t", table(k, v, key_of, val_of))
                    .unwrap();
                assert_eq!(
                    run(&engine),
                    expected(key_of, val_of),
                    "{k:?} keys × {v:?} values, {name}"
                );
            }
        }
    }
}

#[test]
#[ignore = "gpu:groupby — overflow-prone SUM(Int64) keeps the guarded path"]
fn int64_sum_overflow_is_still_reported() {
    let mut engine = Engine::new().expect("ctx");
    engine
        .register_table(
            "t",
            table(K::I32, V::I64, |i| (i % 4) as i64, |_| i64::MAX / 8),
        )
        .unwrap();
    let err = engine
        .sql("SELECT k, SUM(v) FROM t GROUP BY k")
        .expect_err("a group sum beyond i64::MAX must not wrap silently");
    assert!(err.to_string().contains("SUM(integer) overflow"), "{err}");
}

#[test]
#[ignore = "gpu:groupby — large magnitudes that do not overflow stay exact"]
fn int64_sum_large_values_without_overflow() {
    let big = (i64::MAX / (ROWS as i64)) - 1;
    let key_of = |i: usize| (i % 3) as i64;
    let val_of = move |i: usize| if i % 2 == 0 { big } else { -big };
    let mut engine = Engine::new().expect("ctx");
    engine
        .register_table("t", table(K::I64, V::I64, key_of, val_of))
        .unwrap();
    assert_eq!(run(&engine), expected(key_of, val_of));
}

#[test]
#[ignore = "gpu:groupby — negative keys decline the shared-memory path"]
fn negative_keys_fall_back_correctly() {
    let key_of = |i: usize| (i % 10) as i64 - 5;
    let val_of = |i: usize| (i % 7) as i64;
    for k in [K::I32, K::I64] {
        let mut engine = Engine::new().expect("ctx");
        engine
            .register_table("t", table(k, V::I64, key_of, val_of))
            .unwrap();
        assert_eq!(run(&engine), expected(key_of, val_of), "{k:?}");
    }
}
