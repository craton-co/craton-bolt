// SPDX-License-Identifier: Apache-2.0

//! End-to-end ORDER BY tests for the GPU bitonic sort fast path.
//!
//! These tests run `SELECT v FROM t ORDER BY v` (and friends) through the
//! full `Engine::sql` pipeline, exercising the GPU path in
//! `crate::exec::gpu_sort` via the gate in `crate::exec::sort::try_gpu_sort`.
//!
//! Every test is `#[ignore]`'d so non-GPU CI passes. Run with
//! `cargo test --test sort_e2e -- --ignored` on a CUDA host.

use std::sync::Arc;

use arrow_array::{Array, Float64Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::Engine;

/// Build a single-column Int32 batch from the given values.
fn int32_batch(name: &str, values: Vec<i32>) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        name,
        ArrowDataType::Int32,
        false,
    )]));
    RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values))]).unwrap()
}

/// Build a single-column Int64 batch from the given values.
fn int64_batch(name: &str, values: Vec<i64>) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        name,
        ArrowDataType::Int64,
        false,
    )]));
    RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values))]).unwrap()
}

/// Build a single-column Float64 batch from the given values.
fn float64_batch(name: &str, values: Vec<f64>) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        name,
        ArrowDataType::Float64,
        false,
    )]));
    RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap()
}

/// Deterministic Fisher-Yates shuffle so tests are reproducible without
/// pulling a `rand` dev-dep. The LCG constants are Knuth's.
fn shuffle_deterministic<T: Copy>(xs: &mut [T], seed: u64) {
    let mut s = seed;
    for i in (1..xs.len()).rev() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (s as usize) % (i + 1);
        xs.swap(i, j);
    }
}

/// Above the GPU_SORT_MIN_ROWS threshold so the GPU path is taken.
const N_BIG: usize = 16_384;

/// `ORDER BY v ASC` on a 16k-row Int32 column. Validates that the GPU fast
/// path returns a strictly ascending sequence.
#[test]
#[ignore = "requires CUDA device"]
fn e2e_order_by_int32_asc() {
    let mut engine = Engine::new().expect("ctx");

    let mut values: Vec<i32> = (0..N_BIG as i32).collect();
    shuffle_deterministic(&mut values, 0xdeadbeef);
    let batch = int32_batch("v", values.clone());
    engine.register_table("t", batch).unwrap();

    let h = engine
        .sql("SELECT v FROM t ORDER BY v")
        .expect("ORDER BY v");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), N_BIG);

    let arr = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32Array");
    for i in 1..N_BIG {
        assert!(
            arr.value(i - 1) <= arr.value(i),
            "non-ASC at row {i}: {} > {}",
            arr.value(i - 1),
            arr.value(i)
        );
    }
    // And the output is a true permutation of the input.
    let mut expected = values;
    expected.sort();
    let actual: Vec<i32> = (0..N_BIG).map(|i| arr.value(i)).collect();
    assert_eq!(actual, expected);
}

/// `ORDER BY v DESC` on a 16k-row Int32 column.
#[test]
#[ignore = "requires CUDA device"]
fn e2e_order_by_int32_desc() {
    let mut engine = Engine::new().expect("ctx");

    let mut values: Vec<i32> = (0..N_BIG as i32).collect();
    shuffle_deterministic(&mut values, 0xfeedface);
    let batch = int32_batch("v", values.clone());
    engine.register_table("t", batch).unwrap();

    let h = engine
        .sql("SELECT v FROM t ORDER BY v DESC")
        .expect("ORDER BY v DESC");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), N_BIG);

    let arr = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32Array");
    for i in 1..N_BIG {
        assert!(
            arr.value(i - 1) >= arr.value(i),
            "non-DESC at row {i}: {} < {}",
            arr.value(i - 1),
            arr.value(i)
        );
    }
    let mut expected = values;
    expected.sort_by(|a, b| b.cmp(a));
    let actual: Vec<i32> = (0..N_BIG).map(|i| arr.value(i)).collect();
    assert_eq!(actual, expected);
}

/// Non-power-of-two size exercises the padding path. 20_000 rounds up to
/// 32_768, with 12_768 sentinel entries.
#[test]
#[ignore = "requires CUDA device"]
fn e2e_order_by_int64_asc_non_pow2() {
    let mut engine = Engine::new().expect("ctx");

    let n = 20_000usize;
    let mut values: Vec<i64> = (0..n as i64).map(|i| (i * 7919) % 1_000_000).collect();
    shuffle_deterministic(&mut values, 0xc001cafe);
    let batch = int64_batch("v", values.clone());
    engine.register_table("t", batch).unwrap();

    let h = engine
        .sql("SELECT v FROM t ORDER BY v")
        .expect("ORDER BY v");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), n);

    let arr = out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64Array");
    for i in 1..n {
        assert!(arr.value(i - 1) <= arr.value(i));
    }
    let mut expected = values;
    expected.sort();
    let actual: Vec<i64> = (0..n).map(|i| arr.value(i)).collect();
    assert_eq!(actual, expected);
}

/// Float64 ASC on a non-power-of-two size.
#[test]
#[ignore = "requires CUDA device"]
fn e2e_order_by_float64_asc() {
    let mut engine = Engine::new().expect("ctx");

    let n = 17_000usize;
    let values: Vec<f64> = (0..n).map(|i| ((i as f64) * 1.61803398875).sin()).collect();
    let batch = float64_batch("v", values.clone());
    engine.register_table("t", batch).unwrap();

    let h = engine
        .sql("SELECT v FROM t ORDER BY v")
        .expect("ORDER BY v");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), n);

    let arr = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64Array");
    for i in 1..n {
        assert!(
            arr.value(i - 1) <= arr.value(i),
            "non-ASC f64 at row {i}: {} > {}",
            arr.value(i - 1),
            arr.value(i)
        );
    }
}

/// Multi-column projection with `ORDER BY` on one column — confirms the
/// non-key columns get gathered in lockstep so payload tracks the key.
#[test]
#[ignore = "requires CUDA device"]
fn e2e_order_by_keeps_payload_aligned() {
    let mut engine = Engine::new().expect("ctx");

    let n = N_BIG;
    let mut keys: Vec<i32> = (0..n as i32).collect();
    shuffle_deterministic(&mut keys, 0xa5a5a5a5);
    let payload: Vec<i32> = keys.iter().map(|k| k + 1_000_000).collect();

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(keys)),
            Arc::new(Int32Array::from(payload)),
        ],
    )
    .unwrap();
    engine.register_table("t", batch).unwrap();

    let h = engine
        .sql("SELECT k, v FROM t ORDER BY k")
        .expect("ORDER BY k");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), n);

    let k = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let v = out
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    for i in 0..n {
        assert_eq!(
            v.value(i),
            k.value(i) + 1_000_000,
            "payload row {i} drifted from key"
        );
    }
    for i in 1..n {
        assert!(k.value(i - 1) <= k.value(i));
    }
}

/// Below the GPU threshold the host path must still produce correct output.
/// This test guards against an accidental gate inversion that would route
/// small queries through the GPU and break on its preconditions.
#[test]
#[ignore = "requires CUDA device"]
fn e2e_small_input_uses_host_path() {
    let mut engine = Engine::new().expect("ctx");

    // 100 rows is well below GPU_SORT_MIN_ROWS = 16k.
    let mut values: Vec<i32> = (0..100i32).collect();
    shuffle_deterministic(&mut values, 0xabad1dea);
    let batch = int32_batch("v", values.clone());
    engine.register_table("t", batch).unwrap();

    let h = engine
        .sql("SELECT v FROM t ORDER BY v")
        .expect("small ORDER BY");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 100);

    let arr = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let actual: Vec<i32> = (0..100).map(|i| arr.value(i)).collect();
    let mut expected = values;
    expected.sort();
    assert_eq!(actual, expected);
}
