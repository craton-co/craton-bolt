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

// ============================================================================
// Stage 2: multi-key, NULL-aware, shmem variant.
// ============================================================================

/// Build a nullable Int32 batch.
fn int32_nullable_batch(name: &str, values: Vec<Option<i32>>) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        name,
        ArrowDataType::Int32,
        true,
    )]));
    RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values))]).unwrap()
}

/// Build a two-column nullable+non-nullable Int32 batch with names `a, b`.
fn two_int32_batch(a: Vec<i32>, b: Vec<i32>) -> RecordBatch {
    assert_eq!(a.len(), b.len());
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("a", ArrowDataType::Int32, false),
        ArrowField::new("b", ArrowDataType::Int32, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(a)), Arc::new(Int32Array::from(b))],
    )
    .unwrap()
}

/// `ORDER BY a ASC, b DESC` — multi-key lexicographic. Build inputs where
/// the major key has ties so the minor key's polarity is observable.
#[test]
#[ignore = "requires CUDA device"]
fn multi_key_int_int() {
    let mut engine = Engine::new().expect("ctx");

    // n = N_BIG so the GPU threshold is hit. a takes ~64 distinct values
    // so b's DESC ordering inside each tie group is visible.
    let n = N_BIG;
    let a: Vec<i32> = (0..n as i32).map(|i| i % 64).collect();
    let b: Vec<i32> = (0..n as i32).collect();
    let batch = two_int32_batch(a.clone(), b.clone());
    engine.register_table("t", batch).unwrap();

    let h = engine
        .sql("SELECT a, b FROM t ORDER BY a ASC, b DESC")
        .expect("ORDER BY a ASC, b DESC");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), n);

    let a_out = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let b_out = out.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
    for i in 1..n {
        let pa = a_out.value(i - 1);
        let ca = a_out.value(i);
        let pb = b_out.value(i - 1);
        let cb = b_out.value(i);
        assert!(pa <= ca, "ORDER BY a ASC violated at row {i}");
        if pa == ca {
            assert!(
                pb >= cb,
                "ORDER BY b DESC within a-tie violated at row {i}: {} < {}",
                pb,
                cb
            );
        }
    }
}

/// `ORDER BY a NULLS FIRST` — NULL rows must precede every non-NULL row.
#[test]
#[ignore = "requires CUDA device"]
fn null_first_int_with_nulls() {
    let mut engine = Engine::new().expect("ctx");

    let n = N_BIG;
    // ~10% nulls, scattered deterministically.
    let mut values: Vec<Option<i32>> = (0..n)
        .map(|i| if i % 10 == 3 { None } else { Some(i as i32) })
        .collect();
    // shuffle to defeat any input-order assumptions
    let mut rng_state: u64 = 0xc0ffee;
    for i in (1..n).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng_state as usize) % (i + 1);
        values.swap(i, j);
    }
    let batch = int32_nullable_batch("v", values.clone());
    engine.register_table("t", batch).unwrap();

    let h = engine
        .sql("SELECT v FROM t ORDER BY v NULLS FIRST")
        .expect("ORDER BY NULLS FIRST");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), n);

    let arr = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    // Find the boundary: count leading nulls.
    let mut leading_nulls = 0usize;
    while leading_nulls < n && arr.is_null(leading_nulls) {
        leading_nulls += 1;
    }
    // The total NULL count should equal the leading-nulls run.
    let expected_nulls = values.iter().filter(|v| v.is_none()).count();
    assert_eq!(
        leading_nulls, expected_nulls,
        "NULLS FIRST: leading nulls must equal total null count"
    );
    // After the null prefix, the rest must be ASC.
    for i in (leading_nulls + 1)..n {
        assert!(
            arr.value(i - 1) <= arr.value(i),
            "ASC violated at row {i} after null prefix"
        );
    }
}

/// `ORDER BY a NULLS LAST` — NULLs trailing.
#[test]
#[ignore = "requires CUDA device"]
fn null_last_int_with_nulls() {
    let mut engine = Engine::new().expect("ctx");

    let n = N_BIG;
    let mut values: Vec<Option<i32>> = (0..n)
        .map(|i| if i % 11 == 5 { None } else { Some(i as i32) })
        .collect();
    let mut rng_state: u64 = 0xfeed;
    for i in (1..n).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng_state as usize) % (i + 1);
        values.swap(i, j);
    }
    let batch = int32_nullable_batch("v", values.clone());
    engine.register_table("t", batch).unwrap();

    let h = engine
        .sql("SELECT v FROM t ORDER BY v NULLS LAST")
        .expect("ORDER BY NULLS LAST");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), n);

    let arr = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    // Find the boundary: count leading non-nulls.
    let mut leading_non_nulls = 0usize;
    while leading_non_nulls < n && !arr.is_null(leading_non_nulls) {
        leading_non_nulls += 1;
    }
    let expected_nulls = values.iter().filter(|v| v.is_none()).count();
    let trailing_nulls = n - leading_non_nulls;
    assert_eq!(
        trailing_nulls, expected_nulls,
        "NULLS LAST: trailing nulls must equal total null count"
    );
    // The leading non-null prefix must be ASC.
    for i in 1..leading_non_nulls {
        assert!(arr.value(i - 1) <= arr.value(i));
    }
    // The trailing nulls really are NULL.
    for i in leading_non_nulls..n {
        assert!(arr.is_null(i));
    }
}

/// `n_rows = 128` exercises the shmem variant (n_pow2 = 128 <= block_size).
/// Below the GPU_SORT_MIN_ROWS threshold (16384) the executor wouldn't take
/// the GPU path normally, so we drive the shmem dispatcher directly via the
/// public `sort_indices_on_gpu_multi` entry point.
#[test]
#[ignore = "requires CUDA device"]
fn shmem_variant_small_input() {
    use craton_bolt::__test_only_gpu_sort::{sort_indices_on_gpu_multi, GpuSortKey, SortLayout};
    use craton_bolt::__test_only_sort_kernel::SortDirection;
    use craton_bolt::__test_only_logical_plan::DataType;

    let n = 128usize;
    let mut values: Vec<i32> = (0..n as i32).collect();
    let mut s: u64 = 0xdeafbeef;
    for i in (1..n).rev() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (s as usize) % (i + 1);
        values.swap(i, j);
    }
    let arr = Int32Array::from(values.clone());

    let keys = vec![GpuSortKey {
        column: &arr,
        dtype: DataType::Int32,
        direction: SortDirection::Asc,
        nulls_first: false,
    }];
    let (layout, perm) = sort_indices_on_gpu_multi(&keys).expect("shmem sort");
    assert!(
        matches!(layout, SortLayout::Shmem),
        "n_pow2=128 must take the Shmem dispatch path; got {:?}",
        layout
    );
    assert_eq!(perm.len(), n);

    // Apply the permutation and check ASC order + permutation correctness.
    let sorted: Vec<i32> = (0..n).map(|i| values[perm.value(i) as usize]).collect();
    let mut expected = values.clone();
    expected.sort();
    assert_eq!(sorted, expected, "shmem-variant output must equal sorted(input)");
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
