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

mod common;
use common::shuffle_deterministic;

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

/// Above the GPU_SORT_MIN_ROWS threshold so the GPU path is taken.
const N_BIG: usize = 16_384;

/// `ORDER BY v ASC` on a 16k-row Int32 column. Validates that the GPU fast
/// path returns a strictly ascending sequence.
#[test]
#[ignore = "gpu:sort"]
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
#[ignore = "gpu:sort"]
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
#[ignore = "gpu:sort"]
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
#[ignore = "gpu:sort"]
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
#[ignore = "gpu:sort"]
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

    let k = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let v = out.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
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
#[ignore = "gpu:sort"]
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
#[ignore = "gpu:sort"]
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
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
#[ignore = "gpu:sort"]
fn null_last_int_with_nulls() {
    let mut engine = Engine::new().expect("ctx");

    let n = N_BIG;
    let mut values: Vec<Option<i32>> = (0..n)
        .map(|i| if i % 11 == 5 { None } else { Some(i as i32) })
        .collect();
    let mut rng_state: u64 = 0xfeed;
    for i in (1..n).rev() {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
#[ignore = "gpu:sort"]
fn shmem_variant_small_input() {
    use craton_bolt::__test_only_gpu_sort::{sort_indices_on_gpu_multi, GpuSortKey, SortLayout};
    use craton_bolt::__test_only_logical_plan::DataType;
    use craton_bolt::__test_only_sort_kernel::SortDirection;

    let n = 128usize;
    let mut values: Vec<i32> = (0..n as i32).collect();
    let mut s: u64 = 0xdeafbeef;
    for i in (1..n).rev() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
    let (layout, perm) = sort_indices_on_gpu_multi(&keys)
        .expect("shmem sort")
        .expect("shmem sort: non-fallback path on int32");
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
    assert_eq!(
        sorted, expected,
        "shmem-variant output must equal sorted(input)"
    );
}

// ============================================================================
// Stage 3: lifted key cap, padded-row routing, Bool/Utf8-via-dict, packed-bit
// shmem validity.
// ============================================================================

/// `ORDER BY a, b, c, d, e, f, g, h ASC` — 8 keys, well above the Stage-2
/// hard cap of 4. Drives the lifted register-pressure-based cap.
#[test]
#[ignore = "gpu:sort"]
fn eight_key_sort() {
    use arrow_array::Int32Array;
    let mut engine = Engine::new().expect("ctx");

    let n = N_BIG;
    // Build 8 columns; each is `i / mod_k` so successive keys add
    // tiebreakers. The 8th column is unique so the final order is total.
    let mods = [16, 8, 8, 8, 8, 8, 8, 1];
    let cols: Vec<Vec<i32>> = mods
        .iter()
        .map(|m| {
            (0..n as i32)
                .map(|i| if *m > 1 { i % m } else { i })
                .collect()
        })
        .collect();
    let names = ["a", "b", "c", "d", "e", "f", "g", "h"];
    let fields: Vec<ArrowField> = names
        .iter()
        .map(|n| ArrowField::new(*n, ArrowDataType::Int32, false))
        .collect();
    let schema = Arc::new(ArrowSchema::new(fields));
    let arrays: Vec<Arc<dyn Array>> = cols
        .iter()
        .map(|c| Arc::new(Int32Array::from(c.clone())) as Arc<dyn Array>)
        .collect();
    let batch = RecordBatch::try_new(schema, arrays).unwrap();
    engine.register_table("t", batch).unwrap();

    let h = engine
        .sql("SELECT a, b, c, d, e, f, g, h FROM t ORDER BY a, b, c, d, e, f, g, h")
        .expect("ORDER BY 8 keys");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), n);
    // Validate strict lex order key-by-key.
    let downcast = |i: usize| out.column(i).as_any().downcast_ref::<Int32Array>().unwrap();
    let arrs: Vec<&Int32Array> = (0..8).map(downcast).collect();
    for row in 1..n {
        for (ki, a) in arrs.iter().enumerate() {
            let prev = a.value(row - 1);
            let curr = a.value(row);
            if prev != curr {
                assert!(
                    prev < curr,
                    "ORDER BY key #{ki} not ascending at row {row}: {prev} > {curr}"
                );
                break; // later keys may go any direction within this tie
            }
        }
    }
}

/// `ORDER BY b ASC` on a Bool column — Stage 3 added Bool support.
#[test]
#[ignore = "gpu:sort"]
fn bool_key_sort() {
    use arrow_array::{BooleanArray, Int32Array};
    let mut engine = Engine::new().expect("ctx");
    let n = N_BIG;
    // Mostly true, ~30% false, with a payload column that should track.
    let bools: Vec<bool> = (0..n).map(|i| i % 10 < 7).collect();
    let payload: Vec<i32> = (0..n as i32).collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("b", ArrowDataType::Boolean, false),
        ArrowField::new("p", ArrowDataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(BooleanArray::from(bools.clone())),
            Arc::new(Int32Array::from(payload)),
        ],
    )
    .unwrap();
    engine.register_table("t", batch).unwrap();

    let h = engine
        .sql("SELECT b, p FROM t ORDER BY b")
        .expect("ORDER BY b ASC");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), n);
    let b_out = out
        .column(0)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    // ASC: false (0) before true (1). All falses come before any true.
    let mut last_true = false;
    for i in 0..n {
        let v = b_out.value(i);
        if v {
            last_true = true;
        } else {
            assert!(
                !last_true,
                "ASC bool: false at row {i} found after a true was already seen"
            );
        }
    }
    // Count matches.
    let total_true_out = (0..n).filter(|i| b_out.value(*i)).count();
    let total_true_in = bools.iter().filter(|x| **x).count();
    assert_eq!(total_true_out, total_true_in);
}

/// `ORDER BY s ASC` over a dictionary-encoded Utf8 column. Stage 3 wires a
/// host-side adapter that drives the existing numeric kernel using the
/// dictionary's index column. After the sort the dictionary remains intact.
///
/// The engine's SQL pipeline doesn't yet expose DictionaryArray as a
/// directly registerable column (the `arrow_dtype_to_plan` mapping rejects
/// Dictionary types — Stage 4 follow-up). This test therefore drives the
/// host-side adapter directly via the `sort_indices_on_gpu_multi` entry
/// point, the same way `shmem_variant_small_input` does. It still exercises
/// the load-bearing piece: `host_values_for_key` peeling off the index
/// column and routing it through the i32 kernel.
#[test]
#[ignore = "gpu:sort"]
fn dict_utf8_key_sort() {
    use arrow_array::types::Int32Type;
    use arrow_array::{DictionaryArray, Int32Array, StringArray};
    use craton_bolt::__test_only_gpu_sort::{sort_indices_on_gpu_multi, GpuSortKey};
    use craton_bolt::__test_only_logical_plan::DataType;
    use craton_bolt::__test_only_sort_kernel::SortDirection;

    let n = 16_384usize;
    // Dictionary with a few entries; cyclic keys produce a coarse-grained
    // ordering that's trivial to verify.
    let dict_values = vec!["alpha", "bravo", "charlie", "delta", "echo"];
    let keys: Vec<i32> = (0..n as i32)
        .map(|i| i % (dict_values.len() as i32))
        .collect();
    let dict_arr: DictionaryArray<Int32Type> = DictionaryArray::try_new(
        Int32Array::from(keys.clone()),
        Arc::new(StringArray::from(dict_values.clone())),
    )
    .unwrap();

    // Drive the multi-key sort directly. dtype=Int32 because the adapter
    // routes a dict<i32,Utf8> via the i32 numeric kernel.
    let sort_keys = vec![GpuSortKey {
        column: &dict_arr,
        dtype: DataType::Int32,
        direction: SortDirection::Asc,
        nulls_first: false,
    }];
    let (_layout, perm) = sort_indices_on_gpu_multi(&sort_keys)
        .expect("dict-Utf8 sort")
        .expect("dict-Utf8 sort: non-fallback path");
    assert_eq!(perm.len(), n);

    // Apply the perm to the original index column; result must be ASC.
    let sorted: Vec<i32> = (0..n).map(|i| keys[perm.value(i) as usize]).collect();
    for i in 1..n {
        assert!(
            sorted[i - 1] <= sorted[i],
            "dict-Utf8 sort: index column not ASC at row {i}: {} > {}",
            sorted[i - 1],
            sorted[i]
        );
    }
    let mut expected = keys.clone();
    expected.sort();
    assert_eq!(sorted, expected, "dict-Utf8 sort must equal sorted(input)");
}

/// Sentinel-collision test: build an Int32 column whose values include the
/// ASC pad sentinel (`i32::MAX`) as a legitimate datum, then `ORDER BY a
/// ASC`. The Stage-2 path silently dropped these rows because they tied
/// the sentinel; Stage 3 routes padded rows via an explicit bit so real
/// `i32::MAX` values survive.
#[test]
#[ignore = "gpu:sort"]
fn sentinel_collision_does_not_drop_row() {
    use arrow_array::Int32Array;
    let mut engine = Engine::new().expect("ctx");
    // n_rows = N_BIG but pick a non-power-of-2 so n_pow2 > n_rows and the
    // padding is non-trivial.
    let n = N_BIG + 137;
    let mut values: Vec<i32> = (0..n as i32).collect();
    // Sprinkle ~50 i32::MAX values.
    for k in 0..50 {
        values[k * 211 % n] = i32::MAX;
    }
    let target_count = values.iter().filter(|v| **v == i32::MAX).count();
    assert!(
        target_count >= 1,
        "test setup should produce at least one i32::MAX row"
    );
    let batch = int32_batch("a", values.clone());
    engine.register_table("t", batch).unwrap();

    let h = engine
        .sql("SELECT a FROM t ORDER BY a")
        .expect("ORDER BY a ASC");
    let out = h.record_batch();
    assert_eq!(
        out.num_rows(),
        n,
        "Stage-3 padded routing must preserve every real row (including i32::MAX)"
    );
    let arr = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    // Count i32::MAX in output — must equal input.
    let out_max = (0..n).filter(|i| arr.value(*i) == i32::MAX).count();
    assert_eq!(
        out_max,
        target_count,
        "Stage-3: lost {} i32::MAX rows under sentinel collision",
        target_count - out_max
    );
    // And the whole output is ASC.
    for i in 1..n {
        assert!(arr.value(i - 1) <= arr.value(i));
    }
}

/// Symmetric companion to `sentinel_collision_does_not_drop_row`: DESC pads
/// with `i32::MIN`. A real row whose value is `i32::MIN` was the
/// symmetric risk pre-Stage-3 fix — without the explicit `is_padded` bit
/// routing it would have tied the sentinel and could end up at an index
/// past the truncation point. Stage 3's padded routing makes the real
/// value win the tiebreak regardless of direction; this test locks the
/// DESC half of that behaviour.
#[test]
#[ignore = "gpu:sort"]
fn sentinel_collision_desc_i32_min_does_not_drop_row() {
    use arrow_array::Int32Array;
    let mut engine = Engine::new().expect("ctx");
    // Same non-power-of-2 size choice as the ASC test so n_pow2 > n_rows
    // and the padding is non-trivial.
    let n = N_BIG + 137;
    let mut values: Vec<i32> = (0..n as i32).collect();
    // Sprinkle ~50 i32::MIN values. These would have tied the DESC pad
    // sentinel pre-Stage-3.
    for k in 0..50 {
        values[k * 211 % n] = i32::MIN;
    }
    let target_count = values.iter().filter(|v| **v == i32::MIN).count();
    assert!(
        target_count >= 1,
        "test setup should produce at least one i32::MIN row"
    );
    let batch = int32_batch("a", values.clone());
    engine.register_table("t", batch).unwrap();

    let h = engine
        .sql("SELECT a FROM t ORDER BY a DESC")
        .expect("ORDER BY a DESC");
    let out = h.record_batch();
    assert_eq!(
        out.num_rows(),
        n,
        "Stage-3 padded routing must preserve every real row (including i32::MIN) under DESC"
    );
    let arr = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    // Count i32::MIN in output — must equal input.
    let out_min = (0..n).filter(|i| arr.value(*i) == i32::MIN).count();
    assert_eq!(
        out_min,
        target_count,
        "Stage-3 DESC: lost {} i32::MIN rows under sentinel collision",
        target_count - out_min
    );
    // And the whole output is DESC.
    for i in 1..n {
        assert!(
            arr.value(i - 1) >= arr.value(i),
            "DESC monotonic violated at row {i}: {} < {}",
            arr.value(i - 1),
            arr.value(i)
        );
    }
    // i32::MIN values must be the trailing run under DESC.
    let last_idx = n - 1;
    assert_eq!(
        arr.value(last_idx),
        i32::MIN,
        "DESC: last row must be i32::MIN (the smallest value present)"
    );
}

/// Stage 4 — `ORDER BY s ASC` over a plain Utf8 (StringArray) column.
/// Drives the inline-dictionary builder in `gpu_sort::host_values_for_key`
/// via the public SQL path. 16k rows with 16 distinct strings exercises
/// the low-cardinality case where the dict-build cost is amortised across
/// many rows.
#[test]
#[ignore = "gpu:sort"]
fn utf8_sort_via_inline_dictionary() {
    use arrow_array::StringArray;
    let mut engine = Engine::new().expect("ctx");
    let n = N_BIG; // 16_384 — above GPU_SORT_MIN_ROWS so the GPU path engages.
                   // 16 distinct strings, cycled. Pick names with mixed lengths and
                   // mixed alphabetic position so the lex order isn't a no-op of the
                   // dict-build order.
    let alphabet = [
        "delta", "alpha", "echo", "bravo", "foxtrot", "charlie", "golf", "india", "hotel", "kilo",
        "juliet", "mike", "lima", "november", "oscar", "papa",
    ];
    let strings: Vec<String> = (0..n)
        .map(|i| alphabet[i % alphabet.len()].to_string())
        .collect();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        ArrowDataType::Utf8,
        false,
    )]));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(strings.clone()))]).unwrap();
    engine.register_table("t", batch).unwrap();

    let h = engine
        .sql("SELECT s FROM t ORDER BY s")
        .expect("ORDER BY s ASC over Utf8");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), n);

    let arr = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("StringArray");
    // Strictly non-decreasing lex order.
    for i in 1..n {
        let a = arr.value(i - 1);
        let b = arr.value(i);
        assert!(a <= b, "ORDER BY s ASC violated at row {i}: {a:?} > {b:?}");
    }
    // Result must be a true permutation of the input.
    let mut expected: Vec<String> = strings;
    expected.sort();
    let actual: Vec<String> = (0..n).map(|i| arr.value(i).to_string()).collect();
    assert_eq!(actual, expected, "Utf8 ASC sort must equal sorted(input)");
}

/// Below the GPU threshold the host path must still produce correct output.
/// This test guards against an accidental gate inversion that would route
/// small queries through the GPU and break on its preconditions.
#[test]
#[ignore = "gpu:sort"]
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

    let arr = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let actual: Vec<i32> = (0..100).map(|i| arr.value(i)).collect();
    let mut expected = values;
    expected.sort();
    assert_eq!(actual, expected);
}

// ============================================================================
// Stage 5 — adaptive high-cardinality Utf8 gate. Above the threshold the GPU
// path returns Ok(None) from `host_values_for_key`'s Utf8 arm and the
// executor falls through to `lexsort_to_indices`. The result must still be
// a correct sort.
// ============================================================================

/// `ORDER BY s ASC` over a 16k-row Utf8 column with every row distinct.
/// The Stage-5 sample-and-fallback gate must abort the GPU path; the host
/// `lexsort_to_indices` then produces the sort. We can't directly observe
/// the dispatch decision from outside, but we *can* assert the result is
/// correct — which is the load-bearing property.
#[test]
#[ignore = "gpu:sort"]
fn high_cardinality_utf8_e2e_via_host_path() {
    use arrow_array::StringArray;
    let mut engine = Engine::new().expect("ctx");
    // N_BIG (16 384) rows, every one a distinct string. The row count
    // clears `GPU_SORT_MIN_ROWS` (otherwise `try_gpu_sort` would skip on
    // the size gate before reaching the Stage-5 cardinality check), and
    // the all-distinct strings push the sampler ratio to ~1.0 — well past
    // `HIGH_CARDINALITY_THRESHOLD = 0.6` — so the GPU path returns
    // `Ok(None)` and the executor falls through to `lexsort_to_indices`.
    let n = N_BIG;
    let strings: Vec<String> = (0..n).map(|i| format!("payload_{i:010}")).collect();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        ArrowDataType::Utf8,
        false,
    )]));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(strings.clone()))]).unwrap();
    engine.register_table("t", batch).unwrap();

    let h = engine
        .sql("SELECT s FROM t ORDER BY s")
        .expect("ORDER BY s ASC over high-cardinality Utf8");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), n);

    let arr = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("StringArray");
    // Non-decreasing lex order.
    for i in 1..n {
        let a = arr.value(i - 1);
        let b = arr.value(i);
        assert!(
            a <= b,
            "high-cardinality ORDER BY s ASC violated at row {i}: {a:?} > {b:?}"
        );
    }
    // Result is a true permutation of the input.
    let mut expected: Vec<String> = strings;
    expected.sort();
    let actual: Vec<String> = (0..n).map(|i| arr.value(i).to_string()).collect();
    assert_eq!(
        actual, expected,
        "high-cardinality Utf8 sort must equal sorted(input)"
    );
}

// ============================================================================
// Stage 6 — DictionaryArray<Int32, Utf8> end-to-end via the full SQL pipeline.
// The native ingest path (DictRegistry + GpuTable::upload_dict_utf8) is now
// the only route a dict-encoded column takes through the engine, so these
// tests exercise validity preservation across upload + sort + download.
// ============================================================================

/// Build a single-column `DictionaryArray<Int32, Utf8>` with a known mix of
/// values and NULLs. Layout:
///   row 0: "b"
///   row 1: NULL
///   row 2: "a"
///   row 3: "c"
///   row 4: NULL
///   row 5: "a"
///
/// Three distinct non-null values plus two NULLs is small enough to verify
/// the sorted output by inspection.
fn dict_utf8_with_nulls() -> arrow_array::DictionaryArray<arrow_array::types::Int32Type> {
    use arrow_array::builder::StringDictionaryBuilder;
    let mut b: StringDictionaryBuilder<arrow_array::types::Int32Type> =
        StringDictionaryBuilder::new();
    b.append_value("b");
    b.append_null();
    b.append_value("a");
    b.append_value("c");
    b.append_null();
    b.append_value("a");
    b.finish()
}

/// Build a `RecordBatch` whose single column `col` is the dict-encoded
/// fixture above.
fn dict_batch() -> RecordBatch {
    let arr = dict_utf8_with_nulls();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "col",
        ArrowDataType::Dictionary(
            Box::new(ArrowDataType::Int32),
            Box::new(ArrowDataType::Utf8),
        ),
        true,
    )]));
    RecordBatch::try_new(schema, vec![Arc::new(arr)]).expect("build batch")
}

/// Decode the output `RecordBatch`'s first column into a `Vec<Option<String>>`
/// for comparison. Handles both `StringArray` and `DictionaryArray<Int32, Utf8>`
/// output shapes — the engine may downconvert on read, depending on the sort
/// path's output mode.
fn decode_utf8_column(batch: &RecordBatch) -> Vec<Option<String>> {
    use arrow_array::{DictionaryArray, StringArray};
    let arr = batch.column(0);
    if let Some(sa) = arr.as_any().downcast_ref::<StringArray>() {
        return (0..sa.len())
            .map(|i| {
                if sa.is_null(i) {
                    None
                } else {
                    Some(sa.value(i).to_string())
                }
            })
            .collect();
    }
    if let Some(da) = arr
        .as_any()
        .downcast_ref::<DictionaryArray<arrow_array::types::Int32Type>>()
    {
        let values = da
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("dict values must be Utf8");
        let keys = da.keys();
        return (0..da.len())
            .map(|i| {
                if keys.is_null(i) {
                    None
                } else {
                    let k = keys.value(i) as usize;
                    if values.is_null(k) {
                        None
                    } else {
                        Some(values.value(k).to_string())
                    }
                }
            })
            .collect();
    }
    panic!(
        "unexpected output column dtype {:?} — expected Utf8 or Dictionary<Int32, Utf8>",
        arr.data_type()
    );
}

/// Stage 6 — `ORDER BY col ASC NULLS LAST` on a Dict-encoded column with
/// NULLs: NULLs sort to the end, real values ascend lex.
#[test]
#[ignore = "gpu:sort"]
fn order_by_dict_utf8_nulls_last() {
    let mut engine = Engine::new().expect("engine");
    engine.register_table("t", dict_batch()).expect("register");

    let h = engine
        .sql("SELECT col FROM t ORDER BY col NULLS LAST")
        .expect("sql");
    let batch = h.record_batch();
    let got = decode_utf8_column(batch);

    let want = vec![
        Some("a".to_string()),
        Some("a".to_string()),
        Some("b".to_string()),
        Some("c".to_string()),
        None,
        None,
    ];
    assert_eq!(got, want);
}

/// Stage 6 — `ORDER BY col ASC NULLS FIRST` on a Dict-encoded column with
/// NULLs: NULLs sort to the front, real values ascend lex.
#[test]
#[ignore = "gpu:sort"]
fn order_by_dict_utf8_nulls_first() {
    let mut engine = Engine::new().expect("engine");
    engine.register_table("t", dict_batch()).expect("register");

    let h = engine
        .sql("SELECT col FROM t ORDER BY col NULLS FIRST")
        .expect("sql");
    let batch = h.record_batch();
    let got = decode_utf8_column(batch);

    let want = vec![
        None,
        None,
        Some("a".to_string()),
        Some("a".to_string()),
        Some("b".to_string()),
        Some("c".to_string()),
    ];
    assert_eq!(got, want);
}
