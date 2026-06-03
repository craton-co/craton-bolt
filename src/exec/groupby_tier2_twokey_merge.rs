// SPDX-License-Identifier: Apache-2.0

//! Tier-2 hash-partitioned GROUP BY **two-key result merger**.
//!
//! This is the two-key sibling of [`crate::exec::groupby_tier2_merge`].
//! Where the single-key merger emits a 2-column `(Int32, Float64)` batch,
//! this one emits a 3-column `(Int32, Int32, Float64)` batch by unpacking
//! each i64 partial key back into its two Int32 halves.
//!
//! ## Unpacking convention (MUST match `groupby.rs::pack_keys`)
//!
//! `pack_keys` for `(Int32, Int32)` produces:
//!
//! ```text
//! packed_u64 = ((column_0 as u32 as u64) << 32) | (column_1 as u32 as u64)
//! packed_i64 = packed_u64 as i64
//! ```
//!
//! So column 0 lives in the **high** 32 bits and column 1 in the **low** 32
//! bits of the packed i64. The reverse mapping used here is:
//!
//! ```text
//! let u = packed as u64;
//! let key1 = (u >> 32) as u32 as i32;            // column 0 (high)
//! let key2 = (u & 0xFFFF_FFFF) as u32 as i32;     // column 1 (low)
//! ```
//!
//! Casting through `u32 as i32` is the canonical lossless round-trip for
//! Int32 → bitcast → Int32 (a straight `as i32` from `u64` would clip).
//!
//! Sort: ascending by `(key1, key2)`. Matches the SQL canonical row
//! ordering (multi-column ORDER BY with the leading column first), lines
//! up with what DuckDB / Polars produce for `ORDER BY 1, 2`, and keeps the
//! bench comparisons honest.
//!
//! Scope (v0): two Int32 group-by columns + single Float64 SUM. The
//! orchestrator constrains us to that shape; wider shapes are a follow-up.

use std::sync::Arc;

use arrow_array::{Float64Array, Int32Array, RecordBatch};

use crate::error::{BoltError, BoltResult};
use crate::exec::groupby_tier2_twokey_orchestrator::Tier2TwokeyPartial;
use crate::plan::logical_plan::Schema;

/// Unpack a single packed i64 key into its two Int32 halves.
///
/// Layout (matches `groupby.rs::pack_keys`):
///   * high 32 bits = column 0
///   * low  32 bits = column 1
#[inline]
fn unpack_i64(packed: i64) -> (i32, i32) {
    let u = packed as u64;
    let high = (u >> 32) as u32 as i32;
    let low = (u & 0xFFFF_FFFFu64) as u32 as i32;
    (high, low)
}

/// Concatenate per-partition results, unpack each i64 into two Int32
/// columns, sort ascending by `(key1, key2)`, and emit a final
/// `RecordBatch` matching `output_schema`.
pub fn build_tier2_twokey_result(
    partial: Tier2TwokeyPartial,
    output_schema: &Schema,
) -> BoltResult<RecordBatch> {
    // 1. Total result rows across all partitions.
    let total: usize = partial.per_partition.iter().map(|(k, _)| k.len()).sum();

    // 2. Concatenate. We unpack on the fly into separate `key1` / `key2`
    //    vectors so the subsequent sort can be a straight `sort_by_key`
    //    on a tuple of (key1, key2) without re-packing.
    let mut key1_out: Vec<i32> = Vec::with_capacity(total);
    let mut key2_out: Vec<i32> = Vec::with_capacity(total);
    let mut sums_out: Vec<f64> = Vec::with_capacity(total);
    for (keys_i64, sums) in partial.per_partition.into_iter() {
        if keys_i64.len() != sums.len() {
            return Err(BoltError::Other(format!(
                "tier2_twokey_merge: partition length mismatch (keys={}, sums={})",
                keys_i64.len(),
                sums.len()
            )));
        }
        for (k_packed, v) in keys_i64.into_iter().zip(sums.into_iter()) {
            let (k1, k2) = unpack_i64(k_packed);
            key1_out.push(k1);
            key2_out.push(k2);
            sums_out.push(v);
        }
    }

    // 3. Sort ascending by (key1, key2). Same zip/sort/unzip strategy the
    //    single-key merger uses — pdqsort over a single Vec is well below
    //    the GPU-pipeline cost upstream.
    if total > 1 {
        let mut zipped: Vec<(i32, i32, f64)> = Vec::with_capacity(total);
        for ((k1, k2), v) in key1_out
            .into_iter()
            .zip(key2_out.into_iter())
            .zip(sums_out.into_iter())
        {
            zipped.push((k1, k2, v));
        }
        zipped.sort_by(|a, b| {
            // (i32, i32) total order: leading key first, ties broken on
            // the trailing key. Float values play no part in the sort.
            (a.0, a.1).cmp(&(b.0, b.1))
        });
        key1_out = Vec::with_capacity(total);
        key2_out = Vec::with_capacity(total);
        sums_out = Vec::with_capacity(total);
        for (k1, k2, v) in zipped {
            key1_out.push(k1);
            key2_out.push(k2);
            sums_out.push(v);
        }
    }

    // 4. Build the output `RecordBatch` against the planner-supplied
    //    schema. Schema must have exactly 3 fields: (Int32, Int32, Float64).
    if output_schema.fields.len() != 3 {
        return Err(BoltError::Other(format!(
            "tier2_twokey_merge: output_schema must have 3 fields (key1, key2, sum), \
             got {}",
            output_schema.fields.len()
        )));
    }
    // dedup (tier2): shared converter in
    // `groupby_tier2_common::plan_schema_to_arrow_schema`.
    let arrow_schema =
        crate::exec::groupby_tier2_common::plan_schema_to_arrow_schema(output_schema)?;
    let k1_arr = Arc::new(Int32Array::from(key1_out));
    let k2_arr = Arc::new(Int32Array::from(key2_out));
    let sum_arr = Arc::new(Float64Array::from(sums_out));
    RecordBatch::try_new(arrow_schema, vec![k1_arr, k2_arr, sum_arr]).map_err(|e| {
        BoltError::Other(format!(
            "tier2_twokey_merge: failed to build output RecordBatch: {e}"
        ))
    })
}

// ---------------------------------------------------------------------------
// Host-only tests — no CUDA needed.
// ---------------------------------------------------------------------------
// v0.7: these arrow/plan aliases are used only by the #[cfg(test)] modules
// below; the non-test schema conversion now lives in exec::schema_convert.
// cfg(test)-gated so normal builds don't see an unused import.
#[cfg(test)]
use crate::plan::logical_plan::DataType;
#[cfg(test)]
use arrow_schema::DataType as ArrowDataType;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::Field;

    /// Canonical two-key Tier-2 output schema: (Int32, Int32, Float64).
    fn out_schema() -> Schema {
        Schema::new(vec![
            Field::new("id1", DataType::Int32, false),
            Field::new("id2", DataType::Int32, false),
            Field::new("sum_v", DataType::Float64, false),
        ])
    }

    /// Pull (key1, key2, sums) back out of a result batch.
    fn extract(batch: &RecordBatch) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let k1 = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("col 0 must be Int32");
        let k2 = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("col 1 must be Int32");
        let sums = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("col 2 must be Float64");
        (
            k1.values().to_vec(),
            k2.values().to_vec(),
            sums.values().to_vec(),
        )
    }

    /// Encode `(a, b)` the same way `groupby.rs::pack_keys` does so the
    /// test inputs match the production packing convention. A regression
    /// in `unpack_i64` would surface here before it ever reaches the
    /// merger's sort path.
    fn pack(a: i32, b: i32) -> i64 {
        let hi = (a as u32 as u64) << 32;
        let lo = b as u32 as u64;
        (hi | lo) as i64
    }

    #[test]
    fn unpack_roundtrip_basic() {
        let cases: &[(i32, i32)] = &[
            (0, 0),
            (1, 2),
            (-1, -1), // both halves all-ones
            (i32::MIN, 0),
            (0, i32::MIN),
            (i32::MAX, i32::MIN),
        ];
        for &(a, b) in cases {
            let packed = pack(a, b);
            assert_eq!(
                unpack_i64(packed),
                (a, b),
                "round-trip failed for ({a}, {b})"
            );
        }
    }

    #[test]
    fn empty_input_yields_empty_batch() {
        let schema = out_schema();
        let partial = Tier2TwokeyPartial {
            per_partition: vec![],
        };
        let batch = build_tier2_twokey_result(partial, &schema).expect("build ok");
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 3);
    }

    #[test]
    fn single_partition_unpacks_and_sorts() {
        let schema = out_schema();
        // Three packed pairs, deliberately out of (k1, k2) sort order.
        let packed = vec![pack(3, 1), pack(1, 2), pack(1, 1)];
        let sums = vec![30.0, 12.0, 11.0];
        let partial = Tier2TwokeyPartial {
            per_partition: vec![(packed, sums)],
        };
        let batch = build_tier2_twokey_result(partial, &schema).expect("build ok");
        let (k1, k2, s) = extract(&batch);
        assert_eq!(k1, vec![1, 1, 3]);
        assert_eq!(k2, vec![1, 2, 1]);
        assert_eq!(s, vec![11.0, 12.0, 30.0]);
    }

    #[test]
    fn multi_partition_concatenated_and_sorted() {
        let schema = out_schema();
        // Three partitions; sort must be by (k1, k2) ASC across the union.
        let partial = Tier2TwokeyPartial {
            per_partition: vec![
                (vec![pack(7, 1), pack(4, 9)], vec![70.0, 49.0]),
                (vec![pack(1, 5), pack(9, 2)], vec![15.0, 92.0]),
                (
                    vec![pack(5, 5), pack(2, 0), pack(6, 6)],
                    vec![55.0, 20.0, 66.0],
                ),
            ],
        };
        let batch = build_tier2_twokey_result(partial, &schema).expect("build ok");
        let (k1, k2, s) = extract(&batch);
        assert_eq!(k1, vec![1, 2, 4, 5, 6, 7, 9]);
        assert_eq!(k2, vec![5, 0, 9, 5, 6, 1, 2]);
        assert_eq!(s, vec![15.0, 20.0, 49.0, 55.0, 66.0, 70.0, 92.0]);
    }

    #[test]
    fn schema_matches_three_columns() {
        let schema = out_schema();
        let partial = Tier2TwokeyPartial {
            per_partition: vec![(vec![pack(1, 2)], vec![3.0])],
        };
        let batch = build_tier2_twokey_result(partial, &schema).expect("build ok");
        let arrow_schema = batch.schema();
        assert_eq!(arrow_schema.fields().len(), 3);
        assert_eq!(arrow_schema.field(0).data_type(), &ArrowDataType::Int32);
        assert_eq!(arrow_schema.field(1).data_type(), &ArrowDataType::Int32);
        assert_eq!(arrow_schema.field(2).data_type(), &ArrowDataType::Float64);
    }

    #[test]
    fn rejects_schema_with_wrong_arity() {
        let bad_schema = Schema::new(vec![
            Field::new("id1", DataType::Int32, false),
            Field::new("sum_v", DataType::Float64, false),
        ]);
        let partial = Tier2TwokeyPartial {
            per_partition: vec![(vec![pack(1, 2)], vec![3.0])],
        };
        let err = build_tier2_twokey_result(partial, &bad_schema)
            .expect_err("must reject 2-field schema");
        let msg = format!("{}", err);
        assert!(
            msg.contains("3 fields"),
            "error must explain the arity requirement: {msg}"
        );
    }

    #[test]
    fn ties_on_high_key_break_on_low_key() {
        // All rows share k1 = 5; sort must order by k2 ASC.
        let schema = out_schema();
        let partial = Tier2TwokeyPartial {
            per_partition: vec![(
                vec![pack(5, 9), pack(5, 1), pack(5, 4)],
                vec![59.0, 51.0, 54.0],
            )],
        };
        let batch = build_tier2_twokey_result(partial, &schema).expect("build ok");
        let (k1, k2, s) = extract(&batch);
        assert_eq!(k1, vec![5, 5, 5]);
        assert_eq!(k2, vec![1, 4, 9]);
        assert_eq!(s, vec![51.0, 54.0, 59.0]);
    }

    // --- P1b-stage6 wiring smoke test (orchestrator + merge end-to-end) -----
    //
    // The merger itself is pure host code; "async wiring" exercised here is
    // upstream (joint `compute_and_upload_partition_offsets_async`). Runs
    // the twokey orchestrator on a small fixture and validates the merged
    // RecordBatch matches a host oracle. `#[ignore]`-gated.
    // ----------------------------------------------------------------------
    #[test]
    #[ignore = "requires CUDA toolkit + JIT (executes Tier-2 twokey pipeline + merge)"]
    fn stage6_orchestrator_plus_merge_twokey_smoke() {
        use crate::cuda::GpuVec;
        use crate::exec::groupby_tier2_twokey_orchestrator::execute_tier2_twokey_sum;
        use std::collections::HashMap;

        // 8-row fixture with duplicate (k1, k2) pairs.
        let host_packed: Vec<i64> = vec![
            pack(1, 10),
            pack(2, 20),
            pack(1, 10),
            pack(3, 30),
            pack(2, 20),
            pack(1, 10),
            pack(4, 40),
            pack(3, 31),
        ];
        let host_vals: Vec<f64> = vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.25, 4.0, 3.1];
        let n_rows = host_packed.len() as u32;

        let keys = match GpuVec::<i64>::from_slice(&host_packed) {
            Ok(v) => v,
            Err(_) => return,
        };
        let vals = match GpuVec::<f64>::from_slice(&host_vals) {
            Ok(v) => v,
            Err(_) => return,
        };

        let partial = match execute_tier2_twokey_sum(&keys, &vals, n_rows) {
            Ok(p) => p,
            Err(_) => return,
        };

        let schema = out_schema();
        let batch = build_tier2_twokey_result(partial, &schema).expect("merge ok");
        let (got_k1, got_k2, got_s) = extract(&batch);

        // Oracle: unpack, reduce, sort by (k1, k2).
        let mut oracle: HashMap<(i32, i32), f64> = HashMap::new();
        for (kp, v) in host_packed.iter().zip(host_vals.iter()) {
            let (a, b) = unpack_i64(*kp);
            *oracle.entry((a, b)).or_insert(0.0) += *v;
        }
        let mut oracle_rows: Vec<((i32, i32), f64)> = oracle.into_iter().collect();
        oracle_rows.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(got_k1.len(), oracle_rows.len());
        for (i, ((a, b), v)) in oracle_rows.iter().enumerate() {
            assert_eq!(got_k1[i], *a, "k1 mismatch at row {i}");
            assert_eq!(got_k2[i], *b, "k2 mismatch at row {i}");
            assert!(
                (got_s[i] - v).abs() < 1e-9,
                "sum mismatch at ({a}, {b}): oracle={v}, got={}",
                got_s[i]
            );
        }
    }
}
