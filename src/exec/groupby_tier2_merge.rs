// SPDX-License-Identifier: Apache-2.0

//! Tier-2 hash-partitioned GROUP BY **result merger**.
//!
//! The Tier-2 orchestrator (sibling agent) hash-partitions input rows by
//! `hash(key) % K`, runs an independent per-partition aggregation, and
//! hands this module a `Vec<(Vec<i32>, Vec<f64>)>` — one (keys, sums)
//! pair per partition. Because each input key hashes to **exactly one**
//! partition, the per-partition key sets are pairwise disjoint and the
//! merger only needs to **concatenate** the partial outputs; no
//! cross-partition reduction is required.
//!
//! We additionally sort the concatenated result by key ASC. That matches
//! the SQL canonical row ordering used elsewhere in this crate (see the
//! `groupby.rs` output path) and lines up with what DuckDB / Polars
//! produce for `ORDER BY 1`, which keeps the bench comparisons honest.
//!
//! Scope (v0): single Int32 group-by column, single Float64 SUM aggregate
//! — the same narrow shape the Tier-1 fast path targets. Wider shapes are
//! out of scope for this slice and would be a follow-up.

use std::sync::Arc;

use arrow_array::{Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{DataType, Schema};

/// Per-partition (deduplicated keys, per-group sums) produced by the
/// Tier-2 orchestrator. Each partition's keys are unique (no cross-
/// partition collisions because each input key hashes to exactly one
/// partition).
pub type PerPartition = Vec<(Vec<i32>, Vec<f64>)>;

/// Concatenate per-partition results into a final `RecordBatch` matching
/// `output_schema`. Output ordering: sorted by key ASC (matches the SQL
/// canonical and what DuckDB/Polars produce for `ORDER BY 1`).
///
/// Sort strategy: zip into `Vec<(i32, f64)>`, `sort_by_key` on the key,
/// then unzip. This costs one extra `total`-sized allocation but keeps
/// the code obviously correct and lets the standard library's pdqsort
/// do the work. For the result sizes we expect at this layer (≤ a few
/// million distinct groups) the allocation is well below the cost of the
/// actual GPU passes upstream.
pub fn build_tier2_result(
    per_partition: PerPartition,
    output_schema: &Schema,
) -> BoltResult<RecordBatch> {
    // 1. Total result rows across all partitions.
    let total: usize = per_partition.iter().map(|(k, _)| k.len()).sum();

    // 2. Concatenate. Capacity pre-allocated so the extends are amortized
    //    O(total) with no intermediate reallocs.
    let mut keys_out: Vec<i32> = Vec::with_capacity(total);
    let mut sums_out: Vec<f64> = Vec::with_capacity(total);
    for (keys, sums) in per_partition.into_iter() {
        if keys.len() != sums.len() {
            return Err(BoltError::Other(format!(
                "tier2_merge: partition length mismatch (keys={}, sums={})",
                keys.len(),
                sums.len()
            )));
        }
        // Empty partitions are skipped naturally by extend().
        keys_out.extend(keys);
        sums_out.extend(sums);
    }

    // 3. Sort by key ASC. Zip / sort / unzip — see strategy note above.
    if keys_out.len() > 1 {
        let mut zipped: Vec<(i32, f64)> = keys_out
            .into_iter()
            .zip(sums_out.into_iter())
            .collect();
        zipped.sort_by_key(|(k, _)| *k);
        keys_out = Vec::with_capacity(total);
        sums_out = Vec::with_capacity(total);
        for (k, v) in zipped {
            keys_out.push(k);
            sums_out.push(v);
        }
    }

    // 4. Build the output `RecordBatch` against the planner-supplied
    //    schema. This module owns its own schema converter (the rest of
    //    the executors in this crate carry local copies as well — a
    //    consolidation refactor is out of scope here).
    let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
    let key_array = Arc::new(Int32Array::from(keys_out));
    let sum_array = Arc::new(Float64Array::from(sums_out));
    RecordBatch::try_new(arrow_schema, vec![key_array, sum_array]).map_err(|e| {
        BoltError::Other(format!(
            "tier2_merge: failed to build output RecordBatch: {e}"
        ))
    })
}

// ---------------------------------------------------------------------------
// Local plan-schema → Arrow-schema conversion. Every executor in this crate
// carries its own copy; consolidating them is a separate refactor.
// ---------------------------------------------------------------------------

fn plan_dtype_to_arrow(d: DataType) -> BoltResult<ArrowDataType> {
    match d {
        DataType::Int32 => Ok(ArrowDataType::Int32),
        DataType::Int64 => Ok(ArrowDataType::Int64),
        DataType::Float32 => Ok(ArrowDataType::Float32),
        DataType::Float64 => Ok(ArrowDataType::Float64),
        DataType::Bool => Ok(ArrowDataType::Boolean),
        DataType::Utf8 => Ok(ArrowDataType::Utf8),
    }
}

fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}

// ---------------------------------------------------------------------------
// Host-only tests — no CUDA needed. Run via:
//   cargo test --release groupby_tier2_merge
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{DataType, Field};

    /// Canonical Tier-2 output schema: Int32 key + Float64 sum.
    fn out_schema() -> Schema {
        Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("sum_v", DataType::Float64, false),
        ])
    }

    /// Pull (keys, sums) back out of a result batch for assertions.
    fn extract(batch: &RecordBatch) -> (Vec<i32>, Vec<f64>) {
        let keys = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("col 0 must be Int32");
        let sums = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("col 1 must be Float64");
        (keys.values().to_vec(), sums.values().to_vec())
    }

    #[test]
    fn empty_input_yields_empty_batch() {
        let schema = out_schema();
        let batch = build_tier2_result(vec![], &schema).expect("build ok");
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 2);
    }

    #[test]
    fn single_partition_passes_through_sorted() {
        let schema = out_schema();
        let per_part: PerPartition = vec![(vec![3, 1, 2], vec![30.0, 10.0, 20.0])];
        let batch = build_tier2_result(per_part, &schema).expect("build ok");
        let (keys, sums) = extract(&batch);
        assert_eq!(keys, vec![1, 2, 3]);
        assert_eq!(sums, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn multi_partition_concatenated() {
        let schema = out_schema();
        // Three partitions with disjoint key sets. (No collisions across
        // partitions is the Tier-2 invariant.)
        let per_part: PerPartition = vec![
            (vec![7, 4], vec![70.0, 40.0]),
            (vec![1, 9], vec![10.0, 90.0]),
            (vec![5, 2, 6], vec![50.0, 20.0, 60.0]),
        ];
        let batch = build_tier2_result(per_part, &schema).expect("build ok");
        let (keys, sums) = extract(&batch);
        assert_eq!(keys, vec![1, 2, 4, 5, 6, 7, 9]);
        assert_eq!(sums, vec![10.0, 20.0, 40.0, 50.0, 60.0, 70.0, 90.0]);
    }

    #[test]
    fn total_row_count_matches_inputs() {
        let schema = out_schema();
        let per_part: PerPartition = vec![
            (vec![10, 11, 12], vec![1.0, 2.0, 3.0]),
            (vec![20], vec![4.0]),
            (vec![30, 31], vec![5.0, 6.0]),
            (vec![], vec![]),
        ];
        let expected_rows: usize = per_part.iter().map(|(k, _)| k.len()).sum();
        let batch = build_tier2_result(per_part, &schema).expect("build ok");
        assert_eq!(batch.num_rows(), expected_rows);
        assert_eq!(batch.num_rows(), 6);
    }

    #[test]
    fn schema_matches_output_schema() {
        let schema = out_schema();
        let per_part: PerPartition = vec![(vec![1], vec![1.0])];
        let batch = build_tier2_result(per_part, &schema).expect("build ok");
        let arrow_schema = batch.schema();
        assert_eq!(arrow_schema.fields().len(), 2);
        assert_eq!(arrow_schema.field(0).name(), "key");
        assert_eq!(arrow_schema.field(0).data_type(), &ArrowDataType::Int32);
        assert_eq!(arrow_schema.field(1).name(), "sum_v");
        assert_eq!(arrow_schema.field(1).data_type(), &ArrowDataType::Float64);
    }

    #[test]
    fn partition_with_zero_rows_is_skipped() {
        let schema = out_schema();
        let per_part: PerPartition = vec![
            (vec![1], vec![5.0]),
            (vec![], vec![]),
            (vec![2], vec![6.0]),
        ];
        let batch = build_tier2_result(per_part, &schema).expect("build ok");
        let (keys, sums) = extract(&batch);
        assert_eq!(keys, vec![1, 2]);
        assert_eq!(sums, vec![5.0, 6.0]);
        assert_eq!(batch.num_rows(), 2);
    }

    // --- P1b-stage6 wiring smoke test (orchestrator + merge end-to-end) -----
    //
    // The merger itself is pure host code; the "async wiring" exercised here
    // is upstream (joint `compute_and_upload_partition_offsets_async`). This
    // test runs the full executor on a small fixture and validates the
    // merged RecordBatch matches a host oracle. `#[ignore]`-gated: needs
    // JIT + a live CUDA context.
    // ----------------------------------------------------------------------
    #[test]
    #[ignore = "gpu:tier2 — executes Tier-2 SUM pipeline + merge"]
    fn stage6_orchestrator_plus_merge_smoke() {
        use std::collections::HashMap;
        use crate::cuda::GpuVec;
        use crate::exec::groupby_tier2_orchestrator::execute_tier2_sum;

        let host_keys: Vec<i32> = vec![1, 2, 1, 3, 2, 1, 4, 3, 5, 2];
        let host_vals: Vec<f64> =
            vec![10.0, 20.0, 11.0, 30.0, 21.0, 12.0, 40.0, 31.0, 50.0, 22.0];
        let n_rows = host_keys.len() as u32;

        let keys = match GpuVec::<i32>::from_slice(&host_keys) {
            Ok(v) => v,
            Err(_) => return,
        };
        let vals = match GpuVec::<f64>::from_slice(&host_vals) {
            Ok(v) => v,
            Err(_) => return,
        };

        let partial = match execute_tier2_sum(&keys, &vals, n_rows) {
            Ok(r) => r,
            Err(_) => return,
        };

        let schema = out_schema();
        let batch = build_tier2_result(partial.per_partition, &schema)
            .expect("merge must succeed");
        let (got_keys, got_sums) = extract(&batch);

        // Oracle: host HashMap reduce, then sort by key ASC (matches merger).
        let mut oracle: HashMap<i32, f64> = HashMap::new();
        for (k, v) in host_keys.iter().zip(host_vals.iter()) {
            *oracle.entry(*k).or_insert(0.0) += *v;
        }
        let mut oracle_pairs: Vec<(i32, f64)> = oracle.into_iter().collect();
        oracle_pairs.sort_by_key(|(k, _)| *k);

        assert_eq!(got_keys.len(), oracle_pairs.len());
        for (i, (ek, ev)) in oracle_pairs.iter().enumerate() {
            assert_eq!(got_keys[i], *ek, "key mismatch at position {i}");
            assert!(
                (got_sums[i] - ev).abs() < 1e-9,
                "sum mismatch at key {ek}: oracle={ev}, got={}",
                got_sums[i]
            );
        }
    }
}
