// SPDX-License-Identifier: Apache-2.0

//! Tier-2 hash-partitioned GROUP BY **multi-SUM** result merger.
//!
//! Sibling of [`crate::exec::groupby_tier2_merge`]: concatenates per-partition
//! partial results into a final `RecordBatch`, but with `N` Float64 SUM output
//! columns instead of one. The Tier-2 partition invariant (each input key
//! hashes to exactly one partition) still holds — the per-partition key sets
//! are pairwise disjoint, so concatenation suffices; no second-level reduce.
//!
//! Output ordering: sorted by key ASC (matches the SQL canonical and what
//! DuckDB/Polars produce for `ORDER BY 1`). The sort uses a `permutation`
//! over the concatenated keys so we only allocate one Vec<usize> rather than
//! a `Vec<(i32, [f64; N])>` — the per-row tuple would be `8 + 8*N` bytes and
//! we'd be moving it twice (sort + unpack).
//!
//! v0 scope: single Int32 group-by column, 1..=4 Float64 SUM aggregates. The
//! orchestrator validates `n_vals` at the entry point; here we just trust
//! the partial's shape and surface a clear error if the partition lengths
//! don't line up.

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::error::{PatinaError, PatinaResult};
use crate::exec::groupby_tier2_multi_orchestrator::Tier2MultiPartial;
use crate::plan::logical_plan::{DataType, Schema};

/// Concatenate per-partition multi-SUM partials into a final `RecordBatch`
/// matching `output_schema`.
///
/// `output_schema` must declare exactly `1 + partial.n_vals` fields: one
/// Int32 key field followed by `n_vals` Float64 SUM fields, in the order
/// the aggregates were declared on the plan. We surface a structured error
/// rather than panicking if the schema width disagrees.
pub fn build_tier2_multi_result(
    partial: Tier2MultiPartial,
    output_schema: &Schema,
) -> PatinaResult<RecordBatch> {
    let Tier2MultiPartial {
        per_partition,
        n_vals,
    } = partial;

    // 0. Schema width check: 1 key + n_vals sums.
    let expected_fields = 1 + n_vals;
    if output_schema.fields.len() != expected_fields {
        return Err(PatinaError::Other(format!(
            "tier2_multi_merge: output_schema has {} fields, expected {} (1 key + {} sums)",
            output_schema.fields.len(),
            expected_fields,
            n_vals
        )));
    }

    // 1. Total result rows across all partitions.
    let total: usize = per_partition.iter().map(|(k, _)| k.len()).sum();

    // 2. Concatenate. Capacity pre-allocated so the extends are amortised
    //    O(total).
    let mut keys_out: Vec<i32> = Vec::with_capacity(total);
    let mut sums_out: Vec<Vec<f64>> =
        (0..n_vals).map(|_| Vec::with_capacity(total)).collect();
    for (keys, sums) in per_partition.into_iter() {
        // Each partition must carry exactly n_vals inner sum columns, each
        // aligned to `keys`.
        if sums.len() != n_vals {
            return Err(PatinaError::Other(format!(
                "tier2_multi_merge: partition has {} sum columns, expected {}",
                sums.len(),
                n_vals
            )));
        }
        for (j, s) in sums.iter().enumerate() {
            if s.len() != keys.len() {
                return Err(PatinaError::Other(format!(
                    "tier2_multi_merge: partition sums[{}].len()={} != keys.len()={}",
                    j,
                    s.len(),
                    keys.len()
                )));
            }
        }
        keys_out.extend(&keys);
        for (j, s) in sums.into_iter().enumerate() {
            sums_out[j].extend(s);
        }
    }

    // 3. Sort by key ASC using a permutation index. We materialise the
    //    permutation once, then apply it to the key column and each of the
    //    N sum columns. This avoids the `Vec<(i32, [f64; N])>` zip+sort+unzip
    //    intermediate, which would move 8 + 8*N bytes per row twice and is
    //    measurably slower for N >= 2.
    if keys_out.len() > 1 {
        let mut perm: Vec<usize> = (0..keys_out.len()).collect();
        perm.sort_unstable_by_key(|&i| keys_out[i]);

        // Apply the permutation. We allocate fresh output vecs of the same
        // capacity and gather in permutation order; in-place permutation
        // would save the allocation but is fiddly with multiple columns of
        // differing types. The extra allocation is bounded by the result
        // size (already much smaller than the input GPU buffers).
        let mut new_keys: Vec<i32> = Vec::with_capacity(keys_out.len());
        let mut new_sums: Vec<Vec<f64>> = (0..n_vals)
            .map(|_| Vec::with_capacity(keys_out.len()))
            .collect();
        for &i in &perm {
            new_keys.push(keys_out[i]);
            for j in 0..n_vals {
                new_sums[j].push(sums_out[j][i]);
            }
        }
        keys_out = new_keys;
        sums_out = new_sums;
    }

    // 4. Build the output `RecordBatch` against the planner-supplied schema.
    //    Local copy of the converter — every executor in this crate carries
    //    its own; consolidating them is a separate refactor.
    let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(1 + n_vals);
    columns.push(Arc::new(Int32Array::from(keys_out)) as ArrayRef);
    for col in sums_out {
        columns.push(Arc::new(Float64Array::from(col)) as ArrayRef);
    }

    RecordBatch::try_new(arrow_schema, columns).map_err(|e| {
        PatinaError::Other(format!(
            "tier2_multi_merge: failed to build output RecordBatch: {e}"
        ))
    })
}

// ---------------------------------------------------------------------------
// Local plan-schema → Arrow-schema conversion. Per the crate convention.
// ---------------------------------------------------------------------------

fn plan_dtype_to_arrow(d: DataType) -> PatinaResult<ArrowDataType> {
    match d {
        DataType::Int32 => Ok(ArrowDataType::Int32),
        DataType::Int64 => Ok(ArrowDataType::Int64),
        DataType::Float32 => Ok(ArrowDataType::Float32),
        DataType::Float64 => Ok(ArrowDataType::Float64),
        DataType::Bool => Ok(ArrowDataType::Boolean),
        DataType::Utf8 => Ok(ArrowDataType::Utf8),
    }
}

fn plan_schema_to_arrow_schema(s: &Schema) -> PatinaResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}

// ---------------------------------------------------------------------------
// Host-only tests — no CUDA needed. Run via:
//   cargo test --release groupby_tier2_multi_merge
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{DataType, Field};

    fn out_schema_n(n_vals: usize) -> Schema {
        let mut fields = vec![Field::new("key", DataType::Int32, false)];
        for j in 0..n_vals {
            fields.push(Field::new(&format!("sum_v{j}"), DataType::Float64, false));
        }
        Schema::new(fields)
    }

    fn extract(batch: &RecordBatch, n_vals: usize) -> (Vec<i32>, Vec<Vec<f64>>) {
        let keys = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("col 0 must be Int32")
            .values()
            .to_vec();
        let mut sums: Vec<Vec<f64>> = Vec::with_capacity(n_vals);
        for j in 0..n_vals {
            let s = batch
                .column(1 + j)
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("sum col must be Float64")
                .values()
                .to_vec();
            sums.push(s);
        }
        (keys, sums)
    }

    #[test]
    fn empty_input_yields_empty_batch_n2() {
        let schema = out_schema_n(2);
        let partial = Tier2MultiPartial {
            per_partition: Vec::new(),
            n_vals: 2,
        };
        let batch = build_tier2_multi_result(partial, &schema).expect("build ok");
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 3);
    }

    #[test]
    fn single_partition_passes_through_sorted_n2() {
        let schema = out_schema_n(2);
        let partial = Tier2MultiPartial {
            per_partition: vec![(
                vec![3, 1, 2],
                vec![vec![30.0, 10.0, 20.0], vec![300.0, 100.0, 200.0]],
            )],
            n_vals: 2,
        };
        let batch = build_tier2_multi_result(partial, &schema).expect("build ok");
        let (keys, sums) = extract(&batch, 2);
        assert_eq!(keys, vec![1, 2, 3]);
        assert_eq!(sums[0], vec![10.0, 20.0, 30.0]);
        assert_eq!(sums[1], vec![100.0, 200.0, 300.0]);
    }

    #[test]
    fn multi_partition_concatenated_n3() {
        let schema = out_schema_n(3);
        let partial = Tier2MultiPartial {
            per_partition: vec![
                (vec![7, 4], vec![vec![70.0, 40.0], vec![71.0, 41.0], vec![72.0, 42.0]]),
                (vec![1, 9], vec![vec![10.0, 90.0], vec![11.0, 91.0], vec![12.0, 92.0]]),
                (
                    vec![5, 2, 6],
                    vec![
                        vec![50.0, 20.0, 60.0],
                        vec![51.0, 21.0, 61.0],
                        vec![52.0, 22.0, 62.0],
                    ],
                ),
            ],
            n_vals: 3,
        };
        let batch = build_tier2_multi_result(partial, &schema).expect("build ok");
        let (keys, sums) = extract(&batch, 3);
        assert_eq!(keys, vec![1, 2, 4, 5, 6, 7, 9]);
        assert_eq!(sums[0], vec![10.0, 20.0, 40.0, 50.0, 60.0, 70.0, 90.0]);
        assert_eq!(sums[1], vec![11.0, 21.0, 41.0, 51.0, 61.0, 71.0, 91.0]);
        assert_eq!(sums[2], vec![12.0, 22.0, 42.0, 52.0, 62.0, 72.0, 92.0]);
    }

    #[test]
    fn empty_partition_is_skipped() {
        let schema = out_schema_n(2);
        let partial = Tier2MultiPartial {
            per_partition: vec![
                (vec![1], vec![vec![5.0], vec![50.0]]),
                (vec![], vec![vec![], vec![]]),
                (vec![2], vec![vec![6.0], vec![60.0]]),
            ],
            n_vals: 2,
        };
        let batch = build_tier2_multi_result(partial, &schema).expect("build ok");
        let (keys, sums) = extract(&batch, 2);
        assert_eq!(keys, vec![1, 2]);
        assert_eq!(sums[0], vec![5.0, 6.0]);
        assert_eq!(sums[1], vec![50.0, 60.0]);
    }

    #[test]
    fn schema_width_mismatch_errors() {
        // n_vals=2 but only 2 fields in the schema (1 key + 1 sum).
        let schema = out_schema_n(1);
        let partial = Tier2MultiPartial {
            per_partition: vec![(vec![1], vec![vec![1.0], vec![2.0]])],
            n_vals: 2,
        };
        let r = build_tier2_multi_result(partial, &schema);
        assert!(r.is_err());
    }

    #[test]
    fn schema_matches_output_schema_n4() {
        let schema = out_schema_n(4);
        let partial = Tier2MultiPartial {
            per_partition: vec![(
                vec![1],
                vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
            )],
            n_vals: 4,
        };
        let batch = build_tier2_multi_result(partial, &schema).expect("build ok");
        let arrow_schema = batch.schema();
        assert_eq!(arrow_schema.fields().len(), 5);
        assert_eq!(arrow_schema.field(0).data_type(), &ArrowDataType::Int32);
        for j in 0..4 {
            assert_eq!(
                arrow_schema.field(1 + j).data_type(),
                &ArrowDataType::Float64
            );
        }
    }
}
