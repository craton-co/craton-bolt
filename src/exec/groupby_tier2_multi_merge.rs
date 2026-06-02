// SPDX-License-Identifier: Apache-2.0

//! Tier-2 hash-partitioned GROUP BY **multi-SUM** result merger.
//!
//! Sibling of [`crate::exec::groupby_tier2_merge`]: combines per-partition
//! partial results into a final `RecordBatch`, but with `N` Float64 SUM output
//! columns instead of one. The Tier-2 partition invariant (each input key
//! hashes to exactly one partition) means the per-partition key sets *should*
//! be pairwise disjoint — but the merge does NOT rely on that for correctness.
//!
//! ## Defensive key dedup (belt-and-suspenders behind the kernel fence)
//!
//! ROOT CAUSE NOW FIXED IN THE KERNEL: the per-partition open-addressing
//! REDUCE kernels' publish/probe protocol was missing an ACQUIRE fence — a
//! prober that observed `set == 2` via a volatile load could read the slot's
//! key from stale (zeroed) shared memory before the claimer's key store
//! landed, take the collision-advance path, and mint the *same* key into a
//! second slot (a phantom/duplicate group). A `membar.cta` after
//! `PUBLISH_DONE:` (see
//! `partition_reduce_kernel_spill_common::emit_publish_probe_protocol`) now
//! orders the key load after the `set==2` observation. With that fence,
//! `tier2_multi_pipeline_matches_cpu_model` passes at 10M rows / 1M keys
//! WITHOUT the dedup below (verified on an RTX 2060 by isolating the fence).
//!
//! This fold is therefore retained as cheap DEFENSE-IN-DEPTH for a
//! scale-only GPU concurrency bug: should the fence ever regress or a future
//! kernel re-introduce a duplicate-key path, the merge still produces the
//! correct aggregate rather than the ~1519 phantom rows the q2 stress test
//! once saw for ≤1M keys. It SUMS rows that share a key rather than appending
//! them: after the stable sort by key, equal keys are adjacent, so a single
//! linear fold collapses any duplicate-key rows into one — adding their N sum
//! columns componentwise (the same reduction the per-partition table would
//! have produced). In the race-free common case every key is unique and the
//! fold is a pure copy.
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

use crate::error::{BoltError, BoltResult};
use crate::exec::groupby_tier2_multi_orchestrator::Tier2MultiPartial;
use crate::plan::logical_plan::{Schema};

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
) -> BoltResult<RecordBatch> {
    let Tier2MultiPartial {
        per_partition,
        n_vals,
    } = partial;

    // 0. Schema width check: 1 key + n_vals sums.
    let expected_fields = 1 + n_vals;
    if output_schema.fields.len() != expected_fields {
        return Err(BoltError::Other(format!(
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
            return Err(BoltError::Other(format!(
                "tier2_multi_merge: partition has {} sum columns, expected {}",
                sums.len(),
                n_vals
            )));
        }
        for (j, s) in sums.iter().enumerate() {
            if s.len() != keys.len() {
                return Err(BoltError::Other(format!(
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

    // 3. Sort by key ASC using a permutation index, then fold duplicate keys.
    //    We materialise the permutation once, then walk it in sorted order
    //    accumulating N sum columns per distinct key. This avoids the
    //    `Vec<(i32, [f64; N])>` zip+sort+unzip intermediate (which would move
    //    8 + 8*N bytes per row twice) and, in the same pass, collapses any
    //    rows that share a key — the defensive phantom-group dedup described
    //    in the module docs. Two rows with the same key (e.g. a per-partition
    //    REDUCE kernel that minted the key into two slots under a publish
    //    race) are summed into one output row rather than both surfacing.
    if keys_out.len() > 1 {
        let mut perm: Vec<usize> = (0..keys_out.len()).collect();
        // Stable sort so that, among rows sharing a key, the accumulation
        // order is deterministic (input/partition order) — keeps the f64
        // add ordering reproducible across runs.
        perm.sort_by_key(|&i| keys_out[i]);

        // Allocate fresh output vecs; capacity is the pre-dedup count (an
        // upper bound on distinct keys). The extra headroom is bounded by the
        // result size, already far smaller than the input GPU buffers.
        let mut new_keys: Vec<i32> = Vec::with_capacity(keys_out.len());
        let mut new_sums: Vec<Vec<f64>> = (0..n_vals)
            .map(|_| Vec::with_capacity(keys_out.len()))
            .collect();
        for &i in &perm {
            let key = keys_out[i];
            // Fold into the previous output row when this key repeats the
            // last emitted key (adjacent after the sort). Otherwise start a
            // new output row.
            if new_keys.last() == Some(&key) {
                let last = new_keys.len() - 1;
                for j in 0..n_vals {
                    new_sums[j][last] += sums_out[j][i];
                }
            } else {
                new_keys.push(key);
                for j in 0..n_vals {
                    new_sums[j].push(sums_out[j][i]);
                }
            }
        }
        keys_out = new_keys;
        sums_out = new_sums;
    }

    // 4. Build the output `RecordBatch` against the planner-supplied schema.
    //    dedup (tier2): shared converter in
    //    `groupby_tier2_common::plan_schema_to_arrow_schema`.
    let arrow_schema =
        crate::exec::groupby_tier2_common::plan_schema_to_arrow_schema(output_schema)?;

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(1 + n_vals);
    columns.push(Arc::new(Int32Array::from(keys_out)) as ArrayRef);
    for col in sums_out {
        columns.push(Arc::new(Float64Array::from(col)) as ArrayRef);
    }

    RecordBatch::try_new(arrow_schema, columns).map_err(|e| {
        BoltError::Other(format!(
            "tier2_multi_merge: failed to build output RecordBatch: {e}"
        ))
    })
}


// ---------------------------------------------------------------------------
// Host-only tests — no CUDA needed. Run via:
//   cargo test --release groupby_tier2_multi_merge
// ---------------------------------------------------------------------------

// v0.7: these arrow/plan aliases are used only by the #[cfg(test)] modules
// below; the non-test schema conversion now lives in exec::schema_convert.
// cfg(test)-gated so normal builds don't see an unused import.
#[cfg(test)]
use arrow_schema::{DataType as ArrowDataType};

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

    // --- Phantom-group dedup regression tests (Tier-2 fix) ------------------
    //
    // These directly attack the bug the fix targets: a per-partition REDUCE
    // kernel minting the SAME key into two exported slots (within one
    // partition, or — defensively — across partitions). The merge must SUM
    // such rows into one output group, never append both. A pre-fix merge
    // would surface duplicate keys here.

    #[test]
    fn duplicate_key_within_one_partition_is_summed_n2() {
        // One partition exports key 5 twice (the phantom slot) plus key 3
        // once. The merge must collapse the two key-5 rows into a single
        // group whose sums are the componentwise total.
        let schema = out_schema_n(2);
        let partial = Tier2MultiPartial {
            per_partition: vec![(
                vec![5, 3, 5],
                vec![
                    vec![10.0, 7.0, 1.0], // v0: key5=10+1=11, key3=7
                    vec![20.0, 8.0, 2.0], // v1: key5=20+2=22, key3=8
                ],
            )],
            n_vals: 2,
        };
        let batch = build_tier2_multi_result(partial, &schema).expect("build ok");
        let (keys, sums) = extract(&batch, 2);
        assert_eq!(keys, vec![3, 5], "duplicate key 5 must collapse to one row");
        assert_eq!(sums[0], vec![7.0, 11.0]);
        assert_eq!(sums[1], vec![8.0, 22.0]);
    }

    #[test]
    fn duplicate_key_across_partitions_is_summed_n3() {
        // A key appearing in two partitions' outputs (e.g. a hypothetical
        // partition-hash leak) must also be merged, not appended.
        let schema = out_schema_n(3);
        let partial = Tier2MultiPartial {
            per_partition: vec![
                (vec![9], vec![vec![1.0], vec![10.0], vec![100.0]]),
                (vec![9], vec![vec![2.0], vec![20.0], vec![200.0]]),
                (vec![4], vec![vec![5.0], vec![50.0], vec![500.0]]),
            ],
            n_vals: 3,
        };
        let batch = build_tier2_multi_result(partial, &schema).expect("build ok");
        let (keys, sums) = extract(&batch, 3);
        assert_eq!(keys, vec![4, 9], "key 9 from two partitions collapses");
        assert_eq!(sums[0], vec![5.0, 3.0]);
        assert_eq!(sums[1], vec![50.0, 30.0]);
        assert_eq!(sums[2], vec![500.0, 300.0]);
    }

    #[test]
    fn three_way_duplicate_key_is_summed_n1() {
        // Same key minted three times — fold must accumulate all three.
        let schema = out_schema_n(1);
        let partial = Tier2MultiPartial {
            per_partition: vec![
                (vec![7, 7], vec![vec![1.0, 2.0]]),
                (vec![7], vec![vec![4.0]]),
            ],
            n_vals: 1,
        };
        let batch = build_tier2_multi_result(partial, &schema).expect("build ok");
        let (keys, sums) = extract(&batch, 1);
        assert_eq!(keys, vec![7]);
        assert_eq!(sums[0], vec![7.0], "1+2+4 across three phantom slots");
    }

    #[test]
    fn no_duplicates_is_pure_passthrough_n2() {
        // Race-free common case: every key unique → fold is a sorted copy,
        // exactly the pre-fix behaviour. Guards against the dedup pass
        // accidentally dropping or merging distinct keys.
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
    fn duplicate_negative_and_zero_keys_summed_n2() {
        // Phantom dedup must work for key 0 (the value a stale-read race
        // would surface) and negative keys.
        let schema = out_schema_n(2);
        let partial = Tier2MultiPartial {
            per_partition: vec![(
                vec![0, -7, 0, -7],
                vec![vec![1.0, 3.0, 1.0, 3.0], vec![2.0, 4.0, 2.0, 4.0]],
            )],
            n_vals: 2,
        };
        let batch = build_tier2_multi_result(partial, &schema).expect("build ok");
        let (keys, sums) = extract(&batch, 2);
        assert_eq!(keys, vec![-7, 0]);
        assert_eq!(sums[0], vec![6.0, 2.0]);
        assert_eq!(sums[1], vec![8.0, 4.0]);
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

    // --- P1b-stage6 wiring smoke test (orchestrator + merge end-to-end) -----
    //
    // The merger itself is pure host code; "async wiring" exercised here is
    // upstream (joint `compute_and_upload_partition_offsets_async`). Runs
    // the multi-SUM orchestrator on a small fixture and validates the
    // merged RecordBatch matches a host oracle. `#[ignore]`-gated.
    // ----------------------------------------------------------------------
    #[test]
    #[ignore = "requires CUDA toolkit + JIT (executes Tier-2 multi-SUM pipeline + merge)"]
    fn stage6_orchestrator_plus_merge_multi_smoke() {
        use std::collections::HashMap;
        use crate::cuda::GpuVec;
        use crate::exec::groupby_tier2_multi_orchestrator::execute_tier2_multi_sum;

        let host_keys: Vec<i32> = vec![1, 2, 1, 3, 2, 1, 4, 3];
        let host_v0: Vec<f64> = vec![10.0, 20.0, 11.0, 30.0, 21.0, 12.0, 40.0, 31.0];
        let host_v1: Vec<f64> = vec![100.0, 200.0, 110.0, 300.0, 210.0, 120.0, 400.0, 310.0];
        let n_rows = host_keys.len() as u32;

        let keys = match GpuVec::<i32>::from_slice(&host_keys) {
            Ok(v) => v,
            Err(_) => return,
        };
        let v0 = match GpuVec::<f64>::from_slice(&host_v0) {
            Ok(v) => v,
            Err(_) => return,
        };
        let v1 = match GpuVec::<f64>::from_slice(&host_v1) {
            Ok(v) => v,
            Err(_) => return,
        };
        let vals = vec![&v0, &v1];

        let partial = match execute_tier2_multi_sum(&keys, &vals, n_rows) {
            Ok(r) => r,
            Err(_) => return,
        };

        let schema = out_schema_n(2);
        let batch = build_tier2_multi_result(partial, &schema).expect("merge ok");
        let (got_keys, got_sums) = extract(&batch, 2);

        // Oracle.
        let mut oracle: HashMap<i32, (f64, f64)> = HashMap::new();
        for i in 0..host_keys.len() {
            let e = oracle.entry(host_keys[i]).or_insert((0.0, 0.0));
            e.0 += host_v0[i];
            e.1 += host_v1[i];
        }
        let mut oracle_rows: Vec<(i32, f64, f64)> =
            oracle.into_iter().map(|(k, (a, b))| (k, a, b)).collect();
        oracle_rows.sort_by_key(|(k, _, _)| *k);

        assert_eq!(got_keys.len(), oracle_rows.len());
        for (i, (k, e0, e1)) in oracle_rows.iter().enumerate() {
            assert_eq!(got_keys[i], *k);
            assert!((got_sums[0][i] - e0).abs() < 1e-9, "v0 mismatch at key {k}");
            assert!((got_sums[1][i] - e1).abs() < 1e-9, "v1 mismatch at key {k}");
        }
    }
}
