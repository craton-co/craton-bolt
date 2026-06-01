// SPDX-License-Identifier: Apache-2.0

//! Tier-2 hash-partitioned GROUP BY executor (top-level shim).
//!
//! Composes:
//! - eligibility via [`crate::exec::groupby_tier2_dispatch`]
//! - on-device pipeline via [`crate::exec::groupby_tier2_orchestrator`]
//! - result materialisation via [`crate::exec::groupby_tier2_merge`]
//!
//! The shape of this file mirrors [`crate::exec::groupby_shmem_exec`]'s
//! `try_execute` so the caller in `execute_groupby` can layer fast paths
//! uniformly: each returns `None` on eligibility miss; the first to return
//! `Some(_)` wins.

use arrow_array::{Array, Float64Array, Int32Array, RecordBatch};

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::groupby_tier2_dispatch::{
    dispatch_v2, AggOp, DispatchInputsV2, GroupByStrategyV2,
};
use crate::exec::groupby_tier2_merge::build_tier2_result;
use crate::exec::groupby_tier2_orchestrator::execute_tier2_sum;
use crate::exec::launch::CudaStream;
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr};
use crate::plan::physical_plan::PhysicalPlan;

/// Try the Tier-2 fast path. Returns `None` on any precondition miss so
/// the caller falls through to the next strategy.
pub fn try_execute(
    plan: &PhysicalPlan,
    batch: &RecordBatch,
) -> Option<BoltResult<RecordBatch>> {
    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        _ => return None,
    };
    if pre.is_some() {
        return None;
    }
    if aggregate.group_by.len() != 1 || aggregate.aggregates.len() != 1 {
        return None;
    }

    let key_io_idx = aggregate.group_by[0];
    let key_io = match aggregate.inputs.get(key_io_idx) {
        Some(io) if io.dtype == DataType::Int32 => io,
        _ => return None,
    };

    let sum_col_name = match &aggregate.aggregates[0] {
        AggregateExpr::Sum(Expr::Column(name)) => name.as_str(),
        _ => return None,
    };

    let key_arr = batch
        .column_by_name(&key_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;
    let val_arr = batch
        .column_by_name(sum_col_name)
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())?;
    if key_arr.len() != val_arr.len() {
        return None;
    }

    // GB-S1: NULL handling — this fast path reads `key_arr.values()` /
    // `val_arr.values()` straight off the Arrow data buffer, which carries
    // garbage bytes at NULL positions (folding in as 0 / synthesizing a
    // group-0 key). Defer NULL-bearing batches back to
    // `groupby::execute_groupby` → the global-atomic path, which consults
    // the validity bitmap. Mirrors the guard in
    // `groupby_tier2_twokey_exec::try_execute`.
    if key_arr.null_count() > 0 || val_arr.null_count() > 0 {
        return None;
    }

    let n_rows = key_arr.len();

    // Cheap host-side n_groups estimator: distinct keys via bitset. For
    // Tier-2 eligibility we only care about an UPPER bound to differentiate
    // Tier-1 ( ≤ 1024 ) vs Tier-2 ( 1025..100M ) vs GlobalAtomic. A two-pass
    // bitset-over-i32 would be O(2^32) memory; instead we use a HashSet of
    // distinct keys — a few-MB walk is fine on the host versus the GPU
    // pipeline cost we save.
    //
    // But that's expensive at 10M rows (~200ms). Cheaper proxy: take the
    // max key + 1 as an upper bound when keys are dense from 0, OR if max
    // exceeds Tier-2's cap, immediately reject. h2o.ai's id3 is well-bounded
    // (1M distinct values in [0, 1M)), so max-based estimate is fine.
    // dedup (tier2/shmem): the max-nonneg-key scan now lives in
    // `groupby_tier2_common`. `None` = saw a negative key → decline;
    // `Some(-1)` = empty input → decline (this executor's historical
    // empty-input behaviour).
    let max_key = crate::exec::groupby_tier2_common::scan_max_nonneg_key(key_arr.values())?;
    if max_key < 0 {
        return None;
    }
    let n_groups_est = (max_key as u32).saturating_add(1);

    // Tier-2 dispatcher gate.
    let inputs = DispatchInputsV2 {
        n_groups: n_groups_est,
        n_rows: n_rows as u32,
        n_key_cols: 1,
        op: AggOp::Sum,
        value_dtype: DataType::Float64,
        key_dtype: DataType::Int32,
    };
    if dispatch_v2(inputs) != GroupByStrategyV2::Tier2Partitioned {
        return None;
    }

    Some(execute_inner(plan, key_arr, val_arr))
}

fn execute_inner(
    plan: &PhysicalPlan,
    key_arr: &Int32Array,
    val_arr: &Float64Array,
) -> BoltResult<RecordBatch> {
    let n_rows = key_arr.len() as u32;
    // Stage-4 (P1b): mint a per-call stream so the input H2D uploads,
    // kernel launches inside the orchestrator, and final D2H share a
    // single ordering domain. Falls back to NULL if creation fails.
    let stream = CudaStream::null();
    let keys_gpu = GpuVec::<i32>::from_slice_async(key_arr.values(), stream.raw())?;
    let vals_gpu = GpuVec::<f64>::from_slice_async(val_arr.values(), stream.raw())?;

    let partial = execute_tier2_sum(&keys_gpu, &vals_gpu, n_rows)?;

    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => {
            return Err(BoltError::Other(
                "groupby_tier2_exec: non-Aggregate plan reached execute_inner".into(),
            ))
        }
    };

    build_tier2_result(partial.per_partition, &aggregate.output_schema)
}

// ---------------------------------------------------------------------------
// Stage-4 (P1b) async round-trip smoke test.
//
// Verifies that `execute_inner` produces correct per-group sums after the
// async memcpy + pinned D2H plumbing was layered in. Gated `#[ignore]`
// because it needs a live CUDA context; cargo test runs it explicitly with
// `--ignored`.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod stage4_tests {
    use super::*;
    use crate::plan::logical_plan::{AggregateExpr, Expr, Field, Schema};
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};

    #[test]
    #[ignore = "gpu:tier2"]
    fn async_tier2_sum_round_trip() {
        // 300 K rows, ~2 K distinct keys — comfortably above the row + group
        // floor that gates this executor.
        let n: usize = 300_000;
        let n_groups: usize = 2048;
        let keys: Vec<i32> = (0..n).map(|i| (i % n_groups) as i32).collect();
        let vals: Vec<f64> = (0..n).map(|i| i as f64).collect();

        // Closed-form expected sum: for each g in 0..n_groups, sum over
        // i in [0..n) with i % n_groups == g of `i as f64`.
        let mut expected = vec![0.0f64; n_groups];
        for (i, &k) in keys.iter().enumerate() {
            expected[k as usize] += vals[i];
        }

        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![
                    ColumnIO {
                        name: "k".into(),
                        dtype: DataType::Int32,
                    },
                    ColumnIO {
                        name: "v".into(),
                        dtype: DataType::Float64,
                    },
                ],
                group_by: vec![0],
                aggregates: vec![AggregateExpr::Sum(Expr::Column("v".into()))],
                output_schema: Schema::new(vec![
                    Field::new("k", DataType::Int32, false),
                    Field::new("sum_v", DataType::Float64, true),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        let key_arr = Int32Array::from(keys);
        let val_arr = Float64Array::from(vals);

        let out = match execute_inner(&plan, &key_arr, &val_arr) {
            Ok(b) => b,
            Err(_) => return, // no CUDA — skip rather than fail.
        };

        // Output rows are (k, sum_v). Verify by index lookup.
        let ks = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let vs = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..out.num_rows() {
            let k = ks.value(i);
            let v = vs.value(i);
            assert_eq!(v, expected[k as usize], "key={} mismatch", k);
        }
    }

    /// GB-S1: a NULL-bearing input (in the key OR the value column) must
    /// make `try_execute` return `None` so the dispatcher falls through to
    /// `groupby::execute_groupby`'s validity-aware global-atomic path,
    /// rather than reading garbage bytes at NULL positions off the raw
    /// Arrow data buffer. This is a host-only test — the guard fires before
    /// any GPU work, so it runs without CUDA. Mirrors the NULL-defer
    /// contract enforced in `groupby_tier2_twokey_exec`.
    #[test]
    fn try_execute_defers_null_bearing_input() {
        use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
        use std::sync::Arc;

        fn plan() -> PhysicalPlan {
            PhysicalPlan::Aggregate {
                table: "t".into(),
                pre: None,
                aggregate: AggregateSpec {
                    inputs: vec![
                        ColumnIO {
                            name: "k".into(),
                            dtype: DataType::Int32,
                        },
                        ColumnIO {
                            name: "v".into(),
                            dtype: DataType::Float64,
                        },
                    ],
                    group_by: vec![0],
                    aggregates: vec![AggregateExpr::Sum(Expr::Column("v".into()))],
                    output_schema: Schema::new(vec![
                        Field::new("k", DataType::Int32, false),
                        Field::new("sum_v", DataType::Float64, true),
                    ]),
                    input_has_validity: Vec::new(),
                },
            }
        }

        fn batch(keys: Int32Array, vals: Float64Array) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("k", arrow_schema::DataType::Int32, true),
                ArrowField::new("v", arrow_schema::DataType::Float64, true),
            ]));
            RecordBatch::try_new(schema, vec![Arc::new(keys), Arc::new(vals)])
                .expect("null-bearing test batch")
        }

        // NULL in the value column.
        let null_val = batch(
            Int32Array::from(vec![0, 1, 2]),
            Float64Array::from(vec![Some(1.0), None, Some(3.0)]),
        );
        assert!(
            try_execute(&plan(), &null_val).is_none(),
            "NULL value must defer to the global-atomic path"
        );

        // NULL in the key column.
        let null_key = batch(
            Int32Array::from(vec![Some(0), None, Some(2)]),
            Float64Array::from(vec![1.0, 2.0, 3.0]),
        );
        assert!(
            try_execute(&plan(), &null_key).is_none(),
            "NULL key must defer to the global-atomic path"
        );
    }
}
