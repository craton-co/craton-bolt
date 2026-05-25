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

use arrow_array::{Float64Array, Int32Array, RecordBatch};

use crate::cuda::GpuVec;
use crate::error::{JavelinError, JavelinResult};
use crate::exec::groupby_tier2_dispatch::{
    dispatch_v2, AggOp, DispatchInputsV2, GroupByStrategyV2,
};
use crate::exec::groupby_tier2_merge::build_tier2_result;
use crate::exec::groupby_tier2_orchestrator::execute_tier2_sum;
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr};
use crate::plan::physical_plan::PhysicalPlan;

/// Try the Tier-2 fast path. Returns `None` on any precondition miss so
/// the caller falls through to the next strategy.
pub fn try_execute(
    plan: &PhysicalPlan,
    batch: &RecordBatch,
) -> Option<JavelinResult<RecordBatch>> {
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
    let mut max_key: i32 = -1;
    for &k in key_arr.values() {
        if k < 0 {
            return None;
        }
        if k > max_key {
            max_key = k;
        }
    }
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
) -> JavelinResult<RecordBatch> {
    let n_rows = key_arr.len() as u32;
    let keys_gpu = GpuVec::<i32>::from_slice(key_arr.values())?;
    let vals_gpu = GpuVec::<f64>::from_slice(val_arr.values())?;

    let partial = execute_tier2_sum(&keys_gpu, &vals_gpu, n_rows)?;

    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => {
            return Err(JavelinError::Other(
                "groupby_tier2_exec: non-Aggregate plan reached execute_inner".into(),
            ))
        }
    };

    build_tier2_result(partial.per_partition, &aggregate.output_schema)
}
