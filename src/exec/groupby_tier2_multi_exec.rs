// SPDX-License-Identifier: Apache-2.0

//! Tier-2 hash-partitioned GROUP BY executor for **multiple SUM** aggregates
//! (top-level shim).
//!
//! Sibling of [`crate::exec::groupby_tier2_exec`]; composes:
//! - shape eligibility (single Int32 key, 1..=4 SUM(Float64) aggregates)
//! - cardinality gating via [`crate::exec::groupby_tier2_dispatch`]
//! - the multi-SUM orchestrator and merger
//!
//! Returns `Some(Ok(batch))` on success, `None` on eligibility miss (so the
//! caller layers fast paths uniformly: first to return `Some(_)` wins),
//! `Some(Err(_))` only on a genuine GPU/build failure encountered after we
//! committed to the path.
//!
//! Target query: h2o.ai q2 (`SELECT id2, SUM(v1), SUM(v2) FROM x GROUP BY id2`)
//! at medium-to-high cardinality.

use arrow_array::{Float64Array, Int32Array, RecordBatch};

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::groupby_tier2_dispatch::{
    dispatch_v2, AggOp, DispatchInputsV2, GroupByStrategyV2,
};
use crate::exec::groupby_tier2_multi_merge::build_tier2_multi_result;
use crate::exec::groupby_tier2_multi_orchestrator::execute_tier2_multi_sum;
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr};
use crate::plan::physical_plan::PhysicalPlan;

/// Maximum number of SUM aggregates this fast path will accept in one launch.
/// Beyond this we fall through to the next strategy. Matches the v0 scope
/// the orchestrator advertises (1..=4).
pub const MAX_VALS: usize = 4;

/// Try the Tier-2 multi-SUM fast path. Returns `None` on any precondition
/// miss so the caller falls through to the next strategy.
pub fn try_execute(
    plan: &PhysicalPlan,
    batch: &RecordBatch,
) -> Option<BoltResult<RecordBatch>> {
    // --- Plan-shape eligibility ------------------------------------------
    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        _ => return None,
    };
    if pre.is_some() {
        return None;
    }
    if aggregate.group_by.len() != 1 {
        return None;
    }
    let n_vals = aggregate.aggregates.len();
    if n_vals == 0 || n_vals > MAX_VALS {
        return None;
    }

    // The single group-by column must be Int32.
    let key_io_idx = aggregate.group_by[0];
    let key_io = match aggregate.inputs.get(key_io_idx) {
        Some(io) if io.dtype == DataType::Int32 => io,
        _ => return None,
    };

    // Every aggregate must be SUM(<bare-column>). Collect names up-front for
    // a single batch lookup pass.
    let mut sum_col_names: Vec<&str> = Vec::with_capacity(n_vals);
    for agg in &aggregate.aggregates {
        let name = match agg {
            AggregateExpr::Sum(Expr::Column(n)) => n.as_str(),
            _ => return None,
        };
        sum_col_names.push(name);
    }

    // Look up key + value arrays. Every value column must be Float64.
    let key_arr = batch
        .column_by_name(&key_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;
    let mut val_arrs: Vec<&Float64Array> = Vec::with_capacity(n_vals);
    for name in &sum_col_names {
        let arr = batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())?;
        if arr.len() != key_arr.len() {
            return None;
        }
        val_arrs.push(arr);
    }

    let n_rows = key_arr.len();
    if n_rows < 256 * 1024 {
        // Precondition: n_rows >= 256K (matches TIER2_MIN_ROWS in the
        // dispatcher; restated here so eligibility is auditable without
        // tracing through the dispatcher).
        return None;
    }

    // --- Range check on keys ---------------------------------------------
    //
    // Cheap upper-bound estimator for n_groups via max(key) + 1. Same
    // strategy as the single-SUM Tier-2 exec — h2o.ai keys are dense from 0
    // so max-based is tight. Reject negatives and bound the upper end at
    // the Tier-2 dispatcher's cap.
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

    // Tier-1 multi-SUM owns max(key) <= 1024 — by precondition we require
    // max(key) > 1024 so this executor doesn't shadow the better fast path.
    if n_groups_est <= 1024 {
        return None;
    }
    // Tier-2.1 (pass-2-on-GPU) brought the multi-SUM fixed overhead down
    // to roughly the single-SUM Tier-2.1 floor. The historical 100 K-group
    // gate existed to keep host-HashMap pass-2 from regressing q2-class
    // workloads (10 K groups); with GPU pass-2 that gate is obsolete and
    // the floor moves back to Tier-1's cap (BLOCK_GROUPS = 1024), matching
    // the single-SUM dispatcher.
    //
    // The kernel itself doesn't care about absolute group count — it
    // walks each partition's slice with a grid-stride loop. The only
    // residual concern is fixed setup cost (one partition kernel + N
    // scatter launches + one reduce launch), which empirically amortises
    // by ~5 K groups at h2o.ai N=10 M. We pin to 1024 for symmetry.
    const MULTI_SUM_MIN_GROUPS: u32 = 1024;
    if n_groups_est < MULTI_SUM_MIN_GROUPS {
        return None;
    }
    // Tier-2 cap (matches TIER2_MAX_GROUPS in the dispatcher).
    if n_groups_est >= 100_000_000 {
        return None;
    }

    // --- Final dispatcher gate -------------------------------------------
    //
    // The shared dispatcher only knows about single-aggregate queries; we
    // re-use it because every aggregate here has the same op+dtype (SUM +
    // Float64), so a single dispatch call covers all N.
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

    Some(execute_inner(plan, key_arr, &val_arrs))
}

/// All the fallible launch + result-marshal work, factored out so
/// `try_execute` cleanly returns `Option<Result<_>>`.
fn execute_inner(
    plan: &PhysicalPlan,
    key_arr: &Int32Array,
    val_arrs: &[&Float64Array],
) -> BoltResult<RecordBatch> {
    let n_rows = key_arr.len() as u32;

    // Upload key column + each value column independently. Sharing a single
    // buffer would require concatenation; the scatter kernel reads per-row
    // from one `vals_ptr` so independent buffers are the natural shape.
    let keys_gpu = GpuVec::<i32>::from_slice(key_arr.values())?;
    let mut vals_gpus: Vec<GpuVec<f64>> = Vec::with_capacity(val_arrs.len());
    for v in val_arrs {
        vals_gpus.push(GpuVec::<f64>::from_slice(v.values())?);
    }

    // Build the borrow slice the orchestrator wants. The orchestrator never
    // mutates these — `&[&GpuVec<f64>]` matches its signature exactly so we
    // don't need an extra trait or wrapper here.
    let vals_refs: Vec<&GpuVec<f64>> = vals_gpus.iter().collect();

    let partial = execute_tier2_multi_sum(&keys_gpu, &vals_refs, n_rows)?;

    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => {
            return Err(BoltError::Other(
                "groupby_tier2_multi_exec: non-Aggregate plan reached execute_inner".into(),
            ))
        }
    };

    build_tier2_multi_result(partial, &aggregate.output_schema)
}
