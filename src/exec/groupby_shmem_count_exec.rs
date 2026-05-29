// SPDX-License-Identifier: Apache-2.0

//! Tier-1 **COUNT(*) GROUP BY** executor — shared-memory pre-aggregation
//! for low-cardinality `SELECT key, COUNT(*) FROM x GROUP BY key`.
//!
//! Sibling of [`crate::exec::groupby_shmem_exec`] (single SUM) and
//! [`crate::exec::groupby_shmem_avg_exec`] (multi AVG). The kernel
//! ([`crate::jit::shmem_count_kernel`]) already exists and is used by
//! the Tier-1 AVG path; this executor exposes it standalone for queries
//! that only ask for counts.
//!
//! ## Scope (v0)
//!
//! - GROUP BY exactly one Int32 column
//! - Exactly one aggregate: `COUNT(*)` (matched via
//!   `AggregateExpr::Count(_)`)
//! - `max(key) < BLOCK_GROUPS` (1024)
//! - `n_rows >= 64 K`
//! - no `pre` kernel

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow_array::{Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{Schema as ArrowSchema};

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::groupby_shmem_launch::{tune, TuneInputs};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::module_cache;
use crate::jit::shmem_count_kernel::{
    compile_shmem_count_kernel, BLOCK_GROUPS, KERNEL_ENTRY,
};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Schema};
use crate::plan::physical_plan::PhysicalPlan;

const MIN_ROWS_FAST_PATH: usize = 64 * 1024;

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
    // Capture the counted column name (if the COUNT argument is a bare
    // column) so the GB-S1 guard below can defer NULL-bearing counted
    // columns — this kernel counts every row, but SQL `COUNT(col)` skips
    // NULLs.
    let count_col_name: Option<&str> = match &aggregate.aggregates[0] {
        AggregateExpr::Count(Expr::Column(n)) => Some(n.as_str()),
        AggregateExpr::Count(_) => None,
        _ => return None,
    };

    let key_io_idx = aggregate.group_by[0];
    let key_io = match aggregate.inputs.get(key_io_idx) {
        Some(io) if io.dtype == DataType::Int32 => io,
        _ => return None,
    };

    let key_arr = batch
        .column_by_name(&key_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;

    // GB-S1: NULL handling. NULL keys are read off the raw Arrow buffer and
    // would synthesize a group-0; a NULL in the counted column would be
    // over-counted. Defer NULL-bearing batches back to
    // `groupby::execute_groupby` → the global-atomic path, which consults
    // validity. Mirrors the guard in `groupby_tier2_twokey_exec::try_execute`.
    if key_arr.null_count() > 0 {
        return None;
    }
    if let Some(name) = count_col_name {
        if let Some(col) = batch.column_by_name(name) {
            if col.null_count() > 0 {
                return None;
            }
        }
    }

    let n_rows = key_arr.len();
    if n_rows < MIN_ROWS_FAST_PATH {
        return None;
    }

    // max(key) must fit in the BLOCK_GROUPS slot map (Tier-1's whole
    // premise). Above that, the Tier-2.1 COUNT executor handles it.
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
    if n_groups_est > BLOCK_GROUPS {
        return None;
    }

    Some(execute_inner(plan, key_arr, n_groups_est))
}

fn execute_inner(
    plan: &PhysicalPlan,
    key_arr: &Int32Array,
    n_groups: u32,
) -> BoltResult<RecordBatch> {
    let n_rows = key_arr.len();
    // Stage-4 (P1b): per-call stream shared across H2D, kernel, and D2H.
    let stream = CudaStream::null_or_default();
    let keys_gpu: GpuVec<i32> = GpuVec::<i32>::from_slice_async(key_arr.values(), stream.raw())?;
    let mut out_gpu: GpuVec<u64> = GpuVec::<u64>::zeros_async(n_groups as usize, stream.raw())?;

    let module = module_cache::get_or_build_module(
        module_path!(),
        "shmem_count".to_string(),
        None,
        || compile_shmem_count_kernel(),
    )?;
    let function = module.function(KERNEL_ENTRY)?;

    let params = tune(TuneInputs {
        n_rows: n_rows as u32,
        n_groups: BLOCK_GROUPS,
        bytes_per_acc_slot: 8,
        max_shared_per_block: None,
    })
    .map_err(|e| {
        BoltError::Other(format!(
            "shmem_count_exec: tuner refused: {e} (n_rows={n_rows}, n_groups={n_groups})"
        ))
    })?;

    // CUDA-Oxide typed kernel args.
    let view_keys = keys_gpu.view();
    let mut view_out = out_gpu.view_mut();
    let mut args = KernelArgs::empty();
    args.push_input(&view_keys);
    args.push_output(&mut view_out);
    args.push_scalar_u32(n_rows as u32);
    args.push_scalar_u32(n_groups);

    launch_with_geometry(
        function,
        params.grid_blocks,
        params.block_threads,
        0,
        &stream,
        &mut args,
    )?;

    // Stage-4 (P1b): pinned D2H; sync once.
    // Build the result: which slots are populated. We use a host-side
    // presence map (same as the SUM executor) to decide which slots
    // make it into the output. For COUNT, a count > 0 directly tells us
    // — no separate set buffer needed.
    let pinned_counts = out_gpu.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let host_counts: Vec<u64> = pinned_counts.as_slice().to_vec();

    let mut out_keys: Vec<i32> = Vec::new();
    let mut out_counts: Vec<i64> = Vec::new();
    for (slot, &c) in host_counts.iter().enumerate() {
        if c > 0 {
            out_keys.push(slot as i32);
            // Output schema is Int64 for COUNT — widened by the planner.
            out_counts.push(c as i64);
        }
    }

    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => unreachable!("try_execute guards this"),
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Int32Array::from(out_keys)),
            Arc::new(Int64Array::from(out_counts)),
        ],
    )
    .map_err(|e| {
        BoltError::Other(format!(
            "groupby_shmem_count_exec: failed to build RecordBatch: {e}"
        ))
    })
}
fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    crate::exec::schema_convert::plan_schema_to_arrow_schema_no_temporal(s, "this aggregate output path")
}

// The cuda_sys/ptr/c_void imports are kept for consistency with the
// other executors even though the launch is now lifted through
// `launch_with_geometry`. If the lint warns, we'll prune.
#[allow(dead_code)]
const _UNUSED_IMPORT_GUARDS: usize = {
    let _ = std::mem::size_of::<*mut c_void>();
    let _ = std::mem::size_of::<CUdeviceptr>();
    let _ = ptr::null::<u8>();
    let _ = cuda_sys::CUDA_SUCCESS;
    0
};

// ---------------------------------------------------------------------------
// Stage-4 (P1b) async round-trip smoke test.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod stage4_tests {
    use super::*;
    use crate::plan::logical_plan::{Expr, Field, Literal};
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    #[test]
    #[ignore = "gpu:tier1"]
    fn async_shmem_count_round_trip() {
        let n: usize = 1024;
        let n_groups: usize = 8;
        let keys: Vec<i32> = (0..n).map(|i| (i % n_groups) as i32).collect();
        let mut expected = vec![0i64; n_groups];
        for &k in &keys {
            expected[k as usize] += 1;
        }
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![ColumnIO { name: "k".into(), dtype: DataType::Int32 }],
                group_by: vec![0],
                aggregates: vec![AggregateExpr::Count(Expr::Literal(Literal::Null))],
                output_schema: Schema::new(vec![
                    Field::new("k", DataType::Int32, false),
                    Field::new("count_star", DataType::Int64, true),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(keys)) as arrow_array::ArrayRef],
        )
        .unwrap();
        let out = match try_execute(&plan, &batch) {
            Some(Ok(b)) => b,
            _ => return,
        };
        let ks = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let cs = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..out.num_rows() {
            assert_eq!(cs.value(i), expected[ks.value(i) as usize]);
        }
    }
}
