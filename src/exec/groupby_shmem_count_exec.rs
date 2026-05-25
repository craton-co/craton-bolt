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

use arrow_array::{Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{JavelinError, JavelinResult};
use crate::exec::groupby_shmem_launch::{tune, TuneInputs};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::jit::shmem_count_kernel::{
    compile_shmem_count_kernel, BLOCK_GROUPS, KERNEL_ENTRY,
};
use crate::jit::CudaModule;
use crate::plan::logical_plan::{AggregateExpr, DataType, Schema};
use crate::plan::physical_plan::PhysicalPlan;

const MIN_ROWS_FAST_PATH: usize = 64 * 1024;

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
    match &aggregate.aggregates[0] {
        AggregateExpr::Count(_) => {}
        _ => return None,
    }

    let key_io_idx = aggregate.group_by[0];
    let key_io = match aggregate.inputs.get(key_io_idx) {
        Some(io) if io.dtype == DataType::Int32 => io,
        _ => return None,
    };

    let key_arr = batch
        .column_by_name(&key_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;

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
) -> JavelinResult<RecordBatch> {
    let n_rows = key_arr.len();
    let keys_gpu: GpuVec<i32> = GpuVec::<i32>::from_slice(key_arr.values())?;
    let mut out_gpu: GpuVec<u64> = GpuVec::<u64>::zeros(n_groups as usize)?;

    let ptx = compile_shmem_count_kernel()?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(KERNEL_ENTRY)?;

    let params = tune(TuneInputs {
        n_rows: n_rows as u32,
        n_groups: BLOCK_GROUPS,
        bytes_per_acc_slot: 8,
        max_shared_per_block: None,
    })
    .map_err(|e| {
        JavelinError::Other(format!(
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

    let stream = CudaStream::null();
    launch_with_geometry(
        function,
        params.grid_blocks,
        params.block_threads,
        0,
        &stream,
        &mut args,
    )?;

    // Build the result: which slots are populated. We use a host-side
    // presence map (same as the SUM executor) to decide which slots
    // make it into the output. For COUNT, a count > 0 directly tells us
    // — no separate set buffer needed.
    let host_counts: Vec<u64> = out_gpu.to_vec()?;

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
        JavelinError::Other(format!(
            "groupby_shmem_count_exec: failed to build RecordBatch: {e}"
        ))
    })
}

fn plan_dtype_to_arrow(d: DataType) -> JavelinResult<ArrowDataType> {
    match d {
        DataType::Int32 => Ok(ArrowDataType::Int32),
        DataType::Int64 => Ok(ArrowDataType::Int64),
        DataType::Float32 => Ok(ArrowDataType::Float32),
        DataType::Float64 => Ok(ArrowDataType::Float64),
        DataType::Bool => Ok(ArrowDataType::Boolean),
        DataType::Utf8 => Ok(ArrowDataType::Utf8),
    }
}

fn plan_schema_to_arrow_schema(s: &Schema) -> JavelinResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
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
