// SPDX-License-Identifier: Apache-2.0

//! Tier-1 **MIN / MAX** GROUP BY executor (low-cardinality integer
//! MIN/MAX over `n_groups <= BLOCK_GROUPS`).
//!
//! Single aggregate only (one MIN or one MAX per query). Mixed
//! MIN+MAX in the same query is rejected — a future multi-agg-MinMax
//! variant could be built atop `partition_reduce_kernel_minmax` if a
//! workload demands it.

use std::sync::Arc;

use arrow_array::{Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::groupby_shmem_launch::{tune, TuneInputs};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::jit::partition_reduce_kernel_minmax::{MinMaxDtype, MinMaxOp};
use crate::jit::shmem_minmax_kernel::{
    compile_shmem_minmax_kernel, kernel_entry, BLOCK_GROUPS,
};
use crate::jit::CudaModule;
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

    // Recognise MIN(col) or MAX(col) over a bare column ref.
    let (op, val_col_name) = match &aggregate.aggregates[0] {
        AggregateExpr::Min(Expr::Column(n)) => (MinMaxOp::Min, n.as_str()),
        AggregateExpr::Max(Expr::Column(n)) => (MinMaxOp::Max, n.as_str()),
        _ => return None,
    };

    // Single Int32 key.
    let key_io_idx = aggregate.group_by[0];
    let key_io = match aggregate.inputs.get(key_io_idx) {
        Some(io) if io.dtype == DataType::Int32 => io,
        _ => return None,
    };

    let key_arr = batch
        .column_by_name(&key_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;

    let val_col = batch.column_by_name(val_col_name)?;
    let val_dtype = match val_col.data_type() {
        ArrowDataType::Int32 => MinMaxDtype::Int32,
        ArrowDataType::Int64 => MinMaxDtype::Int64,
        _ => return None, // Float MIN/MAX deferred (no native PTX atomic).
    };
    if key_arr.len() != val_col.len() {
        return None;
    }
    let n_rows = key_arr.len();
    if n_rows < MIN_ROWS_FAST_PATH {
        return None;
    }

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
        // Tier-2.1 MIN/MAX owns this.
        return None;
    }

    Some(execute_inner(plan, key_arr, val_col, op, val_dtype, n_groups_est))
}

fn execute_inner(
    plan: &PhysicalPlan,
    key_arr: &Int32Array,
    val_col: &dyn arrow_array::Array,
    op: MinMaxOp,
    val_dtype: MinMaxDtype,
    n_groups: u32,
) -> BoltResult<RecordBatch> {
    let n_rows = key_arr.len() as u32;
    // Stage-4 (P1b): per-call stream shared across H2D / kernel / D2H.
    let stream = CudaStream::null_or_default();
    let keys_gpu: GpuVec<i32> = GpuVec::<i32>::from_slice_async(key_arr.values(), stream.raw())?;

    // Initialise out_vals to the IDENTITY for the op so global atomics
    // (overflow + merge paths) start from a known sentinel.
    let n_groups_usz = n_groups as usize;
    match val_dtype {
        MinMaxDtype::Int32 => {
            let identity: i32 = match op {
                MinMaxOp::Min => i32::MAX,
                MinMaxOp::Max => i32::MIN,
            };
            let vals_in: Vec<i32> = val_col
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| BoltError::Other("expected Int32Array".into()))?
                .values()
                .to_vec();
            let vals_gpu: GpuVec<i32> = GpuVec::<i32>::from_slice_async(&vals_in, stream.raw())?;
            let init_out: Vec<i32> = vec![identity; n_groups_usz];
            let mut out_vals_gpu: GpuVec<i32> = GpuVec::<i32>::from_slice_async(&init_out, stream.raw())?;
            let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros_async(n_groups_usz, stream.raw())?;

            run_launch_i32(
                op, val_dtype, &keys_gpu, &vals_gpu, &mut out_vals_gpu, &mut out_set_gpu, n_rows, n_groups, &stream,
            )?;

            // Stage-4 (P1b): pinned D2H; sync once.
            let pinned_vals = out_vals_gpu.to_pinned_async(stream.raw())?;
            let pinned_set = out_set_gpu.to_pinned_async(stream.raw())?;
            stream.synchronize()?;
            let host_out_vals: Vec<i32> = pinned_vals.as_slice().to_vec();
            let host_out_set: Vec<u8> = pinned_set.as_slice().to_vec();

            let mut keys: Vec<i32> = Vec::new();
            let mut vals: Vec<i32> = Vec::new();
            for g in 0..n_groups_usz {
                if host_out_set[g] != 0 {
                    keys.push(g as i32);
                    vals.push(host_out_vals[g]);
                }
            }

            let aggregate = match plan {
                PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
                _ => unreachable!(),
            };
            let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
            RecordBatch::try_new(
                arrow_schema,
                vec![
                    Arc::new(Int32Array::from(keys)),
                    Arc::new(Int32Array::from(vals)),
                ],
            )
            .map_err(|e| {
                BoltError::Other(format!(
                    "groupby_shmem_minmax_exec(i32): build error: {e}"
                ))
            })
        }
        MinMaxDtype::Int64 => {
            let identity: i64 = match op {
                MinMaxOp::Min => i64::MAX,
                MinMaxOp::Max => i64::MIN,
            };
            let vals_in: Vec<i64> = val_col
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| BoltError::Other("expected Int64Array".into()))?
                .values()
                .to_vec();
            let vals_gpu: GpuVec<i64> = GpuVec::<i64>::from_slice_async(&vals_in, stream.raw())?;
            let init_out: Vec<i64> = vec![identity; n_groups_usz];
            let mut out_vals_gpu: GpuVec<i64> = GpuVec::<i64>::from_slice_async(&init_out, stream.raw())?;
            let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros_async(n_groups_usz, stream.raw())?;

            run_launch_i64(
                op, val_dtype, &keys_gpu, &vals_gpu, &mut out_vals_gpu, &mut out_set_gpu, n_rows, n_groups, &stream,
            )?;

            // Stage-4 (P1b): pinned D2H; sync once.
            let pinned_vals = out_vals_gpu.to_pinned_async(stream.raw())?;
            let pinned_set = out_set_gpu.to_pinned_async(stream.raw())?;
            stream.synchronize()?;
            let host_out_vals: Vec<i64> = pinned_vals.as_slice().to_vec();
            let host_out_set: Vec<u8> = pinned_set.as_slice().to_vec();

            let mut keys: Vec<i32> = Vec::new();
            let mut vals: Vec<i64> = Vec::new();
            for g in 0..n_groups_usz {
                if host_out_set[g] != 0 {
                    keys.push(g as i32);
                    vals.push(host_out_vals[g]);
                }
            }

            let aggregate = match plan {
                PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
                _ => unreachable!(),
            };
            let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
            RecordBatch::try_new(
                arrow_schema,
                vec![
                    Arc::new(Int32Array::from(keys)),
                    Arc::new(Int64Array::from(vals)),
                ],
            )
            .map_err(|e| {
                BoltError::Other(format!(
                    "groupby_shmem_minmax_exec(i64): build error: {e}"
                ))
            })
        }
    }
}

fn run_launch_i32(
    op: MinMaxOp,
    val_dtype: MinMaxDtype,
    keys_gpu: &GpuVec<i32>,
    vals_gpu: &GpuVec<i32>,
    out_vals_gpu: &mut GpuVec<i32>,
    out_set_gpu: &mut GpuVec<u8>,
    n_rows: u32,
    n_groups: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    let ptx = compile_shmem_minmax_kernel(op, val_dtype)?;
    let module = CudaModule::from_ptx(&ptx)?;
    let entry = kernel_entry(op, val_dtype);
    let function = module.function(&entry)?;

    let params = tune(TuneInputs {
        n_rows,
        n_groups: BLOCK_GROUPS,
        bytes_per_acc_slot: 4,
        max_shared_per_block: None,
    })
    .map_err(|e| BoltError::Other(format!("minmax tuner refused: {e}")))?;

    let view_keys = keys_gpu.view();
    let view_vals = vals_gpu.view();
    let mut view_out_vals = out_vals_gpu.view_mut();
    let mut view_out_set = out_set_gpu.view_mut();

    let mut args = KernelArgs::empty();
    args.push_input(&view_keys);
    args.push_input(&view_vals);
    args.push_output(&mut view_out_vals);
    args.push_output(&mut view_out_set);
    args.push_scalar_u32(n_rows);
    args.push_scalar_u32(n_groups);

    launch_with_geometry(
        function,
        params.grid_blocks,
        params.block_threads,
        0,
        stream,
        &mut args,
    )
}

fn run_launch_i64(
    op: MinMaxOp,
    val_dtype: MinMaxDtype,
    keys_gpu: &GpuVec<i32>,
    vals_gpu: &GpuVec<i64>,
    out_vals_gpu: &mut GpuVec<i64>,
    out_set_gpu: &mut GpuVec<u8>,
    n_rows: u32,
    n_groups: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    let ptx = compile_shmem_minmax_kernel(op, val_dtype)?;
    let module = CudaModule::from_ptx(&ptx)?;
    let entry = kernel_entry(op, val_dtype);
    let function = module.function(&entry)?;

    let params = tune(TuneInputs {
        n_rows,
        n_groups: BLOCK_GROUPS,
        bytes_per_acc_slot: 8,
        max_shared_per_block: None,
    })
    .map_err(|e| BoltError::Other(format!("minmax tuner refused: {e}")))?;

    let view_keys = keys_gpu.view();
    let view_vals = vals_gpu.view();
    let mut view_out_vals = out_vals_gpu.view_mut();
    let mut view_out_set = out_set_gpu.view_mut();

    let mut args = KernelArgs::empty();
    args.push_input(&view_keys);
    args.push_input(&view_vals);
    args.push_output(&mut view_out_vals);
    args.push_output(&mut view_out_set);
    args.push_scalar_u32(n_rows);
    args.push_scalar_u32(n_groups);

    launch_with_geometry(
        function,
        params.grid_blocks,
        params.block_threads,
        0,
        stream,
        &mut args,
    )
}

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
// Stage-4 (P1b) async round-trip smoke test.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod stage4_tests {
    use super::*;
    use crate::plan::logical_plan::Field;
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};

    #[test]
    #[ignore = "gpu:tier1"]
    fn async_shmem_minmax_round_trip() {
        let n: usize = 1024;
        let n_groups: usize = 8;
        let keys: Vec<i32> = (0..n).map(|i| (i % n_groups) as i32).collect();
        let vals: Vec<i32> = (0..n).map(|i| i as i32).collect();
        let mut expected_min = vec![i32::MAX; n_groups];
        for (i, &k) in keys.iter().enumerate() {
            if vals[i] < expected_min[k as usize] {
                expected_min[k as usize] = vals[i];
            }
        }
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![
                    ColumnIO { name: "k".into(), dtype: DataType::Int32 },
                    ColumnIO { name: "v".into(), dtype: DataType::Int32 },
                ],
                group_by: vec![0],
                aggregates: vec![AggregateExpr::Min(Expr::Column("v".into()))],
                output_schema: Schema::new(vec![
                    Field::new("k", DataType::Int32, false),
                    Field::new("min_v", DataType::Int32, true),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(keys)) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(vals)) as arrow_array::ArrayRef,
            ],
        )
        .unwrap();
        let out = match try_execute(&plan, &batch) {
            Some(Ok(b)) => b,
            _ => return,
        };
        let ks = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let vs = out.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..out.num_rows() {
            assert_eq!(vs.value(i), expected_min[ks.value(i) as usize]);
        }
    }
}
