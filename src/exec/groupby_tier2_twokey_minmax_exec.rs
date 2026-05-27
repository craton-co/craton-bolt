// SPDX-License-Identifier: Apache-2.0

//! **Two-key MIN / MAX at Tier 2.1** — high-cardinality executor for
//! `SELECT a, b, {MIN,MAX}(v) FROM x GROUP BY a, b` over **integer** value
//! columns (Int32 / Int64).
//!
//! Mirror of [`crate::exec::groupby_tier2_minmax_exec`] adapted for the
//! i64-packed-two-key path. Both group-by columns are Int32 and packed
//! losslessly into a single i64 host-side (matching the convention in
//! `groupby.rs::pack_keys`); the on-device chain then treats them as a
//! single dense key column.
//!
//! Float MIN/MAX over two keys is handled by the sibling executor
//! [`crate::exec::groupby_tier2_twokey_minmax_float_exec`] — PTX has no
//! native `atom.shared.{min,max}.f{32,64}` on sm_70 and the float kernel
//! emits a CAS-loop instead.
//!
//! ## Algorithm
//!
//! 1. Pack `(k1, k2)` → `i64` host-side.
//! 2. Run `partition_kernel_i64` over the packed keys.
//! 3. Round-trip the integer value column through `f64` (the scatter
//!    kernel only takes f64 values; same hack as the single-key int
//!    MIN/MAX exec — `Int32→f64→Int32` is exact for all i32, Int64
//!    is lossy above 2^53 and documented as such).
//! 4. Run `scatter_kernel_i64` (packed i64 keys + dummy f64 vals).
//! 5. Re-upload the integer vals.
//! 6. Run `partition_reduce_kernel_minmax_i64` with the integer vals.
//! 7. Walk slots, unpack `(key_hi, key_lo)`, sort by packed-i64 ASC,
//!    build the output RecordBatch.
//!
//! ## Scope (v0)
//!
//! - GROUP BY exactly two Int32 columns
//! - Exactly one aggregate, `MIN(<bare col>)` or `MAX(<bare col>)`
//! - Value dtype Int32 or Int64
//! - `n_rows >= 256 K`
//! - Combined key cardinality < 100 M (Tier-2 dispatcher cap)

use std::sync::Arc;

use arrow_array::{Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::partition_offsets;
use crate::jit::partition_reduce_kernel_minmax::{MinMaxDtype, MinMaxOp};
use crate::jit::partition_reduce_kernel_minmax_i64::{
    compile_partition_reduce_kernel_minmax_i64, kernel_entry as minmax_i64_entry,
    BLOCK_GROUPS, BLOCK_THREADS as REDUCE_BLOCK_THREADS,
};
use crate::jit::{partition_kernel_i64, scatter_kernel_i64, CudaModule};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Schema};
use crate::plan::physical_plan::PhysicalPlan;

const BLOCK_THREADS: u32 = 256;

/// Try the two-key Tier-2.1 integer MIN/MAX fast path. `None` on any
/// precondition miss so the caller falls through to the next strategy.
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
    if aggregate.group_by.len() != 2 || aggregate.aggregates.len() != 1 {
        return None;
    }

    // Single MIN or MAX over a bare column.
    let (op, val_col_name) = match &aggregate.aggregates[0] {
        AggregateExpr::Min(Expr::Column(n)) => (MinMaxOp::Min, n.as_str()),
        AggregateExpr::Max(Expr::Column(n)) => (MinMaxOp::Max, n.as_str()),
        _ => return None,
    };

    // Both keys must be Int32.
    let k1_io = aggregate.inputs.get(aggregate.group_by[0])?;
    let k2_io = aggregate.inputs.get(aggregate.group_by[1])?;
    if k1_io.dtype != DataType::Int32 || k2_io.dtype != DataType::Int32 {
        return None;
    }

    let k1 = batch
        .column_by_name(&k1_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;
    let k2 = batch
        .column_by_name(&k2_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;
    if k1.len() != k2.len() {
        return None;
    }

    // Value: Int32 or Int64. Float routes through the sibling float exec.
    let val_col = batch.column_by_name(val_col_name)?;
    let val_dtype = match val_col.data_type() {
        ArrowDataType::Int32 => MinMaxDtype::Int32,
        ArrowDataType::Int64 => MinMaxDtype::Int64,
        _ => return None,
    };

    if val_col.len() != k1.len() {
        return None;
    }
    let n_rows = k1.len();
    if n_rows < 256 * 1024 {
        return None;
    }
    if n_rows >= 100_000_000 {
        return None;
    }

    // Int64 correctness gate: the scatter pipeline routes the value
    // column through `f64`, which only represents `i64` losslessly for
    // `|v| <= 2^53`. Decline when any value would lose precision so the
    // caller falls through to a non-lossy (slower) executor — wrong
    // MIN/MAX is far worse than slow MIN/MAX. Shared with the single-key
    // exec so the gate is defined in exactly one place.
    //
    // TODO(c4): wire a typed i64 scatter+reduce path so this guard can
    // be removed. The two scatter kernels in `crate::jit::scatter_kernel*`
    // both take `f64` value columns today; a sibling that takes `i64`
    // value columns (or a generic typed scatter) would let us keep the
    // fast lane for the full Int64 range.
    if val_dtype == MinMaxDtype::Int64 {
        let a = val_col.as_any().downcast_ref::<Int64Array>()?;
        if crate::exec::groupby_tier2_minmax_exec::i64_values_exceed_f64_mantissa(a.values()) {
            return None;
        }
    }

    Some(execute_inner(plan, k1, k2, val_col, op, val_dtype))
}

fn execute_inner(
    plan: &PhysicalPlan,
    k1: &Int32Array,
    k2: &Int32Array,
    val_col: &dyn arrow_array::Array,
    op: MinMaxOp,
    val_dtype: MinMaxDtype,
) -> BoltResult<RecordBatch> {
    let n_rows = k1.len() as u32;

    // ---- Host-side pack ----
    // `(k1 << 32) | (k2 & 0xFFFF_FFFF)`. Matches `groupby.rs::pack_keys`.
    let packed: Vec<i64> = k1
        .values()
        .iter()
        .zip(k2.values().iter())
        .map(|(&a, &b)| ((a as u32 as u64) << 32 | (b as u32 as u64)) as i64)
        .collect();
    let keys_gpu: GpuVec<i64> = GpuVec::<i64>::from_slice(&packed)?;

    // Round-trip ints through f64 for the scatter (same as single-key int
    // MIN/MAX exec). Exact for Int32; for Int64 the round-trip is only
    // sound for `|v| <= 2^53` — `try_execute` declines the batch
    // otherwise (see the `i64_values_exceed_f64_mantissa` gate there).
    // TODO(c4): remove the f64 hop once a typed i64 scatter lands.
    let host_vals_f64: Vec<f64> = match val_dtype {
        MinMaxDtype::Int32 => val_col
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| BoltError::Other("expected Int32Array".into()))?
            .values()
            .iter()
            .map(|&v| v as f64)
            .collect(),
        MinMaxDtype::Int64 => val_col
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| BoltError::Other("expected Int64Array".into()))?
            .values()
            .iter()
            .map(|&v| v as f64)
            .collect(),
    };
    let vals_in_gpu: GpuVec<f64> = GpuVec::<f64>::from_slice(&host_vals_f64)?;

    let num_partitions = partition_kernel_i64::NUM_PARTITIONS;

    // ---- Partition pass (i64) ----
    let mut counts: GpuVec<u32> = GpuVec::<u32>::zeros(num_partitions as usize)?;
    let mut partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros(n_rows as usize)?;
    {
        let ptx = partition_kernel_i64::compile_partition_kernel_i64()?;
        let module = CudaModule::from_ptx(&ptx)?;
        let func = module.function(partition_kernel_i64::KERNEL_ENTRY)?;

        let view_keys = keys_gpu.view();
        let mut view_pids = partition_ids.view_mut();
        let mut view_counts = counts.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_output(&mut view_pids);
        args.push_output(&mut view_counts);
        args.push_scalar_u32(n_rows);

        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);
        let stream = CudaStream::null();
        launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
    }

    let offsets: Vec<u32> = partition_offsets::compute_partition_offsets(&counts)?;
    let offsets_gpu: GpuVec<u32> = partition_offsets::upload_offsets(&offsets)?;

    // ---- Scatter (i64 keys + f64 vals; we discard the scattered f64 vals) ----
    let mut scatter_keys: GpuVec<i64> = GpuVec::<i64>::zeros(n_rows as usize)?;
    let mut scatter_vals_f64: GpuVec<f64> = GpuVec::<f64>::zeros(n_rows as usize)?;
    {
        let ptx = scatter_kernel_i64::compile_scatter_kernel_i64()?;
        let module = CudaModule::from_ptx(&ptx)?;
        let func = module.function(scatter_kernel_i64::KERNEL_ENTRY)?;
        let mut cursors: GpuVec<u32> = GpuVec::<u32>::zeros(num_partitions as usize)?;

        let view_keys = keys_gpu.view();
        let view_vals = vals_in_gpu.view();
        let view_pids = partition_ids.view();
        let view_offsets = offsets_gpu.view();
        let mut view_cursors = cursors.view_mut();
        let mut view_sk = scatter_keys.view_mut();
        let mut view_sv = scatter_vals_f64.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_input(&view_vals);
        args.push_input(&view_pids);
        args.push_input(&view_offsets);
        args.push_output(&mut view_cursors);
        args.push_output(&mut view_sk);
        args.push_output(&mut view_sv);
        args.push_scalar_u32(n_rows);

        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);
        let stream = CudaStream::null();
        launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
    }

    // Convert scattered f64 vals back to the integer dtype for the reduce.
    let scattered_f64: Vec<f64> = scatter_vals_f64.to_vec()?;

    match val_dtype {
        MinMaxDtype::Int32 => {
            let v: Vec<i32> = scattered_f64.iter().map(|&x| x as i32).collect();
            let vals_typed_gpu = GpuVec::<i32>::from_slice(&v)?;
            // Keep these alive past the launch.
            let _ = (vals_in_gpu, scatter_vals_f64);
            run_reduce_phase_i32(plan, op, vals_typed_gpu, scatter_keys, offsets, num_partitions)
        }
        MinMaxDtype::Int64 => {
            let v: Vec<i64> = scattered_f64.iter().map(|&x| x as i64).collect();
            let vals_typed_gpu = GpuVec::<i64>::from_slice(&v)?;
            let _ = (vals_in_gpu, scatter_vals_f64);
            run_reduce_phase_i64(plan, op, vals_typed_gpu, scatter_keys, offsets, num_partitions)
        }
    }
}

/// Reduce phase for Int32 value dtype (i64-key).
fn run_reduce_phase_i32(
    plan: &PhysicalPlan,
    op: MinMaxOp,
    vals_gpu: GpuVec<i32>,
    scatter_keys: GpuVec<i64>,
    offsets: Vec<u32>,
    num_partitions: u32,
) -> BoltResult<RecordBatch> {
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice(&offsets)?;
    let block_groups = BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let mut out_keys_gpu: GpuVec<i64> = GpuVec::<i64>::zeros(n_out_slots)?;
    let mut out_vals_gpu: GpuVec<i32> = GpuVec::<i32>::zeros(n_out_slots)?;
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros(n_out_slots)?;

    {
        let ptx = compile_partition_reduce_kernel_minmax_i64(op, MinMaxDtype::Int32)?;
        let module = CudaModule::from_ptx(&ptx)?;
        let entry = minmax_i64_entry(op, MinMaxDtype::Int32);
        let func = module.function(&entry)?;

        let view_pk = scatter_keys.view();
        let view_pv = vals_gpu.view();
        let view_po = offsets_kp1_gpu.view();
        let mut view_ok = out_keys_gpu.view_mut();
        let mut view_ov = out_vals_gpu.view_mut();
        let mut view_os = out_set_gpu.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_pk);
        args.push_input(&view_pv);
        args.push_input(&view_po);
        args.push_output(&mut view_ok);
        args.push_output(&mut view_ov);
        args.push_output(&mut view_os);

        let stream = CudaStream::null();
        launch_with_geometry(func, num_partitions, REDUCE_BLOCK_THREADS, 0, &stream, &mut args)?;
    }

    let host_out_keys: Vec<i64> = out_keys_gpu.to_vec()?;
    let host_out_vals: Vec<i32> = out_vals_gpu.to_vec()?;
    let host_out_set: Vec<u8> = out_set_gpu.to_vec()?;

    let mut rows: Vec<(i64, i32)> = Vec::new();
    for pid in 0..num_partitions as usize {
        let base = pid * block_groups;
        for slot in 0..block_groups {
            let idx = base + slot;
            if host_out_set[idx] != 0 {
                rows.push((host_out_keys[idx], host_out_vals[idx]));
            }
        }
    }
    rows.sort_by_key(|(k, _)| *k);

    let mut out_k1: Vec<i32> = Vec::with_capacity(rows.len());
    let mut out_k2: Vec<i32> = Vec::with_capacity(rows.len());
    let mut out_v: Vec<i32> = Vec::with_capacity(rows.len());
    for (k, v) in rows {
        let u = k as u64;
        out_k1.push((u >> 32) as u32 as i32);
        out_k2.push((u & 0xFFFF_FFFF) as u32 as i32);
        out_v.push(v);
    }

    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => unreachable!("try_execute guards this"),
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Int32Array::from(out_k1)),
            Arc::new(Int32Array::from(out_k2)),
            Arc::new(Int32Array::from(out_v)),
        ],
    )
    .map_err(|e| {
        BoltError::Other(format!(
            "groupby_tier2_twokey_minmax_exec(i32): build error: {e}"
        ))
    })
}

/// Reduce phase for Int64 value dtype (i64-key).
fn run_reduce_phase_i64(
    plan: &PhysicalPlan,
    op: MinMaxOp,
    vals_gpu: GpuVec<i64>,
    scatter_keys: GpuVec<i64>,
    offsets: Vec<u32>,
    num_partitions: u32,
) -> BoltResult<RecordBatch> {
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice(&offsets)?;
    let block_groups = BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let mut out_keys_gpu: GpuVec<i64> = GpuVec::<i64>::zeros(n_out_slots)?;
    let mut out_vals_gpu: GpuVec<i64> = GpuVec::<i64>::zeros(n_out_slots)?;
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros(n_out_slots)?;

    {
        let ptx = compile_partition_reduce_kernel_minmax_i64(op, MinMaxDtype::Int64)?;
        let module = CudaModule::from_ptx(&ptx)?;
        let entry = minmax_i64_entry(op, MinMaxDtype::Int64);
        let func = module.function(&entry)?;

        let view_pk = scatter_keys.view();
        let view_pv = vals_gpu.view();
        let view_po = offsets_kp1_gpu.view();
        let mut view_ok = out_keys_gpu.view_mut();
        let mut view_ov = out_vals_gpu.view_mut();
        let mut view_os = out_set_gpu.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_pk);
        args.push_input(&view_pv);
        args.push_input(&view_po);
        args.push_output(&mut view_ok);
        args.push_output(&mut view_ov);
        args.push_output(&mut view_os);

        let stream = CudaStream::null();
        launch_with_geometry(func, num_partitions, REDUCE_BLOCK_THREADS, 0, &stream, &mut args)?;
    }

    let host_out_keys: Vec<i64> = out_keys_gpu.to_vec()?;
    let host_out_vals: Vec<i64> = out_vals_gpu.to_vec()?;
    let host_out_set: Vec<u8> = out_set_gpu.to_vec()?;

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for pid in 0..num_partitions as usize {
        let base = pid * block_groups;
        for slot in 0..block_groups {
            let idx = base + slot;
            if host_out_set[idx] != 0 {
                rows.push((host_out_keys[idx], host_out_vals[idx]));
            }
        }
    }
    rows.sort_by_key(|(k, _)| *k);

    let mut out_k1: Vec<i32> = Vec::with_capacity(rows.len());
    let mut out_k2: Vec<i32> = Vec::with_capacity(rows.len());
    let mut out_v: Vec<i64> = Vec::with_capacity(rows.len());
    for (k, v) in rows {
        let u = k as u64;
        out_k1.push((u >> 32) as u32 as i32);
        out_k2.push((u & 0xFFFF_FFFF) as u32 as i32);
        out_v.push(v);
    }

    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => unreachable!("try_execute guards this"),
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Int32Array::from(out_k1)),
            Arc::new(Int32Array::from(out_k2)),
            Arc::new(Int64Array::from(out_v)),
        ],
    )
    .map_err(|e| {
        BoltError::Other(format!(
            "groupby_tier2_twokey_minmax_exec(i64): build error: {e}"
        ))
    })
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
// Correctness-guard tests for the two-key Int64-value path.
//
// We re-export the same predicate the single-key exec uses
// (`groupby_tier2_minmax_exec::i64_values_exceed_f64_mantissa`), so the
// behaviour we care about — "any value > 2^53 declines the fast path" —
// is exercised here in the context of the two-key GROUP BY. No GPU
// required.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use crate::exec::groupby_tier2_minmax_exec::i64_values_exceed_f64_mantissa;

    const F64_EXACT_I64_LIMIT: i64 = 1_i64 << 53;

    /// Two-key `MIN(int64_col)` with a value column containing
    /// `2^53 + 1` must decline the fast path. Without the guard the
    /// executor would silently downcast through f64 and return the
    /// wrong number for one of the (k1, k2) groups.
    #[test]
    fn twokey_min_over_pow2_53_plus_one_is_rejected() {
        let big = F64_EXACT_I64_LIMIT + 1;
        // Sanity: confirm f64 round-trip really is lossy here.
        assert_ne!(big as f64 as i64, big);
        let vals = vec![1_i64, 2, big, 4, 5];
        assert!(
            i64_values_exceed_f64_mantissa(&vals),
            "two-key MIN must trip the guard on a column containing 2^53 + 1"
        );
    }

    /// Two-key `MAX(int64_col)` — same property. The guard fires
    /// regardless of which aggregation op is requested.
    #[test]
    fn twokey_max_over_pow2_53_plus_one_is_rejected() {
        let big = F64_EXACT_I64_LIMIT + 1;
        let vals = vec![big, 10, 20, 30];
        assert!(i64_values_exceed_f64_mantissa(&vals));
    }

    /// Two-key path with an in-range Int64 column must keep using the
    /// fast lane. This is the no-regression test: typical workloads
    /// (small IDs, modest counters) stay on the GPU.
    #[test]
    fn twokey_in_range_int64_keeps_fast_path() {
        let vals: Vec<i64> = (0..512).map(|i| i * 1024).collect();
        assert!(
            !i64_values_exceed_f64_mantissa(&vals),
            "in-range Int64 two-key MIN/MAX must NOT decline"
        );
    }
}

// ---------------------------------------------------------------------------
// Host-only eligibility-gate tests for the two-key integer MIN/MAX exec.
//
// Pure host tests; the actual MIN/MAX numerical correctness lives in the
// e2e suite where a real CUDA context is available. The Int64-precision
// (f64-roundtrip) guard is covered separately in the single-key exec's
// inline tests once that fix lands on the branch — the twokey path will
// inherit the predicate from there.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod eligibility_tests {
    use super::*;
    use crate::plan::logical_plan::Field;
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};

    /// Plan for `SELECT k1, k2, MIN(v) FROM t GROUP BY k1, k2`. The
    /// op argument lets a test swap to MAX without rebuilding.
    fn build_twokey_minmax_plan(op_is_min: bool, val_dtype: DataType) -> PhysicalPlan {
        let inputs = vec![
            ColumnIO {
                name: "k1".into(),
                dtype: DataType::Int32,
            },
            ColumnIO {
                name: "k2".into(),
                dtype: DataType::Int32,
            },
            ColumnIO {
                name: "v".into(),
                dtype: val_dtype,
            },
        ];
        let agg = if op_is_min {
            AggregateExpr::Min(Expr::Column("v".into()))
        } else {
            AggregateExpr::Max(Expr::Column("v".into()))
        };
        let output_schema = Schema::new(vec![
            Field::new("k1", DataType::Int32, false),
            Field::new("k2", DataType::Int32, false),
            Field::new("v", val_dtype, true),
        ]);
        PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs,
                group_by: vec![0, 1],
                aggregates: vec![agg],
                output_schema,
            },
        }
    }

    /// `(k1, k2, v)` batch with `n` rows.
    fn twokey_minmax_batch_i32(n: usize) -> RecordBatch {
        let k1: Vec<i32> = (0..n as i32).collect();
        let k2: Vec<i32> = (0..n as i32).map(|i| i + 1).collect();
        let v: Vec<i32> = (0..n as i32).map(|i| i * 2).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k1", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Int32, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(k1)) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(k2)) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(v)) as arrow_array::ArrayRef,
            ],
        )
        .unwrap()
    }

    /// Non-Aggregate plan: reject.
    #[test]
    fn rejects_non_aggregate_plan() {
        let plan = PhysicalPlan::Union { inputs: vec![] };
        let batch = twokey_minmax_batch_i32(0);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Single-key plan belongs to the single-key sibling.
    #[test]
    fn rejects_single_key_plan() {
        let mut plan = build_twokey_minmax_plan(true, DataType::Int32);
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.group_by = vec![0];
        }
        let batch = twokey_minmax_batch_i32(300_000);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Two aggregates → multi-agg territory.
    #[test]
    fn rejects_two_aggregates() {
        let mut plan = build_twokey_minmax_plan(true, DataType::Int32);
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.aggregates.push(AggregateExpr::Max(Expr::Column("v".into())));
        }
        let batch = twokey_minmax_batch_i32(300_000);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// SUM / COUNT / AVG aggregates → reject.
    #[test]
    fn rejects_sum_aggregate() {
        let mut plan = build_twokey_minmax_plan(true, DataType::Int32);
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.aggregates = vec![AggregateExpr::Sum(Expr::Column("v".into()))];
        }
        let batch = twokey_minmax_batch_i32(300_000);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// MIN over Float64 → goes to the float-minmax sibling, not this exec.
    #[test]
    fn rejects_float_value_column() {
        let plan = build_twokey_minmax_plan(true, DataType::Float64);
        let n = 300_000;
        let k: Vec<i32> = (0..n as i32).collect();
        let v: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k1", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(k.clone())) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(k)) as arrow_array::ArrayRef,
                Arc::new(arrow_array::Float64Array::from(v)) as arrow_array::ArrayRef,
            ],
        )
        .unwrap();
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Below the row threshold → defer.
    #[test]
    fn rejects_below_row_threshold() {
        let plan = build_twokey_minmax_plan(false, DataType::Int32);
        let batch = twokey_minmax_batch_i32(2_048);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// `pre` kernel present → with-pre executor handles this.
    #[test]
    fn rejects_plan_with_pre_kernel() {
        use crate::plan::physical_plan::KernelSpec;
        let mut plan = build_twokey_minmax_plan(true, DataType::Int32);
        if let PhysicalPlan::Aggregate { pre, .. } = &mut plan {
            *pre = Some(KernelSpec {
                inputs: vec![],
                outputs: vec![],
                ops: vec![],
                predicate: None,
                register_count: 0,
            });
        }
        let batch = twokey_minmax_batch_i32(300_000);
        assert!(try_execute(&plan, &batch).is_none());
    }
}
