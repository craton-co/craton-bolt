// SPDX-License-Identifier: Apache-2.0

//! **Two-key float MIN / MAX at Tier 2.1** — high-cardinality executor for
//! `SELECT a, b, {MIN,MAX}(v) FROM x GROUP BY a, b` over `Float64` value
//! columns.
//!
//! Mirror of [`crate::exec::groupby_tier2_minmax_float_exec`] (single
//! Int32 key) adapted for the i64-packed-two-key path. Both group-by
//! columns are Int32 and packed losslessly into a single i64 host-side
//! (matching the convention in `groupby.rs::pack_keys`); the on-device
//! chain then treats them as a single dense key column.
//!
//! Integer MIN/MAX over two keys is handled by the sibling executor
//! [`crate::exec::groupby_tier2_twokey_minmax_exec`]. The split exists
//! because PTX has no native `atom.shared.{min,max}.f{32,64}` on sm_70 —
//! the float kernel emits a CAS-loop instead.
//!
//! ## Algorithm
//!
//! 1. Pack `(k1, k2)` → `i64` host-side.
//! 2. Run `partition_kernel_i64` over the packed keys.
//! 3. Run `scatter_kernel_i64` (packed i64 keys + f64 vals — no
//!    conversion needed since the value column is already f64).
//! 4. Run `partition_reduce_kernel_minmax_float_i64` (CAS-loop atomic
//!    MIN/MAX on the float bit pattern).
//! 5. Walk slots, unpack `(key_hi, key_lo)`, sort by packed-i64 ASC,
//!    build the output RecordBatch.
//!
//! ## Scope (v0)
//!
//! - GROUP BY exactly two Int32 columns
//! - Exactly one aggregate, `MIN(<bare col>)` or `MAX(<bare col>)`
//! - Value dtype `Float64` only (Float32 promotion is a one-line
//!   addition once a workload asks; the kernel handles both widths
//!   already)
//! - `n_rows >= 256 K`
//! - Combined key cardinality < 100 M (Tier-2 dispatcher cap)

use std::sync::Arc;

use arrow_array::{Array, Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::module_cache;
use crate::exec::partition_offsets;
use crate::jit::partition_reduce_kernel_minmax::MinMaxOp;
use crate::jit::partition_reduce_kernel_minmax_float::FloatDtype;
use crate::jit::partition_reduce_kernel_minmax_float_i64::{
    compile_partition_reduce_kernel_minmax_float_i64_with_spill,
    kernel_entry_with_spill as minmax_float_i64_entry, BLOCK_GROUPS,
    BLOCK_THREADS as REDUCE_BLOCK_THREADS,
};
use crate::jit::{partition_kernel_i64, scatter_kernel_i64, CudaModule};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Schema};
use crate::plan::physical_plan::PhysicalPlan;

const BLOCK_THREADS: u32 = 256;

// ---------------------------------------------------------------------------
// Per-executor module cache. Mirror of `groupby_tier2_minmax_float_exec.rs`
// over the i64-key kernel variants.
// ---------------------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
enum ReduceFloatKey {
    MinF32,
    MaxF32,
    MinF64,
    MaxF64,
}

impl ReduceFloatKey {
    fn from_pair(op: MinMaxOp, dt: FloatDtype) -> Self {
        match (op, dt) {
            (MinMaxOp::Min, FloatDtype::Float32) => ReduceFloatKey::MinF32,
            (MinMaxOp::Max, FloatDtype::Float32) => ReduceFloatKey::MaxF32,
            (MinMaxOp::Min, FloatDtype::Float64) => ReduceFloatKey::MinF64,
            (MinMaxOp::Max, FloatDtype::Float64) => ReduceFloatKey::MaxF64,
        }
    }

    fn into_pair(self) -> (MinMaxOp, FloatDtype) {
        match self {
            ReduceFloatKey::MinF32 => (MinMaxOp::Min, FloatDtype::Float32),
            ReduceFloatKey::MaxF32 => (MinMaxOp::Max, FloatDtype::Float32),
            ReduceFloatKey::MinF64 => (MinMaxOp::Min, FloatDtype::Float64),
            ReduceFloatKey::MaxF64 => (MinMaxOp::Max, FloatDtype::Float64),
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
enum KernelSpec {
    PartitionI64,
    ScatterI64,
    ReduceMinMaxFloatI64(ReduceFloatKey),
}

#[cfg(test)]
static LOAD_COUNT: module_cache::LoadCounter = module_cache::LoadCounter::new();

fn get_or_build_module(spec: &KernelSpec) -> BoltResult<CudaModule> {
    #[cfg(test)]
    let counter = Some(&LOAD_COUNT);
    #[cfg(not(test))]
    let counter = None;
    module_cache::get_or_build_module(module_path!(), format!("{:?}", spec), counter, || {
        Ok(match spec {
            KernelSpec::PartitionI64 => partition_kernel_i64::compile_partition_kernel_i64()?,
            KernelSpec::ScatterI64 => scatter_kernel_i64::compile_scatter_kernel_i64()?,
            KernelSpec::ReduceMinMaxFloatI64(rk) => {
                let (op, dt) = rk.into_pair();
                compile_partition_reduce_kernel_minmax_float_i64(op, dt)?
            }
        })
    })
}

/// Try the two-key Tier-2.1 float MIN/MAX fast path. `None` on any miss.
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

    // Value: Float64 only in v0.
    let val_col = batch.column_by_name(val_col_name)?;
    let float_dtype = match val_col.data_type() {
        ArrowDataType::Float64 => FloatDtype::Float64,
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

    // PV-stage-f: NULL handling — `partition_reduce_kernel_minmax_float_i64`
    // has no `_with_validity` companion. Defer NULL-bearing batches
    // back to the no-pre single-key paths. Stage G follow-up.
    if k1.null_count() > 0 || k2.null_count() > 0 || val_col.null_count() > 0 {
        return None;
    }

    Some(execute_inner(plan, k1, k2, val_col, op, float_dtype))
}

fn execute_inner(
    plan: &PhysicalPlan,
    k1: &Int32Array,
    k2: &Int32Array,
    val_col: &dyn arrow_array::Array,
    op: MinMaxOp,
    float_dtype: FloatDtype,
) -> BoltResult<RecordBatch> {
    let n_rows = k1.len() as u32;

    // Stage-4 (P1b): per-call stream shared across every H2D / kernel / D2H.
    let stream = CudaStream::null_or_default();

    // ---- Host-side pack ----
    let packed: Vec<i64> = k1
        .values()
        .iter()
        .zip(k2.values().iter())
        .map(|(&a, &b)| ((a as u32 as u64) << 32 | (b as u32 as u64)) as i64)
        .collect();
    let keys_gpu: GpuVec<i64> = GpuVec::<i64>::from_slice_async(&packed, stream.raw())?;

    let val_arr = val_col
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| BoltError::Other("expected Float64Array".into()))?;
    let vals_gpu: GpuVec<f64> = GpuVec::<f64>::from_slice_async(val_arr.values(), stream.raw())?;

    let num_partitions = partition_kernel_i64::NUM_PARTITIONS;

    // ---- Partition pass (i64) ----
    let mut counts: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;
    let mut partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros_async(n_rows as usize, stream.raw())?;
    let partition_module = get_or_build_module(&KernelSpec::PartitionI64)?;
    {
        let func = partition_module.function(partition_kernel_i64::KERNEL_ENTRY)?;

        let view_keys = keys_gpu.view();
        let mut view_pids = partition_ids.view_mut();
        let mut view_counts = counts.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_output(&mut view_pids);
        args.push_output(&mut view_counts);
        args.push_scalar_u32(n_rows);

        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);
        launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
    }

    // P1b-stage8: joint helper, 2 syncs → 1.
    let (offsets, offsets_gpu): (Vec<u32>, GpuVec<u32>) =
        partition_offsets::compute_and_upload_partition_offsets_async(&counts, stream.raw())?;

    // ---- Scatter (i64 keys + f64 vals — no conversion needed) ----
    let mut scatter_keys: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_rows as usize, stream.raw())?;
    let mut scatter_vals: GpuVec<f64> = GpuVec::<f64>::zeros_async(n_rows as usize, stream.raw())?;
    let scatter_module = get_or_build_module(&KernelSpec::ScatterI64)?;
    {
        let func = scatter_module.function(scatter_kernel_i64::KERNEL_ENTRY)?;
        let mut cursors: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;

        let view_keys = keys_gpu.view();
        let view_vals = vals_gpu.view();
        let view_pids = partition_ids.view();
        let view_offsets = offsets_gpu.view();
        let mut view_cursors = cursors.view_mut();
        let mut view_sk = scatter_keys.view_mut();
        let mut view_sv = scatter_vals.view_mut();

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
        launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
    }

    // ---- Reduce (CAS-loop float MIN/MAX, i64-key) ----
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice_async(&offsets, stream.raw())?;
    let block_groups = BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let mut out_keys_gpu: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_vals_gpu: GpuVec<f64> = GpuVec::<f64>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros_async(n_out_slots, stream.raw())?;
    let mut spill: GpuVec<u32> = GpuVec::<u32>::zeros_async(1, stream.raw())?;

    let reduce_module = get_or_build_module(&KernelSpec::ReduceMinMaxFloatI64(
        ReduceFloatKey::from_pair(op, float_dtype),
    ))?;
    {
        let entry = minmax_float_i64_entry(op, float_dtype);
        let func = reduce_module.function(&entry)?;

        let view_pk = scatter_keys.view();
        let view_pv = scatter_vals.view();
        let view_po = offsets_kp1_gpu.view();
        let mut view_ok = out_keys_gpu.view_mut();
        let mut view_ov = out_vals_gpu.view_mut();
        let mut view_os = out_set_gpu.view_mut();
        let mut view_sp = spill.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_pk);
        args.push_input(&view_pv);
        args.push_input(&view_po);
        args.push_output(&mut view_ok);
        args.push_output(&mut view_ov);
        args.push_output(&mut view_os);
        args.push_output(&mut view_sp);

        launch_with_geometry(func, num_partitions, REDUCE_BLOCK_THREADS, 0, &stream, &mut args)?;
    }

    // Stage-4 (P1b): pinned D2H; sync once.
    let pinned_keys = out_keys_gpu.to_pinned_async(stream.raw())?;
    let pinned_vals = out_vals_gpu.to_pinned_async(stream.raw())?;
    let pinned_set = out_set_gpu.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let spill_count = spill.to_vec()?[0];
    if spill_count > 0 {
        return Err(BoltError::Other(format!(
            "partition_reduce spill: {} rows exceeded MAX_PROBES; result may be incorrect",
            spill_count
        )));
    }
    let host_out_keys: Vec<i64> = pinned_keys.as_slice().to_vec();
    let host_out_vals: Vec<f64> = pinned_vals.as_slice().to_vec();
    let host_out_set: Vec<u8> = pinned_set.as_slice().to_vec();

    let mut rows: Vec<(i64, f64)> = Vec::new();
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
    let mut out_v: Vec<f64> = Vec::with_capacity(rows.len());
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
            Arc::new(Float64Array::from(out_v)),
        ],
    )
    .map_err(|e| {
        BoltError::Other(format!(
            "groupby_tier2_twokey_minmax_float_exec: build error: {e}"
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
// Host-only eligibility-gate tests for the two-key float MIN/MAX exec.
//
// Mirror of the integer-MIN/MAX exec's host tests; we just swap to the
// f64-value path. Numerical correctness is GPU-bound and lives in the
// e2e suite.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::Field;
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};
    use arrow_array::Int64Array;

    /// Plan for `SELECT k1, k2, MIN(v) FROM t GROUP BY k1, k2`. Caller
    /// picks the value column dtype to probe different reject paths.
    fn build_twokey_float_minmax_plan(op_is_min: bool, val_dtype: DataType) -> PhysicalPlan {
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
                input_has_validity: Vec::new(),
            },
        }
    }

    fn twokey_float_batch(n: usize) -> RecordBatch {
        let k1: Vec<i32> = (0..n as i32).collect();
        let k2: Vec<i32> = (0..n as i32).map(|i| i + 1).collect();
        let v: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k1", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(k1)) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(k2)) as arrow_array::ArrayRef,
                Arc::new(Float64Array::from(v)) as arrow_array::ArrayRef,
            ],
        )
        .unwrap()
    }

    /// Non-Aggregate plan: reject.
    #[test]
    fn rejects_non_aggregate_plan() {
        let plan = PhysicalPlan::Union { inputs: vec![] };
        let batch = twokey_float_batch(0);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Single-key plan: reject (single-key sibling owns it).
    #[test]
    fn rejects_single_key_plan() {
        let mut plan = build_twokey_float_minmax_plan(true, DataType::Float64);
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.group_by = vec![0];
        }
        let batch = twokey_float_batch(300_000);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// SUM aggregate → reject.
    #[test]
    fn rejects_sum_aggregate() {
        let mut plan = build_twokey_float_minmax_plan(true, DataType::Float64);
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.aggregates = vec![AggregateExpr::Sum(Expr::Column("v".into()))];
        }
        let batch = twokey_float_batch(300_000);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Two aggregates → reject.
    #[test]
    fn rejects_two_aggregates() {
        let mut plan = build_twokey_float_minmax_plan(true, DataType::Float64);
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate
                .aggregates
                .push(AggregateExpr::Max(Expr::Column("v".into())));
        }
        let batch = twokey_float_batch(300_000);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Int64 value column → integer-minmax sibling, not this exec.
    #[test]
    fn rejects_int64_value_column() {
        let plan = build_twokey_float_minmax_plan(true, DataType::Int64);
        let n = 300_000;
        let k: Vec<i32> = (0..n as i32).collect();
        let v: Vec<i64> = (0..n).map(|i| i as i64).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k1", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(k.clone())) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(k)) as arrow_array::ArrayRef,
                Arc::new(Int64Array::from(v)) as arrow_array::ArrayRef,
            ],
        )
        .unwrap();
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Float32 value column → v0 doesn't accept it yet; reject.
    #[test]
    fn rejects_float32_value_column() {
        let plan = build_twokey_float_minmax_plan(true, DataType::Float32);
        let n = 300_000;
        let k: Vec<i32> = (0..n as i32).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k1", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Float32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(k.clone())) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(k)) as arrow_array::ArrayRef,
                Arc::new(arrow_array::Float32Array::from(v)) as arrow_array::ArrayRef,
            ],
        )
        .unwrap();
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Below row threshold → defer.
    #[test]
    fn rejects_below_row_threshold() {
        let plan = build_twokey_float_minmax_plan(true, DataType::Float64);
        let batch = twokey_float_batch(1_024);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// `pre` kernel present → defer.
    #[test]
    fn rejects_plan_with_pre_kernel() {
        use crate::plan::physical_plan::KernelSpec;
        let mut plan = build_twokey_float_minmax_plan(true, DataType::Float64);
        if let PhysicalPlan::Aggregate { pre, .. } = &mut plan {
            *pre = Some(KernelSpec {
                inputs: vec![],
                outputs: vec![],
                ops: vec![],
                predicate: None,
                register_count: 0,
                input_has_validity: vec![],
                output_has_validity: vec![],
            });
        }
        let batch = twokey_float_batch(300_000);
        assert!(try_execute(&plan, &batch).is_none());
    }
}

// ---------------------------------------------------------------------------
// Module-cache mechanics tests. Skip on CPU-only hosts.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod cache_tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn second_call_same_spec_is_cache_hit() {
        let m1 = match get_or_build_module(&KernelSpec::PartitionI64) {
            Ok(m) => m,
            Err(_) => return,
        };
        let after_first = LOAD_COUNT.load(Ordering::SeqCst);
        let m2 = get_or_build_module(&KernelSpec::PartitionI64).expect("hit");
        assert_eq!(LOAD_COUNT.load(Ordering::SeqCst), after_first);
        assert_eq!(m1.raw(), m2.raw());
    }

    #[test]
    fn op_dtype_combinations_are_distinct_cache_keys() {
        let _ = match get_or_build_module(&KernelSpec::ReduceMinMaxFloatI64(
            ReduceFloatKey::MinF64,
        )) {
            Ok(m) => m,
            Err(_) => return,
        };
        let _ = get_or_build_module(&KernelSpec::ReduceMinMaxFloatI64(ReduceFloatKey::MaxF64))
            .expect("max-f64 build");
        let baseline = LOAD_COUNT.load(Ordering::SeqCst);
        let _ = get_or_build_module(&KernelSpec::ReduceMinMaxFloatI64(ReduceFloatKey::MinF64))
            .expect("min-f64 hit");
        let _ = get_or_build_module(&KernelSpec::ReduceMinMaxFloatI64(ReduceFloatKey::MaxF64))
            .expect("max-f64 hit");
        assert_eq!(LOAD_COUNT.load(Ordering::SeqCst), baseline);
    }
}

// ---------------------------------------------------------------------------
// Stage-4 (P1b) async smoke test.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod stage4_tests {
    use super::*;
    use crate::plan::logical_plan::Field;
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn async_tier2_twokey_minmax_float_round_trip() {
        let n: usize = 300_000;
        let k1: Vec<i32> = (0..n as i32).map(|i| i % 64).collect();
        let k2: Vec<i32> = (0..n as i32).map(|i| (i / 64) % 64).collect();
        let v: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![
                    ColumnIO { name: "k1".into(), dtype: DataType::Int32 },
                    ColumnIO { name: "k2".into(), dtype: DataType::Int32 },
                    ColumnIO { name: "v".into(), dtype: DataType::Float64 },
                ],
                group_by: vec![0, 1],
                aggregates: vec![AggregateExpr::Min(Expr::Column("v".into()))],
                output_schema: Schema::new(vec![
                    Field::new("k1", DataType::Int32, false),
                    Field::new("k2", DataType::Int32, false),
                    Field::new("min_v", DataType::Float64, true),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k1", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(k1)) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(k2)) as arrow_array::ArrayRef,
                Arc::new(Float64Array::from(v)) as arrow_array::ArrayRef,
            ],
        )
        .unwrap();
        let _ = try_execute(&plan, &batch);
    }
}
