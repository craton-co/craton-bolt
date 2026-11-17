// SPDX-License-Identifier: AGPL-3.0-only

//! Fused pre-projection + GROUP BY aggregate execution.
//!
//! Runs the GPU pre-projection kernel (and optional predicate-only mask kernel)
//! from [`KernelSpec`], compacts surviving rows into an Arrow [`RecordBatch`]
//! preserving per-column validity bitmaps, and delegates the grouped reduction
//! to [`crate::exec::groupby::execute_groupby`].
//!
//! Delegating the post-pre stage to [`execute_groupby`](crate::exec::groupby::execute_groupby)
//! gives `pre = Some(_)` queries full feature parity with bare-column `GROUP BY`:
//!
//! - Single-column and lossless 2-column 64-bit packed keys (`Int32`, `Int64`,
//!   `Float32`, `Float64`, `Bool`, `Date32`, `Timestamp`), including all Tier-1
//!   shared-memory and Tier-2 hash-partitioned GPU fast paths.
//! - Wide composite keys (`> 64` bits or `3+` columns, plus `Decimal128`) via
//!   [`crate::exec::groupby_wide::execute_groupby_wide`].
//! - Sentinel-free fallback via [`crate::exec::groupby_valid::execute_groupby_valid`]
//!   when packed keys collide with `I64_EMPTY_SENTINEL` (`i64::MIN`, e.g.
//!   `(i32::MIN, 0)` or `-0.0`) or `NULL_GROUP_KEY` (`i64::MAX`).
//! - Checked `SUM(Int64)` overflow, grouped `Decimal128` `SUM`/`MIN`/`MAX`,
//!   float `MIN`/`MAX` CAS atomics, temporal `MIN`/`MAX`, and `VAR_*`/`STDDEV_*`.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
    Int32Array, Int64Array, RecordBatch, RecordBatchOptions, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray,
};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use bytemuck::Pod;

use crate::cuda::buffer::primitive_to_gpu;
use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::expr_agg;
use crate::exec::launch::{grid_x_for, CudaStream};
use crate::exec::module_cache;
use crate::exec::n_rows_to_u32;
use crate::jit::compile_ptx;
use crate::jit::hash_kernels::packed_validity_word_count;
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Literal, TimeUnit};
use crate::plan::physical_plan::{AggregateSpec, ColumnIO, KernelSpec, PhysicalPlan};

/// PTX entry-point name for the pre-projection kernel.
const PRE_KERNEL_ENTRY: &str = "bolt_pre_kernel";

/// PTX entry-point name for the predicate-only kernel that materialises the mask.
const PRE_PREDICATE_ENTRY: &str = "bolt_pre_predicate";

/// Threads per block for the pre-projection / predicate launches.
const PRE_BLOCK_SIZE: u32 = 256;

/// PV-stage-e: observability counter — increments once per agg launch that
/// dispatches through the native `_with_validity` GPU kernel path during a
/// `groupby_with_pre` execution. Forwarded from
/// [`crate::exec::groupby::NATIVE_VALIDITY_LAUNCHES`].
#[doc(hidden)]
pub static NATIVE_VALIDITY_LAUNCHES: AtomicUsize = AtomicUsize::new(0);

/// Execute an `Aggregate` plan that has BOTH a `pre` kernel AND a non-empty
/// `group_by`.
pub fn execute_groupby_with_pre(
    plan: &PhysicalPlan,
    table_batch: &RecordBatch,
) -> BoltResult<RecordBatch> {
    let (table, pre, aggregate) = match plan {
        PhysicalPlan::Aggregate {
            table,
            pre,
            aggregate,
        } => (table, pre, aggregate),
        other => {
            return Err(BoltError::Other(format!(
                "execute_groupby_with_pre: expected Aggregate plan, got {:?}",
                std::mem::discriminant(other)
            )))
        }
    };

    let pre_spec = pre.as_ref().ok_or_else(|| {
        BoltError::Other("execute_groupby_with_pre: pre kernel is None; use execute_groupby".into())
    })?;

    if aggregate.group_by.is_empty() {
        return Err(BoltError::Other(
            "execute_groupby_with_pre: aggregate has no GROUP BY columns; \
             use execute_aggregate_with_pre"
                .into(),
        ));
    }

    let (rewritten_plan, pre_batch) =
        materialize_pre_stage_batch(table, pre_spec, aggregate, table_batch)?;

    let before = crate::exec::groupby::NATIVE_VALIDITY_LAUNCHES.load(Ordering::Relaxed);
    let res = crate::exec::groupby::execute_groupby(&rewritten_plan, &pre_batch);
    let after = crate::exec::groupby::NATIVE_VALIDITY_LAUNCHES.load(Ordering::Relaxed);
    if after > before {
        NATIVE_VALIDITY_LAUNCHES.fetch_add(after - before, Ordering::Relaxed);
    }
    res
}

/// Run `pre_spec` on `table_batch` (projection + optional predicate mask),
/// compact surviving rows into a host [`RecordBatch`], and rewrite `aggregate`
/// into an equivalent `pre: None` [`PhysicalPlan::Aggregate`].
///
/// Shared with [`crate::exec::agg_with_pre::execute_aggregate_with_pre`] when a
/// scalar pre-aggregation carries temporal, boolean, or `Decimal128` columns.
pub(crate) fn materialize_pre_stage_batch(
    table: &str,
    pre_spec: &KernelSpec,
    aggregate: &AggregateSpec,
    table_batch: &RecordBatch,
) -> BoltResult<(PhysicalPlan, RecordBatch)> {
    let n_rows = table_batch.num_rows();
    let compacted = run_pre_stage(pre_spec, table_batch, n_rows)?;
    let n_compacted = compacted.n_rows(n_rows);
    let name_to_pre_ord = build_pre_output_index(pre_spec)?;

    // Validate and remap GROUP BY ordinals into `pre_spec.outputs` order.
    let mut rewritten_group_by: Vec<usize> = Vec::with_capacity(aggregate.group_by.len());
    for &key_ord in &aggregate.group_by {
        let key_io = aggregate.inputs.get(key_ord).ok_or_else(|| {
            BoltError::Plan(format!(
                "execute_groupby_with_pre: group_by ordinal {} out of range (only {} inputs)",
                key_ord,
                aggregate.inputs.len()
            ))
        })?;
        let &pre_ord = name_to_pre_ord.get(&key_io.name).ok_or_else(|| {
            BoltError::Plan(format!(
                "execute_groupby_with_pre: GROUP BY key '{}' not among pre kernel outputs",
                key_io.name
            ))
        })?;
        let pre_out = &pre_spec.outputs[pre_ord];
        if pre_out.dtype != key_io.dtype {
            return Err(BoltError::Type(format!(
                "GROUP BY key '{}' pre output dtype {:?} does not match plan dtype {:?}",
                key_io.name, pre_out.dtype, key_io.dtype
            )));
        }
        if !host_col_matches_dtype(&compacted.cols[pre_ord], key_io.dtype) {
            return Err(BoltError::Type(format!(
                "GROUP BY key '{}' compacted column dtype does not match plan dtype {:?}",
                key_io.name, key_io.dtype
            )));
        }
        rewritten_group_by.push(pre_ord);
    }

    // Resolve any non-bare aggregate expressions (hand-crafted plans or literal
    // `COUNT(1)` before materialising `compacted.cols` into Arrow arrays).
    let n_group_keys = aggregate.group_by.len();
    let mut extra_cols: Vec<(ColumnIO, HostCol)> = Vec::new();
    let mut rewritten_aggs: Vec<AggregateExpr> = Vec::with_capacity(aggregate.aggregates.len());
    let mut lazy_env: Option<Vec<(String, expr_agg::HostColumn)>> = None;

    for (i, agg) in aggregate.aggregates.iter().enumerate() {
        let inner = inner_expr_of(agg);
        if let Some(name) = expr_agg::try_bare_column(inner) {
            let &pre_ord = name_to_pre_ord.get(name).ok_or_else(|| {
                BoltError::Plan(format!(
                    "aggregate input '{}' not found among pre kernel outputs",
                    name
                ))
            })?;
            let pre_dtype = pre_spec.outputs[pre_ord].dtype;
            if let Some(plan_io) = aggregate.inputs.iter().find(|c| c.name == name) {
                if plan_io.dtype != pre_dtype {
                    return Err(BoltError::Type(format!(
                        "aggregate input '{}' pre output dtype {:?} does not match plan dtype {:?}",
                        name, pre_dtype, plan_io.dtype
                    )));
                }
            }
            if !host_col_matches_dtype(&compacted.cols[pre_ord], pre_dtype) {
                return Err(BoltError::Type(format!(
                    "aggregate input '{}' compacted dtype does not match plan dtype {:?}",
                    name, pre_dtype
                )));
            }
            rewritten_aggs.push(agg.clone());
            continue;
        }

        if let AggregateExpr::Count(Expr::Literal(lit)) = agg {
            if !matches!(lit, Literal::Null) {
                let synth_name = format!("__bolt_pre_count_{i}");
                extra_cols.push((
                    ColumnIO {
                        name: synth_name.clone(),
                        dtype: DataType::Int64,
                    },
                    HostCol {
                        values: HostColValues::I64(vec![1i64; n_compacted]),
                        validity: None,
                    },
                ));
                rewritten_aggs.push(AggregateExpr::Count(Expr::Column(synth_name)));
                continue;
            }
        }

        // Slow path: evaluate expression over compacted pre outputs via `expr_agg`.
        let eval_dtype = slow_path_eval_dtype(agg, aggregate, n_group_keys, i)?;
        if lazy_env.is_none() {
            let wrapped: Vec<(String, expr_agg::HostColumn)> = compacted
                .cols
                .iter()
                .enumerate()
                .map(|(j, c)| (pre_spec.outputs[j].name.clone(), to_expr_host(c)))
                .collect();
            lazy_env = Some(wrapped);
        }
        let wrapped = lazy_env.as_ref().expect("populated above");
        let env: expr_agg::ColumnEnv<'_> = wrapped.iter().map(|(n, c)| (n.clone(), c)).collect();
        let evaluated = expr_agg::eval_expr(inner, &env, eval_dtype, n_compacted)?;
        let host_col = from_expr_host(evaluated)?;
        let synth_name = format!("__bolt_pre_eval_{i}");
        extra_cols.push((
            ColumnIO {
                name: synth_name.clone(),
                dtype: eval_dtype,
            },
            host_col,
        ));
        rewritten_aggs.push(rebuild_agg_with_col(agg, synth_name));
    }
    drop(lazy_env);

    // Convert all compacted + synthetic columns into an Arrow RecordBatch.
    let total_cols = pre_spec.outputs.len() + extra_cols.len();
    let mut rewritten_inputs: Vec<ColumnIO> = Vec::with_capacity(total_cols);
    let mut arrow_fields: Vec<ArrowField> = Vec::with_capacity(total_cols);
    let mut arrow_cols: Vec<ArrayRef> = Vec::with_capacity(total_cols);

    for (io, host_col) in pre_spec.outputs.iter().cloned().zip(compacted.cols) {
        let (arr, has_nulls) = host_col.into_arrow_array()?;
        let arrow_dt = crate::exec::schema_convert::plan_dtype_to_arrow(io.dtype)?;
        arrow_fields.push(ArrowField::new(&io.name, arrow_dt, has_nulls));
        arrow_cols.push(arr);
        rewritten_inputs.push(io);
    }
    for (io, host_col) in extra_cols {
        let (arr, has_nulls) = host_col.into_arrow_array()?;
        let arrow_dt = crate::exec::schema_convert::plan_dtype_to_arrow(io.dtype)?;
        arrow_fields.push(ArrowField::new(&io.name, arrow_dt, has_nulls));
        arrow_cols.push(arr);
        rewritten_inputs.push(io);
    }

    // Propagate the planner's `input_has_validity` signal to the rewritten
    // `AggregateSpec` so `execute_groupby` exercises native `_with_validity`
    // GPU kernels when the planner flagged nullable inputs.
    let any_pre_input_has_validity = pre_spec.input_has_validity.iter().any(|&v| v);
    let input_has_validity = if any_pre_input_has_validity {
        let mut flags: Vec<bool> = arrow_cols.iter().map(|c| c.null_count() > 0).collect();
        if !flags.iter().any(|&b| b) && !flags.is_empty() {
            flags[0] = true;
        }
        flags
    } else {
        Vec::new()
    };

    let arrow_schema = Arc::new(ArrowSchema::new(arrow_fields));
    let options = RecordBatchOptions::new().with_row_count(Some(n_compacted));
    let pre_batch = RecordBatch::try_new_with_options(arrow_schema, arrow_cols, &options)
        .map_err(|e| BoltError::Other(format!("failed to build pre-stage RecordBatch: {e}")))?;

    let rewritten_spec = AggregateSpec {
        inputs: rewritten_inputs,
        group_by: rewritten_group_by,
        aggregates: rewritten_aggs,
        output_schema: aggregate.output_schema.clone(),
        input_has_validity,
    };
    let rewritten_plan = PhysicalPlan::Aggregate {
        table: table.to_string(),
        pre: None,
        aggregate: rewritten_spec,
    };

    Ok((rewritten_plan, pre_batch))
}

/// Determine the evaluation dtype for a slow-path (non-bare) aggregate input.
fn slow_path_eval_dtype(
    agg: &AggregateExpr,
    aggregate: &AggregateSpec,
    n_group_keys: usize,
    agg_idx: usize,
) -> BoltResult<DataType> {
    if n_group_keys + aggregate.aggregates.len() == aggregate.inputs.len() {
        if let Some(io) = aggregate.inputs.get(n_group_keys + agg_idx) {
            return Ok(io.dtype);
        }
    }
    match agg {
        AggregateExpr::Count(_) => Ok(DataType::Int64),
        AggregateExpr::Avg(_)
        | AggregateExpr::VarPop(_)
        | AggregateExpr::VarSamp(_)
        | AggregateExpr::StddevPop(_)
        | AggregateExpr::StddevSamp(_) => Ok(DataType::Float64),
        AggregateExpr::Sum(_) | AggregateExpr::Min(_) | AggregateExpr::Max(_) => aggregate
            .output_schema
            .fields
            .get(n_group_keys + agg_idx)
            .map(|f| f.dtype)
            .ok_or_else(|| {
                BoltError::Plan(format!(
                    "groupby_with_pre: output_schema missing field for aggregate index {}",
                    agg_idx
                ))
            }),
    }
}

/// Rebuild `agg` replacing its operand with `Expr::Column(col_name)`.
fn rebuild_agg_with_col(agg: &AggregateExpr, col_name: String) -> AggregateExpr {
    let col = Expr::Column(col_name);
    match agg {
        AggregateExpr::Count(_) => AggregateExpr::Count(col),
        AggregateExpr::Sum(_) => AggregateExpr::Sum(col),
        AggregateExpr::Min(_) => AggregateExpr::Min(col),
        AggregateExpr::Max(_) => AggregateExpr::Max(col),
        AggregateExpr::Avg(_) => AggregateExpr::Avg(col),
        AggregateExpr::VarPop(_) => AggregateExpr::VarPop(Box::new(col)),
        AggregateExpr::VarSamp(_) => AggregateExpr::VarSamp(Box::new(col)),
        AggregateExpr::StddevPop(_) => AggregateExpr::StddevPop(Box::new(col)),
        AggregateExpr::StddevSamp(_) => AggregateExpr::StddevSamp(Box::new(col)),
    }
}

// ===========================================================================
// Pre stage — GPU projection + optional predicate mask + host compaction.
// ===========================================================================

struct CompactedPreOutputs {
    cols: Vec<HostCol>,
    post_filter_rows: Option<usize>,
}

impl CompactedPreOutputs {
    fn n_rows(&self, source_rows: usize) -> usize {
        if let Some(c) = self.cols.first() {
            c.len()
        } else {
            self.post_filter_rows.unwrap_or(source_rows)
        }
    }
}

fn run_pre_stage(
    spec: &KernelSpec,
    table_batch: &RecordBatch,
    n_rows: usize,
) -> BoltResult<CompactedPreOutputs> {
    for io in &spec.outputs {
        if matches!(io.dtype, DataType::Utf8 | DataType::Extended(_)) {
            return Err(BoltError::Type(format!(
                "groupby_with_pre: pre kernel output dtype {:?} not supported",
                io.dtype
            )));
        }
    }

    let mut input_cols: Vec<PreCol> = Vec::with_capacity(spec.inputs.len());
    for io in &spec.inputs {
        let idx = table_batch.schema().index_of(&io.name).map_err(|e| {
            BoltError::Plan(format!(
                "pre kernel input column '{}' not present in table batch: {}",
                io.name, e
            ))
        })?;
        let arr = table_batch.column(idx);
        let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
        if arr_dtype != io.dtype {
            return Err(BoltError::Type(format!(
                "pre kernel input '{}' dtype mismatch: plan says {:?}, batch has {:?}",
                io.name, io.dtype, arr_dtype
            )));
        }
        input_cols.push(PreCol::upload(&io.name, arr.as_ref(), io.dtype)?);
    }

    for (i, col) in input_cols.iter_mut().enumerate() {
        if spec.input_has_validity.get(i).copied().unwrap_or(false) && !col.has_validity() {
            col.ensure_validity(n_rows)?;
        }
    }
    let input_has_validity: Vec<bool> = input_cols.iter().map(|c| c.has_validity()).collect();
    let any_input_validity: bool = input_has_validity.iter().any(|b| *b);
    let output_has_validity: Vec<bool> = if any_input_validity {
        vec![true; spec.outputs.len()]
    } else {
        vec![false; spec.outputs.len()]
    };

    let mut output_cols: Vec<PreCol> = Vec::with_capacity(spec.outputs.len());
    for io in &spec.outputs {
        output_cols.push(PreCol::alloc_zeros(io.dtype, n_rows, any_input_validity)?);
    }

    let stream = CudaStream::null_or_default();

    if !spec.outputs.is_empty() {
        let pre_spec_for_ptx = KernelSpec {
            input_has_validity: input_has_validity.clone(),
            output_has_validity: output_has_validity.clone(),
            ..spec.clone()
        };
        let module = module_cache::get_or_build_module(
            module_path!(),
            format!("pre_kernel:{}:{:?}", PRE_KERNEL_ENTRY, pre_spec_for_ptx),
            None,
            || compile_ptx(&pre_spec_for_ptx, PRE_KERNEL_ENTRY),
        )?;
        let function = module.function(PRE_KERNEL_ENTRY)?;

        let mut device_ptrs: Vec<CUdeviceptr> = Vec::with_capacity(
            input_cols.len() + output_cols.len() + input_cols.len() + output_cols.len(),
        );
        for c in &input_cols {
            device_ptrs.push(c.device_ptr());
        }
        for c in &output_cols {
            device_ptrs.push(c.device_ptr());
        }
        for (i, has) in input_has_validity.iter().enumerate() {
            if *has {
                let vp = input_cols[i].validity_device_ptr().ok_or_else(|| {
                    BoltError::Other(
                        "internal: input flagged with validity has no valid_mask device pointer"
                            .into(),
                    )
                })?;
                device_ptrs.push(vp);
            }
        }
        for (i, has) in output_has_validity.iter().enumerate() {
            if *has {
                let vp = output_cols[i].validity_device_ptr().ok_or_else(|| {
                    BoltError::Other(
                        "internal: output flagged with validity has no valid_mask device pointer"
                            .into(),
                    )
                })?;
                device_ptrs.push(vp);
            }
        }
        let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;

        let mut kernel_params: Vec<*mut c_void> = Vec::with_capacity(device_ptrs.len() + 1);
        for p in device_ptrs.iter_mut() {
            kernel_params.push(p as *mut CUdeviceptr as *mut c_void);
        }
        kernel_params.push(&mut n_rows_u32 as *mut u32 as *mut c_void);

        if n_rows > 0 {
            let grid_x = grid_x_for(n_rows_u32, PRE_BLOCK_SIZE);
            // SAFETY: `function` is borrowed from a live `CudaModule`; every entry
            // of `kernel_params` points into `device_ptrs` or `n_rows_u32`, both of
            // which outlive the launch + synchronize below.
            unsafe {
                cuda_sys::check(cuda_sys::cuLaunchKernel(
                    function.raw(),
                    grid_x,
                    1,
                    1,
                    PRE_BLOCK_SIZE,
                    1,
                    1,
                    0,
                    stream.raw(),
                    kernel_params.as_mut_ptr(),
                    ptr::null_mut(),
                ))?;
            }
            stream.synchronize()?;
        }
    }

    let host_mask: Option<Vec<bool>> = if spec.predicate.is_some() {
        let pred_spec = KernelSpec {
            input_has_validity: input_has_validity.clone(),
            ..spec.clone()
        };
        let pred_module = module_cache::get_or_build_module(
            module_path!(),
            format!("predicate_kernel:{}:{:?}", PRE_PREDICATE_ENTRY, pred_spec),
            None,
            || crate::jit::scan_kernel::compile_predicate_kernel(&pred_spec, PRE_PREDICATE_ENTRY),
        )?;
        let pred_function = pred_module.function(PRE_PREDICATE_ENTRY)?;

        let mask = crate::exec::compact::alloc_mask_buffer(n_rows)?;
        let input_ptrs: Vec<CUdeviceptr> = input_cols.iter().map(|c| c.device_ptr()).collect();
        let validity_ptrs: Vec<CUdeviceptr> = input_cols
            .iter()
            .filter_map(|c| c.validity_device_ptr())
            .collect();
        crate::exec::compact::launch_predicate_kernel(
            pred_function,
            &input_ptrs,
            mask.device_ptr(),
            &validity_ptrs,
            n_rows_to_u32(n_rows)?,
            &stream,
        )?;
        Some(crate::exec::compact::download_mask(
            mask.device_ptr(),
            n_rows,
            &stream,
        )?)
    } else {
        None
    };

    drop(input_cols);

    let post_filter_rows = host_mask
        .as_ref()
        .map(|m| m.iter().filter(|&&keep| keep).count());

    let mut cols: Vec<HostCol> = Vec::with_capacity(output_cols.len());
    for col in output_cols {
        let host = col.to_host_col(n_rows)?;
        let compact = match &host_mask {
            Some(mask) => host.compact(mask)?,
            None => host,
        };
        cols.push(compact);
    }

    Ok(CompactedPreOutputs {
        cols,
        post_filter_rows,
    })
}

fn build_pre_output_index(pre_spec: &KernelSpec) -> BoltResult<HashMap<String, usize>> {
    let mut map: HashMap<String, usize> = HashMap::with_capacity(pre_spec.outputs.len());
    for (j, io) in pre_spec.outputs.iter().enumerate() {
        if map.insert(io.name.clone(), j).is_some() {
            return Err(BoltError::Other(format!(
                "execute_groupby_with_pre: duplicate pre output name '{}'",
                io.name
            )));
        }
    }
    Ok(map)
}

/// Stage C: pack a `Vec<u8>` validity stream (`1` = valid, `0` = null,
/// one byte per row, Arrow-style row-major) into a packed-bit `Vec<u32>`
/// suitable for upload to the GPU.
///
/// Consumed by [`crate::exec::groupby`] and [`crate::exec::groupby_valid`] on
/// their native-validity kernel paths.
pub(crate) fn pack_validity_bits(bytes: &[u8]) -> Vec<u32> {
    let n_words = packed_validity_word_count(bytes.len());
    let mut out: Vec<u32> = vec![0u32; n_words];
    for (i, &b) in bytes.iter().enumerate() {
        if b != 0 {
            let word = i / 32;
            let bit = (i % 32) as u32;
            out[word] |= 1u32 << bit;
        }
    }
    out
}

fn inner_expr_of(agg: &AggregateExpr) -> &Expr {
    match agg {
        AggregateExpr::Sum(e)
        | AggregateExpr::Min(e)
        | AggregateExpr::Max(e)
        | AggregateExpr::Avg(e)
        | AggregateExpr::Count(e) => e,
        AggregateExpr::VarPop(e)
        | AggregateExpr::VarSamp(e)
        | AggregateExpr::StddevPop(e)
        | AggregateExpr::StddevSamp(e) => e.as_ref(),
    }
}

fn to_expr_host(c: &HostCol) -> expr_agg::HostColumn {
    let valid_at = |i: usize| -> bool {
        match &c.validity {
            Some(v) => v[i] != 0,
            None => true,
        }
    };
    fn lift<T: Copy>(values: &[T], valid_at: impl Fn(usize) -> bool) -> Vec<Option<T>> {
        values
            .iter()
            .enumerate()
            .map(|(i, x)| if valid_at(i) { Some(*x) } else { None })
            .collect()
    }
    match &c.values {
        HostColValues::I32(v) | HostColValues::Date32(v) => {
            expr_agg::HostColumn::I32(lift(v, valid_at))
        }
        HostColValues::I64(v) | HostColValues::Timestamp { values: v, .. } => {
            expr_agg::HostColumn::I64(lift(v, valid_at))
        }
        HostColValues::F32(v) => expr_agg::HostColumn::F32(lift(v, valid_at)),
        HostColValues::F64(v) => expr_agg::HostColumn::F64(lift(v, valid_at)),
        HostColValues::Bool(v) => {
            let lifted = v
                .iter()
                .enumerate()
                .map(|(i, &b)| if valid_at(i) { Some(b != 0) } else { None })
                .collect();
            expr_agg::HostColumn::Bool(lifted)
        }
        HostColValues::Decimal128 { values, scale, .. } => {
            let pow = 10f64.powi(*scale as i32);
            let lifted = values
                .iter()
                .enumerate()
                .map(|(i, &v)| if valid_at(i) { Some((v as f64) / pow) } else { None })
                .collect();
            expr_agg::HostColumn::F64(lifted)
        }
    }
}

fn from_expr_host(c: expr_agg::HostColumn) -> BoltResult<HostCol> {
    fn split<T: Copy + Default>(v: Vec<Option<T>>) -> (Vec<T>, Option<Vec<u8>>) {
        let any_null = v.iter().any(|x| x.is_none());
        if !any_null {
            return (v.into_iter().map(|x| x.unwrap_or_default()).collect(), None);
        }
        let mut values: Vec<T> = Vec::with_capacity(v.len());
        let mut validity: Vec<u8> = Vec::with_capacity(v.len());
        for x in v.into_iter() {
            match x {
                Some(val) => {
                    values.push(val);
                    validity.push(1);
                }
                None => {
                    values.push(T::default());
                    validity.push(0);
                }
            }
        }
        (values, Some(validity))
    }
    match c {
        expr_agg::HostColumn::I32(v) => {
            let (vals, valid) = split::<i32>(v);
            Ok(HostCol {
                values: HostColValues::I32(vals),
                validity: valid,
            })
        }
        expr_agg::HostColumn::I64(v) => {
            let (vals, valid) = split::<i64>(v);
            Ok(HostCol {
                values: HostColValues::I64(vals),
                validity: valid,
            })
        }
        expr_agg::HostColumn::F32(v) => {
            let (vals, valid) = split::<f32>(v);
            Ok(HostCol {
                values: HostColValues::F32(vals),
                validity: valid,
            })
        }
        expr_agg::HostColumn::F64(v) => {
            let (vals, valid) = split::<f64>(v);
            Ok(HostCol {
                values: HostColValues::F64(vals),
                validity: valid,
            })
        }
        expr_agg::HostColumn::Bool(v) => {
            let as_u8: Vec<Option<u8>> = v
                .into_iter()
                .map(|opt| opt.map(|b| if b { 1u8 } else { 0u8 }))
                .collect();
            let (vals, valid) = split::<u8>(as_u8);
            Ok(HostCol {
                values: HostColValues::Bool(vals),
                validity: valid,
            })
        }
        expr_agg::HostColumn::Utf8(_) => Err(BoltError::Type(
            "groupby_with_pre: Utf8 aggregate inputs not supported by the \
             primitive reduction path"
                .into(),
        )),
    }
}

// ===========================================================================
// Heterogeneous device + host columns for the pre kernel.
// ===========================================================================

struct PreCol {
    values: PreColValues,
    valid_mask: Option<GpuVec<u8>>,
}

enum PreColValues {
    I32(GpuVec<i32>),
    I64(GpuVec<i64>),
    F32(GpuVec<f32>),
    F64(GpuVec<f64>),
    Bool(GpuVec<u8>),
    Date32(GpuVec<i32>),
    Timestamp {
        values: GpuVec<i64>,
        unit: TimeUnit,
        tz: Option<&'static str>,
    },
    Decimal128 {
        values: GpuVec<u64>,
        precision: u8,
        scale: i8,
    },
}

impl PreCol {
    fn upload(name: &str, arr: &dyn Array, dtype: DataType) -> BoltResult<Self> {
        let n = arr.len();
        let valid_mask = if arr.null_count() > 0 {
            let mut bytes: Vec<u8> = Vec::with_capacity(n);
            for i in 0..n {
                bytes.push(if arr.is_null(i) { 0 } else { 1 });
            }
            Some(GpuVec::<u8>::from_slice(&bytes)?)
        } else {
            None
        };
        let values = match dtype {
            DataType::Int32 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| downcast_err("input", "Int32"))?;
                PreColValues::I32(GpuVec::from_buffer(primitive_to_gpu(pa)?))
            }
            DataType::Int64 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| downcast_err("input", "Int64"))?;
                PreColValues::I64(GpuVec::from_buffer(primitive_to_gpu(pa)?))
            }
            DataType::Float32 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| downcast_err("input", "Float32"))?;
                PreColValues::F32(GpuVec::from_buffer(primitive_to_gpu(pa)?))
            }
            DataType::Float64 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| downcast_err("input", "Float64"))?;
                PreColValues::F64(GpuVec::from_buffer(primitive_to_gpu(pa)?))
            }
            DataType::Bool => {
                let ba = arr
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| downcast_err("input", "Bool"))?;
                let mut bytes: Vec<u8> = Vec::with_capacity(n);
                for i in 0..n {
                    bytes.push(if !ba.is_null(i) && ba.value(i) { 1 } else { 0 });
                }
                PreColValues::Bool(GpuVec::<u8>::from_slice(&bytes)?)
            }
            DataType::Date32 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .ok_or_else(|| downcast_err("input", "Date32"))?;
                PreColValues::Date32(GpuVec::from_buffer(primitive_to_gpu(pa)?))
            }
            DataType::Timestamp(unit, tz) => {
                let values = match unit {
                    TimeUnit::Second => {
                        let pa = arr
                            .as_any()
                            .downcast_ref::<TimestampSecondArray>()
                            .ok_or_else(|| downcast_err("input", "TimestampSecond"))?;
                        GpuVec::from_buffer(primitive_to_gpu(pa)?)
                    }
                    TimeUnit::Millisecond => {
                        let pa = arr
                            .as_any()
                            .downcast_ref::<TimestampMillisecondArray>()
                            .ok_or_else(|| downcast_err("input", "TimestampMillisecond"))?;
                        GpuVec::from_buffer(primitive_to_gpu(pa)?)
                    }
                    TimeUnit::Microsecond => {
                        let pa = arr
                            .as_any()
                            .downcast_ref::<TimestampMicrosecondArray>()
                            .ok_or_else(|| downcast_err("input", "TimestampMicrosecond"))?;
                        GpuVec::from_buffer(primitive_to_gpu(pa)?)
                    }
                    TimeUnit::Nanosecond => {
                        let pa = arr
                            .as_any()
                            .downcast_ref::<TimestampNanosecondArray>()
                            .ok_or_else(|| downcast_err("input", "TimestampNanosecond"))?;
                        GpuVec::from_buffer(primitive_to_gpu(pa)?)
                    }
                };
                PreColValues::Timestamp { values, unit, tz }
            }
            DataType::Decimal128(precision, scale) => {
                let da = arr
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or_else(|| downcast_err("input", "Decimal128"))?;
                if let ArrowDataType::Decimal128(ap, as_) = da.data_type() {
                    if *ap != precision || *as_ != scale {
                        return Err(BoltError::Type(format!(
                            "Decimal128 column '{}' upload: plan dtype \
                             Decimal128({precision}, {scale}) disagrees with Arrow \
                             dtype Decimal128({ap}, {as_})",
                            name
                        )));
                    }
                }
                let mut packed: Vec<u64> = Vec::with_capacity(2 * n);
                for i in 0..n {
                    let v: i128 = if da.is_null(i) { 0 } else { da.value(i) };
                    let bits = v as u128;
                    packed.push(bits as u64);
                    packed.push((bits >> 64) as u64);
                }
                PreColValues::Decimal128 {
                    values: GpuVec::<u64>::from_slice(&packed)?,
                    precision,
                    scale,
                }
            }
            DataType::Utf8 | DataType::Extended(_) => {
                return Err(BoltError::Type(format!(
                    "groupby_with_pre: pre kernel column dtype {:?} not supported",
                    dtype
                )))
            }
        };
        Ok(PreCol { values, valid_mask })
    }

    fn alloc_zeros(dtype: DataType, n: usize, with_validity: bool) -> BoltResult<Self> {
        let values = match dtype {
            DataType::Int32 => PreColValues::I32(GpuVec::<i32>::zeros(n)?),
            DataType::Int64 => PreColValues::I64(GpuVec::<i64>::zeros(n)?),
            DataType::Float32 => PreColValues::F32(GpuVec::<f32>::zeros(n)?),
            DataType::Float64 => PreColValues::F64(GpuVec::<f64>::zeros(n)?),
            DataType::Bool => PreColValues::Bool(GpuVec::<u8>::zeros(n)?),
            DataType::Date32 => PreColValues::Date32(GpuVec::<i32>::zeros(n)?),
            DataType::Timestamp(unit, tz) => PreColValues::Timestamp {
                values: GpuVec::<i64>::zeros(n)?,
                unit,
                tz,
            },
            DataType::Decimal128(precision, scale) => PreColValues::Decimal128 {
                values: GpuVec::<u64>::zeros(2 * n)?,
                precision,
                scale,
            },
            DataType::Utf8 | DataType::Extended(_) => {
                return Err(BoltError::Type(format!(
                    "groupby_with_pre: pre kernel output dtype {:?} not supported",
                    dtype
                )))
            }
        };
        let valid_mask = if with_validity {
            Some(GpuVec::<u8>::zeros(n)?)
        } else {
            None
        };
        Ok(PreCol { values, valid_mask })
    }

    fn device_ptr(&self) -> CUdeviceptr {
        match &self.values {
            PreColValues::I32(v) | PreColValues::Date32(v) => v.device_ptr(),
            PreColValues::I64(v) | PreColValues::Timestamp { values: v, .. } => v.device_ptr(),
            PreColValues::F32(v) => v.device_ptr(),
            PreColValues::F64(v) => v.device_ptr(),
            PreColValues::Bool(v) => v.device_ptr(),
            PreColValues::Decimal128 { values, .. } => values.device_ptr(),
        }
    }

    fn validity_device_ptr(&self) -> Option<CUdeviceptr> {
        self.valid_mask.as_ref().map(|m| m.device_ptr())
    }

    fn has_validity(&self) -> bool {
        self.valid_mask.is_some()
    }

    fn ensure_validity(&mut self, n_rows: usize) -> BoltResult<()> {
        if self.valid_mask.is_none() {
            self.valid_mask = Some(GpuVec::<u8>::from_slice(&vec![1u8; n_rows])?);
        }
        Ok(())
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_host_col(self, n_rows: usize) -> BoltResult<HostCol> {
        let PreCol { values, valid_mask } = self;
        let validity: Option<Vec<u8>> = match valid_mask {
            Some(v) => Some(copy_back::<u8>(&v, n_rows)?),
            None => None,
        };
        let host_values = match values {
            PreColValues::I32(v) => HostColValues::I32(copy_back::<i32>(&v, n_rows)?),
            PreColValues::I64(v) => HostColValues::I64(copy_back::<i64>(&v, n_rows)?),
            PreColValues::F32(v) => HostColValues::F32(copy_back::<f32>(&v, n_rows)?),
            PreColValues::F64(v) => HostColValues::F64(copy_back::<f64>(&v, n_rows)?),
            PreColValues::Bool(v) => HostColValues::Bool(copy_back::<u8>(&v, n_rows)?),
            PreColValues::Date32(v) => HostColValues::Date32(copy_back::<i32>(&v, n_rows)?),
            PreColValues::Timestamp { values, unit, tz } => HostColValues::Timestamp {
                values: copy_back::<i64>(&values, n_rows)?,
                unit,
                tz,
            },
            PreColValues::Decimal128 {
                values,
                precision,
                scale,
            } => {
                let raw = copy_back::<u64>(&values, 2 * n_rows)?;
                let mut i128_vals: Vec<i128> = Vec::with_capacity(n_rows);
                for pair in raw.chunks_exact(2) {
                    let lo = pair[0] as u128;
                    let hi = (pair[1] as u128) << 64;
                    i128_vals.push((lo | hi) as i128);
                }
                HostColValues::Decimal128 {
                    values: i128_vals,
                    precision,
                    scale,
                }
            }
        };
        Ok(HostCol {
            values: host_values,
            validity,
        })
    }
}

pub(crate) struct HostCol {
    values: HostColValues,
    validity: Option<Vec<u8>>,
}

enum HostColValues {
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<u8>),
    Date32(Vec<i32>),
    Timestamp {
        values: Vec<i64>,
        unit: TimeUnit,
        tz: Option<&'static str>,
    },
    Decimal128 {
        values: Vec<i128>,
        precision: u8,
        scale: i8,
    },
}

impl HostCol {
    fn len(&self) -> usize {
        match &self.values {
            HostColValues::I32(v) | HostColValues::Date32(v) => v.len(),
            HostColValues::I64(v) | HostColValues::Timestamp { values: v, .. } => v.len(),
            HostColValues::F32(v) => v.len(),
            HostColValues::F64(v) => v.len(),
            HostColValues::Bool(v) => v.len(),
            HostColValues::Decimal128 { values, .. } => values.len(),
        }
    }

    fn compact(self, mask: &[bool]) -> BoltResult<HostCol> {
        if mask.len() != self.len() {
            return Err(BoltError::Other(format!(
                "groupby_with_pre: mask length {} != column length {}",
                mask.len(),
                self.len()
            )));
        }
        let HostCol { values, validity } = self;
        let values = match values {
            HostColValues::I32(v) => HostColValues::I32(filter_vec(v, mask)),
            HostColValues::I64(v) => HostColValues::I64(filter_vec(v, mask)),
            HostColValues::F32(v) => HostColValues::F32(filter_vec(v, mask)),
            HostColValues::F64(v) => HostColValues::F64(filter_vec(v, mask)),
            HostColValues::Bool(v) => HostColValues::Bool(filter_vec(v, mask)),
            HostColValues::Date32(v) => HostColValues::Date32(filter_vec(v, mask)),
            HostColValues::Timestamp { values, unit, tz } => HostColValues::Timestamp {
                values: filter_vec(values, mask),
                unit,
                tz,
            },
            HostColValues::Decimal128 {
                values,
                precision,
                scale,
            } => HostColValues::Decimal128 {
                values: filter_vec(values, mask),
                precision,
                scale,
            },
        };
        let validity = validity.map(|v| filter_vec(v, mask));
        Ok(HostCol { values, validity })
    }

    #[cfg(test)]
    fn strip_nulls(self) -> HostCol {
        let HostCol { values, validity } = self;
        let Some(v) = validity else {
            return HostCol {
                values,
                validity: None,
            };
        };
        let keep: Vec<bool> = v.iter().map(|b| *b != 0).collect();
        let values = match values {
            HostColValues::I32(x) => HostColValues::I32(filter_vec(x, &keep)),
            HostColValues::I64(x) => HostColValues::I64(filter_vec(x, &keep)),
            HostColValues::F32(x) => HostColValues::F32(filter_vec(x, &keep)),
            HostColValues::F64(x) => HostColValues::F64(filter_vec(x, &keep)),
            HostColValues::Bool(x) => HostColValues::Bool(filter_vec(x, &keep)),
            HostColValues::Date32(x) => HostColValues::Date32(filter_vec(x, &keep)),
            HostColValues::Timestamp { values, unit, tz } => HostColValues::Timestamp {
                values: filter_vec(values, &keep),
                unit,
                tz,
            },
            HostColValues::Decimal128 {
                values,
                precision,
                scale,
            } => HostColValues::Decimal128 {
                values: filter_vec(values, &keep),
                precision,
                scale,
            },
        };
        HostCol {
            values,
            validity: None,
        }
    }

    /// Materialise this host column as an Arrow [`ArrayRef`], returning
    /// `(array, has_nulls)`. Only attaches a [`NullBuffer`] when `validity`
    /// actually carries at least one `0` byte, keeping non-null columns on the
    /// fast null-free downstream paths.
    fn into_arrow_array(self) -> BoltResult<(ArrayRef, bool)> {
        let HostCol { values, validity } = self;
        let null_buf: Option<NullBuffer> = match validity {
            Some(v) if v.contains(&0) => {
                let bools: Vec<bool> = v.into_iter().map(|b| b != 0).collect();
                Some(NullBuffer::from(bools))
            }
            _ => None,
        };
        let has_nulls = null_buf.is_some();

        let arr: ArrayRef = match values {
            HostColValues::I32(v) => Arc::new(Int32Array::new(v.into(), null_buf)),
            HostColValues::I64(v) => Arc::new(Int64Array::new(v.into(), null_buf)),
            HostColValues::F32(v) => Arc::new(Float32Array::new(v.into(), null_buf)),
            HostColValues::F64(v) => Arc::new(Float64Array::new(v.into(), null_buf)),
            HostColValues::Bool(v) => {
                let bits = BooleanBuffer::from(v.into_iter().map(|b| b != 0).collect::<Vec<_>>());
                Arc::new(BooleanArray::new(bits, null_buf))
            }
            HostColValues::Date32(v) => Arc::new(Date32Array::new(v.into(), null_buf)),
            HostColValues::Timestamp { values, unit, tz } => {
                let tz_arc: Option<Arc<str>> = tz.map(Arc::from);
                match unit {
                    TimeUnit::Second => Arc::new(
                        TimestampSecondArray::new(values.into(), null_buf)
                            .with_timezone_opt(tz_arc),
                    ),
                    TimeUnit::Millisecond => Arc::new(
                        TimestampMillisecondArray::new(values.into(), null_buf)
                            .with_timezone_opt(tz_arc),
                    ),
                    TimeUnit::Microsecond => Arc::new(
                        TimestampMicrosecondArray::new(values.into(), null_buf)
                            .with_timezone_opt(tz_arc),
                    ),
                    TimeUnit::Nanosecond => Arc::new(
                        TimestampNanosecondArray::new(values.into(), null_buf)
                            .with_timezone_opt(tz_arc),
                    ),
                }
            }
            HostColValues::Decimal128 {
                values,
                precision,
                scale,
            } => {
                let da = Decimal128Array::new(values.into(), null_buf)
                    .with_precision_and_scale(precision, scale)
                    .map_err(|e| {
                        BoltError::Type(format!(
                            "Decimal128 pre output precision/scale ({precision}, {scale}) invalid: {e}"
                        ))
                    })?;
                Arc::new(da)
            }
        };
        Ok((arr, has_nulls))
    }
}

fn host_col_matches_dtype(col: &HostCol, dtype: DataType) -> bool {
    match (&col.values, dtype) {
        (HostColValues::I32(_), DataType::Int32)
        | (HostColValues::I64(_), DataType::Int64)
        | (HostColValues::F32(_), DataType::Float32)
        | (HostColValues::F64(_), DataType::Float64)
        | (HostColValues::Bool(_), DataType::Bool)
        | (HostColValues::Date32(_), DataType::Date32) => true,
        (
            HostColValues::Timestamp {
                unit: u1, tz: tz1, ..
            },
            DataType::Timestamp(u2, tz2),
        ) => *u1 == u2 && *tz1 == tz2,
        (
            HostColValues::Decimal128 {
                precision: p1,
                scale: s1,
                ..
            },
            DataType::Decimal128(p2, s2),
        ) => *p1 == p2 && *s1 == s2,
        _ => false,
    }
}

fn filter_vec<T: Copy>(v: Vec<T>, mask: &[bool]) -> Vec<T> {
    v.into_iter()
        .zip(mask.iter())
        .filter_map(|(x, &k)| if k { Some(x) } else { None })
        .collect()
}

fn copy_back<T>(v: &GpuVec<T>, n_rows: usize) -> BoltResult<Vec<T>>
where
    T: Pod,
{
    let host = v.to_vec()?;
    if host.len() != n_rows {
        return Err(BoltError::Other(format!(
            "internal: device buffer length {} did not match expected {}",
            host.len(),
            n_rows
        )));
    }
    Ok(host)
}

fn downcast_err(role: &str, expected: &str) -> BoltError {
    BoltError::Type(format!(
        "groupby_with_pre: pre kernel {} could not be downcast to {}",
        role, expected
    ))
}

fn arrow_dtype_to_plan(d: &ArrowDataType) -> BoltResult<DataType> {
    crate::exec::schema_convert::arrow_dtype_to_plan(d)
}

#[cfg(test)]
mod null_propagation_tests {
    use super::*;
    use crate::exec::groupby_common::{download_pinned_f64, download_pinned_i32, download_pinned_i64};

    #[test]
    fn host_col_compact_validity_alignment() {
        let col = HostCol {
            values: HostColValues::I32(vec![1, 2, 3, 4]),
            validity: Some(vec![1, 0, 1, 0]),
        };
        let mask = vec![true, true, false, true];
        let compact = col.compact(&mask).expect("compact");
        match compact.values {
            HostColValues::I32(v) => assert_eq!(v, vec![1, 2, 4]),
            _ => panic!("dtype changed"),
        }
        assert_eq!(compact.validity.as_deref(), Some(&[1u8, 0, 0][..]));
    }

    #[test]
    fn host_col_strip_nulls_drops_invalid_rows() {
        let col = HostCol {
            values: HostColValues::F64(vec![1.0, 2.0, 3.0]),
            validity: Some(vec![1, 0, 1]),
        };
        let stripped = col.strip_nulls();
        assert!(stripped.validity.is_none());
        match stripped.values {
            HostColValues::F64(v) => assert_eq!(v, vec![1.0, 3.0]),
            _ => panic!("dtype changed"),
        }
    }

    #[test]
    fn host_col_into_arrow_array_preserves_temporal_and_decimal_nulls() {
        let dec_col = HostCol {
            values: HostColValues::Decimal128 {
                values: vec![1234, 0, -5678],
                precision: 10,
                scale: 2,
            },
            validity: Some(vec![1, 0, 1]),
        };
        let (arr, has_nulls) = dec_col.into_arrow_array().expect("decimal array");
        assert!(has_nulls);
        assert_eq!(arr.null_count(), 1);
        assert!(arr.is_null(1));
        let da = arr.as_any().downcast_ref::<Decimal128Array>().unwrap();
        assert_eq!(da.value(0), 1234);
        assert_eq!(da.value(2), -5678);

        let date_col = HostCol {
            values: HostColValues::Date32(vec![19_000, 19_001]),
            validity: Some(vec![1, 1]),
        };
        let (d_arr, d_nulls) = date_col.into_arrow_array().expect("date array");
        assert!(!d_nulls, "all-valid mask must not attach a NullBuffer");
        assert_eq!(d_arr.null_count(), 0);
    }

    #[test]
    fn to_expr_host_surfaces_nulls() {
        let col = HostCol {
            values: HostColValues::I64(vec![10, 20, 30]),
            validity: Some(vec![1, 0, 1]),
        };
        let lifted = to_expr_host(&col);
        match lifted {
            expr_agg::HostColumn::I64(v) => {
                assert_eq!(v, vec![Some(10), None, Some(30)]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn from_expr_host_preserves_validity() {
        let col = expr_agg::HostColumn::F32(vec![Some(1.5f32), None, Some(3.5f32)]);
        let out = from_expr_host(col).expect("ok");
        assert_eq!(out.validity.as_deref(), Some(&[1u8, 0, 1][..]));
        match out.values {
            HostColValues::F32(v) => assert_eq!(v, vec![1.5, 0.0, 3.5]),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn arrow_null_bearing_int64_array_extracts_validity_bytes() {
        let arr = Int64Array::from(vec![Some(1i64), None, Some(3i64), None, Some(5i64)]);
        assert_eq!(arr.null_count(), 2);
        let bytes: Vec<u8> = (0..arr.len())
            .map(|i| if arr.is_null(i) { 0 } else { 1 })
            .collect();
        assert_eq!(bytes, vec![1, 0, 1, 0, 1]);
    }

    #[test]
    fn pack_validity_bits_small_little_endian() {
        let bytes = vec![1u8, 0, 1, 0, 1];
        let packed = pack_validity_bits(&bytes);
        assert_eq!(packed.len(), 1, "5 rows fit in one u32 word");
        assert_eq!(packed[0], 0b10101, "bit 0 = row 0; bit 4 = row 4");
    }

    #[test]
    fn pack_validity_bits_crosses_word_boundary() {
        let mut bytes = vec![1u8; 35];
        bytes[33] = 0;
        let packed = pack_validity_bits(&bytes);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], 0xFFFF_FFFFu32);
        assert_eq!(packed[1], 0b101);
    }

    #[test]
    fn pack_validity_bits_round_trip() {
        let bytes: Vec<u8> = (0..100u8).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        let packed = pack_validity_bits(&bytes);
        for (i, &expected) in bytes.iter().enumerate() {
            let word = packed[i / 32];
            let bit = ((word >> (i % 32)) & 1) as u8;
            assert_eq!(bit, expected, "row {i} round-trip mismatch");
        }
    }

    #[test]
    fn pack_validity_bits_empty_pads_to_one_word() {
        let packed = pack_validity_bits(&[]);
        assert_eq!(packed, vec![0u32]);
    }

    #[test]
    #[ignore = "gpu:tier1"]
    fn v07_async_download_pinned_helpers_do_not_panic() {
        let stream = CudaStream::null_or_default();
        let Ok(v_i32) = GpuVec::<i32>::from_slice_async(&[1, 2, 3, 4], stream.raw()) else {
            return;
        };
        let _ = download_pinned_i32(&v_i32, &stream);
        let Ok(v_i64) = GpuVec::<i64>::from_slice_async(&[1i64, 2, 3, 4], stream.raw()) else {
            return;
        };
        let _ = download_pinned_i64(&v_i64, &stream);
        let Ok(v_f32) = GpuVec::<f32>::from_slice_async(&[1.0f32, 2.0, 3.0, 4.0], stream.raw())
        else {
            return;
        };
        let _ = crate::exec::groupby_common::download_pinned_f32(&v_f32, &stream);
        let Ok(v_f64) = GpuVec::<f64>::from_slice_async(&[1.0f64, 2.0, 3.0, 4.0], stream.raw())
        else {
            return;
        };
        let _ = download_pinned_f64(&v_f64, &stream);
    }

    #[test]
    #[ignore = "gpu:tier1"]
    fn v07_async_groupby_with_pre_round_trip() {
        use crate::Engine;

        let mut engine = Engine::new().expect("ctx");
        let keys: Vec<i32> = (0..9i32).map(|i| i % 3).collect();
        let vals: Vec<i64> = (0..9i64).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(keys)) as ArrayRef,
                Arc::new(Int64Array::from(vals)) as ArrayRef,
            ],
        )
        .unwrap();
        engine.register_table("t", batch).unwrap();

        let h = match engine.sql("SELECT k, SUM(v + v) FROM t GROUP BY k") {
            Ok(h) => h,
            Err(_) => return,
        };
        let out = h.record_batch();
        let mut expected: HashMap<i32, i64> = HashMap::new();
        for v in 0..9i64 {
            *expected.entry((v as i32) % 3).or_default() += v + v;
        }
        let ks = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let ss = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..out.num_rows() {
            let k = ks.value(i);
            let s = ss.value(i);
            assert_eq!(Some(&s), expected.get(&k), "k={k} sum={s}");
        }
    }
}
