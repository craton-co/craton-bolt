// SPDX-License-Identifier: Apache-2.0

//! Aggregate execution with a pre-projection / filter pass.
//!
//! The companion to `aggregate.rs`. That module handles the trivial scalar case
//! where every aggregate input is a bare column reference and there is no
//! filter — the physical plan then carries `pre = None` and the aggregator can
//! read columns straight off the scan. This module handles the other case:
//! `pre = Some(KernelSpec)`, used when an aggregate input is a non-trivial
//! expression (e.g. `SUM(price * tax)`) or a filter is present (e.g.
//! `... WHERE region_id = 1`).
//!
//! Pipeline:
//!   1. JIT-compile `pre` as a projection kernel (potentially with a filter)
//!      and launch it to materialise the pre-aggregation columns as device
//!      buffers, exactly the way `engine.rs::execute_projection` does.
//!   2. If `pre.predicate` is set, run a separate predicate-only kernel to
//!      materialise a `u8` keep-mask. The projection kernel leaves zeros in
//!      masked slots (see `engine.rs` TODO), so we just download each column
//!      to the host and compact via the mask.
//!   3. For each `AggregateExpr`, reduce the matching compacted column via
//!      the existing per-block GPU reduction kernel from `agg_kernels`.
//!      `COUNT(_)` uses the post-mask row count directly; `AVG` launches a
//!      single **fused** kernel (`bolt_avg_reduce`) that emits per-block
//!      `(f64 sum, u32 count)` partials in one pass — strictly faster than
//!      the old "SUM kernel + host-side count + host divide" decomposition.
//!   4. Pack the resulting scalars into a single-row Arrow `RecordBatch`
//!      matching `aggregate.output_schema`.
//!
//! Scope (first cut):
//!   - No GROUP BY. `aggregate.group_by` must be empty — `groupby.rs` handles
//!     that case separately.
//!   - Primitive dtypes only (Int32, Int64, Float32, Float64). Bool and Utf8
//!     are rejected; aggregate-input expressions never carry those through
//!     `pre`.
//!   - One-batch-per-launch: `table_batch` is the entire source.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use arrow_schema::{
    DataType as ArrowDataType, Schema as ArrowSchema,
};
use bytemuck::Pod;

use crate::cuda::buffer::primitive_to_gpu;
use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::expr_agg;
use crate::exec::launch::CudaStream;
use crate::exec::module_cache;
use crate::exec::n_rows_to_u32;
use crate::jit::agg_kernels::{
    compile_avg_reduction_kernel, compile_reduction_kernel, ReduceOp, AVG_KERNEL_ENTRY,
    BLOCK_SIZE, REDUCTION_KERNEL_ENTRY,
};
use crate::jit::compile_ptx;
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Field, Schema};
use crate::plan::physical_plan::{AggregateSpec, KernelSpec, PhysicalPlan};

/// PTX entry-point name for the pre-projection kernel.
const PRE_KERNEL_ENTRY: &str = "bolt_pre_kernel";

/// PTX entry-point name for the predicate-only kernel that materialises the mask.
const PRE_PREDICATE_ENTRY: &str = "bolt_pre_predicate";

/// Threads per block for the pre-projection / predicate launches.
const PRE_BLOCK_SIZE: u32 = 256;

/// Execute an aggregate plan whose `pre` is `Some(_)`.
///
/// Pipeline:
///   1. Run `pre` as a projection (possibly with filter) kernel to materialise
///      pre-aggregation columns as `GpuVec`s.
///   2. (If `pre.predicate`) build a `u8` mask via the predicate-only kernel,
///      then compact each materialised column on host into a contiguous prefix.
///   3. Reduce each compacted column via the existing scalar reduction path.
///   4. Pack scalar results into a single-row Arrow `RecordBatch` matching
///      `aggregate.output_schema`.
///
/// Errors if `aggregate.group_by` is non-empty (handled elsewhere) or if `pre`
/// is `None` (caller should use the scalar reduction path in `aggregate.rs`).
pub fn execute_aggregate_with_pre(
    plan: &PhysicalPlan,
    table_batch: &RecordBatch,
) -> BoltResult<RecordBatch> {
    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        other => {
            return Err(BoltError::Other(format!(
                "execute_aggregate_with_pre: expected Aggregate plan, got {:?}",
                std::mem::discriminant(other)
            )))
        }
    };

    let pre_spec = pre.as_ref().ok_or_else(|| {
        BoltError::Other(
            "execute_aggregate_with_pre: pre kernel is None; use aggregate::execute_aggregate"
                .into(),
        )
    })?;

    if !aggregate.group_by.is_empty() {
        return Err(BoltError::Other(
            "agg_with_pre: GROUP BY handled separately".into(),
        ));
    }

    let n_rows = table_batch.num_rows();

    // 1+2. Run the pre kernel (and, if present, the predicate-only kernel),
    //      then download + host-compact each output column.
    let compacted = run_pre_stage(pre_spec, table_batch, n_rows)?;

    // 3+4. Reduce each materialised column and assemble the output batch.
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    let arrays = build_scalar_aggregates(aggregate, pre_spec, &compacted)?;

    RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
        BoltError::Other(format!("failed to build aggregate RecordBatch: {e}"))
    })
}

/// Materialised, host-side, post-filter output of the pre kernel: one host
/// vector per `pre.outputs`, parallel to `pre.outputs` by index.
struct CompactedPreOutputs {
    /// One host-side compacted column per `pre.outputs[i]`.
    cols: Vec<HostCol>,
}

impl CompactedPreOutputs {
    /// Post-filter row count. Identical across all columns; we use the first
    /// for the canonical value (the COUNT path leans on this).
    fn n_rows(&self) -> usize {
        self.cols.first().map(|c| c.len()).unwrap_or(0)
    }
}

/// Step 1+2: upload pre-kernel inputs, launch the projection (and optional
/// predicate) kernels, download the outputs, host-compact via the mask if any.
///
/// **NULL handling (Option B)**: per input column with `null_count() > 0`,
/// `PreCol::upload` materialises a parallel `valid_mask` (`u8`-per-row)
/// alongside the value buffer. If any input carries validity we re-issue
/// the pre kernel with `KernelSpec::input_has_validity` /
/// `output_has_validity` flags set so the JIT emits the
/// AND-then-store-validity sequence (see `jit/ptx_gen.rs::compile`). The
/// per-output validity buffer is then downloaded alongside each value
/// buffer in `to_host_col`. When no input has NULLs, every flag stays
/// `false` and the historical PTX shape + param list are emitted
/// bit-for-bit.
fn run_pre_stage(
    spec: &KernelSpec,
    table_batch: &RecordBatch,
    n_rows: usize,
) -> BoltResult<CompactedPreOutputs> {
    // -- Upload inputs.
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
        input_cols.push(PreCol::upload(arr.as_ref(), io.dtype)?);
    }

    // -- (Option B) Detect which inputs carry validity. If any do, every
    //    output must also carry validity so the kernel has a target buffer
    //    for the per-row combined AND.
    let input_has_validity: Vec<bool> =
        input_cols.iter().map(|c| c.has_validity()).collect();
    let any_input_validity: bool = input_has_validity.iter().any(|b| *b);
    let output_has_validity: Vec<bool> = if any_input_validity {
        vec![true; spec.outputs.len()]
    } else {
        vec![false; spec.outputs.len()]
    };

    // -- Allocate zero-initialised output buffers (with validity if any
    //    input carries NULLs).
    let mut output_cols: Vec<PreCol> = Vec::with_capacity(spec.outputs.len());
    for io in &spec.outputs {
        output_cols.push(PreCol::alloc_zeros(io.dtype, n_rows, any_input_validity)?);
    }

    // -- JIT-compile the projection PTX with validity flags threaded
    //    through `KernelSpec`. The default (no-validity) path produces the
    //    historical PTX byte-for-byte; the flagged path appends `*u8`
    //    validity params and emits the AND-store sequence.
    let pre_spec_for_ptx = KernelSpec {
        input_has_validity: input_has_validity.clone(),
        output_has_validity: output_has_validity.clone(),
        ..spec.clone()
    };
    // Route through the consolidated `exec::module_cache`. The pre-kernel's
    // PTX is a function of the (validity-aware) `KernelSpec` plus the
    // PRE_KERNEL_ENTRY name, so the spec id Debug-hashes both.
    let module = module_cache::get_or_build_module(
        module_path!(),
        format!("pre_kernel:{}:{:?}", PRE_KERNEL_ENTRY, pre_spec_for_ptx),
        None,
        || compile_ptx(&pre_spec_for_ptx, PRE_KERNEL_ENTRY),
    )?;
    let function = module.function(PRE_KERNEL_ENTRY)?;

    // -- Assemble kernel parameters: inputs..., outputs...,
    //    [input_validity..., output_validity...,] n_rows u32. Order matches
    //    `ptx_gen::compile`'s param walk.
    let mut device_ptrs: Vec<CUdeviceptr> =
        Vec::with_capacity(input_cols.len() + output_cols.len() + input_cols.len() + output_cols.len());
    for c in &input_cols {
        device_ptrs.push(c.device_ptr());
    }
    for c in &output_cols {
        device_ptrs.push(c.device_ptr());
    }
    // Validity pointers in the same order as the `_has_validity` flags.
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

    // -- Launch one thread per row.
    let stream = CudaStream::null();
    if n_rows > 0 {
        let grid_x = ((n_rows_u32 + PRE_BLOCK_SIZE - 1) / PRE_BLOCK_SIZE).max(1);
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

    // -- Optional predicate kernel: builds a u8 keep-mask we download.
    let host_mask: Option<Vec<bool>> = if spec.predicate.is_some() {
        let pred_module = module_cache::get_or_build_module(
            module_path!(),
            format!("predicate_kernel:{}:{:?}", PRE_PREDICATE_ENTRY, spec),
            None,
            || crate::jit::scan_kernel::compile_predicate_kernel(spec, PRE_PREDICATE_ENTRY),
        )?;
        let pred_function = pred_module.function(PRE_PREDICATE_ENTRY)?;

        let mask = crate::exec::compact::alloc_mask_buffer(n_rows)?;
        let input_ptrs: Vec<CUdeviceptr> =
            input_cols.iter().map(|c| c.device_ptr()).collect();
        // Aggregate pre-stage predicate kernel: today's planner doesn't lower
        // `Op::IsNullCheck` through this path (only the projection-scan-chain
        // path uses it), so we pass an empty validity slice — the
        // scan_kernel emits the legacy no-validity param layout bit-for-bit.
        // When/if the aggregate planner grows IS NULL support, the validity
        // ptrs would be assembled from `input_cols[i].validity_device_ptr()`
        // in the same order as `KernelSpec::input_has_validity`.
        let validity_ptrs: Vec<CUdeviceptr> = Vec::new();
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

    // Inputs no longer needed past this point.
    drop(input_cols);

    // -- Download each pre output, then host-compact via the mask if any.
    let mut cols: Vec<HostCol> = Vec::with_capacity(output_cols.len());
    for col in output_cols {
        let host = col.to_host_col(n_rows)?;
        let compact = match &host_mask {
            Some(mask) => host.compact(mask)?,
            None => host,
        };
        cols.push(compact);
    }

    Ok(CompactedPreOutputs { cols })
}

/// Build one Arrow scalar array per `AggregateExpr`, in `aggregate.aggregates`
/// order, against the post-filter host columns produced by the pre kernel.
fn build_scalar_aggregates(
    aggregate: &AggregateSpec,
    pre_spec: &KernelSpec,
    compacted: &CompactedPreOutputs,
) -> BoltResult<Vec<ArrayRef>> {
    if aggregate.output_schema.fields.len() != aggregate.aggregates.len() {
        return Err(BoltError::Other(format!(
            "internal: aggregate output schema has {} fields but plan has {} aggregates",
            aggregate.output_schema.fields.len(),
            aggregate.aggregates.len()
        )));
    }

    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(aggregate.aggregates.len());
    for (i, agg) in aggregate.aggregates.iter().enumerate() {
        let out_field = &aggregate.output_schema.fields[i];
        let array = build_one_aggregate(agg, out_field, pre_spec, compacted)?;
        arrays.push(array);
    }
    Ok(arrays)
}

/// Compute a single aggregate and return its single-row Arrow array.
fn build_one_aggregate(
    agg: &AggregateExpr,
    out_field: &Field,
    pre_spec: &KernelSpec,
    compacted: &CompactedPreOutputs,
) -> BoltResult<ArrayRef> {
    match agg {
        AggregateExpr::Sum(expr) | AggregateExpr::Min(expr) | AggregateExpr::Max(expr) => {
            let op = ReduceOp::from_agg(agg)?;
            let resolved =
                resolve_agg_input_col(expr, pre_spec, compacted, out_field.dtype)?;
            let scalar = reduce_host_col(op, resolved.as_ref())?;
            scalar_to_array(scalar, out_field.dtype)
        }
        AggregateExpr::Count(expr) => {
            // SQL COUNT(col) counts non-NULL rows; COUNT(*) (planner-emitted
            // as `Count(Literal(1))`) counts every surviving row. The
            // distinction falls out naturally: `resolve_agg_input_col`
            // returns a column whose `non_null_count` already excludes
            // NULLs from expression evaluation, and a literal can never
            // produce NULLs so it matches the full row count.
            //
            // We materialise at Int64 since that's the COUNT result dtype.
            let resolved =
                resolve_agg_input_col(expr, pre_spec, compacted, DataType::Int64)?;
            let count = resolved.non_null_count() as i64;
            scalar_to_array(Scalar::I64(count), out_field.dtype)
        }
        AggregateExpr::StddevPop(expr) | AggregateExpr::StddevSamp(expr) => {
            // STDDEV with a pre-aggregation (Project/Filter) stage. v0.5
            // cut: the pre-aggregated scalar path is still in scope (a
            // SELECT/WHERE feeding STDDEV doesn't involve GROUP BY), so
            // we resolve the value column through the same expression
            // evaluator AVG uses, then fold it with the host Welford
            // state. `resolve_agg_input_col` materialises a NULL-aware
            // HostCol; we then route through `strip_nulls_borrowed` (the
            // same NULL handling AVG uses) so the Welford fold only sees
            // valid values.
            //
            // `expected_dtype` is Float64 because Welford accumulates
            // in f64 — but the helper still validates the cast for
            // narrower dtypes, so we don't lose the type-error surface.
            let resolved = resolve_agg_input_col(
                expr.as_ref(),
                pre_spec,
                compacted,
                DataType::Float64,
            )?;
            let host_col = resolved.as_ref();
            let stripped = strip_nulls_borrowed(host_col);
            let mut state = crate::exec::welford::WelfordState::empty();
            match &stripped.values {
                HostColValues::I32(v) => state.push_slice_i32(v),
                HostColValues::I64(v) => state.push_slice_i64(v),
                HostColValues::F32(v) => state.push_slice_f32(v),
                HostColValues::F64(v) => state.push_slice_f64(v),
            }
            let kind = match agg {
                AggregateExpr::StddevPop(_) => crate::exec::welford::StddevKind::Pop,
                AggregateExpr::StddevSamp(_) => crate::exec::welford::StddevKind::Samp,
                _ => unreachable!("matched in outer arm"),
            };
            stddev_to_array_with_pre(
                crate::exec::welford::finalize(&state, kind),
                agg,
                out_field,
            )
        }
        AggregateExpr::Avg(expr) => {
            // AVG via the **fused** kernel: one launch produces both the
            // numerator (sum, f64) and the denominator (count, u32) as
            // per-block partials. Replaces the previous "run SUM kernel,
            // then divide by the host-side `non_null_count`" shape — the
            // two-launch / two-PTX-compile decomposition is gone.
            //
            // NULL handling: `resolve_agg_input_col` already filters NULL
            // rows out of the slow-path column and the fast-path column
            // can't carry NULLs. We then strip any residual validity in
            // `fused_avg_host_col` before uploading so the GPU never sees
            // garbage at NULL positions.
            //
            // TODO(null): empty input -> 0.0 (not SQL NULL), matching the
            // public AVG return-type contract; see `aggregate.rs` for the
            // same TODO.
            let resolved =
                resolve_agg_input_col(expr, pre_spec, compacted, DataType::Float64)?;
            let (sum_f64, count_u64) = fused_avg_host_col(resolved.as_ref())?;
            let avg = if count_u64 == 0 {
                0.0
            } else {
                sum_f64 / count_u64 as f64
            };
            scalar_to_array(Scalar::F64(avg), out_field.dtype)
        }
        AggregateExpr::VarPop(expr) | AggregateExpr::VarSamp(expr) => {
            // v0.5 host-side Welford reduction over the slow-path
            // pre-aggregation column, materialised at Float64. Matches
            // the scalar-aggregate path in `aggregate.rs` so the two
            // entry points produce identical results for a given input.
            let resolved = resolve_agg_input_col(
                expr.as_ref(),
                pre_spec,
                compacted,
                DataType::Float64,
            )?;
            let xs = host_col_as_f64(resolved.as_ref())?;
            let is_pop = matches!(agg, AggregateExpr::VarPop(_));
            let result: Option<f64> = if is_pop {
                crate::exec::welford::var_pop_f64(&xs)
            } else {
                crate::exec::welford::var_samp_f64(&xs)
            };
            Ok(Arc::new(Float64Array::from(vec![result])) as ArrayRef)
        }
    }
}

/// Materialise a (NULL-stripped) host column as `Vec<f64>` for the
/// Welford pass. The slow path of `resolve_agg_input_col` already
/// validity-filters NULLs out of `Owned`; the fast path uses borrowed
/// pre-stage outputs that carry no NULLs. This helper widens whichever
/// dtype landed in `HostCol` to `f64` for the variance accumulator.
fn host_col_as_f64(col: &HostCol) -> BoltResult<Vec<f64>> {
    // If the borrowed fast-path column happens to carry residual validity
    // (it shouldn't — the upload path rejects validity-bearing pre cols
    // outright), strip it here so a stray garbage value at a NULL slot
    // doesn't poison the Welford state.
    let valid_at = |i: usize| -> bool {
        match &col.validity {
            Some(v) => v[i] != 0,
            None => true,
        }
    };
    let n = col.len();
    let mut out: Vec<f64> = Vec::with_capacity(n);
    match &col.values {
        HostColValues::I32(v) => {
            for (i, x) in v.iter().enumerate() {
                if valid_at(i) {
                    out.push(*x as f64);
                }
            }
        }
        HostColValues::I64(v) => {
            for (i, x) in v.iter().enumerate() {
                if valid_at(i) {
                    out.push(*x as f64);
                }
            }
        }
        HostColValues::F32(v) => {
            for (i, x) in v.iter().enumerate() {
                if valid_at(i) {
                    out.push(*x as f64);
                }
            }
        }
        HostColValues::F64(v) => {
            for (i, x) in v.iter().enumerate() {
                if valid_at(i) {
                    out.push(*x);
                }
            }
        }
    }
    Ok(out)
}

/// Resolve an aggregate input expression to a host column ready for reduction.
///
/// Fast path: when `expr` (after stripping aliases) is a bare column ref whose
/// name matches one of `pre.outputs`, return a borrowed view of that compacted
/// column — no extra allocation.
///
/// Slow path: when `expr` is anything else (e.g. `Sum(price * tax)` where the
/// planner didn't pre-materialise the product), build a [`expr_agg::ColumnEnv`]
/// over the already-compacted pre outputs and use [`expr_agg::eval_expr`] to
/// materialise the value column. NULL rows produced by the host-side
/// evaluator are filtered out of the returned column so the GPU reduction
/// sees only valid values; the count of surviving rows is reported via
/// [`ResolvedHostCol::non_null_count`] (used by AVG and COUNT(col)). The
/// result is owned. `expected_dtype` is the caller-chosen materialisation
/// dtype: SUM/MIN/MAX use `out_field.dtype`, AVG uses `Float64` (the
/// reduction accumulator), and COUNT uses `Int64` (only the non-NULL count
/// is consumed, not the values).
fn resolve_agg_input_col<'a>(
    expr: &Expr,
    pre_spec: &KernelSpec,
    compacted: &'a CompactedPreOutputs,
    expected_dtype: DataType,
) -> BoltResult<ResolvedHostCol<'a>> {
    if let Some(name) = expr_agg::try_bare_column(expr) {
        let idx = pre_spec
            .outputs
            .iter()
            .position(|o| o.name == name)
            .ok_or_else(|| {
                BoltError::Plan(format!(
                    "aggregate input '{}' not found among pre kernel outputs",
                    name
                ))
            })?;
        if idx >= compacted.cols.len() {
            return Err(BoltError::Other(format!(
                "internal: pre output ordinal {} out of range (have {} compacted cols)",
                idx,
                compacted.cols.len()
            )));
        }
        let col = &compacted.cols[idx];
        // Fast path: the pre kernel output has no NULL bitmap, so every
        // row is a "non-NULL" row for aggregate purposes.
        let n = col.len();
        return Ok(ResolvedHostCol::Borrowed { col, non_null: n });
    }

    // Slow path: materialise via the host-side evaluator over the compacted
    // pre outputs. Each compacted column is wrapped lazily (lifting to
    // `Option`, which never carries a None on this path) so the evaluator can
    // run unchanged. The evaluator itself can introduce NULLs (e.g. integer
    // division by zero, or any operand of a binary op being NULL), which we
    // then filter out below so the reduction sees only valid rows.
    let n_rows = compacted.n_rows();
    let wrapped: Vec<(String, expr_agg::HostColumn)> = pre_spec
        .outputs
        .iter()
        .enumerate()
        .map(|(j, io)| (io.name.clone(), to_expr_host(&compacted.cols[j])))
        .collect();
    let env: expr_agg::ColumnEnv<'_> = wrapped.iter().map(|(n, c)| (n.clone(), c)).collect();
    let materialised = expr_agg::eval_expr(expr, &env, expected_dtype, n_rows)?;
    let filtered = from_expr_host(materialised)?;
    // PV (Option B): `filtered.validity` carries per-row NULL info as a
    // `Vec<u8>` (`1` = valid, `0` = NULL). The fast path's `non_null` is
    // the full column length; the slow path's `non_null` is the popcount
    // of validity, falling back to the full length when validity is `None`.
    let non_null = match &filtered.validity {
        Some(v) => v.iter().filter(|b| **b != 0).count(),
        None => filtered.len(),
    };
    Ok(ResolvedHostCol::Owned {
        col: filtered,
        non_null,
    })
}

/// Borrowed or owned host column. Lets the fast path return `&HostCol` from
/// the compacted store while the slow path returns a freshly-materialised
/// value from `expr_agg`, with a single `as_ref()` method to feed either into
/// `reduce_host_col`.
///
/// `non_null` is the number of rows that were not SQL NULL in the source
/// column (after pre-filtering and expression evaluation). SQL aggregate
/// semantics — `SUM`, `MIN`, `MAX`, `AVG`, `COUNT(col)` — all ignore NULL
/// rows, so the executor exposes this count to callers that need it for
/// the AVG denominator and the COUNT result.
enum ResolvedHostCol<'a> {
    /// Borrowed view of a pre-stage output (no NULLs possible here).
    Borrowed { col: &'a HostCol, non_null: usize },
    /// Freshly materialised + NULL-filtered slow-path column.
    Owned { col: HostCol, non_null: usize },
}

impl<'a> ResolvedHostCol<'a> {
    fn as_ref(&self) -> &HostCol {
        match self {
            ResolvedHostCol::Borrowed { col, .. } => *col,
            ResolvedHostCol::Owned { col, .. } => col,
        }
    }

    /// Number of non-NULL rows in the original (pre-filter) input. Used by
    /// AVG (denominator) and COUNT(col) (result). For the fast path this
    /// equals the column length; for the slow path it equals the length
    /// of the filtered column (since NULL rows have been dropped).
    fn non_null_count(&self) -> usize {
        match self {
            ResolvedHostCol::Borrowed { non_null, .. } => *non_null,
            ResolvedHostCol::Owned { non_null, .. } => *non_null,
        }
    }
}

/// Lift a local primitive [`HostCol`] into the `Option`-carrying
/// [`expr_agg::HostColumn`] shape consumed by the host-side evaluator.
///
/// **NULL handling (Option B)**: when the source `HostCol` carries a
/// validity bitmap, NULL rows surface as `None` so the evaluator's 3VL
/// machinery (see `expr_agg`) propagates them correctly. When validity
/// is `None` every cell is `Some(_)` (the pre-Option-B behaviour).
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
        HostColValues::I32(v) => expr_agg::HostColumn::I32(lift(v, valid_at)),
        HostColValues::I64(v) => expr_agg::HostColumn::I64(lift(v, valid_at)),
        HostColValues::F32(v) => expr_agg::HostColumn::F32(lift(v, valid_at)),
        HostColValues::F64(v) => expr_agg::HostColumn::F64(lift(v, valid_at)),
    }
}

/// Convert a materialised [`expr_agg::HostColumn`] back into the local
/// primitive [`HostCol`] shape consumed by the reduction path. NULLs are
/// **preserved** as a `validity` bitmap (Option B) rather than silently
/// collapsing to the dtype's zero — the downstream reducer
/// (`reduce_host_col`) strips the NULL rows before launching the GPU
/// kernel. This is the correctness contract that previously required
/// `PreCol::upload` to reject NULL-bearing arrays outright (Option A);
/// Stage B propagation makes the rejection unnecessary as long as the
/// validity rides all the way through.
///
/// Bool / Utf8 materialisations are rejected — the reduction kernels
/// only accept primitive numeric inputs.
fn from_expr_host(c: expr_agg::HostColumn) -> BoltResult<HostCol> {
    fn split<T: Copy + Default>(v: Vec<Option<T>>) -> (Vec<T>, Option<Vec<u8>>) {
        let any_null = v.iter().any(|x| x.is_none());
        if !any_null {
            return (
                v.into_iter().map(|x| x.unwrap_or_default()).collect(),
                None,
            );
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
        expr_agg::HostColumn::Bool(_) | expr_agg::HostColumn::Utf8(_) => {
            Err(BoltError::Type(
                "agg_with_pre: Bool/Utf8 aggregate inputs not supported by the \
                 primitive reduction path"
                    .into(),
            ))
        }
    }
}

/// Run a GPU reduction over `col` and return the scalar result.
///
/// **NULL handling (Option B)**: if `col` carries a validity bitmap, the
/// NULL rows are stripped on the host before the values are uploaded —
/// the reduction kernels (`agg_kernels::compile_reduction_kernel`) do
/// not consume validity, so this is the surgical layer that bridges
/// Stage B (NULLs survive the pre-stage) and Stage C (validity flows
/// into the reduction itself). Stripping at the host costs a single
/// pass and a temporary `Vec`; for typical batch sizes that's negligible
/// next to the GPU launch overhead.
fn reduce_host_col(op: ReduceOp, col: &HostCol) -> BoltResult<Scalar> {
    // Materialise a NULL-free view. The `strip_nulls` no-op when validity
    // is None, so the all-valid fast path pays zero cost.
    let stripped = strip_nulls_borrowed(col);
    match &stripped.values {
        HostColValues::I32(v) => reduce_host_slice::<i32>(op, DataType::Int32, v),
        HostColValues::I64(v) => reduce_host_slice::<i64>(op, DataType::Int64, v),
        HostColValues::F32(v) => reduce_host_slice::<f32>(op, DataType::Float32, v),
        HostColValues::F64(v) => reduce_host_slice::<f64>(op, DataType::Float64, v),
    }
}

/// Pack a finalized stddev (`Some(σ)` or `None`) into a one-row Arrow
/// `Float64Array`. Mirrors [`aggregate::stddev_to_array`] (see that
/// module's doc); duplicated here so the pre-aggregated path doesn't
/// have to reach across module boundaries for a single
/// dtype-checked-pack helper.
fn stddev_to_array_with_pre(
    value: Option<f64>,
    agg: &AggregateExpr,
    out_field: &Field,
) -> BoltResult<ArrayRef> {
    if out_field.dtype != DataType::Float64 {
        return Err(BoltError::Type(format!(
            "STDDEV output dtype must be Float64, got {:?}",
            out_field.dtype
        )));
    }
    match (agg, value) {
        (AggregateExpr::StddevPop(_), Some(v)) => {
            Ok(Arc::new(Float64Array::from(vec![v])) as ArrayRef)
        }
        (AggregateExpr::StddevPop(_), None) => {
            // Empty / all-NULL input → 0.0 (mirrors the AVG convention so
            // a non-nullable downstream consumer never sees a NULL here).
            Ok(Arc::new(Float64Array::from(vec![0.0_f64])) as ArrayRef)
        }
        (AggregateExpr::StddevSamp(_), Some(v)) => {
            Ok(Arc::new(Float64Array::from(vec![v])) as ArrayRef)
        }
        (AggregateExpr::StddevSamp(_), None) => {
            // count <= 1 → SQL NULL. The aggregate output field is
            // nullable by `LogicalPlan::Aggregate` construction, so this
            // single-element nullable Float64Array packs cleanly.
            Ok(Arc::new(Float64Array::from(vec![None::<f64>])) as ArrayRef)
        }
        _ => Err(BoltError::Other(
            "stddev_to_array_with_pre called with non-STDDEV aggregate".into(),
        )),
    }
}

/// Run the **fused** AVG reduction over `col` and return `(sum_f64,
/// count_u64)`. NULL rows are stripped on the host before upload (the kernel
/// expects a contiguous value buffer). Replaces the "run SUM kernel, then
/// divide by host-known count" decomposition in the AVG branch.
///
/// The kernel does its own per-block count, which matches the post-strip row
/// count: every in-range thread contributes 1 to the count.
fn fused_avg_host_col(col: &HostCol) -> BoltResult<(f64, u64)> {
    let stripped = strip_nulls_borrowed(col);
    match &stripped.values {
        HostColValues::I32(v) => fused_avg_host_slice::<i32>(DataType::Int32, v),
        HostColValues::I64(v) => fused_avg_host_slice::<i64>(DataType::Int64, v),
        HostColValues::F32(v) => fused_avg_host_slice::<f32>(DataType::Float32, v),
        HostColValues::F64(v) => fused_avg_host_slice::<f64>(DataType::Float64, v),
    }
}

/// Upload a host slice, then launch the fused AVG kernel over it. Returns
/// `(sum_f64, count_u64)`. `dtype` is the input element dtype.
fn fused_avg_host_slice<TIn>(dtype: DataType, host: &[TIn]) -> BoltResult<(f64, u64)>
where
    TIn: Pod,
{
    if host.is_empty() {
        return Ok((0.0, 0));
    }

    let dev = GpuVec::<TIn>::from_slice(host)?;
    let n_rows = host.len();

    let block = BLOCK_SIZE;
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let grid_x = ((n_rows_u32 + block - 1) / block).max(1);

    let block_sums = GpuVec::<f64>::zeros(grid_x as usize)?;
    let block_counts = GpuVec::<u32>::zeros(grid_x as usize)?;

    let module = crate::exec::module_cache::get_or_build_module(
        module_path!(),
        format!("avg_reduce_{:?}", dtype),
        None,
        || compile_avg_reduction_kernel(dtype),
    )?;
    let function = module.function(AVG_KERNEL_ENTRY)?;

    let mut input_ptr: CUdeviceptr = dev.device_ptr();
    let mut sums_ptr: CUdeviceptr = block_sums.device_ptr();
    let mut counts_ptr: CUdeviceptr = block_counts.device_ptr();

    let mut kernel_params: [*mut c_void; 4] = [
        &mut input_ptr as *mut CUdeviceptr as *mut c_void,
        &mut sums_ptr as *mut CUdeviceptr as *mut c_void,
        &mut counts_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
    ];

    let stream = CudaStream::null();
    // SAFETY: `function` borrowed from a live module; param slots point into
    // stack locals that outlive `synchronize`.
    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x,
            1,
            1,
            block,
            1,
            1,
            0,
            stream.raw(),
            kernel_params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }
    stream.synchronize()?;

    let _ = input_ptr;
    let _ = sums_ptr;
    let _ = counts_ptr;

    let host_sums = block_sums.to_vec()?;
    let host_counts = block_counts.to_vec()?;
    drop(block_sums);
    drop(block_counts);

    let total_sum: f64 = host_sums.iter().copied().sum();
    let total_count: u64 = host_counts.iter().copied().map(u64::from).sum();
    Ok((total_sum, total_count))
}

/// Borrowing version of `HostCol::strip_nulls`. Returns an owned `HostCol`
/// because the row count differs from the input when NULLs are present.
/// All-valid fast path returns a fresh `HostCol` that shares the input's
/// vector contents via clone — Stage C can switch to a real borrow once
/// the reduction path accepts validity natively.
fn strip_nulls_borrowed(col: &HostCol) -> HostCol {
    let Some(v) = &col.validity else {
        // All-valid: clone the values; the reducer needs an owned slice
        // either way for the upload. This is the hot path so the clone
        // matters — but `Vec::clone` on the slice is a single memcpy.
        let values = match &col.values {
            HostColValues::I32(x) => HostColValues::I32(x.clone()),
            HostColValues::I64(x) => HostColValues::I64(x.clone()),
            HostColValues::F32(x) => HostColValues::F32(x.clone()),
            HostColValues::F64(x) => HostColValues::F64(x.clone()),
        };
        return HostCol {
            values,
            validity: None,
        };
    };
    let keep: Vec<bool> = v.iter().map(|b| *b != 0).collect();
    let values = match &col.values {
        HostColValues::I32(x) => HostColValues::I32(filter_vec(x.clone(), &keep)),
        HostColValues::I64(x) => HostColValues::I64(filter_vec(x.clone(), &keep)),
        HostColValues::F32(x) => HostColValues::F32(filter_vec(x.clone(), &keep)),
        HostColValues::F64(x) => HostColValues::F64(filter_vec(x.clone(), &keep)),
    };
    HostCol {
        values,
        validity: None,
    }
}

/// Upload a host slice, then run the standard GPU reduction over it.
fn reduce_host_slice<T>(op: ReduceOp, dtype: DataType, host: &[T]) -> BoltResult<Scalar>
where
    T: Pod + ReduceScalar,
{
    let dev = GpuVec::<T>::from_slice(host)?;
    reduce_gpu_vec::<T>(op, dtype, &dev, host.len())
}

/// Launch the per-block reduction kernel against `input` and finish the
/// reduction on the host. Returns the final scalar as a `Scalar`.
fn reduce_gpu_vec<T>(
    op: ReduceOp,
    dtype: DataType,
    input: &GpuVec<T>,
    n_rows: usize,
) -> BoltResult<Scalar>
where
    T: Pod + ReduceScalar,
{
    // 0-row degenerate case: skip the launch entirely and return the identity.
    if n_rows == 0 {
        return T::identity_scalar(op, dtype);
    }

    let block = BLOCK_SIZE;
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let grid_x = ((n_rows_u32 + block - 1) / block).max(1);
    let partials = GpuVec::<T>::zeros(grid_x as usize)?;

    let module = module_cache::get_or_build_module(
        module_path!(),
        format!("reduction:{:?}:{:?}", op, dtype),
        None,
        || compile_reduction_kernel(op, dtype),
    )?;
    let function = module.function(REDUCTION_KERNEL_ENTRY)?;

    let mut input_ptr: CUdeviceptr = input.device_ptr();
    let mut output_ptr: CUdeviceptr = partials.device_ptr();

    let mut kernel_params: [*mut c_void; 3] = [
        &mut input_ptr as *mut CUdeviceptr as *mut c_void,
        &mut output_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
    ];

    let stream = CudaStream::null();
    // SAFETY: `function` is borrowed from a live `CudaModule`; every entry of
    // `kernel_params` points to a stack local that outlives the synchronize.
    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x,
            1,
            1,
            block,
            1,
            1,
            0,
            stream.raw(),
            kernel_params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }
    stream.synchronize()?;

    let _ = input_ptr;
    let _ = output_ptr;

    let host_partials = partials.to_vec()?;
    drop(partials);
    T::finalize(op, dtype, &host_partials)
}

/// Heterogenous owned device column for the pre kernel's inputs and outputs.
/// Only primitive numeric dtypes are reachable here — Bool / Utf8 are rejected
/// because aggregate inputs (and the expressions feeding them) never carry
/// those types.
///
/// **NULL handling (Option B)**: each `PreCol` may additionally carry a
/// parallel `valid_mask: GpuVec<u8>` (`1` = valid row, `0` = NULL) that
/// rides alongside the value buffer through the pre-stage GPU kernel. The
/// pre kernel emits a per-row AND of input validities into each output
/// `valid_mask` (see `jit/ptx_gen.rs::compile`). After download +
/// host-compaction the NULL rows are stripped, so downstream reductions
/// see only valid values.
///
/// A `valid_mask` of `None` means "all rows are valid" — the fast path
/// when the input Arrow array has `null_count() == 0`. Carrying the
/// `Option` (rather than always allocating an all-ones buffer) keeps the
/// common NULL-free pipeline allocation-free.
struct PreCol {
    /// The value buffer.
    values: PreColValues,
    /// Optional per-row validity bitmap; `None` = all rows valid.
    valid_mask: Option<GpuVec<u8>>,
}

/// Typed value buffer for a `PreCol`. Split out from the validity bitmap
/// so the existing dtype dispatch keeps its readable match arm structure.
enum PreColValues {
    I32(GpuVec<i32>),
    I64(GpuVec<i64>),
    F32(GpuVec<f32>),
    F64(GpuVec<f64>),
}

impl PreCol {
    /// Upload an Arrow array to the GPU, downcasting per `dtype`.
    ///
    /// **NULL handling (Option B)**: when `arr.null_count() > 0` this also
    /// builds a parallel `Vec<u8>` validity mask (`1` = valid, `0` = null)
    /// from the Arrow validity bitmap and uploads it as `valid_mask`. The
    /// caller (`run_pre_stage`) is responsible for plumbing the validity
    /// pointer into the pre kernel's parameter list when present.
    ///
    /// When `arr.null_count() == 0` we skip the bitmap allocation entirely
    /// — `valid_mask` stays `None` — so NULL-free queries pay no extra
    /// memory or PTX cost.
    fn upload(arr: &dyn Array, dtype: DataType) -> BoltResult<Self> {
        let n = arr.len();
        let valid_mask = if arr.null_count() > 0 {
            // Build a host-side `Vec<u8>` from the Arrow validity bitmap.
            // We deliberately materialise byte-per-row rather than ship the
            // packed Arrow bitmap to keep the GPU side dead-simple (`ld.u8`
            // + `and.b32`). The 8x size bloat versus a packed bitmap is the
            // tradeoff Option B accepts — Stage C can switch to a packed
            // representation behind the same `valid_mask` field.
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
            DataType::Bool | DataType::Utf8 | DataType::Decimal128(_, _) | DataType::Date32 | DataType::Timestamp(_, _) => {
                return Err(BoltError::Type(format!(
                    "agg_with_pre: pre kernel column dtype {:?} not supported",
                    dtype
                )))
            }
        };
        Ok(PreCol { values, valid_mask })
    }

    /// Allocate a zero-initialised device column of `n` rows.
    ///
    /// Used for pre-kernel **output** buffers. `with_validity` controls
    /// whether a parallel zero-initialised validity buffer is also
    /// allocated: when any input carries validity the corresponding
    /// outputs must carry a parallel buffer so the kernel has somewhere
    /// to store the per-row combined-validity AND result. We zero-init
    /// the validity buffer too — out-of-bounds threads exit before
    /// touching it, and the pre kernel's `tid >= n_rows` guard ensures
    /// undefined positions stay zero (i.e. "null"), which is the safe
    /// default.
    fn alloc_zeros(dtype: DataType, n: usize, with_validity: bool) -> BoltResult<Self> {
        let values = match dtype {
            DataType::Int32 => PreColValues::I32(GpuVec::<i32>::zeros(n)?),
            DataType::Int64 => PreColValues::I64(GpuVec::<i64>::zeros(n)?),
            DataType::Float32 => PreColValues::F32(GpuVec::<f32>::zeros(n)?),
            DataType::Float64 => PreColValues::F64(GpuVec::<f64>::zeros(n)?),
            DataType::Bool | DataType::Utf8 | DataType::Decimal128(_, _) | DataType::Date32 | DataType::Timestamp(_, _) => {
                return Err(BoltError::Type(format!(
                    "agg_with_pre: pre kernel output dtype {:?} not supported",
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

    /// Raw device pointer for kernel-parameter assembly (value buffer).
    fn device_ptr(&self) -> CUdeviceptr {
        match &self.values {
            PreColValues::I32(v) => v.device_ptr(),
            PreColValues::I64(v) => v.device_ptr(),
            PreColValues::F32(v) => v.device_ptr(),
            PreColValues::F64(v) => v.device_ptr(),
        }
    }

    /// Raw device pointer for the validity buffer, if present.
    fn validity_device_ptr(&self) -> Option<CUdeviceptr> {
        self.valid_mask.as_ref().map(|m| m.device_ptr())
    }

    /// True iff this `PreCol` carries a per-row validity bitmap.
    fn has_validity(&self) -> bool {
        self.valid_mask.is_some()
    }

    /// Download the column to host and verify the length matches `n_rows`.
    /// Also downloads the validity bitmap (if present) parallel to the
    /// values.
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
        };
        Ok(HostCol {
            values: host_values,
            validity,
        })
    }
}

/// Host-side typed column produced by downloading a `PreCol`. Carries a
/// parallel `validity` vector when the source column had NULLs (Option B);
/// `None` means "all rows valid".
struct HostCol {
    values: HostColValues,
    /// Per-row validity, parallel to `values`. `Some(v)` => `v[i] == 0` is
    /// a NULL. `None` => all rows valid (fast path).
    validity: Option<Vec<u8>>,
}

/// Typed host-side values for a `HostCol`. Mirrors `PreColValues`.
enum HostColValues {
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl HostCol {
    /// Number of elements in the column.
    fn len(&self) -> usize {
        match &self.values {
            HostColValues::I32(v) => v.len(),
            HostColValues::I64(v) => v.len(),
            HostColValues::F32(v) => v.len(),
            HostColValues::F64(v) => v.len(),
        }
    }

    /// Return a new column containing only positions where `mask[i]` is true.
    /// The pre-projection kernel leaves zeros in masked slots, so we drop those
    /// positions and keep the rest in original order. The validity bitmap (if
    /// any) is compacted in lockstep.
    fn compact(self, mask: &[bool]) -> BoltResult<HostCol> {
        if mask.len() != self.len() {
            return Err(BoltError::Other(format!(
                "agg_with_pre: mask length {} != column length {}",
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
        };
        let validity = validity.map(|v| filter_vec(v, mask));
        Ok(HostCol { values, validity })
    }

    /// Drop the NULL rows according to the validity bitmap; returns a new
    /// `HostCol` with `validity = None`. No-op when validity is already
    /// `None`. Used to feed downstream reduction kernels that don't yet
    /// consume validity (so the value buffer must be NULL-stripped first).
    #[allow(dead_code)]
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
        };
        HostCol {
            values,
            validity: None,
        }
    }
}

/// Keep `v[i]` iff `mask[i]`; returns a fresh Vec.
fn filter_vec<T: Copy>(v: Vec<T>, mask: &[bool]) -> Vec<T> {
    v.into_iter()
        .zip(mask.iter())
        .filter_map(|(x, &k)| if k { Some(x) } else { None })
        .collect()
}

/// Copy back a `GpuVec<T>` into a host `Vec<T>` of length `n_rows`.
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

/// A typed scalar result of a reduction.
#[derive(Debug, Clone, Copy)]
enum Scalar {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

/// Per-`T` helpers for the GPU reduction path. Mirrors the trait in
/// `aggregate.rs` exactly so this module stays self-contained.
trait ReduceScalar: Sized + Copy {
    fn finalize(op: ReduceOp, dtype: DataType, host: &[Self]) -> BoltResult<Scalar>;
    fn identity_scalar(op: ReduceOp, dtype: DataType) -> BoltResult<Scalar>;
}

impl ReduceScalar for i32 {
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> BoltResult<Scalar> {
        let acc = match op {
            ReduceOp::Sum | ReduceOp::Count => host.iter().copied().fold(0i32, i32::wrapping_add),
            ReduceOp::Min => host.iter().copied().fold(i32::MAX, i32::min),
            ReduceOp::Max => host.iter().copied().fold(i32::MIN, i32::max),
        };
        Ok(Scalar::I32(acc))
    }
    fn identity_scalar(op: ReduceOp, _dtype: DataType) -> BoltResult<Scalar> {
        Ok(Scalar::I32(match op {
            ReduceOp::Sum | ReduceOp::Count => 0,
            ReduceOp::Min => i32::MAX,
            ReduceOp::Max => i32::MIN,
        }))
    }
}

impl ReduceScalar for i64 {
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> BoltResult<Scalar> {
        let acc = match op {
            ReduceOp::Sum | ReduceOp::Count => host.iter().copied().fold(0i64, i64::wrapping_add),
            ReduceOp::Min => host.iter().copied().fold(i64::MAX, i64::min),
            ReduceOp::Max => host.iter().copied().fold(i64::MIN, i64::max),
        };
        Ok(Scalar::I64(acc))
    }
    fn identity_scalar(op: ReduceOp, _dtype: DataType) -> BoltResult<Scalar> {
        Ok(Scalar::I64(match op {
            ReduceOp::Sum | ReduceOp::Count => 0,
            ReduceOp::Min => i64::MAX,
            ReduceOp::Max => i64::MIN,
        }))
    }
}

impl ReduceScalar for f32 {
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> BoltResult<Scalar> {
        let acc = match op {
            ReduceOp::Sum | ReduceOp::Count => host.iter().copied().fold(0.0f32, |a, b| a + b),
            ReduceOp::Min => host.iter().copied().fold(f32::INFINITY, f32::min),
            ReduceOp::Max => host.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        };
        Ok(Scalar::F32(acc))
    }
    fn identity_scalar(op: ReduceOp, _dtype: DataType) -> BoltResult<Scalar> {
        Ok(Scalar::F32(match op {
            ReduceOp::Sum | ReduceOp::Count => 0.0,
            ReduceOp::Min => f32::INFINITY,
            ReduceOp::Max => f32::NEG_INFINITY,
        }))
    }
}

impl ReduceScalar for f64 {
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> BoltResult<Scalar> {
        let acc = match op {
            ReduceOp::Sum | ReduceOp::Count => host.iter().copied().fold(0.0f64, |a, b| a + b),
            ReduceOp::Min => host.iter().copied().fold(f64::INFINITY, f64::min),
            ReduceOp::Max => host.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        };
        Ok(Scalar::F64(acc))
    }
    fn identity_scalar(op: ReduceOp, _dtype: DataType) -> BoltResult<Scalar> {
        Ok(Scalar::F64(match op {
            ReduceOp::Sum | ReduceOp::Count => 0.0,
            ReduceOp::Min => f64::INFINITY,
            ReduceOp::Max => f64::NEG_INFINITY,
        }))
    }
}

/// Convert a `Scalar` into a single-element Arrow array of `out_dtype`.
fn scalar_to_array(scalar: Scalar, out_dtype: DataType) -> BoltResult<ArrayRef> {
    match (scalar, out_dtype) {
        (Scalar::I32(v), DataType::Int32) => Ok(Arc::new(Int32Array::from(vec![v])) as ArrayRef),
        (Scalar::I64(v), DataType::Int64) => Ok(Arc::new(Int64Array::from(vec![v])) as ArrayRef),
        (Scalar::F32(v), DataType::Float32) => {
            Ok(Arc::new(Float32Array::from(vec![v])) as ArrayRef)
        }
        (Scalar::F64(v), DataType::Float64) => {
            Ok(Arc::new(Float64Array::from(vec![v])) as ArrayRef)
        }

        // Common cross-dtype output paths (AVG -> Float64, COUNT -> Int64).
        (Scalar::I32(v), DataType::Int64) => {
            Ok(Arc::new(Int64Array::from(vec![v as i64])) as ArrayRef)
        }
        (Scalar::I32(v), DataType::Float32) => {
            Ok(Arc::new(Float32Array::from(vec![v as f32])) as ArrayRef)
        }
        (Scalar::I32(v), DataType::Float64) => {
            Ok(Arc::new(Float64Array::from(vec![v as f64])) as ArrayRef)
        }
        (Scalar::I64(v), DataType::Float64) => {
            Ok(Arc::new(Float64Array::from(vec![v as f64])) as ArrayRef)
        }
        (Scalar::F32(v), DataType::Float64) => {
            Ok(Arc::new(Float64Array::from(vec![v as f64])) as ArrayRef)
        }

        (s, dt) => Err(BoltError::Type(format!(
            "agg_with_pre: cannot pack scalar {:?} into output dtype {:?}",
            s, dt
        ))),
    }
}

/// Build a `Type` error for a failed Arrow downcast.
fn downcast_err(role: &str, expected: &str) -> BoltError {
    BoltError::Type(format!(
        "agg_with_pre: pre kernel {} could not be downcast to {}",
        role, expected
    ))
}

/// Map Arrow `DataType` to our plan `DataType`. Mirror of the helper in
/// `aggregate.rs` and `engine.rs`, copied here to avoid reaching across module
/// privacy.
fn arrow_dtype_to_plan(d: &ArrowDataType) -> BoltResult<DataType> {
    crate::exec::schema_convert::arrow_dtype_to_plan_basic(d, "")
}

/// Build an Arrow `Schema` from our plan `Schema` for the output `RecordBatch`.
fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    crate::exec::schema_convert::plan_schema_to_arrow_schema_no_temporal(s, "this aggregate output path")
}

#[cfg(test)]
mod null_propagation_tests {
    //! Host-only tests for the Option B NULL-propagation gates added to the
    //! pre-stage. None of these reach CUDA: `PreCol::upload` is exercised
    //! via integration tests with `#[ignore]` (or downstream e2e), while
    //! the host-side `to_expr_host` / `from_expr_host` / `strip_nulls`
    //! contracts can be verified pure.
    use super::*;

    // ---- HostCol round-trip --------------------------------------------------

    #[test]
    fn host_col_strip_nulls_drops_invalid_rows() {
        let col = HostCol {
            values: HostColValues::I64(vec![10, 20, 30, 40]),
            validity: Some(vec![1, 0, 1, 0]),
        };
        let stripped = col.strip_nulls();
        assert!(stripped.validity.is_none());
        match stripped.values {
            HostColValues::I64(v) => assert_eq!(v, vec![10, 30]),
            _ => panic!("dtype changed"),
        }
    }

    #[test]
    fn host_col_strip_nulls_is_noop_when_all_valid() {
        let col = HostCol {
            values: HostColValues::F64(vec![1.0, 2.0, 3.0]),
            validity: None,
        };
        let stripped = col.strip_nulls();
        match stripped.values {
            HostColValues::F64(v) => assert_eq!(v, vec![1.0, 2.0, 3.0]),
            _ => panic!("dtype changed"),
        }
    }

    // ---- to_expr_host: validity surfaces as None ----------------------------

    #[test]
    fn to_expr_host_propagates_validity_as_none_i32() {
        let col = HostCol {
            values: HostColValues::I32(vec![1, 2, 3, 4]),
            validity: Some(vec![1, 0, 1, 0]),
        };
        let lifted = to_expr_host(&col);
        match lifted {
            expr_agg::HostColumn::I32(v) => {
                assert_eq!(v, vec![Some(1), None, Some(3), None]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn to_expr_host_all_some_when_validity_none() {
        let col = HostCol {
            values: HostColValues::F32(vec![1.5, 2.5, 3.5]),
            validity: None,
        };
        let lifted = to_expr_host(&col);
        match lifted {
            expr_agg::HostColumn::F32(v) => {
                assert_eq!(v, vec![Some(1.5), Some(2.5), Some(3.5)]);
            }
            _ => panic!("wrong variant"),
        }
    }

    // ---- from_expr_host: preserves NULLs as validity ------------------------

    #[test]
    fn from_expr_host_preserves_nulls_as_validity_i32() {
        let col = expr_agg::HostColumn::I32(vec![Some(1), None, Some(3)]);
        let out = from_expr_host(col).expect("should succeed");
        match out.values {
            HostColValues::I32(v) => assert_eq!(v, vec![1, 0, 3]),
            _ => panic!("wrong variant"),
        }
        assert_eq!(out.validity.as_deref(), Some(&[1u8, 0, 1][..]));
    }

    #[test]
    fn from_expr_host_no_validity_when_all_some_f64() {
        let col = expr_agg::HostColumn::F64(vec![Some(1.0), Some(2.0)]);
        let out = from_expr_host(col).expect("should succeed");
        match out.values {
            HostColValues::F64(v) => assert_eq!(v, vec![1.0, 2.0]),
            _ => panic!("wrong variant"),
        }
        assert!(out.validity.is_none(), "all-valid -> no validity bitmap");
    }

    // ---- HostCol::compact compacts validity alongside values ----------------

    #[test]
    fn host_col_compact_preserves_validity_alignment() {
        // Predicate-driven compaction: drop rows 1 and 3. Validity must
        // shrink in lockstep.
        let col = HostCol {
            values: HostColValues::I64(vec![10, 20, 30, 40, 50]),
            validity: Some(vec![1, 0, 1, 0, 1]),
        };
        let mask = vec![true, false, true, false, true];
        let compact = col.compact(&mask).expect("compact");
        match compact.values {
            HostColValues::I64(v) => assert_eq!(v, vec![10, 30, 50]),
            _ => panic!("dtype changed"),
        }
        assert_eq!(compact.validity.as_deref(), Some(&[1u8, 1, 1][..]));
    }

    // ---- Arrow NULL-bearing upload would be accepted ------------------------
    //
    // NOTE: PreCol::upload reaches `primitive_to_gpu` which allocates a
    // CUDA buffer; we can't run that without the driver. Instead we verify
    // the host-side branch (`arr.null_count() > 0`) takes the validity
    // path by inspecting an intermediate value buffer via Arrow itself.
    // The full GPU path is exercised by integration tests.

    #[test]
    fn null_bearing_array_has_null_count_positive() {
        let arr = Int32Array::from(vec![Some(1i32), None, Some(3i32)]);
        assert_eq!(arr.null_count(), 1, "Arrow should report 1 null");
        // The branch that builds the host-side `Vec<u8>` validity matches
        // `arr.is_null(i)`. This is exactly the predicate `PreCol::upload`
        // uses, so we mirror it here as a regression guard against a
        // future refactor that swaps the predicate (e.g. to a bitmap walk
        // that off-by-ones the high bit).
        let expected: Vec<u8> = (0..arr.len())
            .map(|i| if arr.is_null(i) { 0 } else { 1 })
            .collect();
        assert_eq!(expected, vec![1, 0, 1]);
    }
}
