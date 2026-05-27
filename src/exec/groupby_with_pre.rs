// SPDX-License-Identifier: Apache-2.0

//! Fused pre-projection + GROUP BY aggregate execution.
//!
//! This module is the union of [`agg_with_pre`](crate::exec::agg_with_pre) and
//! [`groupby`](crate::exec::groupby). The former runs a pre-projection (and
//! optional predicate) kernel ahead of a *scalar* reduction; the latter runs a
//! GPU hash-grouped aggregation but only against bare column references on the
//! source `RecordBatch`. Neither handles a query like:
//!
//! ```sql
//! SELECT region, SUM(price * tax)
//! FROM   sales
//! WHERE  active = 1
//! GROUP  BY region
//! ```
//!
//! which requires *both* a pre kernel (to compute `price * tax` and filter on
//! `active = 1`) *and* a GROUP BY post-stage.
//!
//! ## Pipeline
//!
//! 1. Validate that the plan is an `Aggregate` with `pre = Some(_)` and a
//!    single-column GROUP BY.
//! 2. Upload `pre.inputs` from the batch, allocate zero-initialised output
//!    buffers, JIT-compile and launch the pre kernel.
//! 3. If `pre.predicate` is set, build a `u8` keep-mask via the predicate-only
//!    kernel and download it.
//! 4. Download each pre output to a host `Vec`, compacting via the mask if
//!    present.
//! 5. Build a `name -> j` lookup so each `aggregate.inputs[i].name` resolves to
//!    the compacted host vector produced by `pre.outputs[j]`.
//! 6. Allocate the open-addressing keys table, upload the key column as i64,
//!    JIT + launch the keys kernel.
//! 7. For each aggregate, upload its input from the compacted host vector,
//!    allocate the accumulator, and launch the agg kernel. AVG decomposes
//!    into SUM + COUNT; COUNT(*) synthesises an all-ones i64 column.
//! 8. Download the keys table and accumulators, sort by key, and pack into a
//!    `RecordBatch` matching `aggregate.output_schema`.
//!
//! ## Scope (v1)
//!
//! - Single-column GROUP BY only (multi-column is a separate Step).
//! - Int32 / Int64 keys only.
//! - MIN / MAX over float inputs is rejected (would need a CAS loop on sm_70).
//! - `pre.outputs` carrying Bool / Utf8 is rejected; aggregate-input
//!   expressions don't produce those types in practice.
//!
//! The module is intentionally self-contained: helpers are duplicated from
//! `agg_with_pre` and `groupby` rather than reaching across module privacy.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
};
use bytemuck::Pod;

use crate::cuda::buffer::primitive_to_gpu;
use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::expr_agg;
use crate::exec::launch::CudaStream;
use crate::exec::n_rows_to_u32;
use crate::jit::agg_kernels::ReduceOp;
use crate::jit::hash_kernels::{
    compile_groupby_agg_kernel, compile_groupby_keys_kernel, groupby_block_size,
    AGG_KERNEL_ENTRY, KEYS_KERNEL_ENTRY,
};
use crate::jit::{compile_ptx, CudaModule};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Field, Schema};
use crate::plan::physical_plan::{AggregateSpec, ColumnIO, KernelSpec, PhysicalPlan};

/// PTX entry-point name for the pre-projection kernel.
const PRE_KERNEL_ENTRY: &str = "bolt_pre_kernel";

/// PTX entry-point name for the predicate-only kernel that materialises the mask.
const PRE_PREDICATE_ENTRY: &str = "bolt_pre_predicate";

/// Threads per block for the pre-projection / predicate launches.
const PRE_BLOCK_SIZE: u32 = 256;

/// Empty-slot sentinel; mirrors the literal baked into the keys kernel.
const EMPTY_KEY: i64 = i64::MIN;

/// Execute an `Aggregate` plan that has BOTH a pre kernel AND a non-empty
/// `group_by`. Single-column GROUP BY only; Int32 / Int64 keys only.
///
/// Errors with a clear `BoltError` if the plan does not match this shape
/// — callers should dispatch to one of the other executors for the trivial
/// cases.
pub fn execute_groupby_with_pre(
    plan: &PhysicalPlan,
    table_batch: &RecordBatch,
) -> BoltResult<RecordBatch> {
    // -- 1. Validate plan shape. --
    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        other => {
            return Err(BoltError::Other(format!(
                "execute_groupby_with_pre: expected Aggregate plan, got {:?}",
                std::mem::discriminant(other)
            )))
        }
    };

    let pre_spec = pre.as_ref().ok_or_else(|| {
        BoltError::Other(
            "execute_groupby_with_pre: pre kernel is None; use execute_groupby".into(),
        )
    })?;

    if aggregate.group_by.is_empty() {
        return Err(BoltError::Other(
            "execute_groupby_with_pre: aggregate has no GROUP BY columns; \
             use execute_aggregate_with_pre"
                .into(),
        ));
    }
    if aggregate.group_by.len() > 1 {
        return Err(BoltError::Other(
            "agg pre+groupby: multi-column GROUP BY not yet supported".into(),
        ));
    }

    let n_rows = table_batch.num_rows();

    // -- 2-4. Run the pre kernel (and optional predicate) and host-compact
    //         each output column.
    let compacted = run_pre_stage(pre_spec, table_batch, n_rows)?;

    // -- 5. Build a name -> compacted-column-index lookup.
    let name_to_pre_ord = build_pre_output_index(pre_spec)?;

    // Resolve the single GROUP BY key column to its compacted host vector.
    let key_ord_in_inputs = aggregate.group_by[0];
    let key_io = aggregate.inputs.get(key_ord_in_inputs).ok_or_else(|| {
        BoltError::Plan(format!(
            "execute_groupby_with_pre: group_by ordinal {} out of range (only {} inputs)",
            key_ord_in_inputs,
            aggregate.inputs.len()
        ))
    })?;
    let key_pre_ord = *name_to_pre_ord.get(&key_io.name).ok_or_else(|| {
        BoltError::Plan(format!(
            "execute_groupby_with_pre: GROUP BY key '{}' not among pre kernel outputs",
            key_io.name
        ))
    })?;
    let key_host = &compacted.cols[key_pre_ord];

    let original_key_dtype = key_io.dtype;

    // Validate the key dtype against the compacted column's dtype + reject
    // float / Utf8 / Bool keys with a clear message.
    match original_key_dtype {
        DataType::Int32 | DataType::Int64 => {}
        DataType::Float32 | DataType::Float64 => {
            return Err(BoltError::Type(
                "float GROUP BY keys not yet supported".into(),
            ))
        }
        DataType::Utf8 => {
            return Err(BoltError::Type(
                "Utf8 GROUP BY keys not yet supported".into(),
            ))
        }
        DataType::Bool => {
            return Err(BoltError::Type(format!(
                "GROUP BY key dtype {:?} not supported",
                original_key_dtype
            )))
        }
    }
    if !host_col_matches_dtype(key_host, original_key_dtype) {
        return Err(BoltError::Type(format!(
            "GROUP BY key '{}' compacted column dtype does not match plan dtype {:?}",
            key_io.name, original_key_dtype
        )));
    }

    let host_keys: Vec<i64> = key_host.to_i64_for_key(&key_io.name)?;
    let n_compacted = host_keys.len();
    if n_compacted != compacted.n_rows() {
        return Err(BoltError::Other(format!(
            "internal: key column length {} disagrees with compacted row count {}",
            n_compacted,
            compacted.n_rows()
        )));
    }

    // Validate no key collides with the EMPTY_KEY sentinel.
    if host_keys.iter().any(|&k| k == EMPTY_KEY) {
        return Err(BoltError::Other(format!(
            "GROUP BY key column '{}' contains the reserved sentinel value i64::MIN",
            key_io.name
        )));
    }

    // -- 6. Allocate keys table + upload key column. Launch keys kernel.
    let n_unique = unique_count(&host_keys);
    let k = next_pow2((n_unique.saturating_mul(2)).saturating_add(16)).max(64);
    let k_u32 = u32::try_from(k).map_err(|_| {
        BoltError::Other(format!(
            "GROUP BY hash table size {} exceeds u32::MAX",
            k
        ))
    })?;

    let host_keys_init: Vec<i64> = vec![EMPTY_KEY; k];
    let mut keys_table = GpuVec::<i64>::from_slice(&host_keys_init)?;
    let key_col_gpu = GpuVec::<i64>::from_slice(&host_keys)?;

    let stream = CudaStream::null();
    launch_keys_kernel(&key_col_gpu, &mut keys_table, n_compacted, k_u32, &stream)?;

    // -- 7. For each aggregate, prepare its accumulator and launch. The
    //       per-aggregate input column is at position `group_by.len() + i` in
    //       `aggregate.inputs` (mirroring the feed-order lowering in
    //       `physical_plan::lower_aggregate`).
    let n_group_keys = aggregate.group_by.len();
    let mut acc_results: Vec<AccDownload> = Vec::with_capacity(aggregate.aggregates.len());
    for (i, agg) in aggregate.aggregates.iter().enumerate() {
        let input_ord = n_group_keys + i;
        let acc = run_one_aggregate(
            agg,
            aggregate,
            input_ord,
            &name_to_pre_ord,
            &compacted,
            &key_col_gpu,
            &keys_table,
            n_compacted,
            k,
            k_u32,
            &stream,
        )?;
        acc_results.push(acc);
    }

    // -- 8. Download the keys table to drive output assembly.
    let host_keys_table: Vec<i64> = keys_table.to_vec()?;
    drop(keys_table);
    drop(key_col_gpu);

    // Walk filled slots and sort by key for deterministic output ordering.
    let mut groups: Vec<(i64, usize)> = host_keys_table
        .iter()
        .enumerate()
        .filter_map(|(slot, &k_at)| {
            if k_at == EMPTY_KEY {
                None
            } else {
                Some((k_at, slot))
            }
        })
        .collect();
    groups.sort_unstable_by_key(|(k, _)| *k);

    let n_groups = groups.len();
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(1 + aggregate.aggregates.len());

    // Column 0: the GROUP BY key in its original dtype.
    arrays.push(build_key_array(&groups, original_key_dtype)?);

    // Columns 1..N+1: one per aggregate.
    for (i, agg) in aggregate.aggregates.iter().enumerate() {
        let out_field = aggregate.output_schema.fields.get(1 + i).ok_or_else(|| {
            BoltError::Other(format!(
                "execute_groupby_with_pre: output_schema missing field for aggregate index {}",
                i
            ))
        })?;
        let arr = build_agg_array(agg, out_field, &acc_results[i], &groups, n_groups)?;
        arrays.push(arr);
    }

    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
        BoltError::Other(format!(
            "failed to build GROUP-BY-with-pre RecordBatch: {e}"
        ))
    })
}

// ===========================================================================
// Pre stage — lifted from `agg_with_pre.rs::run_pre_stage`.
// ===========================================================================

/// Materialised, host-side, post-filter output of the pre kernel: one host
/// vector per `pre.outputs`, parallel to `pre.outputs` by index.
struct CompactedPreOutputs {
    /// One host-side compacted column per `pre.outputs[i]`.
    cols: Vec<HostCol>,
}

impl CompactedPreOutputs {
    /// Post-filter row count. Identical across all columns.
    fn n_rows(&self) -> usize {
        self.cols.first().map(|c| c.len()).unwrap_or(0)
    }
}

/// Run the pre-projection kernel, optionally the predicate-only kernel, then
/// download each output and host-compact via the mask if any. Returns one
/// host vector per `pre.outputs` (in the same order).
fn run_pre_stage(
    spec: &KernelSpec,
    table_batch: &RecordBatch,
    n_rows: usize,
) -> BoltResult<CompactedPreOutputs> {
    // Reject pre outputs of Bool / Utf8 up front — the downstream group-by
    // and aggregation paths cannot consume them.
    for io in &spec.outputs {
        match io.dtype {
            DataType::Bool | DataType::Utf8 => {
                return Err(BoltError::Type(
                    "Bool/Utf8 in pre outputs not yet supported".into(),
                ))
            }
            _ => {}
        }
    }

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

    // -- Allocate zero-initialised output buffers.
    let mut output_cols: Vec<PreCol> = Vec::with_capacity(spec.outputs.len());
    for io in &spec.outputs {
        output_cols.push(PreCol::alloc_zeros(io.dtype, n_rows)?);
    }

    // -- JIT-compile the projection PTX and load it.
    let ptx = compile_ptx(spec, PRE_KERNEL_ENTRY)?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(PRE_KERNEL_ENTRY)?;

    // -- Assemble kernel parameters: inputs..., outputs..., n_rows u32.
    let mut device_ptrs: Vec<CUdeviceptr> =
        Vec::with_capacity(input_cols.len() + output_cols.len());
    for c in &input_cols {
        device_ptrs.push(c.device_ptr());
    }
    for c in &output_cols {
        device_ptrs.push(c.device_ptr());
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
        let pred_ptx =
            crate::jit::scan_kernel::compile_predicate_kernel(spec, PRE_PREDICATE_ENTRY)?;
        let pred_module = CudaModule::from_ptx(&pred_ptx)?;
        let pred_function = pred_module.function(PRE_PREDICATE_ENTRY)?;

        let mask = crate::exec::compact::alloc_mask_buffer(n_rows)?;
        let input_ptrs: Vec<CUdeviceptr> =
            input_cols.iter().map(|c| c.device_ptr()).collect();
        crate::exec::compact::launch_predicate_kernel(
            pred_function,
            &input_ptrs,
            mask.device_ptr(),
            n_rows_u32,
            &stream,
        )?;
        Some(crate::exec::compact::download_mask(
            mask.device_ptr(),
            n_rows,
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

// ===========================================================================
// GROUP BY stage — lifted from `groupby.rs`, redirected to consume host
// vectors produced by the pre stage instead of reading directly from the
// `RecordBatch`.
// ===========================================================================

/// Build a `name -> j` index where `j` is the ordinal into `pre.outputs` (and,
/// equivalently, into `CompactedPreOutputs::cols`).
fn build_pre_output_index(pre_spec: &KernelSpec) -> BoltResult<HashMap<String, usize>> {
    let mut map: HashMap<String, usize> = HashMap::with_capacity(pre_spec.outputs.len());
    for (j, io) in pre_spec.outputs.iter().enumerate() {
        // If the lowering ever emitted duplicate output names this would be a
        // logic bug — flag rather than silently overwrite.
        if map.insert(io.name.clone(), j).is_some() {
            return Err(BoltError::Other(format!(
                "execute_groupby_with_pre: duplicate pre output name '{}'",
                io.name
            )));
        }
    }
    Ok(map)
}

/// Launch the keys-only kernel.
fn launch_keys_kernel(
    group_col: &GpuVec<i64>,
    keys_table: &mut GpuVec<i64>,
    n_rows: usize,
    k_u32: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    if n_rows == 0 {
        // Nothing to insert; the empty keys table is already correct.
        return Ok(());
    }

    let ptx = compile_groupby_keys_kernel()?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(KEYS_KERNEL_ENTRY)?;

    let mut group_ptr: CUdeviceptr = group_col.device_ptr();
    let mut keys_ptr: CUdeviceptr = keys_table.device_ptr();
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let mut k_param: u32 = k_u32;

    let mut params: [*mut c_void; 4] = [
        &mut group_ptr as *mut CUdeviceptr as *mut c_void,
        &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
        &mut k_param as *mut u32 as *mut c_void,
    ];

    let block = groupby_block_size();
    let grid_x = ((n_rows_u32 + block - 1) / block).max(1);

    // SAFETY: `function` is borrowed from a live `CudaModule`; every entry of
    // `params` points to a stack local that outlives the synchronize.
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
            params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }
    stream.synchronize()?;
    let _ = (group_ptr, keys_ptr);
    Ok(())
}

/// Launch one aggregate-update kernel against a typed input column.
fn launch_agg_kernel<T: Pod>(
    op: ReduceOp,
    input_dtype: DataType,
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    input_col: &GpuVec<T>,
    acc_table: &mut GpuVec<T>,
    n_rows: usize,
    k_u32: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    if n_rows == 0 {
        return Ok(());
    }

    let ptx = compile_groupby_agg_kernel(op, input_dtype)?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(AGG_KERNEL_ENTRY)?;

    let mut group_ptr: CUdeviceptr = group_col.device_ptr();
    let mut keys_ptr: CUdeviceptr = keys_table.device_ptr();
    let mut input_ptr: CUdeviceptr = input_col.device_ptr();
    let mut acc_ptr: CUdeviceptr = acc_table.device_ptr();
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let mut k_param: u32 = k_u32;

    let mut params: [*mut c_void; 6] = [
        &mut group_ptr as *mut CUdeviceptr as *mut c_void,
        &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut input_ptr as *mut CUdeviceptr as *mut c_void,
        &mut acc_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
        &mut k_param as *mut u32 as *mut c_void,
    ];

    let block = groupby_block_size();
    let grid_x = ((n_rows_u32 + block - 1) / block).max(1);

    // SAFETY: see `launch_keys_kernel`.
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
            params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }
    stream.synchronize()?;
    let _ = (group_ptr, keys_ptr, input_ptr, acc_ptr);
    Ok(())
}

/// Downloaded accumulator table for a single aggregate. The dtype variant
/// tells the host-side assembler how to read each slot.
///
/// `SUM(Int32)` widens to an `i64` accumulator (per
/// [`crate::plan::logical_plan::sum_output_dtype`] /
/// [`crate::jit::agg_kernels::reduction_output_dtype`]) and so always lands
/// in the [`AccDownload::I64`] arm; the [`AccDownload::I32`] arm is reserved
/// for `MIN`/`MAX(Int32)` only.
enum AccDownload {
    /// Int32 MIN/MAX result column (length `k`). `SUM(Int32)` is widened
    /// and arrives as [`AccDownload::I64`].
    I32(Vec<i32>),
    /// Int64 SUM/MIN/MAX/COUNT result column (length `k`). Also carries
    /// the widened SUM accumulator for `SUM(Int32)` inputs.
    I64(Vec<i64>),
    /// Float32 SUM result column (length `k`).
    F32(Vec<f32>),
    /// Float64 SUM result column (length `k`).
    F64(Vec<f64>),
    /// AVG: a SUM accumulator (downloaded as f64) and a COUNT accumulator
    /// (downloaded as i64), both length `k`.
    Avg { sum: Vec<f64>, count: Vec<i64> },
}

/// Compile + launch one aggregate kernel (or, for `Avg`, two), download its
/// accumulator table(s), and return the result. Input data is taken from the
/// pre stage's compacted host columns, located by the input's ordinal in
/// `aggregate.inputs` (which matches `pre.outputs` by name + position).
#[allow(clippy::too_many_arguments)]
fn run_one_aggregate(
    agg: &AggregateExpr,
    aggregate: &AggregateSpec,
    input_ord: usize,
    name_to_pre_ord: &HashMap<String, usize>,
    compacted: &CompactedPreOutputs,
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    n_rows: usize,
    k: usize,
    k_u32: u32,
    stream: &CudaStream,
) -> BoltResult<AccDownload> {
    match agg {
        AggregateExpr::Sum(_) | AggregateExpr::Min(_) | AggregateExpr::Max(_) => {
            let op = ReduceOp::from_agg(agg)?;
            let (col_io, resolved) =
                resolve_agg_input_slow(agg, aggregate, input_ord, name_to_pre_ord, compacted)?;
            run_typed_agg(
                op,
                col_io,
                resolved.as_ref(),
                group_col,
                keys_table,
                n_rows,
                k,
                k_u32,
                stream,
            )
        }

        AggregateExpr::Count(_) => {
            // COUNT(*) over groups: synthesise an all-ones i64 input column of
            // length n_rows (the post-mask row count).
            let ones: Vec<i64> = vec![1i64; n_rows];
            let input_gpu = GpuVec::<i64>::from_slice(&ones)?;
            let identity_init: Vec<i64> = vec![0i64; k];
            let mut acc_table = GpuVec::<i64>::from_slice(&identity_init)?;
            launch_agg_kernel::<i64>(
                ReduceOp::Sum,
                DataType::Int64,
                group_col,
                keys_table,
                &input_gpu,
                &mut acc_table,
                n_rows,
                k_u32,
                stream,
            )?;
            Ok(AccDownload::I64(acc_table.to_vec()?))
        }

        AggregateExpr::Avg(_) => {
            // AVG = SUM(expr) / COUNT(*), both grouped. SUM in f64; COUNT in i64.
            let (col_io, resolved) =
                resolve_agg_input_slow(agg, aggregate, input_ord, name_to_pre_ord, compacted)?;
            let host_col = resolved.as_ref();

            // SUM(expr) cast to f64.
            let sum_input: Vec<f64> = host_col.to_f64(&col_io.name)?;
            let input_gpu = GpuVec::<f64>::from_slice(&sum_input)?;
            let sum_init: Vec<f64> = vec![0.0f64; k];
            let mut sum_acc = GpuVec::<f64>::from_slice(&sum_init)?;
            launch_agg_kernel::<f64>(
                ReduceOp::Sum,
                DataType::Float64,
                group_col,
                keys_table,
                &input_gpu,
                &mut sum_acc,
                n_rows,
                k_u32,
                stream,
            )?;
            let sum_host = sum_acc.to_vec()?;

            // COUNT(*) per group.
            let ones: Vec<i64> = vec![1i64; n_rows];
            let count_input = GpuVec::<i64>::from_slice(&ones)?;
            let count_init: Vec<i64> = vec![0i64; k];
            let mut count_acc = GpuVec::<i64>::from_slice(&count_init)?;
            launch_agg_kernel::<i64>(
                ReduceOp::Sum,
                DataType::Int64,
                group_col,
                keys_table,
                &count_input,
                &mut count_acc,
                n_rows,
                k_u32,
                stream,
            )?;
            let count_host = count_acc.to_vec()?;

            Ok(AccDownload::Avg {
                sum: sum_host,
                count: count_host,
            })
        }
    }
}

/// Common path for SUM/MIN/MAX. Uploads the typed input from the compacted
/// host column, allocates a typed accumulator initialised to the op's
/// identity, launches the agg kernel, and downloads the accumulator.
#[allow(clippy::too_many_arguments)]
fn run_typed_agg(
    op: ReduceOp,
    col_io: &ColumnIO,
    host_col: &HostCol,
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    n_rows: usize,
    k: usize,
    k_u32: u32,
    stream: &CudaStream,
) -> BoltResult<AccDownload> {
    if !host_col_matches_dtype(host_col, col_io.dtype) {
        return Err(BoltError::Type(format!(
            "aggregate input '{}' compacted dtype does not match plan dtype {:?}",
            col_io.name, col_io.dtype
        )));
    }

    match (col_io.dtype, host_col) {
        (DataType::Int32, HostCol::I32(host)) => {
            // SUM(Int32) widens to an i64 accumulator (matching the scalar
            // reducer and the narrow GROUP BY path): the plan declares an
            // Int64 output dtype via `crate::plan::logical_plan::sum_output_dtype`,
            // and the kernel emits `atom.global.add.u64`. MIN/MAX over Int32
            // keep their natural i32 width.
            //
            // See also `crate::jit::agg_kernels::reduction_output_dtype`,
            // which mirrors the widening rule on the kernel-emission side.
            if matches!(op, ReduceOp::Sum) {
                let widened: Vec<i64> = host.iter().map(|&x| x as i64).collect();
                let input_gpu = GpuVec::<i64>::from_slice(&widened)?;
                let init: Vec<i64> = vec![identity_i64(op); k];
                let mut acc = GpuVec::<i64>::from_slice(&init)?;
                launch_agg_kernel::<i64>(
                    op,
                    DataType::Int64,
                    group_col,
                    keys_table,
                    &input_gpu,
                    &mut acc,
                    n_rows,
                    k_u32,
                    stream,
                )?;
                return Ok(AccDownload::I64(acc.to_vec()?));
            }

            let input_gpu = GpuVec::<i32>::from_slice(host)?;
            let init: Vec<i32> = vec![identity_i32(op); k];
            let mut acc = GpuVec::<i32>::from_slice(&init)?;
            launch_agg_kernel::<i32>(
                op,
                DataType::Int32,
                group_col,
                keys_table,
                &input_gpu,
                &mut acc,
                n_rows,
                k_u32,
                stream,
            )?;
            Ok(AccDownload::I32(acc.to_vec()?))
        }
        (DataType::Int64, HostCol::I64(host)) => {
            let input_gpu = GpuVec::<i64>::from_slice(host)?;
            let init: Vec<i64> = vec![identity_i64(op); k];
            let mut acc = GpuVec::<i64>::from_slice(&init)?;
            launch_agg_kernel::<i64>(
                op,
                DataType::Int64,
                group_col,
                keys_table,
                &input_gpu,
                &mut acc,
                n_rows,
                k_u32,
                stream,
            )?;
            Ok(AccDownload::I64(acc.to_vec()?))
        }
        (DataType::Float32, HostCol::F32(host)) => {
            if matches!(op, ReduceOp::Min | ReduceOp::Max) {
                return Err(BoltError::Other(
                    "MIN/MAX over float not yet supported".into(),
                ));
            }
            let input_gpu = GpuVec::<f32>::from_slice(host)?;
            let init: Vec<f32> = vec![identity_f32(op); k];
            let mut acc = GpuVec::<f32>::from_slice(&init)?;
            launch_agg_kernel::<f32>(
                op,
                DataType::Float32,
                group_col,
                keys_table,
                &input_gpu,
                &mut acc,
                n_rows,
                k_u32,
                stream,
            )?;
            Ok(AccDownload::F32(acc.to_vec()?))
        }
        (DataType::Float64, HostCol::F64(host)) => {
            if matches!(op, ReduceOp::Min | ReduceOp::Max) {
                return Err(BoltError::Other(
                    "MIN/MAX over float not yet supported".into(),
                ));
            }
            let input_gpu = GpuVec::<f64>::from_slice(host)?;
            let init: Vec<f64> = vec![identity_f64(op); k];
            let mut acc = GpuVec::<f64>::from_slice(&init)?;
            launch_agg_kernel::<f64>(
                op,
                DataType::Float64,
                group_col,
                keys_table,
                &input_gpu,
                &mut acc,
                n_rows,
                k_u32,
                stream,
            )?;
            Ok(AccDownload::F64(acc.to_vec()?))
        }
        (DataType::Bool, _) | (DataType::Utf8, _) => Err(BoltError::Type(format!(
            "aggregate input dtype {:?} not supported (column '{}')",
            col_io.dtype, col_io.name
        ))),
        (dt, _) => Err(BoltError::Type(format!(
            "internal: aggregate input '{}' compacted column variant disagrees with dtype {:?}",
            col_io.name, dt
        ))),
    }
}

/// Resolve an aggregate's input to a `(ColumnIO, ResolvedHostCol)` pair.
///
/// Fast path: when the aggregate's inner expression (after stripping aliases)
/// is a bare column ref whose name matches one of `pre.outputs`, return a
/// borrowed view of that compacted pre output — no extra allocation.
///
/// Slow path: when the aggregate's inner expression is non-trivial (e.g.
/// `Sum(price * tax)` that the planner did not pre-materialise), build an
/// [`expr_agg::ColumnEnv`] from the already-compacted pre outputs and call
/// [`expr_agg::eval_expr`] to materialise the value column. The materialised
/// dtype is `col_io.dtype` — the same dtype the GPU kernel dispatch in
/// [`run_typed_agg`] expects. The returned `col_io` is still the
/// `aggregate.inputs[input_ord]` entry; it carries the right `(name, dtype)`
/// for kernel dispatch and error messages even though no entry of
/// `pre.outputs` matched its name.
fn resolve_agg_input_slow<'a>(
    agg: &AggregateExpr,
    aggregate: &'a AggregateSpec,
    input_ord: usize,
    name_to_pre_ord: &HashMap<String, usize>,
    compacted: &'a CompactedPreOutputs,
) -> BoltResult<(&'a ColumnIO, ResolvedHostCol<'a>)> {
    let col_io = aggregate.inputs.get(input_ord).ok_or_else(|| {
        BoltError::Plan(format!(
            "groupby_with_pre: aggregate input ordinal {} out of range (only {} inputs)",
            input_ord,
            aggregate.inputs.len()
        ))
    })?;

    let inner = inner_expr_of(agg);

    // Fast path: bare column ref into pre outputs.
    if let Some(name) = expr_agg::try_bare_column(inner) {
        if let Some(&pre_ord) = name_to_pre_ord.get(name) {
            let host_col = compacted.cols.get(pre_ord).ok_or_else(|| {
                BoltError::Other(format!(
                    "internal: pre output ordinal {} out of range (have {} compacted cols)",
                    pre_ord,
                    compacted.cols.len()
                ))
            })?;
            return Ok((col_io, ResolvedHostCol::Borrowed(host_col)));
        }
        // Bare-column but no matching pre output — preserve the legacy error
        // shape so plan-level mismatches surface the same way as before.
        return Err(BoltError::Plan(format!(
            "aggregate input '{}' not found among pre kernel outputs",
            name
        )));
    }

    // Slow path: materialise via the host-side evaluator over the compacted
    // pre outputs. Use `col_io.dtype` as the materialisation target so the
    // downstream GPU kernel dispatch in `run_typed_agg` sees a HostCol variant
    // that matches what the plan declared.
    let n_rows = compacted.n_rows();
    let wrapped: Vec<(String, expr_agg::HostColumn)> = compacted
        .cols
        .iter()
        .enumerate()
        .map(|(j, c)| {
            let name = compacted_pre_output_name(j, name_to_pre_ord);
            (name, to_expr_host(c))
        })
        .collect();
    let env: expr_agg::ColumnEnv<'_> = wrapped.iter().map(|(n, c)| (n.clone(), c)).collect();
    let materialised = expr_agg::eval_expr(inner, &env, col_io.dtype, n_rows)?;
    Ok((col_io, ResolvedHostCol::Owned(from_expr_host(materialised)?)))
}

/// Recover a pre output's name from its ordinal by reverse-searching
/// `name_to_pre_ord`. We need this because the slow path builds the
/// evaluator env from the compacted column store, which is indexed by
/// ordinal rather than name. Linear search is fine: `pre.outputs` is small
/// (typically O(few)).
fn compacted_pre_output_name(
    ord: usize,
    name_to_pre_ord: &HashMap<String, usize>,
) -> String {
    for (name, &o) in name_to_pre_ord.iter() {
        if o == ord {
            return name.clone();
        }
    }
    // The map is bijective by construction (`build_pre_output_index` rejects
    // duplicates), so this branch is unreachable in practice. Return a name
    // that cannot collide with any real column to keep the env consistent if
    // we ever do hit it.
    format!("__pre_output_{ord}")
}

/// Borrowed or owned host column. Lets the fast path return `&HostCol` from
/// the compacted store while the slow path returns a freshly-materialised
/// value from `expr_agg`, with a single `as_ref()` method to feed either into
/// the typed-agg dispatch.
enum ResolvedHostCol<'a> {
    Borrowed(&'a HostCol),
    Owned(HostCol),
}

impl<'a> ResolvedHostCol<'a> {
    fn as_ref(&self) -> &HostCol {
        match self {
            ResolvedHostCol::Borrowed(c) => *c,
            ResolvedHostCol::Owned(c) => c,
        }
    }
}

/// Unwrap the inner expression of an [`AggregateExpr`]. Duplicated from
/// `aggregate.rs::agg_inner_expr` because that helper is module-private.
fn inner_expr_of(agg: &AggregateExpr) -> &Expr {
    match agg {
        AggregateExpr::Sum(e)
        | AggregateExpr::Min(e)
        | AggregateExpr::Max(e)
        | AggregateExpr::Avg(e)
        | AggregateExpr::Count(e) => e,
    }
}

/// Lift a local primitive [`HostCol`] into the `Option`-carrying
/// [`expr_agg::HostColumn`] shape consumed by the host-side evaluator. The
/// `pre`-stage output path has no NULL bitmap support, so every cell is
/// `Some(_)`.
fn to_expr_host(c: &HostCol) -> expr_agg::HostColumn {
    match c {
        HostCol::I32(v) => expr_agg::HostColumn::I32(v.iter().map(|x| Some(*x)).collect()),
        HostCol::I64(v) => expr_agg::HostColumn::I64(v.iter().map(|x| Some(*x)).collect()),
        HostCol::F32(v) => expr_agg::HostColumn::F32(v.iter().map(|x| Some(*x)).collect()),
        HostCol::F64(v) => expr_agg::HostColumn::F64(v.iter().map(|x| Some(*x)).collect()),
    }
}

/// Convert a materialised [`expr_agg::HostColumn`] back into the local
/// primitive [`HostCol`] shape consumed by the GPU agg kernels.
///
/// **NULL handling**: any residual `None` is a logic bug — we are post
/// predicate-mask compaction at this point, and `PreCol::upload` already
/// rejects NULL-bearing input columns (see TODO there about full validity
/// propagation). If a `None` reaches us here it means a NULL slipped past
/// both gates, so we surface it as an error rather than silently collapse
/// it to the dtype's zero (which would re-introduce the C2/C2b silent
/// wrong-answer bug for any future call site that lacks a NULL-stripping
/// predicate).
///
/// Bool / Utf8 materialisations are rejected — the GPU agg kernels only
/// accept primitive numeric inputs.
fn from_expr_host(c: expr_agg::HostColumn) -> BoltResult<HostCol> {
    fn collapse<T: Copy>(label: &str, v: Vec<Option<T>>) -> BoltResult<Vec<T>> {
        let mut out: Vec<T> = Vec::with_capacity(v.len());
        for (i, x) in v.into_iter().enumerate() {
            match x {
                Some(val) => out.push(val),
                None => {
                    return Err(BoltError::Other(format!(
                        "groupby_with_pre: from_expr_host received a NULL at row \
                         {} of {} column; pre-stage NULL propagation is not yet \
                         implemented and predicate compaction did not strip \
                         this row",
                        i, label
                    )));
                }
            }
        }
        Ok(out)
    }

    match c {
        expr_agg::HostColumn::I32(v) => Ok(HostCol::I32(collapse("I32", v)?)),
        expr_agg::HostColumn::I64(v) => Ok(HostCol::I64(collapse("I64", v)?)),
        expr_agg::HostColumn::F32(v) => Ok(HostCol::F32(collapse("F32", v)?)),
        expr_agg::HostColumn::F64(v) => Ok(HostCol::F64(collapse("F64", v)?)),
        expr_agg::HostColumn::Bool(_) | expr_agg::HostColumn::Utf8(_) => {
            Err(BoltError::Type(
                "groupby_with_pre: Bool/Utf8 aggregate inputs not supported by the \
                 primitive reduction path"
                    .into(),
            ))
        }
    }
}

// ===========================================================================
// Output assembly — mirrors `groupby.rs`.
// ===========================================================================

/// Build the group-by key column as an Arrow array of the original key dtype.
fn build_key_array(
    groups: &[(i64, usize)],
    original_key_dtype: DataType,
) -> BoltResult<ArrayRef> {
    match original_key_dtype {
        DataType::Int32 => {
            let mut out: Vec<i32> = Vec::with_capacity(groups.len());
            for (k, _) in groups {
                let v = i32::try_from(*k).map_err(|_| {
                    BoltError::Type(format!(
                        "GROUP BY: key {} does not fit in Int32 on output",
                        k
                    ))
                })?;
                out.push(v);
            }
            Ok(Arc::new(Int32Array::from(out)) as ArrayRef)
        }
        DataType::Int64 => {
            let out: Vec<i64> = groups.iter().map(|(k, _)| *k).collect();
            Ok(Arc::new(Int64Array::from(out)) as ArrayRef)
        }
        other => Err(BoltError::Type(format!(
            "GROUP BY key dtype {:?} not supported on output",
            other
        ))),
    }
}

/// Build the output Arrow array for one aggregate, indexing the downloaded
/// accumulator by each group's slot.
fn build_agg_array(
    agg: &AggregateExpr,
    out_field: &Field,
    acc: &AccDownload,
    groups: &[(i64, usize)],
    n_groups: usize,
) -> BoltResult<ArrayRef> {
    match (agg, acc) {
        (AggregateExpr::Count(_), AccDownload::I64(host)) => {
            let mut out: Vec<i64> = Vec::with_capacity(n_groups);
            for (_, slot) in groups {
                out.push(host[*slot]);
            }
            pack_array(out_field.dtype, Scalars::I64(out))
        }
        (AggregateExpr::Avg(_), AccDownload::Avg { sum, count }) => {
            let mut out: Vec<f64> = Vec::with_capacity(n_groups);
            for (_, slot) in groups {
                let s = sum[*slot];
                let c = count[*slot];
                let v = if c == 0 { 0.0 } else { s / (c as f64) };
                out.push(v);
            }
            pack_array(out_field.dtype, Scalars::F64(out))
        }
        (AggregateExpr::Sum(_) | AggregateExpr::Min(_) | AggregateExpr::Max(_), other) => {
            let scalars = match other {
                AccDownload::I32(host) => Scalars::I32(
                    groups.iter().map(|(_, slot)| host[*slot]).collect(),
                ),
                AccDownload::I64(host) => Scalars::I64(
                    groups.iter().map(|(_, slot)| host[*slot]).collect(),
                ),
                AccDownload::F32(host) => Scalars::F32(
                    groups.iter().map(|(_, slot)| host[*slot]).collect(),
                ),
                AccDownload::F64(host) => Scalars::F64(
                    groups.iter().map(|(_, slot)| host[*slot]).collect(),
                ),
                AccDownload::Avg { .. } => {
                    return Err(BoltError::Other(
                        "internal: AVG accumulator passed to non-AVG aggregate".into(),
                    ))
                }
            };
            pack_array(out_field.dtype, scalars)
        }
        (_, _) => Err(BoltError::Other(
            "internal: aggregate / accumulator-variant mismatch".into(),
        )),
    }
}

/// Typed batch of per-group scalar values, prior to dtype-casting into Arrow.
enum Scalars {
    /// Int32 column.
    I32(Vec<i32>),
    /// Int64 column.
    I64(Vec<i64>),
    /// Float32 column.
    F32(Vec<f32>),
    /// Float64 column.
    F64(Vec<f64>),
}

/// Cast a `Scalars` batch into an Arrow array of `out_dtype`.
fn pack_array(out_dtype: DataType, scalars: Scalars) -> BoltResult<ArrayRef> {
    match (scalars, out_dtype) {
        (Scalars::I32(v), DataType::Int32) => Ok(Arc::new(Int32Array::from(v)) as ArrayRef),
        (Scalars::I64(v), DataType::Int64) => Ok(Arc::new(Int64Array::from(v)) as ArrayRef),
        (Scalars::F32(v), DataType::Float32) => {
            Ok(Arc::new(Float32Array::from(v)) as ArrayRef)
        }
        (Scalars::F64(v), DataType::Float64) => {
            Ok(Arc::new(Float64Array::from(v)) as ArrayRef)
        }

        // Cross-dtype paths consistent with the scalar reducer.
        (Scalars::I32(v), DataType::Int64) => Ok(Arc::new(Int64Array::from(
            v.into_iter().map(|x| x as i64).collect::<Vec<_>>(),
        )) as ArrayRef),
        (Scalars::I32(v), DataType::Float32) => Ok(Arc::new(Float32Array::from(
            v.into_iter().map(|x| x as f32).collect::<Vec<_>>(),
        )) as ArrayRef),
        (Scalars::I32(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(
            v.into_iter().map(|x| x as f64).collect::<Vec<_>>(),
        )) as ArrayRef),
        (Scalars::I64(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(
            v.into_iter().map(|x| x as f64).collect::<Vec<_>>(),
        )) as ArrayRef),
        (Scalars::F32(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(
            v.into_iter().map(|x| x as f64).collect::<Vec<_>>(),
        )) as ArrayRef),

        (_, dt) => Err(BoltError::Type(format!(
            "GROUP BY: cannot pack scalars into output dtype {:?}",
            dt
        ))),
    }
}

// ===========================================================================
// Heterogenous device + host columns. Lifted from `agg_with_pre.rs`.
// ===========================================================================

/// Heterogenous owned device column for the pre kernel's inputs and outputs.
enum PreCol {
    I32(GpuVec<i32>),
    I64(GpuVec<i64>),
    F32(GpuVec<f32>),
    F64(GpuVec<f64>),
}

impl PreCol {
    /// Upload an Arrow array to the GPU, downcasting per `dtype`.
    ///
    /// **NULL handling**: rejects any input array with `null_count() > 0`.
    /// The pre-stage GPU kernels operate on raw value buffers and do not
    /// (yet) carry a validity bitmap, so a NULL would silently feed garbage
    /// into multiplications / additions. The clean error is the safe
    /// surgical fix; full propagation is tracked below.
    ///
    /// TODO(pre-stage-nulls): propagate `arr.nulls()` into a parallel
    /// `valid_mask: GpuBuffer<u8>` per `PreCol`, then have the JIT kernel
    /// emit `value if valid else identity` per op so NULL semantics survive
    /// the pre-stage. Until then, planners that introduce a pre kernel
    /// over a NULL-bearing source column must ensure either (a) the source
    /// has been filtered upstream or (b) the column is unconditionally
    /// non-null.
    fn upload(arr: &dyn Array, dtype: DataType) -> BoltResult<Self> {
        if arr.null_count() > 0 {
            return Err(BoltError::Type(format!(
                "groupby_with_pre: pre-stage kernels do not yet propagate NULL \
                 validity; input column with dtype {:?} has {} NULL(s). \
                 See TODO(pre-stage-nulls) in groupby_with_pre.rs.",
                dtype,
                arr.null_count()
            )));
        }
        match dtype {
            DataType::Int32 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| downcast_err("input", "Int32"))?;
                Ok(PreCol::I32(GpuVec::from_buffer(primitive_to_gpu(pa)?)))
            }
            DataType::Int64 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| downcast_err("input", "Int64"))?;
                Ok(PreCol::I64(GpuVec::from_buffer(primitive_to_gpu(pa)?)))
            }
            DataType::Float32 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| downcast_err("input", "Float32"))?;
                Ok(PreCol::F32(GpuVec::from_buffer(primitive_to_gpu(pa)?)))
            }
            DataType::Float64 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| downcast_err("input", "Float64"))?;
                Ok(PreCol::F64(GpuVec::from_buffer(primitive_to_gpu(pa)?)))
            }
            DataType::Bool | DataType::Utf8 => Err(BoltError::Type(format!(
                "groupby_with_pre: pre kernel column dtype {:?} not supported",
                dtype
            ))),
        }
    }

    /// Allocate a zero-initialised device column of `n` rows.
    fn alloc_zeros(dtype: DataType, n: usize) -> BoltResult<Self> {
        match dtype {
            DataType::Int32 => Ok(PreCol::I32(GpuVec::<i32>::zeros(n)?)),
            DataType::Int64 => Ok(PreCol::I64(GpuVec::<i64>::zeros(n)?)),
            DataType::Float32 => Ok(PreCol::F32(GpuVec::<f32>::zeros(n)?)),
            DataType::Float64 => Ok(PreCol::F64(GpuVec::<f64>::zeros(n)?)),
            DataType::Bool | DataType::Utf8 => Err(BoltError::Type(format!(
                "groupby_with_pre: pre kernel output dtype {:?} not supported",
                dtype
            ))),
        }
    }

    /// Raw device pointer for kernel-parameter assembly.
    fn device_ptr(&self) -> CUdeviceptr {
        match self {
            PreCol::I32(v) => v.device_ptr(),
            PreCol::I64(v) => v.device_ptr(),
            PreCol::F32(v) => v.device_ptr(),
            PreCol::F64(v) => v.device_ptr(),
        }
    }

    /// Download the column to host and verify the length matches `n_rows`.
    fn to_host_col(self, n_rows: usize) -> BoltResult<HostCol> {
        match self {
            PreCol::I32(v) => Ok(HostCol::I32(copy_back::<i32>(&v, n_rows)?)),
            PreCol::I64(v) => Ok(HostCol::I64(copy_back::<i64>(&v, n_rows)?)),
            PreCol::F32(v) => Ok(HostCol::F32(copy_back::<f32>(&v, n_rows)?)),
            PreCol::F64(v) => Ok(HostCol::F64(copy_back::<f64>(&v, n_rows)?)),
        }
    }
}

/// Host-side typed column produced by downloading a `PreCol`.
enum HostCol {
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl HostCol {
    /// Number of elements in the column.
    fn len(&self) -> usize {
        match self {
            HostCol::I32(v) => v.len(),
            HostCol::I64(v) => v.len(),
            HostCol::F32(v) => v.len(),
            HostCol::F64(v) => v.len(),
        }
    }

    /// Return a new column containing only positions where `mask[i]` is true.
    fn compact(self, mask: &[bool]) -> BoltResult<HostCol> {
        if mask.len() != self.len() {
            return Err(BoltError::Other(format!(
                "groupby_with_pre: mask length {} != column length {}",
                mask.len(),
                self.len()
            )));
        }
        Ok(match self {
            HostCol::I32(v) => HostCol::I32(filter_vec(v, mask)),
            HostCol::I64(v) => HostCol::I64(filter_vec(v, mask)),
            HostCol::F32(v) => HostCol::F32(filter_vec(v, mask)),
            HostCol::F64(v) => HostCol::F64(filter_vec(v, mask)),
        })
    }

    /// Convert the column to a `Vec<i64>` for use as a GROUP BY key. Int32
    /// upcasts; everything else is an error.
    fn to_i64_for_key(&self, col_name: &str) -> BoltResult<Vec<i64>> {
        match self {
            HostCol::I32(v) => Ok(v.iter().map(|&x| x as i64).collect()),
            HostCol::I64(v) => Ok(v.clone()),
            HostCol::F32(_) | HostCol::F64(_) => Err(BoltError::Type(format!(
                "float GROUP BY keys not yet supported (column '{}')",
                col_name
            ))),
        }
    }

    /// Convert the column to a `Vec<f64>` for use as an AVG input. Any
    /// numeric variant upcasts. `col_name` is only used to enrich a future
    /// error message if we ever extend this to reject non-numeric variants.
    fn to_f64(&self, _col_name: &str) -> BoltResult<Vec<f64>> {
        match self {
            HostCol::I32(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            HostCol::I64(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            HostCol::F32(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            HostCol::F64(v) => Ok(v.clone()),
        }
    }
}

/// True iff the `HostCol` variant matches the plan dtype it should carry.
fn host_col_matches_dtype(col: &HostCol, dtype: DataType) -> bool {
    matches!(
        (col, dtype),
        (HostCol::I32(_), DataType::Int32)
            | (HostCol::I64(_), DataType::Int64)
            | (HostCol::F32(_), DataType::Float32)
            | (HostCol::F64(_), DataType::Float64)
    )
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

// ===========================================================================
// Identities for the accumulator initialiser.
// ===========================================================================

fn identity_i32(op: ReduceOp) -> i32 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => 0,
        ReduceOp::Min => i32::MAX,
        ReduceOp::Max => i32::MIN,
    }
}

fn identity_i64(op: ReduceOp) -> i64 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => 0,
        ReduceOp::Min => i64::MAX,
        ReduceOp::Max => i64::MIN,
    }
}

fn identity_f32(op: ReduceOp) -> f32 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => 0.0,
        ReduceOp::Min => f32::INFINITY,
        ReduceOp::Max => f32::NEG_INFINITY,
    }
}

fn identity_f64(op: ReduceOp) -> f64 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => 0.0,
        ReduceOp::Min => f64::INFINITY,
        ReduceOp::Max => f64::NEG_INFINITY,
    }
}

// ===========================================================================
// Misc helpers.
// ===========================================================================

/// Count of distinct values in `keys`. Linear, O(n) using a HashSet.
fn unique_count(keys: &[i64]) -> usize {
    let mut set: HashSet<i64> = HashSet::with_capacity(keys.len().min(1 << 20));
    for &k in keys {
        set.insert(k);
    }
    set.len()
}

/// Smallest power of two >= `n`.
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p: usize = 1;
    while p < n {
        match p.checked_mul(2) {
            Some(v) => p = v,
            None => return p,
        }
    }
    p
}

/// Build a `Type` error for a failed Arrow downcast.
fn downcast_err(role: &str, expected: &str) -> BoltError {
    BoltError::Type(format!(
        "groupby_with_pre: pre kernel {} could not be downcast to {}",
        role, expected
    ))
}

/// Map Arrow `DataType` to our plan `DataType`.
fn arrow_dtype_to_plan(d: &ArrowDataType) -> BoltResult<DataType> {
    match d {
        ArrowDataType::Int32 => Ok(DataType::Int32),
        ArrowDataType::Int64 => Ok(DataType::Int64),
        ArrowDataType::Float32 => Ok(DataType::Float32),
        ArrowDataType::Float64 => Ok(DataType::Float64),
        ArrowDataType::Boolean => Ok(DataType::Bool),
        ArrowDataType::Utf8 => Ok(DataType::Utf8),
        other => Err(BoltError::Type(format!(
            "unsupported Arrow dtype {:?}",
            other
        ))),
    }
}

/// Map our plan `DataType` to Arrow `DataType`.
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

/// Build an Arrow `Schema` from our plan `Schema` for the output `RecordBatch`.
fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}

#[cfg(test)]
mod null_handling_tests {
    //! Host-only tests for the NULL-handling gates added in the pre-stage:
    //!   1. `from_expr_host` errors on residual `None` (rather than silently
    //!      collapsing to the dtype zero).
    //!   2. `PreCol::upload` rejects Arrow arrays whose `null_count() > 0`
    //!      (the pre-stage GPU kernels do not yet carry validity, so a NULL
    //!      would multiply garbage in expressions like `price * tax`).
    //!
    //! No CUDA touched: both error paths short-circuit before any device
    //! allocation.
    use super::*;

    #[test]
    fn from_expr_host_errors_on_residual_none_i32() {
        let col = expr_agg::HostColumn::I32(vec![Some(1), None]);
        let err = from_expr_host(col).expect_err("expected NULL to be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("NULL"), "error should mention NULL: {msg}");
        assert!(
            msg.contains("groupby_with_pre"),
            "error should be tagged with the module name, got: {msg}"
        );
    }

    #[test]
    fn from_expr_host_errors_on_residual_none_f64() {
        let col = expr_agg::HostColumn::F64(vec![None, Some(2.0)]);
        let err = from_expr_host(col).expect_err("expected NULL to be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("row 0"), "should point at row 0: {msg}");
    }

    #[test]
    fn from_expr_host_passes_through_all_some_i64() {
        let col = expr_agg::HostColumn::I64(vec![Some(1), Some(2), Some(3)]);
        let out = from_expr_host(col).expect("all-Some input should pass through");
        match out {
            HostCol::I64(v) => assert_eq!(v, vec![1, 2, 3]),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn from_expr_host_rejects_bool_and_utf8() {
        let bool_col = expr_agg::HostColumn::Bool(vec![Some(true)]);
        assert!(from_expr_host(bool_col).is_err());
        let utf8_col = expr_agg::HostColumn::Utf8(vec![Some("x".into())]);
        assert!(from_expr_host(utf8_col).is_err());
    }

    #[test]
    fn pre_col_upload_rejects_null_bearing_input() {
        let arr = Float64Array::from(vec![Some(1.0f64), None, Some(3.0f64)]);
        assert!(arr.null_count() > 0, "test fixture should carry a NULL");
        let err = PreCol::upload(&arr, DataType::Float64)
            .expect_err("PreCol::upload should reject NULL-bearing input");
        let msg = format!("{err}");
        assert!(msg.contains("NULL"), "error should mention NULL: {msg}");
        assert!(
            msg.contains("pre-stage") || msg.contains("validity"),
            "error should explain why, got: {msg}"
        );
    }
}
