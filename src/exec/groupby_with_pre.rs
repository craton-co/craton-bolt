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
//! 7. For each aggregate, resolve the input via [`resolve_agg_input_slow`]
//!    (returning a `(value_col, optional_validity_mask)` pair), filter both
//!    the key column and the value column by the validity mask to drop
//!    NULL rows (SQL: `MIN`/`MAX`/`SUM`/`AVG`/`COUNT(col)` ignore NULLs),
//!    upload, allocate the accumulator, and launch the agg kernel. AVG
//!    decomposes into SUM + COUNT over the same filtered pair; COUNT
//!    synthesises an all-ones i64 column at the post-filter length so
//!    COUNT(col) excludes NULL rows and COUNT(*) (i.e. `Count(Literal(1))`)
//!    counts every surviving row.
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
use crate::exec::launch::{grid_x_for, CudaStream};
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
            &host_keys,
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
    let grid_x = grid_x_for(n_rows_u32, block);

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
    let grid_x = grid_x_for(n_rows_u32, block);

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
///
/// **NULL handling**: SQL aggregate semantics require `SUM`/`MIN`/`MAX`/`AVG`/
/// `COUNT(col)` to ignore NULL rows. The slow-path resolver
/// ([`resolve_agg_input_slow`]) returns a per-row validity mask alongside the
/// value column. When any row is NULL we filter both the value column and
/// the parallel group-key column to keep only valid rows before uploading
/// them to the GPU. The keys-table itself is shared across aggregates and
/// is built once from the unfiltered key column.
///
/// A previous version of this function used a separate `from_expr_host` that
/// coerced NULLs to dtype-zero, which broke `MIN([NULL, 5]) = 5` (returned 0)
/// and `AVG([NULL, 4]) = 4.0` (returned 2.0 because the denominator was
/// inflated by the NULL row). See `agg_with_pre.rs` for the matching scalar
/// fix.
#[allow(clippy::too_many_arguments)]
fn run_one_aggregate(
    agg: &AggregateExpr,
    aggregate: &AggregateSpec,
    input_ord: usize,
    name_to_pre_ord: &HashMap<String, usize>,
    compacted: &CompactedPreOutputs,
    host_keys: &[i64],
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
                &resolved,
                host_keys,
                group_col,
                keys_table,
                n_rows,
                k,
                k_u32,
                stream,
            )
        }

        AggregateExpr::Count(_) => {
            // SQL COUNT(col) counts non-NULL rows per group; COUNT(*) (planner-
            // emitted as `Count(Literal(1))`) counts every surviving row per
            // group. The distinction falls out naturally: `resolve_agg_input_slow`
            // builds a validity mask reflecting NULLs in the resolved
            // expression, and a literal can never produce NULLs so its mask is
            // all-true. We then SUM(ones) per group over the validity-filtered
            // (group, ones) pair.
            let (_col_io, resolved) =
                resolve_agg_input_slow(agg, aggregate, input_ord, name_to_pre_ord, compacted)?;
            let (filtered_keys_gpu, n_valid) =
                filter_keys_if_needed(host_keys, &resolved)?;
            let count_group_col: &GpuVec<i64> =
                filtered_keys_gpu.as_ref().unwrap_or(group_col);
            let ones: Vec<i64> = vec![1i64; n_valid];
            let input_gpu = GpuVec::<i64>::from_slice(&ones)?;
            let identity_init: Vec<i64> = vec![0i64; k];
            let mut acc_table = GpuVec::<i64>::from_slice(&identity_init)?;
            launch_agg_kernel::<i64>(
                ReduceOp::Sum,
                DataType::Int64,
                count_group_col,
                keys_table,
                &input_gpu,
                &mut acc_table,
                n_valid,
                k_u32,
                stream,
            )?;
            Ok(AccDownload::I64(acc_table.to_vec()?))
        }

        AggregateExpr::Avg(_) => {
            // AVG = SUM(expr) / COUNT(expr), both grouped, SQL semantics:
            // NULL rows are excluded from both numerator and denominator.
            // SUM in f64; COUNT in i64.
            let (col_io, resolved) =
                resolve_agg_input_slow(agg, aggregate, input_ord, name_to_pre_ord, compacted)?;
            let host_col = resolved.as_ref();

            // Build the filtered key column once for this aggregate (used by
            // both the SUM and COUNT kernel launches).
            let (filtered_keys_gpu, n_valid) =
                filter_keys_if_needed(host_keys, &resolved)?;
            let avg_group_col: &GpuVec<i64> =
                filtered_keys_gpu.as_ref().unwrap_or(group_col);

            // Build the filtered value column at f64.
            let sum_input: Vec<f64> = match resolved.validity() {
                Some(mask) => host_col.to_f64_filtered(mask, &col_io.name)?,
                None => host_col.to_f64(&col_io.name)?,
            };
            debug_assert_eq!(sum_input.len(), n_valid);
            let input_gpu = GpuVec::<f64>::from_slice(&sum_input)?;
            let sum_init: Vec<f64> = vec![0.0f64; k];
            let mut sum_acc = GpuVec::<f64>::from_slice(&sum_init)?;
            launch_agg_kernel::<f64>(
                ReduceOp::Sum,
                DataType::Float64,
                avg_group_col,
                keys_table,
                &input_gpu,
                &mut sum_acc,
                n_valid,
                k_u32,
                stream,
            )?;
            let sum_host = sum_acc.to_vec()?;

            // COUNT(non-NULL) per group over the same filtered keys.
            let ones: Vec<i64> = vec![1i64; n_valid];
            let count_input = GpuVec::<i64>::from_slice(&ones)?;
            let count_init: Vec<i64> = vec![0i64; k];
            let mut count_acc = GpuVec::<i64>::from_slice(&count_init)?;
            launch_agg_kernel::<i64>(
                ReduceOp::Sum,
                DataType::Int64,
                avg_group_col,
                keys_table,
                &count_input,
                &mut count_acc,
                n_valid,
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

/// If `resolved` carries a validity mask with any false bits, build a fresh
/// GPU key column containing only the rows where validity is true, and
/// return it alongside the number of surviving rows. Otherwise return
/// `(None, host_keys.len())` and the caller should keep using the shared
/// unfiltered group column.
///
/// Used by SUM/MIN/MAX/AVG/COUNT(col) to keep the key column positionally
/// aligned with the NULL-filtered value column at GPU upload time.
fn filter_keys_if_needed(
    host_keys: &[i64],
    resolved: &ResolvedHostCol<'_>,
) -> BoltResult<(Option<GpuVec<i64>>, usize)> {
    match resolved.validity() {
        Some(mask) => {
            if mask.len() != host_keys.len() {
                return Err(BoltError::Other(format!(
                    "groupby_with_pre: validity mask length {} != key column length {}",
                    mask.len(),
                    host_keys.len()
                )));
            }
            let filtered: Vec<i64> = host_keys
                .iter()
                .zip(mask.iter())
                .filter_map(|(&k, &v)| if v { Some(k) } else { None })
                .collect();
            let n = filtered.len();
            let gpu = GpuVec::<i64>::from_slice(&filtered)?;
            Ok((Some(gpu), n))
        }
        None => Ok((None, host_keys.len())),
    }
}

/// Common path for SUM/MIN/MAX. Uploads the typed input from the compacted
/// host column, allocates a typed accumulator initialised to the op's
/// identity, launches the agg kernel, and downloads the accumulator.
///
/// When the resolved column carries a validity mask with any false bits,
/// the value column and the group-key column are both filtered down to
/// the validity-true rows before upload so that NULL rows are skipped from
/// the GROUP BY reduction (SQL semantics: `MIN([NULL, 5]) = 5`).
#[allow(clippy::too_many_arguments)]
fn run_typed_agg(
    op: ReduceOp,
    col_io: &ColumnIO,
    resolved: &ResolvedHostCol<'_>,
    host_keys: &[i64],
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    n_rows: usize,
    k: usize,
    k_u32: u32,
    stream: &CudaStream,
) -> BoltResult<AccDownload> {
    let host_col = resolved.as_ref();
    if !host_col_matches_dtype(host_col, col_io.dtype) {
        return Err(BoltError::Type(format!(
            "aggregate input '{}' compacted dtype does not match plan dtype {:?}",
            col_io.name, col_io.dtype
        )));
    }
    debug_assert_eq!(
        host_keys.len(),
        n_rows,
        "groupby_with_pre: host_keys / n_rows mismatch — invariant violated by caller"
    );

    // If the resolved column has a validity mask with any false bits, filter
    // both (keys, values) together so the GPU sees a parallel pair of rows
    // with no NULL slots. The keys-table is shared and remains correct
    // because it indexes the original key set.
    let (filtered_keys_gpu, n_eff) = filter_keys_if_needed(host_keys, resolved)?;
    let eff_group_col: &GpuVec<i64> = filtered_keys_gpu.as_ref().unwrap_or(group_col);
    let validity = resolved.validity();

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
                let widened: Vec<i64> = match validity {
                    Some(mask) => host
                        .iter()
                        .zip(mask.iter())
                        .filter_map(|(&x, &v)| if v { Some(x as i64) } else { None })
                        .collect(),
                    None => host.iter().map(|&x| x as i64).collect(),
                };
                let input_gpu = GpuVec::<i64>::from_slice(&widened)?;
                let init: Vec<i64> = vec![identity_i64(op); k];
                let mut acc = GpuVec::<i64>::from_slice(&init)?;
                launch_agg_kernel::<i64>(
                    op,
                    DataType::Int64,
                    eff_group_col,
                    keys_table,
                    &input_gpu,
                    &mut acc,
                    n_eff,
                    k_u32,
                    stream,
                )?;
                return Ok(AccDownload::I64(acc.to_vec()?));
            }

            let filtered: Vec<i32> = filter_by_validity(host, validity);
            let input_gpu = GpuVec::<i32>::from_slice(&filtered)?;
            let init: Vec<i32> = vec![identity_i32(op); k];
            let mut acc = GpuVec::<i32>::from_slice(&init)?;
            launch_agg_kernel::<i32>(
                op,
                DataType::Int32,
                eff_group_col,
                keys_table,
                &input_gpu,
                &mut acc,
                n_eff,
                k_u32,
                stream,
            )?;
            Ok(AccDownload::I32(acc.to_vec()?))
        }
        (DataType::Int64, HostCol::I64(host)) => {
            let filtered: Vec<i64> = filter_by_validity(host, validity);
            let input_gpu = GpuVec::<i64>::from_slice(&filtered)?;
            let init: Vec<i64> = vec![identity_i64(op); k];
            let mut acc = GpuVec::<i64>::from_slice(&init)?;
            launch_agg_kernel::<i64>(
                op,
                DataType::Int64,
                eff_group_col,
                keys_table,
                &input_gpu,
                &mut acc,
                n_eff,
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
            let filtered: Vec<f32> = filter_by_validity(host, validity);
            let input_gpu = GpuVec::<f32>::from_slice(&filtered)?;
            let init: Vec<f32> = vec![identity_f32(op); k];
            let mut acc = GpuVec::<f32>::from_slice(&init)?;
            launch_agg_kernel::<f32>(
                op,
                DataType::Float32,
                eff_group_col,
                keys_table,
                &input_gpu,
                &mut acc,
                n_eff,
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
            let filtered: Vec<f64> = filter_by_validity(host, validity);
            let input_gpu = GpuVec::<f64>::from_slice(&filtered)?;
            let init: Vec<f64> = vec![identity_f64(op); k];
            let mut acc = GpuVec::<f64>::from_slice(&init)?;
            launch_agg_kernel::<f64>(
                op,
                DataType::Float64,
                eff_group_col,
                keys_table,
                &input_gpu,
                &mut acc,
                n_eff,
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

/// Return a `Vec<T>` containing `host[i]` only where `validity[i]` is `true`.
/// If `validity` is `None` (fast path / no NULLs), the full slice is cloned.
fn filter_by_validity<T: Copy>(host: &[T], validity: Option<&[bool]>) -> Vec<T> {
    match validity {
        Some(mask) => host
            .iter()
            .zip(mask.iter())
            .filter_map(|(&x, &v)| if v { Some(x) } else { None })
            .collect(),
        None => host.to_vec(),
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

    // Fast path: bare column ref into pre outputs. The pre-stage has no
    // NULL bitmap support — every cell is non-NULL, so no validity mask.
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
    // that matches what the plan declared. The materialised column may
    // contain NULLs (e.g. integer division by zero, or any operand of a
    // binary op being NULL); we surface those as a parallel validity mask
    // rather than coercing them to zero, so downstream `MIN`/`MAX`/`AVG`/
    // `COUNT(col)` get SQL semantics.
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
    let (host, validity) = from_expr_host(materialised)?;
    Ok((col_io, ResolvedHostCol::Owned { col: host, validity }))
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

/// Borrowed or owned host column, plus an optional per-row validity mask.
/// Lets the fast path return `&HostCol` from the compacted store while the
/// slow path returns a freshly-materialised value from `expr_agg` along
/// with the NULL pattern produced by host-side evaluation.
///
/// `validity`, when `Some(_)`, has length equal to the host column and
/// matches row-by-row: `mask[i] == true` means row `i` is non-NULL.
/// `None` means "all rows valid" — used for the fast path where the pre
/// stage cannot produce NULLs.
enum ResolvedHostCol<'a> {
    /// Borrowed view of a pre-stage output (no NULLs possible here).
    Borrowed(&'a HostCol),
    /// Freshly materialised slow-path column with an optional NULL mask.
    Owned {
        col: HostCol,
        /// `None` iff every row materialised non-NULL.
        validity: Option<Vec<bool>>,
    },
}

impl<'a> ResolvedHostCol<'a> {
    fn as_ref(&self) -> &HostCol {
        match self {
            ResolvedHostCol::Borrowed(c) => *c,
            ResolvedHostCol::Owned { col, .. } => col,
        }
    }

    /// Per-row validity mask, or `None` if every row is non-NULL.
    fn validity(&self) -> Option<&[bool]> {
        match self {
            ResolvedHostCol::Borrowed(_) => None,
            ResolvedHostCol::Owned { validity, .. } => validity.as_deref(),
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
/// primitive [`HostCol`] shape consumed by the GPU agg kernels, plus a
/// per-row validity mask.
///
/// **NULL handling**: SQL aggregate semantics require `SUM`/`MIN`/`MAX`/`AVG`/
/// `COUNT(col)` to ignore NULL rows. Unlike the scalar `agg_with_pre`
/// counterpart — which can simply drop NULL rows because each scalar
/// aggregate is independent — the GROUP BY path must keep the value column
/// positionally aligned with the key column. We therefore return:
///
/// - a `HostCol` where each NULL cell holds an arbitrary placeholder
///   (dtype-zero), and
/// - an `Option<Vec<bool>>` validity mask where `mask[i] == true` means
///   row `i` was non-NULL. The mask is `None` (saving an allocation and
///   a filter pass) when every row was non-NULL.
///
/// Downstream, `run_one_aggregate` filters both the keys column and the
/// value column by this mask before uploading to the GPU, so MIN/MAX/SUM/
/// AVG/COUNT(col) only see the validity-true rows.
///
/// A previous version of this function coerced NULLs to dtype-zero with no
/// mask, which was a correctness bug: `MIN([NULL, 5])` returned `0` instead
/// of `5`, `MAX([NULL, -3])` returned `0` instead of `-3`, and `AVG([NULL, 4])`
/// returned `2.0` (4/2) instead of `4.0` (4/1). See `agg_with_pre.rs` for
/// the matching scalar fix.
///
/// Bool / Utf8 materialisations are still rejected — the GPU agg kernels
/// only accept primitive numeric inputs.
fn from_expr_host(c: expr_agg::HostColumn) -> BoltResult<(HostCol, Option<Vec<bool>>)> {
    fn split<T: Copy + Default>(v: Vec<Option<T>>) -> (Vec<T>, Option<Vec<bool>>) {
        let mut any_null = false;
        let mut values: Vec<T> = Vec::with_capacity(v.len());
        let mut mask: Vec<bool> = Vec::with_capacity(v.len());
        for cell in v {
            match cell {
                Some(x) => {
                    values.push(x);
                    mask.push(true);
                }
                None => {
                    values.push(T::default());
                    mask.push(false);
                    any_null = true;
                }
            }
        }
        if any_null {
            (values, Some(mask))
        } else {
            (values, None)
        }
    }

    match c {
        expr_agg::HostColumn::I32(v) => {
            let (vals, mask) = split::<i32>(v);
            Ok((HostCol::I32(vals), mask))
        }
        expr_agg::HostColumn::I64(v) => {
            let (vals, mask) = split::<i64>(v);
            Ok((HostCol::I64(vals), mask))
        }
        expr_agg::HostColumn::F32(v) => {
            let (vals, mask) = split::<f32>(v);
            Ok((HostCol::F32(vals), mask))
        }
        expr_agg::HostColumn::F64(v) => {
            let (vals, mask) = split::<f64>(v);
            Ok((HostCol::F64(vals), mask))
        }
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

    /// Like [`HostCol::to_f64`] but drops every row where `mask[i]` is false.
    /// Used by the AVG path to filter NULL rows out of the numerator before
    /// uploading to the GPU. The result length equals the popcount of `mask`.
    fn to_f64_filtered(&self, mask: &[bool], _col_name: &str) -> BoltResult<Vec<f64>> {
        if mask.len() != self.len() {
            return Err(BoltError::Other(format!(
                "groupby_with_pre: validity mask length {} != column length {}",
                mask.len(),
                self.len()
            )));
        }
        Ok(match self {
            HostCol::I32(v) => v
                .iter()
                .zip(mask.iter())
                .filter_map(|(&x, &m)| if m { Some(x as f64) } else { None })
                .collect(),
            HostCol::I64(v) => v
                .iter()
                .zip(mask.iter())
                .filter_map(|(&x, &m)| if m { Some(x as f64) } else { None })
                .collect(),
            HostCol::F32(v) => v
                .iter()
                .zip(mask.iter())
                .filter_map(|(&x, &m)| if m { Some(x as f64) } else { None })
                .collect(),
            HostCol::F64(v) => v
                .iter()
                .zip(mask.iter())
                .filter_map(|(&x, &m)| if m { Some(x) } else { None })
                .collect(),
        })
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// These tests cover the host-only NULL-handling contract of `from_expr_host`
// (the patch point for the GROUP BY correctness bug) and the host-side
// key/value filtering performed by [`filter_keys_if_needed`] /
// [`filter_by_validity`] / [`HostCol::to_f64_filtered`]. They cannot
// exercise the GPU agg kernels directly (which require CUDA), so they
// emulate the per-group accumulation that the GPU kernel would perform on
// the validity-filtered (group, value) pairs and assert that each group's
// aggregate matches SQL semantics.
//
// The bug being fixed: a previous version of `from_expr_host` coerced
// NULLs to dtype-zero, which made `MIN([NULL, 5])` return `0` instead of
// `5`, `MAX([NULL, -3])` return `0` instead of `-3`, `AVG([NULL, 4])`
// return `2.0` instead of `4.0`, and `COUNT(col)` count NULL rows.

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Emulate the per-group SUM/MIN/MAX/COUNT that the GPU kernel would
    /// perform after we feed it the validity-filtered (key, value) pairs.
    /// Returns a deterministically sorted `(key, agg_value)` Vec.
    fn group_reduce_i32(
        keys: &[i64],
        values: &[i32],
        op: ReduceOp,
    ) -> Vec<(i64, i32)> {
        assert_eq!(keys.len(), values.len());
        let mut acc: HashMap<i64, i32> = HashMap::new();
        for (&k, &v) in keys.iter().zip(values.iter()) {
            let cell = acc.entry(k).or_insert_with(|| match op {
                ReduceOp::Sum | ReduceOp::Count => 0,
                ReduceOp::Min => i32::MAX,
                ReduceOp::Max => i32::MIN,
            });
            *cell = match op {
                ReduceOp::Sum | ReduceOp::Count => cell.wrapping_add(v),
                ReduceOp::Min => (*cell).min(v),
                ReduceOp::Max => (*cell).max(v),
            };
        }
        let mut out: Vec<(i64, i32)> = acc.into_iter().collect();
        out.sort_unstable_by_key(|(k, _)| *k);
        out
    }

    fn group_count(keys: &[i64]) -> Vec<(i64, i64)> {
        let mut acc: HashMap<i64, i64> = HashMap::new();
        for &k in keys {
            *acc.entry(k).or_insert(0) += 1;
        }
        let mut out: Vec<(i64, i64)> = acc.into_iter().collect();
        out.sort_unstable_by_key(|(k, _)| *k);
        out
    }

    fn group_sum_f64(keys: &[i64], values: &[f64]) -> Vec<(i64, f64)> {
        assert_eq!(keys.len(), values.len());
        let mut acc: HashMap<i64, f64> = HashMap::new();
        for (&k, &v) in keys.iter().zip(values.iter()) {
            *acc.entry(k).or_insert(0.0) += v;
        }
        let mut out: Vec<(i64, f64)> = acc.into_iter().collect();
        out.sort_unstable_by_key(|(k, _)| *k);
        out
    }

    /// `from_expr_host` returns a parallel validity mask and dtype-zero
    /// placeholder for NULL cells; no mask is returned when every cell
    /// is non-NULL (small optimisation).
    #[test]
    fn from_expr_host_returns_validity_mask() {
        let col = expr_agg::HostColumn::I32(vec![None, Some(5), Some(7), None]);
        let (host, validity) = from_expr_host(col).expect("ok");
        match host {
            HostCol::I32(v) => assert_eq!(v.len(), 4),
            _ => panic!("expected I32"),
        }
        let mask = validity.expect("validity mask should be present when NULLs exist");
        assert_eq!(mask, vec![false, true, true, false]);
    }

    /// When the input has no NULLs the validity mask is `None` (so the
    /// downstream filter passes are skipped entirely).
    #[test]
    fn from_expr_host_omits_mask_when_all_valid() {
        let col = expr_agg::HostColumn::I32(vec![Some(1), Some(2), Some(3)]);
        let (host, validity) = from_expr_host(col).expect("ok");
        assert!(validity.is_none(), "no NULLs => no mask");
        match host {
            HostCol::I32(v) => assert_eq!(v, vec![1, 2, 3]),
            _ => panic!("expected I32"),
        }
    }

    /// MIN per group ignores NULL rows: `MIN([NULL, 5]) = 5`, not `0`. Pre-fix
    /// the NULL-zeroed column produced `MIN(group=0) = 0`.
    #[test]
    fn group_min_ignores_nulls() {
        // Two groups (1, 2). Group 1 has values [NULL, 5]; group 2 has [10].
        let keys = vec![1i64, 1, 2];
        let values_col = expr_agg::HostColumn::I32(vec![None, Some(5), Some(10)]);
        let (host, validity) = from_expr_host(values_col).expect("ok");
        let mask = validity.as_deref();
        let values: Vec<i32> = filter_by_validity(
            match &host {
                HostCol::I32(v) => v,
                _ => panic!("expected I32"),
            },
            mask,
        );
        let filtered_keys: Vec<i64> = match mask {
            Some(m) => keys.iter().zip(m.iter()).filter_map(|(&k, &v)| if v { Some(k) } else { None }).collect(),
            None => keys.clone(),
        };
        let result = group_reduce_i32(&filtered_keys, &values, ReduceOp::Min);
        assert_eq!(result, vec![(1, 5), (2, 10)]);
    }

    /// MAX per group ignores NULL rows: `MAX([NULL, -3]) = -3`, not `0`.
    /// Pre-fix the NULL-zeroed column made `max(0, -3) == 0`.
    #[test]
    fn group_max_ignores_nulls() {
        // Group 1: [NULL, -3]; group 2: [-5, NULL].
        let keys = vec![1i64, 1, 2, 2];
        let values_col = expr_agg::HostColumn::I32(vec![None, Some(-3), Some(-5), None]);
        let (host, validity) = from_expr_host(values_col).expect("ok");
        let mask = validity.as_deref();
        let values: Vec<i32> = filter_by_validity(
            match &host {
                HostCol::I32(v) => v,
                _ => panic!("expected I32"),
            },
            mask,
        );
        let filtered_keys: Vec<i64> = match mask {
            Some(m) => keys.iter().zip(m.iter()).filter_map(|(&k, &v)| if v { Some(k) } else { None }).collect(),
            None => keys.clone(),
        };
        let result = group_reduce_i32(&filtered_keys, &values, ReduceOp::Max);
        assert_eq!(result, vec![(1, -3), (2, -5)]);
    }

    /// AVG per group: denominator must exclude NULL rows.
    /// Group 1 has [NULL, 4.0] -> AVG = 4.0 (not 2.0).
    /// Group 2 has [3.0, 5.0] -> AVG = 4.0.
    #[test]
    fn group_avg_excludes_nulls_from_denominator() {
        let keys = vec![1i64, 1, 2, 2];
        let values_col = expr_agg::HostColumn::F64(vec![None, Some(4.0), Some(3.0), Some(5.0)]);
        let (host, validity) = from_expr_host(values_col).expect("ok");
        let mask = validity.as_deref();
        // Slow-path AVG materialises at f64 and uses `to_f64_filtered`.
        let values_f64 = match (mask, &host) {
            (Some(m), HostCol::F64(v)) => host_to_f64_filtered_test(v, m),
            (None, HostCol::F64(v)) => v.clone(),
            _ => panic!("expected F64"),
        };
        let filtered_keys: Vec<i64> = match mask {
            Some(m) => keys.iter().zip(m.iter()).filter_map(|(&k, &v)| if v { Some(k) } else { None }).collect(),
            None => keys.clone(),
        };
        let sums = group_sum_f64(&filtered_keys, &values_f64);
        let counts = group_count(&filtered_keys);
        // Combine sum/count per group and compute AVG.
        let mut averages: Vec<(i64, f64)> = sums
            .iter()
            .zip(counts.iter())
            .map(|(&(k1, s), &(k2, c))| {
                assert_eq!(k1, k2);
                let avg = if c == 0 { 0.0 } else { s / (c as f64) };
                (k1, avg)
            })
            .collect();
        averages.sort_unstable_by_key(|(k, _)| *k);
        assert_eq!(averages.len(), 2);
        assert_eq!(averages[0].0, 1);
        assert!((averages[0].1 - 4.0).abs() < 1e-12, "group 1 AVG was {}", averages[0].1);
        assert_eq!(averages[1].0, 2);
        assert!((averages[1].1 - 4.0).abs() < 1e-12, "group 2 AVG was {}", averages[1].1);
    }

    /// Local copy of the `HostCol::to_f64_filtered` logic in test scope so
    /// we can run it on a raw `&[f64]` borrowed from a destructured HostCol
    /// match without colliding with the `to_f64_filtered` method's
    /// `&self` borrow (the test is short, this keeps the assertions
    /// readable).
    fn host_to_f64_filtered_test(v: &[f64], mask: &[bool]) -> Vec<f64> {
        v.iter()
            .zip(mask.iter())
            .filter_map(|(&x, &m)| if m { Some(x) } else { None })
            .collect()
    }

    /// COUNT(col) excludes NULL rows per group.
    /// Group 1 has [NULL, 5, NULL, 7] -> COUNT = 2.
    /// Group 2 has [10] -> COUNT = 1.
    #[test]
    fn group_count_col_excludes_nulls() {
        let keys = vec![1i64, 1, 1, 1, 2];
        let values_col =
            expr_agg::HostColumn::I32(vec![None, Some(5), None, Some(7), Some(10)]);
        let (_host, validity) = from_expr_host(values_col).expect("ok");
        let mask = validity.as_deref();
        let filtered_keys: Vec<i64> = match mask {
            Some(m) => keys.iter().zip(m.iter()).filter_map(|(&k, &v)| if v { Some(k) } else { None }).collect(),
            None => keys.clone(),
        };
        let counts = group_count(&filtered_keys);
        assert_eq!(counts, vec![(1, 2), (2, 1)]);
    }

    /// SUM per group was "accidentally correct" under the old NULL-as-zero
    /// behavior (0 is the SUM identity). Verify it stays correct under the
    /// new filter-then-reduce pipeline. Group 1: [NULL, 5, 7] -> SUM = 12.
    /// Group 2: [3] -> SUM = 3.
    #[test]
    fn group_sum_matches_non_null_sum() {
        let keys = vec![1i64, 1, 1, 2];
        let values_col = expr_agg::HostColumn::I32(vec![None, Some(5), Some(7), Some(3)]);
        let (host, validity) = from_expr_host(values_col).expect("ok");
        let mask = validity.as_deref();
        let values: Vec<i32> = filter_by_validity(
            match &host {
                HostCol::I32(v) => v,
                _ => panic!("expected I32"),
            },
            mask,
        );
        let filtered_keys: Vec<i64> = match mask {
            Some(m) => keys.iter().zip(m.iter()).filter_map(|(&k, &v)| if v { Some(k) } else { None }).collect(),
            None => keys.clone(),
        };
        let sums = group_reduce_i32(&filtered_keys, &values, ReduceOp::Sum);
        assert_eq!(sums, vec![(1, 12), (2, 3)]);
    }

    /// All-NULL aggregate row: a group whose every value is NULL contributes
    /// no rows to the reduction (`non_null == 0` for that group). The GROUP BY
    /// pipeline then leaves the SUM accumulator at its identity (0) and the
    /// COUNT accumulator at 0, so the output row for the group is
    /// `(key, 0)` for SUM/COUNT and `0.0` for AVG (per the existing
    /// `c == 0 ? 0.0 : s/c` guard in `build_agg_array`). SQL says these
    /// should be NULL, but the existing non-nullable output path cannot
    /// express NULL — same caveat documented in `agg_with_pre.rs`.
    #[test]
    fn group_all_null_yields_zero_count_and_no_contribution() {
        let keys = vec![1i64, 1, 2, 2];
        // Group 1 is all-NULL; group 2 has one non-NULL value.
        let values_col = expr_agg::HostColumn::I32(vec![None, None, Some(4), None]);
        let (host, validity) = from_expr_host(values_col).expect("ok");
        let mask = validity.as_deref();
        let values: Vec<i32> = filter_by_validity(
            match &host {
                HostCol::I32(v) => v,
                _ => panic!("expected I32"),
            },
            mask,
        );
        let filtered_keys: Vec<i64> = match mask {
            Some(m) => keys.iter().zip(m.iter()).filter_map(|(&k, &v)| if v { Some(k) } else { None }).collect(),
            None => keys.clone(),
        };
        // Filtered down to just group 2's single row.
        assert_eq!(filtered_keys, vec![2]);
        assert_eq!(values, vec![4]);
        let counts = group_count(&filtered_keys);
        // Only group 2 has a non-NULL row; group 1 contributes nothing
        // to the reduction (its keys-table slot keeps the COUNT identity 0).
        assert_eq!(counts, vec![(2, 1)]);
        let mins = group_reduce_i32(&filtered_keys, &values, ReduceOp::Min);
        assert_eq!(mins, vec![(2, 4)]);
    }

    /// `filter_keys_if_needed` filters host keys in lockstep with the
    /// validity mask, and returns `(None, n_rows)` (i.e. no allocation,
    /// reuse the shared key column) when the resolved column has no NULLs.
    #[test]
    fn filter_keys_skips_allocation_when_all_valid() {
        let keys = vec![1i64, 2, 3];
        let resolved = ResolvedHostCol::Owned {
            col: HostCol::I32(vec![10, 20, 30]),
            validity: None,
        };
        let (gpu, n) = filter_keys_if_needed(&keys, &resolved).expect("ok");
        assert!(gpu.is_none(), "no validity => no fresh GpuVec");
        assert_eq!(n, 3);
    }

    /// Bool / Utf8 are still rejected — the primitive reduction kernels
    /// only accept numeric inputs.
    #[test]
    fn from_expr_host_rejects_bool_and_utf8() {
        let bool_col = expr_agg::HostColumn::Bool(vec![Some(true)]);
        assert!(matches!(from_expr_host(bool_col), Err(BoltError::Type(_))));
        let utf8_col = expr_agg::HostColumn::Utf8(vec![Some("x".to_string())]);
        assert!(matches!(from_expr_host(utf8_col), Err(BoltError::Type(_))));
    }
}

#[cfg(test)]
mod null_handling_tests {
    //! Host-only tests for the `PreCol::upload` NULL-rejection gate added in
    //! the pre-stage: Arrow arrays whose `null_count() > 0` are rejected
    //! before any device allocation because the pre-stage GPU kernels do not
    //! yet carry a validity bitmap and would otherwise multiply garbage in
    //! expressions like `price * tax`.
    //!
    //! No CUDA touched: the error path short-circuits before any device
    //! allocation.
    use super::*;

    #[test]
    fn pre_col_upload_rejects_null_bearing_input() {
        let arr = Float64Array::from(vec![Some(1.0f64), None, Some(3.0f64)]);
        assert!(arr.null_count() > 0, "test fixture should carry a NULL");
        // `PreCol` doesn't implement `Debug`, so we match the Result manually
        // rather than calling `expect_err`.
        let msg = match PreCol::upload(&arr, DataType::Float64) {
            Ok(_) => panic!("PreCol::upload should reject NULL-bearing input"),
            Err(e) => format!("{e}"),
        };
        assert!(msg.contains("NULL"), "error should mention NULL: {msg}");
        assert!(
            msg.contains("pre-stage") || msg.contains("validity"),
            "error should explain why, got: {msg}"
        );
    }
}
