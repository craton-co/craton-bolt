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
//!
//! ## PV-stage-d: host-strip fallback vs native-validity kernels
//!
//! The pre-stage host-strip path below (predicate-only kernel + host-side
//! `mask`-driven `Vec::compact`) is **kept** as the fallback for cases the
//! GPU native-validity kernels can't (yet) handle. Specifically, the
//! pre-kernel itself doesn't yet consume a packed-bit validity bitmap —
//! when stage E lands that, this executor will branch:
//!
//! * If `kernel.input_has_validity[i] == true` for any input AND the
//!   kernel shape is one we have a native-validity emitter for, upload
//!   `pack_validity_bits(...)` (see [`crate::jit::valid_flag_kernels`])
//!   and dispatch to the `_with_validity` variant — no host strip.
//! * Otherwise, fall back to the existing predicate + compact path
//!   below. This handles validity-bearing wide keys, Utf8, and any
//!   shape we haven't taught the codegen yet.
//!
//! The fallback is correctness-safe at the cost of an extra device-to-host
//! roundtrip on the pre outputs; the per-aggregate hash-table launches
//! still run on the GPU.

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
    compile_groupby_agg_kernel, compile_groupby_agg_kernel_with_validity,
    compile_groupby_keys_kernel, compile_groupby_keys_kernel_with_validity,
    groupby_block_size, packed_validity_word_count, AGG_KERNEL_ENTRY,
    KEYS_KERNEL_ENTRY,
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

    // Option-A safety net (Stage B partial): a NULL group-key row would
    // hash to a "NULL group" that the current open-addressing keys kernel
    // cannot represent (it has only the `i64::MIN` empty sentinel + real
    // key slots; NULL is a separate semantic group in SQL). Stage C
    // should add an explicit NULL slot, or pre-strip NULL-keyed rows
    // alongside their aggregate inputs. For now we reject with a clear
    // message rather than silently accumulate NULL-keyed rows into
    // whichever real group happens to collide.
    if key_host.validity.is_some() {
        return Err(BoltError::Other(format!(
            "groupby_with_pre: GROUP BY key column '{}' carries a NULL validity \
             bitmap, but the GPU group-by keys kernel does not yet represent the \
             NULL group. NULL keys are a Stage C follow-up to Option B.",
            key_io.name
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
    // Key column already had its NULL rows rejected upstream (see the
    // Stage-B key-validity gate above), so we pass `None` here — the
    // classic 4-param keys kernel is sufficient. Stage C lifting for
    // NULL keys (per-row, on the device) is a follow-up; today the
    // host-side reject keeps semantics correct.
    launch_keys_kernel(
        &key_col_gpu,
        &mut keys_table,
        None,
        n_compacted,
        k_u32,
        &stream,
    )?;

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

    // -- (Option B) Detect input validity, plumb through KernelSpec.
    //    See `agg_with_pre::run_pre_stage` for the design rationale.
    let input_has_validity: Vec<bool> =
        input_cols.iter().map(|c| c.has_validity()).collect();
    let any_input_validity: bool = input_has_validity.iter().any(|b| *b);
    let output_has_validity: Vec<bool> = if any_input_validity {
        vec![true; spec.outputs.len()]
    } else {
        vec![false; spec.outputs.len()]
    };

    // -- Allocate zero-initialised output buffers (with validity if needed).
    let mut output_cols: Vec<PreCol> = Vec::with_capacity(spec.outputs.len());
    for io in &spec.outputs {
        output_cols.push(PreCol::alloc_zeros(io.dtype, n_rows, any_input_validity)?);
    }

    // -- JIT-compile the projection PTX. Validity flags drive the
    //    additional `*u8` params + AND-store emission in `ptx_gen`.
    let pre_spec_for_ptx = KernelSpec {
        input_has_validity: input_has_validity.clone(),
        output_has_validity: output_has_validity.clone(),
        ..spec.clone()
    };
    let ptx = compile_ptx(&pre_spec_for_ptx, PRE_KERNEL_ENTRY)?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(PRE_KERNEL_ENTRY)?;

    // -- Assemble kernel parameters: inputs..., outputs...,
    //    [input_validity..., output_validity...,] n_rows u32. Order
    //    matches `ptx_gen::compile`'s param walk.
    let mut device_ptrs: Vec<CUdeviceptr> =
        Vec::with_capacity(input_cols.len() + output_cols.len() + input_cols.len() + output_cols.len());
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

/// Launch the keys-only kernel. When `key_validity` is `Some(_)` the
/// Stage C validity-aware kernel variant is used: NULL-keyed rows skip
/// the insert on the device side, matching SQL semantics (NULL keys form
/// no group). When `None` the classic 4-param kernel is used.
fn launch_keys_kernel(
    group_col: &GpuVec<i64>,
    keys_table: &mut GpuVec<i64>,
    key_validity: Option<&GpuVec<u32>>,
    n_rows: usize,
    k_u32: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    if n_rows == 0 {
        // Nothing to insert; the empty keys table is already correct.
        return Ok(());
    }

    let ptx = if key_validity.is_some() {
        compile_groupby_keys_kernel_with_validity()?
    } else {
        compile_groupby_keys_kernel()?
    };
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(KEYS_KERNEL_ENTRY)?;

    let mut group_ptr: CUdeviceptr = group_col.device_ptr();
    let mut keys_ptr: CUdeviceptr = keys_table.device_ptr();
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let mut k_param: u32 = k_u32;

    let block = groupby_block_size();
    let grid_x = grid_x_for(n_rows_u32, block);

    // SAFETY: `function` is borrowed from a live `CudaModule`; every entry of
    // `params` points to a stack local that outlives the synchronize.
    if let Some(vp) = key_validity {
        let mut validity_ptr: CUdeviceptr = vp.device_ptr();
        let mut params: [*mut c_void; 5] = [
            &mut group_ptr as *mut CUdeviceptr as *mut c_void,
            &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
            &mut n_rows_u32 as *mut u32 as *mut c_void,
            &mut k_param as *mut u32 as *mut c_void,
            &mut validity_ptr as *mut CUdeviceptr as *mut c_void,
        ];
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
        let _ = validity_ptr;
    } else {
        let mut params: [*mut c_void; 4] = [
            &mut group_ptr as *mut CUdeviceptr as *mut c_void,
            &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
            &mut n_rows_u32 as *mut u32 as *mut c_void,
            &mut k_param as *mut u32 as *mut c_void,
        ];
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
    launch_agg_kernel_inner(
        op, input_dtype, group_col, keys_table, input_col, acc_table, None, n_rows,
        k_u32, stream,
    )
}

/// Stage C: launch the validity-aware variant of the agg kernel. `value_validity`
/// is a packed-bit `*u32` device pointer; rows whose bit is `0` are skipped
/// on the device (no atomic update issued). Length must equal
/// [`packed_validity_word_count`]`(n_rows)`.
///
/// Currently unused — the production path keeps the host-strip-in-lockstep
/// fallback (see `run_typed_agg`). The wrapper exists so future
/// performance work can flip individual call sites to the native GPU
/// path without re-deriving the kernel-launch boilerplate.
#[allow(dead_code)]
fn launch_agg_kernel_with_validity<T: Pod>(
    op: ReduceOp,
    input_dtype: DataType,
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    input_col: &GpuVec<T>,
    acc_table: &mut GpuVec<T>,
    value_validity: &GpuVec<u32>,
    n_rows: usize,
    k_u32: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    launch_agg_kernel_inner(
        op,
        input_dtype,
        group_col,
        keys_table,
        input_col,
        acc_table,
        Some(value_validity),
        n_rows,
        k_u32,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_agg_kernel_inner<T: Pod>(
    op: ReduceOp,
    input_dtype: DataType,
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    input_col: &GpuVec<T>,
    acc_table: &mut GpuVec<T>,
    value_validity: Option<&GpuVec<u32>>,
    n_rows: usize,
    k_u32: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    if n_rows == 0 {
        return Ok(());
    }

    let ptx = if value_validity.is_some() {
        compile_groupby_agg_kernel_with_validity(op, input_dtype)?
    } else {
        compile_groupby_agg_kernel(op, input_dtype)?
    };
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(AGG_KERNEL_ENTRY)?;

    let mut group_ptr: CUdeviceptr = group_col.device_ptr();
    let mut keys_ptr: CUdeviceptr = keys_table.device_ptr();
    let mut input_ptr: CUdeviceptr = input_col.device_ptr();
    let mut acc_ptr: CUdeviceptr = acc_table.device_ptr();
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let mut k_param: u32 = k_u32;

    // The classic ABI is 6 params (group, keys, input, acc, n_rows, k); the
    // Stage C validity ABI adds a 7th trailing `*u64` packed-bit pointer.
    let validity_ptr_opt: Option<CUdeviceptr> = value_validity.map(|v| v.device_ptr());

    let block = groupby_block_size();
    let grid_x = grid_x_for(n_rows_u32, block);

    if let Some(mut validity_ptr) = validity_ptr_opt {
        let mut params: [*mut c_void; 7] = [
            &mut group_ptr as *mut CUdeviceptr as *mut c_void,
            &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
            &mut input_ptr as *mut CUdeviceptr as *mut c_void,
            &mut acc_ptr as *mut CUdeviceptr as *mut c_void,
            &mut n_rows_u32 as *mut u32 as *mut c_void,
            &mut k_param as *mut u32 as *mut c_void,
            &mut validity_ptr as *mut CUdeviceptr as *mut c_void,
        ];
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
        let _ = validity_ptr;
    } else {
        let mut params: [*mut c_void; 6] = [
            &mut group_ptr as *mut CUdeviceptr as *mut c_void,
            &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
            &mut input_ptr as *mut CUdeviceptr as *mut c_void,
            &mut acc_ptr as *mut CUdeviceptr as *mut c_void,
            &mut n_rows_u32 as *mut u32 as *mut c_void,
            &mut k_param as *mut u32 as *mut c_void,
        ];
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
    }
    stream.synchronize()?;
    let _ = (group_ptr, keys_ptr, input_ptr, acc_ptr);
    Ok(())
}

/// Stage C: pack a `Vec<u8>` validity stream (`1` = valid, `0` = null,
/// one byte per row, Arrow-style row-major) into a packed-bit `Vec<u32>`
/// suitable for upload to the GPU. The layout matches the kernel's
/// expectations (see [`crate::jit::hash_kernels`] module-level docs):
/// bit `tid % 32` of word `tid / 32` describes row `tid`, with
/// little-endian bit order inside each word (bit 0 = first row of the
/// chunk).
///
/// The output length is [`packed_validity_word_count`]`(n_rows)`. Bits
/// past `n_rows` are zero-padded; the kernel never reads them because
/// `tid >= n_rows` short-circuits earlier.
///
/// Currently unused by the production path (the host-strip fallback
/// avoids the per-row pack); exposed for when the native-validity
/// kernel variant becomes the default.
#[allow(dead_code)]
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

    // Stage C: the Stage-B reject error is lifted. NULL value rows are
    // dropped in lockstep with the GROUP BY key column via
    // `filter_keys_if_needed` (above) + `filter_by_validity` (per
    // dtype, below). Both filters consume the same `validity` mask so
    // (keys, values) stay parallel at the kernel call site, and the
    // GPU agg kernel never sees a NULL contribution. A follow-up can
    // switch to the native-validity kernel variant
    // (`compile_groupby_agg_kernel_with_validity` +
    // `launch_agg_kernel_with_validity`, both already wired up) to
    // avoid the host-side strip on hot paths; for correctness the
    // host strip is sufficient and matches SQL aggregate semantics.

    match (col_io.dtype, &host_col.values) {
        (DataType::Int32, HostColValues::I32(host)) => {
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
        (DataType::Int64, HostColValues::I64(host)) => {
            // Stage C: filter NULL rows in lockstep with the key column.
            // Previously this branch uploaded the raw values buffer
            // verbatim (would have summed NULL slots as 0); the
            // Stage-B reject guarded against that. Now that the gate
            // is lifted we must apply the validity mask explicitly.
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
        (DataType::Float32, HostColValues::F32(host)) => {
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
        (DataType::Float64, HostColValues::F64(host)) => {
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
    let host = from_expr_host(materialised)?;
    // PV (Option B): `host.validity` is `Option<Vec<u8>>` (`1` = valid).
    // `ResolvedHostCol::Owned::validity` is `Option<Vec<bool>>`
    // (`true` = valid). Translate. `None` (no NULLs) stays `None` so
    // downstream `filter_keys_if_needed` short-circuits.
    let validity = host
        .validity
        .as_ref()
        .map(|v| v.iter().map(|b| *b != 0).collect::<Vec<bool>>());
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
/// [`expr_agg::HostColumn`] shape consumed by the host-side evaluator.
///
/// **NULL handling (Option B)**: when the source `HostCol` carries a
/// validity bitmap, NULL rows surface as `None` so the evaluator's 3VL
/// machinery (see `expr_agg`) propagates them. When validity is `None`
/// every cell is `Some(_)`.
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
/// primitive [`HostCol`] shape. NULLs are **preserved** as a `validity`
/// bitmap (Option B): downstream `run_typed_agg` rejects validity-bearing
/// inputs with a clear error (Stage C follow-up). Bool / Utf8
/// materialisations are rejected — the GPU agg kernels only accept
/// primitive numeric inputs.
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
///
/// **NULL handling (Option B)**: identical contract to
/// [`crate::exec::agg_with_pre::PreCol`] — see that doc for the rationale.
/// Each `PreCol` may carry a parallel `valid_mask: GpuVec<u8>` (`1` =
/// valid, `0` = null) that rides alongside the value buffer.
struct PreCol {
    values: PreColValues,
    valid_mask: Option<GpuVec<u8>>,
}

enum PreColValues {
    I32(GpuVec<i32>),
    I64(GpuVec<i64>),
    F32(GpuVec<f32>),
    F64(GpuVec<f64>),
}

impl PreCol {
    /// Upload an Arrow array to the GPU, downcasting per `dtype`. Mirrors
    /// `agg_with_pre::PreCol::upload`; see that for the NULL-handling
    /// contract (Option B: validity bitmap propagates to the GPU
    /// alongside the value buffer).
    fn upload(arr: &dyn Array, dtype: DataType) -> BoltResult<Self> {
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
            DataType::Bool | DataType::Utf8 => {
                return Err(BoltError::Type(format!(
                    "groupby_with_pre: pre kernel column dtype {:?} not supported",
                    dtype
                )))
            }
        };
        Ok(PreCol { values, valid_mask })
    }

    /// Allocate a zero-initialised device column of `n` rows.
    fn alloc_zeros(dtype: DataType, n: usize, with_validity: bool) -> BoltResult<Self> {
        let values = match dtype {
            DataType::Int32 => PreColValues::I32(GpuVec::<i32>::zeros(n)?),
            DataType::Int64 => PreColValues::I64(GpuVec::<i64>::zeros(n)?),
            DataType::Float32 => PreColValues::F32(GpuVec::<f32>::zeros(n)?),
            DataType::Float64 => PreColValues::F64(GpuVec::<f64>::zeros(n)?),
            DataType::Bool | DataType::Utf8 => {
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

/// Host-side typed column produced by downloading a `PreCol`. Mirrors
/// `agg_with_pre::HostCol`. Carries a parallel `validity` vector when
/// any input row was NULL.
struct HostCol {
    values: HostColValues,
    /// Per-row validity. `None` => all rows valid.
    validity: Option<Vec<u8>>,
}

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
    /// Validity (if any) is compacted in lockstep.
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
        };
        let validity = validity.map(|v| filter_vec(v, mask));
        Ok(HostCol { values, validity })
    }

    /// Strip NULL rows according to the validity bitmap. No-op when
    /// validity is already `None`. Currently only exercised by the
    /// `null_propagation_tests` module — production code (`run_typed_agg`)
    /// rejects validity-bearing inputs outright until Stage C lifts the
    /// gate by stripping in lockstep with the GROUP BY key column.
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

    /// Convert the column to a `Vec<i64>` for use as a GROUP BY key. Int32
    /// upcasts; everything else is an error.
    fn to_i64_for_key(&self, col_name: &str) -> BoltResult<Vec<i64>> {
        match &self.values {
            HostColValues::I32(v) => Ok(v.iter().map(|&x| x as i64).collect()),
            HostColValues::I64(v) => Ok(v.clone()),
            HostColValues::F32(_) | HostColValues::F64(_) => Err(BoltError::Type(format!(
                "float GROUP BY keys not yet supported (column '{}')",
                col_name
            ))),
        }
    }

    /// Convert the column to a `Vec<f64>` for use as an AVG input. Any
    /// numeric variant upcasts. `col_name` is only used to enrich a future
    /// error message if we ever extend this to reject non-numeric variants.
    fn to_f64(&self, _col_name: &str) -> BoltResult<Vec<f64>> {
        match &self.values {
            HostColValues::I32(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            HostColValues::I64(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            HostColValues::F32(v) => Ok(v.iter().map(|&x| x as f64).collect()),
            HostColValues::F64(v) => Ok(v.clone()),
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
        Ok(match &self.values {
            HostColValues::I32(v) => v
                .iter()
                .zip(mask.iter())
                .filter_map(|(&x, &m)| if m { Some(x as f64) } else { None })
                .collect(),
            HostColValues::I64(v) => v
                .iter()
                .zip(mask.iter())
                .filter_map(|(&x, &m)| if m { Some(x as f64) } else { None })
                .collect(),
            HostColValues::F32(v) => v
                .iter()
                .zip(mask.iter())
                .filter_map(|(&x, &m)| if m { Some(x as f64) } else { None })
                .collect(),
            HostColValues::F64(v) => v
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
        (&col.values, dtype),
        (HostColValues::I32(_), DataType::Int32)
            | (HostColValues::I64(_), DataType::Int64)
            | (HostColValues::F32(_), DataType::Float32)
            | (HostColValues::F64(_), DataType::Float64)
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
mod null_propagation_tests {
    //! Host-only tests for the Option B NULL-propagation gates added to
    //! the groupby-with-pre stage. The GROUP BY GPU path itself (keys
    //! kernel + agg kernel) does NOT yet consume validity (a Stage C
    //! follow-up), so the in-module gates ERROR rather than risk
    //! silently aggregating NULL-keyed rows into the wrong group or
    //! summing garbage.
    use super::*;

    // ---- HostCol mirrors agg_with_pre semantics -----------------------------

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

    // ---- Arrow NULL-bearing arrays now flow through `PreCol::upload` ------
    //
    // PreCol::upload requires CUDA so we don't invoke it here. We test the
    // observable invariant: an Arrow NULL-bearing array can be queried for
    // a host-side validity Vec<u8> that matches what `upload` constructs.

    #[test]
    fn arrow_null_bearing_int64_array_extracts_validity_bytes() {
        let arr = Int64Array::from(vec![Some(1i64), None, Some(3i64), None, Some(5i64)]);
        assert_eq!(arr.null_count(), 2);
        let bytes: Vec<u8> = (0..arr.len())
            .map(|i| if arr.is_null(i) { 0 } else { 1 })
            .collect();
        assert_eq!(bytes, vec![1, 0, 1, 0, 1]);
    }

    // ---- Stage C: packed-bit validity conversion --------------------------

    /// Small byte-stream packs into a single u32 with little-endian bit order
    /// (bit 0 = first row). This matches Arrow's null-buffer convention and
    /// the kernel's `bfe.u32` extraction.
    #[test]
    fn pack_validity_bits_small_little_endian() {
        // 5 rows: 1, 0, 1, 0, 1 → bits 0, 2, 4 set → 0b10101 = 0x15
        let bytes = vec![1u8, 0, 1, 0, 1];
        let packed = pack_validity_bits(&bytes);
        assert_eq!(packed.len(), 1, "5 rows fit in one u32 word");
        assert_eq!(packed[0], 0b10101, "bit 0 = row 0; bit 4 = row 4");
    }

    /// 32-row boundary: word 0 holds rows 0..32, word 1 holds rows 32..
    #[test]
    fn pack_validity_bits_crosses_word_boundary() {
        // Row 0..32 are all valid (word 0 = 0xFFFF_FFFF); row 32 is valid,
        // row 33 is null, row 34 is valid (word 1 = 0b101 = 0x5).
        let mut bytes = vec![1u8; 35];
        bytes[33] = 0;
        let packed = pack_validity_bits(&bytes);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], 0xFFFF_FFFFu32);
        assert_eq!(packed[1], 0b101);
    }

    /// Round-trip: pack then unpack via the same bit-arithmetic the kernel
    /// uses on the device. Catches off-by-one and endianness regressions.
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

    /// Empty input still produces one word of padding so the kernel can
    /// load word 0 unconditionally (its `tid >= n_rows` guard ensures the
    /// bit is never observed).
    #[test]
    fn pack_validity_bits_empty_pads_to_one_word() {
        let packed = pack_validity_bits(&[]);
        assert_eq!(packed, vec![0u32]);
    }
}
