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

// dedup (groupby_common): `HashSet` was used only by the now-relocated
// `unique_count`; the shared copy lives in `crate::exec::groupby_common`.
use std::collections::HashMap;
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
use crate::exec::launch::{grid_x_for, CudaStream};
use crate::exec::module_cache;
use crate::exec::n_rows_to_u32;
use crate::jit::agg_kernels::ReduceOp;
use crate::jit::hash_kernels::{
    compile_groupby_agg_kernel, compile_groupby_agg_kernel_with_validity,
    compile_groupby_keys_kernel, compile_groupby_keys_kernel_with_validity,
    groupby_block_size, packed_validity_word_count, AGG_KERNEL_ENTRY,
    I64_EMPTY_SENTINEL, KEYS_KERNEL_ENTRY,
};
use crate::jit::compile_ptx;
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Field, Schema};
use crate::plan::physical_plan::{AggregateSpec, ColumnIO, KernelSpec, PhysicalPlan};

/// PTX entry-point name for the pre-projection kernel.
const PRE_KERNEL_ENTRY: &str = "bolt_pre_kernel";

/// PTX entry-point name for the predicate-only kernel that materialises the mask.
const PRE_PREDICATE_ENTRY: &str = "bolt_pre_predicate";

/// Threads per block for the pre-projection / predicate launches.
const PRE_BLOCK_SIZE: u32 = 256;

/// Empty-slot sentinel; mirrors the literal baked into the keys kernel.
/// Re-export of [`I64_EMPTY_SENTINEL`] under the legacy local name to keep
/// existing call sites in this module unchanged. Review C7 centralises the
/// value in [`crate::jit::hash_kernels`] so dispatchers + PTX stay in sync.
const EMPTY_KEY: i64 = I64_EMPTY_SENTINEL;

/// Reserved encoded-key sentinel for the synthesised SQL NULL group. Mirrors
/// `crate::exec::groupby::NULL_GROUP_KEY`. All NULL-keyed rows are remapped to
/// this single value so they hash to one hash-table slot; `build_key_array`
/// emits a NULL (not a decoded value) for that slot. `i64::MAX` is safe: a
/// genuine single-column Int32/Int64 key only reaches it if it literally holds
/// `i64::MAX`, which we detect and reject (no NULL-group synthesis there).
const NULL_GROUP_KEY: i64 = i64::MAX;

/// PV-stage-e: observability counter — increments once per agg launch that
/// successfully dispatches through the native `_with_validity` GPU kernel
/// path (i.e. uploads a packed-bit validity bitmap and skips the host strip).
///
/// Used by inline `#[cfg(test)]` tests to assert that the planner-time
/// `KernelSpec::input_has_validity` signal actually drives runtime
/// dispatch, rather than silently falling through to the host-strip
/// fallback. Production code does not read this counter — it's purely
/// for test observability.
#[doc(hidden)]
pub static NATIVE_VALIDITY_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// PV-stage-e: dispatch predicate — should this `(op, dtype, validity, planner_flag)`
/// combination route through the native `_with_validity` kernel variant
/// instead of the host-strip fallback?
///
/// Three conditions must hold:
///
/// 1. The planner signalled that at least one pre-stage input may carry
///    validity (`any_input_has_validity == true`). When the planner is
///    confident no input has nulls (e.g. `TableProvider::has_nulls` returned
///    false for every column), the native dispatch is skipped — the host
///    strip is a no-op in that case anyway, so the test serves as a
///    correctness gate rather than a perf bypass.
///
/// 2. The resolved host column actually carries a validity mask. If the
///    pre stage didn't produce one (e.g. all inputs are non-null in this
///    batch), there's no bitmap to upload and the classic kernel is
///    bit-for-bit equivalent.
///
/// 3. The `(op, dtype)` combination has a native validity-aware emitter.
///    `compile_groupby_agg_kernel_with_validity` covers integer SUM/MIN/MAX
///    and float SUM; float MIN/MAX still routes through `float_atomics`
///    (no validity variant there yet — Stage F follow-up). COUNT goes
///    through a synthetic-ones path that's already correct, so it's not
///    eligible for native validity dispatch here.
fn dispatch_native_validity(
    any_input_has_validity: bool,
    validity: Option<&[bool]>,
    op: ReduceOp,
    dtype: DataType,
) -> bool {
    if !any_input_has_validity {
        return false;
    }
    let Some(mask) = validity else {
        return false;
    };
    // No nulls in this batch even though the planner flagged validity?
    // Skip the native path — classic kernel is equivalent.
    if mask.iter().all(|&v| v) {
        return false;
    }
    // Native `_with_validity` kernel coverage: integer SUM/MIN/MAX and
    // float SUM. Float MIN/MAX is routed elsewhere (float_atomics) and
    // has no `_with_validity` companion yet — fall through to host-strip.
    // Bool/Utf8 are rejected by the kernel itself, so don't dispatch.
    match (op, dtype) {
        (ReduceOp::Sum, DataType::Int32)
        | (ReduceOp::Sum, DataType::Int64)
        | (ReduceOp::Sum, DataType::Float32)
        | (ReduceOp::Sum, DataType::Float64)
        | (ReduceOp::Min, DataType::Int32)
        | (ReduceOp::Max, DataType::Int32)
        | (ReduceOp::Min, DataType::Int64)
        | (ReduceOp::Max, DataType::Int64) => true,
        // Float MIN/MAX, COUNT, Bool/Utf8, anything else: host-strip.
        _ => false,
    }
}

// dedup (groupby_common): the Stage-3 pinned D2H helpers
// (`download_pinned_{i32,i64,f32,f64}`) and the `unique_count` / `next_pow2`
// scan helpers used to be defined locally here (copies of the `groupby.rs`
// originals). They now live in the single canonical module
// `crate::exec::groupby_common`. This executor consumes host vectors from the
// pre stage, so it does NOT share the key-packing functions (`pack_keys` etc.)
// — those stay specific to the no-pre executors. Call sites are unchanged.
use crate::exec::groupby_common::{
    download_pinned_f32, download_pinned_f64, download_pinned_i32, download_pinned_i64,
    next_pow2, unique_count,
};

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
        DataType::Decimal128(_, _) => {
            return Err(BoltError::Plan(
                "Decimal128 not yet lowered to GPU; coming in a follow-up \
                 (GROUP BY key)"
                    .into(),
            ))
        }
        DataType::Date32 | DataType::Timestamp(_, _) => {
            return Err(BoltError::Type(format!(
                "GROUP BY key dtype {:?} not yet supported",
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

    let mut host_keys: Vec<i64> = key_host.to_i64_for_key(&key_io.name)?;
    let n_compacted = host_keys.len();
    if n_compacted != compacted.n_rows() {
        return Err(BoltError::Other(format!(
            "internal: key column length {} disagrees with compacted row count {}",
            n_compacted,
            compacted.n_rows()
        )));
    }

    // NULL-group synthesis (single-column GROUP BY): SQL groups ALL NULL-keyed
    // rows into ONE group whose key is NULL, with aggregates computed over
    // those rows. We remap each NULL-key row's encoded key to the reserved
    // `NULL_GROUP_KEY` sentinel and KEEP every row (preserving row alignment
    // with the per-aggregate value columns). The NULL-key rows then hash to a
    // single slot like any real group; `build_key_array` emits NULL for that
    // slot. Per-aggregate value-NULL masks are handled independently in
    // `run_one_aggregate`, so `COUNT(v)` still excludes value-NULLs WITHIN the
    // NULL group (the all-NULL-key case: COUNT(*) counts every row, COUNT(v)
    // counts only non-NULL `v`).
    //
    // Row NULLs come from `key_host.validity` (the pre-stage's per-output
    // validity, issue B): the key passthrough output depends only on the key
    // `LoadColumn`, so its bitmap is the key's OWN nulls, post-compaction
    // aligned with `host_keys`. We gate on the ORIGINAL key column's
    // `null_count` first as a cheap belt-and-suspenders: a provably null-free
    // source key never enters the remap, so `SUM(<nullable value>) GROUP BY
    // <non-null key>` (the common case) is bit-identical to before.
    let key_src_has_nulls = table_batch
        .schema()
        .index_of(&key_io.name)
        .map(|idx| table_batch.column(idx).null_count() > 0)
        .unwrap_or(false);
    let synthesised_null_group = if key_src_has_nulls {
        match &key_host.validity {
            Some(vmask) if vmask.iter().any(|&b| b == 0) => {
                // Sentinel collision: a genuine (valid) key literally equal to
                // `NULL_GROUP_KEY` can't be told apart from the synthesised
                // NULL group. No valid-flag-with-pre fallback exists yet, so
                // reject with a clear error rather than miscount.
                let collision = host_keys
                    .iter()
                    .zip(vmask.iter())
                    .any(|(&kk, &ok)| ok != 0 && kk == NULL_GROUP_KEY);
                if collision {
                    return Err(BoltError::Other(format!(
                        "GROUP BY key column '{}' contains a genuine key equal to \
                         the reserved NULL-group sentinel (i64::MAX); cannot \
                         synthesise the NULL group",
                        key_io.name
                    )));
                }
                for (kk, &ok) in host_keys.iter_mut().zip(vmask.iter()) {
                    if ok == 0 {
                        *kk = NULL_GROUP_KEY;
                    }
                }
                true
            }
            // Source flagged nulls but the (post-compaction) key bitmap shows
            // none — e.g. a WHERE filter removed every NULL-key row. Nothing to
            // synthesise.
            _ => false,
        }
    } else {
        false
    };

    // Validate no key collides with the EMPTY_KEY sentinel. Review C7:
    // unlike `execute_groupby` (which routes the colliding case to the
    // sentinel-free valid-flag executor), this pre-projected path has no
    // valid-flag-with-pre fallback yet, so we still reject with a clear
    // error rather than silently producing wrong results. A `log::warn!`
    // makes the bail-out observable in production.
    if host_keys.iter().any(|&k| k == EMPTY_KEY) {
        log::warn!(
            "execute_groupby_with_pre: GROUP BY key '{}' contains \
             i64::MIN (classic-kernel empty-slot sentinel); rejecting \
             because no valid-flag-with-pre executor exists yet",
            key_io.name
        );
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

    let stream = CudaStream::null_or_default();
    let host_keys_init: Vec<i64> = vec![EMPTY_KEY; k];
    let mut keys_table =
        GpuVec::<i64>::from_slice_async(&host_keys_init, stream.raw())?;
    let key_col_gpu = GpuVec::<i64>::from_slice_async(&host_keys, stream.raw())?;
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
    //
    // PV-stage-e: read the planner-time validity signal off `pre_spec`. If
    // ANY pre input was flagged by `populate_input_validity`, the pre stage
    // may have produced validity-bearing outputs; the per-aggregate
    // dispatch below uses this together with the resolved column's
    // validity mask to decide between native `_with_validity` and the
    // host-strip fallback. An empty (legacy / safe-default) flag vector
    // collapses to `false` so existing call sites are bit-identical.
    let any_input_has_validity: bool =
        pre_spec.input_has_validity.iter().any(|&v| v);

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
            any_input_has_validity,
        )?;
        acc_results.push(acc);
    }

    // -- 8. Download the keys table to drive output assembly.
    let host_keys_table: Vec<i64> = download_pinned_i64(&keys_table, &stream)?;
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
    arrays.push(build_key_array(
        &groups,
        original_key_dtype,
        synthesised_null_group,
    )?);

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
            DataType::Bool | DataType::Utf8 | DataType::Decimal128(_, _) | DataType::Date32 | DataType::Timestamp(_, _) => {
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
    let module = module_cache::get_or_build_module(
        module_path!(),
        format!("pre_kernel:{}:{:?}", PRE_KERNEL_ENTRY, pre_spec_for_ptx),
        None,
        || compile_ptx(&pre_spec_for_ptx, PRE_KERNEL_ENTRY),
    )?;
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
    let stream = CudaStream::null_or_default();
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
        // GroupBy pre-stage predicate kernel: same situation as
        // `agg_with_pre`. The planner doesn't lower `Op::IsNullCheck`
        // through this aggregate predicate path today (only the projection-
        // scan-chain path uses it), so we pass an empty validity slice and
        // the scan_kernel emits the legacy no-validity param layout
        // bit-for-bit. See the matching comment in `agg_with_pre.rs`.
        let validity_ptrs: Vec<CUdeviceptr> = Vec::new();
        crate::exec::compact::launch_predicate_kernel(
            pred_function,
            &input_ptrs,
            mask.device_ptr(),
            &validity_ptrs,
            n_rows_u32,
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

    let key_validity_flag = key_validity.is_some();
    let module = module_cache::get_or_build_module(
        module_path!(),
        format!("groupby_keys:validity={}", key_validity_flag),
        None,
        || {
            if key_validity_flag {
                compile_groupby_keys_kernel_with_validity()
            } else {
                compile_groupby_keys_kernel()
            }
        },
    )?;
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
/// PV-stage-e: now invoked from `run_typed_agg_native_validity` whenever
/// the planner's `KernelSpec::input_has_validity` signal lights up and
/// the resolved aggregate column has a non-trivial validity mask. The
/// host-strip path in `run_typed_agg` is preserved as the fallback for
/// shapes the native kernel can't handle (Utf8 keys, float MIN/MAX,
/// etc.) — see `dispatch_native_validity` for the gating predicate.
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

    let value_validity_flag = value_validity.is_some();
    let module = module_cache::get_or_build_module(
        module_path!(),
        format!("groupby_agg:{:?}:{:?}:validity={}", op, input_dtype, value_validity_flag),
        None,
        || {
            if value_validity_flag {
                compile_groupby_agg_kernel_with_validity(op, input_dtype)
            } else {
                compile_groupby_agg_kernel(op, input_dtype)
            }
        },
    )?;
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
/// PV-stage-f: now consumed by both `groupby_with_pre` (via
/// `run_typed_agg_native_validity`) and `groupby` / `groupby_valid`,
/// which call this directly from their no-pre native-validity paths.
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
    /// v0.7: VAR_POP / VAR_SAMP / STDDEV_POP / STDDEV_SAMP per-group
    /// Welford state, indexed by GPU slot (length `k`). See the matching
    /// variant in `crate::exec::groupby::AccDownload::Welford` for the
    /// rationale — the (count, mean, M2) triple doesn't fit a single
    /// atomic, so per-group accumulation is folded host-side.
    Welford { states: Vec<crate::exec::welford::WelfordState> },
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
    any_input_has_validity: bool,
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
                any_input_has_validity,
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
                filter_keys_if_needed(host_keys, &resolved, stream)?;
            let count_group_col: &GpuVec<i64> =
                filtered_keys_gpu.as_ref().unwrap_or(group_col);
            let ones: Vec<i64> = vec![1i64; n_valid];
            let input_gpu = GpuVec::<i64>::from_slice_async(&ones, stream.raw())?;
            let identity_init: Vec<i64> = vec![0i64; k];
            let mut acc_table =
                GpuVec::<i64>::from_slice_async(&identity_init, stream.raw())?;
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
            Ok(AccDownload::I64(download_pinned_i64(&acc_table, stream)?))
        }

        AggregateExpr::VarPop(_)
        | AggregateExpr::VarSamp(_)
        | AggregateExpr::StddevPop(_)
        | AggregateExpr::StddevSamp(_) => {
            // v0.7: per-group Welford via the pre-stage path. Same
            // strategy as `crate::exec::groupby::run_welford_aggregate`:
            // download `keys_table` host-side, build a key -> slot map,
            // fold (value, key) pairs into `WelfordState[slot]`. The
            // pre stage has already materialised the value column into
            // a `CompactedPreOutputs` host slice, so we don't need a
            // GPU device buffer for it.
            run_welford_aggregate_with_pre(
                agg,
                aggregate,
                input_ord,
                name_to_pre_ord,
                compacted,
                host_keys,
                keys_table,
                k,
            )
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
                filter_keys_if_needed(host_keys, &resolved, stream)?;
            let avg_group_col: &GpuVec<i64> =
                filtered_keys_gpu.as_ref().unwrap_or(group_col);

            // Build the filtered value column at f64.
            let sum_input: Vec<f64> = match resolved.validity() {
                Some(mask) => host_col.to_f64_filtered(mask, &col_io.name)?,
                None => host_col.to_f64(&col_io.name)?,
            };
            debug_assert_eq!(sum_input.len(), n_valid);
            let input_gpu = GpuVec::<f64>::from_slice_async(&sum_input, stream.raw())?;
            let sum_init: Vec<f64> = vec![0.0f64; k];
            let mut sum_acc =
                GpuVec::<f64>::from_slice_async(&sum_init, stream.raw())?;
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
            let sum_host = download_pinned_f64(&sum_acc, stream)?;

            // COUNT(non-NULL) per group over the same filtered keys.
            let ones: Vec<i64> = vec![1i64; n_valid];
            let count_input = GpuVec::<i64>::from_slice_async(&ones, stream.raw())?;
            let count_init: Vec<i64> = vec![0i64; k];
            let mut count_acc =
                GpuVec::<i64>::from_slice_async(&count_init, stream.raw())?;
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
            let count_host = download_pinned_i64(&count_acc, stream)?;

            Ok(AccDownload::Avg {
                sum: sum_host,
                count: count_host,
            })
        }
    }
}

/// Per-group Welford accumulation for the pre-staged GROUP BY path.
///
/// Reads the value column from the pre-stage's compacted host buffer,
/// downloads `keys_table` to find each row's GPU slot, and folds
/// `(value, slot)` pairs into a per-slot `WelfordState` table.
///
/// See `crate::exec::groupby::run_welford_aggregate` for the design
/// rationale.
fn run_welford_aggregate_with_pre(
    agg: &AggregateExpr,
    aggregate: &AggregateSpec,
    input_ord: usize,
    name_to_pre_ord: &HashMap<String, usize>,
    compacted: &CompactedPreOutputs,
    host_keys: &[i64],
    keys_table: &GpuVec<i64>,
    k: usize,
) -> BoltResult<AccDownload> {
    let (col_io, resolved) =
        resolve_agg_input_slow(agg, aggregate, input_ord, name_to_pre_ord, compacted)?;
    let host_col = resolved.as_ref();

    // Build host-side value column at f64. NULL rows are dropped via the
    // resolver's validity mask (same shape used by AVG).
    let values_f64: Vec<f64> = match resolved.validity() {
        Some(mask) => host_col.to_f64_filtered(mask, &col_io.name)?,
        None => host_col.to_f64(&col_io.name)?,
    };

    // Download keys_table once so we can map slot -> key, then invert to
    // key -> slot for the per-row lookup.
    let host_keys_table: Vec<i64> = keys_table.to_vec()?;
    debug_assert_eq!(host_keys_table.len(), k);
    let mut key_to_slot: HashMap<i64, usize> =
        HashMap::with_capacity(host_keys_table.len().min(1 << 20));
    for (slot, &kk) in host_keys_table.iter().enumerate() {
        if kk != EMPTY_KEY {
            key_to_slot.insert(kk, slot);
        }
    }

    let mut states: Vec<crate::exec::welford::WelfordState> =
        vec![crate::exec::welford::WelfordState::empty(); k];

    // Walk rows: for each non-NULL row, look up its key in the slot map
    // and fold the value into the per-slot Welford state.
    match resolved.validity() {
        None => {
            debug_assert_eq!(values_f64.len(), host_keys.len());
            for (i, &key) in host_keys.iter().enumerate() {
                if let Some(&slot) = key_to_slot.get(&key) {
                    states[slot].push(values_f64[i]);
                }
            }
        }
        Some(mask) => {
            debug_assert_eq!(mask.len(), host_keys.len());
            let mut idx_vals: usize = 0;
            for (i, &key) in host_keys.iter().enumerate() {
                if !mask[i] {
                    continue;
                }
                if let Some(&slot) = key_to_slot.get(&key) {
                    states[slot].push(values_f64[idx_vals]);
                }
                idx_vals += 1;
            }
        }
    }

    Ok(AccDownload::Welford { states })
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
    stream: &CudaStream,
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
            let gpu = GpuVec::<i64>::from_slice_async(&filtered, stream.raw())?;
            Ok((Some(gpu), n))
        }
        None => Ok((None, host_keys.len())),
    }
}

/// PV-stage-e: native validity dispatch — the planner-driven hot path.
///
/// Instead of host-filtering NULL rows before upload, we:
///
/// 1. Upload the FULL value column verbatim (NULL slots may hold garbage
///    bit patterns — the kernel will skip them).
/// 2. Upload the FULL key column (same — kernel won't touch the slot).
/// 3. Pack the resolved column's validity mask into the kernel's
///    expected layout via [`pack_validity_bits`] (Vec<u32> words,
///    little-endian bit order, one bit per row).
/// 4. Dispatch to [`launch_agg_kernel_with_validity`], which routes
///    through `compile_groupby_agg_kernel_with_validity` and emits a
///    `bfe.u32`-based bit-test before the atomic update.
///
/// Increments [`NATIVE_VALIDITY_LAUNCHES`] once per call so inline tests
/// can assert the planner signal actually drove dispatch (rather than
/// falling through to the host-strip path).
///
/// `validity` MUST be `Some(_)` and have at least one false bit — both
/// conditions are checked by [`dispatch_native_validity`] before this
/// function is invoked.
#[allow(clippy::too_many_arguments)]
fn run_typed_agg_native_validity(
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
    let validity = resolved.validity().ok_or_else(|| {
        BoltError::Other(
            "groupby_with_pre: native-validity dispatch invoked without a validity mask"
                .into(),
        )
    })?;
    if validity.len() != host_keys.len() {
        return Err(BoltError::Other(format!(
            "groupby_with_pre: validity mask length {} != key column length {}",
            validity.len(),
            host_keys.len()
        )));
    }
    debug_assert_eq!(
        n_rows,
        host_keys.len(),
        "native-validity dispatch invariant: n_rows == host_keys.len()"
    );

    // Pack the validity mask into the kernel's u32-word layout. The mask
    // is `&[bool]` here; we project to the `u8` representation
    // `pack_validity_bits` consumes (1 = valid, 0 = null) and feed it.
    let validity_bytes: Vec<u8> =
        validity.iter().map(|&v| if v { 1u8 } else { 0u8 }).collect();
    let packed = pack_validity_bits(&validity_bytes);
    let validity_gpu = GpuVec::<u32>::from_slice_async(&packed, stream.raw())?;

    // Account the native-dispatch launch. The counter is observed by
    // inline tests; the production path does not read it.
    NATIVE_VALIDITY_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    match (col_io.dtype, &host_col.values) {
        (DataType::Int32, HostColValues::I32(host)) => {
            // SUM(Int32) widens to i64; MIN/MAX(Int32) keep i32 width.
            // Mirror the host-strip branch above.
            if matches!(op, ReduceOp::Sum) {
                let widened: Vec<i64> = host.iter().map(|&x| x as i64).collect();
                let input_gpu =
                    GpuVec::<i64>::from_slice_async(&widened, stream.raw())?;
                let init: Vec<i64> = vec![identity_i64(op); k];
                let mut acc =
                    GpuVec::<i64>::from_slice_async(&init, stream.raw())?;
                launch_agg_kernel_with_validity::<i64>(
                    op, DataType::Int64, group_col, keys_table, &input_gpu,
                    &mut acc, &validity_gpu, n_rows, k_u32, stream,
                )?;
                return Ok(AccDownload::I64(download_pinned_i64(&acc, stream)?));
            }
            let input_gpu = GpuVec::<i32>::from_slice_async(host, stream.raw())?;
            let init: Vec<i32> = vec![identity_i32(op); k];
            let mut acc = GpuVec::<i32>::from_slice_async(&init, stream.raw())?;
            launch_agg_kernel_with_validity::<i32>(
                op, DataType::Int32, group_col, keys_table, &input_gpu, &mut acc,
                &validity_gpu, n_rows, k_u32, stream,
            )?;
            Ok(AccDownload::I32(download_pinned_i32(&acc, stream)?))
        }
        (DataType::Int64, HostColValues::I64(host)) => {
            let input_gpu = GpuVec::<i64>::from_slice_async(host, stream.raw())?;
            let init: Vec<i64> = vec![identity_i64(op); k];
            let mut acc = GpuVec::<i64>::from_slice_async(&init, stream.raw())?;
            launch_agg_kernel_with_validity::<i64>(
                op, DataType::Int64, group_col, keys_table, &input_gpu, &mut acc,
                &validity_gpu, n_rows, k_u32, stream,
            )?;
            Ok(AccDownload::I64(download_pinned_i64(&acc, stream)?))
        }
        (DataType::Float32, HostColValues::F32(host)) => {
            // Float MIN/MAX is filtered out by `dispatch_native_validity`,
            // so only SUM reaches here.
            debug_assert!(matches!(op, ReduceOp::Sum));
            let input_gpu = GpuVec::<f32>::from_slice_async(host, stream.raw())?;
            let init: Vec<f32> = vec![identity_f32(op); k];
            let mut acc = GpuVec::<f32>::from_slice_async(&init, stream.raw())?;
            launch_agg_kernel_with_validity::<f32>(
                op, DataType::Float32, group_col, keys_table, &input_gpu,
                &mut acc, &validity_gpu, n_rows, k_u32, stream,
            )?;
            Ok(AccDownload::F32(download_pinned_f32(&acc, stream)?))
        }
        (DataType::Float64, HostColValues::F64(host)) => {
            debug_assert!(matches!(op, ReduceOp::Sum));
            let input_gpu = GpuVec::<f64>::from_slice_async(host, stream.raw())?;
            let init: Vec<f64> = vec![identity_f64(op); k];
            let mut acc = GpuVec::<f64>::from_slice_async(&init, stream.raw())?;
            launch_agg_kernel_with_validity::<f64>(
                op, DataType::Float64, group_col, keys_table, &input_gpu,
                &mut acc, &validity_gpu, n_rows, k_u32, stream,
            )?;
            Ok(AccDownload::F64(download_pinned_f64(&acc, stream)?))
        }
        (dt, _) => Err(BoltError::Type(format!(
            "groupby_with_pre: native-validity dispatch reached unsupported dtype {:?} \
             for column '{}'",
            dt, col_io.name
        ))),
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
    any_input_has_validity: bool,
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

    // PV-stage-e: planner signal + dtype gate — when the native
    // `_with_validity` kernel can handle this row, dispatch to it and
    // skip the host strip. The `dispatch_native_validity` helper
    // documents the three preconditions (planner flag, validity mask
    // present, supported `(op, dtype)`); failing any of them falls
    // through to the legacy host-strip path below.
    let validity = resolved.validity();
    if dispatch_native_validity(any_input_has_validity, validity, op, col_io.dtype) {
        return run_typed_agg_native_validity(
            op, col_io, resolved, host_keys, group_col, keys_table, n_rows, k, k_u32,
            stream,
        );
    }

    // Legacy host-strip path. If the resolved column has a validity mask
    // with any false bits, filter both (keys, values) together so the GPU
    // sees a parallel pair of rows with no NULL slots. The keys-table is
    // shared and remains correct because it indexes the original key set.
    let (filtered_keys_gpu, n_eff) = filter_keys_if_needed(host_keys, resolved, stream)?;
    let eff_group_col: &GpuVec<i64> = filtered_keys_gpu.as_ref().unwrap_or(group_col);

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
                let input_gpu =
                    GpuVec::<i64>::from_slice_async(&widened, stream.raw())?;
                let init: Vec<i64> = vec![identity_i64(op); k];
                let mut acc =
                    GpuVec::<i64>::from_slice_async(&init, stream.raw())?;
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
                return Ok(AccDownload::I64(download_pinned_i64(&acc, stream)?));
            }

            let filtered: Vec<i32> = filter_by_validity(host, validity);
            let input_gpu =
                GpuVec::<i32>::from_slice_async(&filtered, stream.raw())?;
            let init: Vec<i32> = vec![identity_i32(op); k];
            let mut acc = GpuVec::<i32>::from_slice_async(&init, stream.raw())?;
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
            Ok(AccDownload::I32(download_pinned_i32(&acc, stream)?))
        }
        (DataType::Int64, HostColValues::I64(host)) => {
            // Stage C: filter NULL rows in lockstep with the key column.
            // Previously this branch uploaded the raw values buffer
            // verbatim (would have summed NULL slots as 0); the
            // Stage-B reject guarded against that. Now that the gate
            // is lifted we must apply the validity mask explicitly.
            let filtered: Vec<i64> = filter_by_validity(host, validity);
            let input_gpu =
                GpuVec::<i64>::from_slice_async(&filtered, stream.raw())?;
            let init: Vec<i64> = vec![identity_i64(op); k];
            let mut acc = GpuVec::<i64>::from_slice_async(&init, stream.raw())?;
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
            Ok(AccDownload::I64(download_pinned_i64(&acc, stream)?))
        }
        (DataType::Float32, HostColValues::F32(host)) => {
            if matches!(op, ReduceOp::Min | ReduceOp::Max) {
                return Err(BoltError::Other(
                    "MIN/MAX over float not yet supported".into(),
                ));
            }
            let filtered: Vec<f32> = filter_by_validity(host, validity);
            let input_gpu =
                GpuVec::<f32>::from_slice_async(&filtered, stream.raw())?;
            let init: Vec<f32> = vec![identity_f32(op); k];
            let mut acc = GpuVec::<f32>::from_slice_async(&init, stream.raw())?;
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
            Ok(AccDownload::F32(download_pinned_f32(&acc, stream)?))
        }
        (DataType::Float64, HostColValues::F64(host)) => {
            if matches!(op, ReduceOp::Min | ReduceOp::Max) {
                return Err(BoltError::Other(
                    "MIN/MAX over float not yet supported".into(),
                ));
            }
            let filtered: Vec<f64> = filter_by_validity(host, validity);
            let input_gpu =
                GpuVec::<f64>::from_slice_async(&filtered, stream.raw())?;
            let init: Vec<f64> = vec![identity_f64(op); k];
            let mut acc = GpuVec::<f64>::from_slice_async(&init, stream.raw())?;
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
            Ok(AccDownload::F64(download_pinned_f64(&acc, stream)?))
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

    // Fast path: bare column ref into pre outputs. The pre-stage DOES carry a
    // per-output NULL bitmap (issue B: each output's validity is the AND of only
    // the inputs it depends on), downloaded into `HostCol::validity` by
    // `to_host_col`. For a bare column `v`, that bitmap is `v`'s own nulls, so we
    // must surface it — otherwise `COUNT(v)`/`MIN(v)`/`AVG(v)` would (wrongly)
    // count/aggregate NULL rows. When the column has no nulls (`validity` is
    // `None` or all-valid) we return the zero-alloc `Borrowed` form, bit-
    // identical to the previous behaviour for the common null-free case.
    if let Some(name) = expr_agg::try_bare_column(inner) {
        if let Some(&pre_ord) = name_to_pre_ord.get(name) {
            let host_col = compacted.cols.get(pre_ord).ok_or_else(|| {
                BoltError::Other(format!(
                    "internal: pre output ordinal {} out of range (have {} compacted cols)",
                    pre_ord,
                    compacted.cols.len()
                ))
            })?;
            let resolved = match &host_col.validity {
                Some(v) if v.iter().any(|&b| b == 0) => {
                    let mask: Vec<bool> = v.iter().map(|&b| b != 0).collect();
                    ResolvedHostCol::BorrowedWithValidity {
                        col: host_col,
                        validity: mask,
                    }
                }
                _ => ResolvedHostCol::Borrowed(host_col),
            };
            return Ok((col_io, resolved));
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
    /// Borrowed view of a pre-stage output with no NULLs (validity absent or
    /// all-valid). Zero-alloc; the common case.
    Borrowed(&'a HostCol),
    /// Borrowed pre-stage output values plus an owned per-row NULL mask
    /// (`true` = valid) converted from the column's `Option<Vec<u8>>` bitmap.
    /// Lets the bare-column fast path surface the pre-stage's per-output
    /// validity without cloning the value buffer.
    BorrowedWithValidity {
        col: &'a HostCol,
        validity: Vec<bool>,
    },
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
            ResolvedHostCol::BorrowedWithValidity { col, .. } => *col,
            ResolvedHostCol::Owned { col, .. } => col,
        }
    }

    /// Per-row validity mask, or `None` if every row is non-NULL.
    fn validity(&self) -> Option<&[bool]> {
        match self {
            ResolvedHostCol::Borrowed(_) => None,
            ResolvedHostCol::BorrowedWithValidity { validity, .. } => Some(validity),
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
        AggregateExpr::VarPop(e) | AggregateExpr::VarSamp(e) => e.as_ref(),
        // STDDEV variants store their operand boxed; deref to the inner
        // `Expr` so this helper's downstream column-resolution logic
        // still composes. The GROUP-BY-with-pre path then rejects the
        // STDDEV op proper at `run_one_aggregate` — but the helper still
        // needs to return *some* operand reference so the per-aggregate
        // expression-feed collection (used by the pre kernel) doesn't
        // panic on an exhaustive-match miss before that rejection fires.
        AggregateExpr::StddevPop(e) | AggregateExpr::StddevSamp(e) => e.as_ref(),
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
    synthesised_null_group: bool,
) -> BoltResult<ArrayRef> {
    // A group whose encoded key is `NULL_GROUP_KEY` is the synthesised SQL NULL
    // group — emit NULL for its key cell instead of decoding the sentinel. We
    // only treat the sentinel as NULL when synthesis actually happened, so a
    // genuine Int64 key literally holding `i64::MAX` (synthesis was not active)
    // still round-trips as its real value.
    let is_null_group = |k: i64| synthesised_null_group && k == NULL_GROUP_KEY;
    match original_key_dtype {
        DataType::Int32 => {
            let out: Vec<Option<i32>> = groups
                .iter()
                .map(|(k, _)| {
                    if is_null_group(*k) {
                        Ok(None)
                    } else {
                        i32::try_from(*k).map(Some).map_err(|_| {
                            BoltError::Type(format!(
                                "GROUP BY: key {} does not fit in Int32 on output",
                                k
                            ))
                        })
                    }
                })
                .collect::<BoltResult<Vec<Option<i32>>>>()?;
            Ok(Arc::new(Int32Array::from(out)) as ArrayRef)
        }
        DataType::Int64 => {
            let out: Vec<Option<i64>> = groups
                .iter()
                .map(|(k, _)| if is_null_group(*k) { None } else { Some(*k) })
                .collect();
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
        // v0.7: VAR_POP / VAR_SAMP / STDDEV_POP / STDDEV_SAMP per-group
        // finalisation from the host-side Welford state. Output is always
        // a nullable Float64 array. See
        // `crate::exec::groupby::finalize_welford_array` for the matching
        // logic.
        (AggregateExpr::VarPop(_), AccDownload::Welford { states }) => {
            finalize_welford_array_with_pre(states, groups, WelfordOutKind::VarPop, out_field)
        }
        (AggregateExpr::VarSamp(_), AccDownload::Welford { states }) => {
            finalize_welford_array_with_pre(states, groups, WelfordOutKind::VarSamp, out_field)
        }
        (AggregateExpr::StddevPop(_), AccDownload::Welford { states }) => {
            finalize_welford_array_with_pre(states, groups, WelfordOutKind::StddevPop, out_field)
        }
        (AggregateExpr::StddevSamp(_), AccDownload::Welford { states }) => {
            finalize_welford_array_with_pre(states, groups, WelfordOutKind::StddevSamp, out_field)
        }
        (
            AggregateExpr::VarPop(_)
            | AggregateExpr::VarSamp(_)
            | AggregateExpr::StddevPop(_)
            | AggregateExpr::StddevSamp(_),
            _,
        ) => Err(BoltError::Other(
            "internal: VAR/STDDEV aggregate received a non-Welford accumulator \
             (with-pre path)"
                .into(),
        )),
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
                AccDownload::Welford { .. } => {
                    return Err(BoltError::Other(
                        "internal: Welford accumulator passed to SUM/MIN/MAX aggregate \
                         (with-pre path)"
                            .into(),
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

/// Tag for finalising a per-group Welford state in the with-pre GROUP BY path.
/// Mirrors `crate::exec::groupby::WelfordOutKind`.
#[derive(Clone, Copy, Debug)]
enum WelfordOutKind {
    VarPop,
    VarSamp,
    StddevPop,
    StddevSamp,
}

/// Build a nullable `Float64Array` from a per-slot Welford state vector
/// indexed by GPU slot (length `k`), reading one cell per `(_, slot)`
/// entry in `groups`. Mirrors `crate::exec::groupby::finalize_welford_array`.
fn finalize_welford_array_with_pre(
    states: &[crate::exec::welford::WelfordState],
    groups: &[(i64, usize)],
    kind: WelfordOutKind,
    out_field: &Field,
) -> BoltResult<ArrayRef> {
    if out_field.dtype != DataType::Float64 {
        return Err(BoltError::Type(format!(
            "GROUP BY (with-pre) VAR/STDDEV output dtype must be Float64, got {:?}",
            out_field.dtype
        )));
    }
    let mut out: Vec<Option<f64>> = Vec::with_capacity(groups.len());
    for (_, slot) in groups {
        let st = states.get(*slot).ok_or_else(|| {
            BoltError::Other(format!(
                "internal: with-pre Welford slot {} out of range (len {})",
                slot,
                states.len()
            ))
        })?;
        let v = match kind {
            WelfordOutKind::VarPop => st.var_pop(),
            WelfordOutKind::VarSamp => st.var_samp(),
            WelfordOutKind::StddevPop => st.stddev_pop(),
            WelfordOutKind::StddevSamp => st.stddev_samp(),
        };
        out.push(v);
    }
    Ok(Arc::new(Float64Array::from(out)) as ArrayRef)
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
            DataType::Bool | DataType::Utf8 | DataType::Decimal128(_, _) | DataType::Date32 | DataType::Timestamp(_, _) => {
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
            DataType::Bool | DataType::Utf8 | DataType::Decimal128(_, _) | DataType::Date32 | DataType::Timestamp(_, _) => {
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

// dedup (groupby_common): `unique_count` and `next_pow2` were copies of the
// `groupby.rs` originals; they now live in `crate::exec::groupby_common`
// (imported at the top of this module). Call sites are unchanged.

/// Build a `Type` error for a failed Arrow downcast.
fn downcast_err(role: &str, expected: &str) -> BoltError {
    BoltError::Type(format!(
        "groupby_with_pre: pre kernel {} could not be downcast to {}",
        role, expected
    ))
}

/// Map Arrow `DataType` to our plan `DataType`.
fn arrow_dtype_to_plan(d: &ArrowDataType) -> BoltResult<DataType> {
    crate::exec::schema_convert::arrow_dtype_to_plan_basic(d, "")
}

/// Build an Arrow `Schema` from our plan `Schema` for the output `RecordBatch`.
fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    crate::exec::schema_convert::plan_schema_to_arrow_schema_no_temporal(s, "this aggregate output path")
}

// v0.7: these arrow/plan aliases are used only by the #[cfg(test)] modules
// below; the non-test schema conversion now lives in exec::schema_convert.
// cfg(test)-gated so normal builds don't see an unused import.
#[cfg(test)]
use arrow_schema::{Field as ArrowField};

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

    // ---- PV-stage-e: dispatch decision (planner flag → native path) ----

    /// When the planner flagged no input as null-bearing, the dispatch
    /// predicate returns false regardless of the resolved-column mask —
    /// the host-strip path remains the default.
    #[test]
    fn dispatch_native_validity_off_when_planner_flag_false() {
        let mask = [true, false, true];
        let chose = dispatch_native_validity(
            /* any_input_has_validity = */ false,
            Some(&mask[..]),
            ReduceOp::Sum,
            DataType::Int64,
        );
        assert!(
            !chose,
            "planner flag is the gate: without it we must stay on host-strip"
        );
    }

    /// When the planner says "may have validity" AND the resolved mask
    /// carries a real NULL row AND the (op, dtype) is covered by the
    /// native `_with_validity` kernel, the predicate returns true.
    #[test]
    fn dispatch_native_validity_on_for_supported_op_with_mask() {
        let mask = [true, false, true];
        for (op, dtype) in [
            (ReduceOp::Sum, DataType::Int32),
            (ReduceOp::Sum, DataType::Int64),
            (ReduceOp::Sum, DataType::Float32),
            (ReduceOp::Sum, DataType::Float64),
            (ReduceOp::Min, DataType::Int32),
            (ReduceOp::Max, DataType::Int64),
        ] {
            assert!(
                dispatch_native_validity(true, Some(&mask[..]), op, dtype),
                "expected native dispatch for ({:?}, {:?})",
                op, dtype
            );
        }
    }

    /// Float MIN/MAX has no `_with_validity` emitter yet (Stage F follow-up
    /// — `float_atomics` would need a sibling), so the predicate must
    /// fall through to host-strip even when planner + mask line up.
    #[test]
    fn dispatch_native_validity_off_for_float_minmax() {
        let mask = [true, false, true];
        for (op, dtype) in [
            (ReduceOp::Min, DataType::Float32),
            (ReduceOp::Max, DataType::Float32),
            (ReduceOp::Min, DataType::Float64),
            (ReduceOp::Max, DataType::Float64),
        ] {
            assert!(
                !dispatch_native_validity(true, Some(&mask[..]), op, dtype),
                "Float MIN/MAX has no _with_validity emitter; expected host-strip for ({:?}, {:?})",
                op, dtype
            );
        }
    }

    /// All-valid masks short-circuit to host-strip — the classic kernel is
    /// bit-for-bit equivalent in that case, no need to allocate a packed
    /// bitmap and use the heavier `_with_validity` PTX.
    #[test]
    fn dispatch_native_validity_off_when_mask_all_true() {
        let mask = [true, true, true, true];
        let chose = dispatch_native_validity(
            true,
            Some(&mask[..]),
            ReduceOp::Sum,
            DataType::Int64,
        );
        assert!(
            !chose,
            "all-true mask means no NULLs in this batch; skip the native dispatch"
        );
    }

    /// Missing mask (`None`) means the pre stage didn't produce one even
    /// though the planner flagged input validity — fall back to host-strip
    /// (which is a no-op for None).
    #[test]
    fn dispatch_native_validity_off_when_mask_none() {
        let chose = dispatch_native_validity(
            true,
            None,
            ReduceOp::Sum,
            DataType::Int64,
        );
        assert!(!chose, "no mask -> no native dispatch");
    }

    // ---- v0.7 async-memcpy migration smoke -------------------------------
    //
    // Exercise the migrated `download_pinned_*` helpers under both stub
    // and real CUDA. The helpers wrap `to_pinned_async` +
    // `stream.synchronize()`; under `--features cuda-stub` the async
    // FFI shim returns `CUDA_ERROR_STUB`, so we expect `Err` rather
    // than a panic. The assertion is "did not panic" + "returned a
    // Result". Under a live CUDA context (`gpu:tier1`) the helpers
    // succeed against a tiny GpuVec round-trip.
    #[test]
    #[ignore = "gpu:tier1"]
    fn v07_async_download_pinned_helpers_do_not_panic() {
        let stream = CudaStream::null_or_default();
        // If we can't even allocate a 4-element GpuVec on this backend
        // (the stub will Err), short-circuit — the migrated pinned path
        // simply wasn't reachable, which is fine for the "no panic"
        // contract.
        let Ok(v_i32) = GpuVec::<i32>::from_slice_async(&[1, 2, 3, 4], stream.raw()) else { return };
        let _ = download_pinned_i32(&v_i32, &stream);
        let Ok(v_i64) = GpuVec::<i64>::from_slice_async(&[1i64, 2, 3, 4], stream.raw()) else { return };
        let _ = download_pinned_i64(&v_i64, &stream);
        let Ok(v_f32) = GpuVec::<f32>::from_slice_async(&[1.0f32, 2.0, 3.0, 4.0], stream.raw()) else { return };
        let _ = download_pinned_f32(&v_f32, &stream);
        let Ok(v_f64) = GpuVec::<f64>::from_slice_async(&[1.0f64, 2.0, 3.0, 4.0], stream.raw()) else { return };
        let _ = download_pinned_f64(&v_f64, &stream);
    }

    /// Engine-level GROUP BY WITH PRE round-trip: the only public entry
    /// to `execute_groupby_with_pre`. Exercises the migrated async H2D
    /// for the keys table + key column + agg input + agg accumulator,
    /// and the migrated pinned D2H for the keys table + accumulator.
    /// Stub mode short-circuits at the first CUDA FFI; the test is
    /// satisfied as long as no panic escapes.
    #[test]
    #[ignore = "gpu:tier1"]
    fn v07_async_groupby_with_pre_round_trip() {
        use crate::Engine;

        let mut engine = Engine::new().expect("ctx");
        // 9 rows, key in {0,1,2}, val computed by the pre-stage as v*2.
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

        // The `v + v` expression forces a pre-projection kernel; the
        // GROUP BY then routes through `execute_groupby_with_pre`.
        let h = match engine.sql("SELECT k, SUM(v + v) FROM t GROUP BY k") {
            Ok(h) => h,
            Err(_) => return,
        };
        let out = h.record_batch();
        let mut expected: std::collections::HashMap<i32, i64> =
            std::collections::HashMap::new();
        for v in 0..9i64 {
            *expected.entry((v as i32) % 3).or_default() += v + v;
        }
        let ks = out
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let ss = out
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..out.num_rows() {
            let k = ks.value(i);
            let s = ss.value(i);
            assert_eq!(Some(&s), expected.get(&k), "k={k} sum={s}");
        }
    }
}
