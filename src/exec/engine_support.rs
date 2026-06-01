// SPDX-License-Identifier: Apache-2.0

//! Standalone free helper functions lifted out of `exec::engine` (pure
//! reorganization; no behavior change).
//!
//! Schema converters, env-var / pool-stats helpers, the passthrough
//! detector, host↔Arrow column bridges, the incremental-cache row helpers,
//! and the debug-sync guard. None of these touch private `Engine`
//! internals, so they live cleanly outside the giant `impl Engine` block.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::error::{BoltError, BoltResult};
use crate::plan::{DataType, Field, KernelSpec, MemTableProvider, Schema};

/// Number of rows the device-side storage of a `GpuColumnData` currently
/// holds. Used by the incremental cache to compare against the host's
/// new row count and decide whether to prefix-extend or fully re-upload.
pub(crate) fn column_storage_rows(data: &crate::exec::gpu_table::GpuColumnData) -> usize {
    use crate::exec::gpu_table::GpuColumnData::*;
    match data {
        I32(v) => v.len(),
        I64(v) => v.len(),
        F32(v) => v.len(),
        F64(v) => v.len(),
        Bool(v) => v.len(),
        BoolNullable { values, .. } => values.len(),
        Utf8 { indices, .. } => indices.len(),
        DictUtf8 { keys, .. } => keys.len(),
        // v0.7 sub-task B: Decimal128 stores `2 * n_rows` u64 values
        // (interleaved [lo, hi] pairs); divide back to get the logical
        // row count.
        Decimal128 { values, .. } => values.len() / 2,
    }
}

/// Try to extend `prev` (a stale GpuColumn whose host data strictly grew)
/// into a fresh column at `n_rows_total` rows by preserving the
/// previously-uploaded prefix and HtoD-uploading only the tail.
///
/// Returns:
///   - `Ok(Some(new_column))` — extension succeeded; caller should
///     prefer this over a full re-upload (no PCIe traffic for the
///     prefix).
///   - `Ok(None)` — the variant can't be safely extended in place (e.g.
///     bit-packed validity bitmap with a non-byte-aligned previous row
///     count). Caller should fall back to a full re-upload.
///   - `Err(_)` — a CUDA / Arrow error.
pub(crate) fn try_extend_column(
    prev: crate::exec::gpu_table::GpuColumn,
    concatenated: &RecordBatch,
    col_idx: usize,
    n_rows_total: usize,
) -> BoltResult<Option<crate::exec::gpu_table::GpuColumn>> {
    use crate::exec::gpu_table::{GpuColumn, GpuColumnData};
    let prev_rows = column_storage_rows(&prev.data);
    // Caller already enforced 0 < prev_rows < n_rows_total but re-check
    // defensively here so the helpers can stand alone.
    if prev_rows == 0 || prev_rows >= n_rows_total {
        return Ok(None);
    }
    let arr = concatenated.column(col_idx);
    let GpuColumn {
        name,
        dtype,
        data,
        host_revision: _,
    } = prev;
    let new_data: GpuColumnData = match data {
        GpuColumnData::I32(old) => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| {
                    BoltError::Type(format!(
                        "incremental extend: column '{name}' was I32 on device but \
                         host array is {:?}",
                        arr.data_type()
                    ))
                })?;
            let tail: Vec<i32> = (prev_rows..n_rows_total)
                .map(|i| pa.value(i))
                .collect();
            let extended = old.extended_with_prefix(n_rows_total, prev_rows, &tail)?;
            GpuColumnData::I32(extended)
        }
        GpuColumnData::I64(old) => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    BoltError::Type(format!(
                        "incremental extend: column '{name}' was I64 on device but \
                         host array is {:?}",
                        arr.data_type()
                    ))
                })?;
            let tail: Vec<i64> = (prev_rows..n_rows_total)
                .map(|i| pa.value(i))
                .collect();
            let extended = old.extended_with_prefix(n_rows_total, prev_rows, &tail)?;
            GpuColumnData::I64(extended)
        }
        GpuColumnData::F32(old) => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    BoltError::Type(format!(
                        "incremental extend: column '{name}' was F32 on device but \
                         host array is {:?}",
                        arr.data_type()
                    ))
                })?;
            let tail: Vec<f32> = (prev_rows..n_rows_total)
                .map(|i| pa.value(i))
                .collect();
            let extended = old.extended_with_prefix(n_rows_total, prev_rows, &tail)?;
            GpuColumnData::F32(extended)
        }
        GpuColumnData::F64(old) => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    BoltError::Type(format!(
                        "incremental extend: column '{name}' was F64 on device but \
                         host array is {:?}",
                        arr.data_type()
                    ))
                })?;
            let tail: Vec<f64> = (prev_rows..n_rows_total)
                .map(|i| pa.value(i))
                .collect();
            let extended = old.extended_with_prefix(n_rows_total, prev_rows, &tail)?;
            GpuColumnData::F64(extended)
        }
        GpuColumnData::Bool(old) => {
            let ba = arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    BoltError::Type(format!(
                        "incremental extend: column '{name}' was Bool on device but \
                         host array is {:?}",
                        arr.data_type()
                    ))
                })?;
            // Only safe for null-free Bool — the variant we have is
            // `Bool` (non-nullable). If the appended batch added nulls,
            // the GpuColumnData variant would need to become
            // `BoolNullable`, and we can't extend across a variant
            // change. Punt to full re-upload.
            use arrow::array::Array as _;
            if ba.null_count() != 0 {
                return Ok(None);
            }
            let tail_rows = n_rows_total - prev_rows;
            let mut tail: Vec<u8> = Vec::with_capacity(tail_rows);
            for i in prev_rows..n_rows_total {
                tail.push(if ba.value(i) { 1 } else { 0 });
            }
            let extended = old.extended_with_prefix(n_rows_total, prev_rows, &tail)?;
            GpuColumnData::Bool(extended)
        }
        GpuColumnData::BoolNullable { values, validity } => {
            let ba = arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    BoltError::Type(format!(
                        "incremental extend: column '{name}' was BoolNullable on \
                         device but host array is {:?}",
                        arr.data_type()
                    ))
                })?;
            let tail_rows = n_rows_total - prev_rows;
            let mut tail_v: Vec<u8> = Vec::with_capacity(tail_rows);
            let mut tail_m: Vec<u8> = Vec::with_capacity(tail_rows);
            use arrow::array::Array as _;
            for i in prev_rows..n_rows_total {
                if ba.is_null(i) {
                    tail_v.push(0);
                    tail_m.push(0);
                } else {
                    tail_v.push(if ba.value(i) { 1 } else { 0 });
                    tail_m.push(1);
                }
            }
            let new_values = values.extended_with_prefix(n_rows_total, prev_rows, &tail_v)?;
            let new_validity =
                validity.extended_with_prefix(n_rows_total, prev_rows, &tail_m)?;
            GpuColumnData::BoolNullable {
                values: new_values,
                validity: new_validity,
            }
        }
        // Utf8 / DictUtf8: the host-side dictionary is rebuilt on every
        // `register_batch` (review C10), and we'd need to re-derive
        // per-row indices from the new dictionary to update the GPU
        // copy. Falling back to a full re-upload is simpler and
        // correct — the prefix optimisation here would require teaching
        // the device-side keys layout about dict offsets, which is
        // out of scope for batch 5. Returning `None` triggers the
        // caller's full re-upload fallback.
        GpuColumnData::Utf8 { .. } | GpuColumnData::DictUtf8 { .. } => {
            return Ok(None);
        }
        // v0.7 sub-task B: Decimal128 prefix-extend isn't wired yet —
        // the tail would need a slice-and-pack helper paralleling
        // `Decimal128Array::value(i)`. Punt to a full re-upload for now;
        // every existing Decimal column test exercises the full-upload
        // path through `GpuColumn::upload`.
        GpuColumnData::Decimal128 { .. } => {
            return Ok(None);
        }
    };
    Ok(Some(GpuColumn {
        name,
        dtype,
        data: new_data,
        host_revision: 0, // caller overwrites
    }))
}

/// Synchronize the default stream and convert any pending CUDA error.
///
/// `cuLaunchKernel` is asynchronous: its return value reflects only whether
/// the launch was *accepted*, not whether the kernel later faulted. If we
/// don't synchronize, a kernel-side fault (illegal address, OOB shared
/// memory access, assertion failure, etc.) surfaces at the *next* CUDA API
/// call — which may be many lines away and in unrelated code, producing
/// extremely misleading error messages and stack traces during debugging.
///
/// In debug builds we call `cuCtxSynchronize` immediately after every
/// launch site so faults are reported at the actual launch that caused
/// them. Release builds skip this entirely: the `cfg!(debug_assertions)`
/// check is a compile-time constant, so the optimiser folds this function
/// into a no-op (`Ok(())`) and any per-launch latency goes to zero.
///
/// Cheap in release: a no-op when `cfg!(debug_assertions)` is false.
#[inline]
pub(crate) fn debug_sync_check() -> crate::error::BoltResult<()> {
    if cfg!(debug_assertions) {
        unsafe { crate::cuda::cuda_sys::check(crate::cuda::cuda_sys::cuCtxSynchronize())? };
    }
    Ok(())
}

/// Map Arrow `DataType` to our plan `DataType`. Errors on unsupported types.
///
/// **Stage 4 / Stage 6** — `Dictionary(_, Utf8)` is accepted as a register-table
/// type and exposed to the planner as `DataType::Utf8`. The fact that the column
/// is dictionary-encoded is a *storage* detail: query planning, projection,
/// filtering, ORDER BY all reason about it as a Utf8 column. SQL frontends
/// see it identically to a flat `StringArray` column.
///
/// Stage 4 accepted any key width (Int32 *or* Int64) and routed through the
/// flatten step. Stage 6 added a native ingest path for `Dictionary<Int32, Utf8>`
/// in `GpuTable::from_record_batch` and `DictRegistry::register_table`, so the
/// flatten in `flatten_dictionary_utf8_columns` is now a deprecated no-op (the
/// dict layout reaches the GPU table directly). Int64-keyed dicts still take
/// the legacy path through `GpuColumn::upload`.
pub(crate) fn arrow_dtype_to_plan(d: &ArrowDataType) -> BoltResult<DataType> {
    crate::exec::schema_convert::arrow_dtype_to_plan(d)
}

/// Stage 4 — rewrite every `Dictionary(_, Utf8)` column in `batch` into a
/// plain `StringArray`, leaving non-dictionary columns untouched. Returns
/// the rewritten `RecordBatch` (cheap if no dict columns: just reuses the
/// original arrays via `Arc`).
///
/// Why flatten at registration time rather than carrying the dict through?
/// The GPU storage (`GpuTable`) already manages its own dictionary for Utf8
/// columns (see `GpuColumnData::Utf8`), so re-using the input dict would
/// require teaching every consumer (GpuTable upload, projection, gather,
/// expression evaluator, ORDER BY's host-side `take`) to read both dict
/// variants. Materialising once at registration is O(n) per dict column —
/// the same cost the engine's own dictionary builder pays — and keeps every
/// downstream stage's Utf8 handling unified on `StringArray`.
///
/// **Stage 5** added a native `GpuColumnData::DictUtf8` variant to
/// `GpuTable` so callers that go directly through `GpuTable::from_record_batch`
/// (skipping the engine's registry / `MemTableProvider`) can preserve the
/// input dictionary instead of materialising it.
///
/// **Stage 6** — DEPRECATED. The dict registry and `GpuTable` now ingest
/// `DictionaryArray<Int32, Utf8>` natively (the registry matches the dict
/// variant directly; `GpuTable::from_record_batch` routes through
/// `upload_dict_utf8`). The engine no longer calls this helper from
/// `register_table` / `replace_table` / `register_batch`, but the body is
/// kept callable so any out-of-tree consumer that imported it still
/// compiles. Will be removed in a wave following Stage 7.
#[deprecated(
    since = "0.3.0",
    note = "DictionaryArray<Int32, Utf8> is now ingested natively by DictRegistry \
            and GpuTable::from_record_batch; this flatten step is no longer \
            invoked by the engine and will be removed in a future release."
)]
#[allow(dead_code)]
pub(crate) fn flatten_dictionary_utf8_columns(batch: RecordBatch) -> BoltResult<RecordBatch> {
    use arrow_array::{Array, DictionaryArray, StringArray};
    use arrow_array::types::{Int32Type, Int64Type};

    let schema = batch.schema();
    let mut changed = false;
    let mut new_fields: Vec<ArrowField> = Vec::with_capacity(schema.fields().len());
    let mut new_cols: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    for (idx, field) in schema.fields().iter().enumerate() {
        let col = batch.column(idx);
        match field.data_type() {
            ArrowDataType::Dictionary(key_ty, value_ty)
                if matches!(value_ty.as_ref(), ArrowDataType::Utf8) =>
            {
                // Decode (key_idx, value_idx) -> StringArray entries.
                // Supports Int32 and Int64 key types (matches `arrow_dtype_to_plan`).
                let n = col.len();
                let mut out: Vec<Option<String>> = Vec::with_capacity(n);
                let decode_into = |out: &mut Vec<Option<String>>,
                                   value_idx: usize,
                                   sa: &StringArray| {
                    if sa.is_null(value_idx) {
                        out.push(None);
                    } else {
                        out.push(Some(sa.value(value_idx).to_string()));
                    }
                };
                match key_ty.as_ref() {
                    ArrowDataType::Int32 => {
                        let da = col
                            .as_any()
                            .downcast_ref::<DictionaryArray<Int32Type>>()
                            .ok_or_else(|| {
                                BoltError::Type(
                                    "register_table: dict<i32,utf8> downcast failed".into(),
                                )
                            })?;
                        let sa = da
                            .values()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| {
                                BoltError::Type(
                                    "register_table: dict values are not StringArray".into(),
                                )
                            })?;
                        let keys = da.keys();
                        for i in 0..n {
                            if keys.is_null(i) {
                                out.push(None);
                            } else {
                                // Finding V-5: validate every key before it
                                // indexes the dictionary. A negative or
                                // out-of-range key would make `sa.value(..)`
                                // panic on OOB inside `decode_into`. Reject it
                                // with a clean error instead, mirroring the
                                // strict bounds checks in `string_ops`.
                                let key = keys.value(i);
                                if key < 0 {
                                    return Err(BoltError::Type(format!(
                                        "register_table: negative dict<i32,utf8> key {} at row {}",
                                        key, i
                                    )));
                                }
                                let pos = key as usize;
                                if pos >= sa.len() {
                                    return Err(BoltError::Type(format!(
                                        "register_table: dict<i32,utf8> key {} at row {} out of range (dictionary size {})",
                                        key,
                                        i,
                                        sa.len()
                                    )));
                                }
                                decode_into(&mut out, pos, sa);
                            }
                        }
                    }
                    ArrowDataType::Int64 => {
                        let da = col
                            .as_any()
                            .downcast_ref::<DictionaryArray<Int64Type>>()
                            .ok_or_else(|| {
                                BoltError::Type(
                                    "register_table: dict<i64,utf8> downcast failed".into(),
                                )
                            })?;
                        let sa = da
                            .values()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| {
                                BoltError::Type(
                                    "register_table: dict values are not StringArray".into(),
                                )
                            })?;
                        let keys = da.keys();
                        for i in 0..n {
                            if keys.is_null(i) {
                                out.push(None);
                            } else {
                                // Finding V-5: validate every key before it
                                // indexes the dictionary. The original `as
                                // usize` cast could feed a negative key (after
                                // sign extension) or an out-of-range key to
                                // `sa.value(..)`, panicking on OOB. Reject
                                // negative, out-of-range, and (for parity with
                                // the upload path's i32 device buffer) keys
                                // above `i32::MAX`.
                                let key = keys.value(i);
                                if key < 0 {
                                    return Err(BoltError::Type(format!(
                                        "register_table: negative dict<i64,utf8> key {} at row {}",
                                        key, i
                                    )));
                                }
                                if key > i32::MAX as i64 {
                                    return Err(BoltError::Type(format!(
                                        "register_table: dict<i64,utf8> key {} at row {} exceeds i32 capacity",
                                        key, i
                                    )));
                                }
                                let pos = key as usize;
                                if pos >= sa.len() {
                                    return Err(BoltError::Type(format!(
                                        "register_table: dict<i64,utf8> key {} at row {} out of range (dictionary size {})",
                                        key,
                                        i,
                                        sa.len()
                                    )));
                                }
                                decode_into(&mut out, pos, sa);
                            }
                        }
                    }
                    other => {
                        return Err(BoltError::Type(format!(
                            "register_table: dict key type {:?} not supported \
                             (expected Int32 or Int64 with Utf8 values)",
                            other
                        )));
                    }
                }
                let sa = StringArray::from(out);
                new_fields.push(ArrowField::new(
                    field.name().clone(),
                    ArrowDataType::Utf8,
                    field.is_nullable(),
                ));
                new_cols.push(Arc::new(sa) as ArrayRef);
                changed = true;
            }
            _ => {
                new_fields.push(field.as_ref().clone());
                new_cols.push(col.clone());
            }
        }
    }
    if !changed {
        return Ok(batch);
    }
    let new_schema = Arc::new(ArrowSchema::new(new_fields));
    RecordBatch::try_new(new_schema, new_cols)
        .map_err(|e| BoltError::Type(format!("register_table: rebuild after dict flatten failed: {e}")))
}

/// Parse the `BOLT_POOL_STATS_INTERVAL_SECS` environment variable into
/// a `Duration`. Missing or unparseable values default to
/// `DEFAULT_POOL_STATS_INTERVAL_SECS`; an explicit `0` disables
/// periodic emission (signalled by `Duration::ZERO`).
///
/// `pub` so the integration test `tests/env_var_smoke.rs` can
/// round-trip the parser against the live env var without going
/// through `Engine::new` (which would also pay an eager CUDA-context
/// init cost we want to keep off host-only smoke runs).
pub fn pool_stats_interval_from_env() -> Duration {
    match std::env::var(crate::exec::engine::POOL_STATS_ENV)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        Some(0) => Duration::ZERO,
        Some(n) => Duration::from_secs(n),
        None => Duration::from_secs(crate::exec::engine::DEFAULT_POOL_STATS_INTERVAL_SECS),
    }
}

/// Install (or clear) the builder's `persistent_cache(path)` directory on
/// the process-wide disk PTX cache.
///
/// This is the single bridge between [`EngineBuilder::persistent_cache`]
/// and the JIT compile path's disk-cache lookup
/// ([`Engine::get_or_build_module`] → [`crate::jit::disk_cache::disk_cache`]).
/// Pulled out of [`EngineBuilder::build`] as a free function so the
/// builder → cache-layer plumbing can be exercised host-side without a
/// live CUDA context (the rest of `build` needs one).
///
/// Semantics, mirroring [`crate::jit::disk_cache::set_override_dir`]:
///   * `Some(path)` — subsequent `disk_cache()` lookups resolve to
///     `path` regardless of the `BOLT_PTX_CACHE_DIR` env var (the
///     builder knob takes precedence; the env var remains the fallback
///     when no builder path is configured).
///   * `None` — clears any prior builder override so the cache
///     re-falls-back to the env var, and stays disabled if that too is
///     unset. This preserves the opt-in "no path → unchanged behaviour"
///     contract: a default-built engine never enables the disk cache on
///     its own.
pub(crate) fn install_persistent_cache_override(path: Option<&std::path::Path>) {
    crate::jit::disk_cache::set_override_dir(path.map(|p| p.to_path_buf()));
}

/// Identity-passthrough analysis for the GPU-projection host fallback
/// ([`Engine::execute_projection_host_fallback`]).
///
/// Returns `Some(out_src)` — where `out_src[output_col_idx] = input_col_idx` —
/// IFF `kernel` is a pure passthrough: no predicate, and its `ops` are exactly
/// `LoadColumn`→`Store` (or 128-bit `LoadColumn128`→`Store128`) pairs that
/// route each input column straight to an output column with no computation.
/// Returns `None` for any other shape (a predicate, or any compute / cast /
/// select / 128-bit-arithmetic op), signalling the caller to re-raise the GPU
/// decline rather than risk a wrong host result. Pulled out as a free function
/// so the passthrough detection is unit-testable without a CUDA context.
pub(crate) fn passthrough_output_sources(kernel: &KernelSpec) -> Option<Vec<usize>> {
    use crate::plan::physical_plan::{Op, Reg};

    // A predicate means rows are filtered — not a pure passthrough.
    if kernel.predicate.is_some() {
        return None;
    }

    // Register → source-column maps for the single and 128-bit load classes.
    let mut loaded: HashMap<Reg, usize> = HashMap::new();
    let mut loaded128: HashMap<(Reg, Reg), usize> = HashMap::new();
    // output_col_idx → input_col_idx, recorded at each Store.
    let mut out_src: HashMap<usize, usize> = HashMap::new();

    for op in &kernel.ops {
        match op {
            Op::LoadColumn { dst, col_idx, .. } => {
                loaded.insert(*dst, *col_idx);
            }
            Op::LoadColumn128 {
                dst_lo,
                dst_hi,
                col_idx,
            } => {
                loaded128.insert((*dst_lo, *dst_hi), *col_idx);
            }
            Op::Store { src, col_idx, .. } => {
                let in_idx = loaded.get(src).copied()?;
                out_src.insert(*col_idx, in_idx);
            }
            Op::Store128 {
                src_lo,
                src_hi,
                col_idx,
            } => {
                let in_idx = loaded128.get(&(*src_lo, *src_hi)).copied()?;
                out_src.insert(*col_idx, in_idx);
            }
            // Any compute / cast / select / 128-bit arithmetic op disqualifies
            // the passthrough fast path.
            _ => return None,
        }
    }

    // Every output must be covered by exactly one passthrough store.
    if out_src.len() != kernel.outputs.len() {
        return None;
    }
    let mut mapping = Vec::with_capacity(kernel.outputs.len());
    for out_idx in 0..kernel.outputs.len() {
        mapping.push(*out_src.get(&out_idx)?);
    }
    Some(mapping)
}

/// Decide whether to emit a pool-stats snapshot at time `now`, advancing
/// the throttle state on a positive decision.
///
/// Pulled out of [`Engine::maybe_emit_pool_stats`] so the throttle
/// semantics can be exercised without a live CUDA context. Side
/// effects: writes `Some(now)` into `last_emit` when emission is due,
/// leaves it untouched otherwise.
///
/// Returns `true` IFF the caller should emit a log line + observer
/// notification right now. Encapsulates three rules:
///   * `interval == 0` → never emit (env-var disables).
///   * `last_emit.is_none()` → always emit (first query on the engine).
///   * `now - last_emit >= interval` → emit and reset.
pub(crate) fn should_emit_pool_stats(
    last_emit: &Mutex<Option<Instant>>,
    interval: Duration,
    now: Instant,
) -> bool {
    if interval.is_zero() {
        return false;
    }
    let mut last = match last_emit.lock() {
        Ok(g) => g,
        Err(_) => return false, // poisoned — best-effort; skip the emit.
    };
    let should = match *last {
        None => true,
        Some(prev) => now.duration_since(prev) >= interval,
    };
    if should {
        *last = Some(now);
    }
    should
}

/// Concatenate a table's host-side batches into a single `RecordBatch`.
///
/// Shared by [`Engine::materialize_table`]'s eager and streaming-overlay
/// paths. Zero batches errors, one batch is cloned cheaply (Arrow arrays are
/// `Arc`-backed), two or more go through `arrow::compute::concat_batches`
/// (which copies every column — the perf cost the field doc on `tables`
/// warns about).
pub(crate) fn concat_table_batches(name: &str, batches: &[RecordBatch]) -> BoltResult<RecordBatch> {
    match batches.len() {
        0 => Err(BoltError::Plan(format!(
            "table '{name}' is registered but contains zero batches"
        ))),
        1 => Ok(batches[0].clone()),
        _ => {
            let schema = batches[0].schema();
            arrow::compute::concat_batches(&schema, batches.iter()).map_err(|e| {
                BoltError::Other(format!(
                    "failed to concatenate {} batches for table '{name}': {e}",
                    batches.len()
                ))
            })
        }
    }
}

/// Stage 6 — walk `batch` and inform `provider` of each column's actual
/// runtime nullability (i.e. whether the source array had any nulls). For
/// `DictionaryArray<_, Utf8>` columns the per-row nullability lives on the
/// keys buffer, not the dictionary values; this helper consults
/// `keys().null_count()` to get the right answer. Called from
/// `register_table` / `replace_table` / `register_batch`, so the
/// engine-backed `TableProvider` (`EngineProvider::has_nulls`) and the
/// codegen-time `populate_input_validity` pass both see truthful claims.
pub(crate) fn propagate_column_nullability(
    provider: &mut MemTableProvider,
    table: &str,
    batch: &RecordBatch,
) {
    // `Array::null_count` is an inherent-trait method; pull the trait into
    // scope locally so we can ask any `&dyn Array` for its null count.
    use arrow_array::Array;
    let schema = batch.schema();
    for (idx, field) in schema.fields().iter().enumerate() {
        let arr = batch.column(idx);
        let has_nulls = match field.data_type() {
            ArrowDataType::Dictionary(key_t, _)
                if key_t.as_ref() == &ArrowDataType::Int32 =>
            {
                // Dict keys carry the per-row validity. Downcast and ask the
                // keys array directly; fall back to the array's own
                // `null_count()` if the downcast fails (shouldn't happen for
                // Int32 keys but defensive).
                match arr
                    .as_any()
                    .downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::Int32Type>>()
                {
                    Some(da) => da.keys().null_count() > 0,
                    None => arr.null_count() > 0,
                }
            }
            _ => arr.null_count() > 0,
        };
        provider.set_column_nullability(table.to_string(), field.name().clone(), has_nulls);
    }
}

/// Convert an `arrow_schema::Schema` into our plan `Schema`.
pub(crate) fn arrow_schema_to_plan_schema(s: &ArrowSchema) -> BoltResult<Schema> {
    let mut fields = Vec::with_capacity(s.fields().len());
    for f in s.fields() {
        let dt = arrow_dtype_to_plan(f.data_type())?;
        fields.push(Field::new(f.name().clone(), dt, f.is_nullable()));
    }
    Ok(Schema::new(fields))
}

/// Convert our plan `Schema` to an `arrow_schema::Schema` (used for output `RecordBatch`).
pub(crate) fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    crate::exec::schema_convert::plan_schema_to_arrow_schema(s)
}

/// Build the single-row `Int64` output batch for a `PhysicalPlan::CountRows`
/// node: one column holding `n_rows` (the materialised child plan's row
/// count). `output_schema` must describe exactly one column (the count); its
/// Arrow shape comes from `plan_schema_to_arrow_schema`, so the column name /
/// nullability follow whatever the lowerer stored (a single Int64 field).
///
/// Factored out of the `execute` arm so the host-side row-count step is unit
/// testable without a GPU / engine.
pub(crate) fn build_count_rows_batch(n_rows: usize, output_schema: &Schema) -> BoltResult<RecordBatch> {
    if output_schema.fields.len() != 1 {
        return Err(BoltError::Plan(format!(
            "CountRows output schema must have exactly one column, got {}",
            output_schema.fields.len()
        )));
    }
    let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
    let arr: ArrayRef = Arc::new(Int64Array::from(vec![n_rows as i64]));
    RecordBatch::try_new(arrow_schema, vec![arr]).map_err(|e| {
        BoltError::Other(format!("failed to build CountRows RecordBatch: {e}"))
    })
}

/// Convert a host-side computed `HostColumn` into an `ArrayRef`.
///
/// Used by the `PhysicalPlan::Project` compute path (string `||`,
/// arithmetic over post-aggregate scalars, …) to fold a freshly
/// materialised column back into the output `RecordBatch`. Mirrors the
/// `arrow_array_to_host_column` shape in `filter.rs` (the inverse
/// direction).
pub(crate) fn host_column_to_arrow_array(col: crate::exec::expr_agg::HostColumn) -> BoltResult<ArrayRef> {
    use crate::exec::expr_agg::HostColumn;
    Ok(match col {
        HostColumn::Bool(v) => Arc::new(BooleanArray::from(v)) as ArrayRef,
        HostColumn::I32(v) => Arc::new(Int32Array::from(v)) as ArrayRef,
        HostColumn::I64(v) => Arc::new(Int64Array::from(v)) as ArrayRef,
        HostColumn::F32(v) => Arc::new(Float32Array::from(v)) as ArrayRef,
        HostColumn::F64(v) => Arc::new(Float64Array::from(v)) as ArrayRef,
        HostColumn::Utf8(v) => {
            let arr = arrow_array::StringArray::from(v);
            Arc::new(arr) as ArrayRef
        }
    })
}
