// SPDX-License-Identifier: Apache-2.0

//! String-projection execution methods for [`Engine`].
//!
//! Pure-reorg split of the former monolithic `engine.rs`: these
//! `impl Engine` methods were moved here verbatim to keep the parent file
//! navigable. No behaviour change — the three executor entry points
//! (`execute_string_length` / `execute_string_project` /
//! `execute_string_like_filter`) had their visibility widened to `pub(crate)`
//! so the top-level dispatch in `engine.rs` can call them across the module
//! boundary; the per-column helpers stay private to this module.
//!
//! The cluster covers the `StringLength` / `StringProject` (`UPPER`/`LOWER`/
//! `SUBSTRING`/`TRIM`/`CONCAT`/`CASE`) and `StringLikeFilter` executors plus
//! their GPU-gather / host-mirror column helpers.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::DataType as ArrowDataType;

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::engine::{Engine, QueryHandle, BLOCK_SIZE};
use crate::exec::engine_device_col::check_len;
use crate::exec::engine_support::{debug_sync_check, plan_schema_to_arrow_schema};
use crate::exec::launch::{grid_x_for, CudaStream};
use crate::exec::n_rows_to_u32;
use crate::jit::CudaModule;
use crate::plan::{PhysicalPlan, Schema};

impl Engine {
    /// Execute a [`PhysicalPlan::StringLength`]: a `SELECT LENGTH(<utf8_col>)`
    /// projection (plus passthrough columns) over a bare scan, with the
    /// `LENGTH` outputs computed on the GPU via the dictionary-index gather
    /// kernel ([`crate::jit::string_kernel::compile_length_gather_kernel`]).
    ///
    /// Passthrough columns are lifted directly from the host-side source batch
    /// (zero-copy `ArrayRef` clone). Each `LENGTH(col)` output runs the gather
    /// against the GPU-resident dictionary column when it is dictionary-encoded
    /// (and, for the native `DictUtf8` layout, null-free); otherwise it falls
    /// back to a host-side gather over the downloaded keys (see
    /// [`crate::exec::string_length`]). Both paths produce an `Int64Array`
    /// matching the logical-plan `LENGTH` output dtype.
    pub(crate) fn execute_string_length(
        &self,
        table: &str,
        outputs: &[crate::plan::physical_plan::StringLengthOutput],
        output_schema: &Schema,
    ) -> BoltResult<QueryHandle> {
        use crate::plan::physical_plan::StringLengthOutput;

        // Source host batch — used for passthrough columns (and as the row
        // count authority so an empty / partial table still works).
        let src_batch = self.materialize_table(table)?;
        let src_schema = src_batch.schema();
        let n_rows = src_batch.num_rows();

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(outputs.len());
        for out in outputs {
            match out {
                StringLengthOutput::Passthrough { source } => {
                    let idx = src_schema.index_of(source).map_err(|_| {
                        BoltError::Plan(format!(
                            "StringLength: passthrough column '{source}' not found in \
                             table '{table}'"
                        ))
                    })?;
                    arrays.push(src_batch.column(idx).clone());
                }
                StringLengthOutput::Length { source } => {
                    arrays.push(self.string_length_column(table, source, n_rows)?);
                }
            }
        }

        let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
        let batch_out = RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
            BoltError::Other(format!(
                "StringLength: failed to build output RecordBatch: {e}"
            ))
        })?;
        Ok(QueryHandle { batch: batch_out })
    }

    /// Compute `LENGTH(<source>)` for the GPU-resident `Utf8` column `source`
    /// of `table`, returning an `Int64Array` of `n_rows` rows.
    ///
    /// GPU path: build the per-dictionary-entry `i32` length table that matches
    /// the column's device key layout, upload it, launch the gather kernel
    /// (`out[row] = length_table[keys[row]]`), download the `Int32` result, and
    /// widen to `Int64`. When the column is not safe to gather on the GPU
    /// (non-dict storage, or a `DictUtf8` column with NULLs — whose zeroed keys
    /// would gather the wrong slot), fall back to a host-side gather over the
    /// downloaded keys, which is byte-for-byte identical for the supported case.
    fn string_length_column(
        &self,
        table: &str,
        source: &str,
        n_rows: usize,
    ) -> BoltResult<ArrayRef> {
        use crate::exec::string_length::{
            build_length_table, gpu_gather_layout, host_gather_lengths, KeyLayout,
        };

        let gpu_table_ref = self.ensure_gpu_table(table)?;
        let gpu_table: &crate::exec::gpu_table::GpuTable = &gpu_table_ref;
        let column = gpu_table.column(source).ok_or_else(|| {
            BoltError::Plan(format!(
                "StringLength: column '{source}' not in GPU table '{table}'"
            ))
        })?;

        // Resolve the host-side dictionary + device key buffer + layout for
        // this column. `None` layout ⇒ host fallback.
        let dict = column.utf8_dictionary().ok_or_else(|| {
            BoltError::Plan(format!(
                "StringLength: column '{source}' is not a Utf8 column (LENGTH requires Utf8)"
            ))
        })?;
        let (keys_vec, layout): (&GpuVec<i32>, Option<KeyLayout>) = match &column.data {
            crate::exec::gpu_table::GpuColumnData::Utf8 { indices, .. } => {
                (indices, gpu_gather_layout(&column.data))
            }
            crate::exec::gpu_table::GpuColumnData::DictUtf8 { keys, .. } => {
                (keys, gpu_gather_layout(&column.data))
            }
            _ => {
                return Err(BoltError::Plan(format!(
                    "StringLength: column '{source}' has non-Utf8 GPU storage"
                )))
            }
        };

        let layout = match layout {
            Some(l) => l,
            None => {
                // Host fallback: download keys and gather over the 1-based
                // NULL-sentinel table. A NULL input row emits SQL NULL (a
                // validity-carrying `None`), distinct from `LENGTH('') = 0` —
                // matching the now-NULL-correct `exec::string_ops::length`
                // (agent-C F-3). Valid rows map to `table[key+1]`.
                let table_lengths = build_length_table(dict, KeyLayout::OneBasedNullSlot0)?;
                let keys_host = keys_vec.to_vec()?;
                // DictUtf8 keys are 0-based; remap to the 1-based table by
                // adding 1 only when the column is the DictUtf8 layout.
                let lens: Vec<Option<i64>> = match &column.data {
                    crate::exec::gpu_table::GpuColumnData::DictUtf8 { valid_mask, .. } => {
                        // Consult validity: NULL rows → SQL NULL, valid rows →
                        // table[key+1].
                        let mask = valid_mask.as_ref().map(|m| m.to_vec()).transpose()?;
                        let mut out: Vec<Option<i64>> = Vec::with_capacity(keys_host.len());
                        for (row, &k) in keys_host.iter().enumerate() {
                            let is_valid = match &mask {
                                None => true,
                                Some(bits) => {
                                    let byte = bits.get(row / 8).copied().unwrap_or(0);
                                    (byte >> (row % 8)) & 1 == 1
                                }
                            };
                            if !is_valid {
                                // SQL NULL, NOT length 0.
                                out.push(None);
                            } else if k < 0 {
                                return Err(BoltError::Other(format!(
                                    "LENGTH: negative dictionary key {k}"
                                )));
                            } else {
                                // table index = key + 1 (slot 0 is NULL).
                                let len = *table_lengths.get(k as usize + 1).ok_or_else(|| {
                                    BoltError::Other(format!("LENGTH: key {k} out of range"))
                                })?;
                                out.push(Some(len as i64));
                            }
                        }
                        out
                    }
                    // Non-DictUtf8 host gather: no per-row validity bitmap at
                    // this layer, so every gathered length is non-NULL.
                    _ => host_gather_lengths(&keys_host, &table_lengths)?
                        .into_iter()
                        .map(Some)
                        .collect(),
                };
                check_len(lens.len(), n_rows)?;
                // `Int64Array::from(Vec<Option<i64>>)` carries the validity
                // bitmap, so NULL rows decode back to SQL NULL.
                return Ok(Arc::new(Int64Array::from(lens)) as ArrayRef);
            }
        };

        // GPU gather path.
        let length_table = build_length_table(dict, layout)?;
        let table_gpu = GpuVec::<i32>::from_slice(&length_table)?;
        let out_gpu = GpuVec::<i32>::zeros(n_rows)?;

        let module =
            CudaModule::from_ptx(&crate::jit::string_kernel::compile_length_gather_kernel()?)?;
        let function = module.function(crate::jit::string_kernel::LENGTH_GATHER_ENTRY)?;

        // ABI: (indices, length_table, out, n_rows). Assemble raw kernel
        // params directly (heterogeneous list; same pattern as
        // `execute_projection`).
        let mut indices_ptr = keys_vec.device_ptr();
        let mut table_ptr = table_gpu.device_ptr();
        let mut out_ptr = out_gpu.device_ptr();
        let mut n_rows_u32 = n_rows_to_u32(n_rows)?;
        let mut kernel_params: Vec<*mut c_void> = vec![
            &mut indices_ptr as *mut CUdeviceptr as *mut c_void,
            &mut table_ptr as *mut CUdeviceptr as *mut c_void,
            &mut out_ptr as *mut CUdeviceptr as *mut c_void,
            &mut n_rows_u32 as *mut u32 as *mut c_void,
        ];

        let stream = CudaStream::null_or_default();
        let grid_x = grid_x_for(n_rows_u32, BLOCK_SIZE);
        unsafe {
            cuda_sys::check(cuda_sys::cuLaunchKernel(
                function.raw(),
                grid_x,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                stream.raw(),
                kernel_params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }
        debug_sync_check()?;
        stream.synchronize()?;

        // Download Int32 lengths and widen to the Int64 SQL contract.
        let lens_i32 = out_gpu.to_vec()?;
        check_len(lens_i32.len(), n_rows)?;

        // Derive the output validity bitmap from the SOURCE column's NULLs so
        // `LENGTH(NULL)` is SQL NULL, not a valid `0`. The kernel gathered a
        // bare length for every row (a NULL row gathered length-table slot 0 =
        // 0), so without re-applying validity here a NULL input would read back
        // as a valid `0`. `LENGTH('')` is unaffected: an empty string is a real
        // dictionary entry whose row is VALID, so it stays a valid `0`.
        //
        // * `OneBasedNullSlot0` (engine-managed `Utf8`): NULL ⇔ key == 0 (the
        //   reserved NULL sentinel slot). Real strings — including `""` — have
        //   key >= 1 and are valid.
        // * `ZeroBased` (`DictUtf8`): `gpu_gather_layout` only selects the GPU
        //   path when `valid_mask` is `None`, i.e. there are no NULLs, so every
        //   row is valid.
        let lens_opt: Vec<Option<i64>> = match layout {
            KeyLayout::OneBasedNullSlot0 => {
                let keys_host = keys_vec.to_vec()?;
                check_len(keys_host.len(), n_rows)?;
                lens_i32
                    .into_iter()
                    .zip(keys_host.into_iter())
                    .map(|(len, key)| if key == 0 { None } else { Some(len as i64) })
                    .collect()
            }
            // No NULLs on this path (see above): every length is valid.
            KeyLayout::ZeroBased => lens_i32.into_iter().map(|v| Some(v as i64)).collect(),
        };
        // `Int64Array::from(Vec<Option<i64>>)` carries the validity bitmap, so
        // NULL rows decode back to SQL NULL.
        Ok(Arc::new(Int64Array::from(lens_opt)) as ArrayRef)
    }

    /// Execute a [`PhysicalPlan::StringProject`]: a `SELECT UPPER(<utf8_col>)` /
    /// `LOWER(<utf8_col>)` projection (plus passthrough columns) over a bare
    /// scan, with the transform outputs produced on the GPU via the two-pass
    /// length/scan/write kernels in [`crate::jit::string_kernel`] (see
    /// [`crate::exec::string_project`]).
    ///
    /// Passthrough columns are lifted directly from the host source batch.
    /// Each `UPPER`/`LOWER(col)` output runs the two-pass GPU producer against a
    /// row-aligned offsets+bytes input materialised from the dictionary-encoded
    /// column — but only when the column's dictionary is pure ASCII (the kernels
    /// ASCII-fold byte-wise; non-ASCII Unicode case mapping can change byte
    /// length, e.g. `'ß'` → `"SS"`). Non-ASCII dictionaries, or columns with no
    /// supported GPU storage, fall back to a full-Unicode host transform. Both
    /// paths produce a `StringArray`.
    pub(crate) fn execute_string_project(
        &self,
        table: &str,
        outputs: &[crate::plan::physical_plan::StringProjectOutput],
        output_schema: &Schema,
    ) -> BoltResult<QueryHandle> {
        use crate::plan::physical_plan::StringProjectOutput;

        let src_batch = self.materialize_table(table)?;
        let src_schema = src_batch.schema();
        let n_rows = src_batch.num_rows();

        // Lazily-built host env (decoded source columns as `HostColumn`s) for
        // the `CaseUtf8` output path; `None` until the first CASE forces the
        // lift. Mirrors the `PhysicalPlan::Project` compute path in `execute`.
        let mut owned_env: Option<Vec<(String, crate::exec::expr_agg::HostColumn)>> = None;

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(outputs.len());
        for out in outputs {
            match out {
                StringProjectOutput::Passthrough { source } => {
                    let idx = src_schema.index_of(source).map_err(|_| {
                        BoltError::Plan(format!(
                            "StringProject: passthrough column '{source}' not found in \
                             table '{table}'"
                        ))
                    })?;
                    let src_col = src_batch.column(idx);
                    // A dictionary-encoded Utf8 column is stored as
                    // `Dictionary(Int32, Utf8)` on the host but projects as
                    // logical `Utf8` (the output schema declares `Utf8`).
                    // Decode it to a plain Utf8 array so the built batch matches
                    // the schema; non-dictionary columns pass through unchanged.
                    if matches!(src_col.data_type(), ArrowDataType::Dictionary(_, _)) {
                        let decoded = arrow::compute::cast(src_col.as_ref(), &ArrowDataType::Utf8)
                            .map_err(|e| {
                                BoltError::Other(format!(
                                "StringProject: decode dictionary '{source}' to Utf8 failed: {e}"
                            ))
                            })?;
                        arrays.push(decoded);
                    } else {
                        arrays.push(src_col.clone());
                    }
                }
                StringProjectOutput::Transform { source, transform } => {
                    arrays.push(self.string_transform_column(table, source, *transform, n_rows)?);
                }
                StringProjectOutput::Concat { sources } => {
                    arrays.push(self.string_concat_column(table, sources, n_rows)?);
                }
                StringProjectOutput::CaseUtf8 {
                    branches,
                    else_branch,
                } => {
                    // Build the host env (decoded source columns) once, lazily.
                    // Dictionary-encoded Utf8 columns are decoded to a plain
                    // Utf8 array first (`arrow_array_to_host_column` has no
                    // Dictionary arm), mirroring the Passthrough decode above.
                    if owned_env.is_none() {
                        let mut v = Vec::with_capacity(src_batch.num_columns());
                        for (i, field) in src_schema.fields().iter().enumerate() {
                            let arr = src_batch.column(i);
                            let decoded: ArrayRef =
                                if matches!(arr.data_type(), ArrowDataType::Dictionary(_, _)) {
                                    arrow::compute::cast(arr.as_ref(), &ArrowDataType::Utf8)
                                        .map_err(|e| {
                                            BoltError::Other(format!(
                                                "StringProject(CaseUtf8): decode dictionary \
                                             '{}' to Utf8 failed: {e}",
                                                field.name()
                                            ))
                                        })?
                                } else {
                                    arr.clone()
                                };
                            let hc = crate::exec::filter::arrow_array_to_host_column(
                                decoded.as_ref(),
                                n_rows,
                            )?;
                            v.push((field.name().clone(), hc));
                        }
                        owned_env = Some(v);
                    }
                    let env_ref = owned_env.as_ref().expect("just built");
                    let env: crate::exec::expr_agg::ColumnEnv<'_> =
                        env_ref.iter().map(|(n, c)| (n.clone(), c)).collect();
                    let arr = crate::exec::string_project::eval_case_utf8(
                        branches,
                        else_branch.as_deref(),
                        &env,
                        n_rows,
                    )?;
                    arrays.push(Arc::new(arr) as ArrayRef);
                }
            }
        }

        let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
        let batch_out = RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
            BoltError::Other(format!(
                "StringProject: failed to build output RecordBatch: {e}"
            ))
        })?;
        Ok(QueryHandle { batch: batch_out })
    }

    /// Compute a [`StringTransform`](crate::exec::string_project::StringTransform)
    /// — `UPPER`/`LOWER`/`SUBSTRING`/`TRIM` — of the GPU-resident `Utf8` column
    /// `source` of `table`, returning a `Utf8` `ArrayRef` of `n_rows` rows.
    ///
    /// `SUBSTRING`/`TRIM` are realised via the byte-identical host mirror
    /// ([`crate::exec::string_project::host_transform_strings`]) regardless of
    /// the dictionary contents — their GPU two-pass producers exist but are
    /// unvalidated on hardware (matching the CONCAT path).
    ///
    /// For `UPPER`/`LOWER`:
    /// GPU path (ASCII dictionaries): materialise a row-aligned offsets+bytes
    /// input from the column's dictionary + device keys, upload, run the length
    /// pass → host exclusive scan of `row_lens` → allocate output bytes → run
    /// the write pass → download → rebuild the `StringArray` (re-applying NULLs).
    /// Host fallback (non-ASCII dictionary, or unsupported GPU storage): apply
    /// the full-Unicode transform host-side. Both paths preserve NULLs as Arrow
    /// NULLs.
    fn string_transform_column(
        &self,
        table: &str,
        source: &str,
        transform: crate::exec::string_project::StringTransform,
        n_rows: usize,
    ) -> BoltResult<ArrayRef> {
        use crate::exec::string_project::{
            build_row_aligned_input, dict_is_ascii, exclusive_scan_lens, host_transform_strings,
            string_array_from_offsets, KeyLayout,
        };

        let gpu_table_ref = self.ensure_gpu_table(table)?;
        let gpu_table: &crate::exec::gpu_table::GpuTable = &gpu_table_ref;
        let column = gpu_table.column(source).ok_or_else(|| {
            BoltError::Plan(format!(
                "StringProject: column '{source}' not in GPU table '{table}'"
            ))
        })?;
        let dict = column.utf8_dictionary().ok_or_else(|| {
            BoltError::Plan(format!(
                "StringProject: column '{source}' is not a Utf8 column"
            ))
        })?;

        // Resolve the host-side keys + layout + per-row validity for this
        // column. For the engine-managed `Utf8` layout NULL is encoded as key 0
        // (1-based dict); for native `DictUtf8` NULL lives on `valid_mask`.
        let (keys_host, layout, validity): (Vec<i32>, KeyLayout, Option<Vec<bool>>) =
            match &column.data {
                crate::exec::gpu_table::GpuColumnData::Utf8 { indices, .. } => {
                    let keys = indices.to_vec()?;
                    // Validity = key != 0 (slot 0 is the NULL sentinel).
                    let valid: Vec<bool> = keys.iter().map(|&k| k != 0).collect();
                    (keys, KeyLayout::OneBasedNullSlot0, Some(valid))
                }
                crate::exec::gpu_table::GpuColumnData::DictUtf8 {
                    keys, valid_mask, ..
                } => {
                    let keys = keys.to_vec()?;
                    let valid = match valid_mask {
                        None => None,
                        Some(mask) => {
                            let bits = mask.to_vec()?;
                            let v: Vec<bool> = (0..keys.len())
                                .map(|row| {
                                    let byte = bits.get(row / 8).copied().unwrap_or(0);
                                    (byte >> (row % 8)) & 1 == 1
                                })
                                .collect();
                            Some(v)
                        }
                    };
                    (keys, KeyLayout::ZeroBased, valid)
                }
                _ => {
                    return Err(BoltError::Plan(format!(
                        "StringProject: column '{source}' has non-Utf8 GPU storage"
                    )))
                }
            };

        check_len(keys_host.len(), n_rows)?;
        let validity_slice = validity.as_deref();

        // SUBSTRING / TRIM are realised host-side (byte-identical to the host
        // helpers in `string_ops_extended`). The GPU two-pass producers for
        // these exist in `jit::string_kernel` and are PTX-shape-tested, but are
        // unvalidated on hardware (like CONCAT / LIKE), so we take the
        // correctness-guaranteed host path here. Results are identical either
        // way; wiring the device launch is a follow-up.
        if transform.is_host_realized() {
            let arr = host_transform_strings(dict, &keys_host, layout, validity_slice, transform)?;
            return Ok(Arc::new(arr) as ArrayRef);
        }

        // Host fallback for non-ASCII dictionaries: the byte-wise GPU fold is
        // only correct for ASCII (Unicode case mapping can change byte length).
        if !dict_is_ascii(dict) {
            let arr = host_transform_strings(dict, &keys_host, layout, validity_slice, transform)?;
            return Ok(Arc::new(arr) as ArrayRef);
        }

        // Gate: the GPU two-pass UPPER/LOWER device path is UNVALIDATED. By
        // default (gate OFF) take the validated host mirror, which is
        // byte-identical for ASCII dictionaries. The device launch below is
        // only reached when `BOLT_GPU_STRING` is truthy.
        if !crate::exec::string_project::gpu_string_enabled() {
            let arr = host_transform_strings(dict, &keys_host, layout, validity_slice, transform)?;
            return Ok(Arc::new(arr) as ArrayRef);
        }

        // ---- GPU two-pass path -------------------------------------------
        // Pass 0 (host): materialise the row-aligned offsets+bytes input.
        let (src_offsets, src_bytes) =
            build_row_aligned_input(dict, &keys_host, layout, validity_slice)?;

        // Empty input (no rows, or all-empty bytes): skip the launch and build
        // the result directly. `from_slice` on an empty slice is brittle and a
        // zero-thread launch is pointless.
        if n_rows == 0 {
            let arr = string_array_from_offsets(&src_offsets, &src_bytes, validity_slice)?;
            return Ok(Arc::new(arr) as ArrayRef);
        }

        let kind = transform.scalar_fn_kind();
        let src_offsets_gpu = GpuVec::<i32>::from_slice(&src_offsets)?;
        // `src_bytes` may be empty (all rows empty/NULL); allocate at least one
        // byte so the device pointer is valid even though no thread reads it.
        let src_bytes_gpu = if src_bytes.is_empty() {
            GpuVec::<u8>::zeros(1)?
        } else {
            GpuVec::<u8>::from_slice(&src_bytes)?
        };
        let row_lens_gpu = GpuVec::<u32>::zeros(n_rows)?;

        let n_rows_u32 = n_rows_to_u32(n_rows)?;
        let stream = CudaStream::null_or_default();
        let grid_x = grid_x_for(n_rows_u32, BLOCK_SIZE);

        // ---- Pass 1: length pass → row_lens. ABI (UPPER/LOWER, 4 params):
        //      (src_offsets, src_bytes, row_lens, n_rows).
        {
            let module =
                CudaModule::from_ptx(&crate::jit::string_kernel::compile_varwidth_len_pass(kind)?)?;
            let entry = crate::jit::string_kernel::len_pass_entry(kind)?;
            let function = module.function(&entry)?;

            let mut p_off = src_offsets_gpu.device_ptr();
            let mut p_bytes = src_bytes_gpu.device_ptr();
            let mut p_lens = row_lens_gpu.device_ptr();
            let mut p_n = n_rows_u32;
            let mut params: Vec<*mut c_void> = vec![
                &mut p_off as *mut CUdeviceptr as *mut c_void,
                &mut p_bytes as *mut CUdeviceptr as *mut c_void,
                &mut p_lens as *mut CUdeviceptr as *mut c_void,
                &mut p_n as *mut u32 as *mut c_void,
            ];
            unsafe {
                cuda_sys::check(cuda_sys::cuLaunchKernel(
                    function.raw(),
                    grid_x,
                    1,
                    1,
                    BLOCK_SIZE,
                    1,
                    1,
                    0,
                    stream.raw(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ))?;
            }
            debug_sync_check()?;
            stream.synchronize()?;
        }

        // ---- Pass 2 (host): exclusive-scan row_lens → out_offsets + total.
        let row_lens = row_lens_gpu.to_vec()?;
        check_len(row_lens.len(), n_rows)?;
        let (out_offsets, total_bytes) = exclusive_scan_lens(&row_lens)?;
        let out_offsets_gpu = GpuVec::<i32>::from_slice(&out_offsets)?;
        let out_bytes_gpu = if total_bytes == 0 {
            GpuVec::<u8>::zeros(1)?
        } else {
            GpuVec::<u8>::zeros(total_bytes)?
        };

        // ---- Pass 3: write pass → out_bytes. ABI (UPPER/LOWER, 5 params):
        //      (src_offsets, src_bytes, out_offsets, out_bytes, n_rows).
        {
            let module = CudaModule::from_ptx(
                &crate::jit::string_kernel::compile_varwidth_write_pass(kind)?,
            )?;
            let entry = crate::jit::string_kernel::write_pass_entry(kind)?;
            let function = module.function(&entry)?;

            let mut p_off = src_offsets_gpu.device_ptr();
            let mut p_bytes = src_bytes_gpu.device_ptr();
            let mut p_out_off = out_offsets_gpu.device_ptr();
            let mut p_out_bytes = out_bytes_gpu.device_ptr();
            let mut p_n = n_rows_u32;
            let mut params: Vec<*mut c_void> = vec![
                &mut p_off as *mut CUdeviceptr as *mut c_void,
                &mut p_bytes as *mut CUdeviceptr as *mut c_void,
                &mut p_out_off as *mut CUdeviceptr as *mut c_void,
                &mut p_out_bytes as *mut CUdeviceptr as *mut c_void,
                &mut p_n as *mut u32 as *mut c_void,
            ];
            unsafe {
                cuda_sys::check(cuda_sys::cuLaunchKernel(
                    function.raw(),
                    grid_x,
                    1,
                    1,
                    BLOCK_SIZE,
                    1,
                    1,
                    0,
                    stream.raw(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ))?;
            }
            debug_sync_check()?;
            stream.synchronize()?;
        }

        // ---- Download + rebuild StringArray (re-applying NULLs).
        let out_bytes = out_bytes_gpu.to_vec()?;
        // `out_bytes_gpu` was padded to >= 1 byte; truncate to the real total.
        let out_bytes = &out_bytes[..total_bytes.min(out_bytes.len())];
        let arr = string_array_from_offsets(&out_offsets, out_bytes, validity_slice)?;
        Ok(Arc::new(arr) as ArrayRef)
    }

    /// Compute `CONCAT(s0, s1, ...)` for the GPU-resident `Utf8` source columns
    /// of `table`, returning a `Utf8` `ArrayRef` of `n_rows` rows with
    /// NULL-if-any-arg-NULL semantics (standard SQL, matching the host path).
    ///
    /// Resolution of each column's dictionary + device keys + layout + per-row
    /// validity mirrors [`string_transform_column`](Self::string_transform_column).
    /// The N-input two-pass GPU concat kernels
    /// ([`crate::jit::string_kernel::compile_concat_len_pass`] /
    /// `compile_concat_write_pass`) exist and are PTX-shape-tested; this executor
    /// currently realises the result via the byte-identical host mirror
    /// ([`crate::exec::string_project::host_concat_strings`]) so the path is
    /// correctness-guaranteed (the device concat kernel is unvalidated on
    /// hardware, like the LIKE matcher). Wiring the device launch here is a
    /// follow-up; results are identical either way.
    fn string_concat_column(
        &self,
        table: &str,
        sources: &[String],
        n_rows: usize,
    ) -> BoltResult<ArrayRef> {
        use crate::exec::string_project::{build_concat_input, host_concat_strings, KeyLayout};

        let gpu_table_ref = self.ensure_gpu_table(table)?;
        let gpu_table: &crate::exec::gpu_table::GpuTable = &gpu_table_ref;

        let mut inputs = Vec::with_capacity(sources.len());
        for source in sources {
            let column = gpu_table.column(source).ok_or_else(|| {
                BoltError::Plan(format!(
                    "StringProject(Concat): column '{source}' not in GPU table '{table}'"
                ))
            })?;
            let dict = column.utf8_dictionary().ok_or_else(|| {
                BoltError::Plan(format!(
                    "StringProject(Concat): column '{source}' is not a Utf8 column"
                ))
            })?;
            // Same (keys, layout, validity) resolution as `string_transform_column`.
            let (keys_host, layout, validity): (Vec<i32>, KeyLayout, Option<Vec<bool>>) =
                match &column.data {
                    crate::exec::gpu_table::GpuColumnData::Utf8 { indices, .. } => {
                        let keys = indices.to_vec()?;
                        let valid: Vec<bool> = keys.iter().map(|&k| k != 0).collect();
                        (keys, KeyLayout::OneBasedNullSlot0, Some(valid))
                    }
                    crate::exec::gpu_table::GpuColumnData::DictUtf8 {
                        keys, valid_mask, ..
                    } => {
                        let keys = keys.to_vec()?;
                        let valid = match valid_mask {
                            None => None,
                            Some(mask) => {
                                let bits = mask.to_vec()?;
                                let v: Vec<bool> = (0..keys.len())
                                    .map(|row| {
                                        let byte = bits.get(row / 8).copied().unwrap_or(0);
                                        (byte >> (row % 8)) & 1 == 1
                                    })
                                    .collect();
                                Some(v)
                            }
                        };
                        (keys, KeyLayout::ZeroBased, valid)
                    }
                    _ => {
                        return Err(BoltError::Plan(format!(
                            "StringProject(Concat): column '{source}' has non-Utf8 GPU storage"
                        )))
                    }
                };
            check_len(keys_host.len(), n_rows)?;
            inputs.push(build_concat_input(
                dict,
                &keys_host,
                layout,
                validity.as_deref(),
            )?);
        }

        let arr = host_concat_strings(&inputs)?;
        Ok(Arc::new(arr) as ArrayRef)
    }

    /// Execute a [`PhysicalPlan::StringLikeFilter`]: a GPU per-row `LIKE` /
    /// `NOT LIKE` over a non-dictionary `Utf8` column, then materialise the
    /// surviving rows.
    ///
    /// ⚠️ UNVALIDATED DEVICE PATH. The matcher kernel
    /// ([`crate::jit::string_kernel::compile_like_match_kernel`]) has not run on
    /// GPU hardware; correctness is guaranteed by the host mirror in
    /// [`crate::exec::string_like`] and by this executor's clean host fallback.
    ///
    /// Flow: execute `input` (a bare scan → row-aligned source batch); pull the
    /// `column` as a host `StringArray`; build a row-aligned offsets+bytes
    /// buffer + validity; upload; launch the matcher (literal baked as a device
    /// buffer); download the 0/1 mask; re-apply NULL 3VL; `arrow::compute::filter`
    /// every column. If the column is absent / not Utf8 at run time, fall back
    /// to the host `LIKE` over the same `StringArray` (no panic).
    pub(crate) fn execute_string_like_filter(
        &self,
        input: &PhysicalPlan,
        _table: &str,
        column: &str,
        literal: &[u8],
        mode: crate::jit::string_kernel::LikeMode,
        negated: bool,
    ) -> BoltResult<QueryHandle> {
        use arrow_array::Array;

        // Execute the inner scan: this is the row-aligned source batch that
        // carries `column` (the lowering required a bare Scan beneath).
        let batch = self.execute(input)?.into_record_batch();
        let schema = batch.schema();

        // Locate the column; if missing or not a StringArray, fall back to the
        // host LIKE over whatever the column decodes to (no panic). Because the
        // lowering already proved `column` is a Utf8 scan column, the common
        // case is the StringArray downcast succeeding.
        let col_idx = match schema.index_of(column) {
            Ok(i) => i,
            Err(_) => {
                return Err(BoltError::Plan(format!(
                    "StringLikeFilter: column '{column}' not found in input batch"
                )))
            }
        };
        let col_arr = batch.column(col_idx);
        // Normalise to a `StringArray`. The common case is a direct downcast;
        // any other Utf8-compatible layout (e.g. a dictionary array that slipped
        // through un-rewritten) is cast to Utf8 so the path stays host-fallback-
        // safe (no panic, no hard error) for unexpected run-time layouts.
        let owned_cast: ArrayRef;
        let str_arr: &arrow_array::StringArray =
            match col_arr.as_any().downcast_ref::<arrow_array::StringArray>() {
                Some(a) => a,
                None => {
                    owned_cast = arrow::compute::cast(col_arr.as_ref(), &ArrowDataType::Utf8)
                        .map_err(|e| {
                            BoltError::Plan(format!(
                                "StringLikeFilter: column '{column}' is not Utf8 and could \
                             not be cast (got {:?}): {e}",
                                col_arr.data_type()
                            ))
                        })?;
                    owned_cast
                        .as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                        .ok_or_else(|| {
                            BoltError::Plan(format!(
                                "StringLikeFilter: cast of column '{column}' did not yield Utf8"
                            ))
                        })?
                }
            };

        // Build the boolean mask: GPU device path, with a host fallback that
        // produces the identical mask if the launch is not viable.
        //
        // The GPU LIKE matcher is an UNVALIDATED device path; by default
        // (gate OFF) we skip it entirely and take the validated host mirror,
        // which produces the byte-identical mask. The device launch is only
        // attempted when `BOLT_GPU_STRING` is truthy.
        let mask: arrow_array::BooleanArray = if !crate::exec::string_like::gpu_string_enabled() {
            crate::exec::string_like::host_mask_via_mirror(str_arr, literal, mode, negated)
        } else {
            match self.string_like_mask_gpu(str_arr, literal, mode, negated) {
                Ok(m) => m,
                Err(e) => {
                    // Host fallback: evaluate the SAME predicate via the validated
                    // host mirror (equivalent to exec::like::host_like for these
                    // shapes). Correctness is unaffected; only the GPU speedup is
                    // lost. Logged so a hardware bring-up notices.
                    log::warn!(
                        "StringLikeFilter: GPU matcher unavailable ({e}); \
                     falling back to host LIKE for column '{column}'"
                    );
                    crate::exec::string_like::host_mask_via_mirror(str_arr, literal, mode, negated)
                }
            }
        };

        // Apply the mask to every column (NULL mask entries drop the row).
        let filtered: Vec<ArrayRef> = batch
            .columns()
            .iter()
            .map(|c| {
                arrow::compute::filter(c.as_ref(), &mask).map_err(|e| {
                    BoltError::Other(format!("StringLikeFilter: arrow filter failed: {e}"))
                })
            })
            .collect::<BoltResult<Vec<_>>>()?;
        let out = RecordBatch::try_new(batch.schema(), filtered).map_err(|e| {
            BoltError::Other(format!(
                "StringLikeFilter: failed to rebuild RecordBatch: {e}"
            ))
        })?;
        Ok(QueryHandle { batch: out })
    }

    /// GPU per-row LIKE matcher: upload the row-aligned column + literal, launch
    /// [`crate::jit::string_kernel::compile_like_match_kernel`], download the
    /// 0/1 mask, and re-apply NULL 3VL into a [`arrow_array::BooleanArray`].
    ///
    /// Returns `Err` (so the caller can host-fall-back) for any non-viable
    /// launch condition. UNVALIDATED device path — see the executor doc.
    fn string_like_mask_gpu(
        &self,
        col: &arrow_array::StringArray,
        literal: &[u8],
        mode: crate::jit::string_kernel::LikeMode,
        negated: bool,
    ) -> BoltResult<arrow_array::BooleanArray> {
        use crate::exec::string_like::{build_row_aligned_from_strings, mask_to_boolean_array};
        use arrow_array::Array;

        let n_rows = col.len();
        let (offsets, bytes, validity) = build_row_aligned_from_strings(col)?;

        // Empty input: nothing to launch; build the (empty) mask directly.
        if n_rows == 0 {
            return Ok(mask_to_boolean_array(&[], &validity));
        }

        // The engine already owns a live `CudaContext` (`self._ctx`), so device
        // allocations below are valid. Any allocation / launch failure returns
        // an `Err`, which the caller turns into a host fallback.
        let offsets_gpu = GpuVec::<i32>::from_slice(&offsets)?;
        let bytes_gpu = if bytes.is_empty() {
            GpuVec::<u8>::zeros(1)?
        } else {
            GpuVec::<u8>::from_slice(&bytes)?
        };
        // Literal: bake as a small device buffer. Pad empty to 1 byte so the
        // device pointer is valid (lit_len==0 short-circuits before any read).
        let lit_len = u32::try_from(literal.len())
            .map_err(|_| BoltError::Other("StringLikeFilter: literal length exceeds u32".into()))?;
        let lit_gpu = if literal.is_empty() {
            GpuVec::<u8>::zeros(1)?
        } else {
            GpuVec::<u8>::from_slice(literal)?
        };
        let mask_gpu = GpuVec::<u8>::zeros(n_rows)?;

        let n_rows_u32 = n_rows_to_u32(n_rows)?;
        let stream = CudaStream::null_or_default();
        let grid_x = grid_x_for(n_rows_u32, BLOCK_SIZE);

        let module = CudaModule::from_ptx(&crate::jit::string_kernel::compile_like_match_kernel(
            mode, negated,
        )?)?;
        let function = module.function(crate::jit::string_kernel::LIKE_MATCH_ENTRY)?;

        let mut p_off = offsets_gpu.device_ptr();
        let mut p_bytes = bytes_gpu.device_ptr();
        let mut p_lit = lit_gpu.device_ptr();
        let mut p_mask = mask_gpu.device_ptr();
        let mut p_n = n_rows_u32;
        let mut p_l = lit_len;
        let mut params: Vec<*mut c_void> = vec![
            &mut p_off as *mut CUdeviceptr as *mut c_void,
            &mut p_bytes as *mut CUdeviceptr as *mut c_void,
            &mut p_lit as *mut CUdeviceptr as *mut c_void,
            &mut p_mask as *mut CUdeviceptr as *mut c_void,
            &mut p_n as *mut u32 as *mut c_void,
            &mut p_l as *mut u32 as *mut c_void,
        ];
        unsafe {
            cuda_sys::check(cuda_sys::cuLaunchKernel(
                function.raw(),
                grid_x,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                stream.raw(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }
        debug_sync_check()?;
        stream.synchronize()?;

        let mask = mask_gpu.to_vec()?;
        check_len(mask.len(), n_rows)?;
        Ok(mask_to_boolean_array(&mask, &validity))
    }
}
