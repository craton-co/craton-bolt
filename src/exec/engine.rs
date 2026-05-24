// SPDX-License-Identifier: Apache-2.0

//! Top-level engine: dispatches per-shape executors (scalar agg, GROUP BY, etc.);
//! performs GPU prefix-scan + gather compaction for filter outputs, or a host-side
//! `arrow::compute::filter` fallback when any output column is Utf8.
//!
//! The engine owns a CUDA context and a registry of host-side Arrow `RecordBatch`es.
//! `Engine::sql` parses, plans, codegens, launches, and returns a `QueryHandle` whose
//! `record_batch()` exposes the result.
//!
//! Projection-with-filter flow: a predicate-only kernel materialises a `u8` mask
//! into a fresh device buffer. When every output column is gather-friendly
//! (primitive or Bool), the engine then runs `gpu_compact::compact_columns_on_gpu`
//! (prefix scan + gather) entirely on the device and downloads only the surviving
//! rows. When any output column is Utf8 — the gather kernel cannot relocate
//! variable-width strings — the engine falls back to downloading the full
//! per-column outputs plus the mask and running `compact::compact_arrays`
//! (Arrow's host-side filter) on the host. Scalar aggregates, group-bys with or
//! without a `WHERE`, and their `extended_agg`/`expr_agg` variants are
//! dispatched to dedicated executors in `Engine::execute`.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray,
};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::buffer::primitive_to_gpu;
use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::dictionary::DictionaryColumn;
use crate::cuda::{CudaContext, GpuVec};
use crate::error::{JavelinError, JavelinResult};
use crate::exec::launch::CudaStream;
use crate::exec::n_rows_to_u32;
use crate::jit::{compile_ptx, CudaModule};
use crate::plan::{
    parse_sql, DataType, Field, KernelSpec, LogicalPlan, MemTableProvider, PhysicalPlan, Schema,
};

/// PTX entry-point name; matches the symbol `ptx_gen` emits.
const KERNEL_ENTRY: &str = "javelin_kernel";

/// Threads per CUDA block for the 1D launch.
const BLOCK_SIZE: u32 = 256;

/// Top-level query engine.
///
/// Field-drop order matters: `dict_registry` owns `DictionaryColumn`s which own
/// `GpuVec`s — those must be freed BEFORE `_ctx` tears down the CUDA context.
/// Rust drops fields in declaration order, so `_ctx` sits last.
pub struct Engine {
    /// Registered tables, keyed by name.
    tables: HashMap<String, RecordBatch>,
    /// Name → Schema provider, kept in sync with `tables`. The schema is
    /// EXTENDED with `__idx_<col>` Int32 columns for every registered Utf8
    /// column so the SQL frontend resolves rewriter-produced column refs.
    provider: MemTableProvider,
    /// Per-table Utf8 dictionaries; drives the string-literal predicate rewrite.
    dict_registry: crate::exec::dict_registry::DictRegistry,
    /// Owned CUDA context — declared LAST so it drops AFTER dictionaries.
    _ctx: CudaContext,
}

impl Engine {
    /// Create an engine on the default CUDA device (ordinal 0).
    ///
    /// Convenience constructor for single-GPU systems. On hosts with more
    /// than one CUDA device, use [`Engine::new_with_device`] to pick a
    /// specific GPU.
    pub fn new() -> JavelinResult<Self> {
        Self::new_with_device(0)
    }

    /// Create an engine bound to the CUDA device at ordinal `device_idx`.
    ///
    /// Use this when running on a multi-GPU host and you want to target a
    /// specific device. The constructor:
    ///   1. Initializes the CUDA driver (idempotent — safe to call repeatedly).
    ///   2. Validates `device_idx` against `cuDeviceGetCount`.
    ///   3. Creates an owned CUDA context on the selected device.
    ///
    /// # Errors
    /// Returns an error if `device_idx < 0` or `device_idx >=
    /// cuDeviceGetCount()`, or if any underlying CUDA driver call fails
    /// (e.g. no CUDA-capable device, driver/runtime mismatch).
    pub fn new_with_device(device_idx: i32) -> JavelinResult<Self> {
        // Initialize the driver up-front so device_count() is callable.
        cuda_sys::init()?;
        let count = cuda_sys::device_count()?;
        if device_idx < 0 || device_idx >= count {
            return Err(JavelinError::Other(format!(
                "CUDA device index {} is out of range: {} device(s) visible to the driver (valid range: 0..{})",
                device_idx, count, count
            )));
        }
        let ctx = CudaContext::new(device_idx)?;
        Ok(Self {
            tables: HashMap::new(),
            provider: MemTableProvider::new(),
            dict_registry: crate::exec::dict_registry::DictRegistry::new(),
            _ctx: ctx,
        })
    }

    /// Register a host-side `RecordBatch` under `name`. Replaces any existing entry.
    /// Also builds Utf8 dictionaries for the table and extends the engine-side
    /// schema with `__idx_<col>` Int32 columns so the rewriter's emitted column
    /// references resolve at parse time.
    pub fn register_table(
        &mut self,
        name: impl Into<String>,
        batch: RecordBatch,
    ) -> JavelinResult<()> {
        let name = name.into();
        // Build Utf8 dictionaries first (may fail — surface before we mutate
        // tables/provider).
        self.dict_registry.register_table(name.clone(), &batch)?;
        let base_schema = arrow_schema_to_plan_schema(batch.schema().as_ref())?;
        let extended = self.dict_registry.extended_schema(&name, &base_schema);
        self.provider.register(name.clone(), extended);
        self.tables.insert(name, batch);
        Ok(())
    }

    /// Compile and execute a SQL query string.
    pub fn sql(&self, query: &str) -> JavelinResult<QueryHandle> {
        let plan: LogicalPlan = parse_sql(query, &self.provider)?;
        // String-literal predicates against Utf8 columns are folded into
        // integer equality against the corresponding __idx_<col> i32 column.
        let plan = self.dict_registry.rewrite_plan(&plan)?;
        let phys = crate::plan::lower_physical(&plan)?;
        self.execute(&phys)
    }

    /// Execute a pre-built `PhysicalPlan`.
    pub fn execute(&self, phys: &PhysicalPlan) -> JavelinResult<QueryHandle> {
        match phys {
            PhysicalPlan::Projection {
                table,
                kernel,
                output_schema,
            } => self.execute_projection(table, kernel, output_schema),
            PhysicalPlan::Aggregate {
                table,
                pre,
                aggregate,
            } => {
                let batch = self.tables.get(table).ok_or_else(|| {
                    JavelinError::Plan(format!(
                        "table '{table}' is not registered with the engine"
                    ))
                })?;
                let out = match (!aggregate.group_by.is_empty(), pre.is_some()) {
                    (true, true) => {
                        crate::exec::groupby_with_pre::execute_groupby_with_pre(phys, batch)?
                    }
                    (true, false) => crate::exec::groupby::execute_groupby(phys, batch)?,
                    (false, true) => {
                        crate::exec::agg_with_pre::execute_aggregate_with_pre(phys, batch)?
                    }
                    (false, false) => crate::exec::aggregate::execute_aggregate(phys, batch)?,
                };
                Ok(QueryHandle { batch: out })
            }
        }
    }

    /// Execute a single fused projection (optionally with filter) kernel.
    fn execute_projection(
        &self,
        table: &str,
        kernel: &KernelSpec,
        output_schema: &Schema,
    ) -> JavelinResult<QueryHandle> {
        let batch = self.tables.get(table).ok_or_else(|| {
            JavelinError::Plan(format!("table '{table}' is not registered with the engine"))
        })?;
        let n_rows = batch.num_rows();

        // 1. Upload inputs. Keep the owned GpuVecs in `input_cols` so they outlive the launch.
        //
        // `__idx_<col>` inputs come from the dict_registry (they don't exist
        // in the source RecordBatch). They were synthesized by the
        // string-literal rewriter and resolve to the i32/i64 dictionary index
        // column already on the device — we hand the launch a `Borrowed`
        // device pointer into the registry's `GpuVec` rather than bouncing the
        // index column through the host. `&self` is borrowed for the entire
        // `execute_projection`, so the dictionary's GpuVec outlives the launch.
        let mut input_cols: Vec<DeviceCol> = Vec::with_capacity(kernel.inputs.len());
        for io in &kernel.inputs {
            if let Some(original) = io.name.strip_prefix("__idx_") {
                let dict = self.dict_registry.dictionary(table, original).ok_or_else(|| {
                    JavelinError::Plan(format!(
                        "rewriter-emitted column '{}' has no dictionary in registry",
                        io.name
                    ))
                })?;
                // Fail fast on plan/dict dtype mismatch BEFORE doing any I/O —
                // this catches a stale plan that names __idx_X with the wrong
                // width without paying the cost of touching the device.
                if io.dtype != dict.index_dtype() {
                    return Err(JavelinError::Plan(format!(
                        "rewriter-emitted column '{}' dtype mismatch: plan says {:?}, dictionary is {:?}",
                        io.name, io.dtype, dict.index_dtype()
                    )));
                }
                // Borrow the device pointer from the registry's existing
                // index column — no host bounce, no fresh allocation.
                let ptr = match dict {
                    crate::cuda::dictionary_any::DictionaryColumnAny::I32(d) => {
                        d.indices.device_ptr()
                    }
                    crate::cuda::dictionary_any::DictionaryColumnAny::I64(d) => {
                        d.indices.device_ptr()
                    }
                };
                input_cols.push(DeviceCol::Borrowed { ptr });
                continue;
            }
            let idx = batch
                .schema()
                .index_of(&io.name)
                .map_err(|e| JavelinError::Plan(format!("column '{}' not in table '{}': {e}", io.name, table)))?;
            let arr = batch.column(idx);
            let dev = DeviceCol::upload(arr.as_ref(), io.dtype)?;
            input_cols.push(dev);
        }

        // 2. Allocate output buffers, zero-initialised. For Utf8 passthrough
        //    columns (output dtype Utf8 AND name matches an input column),
        //    clone the source dictionary so download can decode indices back
        //    to strings. (Computed Utf8 outputs aren't supported yet.)
        let mut output_cols: Vec<DeviceCol> = Vec::with_capacity(kernel.outputs.len());
        for io in &kernel.outputs {
            let mut col = DeviceCol::alloc_zeros(io.dtype, n_rows)?;
            if io.dtype == DataType::Utf8 {
                if let Some(src) = input_cols
                    .iter()
                    .zip(kernel.inputs.iter())
                    .find(|(_, in_io)| in_io.name == io.name && in_io.dtype == DataType::Utf8)
                    .and_then(|(c, _)| c.utf8_dictionary())
                {
                    col.set_utf8_dictionary(src.to_vec());
                }
            }
            output_cols.push(col);
        }

        // 3. JIT-compile the kernel to PTX and load it.
        let ptx = compile_ptx(kernel, KERNEL_ENTRY)?;
        let module = CudaModule::from_ptx(&ptx)?;
        let function = module.function(KERNEL_ENTRY)?;

        // 4. Build the kernel-parameter list.
        //
        // `KernelArgs` is monomorphic on `T` per push and cannot store heterogenous
        // column types in one list. We bypass it and assemble raw kernel params
        // directly: inputs first, then outputs, then the row-count `u32`.
        let mut device_ptrs: Vec<CUdeviceptr> = Vec::with_capacity(input_cols.len() + output_cols.len());
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

        // 5. Launch with one thread per row, block size 256.
        let stream = CudaStream::null();
        let grid_x = ((n_rows_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);
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
        stream.synchronize()?;

        // 6. If the kernel has a predicate, run a separate predicate-only
        //    kernel to materialise a u8 mask. We default to GPU-side compaction
        //    (prefix scan + gather) when every output column is gather-friendly
        //    (primitive + Bool); Utf8 outputs fall back to the host-side path
        //    because the gather kernel can't move variable-width strings.
        //
        //    `input_cols` must outlive the predicate launch.
        let arrays: Vec<ArrayRef> = if kernel.predicate.is_some() {
            let pred_ptx =
                crate::jit::scan_kernel::compile_predicate_kernel(kernel, "javelin_predicate")?;
            let pred_module = CudaModule::from_ptx(&pred_ptx)?;
            let pred_function = pred_module.function("javelin_predicate")?;

            let mask = crate::exec::compact::alloc_mask_buffer(n_rows)?;
            let input_ptrs: Vec<CUdeviceptr> =
                input_cols.iter().map(|c| c.device_ptr()).collect();
            crate::exec::compact::launch_predicate_kernel(
                pred_function,
                &input_ptrs,
                mask.device_ptr(),
                n_rows_to_u32(n_rows)?,
                &stream,
            )?;

            let has_utf8_output = kernel.outputs.iter().any(|c| c.dtype == DataType::Utf8);
            if has_utf8_output {
                // Host-side fallback: download mask + outputs, then filter.
                let host_mask =
                    crate::exec::compact::download_mask(mask.device_ptr(), n_rows)?;
                drop(input_cols);
                let mut full: Vec<ArrayRef> = Vec::with_capacity(output_cols.len());
                for col in output_cols {
                    full.push(col.download(n_rows)?);
                }
                crate::exec::compact::compact_arrays(&full, &host_mask)?
            } else {
                // GPU-side path: prefix-scan + gather, download the compacted output.
                let cols: Vec<(CUdeviceptr, DataType)> = output_cols
                    .iter()
                    .zip(kernel.outputs.iter())
                    .map(|(c, io)| (c.device_ptr(), io.dtype))
                    .collect();
                let (gathered, _total) = crate::exec::gpu_compact::compact_columns_on_gpu(
                    mask.device_ptr(),
                    n_rows,
                    &cols,
                    &stream,
                )?;
                // Output buffers can drop now; gathered owns the compacted data.
                drop(input_cols);
                drop(output_cols);
                let mut out: Vec<ArrayRef> = Vec::with_capacity(gathered.len());
                for g in &gathered {
                    out.push(g.download()?);
                }
                out
            }
        } else {
            drop(input_cols);
            let mut full: Vec<ArrayRef> = Vec::with_capacity(output_cols.len());
            for col in output_cols {
                full.push(col.download(n_rows)?);
            }
            full
        };

        // 9. Build the result RecordBatch.
        let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
        let batch_out = RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
            JavelinError::Other(format!("failed to build output RecordBatch: {e}"))
        })?;
        Ok(QueryHandle { batch: batch_out })
    }
}

/// Result of a query — wraps the output Arrow `RecordBatch`.
pub struct QueryHandle {
    /// The materialised result.
    batch: RecordBatch,
}

impl QueryHandle {
    /// Borrow the underlying record batch.
    pub fn record_batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// Consume the handle and return the owned record batch.
    pub fn into_record_batch(self) -> RecordBatch {
        self.batch
    }

    /// Number of rows in the result.
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }
}

/// Heterogenous owned device column. Keeps each `GpuVec<T>` alive past the kernel launch.
enum DeviceCol {
    /// 32-bit signed integer column.
    I32(GpuVec<i32>),
    /// 64-bit signed integer column.
    I64(GpuVec<i64>),
    /// 32-bit float column.
    F32(GpuVec<f32>),
    /// 64-bit float column.
    F64(GpuVec<f64>),
    /// Bool stored as one byte per row (0 / 1). Used when the source Arrow
    /// array has no nulls.
    Bool(GpuVec<u8>),
    /// Bool stored as TWO parallel byte-per-row buffers:
    ///   * `values[i] = 1` iff row `i` is `true`, `0` otherwise (incl. null).
    ///   * `validity[i] = 1` iff row `i` is non-null, `0` if null.
    /// Both buffers have the row-count length. The kernel ABI continues to
    /// see only the values pointer via `device_ptr()`; validity is consumed
    /// host-side on download and (TODO post-w5) threaded through filter and
    /// aggregate kernels.
    BoolNullable {
        values: GpuVec<u8>,
        validity: GpuVec<u8>,
    },
    /// Utf8 stored as i32 dictionary indices; host dictionary lives alongside.
    Utf8(DictionaryColumn),
    /// Borrowed device pointer — the underlying buffer is owned elsewhere
    /// (today: a dictionary in `dict_registry`). Use ONLY as a kernel input;
    /// `download()` is unreachable because we drop `input_cols` before reading
    /// outputs. The lifetime of the owning buffer is enforced by `&self`
    /// borrowing for the entire duration of `execute_projection`.
    Borrowed { ptr: CUdeviceptr },
}

impl DeviceCol {
    /// Upload an Arrow array to the GPU, downcasting per `dtype`.
    fn upload(arr: &dyn Array, dtype: DataType) -> JavelinResult<Self> {
        match dtype {
            DataType::Int32 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| type_mismatch_err(arr, "Int32"))?;
                let buf = primitive_to_gpu(pa)?;
                Ok(DeviceCol::I32(GpuVec::from_buffer(buf)))
            }
            DataType::Int64 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| type_mismatch_err(arr, "Int64"))?;
                let buf = primitive_to_gpu(pa)?;
                Ok(DeviceCol::I64(GpuVec::from_buffer(buf)))
            }
            DataType::Float32 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| type_mismatch_err(arr, "Float32"))?;
                let buf = primitive_to_gpu(pa)?;
                Ok(DeviceCol::F32(GpuVec::from_buffer(buf)))
            }
            DataType::Float64 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| type_mismatch_err(arr, "Float64"))?;
                let buf = primitive_to_gpu(pa)?;
                Ok(DeviceCol::F64(GpuVec::from_buffer(buf)))
            }
            DataType::Bool => {
                let ba = arr
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| type_mismatch_err(arr, "Bool"))?;
                let n = ba.len();
                // No-null fast path: single value buffer, the legacy `Bool`
                // variant. Existing kernels and the gather/compact paths
                // continue to see the same one-byte-per-row layout.
                if ba.null_count() == 0 {
                    let mut bytes: Vec<u8> = Vec::with_capacity(n);
                    for i in 0..n {
                        bytes.push(if ba.value(i) { 1 } else { 0 });
                    }
                    return Ok(DeviceCol::Bool(GpuVec::<u8>::from_slice(&bytes)?));
                }
                // Nullable path: build BOTH a value buffer (0 for false-or-null
                // so value-only kernels see a defined byte) AND a parallel
                // validity buffer (1 = non-null, 0 = null), then upload both
                // and produce a `BoolNullable` device column.
                //
                // TODO(post-w5): wire validity through filter/agg kernels —
                // today only the projection download path consumes it to
                // reconstruct a nullable BooleanArray. Filter/compact and the
                // aggregate executors still see the value buffer alone via
                // `device_ptr()` and will treat null rows as `false`.
                let mut values: Vec<u8> = Vec::with_capacity(n);
                let mut validity: Vec<u8> = Vec::with_capacity(n);
                for i in 0..n {
                    if ba.is_null(i) {
                        values.push(0);
                        validity.push(0);
                    } else {
                        values.push(if ba.value(i) { 1 } else { 0 });
                        validity.push(1);
                    }
                }
                let v_gpu = GpuVec::<u8>::from_slice(&values)?;
                let m_gpu = GpuVec::<u8>::from_slice(&validity)?;
                Ok(DeviceCol::BoolNullable {
                    values: v_gpu,
                    validity: m_gpu,
                })
            }
            DataType::Utf8 => {
                let sa = arr
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| type_mismatch_err(arr, "Utf8"))?;
                Ok(DeviceCol::Utf8(DictionaryColumn::from_string_array(sa)?))
            }
        }
    }

    /// Allocate a zero-initialised device column of `n` rows.
    ///
    /// Utf8 outputs allocate an empty dictionary; the engine must replace it
    /// with the source column's dictionary before download (today this only
    /// works for pure column-passthrough projections — `output_schema` field
    /// name matching an input column name).
    fn alloc_zeros(dtype: DataType, n: usize) -> JavelinResult<Self> {
        match dtype {
            DataType::Int32 => Ok(DeviceCol::I32(GpuVec::<i32>::zeros(n)?)),
            DataType::Int64 => Ok(DeviceCol::I64(GpuVec::<i64>::zeros(n)?)),
            DataType::Float32 => Ok(DeviceCol::F32(GpuVec::<f32>::zeros(n)?)),
            DataType::Float64 => Ok(DeviceCol::F64(GpuVec::<f64>::zeros(n)?)),
            DataType::Bool => Ok(DeviceCol::Bool(GpuVec::<u8>::zeros(n)?)),
            DataType::Utf8 => Ok(DeviceCol::Utf8(DictionaryColumn {
                dictionary: Vec::new(),
                indices: GpuVec::<i32>::zeros(n)?,
                n_rows: n,
            })),
        }
    }

    /// Raw device pointer for kernel-parameter assembly.
    ///
    /// For `BoolNullable`, this returns the values pointer only — the
    /// validity buffer is not yet exposed to kernels (see
    /// TODO(post-w5) in the upload path). The buffer's lifetime is
    /// preserved by `self` because the variant owns both `GpuVec`s.
    fn device_ptr(&self) -> CUdeviceptr {
        match self {
            DeviceCol::I32(v) => v.device_ptr(),
            DeviceCol::I64(v) => v.device_ptr(),
            DeviceCol::F32(v) => v.device_ptr(),
            DeviceCol::F64(v) => v.device_ptr(),
            DeviceCol::Bool(v) => v.device_ptr(),
            DeviceCol::BoolNullable { values, .. } => values.device_ptr(),
            DeviceCol::Utf8(d) => d.indices.device_ptr(),
            DeviceCol::Borrowed { ptr } => *ptr,
        }
    }

    /// Install a dictionary on a Utf8 column (for output columns whose source dictionary
    /// the engine knows). No-op for non-Utf8 columns.
    fn set_utf8_dictionary(&mut self, dict: Vec<String>) {
        if let DeviceCol::Utf8(d) = self {
            d.dictionary = dict;
        }
    }

    /// Borrow the inner dictionary if this is a Utf8 column.
    fn utf8_dictionary(&self) -> Option<&[String]> {
        match self {
            DeviceCol::Utf8(d) => Some(&d.dictionary),
            _ => None,
        }
    }

    /// Copy the device column back to a host Arrow array of length `n_rows`.
    fn download(self, n_rows: usize) -> JavelinResult<ArrayRef> {
        match self {
            DeviceCol::I32(v) => {
                let host = copy_back::<i32>(&v, n_rows)?;
                Ok(Arc::new(Int32Array::from(host)) as ArrayRef)
            }
            DeviceCol::I64(v) => {
                let host = copy_back::<i64>(&v, n_rows)?;
                Ok(Arc::new(Int64Array::from(host)) as ArrayRef)
            }
            DeviceCol::F32(v) => {
                let host = copy_back::<f32>(&v, n_rows)?;
                Ok(Arc::new(Float32Array::from(host)) as ArrayRef)
            }
            DeviceCol::F64(v) => {
                let host = copy_back::<f64>(&v, n_rows)?;
                Ok(Arc::new(Float64Array::from(host)) as ArrayRef)
            }
            DeviceCol::Bool(v) => {
                let host = copy_back::<u8>(&v, n_rows)?;
                let bools: Vec<bool> = host.into_iter().map(|b| b != 0).collect();
                Ok(Arc::new(BooleanArray::from(bools)) as ArrayRef)
            }
            DeviceCol::BoolNullable { values, validity } => {
                let host_values = copy_back::<u8>(&values, n_rows)?;
                let host_validity = copy_back::<u8>(&validity, n_rows)?;
                // Reconstruct a nullable BooleanArray by zipping values with
                // the validity buffer: null rows become `None`, valid rows
                // become `Some(value != 0)`.
                let arr: BooleanArray = host_values
                    .into_iter()
                    .zip(host_validity.into_iter())
                    .map(|(v, m)| if m == 1 { Some(v == 1) } else { None })
                    .collect();
                Ok(Arc::new(arr) as ArrayRef)
            }
            DeviceCol::Utf8(d) => {
                let arr = d.to_string_array()?;
                Ok(Arc::new(arr) as ArrayRef)
            }
            DeviceCol::Borrowed { .. } => Err(JavelinError::Other(
                "internal: cannot download a borrowed device column — \
                 Borrowed variants are kernel inputs only and must be dropped \
                 before any output download"
                    .into(),
            )),
        }
    }
}

/// Copy back a `GpuVec<T>` into a host `Vec<T>` of length `n_rows`.
///
/// Output buffers are allocated via `GpuVec::zeros(n_rows)`, whose `len()` is `n_rows`,
/// so `to_vec()` returns exactly that many elements.
fn copy_back<T>(v: &GpuVec<T>, n_rows: usize) -> JavelinResult<Vec<T>>
where
    T: bytemuck::Pod,
{
    let host = v.to_vec()?;
    if host.len() != n_rows {
        return Err(JavelinError::Other(format!(
            "internal: device buffer length {} did not match expected {}",
            host.len(),
            n_rows
        )));
    }
    Ok(host)
}

/// Build a `Type` error for an Arrow downcast failure.
fn type_mismatch_err(arr: &dyn Array, expected: &str) -> JavelinError {
    JavelinError::Type(format!(
        "Arrow array dtype {:?} does not match expected {}",
        arr.data_type(),
        expected
    ))
}

/// Map our plan `DataType` to Arrow `DataType`.
fn plan_dtype_to_arrow(d: DataType) -> JavelinResult<ArrowDataType> {
    match d {
        DataType::Int32 => Ok(ArrowDataType::Int32),
        DataType::Int64 => Ok(ArrowDataType::Int64),
        DataType::Float32 => Ok(ArrowDataType::Float32),
        DataType::Float64 => Ok(ArrowDataType::Float64),
        DataType::Bool => Ok(ArrowDataType::Boolean),
        DataType::Utf8 => Ok(ArrowDataType::Utf8),
    }
}

/// Map Arrow `DataType` to our plan `DataType`. Errors on unsupported types.
fn arrow_dtype_to_plan(d: &ArrowDataType) -> JavelinResult<DataType> {
    match d {
        ArrowDataType::Int32 => Ok(DataType::Int32),
        ArrowDataType::Int64 => Ok(DataType::Int64),
        ArrowDataType::Float32 => Ok(DataType::Float32),
        ArrowDataType::Float64 => Ok(DataType::Float64),
        ArrowDataType::Boolean => Ok(DataType::Bool),
        ArrowDataType::Utf8 => Ok(DataType::Utf8),
        other => Err(JavelinError::Type(format!(
            "unsupported Arrow dtype {:?}",
            other
        ))),
    }
}

/// Convert an `arrow_schema::Schema` into our plan `Schema`.
fn arrow_schema_to_plan_schema(s: &ArrowSchema) -> JavelinResult<Schema> {
    let mut fields = Vec::with_capacity(s.fields().len());
    for f in s.fields() {
        let dt = arrow_dtype_to_plan(f.data_type())?;
        fields.push(Field::new(f.name().clone(), dt, f.is_nullable()));
    }
    Ok(Schema::new(fields))
}

/// Convert our plan `Schema` to an `arrow_schema::Schema` (used for output `RecordBatch`).
fn plan_schema_to_arrow_schema(s: &Schema) -> JavelinResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}
