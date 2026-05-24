// SPDX-License-Identifier: Apache-2.0

//! GPU-side filter compaction: prefix-scan + gather, end-to-end.
//!
//! Pairs with [`crate::jit::prefix_scan`], which emits the PTX. The flow:
//!
//! ```text
//!  mask(u8,n)  ──►  per-block exclusive scan
//!                       │             │
//!                       ▼             ▼
//!                 local_indices  block_sums       (device)
//!                                     │
//!                                     ▼  d2h + host exclusive scan
//!                                block_bases (host) + total_count
//!                                     │  h2d
//!                                     ▼
//!                                block_bases       (device)
//!                                     │
//!  input(T,n) ──────────────► gather_one ──► output(T, total_count)
//! ```
//!
//! The host-side scan over `block_sums` is trivial at the row counts the
//! engine handles per batch: with `BLOCK_SIZE = 256`, `n_rows = 16_777_215`
//! produces `65_535` blocks, which serial-sums in microseconds. Larger inputs
//! should be batched upstream — we reject `n_rows > u32::MAX / BLOCK_SIZE`
//! up-front rather than silently degrade.
//!
//! ## Mask lifetime
//!
//! The mask device pointer is captured inside [`ScanResult`] alongside the
//! per-row local indices. The caller owns the underlying `GpuVec<u8>` and must
//! keep it alive for as long as the `ScanResult` is used (every `gather_one`
//! call dereferences it). We capture the *raw pointer* rather than borrowing
//! the `GpuVec` so the scan result has no lifetime parameter and is easy to
//! stash in engine state.

use std::ffi::c_void;
use std::ptr;

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{JavelinError, JavelinResult};
use crate::exec::launch::CudaStream;
use crate::jit::jit_compiler::CudaModule;
use crate::jit::prefix_scan::{
    compile_gather_kernel, compile_prefix_scan_kernel, gather_kernel_entry, BLOCK_SIZE,
    SCAN_KERNEL_ENTRY,
};
use crate::plan::logical_plan::DataType;

/// Outputs of [`prefix_scan_mask`]: the per-row exclusive prefixes, the
/// per-block bases (already exclusive-summed on the host and re-uploaded), and
/// the total number of surviving rows.
///
/// The mask device pointer is also captured so [`gather_one`] does not need it
/// as a separate argument; the caller-owned mask `GpuVec<u8>` must outlive
/// this struct.
pub struct ScanResult {
    /// Per-row, block-local exclusive prefix sum of the mask. Length = `n_rows`.
    pub local_indices: GpuVec<u32>,
    /// Per-block exclusive prefix sum of `block_sums`. Length = `n_blocks`.
    pub block_bases: GpuVec<u32>,
    /// Total surviving rows = sum of all `block_sums`.
    pub total_count: usize,
    /// Device pointer of the u8 mask the scan was computed over. Re-used by
    /// every `gather_one` call. The caller owns the underlying allocation.
    pub mask_ptr: CUdeviceptr,
    /// Number of rows the mask covers. Cached so gather launches can validate
    /// without re-deriving it from `local_indices.len()`.
    pub n_rows: usize,
}

/// Owned, typed gather output column. Keep it alive past the gather launch.
///
/// The variants exist so the public API doesn't have to be generic over `T`
/// at every call site; the engine can branch on dtype once and then carry
/// around a single enum value.
pub enum GatheredCol {
    /// Compacted column of `i32` values.
    I32(GpuVec<i32>),
    /// Compacted column of `i64` values.
    I64(GpuVec<i64>),
    /// Compacted column of `f32` values.
    F32(GpuVec<f32>),
    /// Compacted column of `f64` values.
    F64(GpuVec<f64>),
    /// Compacted column of `u8` values (used for `Bool`).
    Bool(GpuVec<u8>),
}

impl GatheredCol {
    /// Raw device pointer of the underlying GpuVec.
    pub fn device_ptr(&self) -> CUdeviceptr {
        match self {
            GatheredCol::I32(v) => v.device_ptr(),
            GatheredCol::I64(v) => v.device_ptr(),
            GatheredCol::F32(v) => v.device_ptr(),
            GatheredCol::F64(v) => v.device_ptr(),
            GatheredCol::Bool(v) => v.device_ptr(),
        }
    }

    /// Element count of the underlying GpuVec.
    pub fn len(&self) -> usize {
        match self {
            GatheredCol::I32(v) => v.len(),
            GatheredCol::I64(v) => v.len(),
            GatheredCol::F32(v) => v.len(),
            GatheredCol::F64(v) => v.len(),
            GatheredCol::Bool(v) => v.len(),
        }
    }

    /// Whether the gather produced zero output rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy this GPU column back to a host Arrow array.
    ///
    /// Mirrors the per-variant logic in `engine.rs::DeviceCol::download` so the
    /// engine can use a single code path after `compact_columns_on_gpu`. The
    /// `Bool` variant goes through `Vec<u8> -> Vec<bool>` because Arrow's
    /// `BooleanArray::from` expects a `Vec<bool>`, not a packed byte buffer.
    pub fn download(&self) -> crate::error::JavelinResult<arrow_array::ArrayRef> {
        use std::sync::Arc;
        match self {
            GatheredCol::I32(v) => {
                let host = v.to_vec()?;
                Ok(Arc::new(arrow_array::Int32Array::from(host)) as arrow_array::ArrayRef)
            }
            GatheredCol::I64(v) => {
                let host = v.to_vec()?;
                Ok(Arc::new(arrow_array::Int64Array::from(host)) as arrow_array::ArrayRef)
            }
            GatheredCol::F32(v) => {
                let host = v.to_vec()?;
                Ok(Arc::new(arrow_array::Float32Array::from(host)) as arrow_array::ArrayRef)
            }
            GatheredCol::F64(v) => {
                let host = v.to_vec()?;
                Ok(Arc::new(arrow_array::Float64Array::from(host)) as arrow_array::ArrayRef)
            }
            GatheredCol::Bool(v) => {
                let host = v.to_vec()?;
                let bools: Vec<bool> = host.into_iter().map(|b| b != 0).collect();
                Ok(Arc::new(arrow_array::BooleanArray::from(bools)) as arrow_array::ArrayRef)
            }
        }
    }
}

/// Run the device-side prefix scan over an existing u8 mask.
///
/// The caller is responsible for keeping `mask_ptr`'s allocation alive across
/// every subsequent `gather_one` call. The returned `ScanResult` only owns the
/// scan products.
pub fn prefix_scan_mask(
    mask_ptr: CUdeviceptr,
    n_rows: usize,
    stream: &CudaStream,
) -> JavelinResult<ScanResult> {
    if n_rows == 0 {
        return Ok(ScanResult {
            local_indices: GpuVec::<u32>::empty(),
            block_bases: GpuVec::<u32>::empty(),
            total_count: 0,
            mask_ptr,
            n_rows: 0,
        });
    }
    let max_rows = (u32::MAX as usize) / (BLOCK_SIZE as usize);
    if n_rows > max_rows {
        // Single-pass topped out; delegate to the recursive multi-pass path.
        return crate::exec::gpu_compact_multipass::prefix_scan_mask_multipass(
            mask_ptr, n_rows, stream,
        );
    }

    let block_size = BLOCK_SIZE as usize;
    let n_blocks = n_rows.div_ceil(block_size);

    // Allocate the two device output buffers.
    let local_indices = GpuVec::<u32>::zeros(n_rows)?;
    let block_sums = GpuVec::<u32>::zeros(n_blocks)?;

    // JIT-compile and load the scan kernel.
    let ptx = compile_prefix_scan_kernel()?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(SCAN_KERNEL_ENTRY)?;

    // Launch. cuLaunchKernel ABI: pointer-to-each-arg in a *mut c_void array.
    let mut p_mask: CUdeviceptr = mask_ptr;
    let mut p_local: CUdeviceptr = local_indices.device_ptr();
    let mut p_block: CUdeviceptr = block_sums.device_ptr();
    let mut n_rows_u32: u32 = n_rows as u32;

    let mut kernel_params: [*mut c_void; 4] = [
        &mut p_mask as *mut CUdeviceptr as *mut c_void,
        &mut p_local as *mut CUdeviceptr as *mut c_void,
        &mut p_block as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
    ];

    let grid_x: u32 = n_blocks as u32;
    // SAFETY: every entry in `kernel_params` points at a stack local that
    // outlives the launch+synchronize below; `function` is borrowed from a
    // live `CudaModule`.
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

    // Download block_sums, exclusive-scan on host, compute total, re-upload.
    let sums_host: Vec<u32> = block_sums.to_vec()?;
    let mut bases_host: Vec<u32> = Vec::with_capacity(sums_host.len());
    let mut running: u64 = 0;
    for s in &sums_host {
        // u64 accumulator is overkill (we already bounded n_rows <= u32::MAX),
        // but it's cheap and makes the bound obvious.
        bases_host.push(running as u32);
        running += *s as u64;
    }
    let total_count = running as usize;

    let block_bases = GpuVec::<u32>::from_slice(&bases_host)?;

    // `block_sums` and the temporary host vecs drop here; their device memory
    // is freed before we hand back the ScanResult. Keep `local_indices` and
    // `block_bases` alive by moving them into the result.
    drop(block_sums);
    drop(sums_host);
    drop(bases_host);

    Ok(ScanResult {
        local_indices,
        block_bases,
        total_count,
        mask_ptr,
        n_rows,
    })
}

/// Gather one column on the device into a freshly allocated `GpuVec` of length
/// `scan.total_count`.
///
/// `input_ptr` must point at a device allocation of `n_rows * size_of::<T>()`
/// bytes where `T` matches `dtype`. The mask buffer captured by
/// `scan.mask_ptr` must still be alive (the caller owns it).
pub fn gather_one(
    input_ptr: CUdeviceptr,
    n_rows: usize,
    scan: &ScanResult,
    dtype: DataType,
    stream: &CudaStream,
) -> JavelinResult<GatheredCol> {
    if matches!(dtype, DataType::Utf8) {
        return Err(JavelinError::Other(
            "gpu_compact: gather Utf8 not supported (variable-width)".into(),
        ));
    }
    if scan.n_rows != n_rows {
        return Err(JavelinError::Other(format!(
            "gpu_compact: scan.n_rows={} mismatches input n_rows={}",
            scan.n_rows, n_rows
        )));
    }
    if scan.local_indices.len() != n_rows {
        return Err(JavelinError::Other(format!(
            "gpu_compact: scan.local_indices.len()={} mismatches n_rows={}",
            scan.local_indices.len(),
            n_rows
        )));
    }

    // Allocate the typed output and pick the device pointer to launch with.
    let col = alloc_gathered(dtype, scan.total_count)?;
    let output_ptr = col.device_ptr();

    if n_rows == 0 || scan.total_count == 0 {
        // Nothing to copy. The pre-allocated output (length 0 or untouched
        // zeros) is the right answer; skip the launch entirely.
        return Ok(col);
    }

    // JIT-compile + load the gather kernel for this dtype.
    let ptx = compile_gather_kernel(dtype)?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(gather_kernel_entry(dtype))?;

    // Assemble the cuLaunchKernel argument array. Order matches the kernel
    // ABI in `compile_gather_kernel`.
    let mut p_mask: CUdeviceptr = scan.mask_ptr;
    let mut p_local: CUdeviceptr = scan.local_indices.device_ptr();
    let mut p_bases: CUdeviceptr = scan.block_bases.device_ptr();
    let mut p_input: CUdeviceptr = input_ptr;
    let mut p_output: CUdeviceptr = output_ptr;
    let mut n_rows_u32: u32 = n_rows as u32;

    let mut kernel_params: [*mut c_void; 6] = [
        &mut p_mask as *mut CUdeviceptr as *mut c_void,
        &mut p_local as *mut CUdeviceptr as *mut c_void,
        &mut p_bases as *mut CUdeviceptr as *mut c_void,
        &mut p_input as *mut CUdeviceptr as *mut c_void,
        &mut p_output as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
    ];

    let block_size = BLOCK_SIZE as usize;
    let n_blocks = n_rows.div_ceil(block_size);
    let grid_x = n_blocks as u32;

    // SAFETY: each kernel_params entry points at a live stack local; `function`
    // is borrowed from a live `CudaModule`; `stream` is live; the device
    // buffers behind every pointer outlive the synchronize below (the caller
    // owns mask/input, `scan` owns local/bases, `col` owns output).
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

    Ok(col)
}

/// Compact a set of pre-allocated, pre-launched output columns end-to-end on
/// the GPU.
///
/// Inputs:
///   - `mask_ptr` / `n_rows`: the device-side u8 mask the projection kernel's
///     predicate emitted, of length `n_rows`. The caller owns this buffer and
///     must keep it alive for the duration of the call (every `gather_one`
///     launch reads through the captured pointer).
///   - `columns`: one device pointer + dtype per output column to compact.
///
/// Pipeline:
///   1. [`prefix_scan_mask`] over `mask_ptr` produces per-row local indices
///      and per-block bases (and the total surviving-row count).
///   2. For each `(ptr, dtype)`, [`gather_one`] launches a typed gather into
///      a freshly allocated `GpuVec` of length `scan.total_count`.
///   3. Returns the `Vec<GatheredCol>` (parallel to `columns`) and the total
///      count. The caller downloads each column to host with `GatheredCol::download`.
///
/// `Utf8` columns return [`JavelinError::Other`] — the gather kernel can only
/// move fixed-width values, so variable-width strings have to go through the
/// host-side `compact_arrays` fallback.
pub fn compact_columns_on_gpu(
    mask_ptr: CUdeviceptr,
    n_rows: usize,
    columns: &[(CUdeviceptr, DataType)],
    stream: &CudaStream,
) -> JavelinResult<(Vec<GatheredCol>, usize)> {
    // Validate dtypes BEFORE launching the scan so a Utf8 column can't waste a
    // kernel launch + sync. `prefix_scan_mask` already short-circuits on
    // n_rows == 0, so the empty-columns + zero-rows path costs just the scan
    // call's early return and the Vec allocation below.
    for (_, dtype) in columns {
        if matches!(dtype, DataType::Utf8) {
            return Err(JavelinError::Other(
                "Utf8 gather not supported on GPU (use host-side compact_arrays)".into(),
            ));
        }
    }

    let scan = prefix_scan_mask(mask_ptr, n_rows, stream)?;

    let mut out: Vec<GatheredCol> = Vec::with_capacity(columns.len());
    for (ptr, dtype) in columns {
        out.push(gather_one(*ptr, n_rows, &scan, *dtype, stream)?);
    }

    Ok((out, scan.total_count))
}

/// Allocate a `GpuVec<T>` matching `dtype` with `len` elements and wrap it.
fn alloc_gathered(dtype: DataType, len: usize) -> JavelinResult<GatheredCol> {
    Ok(match dtype {
        DataType::Bool => GatheredCol::Bool(GpuVec::<u8>::zeros(len)?),
        DataType::Int32 => GatheredCol::I32(GpuVec::<i32>::zeros(len)?),
        DataType::Int64 => GatheredCol::I64(GpuVec::<i64>::zeros(len)?),
        DataType::Float32 => GatheredCol::F32(GpuVec::<f32>::zeros(len)?),
        DataType::Float64 => GatheredCol::F64(GpuVec::<f64>::zeros(len)?),
        DataType::Utf8 => {
            return Err(JavelinError::Other(
                "gpu_compact: gather Utf8 not supported (variable-width)".into(),
            ))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Replicate the host-side exclusive scan that `prefix_scan_mask` runs
    /// over the downloaded `block_sums`. This is the only piece of compaction
    /// logic we can exercise without CUDA, but it's the load-bearing arithmetic
    /// that turns per-block counts into per-block bases — get this wrong and
    /// gather writes overlap.
    fn host_exclusive_scan(sums: &[u32]) -> (Vec<u32>, usize) {
        let mut bases = Vec::with_capacity(sums.len());
        let mut running: u64 = 0;
        for s in sums {
            bases.push(running as u32);
            running += *s as u64;
        }
        (bases, running as usize)
    }

    #[test]
    fn host_scan_empty() {
        let (bases, total) = host_exclusive_scan(&[]);
        assert!(bases.is_empty());
        assert_eq!(total, 0);
    }

    #[test]
    fn host_scan_single_block() {
        let (bases, total) = host_exclusive_scan(&[7]);
        assert_eq!(bases, vec![0]);
        assert_eq!(total, 7);
    }

    #[test]
    fn host_scan_multi_block() {
        // Two blocks of 256 mask bytes; block 0 keeps 100 rows, block 1 keeps
        // 200 rows. Block 0 should land at base 0; block 1 at base 100; and
        // total = 300.
        let sums = vec![100u32, 200u32];
        let (bases, total) = host_exclusive_scan(&sums);
        assert_eq!(bases, vec![0, 100]);
        assert_eq!(total, 300);
    }

    #[test]
    fn host_scan_matches_sum() {
        // Random-ish counts; total must equal the simple sum and bases must
        // be the exclusive prefix.
        let sums = vec![3u32, 0, 5, 256, 1, 9, 9];
        let (bases, total) = host_exclusive_scan(&sums);
        assert_eq!(bases, vec![0, 3, 3, 8, 264, 265, 274]);
        assert_eq!(total, 283);
        assert_eq!(total as u32, sums.iter().sum::<u32>());
    }

    #[test]
    #[ignore = "requires CUDA toolkit at runtime (zeros)"]
    fn gather_col_dispatch_matches_alloc() {
        // alloc_gathered must produce a variant whose len matches the request.
        let g = alloc_gathered(DataType::Int32, 4).expect("alloc i32");
        assert!(matches!(g, GatheredCol::I32(_)));
        assert_eq!(g.len(), 4);

        let g = alloc_gathered(DataType::Float64, 0).expect("alloc f64 empty");
        assert!(matches!(g, GatheredCol::F64(_)));
        assert!(g.is_empty());

        // `expect_err` would require `GatheredCol: Debug`, which we can't
        // derive because `GpuVec<T>` doesn't impl Debug. Match instead.
        match alloc_gathered(DataType::Utf8, 1) {
            Ok(_) => panic!("utf8 should not be supported"),
            Err(e) => assert!(format!("{}", e).contains("Utf8")),
        }
    }

    /// `compact_columns_on_gpu` with no columns and `n_rows = 0` must take the
    /// `prefix_scan_mask` n_rows-shortcut and never reach a kernel launch.
    /// This is the only end-to-end behavior we can assert without a GPU: an
    /// empty input pair returns `(vec![], 0)` and propagates no Cuda error.
    /// We pass `mask_ptr = 0` (NULL device pointer) deliberately — if the
    /// shortcut ever regresses, the first launch will fault on the NULL mask
    /// and the test will start failing instead of silently passing.
    #[test]
    fn compact_empty_inputs_skips_launch() {
        let stream = CudaStream::null();
        let res = compact_columns_on_gpu(0, 0, &[], &stream);
        match res {
            Ok((cols, total)) => {
                assert!(cols.is_empty());
                assert_eq!(total, 0);
            }
            Err(e) => panic!("expected Ok for empty inputs, got {e}"),
        }
    }

    /// A Utf8 entry in `columns` is rejected up-front with the documented
    /// error message — before any scan or gather launches.
    #[test]
    fn compact_utf8_column_rejected() {
        let stream = CudaStream::null();
        // n_rows = 0 keeps the scan's launch path inert in case the Utf8 check
        // ever moves below the scan call; the assertion is about the error
        // message, not which line raised it.
        let cols = [(0u64, DataType::Utf8)];
        match compact_columns_on_gpu(0, 0, &cols, &stream) {
            Ok(_) => panic!("expected Utf8 rejection"),
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains("Utf8 gather not supported on GPU"),
                    "unexpected error message: {msg}"
                );
            }
        }
    }
}
