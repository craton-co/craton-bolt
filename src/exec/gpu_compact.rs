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
use crate::exec::n_rows_to_u32;
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
    /// Compacted column of `u8` values (used for `Bool`, no nulls).
    Bool(GpuVec<u8>),
    /// Compacted nullable bool: two parallel `u8`-per-row buffers, both of
    /// length `n_surviving_rows`. Produced by [`gather_bool_nullable`] when
    /// the source column was a `BoolNullable` device column. The same
    /// gather indices are used for both buffers — for any surviving row
    /// `j`, `values[j]` and `validity[j]` come from the same source row,
    /// so the per-row null-ness contract is preserved end-to-end:
    ///
    /// * `validity[j] == 1` &rarr; row `j` is non-null, `values[j]` is the
    ///   real bool byte (`0` / `1`).
    /// * `validity[j] == 0` &rarr; row `j` is null. The byte at
    ///   `values[j]` is conservatively `0` to keep value-only kernels
    ///   well-defined, but consumers MUST check `validity[j]` first.
    BoolNullable {
        /// Gathered value bytes (`0` = false-or-null, `1` = true).
        values: GpuVec<u8>,
        /// Gathered validity bytes (`0` = null, `1` = non-null). Same
        /// length as `values`.
        validity: GpuVec<u8>,
    },
}

impl GatheredCol {
    /// Raw device pointer of the underlying GpuVec.
    ///
    /// For [`GatheredCol::BoolNullable`] this returns the *values* buffer's
    /// pointer only — the validity buffer is reachable via
    /// [`Self::validity_device_ptr`]. This mirrors the
    /// `engine.rs::DeviceCol::BoolNullable::device_ptr()` convention so
    /// kernels that don't consume validity see the same byte layout as the
    /// no-null `Bool` variant.
    pub fn device_ptr(&self) -> CUdeviceptr {
        match self {
            GatheredCol::I32(v) => v.device_ptr(),
            GatheredCol::I64(v) => v.device_ptr(),
            GatheredCol::F32(v) => v.device_ptr(),
            GatheredCol::F64(v) => v.device_ptr(),
            GatheredCol::Bool(v) => v.device_ptr(),
            GatheredCol::BoolNullable { values, .. } => values.device_ptr(),
        }
    }

    /// Raw device pointer to the validity buffer, if this column carries one.
    /// Only [`GatheredCol::BoolNullable`] has a validity buffer; all other
    /// variants return `None`.
    pub fn validity_device_ptr(&self) -> Option<CUdeviceptr> {
        match self {
            GatheredCol::BoolNullable { validity, .. } => Some(validity.device_ptr()),
            _ => None,
        }
    }

    /// Element count of the underlying GpuVec.
    ///
    /// For [`GatheredCol::BoolNullable`] the values and validity buffers are
    /// gathered with the same indices, so they have identical lengths; this
    /// returns the shared length (i.e. the row count).
    pub fn len(&self) -> usize {
        match self {
            GatheredCol::I32(v) => v.len(),
            GatheredCol::I64(v) => v.len(),
            GatheredCol::F32(v) => v.len(),
            GatheredCol::F64(v) => v.len(),
            GatheredCol::Bool(v) => v.len(),
            // Invariant: gather_bool_nullable launches the same gather kernel
            // with the same scan over both buffers, so `values.len() ==
            // validity.len() == scan.total_count`. We pick `values` here as
            // the canonical row count.
            GatheredCol::BoolNullable { values, .. } => values.len(),
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
    ///
    /// [`GatheredCol::BoolNullable`] materialises a *nullable* `BooleanArray`
    /// by zipping the values and validity bytes — the same reconstruction
    /// that `ExtendedDeviceCol::BoolNullable::download` uses for the
    /// uncompacted upload-side path. This is what preserves W5A2's
    /// per-row null-ness across the GPU prefix-scan + gather pipeline.
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
            GatheredCol::BoolNullable { values, validity } => {
                let host_values: Vec<u8> = values.to_vec()?;
                let host_validity: Vec<u8> = validity.to_vec()?;
                // Defensive: the two buffers must agree on length. The
                // invariant is enforced at construction (gather_bool_nullable
                // gathers both with the same `scan`), but if a future caller
                // hand-builds a `BoolNullable` with mismatched buffers we
                // want a clean error instead of a silent truncation in
                // `zip`.
                if host_values.len() != host_validity.len() {
                    return Err(JavelinError::Other(format!(
                        "GatheredCol::BoolNullable buffer length mismatch: \
                         values={}, validity={}",
                        host_values.len(),
                        host_validity.len(),
                    )));
                }
                let arr: arrow_array::BooleanArray = host_values
                    .into_iter()
                    .zip(host_validity.into_iter())
                    .map(|(v, m)| if m == 1 { Some(v == 1) } else { None })
                    .collect();
                Ok(Arc::new(arr) as arrow_array::ArrayRef)
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
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;

    let mut kernel_params: [*mut c_void; 4] = [
        &mut p_mask as *mut CUdeviceptr as *mut c_void,
        &mut p_local as *mut CUdeviceptr as *mut c_void,
        &mut p_block as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
    ];

    let grid_x: u32 = n_rows_to_u32(n_blocks)?;
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
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;

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
    let grid_x = n_rows_to_u32(n_blocks)?;

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

/// Gather BOTH halves of a nullable bool column (values + validity) using a
/// single shared `ScanResult`, returning a [`GatheredCol::BoolNullable`].
///
/// This is the GPU analogue of the host-side path described in
/// `compact.rs::apply_mask`: per-row nullness is preserved because both
/// buffers are gathered with the *same* `scan`. For any surviving output
/// row `j`, both `values[j]` and `validity[j]` are pulled from the
/// identical source row `i`, so the value/validity correspondence the
/// `BoolNullable` contract requires is invariant under compaction.
///
/// Invariants enforced at call sites:
///   * `values_ptr` and `validity_ptr` point to device allocations of
///     length `n_rows` bytes each. The caller owns both — they must
///     outlive the synchronize inside each `gather_one` call.
///   * `scan` was produced from a mask of length `n_rows` (the same
///     `n_rows` argument passed here).
///
/// The gather kernel itself is unchanged — it's a generic per-dtype gather.
/// We just launch it twice with the same scan and box the pair into a
/// `BoolNullable` variant. Two kernel launches are intentional: the gather
/// kernel ABI takes a single input pointer, and the two buffers are
/// physically separate allocations on the device.
///
/// Wired in by W7A8. The engine should branch on
/// `engine.rs::DeviceCol::BoolNullable` and call this in place of
/// `gather_one` so the validity bitmap survives the filter compaction. Until
/// the engine plumbing lands, this helper is callable from any code path
/// that already has the two device pointers in hand.
pub fn gather_bool_nullable(
    values_ptr: CUdeviceptr,
    validity_ptr: CUdeviceptr,
    n_rows: usize,
    scan: &ScanResult,
    stream: &CudaStream,
) -> JavelinResult<GatheredCol> {
    // Two independent gather launches, both keyed off the same `scan`. The
    // kernel ABI handles one buffer at a time; we re-use the scan products
    // (local_indices + block_bases + mask_ptr + total_count) so the second
    // launch is just `compile_gather_kernel(Bool)` again on a different
    // input pointer. JIT caching at the `compile_gather_kernel` layer
    // means we don't re-compile the PTX for the second call in practice.
    let gathered_values = gather_one(values_ptr, n_rows, scan, DataType::Bool, stream)?;
    let gathered_validity = gather_one(validity_ptr, n_rows, scan, DataType::Bool, stream)?;

    // Unwrap to the inner `GpuVec<u8>` for the new variant. Both must come
    // out of `gather_one(DataType::Bool, ...)` as `GatheredCol::Bool`; any
    // other shape is a programming error in this file, not a runtime
    // condition, so we panic-with-message rather than threading a Result.
    let values = match gathered_values {
        GatheredCol::Bool(v) => v,
        _ => unreachable!(
            "gather_one(DataType::Bool, ...) must return GatheredCol::Bool; \
             see alloc_gathered match arm"
        ),
    };
    let validity = match gathered_validity {
        GatheredCol::Bool(v) => v,
        _ => unreachable!(
            "gather_one(DataType::Bool, ...) must return GatheredCol::Bool; \
             see alloc_gathered match arm"
        ),
    };

    // Defensive length-equality assertion: both gathers used the same scan,
    // so they MUST have the same length. If they don't, the downstream
    // `download()` zip would silently truncate to the shorter buffer.
    debug_assert_eq!(
        values.len(),
        validity.len(),
        "gather_bool_nullable: values/validity length mismatch despite shared scan"
    );

    Ok(GatheredCol::BoolNullable { values, validity })
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
///
/// ## Nullable bool (W7A8)
///
/// This entry point only takes a single device pointer per column, so it
/// can't directly compact a `BoolNullable` device column (which has a
/// parallel validity buffer the gather pipeline must also visit). The
/// engine should branch on `DeviceCol::BoolNullable` BEFORE calling this
/// function, call [`prefix_scan_mask`] once to amortise the scan, then
/// call [`gather_bool_nullable`] for the bool-nullable column and
/// [`gather_one`] for everything else, assembling the resulting
/// `Vec<GatheredCol>` itself.
///
/// We don't add a `(values, validity, DataType)` overload here because
/// `engine.rs::DeviceCol` is private to the engine module — this file
/// can't pattern-match on it. Keeping the validity wire-up at the
/// engine-callsite layer avoids leaking the variant boundary across
/// modules. (TODO(post-w7): if the engine ever lifts `DeviceCol` to
/// `pub(crate)`, fold the branch into this function so callers stop
/// having to know about the two-launch dance.)
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
    // `Array` is the trait that supplies `.as_any()`, `.len()`,
    // `.null_count()`, `.is_null()` on every concrete Arrow array. The
    // BoolNullable download tests below need it; importing here keeps the
    // test module self-contained without polluting the parent module.
    #[allow(unused_imports)]
    use arrow_array::Array;

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

    /// Pure-host check on the `GatheredCol::BoolNullable` download path: we
    /// hand-build a fake `BoolNullable` GpuVec pair from known bytes,
    /// call `download`, and confirm the resulting Arrow array preserves
    /// per-row null-ness with no `Some(false)` vs `None` collapse. Runs
    /// only with CUDA available because `GpuVec::from_slice` allocates on
    /// the device; under `#[ignore]` so non-GPU CI passes.
    ///
    /// This guards against any future change to `GatheredCol::download`'s
    /// zip logic that would re-introduce the W5A2-pre regression
    /// (dropping validity during compaction).
    #[test]
    #[ignore = "requires CUDA toolkit at runtime (GpuVec::from_slice allocates on device)"]
    fn gathered_bool_nullable_download_preserves_validity() {
        // values:   [1, 0, 0, 1]
        // validity: [1, 0, 1, 0]
        // -> Some(true), None, Some(false), None
        let values = GpuVec::<u8>::from_slice(&[1u8, 0, 0, 1]).expect("upload values");
        let validity = GpuVec::<u8>::from_slice(&[1u8, 0, 1, 0]).expect("upload validity");
        let col = GatheredCol::BoolNullable { values, validity };

        // device_ptr + validity_device_ptr must both surface non-NULL
        // device addresses that DISAGREE (two separate allocations).
        let vptr = col.device_ptr();
        let mptr = col.validity_device_ptr().expect("validity must be Some");
        assert_ne!(vptr, 0, "values device pointer must be non-NULL");
        assert_ne!(mptr, 0, "validity device pointer must be non-NULL");
        assert_ne!(
            vptr, mptr,
            "values and validity must be distinct device allocations"
        );

        assert_eq!(col.len(), 4);
        assert!(!col.is_empty());

        let arr = col.download().expect("download");
        let ba = arr
            .as_any()
            .downcast_ref::<arrow_array::BooleanArray>()
            .expect("BooleanArray");
        assert_eq!(ba.len(), 4);
        assert_eq!(ba.null_count(), 2);
        assert_eq!(ba.is_null(0), false);
        assert_eq!(ba.value(0), true);
        assert!(ba.is_null(1));
        assert_eq!(ba.is_null(2), false);
        assert_eq!(ba.value(2), false);
        assert!(ba.is_null(3));
    }

    /// End-to-end GPU-side test: build a u8 mask + a paired (values,
    /// validity) bool-nullable column, run `prefix_scan_mask`, call
    /// `gather_bool_nullable`, and verify BOTH buffers were gathered with
    /// the same indices — i.e. the value/validity correspondence is
    /// preserved across the GPU prefix-scan + gather pipeline. This is the
    /// W7A8 acceptance test for the GPU path; ignored on non-GPU CI.
    ///
    /// Setup mirrors the host-side `compact_bool_with_nulls_preserves_validity`
    /// test in `compact.rs`:
    ///   Source (6 rows):    [true, null, false, true, null, false]
    ///   Mask (keep/drop):   [keep, keep, drop, keep, keep, drop]
    ///   Expected output:    [Some(true), None, Some(true), None]
    ///
    /// Concretely, the values/validity byte buffers we upload are:
    ///   values:   [1, 0, 0, 1, 0, 0]   (0 for both false and null)
    ///   validity: [1, 0, 1, 1, 0, 1]   (1 = non-null)
    ///   mask:     [1, 1, 0, 1, 1, 0]
    /// After gather the expected device buffers (length 4) are:
    ///   values:   [1, 0, 1, 0]
    ///   validity: [1, 0, 1, 0]
    /// which the download zip then turns into [Some(true), None, Some(true), None].
    #[test]
    #[ignore = "requires CUDA toolkit + driver at runtime"]
    fn gpu_compact_bool_nullable_gathers_both_buffers() {
        let stream = CudaStream::null();

        // Upload mask, values, validity.
        let mask_buf =
            GpuVec::<u8>::from_slice(&[1u8, 1, 0, 1, 1, 0]).expect("upload mask");
        let values_buf =
            GpuVec::<u8>::from_slice(&[1u8, 0, 0, 1, 0, 0]).expect("upload values");
        let validity_buf =
            GpuVec::<u8>::from_slice(&[1u8, 0, 1, 1, 0, 1]).expect("upload validity");

        let n_rows = 6usize;

        // Single prefix scan, shared by both gather launches inside
        // gather_bool_nullable.
        let scan = prefix_scan_mask(mask_buf.device_ptr(), n_rows, &stream)
            .expect("prefix_scan_mask");
        assert_eq!(
            scan.total_count, 4,
            "mask keeps 4 of 6 rows; scan total_count must match"
        );

        let gathered = gather_bool_nullable(
            values_buf.device_ptr(),
            validity_buf.device_ptr(),
            n_rows,
            &scan,
            &stream,
        )
        .expect("gather_bool_nullable");

        // Variant shape: must be BoolNullable, both buffers length 4.
        match &gathered {
            GatheredCol::BoolNullable { values, validity } => {
                assert_eq!(values.len(), 4, "values gathered to total_count rows");
                assert_eq!(
                    validity.len(),
                    4,
                    "validity gathered to total_count rows"
                );
            }
            _ => panic!("expected GatheredCol::BoolNullable, got a different variant"),
        }
        assert_eq!(gathered.len(), 4);
        assert!(!gathered.is_empty());
        // values and validity must occupy distinct device allocations —
        // critical because otherwise the second gather would have
        // clobbered the first.
        assert_ne!(
            gathered.device_ptr(),
            gathered
                .validity_device_ptr()
                .expect("BoolNullable must expose validity"),
        );

        // End-to-end download check: per-row nullness preserved.
        let arr = gathered.download().expect("download");
        let ba = arr
            .as_any()
            .downcast_ref::<arrow_array::BooleanArray>()
            .expect("BooleanArray");
        assert_eq!(ba.len(), 4);
        let expected: Vec<Option<bool>> =
            vec![Some(true), None, Some(true), None];
        let actual: Vec<Option<bool>> = (0..ba.len())
            .map(|i| if ba.is_null(i) { None } else { Some(ba.value(i)) })
            .collect();
        assert_eq!(actual, expected, "per-row validity preserved end-to-end");
        assert_eq!(ba.null_count(), 2);

        // Keep source buffers alive past the assertions — drop here so
        // any CUDA double-free surfaces in this test, not somewhere
        // downstream.
        drop(gathered);
        drop(scan);
        drop(validity_buf);
        drop(values_buf);
        drop(mask_buf);
    }
}
