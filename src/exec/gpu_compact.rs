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
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::CudaStream;
use crate::exec::module_cache;
use crate::exec::n_rows_to_u32;
use crate::jit::prefix_scan::{
    compile_gather_kernel, compile_prefix_scan_kernel, compile_prefix_scan_kernel_blelloch,
    compile_prefix_scan_kernel_lookback, gather_kernel_entry, BLOCK_SIZE, SCAN_KERNEL_ENTRY,
    SCAN_KERNEL_ENTRY_BLELLOCH, SCAN_KERNEL_ENTRY_LOOKBACK,
};
use crate::plan::logical_plan::DataType;
use crate::plan::physical_plan::{
    CompactionKernelKind, CompactionKernelSpec, PrefixScanAlgoTag,
};

/// Build the `CompactionKernelSpec` for a `PrefixScan(algo)` cache
/// lookup. Inline `kind: ..` construction at each call site reads
/// noisy; this helper keeps the spec build at the launch site
/// readable.
fn prefix_scan_spec(algo: PrefixScanAlgoTag) -> CompactionKernelSpec {
    CompactionKernelSpec {
        kind: CompactionKernelKind::PrefixScan(algo),
    }
}

/// Map the runtime algorithm selector to its `PrefixScanAlgoTag`.
/// The two enums carry the same set of variants intentionally —
/// `PrefixScanAlgo` is the local executor-side dispatch tag while
/// `PrefixScanAlgoTag` is the IR-side cache-key tag. Keeping them as
/// distinct types lets the IR live in `plan::physical_plan` without
/// leaking the local `PrefixScanAlgo` dispatch enum out of `exec`.
fn algo_to_tag(algo: PrefixScanAlgo) -> PrefixScanAlgoTag {
    match algo {
        PrefixScanAlgo::HillisSteele => PrefixScanAlgoTag::HillisSteele,
        PrefixScanAlgo::Blelloch => PrefixScanAlgoTag::Blelloch,
        PrefixScanAlgo::Lookback => PrefixScanAlgoTag::Lookback,
    }
}

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
    pub fn download(&self) -> crate::error::BoltResult<arrow_array::ArrayRef> {
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
                    return Err(BoltError::Other(format!(
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
) -> BoltResult<ScanResult> {
    if n_rows == 0 {
        return Ok(ScanResult {
            local_indices: GpuVec::<u32>::empty(),
            block_bases: GpuVec::<u32>::empty(),
            total_count: 0,
            mask_ptr,
            n_rows: 0,
        });
    }
    // Single-pass dispatch limit: ensures `running` fits in u32. Multipass at
    // multipass.rs handles n_rows above this.
    let max_rows = (u32::MAX as usize) / (BLOCK_SIZE as usize);
    if n_rows > max_rows {
        // Single-pass topped out; delegate to the recursive multi-pass path.
        return crate::exec::gpu_compact_multipass::prefix_scan_mask_multipass(
            mask_ptr, n_rows, stream,
        );
    }

    // Decoupled-lookback takes a 5th `partial_status` argument and bakes the
    // global prefix into `local_indices` in one launch, so it has its own
    // launcher. Hillis-Steele / Blelloch share the 4-arg launcher below.
    let (algo, _spec_id, entry) = prefix_scan_algo_selection();
    if algo == PrefixScanAlgo::Lookback {
        return prefix_scan_mask_lookback(mask_ptr, n_rows, stream);
    }

    let block_size = BLOCK_SIZE as usize;
    let n_blocks = n_rows.div_ceil(block_size);

    // Allocate the two device output buffers.
    let local_indices = GpuVec::<u32>::zeros(n_rows)?;
    let block_sums = GpuVec::<u32>::zeros(n_blocks)?;

    // JIT-compile and load the scan kernel via the v0.7
    // `CompactionKernelSpec`-keyed cache. On a warm hit the codegen +
    // PTX-load round-trip is skipped — the cached `CudaModule` clone
    // returns in sub-microsecond time. Algorithm selected by
    // `BOLT_PREFIX_SCAN_ALGO` env var:
    //   * unset / "hillis" / "hillis-steele" -> Hillis-Steele (default, O(n log n))
    //   * "blelloch"                         -> Blelloch upsweep+downsweep (O(n))
    // Both kernels expose the same 4-arg ABI; only the PTX entry name
    // and the kind tag inside the cache key change between them.
    let spec = prefix_scan_spec(algo_to_tag(algo));
    let module = module_cache::get_or_build_module_for_compaction(&spec, entry, |_s| {
        let (ptx, _e) = compile_prefix_scan_for_algo()?;
        Ok(ptx)
    })?;
    let function = module.function(entry)?;

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
    //
    // Defensive: a buggy GPU prefix-scan kernel (or a corrupted download)
    // could in principle push `running` past `u32::MAX`. The downstream
    // `bases_host.push(running as u32)` would then silently wrap, producing
    // a non-monotonic bases vec — and gather_one would scatter writes at
    // the wrong device offset. We accumulate with `checked_add` (on u64,
    // which dominates the per-block u32 sums) and surface overflow as a
    // structured error rather than wrap. Each `running as u32` cast that
    // *does* execute is then guaranteed to be lossless because we bound the
    // accumulator below `u32::MAX` at the previous iteration.
    let sums_host: Vec<u32> = block_sums.to_vec()?;
    let mut bases_host: Vec<u32> = Vec::with_capacity(sums_host.len());
    let mut running: u64 = 0;
    let mut prev_base: u32 = 0;
    for (i, s) in sums_host.iter().enumerate() {
        let base_u32 = u32::try_from(running).map_err(|_| BoltError::Other(format!(
            "gpu_compact: per-block base {running} exceeds u32::MAX at block {i}; \
             this is a kernel-contract violation"
        )))?;
        // Monotonicity guard: bases must be non-decreasing. The u64 accumulator
        // can only ever grow (we add unsigned u32 values), so a decrease here
        // would be a host-arithmetic bug, not a GPU bug — surface it loudly.
        if i > 0 && base_u32 < prev_base {
            return Err(BoltError::Other(format!(
                "gpu_compact: non-monotonic block_bases at block {i}: {base_u32} < {prev_base}"
            )));
        }
        bases_host.push(base_u32);
        prev_base = base_u32;
        running = running
            .checked_add(*s as u64)
            .ok_or_else(|| BoltError::Other(format!(
                "gpu_compact: prefix-sum u64 overflow at block {i} (running={running}, +{s})"
            )))?;
    }
    let total_count = usize::try_from(running).map_err(|_| BoltError::Other(format!(
        "gpu_compact: total_count {running} exceeds usize::MAX on this host"
    )))?;

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

/// Single-pass decoupled-lookback variant of [`prefix_scan_mask`].
///
/// Activated by `BOLT_PREFIX_SCAN_ALGO=lookback` via the dispatch in
/// [`prefix_scan_algo_selection`]. Unlike the Hillis-Steele / Blelloch
/// kernels — which compute per-block sums and require a host
/// download + exclusive-scan + re-upload of `block_sums` — the lookback
/// kernel publishes its block aggregate to a per-block status array and
/// walks the array backwards to derive the global prefix in the same grid
/// launch. The returned `local_indices` already holds the **global**
/// exclusive prefix, so the host pass and `block_bases` allocation are
/// skipped entirely.
///
/// ## Output contract
///
/// The returned [`ScanResult`] has:
///   * `local_indices[gid]` = global exclusive prefix of the mask at row `gid`;
///   * `block_bases` = a length-`n_blocks` u32 buffer filled with zeros.
///     This is load-bearing: the gather kernel reads
///     `block_bases[blockIdx.x] + local_indices[gid]` as the output index,
///     and lookback bakes the block prefix into `local_indices` so the
///     block-bases term must contribute zero to leave the math correct;
///   * `total_count` = the inclusive prefix at row `n_rows - 1` plus the
///     mask byte at that row. The kernel publishes the block-`n_blocks-1`
///     inclusive prefix to `partial_status` and we recover it by reading
///     that slot's low 30 bits after the launch syncs.
///
/// ## Size bound
///
/// The 30-bit value field in each status slot caps any prefix at
/// `(1 << 30) - 1 = 1_073_741_823`. We refuse `n_rows >= 1 << 30` up-front
/// so a saturated prefix is impossible. Larger inputs should use the
/// multipass path (Hillis-Steele) or one of the in-block scans.
fn prefix_scan_mask_lookback(
    mask_ptr: CUdeviceptr,
    n_rows: usize,
    stream: &CudaStream,
) -> BoltResult<ScanResult> {
    // 30-bit value field => max prefix is (1 << 30) - 1. Refuse anything
    // that could conceivably saturate (the prefix never exceeds n_rows
    // since every contribution is 0 or 1).
    const LOOKBACK_ROW_LIMIT: usize = 1 << 30;
    if n_rows >= LOOKBACK_ROW_LIMIT {
        return Err(BoltError::Other(format!(
            "decoupled-lookback scan requires n_rows < 2^30 (got {n_rows}); \
             use multipass for larger inputs"
        )));
    }

    let block_size = BLOCK_SIZE as usize;
    let n_blocks = n_rows.div_ceil(block_size);

    let local_indices = GpuVec::<u32>::zeros(n_rows)?;
    let block_sums = GpuVec::<u32>::zeros(n_blocks)?;
    // partial_status[i] starts at INVALID (= 0). Zero-init is exactly the
    // initial state the kernel expects; any other value would be observed
    // as a stale AGGREGATE/INCLUSIVE by a peer block.
    let partial_status = GpuVec::<u32>::zeros(n_blocks)?;

    // v0.7: route through the `CompactionKernelSpec`-keyed cache.
    // Lookback owns its own `PrefixScanAlgoTag::Lookback` slot so
    // re-entering this function on the warm path skips codegen
    // entirely.
    let spec = prefix_scan_spec(PrefixScanAlgoTag::Lookback);
    let module = module_cache::get_or_build_module_for_compaction(
        &spec,
        SCAN_KERNEL_ENTRY_LOOKBACK,
        |_s| compile_prefix_scan_kernel_lookback(),
    )?;
    let function = module.function(SCAN_KERNEL_ENTRY_LOOKBACK)?;

    let mut p_mask: CUdeviceptr = mask_ptr;
    let mut p_local: CUdeviceptr = local_indices.device_ptr();
    let mut p_block: CUdeviceptr = block_sums.device_ptr();
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let mut p_status: CUdeviceptr = partial_status.device_ptr();

    let mut kernel_params: [*mut c_void; 5] = [
        &mut p_mask as *mut CUdeviceptr as *mut c_void,
        &mut p_local as *mut CUdeviceptr as *mut c_void,
        &mut p_block as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
        &mut p_status as *mut CUdeviceptr as *mut c_void,
    ];

    let grid_x: u32 = n_rows_to_u32(n_blocks)?;
    // SAFETY: every entry in `kernel_params` points at a stack local that
    // outlives the launch+synchronize below; `function` is borrowed from a
    // live `CudaModule`. The kernel writes only to `local_indices`,
    // `block_sums`, and `partial_status`, all owned here.
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

    // Total count: read `partial_status[n_blocks - 1]`'s low 30 bits. The
    // last block's PUBLISH_INCLUSIVE stored `(block_prefix +
    // block_aggregate) & VALUE_MASK` there, which is the inclusive prefix
    // at row `n_rows - 1`. Equivalently `n_rows - mask_zero_count`.
    let status_host: Vec<u32> = partial_status.to_vec()?;
    debug_assert_eq!(status_host.len(), n_blocks);
    let last_slot = *status_host
        .last()
        .expect("n_blocks > 0 because n_rows > 0 and div_ceil");
    let total_count = (last_slot & crate::jit::prefix_scan::LOOKBACK_VALUE_MASK) as usize;

    // `block_bases` is zero-filled: lookback baked the block prefix into
    // `local_indices`, so the gather kernel's `block_bases + local_indices`
    // sum must contribute only the local term. We still allocate it (rather
    // than reusing one global zero buffer) so the `ScanResult` API stays
    // uniform across all three algorithms.
    let block_bases = GpuVec::<u32>::zeros(n_blocks)?;

    drop(block_sums);
    drop(partial_status);
    drop(status_host);

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
///
/// This call is synchronous: it launches the gather kernel and then
/// `stream.synchronize()`s before returning. Callers that want to batch
/// multiple gather launches back-to-back on the same stream should use
/// [`gather_one_async`] instead and synchronize once at the end of the
/// batch. See [`compact_columns_on_gpu`] for the canonical batched pattern.
pub fn gather_one(
    input_ptr: CUdeviceptr,
    n_rows: usize,
    scan: &ScanResult,
    dtype: DataType,
    stream: &CudaStream,
) -> BoltResult<GatheredCol> {
    // Delegate the launch to `gather_one_async`, then synchronize once so the
    // returned `GatheredCol` is host-observable. The split exists so callers
    // batching N columns can amortise the sync; this thin wrapper preserves
    // the single-shot contract for callers that aren't ready to manage their
    // own synchronization point.
    let col = gather_one_async(input_ptr, n_rows, scan, dtype, stream)?;
    stream.synchronize()?;
    Ok(col)
}

/// Asynchronous variant of [`gather_one`]: launches the gather kernel on the
/// given stream and returns the freshly allocated output column **without
/// synchronizing**. The caller MUST call `stream.synchronize()` (or otherwise
/// wait for the stream) before reading the output buffer from the host or
/// dropping the inputs.
///
/// ## Why this is safe to chain without per-launch sync
///
/// The gather kernel ABI reads `(mask, local_indices, block_bases, input)`
/// and writes only to `output`. Across back-to-back launches on the same
/// stream:
///   * `mask`, `local_indices`, `block_bases` come from the shared `scan` and
///     are READ-ONLY in the kernel — no WAW hazard between launches that
///     share them, and no RAW hazard either (no launch writes them).
///   * `input` is per-launch and is also READ-ONLY in the kernel.
///   * `output` is per-launch (each call allocates a fresh `GpuVec` via
///     `alloc_gathered`), so no two launches write to overlapping bytes —
///     no WAW hazard.
///   * The output of launch *i* is never read as input by launch *j>i* in
///     this pipeline, so no cross-launch RAW hazard either.
///
/// Combined with the CUDA stream's in-order execution guarantee, this means
/// the kernel calls can be enqueued back-to-back and a single
/// `stream.synchronize()` at the end of the batch is sufficient to make all
/// outputs host-observable. The per-launch sync that `gather_one` does is
/// purely a convenience for single-shot callers; in batched paths it costs
/// one host-device round trip per column for no correctness benefit.
pub fn gather_one_async(
    input_ptr: CUdeviceptr,
    n_rows: usize,
    scan: &ScanResult,
    dtype: DataType,
    stream: &CudaStream,
) -> BoltResult<GatheredCol> {
    if matches!(dtype, DataType::Utf8) {
        return Err(BoltError::Other(
            "gpu_compact: gather Utf8 not supported (variable-width)".into(),
        ));
    }
    if scan.n_rows != n_rows {
        return Err(BoltError::Other(format!(
            "gpu_compact: scan.n_rows={} mismatches input n_rows={}",
            scan.n_rows, n_rows
        )));
    }
    if scan.local_indices.len() != n_rows {
        return Err(BoltError::Other(format!(
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

    // JIT-compile + load the gather kernel for this dtype via the
    // v0.7 `CompactionKernelSpec`-keyed cache. The PTX is keyed by
    // `dtype` only; the `Gather(dtype)` variant of
    // `CompactionKernelKind` owns one cache slot per supported
    // dtype so back-to-back gather launches over a wide projection
    // hit the warm path after the first launch.
    let spec = CompactionKernelSpec {
        kind: CompactionKernelKind::Gather(dtype),
    };
    let module = module_cache::get_or_build_module_for_compaction(
        &spec,
        gather_kernel_entry(dtype),
        |s| match s.kind {
            CompactionKernelKind::Gather(d) => compile_gather_kernel(d),
            // Other variants never reach this closure because the
            // `spec` we built above has `Gather(dtype)` baked in;
            // surface a structured error rather than panic if a
            // future refactor accidentally widens the spec.
            other => Err(BoltError::Other(format!(
                "gpu_compact: gather closure invoked on non-Gather spec {:?}",
                other
            ))),
        },
    )?;
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

    // SAFETY: each kernel_params entry points at a live stack local that
    // outlives this `cuLaunchKernel` call (the driver copies the args before
    // returning); `function` is borrowed from a live `CudaModule`; `stream`
    // is live. The device buffers behind every pointer must outlive the
    // *eventual* stream synchronize the caller is responsible for — the
    // caller owns mask/input, `scan` owns local/bases, and `col` owns the
    // output, all moved/returned past this call.
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
) -> BoltResult<GatheredCol> {
    // Two independent gather launches, both keyed off the same `scan`. The
    // kernel ABI handles one buffer at a time; we re-use the scan products
    // (local_indices + block_bases + mask_ptr + total_count) so the second
    // launch is just `compile_gather_kernel(Bool)` again on a different
    // input pointer. JIT caching at the `compile_gather_kernel` layer
    // means we don't re-compile the PTX for the second call in practice.
    //
    // Use `gather_one_async` for both and synchronize once at the end: the
    // two launches write to disjoint output buffers (`values` and
    // `validity`) and only READ the shared scan + mask, so back-to-back
    // launches on the same stream are hazard-free. See
    // `gather_one_async`'s safety comment for the full RAW/WAW argument.
    let gathered_values = gather_one_async(values_ptr, n_rows, scan, DataType::Bool, stream)?;
    let gathered_validity =
        gather_one_async(validity_ptr, n_rows, scan, DataType::Bool, stream)?;
    stream.synchronize()?;

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
/// `Utf8` columns return [`BoltError::Other`] — the gather kernel can only
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
) -> BoltResult<(Vec<GatheredCol>, usize)> {
    // Validate dtypes BEFORE launching the scan so a Utf8 column can't waste a
    // kernel launch + sync. `prefix_scan_mask` already short-circuits on
    // n_rows == 0, so the empty-columns + zero-rows path costs just the scan
    // call's early return and the Vec allocation below.
    for (_, dtype) in columns {
        if matches!(dtype, DataType::Utf8) {
            return Err(BoltError::Other(
                "Utf8 gather not supported on GPU (use host-side compact_arrays)".into(),
            ));
        }
    }

    let scan = prefix_scan_mask(mask_ptr, n_rows, stream)?;

    // Batch the per-column gather launches: enqueue all N kernels on the
    // same stream without per-launch synchronize, then do ONE
    // `stream.synchronize()` at the end. The gather kernel is a pure
    // read-then-write with disjoint output buffers per column (see
    // `gather_one_async` for the full hazard argument), so back-to-back
    // launches are safe under CUDA's in-stream ordering. The win versus the
    // old per-launch sync is N-1 host-device round trips per
    // `compact_columns_on_gpu` call — material on wide projections.
    let mut out: Vec<GatheredCol> = Vec::with_capacity(columns.len());
    for (ptr, dtype) in columns {
        out.push(gather_one_async(*ptr, n_rows, &scan, *dtype, stream)?);
    }
    if !out.is_empty() {
        // Skip the sync when no launches happened: `gather_one_async`
        // short-circuits to a zero-length allocation when
        // `scan.total_count == 0` OR `n_rows == 0`, but a non-empty
        // `columns` slice with a non-empty scan still enqueues at least one
        // kernel — sync to make the outputs host-observable. The
        // `is_empty` guard is just to avoid a no-op syscall in the
        // pathological N-column-but-all-empty case.
        stream.synchronize()?;
    }

    Ok((out, scan.total_count))
}

/// Identifier for which prefix-scan algorithm should service the current
/// call. Selected from the `BOLT_PREFIX_SCAN_ALGO` env var on every call —
/// see [`prefix_scan_algo_selection`] for the resolution rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefixScanAlgo {
    /// O(n log n) ping-pong scan ([`SCAN_KERNEL_ENTRY`]). Default.
    HillisSteele,
    /// O(n) upsweep/downsweep scan ([`SCAN_KERNEL_ENTRY_BLELLOCH`]).
    Blelloch,
    /// Single-pass decoupled-lookback scan ([`SCAN_KERNEL_ENTRY_LOOKBACK`]).
    /// Folds the per-block reduce + global scan into one grid launch — no
    /// host round-trip on `block_sums`.
    Lookback,
}

/// Env-var-driven dispatch between the three prefix-scan kernels. Returns
/// the resolved [`PrefixScanAlgo`] plus matching `(spec_id, entry_name)` for
/// the consolidated module cache.
///
/// Selection rules for `BOLT_PREFIX_SCAN_ALGO`:
///   * `"blelloch"` (case-insensitive) -> Blelloch upsweep+downsweep
///     (O(n) work). New code path; not the default.
///   * `"lookback"` (case-insensitive) -> single-pass decoupled-lookback
///     ([`SCAN_KERNEL_ENTRY_LOOKBACK`]). Routes through
///     [`prefix_scan_mask_lookback`] which allocates an extra
///     `partial_status` buffer; the kernel returns GLOBAL exclusive
///     prefixes directly in `local_indices`.
///   * Unset, `"hillis"`, `"hillis-steele"`, or any other value ->
///     Hillis-Steele (default; O(n log n) work, in production use since the
///     initial GPU compaction landing).
///
/// The Hillis-Steele default is intentional while the alternative kernels
/// are in shake-out: the existing path has soak time across the e2e tests,
/// and the host-side validation we can do without a GPU (substring + golden
/// tests) only catches structural regressions, not numerical ones. Once the
/// alternative paths have been exercised end-to-end on real hardware the
/// default should flip and this helper can collapse to a single kernel.
fn prefix_scan_algo_selection() -> (PrefixScanAlgo, &'static str, &'static str) {
    let env = std::env::var("BOLT_PREFIX_SCAN_ALGO").ok();
    match env.as_deref() {
        Some(s) if s.eq_ignore_ascii_case("blelloch") => (
            PrefixScanAlgo::Blelloch,
            "prefix_scan_blelloch",
            SCAN_KERNEL_ENTRY_BLELLOCH,
        ),
        Some(s) if s.eq_ignore_ascii_case("lookback") => (
            PrefixScanAlgo::Lookback,
            "prefix_scan_lookback",
            SCAN_KERNEL_ENTRY_LOOKBACK,
        ),
        _ => (
            PrefixScanAlgo::HillisSteele,
            "prefix_scan",
            SCAN_KERNEL_ENTRY,
        ),
    }
}

fn compile_prefix_scan_for_algo() -> BoltResult<(String, &'static str)> {
    // Read at every call. Cheap (an env lookup) and lets the algorithm be
    // changed without restart for ad-hoc benchmarking. If this lookup ever
    // shows up in a flamegraph the right fix is to cache the resolved
    // choice in a `OnceLock`, not to bake the default at compile time.
    //
    // Lookback is dispatched through `prefix_scan_mask_lookback` because it
    // takes a 5th `partial_status` argument; the 4-arg call sites in
    // `prefix_scan_mask` use only the other two variants.
    let (algo, _spec, _entry) = prefix_scan_algo_selection();
    match algo {
        PrefixScanAlgo::Blelloch => Ok((
            compile_prefix_scan_kernel_blelloch()?,
            SCAN_KERNEL_ENTRY_BLELLOCH,
        )),
        PrefixScanAlgo::Lookback | PrefixScanAlgo::HillisSteele => {
            // For `HillisSteele` this is the genuine path. The `Lookback`
            // arm should never reach here in practice (`prefix_scan_mask`
            // delegates to `prefix_scan_mask_lookback` first); if it does
            // we fall back to Hillis-Steele rather than emit a kernel with
            // the wrong ABI for the 4-arg launcher.
            Ok((compile_prefix_scan_kernel()?, SCAN_KERNEL_ENTRY))
        }
    }
}

/// Allocate a `GpuVec<T>` matching `dtype` with `len` elements and wrap it.
fn alloc_gathered(dtype: DataType, len: usize) -> BoltResult<GatheredCol> {
    Ok(match dtype {
        DataType::Bool => GatheredCol::Bool(GpuVec::<u8>::zeros(len)?),
        DataType::Int32 => GatheredCol::I32(GpuVec::<i32>::zeros(len)?),
        DataType::Int64 => GatheredCol::I64(GpuVec::<i64>::zeros(len)?),
        DataType::Float32 => GatheredCol::F32(GpuVec::<f32>::zeros(len)?),
        DataType::Float64 => GatheredCol::F64(GpuVec::<f64>::zeros(len)?),
        DataType::Utf8 => {
            return Err(BoltError::Other(
                "gpu_compact: gather Utf8 not supported (variable-width)".into(),
            ))
        }
        DataType::Decimal128(_, _) => {
            return Err(BoltError::Plan(
                "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
            ))
        }
        DataType::Date32 | DataType::Timestamp(_, _) => {
            return Err(BoltError::Other(
                "Date/Timestamp not yet lowered to GPU".into(),
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

    /// Mirror of the in-kernel-driver post-scan with the same defensive checks
    /// the production path applies: overflow surfaces as `Err`, monotonicity
    /// is enforced. Keeping it as a local helper avoids exposing the
    /// arithmetic on the public surface while still letting the unit tests
    /// pin down the contract.
    fn host_exclusive_scan_checked(sums: &[u32]) -> BoltResult<(Vec<u32>, usize)> {
        let mut bases = Vec::with_capacity(sums.len());
        let mut running: u64 = 0;
        let mut prev_base: u32 = 0;
        for (i, s) in sums.iter().enumerate() {
            if running > u32::MAX as u64 {
                return Err(BoltError::Other(format!(
                    "gpu_compact: prefix-sum overflowed u32 at block {i} \
                     (running={running}, total exceeds u32::MAX)"
                )));
            }
            let base_u32 = running as u32;
            if i > 0 && base_u32 < prev_base {
                return Err(BoltError::Other(format!(
                    "gpu_compact: non-monotonic block_bases at block {i}: {base_u32} < {prev_base}"
                )));
            }
            bases.push(base_u32);
            prev_base = base_u32;
            running = running
                .checked_add(*s as u64)
                .ok_or_else(|| BoltError::Other(format!(
                    "gpu_compact: prefix-sum u64 overflow at block {i} (running={running}, +{s})"
                )))?;
        }
        if running > usize::MAX as u64 {
            return Err(BoltError::Other(format!(
                "gpu_compact: total_count {running} exceeds usize::MAX"
            )));
        }
        Ok((bases, running as usize))
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

    // --- Defensive: checked scan -------------------------------------------
    //
    // Mirrors the host post-scan in `prefix_scan_mask`. Well-formed inputs
    // must match the un-checked oracle exactly; pathological inputs must
    // surface a structured error rather than silently wrap.
    // -----------------------------------------------------------------------

    #[test]
    fn host_scan_checked_matches_unchecked_on_clean_inputs() {
        let sums = vec![3u32, 0, 5, 256, 1, 9, 9];
        let (bases_unchecked, total_unchecked) = host_exclusive_scan(&sums);
        let (bases_checked, total_checked) =
            host_exclusive_scan_checked(&sums).expect("clean input must pass");
        assert_eq!(bases_checked, bases_unchecked);
        assert_eq!(total_checked, total_unchecked);
    }

    #[test]
    fn host_scan_checked_rejects_u32_overflow() {
        // Three large block sums force the prefix to exceed u32::MAX on
        // iteration 2 (running = u32::MAX + u32::MAX > u32::MAX), at which
        // point the checked cast must surface a structured error rather
        // than silently wrap.
        let sums = vec![u32::MAX, u32::MAX, 1];
        let err = host_exclusive_scan_checked(&sums).expect_err(
            "triple-u32::MAX must overflow the u32 base cast on iteration 2",
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("overflowed u32") || msg.contains("u64 overflow"),
            "error must mention u32/u64 overflow, got: {msg}"
        );
    }

    #[test]
    fn host_scan_checked_zero_sums_all_monotonic() {
        // All-zero per-block sums is the empty-mask case: bases are all 0,
        // total is 0, and no monotonicity violation is possible.
        let sums = vec![0u32; 16];
        let (bases, total) =
            host_exclusive_scan_checked(&sums).expect("all-zero must pass");
        assert!(bases.iter().all(|&b| b == 0));
        assert_eq!(total, 0);
    }

    #[test]
    #[ignore = "gpu:projection — zeros"]
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
    #[ignore = "gpu:projection — GpuVec::from_slice allocates on device"]
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
    #[ignore = "gpu:projection"]
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

    /// End-to-end GPU-side test for the batched multi-column gather path:
    /// build an 8-row mask plus 3 fixed-width columns of different dtypes
    /// (`Int32`, `Float64`, `Bool`), run `compact_columns_on_gpu`, and
    /// verify all three outputs are correct AND were produced by ONE
    /// end-of-batch synchronize rather than three per-column syncs.
    ///
    /// The correctness contract we exercise here is the same one
    /// `gather_one_async`'s safety comment claims: the three kernel
    /// launches share `mask` / `local_indices` / `block_bases` read-only
    /// and each writes to its own disjoint output buffer, so batched
    /// launches must produce byte-identical results to the old per-launch
    /// synchronize loop.
    ///
    /// Setup (8 rows):
    ///   col_i32:   [10, 20, 30, 40, 50, 60, 70, 80]
    ///   col_f64:   [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    ///   col_bool:  [1, 0, 1, 0, 1, 0, 1, 0]   (encoded as u8)
    ///   mask:      [1, 0, 1, 1, 0, 1, 0, 1]   -> keeps rows 0, 2, 3, 5, 7
    /// Expected outputs (length 5):
    ///   col_i32:   [10, 30, 40, 60, 80]
    ///   col_f64:   [1.5, 3.5, 4.5, 6.5, 8.5]
    ///   col_bool:  [1, 1, 0, 0, 0]
    #[test]
    #[ignore = "gpu:projection"]
    fn gpu_compact_three_columns_batched_launches() {
        let stream = CudaStream::null();

        // Upload mask + three input columns of distinct dtypes.
        let mask_buf =
            GpuVec::<u8>::from_slice(&[1u8, 0, 1, 1, 0, 1, 0, 1]).expect("upload mask");
        let col_i32 =
            GpuVec::<i32>::from_slice(&[10i32, 20, 30, 40, 50, 60, 70, 80])
                .expect("upload i32");
        let col_f64 =
            GpuVec::<f64>::from_slice(&[1.5f64, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
                .expect("upload f64");
        let col_bool = GpuVec::<u8>::from_slice(&[1u8, 0, 1, 0, 1, 0, 1, 0])
            .expect("upload bool");

        let n_rows = 8usize;
        let columns = [
            (col_i32.device_ptr(), DataType::Int32),
            (col_f64.device_ptr(), DataType::Float64),
            (col_bool.device_ptr(), DataType::Bool),
        ];

        let (gathered, total) =
            compact_columns_on_gpu(mask_buf.device_ptr(), n_rows, &columns, &stream)
                .expect("compact_columns_on_gpu");
        assert_eq!(total, 5, "mask keeps 5 of 8 rows");
        assert_eq!(gathered.len(), 3, "one GatheredCol per input column");

        // Outputs preserve column order. Each column's allocation is
        // disjoint from the others; if the batched-launch path ever
        // started reusing a single scratch buffer this assertion would
        // catch it.
        let i32_ptr = gathered[0].device_ptr();
        let f64_ptr = gathered[1].device_ptr();
        let bool_ptr = gathered[2].device_ptr();
        assert_ne!(i32_ptr, f64_ptr, "i32 / f64 outputs must be distinct");
        assert_ne!(f64_ptr, bool_ptr, "f64 / bool outputs must be distinct");
        assert_ne!(i32_ptr, bool_ptr, "i32 / bool outputs must be distinct");

        // i32 column.
        match &gathered[0] {
            GatheredCol::I32(v) => {
                assert_eq!(v.len(), 5);
                let host = v.to_vec().expect("download i32");
                assert_eq!(host, vec![10i32, 30, 40, 60, 80]);
            }
            _ => panic!("col 0 must be GatheredCol::I32"),
        }

        // f64 column.
        match &gathered[1] {
            GatheredCol::F64(v) => {
                assert_eq!(v.len(), 5);
                let host = v.to_vec().expect("download f64");
                assert_eq!(host, vec![1.5f64, 3.5, 4.5, 6.5, 8.5]);
            }
            _ => panic!("col 1 must be GatheredCol::F64"),
        }

        // bool column (u8-encoded).
        match &gathered[2] {
            GatheredCol::Bool(v) => {
                assert_eq!(v.len(), 5);
                let host = v.to_vec().expect("download bool");
                assert_eq!(host, vec![1u8, 1, 0, 0, 0]);
            }
            _ => panic!("col 2 must be GatheredCol::Bool"),
        }

        // Drop in reverse to surface any CUDA double-free here rather
        // than at end-of-test.
        drop(gathered);
        drop(col_bool);
        drop(col_f64);
        drop(col_i32);
        drop(mask_buf);
    }
}
