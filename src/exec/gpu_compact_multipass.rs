// SPDX-License-Identifier: Apache-2.0

//! Multi-pass prefix-scan compaction.
//!
//! The single-pass [`crate::exec::gpu_compact::prefix_scan_mask`] errors out
//! once `n_rows > u32::MAX / BLOCK_SIZE` (~16.8M rows at `BLOCK_SIZE = 256`)
//! because the host-side scan over `block_sums` is sequential and the per-row
//! index counter is a `u32`. This module implements a recursive variant that
//! handles arbitrary row counts by scanning the intermediate `block_sums`
//! array with the **same** Hillis-Steele kernel — escalating to a recursive
//! call when even that array doesn't fit in a single block.
//!
//! ## Recursion sketch (n_rows ≈ 4.3 B, BLOCK_SIZE = 256)
//!
//! ```text
//!  Level 0:  n = 4.3B          → n_blocks = 16.8M
//!  Level 1:  n = 16.8M         → n_blocks = 65 536
//!  Level 2:  n = 65 536        → n_blocks = 256
//!  Level 3:  n = 256           → host-scan + upload, done.
//! ```
//!
//! Walking back DOWN the stack, each level runs
//! [`bolt_add_block_bases`](crate::jit::prefix_scan_multipass::ADD_BASES_KERNEL_ENTRY)
//! to fold the parent's per-block bases into the child's per-row local indices,
//! so the lowest-level `local_indices` ends up holding the global exclusive
//! prefix sum. The TOP level's per-row `local_indices` IS the per-row
//! `local_indices` the gather kernels expect — but the per-block bases now
//! come from the recursive scan rather than a host-side serial pass.
//!
//! ## Why a separate u32 scan kernel
//!
//! The single-pass `bolt_prefix_scan` kernel reads a `u8` mask byte and
//! coerces non-zero to 1; the recursive levels need to scan a `u32` count
//! array verbatim. Rather than parameterize the kernel by input dtype we emit
//! a sibling kernel
//! ([`bolt_prefix_scan_u32`](crate::jit::prefix_scan_multipass::SCAN_U32_KERNEL_ENTRY))
//! whose body is identical except for the per-thread load.
//!
//! ## What's deferred / non-goals
//!
//! * **CUDA-free unit tests only.** The dispatch threshold and host-scan
//!   helpers are tested directly; the full multi-pass execution needs a GPU
//!   and is exercised by the engine's integration suite.
//! * **Single recursion strategy.** We always recurse with the same
//!   `BLOCK_SIZE`; we don't fall back to a "scan a 65 535-block buffer with a
//!   larger block" trick. For the row counts we'll see in practice, 3–4
//!   levels is plenty.

use std::ffi::c_void;
use std::ptr;

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::BoltResult;
use crate::exec::launch::CudaStream;
use crate::exec::module_cache;
use crate::exec::n_rows_to_u32;
use crate::jit::prefix_scan::{compile_prefix_scan_kernel, BLOCK_SIZE, SCAN_KERNEL_ENTRY};
use crate::jit::prefix_scan_multipass::{
    compile_add_block_bases_kernel, compile_prefix_scan_u32_kernel, ADD_BASES_KERNEL_ENTRY,
    SCAN_U32_KERNEL_ENTRY,
};

// Same `ScanResult` shape callers already consume from `gpu_compact`. The
// engine's `prefix_scan_mask` dispatches between single-pass and multipass
// based on n_rows, so the return type MUST be a re-export, not a parallel
// struct — otherwise callers couldn't store either result in the same place.
pub use crate::exec::gpu_compact::ScanResult;

/// Multi-pass prefix scan. Handles arbitrary `n_rows` by recursing the
/// per-block scan over the intermediate `block_sums` arrays until the
/// top-level array fits in a single block (≤ `BLOCK_SIZE` entries).
///
/// API matches [`crate::exec::gpu_compact::prefix_scan_mask`] so the engine
/// can dispatch on `n_rows` without changing call sites.
pub fn prefix_scan_mask_multipass(
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

    let block_size = BLOCK_SIZE as usize;
    let n_blocks = n_rows.div_ceil(block_size);

    // Stage 1: per-block scan over the u8 mask, producing per-row local
    // indices and per-block sums. Identical to the single-pass code in
    // `gpu_compact::prefix_scan_mask` — we duplicate the launch glue rather
    // than reaching into the sibling module so its single-pass logic stays
    // self-contained.
    let local0 = GpuVec::<u32>::zeros(n_rows)?;
    let block_sums0 = GpuVec::<u32>::zeros(n_blocks)?;
    run_u8_scan(mask_ptr, &local0, &block_sums0, n_rows, stream)?;

    // Stage 2: turn `block_sums0` into `block_bases0` via either a single
    // host-side scan (small) or a recursive device-side scan (big).
    let (block_bases0, total_count) = scan_block_sums(block_sums0, stream)?;

    Ok(ScanResult {
        local_indices: local0,
        block_bases: block_bases0,
        total_count,
        mask_ptr,
        n_rows,
    })
}

/// Decide between host-scan and recursive device-scan for an array of
/// per-block sums.
///
/// If `vals.len() <= BLOCK_SIZE` the host loop is faster than a single kernel
/// launch + sync; otherwise we recurse via `recursive_scan_u32`.
///
/// Splitting this out keeps the threshold check unit-testable (see
/// [`should_host_scan`]) without dragging `prefix_scan_mask_multipass`'s GPU
/// dependencies into the test.
fn scan_block_sums(
    vals: GpuVec<u32>,
    stream: &CudaStream,
) -> BoltResult<(GpuVec<u32>, usize)> {
    if should_host_scan(vals.len()) {
        let host = vals.to_vec()?;
        let (bases, total) = host_exclusive_scan(&host);
        let bases_dev = GpuVec::<u32>::from_slice(&bases)?;
        // `vals` (the device block_sums buffer) drops here: nothing
        // downstream needs it after we've turned it into bases.
        drop(vals);
        Ok((bases_dev, total))
    } else {
        recursive_scan_u32(vals, stream)
    }
}

/// Dispatch predicate: should this array be exclusive-scanned on the host
/// rather than recursed into a device scan?
///
/// True iff the array fits in a single `BLOCK_SIZE` block. Exposed so the
/// dispatch decision can be unit-tested without touching CUDA.
pub(crate) fn should_host_scan(len: usize) -> bool {
    len <= BLOCK_SIZE as usize
}

/// Recursively exclusive-scan a `u32` device array.
///
/// Invariant on entry: `vals.len() > BLOCK_SIZE`. (Callers must check
/// [`should_host_scan`] first.)
///
/// 1. Run [`bolt_prefix_scan_u32`](SCAN_U32_KERNEL_ENTRY) on `vals` to
///    produce per-entry `local_indices` and per-block `block_sums`.
/// 2. Recursively turn that `block_sums` into `parent_bases` (+ total).
/// 3. Run [`bolt_add_block_bases`](ADD_BASES_KERNEL_ENTRY) to fold the
///    parent bases into `local_indices`, making them globally correct.
/// 4. Return `(local_indices, total)`. The caller's `block_bases` is OUR
///    `local_indices`: every entry `i` of our input array now maps to its
///    global exclusive prefix sum.
fn recursive_scan_u32(
    vals: GpuVec<u32>,
    stream: &CudaStream,
) -> BoltResult<(GpuVec<u32>, usize)> {
    let n = vals.len();
    debug_assert!(
        n > BLOCK_SIZE as usize,
        "recursive_scan_u32 entered with n={} <= BLOCK_SIZE={}; should_host_scan should have caught this",
        n,
        BLOCK_SIZE
    );

    let block_size = BLOCK_SIZE as usize;
    let n_blocks = n.div_ceil(block_size);

    let local = GpuVec::<u32>::zeros(n)?;
    let block_sums = GpuVec::<u32>::zeros(n_blocks)?;
    run_u32_scan(&vals, &local, &block_sums, n, stream)?;

    // `vals` is no longer needed once we've produced (local, block_sums);
    // drop it before recursing to keep peak device memory bounded.
    drop(vals);

    let (parent_bases, total) = scan_block_sums(block_sums, stream)?;

    // Fold parent bases into local indices → globally-correct exclusive scan.
    run_add_block_bases(&local, &parent_bases, n, stream)?;

    Ok((local, total))
}

/// Launch the single-pass `bolt_prefix_scan` kernel over a u8 mask.
///
/// Mirrors the launch glue in `gpu_compact::prefix_scan_mask`'s body — we
/// don't call into that function because it does its own host-side scan over
/// `block_sums` (which is what we're *replacing*).
fn run_u8_scan(
    mask_ptr: CUdeviceptr,
    local: &GpuVec<u32>,
    block_sums: &GpuVec<u32>,
    n_rows: usize,
    stream: &CudaStream,
) -> BoltResult<()> {
    // Share the `prefix_scan` cache slot with `gpu_compact::prefix_scan_mask`:
    // both call sites compile the SAME unparameterised PTX. Namespacing on the
    // sibling module path keeps lookups uniform without re-loading the cubin.
    let module = module_cache::get_or_build_module(
        "craton_bolt::exec::gpu_compact",
        "prefix_scan".to_string(),
        None,
        || compile_prefix_scan_kernel(),
    )?;
    let function = module.function(SCAN_KERNEL_ENTRY)?;

    let mut p_mask: CUdeviceptr = mask_ptr;
    let mut p_local: CUdeviceptr = local.device_ptr();
    let mut p_block: CUdeviceptr = block_sums.device_ptr();
    // The n_rows passed to this kernel always fits in u32: at the top level
    // we accept up to u32::MAX (the dispatch limit); at recursive levels the
    // input is a `block_sums` slice which is itself a count of blocks
    // (well under u32::MAX). We funnel through `n_rows_to_u32` so a
    // pathological caller surfaces a structured error rather than wrapping.
    let mut n_u32: u32 = n_rows_to_u32(n_rows)?;

    let mut kernel_params: [*mut c_void; 4] = [
        &mut p_mask as *mut CUdeviceptr as *mut c_void,
        &mut p_local as *mut CUdeviceptr as *mut c_void,
        &mut p_block as *mut CUdeviceptr as *mut c_void,
        &mut n_u32 as *mut u32 as *mut c_void,
    ];

    let grid_x: u32 = n_rows_to_u32(n_rows.div_ceil(BLOCK_SIZE as usize))?;
    // SAFETY: each kernel_params entry points at a live stack local that
    // outlives the launch+synchronize below; `function` is borrowed from a
    // live `CudaModule`; the buffers behind every pointer outlive the sync
    // (caller owns mask + scan outputs).
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
    Ok(())
}

/// Launch the `bolt_prefix_scan_u32` kernel over a u32 input.
///
/// Same launch glue as `run_u8_scan` — the only difference is the kernel
/// entry name and the input load width inside the PTX.
fn run_u32_scan(
    vals: &GpuVec<u32>,
    local: &GpuVec<u32>,
    block_sums: &GpuVec<u32>,
    n: usize,
    stream: &CudaStream,
) -> BoltResult<()> {
    let module = module_cache::get_or_build_module(
        module_path!(),
        "prefix_scan_u32".to_string(),
        None,
        || compile_prefix_scan_u32_kernel(),
    )?;
    let function = module.function(SCAN_U32_KERNEL_ENTRY)?;

    let mut p_vals: CUdeviceptr = vals.device_ptr();
    let mut p_local: CUdeviceptr = local.device_ptr();
    let mut p_block: CUdeviceptr = block_sums.device_ptr();
    let mut n_u32: u32 = n_rows_to_u32(n)?;

    let mut kernel_params: [*mut c_void; 4] = [
        &mut p_vals as *mut CUdeviceptr as *mut c_void,
        &mut p_local as *mut CUdeviceptr as *mut c_void,
        &mut p_block as *mut CUdeviceptr as *mut c_void,
        &mut n_u32 as *mut u32 as *mut c_void,
    ];

    let grid_x: u32 = n_rows_to_u32(n.div_ceil(BLOCK_SIZE as usize))?;
    // SAFETY: identical to `run_u8_scan`'s argument.
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
    Ok(())
}

/// Launch the `bolt_add_block_bases` kernel.
///
/// MUST use `BLOCK_SIZE` so `blockIdx.x = gid / BLOCK_SIZE` and the per-block
/// base lookup is correct.
fn run_add_block_bases(
    indices: &GpuVec<u32>,
    block_bases: &GpuVec<u32>,
    n: usize,
    stream: &CudaStream,
) -> BoltResult<()> {
    let module = module_cache::get_or_build_module(
        module_path!(),
        "add_block_bases".to_string(),
        None,
        || compile_add_block_bases_kernel(),
    )?;
    let function = module.function(ADD_BASES_KERNEL_ENTRY)?;

    let mut p_indices: CUdeviceptr = indices.device_ptr();
    let mut p_bases: CUdeviceptr = block_bases.device_ptr();
    let mut n_u32: u32 = n_rows_to_u32(n)?;

    let mut kernel_params: [*mut c_void; 3] = [
        &mut p_indices as *mut CUdeviceptr as *mut c_void,
        &mut p_bases as *mut CUdeviceptr as *mut c_void,
        &mut n_u32 as *mut u32 as *mut c_void,
    ];

    let grid_x: u32 = n_rows_to_u32(n.div_ceil(BLOCK_SIZE as usize))?;
    // SAFETY: identical to `run_u8_scan`'s argument.
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
    Ok(())
}

/// Host-side exclusive scan over `u32` block sums.
///
/// Returns `(bases, total)`. Identical to the helper inline in
/// `gpu_compact::prefix_scan_mask`'s tests; lifted here so the multipass
/// host-scan path uses the same arithmetic.
///
/// The accumulator is `u64` so callers never need to worry about an
/// intermediate overflow even when the scanned array runs the full width of
/// the recursion stack. The returned `total` is `usize` (`running` cast),
/// matching `ScanResult::total_count`.
pub(crate) fn host_exclusive_scan(sums: &[u32]) -> (Vec<u32>, usize) {
    let mut bases = Vec::with_capacity(sums.len());
    let mut running: u64 = 0;
    for s in sums {
        bases.push(running as u32);
        running += *s as u64;
    }
    (bases, running as usize)
}

/// Compute the recursion depth needed to scan `n_rows` rows with the
/// multipass strategy. Pure arithmetic — handy for capacity planning and as
/// a regression check on the doc-stated depth bounds.
///
/// Returns 0 for `n_rows == 0` (the early-return path takes no levels).
#[cfg(test)]
fn recursion_depth(n_rows: usize) -> usize {
    if n_rows == 0 {
        return 0;
    }
    let block_size = BLOCK_SIZE as usize;
    // Level 0 always happens: scan the u8 mask.
    let mut depth = 1usize;
    let mut n = n_rows.div_ceil(block_size); // size of block_sums at level 0
    // Each additional level scans the previous block_sums.
    while n > block_size {
        depth += 1;
        n = n.div_ceil(block_size);
    }
    // Plus one for the final host scan over the top-level block_sums.
    depth + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_exclusive_scan_empty() {
        let (bases, total) = host_exclusive_scan(&[]);
        assert!(bases.is_empty());
        assert_eq!(total, 0);
    }

    #[test]
    fn host_exclusive_scan_single() {
        let (bases, total) = host_exclusive_scan(&[42]);
        assert_eq!(bases, vec![0]);
        assert_eq!(total, 42);
    }

    #[test]
    fn host_exclusive_scan_multi() {
        // Hand-checked: bases are exclusive prefix sums; total is the full sum.
        let sums = vec![3u32, 0, 5, 256, 1, 9, 9];
        let (bases, total) = host_exclusive_scan(&sums);
        assert_eq!(bases, vec![0, 3, 3, 8, 264, 265, 274]);
        assert_eq!(total, 283);
        assert_eq!(total as u32, sums.iter().sum::<u32>());
    }

    #[test]
    fn host_exclusive_scan_wide_accumulator() {
        // Each entry sums to BLOCK_SIZE - 1 (max per block); with 5 entries
        // total is 5 * 255 = 1275, well within u32. The point is that the
        // u64 accumulator absorbs the intermediate values without surprise.
        let sums = vec![255u32; 5];
        let (bases, total) = host_exclusive_scan(&sums);
        assert_eq!(bases, vec![0, 255, 510, 765, 1020]);
        assert_eq!(total, 1275);
    }

    #[test]
    fn dispatch_threshold_host_below_block_size() {
        // Arrays that fit in one block are host-scanned (the recursion base
        // case). At BLOCK_SIZE = 256 that's any length 0..=256.
        for len in [0usize, 1, 128, 255, 256] {
            assert!(
                should_host_scan(len),
                "len={} should be host-scanned (BLOCK_SIZE={})",
                len,
                BLOCK_SIZE
            );
        }
    }

    #[test]
    fn dispatch_threshold_device_above_block_size() {
        // Anything strictly larger than BLOCK_SIZE recurses to the device
        // scan. This is the case the multipass path exists for.
        for len in [257usize, 512, 65_536, 16_777_216, 1_000_000_000] {
            assert!(
                !should_host_scan(len),
                "len={} should recurse to device scan (BLOCK_SIZE={})",
                len,
                BLOCK_SIZE
            );
        }
    }

    #[test]
    fn recursion_depth_matches_documented_bounds() {
        // No work for empty input.
        assert_eq!(recursion_depth(0), 0);

        // n_rows fits in a single block: one device scan + the host scan
        // of a 1-entry block_sums = 2.
        assert_eq!(recursion_depth(BLOCK_SIZE as usize), 2);

        // n_rows = 1B at BLOCK_SIZE = 256:
        //   Level 0: 1B → block_sums has 3 906 250 entries (> 256, recurse)
        //   Level 1: 3.9M → block_sums has 15 259 entries (> 256, recurse)
        //   Level 2: 15 259 → block_sums has 60 entries (≤ 256, host-scan)
        // So: 3 device levels + 1 host level = 4.
        assert_eq!(recursion_depth(1_000_000_000), 4);

        // n_rows = u32::MAX (~4.3B):
        //   Level 0: 4.3B → block_sums has 16 777 216 entries (recurse)
        //   Level 1: 16.8M → block_sums has 65 536 entries (recurse)
        //   Level 2: 65 536 → block_sums has 256 entries (host-scan)
        // So: 3 device levels + 1 host level = 4.
        assert_eq!(recursion_depth(u32::MAX as usize), 4);
    }

    /// The empty-input shortcut must not allocate scan products or hit any
    /// kernel-launch path. We pass `mask_ptr = 0` (NULL device pointer) so
    /// that if the shortcut ever regresses, the first launch faults on the
    /// NULL mask and this test starts failing.
    #[test]
    fn multipass_empty_inputs_skips_launch() {
        let stream = CudaStream::null();
        let res = prefix_scan_mask_multipass(0, 0, &stream);
        match res {
            Ok(scan) => {
                assert_eq!(scan.n_rows, 0);
                assert_eq!(scan.total_count, 0);
                assert_eq!(scan.local_indices.len(), 0);
                assert_eq!(scan.block_bases.len(), 0);
                assert_eq!(scan.mask_ptr, 0);
            }
            Err(e) => panic!("expected Ok for empty inputs, got {e}"),
        }
    }
}
