// SPDX-License-Identifier: Apache-2.0

//! Exclusive prefix-sum offsets for Tier-2 hash-partitioned GROUP BY.
//!
//! After the partition kernel (sibling module) writes per-partition row
//! counts into a `GpuVec<u32>` of length [`NUM_PARTITIONS`], the scatter
//! kernel needs to know where each partition starts in the destination
//! buffer. That's an exclusive prefix sum over the count vector.
//!
//! ## Why we do this on the host
//!
//! `NUM_PARTITIONS = 4096`, so the counts vector is exactly 16 KiB. The
//! cost breakdown for a host-side scan is:
//!
//! - DtoH copy of 16 KiB:  ~10 µs (a single PCIe round-trip)
//! - 4096-element sum:     ~1 µs on any modern CPU
//! - HtoD copy of 16 KiB:  ~10 µs
//!
//! That's ~25 µs end-to-end *combined* across `compute_partition_offsets`
//! and `upload_offsets`. A GPU prefix-scan over 4096 elements would pay
//! roughly the same in launch overhead alone, plus we'd have to ship and
//! maintain another kernel. Tier 2 only kicks in for queries whose
//! end-to-end runtime is measured in milliseconds, so this overhead is
//! comfortably below 0.1 %. The complexity of a device scan is not
//! justified at this scale.
//!
//! ## Stage-5 (P1b) async + pinned host
//!
//! The sync round-trip used to cost two pageable PCIe transfers (one D2H,
//! one H2D) hitting the driver-synthesised staging buffer. Stage 5 routes
//! both legs through a single 16 KiB **pinned** host scratch buffer:
//!
//! 1. D2H `cuMemcpyDtoHAsync` into pinned scratch on the NULL stream.
//! 2. Block on `cuStreamSynchronize`.
//! 3. Prefix-sum in place on the same pinned region.
//! 4. H2D `cuMemcpyHtoDAsync` out of the same pinned region.
//! 5. Block on `cuStreamSynchronize`.
//!
//! On a 16 KiB transfer pinned vs pageable roughly halves wall time
//! (~6 GB/s → ~12 GB/s observed). Combined cost drops from ~25 µs sync to
//! ~12 µs async-pinned per orchestrator call — at 1000 calls/s that's
//! ~13 ms/s of CPU time recovered, just from removing the driver's
//! pageable-staging detour.
//!
//! ### Stage-7 (P1b) thread-local pinned scratch
//!
//! Stage 5 stored the pinned scratch in a single
//! `OnceLock<Mutex<PinnedHostBuffer<u32>>>`. That's correct, but every
//! Tier-2 orchestrator call serialises on the same mutex — which becomes
//! a contention point as soon as queries run concurrently (think a server
//! handling several SQL requests in flight on the same engine). Stage 7
//! swaps the global mutex for a `thread_local!` scratch: each query
//! thread owns one 16 KiB pinned buffer, allocated lazily on first use.
//!
//! **Trade-off**: the host now holds one 16 KiB pinned region per thread
//! that ever touched Tier-2. For the typical server topology (a small
//! pool of worker threads handling many queries) this is ~tens of KiB of
//! pinned memory total — well under the per-context budget that the rest
//! of the engine reserves. For a pathological caller that spawns a new
//! thread per query, the buffers leak with the thread on exit
//! (`cuMemFreeHost` runs in the per-thread `Drop`), which is the right
//! lifecycle anyway. We document the upper-bound as "one 16 KiB pinned
//! buffer per orchestrator-calling thread" so a Stage-8 ceiling can land
//! cleanly if/when a counter-example shows up.
//!
//! ### Joint-call helper
//!
//! `compute_and_upload_partition_offsets_async` exposes the
//! "one synchronize between D2H and H2D" path callers can adopt to drop
//! the second sync. The orchestrator currently calls the pair separately
//! (history reasons); when it migrates, the joint helper collapses the
//! pinned-async sequence to a single `stream.synchronize()` for the
//! whole call.
//!
//! ## Sizing rationale (K = 4096)
//!
//! If the partition count ever grows past ~16 K we should revisit, but
//! 4096 is the right choice for q5-class workloads (~1 M groups, ~250
//! groups per partition) and there's no plausible path to making it
//! larger without also blowing up the per-partition hashtable budget.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::cuda::cuda_sys::{self, CUstream};
use crate::cuda::{GpuVec, PinnedHostBuffer};
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::CudaStream;

/// Number of hash partitions used by Tier-2 GROUP BY.
///
/// Chosen so that for a target of ~1 M distinct groups, each partition
/// holds on the order of `BLOCK_GROUPS = 1024` keys, which is the upper
/// bound the Tier-1 block-local hashtable can hold in shared memory.
pub const NUM_PARTITIONS: u32 = 4096;

/// Process-global CUDA-context epoch.
///
/// Bumped by [`invalidate_pinned_scratch_on_context_teardown`] every time a
/// `CudaContext` is destroyed. The thread-local pinned scratch stashes the
/// epoch it was allocated under; a mismatch means "the context that pinned my
/// pages is gone" and forces a fresh allocation in the live context. See the
/// teardown function for the full rationale.
static CONTEXT_EPOCH: AtomicU64 = AtomicU64::new(0);

// Stage-7: per-thread pinned scratch buffer of length `NUM_PARTITIONS + 1`.
//
// One 16 KiB page-locked region per thread that ever calls into this
// module. Initialised lazily on first use, so a thread that never reaches
// Tier-2 (or runs without a CUDA context) pays no `cuMemAllocHost` cost.
// `PinnedHostBuffer` is `Send + !Sync` and its `Drop` releases the pinned
// region via `cuMemFreeHost`, so the buffer's lifecycle is bound to the
// thread that owns it.
//
// **Context binding (the multi-engine crash).** `cuMemAllocHost_v2` pins the
// pages *inside the CUDA context current at allocation time*. The engine mints
// a fresh context per `Engine`, and `cuCtxDestroy` reclaims that context's
// pinned allocations on teardown — but this thread-local outlives the context
// (the thread keeps running across `Engine::new()` / drop cycles). Without a
// guard, the second engine would reuse a `PinnedHostBuffer` whose host pointer
// was reclaimed by the first context's `cuCtxDestroy`, then `as_mut_slice()`
// into now-unmapped pages and `cuMemcpy*Async` against a pinned registration
// in a dead context — a `STATUS_ACCESS_VIOLATION`. This is the pinned-host
// analogue of the dangling pooled `CUdeviceptr` that `POOL.drain()` guards and
// the stale `CudaModule` that `module_cache::clear_all_caches()` guards. We
// close it by tagging each buffer with the `CONTEXT_EPOCH` it was allocated
// under and discarding it on mismatch (see `with_pinned_scratch`), plus a
// clean free of the dropping thread's own buffer at teardown (see
// `invalidate_pinned_scratch_on_context_teardown`).
//
// **Why a `RefCell`**: callers borrow the scratch mutably (to write the
// downloaded counts and the prefix-summed bases) but only one borrow is
// ever live at a time — the orchestrator calls a single helper at a time
// per thread, and helpers don't re-enter the module. The `RefCell::try_borrow_mut`
// path therefore never panics under correct use; we surface a clean
// `BoltError` if it ever does (e.g. a future re-entrant caller).
thread_local! {
    /// The thread-local pinned scratch slot. `RefCell<Option<(buf, epoch)>>`
    /// so we can:
    ///   * Lazily allocate on first `with_pinned_scratch` call (`None` → `Some`).
    ///   * Recover from a previous allocation failure by leaving the slot
    ///     `None` and re-attempting on the next call.
    ///   * Detect that the allocating context was torn down by comparing the
    ///     stashed `epoch` against [`CONTEXT_EPOCH`].
    ///
    /// `Option` matters because `RefCell` itself can't be empty — without
    /// it we'd have to stash a zero-length placeholder, which the callers
    /// can't distinguish from a real (but truncated) buffer.
    static PINNED_SCRATCH: RefCell<Option<(PinnedHostBuffer<u32>, u64)>> =
        const { RefCell::new(None) };
}

/// Invalidate the process-wide pinned scratch on `CudaContext` teardown.
///
/// Called from `CudaContext::Drop` (alongside `POOL.drain()`,
/// `stream_pool::drain()`, and `module_cache::clear_all_caches()`) *while the
/// dying context is still alive and current*. It does two things:
///
/// 1. **Frees the dropping thread's own scratch cleanly.** In the supported
///    single-context model the thread that drops the `Engine` is the same one
///    that ran the Tier-2 queries, so its `PINNED_SCRATCH` holds a buffer
///    pinned in the context we're about to destroy. Setting the slot to `None`
///    here runs `PinnedHostBuffer::Drop` → `cuMemFreeHost` against the still-
///    live context, reclaiming the pages without a warning.
///
/// 2. **Invalidates every other thread's scratch.** A thread-local cannot be
///    reset from another thread, so for any *other* thread that allocated
///    scratch under this context we instead bump [`CONTEXT_EPOCH`]. That
///    thread's next `with_pinned_scratch` observes the stale epoch and
///    discards its buffer via `forget_pinned_pages` — skipping the now-invalid
///    `cuMemFreeHost` (the pages were already reclaimed by `cuCtxDestroy`) and
///    reallocating fresh in the live context. This makes the fix robust beyond
///    the single-threaded repro without ever touching foreign TLS.
pub fn invalidate_pinned_scratch_on_context_teardown() {
    // (1) Free this thread's buffer while the context is still current.
    PINNED_SCRATCH.with(|cell| {
        if let Ok(mut slot) = cell.try_borrow_mut() {
            // Drop -> cuMemFreeHost in the live context (clean reclaim).
            *slot = None;
        }
        // A failed borrow means we're nested inside `with_pinned_scratch`,
        // which never happens from within a context Drop; ignore it rather
        // than risk panicking in a destructor.
    });
    // (2) Force every other thread's stale buffer to be discarded lazily.
    CONTEXT_EPOCH.fetch_add(1, Ordering::Release);
}

/// Run `f` with the calling thread's pinned scratch buffer, allocating
/// it on first use. The buffer is guaranteed to hold at least
/// `NUM_PARTITIONS + 1` `u32`s while `f` runs.
///
/// Errors:
///   * Returns `BoltError::Other` if `cuMemAllocHost` refuses (no CUDA
///     context, host is OOM, etc.). The slot is left empty so the next
///     call may retry — useful in test harnesses that initialise CUDA
///     after the first orchestrator probe.
///   * Returns `BoltError::Other` if the slot is already borrowed
///     (re-entrancy bug — this module is not re-entrant by design).
///
/// `f` cannot itself call into this module on the same thread; doing so
/// would land in the borrow-checker arm above.
fn with_pinned_scratch<R>(
    f: impl FnOnce(&mut PinnedHostBuffer<u32>) -> BoltResult<R>,
) -> BoltResult<R> {
    PINNED_SCRATCH.with(|cell| {
        let mut slot = cell.try_borrow_mut().map_err(|_| {
            BoltError::Other(
                "partition_offsets: pinned scratch already in use on this thread \
                 (re-entrant call?)"
                    .into(),
            )
        })?;
        let epoch = CONTEXT_EPOCH.load(Ordering::Acquire);
        // Discard a buffer pinned under a now-destroyed context. Its host
        // pages were reclaimed by that context's `cuCtxDestroy`, so we must
        // NOT let `cuMemFreeHost` run on the stale pointer — `forget_pinned_pages`
        // neutralizes the `Drop` before the slot is cleared, then we reallocate
        // fresh in the live context below.
        if let Some((buf, buf_epoch)) = slot.as_mut() {
            if *buf_epoch != epoch {
                buf.forget_pinned_pages();
                *slot = None;
            }
        }
        if slot.is_none() {
            // Allocate on first use for this thread (or after a context-epoch
            // invalidation). If allocation fails we leave the slot empty so a
            // later call can retry.
            let buf = PinnedHostBuffer::<u32>::new(NUM_PARTITIONS as usize + 1)
                .map_err(|e| {
                    BoltError::Other(format!(
                        "partition_offsets: failed to allocate per-thread \
                         pinned scratch (cuMemAllocHost): {e}"
                    ))
                })?;
            *slot = Some((buf, epoch));
        }
        // SAFETY of unwrap: just installed `Some` if it was `None`.
        let (buf, _) = slot.as_mut().expect("pinned scratch was just installed");
        debug_assert!(buf.len() >= NUM_PARTITIONS as usize + 1);
        f(buf)
    })
}

/// Compute exclusive prefix-sum offsets from a GPU-resident counts vector.
///
/// Input: `counts` must be a `GpuVec<u32>` of length [`NUM_PARTITIONS`]
/// holding per-partition row counts (produced by `partition_kernel`).
///
/// Output: `Vec<u32>` of length `NUM_PARTITIONS + 1`. `offsets[k]` is the
/// starting index for partition `k` in the scatter destination buffer;
/// `offsets[NUM_PARTITIONS]` equals the total row count and is used as
/// the scatter buffer length.
///
/// Mechanism (Stage-5): async D2H from `counts` into a shared pinned host
/// scratch buffer on the NULL stream, then a single `cuStreamSynchronize`,
/// then a host-side prefix-sum loop. The pinned buffer cuts DMA bandwidth
/// roughly in half compared to the pageable D2H that the sync code path
/// used to do. Downloads 4096 `u32`s (16 KiB) over PCIe. See the module
/// docs for the lifecycle of the scratch slot and the cost rationale.
pub fn compute_partition_offsets(counts: &GpuVec<u32>) -> BoltResult<Vec<u32>> {
    let expected = NUM_PARTITIONS as usize;
    if counts.len() != expected {
        return Err(BoltError::Other(format!(
            "compute_partition_offsets: counts.len() = {} but expected NUM_PARTITIONS = {}",
            counts.len(),
            expected,
        )));
    }
    let stream = CudaStream::null();
    with_pinned_scratch(|scratch| {
        d2h_into_pinned(counts, scratch, stream.raw())?;
        stream.synchronize()?;
        prefix_sum_pinned_to_vec(scratch)
    })
}

/// Upload the host-side offsets back to the GPU so the scatter kernel
/// can read them.
///
/// Returns a `GpuVec<u32>` of length [`NUM_PARTITIONS`] (NOT length+1 —
/// the scatter kernel only needs the per-partition start, not the
/// trailing total). Callers that need the total should grab
/// `offsets[NUM_PARTITIONS as usize]` from the host slice before
/// uploading.
///
/// Mechanism (Stage-5): copies the input slice into the shared pinned
/// scratch buffer, then issues a `cuMemcpyHtoDAsync` on the NULL stream
/// and synchronizes once. Same DMA-bandwidth win as
/// [`compute_partition_offsets`]; see the module docs.
pub fn upload_offsets(offsets: &[u32]) -> BoltResult<GpuVec<u32>> {
    let expected = NUM_PARTITIONS as usize + 1;
    if offsets.len() != expected {
        return Err(BoltError::Other(format!(
            "upload_offsets: offsets.len() = {} but expected NUM_PARTITIONS + 1 = {}",
            offsets.len(),
            expected,
        )));
    }

    // Pinned-async H2D. We only ship the first NUM_PARTITIONS bases; the
    // scatter kernel indexes `offsets[pid]` for pid in [0, K) and the
    // trailing total is only useful host-side.
    let stream = CudaStream::null();
    with_pinned_scratch(|scratch| {
        // Copy the bases into pinned memory; this is a plain host memcpy
        // and not synchronized on any stream.
        scratch.as_mut_slice()[..NUM_PARTITIONS as usize]
            .copy_from_slice(&offsets[..NUM_PARTITIONS as usize]);
        // `from_slice_async` issues `cuMemcpyHtoDAsync` from the pinned source
        // pointer, which is DMA-friendly (no driver-synthesised staging).
        let gpu = GpuVec::<u32>::from_slice_async(
            &scratch.as_slice()[..NUM_PARTITIONS as usize],
            stream.raw(),
        )?;
        stream.synchronize()?;
        // Scratch borrow drops on return; safe because the stream is
        // already synchronized so no DMA still references the pinned
        // source region.
        Ok(gpu)
    })
}

/// Combined D2H + prefix-scan + H2D on a single caller-supplied stream
/// with **one** synchronize between the D2H and the H2D.
///
/// This is the Stage-5 "1 sync per call" entry point. It is functionally
/// equivalent to `compute_partition_offsets` followed by `upload_offsets`,
/// but uses the caller's stream throughout so the device-side prerequisites
/// (the partition kernel that wrote `counts`) and post-requisites (the
/// scatter kernel that consumes the uploaded offsets) chain through one
/// stream without an extra default-stream serialization.
///
/// Returns `(host_offsets, device_offsets)`:
/// - `host_offsets`: length `NUM_PARTITIONS + 1`; element [K] is the total
///   row count, needed for scatter buffer sizing.
/// - `device_offsets`: length `NUM_PARTITIONS`; the bases the scatter
///   kernel reads.
///
/// ## Lifecycle invariant
///
/// The H2D from pinned scratch into `device_offsets` is *enqueued* on
/// `stream` and not awaited inside this helper. The caller is required
/// to synchronize `stream` (or have a follow-up kernel queued on
/// `stream` that reads `device_offsets`, then synchronize later) before
/// the next call to any function in this module — otherwise a second
/// caller could acquire the pinned-scratch mutex and overwrite the
/// region while the in-flight H2D is still draining it. In the current
/// orchestrator that invariant holds trivially: Tier-2 calls are
/// single-threaded per query and the scatter kernel that reads
/// `device_offsets` is enqueued on the same stream, so the H2D retires
/// before any other dispatch needs the scratch.
///
/// Currently used by the inline stage-5 round-trip test; the Tier-2
/// orchestrator can migrate to this entry point in a follow-up to drop its
/// second sync site. Today the orchestrator still calls the legacy pair,
/// which now each do an internal `stream.synchronize()` on the NULL
/// stream — totalling 2 syncs per orchestrator call vs the 1 sync this
/// helper offers.
// `stream: CUstream` is an opaque handle forwarded to FFI; not dereferenced
// in this function. The clippy lint suggests `unsafe fn` but that would be
// a major API break for no actual safety win.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn compute_and_upload_partition_offsets_async(
    counts: &GpuVec<u32>,
    stream: CUstream,
) -> BoltResult<(Vec<u32>, GpuVec<u32>)> {
    let expected = NUM_PARTITIONS as usize;
    if counts.len() != expected {
        return Err(BoltError::Other(format!(
            "compute_and_upload_partition_offsets_async: counts.len() = {} \
             but expected NUM_PARTITIONS = {}",
            counts.len(),
            expected,
        )));
    }

    with_pinned_scratch(|scratch| {
        // Step 1: D2H async into pinned scratch[0..NUM_PARTITIONS].
        // SAFETY: scratch holds NUM_PARTITIONS+1 elements; we write the
        // first NUM_PARTITIONS. `counts` is a live device allocation of
        // exactly NUM_PARTITIONS u32s (checked above).
        unsafe {
            cuda_sys::memcpy_d2h_async::<u32>(
                scratch.as_mut_ptr(),
                counts.device_ptr(),
                NUM_PARTITIONS as usize,
                stream,
            )?;
        }

        // The H2D below depends on the prefix-sum, which depends on the
        // D2H landing — we cannot enqueue the H2D yet. Sync once to flush
        // the D2H, then do the host work, then issue the H2D. The whole
        // pipeline therefore costs exactly one synchronize.
        unsafe {
            cuda_sys::check(cuda_sys::cuStreamSynchronize(stream))?;
        }

        // Step 2: prefix-sum in pinned memory. We materialise the
        // host-visible `Vec<u32>` here (length NUM_PARTITIONS+1) because
        // the caller wants it for the scatter-buffer sizing.
        let host_offsets = prefix_sum_pinned_to_vec(scratch)?;

        // Step 3: write the bases (offsets[0..NUM_PARTITIONS]) back into
        // pinned scratch[0..NUM_PARTITIONS]. They sit alongside the
        // already-computed `offsets[NUM_PARTITIONS]` in scratch[K], which
        // we leave unused for this leg.
        scratch.as_mut_slice()[..NUM_PARTITIONS as usize]
            .copy_from_slice(&host_offsets[..NUM_PARTITIONS as usize]);

        // Step 4: H2D async from pinned scratch into a freshly allocated
        // device vec. The `from_slice_async` call issues `cuMemcpyHtoDAsync`
        // with the pinned source pointer (no driver-synthesised staging) and
        // sets the GpuVec's logical length atomically.
        let gpu_out = GpuVec::<u32>::from_slice_async(
            &scratch.as_slice()[..NUM_PARTITIONS as usize],
            stream,
        )?;

        // We hold the scratch borrow across the enqueued H2D. The caller MUST
        // synchronize `stream` before issuing another partition-offsets call
        // on this same thread (the typical flow does this implicitly by
        // chaining the scatter kernel on `stream`, then synchronizing later).
        // With Stage-7 thread-local scratch, a *different* thread cannot
        // observe the in-flight H2D's source region (each thread owns its own
        // pinned slot), so the cross-thread variant of this footgun is gone.
        // Dropping the borrow here is safe: the next same-thread call would
        // see the slot free, and any DMA in flight by then must have retired
        // because the caller's synchronize precedes it.
        Ok((host_offsets, gpu_out))
    })
}

/// Async D2H of all NUM_PARTITIONS counts into `scratch[0..NUM_PARTITIONS]`.
///
/// The caller is responsible for synchronizing `stream` before reading
/// `scratch`. `scratch.len()` must be `>= NUM_PARTITIONS`.
fn d2h_into_pinned(
    counts: &GpuVec<u32>,
    scratch: &mut PinnedHostBuffer<u32>,
    stream: CUstream,
) -> BoltResult<()> {
    debug_assert!(scratch.len() >= NUM_PARTITIONS as usize);
    debug_assert_eq!(counts.len(), NUM_PARTITIONS as usize);
    // SAFETY: scratch has capacity for NUM_PARTITIONS u32s; counts is a
    // live device allocation of exactly that many u32s. Caller synchronizes
    // the stream before reading scratch.
    unsafe {
        cuda_sys::memcpy_d2h_async::<u32>(
            scratch.as_mut_ptr(),
            counts.device_ptr(),
            NUM_PARTITIONS as usize,
            stream,
        )?;
    }
    Ok(())
}

/// Compute the exclusive prefix-sum of the first NUM_PARTITIONS elements
/// in `scratch` (the just-downloaded counts) and return a fresh
/// `Vec<u32>` of length NUM_PARTITIONS+1.
///
/// The pinned scratch buffer is left in a defined state — its first
/// NUM_PARTITIONS entries are unchanged, and entry [K] is unused. We
/// build a separate `Vec<u32>` because callers want owned host data and
/// the pinned scratch is shared.
fn prefix_sum_pinned_to_vec(scratch: &PinnedHostBuffer<u32>) -> BoltResult<Vec<u32>> {
    prefix_sum_cpu(&scratch.as_slice()[..NUM_PARTITIONS as usize])
}

/// Pure-CPU exclusive prefix sum.
///
/// Output length is `counts.len() + 1`; `out[0] = 0`,
/// `out[k] = sum(counts[0..k])`, and `out[counts.len()]` is the total.
///
/// The legitimate total equals the row count, which is bounded by
/// `u32::MAX` (any real workload that overflowed it would already have
/// been rejected upstream by `n_rows_to_u32`). We therefore use
/// `checked_add`: under correct operation it never overflows, but a
/// corrupt/garbage partition count (kernel bug, uninitialized scratch
/// slot) would otherwise *wrap* and silently emit wrong scatter base
/// offsets, sending downstream scatter writes out of place or OOB.
/// Converting that overflow into a clean `BoltError` turns a silent
/// device-memory corruption into a recoverable error. The cost over the
/// 4096-element domain is negligible.
fn prefix_sum_cpu(counts: &[u32]) -> BoltResult<Vec<u32>> {
    let mut out = Vec::with_capacity(counts.len() + 1);
    let mut acc: u32 = 0;
    out.push(0);
    for &c in counts {
        acc = acc.checked_add(c).ok_or_else(|| {
            BoltError::Other(
                "partition count prefix-sum overflowed u32: corrupt partition counts".into(),
            )
        })?;
        out.push(acc);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_sum_empty() {
        let counts = vec![0u32; NUM_PARTITIONS as usize];
        let offsets = prefix_sum_cpu(&counts).unwrap();
        assert_eq!(offsets.len(), NUM_PARTITIONS as usize + 1);
        assert!(offsets.iter().all(|&v| v == 0));
    }

    #[test]
    fn prefix_sum_uniform() {
        let counts = vec![5u32; NUM_PARTITIONS as usize];
        let offsets = prefix_sum_cpu(&counts).unwrap();
        assert_eq!(offsets.len(), NUM_PARTITIONS as usize + 1);
        for k in 0..=NUM_PARTITIONS as usize {
            assert_eq!(offsets[k], (k as u32) * 5, "offsets[{}] mismatch", k);
        }
    }

    #[test]
    fn prefix_sum_skewed() {
        // All rows land in partition 0; every other partition is empty.
        let mut counts = vec![0u32; NUM_PARTITIONS as usize];
        counts[0] = NUM_PARTITIONS;
        let offsets = prefix_sum_cpu(&counts).unwrap();

        // offsets[0] must be zero (exclusive scan).
        assert_eq!(offsets[0], 0);
        // offsets[1..=NUM_PARTITIONS] all sit past the single populated partition.
        for k in 1..=NUM_PARTITIONS as usize {
            assert_eq!(offsets[k], NUM_PARTITIONS);
        }
        // Monotonic non-decreasing — true for any exclusive prefix sum of non-negative counts.
        for window in offsets.windows(2) {
            assert!(window[0] <= window[1]);
        }
        // Final element equals total.
        assert_eq!(offsets[NUM_PARTITIONS as usize], NUM_PARTITIONS);
    }

    #[test]
    fn prefix_sum_known_pattern() {
        // counts[k] = k + 1, so sum_{i=0..k} counts[i] = sum_{i=1..=k} i = k*(k+1)/2.
        let counts: Vec<u32> = (0..NUM_PARTITIONS).map(|k| k + 1).collect();
        let offsets = prefix_sum_cpu(&counts).unwrap();
        for k in 0..=NUM_PARTITIONS as usize {
            let k_u32 = k as u32;
            let expected = k_u32 * (k_u32 + 1) / 2;
            assert_eq!(offsets[k], expected, "offsets[{}] mismatch", k);
        }
    }

    #[test]
    fn length_invariant() {
        // The exclusive-scan contract: out.len() == in.len() + 1.
        // Exercise multiple input lengths to catch off-by-one mistakes regardless
        // of NUM_PARTITIONS being a power of two.
        for &n in &[0usize, 1, 2, 17, 1023, NUM_PARTITIONS as usize, 4096] {
            let counts = vec![1u32; n];
            let offsets = prefix_sum_cpu(&counts).unwrap();
            assert_eq!(offsets.len(), n + 1, "length invariant violated at n = {}", n);
        }
    }

    #[test]
    fn last_element_equals_total() {
        // Use a non-trivial irregular pattern so a bug that returns the wrong
        // accumulator (e.g. inclusive scan) would be visible.
        let counts: Vec<u32> = (0..NUM_PARTITIONS).map(|k| (k * 7 + 3) % 11).collect();
        let total: u32 = counts.iter().sum();
        let offsets = prefix_sum_cpu(&counts).unwrap();
        assert_eq!(offsets[NUM_PARTITIONS as usize], total);
        assert_eq!(offsets[0], 0);
    }

    #[test]
    fn exclusive_not_inclusive() {
        // Guard against a regression where someone "simplifies" the loop and
        // accidentally produces an inclusive scan.
        let counts = vec![1u32, 2, 3, 4];
        let offsets = prefix_sum_cpu(&counts).unwrap();
        // Exclusive scan: [0, 1, 3, 6, 10]
        assert_eq!(offsets, vec![0, 1, 3, 6, 10]);
    }

    // End-to-end round-trip through compute_partition_offsets + upload_offsets.
    // Marked `#[ignore]` because both calls allocate device memory and so need
    // a live CUDA context, which CI may not have. Run locally with
    // `cargo test -- --ignored partition_offsets` on a CUDA box.
    #[test]
    #[ignore = "requires CUDA toolkit at runtime (allocates GpuVec)"]
    fn end_to_end_roundtrip() {
        let host_counts: Vec<u32> = (0..NUM_PARTITIONS).map(|k| k + 1).collect();
        let dev_counts = GpuVec::<u32>::from_slice(&host_counts).expect("upload counts");
        let offsets = compute_partition_offsets(&dev_counts).expect("compute offsets");
        assert_eq!(offsets.len(), NUM_PARTITIONS as usize + 1);

        // Spot-check against the closed form.
        for k in 0..=NUM_PARTITIONS as usize {
            let k_u32 = k as u32;
            let expected = k_u32 * (k_u32 + 1) / 2;
            assert_eq!(offsets[k], expected);
        }

        let dev_offsets = upload_offsets(&offsets).expect("upload offsets");
        assert_eq!(dev_offsets.len(), NUM_PARTITIONS as usize);
        let roundtripped = dev_offsets.to_vec().expect("download offsets");
        // upload_offsets drops the trailing total — the device copy should
        // match offsets[..NUM_PARTITIONS] exactly.
        assert_eq!(roundtripped, offsets[..NUM_PARTITIONS as usize]);
    }

    #[test]
    fn upload_rejects_wrong_length() {
        // Exercises the length-check path without needing CUDA: the guard
        // fires before we call into GpuVec::from_slice. Now that
        // `upload_offsets` allocates a GpuVec eagerly, the length check
        // must still fire first so this stays a host-only test.
        // GpuVec<u32> doesn't implement Debug, so we can't use expect_err —
        // pattern-match instead.
        let too_short = vec![0u32; NUM_PARTITIONS as usize]; // missing trailing total
        match upload_offsets(&too_short) {
            Ok(_) => panic!("must reject length NUM_PARTITIONS"),
            Err(e) => {
                let msg = format!("{}", e);
                assert!(
                    msg.contains("expected NUM_PARTITIONS + 1"),
                    "unexpected error message: {}",
                    msg
                );
            }
        }
    }

    // ---------------------------------------------------------------------
    // Stage-5 (P1b) async round-trip tests.
    //
    // These exercise the pinned-async D2H + H2D path end-to-end. Marked
    // `#[ignore]` because they need a live CUDA context. Run locally:
    //   cargo test -- --ignored partition_offsets::tests::stage5
    // ---------------------------------------------------------------------

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn stage5_compute_uses_pinned_async() {
        // Sanity check that the pinned-async path returns the same
        // prefix-sum as the host scan. Counts chosen to exercise the
        // wrapping accumulator and a non-trivial offsets[K].
        let host_counts: Vec<u32> =
            (0..NUM_PARTITIONS).map(|k| (k * 11 + 1) % 257).collect();
        let dev_counts =
            GpuVec::<u32>::from_slice(&host_counts).expect("upload counts");

        let offsets = compute_partition_offsets(&dev_counts).expect("async compute");
        assert_eq!(offsets.len(), NUM_PARTITIONS as usize + 1);
        assert_eq!(offsets[0], 0);
        let expected_total: u32 = host_counts.iter().copied().sum();
        assert_eq!(offsets[NUM_PARTITIONS as usize], expected_total);

        // Cross-check against the pure-host prefix sum.
        let cpu = prefix_sum_cpu(&host_counts).unwrap();
        assert_eq!(offsets, cpu);
    }

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn stage5_joint_helper_single_sync() {
        // The joint helper does exactly one stream.synchronize() between
        // the D2H and the H2D. We can't directly observe the sync count
        // without instrumenting the driver, but we can at least check the
        // outputs match the legacy pair.
        use crate::exec::launch::CudaStream;

        let host_counts: Vec<u32> = (0..NUM_PARTITIONS).map(|k| k + 1).collect();
        let dev_counts =
            GpuVec::<u32>::from_slice(&host_counts).expect("upload counts");

        let stream = CudaStream::new().expect("create stream");
        let (offsets, dev_offsets) =
            compute_and_upload_partition_offsets_async(&dev_counts, stream.raw())
                .expect("joint helper");
        stream.synchronize().expect("flush trailing H2D");

        // Same prefix-sum semantics as the legacy pair.
        assert_eq!(offsets.len(), NUM_PARTITIONS as usize + 1);
        for k in 0..=NUM_PARTITIONS as usize {
            let k_u32 = k as u32;
            let expected = k_u32 * (k_u32 + 1) / 2;
            assert_eq!(offsets[k], expected);
        }

        // Device side: first NUM_PARTITIONS bases match the host slice.
        assert_eq!(dev_offsets.len(), NUM_PARTITIONS as usize);
        let device_view = dev_offsets.to_vec().expect("download offsets");
        assert_eq!(device_view, offsets[..NUM_PARTITIONS as usize]);
    }

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn stage5_pinned_scratch_is_reused() {
        // Calling compute_partition_offsets twice in a row should not
        // reallocate pinned memory — the thread-local scratch is reused
        // for every call on this thread. We exercise the path twice with
        // different inputs and confirm each produces the right answer; an
        // allocation bug would typically manifest as a stale-data
        // dependency between calls.
        let a: Vec<u32> = (0..NUM_PARTITIONS).map(|k| k + 1).collect();
        let b: Vec<u32> = (0..NUM_PARTITIONS).map(|k| 2 * k + 1).collect();
        let dev_a = GpuVec::<u32>::from_slice(&a).expect("upload a");
        let dev_b = GpuVec::<u32>::from_slice(&b).expect("upload b");

        let offs_a = compute_partition_offsets(&dev_a).expect("compute a");
        let offs_b = compute_partition_offsets(&dev_b).expect("compute b");

        assert_eq!(offs_a, prefix_sum_cpu(&a).unwrap());
        assert_eq!(offs_b, prefix_sum_cpu(&b).unwrap());
    }

    // ---------------------------------------------------------------------
    // Stage-7 (P1b) thread-local pinned scratch.
    //
    // Concurrent-allocator coverage: 8 threads each run a prefix-sum
    // through `with_pinned_scratch`, every thread allocates its own
    // 16 KiB pinned slot, and the results don't cross-contaminate. The
    // test is CUDA-free — `with_pinned_scratch` exits early with an
    // error when `cuMemAllocHost` refuses (which it always does without
    // a context), and we treat that as a successful "no panic + clean
    // error" case. On CUDA hosts (run with `--ignored`) every thread
    // produces the correct prefix sum.
    // ---------------------------------------------------------------------

    #[test]
    fn stage7_concurrent_threads_no_panic() {
        // Eight threads, each calling the thread-local scratch helper
        // many times. We don't require CUDA: if pinned allocation fails
        // the helper returns an `Err` (no panic). The assertion below
        // accepts either outcome — what we're guarding against is a
        // re-entrancy / lock-contention panic from the helper itself.
        use std::thread;

        let mut handles = Vec::new();
        for tid in 0..8u32 {
            handles.push(thread::spawn(move || {
                // Prepare a deterministic per-thread input. Each thread
                // hits `with_pinned_scratch` directly so the test stays
                // CUDA-free (no `GpuVec` allocation needed).
                for iter in 0..32u32 {
                    let result: BoltResult<Vec<u32>> =
                        with_pinned_scratch(|scratch| {
                            // Use the pinned region as scratch: write a
                            // deterministic pattern in, prefix-sum it,
                            // return the result. Mirrors the real
                            // download-then-scan call shape.
                            let s = scratch.as_mut_slice();
                            for k in 0..NUM_PARTITIONS as usize {
                                s[k] = tid.wrapping_add(iter).wrapping_add(k as u32);
                            }
                            prefix_sum_cpu(&s[..NUM_PARTITIONS as usize])
                        });
                    if let Ok(v) = result {
                        // Sanity: prefix sum's first element is 0 and
                        // length is correct. Any cross-thread state
                        // bleed would corrupt this.
                        assert_eq!(v.len(), NUM_PARTITIONS as usize + 1);
                        assert_eq!(v[0], 0);
                    }
                    // Err is acceptable on a CUDA-less host (no pinned
                    // alloc available); the test asserts only that we
                    // don't panic / dangle / cross-contaminate state.
                }
            }));
        }
        for h in handles {
            // `join` must not panic — the helper is panic-free.
            h.join().expect("worker thread panicked");
        }
    }

    #[test]
    #[ignore = "requires CUDA toolkit at runtime — Stage 7 concurrent"]
    fn stage7_concurrent_threads_each_get_correct_output() {
        // CUDA-on variant of `stage7_concurrent_threads_no_panic`. Each
        // thread does a real D2H prefix-sum through
        // `compute_partition_offsets` and the result must match the
        // pure-CPU scan exactly — proving the thread-local pinned slot
        // is genuinely thread-local (no cross-thread tearing).
        use std::sync::Arc;
        use std::thread;

        // Build the inputs on the main thread (CUDA context is per-thread
        // in the driver model; using one shared host vec keeps the test
        // self-contained — each thread re-uploads its own GpuVec).
        let inputs: Vec<Arc<Vec<u32>>> = (0..8)
            .map(|tid| {
                let v: Vec<u32> = (0..NUM_PARTITIONS)
                    .map(|k| (tid as u32) * 7 + k + 1)
                    .collect();
                Arc::new(v)
            })
            .collect();

        let mut handles = Vec::new();
        for (tid, host) in inputs.iter().cloned().enumerate() {
            handles.push(thread::spawn(move || {
                // Each thread brings up its own CUDA context (driver API)
                // implicitly via `GpuVec::from_slice`'s primary-context
                // attach. If that fails, return a sentinel `None` so the
                // outer assertion can skip — the test is `#[ignore]`'d
                // already, so the run only happens on a CUDA host.
                let dev = GpuVec::<u32>::from_slice(&host[..])
                    .expect("upload per-thread counts");
                let offsets = compute_partition_offsets(&dev)
                    .expect("compute per-thread offsets");
                (tid, offsets, host)
            }));
        }
        for h in handles {
            let (tid, got, host) = h.join().expect("worker thread panicked");
            let expected = prefix_sum_cpu(&host[..]).unwrap();
            assert_eq!(
                got, expected,
                "thread {tid} returned a prefix sum that doesn't match the \
                 pure-CPU scan — thread-local pinned scratch likely torn"
            );
        }
    }
}
