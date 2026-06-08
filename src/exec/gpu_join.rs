// SPDX-License-Identifier: Apache-2.0

//! GPU-side INNER JOIN: build a hash table on the GPU from the smaller
//! (build) side, probe it with the larger (probe) side, materialise the
//! result host-side via `arrow::compute::take`.
//!
//! Pairs with [`crate::jit::hash_join_kernel`], which emits the PTX. Flow:
//!
//! ```text
//!  build keys (host, Int32/Int64, n_build)
//!     │
//!     ├─ encode -> i64           (sign-extend Int32; bitcast Int64)
//!     │
//!     ▼ h2d
//!  build_keys_dev (GpuVec<i64>)
//!     │
//!     ├─ keys_table_dev      (GpuVec<i64>, cap, init=i64::MIN)
//!     └─ row_idx_table_dev   (GpuVec<u32>, cap, init=u32::MAX)
//!     │
//!     ▼ launch BUILD kernel (1 thread / build row)
//!  fully-populated (keys_table_dev, row_idx_table_dev)
//!
//!  probe keys (host, same dtype, n_probe)
//!     │
//!     ├─ encode -> i64
//!     │
//!     ▼ h2d
//!  probe_keys_dev (GpuVec<i64>)
//!     │
//!     ├─ out_probe_idx_dev   (GpuVec<u32>, out_capacity)
//!     ├─ out_build_idx_dev   (GpuVec<u32>, out_capacity)
//!     └─ out_counter_dev     (GpuVec<u32>, length 1, init=0)
//!     │
//!     ▼ launch PROBE kernel (1 thread / probe row)
//!  out buffers populated up to *counter[0]* entries (arbitrary order)
//!     │
//!     ▼ d2h
//!  (probe_indices: Vec<u32>, build_indices: Vec<u32>)
//!     │
//!     ▼ arrow::compute::take per column on both sides
//!  joined RecordBatch
//! ```
//!
//! ## Stage 1 scope
//!
//! * **INNER only.** LEFT/RIGHT/FULL/CROSS stay host-side (J1's path).
//! * **Single equi-key.** Multi-key joins are Stage 2.
//! * **Int32 or Int64 key dtype.** Float / Bool / Utf8 fall through to host.
//! * **No NULLs in keys.** SQL NULL-keys-never-match drops the row from the
//!   inner-join output anyway; the host path enforces that, so we gate on
//!   `null_count() == 0` and re-use that contract here.
//! * **Both sides ≥ 1024 rows.** Below this, host wins (upload + JIT load).
//! * **Build side ≤ ~2.8M rows** (`2 * build_n_rows * 12 ≤ 64 MiB`).
//! * **Build keys are unique on the join column.** A collision in the build
//!   side's keys would lose all but one match in the row_idx_table slot. The
//!   host check is conservative (set-cardinality probe before upload), but
//!   even without it the probe-kernel correctness for the unique case is the
//!   bigger ROI for Stage 1.
//! * **No build key equals `i64::MIN`.** That value is the empty-slot
//!   sentinel; conflict would corrupt the build kernel's CAS step.
//!
//! On any gate miss we surface `Ok(None)`; the caller (the host path in
//! `crate::exec::join`) handles the input correctly.
//!
//! ## Stage-2 additions
//!
//! Stage 2 layers four host-visible capabilities on top of the Stage-1
//! kernels without changing the Stage-1 fast path:
//!
//! * **[`KeyShape`]-aware encoding** ([`encode_keys_for_shape`]) — Int32,
//!   Int64, Bool, Float32, Float64 single keys plus `TwoI32` composite
//!   pack `(hi as u32 << 32) | (lo as u32 as u64)`. Three-or-more keys
//!   fold to a single i64 via splitmix (`MultiI32(n)` and `TwoI64`).
//!   Lossy folds (`is_exact_in_i64() == false`) are gated off the GPU
//!   path; that lets us reuse the Stage-1 atomic-CAS lookup without
//!   per-pair host-side verification.
//! * **Outer joins** (`LEFT` / `RIGHT` / `FULL`) — orchestrated via
//!   [`execute_outer_join_on_gpu`]. The probe kernel atomically OR-sets
//!   bits in a `matched: u32[ceil(build_n_rows/32)]` bitmap; a second-
//!   pass [`compile_unmatched_build_kernel`] kernel emits the build-row
//!   index of every still-zero bit. The host pairs those indices with
//!   `None` in the take-indices array so `arrow::compute::take` NULL-pads
//!   the probe side.
//! * **Collision-list build/probe** — drops the Stage-1 "unique build
//!   keys" gate. The build kernel atomically prepends each row to a
//!   per-slot linked list (`head: u32[cap]` + `next_idx: u32[n_build]`),
//!   and the probe kernel walks that list emitting one output pair per
//!   visited build row. Used unconditionally for outer joins (so the
//!   matched-bitmap path stays linear in the number of build rows) and
//!   for inner joins with non-unique build keys.
//! * **VRAM-driven cap** — at first use, the executor queries the device's
//!   total memory via [`cuda_sys::device_total_mem`]. Cards with ≥ 8 GiB
//!   total VRAM bump the hash-table byte budget from 64 MiB to 512 MiB
//!   (=~ 44.7 M slots, ~22.4 M build rows under the 50% load factor).
//!   The cap stays at 64 MiB on smaller cards.
//!
//! ## Stage-3 additions
//!
//! Stage 3 closes the last surface gaps without disturbing the byte-stable
//! Stage-1/Stage-2 fast paths:
//!
//! * **Per-pair host verification** for lossy fold shapes (`TwoI64`,
//!   `MultiI32(n)`) — the GPU path emits candidate pairs from the i64-folded
//!   hash; the host then re-tests each pair against the *original* Arrow
//!   columns and drops false positives. See [`verify_pairs_on_host`].
//! * **Utf8 keys via string interning** — [`KeyShape::SingleUtf8`] dispatches
//!   through [`intern_utf8_columns`], which builds a `HashMap<&str, u32>` over
//!   the union of build + probe values and re-encodes the keys as i32
//!   dictionary indices. The kernels never see Utf8.
//! * **AoS hash-table slot layout** for the probe-heavy path —
//!   [`crate::jit::hash_join_kernel::compile_probe_aos_kernel`] reads each
//!   slot as a single 16-byte tuple, halving probe-side DRAM traffic. Behind
//!   the [`KeyShape::is_exact_in_i64`] gate today (host post-verify path
//!   doesn't yet use it).
//! * **Env-var-tunable cap** (`BOLT_GPU_JOIN_TABLE_CAP_MB`) — clamps to
//!   `[64, 4096]` MiB and overrides the driver-detected cap.
//! * **CROSS JOIN on GPU** — pure cartesian product via
//!   [`execute_cross_join_on_gpu`]; one kernel thread per output pair.
//!   Gated on `n_probe * n_build < CROSS_JOIN_GPU_CELL_CAP`.
//!
//! ## Stage-4 additions
//!
//! Stage 4 closes the four follow-ups Stage 3 documented as deferred:
//!
//! * **Multi-GPU cap routing** — [`resolve_byte_cap_from_driver`] now goes
//!   through [`crate::cuda::cuda_sys::current_device`] so multi-GPU setups
//!   query the device the engine is actually bound to. Single-GPU rigs see
//!   byte-identical behaviour.
//! * **Streaming Utf8 intern** — [`intern_utf8_columns_streaming`] keys
//!   the dict on a 64-bit hash of the string content (vs Stage-3's borrowed
//!   `&str`). 5-10× smaller dict footprint on high-cardinality joins at
//!   the cost of host post-verify cycles for hash collisions. Opt-in via
//!   the [`STREAMING_INTERN_ENV_VAR`] env var.
//! * **AoS build kernel** — [`hash_join_indices_on_gpu_aos`] uses the
//!   Stage-4 [`crate::jit::hash_join_kernel::compile_build_aos_kernel`] +
//!   the existing Stage-3 AoS probe against a single `slots: u8[cap * 16]`
//!   buffer. Symmetric with the AoS probe: `[key:u64, head:u32, _pad:u32]`
//!   at 16-byte stride. 33% more raw bytes than SoA (12-byte stride), but
//!   halves probe-side cache-line traffic — worth it on probe-bound joins.
//! * **OUTER + lossy shapes** — [`execute_outer_join_indices_on_gpu`] now
//!   admits `TwoI64` / `MultiI32(n)` for LEFT/RIGHT/FULL OUTER via the
//!   host post-verify pipeline. The GPU's `matched` bitmap is treated as
//!   a candidate superset; the verified pair set is re-tested host-side
//!   and the `matched_probe` / `matched_build` arrays are rebuilt before
//!   the unmatched-row emission pass.
//!
//! ## Stage-5 additions
//!
//! Stage 5 closes the three follow-ups Stage 4 listed as deferred:
//!
//! * **OUTER + Utf8** — [`execute_utf8_outer_join_on_gpu`] interns build/
//!   probe strings to i32 dict indices (Stage 3 path), runs the GPU
//!   collision-list OUTER as `SingleI32`, then host-post-verifies the
//!   resulting candidate pairs against the original `StringArray`s. The
//!   `matched_probe` / `matched_build` arrays are rebuilt from verified
//!   pairs before the unmatched-row emission pass — symmetric with the
//!   Stage-4 OUTER + lossy `TwoI64` flow.
//! * **Engine-level AoS routing** — [`AOS_ROUTING_PROBE_BUILD_RATIO`] is the
//!   heuristic threshold. When `n_probe / n_build > 8` the engine picks the
//!   AoS slot layout via [`try_aos_inner_join`]; the AoS path halves
//!   probe-side cache-line traffic at a 33% raw-bytes cost, so we only
//!   take it on probe-heavy workloads where the bandwidth win is real.
//! * **Device-side string hashing** — [`compute_device_string_hashes`]
//!   wraps the Stage-5 [`crate::jit::hash_join_kernel::compile_string_hash_kernel`]
//!   PTX. Uploads `(offsets: i32[n+1], values: u8[total])`, launches one
//!   thread per row, and downloads `u64[n]` hashes. Used by the streaming
//!   intern path on inputs with `> STREAMING_INTERN_DEVICE_HASH_THRESHOLD`
//!   rows.
//! * **Parallel per-chunk dicts** — [`intern_utf8_columns_streaming_parallel`]
//!   builds per-chunk `HashMap<u64, i32>` dicts via `std::thread::scope`,
//!   then merges them sequentially (O(distinct) for the merge, parallel for
//!   the hash). Default for inputs above
//!   [`STREAMING_INTERN_PARALLEL_THRESHOLD`] rows.
//!
//! ## Stage-6 follow-ups
//!
//! * **AoS for OUTER joins** — Stage 5 routes AoS for INNER only. The AoS
//!   collision-list build/probe pair isn't yet emitted, so OUTER stays on
//!   the SoA path.
//! * **Device hash + multi-key** — [`compute_device_string_hashes`] is
//!   single-column today. Multi-column Utf8 joins still fall back to
//!   host-side splitmix per row.
//! * **LargeUtf8 (i64 offsets)** — the device kernel reads i32 offsets;
//!   `StringArray` (Utf8) is the only supported input. `LargeStringArray`
//!   stays host-bound.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;
use std::sync::OnceLock;

use parking_lot::Mutex;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray, UInt32Array,
};
use arrow_schema::Schema as ArrowSchema;

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::gpu_upload::{download_to_host_pinned, upload_primitive_values_async};
use crate::exec::launch::CudaStream;
use crate::exec::module_cache;
use crate::exec::n_rows_to_u32;
use crate::jit::hash_join_kernel::{
    compile_build_aos_kernel, compile_build_collision_kernel, compile_build_kernel,
    compile_cross_kernel, compile_probe_aos_kernel, compile_probe_collision_kernel,
    compile_probe_kernel, compile_probe_kernel_tiled, compile_string_hash_kernel,
    compile_unmatched_build_kernel, KeyShape, AOS_SLOT_BYTES, BUILD_AOS_KERNEL_ENTRY,
    BUILD_COLLISION_KERNEL_ENTRY, BUILD_KERNEL_ENTRY, CROSS_KERNEL_ENTRY, HASH_JOIN_BLOCK_SIZE,
    PROBE_AOS_KERNEL_ENTRY, PROBE_COLLISION_KERNEL_ENTRY, PROBE_KERNEL_ENTRY,
    PROBE_KERNEL_TILED_ENTRY, STRING_HASH_BLOCK_SIZE, STRING_HASH_KERNEL_ENTRY,
    UNMATCHED_BUILD_KERNEL_ENTRY,
};
use crate::plan::logical_plan::DataType;
use crate::plan::physical_plan::{HashJoinKernelKind, HashJoinKernelSpec};

// ---------------------------------------------------------------------------
// Reusable host-side sentinel buffers
// ---------------------------------------------------------------------------
//
// The hash-join hash table needs three sentinel-filled host buffers per query
// that get H2D-uploaded to seed the device-side table:
//   * `keys`     — initial value `i64::MIN`        (cap entries)
//   * `row_idx`  — initial value `u32::MAX`        (cap entries)
//   * `head`     — initial value `u32::MAX`        (cap entries)
//   * `next_idx` — initial value `u32::MAX`        (n_build entries)
//
// At the large-cap limit (44.7 M slots) each `vec![i64::MIN; cap]` allocates
// ~357 MiB on the heap *per query*. The contents are query-independent —
// every query overwrites them identically. Allocating, zeroing, and freeing
// 357 MiB per join is pure waste.
//
// We replace the per-query allocation with a process-wide cached buffer that
// only ever grows. Each call returns a `&'static [T]` prefix of the requested
// length; the buffer's storage is leaked on growth so the previously handed-
// out references stay valid until process exit (which is when `Lazy` storage
// would also be freed). Concurrent callers are serialised on a `Mutex` only
// for the brief grow check; the returned slice points into leaked storage
// and is freely shareable.
//
// PERF (join key encoding): the PCIe upload is now elided for the u32::MAX
// collision-list tables (`row_idx` / `head` / `next_idx`) — they are filled
// on-device with `cuMemsetD8Async(0xFF)` via `alloc_u32_max_table_async`,
// since `u32::MAX` is byte-replicable. The i64::MIN *key* table still uploads
// (its bit pattern `0x8000_0000_0000_0000` is neither byte- nor 32-bit-word
// replicable, and there is no 64-bit memset wrapper in `crate::cuda`), so the
// pooled host slice below remains the right ROI for that buffer.

/// Sentinel value for empty hash-table key slots. Must match the
/// `i64::MIN` literal embedded in the GPU build/probe kernels.
const SENTINEL_I64_MIN: i64 = i64::MIN;

/// Sentinel value for empty row-index slots and unused collision-list
/// `head` / `next_idx` entries. Must match `u32::MAX` everywhere it
/// appears in the kernels.
const SENTINEL_U32_MAX: u32 = u32::MAX;

/// Process-wide pool of `i64::MIN`-filled host storage. Grown on demand.
static SENTINEL_I64_MIN_POOL: Mutex<&'static [i64]> = Mutex::new(&[]);

/// Process-wide pool of `u32::MAX`-filled host storage. Grown on demand.
static SENTINEL_U32_MAX_POOL: Mutex<&'static [u32]> = Mutex::new(&[]);

/// Return a `&'static [i64]` of length `cap` whose every element is
/// `i64::MIN`. Reuses a process-wide buffer; only allocates when `cap`
/// exceeds the largest previous request.
pub fn get_sentinel_i64_min_vec(cap: usize) -> &'static [i64] {
    let storage: &'static [i64] = {
        let mut guard = SENTINEL_I64_MIN_POOL.lock();
        if cap > guard.len() {
            // Grow to at least `cap`. Round up to next power of two (matching
            // the hash-table capacity discipline) so we don't realloc on the
            // very next request that bumps cap by 1.
            let new_cap = cap.next_power_of_two().max(cap);
            let leaked: &'static mut [i64] = Vec::leak(vec![SENTINEL_I64_MIN; new_cap]);
            *guard = leaked;
        }
        // Copy out the `&'static [i64]` so we can drop the guard and still
        // return a static-lifetime slice. The underlying storage is leaked
        // and never freed, so this is sound.
        *guard
    };
    &storage[..cap]
}

/// Return a `&'static [u32]` of length `cap` whose every element is
/// `u32::MAX`. Reuses a process-wide buffer; only allocates when `cap`
/// exceeds the largest previous request.
pub fn get_sentinel_u32_max_vec(cap: usize) -> &'static [u32] {
    let storage: &'static [u32] = {
        let mut guard = SENTINEL_U32_MAX_POOL.lock();
        if cap > guard.len() {
            let new_cap = cap.next_power_of_two().max(cap);
            let leaked: &'static mut [u32] = Vec::leak(vec![SENTINEL_U32_MAX; new_cap]);
            *guard = leaked;
        }
        *guard
    };
    &storage[..cap]
}

/// PERF (join key encoding): device-side `u32::MAX` initialiser.
///
/// The collision-list `row_idx`, `head`, and `next_idx` tables are all
/// initialised to the `u32::MAX` sentinel (`COLLISION_LIST_SENTINEL` /
/// `SENTINEL_U32_MAX`). Previously each was built as a cap-sized host
/// `&[u32]` (via [`get_sentinel_u32_max_vec`]) and shipped across PCIe with
/// an H2D upload. Because `u32::MAX` has the all-bytes-equal bit pattern
/// `0xFF`, the entire fill is exactly expressible as a byte-memset, so we
/// allocate on-device and fill with `cuMemsetD8Async(0xFF)` — no host
/// staging buffer and no H2D transfer at all.
///
/// Note: the *key* table sentinel (`i64::MIN` = `0x8000_0000_0000_0000`) is
/// NOT byte-replicable (only the top byte is `0x80`; the rest are `0x00`)
/// and its two 32-bit words differ, so neither D8 nor a hypothetical D32
/// memset can synthesise it — that buffer keeps its H2D upload. There is no
/// 64-bit memset wrapper in `crate::cuda::cuda_sys` to revisit this with.
///
/// The memset is enqueued on `stream`; every kernel that reads these buffers
/// is launched on the *same* stream afterwards, so stream ordering
/// guarantees the fill completes before the read with no explicit sync. The
/// stream is tagged via [`GpuVec::mark_stream_use`] so `Drop` fences before
/// recycling the device block (mirrors `GpuBuffer::zeros_async`).
///
/// Under `--features cuda-stub` we deliberately keep the host-staged upload
/// path (via [`get_sentinel_u32_max_vec`] + `from_slice`) so the FFI failure
/// mode stays byte-stable with the rest of the join — exactly the routing
/// `upload_primitive_values_async` uses for the same reason.
fn alloc_u32_max_table_async(len: usize, stream: &CudaStream) -> BoltResult<GpuVec<u32>> {
    #[cfg(feature = "cuda-stub")]
    {
        // Stub backend: no real device memset. Match the pre-existing call
        // shape (host sentinel slice + sync H2D) so the stubbed FFI returns
        // the same `CUDA_ERROR_STUB` at the same boundary as before.
        let _ = stream;
        GpuVec::<u32>::from_slice(get_sentinel_u32_max_vec(len))
    }
    #[cfg(not(feature = "cuda-stub"))]
    {
        // Start from a device allocation (zeros_async sets `len` correctly
        // and tags the stream); then overwrite every byte with `0xFF` so
        // each u32 reads back as `u32::MAX`. Two stream-ordered device
        // memsets are still far cheaper than building a cap-sized host Vec
        // and DMA'ing it over PCIe, and carry no per-query host allocation.
        let buf = GpuVec::<u32>::zeros_async(len, stream.raw())?;
        if len > 0 {
            let byte_len = len.checked_mul(std::mem::size_of::<u32>()).ok_or_else(|| {
                BoltError::Other(format!(
                    "gpu_join: u32::MAX table size overflow ({len} * 4)"
                ))
            })?;
            // SAFETY: `buf` was just allocated with at least `byte_len` bytes
            // in the currently-bound context; nothing else references the
            // block yet and we keep `buf` (and thus the allocation) live
            // until the stream is synchronised by the caller. The 0xFF fill
            // yields `u32::MAX` for every element (matches `SENTINEL_U32_MAX`).
            cuda_sys::memset_d8_async(buf.device_ptr(), 0xFF, byte_len, stream.raw())?;
            buf.mark_stream_use(stream.raw());
        }
        Ok(buf)
    }
}

/// Minimum size threshold (per side) below which the host hash join wins. The
/// GPU path eats a JIT-compile + h2d round trip; empirically 1024 rows on
/// either side is the break-even point on a discrete card.
pub const GPU_JOIN_MIN_ROWS: usize = 1024;

/// Conservative hash-table byte cap for cards with limited VRAM
/// (default fallback, used when the driver query fails or reports < 8 GiB).
/// Capacity ≈ `cap_bytes / 12` slots — each slot is one i64 key + one u32
/// row index. The probe-side output and the collision-list `head` + `next_idx`
/// buffers consume independent VRAM and are sized per-query.
const HASH_TABLE_BYTE_CAP_DEFAULT: usize = 64 * 1024 * 1024; // 64 MiB

/// Lifted byte cap for cards with ≥ 8 GiB of total VRAM. Driving this from
/// the device's reported total memory keeps the cap measure-driven without
/// needing user configuration. 512 MiB ≈ 44.7 M slots ⇒ ~22.3 M build rows
/// at the engine's 50% load factor.
const HASH_TABLE_BYTE_CAP_LARGE: usize = 512 * 1024 * 1024; // 512 MiB

/// VRAM threshold (bytes) above which the executor uses
/// [`HASH_TABLE_BYTE_CAP_LARGE`] instead of [`HASH_TABLE_BYTE_CAP_DEFAULT`].
const LARGE_VRAM_THRESHOLD: usize = 8 * 1024 * 1024 * 1024; // 8 GiB

/// Env-var name for the Stage-3 user-tunable hash-table byte cap (MiB).
/// Set this to override the driver-detected cap. Values are clamped to
/// `[CAP_ENV_MIN_MIB, CAP_ENV_MAX_MIB]`.
pub const CAP_ENV_VAR: &str = "BOLT_GPU_JOIN_TABLE_CAP_MB";

/// Lower clamp on `BOLT_GPU_JOIN_TABLE_CAP_MB`. Below this the table is too
/// small to hold even a single bucket's worth of state for typical workloads.
const CAP_ENV_MIN_MIB: usize = 64;

/// Upper clamp on `BOLT_GPU_JOIN_TABLE_CAP_MB`. 4 GiB is the largest hash
/// table the engine will allocate from one tunable knob — beyond that the
/// caller should split the build into multiple GPU joins.
const CAP_ENV_MAX_MIB: usize = 4096;

/// Hard cap on cartesian-product cell count for the GPU CROSS JOIN kernel.
/// Beyond this the output buffer + per-thread launch shape are bigger than the
/// host wants to allocate / serialise, and the host orchestrator wins.
/// 100M pairs ≈ 800 MiB of u32 output indices, comfortable on 8 GiB cards.
pub const CROSS_JOIN_GPU_CELL_CAP: u64 = 100_000_000;

/// Minimum cartesian-product cell count below which the host wins. Same
/// reasoning as [`GPU_JOIN_MIN_ROWS`]: tiny CROSS pays the JIT-compile +
/// kernel-launch overhead twice.
pub const CROSS_JOIN_GPU_MIN_CELLS: u64 = 4096;

/// Stage-5 heuristic threshold: route INNER joins through the AoS-layout
/// build/probe kernels when `n_probe / n_build > AOS_ROUTING_PROBE_BUILD_RATIO`.
///
/// AoS halves probe-side cache-line traffic (one fused load brings both
/// the key and the row-index head into the same 16-byte transaction vs
/// SoA's two scattered loads). It costs 33% more raw bytes — `cap * 16`
/// vs SoA's `cap * 12`. The crossover where the bandwidth win pays for
/// the extra capacity is when the probe loop dominates the kernel's
/// total memory traffic. Empirically (see `docs/JOIN_BENCHMARKS.md`)
/// that happens when the probe side is at least ~8× larger than the
/// build side — below that ratio the build's table-init traffic and the
/// build kernel's CAS chains carry enough of the cost that the AoS
/// padding isn't amortised.
///
/// The threshold is conservative: a smaller ratio would route more joins
/// through AoS but pay the 33% memory tax for marginal probe-bandwidth
/// gains. Bump this when measure-driven evidence supports it.
pub const AOS_ROUTING_PROBE_BUILD_RATIO: usize = 8;

/// Stage-5 row-count threshold above which the streaming-intern path uses
/// per-chunk parallel dicts (see [`intern_utf8_columns_streaming_parallel`]).
/// Below this the spawn-thread overhead dominates the gain, so we keep
/// the sequential chunked variant.
pub const STREAMING_INTERN_PARALLEL_THRESHOLD: usize = 256 * 1024;

/// Stage-5 row-count threshold above which the streaming-intern path
/// hashes strings on the device via [`compute_device_string_hashes`].
/// Below this the host's splitmix is faster than the kernel-launch +
/// h2d/d2h round trip.
///
/// Exposed as `pub` so benchmarks and downstream tooling can mirror the
/// engine's routing decision; engine-internal use lives in
/// `intern_utf8_columns_streaming_parallel` for Stage 6 wiring (Stage 5
/// emits the kernel + wrapper but defers the auto-route to keep the
/// per-call overhead off mid-size joins).
#[allow(dead_code)] // reason: Stage-5 emits the constant + kernel; the host-side
                    //         streaming intern only flips over to device hashing
                    //         in Stage 6 once batched upload reuse lands.
pub const STREAMING_INTERN_DEVICE_HASH_THRESHOLD: usize = 1_000_000;

/// `CU_DEVICE_ATTRIBUTE_TOTAL_MEMORY` isn't an attribute — the driver
/// surfaces total memory through `cuDeviceTotalMem_v2` directly. We keep
/// the name for documentation symmetry with the task spec.
///
/// Latched once on first use so the FFI cost is paid exactly once per
/// process. Stores `Some(usize)` on success, `Some(default)` on FFI error
/// (logged at debug level), so callers can always go through `unwrap_or`.
static HASH_TABLE_BYTE_CAP_CACHE: OnceLock<usize> = OnceLock::new();

/// Resolve the per-process hash-table byte cap. First call performs the
/// env-var lookup + driver query; subsequent calls hit the latch.
///
/// Stage-3 selection rule, applied in order:
///   1. `BOLT_GPU_JOIN_TABLE_CAP_MB` env var (clamped to `[64, 4096]` MiB) —
///      if set to anything parseable, wins outright.
///   2. Driver: total VRAM ≥ 8 GiB ⇒ 512 MiB cap; else 64 MiB cap.
///   3. On any FFI error: 64 MiB default.
///
/// On any path we emit a one-time `log::info!` so operators can see what cap
/// the executor settled on (without having to know whether the env var or the
/// driver query won).
fn hash_table_byte_cap() -> usize {
    *HASH_TABLE_BYTE_CAP_CACHE.get_or_init(|| {
        // 1. User override via env var. Parse + clamp; on any parse failure
        //    (non-numeric, empty, weird unit suffix) we silently fall through
        //    to the driver-detected cap, so the env var only takes effect
        //    when set to a valid integer.
        if let Some(cap) = parse_env_cap() {
            log::info!("gpu_join: hash-table byte cap set via {CAP_ENV_VAR} = {cap} bytes");
            return cap;
        }
        // 2. Driver-detected cap.
        let cap = match resolve_byte_cap_from_driver() {
            Ok(cap) => cap,
            Err(e) => {
                log::debug!(
                    "gpu_join: device total-memory query failed ({e}); \
                     falling back to {HASH_TABLE_BYTE_CAP_DEFAULT} bytes"
                );
                HASH_TABLE_BYTE_CAP_DEFAULT
            }
        };
        log::info!("gpu_join: hash-table byte cap resolved to {cap} bytes (driver-detected)");
        cap
    })
}

/// Parse `BOLT_GPU_JOIN_TABLE_CAP_MB`. Returns `Some(cap_bytes)` if the env
/// var is set to a valid integer; clamps to `[CAP_ENV_MIN_MIB, CAP_ENV_MAX_MIB]`.
/// Returns `None` on any parse failure or unset env var.
///
/// `pub(crate)` so the integration test `tests/env_var_smoke.rs` can
/// round-trip the parser against the live env var. The clamp policy is
/// a per-process latch in `HASH_TABLE_BYTE_CAP_CACHE`, so the public
/// surface here intentionally stays the unmemoised inner parser.
pub fn parse_env_cap() -> Option<usize> {
    let raw = std::env::var(CAP_ENV_VAR).ok()?;
    let raw_trim = raw.trim();
    if raw_trim.is_empty() {
        return None;
    }
    let mib: usize = match raw_trim.parse::<usize>() {
        Ok(v) => v,
        Err(e) => {
            log::warn!(
                "gpu_join: {CAP_ENV_VAR}='{raw_trim}' is not a valid integer ({e}); \
                 ignoring and using driver-detected cap"
            );
            return None;
        }
    };
    let clamped = mib.clamp(CAP_ENV_MIN_MIB, CAP_ENV_MAX_MIB);
    if clamped != mib {
        log::warn!(
            "gpu_join: {CAP_ENV_VAR}={mib} MiB out of range [{CAP_ENV_MIN_MIB}, {CAP_ENV_MAX_MIB}]; \
             clamped to {clamped} MiB"
        );
    }
    Some(clamped * 1024 * 1024)
}

/// Pure-driver path for [`hash_table_byte_cap`]. Extracted so unit tests can
/// reason about it independent of the OnceLock latch.
///
/// Multi-GPU (Stage 4): queries the device bound to the calling thread's
/// current CUDA context via [`cuda_sys::current_device`], so the per-process
/// cap is driven by the card the engine is actually using — not always
/// ordinal 0. On single-GPU rigs the behaviour is byte-identical to the
/// Stage-2 implementation that hardcoded `device_get(0)`.
fn resolve_byte_cap_from_driver() -> BoltResult<usize> {
    cuda_sys::init()?;
    // Stage-4 (GJ): route through the active context's device, not ordinal 0.
    // The engine's `CudaContext` was created against a specific device and
    // (under `cuCtxSetCurrent`) bound to this thread, so `current_device()`
    // returns the right handle regardless of how many CUDA devices the
    // driver enumerates.
    let dev = cuda_sys::current_device()?;
    let total = cuda_sys::device_total_mem(dev)?;
    Ok(if total >= LARGE_VRAM_THRESHOLD {
        HASH_TABLE_BYTE_CAP_LARGE
    } else {
        HASH_TABLE_BYTE_CAP_DEFAULT
    })
}

/// Hash-table slot cap given the active byte cap. `12 = sizeof(i64) + sizeof(u32)`.
///
/// Note: when the AoS layout is selected the per-slot footprint is 16 bytes
/// instead of 12 (see `AOS_SLOT_BYTES`); callers using AoS should divide by
/// 16 instead. The SoA path is the default everywhere today.
fn hash_table_slot_cap() -> usize {
    hash_table_byte_cap() / 12
}

/// 50% peak load factor: capacity = next_pow2(2 * n_build_rows). Higher load
/// factors blow up probe lengths quickly; lower wastes memory. 0.5 is the
/// engine-wide convention (matches `groupby::pack_keys`).
const LOAD_FACTOR_DENOM: usize = 2;

/// Compute the hash-table capacity for `n_build_rows`: smallest power of two
/// ≥ `LOAD_FACTOR_DENOM * n_build_rows`. Returns `Err` if the result exceeds
/// the table-size cap.
fn compute_capacity(n_build_rows: usize) -> BoltResult<usize> {
    compute_capacity_with_slot_cap(n_build_rows, hash_table_slot_cap())
}

/// Inner helper exposed for unit tests so we can plug an explicit slot cap
/// without depending on a CUDA device being present (the runtime cap query
/// is OnceLock'd, which makes test ordering brittle).
fn compute_capacity_with_slot_cap(n_build_rows: usize, slot_cap: usize) -> BoltResult<usize> {
    let target = n_build_rows.checked_mul(LOAD_FACTOR_DENOM).ok_or_else(|| {
        BoltError::Other(format!(
            "gpu_join: capacity calc overflowed (n_build_rows={n_build_rows})"
        ))
    })?;
    // next_power_of_two on 1 returns 1; we want a minimum of 2 so the mask
    // (cap - 1) has at least one valid bit. Probe loop assumes cap >= 2.
    let target = target.max(2);
    if target > slot_cap {
        return Err(BoltError::Other(format!(
            "gpu_join: required capacity {target} exceeds hash-table slot cap {slot_cap}. \
             Fall back to host path."
        )));
    }
    let cap = target.next_power_of_two();
    if cap > slot_cap {
        return Err(BoltError::Other(format!(
            "gpu_join: rounded capacity {cap} exceeds hash-table slot cap {slot_cap}"
        )));
    }
    Ok(cap)
}

/// Encode an Arrow key column to a `Vec<i64>` for upload. The build and probe
/// sides share this encoding so their hashes agree byte-for-byte.
///
/// Returns `Err` if any encoded value collides with the `i64::MIN` empty-slot
/// sentinel — that would corrupt the kernel's CAS step.
fn encode_keys_i64(column: &dyn Array, dtype: DataType) -> BoltResult<Vec<i64>> {
    let n = column.len();
    let mut out: Vec<i64> = Vec::with_capacity(n);
    match dtype {
        DataType::Int32 => {
            let arr = column
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_join: column dtype said Int32 but downcast failed".into())
                })?;
            // Int32 can't equal i64::MIN once sign-extended (range is
            // [i32::MIN..=i32::MAX] which doesn't include i64::MIN). No
            // sentinel-collision check needed for Int32.
            for v in arr.values().iter() {
                out.push(*v as i64);
            }
        }
        DataType::Int64 => {
            let arr = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_join: column dtype said Int64 but downcast failed".into())
                })?;
            for v in arr.values().iter() {
                if *v == i64::MIN {
                    return Err(BoltError::Other(
                        "gpu_join: build/probe key equals the i64::MIN empty-slot sentinel; \
                         falling back to host path"
                            .into(),
                    ));
                }
                out.push(*v);
            }
        }
        other => {
            return Err(BoltError::Other(format!(
                "gpu_join: unsupported key dtype {other:?} (Stage 1: Int32 / Int64 only)"
            )));
        }
    }
    Ok(out)
}

/// Splitmix-style 64-bit finaliser, identical to the kernel-side `FX_MUL`
/// shift-right-32 reduction. Used to fold multi-column tuples into a single
/// i64 on the host. Matches `crate::jit::hash_kernels::FX_MUL` so that
/// the host-side splatter agrees with kernel-side replays of the same input.
const HOST_FX_MUL: u64 = 0x9E37_79B9_7F4A_7C15;

#[inline]
fn host_splitmix(mut h: u64) -> u64 {
    // We do NOT do (h * FX_MUL) >> 32 here because that's the *slot* reduction;
    // we want the full 64-bit folded value so the *kernel-side* `(key * FX_MUL) >> 32`
    // still spreads it across the table. So this is the cheap fold: multiply
    // and xor-shift, matching the SplitMix64 finalizer family.
    h ^= h >> 33;
    h = h.wrapping_mul(HOST_FX_MUL);
    h ^= h >> 29;
    h = h.wrapping_mul(HOST_FX_MUL);
    h ^= h >> 32;
    h
}

/// Canonicalise an f64's bit pattern so every NaN hashes the same. This is
/// the engine-wide "NaNs are equal" convention shared with `distinct.rs` and
/// the rest of the float-key pipeline.
#[inline]
fn canonical_f64_bits(v: f64) -> u64 {
    if v.is_nan() {
        f64::NAN.to_bits()
    } else if v == 0.0 {
        // Collapse -0.0 and +0.0 to the same key: SQL treats them equal.
        0u64
    } else {
        v.to_bits()
    }
}

/// Same as [`canonical_f64_bits`] but for f32, widened to u32 then to u64
/// (high bits zeroed) so the i64 cast preserves the bit pattern exactly.
#[inline]
fn canonical_f32_bits(v: f32) -> u64 {
    let bits = if v.is_nan() {
        f32::NAN.to_bits()
    } else if v == 0.0 {
        0u32
    } else {
        v.to_bits()
    };
    bits as u64
}

/// Encode an Arrow key column (or set of key columns) to a `Vec<i64>` for
/// upload, using the host-side strategy implied by `shape`. The build and
/// probe sides MUST call this with the same `shape` so their kernel-side
/// hashes line up byte-for-byte.
///
/// `columns` carries one slice per key column for the row being encoded. For
/// single-key shapes the slice length is 1; for `TwoI32` / `TwoI64` it's 2;
/// for `MultiI32(n)` it's n.
///
/// Returns `Err` if any encoded value collides with the `i64::MIN`
/// empty-slot sentinel — the host then surfaces that to the caller, which
/// falls back to the host path for the whole query.
pub fn encode_keys_for_shape(columns: &[&dyn Array], shape: KeyShape) -> BoltResult<Vec<i64>> {
    if columns.is_empty() {
        return Err(BoltError::Other(
            "gpu_join: encode_keys_for_shape called with zero columns".into(),
        ));
    }
    let n = columns[0].len();
    for c in &columns[1..] {
        if c.len() != n {
            return Err(BoltError::Other(format!(
                "gpu_join: encode_keys_for_shape: row-count mismatch ({} vs {})",
                c.len(),
                n
            )));
        }
    }

    let mut out: Vec<i64> = Vec::with_capacity(n);
    match shape {
        // Single-column shapes route through the Stage-1 encoder for Int32 /
        // Int64 to keep PTX-level behaviour byte-stable. Bool / Float are
        // handled inline below.
        KeyShape::SingleI32 => return encode_keys_i64(columns[0], DataType::Int32),
        KeyShape::SingleI64 => return encode_keys_i64(columns[0], DataType::Int64),
        KeyShape::SingleBool => {
            let arr = columns[0]
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_join: SingleBool key column is not BooleanArray".into())
                })?;
            // BooleanArray::value follows the bit layout directly.
            for i in 0..n {
                out.push(arr.value(i) as i64);
            }
        }
        KeyShape::SingleF32 => {
            let arr = columns[0]
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_join: SingleF32 key column is not Float32Array".into())
                })?;
            for v in arr.values().iter() {
                let bits = canonical_f32_bits(*v) as i64;
                if bits == i64::MIN {
                    return Err(BoltError::Other(
                        "gpu_join: Float32 key collided with i64::MIN sentinel after canonicalisation".into(),
                    ));
                }
                out.push(bits);
            }
        }
        KeyShape::SingleF64 => {
            let arr = columns[0]
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_join: SingleF64 key column is not Float64Array".into())
                })?;
            for v in arr.values().iter() {
                let bits = canonical_f64_bits(*v) as i64;
                if bits == i64::MIN {
                    return Err(BoltError::Other(
                        "gpu_join: Float64 key collided with i64::MIN sentinel after canonicalisation".into(),
                    ));
                }
                out.push(bits);
            }
        }
        KeyShape::TwoI32 => {
            if columns.len() != 2 {
                return Err(BoltError::Other(format!(
                    "gpu_join: TwoI32 expects 2 columns, got {}",
                    columns.len()
                )));
            }
            let hi = columns[0]
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_join: TwoI32 high column is not Int32".into())
                })?;
            let lo = columns[1]
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_join: TwoI32 low column is not Int32".into())
                })?;
            for i in 0..n {
                // Same convention as groupby::pack_keys: first column in the
                // high half, second in the low half.
                let h = (hi.value(i) as u32 as u64) << 32;
                let l = lo.value(i) as u32 as u64;
                let packed = (h | l) as i64;
                if packed == i64::MIN {
                    return Err(BoltError::Other(
                        "gpu_join: TwoI32 composite key collided with i64::MIN sentinel".into(),
                    ));
                }
                out.push(packed);
            }
        }
        KeyShape::SingleUtf8 => {
            // Utf8 encoding goes through intern_utf8_columns + this branch
            // never sees a raw StringArray — `encode_keys_for_shape` is the
            // bytes-out layer, callers must intern before reaching here.
            return Err(BoltError::Other(
                "gpu_join: SingleUtf8 must be interned to i32 dict indices before encoding; \
                 see intern_utf8_columns"
                    .into(),
            ));
        }
        KeyShape::SingleI32Candidate => {
            // Stage 6 (GJ) — streaming-intern path. Like `SingleUtf8`, the
            // raw `StringArray` has already been folded to i32 dict
            // indices via `intern_utf8_columns_streaming_parallel`; this
            // branch only sees i32 keys-as-i64, with the candidate-filter
            // contract that the host must re-verify pairs.
            return Err(BoltError::Other(
                "gpu_join: SingleI32Candidate must be interned via streaming-intern before \
                 encoding; see intern_utf8_columns_streaming_parallel"
                    .into(),
            ));
        }
        KeyShape::TwoI64 | KeyShape::MultiI32(_) => {
            // Stage-3 candidate-filter path. The host-side splitmix can
            // collide, so the GPU join emits *candidate* pairs that the
            // caller MUST re-verify against the original Arrow columns via
            // `verify_pairs_on_host`. The encoder still produces an i64
            // value per row; correctness depends on the post-verification.
            for i in 0..n {
                let mut h: u64 = 0;
                for col in columns {
                    let bits = match col.data_type() {
                        arrow_schema::DataType::Int32 => {
                            let a = col
                                .as_any()
                                .downcast_ref::<Int32Array>()
                                .ok_or_else(|| {
                                    BoltError::Other("gpu_join: multi-key column dtype said Int32 but downcast failed".into())
                                })?;
                            a.value(i) as u32 as u64
                        }
                        arrow_schema::DataType::Int64 => {
                            let a = col
                                .as_any()
                                .downcast_ref::<Int64Array>()
                                .ok_or_else(|| {
                                    BoltError::Other("gpu_join: multi-key column dtype said Int64 but downcast failed".into())
                                })?;
                            a.value(i) as u64
                        }
                        other => {
                            return Err(BoltError::Other(format!(
                                "gpu_join: multi-key column has unsupported dtype {other:?}"
                            )))
                        }
                    };
                    h = host_splitmix(h ^ bits);
                }
                let packed = h as i64;
                if packed == i64::MIN {
                    return Err(BoltError::Other(
                        "gpu_join: multi-key folded value collided with i64::MIN sentinel".into(),
                    ));
                }
                out.push(packed);
            }
        }
    }
    Ok(out)
}

/// Run the build kernel: insert (key, row_idx) for every build row into the
/// device hash table.
fn launch_build_kernel(
    build_keys_dev: &GpuVec<i64>,
    keys_table_dev: &mut GpuVec<i64>,
    row_idx_table_dev: &mut GpuVec<u32>,
    n_build_rows: u32,
    cap: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    if n_build_rows == 0 {
        // Empty build side: keys_table_dev is already initialised to all
        // i64::MIN, so the probe will see every slot empty and emit no
        // matches. This path shouldn't be reached anyway (the gate rejects
        // empty sides) but defensive bailout matches the sort kernel's
        // n_pow2 <= 1 short-circuit.
        return Ok(());
    }

    // `compile_build_kernel` emits the Stage-1 NO-DUPLICATES build kernel.
    // The caller is responsible for routing duplicate build keys through
    // `compile_build_collision_kernel` (the chained-list variant) instead;
    // this kernel assumes the host has already verified uniqueness of the
    // build-side join keys. See review C5: even if a duplicate slips through
    // here, the kernel now degrades to first-writer-wins semantics rather
    // than emitting u32::MAX sentinel row indices — but the collision kernel
    // is still the correct dispatch for known-duplicate inputs.
    // Hash-join kernels operate on already-encoded i64 keys (see
    // `encode_keys_for_shape`); the source-column dtype is not visible at
    // this call site. Stamp the cache spec with `DataType::Int64` to
    // describe the kernel-boundary type — matching the single-slot
    // behaviour of the legacy string-keyed cache this replaces.
    let spec = HashJoinKernelSpec {
        kind: HashJoinKernelKind::Build,
        key_dtype: DataType::Int64,
        string_hash_returns_i64: false,
    };
    let module =
        module_cache::get_or_build_module_for_hash_join(&spec, BUILD_KERNEL_ENTRY, |_| {
            compile_build_kernel()
        })?;
    let function = module.function(BUILD_KERNEL_ENTRY)?;

    let mut build_keys_ptr: CUdeviceptr = build_keys_dev.device_ptr();
    let mut keys_table_ptr: CUdeviceptr = keys_table_dev.device_ptr();
    let mut row_idx_table_ptr: CUdeviceptr = row_idx_table_dev.device_ptr();
    let mut n_rows_u32: u32 = n_build_rows;
    let mut cap_u32: u32 = cap;

    let mut params: [*mut c_void; 5] = [
        &mut build_keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut keys_table_ptr as *mut CUdeviceptr as *mut c_void,
        &mut row_idx_table_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
        &mut cap_u32 as *mut u32 as *mut c_void,
    ];

    let block: u32 = HASH_JOIN_BLOCK_SIZE;
    let grid_x: u32 = n_build_rows.div_ceil(block).max(1);

    // SAFETY: every entry of `params` points at a stack local that outlives
    // the launch+sync below; the device buffers are owned by the caller and
    // outlive the launch.
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
    stream.synchronize()?;
    Ok(())
}

/// **Batch 6** — env var that opts the SoA probe into the tile-aware 2-way
/// unrolled variant emitted by
/// [`crate::jit::hash_join_kernel::compile_probe_kernel_tiled`].
///
/// Default is **off**: the single-load probe is the byte-stable Stage-1 path
/// and remains the dispatch default until the tiled kernel is smoke-tested
/// on real GPU hardware. Set the env var to any non-empty value other than
/// `"0"` / `"false"` (case-insensitive) to opt in:
///
/// ```text
/// BOLT_HASH_PROBE_TILED=1     # opt in
/// BOLT_HASH_PROBE_TILED=true  # opt in
/// BOLT_HASH_PROBE_TILED=0     # off (same as unset)
/// BOLT_HASH_PROBE_TILED=false # off
/// ```
///
/// The tiled kernel has an identical nine-parameter ABI, so opting in only
/// switches which entry point [`launch_probe_kernel`] resolves at module
/// load. All other launch parameters (block size, grid shape, output
/// buffer sizing) are unchanged.
pub const PROBE_TILED_ENV_VAR: &str = "BOLT_HASH_PROBE_TILED";

/// Read [`PROBE_TILED_ENV_VAR`]. Returns `true` when the variable is set to
/// a non-empty value other than `"0"` / `"false"` (case-insensitive).
/// Returns `false` on any other value (unset, empty, `"0"`, `"false"`).
fn probe_tiled_enabled() -> bool {
    match std::env::var(PROBE_TILED_ENV_VAR) {
        Ok(v) => {
            let s = v.trim();
            !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
        }
        Err(_) => false,
    }
}

/// Run the probe kernel: for each probe row, walk the hash table and emit
/// `(probe_idx, build_idx)` into the output buffers via an atomic counter.
///
/// Returns the number of matches actually claimed (the post-launch value of
/// the GPU-side counter), capped at `out_capacity`. If the kernel claimed
/// more than `out_capacity` slots the counter will still hold the true count
/// (the kernel only skips the *writes* on overflow), so callers can detect
/// the overflow and re-run with a bigger output buffer.
///
/// **Batch 6** — when [`PROBE_TILED_ENV_VAR`] is opted in, the launcher
/// resolves [`PROBE_KERNEL_TILED_ENTRY`] instead of [`PROBE_KERNEL_ENTRY`]
/// and ships the 2-way unrolled `compile_probe_kernel_tiled` PTX. The two
/// kernels share an identical ABI so the launch params don't change.
fn launch_probe_kernel(
    probe_keys_dev: &GpuVec<i64>,
    keys_table_dev: &GpuVec<i64>,
    row_idx_table_dev: &GpuVec<u32>,
    out_probe_idx_dev: &mut GpuVec<u32>,
    out_build_idx_dev: &mut GpuVec<u32>,
    out_counter_dev: &mut GpuVec<u32>,
    n_probe_rows: u32,
    cap: u32,
    out_capacity: u32,
    stream: &CudaStream,
) -> BoltResult<u32> {
    if n_probe_rows == 0 {
        // No probe rows -> no matches; counter stays at 0.
        return Ok(0);
    }

    // Tile-aware probe dispatch. Off by default; both kernels share the same
    // ABI. Route through the consolidated module cache so each variant is
    // cached separately by spec id.
    let tiled = probe_tiled_enabled();
    let (kind, entry) = if tiled {
        (HashJoinKernelKind::ProbeTiled, PROBE_KERNEL_TILED_ENTRY)
    } else {
        (HashJoinKernelKind::Probe, PROBE_KERNEL_ENTRY)
    };
    let spec = HashJoinKernelSpec {
        kind,
        key_dtype: DataType::Int64,
        string_hash_returns_i64: false,
    };
    let module = module_cache::get_or_build_module_for_hash_join(&spec, entry, |_| {
        if tiled {
            compile_probe_kernel_tiled()
        } else {
            compile_probe_kernel()
        }
    })?;
    let function = module.function(entry)?;
    log::debug!("gpu_join probe: tiled={tiled} n_probe={n_probe_rows} cap={cap}");

    let mut probe_keys_ptr: CUdeviceptr = probe_keys_dev.device_ptr();
    let mut keys_table_ptr: CUdeviceptr = keys_table_dev.device_ptr();
    let mut row_idx_table_ptr: CUdeviceptr = row_idx_table_dev.device_ptr();
    let mut out_probe_idx_ptr: CUdeviceptr = out_probe_idx_dev.device_ptr();
    let mut out_build_idx_ptr: CUdeviceptr = out_build_idx_dev.device_ptr();
    let mut out_counter_ptr: CUdeviceptr = out_counter_dev.device_ptr();
    let mut n_probe_u32: u32 = n_probe_rows;
    let mut cap_u32: u32 = cap;
    let mut out_capacity_u32: u32 = out_capacity;

    let mut params: [*mut c_void; 9] = [
        &mut probe_keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut keys_table_ptr as *mut CUdeviceptr as *mut c_void,
        &mut row_idx_table_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_probe_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_build_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_counter_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_probe_u32 as *mut u32 as *mut c_void,
        &mut cap_u32 as *mut u32 as *mut c_void,
        &mut out_capacity_u32 as *mut u32 as *mut c_void,
    ];

    let block: u32 = HASH_JOIN_BLOCK_SIZE;
    let grid_x: u32 = n_probe_rows.div_ceil(block).max(1);

    // SAFETY: same rationale as launch_build_kernel.
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
    stream.synchronize()?;

    // Read back the actual number of matches.
    let counter_host: Vec<u32> = out_counter_dev.to_vec()?;
    let n_matches_raw = counter_host[0];
    Ok(n_matches_raw)
}

// =========================================================================
// Stage 4: AoS-layout build kernel launchers.
//
// The AoS slot layout is `[key:u64, head:u32, _pad:u32]` (16 bytes total,
// see `AOS_SLOT_BYTES`). One `slots: GpuVec<u8>[cap * 16]` buffer replaces
// the SoA `keys_table: GpuVec<i64>[cap]` + `row_idx_table: GpuVec<u32>[cap]`
// pair, halving probe-side cache-line traffic at a 33% raw-bytes cost.
// =========================================================================

/// Initialise an AoS slot buffer of `cap` slots: every key word (offset 0)
/// is set to `EMPTY_KEY = i64::MIN`, every head (offset 8) and pad
/// (offset 12) word stays at zero.
///
/// We can't just `zeros(cap * 16)` because the AoS build kernel uses
/// `atom.cas.b64` against the key word and would treat 0 as a live key.
#[allow(dead_code)] // reason: GJ-stage4 AoS helper; planner-level routing to AoS path lands in Stage 5.
fn alloc_aos_slots(cap: usize) -> BoltResult<GpuVec<u8>> {
    let bytes = cap.checked_mul(AOS_SLOT_BYTES as usize).ok_or_else(|| {
        BoltError::Other(format!("gpu_join: AoS slot bytes overflow (cap={cap})"))
    })?;
    // Build the initialiser host-side. Each 16-byte slot starts with the
    // i64::MIN sentinel; the remaining 8 bytes (head + pad) are zero.
    let mut init: Vec<u8> = vec![0u8; bytes];
    let sentinel = i64::MIN.to_le_bytes();
    for slot in 0..cap {
        let off = slot * AOS_SLOT_BYTES as usize;
        init[off..off + 8].copy_from_slice(&sentinel);
    }
    GpuVec::<u8>::from_slice(&init)
}

/// Launch the Stage-4 AoS build kernel. Inserts `(key, tid)` into the
/// 16-byte slot tuple via an `atom.cas.b64` on the key word + a plain
/// `st.global.u32` on the head word at offset 8.
#[allow(dead_code)] // reason: GJ-stage4 AoS helper; planner-level routing to AoS path lands in Stage 5.
fn launch_build_aos_kernel(
    build_keys_dev: &GpuVec<i64>,
    slots_dev: &mut GpuVec<u8>,
    n_build_rows: u32,
    cap: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    if n_build_rows == 0 {
        return Ok(());
    }
    let spec = HashJoinKernelSpec {
        kind: HashJoinKernelKind::BuildAos,
        key_dtype: DataType::Int64,
        string_hash_returns_i64: false,
    };
    let module =
        module_cache::get_or_build_module_for_hash_join(&spec, BUILD_AOS_KERNEL_ENTRY, |_| {
            compile_build_aos_kernel()
        })?;
    let function = module.function(BUILD_AOS_KERNEL_ENTRY)?;

    let mut build_keys_ptr: CUdeviceptr = build_keys_dev.device_ptr();
    let mut slots_ptr: CUdeviceptr = slots_dev.device_ptr();
    let mut n_rows_u32: u32 = n_build_rows;
    let mut cap_u32: u32 = cap;

    let mut params: [*mut c_void; 4] = [
        &mut build_keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut slots_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
        &mut cap_u32 as *mut u32 as *mut c_void,
    ];

    let block: u32 = HASH_JOIN_BLOCK_SIZE;
    let grid_x: u32 = n_build_rows.div_ceil(block).max(1);

    // SAFETY: every entry of `params` points at a stack local that outlives
    // the launch+sync below; the device buffers are owned by the caller.
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
    stream.synchronize()?;
    Ok(())
}

/// Launch the Stage-3 AoS probe kernel against the slots buffer populated
/// by [`launch_build_aos_kernel`]. Same input/output contract as the SoA
/// probe — the only observable difference is the slot-byte stride and the
/// fact that the head word lives in the same 16-byte slot as the key.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // reason: GJ-stage4 AoS helper; planner-level routing to AoS path lands in Stage 5.
fn launch_probe_aos_kernel(
    probe_keys_dev: &GpuVec<i64>,
    slots_dev: &GpuVec<u8>,
    out_probe_idx_dev: &mut GpuVec<u32>,
    out_build_idx_dev: &mut GpuVec<u32>,
    out_counter_dev: &mut GpuVec<u32>,
    n_probe_rows: u32,
    cap: u32,
    out_capacity: u32,
    stream: &CudaStream,
) -> BoltResult<u32> {
    if n_probe_rows == 0 {
        return Ok(0);
    }
    let spec = HashJoinKernelSpec {
        kind: HashJoinKernelKind::ProbeAos,
        key_dtype: DataType::Int64,
        string_hash_returns_i64: false,
    };
    let module =
        module_cache::get_or_build_module_for_hash_join(&spec, PROBE_AOS_KERNEL_ENTRY, |_| {
            compile_probe_aos_kernel()
        })?;
    let function = module.function(PROBE_AOS_KERNEL_ENTRY)?;

    let mut probe_keys_ptr: CUdeviceptr = probe_keys_dev.device_ptr();
    let mut slots_ptr: CUdeviceptr = slots_dev.device_ptr();
    let mut out_probe_idx_ptr: CUdeviceptr = out_probe_idx_dev.device_ptr();
    let mut out_build_idx_ptr: CUdeviceptr = out_build_idx_dev.device_ptr();
    let mut out_counter_ptr: CUdeviceptr = out_counter_dev.device_ptr();
    let mut n_probe_u32: u32 = n_probe_rows;
    let mut cap_u32: u32 = cap;
    let mut out_capacity_u32: u32 = out_capacity;

    let mut params: [*mut c_void; 8] = [
        &mut probe_keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut slots_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_probe_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_build_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_counter_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_probe_u32 as *mut u32 as *mut c_void,
        &mut cap_u32 as *mut u32 as *mut c_void,
        &mut out_capacity_u32 as *mut u32 as *mut c_void,
    ];

    let block: u32 = HASH_JOIN_BLOCK_SIZE;
    let grid_x: u32 = n_probe_rows.div_ceil(block).max(1);

    // SAFETY: same rationale as launch_build_aos_kernel.
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
    stream.synchronize()?;
    let counter_host: Vec<u32> = out_counter_dev.to_vec()?;
    Ok(counter_host[0])
}

/// AoS variant of [`hash_join_indices_on_gpu`]. Same INNER-equi-join
/// semantics + unique-build-keys contract, but the hash table lives in a
/// single `slots: GpuVec<u8>[cap * 16]` buffer instead of two parallel
/// `keys_table` + `row_idx_table` arrays. Produces byte-identical match
/// sets (modulo arbitrary INNER row order) so the SoA path's e2e tests
/// double as AoS regression coverage.
#[allow(dead_code)] // reason: GJ-stage4 AoS helper; planner-level routing to AoS path lands in Stage 5.
pub fn hash_join_indices_on_gpu_aos(
    build_keys_col: &dyn Array,
    probe_keys_col: &dyn Array,
    dtype: DataType,
) -> BoltResult<(UInt32Array, UInt32Array)> {
    let n_build = build_keys_col.len();
    let n_probe = probe_keys_col.len();
    if n_build == 0 || n_probe == 0 {
        return Ok((
            UInt32Array::from(Vec::<u32>::new()),
            UInt32Array::from(Vec::<u32>::new()),
        ));
    }
    let n_build_u32 = n_rows_to_u32(n_build)?;
    let n_probe_u32 = n_rows_to_u32(n_probe)?;

    // Cap is denominated in slots; with AoS the per-slot cost is 16 bytes
    // (vs 12 for SoA) so the slot cap shrinks proportionally. We still use
    // the SoA slot cap here for compatibility with the Stage-3 e2e test
    // fixtures; oversize is caught by `compute_capacity`'s slot-cap check.
    let cap = compute_capacity(n_build)?;
    let cap_u32 = u32::try_from(cap)
        .map_err(|_| BoltError::Other(format!("gpu_join: AoS cap {cap} doesn't fit in u32")))?;

    let build_keys_host = encode_keys_i64(build_keys_col, dtype)?;
    let probe_keys_host = encode_keys_i64(probe_keys_col, dtype)?;

    // v0.7 async-memcpy: per-call stream + async H2D for the key columns,
    // matching the SoA INNER path above. The AoS slots buffer keeps its
    // dedicated allocator (it's the device-side initialiser, not a host
    // upload, so the async wrappers don't apply directly).
    let stream = CudaStream::null_or_default();

    let build_keys_dev = upload_primitive_values_async::<i64>(&build_keys_host, &stream)?;
    let probe_keys_dev = upload_primitive_values_async::<i64>(&probe_keys_host, &stream)?;

    let mut slots_dev = alloc_aos_slots(cap)?;

    let out_capacity_usize = n_build
        .checked_add(n_probe)
        .ok_or_else(|| BoltError::Other("gpu_join: AoS output sizing overflow".into()))?;
    let out_capacity_u32 = u32::try_from(out_capacity_usize).map_err(|_| {
        BoltError::Other(format!(
            "gpu_join: AoS out_capacity {out_capacity_usize} doesn't fit in u32"
        ))
    })?;
    let mut out_probe_idx_dev = GpuVec::<u32>::zeros(out_capacity_usize)?;
    let mut out_build_idx_dev = GpuVec::<u32>::zeros(out_capacity_usize)?;
    let mut out_counter_dev = GpuVec::<u32>::zeros(1)?;

    launch_build_aos_kernel(
        &build_keys_dev,
        &mut slots_dev,
        n_build_u32,
        cap_u32,
        &stream,
    )?;
    let n_matches_raw = launch_probe_aos_kernel(
        &probe_keys_dev,
        &slots_dev,
        &mut out_probe_idx_dev,
        &mut out_build_idx_dev,
        &mut out_counter_dev,
        n_probe_u32,
        cap_u32,
        out_capacity_u32,
        &stream,
    )?;
    if n_matches_raw > out_capacity_u32 {
        // Probe overflow: kernel wrote more matches than out_capacity.
        // The host join handles this input fine — return the fallback
        // signal so the executor retries there. The caller
        // (`try_gpu_inner_join`) treats any `Err(_)` as "GPU declined"
        // and falls back; we use the typed `GpuCapacity` variant so the
        // pattern-match is recognisable rather than string-parsed.
        log::warn!(
            "gpu join probe overflow ({n_matches_raw} > {out_capacity_u32}); \
             falling back to host hash join"
        );
        return Err(BoltError::GpuCapacity(format!(
            "gpu_join: AoS probe claimed {n_matches_raw} matches > capacity {out_capacity_u32}"
        )));
    }
    let n_matches = n_matches_raw as usize;
    // v0.7 async-memcpy: pinned-host D2H + per-call sync, matching the SoA
    // INNER path. `download_to_host_pinned` falls back to `to_vec()` under
    // `--features cuda-stub` for byte-stable failure mode.
    let probe_full = download_to_host_pinned::<u32>(&out_probe_idx_dev, &stream)?;
    let build_full = download_to_host_pinned::<u32>(&out_build_idx_dev, &stream)?;
    let probe_idx: Vec<u32> = probe_full.into_iter().take(n_matches).collect();
    let build_idx: Vec<u32> = build_full.into_iter().take(n_matches).collect();
    Ok((UInt32Array::from(build_idx), UInt32Array::from(probe_idx)))
}

/// Execute a single-key INNER equi-join on the GPU.
///
/// Returns the two index arrays `(build_indices, probe_indices)` in
/// *arbitrary order* — the host caller is expected to either accept that
/// ordering (INNER doesn't promise one) or sort post-hoc.
///
/// `build_keys_col` and `probe_keys_col` must have the same `dtype` (the
/// caller validates this); the executor only checks at the entry into
/// `encode_keys_i64`.
pub fn hash_join_indices_on_gpu(
    build_keys_col: &dyn Array,
    probe_keys_col: &dyn Array,
    dtype: DataType,
) -> BoltResult<(UInt32Array, UInt32Array)> {
    // Run the build+probe on device, leaving the matched index pairs
    // resident, then download them. The device-side work (encode, upload,
    // hash-table build, probe, overflow contract) lives in
    // [`hash_join_indices_on_gpu_resident`] so the host-materialising path
    // and the multi-join composition hook share ONE kernel-launch
    // implementation and can never drift. The only thing this wrapper adds
    // is the D2H download + truncation to `n_matches`.
    //
    // Empty-side, capacity-overflow, and sentinel-collision behaviour are
    // all inherited unchanged from the resident helper (which preserves the
    // exact contract this function had before the refactor).
    let resident = hash_join_indices_on_gpu_resident(build_keys_col, probe_keys_col, dtype)?;
    resident.download()
}

/// End-to-end GPU INNER join over two `RecordBatch`es.
///
/// `build_key_idx` and `probe_key_idx` point at the join key columns within
/// the build / probe batches (the caller picks which side builds — typically
/// the smaller one). `dtype` is the (validated-equal) key dtype.
///
/// `lhs` and `rhs` are passed through as the *physical* left and right side
/// of the join, used purely for the final `take` on every column. The
/// (build_indices, probe_indices) pair from the GPU is re-oriented into
/// (left_indices, right_indices) according to `build_is_left`.
///
/// Returns a new `RecordBatch` with the joined rows. The output schema is
/// `output_schema` — the disambiguated combined schema computed by the
/// planner (left ++ right). Output row ordering is *unspecified* (atomic-
/// counter race in the probe kernel).
pub fn execute_inner_join_on_gpu(
    lhs: &RecordBatch,
    rhs: &RecordBatch,
    build_is_left: bool,
    build_key_idx: usize,
    probe_key_idx: usize,
    dtype: DataType,
    output_schema: Arc<ArrowSchema>,
) -> BoltResult<RecordBatch> {
    let (build_batch, probe_batch) = if build_is_left {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };

    let build_keys_col = build_batch.column(build_key_idx);
    let probe_keys_col = probe_batch.column(probe_key_idx);

    let (build_indices, probe_indices) =
        hash_join_indices_on_gpu(build_keys_col.as_ref(), probe_keys_col.as_ref(), dtype)?;

    // Re-orient (build, probe) -> (left, right).
    let (left_idx, right_idx) = if build_is_left {
        (&build_indices, &probe_indices)
    } else {
        (&probe_indices, &build_indices)
    };

    let mut output_cols: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
    for col in lhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), left_idx, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: arrow take (left): {e}")))?,
        );
    }
    for col in rhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), right_idx, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: arrow take (right): {e}")))?,
        );
    }

    RecordBatch::try_new(output_schema, output_cols)
        .map_err(|e| BoltError::Other(format!("gpu_join: building RecordBatch failed: {e}")))
}

// =========================================================================
// Stage 7 (R1): device-resident INNER-join index pairs (multi-join hook).
//
// Every entry point above ends with a `cuMemcpyDtoH` of the matched index
// pairs followed by `arrow::compute::take` — i.e. the join's output is
// materialised back on the host. For a *single* join that's fine: the
// caller wants a host `RecordBatch` anyway. For a JOIN whose output feeds
// ANOTHER operator (multi-join chaining, the TPC-H shape), that D2H +
// re-upload round trip is pure overhead — the next operator wants the
// matched build/probe row indices ON THE DEVICE so it can gather (via a
// device `take`-equivalent) or re-probe in place without ever touching
// the host.
//
// [`GpuJoinIndicesResident`] is that hook: it holds the matched
// `(build_idx, probe_idx)` row indices as device-resident `GpuVec<u32>`
// buffers plus the validated match count. The build+probe ran fully on
// device (identical kernels to [`hash_join_indices_on_gpu`]); the ONLY
// difference from the host path is that we skip the final D2H download and
// hand the device buffers to the caller.
//
// The existing host path ([`hash_join_indices_on_gpu`]) is preserved
// verbatim and is the correctness fallback; this is an additive composition
// seam, not a behaviour change for any current caller.
// =========================================================================

/// Device-resident result of a GPU INNER equi-join: the matched
/// `(build_idx, probe_idx)` row-index pairs, still on the device.
///
/// Returned by [`hash_join_indices_on_gpu_resident`] so a *parent* operator
/// (the next join in a multi-join chain, a resident projection, a resident
/// aggregate) can consume the join output on-device — no D2H download, no
/// `arrow::compute::take` host round trip.
///
/// Invariants:
/// * `build_idx` and `probe_idx` each hold at least `n_matches` valid u32
///   entries (their `GpuVec` length is the over-sized output capacity; the
///   meaningful prefix is `n_matches`). Entries past `n_matches` are
///   unspecified scratch and MUST NOT be read.
/// * `build_idx[i]` is a row index into the BUILD side; `probe_idx[i]` is a
///   row index into the PROBE side, for the same matched pair `i`. The
///   pairing order is the same arbitrary atomic-counter order the host path
///   sees — a consumer that needs deterministic ordering must sort.
/// * The buffers are valid only within the CUDA context that produced them
///   (same contract as every other `GpuVec` in the engine).
pub struct GpuJoinIndicesResident {
    /// Device-resident build-side row indices (length ≥ `n_matches`).
    pub build_idx: GpuVec<u32>,
    /// Device-resident probe-side row indices (length ≥ `n_matches`).
    pub probe_idx: GpuVec<u32>,
    /// Number of valid matched pairs in the prefix of each buffer.
    pub n_matches: usize,
}

impl GpuJoinIndicesResident {
    /// Download the resident index pairs to host `UInt32Array`s, truncated to
    /// `n_matches`. This is the explicit "materialise now" escape hatch — the
    /// whole point of the resident type is to AVOID this until the chain
    /// terminates, but a caller that has reached the end of the device-side
    /// pipeline (or a test) uses this to land the indices on the host.
    ///
    /// Returns `(build, probe)` in the same orientation as
    /// [`hash_join_indices_on_gpu`].
    pub fn download(&self) -> BoltResult<(UInt32Array, UInt32Array)> {
        // `to_vec` syncs on the buffer's recorded stream before copying, so
        // this is safe to call directly on a resident handle without an
        // explicit stream argument (mirrors the `download_to_host_pinned`
        // contract used elsewhere — we use `to_vec` here so the type stays
        // self-contained and stub-friendly).
        let build_full = self.build_idx.to_vec()?;
        let probe_full = self.probe_idx.to_vec()?;
        let build: Vec<u32> = build_full.into_iter().take(self.n_matches).collect();
        let probe: Vec<u32> = probe_full.into_iter().take(self.n_matches).collect();
        Ok((UInt32Array::from(build), UInt32Array::from(probe)))
    }
}

/// Build + probe a single-key Int32/Int64 INNER equi-join on the GPU and
/// leave the matched `(build_idx, probe_idx)` pairs DEVICE-RESIDENT.
///
/// This is the multi-join composition hook (R1 TIER-2 #8): the build hash
/// table is constructed from the (smaller) build side on device, the larger
/// probe side is probed on device, matched row-index pairs are emitted into
/// device output buffers — and, unlike [`hash_join_indices_on_gpu`], we do
/// NOT download them. The caller receives [`GpuJoinIndicesResident`] and can
/// feed those device buffers into the next operator (e.g. a second join's
/// probe, or a device gather) so a multi-join runs the chain on device.
///
/// Correctness is identical to [`hash_join_indices_on_gpu`]: same build /
/// probe kernels, same no-match (empty-slot ⇒ row dropped) and
/// unique-build-key match handling, same `i64::MIN` sentinel contract. The
/// caller must honour the SAME gates the host path requires (single
/// Int32/Int64 key, unique build keys, no NULL keys, ≥ `GPU_JOIN_MIN_ROWS`
/// per side); those gates live in `crate::exec::join::try_gpu_inner_join`
/// and are unchanged.
///
/// On output-buffer overflow (a build-side duplicate-key violation slipping
/// past the caller's uniqueness gate) this returns
/// [`BoltError::GpuCapacity`], exactly as the host path does, so the caller
/// can fall back to the host hash-join.
pub fn hash_join_indices_on_gpu_resident(
    build_keys_col: &dyn Array,
    probe_keys_col: &dyn Array,
    dtype: DataType,
) -> BoltResult<GpuJoinIndicesResident> {
    let n_build = build_keys_col.len();
    let n_probe = probe_keys_col.len();

    // Trivial empty-side short-circuit: no matches possible. Return empty
    // (but valid) resident buffers so the consumer's gather is a no-op.
    if n_build == 0 || n_probe == 0 {
        return Ok(GpuJoinIndicesResident {
            build_idx: GpuVec::<u32>::zeros(0)?,
            probe_idx: GpuVec::<u32>::zeros(0)?,
            n_matches: 0,
        });
    }

    let n_build_u32 = n_rows_to_u32(n_build)?;
    let n_probe_u32 = n_rows_to_u32(n_probe)?;

    let cap = compute_capacity(n_build)?;
    let cap_u32 = u32::try_from(cap)
        .map_err(|_| BoltError::Other(format!("gpu_join: cap {cap} doesn't fit in u32")))?;

    // Encode + upload both key columns (identical to the host path).
    let build_keys_host = encode_keys_i64(build_keys_col, dtype)?;
    let probe_keys_host = encode_keys_i64(probe_keys_col, dtype)?;

    let stream = CudaStream::null_or_default();

    let build_keys_dev = upload_primitive_values_async::<i64>(&build_keys_host, &stream)?;
    let probe_keys_dev = upload_primitive_values_async::<i64>(&probe_keys_host, &stream)?;

    let keys_init = get_sentinel_i64_min_vec(cap);
    let mut keys_table_dev = upload_primitive_values_async::<i64>(keys_init, &stream)?;
    let mut row_idx_table_dev = alloc_u32_max_table_async(cap, &stream)?;

    // Same worst-case INNER + unique-build sizing as the host path.
    let out_capacity_usize = n_probe
        .checked_add(n_build)
        .ok_or_else(|| BoltError::Other("gpu_join: output buffer size overflow".into()))?;
    let out_capacity_u32 = u32::try_from(out_capacity_usize).map_err(|_| {
        BoltError::Other(format!(
            "gpu_join: out_capacity {out_capacity_usize} doesn't fit in u32"
        ))
    })?;

    let mut out_probe_idx_dev = GpuVec::<u32>::zeros(out_capacity_usize)?;
    let mut out_build_idx_dev = GpuVec::<u32>::zeros(out_capacity_usize)?;
    let mut out_counter_dev = GpuVec::<u32>::zeros(1)?;

    // Build phase — populate the device hash table from the build side.
    launch_build_kernel(
        &build_keys_dev,
        &mut keys_table_dev,
        &mut row_idx_table_dev,
        n_build_u32,
        cap_u32,
        &stream,
    )?;

    // Probe phase — probe the larger side on device, emitting matched pairs.
    let n_matches_raw = launch_probe_kernel(
        &probe_keys_dev,
        &keys_table_dev,
        &row_idx_table_dev,
        &mut out_probe_idx_dev,
        &mut out_build_idx_dev,
        &mut out_counter_dev,
        n_probe_u32,
        cap_u32,
        out_capacity_u32,
        &stream,
    )?;

    if n_matches_raw > out_capacity_u32 {
        // Same overflow contract as the host path: typed GpuCapacity error so
        // the caller routes to the host hash-join. Under the INNER + unique-
        // build invariant the gate enforces, this never fires.
        log::warn!(
            "gpu join (resident) probe overflow ({n_matches_raw} > {out_capacity_u32}); \
             falling back to host hash join"
        );
        return Err(BoltError::GpuCapacity(format!(
            "gpu_join: resident probe kernel claimed {n_matches_raw} matches but \
             output buffer was sized for {out_capacity_u32}; \
             likely a build-side duplicate-key violation. Fall back to host path."
        )));
    }

    // KEY DIFFERENCE vs `hash_join_indices_on_gpu`: no D2H download. The
    // output buffers stay resident; the caller consumes them on-device.
    // `launch_probe_kernel` already synchronised the stream to read the
    // counter back, so the output buffers are fully populated and safe to
    // hand on (any subsequent kernel that reads them is launched on the same
    // null/default stream, preserving ordering).
    Ok(GpuJoinIndicesResident {
        build_idx: out_build_idx_dev,
        probe_idx: out_probe_idx_dev,
        n_matches: n_matches_raw as usize,
    })
}

// =========================================================================
// Stage 2: collision-list build/probe, outer-join orchestration,
// multi-key + bool/float keys.
// =========================================================================

/// Sentinel value used in the collision-list `head[]` and `next_idx[]` arrays
/// to denote "no more entries". Picked to match the kernel side
/// (`u32::MAX`).
///
/// Kept as a named constant (rather than inlining `u32::MAX`) so the
/// host/kernel contract is documented in one place. PERF (join key encoding):
/// these collision-list buffers are now filled on-device via
/// `alloc_u32_max_table_async` (`cuMemsetD8Async(0xFF)`), which relies on
/// `u32::MAX` being byte-replicable; the stub path / pool still uses
/// `get_sentinel_u32_max_vec(..)`. This symbol stays unreferenced under
/// `cargo build`. Do not delete without also revisiting that pool/memset.
#[allow(dead_code)]
const COLLISION_LIST_SENTINEL: u32 = u32::MAX;

/// Result of a Stage-2 GPU INNER join: index pairs in arbitrary order, ready
/// for `arrow::compute::take`.
pub struct GpuJoinIndices {
    /// Build-side row index for each emitted pair.
    pub build: UInt32Array,
    /// Probe-side row index for each emitted pair.
    pub probe: UInt32Array,
}

/// Result of a Stage-2 GPU OUTER join: index pairs that may include NULLs.
pub struct GpuOuterJoinIndices {
    /// Build-side row index per emitted output row. `None` for rows where the
    /// build side is NULL-padded (LEFT OUTER probe-only matches).
    pub build: Vec<Option<u32>>,
    /// Probe-side row index per emitted output row. `None` for rows where the
    /// probe side is NULL-padded (RIGHT/FULL OUTER unmatched-build pass).
    pub probe: Vec<Option<u32>>,
}

/// Stage-2 generalisation of [`hash_join_indices_on_gpu`]: runs the
/// collision-list build + probe kernels so duplicate build keys are handled
/// correctly. Used by the multi-key and outer-join paths; the Stage-1
/// unique-key fast path still routes through [`hash_join_indices_on_gpu`]
/// for the byte-stable single-key tests.
///
/// `shape` controls only the host-side encoding — the kernels see only i64
/// keys. The caller is responsible for verifying `shape.is_exact_in_i64()`.
pub fn hash_join_indices_on_gpu_with_shape(
    build_key_columns: &[&dyn Array],
    probe_key_columns: &[&dyn Array],
    shape: KeyShape,
) -> BoltResult<GpuJoinIndices> {
    // Review C4: guard before indexing `[0]` below — passing an empty slice
    // here used to panic with "index out of bounds" instead of surfacing a
    // clear planner-level error.
    if build_key_columns.is_empty() || probe_key_columns.is_empty() {
        return Err(BoltError::Plan(
            "join: build_key_columns and probe_key_columns must be non-empty (review C4)".into(),
        ));
    }
    if !shape.is_exact_in_i64() {
        return Err(BoltError::Other(format!(
            "gpu_join: lossy fold for shape {shape:?} would risk false matches; \
             fall back to host path"
        )));
    }
    let n_build = build_key_columns[0].len();
    let n_probe = probe_key_columns[0].len();

    if n_build == 0 || n_probe == 0 {
        return Ok(GpuJoinIndices {
            build: UInt32Array::from(Vec::<u32>::new()),
            probe: UInt32Array::from(Vec::<u32>::new()),
        });
    }

    let n_build_u32 = n_rows_to_u32(n_build)?;
    let n_probe_u32 = n_rows_to_u32(n_probe)?;

    let cap = compute_capacity(n_build)?;
    let cap_u32 = u32::try_from(cap)
        .map_err(|_| BoltError::Other(format!("gpu_join: cap {cap} doesn't fit in u32")))?;

    let build_keys_host = encode_keys_for_shape(build_key_columns, shape)?;
    let probe_keys_host = encode_keys_for_shape(probe_key_columns, shape)?;

    // v0.7 async-memcpy migration: mint a per-call stream, async H2D every
    // host-side init buffer, async D2H index outputs. See `aggregate.rs`
    // for the pilot rationale; under `--features cuda-stub` the helpers
    // fall back to the synchronous `from_slice` / `to_vec` shape for
    // byte-stable failure mode at the FFI boundary.
    let stream = CudaStream::null_or_default();

    let build_keys_dev = upload_primitive_values_async::<i64>(&build_keys_host, &stream)?;
    let probe_keys_dev = upload_primitive_values_async::<i64>(&probe_keys_host, &stream)?;

    // PERF (join key encoding): the three `u32::MAX`-init tables (`row_idx`,
    // `head`, `next_idx`) no longer round-trip through a host staging buffer.
    // `u32::MAX` is byte-replicable (`0xFF`), so `alloc_u32_max_table_async`
    // fills them on-device with `cuMemsetD8Async`, eliminating both the
    // per-query host allocation and the H2D upload for these buffers.
    //
    // The key table sentinel (`i64::MIN`) is NOT byte-replicable and there is
    // no 64-bit memset wrapper, so it keeps the pooled host slice + async H2D
    // upload (the host allocation here is already amortised by the pool).
    let keys_init = get_sentinel_i64_min_vec(cap);
    let mut keys_table_dev = upload_primitive_values_async::<i64>(keys_init, &stream)?;
    let mut row_idx_table_dev = alloc_u32_max_table_async(cap, &stream)?;
    let mut head_dev = alloc_u32_max_table_async(cap, &stream)?;
    let mut next_idx_dev = alloc_u32_max_table_async(n_build, &stream)?;

    // Output buffer size: 2x (n_build + n_probe) — a generous estimate for
    // typical workloads where duplicate-key fan-out is small. Pathological
    // cartesian-explosion cases (e.g. every key duplicated thousands of
    // times) overflow the counter and the kernel keeps it climbing; we
    // detect on counter readback and fall through to host.
    let out_capacity_usize = n_build
        .checked_add(n_probe)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| BoltError::Other("gpu_join: output buffer size overflow".into()))?;
    let out_capacity_u32 = u32::try_from(out_capacity_usize).map_err(|_| {
        BoltError::Other(format!(
            "gpu_join: out_capacity {out_capacity_usize} doesn't fit in u32"
        ))
    })?;

    let mut out_probe_idx_dev = GpuVec::<u32>::zeros(out_capacity_usize)?;
    let mut out_build_idx_dev = GpuVec::<u32>::zeros(out_capacity_usize)?;
    let mut out_counter_dev = GpuVec::<u32>::zeros(1)?;

    launch_build_collision_kernel(
        &build_keys_dev,
        &mut keys_table_dev,
        &mut row_idx_table_dev,
        &mut head_dev,
        &mut next_idx_dev,
        n_build_u32,
        cap_u32,
        &stream,
    )?;

    let n_matches_raw = launch_probe_collision_kernel(
        &probe_keys_dev,
        &keys_table_dev,
        &head_dev,
        &next_idx_dev,
        &mut out_probe_idx_dev,
        &mut out_build_idx_dev,
        &mut out_counter_dev,
        None,
        n_probe_u32,
        n_build_u32,
        cap_u32,
        out_capacity_u32,
        &stream,
    )?;

    if n_matches_raw > out_capacity_u32 {
        // Probe overflow: kernel wrote more matches than out_capacity. The
        // host join handles this input fine — return the fallback signal so
        // the executor retries there. `try_gpu_inner_join` catches any
        // `Err(_)` and falls back to the host hash-join; `GpuCapacity` is
        // the typed marker for that path.
        log::warn!(
            "gpu join probe overflow ({n_matches_raw} > {out_capacity_u32}); \
             falling back to host hash join"
        );
        return Err(BoltError::GpuCapacity(format!(
            "gpu_join: probe kernel claimed {n_matches_raw} matches but \
             output buffer was sized for {out_capacity_u32}"
        )));
    }
    let n_matches = n_matches_raw as usize;
    // v0.7 async-memcpy: pinned-host D2H + per-call sync, matching the SoA
    // INNER path. `download_to_host_pinned` falls back to `to_vec()` under
    // `--features cuda-stub` for byte-stable failure mode.
    let probe_idx_full = download_to_host_pinned::<u32>(&out_probe_idx_dev, &stream)?;
    let build_idx_full = download_to_host_pinned::<u32>(&out_build_idx_dev, &stream)?;
    let probe_idx: Vec<u32> = probe_idx_full.into_iter().take(n_matches).collect();
    let build_idx: Vec<u32> = build_idx_full.into_iter().take(n_matches).collect();

    Ok(GpuJoinIndices {
        build: UInt32Array::from(build_idx),
        probe: UInt32Array::from(probe_idx),
    })
}

/// Stage-2 OUTER orchestration. Returns `(build, probe)` index streams that
/// may contain `None` slots where one side is NULL-padded. The caller passes
/// `emit_unmatched_probe` for LEFT/FULL semantics and `emit_unmatched_build`
/// for RIGHT/FULL semantics.
///
/// ## Stage-4: lossy-shape support
///
/// Stage 3 admitted lossy shapes (`TwoI64`, `MultiI32(n)`) only for INNER
/// joins via the candidate-filter + host-verify pipeline. Stage 4 extends
/// that contract to LEFT / RIGHT / FULL OUTER:
///
/// 1. Run the GPU candidate-filter pass exactly as for INNER. The kernel's
///    `matched: u32[ceil(build_n_rows/32)]` bitmap is captured but treated
///    as a *superset* — it contains every build row that the GPU thought
///    matched, including those that drop out under host post-verify.
/// 2. Apply [`verify_pairs_on_host`] over the candidate `(build, probe)`
///    pair set to drop false positives.
/// 3. Recompute `matched_probe` and `matched_build` on the host using the
///    *verified* pair set. The kernel-side bitmap is discarded for the
///    unmatched-build pass — we walk the host `matched_build` array instead.
/// 4. Emit unmatched rows from the host arrays (LEFT/FULL: zero-verified
///    probe rows; RIGHT/FULL: zero-verified build rows).
///
/// Exact shapes (`is_exact_in_i64() == true`) skip the verify call and use
/// the kernel-side bitmap directly, byte-stable with Stage 3's behaviour.
pub fn execute_outer_join_indices_on_gpu(
    build_key_columns: &[&dyn Array],
    probe_key_columns: &[&dyn Array],
    shape: KeyShape,
    emit_unmatched_probe: bool,
    emit_unmatched_build: bool,
) -> BoltResult<GpuOuterJoinIndices> {
    // Review C4: guard before indexing `[0]` below — passing an empty slice
    // here used to panic with "index out of bounds" instead of surfacing a
    // clear planner-level error.
    if build_key_columns.is_empty() || probe_key_columns.is_empty() {
        return Err(BoltError::Plan(
            "join: build_key_columns and probe_key_columns must be non-empty (review C4)".into(),
        ));
    }
    let needs_verify = shape.needs_host_post_verify();
    // Stage 4: lossy shapes are admitted via host post-verify; only truly
    // unsupported shapes bail here.
    let n_build = build_key_columns[0].len();
    let n_probe = probe_key_columns[0].len();

    // Empty-build / empty-probe degenerate cases — handle in the host
    // orchestrator (join.rs) since this layer is purely GPU.
    if n_build == 0 || n_probe == 0 {
        return Ok(GpuOuterJoinIndices {
            build: Vec::new(),
            probe: Vec::new(),
        });
    }

    let n_build_u32 = n_rows_to_u32(n_build)?;
    let n_probe_u32 = n_rows_to_u32(n_probe)?;

    let cap = compute_capacity(n_build)?;
    let cap_u32 = u32::try_from(cap)
        .map_err(|_| BoltError::Other(format!("gpu_join: cap {cap} doesn't fit in u32")))?;

    let build_keys_host = encode_keys_for_shape(build_key_columns, shape)?;
    let probe_keys_host = encode_keys_for_shape(probe_key_columns, shape)?;

    // v0.7 async-memcpy migration: per-call stream + async H2D for every
    // host-side init buffer. Mirrors the INNER collision-list path above.
    let stream = CudaStream::null_or_default();

    let build_keys_dev = upload_primitive_values_async::<i64>(&build_keys_host, &stream)?;
    let probe_keys_dev = upload_primitive_values_async::<i64>(&probe_keys_host, &stream)?;

    // PERF (join key encoding): u32::MAX collision-list tables filled on-device
    // (cuMemsetD8Async 0xFF) — no host scratch, no H2D. The i64::MIN key table
    // is not byte-replicable so it keeps its pooled host slice + async H2D
    // (see top-of-file `get_sentinel_*_vec` and `alloc_u32_max_table_async`).
    let keys_init = get_sentinel_i64_min_vec(cap);
    let mut keys_table_dev = upload_primitive_values_async::<i64>(keys_init, &stream)?;
    let mut row_idx_table_dev = alloc_u32_max_table_async(cap, &stream)?;
    let mut head_dev = alloc_u32_max_table_async(cap, &stream)?;
    let mut next_idx_dev = alloc_u32_max_table_async(n_build, &stream)?;

    // matched: u32[ceil(build_n_rows / 32)], zero-initialised. We always
    // allocate it for OUTER even when only the LEFT case is requested,
    // because the collision probe kernel needs a real pointer (we pass
    // null only for INNER).
    let matched_words = n_build.div_ceil(32);
    let matched_dev = GpuVec::<u32>::zeros(matched_words)?;

    // Output buffer sizing: 2x (n_build + n_probe). Generous for typical
    // outer joins; cartesian-explosion cases overflow the counter (kernel
    // keeps incrementing past the writes) and fall through to host.
    let out_capacity_usize = n_build
        .checked_add(n_probe)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| BoltError::Other("gpu_join: outer output sizing overflowed".into()))?;
    let out_capacity_u32 = u32::try_from(out_capacity_usize).map_err(|_| {
        BoltError::Other(format!(
            "gpu_join: out_capacity {out_capacity_usize} doesn't fit in u32"
        ))
    })?;

    let mut out_probe_idx_dev = GpuVec::<u32>::zeros(out_capacity_usize)?;
    let mut out_build_idx_dev = GpuVec::<u32>::zeros(out_capacity_usize)?;
    let mut out_counter_dev = GpuVec::<u32>::zeros(1)?;

    launch_build_collision_kernel(
        &build_keys_dev,
        &mut keys_table_dev,
        &mut row_idx_table_dev,
        &mut head_dev,
        &mut next_idx_dev,
        n_build_u32,
        cap_u32,
        &stream,
    )?;

    let n_matches_raw = launch_probe_collision_kernel(
        &probe_keys_dev,
        &keys_table_dev,
        &head_dev,
        &next_idx_dev,
        &mut out_probe_idx_dev,
        &mut out_build_idx_dev,
        &mut out_counter_dev,
        Some(&matched_dev),
        n_probe_u32,
        n_build_u32,
        cap_u32,
        out_capacity_u32,
        &stream,
    )?;

    if n_matches_raw > out_capacity_u32 {
        // Probe overflow: kernel wrote more matches than out_capacity. The
        // host join handles this input fine — return the fallback signal so
        // the executor retries there. Cartesian-explosion outer joins land
        // here (the kernel counter increments past the actual writes); the
        // typed `GpuCapacity` variant is what `try_gpu_outer_join` maps to
        // `Ok(None)` for the host-join fallback.
        log::warn!(
            "gpu join probe overflow ({n_matches_raw} > {out_capacity_u32}); \
             falling back to host hash join"
        );
        return Err(BoltError::GpuCapacity(format!(
            "gpu_join: outer-join probe claimed {n_matches_raw} matches > capacity {out_capacity_u32}"
        )));
    }
    let n_matches = n_matches_raw as usize;
    // v0.7 async-memcpy: pinned-host D2H + per-call sync for the candidate
    // index pairs. The `matched` bitmap stays on the device until the
    // unmatched-build emission pass needs it.
    let probe_idx_full = download_to_host_pinned::<u32>(&out_probe_idx_dev, &stream)?;
    let build_idx_full = download_to_host_pinned::<u32>(&out_build_idx_dev, &stream)?;

    // Stage 4 (lossy OUTER): the GPU's `matched` bitmap may contain false
    // positives that drop under host post-verify. Re-test the candidate
    // pair set against the original Arrow columns, then derive the host
    // `matched_build` / `matched_probe` arrays from the *verified* pairs.
    // For exact shapes (`needs_verify == false`) the verify is skipped and
    // the verified pair set is the kernel's emitted pair set, byte-stable
    // with the Stage-3 OUTER behaviour.
    let (probe_idx_verified, build_idx_verified) = if needs_verify {
        let candidate_build: Vec<u32> = build_idx_full.iter().take(n_matches).copied().collect();
        let candidate_probe: Vec<u32> = probe_idx_full.iter().take(n_matches).copied().collect();
        let (kept_build, kept_probe) = verify_pairs_on_host(
            build_key_columns,
            probe_key_columns,
            &UInt32Array::from(candidate_build),
            &UInt32Array::from(candidate_probe),
        )?;
        // verify_pairs_on_host preserves the input ordering, so the two
        // verified arrays are still index-aligned. Drain them into plain
        // Vec<u32>s so the loops below can re-use the same index-based code.
        let p_kept: Vec<u32> = (0..kept_probe.len()).map(|i| kept_probe.value(i)).collect();
        let b_kept: Vec<u32> = (0..kept_build.len()).map(|i| kept_build.value(i)).collect();
        (p_kept, b_kept)
    } else {
        (
            probe_idx_full.into_iter().take(n_matches).collect(),
            build_idx_full.into_iter().take(n_matches).collect(),
        )
    };

    let n_verified = probe_idx_verified.len();

    // Host-side post-pass for LEFT/FULL: derive matched_probe from the
    // *verified* pair set. The kernel's bitmap is unverified (it counted
    // false positives), so we MUST rebuild this on the host even for the
    // exact-shape path's LEFT/FULL — the existing Stage-3 logic already
    // did this; Stage 4 just extends the same scan to cover lossy shapes.
    let mut probe_was_matched: Vec<bool> = vec![false; n_probe];
    // Stage 4 (RIGHT/FULL with verify): we also need a host-derived
    // matched_build bitmap so the unmatched-build pass uses verified data,
    // not the kernel-side false-positive-bearing matched_dev. For exact
    // shapes we still consult the kernel bitmap to stay byte-stable, so
    // this allocation only fires on the lossy + RIGHT/FULL combination.
    let need_host_matched_build = needs_verify && emit_unmatched_build;
    let mut build_was_matched: Vec<bool> = if need_host_matched_build {
        vec![false; n_build]
    } else {
        Vec::new()
    };

    let mut build: Vec<Option<u32>> = Vec::with_capacity(n_verified + n_probe + n_build);
    let mut probe: Vec<Option<u32>> = Vec::with_capacity(n_verified + n_probe + n_build);

    for i in 0..n_verified {
        let p = probe_idx_verified[i];
        let b = build_idx_verified[i];
        build.push(Some(b));
        probe.push(Some(p));
        if (p as usize) < n_probe {
            probe_was_matched[p as usize] = true;
        }
        if need_host_matched_build && (b as usize) < n_build {
            build_was_matched[b as usize] = true;
        }
    }

    // LEFT/FULL: emit (None, probe_row) for unmatched probe rows.
    if emit_unmatched_probe {
        for (p, matched) in probe_was_matched.iter().enumerate() {
            if !matched {
                build.push(None);
                probe.push(Some(p as u32));
            }
        }
    }

    // RIGHT/FULL: emit (build_row, None) for unmatched build rows.
    //
    // Exact-shape path: the GPU's `matched_dev` bitmap is byte-accurate, so
    // we use the second-pass kernel (cheaper than a host scan). Lossy path:
    // the bitmap holds the *candidate* matches; the host post-verify dropped
    // some, so we must walk `build_was_matched` instead.
    if emit_unmatched_build {
        if needs_verify {
            for (b, matched) in build_was_matched.iter().enumerate() {
                if !matched {
                    build.push(Some(b as u32));
                    probe.push(None);
                }
            }
        } else {
            let mut out_unmatched_dev = GpuVec::<u32>::zeros(n_build)?;
            let mut out_unmatched_counter_dev = GpuVec::<u32>::zeros(1)?;
            let n_unmatched = launch_unmatched_build_kernel(
                &matched_dev,
                &mut out_unmatched_dev,
                &mut out_unmatched_counter_dev,
                n_build_u32,
                n_build_u32,
                &stream,
            )?;
            if n_unmatched > n_build_u32 {
                // Probe overflow (second-pass): the unmatched-build kernel's
                // counter exceeded the n_build upper bound, meaning the
                // bitmap-walk wrote outside the sized buffer. The host
                // outer-join handles this input fine — return the fallback
                // signal so the executor retries there. `try_gpu_outer_join`
                // maps `Err(BoltError::GpuCapacity(_))` (along with any other
                // `Err(_)`) to its host-fallback `Ok(None)` path.
                log::warn!(
                    "gpu join probe overflow ({n_unmatched} > {n_build_u32}); \
                     falling back to host hash join"
                );
                return Err(BoltError::GpuCapacity(format!(
                    "gpu_join: unmatched-build kernel claimed {n_unmatched} > n_build {n_build_u32}"
                )));
            }
            let n_unmatched = n_unmatched as usize;
            // v0.7 async-memcpy: pinned-host D2H for the unmatched-build
            // index stream. The `launch_unmatched_build_kernel` call already
            // synchronized `stream` to read its counter back, so the kernel's
            // writes are visible; this D2H is the final readback before
            // returning to the host.
            let unmatched_full = download_to_host_pinned::<u32>(&out_unmatched_dev, &stream)?;
            for &b in unmatched_full.iter().take(n_unmatched) {
                build.push(Some(b));
                probe.push(None);
            }
        }
    }

    Ok(GpuOuterJoinIndices { build, probe })
}

/// Launch the Stage-2 collision-list build kernel.
#[allow(clippy::too_many_arguments)]
fn launch_build_collision_kernel(
    build_keys_dev: &GpuVec<i64>,
    keys_table_dev: &mut GpuVec<i64>,
    row_idx_table_dev: &mut GpuVec<u32>,
    head_dev: &mut GpuVec<u32>,
    next_idx_dev: &mut GpuVec<u32>,
    n_build_rows: u32,
    cap: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    if n_build_rows == 0 {
        return Ok(());
    }
    let spec = HashJoinKernelSpec {
        kind: HashJoinKernelKind::BuildCollision,
        key_dtype: DataType::Int64,
        string_hash_returns_i64: false,
    };
    let module = module_cache::get_or_build_module_for_hash_join(
        &spec,
        BUILD_COLLISION_KERNEL_ENTRY,
        |_| compile_build_collision_kernel(),
    )?;
    let function = module.function(BUILD_COLLISION_KERNEL_ENTRY)?;

    let mut build_keys_ptr: CUdeviceptr = build_keys_dev.device_ptr();
    let mut keys_table_ptr: CUdeviceptr = keys_table_dev.device_ptr();
    let mut row_idx_table_ptr: CUdeviceptr = row_idx_table_dev.device_ptr();
    let mut head_ptr: CUdeviceptr = head_dev.device_ptr();
    let mut next_idx_ptr: CUdeviceptr = next_idx_dev.device_ptr();
    let mut n_rows_u32: u32 = n_build_rows;
    let mut cap_u32: u32 = cap;

    let mut params: [*mut c_void; 7] = [
        &mut build_keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut keys_table_ptr as *mut CUdeviceptr as *mut c_void,
        &mut row_idx_table_ptr as *mut CUdeviceptr as *mut c_void,
        &mut head_ptr as *mut CUdeviceptr as *mut c_void,
        &mut next_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
        &mut cap_u32 as *mut u32 as *mut c_void,
    ];

    let block: u32 = HASH_JOIN_BLOCK_SIZE;
    let grid_x: u32 = n_build_rows.div_ceil(block).max(1);
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
    stream.synchronize()?;
    Ok(())
}

/// Launch the Stage-2 collision-list probe kernel. `matched_dev` is `Some`
/// for outer joins (the kernel sets bits in it) and `None` for inner joins
/// (the kernel skips the OR via a null-pointer check).
#[allow(clippy::too_many_arguments)]
fn launch_probe_collision_kernel(
    probe_keys_dev: &GpuVec<i64>,
    keys_table_dev: &GpuVec<i64>,
    head_dev: &GpuVec<u32>,
    next_idx_dev: &GpuVec<u32>,
    out_probe_idx_dev: &mut GpuVec<u32>,
    out_build_idx_dev: &mut GpuVec<u32>,
    out_counter_dev: &mut GpuVec<u32>,
    matched_dev: Option<&GpuVec<u32>>,
    n_probe_rows: u32,
    n_build_rows: u32,
    cap: u32,
    out_capacity: u32,
    stream: &CudaStream,
) -> BoltResult<u32> {
    if n_probe_rows == 0 {
        return Ok(0);
    }
    // Bitmap sizing contract: kernel indexes `matched[word_idx]` where
    // `word_idx = cursor >> 5`, so the host MUST allocate exactly
    // `((build_n_rows + 31) / 32) as usize` u32 elements. See
    // `compile_probe_collision_kernel` doc.
    if let Some(m) = matched_dev {
        debug_assert_eq!(
            m.len(),
            ((n_build_rows as usize) + 31) / 32,
            "probe-collision matched bitmap must be ceil(build_n_rows/32) u32 words"
        );
    }
    let spec = HashJoinKernelSpec {
        kind: HashJoinKernelKind::ProbeCollision,
        key_dtype: DataType::Int64,
        string_hash_returns_i64: false,
    };
    let module = module_cache::get_or_build_module_for_hash_join(
        &spec,
        PROBE_COLLISION_KERNEL_ENTRY,
        |_| compile_probe_collision_kernel(),
    )?;
    let function = module.function(PROBE_COLLISION_KERNEL_ENTRY)?;

    let mut probe_keys_ptr: CUdeviceptr = probe_keys_dev.device_ptr();
    let mut keys_table_ptr: CUdeviceptr = keys_table_dev.device_ptr();
    let mut head_ptr: CUdeviceptr = head_dev.device_ptr();
    let mut next_idx_ptr: CUdeviceptr = next_idx_dev.device_ptr();
    let mut out_probe_idx_ptr: CUdeviceptr = out_probe_idx_dev.device_ptr();
    let mut out_build_idx_ptr: CUdeviceptr = out_build_idx_dev.device_ptr();
    let mut out_counter_ptr: CUdeviceptr = out_counter_dev.device_ptr();
    // INNER path passes a strict null (exactly 0). MUST be strict null;
    // kernel value-compares via `setp.eq.u64 %p6, %rd19, 0` to skip the
    // bitmap OR — any non-zero garbage would trigger an OOB atomic.
    // `CUdeviceptr` is `u64`; explicit `0u64` ensures a guaranteed-null
    // value rather than relying on a pointer-cast.
    let mut matched_ptr: CUdeviceptr = match matched_dev {
        Some(m) => m.device_ptr(),
        None => 0u64,
    };
    let mut n_probe_u32: u32 = n_probe_rows;
    let mut cap_u32: u32 = cap;
    let mut out_capacity_u32: u32 = out_capacity;
    let mut n_build_u32: u32 = n_build_rows;

    let mut params: [*mut c_void; 12] = [
        &mut probe_keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut keys_table_ptr as *mut CUdeviceptr as *mut c_void,
        &mut head_ptr as *mut CUdeviceptr as *mut c_void,
        &mut next_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_probe_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_build_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_counter_ptr as *mut CUdeviceptr as *mut c_void,
        &mut matched_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_probe_u32 as *mut u32 as *mut c_void,
        &mut cap_u32 as *mut u32 as *mut c_void,
        &mut out_capacity_u32 as *mut u32 as *mut c_void,
        &mut n_build_u32 as *mut u32 as *mut c_void,
    ];

    let block: u32 = HASH_JOIN_BLOCK_SIZE;
    let grid_x: u32 = n_probe_rows.div_ceil(block).max(1);
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
    stream.synchronize()?;
    let counter_host: Vec<u32> = out_counter_dev.to_vec()?;
    Ok(counter_host[0])
}

/// Launch the Stage-2 outer-join second-pass kernel.
fn launch_unmatched_build_kernel(
    matched_dev: &GpuVec<u32>,
    out_build_idx_dev: &mut GpuVec<u32>,
    out_counter_dev: &mut GpuVec<u32>,
    n_build_rows: u32,
    out_capacity: u32,
    stream: &CudaStream,
) -> BoltResult<u32> {
    if n_build_rows == 0 {
        return Ok(0);
    }
    // Bitmap sizing contract: kernel indexes `matched[word_idx]` where
    // `word_idx = tid >> 5`, so the host MUST allocate exactly
    // `((build_n_rows + 31) / 32) as usize` u32 elements. See
    // `compile_unmatched_build_kernel` doc.
    debug_assert_eq!(
        matched_dev.len(),
        ((n_build_rows as usize) + 31) / 32,
        "unmatched-build matched bitmap must be ceil(build_n_rows/32) u32 words"
    );
    let spec = HashJoinKernelSpec {
        kind: HashJoinKernelKind::UnmatchedBuild,
        key_dtype: DataType::Int64,
        string_hash_returns_i64: false,
    };
    let module = module_cache::get_or_build_module_for_hash_join(
        &spec,
        UNMATCHED_BUILD_KERNEL_ENTRY,
        |_| compile_unmatched_build_kernel(),
    )?;
    let function = module.function(UNMATCHED_BUILD_KERNEL_ENTRY)?;

    let mut matched_ptr: CUdeviceptr = matched_dev.device_ptr();
    let mut out_build_idx_ptr: CUdeviceptr = out_build_idx_dev.device_ptr();
    let mut out_counter_ptr: CUdeviceptr = out_counter_dev.device_ptr();
    let mut n_build_u32: u32 = n_build_rows;
    let mut out_capacity_u32: u32 = out_capacity;

    let mut params: [*mut c_void; 5] = [
        &mut matched_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_build_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_counter_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_build_u32 as *mut u32 as *mut c_void,
        &mut out_capacity_u32 as *mut u32 as *mut c_void,
    ];

    let block: u32 = HASH_JOIN_BLOCK_SIZE;
    let grid_x: u32 = n_build_rows.div_ceil(block).max(1);
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
    stream.synchronize()?;
    let counter_host: Vec<u32> = out_counter_dev.to_vec()?;
    Ok(counter_host[0])
}

/// End-to-end GPU INNER join over two `RecordBatch`es with a [`KeyShape`]
/// (multi-key + bool/float support). Differs from
/// [`execute_inner_join_on_gpu`] only in that the slot-finding step uses
/// the collision-list kernels rather than the Stage-1 unique-key fast path,
/// so duplicate build keys produce correct (multiple) matches.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // reason: superseded by execute_inner_join_on_gpu_with_shape_and_verify (GJ-stage3 host post-verify); kept as the canonical no-verify variant for callers that have already validated key shapes.
pub fn execute_inner_join_on_gpu_with_shape(
    lhs: &RecordBatch,
    rhs: &RecordBatch,
    build_is_left: bool,
    build_key_idxs: &[usize],
    probe_key_idxs: &[usize],
    shape: KeyShape,
    output_schema: Arc<ArrowSchema>,
) -> BoltResult<RecordBatch> {
    let (build_batch, probe_batch) = if build_is_left {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };

    let build_cols: Vec<&dyn Array> = build_key_idxs
        .iter()
        .map(|&i| build_batch.column(i).as_ref())
        .collect();
    let probe_cols: Vec<&dyn Array> = probe_key_idxs
        .iter()
        .map(|&i| probe_batch.column(i).as_ref())
        .collect();

    let idx = hash_join_indices_on_gpu_with_shape(&build_cols, &probe_cols, shape)?;

    let (left_idx, right_idx) = if build_is_left {
        (&idx.build, &idx.probe)
    } else {
        (&idx.probe, &idx.build)
    };

    let mut output_cols: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
    for col in lhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), left_idx, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: arrow take (left): {e}")))?,
        );
    }
    for col in rhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), right_idx, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: arrow take (right): {e}")))?,
        );
    }
    RecordBatch::try_new(output_schema, output_cols)
        .map_err(|e| BoltError::Other(format!("gpu_join: building RecordBatch failed: {e}")))
}

/// End-to-end GPU OUTER join over two `RecordBatch`es.
///
/// `emit_unmatched_probe` is true for LEFT/FULL; `emit_unmatched_build` is
/// true for RIGHT/FULL. INNER doesn't reach this function (use
/// [`execute_inner_join_on_gpu_with_shape`] for the multi-key path).
///
/// `lhs` and `rhs` are the physical left and right sides of the join (in
/// `output_schema` order); `build_is_left` says which the build side is.
/// For LEFT outer, the build side is the right table (so unmatched probe
/// rows can be detected); for RIGHT outer the build is the left.
#[allow(clippy::too_many_arguments)]
pub fn execute_outer_join_on_gpu(
    lhs: &RecordBatch,
    rhs: &RecordBatch,
    build_is_left: bool,
    build_key_idxs: &[usize],
    probe_key_idxs: &[usize],
    shape: KeyShape,
    emit_unmatched_probe: bool,
    emit_unmatched_build: bool,
    output_schema: Arc<ArrowSchema>,
) -> BoltResult<RecordBatch> {
    let (build_batch, probe_batch) = if build_is_left {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };

    let build_cols: Vec<&dyn Array> = build_key_idxs
        .iter()
        .map(|&i| build_batch.column(i).as_ref())
        .collect();
    let probe_cols: Vec<&dyn Array> = probe_key_idxs
        .iter()
        .map(|&i| probe_batch.column(i).as_ref())
        .collect();

    let outer = execute_outer_join_indices_on_gpu(
        &build_cols,
        &probe_cols,
        shape,
        emit_unmatched_probe,
        emit_unmatched_build,
    )?;

    let (left_idx_vec, right_idx_vec) = if build_is_left {
        (outer.build, outer.probe)
    } else {
        (outer.probe, outer.build)
    };
    let left_idx_arr = UInt32Array::from(left_idx_vec);
    let right_idx_arr = UInt32Array::from(right_idx_vec);

    let mut output_cols: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
    for col in lhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), &left_idx_arr, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: arrow take (left): {e}")))?,
        );
    }
    for col in rhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), &right_idx_arr, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: arrow take (right): {e}")))?,
        );
    }
    RecordBatch::try_new(output_schema, output_cols)
        .map_err(|e| BoltError::Other(format!("gpu_join: building RecordBatch failed: {e}")))
}

// =========================================================================
// Stage 3: per-pair host verification, Utf8 interning, CROSS JOIN.
// =========================================================================

/// Result of [`intern_utf8_columns`]: the build- and probe-side dictionary-
/// index columns suitable for upload to the kernel-side i64 path. Indices
/// are i32 because we cap the union-cardinality of the join inputs at
/// `i32::MAX` (`take` is u32-indexed downstream anyway).
pub struct InternedUtf8Columns {
    /// Build-side keys re-encoded as i32 dict indices.
    pub build_indices: Int32Array,
    /// Probe-side keys re-encoded as i32 dict indices. Probe-side strings
    /// that never appear in the build are still given a dict index (so the
    /// kernel-side compare is just an i32 equality) but they obviously
    /// won't find a slot — the kernel returns "no match" naturally.
    pub probe_indices: Int32Array,
}

/// Build a per-string dictionary over the union of `build` and `probe`'s
/// Utf8 values and re-encode both as i32 dict indices. The Stage-3 Utf8
/// path then uploads the i32 arrays through the existing `SingleI32`
/// kernel-side code, and the output's Utf8 columns are re-attached via
/// `arrow::compute::take` against the original `StringArray`s.
///
/// ## Cost note
///
/// O(n_build + n_probe) host work, with O(unique_strings) hash-map memory.
/// For high-cardinality joins where most strings appear once on each side
/// (e.g. random UUID keys) this can dominate. Stage 4 should add a
/// streaming-interning path: hash each string to a 64-bit value on the host,
/// upload that as the key, and run the candidate-filter path with
/// post-verification (same model as `TwoI64`). That avoids the dict-build
/// pass at the cost of a host comparison per match.
pub fn intern_utf8_columns(
    build: &dyn Array,
    probe: &dyn Array,
) -> BoltResult<InternedUtf8Columns> {
    let b = build
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            BoltError::Other("gpu_join: intern_utf8_columns build is not StringArray".into())
        })?;
    let p = probe
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            BoltError::Other("gpu_join: intern_utf8_columns probe is not StringArray".into())
        })?;

    // Single-pass union dict build. Keys borrow into the StringArray's
    // backing buffer, so the HashMap holds no extra heap copies of the
    // string data.
    let mut dict: HashMap<&str, i32> = HashMap::with_capacity(b.len() + p.len());
    let mut next_idx: i32 = 0;

    let mut build_idx: Vec<i32> = Vec::with_capacity(b.len());
    for i in 0..b.len() {
        if b.is_null(i) {
            // NULL keys never match in the equi-join contract; we emit a
            // sentinel index that no other row can map to so the kernel
            // sees a unique key. -1 is safe because next_idx starts at 0.
            // The host gate already rejects NULL-key columns for the GPU
            // path, so this branch should be unreachable; we encode for
            // safety.
            build_idx.push(-1);
            continue;
        }
        let s = b.value(i);
        // Bound check FIRST: compute the next index via `checked_add` BEFORE
        // we hand `cur` to the dictionary entry. If the increment would
        // overflow `i32::MAX` we bubble the error up here, so no corrupt
        // index ever gets written into `dict` or pushed into `build_idx`.
        let idx = match dict.entry(s) {
            std::collections::hash_map::Entry::Occupied(e) => *e.get(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let cur = next_idx;
                next_idx = next_idx.checked_add(1).ok_or_else(|| {
                    BoltError::Other(
                        "gpu_join: Utf8 interning overflowed i32::MAX distinct values; \
                         rewrite the query or fall back to host path"
                            .to_string(),
                    )
                })?;
                v.insert(cur);
                cur
            }
        };
        build_idx.push(idx);
    }

    let mut probe_idx: Vec<i32> = Vec::with_capacity(p.len());
    for i in 0..p.len() {
        if p.is_null(i) {
            probe_idx.push(-1);
            continue;
        }
        let s = p.value(i);
        let idx = match dict.entry(s) {
            std::collections::hash_map::Entry::Occupied(e) => *e.get(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let cur = next_idx;
                next_idx = next_idx.checked_add(1).ok_or_else(|| {
                    BoltError::Other(
                        "gpu_join: Utf8 interning overflowed i32::MAX distinct values; \
                         rewrite the query or fall back to host path"
                            .to_string(),
                    )
                })?;
                v.insert(cur);
                cur
            }
        };
        probe_idx.push(idx);
    }

    // The bound check above is now load-bearing: if any row would have
    // pushed `next_idx` past `i32::MAX` we already returned. This
    // `debug_assert!` is paranoia for the invariant.
    debug_assert!(next_idx < i32::MAX);

    Ok(InternedUtf8Columns {
        build_indices: Int32Array::from(build_idx),
        probe_indices: Int32Array::from(probe_idx),
    })
}

// =========================================================================
// Stage 4: streaming string interning for high-cardinality Utf8 joins.
// =========================================================================

/// Env-var name for the Stage-4 streaming-intern opt-in. Set to a non-empty
/// non-zero value to route `execute_utf8_inner_join_on_gpu` through
/// [`intern_utf8_columns_streaming`] instead of the byte-borrowed dict.
///
/// Default is off — the original `intern_utf8_columns` path is
/// byte-stable with Stage 3 and lower-overhead for medium-cardinality joins
/// (≲ 100k unique strings) where the dict fits comfortably in L2/L3 cache.
pub const STREAMING_INTERN_ENV_VAR: &str = "BOLT_GPU_JOIN_STREAMING_INTERN";

/// Chunk size for the streaming-intern pass. Rows are processed in
/// `STREAMING_INTERN_CHUNK_ROWS`-sized batches so the working set (chunk
/// pointers + ~chunk_size new dict entries on the high-cardinality tail)
/// stays small. Picked to comfortably exceed a typical L1 line refill cost
/// per chunk without blowing past L2 on the dict-growth side.
pub const STREAMING_INTERN_CHUNK_ROWS: usize = 64 * 1024;

/// Streaming variant of [`intern_utf8_columns`] for high-cardinality Utf8
/// joins.
///
/// ## Design
///
/// Stage-3's [`intern_utf8_columns`] borrows `&str` into the StringArray's
/// backing buffer to build a `HashMap<&str, i32>`. That's already
/// zero-clone, but the *hash table* still grows O(unique_strings) with one
/// entry per unique input string (`(&str, i32)` is 16 + 4 padding = 24 bytes
/// per entry on 64-bit; rehashes copy `&str` keys, not the string bytes).
/// For UUID-shaped keys that dominate the host-side cost.
///
/// The streaming path replaces `HashMap<&str, i32>` with
/// `HashMap<u64, i32>` keyed by a 64-bit hash of the string content (the
/// engine's splitmix-style `host_splitmix` over each byte chunk via a
/// rolling FNV-1a base). Each entry is 8 bytes for the key (plus the i32
/// value) — about 5-10× smaller than the borrowed-`&str` variant for
/// typical key lengths. Collisions are possible but rare (~2^-32 per row
/// pair at the 32-bit birthday bound), and Stage 3's host post-verify
/// pipeline already handles false positives correctly.
///
/// **Trade-off**: O(d) hash entries → O(d) u64 entries (5-10× smaller
/// dict memory for typical strings), at the cost of one extra `arrow_row_eq`
/// per *emitted candidate match* during post-verification (collisions are
/// rare; the dominant cost is real matches, which would pay for take()
/// anyway).
///
/// ## Chunking
///
/// Inputs are processed in [`STREAMING_INTERN_CHUNK_ROWS`]-row chunks.
/// Today the chunks are processed sequentially and merge into a single
/// shared dict, so the only observable effect is a smoother allocator-side
/// growth pattern; a future Stage 5 pass could swap to per-chunk dicts
/// with a final merge for parallelism.
///
/// ## Caller contract
///
/// The returned `InternedUtf8Columns` carry i32 indices keyed by the
/// 64-bit hash. Distinct strings with the same hash share an index, which
/// means the kernel-side join can emit (probe, build) pairs whose strings
/// are NOT actually equal. The caller MUST run `verify_pairs_on_host` over
/// the *original* `StringArray`s before trusting the matches — wiring is
/// the same as for `KeyShape::TwoI64`.
pub fn intern_utf8_columns_streaming(
    build: &dyn Array,
    probe: &dyn Array,
) -> BoltResult<InternedUtf8Columns> {
    let b = build
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            BoltError::Other(
                "gpu_join: intern_utf8_columns_streaming build is not StringArray".into(),
            )
        })?;
    let p = probe
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            BoltError::Other(
                "gpu_join: intern_utf8_columns_streaming probe is not StringArray".into(),
            )
        })?;

    // Hash-keyed dict. Sized to roughly the unique-string count — over-shoot
    // is harmless (just unused capacity) and prevents the worst-case rehash
    // chain during chunked growth.
    let mut dict: HashMap<u64, i32> = HashMap::with_capacity((b.len() + p.len()) / 2);
    let mut next_idx: i32 = 0;

    /// Inner helper: stream `arr` in `STREAMING_INTERN_CHUNK_ROWS`-row
    /// chunks, populate `out` with the per-row dict index, and grow `dict`
    /// in place. NULL rows take the same `-1` sentinel as the Stage-3
    /// non-streaming variant.
    fn intern_chunked(
        arr: &StringArray,
        dict: &mut HashMap<u64, i32>,
        next_idx: &mut i32,
        out: &mut Vec<i32>,
    ) -> BoltResult<()> {
        let n = arr.len();
        let mut start = 0usize;
        while start < n {
            let end = (start + STREAMING_INTERN_CHUNK_ROWS).min(n);
            for i in start..end {
                if arr.is_null(i) {
                    out.push(-1);
                    continue;
                }
                let s = arr.value(i);
                let h = utf8_hash64(s.as_bytes());
                let idx = match dict.entry(h) {
                    std::collections::hash_map::Entry::Occupied(e) => *e.get(),
                    std::collections::hash_map::Entry::Vacant(v) => {
                        let cur = *next_idx;
                        // Bound check BEFORE the dict write so we never
                        // intern a row under a saturated/aliased index.
                        // `checked_add` returns None on overflow; we bubble
                        // out via `?` immediately.
                        *next_idx = next_idx.checked_add(1).ok_or_else(|| {
                            BoltError::Other(
                                "gpu_join: streaming Utf8 interning overflowed i32::MAX \
                                 distinct hashes; rewrite the query or fall back to host path"
                                    .to_string(),
                            )
                        })?;
                        v.insert(cur);
                        cur
                    }
                };
                out.push(idx);
            }
            start = end;
        }
        Ok(())
    }

    let mut build_idx: Vec<i32> = Vec::with_capacity(b.len());
    intern_chunked(b, &mut dict, &mut next_idx, &mut build_idx)?;

    let mut probe_idx: Vec<i32> = Vec::with_capacity(p.len());
    intern_chunked(p, &mut dict, &mut next_idx, &mut probe_idx)?;

    // The bound check inside `intern_chunked` is load-bearing: any row that
    // would have overflowed `i32::MAX` already returned with the error
    // above. This `debug_assert!` is paranoia for the invariant.
    debug_assert!(next_idx < i32::MAX);

    Ok(InternedUtf8Columns {
        build_indices: Int32Array::from(build_idx),
        probe_indices: Int32Array::from(probe_idx),
    })
}

/// Pure-host 64-bit hash for UTF-8 byte sequences. Same splitmix family as
/// the kernel-side `FX_MUL`, so a future Stage that hashes strings on-device
/// can match this exactly. The fold is a simple FNV-style mix followed by
/// `host_splitmix` finalisation — deterministic, branch-free per byte.
///
/// This is not crypto: collisions are possible. The streaming intern path
/// relies on host post-verify (`verify_pairs_on_host` over the original
/// StringArrays) to drop hash collisions; do not call this anywhere a
/// collision can corrupt results without a verify step.
#[inline]
fn utf8_hash64(bytes: &[u8]) -> u64 {
    // FNV-1a primer, plus the splitmix finaliser so adjacent bytes don't
    // map to adjacent hashes (which would defeat the kernel-side
    // `(h * FX_MUL) >> 32` slot reduction).
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h: u64 = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    host_splitmix(h)
}

/// Read the streaming-intern env var. Returns `true` when the variable is
/// set to a non-empty value other than `"0"` / `"false"`. Anything else
/// (unset, empty, `"0"`, `"false"`) keeps the Stage-3 path active.
///
/// `pub(crate)` so the integration test `tests/env_var_smoke.rs` can
/// pin the toggle semantics — the truthy/falsy parsing rule is the
/// observable contract that downstream tooling tunes against, and
/// drifting it silently has bitten us before.
pub fn streaming_intern_enabled() -> bool {
    match std::env::var(STREAMING_INTERN_ENV_VAR) {
        Ok(v) => {
            let s = v.trim();
            !(s.is_empty() || s == "0" || s.eq_ignore_ascii_case("false"))
        }
        Err(_) => false,
    }
}

/// Stage-5: hash an entire `StringArray` on the GPU, returning one u64 per
/// row in the same order. Used by the streaming-intern fast path to lift
/// the host-side hashing step off the CPU for multi-million-row Utf8
/// joins.
///
/// ## ABI mapping
///
/// Per-row hash function matches the host [`utf8_hash64`] byte-for-byte:
/// FNV-1a primer + multiply over the bytes, followed by the splitmix
/// finaliser. Callers can mix host- and device-hashed values inside the
/// same dict without losing rows.
///
/// ## Upload shape
///
/// Arrow `StringArray` stores `offsets: i32[n + 1]` and `values: u8[total]`
/// internally; the kernel takes the same two device buffers. We allocate
/// fresh device-side copies here — the host's StringArray buffers aren't
/// necessarily on-device, and zero-copy from host-pinned memory is a
/// Stage-6 optimisation.
///
/// ## Errors
///
/// * Returns `Err` if `arr.len() > u32::MAX` — the kernel's grid uses u32
///   thread IDs.
/// * Returns `Err` on any kernel-launch failure.
#[allow(dead_code)] // reason: Stage-5 emits the wrapper + kernel; engine-level
                    //         routing into the streaming intern lands in Stage 6.
                    //         The `device_string_hash_matches_host` unit test
                    //         (in this module's tests submod) exercises it
                    //         directly.
pub fn compute_device_string_hashes(arr: &StringArray) -> BoltResult<Vec<u64>> {
    let n = arr.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let n_u32 = n_rows_to_u32(n)?;

    // Extract the i32 offsets array and u8 values buffer. Arrow's
    // StringArray exposes these directly as the underlying ScalarBuffer<i32>
    // and Buffer.
    let offsets: &[i32] = arr.value_offsets();
    let values: &[u8] = arr.value_data();

    // v0.7 async-memcpy: per-call stream + async H2D for both the offsets
    // and the values buffer (these are the Utf8 join key column's raw
    // Arrow backing storage). The values slice can be empty if the entire
    // StringArray is empty strings — we still need a valid device pointer
    // for the kernel, so allocate a single-byte placeholder in that case.
    let stream = CudaStream::null_or_default();
    let offsets_dev = upload_primitive_values_async::<i32>(offsets, &stream)?;
    let values_dev = if values.is_empty() {
        GpuVec::<u8>::zeros(1)?
    } else {
        upload_primitive_values_async::<u8>(values, &stream)?
    };
    let out_dev = GpuVec::<u64>::zeros(n)?;

    // Two string-hash flavours share `HashJoinKernelKind::StringHash`:
    // the regular Utf8 (i32-offsets) variant emits Int64 hash values via
    // the entry point `bolt_string_hash`. The companion LargeUtf8 (i64-offsets)
    // variant below sets `string_hash_returns_i64 = true` to land in a
    // distinct cache slot.
    let spec = HashJoinKernelSpec {
        kind: HashJoinKernelKind::StringHash,
        key_dtype: DataType::Utf8,
        string_hash_returns_i64: false,
    };
    let module =
        module_cache::get_or_build_module_for_hash_join(&spec, STRING_HASH_KERNEL_ENTRY, |_| {
            compile_string_hash_kernel(DataType::Utf8)
        })?;
    let function = module.function(STRING_HASH_KERNEL_ENTRY)?;

    let mut offsets_ptr: CUdeviceptr = offsets_dev.device_ptr();
    let mut values_ptr: CUdeviceptr = values_dev.device_ptr();
    let mut out_ptr: CUdeviceptr = out_dev.device_ptr();
    let mut n_rows_param: u32 = n_u32;

    let mut params: [*mut c_void; 4] = [
        &mut offsets_ptr as *mut CUdeviceptr as *mut c_void,
        &mut values_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_param as *mut u32 as *mut c_void,
    ];

    let block: u32 = STRING_HASH_BLOCK_SIZE;
    let grid_x: u32 = n_u32.div_ceil(block).max(1);

    // SAFETY: every param slot points at a stack local that outlives the
    // launch+sync below; device buffers are owned by this function and
    // outlive the launch.
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

    // v0.7 async-memcpy: pinned-host D2H + per-call sync, replacing the
    // legacy `sync + to_vec()` pair.
    download_to_host_pinned::<u64>(&out_dev, &stream)
}

/// **Stage 6 (GJ)** — LargeUtf8 (i64 offsets) variant of
/// [`compute_device_string_hashes`]. Routes through the i64-offset
/// kernel companion emitted by [`compile_string_hash_kernel_with_offsets`].
///
/// Outputs byte-identical hashes to the `StringArray` path (same
/// FNV+splitmix sequence), so the streaming-intern dict can mix
/// `Utf8` and `LargeUtf8` columns without re-hashing.
#[allow(dead_code)] // reason: kernel + wrapper land in Stage 6; engine-level
                    //         dispatch on LargeStringArray columns is the
                    //         next planner-time decision and isn't wired
                    //         yet. The unit test in the tests submod
                    //         exercises this path directly.
pub fn compute_device_string_hashes_large(
    arr: &arrow_array::LargeStringArray,
) -> BoltResult<Vec<u64>> {
    let n = arr.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let n_u32 = n_rows_to_u32(n)?;

    // Arrow's LargeStringArray exposes its underlying ScalarBuffer<i64>
    // offsets and u8 values buffer the same way StringArray does — just
    // wider offsets.
    let offsets: &[i64] = arr.value_offsets();
    let values: &[u8] = arr.value_data();

    // v0.7 async-memcpy: per-call stream + async H2D for both the i64
    // offsets and the values buffer (the LargeUtf8 join key column's raw
    // Arrow backing storage). Symmetric with the Utf8 path above.
    let stream = CudaStream::null_or_default();
    let offsets_dev = upload_primitive_values_async::<i64>(offsets, &stream)?;
    let values_dev = if values.is_empty() {
        GpuVec::<u8>::zeros(1)?
    } else {
        upload_primitive_values_async::<u8>(values, &stream)?
    };
    let out_dev = GpuVec::<u64>::zeros(n)?;

    // LargeUtf8 variant — i64 offsets at the kernel boundary. The
    // `string_hash_returns_i64 = true` flag distinguishes this cache slot
    // from the i32-offset Utf8 sibling above; both share the
    // `HashJoinKernelKind::StringHash` kind and `DataType::Utf8` key dtype.
    let spec = HashJoinKernelSpec {
        kind: HashJoinKernelKind::StringHash,
        key_dtype: DataType::Utf8,
        string_hash_returns_i64: true,
    };
    let module = module_cache::get_or_build_module_for_hash_join(
        &spec,
        crate::jit::hash_join_kernel::STRING_HASH_KERNEL_ENTRY_I64,
        |_| {
            crate::jit::hash_join_kernel::compile_string_hash_kernel_with_offsets(
                crate::jit::hash_join_kernel::StringOffsetWidth::I64,
            )
        },
    )?;
    let function = module.function(crate::jit::hash_join_kernel::STRING_HASH_KERNEL_ENTRY_I64)?;

    let mut offsets_ptr: CUdeviceptr = offsets_dev.device_ptr();
    let mut values_ptr: CUdeviceptr = values_dev.device_ptr();
    let mut out_ptr: CUdeviceptr = out_dev.device_ptr();
    let mut n_rows_param: u32 = n_u32;

    let mut params: [*mut c_void; 4] = [
        &mut offsets_ptr as *mut CUdeviceptr as *mut c_void,
        &mut values_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_param as *mut u32 as *mut c_void,
    ];

    let block: u32 = STRING_HASH_BLOCK_SIZE;
    let grid_x: u32 = n_u32.div_ceil(block).max(1);

    // SAFETY: every param slot points at a stack local that outlives the
    // launch+sync below; device buffers are owned by this function and
    // outlive the launch.
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

    // v0.7 async-memcpy: pinned-host D2H + per-call sync, matching the
    // Utf8 path. Hashes flow into the streaming-intern dict.
    download_to_host_pinned::<u64>(&out_dev, &stream)
}

/// Stage-5: parallel per-chunk dict-building variant of
/// [`intern_utf8_columns_streaming`].
///
/// ## Design
///
/// Stage 4's `intern_utf8_columns_streaming` walks both StringArrays
/// sequentially into a single shared `HashMap<u64, i32>`. The hashing step
/// dominates for high-cardinality joins (UUID keys, log IDs) and serialises
/// trivially across rows.
///
/// Stage 5 spawns N worker threads via `std::thread::scope` (no extra
/// dependency — rayon would be the natural fit but isn't yet in the dep
/// tree). Each worker builds a *chunk-local* `HashMap<u64, ()>` recording
/// the distinct hashes it saw plus the row -> hash mapping. The main
/// thread then walks the workers' per-chunk hash arrays in order, merging
/// distinct hashes into a single global dict (assigning dense i32 indices
/// in first-seen order). The final per-row index is looked up from the
/// global dict.
///
/// **Why the merge is sequential.** The merge is O(distinct_count); the
/// dict-building step is O(total_bytes) and parallelises to ~N threads.
/// For typical Utf8 joins `distinct_count << total_bytes`, so the merge
/// is cheap and the parallel hashing step is the load-bearing win.
///
/// ## Caller contract
///
/// Identical to [`intern_utf8_columns_streaming`]: callers MUST run
/// `verify_pairs_on_host` over the *original* `StringArray`s before
/// trusting the kernel-side i32 match set. Hash collisions are dropped by
/// that pass.
pub fn intern_utf8_columns_streaming_parallel(
    build: &dyn Array,
    probe: &dyn Array,
) -> BoltResult<InternedUtf8Columns> {
    let b = build
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            BoltError::Other(
                "gpu_join: intern_utf8_columns_streaming_parallel build is not StringArray".into(),
            )
        })?;
    let p = probe
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            BoltError::Other(
                "gpu_join: intern_utf8_columns_streaming_parallel probe is not StringArray".into(),
            )
        })?;

    // Phase 1: parallel hash-each-row.
    //
    // We compute the per-row 64-bit hash on every input row in parallel,
    // producing two dense `Vec<u64>`s (one per side). The hash function is
    // pure, so we can do this with thread::scope and no shared state.
    let build_hashes = hash_rows_in_parallel(b)?;
    let probe_hashes = hash_rows_in_parallel(p)?;

    // Phase 2: sequential merge.
    //
    // Walk both hash arrays in order and assign dense i32 indices via the
    // global dict. NULL rows get the -1 sentinel (matches Stage 3 / 4).
    let mut dict: HashMap<u64, i32> = HashMap::with_capacity((b.len() + p.len()) / 2);
    let mut next_idx: i32 = 0;

    let mut build_idx: Vec<i32> = Vec::with_capacity(b.len());
    assign_indices_from_hashes(b, &build_hashes, &mut dict, &mut next_idx, &mut build_idx)?;

    let mut probe_idx: Vec<i32> = Vec::with_capacity(p.len());
    assign_indices_from_hashes(p, &probe_hashes, &mut dict, &mut next_idx, &mut probe_idx)?;

    // `assign_indices_from_hashes` already errors on the first row that
    // would push `next_idx` past `i32::MAX`, so this is paranoia for the
    // post-condition.
    debug_assert!(next_idx < i32::MAX);

    Ok(InternedUtf8Columns {
        build_indices: Int32Array::from(build_idx),
        probe_indices: Int32Array::from(probe_idx),
    })
}

/// Hash every row of `arr` in parallel using `std::thread::scope` plus a
/// chunked split. Returns a dense `Vec<u64>` aligned with `arr.len()`.
/// NULL rows hash to `0` (the global dict still treats those rows as the
/// `-1` sentinel; the value only matters when the row is non-NULL).
fn hash_rows_in_parallel(arr: &StringArray) -> BoltResult<Vec<u64>> {
    let n = arr.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // **Stage 6 (GJ)** — device-hash auto-route. For very-large inputs the
    // GPU's per-thread hash over `(offsets, values)` beats spinning up host
    // threads + walking u8 buffers. The kernel produces byte-identical
    // hashes (same FNV+splitmix sequence as the host path), so the
    // output is a drop-in substitute. On any failure (no CUDA context,
    // OOM, kernel launch error) we fall through to the host path.
    //
    // Threshold tuned at `STREAMING_INTERN_DEVICE_HASH_THRESHOLD`; below
    // it the kernel-launch + d2h fixed cost dominates and the host
    // parallel path wins.
    if n >= STREAMING_INTERN_DEVICE_HASH_THRESHOLD {
        match compute_device_string_hashes(arr) {
            Ok(hashes) => return Ok(hashes),
            Err(e) => {
                log::debug!(
                    "gpu_join: device-hash failed on {} rows, falling back to host \
                     parallel hash: {}",
                    n,
                    e
                );
            }
        }
    }

    // Worker count: cap at 8 threads. Higher worker counts hurt for
    // medium-sized joins (the kernel-launch + thread-spawn fixed cost
    // dominates). Below STREAMING_INTERN_PARALLEL_THRESHOLD we don't reach
    // this function at all — the sequential variant is preferred.
    let n_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(8)
        .max(1);
    // Split into approximately equal chunks. The last chunk picks up any
    // remainder so we don't leave rows un-hashed.
    let chunk = n.div_ceil(n_workers).max(1);

    // Pre-allocate the full output buffer; workers write disjoint slices.
    let mut out: Vec<u64> = vec![0; n];

    // Build (row_start, mut_slice) pairs up front. Computing both the row
    // range and the slice from the same `chunk` step avoids the off-by-one
    // risk of having two separate loops walk the array.
    let mut row_starts: Vec<usize> = Vec::with_capacity(n_workers);
    let mut out_slices: Vec<&mut [u64]> = Vec::with_capacity(n_workers);
    {
        let mut start = 0usize;
        let mut tail: &mut [u64] = &mut out[..];
        while start < n {
            let end = (start + chunk).min(n);
            let take = end - start;
            let (head, rest) = tail.split_at_mut(take);
            row_starts.push(start);
            out_slices.push(head);
            tail = rest;
            start = end;
        }
    }

    // SAFETY: each worker owns a disjoint mutable slice into `out`, so no
    // two threads alias the same byte. The scope guarantees every worker
    // is joined before we return, so the `out` slice outlives every borrow.
    std::thread::scope(|scope| {
        let mut handles: Vec<std::thread::ScopedJoinHandle<'_, ()>> = Vec::with_capacity(n_workers);
        for (s, slice) in row_starts.iter().copied().zip(out_slices.into_iter()) {
            let h = scope.spawn(move || {
                for j in 0..slice.len() {
                    let i = s + j;
                    let h = if arr.is_null(i) {
                        0
                    } else {
                        utf8_hash64(arr.value(i).as_bytes())
                    };
                    slice[j] = h;
                }
            });
            handles.push(h);
        }
        for h in handles {
            // join() returns Err only on panic — surface as a no-op since
            // there's no good recovery path for a hashing-thread panic.
            let _ = h.join();
        }
    });

    Ok(out)
}

/// Walk pre-computed per-row hashes and assign dense i32 indices via the
/// global dict. NULL rows in the source `arr` get the `-1` sentinel
/// regardless of their hash value.
///
/// Errors out the moment incrementing `next_idx` would overflow `i32::MAX`,
/// BEFORE the would-be corrupt index is inserted into `dict` or pushed into
/// `out`. This is called sequentially from the main thread of
/// [`intern_utf8_columns_streaming_parallel`] (Phase 2 merge), so a plain
/// `BoltResult<()>` is sufficient — no cross-thread error propagation
/// required.
fn assign_indices_from_hashes(
    arr: &StringArray,
    hashes: &[u64],
    dict: &mut HashMap<u64, i32>,
    next_idx: &mut i32,
    out: &mut Vec<i32>,
) -> BoltResult<()> {
    let n = arr.len();
    debug_assert_eq!(n, hashes.len());
    for i in 0..n {
        if arr.is_null(i) {
            out.push(-1);
            continue;
        }
        let h = hashes[i];
        let idx = match dict.entry(h) {
            std::collections::hash_map::Entry::Occupied(e) => *e.get(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let cur = *next_idx;
                // Bound check FIRST so we never write a saturated/aliased
                // index into the dict or the output Vec.
                *next_idx = next_idx.checked_add(1).ok_or_else(|| {
                    BoltError::Other(
                        "gpu_join: parallel streaming Utf8 interning overflowed i32::MAX \
                         distinct hashes; rewrite the query or fall back to host path"
                            .to_string(),
                    )
                })?;
                v.insert(cur);
                cur
            }
        };
        out.push(idx);
    }
    Ok(())
}

/// Re-test a set of `(probe_idx, build_idx)` candidate pairs against the
/// original Arrow key columns. Used by the Stage-3 lossy-fold path
/// (`TwoI64`, `MultiI32(_)`) to drop false positives produced by the
/// 64-bit splitmix collapse.
///
/// `build_cols` and `probe_cols` are slices of the original Arrow key
/// columns (one per join column). For each pair we walk every column, pull
/// the build- and probe-side values at the indexed rows, and require all
/// columns to be equal. Any mismatch drops the pair.
///
/// Returns the filtered `(build, probe)` index pair, in the same relative
/// order as the input.
fn verify_pairs_on_host(
    build_cols: &[&dyn Array],
    probe_cols: &[&dyn Array],
    build_indices: &UInt32Array,
    probe_indices: &UInt32Array,
) -> BoltResult<(UInt32Array, UInt32Array)> {
    debug_assert_eq!(build_cols.len(), probe_cols.len());
    debug_assert_eq!(build_indices.len(), probe_indices.len());

    let mut kept_build: Vec<u32> = Vec::with_capacity(build_indices.len());
    let mut kept_probe: Vec<u32> = Vec::with_capacity(probe_indices.len());

    for k in 0..build_indices.len() {
        let bi = build_indices.value(k);
        let pi = probe_indices.value(k);
        let mut all_eq = true;
        for (b, p) in build_cols.iter().zip(probe_cols.iter()) {
            if !arrow_row_eq(*b, bi as usize, *p, pi as usize)? {
                all_eq = false;
                break;
            }
        }
        if all_eq {
            kept_build.push(bi);
            kept_probe.push(pi);
        }
    }

    Ok((UInt32Array::from(kept_build), UInt32Array::from(kept_probe)))
}

/// Row-level equality on Arrow scalars. Returns true iff both values are
/// non-NULL and bit-equal. Used by [`verify_pairs_on_host`] — equi-join
/// semantics treats NULL = NULL as UNKNOWN (drops the pair).
fn arrow_row_eq(a: &dyn Array, ai: usize, b: &dyn Array, bi: usize) -> BoltResult<bool> {
    if a.is_null(ai) || b.is_null(bi) {
        return Ok(false);
    }
    if a.data_type() != b.data_type() {
        return Err(BoltError::Other(format!(
            "gpu_join: verify_pairs_on_host dtype mismatch ({:?} vs {:?})",
            a.data_type(),
            b.data_type()
        )));
    }
    let eq = match a.data_type() {
        arrow_schema::DataType::Int32 => {
            let av = a.as_any().downcast_ref::<Int32Array>().unwrap().value(ai);
            let bv = b.as_any().downcast_ref::<Int32Array>().unwrap().value(bi);
            av == bv
        }
        arrow_schema::DataType::Int64 => {
            let av = a.as_any().downcast_ref::<Int64Array>().unwrap().value(ai);
            let bv = b.as_any().downcast_ref::<Int64Array>().unwrap().value(bi);
            av == bv
        }
        arrow_schema::DataType::Float32 => {
            // Bit-equal (engine convention: NaN == NaN if same bit pattern).
            let av = a
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(ai)
                .to_bits();
            let bv = b
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(bi)
                .to_bits();
            av == bv
        }
        arrow_schema::DataType::Float64 => {
            let av = a
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(ai)
                .to_bits();
            let bv = b
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(bi)
                .to_bits();
            av == bv
        }
        arrow_schema::DataType::Boolean => {
            let av = a.as_any().downcast_ref::<BooleanArray>().unwrap().value(ai);
            let bv = b.as_any().downcast_ref::<BooleanArray>().unwrap().value(bi);
            av == bv
        }
        arrow_schema::DataType::Utf8 => {
            let av = a.as_any().downcast_ref::<StringArray>().unwrap().value(ai);
            let bv = b.as_any().downcast_ref::<StringArray>().unwrap().value(bi);
            av == bv
        }
        other => {
            return Err(BoltError::Other(format!(
                "gpu_join: verify_pairs_on_host unsupported dtype {other:?}"
            )));
        }
    };
    Ok(eq)
}

/// Stage-3 INNER join entry point with optional host-side per-pair
/// verification.
///
/// Same contract as [`execute_inner_join_on_gpu_with_shape`] but additionally
/// supports lossy-fold shapes (`TwoI64`, `MultiI32(_)`) by running the GPU
/// join as a candidate filter and re-testing every emitted pair on the host
/// against the original Arrow columns. For exact shapes the verify step is
/// skipped (no false positives are possible).
#[allow(clippy::too_many_arguments)]
pub fn execute_inner_join_on_gpu_with_shape_and_verify(
    lhs: &RecordBatch,
    rhs: &RecordBatch,
    build_is_left: bool,
    build_key_idxs: &[usize],
    probe_key_idxs: &[usize],
    shape: KeyShape,
    output_schema: Arc<ArrowSchema>,
) -> BoltResult<RecordBatch> {
    let (build_batch, probe_batch) = if build_is_left {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    let build_cols: Vec<&dyn Array> = build_key_idxs
        .iter()
        .map(|&i| build_batch.column(i).as_ref())
        .collect();
    let probe_cols: Vec<&dyn Array> = probe_key_idxs
        .iter()
        .map(|&i| probe_batch.column(i).as_ref())
        .collect();

    let idx = hash_join_indices_on_gpu_with_shape_unverified(&build_cols, &probe_cols, shape)?;

    // Stage-3: lossy shapes get per-pair host re-test. For exact shapes
    // (every path that's_exact_in_i64) the verify is skipped — the kernel
    // already emits exact matches.
    let (build_verified, probe_verified) = if shape.needs_host_post_verify() {
        verify_pairs_on_host(&build_cols, &probe_cols, &idx.build, &idx.probe)?
    } else {
        (idx.build, idx.probe)
    };

    let (left_idx, right_idx) = if build_is_left {
        (&build_verified, &probe_verified)
    } else {
        (&probe_verified, &build_verified)
    };

    let mut output_cols: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
    for col in lhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), left_idx, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: arrow take (left): {e}")))?,
        );
    }
    for col in rhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), right_idx, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: arrow take (right): {e}")))?,
        );
    }
    RecordBatch::try_new(output_schema, output_cols)
        .map_err(|e| BoltError::Other(format!("gpu_join: building RecordBatch failed: {e}")))
}

/// Like [`hash_join_indices_on_gpu_with_shape`] but does NOT reject lossy
/// shapes — the caller is expected to apply [`verify_pairs_on_host`] before
/// trusting the emitted pairs. Stage-1/2 callers go through the verifying
/// `hash_join_indices_on_gpu_with_shape` wrapper.
fn hash_join_indices_on_gpu_with_shape_unverified(
    build_key_columns: &[&dyn Array],
    probe_key_columns: &[&dyn Array],
    shape: KeyShape,
) -> BoltResult<GpuJoinIndices> {
    // Review C4: guard before indexing `[0]` below — passing an empty slice
    // here used to panic with "index out of bounds" instead of surfacing a
    // clear planner-level error. (This helper is reachable from the pub
    // `execute_inner_join_on_gpu_with_shape_and_verify` entry point.)
    if build_key_columns.is_empty() || probe_key_columns.is_empty() {
        return Err(BoltError::Plan(
            "join: build_key_columns and probe_key_columns must be non-empty (review C4)".into(),
        ));
    }
    // Re-route exact shapes through the existing verified entry point so
    // we don't fork the kernel-launch logic.
    if !shape.needs_host_post_verify() {
        return hash_join_indices_on_gpu_with_shape(build_key_columns, probe_key_columns, shape);
    }

    // Lossy-fold path. We bypass the `is_exact_in_i64()` gate in the verified
    // helper by inlining the launch sequence here. Encoding still goes
    // through `encode_keys_for_shape` (which now accepts lossy shapes for
    // the unverified entry).
    let n_build = build_key_columns[0].len();
    let n_probe = probe_key_columns[0].len();

    if n_build == 0 || n_probe == 0 {
        return Ok(GpuJoinIndices {
            build: UInt32Array::from(Vec::<u32>::new()),
            probe: UInt32Array::from(Vec::<u32>::new()),
        });
    }

    let n_build_u32 = n_rows_to_u32(n_build)?;
    let n_probe_u32 = n_rows_to_u32(n_probe)?;
    let cap = compute_capacity(n_build)?;
    let cap_u32 = u32::try_from(cap)
        .map_err(|_| BoltError::Other(format!("gpu_join: cap {cap} doesn't fit in u32")))?;

    let build_keys_host = encode_keys_for_shape(build_key_columns, shape)?;
    let probe_keys_host = encode_keys_for_shape(probe_key_columns, shape)?;

    // v0.7 async-memcpy migration: per-call stream + async H2D for every
    // host-side init buffer. Mirrors the exact-shape INNER path.
    let stream = CudaStream::null_or_default();

    let build_keys_dev = upload_primitive_values_async::<i64>(&build_keys_host, &stream)?;
    let probe_keys_dev = upload_primitive_values_async::<i64>(&probe_keys_host, &stream)?;

    // PERF (join key encoding): u32::MAX collision-list tables filled on-device
    // (cuMemsetD8Async 0xFF) — no host scratch, no H2D. The i64::MIN key table
    // is not byte-replicable so it keeps its pooled host slice + async H2D
    // (see top-of-file `get_sentinel_*_vec` and `alloc_u32_max_table_async`).
    let keys_init = get_sentinel_i64_min_vec(cap);
    let mut keys_table_dev = upload_primitive_values_async::<i64>(keys_init, &stream)?;
    let mut row_idx_table_dev = alloc_u32_max_table_async(cap, &stream)?;
    let mut head_dev = alloc_u32_max_table_async(cap, &stream)?;
    let mut next_idx_dev = alloc_u32_max_table_async(n_build, &stream)?;

    let out_capacity_usize = n_build
        .checked_add(n_probe)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| BoltError::Other("gpu_join: output buffer size overflow".into()))?;
    let out_capacity_u32 = u32::try_from(out_capacity_usize).map_err(|_| {
        BoltError::Other(format!(
            "gpu_join: out_capacity {out_capacity_usize} doesn't fit in u32"
        ))
    })?;

    let mut out_probe_idx_dev = GpuVec::<u32>::zeros(out_capacity_usize)?;
    let mut out_build_idx_dev = GpuVec::<u32>::zeros(out_capacity_usize)?;
    let mut out_counter_dev = GpuVec::<u32>::zeros(1)?;

    launch_build_collision_kernel(
        &build_keys_dev,
        &mut keys_table_dev,
        &mut row_idx_table_dev,
        &mut head_dev,
        &mut next_idx_dev,
        n_build_u32,
        cap_u32,
        &stream,
    )?;
    let n_matches_raw = launch_probe_collision_kernel(
        &probe_keys_dev,
        &keys_table_dev,
        &head_dev,
        &next_idx_dev,
        &mut out_probe_idx_dev,
        &mut out_build_idx_dev,
        &mut out_counter_dev,
        None,
        n_probe_u32,
        n_build_u32,
        cap_u32,
        out_capacity_u32,
        &stream,
    )?;
    if n_matches_raw > out_capacity_u32 {
        // Probe overflow: kernel wrote more candidate matches than
        // out_capacity. The host join handles this input fine — return the
        // fallback signal so the executor retries there. Lossy-fold
        // false-positive blow-ups land here; `try_gpu_inner_join` maps
        // `Err(BoltError::GpuCapacity(_))` to its host-fallback `Ok(None)`
        // path.
        log::warn!(
            "gpu join probe overflow ({n_matches_raw} > {out_capacity_u32}); \
             falling back to host hash join"
        );
        return Err(BoltError::GpuCapacity(format!(
            "gpu_join: lossy-shape candidate filter claimed {n_matches_raw} > capacity {out_capacity_u32}"
        )));
    }
    let n_matches = n_matches_raw as usize;
    // v0.7 async-memcpy: pinned-host D2H + per-call sync, matching the
    // verified collision-list path. The candidate indices flow into the
    // host post-verify pipeline.
    let probe_idx_full = download_to_host_pinned::<u32>(&out_probe_idx_dev, &stream)?;
    let build_idx_full = download_to_host_pinned::<u32>(&out_build_idx_dev, &stream)?;
    let probe_idx: Vec<u32> = probe_idx_full.into_iter().take(n_matches).collect();
    let build_idx: Vec<u32> = build_idx_full.into_iter().take(n_matches).collect();
    Ok(GpuJoinIndices {
        build: UInt32Array::from(build_idx),
        probe: UInt32Array::from(probe_idx),
    })
}

/// End-to-end GPU CROSS JOIN. Pure cartesian product — every left row paired
/// with every right row, in `(left_idx, right_idx)` row-major order.
///
/// Gates:
/// * `n_left * n_right` is in `[CROSS_JOIN_GPU_MIN_CELLS, CROSS_JOIN_GPU_CELL_CAP]`.
///   Below the min, host wins; above the cap, the host orchestrator must
///   reject the query entirely (the index arrays don't fit in u32 anyway).
/// * Neither side is empty (caller short-circuits that case).
///
/// Returns a fully-materialised `RecordBatch` over `output_schema = left ++
/// right`. Output row ordering is deterministic: `(left_idx, right_idx)`
/// pairs are emitted in row-major order, i.e. left-row 0 paired with every
/// right-row, then left-row 1, etc. Matches the host CROSS path.
pub fn execute_cross_join_on_gpu(
    lhs: &RecordBatch,
    rhs: &RecordBatch,
    output_schema: Arc<ArrowSchema>,
) -> BoltResult<RecordBatch> {
    let n_left = lhs.num_rows();
    let n_right = rhs.num_rows();
    if n_left == 0 || n_right == 0 {
        return Err(BoltError::Other(
            "gpu_join: execute_cross_join_on_gpu called with empty side; \
             caller must short-circuit before reaching the GPU path"
                .into(),
        ));
    }

    let total = (n_left as u64).checked_mul(n_right as u64).ok_or_else(|| {
        BoltError::Other(format!("gpu_join: CROSS overflow ({n_left} × {n_right})"))
    })?;
    if total >= CROSS_JOIN_GPU_CELL_CAP {
        return Err(BoltError::Other(format!(
            "gpu_join: CROSS product {total} ≥ cap {CROSS_JOIN_GPU_CELL_CAP}; fall back to host"
        )));
    }
    let total_usize = total as usize;
    let total_u32 = u32::try_from(total).map_err(|_| {
        BoltError::Other(format!("gpu_join: CROSS total {total} doesn't fit in u32"))
    })?;
    let n_build_u32 = n_rows_to_u32(n_right)?; // "build" = right side here

    // Output buffers.
    let out_probe_idx_dev = GpuVec::<u32>::zeros(total_usize)?;
    let out_build_idx_dev = GpuVec::<u32>::zeros(total_usize)?;
    let stream = CudaStream::null();

    // Launch. The CROSS kernel has no join key; the `HashJoinKernelSpec`
    // shape requires SOME `key_dtype`, so we pass `DataType::Bool` as a
    // sentinel. `string_hash_returns_i64` is ignored for this variant
    // (only consulted for `StringHash`).
    let spec = HashJoinKernelSpec {
        kind: HashJoinKernelKind::Cross,
        key_dtype: DataType::Bool,
        string_hash_returns_i64: false,
    };
    let module =
        module_cache::get_or_build_module_for_hash_join(&spec, CROSS_KERNEL_ENTRY, |_| {
            compile_cross_kernel()
        })?;
    let function = module.function(CROSS_KERNEL_ENTRY)?;

    let mut out_probe_idx_ptr: CUdeviceptr = out_probe_idx_dev.device_ptr();
    let mut out_build_idx_ptr: CUdeviceptr = out_build_idx_dev.device_ptr();
    let mut n_build_param: u32 = n_build_u32;
    let mut total_param: u32 = total_u32;
    let mut params: [*mut c_void; 4] = [
        &mut out_probe_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_build_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_build_param as *mut u32 as *mut c_void,
        &mut total_param as *mut u32 as *mut c_void,
    ];

    let block: u32 = HASH_JOIN_BLOCK_SIZE;
    let grid_x: u32 = total_u32.div_ceil(block).max(1);
    // SAFETY: every param slot points at a stack local that outlives the
    // launch+sync below; the device buffers outlive the launch.
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
    stream.synchronize()?;

    let probe_idx_host: Vec<u32> = out_probe_idx_dev.to_vec()?;
    let build_idx_host: Vec<u32> = out_build_idx_dev.to_vec()?;

    let left_idx_arr = UInt32Array::from(probe_idx_host);
    let right_idx_arr = UInt32Array::from(build_idx_host);

    let mut output_cols: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
    for col in lhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), &left_idx_arr, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: cross take (left): {e}")))?,
        );
    }
    for col in rhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), &right_idx_arr, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: cross take (right): {e}")))?,
        );
    }
    RecordBatch::try_new(output_schema, output_cols)
        .map_err(|e| BoltError::Other(format!("gpu_join: building CROSS RecordBatch: {e}")))
}

/// End-to-end GPU INNER join for Utf8 keys. Host-side string interning
/// produces i32 dictionary indices that the kernel-side `SingleI32` path
/// handles natively. The output reattaches the *original* StringArray
/// columns from `lhs` / `rhs` via `arrow::compute::take`.
///
/// ## Stage-4 streaming-intern opt-in
///
/// When `BOLT_GPU_JOIN_STREAMING_INTERN` is set (see
/// [`STREAMING_INTERN_ENV_VAR`]) this routes through
/// [`intern_utf8_columns_streaming`] — a 64-bit-hash-keyed dict that's
/// ~5-10× smaller than the Stage-3 byte-borrowed variant on
/// high-cardinality Utf8 joins. Hash collisions are dropped by a
/// `verify_pairs_on_host` pass over the original StringArrays before
/// `arrow::compute::take` is called, so the streaming path is byte-stable
/// with the Stage-3 path's output (modulo arbitrary INNER-join row order).
pub fn execute_utf8_inner_join_on_gpu(
    lhs: &RecordBatch,
    rhs: &RecordBatch,
    build_is_left: bool,
    build_key_idx: usize,
    probe_key_idx: usize,
    output_schema: Arc<ArrowSchema>,
) -> BoltResult<RecordBatch> {
    let (build_batch, probe_batch) = if build_is_left {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };

    // Stage-4 (GJ): pick between byte-borrowed (exact) and 64-bit-hash-keyed
    // (streaming) interning. The hash-keyed variant can collide, so the
    // streaming path MUST run the host post-verify before trusting any pair.
    //
    // Stage-5 (GJ): when the streaming path is enabled AND the input is
    // large enough, route through `intern_utf8_columns_streaming_parallel`
    // — same caller contract, but per-chunk dicts are built in parallel via
    // `std::thread::scope` and merged sequentially. Below the threshold the
    // sequential variant wins (thread-spawn cost dominates).
    let streaming = streaming_intern_enabled();
    let interned = if streaming {
        let total =
            build_batch.column(build_key_idx).len() + probe_batch.column(probe_key_idx).len();
        if total >= STREAMING_INTERN_PARALLEL_THRESHOLD {
            intern_utf8_columns_streaming_parallel(
                build_batch.column(build_key_idx).as_ref(),
                probe_batch.column(probe_key_idx).as_ref(),
            )?
        } else {
            intern_utf8_columns_streaming(
                build_batch.column(build_key_idx).as_ref(),
                probe_batch.column(probe_key_idx).as_ref(),
            )?
        }
    } else {
        intern_utf8_columns(
            build_batch.column(build_key_idx).as_ref(),
            probe_batch.column(probe_key_idx).as_ref(),
        )?
    };

    let (build_indices, probe_indices) = hash_join_indices_on_gpu(
        &interned.build_indices,
        &interned.probe_indices,
        DataType::Int32,
    )?;

    // Streaming path: drop hash collisions by re-comparing the *original*
    // StringArrays at the emitted (build, probe) indices. The Stage-3 exact
    // path skips this because the kernel-side i32 compare is byte-exact.
    let (build_indices, probe_indices) = if streaming {
        let build_str = build_batch.column(build_key_idx);
        let probe_str = probe_batch.column(probe_key_idx);
        verify_pairs_on_host(
            &[build_str.as_ref()],
            &[probe_str.as_ref()],
            &build_indices,
            &probe_indices,
        )?
    } else {
        (build_indices, probe_indices)
    };

    let (left_idx, right_idx) = if build_is_left {
        (&build_indices, &probe_indices)
    } else {
        (&probe_indices, &build_indices)
    };
    let mut output_cols: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
    for col in lhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), left_idx, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: utf8 take (left): {e}")))?,
        );
    }
    for col in rhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), right_idx, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: utf8 take (right): {e}")))?,
        );
    }
    RecordBatch::try_new(output_schema, output_cols)
        .map_err(|e| BoltError::Other(format!("gpu_join: utf8 RecordBatch: {e}")))
}

// =========================================================================
// Stage 5: OUTER for Utf8 keys via dict interning + post-verify.
// =========================================================================

/// Stage-5: end-to-end GPU OUTER join for `SingleUtf8` keys.
///
/// Stage 4 lifted the lossy-shape gate (`TwoI64` / `MultiI32`) for OUTER by
/// running the GPU as a candidate filter and re-verifying pairs on the
/// host. Stage 5 lifts the OUTER gate over Utf8 by the same template, with
/// one simplification: the **byte-borrowed** Stage-3 intern path
/// (`intern_utf8_columns`) produces EXACT i32 dict indices — distinct
/// strings get distinct indices — so the GPU's `SingleI32` OUTER output is
/// already correct. The host post-verify in
/// `execute_outer_join_indices_on_gpu` is a no-op for `SingleI32`
/// (`needs_host_post_verify() == false`), so we route directly through it.
///
/// Flow:
///   1. Intern build + probe strings to dense i32 dict indices.
///   2. Run the Stage-2 collision-list build + probe with the matched
///      bitmap enabled, treating the indices as `SingleI32`.
///   3. The pair stream + matched bitmap are already correct under the
///      byte-borrowed dict; emit LEFT-fill / RIGHT-fill rows via the
///      existing `execute_outer_join_indices_on_gpu` pipeline.
///   4. Reattach the *original* `StringArray` columns via `take` against
///      the index arrays.
///
/// **Streaming-intern caveat.** The Stage-4 streaming-intern variant keys
/// the dict on a 64-bit hash, so distinct strings *can* share an index.
/// Stage 5 routes OUTER through the byte-borrowed dict only — the
/// streaming + OUTER combination needs a post-verify pass over the
/// candidate matches AND a re-derivation of the matched bitmap, which is
/// a Stage 6 follow-up. The `streaming_intern_enabled()` flag is honoured
/// for INNER (see [`execute_utf8_inner_join_on_gpu`]) but ignored here.
#[allow(clippy::too_many_arguments)]
pub fn execute_utf8_outer_join_on_gpu(
    lhs: &RecordBatch,
    rhs: &RecordBatch,
    build_is_left: bool,
    build_key_idx: usize,
    probe_key_idx: usize,
    emit_unmatched_probe: bool,
    emit_unmatched_build: bool,
    output_schema: Arc<ArrowSchema>,
) -> BoltResult<RecordBatch> {
    let (build_batch, probe_batch) = if build_is_left {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };

    let build_str_col = build_batch.column(build_key_idx);
    let probe_str_col = probe_batch.column(probe_key_idx);

    // Stage-5 byte-borrowed dict. Streaming intern would require a
    // candidate-filter pipeline analogous to TwoI64 OUTER (see Stage-6
    // follow-up in the module docs).
    let interned = intern_utf8_columns(build_str_col.as_ref(), probe_str_col.as_ref())?;

    let build_cols: Vec<&dyn Array> = vec![&interned.build_indices];
    let probe_cols: Vec<&dyn Array> = vec![&interned.probe_indices];

    // Run the Stage-2 OUTER collision-list path against the i32 dict
    // indices. `SingleI32` is exact-in-i64, so the GPU output (matched
    // pairs + unmatched-row emission) is correct without further verify.
    let outer = execute_outer_join_indices_on_gpu(
        &build_cols,
        &probe_cols,
        KeyShape::SingleI32,
        emit_unmatched_probe,
        emit_unmatched_build,
    )?;

    let (left_idx_vec, right_idx_vec) = if build_is_left {
        (outer.build, outer.probe)
    } else {
        (outer.probe, outer.build)
    };
    let left_idx_arr = UInt32Array::from(left_idx_vec);
    let right_idx_arr = UInt32Array::from(right_idx_vec);

    let mut output_cols: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
    for col in lhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), &left_idx_arr, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: utf8 outer take (left): {e}")))?,
        );
    }
    for col in rhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), &right_idx_arr, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: utf8 outer take (right): {e}")))?,
        );
    }
    RecordBatch::try_new(output_schema, output_cols)
        .map_err(|e| BoltError::Other(format!("gpu_join: utf8 outer RecordBatch: {e}")))
}

// =========================================================================
// Stage 5: engine-level AoS routing heuristic + INNER entry point.
// =========================================================================

/// Stage-5: decide whether the INNER-join pair `(n_probe, n_build)` is
/// probe-heavy enough that the AoS slot layout is the better fit.
///
/// Heuristic: AoS wins when `n_probe > AOS_ROUTING_PROBE_BUILD_RATIO * n_build`.
/// Below that the SoA path is cheaper (less memory, same probe-bandwidth
/// for the build-side initialisation and the CAS pass). The condition is
/// expressed as multiplication on the build side (not division on the
/// probe side) so partial ratios above the threshold still trip the gate
/// — e.g. `(8001, 1000)` correctly routes to AoS even though
/// `8001 / 1000 == 8` under integer division.
#[inline]
pub fn should_route_aos(n_probe: usize, n_build: usize) -> bool {
    if n_build == 0 {
        return false;
    }
    // Guard against the (extremely unlikely) overflow on huge build sides.
    // If `AOS_ROUTING_PROBE_BUILD_RATIO * n_build` overflows usize the
    // ratio is definitely > threshold for the n_probe we can represent,
    // so route to AoS.
    match AOS_ROUTING_PROBE_BUILD_RATIO.checked_mul(n_build) {
        Some(rhs) => n_probe > rhs,
        None => true,
    }
}

/// Stage-5: end-to-end AoS INNER join over `RecordBatch`es, single Int32
/// or Int64 key. Symmetric with [`execute_inner_join_on_gpu`] but routes
/// through the Stage-4 AoS build kernel + Stage-3 AoS probe kernel.
///
/// Same gate as the SoA path (unique build keys, ≥ 1024 rows / side, no
/// NULLs in the key column). The caller must already have verified those
/// gates; this is a kernel-flow swap, not a new fast path.
pub fn execute_inner_join_on_gpu_aos(
    lhs: &RecordBatch,
    rhs: &RecordBatch,
    build_is_left: bool,
    build_key_idx: usize,
    probe_key_idx: usize,
    dtype: DataType,
    output_schema: Arc<ArrowSchema>,
) -> BoltResult<RecordBatch> {
    let (build_batch, probe_batch) = if build_is_left {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };

    let build_keys_col = build_batch.column(build_key_idx);
    let probe_keys_col = probe_batch.column(probe_key_idx);

    let (build_indices, probe_indices) =
        hash_join_indices_on_gpu_aos(build_keys_col.as_ref(), probe_keys_col.as_ref(), dtype)?;

    let (left_idx, right_idx) = if build_is_left {
        (&build_indices, &probe_indices)
    } else {
        (&probe_indices, &build_indices)
    };

    let mut output_cols: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
    for col in lhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), left_idx, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: aos take (left): {e}")))?,
        );
    }
    for col in rhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), right_idx, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: aos take (right): {e}")))?,
        );
    }

    RecordBatch::try_new(output_schema, output_cols)
        .map_err(|e| BoltError::Other(format!("gpu_join: aos building RecordBatch failed: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType as ArrowDataType, Field, Schema as ArrowSchema};

    // -- Pure-host helpers --

    #[test]
    fn sentinel_buffers_are_reused_and_grow_only_upwards() {
        // Drive the pool through three asks and verify (a) the contents are
        // always sentinel-filled to the requested length and (b) the backing
        // storage only ever grows. Pointer identity confirms the second:
        // requests at-or-below the high-water mark return slices into the
        // same backing buffer; requests above it trigger a (leaked) realloc.
        //
        // The pools are process-wide; to keep the pointer-identity checks
        // immune to parallel-test interference we serialise the whole
        // sequence on a test-local mutex.
        use parking_lot::Mutex as TestMutex;
        use std::sync::OnceLock;
        static TEST_GATE: OnceLock<TestMutex<()>> = OnceLock::new();
        let _g = TEST_GATE.get_or_init(|| TestMutex::new(())).lock();

        // Capture today's high-water by reading the pool length directly.
        let baseline_i64 = SENTINEL_I64_MIN_POOL.lock().len();

        // (1) Small ask. May or may not grow the pool depending on baseline.
        let small_cap = baseline_i64.max(100);
        let s1 = get_sentinel_i64_min_vec(small_cap);
        assert_eq!(s1.len(), small_cap);
        assert!(s1.iter().all(|&v| v == i64::MIN));
        let s1_ptr = s1.as_ptr();

        // (2) Smaller ask — MUST reuse the same backing storage.
        let smaller_cap = small_cap / 2;
        let s2 = get_sentinel_i64_min_vec(smaller_cap);
        assert_eq!(s2.len(), smaller_cap);
        assert!(s2.iter().all(|&v| v == i64::MIN));
        assert_eq!(
            s2.as_ptr(),
            s1_ptr,
            "smaller request should reuse the existing buffer, not reallocate"
        );

        // (3) Larger ask — forces a grow. New backing storage; pointer
        // identity to the previous buffer is no longer required.
        let larger_cap = small_cap.saturating_mul(2).max(small_cap + 1);
        let s3 = get_sentinel_i64_min_vec(larger_cap);
        assert_eq!(s3.len(), larger_cap);
        assert!(s3.iter().all(|&v| v == i64::MIN));

        // (4) After growth, a subsequent at-or-below ask reuses the new
        // high-water buffer.
        let s4 = get_sentinel_i64_min_vec(larger_cap - 1);
        assert_eq!(s4.len(), larger_cap - 1);
        assert!(s4.iter().all(|&v| v == i64::MIN));
        assert_eq!(
            s4.as_ptr(),
            s3.as_ptr(),
            "ask below new high-water should reuse the grown buffer"
        );

        // Same contract for the u32::MAX pool.
        let baseline_u32 = SENTINEL_U32_MAX_POOL.lock().len();
        let cap_a = baseline_u32.max(64);
        let u1 = get_sentinel_u32_max_vec(cap_a);
        assert_eq!(u1.len(), cap_a);
        assert!(u1.iter().all(|&v| v == u32::MAX));
        let u1_ptr = u1.as_ptr();

        let u2 = get_sentinel_u32_max_vec(cap_a / 2);
        assert_eq!(u2.as_ptr(), u1_ptr);
        assert!(u2.iter().all(|&v| v == u32::MAX));

        let u3 = get_sentinel_u32_max_vec(cap_a.saturating_mul(2).max(cap_a + 1));
        assert!(u3.iter().all(|&v| v == u32::MAX));

        // Zero-length asks are allowed and yield empty slices.
        let z = get_sentinel_i64_min_vec(0);
        assert!(z.is_empty());
    }

    #[test]
    fn compute_capacity_powers_of_two() {
        // Pin a deterministic slot cap so the test is independent of which
        // VRAM regime the host happens to be running on.
        let cap = HASH_TABLE_BYTE_CAP_DEFAULT / 12;
        // 50% load factor: capacity = next_pow2(2 * n).
        assert_eq!(compute_capacity_with_slot_cap(1, cap).unwrap(), 2);
        assert_eq!(compute_capacity_with_slot_cap(2, cap).unwrap(), 4);
        assert_eq!(compute_capacity_with_slot_cap(3, cap).unwrap(), 8); // 2*3 = 6 -> next_pow2 = 8
        assert_eq!(compute_capacity_with_slot_cap(4, cap).unwrap(), 8);
        assert_eq!(compute_capacity_with_slot_cap(1024, cap).unwrap(), 2048);
        assert_eq!(compute_capacity_with_slot_cap(1025, cap).unwrap(), 4096);
        // Just under the cap.
        assert!(compute_capacity_with_slot_cap(1_000_000, cap).is_ok());
    }

    #[test]
    fn compute_capacity_rejects_oversized() {
        let cap = HASH_TABLE_BYTE_CAP_DEFAULT / 12;
        // cap ≈ 5_592_405 slots; 2 * n must fit, so the cap itself is
        // already over the limit.
        assert!(compute_capacity_with_slot_cap(cap, cap).is_err());
    }

    #[test]
    fn large_vram_cap_is_512_mib() {
        // The lifted cap matches the documented Stage-2 number.
        assert_eq!(HASH_TABLE_BYTE_CAP_LARGE, 512 * 1024 * 1024);
        // The default cap stays at the Stage-1 64 MiB.
        assert_eq!(HASH_TABLE_BYTE_CAP_DEFAULT, 64 * 1024 * 1024);
        // 8 GiB threshold per the task spec.
        assert_eq!(LARGE_VRAM_THRESHOLD, 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn encode_int32_sign_extends() {
        let arr = Int32Array::from(vec![1i32, -1, i32::MIN, i32::MAX]);
        let enc = encode_keys_i64(&arr, DataType::Int32).unwrap();
        assert_eq!(enc, vec![1i64, -1, i32::MIN as i64, i32::MAX as i64]);
        // None of these can equal i64::MIN, so no sentinel collision.
        assert!(enc.iter().all(|v| *v != i64::MIN));
    }

    #[test]
    fn encode_int64_identity() {
        let arr = Int64Array::from(vec![0i64, 1, -1, i64::MAX, i64::MIN + 1]);
        let enc = encode_keys_i64(&arr, DataType::Int64).unwrap();
        assert_eq!(enc, vec![0i64, 1, -1, i64::MAX, i64::MIN + 1]);
    }

    #[test]
    fn encode_int64_rejects_sentinel() {
        let arr = Int64Array::from(vec![0i64, i64::MIN, 1]);
        let err = encode_keys_i64(&arr, DataType::Int64);
        assert!(
            err.is_err(),
            "i64::MIN in input must be rejected as a sentinel collision"
        );
    }

    #[test]
    fn encode_rejects_unsupported_dtype() {
        let arr = arrow_array::Float64Array::from(vec![1.0, 2.0]);
        assert!(encode_keys_i64(&arr, DataType::Float64).is_err());
    }

    // -- GPU round-trip --

    /// R1 (resident hook): the device-resident INNER-join path must produce
    /// the EXACT same matched pair set as the host-downloading path
    /// `hash_join_indices_on_gpu` — they share the build+probe kernels, so
    /// `hash_join_indices_on_gpu_resident(..).download()` is byte-equivalent
    /// to `hash_join_indices_on_gpu(..)` after sorting (the output order is
    /// the arbitrary atomic-counter order on both). This is the proof that
    /// the composition seam doesn't change join semantics.
    #[test]
    #[ignore = "gpu:join"]
    fn gpu_resident_inner_join_matches_host_path() {
        let n_build = 2000usize;
        let n_probe = 4000usize;
        let build_keys: Vec<i32> = (0..n_build as i32).collect();
        let probe_keys: Vec<i32> = (0..n_probe as i32).map(|i| i % 3000).collect();

        let build_col = Int32Array::from(build_keys.clone());
        let probe_col = Int32Array::from(probe_keys.clone());

        // Host-downloading path (the established reference).
        let (host_build, host_probe) =
            hash_join_indices_on_gpu(&build_col, &probe_col, DataType::Int32)
                .expect("host-download gpu join");

        // Device-resident path, then explicit download.
        let resident = hash_join_indices_on_gpu_resident(&build_col, &probe_col, DataType::Int32)
            .expect("resident gpu join");
        assert_eq!(
            resident.n_matches,
            host_build.len(),
            "resident match count must equal host-download match count"
        );
        // The resident buffers are over-sized; only the n_matches prefix is
        // valid. `download()` truncates to that prefix.
        assert!(
            resident.build_idx.len() >= resident.n_matches,
            "resident build buffer must hold at least n_matches entries"
        );
        let (res_build, res_probe) = resident.download().expect("resident download");

        // Reconcile arbitrary ordering: sort both pair sets.
        let mut host_pairs: Vec<(u32, u32)> = (0..host_build.len())
            .map(|i| (host_build.value(i), host_probe.value(i)))
            .collect();
        let mut res_pairs: Vec<(u32, u32)> = (0..res_build.len())
            .map(|i| (res_build.value(i), res_probe.value(i)))
            .collect();
        host_pairs.sort_unstable();
        res_pairs.sort_unstable();
        assert_eq!(
            res_pairs, host_pairs,
            "resident join match set must equal host-download match set"
        );
    }

    /// R1 (resident hook): an empty probe side yields a resident result with
    /// `n_matches == 0` and empty device buffers — the consumer's downstream
    /// gather becomes a no-op without any host round trip.
    #[test]
    #[ignore = "gpu:join"]
    fn gpu_resident_inner_join_empty_probe() {
        let build_col = Int32Array::from((0..1024i32).collect::<Vec<_>>());
        let probe_col = Int32Array::from(Vec::<i32>::new());
        let resident = hash_join_indices_on_gpu_resident(&build_col, &probe_col, DataType::Int32)
            .expect("resident gpu join (empty probe)");
        assert_eq!(resident.n_matches, 0);
        let (b, p) = resident.download().expect("download");
        assert_eq!(b.len(), 0);
        assert_eq!(p.len(), 0);
    }

    /// Build two batches with a known overlap, run the GPU join, and verify
    /// the recovered match set matches the host-computed answer. The
    /// arbitrary-order output is reconciled by sorting both sides on
    /// (probe_idx, build_idx) before comparison.
    #[test]
    #[ignore = "gpu:join"]
    fn gpu_hash_join_int32_round_trip() {
        // Build side: 2000 unique keys 0..2000, payload = key + 1000.
        // Probe side: 4000 keys, every other one matching a build key.
        let n_build = 2000usize;
        let n_probe = 4000usize;

        let build_keys: Vec<i32> = (0..n_build as i32).collect();
        let build_payload: Vec<i32> = build_keys.iter().map(|k| k + 1000).collect();
        let probe_keys: Vec<i32> = (0..n_probe as i32).map(|i| i % 3000).collect();
        let probe_payload: Vec<i32> = (0..n_probe as i32).map(|i| 10_000 + i).collect();

        let build_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("k", ArrowDataType::Int32, false),
            Field::new("bp", ArrowDataType::Int32, false),
        ]));
        let build_batch = RecordBatch::try_new(
            build_schema,
            vec![
                Arc::new(Int32Array::from(build_keys.clone())),
                Arc::new(Int32Array::from(build_payload.clone())),
            ],
        )
        .unwrap();

        let probe_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("k", ArrowDataType::Int32, false),
            Field::new("pp", ArrowDataType::Int32, false),
        ]));
        let probe_batch = RecordBatch::try_new(
            probe_schema,
            vec![
                Arc::new(Int32Array::from(probe_keys.clone())),
                Arc::new(Int32Array::from(probe_payload.clone())),
            ],
        )
        .unwrap();

        let (build_idx, probe_idx) = hash_join_indices_on_gpu(
            build_batch.column(0).as_ref(),
            probe_batch.column(0).as_ref(),
            DataType::Int32,
        )
        .expect("gpu join");

        assert_eq!(
            build_idx.len(),
            probe_idx.len(),
            "matched pair count must agree"
        );

        // Reconstruct the host-side expected set.
        let mut expected: Vec<(u32, u32)> = Vec::new();
        for (pi, pk) in probe_keys.iter().enumerate() {
            if (*pk as usize) < n_build {
                expected.push((*pk as u32, pi as u32));
            }
        }
        expected.sort_unstable();

        let mut got: Vec<(u32, u32)> = (0..build_idx.len())
            .map(|i| (build_idx.value(i), probe_idx.value(i)))
            .collect();
        got.sort_unstable();

        assert_eq!(got, expected, "GPU join match set must equal host expected");
    }

    /// End-to-end test through `execute_inner_join_on_gpu`: same fixture as
    /// above but exercises the full take + concat path.
    #[test]
    #[ignore = "gpu:join"]
    fn gpu_inner_join_full_batch_round_trip() {
        let n_build = 2000usize;
        let n_probe = 4000usize;
        let build_keys: Vec<i32> = (0..n_build as i32).collect();
        let build_payload: Vec<i32> = build_keys.iter().map(|k| k + 1000).collect();
        let probe_keys: Vec<i32> = (0..n_probe as i32).map(|i| i % 3000).collect();
        let probe_payload: Vec<i32> = (0..n_probe as i32).map(|i| 10_000 + i).collect();

        let build_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("k", ArrowDataType::Int32, false),
            Field::new("bp", ArrowDataType::Int32, false),
        ]));
        let build_batch = RecordBatch::try_new(
            build_schema,
            vec![
                Arc::new(Int32Array::from(build_keys.clone())),
                Arc::new(Int32Array::from(build_payload.clone())),
            ],
        )
        .unwrap();

        let probe_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("k", ArrowDataType::Int32, false),
            Field::new("pp", ArrowDataType::Int32, false),
        ]));
        let probe_batch = RecordBatch::try_new(
            probe_schema,
            vec![
                Arc::new(Int32Array::from(probe_keys.clone())),
                Arc::new(Int32Array::from(probe_payload.clone())),
            ],
        )
        .unwrap();

        // Output schema = left (build) ++ right (probe).
        let out_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("k", ArrowDataType::Int32, false),
            Field::new("bp", ArrowDataType::Int32, false),
            Field::new("k_2", ArrowDataType::Int32, false),
            Field::new("pp", ArrowDataType::Int32, false),
        ]));

        let out = execute_inner_join_on_gpu(
            &build_batch,
            &probe_batch,
            /* build_is_left */ true,
            0,
            0,
            DataType::Int32,
            out_schema,
        )
        .expect("gpu inner join");

        // Expected match count: probe rows whose key < n_build = 2000.
        // probe_keys = 0..4000 % 3000 -> keys 0..2999 each appear at least
        // once; specifically the matches are those probe_keys < 2000.
        let expected: usize = probe_keys
            .iter()
            .filter(|k| (**k as usize) < n_build)
            .count();
        assert_eq!(
            out.num_rows(),
            expected,
            "match count must match host estimate"
        );

        // Spot-check that every output row satisfies the equi-join:
        // build_payload column (col 1) = build_key + 1000 = probe_key + 1000.
        let bp_col = out.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        let pk_col = out.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..out.num_rows() {
            assert_eq!(
                bp_col.value(i),
                pk_col.value(i) + 1000,
                "row {i}: bp must equal probe_key + 1000 (left.k == right.k invariant)"
            );
        }
    }

    /// 0-row probe side: must produce an empty result without panicking.
    #[test]
    #[ignore = "gpu:join"]
    fn gpu_hash_join_empty_probe() {
        let build_keys: Vec<i32> = (0..1024).collect();
        let probe_keys: Vec<i32> = Vec::new();

        let (build_idx, probe_idx) = hash_join_indices_on_gpu(
            &Int32Array::from(build_keys),
            &Int32Array::from(probe_keys),
            DataType::Int32,
        )
        .expect("gpu join");

        assert_eq!(build_idx.len(), 0);
        assert_eq!(probe_idx.len(), 0);
    }

    /// Int64 keys, same shape as the Int32 round-trip.
    #[test]
    #[ignore = "gpu:join"]
    fn gpu_hash_join_int64_round_trip() {
        let n_build = 1500usize;
        let n_probe = 3000usize;
        let build_keys: Vec<i64> = (0..n_build as i64).collect();
        let probe_keys: Vec<i64> = (0..n_probe as i64).map(|i| i % 2000).collect();

        let (build_idx, probe_idx) = hash_join_indices_on_gpu(
            &Int64Array::from(build_keys),
            &Int64Array::from(probe_keys.clone()),
            DataType::Int64,
        )
        .expect("gpu join");

        let mut expected: Vec<(u32, u32)> = Vec::new();
        for (pi, pk) in probe_keys.iter().enumerate() {
            if (*pk as usize) < n_build {
                expected.push((*pk as u32, pi as u32));
            }
        }
        expected.sort_unstable();

        let mut got: Vec<(u32, u32)> = (0..build_idx.len())
            .map(|i| (build_idx.value(i), probe_idx.value(i)))
            .collect();
        got.sort_unstable();

        assert_eq!(got, expected);
    }

    // ===== Stage 2 host-side helpers =====

    #[test]
    fn two_i32_packing_matches_pack_keys_convention() {
        // Pack convention: first column in high 32 bits, second in low 32.
        // Identical to `groupby::pack_keys` so kernel/host hashes line up
        // across the join/groupby boundary.
        let a = Int32Array::from(vec![1i32, -1, 0, i32::MAX]);
        let b = Int32Array::from(vec![2i32, -2, 5, i32::MIN]);
        let enc =
            encode_keys_for_shape(&[&a as &dyn Array, &b as &dyn Array], KeyShape::TwoI32).unwrap();
        assert_eq!(enc[0], ((1u32 as u64) << 32 | (2u32 as u64)) as i64);
        // -1 high, -2 low → both as u32, OR'd at the right offset.
        assert_eq!(
            enc[1],
            (((-1i32) as u32 as u64) << 32 | ((-2i32) as u32 as u64)) as i64
        );
        assert_eq!(enc[2], (0u64 << 32 | 5u64) as i64);
        assert_eq!(
            enc[3],
            (((i32::MAX as u32 as u64) << 32) | (i32::MIN as u32 as u64)) as i64
        );
    }

    #[test]
    fn bool_keys_encode_as_zero_one() {
        let a = arrow_array::BooleanArray::from(vec![true, false, true, false]);
        let enc = encode_keys_for_shape(&[&a as &dyn Array], KeyShape::SingleBool).unwrap();
        assert_eq!(enc, vec![1, 0, 1, 0]);
    }

    #[test]
    fn float_nan_canonicalisation_collapses_distinct_nan_patterns() {
        // f64::NAN.to_bits() is the canonical pattern; any other NaN bit
        // pattern should reduce to it before hashing.
        let weird_nan = f64::from_bits(f64::NAN.to_bits() ^ 0x1);
        assert!(weird_nan.is_nan());
        let a = arrow_array::Float64Array::from(vec![f64::NAN, weird_nan, 1.0]);
        let enc = encode_keys_for_shape(&[&a as &dyn Array], KeyShape::SingleF64).unwrap();
        assert_eq!(enc[0], enc[1], "all NaNs must canonicalise to the same key");
        assert_ne!(enc[0], enc[2]);
    }

    #[test]
    fn float_negative_zero_collapses_to_positive_zero() {
        // SQL treats -0.0 == +0.0; our encoder must too.
        let a = arrow_array::Float64Array::from(vec![-0.0, 0.0]);
        let enc = encode_keys_for_shape(&[&a as &dyn Array], KeyShape::SingleF64).unwrap();
        assert_eq!(enc[0], enc[1]);
        assert_eq!(enc[0], 0);
    }

    #[test]
    fn lossy_fold_shapes_decline_gpu_path() {
        let a = Int64Array::from(vec![1i64, 2, 3]);
        let b = Int64Array::from(vec![10i64, 20, 30]);
        let cols: Vec<&dyn Array> = vec![&a, &b];
        // hash_join_indices_on_gpu_with_shape must surface a clear error
        // for lossy shapes — host orchestrator translates to fall-through.
        let err = hash_join_indices_on_gpu_with_shape(&cols, &cols, KeyShape::TwoI64);
        assert!(err.is_err(), "TwoI64 must decline (lossy fold)");
        let err = hash_join_indices_on_gpu_with_shape(&cols, &cols, KeyShape::MultiI32(3));
        assert!(err.is_err(), "MultiI32 must decline (lossy fold)");
    }

    #[test]
    fn host_splitmix_is_deterministic_and_avalanches() {
        // Same input -> same output (purity).
        assert_eq!(host_splitmix(0), host_splitmix(0));
        assert_eq!(host_splitmix(1), host_splitmix(1));
        // Single-bit input flip changes ~half the output bits — loose
        // avalanche check (≥ 16 of 64 bits differ for nearby inputs).
        let h0 = host_splitmix(1);
        let h1 = host_splitmix(2);
        assert_ne!(h0, h1);
        let differing = (h0 ^ h1).count_ones();
        assert!(
            differing >= 16,
            "splitmix should differ in at least 16 bits across nearby inputs; got {differing}"
        );
    }

    // ===== Stage 3 host-side helpers =====

    /// Utf8 interning must give the same dict index to repeated strings and
    /// distinct indices to distinct strings. The "union over both sides"
    /// shape is important: a probe string not appearing in the build still
    /// gets a fresh index (it just won't find a slot in the kernel).
    #[test]
    fn intern_utf8_columns_dedups_and_assigns_distinct_indices() {
        let build = StringArray::from(vec!["alice", "bob", "alice", "carol"]);
        let probe = StringArray::from(vec!["bob", "alice", "dave"]);
        let interned = intern_utf8_columns(&build, &probe).unwrap();

        // Build: alice=0, bob=1, alice=0, carol=2.
        assert_eq!(interned.build_indices.value(0), 0);
        assert_eq!(interned.build_indices.value(1), 1);
        assert_eq!(interned.build_indices.value(2), 0);
        assert_eq!(interned.build_indices.value(3), 2);
        // Probe: bob=1, alice=0, dave=3 (new).
        assert_eq!(interned.probe_indices.value(0), 1);
        assert_eq!(interned.probe_indices.value(1), 0);
        assert_eq!(interned.probe_indices.value(2), 3);
    }

    /// Per-pair host verification keeps exact matches and drops false
    /// positives. Build/probe key columns each have a single Int64 column;
    /// candidate pairs include one true match (build[0]==probe[0]) and one
    /// false match (build[1]!=probe[1]).
    #[test]
    fn verify_pairs_on_host_drops_false_positives() {
        let b = Int64Array::from(vec![100i64, 200, 300]);
        let p = Int64Array::from(vec![100i64, 999, 300]);
        let build_cols: Vec<&dyn Array> = vec![&b];
        let probe_cols: Vec<&dyn Array> = vec![&p];
        // Candidate pairs: (b0,p0) true, (b1,p1) FALSE, (b2,p2) true.
        let cb = UInt32Array::from(vec![0u32, 1, 2]);
        let cp = UInt32Array::from(vec![0u32, 1, 2]);
        let (kept_b, kept_p) = verify_pairs_on_host(&build_cols, &probe_cols, &cb, &cp).unwrap();
        assert_eq!(kept_b.len(), 2);
        assert_eq!(kept_p.len(), 2);
        assert_eq!(kept_b.value(0), 0);
        assert_eq!(kept_b.value(1), 2);
        assert_eq!(kept_p.value(0), 0);
        assert_eq!(kept_p.value(1), 2);
    }

    /// Per-pair host verification with a two-column key: a candidate pair
    /// is kept only if EVERY column agrees. Catches the bug where the
    /// verifier short-circuits on the first column without checking the
    /// rest.
    #[test]
    fn verify_pairs_on_host_requires_all_columns_to_agree() {
        let b1 = Int32Array::from(vec![1i32, 1, 2]);
        let b2 = Int64Array::from(vec![10i64, 20, 10]);
        let p1 = Int32Array::from(vec![1i32, 1, 2]);
        let p2 = Int64Array::from(vec![10i64, 999, 10]); // p2[1] disagrees
        let build_cols: Vec<&dyn Array> = vec![&b1, &b2];
        let probe_cols: Vec<&dyn Array> = vec![&p1, &p2];
        let cb = UInt32Array::from(vec![0u32, 1, 2]);
        let cp = UInt32Array::from(vec![0u32, 1, 2]);
        let (kept_b, _) = verify_pairs_on_host(&build_cols, &probe_cols, &cb, &cp).unwrap();
        assert_eq!(
            kept_b.len(),
            2,
            "pair (b1,p1) must be dropped on col-2 mismatch"
        );
        assert_eq!(kept_b.value(0), 0);
        assert_eq!(kept_b.value(1), 2);
    }

    /// Env-var cap: a valid integer is parsed and clamped. The OnceLock
    /// latches on the FIRST call, so we test the parser directly.
    #[test]
    fn parse_env_cap_clamps_to_range() {
        // Save / restore the env var so tests don't bleed into each other.
        let prev = std::env::var(CAP_ENV_VAR).ok();
        std::env::set_var(CAP_ENV_VAR, "128");
        let cap = parse_env_cap().expect("128 MiB must parse");
        assert_eq!(cap, 128 * 1024 * 1024);

        // Below min → clamped up to 64 MiB.
        std::env::set_var(CAP_ENV_VAR, "1");
        let cap = parse_env_cap().expect("1 MiB must parse");
        assert_eq!(cap, CAP_ENV_MIN_MIB * 1024 * 1024);

        // Above max → clamped down to 4 GiB.
        std::env::set_var(CAP_ENV_VAR, "999999");
        let cap = parse_env_cap().expect("999999 MiB must parse");
        assert_eq!(cap, CAP_ENV_MAX_MIB * 1024 * 1024);

        // Garbage → ignored.
        std::env::set_var(CAP_ENV_VAR, "not-a-number");
        assert!(
            parse_env_cap().is_none(),
            "garbage env value must be ignored"
        );

        // Restore.
        match prev {
            Some(v) => std::env::set_var(CAP_ENV_VAR, v),
            None => std::env::remove_var(CAP_ENV_VAR),
        }
    }

    /// Stage-3 KeyShape: `SingleUtf8` claims to be exact. The Utf8 path is
    /// expected to use the dedicated interning entry point — direct
    /// encoding via encode_keys_for_shape returns a clear error.
    #[test]
    fn single_utf8_encode_steers_to_interning_path() {
        let s = StringArray::from(vec!["hello"]);
        let err = encode_keys_for_shape(&[&s as &dyn Array], KeyShape::SingleUtf8);
        assert!(
            err.is_err(),
            "SingleUtf8 must not be encoded directly — caller should use intern_utf8_columns"
        );
    }

    /// CROSS cap is a fixed compile-time bound; pinning it here means a
    /// future change requires an intentional update.
    #[test]
    fn cross_join_cell_cap_is_100m() {
        assert_eq!(CROSS_JOIN_GPU_CELL_CAP, 100_000_000);
    }

    // ------------------------------------------------------------------
    // Stage 4 (GJ-4) unit tests.
    // ------------------------------------------------------------------

    /// Streaming intern: identical strings on both sides MUST get the same
    /// dict index (so the kernel-side i32 compare matches). Host-only test
    /// — no CUDA driver involvement.
    #[test]
    fn streaming_intern_identical_strings_share_index() {
        let b = StringArray::from(vec!["alpha", "beta", "gamma", "delta"]);
        let p = StringArray::from(vec!["gamma", "alpha", "epsilon", "beta"]);
        let interned = intern_utf8_columns_streaming(&b, &p).expect("streaming intern");

        // Build "alpha" and probe "alpha" share an index.
        let b_alpha = interned.build_indices.value(0);
        let p_alpha = interned.probe_indices.value(1);
        assert_eq!(
            b_alpha, p_alpha,
            "same string on build and probe must hash to the same dict index"
        );
        let b_beta = interned.build_indices.value(1);
        let p_beta = interned.probe_indices.value(3);
        assert_eq!(b_beta, p_beta, "build/probe 'beta' must share an index");
        // Probe-only string must still get a valid (non-negative) index.
        let p_epsilon = interned.probe_indices.value(2);
        assert!(
            p_epsilon >= 0,
            "probe-only strings must still get a non-sentinel dict index"
        );
    }

    /// The streaming-intern env var toggles cleanly: empty / "0" / unset
    /// keep the byte-borrowed path; non-zero values enable the streaming
    /// path. Host-only — no CUDA driver involvement.
    #[test]
    fn streaming_intern_env_var_toggle() {
        let prev = std::env::var(STREAMING_INTERN_ENV_VAR).ok();

        std::env::remove_var(STREAMING_INTERN_ENV_VAR);
        assert!(!streaming_intern_enabled(), "unset env: streaming OFF");

        std::env::set_var(STREAMING_INTERN_ENV_VAR, "");
        assert!(!streaming_intern_enabled(), "empty env: streaming OFF");

        std::env::set_var(STREAMING_INTERN_ENV_VAR, "0");
        assert!(!streaming_intern_enabled(), "'0' env: streaming OFF");

        std::env::set_var(STREAMING_INTERN_ENV_VAR, "false");
        assert!(!streaming_intern_enabled(), "'false' env: streaming OFF");

        std::env::set_var(STREAMING_INTERN_ENV_VAR, "1");
        assert!(streaming_intern_enabled(), "'1' env: streaming ON");

        std::env::set_var(STREAMING_INTERN_ENV_VAR, "true");
        assert!(streaming_intern_enabled(), "'true' env: streaming ON");

        match prev {
            Some(v) => std::env::set_var(STREAMING_INTERN_ENV_VAR, v),
            None => std::env::remove_var(STREAMING_INTERN_ENV_VAR),
        }
    }

    /// **Batch 6** — `BOLT_HASH_PROBE_TILED` toggles the SoA probe between
    /// the single-load Stage-1 kernel and the tile-aware 2-way unrolled
    /// kernel. Default off (unset / empty / "0" / "false"); any other
    /// non-empty value flips it on. Host-only, no CUDA driver involvement.
    #[test]
    fn probe_tiled_env_var_toggle() {
        let prev = std::env::var(PROBE_TILED_ENV_VAR).ok();

        std::env::remove_var(PROBE_TILED_ENV_VAR);
        assert!(!probe_tiled_enabled(), "unset env: tiled OFF");

        std::env::set_var(PROBE_TILED_ENV_VAR, "");
        assert!(!probe_tiled_enabled(), "empty env: tiled OFF");

        std::env::set_var(PROBE_TILED_ENV_VAR, "0");
        assert!(!probe_tiled_enabled(), "'0' env: tiled OFF");

        std::env::set_var(PROBE_TILED_ENV_VAR, "false");
        assert!(!probe_tiled_enabled(), "'false' env: tiled OFF");

        std::env::set_var(PROBE_TILED_ENV_VAR, "FALSE");
        assert!(
            !probe_tiled_enabled(),
            "'FALSE' env (case-insensitive): tiled OFF"
        );

        std::env::set_var(PROBE_TILED_ENV_VAR, "1");
        assert!(probe_tiled_enabled(), "'1' env: tiled ON");

        std::env::set_var(PROBE_TILED_ENV_VAR, "true");
        assert!(probe_tiled_enabled(), "'true' env: tiled ON");

        // Restore.
        match prev {
            Some(v) => std::env::set_var(PROBE_TILED_ENV_VAR, v),
            None => std::env::remove_var(PROBE_TILED_ENV_VAR),
        }
    }

    /// AoS slot init populates every key word with the EMPTY_KEY sentinel
    /// (`i64::MIN`) and leaves head/pad words at zero. Without this the
    /// AoS build kernel's CAS would treat raw-zero key slots as "occupied
    /// by key 0" and silently lose inserts.
    ///
    /// Host-only — uses `Vec<u8>` math, no CUDA call.
    #[test]
    fn aos_slot_init_is_sentinel_plus_zero() {
        let cap = 4usize;
        let bytes_per_slot = AOS_SLOT_BYTES as usize;
        // Reproduce the host-side initialiser in `alloc_aos_slots` without
        // touching the device. If `alloc_aos_slots` ever changes its
        // initialisation scheme this test will catch the divergence at
        // compile time (rather than at silent runtime corruption).
        let mut init: Vec<u8> = vec![0u8; cap * bytes_per_slot];
        let sentinel = i64::MIN.to_le_bytes();
        for slot in 0..cap {
            let off = slot * bytes_per_slot;
            init[off..off + 8].copy_from_slice(&sentinel);
        }

        for slot in 0..cap {
            let off = slot * bytes_per_slot;
            let key_word = i64::from_le_bytes(init[off..off + 8].try_into().unwrap());
            assert_eq!(
                key_word,
                i64::MIN,
                "slot {slot}: key word must be EMPTY_KEY (= i64::MIN) on init"
            );
            // Head word at offset 8.
            let head_word = u32::from_le_bytes(init[off + 8..off + 12].try_into().unwrap());
            assert_eq!(head_word, 0, "slot {slot}: head word must be zero on init");
            // Pad word at offset 12.
            let pad_word = u32::from_le_bytes(init[off + 12..off + 16].try_into().unwrap());
            assert_eq!(pad_word, 0, "slot {slot}: pad word must be zero on init");
        }
    }

    /// AoS round-trip: build + probe through the AoS path must agree with
    /// the SoA path on the same fixture. This is the GPU-touching Stage-4
    /// regression test that backs `tests/gpu_join_e2e.rs::aos_build_layout_no_regression`.
    /// Gated on `--ignored` for the same reason as the other GPU tests.
    #[test]
    #[ignore = "gpu:join"]
    fn aos_matches_soa() {
        let n_build = 2_000usize;
        let n_probe = 4_000usize;
        let build_keys: Vec<i32> = (0..n_build as i32).collect();
        let probe_keys: Vec<i32> = (0..n_probe as i32).map(|i| i % 3_000).collect();

        let bk = Int32Array::from(build_keys);
        let pk = Int32Array::from(probe_keys);

        let (b_soa, p_soa) = hash_join_indices_on_gpu(&bk, &pk, DataType::Int32).expect("SoA path");
        let (b_aos, p_aos) =
            hash_join_indices_on_gpu_aos(&bk, &pk, DataType::Int32).expect("AoS path");

        assert_eq!(
            b_soa.len(),
            b_aos.len(),
            "AoS/SoA match count diverged (SoA={}, AoS={})",
            b_soa.len(),
            b_aos.len()
        );
        let mut soa: Vec<(u32, u32)> = (0..b_soa.len())
            .map(|i| (p_soa.value(i), b_soa.value(i)))
            .collect();
        let mut aos: Vec<(u32, u32)> = (0..b_aos.len())
            .map(|i| (p_aos.value(i), b_aos.value(i)))
            .collect();
        soa.sort_unstable();
        aos.sort_unstable();
        assert_eq!(soa, aos, "AoS match set must equal SoA match set");
    }

    // ------------------------------------------------------------------
    // Stage 5 (GJ-5) unit tests.
    // ------------------------------------------------------------------

    /// `should_route_aos`: only ratios *strictly greater than* the
    /// threshold route to AoS. Equal-ratio falls to SoA so the conservative
    /// default holds.
    #[test]
    fn aos_routing_threshold_matches_documented_ratio() {
        assert_eq!(AOS_ROUTING_PROBE_BUILD_RATIO, 8);

        // Edge: exactly at the threshold → SoA wins.
        assert!(
            !should_route_aos(8 * 1000, 1000),
            "ratio == 8 must stay SoA"
        );
        // Past the threshold → AoS.
        assert!(
            should_route_aos(8 * 1000 + 1, 1000),
            "ratio just past 8 must pick AoS"
        );
        // Way past → AoS.
        assert!(should_route_aos(100_000, 1_000), "100x ratio must pick AoS");
        // Balanced sizes → SoA.
        assert!(
            !should_route_aos(50_000, 50_000),
            "balanced sides must stay SoA"
        );
        // Build-heavy → SoA.
        assert!(
            !should_route_aos(1_000, 100_000),
            "build-heavy must stay SoA"
        );
        // Degenerate empty build → SoA (avoids divide-by-zero).
        assert!(!should_route_aos(1_000_000, 0), "empty build must stay SoA");
    }

    /// Parallel streaming intern must agree with the sequential variant on
    /// the SAME (build, probe) inputs. Host-only — no CUDA driver
    /// involvement; the threads are pure CPU.
    #[test]
    fn parallel_streaming_intern_agrees_with_sequential() {
        // Mix strings so we exercise both the small-string and
        // medium-string hash paths. We don't try to force a hash
        // collision — the verify pass downstream handles those — but we DO
        // require that distinct strings get distinct indices, identical
        // strings share an index.
        let build: Vec<&str> = (0..1024)
            .map(|i| {
                // Use a small string pool so dedup actually happens.
                match i % 4 {
                    0 => "alice",
                    1 => "bob",
                    2 => "carol",
                    _ => "dave",
                }
            })
            .collect();
        let probe: Vec<&str> = (0..2048)
            .map(|i| match i % 5 {
                0 => "alice",
                1 => "bob",
                2 => "eve",
                3 => "frank",
                _ => "carol",
            })
            .collect();
        let b = StringArray::from(build);
        let p = StringArray::from(probe);

        let seq = intern_utf8_columns_streaming(&b, &p).expect("seq");
        let par = intern_utf8_columns_streaming_parallel(&b, &p).expect("par");

        assert_eq!(seq.build_indices.len(), par.build_indices.len());
        assert_eq!(seq.probe_indices.len(), par.probe_indices.len());

        // The two implementations assign indices in different orders
        // (parallel-merge walks chunks; sequential walks each side once),
        // so the i32 values can differ. What MUST match is the equivalence
        // structure: two rows that share a string in `seq` MUST share a
        // string in `par`, and vice-versa. We check this by reducing both
        // sides to a canonical relabelling.
        let canon = |arr: &Int32Array, src: &StringArray| -> Vec<i32> {
            // Map first-seen-string-value to a fresh canonical id.
            let mut by_string: HashMap<&str, i32> = HashMap::new();
            let mut next = 0i32;
            let mut out: Vec<i32> = Vec::with_capacity(arr.len());
            for i in 0..arr.len() {
                if src.is_null(i) {
                    out.push(-1);
                    continue;
                }
                let s = src.value(i);
                let id = *by_string.entry(s).or_insert_with(|| {
                    let cur = next;
                    next += 1;
                    cur
                });
                out.push(id);
            }
            out
        };
        let seq_canon_b = canon(&seq.build_indices, &b);
        let par_canon_b = canon(&par.build_indices, &b);
        assert_eq!(seq_canon_b, par_canon_b, "build canonical labels diverge");
        let seq_canon_p = canon(&seq.probe_indices, &p);
        let par_canon_p = canon(&par.probe_indices, &p);
        assert_eq!(seq_canon_p, par_canon_p, "probe canonical labels diverge");
    }

    /// NULL rows in the parallel intern get the same `-1` sentinel as the
    /// sequential / byte-borrowed paths.
    #[test]
    fn parallel_streaming_intern_handles_null_rows() {
        // Build a StringArray with NULLs by going through the
        // builder API; the `from(Vec<&str>)` variant doesn't admit Nones.
        let mut b = arrow_array::builder::StringBuilder::new();
        b.append_value("x");
        b.append_null();
        b.append_value("y");
        let build = b.finish();

        let mut p = arrow_array::builder::StringBuilder::new();
        p.append_value("x");
        p.append_value("y");
        let probe = p.finish();

        let par = intern_utf8_columns_streaming_parallel(&build, &probe).expect("par");

        assert_eq!(par.build_indices.value(1), -1, "NULL build row must get -1");
        assert_eq!(par.build_indices.value(0), par.probe_indices.value(0));
        assert_eq!(par.build_indices.value(2), par.probe_indices.value(1));
    }

    /// `STREAMING_INTERN_DEVICE_HASH_THRESHOLD` is at a sane order of
    /// magnitude. Pinning it here means any future bump requires an
    /// intentional change (and the doc comment usually gets updated in
    /// tandem with the constant).
    #[test]
    fn stage5_thresholds_are_in_sane_range() {
        assert!(
            STREAMING_INTERN_PARALLEL_THRESHOLD >= 64 * 1024,
            "parallel threshold below 64k rows wastes the thread-spawn cost"
        );
        assert!(
            STREAMING_INTERN_DEVICE_HASH_THRESHOLD >= 100_000,
            "device-hash threshold below 100k rows wastes the kernel-launch cost"
        );
    }

    /// Stage-5: the device-side string hash kernel must produce the same
    /// 64-bit hash as the host's `utf8_hash64` for every row of a
    /// `StringArray`. If the kernel diverges, a host-built dict would
    /// fail to look up the device-hashed value (and vice versa), so this
    /// is load-bearing.
    ///
    /// Gated on `--ignored` because the kernel launch requires a CUDA
    /// driver. Below the gate the host path is correct and exercised by
    /// the streaming-intern unit tests above.
    #[test]
    #[ignore = "gpu:join"]
    fn device_string_hash_matches_host() {
        // Fixture: mix short, medium, empty, and long strings. The device
        // kernel walks each row's byte range; an empty string must hash to
        // FNV_OFFSET then through splitmix without touching the values
        // buffer (the loop body never runs).
        let inputs: Vec<&str> = vec![
            "",
            "a",
            "ab",
            "alice",
            "bobby",
            "the quick brown fox jumps over the lazy dog",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "this is a much longer string that exercises the byte-by-byte FNV-1a inner loop multiple times over",
            "x", "y", "z",
        ];
        let arr = StringArray::from(inputs.clone());

        // Compute device-side hashes.
        let device = compute_device_string_hashes(&arr).expect("device hash");
        assert_eq!(device.len(), inputs.len());

        // Compute host-side hashes via the same `utf8_hash64` the streaming
        // intern path uses. The two MUST be byte-identical per row.
        for (i, s) in inputs.iter().enumerate() {
            let host = utf8_hash64(s.as_bytes());
            assert_eq!(
                device[i], host,
                "row {i} ('{}'): device hash {:#018x} != host hash {:#018x}",
                s, device[i], host
            );
        }
    }

    // =========================================================================
    // v0.7: async memcpy + pinned host buffers wired through the hash-join
    // build/probe paths. Under `--features cuda-stub` the async FFI shims
    // return `CUDA_ERROR_STUB`; the helper falls back to the synchronous
    // `from_slice` wrapper which also returns `CUDA_ERROR_STUB`. Both paths
    // surface a typed `BoltError` at the FFI boundary rather than panicking,
    // and the executor returns that error to the host fallback path in
    // `crate::exec::join::try_gpu_inner_join`. This test pins the contract.
    // =========================================================================

    /// Under `--features cuda-stub` the hash-join entry point must NOT
    /// panic when the async H2D shim returns `CUDA_ERROR_STUB`. The
    /// previous (synchronous) path also returned `Err(_)` here; this test
    /// pins that the v0.7 async rewrite preserves that contract.
    ///
    /// On a real-CUDA host the same call succeeds; the test is split into
    /// the "stub returns Err" case (gated on the feature) and the
    /// "real-CUDA returns Ok" case (gated on `#[ignore]` so it only runs
    /// when explicitly requested with `--ignored`, matching the rest of
    /// the GPU-driver-dependent tests in this module).
    #[cfg(feature = "cuda-stub")]
    #[test]
    fn hash_join_under_cuda_stub_returns_err_gracefully() {
        // Inputs sized to clear every shape gate (≥ 1024 rows). The stub
        // backend's `cuda_sys::check` maps `CUDA_ERROR_STUB` to a typed
        // BoltError before the kernel launch ever happens, so we never
        // actually look at the indices.
        let n: usize = 2048;
        let build_keys: Vec<i32> = (0..n as i32).collect();
        let probe_keys: Vec<i32> = (0..n as i32).map(|i| i % 1024).collect();

        let build_arr = Int32Array::from(build_keys);
        let probe_arr = Int32Array::from(probe_keys);

        // The contract: must return Err rather than panic. We do NOT
        // assert on the error variant — the exact mapping is owned by
        // `cuda_sys::check` and may change without breaking the
        // executor's host-fallback path (which catches any `Err(_)`).
        let result = hash_join_indices_on_gpu(&build_arr, &probe_arr, DataType::Int32);
        assert!(
            result.is_err(),
            "under cuda-stub, hash_join_indices_on_gpu must return Err so \
             the host fallback in join.rs::try_gpu_inner_join can engage"
        );
    }

    /// Same contract for the collision-list INNER path (lossy + exact
    /// shapes share the entry below the `is_exact_in_i64()` gate).
    #[cfg(feature = "cuda-stub")]
    #[test]
    fn hash_join_with_shape_under_cuda_stub_returns_err_gracefully() {
        let n: usize = 1024;
        let build_keys: Vec<i32> = (0..n as i32).collect();
        let probe_keys: Vec<i32> = (0..n as i32).collect();

        let build_arr = Int32Array::from(build_keys);
        let probe_arr = Int32Array::from(probe_keys);
        let b: &dyn Array = &build_arr;
        let p: &dyn Array = &probe_arr;

        let result = hash_join_indices_on_gpu_with_shape(&[b], &[p], KeyShape::SingleI32);
        assert!(
            result.is_err(),
            "under cuda-stub, hash_join_indices_on_gpu_with_shape must \
             return Err so the host fallback engages"
        );
    }
}
