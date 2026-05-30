// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for GPU-side filter compaction (prefix scan + gather).
//!
//! Today's compaction path (see `crate::exec::compact`) downloads the u8 mask
//! to the host and uses `arrow::compute::filter` to drop masked-out rows. That
//! requires a per-row d2h copy plus a host-side O(n) scan per column. This
//! module emits the kernels for an entirely-on-device compaction:
//!
//! 1. **Per-block Hillis-Steele exclusive prefix-sum** over the u8 mask. Each
//!    block of `BLOCK_SIZE` threads writes its thread's exclusive prefix into
//!    `local_indices[gid]` and the block's running sum (the inclusive scan at
//!    the last thread) into `block_sums[blockIdx.x]`.
//! 2. (Host stage — see `crate::exec::gpu_compact`) Download the block_sums,
//!    exclusive-scan on the CPU, and re-upload as `block_bases`. The total
//!    count is the sum of the block_sums.
//! 3. **Per-dtype gather kernel**: for each surviving row, write
//!    `output[block_bases[blockIdx.x] + local_indices[gid]] = input[gid]`.
//!
//! The shared-memory ping-pong here is deliberately the simpler Hillis-Steele
//! variant (O(n log n) work; fine at BLOCK_SIZE = 256) rather than Blelloch.
//! Two `sdata` buffers swap each round to avoid a redundant `bar.sync`.
//!
//! ## What's deferred
//!
//! The host-side scan of `block_sums` assumes the array fits comfortably in
//! host memory and runs serially — at `BLOCK_SIZE = 256` we hit 16M rows per
//! 65 535 blocks, which is plenty for one batch. For multi-batch row counts
//! beyond `u32::MAX / 256` the caller should split the input; this module
//! validates the bound up-front in `compile_*` callers (the limit lives in
//! `gpu_compact.rs`).

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::DataType;

/// PTX target metadata baked into every emitted module.
const PTX_VERSION: &str = ".version 7.5";
/// Target SM architecture string.
const PTX_TARGET: &str = ".target sm_70";
/// Address size directive (we always use 64-bit pointers).
const PTX_ADDRESS_SIZE: &str = ".address_size 64";

/// Threads per block used by both the prefix-scan and the gather kernels.
///
/// Chosen so a single Hillis-Steele block scan fits easily in shared memory
/// (`BLOCK_SIZE * 4 bytes` for u32 entries times two ping-pong buffers).
///
/// MUST be a power of two: the Hillis-Steele scan walks strides `1, 2, 4, ...`
/// up to `BLOCK_SIZE/2`, and the lane-`tid >= offset` predicate assumes the
/// final stride exactly halves the block.
pub const BLOCK_SIZE: u32 = 256;

// Compile-time invariant for the Hillis-Steele block scan. See the doc
// comment on `BLOCK_SIZE` above.
const _: () = assert!(
    BLOCK_SIZE.is_power_of_two(),
    "Hillis-Steele block scan requires power-of-two BLOCK_SIZE"
);

/// Entry-point name for the per-block prefix-scan kernel.
pub const SCAN_KERNEL_ENTRY: &str = "bolt_prefix_scan";

/// Entry-point name for the Blelloch upsweep+downsweep variant of the
/// per-block prefix-scan kernel.
///
/// Same ABI as [`SCAN_KERNEL_ENTRY`], so host code can swap one PTX for the
/// other without re-plumbing argument arrays. Activated via the
/// `BOLT_PREFIX_SCAN_ALGO=blelloch` env var (see
/// [`crate::exec::gpu_compact::prefix_scan_mask`]); the Hillis-Steele kernel
/// remains the default while the Blelloch path bakes.
pub const SCAN_KERNEL_ENTRY_BLELLOCH: &str = "bolt_prefix_scan_blelloch";

/// Entry-point name for the **single-pass decoupled-lookback** prefix-scan
/// kernel.
///
/// Unlike the Hillis-Steele and Blelloch variants this kernel does NOT
/// require a host round-trip to exclusive-scan `block_sums`: each block
/// publishes its own aggregate to a per-block status slot and then walks
/// previous slots backwards (the "decoupled lookback" of Merrill & Garland,
/// "Single-pass Parallel Prefix Scan with Decoupled Look-back", NVR-2016-002)
/// to derive its global prefix in one grid launch.
///
/// 5-arg ABI:
/// ```text
/// .visible .entry bolt_prefix_scan_lookback(
///     .param .u64 ..._param_0,   // mask_ptr        (u8*)
///     .param .u64 ..._param_1,   // local_indices   (u32*)  -- holds GLOBAL exclusive prefix on return
///     .param .u64 ..._param_2,   // block_sums      (u32*)  -- written but unused by caller
///     .param .u32 ..._param_3,   // n_rows
///     .param .u64 ..._param_4    // partial_status  (u32*)  -- per-block decoupled-lookback slots
/// )
/// ```
///
/// `partial_status[blockIdx.x]` is a `u32` whose top 2 bits encode the
/// publication status and the low 30 bits carry the aggregate or inclusive
/// prefix value:
///
/// | bits  | meaning                                              |
/// |-------|------------------------------------------------------|
/// | 31:30 | status: 0 = INVALID, 1 = AGGREGATE, 2 = INCLUSIVE     |
/// | 29:0  | value (block aggregate OR inclusive global prefix)   |
///
/// See [`LOOKBACK_VALUE_MASK`] / [`LOOKBACK_STATUS_AGGREGATE`] /
/// [`LOOKBACK_STATUS_INCLUSIVE`] / [`LOOKBACK_STATUS_INVALID`].
///
/// On return `local_indices[gid]` holds the **global** exclusive prefix
/// (already added the block prefix), so the host can skip the
/// download-scan-upload round trip the other variants need. `block_sums`
/// is still written (block aggregate) for diagnostic compatibility but
/// is not required by the gather kernel — the caller passes an all-zero
/// `block_bases` to `gather_one`.
pub const SCAN_KERNEL_ENTRY_LOOKBACK: &str = "bolt_prefix_scan_lookback";

/// Bitmask for the value (low 30) bits in a `partial_status` slot.
pub const LOOKBACK_VALUE_MASK: u32 = 0x3FFF_FFFF;

/// Status code: slot has not been published yet. The zero-init of the
/// `partial_status` buffer naturally encodes this state.
pub const LOOKBACK_STATUS_INVALID: u32 = 0;

/// Status code: block published its own aggregate but has not yet
/// resolved its global prefix. The aggregate value is in the low 30 bits.
pub const LOOKBACK_STATUS_AGGREGATE: u32 = 1;

/// Status code: block has resolved its global prefix; the low 30 bits
/// carry the **inclusive** global prefix (`block_prefix + block_aggregate`),
/// which is what successor blocks consume to short-circuit the lookback.
pub const LOOKBACK_STATUS_INCLUSIVE: u32 = 2;

/// Entry-point name for the per-dtype gather kernel.
///
/// Returns a static string so callers can pass it straight to
/// `CudaModule::function`. Errors are deferred to `compile_gather_kernel` so
/// invalid dtypes surface during PTX generation rather than name lookup.
pub fn gather_kernel_entry(dtype: DataType) -> &'static str {
    match dtype {
        DataType::Bool => "bolt_gather_bool",
        DataType::Int32 => "bolt_gather_i32",
        DataType::Int64 => "bolt_gather_i64",
        DataType::Float32 => "bolt_gather_f32",
        DataType::Float64 => "bolt_gather_f64",
        DataType::Utf8 => "bolt_gather_utf8_unsupported",
        // Decimal128 has no GPU gather kernel yet (v0.6 / M4 plan-only);
        // surface an unsupported-named entry so any dispatch attempt fails
        // loudly. The actual rejection happens in the executor before any
        // gather is launched.
        DataType::Decimal128(_, _) => "bolt_gather_decimal128_unsupported",
        // v0.7: temporal gather reuses the matching integer kernel. The
        // gather kernel is type-agnostic apart from the load/store width,
        // so we route Date32 to the i32 kernel and Timestamp to the i64
        // kernel — they emit the same PTX shape and same SM occupancy.
        DataType::Date32 => "bolt_gather_i32",
        DataType::Timestamp(_, _) => "bolt_gather_i64",
    }
}

/// Generate the per-block exclusive prefix-scan PTX module.
///
/// ABI:
/// ```text
/// .visible .entry bolt_prefix_scan(
///     .param .u64 bolt_prefix_scan_param_0,   // mask_ptr      (u8*)
///     .param .u64 bolt_prefix_scan_param_1,   // local_indices (u32*)
///     .param .u64 bolt_prefix_scan_param_2,   // block_sums    (u32*)
///     .param .u32 bolt_prefix_scan_param_3    // n_rows
/// )
/// ```
///
/// `local_indices[i]` holds the exclusive prefix sum within the row's block;
/// `block_sums[blockIdx.x]` holds the inclusive sum of the entire block (the
/// number of "kept" rows in that block).
pub fn compile_prefix_scan_kernel() -> BoltResult<String> {
    // Two u32 buffers of BLOCK_SIZE entries each: ping-pong avoids one
    // bar.sync per round vs. reading and writing the same buffer.
    let elem_bytes: u32 = 4;
    let shared_bytes_per_buf: u32 = BLOCK_SIZE * elem_bytes;
    let shared_bytes_total: u32 = shared_bytes_per_buf * 2;

    let mut ptx = String::new();
    writeln!(ptx, "{}", PTX_VERSION).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_TARGET).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_ADDRESS_SIZE).map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(
        ptx,
        ".shared .align 4 .b8 sdata[{shared}];",
        shared = shared_bytes_total
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Signature: mask, local_indices, block_sums, n_rows.
    writeln!(ptx, ".visible .entry {}(", SCAN_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", SCAN_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", SCAN_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_2,", SCAN_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_3", SCAN_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Register pool. PTX `.reg` decls only allocate names; pick generous sizes.
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b16   %rs<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<32>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // -------- Indices: tid_x = %tid.x ; ctaid = %ctaid.x ; gid = ctaid*ntid+tid
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    // n_rows
    writeln!(
        ptx,
        "\tld.param.u32 %r4, [{}_param_3];",
        SCAN_KERNEL_ENTRY
    )
    .map_err(write_err)?;

    // -------- Globalize parameter pointers.
    writeln!(
        ptx,
        "\tld.param.u64 %rd0, [{}_param_0];",
        SCAN_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u64 %rd1, [{}_param_1];",
        SCAN_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u64 %rd2, [{}_param_2];",
        SCAN_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;

    // -------- Load this thread's mask byte (0 if past the end). Treat any
    // non-zero byte as "keep" by emitting (m != 0) ? 1 : 0 to make the scan
    // robust to predicate kernels that emit truthy values other than 1.
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r5, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra AFTER_LOAD;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd3, %r3, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd4, %rd0, %rd3;").map_err(write_err)?;
    // mask is a read-only input (the host side passes a distinct GpuVec<u8>;
    // local_indices and block_sums in this kernel are write-only), so route
    // the load through the read-only cache via `ld.global.nc`.
    writeln!(ptx, "\tld.global.nc.u8 %rs0, [%rd4];").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u16 %r6, %rs0;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p1, %r6, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.s32 %r5, 1, 0, %p1;").map_err(write_err)?;
    writeln!(ptx, "AFTER_LOAD:").map_err(write_err)?;

    // %r5 is now the 0/1 value for this thread. We also keep it for the
    // inclusive->exclusive conversion at the end.

    // -------- Stash the value into ping buffer (sdata[0..BLOCK_SIZE]).
    // base0 = sdata ; base1 = sdata + BLOCK_SIZE*4 ; idx_off = tid * 4
    // NOTE: keep `mov.u64` here (not `cvta.shared.u64`). %rd5 (and its derived
    // addresses %rd8/%rd9) feed `st.shared.u32` / `ld.shared.u32` below, which
    // require a shared-state-space address. `cvta.shared.u64` produces a
    // generic-space pointer; switching would force every shared ld/st in this
    // kernel to drop its `.shared` qualifier — outside this cleanup's scope.
    writeln!(ptx, "\tmov.u64 %rd5, sdata;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tadd.s64 %rd6, %rd5, {off};",
        off = shared_bytes_per_buf
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd7, %r2, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd8, %rd5, %rd7;").map_err(write_err)?; // ping addr
    writeln!(ptx, "\tadd.s64 %rd9, %rd6, %rd7;").map_err(write_err)?; // pong addr
    writeln!(ptx, "\tst.shared.u32 [%rd8], %r5;").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;

    // -------- Hillis-Steele scan: for offset in {1,2,4,...,BLOCK_SIZE/2}:
    //   v = (tid >= offset) ? ping[tid - offset] : 0
    //   pong[tid] = ping[tid] + v
    //   bar.sync
    //   swap ping/pong
    //
    // We unroll the loop. At the start of each round, %rd8 points into the
    // "read" buffer at this thread's slot and %rd9 into the "write" buffer.
    // After the round we swap them by xchg'ing the registers via a temp.
    let mut offset: u32 = 1;
    let mut step: usize = 0;
    while offset < BLOCK_SIZE {
        let off_bytes = offset * elem_bytes;
        // Load own value: %r7 = read[tid]
        writeln!(ptx, "\tld.shared.u32 %r7, [%rd8];").map_err(write_err)?;
        // Conditionally load neighbor at offset behind: %r8 = (tid >= offset)
        // ? read[tid - offset] : 0
        writeln!(
            ptx,
            "\tsetp.lt.s32 %p{p}, %r2, {off};",
            p = 2 + (step % 4),
            off = offset
        )
        .map_err(write_err)?;
        // Default to 0 then conditionally overwrite from shared memory.
        writeln!(ptx, "\tmov.u32 %r8, 0;").map_err(write_err)?;
        writeln!(
            ptx,
            "\t@%p{p} bra SKIP_LOAD_{step};",
            p = 2 + (step % 4),
            step = step
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tsub.s64 %rd10, %rd8, {off};",
            off = off_bytes
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tld.shared.u32 %r8, [%rd10];").map_err(write_err)?;
        writeln!(ptx, "SKIP_LOAD_{step}:", step = step).map_err(write_err)?;
        // sum = own + neighbor (or own + 0)
        writeln!(ptx, "\tadd.s32 %r9, %r7, %r8;").map_err(write_err)?;
        writeln!(ptx, "\tst.shared.u32 [%rd9], %r9;").map_err(write_err)?;
        writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
        // Swap ping/pong slot pointers: tmp = rd8 ; rd8 = rd9 ; rd9 = tmp
        writeln!(ptx, "\tmov.u64 %rd11, %rd8;").map_err(write_err)?;
        writeln!(ptx, "\tmov.u64 %rd8, %rd9;").map_err(write_err)?;
        writeln!(ptx, "\tmov.u64 %rd9, %rd11;").map_err(write_err)?;
        offset <<= 1;
        step += 1;
    }

    // After the loop, %rd8 points at this thread's slot in the buffer holding
    // the *inclusive* scan. Load it.
    writeln!(ptx, "\tld.shared.u32 %r10, [%rd8];").map_err(write_err)?;
    // Convert inclusive -> exclusive: excl = incl - own_value (own is %r5).
    writeln!(ptx, "\tsub.s32 %r11, %r10, %r5;").map_err(write_err)?;

    // -------- Write exclusive scan to local_indices[gid] (only in-range).
    writeln!(ptx, "\tsetp.ge.u32 %p5, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra AFTER_LOCAL_STORE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd12, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd13, %rd1, %rd12;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd13], %r11;").map_err(write_err)?;
    writeln!(ptx, "AFTER_LOCAL_STORE:").map_err(write_err)?;

    // -------- Thread (BLOCK_SIZE - 1) writes block_sums[blockIdx.x] = incl.
    //
    // For partial blocks (the last block when n_rows isn't a multiple of
    // BLOCK_SIZE) the last thread's inclusive scan still equals the count of
    // 1s in the block, because out-of-range lanes contributed zeros via the
    // AFTER_LOAD guard above.
    writeln!(
        ptx,
        "\tsetp.ne.s32 %p6, %r2, {last};",
        last = BLOCK_SIZE - 1
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p6 bra DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd14, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd15, %rd2, %rd14;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd15], %r10;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Generate the per-block exclusive prefix-scan PTX module using a **Blelloch
/// upsweep + downsweep** scan instead of the Hillis-Steele variant emitted by
/// [`compile_prefix_scan_kernel`].
///
/// ## Why
///
/// Hillis-Steele performs O(n log n) work and `log2(BLOCK_SIZE)` barriers;
/// Blelloch performs O(n) work with the same number of barriers, so for
/// `BLOCK_SIZE = 256` it issues ~8x fewer adds in shared memory. Both
/// kernels have identical ABI:
///
/// ```text
/// .visible .entry bolt_prefix_scan_blelloch(
///     .param .u64 ..._param_0,   // mask_ptr      (u8*)
///     .param .u64 ..._param_1,   // local_indices (u32*)
///     .param .u64 ..._param_2,   // block_sums    (u32*)
///     .param .u32 ..._param_3    // n_rows
/// )
/// ```
///
/// so host code can swap one PTX for the other transparently.
///
/// ## Algorithm
///
/// Operates on a single shared-memory buffer `arr[BLOCK_SIZE]` of u32. With
/// `BLOCK_SIZE = 256` we have `K = log2(256) = 8` levels.
///
/// 1. **Seed**: every thread loads its mask byte (0 if out of range),
///    normalises to 0/1, stores into `arr[tid]`. One `bar.sync`.
///
/// 2. **Upsweep (reduce tree)**: for `d in 0..K`:
///       stride = 1 << (d + 1)        // 2, 4, 8, ..., BLOCK_SIZE
///       idx    = tid * stride + stride - 1
///       if idx < BLOCK_SIZE:
///           arr[idx] += arr[idx - (stride / 2)]
///       bar.sync
///    After K levels `arr[BLOCK_SIZE - 1]` holds the inclusive sum of the
///    whole block (i.e. the count of kept rows = the value
///    `block_sums[blockIdx.x]` needs).
///
/// 3. **Pivot**: thread 0 saves `arr[BLOCK_SIZE - 1]` into a private register
///    (so the block-sum survives the zero-out), then writes
///    `arr[BLOCK_SIZE - 1] = 0`. One `bar.sync`. The zero is the additive
///    identity for the downsweep that turns the upsweep tree into the
///    exclusive scan.
///
/// 4. **Downsweep**: for `d in (K-1)..=0` (i.e. strides `BLOCK_SIZE, ..., 2`):
///       stride = 1 << (d + 1)
///       idx_r  = tid * stride + stride - 1
///       idx_l  = tid * stride + (stride / 2) - 1
///       if idx_r < BLOCK_SIZE:
///           t          = arr[idx_l];
///           arr[idx_l] = arr[idx_r];
///           arr[idx_r] = arr[idx_r] + t;
///       bar.sync
///    After K levels `arr[tid]` holds the exclusive prefix sum of the
///    original 0/1 values across the block.
///
/// 5. **Stores**: each in-range thread writes `arr[tid]` to
///    `local_indices[gid]`. Thread 0 writes the saved inclusive sum to
///    `block_sums[blockIdx.x]`.
///
/// ## Correctness notes
///
/// * Partial last block: out-of-range lanes seed `arr[tid] = 0` (the
///   `AFTER_LOAD` predicate is identical to Hillis-Steele's), so the
///   inclusive sum at `arr[BLOCK_SIZE - 1]` after upsweep still counts only
///   in-range "keep" bits.
/// * The downsweep predicate `(tid * stride + stride - 1) < BLOCK_SIZE` is
///   equivalent to `tid < BLOCK_SIZE / stride`, which we compute once per
///   level as an immediate comparison against the corresponding small
///   constant — this lets the level loops be unrolled with no per-iteration
///   strength reduction in the PTX.
/// * Only thread 0 writes the block sum, mirroring Hillis-Steele's last-
///   thread store but issued from the same thread that did the pivot
///   zero-out (no extra synchronisation needed).
pub fn compile_prefix_scan_kernel_blelloch() -> BoltResult<String> {
    // Single u32 buffer of BLOCK_SIZE entries. Blelloch does in-place
    // reads-and-writes against the same locations, so unlike Hillis-Steele
    // there is no ping-pong; the per-level `bar.sync` provides the ordering
    // between read and write across threads.
    let elem_bytes: u32 = 4;
    let shared_bytes_total: u32 = BLOCK_SIZE * elem_bytes;

    // log2(BLOCK_SIZE). BLOCK_SIZE is a compile-time pow2 by construction
    // (= 256), so this is exact.
    debug_assert!(BLOCK_SIZE.is_power_of_two(), "Blelloch requires pow2 BLOCK_SIZE");
    let k_levels: u32 = BLOCK_SIZE.trailing_zeros();

    let mut ptx = String::new();
    writeln!(ptx, "{}", PTX_VERSION).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_TARGET).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_ADDRESS_SIZE).map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(
        ptx,
        ".shared .align 4 .b8 sdata[{shared}];",
        shared = shared_bytes_total
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Signature mirrors the Hillis-Steele kernel exactly so the host launcher
    // can swap PTX without touching argument plumbing.
    writeln!(ptx, ".visible .entry {}(", SCAN_KERNEL_ENTRY_BLELLOCH).map_err(write_err)?;
    writeln!(
        ptx,
        "\t.param .u64 {}_param_0,",
        SCAN_KERNEL_ENTRY_BLELLOCH
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\t.param .u64 {}_param_1,",
        SCAN_KERNEL_ENTRY_BLELLOCH
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\t.param .u64 {}_param_2,",
        SCAN_KERNEL_ENTRY_BLELLOCH
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\t.param .u32 {}_param_3",
        SCAN_KERNEL_ENTRY_BLELLOCH
    )
    .map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Generous register pool: upsweep + downsweep each unroll to ~K levels
    // with a handful of address/value regs per level, but PTX `.reg` decls
    // only allocate names so over-sizing is free.
    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b16   %rs<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<48>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<48>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // -------- Indices: gid = ctaid.x * ntid.x + tid.x ; load n_rows.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u32 %r4, [{}_param_3];",
        SCAN_KERNEL_ENTRY_BLELLOCH
    )
    .map_err(write_err)?;

    // -------- Globalize parameter pointers.
    writeln!(
        ptx,
        "\tld.param.u64 %rd0, [{}_param_0];",
        SCAN_KERNEL_ENTRY_BLELLOCH
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u64 %rd1, [{}_param_1];",
        SCAN_KERNEL_ENTRY_BLELLOCH
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u64 %rd2, [{}_param_2];",
        SCAN_KERNEL_ENTRY_BLELLOCH
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;

    // -------- Load this thread's mask byte (0 if past the end). Normalise
    // non-zero bytes to 1 so the scan is robust to predicate kernels that
    // emit truthy values other than literal 1.
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r5, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra AFTER_LOAD;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd3, %r3, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd4, %rd0, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u8 %rs0, [%rd4];").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u16 %r6, %rs0;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p1, %r6, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.s32 %r5, 1, 0, %p1;").map_err(write_err)?;
    writeln!(ptx, "AFTER_LOAD:").map_err(write_err)?;

    // %r5 now holds the per-thread 0/1 value. Stash into arr[tid].
    //
    // Address arithmetic: we keep base = sdata (state-space `.shared`) in
    // %rd5 and per-thread `tid * 4` in %rd6. The thread's own slot is
    // %rd5 + %rd6 (stored in %rd7). For the upsweep/downsweep we need slot
    // pointers at offsets keyed off `idx`, computed per-level below. Keep
    // shared-state addresses (no `cvta.shared.u64`) so `st.shared.u32` /
    // `ld.shared.u32` see the right state-space qualifier.
    writeln!(ptx, "\tmov.u64 %rd5, sdata;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd6, %r2, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd7, %rd5, %rd6;").map_err(write_err)?; // arr[tid]
    writeln!(ptx, "\tst.shared.u32 [%rd7], %r5;").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;

    // ============================================================
    //   UPSWEEP (reduce tree). Levels d = 0 .. K-1.
    //
    //   stride       = 1 << (d + 1)   (2, 4, 8, ..., BLOCK_SIZE)
    //   half_stride  = 1 << d         (1, 2, 4, ..., BLOCK_SIZE/2)
    //   active mask  = tid < BLOCK_SIZE / stride
    //
    //   active thread does:
    //     idx_r = tid * stride + stride - 1
    //     idx_l = idx_r - half_stride
    //     arr[idx_r] += arr[idx_l]
    // ============================================================
    writeln!(ptx, "\t// ---- BLELLOCH UPSWEEP ----").map_err(write_err)?;
    for d in 0..k_levels {
        let stride: u32 = 1u32 << (d + 1);
        let half_stride: u32 = 1u32 << d;
        // Number of active threads at this level. Equivalently the upper
        // bound on `tid` such that `tid * stride + stride - 1 < BLOCK_SIZE`.
        let active_threads: u32 = BLOCK_SIZE / stride;

        writeln!(
            ptx,
            "\t// upsweep level d={d}, stride={stride}, half={half_stride}, active={active_threads}",
            d = d,
            stride = stride,
            half_stride = half_stride,
            active_threads = active_threads,
        )
        .map_err(write_err)?;

        // Predicate: skip if `tid >= active_threads`.
        writeln!(
            ptx,
            "\tsetp.ge.u32 %p2, %r2, {at};",
            at = active_threads
        )
        .map_err(write_err)?;
        writeln!(ptx, "\t@%p2 bra UPSWEEP_SKIP_{d};", d = d).map_err(write_err)?;

        // idx_r = tid * stride + (stride - 1); byte offset = idx_r * 4.
        // We compute the address directly to avoid a redundant intermediate.
        // %r10 = tid * stride
        writeln!(ptx, "\tmul.lo.u32 %r10, %r2, {stride};", stride = stride)
            .map_err(write_err)?;
        // %r11 = idx_r = %r10 + (stride - 1)
        writeln!(
            ptx,
            "\tadd.s32 %r11, %r10, {sm1};",
            sm1 = stride - 1
        )
        .map_err(write_err)?;
        // %r12 = idx_l = idx_r - half_stride
        writeln!(
            ptx,
            "\tsub.s32 %r12, %r11, {hs};",
            hs = half_stride
        )
        .map_err(write_err)?;
        // Byte offsets and shared addresses for idx_r / idx_l.
        writeln!(ptx, "\tmul.wide.u32 %rd10, %r11, 4;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd11, %rd5, %rd10;").map_err(write_err)?; // &arr[idx_r]
        writeln!(ptx, "\tmul.wide.u32 %rd12, %r12, 4;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd13, %rd5, %rd12;").map_err(write_err)?; // &arr[idx_l]
        // arr[idx_r] += arr[idx_l]
        writeln!(ptx, "\tld.shared.u32 %r13, [%rd11];").map_err(write_err)?;
        writeln!(ptx, "\tld.shared.u32 %r14, [%rd13];").map_err(write_err)?;
        writeln!(ptx, "\tadd.s32 %r15, %r13, %r14;").map_err(write_err)?;
        writeln!(ptx, "\tst.shared.u32 [%rd11], %r15;").map_err(write_err)?;

        writeln!(ptx, "UPSWEEP_SKIP_{d}:", d = d).map_err(write_err)?;
        writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    }

    // ============================================================
    //   PIVOT: capture inclusive block sum into a register on thread 0,
    //   then zero arr[BLOCK_SIZE-1] so the downsweep produces an
    //   exclusive scan.
    //
    //   Why thread 0? Any single thread works; thread 0 is convenient
    //   because it can also issue the final `block_sums[blockIdx.x]`
    //   store later (no extra barrier needed: only thread 0 reads %r20).
    // ============================================================
    writeln!(ptx, "\t// ---- BLELLOCH ZERO-INIT (exclusive-scan identity) ----")
        .map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p3, %r2, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra AFTER_PIVOT;").map_err(write_err)?;
    // Last-element address: %rd5 + (BLOCK_SIZE - 1) * 4
    writeln!(
        ptx,
        "\tadd.s64 %rd14, %rd5, {off};",
        off = (BLOCK_SIZE - 1) * elem_bytes
    )
    .map_err(write_err)?;
    // %r20 = arr[BLOCK_SIZE - 1] (the inclusive block sum). Survives in
    // thread 0's register file across the downsweep.
    writeln!(ptx, "\tld.shared.u32 %r20, [%rd14];").map_err(write_err)?;
    // arr[BLOCK_SIZE - 1] = 0
    writeln!(ptx, "\tmov.u32 %r21, 0;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd14], %r21;").map_err(write_err)?;
    writeln!(ptx, "AFTER_PIVOT:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;

    // ============================================================
    //   DOWNSWEEP. Levels d = K-1 .. 0 (largest stride first).
    //
    //   stride       = 1 << (d + 1)
    //   half_stride  = 1 << d
    //   active mask  = tid < BLOCK_SIZE / stride
    //
    //   active thread does:
    //     idx_r = tid * stride + stride - 1
    //     idx_l = idx_r - half_stride
    //     t          = arr[idx_l]
    //     arr[idx_l] = arr[idx_r]
    //     arr[idx_r] = arr[idx_r] + t
    // ============================================================
    writeln!(ptx, "\t// ---- BLELLOCH DOWNSWEEP ----").map_err(write_err)?;
    for d in (0..k_levels).rev() {
        let stride: u32 = 1u32 << (d + 1);
        let half_stride: u32 = 1u32 << d;
        let active_threads: u32 = BLOCK_SIZE / stride;

        writeln!(
            ptx,
            "\t// downsweep level d={d}, stride={stride}, half={half_stride}, active={active_threads}",
            d = d,
            stride = stride,
            half_stride = half_stride,
            active_threads = active_threads,
        )
        .map_err(write_err)?;

        writeln!(
            ptx,
            "\tsetp.ge.u32 %p4, %r2, {at};",
            at = active_threads
        )
        .map_err(write_err)?;
        writeln!(ptx, "\t@%p4 bra DOWNSWEEP_SKIP_{d};", d = d).map_err(write_err)?;

        // idx_r / idx_l + addresses (same shape as upsweep).
        writeln!(ptx, "\tmul.lo.u32 %r25, %r2, {stride};", stride = stride)
            .map_err(write_err)?;
        writeln!(
            ptx,
            "\tadd.s32 %r26, %r25, {sm1};",
            sm1 = stride - 1
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tsub.s32 %r27, %r26, {hs};",
            hs = half_stride
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tmul.wide.u32 %rd20, %r26, 4;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd21, %rd5, %rd20;").map_err(write_err)?; // &arr[idx_r]
        writeln!(ptx, "\tmul.wide.u32 %rd22, %r27, 4;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd23, %rd5, %rd22;").map_err(write_err)?; // &arr[idx_l]
        // t = arr[idx_l]
        writeln!(ptx, "\tld.shared.u32 %r28, [%rd23];").map_err(write_err)?;
        // r29 = arr[idx_r]
        writeln!(ptx, "\tld.shared.u32 %r29, [%rd21];").map_err(write_err)?;
        // arr[idx_l] = r29
        writeln!(ptx, "\tst.shared.u32 [%rd23], %r29;").map_err(write_err)?;
        // arr[idx_r] = r29 + t
        writeln!(ptx, "\tadd.s32 %r30, %r29, %r28;").map_err(write_err)?;
        writeln!(ptx, "\tst.shared.u32 [%rd21], %r30;").map_err(write_err)?;

        writeln!(ptx, "DOWNSWEEP_SKIP_{d}:", d = d).map_err(write_err)?;
        writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    }

    // -------- Each in-range thread writes arr[tid] (its exclusive prefix)
    // to local_indices[gid]. Mirrors the Hillis-Steele tail.
    writeln!(ptx, "\tld.shared.u32 %r35, [%rd7];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p5, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra AFTER_LOCAL_STORE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd30, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd31, %rd1, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd31], %r35;").map_err(write_err)?;
    writeln!(ptx, "AFTER_LOCAL_STORE:").map_err(write_err)?;

    // -------- Thread 0 (the pivot thread) writes block_sums[blockIdx.x]
    // from %r20, which captured arr[BLOCK_SIZE - 1] just before the
    // zero-out. For partial last blocks, out-of-range lanes seeded 0, so
    // the inclusive sum is still the correct kept-row count.
    writeln!(ptx, "\tsetp.ne.s32 %p6, %r2, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p6 bra DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd32, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd33, %rd2, %rd32;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd33], %r20;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Generate the single-pass **decoupled-lookback** prefix-scan PTX module.
///
/// ## Why
///
/// The Hillis-Steele and Blelloch kernels compute per-block exclusive scans
/// and `block_sums`; the global prefix step then requires a host
/// download + exclusive-scan + re-upload of `block_sums`. That round trip
/// costs a `cudaMemcpy` each way plus a kernel sync. The decoupled-lookback
/// algorithm (Merrill & Garland, NVR-2016-002) folds the global pass into the
/// same kernel launch by having each block publish its own aggregate to a
/// status array and then walk previous slots backwards until it finds the
/// nearest already-INCLUSIVE block, summing intervening aggregates as it
/// goes. The result: one kernel launch, no host round-trip, and the
/// resulting `local_indices[gid]` already holds the global exclusive prefix.
///
/// ## ABI
///
/// See [`SCAN_KERNEL_ENTRY_LOOKBACK`] for the full 5-arg signature
/// (`mask`, `local_indices`, `block_sums`, `n_rows`, `partial_status`).
///
/// Status-slot encoding (a single `u32` per block):
///
/// ```text
///   31:30  status (0 = INVALID, 1 = AGGREGATE, 2 = INCLUSIVE)
///   29:0   value  (block aggregate OR inclusive global prefix)
/// ```
///
/// See [`LOOKBACK_VALUE_MASK`], [`LOOKBACK_STATUS_INVALID`],
/// [`LOOKBACK_STATUS_AGGREGATE`], [`LOOKBACK_STATUS_INCLUSIVE`].
///
/// ## Algorithm
///
/// 1. **Block scan**. Re-uses the Hillis-Steele ping-pong scan from
///    [`compile_prefix_scan_kernel`] to compute the per-block exclusive
///    prefix in `local_indices_excl` (low) plus the block's inclusive sum
///    `block_aggregate`.
///
/// 2. **PUBLISH_AGGREGATE** (thread 0). Stores the block aggregate to
///    `partial_status[blockIdx.x]` tagged with `STATUS_AGGREGATE`, then
///    issues `membar.gl` so the publication is observable to peers BEFORE
///    we start reading their slots.
///
/// 3. **LOOKBACK_SPIN** (thread 0). For `pred = blockIdx.x - 1` down to
///    `0`, spin-loads `partial_status[pred]` with `ld.acquire.gpu.u32`
///    until the status is non-INVALID, then:
///       * if INCLUSIVE: add the inclusive value, **break** (we've
///         absorbed every prior block in one step);
///       * if AGGREGATE: add the aggregate and continue to `pred - 1`.
///    The result accumulates `block_prefix` (the exclusive global prefix
///    for this block). Block 0 simply skips the loop with `block_prefix = 0`.
///
/// 4. **PUBLISH_INCLUSIVE** (thread 0). Stores
///    `(block_prefix + block_aggregate)` to `partial_status[blockIdx.x]`
///    tagged with `STATUS_INCLUSIVE`, then `membar.gl`. Stashes
///    `block_prefix` in a shared scalar so other threads can read it after
///    the broadcast barrier.
///
/// 5. **BROADCAST**. `bar.sync 0` so every thread in the block has visibility
///    on `block_prefix_slot`. Each in-range thread writes
///    `local_indices[gid] = within_block_excl + block_prefix` (i.e. the
///    GLOBAL exclusive prefix). Thread `BLOCK_SIZE-1` also stores the
///    block aggregate into `block_sums[blockIdx.x]` for diagnostic
///    compatibility with the other two kernels (the host-side caller does
///    not consume it).
///
/// ## Correctness notes
///
/// * **Value bit budget**: the low-30-bits encoding caps the cumulative
///   prefix at `(1 << 30) - 1 = 1_073_741_823`. The host-side caller
///   ([`crate::exec::gpu_compact::prefix_scan_mask_lookback`]) refuses
///   `n_rows >= (1 << 30)` so the prefix cannot saturate.
/// * **Aggregate-first publish**. Publishing AGGREGATE *before* lookback
///   ensures every block's predecessor walk is well-founded: peers can
///   always make progress because the previous block has already advertised
///   at least its own aggregate.
/// * **membar.gl on the publisher + ld.acquire on the reader** together
///   provide the cross-CTA ordering guarantee on sm_70+. We use the GPU
///   scope (`.gpu` qualifier) on the load because peer blocks may execute
///   on different SMs.
/// * **Single-thread lookback** keeps the algorithm simple. A warp-wide
///   lookback (one thread per `pred`, ballot to find the leftmost
///   INCLUSIVE) is a known optimisation; we defer it to a follow-up.
/// * **Out-of-range lanes** in a partial last block contribute zero to the
///   block aggregate (the existing AFTER_LOAD predicate handles it), so
///   the published aggregate is correct for any `n_rows`.
///
/// ## Host launch contract — forward-progress / no-deadlock (review C-7 / E §2)
///
/// The LOOKBACK_SPIN step (algorithm step 3) spins on a *predecessor* block's
/// `partial_status[blockIdx.x - 1]` with `ld.acquire.gpu.u32` and has **no
/// timeout / fallback**. Forward progress therefore depends on every block
/// being able to make progress without waiting on a block that has not been
/// scheduled. This is only guaranteed when the whole grid is **co-resident** —
/// i.e. the launch is a single occupancy-bounded wave. The host launch site
/// MUST enforce, before selecting this kernel:
///
/// 1. **Single-wave occupancy bound.** `gridDim.x <= max_resident_blocks`,
///    where
///    `max_resident_blocks = num_SMs * maxActiveBlocksPerSM`
///    for THIS kernel at the chosen `blockDim` and shared-mem usage (query via
///    `cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, blockDim,
///    smem)` then `* deviceProp.multiProcessorCount`). If `gridDim.x` would
///    exceed that bound, a predecessor block may never be scheduled while a
///    successor spins on its status slot — a **deadlock**. The launch site
///    must instead fall back to the multipass scan
///    ([`compile_prefix_scan_kernel`] / [`compile_prefix_scan_kernel_blelloch`]
///    driven by the host download-scan-upload path, exposed in
///    `gpu_compact` as `prefix_scan_multipass`).
///
/// 2. **`n_rows <= i32::MAX`.** The global thread id is computed in **s32**
///    (`mad.lo.s32 %r3, %ctaid.x, %ntid.x, %tid.x`). For `n_rows > i32::MAX`
///    the id wraps negative and mis-addresses the mask / index buffers. Every
///    ptx_gen-/prefix-scan-emitted kernel shares this cap (see the C-3/C-4
///    note in `ptx_gen::compile`); the launch site must assert it.
///
/// 3. **Value bit budget** (also a correctness note above):
///    `n_rows < (1 << 30)` so the 30-bit prefix value cannot saturate.
///
/// The host helper [`lookback_launch_is_safe`] encodes bounds (1)–(3) as a
/// single predicate the launch site can `debug_assert!` on and branch to the
/// multipass fallback when it returns `false`.
pub fn compile_prefix_scan_kernel_lookback() -> BoltResult<String> {
    // Re-use Hillis-Steele's shared-memory layout for the block scan: two
    // ping-pong u32 buffers of BLOCK_SIZE entries each. We also need ONE
    // extra u32 of shared scratch to broadcast `block_prefix` from thread 0
    // to every lane after the lookback completes, and ONE extra u32 to
    // stash the block aggregate (computed by thread BLOCK_SIZE-1) where
    // thread 0 can read it.
    let elem_bytes: u32 = 4;
    let shared_bytes_per_buf: u32 = BLOCK_SIZE * elem_bytes;
    let shared_scan_bytes: u32 = shared_bytes_per_buf * 2;
    // Offset of the broadcast slot (`block_prefix_slot`) in `sdata`.
    let block_prefix_off: u32 = shared_scan_bytes;
    // Offset of the aggregate slot (`block_aggregate_slot`) in `sdata`.
    let block_aggregate_off: u32 = shared_scan_bytes + 4;
    let shared_bytes_total: u32 = shared_scan_bytes + 8;

    let mut ptx = String::new();
    writeln!(ptx, "{}", PTX_VERSION).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_TARGET).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_ADDRESS_SIZE).map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(
        ptx,
        ".shared .align 4 .b8 sdata[{shared}];",
        shared = shared_bytes_total
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Signature: mask, local_indices, block_sums, n_rows, partial_status.
    writeln!(ptx, ".visible .entry {}(", SCAN_KERNEL_ENTRY_LOOKBACK).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", SCAN_KERNEL_ENTRY_LOOKBACK).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", SCAN_KERNEL_ENTRY_LOOKBACK).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_2,", SCAN_KERNEL_ENTRY_LOOKBACK).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_3,", SCAN_KERNEL_ENTRY_LOOKBACK).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_4", SCAN_KERNEL_ENTRY_LOOKBACK).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Register pool. The lookback adds a spin loop with status decoding;
    // size the pool generously to cover it plus the inherited Hillis-Steele
    // body.
    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b16   %rs<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<64>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<64>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // -------- Indices: tid_x = %tid.x ; ctaid = %ctaid.x ; gid = ctaid*ntid+tid
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    // n_rows
    writeln!(
        ptx,
        "\tld.param.u32 %r4, [{}_param_3];",
        SCAN_KERNEL_ENTRY_LOOKBACK
    )
    .map_err(write_err)?;

    // -------- Globalize parameter pointers.
    writeln!(
        ptx,
        "\tld.param.u64 %rd0, [{}_param_0];",
        SCAN_KERNEL_ENTRY_LOOKBACK
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u64 %rd1, [{}_param_1];",
        SCAN_KERNEL_ENTRY_LOOKBACK
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u64 %rd2, [{}_param_2];",
        SCAN_KERNEL_ENTRY_LOOKBACK
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u64 %rd40, [{}_param_4];",
        SCAN_KERNEL_ENTRY_LOOKBACK
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd40, %rd40;").map_err(write_err)?;

    // -------- Load this thread's mask byte (0 if past the end). Mirror the
    // Hillis-Steele normalisation: any non-zero byte counts as `1`.
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r5, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra AFTER_LOAD;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd3, %r3, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd4, %rd0, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u8 %rs0, [%rd4];").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u16 %r6, %rs0;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p1, %r6, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.s32 %r5, 1, 0, %p1;").map_err(write_err)?;
    writeln!(ptx, "AFTER_LOAD:").map_err(write_err)?;

    // -------- Stash own 0/1 value into ping buffer (sdata[0..BLOCK_SIZE]).
    writeln!(ptx, "\tmov.u64 %rd5, sdata;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tadd.s64 %rd6, %rd5, {off};",
        off = shared_bytes_per_buf
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd7, %r2, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd8, %rd5, %rd7;").map_err(write_err)?; // ping addr
    writeln!(ptx, "\tadd.s64 %rd9, %rd6, %rd7;").map_err(write_err)?; // pong addr
    writeln!(ptx, "\tst.shared.u32 [%rd8], %r5;").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;

    // -------- Hillis-Steele block scan (same as the default kernel).
    let mut offset: u32 = 1;
    let mut step: usize = 0;
    while offset < BLOCK_SIZE {
        let off_bytes = offset * elem_bytes;
        writeln!(ptx, "\tld.shared.u32 %r7, [%rd8];").map_err(write_err)?;
        writeln!(
            ptx,
            "\tsetp.lt.s32 %p{p}, %r2, {off};",
            p = 2 + (step % 4),
            off = offset
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tmov.u32 %r8, 0;").map_err(write_err)?;
        writeln!(
            ptx,
            "\t@%p{p} bra LB_SKIP_LOAD_{step};",
            p = 2 + (step % 4),
            step = step
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tsub.s64 %rd10, %rd8, {off};",
            off = off_bytes
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tld.shared.u32 %r8, [%rd10];").map_err(write_err)?;
        writeln!(ptx, "LB_SKIP_LOAD_{step}:", step = step).map_err(write_err)?;
        writeln!(ptx, "\tadd.s32 %r9, %r7, %r8;").map_err(write_err)?;
        writeln!(ptx, "\tst.shared.u32 [%rd9], %r9;").map_err(write_err)?;
        writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
        // Swap ping/pong pointers via tmp.
        writeln!(ptx, "\tmov.u64 %rd11, %rd8;").map_err(write_err)?;
        writeln!(ptx, "\tmov.u64 %rd8, %rd9;").map_err(write_err)?;
        writeln!(ptx, "\tmov.u64 %rd9, %rd11;").map_err(write_err)?;
        offset <<= 1;
        step += 1;
    }

    // %r10 = inclusive scan at this thread; %r11 = exclusive (incl - own).
    writeln!(ptx, "\tld.shared.u32 %r10, [%rd8];").map_err(write_err)?;
    writeln!(ptx, "\tsub.s32 %r11, %r10, %r5;").map_err(write_err)?;

    // -------- Thread (BLOCK_SIZE - 1) publishes the block aggregate to a
    // shared slot so thread 0 can read it (different lanes own those two
    // duties; the slot lets us avoid a warp shuffle).
    writeln!(
        ptx,
        "\tsetp.ne.s32 %p5, %r2, {last};",
        last = BLOCK_SIZE - 1
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra AFTER_AGG_STASH;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tadd.s64 %rd12, %rd5, {off};",
        off = block_aggregate_off
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd12], %r10;").map_err(write_err)?;
    // Diagnostic compatibility: also publish to block_sums[blockIdx.x].
    writeln!(ptx, "\tmul.wide.u32 %rd13, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd14, %rd2, %rd13;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd14], %r10;").map_err(write_err)?;
    writeln!(ptx, "AFTER_AGG_STASH:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;

    // ============================================================
    //   Thread 0: PUBLISH_AGGREGATE, LOOKBACK_SPIN, PUBLISH_INCLUSIVE.
    //   Every other lane jumps straight to BROADCAST.
    // ============================================================
    writeln!(ptx, "\tsetp.ne.s32 %p6, %r2, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p6 bra BROADCAST;").map_err(write_err)?;

    // ---- Read block aggregate from the shared slot (own thread did the
    // bar.sync above, so this read is safe).
    writeln!(
        ptx,
        "\tadd.s64 %rd15, %rd5, {off};",
        off = block_aggregate_off
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r20, [%rd15];").map_err(write_err)?;

    // ---- Compute &partial_status[blockIdx.x] in %rd16.
    writeln!(ptx, "\tmul.wide.u32 %rd17, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd16, %rd40, %rd17;").map_err(write_err)?;

    // ---- PUBLISH_AGGREGATE: pack (1 << 30) | (aggregate & 0x3FFFFFFF).
    writeln!(ptx, "PUBLISH_AGGREGATE:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tand.b32 %r21, %r20, {mask};",
        mask = LOOKBACK_VALUE_MASK
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tor.b32 %r22, %r21, {flag};",
        flag = LOOKBACK_STATUS_AGGREGATE << 30
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd16], %r22;").map_err(write_err)?;
    // Make the publication observable to peer CTAs BEFORE we start reading
    // their slots.
    writeln!(ptx, "\tmembar.gl;").map_err(write_err)?;

    // ---- LOOKBACK: pred = blockIdx.x - 1 ; block_prefix = 0.
    //   while pred >= 0:
    //     LOOKBACK_SPIN: status = ld.acquire.gpu.u32 [partial_status + pred*4]
    //       if status == INVALID: spin
    //       else: value = status & 0x3FFFFFFF
    //             block_prefix += value
    //             if (status >> 30) == INCLUSIVE: break
    //             pred -= 1
    //
    // Block 0 (%r0 == 0) skips this loop entirely with block_prefix = 0.
    writeln!(ptx, "\tmov.u32 %r30, 0;").map_err(write_err)?; // block_prefix
    writeln!(ptx, "\tsetp.eq.s32 %p7, %r0, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p7 bra AFTER_LOOKBACK;").map_err(write_err)?;
    writeln!(ptx, "\tsub.s32 %r31, %r0, 1;").map_err(write_err)?; // pred

    writeln!(ptx, "LOOKBACK_OUTER:").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd18, %r31, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd19, %rd40, %rd18;").map_err(write_err)?;
    writeln!(ptx, "LOOKBACK_SPIN:").map_err(write_err)?;
    // Acquire load gives the matching ordering for the publisher's
    // membar.gl + st.global on the same address. `.gpu` scope because peers
    // may live on a different SM.
    writeln!(ptx, "\tld.acquire.gpu.u32 %r32, [%rd19];").map_err(write_err)?;
    writeln!(ptx, "\tshr.u32 %r33, %r32, 30;").map_err(write_err)?; // status
    writeln!(
        ptx,
        "\tsetp.eq.s32 %p8, %r33, {inv};",
        inv = LOOKBACK_STATUS_INVALID
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p8 bra LOOKBACK_SPIN;").map_err(write_err)?;
    // Non-INVALID -> accumulate value into block_prefix.
    writeln!(
        ptx,
        "\tand.b32 %r34, %r32, {mask};",
        mask = LOOKBACK_VALUE_MASK
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s32 %r30, %r30, %r34;").map_err(write_err)?;
    // INCLUSIVE -> done.
    writeln!(
        ptx,
        "\tsetp.eq.s32 %p9, %r33, {inc};",
        inc = LOOKBACK_STATUS_INCLUSIVE
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p9 bra AFTER_LOOKBACK;").map_err(write_err)?;
    // AGGREGATE -> walk further left. Stop after pred == 0.
    writeln!(ptx, "\tsetp.eq.s32 %p10, %r31, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p10 bra AFTER_LOOKBACK;").map_err(write_err)?;
    writeln!(ptx, "\tsub.s32 %r31, %r31, 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOKBACK_OUTER;").map_err(write_err)?;
    writeln!(ptx, "AFTER_LOOKBACK:").map_err(write_err)?;

    // ---- PUBLISH_INCLUSIVE: (block_prefix + block_aggregate) tagged 0b10.
    writeln!(ptx, "PUBLISH_INCLUSIVE:").map_err(write_err)?;
    writeln!(ptx, "\tadd.s32 %r35, %r30, %r20;").map_err(write_err)?; // incl_value
    writeln!(
        ptx,
        "\tand.b32 %r36, %r35, {mask};",
        mask = LOOKBACK_VALUE_MASK
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tor.b32 %r37, %r36, {flag};",
        flag = LOOKBACK_STATUS_INCLUSIVE << 30
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd16], %r37;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.gl;").map_err(write_err)?;

    // Stash block_prefix into the shared broadcast slot so every lane can
    // read it after the upcoming bar.sync.
    writeln!(
        ptx,
        "\tadd.s64 %rd20, %rd5, {off};",
        off = block_prefix_off
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd20], %r30;").map_err(write_err)?;

    // ============================================================
    //   BROADCAST: every thread reads block_prefix and writes the global
    //   exclusive prefix to local_indices[gid].
    // ============================================================
    writeln!(ptx, "BROADCAST:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;

    writeln!(
        ptx,
        "\tadd.s64 %rd21, %rd5, {off};",
        off = block_prefix_off
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r40, [%rd21];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s32 %r41, %r11, %r40;").map_err(write_err)?;

    // Only in-range threads write.
    writeln!(ptx, "\tsetp.ge.u32 %p11, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p11 bra DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd22, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd22;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd23], %r41;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Exclusive upper bound on `n_rows` for the decoupled-lookback kernel imposed
/// by its 30-bit `partial_status` value field: the cumulative prefix must fit
/// in `LOOKBACK_VALUE_MASK` (`< 1 << 30`), so a launch with `n_rows >=
/// LOOKBACK_MAX_ROWS` could saturate the prefix and MUST take the multipass
/// fallback. See the "Host launch contract" section on
/// [`compile_prefix_scan_kernel_lookback`].
pub const LOOKBACK_MAX_ROWS: u32 = 1 << 30;

/// Forward-progress / no-deadlock guard for the decoupled-lookback scan
/// (review C-7 / E §2).
///
/// Returns `true` iff it is safe to launch [`SCAN_KERNEL_ENTRY_LOOKBACK`] with
/// `grid_dim_x` blocks over `n_rows` rows given `max_resident_blocks` (the
/// occupancy-bounded co-residency capacity, `num_SMs * maxActiveBlocksPerSM`
/// for this kernel at the launch `blockDim`). The host launch site MUST gate
/// on this predicate and, when it returns `false`, fall back to the multipass
/// scan (`gpu_compact::prefix_scan_multipass`) instead of the single-pass
/// lookback kernel:
///
/// ```ignore
/// // At the lookback launch site (in the executor that owns the launch):
/// debug_assert!(
///     prefix_scan::lookback_launch_is_safe(grid_dim_x, max_resident_blocks, n_rows),
///     "lookback scan grid {grid_dim_x} / n_rows {n_rows} violates the \
///      forward-progress contract; would deadlock or saturate"
/// );
/// if !prefix_scan::lookback_launch_is_safe(grid_dim_x, max_resident_blocks, n_rows) {
///     return prefix_scan_multipass(/* … */);
/// }
/// ```
///
/// The three bounds checked, in order:
///
/// 1. **Co-residency**: `grid_dim_x <= max_resident_blocks` — every block must
///    be resident so a successor never spins on an unscheduled predecessor's
///    status slot (the deadlock case). A `max_resident_blocks == 0` query
///    result (no capacity reported) is treated as unsafe.
/// 2. **Row addressing**: `n_rows <= i32::MAX` — the kernel computes the global
///    thread id in s32; larger counts wrap negative and mis-address.
/// 3. **Value budget**: `n_rows < LOOKBACK_MAX_ROWS` — the 30-bit prefix value
///    field must not saturate.
///
/// Pure host arithmetic (no CUDA context), so the launch-site owner gets a
/// single tested predicate rather than re-deriving the three bounds inline.
#[must_use]
pub fn lookback_launch_is_safe(grid_dim_x: u32, max_resident_blocks: u32, n_rows: usize) -> bool {
    grid_dim_x <= max_resident_blocks
        && max_resident_blocks > 0
        && n_rows <= i32::MAX as usize
        && (n_rows as u64) < LOOKBACK_MAX_ROWS as u64
}

/// Generate the per-dtype gather PTX module.
///
/// ABI (per `<dtype>`):
/// ```text
/// .visible .entry bolt_gather_<dtype>(
///     .param .u64 ..._param_0,   // mask_ptr          (u8*)
///     .param .u64 ..._param_1,   // local_indices_ptr (u32*)
///     .param .u64 ..._param_2,   // block_bases_ptr   (u32*)
///     .param .u64 ..._param_3,   // input_ptr         (T*)
///     .param .u64 ..._param_4,   // output_ptr        (T*)
///     .param .u32 ..._param_5    // n_rows
/// )
/// ```
///
/// Each thread reads its mask byte; on `m != 0` it computes
/// `idx = block_bases[blockIdx.x] + local_indices[gid]` and copies
/// `input[gid]` to `output[idx]`.
pub fn compile_gather_kernel(dtype: DataType) -> BoltResult<String> {
    let entry = gather_kernel_entry(dtype);
    let elem_bytes = dtype.byte_width().ok_or_else(|| {
        BoltError::Other(format!(
            "prefix_scan: gather not supported for variable-width dtype {:?}",
            dtype
        ))
    })?;
    if matches!(dtype, DataType::Utf8) {
        return Err(BoltError::Other(
            "prefix_scan: gather Utf8 not supported (variable-width)".into(),
        ));
    }
    let (ld_suffix, reg_class, reg_ty) = gather_type_info(dtype)?;

    let mut ptx = String::new();
    writeln!(ptx, "{}", PTX_VERSION).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_TARGET).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_ADDRESS_SIZE).map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {}(", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_2,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_3,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_4,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_5", entry).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b16   %rs<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<24>;").map_err(write_err)?;
    // The value register class for Bool / Int32 reuses one of the generic
    // decls above (`rs` and `r` respectively), so we only emit a fresh decl
    // when the class differs (rl for i64, f for f32, fd for f64).
    if reg_class != "rs" && reg_class != "r" && reg_class != "rd" {
        writeln!(ptx, "\t.reg .{ty}   %{rc}<4>;", ty = reg_ty, rc = reg_class)
            .map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    // gid = ctaid.x * ntid.x + tid.x
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{}_param_5];", entry).map_err(write_err)?;
    // Bail out-of-range.
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // Globalize the five pointer params.
    writeln!(ptx, "\tld.param.u64 %rd0, [{}_param_0];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd1, [{}_param_1];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd2, [{}_param_2];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd3, [{}_param_3];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd4, [{}_param_4];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;

    // Load mask byte; bail if zero. Mask, local_indices, block_bases and the
    // input column are all read-only inputs (the host side allocates them as
    // distinct GpuVec buffers and the planner never aliases them with the
    // output), so route the loads through the read-only cache via `ld.global.nc`.
    writeln!(ptx, "\tmul.wide.u32 %rd5, %r3, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd6, %rd0, %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u8 %rs0, [%rd6];").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u16 %r5, %rs0;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p1, %r5, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra DONE;").map_err(write_err)?;

    // local_idx = local_indices[gid]
    writeln!(ptx, "\tmul.wide.u32 %rd7, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd8, %rd1, %rd7;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u32 %r6, [%rd8];").map_err(write_err)?;

    // block_base = block_bases[blockIdx.x]
    writeln!(ptx, "\tmul.wide.u32 %rd9, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd10, %rd2, %rd9;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u32 %r7, [%rd10];").map_err(write_err)?;

    // out_idx = block_base + local_idx
    writeln!(ptx, "\tadd.s32 %r8, %r6, %r7;").map_err(write_err)?;

    // input_addr  = input  + gid * elem_bytes
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd11, %r3, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd3, %rd11;").map_err(write_err)?;

    // output_addr = output + out_idx * elem_bytes
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd13, %r8, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd14, %rd4, %rd13;").map_err(write_err)?;

    // value = *input_addr ; *output_addr = value
    writeln!(
        ptx,
        "\tld.global.nc.{ld} %{rc}0, [%rd12];",
        ld = ld_suffix,
        rc = reg_class
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tst.global.{ld} [%rd14], %{rc}0;",
        ld = ld_suffix,
        rc = reg_class
    )
    .map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// `(ld/st suffix, register class, .reg type)` triple for gather.
///
/// The suffix is used for both loads and stores of the value (e.g. `s32` for
/// Int32). The register class matches `ptx_gen.rs`'s conventions so the same
/// `.reg` decls work.
fn gather_type_info(dtype: DataType) -> BoltResult<(&'static str, &'static str, &'static str)> {
    Ok(match dtype {
        DataType::Bool => ("u8", "rs", "b16"),
        DataType::Int32 => ("s32", "r", "b32"),
        DataType::Int64 => ("s64", "rl", "b64"),
        DataType::Float32 => ("f32", "f", "f32"),
        DataType::Float64 => ("f64", "fd", "f64"),
        DataType::Utf8 => {
            return Err(BoltError::Other(
                "prefix_scan: gather Utf8 not supported (variable-width)".into(),
            ))
        }
        DataType::Decimal128(_, _) => {
            return Err(BoltError::Plan(
                "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
            ))
        }
        // v0.7: temporal gather lowers to integer gather on the underlying
        // days / ticks. Same register classes as Int32 / Int64.
        DataType::Date32 => ("s32", "r", "b32"),
        DataType::Timestamp(_, _) => ("s64", "rl", "b64"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_ptx_has_shape() {
        let ptx = compile_prefix_scan_kernel().expect("scan PTX compiles");

        // Header.
        assert!(ptx.contains(".version 7.5"));
        assert!(ptx.contains(".target sm_70"));
        assert!(ptx.contains(".address_size 64"));

        // Shared-memory scratchpad (two BLOCK_SIZE * 4 byte buffers).
        let total = (BLOCK_SIZE * 4 * 2) as u64;
        assert!(
            ptx.contains(&format!(".shared .align 4 .b8 sdata[{total}]")),
            "shared decl missing or wrong size; got:\n{ptx}"
        );

        // Signature.
        assert!(ptx.contains(".visible .entry bolt_prefix_scan("));
        assert!(ptx.contains(".param .u64 bolt_prefix_scan_param_0,"));
        assert!(ptx.contains(".param .u64 bolt_prefix_scan_param_1,"));
        assert!(ptx.contains(".param .u64 bolt_prefix_scan_param_2,"));
        assert!(ptx.contains(".param .u32 bolt_prefix_scan_param_3"));

        // Hillis-Steele: one bar.sync per round + the seed sync. BLOCK_SIZE=256
        // -> log2(256)=8 rounds, plus one for the initial load.
        let n_sync = ptx.matches("bar.sync 0;").count();
        assert!(
            n_sync >= 9,
            "expected >=9 bar.syncs (one seed + 8 rounds), got {n_sync}"
        );

        // Shared-memory ops + the final inclusive->exclusive subtract.
        assert!(ptx.contains("ld.shared.u32"), "missing ld.shared.u32");
        assert!(ptx.contains("st.shared.u32"), "missing st.shared.u32");
        assert!(
            ptx.contains("sub.s32 %r11, %r10, %r5;"),
            "missing inclusive->exclusive subtract"
        );

        // Block-sum store at thread (BLOCK_SIZE-1).
        assert!(ptx.contains(&format!("setp.ne.s32 %p6, %r2, {};", BLOCK_SIZE - 1)));
        assert!(ptx.contains("st.global.u32 [%rd15], %r10;"));

        assert!(ptx.contains("DONE:"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn gather_i32_ptx_has_shape() {
        let ptx = compile_gather_kernel(DataType::Int32).expect("i32 gather PTX compiles");

        assert!(ptx.contains(".visible .entry bolt_gather_i32("));
        // Six params: mask, local_indices, block_bases, input, output, n_rows.
        for i in 0..=4 {
            assert!(
                ptx.contains(&format!(".param .u64 bolt_gather_i32_param_{i},")),
                "missing u64 param {i}"
            );
        }
        assert!(ptx.contains(".param .u32 bolt_gather_i32_param_5"));

        // i32 load/store uses .s32. Input load is routed through the read-only
        // cache (`ld.global.nc`); output store stays as `st.global`.
        assert!(ptx.contains("ld.global.nc.s32"), "missing typed input load");
        assert!(ptx.contains("st.global.s32"), "missing typed output store");

        // Mask gate (read-only-cache load).
        assert!(ptx.contains("ld.global.nc.u8 %rs0,"));
        assert!(
            ptx.contains("setp.eq.s32 %p1, %r5, 0;"),
            "missing mask==0 short-circuit"
        );

        // Out-of-range guard at the top.
        assert!(ptx.contains("setp.ge.u32 %p0, %r3, %r4;"));
    }

    #[test]
    fn gather_entry_names_match_dtype() {
        assert_eq!(gather_kernel_entry(DataType::Bool), "bolt_gather_bool");
        assert_eq!(gather_kernel_entry(DataType::Int32), "bolt_gather_i32");
        assert_eq!(gather_kernel_entry(DataType::Int64), "bolt_gather_i64");
        assert_eq!(gather_kernel_entry(DataType::Float32), "bolt_gather_f32");
        assert_eq!(gather_kernel_entry(DataType::Float64), "bolt_gather_f64");

        // Every supported dtype's PTX should use the matching entry name.
        for dtype in [
            DataType::Bool,
            DataType::Int32,
            DataType::Int64,
            DataType::Float32,
            DataType::Float64,
        ] {
            let ptx = compile_gather_kernel(dtype).unwrap_or_else(|e| {
                panic!("compile_gather_kernel({:?}) failed: {}", dtype, e)
            });
            let entry = gather_kernel_entry(dtype);
            assert!(
                ptx.contains(&format!(".visible .entry {entry}(")),
                "{dtype:?} PTX missing entry {entry}"
            );
        }
    }

    #[test]
    fn gather_rejects_utf8() {
        let err = compile_gather_kernel(DataType::Utf8)
            .expect_err("Utf8 gather must error: variable-width");
        assert!(format!("{}", err).contains("Utf8"));
    }

    /// Structural smoke test for the Blelloch variant: header, signature,
    /// shared-memory size, and the upsweep/downsweep/pivot markers must all
    /// be present. We do not run the kernel here — it's compiled and
    /// inspected as a string.
    #[test]
    fn blelloch_scan_ptx_has_shape() {
        let ptx = compile_prefix_scan_kernel_blelloch().expect("blelloch PTX compiles");

        // Header.
        assert!(ptx.contains(".version 7.5"));
        assert!(ptx.contains(".target sm_70"));
        assert!(ptx.contains(".address_size 64"));

        // Single buffer (no ping-pong): BLOCK_SIZE * 4 bytes.
        let total = (BLOCK_SIZE * 4) as u64;
        assert!(
            ptx.contains(&format!(".shared .align 4 .b8 sdata[{total}]")),
            "shared decl missing or wrong size; got:\n{ptx}"
        );

        // Signature must match the Blelloch entry name and the same 4-arg
        // ABI as Hillis-Steele.
        assert!(ptx.contains(".visible .entry bolt_prefix_scan_blelloch("));
        assert!(ptx.contains(".param .u64 bolt_prefix_scan_blelloch_param_0,"));
        assert!(ptx.contains(".param .u64 bolt_prefix_scan_blelloch_param_1,"));
        assert!(ptx.contains(".param .u64 bolt_prefix_scan_blelloch_param_2,"));
        assert!(ptx.contains(".param .u32 bolt_prefix_scan_blelloch_param_3"));

        // Algorithm-section markers we emit as comments so reviewers and this
        // test can both confirm the upsweep and downsweep halves are wired
        // up. If these vanish we either dropped a phase or renamed it
        // unexpectedly.
        assert!(
            ptx.contains("BLELLOCH UPSWEEP"),
            "missing upsweep marker comment:\n{ptx}"
        );
        assert!(
            ptx.contains("BLELLOCH DOWNSWEEP"),
            "missing downsweep marker comment:\n{ptx}"
        );
        assert!(
            ptx.contains("BLELLOCH ZERO-INIT"),
            "missing exclusive-scan zero-init marker:\n{ptx}"
        );

        // Per-level skip labels: BLOCK_SIZE = 256 -> log2 = 8 levels per phase.
        // Each level emits UPSWEEP_SKIP_<d>: / DOWNSWEEP_SKIP_<d>: labels.
        for d in 0..BLOCK_SIZE.trailing_zeros() {
            assert!(
                ptx.contains(&format!("UPSWEEP_SKIP_{d}:")),
                "missing UPSWEEP_SKIP_{d}: label\n{ptx}"
            );
            assert!(
                ptx.contains(&format!("DOWNSWEEP_SKIP_{d}:")),
                "missing DOWNSWEEP_SKIP_{d}: label\n{ptx}"
            );
        }

        // Barrier accounting. The Blelloch pattern is:
        //   * one bar.sync after the seed store
        //   * one bar.sync per upsweep level (K = log2(BLOCK_SIZE))
        //   * one bar.sync after the pivot zero-init
        //   * one bar.sync per downsweep level (K)
        // Total = 2 * K + 2 = 2*8 + 2 = 18.
        let k = BLOCK_SIZE.trailing_zeros();
        let expected_syncs = (2 * k + 2) as usize;
        let n_sync = ptx.matches("bar.sync 0;").count();
        assert_eq!(
            n_sync, expected_syncs,
            "expected {expected_syncs} bar.syncs (seed + K upsweep + pivot + K downsweep, K={k}), got {n_sync}\n{ptx}"
        );

        // Shared-memory ops must be present (this is the load-bearing
        // mnemonic for both phases).
        assert!(ptx.contains("ld.shared.u32"), "missing ld.shared.u32");
        assert!(ptx.contains("st.shared.u32"), "missing st.shared.u32");

        // Thread 0 captures the inclusive sum into %r20 before zero-out and
        // the same register feeds the final block_sums store. Pin both ends
        // of that dataflow so a refactor that drops one side surfaces here.
        assert!(
            ptx.contains("ld.shared.u32 %r20, [%rd14];"),
            "missing pivot capture of inclusive block sum\n{ptx}"
        );
        assert!(
            ptx.contains("st.global.u32 [%rd33], %r20;"),
            "missing block_sums store from captured inclusive sum\n{ptx}"
        );

        assert!(ptx.contains("DONE:"));
        assert!(ptx.contains("ret;"));
    }

    /// Sanity check that the Blelloch PTX is non-empty and structurally
    /// distinct from the Hillis-Steele PTX. The two kernels share an ABI
    /// but should never be byte-identical.
    #[test]
    fn blelloch_scan_ptx_differs_from_hillis_steele() {
        let blelloch = compile_prefix_scan_kernel_blelloch().expect("blelloch compiles");
        let hillis = compile_prefix_scan_kernel().expect("hillis-steele compiles");
        assert!(!blelloch.is_empty());
        assert_ne!(blelloch, hillis, "Blelloch and Hillis-Steele PTX must differ");

        // Hillis-Steele uses two ping-pong shmem buffers (2048 bytes);
        // Blelloch uses one (1024 bytes). The shared decl is the most
        // load-bearing structural difference.
        assert!(hillis.contains(".shared .align 4 .b8 sdata[2048]"));
        assert!(blelloch.contains(".shared .align 4 .b8 sdata[1024]"));
    }

    /// The three status codes plus the 30-bit value mask must form a
    /// disjoint encoding inside a single u32. If any of these constants
    /// drifts to overlap with another the lookback kernel's pack/unpack
    /// logic silently corrupts running prefixes.
    #[test]
    fn lookback_constants_are_distinct() {
        // Three status codes are pairwise different.
        assert_ne!(LOOKBACK_STATUS_INVALID, LOOKBACK_STATUS_AGGREGATE);
        assert_ne!(LOOKBACK_STATUS_INVALID, LOOKBACK_STATUS_INCLUSIVE);
        assert_ne!(LOOKBACK_STATUS_AGGREGATE, LOOKBACK_STATUS_INCLUSIVE);

        // INVALID is the implicit value of a freshly zeroed slot.
        assert_eq!(LOOKBACK_STATUS_INVALID, 0);

        // Each status fits in 2 bits.
        for s in [
            LOOKBACK_STATUS_INVALID,
            LOOKBACK_STATUS_AGGREGATE,
            LOOKBACK_STATUS_INCLUSIVE,
        ] {
            assert!(s < 4, "status {s} does not fit in 2 bits");
        }

        // Value mask covers exactly the low 30 bits — no overlap with the
        // top-2-bits status field.
        assert_eq!(LOOKBACK_VALUE_MASK, (1u32 << 30) - 1);
        let status_field_mask: u32 = !LOOKBACK_VALUE_MASK;
        assert_eq!(
            status_field_mask, 0xC000_0000,
            "status field must cover bits 30:31"
        );
        assert_eq!(
            LOOKBACK_VALUE_MASK & status_field_mask,
            0,
            "value and status fields overlap"
        );
        // Packed AGGREGATE/INCLUSIVE shifted into the status field never
        // intersect a value in the low 30 bits.
        let packed_agg = LOOKBACK_STATUS_AGGREGATE << 30;
        let packed_inc = LOOKBACK_STATUS_INCLUSIVE << 30;
        assert_eq!(packed_agg & LOOKBACK_VALUE_MASK, 0);
        assert_eq!(packed_inc & LOOKBACK_VALUE_MASK, 0);
        assert_ne!(packed_agg, packed_inc);
    }

    /// Structural smoke test for the decoupled-lookback variant: header,
    /// 5-arg signature, the three publication-protocol labels, and the
    /// memory-fence / acquire-load mnemonics that make the cross-CTA
    /// synchronization correct on sm_70+.
    #[test]
    fn lookback_ptx_has_shape() {
        let ptx =
            compile_prefix_scan_kernel_lookback().expect("lookback PTX compiles");

        // Header.
        assert!(ptx.contains(".version 7.5"));
        assert!(ptx.contains(".target sm_70"));
        assert!(ptx.contains(".address_size 64"));

        // Entry name + 5-param ABI (4x u64 ptr + 1x u32).
        assert!(
            ptx.contains(".visible .entry bolt_prefix_scan_lookback("),
            "missing lookback entry name:\n{ptx}"
        );
        assert!(ptx.contains(".param .u64 bolt_prefix_scan_lookback_param_0,"));
        assert!(ptx.contains(".param .u64 bolt_prefix_scan_lookback_param_1,"));
        assert!(ptx.contains(".param .u64 bolt_prefix_scan_lookback_param_2,"));
        assert!(ptx.contains(".param .u32 bolt_prefix_scan_lookback_param_3,"));
        assert!(ptx.contains(".param .u64 bolt_prefix_scan_lookback_param_4"));

        // Publication / spin / broadcast labels — the load-bearing markers
        // for the decoupled-lookback protocol.
        for label in [
            "PUBLISH_AGGREGATE:",
            "LOOKBACK_SPIN:",
            "PUBLISH_INCLUSIVE:",
            "BROADCAST:",
        ] {
            assert!(
                ptx.contains(label),
                "missing label `{label}` in lookback PTX:\n{ptx}"
            );
        }

        // At least two membar.gl: one after PUBLISH_AGGREGATE, one after
        // PUBLISH_INCLUSIVE. Drop either fence and peer blocks can observe
        // a stale slot.
        let n_membar = ptx.matches("membar.gl;").count();
        assert!(
            n_membar >= 2,
            "expected >=2 membar.gl (one per publish), got {n_membar}:\n{ptx}"
        );

        // Acquire-scope read of the partial_status slot. Without it the
        // spin loop could observe a torn write on sm_70+.
        assert!(
            ptx.contains("ld.acquire.gpu.u32"),
            "missing ld.acquire.gpu.u32 on partial_status read:\n{ptx}"
        );
    }

    /// The lookback forward-progress guard (review C-7 / E §2) accepts a
    /// single occupancy-bounded wave and rejects an over-subscribed grid.
    #[test]
    fn lookback_guard_enforces_coresidency() {
        // grid fits within resident capacity, small n_rows → safe.
        assert!(lookback_launch_is_safe(64, 64, 100_000));
        assert!(lookback_launch_is_safe(1, 80, 1));
        // grid exceeds resident capacity → would deadlock → unsafe.
        assert!(!lookback_launch_is_safe(65, 64, 100_000));
        // zero reported capacity is never safe.
        assert!(!lookback_launch_is_safe(0, 0, 0));
    }

    /// The guard also enforces the s32 row-addressing and 30-bit value-budget
    /// bounds, falling back to multipass for oversize row counts.
    #[test]
    fn lookback_guard_enforces_row_bounds() {
        // n_rows at/over the 30-bit value budget saturates the prefix → unsafe
        // even with ample co-residency.
        assert!(lookback_launch_is_safe(1, 1, LOOKBACK_MAX_ROWS as usize - 1));
        assert!(!lookback_launch_is_safe(1, 1, LOOKBACK_MAX_ROWS as usize));
        // n_rows above i32::MAX mis-addresses in the s32 tid math → unsafe.
        assert!(!lookback_launch_is_safe(1, 1, i32::MAX as usize + 1));
        // The value budget is the binding (smaller) of the two row caps.
        assert!((LOOKBACK_MAX_ROWS as usize) < i32::MAX as usize);
    }
}

/// Adapt a `std::fmt::Error` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("prefix_scan: write failed: {}", e))
}
