// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for GPU-side **framed window-function** execution.
//!
//! Pairs with [`crate::exec::window`], which drives the GPU path: it sorts the
//! input by `(partition_key, order_key)` (reusing the existing GPU/host sort),
//! then runs the kernels emitted here to compute the per-row window output in
//! *permuted* order before scattering back to original row positions.
//!
//! Today this module emits two device kernels, both single-block 1-D grids of
//! [`WINDOW_BLOCK_SIZE`] threads with one thread per row. The host driver only
//! launches them for inputs that fit a single block (the partition/order scan
//! is intra-block only — a multi-block segmented scan with cross-block carry is
//! deferred; see `crate::exec::window` dispatch gate and the "What's deferred"
//! list at the bottom of this comment):
//!
//! 1. **Boundary-flag kernel** ([`compile_boundary_flag_kernel`],
//!    entry [`BOUNDARY_FLAG_ENTRY`]). Given the permuted partition-key and
//!    order-key columns (both `i64`, NULLs pre-encoded by the host into a
//!    sentinel-free comparable lane — see the host driver), it writes two `u8`
//!    flag columns:
//!      * `part_head[i]` = 1 iff row `i` starts a new partition
//!        (`i == 0 || part_key[i] != part_key[i-1]`);
//!      * `peer_head[i]` = 1 iff row `i` starts a new ordering peer group
//!        *within* its partition (`part_head[i] || order_key[i] !=
//!        order_key[i-1]`).
//!    These flags are the segment descriptors the scan kernel consumes.
//!
//! 2. **Segmented running-aggregate / rank kernel**
//!    ([`compile_segmented_scan_kernel`], entry [`SEGMENTED_SCAN_ENTRY`]).
//!    A Hillis-Steele inclusive scan whose combine operator is *segmented*: a
//!    per-element `part_head` flag resets the running accumulator at every
//!    partition boundary (the classic Schwartz segmented-scan trick — carry a
//!    `(flag, value)` pair and `combine((fa,va),(fb,vb)) = (fa|fb, fb ? vb :
//!    va+vb)`). One launch computes, per row, the **partition-local inclusive
//!    prefix sum** of an `i64` value column. The host derives every supported
//!    window function from this single primitive:
//!      * `ROW_NUMBER`  — value column = all 1s; output = inclusive prefix.
//!      * running `COUNT(x)` — value column = `is_non_null(x) ? 1 : 0`.
//!      * running `SUM(x)`   — value column = `x` (NULLs contribute 0); the
//!        host post-processes peer groups so every peer sees the
//!        through-end-of-peer-group sum (RANGE frame), matching the host path.
//!      * `RANK`        — value column = `peer_head ? 1 : 0` would give
//!        DENSE_RANK; for RANK the host instead scans `peer_head`-gated row
//!        counts. The host derives both from the same inclusive prefix plus
//!        the boundary flags (see `crate::exec::window`); the kernel itself
//!        only provides the segmented inclusive sum.
//!
//! ## ABI stability
//!
//! Both entry-point names and the parameter ordering are a codegen contract
//! pinned by the golden/substring tests in `tests/ptx_golden_tests.rs` and the
//! in-module `#[cfg(test)]` substring tests. Changing them is an intentional
//! contract change.
//!
//! ## What's deferred
//!
//! * **Multi-block scan.** The segmented scan is intra-block (Hillis-Steele,
//!   `WINDOW_BLOCK_SIZE` rows max). A decoupled-lookback segmented variant
//!   (mirroring `prefix_scan::compile_prefix_scan_kernel_lookback`) is the
//!   natural follow-up for partitions spanning many blocks.
//! * **General frames** (`ROWS`/`RANGE BETWEEN`), `LAG`/`LEAD`, `MIN`/`MAX`
//!   /`AVG` segmented variants. The host path covers all of these; the GPU
//!   path declines and falls back.
//! * **Wide / Utf8 keys.** The boundary kernel compares a single `i64` lane;
//!   the host driver gates non-`i64`-encodable keys to the host path.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

/// PTX target metadata baked into every emitted module. Mirrors
/// `crate::jit::prefix_scan` so the modules share one SM target / ptxas path.
const PTX_VERSION: &str = ".version 7.5";
/// Target SM architecture string.
const PTX_TARGET: &str = ".target sm_70";
/// Address size directive (we always use 64-bit pointers).
const PTX_ADDRESS_SIZE: &str = ".address_size 64";

/// Threads per block for both window kernels. One thread per row.
///
/// MUST be a power of two: the segmented Hillis-Steele scan walks strides
/// `1, 2, 4, ... BLOCK_SIZE/2`, exactly like
/// [`crate::jit::prefix_scan::BLOCK_SIZE`].
pub const WINDOW_BLOCK_SIZE: u32 = 256;

const _: () = assert!(
    WINDOW_BLOCK_SIZE.is_power_of_two(),
    "segmented Hillis-Steele scan requires power-of-two WINDOW_BLOCK_SIZE"
);

/// Entry-point name for the partition/peer boundary-flag kernel.
pub const BOUNDARY_FLAG_ENTRY: &str = "bolt_window_boundary_flags";

/// Entry-point name for the segmented running-sum (inclusive prefix) kernel.
pub const SEGMENTED_SCAN_ENTRY: &str = "bolt_window_segmented_scan";

#[inline]
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("window_kernel: PTX write failed: {e}"))
}

/// Emit the boundary-flag kernel.
///
/// ABI:
/// ```text
/// .visible .entry bolt_window_boundary_flags(
///     .param .u64 ..._param_0,   // part_key  (i64*, permuted order)
///     .param .u64 ..._param_1,   // order_key (i64*, permuted order)
///     .param .u64 ..._param_2,   // part_head (u8*, out)
///     .param .u64 ..._param_3,   // peer_head (u8*, out)
///     .param .u32 ..._param_4    // n_rows
/// )
/// ```
///
/// One thread per row `i` (`gid = ctaid.x * ntid.x + tid.x`):
///   * `part_head[i] = (i == 0) || (part_key[i] != part_key[i-1])`
///   * `peer_head[i] = part_head[i] || (order_key[i] != order_key[i-1])`
///
/// Rows `>= n_rows` early-return. The host pre-encodes NULL keys into a
/// reserved comparable `i64` value so two NULLs compare equal here (matching
/// the host path's NULL-grouping semantics).
pub fn compile_boundary_flag_kernel() -> BoltResult<String> {
    let e = BOUNDARY_FLAG_ENTRY;
    let mut ptx = String::new();
    writeln!(ptx, "{}", PTX_VERSION).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_TARGET).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_ADDRESS_SIZE).map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {e}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {e}_param_0,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {e}_param_1,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {e}_param_2,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {e}_param_3,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {e}_param_4").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b16   %rs<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<32>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // gid = ctaid.x * ntid.x + tid.x ; load n_rows.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{e}_param_4];").map_err(write_err)?;
    // Out-of-range guard.
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // Globalize pointers.
    writeln!(ptx, "\tld.param.u64 %rd0, [{e}_param_0];").map_err(write_err)?; // part_key
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd1, [{e}_param_1];").map_err(write_err)?; // order_key
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd2, [{e}_param_2];").map_err(write_err)?; // part_head
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd3, [{e}_param_3];").map_err(write_err)?; // peer_head
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;

    // Byte offset for i64 element i: gid * 8.
    writeln!(ptx, "\tmul.wide.u32 %rd4, %r3, 8;").map_err(write_err)?;
    // &part_key[i], &order_key[i]
    writeln!(ptx, "\tadd.s64 %rd5, %rd0, %rd4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd6, %rd1, %rd4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u64 %rd7, [%rd5];").map_err(write_err)?; // part_key[i]
    writeln!(ptx, "\tld.global.nc.u64 %rd8, [%rd6];").map_err(write_err)?; // order_key[i]

    // is_first = (gid == 0). If so, both heads are 1; skip neighbor loads.
    writeln!(ptx, "\tsetp.eq.s32 %p1, %r3, 0;").map_err(write_err)?;
    // Default: part_head = 1, peer_head = 1 (head-of-input case).
    writeln!(ptx, "\tmov.u32 %r5, 1;").map_err(write_err)?; // part_head value
    writeln!(ptx, "\tmov.u32 %r6, 1;").map_err(write_err)?; // peer_head value
    writeln!(ptx, "\t@%p1 bra STORE;").map_err(write_err)?;

    // Load neighbor (i-1) keys: offset (gid-1)*8 = gid*8 - 8.
    writeln!(ptx, "\tadd.s64 %rd9, %rd5, -8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd10, %rd6, -8;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u64 %rd11, [%rd9];").map_err(write_err)?; // part_key[i-1]
    writeln!(ptx, "\tld.global.nc.u64 %rd12, [%rd10];").map_err(write_err)?; // order_key[i-1]

    // part_head = (part_key[i] != part_key[i-1]) ? 1 : 0
    writeln!(ptx, "\tsetp.ne.s64 %p2, %rd7, %rd11;").map_err(write_err)?;
    writeln!(ptx, "\tselp.s32 %r5, 1, 0, %p2;").map_err(write_err)?;
    // order_changed = (order_key[i] != order_key[i-1])
    writeln!(ptx, "\tsetp.ne.s64 %p3, %rd8, %rd12;").map_err(write_err)?;
    writeln!(ptx, "\tselp.s32 %r7, 1, 0, %p3;").map_err(write_err)?;
    // peer_head = part_head | order_changed
    writeln!(ptx, "\tor.b32 %r6, %r5, %r7;").map_err(write_err)?;

    writeln!(ptx, "STORE:").map_err(write_err)?;
    // Store the two u8 flags at &part_head[i] / &peer_head[i] (1-byte stride).
    writeln!(ptx, "\tmul.wide.u32 %rd13, %r3, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd14, %rd2, %rd13;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd15, %rd3, %rd13;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u16.u32 %rs0, %r5;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd14], %rs0;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u16.u32 %rs1, %r6;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd15], %rs1;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;
    Ok(ptx)
}

/// Emit the segmented inclusive-prefix-sum kernel.
///
/// ABI:
/// ```text
/// .visible .entry bolt_window_segmented_scan(
///     .param .u64 ..._param_0,   // values    (i64*, permuted order)
///     .param .u64 ..._param_1,   // seg_head  (u8*,  segment-head flags)
///     .param .u64 ..._param_2,   // out       (i64*, inclusive segmented prefix)
///     .param .u32 ..._param_3    // n_rows
/// )
/// ```
///
/// Single block, `WINDOW_BLOCK_SIZE` threads, one row per thread. Computes the
/// **partition-local inclusive prefix sum** of `values` where `seg_head[i] ==
/// 1` resets the running sum at row `i` (segment boundary). Uses the
/// segmented-scan combine over `(flag, value)` pairs in shared memory:
///
/// ```text
///   combine((fa, va), (fb, vb)) = (fa | fb, fb ? vb : va + vb)
/// ```
///
/// which is associative, so the Hillis-Steele tree (strides `1, 2, ...,
/// BLOCK_SIZE/2`) produces the correct per-segment inclusive scan.
///
/// Layout: two ping-pong shared buffers, each `WINDOW_BLOCK_SIZE` entries of
/// `(i64 value, u32 flag)` — we store value (8 bytes) and flag (4 bytes,
/// padded to 8 for alignment) interleaved as 16-byte records, so each buffer
/// is `WINDOW_BLOCK_SIZE * 16` bytes.
pub fn compile_segmented_scan_kernel() -> BoltResult<String> {
    let e = SEGMENTED_SCAN_ENTRY;
    // 16-byte records: [i64 value @ +0][u32 flag @ +8][u32 pad @ +12].
    let rec_bytes: u32 = 16;
    let buf_bytes: u32 = WINDOW_BLOCK_SIZE * rec_bytes;
    let shared_bytes_total: u32 = buf_bytes * 2;

    let mut ptx = String::new();
    writeln!(ptx, "{}", PTX_VERSION).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_TARGET).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_ADDRESS_SIZE).map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".shared .align 8 .b8 sdata[{shared_bytes_total}];").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {e}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {e}_param_0,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {e}_param_1,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {e}_param_2,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {e}_param_3").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b16   %rs<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<48>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<64>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // gid = ctaid.x * ntid.x + tid.x ; load n_rows ; tid in %r2.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{e}_param_3];").map_err(write_err)?;

    // Globalize pointers.
    writeln!(ptx, "\tld.param.u64 %rd0, [{e}_param_0];").map_err(write_err)?; // values
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd1, [{e}_param_1];").map_err(write_err)?; // seg_head
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd2, [{e}_param_2];").map_err(write_err)?; // out
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;

    // Load this thread's (value, flag); out-of-range lanes seed (0, 1) so they
    // form their own segment and never leak a carry into a real row.
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd5, 0;").map_err(write_err)?; // value
    writeln!(ptx, "\tmov.u32 %r5, 1;").map_err(write_err)?; // flag
    writeln!(ptx, "\t@%p0 bra AFTER_LOAD;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd6, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd7, %rd0, %rd6;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u64 %rd5, [%rd7];").map_err(write_err)?; // value
    writeln!(ptx, "\tmul.wide.u32 %rd8, %r3, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd9, %rd1, %rd8;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u8 %rs0, [%rd9];").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u16 %r5, %rs0;").map_err(write_err)?;
    // Normalise any non-zero flag byte to 1.
    writeln!(ptx, "\tsetp.ne.s32 %p1, %r5, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.s32 %r5, 1, 0, %p1;").map_err(write_err)?;
    writeln!(ptx, "AFTER_LOAD:").map_err(write_err)?;

    // Shared-record addresses. base0 = sdata, base1 = sdata + buf_bytes.
    // rec offset = tid * 16. ping addr in %rd12, pong addr in %rd13.
    writeln!(ptx, "\tmov.u64 %rd10, sdata;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd10, {buf_bytes};").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd14, %r2, {rec_bytes};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd10, %rd14;").map_err(write_err)?; // ping rec
    writeln!(ptx, "\tadd.s64 %rd13, %rd11, %rd14;").map_err(write_err)?; // pong rec
    // Store initial (value, flag) into ping.
    writeln!(ptx, "\tst.shared.u64 [%rd12], %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd12+8], %r5;").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;

    // Hillis-Steele segmented scan, strides 1..BLOCK_SIZE/2.
    //
    // Per round, this thread combines its left-neighbor-at-offset record into
    // its own under the segmented operator:
    //   if tid >= offset:
    //     (fl, vl) = ping[tid - offset]
    //     (fs, vs) = ping[tid]
    //     out_flag  = fl | fs   ... wait — segmented combine puts the *earlier*
    //                                element on the LEFT. Here read[tid-offset]
    //                                is the earlier element (left = a), read[tid]
    //                                is the later (right = b).
    //     combine((fa,va),(fb,vb)) = (fa|fb, fb ? vb : va+vb)
    //   else:
    //     copy own record through unchanged.
    let mut offset: u32 = 1;
    let mut step: usize = 0;
    while offset < WINDOW_BLOCK_SIZE {
        let off_bytes = offset * rec_bytes;
        // Load own (b) record: value %rd20, flag %r20.
        writeln!(ptx, "\tld.shared.u64 %rd20, [%rd12];").map_err(write_err)?;
        writeln!(ptx, "\tld.shared.u32 %r20, [%rd12+8];").map_err(write_err)?;
        // Default write = own record (the tid < offset case).
        writeln!(ptx, "\tmov.u64 %rd21, %rd20;").map_err(write_err)?; // out value
        writeln!(ptx, "\tmov.u32 %r21, %r20;").map_err(write_err)?; // out flag
        // If tid < offset, skip the combine.
        let p = 2 + (step % 4);
        writeln!(ptx, "\tsetp.lt.s32 %p{p}, %r2, {offset};").map_err(write_err)?;
        writeln!(ptx, "\t@%p{p} bra SEG_SKIP_{step};").map_err(write_err)?;
        // Load left (a) neighbor record at tid - offset.
        writeln!(ptx, "\tsub.s64 %rd22, %rd12, {off_bytes};").map_err(write_err)?;
        writeln!(ptx, "\tld.shared.u64 %rd23, [%rd22];").map_err(write_err)?; // va
        writeln!(ptx, "\tld.shared.u32 %r22, [%rd22+8];").map_err(write_err)?; // fa
        // out_flag = fa | fb
        writeln!(ptx, "\tor.b32 %r21, %r22, %r20;").map_err(write_err)?;
        // out_value = fb ? vb : (va + vb)
        writeln!(ptx, "\tadd.s64 %rd24, %rd23, %rd20;").map_err(write_err)?; // va + vb
        writeln!(ptx, "\tsetp.ne.s32 %p{q}, %r20, 0;", q = 8 + (step % 4)).map_err(write_err)?; // fb != 0
        writeln!(ptx, "\tselp.b64 %rd21, %rd20, %rd24, %p{q};", q = 8 + (step % 4))
            .map_err(write_err)?;
        writeln!(ptx, "SEG_SKIP_{step}:").map_err(write_err)?;
        // Write combined record into pong.
        writeln!(ptx, "\tst.shared.u64 [%rd13], %rd21;").map_err(write_err)?;
        writeln!(ptx, "\tst.shared.u32 [%rd13+8], %r21;").map_err(write_err)?;
        writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
        // Swap ping/pong record pointers.
        writeln!(ptx, "\tmov.u64 %rd25, %rd12;").map_err(write_err)?;
        writeln!(ptx, "\tmov.u64 %rd12, %rd13;").map_err(write_err)?;
        writeln!(ptx, "\tmov.u64 %rd13, %rd25;").map_err(write_err)?;
        offset <<= 1;
        step += 1;
    }

    // %rd12 now points at this thread's record in the buffer holding the
    // inclusive segmented scan. Load its value and store to out[gid].
    writeln!(ptx, "\tld.shared.u64 %rd30, [%rd12];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p6, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p6 bra DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd31, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd32, %rd2, %rd31;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u64 [%rd32], %rd30;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;
    Ok(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_kernel_has_entry_and_abi() {
        let ptx = compile_boundary_flag_kernel().unwrap();
        assert!(ptx.contains(".visible .entry bolt_window_boundary_flags("));
        // 5-arg ABI: 4 pointers + n_rows.
        assert!(ptx.contains("bolt_window_boundary_flags_param_4"));
        assert!(!ptx.contains("bolt_window_boundary_flags_param_5"));
        // Header directives present.
        assert!(ptx.contains(".version 7.5"));
        assert!(ptx.contains(".target sm_70"));
        assert!(ptx.contains(".address_size 64"));
    }

    #[test]
    fn boundary_kernel_compares_both_keys() {
        let ptx = compile_boundary_flag_kernel().unwrap();
        // Two distinct i64 inequality compares: one for partition, one for order.
        let cnt = ptx.matches("setp.ne.s64").count();
        assert!(cnt >= 2, "expected >=2 i64 ne compares, got {cnt}");
        // peer_head OR of part_head and order_changed.
        assert!(ptx.contains("or.b32 %r6, %r5, %r7"));
        // Two u8 stores (part_head + peer_head).
        assert_eq!(ptx.matches("st.global.u8").count(), 2);
        // Read-only cached loads for the key columns.
        assert!(ptx.contains("ld.global.nc.u64"));
    }

    #[test]
    fn segmented_scan_has_entry_and_abi() {
        let ptx = compile_segmented_scan_kernel().unwrap();
        assert!(ptx.contains(".visible .entry bolt_window_segmented_scan("));
        assert!(ptx.contains("bolt_window_segmented_scan_param_3"));
        assert!(!ptx.contains("bolt_window_segmented_scan_param_4"));
    }

    #[test]
    fn segmented_scan_shared_buffer_sized_for_two_pingpong_records() {
        let ptx = compile_segmented_scan_kernel().unwrap();
        // 2 buffers * BLOCK_SIZE * 16-byte records.
        let expect = WINDOW_BLOCK_SIZE * 16 * 2;
        assert!(
            ptx.contains(&format!(".shared .align 8 .b8 sdata[{expect}];")),
            "missing/incorrect shared decl for {expect} bytes"
        );
    }

    #[test]
    fn segmented_scan_unrolls_log2_block_rounds() {
        let ptx = compile_segmented_scan_kernel().unwrap();
        // log2(256) = 8 Hillis-Steele rounds -> labels SEG_SKIP_0..SEG_SKIP_7.
        let rounds = WINDOW_BLOCK_SIZE.trailing_zeros() as usize;
        for s in 0..rounds {
            assert!(
                ptx.contains(&format!("SEG_SKIP_{s}:")),
                "missing round label SEG_SKIP_{s}"
            );
        }
        assert!(!ptx.contains(&format!("SEG_SKIP_{rounds}:")));
    }

    #[test]
    fn segmented_scan_uses_segmented_combine() {
        let ptx = compile_segmented_scan_kernel().unwrap();
        // Flag OR + value add + segment-reset select are the combine's three
        // load-bearing instructions.
        assert!(ptx.contains("or.b32 %r21, %r22, %r20"));
        assert!(ptx.contains("add.s64 %rd24, %rd23, %rd20"));
        assert!(ptx.contains("selp.b64 %rd21, %rd20, %rd24"));
        // Barriers: one seed + one per round.
        let expect_bars = 1 + WINDOW_BLOCK_SIZE.trailing_zeros() as usize;
        assert_eq!(ptx.matches("bar.sync 0;").count(), expect_bars);
    }
}
