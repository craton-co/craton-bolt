// SPDX-License-Identifier: Apache-2.0

//! Per-partition shared-memory GROUP BY **COUNT(*)** kernel — Tier 2.1
//! companion to `partition_reduce_kernel_multi`.
//!
//! ## Why this exists
//!
//! AVG decomposes into `SUM(x) / COUNT(*)`. The Tier-1 AVG executor
//! launches the SUM kernel for each AVG plus one COUNT(*) kernel for the
//! shared denominator. Tier-2.1 needs the same shape at the partition-
//! reduce stage: one launch produces per-group SUMs (via
//! `partition_reduce_kernel_multi`), and a sibling launch produces
//! per-group COUNTs (this kernel).
//!
//! It also stands alone: `SELECT key, COUNT(*) FROM x GROUP BY key` over
//! high-cardinality keys benefits from the Tier-2 partitioning + GPU
//! pass-2 pattern just as well as SUM does.
//!
//! ## Algorithm
//!
//! Identical to the single-value SUM kernel, except:
//!   * No `partition_vals` input — COUNT(*) doesn't read a value column.
//!   * Per row: `atom.shared.add.u64 block_counts[slot], 1`.
//!   * Slot accumulator is `u64`, not `f64`. PTX `atom.add.u64` is
//!     supported on sm_70+ (we already use it elsewhere — see
//!     `shmem_count_kernel.rs`).
//!
//! ## Shared-memory layout
//!
//! Per block (4 + 8 + 4 = 16 KiB, identical to the single-SUM variant):
//!   block_keys   : i32 × 1024 =  4 KiB
//!   block_counts : u64 × 1024 =  8 KiB
//!   block_set    : u32 × 1024 =  4 KiB

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

/// Open-addressing slot count per block. Must match
/// `partition_reduce_kernel{,_multi,_i64}::BLOCK_GROUPS`.
pub const BLOCK_GROUPS: u32 = 1024;

/// Threads per block. Matches the SUM kernels.
pub const BLOCK_THREADS: u32 = 256;

/// Partition count for downstream sizing. Matches
/// `partition_kernel::NUM_PARTITIONS` after Tier-2.1 tuning.
pub const NUM_PARTITIONS: u32 = 4096;

/// Probe bound — same v0 policy as the other reduce kernels.
const MAX_PROBES: u32 = BLOCK_GROUPS;

/// Entry-point name.
pub const KERNEL_ENTRY: &str = "bolt_partition_reduce_count";

/// Entry-point name for the spill-counter variant of the COUNT kernel.
/// Distinct from [`KERNEL_ENTRY`] so both PTX modules can coexist in the
/// JIT cache.
pub const KERNEL_ENTRY_WITH_SPILL: &str = "bolt_partition_reduce_count_spill";

/// Generate PTX for the COUNT(*) per-partition reduce kernel.
///
/// Kernel signature (PTX-level):
/// ```text
/// .visible .entry bolt_partition_reduce_count(
///     .param .u64 partition_keys,    // const int32_t* scatter_keys[n_rows]
///     .param .u64 partition_offsets, // const uint32_t* offsets[NUM_PARTITIONS+1]
///     .param .u64 out_keys,          //       int32_t* [NUM_PARTITIONS*BLOCK_GROUPS]
///     .param .u64 out_counts,        //       uint64_t* [NUM_PARTITIONS*BLOCK_GROUPS]
///     .param .u64 out_set            //       uint8_t*  [NUM_PARTITIONS*BLOCK_GROUPS]
/// )
/// ```
///
/// Launch geometry: `grid = NUM_PARTITIONS, block = BLOCK_THREADS`.
pub fn compile_partition_reduce_kernel_count() -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY;
    let block_groups = BLOCK_GROUPS;
    let mask = BLOCK_GROUPS - 1;
    let block_threads = BLOCK_THREADS;
    let keys_bytes = BLOCK_GROUPS * 4;
    let counts_bytes = BLOCK_GROUPS * 8;
    let set_bytes = BLOCK_GROUPS * 4;
    let max_probes = MAX_PROBES;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(
        ptx,
        ".shared .align 4 .b8 block_keys_buf[{bytes}];",
        bytes = keys_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 8 .b8 block_counts_buf[{bytes}];",
        bytes = counts_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 4 .b8 block_set_buf[{bytes}];",
        bytes = set_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_3,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_4").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<64>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<64>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;

    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_counts_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf;").map_err(write_err)?;

    // Global pointers:
    //   %rd3 = partition_keys      (param_0, i32*)
    //   %rd4 = partition_offsets   (param_1, u32*)
    //   %rd5 = out_keys            (param_2, i32*)
    //   %rd6 = out_counts          (param_3, u64*)
    //   %rd7 = out_set             (param_4, u8*)
    writeln!(ptx, "\tld.param.u64 %rd3, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd4, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd5, [{entry}_param_2];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd5, %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd6, [{entry}_param_3];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd6, %rd6;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd7, [{entry}_param_4];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd7, %rd7;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // start, end = offsets[pid], offsets[pid+1]
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd4, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd11, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd12];").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ---------- Phase 1: zero shared arrays --------------------------------
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p0, %r20, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd21], 0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd22, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd22;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd23], 0;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd24], 0;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tadd.u32 %r20, %r20, {bt};",
        bt = block_threads
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra ZERO_TOP;").map_err(write_err)?;
    writeln!(ptx, "ZERO_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ---------- Phase 2: probe + atomic count ------------------------------
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    // key = partition_keys[i]
    writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s32 %r31, [%rd31];").map_err(write_err)?;

    // slot = key & mask
    writeln!(
        ptx,
        "\tand.b32 %r32, %r31, 0x{mask:X};",
        mask = mask
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r33, 0;").map_err(write_err)?;

    writeln!(ptx, "PROBE_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r33, %r33, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.gt.u32 %p2, %r33, {mp};",
        mp = max_probes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra LOOP_NEXT;").map_err(write_err)?;

    // Slot addresses:
    //   addr_set   = block_set    + slot * 4
    //   addr_key   = block_keys   + slot * 4
    //   addr_count = block_counts + slot * 8
    writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd34;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?;

    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;

    // MATCH path: fence between observing set==1 (via CAS) and loading
    // the key. CAS and the key store touch different shared addresses,
    // so PTX on sm_70 requires an explicit membar.cta to order them
    // across threads. Without this, a racing thread could read a still-
    // zeroed key and false-match key 0.
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    // ACQUIRE-LOAD: pairs with the publisher's atomic store; makes the
    // read-of-published-value contract explicit (sm_70+). Replaces the
    // plain `ld.<space>.<ty>` which relied on the publisher's release +
    // SASS-level implicit acquire — sound in practice but not promised
    // by PTX semantics.
    writeln!(ptx, "\tld.acquire.cta.s32 %r35, [%rd36];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p4, %r35, %r31;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra MATCH;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r32, %r32, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tand.b32 %r32, %r32, 0x{mask:X};",
        mask = mask
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    // CLAIM: publish the key, fence so racing readers see it, then
    // atomically add 1 to the count.
    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd36], %r31;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.u64 %rd40, [%rd38], 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    writeln!(ptx, "MATCH:").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.u64 %rd41, [%rd38], 1;").map_err(write_err)?;

    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ---------- Phase 3: export ---------------------------------------------
    writeln!(
        ptx,
        "\tmul.lo.u32 %r40, %r0, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r41, %r2;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p5, %r41, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra EXPORT_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r42, %r40, %r41;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd42, %r41, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd43, %rd0, %rd42;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s32 %r43, [%rd43];").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd44, %r41, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd45, %rd1, %rd44;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u64 %rd46, [%rd45];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd47, %rd2, %rd42;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r44, [%rd47];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p6, %r44, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r45, 1, 0, %p6;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd48, %r42, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd49, %rd5, %rd48;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s32 [%rd49], %r43;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd50, %r42, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd51, %rd6, %rd50;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u64 [%rd51], %rd46;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u64.u32 %rd52, %r42;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd53, %rd7, %rd52;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd53], %r45;").map_err(write_err)?;

    writeln!(
        ptx,
        "\tadd.u32 %r41, %r41, {bt};",
        bt = block_threads
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra EXPORT_TOP;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_DONE:").map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Spill-counter-aware sibling of [`compile_partition_reduce_kernel_count`].
///
/// Same algorithm; adds a 6th `.param .u64 spill_counter` pointer to the
/// kernel signature. When a row's linear probe exceeds [`MAX_PROBES`]
/// without claiming or matching a slot, the kernel atomically increments
/// `*spill_counter` instead of silently dropping the row, so host
/// orchestrators can detect the corruption and surface a structured
/// error.
///
/// Exports [`KERNEL_ENTRY_WITH_SPILL`]; both variants coexist in the JIT
/// cache.
pub fn compile_partition_reduce_kernel_count_with_spill() -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY_WITH_SPILL;
    let block_groups = BLOCK_GROUPS;
    let mask = BLOCK_GROUPS - 1;
    let block_threads = BLOCK_THREADS;
    let keys_bytes = BLOCK_GROUPS * 4;
    let counts_bytes = BLOCK_GROUPS * 8;
    let set_bytes = BLOCK_GROUPS * 4;
    let max_probes = MAX_PROBES;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(
        ptx,
        ".shared .align 4 .b8 block_keys_buf_csp[{bytes}];",
        bytes = keys_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 8 .b8 block_counts_buf_csp[{bytes}];",
        bytes = counts_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 4 .b8 block_set_buf_csp[{bytes}];",
        bytes = set_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_3,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_4,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_5").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<64>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<64>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;

    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf_csp;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_counts_buf_csp;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf_csp;").map_err(write_err)?;

    // Globals: rd3=keys, rd4=offsets, rd5=out_keys, rd6=out_counts,
    //          rd7=out_set, rd8=spill_counter.
    writeln!(ptx, "\tld.param.u64 %rd3, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd4, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd5, [{entry}_param_2];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd5, %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd6, [{entry}_param_3];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd6, %rd6;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd7, [{entry}_param_4];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd7, %rd7;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd8, [{entry}_param_5];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd8, %rd8;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd10, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd4, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd11, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd12];").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Zero shared arrays.
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p0, %r20, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd21], 0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd22, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd22;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd23], 0;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd24], 0;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tadd.u32 %r20, %r20, {bt};",
        bt = block_threads
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra ZERO_TOP;").map_err(write_err)?;
    writeln!(ptx, "ZERO_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Probe + count.
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s32 %r31, [%rd31];").map_err(write_err)?;

    writeln!(
        ptx,
        "\tand.b32 %r32, %r31, 0x{mask:X};",
        mask = mask
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r33, 0;").map_err(write_err)?;

    writeln!(ptx, "PROBE_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r33, %r33, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.gt.u32 %p2, %r33, {mp};",
        mp = max_probes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra SPILL_BUMP;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd34;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?;

    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;

    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s32 %r35, [%rd36];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p4, %r35, %r31;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra MATCH;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r32, %r32, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tand.b32 %r32, %r32, 0x{mask:X};",
        mask = mask
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd36], %r31;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.u64 %rd40, [%rd38], 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    writeln!(ptx, "MATCH:").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.u64 %rd41, [%rd38], 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    writeln!(ptx, "SPILL_BUMP:").map_err(write_err)?;
    writeln!(ptx, "\tatom.global.add.u32 %r36, [%rd8], 1;").map_err(write_err)?;

    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Export.
    writeln!(
        ptx,
        "\tmul.lo.u32 %r40, %r0, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r41, %r2;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p5, %r41, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra EXPORT_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r42, %r40, %r41;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd42, %r41, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd43, %rd0, %rd42;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s32 %r43, [%rd43];").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd44, %r41, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd45, %rd1, %rd44;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u64 %rd46, [%rd45];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd47, %rd2, %rd42;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r44, [%rd47];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p6, %r44, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r45, 1, 0, %p6;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd48, %r42, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd49, %rd5, %rd48;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s32 [%rd49], %r43;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd50, %r42, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd51, %rd6, %rd50;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u64 [%rd51], %rd46;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u64.u32 %rd52, %r42;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd53, %rd7, %rd52;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd53], %r45;").map_err(write_err)?;

    writeln!(
        ptx,
        "\tadd.u32 %r41, %r41, {bt};",
        bt = block_threads
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra EXPORT_TOP;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_DONE:").map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!(
        "partition_reduce_kernel_count: write failed: {}",
        e
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_ok() {
        assert!(compile_partition_reduce_kernel_count().is_ok());
    }

    #[test]
    fn has_correct_entry() {
        let ptx = compile_partition_reduce_kernel_count().unwrap();
        assert!(
            ptx.contains(".visible .entry bolt_partition_reduce_count("),
            "entry-point not found"
        );
    }

    #[test]
    fn uses_atom_shared_add_u64() {
        let ptx = compile_partition_reduce_kernel_count().unwrap();
        // CLAIM + MATCH paths each issue one atom.shared.add.u64.
        let count = ptx.matches("atom.shared.add.u64").count();
        assert!(count >= 2, "expected ≥ 2 atom.shared.add.u64, got {count}");
    }

    #[test]
    fn no_value_pointer_load() {
        // The COUNT kernel takes no value column — its row loop must read
        // only the key. We assert the parameter count: 5 .u64 lines (keys,
        // offsets, out_keys, out_counts, out_set).
        let ptx = compile_partition_reduce_kernel_count().unwrap();
        let n_params = ptx.matches(".param .u64 ").count();
        assert_eq!(n_params, 5, "expected 5 .u64 params, got {n_params}");
    }

    #[test]
    fn writes_u64_counts_to_global() {
        let ptx = compile_partition_reduce_kernel_count().unwrap();
        assert!(
            ptx.contains("st.global.u64"),
            "u64 count store to global missing"
        );
    }

    #[test]
    fn has_two_barriers() {
        let ptx = compile_partition_reduce_kernel_count().unwrap();
        let bars = ptx.matches("bar.sync 0").count();
        assert!(bars >= 2, "expected ≥ 2 bar.sync 0, got {bars}");
    }

    // ----- _with_spill variant shape tests ---------------------------------

    #[test]
    fn with_spill_uses_distinct_entry_name() {
        let ptx = compile_partition_reduce_kernel_count_with_spill().unwrap();
        assert_eq!(KERNEL_ENTRY_WITH_SPILL, "bolt_partition_reduce_count_spill");
        assert!(
            ptx.contains(".visible .entry bolt_partition_reduce_count_spill("),
            "spill variant must declare its own entry point:\n{ptx}"
        );
        assert!(
            !ptx.contains(".visible .entry bolt_partition_reduce_count("),
            "spill variant must NOT export the base entry name:\n{ptx}"
        );
    }

    #[test]
    fn with_spill_has_six_params_and_global_atomic() {
        let ptx = compile_partition_reduce_kernel_count_with_spill().unwrap();
        let n_params = ptx.matches(".param .u64 ").count();
        assert_eq!(
            n_params, 6,
            "COUNT spill variant must expose 6 .u64 params (5 + spill_counter), got {n_params}\n{ptx}"
        );
        assert!(ptx.contains("atom.global.add.u32"), "{ptx}");
        assert!(ptx.contains("SPILL_BUMP:"), "{ptx}");
        // Counts still use u64 shared add.
        assert!(ptx.contains("atom.shared.add.u64"), "{ptx}");
    }
}
