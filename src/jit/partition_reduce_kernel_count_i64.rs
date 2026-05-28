// SPDX-License-Identifier: Apache-2.0

//! Per-partition COUNT(*) reduce kernel — **i64 key variant**.
//!
//! Mirror of [`crate::jit::partition_reduce_kernel_count`] (i32 key)
//! adapted for the i64-packed-two-key Tier-2.1 path. Identical to its
//! i32 sibling except:
//!
//!   * Keys are loaded with `ld.global.s64` and stored with `st.global.s64`
//!   * Slot computation uses `cvt.u32.u64` on the low 32 bits then masks
//!   * `block_keys_buf` is 8 KiB (vs 4 KiB)
//!   * Output buffer's per-slot stride for keys is 8 B (vs 4 B)
//!
//! Used by the two-key COUNT(*) executor (`SELECT a, b, COUNT(*) FROM
//! x GROUP BY a, b`) and as the COUNT denominator for the two-key AVG
//! executor (when added).

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

pub const BLOCK_GROUPS: u32 = 1024;
pub const BLOCK_THREADS: u32 = 256;
pub const NUM_PARTITIONS: u32 = 4096;
const MAX_PROBES: u32 = BLOCK_GROUPS;

/// Per-iteration `nanosleep.u32` operand for the collision-advance path
/// (sm_70+). See `partition_reduce_kernel::SPIN_BACKOFF_NS` for full
/// rationale. TODO(perf): exponential back-off.
const SPIN_BACKOFF_NS: u32 = 32;

pub const KERNEL_ENTRY: &str = "bolt_partition_reduce_count_i64";

/// Entry-point name for the spill-counter variant.
pub const KERNEL_ENTRY_WITH_SPILL: &str = "bolt_partition_reduce_count_i64_spill";

/// Generate PTX for the i64-key COUNT(*) per-partition reduce kernel.
///
/// Signature:
/// ```text
/// .visible .entry bolt_partition_reduce_count_i64(
///     .param .u64 partition_keys,    // const int64_t* scatter_keys[n_rows]
///     .param .u64 partition_offsets, // const uint32_t* offsets[K+1]
///     .param .u64 out_keys,          //       int64_t* [K*BG]
///     .param .u64 out_counts,        //       uint64_t* [K*BG]
///     .param .u64 out_set            //       uint8_t* [K*BG]
/// )
/// ```
pub fn compile_partition_reduce_kernel_count_i64() -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY;
    let block_groups = BLOCK_GROUPS;
    let mask = BLOCK_GROUPS - 1;
    let block_threads = BLOCK_THREADS;
    let keys_bytes = BLOCK_GROUPS * 8; // i64
    let counts_bytes = BLOCK_GROUPS * 8;
    let set_bytes = BLOCK_GROUPS * 4;
    let max_probes = MAX_PROBES;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(
        ptx,
        ".shared .align 8 .b8 block_keys_buf[{bytes}];",
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
    for p in 0..4 {
        writeln!(ptx, "\t.param .u64 {entry}_param_{p},").map_err(write_err)?;
    }
    writeln!(ptx, "\t.param .u64 {entry}_param_4").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<64>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<64>;").map_err(write_err)?;
    // Operand register for the per-collision `nanosleep.u32` back-off.
    writeln!(ptx, "\t.reg .u32   %nstime;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_counts_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf;").map_err(write_err)?;

    for (rd, p) in (3..=7).zip(0..5) {
        writeln!(ptx, "\tld.param.u64 %rd{rd}, [{entry}_param_{p}];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd{rd}, %rd{rd};").map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd10, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd4, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd11, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd12];").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 1: zero shared.
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p0, %r20, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd21], 0;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd23], 0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd22, %r20, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd22;").map_err(write_err)?;
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

    // Phase 2: probe + atomic count.
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    // key = partition_keys[i] (i64)
    writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rd60, [%rd31];").map_err(write_err)?;

    // slot from low 32 bits.
    writeln!(ptx, "\tcvt.u32.u64 %r31, %rd60;").map_err(write_err)?;
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

    // Addrs (set: *4, key: *8 i64, count: *8).
    writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd37;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?;

    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;

    // MATCH path: membar.cta to order the CAS (on block_set) against
    // the subsequent key load (on block_keys). Different addresses, so
    // PTX sm_70 requires an explicit fence — without it a racing thread
    // can read a zero key under set==1 and false-match key 0.
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    // ACQUIRE-LOAD: pairs with the publisher's atomic store; makes the
    // read-of-published-value contract explicit (sm_70+). Replaces the
    // plain `ld.<space>.<ty>` which relied on the publisher's release +
    // SASS-level implicit acquire — sound in practice but not promised
    // by PTX semantics.
    writeln!(ptx, "\tld.acquire.cta.s64 %rd61, [%rd36];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s64 %p4, %rd61, %rd60;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra MATCH;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r32, %r32, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tand.b32 %r32, %r32, 0x{mask:X};",
        mask = mask
    )
    .map_err(write_err)?;
    // Occupancy-friendly back-off on the collision-advance path.
    writeln!(
        ptx,
        "\tmov.u32 %nstime, {ns};",
        ns = SPIN_BACKOFF_NS
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tnanosleep.u32 %nstime;").map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd36], %rd60;").map_err(write_err)?;
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

    // Phase 3: export populated slots.
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

    writeln!(ptx, "\tmul.wide.u32 %rd44, %r41, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd43, %rd0, %rd44;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s64 %rd62, [%rd43];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd45, %rd1, %rd44;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u64 %rd46, [%rd45];").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd42, %r41, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd47, %rd2, %rd42;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r44, [%rd47];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p6, %r44, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r45, 1, 0, %p6;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd48, %r42, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd49, %rd5, %rd48;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s64 [%rd49], %rd62;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd51, %rd6, %rd48;").map_err(write_err)?;
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

/// Spill-counter-aware sibling of [`compile_partition_reduce_kernel_count_i64`].
///
/// Identical algorithm; adds one extra `.param .u64 spill_counter`
/// (uint32_t* &spill_counter[1]; may be 0/null). On MAX_PROBES overflow
/// the kernel issues `atom.global.add.u32 [spill_counter], 1` after a
/// null check, then drops the row.
pub fn compile_partition_reduce_kernel_count_i64_with_spill() -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY_WITH_SPILL;
    let block_groups = BLOCK_GROUPS;
    let mask = BLOCK_GROUPS - 1;
    let block_threads = BLOCK_THREADS;
    let keys_bytes = BLOCK_GROUPS * 8;
    let counts_bytes = BLOCK_GROUPS * 8;
    let set_bytes = BLOCK_GROUPS * 4;
    let max_probes = MAX_PROBES;

    super::partition_reduce_kernel_spill_common::emit_ptx_header(&mut ptx)?;

    writeln!(
        ptx,
        ".shared .align 8 .b8 block_keys_buf_sp[{bytes}];",
        bytes = keys_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 8 .b8 block_counts_buf_sp[{bytes}];",
        bytes = counts_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 4 .b8 block_set_buf_sp[{bytes}];",
        bytes = set_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // 5 base params + 1 spill_counter = 6 .u64 params.
    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    for p in 0..5 {
        writeln!(ptx, "\t.param .u64 {entry}_param_{p},").map_err(write_err)?;
    }
    writeln!(ptx, "\t.param .u64 {entry}_param_5").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<64>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<80>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    super::partition_reduce_kernel_spill_common::emit_thread_block_ids(&mut ptx)?;
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf_sp;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_counts_buf_sp;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf_sp;").map_err(write_err)?;

    // %rd3..=%rd7 mirror the base kernel's param mapping.
    // %rd8 is the spill counter pointer.
    for (rd, p) in (3..=7).zip(0..5) {
        writeln!(ptx, "\tld.param.u64 %rd{rd}, [{entry}_param_{p}];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd{rd}, %rd{rd};").map_err(write_err)?;
    }
    writeln!(ptx, "\tld.param.u64 %rd8, [{entry}_param_5];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd8, %rd8;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd10, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd4, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd11, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd12];").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 1: zero shared.
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p0, %r20, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd21], 0;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd23], 0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd22, %r20, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd22;").map_err(write_err)?;
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

    // Phase 2.
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rd60, [%rd31];").map_err(write_err)?;

    writeln!(ptx, "\tcvt.u32.u64 %r31, %rd60;").map_err(write_err)?;
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
    writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd37;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?;

    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;

    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s64 %rd61, [%rd36];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s64 %p4, %rd61, %rd60;").map_err(write_err)?;
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
    writeln!(ptx, "\tst.shared.u64 [%rd36], %rd60;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.u64 %rd40, [%rd38], 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    writeln!(ptx, "MATCH:").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.u64 %rd41, [%rd38], 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    super::partition_reduce_kernel_spill_common::emit_spill_bump_with_null_check(&mut ptx, 8)?;

    super::partition_reduce_kernel_spill_common::emit_loop_next_done(&mut ptx)?;

    // Phase 3: export.
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
        "\tsetp.ge.u32 %p6, %r41, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p6 bra EXPORT_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r42, %r40, %r41;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd44, %r41, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd43, %rd0, %rd44;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s64 %rd62, [%rd43];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd45, %rd1, %rd44;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u64 %rd46, [%rd45];").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd42, %r41, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd47, %rd2, %rd42;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r44, [%rd47];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p7, %r44, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r45, 1, 0, %p7;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd48, %r42, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd49, %rd5, %rd48;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s64 [%rd49], %rd62;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd51, %rd6, %rd48;").map_err(write_err)?;
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
        "partition_reduce_kernel_count_i64: write failed: {}",
        e
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_ok() {
        assert!(compile_partition_reduce_kernel_count_i64().is_ok());
    }

    #[test]
    fn has_correct_entry() {
        let ptx = compile_partition_reduce_kernel_count_i64().unwrap();
        assert!(ptx.contains(".visible .entry bolt_partition_reduce_count_i64("));
    }

    #[test]
    fn uses_i64_key_loads_and_stores() {
        let ptx = compile_partition_reduce_kernel_count_i64().unwrap();
        assert!(ptx.contains("ld.global.s64"));
        assert!(ptx.contains("st.global.s64"));
        assert!(ptx.contains("ld.shared.s64"));
        assert!(ptx.contains("st.shared.u64"));
    }

    #[test]
    fn uses_atom_shared_add_u64() {
        let ptx = compile_partition_reduce_kernel_count_i64().unwrap();
        assert!(ptx.matches("atom.shared.add.u64").count() >= 2);
    }

    // ----- _with_spill variant shape tests ---------------------------------

    #[test]
    fn with_spill_uses_distinct_entry_name() {
        let ptx = compile_partition_reduce_kernel_count_i64_with_spill().expect("compiles");
        assert_eq!(
            KERNEL_ENTRY_WITH_SPILL,
            "bolt_partition_reduce_count_i64_spill"
        );
        let needle = format!(".visible .entry {}(", KERNEL_ENTRY_WITH_SPILL);
        assert!(ptx.contains(&needle));
        assert!(!ptx.contains(".visible .entry bolt_partition_reduce_count_i64("));
    }

    #[test]
    fn with_spill_has_six_params_and_global_atomic() {
        let ptx = compile_partition_reduce_kernel_count_i64_with_spill().expect("compiles");
        let n = ptx.matches(".param .u64 ").count();
        assert_eq!(n, 6, "spill variant must add one .u64 param (got {n})");
        assert!(ptx.contains("atom.global.add.u32"));
        assert!(ptx.contains("SPILL_BUMP:"));
        assert!(ptx.contains("setp.eq.u64"));
    }
}
