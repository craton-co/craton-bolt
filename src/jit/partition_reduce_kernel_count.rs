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
//!
//! ## Intra-file dedup
//!
//! The non-spill [`compile_partition_reduce_kernel_count`] and the
//! [`compile_partition_reduce_kernel_count_with_spill`] sibling share ~80%
//! of their PTX body. The identical phases — the per-partition slice read,
//! the shared-array zero-init, the probe + atomic-count loop, and the
//! export — are factored here into the private `emit_*` helpers below.
//! Each helper emits the exact bytes both emitters previously wrote inline;
//! the few divergent tokens (the buffer-name suffix, the over-probe branch
//! target, the collision back-off, and the `MATCH` trailing branch) are
//! threaded through as parameters so the PTX golden snapshots
//! (`tests/ptx_golden_partition_snapshots.rs`) stay byte-identical.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

/// Key width selector for the unified COUNT(*) generator. Shared with the SUM
/// generator so the i32-key and i64-key COUNT wrappers (in this file and its
/// i64 sibling) both delegate to a single [`emit_count_kernel`].
use super::partition_reduce_kernel::KeyWidth;

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

/// Per-iteration `nanosleep.u32` operand for the collision-advance path
/// (sm_70+). See `partition_reduce_kernel::SPIN_BACKOFF_NS` for the full
/// rationale. TODO(perf): exponential back-off.
const SPIN_BACKOFF_NS: u32 = 32;

/// Entry-point name.
pub const KERNEL_ENTRY: &str = "bolt_partition_reduce_count";

/// Entry-point name for the spill-counter variant of the COUNT kernel.
/// Distinct from [`KERNEL_ENTRY`] so both PTX modules can coexist in the
/// JIT cache.
pub const KERNEL_ENTRY_WITH_SPILL: &str = "bolt_partition_reduce_count_spill";

// ---------------------------------------------------------------------------
// Unified key-width-parameterised COUNT(*) generator.
//
// `emit_count_kernel` emits BOTH the i32-key COUNT kernel (this file's public
// `compile_partition_reduce_kernel_count{,_with_spill}`) and the i64-key COUNT
// kernel (`partition_reduce_kernel_count_i64`'s
// `compile_partition_reduce_kernel_count_i64{,_with_spill}`, which delegate
// here). The whole scaffold — header, entry/regs framing, shmem-base +
// global-pointer setup, the per-partition offsets read, the publish/probe
// protocol, the CLAIM/MATCH accumulate, the LOOP epilogue, and the export-loop
// control flow — is written ONCE. Only the genuinely key-dependent bytes branch
// on `key_width`:
//
//   * shmem `block_keys` align + byte size (4 B/slot vs 8 B/slot) AND the
//     spill-variant buffer-name suffix (`_csp` for the older i32 COUNT kernel,
//     `_sp` for the i64 COUNT kernel — an asymmetry preserved verbatim),
//   * the spill `%rd<N>` register-file width (i32 spill keeps `%rd<64>`; i64
//     spill needs `%rd<80>`),
//   * the zero-init key store width + scratch-register order,
//   * the probe-prologue key load (`s32`+direct slot vs `s64`+`cvt.u32.u64`)
//     and the addr-compute scratch-register order,
//   * the publish/probe `PublishRegs` + key-type token,
//   * the CLAIM key store width,
//   * the export key load/store width + scratch-register order, and the
//     export-loop predicate numbers (the i64 spill variant's null-check
//     predicate shifts the export predicates by one).
//
// Every other byte is identical across all four (key_width × spill) variants.
// The 4 golden snapshots
// (`partition_reduce_count{,_spill}`, `partition_reduce_count_i64{,_spill}`) in
// `tests/ptx_golden_partition_snapshots.rs` pin the emitted bytes.
// ---------------------------------------------------------------------------

/// Emit the full per-partition COUNT(*) kernel for the given
/// `key_width`/`spill`/`entry`.
///
/// `spill == true` appends the trailing `spill_counter` `.u64` param + the
/// `SPILL_BUMP` overflow handler and drops the collision-advance back-off; the
/// i64 spill variant null-checks the counter pointer (the i32 spill variant,
/// older, bumps unconditionally — preserved for byte-stable golden parity).
pub(crate) fn emit_count_kernel(
    key_width: KeyWidth,
    spill: bool,
    entry: &str,
) -> BoltResult<String> {
    let mut ptx = String::new();
    let block_groups = BLOCK_GROUPS;
    let mask = BLOCK_GROUPS - 1;
    let block_threads = BLOCK_THREADS;
    let max_probes = MAX_PROBES;
    let n_params = if spill { 6 } else { 5 };
    // Buffer-name suffix asymmetry: the older i32 COUNT spill kernel uses
    // `_csp`; the i64 COUNT spill kernel uses `_sp`. Non-spill: no suffix.
    let suffix = match (spill, key_width) {
        (false, _) => "",
        (true, KeyWidth::I32) => "_csp",
        (true, KeyWidth::I64) => "_sp",
    };
    let overflow_target = if spill { "SPILL_BUMP" } else { "LOOP_NEXT" };

    super::partition_reduce_kernel_spill_common::emit_ptx_header(&mut ptx)?;

    // ---- shared-memory open-addressing table declarations -----------------
    // Three parallel arrays so each is naturally aligned. The key array's
    // align + byte size is the only key-width divergence here (4 B/slot for
    // i32, 8 B/slot for i64); counts (u64) and set (u32) are width-agnostic.
    let keys_align = match key_width {
        KeyWidth::I32 => 4,
        KeyWidth::I64 => 8,
    };
    let keys_bytes = match key_width {
        KeyWidth::I32 => block_groups * 4,
        KeyWidth::I64 => block_groups * 8,
    };
    let counts_bytes = block_groups * 8;
    let set_bytes = block_groups * 4;
    writeln!(
        ptx,
        ".shared .align {al} .b8 block_keys_buf{suffix}[{bytes}];",
        al = keys_align,
        bytes = keys_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 8 .b8 block_counts_buf{suffix}[{bytes}];",
        bytes = counts_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 4 .b8 block_set_buf{suffix}[{bytes}];",
        bytes = set_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ---- entry signature + register-file declarations ---------------------
    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    for i in 0..n_params {
        let comma = if i + 1 == n_params { "" } else { "," };
        writeln!(ptx, "\t.param .u64 {entry}_param_{i}{comma}").map_err(write_err)?;
    }
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<64>;").map_err(write_err)?;
    // Spill `%rd` width: i32 keeps the original `%rd<64>`; i64's spill variant
    // declares the wider `%rd<80>`. Non-spill kernels both use `%rd<64>` and
    // additionally declare the `%nstime` back-off operand.
    let rd_count = if spill && key_width == KeyWidth::I64 {
        80
    } else {
        64
    };
    writeln!(ptx, "\t.reg .b64   %rd<{rd_count}>;").map_err(write_err)?;
    if !spill {
        // Operand register for the per-collision `nanosleep.u32` back-off
        // (sm_70+). See partition_reduce_kernel.rs for full rationale.
        writeln!(ptx, "\t.reg .u32   %nstime;").map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    // --- thread coordinates -------------------------------------------------
    // %r0 = blockIdx.x = partition id, %r1 = blockDim.x, %r2 = threadIdx.x.
    super::partition_reduce_kernel_spill_common::emit_thread_block_ids(&mut ptx)?;

    // --- shared-memory base addresses --------------------------------------
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf{suffix};").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_counts_buf{suffix};").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf{suffix};").map_err(write_err)?;

    // --- global pointer setup (cvta from .param) ---------------------------
    // Param j lands in %rd{3 + j}: keys/offsets/out_keys/out_counts/out_set
    // (+ the spill_counter pointer in %rd8 for the spill variant). The COUNT
    // kernels read no value column, so the offsets pointer is %rd4 (not %rd5).
    for j in 0..n_params {
        let rd = 3 + j;
        writeln!(ptx, "\tld.param.u64 %rd{rd}, [{entry}_param_{j}];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd{rd}, %rd{rd};").map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    // --- read this block's partition slice [start, end) into %r10/%r11 -----
    emit_offsets_read(&mut ptx)?;

    // ---- Phase 1: cooperatively zero the shared arrays --------------------
    emit_zero_phase(&mut ptx, key_width, block_groups, block_threads)?;

    // ---- Phase 2: probe + atomic count over this partition's rows ---------
    emit_probe_loop(&mut ptx, key_width, mask, max_probes, overflow_target, spill)?;

    if spill {
        // SPILL_BUMP: bump the spill counter, then fall to the epilogue. The
        // i32 variant (older) bumps unconditionally; the i64 variant
        // null-checks the pointer so callers can opt out with 0. Both shapes
        // are byte-stable against their golden snapshots.
        match key_width {
            KeyWidth::I32 => {
                super::partition_reduce_kernel_spill_common::emit_spill_bump_unchecked(
                    &mut ptx, 8,
                )?;
            }
            KeyWidth::I64 => {
                super::partition_reduce_kernel_spill_common::emit_spill_bump_with_null_check(
                    &mut ptx, 8,
                )?;
            }
        }
        super::partition_reduce_kernel_spill_common::emit_loop_next_done(&mut ptx)?;
    } else {
        writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
        writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
        writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
        writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
        writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
        writeln!(ptx).map_err(write_err)?;
    }

    // ---- Phase 3: per-slot export to global memory ------------------------
    // The export loop/set predicates are normally `%p5`/`%p6`. ONLY the i64
    // spill variant shifts them to `%p6`/`%p7`: its `SPILL_BUMP` handler emits
    // a `setp.eq.u64 %p5` null-check that consumes `%p5`. The i32 spill variant
    // (older, unchecked `SPILL_BUMP`) emits NO such predicate, so it keeps the
    // base `%p5`/`%p6` — matching its golden snapshot exactly.
    let shift_export_preds = spill && key_width == KeyWidth::I64;
    let (loop_pred, set_pred) = if shift_export_preds {
        ("%p6", "%p7")
    } else {
        ("%p5", "%p6")
    };
    emit_export_phase(
        &mut ptx,
        key_width,
        block_groups,
        block_threads,
        loop_pred,
        set_pred,
    )?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Emit the per-partition slice read: `start = offsets[pid]`,
/// `end = offsets[pid+1]` into `%r10` / `%r11`. The COUNT kernels keep the
/// offsets pointer in `%rd4` (their reduced param list has no values array),
/// so this is NOT the `spill_common::emit_partition_slice_read` helper, which
/// expects `%rd5`. Byte-identical across key widths and spill variants.
fn emit_offsets_read(ptx: &mut String) -> BoltResult<()> {
    // start, end = offsets[pid], offsets[pid+1]
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd4, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd11, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd12];").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;
    Ok(())
}

/// Emit Phase 1 — the strided zero-init of `block_keys` / `block_counts` /
/// `block_set` followed by `bar.sync 0`. The key store width + scratch-register
/// order branch on `key_width`; everything else is shared.
fn emit_zero_phase(
    ptx: &mut String,
    key_width: KeyWidth,
    block_groups: u32,
    block_threads: u32,
) -> BoltResult<()> {
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r20, {bg};", bg = block_groups).map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    match key_width {
        KeyWidth::I32 => {
            // block_keys[s] = 0  (i32, 4 B)
            writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u32 [%rd21], 0;").map_err(write_err)?;
            // block_counts[s] = 0  (u64, 8 B)
            writeln!(ptx, "\tmul.wide.u32 %rd22, %r20, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd22;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u64 [%rd23], 0;").map_err(write_err)?;
            // block_set[s] = 0  (u32, 4 B, addressed at rd2 + s*4)
            writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd20;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u32 [%rd24], 0;").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            // block_keys[s] = 0  (i64, 8 B)
            writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u64 [%rd21], 0;").map_err(write_err)?;
            // block_counts[s] = 0  (u64, 8 B) — same stride, reuses %rd20.
            writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd20;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u64 [%rd23], 0;").map_err(write_err)?;
            // block_set[s] = 0  (u32, 4 B)
            writeln!(ptx, "\tmul.wide.u32 %rd22, %r20, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd22;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u32 [%rd24], 0;").map_err(write_err)?;
        }
    }
    writeln!(ptx, "\tadd.u32 %r20, %r20, {bt};", bt = block_threads).map_err(write_err)?;
    writeln!(ptx, "\tbra ZERO_TOP;").map_err(write_err)?;
    writeln!(ptx, "ZERO_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;
    Ok(())
}

/// Emit Phase 2 — the probe + atomic-count loop, from the `LOOP_TOP` setup
/// through the `MATCH` accumulate. The key load (`s32`+direct slot vs
/// `s64`+`cvt.u32.u64`), the addr-compute scratch-register order, the
/// publish/probe register tuple + key-type token, and the CLAIM key store width
/// branch on `key_width`. The remaining divergence points are:
///
///   * `over_probe_target` — the `@%p2 bra <target>` taken when the linear
///     probe exceeds `max_probes`: `"LOOP_NEXT"` (non-spill, silently drops
///     the row) vs `"SPILL_BUMP"` (spill, bumps the counter).
///   * `spill` — when `false`, the collision-advance path emits the
///     `nanosleep.u32` occupancy back-off before `bra PROBE_TOP`; the spill
///     path drops it. When `true`, the `MATCH` accumulate ends with an
///     explicit `bra LOOP_NEXT;` (the spill emitter follows `MATCH` with the
///     `SPILL_BUMP` block, so it cannot fall through); the non-spill emitter
///     lets `MATCH` fall straight into the `LOOP_NEXT` label that immediately
///     follows it.
fn emit_probe_loop(
    ptx: &mut String,
    key_width: KeyWidth,
    mask: u32,
    max_probes: u32,
    over_probe_target: &str,
    spill: bool,
) -> BoltResult<()> {
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    match key_width {
        KeyWidth::I32 => {
            // key = partition_keys[i] (i32)
            writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
            writeln!(ptx, "\tld.global.s32 %r31, [%rd31];").map_err(write_err)?;

            // slot = key & mask
            writeln!(ptx, "\tand.b32 %r32, %r31, 0x{mask:X};", mask = mask).map_err(write_err)?;
        }
        KeyWidth::I64 => {
            // key = partition_keys[i] (i64)
            writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
            writeln!(ptx, "\tld.global.s64 %rd60, [%rd31];").map_err(write_err)?;

            // slot from low 32 bits.
            writeln!(ptx, "\tcvt.u32.u64 %r31, %rd60;").map_err(write_err)?;
            writeln!(ptx, "\tand.b32 %r32, %r31, 0x{mask:X};", mask = mask).map_err(write_err)?;
        }
    }
    writeln!(ptx, "\tmov.u32 %r33, 0;").map_err(write_err)?;

    writeln!(ptx, "PROBE_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r33, %r33, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.gt.u32 %p2, %r33, {mp};", mp = max_probes).map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra {target};", target = over_probe_target).map_err(write_err)?;

    // Slot addresses. The set/count strides (×4 / ×8) are width-agnostic; the
    // key slot stride (×4 for i32, ×8 for i64) and the resulting
    // scratch-register order are the only divergence.
    match key_width {
        KeyWidth::I32 => {
            //   addr_set   = block_set    + slot * 4
            //   addr_key   = block_keys   + slot * 4
            //   addr_count = block_counts + slot * 8
            writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd34;").map_err(write_err)?;
            writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            //   addr_set   = block_set    + slot * 4
            //   addr_key   = block_keys   + slot * 8   (i64 slot stride)
            //   addr_count = block_counts + slot * 8
            writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?;
            writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd37;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?;
        }
    }

    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;

    // 3-state publish protocol (claim-then-write race fix; see
    // partition_reduce_kernel.rs). Slot occupied (set is 1=claiming or
    // 2=ready). Spin on a VOLATILE SHARED re-read of set (a bare
    // ld.acquire.cta defaults to global space and faults on the shared
    // offset) until the claimer publishes set:=2, yielding via nanosleep so
    // the same-warp claimer can run, THEN read the key.
    let (key_dst_reg, probe_key_reg, key_ty) = match key_width {
        KeyWidth::I32 => ("%r35", "%r31", "s32"),
        KeyWidth::I64 => ("%rd61", "%rd60", "s64"),
    };
    super::partition_reduce_kernel_spill_common::emit_publish_probe_protocol(
        ptx,
        &super::partition_reduce_kernel_spill_common::PublishRegs {
            set_flag_reg: "%r36",
            set_addr_reg: "%rd35",
            key_addr_reg: "%rd36",
            key_dst_reg,
            probe_key_reg,
        },
        key_ty,
    )?;
    writeln!(ptx, "\tadd.u32 %r32, %r32, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r32, %r32, 0x{mask:X};", mask = mask).map_err(write_err)?;
    if !spill {
        // Occupancy-friendly back-off on the collision-advance path
        // (sm_70+). See partition_reduce_kernel.rs for full rationale. The
        // spill variant omits the back-off and jumps straight to PROBE_TOP.
        super::partition_reduce_kernel_spill_common::emit_spin_backoff(ptx, SPIN_BACKOFF_NS)?;
    }
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    // CLAIM: publish the key, fence so racing readers see it, then
    // atomically add 1 to the count.
    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    match key_width {
        KeyWidth::I32 => {
            writeln!(ptx, "\tst.shared.u32 [%rd36], %r31;").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            writeln!(ptx, "\tst.shared.u64 [%rd36], %rd60;").map_err(write_err)?;
        }
    }
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd35], 2;").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.u64 %rd40, [%rd38], 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    writeln!(ptx, "MATCH:").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.u64 %rd41, [%rd38], 1;").map_err(write_err)?;
    if spill {
        // The spill emitter places the SPILL_BUMP block immediately after
        // MATCH, so MATCH must branch over it; the non-spill emitter falls
        // straight through into the LOOP_NEXT label that follows.
        writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;
    }
    Ok(())
}

/// Emit Phase 3 — the strided export of `(key, count, set)` from shared to
/// global. The key load/store width and the export-block scratch-register order
/// branch on `key_width`; `loop_pred`/`set_pred` are the bound / set-coercion
/// predicates (the i64 spill variant's null-check shifts them by one).
fn emit_export_phase(
    ptx: &mut String,
    key_width: KeyWidth,
    block_groups: u32,
    block_threads: u32,
    loop_pred: &str,
    set_pred: &str,
) -> BoltResult<()> {
    writeln!(ptx, "\tmul.lo.u32 %r40, %r0, {bg};", bg = block_groups).map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r41, %r2;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 {loop_pred}, %r41, {bg};", bg = block_groups)
        .map_err(write_err)?;
    writeln!(ptx, "\t@{loop_pred} bra EXPORT_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r42, %r40, %r41;").map_err(write_err)?;

    match key_width {
        KeyWidth::I32 => {
            writeln!(ptx, "\tmul.wide.u32 %rd42, %r41, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd43, %rd0, %rd42;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.s32 %r43, [%rd43];").map_err(write_err)?;
            writeln!(ptx, "\tmul.wide.u32 %rd44, %r41, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd45, %rd1, %rd44;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.u64 %rd46, [%rd45];").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd47, %rd2, %rd42;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.u32 %r44, [%rd47];").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            // i64 key (8 B), u64 count (8 B, reuses the key offset %rd44), u32 set.
            writeln!(ptx, "\tmul.wide.u32 %rd44, %r41, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd43, %rd0, %rd44;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.s64 %rd62, [%rd43];").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd45, %rd1, %rd44;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.u64 %rd46, [%rd45];").map_err(write_err)?;
            writeln!(ptx, "\tmul.wide.u32 %rd42, %r41, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd47, %rd2, %rd42;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.u32 %r44, [%rd47];").map_err(write_err)?;
        }
    }
    writeln!(ptx, "\tsetp.ne.s32 {set_pred}, %r44, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r45, 1, 0, {set_pred};").map_err(write_err)?;

    match key_width {
        KeyWidth::I32 => {
            writeln!(ptx, "\tmul.wide.u32 %rd48, %r42, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd49, %rd5, %rd48;").map_err(write_err)?;
            writeln!(ptx, "\tst.global.s32 [%rd49], %r43;").map_err(write_err)?;
            writeln!(ptx, "\tmul.wide.u32 %rd50, %r42, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd51, %rd6, %rd50;").map_err(write_err)?;
            writeln!(ptx, "\tst.global.u64 [%rd51], %rd46;").map_err(write_err)?;
            writeln!(ptx, "\tcvt.u64.u32 %rd52, %r42;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd53, %rd7, %rd52;").map_err(write_err)?;
            writeln!(ptx, "\tst.global.u8 [%rd53], %r45;").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            // out_keys + global_slot * 8   (i64; count reuses the offset %rd48)
            writeln!(ptx, "\tmul.wide.u32 %rd48, %r42, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd49, %rd5, %rd48;").map_err(write_err)?;
            writeln!(ptx, "\tst.global.s64 [%rd49], %rd62;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd51, %rd6, %rd48;").map_err(write_err)?;
            writeln!(ptx, "\tst.global.u64 [%rd51], %rd46;").map_err(write_err)?;
            writeln!(ptx, "\tcvt.u64.u32 %rd52, %r42;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd53, %rd7, %rd52;").map_err(write_err)?;
            writeln!(ptx, "\tst.global.u8 [%rd53], %r45;").map_err(write_err)?;
        }
    }

    writeln!(ptx, "\tadd.u32 %r41, %r41, {bt};", bt = block_threads).map_err(write_err)?;
    writeln!(ptx, "\tbra EXPORT_TOP;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_DONE:").map_err(write_err)?;
    Ok(())
}

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
/// Launch geometry: `grid = NUM_PARTITIONS, block = BLOCK_THREADS`. Delegates to
/// the unified [`emit_count_kernel`] with the `I32` key width.
pub fn compile_partition_reduce_kernel_count() -> BoltResult<String> {
    emit_count_kernel(KeyWidth::I32, false, KERNEL_ENTRY)
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
/// cache. Delegates to the unified [`emit_count_kernel`] with the `I32` key
/// width and `spill = true`.
pub fn compile_partition_reduce_kernel_count_with_spill() -> BoltResult<String> {
    emit_count_kernel(KeyWidth::I32, true, KERNEL_ENTRY_WITH_SPILL)
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

    /// Byte-stable refactor guard: the header / thread-id / spin-back-off
    /// fragments now come from the shared `spill_common` helpers. Assert the
    /// exact bytes they previously emitted inline are unchanged.
    #[test]
    fn shared_fragment_bytes_are_byte_stable() {
        let ptx = compile_partition_reduce_kernel_count().unwrap();
        assert!(ptx.starts_with(".version 7.5\n.target sm_70\n.address_size 64\n\n"));
        assert!(ptx.contains(
            "\tmov.u32 %r0, %ctaid.x;\n\tmov.u32 %r1, %ntid.x;\n\tmov.u32 %r2, %tid.x;\n"
        ));
        assert!(ptx.contains("\tmov.u32 %nstime, 32;\n\tnanosleep.u32 %nstime;\n"));
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
