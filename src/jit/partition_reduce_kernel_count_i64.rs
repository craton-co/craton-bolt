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
//!
//! ## Shared emit helpers
//!
//! The non-spill [`compile_partition_reduce_kernel_count_i64`] and the
//! spill-aware [`compile_partition_reduce_kernel_count_i64_with_spill`]
//! are ~80% byte-identical. The shared phases are factored into the
//! `emit_*` private helpers below; each helper emits bytes that are
//! literally identical across both callers. The publish/probe protocol
//! is shared one level up via
//! `super::partition_reduce_kernel_spill_common::emit_publish_probe_protocol`.
//! Only the genuinely divergent bits (shared-buffer symbol suffix,
//! register-bank sizes, the PROBE_TOP overflow target, the spin-backoff,
//! the MATCH fall-through, the spill param/bump, and the two export
//! predicate numbers) remain per-emitter.

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

// ---------------------------------------------------------------------------
// Shared emit helpers (THIS FILE ONLY).
//
// Each helper emits the exact byte sequence that was previously inlined,
// identically, into both emitters. Where a phase had identical bytes in
// both, it takes no extra parameters; where two tokens genuinely differed
// (export predicate numbers), those are passed in so the helper still
// reproduces today's output verbatim per-caller.
// ---------------------------------------------------------------------------

/// Shared-buffer symbol suffix: `""` for the non-spill kernel,
/// `"_sp"` for the spill kernel. Used to keep the `.shared` decls and
/// the `mov.u64 %rd0..2, <buf>` lines byte-identical to today.
type BufSuffix = &'static str;

/// Emit the three `.shared` buffer declarations (keys/counts/set) plus a
/// trailing blank line. The symbol names take `suffix` so the spill
/// kernel emits the `_sp`-suffixed names exactly as before.
fn emit_shared_decls(ptx: &mut String, suffix: BufSuffix) -> BoltResult<()> {
    let keys_bytes = BLOCK_GROUPS * 8; // i64
    let counts_bytes = BLOCK_GROUPS * 8;
    let set_bytes = BLOCK_GROUPS * 4;
    writeln!(
        ptx,
        ".shared .align 8 .b8 block_keys_buf{suffix}[{bytes}];",
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
    Ok(())
}

/// Emit the thread/block id setup, the three `mov.u64 %rd0..2, <buf>`
/// lines (suffix-aware), and the `%rd3..=%rd7` param load + `cvta` loop.
/// Stops *before* the spill kernel's extra `%rd8` spill-pointer load so
/// the divergent tail stays per-caller.
fn emit_ids_and_base_params(
    ptx: &mut String,
    entry: &str,
    suffix: BufSuffix,
) -> BoltResult<()> {
    super::partition_reduce_kernel_spill_common::emit_thread_block_ids(ptx)?;
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf{suffix};").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_counts_buf{suffix};").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf{suffix};").map_err(write_err)?;

    for (rd, p) in (3..=7).zip(0..5) {
        writeln!(ptx, "\tld.param.u64 %rd{rd}, [{entry}_param_{p}];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd{rd}, %rd{rd};").map_err(write_err)?;
    }
    Ok(())
}

/// Emit the per-partition offset load (`[lo, hi)` row bounds into
/// `%r10`/`%r11`) plus a trailing blank line. Byte-identical in both.
fn emit_offsets_load(ptx: &mut String) -> BoltResult<()> {
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd4, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd11, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd12];").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;
    Ok(())
}

/// Phase 1: cooperative zero-init of the three shared buffers
/// (ZERO_TOP/ZERO_DONE), ending with `bar.sync 0;` and a blank line.
/// Byte-identical in both emitters.
fn emit_zero_shared(ptx: &mut String) -> BoltResult<()> {
    let block_groups = BLOCK_GROUPS;
    let block_threads = BLOCK_THREADS;
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r20, {bg};", bg = block_groups).map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd21], 0;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd23], 0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd22, %r20, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd22;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd24], 0;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r20, %r20, {bt};", bt = block_threads).map_err(write_err)?;
    writeln!(ptx, "\tbra ZERO_TOP;").map_err(write_err)?;
    writeln!(ptx, "ZERO_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;
    Ok(())
}

/// Phase 2 prologue: loop setup, the i64 key load, slot derivation, the
/// PROBE_TOP probe-count guard (whose overflow target differs), the slot
/// address math, the CAS-claim, and the claim-success branch to `CLAIM`.
/// Emits everything up to (but not including) the publish/probe protocol.
///
/// `overflow_target` is the `@%p2 bra <label>;` destination on probe
/// exhaustion: `"LOOP_NEXT"` for the non-spill kernel (drop the row),
/// `"SPILL_BUMP"` for the spill kernel (bump the spill counter).
fn emit_probe_prologue(ptx: &mut String, overflow_target: &str) -> BoltResult<()> {
    let mask = BLOCK_GROUPS - 1;
    let max_probes = MAX_PROBES;

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
    writeln!(ptx, "\tand.b32 %r32, %r31, 0x{mask:X};", mask = mask).map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r33, 0;").map_err(write_err)?;

    writeln!(ptx, "PROBE_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r33, %r33, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.gt.u32 %p2, %r33, {mp};", mp = max_probes).map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra {overflow_target};").map_err(write_err)?;

    // Addrs (set: *4, key: *8 i64, count: *8).
    writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd37;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?;

    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;
    Ok(())
}

/// Emit the shared publish/probe protocol call followed by the
/// collision-advance slot bump (`+1` then mask). Byte-identical in both.
/// The non-spill kernel emits a spin-backoff *after* this (divergent),
/// so the trailing `bra PROBE_TOP;` is left to the caller.
fn emit_protocol_and_advance(ptx: &mut String) -> BoltResult<()> {
    let mask = BLOCK_GROUPS - 1;
    super::partition_reduce_kernel_spill_common::emit_publish_probe_protocol(
        ptx,
        &super::partition_reduce_kernel_spill_common::PublishRegs {
            set_flag_reg: "%r36",
            set_addr_reg: "%rd35",
            key_addr_reg: "%rd36",
            key_dst_reg: "%rd61",
            probe_key_reg: "%rd60",
        },
        "s64",
    )?;
    writeln!(ptx, "\tadd.u32 %r32, %r32, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r32, %r32, 0x{mask:X};", mask = mask).map_err(write_err)?;
    Ok(())
}

/// Emit the `CLAIM:` block: publish the i64 key, fence, publish set:=2,
/// bump the count, branch to `LOOP_NEXT`. Byte-identical in both.
fn emit_claim_block(ptx: &mut String) -> BoltResult<()> {
    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd36], %rd60;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd35], 2;").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.u64 %rd40, [%rd38], 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;
    Ok(())
}

/// Phase 3: cooperative export of populated slots
/// (EXPORT_TOP/EXPORT_DONE) plus the trailing `ret;` and closing brace.
///
/// The two export predicates differ between emitters (`%p5`/`%p6` in the
/// non-spill kernel, `%p6`/`%p7` in the spill kernel), so they are passed
/// in; every other byte is identical.
fn emit_export_and_return(
    ptx: &mut String,
    loop_pred: &str,
    set_pred: &str,
) -> BoltResult<()> {
    let block_groups = BLOCK_GROUPS;
    let block_threads = BLOCK_THREADS;

    writeln!(ptx, "\tmul.lo.u32 %r40, %r0, {bg};", bg = block_groups).map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r41, %r2;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 {loop_pred}, %r41, {bg};", bg = block_groups)
        .map_err(write_err)?;
    writeln!(ptx, "\t@{loop_pred} bra EXPORT_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r42, %r40, %r41;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd44, %r41, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd43, %rd0, %rd44;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s64 %rd62, [%rd43];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd45, %rd1, %rd44;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u64 %rd46, [%rd45];").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd42, %r41, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd47, %rd2, %rd42;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r44, [%rd47];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 {set_pred}, %r44, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r45, 1, 0, {set_pred};").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd48, %r42, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd49, %rd5, %rd48;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s64 [%rd49], %rd62;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd51, %rd6, %rd48;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u64 [%rd51], %rd46;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u64.u32 %rd52, %r42;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd53, %rd7, %rd52;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd53], %r45;").map_err(write_err)?;

    writeln!(ptx, "\tadd.u32 %r41, %r41, {bt};", bt = block_threads).map_err(write_err)?;
    writeln!(ptx, "\tbra EXPORT_TOP;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_DONE:").map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;
    Ok(())
}

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

    super::partition_reduce_kernel_spill_common::emit_ptx_header(&mut ptx)?;

    emit_shared_decls(&mut ptx, "")?;

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

    emit_ids_and_base_params(&mut ptx, entry, "")?;
    writeln!(ptx).map_err(write_err)?;

    emit_offsets_load(&mut ptx)?;

    // Phase 1: zero shared.
    emit_zero_shared(&mut ptx)?;

    // Phase 2: probe + atomic count. Non-spill drops rows on overflow.
    emit_probe_prologue(&mut ptx, "LOOP_NEXT")?;

    // MATCH path: membar.cta to order the CAS (on block_set) against
    // the subsequent key load (on block_keys). Different addresses, so
    // PTX sm_70 requires an explicit fence — without it a racing thread
    // can read a zero key under set==1 and false-match key 0.
    // 3-state publish protocol (claim-then-write race fix; set is u32 at
    // %rd35, key is i64 at %rd36). VOLATILE SHARED re-read of set + nanosleep
    // yield until the claimer publishes set:=2, THEN read the i64 key.
    emit_protocol_and_advance(&mut ptx)?;
    // Occupancy-friendly back-off on the collision-advance path.
    super::partition_reduce_kernel_spill_common::emit_spin_backoff(&mut ptx, SPIN_BACKOFF_NS)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    emit_claim_block(&mut ptx)?;

    writeln!(ptx, "MATCH:").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.u64 %rd41, [%rd38], 1;").map_err(write_err)?;

    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 3: export populated slots.
    emit_export_and_return(&mut ptx, "%p5", "%p6")?;

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

    super::partition_reduce_kernel_spill_common::emit_ptx_header(&mut ptx)?;

    emit_shared_decls(&mut ptx, "_sp")?;

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

    emit_ids_and_base_params(&mut ptx, entry, "_sp")?;
    // %rd8 is the spill counter pointer (spill-only tail of param setup).
    writeln!(ptx, "\tld.param.u64 %rd8, [{entry}_param_5];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd8, %rd8;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    emit_offsets_load(&mut ptx)?;

    // Phase 1: zero shared.
    emit_zero_shared(&mut ptx)?;

    // Phase 2. Spill kernel bumps the spill counter on overflow.
    emit_probe_prologue(&mut ptx, "SPILL_BUMP")?;

    // 3-state publish protocol (claim-then-write race fix).
    emit_protocol_and_advance(&mut ptx)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    emit_claim_block(&mut ptx)?;

    writeln!(ptx, "MATCH:").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.u64 %rd41, [%rd38], 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    super::partition_reduce_kernel_spill_common::emit_spill_bump_with_null_check(&mut ptx, 8)?;

    super::partition_reduce_kernel_spill_common::emit_loop_next_done(&mut ptx)?;

    // Phase 3: export.
    emit_export_and_return(&mut ptx, "%p6", "%p7")?;

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
