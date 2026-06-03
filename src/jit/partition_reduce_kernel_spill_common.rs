// SPDX-License-Identifier: Apache-2.0

//! Shared PTX-emission helpers for `partition_reduce_kernel*` `_with_spill`
//! variants.
//!
//! ## Why
//!
//! The 10 `compile_*_with_spill()` emitters in `src/jit/partition_reduce_kernel*.rs`
//! each hand-write the same probing/CAS/membar.cta scaffolding for their
//! own (key-type, value-type, op) tuple. Most of those bodies differ — the
//! shmem layout, register file, and atomic op are per-kernel — but a
//! handful of fragments are emitted byte-for-byte identically across
//! every spill variant:
//!
//! 1. The PTX module header (`.version 7.5` / `.target sm_70` /
//!    `.address_size 64` + blank line).
//! 2. The block/thread-id register setup (`mov.u32 %r0,%ctaid.x` etc.).
//! 3. The `SPILL_BUMP:` label + null-check + `atom.global.add.u32`
//!    sequence (the 8 kernels that null-check; 2 omit the check).
//! 4. The `LOOP_NEXT:` / `LOOP_DONE:` epilogue.
//!
//! This module factors those exact byte sequences into helpers so the
//! probing scaffolding is written once. Each helper emits the same bytes
//! the inline `writeln!` calls would produce — the PTX-shape tests in
//! every sibling file continue to pass without modification.
//!
//! ## Scope limits
//!
//! Many further-similar-looking chunks (the zero-init phase, the probe
//! loop body, the export phase) **are not** byte-for-byte identical
//! across kernels because they encode the per-kernel key/value widths,
//! the atomic op, and the `multi`-variant's parametric value count. We
//! deliberately do NOT try to share those — a parameterised helper would
//! either change the emitted PTX byte sequence or fan out into a giant
//! match that defeats the dedup goal.
//!
//! ## Contract
//!
//! Every helper here MUST emit bytes that match exactly what the inline
//! code in the caller previously wrote. The PTX-shape tests in each
//! sibling file (which check for substrings like `"SPILL_BUMP:"`,
//! `"atom.global.add.u32"`, `"setp.eq.u64"`, `".version 7.5"`, etc.)
//! verify this in aggregate.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

/// Convert a `std::fmt::Error` from `writeln!` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!(
        "partition_reduce_kernel_spill_common: write failed: {}",
        e
    ))
}

/// Emit the PTX module header: `.version 7.5`, `.target sm_70`,
/// `.address_size 64`, then a blank line.
///
/// Identical across all 10 `_with_spill` kernels (and their non-spill
/// siblings, though we only call from the spill paths to avoid touching
/// the base emitters).
pub(crate) fn emit_ptx_header(ptx: &mut String) -> BoltResult<()> {
    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;
    Ok(())
}

/// Emit the thread/block-id register setup, the first three `mov.u32`
/// instructions inside the kernel body:
///
/// ```text
/// \tmov.u32 %r0, %ctaid.x;
/// \tmov.u32 %r1, %ntid.x;
/// \tmov.u32 %r2, %tid.x;
/// ```
///
/// Identical across all 11 `partition_reduce_kernel*` emitters (both the
/// non-spill base kernels and their `_with_spill` siblings).
pub(crate) fn emit_thread_block_ids(ptx: &mut String) -> BoltResult<()> {
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    Ok(())
}

/// Emit the per-collision occupancy back-off pair on the probe
/// collision-advance path:
///
/// ```text
/// \tmov.u32 %nstime, 32;
/// \tnanosleep.u32 %nstime;
/// ```
///
/// `ns` is the `SPIN_BACKOFF_NS` constant (32 in every current emitter).
/// This pair is emitted byte-for-byte identically by every non-spill
/// integer/SUM/COUNT/MIN-MAX emitter right after the collision-advance
/// `and.b32` and before the `bra PROBE_TOP`. The spill siblings drop the
/// back-off (their collision path jumps straight back to `PROBE_TOP`), so
/// this helper is only used by the non-spill base kernels.
pub(crate) fn emit_spin_backoff(ptx: &mut String, ns: u32) -> BoltResult<()> {
    writeln!(ptx, "\tmov.u32 %nstime, {ns};").map_err(write_err)?;
    writeln!(ptx, "\tnanosleep.u32 %nstime;").map_err(write_err)?;
    Ok(())
}

/// Emit the `SPILL_BUMP:` label, a null-check that skips to `LOOP_NEXT`
/// when the spill counter pointer is `0`, and the
/// `atom.global.add.u32 ..., 1` that bumps the counter when the pointer
/// is non-null.
///
/// `rd_spill_idx` is the index of the `%rd*` register holding the
/// (already cvta-ed) spill-counter pointer. For example, the i64-key
/// SUM kernel uses `%rd9`; pass `9` here.
///
/// Emitted bytes (with `rd_spill_idx = 9`):
///
/// ```text
/// SPILL_BUMP:
/// \tsetp.eq.u64 %p5, %rd9, 0;
/// \t@%p5 bra LOOP_NEXT;
/// \tatom.global.add.u32 %r36, [%rd9], 1;
/// ```
///
/// Used by 8 of 10 spill kernels (all except the i32-key SUM and the
/// i32-key COUNT, which were written before the null-check was added
/// and remain that way to keep their golden PTX byte-stable).
pub(crate) fn emit_spill_bump_with_null_check(
    ptx: &mut String,
    rd_spill_idx: u32,
) -> BoltResult<()> {
    writeln!(ptx, "SPILL_BUMP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.u64 %p5, %rd{rd_spill_idx}, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra LOOP_NEXT;").map_err(write_err)?;
    writeln!(ptx, "\tatom.global.add.u32 %r36, [%rd{rd_spill_idx}], 1;").map_err(write_err)?;
    Ok(())
}

/// Emit the `SPILL_BUMP:` label followed by an unconditional
/// `atom.global.add.u32` bump (no null check). Two of the older spill
/// emitters (i32-key SUM, i32-key COUNT) ship this shape; the helper
/// preserves their exact bytes.
///
/// Emitted bytes (with `rd_spill_idx = 9`):
///
/// ```text
/// SPILL_BUMP:
/// \tatom.global.add.u32 %r36, [%rd9], 1;
/// ```
pub(crate) fn emit_spill_bump_unchecked(ptx: &mut String, rd_spill_idx: u32) -> BoltResult<()> {
    writeln!(ptx, "SPILL_BUMP:").map_err(write_err)?;
    writeln!(ptx, "\tatom.global.add.u32 %r36, [%rd{rd_spill_idx}], 1;").map_err(write_err)?;
    Ok(())
}

/// Emit the per-block partition-slice read that loads `[start, end)` from
/// the `partition_offsets` buffer (held in `%rd5`) for this block's
/// partition id (`%r0`):
///
/// ```text
/// \tmul.wide.u32 %rd10, %r0, 4;
/// \tadd.s64 %rd11, %rd5, %rd10;
/// \tld.global.u32 %r10, [%rd11];
/// \tadd.s64 %rd12, %rd11, 4;
/// \tld.global.u32 %r11, [%rd12];
/// ```
///
/// `%r10` receives `start = partition_offsets[pid]` and `%r11` receives
/// `end = partition_offsets[pid + 1]`. The fixed register allocation
/// (`%rd10`/`%rd11`/`%rd12`, `%r10`/`%r11`) and the offsets pointer in
/// `%rd5` are shared verbatim by the 6 scalar-key SUM / MIN-MAX emitters
/// (SUM and MIN/MAX-int and MIN/MAX-float, each in its i32-key and
/// i64-key form, and each in its non-spill and `_with_spill` variant —
/// 12 call sites total).
///
/// The `count` kernels are deliberately NOT callers: their reduced
/// parameter list (no separate values array) shifts the offsets pointer
/// to `%rd4`, so they emit `add.s64 %rd11, %rd4, %rd10` and would drift
/// by one register if routed through this helper. The `multi` kernels
/// place the offsets pointer in a parametric `%rd` register and use a
/// different scratch register block (`%rd80`..). Both keep their own
/// inline emission.
pub(crate) fn emit_partition_slice_read(ptx: &mut String) -> BoltResult<()> {
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd5, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd11, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd12];").map_err(write_err)?;
    Ok(())
}

/// Emit the post-loop epilogue:
///
/// ```text
/// LOOP_NEXT:
/// \tadd.u32 %r30, %r30, %r1;
/// \tbra LOOP_TOP;
/// LOOP_DONE:
/// \tbar.sync 0;
/// (blank line)
/// ```
///
/// Identical across every `_with_spill` kernel (and across the base
/// emitters too — they share the same post-loop boilerplate).
pub(crate) fn emit_loop_next_done(ptx: &mut String) -> BoltResult<()> {
    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;
    Ok(())
}

/// Emit the grid-stride row-loop head, shared verbatim by every
/// `partition_reduce_kernel*` op family (SUM, COUNT, MIN/MAX-int,
/// MIN/MAX-float, MULTI), in both key widths and spill variants:
///
/// ```text
/// \tadd.u32 %r30, %r10, %r2;
/// LOOP_TOP:
/// \tsetp.ge.u32 %p1, %r30, %r11;
/// \t@%p1 bra LOOP_DONE;
/// ```
///
/// `%r30` is the per-thread row cursor (`start + tid`), `%r10`/`%r11` hold
/// the partition slice `[start, end)`, and the loop exits to `LOOP_DONE`.
/// Every byte is fixed (no register or label substitution), so all callers
/// share the identical sequence.
pub(crate) fn emit_loop_head(ptx: &mut String) -> BoltResult<()> {
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;
    Ok(())
}

/// Emit the probe-counter init + `PROBE_TOP` bound-check head shared verbatim
/// by every op family:
///
/// ```text
/// \tmov.u32 %r33, 0;
/// PROBE_TOP:
/// \tadd.u32 %r33, %r33, 1;
/// \tsetp.gt.u32 %p2, %r33, {max_probes};
/// \t@%p2 bra {overflow_target};
/// ```
///
/// `%r33` is the per-row probe counter; on exceeding `max_probes` the kernel
/// branches to `overflow_target` (`"LOOP_NEXT"` for the non-spill kernels,
/// which silently drop the row, or `"SPILL_BUMP"` for the spill kernels).
/// Both operands are substituted verbatim, reproducing the prior inline bytes.
pub(crate) fn emit_probe_bound_check(
    ptx: &mut String,
    max_probes: u32,
    overflow_target: &str,
) -> BoltResult<()> {
    writeln!(ptx, "\tmov.u32 %r33, 0;").map_err(write_err)?;
    writeln!(ptx, "PROBE_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r33, %r33, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.gt.u32 %p2, %r33, {max_probes};").map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra {overflow_target};").map_err(write_err)?;
    Ok(())
}

/// Emit the slot-claim `atom.shared.cas.b32` + the `@%p3 bra CLAIM;` branch,
/// shared verbatim by every op family (only the set-flag address operand
/// differs — `%rd35` for the scalar/min-max kernels, `%rd93` for MULTI):
///
/// ```text
/// \tatom.shared.cas.b32 %r34, [{set_addr_reg}], 0, 1;
/// \tsetp.eq.s32 %p3, %r34, 0;
/// \t@%p3 bra CLAIM;
/// ```
///
/// `%r34` receives the prior set value; `%p3` is true (→ `CLAIM`) when this
/// thread won the empty slot. `set_addr_reg` is the full `%rd*` token for the
/// `block_set[slot]` address.
pub(crate) fn emit_slot_claim_cas(ptx: &mut String, set_addr_reg: &str) -> BoltResult<()> {
    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [{set_addr_reg}], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;
    Ok(())
}

/// Emit the CLAIM-path publish fence + `set:=2` release, shared verbatim by
/// every op family (only the set-flag address operand differs — `%rd35` for
/// the scalar/min-max kernels, `%rd93` for MULTI):
///
/// ```text
/// \tmembar.cta;
/// \tst.shared.u32 [{set_addr_reg}], 2;
/// ```
///
/// This is emitted AFTER the CLAIM key store and BEFORE the op-specific value
/// accumulate. The `membar.cta` orders the key store ahead of the `set:=2`
/// publish so a racing prober that observes `set==2` is guaranteed to read the
/// published key (the 3-state publish protocol's release half).
pub(crate) fn emit_claim_publish(ptx: &mut String, set_addr_reg: &str) -> BoltResult<()> {
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [{set_addr_reg}], 2;").map_err(write_err)?;
    Ok(())
}

/// Register / type tokens that vary between the i32-key and i64-key SUM
/// kernels when emitting the 3-state publish/probe protocol
/// ([`emit_publish_probe_protocol`]).
///
/// Every field is the *full* PTX register token (including the `%r` / `%rd`
/// prefix) or a bare type token, so the helper never has to know a width —
/// it just substitutes the strings verbatim.
pub(crate) struct PublishRegs<'a> {
    /// Set-flag scratch register, re-used as the `nanosleep` operand. `%r36`
    /// in both the i32 and i64 SUM kernels.
    pub set_flag_reg: &'a str,
    /// Address register holding `addr_set` (the set-flag slot). `%rd35` in
    /// both SUM kernels.
    pub set_addr_reg: &'a str,
    /// Address register holding `addr_key` (the key slot). `%rd36` in both
    /// SUM kernels.
    pub key_addr_reg: &'a str,
    /// Destination register for the published key load. `%r35` (i32) /
    /// `%rd61` (i64).
    pub key_dst_reg: &'a str,
    /// The probe key register compared against the loaded key. `%r31` (i32) /
    /// `%rd60` (i64).
    pub probe_key_reg: &'a str,
}

/// Emit the 3-state publish/probe protocol for the open-addressing hash
/// table — the claim-then-publish race fix shared verbatim (modulo register
/// numbers and the key-type token) by all four SUM emitters (i32 / i64, each
/// in its non-spill and `_with_spill` form).
///
/// Emits, starting at the `PUBLISH_WAIT:` label and ending with the
/// `@%p4 bra MATCH;` collision/match branch:
///
/// ```text
/// PUBLISH_WAIT:
/// \tld.volatile.shared.u32 {set_flag_reg}, [{set_addr_reg}];
/// \tsetp.eq.u32 %p7, {set_flag_reg}, 2;
/// \t@%p7 bra PUBLISH_DONE;
/// \tmov.u32 {set_flag_reg}, 32;
/// \tnanosleep.u32 {set_flag_reg};
/// \tbra PUBLISH_WAIT;
/// PUBLISH_DONE:
/// \tmembar.cta;
/// \tld.shared.{key_ty} {key_dst_reg}, [{key_addr_reg}];
/// \tsetp.eq.{key_ty} %p4, {key_dst_reg}, {probe_key_reg};
/// \t@%p4 bra MATCH;
/// ```
///
/// The `membar.cta` after `PUBLISH_DONE:` is the ACQUIRE half of the
/// release/acquire pair with the writer's `st key; membar.cta; st [set],2`
/// (see [`emit_claim_publish`]). The reader observes `set == 2` via a
/// `ld.volatile`, but on sm_70 a *volatile* load carries no ordering for the
/// subsequent plain `ld.shared` of the key — without this fence the key load
/// may be satisfied from stale (zero-initialised) shared memory, returning
/// key 0. Since real keys include 0, a prober would then take the
/// collision-advance path and CLAIM a duplicate slot for an already-present
/// key, minting a phantom group (the Tier-2 ~1519-phantom bug at 1M-key
/// scale). The fence orders the `set==2` observation ahead of the key read.
///
/// `key_ty` is the signed-integer key-type token (`s32` for the i32-key
/// kernel, `s64` for the i64-key kernel); it appears both as the
/// `ld.shared` width and the `setp.eq` width. Every other character is
/// literal text shared by all four call sites — only the register tokens
/// and `key_ty` are substituted, so the emitted bytes are unchanged.
pub(crate) fn emit_publish_probe_protocol(
    ptx: &mut String,
    regs: &PublishRegs,
    key_ty: &str,
) -> BoltResult<()> {
    let PublishRegs {
        set_flag_reg,
        set_addr_reg,
        key_addr_reg,
        key_dst_reg,
        probe_key_reg,
    } = regs;
    writeln!(ptx, "PUBLISH_WAIT:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.volatile.shared.u32 {set_flag_reg}, [{set_addr_reg}];"
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.u32 %p7, {set_flag_reg}, 2;").map_err(write_err)?;
    writeln!(ptx, "\t@%p7 bra PUBLISH_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 {set_flag_reg}, 32;").map_err(write_err)?;
    writeln!(ptx, "\tnanosleep.u32 {set_flag_reg};").map_err(write_err)?;
    writeln!(ptx, "\tbra PUBLISH_WAIT;").map_err(write_err)?;
    writeln!(ptx, "PUBLISH_DONE:").map_err(write_err)?;
    // ACQUIRE fence: pair with the writer's release (st key; membar.cta;
    // st [set],2). A volatile load of `set==2` does not order the following
    // plain `ld.shared` of the key on sm_70, so without this the key read can
    // come from stale (zeroed) shared memory and mint a phantom group.
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.{key_ty} {key_dst_reg}, [{key_addr_reg}];").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.eq.{key_ty} %p4, {key_dst_reg}, {probe_key_reg};"
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra MATCH;").map_err(write_err)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests — exact byte-output assertions so regressions surface immediately.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_emits_expected_bytes() {
        let mut s = String::new();
        emit_ptx_header(&mut s).unwrap();
        assert_eq!(s, ".version 7.5\n.target sm_70\n.address_size 64\n\n");
    }

    #[test]
    fn thread_block_ids_emit_expected_bytes() {
        let mut s = String::new();
        emit_thread_block_ids(&mut s).unwrap();
        assert_eq!(
            s,
            "\tmov.u32 %r0, %ctaid.x;\n\
             \tmov.u32 %r1, %ntid.x;\n\
             \tmov.u32 %r2, %tid.x;\n"
        );
    }

    #[test]
    fn spill_bump_with_null_check_emits_expected_bytes_rd9() {
        let mut s = String::new();
        emit_spill_bump_with_null_check(&mut s, 9).unwrap();
        assert_eq!(
            s,
            "SPILL_BUMP:\n\
             \tsetp.eq.u64 %p5, %rd9, 0;\n\
             \t@%p5 bra LOOP_NEXT;\n\
             \tatom.global.add.u32 %r36, [%rd9], 1;\n"
        );
    }

    #[test]
    fn spill_bump_with_null_check_emits_expected_bytes_rd8() {
        let mut s = String::new();
        emit_spill_bump_with_null_check(&mut s, 8).unwrap();
        assert_eq!(
            s,
            "SPILL_BUMP:\n\
             \tsetp.eq.u64 %p5, %rd8, 0;\n\
             \t@%p5 bra LOOP_NEXT;\n\
             \tatom.global.add.u32 %r36, [%rd8], 1;\n"
        );
    }

    #[test]
    fn spill_bump_unchecked_emits_expected_bytes_rd9() {
        let mut s = String::new();
        emit_spill_bump_unchecked(&mut s, 9).unwrap();
        assert_eq!(
            s,
            "SPILL_BUMP:\n\
             \tatom.global.add.u32 %r36, [%rd9], 1;\n"
        );
    }

    #[test]
    fn spin_backoff_emits_expected_bytes() {
        let mut s = String::new();
        emit_spin_backoff(&mut s, 32).unwrap();
        assert_eq!(s, "\tmov.u32 %nstime, 32;\n\tnanosleep.u32 %nstime;\n");
    }

    #[test]
    fn partition_slice_read_emits_expected_bytes() {
        let mut s = String::new();
        emit_partition_slice_read(&mut s).unwrap();
        assert_eq!(
            s,
            "\tmul.wide.u32 %rd10, %r0, 4;\n\
             \tadd.s64 %rd11, %rd5, %rd10;\n\
             \tld.global.u32 %r10, [%rd11];\n\
             \tadd.s64 %rd12, %rd11, 4;\n\
             \tld.global.u32 %r11, [%rd12];\n"
        );
    }

    #[test]
    fn publish_probe_protocol_i32_emits_expected_bytes() {
        let mut s = String::new();
        emit_publish_probe_protocol(
            &mut s,
            &PublishRegs {
                set_flag_reg: "%r36",
                set_addr_reg: "%rd35",
                key_addr_reg: "%rd36",
                key_dst_reg: "%r35",
                probe_key_reg: "%r31",
            },
            "s32",
        )
        .unwrap();
        assert_eq!(
            s,
            "PUBLISH_WAIT:\n\
             \tld.volatile.shared.u32 %r36, [%rd35];\n\
             \tsetp.eq.u32 %p7, %r36, 2;\n\
             \t@%p7 bra PUBLISH_DONE;\n\
             \tmov.u32 %r36, 32;\n\
             \tnanosleep.u32 %r36;\n\
             \tbra PUBLISH_WAIT;\n\
             PUBLISH_DONE:\n\
             \tmembar.cta;\n\
             \tld.shared.s32 %r35, [%rd36];\n\
             \tsetp.eq.s32 %p4, %r35, %r31;\n\
             \t@%p4 bra MATCH;\n"
        );
    }

    #[test]
    fn publish_probe_protocol_i64_emits_expected_bytes() {
        let mut s = String::new();
        emit_publish_probe_protocol(
            &mut s,
            &PublishRegs {
                set_flag_reg: "%r36",
                set_addr_reg: "%rd35",
                key_addr_reg: "%rd36",
                key_dst_reg: "%rd61",
                probe_key_reg: "%rd60",
            },
            "s64",
        )
        .unwrap();
        assert_eq!(
            s,
            "PUBLISH_WAIT:\n\
             \tld.volatile.shared.u32 %r36, [%rd35];\n\
             \tsetp.eq.u32 %p7, %r36, 2;\n\
             \t@%p7 bra PUBLISH_DONE;\n\
             \tmov.u32 %r36, 32;\n\
             \tnanosleep.u32 %r36;\n\
             \tbra PUBLISH_WAIT;\n\
             PUBLISH_DONE:\n\
             \tmembar.cta;\n\
             \tld.shared.s64 %rd61, [%rd36];\n\
             \tsetp.eq.s64 %p4, %rd61, %rd60;\n\
             \t@%p4 bra MATCH;\n"
        );
    }

    #[test]
    fn loop_head_emits_expected_bytes() {
        let mut s = String::new();
        emit_loop_head(&mut s).unwrap();
        assert_eq!(
            s,
            "\tadd.u32 %r30, %r10, %r2;\n\
             LOOP_TOP:\n\
             \tsetp.ge.u32 %p1, %r30, %r11;\n\
             \t@%p1 bra LOOP_DONE;\n"
        );
    }

    #[test]
    fn probe_bound_check_emits_expected_bytes_loop_next() {
        let mut s = String::new();
        emit_probe_bound_check(&mut s, 1024, "LOOP_NEXT").unwrap();
        assert_eq!(
            s,
            "\tmov.u32 %r33, 0;\n\
             PROBE_TOP:\n\
             \tadd.u32 %r33, %r33, 1;\n\
             \tsetp.gt.u32 %p2, %r33, 1024;\n\
             \t@%p2 bra LOOP_NEXT;\n"
        );
    }

    #[test]
    fn probe_bound_check_emits_expected_bytes_spill_bump() {
        let mut s = String::new();
        emit_probe_bound_check(&mut s, 1024, "SPILL_BUMP").unwrap();
        assert_eq!(
            s,
            "\tmov.u32 %r33, 0;\n\
             PROBE_TOP:\n\
             \tadd.u32 %r33, %r33, 1;\n\
             \tsetp.gt.u32 %p2, %r33, 1024;\n\
             \t@%p2 bra SPILL_BUMP;\n"
        );
    }

    #[test]
    fn slot_claim_cas_emits_expected_bytes_rd35() {
        let mut s = String::new();
        emit_slot_claim_cas(&mut s, "%rd35").unwrap();
        assert_eq!(
            s,
            "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;\n\
             \tsetp.eq.s32 %p3, %r34, 0;\n\
             \t@%p3 bra CLAIM;\n"
        );
    }

    #[test]
    fn slot_claim_cas_emits_expected_bytes_rd93() {
        let mut s = String::new();
        emit_slot_claim_cas(&mut s, "%rd93").unwrap();
        assert_eq!(
            s,
            "\tatom.shared.cas.b32 %r34, [%rd93], 0, 1;\n\
             \tsetp.eq.s32 %p3, %r34, 0;\n\
             \t@%p3 bra CLAIM;\n"
        );
    }

    #[test]
    fn claim_publish_emits_expected_bytes_rd35() {
        let mut s = String::new();
        emit_claim_publish(&mut s, "%rd35").unwrap();
        assert_eq!(s, "\tmembar.cta;\n\tst.shared.u32 [%rd35], 2;\n");
    }

    #[test]
    fn claim_publish_emits_expected_bytes_rd93() {
        let mut s = String::new();
        emit_claim_publish(&mut s, "%rd93").unwrap();
        assert_eq!(s, "\tmembar.cta;\n\tst.shared.u32 [%rd93], 2;\n");
    }

    #[test]
    fn loop_next_done_emits_expected_bytes() {
        let mut s = String::new();
        emit_loop_next_done(&mut s).unwrap();
        assert_eq!(
            s,
            "LOOP_NEXT:\n\
             \tadd.u32 %r30, %r30, %r1;\n\
             \tbra LOOP_TOP;\n\
             LOOP_DONE:\n\
             \tbar.sync 0;\n\
             \n"
        );
    }
}
