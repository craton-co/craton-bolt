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
    writeln!(
        ptx,
        "\tatom.global.add.u32 %r36, [%rd{rd_spill_idx}], 1;"
    )
    .map_err(write_err)?;
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
pub(crate) fn emit_spill_bump_unchecked(
    ptx: &mut String,
    rd_spill_idx: u32,
) -> BoltResult<()> {
    writeln!(ptx, "SPILL_BUMP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tatom.global.add.u32 %r36, [%rd{rd_spill_idx}], 1;"
    )
    .map_err(write_err)?;
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
