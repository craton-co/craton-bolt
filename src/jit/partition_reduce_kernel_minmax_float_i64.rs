// SPDX-License-Identifier: Apache-2.0

//! Per-partition shared-memory **float MIN / MAX** kernel — i64-key
//! variant (Tier 2.1 for two-key float MIN/MAX).
//!
//! Sibling of [`crate::jit::partition_reduce_kernel_minmax_float`] (i32
//! key). Adapts the same CAS-loop atomic pattern (PTX has no native
//! `atom.shared.{min,max}.f{32,64}` on sm_70) to the i64-packed-two-key
//! layout used by the two-key Tier-2.1 pipeline.
//!
//! ## Layout differences from the i32-key variant
//!
//! | What                  | i32-key variant        | i64-key variant         |
//! | --------------------- | ---------------------- | ----------------------- |
//! | `block_keys` slot     | 4 B                    | **8 B**                 |
//! | `block_keys_buf`      | 4 KiB (1024 × 4 B)     | **8 KiB** (1024 × 8 B)  |
//! | Key load (rows)       | `ld.global.s32`        | `ld.global.s64`         |
//! | Key compare           | `setp.eq.s32`          | `setp.eq.s64`           |
//! | Key store (shared)    | `st.shared.u32`        | `st.shared.u64`         |
//! | Output key store      | `st.global.s32`        | `st.global.s64`         |
//! | Slot mapping          | `& mask` on i32 key    | `cvt.u32.u64` + mask    |
//!
//! The CAS-loop body itself operates on the float value bits and is
//! identical to the i32-key sibling.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
use crate::jit::partition_reduce_kernel_minmax::MinMaxOp;
use crate::jit::partition_reduce_kernel_minmax_float::FloatDtype;

pub const BLOCK_GROUPS: u32 = 1024;
pub const BLOCK_THREADS: u32 = 256;
pub const NUM_PARTITIONS: u32 = 4096;
const MAX_PROBES: u32 = BLOCK_GROUPS;

/// Per-iteration `nanosleep.u32` operand for the collision-advance and
/// CAS-retry paths (sm_70+). TODO(perf): exponential back-off.
const SPIN_BACKOFF_NS: u32 = 32;

/// Entry-point name for the (op, dtype) combination. Distinct from the
/// i32-key sibling via the `_keyi64` suffix.
pub fn kernel_entry(op: MinMaxOp, dtype: FloatDtype) -> String {
    let opn = match op {
        MinMaxOp::Min => "min",
        MinMaxOp::Max => "max",
    };
    let dt = match dtype {
        FloatDtype::Float32 => "f32",
        FloatDtype::Float64 => "f64",
    };
    format!("bolt_partition_reduce_{}_{}_keyi64", opn, dt)
}

/// Entry-point name for the spill-counter variant.
pub fn kernel_entry_with_spill(op: MinMaxOp, dtype: FloatDtype) -> String {
    format!("{}_spill", kernel_entry(op, dtype))
}

fn val_bytes(dtype: FloatDtype) -> u32 {
    match dtype {
        FloatDtype::Float32 => 4,
        FloatDtype::Float64 => 8,
    }
}

fn ptx_load(dtype: FloatDtype) -> &'static str {
    match dtype {
        FloatDtype::Float32 => "ld.global.f32",
        FloatDtype::Float64 => "ld.global.f64",
    }
}

fn ptx_cas_suffix(dtype: FloatDtype) -> &'static str {
    match dtype {
        FloatDtype::Float32 => "b32",
        FloatDtype::Float64 => "b64",
    }
}

fn ptx_setp_suffix(dtype: FloatDtype) -> &'static str {
    match dtype {
        FloatDtype::Float32 => "f32",
        FloatDtype::Float64 => "f64",
    }
}

/// Identity bit pattern literal (MIN = +inf, MAX = -inf) for the (op, dtype)
/// pair. Shared verbatim by both emitters.
fn identity_lit(op: MinMaxOp, dtype: FloatDtype) -> String {
    match (op, dtype) {
        (MinMaxOp::Min, FloatDtype::Float32) => "0x7F800000".to_string(),
        (MinMaxOp::Max, FloatDtype::Float32) => "0xFF800000".to_string(),
        (MinMaxOp::Min, FloatDtype::Float64) => "0x7FF0000000000000".to_string(),
        (MinMaxOp::Max, FloatDtype::Float64) => "0xFFF0000000000000".to_string(),
    }
}

/// Emit the three `.shared .align` buffer declarations + trailing blank line.
/// `suffix` is `""` for the non-spill kernel and `"_sp"` for the spill kernel
/// (the only divergence: the buffer symbol names).
fn emit_shared_decls(
    ptx: &mut String,
    suffix: &str,
    val_align: u32,
    keys_bytes: u32,
    vals_bytes: u32,
    set_bytes: u32,
) -> BoltResult<()> {
    writeln!(
        ptx,
        ".shared .align 8 .b8 block_keys_buf{suffix}[{bytes}];",
        bytes = keys_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align {a} .b8 block_vals_buf{suffix}[{bytes}];",
        a = val_align,
        bytes = vals_bytes
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

/// Emit Phase 1 (zero shared; vals init to ±inf identity) — the
/// `mov.u32 %r20, %r2;` seed through the `ZERO_DONE:` / `bar.sync 0;` and
/// trailing blank line. Byte-identical across both emitters.
fn emit_zero_init_phase(
    ptx: &mut String,
    dtype: FloatDtype,
    vbpw: u32,
    identity_lit: &str,
    block_groups: u32,
    block_threads: u32,
) -> BoltResult<()> {
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r20, {bg};", bg = block_groups).map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    // block_keys[s] = 0  (i64, 8 B)
    writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd21], 0;").map_err(write_err)?;
    // block_vals[s] = identity
    writeln!(ptx, "\tmul.wide.u32 %rd22, %r20, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd22;").map_err(write_err)?;
    match dtype {
        FloatDtype::Float32 => {
            writeln!(ptx, "\tst.shared.u32 [%rd23], {identity_lit};").map_err(write_err)?;
        }
        FloatDtype::Float64 => {
            writeln!(ptx, "\tst.shared.u64 [%rd23], {identity_lit};").map_err(write_err)?;
        }
    }
    // block_set[s] = 0  (u32, 4 B)
    writeln!(ptx, "\tmul.wide.u32 %rd25, %r20, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd25;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd24], 0;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r20, %r20, {bt};", bt = block_threads).map_err(write_err)?;
    writeln!(ptx, "\tbra ZERO_TOP;").map_err(write_err)?;
    writeln!(ptx, "ZERO_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;
    Ok(())
}

/// Emit the Phase-2 probe prologue: `LOOP_TOP:` through the
/// `mov.u32 %r33, 0;` probe-counter seed. Returns the chosen value register
/// (`%f0` / `%fd0`) for the caller's CAS loops. Byte-identical across both
/// emitters (the `val_load` token string is the same for a given dtype).
fn emit_probe_prologue(
    ptx: &mut String,
    dtype: FloatDtype,
    vbpw: u32,
    val_load: &str,
    mask: u32,
) -> BoltResult<&'static str> {
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    // key = partition_keys[i] (i64)
    writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rd60, [%rd31];").map_err(write_err)?;
    // val (typed float)
    writeln!(ptx, "\tmul.wide.u32 %rd32, %r30, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd33, %rd4, %rd32;").map_err(write_err)?;
    let val_reg = match dtype {
        FloatDtype::Float32 => "%f0",
        FloatDtype::Float64 => "%fd0",
    };
    writeln!(ptx, "\t{val_load} {val_reg}, [%rd33];").map_err(write_err)?;
    // slot = low32(key) & mask
    writeln!(ptx, "\tcvt.u32.u64 %r31, %rd60;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r32, %r31, 0x{mask:X};", mask = mask).map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r33, 0;").map_err(write_err)?;

    writeln!(ptx, "PROBE_TOP:").map_err(write_err)?;
    Ok(val_reg)
}

/// Emit the probe-slot address computation (`addr_set` / `addr_key` /
/// `addr_val`) followed by the claim CAS + `@%p3 bra CLAIM;` branch.
/// Byte-identical across both emitters.
fn emit_probe_slot_and_claim(ptx: &mut String, vbpw: u32) -> BoltResult<()> {
    writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?; // addr_set
    writeln!(ptx, "\tmul.wide.u32 %rd39, %r32, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd39;").map_err(write_err)?; // addr_key (i64)
    writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?; // addr_val

    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;
    Ok(())
}

/// Emit the `CLAIM:` block (publish key, fence, publish set:=2, CAS-loop the
/// val) then `bra LOOP_NEXT;`, and the `MATCH:` block (CAS-loop the val).
/// `match_trailer` is appended after the MATCH CAS loop: empty in the
/// non-spill kernel (MATCH falls through into the inline `LOOP_NEXT:`), or
/// `"\tbra LOOP_NEXT;\n"` in the spill kernel (whose `LOOP_NEXT:` is emitted
/// later, after `SPILL_BUMP:`).
fn emit_claim_and_match(
    ptx: &mut String,
    op: MinMaxOp,
    dtype: FloatDtype,
    val_reg: &str,
    match_has_trailing_bra: bool,
) -> BoltResult<()> {
    // CLAIM: publish key (i64), fence, then CAS-loop the val.
    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd36], %rd60;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd35], 2;").map_err(write_err)?;
    emit_cas_loop(ptx, op, dtype, "CLAIM_CAS", val_reg)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    // MATCH: slot already holds our key. CAS-loop to update the val.
    writeln!(ptx, "MATCH:").map_err(write_err)?;
    emit_cas_loop(ptx, op, dtype, "MATCH_CAS", val_reg)?;
    if match_has_trailing_bra {
        writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;
    }
    Ok(())
}

/// Emit Phase 3 (export). `ge_pred` / `ne_pred` are the predicate registers
/// for the loop-guard `setp.ge.u32` and the presence `setp.ne.s32`: the
/// non-spill kernel uses `%p5` / `%p6`, the spill kernel `%p6` / `%p7` (their
/// only divergence — every instruction is otherwise identical).
fn emit_export_phase(
    ptx: &mut String,
    dtype: FloatDtype,
    vbpw: u32,
    block_groups: u32,
    block_threads: u32,
    ge_pred: &str,
    ne_pred: &str,
) -> BoltResult<()> {
    writeln!(ptx, "\tmul.lo.u32 %r40, %r0, {bg};", bg = block_groups).map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r41, %r2;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 {ge_pred}, %r41, {bg};", bg = block_groups).map_err(write_err)?;
    writeln!(ptx, "\t@{ge_pred} bra EXPORT_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r42, %r40, %r41;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd44, %r41, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd45, %rd0, %rd44;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s64 %rd62, [%rd45];").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd46, %r41, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd47, %rd1, %rd46;").map_err(write_err)?;
    let export_val_reg = match dtype {
        FloatDtype::Float32 => "%f8",
        FloatDtype::Float64 => "%fd8",
    };
    match dtype {
        FloatDtype::Float32 => {
            writeln!(ptx, "\tld.shared.f32 {export_val_reg}, [%rd47];").map_err(write_err)?;
        }
        FloatDtype::Float64 => {
            writeln!(ptx, "\tld.shared.f64 {export_val_reg}, [%rd47];").map_err(write_err)?;
        }
    }
    writeln!(ptx, "\tmul.wide.u32 %rd48, %r41, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd49, %rd2, %rd48;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r49, [%rd49];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 {ne_pred}, %r49, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r50, 1, 0, {ne_pred};").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd50, %r42, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd51, %rd6, %rd50;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s64 [%rd51], %rd62;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd52, %r42, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd53, %rd7, %rd52;").map_err(write_err)?;
    match dtype {
        FloatDtype::Float32 => {
            writeln!(ptx, "\tst.global.f32 [%rd53], {export_val_reg};").map_err(write_err)?;
        }
        FloatDtype::Float64 => {
            writeln!(ptx, "\tst.global.f64 [%rd53], {export_val_reg};").map_err(write_err)?;
        }
    }
    writeln!(ptx, "\tcvt.u64.u32 %rd54, %r42;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd55, %rd8, %rd54;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd55], %r50;").map_err(write_err)?;

    writeln!(ptx, "\tadd.u32 %r41, %r41, {bt};", bt = block_threads).map_err(write_err)?;
    writeln!(ptx, "\tbra EXPORT_TOP;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_DONE:").map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;
    Ok(())
}

/// The `PublishRegs` used by both emitters' collision path (identical
/// register tokens and key-type token in both kernels).
fn publish_regs() -> super::partition_reduce_kernel_spill_common::PublishRegs<'static> {
    super::partition_reduce_kernel_spill_common::PublishRegs {
        set_flag_reg: "%r36",
        set_addr_reg: "%rd35",
        key_addr_reg: "%rd36",
        key_dst_reg: "%rd61",
        probe_key_reg: "%rd60",
    }
}

pub fn compile_partition_reduce_kernel_minmax_float_i64(
    op: MinMaxOp,
    dtype: FloatDtype,
) -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = kernel_entry(op, dtype);
    let entry = entry.as_str();
    let block_groups = BLOCK_GROUPS;
    let mask = BLOCK_GROUPS - 1;
    let block_threads = BLOCK_THREADS;
    let keys_bytes = BLOCK_GROUPS * 8; // i64 slot stride
    let val_bytes_per_slot = val_bytes(dtype);
    let vals_bytes = BLOCK_GROUPS * val_bytes_per_slot;
    let set_bytes = BLOCK_GROUPS * 4;
    let max_probes = MAX_PROBES;
    let val_load = ptx_load(dtype);

    // Identity bit pattern: MIN = +inf, MAX = -inf.
    let identity_lit = identity_lit(op, dtype);

    super::partition_reduce_kernel_spill_common::emit_ptx_header(&mut ptx)?;

    let val_align = val_bytes_per_slot;
    // Keys are 8-byte aligned (i64).
    emit_shared_decls(&mut ptx, "", val_align, keys_bytes, vals_bytes, set_bytes)?;

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
    writeln!(ptx, "\t.reg .f32   %f<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<16>;").map_err(write_err)?;
    // Operand register for the per-collision / per-retry `nanosleep.u32`
    // back-off (sm_70+).
    writeln!(ptx, "\t.reg .u32   %nstime;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    super::partition_reduce_kernel_spill_common::emit_thread_block_ids(&mut ptx)?;
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_vals_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf;").map_err(write_err)?;

    for (rd, p) in (3..=8).zip(0..6) {
        writeln!(ptx, "\tld.param.u64 %rd{rd}, [{entry}_param_{p}];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd{rd}, %rd{rd};").map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    // Read partition slice [start, end).
    super::partition_reduce_kernel_spill_common::emit_partition_slice_read(&mut ptx)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 1: zero shared. Vals init to ±inf identity.
    let vbpw = val_bytes_per_slot;
    emit_zero_init_phase(
        &mut ptx,
        dtype,
        vbpw,
        &identity_lit,
        block_groups,
        block_threads,
    )?;

    // Phase 2: probe + CAS-loop atomic MIN/MAX.
    let val_reg = emit_probe_prologue(&mut ptx, dtype, vbpw, val_load, mask)?;
    writeln!(ptx, "\tadd.u32 %r33, %r33, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.gt.u32 %p2, %r33, {mp};",
        mp = max_probes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra LOOP_NEXT;").map_err(write_err)?;

    emit_probe_slot_and_claim(&mut ptx, vbpw)?;

    // Else: slot occupied. membar.cta orders the set CAS against the
    // i64 key load (different addresses) — without this PTX sm_70 lets
    // a racing thread observe set==1 with a zero key and false-match.
    // 3-state publish protocol (claim-then-write race fix; set u32 at %rd35,
    // key i64 at %rd36). VOLATILE SHARED re-read of set + nanosleep yield
    // until the claimer publishes set:=2, THEN read the i64 key.
    super::partition_reduce_kernel_spill_common::emit_publish_probe_protocol(
        &mut ptx,
        &publish_regs(),
        "s64",
    )?;
    writeln!(ptx, "\tadd.u32 %r32, %r32, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tand.b32 %r32, %r32, 0x{mask:X};",
        mask = mask
    )
    .map_err(write_err)?;
    // Occupancy-friendly back-off on the collision-advance path.
    super::partition_reduce_kernel_spill_common::emit_spin_backoff(&mut ptx, SPIN_BACKOFF_NS)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    // CLAIM + MATCH CAS-loops. Non-spill MATCH falls through into the inline
    // LOOP_NEXT below, so no trailing `bra LOOP_NEXT;`.
    emit_claim_and_match(&mut ptx, op, dtype, val_reg, false)?;

    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 3: export.
    emit_export_phase(
        &mut ptx,
        dtype,
        vbpw,
        block_groups,
        block_threads,
        "%p5",
        "%p6",
    )?;

    Ok(ptx)
}

/// Emit one CAS retry loop updating `[%rd38]` with the chosen (MIN/MAX)
/// of the existing value vs `val_reg`. Identical to the i32-key sibling
/// — it operates on the float value bits, not the key.
fn emit_cas_loop(
    ptx: &mut String,
    op: MinMaxOp,
    dtype: FloatDtype,
    label_prefix: &str,
    val_reg: &str,
) -> BoltResult<()> {
    let cas_suffix = ptx_cas_suffix(dtype);
    let setp_dt = ptx_setp_suffix(dtype);
    let cmp = match op {
        MinMaxOp::Min => "lt",
        MinMaxOp::Max => "gt",
    };

    let (old_bit_reg, new_bit_reg, val_bit_reg, old_typed_reg) = match dtype {
        FloatDtype::Float32 => ("%r36", "%r37", "%r38", "%f4"),
        FloatDtype::Float64 => ("%rd40", "%rd41", "%rd42", "%fd4"),
    };

    writeln!(ptx, "{lp}_LOAD:", lp = label_prefix).map_err(write_err)?;
    match dtype {
        FloatDtype::Float32 => {
            writeln!(ptx, "\tld.shared.b32 {old_bit_reg}, [%rd38];").map_err(write_err)?;
            writeln!(ptx, "\tmov.b32 {old_typed_reg}, {old_bit_reg};").map_err(write_err)?;
            writeln!(ptx, "\tmov.b32 {val_bit_reg}, {val_reg};").map_err(write_err)?;
        }
        FloatDtype::Float64 => {
            writeln!(ptx, "\tld.shared.b64 {old_bit_reg}, [%rd38];").map_err(write_err)?;
            writeln!(ptx, "\tmov.b64 {old_typed_reg}, {old_bit_reg};").map_err(write_err)?;
            writeln!(ptx, "\tmov.b64 {val_bit_reg}, {val_reg};").map_err(write_err)?;
        }
    }
    writeln!(
        ptx,
        "\tsetp.{cmp}.{setp_dt} %p7, {val_reg}, {old_typed_reg};"
    )
    .map_err(write_err)?;
    match dtype {
        FloatDtype::Float32 => {
            writeln!(
                ptx,
                "\tselp.b32 {new_bit_reg}, {val_bit_reg}, {old_bit_reg}, %p7;"
            )
            .map_err(write_err)?;
        }
        FloatDtype::Float64 => {
            writeln!(
                ptx,
                "\tselp.b64 {new_bit_reg}, {val_bit_reg}, {old_bit_reg}, %p7;"
            )
            .map_err(write_err)?;
        }
    }
    let eq_pred = "%p8";
    match dtype {
        FloatDtype::Float32 => {
            writeln!(
                ptx,
                "\tsetp.eq.b32 {eq_pred}, {new_bit_reg}, {old_bit_reg};"
            )
            .map_err(write_err)?;
        }
        FloatDtype::Float64 => {
            writeln!(
                ptx,
                "\tsetp.eq.b64 {eq_pred}, {new_bit_reg}, {old_bit_reg};"
            )
            .map_err(write_err)?;
        }
    }
    writeln!(ptx, "\t@{eq_pred} bra {lp}_DONE;", lp = label_prefix).map_err(write_err)?;
    let swap_reg = match dtype {
        FloatDtype::Float32 => "%r39",
        FloatDtype::Float64 => "%rd43",
    };
    writeln!(
        ptx,
        "\tatom.shared.cas.{cas_suffix} {swap_reg}, [%rd38], {old_bit_reg}, {new_bit_reg};"
    )
    .map_err(write_err)?;
    let won_pred = "%p9";
    match dtype {
        FloatDtype::Float32 => {
            writeln!(
                ptx,
                "\tsetp.eq.b32 {won_pred}, {swap_reg}, {old_bit_reg};"
            )
            .map_err(write_err)?;
        }
        FloatDtype::Float64 => {
            writeln!(
                ptx,
                "\tsetp.eq.b64 {won_pred}, {swap_reg}, {old_bit_reg};"
            )
            .map_err(write_err)?;
        }
    }
    // Occupancy-friendly back-off on the CAS-loss retry path. When CAS
    // lost, another warp updated the slot between our load and our CAS;
    // yielding SM cycles here gives that warp room to drain its update
    // instead of all warps storming the same cache line.
    writeln!(
        ptx,
        "\t@!{won_pred} mov.u32 %nstime, {ns};",
        ns = SPIN_BACKOFF_NS
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\t@!{won_pred} nanosleep.u32 %nstime;"
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@!{won_pred} bra {lp}_LOAD;", lp = label_prefix).map_err(write_err)?;
    writeln!(ptx, "{lp}_DONE:", lp = label_prefix).map_err(write_err)?;
    Ok(())
}

/// Spill-counter-aware sibling. Identical to
/// [`compile_partition_reduce_kernel_minmax_float_i64`] with one extra
/// `.param .u64 spill_counter` (uint32_t*, may be null). On MAX_PROBES
/// overflow null-checks then `atom.global.add.u32` it.
pub fn compile_partition_reduce_kernel_minmax_float_i64_with_spill(
    op: MinMaxOp,
    dtype: FloatDtype,
) -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = kernel_entry_with_spill(op, dtype);
    let entry = entry.as_str();
    let block_groups = BLOCK_GROUPS;
    let mask = BLOCK_GROUPS - 1;
    let block_threads = BLOCK_THREADS;
    let keys_bytes = BLOCK_GROUPS * 8;
    let val_bytes_per_slot = val_bytes(dtype);
    let vals_bytes = BLOCK_GROUPS * val_bytes_per_slot;
    let set_bytes = BLOCK_GROUPS * 4;
    let max_probes = MAX_PROBES;
    let vload = ptx_load(dtype);

    let identity_lit = identity_lit(op, dtype);

    super::partition_reduce_kernel_spill_common::emit_ptx_header(&mut ptx)?;

    let val_align = val_bytes_per_slot;
    emit_shared_decls(&mut ptx, "_sp", val_align, keys_bytes, vals_bytes, set_bytes)?;

    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    for p in 0..6 {
        writeln!(ptx, "\t.param .u64 {entry}_param_{p},").map_err(write_err)?;
    }
    writeln!(ptx, "\t.param .u64 {entry}_param_6").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<64>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<96>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f32   %f<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<16>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    super::partition_reduce_kernel_spill_common::emit_thread_block_ids(&mut ptx)?;
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf_sp;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_vals_buf_sp;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf_sp;").map_err(write_err)?;

    for (rd, p) in (3..=8).zip(0..6) {
        writeln!(ptx, "\tld.param.u64 %rd{rd}, [{entry}_param_{p}];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd{rd}, %rd{rd};").map_err(write_err)?;
    }
    writeln!(ptx, "\tld.param.u64 %rd9, [{entry}_param_6];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd9, %rd9;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    super::partition_reduce_kernel_spill_common::emit_partition_slice_read(&mut ptx)?;
    writeln!(ptx).map_err(write_err)?;

    let vbpw = val_bytes_per_slot;
    emit_zero_init_phase(
        &mut ptx,
        dtype,
        vbpw,
        &identity_lit,
        block_groups,
        block_threads,
    )?;

    let val_reg = emit_probe_prologue(&mut ptx, dtype, vbpw, vload, mask)?;
    writeln!(ptx, "\tadd.u32 %r33, %r33, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.gt.u32 %p2, %r33, {mp};",
        mp = max_probes
    )
    .map_err(write_err)?;
    // Spill divergence: overflow jumps to SPILL_BUMP instead of LOOP_NEXT.
    writeln!(ptx, "\t@%p2 bra SPILL_BUMP;").map_err(write_err)?;

    emit_probe_slot_and_claim(&mut ptx, vbpw)?;

    // 3-state publish protocol (claim-then-write race fix).
    super::partition_reduce_kernel_spill_common::emit_publish_probe_protocol(
        &mut ptx,
        &publish_regs(),
        "s64",
    )?;
    writeln!(ptx, "\tadd.u32 %r32, %r32, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tand.b32 %r32, %r32, 0x{mask:X};",
        mask = mask
    )
    .map_err(write_err)?;
    // Spill divergence: collision path has no back-off; jump straight back.
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    // CLAIM + MATCH CAS-loops. Spill MATCH branches to LOOP_NEXT (which is
    // emitted after SPILL_BUMP), so it needs a trailing `bra LOOP_NEXT;`.
    emit_claim_and_match(&mut ptx, op, dtype, val_reg, true)?;

    super::partition_reduce_kernel_spill_common::emit_spill_bump_with_null_check(&mut ptx, 9)?;

    super::partition_reduce_kernel_spill_common::emit_loop_next_done(&mut ptx)?;

    emit_export_phase(
        &mut ptx,
        dtype,
        vbpw,
        block_groups,
        block_threads,
        "%p6",
        "%p7",
    )?;

    Ok(ptx)
}

fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!(
        "partition_reduce_kernel_minmax_float_i64: write failed: {}",
        e
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_all_four_combos() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [FloatDtype::Float32, FloatDtype::Float64] {
                let ptx = compile_partition_reduce_kernel_minmax_float_i64(op, dt)
                    .unwrap_or_else(|e| panic!("{:?}/{:?}: {e}", op, dt));
                assert!(!ptx.is_empty());
            }
        }
    }

    #[test]
    fn has_keyi64_entry_name() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [FloatDtype::Float32, FloatDtype::Float64] {
                let entry = kernel_entry(op, dt);
                assert!(
                    entry.ends_with("_keyi64"),
                    "i64-key entry must end with `_keyi64`, got {entry}"
                );
                let ptx = compile_partition_reduce_kernel_minmax_float_i64(op, dt).unwrap();
                let needle = format!(".visible .entry {entry}(");
                assert!(ptx.contains(&needle), "PTX missing `{needle}`");
            }
        }
    }

    #[test]
    fn uses_i64_key_load_and_compare() {
        let ptx = compile_partition_reduce_kernel_minmax_float_i64(
            MinMaxOp::Min,
            FloatDtype::Float64,
        )
        .unwrap();
        assert!(
            ptx.contains("ld.global.s64"),
            "i64-key float kernel must use `ld.global.s64` for keys"
        );
        assert!(
            ptx.contains("setp.eq.s64"),
            "i64-key float kernel must use `setp.eq.s64`"
        );
        assert!(
            ptx.contains("st.global.s64"),
            "i64-key float kernel must store keys as s64"
        );
    }

    #[test]
    fn uses_atom_shared_cas() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [FloatDtype::Float32, FloatDtype::Float64] {
                let ptx = compile_partition_reduce_kernel_minmax_float_i64(op, dt).unwrap();
                let want = format!("atom.shared.cas.{}", ptx_cas_suffix(dt));
                assert!(
                    ptx.contains(&want),
                    "{:?}/{:?}: PTX missing `{want}`",
                    op,
                    dt
                );
            }
        }
    }

    // ----- _with_spill variant shape tests ---------------------------------

    #[test]
    fn with_spill_compiles_all_four_combos() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [FloatDtype::Float32, FloatDtype::Float64] {
                let ptx = compile_partition_reduce_kernel_minmax_float_i64_with_spill(op, dt)
                    .unwrap_or_else(|e| panic!("{:?}/{:?}: {e}", op, dt));
                assert!(!ptx.is_empty());
            }
        }
    }

    #[test]
    fn with_spill_has_distinct_entry_and_spill_atomic() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [FloatDtype::Float32, FloatDtype::Float64] {
                let base = kernel_entry(op, dt);
                let spill = kernel_entry_with_spill(op, dt);
                assert_ne!(base, spill);
                assert!(spill.ends_with("_spill"));
                let ptx =
                    compile_partition_reduce_kernel_minmax_float_i64_with_spill(op, dt).unwrap();
                let needle = format!(".visible .entry {spill}(");
                assert!(ptx.contains(&needle));
                assert!(ptx.contains("atom.global.add.u32"));
                assert!(ptx.contains("SPILL_BUMP:"));
                assert!(ptx.contains("setp.eq.u64"));
            }
        }
    }
}
