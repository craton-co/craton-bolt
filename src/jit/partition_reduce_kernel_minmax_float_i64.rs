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
    let identity_lit: String = match (op, dtype) {
        (MinMaxOp::Min, FloatDtype::Float32) => "0x7F800000".to_string(),
        (MinMaxOp::Max, FloatDtype::Float32) => "0xFF800000".to_string(),
        (MinMaxOp::Min, FloatDtype::Float64) => "0x7FF0000000000000".to_string(),
        (MinMaxOp::Max, FloatDtype::Float64) => "0xFFF0000000000000".to_string(),
    };

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    let val_align = val_bytes_per_slot;
    // Keys are 8-byte aligned (i64).
    writeln!(
        ptx,
        ".shared .align 8 .b8 block_keys_buf[{bytes}];",
        bytes = keys_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align {a} .b8 block_vals_buf[{bytes}];",
        a = val_align,
        bytes = vals_bytes
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
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_vals_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf;").map_err(write_err)?;

    for (rd, p) in (3..=8).zip(0..6) {
        writeln!(ptx, "\tld.param.u64 %rd{rd}, [{entry}_param_{p}];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd{rd}, %rd{rd};").map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    // Read partition slice [start, end).
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd5, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd11, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd12];").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 1: zero shared. Vals init to ±inf identity.
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p0, %r20, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    // block_keys[s] = 0  (i64, 8 B)
    writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd21], 0;").map_err(write_err)?;
    // block_vals[s] = identity
    let vbpw = val_bytes_per_slot;
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

    // Phase 2: probe + CAS-loop atomic MIN/MAX.
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

    writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?; // addr_set
    writeln!(ptx, "\tmul.wide.u32 %rd39, %r32, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd39;").map_err(write_err)?; // addr_key (i64)
    writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?; // addr_val

    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;

    // Else: slot occupied. membar.cta orders the set CAS against the
    // i64 key load (different addresses) — without this PTX sm_70 lets
    // a racing thread observe set==1 with a zero key and false-match.
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    // CAS-LOSER ACQUIRE: the ld.shared.s64 below races against the
    // publishing thread's st.shared.u64 + membar.cta + atom.shared.cas.b32.
    // PTX does NOT guarantee acquire on plain ld.shared; the publishing
    // chain's membar.cta sequenced before atom.cas carries release-acquire
    // on Volta+. TODO: ld.acquire.cta when sm_60 is dropped.
    writeln!(
        ptx,
        "\t// CAS-LOSER ACQUIRE: see partition_reduce_kernel_minmax_float_i64.rs"
    )
    .map_err(write_err)?;
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

    // CLAIM: publish key (i64), fence, then CAS-loop the val.
    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd36], %rd60;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    emit_cas_loop(&mut ptx, op, dtype, "CLAIM_CAS", val_reg)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    // MATCH: slot already holds our key. CAS-loop to update the val.
    writeln!(ptx, "MATCH:").map_err(write_err)?;
    emit_cas_loop(&mut ptx, op, dtype, "MATCH_CAS", val_reg)?;

    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

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
        "\tsetp.ge.u32 %p5, %r41, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra EXPORT_DONE;").map_err(write_err)?;
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
    writeln!(ptx, "\tsetp.ne.s32 %p6, %r49, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r50, 1, 0, %p6;").map_err(write_err)?;

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
    writeln!(ptx, "\t@!{won_pred} bra {lp}_LOAD;", lp = label_prefix).map_err(write_err)?;
    writeln!(ptx, "{lp}_DONE:", lp = label_prefix).map_err(write_err)?;
    Ok(())
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
}
