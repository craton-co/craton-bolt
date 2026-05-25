// SPDX-License-Identifier: Apache-2.0

//! Per-partition shared-memory **float MIN / MAX** kernel — Tier 2.1
//! variant for `Float32` and `Float64` value dtypes.
//!
//! ## Why this is a separate kernel from the integer MIN/MAX path
//!
//! PTX has no `atom.shared.{min,max}.f{32,64}` instruction on sm_70.
//! The integer atomic `atom.shared.{min,max}.{s32,u32,s64,u64}` does
//! what we want for fixed-width signed integers, but for floats we
//! have to roll our own via a `atom.shared.cas.b{32,64}` retry loop:
//!
//! ```text
//!   loop:
//!     old   = ld.shared.bXX   [slot]
//!     newv  = chosen(old, val)        // host-MIN or host-MAX semantics
//!     if newv == old goto done        // nothing to update
//!     swapped = atom.shared.cas.bXX  [slot], old, newv
//!     if swapped == old goto done     // we won the race
//!     goto loop                       // someone else updated; retry
//! ```
//!
//! The CAS reinterprets the bits as `b32` / `b64`. For floats we have
//! to choose the new value via `setp.{lt,gt}.fXX` on the typed loads
//! and then `selp.bXX` to pick the bit pattern. The pattern is well
//! established by `src/jit/float_atomics.rs` for the non-grouped
//! `atom.global.cas` SUM kernels; here we apply it to shared memory
//! and per-partition aggregation.
//!
//! ## Algorithm
//!
//! Mirrors `partition_reduce_kernel_minmax`. The only difference is
//! the atomic step: instead of `atom.shared.<op>.<itype>`, we emit a
//! CAS loop. Open-addressing slot map, identity-initialised, output
//! exported per slot.
//!
//! ## Scope (v0)
//!
//! - Op ∈ {Min, Max}
//! - Value dtype ∈ {Float32, Float64}
//! - Keys are still `i32` (matches the rest of the i32-key Tier-2
//!   pipeline). i64-key float MIN/MAX is the obvious next sibling but
//!   has no workload driver.
//!
//! ## NaN handling
//!
//! Following IEEE-754 conventions, comparisons with NaN return false.
//! Our `setp.lt.fXX` mirrors that: if either operand is NaN, the
//! "chosen" value defaults to `val` (the new candidate). NaN values
//! therefore propagate into the slot if encountered — same behaviour
//! the CPU reference would produce.

use std::fmt::Write;

use crate::error::{PatinaError, PatinaResult};
use crate::jit::partition_reduce_kernel_minmax::MinMaxOp;

pub const BLOCK_GROUPS: u32 = 1024;
pub const BLOCK_THREADS: u32 = 256;
pub const NUM_PARTITIONS: u32 = 4096;
const MAX_PROBES: u32 = BLOCK_GROUPS;

/// Float value dtype variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatDtype {
    Float32,
    Float64,
}

impl FloatDtype {
    fn bytes(self) -> u32 {
        match self {
            FloatDtype::Float32 => 4,
            FloatDtype::Float64 => 8,
        }
    }
    fn ptx_load(self) -> &'static str {
        match self {
            FloatDtype::Float32 => "ld.global.f32",
            FloatDtype::Float64 => "ld.global.f64",
        }
    }
    fn ptx_cas_suffix(self) -> &'static str {
        match self {
            FloatDtype::Float32 => "b32",
            FloatDtype::Float64 => "b64",
        }
    }
    fn ptx_setp_suffix(self) -> &'static str {
        match self {
            FloatDtype::Float32 => "f32",
            FloatDtype::Float64 => "f64",
        }
    }
}

pub fn kernel_entry(op: MinMaxOp, dtype: FloatDtype) -> String {
    let opn = match op {
        MinMaxOp::Min => "min",
        MinMaxOp::Max => "max",
    };
    let dt = match dtype {
        FloatDtype::Float32 => "f32",
        FloatDtype::Float64 => "f64",
    };
    format!("patina_partition_reduce_{}_{}", opn, dt)
}

pub fn compile_partition_reduce_kernel_minmax_float(
    op: MinMaxOp,
    dtype: FloatDtype,
) -> PatinaResult<String> {
    let mut ptx = String::new();
    let entry = kernel_entry(op, dtype);
    let entry = entry.as_str();
    let block_groups = BLOCK_GROUPS;
    let mask = BLOCK_GROUPS - 1;
    let block_threads = BLOCK_THREADS;
    let keys_bytes = BLOCK_GROUPS * 4;
    let val_bytes_per_slot = dtype.bytes();
    let vals_bytes = BLOCK_GROUPS * val_bytes_per_slot;
    let set_bytes = BLOCK_GROUPS * 4;
    let max_probes = MAX_PROBES;
    let cas_suffix = dtype.ptx_cas_suffix();
    let setp_dt = dtype.ptx_setp_suffix();
    let val_load = dtype.ptx_load();

    // Identity bit pattern (for the chosen op).
    // MIN identity = +infinity ; MAX identity = -infinity. Hex literals
    // for each dtype:
    let identity_lit: String = match (op, dtype) {
        (MinMaxOp::Min, FloatDtype::Float32) => "0x7F800000".to_string(), // +inf f32
        (MinMaxOp::Max, FloatDtype::Float32) => "0xFF800000".to_string(), // -inf f32
        (MinMaxOp::Min, FloatDtype::Float64) => "0x7FF0000000000000".to_string(), // +inf f64
        (MinMaxOp::Max, FloatDtype::Float64) => "0xFFF0000000000000".to_string(), // -inf f64
    };

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    let val_align = val_bytes_per_slot;
    writeln!(
        ptx,
        ".shared .align 4 .b8 block_keys_buf[{bytes}];",
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
    writeln!(ptx, "\t.reg .b64   %rd<64>;").map_err(write_err)?;
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

    // Read partition slice.
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd5, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd11, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd12];").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 1: zero shared. Vals init to ±infinity identity.
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

    // Phase 2: probe + CAS-loop atomic MIN/MAX.
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    // key
    writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s32 %r31, [%rd31];").map_err(write_err)?;
    // val (typed float)
    writeln!(ptx, "\tmul.wide.u32 %rd32, %r30, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd33, %rd4, %rd32;").map_err(write_err)?;
    let val_reg = match dtype {
        FloatDtype::Float32 => "%f0",
        FloatDtype::Float64 => "%fd0",
    };
    writeln!(ptx, "\t{val_load} {val_reg}, [%rd33];").map_err(write_err)?;
    // slot
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
    writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd34;").map_err(write_err)?; // addr_key
    writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?; // addr_val

    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;

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

    // CLAIM: publish key, then enter the CAS-loop to set the val.
    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd36], %r31;").map_err(write_err)?;
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

    writeln!(ptx, "\tmul.wide.u32 %rd44, %r41, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd45, %rd0, %rd44;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s32 %r47, [%rd45];").map_err(write_err)?;
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
    writeln!(ptx, "\tadd.s64 %rd49, %rd2, %rd44;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r49, [%rd49];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p6, %r49, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r50, 1, 0, %p6;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd50, %r42, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd51, %rd6, %rd50;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s32 [%rd51], %r47;").map_err(write_err)?;
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

    let _ = cas_suffix;
    let _ = setp_dt;

    Ok(ptx)
}

/// Emit one CAS retry loop that updates `[%rd38]` with the chosen
/// (MIN or MAX) of the existing value vs `val_reg`. `label_prefix` is
/// used to namespace the loop's labels so multiple CAS loops in the
/// same function don't clash.
fn emit_cas_loop(
    ptx: &mut String,
    op: MinMaxOp,
    dtype: FloatDtype,
    label_prefix: &str,
    val_reg: &str,
) -> PatinaResult<()> {
    let cas_suffix = dtype.ptx_cas_suffix();
    let setp_dt = dtype.ptx_setp_suffix();
    // Comparison: MIN keeps the smaller; MAX keeps the larger. We use
    // setp.lt for MIN (i.e. "is val < old?") and setp.gt for MAX.
    let cmp = match op {
        MinMaxOp::Min => "lt",
        MinMaxOp::Max => "gt",
    };

    // Bitcast registers for CAS: we operate on the shared-mem cell as
    // a `bXX` blob.
    let (old_bit_reg, new_bit_reg, val_bit_reg, old_typed_reg) = match dtype {
        FloatDtype::Float32 => ("%r36", "%r37", "%r38", "%f4"),
        FloatDtype::Float64 => ("%rd40", "%rd41", "%rd42", "%fd4"),
    };

    writeln!(
        ptx,
        "{lp}_LOAD:",
        lp = label_prefix
    )
    .map_err(write_err)?;
    // Read the current bits.
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
    // Pick: if (val OP old) → newv = val else newv = old.
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
    // If new == old we'd be a no-op CAS — skip to save the round trip.
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
    writeln!(
        ptx,
        "\t@{eq_pred} bra {lp}_DONE;",
        lp = label_prefix
    )
    .map_err(write_err)?;
    // CAS. If we win (swapped == old) the slot now holds newv. If we
    // lose, someone else updated; re-read and try again.
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
    writeln!(
        ptx,
        "\t@!{won_pred} bra {lp}_LOAD;",
        lp = label_prefix
    )
    .map_err(write_err)?;
    writeln!(ptx, "{lp}_DONE:", lp = label_prefix).map_err(write_err)?;
    Ok(())
}

fn write_err(e: std::fmt::Error) -> PatinaError {
    PatinaError::Other(format!(
        "partition_reduce_kernel_minmax_float: write failed: {}",
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
                let ptx = compile_partition_reduce_kernel_minmax_float(op, dt)
                    .unwrap_or_else(|e| panic!("{:?}/{:?}: {e}", op, dt));
                assert!(!ptx.is_empty());
            }
        }
    }

    #[test]
    fn uses_atom_shared_cas() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [FloatDtype::Float32, FloatDtype::Float64] {
                let ptx = compile_partition_reduce_kernel_minmax_float(op, dt).unwrap();
                let want = format!("atom.shared.cas.{}", dt.ptx_cas_suffix());
                assert!(
                    ptx.contains(&want),
                    "{:?}/{:?}: missing {want}",
                    op,
                    dt
                );
            }
        }
    }

    #[test]
    fn min_uses_setp_lt_max_uses_setp_gt() {
        let ptx_min =
            compile_partition_reduce_kernel_minmax_float(MinMaxOp::Min, FloatDtype::Float32)
                .unwrap();
        assert!(ptx_min.contains("setp.lt.f32"));
        let ptx_max =
            compile_partition_reduce_kernel_minmax_float(MinMaxOp::Max, FloatDtype::Float64)
                .unwrap();
        assert!(ptx_max.contains("setp.gt.f64"));
    }

    #[test]
    fn identity_initialised_to_signed_infinity() {
        let ptx_min_64 =
            compile_partition_reduce_kernel_minmax_float(MinMaxOp::Min, FloatDtype::Float64)
                .unwrap();
        // +inf f64 bit pattern.
        assert!(ptx_min_64.contains("0x7FF0000000000000"));
        let ptx_max_64 =
            compile_partition_reduce_kernel_minmax_float(MinMaxOp::Max, FloatDtype::Float64)
                .unwrap();
        // -inf f64 bit pattern.
        assert!(ptx_max_64.contains("0xFFF0000000000000"));
    }
}
