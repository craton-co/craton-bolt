// SPDX-License-Identifier: Apache-2.0

//! Tier-1 per-block shared-memory **MIN / MAX** kernel.
//!
//! Sibling of [`crate::jit::shmem_sum_kernel`] and
//! [`crate::jit::shmem_count_kernel`]. Same shape — `n_groups <=
//! BLOCK_GROUPS = 1024` direct-mapped slot table, key value IS slot
//! index — but the slot accumulator is `Int32` or `Int64` updated via
//! `atom.shared.{min,max}.{s32,s64}` instead of `atom.shared.add.f64`.
//!
//! Float MIN/MAX is deferred (no native PTX float atomic; needs a CAS
//! loop). See `partition_reduce_kernel_minmax.rs` module doc for the
//! same reasoning at the Tier-2 scale.

use std::fmt::Write;

use crate::error::{JavelinError, JavelinResult};
use crate::jit::partition_reduce_kernel_minmax::{MinMaxDtype, MinMaxOp};

pub const BLOCK_GROUPS: u32 = 1024;
pub const BLOCK_THREADS: u32 = 256;

/// Entry name for the given (op, dtype) combination — distinct per
/// variant so the PTX cache keys correctly.
pub fn kernel_entry(op: MinMaxOp, dtype: MinMaxDtype) -> String {
    let dt = match dtype {
        MinMaxDtype::Int32 => "i32",
        MinMaxDtype::Int64 => "i64",
    };
    let opn = match op {
        MinMaxOp::Min => "min",
        MinMaxOp::Max => "max",
    };
    format!("javelin_groupby_shmem_{}_{}", opn, dt)
}

/// Emit PTX for the Tier-1 MIN/MAX shared-mem kernel.
///
/// Kernel signature:
/// ```text
/// .visible .entry <entry>(
///     .param .u64 keys,       // const int32_t* keys[n_rows]
///     .param .u64 vals,       // const {int32_t|int64_t}* vals[n_rows]
///     .param .u64 out_vals,   //       {int32_t|int64_t}* out[n_groups]
///     .param .u64 out_set,    //       uint8_t* out_set[n_groups]
///     .param .u32 n_rows,
///     .param .u32 n_groups
/// )
/// ```
///
/// Launch: `grid = ceil(n_rows / block) × stride`, `block = 256`,
/// shared-mem static.
pub fn compile_shmem_minmax_kernel(
    op: MinMaxOp,
    dtype: MinMaxDtype,
) -> JavelinResult<String> {
    let mut ptx = String::new();
    let entry = kernel_entry(op, dtype);
    let entry = entry.as_str();
    let block_groups = BLOCK_GROUPS;
    let val_bytes = match dtype {
        MinMaxDtype::Int32 => 4,
        MinMaxDtype::Int64 => 8,
    };
    let vals_buf_bytes = block_groups * val_bytes;
    let set_buf_bytes = block_groups * 4;
    let opn = match op {
        MinMaxOp::Min => "min",
        MinMaxOp::Max => "max",
    };
    let atom_suffix = match dtype {
        MinMaxDtype::Int32 => "s32",
        MinMaxDtype::Int64 => "s64",
    };
    let val_load = match dtype {
        MinMaxDtype::Int32 => "ld.global.s32",
        MinMaxDtype::Int64 => "ld.global.s64",
    };

    // Identity literal for the type/op. For MIN it's the type's MAX
    // value (so any input wins on the first atom.min); for MAX it's
    // the type's MIN value.
    let identity: String = match (op, dtype) {
        (MinMaxOp::Min, MinMaxDtype::Int32) => format!("0x{:X}", i32::MAX as u32),
        (MinMaxOp::Max, MinMaxDtype::Int32) => format!("0x{:X}", i32::MIN as u32),
        (MinMaxOp::Min, MinMaxDtype::Int64) => format!("0x{:X}", i64::MAX as u64),
        (MinMaxOp::Max, MinMaxDtype::Int64) => format!("0x{:X}", i64::MIN as u64),
    };

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    let val_align = if dtype == MinMaxDtype::Int64 { 8 } else { 4 };
    writeln!(
        ptx,
        ".shared .align {a} .b8 block_vals_buf[{bytes}];",
        a = val_align,
        bytes = vals_buf_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 4 .b8 block_set_buf[{bytes}];",
        bytes = set_buf_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    for p in 0..3 {
        writeln!(ptx, "\t.param .u64 {entry}_param_{p},").map_err(write_err)?;
    }
    writeln!(ptx, "\t.param .u64 {entry}_param_3,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_4,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_5").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<32>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, "\tmov.u64 %rd0, block_vals_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_set_buf;").map_err(write_err)?;
    // Param → global ptrs.
    //   %rd2 = keys (i32*)
    //   %rd3 = vals (typed*)
    //   %rd4 = out_vals (typed*)
    //   %rd5 = out_set (u8*)
    for (rd, p) in (2..=5).zip(0..4) {
        writeln!(ptx, "\tld.param.u64 %rd{rd}, [{entry}_param_{p}];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd{rd}, %rd{rd};").map_err(write_err)?;
    }
    writeln!(ptx, "\tld.param.u32 %r0, [{entry}_param_4];").map_err(write_err)?; // n_rows
    writeln!(ptx, "\tld.param.u32 %r1, [{entry}_param_5];").map_err(write_err)?; // n_groups

    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r3, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r4, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r5, %nctaid.x;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 1: zero block_set, init block_vals to identity.
    writeln!(ptx, "\tmov.u32 %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p0, %r10, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    let vbpw = val_bytes;
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r10, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd0, %rd10;").map_err(write_err)?;
    match dtype {
        MinMaxDtype::Int32 => {
            writeln!(ptx, "\tst.shared.u32 [%rd11], {identity};").map_err(write_err)?;
        }
        MinMaxDtype::Int64 => {
            writeln!(ptx, "\tst.shared.u64 [%rd11], {identity};").map_err(write_err)?;
        }
    }
    writeln!(ptx, "\tmul.wide.u32 %rd12, %r10, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd13, %rd1, %rd12;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd13], 0;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r10, %r10, %r3;").map_err(write_err)?;
    writeln!(ptx, "\tbra ZERO_TOP;").map_err(write_err)?;
    writeln!(ptx, "ZERO_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 2: grid-stride loop. For each row: load key, load val,
    // atomic min/max into block_vals[key], set block_set[key].
    writeln!(ptx, "\tmul.lo.u32 %r11, %r4, %r3;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r11, %r11, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.u32 %r12, %r3, %r5;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r11, %r0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    // key = keys[i]
    writeln!(ptx, "\tmul.wide.u32 %rd14, %r11, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd15, %rd2, %rd14;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s32 %r13, [%rd15];").map_err(write_err)?;
    // val = vals[i]
    writeln!(ptx, "\tmul.wide.u32 %rd16, %r11, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd17, %rd3, %rd16;").map_err(write_err)?;
    let val_reg = if dtype == MinMaxDtype::Int64 { "%rd20" } else { "%r20" };
    writeln!(ptx, "\t{val_load} {val_reg}, [%rd17];").map_err(write_err)?;

    // Out-of-range key (key >= BLOCK_GROUPS): write directly to
    // global out_vals[key] using atom.global.<op>.<dtype>. Slot map
    // can't hold them.
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p2, %r13, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra OVERFLOW;").map_err(write_err)?;

    // In-range: shared-mem path.
    writeln!(ptx, "\tmul.wide.u32 %rd18, %r13, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd19, %rd0, %rd18;").map_err(write_err)?;
    let scratch = if dtype == MinMaxDtype::Int64 { "%rd21" } else { "%r21" };
    writeln!(
        ptx,
        "\tatom.shared.{opn}.{at} {scratch}, [%rd19], {val_reg};",
        opn = opn, at = atom_suffix, scratch = scratch, val_reg = val_reg
    )
    .map_err(write_err)?;
    // Mark slot as populated (idempotent; many threads can hit the
    // same slot — they all write 1).
    writeln!(ptx, "\tmul.wide.u32 %rd22, %r13, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd22;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd23], 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    // Overflow: direct global atomic into out_vals[key].
    writeln!(ptx, "OVERFLOW:").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd24, %r13, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd25, %rd4, %rd24;").map_err(write_err)?;
    let scratch2 = if dtype == MinMaxDtype::Int64 { "%rd26" } else { "%r26" };
    writeln!(
        ptx,
        "\tatom.global.{opn}.{at} {scratch2}, [%rd25], {val_reg};",
        opn = opn, at = atom_suffix, scratch2 = scratch2, val_reg = val_reg
    )
    .map_err(write_err)?;
    // For overflow rows we also need to mark out_set[key] populated.
    writeln!(ptx, "\tmul.wide.u32 %rd27, %r13, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd28, %rd5, %rd27;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd28], 1;").map_err(write_err)?;

    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r11, %r11, %r12;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 3: merge populated slots to global. Each first-BLOCK_GROUPS
    // thread reads its shared slot and atom-merges into out_vals[slot]
    // if the set flag is 1.
    writeln!(ptx, "\tmov.u32 %r14, %r2;").map_err(write_err)?;
    writeln!(ptx, "MERGE_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p3, %r14, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra MERGE_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd29, %r14, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd30, %rd1, %rd29;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r15, [%rd30];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p4, %r15, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra MERGE_NEXT;").map_err(write_err)?;

    // Slot populated. Load shared val, merge to global.
    writeln!(ptx, "\tmul.wide.u32 %rd31, %r14, {vbpw};").map_err(write_err)?;
    writeln!(
        ptx,
        "\tadd.s64 %rd29, %rd0, %rd31;"
    )
    .map_err(write_err)?;
    let slot_val_reg = if dtype == MinMaxDtype::Int64 { "%rd22" } else { "%r22" };
    match dtype {
        MinMaxDtype::Int32 => {
            writeln!(ptx, "\tld.shared.s32 {slot_val_reg}, [%rd29];").map_err(write_err)?;
        }
        MinMaxDtype::Int64 => {
            writeln!(ptx, "\tld.shared.s64 {slot_val_reg}, [%rd29];").map_err(write_err)?;
        }
    }
    writeln!(
        ptx,
        "\tadd.s64 %rd29, %rd4, %rd31;"
    )
    .map_err(write_err)?;
    let scratch3 = if dtype == MinMaxDtype::Int64 { "%rd23" } else { "%r23" };
    writeln!(
        ptx,
        "\tatom.global.{opn}.{at} {scratch3}, [%rd29], {slot_val_reg};",
        opn = opn,
        at = atom_suffix
    )
    .map_err(write_err)?;
    // Mark out_set[slot] populated.
    writeln!(ptx, "\tadd.s64 %rd29, %rd5, %r14;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd29], 1;").map_err(write_err)?;

    writeln!(ptx, "MERGE_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r14, %r14, %r3;").map_err(write_err)?;
    writeln!(ptx, "\tbra MERGE_TOP;").map_err(write_err)?;
    writeln!(ptx, "MERGE_DONE:").map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    // Silence unused-variable lint for the dtype param (consumed via
    // `match` arms).
    let _ = atom_suffix;
    let _ = opn;

    Ok(ptx)
}

fn write_err(e: std::fmt::Error) -> JavelinError {
    JavelinError::Other(format!("shmem_minmax_kernel: write failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_all_four_combos() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [MinMaxDtype::Int32, MinMaxDtype::Int64] {
                let ptx = compile_shmem_minmax_kernel(op, dt).unwrap();
                assert!(!ptx.is_empty(), "{:?}/{:?} produced empty PTX", op, dt);
            }
        }
    }

    #[test]
    fn entries_are_distinct() {
        let mut names: Vec<String> = Vec::new();
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [MinMaxDtype::Int32, MinMaxDtype::Int64] {
                names.push(kernel_entry(op, dt));
            }
        }
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "want 4 distinct entries, got {sorted:?}");
    }

    #[test]
    fn emits_atomic_op() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [MinMaxDtype::Int32, MinMaxDtype::Int64] {
                let ptx = compile_shmem_minmax_kernel(op, dt).unwrap();
                let opn = if op == MinMaxOp::Min { "min" } else { "max" };
                let at = if dt == MinMaxDtype::Int32 { "s32" } else { "s64" };
                let want = format!("atom.shared.{opn}.{at}");
                assert!(
                    ptx.contains(&want),
                    "{:?}/{:?} missing {want}",
                    op,
                    dt
                );
            }
        }
    }

    #[test]
    fn has_overflow_path() {
        let ptx =
            compile_shmem_minmax_kernel(MinMaxOp::Min, MinMaxDtype::Int32).unwrap();
        assert!(ptx.contains("OVERFLOW:"), "missing OVERFLOW label");
        assert!(
            ptx.contains("atom.global.min.s32"),
            "missing global atomic on overflow path"
        );
    }
}
