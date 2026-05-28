// SPDX-License-Identifier: Apache-2.0

//! Per-partition shared-memory GROUP BY **MIN / MAX** kernel — i64-key
//! variant (Tier 2.1 for two-key MIN/MAX).
//!
//! Sibling of [`crate::jit::partition_reduce_kernel_minmax`] (i32 key).
//! The i32 variant handles single-Int32-key Tier-2 MIN/MAX; this file
//! handles the two-Int32-keys-packed-into-Int64 case used by the
//! two-key Tier-2.1 path.
//!
//! ## Layout differences from the i32 variant
//!
//! | What                  | i32 variant            | i64 variant (here)      |
//! | --------------------- | ---------------------- | ----------------------- |
//! | `block_keys` slot     | 4 B                    | **8 B**                 |
//! | `block_keys_buf`      | 4 KiB (1024 × 4 B)     | **8 KiB** (1024 × 8 B)  |
//! | Key load              | `ld.global.s32`        | `ld.global.s64`         |
//! | Key compare           | `setp.eq.s32`          | `setp.eq.s64`           |
//! | Key store (shared)    | `st.shared.u32`        | `st.shared.u64`         |
//! | Output key store      | `st.global.s32`        | `st.global.s64`         |
//! | Slot mapping          | `& mask` on key        | `cvt.u32.u64` low + mask|
//!
//! ## Slot mapping
//!
//! The partition kernel hashes i64 keys via Knuth's 64-bit Fibonacci
//! multiplier and takes the HIGH bits to pick the partition. Inside a
//! partition we use the LOW 32 bits of the packed key — i.e. the
//! second Int32 column — as the open-addressing slot index, exactly
//! mirroring [`crate::jit::partition_reduce_kernel_i64`] and
//! [`crate::jit::partition_reduce_kernel_count_i64`].

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
use crate::jit::partition_reduce_kernel_minmax::{MinMaxDtype, MinMaxOp};

pub const BLOCK_GROUPS: u32 = 1024;
pub const BLOCK_THREADS: u32 = 256;
pub const NUM_PARTITIONS: u32 = 4096;
const MAX_PROBES: u32 = BLOCK_GROUPS;

/// Per-iteration `nanosleep.u32` operand for the collision-advance path
/// (sm_70+). See `partition_reduce_kernel::SPIN_BACKOFF_NS` for full
/// rationale. TODO(perf): exponential back-off.
const SPIN_BACKOFF_NS: u32 = 32;

/// Entry-point name for the (op, dtype) combination. Distinct from the
/// i32-key sibling's entries via the `_keyi64` suffix so both can co-exist
/// in the same CUDA context.
pub fn kernel_entry(op: MinMaxOp, dtype: MinMaxDtype) -> String {
    let opn = match op {
        MinMaxOp::Min => "min",
        MinMaxOp::Max => "max",
    };
    let dt = match dtype {
        MinMaxDtype::Int32 => "i32",
        MinMaxDtype::Int64 => "i64",
    };
    format!("bolt_partition_reduce_{}_{}_keyi64", opn, dt)
}

fn val_load(dtype: MinMaxDtype) -> &'static str {
    match dtype {
        MinMaxDtype::Int32 => "ld.global.s32",
        MinMaxDtype::Int64 => "ld.global.s64",
    }
}

fn atom_suffix(dtype: MinMaxDtype) -> &'static str {
    match dtype {
        MinMaxDtype::Int32 => "s32",
        MinMaxDtype::Int64 => "s64",
    }
}

fn val_bytes(dtype: MinMaxDtype) -> u32 {
    match dtype {
        MinMaxDtype::Int32 => 4,
        MinMaxDtype::Int64 => 8,
    }
}

fn identity_i32(op: MinMaxOp) -> i32 {
    match op {
        MinMaxOp::Min => i32::MAX,
        MinMaxOp::Max => i32::MIN,
    }
}

fn identity_i64(op: MinMaxOp) -> i64 {
    match op {
        MinMaxOp::Min => i64::MAX,
        MinMaxOp::Max => i64::MIN,
    }
}

fn op_name(op: MinMaxOp) -> &'static str {
    match op {
        MinMaxOp::Min => "min",
        MinMaxOp::Max => "max",
    }
}

/// Emit PTX for the i64-key MIN/MAX per-partition reduce kernel.
///
/// Signature:
/// ```text
/// .visible .entry <entry>(
///     .param .u64 partition_keys,     // const int64_t*
///     .param .u64 partition_vals,     // const {int32_t|int64_t}*
///     .param .u64 partition_offsets,  // const uint32_t* [K+1]
///     .param .u64 out_keys,           //       int64_t* [K*BG]
///     .param .u64 out_vals,           //       {int32_t|int64_t}* [K*BG]
///     .param .u64 out_set             //       uint8_t* [K*BG]
/// )
/// ```
pub fn compile_partition_reduce_kernel_minmax_i64(
    op: MinMaxOp,
    dtype: MinMaxDtype,
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
    let atom_op = format!("atom.shared.{}.{}", op_name(op), atom_suffix(dtype));

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Keys are 8-byte aligned (i64). Vals follow the dtype.
    writeln!(
        ptx,
        ".shared .align 8 .b8 block_keys_buf[{bytes}];",
        bytes = keys_bytes
    )
    .map_err(write_err)?;
    let val_align = if dtype == MinMaxDtype::Int64 { 8 } else { 4 };
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
    // Operand register for the per-collision `nanosleep.u32` back-off.
    writeln!(ptx, "\t.reg .u32   %nstime;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_vals_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf;").map_err(write_err)?;

    // Global pointer setup.
    //  %rd3 = partition_keys (i64*)
    //  %rd4 = partition_vals (i32* or i64*)
    //  %rd5 = partition_offsets (u32*)
    //  %rd6 = out_keys (i64*), %rd7 = out_vals, %rd8 = out_set
    for (rd, p) in (3..=8).zip(0..6) {
        writeln!(ptx, "\tld.param.u64 %rd{rd}, [{entry}_param_{p}];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd{rd}, %rd{rd};").map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    // start, end = offsets[pid], offsets[pid+1]
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd5, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd11, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd12];").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 1: zero shared. Keys (i64) + set = 0; vals = IDENTITY.
    let identity_lit: String = match dtype {
        MinMaxDtype::Int32 => format!("0x{:X}", identity_i32(op) as u32),
        MinMaxDtype::Int64 => format!("0x{:X}", identity_i64(op) as u64),
    };
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p0, %r20, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    // block_keys[s] = 0   (i64, 8 B)
    writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd21], 0;").map_err(write_err)?;
    // block_vals[s] = identity
    let vbpw = val_bytes_per_slot;
    writeln!(ptx, "\tmul.wide.u32 %rd22, %r20, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd22;").map_err(write_err)?;
    match dtype {
        MinMaxDtype::Int32 => {
            writeln!(ptx, "\tst.shared.u32 [%rd23], {identity_lit};").map_err(write_err)?;
        }
        MinMaxDtype::Int64 => {
            writeln!(ptx, "\tst.shared.u64 [%rd23], {identity_lit};").map_err(write_err)?;
        }
    }
    // block_set[s] = 0   (u32, 4 B)
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

    // Phase 2: probe + atomic {min,max}.
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    // key = partition_keys[i] (i64)
    writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rd60, [%rd31];").map_err(write_err)?; // %rd60 = key

    // val = partition_vals[i] (typed)
    writeln!(ptx, "\tmul.wide.u32 %rd32, %r30, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd33, %rd4, %rd32;").map_err(write_err)?;
    let val_reg = if dtype == MinMaxDtype::Int64 { "%rd70" } else { "%r40" };
    let vload = val_load(dtype);
    writeln!(ptx, "\t{vload} {val_reg}, [%rd33];").map_err(write_err)?;

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

    // Addrs (set: *4, key: *8 i64, val: *vbpw).
    writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?; // addr_set
    writeln!(ptx, "\tmul.wide.u32 %rd39, %r32, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd39;").map_err(write_err)?; // addr_key (i64)
    writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?; // addr_val (typed)

    // CAS the set flag.
    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;

    // Else: slot occupied. membar.cta orders the CAS (block_set) against
    // the upcoming i64 key load (block_keys, a different address) — PTX
    // sm_70 has no inter-address ordering otherwise, and a racing thread
    // could observe set==1 but read a zero key and false-match key 0.
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s64 %rd61, [%rd36];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s64 %p4, %rd61, %rd60;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra MATCH;").map_err(write_err)?;
    // Collision: advance.
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

    // CLAIM: publish key (i64), fence, then atom.<op> the val.
    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd36], %rd60;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    let scratch_reg = if dtype == MinMaxDtype::Int64 { "%rd72" } else { "%r42" };
    writeln!(ptx, "\t{atom_op} {scratch_reg}, [%rd38], {val_reg};").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    writeln!(ptx, "MATCH:").map_err(write_err)?;
    let scratch_reg2 = if dtype == MinMaxDtype::Int64 { "%rd73" } else { "%r43" };
    writeln!(ptx, "\t{atom_op} {scratch_reg2}, [%rd38], {val_reg};").map_err(write_err)?;

    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 3: export.
    writeln!(
        ptx,
        "\tmul.lo.u32 %r44, %r0, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r45, %r2;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p5, %r45, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra EXPORT_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r46, %r44, %r45;").map_err(write_err)?;

    // Load shared slot's i64 key + typed val + u32 set.
    writeln!(ptx, "\tmul.wide.u32 %rd44, %r45, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd45, %rd0, %rd44;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s64 %rd62, [%rd45];").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd46, %r45, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd47, %rd1, %rd46;").map_err(write_err)?;
    let export_val_reg = if dtype == MinMaxDtype::Int64 { "%rd74" } else { "%r48" };
    match dtype {
        MinMaxDtype::Int32 => {
            writeln!(ptx, "\tld.shared.s32 {export_val_reg}, [%rd47];").map_err(write_err)?;
        }
        MinMaxDtype::Int64 => {
            writeln!(ptx, "\tld.shared.s64 {export_val_reg}, [%rd47];").map_err(write_err)?;
        }
    }
    writeln!(ptx, "\tmul.wide.u32 %rd48, %r45, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd49, %rd2, %rd48;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r49, [%rd49];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p6, %r49, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r50, 1, 0, %p6;").map_err(write_err)?;

    // Store: out_keys (i64), out_vals (typed), out_set (u8).
    writeln!(ptx, "\tmul.wide.u32 %rd50, %r46, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd51, %rd6, %rd50;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s64 [%rd51], %rd62;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd52, %r46, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd53, %rd7, %rd52;").map_err(write_err)?;
    match dtype {
        MinMaxDtype::Int32 => {
            writeln!(ptx, "\tst.global.s32 [%rd53], {export_val_reg};").map_err(write_err)?;
        }
        MinMaxDtype::Int64 => {
            writeln!(ptx, "\tst.global.s64 [%rd53], {export_val_reg};").map_err(write_err)?;
        }
    }
    writeln!(ptx, "\tcvt.u64.u32 %rd54, %r46;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd55, %rd8, %rd54;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd55], %r50;").map_err(write_err)?;

    writeln!(
        ptx,
        "\tadd.u32 %r45, %r45, {bt};",
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
        "partition_reduce_kernel_minmax_i64: write failed: {}",
        e
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_all_four_combos() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [MinMaxDtype::Int32, MinMaxDtype::Int64] {
                let ptx = compile_partition_reduce_kernel_minmax_i64(op, dt)
                    .unwrap_or_else(|e| panic!("{:?}/{:?} should compile: {e}", op, dt));
                assert!(!ptx.is_empty());
            }
        }
    }

    #[test]
    fn has_keyi64_entry_name() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [MinMaxDtype::Int32, MinMaxDtype::Int64] {
                let ptx = compile_partition_reduce_kernel_minmax_i64(op, dt).unwrap();
                let entry = kernel_entry(op, dt);
                assert!(
                    entry.ends_with("_keyi64"),
                    "i64-key entry must end with `_keyi64`, got {entry}"
                );
                let needle = format!(".visible .entry {entry}(");
                assert!(
                    ptx.contains(&needle),
                    "{:?}/{:?}: PTX missing `{needle}`",
                    op,
                    dt
                );
            }
        }
    }

    #[test]
    fn uses_i64_key_load() {
        let ptx =
            compile_partition_reduce_kernel_minmax_i64(MinMaxOp::Min, MinMaxDtype::Int32).unwrap();
        assert!(
            ptx.contains("ld.global.s64"),
            "i64-key kernel must use `ld.global.s64` for keys"
        );
        assert!(
            ptx.contains("setp.eq.s64"),
            "i64-key kernel must compare keys with `setp.eq.s64`"
        );
        assert!(
            ptx.contains("st.global.s64"),
            "i64-key kernel must store keys with `st.global.s64`"
        );
    }

    #[test]
    fn emits_expected_atomic_op() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [MinMaxDtype::Int32, MinMaxDtype::Int64] {
                let ptx = compile_partition_reduce_kernel_minmax_i64(op, dt).unwrap();
                let want = format!("atom.shared.{}.{}", op_name(op), atom_suffix(dt));
                assert!(
                    ptx.contains(&want),
                    "{:?}/{:?}: missing `{want}` in emitted PTX",
                    op,
                    dt
                );
            }
        }
    }
}
