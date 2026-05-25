// SPDX-License-Identifier: Apache-2.0

//! Per-partition shared-memory GROUP BY **MIN / MAX** kernel (Tier 2.1).
//!
//! Parameterised over `(op, value_dtype)` where:
//!   * `op ∈ {Min, Max}`
//!   * `value_dtype ∈ {Int32, Int64}` — floats deferred (see below).
//!
//! ## Why no float MIN/MAX
//!
//! PTX `atom.shared.{min,max}.f32 / .f64` do NOT exist on sm_70. The
//! integer variants (`.s32`, `.s64`, `.u32`, `.u64`) are first-class;
//! float min/max requires a `atom.shared.cas.b32` / `.b64` retry loop
//! (load → compare → CAS until win). That's a different kernel
//! template and we don't have a benchmark driver for it. Documented
//! and left for a future workload.
//!
//! ## Algorithm
//!
//! Identical to the SUM kernel except:
//!   * Per-row slot atomic is `atom.shared.{min,max}.{s32,s64}` instead
//!     of `atom.shared.add.f64`.
//!   * Slot accumulator is initialised to the IDENTITY for the op:
//!     - MIN → `INT_MAX` (any incoming value is ≤ this)
//!     - MAX → `INT_MIN` (any incoming value is ≥ this)
//!   * `block_set[slot] == 1` distinguishes "no rows here" from "the
//!     identity value happens to be the answer".
//!
//! Output slot's value is the per-partition MIN or MAX of that key's
//! values. The host merges partitions trivially (each key hashes to
//! exactly one partition; no cross-partition reduction needed).

use std::fmt::Write;

use crate::error::{JavelinError, JavelinResult};

pub const BLOCK_GROUPS: u32 = 1024;
pub const BLOCK_THREADS: u32 = 256;
pub const NUM_PARTITIONS: u32 = 4096;
const MAX_PROBES: u32 = BLOCK_GROUPS;

/// Reduction op for this kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinMaxOp {
    Min,
    Max,
}

impl MinMaxOp {
    fn name(self) -> &'static str {
        match self {
            MinMaxOp::Min => "min",
            MinMaxOp::Max => "max",
        }
    }

    /// Initial value used to zero the shared-memory accumulator. MIN
    /// starts at the largest representable value; MAX at the smallest.
    fn identity_i32(self) -> i32 {
        match self {
            MinMaxOp::Min => i32::MAX,
            MinMaxOp::Max => i32::MIN,
        }
    }

    fn identity_i64(self) -> i64 {
        match self {
            MinMaxOp::Min => i64::MAX,
            MinMaxOp::Max => i64::MIN,
        }
    }
}

/// Supported value dtypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinMaxDtype {
    Int32,
    Int64,
}

impl MinMaxDtype {
    fn ptx_load(self) -> &'static str {
        match self {
            MinMaxDtype::Int32 => "ld.global.s32",
            MinMaxDtype::Int64 => "ld.global.s64",
        }
    }
    fn ptx_atom_suffix(self) -> &'static str {
        match self {
            MinMaxDtype::Int32 => "s32",
            MinMaxDtype::Int64 => "s64",
        }
    }
    fn bytes(self) -> u32 {
        match self {
            MinMaxDtype::Int32 => 4,
            MinMaxDtype::Int64 => 8,
        }
    }
    /// Register class for a typed temp (`b32` for i32, `b64` for i64).
    fn reg_class(self) -> &'static str {
        match self {
            MinMaxDtype::Int32 => "b32",
            MinMaxDtype::Int64 => "b64",
        }
    }
}

/// Entry-point name for the given (op, dtype) combination. The
/// dtype-specific suffix lets the PTX cache key off the full kernel
/// signature without ambiguity.
pub fn kernel_entry(op: MinMaxOp, dtype: MinMaxDtype) -> String {
    let dt = match dtype {
        MinMaxDtype::Int32 => "i32",
        MinMaxDtype::Int64 => "i64",
    };
    format!("javelin_partition_reduce_{}_{}", op.name(), dt)
}

/// Emit PTX for the MIN/MAX per-partition reduce kernel.
///
/// Signature:
/// ```text
/// .visible .entry <entry>(
///     .param .u64 partition_keys,     // const int32_t*
///     .param .u64 partition_vals,     // const {int32_t|int64_t}*
///     .param .u64 partition_offsets,  // const uint32_t* [K+1]
///     .param .u64 out_keys,           //       int32_t* [K*BG]
///     .param .u64 out_vals,           //       {int32_t|int64_t}* [K*BG]
///     .param .u64 out_set             //       uint8_t* [K*BG]
/// )
/// ```
pub fn compile_partition_reduce_kernel_minmax(
    op: MinMaxOp,
    dtype: MinMaxDtype,
) -> JavelinResult<String> {
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
    let atom_op = format!("atom.shared.{}.{}", op.name(), dtype.ptx_atom_suffix());

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(
        ptx,
        ".shared .align 4 .b8 block_keys_buf[{bytes}];",
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
    writeln!(ptx, "\t.reg .b64   %rd<64>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_vals_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf;").map_err(write_err)?;

    // Global pointers.
    //  %rd3 = partition_keys (i32*)
    //  %rd4 = partition_vals (i32* or i64*)
    //  %rd5 = partition_offsets (u32*)
    //  %rd6 = out_keys, %rd7 = out_vals, %rd8 = out_set
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

    // Phase 1: zero shared. Keys + set = 0; vals = IDENTITY.
    let identity_lit: String = match dtype {
        MinMaxDtype::Int32 => format!("0x{:X}", op.identity_i32() as u32),
        MinMaxDtype::Int64 => format!("0x{:X}", op.identity_i64() as u64),
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
    // block_keys[s] = 0
    writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd21], 0;").map_err(write_err)?;
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
    // block_set[s] = 0
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

    // Phase 2: probe + atomic {min,max}.
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    // key = partition_keys[i] (i32)
    writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s32 %r31, [%rd31];").map_err(write_err)?;

    // val = partition_vals[i] (typed)
    writeln!(ptx, "\tmul.wide.u32 %rd32, %r30, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd33, %rd4, %rd32;").map_err(write_err)?;
    let val_reg = if dtype == MinMaxDtype::Int64 { "%rd40" } else { "%r40" };
    let val_load = dtype.ptx_load();
    writeln!(ptx, "\t{val_load} {val_reg}, [%rd33];").map_err(write_err)?;

    // slot = key & mask
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

    // Addrs
    writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?; // addr_set
    writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd34;").map_err(write_err)?; // addr_key
    writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?; // addr_val

    // CAS the set flag.
    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;

    // Else: slot occupied. Compare keys.
    writeln!(ptx, "\tld.shared.s32 %r35, [%rd36];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p4, %r35, %r31;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra MATCH;").map_err(write_err)?;
    // Collision: advance.
    writeln!(ptx, "\tadd.u32 %r32, %r32, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tand.b32 %r32, %r32, 0x{mask:X};",
        mask = mask
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    // CLAIM: publish key, then atom.<op> the val.
    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd36], %r31;").map_err(write_err)?;
    let scratch_reg = if dtype == MinMaxDtype::Int64 { "%rd42" } else { "%r42" };
    writeln!(ptx, "\t{atom_op} {scratch_reg}, [%rd38], {val_reg};").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    writeln!(ptx, "MATCH:").map_err(write_err)?;
    let scratch_reg2 = if dtype == MinMaxDtype::Int64 { "%rd43" } else { "%r43" };
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

    writeln!(ptx, "\tmul.wide.u32 %rd44, %r45, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd45, %rd0, %rd44;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s32 %r47, [%rd45];").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd46, %r45, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd47, %rd1, %rd46;").map_err(write_err)?;
    let export_val_reg = if dtype == MinMaxDtype::Int64 { "%rd48" } else { "%r48" };
    match dtype {
        MinMaxDtype::Int32 => {
            writeln!(ptx, "\tld.shared.s32 {export_val_reg}, [%rd47];").map_err(write_err)?;
        }
        MinMaxDtype::Int64 => {
            writeln!(ptx, "\tld.shared.s64 {export_val_reg}, [%rd47];").map_err(write_err)?;
        }
    }
    writeln!(ptx, "\tadd.s64 %rd49, %rd2, %rd44;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r49, [%rd49];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p6, %r49, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r50, 1, 0, %p6;").map_err(write_err)?;

    // Store out_keys[gs] (i32), out_vals[gs] (typed), out_set[gs] (u8).
    writeln!(ptx, "\tmul.wide.u32 %rd50, %r46, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd51, %rd6, %rd50;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s32 [%rd51], %r47;").map_err(write_err)?;
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

    // The val_reg_class let-binding hint isn't actually used outside this
    // local-stringification context; reference it to keep the lint clean.
    let _ = dtype.reg_class();

    Ok(ptx)
}

fn write_err(e: std::fmt::Error) -> JavelinError {
    JavelinError::Other(format!(
        "partition_reduce_kernel_minmax: write failed: {}",
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
                let ptx = compile_partition_reduce_kernel_minmax(op, dt)
                    .unwrap_or_else(|e| panic!("{:?}/{:?} should compile: {e}", op, dt));
                assert!(!ptx.is_empty());
            }
        }
    }

    #[test]
    fn emits_expected_atomic_op() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [MinMaxDtype::Int32, MinMaxDtype::Int64] {
                let ptx = compile_partition_reduce_kernel_minmax(op, dt).unwrap();
                let want = format!("atom.shared.{}.{}", op.name(), dt.ptx_atom_suffix());
                assert!(
                    ptx.contains(&want),
                    "{:?}/{:?}: emitted PTX missing `{want}`",
                    op,
                    dt
                );
            }
        }
    }

    #[test]
    fn entry_names_are_distinct() {
        let names: Vec<String> = [MinMaxOp::Min, MinMaxOp::Max]
            .iter()
            .flat_map(|op| {
                [MinMaxDtype::Int32, MinMaxDtype::Int64]
                    .iter()
                    .map(|dt| kernel_entry(*op, *dt))
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "expected 4 distinct entry names, got {sorted:?}");
    }

    #[test]
    fn identity_initialised_per_op() {
        // The MIN kernel initialises block_vals to the type's max value
        // (so the first incoming val always wins). MAX uses min.
        // Verify the literal appears in the PTX.
        let ptx_min_32 =
            compile_partition_reduce_kernel_minmax(MinMaxOp::Min, MinMaxDtype::Int32).unwrap();
        let want_min_32 = format!("0x{:X}", i32::MAX as u32);
        assert!(
            ptx_min_32.contains(&want_min_32),
            "MIN i32 kernel must initialise block_vals to {want_min_32}"
        );

        let ptx_max_32 =
            compile_partition_reduce_kernel_minmax(MinMaxOp::Max, MinMaxDtype::Int32).unwrap();
        let want_max_32 = format!("0x{:X}", i32::MIN as u32);
        assert!(
            ptx_max_32.contains(&want_max_32),
            "MAX i32 kernel must initialise block_vals to {want_max_32}"
        );
    }

    #[test]
    fn correct_param_count() {
        let ptx =
            compile_partition_reduce_kernel_minmax(MinMaxOp::Min, MinMaxDtype::Int32).unwrap();
        let count = ptx.matches(".param .u64 ").count();
        assert_eq!(count, 6, "expected 6 .u64 params, got {count}");
    }
}
