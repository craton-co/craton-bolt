// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for GROUP BY `MIN` / `MAX` aggregates over floating-point
//! inputs.
//!
//! sm_70 has no `atom.global.min.fXX` / `atom.global.max.fXX` instructions —
//! only the integer variants and `atom.global.add.f{32,64}` are native. To
//! close the gap for `MIN(float)` / `MAX(float)`, this module emits a kernel
//! whose accumulator update is a CAS loop on the raw bit pattern of the slot:
//!
//! ```text
//! LOOP:
//!     ld.global.b32 old_bits, [addr]
//!     mov.b32 old_f, old_bits             // reinterpret bits as float
//!     setp.lt.f32 p_less, candidate, old_f
//!     selp.f32 new_f, candidate, old_f, p_less
//!     mov.b32 new_bits, new_f
//!     setp.eq.b32 p_same, old_bits, new_bits
//!     @p_same bra DONE                    // already optimal — nothing to do
//!     atom.global.cas.b32 actual, [addr], old_bits, new_bits
//!     setp.eq.b32 p_won, actual, old_bits
//!     @!p_won bra LOOP                    // another thread raced; retry
//! DONE:
//! ```
//!
//! The same shape is used for `Float64`, with `b64` / `f64` ops.
//!
//! ## NaN behaviour
//!
//! `setp.lt.fXX NaN, x` and `setp.lt.fXX x, NaN` are both `false`. As a
//! consequence:
//!
//! * If the SLOT holds a real value and the CANDIDATE is NaN, `p_less` is
//!   false → `new = old` → `p_same` is true → we bail out without writing.
//! * If the SLOT holds NaN (only possible if NaN was the identity, which it
//!   is not — we initialise to ±inf in `agg_kernels::ReduceOp::identity_ptx`)
//!   and the CANDIDATE is real, `p_less` is false → `new = old (NaN)` →
//!   `p_same` is true → again, no write.
//!
//! In other words: NaN inputs are silently ignored. That matches the standard
//! SQL semantic ("MIN/MAX ignore NaN") and is acceptable for v1. A future
//! revision could promote NaN to a propagating sentinel if a dialect ever
//! demands it.
//!
//! ## ABI
//!
//! The emitted kernel has the same six-parameter signature as
//! [`hash_kernels::compile_groupby_agg_kernel`], so the host launcher can
//! dispatch through a single code path:
//!
//! ```text
//! .visible .entry javelin_groupby_agg(
//!     .param .u64 group_col_ptr,   // i64 group keys, length n_rows
//!     .param .u64 keys_table_ptr,  // i64, length k, fully populated
//!     .param .u64 input_col_ptr,   // T (Float32 or Float64), length n_rows
//!     .param .u64 acc_table_ptr,   // T, length k, init'd to identity(op)
//!     .param .u32 n_rows,
//!     .param .u32 k                // power-of-two table size
//! )
//! ```

use std::fmt::Write;

use crate::error::{JavelinError, JavelinResult};
use crate::jit::agg_kernels::ReduceOp;
use crate::plan::logical_plan::DataType;

/// Splitmix-style multiplier used by the per-row hash. Must match
/// `hash_kernels::FX_MUL` so the probe lands on the slot the keys kernel
/// populated. Re-declared rather than imported to keep this module standalone
/// (no cross-module coupling beyond `ReduceOp` and `DataType`).
const FX_MUL: i64 = 0x9E3779B97F4A7C15u64 as i64;

/// PTX `i64::MIN` literal used as the "empty slot" sentinel by the keys
/// kernel. Mirrors `hash_kernels::EMPTY_KEY_LITERAL`.
const EMPTY_KEY_LITERAL: &str = "-9223372036854775808";

/// Entry-point name of the emitted kernel. Matches
/// `hash_kernels::AGG_KERNEL_ENTRY` so the host can look the symbol up under a
/// single name regardless of which compiler produced the PTX.
pub const FLOAT_ATOMIC_AGG_ENTRY: &str = "javelin_groupby_agg";

/// Generate a PTX kernel for `GROUP BY MIN(float)` / `MAX(float)`.
///
/// Performs the same hash + linear probe against `keys_table_ptr` as the
/// integer agg kernel, then runs a `atom.global.cas.bXX` retry loop to update
/// the accumulator slot with `MIN`/`MAX` of `(slot_value, candidate)`.
///
/// # Errors
///
/// Returns `JavelinError::Other` for any `(op, dtype)` combination outside
/// `(Min | Max, Float32 | Float64)`. Sum/Count and integer dtypes are handled
/// by `hash_kernels::compile_groupby_agg_kernel`; routing the wrong case here
/// is a programmer error and we surface it loudly instead of silently
/// producing the wrong code.
pub fn compile_groupby_float_atomic_kernel(
    op: ReduceOp,
    dtype: DataType,
) -> JavelinResult<String> {
    // Validate inputs up front so the rest of the function can assume them.
    let cmp_setp = match (op, dtype) {
        (ReduceOp::Min, DataType::Float32) => "setp.lt.f32",
        (ReduceOp::Max, DataType::Float32) => "setp.gt.f32",
        (ReduceOp::Min, DataType::Float64) => "setp.lt.f64",
        (ReduceOp::Max, DataType::Float64) => "setp.gt.f64",
        (ReduceOp::Sum, _) | (ReduceOp::Count, _) => {
            return Err(JavelinError::Other(format!(
                "float_atomics: only MIN/MAX are supported here (got {:?}); \
                 use hash_kernels::compile_groupby_agg_kernel for SUM/COUNT",
                op
            )));
        }
        (_, DataType::Bool)
        | (_, DataType::Int32)
        | (_, DataType::Int64)
        | (_, DataType::Utf8) => {
            return Err(JavelinError::Other(format!(
                "float_atomics: dtype {:?} is not a floating-point type; \
                 use hash_kernels::compile_groupby_agg_kernel for integer MIN/MAX",
                dtype
            )));
        }
    };

    // Per-dtype PTX type info. `bits_ty` is the integer width used for the
    // CAS, `float_ty` is the matching float type for the comparison, `elem_bytes`
    // is the stride for both the input column and accumulator table.
    let (bits_ty, float_ty, elem_bytes, atom_cas, bits_reg, float_reg) = match dtype {
        DataType::Float32 => ("b32", "f32", 4usize, "atom.global.cas.b32", "vr", "vf"),
        DataType::Float64 => ("b64", "f64", 8usize, "atom.global.cas.b64", "vrl", "vfd"),
        // Unreachable thanks to the validation above, but keep the match total.
        _ => {
            return Err(JavelinError::Other(format!(
                "float_atomics: unexpected dtype {:?}",
                dtype
            )));
        }
    };

    let mut ptx = String::new();

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {}(", FLOAT_ATOMIC_AGG_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", FLOAT_ATOMIC_AGG_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", FLOAT_ATOMIC_AGG_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_2,", FLOAT_ATOMIC_AGG_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_3,", FLOAT_ATOMIC_AGG_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_4,", FLOAT_ATOMIC_AGG_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_5", FLOAT_ATOMIC_AGG_ENTRY).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // `.reg` declarations. Generous because PTX `.reg` decls only allocate
    // names, not real hardware registers.
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    // Bit-pattern registers used for the CAS itself. For f32 these are
    // `.b32 %vrN`; for f64 they are `.b64 %vrlN`. Distinct namespaces avoid
    // collisions with the `%r` / `%rl` registers above.
    writeln!(ptx, "\t.reg .{ty}   %{rc}<8>;", ty = bits_ty, rc = bits_reg)
        .map_err(write_err)?;
    // Float-typed view of the same value for the comparison + select.
    writeln!(
        ptx,
        "\t.reg .{ty}   %{rc}<8>;",
        ty = float_ty,
        rc = float_reg
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // tid = ctaid.x * ntid.x + tid.x ; bail if tid >= n_rows.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u32 %r4, [{}_param_4];",
        FLOAT_ATOMIC_AGG_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.s32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // k and mask = k - 1.
    writeln!(
        ptx,
        "\tld.param.u32 %r5, [{}_param_5];",
        FLOAT_ATOMIC_AGG_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;

    // Load the i64-encoded key for this row.
    writeln!(
        ptx,
        "\tld.param.u64 %rd0, [{}_param_0];",
        FLOAT_ATOMIC_AGG_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.s32 %rd1, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rl0, [%rd2];").map_err(write_err)?;

    // Hash: h = (key * FX_MUL) >> 32 ; then & (k-1). Matches the keys kernel.
    writeln!(ptx, "\tmov.s64 %rl1, {};", FX_MUL).map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s64 %rl2, %rl0, %rl1;").map_err(write_err)?;
    writeln!(ptx, "\tshr.u64 %rl3, %rl2, 32;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u64 %r7, %rl3;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r7, %r6;").map_err(write_err)?;

    // Keys-table base pointer.
    writeln!(
        ptx,
        "\tld.param.u64 %rd3, [{}_param_1];",
        FLOAT_ATOMIC_AGG_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;

    // Empty-slot sentinel, kept around for the defensive check inside the
    // probe loop.
    writeln!(ptx, "\tmov.s64 %rl4, {};", EMPTY_KEY_LITERAL).map_err(write_err)?;

    // Probe loop. Non-mutating: keys kernel already populated the table; we
    // just walk slots until we find the one whose key matches ours.
    writeln!(ptx, "PROBE_LOOP:").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd4, %r8, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd5, %rd3, %rd4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rl5, [%rd5];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s64 %p1, %rl5, %rl0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra FOUND;").map_err(write_err)?;
    // Defensive: if we hit an EMPTY sentinel during the probe the keys kernel
    // didn't populate this row's slot — shouldn't happen in practice, but bail
    // to avoid spinning forever.
    writeln!(ptx, "\tsetp.eq.s64 %p2, %rl5, %rl4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra DONE;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_LOOP;").map_err(write_err)?;
    writeln!(ptx, "FOUND:").map_err(write_err)?;

    // Compute the accumulator slot address (acc_table + slot * elem_bytes).
    writeln!(
        ptx,
        "\tld.param.u64 %rd9, [{}_param_3];",
        FLOAT_ATOMIC_AGG_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd9, %rd9;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd10, %r8, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd9, %rd10;").map_err(write_err)?;

    // Load the candidate value (input_col[tid]) into the float-view register.
    writeln!(
        ptx,
        "\tld.param.u64 %rd6, [{}_param_2];",
        FLOAT_ATOMIC_AGG_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd6, %rd6;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.s32 %rd7, %r3, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd8, %rd6, %rd7;").map_err(write_err)?;
    // Candidate float lives in %{float_reg}0 for the rest of the kernel.
    writeln!(
        ptx,
        "\tld.global.{fty} %{fr}0, [%rd8];",
        fty = float_ty,
        fr = float_reg
    )
    .map_err(write_err)?;

    // === CAS retry loop. ===
    //
    //   %{bits_reg}0 = old_bits      (snapshot of accumulator)
    //   %{float_reg}1 = old_f        (same value reinterpreted as float)
    //   %{float_reg}2 = new_f        (min/max of old_f and candidate)
    //   %{bits_reg}1 = new_bits      (new_f reinterpreted back to bits)
    //   %{bits_reg}2 = actual_old    (value CAS observed at the slot)
    writeln!(ptx, "CAS_LOOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.global.{bty} %{br}0, [%rd11];",
        bty = bits_ty,
        br = bits_reg
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tmov.{bty} %{fr}1, %{br}0;",
        bty = bits_ty,
        fr = float_reg,
        br = bits_reg
    )
    .map_err(write_err)?;
    // %p3 = (candidate <op> old). For MIN, op is `<`; for MAX, op is `>`.
    // NaN-comparison semantics: setp.lt/gt with a NaN operand is always
    // false, so a NaN candidate leaves %p3 false → new_f := old_f → bail.
    writeln!(
        ptx,
        "\t{cmp} %p3, %{fr}0, %{fr}1;",
        cmp = cmp_setp,
        fr = float_reg
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tselp.{fty} %{fr}2, %{fr}0, %{fr}1, %p3;",
        fty = float_ty,
        fr = float_reg
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tmov.{bty} %{br}1, %{fr}2;",
        bty = bits_ty,
        br = bits_reg,
        fr = float_reg
    )
    .map_err(write_err)?;
    // If new_bits == old_bits the candidate did not improve the slot —
    // including the NaN case above — skip the atomic.
    writeln!(
        ptx,
        "\tsetp.eq.{bty} %p4, %{br}1, %{br}0;",
        bty = bits_ty,
        br = bits_reg
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra DONE;").map_err(write_err)?;
    // Try to swap old_bits -> new_bits at the slot. `atom.cas` returns the
    // pre-existing value.
    writeln!(
        ptx,
        "\t{atom} %{br}2, [%rd11], %{br}0, %{br}1;",
        atom = atom_cas,
        br = bits_reg
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.eq.{bty} %p5, %{br}2, %{br}0;",
        bty = bits_ty,
        br = bits_reg
    )
    .map_err(write_err)?;
    // If we did NOT win the race, someone updated the slot since our load —
    // retry with their value as the new baseline.
    writeln!(ptx, "\t@!%p5 bra CAS_LOOP;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Adapt a `std::fmt::Error` into a `JavelinError`.
fn write_err(e: std::fmt::Error) -> JavelinError {
    JavelinError::Other(format!("float_atomics: write failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_f32_contains_cas_loop() {
        let ptx = compile_groupby_float_atomic_kernel(ReduceOp::Min, DataType::Float32)
            .expect("kernel should compile");
        assert!(
            ptx.contains("atom.global.cas.b32"),
            "expected CAS.b32 in emitted PTX, got:\n{ptx}"
        );
        assert!(
            ptx.contains("setp.lt.f32"),
            "expected setp.lt.f32 (MIN comparison) in emitted PTX, got:\n{ptx}"
        );
        assert!(
            ptx.contains("javelin_groupby_agg"),
            "expected entry point name in emitted PTX, got:\n{ptx}"
        );
        assert!(
            ptx.contains("CAS_LOOP"),
            "expected CAS_LOOP label in emitted PTX, got:\n{ptx}"
        );
    }

    #[test]
    fn max_f64_contains_cas_loop() {
        let ptx = compile_groupby_float_atomic_kernel(ReduceOp::Max, DataType::Float64)
            .expect("kernel should compile");
        assert!(
            ptx.contains("atom.global.cas.b64"),
            "expected CAS.b64 in emitted PTX, got:\n{ptx}"
        );
        assert!(
            ptx.contains("setp.gt.f64"),
            "expected setp.gt.f64 (MAX comparison) in emitted PTX, got:\n{ptx}"
        );
        assert!(
            ptx.contains("javelin_groupby_agg"),
            "expected entry point name in emitted PTX, got:\n{ptx}"
        );
    }

    #[test]
    fn rejects_int_dtype() {
        let err = compile_groupby_float_atomic_kernel(ReduceOp::Min, DataType::Int32)
            .expect_err("Int32 should be rejected by float-only kernel");
        let msg = err.to_string();
        assert!(
            msg.contains("Int32") || msg.contains("floating-point"),
            "error message should mention dtype mismatch, got: {msg}"
        );
    }

    #[test]
    fn rejects_sum() {
        let err = compile_groupby_float_atomic_kernel(ReduceOp::Sum, DataType::Float64)
            .expect_err("Sum should be rejected by MIN/MAX-only kernel");
        let msg = err.to_string();
        assert!(
            msg.contains("MIN/MAX") || msg.contains("Sum"),
            "error message should mention op mismatch, got: {msg}"
        );
    }

    #[test]
    fn entry_constant_matches_emitted_name() {
        let ptx = compile_groupby_float_atomic_kernel(ReduceOp::Min, DataType::Float32).unwrap();
        let entry = format!(".visible .entry {}(", FLOAT_ATOMIC_AGG_ENTRY);
        assert!(
            ptx.contains(&entry),
            "PTX should declare entry as {entry:?}, got:\n{ptx}"
        );
    }
}
