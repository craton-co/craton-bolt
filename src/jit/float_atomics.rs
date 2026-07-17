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
//! This kernel implements the **DuckDB total order** used by the scalar /
//! window MIN/MAX path (`aggregate.rs::float_total_cmp`): **NaN sorts as the
//! largest value** (NaN > +inf), and all NaN bit-patterns are treated as
//! equal. Concretely:
//!
//! * **MAX** surfaces NaN whenever one is present in a group (NaN beats every
//!   finite value and ±inf).
//! * **MIN** skips NaN unless the group is all-NaN (any real value beats NaN
//!   because NaN is the maximum; if every value is NaN the slot keeps NaN).
//!
//! IEEE `setp.lt`/`setp.gt` return **false** for any NaN operand, so a naive
//! ordered compare would silently *ignore* NaN — disagreeing with the scalar
//! path. We therefore compute the "should the candidate replace the slot?"
//! predicate explicitly, testing each operand for NaN with
//! `testp.notanumber.fXX` and folding NaN to the top of the order:
//!
//! ```text
//! cand_nan = testp.notanumber(candidate)
//! slot_nan = testp.notanumber(slot)
//! // MAX: replace when candidate sorts strictly ABOVE slot:
//! //   (cand_nan & !slot_nan)            NaN candidate beats a finite slot
//! //   | (!cand_nan & !slot_nan & cand > slot)   ordered case, neither NaN
//! // MIN: replace when candidate sorts strictly BELOW slot:
//! //   (!cand_nan & slot_nan)            finite candidate beats a NaN slot
//! //   | (!cand_nan & !slot_nan & cand < slot)   ordered case, neither NaN
//! ```
//!
//! In both ops, a NaN-vs-NaN pair never replaces (they compare equal), and the
//! ordered branch is gated on *both* operands being non-NaN so the IEEE
//! "false on NaN" quirk can never leak a spurious decision. The accumulator is
//! still initialised to ±inf by `agg_kernels::ReduceOp::identity_ptx`; with
//! the rules above a single NaN in a MAX group overwrites that +(-)inf seed and
//! propagates to the result, exactly matching the scalar reduction.
//!
//! ## ABI
//!
//! The emitted kernel has the same six-parameter signature as
//! [`hash_kernels::compile_groupby_agg_kernel`], so the host launcher can
//! dispatch through a single code path:
//!
//! ```text
//! .visible .entry bolt_groupby_agg(
//!     .param .u64 group_col_ptr,   // i64 group keys, length n_rows
//!     .param .u64 keys_table_ptr,  // i64, length k, fully populated
//!     .param .u64 input_col_ptr,   // T (Float32 or Float64), length n_rows
//!     .param .u64 acc_table_ptr,   // T, length k, init'd to identity(op)
//!     .param .u32 n_rows,
//!     .param .u32 k                // power-of-two table size
//! )
//! ```

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
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
pub const FLOAT_ATOMIC_AGG_ENTRY: &str = "bolt_groupby_agg";

/// Per-iteration `nanosleep.u32` operand for the slot-probe and CAS-retry
/// loops. PTX `nanosleep.u32` (sm_70+) yields SM cycles so peer warps
/// contending the same accumulator slot can make progress instead of all
/// warps burning instruction-issue slots on hot CAS retries.
///
/// TODO(perf): exponential back-off (shift left by 1 per iteration, capped
/// at 256). The exponential variant requires a register that survives the
/// loop body across the back-edge, complicating the PTX. The fixed 32 ns
/// constant captures the bulk of the occupancy win at a fraction of the
/// codegen complexity.
const SPIN_BACKOFF_NS: u32 = 32;

/// Maximum number of linear-probe steps before the kernel gives up on a row
/// and records an overflow instead of spinning forever.
///
/// Expressed as a multiple of the table size `k`: we allow up to `k *
/// MAX_PROBE_FACTOR` probe steps. Mirrors the `MAX_PROBE_FACTOR` convention in
/// `hash_kernels.rs` so the two kernels bail under the same load conditions.
/// Because the keys kernel fully populates the table before this kernel runs,
/// any walk longer than the whole table (factor >= 1) means the row's key is
/// genuinely absent — a corrupt/under-populated keys table — so a small factor
/// is sufficient and keeps the bound cheap to evaluate.
const MAX_PROBE_FACTOR: u32 = 2;

/// Generate a PTX kernel for `GROUP BY MIN(float)` / `MAX(float)`.
///
/// Performs the same hash + linear probe against `keys_table_ptr` as the
/// integer agg kernel, then runs a `atom.global.cas.bXX` retry loop to update
/// the accumulator slot with `MIN`/`MAX` of `(slot_value, candidate)`.
///
/// # Errors
///
/// Returns `BoltError::Other` for any `(op, dtype)` combination outside
/// `(Min | Max, Float32 | Float64)`. Sum/Count and integer dtypes are handled
/// by `hash_kernels::compile_groupby_agg_kernel`; routing the wrong case here
/// is a programmer error and we surface it loudly instead of silently
/// producing the wrong code.
pub fn compile_groupby_float_atomic_kernel(op: ReduceOp, dtype: DataType) -> BoltResult<String> {
    // Validate inputs up front so the rest of the function can assume them.
    // `ordered_cmp` is the IEEE ordered comparison used only when BOTH operands
    // are known non-NaN (we test for NaN separately to honour the NaN-as-largest
    // total order — see the module-level "NaN behaviour" note). The float suffix
    // is filled in below from `dtype`.
    let is_min = match op {
        ReduceOp::Min => true,
        ReduceOp::Max => false,
        ReduceOp::Sum | ReduceOp::Count => false, // unused; rejected below
    };
    match (op, dtype) {
        (ReduceOp::Min, DataType::Float32)
        | (ReduceOp::Max, DataType::Float32)
        | (ReduceOp::Min, DataType::Float64)
        | (ReduceOp::Max, DataType::Float64) => {}
        (ReduceOp::Sum, _) | (ReduceOp::Count, _) => {
            return Err(BoltError::Other(format!(
                "float_atomics: only MIN/MAX are supported here (got {:?}); \
                 use hash_kernels::compile_groupby_agg_kernel for SUM/COUNT",
                op
            )));
        }
        (_, DataType::Bool)
        | (_, DataType::Int32)
        | (_, DataType::Int64)
        | (_, DataType::Utf8)
        | (_, DataType::Decimal128(_, _))
        | (_, DataType::Date32)
        | (_, DataType::Timestamp(_, _)) => {
            return Err(BoltError::Other(format!(
                "float_atomics: dtype {:?} is not a floating-point type; \
                 use hash_kernels::compile_groupby_agg_kernel for integer MIN/MAX",
                dtype
            )));
        }
    };

    // Per-dtype PTX type info. `bits_ty` is the integer width used for the
    // CAS, `float_ty` is the matching float type for the comparison, `elem_bytes`
    // is the stride for both the input column and accumulator table. `pos_zero`
    // is the PTX hex-float literal for `+0.0` of this type, used to canonicalise
    // `-0.0 -> +0.0` so the total order is deterministic (see the `-0.0`
    // canonicalisation note below).
    let (bits_ty, float_ty, elem_bytes, atom_cas, bits_reg, float_reg, pos_zero) = match dtype {
        DataType::Float32 => (
            "b32",
            "f32",
            4usize,
            "atom.global.cas.b32",
            "vr",
            "vf",
            "0f00000000",
        ),
        DataType::Float64 => (
            "b64",
            "f64",
            8usize,
            "atom.global.cas.b64",
            "vrl",
            "vfd",
            "0d0000000000000000",
        ),
        // Unreachable thanks to the validation above, but keep the match total.
        _ => {
            return Err(BoltError::Other(format!(
                "float_atomics: unexpected dtype {:?}",
                dtype
            )));
        }
    };

    // IEEE *ordered* comparison used only on the both-operands-non-NaN branch.
    // For MIN we ask `candidate < slot`; for MAX, `candidate > slot`.
    let ordered_cmp = if is_min {
        format!("setp.lt.{}", float_ty)
    } else {
        format!("setp.gt.{}", float_ty)
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
    writeln!(ptx, "\t.reg .pred  %p<9>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    // Operand register for the per-iteration `nanosleep.u32` back-off in
    // the PROBE_LOOP (slot walk) and the CAS_LOOP (contention retry).
    // PTX `nanosleep.u32` (sm_70+) suspends the warp for ~NS nanoseconds,
    // yielding SM cycles to the warp scheduler so peer warps can make
    // progress on the slot we're contending. Portable form is register
    // operand; the immediate form is rejected by some toolchains.
    writeln!(ptx, "\t.reg .u32   %nstime;").map_err(write_err)?;
    // Bit-pattern registers used for the CAS itself. For f32 these are
    // `.b32 %vrN`; for f64 they are `.b64 %vrlN`. Distinct namespaces avoid
    // collisions with the `%r` / `%rl` registers above.
    writeln!(ptx, "\t.reg .{ty}   %{rc}<8>;", ty = bits_ty, rc = bits_reg).map_err(write_err)?;
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
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
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
    writeln!(ptx, "\tmul.wide.u32 %rd1, %r3, 8;").map_err(write_err)?;
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

    // Probe bound: allow at most `k * MAX_PROBE_FACTOR` steps before declaring
    // the row's key absent and dropping its contribution. `%r9` is the live
    // probe budget, decremented each iteration and tested against zero in the
    // loop. Mirrors the MAX_PROBE_FACTOR bail convention in `hash_kernels.rs`
    // so both kernels give up under the same conditions instead of spinning
    // forever on a corrupt / under-populated keys table.
    //
    // LIMITATION: probe exhaustion (or hitting an EMPTY sentinel) silently
    // drops the row's contribution by branching to DONE — the documented,
    // pre-existing behaviour. There is no overflow counter, so this kernel
    // requires no host-side over-allocation beyond the `k`-element accumulator
    // table described in the module-level ABI note.
    writeln!(ptx, "\tmul.lo.u32 %r9, %r5, {f};", f = MAX_PROBE_FACTOR).map_err(write_err)?;

    // Probe loop. Non-mutating: keys kernel already populated the table; we
    // just walk slots until we find the one whose key matches ours, bounded by
    // the probe budget in %r9.
    writeln!(ptx, "PROBE_LOOP:").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd4, %r8, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd5, %rd3, %rd4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rl5, [%rd5];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s64 %p1, %rl5, %rl0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra FOUND;").map_err(write_err)?;
    // Defensive: if we hit an EMPTY sentinel during the probe the keys kernel
    // didn't populate this row's slot — shouldn't happen in practice. Drop the
    // row's contribution by bailing to DONE. This is a silent drop (a
    // documented limitation, unchanged from before this branch); there is no
    // overflow counter and therefore no host-side over-allocation to keep the
    // write in bounds.
    writeln!(ptx, "\tsetp.eq.s64 %p2, %rl5, %rl4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra DONE;").map_err(write_err)?;
    // Exhausted the probe budget without finding our key: also drop the row's
    // contribution (silent, same as the sentinel-miss case above).
    writeln!(ptx, "\tsetp.eq.u32 %p3, %r9, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra DONE;").map_err(write_err)?;
    writeln!(ptx, "\tsub.u32 %r9, %r9, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    // Occupancy-friendly back-off on the probe-advance path. Reached only
    // when the probed slot held the wrong key (collision) — yielding SM
    // cycles here frees the warp scheduler to run peer warps that may be
    // populating slots ahead of us. The FOUND and bail paths
    // skip this via early branches above.
    writeln!(ptx, "\tmov.u32 %nstime, {ns};", ns = SPIN_BACKOFF_NS).map_err(write_err)?;
    writeln!(ptx, "\tnanosleep.u32 %nstime;").map_err(write_err)?;
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
        "\tmul.wide.u32 %rd7, %r3, {bytes};",
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

    // Canonicalise the candidate's signed zero: -0.0 -> +0.0. Under IEEE,
    // `setp.lt`/`setp.gt` treat -0.0 and +0.0 as EQUAL (both `-0.0 < +0.0` and
    // `+0.0 < -0.0` are false), and the bits-equal short-circuit below then
    // skips the CAS — so a group containing both signs would keep whichever
    // landed first, a nondeterministic result that disagrees with the module's
    // float_total_cmp total order. We fold -0.0 to +0.0 so the stored bits are
    // deterministic. The test `setp.eq.{fty} cand, +0.0` is true for EXACTLY
    // +0.0 and -0.0 and false for NaN, so the `selp` rewrites only signed zero
    // and leaves every other value — including all NaN bit patterns — untouched,
    // preserving the NaN handling documented above.
    writeln!(
        ptx,
        "\tmov.{fty} %{fr}3, {z};",
        fty = float_ty,
        fr = float_reg,
        z = pos_zero
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.eq.{fty} %p8, %{fr}0, %{fr}3;",
        fty = float_ty,
        fr = float_reg
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tselp.{fty} %{fr}0, %{fr}3, %{fr}0, %p8;",
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
    // Canonicalise the slot's signed zero the same way as the candidate above,
    // so a -0.0 already resident in the accumulator is compared/stored as +0.0
    // and the result is independent of arrival order. NaN-safe: `setp.eq.{fty}`
    // is true only for ±0.0 and false for NaN, so NaN slot bits are preserved.
    // %{fr}3 holds +0.0 (re-materialised here so the canonicalisation does not
    // depend on register liveness across the CAS back-edge).
    writeln!(
        ptx,
        "\tmov.{fty} %{fr}3, {z};",
        fty = float_ty,
        fr = float_reg,
        z = pos_zero
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.eq.{fty} %p8, %{fr}1, %{fr}3;",
        fty = float_ty,
        fr = float_reg
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tselp.{fty} %{fr}1, %{fr}3, %{fr}1, %p8;",
        fty = float_ty,
        fr = float_reg
    )
    .map_err(write_err)?;
    // Compute %p3 = "candidate sorts strictly past the slot under the DuckDB
    // total order (NaN as largest)". A bare `setp.lt/gt` is FALSE for any NaN
    // operand, which would silently ignore NaN; we instead test each operand
    // for NaN explicitly and fold NaN to the top of the order. Predicate map:
    //   %p3  = cand_nan  = testp.notanumber(candidate, %{fr}0)
    //   %p4  = slot_nan  = testp.notanumber(old,       %{fr}1)
    //   %p5  = !cand_nan ;  %p6 = !slot_nan   (materialised — PTX `and.pred` /
    //         `or.pred` do not accept an inline `!` source negation)
    //   %p7  = ordered   = candidate <lt|gt> old   (valid only when neither NaN)
    // Final replace decision (rebuilt into %p3):
    //   MAX: (cand_nan & !slot_nan) | (!cand_nan & !slot_nan & ordered)
    //   MIN: (!cand_nan & slot_nan) | (!cand_nan & !slot_nan & ordered)
    writeln!(
        ptx,
        "\ttestp.notanumber.{fty} %p3, %{fr}0;",
        fty = float_ty,
        fr = float_reg
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\ttestp.notanumber.{fty} %p4, %{fr}1;",
        fty = float_ty,
        fr = float_reg
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tnot.pred %p5, %p3;").map_err(write_err)?; // !cand_nan
    writeln!(ptx, "\tnot.pred %p6, %p4;").map_err(write_err)?; // !slot_nan
    writeln!(
        ptx,
        "\t{cmp} %p7, %{fr}0, %{fr}1;",
        cmp = ordered_cmp,
        fr = float_reg
    )
    .map_err(write_err)?;
    // ordered term (common to MIN/MAX): ordered & !cand_nan & !slot_nan → %p7
    writeln!(ptx, "\tand.pred %p7, %p7, %p5;").map_err(write_err)?;
    writeln!(ptx, "\tand.pred %p7, %p7, %p6;").map_err(write_err)?;
    if is_min {
        // NaN-slot term: finite candidate beats a NaN slot = !cand_nan & slot_nan
        writeln!(ptx, "\tand.pred %p4, %p4, %p5;").map_err(write_err)?;
        // replace = ordered_term | nan_slot_term → %p3
        writeln!(ptx, "\tor.pred  %p3, %p7, %p4;").map_err(write_err)?;
    } else {
        // NaN-candidate term: NaN candidate beats a finite slot = cand_nan & !slot_nan
        writeln!(ptx, "\tand.pred %p3, %p3, %p6;").map_err(write_err)?;
        // replace = nan_cand_term | ordered_term → %p3
        writeln!(ptx, "\tor.pred  %p3, %p3, %p7;").map_err(write_err)?;
    }
    // new_f := candidate if we should replace, else keep old.
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
    // Occupancy-friendly back-off on the CAS-loss retry path. When the
    // CAS lost we know another warp updated the slot between our load
    // and our CAS; yielding SM cycles here gives that warp room to drain
    // its update instead of all warps storming the same cache line. The
    // CAS-won path skips the back-off by branching past via @%p5.
    writeln!(ptx, "\t@!%p5 mov.u32 %nstime, {ns};", ns = SPIN_BACKOFF_NS).map_err(write_err)?;
    writeln!(ptx, "\t@!%p5 nanosleep.u32 %nstime;").map_err(write_err)?;
    // If we did NOT win the race, someone updated the slot since our load —
    // retry with their value as the new baseline.
    writeln!(ptx, "\t@!%p5 bra CAS_LOOP;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Adapt a `std::fmt::Error` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("float_atomics: write failed: {}", e))
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
            ptx.contains("bolt_groupby_agg"),
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
            ptx.contains("bolt_groupby_agg"),
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

    /// Finding (a): the probe walk must be bounded so a corrupt /
    /// under-populated keys table can never spin the kernel forever. On probe
    /// exhaustion or an EMPTY-sentinel miss the kernel bails to DONE, silently
    /// dropping the row's contribution (a documented limitation unchanged from
    /// before this branch). There is NO overflow counter, so the kernel needs
    /// no host-side over-allocation past the `k`-element accumulator table.
    #[test]
    fn probe_loop_is_bounded_and_bails_without_overflow_counter() {
        for (op, dt) in [
            (ReduceOp::Min, DataType::Float32),
            (ReduceOp::Max, DataType::Float32),
            (ReduceOp::Min, DataType::Float64),
            (ReduceOp::Max, DataType::Float64),
        ] {
            let ptx = compile_groupby_float_atomic_kernel(op, dt)
                .unwrap_or_else(|e| panic!("kernel {op:?}/{dt:?} should compile: {e}"));

            // The probe budget is derived from k (param_5, loaded into %r5)
            // scaled by MAX_PROBE_FACTOR.
            assert!(
                ptx.contains(&format!("mul.lo.u32 %r9, %r5, {MAX_PROBE_FACTOR};")),
                "expected probe budget = k * MAX_PROBE_FACTOR in PTX for {op:?}/{dt:?}, got:\n{ptx}"
            );
            // Budget is tested against zero and decremented inside the loop.
            assert!(
                ptx.contains("setp.eq.u32 %p3, %r9, 0;"),
                "expected probe-budget exhaustion test in PTX for {op:?}/{dt:?}, got:\n{ptx}"
            );
            assert!(
                ptx.contains("sub.u32 %r9, %r9, 1;"),
                "expected probe-budget decrement in PTX for {op:?}/{dt:?}, got:\n{ptx}"
            );
            // Both the exhaustion and the EMPTY-sentinel-miss paths bail to DONE,
            // dropping the row's contribution.
            assert!(
                ptx.contains("@%p3 bra DONE;"),
                "probe-budget exhaustion should bail to DONE for {op:?}/{dt:?}, got:\n{ptx}"
            );
            assert!(
                ptx.contains("@%p2 bra DONE;"),
                "EMPTY-sentinel miss should bail to DONE for {op:?}/{dt:?}, got:\n{ptx}"
            );
            // There must be NO overflow counter: no OVERFLOW sink, no trailing
            // u32 atomic-add into an over-allocated accumulator slot. This is
            // what keeps the kernel from writing out of bounds when the host
            // allocates exactly `k` accumulator entries.
            assert!(
                !ptx.contains("OVERFLOW"),
                "no OVERFLOW sink should remain for {op:?}/{dt:?}, got:\n{ptx}"
            );
            assert!(
                !ptx.contains("atom.global.add.u32"),
                "no overflow-counter atomic add should remain for {op:?}/{dt:?}, got:\n{ptx}"
            );
        }
    }

    /// Finding (b): -0.0 must be canonicalised to +0.0 before the compare/CAS
    /// so MIN/MAX over a group containing both signed zeros is deterministic.
    /// The canonicalisation must be NaN-safe (only ±0.0 is rewritten).
    #[test]
    fn signed_zero_is_canonicalized() {
        // f32: +0.0 literal is 0f00000000.
        let ptx32 = compile_groupby_float_atomic_kernel(ReduceOp::Min, DataType::Float32).unwrap();
        assert!(
            ptx32.contains("mov.f32 %vf3, 0f00000000;"),
            "expected +0.0 f32 literal materialised for canonicalisation, got:\n{ptx32}"
        );
        // ±0.0-only test (false for NaN) then selp folds -0.0 -> +0.0 for both
        // the candidate (%vf0) and the resident slot value (%vf1).
        assert!(
            ptx32.contains("setp.eq.f32 %p8, %vf0, %vf3;")
                && ptx32.contains("selp.f32 %vf0, %vf3, %vf0, %p8;"),
            "expected candidate -0.0 canonicalisation for f32, got:\n{ptx32}"
        );
        assert!(
            ptx32.contains("setp.eq.f32 %p8, %vf1, %vf3;")
                && ptx32.contains("selp.f32 %vf1, %vf3, %vf1, %p8;"),
            "expected slot -0.0 canonicalisation for f32, got:\n{ptx32}"
        );

        // f64: +0.0 literal is 0d0000000000000000.
        let ptx64 = compile_groupby_float_atomic_kernel(ReduceOp::Max, DataType::Float64).unwrap();
        assert!(
            ptx64.contains("mov.f64 %vfd3, 0d0000000000000000;"),
            "expected +0.0 f64 literal materialised for canonicalisation, got:\n{ptx64}"
        );
        assert!(
            ptx64.contains("setp.eq.f64 %p8, %vfd0, %vfd3;")
                && ptx64.contains("selp.f64 %vfd0, %vfd3, %vfd0, %p8;"),
            "expected candidate -0.0 canonicalisation for f64, got:\n{ptx64}"
        );
        assert!(
            ptx64.contains("setp.eq.f64 %p8, %vfd1, %vfd3;")
                && ptx64.contains("selp.f64 %vfd1, %vfd3, %vfd1, %p8;"),
            "expected slot -0.0 canonicalisation for f64, got:\n{ptx64}"
        );

        // NaN handling must remain intact alongside the new canonicalisation.
        assert!(
            ptx32.contains("testp.notanumber.f32 %p3, %vf0;")
                && ptx32.contains("testp.notanumber.f32 %p4, %vf1;"),
            "NaN total-order tests should be preserved, got:\n{ptx32}"
        );
    }
}
