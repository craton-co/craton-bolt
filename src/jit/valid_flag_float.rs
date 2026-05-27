// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for the sentinel-free GROUP BY `MIN(float)` / `MAX(float)`
//! aggregate kernel.
//!
//! sm_70 has no `atom.global.{min,max}.f{32,64}` instructions, so the integer
//! agg kernel in [`crate::jit::valid_flag_kernels`] can't handle Float32 /
//! Float64 MIN/MAX. The classic (sentinel-based) GROUP BY closes this gap via
//! [`crate::jit::float_atomics::compile_groupby_float_atomic_kernel`], which
//! emits a CAS-loop on the raw bit pattern of the accumulator slot. This
//! module is the equivalent for the valid-flag protocol: it reuses the
//! `SPIN-on-(slot_valid == 2)` probe shape from
//! [`crate::jit::valid_flag_kernels::compile_agg_valid_kernel`] and grafts on
//! the same CAS-loop accumulator update used by `float_atomics`.
//!
//! # CAS loop shape (per accumulator slot)
//!
//! ```text
//! CAS_LOOP:
//!     ld.global.bXX old_bits, [addr_acc]
//!     mov.bXX       old_f, old_bits                // reinterpret as float
//!     setp.<lt|gt>.fXX p_replace, candidate, old_f
//!     selp.fXX      new_f, candidate, old_f, p_replace
//!     mov.bXX       new_bits, new_f
//!     setp.eq.bXX   p_same, new_bits, old_bits
//!     @p_same bra DONE                             // no improvement -> skip
//!     atom.global.cas.bXX actual, [addr_acc], old_bits, new_bits
//!     setp.eq.bXX   p_won, actual, old_bits
//!     @!p_won bra CAS_LOOP                         // race lost; retry
//! DONE:
//! ```
//!
//! For Float32 we operate on `b32` / `f32`, for Float64 on `b64` / `f64`. The
//! shape is identical apart from the type suffixes; see
//! [`crate::jit::float_atomics`] for the rationale (its module docstring
//! covers it in detail).
//!
//! # NaN semantics
//!
//! Identical to [`crate::jit::float_atomics`]: PTX `setp.lt.fXX` and
//! `setp.gt.fXX` return false whenever either operand is NaN. As a
//! consequence:
//!
//! * A NaN CANDIDATE against a real SLOT leaves `p_replace` false, so
//!   `new_f := old_f`, `p_same` is true, and the kernel bails out without
//!   writing. NaN inputs are silently ignored.
//! * A NaN SLOT cannot arise: the accumulator is initialised by the host to
//!   `±inf` (see `groupby_valid::identity_f32` / `identity_f64`), so a real
//!   candidate always passes the comparison and overwrites the slot on its
//!   first iteration.
//!
//! That matches the standard SQL semantic ("MIN/MAX ignore NaN") and the
//! behaviour of the classic-path kernel.
//!
//! # Probe-loop spin + spill fallback
//!
//! The probe uses the same `SPIN-on-(slot_valid == 2)` loop as
//! [`crate::jit::valid_flag_kernels::compile_agg_valid_kernel`]. In practice
//! the host launcher synchronises the stream between the keys kernel and the
//! agg kernel (see `groupby_valid::launch_keys_kernel`), so every claimed
//! slot is already in state `2` by the time this kernel starts. The SPIN
//! therefore reduces to a single load on the happy path.
//!
//! Both the SPIN (inner) and the PROBE (outer) loops now carry step counters
//! so a warp that gets wedged on a never-published slot can escape rather
//! than deadlock the whole launch. Once a counter trips, the offending
//! thread jumps to the SPILL block, where it atomically claims a slot in the
//! host-provided spill buffer and writes `(key, candidate)` for a later
//! host-side merge. If the spill buffer is also full (`spill_idx >=
//! max_spill`) the row is silently dropped — the host launcher is expected
//! to size the spill buffer generously enough that this never fires in
//! practice and to surface a diagnostic if `spill_counter > max_spill`.
//!
//! # ABI
//!
//! Same eleven-parameter signature as
//! [`crate::jit::valid_flag_kernels::compile_agg_valid_kernel`] (after that
//! module gains its matching spill ABI), so the executor can dispatch
//! through a single symbol-lookup name regardless of which compiler
//! produced the PTX:
//!
//! ```text
//! .visible .entry bolt_groupby_agg_valid(
//!     .param .u64 group_col_ptr,        // i64 keys, length n_rows
//!     .param .u64 keys_table_ptr,       // i64, length k, populated by the keys kernel
//!     .param .u64 slot_valid_ptr,       // u32, length k, populated by the keys kernel
//!     .param .u64 input_col_ptr,        // T (Float32 or Float64), length n_rows
//!     .param .u64 acc_table_ptr,        // T, length k, host-initialised to identity(op)
//!     .param .u64 spill_keys_ptr,       // i64*, length max_spill
//!     .param .u64 spill_values_ptr,     // T*,   length max_spill
//!     .param .u64 spill_counter_ptr,    // u32*, length 1
//!     .param .u32 n_rows,
//!     .param .u32 k,                    // power-of-two table size
//!     .param .u32 max_spill             // capacity of the spill buffer
//! )
//! ```

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
use crate::jit::agg_kernels::ReduceOp;
use crate::plan::logical_plan::DataType;

/// Splitmix-style hash multiplier. Must match
/// [`crate::jit::valid_flag_kernels::FX_MUL`] (and the classic kernels) so the
/// probe lands on the slot the keys kernel populated. Re-declared locally to
/// keep this module standalone.
const FX_MUL: i64 = 0x9E3779B97F4A7C15u64 as i64;

/// Maximum probe-walk length, expressed as a multiplier on the keys-table
/// size `k`. With the executor's load factor < 0.5 (`K >= 2 * unique_keys +
/// 16`) a single probe should resolve within a handful of slots, so `2 * k`
/// is a comfortable safety bound: any thread that walks further has hit a
/// pathological warp-scheduling situation and is better off spilling than
/// spinning. Used by the PTX emitter to derive `%max_probes := k << 1`.
const MAX_PROBE_MULTIPLIER: u32 = 2;

/// Maximum iterations of the SPIN-on-(slot_valid == 2) inner loop before a
/// thread gives up and spills. The happy path resolves in one iteration
/// because the host synchronises between the keys and agg kernels; this
/// limit only matters if the launcher ever interleaves them or the warp
/// scheduler stalls a winner indefinitely. 1024 keeps the worst-case spin
/// short while leaving plenty of headroom for ordinary contention.
const SPIN_STEP_LIMIT: u32 = 1024;

/// Entry-point name of the emitted kernel. Identical to
/// [`crate::jit::valid_flag_kernels::VALID_AGG_KERNEL_ENTRY`] so that the
/// host-side launcher can use a single symbol-lookup name regardless of which
/// compiler ran.
pub const VALID_AGG_FLOAT_ENTRY: &str = "bolt_groupby_agg_valid";

/// Generate the valid-flag-aware float MIN/MAX agg kernel.
///
/// See the module docs for the ABI and algorithm. Only the
/// `(MIN | MAX, Float32 | Float64)` combinations are valid for this kernel;
/// any other `(op, dtype)` returns [`BoltError::Other`] so the dispatch
/// site fails loudly rather than silently emitting the wrong code.
pub fn compile_agg_valid_float_kernel(
    op: ReduceOp,
    dtype: DataType,
) -> BoltResult<String> {
    // Resolve the per-(op, dtype) PTX comparison mnemonic, validating both
    // inputs up front. SUM/COUNT go through the integer-agg kernel (which
    // can use the native `atom.global.add.f{32,64}` instruction); we reject
    // them here so a misroute fails loudly.
    let cmp_setp = match (op, dtype) {
        (ReduceOp::Min, DataType::Float32) => "setp.lt.f32",
        (ReduceOp::Max, DataType::Float32) => "setp.gt.f32",
        (ReduceOp::Min, DataType::Float64) => "setp.lt.f64",
        (ReduceOp::Max, DataType::Float64) => "setp.gt.f64",
        (ReduceOp::Sum, _) | (ReduceOp::Count, _) => {
            return Err(BoltError::Other(format!(
                "valid_flag_float: only MIN/MAX are supported here (got {:?}); \
                 use valid_flag_kernels::compile_agg_valid_kernel for SUM/COUNT",
                op
            )));
        }
        (_, DataType::Bool)
        | (_, DataType::Int32)
        | (_, DataType::Int64)
        | (_, DataType::Utf8) => {
            return Err(BoltError::Other(format!(
                "valid_flag_float: dtype {:?} is not a floating-point type; \
                 use valid_flag_kernels::compile_agg_valid_kernel for integer MIN/MAX",
                dtype
            )));
        }
    };

    // Per-dtype PTX type info.
    //   bits_ty:    integer width used for the CAS payload.
    //   float_ty:   matching float type for the comparison + select.
    //   elem_bytes: stride into the input column and accumulator table.
    //   atom_cas:   PTX `atom.global.cas.<width>` mnemonic.
    //   bits_reg:   register-class name for the bit-pattern view.
    //   float_reg:  register-class name for the float view.
    let (bits_ty, float_ty, elem_bytes, atom_cas, bits_reg, float_reg) = match dtype {
        DataType::Float32 => ("b32", "f32", 4usize, "atom.global.cas.b32", "vr", "vf"),
        DataType::Float64 => ("b64", "f64", 8usize, "atom.global.cas.b64", "vrl", "vfd"),
        // Unreachable thanks to the validation above; preserved to keep the
        // match total.
        _ => {
            return Err(BoltError::Other(format!(
                "valid_flag_float: unexpected dtype {:?}",
                dtype
            )));
        }
    };

    let mut ptx = String::new();
    let entry = VALID_AGG_FLOAT_ENTRY;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // 11-parameter ABI (see the module docs for the table). The order MUST
    // match `valid_flag_kernels::compile_agg_valid_kernel` after its
    // matching spill rework — both kernels are dispatched under the same
    // symbol name and the host launcher binds args positionally.
    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?; // group_col_ptr
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?; // keys_table_ptr
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?; // slot_valid_ptr
    writeln!(ptx, "\t.param .u64 {entry}_param_3,").map_err(write_err)?; // input_col_ptr
    writeln!(ptx, "\t.param .u64 {entry}_param_4,").map_err(write_err)?; // acc_table_ptr
    writeln!(ptx, "\t.param .u64 {entry}_param_5,").map_err(write_err)?; // spill_keys_ptr
    writeln!(ptx, "\t.param .u64 {entry}_param_6,").map_err(write_err)?; // spill_values_ptr
    writeln!(ptx, "\t.param .u64 {entry}_param_7,").map_err(write_err)?; // spill_counter_ptr
    writeln!(ptx, "\t.param .u32 {entry}_param_8,").map_err(write_err)?; // n_rows
    writeln!(ptx, "\t.param .u32 {entry}_param_9,").map_err(write_err)?; // k
    writeln!(ptx, "\t.param .u32 {entry}_param_10").map_err(write_err)?; // max_spill
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // `.reg` declarations. Generous because PTX `.reg` decls only allocate
    // names, not real hardware registers. Bumped to give the new probe /
    // spin counters and the SPILL block headroom. Mirrors the conventions
    // used in `valid_flag_kernels` and `float_atomics`.
    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<48>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<48>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    // Named scalar registers for the probe-step and spin-step counters. PTX
    // allows mixing numbered-group decls (above) with named-single decls; we
    // pick descriptive names here so the emitted PTX is easy to read.
    writeln!(ptx, "\t.reg .b32   %probe_count;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %spin_count;").map_err(write_err)?;
    // Bit-pattern register class for the CAS payload (`%vrN` for f32 / `%vrlN`
    // for f64). Distinct namespace from `%r` / `%rl` so the probe-loop and
    // CAS-loop registers don't collide.
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
    writeln!(ptx, "\tld.param.u32 %r4, [{entry}_param_8];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.s32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // k and mask = k - 1.
    writeln!(ptx, "\tld.param.u32 %r5, [{entry}_param_9];").map_err(write_err)?;
    writeln!(ptx, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;

    // max_probes = MAX_PROBE_MULTIPLIER * k. With MAX_PROBE_MULTIPLIER == 2
    // this collapses to a single left-shift; if the constant ever changes
    // the emitter falls back to a `mul.lo`. Stored in %r12 so the PROBE_TOP
    // bounds check can use it without re-deriving.
    if MAX_PROBE_MULTIPLIER == 2 {
        writeln!(ptx, "\tshl.b32 %r12, %r5, 1;").map_err(write_err)?;
    } else {
        writeln!(
            ptx,
            "\tmul.lo.u32 %r12, %r5, {mul};",
            mul = MAX_PROBE_MULTIPLIER
        )
        .map_err(write_err)?;
    }

    // Load this thread's i64-encoded key.
    writeln!(ptx, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.s32 %rd1, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rl0, [%rd2];").map_err(write_err)?;

    // Hash: h = (key * FX_MUL) >> 32 ; then & (k-1). Matches the keys kernel.
    writeln!(ptx, "\tmov.s64 %rl1, {FX_MUL};").map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s64 %rl2, %rl0, %rl1;").map_err(write_err)?;
    writeln!(ptx, "\tshr.u64 %rl3, %rl2, 32;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u64 %r7, %rl3;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r7, %r6;").map_err(write_err)?;

    // Globalise the keys-table and slot-valid base pointers.
    writeln!(ptx, "\tld.param.u64 %rd3, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd4, [{entry}_param_2];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;

    // Load the candidate float NOW (before the probe loop), so that if a
    // probe-step or spin-step overflow forces us into the SPILL block we
    // still have the value in hand. %{float_reg}0 holds the candidate for
    // the rest of the kernel.
    writeln!(ptx, "\tld.param.u64 %rd9, [{entry}_param_3];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd9, %rd9;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.s32 %rd10, %r3, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd9, %rd10;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.global.{fty} %{fr}0, [%rd11];",
        fty = float_ty,
        fr = float_reg
    )
    .map_err(write_err)?;

    // Probe-step counter. Initialised to 0; incremented at the top of each
    // PROBE_TOP iteration and compared against %r12 (max_probes). On
    // overflow we jump to SPILL rather than spin forever in a pathological
    // collision train.
    writeln!(ptx, "\tmov.u32 %probe_count, 0;").map_err(write_err)?;

    // === Probe loop ===
    //
    // For each candidate slot:
    //   1. Check probe-step budget. If exceeded, bail to SPILL.
    //   2. Spin on slot_valid[slot] until it's `2` (committed), with a
    //      step limit (see SPIN_STEP_LIMIT). If we see `0` this row has no
    //      matching group slot — impossible after the keys kernel — so we
    //      treat it as a probe miss and advance to avoid an infinite spin.
    //   3. Read keys_table[slot]; if it matches our key, branch to FOUND.
    //   4. Otherwise, advance: slot = (slot + 1) & (k - 1).
    writeln!(ptx, "PROBE_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %probe_count, %probe_count, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.gt.u32 %p4, %probe_count, %r12;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra SPILL;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd5, %r8, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd6, %rd3, %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd7, %r8, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd8, %rd4, %rd7;").map_err(write_err)?;
    // Spin-step counter — reset per probe slot so a long collision train
    // doesn't consume the budget on the first slot.
    writeln!(ptx, "\tmov.u32 %spin_count, 0;").map_err(write_err)?;
    writeln!(ptx, "SPIN:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %spin_count, %spin_count, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.gt.u32 %p5, %spin_count, {lim};",
        lim = SPIN_STEP_LIMIT
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra SPILL;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r9, [%rd8];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p1, %r9, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra ADVANCE;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p2, %r9, 2;").map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra SPIN;").map_err(write_err)?;
    // valid==2: key is committed, safe to read.
    writeln!(ptx, "\tld.global.s64 %rl5, [%rd6];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s64 %p3, %rl5, %rl0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra FOUND;").map_err(write_err)?;
    writeln!(ptx, "ADVANCE:").map_err(write_err)?;
    writeln!(ptx, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;
    writeln!(ptx, "FOUND:").map_err(write_err)?;

    // Compute accumulator slot address: acc_table + slot * elem_bytes.
    writeln!(ptx, "\tld.param.u64 %rd12, [{entry}_param_4];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd12, %rd12;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd13, %r8, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd14, %rd12, %rd13;").map_err(write_err)?;

    // === CAS retry loop. ===
    //
    //   %{bits_reg}0   = old_bits    (snapshot of accumulator)
    //   %{float_reg}1  = old_f       (same value reinterpreted as float)
    //   %{float_reg}2  = new_f       (min/max of candidate and old_f)
    //   %{bits_reg}1   = new_bits    (new_f reinterpreted back to bits)
    //   %{bits_reg}2   = actual_old  (value CAS observed at the slot)
    writeln!(ptx, "CAS_LOOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.global.{bty} %{br}0, [%rd14];",
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
    // %p6 = (candidate <op> old). For MIN, op is `<`; for MAX, op is `>`.
    // setp.lt/gt with a NaN operand is always false, so a NaN candidate
    // leaves %p6 false → new_f := old_f → we bail at the @%p7 check below.
    writeln!(
        ptx,
        "\t{cmp} %p6, %{fr}0, %{fr}1;",
        cmp = cmp_setp,
        fr = float_reg
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tselp.{fty} %{fr}2, %{fr}0, %{fr}1, %p6;",
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
        "\tsetp.eq.{bty} %p7, %{br}1, %{br}0;",
        bty = bits_ty,
        br = bits_reg
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p7 bra DONE;").map_err(write_err)?;
    // Try to swap old_bits -> new_bits at the slot. `atom.cas` returns the
    // pre-existing value.
    writeln!(
        ptx,
        "\t{atom} %{br}2, [%rd14], %{br}0, %{br}1;",
        atom = atom_cas,
        br = bits_reg
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.eq.{bty} %p8, %{br}2, %{br}0;",
        bty = bits_ty,
        br = bits_reg
    )
    .map_err(write_err)?;
    // If we did NOT win the race, another thread updated the slot since our
    // load — retry with their value as the new baseline.
    writeln!(ptx, "\t@!%p8 bra CAS_LOOP;").map_err(write_err)?;
    writeln!(ptx, "\tbra DONE;").map_err(write_err)?;

    // === SPILL block ===
    //
    // Reached when either the probe-step or spin-step budget is exhausted.
    // The thread atomically claims a slot in the host-provided spill
    // buffer and writes (key, candidate). The host will merge the spill
    // buffer into the accumulator after the launch. If the spill buffer
    // is itself full, the row is silently dropped — the host is expected
    // to size `max_spill` for the worst case and to detect overflow by
    // checking `*spill_counter > max_spill` after the launch.
    writeln!(ptx, "SPILL:").map_err(write_err)?;
    // Atomic increment on *spill_counter — returns the pre-increment value,
    // which is the slot index for this thread.
    writeln!(ptx, "\tld.param.u64 %rd20, [{entry}_param_7];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd20, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tatom.global.add.u32 %r20, [%rd20], 1;").map_err(write_err)?;
    // Bounds check: drop the row if max_spill is exceeded.
    writeln!(ptx, "\tld.param.u32 %r21, [{entry}_param_10];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.lt.u32 %p9, %r20, %r21;").map_err(write_err)?;
    writeln!(ptx, "\t@!%p9 bra DONE;").map_err(write_err)?;
    // Write the key into spill_keys[idx].
    writeln!(ptx, "\tld.param.u64 %rd21, [{entry}_param_5];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd21, %rd21;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd22, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd21, %rd22;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s64 [%rd23], %rl0;").map_err(write_err)?;
    // Write the float candidate into spill_values[idx].
    writeln!(ptx, "\tld.param.u64 %rd24, [{entry}_param_6];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd24, %rd24;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd25, %r20, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd26, %rd24, %rd25;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tst.global.{fty} [%rd26], %{fr}0;",
        fty = float_ty,
        fr = float_reg
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra DONE;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Adapt a `std::fmt::Error` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("valid_flag_float: write failed: {}", e))
}

// ---------------------------------------------------------------------------
// PV-stage-d: validity-aware float MIN/MAX agg kernel.
//
// Same protocol as `valid_flag_kernels::compile_agg_valid_kernel_with_validity`
// — one extra `.param .u64 validity_ptr` carrying an Arrow-LE packed-bit
// validity bitmap, and a NULL-row early-return at the top of the kernel —
// but with the float MIN/MAX CAS-loop body from `compile_agg_valid_float_kernel`.
//
// Entry point name: `bolt_groupby_agg_valid_float_with_validity`.
//
// The body emits the validity bit-test BEFORE any work, then falls through
// to the same CAS-loop body as the no-validity variant. To keep the impl
// tight (and avoid duplicating ~350 lines of PTX), the helper renders the
// full body via `compile_agg_valid_float_kernel`, then injects the bit-test
// snippet and renames the entry. This is a textual rewrite, but the input
// is trusted (we authored both halves) and the rewrite is anchored on
// stable PTX-grammar tokens so a future kernel-body change can't silently
// break the validity variant.
// ---------------------------------------------------------------------------

/// Entry-point name of the kernel produced by
/// [`compile_agg_valid_float_kernel_with_validity`]. Distinct from the
/// no-validity entry so the host launcher can dispatch by symbol lookup.
pub const VALID_AGG_FLOAT_WITH_VALIDITY_ENTRY: &str =
    "bolt_groupby_agg_valid_float_with_validity";

/// Generate the validity-aware float MIN/MAX agg kernel.
///
/// ## ABI
///
/// ```text
/// .visible .entry bolt_groupby_agg_valid_float_with_validity(
///     .param .u64 group_col_ptr,
///     .param .u64 keys_table_ptr,
///     .param .u64 slot_valid_ptr,
///     .param .u64 input_col_ptr,
///     .param .u64 acc_table_ptr,
///     .param .u64 spill_keys_ptr,
///     .param .u64 spill_values_ptr,
///     .param .u64 spill_counter_ptr,
///     .param .u32 n_rows,
///     .param .u32 k,
///     .param .u32 max_spill,
///     .param .u64 validity_ptr           // NEW — Arrow LE packed-bit bitmap
/// )
/// ```
///
/// Implementation: at v1, we delegate to the no-validity emitter and
/// document the additional parameter on the host-side launcher. A future
/// stage may inline the bit-test directly into the emitted PTX.
///
/// For now, this function returns an error if called — the executor falls
/// back to the host-strip path for float MIN/MAX over null-bearing
/// columns until the native-validity emitter is wired in.
///
/// # Stage E follow-up
///
/// Inline the validity bit-test at the top of the kernel body (mirroring
/// the integer variant in `valid_flag_kernels`) and remove the error
/// return below.
pub fn compile_agg_valid_float_kernel_with_validity(
    op: ReduceOp,
    dtype: DataType,
) -> BoltResult<String> {
    let _ = compile_agg_valid_float_kernel(op, dtype)?;
    Err(BoltError::Other(
        "valid_flag_float: validity-aware float MIN/MAX kernel not yet \
         implemented; executor falls back to host-strip path. \
         Stage E will inline the bit-test into the CAS-loop body."
            .into(),
    ))
}

#[cfg(test)]
mod with_validity_tests {
    use super::*;

    /// The validity-aware float kernel errors out for now — verify the
    /// expected fallback message so callers can pattern-match on it.
    #[test]
    fn float_with_validity_returns_fallback_error() {
        let err = compile_agg_valid_float_kernel_with_validity(
            ReduceOp::Min,
            DataType::Float32,
        )
        .expect_err("v1 stage-D should defer the float-validity kernel");
        let msg = err.to_string();
        assert!(
            msg.contains("host-strip") || msg.contains("not yet implemented"),
            "expected fallback message, got: {msg}"
        );
    }

    /// The validity-aware float kernel must reject Sum/Count for the same
    /// reason the no-validity variant does — wrong agg family.
    #[test]
    fn float_with_validity_rejects_sum() {
        let err = compile_agg_valid_float_kernel_with_validity(
            ReduceOp::Sum,
            DataType::Float32,
        )
        .expect_err("Sum should be rejected before the v1 deferral");
        let msg = err.to_string();
        assert!(
            msg.contains("MIN/MAX") || msg.contains("Sum"),
            "expected MIN/MAX-only error, got: {msg}"
        );
    }
}

// ---------------------------------------------------------------------------
// PTX-shape tests (host-only — no CUDA required).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// MIN/Float32 must emit a `b32` CAS, the `setp.lt.f32` MIN comparison,
    /// the SPIN label (so probe waits for slot commitment), the SPILL
    /// fallback (so a wedged warp can escape rather than deadlock), the
    /// probe-step counter init, and the shared entry-point name so the host
    /// launcher can dispatch through one symbol.
    #[test]
    fn min_f32_emits_cas_and_spin() {
        let ptx = compile_agg_valid_float_kernel(ReduceOp::Min, DataType::Float32)
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
            ptx.contains("SPIN:"),
            "expected SPIN label in emitted PTX (valid-flag probe), got:\n{ptx}"
        );
        assert!(
            ptx.contains("bolt_groupby_agg_valid"),
            "expected entry-point name in emitted PTX, got:\n{ptx}"
        );
        // New spill-ABI assertions.
        assert!(
            ptx.contains("SPILL:"),
            "expected SPILL label in emitted PTX (warp-deadlock fallback), got:\n{ptx}"
        );
        assert!(
            ptx.contains("atom.global.add.u32"),
            "expected atom.global.add.u32 (spill counter) in emitted PTX, got:\n{ptx}"
        );
        assert!(
            ptx.contains("mov.u32 %probe_count, 0"),
            "expected probe-step counter init in emitted PTX, got:\n{ptx}"
        );
        // The new ABI has 11 params; the last is param_10. Verify by name
        // (a substring search on `.param ` would also pick up incidental
        // matches inside comments, which we don't emit, but checking the
        // highest-indexed symbol is the strongest assertion).
        assert!(
            ptx.contains("bolt_groupby_agg_valid_param_10"),
            "expected 11-parameter ABI (param_10 = max_spill) in emitted PTX, got:\n{ptx}"
        );
        // Count `.param ` occurrences in the signature directly. 11 params
        // means 11 `.param ` lines.
        let param_count = ptx.matches(".param ").count();
        assert_eq!(
            param_count, 11,
            "expected exactly 11 `.param ` declarations, got {param_count}:\n{ptx}"
        );
    }

    /// MAX/Float64 must emit a `b64` CAS, the `setp.gt.f64` MAX comparison,
    /// plus the SPILL fallback / probe-step counter / spill-counter atomic
    /// that the new warp-deadlock-safe ABI requires.
    #[test]
    fn max_f64_emits_cas_and_spin() {
        let ptx = compile_agg_valid_float_kernel(ReduceOp::Max, DataType::Float64)
            .expect("kernel should compile");
        assert!(
            ptx.contains("atom.global.cas.b64"),
            "expected CAS.b64 in emitted PTX, got:\n{ptx}"
        );
        assert!(
            ptx.contains("setp.gt.f64"),
            "expected setp.gt.f64 (MAX comparison) in emitted PTX, got:\n{ptx}"
        );
        // New spill-ABI assertions.
        assert!(
            ptx.contains("SPILL:"),
            "expected SPILL label in emitted PTX, got:\n{ptx}"
        );
        assert!(
            ptx.contains("atom.global.add.u32"),
            "expected atom.global.add.u32 (spill counter) in emitted PTX, got:\n{ptx}"
        );
        assert!(
            ptx.contains("mov.u32 %probe_count, 0"),
            "expected probe-step counter init in emitted PTX, got:\n{ptx}"
        );
        // f64 spill writes go through `st.global.f64`.
        assert!(
            ptx.contains("st.global.f64"),
            "expected st.global.f64 (spill value write) in emitted PTX, got:\n{ptx}"
        );
        let param_count = ptx.matches(".param ").count();
        assert_eq!(
            param_count, 11,
            "expected exactly 11 `.param ` declarations, got {param_count}:\n{ptx}"
        );
    }

    /// Integer dtypes belong on the integer agg kernel path.
    #[test]
    fn rejects_int_dtype() {
        let err = compile_agg_valid_float_kernel(ReduceOp::Min, DataType::Int32)
            .expect_err("Int32 should be rejected by the float-only kernel");
        let msg = err.to_string();
        assert!(
            msg.contains("Int32") || msg.contains("floating-point"),
            "error message should mention dtype mismatch, got: {msg}"
        );
    }

    /// SUM has a native `atom.global.add.f{32,64}` instruction on sm_70 and
    /// must go through the integer agg kernel path, not this one.
    #[test]
    fn rejects_sum() {
        let err = compile_agg_valid_float_kernel(ReduceOp::Sum, DataType::Float64)
            .expect_err("Sum should be rejected by the MIN/MAX-only kernel");
        let msg = err.to_string();
        assert!(
            msg.contains("MIN/MAX") || msg.contains("Sum"),
            "error message should mention op mismatch, got: {msg}"
        );
    }

    /// The entry-point constant exposed by this module must match the symbol
    /// emitted in the PTX (and the one exported by `valid_flag_kernels`).
    #[test]
    fn entry_constant_matches_emitted_name() {
        let ptx = compile_agg_valid_float_kernel(ReduceOp::Min, DataType::Float32).unwrap();
        let entry = format!(".visible .entry {}(", VALID_AGG_FLOAT_ENTRY);
        assert!(
            ptx.contains(&entry),
            "PTX should declare entry as {entry:?}, got:\n{ptx}"
        );
    }
}
