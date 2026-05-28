// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for the single-pass open-addressing GPU hash-grouping kernels.
//!
//! Two kernels back the GROUP BY executor in `crate::exec::groupby`:
//!
//! 1. [`KEYS_KERNEL_ENTRY`] — `bolt_groupby_keys`. One thread per input row.
//!    Each thread hashes its key, then performs a linear-probe `atom.cas` loop
//!    on the keys table until either (a) it inserts the key into an empty slot
//!    or (b) it finds an existing slot containing the same key. No aggregate
//!    table is touched.
//!
//! 2. [`AGG_KERNEL_ENTRY`] — `bolt_groupby_agg`. Re-runs the same hash +
//!    probe sequence against an already-populated keys table to find the slot
//!    for this row, then issues a single `atom.global.<op>.<dtype>` on that
//!    slot in the accumulator table. The kernel handles ONE aggregate at a
//!    time; the host launches it once per aggregate.
//!
//! Splitting kernels this way keeps the parameter list small and the PTX
//! template manageable: every kernel takes pointers plus `n_rows` and `k`
//! (the table size, always a power of two so the probe can mask instead of
//! mod).
//!
//! ## ABIs
//!
//! Keys kernel:
//! ```text
//! .visible .entry bolt_groupby_keys(
//!     .param .u64 group_col_ptr,   // i64 group keys, length n_rows
//!     .param .u64 keys_table_ptr,  // i64, length k, init'd to EMPTY_KEY
//!     .param .u32 n_rows,
//!     .param .u32 k                // power-of-two table size
//! )
//! ```
//!
//! Agg kernel (input dtype `T` parameterises the load + atomic instruction):
//! ```text
//! .visible .entry bolt_groupby_agg(
//!     .param .u64 group_col_ptr,   // i64 group keys, length n_rows
//!     .param .u64 keys_table_ptr,  // i64, length k, fully populated
//!     .param .u64 input_col_ptr,   // T, length n_rows
//!     .param .u64 acc_table_ptr,   // T, length k, init'd to identity(op)
//!     .param .u32 n_rows,
//!     .param .u32 k
//! )
//! ```
//!
//! ## Sentinel
//!
//! The keys table is initialised on the host to `i64::MIN` and that value is
//! reserved as the "empty" sentinel; the executor validates no input key
//! equals this before launching.
//!
//! ## Restrictions
//!
//! `MIN` / `MAX` over floating-point inputs is implemented via a CAS loop in
//! [`crate::jit::float_atomics`]; this module only emits integer atomic
//! kernels. PTX `atom.global.min/max.fXX` does not exist for `Float32` /
//! `Float64` on sm_70, so float MIN/MAX combinations are rejected here and
//! the executor dispatches them to the float-atomics path instead.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
use crate::jit::agg_kernels::ReduceOp;
use crate::plan::logical_plan::DataType;

/// Splitmix-style multiplier used by the per-row hash. Public so tests in the
/// executor can replay the hash on the host while building the expected
/// `(key -> slot)` mapping.
///
/// This is the canonical declaration of the constant — sibling kernel modules
/// (notably [`crate::jit::valid_flag_kernels`]) redeclare the same value so
/// they can be compiled / tested standalone, but the bit pattern must match
/// the one here byte-for-byte, otherwise host-side hash replay against a
/// classic-kernel-built table will disagree with a valid-flag-built one.
// NOTE: this value must match valid_flag_kernels::FX_MUL.
pub const FX_MUL: i64 = 0x9E3779B97F4A7C15u64 as i64;

/// Entry point of the keys-only kernel emitted by [`compile_groupby_keys_kernel`].
pub const KEYS_KERNEL_ENTRY: &str = "bolt_groupby_keys";

/// Entry point of the aggregate-update kernel emitted by
/// [`compile_groupby_agg_kernel`].
pub const AGG_KERNEL_ENTRY: &str = "bolt_groupby_agg";

/// Threads per block for both grouping kernels.
const BLOCK_SIZE: u32 = 256;

/// PTX `i64::MIN` literal used as the "empty slot" sentinel.
const EMPTY_KEY_LITERAL: &str = "-9223372036854775808";

/// Upper bound on the linear-probe loop, expressed as a multiple of `k`.
/// At load factor < 0.5 (enforced by the executor) the expected probe length
/// is well under `log2(k)`, so a full table sweep is generous. The bound
/// exists purely to prevent a runaway kernel — if the host's load-factor
/// invariant is honoured, the bound never triggers. Mirrors the
/// `MAX_PROBE_FACTOR` constant in [`crate::jit::valid_flag_kernels`].
const MAX_PROBE_FACTOR: u32 = 2;

/// Generate PTX for the keys-building kernel. The kernel writes only to the
/// keys table; the accumulator tables are untouched.
///
/// # Encoding contract
///
/// The kernel treats every entry of `group_col_ptr` as an opaque `i64` and
/// uses bitwise equality (via `atom.cas.b64`) to compare keys. The host is
/// responsible for ENCODING the user's group-by columns into i64s before
/// upload. The currently used encodings (see `exec::groupby::pack_keys`):
///
/// * Single Int32 → sign-extended to i64.
/// * Single Int64 → identity bitcast.
/// * Single Float32 → `(f.to_bits() as u32) as i64` (bitwise-equal floats
///   group together; `-0.0` vs `+0.0` and NaN bit patterns differ).
/// * Single Float64 → `f.to_bits() as i64`.
/// * Two columns whose combined width fits in 64 bits → high 32 bits = first
///   column, low 32 bits = second column, each using the same bit
///   representation as the single-column case.
///
/// Because every supported encoding is LOSSLESS (distinct tuples ↦ distinct
/// i64), this kernel needs no awareness of the per-row column count.
pub fn compile_groupby_keys_kernel() -> BoltResult<String> {
    let mut ptx = String::new();

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {}(", KEYS_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", KEYS_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", KEYS_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_2,", KEYS_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_3", KEYS_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Generous `.reg` decls — only names, not real allocations.
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // tid = ctaid.x * ntid.x + tid.x ; bail if tid >= n_rows.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u32 %r4, [{}_param_2];",
        KEYS_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // Load k and compute k-1 (mask).
    writeln!(
        ptx,
        "\tld.param.u32 %r5, [{}_param_3];",
        KEYS_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;

    // max_probes = k * MAX_PROBE_FACTOR. Computed once at kernel entry so
    // the bounded PROBE_LOOP can compare against it cheaply. The host
    // enforces load factor < 0.5, so this bound is purely defensive — if it
    // ever triggers, the thread gives up silently rather than spin forever.
    writeln!(
        ptx,
        "\tmul.lo.u32 %r20, %r5, {factor};",
        factor = MAX_PROBE_FACTOR
    )
    .map_err(write_err)?;

    // Load this thread's key value from group_col.
    writeln!(
        ptx,
        "\tld.param.u64 %rd0, [{}_param_0];",
        KEYS_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd1, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rl0, [%rd2];").map_err(write_err)?;

    // Hash: h = (key * FX_MUL) >> 32 ; then & (k-1).
    writeln!(ptx, "\tmov.s64 %rl1, {};", FX_MUL).map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s64 %rl2, %rl0, %rl1;").map_err(write_err)?;
    writeln!(ptx, "\tshr.u64 %rl3, %rl2, 32;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u64 %r7, %rl3;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r7, %r6;").map_err(write_err)?;

    // Load keys_table base ptr.
    writeln!(
        ptx,
        "\tld.param.u64 %rd3, [{}_param_1];",
        KEYS_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;

    // %rl4 = EMPTY_KEY ; %rl0 still holds the key.
    writeln!(ptx, "\tmov.s64 %rl4, {};", EMPTY_KEY_LITERAL).map_err(write_err)?;

    // Bounded-probe counter. %r21 increments once per slot examined; on
    // overflow the thread bails to DONE rather than spinning indefinitely.
    writeln!(ptx, "\tmov.u32 %r21, 0;").map_err(write_err)?;

    // Probe loop. %r8 is the current slot; loops on collision.
    writeln!(ptx, "PROBE_LOOP:").map_err(write_err)?;
    // Bound check: probe_count += 1 ; if probe_count > max_probes -> DONE.
    // Give-up-silently semantics — the success path is unchanged. Host-side
    // post-launch detection of "did every key get placed?" is a separate
    // concern (see the valid-flag SPILL path for the version that surfaces
    // the overflow to the host).
    writeln!(ptx, "\tadd.u32 %r21, %r21, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.gt.u32 %p3, %r21, %r20;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra DONE;").map_err(write_err)?;
    // addr = keys_table + slot * 8
    writeln!(ptx, "\tmul.wide.u32 %rd4, %r8, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd5, %rd3, %rd4;").map_err(write_err)?;
    // atom.cas: try EMPTY -> key. Returns previous value.
    writeln!(
        ptx,
        "\tatom.global.cas.b64 %rl5, [%rd5], %rl4, %rl0;"
    )
    .map_err(write_err)?;
    // If old == EMPTY, we inserted.
    writeln!(ptx, "\tsetp.eq.s64 %p1, %rl5, %rl4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra DONE;").map_err(write_err)?;
    // If old == key, slot already holds our group.
    writeln!(ptx, "\tsetp.eq.s64 %p2, %rl5, %rl0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra DONE;").map_err(write_err)?;
    // Collision: advance slot (linear probe, masked to k).
    writeln!(ptx, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_LOOP;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Generate PTX for an aggregate-update kernel parameterised over `op` +
/// `input_dtype`. Assumes the keys table referenced by `keys_table_ptr` is
/// already fully populated by a prior [`compile_groupby_keys_kernel`] launch.
///
/// ## Cross-kernel synchronisation contract
///
/// The keys-kernel writes to `keys_table_ptr` and the agg-kernel reads from
/// it. The two kernels cooperate via the table but DO NOT synchronise
/// internally — the agg-kernel's probe loop assumes every slot that will
/// ever be written has already been written. The host is responsible for
/// enforcing that ordering, which means one of the following MUST hold
/// between the two launches:
///
/// * Both launches go on the same CUDA stream (CUDA's default in-order
///   semantics make this a memory-ordering no-op — the agg kernel's loads
///   are guaranteed to observe every store from the keys kernel), OR
/// * The host explicitly calls `cuStreamSynchronize` (or an equivalent
///   event-wait) between the two launches.
///
/// Cross-stream launches WITHOUT an explicit synchronise are a bug: the agg
/// kernel will see a partially-populated keys table, miss its slot during
/// linear probe, and either spin to the new bounded-probe limit and give up
/// silently OR (depending on probe path) atomically update the wrong slot.
/// Neither outcome is recoverable post-hoc. This invariant previously lived
/// only in scattered executor docstrings; it is restated here because the
/// agg-kernel PTX itself bakes it in as a pre-condition.
pub fn compile_groupby_agg_kernel(
    op: ReduceOp,
    input_dtype: DataType,
) -> BoltResult<String> {
    // Reject unsupported (op, dtype) combinations up front with explicit errors.
    let atomic = atomic_for(op, input_dtype)?;

    let (load_suffix, reg_class) = ptx_type_info(input_dtype)?;
    let elem_bytes = input_dtype.byte_width().ok_or_else(|| {
        BoltError::Other(format!(
            "hash_kernels: variable-width dtype {:?} not supported",
            input_dtype
        ))
    })?;

    let mut ptx = String::new();

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {}(", AGG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", AGG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", AGG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_2,", AGG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_3,", AGG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_4,", AGG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_5", AGG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    // A typed value register for the input column load. For Int64 inputs the
    // input value lives in the `%rl` namespace already used by the key path —
    // emitting a separate value register keeps the PTX uniform across dtypes.
    writeln!(
        ptx,
        "\t.reg .{ty}   %{rc}<4>;",
        ty = reg_decl_ty(input_dtype)?,
        rc = reg_class
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
        AGG_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // k and mask = k - 1.
    writeln!(
        ptx,
        "\tld.param.u32 %r5, [{}_param_5];",
        AGG_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;

    // Load the key for this row.
    writeln!(
        ptx,
        "\tld.param.u64 %rd0, [{}_param_0];",
        AGG_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd1, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rl0, [%rd2];").map_err(write_err)?;

    // Hash.
    writeln!(ptx, "\tmov.s64 %rl1, {};", FX_MUL).map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s64 %rl2, %rl0, %rl1;").map_err(write_err)?;
    writeln!(ptx, "\tshr.u64 %rl3, %rl2, 32;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u64 %r7, %rl3;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r7, %r6;").map_err(write_err)?;

    // Keys table base.
    writeln!(
        ptx,
        "\tld.param.u64 %rd3, [{}_param_1];",
        AGG_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;

    // Probe loop — non-mutating; the keys table is read-only here. We just
    // walk slots until we find the one whose key matches ours. (Keys kernel
    // ran first so we are guaranteed to find a matching slot.)
    writeln!(ptx, "PROBE_LOOP:").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd4, %r8, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd5, %rd3, %rd4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rl5, [%rd5];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s64 %p1, %rl5, %rl0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra FOUND;").map_err(write_err)?;
    // Otherwise advance.
    writeln!(ptx, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_LOOP;").map_err(write_err)?;
    writeln!(ptx, "FOUND:").map_err(write_err)?;

    // Load the input column value for this row.
    writeln!(
        ptx,
        "\tld.param.u64 %rd6, [{}_param_2];",
        AGG_KERNEL_ENTRY
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
    writeln!(
        ptx,
        "\tld.global.{ld} %{rc}0, [%rd8];",
        ld = load_suffix,
        rc = reg_class
    )
    .map_err(write_err)?;

    // Compute the accumulator slot address (acc_table + slot * elem_bytes).
    writeln!(
        ptx,
        "\tld.param.u64 %rd9, [{}_param_3];",
        AGG_KERNEL_ENTRY
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

    // Atomic update. PTX `atom` returns the old value into a destination
    // register; we don't need it, but the form requires one — reuse the
    // value register class with a fresh index.
    writeln!(
        ptx,
        "\t{atomic} %{rc}1, [%rd11], %{rc}0;",
        atomic = atomic,
        rc = reg_class
    )
    .map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Block size accessor for the host-side launcher. Kept private to the module
/// for now; the executor reads it via [`groupby_block_size`].
pub fn groupby_block_size() -> u32 {
    BLOCK_SIZE
}

/// PTX `atom.global.*` mnemonic (with no operands) for the given op + dtype.
/// Returns an error for combinations the v1 implementation does not support
/// (most notably float MIN/MAX, which would need a CAS loop).
fn atomic_for(op: ReduceOp, dtype: DataType) -> BoltResult<&'static str> {
    use DataType::*;
    use ReduceOp::*;
    Ok(match (op, dtype) {
        (Sum, Int32) | (Count, Int32) => "atom.global.add.s32",
        // PTX has no `atom.add.s64` — only `.u64`. Two's-complement signed
        // addition is bit-identical to unsigned addition, so emitting `.u64`
        // for an `Int64` accumulator is sound. See PTX ISA, "atom" —
        // supported types are {u32, s32, u64, f16, f16x2, f32, f64, bf16, bf16x2}.
        (Sum, Int64) | (Count, Int64) => "atom.global.add.u64",
        (Sum, Float32) | (Count, Float32) => "atom.global.add.f32",
        (Sum, Float64) | (Count, Float64) => "atom.global.add.f64",

        (Min, Int32) => "atom.global.min.s32",
        (Min, Int64) => "atom.global.min.s64",
        (Max, Int32) => "atom.global.max.s32",
        (Max, Int64) => "atom.global.max.s64",

        (Min, Float32) | (Min, Float64) | (Max, Float32) | (Max, Float64) => {
            return Err(BoltError::Other(
                "MIN/MAX over float not yet supported in GROUP BY".into(),
            ))
        }

        (_, Bool) | (_, Utf8) => {
            return Err(BoltError::Type(format!(
                "hash_kernels: aggregate over dtype {:?} not supported",
                dtype
            )))
        }
    })
}

/// `(ld_suffix, reg_class)` for the input column / accumulator value type.
///
/// The register class is intentionally distinct from the `%r`, `%rl`, `%rd`
/// classes used for hashing/probing so the two namespaces don't collide.
fn ptx_type_info(dtype: DataType) -> BoltResult<(&'static str, &'static str)> {
    Ok(match dtype {
        DataType::Int32 => ("s32", "vr"),
        DataType::Int64 => ("s64", "vl"),
        DataType::Float32 => ("f32", "vf"),
        DataType::Float64 => ("f64", "vd"),
        DataType::Bool | DataType::Utf8 => {
            return Err(BoltError::Type(format!(
                "hash_kernels: dtype {:?} not supported in aggregate kernel",
                dtype
            )))
        }
    })
}

/// PTX `.reg` declaration type for the input-value register class returned by
/// [`ptx_type_info`].
fn reg_decl_ty(dtype: DataType) -> BoltResult<&'static str> {
    Ok(match dtype {
        DataType::Int32 => "b32",
        DataType::Int64 => "b64",
        DataType::Float32 => "f32",
        DataType::Float64 => "f64",
        DataType::Bool | DataType::Utf8 => {
            return Err(BoltError::Type(format!(
                "hash_kernels: dtype {:?} not supported in aggregate kernel",
                dtype
            )))
        }
    })
}

/// Adapt a `std::fmt::Error` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("hash_kernels: write failed: {}", e))
}
