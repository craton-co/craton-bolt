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

use crate::error::{BoltError, BoltResult};
use crate::jit::partition_reduce_kernel::KeyWidth;
use crate::jit::partition_reduce_kernel_minmax::MinMaxOp;

pub const BLOCK_GROUPS: u32 = 1024;
pub const BLOCK_THREADS: u32 = 256;
pub const NUM_PARTITIONS: u32 = 4096;
const MAX_PROBES: u32 = BLOCK_GROUPS;

/// Per-iteration `nanosleep.u32` operand for the collision-advance path
/// (sm_70+). See `partition_reduce_kernel::SPIN_BACKOFF_NS` for full
/// rationale. TODO(perf): exponential back-off.
const SPIN_BACKOFF_NS: u32 = 32;

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
    format!("bolt_partition_reduce_{}_{}", opn, dt)
}

/// Entry-point name for the spill-counter variant.
pub fn kernel_entry_with_spill(op: MinMaxOp, dtype: FloatDtype) -> String {
    format!("{}_spill", kernel_entry(op, dtype))
}

pub fn compile_partition_reduce_kernel_minmax_float(
    op: MinMaxOp,
    dtype: FloatDtype,
) -> BoltResult<String> {
    let entry = kernel_entry(op, dtype);
    emit_minmax_float_kernel(KeyWidth::I32, op, dtype, entry.as_str(), "", false)
}

/// Unified key-width-parameterised float MIN/MAX generator.
///
/// Emits BOTH the i32-key kernel (this file's public
/// `compile_partition_reduce_kernel_minmax_float{,_with_spill}`) and the
/// i64-key kernel (`partition_reduce_kernel_minmax_float_i64`'s
/// `compile_partition_reduce_kernel_minmax_float_i64{,_with_spill}`, which
/// delegate here). The whole scaffold — header, entry/regs framing,
/// shmem-base + global-pointer setup, partition-slice read, the
/// publish/probe protocol, the float CAS-loop accumulate, and the export
/// control flow — is written ONCE. Only the genuinely key-dependent bytes
/// branch on `key_width`:
///
///   * shmem `block_keys` align + byte size (4 B/slot vs 8 B/slot),
///   * the `%rd<N>` register-file width (i32 keeps `%rd<64>`; i64 non-spill
///     `%rd<80>`, i64 spill `%rd<96>`),
///   * the zero-init key store width + scratch-register order,
///   * the probe-prologue key load (`s32`+direct slot vs `s64`+`cvt.u32.u64`)
///     and the slot-addr scratch-register order,
///   * the publish/probe `PublishRegs` + key-type token,
///   * the CLAIM key store width,
///   * the export key load/store width + scratch-register order.
///
/// The float value handling — the CAS retry loop, the `setp.{lt,gt}.f{32,64}`
/// comparison, the `selp.b{32,64}` pick, and the `±inf` identity init — is
/// INDEPENDENT of the key width and is emitted identically for both. The
/// `op`/`dtype`/`spill` runtime parameters are preserved exactly as the prior
/// within-file dedup left them.
///
/// `buf_suffix` distinguishes the shared-memory buffer symbol names (`""` for
/// the base kernel, `"_sp"` for the spill sibling) so the two compilation
/// units do not collide.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_minmax_float_kernel(
    key_width: KeyWidth,
    op: MinMaxOp,
    dtype: FloatDtype,
    entry: &str,
    buf_suffix: &str,
    spill: bool,
) -> BoltResult<String> {
    let mut ptx = String::new();
    let block_groups = BLOCK_GROUPS;
    let mask = BLOCK_GROUPS - 1;
    let block_threads = BLOCK_THREADS;
    let keys_bytes = match key_width {
        KeyWidth::I32 => BLOCK_GROUPS * 4,
        KeyWidth::I64 => BLOCK_GROUPS * 8,
    };
    let val_bytes_per_slot = dtype.bytes();
    let vals_bytes = BLOCK_GROUPS * val_bytes_per_slot;
    let set_bytes = BLOCK_GROUPS * 4;
    let max_probes = MAX_PROBES;
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

    super::partition_reduce_kernel_spill_common::emit_ptx_header(&mut ptx)?;

    emit_shared_buffers(
        &mut ptx,
        key_width,
        buf_suffix,
        val_bytes_per_slot,
        keys_bytes,
        vals_bytes,
        set_bytes,
    )?;

    emit_entry_and_regs(&mut ptx, key_width, entry, spill)?;

    emit_prologue(&mut ptx, entry, buf_suffix, spill)?;

    // Phase 1: zero shared. Vals init to ±infinity identity.
    emit_zero_phase(
        &mut ptx,
        key_width,
        dtype,
        &identity_lit,
        val_bytes_per_slot,
        block_groups,
        block_threads,
    )?;

    // Phase 2: probe + CAS-loop atomic MIN/MAX.
    let vbpw = val_bytes_per_slot;
    let val_reg = match dtype {
        FloatDtype::Float32 => "%f0",
        FloatDtype::Float64 => "%fd0",
    };
    emit_probe_loop(
        &mut ptx, key_width, op, dtype, val_load, val_reg, mask, max_probes, vbpw, spill,
    )?;

    // Phase 3: export.
    emit_export_phase(
        &mut ptx,
        key_width,
        dtype,
        vbpw,
        block_groups,
        block_threads,
        spill,
    )?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Emit the three `.shared` buffer declarations + trailing blank line.
/// `suffix` is appended to each symbol name (`""` base / `"_sp"` spill). The
/// `block_keys` align is the only key-width divergence (4 B/slot for i32,
/// 8 B/slot for i64).
fn emit_shared_buffers(
    ptx: &mut String,
    key_width: KeyWidth,
    suffix: &str,
    val_align: u32,
    keys_bytes: u32,
    vals_bytes: u32,
    set_bytes: u32,
) -> BoltResult<()> {
    let keys_align = match key_width {
        KeyWidth::I32 => 4,
        KeyWidth::I64 => 8,
    };
    writeln!(
        ptx,
        ".shared .align {ka} .b8 block_keys_buf{suffix}[{bytes}];",
        ka = keys_align,
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

/// Emit the `.visible .entry` header (5 params for the base kernel, 6 for
/// the spill sibling), the opening brace, and the register-file
/// declarations. The non-spill kernel additionally declares the
/// `%nstime` back-off operand; the spill kernel drops it (its collision
/// path has no back-off).
fn emit_entry_and_regs(
    ptx: &mut String,
    key_width: KeyWidth,
    entry: &str,
    spill: bool,
) -> BoltResult<()> {
    let last_param = if spill { 6 } else { 5 };
    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    for p in 0..last_param {
        writeln!(ptx, "\t.param .u64 {entry}_param_{p},").map_err(write_err)?;
    }
    writeln!(ptx, "\t.param .u64 {entry}_param_{last_param}").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // `%rd` register-file width. The i32-key kernel uses `%rd<64>` for both
    // variants; the i64-key kernel needs the wider key-handling slots —
    // `%rd<80>` non-spill, `%rd<96>` spill (the spill null-check + bump
    // consume the extra `%rd` slots).
    let rd_count = match (key_width, spill) {
        (KeyWidth::I32, _) => 64,
        (KeyWidth::I64, false) => 80,
        (KeyWidth::I64, true) => 96,
    };
    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<64>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<{rd_count}>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f32   %f<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<16>;").map_err(write_err)?;
    if !spill {
        // Operand register for the per-collision / per-retry `nanosleep.u32`
        // back-off (sm_70+).
        writeln!(ptx, "\t.reg .u32   %nstime;").map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;
    Ok(())
}

/// Emit the kernel prologue: thread/block-id setup, the three shared-buffer
/// base-pointer movs, the six global-param loads (+ the seventh spill-counter
/// load when `spill`), and the partition-slice read. `suffix` matches the
/// shared-buffer symbol suffix from [`emit_shared_buffers`].
fn emit_prologue(ptx: &mut String, entry: &str, suffix: &str, spill: bool) -> BoltResult<()> {
    super::partition_reduce_kernel_spill_common::emit_thread_block_ids(ptx)?;
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf{suffix};").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_vals_buf{suffix};").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf{suffix};").map_err(write_err)?;

    for (rd, p) in (3..=8).zip(0..6) {
        writeln!(ptx, "\tld.param.u64 %rd{rd}, [{entry}_param_{p}];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd{rd}, %rd{rd};").map_err(write_err)?;
    }
    if spill {
        writeln!(ptx, "\tld.param.u64 %rd9, [{entry}_param_6];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd9, %rd9;").map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    // Read partition slice.
    super::partition_reduce_kernel_spill_common::emit_partition_slice_read(ptx)?;
    writeln!(ptx).map_err(write_err)?;
    Ok(())
}

/// Emit Phase 1 — the shared-memory zero/identity-init loop. Identical for
/// both variants.
fn emit_zero_phase(
    ptx: &mut String,
    key_width: KeyWidth,
    dtype: FloatDtype,
    identity_lit: &str,
    vbpw: u32,
    block_groups: u32,
    block_threads: u32,
) -> BoltResult<()> {
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r20, {bg};", bg = block_groups).map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    // block_keys[s] = 0 — i32 stores 4 B (offset %rd20 also serves the set
    // store), i64 stores 8 B (set needs its own ×4 offset %rd25).
    match key_width {
        KeyWidth::I32 => {
            writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u32 [%rd21], 0;").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u64 [%rd21], 0;").map_err(write_err)?;
        }
    }
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
    match key_width {
        KeyWidth::I32 => {
            // block_set addressed at rd2 + s*4, reusing the key offset %rd20.
            writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd20;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u32 [%rd24], 0;").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            // block_set needs its own ×4 offset (%rd20 holds the ×8 key offset).
            writeln!(ptx, "\tmul.wide.u32 %rd25, %r20, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd25;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u32 [%rd24], 0;").map_err(write_err)?;
        }
    }
    writeln!(ptx, "\tadd.u32 %r20, %r20, {bt};", bt = block_threads).map_err(write_err)?;
    writeln!(ptx, "\tbra ZERO_TOP;").map_err(write_err)?;
    writeln!(ptx, "ZERO_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;
    Ok(())
}

/// Emit Phase 2 — the probe loop: load key/val, open-address probe with
/// CAS-claim, the shared publish/probe protocol, collision advance, and the
/// `CLAIM`/`MATCH` CAS-loop accumulate. Divergence between the two variants:
///
/// * `spill` redirects the `MAX_PROBES`-overflow branch to `SPILL_BUMP`
///   (instead of `LOOP_NEXT`), drops the collision-advance back-off, and
///   emits an explicit `bra LOOP_NEXT` after the `MATCH` CAS loop followed
///   by the `SPILL_BUMP` block; the non-spill variant lets `MATCH` fall
///   through into `LOOP_NEXT`.
/// * Both variants end with the shared `LOOP_NEXT:`/`LOOP_DONE:` epilogue.
#[allow(clippy::too_many_arguments)]
fn emit_probe_loop(
    ptx: &mut String,
    key_width: KeyWidth,
    op: MinMaxOp,
    dtype: FloatDtype,
    val_load: &str,
    val_reg: &str,
    mask: u32,
    max_probes: u32,
    vbpw: u32,
    spill: bool,
) -> BoltResult<()> {
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    // key load + slot derivation. i32 loads an s32 key straight into %r31 and
    // masks it directly; i64 loads an s64 key into %rd60 then derives the slot
    // from the low 32 bits via cvt.u32.u64.
    match key_width {
        KeyWidth::I32 => {
            writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
            writeln!(ptx, "\tld.global.s32 %r31, [%rd31];").map_err(write_err)?;
            // val (typed float)
            writeln!(ptx, "\tmul.wide.u32 %rd32, %r30, {vbpw};").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd33, %rd4, %rd32;").map_err(write_err)?;
            writeln!(ptx, "\t{val_load} {val_reg}, [%rd33];").map_err(write_err)?;
            // slot = key & mask
            writeln!(ptx, "\tand.b32 %r32, %r31, 0x{mask:X};", mask = mask).map_err(write_err)?;
        }
        KeyWidth::I64 => {
            writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
            writeln!(ptx, "\tld.global.s64 %rd60, [%rd31];").map_err(write_err)?;
            // val (typed float)
            writeln!(ptx, "\tmul.wide.u32 %rd32, %r30, {vbpw};").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd33, %rd4, %rd32;").map_err(write_err)?;
            writeln!(ptx, "\t{val_load} {val_reg}, [%rd33];").map_err(write_err)?;
            // slot = low32(key) & mask
            writeln!(ptx, "\tcvt.u32.u64 %r31, %rd60;").map_err(write_err)?;
            writeln!(ptx, "\tand.b32 %r32, %r31, 0x{mask:X};", mask = mask).map_err(write_err)?;
        }
    }
    writeln!(ptx, "\tmov.u32 %r33, 0;").map_err(write_err)?;

    writeln!(ptx, "PROBE_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r33, %r33, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.gt.u32 %p2, %r33, {mp};", mp = max_probes).map_err(write_err)?;
    let overflow_target = if spill { "SPILL_BUMP" } else { "LOOP_NEXT" };
    writeln!(ptx, "\t@%p2 bra {overflow_target};").map_err(write_err)?;

    // Slot-address compute. addr_set is ×4 in both; addr_key is ×4 reusing
    // %rd34 for i32 vs a separate ×8 (%rd39) for i64; addr_val is ×vbpw.
    match key_width {
        KeyWidth::I32 => {
            writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?; // addr_set
            writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd34;").map_err(write_err)?; // addr_key
            writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, {vbpw};").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?; // addr_val
        }
        KeyWidth::I64 => {
            writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?; // addr_set
            writeln!(ptx, "\tmul.wide.u32 %rd39, %r32, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd39;").map_err(write_err)?; // addr_key (i64)
            writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, {vbpw};").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?; // addr_val
        }
    }

    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;

    // 3-state publish protocol (claim-then-write race fix; see
    // partition_reduce_kernel.rs). VOLATILE SHARED re-read of set (not
    // ld.acquire.cta — that defaults to global and faults on the shared
    // offset) + nanosleep yield until set:=2, THEN read the key. Only the key
    // destination / probe-key registers and the key-type token branch on width.
    let (key_dst_reg, probe_key_reg, key_ty) = match key_width {
        KeyWidth::I32 => ("%r35", "%r31", "s32"),
        KeyWidth::I64 => ("%rd61", "%rd60", "s64"),
    };
    super::partition_reduce_kernel_spill_common::emit_publish_probe_protocol(
        ptx,
        &super::partition_reduce_kernel_spill_common::PublishRegs {
            set_flag_reg: "%r36",
            set_addr_reg: "%rd35",
            key_addr_reg: "%rd36",
            key_dst_reg,
            probe_key_reg,
        },
        key_ty,
    )?;
    writeln!(ptx, "\tadd.u32 %r32, %r32, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r32, %r32, 0x{mask:X};", mask = mask).map_err(write_err)?;
    if !spill {
        // Occupancy-friendly back-off on the collision-advance path.
        super::partition_reduce_kernel_spill_common::emit_spin_backoff(ptx, SPIN_BACKOFF_NS)?;
    }
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    // CLAIM: publish key, fence, then enter the CAS-loop to set the val. Only
    // the key store width (u32+%r31 vs u64+%rd60) branches on width.
    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    match key_width {
        KeyWidth::I32 => {
            writeln!(ptx, "\tst.shared.u32 [%rd36], %r31;").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            writeln!(ptx, "\tst.shared.u64 [%rd36], %rd60;").map_err(write_err)?;
        }
    }
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd35], 2;").map_err(write_err)?;
    emit_cas_loop(ptx, op, dtype, "CLAIM_CAS", val_reg)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    // MATCH: slot already holds our key. CAS-loop to update the val.
    writeln!(ptx, "MATCH:").map_err(write_err)?;
    emit_cas_loop(ptx, op, dtype, "MATCH_CAS", val_reg)?;

    if spill {
        // Spill variant: MATCH branches out explicitly (the SPILL_BUMP block
        // is interposed before the shared LOOP_NEXT epilogue).
        writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;
        super::partition_reduce_kernel_spill_common::emit_spill_bump_with_null_check(ptx, 9)?;
        super::partition_reduce_kernel_spill_common::emit_loop_next_done(ptx)?;
    } else {
        // Non-spill: MATCH falls straight through into LOOP_NEXT.
        writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
        writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
        writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
        writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
        writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
        writeln!(ptx).map_err(write_err)?;
    }
    Ok(())
}

/// Emit Phase 3 — the export loop. The only divergence is the predicate
/// numbering: the spill variant has already consumed `%p5` in its
/// `SPILL_BUMP` null-check, so its export loop-guard / set-test predicates
/// are `%p6`/`%p7` instead of `%p5`/`%p6`.
fn emit_export_phase(
    ptx: &mut String,
    key_width: KeyWidth,
    dtype: FloatDtype,
    vbpw: u32,
    block_groups: u32,
    block_threads: u32,
    spill: bool,
) -> BoltResult<()> {
    let (p_guard, p_set) = if spill { ("%p6", "%p7") } else { ("%p5", "%p6") };
    writeln!(ptx, "\tmul.lo.u32 %r40, %r0, {bg};", bg = block_groups).map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r41, %r2;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 {p_guard}, %r41, {bg};", bg = block_groups).map_err(write_err)?;
    writeln!(ptx, "\t@{p_guard} bra EXPORT_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r42, %r40, %r41;").map_err(write_err)?;

    // Load shared slot key + val + set. The key load width / offset stride and
    // the set-offset scratch register branch on key width; i32 reuses the ×4
    // key offset %rd44 for the set load, i64 needs its own ×4 offset %rd48.
    let export_val_reg = match dtype {
        FloatDtype::Float32 => "%f8",
        FloatDtype::Float64 => "%fd8",
    };
    match key_width {
        KeyWidth::I32 => {
            writeln!(ptx, "\tmul.wide.u32 %rd44, %r41, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd45, %rd0, %rd44;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.s32 %r47, [%rd45];").map_err(write_err)?;
            writeln!(ptx, "\tmul.wide.u32 %rd46, %r41, {vbpw};").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd47, %rd1, %rd46;").map_err(write_err)?;
            match dtype {
                FloatDtype::Float32 => {
                    writeln!(ptx, "\tld.shared.f32 {export_val_reg}, [%rd47];")
                        .map_err(write_err)?;
                }
                FloatDtype::Float64 => {
                    writeln!(ptx, "\tld.shared.f64 {export_val_reg}, [%rd47];")
                        .map_err(write_err)?;
                }
            }
            writeln!(ptx, "\tadd.s64 %rd49, %rd2, %rd44;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.u32 %r49, [%rd49];").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            writeln!(ptx, "\tmul.wide.u32 %rd44, %r41, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd45, %rd0, %rd44;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.s64 %rd62, [%rd45];").map_err(write_err)?;
            writeln!(ptx, "\tmul.wide.u32 %rd46, %r41, {vbpw};").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd47, %rd1, %rd46;").map_err(write_err)?;
            match dtype {
                FloatDtype::Float32 => {
                    writeln!(ptx, "\tld.shared.f32 {export_val_reg}, [%rd47];")
                        .map_err(write_err)?;
                }
                FloatDtype::Float64 => {
                    writeln!(ptx, "\tld.shared.f64 {export_val_reg}, [%rd47];")
                        .map_err(write_err)?;
                }
            }
            writeln!(ptx, "\tmul.wide.u32 %rd48, %r41, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd49, %rd2, %rd48;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.u32 %r49, [%rd49];").map_err(write_err)?;
        }
    }
    writeln!(ptx, "\tsetp.ne.s32 {p_set}, %r49, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r50, 1, 0, {p_set};").map_err(write_err)?;

    // Store the slot out to global. Key store width / offset stride branch on
    // key width; the val (×vbpw) and set (×1) stores are shared.
    match key_width {
        KeyWidth::I32 => {
            writeln!(ptx, "\tmul.wide.u32 %rd50, %r42, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd51, %rd6, %rd50;").map_err(write_err)?;
            writeln!(ptx, "\tst.global.s32 [%rd51], %r47;").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            writeln!(ptx, "\tmul.wide.u32 %rd50, %r42, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd51, %rd6, %rd50;").map_err(write_err)?;
            writeln!(ptx, "\tst.global.s64 [%rd51], %rd62;").map_err(write_err)?;
        }
    }
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
    Ok(())
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
) -> BoltResult<()> {
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
    writeln!(
        ptx,
        "\t@!{won_pred} bra {lp}_LOAD;",
        lp = label_prefix
    )
    .map_err(write_err)?;
    writeln!(ptx, "{lp}_DONE:", lp = label_prefix).map_err(write_err)?;
    Ok(())
}

/// Spill-counter-aware sibling. Identical to
/// [`compile_partition_reduce_kernel_minmax_float`] with one extra
/// `.param .u64 spill_counter` (uint32_t*, may be null). On MAX_PROBES
/// overflow null-checks the pointer then `atom.global.add.u32` it.
pub fn compile_partition_reduce_kernel_minmax_float_with_spill(
    op: MinMaxOp,
    dtype: FloatDtype,
) -> BoltResult<String> {
    let entry = kernel_entry_with_spill(op, dtype);
    emit_minmax_float_kernel(KeyWidth::I32, op, dtype, entry.as_str(), "_sp", true)
}

fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!(
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

    // ----- _with_spill variant shape tests ---------------------------------

    #[test]
    fn with_spill_compiles_all_four_combos() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [FloatDtype::Float32, FloatDtype::Float64] {
                let ptx = compile_partition_reduce_kernel_minmax_float_with_spill(op, dt)
                    .unwrap_or_else(|e| panic!("{:?}/{:?}: {e}", op, dt));
                assert!(!ptx.is_empty());
            }
        }
    }

    #[test]
    fn with_spill_distinct_entry_and_spill_atomic() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [FloatDtype::Float32, FloatDtype::Float64] {
                let base = kernel_entry(op, dt);
                let spill = kernel_entry_with_spill(op, dt);
                assert_ne!(base, spill);
                assert!(spill.ends_with("_spill"));
                let ptx = compile_partition_reduce_kernel_minmax_float_with_spill(op, dt).unwrap();
                let needle = format!(".visible .entry {spill}(");
                assert!(ptx.contains(&needle));
                assert!(ptx.contains("atom.global.add.u32"));
                assert!(ptx.contains("SPILL_BUMP:"));
                assert!(ptx.contains("setp.eq.u64"));
            }
        }
    }
}
