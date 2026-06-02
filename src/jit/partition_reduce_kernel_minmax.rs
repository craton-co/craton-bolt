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

use crate::error::{BoltError, BoltResult};
use super::partition_reduce_kernel::KeyWidth;

pub const BLOCK_GROUPS: u32 = 1024;
pub const BLOCK_THREADS: u32 = 256;
pub const NUM_PARTITIONS: u32 = 4096;
const MAX_PROBES: u32 = BLOCK_GROUPS;

/// Per-iteration `nanosleep.u32` operand for the collision-advance path
/// (sm_70+). See `partition_reduce_kernel::SPIN_BACKOFF_NS` for the full
/// rationale. TODO(perf): exponential back-off.
const SPIN_BACKOFF_NS: u32 = 32;

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
    #[allow(dead_code)]
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
    format!("bolt_partition_reduce_{}_{}", op.name(), dt)
}

/// Entry-point name for the spill-counter variant.
pub fn kernel_entry_with_spill(op: MinMaxOp, dtype: MinMaxDtype) -> String {
    format!("{}_spill", kernel_entry(op, dtype))
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
) -> BoltResult<String> {
    let entry = kernel_entry(op, dtype);
    emit_minmax_kernel(KeyWidth::I32, op, /* spill = */ false, &entry, dtype)
}

/// Spill-counter-aware sibling of
/// [`compile_partition_reduce_kernel_minmax`]. Same body with one extra
/// `.param .u64 spill_counter` (uint32_t*, may be 0/null). On
/// MAX_PROBES overflow, atomically bumps the counter then drops the row.
pub fn compile_partition_reduce_kernel_minmax_with_spill(
    op: MinMaxOp,
    dtype: MinMaxDtype,
) -> BoltResult<String> {
    let entry = kernel_entry_with_spill(op, dtype);
    emit_minmax_kernel(KeyWidth::I32, op, /* spill = */ true, &entry, dtype)
}

// ===========================================================================
// Unified key-width-parameterised MIN/MAX-int generator.
//
// `emit_minmax_kernel` emits ALL EIGHT integer MIN/MAX variants:
//   key_width ∈ {I32, I64} × spill ∈ {false, true} × the runtime `op` and
//   `dtype` params. The i32-key kernels are emitted directly from this file's
//   public `compile_partition_reduce_kernel_minmax{,_with_spill}`; the i64-key
//   kernels delegate here from `partition_reduce_kernel_minmax_i64`'s
//   `compile_partition_reduce_kernel_minmax_i64{,_with_spill}`.
//
// The whole scaffold — header, shmem decls, entry/param framing, register
// file, thread/pointer setup, partition-slice read, zero-init, the probe
// loop, the publish/probe protocol, CLAIM/MATCH, the spill handler, and the
// export loop — is written ONCE. Only the genuinely KEY-width-dependent bytes
// branch on `key_width`:
//
//   * shmem `block_keys` align + byte size (4 B/slot vs 8 B/slot),
//   * the `%rd<N>` register-file width (i32 keeps `%rd<64>`; i64 non-spill
//     `%rd<80>`, i64 spill `%rd<96>`),
//   * the zero-init key store width + the set-offset scratch register,
//   * the probe-prologue key load (`s32`+direct slot vs `s64`+`cvt.u32.u64`),
//     the per-dtype value register (`%r40`/`%rd40` vs `%r40`/`%rd70`), and the
//     key-slot address scratch register,
//   * the publish/probe `PublishRegs` + key-type token,
//   * the CLAIM/MATCH key store width + the per-dtype atomic scratch dsts,
//   * the export key load/store width + scratch-register order + per-dtype
//     export value register.
//
// `op`/`dtype` stay runtime params throughout (the atomic op, the identity
// literal, the value load/store width). The export-loop predicates are passed
// down per spill state — the i64 spill variant shifts them by one because its
// null-checked `SPILL_BUMP` consumes `%p5`; the i32 spill variant does the
// same (its `SPILL_BUMP` is also null-checked here). The 12 golden snapshots
// in `tests/ptx_golden_partition_snapshots.rs` pin the emitted bytes.
// ===========================================================================

/// Emit the full per-partition integer MIN/MAX kernel for the given
/// `key_width`/`op`/`spill`/`entry`/`dtype`.
///
/// `spill == true` appends the trailing `spill_counter` `.u64` param + the
/// null-checked `SPILL_BUMP` overflow handler and drops the collision-advance
/// back-off. `dtype` selects the VALUE width (independent of the KEY width):
/// the value load/store/atomic width and the value/export registers.
pub(crate) fn emit_minmax_kernel(
    key_width: KeyWidth,
    op: MinMaxOp,
    spill: bool,
    entry: &str,
    dtype: MinMaxDtype,
) -> BoltResult<String> {
    let mut ptx = String::new();
    let layout = Layout::new(key_width, dtype, op, spill);

    emit_prologue(&mut ptx, entry, &layout, op, dtype)?;
    emit_probe_loop(&mut ptx, &layout, spill)?;

    // MATCH: non-spill falls straight through into the inline `LOOP_NEXT:`
    // label, so it emits the atomic with NO trailing `bra` and then inlines the
    // loop-next/done epilogue. The spill variant `bra`s to LOOP_NEXT (its
    // SPILL_BUMP block sits between MATCH and the epilogue).
    writeln!(ptx, "MATCH:").map_err(write_err)?;
    writeln!(
        ptx,
        "\t{atom_op} {scratch_reg2}, [%rd38], {val_reg};",
        atom_op = layout.atom_op,
        scratch_reg2 = layout.match_scratch_reg,
        val_reg = layout.val_reg,
    )
    .map_err(write_err)?;

    if spill {
        writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;
        super::partition_reduce_kernel_spill_common::emit_spill_bump_with_null_check(&mut ptx, 9)?;
        super::partition_reduce_kernel_spill_common::emit_loop_next_done(&mut ptx)?;
    } else {
        writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
        writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
        writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
        writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
        writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
        writeln!(ptx).map_err(write_err)?;
    }

    // Phase 3: export. The base predicates are `%p5`/`%p6`. ONLY the spill
    // variants shift to `%p6`/`%p7`: their null-checked `SPILL_BUMP` consumes
    // `%p5`.
    let (p_export, p_set) = if spill { ("%p6", "%p7") } else { ("%p5", "%p6") };
    emit_export(&mut ptx, &layout, p_export, p_set)?;

    Ok(ptx)
}

// ===========================================================================
// Private shared emitters for the unified integer MIN/MAX generator.
//
// `emit_prologue` / `emit_probe_loop` / `emit_export` carry every shared byte
// once. The spill/non-spill divergences (buffer suffix, extra param +
// spill-pointer load, the `%nstime` reg decl, the overflow branch target, the
// collision back-off, the MATCH trailing branch, the export predicate numbers)
// AND the key-width divergences (key slot stride, the `%rd<N>` reg width, the
// key load/store widths, the slot-derivation, and the per-key scratch
// registers) are all carried on the `Layout` (or a `spill: bool` param), so
// each (key_width × spill × op × dtype) variant reproduces its exact bytes.
// ===========================================================================

/// Per-emitter constants + the key-width / spill / dtype-divergent tokens.
struct Layout {
    /// Which key width (i32 single key vs i64 packed two-key).
    key_width: KeyWidth,
    /// `BLOCK_GROUPS`.
    block_groups: u32,
    /// `BLOCK_GROUPS - 1`, the slot mask.
    mask: u32,
    /// `BLOCK_THREADS`, the zero-init / export stride.
    block_threads: u32,
    /// Value width in bytes (4 for i32, 8 for i64) — the `mul.wide` multiplier.
    vbpw: u32,
    /// `MAX_PROBES`.
    max_probes: u32,
    /// `atom.shared.{min,max}.{s32,s64}` for this (op, dtype).
    atom_op: String,
    /// Shared-buffer name suffix: `""` (non-spill) or `"_sp"` (spill).
    buf_suffix: &'static str,
    /// Whether this is the `_with_spill` variant.
    spill: bool,
    /// Typed value register (per dtype, and per key width for i64-dtype).
    val_reg: &'static str,
    /// CLAIM-path atomic scratch dst (per dtype, per key width for i64-dtype).
    scratch_reg: &'static str,
    /// MATCH-path atomic scratch dst (per dtype, per key width for i64-dtype).
    match_scratch_reg: &'static str,
    /// Export typed value register (per dtype, per key width for i64-dtype).
    export_val_reg: &'static str,
}

impl Layout {
    fn new(key_width: KeyWidth, dtype: MinMaxDtype, op: MinMaxOp, spill: bool) -> Self {
        let i64_val = dtype == MinMaxDtype::Int64;
        let i64_key = key_width == KeyWidth::I64;
        Layout {
            key_width,
            block_groups: BLOCK_GROUPS,
            mask: BLOCK_GROUPS - 1,
            block_threads: BLOCK_THREADS,
            vbpw: dtype.bytes(),
            max_probes: MAX_PROBES,
            atom_op: format!("atom.shared.{}.{}", op.name(), dtype.ptx_atom_suffix()),
            buf_suffix: if spill { "_sp" } else { "" },
            spill,
            // The i64-key file uses a higher `%rd` block for its typed value /
            // scratch / export registers (`%rd70`/`%rd72`/`%rd73`/`%rd74`) to
            // stay clear of the i64 KEY registers (`%rd60`/`%rd61`/`%rd62`);
            // the i32-key file packs them lower (`%rd40`/`%rd42`/`%rd43`/`%rd48`).
            val_reg: match (i64_val, i64_key) {
                (false, _) => "%r40",
                (true, false) => "%rd40",
                (true, true) => "%rd70",
            },
            scratch_reg: match (i64_val, i64_key) {
                (false, _) => "%r42",
                (true, false) => "%rd42",
                (true, true) => "%rd72",
            },
            match_scratch_reg: match (i64_val, i64_key) {
                (false, _) => "%r43",
                (true, false) => "%rd43",
                (true, true) => "%rd73",
            },
            export_val_reg: match (i64_val, i64_key) {
                (false, _) => "%r48",
                (true, false) => "%rd48",
                (true, true) => "%rd74",
            },
        }
    }
}

/// Emit phases 0–1 (header through zero-init). The key-width divergences are
/// the `block_keys` align + byte size, the `%rd<N>` register-file width, the
/// zero-init key store width, and the set-offset scratch register; the
/// spill divergences (carried on the `Layout`) are the buffer suffix, the
/// extra param + `%rd9` load, and the `%nstime` reg decl.
fn emit_prologue(
    ptx: &mut String,
    entry: &str,
    layout: &Layout,
    op: MinMaxOp,
    dtype: MinMaxDtype,
) -> BoltResult<()> {
    let block_groups = layout.block_groups;
    let block_threads = layout.block_threads;
    let vbpw = layout.vbpw;
    let i64_key = layout.key_width == KeyWidth::I64;
    let keys_align = if i64_key { 8 } else { 4 };
    let keys_bytes = if i64_key { BLOCK_GROUPS * 8 } else { BLOCK_GROUPS * 4 };
    let vals_bytes = BLOCK_GROUPS * vbpw;
    let set_bytes = BLOCK_GROUPS * 4;
    let sfx = layout.buf_suffix;

    super::partition_reduce_kernel_spill_common::emit_ptx_header(ptx)?;

    writeln!(
        ptx,
        ".shared .align {al} .b8 block_keys_buf{sfx}[{bytes}];",
        al = keys_align,
        bytes = keys_bytes
    )
    .map_err(write_err)?;
    let val_align = if dtype == MinMaxDtype::Int64 { 8 } else { 4 };
    writeln!(
        ptx,
        ".shared .align {a} .b8 block_vals_buf{sfx}[{bytes}];",
        a = val_align,
        bytes = vals_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 4 .b8 block_set_buf{sfx}[{bytes}];",
        bytes = set_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // The non-spill kernel has 6 params (0..=5); the spill kernel has 7
    // (0..=6). `last_param` is the trailing comma-less one.
    let last_param = if layout.spill { 6 } else { 5 };
    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    for p in 0..last_param {
        writeln!(ptx, "\t.param .u64 {entry}_param_{p},").map_err(write_err)?;
    }
    writeln!(ptx, "\t.param .u64 {entry}_param_{last_param}").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<64>;").map_err(write_err)?;
    // `%rd` width: i32 key keeps `%rd<64>` for both spill/non-spill. The i64
    // key needs the wider key registers: `%rd<80>` non-spill, `%rd<96>` spill.
    let rd_count = match (i64_key, layout.spill) {
        (false, _) => 64,
        (true, false) => 80,
        (true, true) => 96,
    };
    writeln!(ptx, "\t.reg .b64   %rd<{rd_count}>;").map_err(write_err)?;
    if !layout.spill {
        // Operand register for the per-collision `nanosleep.u32` back-off.
        // (The spill variant has no collision back-off, so it omits this.)
        writeln!(ptx, "\t.reg .u32   %nstime;").map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    super::partition_reduce_kernel_spill_common::emit_thread_block_ids(ptx)?;
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf{sfx};").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_vals_buf{sfx};").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf{sfx};").map_err(write_err)?;

    // Global pointers.
    //  %rd3 = partition_keys (i32* or i64*)
    //  %rd4 = partition_vals (i32* or i64*)
    //  %rd5 = partition_offsets (u32*)
    //  %rd6 = out_keys, %rd7 = out_vals, %rd8 = out_set
    //  (spill only) %rd9 = spill counter.
    for (rd, p) in (3..=8).zip(0..6) {
        writeln!(ptx, "\tld.param.u64 %rd{rd}, [{entry}_param_{p}];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd{rd}, %rd{rd};").map_err(write_err)?;
    }
    if layout.spill {
        writeln!(ptx, "\tld.param.u64 %rd9, [{entry}_param_6];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd9, %rd9;").map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    // start, end = offsets[pid], offsets[pid+1]
    super::partition_reduce_kernel_spill_common::emit_partition_slice_read(ptx)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 1: zero shared. Keys + set = 0; vals = IDENTITY.
    let identity_lit: String = match dtype {
        MinMaxDtype::Int32 => format!("0x{:X}", op.identity_i32() as u32),
        MinMaxDtype::Int64 => format!("0x{:X}", op.identity_i64() as u64),
    };
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r20, {bg};", bg = block_groups).map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    match layout.key_width {
        KeyWidth::I32 => {
            // block_keys[s] = 0  (i32, 4 B). The key offset (%rd20, ×4) is
            // reused for block_set below.
            writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u32 [%rd21], 0;").map_err(write_err)?;
            // block_vals[s] = identity
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
            // block_set[s] = 0  (reuses the ×4 key offset %rd20)
            writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd20;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u32 [%rd24], 0;").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            // block_keys[s] = 0  (i64, 8 B). The key offset (%rd20, ×8) cannot
            // be reused for block_set, so a separate ×4 offset (%rd25) is cut.
            writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u64 [%rd21], 0;").map_err(write_err)?;
            // block_vals[s] = identity
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
            // block_set[s] = 0  (u32, 4 B — needs its own ×4 offset %rd25)
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

/// Emit Phase 2: the probe loop from `add.u32 %r30, ...` through the CLAIM
/// block (the MATCH block is emitted per-caller). The key-width divergences are
/// the key load (`s32`+direct slot vs `s64`+`cvt.u32.u64`), the key-slot
/// address scratch register, the publish/probe register tuple + key-type token,
/// and the CLAIM key store width. The spill divergences are the
/// MAX_PROBES-overflow branch target and the collision back-off.
fn emit_probe_loop(ptx: &mut String, layout: &Layout, spill: bool) -> BoltResult<()> {
    let mask = layout.mask;
    let vbpw = layout.vbpw;
    let max_probes = layout.max_probes;
    let atom_op = &layout.atom_op;
    let val_reg = layout.val_reg;

    // Phase 2: probe + atomic {min,max}.
    super::partition_reduce_kernel_spill_common::emit_loop_head(ptx)?;

    let val_load = layout_load(val_reg);
    match layout.key_width {
        KeyWidth::I32 => {
            // key = partition_keys[i] (i32)
            writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
            writeln!(ptx, "\tld.global.s32 %r31, [%rd31];").map_err(write_err)?;

            // val = partition_vals[i] (typed)
            writeln!(ptx, "\tmul.wide.u32 %rd32, %r30, {vbpw};").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd33, %rd4, %rd32;").map_err(write_err)?;
            writeln!(ptx, "\t{val_load} {val_reg}, [%rd33];").map_err(write_err)?;

            // slot = key & mask
            writeln!(ptx, "\tand.b32 %r32, %r31, 0x{mask:X};", mask = mask).map_err(write_err)?;
        }
        KeyWidth::I64 => {
            // key = partition_keys[i] (i64)
            writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
            writeln!(ptx, "\tld.global.s64 %rd60, [%rd31];").map_err(write_err)?; // %rd60 = key

            // val = partition_vals[i] (typed)
            writeln!(ptx, "\tmul.wide.u32 %rd32, %r30, {vbpw};").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd33, %rd4, %rd32;").map_err(write_err)?;
            writeln!(ptx, "\t{val_load} {val_reg}, [%rd33];").map_err(write_err)?;

            // slot = low32(key) & mask
            writeln!(ptx, "\tcvt.u32.u64 %r31, %rd60;").map_err(write_err)?;
            writeln!(ptx, "\tand.b32 %r32, %r31, 0x{mask:X};", mask = mask).map_err(write_err)?;
        }
    }
    // Overflow: non-spill drops the row (`LOOP_NEXT`); spill bumps a counter.
    let overflow_label = if spill { "SPILL_BUMP" } else { "LOOP_NEXT" };
    super::partition_reduce_kernel_spill_common::emit_probe_bound_check(
        ptx,
        max_probes,
        overflow_label,
    )?;

    // Addrs. set: ×4, key: ×4 (i32) or ×8 (i64), val: ×vbpw. The key-slot
    // address scratch register order is the only key-width divergence.
    match layout.key_width {
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
            writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?; // addr_val (typed)
        }
    }

    // CAS the set flag.
    super::partition_reduce_kernel_spill_common::emit_slot_claim_cas(ptx, "%rd35")?;

    // 3-state publish protocol (claim-then-write race fix; see
    // partition_reduce_kernel.rs). Spin on a VOLATILE SHARED re-read of set
    // (ld.acquire.cta would default to global and fault on the shared offset)
    // until the claimer publishes set:=2, yielding via nanosleep, THEN read
    // the key. Already factored into spill_common; both key widths share it,
    // differing only in the key dst/probe register and the key-type token.
    let (key_dst_reg, probe_key_reg, key_ty) = match layout.key_width {
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
    // Collision: advance.
    writeln!(ptx, "\tadd.u32 %r32, %r32, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r32, %r32, 0x{mask:X};", mask = mask).map_err(write_err)?;
    if !spill {
        // Occupancy-friendly back-off on the collision-advance path. The spill
        // variant jumps straight back to PROBE_TOP without it.
        super::partition_reduce_kernel_spill_common::emit_spin_backoff(ptx, SPIN_BACKOFF_NS)?;
    }
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    // CLAIM: publish key, fence, then atom.<op> the val.
    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    match layout.key_width {
        KeyWidth::I32 => {
            writeln!(ptx, "\tst.shared.u32 [%rd36], %r31;").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            writeln!(ptx, "\tst.shared.u64 [%rd36], %rd60;").map_err(write_err)?;
        }
    }
    super::partition_reduce_kernel_spill_common::emit_claim_publish(ptx, "%rd35")?;
    writeln!(
        ptx,
        "\t{atom_op} {scratch_reg}, [%rd38], {val_reg};",
        scratch_reg = layout.scratch_reg,
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;
    Ok(())
}

/// Recover the typed `ld.global` mnemonic from the value register width. Both
/// callers already select `val_reg` by dtype, so keying the load off it keeps
/// the load/reg widths in lockstep and reproduces the original `ptx_load()`
/// bytes (`ld.global.s32` for `%r*`, `ld.global.s64` for `%rd*`).
fn layout_load(val_reg: &str) -> &'static str {
    if val_reg.starts_with("%rd") {
        MinMaxDtype::Int64.ptx_load()
    } else {
        MinMaxDtype::Int32.ptx_load()
    }
}

/// Emit Phase 3 (export). The key-width divergences are the key load/store
/// width + the set-offset scratch-register order + the per-key export key
/// register; the spill divergence is the two predicate register numbers, which
/// the caller passes in (`%p5`/`%p6` non-spill; `%p6`/`%p7` spill).
fn emit_export(
    ptx: &mut String,
    layout: &Layout,
    p_export: &str,
    p_set: &str,
) -> BoltResult<()> {
    let block_groups = layout.block_groups;
    let block_threads = layout.block_threads;
    let vbpw = layout.vbpw;
    let export_val_reg = layout.export_val_reg;
    let i64_val = export_val_reg.starts_with("%rd");

    writeln!(ptx, "\tmul.lo.u32 %r44, %r0, {bg};", bg = block_groups).map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r45, %r2;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 {p_export}, %r45, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@{p_export} bra EXPORT_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r46, %r44, %r45;").map_err(write_err)?;

    // Load shared slot's key + typed val + u32 set.
    match layout.key_width {
        KeyWidth::I32 => {
            // i32 key (×4); set reuses the ×4 key offset %rd44.
            writeln!(ptx, "\tmul.wide.u32 %rd44, %r45, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd45, %rd0, %rd44;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.s32 %r47, [%rd45];").map_err(write_err)?;
            writeln!(ptx, "\tmul.wide.u32 %rd46, %r45, {vbpw};").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd47, %rd1, %rd46;").map_err(write_err)?;
            if i64_val {
                writeln!(ptx, "\tld.shared.s64 {export_val_reg}, [%rd47];").map_err(write_err)?;
            } else {
                writeln!(ptx, "\tld.shared.s32 {export_val_reg}, [%rd47];").map_err(write_err)?;
            }
            writeln!(ptx, "\tadd.s64 %rd49, %rd2, %rd44;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.u32 %r49, [%rd49];").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            // i64 key (×8); set needs its own ×4 offset %rd48.
            writeln!(ptx, "\tmul.wide.u32 %rd44, %r45, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd45, %rd0, %rd44;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.s64 %rd62, [%rd45];").map_err(write_err)?;
            writeln!(ptx, "\tmul.wide.u32 %rd46, %r45, {vbpw};").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd47, %rd1, %rd46;").map_err(write_err)?;
            if i64_val {
                writeln!(ptx, "\tld.shared.s64 {export_val_reg}, [%rd47];").map_err(write_err)?;
            } else {
                writeln!(ptx, "\tld.shared.s32 {export_val_reg}, [%rd47];").map_err(write_err)?;
            }
            writeln!(ptx, "\tmul.wide.u32 %rd48, %r45, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd49, %rd2, %rd48;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.u32 %r49, [%rd49];").map_err(write_err)?;
        }
    }
    writeln!(ptx, "\tsetp.ne.s32 {p_set}, %r49, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r50, 1, 0, {p_set};").map_err(write_err)?;

    // Store out_keys[gs] (i32 or i64), out_vals[gs] (typed), out_set[gs] (u8).
    match layout.key_width {
        KeyWidth::I32 => {
            writeln!(ptx, "\tmul.wide.u32 %rd50, %r46, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd51, %rd6, %rd50;").map_err(write_err)?;
            writeln!(ptx, "\tst.global.s32 [%rd51], %r47;").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            writeln!(ptx, "\tmul.wide.u32 %rd50, %r46, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd51, %rd6, %rd50;").map_err(write_err)?;
            writeln!(ptx, "\tst.global.s64 [%rd51], %rd62;").map_err(write_err)?;
        }
    }
    writeln!(ptx, "\tmul.wide.u32 %rd52, %r46, {vbpw};").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd53, %rd7, %rd52;").map_err(write_err)?;
    if i64_val {
        writeln!(ptx, "\tst.global.s64 [%rd53], {export_val_reg};").map_err(write_err)?;
    } else {
        writeln!(ptx, "\tst.global.s32 [%rd53], {export_val_reg};").map_err(write_err)?;
    }
    writeln!(ptx, "\tcvt.u64.u32 %rd54, %r46;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd55, %rd8, %rd54;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd55], %r50;").map_err(write_err)?;

    writeln!(ptx, "\tadd.u32 %r45, %r45, {bt};", bt = block_threads).map_err(write_err)?;
    writeln!(ptx, "\tbra EXPORT_TOP;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_DONE:").map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;
    Ok(())
}

fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!(
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

    /// Byte-stable refactor guard: header / thread-id / spin-back-off
    /// fragments are now emitted via the shared `spill_common` helpers; the
    /// exact bytes must match what was previously inlined here.
    #[test]
    fn shared_fragment_bytes_are_byte_stable() {
        let ptx =
            compile_partition_reduce_kernel_minmax(MinMaxOp::Max, MinMaxDtype::Int64).unwrap();
        assert!(ptx.starts_with(".version 7.5\n.target sm_70\n.address_size 64\n\n"));
        assert!(ptx.contains(
            "\tmov.u32 %r0, %ctaid.x;\n\tmov.u32 %r1, %ntid.x;\n\tmov.u32 %r2, %tid.x;\n"
        ));
        assert!(ptx.contains("\tmov.u32 %nstime, 32;\n\tnanosleep.u32 %nstime;\n"));
    }

    // ----- _with_spill variant shape tests ---------------------------------

    #[test]
    fn with_spill_compiles_all_four_combos() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [MinMaxDtype::Int32, MinMaxDtype::Int64] {
                let ptx = compile_partition_reduce_kernel_minmax_with_spill(op, dt)
                    .unwrap_or_else(|e| panic!("{:?}/{:?} should compile: {e}", op, dt));
                assert!(!ptx.is_empty());
            }
        }
    }

    #[test]
    fn with_spill_has_distinct_entry_names() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [MinMaxDtype::Int32, MinMaxDtype::Int64] {
                let base = kernel_entry(op, dt);
                let spill = kernel_entry_with_spill(op, dt);
                assert_ne!(base, spill);
                assert!(spill.ends_with("_spill"));
                let ptx = compile_partition_reduce_kernel_minmax_with_spill(op, dt).unwrap();
                let needle = format!(".visible .entry {spill}(");
                assert!(ptx.contains(&needle));
            }
        }
    }

    #[test]
    fn with_spill_has_seven_params_and_global_atomic() {
        let ptx = compile_partition_reduce_kernel_minmax_with_spill(
            MinMaxOp::Min,
            MinMaxDtype::Int32,
        )
        .unwrap();
        let count = ptx.matches(".param .u64 ").count();
        assert_eq!(count, 7, "expected 7 .u64 params, got {count}");
        assert!(ptx.contains("atom.global.add.u32"));
        assert!(ptx.contains("SPILL_BUMP:"));
        assert!(ptx.contains("setp.eq.u64"));
    }
}
