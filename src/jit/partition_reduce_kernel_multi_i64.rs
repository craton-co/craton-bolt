// SPDX-License-Identifier: Apache-2.0

//! Per-partition shared-memory reduce kernel — **i64-key, multi-value
//! SUM**. The intersection of `partition_reduce_kernel_multi` (i32-key,
//! N-value) and `partition_reduce_kernel_i64` (i64-key, single-value).
//!
//! Used by the two-key multi-aggregate Tier-2.1 path: pack two i32 keys
//! into an i64 (per `groupby.rs::pack_keys`), partition + scatter
//! through the i64 pipeline, then this kernel reduces each partition
//! into N parallel f64 SUMs keyed by i64.
//!
//! ## Shared-memory layout per block
//!
//! block_keys    : i64 × 1024 =  8 KiB
//! block_vals_0  : f64 × 1024 =  8 KiB
//! block_vals_1  : f64 × 1024 =  8 KiB   (only if n_vals ≥ 2)
//! block_vals_2  : f64 × 1024 =  8 KiB   (only if n_vals ≥ 3)
//! block_vals_3  : f64 × 1024 =  8 KiB   (only if n_vals ≥ 4)
//! block_set     : u32 × 1024 =  4 KiB
//!
//! Totals: N=1 → 20 KiB ; N=2 → 28 KiB ; N=3 → 36 KiB ; N=4 → 44 KiB.
//! All within sm_70's 48 KiB static-shared-mem budget.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

pub const BLOCK_GROUPS: u32 = 1024;
pub const BLOCK_THREADS: u32 = 256;
pub const MAX_VALS: u32 = 4;
pub const NUM_PARTITIONS: u32 = 4096;
const MAX_PROBES: u32 = BLOCK_GROUPS;

/// Per-iteration `nanosleep.u32` operand for the collision-advance path
/// (sm_70+). See `partition_reduce_kernel::SPIN_BACKOFF_NS` for full
/// rationale. TODO(perf): exponential back-off.
const SPIN_BACKOFF_NS: u32 = 32;

pub fn kernel_entry(n_vals: u32) -> String {
    format!("bolt_partition_reduce_multi_sum_i64_{}", n_vals)
}

/// Entry-point name for the spill-counter variant.
pub fn kernel_entry_with_spill(n_vals: u32) -> String {
    format!("{}_spill", kernel_entry(n_vals))
}

/// Register-index layout for the i64-key multi-SUM kernel body. Every index
/// is a pure function of `n_vals` and is computed identically by both the
/// non-spill and `_with_spill` emitters, so it lives in one place. Each field
/// is the bare `%rd<idx>` register number used downstream.
#[derive(Clone, Copy)]
struct RegLayout {
    rd_set: u32,
    rd_pkeys: u32,
    rd_pvals_base: u32,
    rd_poff: u32,
    rd_okeys: u32,
    rd_ovals_base: u32,
    rd_oset: u32,
}

impl RegLayout {
    fn new(n_vals: u32) -> Self {
        let rd_set = 1 + n_vals;
        let rd_pkeys = rd_set + 1;
        let rd_pvals_base = rd_pkeys + 1;
        let rd_poff = rd_pvals_base + n_vals;
        let rd_okeys = rd_poff + 1;
        let rd_ovals_base = rd_okeys + 1;
        let rd_oset = rd_ovals_base + n_vals;
        Self {
            rd_set,
            rd_pkeys,
            rd_pvals_base,
            rd_poff,
            rd_okeys,
            rd_ovals_base,
            rd_oset,
        }
    }
}

/// Emit the shared-memory table declarations (`block_keys`, `n_vals` ×
/// `block_vals`, `block_set`) plus the trailing blank line. `block_keys` is
/// 8-byte aligned here (i64 keys), unlike the i32-key sibling.
///
/// `suffix` is appended to every array name: `""` for the non-spill kernel,
/// `"_sp"` for the `_with_spill` kernel — the only thing that differs between
/// the two emitters here. Byte-for-byte equal to the inline `writeln!`s it
/// replaces.
fn emit_shared_decls(
    ptx: &mut String,
    n_vals: u32,
    suffix: &str,
    keys_bytes: u32,
    vals_bytes: u32,
    set_bytes: u32,
) -> BoltResult<()> {
    writeln!(
        ptx,
        ".shared .align 8 .b8 block_keys_buf{suffix}[{bytes}];",
        bytes = keys_bytes
    )
    .map_err(write_err)?;
    for j in 0..n_vals {
        writeln!(
            ptx,
            ".shared .align 8 .b8 block_vals{j}_buf{suffix}[{bytes}];",
            j = j,
            bytes = vals_bytes
        )
        .map_err(write_err)?;
    }
    writeln!(
        ptx,
        ".shared .align 4 .b8 block_set_buf{suffix}[{bytes}];",
        bytes = set_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;
    Ok(())
}

/// Emit the `.visible .entry` line, the `total_params` `.param .u64` lines
/// (comma-separated, last bare), and the opening `(`/`)`/`{` framing.
///
/// `total_params` is `4 + 2*n_vals` for the base kernel and one more for the
/// spill kernel — passed in so the loop body stays identical.
fn emit_entry_signature(ptx: &mut String, entry: &str, total_params: u32) -> BoltResult<()> {
    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    for p in 0..total_params {
        let trailing = if p == total_params - 1 { "" } else { "," };
        writeln!(ptx, "\t.param .u64 {entry}_param_{p}{trailing}").map_err(write_err)?;
    }
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;
    Ok(())
}

/// Emit the shared-base-address `mov.u64`s: `%rd0 = block_keys`,
/// `%rd1..%rd{n_vals} = block_vals*`, `%rd{1+n_vals} = block_set`. `suffix`
/// selects the non-spill (`""`) vs spill (`"_sp"`) array names.
fn emit_shared_base_addrs(ptx: &mut String, n_vals: u32, suffix: &str) -> BoltResult<()> {
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf{suffix};").map_err(write_err)?;
    for j in 0..n_vals {
        let rd = 1 + j;
        writeln!(ptx, "\tmov.u64 %rd{rd}, block_vals{j}_buf{suffix};").map_err(write_err)?;
    }
    let rd_set = 1 + n_vals;
    writeln!(ptx, "\tmov.u64 %rd{rd_set}, block_set_buf{suffix};").map_err(write_err)?;
    Ok(())
}

/// Emit the global-pointer `ld.param` + `cvta.to.global` pairs shared by both
/// emitters: partition_keys, the `n_vals` partition_vals, partition_offsets,
/// out_keys, the `n_vals` out_vals, and out_set. The spill kernel emits one
/// further pointer (the spill counter) inline after calling this.
fn emit_global_ptr_setup(
    ptx: &mut String,
    entry: &str,
    n_vals: u32,
    layout: &RegLayout,
) -> BoltResult<()> {
    let RegLayout {
        rd_pkeys,
        rd_pvals_base,
        rd_poff,
        rd_okeys,
        rd_ovals_base,
        rd_oset,
        ..
    } = *layout;
    writeln!(ptx, "\tld.param.u64 %rd{rd_pkeys}, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd{rd_pkeys}, %rd{rd_pkeys};").map_err(write_err)?;
    for j in 0..n_vals {
        let rd = rd_pvals_base + j;
        let p = 1 + j;
        writeln!(ptx, "\tld.param.u64 %rd{rd}, [{entry}_param_{p}];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd{rd}, %rd{rd};").map_err(write_err)?;
    }
    let p_off = 1 + n_vals;
    writeln!(ptx, "\tld.param.u64 %rd{rd_poff}, [{entry}_param_{p_off}];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd{rd_poff}, %rd{rd_poff};").map_err(write_err)?;
    let p_ok = 2 + n_vals;
    writeln!(ptx, "\tld.param.u64 %rd{rd_okeys}, [{entry}_param_{p_ok}];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd{rd_okeys}, %rd{rd_okeys};").map_err(write_err)?;
    for j in 0..n_vals {
        let rd = rd_ovals_base + j;
        let p = 3 + n_vals + j;
        writeln!(ptx, "\tld.param.u64 %rd{rd}, [{entry}_param_{p}];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd{rd}, %rd{rd};").map_err(write_err)?;
    }
    let p_os = 3 + 2 * n_vals;
    writeln!(ptx, "\tld.param.u64 %rd{rd_oset}, [{entry}_param_{p_os}];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd{rd_oset}, %rd{rd_oset};").map_err(write_err)?;
    Ok(())
}

/// Emit the per-block partition-slice read (`[start,end)` into `%r10`/`%r11`)
/// using the parametric offsets register and the `%rd80`.. scratch block, then
/// a blank line. Identical in both emitters.
fn emit_slice_read(ptx: &mut String, rd_poff: u32) -> BoltResult<()> {
    writeln!(ptx, "\tmul.wide.u32 %rd80, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd81, %rd{rd_poff}, %rd80;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd81];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd82, %rd81, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd82];").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;
    Ok(())
}

/// Emit Phase 1 — the cooperative shared-memory zero-init loop
/// (`ZERO_TOP`/`ZERO_DONE`), zeroing the i64 key slot, the `n_vals` value
/// slots, and the set slot, ending with `bar.sync 0` + blank line. Identical
/// bytes in both emitters. Keys and vals are i64-wide (`%rd83 = slot*8`); set
/// is u32 (`%rd85 = slot*4`).
fn emit_zero_init(
    ptx: &mut String,
    n_vals: u32,
    layout: &RegLayout,
    block_groups: u32,
    block_threads: u32,
) -> BoltResult<()> {
    let rd_set = layout.rd_set;
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r20, {bg};", bg = block_groups).map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    // block_keys[s] = 0 (i64)
    writeln!(ptx, "\tmul.wide.u32 %rd83, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd84, %rd0, %rd83;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd84], 0;").map_err(write_err)?;
    for j in 0..n_vals {
        let rd_v = 1 + j;
        writeln!(ptx, "\tadd.s64 %rd86, %rd{rd_v}, %rd83;").map_err(write_err)?;
        writeln!(ptx, "\tst.shared.u64 [%rd86], 0;").map_err(write_err)?;
    }
    writeln!(ptx, "\tmul.wide.u32 %rd85, %r20, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd87, %rd{rd_set}, %rd85;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd87], 0;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r20, %r20, {bt};", bt = block_threads).map_err(write_err)?;
    writeln!(ptx, "\tbra ZERO_TOP;").map_err(write_err)?;
    writeln!(ptx, "ZERO_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;
    Ok(())
}

/// Emit the Phase-2 head: the `LOOP_TOP` bound check, the i64 key load, the
/// `n_vals` value loads, the `slot = low32(key) & mask` compute, and the
/// `PROBE_TOP` label through the probe-bound check, ending at the
/// `@%p2 bra {overflow_target};` branch. `overflow_target` is `LOOP_NEXT` for
/// the non-spill kernel (it has no spill block) and `SPILL_BUMP` for the spill
/// kernel — the only divergence in this phase.
fn emit_probe_head(
    ptx: &mut String,
    n_vals: u32,
    layout: &RegLayout,
    mask: u32,
    max_probes: u32,
    overflow_target: &str,
) -> BoltResult<()> {
    let rd_pkeys = layout.rd_pkeys;
    let rd_pvals_base = layout.rd_pvals_base;
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    // key = partition_keys[i] (i64)
    writeln!(ptx, "\tmul.wide.u32 %rd88, %r30, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd89, %rd{rd_pkeys}, %rd88;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rd60, [%rd89];").map_err(write_err)?; // %rd60 = key

    // val_j = partition_vals_j[i]
    for j in 0..n_vals {
        let rd_v = rd_pvals_base + j;
        let fd_v = j;
        writeln!(ptx, "\tadd.s64 %rd91, %rd{rd_v}, %rd88;").map_err(write_err)?;
        writeln!(ptx, "\tld.global.f64 %fd{fd_v}, [%rd91];").map_err(write_err)?;
    }

    // slot = (low_32(key)) & mask
    writeln!(ptx, "\tcvt.u32.u64 %r31, %rd60;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r32, %r31, 0x{mask:X};", mask = mask).map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r33, 0;").map_err(write_err)?;

    writeln!(ptx, "PROBE_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r33, %r33, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.gt.u32 %p2, %r33, {mp};", mp = max_probes).map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra {overflow_target};").map_err(write_err)?;
    Ok(())
}

/// Emit the slot-address compute + `atom.shared.cas.b32` slot claim and the
/// `@%p3 bra CLAIM;` branch. Keys are i64 (×8 at `%rd94`); set is u32 (×4 at
/// `%rd93`). `%rd95 = slot*8` is also computed here for downstream val
/// addressing. Identical bytes in both emitters.
fn emit_slot_cas(ptx: &mut String, layout: &RegLayout) -> BoltResult<()> {
    let rd_set = layout.rd_set;
    writeln!(ptx, "\tmul.wide.u32 %rd92, %r32, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd93, %rd{rd_set}, %rd92;").map_err(write_err)?; // addr_set
    writeln!(ptx, "\tmul.wide.u32 %rd95, %r32, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd94, %rd0, %rd95;").map_err(write_err)?; // addr_key (i64)

    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd93], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;
    Ok(())
}

/// Emit the call into the shared publish/probe protocol (the claim-then-write
/// race fix) with the i64-key register tokens (`key_dst = %rd61`,
/// `probe_key = %rd60`, key-type `s64`). Both emitters pass identical tokens.
fn emit_publish_protocol(ptx: &mut String) -> BoltResult<()> {
    super::partition_reduce_kernel_spill_common::emit_publish_probe_protocol(
        ptx,
        &super::partition_reduce_kernel_spill_common::PublishRegs {
            set_flag_reg: "%r36",
            set_addr_reg: "%rd93",
            key_addr_reg: "%rd94",
            key_dst_reg: "%rd61",
            probe_key_reg: "%rd60",
        },
        "s64",
    )
}

/// Emit the collision-advance `add`/`and` slot bump. `mask` is the slot mask.
/// Shared by both; the non-spill emitter follows this with `emit_spin_backoff`
/// and the spill emitter does not — that divergence stays in the callers.
fn emit_collision_advance(ptx: &mut String, mask: u32) -> BoltResult<()> {
    writeln!(ptx, "\tadd.u32 %r32, %r32, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r32, %r32, 0x{mask:X};", mask = mask).map_err(write_err)?;
    Ok(())
}

/// Emit the N-way `atom.shared.add.f64` accumulate body shared by the CLAIM and
/// MATCH paths: for each value column, `addr = block_vals_j + slot*8` (from
/// `%rd95`) then the shared atomic add. `fd_scratch_base` is the first scratch
/// f64 index (16 for the CLAIM path, 24 for the MATCH path — distinct so the
/// two paths don't alias) — the only thing that differs between the two call
/// sites.
fn emit_accumulate(ptx: &mut String, n_vals: u32, fd_scratch_base: u32) -> BoltResult<()> {
    for j in 0..n_vals {
        let rd_v = 1 + j;
        let fd_v = j;
        writeln!(ptx, "\tadd.s64 %rd96, %rd{rd_v}, %rd95;").map_err(write_err)?;
        let fd_scratch = fd_scratch_base + j;
        writeln!(
            ptx,
            "\tatom.shared.add.f64 %fd{fd_scratch}, [%rd96], %fd{fd_v};"
        )
        .map_err(write_err)?;
    }
    Ok(())
}

/// Emit the `CLAIM:` block: store the i64 key, `membar.cta`, publish `set:=2`,
/// then the N-way accumulate (CLAIM scratch base 16) and `bra LOOP_NEXT;`.
/// Identical bytes in both emitters.
fn emit_claim(ptx: &mut String, n_vals: u32) -> BoltResult<()> {
    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd94], %rd60;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd93], 2;").map_err(write_err)?;
    emit_accumulate(ptx, n_vals, 16)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;
    Ok(())
}

/// Emit Phase 3 — the populated-slot export loop (`EXPORT_TOP`/`EXPORT_DONE`):
/// load each shared slot's i64 key + `n_vals` vals + set, derive the 1-byte set
/// flag, and store key/vals/set to global. `p_ge`/`p_ne` are the two predicate
/// tokens this phase consumes; they differ between the emitters (`%p5`/`%p6`
/// non-spill vs `%p6`/`%p7` spill, because the spill kernel burned `%p5` on the
/// spill-bump null check) — the only divergence here.
fn emit_export(
    ptx: &mut String,
    n_vals: u32,
    layout: &RegLayout,
    block_groups: u32,
    block_threads: u32,
    p_ge: &str,
    p_ne: &str,
) -> BoltResult<()> {
    let RegLayout {
        rd_set,
        rd_okeys,
        rd_ovals_base,
        rd_oset,
        ..
    } = *layout;
    writeln!(ptx, "\tmul.lo.u32 %r40, %r0, {bg};", bg = block_groups).map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r41, %r2;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 {p_ge}, %r41, {bg};", bg = block_groups).map_err(write_err)?;
    writeln!(ptx, "\t@{p_ge} bra EXPORT_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r42, %r40, %r41;").map_err(write_err)?;

    // Load shared slot's i64 key + N vals + set.
    writeln!(ptx, "\tmul.wide.u32 %rd99, %r41, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd98, %rd0, %rd99;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s64 %rd62, [%rd98];").map_err(write_err)?;
    for j in 0..n_vals {
        let rd_v = 1 + j;
        let fd_v = j;
        writeln!(ptx, "\tadd.s64 %rd100, %rd{rd_v}, %rd99;").map_err(write_err)?;
        writeln!(ptx, "\tld.shared.f64 %fd{fd_v}, [%rd100];").map_err(write_err)?;
    }
    writeln!(ptx, "\tmul.wide.u32 %rd97, %r41, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd101, %rd{rd_set}, %rd97;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r44, [%rd101];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 {p_ne}, %r44, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r45, 1, 0, {p_ne};").map_err(write_err)?;

    // Store: i64 key, N f64 vals, u8 set.
    writeln!(ptx, "\tmul.wide.u32 %rd104, %r42, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd103, %rd{rd_okeys}, %rd104;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s64 [%rd103], %rd62;").map_err(write_err)?;
    for j in 0..n_vals {
        let rd_ov = rd_ovals_base + j;
        let fd_v = j;
        writeln!(ptx, "\tadd.s64 %rd105, %rd{rd_ov}, %rd104;").map_err(write_err)?;
        writeln!(ptx, "\tst.global.f64 [%rd105], %fd{fd_v};").map_err(write_err)?;
    }
    writeln!(ptx, "\tcvt.u64.u32 %rd106, %r42;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd107, %rd{rd_oset}, %rd106;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd107], %r45;").map_err(write_err)?;

    writeln!(ptx, "\tadd.u32 %r41, %r41, {bt};", bt = block_threads).map_err(write_err)?;
    writeln!(ptx, "\tbra EXPORT_TOP;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_DONE:").map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;
    Ok(())
}

pub fn compile_partition_reduce_kernel_multi_i64(n_vals: u32) -> BoltResult<String> {
    if n_vals == 0 || n_vals > MAX_VALS {
        return Err(BoltError::Other(format!(
            "partition_reduce_kernel_multi_i64: n_vals must be 1..={MAX_VALS}, got {n_vals}"
        )));
    }
    let mut ptx = String::new();
    let entry = kernel_entry(n_vals);
    let entry = entry.as_str();
    let block_groups = BLOCK_GROUPS;
    let mask = BLOCK_GROUPS - 1;
    let block_threads = BLOCK_THREADS;
    let keys_bytes = BLOCK_GROUPS * 8; // i64 keys
    let vals_bytes = BLOCK_GROUPS * 8;
    let set_bytes = BLOCK_GROUPS * 4;
    let max_probes = MAX_PROBES;
    let layout = RegLayout::new(n_vals);

    super::partition_reduce_kernel_spill_common::emit_ptx_header(&mut ptx)?;

    // Shared-memory tables: block_keys (i64, .align 8) + n_vals × block_vals
    // (f64) + block_set (u32). No suffix on the base kernel's array names.
    emit_shared_decls(&mut ptx, n_vals, "", keys_bytes, vals_bytes, set_bytes)?;

    // Kernel signature: keys + N val ptrs + offsets + out_keys + N out_val ptrs
    // + out_set = 4 + 2N.
    emit_entry_signature(&mut ptx, entry, 4 + 2 * n_vals)?;

    writeln!(ptx, "\t.reg .pred  %p<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<96>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<128>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<32>;").map_err(write_err)?;
    // Operand register for the per-collision `nanosleep.u32` back-off. Present
    // only in the non-spill kernel — its collision path runs the spin back-off,
    // which the spill kernel drops, so the spill emitter omits this `.reg` line.
    writeln!(ptx, "\t.reg .u32   %nstime;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    super::partition_reduce_kernel_spill_common::emit_thread_block_ids(&mut ptx)?;

    // Shared bases: %rd0=keys, %rd1..%rd{n_vals}=vals_j, %rd{n_vals+1}=set
    emit_shared_base_addrs(&mut ptx, n_vals, "")?;

    // Global pointer setup. (The spill kernel emits one further pointer after
    // this.)
    emit_global_ptr_setup(&mut ptx, entry, n_vals, &layout)?;
    writeln!(ptx).map_err(write_err)?;

    // Read partition slice [start, end).
    emit_slice_read(&mut ptx, layout.rd_poff)?;

    // ----------------- Phase 1: zero shared. ------------------------------
    emit_zero_init(&mut ptx, n_vals, &layout, block_groups, block_threads)?;

    // ----------------- Phase 2: probe + N-way sum (i64 keys). -------------
    // Non-spill: probe-bound overflow jumps straight to LOOP_NEXT (no spill
    // block to bump a counter).
    emit_probe_head(&mut ptx, n_vals, &layout, mask, max_probes, "LOOP_NEXT")?;

    // Slot addresses + CAS the slot flag.
    emit_slot_cas(&mut ptx, &layout)?;

    // Slot occupied — membar.cta orders the set CAS against the i64
    // key load (different addresses). PTX sm_70 has no inter-address
    // ordering; without this fence a racing thread can see set==1 with
    // a zero key and false-match.
    // 3-state publish protocol (claim-then-write race fix; set u32 at %rd93,
    // key i64 at %rd94). VOLATILE SHARED re-read of set + nanosleep yield
    // until the claimer publishes set:=2, THEN read the i64 key.
    emit_publish_protocol(&mut ptx)?;
    // Collision: advance.
    emit_collision_advance(&mut ptx, mask)?;
    // Occupancy-friendly back-off on the collision-advance path (non-spill only).
    super::partition_reduce_kernel_spill_common::emit_spin_backoff(&mut ptx, SPIN_BACKOFF_NS)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    // CLAIM: won the slot (CAS set 0->1). Store key, fence, publish set:=2,
    // then sum N vals.
    emit_claim(&mut ptx, n_vals)?;

    // MATCH: slot already holds our key — just sum. Non-spill falls through to
    // the immediately-following LOOP_NEXT label (no trailing `bra`).
    writeln!(ptx, "MATCH:").map_err(write_err)?;
    emit_accumulate(&mut ptx, n_vals, 24)?;

    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ----------------- Phase 3: export populated slots. -------------------
    // Non-spill predicates: %p5 (bound) / %p6 (set-flag). The spill kernel
    // shifts these to %p6 / %p7 (it burned %p5 on the spill-bump null check).
    emit_export(
        &mut ptx,
        n_vals,
        &layout,
        block_groups,
        block_threads,
        "%p5",
        "%p6",
    )?;

    Ok(ptx)
}

/// Spill-counter-aware sibling of
/// [`compile_partition_reduce_kernel_multi_i64`]. Trailing
/// `.param .u64 spill_counter` (uint32_t*, may be null). On MAX_PROBES
/// overflow null-checks then `atom.global.add.u32` it.
pub fn compile_partition_reduce_kernel_multi_i64_with_spill(n_vals: u32) -> BoltResult<String> {
    if n_vals == 0 || n_vals > MAX_VALS {
        return Err(BoltError::Other(format!(
            "partition_reduce_kernel_multi_i64_with_spill: n_vals must be 1..={MAX_VALS}, got {n_vals}"
        )));
    }
    let mut ptx = String::new();
    let entry = kernel_entry_with_spill(n_vals);
    let entry = entry.as_str();
    let block_groups = BLOCK_GROUPS;
    let mask = BLOCK_GROUPS - 1;
    let block_threads = BLOCK_THREADS;
    let keys_bytes = BLOCK_GROUPS * 8;
    let vals_bytes = BLOCK_GROUPS * 8;
    let set_bytes = BLOCK_GROUPS * 4;
    let max_probes = MAX_PROBES;
    let layout = RegLayout::new(n_vals);
    // The spill counter pointer lives one register past out_set.
    let rd_spill = layout.rd_oset + 1;

    super::partition_reduce_kernel_spill_common::emit_ptx_header(&mut ptx)?;

    // Shared tables with the `_sp` array-name suffix (the only thing that
    // differs from the base kernel's declarations).
    emit_shared_decls(&mut ptx, n_vals, "_sp", keys_bytes, vals_bytes, set_bytes)?;

    // 4 + 2*n_vals base params + 1 spill_counter trailing.
    emit_entry_signature(&mut ptx, entry, 4 + 2 * n_vals + 1)?;

    // Register pool — note: NO `%nstime` register. The spill kernel's collision
    // path skips the spin back-off, so it never needs the nanosleep operand.
    writeln!(ptx, "\t.reg .pred  %p<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<96>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<128>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<32>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    super::partition_reduce_kernel_spill_common::emit_thread_block_ids(&mut ptx)?;

    emit_shared_base_addrs(&mut ptx, n_vals, "_sp")?;

    // Shared global-pointer setup, then the spill-counter pointer (the one
    // extra param this variant carries).
    emit_global_ptr_setup(&mut ptx, entry, n_vals, &layout)?;
    let p_sp = 4 + 2 * n_vals;
    writeln!(ptx, "\tld.param.u64 %rd{rd_spill}, [{entry}_param_{p_sp}];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd{rd_spill}, %rd{rd_spill};").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    emit_slice_read(&mut ptx, layout.rd_poff)?;

    // Phase 1: zero.
    emit_zero_init(&mut ptx, n_vals, &layout, block_groups, block_threads)?;

    // Phase 2. Spill: probe-bound overflow jumps to SPILL_BUMP (bumps the
    // overflow counter) instead of straight to LOOP_NEXT.
    emit_probe_head(&mut ptx, n_vals, &layout, mask, max_probes, "SPILL_BUMP")?;

    emit_slot_cas(&mut ptx, &layout)?;

    // 3-state publish protocol (claim-then-write race fix).
    emit_publish_protocol(&mut ptx)?;
    // Collision: advance. Spill kernel drops the spin back-off here, so it
    // jumps straight back to PROBE_TOP (no `emit_spin_backoff`).
    emit_collision_advance(&mut ptx, mask)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    // CLAIM: won the slot. Store key, fence, publish set:=2, then sum N vals.
    emit_claim(&mut ptx, n_vals)?;

    // MATCH: spill variant has the SPILL_BUMP block between MATCH and LOOP_NEXT,
    // so it needs an explicit `bra LOOP_NEXT;` here (the base kernel falls
    // through instead).
    writeln!(ptx, "MATCH:").map_err(write_err)?;
    emit_accumulate(&mut ptx, n_vals, 24)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    super::partition_reduce_kernel_spill_common::emit_spill_bump_with_null_check(
        &mut ptx, rd_spill,
    )?;

    super::partition_reduce_kernel_spill_common::emit_loop_next_done(&mut ptx)?;

    // Phase 3. Spill predicates are %p6 (bound) / %p7 (set-flag) — shifted up
    // one from the base kernel because %p5 was consumed by the spill-bump
    // null check.
    emit_export(
        &mut ptx,
        n_vals,
        &layout,
        block_groups,
        block_threads,
        "%p6",
        "%p7",
    )?;

    Ok(ptx)
}

fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!(
        "partition_reduce_kernel_multi_i64: write failed: {}",
        e
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_for_all_n_vals() {
        for n in 1..=MAX_VALS {
            let ptx = compile_partition_reduce_kernel_multi_i64(n)
                .unwrap_or_else(|e| panic!("n_vals={n} should compile: {e}"));
            assert!(!ptx.is_empty());
        }
    }

    #[test]
    fn rejects_bad_n_vals() {
        assert!(compile_partition_reduce_kernel_multi_i64(0).is_err());
        assert!(compile_partition_reduce_kernel_multi_i64(MAX_VALS + 1).is_err());
    }

    #[test]
    fn uses_i64_key_loads_and_stores() {
        let ptx = compile_partition_reduce_kernel_multi_i64(2).unwrap();
        assert!(ptx.contains("ld.global.s64"));
        assert!(ptx.contains("st.global.s64"));
        assert!(ptx.contains("ld.shared.s64"));
        assert!(ptx.contains("st.shared.u64"));
    }

    #[test]
    fn emits_n_atomic_adds_per_path() {
        // CLAIM + MATCH each issue n_vals atom.shared.add.f64.
        for n in 1..=MAX_VALS {
            let ptx = compile_partition_reduce_kernel_multi_i64(n).unwrap();
            let c = ptx.matches("atom.shared.add.f64").count();
            assert_eq!(c, (2 * n) as usize, "n={n}: want {} atomics", 2 * n);
        }
    }

    // ----- _with_spill variant shape tests ---------------------------------

    #[test]
    fn with_spill_compiles_for_all_n_vals() {
        for n in 1..=MAX_VALS {
            let ptx = compile_partition_reduce_kernel_multi_i64_with_spill(n)
                .unwrap_or_else(|e| panic!("n_vals={n} should compile: {e}"));
            assert!(!ptx.is_empty());
        }
    }

    #[test]
    fn with_spill_rejects_bad_n_vals() {
        assert!(compile_partition_reduce_kernel_multi_i64_with_spill(0).is_err());
        assert!(compile_partition_reduce_kernel_multi_i64_with_spill(MAX_VALS + 1).is_err());
    }

    #[test]
    fn with_spill_has_extra_param_and_spill_atomic() {
        for n in 1..=MAX_VALS {
            let ptx = compile_partition_reduce_kernel_multi_i64_with_spill(n).unwrap();
            let expected = 4 + 2 * n as usize + 1;
            let c = ptx.matches(".param .u64 ").count();
            assert_eq!(c, expected, "n_vals={n}");
            assert!(ptx.contains("atom.global.add.u32"));
            assert!(ptx.contains("SPILL_BUMP:"));
            assert!(ptx.contains("setp.eq.u64"));
            let entry = kernel_entry_with_spill(n);
            assert!(ptx.contains(&format!(".visible .entry {entry}(")));
        }
    }
}
