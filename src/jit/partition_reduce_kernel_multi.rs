// SPDX-License-Identifier: Apache-2.0

//! Per-partition shared-memory GROUP BY SUM kernel — **multi-value
//! variant** (Tier 2.1 for multi-aggregate workloads).
//!
//! Sibling of [`crate::jit::partition_reduce_kernel`] (single i32 key,
//! single f64 value) and [`crate::jit::partition_reduce_kernel_i64`]
//! (single i64 key, single f64 value). This file emits PTX for the
//! **N-value** generalization: one i32 key column plus `n_vals` f64
//! value columns, all reduced into per-partition open-addressing tables
//! in one launch.
//!
//! ## Why this exists
//!
//! Multi-SUM Tier-2 (`groupby_tier2_multi_orchestrator.rs`) used to do
//! pass-2 on the host: download the scatter buffers, build N=4096 small
//! `HashMap<i32, [f64; N]>`s, push the result. That worked but couldn't
//! win below ~100 K groups — the gate `MULTI_SUM_MIN_GROUPS` in
//! `groupby_tier2_multi_exec.rs` exists to keep workloads like q2
//! (10 K groups) from hitting the slow path.
//!
//! This kernel replaces that host loop with one GPU launch — one block
//! per partition, N parallel f64 accumulators per slot. With pass-2
//! moved to the GPU, the fixed multi-SUM Tier-2 overhead drops enough
//! that the gate can be relaxed back to Tier-1's cap (`> BLOCK_GROUPS`).
//!
//! ## Shared-memory layout
//!
//! Per block:
//!   block_keys     : i32   × 1024 =  4 KiB
//!   block_vals_0   : f64   × 1024 =  8 KiB
//!   block_vals_1   : f64   × 1024 =  8 KiB   (only if n_vals ≥ 2)
//!   block_vals_2   : f64   × 1024 =  8 KiB   (only if n_vals ≥ 3)
//!   block_vals_3   : f64   × 1024 =  8 KiB   (only if n_vals ≥ 4)
//!   block_set      : u32   × 1024 =  4 KiB
//!
//! Totals: N=1 → 16 KiB ; N=2 → 24 KiB ; N=3 → 32 KiB ; N=4 → 40 KiB.
//! All comfortably under sm_70's 48 KiB static-shared-mem budget.
//!
//! ## Algorithm
//!
//! Identical to the single-value variant, just with N parallel atomic
//! adds per row instead of 1. The probe slot is determined by the key
//! alone (low bits, masked); on claim or match, each thread issues
//! `n_vals` `atom.shared.add.f64` instructions in sequence, one per
//! value column. The shared-memory atomic unit handles them
//! independently, so the per-row cost grows roughly linearly in N (no
//! hidden quadratic).
//!
//! ## PTX size
//!
//! The kernel text grows with N because we emit N val-loads, N atomic
//! adds, and N export-stores. For N=4 the PTX is ~600 lines vs ~430 for
//! N=1. All emitted at compile time and cached by the existing PTX
//! cache in `jit_compiler.rs`.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

/// Open-addressing slot count per block. Must match
/// [`crate::jit::partition_reduce_kernel::BLOCK_GROUPS`].
pub const BLOCK_GROUPS: u32 = 1024;

/// Threads per block. Matches the single-value sibling.
pub const BLOCK_THREADS: u32 = 256;

/// Maximum value columns supported. Mirrors `shmem_multi_sum_kernel::MAX_VALS`
/// so the multi-SUM eligibility check upstream is consistent.
pub const MAX_VALS: u32 = 4;

/// Number of partitions launched per query. Must match
/// [`crate::jit::partition_kernel::NUM_PARTITIONS`] (4096 after Tier-2.1
/// tuning).
pub const NUM_PARTITIONS: u32 = 4096;

/// Probe-chain bound — same v0 policy as the single-value variant.
const MAX_PROBES: u32 = BLOCK_GROUPS;

/// Per-iteration `nanosleep.u32` operand for the collision-advance path
/// (sm_70+). See `partition_reduce_kernel::SPIN_BACKOFF_NS` for full
/// rationale. TODO(perf): exponential back-off.
const SPIN_BACKOFF_NS: u32 = 32;

/// Entry-point name for the emitted PTX. Includes `n_vals` so each variant
/// is cache-distinct in the PTX cache.
pub fn kernel_entry(n_vals: u32) -> String {
    format!("bolt_partition_reduce_multi_sum_{}", n_vals)
}

/// Generate PTX for the multi-value per-partition reduce kernel.
///
/// `n_vals` must be in `1..=MAX_VALS` (1..=4). The emitted kernel has
/// `4 + 2*n_vals` pointer parameters in this order:
///
/// ```text
/// .param .u64 partition_keys
/// .param .u64 partition_vals_0   ..   partition_vals_{n_vals-1}
/// .param .u64 partition_offsets
/// .param .u64 out_keys
/// .param .u64 out_vals_0   ..   out_vals_{n_vals-1}
/// .param .u64 out_set
/// ```
///
/// Launch geometry: `grid = NUM_PARTITIONS`, `block = BLOCK_THREADS`.
pub fn compile_partition_reduce_kernel_multi(n_vals: u32) -> BoltResult<String> {
    if n_vals == 0 || n_vals > MAX_VALS {
        return Err(BoltError::Other(format!(
            "partition_reduce_kernel_multi: n_vals must be 1..={MAX_VALS}, got {n_vals}"
        )));
    }
    let mut ptx = String::new();
    let entry = kernel_entry(n_vals);
    let entry = entry.as_str();
    let block_groups = BLOCK_GROUPS;
    let mask = BLOCK_GROUPS - 1;
    let block_threads = BLOCK_THREADS;
    let keys_bytes = BLOCK_GROUPS * 4;
    let vals_bytes = BLOCK_GROUPS * 8;
    let set_bytes = BLOCK_GROUPS * 4;
    let max_probes = MAX_PROBES;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Shared-memory tables: block_keys (i32) + n_vals × block_vals (f64) +
    // block_set (u32). One declaration per array.
    writeln!(
        ptx,
        ".shared .align 4 .b8 block_keys_buf[{bytes}];",
        bytes = keys_bytes
    )
    .map_err(write_err)?;
    for j in 0..n_vals {
        writeln!(
            ptx,
            ".shared .align 8 .b8 block_vals{j}_buf[{bytes}];",
            j = j,
            bytes = vals_bytes
        )
        .map_err(write_err)?;
    }
    writeln!(
        ptx,
        ".shared .align 4 .b8 block_set_buf[{bytes}];",
        bytes = set_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Kernel signature. Param count = 4 + 2*n_vals.
    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    let total_params = 4 + 2 * n_vals;
    for p in 0..total_params {
        let trailing = if p == total_params - 1 { "" } else { "," };
        writeln!(ptx, "\t.param .u64 {entry}_param_{p}{trailing}").map_err(write_err)?;
    }
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Register pool — extra-wide to give us room for N parallel val streams.
    // The single-value variant's pool was %r<64> / %rd<64>; we widen both
    // here so per-val temporaries (%rd_addr_val_j, %fd_val_j) don't clash.
    writeln!(ptx, "\t.reg .pred  %p<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<96>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<128>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<32>;").map_err(write_err)?;
    // Operand register for the per-collision `nanosleep.u32` back-off.
    writeln!(ptx, "\t.reg .u32   %nstime;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Thread coordinates.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;

    // Shared base addresses:
    //   %rd0 = block_keys_buf
    //   %rd1..%rd{n_vals} = block_vals0_buf .. block_vals{n_vals-1}_buf
    //   %rd_set = block_set_buf  (we'll put it at %rd{n_vals+1})
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf;").map_err(write_err)?;
    for j in 0..n_vals {
        let rd = 1 + j;
        writeln!(ptx, "\tmov.u64 %rd{rd}, block_vals{j}_buf;").map_err(write_err)?;
    }
    let rd_set = 1 + n_vals;
    writeln!(ptx, "\tmov.u64 %rd{rd_set}, block_set_buf;").map_err(write_err)?;

    // Global pointer setup. `partition_keys` is at param_0, then n_vals
    // value buffers, then offsets, then out_keys, then n_vals out_vals,
    // then out_set.
    //
    // %rd_pkeys      = partition_keys      (param_0)
    // %rd_pvals[j]   = partition_vals_j    (param_{1+j})
    // %rd_poff       = partition_offsets   (param_{1+n_vals})
    // %rd_okeys      = out_keys            (param_{2+n_vals})
    // %rd_ovals[j]   = out_vals_j          (param_{3+n_vals+j})
    // %rd_oset       = out_set             (param_{3+2*n_vals})
    let rd_pkeys = rd_set + 1; // first free register after shared bases
    let rd_pvals_base = rd_pkeys + 1; // pvals[j] = %rd{rd_pvals_base + j}
    let rd_poff = rd_pvals_base + n_vals;
    let rd_okeys = rd_poff + 1;
    let rd_ovals_base = rd_okeys + 1;
    let rd_oset = rd_ovals_base + n_vals;

    writeln!(
        ptx,
        "\tld.param.u64 %rd{rd_pkeys}, [{entry}_param_0];"
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tcvta.to.global.u64 %rd{rd_pkeys}, %rd{rd_pkeys};"
    )
    .map_err(write_err)?;
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
    writeln!(ptx).map_err(write_err)?;

    // Read this block's partition slice [start, end).
    writeln!(ptx, "\tmul.wide.u32 %rd80, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd81, %rd{rd_poff}, %rd80;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd81];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd82, %rd81, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd82];").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ----------------- Phase 1: cooperative shared-mem zero ---------------
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p0, %r20, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    // block_keys[s] = 0  (i32, 4 B)
    writeln!(ptx, "\tmul.wide.u32 %rd83, %r20, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd84, %rd0, %rd83;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd84], 0;").map_err(write_err)?;
    // block_vals_j[s] = 0.0  (f64, 8 B)
    writeln!(ptx, "\tmul.wide.u32 %rd85, %r20, 8;").map_err(write_err)?;
    for j in 0..n_vals {
        let rd_v = 1 + j;
        writeln!(ptx, "\tadd.s64 %rd86, %rd{rd_v}, %rd85;").map_err(write_err)?;
        writeln!(ptx, "\tst.shared.u64 [%rd86], 0;").map_err(write_err)?;
    }
    // block_set[s] = 0
    writeln!(ptx, "\tadd.s64 %rd87, %rd{rd_set}, %rd83;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd87], 0;").map_err(write_err)?;
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

    // ----------------- Phase 2: probe + N-way sum -------------------------
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    // key = partition_keys[i]
    writeln!(ptx, "\tmul.wide.u32 %rd88, %r30, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd89, %rd{rd_pkeys}, %rd88;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s32 %r31, [%rd89];").map_err(write_err)?; // key

    // val_j = partition_vals_j[i]   (per-j load — %fd{j} holds val_j)
    writeln!(ptx, "\tmul.wide.u32 %rd90, %r30, 8;").map_err(write_err)?;
    for j in 0..n_vals {
        let rd_v = rd_pvals_base + j;
        let fd_v = j; // %fd0 .. %fd{n_vals-1}
        writeln!(ptx, "\tadd.s64 %rd91, %rd{rd_v}, %rd90;").map_err(write_err)?;
        writeln!(ptx, "\tld.global.f64 %fd{fd_v}, [%rd91];").map_err(write_err)?;
    }

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

    // Slot addresses (set: ×4, key: ×4, val_j: ×8 each).
    writeln!(ptx, "\tmul.wide.u32 %rd92, %r32, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd93, %rd{rd_set}, %rd92;").map_err(write_err)?; // addr_set
    writeln!(ptx, "\tadd.s64 %rd94, %rd0, %rd92;").map_err(write_err)?; // addr_key
    writeln!(ptx, "\tmul.wide.u32 %rd95, %r32, 8;").map_err(write_err)?;
    // Per-j val addresses: %rd96 reused as scratch then stored into %rd_addr_val_j
    // We compute them lazily after the CAS branches so we don't waste regs on
    // the collision path. (Done below.)

    // CAS the slot flag.
    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd93], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;

    // Else: slot occupied — membar.cta orders the CAS (block_set)
    // against the key load (block_keys, different address). PTX sm_70
    // requires this fence; without it a racing thread can read a zero
    // key under set==1 and false-match key 0.
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s32 %r35, [%rd94];").map_err(write_err)?;
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
    // Occupancy-friendly back-off on the collision-advance path.
    writeln!(
        ptx,
        "\tmov.u32 %nstime, {ns};",
        ns = SPIN_BACKOFF_NS
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tnanosleep.u32 %nstime;").map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    // CLAIM: this thread won the slot. Write the key, fence, then sum
    // N vals.
    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd94], %r31;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    for j in 0..n_vals {
        let rd_v = 1 + j;
        let fd_v = j;
        // addr_val_j = block_vals_j + slot*8 = %rd_v + %rd95
        writeln!(ptx, "\tadd.s64 %rd96, %rd{rd_v}, %rd95;").map_err(write_err)?;
        // atom.shared.add.f64 writes the sum into a scratch f64.
        // We reuse %fd{16 + j} as the scratch — distinct from %fd{j} (the input).
        let fd_scratch = 16 + j;
        writeln!(
            ptx,
            "\tatom.shared.add.f64 %fd{fd_scratch}, [%rd96], %fd{fd_v};"
        )
        .map_err(write_err)?;
    }
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    // MATCH: slot already holds our key — just sum.
    writeln!(ptx, "MATCH:").map_err(write_err)?;
    for j in 0..n_vals {
        let rd_v = 1 + j;
        let fd_v = j;
        writeln!(ptx, "\tadd.s64 %rd96, %rd{rd_v}, %rd95;").map_err(write_err)?;
        let fd_scratch = 24 + j;
        writeln!(
            ptx,
            "\tatom.shared.add.f64 %fd{fd_scratch}, [%rd96], %fd{fd_v};"
        )
        .map_err(write_err)?;
    }

    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ----------------- Phase 3: export populated slots --------------------
    writeln!(
        ptx,
        "\tmul.lo.u32 %r40, %r0, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r41, %r2;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p5, %r41, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra EXPORT_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r42, %r40, %r41;").map_err(write_err)?;

    // Load shared slot's key + N vals + set.
    writeln!(ptx, "\tmul.wide.u32 %rd97, %r41, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd98, %rd0, %rd97;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s32 %r43, [%rd98];").map_err(write_err)?; // key
    writeln!(ptx, "\tmul.wide.u32 %rd99, %r41, 8;").map_err(write_err)?;
    for j in 0..n_vals {
        let rd_v = 1 + j;
        let fd_v = j; // reuse low fd regs for slot vals
        writeln!(ptx, "\tadd.s64 %rd100, %rd{rd_v}, %rd99;").map_err(write_err)?;
        writeln!(ptx, "\tld.shared.f64 %fd{fd_v}, [%rd100];").map_err(write_err)?;
    }
    writeln!(ptx, "\tadd.s64 %rd101, %rd{rd_set}, %rd97;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r44, [%rd101];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p6, %r44, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r45, 1, 0, %p6;").map_err(write_err)?;

    // Store key (4 B), vals (8 B each), set (1 B).
    writeln!(ptx, "\tmul.wide.u32 %rd102, %r42, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd103, %rd{rd_okeys}, %rd102;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s32 [%rd103], %r43;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd104, %r42, 8;").map_err(write_err)?;
    for j in 0..n_vals {
        let rd_ov = rd_ovals_base + j;
        let fd_v = j;
        writeln!(ptx, "\tadd.s64 %rd105, %rd{rd_ov}, %rd104;").map_err(write_err)?;
        writeln!(ptx, "\tst.global.f64 [%rd105], %fd{fd_v};").map_err(write_err)?;
    }
    writeln!(ptx, "\tcvt.u64.u32 %rd106, %r42;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd107, %rd{rd_oset}, %rd106;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd107], %r45;").map_err(write_err)?;

    writeln!(
        ptx,
        "\tadd.u32 %r41, %r41, {bt};",
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
        "partition_reduce_kernel_multi: write failed: {}",
        e
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_for_all_n_vals() {
        for n in 1..=MAX_VALS {
            let ptx = compile_partition_reduce_kernel_multi(n)
                .unwrap_or_else(|e| panic!("n_vals={n} should compile but: {e}"));
            assert!(!ptx.is_empty(), "n_vals={n} produced empty PTX");
        }
    }

    #[test]
    fn rejects_zero_and_overflow() {
        assert!(compile_partition_reduce_kernel_multi(0).is_err());
        assert!(compile_partition_reduce_kernel_multi(MAX_VALS + 1).is_err());
    }

    #[test]
    fn distinct_entry_per_n_vals() {
        for n in 1..=MAX_VALS {
            let ptx = compile_partition_reduce_kernel_multi(n).unwrap();
            let want = kernel_entry(n);
            assert!(
                ptx.contains(&format!(".visible .entry {want}(")),
                "n_vals={n}: entry-point name missing from emitted PTX"
            );
        }
    }

    #[test]
    fn emits_n_shared_val_arrays() {
        for n in 1..=MAX_VALS {
            let ptx = compile_partition_reduce_kernel_multi(n).unwrap();
            for j in 0..n {
                assert!(
                    ptx.contains(&format!("block_vals{j}_buf")),
                    "n_vals={n}: missing block_vals{j}_buf declaration"
                );
            }
            // And NO extras for unused j.
            for j in n..MAX_VALS {
                assert!(
                    !ptx.contains(&format!("block_vals{j}_buf")),
                    "n_vals={n}: stray block_vals{j}_buf declaration"
                );
            }
        }
    }

    #[test]
    fn emits_n_atomic_adds_in_claim_path() {
        // The CLAIM path issues exactly n_vals atom.shared.add.f64 instructions.
        // Counting all atom.shared.add.f64 in the file gives 2*n_vals (CLAIM
        // + MATCH paths) — verify that linear scaling.
        for n in 1..=MAX_VALS {
            let ptx = compile_partition_reduce_kernel_multi(n).unwrap();
            let count = ptx.matches("atom.shared.add.f64").count();
            assert_eq!(
                count,
                (2 * n) as usize,
                "n_vals={n}: expected {} atom.shared.add.f64 (CLAIM + MATCH × n_vals), got {count}",
                2 * n,
            );
        }
    }

    #[test]
    fn uses_atom_shared_cas_for_slot_claim() {
        let ptx = compile_partition_reduce_kernel_multi(2).unwrap();
        assert!(
            ptx.contains("atom.shared.cas.b32"),
            "atom.shared.cas.b32 (slot-claim) missing"
        );
    }

    #[test]
    fn has_two_barriers() {
        let ptx = compile_partition_reduce_kernel_multi(2).unwrap();
        assert!(
            ptx.matches("bar.sync 0").count() >= 2,
            "expected ≥2 bar.sync 0"
        );
    }

    #[test]
    fn param_count_matches_signature() {
        // 4 + 2*n_vals .param .u64 lines should appear inside the entry().
        for n in 1..=MAX_VALS {
            let ptx = compile_partition_reduce_kernel_multi(n).unwrap();
            let expected = 4 + 2 * n as usize;
            let count = ptx.matches(".param .u64 ").count();
            assert_eq!(
                count, expected,
                "n_vals={n}: expected {expected} .param .u64 lines, got {count}"
            );
        }
    }
}
