// SPDX-License-Identifier: Apache-2.0

//! Per-partition shared-memory GROUP BY SUM kernel — **Int64-key variant**
//! (Tier 2.1 for two-key GROUP BY).
//!
//! Sibling of [`crate::jit::partition_reduce_kernel`]. The Int32 sibling
//! handles single-Int32-key Tier-2 (q5); this file handles the two-Int32-
//! keys-packed-into-Int64 case used by the two-key Tier-2 path (q3).
//!
//! ## Why this exists
//!
//! Before this kernel landed, `groupby_tier2_twokey_orchestrator.rs`
//! reduced its scattered partitions on the **host** (download both keys
//! and vals, build N=4096 small `HashMap<i64, f64>`s, push the result).
//! That cost roughly the same as host-pass-2 did for the i32 path:
//! ~150 ms of D2H + HashMap work on a 10 M-row workload. Measured: q3
//! went from 807 ms baseline → 953 ms with twokey-Tier-2 enabled (host
//! pass-2), so the two-key path was disabled at integration time.
//!
//! This kernel is the i64-key analog of the i32 per-partition reduce
//! kernel that already won 3.7× on q5. Wiring it into the twokey
//! orchestrator restores Tier-2's structural advantage at the i64 key
//! width.
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
//! | Output buffer / slot  | 4 B                    | **8 B**                 |
//! | Output buffer total   | 4 MiB (4096×1024×4 B)  | **8 MiB** (×8 B)        |
//! | Total per-block shmem | 16 KiB                 | **20 KiB**              |
//!
//! 20 KiB per block is comfortably under the 48 KiB sm_70 static budget.
//! Total output footprint is 8 + 8 + 4 = 20 B × 4096 × 1024 = 80 MiB
//! (vs 52 MiB for i32). The extra 28 MiB D2H is ~3 ms at PCIe Gen3 x16 —
//! negligible relative to the wins we expect.
//!
//! ## Slot mapping
//!
//! The partition kernel hashes i64 keys using Knuth's 64-bit Fibonacci
//! multiplier and takes the HIGH log₂(K) bits to select the partition.
//! Inside a partition the keys are "random" w.r.t. that selection, so we
//! can route them to shared-table slots using the **low 32 bits** of the
//! key as a direct slot index (`& (BLOCK_GROUPS - 1)`). For h2o.ai q3
//! (two i32 keys packed as `(id1 << 32) | id2`) this means the slot is
//! dictated by `id2 & 0x3FF` — id2 ranges 0..999 and is dense, giving an
//! evenly-distributed slot map.
//!
//! If a future workload arrives where the low 32 bits cluster, we can
//! drop in a `mul.lo.u32 %r_slot, %r_low_key, KNUTH_MULTIPLIER` between
//! the cast and the mask. That's a 2-line PTX change.
//!
//! ## Probe / collision / drop policy
//!
//! Identical to the i32 variant: open-addressing linear probe bounded
//! by `MAX_PROBES = BLOCK_GROUPS`, drop on overflow (deliberate v0).

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

/// Number of slots in each block's shared-memory open-addressing table.
/// Must match [`crate::jit::partition_reduce_kernel::BLOCK_GROUPS`] so the
/// orchestrator can share the launch-geometry / output-sizing assumptions.
pub const BLOCK_GROUPS: u32 = 1024;

/// Threads per block. Matches the i32 sibling.
pub const BLOCK_THREADS: u32 = 256;

/// Number of partitions. Must match [`crate::jit::partition_kernel_i64::NUM_PARTITIONS`]
/// (which is 4096 after the Tier-2.1 NUM_PARTITIONS tuning).
pub const NUM_PARTITIONS: u32 = 4096;

/// Entry-point name embedded in the emitted PTX.
pub const KERNEL_ENTRY: &str = "bolt_partition_reduce_i64";

/// Entry-point name for the spill-counter variant. Distinct from
/// [`KERNEL_ENTRY`] so both kernels can coexist in the JIT module cache.
pub const KERNEL_ENTRY_WITH_SPILL: &str = "bolt_partition_reduce_i64_spill";

/// Probe bound. Same as i32 sibling.
const MAX_PROBES: u32 = BLOCK_GROUPS;

/// Per-iteration `nanosleep.u32` operand for the collision-advance path.
/// See `partition_reduce_kernel::SPIN_BACKOFF_NS` for full rationale.
/// TODO(perf): exponential back-off.
const SPIN_BACKOFF_NS: u32 = 32;

/// Generate PTX for the i64-key per-partition reduce kernel.
///
/// Kernel signature (PTX-level):
/// ```text
/// .visible .entry bolt_partition_reduce_i64(
///     .param .u64 partition_keys,    // const int64_t*  scatter_keys[n_rows]
///     .param .u64 partition_vals,    // const double*   scatter_vals[n_rows]
///     .param .u64 partition_offsets, // const uint32_t* offsets[NUM_PARTITIONS+1]
///     .param .u64 out_keys,          //       int64_t*  [NUM_PARTITIONS*BLOCK_GROUPS]
///     .param .u64 out_vals,          //       double*   [NUM_PARTITIONS*BLOCK_GROUPS]
///     .param .u64 out_set            //       uint8_t*  [NUM_PARTITIONS*BLOCK_GROUPS]
/// )
/// ```
///
/// Launch geometry: `grid = NUM_PARTITIONS, block = BLOCK_THREADS`. One
/// block per partition.
pub fn compile_partition_reduce_kernel_i64() -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY;
    let block_groups = BLOCK_GROUPS;
    let mask = BLOCK_GROUPS - 1;
    let block_threads = BLOCK_THREADS;
    let keys_bytes = BLOCK_GROUPS * 8; // i64 slot stride
    let vals_bytes = BLOCK_GROUPS * 8;
    let set_bytes = BLOCK_GROUPS * 4;
    let max_probes = MAX_PROBES;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Shared-memory open-addressing table. block_keys is i64-aligned
    // (8 bytes per slot); the others match the i32 variant.
    writeln!(
        ptx,
        ".shared .align 8 .b8 block_keys_buf[{bytes}];",
        bytes = keys_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 8 .b8 block_vals_buf[{bytes}];",
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
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_3,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_4,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_5").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Wider register pool — same generous budget as the i32 sibling.
    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<64>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<64>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<8>;").map_err(write_err)?;
    // Operand register for the per-collision `nanosleep.u32` back-off
    // (sm_70+). See partition_reduce_kernel.rs for rationale.
    writeln!(ptx, "\t.reg .u32   %nstime;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // %r0 = blockIdx.x = partition id
    // %r1 = blockDim.x
    // %r2 = threadIdx.x
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;

    // Shared-memory base addresses.
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_vals_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf;").map_err(write_err)?;

    // Param → global pointer setup.
    writeln!(ptx, "\tld.param.u64 %rd3, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd4, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd5, [{entry}_param_2];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd5, %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd6, [{entry}_param_3];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd6, %rd6;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd7, [{entry}_param_4];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd7, %rd7;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd8, [{entry}_param_5];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd8, %rd8;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Read this block's partition slice [start, end).
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd5, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd11, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd12];").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ---------------- Phase 1: cooperatively zero shared arrays ------------
    // block_keys[s] (8 B), block_vals[s] (8 B), block_set[s] (4 B).
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p0, %r20, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    // block_keys[s] = 0  (i64, 8 B)
    writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd21], 0;").map_err(write_err)?;
    // block_vals[s] = 0.0  (f64, 8 B)
    writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd23], 0;").map_err(write_err)?;
    // block_set[s] = 0  (u32, 4 B)
    writeln!(ptx, "\tmul.wide.u32 %rd22, %r20, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd22;").map_err(write_err)?;
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

    // ---------------- Phase 2: probe + sum over partition rows -------------
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    // key = partition_keys[i]  (i64)
    writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rd60, [%rd31];").map_err(write_err)?; // %rd60 = key

    // val = partition_vals[i]  (f64)
    writeln!(ptx, "\tadd.s64 %rd33, %rd4, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.f64 %fd0, [%rd33];").map_err(write_err)?;

    // slot = (key as u32) & mask    — direct slot from low 32 bits.
    // See module docs: for the h2o.ai twokey workload the packed key's
    // low 32 bits are id2 (dense 0..999), giving an even distribution.
    writeln!(ptx, "\tcvt.u32.u64 %r31, %rd60;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tand.b32 %r32, %r31, 0x{mask:X};",
        mask = mask
    )
    .map_err(write_err)?;
    // probe_count = 0
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

    // addr_set  = block_set  + slot * 4
    // addr_key  = block_keys + slot * 8     (i64 slot stride)
    // addr_val  = block_vals + slot * 8
    writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?; // addr_set
    writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd37;").map_err(write_err)?; // addr_key (i64)
    writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?; // addr_val (f64)

    // old = atomicCAS(&block_set[slot], 0, 1)
    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;

    // Else: slot occupied. PTX gives no inter-address ordering between
    // the CAS (block_set) and the key store (block_keys) — different
    // addresses. Insert membar.cta before the key load so a racing
    // thread that observes set==1 can never read a still-zeroed i64 key
    // and false-match key 0.
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    // ACQUIRE-LOAD: pairs with the publisher's atomic store; makes the
    // read-of-published-value contract explicit (sm_70+). Replaces the
    // plain `ld.<space>.<ty>` which relied on the publisher's release +
    // SASS-level implicit acquire — sound in practice but not promised
    // by PTX semantics.
    writeln!(ptx, "\tld.acquire.cta.s64 %rd61, [%rd36];").map_err(write_err)?;
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
    // Occupancy-friendly back-off on the collision-advance path
    // (sm_70+). See partition_reduce_kernel.rs for full rationale.
    writeln!(
        ptx,
        "\tmov.u32 %nstime, {ns};",
        ns = SPIN_BACKOFF_NS
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tnanosleep.u32 %nstime;").map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd36], %rd60;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.f64 %fd1, [%rd38], %fd0;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    writeln!(ptx, "MATCH:").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.f64 %fd2, [%rd38], %fd0;").map_err(write_err)?;

    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ---------------- Phase 3: export populated slots to global ----------
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

    // Load shared slot's i64 key + f64 val + u32 set.
    writeln!(ptx, "\tmul.wide.u32 %rd40, %r41, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd41, %rd0, %rd40;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s64 %rd62, [%rd41];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd43, %rd1, %rd40;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.f64 %fd3, [%rd43];").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd42, %r41, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd44, %rd2, %rd42;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r44, [%rd44];").map_err(write_err)?;

    // Coerce set to 0/1.
    writeln!(ptx, "\tsetp.ne.s32 %p6, %r44, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r45, 1, 0, %p6;").map_err(write_err)?;

    // Store: out_keys[global_slot] (i64), out_vals[global_slot] (f64),
    // out_set[global_slot] (u8).
    writeln!(ptx, "\tmul.wide.u32 %rd45, %r42, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd46, %rd6, %rd45;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s64 [%rd46], %rd62;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd48, %rd7, %rd45;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.f64 [%rd48], %fd3;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u64.u32 %rd49, %r42;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd50, %rd8, %rd49;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd50], %r45;").map_err(write_err)?;

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

/// Spill-counter-aware sibling of [`compile_partition_reduce_kernel_i64`].
///
/// Identical algorithm and shared-memory layout, plus one extra kernel
/// parameter:
///
/// ```text
/// .param .u64 spill_counter   // uint32_t* &spill_counter[1]  (may be 0)
/// ```
///
/// On a probe overflow (MAX_PROBES exceeded without a free or matching
/// slot), the kernel issues `atom.global.add.u32 [spill_counter], 1`
/// before dropping the row — but only if the pointer is non-null. Host
/// orchestrators read the counter after launch+sync; any non-zero value
/// indicates the partition table overflowed and the per-group sums for
/// the spilled key would be silently incorrect.
///
/// Exports the distinct entry [`KERNEL_ENTRY_WITH_SPILL`] so both can
/// coexist in the same JIT cache.
pub fn compile_partition_reduce_kernel_i64_with_spill() -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY_WITH_SPILL;
    let block_groups = BLOCK_GROUPS;
    let mask = BLOCK_GROUPS - 1;
    let block_threads = BLOCK_THREADS;
    let keys_bytes = BLOCK_GROUPS * 8;
    let vals_bytes = BLOCK_GROUPS * 8;
    let set_bytes = BLOCK_GROUPS * 4;
    let max_probes = MAX_PROBES;

    super::partition_reduce_kernel_spill_common::emit_ptx_header(&mut ptx)?;

    // Distinct shmem symbol names so this PTX module can coexist with
    // the non-spill variant if both are loaded into the same context.
    writeln!(
        ptx,
        ".shared .align 8 .b8 block_keys_buf_sp[{bytes}];",
        bytes = keys_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 8 .b8 block_vals_buf_sp[{bytes}];",
        bytes = vals_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 4 .b8 block_set_buf_sp[{bytes}];",
        bytes = set_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_3,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_4,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_5,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_6").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<64>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<80>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<8>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    super::partition_reduce_kernel_spill_common::emit_thread_block_ids(&mut ptx)?;

    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf_sp;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_vals_buf_sp;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf_sp;").map_err(write_err)?;

    // Global pointer setup. %rd3..=%rd8 mirror the non-spill kernel,
    // %rd9 carries the spill_counter pointer.
    writeln!(ptx, "\tld.param.u64 %rd3, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd4, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd5, [{entry}_param_2];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd5, %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd6, [{entry}_param_3];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd6, %rd6;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd7, [{entry}_param_4];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd7, %rd7;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd8, [{entry}_param_5];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd8, %rd8;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd9, [{entry}_param_6];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd9, %rd9;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd10, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd5, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd11, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd12];").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 1: zero shared.
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p0, %r20, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd21], 0;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd23], 0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd22, %r20, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd22;").map_err(write_err)?;
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

    // Phase 2: probe + sum over partition rows.
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rd60, [%rd31];").map_err(write_err)?;

    writeln!(ptx, "\tadd.s64 %rd33, %rd4, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.f64 %fd0, [%rd33];").map_err(write_err)?;

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
    writeln!(ptx, "\t@%p2 bra SPILL_BUMP;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd37;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?;

    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;

    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s64 %rd61, [%rd36];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s64 %p4, %rd61, %rd60;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra MATCH;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r32, %r32, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tand.b32 %r32, %r32, 0x{mask:X};",
        mask = mask
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd36], %rd60;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.f64 %fd1, [%rd38], %fd0;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    writeln!(ptx, "MATCH:").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.f64 %fd2, [%rd38], %fd0;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    // SPILL_BUMP: probe-overflow path. Null-check the spill counter so
    // callers can opt out by passing 0. atom.global.add.u32 is sm_60+.
    super::partition_reduce_kernel_spill_common::emit_spill_bump_with_null_check(&mut ptx, 9)?;

    super::partition_reduce_kernel_spill_common::emit_loop_next_done(&mut ptx)?;

    // Phase 3: export (identical to non-spill).
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
        "\tsetp.ge.u32 %p6, %r41, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p6 bra EXPORT_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r42, %r40, %r41;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd40, %r41, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd41, %rd0, %rd40;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s64 %rd62, [%rd41];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd43, %rd1, %rd40;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.f64 %fd3, [%rd43];").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd42, %r41, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd44, %rd2, %rd42;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r44, [%rd44];").map_err(write_err)?;

    writeln!(ptx, "\tsetp.ne.s32 %p7, %r44, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r45, 1, 0, %p7;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd45, %r42, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd46, %rd6, %rd45;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s64 [%rd46], %rd62;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd48, %rd7, %rd45;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.f64 [%rd48], %fd3;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u64.u32 %rd49, %r42;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd50, %rd8, %rd49;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd50], %r45;").map_err(write_err)?;

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
    BoltError::Other(format!("partition_reduce_kernel_i64: write failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_returns_ok() {
        let ptx = compile_partition_reduce_kernel_i64().expect("compile");
        assert!(!ptx.is_empty());
    }

    #[test]
    fn has_correct_entry() {
        let ptx = compile_partition_reduce_kernel_i64().expect("compile");
        assert!(
            ptx.contains(".visible .entry bolt_partition_reduce_i64("),
            "entry point not found"
        );
    }

    #[test]
    fn uses_i64_key_loads() {
        let ptx = compile_partition_reduce_kernel_i64().expect("compile");
        assert!(
            ptx.contains("ld.global.s64") || ptx.contains("ld.shared.s64"),
            "i64 key loads not found:\n{ptx}"
        );
    }

    #[test]
    fn uses_i64_key_stores_in_output() {
        let ptx = compile_partition_reduce_kernel_i64().expect("compile");
        assert!(
            ptx.contains("st.global.s64"),
            "i64 key store to global not found:\n{ptx}"
        );
    }

    #[test]
    fn uses_atom_shared_cas() {
        let ptx = compile_partition_reduce_kernel_i64().expect("compile");
        assert!(
            ptx.contains("atom.shared.cas.b32"),
            "atom.shared.cas.b32 (slot-claim) not found"
        );
    }

    #[test]
    fn uses_atom_shared_add_f64() {
        let ptx = compile_partition_reduce_kernel_i64().expect("compile");
        assert!(
            ptx.contains("atom.shared.add.f64"),
            "atom.shared.add.f64 (sum) not found"
        );
    }

    #[test]
    fn has_two_syncthreads() {
        let ptx = compile_partition_reduce_kernel_i64().expect("compile");
        let count = ptx.matches("bar.sync 0").count();
        assert!(count >= 2, "expected ≥ 2 bar.sync 0, got {count}");
    }

    /// 1024 slots × 8 B keys + 1024 × 8 B vals + 1024 × 4 B set = 20 KiB.
    #[test]
    fn shared_mem_under_48k() {
        let ptx = compile_partition_reduce_kernel_i64().expect("compile");
        // Coarse check: shared arrays sized to expected bytes.
        assert!(ptx.contains("block_keys_buf[8192]"));
        assert!(ptx.contains("block_vals_buf[8192]"));
        assert!(ptx.contains("block_set_buf[4096]"));
    }

    // ----- _with_spill variant shape tests ---------------------------------

    /// Spill variant exposes a different entry name so both can live in
    /// the same JIT cache.
    #[test]
    fn with_spill_uses_distinct_entry_name() {
        let ptx = compile_partition_reduce_kernel_i64_with_spill().expect("compiles");
        assert_eq!(KERNEL_ENTRY_WITH_SPILL, "bolt_partition_reduce_i64_spill");
        let needle = format!(".visible .entry {}(", KERNEL_ENTRY_WITH_SPILL);
        assert!(ptx.contains(&needle), "PTX missing spill entry:\n{ptx}");
        assert!(
            !ptx.contains(".visible .entry bolt_partition_reduce_i64("),
            "spill variant must not also export base entry"
        );
    }

    /// Adds one .u64 param + the global atomic on overflow.
    #[test]
    fn with_spill_has_seven_params_and_global_atomic() {
        let ptx = compile_partition_reduce_kernel_i64_with_spill().expect("compiles");
        let n = ptx.matches(".param .u64 ").count();
        assert_eq!(n, 7, "spill variant must expose 7 .u64 params, got {n}");
        assert!(
            ptx.contains("atom.global.add.u32"),
            "spill kernel must bump the counter atomically:\n{ptx}"
        );
        assert!(
            ptx.contains("SPILL_BUMP:"),
            "spill kernel must label the overflow path"
        );
        assert!(
            ptx.contains("setp.eq.u64"),
            "spill kernel must null-check the spill_counter pointer"
        );
    }
}
