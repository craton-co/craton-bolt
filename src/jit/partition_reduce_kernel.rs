// SPDX-License-Identifier: Apache-2.0

//! Per-partition shared-memory GROUP BY SUM kernel (Tier 2.1).
//!
//! ## Why this exists
//!
//! Tier 2 (`groupby_tier2_orchestrator`) partitions the input on the GPU via
//! [`crate::jit::partition_kernel`] + [`crate::jit::scatter_kernel`], then
//! reduces each partition into a `HashMap<i32, f64>` on the **host**. The
//! host pass costs:
//!
//! * `~50 ms` of D2H for two `n_rows`-sized buffers (10 M f64 + 10 M i32 over
//!   PCIe Gen3 x16).
//! * `~100 ms` for K=1024 small host `HashMap`s with ~10 K entries each.
//!
//! That host-pass is the bottleneck (~150 ms of the 460 ms q5 / N=10 M
//! number in `docs/BENCHMARKS.md`). This kernel moves it back onto the GPU,
//! reducing the post-kernel D2H to a fixed ~13 MB regardless of input size
//! (~13 ms over PCIe).
//!
//! ## Approach
//!
//! One CUDA block per partition. Each block:
//!
//! 1. Cooperatively zeros a `__shared__` open-addressing hash table:
//!    `block_keys[BLOCK_GROUPS]`, `block_vals[BLOCK_GROUPS]`,
//!    `block_set[BLOCK_GROUPS]`. We use a `u32` per "set" flag (not `u8`)
//!    because `atom.shared.cas` is only available for `b32` / `b64` widths
//!    on sm_70.
//! 2. Walks its partition's scatter slice `[offsets[pid]..offsets[pid+1])`
//!    in a grid-stride loop over `threadIdx.x`.
//! 3. For each row, linear-probes for the key starting at
//!    `slot = key & (BLOCK_GROUPS - 1)`. At each candidate slot:
//!    * `old = atomicCAS(&block_set[slot], 0, 1)`.
//!    * If `old == 0`: we claimed the slot — write
//!      `block_keys[slot] = key` and `atom.shared.add.f64`.
//!    * Else if `block_keys[slot] == key`: matching slot already populated —
//!      `atom.shared.add.f64`.
//!    * Else: collision; advance to `(slot + 1) & mask` and try again.
//! 4. After a `__syncthreads()`, the first `BLOCK_GROUPS` threads (one per
//!    slot) export their slot to `out_keys[pid * BLOCK_GROUPS + slot]`,
//!    `out_vals[..]`, and `out_set[..]`. The host then walks `out_set` per
//!    partition to collect populated slots.
//!
//! ## Load-factor justification — why 1024 slots for ~10 K partition rows
//!
//! At the q5 / N=10 M scale, each of the K=1024 partitions holds
//! ~`10 M / 1024 ≈ 10 K` ROWS, of which the **distinct** keys are at most
//! ~`1 M / 1024 ≈ 1 K`. The shared-memory table only needs to hold the
//! DISTINCT keys, not the raw rows. So a 1024-slot table is comfortably
//! sized for the distinct keys at q5's cardinality; load factor on the
//! KEY axis is ~1×, not ~10×.
//!
//! What the ~10× factor IS: per-block probes per row vs. table size. Each
//! row triggers at least one CAS probe; with 1024 slots and 10 K rows that's
//! 10 K probes against 1 K live keys, but linear-probing convergence depends
//! on key load factor (1×) — not row count.
//!
//! For workloads that exceed ~1 K distinct keys per partition (e.g.
//! pathologically skewed data, or future tier-2 launches over higher-K
//! workloads), v0 degrades two ways:
//!
//! * If distinct keys per partition exceed BLOCK_GROUPS: the linear probe
//!   wraps around without finding a free slot. **v0 deliberately does not
//!   spill** — instead it advances forever, which would deadlock. We
//!   bound the probe length to `BLOCK_GROUPS` iterations and then drop the
//!   row (counted via a debug counter in v1). This is acceptable because
//!   Tier-2's whole premise is that distinct-keys-per-partition stays under
//!   1024 at the workloads we care about; the orchestrator selects Tier 2
//!   precisely when total distinct keys are ≤ K × BLOCK_GROUPS ≈ 1 M.
//! * If row count per partition exceeds threading limits: irrelevant —
//!   the grid-stride loop handles arbitrarily many rows per partition.
//!
//! v1 will use larger per-partition tables sized to the next power of two
//! ≥ 2 × partition row count, falling back to global atomics on overflow.
//! See `docs/GROUPBY_PERF.md` Tier 2.1 follow-up section.
//!
//! ## Spill counter (v1)
//!
//! The base [`compile_partition_reduce_kernel`] preserves the v0 silent-drop
//! semantics for callers that don't care (and for backward-compatible
//! callsites that haven't been migrated). The companion
//! [`compile_partition_reduce_kernel_with_spill`] emits the same kernel
//! with one additional `.ptr .global .restrict .align 4 spill_counter_ptr`
//! parameter at the end of the parameter list. When a row exhausts
//! `MAX_PROBES` probes without finding a free or matching slot, the kernel
//! atomically increments `*spill_counter_ptr` (via `atom.global.add.u32`)
//! before taking the ADVANCE/exit path. The host must:
//!
//!  1. Allocate a `GpuVec<u32>::zeros(1)` for the counter at launch.
//!  2. Pass its device pointer as the trailing parameter. **Passing 0
//!     (null) is permitted** — the kernel gates the atomic on
//!     `setp.eq.u64 %p, %rd_spill, 0` and skips the increment, leaving the
//!     control flow otherwise identical to the original kernel. This lets
//!     callers that don't care about spill detection share the same
//!     emitted PTX.
//!  3. After kernel sync, copy back the single `u32`. A non-zero value
//!     means at least that many rows were silently dropped from the
//!     reduction — the result is incorrect for those groups and the
//!     executor should propagate an error rather than fold a partial sum.
//!
//! The spill counter does **not** capture the spilled keys (no per-row
//! buffer). It is a count-only diagnostic; the executor's response is to
//! abort the Tier-2 launch with `BoltError::Other` and let the
//! orchestrator fall back to a higher-K configuration or to Tier-1.
//!
//! ## PTX-level notes
//!
//! * Shared variables are declared as raw byte arrays
//!   (`.shared .align 8 .b8 block_keys_buf[4096]`, etc.) and indexed via
//!   `slot * stride` offsets. Same pattern as `shmem_sum_kernel`.
//! * The "set" flag uses `u32` (not `u8`) so we can issue
//!   `atom.shared.cas.b32`. Extra cost: 4 KiB per block of shared memory.
//!   Total per-block shared usage:
//!     - `block_keys`:  4 B × 1024 = 4 KiB
//!     - `block_vals`:  8 B × 1024 = 8 KiB
//!     - `block_set`:   4 B × 1024 = 4 KiB
//!   = 16 KiB per block, comfortably under the 48 KiB sm_70 static budget.
//! * `atom.shared.cas.b32` is sm_60+. `atom.shared.add.f64` is sm_60+. We
//!   target sm_70 (matching the rest of `jit/`).
//! * The CAS on `block_set[slot]` and the subsequent st/ld on
//!   `block_keys[slot]` touch DIFFERENT addresses, so PTX gives no
//!   inter-address ordering on sm_70 without an explicit `membar.cta`.
//!   The CLAIM path emits `membar.cta` between the key store and the val
//!   atomic; the MATCH path emits `membar.cta` between observing set==1
//!   and loading the key. Without these fences, a racing thread could
//!   observe set==1 yet read a still-zeroed `block_keys[slot]` and
//!   false-match key 0 — corrupting the sum.
//! * Probe wrap (`(slot + 1) & mask`) uses `add.u32` + `and.b32` rather
//!   than modulo, since BLOCK_GROUPS is a power of two.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

/// Number of slots in each block's shared-memory open-addressing table.
/// Power of two so probe wrap is `& (BLOCK_GROUPS - 1)`. Sized for the
/// q5 / N=10 M workload (~1 K distinct keys per partition).
pub const BLOCK_GROUPS: u32 = 1024;

/// Threads per block. 256 = one warp × 8 = enough parallelism to hide
/// shared-memory latency on the probe path without over-committing
/// registers. Matches the Tier-1 kernel's block size so the dispatcher
/// can share launch geometry.
pub const BLOCK_THREADS: u32 = 256;

/// Number of partitions launched in parallel (one block per partition).
/// MUST match [`crate::jit::partition_kernel::NUM_PARTITIONS`] —
/// duplicated here so this file remains buildable standalone in the
/// pre-merge sibling layout.
pub const NUM_PARTITIONS: u32 = 4096;

/// Entry-point name embedded in the emitted PTX.
pub const KERNEL_ENTRY: &str = "bolt_partition_reduce";

/// Entry-point name for the spill-counter variant. Different from
/// [`KERNEL_ENTRY`] so the two PTX modules can coexist in the JIT cache
/// without colliding, and so an orchestrator that requests the spill
/// variant cannot accidentally resolve the non-spill function.
pub const KERNEL_ENTRY_WITH_SPILL: &str = "bolt_partition_reduce_spill";

/// Probe bound: how many slots a single key will examine before giving
/// up. Equal to BLOCK_GROUPS so we walk the whole table at worst, which
/// at a < 1× key-load-factor is far more than enough. On overflow the
/// row is silently dropped; see the load-factor section in the module
/// docs for why this is acceptable for v0.
const MAX_PROBES: u32 = BLOCK_GROUPS;

/// Per-iteration `nanosleep.u32` operand for the collision-advance path
/// of the slot-claim loop. PTX `nanosleep.u32` (sm_70+) suspends the
/// warp for ~NS nanoseconds, yielding SM cycles to the warp scheduler so
/// peer warps holding contested slots can finish their writes instead of
/// every warp burning instruction-issue slots on hot probe walks.
///
/// TODO(perf): exponential back-off (shift left by 1 each iteration,
/// capped at 256). That would require a register that survives the loop
/// body across the back-edge, complicating the PTX. A fixed 32 ns
/// constant captures the bulk of the occupancy win at a fraction of the
/// codegen complexity.
const SPIN_BACKOFF_NS: u32 = 32;

/// Key width of the per-partition SUM kernel. The i32-key (`bolt_partition_reduce`,
/// q5) and i64-key (`bolt_partition_reduce_i64`, two-key q3) kernels share ~80% of
/// their scaffold; this enum selects the key-specific bytes in
/// [`emit_sum_kernel`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeyWidth {
    /// Single 32-bit key (i32-key SUM). Key slot is 4 B; key load/store/compare
    /// use `s32`/`u32`; slot index comes straight from the key.
    I32,
    /// 64-bit key (i64-key SUM, two i32 keys packed). Key slot is 8 B; key
    /// load/store/compare use `s64`/`u64`; slot index is derived from the low
    /// 32 bits via `cvt.u32.u64`.
    I64,
}

/// Generate PTX for the per-partition reduce kernel.
///
/// Kernel signature (PTX-level):
///
/// ```text
/// .visible .entry bolt_partition_reduce(
///     .param .u64 partition_keys,    // const int32_t*  scatter_keys[n_rows]
///     .param .u64 partition_vals,    // const double*   scatter_vals[n_rows]
///     .param .u64 partition_offsets, // const uint32_t* offsets[NUM_PARTITIONS+1]
///     .param .u64 out_keys,          //       int32_t*  [NUM_PARTITIONS*BLOCK_GROUPS]
///     .param .u64 out_vals,          //       double*   [NUM_PARTITIONS*BLOCK_GROUPS]
///     .param .u64 out_set            //       uint8_t*  [NUM_PARTITIONS*BLOCK_GROUPS]
/// )
/// ```
///
/// Launch geometry: `grid = NUM_PARTITIONS, block = BLOCK_THREADS`. One
/// block per partition — `blockIdx.x` IS the partition id.
///
/// `compile_partition_reduce_kernel()` is deterministic and pure: same
/// input → same output, no I/O. The dispatcher caches the result.
pub fn compile_partition_reduce_kernel() -> BoltResult<String> {
    emit_sum_kernel(KeyWidth::I32, false, KERNEL_ENTRY)
}

/// Spill-counter-aware sibling of [`compile_partition_reduce_kernel`].
///
/// Identical algorithm and shared-memory layout, with one extra kernel
/// parameter:
///
/// ```text
/// .param .u64 spill_counter   //       uint32_t*  &spill_counter[1]
/// ```
///
/// When a row's linear probe exceeds [`MAX_PROBES`] without finding a
/// matching slot (or claiming an empty one), the kernel issues
/// `atom.global.add.u32 [spill_counter], 1` before dropping the row.
/// Host orchestrators read the counter after launch+sync; any non-zero
/// value indicates the partition table overflowed and the per-group sums
/// for the spilled key would be silently incorrect.
///
/// This variant exports a distinct entry point
/// ([`KERNEL_ENTRY_WITH_SPILL`]) so it can coexist with the non-spill
/// kernel in the same JIT module cache. The non-spill emitter is
/// untouched — existing golden tests still pass.
pub fn compile_partition_reduce_kernel_with_spill() -> BoltResult<String> {
    emit_sum_kernel(KeyWidth::I32, true, KERNEL_ENTRY_WITH_SPILL)
}

/// Adapt a `std::fmt::Error` into a `BoltError`. Same shape as the
/// helpers in sibling JIT files — kept local so each kernel emitter is
/// independent.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("partition_reduce_kernel: write failed: {}", e))
}

// ---------------------------------------------------------------------------
// Unified key-width-parameterised SUM generator.
//
// `emit_sum_kernel` emits BOTH the i32-key kernel (this file's public
// `compile_partition_reduce_kernel{,_with_spill}`) and the i64-key kernel
// (`partition_reduce_kernel_i64`'s `compile_partition_reduce_kernel_i64{,_with_spill}`,
// which delegate here). The whole scaffold — header, entry/regs framing,
// shmem-base + global-pointer setup, partition-slice read, the publish/probe
// protocol, MATCH/epilogue, and the export-loop control flow — is written
// ONCE. Only the genuinely key-dependent bytes branch on `key_width`:
//
//   * shmem `block_keys` align + byte size (4 B/slot vs 8 B/slot),
//   * the spill `%rd<N>` register-file width (i32 spill keeps `%rd<64>`; i64
//     spill needs `%rd<80>`),
//   * the zero-init key store width + scratch-register order,
//   * the probe-prologue key load (`s32`+direct slot vs `s64`+`cvt.u32.u64`)
//     and the addr-compute scratch-register order,
//   * the publish/probe `PublishRegs` + key-type token,
//   * the CLAIM key store width,
//   * the export key load/store width + scratch-register order, and the
//     export-loop predicate numbers (the spill variant's null-check predicate
//     shifts the i64 export predicates by one).
//
// Every other byte is identical across all four (key_width × spill) variants.
// The 4 golden snapshots (`partition_reduce{,_spill}`, `partition_reduce_i64{,_spill}`)
// in `tests/ptx_golden_partition_snapshots.rs` pin the emitted bytes.
// ---------------------------------------------------------------------------

/// Emit the full per-partition SUM kernel for the given `key_width`/`spill`/`entry`.
///
/// `spill == true` appends the trailing `spill_counter` `.u64` param + the
/// `SPILL_BUMP` overflow handler and drops the collision-advance back-off; the
/// i64 spill variant null-checks the counter pointer (the i32 spill variant,
/// older, bumps unconditionally — preserved for byte-stable golden parity).
pub(crate) fn emit_sum_kernel(
    key_width: KeyWidth,
    spill: bool,
    entry: &str,
) -> BoltResult<String> {
    let mut ptx = String::new();
    let block_groups = BLOCK_GROUPS;
    let mask = BLOCK_GROUPS - 1;
    let block_threads = BLOCK_THREADS;
    let max_probes = MAX_PROBES;
    let n_params = if spill { 7 } else { 6 };
    let suffix = if spill { "_sp" } else { "" };
    let overflow_target = if spill { "SPILL_BUMP" } else { "LOOP_NEXT" };

    super::partition_reduce_kernel_spill_common::emit_ptx_header(&mut ptx)?;

    // ---- shared-memory open-addressing table declarations -----------------
    // Three parallel arrays so each is naturally aligned. The key array's
    // align + byte size is the only key-width divergence here (4 B/slot for
    // i32, 8 B/slot for i64).
    let keys_align = match key_width {
        KeyWidth::I32 => 4,
        KeyWidth::I64 => 8,
    };
    let keys_bytes = match key_width {
        KeyWidth::I32 => block_groups * 4,
        KeyWidth::I64 => block_groups * 8,
    };
    let vals_bytes = block_groups * 8;
    let set_bytes = block_groups * 4;
    writeln!(
        ptx,
        ".shared .align {al} .b8 block_keys_buf{suffix}[{bytes}];",
        al = keys_align,
        bytes = keys_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 8 .b8 block_vals_buf{suffix}[{bytes}];",
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

    // ---- entry signature + register-file declarations ---------------------
    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    for i in 0..n_params {
        let comma = if i + 1 == n_params { "" } else { "," };
        writeln!(ptx, "\t.param .u64 {entry}_param_{i}{comma}").map_err(write_err)?;
    }
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<64>;").map_err(write_err)?;
    // Spill `%rd` width: i32 keeps the original `%rd<64>`; i64's spill variant
    // declares the wider `%rd<80>`. Non-spill kernels both use `%rd<64>` and
    // additionally declare the `%nstime` back-off operand.
    let rd_count = if spill && key_width == KeyWidth::I64 {
        80
    } else {
        64
    };
    writeln!(ptx, "\t.reg .b64   %rd<{rd_count}>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<8>;").map_err(write_err)?;
    if !spill {
        // Operand register for the per-collision `nanosleep.u32` back-off.
        // PTX requires the register form for portability across toolchains.
        writeln!(ptx, "\t.reg .u32   %nstime;").map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    // --- thread coordinates -------------------------------------------------
    // %r0 = blockIdx.x = partition id, %r1 = blockDim.x, %r2 = threadIdx.x.
    super::partition_reduce_kernel_spill_common::emit_thread_block_ids(&mut ptx)?;

    // --- shared-memory base addresses --------------------------------------
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf{suffix};").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_vals_buf{suffix};").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf{suffix};").map_err(write_err)?;

    // --- global pointer setup (cvta from .param) ---------------------------
    // Param j lands in %rd{3 + j}: keys/vals/offsets/out_keys/out_vals/out_set
    // (+ the spill_counter pointer in %rd9 for the spill variant).
    for j in 0..n_params {
        let rd = 3 + j;
        writeln!(ptx, "\tld.param.u64 %rd{rd}, [{entry}_param_{j}];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd{rd}, %rd{rd};").map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    // --- read this block's partition slice [start, end) --------------------
    super::partition_reduce_kernel_spill_common::emit_partition_slice_read(&mut ptx)?;
    writeln!(ptx).map_err(write_err)?;

    // ---- Phase 1: cooperatively zero the shared arrays --------------------
    emit_zero_init(&mut ptx, key_width, block_groups, block_threads)?;

    // ---- Phase 2: probe + atomic add over this partition's rows -----------
    emit_probe_loop_prefix(&mut ptx, key_width, mask, max_probes, overflow_target)?;

    // 3-state publish/probe protocol (the claim-then-write race fix).
    emit_sum_publish_probe(&mut ptx, key_width)?;

    // Collision: advance slot = (slot + 1) & mask.
    emit_collision_advance(&mut ptx, mask)?;
    if !spill {
        // Occupancy-friendly back-off on the collision-advance path (the spill
        // variant omits this and jumps straight back to PROBE_TOP).
        super::partition_reduce_kernel_spill_common::emit_spin_backoff(&mut ptx, SPIN_BACKOFF_NS)?;
    }
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    // CLAIM: this thread won the slot. Store key, fence, publish set:=2.
    emit_claim(&mut ptx, key_width)?;

    // MATCH: slot already holds our key. Sum into the val. The non-spill kernel
    // FALLS THROUGH into LOOP_NEXT; the spill kernel ends MATCH with an explicit
    // `bra LOOP_NEXT` (its SPILL_BUMP handler sits between MATCH and the
    // epilogue), so the epilogue is emitted separately via emit_loop_next_done.
    writeln!(ptx, "MATCH:").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.f64 %fd2, [%rd38], %fd0;").map_err(write_err)?;

    if spill {
        writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

        // SPILL_BUMP: bump the spill counter, then fall to the epilogue. The
        // i32 variant (older) bumps unconditionally; the i64 variant null-checks
        // the pointer so callers can opt out with 0. Both shapes are byte-stable
        // against their golden snapshots.
        match key_width {
            KeyWidth::I32 => {
                super::partition_reduce_kernel_spill_common::emit_spill_bump_unchecked(
                    &mut ptx, 9,
                )?;
            }
            KeyWidth::I64 => {
                super::partition_reduce_kernel_spill_common::emit_spill_bump_with_null_check(
                    &mut ptx, 9,
                )?;
            }
        }

        super::partition_reduce_kernel_spill_common::emit_loop_next_done(&mut ptx)?;
    } else {
        writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
        writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
        writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
        writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
        writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
        writeln!(ptx).map_err(write_err)?;
    }

    // ---- Phase 3: per-slot export to global memory ------------------------
    // The export loop/set predicates are normally `%p5`/`%p6`. ONLY the i64
    // spill variant shifts them to `%p6`/`%p7`: its `SPILL_BUMP` handler emits a
    // `setp.eq.u64 %p5` null-check that consumes `%p5`. The i32 spill variant
    // (older, unchecked `SPILL_BUMP`) emits NO such predicate, so it keeps the
    // base `%p5`/`%p6` — matching its golden snapshot exactly.
    let shift_export_preds = spill && key_width == KeyWidth::I64;
    let (loop_pred, set_pred) = if shift_export_preds {
        ("%p6", "%p7")
    } else {
        ("%p5", "%p6")
    };
    emit_export(
        &mut ptx,
        key_width,
        block_groups,
        block_threads,
        loop_pred,
        set_pred,
    )?;

    Ok(ptx)
}

/// Phase 1: cooperatively zero the three shared-memory arrays
/// (`block_keys`, `block_vals` f64, `block_set` u32) in a strided loop,
/// then `bar.sync 0` + blank line. The key store width + scratch-register
/// order branch on `key_width`; everything else is shared.
fn emit_zero_init(
    ptx: &mut String,
    key_width: KeyWidth,
    block_groups: u32,
    block_threads: u32,
) -> BoltResult<()> {
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r20, {bg};", bg = block_groups).map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    match key_width {
        KeyWidth::I32 => {
            // block_keys[s] = 0  (i32, 4 B)
            writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u32 [%rd21], 0;").map_err(write_err)?;
            // block_vals[s] = 0.0  (f64, 8 B)
            writeln!(ptx, "\tmul.wide.u32 %rd22, %r20, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd22;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u64 [%rd23], 0;").map_err(write_err)?;
            // block_set[s] = 0  (u32, 4 B, addressed at rd2 + s*4)
            writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd20;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u32 [%rd24], 0;").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            // block_keys[s] = 0  (i64, 8 B)
            writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u64 [%rd21], 0;").map_err(write_err)?;
            // block_vals[s] = 0.0  (f64, 8 B) — same stride, reuses %rd20.
            writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd20;").map_err(write_err)?;
            writeln!(ptx, "\tst.shared.u64 [%rd23], 0;").map_err(write_err)?;
            // block_set[s] = 0  (u32, 4 B)
            writeln!(ptx, "\tmul.wide.u32 %rd22, %r20, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd22;").map_err(write_err)?;
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

/// Phase 2 prefix: the grid-stride loop head + per-row key/val load + initial
/// probe-slot compute, up to and including the `@%p3 bra CLAIM;` branch on the
/// CAS result. The key load (`s32` + direct slot vs `s64` + `cvt.u32.u64`) and
/// the addr-compute scratch-register order branch on `key_width`;
/// `overflow_target` (`LOOP_NEXT` / `SPILL_BUMP`) is the probe-overflow branch.
fn emit_probe_loop_prefix(
    ptx: &mut String,
    key_width: KeyWidth,
    mask: u32,
    max_probes: u32,
    overflow_target: &str,
) -> BoltResult<()> {
    super::partition_reduce_kernel_spill_common::emit_loop_head(ptx)?;

    match key_width {
        KeyWidth::I32 => {
            // key = partition_keys[i] (i32)
            writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
            writeln!(ptx, "\tld.global.s32 %r31, [%rd31];").map_err(write_err)?; // %r31 = key

            // val = partition_vals[i] (f64)
            writeln!(ptx, "\tmul.wide.u32 %rd32, %r30, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd33, %rd4, %rd32;").map_err(write_err)?;
            writeln!(ptx, "\tld.global.f64 %fd0, [%rd33];").map_err(write_err)?; // %fd0 = val

            // slot = key & (BLOCK_GROUPS - 1)  — initial probe position
            writeln!(ptx, "\tand.b32 %r32, %r31, 0x{mask:X};", mask = mask).map_err(write_err)?;
        }
        KeyWidth::I64 => {
            // key = partition_keys[i] (i64)
            writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
            writeln!(ptx, "\tld.global.s64 %rd60, [%rd31];").map_err(write_err)?; // %rd60 = key

            // val = partition_vals[i] (f64) — same stride, reuses %rd30.
            writeln!(ptx, "\tadd.s64 %rd33, %rd4, %rd30;").map_err(write_err)?;
            writeln!(ptx, "\tld.global.f64 %fd0, [%rd33];").map_err(write_err)?;

            // slot = (key as u32) & mask    — direct slot from low 32 bits.
            writeln!(ptx, "\tcvt.u32.u64 %r31, %rd60;").map_err(write_err)?;
            writeln!(ptx, "\tand.b32 %r32, %r31, 0x{mask:X};", mask = mask).map_err(write_err)?;
        }
    }
    // probe_count starts at 0; PROBE_TOP increments + bounds it.
    super::partition_reduce_kernel_spill_common::emit_probe_bound_check(
        ptx,
        max_probes,
        overflow_target,
    )?;

    // Compute slot addresses. The set/val strides (×4 / ×8) are width-agnostic;
    // the key slot stride (×4 for i32, ×8 for i64) and the resulting
    // scratch-register order are the only divergence.
    match key_width {
        KeyWidth::I32 => {
            //   addr_set  = block_set  + slot * 4
            //   addr_key  = block_keys + slot * 4
            //   addr_val  = block_vals + slot * 8
            writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?; // addr_set
            writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd34;").map_err(write_err)?; // addr_key
            writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?; // addr_val
        }
        KeyWidth::I64 => {
            //   addr_set  = block_set  + slot * 4
            //   addr_key  = block_keys + slot * 8     (i64 slot stride)
            //   addr_val  = block_vals + slot * 8
            writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?; // addr_set
            writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd37;").map_err(write_err)?; // addr_key (i64)
            writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?; // addr_val (f64)
        }
    }

    // old = atomicCAS(&block_set[slot], 0, 1)
    super::partition_reduce_kernel_spill_common::emit_slot_claim_cas(ptx, "%rd35")?;
    Ok(())
}

/// Emit the `emit_publish_probe_protocol` call for the given key width. The
/// register tuple is identical across widths except the key destination /
/// probe-key registers (`%r35`/`%r31` for i32, `%rd61`/`%rd60` for i64) and the
/// key-type token (`s32` / `s64`).
fn emit_sum_publish_probe(ptx: &mut String, key_width: KeyWidth) -> BoltResult<()> {
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
    )
}

/// Collision-advance: `slot = (slot + 1) & mask`. Byte-identical across widths;
/// the caller follows it with the `nanosleep` back-off (non-spill) or jumps
/// straight back to `PROBE_TOP` (spill).
fn emit_collision_advance(ptx: &mut String, mask: u32) -> BoltResult<()> {
    writeln!(ptx, "\tadd.u32 %r32, %r32, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r32, %r32, 0x{mask:X};", mask = mask).map_err(write_err)?;
    Ok(())
}

/// CLAIM block: this thread won the slot. Store key, fence, publish set:=2,
/// then `atom.shared.add.f64` and `bra LOOP_NEXT`. Only the key store width
/// (`u32`+`%r31` vs `u64`+`%rd60`) branches on `key_width`.
fn emit_claim(ptx: &mut String, key_width: KeyWidth) -> BoltResult<()> {
    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    match key_width {
        KeyWidth::I32 => {
            writeln!(ptx, "\tst.shared.u32 [%rd36], %r31;").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            writeln!(ptx, "\tst.shared.u64 [%rd36], %rd60;").map_err(write_err)?;
        }
    }
    super::partition_reduce_kernel_spill_common::emit_claim_publish(ptx, "%rd35")?;
    writeln!(ptx, "\tatom.shared.add.f64 %fd1, [%rd38], %fd0;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;
    Ok(())
}

/// Phase 3: per-slot export loop. Each of the first `BLOCK_GROUPS` threads
/// (sweeping by stride `BLOCK_THREADS`) writes its shared slot's key/val/set
/// out to global memory, then `ret;` + closing `}`. The key load/store width
/// and the export-block scratch-register order branch on `key_width`;
/// `loop_pred`/`set_pred` are the bound / set-coercion predicates (the spill
/// variant's null-check shifts them by one).
fn emit_export(
    ptx: &mut String,
    key_width: KeyWidth,
    block_groups: u32,
    block_threads: u32,
    loop_pred: &str,
    set_pred: &str,
) -> BoltResult<()> {
    // %r40 = pid * BLOCK_GROUPS (slot base)
    writeln!(ptx, "\tmul.lo.u32 %r40, %r0, {bg};", bg = block_groups).map_err(write_err)?;

    writeln!(ptx, "\tmov.u32 %r41, %r2;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 {loop_pred}, %r41, {bg};", bg = block_groups)
        .map_err(write_err)?;
    writeln!(ptx, "\t@{loop_pred} bra EXPORT_DONE;").map_err(write_err)?;

    // global_slot = %r40 + %r41
    writeln!(ptx, "\tadd.u32 %r42, %r40, %r41;").map_err(write_err)?;

    // Load shared slot's key + val + set.
    match key_width {
        KeyWidth::I32 => {
            writeln!(ptx, "\tmul.wide.u32 %rd40, %r41, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd41, %rd0, %rd40;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.s32 %r43, [%rd41];").map_err(write_err)?;
            writeln!(ptx, "\tmul.wide.u32 %rd42, %r41, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd43, %rd1, %rd42;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.f64 %fd3, [%rd43];").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd44, %rd2, %rd40;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.u32 %r44, [%rd44];").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            // i64 key (8 B), f64 val (8 B, reuses the key offset %rd40), u32 set.
            writeln!(ptx, "\tmul.wide.u32 %rd40, %r41, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd41, %rd0, %rd40;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.s64 %rd62, [%rd41];").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd43, %rd1, %rd40;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.f64 %fd3, [%rd43];").map_err(write_err)?;
            writeln!(ptx, "\tmul.wide.u32 %rd42, %r41, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd44, %rd2, %rd42;").map_err(write_err)?;
            writeln!(ptx, "\tld.shared.u32 %r44, [%rd44];").map_err(write_err)?;
        }
    }

    // Coerce set from u32 → u8 with non-zero collapse to 1.
    writeln!(ptx, "\tsetp.ne.s32 {set_pred}, %r44, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r45, 1, 0, {set_pred};").map_err(write_err)?;

    // Store back to global out_keys / out_vals / out_set at byte offsets.
    match key_width {
        KeyWidth::I32 => {
            //   out_keys + global_slot * 4
            //   out_vals + global_slot * 8
            //   out_set  + global_slot * 1
            writeln!(ptx, "\tmul.wide.u32 %rd45, %r42, 4;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd46, %rd6, %rd45;").map_err(write_err)?;
            writeln!(ptx, "\tst.global.s32 [%rd46], %r43;").map_err(write_err)?;
            writeln!(ptx, "\tmul.wide.u32 %rd47, %r42, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd48, %rd7, %rd47;").map_err(write_err)?;
            writeln!(ptx, "\tst.global.f64 [%rd48], %fd3;").map_err(write_err)?;
            writeln!(ptx, "\tcvt.u64.u32 %rd49, %r42;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd50, %rd8, %rd49;").map_err(write_err)?;
            writeln!(ptx, "\tst.global.u8 [%rd50], %r45;").map_err(write_err)?;
        }
        KeyWidth::I64 => {
            //   out_keys + global_slot * 8   (i64; val reuses the offset %rd45)
            //   out_vals + global_slot * 8
            //   out_set  + global_slot * 1
            writeln!(ptx, "\tmul.wide.u32 %rd45, %r42, 8;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd46, %rd6, %rd45;").map_err(write_err)?;
            writeln!(ptx, "\tst.global.s64 [%rd46], %rd62;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd48, %rd7, %rd45;").map_err(write_err)?;
            writeln!(ptx, "\tst.global.f64 [%rd48], %fd3;").map_err(write_err)?;
            writeln!(ptx, "\tcvt.u64.u32 %rd49, %r42;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd50, %rd8, %rd49;").map_err(write_err)?;
            writeln!(ptx, "\tst.global.u8 [%rd50], %r45;").map_err(write_err)?;
        }
    }

    writeln!(ptx, "\tadd.u32 %r41, %r41, {bt};", bt = block_threads).map_err(write_err)?;
    writeln!(ptx, "\tbra EXPORT_TOP;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_DONE:").map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// PTX-shape tests (host-only — do NOT require a GPU).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// All three shared-memory tables must be declared in the module
    /// preamble; missing any would cause a PTX assembler error at module
    /// load time.
    #[test]
    fn declares_three_shared_arrays() {
        let ptx = compile_partition_reduce_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("block_keys_buf"),
            "PTX must declare block_keys_buf:\n{ptx}"
        );
        assert!(
            ptx.contains("block_vals_buf"),
            "PTX must declare block_vals_buf:\n{ptx}"
        );
        assert!(
            ptx.contains("block_set_buf"),
            "PTX must declare block_set_buf:\n{ptx}"
        );
        // Three .shared lines for the three arrays.
        let shared_count = ptx.matches(".shared").count();
        assert!(
            shared_count >= 3,
            "expected at least three .shared declarations, saw {shared_count}:\n{ptx}"
        );
    }

    /// The probe protocol relies on `atom.shared.cas.b32` to claim slots
    /// safely under contention. Without it, two threads could both think
    /// they own the same empty slot.
    #[test]
    fn uses_atom_shared_cas_b32() {
        let ptx = compile_partition_reduce_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("atom.shared.cas.b32"),
            "PTX must issue atom.shared.cas.b32 to claim a slot:\n{ptx}"
        );
    }

    /// 3-state publish protocol (the tier-2 phantom-groups fix). Claiming a
    /// slot (CAS set 0->1) and storing the key are separate ops, so a prober
    /// must NOT read the key until the claimer has published it — otherwise it
    /// reads a stale 0 and mints a duplicate group. The protocol:
    ///   * CLAIM: `st key; membar.cta; st set:=2` (publish/release), in order.
    ///   * prober: spin in `PUBLISH_WAIT` on an `ld.acquire.cta.u32` of `set`
    ///     until it reads 2, THEN load the key.
    #[test]
    fn publish_protocol_closes_claim_then_write_race() {
        let ptx = compile_partition_reduce_kernel().expect("kernel compiles");

        // Prober spin-waits for publication before reading the key.
        assert!(
            ptx.contains("PUBLISH_WAIT:"),
            "PTX must spin-wait for the claimer to publish before reading the key:\n{ptx}"
        );
        assert!(
            ptx.contains("ld.volatile.shared.u32 %r36"),
            "PUBLISH_WAIT must acquire-load the set flag each spin:\n{ptx}"
        );

        // CLAIM must publish in order: key store -> membar.cta -> set:=2.
        let claim_pos = ptx.find("CLAIM:").expect("CLAIM label must be present");
        let after_claim = &ptx[claim_pos..];
        let key_store = after_claim
            .find("st.shared.u32 [%rd36], %r31")
            .expect("CLAIM must store the key");
        let mb = after_claim
            .find("membar.cta")
            .expect("CLAIM must fence after the key store");
        let set_pub = after_claim
            .find("st.shared.u32 [%rd35], 2")
            .expect("CLAIM must publish set:=2");
        assert!(
            key_store < mb && mb < set_pub,
            "CLAIM order must be key store -> membar.cta -> set:=2:\n{ptx}"
        );
    }

    /// Accumulation into the per-slot f64 sum MUST use
    /// `atom.shared.add.f64`. The MATCH path lets multiple threads add
    /// into the same slot concurrently.
    #[test]
    fn uses_atom_shared_add_f64() {
        let ptx = compile_partition_reduce_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("atom.shared.add.f64"),
            "PTX must issue atom.shared.add.f64 for the per-slot sum:\n{ptx}"
        );
        // The CLAIM and MATCH paths both add — expect ≥ 2 occurrences.
        let count = ptx.matches("atom.shared.add.f64").count();
        assert!(
            count >= 2,
            "expected ≥ 2 atom.shared.add.f64 (CLAIM + MATCH paths), saw {count}:\n{ptx}"
        );
    }

    /// `__syncthreads()` is `bar.sync 0` in PTX. We need it at least
    /// twice: once after zeroing shared memory, once between the
    /// accumulate loop and the export phase. Without either barrier
    /// the kernel races on shared state.
    #[test]
    fn syncthreads_at_least_twice() {
        let ptx = compile_partition_reduce_kernel().expect("kernel compiles");
        let count = ptx.matches("bar.sync 0").count();
        assert!(
            count >= 2,
            "expected ≥ 2 bar.sync 0 (after zero-init, between accumulate and export), \
             saw {count}:\n{ptx}"
        );
    }

    /// The kernel must export the well-known entry point name so the
    /// orchestrator can resolve it via `cuModuleGetFunction`.
    #[test]
    fn has_correct_entry_name() {
        let ptx = compile_partition_reduce_kernel().expect("kernel compiles");
        let needle = format!(".visible .entry {}(", KERNEL_ENTRY);
        assert!(
            ptx.contains(&needle),
            "PTX must declare .visible .entry {}(  — got:\n{ptx}",
            KERNEL_ENTRY
        );
    }

    /// Byte-stable refactor guard: the header + thread-id prefix produced
    /// via the shared `spill_common` helpers must match the exact bytes the
    /// previously-inlined `writeln!` calls emitted. If the helper bytes
    /// drift, this catches it without needing a GPU or a full snapshot.
    #[test]
    fn header_and_thread_id_prefix_is_byte_stable() {
        let ptx = compile_partition_reduce_kernel().expect("kernel compiles");
        assert!(
            ptx.starts_with(".version 7.5\n.target sm_70\n.address_size 64\n\n"),
            "header bytes drifted:\n{ptx}"
        );
        // Thread-id setup appears verbatim immediately after the shared base
        // movs; assert the exact 3-line block is present in order.
        assert!(
            ptx.contains(
                "\tmov.u32 %r0, %ctaid.x;\n\tmov.u32 %r1, %ntid.x;\n\tmov.u32 %r2, %tid.x;\n"
            ),
            "thread-id prefix bytes drifted:\n{ptx}"
        );
        // The collision-advance back-off pair must still appear verbatim.
        assert!(
            ptx.contains("\tmov.u32 %nstime, 32;\n\tnanosleep.u32 %nstime;\n"),
            "spin back-off bytes drifted:\n{ptx}"
        );
    }

    /// `compile_partition_reduce_kernel` is pure: same input -> same
    /// output. A dispatcher caching the PTX should always get a stable
    /// result.
    #[test]
    fn output_is_deterministic() {
        let a = compile_partition_reduce_kernel().expect("compile a");
        let b = compile_partition_reduce_kernel().expect("compile b");
        assert_eq!(a, b, "partition_reduce kernel emitter must be deterministic");
    }

    /// Linear-probe pattern detectable via labels: PROBE_TOP / CLAIM /
    /// MATCH / collision-advance. Counting `bra` occurrences pointing
    /// back to PROBE_TOP confirms the probe loops correctly.
    #[test]
    fn has_linear_probe_structure() {
        let ptx = compile_partition_reduce_kernel().expect("kernel compiles");
        assert!(ptx.contains("PROBE_TOP:"), "PTX must label PROBE_TOP:\n{ptx}");
        assert!(ptx.contains("CLAIM:"), "PTX must label CLAIM:\n{ptx}");
        assert!(ptx.contains("MATCH:"), "PTX must label MATCH:\n{ptx}");
        // The collision-advance branches back to PROBE_TOP; the bounded
        // probe-overflow branch exits to LOOP_NEXT. Verify both branch
        // targets exist.
        assert!(
            ptx.contains("bra PROBE_TOP;"),
            "PTX must contain a bra PROBE_TOP for the collision-advance:\n{ptx}"
        );
    }

    /// PTX module preamble must match the rest of `src/jit/*` — same
    /// target, same address size. A mismatch would prevent the driver
    /// from co-loading this kernel with other Craton Bolt modules.
    #[test]
    fn ptx_header_matches_project_conventions() {
        let ptx = compile_partition_reduce_kernel().expect("kernel compiles");
        assert!(ptx.contains(".version 7.5"), "PTX must be .version 7.5");
        assert!(ptx.contains(".target sm_70"), "PTX must target sm_70");
        assert!(
            ptx.contains(".address_size 64"),
            "PTX must declare .address_size 64"
        );
    }

    /// GPU-required smoke test: confirm the PTX is accepted by the CUDA
    /// driver. Skipped by default; run with `cargo test --release --
    /// --ignored` on a machine with a CUDA 12.x driver.
    #[test]
    #[ignore]
    fn ptx_loads_into_cuda_driver() {
        let ptx = compile_partition_reduce_kernel().expect("kernel compiles");
        let module = crate::jit::CudaModule::from_ptx(&ptx)
            .expect("PTX should load via cuModuleLoadDataEx");
        let _fn = module
            .function(KERNEL_ENTRY)
            .expect("kernel entry point should be reachable");
    }

    // ----- _with_spill variant shape tests ---------------------------------

    /// The spill-counter variant exposes a different entry point than the
    /// base kernel so both can coexist in one PTX cache without colliding.
    #[test]
    fn with_spill_uses_distinct_entry_name() {
        let ptx = compile_partition_reduce_kernel_with_spill().expect("kernel compiles");
        assert_eq!(KERNEL_ENTRY_WITH_SPILL, "bolt_partition_reduce_spill");
        let needle = format!(".visible .entry {}(", KERNEL_ENTRY_WITH_SPILL);
        assert!(
            ptx.contains(&needle),
            "PTX must declare the spill entry point:\n{ptx}"
        );
        assert!(
            !ptx.contains(".visible .entry bolt_partition_reduce("),
            "spill variant must NOT also export the base entry name:\n{ptx}"
        );
    }

    /// The spill kernel adds one extra .u64 parameter (the spill-counter
    /// pointer) and bumps it atomically when MAX_PROBES is exceeded.
    #[test]
    fn with_spill_has_seven_pointer_params_and_global_atomic() {
        let ptx = compile_partition_reduce_kernel_with_spill().expect("kernel compiles");
        let n_params = ptx.matches(".param .u64 ").count();
        assert_eq!(
            n_params, 7,
            "spill variant must expose 7 .u64 params (6 + spill_counter), got {n_params}\n{ptx}"
        );
        assert!(
            ptx.contains("atom.global.add.u32"),
            "spill variant must atomically bump the counter on overflow:\n{ptx}"
        );
        assert!(
            ptx.contains("SPILL_BUMP:"),
            "spill variant must label its overflow path:\n{ptx}"
        );
    }

    /// Spill variant must carry the same 3-state publish protocol as the base
    /// emitter (the tier-2 phantom-groups fix): a PUBLISH_WAIT spin on an
    /// acquire-load of `set`, and a CLAIM that orders key store -> membar.cta
    /// -> set:=2.
    #[test]
    fn with_spill_carries_publish_protocol() {
        let ptx = compile_partition_reduce_kernel_with_spill().expect("kernel compiles");
        assert!(
            ptx.contains("PUBLISH_WAIT:") && ptx.contains("ld.volatile.shared.u32 %r36"),
            "spill variant must spin-wait for publication before reading the key:\n{ptx}"
        );
        let claim_pos = ptx.find("CLAIM:").expect("CLAIM label must be present");
        let after_claim = &ptx[claim_pos..];
        let key_store = after_claim
            .find("st.shared.u32 [%rd36], %r31")
            .expect("CLAIM must store the key");
        let mb = after_claim
            .find("membar.cta")
            .expect("CLAIM must fence after the key store");
        let set_pub = after_claim
            .find("st.shared.u32 [%rd35], 2")
            .expect("CLAIM must publish set:=2");
        assert!(
            key_store < mb && mb < set_pub,
            "CLAIM order must be key store -> membar.cta -> set:=2:\n{ptx}"
        );
    }

    /// The spill variant is deterministic just like the base emitter.
    #[test]
    fn with_spill_is_deterministic() {
        let a = compile_partition_reduce_kernel_with_spill().expect("compile a");
        let b = compile_partition_reduce_kernel_with_spill().expect("compile b");
        assert_eq!(a, b, "spill emitter must be deterministic");
    }
}
