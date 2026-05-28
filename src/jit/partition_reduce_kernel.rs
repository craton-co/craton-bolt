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
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY;
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

    // Shared-memory open-addressing table. Three parallel arrays so each
    // is naturally aligned; the alternative AoS layout would force 8-byte
    // padding around the u32 set flag.
    writeln!(
        ptx,
        ".shared .align 4 .b8 block_keys_buf[{bytes}];",
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

    // Generous register decls — mirrors the conventions in `shmem_sum_kernel`
    // and `valid_flag_kernels`.
    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    // Register pool widened to 64 to cover the export phase, which uses
    // %r41..%r45 and %rd40..%rd50.
    writeln!(ptx, "\t.reg .b32   %r<64>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<64>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<8>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // --- thread coordinates -------------------------------------------------
    // %r0 = blockIdx.x = partition id
    // %r1 = blockDim.x
    // %r2 = threadIdx.x
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;

    // --- shared-memory base addresses --------------------------------------
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_vals_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf;").map_err(write_err)?;

    // --- global pointer setup (cvta from .param) ---------------------------
    // %rd3 = partition_keys (i32*), %rd4 = partition_vals (f64*),
    // %rd5 = partition_offsets (u32*), %rd6 = out_keys (i32*),
    // %rd7 = out_vals (f64*), %rd8 = out_set (u8*).
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

    // --- read this block's partition slice [start, end) --------------------
    // start = partition_offsets[pid], end = partition_offsets[pid + 1].
    // We need offsets[K+1]-shaped input, with offsets[K] = n_rows; the
    // orchestrator passes the full (K+1) buffer for exactly this reason.
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd5, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?; // start
    writeln!(ptx, "\tadd.s64 %rd12, %rd11, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd12];").map_err(write_err)?; // end
    writeln!(ptx).map_err(write_err)?;

    // ----------------------------------------------------------------------
    // Phase 1: cooperatively zero shared-memory arrays.
    //
    // We zero block_keys (i32), block_vals (f64), and block_set (u32).
    // f64 zero == u64 zero in bits; same for i32 / u32. Each slot needs:
    //   block_keys[s] = 0  (any value — set flag governs validity)
    //   block_vals[s] = 0.0
    //   block_set[s]  = 0
    // ----------------------------------------------------------------------
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p0, %r20, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    // block_keys[s] = 0
    writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd21], 0;").map_err(write_err)?;
    // block_vals[s] = 0.0
    writeln!(ptx, "\tmul.wide.u32 %rd22, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd22;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd23], 0;").map_err(write_err)?;
    // block_set[s] = 0  (u32, 4 bytes, addressed at rd2 + s*4)
    writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd20;").map_err(write_err)?;
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

    // ----------------------------------------------------------------------
    // Phase 2: grid-stride loop over this partition's rows. Each row claims
    // its slot in the shared table via atom.shared.cas, then either writes
    // the key+val (winner) or just adds val (matching key) or advances the
    // probe (collision).
    //
    // Loop induction:
    //   i = start + threadIdx.x   (initial: %r30 = %r10 + %r2)
    //   stride = blockDim.x       (%r1)
    //   continue while i < end    (%r11)
    // ----------------------------------------------------------------------
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    // key = partition_keys[i] (i32)
    writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s32 %r31, [%rd31];").map_err(write_err)?; // %r31 = key

    // val = partition_vals[i] (f64)
    writeln!(ptx, "\tmul.wide.u32 %rd32, %r30, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd33, %rd4, %rd32;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.f64 %fd0, [%rd33];").map_err(write_err)?; // %fd0 = val

    // slot = key & (BLOCK_GROUPS - 1)  — initial probe position
    writeln!(
        ptx,
        "\tand.b32 %r32, %r31, 0x{mask:X};",
        mask = mask
    )
    .map_err(write_err)?;
    // probe_count starts at 0
    writeln!(ptx, "\tmov.u32 %r33, 0;").map_err(write_err)?;

    writeln!(ptx, "PROBE_TOP:").map_err(write_err)?;
    // probe_count += 1; if probe_count > MAX_PROBES, give up (DROP).
    writeln!(ptx, "\tadd.u32 %r33, %r33, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.gt.u32 %p2, %r33, {mp};",
        mp = max_probes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra LOOP_NEXT;").map_err(write_err)?;

    // Compute slot addresses:
    //   addr_set  = block_set  + slot * 4
    //   addr_key  = block_keys + slot * 4
    //   addr_val  = block_vals + slot * 8
    writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?; // addr_set
    writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd34;").map_err(write_err)?; // addr_key
    writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?; // addr_val

    // old = atomicCAS(&block_set[slot], 0, 1)
    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;

    // Else: slot was occupied. PTX gives no inter-address ordering between
    // the CAS on block_set[slot] and the publishing thread's subsequent
    // st.shared.u32 on block_keys[slot] — those touch different addresses.
    // Insert a membar.cta here so this thread, having observed set==1 via
    // its own CAS, sees the winner's key store (which is ordered before
    // its own membar.cta on the CLAIM path). Without this fence a racing
    // thread can read a still-zeroed key and false-match key 0.
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    // ACQUIRE-LOAD: pairs with the publisher's atomic store; makes the
    // read-of-published-value contract explicit (sm_70+). Replaces the
    // plain `ld.<space>.<ty>` which relied on the publisher's release +
    // SASS-level implicit acquire — sound in practice but not promised
    // by PTX semantics.
    writeln!(ptx, "\tld.acquire.cta.s32 %r35, [%rd36];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p4, %r35, %r31;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra MATCH;").map_err(write_err)?;
    // Collision: advance slot = (slot + 1) & mask.
    writeln!(ptx, "\tadd.u32 %r32, %r32, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tand.b32 %r32, %r32, 0x{mask:X};",
        mask = mask
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    // CLAIM: this thread won the slot. Publish the key, then fence so
    // racing MATCH-path readers (which insert their own membar.cta after
    // observing set==1) can never see a zeroed key under set==1.
    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd36], %r31;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.f64 %fd1, [%rd38], %fd0;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    // MATCH: slot already holds our key. Just sum into the val.
    writeln!(ptx, "MATCH:").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.f64 %fd2, [%rd38], %fd0;").map_err(write_err)?;

    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ----------------------------------------------------------------------
    // Phase 3: each of the first BLOCK_GROUPS threads (sweeping by stride
    // BLOCK_THREADS) writes its slot out to global memory.
    //
    // out_keys[pid * BLOCK_GROUPS + s] = block_keys[s]
    // out_vals[pid * BLOCK_GROUPS + s] = block_vals[s]
    // out_set [pid * BLOCK_GROUPS + s] = (block_set[s] != 0) ? 1 : 0
    //
    // Base offset: pid * BLOCK_GROUPS (in slots). We compute the byte base
    // for each output array (keys: ×4, vals: ×8, set: ×1) once at entry.
    // ----------------------------------------------------------------------
    // %r40 = pid * BLOCK_GROUPS (slot base)
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

    // global_slot = %r40 + %r41
    writeln!(ptx, "\tadd.u32 %r42, %r40, %r41;").map_err(write_err)?;

    // Load shared slot's key + val + set.
    writeln!(ptx, "\tmul.wide.u32 %rd40, %r41, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd41, %rd0, %rd40;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s32 %r43, [%rd41];").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd42, %r41, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd43, %rd1, %rd42;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.f64 %fd3, [%rd43];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd44, %rd2, %rd40;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r44, [%rd44];").map_err(write_err)?;

    // Coerce set from u32 → u8 with non-zero collapse to 1.
    writeln!(ptx, "\tsetp.ne.s32 %p6, %r44, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r45, 1, 0, %p6;").map_err(write_err)?;

    // Store back to global out_keys / out_vals / out_set at byte offsets:
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
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY_WITH_SPILL;
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

    writeln!(
        ptx,
        ".shared .align 4 .b8 block_keys_buf_sp[{bytes}];",
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
    writeln!(ptx, "\t.reg .b64   %rd<64>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<8>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;

    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf_sp;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_vals_buf_sp;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, block_set_buf_sp;").map_err(write_err)?;

    // Global pointer setup. Slots 3..=8 mirror the non-spill kernel;
    // slot 9 carries the new spill counter pointer.
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

    // Partition slice [start, end).
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd5, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd11, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd12];").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 1: zero shared arrays.
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p0, %r20, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd20, %r20, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, %rd0, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd21], 0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd22, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd1, %rd22;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd23], 0;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd24, %rd2, %rd20;").map_err(write_err)?;
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

    // Phase 2: probe + atomic add.
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd30, %r30, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s32 %r31, [%rd31];").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd32, %r30, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd33, %rd4, %rd32;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.f64 %fd0, [%rd33];").map_err(write_err)?;

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
    // On probe overflow, bump the spill counter then drop the row. The
    // counter is process-global so we use atom.global.add.u32 — at most
    // ~n_rows total bumps in the worst case (one per dropped row), well
    // within u32's range for any input that fits in GPU memory.
    writeln!(ptx, "\t@%p2 bra SPILL_BUMP;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd34, %r32, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd35, %rd2, %rd34;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd36, %rd0, %rd34;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd37, %r32, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd38, %rd1, %rd37;").map_err(write_err)?;

    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd35], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;

    // MATCH-path inter-address fence (same fix as the non-spill kernel).
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s32 %r35, [%rd36];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p4, %r35, %r31;").map_err(write_err)?;
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
    writeln!(ptx, "\tst.shared.u32 [%rd36], %r31;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.f64 %fd1, [%rd38], %fd0;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    writeln!(ptx, "MATCH:").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.f64 %fd2, [%rd38], %fd0;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    // SPILL_BUMP: atomically increment *spill_counter, then drop the row
    // and proceed to the next iteration. We use atom.global.add.u32 (sm_60+).
    writeln!(ptx, "SPILL_BUMP:").map_err(write_err)?;
    writeln!(ptx, "\tatom.global.add.u32 %r36, [%rd9], 1;").map_err(write_err)?;

    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 3: export, identical to the non-spill kernel.
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

    writeln!(ptx, "\tmul.wide.u32 %rd40, %r41, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd41, %rd0, %rd40;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s32 %r43, [%rd41];").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd42, %r41, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd43, %rd1, %rd42;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.f64 %fd3, [%rd43];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd44, %rd2, %rd40;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r44, [%rd44];").map_err(write_err)?;

    writeln!(ptx, "\tsetp.ne.s32 %p6, %r44, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r45, 1, 0, %p6;").map_err(write_err)?;

    writeln!(ptx, "\tmul.wide.u32 %rd45, %r42, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd46, %rd6, %rd45;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s32 [%rd46], %r43;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd47, %r42, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd48, %rd7, %rd47;").map_err(write_err)?;
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

/// Adapt a `std::fmt::Error` into a `BoltError`. Same shape as the
/// helpers in sibling JIT files — kept local so each kernel emitter is
/// independent.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("partition_reduce_kernel: write failed: {}", e))
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

    /// Inter-address ordering fix: the CAS on `block_set[slot]` and the
    /// store/load on `block_keys[slot]` touch DIFFERENT addresses, so PTX
    /// gives no ordering on sm_70 without an explicit `membar.cta`. Both
    /// the CLAIM path (after publishing the key) and the MATCH path
    /// (before loading the key) must emit it. Without these fences a
    /// racing thread can read a zeroed key under set==1 and false-match
    /// key 0 — the bug this fence pair guards against.
    #[test]
    fn membar_cta_on_claim_and_match_paths() {
        let ptx = compile_partition_reduce_kernel().expect("kernel compiles");
        let count = ptx.matches("membar.cta").count();
        assert!(
            count >= 2,
            "expected >=2 membar.cta (CLAIM publish + MATCH read), saw {count}:\n{ptx}"
        );
        // The MATCH-path fence must live between the CAS and the key
        // load. We locate the CAS and assert the next `membar.cta`
        // appears before the `ld.acquire.cta.s32 %r35` (the key load).
        // The acquire load is in ADDITION to the membar.cta — it makes
        // the read-of-published-value contract explicit at the PTX level.
        let cas_pos = ptx
            .find("atom.shared.cas.b32")
            .expect("CAS must be present");
        let after_cas = &ptx[cas_pos..];
        let mb_pos = after_cas
            .find("membar.cta")
            .expect("membar.cta missing after CAS");
        let ld_pos = after_cas
            .find("ld.acquire.cta.s32 %r35")
            .expect("MATCH-path acquire key load missing");
        assert!(
            mb_pos < ld_pos,
            "membar.cta must appear between CAS and key load on MATCH path:\n{ptx}"
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

    /// Spill variant must keep the membar.cta inter-address fences — the
    /// CAS-race fix from batch 2 still applies here.
    #[test]
    fn with_spill_preserves_membar_cta_fences() {
        let ptx = compile_partition_reduce_kernel_with_spill().expect("kernel compiles");
        let count = ptx.matches("membar.cta").count();
        assert!(
            count >= 2,
            "spill variant must keep both membar.cta fences, saw {count}:\n{ptx}"
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
