// SPDX-License-Identifier: Apache-2.0

//! Hash-partition pass-1 kernel for **64-bit packed keys** (Tier 2 of the
//! GROUP BY perf plan).
//!
//! This is the i64 sibling of [`crate::jit::partition_kernel`]. The single-
//! Int32 variant hashes 32-bit keys into one of `NUM_PARTITIONS = 4096`
//! buckets using `(u32)key * 0x9E3779B1 & 0xFFF`. This file does the same
//! pipeline for `int64_t` keys produced by the host-side
//! `groupby.rs::pack_keys` packing convention — two Int32 columns packed
//! losslessly into an Int64 (high 32 bits = column 0, low 32 bits = column 1).
//!
//! ## Why a separate kernel
//!
//! We MUST hash the *full* 64-bit packed key, not just its low half. If we
//! reused the i32 partition kernel by sign-extension truncation we'd lose
//! the entire high 32 bits of the key — meaning every distinct pair sharing
//! a low half (e.g. all `(*, 7)` pairs in q3's two-key GROUP BY) would alias
//! into the same partition and Tier-2's "each partition fits in shmem"
//! invariant would collapse to "one mega-partition holds half the keys".
//!
//! ## Algorithm
//!
//! ```text
//! const uint32_t K                 = 4096;
//! const uint64_t HASH_MULTIPLIER64 = 0x9E3779B97F4A7C15ull;  // floor(2^64 / phi)
//!
//! for (i = blockIdx.x*blockDim.x + tid; i < n_rows; i += gridDim.x*blockDim.x) {
//!     int64_t  key  = keys[i];
//!     uint64_t hash = (uint64_t)key * HASH_MULTIPLIER64;  // wraps mod 2^64
//!     uint32_t pid  = (uint32_t)(hash >> 52);             // top log2(K)=12 bits
//!     atomicAdd(&counts[pid], 1u);
//!     partition_ids[i] = pid;
//! }
//! ```
//!
//! Why top-bits-of-multiplicative-hash instead of mask-the-low-bits?
//! Knuth's high-bit trick exploits the fact that integer multiplication
//! distributes good mixing into the **high** bits of the product; the low
//! bits remain heavily structured (the low bit of `x * M` is just the low
//! bit of `x`, since M is odd). For a 64-bit key we therefore right-shift
//! by `64 - log2(K) = 52` to read the top 12 bits as the partition id.
//! The i32 sibling can get away with `(hash & 0xFFF)` because its 32-bit
//! multiply already wraps and the wrap acts as a poor-man's mix; at 64
//! bits we'd be reading entirely raw key bits otherwise.
//!
//! ## PTX-level notes
//!
//! * Keys are loaded with `ld.global.s64` (8-byte loads). Stride per row is 8.
//! * The 64-bit multiply uses `mul.lo.u64`, which writes the low 64 bits of
//!   the 128-bit product — exactly the wrapping multiply we want.
//! * The `log2(K)`-bit shift to extract the partition id is
//!   `shr.u64 %rdN, %rd_hash, 52`, followed by `cvt.u32.u64` to land in a
//!   32-bit register for the atomic. We deliberately use the unsigned
//!   `shr.u64` so the top bits zero-fill (no sign extension hazard from
//!   `shr.s64`).
//! * Two emitter variants — see [`compile_partition_kernel_i64_global_atomics`]
//!   (per-row `atom.global.add.u32`, simplest and best for small inputs) and
//!   [`compile_partition_kernel_i64_shmem_staging`] (per-row
//!   `atom.shared.add.u32` into a block-private 16 KiB `block_counts[]`,
//!   then a flush-phase `atom.global.add.u32` per (block, partition) at
//!   end-of-block; cuts L2 atomic traffic by `gridDim.x` for the hot
//!   partitions). The host-side dispatcher
//!   [`compile_partition_kernel_i64_for_n_rows`] picks between them.
//! * `counts[]` is still u32-indexed (per-partition row count, capped well
//!   under `u32::MAX` for any realistic Tier-2 input). The 64-bit width is
//!   in the key column only.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

/// Number of hash partitions. MUST match the i32 sibling so the host-side
/// orchestrator can use the same `compute_partition_offsets` and Tier-2
/// invariants ("each partition fits in shmem"). MUST be a power of two so
/// the `shr.u64` extraction yields a partition id in `[0, NUM_PARTITIONS)`
/// without an explicit mask.
pub const NUM_PARTITIONS: u32 = 4096;

/// 64-bit Fibonacci / Knuth multiplicative-hash constant —
/// `floor(2^64 / phi)` rounded to the nearest odd integer. Mixes the low
/// bits of structured inputs up into the high bits where we'll extract
/// them via `shr.u64` for the partition id.
pub const HASH_MULTIPLIER_64: u64 = 0x9E37_79B9_7F4A_7C15;

/// Threads per block. Matches the i32 sibling so launch geometry is shared.
pub const BLOCK_THREADS: u32 = 256;

/// Entry-point name embedded in the emitted PTX. Distinct from the i32
/// sibling so both modules can be loaded into the same CUDA context.
///
/// Both kernel variants ([`compile_partition_kernel_i64_global_atomics`] and
/// [`compile_partition_kernel_i64_shmem_staging`]) export this same symbol
/// name so a single `function(KERNEL_ENTRY)` lookup works regardless of
/// which variant the host dispatched. The two variants are interchangeable
/// from the launch site's perspective: same signature, same outputs, same
/// launch geometry.
pub const KERNEL_ENTRY: &str = "bolt_partition_i64";

/// Threshold below which the dispatcher picks the global-atomics kernel
/// over the shared-memory-staging kernel.
///
/// For `n_rows < NUM_PARTITIONS * 2`, the shmem-staging kernel's fixed
/// zero-init + flush overhead (`2 * NUM_PARTITIONS` shared-memory writes
/// + `NUM_PARTITIONS` global atomics per block, regardless of `n_rows`)
/// dominates the saving on global-atomic traffic from collapsing the
/// per-row `atom.global.add.u32` into a per-(block, partition)
/// accumulator. At 4096 partitions and BLOCK_THREADS = 256 the
/// per-block overhead is 16 partitions per thread for both the zero-init
/// and the flush; for very small inputs that's strictly more work than
/// the original kernel does in total.
///
/// The threshold is set conservatively at `2 * NUM_PARTITIONS = 8192`
/// rows: above this, the shmem-staging kernel's amortised cost is
/// strictly lower; below it, the global-atomics kernel wins. Kept
/// numerically identical to the i32 sibling's threshold so the dispatch
/// decision is uniform regardless of key width.
pub const SHMEM_STAGING_MIN_ROWS: u32 = NUM_PARTITIONS * 2;

/// Generate PTX for the **i64-key** hash-partition pass-1 kernel —
/// back-compat wrapper.
///
/// Equivalent to [`compile_partition_kernel_i64_global_atomics`], retained
/// so existing callers (and existing PTX-shape tests) that compiled the
/// global-atomics kernel via the unparameterised name continue to work
/// unchanged. New call sites should prefer the host-side dispatcher
/// [`compile_partition_kernel_i64_for_n_rows`] which selects the right
/// variant for the input size.
///
/// Kernel signature (PTX-level):
///
/// ```text
/// .visible .entry bolt_partition_i64(
///     .param .u64 keys_ptr,           // const int64_t*  keys[n_rows]
///     .param .u64 partition_ids_ptr,  //       uint32_t* partition_ids[n_rows]
///     .param .u64 counts_ptr,         //       uint32_t* counts[NUM_PARTITIONS]
///     .param .u32 n_rows
/// )
/// ```
///
/// `counts` MUST be zero-initialised by the caller before launch.
///
/// Deterministic and pure: same input → same output, no I/O.
pub fn compile_partition_kernel_i64() -> BoltResult<String> {
    compile_partition_kernel_i64_global_atomics()
}

/// Pick the i64-key partition-kernel PTX best suited to `n_rows`.
///
/// For `n_rows < SHMEM_STAGING_MIN_ROWS` the global-atomics kernel wins
/// (the shmem-staging zero-init + flush overhead would otherwise
/// dominate). For larger inputs the shmem-staging kernel cuts global
/// atomic traffic from `n_rows` down to `n_blocks * NUM_PARTITIONS`,
/// which is the limiting factor on the L2 atomic unit for dense Tier-2
/// workloads (q3 two-key / q5 two-key).
///
/// Returns PTX for a kernel that exports the same [`KERNEL_ENTRY`]
/// symbol regardless of which variant was selected — the launch site
/// does not need to branch on the choice.
pub fn compile_partition_kernel_i64_for_n_rows(n_rows: u32) -> BoltResult<String> {
    if n_rows < SHMEM_STAGING_MIN_ROWS {
        compile_partition_kernel_i64_global_atomics()
    } else {
        compile_partition_kernel_i64_shmem_staging()
    }
}

/// Hand-emit the original "global atomics" i64-key partition kernel.
///
/// Kernel signature (PTX-level):
///
/// ```text
/// .visible .entry bolt_partition_i64(
///     .param .u64 keys_ptr,           // const int64_t*  keys[n_rows]
///     .param .u64 partition_ids_ptr,  //       uint32_t* partition_ids[n_rows]
///     .param .u64 counts_ptr,         //       uint32_t* counts[NUM_PARTITIONS]
///     .param .u32 n_rows
/// )
/// ```
///
/// `counts` MUST be zero-initialised by the caller before launch — the
/// kernel only issues `atom.global.add.u32` against it.
///
/// Deterministic and pure: same input → same output, no I/O.
pub fn compile_partition_kernel_i64_global_atomics() -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY;
    // Shift count = 64 - log2(NUM_PARTITIONS). For K=4096 → 52.
    let shift = 64u32 - (NUM_PARTITIONS.trailing_zeros());
    debug_assert_eq!(shift, 52);

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_3").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Register decls — generous, same convention as the i32 sibling. The
    // SASS pass will trim. f64 not used (no value column in pass-1) but
    // declared for symmetry with the other Tier-2 kernels.
    writeln!(ptx, "\t.reg .pred  %p<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<32>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // --- thread coordinates -------------------------------------------------
    //   %r3 = gtid    = blockIdx.x * blockDim.x + threadIdx.x
    //   %r5 = stride  = gridDim.x * blockDim.x
    //   %r6 = n_rows  (loaded from .param)
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r4, %nctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s32 %r5, %r4, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r6, [{entry}_param_3];").map_err(write_err)?;

    // --- global pointer setup ---------------------------------------------
    //   %rd0 = keys (i64*),   %rd1 = partition_ids (u32*),
    //   %rd2 = counts (u32*).
    writeln!(ptx, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd1, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd2, [{entry}_param_2];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // 64-bit multiplier materialised once outside the loop. PTX requires the
    // explicit `0x9E3779B97F4A7C15` literal in a register before
    // `mul.lo.u64`; folding a 64-bit immediate is unreliable across
    // toolchains.
    writeln!(
        ptx,
        "\tmov.u64 %rd9, 0x{mult:016X};",
        mult = HASH_MULTIPLIER_64
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ----------------------------------------------------------------------
    // Grid-stride loop:
    //
    //   key   = keys[i]                       // ld.global.s64 (8-byte)
    //   hash  = (u64)key * 0x9E3779B97F4A7C15 // mul.lo.u64, wraps mod 2^64
    //   pid   = (u32)(hash >> 52)             // top log2(K) bits = partition id
    //   atom.global.add.u32 [counts + pid*4], 1
    //   partition_ids[i] = pid                // st.global.u32
    // ----------------------------------------------------------------------
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r6;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra LOOP_DONE;").map_err(write_err)?;

    // key = keys[i]   (i64 load, stride 8)
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd0, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rd20, [%rd11];").map_err(write_err)?;

    // hash = (u64)key * HASH_MULTIPLIER_64    (mul.lo.u64 wraps to low 64 bits)
    writeln!(ptx, "\tmul.lo.u64 %rd21, %rd20, %rd9;").map_err(write_err)?;

    // pid_u64 = hash >> 52    (top log2(K)=12 bits of the multiplied product)
    writeln!(
        ptx,
        "\tshr.u64 %rd22, %rd21, {shift};",
        shift = shift
    )
    .map_err(write_err)?;
    // Narrow to u32 for the atomic + the per-row store.
    writeln!(ptx, "\tcvt.u32.u64 %r10, %rd22;").map_err(write_err)?;

    // atomicAdd(&counts[pid], 1)
    writeln!(ptx, "\tmul.wide.u32 %rd12, %r10, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd13, %rd2, %rd12;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r12, 1;").map_err(write_err)?;
    writeln!(ptx, "\tatom.global.add.u32 %r11, [%rd13], %r12;").map_err(write_err)?;

    // partition_ids[i] = pid
    writeln!(ptx, "\tmul.wide.u32 %rd14, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd15, %rd1, %rd14;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd15], %r10;").map_err(write_err)?;

    // i += stride; loop.
    writeln!(ptx, "\tadd.u32 %r3, %r3, %r5;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Hand-emit the **shared-memory-staging** i64-key partition kernel.
///
/// Same signature and exported symbol as
/// [`compile_partition_kernel_i64_global_atomics`] — the launch site does
/// not have to change. The pipeline is:
///
/// 1. Each block keeps its private `block_counts[NUM_PARTITIONS]` (u32)
///    in shared memory (16 KiB at NUM_PARTITIONS = 4096). The shape is
///    identical to the i32 sibling's shmem-staging variant; only the
///    key-side load + hash differs.
/// 2. All 256 threads collaboratively zero `block_counts`. Each thread
///    handles `NUM_PARTITIONS / BLOCK_THREADS = 16` slots in a small
///    unrolled loop. `bar.sync 0` after.
/// 3. Grid-stride loop: each row issues `atom.shared.add.u32` against
///    its partition's slot in `block_counts`. Shared atomic adds remain
///    atomic but never go through L2 — they're orders of magnitude
///    cheaper than `atom.global.add.u32` under contention.
/// 4. `bar.sync 0` to make every shared add globally visible inside
///    the block.
/// 5. Flush: each thread loops over its 16 slots in `block_counts` and
///    issues `atom.global.add.u32 counts[pid], block_counts[pid]`. The
///    global atomic count drops from `n_rows` to
///    `n_blocks * NUM_PARTITIONS`.
///
/// `counts` MUST be zero-initialised by the caller before launch — the
/// final flush only *adds* into it.
///
/// Kernel signature (PTX-level): identical to the global-atomics
/// variant — see the doc on [`compile_partition_kernel_i64_global_atomics`].
///
/// Deterministic and pure: same input → same output, no I/O.
pub fn compile_partition_kernel_i64_shmem_staging() -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY;
    let block_threads = BLOCK_THREADS;
    let num_partitions = NUM_PARTITIONS;
    // Shift count = 64 - log2(NUM_PARTITIONS). For K=4096 → 52.
    let shift = 64u32 - (NUM_PARTITIONS.trailing_zeros());
    debug_assert_eq!(shift, 52);
    // Per-thread fan-out for zero-init and flush: each thread handles
    // exactly this many of the NUM_PARTITIONS partition slots. With
    // 4096 partitions and 256 threads, each thread does 16 slots.
    debug_assert!(num_partitions % block_threads == 0);
    let slots_per_thread = num_partitions / block_threads;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Block-private counts table, NUM_PARTITIONS u32s = 16 KiB. The
    // .align 4 keeps each 4-byte slot naturally aligned so shared
    // atomics land at addressable u32 offsets without bank-mux fixups.
    writeln!(
        ptx,
        ".shared .align 4 .b32 block_counts[{n}];",
        n = num_partitions
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_3").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Register decls. We need a few more than the global-atomics variant
    // because the zero-init and flush phases each carry their own
    // induction registers.
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<48>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<48>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // --- thread coordinates -------------------------------------------------
    // %r0 = blockIdx.x, %r1 = blockDim.x, %r2 = threadIdx.x
    // %r3 = gtid       = blockIdx.x * blockDim.x + threadIdx.x
    // %r4 = gridDim.x
    // %r5 = stride     = gridDim.x * blockDim.x
    // %r6 = n_rows
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r4, %nctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s32 %r5, %r4, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r6, [{entry}_param_3];").map_err(write_err)?;

    // --- global pointer setup (cvta from .param) ---------------------------
    writeln!(ptx, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd1, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd2, [{entry}_param_2];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Materialise &block_counts[0] once. We use plain `mov.u64`, NOT
    // `cvta.to.shared.u64` — the resulting %rd3 is then used downstream
    // with `st.shared.u32` / `atom.shared.add.u32` / `ld.shared.u32`,
    // which all expect a shared-state-space address. `cvta.shared.u64`
    // would produce a generic-space address and require a matching
    // state-space change at every shared op below. Same pattern as
    // `agg_kernels.rs` and the partition-reduce kernels.
    writeln!(ptx, "\tmov.u64 %rd3, block_counts;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // 64-bit multiplier materialised once outside the loop. PTX requires
    // the explicit `0x9E3779B97F4A7C15` literal in a register before
    // `mul.lo.u64`; folding a 64-bit immediate is unreliable across
    // toolchains. Same materialisation as the global-atomics variant.
    writeln!(
        ptx,
        "\tmov.u64 %rd9, 0x{mult:016X};",
        mult = HASH_MULTIPLIER_64
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ----------------------------------------------------------------------
    // Phase 1: zero-init block_counts.
    //
    // Each thread writes `slots_per_thread` u32s with stride `block_threads`,
    // so consecutive threads write consecutive slots and there are no
    // bank conflicts. The pattern is:
    //
    //   for (k = 0; k < slots_per_thread; k++) {
    //       block_counts[tid + k * block_threads] = 0;
    //   }
    //
    // We unroll the loop because slots_per_thread is a small compile-time
    // constant (16 at the default geometry); unrolled code keeps the
    // induction register chain short and lets ptxas schedule the stores
    // back-to-back.
    // ----------------------------------------------------------------------
    writeln!(ptx, "\tmov.u32 %r13, 0;").map_err(write_err)?; // zero literal
    for k in 0..slots_per_thread {
        // slot_index = tid + k * block_threads
        let offset = k * block_threads;
        writeln!(
            ptx,
            "\tadd.u32 %r14, %r2, {off};",
            off = offset
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tmul.wide.u32 %rd20, %r14, 4;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd21, %rd3, %rd20;").map_err(write_err)?;
        writeln!(ptx, "\tst.shared.u32 [%rd21], %r13;").map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ----------------------------------------------------------------------
    // Phase 2: grid-stride loop, accumulate into shared block_counts.
    //
    //   key  = keys[i]                       // ld.global.s64 (8-byte)
    //   hash = (u64)key * HASH_MULTIPLIER_64  // mul.lo.u64
    //   pid  = (u32)(hash >> 52)              // top log2(K) bits
    //   atom.shared.add.u32 [block_counts + pid*4], 1
    //   partition_ids[i] = pid
    //
    // Only the key load + hash differs from the i32 sibling — the
    // counts side stays u32-indexed by partition id.
    // ----------------------------------------------------------------------
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r6;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra LOOP_DONE;").map_err(write_err)?;

    // key = keys[i]   (i64 load, stride 8)
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd0, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rd23, [%rd11];").map_err(write_err)?;

    // hash = (u64)key * HASH_MULTIPLIER_64    (mul.lo.u64 wraps to low 64 bits)
    writeln!(ptx, "\tmul.lo.u64 %rd24, %rd23, %rd9;").map_err(write_err)?;

    // pid_u64 = hash >> 52    (top log2(K)=12 bits of the multiplied product)
    writeln!(
        ptx,
        "\tshr.u64 %rd25, %rd24, {shift};",
        shift = shift
    )
    .map_err(write_err)?;
    // Narrow to u32 for the atomic + the per-row store.
    writeln!(ptx, "\tcvt.u32.u64 %r10, %rd25;").map_err(write_err)?;

    // atomicAdd(&block_counts[pid], 1). Shared atomic add — never hits L2.
    // The destination register %r11 captures the pre-increment value
    // (required syntactically) but is never read.
    writeln!(ptx, "\tmul.wide.u32 %rd12, %r10, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd13, %rd3, %rd12;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r12, 1;").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.u32 %r11, [%rd13], %r12;").map_err(write_err)?;

    // partition_ids[i] = pid
    writeln!(ptx, "\tmul.wide.u32 %rd14, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd15, %rd1, %rd14;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd15], %r10;").map_err(write_err)?;

    // i += stride; loop.
    writeln!(ptx, "\tadd.u32 %r3, %r3, %r5;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Ensure every shared add in this block is visible to the flush phase.
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ----------------------------------------------------------------------
    // Phase 3: flush block_counts → global counts.
    //
    //   for (k = 0; k < slots_per_thread; k++) {
    //       slot = tid + k * block_threads;
    //       v    = block_counts[slot];        // ld.shared.u32
    //       atom.global.add.u32 counts[slot], v;
    //   }
    //
    // We could skip the atomic when v == 0 to spare the address ALU on
    // empty partitions, but the predicate cost roughly matches the saved
    // atomic on the L2 pre-coalescer (atomics with value=0 are special-
    // cased on Volta+). Issue the unconditional add; it keeps the code
    // shape simple and the worst case bounded.
    // ----------------------------------------------------------------------
    for k in 0..slots_per_thread {
        let offset = k * block_threads;
        // slot_index = tid + k * block_threads
        writeln!(
            ptx,
            "\tadd.u32 %r20, %r2, {off};",
            off = offset
        )
        .map_err(write_err)?;
        // shared_addr = &block_counts[slot_index]
        writeln!(ptx, "\tmul.wide.u32 %rd30, %r20, 4;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd31, %rd3, %rd30;").map_err(write_err)?;
        writeln!(ptx, "\tld.shared.u32 %r21, [%rd31];").map_err(write_err)?;
        // global_addr = &counts[slot_index]
        writeln!(ptx, "\tmul.wide.u32 %rd32, %r20, 4;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd33, %rd2, %rd32;").map_err(write_err)?;
        // atom.global.add.u32 — discard the destination.
        writeln!(ptx, "\tatom.global.add.u32 %r22, [%rd33], %r21;").map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("partition_kernel_i64: write failed: {}", e))
}

// ---------------------------------------------------------------------------
// PTX-shape tests (host-only — no GPU required).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// The 64-bit Knuth/Fibonacci constant MUST appear verbatim in PTX.
    /// Regressing to the 32-bit `0x9E3779B1` (or worse, a sign-extended
    /// truncation) would alias every two-key pack into a handful of
    /// partitions.
    #[test]
    fn contains_hash_multiplier_64_literal() {
        let ptx = compile_partition_kernel_i64().expect("kernel compiles");
        assert!(
            ptx.contains("0x9E3779B97F4A7C15"),
            "PTX must contain the 64-bit Fibonacci multiplier 0x9E3779B97F4A7C15:\n{ptx}"
        );
    }

    /// Cross-block counting requires `atom.global.add.u32` — same
    /// instruction the i32 sibling uses (counts stay u32-indexed by
    /// partition id).
    #[test]
    fn uses_atom_global_add_u32() {
        let ptx = compile_partition_kernel_i64().expect("kernel compiles");
        assert!(
            ptx.contains("atom.global.add.u32"),
            "PTX must use atom.global.add.u32 for the per-partition counter:\n{ptx}"
        );
    }

    /// Entry name must be the i64-specific symbol so the dispatcher can
    /// resolve it without colliding with the i32 sibling when both modules
    /// are loaded into the same CUDA context.
    #[test]
    fn has_correct_entry_name() {
        let ptx = compile_partition_kernel_i64().expect("kernel compiles");
        let needle = format!(".visible .entry {}(", KERNEL_ENTRY);
        assert!(
            ptx.contains(&needle),
            "PTX must declare .visible .entry {}(  — got:\n{ptx}",
            KERNEL_ENTRY
        );
        assert_eq!(KERNEL_ENTRY, "bolt_partition_i64");
    }

    /// The partition id is the TOP 12 bits of the 64-bit multiplied product,
    /// extracted via an unsigned right shift. A regression that swaps in
    /// `shr.s64` (signed) or drops the shift entirely (reading raw low bits)
    /// would silently destroy hash quality.
    #[test]
    fn uses_unsigned_shr_for_high_bits() {
        let ptx = compile_partition_kernel_i64().expect("kernel compiles");
        // Accept either spelling the PTX assembler may produce; we use
        // shr.u64 explicitly in the emitter.
        let has_u64 = ptx.contains("shr.u64") || ptx.contains("shr.b64");
        assert!(
            has_u64,
            "PTX must use shr.u64 / shr.b64 to extract the top bits as pid:\n{ptx}"
        );
    }

    /// `compile_partition_kernel_i64()` must succeed and return a non-empty
    /// PTX string with the expected header. Cheapest possible regression
    /// guard.
    #[test]
    fn compiles_to_non_empty_ptx() {
        let ptx = compile_partition_kernel_i64().expect("kernel compiles");
        assert!(!ptx.is_empty(), "compile returned empty PTX");
        assert!(
            ptx.contains(".version 7.5") && ptx.contains(".target sm_70"),
            "PTX must include the standard Craton Bolt module header:\n{ptx}"
        );
    }

    // -----------------------------------------------------------------------
    // Shared-memory-staging variant tests (mirror of the i32 sibling).
    //
    // The shmem-staging kernel exports the same entry symbol as the
    // global-atomics kernel, so the launch site doesn't care which it
    // gets back. The contract that DOES differ — and these tests pin —
    // is the per-row atomic being a *shared* atomic, plus the presence
    // of the zero-init phase + post-loop flush back to global, plus the
    // i64 key load.
    // -----------------------------------------------------------------------

    /// The shmem-staging kernel must declare a 16 KiB shared-memory
    /// staging buffer. Regressing this (e.g. accidentally allocating
    /// dynamic shared memory or dropping the declaration) would silently
    /// break the kernel — the shared atomics would land at address 0.
    #[test]
    fn shmem_staging_declares_block_counts() {
        let ptx = compile_partition_kernel_i64_shmem_staging().expect("kernel compiles");
        let needle = format!(".shared .align 4 .b32 block_counts[{}]", NUM_PARTITIONS);
        assert!(
            ptx.contains(&needle),
            "shmem-staging PTX must declare block_counts[NUM_PARTITIONS]:\n{ptx}"
        );
    }

    /// The per-row counter in the shmem-staging kernel MUST be a shared
    /// atomic, not a global one. A regression that swaps in
    /// `atom.global.add.u32` for the per-row increment would defeat the
    /// whole optimisation (and still produce correct output, so this
    /// kind of regression is exactly what we need a substring check for).
    #[test]
    fn shmem_staging_uses_atom_shared_add_u32() {
        let ptx = compile_partition_kernel_i64_shmem_staging().expect("kernel compiles");
        assert!(
            ptx.contains("atom.shared.add.u32"),
            "shmem-staging PTX must use atom.shared.add.u32 for the per-row counter:\n{ptx}"
        );
    }

    /// The flush phase MUST still issue a global atomic add — that's
    /// how the per-block private counts get merged into the shared
    /// global `counts[]` array that the host downloads.
    #[test]
    fn shmem_staging_flushes_via_atom_global_add_u32() {
        let ptx = compile_partition_kernel_i64_shmem_staging().expect("kernel compiles");
        assert!(
            ptx.contains("atom.global.add.u32"),
            "shmem-staging PTX must end with atom.global.add.u32 to flush \
             block_counts into the per-grid counts[]:\n{ptx}"
        );
    }

    /// Both the zero-init phase (between block_counts setup and the loop)
    /// AND the post-loop flush MUST be preceded by a `bar.sync 0`. Without
    /// the second barrier, racing threads would flush a still-being-
    /// updated block_counts.
    #[test]
    fn shmem_staging_has_two_barsync_barriers() {
        let ptx = compile_partition_kernel_i64_shmem_staging().expect("kernel compiles");
        let bar_count = ptx.matches("bar.sync 0").count();
        assert!(
            bar_count >= 2,
            "shmem-staging PTX must emit >=2 bar.sync 0 (post-zero-init + \
             post-grid-stride-loop); saw {bar_count}:\n{ptx}"
        );
    }

    /// The shmem-staging variant must still load i64 keys via
    /// `ld.global.s64` and apply the 64-bit Knuth multiplier. A regression
    /// that copy-pasted the i32 sibling's 32-bit pipeline here would
    /// silently corrupt every two-key partition assignment.
    #[test]
    fn shmem_staging_keeps_i64_hash_pipeline() {
        let ptx = compile_partition_kernel_i64_shmem_staging().expect("kernel compiles");
        assert!(
            ptx.contains("ld.global.s64"),
            "shmem-staging PTX must load i64 keys via ld.global.s64:\n{ptx}"
        );
        assert!(
            ptx.contains("0x9E3779B97F4A7C15"),
            "shmem-staging PTX must use the 64-bit Knuth multiplier:\n{ptx}"
        );
        let has_u64_shr = ptx.contains("shr.u64") || ptx.contains("shr.b64");
        assert!(
            has_u64_shr,
            "shmem-staging PTX must extract top bits via shr.u64 / shr.b64:\n{ptx}"
        );
    }

    /// Dispatcher: small inputs route to the global-atomics variant
    /// (the shmem-staging fixed overhead dominates below the threshold).
    #[test]
    fn dispatcher_small_input_picks_global_atomics() {
        let ptx = compile_partition_kernel_i64_for_n_rows(SHMEM_STAGING_MIN_ROWS - 1)
            .expect("dispatcher compiles");
        // Global-atomics kernel never touches shared memory.
        assert!(
            !ptx.contains(".shared"),
            "small-n_rows dispatch must pick the global-atomics kernel \
             (no .shared declarations expected):\n{ptx}"
        );
        assert!(
            !ptx.contains("atom.shared.add.u32"),
            "small-n_rows dispatch must NOT use shared atomics:\n{ptx}"
        );
    }

    /// Dispatcher: large inputs route to the shmem-staging variant.
    #[test]
    fn dispatcher_large_input_picks_shmem_staging() {
        let ptx = compile_partition_kernel_i64_for_n_rows(SHMEM_STAGING_MIN_ROWS)
            .expect("dispatcher compiles");
        assert!(
            ptx.contains("atom.shared.add.u32"),
            "large-n_rows dispatch must pick the shmem-staging kernel \
             (atom.shared.add.u32 expected):\n{ptx}"
        );
    }

    /// Backwards-compatibility: the unparameterised
    /// `compile_partition_kernel_i64()` wrapper must still emit the
    /// global-atomics variant. Existing callers in the orchestrators
    /// depend on this name continuing to resolve to the original PTX
    /// shape until they switch to the n_rows-aware dispatcher.
    #[test]
    fn back_compat_wrapper_still_emits_global_atomics() {
        let wrapper = compile_partition_kernel_i64().expect("compile");
        let global = compile_partition_kernel_i64_global_atomics().expect("compile");
        assert_eq!(
            wrapper, global,
            "compile_partition_kernel_i64() must alias \
             compile_partition_kernel_i64_global_atomics() for back-compat"
        );
    }
}
