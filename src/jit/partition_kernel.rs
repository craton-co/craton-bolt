// SPDX-License-Identifier: Apache-2.0

//! Hash-partition pass-1 kernel (Tier 2 of the GROUP BY perf plan).
//!
//! ## Why this exists
//!
//! Tier 1 (`shmem_sum_kernel`) fixes low-cardinality GROUP BY by reducing
//! global atomic contention through per-block shared-memory pre-aggregation.
//! It does **not** help when the group count exceeds what fits in shared
//! memory (q3 ~1 M two-key groups, q5 1 M groups): the block-local table
//! overflows and the kernel degrades to single-global-hash-table behaviour.
//!
//! Tier 2 attacks this case with a two-pass scheme: first partition the
//! input rows by `hash(key) % K` into K disjoint buckets, then run the
//! Tier-1 block-local groupby once per partition. Each per-partition
//! hashtable has at most `n_groups / K` distinct keys — small enough to
//! live in L2.
//!
//! This file is the **pass-1 partition kernel**: count how many rows land
//! in each of the K = 4096 partitions and remember which partition each row
//! was assigned to. A sibling scatter kernel (written by another agent) then
//! uses the counts (after a prefix-sum) and the per-row partition ids to
//! actually rearrange the rows. Keeping count + assignment in one kernel
//! avoids re-hashing every row twice.
//!
//! ## Algorithm
//!
//! ```text
//! const uint32_t K              = 4096;          // power of 2 → mask, not mod
//! const uint32_t HASH_MULTIPLIER = 0x9E3779B1;   // Knuth multiplicative
//!
//! for (i = blockIdx.x*blockDim.x + tid; i < n_rows; i += gridDim.x*blockDim.x) {
//!     int32_t  key  = keys[i];
//!     uint32_t hash = (uint32_t)key * HASH_MULTIPLIER;
//!     uint32_t pid  = hash & (K - 1);            // == hash % K
//!     atomicAdd(&counts[pid], 1u);               // discard return value
//!     partition_ids[i] = pid;
//! }
//! ```
//!
//! ## PTX-level notes
//!
//! * `atom.global.add.u32` (no result register written) is the cheapest form
//!   of increment-and-discard. We still need a destination register because
//!   PTX requires one syntactically; the value is simply never read.
//! * The mask form `and.b32 %r_pid, %r_hash, 0xFFF` replaces `rem.u32`
//!   because `NUM_PARTITIONS = 4096 = 2^12`. This avoids the expensive
//!   software-emulated integer modulo SASS sequence.
//! * Knuth's multiplicative hash (`x * 2654435761u32`) is emitted as a
//!   single `mul.lo.u32`. We deliberately want the wrapping low 32 bits;
//!   `mul.lo` is precisely that.
//! * Bank-conflict / contention notes:
//!   - `counts[]` has 4096 slots. The original kernel (preserved here as
//!     [`compile_partition_kernel_global_atomics`]) issues one
//!     `atom.global.add.u32` per row, which serialises on the L2 atomic
//!     unit when many threads land in the same partition.
//!   - The [`compile_partition_kernel_shmem_staging`] variant stages
//!     `counts[]` in shared memory: each block keeps a private 16 KiB
//!     `block_counts[NUM_PARTITIONS]`, all per-row increments hit shared
//!     atomics (which never go through L2), and at the end the block
//!     flushes its private counts back to global with one
//!     `atom.global.add.u32 counts[pid], block_counts[pid]` per partition
//!     per block. Global atomic traffic drops from `n_rows` to
//!     `n_blocks * NUM_PARTITIONS`, which for `n_rows >> NUM_PARTITIONS`
//!     is a large win. The dispatcher selects between the two variants
//!     based on `n_rows`: very small inputs prefer the global-atomic
//!     kernel (the flush phase's `n_blocks * NUM_PARTITIONS` overhead
//!     dominates when `n_rows` is small).
//!   - The `st.global.u32` to `partition_ids[i]` is fully coalesced (one
//!     warp writes 32 contiguous u32s per step). No hazard there.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

/// Number of hash partitions. MUST be a power of two so `% NUM_PARTITIONS`
/// can be implemented as `and.b32 %r, %r, (NUM_PARTITIONS - 1)`.
///
/// **Sized to 4096** so each partition holds ~250 distinct keys at the
/// 1 M-group target (q5). The Tier-2.1 per-partition reduce kernel uses a
/// `BLOCK_GROUPS = 1024`-slot open-addressing table per block; at 4096
/// partitions the load factor is ~25 %, giving near-zero probe chains.
/// Bumping from the original 1024 → 4096 traded a slightly larger
/// partition-counts array (16 KiB on device, host download is 16 KiB ≈
/// 50 µs) for a much-better-behaved per-partition hash table.
pub const NUM_PARTITIONS: u32 = 4096;

/// Knuth's multiplicative hash constant: `floor(2^32 / phi)` rounded to
/// the nearest odd integer. Mixes the low bits of small integer keys
/// (which would otherwise alias hard against a power-of-two mask) up into
/// the high bits where the mask actually sees them.
pub const HASH_MULTIPLIER: u32 = 0x9E37_79B1;

/// Threads per block for the partition kernel. Matches `shmem_sum_kernel`
/// so the dispatcher launch geometry doesn't have to track per-kernel
/// block sizes.
pub const BLOCK_THREADS: u32 = 256;

/// Entry-point name embedded in the emitted PTX. The dispatcher resolves
/// this via `cuModuleGetFunction` after `cuModuleLoadDataEx`.
///
/// Both kernel variants ([`compile_partition_kernel_global_atomics`] and
/// [`compile_partition_kernel_shmem_staging`]) export this same symbol
/// name so a single `function(KERNEL_ENTRY)` lookup works regardless of
/// which variant the host dispatched. The two variants are
/// interchangeable from the launch site's perspective: same signature,
/// same outputs, same launch geometry.
pub const KERNEL_ENTRY: &str = "bolt_partition";

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
/// strictly lower; below it, the global-atomics kernel wins.
pub const SHMEM_STAGING_MIN_ROWS: u32 = NUM_PARTITIONS * 2;

/// Generate PTX for the hash-partition pass-1 kernel — back-compat wrapper.
///
/// Equivalent to [`compile_partition_kernel_global_atomics`], retained so
/// existing callers (and existing PTX golden tests) that compiled the
/// global-atomics kernel via the unparameterised name continue to work
/// unchanged. New call sites should prefer the host-side dispatcher
/// [`compile_partition_kernel_for_n_rows`] which selects the right
/// variant for the input size.
#[cfg(not(feature = "rust-cuda"))]
pub fn compile_partition_kernel() -> BoltResult<String> {
    compile_partition_kernel_global_atomics()
}

/// Pick the partition-kernel PTX best suited to `n_rows`.
///
/// For `n_rows < SHMEM_STAGING_MIN_ROWS` the global-atomics kernel wins
/// (the shmem-staging zero-init + flush overhead would otherwise
/// dominate). For larger inputs the shmem-staging kernel cuts global
/// atomic traffic from `n_rows` down to `n_blocks * NUM_PARTITIONS`,
/// which is the limiting factor on the L2 atomic unit for dense Tier-2
/// workloads (q3 / q5).
///
/// Returns PTX for a kernel that exports the same [`KERNEL_ENTRY`]
/// symbol regardless of which variant was selected — the launch site
/// does not need to branch on the choice.
#[cfg(not(feature = "rust-cuda"))]
pub fn compile_partition_kernel_for_n_rows(n_rows: u32) -> BoltResult<String> {
    if n_rows < SHMEM_STAGING_MIN_ROWS {
        compile_partition_kernel_global_atomics()
    } else {
        compile_partition_kernel_shmem_staging()
    }
}

/// `rust-cuda` feature variant — see the non-rust-cuda doc on
/// [`compile_partition_kernel_global_atomics`] for the algorithm and the
/// kernel signature. When the `rust-cuda` feature is on, the PTX is
/// compiled at build time from `kernels/src/lib.rs` via `cuda_builder` +
/// `rustc_codegen_nvvm` and we just emit the embedded artefact here.
/// The shmem-staging variant is hand-emit-only — under `rust-cuda` we
/// fall back to the single embedded global-atomics PTX.
///
/// See docs/rust_cuda/03_partition_kernel_spike.md and
/// docs/rust_cuda/08_wave_a_outcome.md.
#[cfg(feature = "rust-cuda")]
pub fn compile_partition_kernel() -> BoltResult<String> {
    // Compiled by build.rs via cuda_builder when --features rust-cuda is on.
    // Layout: kernels/src/lib.rs --rustc_codegen_nvvm--> $OUT_DIR/partition.ptx.
    const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/partition.ptx"));

    if PTX.is_empty() {
        // Defensive: build.rs must have populated the file when the
        // feature is on. An empty string means cuda_builder silently
        // failed or the feature gate logic in build.rs is broken.
        return Err(BoltError::Other(
            "partition_kernel: rust-cuda PTX artefact is empty — \
             cuda_builder did not produce a valid PTX file. See \
             docs/rust_cuda/08_wave_a_outcome.md."
                .to_string(),
        ));
    }

    Ok(PTX.to_owned())
}

/// `rust-cuda` mirror of [`compile_partition_kernel_for_n_rows`]. The
/// rust-cuda path ships a single embedded PTX; under that feature both
/// "variants" map to the same artefact, so the dispatcher choice is a
/// no-op. See the `#[cfg(feature = "rust-cuda")]` doc on
/// [`compile_partition_kernel`].
#[cfg(feature = "rust-cuda")]
pub fn compile_partition_kernel_for_n_rows(_n_rows: u32) -> BoltResult<String> {
    compile_partition_kernel()
}

/// `rust-cuda` mirror of [`compile_partition_kernel_global_atomics`].
/// Same artefact as [`compile_partition_kernel`] — see that doc.
#[cfg(feature = "rust-cuda")]
pub fn compile_partition_kernel_global_atomics() -> BoltResult<String> {
    compile_partition_kernel()
}

/// `rust-cuda` mirror of [`compile_partition_kernel_shmem_staging`]. The
/// hand-emit shmem-staging PTX has no rust-cuda counterpart yet; under
/// the feature we fall back to the global-atomics artefact. The
/// host-side dispatcher will therefore always select the latter under
/// `rust-cuda`, which is the safe default until Wave B.
#[cfg(feature = "rust-cuda")]
pub fn compile_partition_kernel_shmem_staging() -> BoltResult<String> {
    compile_partition_kernel()
}

/// Hand-emit the original "global atomics" partition kernel.
///
/// Kernel signature (PTX-level):
///
/// ```text
/// .visible .entry bolt_partition(
///     .param .u64 keys_ptr,           // const int32_t*  keys[n_rows]
///     .param .u64 partition_ids_ptr,  //       uint32_t* partition_ids[n_rows]
///     .param .u64 counts_ptr,         //       uint32_t* counts[NUM_PARTITIONS]
///     .param .u32 n_rows
/// )
/// ```
///
/// `counts` MUST be zero-initialised by the caller before launch — the
/// kernel only issues `atom.global.add.u32` against it.
///
/// Deterministic and pure: same input → same output, no I/O. The
/// dispatcher can cache the result indefinitely.
#[cfg(not(feature = "rust-cuda"))]
pub fn compile_partition_kernel_global_atomics() -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY;
    let mask = NUM_PARTITIONS - 1; // 0xFFF for NUM_PARTITIONS = 4096
    let block_threads = BLOCK_THREADS;
    let _ = block_threads; // launch geometry constant; not emitted into PTX

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

    // Register decls. Counts are generous and mirror `shmem_sum_kernel` so a
    // reader switching between the two files doesn't have to re-learn the
    // register naming convention.
    writeln!(ptx, "\t.reg .pred  %p<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<32>;").map_err(write_err)?;
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
    // %rd0 = keys (i32*), %rd1 = partition_ids (u32*), %rd2 = counts (u32*).
    writeln!(ptx, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd1, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd2, [{entry}_param_2];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Constant: Knuth multiplier. Materialised in %r7 once outside the loop;
    // the assembler doesn't reliably fold a 32-bit immediate into `mul.lo.u32`
    // on every toolchain we target.
    writeln!(ptx, "\tmov.u32 %r7, 0x{mult:08X};", mult = HASH_MULTIPLIER).map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ----------------------------------------------------------------------
    // Grid-stride loop:
    //
    //   key   = keys[i]                       // ld.global.s32
    //   hash  = (u32)key * 0x9E3779B1         // mul.lo.u32 with wrap
    //   pid   = hash & 0xFFF                  // and.b32  (== hash % 4096)
    //   atom.global.add.u32 [counts + pid*4], 1
    //   partition_ids[i] = pid                // st.global.u32
    // ----------------------------------------------------------------------
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r6;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra LOOP_DONE;").map_err(write_err)?;

    // key = keys[i]
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd0, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s32 %r8, [%rd11];").map_err(write_err)?;

    // hash = (uint32_t)key * HASH_MULTIPLIER (low-32 multiply, wraps)
    writeln!(ptx, "\tmul.lo.u32 %r9, %r8, %r7;").map_err(write_err)?;

    // pid = hash & (NUM_PARTITIONS - 1)
    writeln!(ptx, "\tand.b32 %r10, %r9, 0x{mask:X};", mask = mask).map_err(write_err)?;

    // atomicAdd(&counts[pid], 1). The "%r11" destination receives the old
    // value but we never read it — required syntactically by PTX.
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

/// Hand-emit the **shared-memory-staging** partition kernel.
///
/// Same signature and exported symbol as
/// [`compile_partition_kernel_global_atomics`] — the launch site does
/// not have to change. The pipeline is:
///
/// 1. Each block keeps its private `block_counts[NUM_PARTITIONS]` (u32)
///    in shared memory (16 KiB at NUM_PARTITIONS = 4096).
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
/// variant — see the doc on [`compile_partition_kernel_global_atomics`].
///
/// Deterministic and pure: same input → same output, no I/O.
#[cfg(not(feature = "rust-cuda"))]
pub fn compile_partition_kernel_shmem_staging() -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY;
    let mask = NUM_PARTITIONS - 1; // 0xFFF for NUM_PARTITIONS = 4096
    let block_threads = BLOCK_THREADS;
    let num_partitions = NUM_PARTITIONS;
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

    // Knuth multiplier — same as the global-atomics variant.
    writeln!(ptx, "\tmov.u32 %r7, 0x{mult:08X};", mult = HASH_MULTIPLIER).map_err(write_err)?;
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
        writeln!(ptx, "\tadd.u32 %r14, %r2, {off};", off = offset).map_err(write_err)?;
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
    //   key  = keys[i]
    //   hash = (u32)key * HASH_MULTIPLIER
    //   pid  = hash & (NUM_PARTITIONS - 1)
    //   atom.shared.add.u32 [block_counts + pid*4], 1
    //   partition_ids[i] = pid
    // ----------------------------------------------------------------------
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r6;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra LOOP_DONE;").map_err(write_err)?;

    // key = keys[i]
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd0, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s32 %r8, [%rd11];").map_err(write_err)?;

    // hash = (uint32_t)key * HASH_MULTIPLIER
    writeln!(ptx, "\tmul.lo.u32 %r9, %r8, %r7;").map_err(write_err)?;

    // pid = hash & (NUM_PARTITIONS - 1)
    writeln!(ptx, "\tand.b32 %r10, %r9, 0x{mask:X};", mask = mask).map_err(write_err)?;

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
        writeln!(ptx, "\tadd.u32 %r20, %r2, {off};", off = offset).map_err(write_err)?;
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

/// Adapt a `std::fmt::Error` into a `BoltError`. Mirrors the helper in
/// `shmem_sum_kernel.rs` — kept local so the two files stay independent.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("partition_kernel: write failed: {}", e))
}

// ---------------------------------------------------------------------------
// PTX-shape tests (host-only — do NOT require a GPU).
//
// These assert the *hand-emitted* PTX shape (specific opcodes, register
// names, header `.version 7.5`). The rust-cuda variant produces
// semantically equivalent but textually different PTX (NVVM picks its
// own register allocator, header is `.version 7.8+`, etc.), so we gate
// these tests off when `--features rust-cuda` is on. The Wave A outcome
// doc tracks what replaces them on the rust-cuda path.
// ---------------------------------------------------------------------------
#[cfg(all(test, not(feature = "rust-cuda")))]
mod tests {
    use super::*;

    /// Pass-1 counts every row's partition. The only way to do that
    /// correctly across blocks is `atom.global.add.u32` — verify it's
    /// present.
    #[test]
    fn uses_atom_global_add_u32() {
        let ptx = compile_partition_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("atom.global.add.u32"),
            "PTX must use atom.global.add.u32 for the per-partition counter:\n{ptx}"
        );
    }

    /// The Knuth multiplicative constant MUST appear as a literal in the
    /// emitted PTX — either in hex (`0x9E3779B1`) or decimal
    /// (`2654435761`). A regression that swaps the constant would silently
    /// degrade hash quality to "low bits only", aliasing every key into a
    /// handful of partitions.
    #[test]
    fn contains_knuth_multiplier_literal() {
        let ptx = compile_partition_kernel().expect("kernel compiles");
        let has_hex = ptx.contains("0x9E3779B1");
        let has_dec = ptx.contains("2654435761");
        assert!(
            has_hex || has_dec,
            "PTX must contain the Knuth multiplier 0x9E3779B1 / 2654435761:\n{ptx}"
        );
    }

    /// NUM_PARTITIONS is a power of two, so the modulo MUST be a mask, not
    /// a `rem.u32`. Verifying `and.b32` appears guards against a future
    /// refactor accidentally swapping in the (much slower) remainder op.
    #[test]
    fn uses_and_mask_for_modulo() {
        let ptx = compile_partition_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("and.b32"),
            "PTX must use and.b32 for the (NUM_PARTITIONS-1) mask:\n{ptx}"
        );
    }

    /// The kernel must export the well-known entry point name so the
    /// dispatcher can resolve it via `cuModuleGetFunction`.
    #[test]
    fn has_correct_entry_name() {
        let ptx = compile_partition_kernel().expect("kernel compiles");
        let needle = format!(".visible .entry {}(", KERNEL_ENTRY);
        assert!(
            ptx.contains(&needle),
            "PTX must declare .visible .entry {}(  — got:\n{ptx}",
            KERNEL_ENTRY
        );
    }

    /// PTX module preamble must match the rest of `src/jit/*` — same
    /// target, same version. A mismatched header would prevent the driver
    /// from co-loading this module with other Craton Bolt kernels.
    #[test]
    fn ptx_header_matches_project_conventions() {
        let ptx = compile_partition_kernel().expect("kernel compiles");
        assert!(ptx.contains(".version 7.5"), "PTX must be .version 7.5");
        assert!(ptx.contains(".target sm_70"), "PTX must target sm_70");
    }

    /// `compile_partition_kernel` must succeed and produce a non-empty PTX
    /// string. Trivially required but catches regressions where a
    /// refactor accidentally short-circuits the emitter.
    #[test]
    fn compiles_to_non_empty_ptx() {
        let ptx = compile_partition_kernel().expect("kernel compiles");
        assert!(
            !ptx.is_empty(),
            "compile_partition_kernel returned empty PTX"
        );
    }

    // -----------------------------------------------------------------------
    // Shared-memory-staging variant tests.
    //
    // The shmem-staging kernel exports the same entry symbol as the
    // global-atomics kernel, so the launch site doesn't care which it
    // gets back. The contract that DOES differ — and these tests pin —
    // is the per-row atomic being a *shared* atomic, plus the presence
    // of the zero-init phase + post-loop flush back to global.
    // -----------------------------------------------------------------------

    /// The shmem-staging kernel must declare a 16 KiB shared-memory
    /// staging buffer. Regressing this (e.g. accidentally allocating
    /// dynamic shared memory or dropping the declaration) would silently
    /// break the kernel — the shared atomics would land at address 0.
    #[test]
    fn shmem_staging_declares_block_counts() {
        let ptx = compile_partition_kernel_shmem_staging().expect("kernel compiles");
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
    /// kind of regression is exactly what we need a substring check
    /// for).
    #[test]
    fn shmem_staging_uses_atom_shared_add_u32() {
        let ptx = compile_partition_kernel_shmem_staging().expect("kernel compiles");
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
        let ptx = compile_partition_kernel_shmem_staging().expect("kernel compiles");
        assert!(
            ptx.contains("atom.global.add.u32"),
            "shmem-staging PTX must end with atom.global.add.u32 to flush \
             block_counts into the per-grid counts[]:\n{ptx}"
        );
    }

    /// Both the zero-init phase (between block_counts setup and the
    /// loop) AND the post-loop flush MUST be preceded by a
    /// `bar.sync 0`. Without the second barrier, racing threads would
    /// flush a still-being-updated block_counts.
    #[test]
    fn shmem_staging_has_two_barsync_barriers() {
        let ptx = compile_partition_kernel_shmem_staging().expect("kernel compiles");
        let bar_count = ptx.matches("bar.sync 0").count();
        assert!(
            bar_count >= 2,
            "shmem-staging PTX must emit >=2 bar.sync 0 (post-zero-init + \
             post-grid-stride-loop); saw {bar_count}:\n{ptx}"
        );
    }

    /// The shmem-staging kernel exports the same entry symbol as the
    /// global-atomics kernel — both variants are interchangeable from
    /// the dispatcher's point of view.
    #[test]
    fn shmem_staging_has_correct_entry_name() {
        let ptx = compile_partition_kernel_shmem_staging().expect("kernel compiles");
        let needle = format!(".visible .entry {}(", KERNEL_ENTRY);
        assert!(
            ptx.contains(&needle),
            "shmem-staging PTX must declare .visible .entry {}(  — got:\n{ptx}",
            KERNEL_ENTRY
        );
    }

    /// The shmem-staging kernel must use the Knuth multiplier mask just
    /// like the global-atomics kernel — same hash, same partition
    /// assignment. A regression that diverges the hash between variants
    /// would silently break the cross-variant correctness invariant.
    #[test]
    fn shmem_staging_uses_same_hash_as_global_atomics() {
        let ptx = compile_partition_kernel_shmem_staging().expect("kernel compiles");
        let has_hex = ptx.contains("0x9E3779B1");
        let has_dec = ptx.contains("2654435761");
        assert!(
            has_hex || has_dec,
            "shmem-staging PTX must use the Knuth multiplier 0x9E3779B1:\n{ptx}"
        );
        assert!(
            ptx.contains("and.b32"),
            "shmem-staging PTX must mask via and.b32 (power-of-two NUM_PARTITIONS):\n{ptx}"
        );
    }

    /// Dispatcher: small inputs route to the global-atomics variant
    /// (the shmem-staging fixed overhead dominates below the
    /// threshold).
    #[test]
    fn dispatcher_small_input_picks_global_atomics() {
        let ptx = compile_partition_kernel_for_n_rows(SHMEM_STAGING_MIN_ROWS - 1)
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
        let ptx = compile_partition_kernel_for_n_rows(SHMEM_STAGING_MIN_ROWS)
            .expect("dispatcher compiles");
        assert!(
            ptx.contains("atom.shared.add.u32"),
            "large-n_rows dispatch must pick the shmem-staging kernel \
             (atom.shared.add.u32 expected):\n{ptx}"
        );
    }

    /// Dispatcher boundary: `SHMEM_STAGING_MIN_ROWS` exactly is the
    /// first row count that crosses into the shmem-staging path.
    #[test]
    fn dispatcher_boundary_is_inclusive_at_threshold() {
        let below =
            compile_partition_kernel_for_n_rows(SHMEM_STAGING_MIN_ROWS - 1).expect("compile");
        let at_threshold =
            compile_partition_kernel_for_n_rows(SHMEM_STAGING_MIN_ROWS).expect("compile");
        // The two PTX strings must differ — proves the dispatcher
        // actually switches variants at the threshold.
        assert_ne!(
            below, at_threshold,
            "dispatcher must produce different PTX on either side of \
             SHMEM_STAGING_MIN_ROWS = {}",
            SHMEM_STAGING_MIN_ROWS
        );
    }

    /// Backwards-compatibility: the unparameterised `compile_partition_kernel()`
    /// wrapper must still emit the global-atomics variant. Existing PTX
    /// golden tests (in `tests/ptx_golden_tests.rs`) and existing
    /// callers in the orchestrators depend on this name continuing to
    /// resolve to the original PTX shape.
    #[test]
    fn back_compat_wrapper_still_emits_global_atomics() {
        let wrapper = compile_partition_kernel().expect("compile");
        let global = compile_partition_kernel_global_atomics().expect("compile");
        assert_eq!(
            wrapper, global,
            "compile_partition_kernel() must alias \
             compile_partition_kernel_global_atomics() for back-compat"
        );
    }
}
