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
//! in each of the K = `NUM_PARTITIONS` (4096) partitions and remember which
//! partition each row was assigned to. A sibling scatter kernel then
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
//! * Bank-conflict / contention notes (acceptable for Tier-2 v0):
//!   - `counts[]` has only `NUM_PARTITIONS` (4096) slots. Concurrent
//!     `atom.global.add.u32` traffic across blocks IS real and serialises on
//!     the L2 atomic unit, same hazard the Tier-1 kernel exists to mitigate.
//!   - We intentionally do **not** stage `counts[]` in shared memory yet —
//!     it would mean a per-block reduce-and-merge phase that doubles the
//!     kernel's code surface for a step that the sibling prefix-sum pass
//!     dominates anyway. If the partition kernel shows up high in the
//!     profile after Tier 2 lands, shared-mem staging is the obvious next
//!     step.
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

// Compile-time invariant: the PTX emits `and.b32 %r_pid, %r_hash, (K-1)`
// in place of `rem.u32`, which is only equivalent to `% K` when K is a
// power of two. See the docstring on `NUM_PARTITIONS` above.
const _: () = assert!(
    NUM_PARTITIONS.is_power_of_two(),
    "partition mask requires power-of-two count"
);

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
pub const KERNEL_ENTRY: &str = "bolt_partition";

/// Generate PTX for the hash-partition pass-1 kernel.
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
/// `compile_partition_kernel()` is deterministic and pure: same input → same
/// output, no I/O. The dispatcher can cache the result indefinitely.
///
/// ## `rust-cuda` feature variant (Wave A spike)
///
/// When the `rust-cuda` Cargo feature is enabled, this function returns
/// PTX compiled at build time from `kernels/src/lib.rs` (via
/// `cuda_builder` + `rustc_codegen_nvvm`) instead of the hand-emitted
/// string below. The two paths are interchangeable from the dispatcher's
/// point of view: both produce a PTX module that exports a single
/// `.visible .entry bolt_partition(.param .u64, .param .u64, .param .u64, .param .u32)`
/// symbol, and both load via `CudaModule::from_ptx`. The hand-emit path
/// is the default and remains the only path until Wave B.
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

#[cfg(not(feature = "rust-cuda"))]
pub fn compile_partition_kernel() -> BoltResult<String> {
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
    writeln!(
        ptx,
        "\tmov.u32 %r7, 0x{mult:08X};",
        mult = HASH_MULTIPLIER
    )
    .map_err(write_err)?;
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
    writeln!(
        ptx,
        "\tand.b32 %r10, %r9, 0x{mask:X};",
        mask = mask
    )
    .map_err(write_err)?;

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
}
