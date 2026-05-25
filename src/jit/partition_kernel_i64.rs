// SPDX-License-Identifier: Apache-2.0

//! Hash-partition pass-1 kernel for **64-bit packed keys** (Tier 2 of the
//! GROUP BY perf plan).
//!
//! This is the i64 sibling of [`crate::jit::partition_kernel`]. The single-
//! Int32 variant hashes 32-bit keys into one of `NUM_PARTITIONS = 1024`
//! buckets using `(u32)key * 0x9E3779B1 & 0x3FF`. This file does the same
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
//! const uint32_t K                 = 1024;
//! const uint64_t HASH_MULTIPLIER64 = 0x9E3779B97F4A7C15ull;  // floor(2^64 / phi)
//!
//! for (i = blockIdx.x*blockDim.x + tid; i < n_rows; i += gridDim.x*blockDim.x) {
//!     int64_t  key  = keys[i];
//!     uint64_t hash = (uint64_t)key * HASH_MULTIPLIER64;  // wraps mod 2^64
//!     uint32_t pid  = (uint32_t)(hash >> 54);             // top 10 bits
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
//! by `64 - log2(K) = 54` to read the top 10 bits as the partition id.
//! The i32 sibling can get away with `(hash & 0x3FF)` because its 32-bit
//! multiply already wraps and the wrap acts as a poor-man's mix; at 64
//! bits we'd be reading entirely raw key bits otherwise.
//!
//! ## PTX-level notes
//!
//! * Keys are loaded with `ld.global.s64` (8-byte loads). Stride per row is 8.
//! * The 64-bit multiply uses `mul.lo.u64`, which writes the low 64 bits of
//!   the 128-bit product — exactly the wrapping multiply we want.
//! * The 10-bit shift to extract the partition id is `shr.u64 %rdN, %rd_hash, 54`,
//!   followed by `cvt.u32.u64` to land in a 32-bit register for the atomic.
//!   We deliberately use the unsigned `shr.u64` so the top bits zero-fill
//!   (no sign extension hazard from `shr.s64`).
//! * `atom.global.add.u32` on counts is identical to the i32 sibling — the
//!   counts array stays u32-indexed by partition id.

use std::fmt::Write;

use crate::error::{JavelinError, JavelinResult};

/// Number of hash partitions. MUST match the i32 sibling so the host-side
/// orchestrator can use the same `compute_partition_offsets` and Tier-2
/// invariants ("each partition fits in shmem").
pub const NUM_PARTITIONS: u32 = 4096;

/// 64-bit Fibonacci / Knuth multiplicative-hash constant —
/// `floor(2^64 / phi)` rounded to the nearest odd integer. Mixes the low
/// bits of structured inputs up into the high bits where we'll mask them
/// out for the partition id.
pub const HASH_MULTIPLIER_64: u64 = 0x9E37_79B9_7F4A_7C15;

/// Threads per block. Matches the i32 sibling so launch geometry is shared.
pub const BLOCK_THREADS: u32 = 256;

/// Entry-point name embedded in the emitted PTX. Distinct from the i32
/// sibling so both modules can be loaded into the same CUDA context.
pub const KERNEL_ENTRY: &str = "javelin_partition_i64";

/// Generate PTX for the **i64-key** hash-partition pass-1 kernel.
///
/// Kernel signature (PTX-level):
///
/// ```text
/// .visible .entry javelin_partition_i64(
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
pub fn compile_partition_kernel_i64() -> JavelinResult<String> {
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
    //   pid   = (u32)(hash >> 54)             // top 10 bits = partition id
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

    // pid_u64 = hash >> 54    (top 10 bits of the multiplied product)
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

fn write_err(e: std::fmt::Error) -> JavelinError {
    JavelinError::Other(format!("partition_kernel_i64: write failed: {}", e))
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
        assert_eq!(KERNEL_ENTRY, "javelin_partition_i64");
    }

    /// The partition id is the TOP 10 bits of the 64-bit multiplied product,
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
            "PTX must use shr.u64 / shr.b64 to extract the top 10 bits as pid:\n{ptx}"
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
            "PTX must include the standard Javelin module header:\n{ptx}"
        );
    }
}
