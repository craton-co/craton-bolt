// SPDX-License-Identifier: Apache-2.0

//! Per-block shared-memory GROUP BY **COUNT** kernel (Tier 1 fast path).
//!
//! This is the sibling of [`crate::jit::shmem_sum_kernel`]: same per-block
//! shared-mem pre-aggregation structure, but the per-row update is
//! `count[key] += 1` instead of `sum[key] += val`. The output type is `u64`
//! (the SQL standard COUNT-returns-Int64 semantics) so the host-side
//! division for `AVG = SUM / COUNT` is a straight `f64 / u64 as f64`.
//!
//! ## Algorithm (mirrors the SUM kernel)
//!
//! ```text
//! __shared__ uint64_t block_count[BLOCK_GROUPS]; // initialised to 0
//! __shared__ uint8_t  block_set[BLOCK_GROUPS];   // initialised to 0
//!
//! // 1. Zero shared memory.
//! for (s = tid; s < BLOCK_GROUPS; s += blockDim.x) {
//!     block_count[s] = 0;
//!     block_set[s]   = 0;
//! }
//! __syncthreads();
//!
//! // 2. Grid-stride accumulate. COUNT(*) only needs the key.
//! for (i = blockIdx.x*blockDim.x + tid; i < n_rows; i += gridDim.x*blockDim.x) {
//!     int32_t key = keys[i];
//!     if (key >= BLOCK_GROUPS) {
//!         atomicAdd_u64(&out_count[key], 1);   // overflow path
//!     } else {
//!         atomicAdd_shared_u64(&block_count[key], 1);
//!         block_set[key] = 1;
//!     }
//! }
//! __syncthreads();
//!
//! // 3. Merge non-empty slots into the global counter array.
//! for (s = tid; s < BLOCK_GROUPS; s += blockDim.x) {
//!     if (block_set[s]) {
//!         atomicAdd_u64(&out_count[s], block_count[s]);
//!     }
//! }
//! ```
//!
//! ## PTX-level notes
//!
//! * `atom.shared.add.u64` and `atom.global.add.u64` are both supported on
//!   sm_70+. There is **no** `.s64` variant of `atom.add` in PTX — we
//!   intentionally use the unsigned form (counts are non-negative anyway).
//! * The accumulator buffer is `8 * BLOCK_GROUPS` bytes (u64), the set-flag
//!   buffer is `BLOCK_GROUPS` bytes (u8). Same shared-mem footprint as the
//!   SUM kernel, so the launch-tuner sizes don't need to change.
//! * Value pointer is intentionally absent from the signature: COUNT(*)
//!   ignores values, and any non-`*` COUNT semantics (`COUNT(col)`) is the
//!   safe-path's responsibility for now.
//! * The non-atomic `block_set[key] = 1` stores race exactly the same way
//!   they do in the SUM kernel — all racing stores have the same value, so
//!   the byte-level outcome is well-defined.

use std::fmt::Write;

use crate::error::{JavelinError, JavelinResult};

/// Number of slots in each block's shared-memory counter table. Must match
/// the SUM kernel's `BLOCK_GROUPS` so the AVG executor can share a single
/// presence map and slot layout across both passes.
pub const BLOCK_GROUPS: u32 = 1024;

/// Threads per block. Same value as the SUM kernel for the same reason —
/// 8 warps balances hiding shared-mem-atomic latency against keeping the
/// merge tail short.
pub const BLOCK_THREADS: u32 = 256;

/// Entry-point name embedded in the emitted PTX. Distinct from the SUM
/// kernel so the AVG executor can load both into a single module-builder
/// session and look them up by name.
pub const KERNEL_ENTRY: &str = "javelin_groupby_shmem_count_u64";

/// Generate PTX for the shared-memory per-block COUNT kernel.
///
/// Kernel signature (PTX-level):
///
/// ```text
/// .visible .entry javelin_groupby_shmem_count_u64(
///     .param .u64 keys_ptr,      // const int32_t* keys, length n_rows
///     .param .u64 out_count_ptr, // uint64_t*      out_count, length n_groups
///     .param .u32 n_rows,
///     .param .u32 n_groups
/// )
/// ```
///
/// Pure / deterministic: same input -> same output.
pub fn compile_shmem_count_kernel() -> JavelinResult<String> {
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY;
    let block_groups = BLOCK_GROUPS;
    let acc_bytes = BLOCK_GROUPS * 8; // u64
    let set_bytes = BLOCK_GROUPS;
    let block_threads = BLOCK_THREADS;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Shared-memory counter + per-slot "set" flag. Static `.shared` —
    // 9 KiB total at BLOCK_GROUPS=1024, well under the 48 KiB sm_70 floor.
    writeln!(
        ptx,
        ".shared .align 8 .b8 block_count_buf[{bytes}];",
        bytes = acc_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 1 .b8 block_set_buf[{bytes}];",
        bytes = set_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_2,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_3").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Register decls. Mirror the SUM kernel's generous counts so future
    // extensions (e.g. predicated count) have headroom without renumbering.
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
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
    writeln!(ptx, "\tld.param.u32 %r6, [{entry}_param_2];").map_err(write_err)?;

    // --- shared-memory base addresses --------------------------------------
    writeln!(ptx, "\tmov.u64 %rd0, block_count_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_set_buf;").map_err(write_err)?;

    // --- global pointer setup ----------------------------------------------
    // %rd2 = keys (i32*), %rd4 = out_count (u64*).
    writeln!(ptx, "\tld.param.u64 %rd2, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd4, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ----------------------------------------------------------------------
    // Phase 1: cooperatively zero the shared counter + set flags.
    // ----------------------------------------------------------------------
    writeln!(ptx, "\tmov.u32 %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p0, %r10, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    // block_count[%r10] = 0  (u64 zero)
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r10, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd0, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd11], 0;").map_err(write_err)?;
    // block_set[%r10] = 0
    writeln!(ptx, "\tcvt.u64.u32 %rd12, %r10;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd13, %rd1, %rd12;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u8 [%rd13], 0;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tadd.u32 %r10, %r10, {bt};",
        bt = block_threads
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra ZERO_TOP;").map_err(write_err)?;
    writeln!(ptx, "ZERO_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ----------------------------------------------------------------------
    // Phase 2: grid-stride loop over input rows.
    //
    //   key = keys[i]   (i32, loaded as s32)
    //   if (unsigned)key >= BLOCK_GROUPS:
    //       atom.global.add.u64 [out_count + key*8], 1
    //   else:
    //       atom.shared.add.u64 [block_count + key*8], 1
    //       st.shared.u8        [block_set   + key],   1
    //
    // The "+= 1" increment is materialised as a `mov.u64 %rd_one, 1` once
    // outside the loop body wouldn't actually help — PTX has no per-block
    // constant pool, and the SASS optimiser folds it anyway. We mov-into a
    // register per-iteration for clarity; the SASS compiler hoists it.
    // ----------------------------------------------------------------------
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r3, %r6;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    // key = keys[i] (i32 -> s32 register %r12)
    writeln!(ptx, "\tmul.wide.u32 %rd14, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd15, %rd2, %rd14;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s32 %r12, [%rd15];").map_err(write_err)?;

    // Overflow check: unsigned compare so a (defensively-allowed) negative
    // key takes the safe global path instead of corrupting a shared slot.
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p2, %r12, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra OVERFLOW;").map_err(write_err)?;

    // Shared increment.  block_count[key] += 1
    writeln!(ptx, "\tmul.wide.s32 %rd18, %r12, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd19, %rd0, %rd18;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd30, 1;").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.u64 %rd31, [%rd19], %rd30;").map_err(write_err)?;
    //   block_set[key] = 1  (unsynchronised; benign race)
    writeln!(ptx, "\tcvt.s64.s32 %rd20, %r12;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, %rd1, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r13, 1;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u8 [%rd21], %r13;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    // Overflow path: hit out_count directly with a global u64 atomic add.
    writeln!(ptx, "OVERFLOW:").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.s32 %rd22, %r12, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd4, %rd22;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd28, 1;").map_err(write_err)?;
    writeln!(ptx, "\tatom.global.add.u64 %rd29, [%rd23], %rd28;").map_err(write_err)?;

    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r3, %r3, %r5;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ----------------------------------------------------------------------
    // Phase 3: merge block-local counts into the global counter array.
    //
    // Same sweep pattern as the SUM kernel: every thread takes a stride of
    // BLOCK_THREADS, only contributes to slots whose `block_set[s]` is set.
    // ----------------------------------------------------------------------
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "MERGE_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p3, %r20, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra MERGE_DONE;").map_err(write_err)?;

    // Load block_set[%r20] (zero-extended into 32-bit %r21).
    writeln!(ptx, "\tcvt.u64.u32 %rd24, %r20;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd25, %rd1, %rd24;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u8 %r21, [%rd25];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p4, %r21, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra MERGE_NEXT;").map_err(write_err)?;

    // Load shared count (u64), atomic-add into global out_count.
    writeln!(ptx, "\tmul.wide.u32 %rd26, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd27, %rd0, %rd26;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u64 %rd16, [%rd27];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd17, %rd4, %rd26;").map_err(write_err)?;
    writeln!(ptx, "\tatom.global.add.u64 %rd6, [%rd17], %rd16;").map_err(write_err)?;

    writeln!(ptx, "MERGE_NEXT:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tadd.u32 %r20, %r20, {bt};",
        bt = block_threads
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra MERGE_TOP;").map_err(write_err)?;
    writeln!(ptx, "MERGE_DONE:").map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// `std::fmt::Error` -> `JavelinError`; matches `shmem_sum_kernel`.
fn write_err(e: std::fmt::Error) -> JavelinError {
    JavelinError::Other(format!("shmem_count_kernel: write failed: {}", e))
}

// ---------------------------------------------------------------------------
// PTX-shape tests (host-only — do NOT require a GPU).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the kernel: per-block shared-mem `+= 1`.
    /// Note we deliberately use `.u64` not `.s64` — PTX has no signed
    /// variant of `atom.add` for 64-bit ints.
    #[test]
    fn uses_atom_shared_add_u64() {
        let ptx = compile_shmem_count_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("atom.shared.add.u64"),
            "PTX must use atom.shared.add.u64 for the block-local count:\n{ptx}"
        );
        // Belt-and-braces: a `.s64` variant would be a PTX-level rejection
        // at module load; catch it at the shape level too.
        assert!(
            !ptx.contains("atom.shared.add.s64"),
            "atom.shared.add.s64 does not exist in PTX — must use .u64:\n{ptx}"
        );
    }

    /// The merge phase issues `atom.global.add.u64` to fold the per-block
    /// counts into the device-global counter array.
    #[test]
    fn uses_atom_global_add_u64_for_merge() {
        let ptx = compile_shmem_count_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("atom.global.add.u64"),
            "PTX must use atom.global.add.u64 for the per-block merge:\n{ptx}"
        );
        assert!(
            !ptx.contains("atom.global.add.s64"),
            "atom.global.add.s64 does not exist in PTX — must use .u64:\n{ptx}"
        );
    }

    /// Resolvable entry-point name so the AVG executor can look the kernel
    /// up via `cuModuleGetFunction`.
    #[test]
    fn has_correct_entry_name() {
        let ptx = compile_shmem_count_kernel().expect("kernel compiles");
        let needle = format!(".visible .entry {}(", KERNEL_ENTRY);
        assert!(
            ptx.contains(&needle),
            "PTX must declare .visible .entry {}(  — got:\n{ptx}",
            KERNEL_ENTRY
        );
    }

    /// `__syncthreads()` in PTX is `bar.sync 0`. We need it after zero-init
    /// and between accumulate and merge.
    #[test]
    fn syncthreads_present() {
        let ptx = compile_shmem_count_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("bar.sync 0"),
            "PTX must include bar.sync 0 (__syncthreads()):\n{ptx}"
        );
    }
}
