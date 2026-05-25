// SPDX-License-Identifier: Apache-2.0

//! Per-block shared-memory GROUP BY SUM kernel (Tier 1 of the GROUP BY perf
//! plan).
//!
//! ## Why this exists
//!
//! Craton Patina's previous GROUP BY path issues one `atom.global.add` per input row
//! against a single device-global hash table. With the h2o.ai db-benchmark
//! workload (10 M rows × 100 distinct groups) every row contends for one of
//! 100 cache lines — the GPU's atomic unit serialises hard and Craton Patina runs
//! 2–44× slower than Polars / DuckDB.
//!
//! This kernel implements the standard fix: each CUDA block first reduces its
//! slice of the input into a `__shared__` hash table, then atomically merges
//! only its *non-empty* slots into the device-global output. Global atomic
//! traffic drops by a factor of roughly `n_rows_per_block / n_groups` — for
//! the 10 M × 100 case, ~300×.
//!
//! ## Scope
//!
//! Intentionally narrow first cut. The dispatcher (sibling agent) will fall
//! back to the existing hash-kernel path for everything outside this
//! envelope:
//!
//! * Op:      `SUM`
//! * Value:   `Float64`
//! * Key:     `Int32`, **direct-mapped**, bounded by `n_groups` from the
//!            caller. No hashing; the key IS the slot index modulo
//!            `BLOCK_GROUPS`. Keys whose value is `>= BLOCK_GROUPS` take the
//!            overflow path (one `atom.global.add.f64` straight to the
//!            output array) so q2 (10 K groups) still produces correct
//!            results, just without the shared-mem speedup for the
//!            overflowed rows.
//!
//! ## Algorithm
//!
//! ```text
//! __shared__ double  block_acc[BLOCK_GROUPS];  // initialised to 0.0
//! __shared__ uint8_t block_set[BLOCK_GROUPS];  // initialised to 0
//!
//! // 1. Cooperatively zero shared memory.
//! for (s = tid; s < BLOCK_GROUPS; s += blockDim.x) {
//!     block_acc[s] = 0.0;
//!     block_set[s] = 0;
//! }
//! __syncthreads();
//!
//! // 2. Grid-stride accumulate.
//! for (i = blockIdx.x*blockDim.x + tid; i < n_rows; i += gridDim.x*blockDim.x) {
//!     int32_t key = keys[i];
//!     double  val = vals[i];
//!     if (key >= BLOCK_GROUPS) {
//!         // Overflow path: skip the shared table for this row.
//!         atomicAdd(&out[key], val);
//!     } else {
//!         atomicAdd_shared(&block_acc[key], val);
//!         block_set[key] = 1;   // unsynchronised single-byte store; harmless race
//!     }
//! }
//! __syncthreads();
//!
//! // 3. Merge block-local table into global output. One global atomic per
//! //    non-empty slot, vs one global atomic per ROW in the old kernel.
//! for (s = tid; s < BLOCK_GROUPS; s += blockDim.x) {
//!     if (block_set[s]) {
//!         atomicAdd(&out[s], block_acc[s]);
//!     }
//! }
//! ```
//!
//! ## PTX-level notes
//!
//! * Shared variables are declared `.shared .align 8 .b8 block_acc_buf[8192]`
//!   and `.shared .align 1 .b8 block_set_buf[1024]`. The buffers are treated
//!   as raw byte arrays at declaration time; ops cast through the address by
//!   indexing with `slot * 8` or `slot`.
//! * `atom.shared.add.f64` is available on sm_60+. We target sm_70 (matching
//!   the rest of `jit/`).
//! * `ld.global` for the `keys` / `vals` pointers uses the standard
//!   `cvta.to.global.u64` dance from `.param`. Shared variables don't need
//!   that — their address resolves to a shared-state pointer directly via
//!   `mov.u64`.
//! * The `block_set` flag is written non-atomically. Multiple threads can
//!   race to store `1` to the same byte; since every racing store has the
//!   same value the outcome is well-defined even at the byte level.
//! * The merge loop's bounds check (`s < BLOCK_GROUPS`) is what keeps us from
//!   writing past `out[n_groups - 1]` when `BLOCK_GROUPS > n_groups`: an
//!   unwritten slot still has `block_set[s] == 0`, so it is skipped. The
//!   caller is responsible for sizing `out` to at least `n_groups`.

use std::fmt::Write;

use crate::error::{PatinaError, PatinaResult};

/// Number of slots in each block's shared-memory accumulator. Must be a
/// compile-time constant for shared-memory allocation. Sized to comfortably
/// hold h2o.ai q1/q4 (100 groups) entirely in the fast path; q2 (10 K groups)
/// falls through to the overflow path for keys `>= BLOCK_GROUPS`.
pub const BLOCK_GROUPS: u32 = 1024;

/// Threads per block for the shared-mem kernel. 256 threads × 1024 slots
/// means the final merge loop runs 4 iterations per thread, which keeps the
/// kernel's tail short while leaving enough threads to hide load latency in
/// the row-accumulate loop.
pub const BLOCK_THREADS: u32 = 256;

/// Entry-point name embedded in the emitted PTX.
pub const KERNEL_ENTRY: &str = "patina_groupby_shmem_sum_f64";

/// Generate PTX for the shared-memory per-block SUM kernel.
///
/// Kernel signature (PTX-level):
///
/// ```text
/// .visible .entry patina_groupby_shmem_sum_f64(
///     .param .u64 keys_ptr,     // const int32_t* keys, length n_rows
///     .param .u64 vals_ptr,     // const double*  vals, length n_rows
///     .param .u64 out_ptr,      // double*        out,  length n_groups
///     .param .u32 n_rows,
///     .param .u32 n_groups
/// )
/// ```
///
/// See module-level docs for the algorithm. `compile_shmem_sum_kernel()` is
/// deterministic and pure: it returns a fixed PTX string and has no I/O.
pub fn compile_shmem_sum_kernel() -> PatinaResult<String> {
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY;
    let block_groups = BLOCK_GROUPS;
    let acc_bytes = BLOCK_GROUPS * 8;
    let set_bytes = BLOCK_GROUPS;
    let block_threads = BLOCK_THREADS;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Shared-memory accumulator + per-slot "set" flag. Function-scope
    // `.shared` variables are allocated by the driver from the launch's
    // configured shared-memory pool; with the sizes below the kernel needs
    // BLOCK_GROUPS * 8 + BLOCK_GROUPS bytes = 9 * BLOCK_GROUPS bytes
    // (9 KiB at BLOCK_GROUPS=1024). Well under the 48 KiB default static
    // shared-mem budget on every sm_70+ device.
    writeln!(
        ptx,
        ".shared .align 8 .b8 block_acc_buf[{bytes}];",
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
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_3,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_4").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Register decls. Generous counts mirror the conventions of
    // `valid_flag_kernels::compile_keys_valid_kernel`.
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<8>;").map_err(write_err)?;
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

    // --- shared-memory base addresses --------------------------------------
    // `mov.u64 %rd, var` yields the shared-state address of `var`. No
    // `cvta` needed — atom.shared / ld.shared / st.shared all consume
    // shared-state pointers directly.
    writeln!(ptx, "\tmov.u64 %rd0, block_acc_buf;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, block_set_buf;").map_err(write_err)?;

    // --- global pointer setup (cvta from .param) ---------------------------
    // %rd2 = keys (i32*), %rd3 = vals (f64*), %rd4 = out (f64*).
    writeln!(ptx, "\tld.param.u64 %rd2, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd3, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd4, [{entry}_param_2];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ----------------------------------------------------------------------
    // Phase 1: cooperatively zero the shared accumulator + set flags.
    // ----------------------------------------------------------------------
    // %r10 = zero-loop index (start at threadIdx.x, stride BLOCK_THREADS).
    // We zero `block_acc[s] = 0.0` as `st.shared.u64 [.], 0` (bit pattern of
    // 0.0 == 0) and `block_set[s] = 0` as `st.shared.u8 [.], 0`.
    writeln!(ptx, "\tmov.u32 %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p0, %r10, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    // block_acc[%r10] = 0.0  (f64 zero == u64 zero in bits)
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
    // For each row this block touches:
    //   key = keys[i]   (i32, loaded as s32)
    //   val = vals[i]   (f64)
    //   if key >= BLOCK_GROUPS:
    //       atom.global.add.f64 [out + key*8], val
    //   else:
    //       atom.shared.add.f64 [block_acc + key*8], val
    //       st.shared.u8        [block_set + key],   1
    // ----------------------------------------------------------------------
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r3, %r6;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    // key = keys[i] (i32 -> s32 register %r12)
    writeln!(ptx, "\tmul.wide.u32 %rd14, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd15, %rd2, %rd14;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s32 %r12, [%rd15];").map_err(write_err)?;

    // val = vals[i] (f64 -> %fd0)
    writeln!(ptx, "\tmul.wide.u32 %rd16, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd17, %rd3, %rd16;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.f64 %fd0, [%rd17];").map_err(write_err)?;

    // Overflow check: if (unsigned)key >= BLOCK_GROUPS, take the global path.
    // We compare unsigned: a negative key (which shouldn't happen because the
    // caller bounds keys to [0, n_groups)) would otherwise wrap into the
    // shared table and corrupt arbitrary slots. Treating it as unsigned makes
    // any out-of-range key (negative or >= BLOCK_GROUPS) take the safe path.
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p2, %r12, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra OVERFLOW;").map_err(write_err)?;

    // Shared accumulate.
    //   addr_acc = block_acc + key * 8
    writeln!(ptx, "\tmul.wide.s32 %rd18, %r12, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd19, %rd0, %rd18;").map_err(write_err)?;
    writeln!(ptx, "\tatom.shared.add.f64 %fd1, [%rd19], %fd0;").map_err(write_err)?;
    //   block_set[key] = 1  (unsynchronised; all stores have the same value)
    writeln!(ptx, "\tcvt.s64.s32 %rd20, %r12;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, %rd1, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r13, 1;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u8 [%rd21], %r13;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    // Overflow path: skip the shared table, hit `out` directly. This is the
    // SAME atomic the legacy single-table kernel issues — so even worst-case
    // (every key overflows) we degrade to parity with the old kernel, never
    // worse.
    writeln!(ptx, "OVERFLOW:").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.s32 %rd22, %r12, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd4, %rd22;").map_err(write_err)?;
    writeln!(ptx, "\tatom.global.add.f64 %fd2, [%rd23], %fd0;").map_err(write_err)?;

    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r3, %r3, %r5;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ----------------------------------------------------------------------
    // Phase 3: merge the block-local table into the global output.
    //
    // Each thread sweeps a stride of BLOCK_THREADS slots. With
    // BLOCK_GROUPS=1024 and BLOCK_THREADS=256 every thread processes exactly
    // 4 slots; if either constant changes the stride pattern still covers
    // the full range.
    //
    // A slot only contributes an atom.global.add when block_set[s] != 0,
    // so unused slots (or slots that fell off the overflow cliff and were
    // never touched in shared) cost nothing.
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

    // Load shared accumulator, atomic-add into the global output.
    writeln!(ptx, "\tmul.wide.u32 %rd26, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd27, %rd0, %rd26;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.f64 %fd3, [%rd27];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd28, %rd4, %rd26;").map_err(write_err)?;
    writeln!(ptx, "\tatom.global.add.f64 %fd4, [%rd28], %fd3;").map_err(write_err)?;

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

/// Adapt a `std::fmt::Error` into a `PatinaError`. Same shape as the helper
/// in `valid_flag_kernels.rs` — kept local so changes to one file don't
/// surprise the other.
fn write_err(e: std::fmt::Error) -> PatinaError {
    PatinaError::Other(format!("shmem_sum_kernel: write failed: {}", e))
}

// ---------------------------------------------------------------------------
// PTX-shape tests (host-only — do NOT require a GPU).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Both shared-memory buffers (`block_acc_buf` and `block_set_buf`) must
    /// be declared in the module preamble — without them the kernel can't
    /// allocate its block-local table at launch.
    #[test]
    fn emits_shared_mem_directives() {
        let ptx = compile_shmem_sum_kernel().expect("kernel compiles");
        assert!(
            ptx.contains(".shared") && ptx.contains("block_acc_buf"),
            "PTX must declare a `.shared` block_acc_buf:\n{ptx}"
        );
        assert!(
            ptx.contains(".shared") && ptx.contains("block_set_buf"),
            "PTX must declare a `.shared` block_set_buf:\n{ptx}"
        );
        // Sanity: the accumulator must be 8 * BLOCK_GROUPS bytes; the set
        // flags must be BLOCK_GROUPS bytes. A typo'd literal here would be a
        // silent corruption bug at launch time.
        let acc_bytes = (BLOCK_GROUPS * 8).to_string();
        let set_bytes = BLOCK_GROUPS.to_string();
        assert!(
            ptx.contains(&format!("block_acc_buf[{}]", acc_bytes)),
            "block_acc_buf must be sized {} bytes:\n{ptx}",
            acc_bytes
        );
        assert!(
            ptx.contains(&format!("block_set_buf[{}]", set_bytes)),
            "block_set_buf must be sized {} bytes:\n{ptx}",
            set_bytes
        );
    }

    /// The shared-memory pre-aggregate uses `atom.shared.add.f64` — that's
    /// the whole point of the kernel.
    #[test]
    fn uses_atom_shared_add_f64() {
        let ptx = compile_shmem_sum_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("atom.shared.add.f64"),
            "PTX must use atom.shared.add.f64 for the block-local accumulate:\n{ptx}"
        );
    }

    /// The merge phase issues `atom.global.add.f64` to fold the block-local
    /// table into the device-global output.
    #[test]
    fn uses_atom_global_add_f64_for_merge() {
        let ptx = compile_shmem_sum_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("atom.global.add.f64"),
            "PTX must use atom.global.add.f64 for the per-block merge:\n{ptx}"
        );
    }

    /// The kernel must export the well-known entry point name so the
    /// dispatcher can resolve it via `cuModuleGetFunction`.
    #[test]
    fn has_correct_entry_name() {
        let ptx = compile_shmem_sum_kernel().expect("kernel compiles");
        let needle = format!(".visible .entry {}(", KERNEL_ENTRY);
        assert!(
            ptx.contains(&needle),
            "PTX must declare .visible .entry {}(  — got:\n{ptx}",
            KERNEL_ENTRY
        );
    }

    /// `__syncthreads()` in PTX is `bar.sync 0`. We need it twice: once
    /// after the zero-init phase, once between accumulate and merge. The
    /// merge phase only reads block_set/block_acc values written by other
    /// threads, so the barrier is load-bearing for correctness.
    #[test]
    fn syncthreads_present() {
        let ptx = compile_shmem_sum_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("bar.sync 0"),
            "PTX must include bar.sync 0 (__syncthreads()):\n{ptx}"
        );
        let count = ptx.matches("bar.sync 0").count();
        assert!(
            count >= 2,
            "PTX should include at least two bar.sync 0 calls (one after zero-init, one between accumulate and merge), saw {count}:\n{ptx}"
        );
    }

    /// Both an inside-the-row-loop `atom.global.add.f64` (the OVERFLOW path)
    /// AND a merge-loop `atom.global.add.f64` (final reduction) must appear.
    /// Counting at least two occurrences is the cheapest way to assert this
    /// without parsing the PTX.
    #[test]
    fn overflow_path_exists() {
        let ptx = compile_shmem_sum_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("atom.shared.add.f64"),
            "shared-mem accumulate must be present:\n{ptx}"
        );
        let global_atomic_count = ptx.matches("atom.global.add.f64").count();
        assert!(
            global_atomic_count >= 2,
            "expected at least two atom.global.add.f64 instructions (overflow path + final merge), saw {global_atomic_count}:\n{ptx}"
        );
        assert!(
            ptx.contains("OVERFLOW:"),
            "PTX should declare an OVERFLOW: label so the row loop has somewhere to branch when key >= BLOCK_GROUPS:\n{ptx}"
        );
    }

    /// `compile_shmem_sum_kernel` is pure: same input -> same output. A
    /// dispatcher caching the PTX should always get a stable result.
    #[test]
    fn output_is_deterministic() {
        let a = compile_shmem_sum_kernel().expect("compile a");
        let b = compile_shmem_sum_kernel().expect("compile b");
        assert_eq!(a, b, "shmem_sum kernel emitter must be deterministic");
    }

    /// PTX module preamble must match the rest of `src/jit/*` — same target,
    /// same address size. The dispatcher links other modules at the same
    /// version so a mismatch here would prevent driver-level co-compile.
    #[test]
    fn ptx_header_matches_project_conventions() {
        let ptx = compile_shmem_sum_kernel().expect("kernel compiles");
        assert!(ptx.contains(".version 7.5"), "PTX must be .version 7.5");
        assert!(ptx.contains(".target sm_70"), "PTX must target sm_70");
        assert!(
            ptx.contains(".address_size 64"),
            "PTX must declare .address_size 64"
        );
    }

    /// GPU-required smoke test: confirm the PTX is accepted by the CUDA
    /// driver. Skipped by default; run with `cargo test --release -- --ignored`
    /// on a machine with a CUDA 12.x driver.
    #[test]
    #[ignore]
    fn ptx_loads_into_cuda_driver() {
        let ptx = compile_shmem_sum_kernel().expect("kernel compiles");
        let module = crate::jit::CudaModule::from_ptx(&ptx)
            .expect("PTX should load via cuModuleLoadDataEx");
        // The entry point must also be resolvable via cuModuleGetFunction —
        // a typo'd `.visible .entry` name only fails here, not at load time.
        let _fn = module
            .function(KERNEL_ENTRY)
            .expect("kernel entry point should be reachable");
    }
}
