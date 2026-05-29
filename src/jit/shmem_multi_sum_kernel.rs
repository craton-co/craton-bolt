// SPDX-License-Identifier: Apache-2.0

//! Per-block shared-memory GROUP BY kernel with **multiple SUM aggregates**
//! computed in a single pass (Tier-1 extension).
//!
//! ## Why this exists
//!
//! `shmem_sum_kernel` handles `SELECT id, SUM(v) FROM x GROUP BY id` — one
//! value column, one output. Real workloads (notably h2o.ai q2,
//! `SELECT id2, SUM(v1), SUM(v2) FROM x GROUP BY id2`) ask for several SUMs
//! over the *same* grouping in a single query. Running the single-SUM kernel
//! N times wastes N× the global-memory bandwidth on the key column and N× the
//! shared-mem zero / merge cost. Folding all aggregates into one kernel lets
//! the row loop pay the key load once and the merge phase amortise its
//! overhead across every output.
//!
//! ## Scope
//!
//! * Op:      `SUM` only.
//! * Value:   `Float64` (every value column).
//! * Key:     `Int32`, direct-mapped, bounded by `BLOCK_GROUPS` from the
//!            caller (overflowed keys go straight to the per-aggregate
//!            global atomics, same as the single-SUM kernel).
//! * Width:   `1..=MAX_VALS` value columns. `MAX_VALS=4` keeps the worst-case
//!            shared-mem footprint comfortably under the sm_70 48 KiB static
//!            limit: `n_vals * BLOCK_GROUPS * 8 + BLOCK_GROUPS` =
//!            `4 * 1024 * 8 + 1024` = 33 KiB at the cap.
//!
//! ## Algorithm
//!
//! Identical in structure to the single-SUM kernel — the only difference is
//! that everywhere the single-SUM kernel touches `block_acc[slot]` or
//! `out[slot]`, the multi-SUM kernel touches `block_acc_j[slot]` /
//! `out_j[slot]` for every `j` in `0..n_vals`. The `block_set[slot]` flag is
//! shared across all aggregates: a slot is "live" if **any** value column
//! contributed to it (in practice every row touches every value column, so
//! "any" and "all" coincide).
//!
//! ```text
//! __shared__ double  block_acc_j[BLOCK_GROUPS];   // for j in 0..n_vals
//! __shared__ uint8_t block_set[BLOCK_GROUPS];
//!
//! // 1. Cooperatively zero shared memory.
//! //    (Each j's accumulator + the shared set-flag array.)
//!
//! // 2. Grid-stride accumulate.
//! for (i = ...; i < n_rows; ...) {
//!     int32_t key = keys[i];
//!     if (key >= BLOCK_GROUPS) {
//!         for (j = 0; j < n_vals; j++)
//!             atomicAdd(&out_j[key], vals_j[i]);
//!     } else {
//!         for (j = 0; j < n_vals; j++)
//!             atomicAdd_shared(&block_acc_j[key], vals_j[i]);
//!         block_set[key] = 1;
//!     }
//! }
//!
//! // 3. Merge.
//! for (s = ...; s < BLOCK_GROUPS; ...) {
//!     if (block_set[s]) {
//!         for (j = 0; j < n_vals; j++)
//!             atomicAdd(&out_j[s], block_acc_j[s]);
//!     }
//! }
//! ```
//!
//! ## PTX-level notes
//!
//! * The N shared accumulators are emitted as N independent `.shared .align 8
//!   .b8 block_acc_j_buf[8192]` declarations. Keeping them as named symbols
//!   (rather than one giant array sliced by `j*BLOCK_GROUPS*8`) makes the PTX
//!   easier to read, and the driver lays them out contiguously regardless.
//! * `block_set_buf` is shared across all aggregates — every value column
//!   writes the same `1` to the same byte for any given key, so there's no
//!   correctness concern.
//! * The row loop "unrolls" the per-aggregate work at PTX-emit time: with
//!   `n_vals=4` the loop body contains 4 `ld.global.f64` + 4
//!   `atom.shared.add.f64` instructions inline. There's no PTX-level `for`;
//!   we just write each iteration into the emitter's string.
//! * The kernel name carries `n_vals` so each width gets its own JIT module
//!   in the cache. `MAX_VALS=4` keeps the cache to four entries, all
//!   compiled once on first use.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

/// Number of slots in each block's shared-memory accumulator table. Matches
/// the single-SUM kernel so the two share dispatcher constants.
pub const BLOCK_GROUPS: u32 = 1024;

/// Threads per block. Same reasoning as the single-SUM kernel — 256 threads
/// keeps the zero/merge tail short while leaving enough warps to hide
/// shared-mem-atomic latency in the row loop.
pub const BLOCK_THREADS: u32 = 256;

/// Maximum number of value columns this kernel will fold into one launch.
/// Capped at 4 so the worst-case shared-mem footprint
/// (`MAX_VALS * BLOCK_GROUPS * 8 + BLOCK_GROUPS = 33 KiB`) fits inside the
/// portable sm_70 48 KiB per-block static shared-mem limit with room to
/// spare. Queries with more SUMs fall back to the global-atomic path.
pub const MAX_VALS: u32 = 4;

/// Emit the PTX entry-point name for a multi-SUM kernel of the given width.
///
/// We carry `n_vals` in the name so the JIT cache can keep one compiled
/// module per width without name collisions.
pub fn kernel_entry(n_vals: u32) -> String {
    format!("bolt_groupby_shmem_multi_sum_f64_{n_vals}")
}

/// Generate PTX for the multi-SUM shared-memory GROUP BY kernel.
///
/// Kernel signature (PTX-level), with `N = n_vals`:
///
/// ```text
/// .visible .entry bolt_groupby_shmem_multi_sum_f64_N(
///     .param .u64 keys_ptr,            // const int32_t* keys
///     .param .u64 vals_0_ptr,          // const double*  vals_0
///     .param .u64 vals_1_ptr,          // (only if N >= 2)
///     .param .u64 vals_2_ptr,          // (only if N >= 3)
///     .param .u64 vals_3_ptr,          // (only if N >= 4)
///     .param .u64 out_0_ptr,           // double*        out_0[n_groups]
///     .param .u64 out_1_ptr,           // (only if N >= 2)
///     .param .u64 out_2_ptr,           // (only if N >= 3)
///     .param .u64 out_3_ptr,           // (only if N >= 4)
///     .param .u32 n_rows,
///     .param .u32 n_groups
/// )
/// ```
///
/// # Errors
///
/// Returns an error if `n_vals` is outside `1..=MAX_VALS`. Same-input ->
/// same-output: a JIT cache can memoise on `n_vals` alone.
pub fn compile_shmem_multi_sum_kernel(n_vals: u32) -> BoltResult<String> {
    if n_vals == 0 || n_vals > MAX_VALS {
        return Err(BoltError::Other(format!(
            "shmem_multi_sum_kernel: n_vals must be in 1..={MAX_VALS}, got {n_vals}"
        )));
    }

    let mut ptx = String::new();
    let entry = kernel_entry(n_vals);
    let block_groups = BLOCK_GROUPS;
    let acc_bytes = BLOCK_GROUPS * 8;
    let set_bytes = BLOCK_GROUPS;
    let block_threads = BLOCK_THREADS;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // --- Shared-memory layout --------------------------------------------
    // One f64 accumulator per value column, plus one shared `set` flag
    // array (shared across all value columns — see module docs).
    //
    // Worst case (n_vals=MAX_VALS=4): 4*8192 + 1024 = 33792 bytes,
    // comfortably below 48 KiB.
    for j in 0..n_vals {
        writeln!(
            ptx,
            ".shared .align 8 .b8 block_acc_{j}_buf[{bytes}];",
            j = j,
            bytes = acc_bytes
        )
        .map_err(write_err)?;
    }
    writeln!(
        ptx,
        ".shared .align 1 .b8 block_set_buf[{bytes}];",
        bytes = set_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // --- Entry point + parameter list ------------------------------------
    // Param order:
    //   0           : keys_ptr
    //   1..=n_vals  : vals_j_ptr  for j in 0..n_vals
    //   n_vals+1..=2*n_vals : out_j_ptr  for j in 0..n_vals
    //   2*n_vals+1  : n_rows
    //   2*n_vals+2  : n_groups
    let total_params = 1 + n_vals + n_vals + 2;
    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    for i in 0..total_params {
        // .u32 for the trailing two scalars, .u64 for the rest.
        let is_scalar = i >= total_params - 2;
        let comma = if i + 1 < total_params { "," } else { "" };
        let ty = if is_scalar { "u32" } else { "u64" };
        writeln!(ptx, "\t.param .{ty} {entry}_param_{i}{comma}").map_err(write_err)?;
    }
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Generous register pools. We index registers numerically and pack
    // per-aggregate temporaries densely so the count scales linearly with
    // `n_vals`. MAX_VALS=4 -> at most ~16 extra .b64 regs over the
    // single-SUM kernel.
    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<64>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<128>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<32>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // --- thread coordinates ----------------------------------------------
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

    // n_rows lives at param index 2*n_vals + 1.
    let n_rows_param = 1 + n_vals + n_vals;
    writeln!(
        ptx,
        "\tld.param.u32 %r6, [{entry}_param_{p}];",
        p = n_rows_param
    )
    .map_err(write_err)?;

    // --- Shared base addresses -------------------------------------------
    // %rd_acc[j] = base of block_acc_j_buf (we use %rd(10+j))
    // %rd_set    = base of block_set_buf   (we use %rd9)
    for j in 0..n_vals {
        writeln!(
            ptx,
            "\tmov.u64 %rd{idx}, block_acc_{j}_buf;",
            idx = 10 + j,
            j = j
        )
        .map_err(write_err)?;
    }
    writeln!(ptx, "\tmov.u64 %rd9, block_set_buf;").map_err(write_err)?;

    // --- Global pointers (one keys_ptr + n_vals vals + n_vals outs) ------
    // %rd20      = keys  pointer
    // %rd30+j    = vals_j pointer
    // %rd40+j    = out_j  pointer
    writeln!(
        ptx,
        "\tld.param.u64 %rd20, [{entry}_param_0];"
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd20, %rd20;").map_err(write_err)?;
    for j in 0..n_vals {
        let pidx = 1 + j;
        writeln!(
            ptx,
            "\tld.param.u64 %rd{rd}, [{entry}_param_{p}];",
            rd = 30 + j,
            p = pidx
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tcvta.to.global.u64 %rd{rd}, %rd{rd};",
            rd = 30 + j
        )
        .map_err(write_err)?;
    }
    for j in 0..n_vals {
        let pidx = 1 + n_vals + j;
        writeln!(
            ptx,
            "\tld.param.u64 %rd{rd}, [{entry}_param_{p}];",
            rd = 40 + j,
            p = pidx
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tcvta.to.global.u64 %rd{rd}, %rd{rd};",
            rd = 40 + j
        )
        .map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    // ---------------------------------------------------------------------
    // Phase 1: zero shared memory (every accumulator + the set flags).
    // ---------------------------------------------------------------------
    // %r10 = zero-loop index (start at threadIdx.x, stride BLOCK_THREADS).
    writeln!(ptx, "\tmov.u32 %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p0, %r10, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    // 8-byte slot offset once, reused for every accumulator.
    writeln!(ptx, "\tmul.wide.u32 %rd50, %r10, 8;").map_err(write_err)?;
    for j in 0..n_vals {
        // block_acc_j[%r10] = 0.0 (f64 zero == u64 zero bits)
        writeln!(
            ptx,
            "\tadd.s64 %rd{rd}, %rd{base}, %rd50;",
            rd = 51 + j,
            base = 10 + j
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tst.shared.u64 [%rd{rd}], 0;",
            rd = 51 + j
        )
        .map_err(write_err)?;
    }
    // block_set[%r10] = 0
    writeln!(ptx, "\tcvt.u64.u32 %rd60, %r10;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd61, %rd9, %rd60;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u8 [%rd61], 0;").map_err(write_err)?;
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

    // ---------------------------------------------------------------------
    // Phase 2: grid-stride loop over rows.
    // ---------------------------------------------------------------------
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r3, %r6;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    // key = keys[i]
    writeln!(ptx, "\tmul.wide.u32 %rd70, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd71, %rd20, %rd70;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s32 %r12, [%rd71];").map_err(write_err)?;

    // Per-aggregate row value load: %fd(j) = vals_j[i].
    writeln!(ptx, "\tmul.wide.u32 %rd72, %r3, 8;").map_err(write_err)?;
    for j in 0..n_vals {
        writeln!(
            ptx,
            "\tadd.s64 %rd{rd}, %rd{base}, %rd72;",
            rd = 73 + j,
            base = 30 + j
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tld.global.f64 %fd{fd}, [%rd{rd}];",
            fd = j,
            rd = 73 + j
        )
        .map_err(write_err)?;
    }

    // Overflow check on key (unsigned comparison — see single-SUM notes).
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p2, %r12, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra OVERFLOW;").map_err(write_err)?;

    // Shared-mem accumulate: 8-byte offset reused across every accumulator.
    writeln!(ptx, "\tmul.wide.s32 %rd80, %r12, 8;").map_err(write_err)?;
    for j in 0..n_vals {
        writeln!(
            ptx,
            "\tadd.s64 %rd{rd}, %rd{base}, %rd80;",
            rd = 81 + j,
            base = 10 + j
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tatom.shared.add.f64 %fd{tmp}, [%rd{rd}], %fd{src};",
            tmp = 10 + j,
            rd = 81 + j,
            src = j
        )
        .map_err(write_err)?;
    }
    // block_set[key] = 1 (shared across all aggregates).
    writeln!(ptx, "\tcvt.s64.s32 %rd90, %r12;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd91, %rd9, %rd90;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r13, 1;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u8 [%rd91], %r13;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    // Overflow path: skip the shared table for this row; emit one
    // atom.global.add.f64 per aggregate, straight into its `out_j`.
    writeln!(ptx, "OVERFLOW:").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.s32 %rd92, %r12, 8;").map_err(write_err)?;
    for j in 0..n_vals {
        writeln!(
            ptx,
            "\tadd.s64 %rd{rd}, %rd{base}, %rd92;",
            rd = 93 + j,
            base = 40 + j
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tatom.global.add.f64 %fd{tmp}, [%rd{rd}], %fd{src};",
            tmp = 20 + j,
            rd = 93 + j,
            src = j
        )
        .map_err(write_err)?;
    }

    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r3, %r3, %r5;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ---------------------------------------------------------------------
    // Phase 3: merge block-local tables into the per-aggregate global outs.
    // ---------------------------------------------------------------------
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "MERGE_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p3, %r20, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra MERGE_DONE;").map_err(write_err)?;

    // Load block_set[%r20].
    writeln!(ptx, "\tcvt.u64.u32 %rd100, %r20;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd101, %rd9, %rd100;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u8 %r21, [%rd101];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p4, %r21, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra MERGE_NEXT;").map_err(write_err)?;

    // Slot is live: for each aggregate, atom.global.add.f64 into out_j.
    writeln!(ptx, "\tmul.wide.u32 %rd102, %r20, 8;").map_err(write_err)?;
    for j in 0..n_vals {
        // shared addr (block_acc_j + slot*8)
        writeln!(
            ptx,
            "\tadd.s64 %rd{rd}, %rd{base}, %rd102;",
            rd = 103 + j * 2,
            base = 10 + j
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tld.shared.f64 %fd{fd}, [%rd{rd}];",
            fd = 25 + j,
            rd = 103 + j * 2
        )
        .map_err(write_err)?;
        // global addr (out_j + slot*8)
        writeln!(
            ptx,
            "\tadd.s64 %rd{rd}, %rd{base}, %rd102;",
            rd = 104 + j * 2,
            base = 40 + j
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tatom.global.add.f64 %fd{tmp}, [%rd{rd}], %fd{src};",
            tmp = 29,
            rd = 104 + j * 2,
            src = 25 + j
        )
        .map_err(write_err)?;
    }

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

fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("shmem_multi_sum_kernel: write failed: {}", e))
}

// ---------------------------------------------------------------------------
// PTX-shape tests (host-only — do NOT require a GPU).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: collect every successful PTX emission across the full valid
    /// width range so each test can assert a property uniformly.
    fn compile_all() -> Vec<(u32, String)> {
        (1..=MAX_VALS)
            .map(|n| (n, compile_shmem_multi_sum_kernel(n).expect("compiles")))
            .collect()
    }

    /// Width-keyed entry-point names must collide with neither the
    /// single-SUM kernel name nor each other — the JIT cache keys on it.
    #[test]
    fn entry_name_matches_kernel_entry_helper() {
        for n in 1..=MAX_VALS {
            let ptx = compile_shmem_multi_sum_kernel(n).expect("compiles");
            let name = kernel_entry(n);
            let needle = format!(".visible .entry {}(", name);
            assert!(
                ptx.contains(&needle),
                "n_vals={n}: PTX must declare .visible .entry {name}(  — got:\n{ptx}"
            );
        }
    }

    /// Every width must declare exactly `n_vals` accumulator buffers plus
    /// one set-flag buffer in shared memory.
    #[test]
    fn emits_one_shared_accumulator_per_value_column() {
        for (n, ptx) in compile_all() {
            // n distinct accumulator declarations
            for j in 0..n {
                let needle = format!(".shared .align 8 .b8 block_acc_{j}_buf");
                assert!(
                    ptx.contains(&needle),
                    "n_vals={n}: missing shared accumulator for j={j}:\n{ptx}"
                );
            }
            // shared set-flag array shared across all accumulators
            assert!(
                ptx.contains(".shared .align 1 .b8 block_set_buf"),
                "n_vals={n}: missing block_set_buf:\n{ptx}"
            );
            // Count the total number of `.shared` declarations: must be
            // exactly n_vals (accumulators) + 1 (set flags). A stray extra
            // would silently bloat the per-block shared-mem budget.
            let shared_count = ptx.matches(".shared ").count();
            assert_eq!(
                shared_count,
                (n + 1) as usize,
                "n_vals={n}: expected {} shared decls, saw {}:\n{ptx}",
                n + 1,
                shared_count
            );
        }
    }

    /// Each width emits `2 * n_vals` shared-mem atomics in total:
    /// one per aggregate in the row loop (Phase 2).
    #[test]
    fn shared_atomic_count_scales_with_n_vals() {
        for (n, ptx) in compile_all() {
            let count = ptx.matches("atom.shared.add.f64").count();
            assert_eq!(
                count, n as usize,
                "n_vals={n}: expected {n} atom.shared.add.f64 (one per aggregate in row loop), saw {count}:\n{ptx}"
            );
        }
    }

    /// Each width emits `2 * n_vals` global atomics in total:
    /// n_vals in the OVERFLOW row-loop path + n_vals in the MERGE phase.
    #[test]
    fn global_atomic_count_scales_with_n_vals() {
        for (n, ptx) in compile_all() {
            let count = ptx.matches("atom.global.add.f64").count();
            assert_eq!(
                count,
                (2 * n) as usize,
                "n_vals={n}: expected {} atom.global.add.f64 (n_vals overflow + n_vals merge), saw {}:\n{ptx}",
                2 * n, count
            );
        }
    }

    /// `__syncthreads()` (`bar.sync 0`) must appear after both the
    /// zero-init phase and the accumulate phase — without them the merge
    /// would read uninitialised / racing state.
    #[test]
    fn syncthreads_between_phases() {
        for (n, ptx) in compile_all() {
            let count = ptx.matches("bar.sync 0").count();
            assert!(
                count >= 2,
                "n_vals={n}: need at least 2 bar.sync 0 (post-zero, post-accumulate), saw {count}:\n{ptx}"
            );
        }
    }

    /// PTX module preamble must match the rest of `src/jit/*`.
    #[test]
    fn ptx_header_matches_project_conventions() {
        for (n, ptx) in compile_all() {
            assert!(ptx.contains(".version 7.5"), "n_vals={n}: missing .version 7.5");
            assert!(ptx.contains(".target sm_70"), "n_vals={n}: missing .target sm_70");
            assert!(
                ptx.contains(".address_size 64"),
                "n_vals={n}: missing .address_size 64"
            );
        }
    }

    /// At MAX_VALS, total static shared memory MUST fit in sm_70's portable
    /// 48 KiB per-block budget. If a future bump pushes us over, the
    /// dispatcher would have to start asking the driver for the dynamic
    /// limit before launching — this test pins the boundary.
    #[test]
    fn worst_case_shared_mem_fits_under_48kb() {
        let bytes = MAX_VALS * BLOCK_GROUPS * 8 + BLOCK_GROUPS;
        assert!(
            bytes <= 49_152,
            "MAX_VALS={MAX_VALS} + BLOCK_GROUPS={BLOCK_GROUPS} requires {bytes} bytes of shared-mem; sm_70 portable limit is 49152"
        );
    }

    /// `compile_shmem_multi_sum_kernel(n_vals)` is pure: same input ->
    /// same output. The JIT cache memoises on `n_vals` alone.
    #[test]
    fn output_is_deterministic_per_width() {
        for n in 1..=MAX_VALS {
            let a = compile_shmem_multi_sum_kernel(n).expect("a");
            let b = compile_shmem_multi_sum_kernel(n).expect("b");
            assert_eq!(a, b, "n_vals={n}: emitter must be deterministic");
        }
    }

    /// Out-of-range widths must be rejected at emit time — silently falling
    /// through with `n_vals=0` (no shared accumulators!) would produce a
    /// kernel that compiles but never writes anything to `out`.
    #[test]
    fn rejects_out_of_range_widths() {
        assert!(compile_shmem_multi_sum_kernel(0).is_err());
        assert!(compile_shmem_multi_sum_kernel(MAX_VALS + 1).is_err());
        assert!(compile_shmem_multi_sum_kernel(99).is_err());
    }
}
