// SPDX-License-Identifier: Apache-2.0

//! Scatter kernel for Tier-2 hash-partitioned GROUP BY.
//!
//! ## Why this exists
//!
//! Tier-2 of the GROUP BY perf plan
//! ([`docs/GROUPBY_PERF.md`](../../../docs/GROUPBY_PERF.md)) is a two-pass
//! aggregation: a *partition* kernel assigns each input row to one of
//! `NUM_PARTITIONS` hash partitions, then this *scatter* kernel moves each
//! `(key, val)` pair into its partition's contiguous slot in the output
//! buffers. A subsequent per-partition Tier-1 shared-mem GROUP BY then runs
//! over the now-coalesced partition slices.
//!
//! The host (between the partition and scatter passes) does a single prefix
//! sum over the partition counts to produce `partition_offsets[k]`. The
//! scatter kernel does NOT recompute that prefix sum — it just uses the
//! offset table to place rows.
//!
//! ## Algorithm
//!
//! ```text
//! for (i = blockIdx.x*blockDim.x + tid; i < n_rows; i += gridDim.x*blockDim.x) {
//!     uint32_t pid       = partition_ids[i];
//!     uint32_t local_idx = atomicAdd(&partition_cursors[pid], 1);  // OLD value
//!     uint32_t out_pos   = partition_offsets[pid] + local_idx;
//!     out_keys[out_pos]  = keys[i];
//!     out_vals[out_pos]  = vals[i];
//! }
//! ```
//!
//! The trick is that `atom.global.add.u32` returns the OLD value of the
//! cursor. That OLD value is exactly the row's index *within* the partition:
//! the first thread to touch partition `k` gets 0, the second gets 1, and
//! so on. Adding `partition_offsets[k]` then yields a globally unique slot
//! across the whole output buffer.
//!
//! ## PTX-level notes
//!
//! * All pointers are passed via `.param .u64` and converted to global-state
//!   pointers via `cvta.to.global.u64`. We do not pass any pointer as
//!   `.restrict` — the rest of `src/jit/*` avoids it for portability with
//!   older driver versions.
//! * The cursor atomic is `atom.global.add.u32`. It needs to be `.global`
//!   because the cursor array lives in device memory shared across the whole
//!   grid (not per-block). The destination register receives the OLD value;
//!   we feed that straight into the slot computation.
//! * `out_keys` is `int32_t*` — store with `st.global.u32`. `out_vals` is
//!   `double*` — store with `st.global.f64`. The two stores are independent
//!   so the compiler can interleave them.
//! * No barriers needed: every thread only writes to a slot it just
//!   reserved via the atomic, so there's no inter-thread ordering to enforce
//!   within this kernel.

use std::fmt::Write;

use crate::error::{PatinaError, PatinaResult};

/// Number of hash partitions. Matches Tier-1 `BLOCK_GROUPS` so a downstream
/// per-partition Tier-1 kernel can fit each partition's group set in a
/// single shared-memory table.
pub const NUM_PARTITIONS: u32 = 4096;

/// Threads per block for the scatter kernel. The work per thread is
/// dominated by one `ld.global` + one `atom.global.add` + two `st.global`,
/// so 256 threads/block strikes a good balance between occupancy and
/// register pressure.
pub const BLOCK_THREADS: u32 = 256;

/// Entry-point name embedded in the emitted PTX.
pub const KERNEL_ENTRY: &str = "patina_scatter";

/// Generate PTX for the partition-scatter kernel.
///
/// Kernel signature (PTX-level):
///
/// ```text
/// .visible .entry patina_scatter(
///     .param .u64 keys_ptr,                // const int32_t*  keys[n_rows]
///     .param .u64 vals_ptr,                // const double*   vals[n_rows]
///     .param .u64 partition_ids_ptr,       // const uint32_t* partition_ids[n_rows]
///     .param .u64 partition_offsets_ptr,   // const uint32_t* offsets[NUM_PARTITIONS]
///     .param .u64 partition_cursors_ptr,   // uint32_t*       cursors[NUM_PARTITIONS] (init 0)
///     .param .u64 out_keys_ptr,            // int32_t*        out_keys[n_rows]
///     .param .u64 out_vals_ptr,            // double*         out_vals[n_rows]
///     .param .u32 n_rows
/// )
/// ```
///
/// `compile_scatter_kernel()` is deterministic and pure: it returns a fixed
/// PTX string with no I/O.
pub fn compile_scatter_kernel() -> PatinaResult<String> {
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // ---------------------------------------------------------------------
    // Entry point: 7 pointer params + 1 u32 length param = 8 .param lines.
    // ---------------------------------------------------------------------
    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?; // keys
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?; // vals
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?; // partition_ids
    writeln!(ptx, "\t.param .u64 {entry}_param_3,").map_err(write_err)?; // partition_offsets
    writeln!(ptx, "\t.param .u64 {entry}_param_4,").map_err(write_err)?; // partition_cursors
    writeln!(ptx, "\t.param .u64 {entry}_param_5,").map_err(write_err)?; // out_keys
    writeln!(ptx, "\t.param .u64 {entry}_param_6,").map_err(write_err)?; // out_vals
    writeln!(ptx, "\t.param .u32 {entry}_param_7").map_err(write_err)?;  // n_rows
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Register decls. Same generous counts as shmem_sum_kernel; the SASS
    // compiler will trim what we don't use.
    writeln!(ptx, "\t.reg .pred  %p<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<4>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // -----------------------------------------------------------------
    // Thread coordinates
    //   %r0 = blockIdx.x
    //   %r1 = blockDim.x
    //   %r2 = threadIdx.x
    //   %r3 = gtid    = blockIdx.x * blockDim.x + threadIdx.x
    //   %r4 = gridDim.x
    //   %r5 = stride  = gridDim.x * blockDim.x
    //   %r6 = n_rows  (loaded from .param)
    // -----------------------------------------------------------------
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r4, %nctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s32 %r5, %r4, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r6, [{entry}_param_7];").map_err(write_err)?;

    // -----------------------------------------------------------------
    // Global pointer setup (cvta.to.global.u64 from each .param)
    //   %rd0 = keys (i32*)
    //   %rd1 = vals (f64*)
    //   %rd2 = partition_ids (u32*)
    //   %rd3 = partition_offsets (u32*)
    //   %rd4 = partition_cursors (u32*)
    //   %rd5 = out_keys (i32*)
    //   %rd6 = out_vals (f64*)
    // -----------------------------------------------------------------
    writeln!(ptx, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd1, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd2, [{entry}_param_2];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd3, [{entry}_param_3];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd4, [{entry}_param_4];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd5, [{entry}_param_5];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd5, %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd6, [{entry}_param_6];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd6, %rd6;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // -----------------------------------------------------------------
    // Grid-stride loop:
    //   while (gtid < n_rows) {
    //       pid       = partition_ids[gtid];
    //       local_idx = atom.global.add.u32 [cursors + pid*4], 1;  // OLD
    //       out_pos   = partition_offsets[pid] + local_idx;
    //       out_keys[out_pos] = keys[gtid];
    //       out_vals[out_pos] = vals[gtid];
    //       gtid += stride;
    //   }
    // -----------------------------------------------------------------
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r6;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra LOOP_DONE;").map_err(write_err)?;

    // pid = partition_ids[gtid]   (u32)
    //   byte offset = gtid * 4
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd2, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;

    // cursor_addr = partition_cursors + pid * 4
    writeln!(ptx, "\tmul.wide.u32 %rd12, %r10, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd13, %rd4, %rd12;").map_err(write_err)?;

    // local_idx = atom.global.add.u32 [cursor_addr], 1  -- returns OLD value
    //
    // This is the load-bearing instruction of the kernel. PTX atomic-add
    // returns the value that was at the address BEFORE the add, which is
    // exactly the row's slot index within its partition (0 for the first
    // arrival, 1 for the second, ...). No further per-partition coordination
    // is needed.
    writeln!(ptx, "\tatom.global.add.u32 %r11, [%rd13], 1;").map_err(write_err)?;

    // offset = partition_offsets[pid]
    writeln!(ptx, "\tadd.s64 %rd14, %rd3, %rd12;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r12, [%rd14];").map_err(write_err)?;

    // out_pos = offset + local_idx   (u32)
    writeln!(ptx, "\tadd.u32 %r13, %r12, %r11;").map_err(write_err)?;

    // key = keys[gtid]
    //   byte offset already computed as %rd10 = gtid * 4
    writeln!(ptx, "\tadd.s64 %rd15, %rd0, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r14, [%rd15];").map_err(write_err)?;

    // val = vals[gtid]
    //   byte offset = gtid * 8
    writeln!(ptx, "\tmul.wide.u32 %rd16, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd17, %rd1, %rd16;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.f64 %fd0, [%rd17];").map_err(write_err)?;

    // out_keys[out_pos] = key
    //   byte offset = out_pos * 4
    writeln!(ptx, "\tmul.wide.u32 %rd18, %r13, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd19, %rd5, %rd18;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd19], %r14;").map_err(write_err)?;

    // out_vals[out_pos] = val
    //   byte offset = out_pos * 8
    writeln!(ptx, "\tmul.wide.u32 %rd20, %r13, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, %rd6, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.f64 [%rd21], %fd0;").map_err(write_err)?;

    // Advance gtid by the grid stride.
    writeln!(ptx, "\tadd.u32 %r3, %r3, %r5;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Adapt a `std::fmt::Error` into a `PatinaError`. Same shape as the helper
/// in `shmem_sum_kernel.rs`.
fn write_err(e: std::fmt::Error) -> PatinaError {
    PatinaError::Other(format!("scatter_kernel: write failed: {}", e))
}

// ---------------------------------------------------------------------------
// PTX-shape tests (host-only — no GPU required).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// The cursor reservation must use `atom.global.add.u32`. Returning the
    /// OLD value of the cursor is what gives us a unique within-partition
    /// slot index per row without any further synchronisation.
    #[test]
    fn uses_atom_global_add_u32_for_cursor() {
        let ptx = compile_scatter_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("atom.global.add.u32"),
            "PTX must use atom.global.add.u32 to reserve partition slots \
             (returns OLD value as local_idx):\n{ptx}"
        );
    }

    /// The kernel takes 7 pointer params (keys, vals, partition_ids,
    /// partition_offsets, partition_cursors, out_keys, out_vals) plus 1
    /// scalar param (n_rows). That's exactly 8 `.param` declarations.
    #[test]
    fn has_eight_param_declarations() {
        let ptx = compile_scatter_kernel().expect("kernel compiles");
        let param_count = ptx.matches(".param ").count();
        assert_eq!(
            param_count, 8,
            "PTX must declare exactly 8 .param entries (7 pointers + 1 u32), \
             saw {param_count}:\n{ptx}"
        );
    }

    /// Both output stores must appear: `st.global.u32` for the int32 key
    /// column and `st.global.f64` for the f64 value column. Forgetting
    /// either store would silently drop one of the scattered columns.
    #[test]
    fn emits_both_output_stores() {
        let ptx = compile_scatter_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("st.global.u32"),
            "PTX must contain st.global.u32 to write the int32 key column:\n{ptx}"
        );
        assert!(
            ptx.contains("st.global.f64"),
            "PTX must contain st.global.f64 to write the f64 value column:\n{ptx}"
        );
    }

    /// The entry point name must match `KERNEL_ENTRY` so the dispatcher can
    /// look it up with `cuModuleGetFunction`.
    #[test]
    fn has_correct_entry_declaration() {
        let ptx = compile_scatter_kernel().expect("kernel compiles");
        let needle = format!(".visible .entry {}(", KERNEL_ENTRY);
        assert!(
            ptx.contains(&needle),
            "PTX must declare .visible .entry {}(  — got:\n{ptx}",
            KERNEL_ENTRY
        );
    }

    /// `compile_scatter_kernel()` must succeed and return a non-empty PTX
    /// string. Cheapest possible sanity check.
    #[test]
    fn compile_returns_non_empty_ptx() {
        let ptx = compile_scatter_kernel().expect("kernel compiles");
        assert!(
            !ptx.is_empty(),
            "compile_scatter_kernel() must return a non-empty PTX string"
        );
        // Belt-and-suspenders: header must be present too, since "non-empty"
        // could otherwise hide a half-baked emit.
        assert!(
            ptx.contains(".version 7.5") && ptx.contains(".target sm_70"),
            "PTX must include the standard header lines:\n{ptx}"
        );
    }
}
