// SPDX-License-Identifier: Apache-2.0

//! Scatter kernel **that also records the per-row destination index**, for
//! deterministic multi-column scatter.
//!
//! ## Why this exists (correctness, not perf)
//!
//! The plain [`crate::jit::scatter_kernel`] uses `atomicAdd` on a
//! per-partition cursor to claim a destination slot for each row. When we
//! need to scatter N value columns aligned to the same key column, the
//! previous design re-ran scatter N times (once per value column) and
//! *assumed* that two scatter launches with identical inputs (same
//! `partition_ids`, same `offsets`, zeroed cursors, same launch geometry)
//! would assign the same destination slot to each row.
//!
//! That assumption is **not** part of the CUDA contract. The ordering of
//! concurrent `atomicAdd` operations is unspecified by NVIDIA — any driver
//! release, warp-scheduler tweak, or block-count change can break it.
//! Under such a break, row `i`'s `v1` would land in one slot and its `v2`
//! in a different slot, so `SUM(v1)` and `SUM(v2)` would be paired with the
//! wrong keys. This is silent data corruption with no easy in-flight check.
//!
//! ## The fix
//!
//! We make the destination-slot assignment deterministic *by construction*:
//! run the atomic-claim pass **once**, write the resulting slot index to a
//! `dest_idx[n_rows]` buffer, and then scatter every value column using
//! that buffer with no atomics at all (see
//! [`crate::jit::scatter_values_by_dest_idx_kernel`]). Each row's slot is
//! decided exactly once, by exactly one thread, in exactly one kernel
//! launch — there is no opportunity for cross-launch divergence.
//!
//! This kernel is "scatter + record destination". It writes the key column
//! to its claimed slot (same as the original scatter kernel) and also
//! records the slot index in `dest_idx[row]`. We deliberately do NOT take
//! a value column here — the value-scatter pass owns that — to keep the
//! atomic-claim launch single-purpose. (For multi-SUM we save the launch
//! we'd otherwise spend on column 0 by reusing this kernel's output, since
//! the indexed value scatter then runs N times instead of N-1+1.)
//!
//! ## Algorithm
//!
//! ```text
//! for (i = blockIdx.x*blockDim.x + tid; i < n_rows; i += gridDim.x*blockDim.x) {
//!     uint32_t pid       = partition_ids[i];
//!     uint32_t local_idx = atomicAdd(&partition_cursors[pid], 1);  // OLD
//!     uint32_t out_pos   = partition_offsets[pid] + local_idx;
//!     dest_idx[i]        = out_pos;          // <-- the new write
//!     out_keys[out_pos]  = keys[i];
//! }
//! ```
//!
//! ## PTX-level notes
//!
//! Same idioms as `scatter_kernel`:
//!   * Pointers via `.param .u64` → `cvta.to.global.u64`.
//!   * Cursor reservation via `atom.global.add.u32` — returns OLD value =
//!     per-partition slot index.
//!   * Stores: `st.global.u32` for the i32 key column and for the u32
//!     `dest_idx` write. No f64 store here; value-column scatter is a
//!     separate kernel.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

/// Threads per block. Matches `scatter_kernel`'s choice for symmetry.
pub const BLOCK_THREADS: u32 = 256;

/// Entry-point name embedded in the emitted PTX.
pub const KERNEL_ENTRY: &str = "bolt_scatter_with_dest_idx";

/// Generate PTX for the "scatter + record destination" kernel.
///
/// Kernel signature (PTX-level):
///
/// ```text
/// .visible .entry bolt_scatter_with_dest_idx(
///     .param .u64 keys_ptr,                // const int32_t*  keys[n_rows]
///     .param .u64 partition_ids_ptr,       // const uint32_t* partition_ids[n_rows]
///     .param .u64 partition_offsets_ptr,   // const uint32_t* offsets[NUM_PARTITIONS]
///     .param .u64 partition_cursors_ptr,   // uint32_t*       cursors[NUM_PARTITIONS] (init 0)
///     .param .u64 out_keys_ptr,            // int32_t*        out_keys[n_rows]
///     .param .u64 dest_idx_ptr,            // uint32_t*       dest_idx[n_rows]
///     .param .u32 n_rows
/// )
/// ```
///
/// Deterministic and pure — returns a fixed PTX string with no I/O.
pub fn compile_scatter_with_dest_idx_kernel() -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // 6 pointer params + 1 u32 length param = 7 .param lines.
    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?; // keys
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?; // partition_ids
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?; // partition_offsets
    writeln!(ptx, "\t.param .u64 {entry}_param_3,").map_err(write_err)?; // partition_cursors
    writeln!(ptx, "\t.param .u64 {entry}_param_4,").map_err(write_err)?; // out_keys
    writeln!(ptx, "\t.param .u64 {entry}_param_5,").map_err(write_err)?; // dest_idx
    writeln!(ptx, "\t.param .u32 {entry}_param_6").map_err(write_err)?;  // n_rows
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<32>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Thread coordinates.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r4, %nctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s32 %r5, %r4, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r6, [{entry}_param_6];").map_err(write_err)?;

    // Global pointer setup.
    //   %rd0 = keys (i32*)
    //   %rd1 = partition_ids (u32*)
    //   %rd2 = partition_offsets (u32*)
    //   %rd3 = partition_cursors (u32*)
    //   %rd4 = out_keys (i32*)
    //   %rd5 = dest_idx (u32*)
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
    writeln!(ptx).map_err(write_err)?;

    // Grid-stride loop:
    //   while (gtid < n_rows) {
    //       pid       = partition_ids[gtid];
    //       local_idx = atom.global.add.u32 [cursors + pid*4], 1;  // OLD
    //       out_pos   = partition_offsets[pid] + local_idx;
    //       dest_idx[gtid]    = out_pos;
    //       out_keys[out_pos] = keys[gtid];
    //       gtid += stride;
    //   }
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r6;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra LOOP_DONE;").map_err(write_err)?;

    // pid = partition_ids[gtid]
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd1, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;

    // cursor_addr = partition_cursors + pid * 4
    writeln!(ptx, "\tmul.wide.u32 %rd12, %r10, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd13, %rd3, %rd12;").map_err(write_err)?;

    // local_idx = atomic-add OLD value
    writeln!(ptx, "\tatom.global.add.u32 %r11, [%rd13], 1;").map_err(write_err)?;

    // offset = partition_offsets[pid]
    writeln!(ptx, "\tadd.s64 %rd14, %rd2, %rd12;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r12, [%rd14];").map_err(write_err)?;

    // out_pos = offset + local_idx
    writeln!(ptx, "\tadd.u32 %r13, %r12, %r11;").map_err(write_err)?;

    // dest_idx[gtid] = out_pos
    //   byte offset = gtid * 4  (already in %rd10)
    writeln!(ptx, "\tadd.s64 %rd15, %rd5, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd15], %r13;").map_err(write_err)?;

    // key = keys[gtid]
    writeln!(ptx, "\tadd.s64 %rd16, %rd0, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r14, [%rd16];").map_err(write_err)?;

    // out_keys[out_pos] = key
    writeln!(ptx, "\tmul.wide.u32 %rd17, %r13, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd18, %rd4, %rd17;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd18], %r14;").map_err(write_err)?;

    // Advance gtid by the grid stride.
    writeln!(ptx, "\tadd.u32 %r3, %r3, %r5;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("scatter_with_dest_idx_kernel: write failed: {}", e))
}

// ---------------------------------------------------------------------------
// PTX-shape tests (host-only — no GPU required).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Cursor reservation MUST use `atom.global.add.u32` so the OLD value
    /// becomes our local slot index.
    #[test]
    fn uses_atom_global_add_u32_for_cursor() {
        let ptx = compile_scatter_with_dest_idx_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("atom.global.add.u32"),
            "PTX must use atom.global.add.u32 to reserve partition slots:\n{ptx}"
        );
    }

    /// 6 pointer params + 1 u32 = 7 `.param` lines.
    #[test]
    fn has_seven_param_declarations() {
        let ptx = compile_scatter_with_dest_idx_kernel().expect("kernel compiles");
        let param_count = ptx.matches(".param ").count();
        assert_eq!(
            param_count, 7,
            "PTX must declare exactly 7 .param entries (6 pointers + 1 u32), \
             saw {param_count}:\n{ptx}"
        );
    }

    /// We write two i32-shape stores (key + dest_idx, both u32 byte width).
    /// We must NOT emit an f64 store — values are scattered by a separate
    /// kernel.
    #[test]
    fn emits_key_and_dest_idx_stores_but_no_value_store() {
        let ptx = compile_scatter_with_dest_idx_kernel().expect("kernel compiles");
        let u32_stores = ptx.matches("st.global.u32").count();
        assert!(
            u32_stores >= 2,
            "PTX must contain at least 2 st.global.u32 (for key and dest_idx writes), \
             saw {u32_stores}:\n{ptx}"
        );
        assert!(
            !ptx.contains("st.global.f64"),
            "PTX must NOT contain st.global.f64 — value scatter is a separate kernel:\n{ptx}"
        );
    }

    /// Entry-point name resolves via `cuModuleGetFunction`.
    #[test]
    fn has_correct_entry_declaration() {
        let ptx = compile_scatter_with_dest_idx_kernel().expect("kernel compiles");
        let needle = format!(".visible .entry {}(", KERNEL_ENTRY);
        assert!(
            ptx.contains(&needle),
            "PTX must declare .visible .entry {}(  — got:\n{ptx}",
            KERNEL_ENTRY
        );
    }

    /// Cheapest possible sanity check.
    #[test]
    fn compile_returns_non_empty_ptx() {
        let ptx = compile_scatter_with_dest_idx_kernel().expect("kernel compiles");
        assert!(!ptx.is_empty());
        assert!(
            ptx.contains(".version 7.5") && ptx.contains(".target sm_70"),
            "PTX must include the standard header lines:\n{ptx}"
        );
    }
}
