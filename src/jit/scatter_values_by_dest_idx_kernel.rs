// SPDX-License-Identifier: Apache-2.0

//! Deterministic value-column scatter using a precomputed `dest_idx[n_rows]`
//! buffer.
//!
//! Companion to [`crate::jit::scatter_with_dest_idx_kernel`]. The atomic-claim
//! pass runs once and writes `dest_idx[i] = out_pos` for every input row.
//! This kernel then writes one value column per launch, with no atomics:
//!
//! ```text
//! for (i = blockIdx.x*blockDim.x + tid; i < n_rows; i += gridDim.x*blockDim.x) {
//!     uint32_t out_pos    = dest_idx[i];
//!     out_vals[out_pos]   = vals[i];
//! }
//! ```
//!
//! ## Why this matters for correctness
//!
//! The previous multi-SUM design called the atomic-based scatter kernel N
//! times (once per value column) and relied on identical `atomicAdd` orderings
//! across launches to keep `(key, v1, v2, …)` aligned. That ordering is NOT
//! a CUDA contract; any driver/scheduler change can break it, silently
//! pairing `SUM(v1)` with the wrong key's `v2`. By precomputing `dest_idx`
//! once and reading it from this kernel, every value column lands at a
//! deterministic-by-construction slot — alignment cannot drift.
//!
//! ## PTX-level notes
//!
//! * No atomics. Pure load-from-dest_idx → store-to-vals.
//! * `out_vals[out_pos] = vals[i]` is one `ld.global.f64` followed by one
//!   `st.global.f64`. The store is scattered (out_pos varies), so it's the
//!   non-coalesced side; that's unavoidable for any scatter and is the same
//!   memory pattern the original kernel had.
//! * No barriers needed — each thread writes a unique slot (guaranteed by
//!   the upstream atomic-claim pass).

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

/// Threads per block. Matches the atomic-claim kernel.
pub const BLOCK_THREADS: u32 = 256;

/// Entry-point name embedded in the emitted PTX.
pub const KERNEL_ENTRY: &str = "bolt_scatter_values_by_dest_idx";

/// Generate PTX for the indexed value-column scatter kernel.
///
/// Kernel signature (PTX-level):
///
/// ```text
/// .visible .entry bolt_scatter_values_by_dest_idx(
///     .param .u64 vals_ptr,         // const double*   vals[n_rows]
///     .param .u64 dest_idx_ptr,     // const uint32_t* dest_idx[n_rows]
///     .param .u64 out_vals_ptr,     // double*         out_vals[n_rows]
///     .param .u32 n_rows
/// )
/// ```
///
/// Deterministic and pure — returns a fixed PTX string with no I/O.
pub fn compile_scatter_values_by_dest_idx_kernel() -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // 3 pointer params + 1 u32 length param = 4 .param lines.
    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?; // vals
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?; // dest_idx
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?; // out_vals
    writeln!(ptx, "\t.param .u32 {entry}_param_3").map_err(write_err)?;  // n_rows
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<4>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Thread coordinates.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r4, %nctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s32 %r5, %r4, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r6, [{entry}_param_3];").map_err(write_err)?;

    // Global pointer setup.
    //   %rd0 = vals (f64*)
    //   %rd1 = dest_idx (u32*)
    //   %rd2 = out_vals (f64*)
    writeln!(ptx, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd1, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd2, [{entry}_param_2];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Grid-stride loop:
    //   while (gtid < n_rows) {
    //       out_pos          = dest_idx[gtid];
    //       v                = vals[gtid];
    //       out_vals[out_pos] = v;
    //       gtid += stride;
    //   }
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r6;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra LOOP_DONE;").map_err(write_err)?;

    // out_pos = dest_idx[gtid]
    //   byte offset (u32) = gtid * 4
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd1, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;

    // v = vals[gtid]
    //   byte offset (f64) = gtid * 8
    writeln!(ptx, "\tmul.wide.u32 %rd12, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd13, %rd0, %rd12;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.f64 %fd0, [%rd13];").map_err(write_err)?;

    // out_vals[out_pos] = v
    //   byte offset = out_pos * 8
    writeln!(ptx, "\tmul.wide.u32 %rd14, %r10, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd15, %rd2, %rd14;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.f64 [%rd15], %fd0;").map_err(write_err)?;

    // Advance gtid by the grid stride.
    writeln!(ptx, "\tadd.u32 %r3, %r3, %r5;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!(
        "scatter_values_by_dest_idx_kernel: write failed: {}",
        e
    ))
}

// ---------------------------------------------------------------------------
// PTX-shape tests (host-only — no GPU required).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Critical: this kernel must NOT use atomics. The whole point of the
    /// dest_idx indirection is to eliminate atomicAdd from the value-scatter
    /// path so per-row alignment is deterministic across N launches.
    #[test]
    fn does_not_use_atomics() {
        let ptx = compile_scatter_values_by_dest_idx_kernel().expect("kernel compiles");
        assert!(
            !ptx.contains("atom."),
            "PTX must not contain any atomic op — the dest_idx indirection \
             must give deterministic placement across launches:\n{ptx}"
        );
    }

    /// 3 pointer params + 1 u32 = 4 `.param` lines.
    #[test]
    fn has_four_param_declarations() {
        let ptx = compile_scatter_values_by_dest_idx_kernel().expect("kernel compiles");
        let param_count = ptx.matches(".param ").count();
        assert_eq!(
            param_count, 4,
            "PTX must declare exactly 4 .param entries (3 pointers + 1 u32), \
             saw {param_count}:\n{ptx}"
        );
    }

    /// Must emit exactly one f64 store (value-column write).
    #[test]
    fn emits_one_f64_store() {
        let ptx = compile_scatter_values_by_dest_idx_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("st.global.f64"),
            "PTX must contain st.global.f64 for the value-column write:\n{ptx}"
        );
    }

    /// Must NOT touch the key column — that's the atomic-claim kernel's job.
    /// We verify by checking there's only one u32 load (dest_idx) and no u32
    /// store.
    #[test]
    fn does_not_write_key_column() {
        let ptx = compile_scatter_values_by_dest_idx_kernel().expect("kernel compiles");
        assert!(
            !ptx.contains("st.global.u32"),
            "PTX must NOT contain st.global.u32 — key column is written by \
             scatter_with_dest_idx, not by this kernel:\n{ptx}"
        );
    }

    /// Entry point name matches `KERNEL_ENTRY`.
    #[test]
    fn has_correct_entry_declaration() {
        let ptx = compile_scatter_values_by_dest_idx_kernel().expect("kernel compiles");
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
        let ptx = compile_scatter_values_by_dest_idx_kernel().expect("kernel compiles");
        assert!(!ptx.is_empty());
        assert!(
            ptx.contains(".version 7.5") && ptx.contains(".target sm_70"),
            "PTX must include the standard header lines:\n{ptx}"
        );
    }
}
