// SPDX-License-Identifier: Apache-2.0

//! Scatter kernel for Tier-2 hash-partitioned GROUP BY — **i64-key sibling**.
//!
//! Functionally identical to [`crate::jit::scatter_kernel`] except:
//!   * `keys`     is `int64_t*`  (8-byte load, `ld.global.s64`).
//!   * `out_keys` is `int64_t*`  (8-byte store, `st.global.u64`).
//!
//! Everything else — `vals` (f64), `partition_ids` (u32), `partition_offsets`
//! (u32), `partition_cursors` (u32), the per-partition cursor reservation via
//! `atom.global.add.u32`, the grid-stride loop — is unchanged. See the i32
//! sibling's module docs for the full algorithm rationale.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

/// Number of hash partitions. Matches both Tier-1 `BLOCK_GROUPS` and the
/// partition-kernel sibling so all three kernels in the Tier-2 chain agree
/// on K.
pub const NUM_PARTITIONS: u32 = 4096;

/// Threads per block. Mirrors the i32 scatter sibling.
pub const BLOCK_THREADS: u32 = 256;

/// Entry-point name embedded in the emitted PTX. Distinct from the i32
/// sibling so both kernels can co-exist in the same CUDA context.
pub const KERNEL_ENTRY: &str = "bolt_scatter_i64";

/// Entry-point name for the **typed i64-value** variant. The kernel shape
/// matches `bolt_scatter_i64` (i64 keys, i64 vals) but the val element is
/// `int64_t` instead of `double` — `ld.global.s64` / `st.global.u64` on the
/// val path. Used by Tier-2 MIN/MAX Int64 to skip the lossy `i64 -> f64`
/// host round-trip that the f64-val sibling forces.
pub const KERNEL_ENTRY_I64_TO_I64: &str = "bolt_scatter_i64_to_i64";

/// Generate PTX for the i64-key partition-scatter kernel.
///
/// Kernel signature (PTX-level):
///
/// ```text
/// .visible .entry bolt_scatter_i64(
///     .param .u64 keys_ptr,                // const int64_t*  keys[n_rows]
///     .param .u64 vals_ptr,                // const double*   vals[n_rows]
///     .param .u64 partition_ids_ptr,       // const uint32_t* partition_ids[n_rows]
///     .param .u64 partition_offsets_ptr,   // const uint32_t* offsets[NUM_PARTITIONS]
///     .param .u64 partition_cursors_ptr,   // uint32_t*       cursors[NUM_PARTITIONS] (init 0)
///     .param .u64 out_keys_ptr,            // int64_t*        out_keys[n_rows]
///     .param .u64 out_vals_ptr,            // double*         out_vals[n_rows]
///     .param .u32 n_rows
/// )
/// ```
///
/// `compile_scatter_kernel_i64()` is deterministic and pure.
pub fn compile_scatter_kernel_i64() -> BoltResult<String> {
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
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?; // keys (i64)
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?; // vals (f64)
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?; // partition_ids
    writeln!(ptx, "\t.param .u64 {entry}_param_3,").map_err(write_err)?; // partition_offsets
    writeln!(ptx, "\t.param .u64 {entry}_param_4,").map_err(write_err)?; // partition_cursors
    writeln!(ptx, "\t.param .u64 {entry}_param_5,").map_err(write_err)?; // out_keys (i64)
    writeln!(ptx, "\t.param .u64 {entry}_param_6,").map_err(write_err)?; // out_vals (f64)
    writeln!(ptx, "\t.param .u32 {entry}_param_7").map_err(write_err)?; // n_rows
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Register decls — same generous shape as the i32 sibling.
    writeln!(ptx, "\t.reg .pred  %p<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<48>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<4>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // -----------------------------------------------------------------
    // Thread coordinates (identical to i32 sibling).
    // -----------------------------------------------------------------
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r4, %nctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s32 %r5, %r4, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r6, [{entry}_param_7];").map_err(write_err)?;

    // -----------------------------------------------------------------
    // Global pointer setup.
    //   %rd0 = keys (i64*)
    //   %rd1 = vals (f64*)
    //   %rd2 = partition_ids (u32*)
    //   %rd3 = partition_offsets (u32*)
    //   %rd4 = partition_cursors (u32*)
    //   %rd5 = out_keys (i64*)
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
    // Grid-stride loop — identical layout to the i32 sibling except for
    // the i64 key load/store (8-byte stride, ld.global.s64 / st.global.u64).
    // -----------------------------------------------------------------
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r6;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra LOOP_DONE;").map_err(write_err)?;

    // pid = partition_ids[gtid]   (u32, stride 4)
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd2, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;

    // cursor_addr = partition_cursors + pid * 4
    writeln!(ptx, "\tmul.wide.u32 %rd12, %r10, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd13, %rd4, %rd12;").map_err(write_err)?;

    // local_idx = atom.global.add.u32 [cursor_addr], 1  -- returns OLD value.
    writeln!(ptx, "\tatom.global.add.u32 %r11, [%rd13], 1;").map_err(write_err)?;

    // offset = partition_offsets[pid]
    writeln!(ptx, "\tadd.s64 %rd14, %rd3, %rd12;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r12, [%rd14];").map_err(write_err)?;

    // out_pos = offset + local_idx   (u32)
    writeln!(ptx, "\tadd.u32 %r13, %r12, %r11;").map_err(write_err)?;

    // key = keys[gtid]   (i64 load, stride 8)
    writeln!(ptx, "\tmul.wide.u32 %rd16, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd17, %rd0, %rd16;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rd30, [%rd17];").map_err(write_err)?;

    // val = vals[gtid]   (f64 load, stride 8 — same byte stride as i64 key)
    writeln!(ptx, "\tadd.s64 %rd18, %rd1, %rd16;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.f64 %fd0, [%rd18];").map_err(write_err)?;

    // out_keys[out_pos] = key   (i64 store, stride 8)
    writeln!(ptx, "\tmul.wide.u32 %rd19, %r13, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd20, %rd5, %rd19;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u64 [%rd20], %rd30;").map_err(write_err)?;

    // out_vals[out_pos] = val   (f64 store, stride 8)
    writeln!(ptx, "\tadd.s64 %rd21, %rd6, %rd19;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.f64 [%rd21], %fd0;").map_err(write_err)?;

    // Advance gtid by the grid stride.
    writeln!(ptx, "\tadd.u32 %r3, %r3, %r5;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Generate PTX for the i64-key + **i64-val** partition-scatter kernel.
///
/// Mirrors [`compile_scatter_kernel_i64`] in shape and algorithm. The only
/// difference is the val element type:
///   * `vals`     is `int64_t*` (8-byte load, `ld.global.s64`).
///   * `out_vals` is `int64_t*` (8-byte store, `st.global.u64`).
///
/// Used by the Tier-2 MIN/MAX Int64 chain so values >2^53 round-trip
/// losslessly through the scatter (the f64-val sibling silently narrows
/// them via the host-side `i64 as f64` cast).
///
/// Kernel signature (PTX-level):
///
/// ```text
/// .visible .entry bolt_scatter_i64_to_i64(
///     .param .u64 keys_ptr,                // const int64_t*  keys[n_rows]
///     .param .u64 vals_ptr,                // const int64_t*  vals[n_rows]
///     .param .u64 partition_ids_ptr,       // const uint32_t* partition_ids[n_rows]
///     .param .u64 partition_offsets_ptr,   // const uint32_t* offsets[NUM_PARTITIONS]
///     .param .u64 partition_cursors_ptr,   // uint32_t*       cursors[NUM_PARTITIONS] (init 0)
///     .param .u64 out_keys_ptr,            // int64_t*        out_keys[n_rows]
///     .param .u64 out_vals_ptr,            // int64_t*        out_vals[n_rows]
///     .param .u32 n_rows
/// )
/// ```
pub fn compile_scatter_kernel_i64_to_i64() -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = KERNEL_ENTRY_I64_TO_I64;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?; // keys (i64)
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?; // vals (i64)
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?; // partition_ids
    writeln!(ptx, "\t.param .u64 {entry}_param_3,").map_err(write_err)?; // partition_offsets
    writeln!(ptx, "\t.param .u64 {entry}_param_4,").map_err(write_err)?; // partition_cursors
    writeln!(ptx, "\t.param .u64 {entry}_param_5,").map_err(write_err)?; // out_keys (i64)
    writeln!(ptx, "\t.param .u64 {entry}_param_6,").map_err(write_err)?; // out_vals (i64)
    writeln!(ptx, "\t.param .u32 {entry}_param_7").map_err(write_err)?; // n_rows
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Register decls — same shape as the f64-val sibling but no .f64 regs
    // (vals are loaded into a .b64 GPR instead).
    writeln!(ptx, "\t.reg .pred  %p<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<48>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Thread coordinates.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r4, %nctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s32 %r5, %r4, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r6, [{entry}_param_7];").map_err(write_err)?;

    // Global pointer setup.
    //   %rd0 = keys (i64*)
    //   %rd1 = vals (i64*)
    //   %rd2 = partition_ids (u32*)
    //   %rd3 = partition_offsets (u32*)
    //   %rd4 = partition_cursors (u32*)
    //   %rd5 = out_keys (i64*)
    //   %rd6 = out_vals (i64*)
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

    // Grid-stride loop. Same layout as the f64-val sibling except the val
    // load/store uses .s64/.u64 instead of .f64.
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r6;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra LOOP_DONE;").map_err(write_err)?;

    // pid = partition_ids[gtid]   (u32, stride 4)
    writeln!(ptx, "\tmul.wide.u32 %rd10, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd2, %rd10;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd11];").map_err(write_err)?;

    // cursor_addr = partition_cursors + pid * 4
    writeln!(ptx, "\tmul.wide.u32 %rd12, %r10, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd13, %rd4, %rd12;").map_err(write_err)?;

    // local_idx = atom.global.add.u32 [cursor_addr], 1  -- returns OLD value.
    writeln!(ptx, "\tatom.global.add.u32 %r11, [%rd13], 1;").map_err(write_err)?;

    // offset = partition_offsets[pid]
    writeln!(ptx, "\tadd.s64 %rd14, %rd3, %rd12;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r12, [%rd14];").map_err(write_err)?;

    // out_pos = offset + local_idx   (u32)
    writeln!(ptx, "\tadd.u32 %r13, %r12, %r11;").map_err(write_err)?;

    // key = keys[gtid]   (i64 load, stride 8)
    writeln!(ptx, "\tmul.wide.u32 %rd16, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd17, %rd0, %rd16;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rd30, [%rd17];").map_err(write_err)?;

    // val = vals[gtid]   (i64 load, stride 8)
    writeln!(ptx, "\tadd.s64 %rd18, %rd1, %rd16;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rd31, [%rd18];").map_err(write_err)?;

    // out_keys[out_pos] = key   (i64 store, stride 8)
    writeln!(ptx, "\tmul.wide.u32 %rd19, %r13, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd20, %rd5, %rd19;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u64 [%rd20], %rd30;").map_err(write_err)?;

    // out_vals[out_pos] = val   (i64 store, stride 8)
    writeln!(ptx, "\tadd.s64 %rd21, %rd6, %rd19;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u64 [%rd21], %rd31;").map_err(write_err)?;

    // Advance gtid by the grid stride.
    writeln!(ptx, "\tadd.u32 %r3, %r3, %r5;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("scatter_kernel_i64: write failed: {}", e))
}

// ---------------------------------------------------------------------------
// PTX-shape tests (host-only — no GPU required).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// The cursor reservation is the load-bearing atomic — same as the i32
    /// sibling. Returning the OLD value of the cursor gives each row a unique
    /// within-partition slot.
    #[test]
    fn uses_atom_global_add_u32_for_cursor() {
        let ptx = compile_scatter_kernel_i64().expect("kernel compiles");
        assert!(
            ptx.contains("atom.global.add.u32"),
            "PTX must use atom.global.add.u32 for the cursor:\n{ptx}"
        );
    }

    /// Same param layout as the i32 sibling: 7 pointer params + 1 u32. The
    /// orchestrator passes args positionally; a regression in the count
    /// would mis-bind every pointer.
    #[test]
    fn has_eight_param_declarations() {
        let ptx = compile_scatter_kernel_i64().expect("kernel compiles");
        let param_count = ptx.matches(".param ").count();
        assert_eq!(
            param_count, 8,
            "PTX must declare exactly 8 .param entries (7 pointers + 1 u32), \
             saw {param_count}:\n{ptx}"
        );
    }

    /// i64 keys MUST round-trip via 8-byte load + 8-byte store. A
    /// regression that left a 4-byte load in place (copy-paste from the
    /// i32 sibling) would silently truncate every output key.
    #[test]
    fn uses_i64_key_load_and_store() {
        let ptx = compile_scatter_kernel_i64().expect("kernel compiles");
        assert!(
            ptx.contains("ld.global.s64"),
            "PTX must use ld.global.s64 to read int64 keys:\n{ptx}"
        );
        assert!(
            ptx.contains("st.global.u64"),
            "PTX must use st.global.u64 to write int64 keys:\n{ptx}"
        );
    }

    /// Entry name must be the i64-specific symbol so both kernels can be
    /// loaded into the same CUDA context without colliding.
    #[test]
    fn has_correct_entry_declaration() {
        let ptx = compile_scatter_kernel_i64().expect("kernel compiles");
        let needle = format!(".visible .entry {}(", KERNEL_ENTRY);
        assert!(
            ptx.contains(&needle),
            "PTX must declare .visible .entry {}(  — got:\n{ptx}",
            KERNEL_ENTRY
        );
        assert_eq!(KERNEL_ENTRY, "bolt_scatter_i64");
    }

    /// `compile_scatter_kernel_i64()` must succeed and return non-empty PTX
    /// with the standard Craton Bolt module header.
    #[test]
    fn compile_returns_non_empty_ptx() {
        let ptx = compile_scatter_kernel_i64().expect("kernel compiles");
        assert!(!ptx.is_empty(), "compile returned empty PTX");
        assert!(
            ptx.contains(".version 7.5") && ptx.contains(".target sm_70"),
            "PTX must include the standard Craton Bolt module header:\n{ptx}"
        );
    }

    // -----------------------------------------------------------------
    // Tests for the typed i64-key + i64-val variant
    // (`compile_scatter_kernel_i64_to_i64`).
    // -----------------------------------------------------------------

    /// Typed variant must round-trip i64 vals via `.s64` load + `.u64`
    /// store — NOT `.f64`. A regression that emitted the f64 load (copy-
    /// paste from the f64-val sibling) would silently re-introduce the
    /// >2^53 precision bug the C4 guard was working around.
    #[test]
    fn i64_to_i64_uses_typed_val_load_and_store() {
        let ptx = compile_scatter_kernel_i64_to_i64().expect("kernel compiles");
        // Two `ld.global.s64` occurrences: one for the key, one for the val.
        assert!(
            ptx.matches("ld.global.s64").count() >= 2,
            "typed i64-val PTX must use ld.global.s64 for BOTH key and val:\n{ptx}"
        );
        // Two `st.global.u64` occurrences: one for the key, one for the val.
        assert!(
            ptx.matches("st.global.u64").count() >= 2,
            "typed i64-val PTX must use st.global.u64 for BOTH key and val:\n{ptx}"
        );
    }

    /// The typed variant MUST NOT emit any `.f64` instructions — the whole
    /// point of this kernel is to keep i64 values out of f64 registers.
    #[test]
    fn i64_to_i64_has_no_f64_instructions() {
        let ptx = compile_scatter_kernel_i64_to_i64().expect("kernel compiles");
        assert!(
            !ptx.contains(".f64") && !ptx.contains("%fd"),
            "typed i64-val PTX must not reference any .f64 type or %fd register:\n{ptx}"
        );
    }

    /// Same param layout as the f64-val sibling: 7 pointer params + 1 u32.
    #[test]
    fn i64_to_i64_has_eight_param_declarations() {
        let ptx = compile_scatter_kernel_i64_to_i64().expect("kernel compiles");
        let param_count = ptx.matches(".param ").count();
        assert_eq!(
            param_count, 8,
            "typed PTX must declare exactly 8 .param entries (7 pointers + 1 u32), \
             saw {param_count}:\n{ptx}"
        );
    }

    /// Entry name must be the i64-to-i64 specific symbol so all three
    /// scatter kernels (i32-key/f64-val, i64-key/f64-val, i64-key/i64-val)
    /// can be loaded into the same CUDA context without colliding.
    #[test]
    fn i64_to_i64_has_correct_entry_declaration() {
        let ptx = compile_scatter_kernel_i64_to_i64().expect("kernel compiles");
        let needle = format!(".visible .entry {}(", KERNEL_ENTRY_I64_TO_I64);
        assert!(
            ptx.contains(&needle),
            "PTX must declare .visible .entry {}(  — got:\n{ptx}",
            KERNEL_ENTRY_I64_TO_I64
        );
        assert_eq!(KERNEL_ENTRY_I64_TO_I64, "bolt_scatter_i64_to_i64");
    }

    /// Cursor reservation is still the load-bearing atomic for the typed
    /// variant.
    #[test]
    fn i64_to_i64_uses_atom_global_add_u32_for_cursor() {
        let ptx = compile_scatter_kernel_i64_to_i64().expect("kernel compiles");
        assert!(
            ptx.contains("atom.global.add.u32"),
            "typed PTX must use atom.global.add.u32 for the cursor:\n{ptx}"
        );
    }
}
