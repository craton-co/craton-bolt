// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for the multi-pass prefix-scan compaction path.
//!
//! The single-pass scan in [`crate::jit::prefix_scan`] tops out at
//! `n_rows <= u32::MAX / BLOCK_SIZE` (≈16.8M rows at `BLOCK_SIZE = 256`)
//! because the host-side scan over `block_sums` assumes the array fits in
//! a single host pass and the per-block index counter is a `u32`.
//!
//! For larger inputs we **recurse** the per-block scan over the
//! `block_sums` array itself, producing a smaller `block_sums_of_block_sums`,
//! until the top-level array fits in a single block (≤ `BLOCK_SIZE` entries)
//! and can be exclusive-scanned on the host. Walking back DOWN the
//! recursion, each level adds the parent level's `block_bases` into its own
//! per-row local indices to make them globally correct.
//!
//! This module emits the two PTX kernels that path needs *beyond* what
//! `prefix_scan.rs` already provides:
//!
//! 1. [`SCAN_U32_KERNEL_ENTRY`] — `bolt_prefix_scan_u32`: identical shape
//!    to the existing scan kernel, but reads a `u32*` input directly (no u8
//!    load, no truthiness coercion). Used to scan the intermediate
//!    `block_sums` arrays.
//! 2. [`ADD_BASES_KERNEL_ENTRY`] — `bolt_add_block_bases`: per-row,
//!    `indices[i] += block_bases[i / BLOCK_SIZE]`. Used between recursion
//!    levels to fold a parent's bases into a child's per-row local indices.
//!
//! The single-pass `bolt_prefix_scan` kernel is re-exported as-is so the
//! multipass execution module only needs one import surface.
//!
//! No new FFI symbols are introduced — these are JIT-compiled by the same
//! NVRTC path the existing scan kernel uses.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

// Re-export the single-pass scan kernel so callers can keep one import.
pub use crate::jit::prefix_scan::{compile_prefix_scan_kernel, SCAN_KERNEL_ENTRY};

use crate::jit::prefix_scan::BLOCK_SIZE;

/// PTX target metadata. Must match `prefix_scan.rs` so all three modules
/// load identically.
const PTX_VERSION: &str = ".version 7.5";
/// Target SM architecture string.
const PTX_TARGET: &str = ".target sm_70";
/// Address size directive (we always use 64-bit pointers).
const PTX_ADDRESS_SIZE: &str = ".address_size 64";

/// Entry-point name for the `u32`-input per-block prefix-scan kernel.
///
/// Used by [`prefix_scan_mask_multipass`](crate::exec::gpu_compact_multipass::prefix_scan_mask_multipass)
/// to scan the intermediate `block_sums` arrays produced by the previous
/// recursion level.
pub const SCAN_U32_KERNEL_ENTRY: &str = "bolt_prefix_scan_u32";

/// Entry-point name for the `add per-block base` kernel.
///
/// Per row: `indices[i] += block_bases[i / BLOCK_SIZE]`. Used to fold a
/// parent level's bases into a child level's per-row local indices on the
/// way back down the recursion.
pub const ADD_BASES_KERNEL_ENTRY: &str = "bolt_add_block_bases";

/// Generate the per-block exclusive prefix-scan PTX module for a `u32` input.
///
/// ABI:
/// ```text
/// .visible .entry bolt_prefix_scan_u32(
///     .param .u64 bolt_prefix_scan_u32_param_0,   // vals_ptr      (u32*)
///     .param .u64 bolt_prefix_scan_u32_param_1,   // local_indices (u32*)
///     .param .u64 bolt_prefix_scan_u32_param_2,   // block_sums    (u32*)
///     .param .u32 bolt_prefix_scan_u32_param_3    // n
/// )
/// ```
///
/// Identical in shape to [`compile_prefix_scan_kernel`] except the per-thread
/// input load reads a `u32` directly rather than a `u8` mask byte:
///
/// ```text
///   // prefix_scan.rs (u8 mask):                this kernel (u32 vals):
///   ld.global.u8  %rs0, [mask + gid]            ld.global.u32 %r5, [vals + gid*4]
///   cvt.u32.u16   %r6,  %rs0
///   setp.ne.s32   %p1,  %r6, 0
///   selp.s32      %r5,  1, 0, %p1               (no truthiness coercion — the
///                                                 raw count IS what we scan)
/// ```
///
/// All other PTX (Hillis-Steele rounds, ping-pong buffers, inclusive→exclusive
/// conversion, block-sum store at thread `BLOCK_SIZE - 1`) matches the
/// single-pass kernel beat-for-beat. Out-of-range lanes contribute zero, so
/// partial blocks remain correct.
pub fn compile_prefix_scan_u32_kernel() -> BoltResult<String> {
    // Two u32 buffers of BLOCK_SIZE entries each: ping-pong avoids one
    // bar.sync per round vs. reading and writing the same buffer.
    let elem_bytes: u32 = 4;
    let shared_bytes_per_buf: u32 = BLOCK_SIZE * elem_bytes;
    let shared_bytes_total: u32 = shared_bytes_per_buf * 2;

    let mut ptx = String::new();
    writeln!(ptx, "{}", PTX_VERSION).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_TARGET).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_ADDRESS_SIZE).map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(
        ptx,
        ".shared .align 4 .b8 sdata[{shared}];",
        shared = shared_bytes_total
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Signature: vals, local_indices, block_sums, n.
    writeln!(ptx, ".visible .entry {}(", SCAN_U32_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", SCAN_U32_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", SCAN_U32_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_2,", SCAN_U32_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_3", SCAN_U32_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Register pool. Matches `compile_prefix_scan_kernel`'s sizes.
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<32>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // -------- Indices: tid_x = %tid.x ; ctaid = %ctaid.x ; gid = ctaid*ntid+tid
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    // n
    writeln!(
        ptx,
        "\tld.param.u32 %r4, [{}_param_3];",
        SCAN_U32_KERNEL_ENTRY
    )
    .map_err(write_err)?;

    // -------- Globalize parameter pointers.
    writeln!(
        ptx,
        "\tld.param.u64 %rd0, [{}_param_0];",
        SCAN_U32_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u64 %rd1, [{}_param_1];",
        SCAN_U32_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u64 %rd2, [{}_param_2];",
        SCAN_U32_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;

    // -------- Load this thread's u32 value (0 if past the end). No mask
    // truthiness coercion — every value IS the count we're scanning.
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r5, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra AFTER_LOAD;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd3, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd4, %rd0, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r5, [%rd4];").map_err(write_err)?;
    writeln!(ptx, "AFTER_LOAD:").map_err(write_err)?;

    // %r5 is now this thread's input value. We also keep it for the
    // inclusive->exclusive conversion at the end.

    // -------- Stash the value into ping buffer (sdata[0..BLOCK_SIZE]).
    // base0 = sdata ; base1 = sdata + BLOCK_SIZE*4 ; idx_off = tid * 4
    // NOTE: keep `mov.u64` here (not `cvta.shared.u64`). %rd5 (and its derived
    // addresses %rd8/%rd9) feed `st.shared.u32` / `ld.shared.u32` below, which
    // require a shared-state-space address. `cvta.shared.u64` produces a
    // generic-space pointer; switching would force every shared ld/st in this
    // kernel to drop its `.shared` qualifier — outside this cleanup's scope.
    writeln!(ptx, "\tmov.u64 %rd5, sdata;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tadd.s64 %rd6, %rd5, {off};",
        off = shared_bytes_per_buf
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd7, %r2, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd8, %rd5, %rd7;").map_err(write_err)?; // ping addr
    writeln!(ptx, "\tadd.s64 %rd9, %rd6, %rd7;").map_err(write_err)?; // pong addr
    writeln!(ptx, "\tst.shared.u32 [%rd8], %r5;").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;

    // -------- Hillis-Steele scan: identical to the u8 variant.
    let mut offset: u32 = 1;
    let mut step: usize = 0;
    while offset < BLOCK_SIZE {
        let off_bytes = offset * elem_bytes;
        writeln!(ptx, "\tld.shared.u32 %r7, [%rd8];").map_err(write_err)?;
        writeln!(
            ptx,
            "\tsetp.lt.s32 %p{p}, %r2, {off};",
            p = 2 + (step % 4),
            off = offset
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tmov.u32 %r8, 0;").map_err(write_err)?;
        writeln!(
            ptx,
            "\t@%p{p} bra SKIP_LOAD_{step};",
            p = 2 + (step % 4),
            step = step
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tsub.s64 %rd10, %rd8, {off};",
            off = off_bytes
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tld.shared.u32 %r8, [%rd10];").map_err(write_err)?;
        writeln!(ptx, "SKIP_LOAD_{step}:", step = step).map_err(write_err)?;
        writeln!(ptx, "\tadd.s32 %r9, %r7, %r8;").map_err(write_err)?;
        writeln!(ptx, "\tst.shared.u32 [%rd9], %r9;").map_err(write_err)?;
        writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
        // Swap ping/pong slot pointers.
        writeln!(ptx, "\tmov.u64 %rd11, %rd8;").map_err(write_err)?;
        writeln!(ptx, "\tmov.u64 %rd8, %rd9;").map_err(write_err)?;
        writeln!(ptx, "\tmov.u64 %rd9, %rd11;").map_err(write_err)?;
        offset <<= 1;
        step += 1;
    }

    // After the loop, %rd8 points at this thread's slot in the buffer holding
    // the inclusive scan.
    writeln!(ptx, "\tld.shared.u32 %r10, [%rd8];").map_err(write_err)?;
    // Convert inclusive -> exclusive: excl = incl - own_value.
    writeln!(ptx, "\tsub.s32 %r11, %r10, %r5;").map_err(write_err)?;

    // -------- Write exclusive scan to local_indices[gid] (only in-range).
    writeln!(ptx, "\tsetp.ge.u32 %p5, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra AFTER_LOCAL_STORE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd12, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd13, %rd1, %rd12;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd13], %r11;").map_err(write_err)?;
    writeln!(ptx, "AFTER_LOCAL_STORE:").map_err(write_err)?;

    // -------- Thread (BLOCK_SIZE - 1) writes block_sums[blockIdx.x] = incl.
    writeln!(
        ptx,
        "\tsetp.ne.s32 %p6, %r2, {last};",
        last = BLOCK_SIZE - 1
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p6 bra DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.s32 %rd14, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd15, %rd2, %rd14;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd15], %r10;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Generate the `add per-block base` PTX module.
///
/// ABI:
/// ```text
/// .visible .entry bolt_add_block_bases(
///     .param .u64 bolt_add_block_bases_param_0,   // indices_ptr     (u32* in/out)
///     .param .u64 bolt_add_block_bases_param_1,   // block_bases_ptr (u32*)
///     .param .u32 bolt_add_block_bases_param_2    // n
/// )
/// ```
///
/// Per row: `indices[i] += block_bases[blockIdx.x]`. The launch grid MUST
/// use the same `BLOCK_SIZE` that produced the indices in the parent scan,
/// so `blockIdx.x` equals `i / BLOCK_SIZE` exactly. There is one
/// `block_bases` entry per block of `indices`.
///
/// Out-of-range lanes (the last block when `n` is not a multiple of
/// `BLOCK_SIZE`) early-return without touching memory.
pub fn compile_add_block_bases_kernel() -> BoltResult<String> {
    let mut ptx = String::new();
    writeln!(ptx, "{}", PTX_VERSION).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_TARGET).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_ADDRESS_SIZE).map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {}(", ADD_BASES_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", ADD_BASES_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", ADD_BASES_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_2", ADD_BASES_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<16>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // gid = ctaid.x * ntid.x + tid.x ; n in %r4
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u32 %r4, [{}_param_2];",
        ADD_BASES_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.s32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // Globalize pointers.
    writeln!(
        ptx,
        "\tld.param.u64 %rd0, [{}_param_0];",
        ADD_BASES_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u64 %rd1, [{}_param_1];",
        ADD_BASES_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;

    // base = block_bases[blockIdx.x]
    writeln!(ptx, "\tmul.wide.s32 %rd2, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd3, %rd1, %rd2;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r5, [%rd3];").map_err(write_err)?;

    // v = indices[gid]
    writeln!(ptx, "\tmul.wide.s32 %rd4, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd5, %rd0, %rd4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r6, [%rd5];").map_err(write_err)?;

    // sum = v + base ; indices[gid] = sum
    writeln!(ptx, "\tadd.u32 %r7, %r6, %r5;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd5], %r7;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Adapt a `std::fmt::Error` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("prefix_scan_multipass: write failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_u32_kernel_shape() {
        let ptx = compile_prefix_scan_u32_kernel().expect("u32 scan PTX compiles");

        // Header.
        assert!(ptx.contains(".version 7.5"));
        assert!(ptx.contains(".target sm_70"));
        assert!(ptx.contains(".address_size 64"));

        // Signature.
        assert!(ptx.contains(".visible .entry bolt_prefix_scan_u32("));
        assert!(ptx.contains(".param .u64 bolt_prefix_scan_u32_param_0,"));
        assert!(ptx.contains(".param .u64 bolt_prefix_scan_u32_param_1,"));
        assert!(ptx.contains(".param .u64 bolt_prefix_scan_u32_param_2,"));
        assert!(ptx.contains(".param .u32 bolt_prefix_scan_u32_param_3"));

        // Direct u32 load — no mask-byte machinery.
        assert!(
            ptx.contains("ld.global.u32"),
            "expected direct u32 input load"
        );
        assert!(
            !ptx.contains("ld.global.u8"),
            "u32 scan should not load u8 bytes"
        );
        assert!(
            !ptx.contains("cvt.u32.u16"),
            "u32 scan should not need u8->u32 conversion"
        );
        assert!(
            !ptx.contains("selp.s32 %r5, 1, 0"),
            "u32 scan should not coerce truthiness"
        );

        // Hillis-Steele core shape (same as single-pass scan).
        let n_sync = ptx.matches("bar.sync 0;").count();
        assert!(
            n_sync >= 9,
            "expected >=9 bar.syncs (one seed + 8 rounds), got {n_sync}"
        );
        assert!(ptx.contains("ld.shared.u32"));
        assert!(ptx.contains("st.shared.u32"));
        assert!(
            ptx.contains("sub.s32 %r11, %r10, %r5;"),
            "missing inclusive->exclusive subtract"
        );

        // Block-sum store at thread (BLOCK_SIZE-1).
        assert!(ptx.contains(&format!("setp.ne.s32 %p6, %r2, {};", BLOCK_SIZE - 1)));
        assert!(ptx.contains("st.global.u32 [%rd15], %r10;"));

        assert!(ptx.contains("DONE:"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn add_block_bases_kernel_shape() {
        let ptx = compile_add_block_bases_kernel().expect("add-bases PTX compiles");

        // Header.
        assert!(ptx.contains(".version 7.5"));
        assert!(ptx.contains(".target sm_70"));
        assert!(ptx.contains(".address_size 64"));

        // Signature.
        assert!(ptx.contains(".visible .entry bolt_add_block_bases("));
        assert!(ptx.contains(".param .u64 bolt_add_block_bases_param_0,"));
        assert!(ptx.contains(".param .u64 bolt_add_block_bases_param_1,"));
        assert!(ptx.contains(".param .u32 bolt_add_block_bases_param_2"));

        // Core ops: load base for the block, load the index, add, store back.
        assert!(
            ptx.contains("ld.global.u32"),
            "expected u32 loads (indices + bases)"
        );
        assert!(
            ptx.contains("add.u32 %r7, %r6, %r5;"),
            "expected u32 sum of index + base"
        );
        assert!(
            ptx.contains("st.global.u32 [%rd5], %r7;"),
            "expected u32 store back to indices[gid]"
        );

        // Out-of-range guard.
        assert!(ptx.contains("setp.ge.s32 %p0, %r3, %r4;"));
        assert!(ptx.contains("@%p0 bra DONE;"));

        assert!(ptx.contains("DONE:"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn reexport_uses_single_pass_entry_name() {
        // Re-export should be the same constant the single-pass module exposes.
        assert_eq!(SCAN_KERNEL_ENTRY, "bolt_prefix_scan");

        // And it should compile — we re-export the *function*, not just the name.
        let ptx = compile_prefix_scan_kernel().expect("re-exported scan PTX compiles");
        assert!(ptx.contains(".visible .entry bolt_prefix_scan("));
    }
}
