// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for GPU-side filter compaction (prefix scan + gather).
//!
//! Today's compaction path (see `crate::exec::compact`) downloads the u8 mask
//! to the host and uses `arrow::compute::filter` to drop masked-out rows. That
//! requires a per-row d2h copy plus a host-side O(n) scan per column. This
//! module emits the kernels for an entirely-on-device compaction:
//!
//! 1. **Per-block Hillis-Steele exclusive prefix-sum** over the u8 mask. Each
//!    block of `BLOCK_SIZE` threads writes its thread's exclusive prefix into
//!    `local_indices[gid]` and the block's running sum (the inclusive scan at
//!    the last thread) into `block_sums[blockIdx.x]`.
//! 2. (Host stage — see `crate::exec::gpu_compact`) Download the block_sums,
//!    exclusive-scan on the CPU, and re-upload as `block_bases`. The total
//!    count is the sum of the block_sums.
//! 3. **Per-dtype gather kernel**: for each surviving row, write
//!    `output[block_bases[blockIdx.x] + local_indices[gid]] = input[gid]`.
//!
//! The shared-memory ping-pong here is deliberately the simpler Hillis-Steele
//! variant (O(n log n) work; fine at BLOCK_SIZE = 256) rather than Blelloch.
//! Two `sdata` buffers swap each round to avoid a redundant `bar.sync`.
//!
//! ## What's deferred
//!
//! The host-side scan of `block_sums` assumes the array fits comfortably in
//! host memory and runs serially — at `BLOCK_SIZE = 256` we hit 16M rows per
//! 65 535 blocks, which is plenty for one batch. For multi-batch row counts
//! beyond `u32::MAX / 256` the caller should split the input; this module
//! validates the bound up-front in `compile_*` callers (the limit lives in
//! `gpu_compact.rs`).

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::DataType;

/// PTX target metadata baked into every emitted module.
const PTX_VERSION: &str = ".version 7.5";
/// Target SM architecture string.
const PTX_TARGET: &str = ".target sm_70";
/// Address size directive (we always use 64-bit pointers).
const PTX_ADDRESS_SIZE: &str = ".address_size 64";

/// Threads per block used by both the prefix-scan and the gather kernels.
///
/// Chosen so a single Hillis-Steele block scan fits easily in shared memory
/// (`BLOCK_SIZE * 4 bytes` for u32 entries times two ping-pong buffers).
pub const BLOCK_SIZE: u32 = 256;

/// Entry-point name for the per-block prefix-scan kernel.
pub const SCAN_KERNEL_ENTRY: &str = "bolt_prefix_scan";

/// Entry-point name for the per-dtype gather kernel.
///
/// Returns a static string so callers can pass it straight to
/// `CudaModule::function`. Errors are deferred to `compile_gather_kernel` so
/// invalid dtypes surface during PTX generation rather than name lookup.
pub fn gather_kernel_entry(dtype: DataType) -> &'static str {
    match dtype {
        DataType::Bool => "bolt_gather_bool",
        DataType::Int32 => "bolt_gather_i32",
        DataType::Int64 => "bolt_gather_i64",
        DataType::Float32 => "bolt_gather_f32",
        DataType::Float64 => "bolt_gather_f64",
        DataType::Utf8 => "bolt_gather_utf8_unsupported",
    }
}

/// Generate the per-block exclusive prefix-scan PTX module.
///
/// ABI:
/// ```text
/// .visible .entry bolt_prefix_scan(
///     .param .u64 bolt_prefix_scan_param_0,   // mask_ptr      (u8*)
///     .param .u64 bolt_prefix_scan_param_1,   // local_indices (u32*)
///     .param .u64 bolt_prefix_scan_param_2,   // block_sums    (u32*)
///     .param .u32 bolt_prefix_scan_param_3    // n_rows
/// )
/// ```
///
/// `local_indices[i]` holds the exclusive prefix sum within the row's block;
/// `block_sums[blockIdx.x]` holds the inclusive sum of the entire block (the
/// number of "kept" rows in that block).
pub fn compile_prefix_scan_kernel() -> BoltResult<String> {
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

    // Signature: mask, local_indices, block_sums, n_rows.
    writeln!(ptx, ".visible .entry {}(", SCAN_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", SCAN_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", SCAN_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_2,", SCAN_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_3", SCAN_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Register pool. PTX `.reg` decls only allocate names; pick generous sizes.
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b16   %rs<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<32>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // -------- Indices: tid_x = %tid.x ; ctaid = %ctaid.x ; gid = ctaid*ntid+tid
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    // n_rows
    writeln!(
        ptx,
        "\tld.param.u32 %r4, [{}_param_3];",
        SCAN_KERNEL_ENTRY
    )
    .map_err(write_err)?;

    // -------- Globalize parameter pointers.
    writeln!(
        ptx,
        "\tld.param.u64 %rd0, [{}_param_0];",
        SCAN_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u64 %rd1, [{}_param_1];",
        SCAN_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u64 %rd2, [{}_param_2];",
        SCAN_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;

    // -------- Load this thread's mask byte (0 if past the end). Treat any
    // non-zero byte as "keep" by emitting (m != 0) ? 1 : 0 to make the scan
    // robust to predicate kernels that emit truthy values other than 1.
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r5, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra AFTER_LOAD;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd3, %r3, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd4, %rd0, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u8 %rs0, [%rd4];").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u16 %r6, %rs0;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p1, %r6, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.s32 %r5, 1, 0, %p1;").map_err(write_err)?;
    writeln!(ptx, "AFTER_LOAD:").map_err(write_err)?;

    // %r5 is now the 0/1 value for this thread. We also keep it for the
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

    // -------- Hillis-Steele scan: for offset in {1,2,4,...,BLOCK_SIZE/2}:
    //   v = (tid >= offset) ? ping[tid - offset] : 0
    //   pong[tid] = ping[tid] + v
    //   bar.sync
    //   swap ping/pong
    //
    // We unroll the loop. At the start of each round, %rd8 points into the
    // "read" buffer at this thread's slot and %rd9 into the "write" buffer.
    // After the round we swap them by xchg'ing the registers via a temp.
    let mut offset: u32 = 1;
    let mut step: usize = 0;
    while offset < BLOCK_SIZE {
        let off_bytes = offset * elem_bytes;
        // Load own value: %r7 = read[tid]
        writeln!(ptx, "\tld.shared.u32 %r7, [%rd8];").map_err(write_err)?;
        // Conditionally load neighbor at offset behind: %r8 = (tid >= offset)
        // ? read[tid - offset] : 0
        writeln!(
            ptx,
            "\tsetp.lt.s32 %p{p}, %r2, {off};",
            p = 2 + (step % 4),
            off = offset
        )
        .map_err(write_err)?;
        // Default to 0 then conditionally overwrite from shared memory.
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
        // sum = own + neighbor (or own + 0)
        writeln!(ptx, "\tadd.s32 %r9, %r7, %r8;").map_err(write_err)?;
        writeln!(ptx, "\tst.shared.u32 [%rd9], %r9;").map_err(write_err)?;
        writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
        // Swap ping/pong slot pointers: tmp = rd8 ; rd8 = rd9 ; rd9 = tmp
        writeln!(ptx, "\tmov.u64 %rd11, %rd8;").map_err(write_err)?;
        writeln!(ptx, "\tmov.u64 %rd8, %rd9;").map_err(write_err)?;
        writeln!(ptx, "\tmov.u64 %rd9, %rd11;").map_err(write_err)?;
        offset <<= 1;
        step += 1;
    }

    // After the loop, %rd8 points at this thread's slot in the buffer holding
    // the *inclusive* scan. Load it.
    writeln!(ptx, "\tld.shared.u32 %r10, [%rd8];").map_err(write_err)?;
    // Convert inclusive -> exclusive: excl = incl - own_value (own is %r5).
    writeln!(ptx, "\tsub.s32 %r11, %r10, %r5;").map_err(write_err)?;

    // -------- Write exclusive scan to local_indices[gid] (only in-range).
    writeln!(ptx, "\tsetp.ge.u32 %p5, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra AFTER_LOCAL_STORE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd12, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd13, %rd1, %rd12;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd13], %r11;").map_err(write_err)?;
    writeln!(ptx, "AFTER_LOCAL_STORE:").map_err(write_err)?;

    // -------- Thread (BLOCK_SIZE - 1) writes block_sums[blockIdx.x] = incl.
    //
    // For partial blocks (the last block when n_rows isn't a multiple of
    // BLOCK_SIZE) the last thread's inclusive scan still equals the count of
    // 1s in the block, because out-of-range lanes contributed zeros via the
    // AFTER_LOAD guard above.
    writeln!(
        ptx,
        "\tsetp.ne.s32 %p6, %r2, {last};",
        last = BLOCK_SIZE - 1
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p6 bra DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd14, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd15, %rd2, %rd14;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd15], %r10;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Generate the per-dtype gather PTX module.
///
/// ABI (per `<dtype>`):
/// ```text
/// .visible .entry bolt_gather_<dtype>(
///     .param .u64 ..._param_0,   // mask_ptr          (u8*)
///     .param .u64 ..._param_1,   // local_indices_ptr (u32*)
///     .param .u64 ..._param_2,   // block_bases_ptr   (u32*)
///     .param .u64 ..._param_3,   // input_ptr         (T*)
///     .param .u64 ..._param_4,   // output_ptr        (T*)
///     .param .u32 ..._param_5    // n_rows
/// )
/// ```
///
/// Each thread reads its mask byte; on `m != 0` it computes
/// `idx = block_bases[blockIdx.x] + local_indices[gid]` and copies
/// `input[gid]` to `output[idx]`.
pub fn compile_gather_kernel(dtype: DataType) -> BoltResult<String> {
    let entry = gather_kernel_entry(dtype);
    let elem_bytes = dtype.byte_width().ok_or_else(|| {
        BoltError::Other(format!(
            "prefix_scan: gather not supported for variable-width dtype {:?}",
            dtype
        ))
    })?;
    if matches!(dtype, DataType::Utf8) {
        return Err(BoltError::Other(
            "prefix_scan: gather Utf8 not supported (variable-width)".into(),
        ));
    }
    let (ld_suffix, reg_class, reg_ty) = gather_type_info(dtype)?;

    let mut ptx = String::new();
    writeln!(ptx, "{}", PTX_VERSION).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_TARGET).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_ADDRESS_SIZE).map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {}(", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_2,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_3,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_4,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_5", entry).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b16   %rs<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<24>;").map_err(write_err)?;
    // The value register class for Bool / Int32 reuses one of the generic
    // decls above (`rs` and `r` respectively), so we only emit a fresh decl
    // when the class differs (rl for i64, f for f32, fd for f64).
    if reg_class != "rs" && reg_class != "r" && reg_class != "rd" {
        writeln!(ptx, "\t.reg .{ty}   %{rc}<4>;", ty = reg_ty, rc = reg_class)
            .map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    // gid = ctaid.x * ntid.x + tid.x
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{}_param_5];", entry).map_err(write_err)?;
    // Bail out-of-range.
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // Globalize the five pointer params.
    writeln!(ptx, "\tld.param.u64 %rd0, [{}_param_0];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd1, [{}_param_1];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd2, [{}_param_2];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd3, [{}_param_3];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd4, [{}_param_4];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;

    // Load mask byte; bail if zero.
    writeln!(ptx, "\tmul.wide.s32 %rd5, %r3, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd6, %rd0, %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u8 %rs0, [%rd6];").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u16 %r5, %rs0;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p1, %r5, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra DONE;").map_err(write_err)?;

    // local_idx = local_indices[gid]
    writeln!(ptx, "\tmul.wide.s32 %rd7, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd8, %rd1, %rd7;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r6, [%rd8];").map_err(write_err)?;

    // block_base = block_bases[blockIdx.x]
    writeln!(ptx, "\tmul.wide.s32 %rd9, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd10, %rd2, %rd9;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r7, [%rd10];").map_err(write_err)?;

    // out_idx = block_base + local_idx
    writeln!(ptx, "\tadd.s32 %r8, %r6, %r7;").map_err(write_err)?;

    // input_addr  = input  + gid * elem_bytes
    writeln!(
        ptx,
        "\tmul.wide.s32 %rd11, %r3, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd3, %rd11;").map_err(write_err)?;

    // output_addr = output + out_idx * elem_bytes
    writeln!(
        ptx,
        "\tmul.wide.s32 %rd13, %r8, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd14, %rd4, %rd13;").map_err(write_err)?;

    // value = *input_addr ; *output_addr = value
    writeln!(
        ptx,
        "\tld.global.{ld} %{rc}0, [%rd12];",
        ld = ld_suffix,
        rc = reg_class
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tst.global.{ld} [%rd14], %{rc}0;",
        ld = ld_suffix,
        rc = reg_class
    )
    .map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// `(ld/st suffix, register class, .reg type)` triple for gather.
///
/// The suffix is used for both loads and stores of the value (e.g. `s32` for
/// Int32). The register class matches `ptx_gen.rs`'s conventions so the same
/// `.reg` decls work.
fn gather_type_info(dtype: DataType) -> BoltResult<(&'static str, &'static str, &'static str)> {
    Ok(match dtype {
        DataType::Bool => ("u8", "rs", "b16"),
        DataType::Int32 => ("s32", "r", "b32"),
        DataType::Int64 => ("s64", "rl", "b64"),
        DataType::Float32 => ("f32", "f", "f32"),
        DataType::Float64 => ("f64", "fd", "f64"),
        DataType::Utf8 => {
            return Err(BoltError::Other(
                "prefix_scan: gather Utf8 not supported (variable-width)".into(),
            ))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_ptx_has_shape() {
        let ptx = compile_prefix_scan_kernel().expect("scan PTX compiles");

        // Header.
        assert!(ptx.contains(".version 7.5"));
        assert!(ptx.contains(".target sm_70"));
        assert!(ptx.contains(".address_size 64"));

        // Shared-memory scratchpad (two BLOCK_SIZE * 4 byte buffers).
        let total = (BLOCK_SIZE * 4 * 2) as u64;
        assert!(
            ptx.contains(&format!(".shared .align 4 .b8 sdata[{total}]")),
            "shared decl missing or wrong size; got:\n{ptx}"
        );

        // Signature.
        assert!(ptx.contains(".visible .entry bolt_prefix_scan("));
        assert!(ptx.contains(".param .u64 bolt_prefix_scan_param_0,"));
        assert!(ptx.contains(".param .u64 bolt_prefix_scan_param_1,"));
        assert!(ptx.contains(".param .u64 bolt_prefix_scan_param_2,"));
        assert!(ptx.contains(".param .u32 bolt_prefix_scan_param_3"));

        // Hillis-Steele: one bar.sync per round + the seed sync. BLOCK_SIZE=256
        // -> log2(256)=8 rounds, plus one for the initial load.
        let n_sync = ptx.matches("bar.sync 0;").count();
        assert!(
            n_sync >= 9,
            "expected >=9 bar.syncs (one seed + 8 rounds), got {n_sync}"
        );

        // Shared-memory ops + the final inclusive->exclusive subtract.
        assert!(ptx.contains("ld.shared.u32"), "missing ld.shared.u32");
        assert!(ptx.contains("st.shared.u32"), "missing st.shared.u32");
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
    fn gather_i32_ptx_has_shape() {
        let ptx = compile_gather_kernel(DataType::Int32).expect("i32 gather PTX compiles");

        assert!(ptx.contains(".visible .entry bolt_gather_i32("));
        // Six params: mask, local_indices, block_bases, input, output, n_rows.
        for i in 0..=4 {
            assert!(
                ptx.contains(&format!(".param .u64 bolt_gather_i32_param_{i},")),
                "missing u64 param {i}"
            );
        }
        assert!(ptx.contains(".param .u32 bolt_gather_i32_param_5"));

        // i32 load/store uses .s32.
        assert!(ptx.contains("ld.global.s32"), "missing typed input load");
        assert!(ptx.contains("st.global.s32"), "missing typed output store");

        // Mask gate.
        assert!(ptx.contains("ld.global.u8 %rs0,"));
        assert!(
            ptx.contains("setp.eq.s32 %p1, %r5, 0;"),
            "missing mask==0 short-circuit"
        );

        // Out-of-range guard at the top.
        assert!(ptx.contains("setp.ge.s32 %p0, %r3, %r4;"));
    }

    #[test]
    fn gather_entry_names_match_dtype() {
        assert_eq!(gather_kernel_entry(DataType::Bool), "bolt_gather_bool");
        assert_eq!(gather_kernel_entry(DataType::Int32), "bolt_gather_i32");
        assert_eq!(gather_kernel_entry(DataType::Int64), "bolt_gather_i64");
        assert_eq!(gather_kernel_entry(DataType::Float32), "bolt_gather_f32");
        assert_eq!(gather_kernel_entry(DataType::Float64), "bolt_gather_f64");

        // Every supported dtype's PTX should use the matching entry name.
        for dtype in [
            DataType::Bool,
            DataType::Int32,
            DataType::Int64,
            DataType::Float32,
            DataType::Float64,
        ] {
            let ptx = compile_gather_kernel(dtype).unwrap_or_else(|e| {
                panic!("compile_gather_kernel({:?}) failed: {}", dtype, e)
            });
            let entry = gather_kernel_entry(dtype);
            assert!(
                ptx.contains(&format!(".visible .entry {entry}(")),
                "{dtype:?} PTX missing entry {entry}"
            );
        }
    }

    #[test]
    fn gather_rejects_utf8() {
        let err = compile_gather_kernel(DataType::Utf8)
            .expect_err("Utf8 gather must error: variable-width");
        assert!(format!("{}", err).contains("Utf8"));
    }
}

/// Adapt a `std::fmt::Error` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("prefix_scan: write failed: {}", e))
}
