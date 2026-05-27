// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for the GPU INNER-JOIN build + probe kernels.
//!
//! Backs `crate::exec::gpu_join`: the Stage 1 GPU fast path for
//! single-key, Int32/Int64 INNER joins. Two kernels cooperate via an open-
//! addressing, linear-probed hash table held entirely in device memory.
//!
//! ## Hash table layout (SoA)
//!
//! Two parallel device buffers of length `cap` (power of two):
//!
//! * `keys_table: i64[cap]`     — initialised to `i64::MIN` (empty sentinel).
//! * `row_idx_table: u32[cap]`  — initialised to `u32::MAX` (defensive — the
//!   keys slot is the source of truth for occupancy, but a sentinel row index
//!   makes any host-side debug print of the table obvious).
//!
//! `cap = next_power_of_two(2 * build_n_rows)` (≈ 50% peak load factor). The
//! executor caps the table at 64 MiB total (Int64 keys + u32 indices = 12
//! bytes/slot ⇒ ~5.6M slots, ~2.8M build rows).
//!
//! ## Build kernel — `bolt_hash_join_build`
//!
//! One thread per build row:
//!
//! 1. Loads `key = keys_col[tid]`.
//! 2. Computes initial `slot = hash64(key) & (cap - 1)`.
//! 3. Performs a bounded linear probe (`MAX_PROBE_FACTOR * cap` slots):
//!    * `atom.global.cas.b64 prev, [keys_table[slot]], EMPTY_KEY, key`.
//!    * `prev == EMPTY_KEY` ⇒ insertion succeeded; write `row_idx_table[slot] = tid`
//!      and return.
//!    * `prev == key` ⇒ another thread already inserted this key. We keep the
//!      first-writer-wins behaviour by leaving `row_idx_table[slot]` alone.
//!      However the host treats build-key collisions as a *fall-through* miss
//!      (see Stage-2 note below) so duplicate build keys disable the GPU fast
//!      path entirely; if execution reaches this kernel the host has already
//!      asserted uniqueness.
//!    * Otherwise advance `slot = (slot + 1) & (cap - 1)` and retry.
//! 4. If the bounded probe exhausts without inserting, the thread silently
//!    bails. The host enforces load factor < 0.5 so this never triggers in
//!    practice; the bound is purely a runaway-kernel safety net.
//!
//! ## Probe kernel — `bolt_hash_join_probe`
//!
//! One thread per probe row:
//!
//! 1. Loads `key = probe_keys[tid]`.
//! 2. Hashes + probes through `keys_table` (same hash + mask as build):
//!    * Slot is empty (`keys_table[slot] == EMPTY_KEY`) ⇒ no match; return.
//!    * Slot key matches ⇒ atomically claim an output index via
//!      `atom.global.add.u32 idx, [out_counter], 1`, then write
//!      `(out_probe_idx[idx], out_build_idx[idx]) = (tid, row_idx_table[slot])`.
//!      Bounded by the host-provided `out_capacity`; on overflow the kernel
//!      bails silently and the host re-launches with a larger output buffer
//!      (the size estimate from `build_n_rows * probe_n_rows` is conservative
//!      for the unique-key case).
//!    * Slot key mismatches ⇒ advance and retry, still bounded.
//!
//! The output ordering is *non-deterministic* (atomic counter races) — the
//! host either sorts on the probe index post-hoc or, in the INNER case,
//! accepts arbitrary row ordering because the join doesn't promise one.
//!
//! ## Hash function
//!
//! Splitmix-style 64-bit multiply, identical to
//! `crate::jit::hash_kernels::FX_MUL`. Host-side replay against the same key
//! type produces identical slot assignments, which keeps the round-trip
//! `#[ignore]`'d test deterministic.
//!
//! ## Stage-2 follow-ups
//!
//! * **Multi-key joins.** Stage 1 hashes a single 64-bit key. Multi-key joins
//!   need either a composite key packing (similar to `groupby::pack_keys`) or
//!   a kernel that hashes a vector of inputs per row.
//! * **OUTER joins** (LEFT / RIGHT / FULL) stay host-side. The natural
//!   extension is a "build-matched" bitmap parallel to `row_idx_table` and a
//!   post-pass that emits unmatched build rows.
//! * **Build-side duplicates.** Stage 1 errors out to the host fast path when
//!   the build side contains duplicate keys (only the *first* row index would
//!   be retrievable from `row_idx_table[slot]`, dropping the rest). The
//!   right fix is a per-slot collision list (e.g. a separate "spill" buffer
//!   keyed by slot, gathered by the probe via a second indirection).
//! * **Larger hash tables.** The 64 MiB cap holds for sm_70-class boards;
//!   newer cards have plenty of headroom and the cap can lift after measuring.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::DataType;

/// PTX target metadata baked into every emitted module. Matches the rest of
/// the JIT pipeline (see `hash_kernels.rs`, `sort_kernel.rs`).
const PTX_VERSION: &str = ".version 7.5";
/// Target SM architecture string.
const PTX_TARGET: &str = ".target sm_70";
/// Address-size directive (we always use 64-bit pointers).
const PTX_ADDRESS_SIZE: &str = ".address_size 64";

/// Threads per block for both the build and probe kernels. Matches `BLOCK_SIZE`
/// elsewhere so occupancy tuning stays uniform across the engine's kernels.
pub const HASH_JOIN_BLOCK_SIZE: u32 = 256;

/// PTX `i64::MIN` literal used as the "empty slot" sentinel in `keys_table`.
/// Identical to the GROUP BY hash kernel's `EMPTY_KEY_LITERAL` so the same
/// host-side initialization helper can serve either consumer.
pub const EMPTY_KEY_LITERAL: &str = "-9223372036854775808";

/// Splitmix multiplier — identical to `crate::jit::hash_kernels::FX_MUL`. Lifted
/// here as a copy so this module compiles standalone (matches the same
/// duplication in `valid_flag_kernels`).
// NOTE: this value must match hash_kernels::FX_MUL.
const FX_MUL: i64 = 0x9E3779B97F4A7C15u64 as i64;

/// Upper bound on the linear-probe loop, expressed as a multiple of `cap`.
/// At load factor < 0.5 (enforced by the host) the expected probe length is
/// well under `log2(cap)`, so a full table sweep is generous. The bound exists
/// purely as a runaway-kernel safety net — if the host's load-factor invariant
/// is honoured, it never triggers.
const MAX_PROBE_FACTOR: u32 = 2;

/// Entry-point name of the build kernel.
pub const BUILD_KERNEL_ENTRY: &str = "bolt_hash_join_build";

/// Entry-point name of the probe kernel.
pub const PROBE_KERNEL_ENTRY: &str = "bolt_hash_join_probe";

/// Validate that `dtype` is a supported single-key dtype for the GPU hash join.
///
/// Stage 1 only handles `Int32` / `Int64` keys. Int32 keys are sign-extended to
/// i64 on the host before upload, so the kernel itself only knows about i64.
pub fn is_supported_key_dtype(dtype: DataType) -> bool {
    matches!(dtype, DataType::Int32 | DataType::Int64)
}

/// Compile the build kernel's PTX.
///
/// The emitted module exports exactly one entry point — [`BUILD_KERNEL_ENTRY`]
/// — with the following ABI:
///
/// ```text
/// .visible .entry bolt_hash_join_build(
///     .param .u64 keys_col_ptr,      // i64, length n_rows (encoded keys)
///     .param .u64 keys_table_ptr,    // i64, length cap, init=i64::MIN
///     .param .u64 row_idx_table_ptr, // u32, length cap
///     .param .u32 n_rows,
///     .param .u32 cap                // power-of-two
/// )
/// ```
///
/// Grid: 1D, one thread per row, block size [`HASH_JOIN_BLOCK_SIZE`].
pub fn compile_build_kernel() -> BoltResult<String> {
    let mut p = String::new();
    writeln!(p, "{PTX_VERSION}").map_err(write_err)?;
    writeln!(p, "{PTX_TARGET}").map_err(write_err)?;
    writeln!(p, "{PTX_ADDRESS_SIZE}").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    let entry = BUILD_KERNEL_ENTRY;
    writeln!(p, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_3,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_4").map_err(write_err)?;
    writeln!(p, ")").map_err(write_err)?;
    writeln!(p, "{{").map_err(write_err)?;

    // Register file. Generous declarations — only names, not real allocations.
    //
    //   pred: %p0 oob, %p1 inserted, %p2 collision (same-key), %p3 probe-overflow
    //   b32 : %r0..%r3 tid math, %r4 n_rows, %r5 cap, %r6 mask=cap-1,
    //         %r7 hash u32, %r8 slot, %r20 max_probes, %r21 probe_count
    //   b64 : %rd0..%rd7 device-pointer scratch
    //   key : %rl0 key, %rl1 FX_MUL, %rl2 key*FX_MUL, %rl3 (>>32), %rl4 EMPTY_KEY,
    //         %rl5 atom.cas result
    writeln!(p, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32   %r<24>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64   %rd<16>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    // tid = ctaid.x * ntid.x + tid.x ; bail if tid >= n_rows.
    writeln!(p, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(p, "\tld.param.u32 %r4, [{entry}_param_3];").map_err(write_err)?;
    writeln!(p, "\tsetp.ge.s32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(p, "\t@%p0 bra DONE;").map_err(write_err)?;

    // cap and mask = cap - 1.
    writeln!(p, "\tld.param.u32 %r5, [{entry}_param_4];").map_err(write_err)?;
    writeln!(p, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;

    // max_probes = cap * MAX_PROBE_FACTOR.
    writeln!(p, "\tmul.lo.u32 %r20, %r5, {MAX_PROBE_FACTOR};").map_err(write_err)?;

    // Load key for this row.
    writeln!(p, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.s32 %rd1, %r3, 8;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(p, "\tld.global.s64 %rl0, [%rd2];").map_err(write_err)?;

    // hash = (key * FX_MUL) >> 32 ; slot = hash & mask.
    writeln!(p, "\tmov.s64 %rl1, {FX_MUL};").map_err(write_err)?;
    writeln!(p, "\tmul.lo.s64 %rl2, %rl0, %rl1;").map_err(write_err)?;
    writeln!(p, "\tshr.u64 %rl3, %rl2, 32;").map_err(write_err)?;
    writeln!(p, "\tcvt.u32.u64 %r7, %rl3;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r8, %r7, %r6;").map_err(write_err)?;

    // keys_table base pointer.
    writeln!(p, "\tld.param.u64 %rd3, [{entry}_param_1];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    // row_idx_table base pointer.
    writeln!(p, "\tld.param.u64 %rd4, [{entry}_param_2];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;

    // EMPTY_KEY sentinel.
    writeln!(p, "\tmov.s64 %rl4, {EMPTY_KEY_LITERAL};").map_err(write_err)?;

    // probe counter.
    writeln!(p, "\tmov.u32 %r21, 0;").map_err(write_err)?;

    // Probe loop.
    writeln!(p, "PROBE_LOOP:").map_err(write_err)?;
    writeln!(p, "\tadd.u32 %r21, %r21, 1;").map_err(write_err)?;
    writeln!(p, "\tsetp.gt.u32 %p3, %r21, %r20;").map_err(write_err)?;
    writeln!(p, "\t@%p3 bra DONE;").map_err(write_err)?;

    // addr_keys = keys_table + slot * 8
    writeln!(p, "\tmul.wide.u32 %rd5, %r8, 8;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd6, %rd3, %rd5;").map_err(write_err)?;

    // atom.cas: EMPTY -> key. Returns the previous value.
    writeln!(p, "\tatom.global.cas.b64 %rl5, [%rd6], %rl4, %rl0;").map_err(write_err)?;

    // If prev == EMPTY -> we inserted; record row index and done.
    writeln!(p, "\tsetp.eq.s64 %p1, %rl5, %rl4;").map_err(write_err)?;
    writeln!(p, "\t@%p1 bra INSERTED;").map_err(write_err)?;

    // If prev == key -> someone else has this exact key already. The host
    // forbids duplicate build keys at this point in Stage 1, so this branch
    // is unreachable in practice; we still test for it so a future
    // duplicate-aware path can clear out cleanly. Either way: don't advance.
    writeln!(p, "\tsetp.eq.s64 %p2, %rl5, %rl0;").map_err(write_err)?;
    writeln!(p, "\t@%p2 bra DONE;").map_err(write_err)?;

    // Collision with a different key: advance slot (linear probe, masked).
    writeln!(p, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(p, "\tbra PROBE_LOOP;").map_err(write_err)?;

    // INSERTED: write the row index. The slot is now exclusively ours
    // (the cas above wrote a non-EMPTY value); no concurrent writer can win
    // a later cas for the same slot.
    writeln!(p, "INSERTED:").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd5, %r8, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd7, %rd4, %rd5;").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd7], %r3;").map_err(write_err)?;

    writeln!(p, "DONE:").map_err(write_err)?;
    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;

    Ok(p)
}

/// Compile the probe kernel's PTX.
///
/// The emitted module exports exactly one entry point — [`PROBE_KERNEL_ENTRY`]
/// — with the following ABI:
///
/// ```text
/// .visible .entry bolt_hash_join_probe(
///     .param .u64 probe_keys_ptr,    // i64, length n_probe (encoded keys)
///     .param .u64 keys_table_ptr,    // i64, length cap (populated)
///     .param .u64 row_idx_table_ptr, // u32, length cap (populated)
///     .param .u64 out_probe_idx_ptr, // u32, length out_capacity
///     .param .u64 out_build_idx_ptr, // u32, length out_capacity
///     .param .u64 out_counter_ptr,   // u32, single counter (init=0)
///     .param .u32 n_probe,
///     .param .u32 cap,               // power-of-two
///     .param .u32 out_capacity       // guard against output buffer overflow
/// )
/// ```
///
/// Grid: 1D, one thread per probe row, block size [`HASH_JOIN_BLOCK_SIZE`].
///
/// On overflow (counter exceeds `out_capacity`) the kernel writes the actual
/// count into the counter — the host detects this and re-launches with a
/// resized output buffer. For Stage 1 the host pre-sizes the output to
/// `build_n_rows + probe_n_rows` which is loose enough to never overflow
/// in the INNER-equi-join-with-unique-build case.
pub fn compile_probe_kernel() -> BoltResult<String> {
    let mut p = String::new();
    writeln!(p, "{PTX_VERSION}").map_err(write_err)?;
    writeln!(p, "{PTX_TARGET}").map_err(write_err)?;
    writeln!(p, "{PTX_ADDRESS_SIZE}").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    let entry = PROBE_KERNEL_ENTRY;
    writeln!(p, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_3,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_4,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_5,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_6,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_7,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_8").map_err(write_err)?;
    writeln!(p, ")").map_err(write_err)?;
    writeln!(p, "{{").map_err(write_err)?;

    writeln!(p, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64   %rd<24>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    // tid = ctaid.x * ntid.x + tid.x ; bail if tid >= n_probe.
    writeln!(p, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(p, "\tld.param.u32 %r4, [{entry}_param_6];").map_err(write_err)?;
    writeln!(p, "\tsetp.ge.s32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(p, "\t@%p0 bra DONE;").map_err(write_err)?;

    // cap and mask = cap - 1.
    writeln!(p, "\tld.param.u32 %r5, [{entry}_param_7];").map_err(write_err)?;
    writeln!(p, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;

    // max_probes = cap * MAX_PROBE_FACTOR.
    writeln!(p, "\tmul.lo.u32 %r20, %r5, {MAX_PROBE_FACTOR};").map_err(write_err)?;

    // out_capacity.
    writeln!(p, "\tld.param.u32 %r22, [{entry}_param_8];").map_err(write_err)?;

    // Load probe key.
    writeln!(p, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.s32 %rd1, %r3, 8;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(p, "\tld.global.s64 %rl0, [%rd2];").map_err(write_err)?;

    // hash + slot.
    writeln!(p, "\tmov.s64 %rl1, {FX_MUL};").map_err(write_err)?;
    writeln!(p, "\tmul.lo.s64 %rl2, %rl0, %rl1;").map_err(write_err)?;
    writeln!(p, "\tshr.u64 %rl3, %rl2, 32;").map_err(write_err)?;
    writeln!(p, "\tcvt.u32.u64 %r7, %rl3;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r8, %r7, %r6;").map_err(write_err)?;

    // keys_table base + row_idx_table base.
    writeln!(p, "\tld.param.u64 %rd3, [{entry}_param_1];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(p, "\tld.param.u64 %rd4, [{entry}_param_2];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;

    // EMPTY_KEY for the no-match test.
    writeln!(p, "\tmov.s64 %rl4, {EMPTY_KEY_LITERAL};").map_err(write_err)?;

    // probe counter.
    writeln!(p, "\tmov.u32 %r21, 0;").map_err(write_err)?;

    // Probe loop — non-mutating walk over keys_table looking for our key.
    writeln!(p, "PROBE_LOOP:").map_err(write_err)?;
    writeln!(p, "\tadd.u32 %r21, %r21, 1;").map_err(write_err)?;
    writeln!(p, "\tsetp.gt.u32 %p3, %r21, %r20;").map_err(write_err)?;
    writeln!(p, "\t@%p3 bra DONE;").map_err(write_err)?;

    // addr_keys = keys_table + slot * 8
    writeln!(p, "\tmul.wide.u32 %rd5, %r8, 8;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd6, %rd3, %rd5;").map_err(write_err)?;
    writeln!(p, "\tld.global.s64 %rl5, [%rd6];").map_err(write_err)?;

    // If slot is empty -> no match, done.
    writeln!(p, "\tsetp.eq.s64 %p1, %rl5, %rl4;").map_err(write_err)?;
    writeln!(p, "\t@%p1 bra DONE;").map_err(write_err)?;

    // If slot matches our key -> match.
    writeln!(p, "\tsetp.eq.s64 %p2, %rl5, %rl0;").map_err(write_err)?;
    writeln!(p, "\t@%p2 bra MATCH;").map_err(write_err)?;

    // Advance slot.
    writeln!(p, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(p, "\tbra PROBE_LOOP;").map_err(write_err)?;

    // MATCH: claim an output index via atomic increment of the global counter,
    // then write (probe_idx, build_idx) at that index — but only if the index
    // is within out_capacity. On overflow, the counter keeps climbing so the
    // host can detect the overflow via cuMemcpyDtoH on the counter; we just
    // don't write past the buffer.
    writeln!(p, "MATCH:").map_err(write_err)?;

    // Load build_idx = row_idx_table[slot].
    writeln!(p, "\tmul.wide.u32 %rd5, %r8, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd7, %rd4, %rd5;").map_err(write_err)?;
    writeln!(p, "\tld.global.u32 %r9, [%rd7];").map_err(write_err)?;

    // atom.add on counter: claim slot.
    writeln!(p, "\tld.param.u64 %rd8, [{entry}_param_5];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd8, %rd8;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r10, 1;").map_err(write_err)?;
    writeln!(p, "\tatom.global.add.u32 %r11, [%rd8], %r10;").map_err(write_err)?;

    // If r11 (claimed slot) >= out_capacity, skip the writes.
    writeln!(p, "\tsetp.ge.u32 %p4, %r11, %r22;").map_err(write_err)?;
    writeln!(p, "\t@%p4 bra DONE;").map_err(write_err)?;

    // out_probe_idx[claimed] = tid (== %r3)
    writeln!(p, "\tld.param.u64 %rd9, [{entry}_param_3];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd9, %rd9;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd10, %r11, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd11, %rd9, %rd10;").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd11], %r3;").map_err(write_err)?;

    // out_build_idx[claimed] = build_idx (== %r9)
    writeln!(p, "\tld.param.u64 %rd12, [{entry}_param_4];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd12, %rd12;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd13, %rd12, %rd10;").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd13], %r9;").map_err(write_err)?;

    writeln!(p, "DONE:").map_err(write_err)?;
    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;

    Ok(p)
}

/// Adapt a `std::fmt::Error` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("hash_join_kernel: write failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Entry-point names are stable — `gpu_join.rs` consumes them by string.
    /// Any rename here would silently fail to resolve the kernel at module
    /// load time.
    #[test]
    fn entry_names_are_stable() {
        assert_eq!(BUILD_KERNEL_ENTRY, "bolt_hash_join_build");
        assert_eq!(PROBE_KERNEL_ENTRY, "bolt_hash_join_probe");
    }

    #[test]
    fn supported_key_dtypes() {
        assert!(is_supported_key_dtype(DataType::Int32));
        assert!(is_supported_key_dtype(DataType::Int64));
        // Stage 1: float/bool/utf8 fall through to the host path.
        assert!(!is_supported_key_dtype(DataType::Float32));
        assert!(!is_supported_key_dtype(DataType::Float64));
        assert!(!is_supported_key_dtype(DataType::Bool));
        assert!(!is_supported_key_dtype(DataType::Utf8));
    }

    /// Header + signature shape — the byte-stable bits of every emitted PTX
    /// module. If anything here changes we want a test failure forcing an
    /// intentional update rather than a silent ABI drift.
    #[test]
    fn build_ptx_header_and_signature_shape() {
        let ptx = compile_build_kernel().unwrap();

        // Header.
        assert!(ptx.contains(".version 7.5"));
        assert!(ptx.contains(".target sm_70"));
        assert!(ptx.contains(".address_size 64"));

        // Entry point.
        assert!(ptx.contains(".visible .entry bolt_hash_join_build("));

        // Param list — five params, three .u64 (pointers) then two .u32.
        assert!(ptx.contains(".param .u64 bolt_hash_join_build_param_0,"));
        assert!(ptx.contains(".param .u64 bolt_hash_join_build_param_1,"));
        assert!(ptx.contains(".param .u64 bolt_hash_join_build_param_2,"));
        assert!(ptx.contains(".param .u32 bolt_hash_join_build_param_3,"));
        assert!(ptx.contains(".param .u32 bolt_hash_join_build_param_4"));
    }

    /// Build kernel MUST emit an `atom.global.cas.b64` against the keys
    /// table — this is the load-bearing instruction for slot insertion. If
    /// it disappears (e.g. someone "optimises" to a plain st.global.s64)
    /// concurrent build threads will race and corrupt the table.
    #[test]
    fn build_ptx_uses_atom_cas_for_insertion() {
        let ptx = compile_build_kernel().unwrap();
        assert!(
            ptx.contains("atom.global.cas.b64"),
            "build kernel must use atom.global.cas.b64 for slot insertion; got:\n{ptx}"
        );
    }

    /// Build kernel must consult the row count to guard OOB threads.
    #[test]
    fn build_ptx_has_oob_guard() {
        let ptx = compile_build_kernel().unwrap();
        assert!(ptx.contains("setp.ge.s32"), "missing OOB compare against n_rows");
        assert!(ptx.contains("bra DONE"), "missing branch to DONE label");
        assert!(ptx.contains("DONE:"), "missing DONE label");
    }

    /// Build kernel must use `cap - 1` as a mask for power-of-two table
    /// indexing. Anything else (e.g. mod) is both slower and a regression
    /// vs the load-factor invariant.
    #[test]
    fn build_ptx_uses_mask_for_slot_indexing() {
        let ptx = compile_build_kernel().unwrap();
        assert!(
            ptx.contains("sub.s32 %r6, %r5, 1;"),
            "build kernel must compute mask = cap - 1; got:\n{ptx}"
        );
        assert!(
            ptx.contains("and.b32 %r8, %r8, %r6;"),
            "build kernel must mask slot with (cap-1) on advance"
        );
    }

    /// Build kernel must use the splitmix multiplier matching `hash_kernels`.
    /// The probe and build hashes MUST agree byte-for-byte; any divergence
    /// makes the probe miss inserted keys.
    #[test]
    fn build_ptx_uses_splitmix_multiplier() {
        let ptx = compile_build_kernel().unwrap();
        // FX_MUL displayed as a decimal i64 literal.
        let expected = format!("mov.s64 %rl1, {FX_MUL};");
        assert!(
            ptx.contains(&expected),
            "build kernel must materialise FX_MUL ({FX_MUL}); got:\n{ptx}"
        );
    }

    /// Build kernel emits the EMPTY_KEY sentinel `i64::MIN`.
    #[test]
    fn build_ptx_uses_i64_min_sentinel() {
        let ptx = compile_build_kernel().unwrap();
        assert!(
            ptx.contains(&format!("mov.s64 %rl4, {EMPTY_KEY_LITERAL};")),
            "build kernel must use i64::MIN as EMPTY_KEY sentinel; got:\n{ptx}"
        );
    }

    /// Probe kernel signature has nine parameters: six .u64 (three table
    /// pointers + two output pointers + counter) and three .u32 (n_probe,
    /// cap, out_capacity).
    #[test]
    fn probe_ptx_header_and_signature_shape() {
        let ptx = compile_probe_kernel().unwrap();

        assert!(ptx.contains(".version 7.5"));
        assert!(ptx.contains(".target sm_70"));
        assert!(ptx.contains(".address_size 64"));

        assert!(ptx.contains(".visible .entry bolt_hash_join_probe("));

        assert!(ptx.contains(".param .u64 bolt_hash_join_probe_param_0,"));
        assert!(ptx.contains(".param .u64 bolt_hash_join_probe_param_5,"));
        assert!(ptx.contains(".param .u32 bolt_hash_join_probe_param_6,"));
        assert!(ptx.contains(".param .u32 bolt_hash_join_probe_param_7,"));
        assert!(ptx.contains(".param .u32 bolt_hash_join_probe_param_8"));
    }

    /// Probe kernel MUST contain the linear-probe lookup loop, identifiable by
    /// (a) the s64 equality test against EMPTY_KEY (no match) and (b) the
    /// s64 equality test against the loaded key (match). Both are required —
    /// dropping either breaks correctness in opposing ways.
    #[test]
    fn probe_ptx_has_lookup_loop_structure() {
        let ptx = compile_probe_kernel().unwrap();
        // setp.eq.s64 appears twice: once for empty-slot check, once for
        // key-match check.
        let n_eq = ptx.matches("setp.eq.s64").count();
        assert!(
            n_eq >= 2,
            "probe kernel must emit at least two s64 equality tests \
             (EMPTY_KEY + key match); saw {n_eq}\n{ptx}"
        );
        assert!(ptx.contains("PROBE_LOOP:"));
        assert!(ptx.contains("MATCH:"));
    }

    /// Probe kernel must atomically increment the global output counter so
    /// concurrent matching threads claim disjoint output slots.
    #[test]
    fn probe_ptx_uses_atom_add_for_output_counter() {
        let ptx = compile_probe_kernel().unwrap();
        assert!(
            ptx.contains("atom.global.add.u32"),
            "probe kernel must use atom.global.add.u32 for output counter; got:\n{ptx}"
        );
    }

    /// Probe kernel must guard against output-buffer overflow before storing.
    /// The guard is `setp.ge.u32 ..., claimed, out_capacity` — if the
    /// returned counter value is >= out_capacity, the writes must be
    /// skipped.
    #[test]
    fn probe_ptx_guards_against_output_overflow() {
        let ptx = compile_probe_kernel().unwrap();
        // The kernel must compare the claimed slot against out_capacity. The
        // exact register pairing comes out as `setp.ge.u32 %p4, %r11, %r22;`
        // in the current emit; we test the shape, not the register pairing,
        // so allocator tweaks don't break the test.
        assert!(
            ptx.contains("setp.ge.u32"),
            "probe kernel must guard claimed >= out_capacity; got:\n{ptx}"
        );
    }

    /// Probe kernel writes both output streams (probe_idx + build_idx) as
    /// u32. If either store disappears the host-side gather produces
    /// garbage; if either becomes a 64-bit store it overruns the u32
    /// output buffers.
    #[test]
    fn probe_ptx_writes_both_output_streams_as_u32() {
        let ptx = compile_probe_kernel().unwrap();
        // Two u32 stores on the match path (one per output stream).
        let n_st = ptx.matches("st.global.u32").count();
        assert!(
            n_st >= 2,
            "probe kernel must store probe_idx and build_idx as u32; saw {n_st}\n{ptx}"
        );
    }

    /// Both kernels must agree on hash + mask — the probe replays the build's
    /// slot computation. Same FX_MUL literal, same mask = cap - 1, same
    /// shift-right-32 reduction.
    #[test]
    fn build_and_probe_share_hash_function() {
        let build = compile_build_kernel().unwrap();
        let probe = compile_probe_kernel().unwrap();
        // Both materialise the same multiplier.
        let mul_literal = format!("mov.s64 %rl1, {FX_MUL};");
        assert!(build.contains(&mul_literal));
        assert!(probe.contains(&mul_literal));
        // Both reduce by shr 32.
        assert!(build.contains("shr.u64 %rl3, %rl2, 32;"));
        assert!(probe.contains("shr.u64 %rl3, %rl2, 32;"));
        // Both mask with cap - 1.
        assert!(build.contains("and.b32 %r8, %r7, %r6;"));
        assert!(probe.contains("and.b32 %r8, %r7, %r6;"));
    }
}
