// SPDX-License-Identifier: Apache-2.0

//! Sentinel-free open-addressing GROUP BY kernels.
//!
//! The kernels in [`crate::jit::hash_kernels`] use `i64::MIN` as an
//! "empty slot" sentinel inside the keys table. That works for arbitrary
//! integer or packed-tuple keys IF the host first validates that no input row
//! happens to encode to the sentinel; when an input does collide (notably a
//! `Float64` column containing `-0.0`, whose bit pattern equals `i64::MIN`)
//! the executor must reject the batch.
//!
//! This module replaces that scheme with a parallel "valid-flag" table:
//!
//! ```text
//! slot_valid: u32[k]   // host-initialised to 0
//! keys_table: i64[k]   // host-initialised contents are irrelevant
//! ```
//!
//! Each slot has a three-state lifecycle stored in `slot_valid[slot]`:
//!
//! | value | meaning                                                            |
//! |-------|--------------------------------------------------------------------|
//! | `0`   | empty — no thread has claimed this slot yet                        |
//! | `1`   | claimed — a winner thread has CAS'd 0→1 and is about to write key  |
//! | `2`   | committed — the winner has written the key AND issued a memory     |
//! |       | barrier, so other threads may now read `keys_table[slot]` safely   |
//!
//! The keys kernel uses `atom.global.cas.b32` on `slot_valid[slot]` rather
//! than on the i64 key itself. PTX `atom.cas` is only available for the
//! `b32` / `b64` widths on sm_70 — a u8 per slot wouldn't work, so the cost
//! of removing the sentinel is 4 bytes per slot of extra device memory for
//! the `slot_valid` table.
//!
//! ## Probe + commit protocol (keys kernel)
//!
//! For each input row, one thread does:
//!
//! ```text
//! PROBE_TOP:
//!     atom.cas.b32 [slot_valid+slot*4], 0, 1   -> old
//!     if old == 0:                              // we won the slot
//!         st.global.s64 [keys_table+slot*8] = key
//!         membar.gl
//!         atom.global.exch.b32 [slot_valid+slot*4], 2   // publish
//!         goto DONE
//!     else:                                     // someone else owns this slot
//!         SPIN: ld.global.u32 v = [slot_valid+slot*4]
//!               if v != 2: goto SPIN            // wait for commit
//!         ld.global.s64 existing = [keys_table+slot*8]
//!         if existing == key: goto DONE         // same group
//!         slot = (slot + 1) & (k - 1)           // collision; advance
//!         goto PROBE_TOP
//! ```
//!
//! The agg kernel uses the same probe shape but only reads (never writes)
//! `slot_valid` and `keys_table`, then runs the atomic accumulator update on
//! the matching slot.
//!
//! ## Race-condition note
//!
//! Between a winner's CAS-success and its `st.global.s64`, a loser thread
//! that probes the same slot sees `slot_valid == 1`. The SPIN-on-2 loop is
//! what makes this correct: losers wait for the published `2` (which only
//! appears AFTER the winner's `membar.gl` retires its store), so when they
//! load the key they always see the committed value.
//!
//! ## Bounded probe + spill (deadlock hardening)
//!
//! Naked spin loops on the GPU can deadlock if every thread in a warp is
//! waiting on the same un-published slot — both the winner and a loser
//! living in the SAME warp AND a scheduler that favours the loser
//! indefinitely is sufficient. With load factor below 0.5 (which the
//! executor enforces via `K >= 2 * unique_keys + 16`) the probability is
//! negligible, but we belt-and-brace it with two bounded counters:
//!
//! * The outer PROBE loop bails after `MAX_PROBE_FACTOR * k` steps (i.e.
//!   a full table traversal — anything that takes longer is genuinely
//!   stuck). One full sweep is enough because, at load factor < 0.5, the
//!   expected probe length is well under `log2(k)`.
//! * The inner SPIN loop bails after [`SPIN_LIMIT`] iterations. At GHz
//!   clock rates the actual writer-to-loser publish window is on the order
//!   of nanoseconds; `1024` is generous by orders of magnitude.
//!
//! On either bound, the thread takes a "spill" path: it atomically
//! increments a host-allocated `spill_counter` and, if the returned slot is
//! within `max_spill`, writes its `(key)` (keys kernel) or `(key, value)`
//! (agg kernel) into the host-allocated spill buffer. The host
//! post-processes the spill after kernel sync, folding spilled rows into
//! the final GROUP BY result. Spill overflow (`final_counter > max_spill`)
//! is treated as a defensive error on the host side.
//!
//! We deliberately keep the spin in the loop (rather than treating
//! `slot_valid == 1` as "advance to next slot"): if losers advanced past a
//! claimed-but-not-yet-published slot they could write the SAME key into a
//! DIFFERENT slot, splitting the group across multiple buckets. Bounded
//! spin + spill preserves correctness without risking deadlock.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
use crate::jit::agg_kernels::ReduceOp;
use crate::plan::logical_plan::DataType;

/// Splitmix-style multiplier used by the per-row hash. Identical to the
/// constant in [`crate::jit::hash_kernels`] so the executor can replay the
/// hash host-side regardless of which kernel variant ran.
pub const FX_MUL: i64 = 0x9E3779B97F4A7C15u64 as i64;

/// Entry-point name of the keys kernel produced by
/// [`compile_keys_valid_kernel`].
pub const VALID_KEYS_KERNEL_ENTRY: &str = "bolt_groupby_keys_valid";

/// Entry-point name of the aggregate kernel produced by
/// [`compile_agg_valid_kernel`].
pub const VALID_AGG_KERNEL_ENTRY: &str = "bolt_groupby_agg_valid";

/// Threads per block. Matched to the classic variant for parity.
const BLOCK_SIZE: u32 = 256;

/// Multiplier on `k` for the bounded PROBE loop. A full table traversal
/// (`2 * k` steps) gives more than enough headroom at load factor < 0.5,
/// where the expected probe length is well under `log2(k)`.
const MAX_PROBE_FACTOR: u32 = 2;

/// Bound on the inner SPIN-on-`slot_valid == 2` loop. The actual
/// writer-to-loser publish window is nanoseconds at GHz clock rates;
/// `1024` iterations is generous by orders of magnitude and is reached
/// only when the warp scheduler genuinely refuses to make progress on
/// the writer.
const SPIN_LIMIT: u32 = 1024;

/// Block-size accessor for the host-side launcher.
pub fn valid_block_size() -> u32 {
    BLOCK_SIZE
}

/// Generate PTX for the sentinel-free keys-building kernel.
///
/// ## ABI
///
/// ```text
/// .visible .entry bolt_groupby_keys_valid(
///     .param .u64 group_col_ptr,      // i64 keys, length n_rows
///     .param .u64 keys_table_ptr,     // i64, length k, contents arbitrary at entry
///     .param .u64 slot_valid_ptr,     // u32, length k, host-initialised to 0
///     .param .u32 n_rows,
///     .param .u32 k,                  // power-of-two table size
///     .param .u64 spill_keys_ptr,     // i64, length max_spill, host-init irrelevant
///     .param .u64 spill_counter_ptr,  // u32, length 1, host-init to 0
///     .param .u32 max_spill           // capacity of spill_keys
/// )
/// ```
///
/// The kernel writes to BOTH `keys_table_ptr` (the winner of each slot) AND
/// `slot_valid_ptr` (state machine described in the module docs). It does
/// not touch any accumulator tables.
///
/// On bounded-probe / bounded-spin overflow the thread atomically
/// increments `spill_counter_ptr` and (if `< max_spill`) writes its key
/// into `spill_keys_ptr`. The host folds the spill buffer into the result
/// after kernel sync.
pub fn compile_keys_valid_kernel() -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = VALID_KEYS_KERNEL_ENTRY;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_3,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_4,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_5,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_6,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_7").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Generous register decls. Match the conventions used in hash_kernels.
    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<48>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<48>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // tid = ctaid.x * ntid.x + tid.x; bail if tid >= n_rows.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{entry}_param_3];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // k and mask = k-1.
    writeln!(ptx, "\tld.param.u32 %r5, [{entry}_param_4];").map_err(write_err)?;
    writeln!(ptx, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;

    // max_probes = k * MAX_PROBE_FACTOR. Computed once at kernel entry so
    // the bounded PROBE loop can compare against it cheaply.
    writeln!(
        ptx,
        "\tmul.lo.u32 %r20, %r5, {factor};",
        factor = MAX_PROBE_FACTOR
    )
    .map_err(write_err)?;

    // Load this thread's key.
    writeln!(ptx, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd1, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rl0, [%rd2];").map_err(write_err)?;

    // Hash: h = (key * FX_MUL) >> 32 ; then & (k-1).
    writeln!(ptx, "\tmov.s64 %rl1, {FX_MUL};").map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s64 %rl2, %rl0, %rl1;").map_err(write_err)?;
    writeln!(ptx, "\tshr.u64 %rl3, %rl2, 32;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u64 %r7, %rl3;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r7, %r6;").map_err(write_err)?;

    // Load keys_table and slot_valid base pointers.
    writeln!(ptx, "\tld.param.u64 %rd3, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd4, [{entry}_param_2];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;

    // Probe-step counter (bounded probe — bails into SPILL on overflow).
    // %r21 holds the running probe count; %r22 is the spin counter (set
    // per slot inside SPIN). Both share the %r register file declared above.
    writeln!(ptx, "\tmov.u32 %r21, 0;").map_err(write_err)?;

    // PROBE_TOP: compute slot addresses, try CAS on slot_valid.
    writeln!(ptx, "PROBE_TOP:").map_err(write_err)?;
    // Bound check: probe_count += 1 ; if probe_count > max_probes -> SPILL.
    writeln!(ptx, "\tadd.u32 %r21, %r21, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.gt.u32 %p4, %r21, %r20;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra SPILL;").map_err(write_err)?;
    // addr_key   = keys_table  + slot * 8
    writeln!(ptx, "\tmul.wide.u32 %rd5, %r8, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd6, %rd3, %rd5;").map_err(write_err)?;
    // addr_valid = slot_valid + slot * 4
    writeln!(ptx, "\tmul.wide.u32 %rd7, %r8, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd8, %rd4, %rd7;").map_err(write_err)?;
    // atom.cas.b32 — try to flip valid from 0 (empty) to 1 (claimed).
    writeln!(ptx, "\tatom.global.cas.b32 %r9, [%rd8], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p1, %r9, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra CLAIM_SLOT;").map_err(write_err)?;

    // SPIN: another thread owns this slot. Wait for it to publish (valid==2).
    // Bounded by SPIN_LIMIT iterations; on timeout the thread spills rather
    // than risk a warp-scheduler deadlock. %r22 is the per-slot spin counter.
    writeln!(ptx, "\tmov.u32 %r22, 0;").map_err(write_err)?;
    writeln!(ptx, "SPIN:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r22, %r22, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.gt.u32 %p5, %r22, {limit};",
        limit = SPIN_LIMIT
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra SPILL;").map_err(write_err)?;
    // ACQUIRE-LOAD: pairs with the publisher's atomic store; makes the
    // read-of-published-value contract explicit (sm_70+). Replaces the
    // plain `ld.<space>.<ty>` which relied on the publisher's release +
    // SASS-level implicit acquire — sound in practice but not promised
    // by PTX semantics.
    writeln!(ptx, "\tld.acquire.gpu.u32 %r10, [%rd8];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p2, %r10, 2;").map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra SPIN;").map_err(write_err)?;
    // Now safe to read the committed key.
    writeln!(ptx, "\tld.global.s64 %rl5, [%rd6];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s64 %p3, %rl5, %rl0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra DONE;").map_err(write_err)?;
    // Different key — collision — advance probe.
    writeln!(ptx, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    // CLAIM_SLOT: we own the slot. Non-atomic store the key, fence, then
    // atomically publish valid := 2 so other threads can read the key.
    writeln!(ptx, "CLAIM_SLOT:").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s64 [%rd6], %rl0;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.gl;").map_err(write_err)?;
    // exch returns the previous value into a dummy register; we discard it.
    writeln!(ptx, "\tatom.global.exch.b32 %r11, [%rd8], 2;").map_err(write_err)?;
    writeln!(ptx, "\tbra DONE;").map_err(write_err)?;

    // SPILL: bounded probe / spin exceeded — atomically claim an entry in
    // the host-allocated spill buffer and write our key. If the spill
    // buffer is full (`spill_idx >= max_spill`) we silently drop; the host
    // detects this via `final counter > max_spill`.
    writeln!(ptx, "SPILL:").map_err(write_err)?;
    // Load spill_counter pointer + max_spill scalar.
    writeln!(ptx, "\tld.param.u64 %rd20, [{entry}_param_6];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd20, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r23, [{entry}_param_7];").map_err(write_err)?;
    // atom.add returns the previous value, which is our slot index.
    writeln!(ptx, "\tatom.global.add.u32 %r24, [%rd20], 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p6, %r24, %r23;").map_err(write_err)?;
    writeln!(ptx, "\t@%p6 bra DONE;").map_err(write_err)?;
    // Write key into spill_keys[%r24].
    writeln!(ptx, "\tld.param.u64 %rd21, [{entry}_param_5];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd21, %rd21;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd22, %r24, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd21, %rd22;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s64 [%rd23], %rl0;").map_err(write_err)?;
    writeln!(ptx, "\tbra DONE;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Generate PTX for the sentinel-free aggregate-update kernel.
///
/// ## ABI
///
/// ```text
/// .visible .entry bolt_groupby_agg_valid(
///     .param .u64 group_col_ptr,      // i64 keys, length n_rows
///     .param .u64 keys_table_ptr,     // i64, length k, populated by the keys kernel
///     .param .u64 slot_valid_ptr,     // u32, length k, populated by the keys kernel
///     .param .u64 input_col_ptr,      // T,   length n_rows
///     .param .u64 acc_table_ptr,      // T,   length k, host-initialised to identity(op)
///     .param .u32 n_rows,
///     .param .u32 k,
///     .param .u64 spill_keys_ptr,     // i64, length max_spill
///     .param .u64 spill_values_ptr,   // T,   length max_spill (matching input dtype)
///     .param .u64 spill_counter_ptr,  // u32, length 1
///     .param .u32 max_spill
/// )
/// ```
///
/// The kernel only READS `keys_table_ptr` and `slot_valid_ptr`; it writes
/// the accumulator at the matching slot via `atom.global.<op>.<dtype>`.
///
/// On bounded-probe / bounded-spin overflow the thread atomically
/// increments `spill_counter_ptr` and (if `< max_spill`) writes its
/// `(key, candidate_value)` pair into the parallel spill buffers. The
/// host folds these into the per-group accumulator after kernel sync.
pub fn compile_agg_valid_kernel(
    op: ReduceOp,
    input_dtype: DataType,
) -> BoltResult<String> {
    // Reject unsupported (op, dtype) combinations up front with explicit errors.
    let atomic = atomic_for(op, input_dtype)?;

    let (load_suffix, reg_class) = ptx_type_info(input_dtype)?;
    let elem_bytes = input_dtype.byte_width().ok_or_else(|| {
        BoltError::Other(format!(
            "valid_flag_kernels: variable-width dtype {:?} not supported",
            input_dtype
        ))
    })?;
    let store_suffix = ptx_store_suffix(input_dtype)?;

    let mut ptx = String::new();
    let entry = VALID_AGG_KERNEL_ENTRY;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_3,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_4,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_5,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_6,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_7,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_8,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_9,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_10").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<48>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<48>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    // Typed value register class for the input column load + atomic update.
    writeln!(
        ptx,
        "\t.reg .{ty}   %{rc}<4>;",
        ty = reg_decl_ty(input_dtype)?,
        rc = reg_class
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // tid = ctaid.x * ntid.x + tid.x; bail if tid >= n_rows.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{entry}_param_5];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // k and mask = k-1.
    writeln!(ptx, "\tld.param.u32 %r5, [{entry}_param_6];").map_err(write_err)?;
    writeln!(ptx, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;

    // max_probes = k * MAX_PROBE_FACTOR (bounded probe loop).
    writeln!(
        ptx,
        "\tmul.lo.u32 %r20, %r5, {factor};",
        factor = MAX_PROBE_FACTOR
    )
    .map_err(write_err)?;

    // Load this thread's key.
    writeln!(ptx, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd1, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rl0, [%rd2];").map_err(write_err)?;

    // Hash.
    writeln!(ptx, "\tmov.s64 %rl1, {FX_MUL};").map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s64 %rl2, %rl0, %rl1;").map_err(write_err)?;
    writeln!(ptx, "\tshr.u64 %rl3, %rl2, 32;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u64 %r7, %rl3;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r7, %r6;").map_err(write_err)?;

    // Keys + valid base pointers.
    writeln!(ptx, "\tld.param.u64 %rd3, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd4, [{entry}_param_2];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;

    // Load this thread's input value EARLY (before the probe) so the SPILL
    // path can write it without re-fetching from the input column.
    writeln!(ptx, "\tld.param.u64 %rd9, [{entry}_param_3];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd9, %rd9;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd10, %r3, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd9, %rd10;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.global.{ld} %{rc}0, [%rd11];",
        ld = load_suffix,
        rc = reg_class
    )
    .map_err(write_err)?;

    // Probe-step counter (bounded probe — bails into SPILL on overflow).
    // %r21 holds the running probe count; %r22 is the per-slot spin counter
    // (set inside SPIN).
    writeln!(ptx, "\tmov.u32 %r21, 0;").map_err(write_err)?;

    // Probe: walk slots, spinning on valid==2 at each, until we find the
    // matching key. We are guaranteed to find one (keys kernel already ran),
    // unless the keys kernel itself spilled this row — then this probe also
    // exhausts and we spill from here too.
    writeln!(ptx, "PROBE_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r21, %r21, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.gt.u32 %p4, %r21, %r20;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra SPILL;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd5, %r8, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd6, %rd3, %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd7, %r8, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd8, %rd4, %rd7;").map_err(write_err)?;
    // Spin on this slot's valid flag until it's committed. The keys kernel
    // synchronises on the stream before we launch, so for occupied slots
    // valid==2 already; this check also tolerates concurrent keys+agg
    // launches if a future scheduler interleaves them. Bounded by
    // SPIN_LIMIT iterations.
    writeln!(ptx, "\tmov.u32 %r22, 0;").map_err(write_err)?;
    writeln!(ptx, "SPIN:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r22, %r22, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.gt.u32 %p5, %r22, {limit};",
        limit = SPIN_LIMIT
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra SPILL;").map_err(write_err)?;
    // ACQUIRE-LOAD: pairs with the publisher's atomic store; makes the
    // read-of-published-value contract explicit (sm_70+). Replaces the
    // plain `ld.<space>.<ty>` which relied on the publisher's release +
    // SASS-level implicit acquire — sound in practice but not promised
    // by PTX semantics.
    writeln!(ptx, "\tld.acquire.gpu.u32 %r9, [%rd8];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p1, %r9, 0;").map_err(write_err)?;
    // valid==0 means this row has no matching group slot — impossible after
    // a successful keys kernel UNLESS the keys kernel itself spilled this
    // row. Treat it as a probe miss and advance rather than spin forever;
    // if we exhaust the bounded probe we end up in SPILL (which is the
    // correct outcome — the spilled keys kernel never placed this key in
    // the table).
    writeln!(ptx, "\t@%p1 bra ADVANCE;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p2, %r9, 2;").map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra SPIN;").map_err(write_err)?;
    // valid==2 here: key is committed. Compare.
    writeln!(ptx, "\tld.global.s64 %rl5, [%rd6];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s64 %p3, %rl5, %rl0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra FOUND;").map_err(write_err)?;
    writeln!(ptx, "ADVANCE:").map_err(write_err)?;
    writeln!(ptx, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;
    writeln!(ptx, "FOUND:").map_err(write_err)?;

    // acc_table + slot * elem_bytes.
    writeln!(ptx, "\tld.param.u64 %rd12, [{entry}_param_4];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd12, %rd12;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd13, %r8, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd14, %rd12, %rd13;").map_err(write_err)?;

    // Atomic update. PTX `atom` returns the old value into a destination
    // register; we don't need it but the form requires one — reuse the
    // value register class with a fresh index.
    writeln!(
        ptx,
        "\t{atomic} %{rc}1, [%rd14], %{rc}0;",
        atomic = atomic,
        rc = reg_class
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra DONE;").map_err(write_err)?;

    // SPILL: bounded probe / spin exceeded — atomically claim an entry in
    // the host-allocated spill buffer and write our (key, candidate value)
    // pair. Host folds these into the per-group accumulator after sync.
    writeln!(ptx, "SPILL:").map_err(write_err)?;
    // Load spill_counter pointer + max_spill scalar.
    writeln!(ptx, "\tld.param.u64 %rd30, [{entry}_param_9];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd30, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r23, [{entry}_param_10];").map_err(write_err)?;
    writeln!(ptx, "\tatom.global.add.u32 %r24, [%rd30], 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p6, %r24, %r23;").map_err(write_err)?;
    writeln!(ptx, "\t@%p6 bra DONE;").map_err(write_err)?;
    // Write key into spill_keys[%r24].
    writeln!(ptx, "\tld.param.u64 %rd31, [{entry}_param_7];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd31, %rd31;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd32, %r24, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd33, %rd31, %rd32;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s64 [%rd33], %rl0;").map_err(write_err)?;
    // Write candidate value into spill_values[%r24].
    writeln!(ptx, "\tld.param.u64 %rd34, [{entry}_param_8];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd34, %rd34;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd35, %r24, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd36, %rd34, %rd35;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tst.global.{st} [%rd36], %{rc}0;",
        st = store_suffix,
        rc = reg_class
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra DONE;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// PTX `atom.global.*` mnemonic (with no operands) for the given op + dtype.
/// Mirrors `hash_kernels::atomic_for` — kept private here so that file stays
/// the source of truth for which combinations the classic path supports too.
fn atomic_for(op: ReduceOp, dtype: DataType) -> BoltResult<&'static str> {
    use DataType::*;
    use ReduceOp::*;
    Ok(match (op, dtype) {
        (Sum, Int32) | (Count, Int32) => "atom.global.add.s32",
        // PTX has no `atom.add.s64` — only `.u64`. Two's-complement signed
        // addition is bit-identical to unsigned addition (both wrap modulo
        // 2^64), so emitting `.u64` for a signed accumulator is sound: any
        // signed-overflow behavior the user could rely on is undefined in
        // Rust / C anyway. See PTX ISA, "atom" — supported types are
        // {u32, s32, u64, f16, f16x2, f32, f64, bf16, bf16x2}.
        (Sum, Int64) | (Count, Int64) => "atom.global.add.u64",
        (Sum, Float32) | (Count, Float32) => "atom.global.add.f32",
        (Sum, Float64) | (Count, Float64) => "atom.global.add.f64",

        (Min, Int32) => "atom.global.min.s32",
        (Min, Int64) => "atom.global.min.s64",
        (Max, Int32) => "atom.global.max.s32",
        (Max, Int64) => "atom.global.max.s64",

        (Min, Float32) | (Min, Float64) | (Max, Float32) | (Max, Float64) => {
            return Err(BoltError::Other(
                "MIN/MAX over float not yet supported in GROUP BY (valid-flag variant)"
                    .into(),
            ))
        }

        (_, Bool) | (_, Utf8) => {
            return Err(BoltError::Type(format!(
                "valid_flag_kernels: aggregate over dtype {:?} not supported",
                dtype
            )))
        }
    })
}

/// `(ld_suffix, reg_class)` for the input column / accumulator value type.
fn ptx_type_info(dtype: DataType) -> BoltResult<(&'static str, &'static str)> {
    Ok(match dtype {
        DataType::Int32 => ("s32", "vr"),
        DataType::Int64 => ("s64", "vl"),
        DataType::Float32 => ("f32", "vf"),
        DataType::Float64 => ("f64", "vd"),
        DataType::Bool | DataType::Utf8 => {
            return Err(BoltError::Type(format!(
                "valid_flag_kernels: dtype {:?} not supported in aggregate kernel",
                dtype
            )))
        }
    })
}

/// PTX `st.global.<suffix>` mnemonic for storing a value of `dtype` into the
/// spill_values buffer. Same width as the `ld.global.<suffix>` used to load
/// the input column.
fn ptx_store_suffix(dtype: DataType) -> BoltResult<&'static str> {
    Ok(match dtype {
        DataType::Int32 => "s32",
        DataType::Int64 => "s64",
        DataType::Float32 => "f32",
        DataType::Float64 => "f64",
        DataType::Bool | DataType::Utf8 => {
            return Err(BoltError::Type(format!(
                "valid_flag_kernels: dtype {:?} not supported in aggregate kernel spill",
                dtype
            )))
        }
    })
}

/// PTX `.reg` declaration type for the input-value register class.
fn reg_decl_ty(dtype: DataType) -> BoltResult<&'static str> {
    Ok(match dtype {
        DataType::Int32 => "b32",
        DataType::Int64 => "b64",
        DataType::Float32 => "f32",
        DataType::Float64 => "f64",
        DataType::Bool | DataType::Utf8 => {
            return Err(BoltError::Type(format!(
                "valid_flag_kernels: dtype {:?} not supported in aggregate kernel",
                dtype
            )))
        }
    })
}

/// Adapt a `std::fmt::Error` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("valid_flag_kernels: write failed: {}", e))
}

// ---------------------------------------------------------------------------
// PV-stage-d: validity-aware kernel companions.
//
// These mirror the sentinel-free keys / agg kernels above but add a packed-
// bit validity input — Arrow-compatible LE layout (bit i of byte i/8 is the
// validity flag for row i; 0 = NULL, 1 = present). The kernel skips work
// for any row whose validity bit is 0 instead of relying on a host-side
// strip pass to drop NULL rows before upload.
//
// ## Packed-bit layout
//
// ```text
// validity: u8[ceil(n_rows / 8)]
//   bit j of validity[i/8] is row i's validity flag (Arrow LE convention)
// ```
//
// This matches `arrow_buffer::NullBuffer::buffer()`'s wire format exactly,
// so the host can hand the Arrow null buffer straight to the GPU without
// rebuilding it.
//
// ## ABI delta (vs `compile_keys_valid_kernel` / `compile_agg_valid_kernel`)
//
// One extra `.param .u64 validity_ptr` is appended to the parameter list
// (just before the spill block). A new entry-point name is used so the
// host launcher can dispatch via symbol lookup without disambiguating the
// shape from outside.
//
// ## Entry-point names
//
// * Keys: `bolt_groupby_keys_valid_with_validity`
// * Agg:  `bolt_groupby_agg_valid_with_validity`
// ---------------------------------------------------------------------------

/// Entry-point name of the keys kernel produced by
/// [`compile_keys_valid_kernel_with_validity`].
pub const VALID_KEYS_KERNEL_WITH_VALIDITY_ENTRY: &str =
    "bolt_groupby_keys_valid_with_validity";

/// Entry-point name of the aggregate-update kernel produced by
/// [`compile_agg_valid_kernel_with_validity`].
pub const VALID_AGG_KERNEL_WITH_VALIDITY_ENTRY: &str =
    "bolt_groupby_agg_valid_with_validity";

/// Pack a per-row boolean validity vector into the Arrow-compatible
/// little-endian packed-bit layout the `_with_validity` kernels expect.
///
/// Bit `i` of byte `i / 8` is the validity flag for row `i`:
///   * `1` — value is present.
///   * `0` — value is NULL.
///
/// The output length is `ceil(n_rows / 8)` bytes. Trailing bits past
/// `validity.len()` in the last byte are zero (treated as "row not
/// present" by the kernel — those rows are guarded by the `n_rows`
/// bound anyway).
///
/// This is a host-side helper; the kernel only needs the produced `Vec<u8>`
/// uploaded as a `GpuVec<u8>`. The implementation is bit-identical to
/// `arrow_buffer::BooleanBuffer::from_iter`, so existing Arrow NullBuffer
/// payloads can be reused directly if the caller already has one.
pub fn pack_validity_bits(validity: &[bool]) -> Vec<u8> {
    let n_bytes = (validity.len() + 7) / 8;
    let mut out = vec![0u8; n_bytes];
    for (i, &v) in validity.iter().enumerate() {
        if v {
            out[i / 8] |= 1u8 << (i % 8);
        }
    }
    out
}

/// Generate PTX for the keys-building kernel with a packed-bit validity
/// input.
///
/// ## ABI
///
/// ```text
/// .visible .entry bolt_groupby_keys_valid_with_validity(
///     .param .u64 group_col_ptr,      // i64 keys, length n_rows
///     .param .u64 keys_table_ptr,     // i64, length k
///     .param .u64 slot_valid_ptr,     // u32, length k, host-init 0
///     .param .u32 n_rows,
///     .param .u32 k,                  // power-of-two table size
///     .param .u64 validity_ptr,       // u8, length ceil(n_rows/8), Arrow LE bits
///     .param .u64 spill_keys_ptr,     // i64, length max_spill
///     .param .u64 spill_counter_ptr,  // u32, length 1
///     .param .u32 max_spill
/// )
/// ```
///
/// Validity check: the kernel computes `byte = validity[tid/8]` and tests
/// `byte & (1 << (tid & 7))`. If zero, the thread returns immediately
/// (no slot claim, no spill) — the row is NULL and contributes no group.
///
/// The rest of the kernel body is identical to
/// [`compile_keys_valid_kernel`].
pub fn compile_keys_valid_kernel_with_validity() -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = VALID_KEYS_KERNEL_WITH_VALIDITY_ENTRY;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_3,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_4,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_5,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_6,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_7,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_8").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Same register decls as the no-validity variant.
    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<48>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<48>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // tid = ctaid.x * ntid.x + tid.x; bail if tid >= n_rows.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{entry}_param_3];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.s32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // -------- Validity bit check.
    //   byte_idx = tid >> 3
    //   bit_idx  = tid & 7
    //   byte     = ld.global.u8 [validity + byte_idx]
    //   if ((byte >> bit_idx) & 1) == 0 -> DONE (this row is NULL).
    //
    // PTX has no `ld.global.u8` of a single bit, so we load a byte and
    // mask. The same pattern is used by the agg kernel below.
    writeln!(ptx, "\tld.param.u64 %rd30, [{entry}_param_5];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd30, %rd30;").map_err(write_err)?;
    // byte_idx = tid / 8 -> %r30; bit_idx = tid & 7 -> %r31
    writeln!(ptx, "\tshr.u32 %r30, %r3, 3;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r31, %r3, 7;").map_err(write_err)?;
    // addr = validity + byte_idx
    writeln!(ptx, "\tmul.wide.u32 %rd31, %r30, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd32, %rd30, %rd31;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u8 %r32, [%rd32];").map_err(write_err)?;
    // mask = 1 << bit_idx ; test = byte & mask
    writeln!(ptx, "\tshl.b32 %r33, 1, %r31;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r34, %r32, %r33;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p7, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p7 bra DONE;").map_err(write_err)?;

    // k and mask = k-1.
    writeln!(ptx, "\tld.param.u32 %r5, [{entry}_param_4];").map_err(write_err)?;
    writeln!(ptx, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.lo.u32 %r20, %r5, {factor};",
        factor = MAX_PROBE_FACTOR
    )
    .map_err(write_err)?;

    // Load this thread's key.
    writeln!(ptx, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.s32 %rd1, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rl0, [%rd2];").map_err(write_err)?;

    // Hash.
    writeln!(ptx, "\tmov.s64 %rl1, {FX_MUL};").map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s64 %rl2, %rl0, %rl1;").map_err(write_err)?;
    writeln!(ptx, "\tshr.u64 %rl3, %rl2, 32;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u64 %r7, %rl3;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r7, %r6;").map_err(write_err)?;

    // Keys + valid base pointers.
    writeln!(ptx, "\tld.param.u64 %rd3, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd4, [{entry}_param_2];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;

    writeln!(ptx, "\tmov.u32 %r21, 0;").map_err(write_err)?;
    writeln!(ptx, "PROBE_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r21, %r21, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.gt.u32 %p4, %r21, %r20;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra SPILL;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd5, %r8, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd6, %rd3, %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd7, %r8, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd8, %rd4, %rd7;").map_err(write_err)?;
    writeln!(ptx, "\tatom.global.cas.b32 %r9, [%rd8], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p1, %r9, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra CLAIM_SLOT;").map_err(write_err)?;

    writeln!(ptx, "\tmov.u32 %r22, 0;").map_err(write_err)?;
    writeln!(ptx, "SPIN:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r22, %r22, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.gt.u32 %p5, %r22, {limit};",
        limit = SPIN_LIMIT
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra SPILL;").map_err(write_err)?;
    // ACQUIRE-LOAD: pairs with the publisher's atomic store; makes the
    // read-of-published-value contract explicit (sm_70+). Replaces the
    // plain `ld.<space>.<ty>` which relied on the publisher's release +
    // SASS-level implicit acquire — sound in practice but not promised
    // by PTX semantics.
    writeln!(ptx, "\tld.acquire.gpu.u32 %r10, [%rd8];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p2, %r10, 2;").map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra SPIN;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rl5, [%rd6];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s64 %p3, %rl5, %rl0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra DONE;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    writeln!(ptx, "CLAIM_SLOT:").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s64 [%rd6], %rl0;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.gl;").map_err(write_err)?;
    writeln!(ptx, "\tatom.global.exch.b32 %r11, [%rd8], 2;").map_err(write_err)?;
    writeln!(ptx, "\tbra DONE;").map_err(write_err)?;

    writeln!(ptx, "SPILL:").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd20, [{entry}_param_7];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd20, %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r23, [{entry}_param_8];").map_err(write_err)?;
    writeln!(ptx, "\tatom.global.add.u32 %r24, [%rd20], 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p6, %r24, %r23;").map_err(write_err)?;
    writeln!(ptx, "\t@%p6 bra DONE;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd21, [{entry}_param_6];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd21, %rd21;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd22, %r24, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd23, %rd21, %rd22;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s64 [%rd23], %rl0;").map_err(write_err)?;
    writeln!(ptx, "\tbra DONE;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Generate PTX for the aggregate kernel with a packed-bit validity input.
///
/// ## ABI
///
/// ```text
/// .visible .entry bolt_groupby_agg_valid_with_validity(
///     .param .u64 group_col_ptr,        // i64 keys, length n_rows
///     .param .u64 keys_table_ptr,       // i64, length k
///     .param .u64 slot_valid_ptr,       // u32, length k
///     .param .u64 input_col_ptr,        // T,   length n_rows
///     .param .u64 acc_table_ptr,        // T,   length k
///     .param .u32 n_rows,
///     .param .u32 k,
///     .param .u64 validity_ptr,         // u8,  length ceil(n_rows/8), Arrow LE
///     .param .u64 spill_keys_ptr,
///     .param .u64 spill_values_ptr,
///     .param .u64 spill_counter_ptr,
///     .param .u32 max_spill
/// )
/// ```
///
/// As with the keys variant, a NULL row (validity bit 0) returns at the
/// top of the kernel before touching the accumulator. The remaining
/// body matches [`compile_agg_valid_kernel`] one-for-one (with the
/// param indices shifted by +1 to make room for `validity_ptr`).
///
/// Float MIN/MAX is delegated to `valid_flag_float` — same constraint
/// as the no-validity variant.
pub fn compile_agg_valid_kernel_with_validity(
    op: ReduceOp,
    input_dtype: DataType,
) -> BoltResult<String> {
    let atomic = atomic_for(op, input_dtype)?;
    let (load_suffix, reg_class) = ptx_type_info(input_dtype)?;
    let elem_bytes = input_dtype.byte_width().ok_or_else(|| {
        BoltError::Other(format!(
            "valid_flag_kernels: variable-width dtype {:?} not supported",
            input_dtype
        ))
    })?;
    let store_suffix = ptx_store_suffix(input_dtype)?;

    let mut ptx = String::new();
    let entry = VALID_AGG_KERNEL_WITH_VALIDITY_ENTRY;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_3,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_4,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_5,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_6,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_7,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_8,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_9,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_10,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_11").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<48>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<48>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    writeln!(
        ptx,
        "\t.reg .{ty}   %{rc}<4>;",
        ty = reg_decl_ty(input_dtype)?,
        rc = reg_class
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // tid = ctaid.x * ntid.x + tid.x; bail if tid >= n_rows.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{entry}_param_5];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.s32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // Validity bit check — same shape as in the keys-with-validity kernel.
    writeln!(ptx, "\tld.param.u64 %rd30, [{entry}_param_7];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd30, %rd30;").map_err(write_err)?;
    writeln!(ptx, "\tshr.u32 %r30, %r3, 3;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r31, %r3, 7;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd31, %r30, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd32, %rd30, %rd31;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u8 %r32, [%rd32];").map_err(write_err)?;
    writeln!(ptx, "\tshl.b32 %r33, 1, %r31;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r34, %r32, %r33;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p7, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p7 bra DONE;").map_err(write_err)?;

    writeln!(ptx, "\tld.param.u32 %r5, [{entry}_param_6];").map_err(write_err)?;
    writeln!(ptx, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.lo.u32 %r20, %r5, {factor};",
        factor = MAX_PROBE_FACTOR
    )
    .map_err(write_err)?;

    writeln!(ptx, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.s32 %rd1, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rl0, [%rd2];").map_err(write_err)?;

    writeln!(ptx, "\tmov.s64 %rl1, {FX_MUL};").map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s64 %rl2, %rl0, %rl1;").map_err(write_err)?;
    writeln!(ptx, "\tshr.u64 %rl3, %rl2, 32;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u64 %r7, %rl3;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r7, %r6;").map_err(write_err)?;

    writeln!(ptx, "\tld.param.u64 %rd3, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd4, [{entry}_param_2];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;

    // Load input value early (so SPILL has it ready).
    writeln!(ptx, "\tld.param.u64 %rd9, [{entry}_param_3];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd9, %rd9;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.s32 %rd10, %r3, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd9, %rd10;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.global.{ld} %{rc}0, [%rd11];",
        ld = load_suffix,
        rc = reg_class
    )
    .map_err(write_err)?;

    writeln!(ptx, "\tmov.u32 %r21, 0;").map_err(write_err)?;
    writeln!(ptx, "PROBE_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r21, %r21, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.gt.u32 %p4, %r21, %r20;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra SPILL;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd5, %r8, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd6, %rd3, %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd7, %r8, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd8, %rd4, %rd7;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r22, 0;").map_err(write_err)?;
    writeln!(ptx, "SPIN:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r22, %r22, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.gt.u32 %p5, %r22, {limit};",
        limit = SPIN_LIMIT
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra SPILL;").map_err(write_err)?;
    // ACQUIRE-LOAD: pairs with the publisher's atomic store; makes the
    // read-of-published-value contract explicit (sm_70+). Replaces the
    // plain `ld.<space>.<ty>` which relied on the publisher's release +
    // SASS-level implicit acquire — sound in practice but not promised
    // by PTX semantics.
    writeln!(ptx, "\tld.acquire.gpu.u32 %r9, [%rd8];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p1, %r9, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra ADVANCE;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p2, %r9, 2;").map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra SPIN;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rl5, [%rd6];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s64 %p3, %rl5, %rl0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra FOUND;").map_err(write_err)?;
    writeln!(ptx, "ADVANCE:").map_err(write_err)?;
    writeln!(ptx, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;
    writeln!(ptx, "FOUND:").map_err(write_err)?;

    writeln!(ptx, "\tld.param.u64 %rd12, [{entry}_param_4];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd12, %rd12;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd13, %r8, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd14, %rd12, %rd13;").map_err(write_err)?;
    writeln!(
        ptx,
        "\t{atomic} %{rc}1, [%rd14], %{rc}0;",
        atomic = atomic,
        rc = reg_class
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra DONE;").map_err(write_err)?;

    writeln!(ptx, "SPILL:").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd40, [{entry}_param_10];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd40, %rd40;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r23, [{entry}_param_11];").map_err(write_err)?;
    writeln!(ptx, "\tatom.global.add.u32 %r24, [%rd40], 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p6, %r24, %r23;").map_err(write_err)?;
    writeln!(ptx, "\t@%p6 bra DONE;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd41, [{entry}_param_8];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd41, %rd41;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd42, %r24, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd43, %rd41, %rd42;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s64 [%rd43], %rl0;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd44, [{entry}_param_9];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd44, %rd44;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd45, %r24, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd46, %rd44, %rd45;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tst.global.{st} [%rd46], %{rc}0;",
        st = store_suffix,
        rc = reg_class
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra DONE;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

#[cfg(test)]
mod with_validity_tests {
    use super::*;

    /// `pack_validity_bits` should match the Arrow LE bit-packing convention.
    #[test]
    fn pack_validity_bits_arrow_le() {
        // Eight rows, alternating present/null: present=1, null=0.
        // Expected byte: bit 0 = 1, bit 1 = 0, bit 2 = 1, ... = 0b01010101 = 0x55.
        let validity = vec![true, false, true, false, true, false, true, false];
        let packed = pack_validity_bits(&validity);
        assert_eq!(packed, vec![0x55u8]);

        // A 17-row vector should produce ceil(17/8) = 3 bytes.
        let validity = vec![true; 17];
        let packed = pack_validity_bits(&validity);
        assert_eq!(packed.len(), 3);
        assert_eq!(packed[0], 0xFF);
        assert_eq!(packed[1], 0xFF);
        // Bit 0 of byte 2 = row 16, the only valid row in byte 2.
        assert_eq!(packed[2], 0x01);

        // Empty input -> empty output.
        let packed = pack_validity_bits(&[]);
        assert!(packed.is_empty());
    }

    /// The keys-with-validity kernel must export its entry name AND test
    /// the validity bit before the slot-claim CAS.
    #[test]
    fn keys_with_validity_emits_bit_check() {
        let ptx = compile_keys_valid_kernel_with_validity().expect("compile");
        assert!(ptx.contains(VALID_KEYS_KERNEL_WITH_VALIDITY_ENTRY));
        // Validity byte load + mask test.
        assert!(
            ptx.contains("ld.global.u8 %r32, [%rd32];"),
            "missing validity byte load:\n{ptx}"
        );
        assert!(
            ptx.contains("setp.eq.s32 %p7, %r34, 0;"),
            "missing validity bit-test predicate:\n{ptx}"
        );
        // The classic CAS is still present (we kept the same probe shape).
        assert!(ptx.contains("atom.global.cas.b32"));
        assert!(ptx.contains("atom.global.exch.b32"));
    }

    /// The agg-with-validity kernel must declare 12 params (vs 11 for
    /// the no-validity variant) and bit-test before the probe.
    #[test]
    fn agg_with_validity_param_count_and_bit_check() {
        let ptx = compile_agg_valid_kernel_with_validity(ReduceOp::Sum, DataType::Int64)
            .expect("compile");
        assert!(ptx.contains(VALID_AGG_KERNEL_WITH_VALIDITY_ENTRY));
        // param_11 is the new max_spill slot (validity_ptr is param_7).
        assert!(ptx.contains("_param_11"), "missing param_11:\n{ptx}");
        // No `_param_12` — defensive that we didn't accidentally bump too far.
        assert!(
            !ptx.contains("_param_12"),
            "kernel should declare exactly 12 params (0..11):\n{ptx}"
        );
        // The bit-test setp predicate %p7.
        assert!(
            ptx.contains("setp.eq.s32 %p7, %r34, 0;"),
            "missing bit-test predicate:\n{ptx}"
        );
        // Sum, Int64 -> atom.global.add.u64.
        assert!(ptx.contains("atom.global.add.u64"));
    }
}

// ---------------------------------------------------------------------------
// PTX-shape tests (host-only — no CUDA required).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// The keys kernel must use `atom.global.cas.b32` for slot claim and
    /// `atom.global.exch.b32` for the commit publish, and must export the
    /// expected entry-point name.
    #[test]
    fn keys_valid_kernel_contains_cas_and_exch() {
        let ptx = compile_keys_valid_kernel().expect("keys kernel compiles");
        assert!(
            ptx.contains("atom.global.cas.b32"),
            "keys kernel must use atom.global.cas.b32 to claim a slot:\n{ptx}"
        );
        assert!(
            ptx.contains("atom.global.exch.b32"),
            "keys kernel must use atom.global.exch.b32 to publish valid=2:\n{ptx}"
        );
        assert!(
            ptx.contains(VALID_KEYS_KERNEL_ENTRY),
            "keys kernel must export the {VALID_KEYS_KERNEL_ENTRY} entry point"
        );
        // Sentinel sanity: the EMPTY_KEY value (`-9223372036854775808`) must
        // NOT appear anywhere in the kernel — the whole point of this variant
        // is that it doesn't reserve a sentinel value.
        assert!(
            !ptx.contains("-9223372036854775808"),
            "valid-flag kernel must NOT reference the i64::MIN sentinel:\n{ptx}"
        );
    }

    /// The agg kernel must read the slot-valid flag before reading the key.
    /// The read uses `ld.acquire.gpu.u32` (sm_70+) so the publisher's
    /// `atom.global.exch.b32` + reader-acquire pair gives explicit PTX-level
    /// ordering rather than relying on SASS-level implicit acquire.
    #[test]
    fn agg_valid_kernel_reads_valid_flag() {
        let ptx = compile_agg_valid_kernel(ReduceOp::Sum, DataType::Int64)
            .expect("agg kernel compiles");
        assert!(
            ptx.contains("ld.acquire.gpu.u32"),
            "agg kernel must issue ld.acquire.gpu.u32 to inspect slot_valid \
             (acquire pairs with the keys-kernel publish):\n{ptx}"
        );
        assert!(
            ptx.contains(VALID_AGG_KERNEL_ENTRY),
            "agg kernel must export the {VALID_AGG_KERNEL_ENTRY} entry point"
        );
        assert!(
            ptx.contains("atom.global.add.u64"),
            "agg kernel for (Sum, Int64) must emit atom.global.add.u64 \
             (PTX has no .s64 variant for atom.add; .u64 is bit-identical \
             for two's-complement signed addition):\n{ptx}"
        );
    }

    /// Utf8 (variable width) is rejected up front by the agg kernel.
    #[test]
    fn valid_kernel_rejects_utf8_input_dtype() {
        let err = compile_agg_valid_kernel(ReduceOp::Sum, DataType::Utf8)
            .expect_err("Utf8 input dtype should be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("Utf8") || msg.contains("not supported"),
            "expected dtype-rejection error, got: {msg}"
        );
    }

    /// Sanity: keys kernel includes the spin-on-2 wait so losers don't read
    /// uncommitted key memory.
    #[test]
    fn keys_kernel_contains_spin_on_commit() {
        let ptx = compile_keys_valid_kernel().expect("compile");
        // The loser-spin path should compare the loaded valid against 2.
        assert!(
            ptx.contains("setp.ne.s32 %p2, %r10, 2"),
            "keys kernel should spin until slot_valid == 2:\n{ptx}"
        );
        assert!(
            ptx.contains("membar.gl"),
            "keys kernel must issue membar.gl between key store and valid publish:\n{ptx}"
        );
    }

    /// The keys kernel must include the bounded-probe counter, the SPILL
    /// label, and the atomic spill-counter increment. Together these are the
    /// deadlock-hardening contract: no thread can spin or probe indefinitely.
    #[test]
    fn keys_kernel_contains_probe_limit_and_spill() {
        let ptx = compile_keys_valid_kernel().expect("compile");
        // The probe-count register (%r21) is zeroed at probe entry.
        assert!(
            ptx.contains("mov.u32 %r21, 0;"),
            "keys kernel must initialise the bounded-probe counter:\n{ptx}"
        );
        // Probe overflow predicate: probe_count > max_probes.
        assert!(
            ptx.contains("setp.gt.u32 %p4, %r21, %r20"),
            "keys kernel must compare probe_count (%r21) against max_probes (%r20):\n{ptx}"
        );
        assert!(
            ptx.contains("SPILL:"),
            "keys kernel must declare a SPILL label for overflowed probes:\n{ptx}"
        );
        assert!(
            ptx.contains("atom.global.add.u32"),
            "keys kernel SPILL path must atomically claim a spill slot:\n{ptx}"
        );
        // The inner SPIN loop must also be bounded; its counter (%r22) is
        // compared against the literal SPIN_LIMIT.
        assert!(
            ptx.contains("setp.gt.u32 %p5, %r22,"),
            "keys kernel must bound the SPIN loop too:\n{ptx}"
        );
    }

    /// The agg kernel must include the same hardening: bounded probe,
    /// bounded spin, SPILL label, atomic spill-counter increment, and a
    /// write of the candidate value to the spill_values buffer.
    #[test]
    fn agg_kernel_contains_probe_limit_and_spill() {
        let ptx = compile_agg_valid_kernel(ReduceOp::Sum, DataType::Int64)
            .expect("compile");
        assert!(
            ptx.contains("mov.u32 %r21, 0;"),
            "agg kernel must initialise the bounded-probe counter:\n{ptx}"
        );
        assert!(
            ptx.contains("setp.gt.u32 %p4, %r21, %r20"),
            "agg kernel must compare probe_count (%r21) against max_probes (%r20):\n{ptx}"
        );
        assert!(
            ptx.contains("SPILL:"),
            "agg kernel must declare a SPILL label for overflowed probes:\n{ptx}"
        );
        assert!(
            ptx.contains("atom.global.add.u32"),
            "agg kernel SPILL path must atomically claim a spill slot:\n{ptx}"
        );
        assert!(
            ptx.contains("setp.gt.u32 %p5, %r22,"),
            "agg kernel must bound the SPIN loop too:\n{ptx}"
        );
        // The agg kernel ABI now declares param_10 (max_spill); without it
        // the SPILL path can't bound-check the atomic counter.
        assert!(
            ptx.contains("_param_10"),
            "agg kernel must declare param_10 (max_spill):\n{ptx}"
        );
    }
}
