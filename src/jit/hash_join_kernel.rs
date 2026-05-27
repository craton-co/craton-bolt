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
//! ## Stage-2 additions
//!
//! Stage 2 layers four new capabilities on top of the Stage-1 emitter without
//! perturbing the Stage-1 entry points (`bolt_hash_join_build` /
//! `bolt_hash_join_probe`) — they keep emitting byte-identical PTX so the
//! Stage-1 unique-key INNER fast path is preserved. Stage 2 adds *separate*
//! entry points:
//!
//! * **Collision-list build kernel** — [`BUILD_COLLISION_KERNEL_ENTRY`].
//!   Drops the "unique build keys" gate. Each slot stores a list head
//!   `head[slot]` into a parallel `next_idx[row]` array; insertion walks the
//!   linear probe to the first slot whose key equals ours (or an EMPTY
//!   slot), then atomically prepends our row to that slot's chain.
//! * **Collision-list probe kernel** — [`PROBE_COLLISION_KERNEL_ENTRY`].
//!   Walks the linked list rooted at `head[slot]` via `next_idx[]`, emitting
//!   one output pair per visited build row. Each visited row may also set
//!   the per-build-row "matched" bitmap.
//! * **Matched-bitmap atomic set** — folded into the collision-list probe.
//!   `matched: u8[build_n_rows]` is zero-initialised by the host and the
//!   probe uses `atom.global.or.b32` (with a 4-byte aligned address)
//!   to set the matched bit. Outer-join orchestration only needs a "was
//!   this build row ever touched?" view; we don't care about ordering, so
//!   the cheap atomic-OR is correct.
//! * **Outer-join second-pass kernel** —
//!   [`UNMATCHED_BUILD_KERNEL_ENTRY`]. One thread per build row; if
//!   `matched[tid] == 0`, atomically claims an output slot and writes
//!   `out_build_idx[claimed] = tid`. The probe-side index is left to the
//!   host (it materialises NULL via `arrow::compute::take` with a `Null`
//!   index).
//!
//! ## Multi-key joins
//!
//! Multi-key joins fold their key tuple into a single i64 *on the host*
//! (see `gpu_join::encode_keys_for_shape`) and then reuse the existing
//! kernels exactly as-is — the kernels never see anything other than i64
//! keys. Two-i32 keys pack as `(k1 as u32 as u64) << 32 | (k2 as u32 as
//! u64)`; wider tuples fold via the same splitmix multiplier the kernels
//! use, which means kernel and host hash agree byte-for-byte on the
//! single i64 they share.
//!
//! ## Float / Bool / sentinel handling
//!
//! Float keys go through `f64::NAN.to_bits()` canonicalisation on the host
//! before being reinterpreted as i64; the kernels see only i64 values, so
//! there's no float arithmetic on the GPU. Bool keys upload as 0/1
//! i64. `i64::MIN` remains reserved as the empty-slot sentinel; the host
//! rejects any tuple whose packed encoding collides.
//!
//! ## Stage-3 follow-ups
//!
//! * **Lift the slot-table layout to AoS.** SoA (parallel keys + head + chain)
//!   wastes a cacheline per slot when most slots have a single occupant.
//!   AoS-with-inline-first-row would halve memory traffic on the probe.
//! * **Larger hash tables.** Stage 2 already lifts the cap to 512 MiB on
//!   ≥ 8 GiB cards; a future pass should be measure-driven per workload.
//! * **Utf8 keys.** Currently still host-only.

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

/// Entry-point name of the Stage-2 collision-list build kernel — handles
/// duplicate build keys by chaining them in a per-slot linked list rooted
/// at `head[slot]` with edges stored in `next_idx[]`.
pub const BUILD_COLLISION_KERNEL_ENTRY: &str = "bolt_hash_join_build_collision";

/// Entry-point name of the Stage-2 collision-list probe kernel — walks the
/// linked list at the matching slot and emits one output pair per visited
/// build row, optionally setting the build-side "matched" bitmap.
pub const PROBE_COLLISION_KERNEL_ENTRY: &str = "bolt_hash_join_probe_collision";

/// Entry-point name of the Stage-2 outer-join second-pass kernel — emits the
/// build-row index of every build row whose `matched[tid]` byte is still 0.
pub const UNMATCHED_BUILD_KERNEL_ENTRY: &str = "bolt_hash_join_emit_unmatched_build";

/// Entry-point name of the Stage-3 CROSS JOIN kernel — one thread per output
/// pair `(probe_idx, build_idx)`. No hash table involved.
pub const CROSS_KERNEL_ENTRY: &str = "bolt_cross_join";

/// Entry-point name of the Stage-3 AoS-layout probe kernel — same semantics
/// as [`PROBE_KERNEL_ENTRY`] but reads each hash-table slot as a single
/// 16-byte tuple `[key:u64, head:u32, _pad:u32]`. Halves probe-side memory
/// traffic on workloads where the probe loop is DRAM-bound.
pub const PROBE_AOS_KERNEL_ENTRY: &str = "bolt_hash_join_probe_aos";

/// Entry-point name of the Stage-4 AoS-layout build kernel — symmetric with
/// [`PROBE_AOS_KERNEL_ENTRY`]. Inserts `(key, row_idx)` into the same
/// `[key:u64, head:u32, _pad:u32]` slot layout: the i64-aligned CAS at
/// slot offset 0 still drives slot ownership, but instead of writing into
/// a parallel `row_idx_table[cap]` the winning thread writes its tid into
/// the same 16-byte slot at offset 8. The `_pad` word at offset 12 is left
/// zero so the slot stays compatible with the AoS probe's 16-byte fused
/// load.
pub const BUILD_AOS_KERNEL_ENTRY: &str = "bolt_hash_join_build_aos";

/// AoS slot footprint in bytes: `i64 key (8) + u32 head (4) + u32 _pad (4)
/// = 16`. The pad keeps each slot 16-byte aligned for the `ld.global.v4.u32`
/// fused load the AoS probe issues. Memory cost vs. SoA is 16/12 = 1.33× —
/// the extra padding buys one fused load per slot read instead of two
/// scattered loads against parallel arrays.
///
/// **Stage-4 memory analysis** (paired with [`compile_build_aos_kernel`]):
/// * SoA (Stage 1/2/3): `keys_table: i64[cap]` + `row_idx_table: u32[cap]`
///   = `cap * 12` bytes.
/// * AoS (Stage 4): `slots: u8[cap * 16]` = `cap * 16` bytes.
///
/// AoS is **33% larger in raw bytes** (`16/12 = 1.333…`) but halves the
/// number of probe-side cache lines touched: one load brings both `key`
/// and `head` (= `row_idx`) into the same 16-byte transaction, vs the
/// SoA path's two scattered loads against parallel arrays. On probe-bound
/// workloads (which dominate join cost) the cache-line saving is worth
/// the extra padding.
pub const AOS_SLOT_BYTES: u32 = 16;

/// Shape of the join key, as seen by the host-side encoder. The kernels
/// themselves never branch on this — every shape is folded into a single
/// i64 on the host before upload — but the value carries the host-side
/// encoding strategy through the join executor.
///
/// * [`SingleI32`] / [`SingleI64`] — Stage 1 single-key inputs, identity /
///   sign-extension into i64.
/// * [`SingleBool`] — bool re-encoded as 0/1 i64.
/// * [`SingleF32`] / [`SingleF64`] — float bit-pattern reinterpreted as i64
///   after NaN canonicalisation on the host.
/// * [`TwoI32`] — two Int32 columns packed as `(hi as u32 as u64) << 32
///   | (lo as u32 as u64)`. Matches `groupby::pack_keys` exactly.
/// * [`TwoI64`] — two Int64 columns folded into a 64-bit splitmix hash
///   on the host. Stage 3 admits this as a *candidate filter*: the GPU
///   emits a superset of true matches; host post-verification re-tests
///   each pair against the full tuple and drops false positives.
/// * [`MultiI32(n)`] — three or more Int32 columns; same candidate-filter
///   model as `TwoI64`. Host falls back wholesale only when interning /
///   re-encoding the candidate set is more expensive than the host join.
/// * [`SingleUtf8`] — Single string key. Host interns via
///   `gpu_join::intern_utf8_columns` to produce a dictionary index per row,
///   then routes through the kernel-side i64 path (the kernel never sees
///   Utf8). Output reattaches the original `StringArray` columns via
///   `take`.
///
/// [`SingleI32`]: KeyShape::SingleI32
/// [`SingleI64`]: KeyShape::SingleI64
/// [`SingleBool`]: KeyShape::SingleBool
/// [`SingleF32`]: KeyShape::SingleF32
/// [`SingleF64`]: KeyShape::SingleF64
/// [`TwoI32`]: KeyShape::TwoI32
/// [`TwoI64`]: KeyShape::TwoI64
/// [`MultiI32(n)`]: KeyShape::MultiI32
/// [`SingleUtf8`]: KeyShape::SingleUtf8
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyShape {
    /// Single Int32 key, sign-extended to i64 on the host.
    SingleI32,
    /// Single Int64 key, identity-encoded as i64.
    SingleI64,
    /// Single Boolean key, encoded as `i64` 0 or 1.
    SingleBool,
    /// Single Float32 key — bit-pattern after NaN canonicalisation.
    SingleF32,
    /// Single Float64 key — bit-pattern after NaN canonicalisation.
    SingleF64,
    /// Single Utf8 key — interned to i32 dict indices on the host before
    /// upload. Stage-3.
    SingleUtf8,
    /// Two Int32 columns packed `(hi << 32) | lo`.
    TwoI32,
    /// Two Int64 columns folded into one i64 by the host via splitmix.
    /// Stage-3 admits this with mandatory host post-verification.
    TwoI64,
    /// `n` Int32 columns folded into one i64 by the host via splitmix.
    /// `n >= 3`. Same candidate-filter contract as `TwoI64`.
    MultiI32(u8),
}

impl KeyShape {
    /// True if the kernel-side path is exact (no fold-collision risk). When
    /// false (`TwoI64`, `MultiI32(_)`), the host folds the tuple to i64 by
    /// splitmix — collisions are possible and the host must verify any
    /// match before accepting it.
    ///
    /// Stage 2's GPU path declines the fast path when this returns false
    /// (the host hash-join takes over). A future Stage emits an additional
    /// per-pair host-side verification step so the lossy fold can be
    /// accepted on the GPU side as a candidate filter.
    pub fn is_exact_in_i64(self) -> bool {
        matches!(
            self,
            KeyShape::SingleI32
                | KeyShape::SingleI64
                | KeyShape::SingleBool
                | KeyShape::SingleF32
                | KeyShape::SingleF64
                | KeyShape::TwoI32
                // SingleUtf8 is exact in i64 once the host has interned the
                // strings to dictionary indices — the kernels see only
                // i32-as-i64 keys at that point, with no collision risk.
                | KeyShape::SingleUtf8
        )
    }

    /// True if this shape needs per-pair host-side verification after the
    /// GPU emits candidate matches. Stage-3 routes `TwoI64` and
    /// `MultiI32(n)` through the GPU as a fast pre-filter; the host then
    /// re-tests each `(probe_idx, build_idx)` pair against the original
    /// Arrow columns and drops false positives produced by the lossy
    /// 64-bit fold. Other shapes are exact and return `false`.
    pub fn needs_host_post_verify(self) -> bool {
        matches!(self, KeyShape::TwoI64 | KeyShape::MultiI32(_))
    }
}

/// Validate that `dtype` is a supported single-key dtype for the GPU hash join.
///
/// Stage 1 handled `Int32` / `Int64`. Stage 2 widens to `Bool`, `Float32`
/// and `Float64` — every shape still ends up as an i64 on the GPU after
/// host-side encoding, so the kernel itself only knows about i64.
/// Stage 3 adds `Utf8` via host-side string interning (the kernel still
/// only sees i32 dict indices).
pub fn is_supported_key_dtype(dtype: DataType) -> bool {
    matches!(
        dtype,
        DataType::Int32
            | DataType::Int64
            | DataType::Bool
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
    )
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

/// Compile the Stage-2 collision-list build kernel's PTX.
///
/// Differences vs the Stage-1 [`compile_build_kernel`]:
///
/// * Two extra `.u64` parameters at the tail — `head: u32[cap]` (initialised
///   to `u32::MAX` by the host = sentinel "empty list") and `next_idx:
///   u32[build_n_rows]` (uninitialised; every row writes its own entry
///   exactly once).
/// * The probe loop walks until *either* (a) the slot is EMPTY and we own
///   it after a CAS, *or* (b) the slot's key already equals ours (someone
///   else owns the slot but it's the same key we're trying to insert).
///   Once the destination slot is known, the kernel atomically prepends
///   this row to the chain: `next_idx[tid] = atom.global.exch.b32 head[slot],
///   tid`.
///
/// ```text
/// .visible .entry bolt_hash_join_build_collision(
///     .param .u64 keys_col_ptr,        // i64, length n_rows
///     .param .u64 keys_table_ptr,      // i64, length cap, init=i64::MIN
///     .param .u64 row_idx_table_ptr,   // u32, length cap (unused but kept
///                                      //   for ABI symmetry; first-writer's
///                                      //   tid only)
///     .param .u64 head_ptr,            // u32, length cap, init=u32::MAX
///     .param .u64 next_idx_ptr,        // u32, length n_rows
///     .param .u32 n_rows,
///     .param .u32 cap
/// )
/// ```
pub fn compile_build_collision_kernel() -> BoltResult<String> {
    let mut p = String::new();
    writeln!(p, "{PTX_VERSION}").map_err(write_err)?;
    writeln!(p, "{PTX_TARGET}").map_err(write_err)?;
    writeln!(p, "{PTX_ADDRESS_SIZE}").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    let entry = BUILD_COLLISION_KERNEL_ENTRY;
    writeln!(p, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_3,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_4,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_5,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_6").map_err(write_err)?;
    writeln!(p, ")").map_err(write_err)?;
    writeln!(p, "{{").map_err(write_err)?;

    // Register file — same shape as Stage-1 build plus a few scratch regs for
    // the head/next pointer arithmetic.
    writeln!(p, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64   %rd<24>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    // tid = ctaid * ntid + tid_x ; bail if tid >= n_rows.
    writeln!(p, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(p, "\tld.param.u32 %r4, [{entry}_param_5];").map_err(write_err)?;
    writeln!(p, "\tsetp.ge.s32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(p, "\t@%p0 bra DONE;").map_err(write_err)?;

    // cap and mask = cap - 1.
    writeln!(p, "\tld.param.u32 %r5, [{entry}_param_6];").map_err(write_err)?;
    writeln!(p, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;
    writeln!(p, "\tmul.lo.u32 %r20, %r5, {MAX_PROBE_FACTOR};").map_err(write_err)?;

    // Load this row's key.
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

    // base pointers.
    writeln!(p, "\tld.param.u64 %rd3, [{entry}_param_1];").map_err(write_err)?; // keys_table
    writeln!(p, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(p, "\tld.param.u64 %rd4, [{entry}_param_2];").map_err(write_err)?; // row_idx_table
    writeln!(p, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;
    writeln!(p, "\tld.param.u64 %rd14, [{entry}_param_3];").map_err(write_err)?; // head
    writeln!(p, "\tcvta.to.global.u64 %rd14, %rd14;").map_err(write_err)?;
    writeln!(p, "\tld.param.u64 %rd15, [{entry}_param_4];").map_err(write_err)?; // next_idx
    writeln!(p, "\tcvta.to.global.u64 %rd15, %rd15;").map_err(write_err)?;

    writeln!(p, "\tmov.s64 %rl4, {EMPTY_KEY_LITERAL};").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r21, 0;").map_err(write_err)?;

    // Probe loop: find a slot we can claim or that already holds our key.
    writeln!(p, "PROBE_LOOP:").map_err(write_err)?;
    writeln!(p, "\tadd.u32 %r21, %r21, 1;").map_err(write_err)?;
    writeln!(p, "\tsetp.gt.u32 %p3, %r21, %r20;").map_err(write_err)?;
    writeln!(p, "\t@%p3 bra DONE;").map_err(write_err)?;

    // addr_keys = keys_table + slot * 8
    writeln!(p, "\tmul.wide.u32 %rd5, %r8, 8;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd6, %rd3, %rd5;").map_err(write_err)?;
    writeln!(p, "\tatom.global.cas.b64 %rl5, [%rd6], %rl4, %rl0;").map_err(write_err)?;

    // prev == EMPTY  -> we now own the slot.
    writeln!(p, "\tsetp.eq.s64 %p1, %rl5, %rl4;").map_err(write_err)?;
    writeln!(p, "\t@%p1 bra OWNED_SLOT;").map_err(write_err)?;

    // prev == key    -> someone else owns the slot but it's our key. Chain.
    writeln!(p, "\tsetp.eq.s64 %p2, %rl5, %rl0;").map_err(write_err)?;
    writeln!(p, "\t@%p2 bra CHAIN;").map_err(write_err)?;

    // Collision: linear-probe to the next slot.
    writeln!(p, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(p, "\tbra PROBE_LOOP;").map_err(write_err)?;

    writeln!(p, "OWNED_SLOT:").map_err(write_err)?;
    // Record the *first* row index for backward-compat with the Stage-1
    // row_idx_table layout — readers that don't walk the chain still see
    // the head.
    writeln!(p, "\tmul.wide.u32 %rd7, %r8, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd8, %rd4, %rd7;").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd8], %r3;").map_err(write_err)?;
    // Fall through to chain insertion so this row is also reachable via head.

    writeln!(p, "CHAIN:").map_err(write_err)?;
    // addr_head = head + slot * 4
    writeln!(p, "\tmul.wide.u32 %rd9, %r8, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd10, %rd14, %rd9;").map_err(write_err)?;
    // Atomic exchange: prev_head = head[slot]; head[slot] = tid.
    writeln!(p, "\tatom.global.exch.b32 %r22, [%rd10], %r3;").map_err(write_err)?;
    // next_idx[tid] = prev_head.
    writeln!(p, "\tmul.wide.s32 %rd11, %r3, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd12, %rd15, %rd11;").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd12], %r22;").map_err(write_err)?;

    writeln!(p, "DONE:").map_err(write_err)?;
    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;
    Ok(p)
}

/// Compile the Stage-2 collision-list probe kernel's PTX.
///
/// Reuses the Stage-1 keys_table layout for slot lookup, then walks the
/// per-slot linked list `head[slot] -> next_idx[head] -> ...` until a
/// `u32::MAX` sentinel is reached, emitting one output pair per node.
/// Optionally sets `matched[build_idx] = 1` via an atomic OR — the host
/// passes a null `matched_ptr` for INNER and a real bitmap for outer.
///
/// ```text
/// .visible .entry bolt_hash_join_probe_collision(
///     .param .u64 probe_keys_ptr,    // i64, length n_probe
///     .param .u64 keys_table_ptr,    // i64, length cap
///     .param .u64 head_ptr,          // u32, length cap (init=u32::MAX)
///     .param .u64 next_idx_ptr,      // u32, length build_n_rows
///     .param .u64 out_probe_idx_ptr, // u32, length out_capacity
///     .param .u64 out_build_idx_ptr, // u32, length out_capacity
///     .param .u64 out_counter_ptr,   // u32, single counter (init=0)
///     .param .u64 matched_ptr,       // u32, ceil(build_n_rows/4) — may be 0
///     .param .u32 n_probe,
///     .param .u32 cap,
///     .param .u32 out_capacity,
///     .param .u32 build_n_rows       // for sentinel check on chain walk
/// )
/// ```
pub fn compile_probe_collision_kernel() -> BoltResult<String> {
    let mut p = String::new();
    writeln!(p, "{PTX_VERSION}").map_err(write_err)?;
    writeln!(p, "{PTX_TARGET}").map_err(write_err)?;
    writeln!(p, "{PTX_ADDRESS_SIZE}").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    let entry = PROBE_COLLISION_KERNEL_ENTRY;
    writeln!(p, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_3,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_4,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_5,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_6,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_7,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_8,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_9,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_10,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_11").map_err(write_err)?;
    writeln!(p, ")").map_err(write_err)?;
    writeln!(p, "{{").map_err(write_err)?;

    writeln!(p, "\t.reg .pred  %p<16>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32   %r<48>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64   %rd<32>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    // tid bounds.
    writeln!(p, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(p, "\tld.param.u32 %r4, [{entry}_param_8];").map_err(write_err)?;
    writeln!(p, "\tsetp.ge.s32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(p, "\t@%p0 bra DONE;").map_err(write_err)?;

    // cap + mask, out_capacity, build_n_rows.
    writeln!(p, "\tld.param.u32 %r5, [{entry}_param_9];").map_err(write_err)?;
    writeln!(p, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;
    writeln!(p, "\tmul.lo.u32 %r20, %r5, {MAX_PROBE_FACTOR};").map_err(write_err)?;
    writeln!(p, "\tld.param.u32 %r23, [{entry}_param_10];").map_err(write_err)?;
    writeln!(p, "\tld.param.u32 %r24, [{entry}_param_11];").map_err(write_err)?;

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

    // base pointers.
    writeln!(p, "\tld.param.u64 %rd3, [{entry}_param_1];").map_err(write_err)?; // keys_table
    writeln!(p, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(p, "\tld.param.u64 %rd14, [{entry}_param_2];").map_err(write_err)?; // head
    writeln!(p, "\tcvta.to.global.u64 %rd14, %rd14;").map_err(write_err)?;
    writeln!(p, "\tld.param.u64 %rd15, [{entry}_param_3];").map_err(write_err)?; // next_idx
    writeln!(p, "\tcvta.to.global.u64 %rd15, %rd15;").map_err(write_err)?;
    writeln!(p, "\tld.param.u64 %rd16, [{entry}_param_4];").map_err(write_err)?; // out_probe_idx
    writeln!(p, "\tcvta.to.global.u64 %rd16, %rd16;").map_err(write_err)?;
    writeln!(p, "\tld.param.u64 %rd17, [{entry}_param_5];").map_err(write_err)?; // out_build_idx
    writeln!(p, "\tcvta.to.global.u64 %rd17, %rd17;").map_err(write_err)?;
    writeln!(p, "\tld.param.u64 %rd18, [{entry}_param_6];").map_err(write_err)?; // counter
    writeln!(p, "\tcvta.to.global.u64 %rd18, %rd18;").map_err(write_err)?;
    writeln!(p, "\tld.param.u64 %rd19, [{entry}_param_7];").map_err(write_err)?; // matched
    // We *do not* convert matched_ptr unconditionally — the host may pass a
    // raw 0 for INNER joins, and cvta on null can trap on some drivers. We
    // gate on rd19 != 0 below.

    writeln!(p, "\tmov.s64 %rl4, {EMPTY_KEY_LITERAL};").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r21, 0;").map_err(write_err)?;

    // Outer loop: linear-probe keys_table to find the slot whose key matches
    // or the first empty slot (no-match).
    writeln!(p, "PROBE_LOOP:").map_err(write_err)?;
    writeln!(p, "\tadd.u32 %r21, %r21, 1;").map_err(write_err)?;
    writeln!(p, "\tsetp.gt.u32 %p3, %r21, %r20;").map_err(write_err)?;
    writeln!(p, "\t@%p3 bra DONE;").map_err(write_err)?;

    writeln!(p, "\tmul.wide.u32 %rd5, %r8, 8;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd6, %rd3, %rd5;").map_err(write_err)?;
    writeln!(p, "\tld.global.s64 %rl5, [%rd6];").map_err(write_err)?;

    // Empty -> no match.
    writeln!(p, "\tsetp.eq.s64 %p1, %rl5, %rl4;").map_err(write_err)?;
    writeln!(p, "\t@%p1 bra DONE;").map_err(write_err)?;

    // Match -> walk the chain.
    writeln!(p, "\tsetp.eq.s64 %p2, %rl5, %rl0;").map_err(write_err)?;
    writeln!(p, "\t@%p2 bra WALK_CHAIN;").map_err(write_err)?;

    // Mismatch -> advance slot.
    writeln!(p, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(p, "\tbra PROBE_LOOP;").map_err(write_err)?;

    // Chain walk: cursor = head[slot]; while cursor != u32::MAX, emit and
    // advance via next_idx[cursor].
    writeln!(p, "WALK_CHAIN:").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd20, %r8, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd21, %rd14, %rd20;").map_err(write_err)?;
    writeln!(p, "\tld.global.u32 %r25, [%rd21];").map_err(write_err)?; // cursor

    // CHAIN_LOOP — bail on cursor == u32::MAX (== -1 as i32) OR cursor >= build_n_rows.
    writeln!(p, "CHAIN_LOOP:").map_err(write_err)?;
    writeln!(p, "\tsetp.ge.u32 %p4, %r25, %r24;").map_err(write_err)?;
    writeln!(p, "\t@%p4 bra DONE;").map_err(write_err)?;

    // Atomic claim an output index.
    writeln!(p, "\tmov.u32 %r26, 1;").map_err(write_err)?;
    writeln!(p, "\tatom.global.add.u32 %r27, [%rd18], %r26;").map_err(write_err)?;
    // Skip stores on overflow but keep counter climbing so host can detect.
    writeln!(p, "\tsetp.ge.u32 %p5, %r27, %r23;").map_err(write_err)?;
    writeln!(p, "\t@%p5 bra ADVANCE;").map_err(write_err)?;

    // out_probe_idx[claimed] = tid (%r3).
    writeln!(p, "\tmul.wide.u32 %rd22, %r27, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd23, %rd16, %rd22;").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd23], %r3;").map_err(write_err)?;
    // out_build_idx[claimed] = cursor (%r25).
    writeln!(p, "\tadd.s64 %rd24, %rd17, %rd22;").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd24], %r25;").map_err(write_err)?;

    // matched[cursor] |= 1 (only when matched_ptr non-null). The bitmap is
    // a u32[ceil(build_n_rows/32)] viewed as u8[ceil(build_n_rows/4)] for
    // the atomic-OR alignment; here we use the byte-resolution version:
    // word_idx = cursor >> 5, bit = 1 << (cursor & 31). atom.global.or.b32
    // on a u32 word is the simplest correct approach.
    writeln!(p, "\tsetp.eq.u64 %p6, %rd19, 0;").map_err(write_err)?;
    writeln!(p, "\t@%p6 bra ADVANCE;").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd25, %rd19;").map_err(write_err)?;
    writeln!(p, "\tshr.u32 %r28, %r25, 5;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd26, %r28, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd27, %rd25, %rd26;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r29, %r25, 31;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r30, 1;").map_err(write_err)?;
    writeln!(p, "\tshl.b32 %r31, %r30, %r29;").map_err(write_err)?;
    writeln!(p, "\tatom.global.or.b32 %r32, [%rd27], %r31;").map_err(write_err)?;

    writeln!(p, "ADVANCE:").map_err(write_err)?;
    // cursor = next_idx[cursor]
    writeln!(p, "\tmul.wide.u32 %rd28, %r25, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd29, %rd15, %rd28;").map_err(write_err)?;
    writeln!(p, "\tld.global.u32 %r25, [%rd29];").map_err(write_err)?;
    writeln!(p, "\tbra CHAIN_LOOP;").map_err(write_err)?;

    writeln!(p, "DONE:").map_err(write_err)?;
    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;
    Ok(p)
}

/// Compile the Stage-2 outer-join second-pass kernel's PTX.
///
/// One thread per build row. If `matched[tid] == 0`, claims a slot in the
/// output buffer and writes `out_build_idx[claimed] = tid`. The host then
/// pairs each claimed entry with a NULL probe-side index via
/// `arrow::compute::take`.
///
/// ```text
/// .visible .entry bolt_hash_join_emit_unmatched_build(
///     .param .u64 matched_ptr,       // u32, ceil(build_n_rows/32)
///     .param .u64 out_build_idx_ptr, // u32, length out_capacity
///     .param .u64 out_counter_ptr,   // u32, single counter (init=0)
///     .param .u32 build_n_rows,
///     .param .u32 out_capacity
/// )
/// ```
pub fn compile_unmatched_build_kernel() -> BoltResult<String> {
    let mut p = String::new();
    writeln!(p, "{PTX_VERSION}").map_err(write_err)?;
    writeln!(p, "{PTX_TARGET}").map_err(write_err)?;
    writeln!(p, "{PTX_ADDRESS_SIZE}").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    let entry = UNMATCHED_BUILD_KERNEL_ENTRY;
    writeln!(p, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_3,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_4").map_err(write_err)?;
    writeln!(p, ")").map_err(write_err)?;
    writeln!(p, "{{").map_err(write_err)?;

    writeln!(p, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32   %r<24>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64   %rd<16>;").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    writeln!(p, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(p, "\tld.param.u32 %r4, [{entry}_param_3];").map_err(write_err)?; // build_n_rows
    writeln!(p, "\tsetp.ge.s32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(p, "\t@%p0 bra DONE;").map_err(write_err)?;

    // Load matched-bitmap word for this row.
    writeln!(p, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(p, "\tshr.u32 %r5, %r3, 5;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd1, %r5, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(p, "\tld.global.u32 %r6, [%rd2];").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r7, %r3, 31;").map_err(write_err)?;
    writeln!(p, "\tshr.u32 %r8, %r6, %r7;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r9, %r8, 1;").map_err(write_err)?;
    // matched bit set -> nothing to do.
    writeln!(p, "\tsetp.ne.u32 %p1, %r9, 0;").map_err(write_err)?;
    writeln!(p, "\t@%p1 bra DONE;").map_err(write_err)?;

    // Claim a slot.
    writeln!(p, "\tld.param.u64 %rd3, [{entry}_param_2];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r10, 1;").map_err(write_err)?;
    writeln!(p, "\tatom.global.add.u32 %r11, [%rd3], %r10;").map_err(write_err)?;
    writeln!(p, "\tld.param.u32 %r12, [{entry}_param_4];").map_err(write_err)?;
    writeln!(p, "\tsetp.ge.u32 %p2, %r11, %r12;").map_err(write_err)?;
    writeln!(p, "\t@%p2 bra DONE;").map_err(write_err)?;

    writeln!(p, "\tld.param.u64 %rd4, [{entry}_param_1];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd4, %rd4;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd5, %r11, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd6, %rd4, %rd5;").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd6], %r3;").map_err(write_err)?;

    writeln!(p, "DONE:").map_err(write_err)?;
    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;
    Ok(p)
}

/// Compile the Stage-3 CROSS JOIN kernel. Pure cartesian product — one
/// thread per output pair, no hash table. The kernel writes
/// `out_probe_idx[tid] = tid / n_build` and `out_build_idx[tid] = tid %
/// n_build`. The host guarantees `n_probe * n_build < u32::MAX` before
/// launch (see [`crate::exec::gpu_join::CROSS_JOIN_GPU_CELL_CAP`]).
///
/// ```text
/// .visible .entry bolt_cross_join(
///     .param .u64 out_probe_idx_ptr,  // u32, length total
///     .param .u64 out_build_idx_ptr,  // u32, length total
///     .param .u32 n_build,
///     .param .u32 total              // n_probe * n_build, fits u32
/// )
/// ```
pub fn compile_cross_kernel() -> BoltResult<String> {
    let mut p = String::new();
    writeln!(p, "{PTX_VERSION}").map_err(write_err)?;
    writeln!(p, "{PTX_TARGET}").map_err(write_err)?;
    writeln!(p, "{PTX_ADDRESS_SIZE}").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    let entry = CROSS_KERNEL_ENTRY;
    writeln!(p, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_2,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_3").map_err(write_err)?;
    writeln!(p, ")").map_err(write_err)?;
    writeln!(p, "{{").map_err(write_err)?;

    writeln!(p, "\t.reg .pred  %p<4>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32   %r<24>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64   %rd<16>;").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    // tid = ctaid * ntid + tid_x ; bail if tid >= total.
    writeln!(p, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(p, "\tld.param.u32 %r4, [{entry}_param_3];").map_err(write_err)?; // total
    writeln!(p, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(p, "\t@%p0 bra DONE;").map_err(write_err)?;

    // n_build
    writeln!(p, "\tld.param.u32 %r5, [{entry}_param_2];").map_err(write_err)?;
    // probe_idx = tid / n_build  ; build_idx = tid % n_build.
    writeln!(p, "\tdiv.u32 %r6, %r3, %r5;").map_err(write_err)?;
    writeln!(p, "\trem.u32 %r7, %r3, %r5;").map_err(write_err)?;

    // out_probe_idx[tid] = probe_idx.
    writeln!(p, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd1, %r3, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd2], %r6;").map_err(write_err)?;

    // out_build_idx[tid] = build_idx.
    writeln!(p, "\tld.param.u64 %rd3, [{entry}_param_1];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd4, %rd3, %rd1;").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd4], %r7;").map_err(write_err)?;

    writeln!(p, "DONE:").map_err(write_err)?;
    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;
    Ok(p)
}

/// Compile the Stage-3 AoS-layout probe kernel.
///
/// Reads each hash-table slot as a single 16-byte tuple `[key:u64,
/// head:u32, _pad:u32]` via one fused load. Compared with the Stage-2 SoA
/// probe (separate `keys_table` and `row_idx_table` loads per slot), AoS
/// halves the DRAM bytes pulled per probe step — the head sits in the same
/// 16-byte transaction as the key.
///
/// Slot address math: `slot_addr = base + slot_idx * 16`. The kernel must
/// emit `mul.lo.u32 slot_offset, slot_idx, 16` (the AoS PTX golden tests
/// this) — Stage-2's `mul.wide.u32 ..., 8` (SoA) is the inverse golden.
///
/// ABI:
///
/// ```text
/// .visible .entry bolt_hash_join_probe_aos(
///     .param .u64 probe_keys_ptr,    // i64, length n_probe
///     .param .u64 slots_ptr,         // [u8; cap * 16] AoS slots
///     .param .u64 out_probe_idx_ptr, // u32, length out_capacity
///     .param .u64 out_build_idx_ptr, // u32, length out_capacity
///     .param .u64 out_counter_ptr,   // u32, single counter
///     .param .u32 n_probe,
///     .param .u32 cap,
///     .param .u32 out_capacity
/// )
/// ```
///
/// **AoS slot layout note:** padding 4 bytes per slot raises hash-table
/// memory by a factor of 16 / 12 = 1.33× vs SoA, but the fused-load wins
/// on probe-bound workloads outweigh the extra capacity. Build kernel
/// stays SoA in Stage 3 (CAS at offset 0 is layout-agnostic; we just
/// haven't ported it).
pub fn compile_probe_aos_kernel() -> BoltResult<String> {
    let mut p = String::new();
    writeln!(p, "{PTX_VERSION}").map_err(write_err)?;
    writeln!(p, "{PTX_TARGET}").map_err(write_err)?;
    writeln!(p, "{PTX_ADDRESS_SIZE}").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    let entry = PROBE_AOS_KERNEL_ENTRY;
    writeln!(p, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_2,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_3,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_4,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_5,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_6,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_7").map_err(write_err)?;
    writeln!(p, ")").map_err(write_err)?;
    writeln!(p, "{{").map_err(write_err)?;

    writeln!(p, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64   %rd<24>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    writeln!(p, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(p, "\tld.param.u32 %r4, [{entry}_param_5];").map_err(write_err)?; // n_probe
    writeln!(p, "\tsetp.ge.s32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(p, "\t@%p0 bra DONE;").map_err(write_err)?;

    writeln!(p, "\tld.param.u32 %r5, [{entry}_param_6];").map_err(write_err)?; // cap
    writeln!(p, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;                  // mask
    writeln!(p, "\tmul.lo.u32 %r20, %r5, {MAX_PROBE_FACTOR};").map_err(write_err)?;
    writeln!(p, "\tld.param.u32 %r22, [{entry}_param_7];").map_err(write_err)?; // out_capacity

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

    // slots base pointer.
    writeln!(p, "\tld.param.u64 %rd3, [{entry}_param_1];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;

    writeln!(p, "\tmov.s64 %rl4, {EMPTY_KEY_LITERAL};").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r21, 0;").map_err(write_err)?;

    // PROBE_LOOP — AoS edition. Slot offset = slot_idx * 16. This is the
    // load-bearing instruction the PTX golden test pins.
    writeln!(p, "PROBE_LOOP:").map_err(write_err)?;
    writeln!(p, "\tadd.u32 %r21, %r21, 1;").map_err(write_err)?;
    writeln!(p, "\tsetp.gt.u32 %p3, %r21, %r20;").map_err(write_err)?;
    writeln!(p, "\t@%p3 bra DONE;").map_err(write_err)?;

    // slot_offset = slot_idx * 16. Golden: `mul.lo.u32 ..., ..., 16`.
    writeln!(p, "\tmul.lo.u32 %r10, %r8, {AOS_SLOT_BYTES};").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd5, %r10, 1;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd6, %rd3, %rd5;").map_err(write_err)?;

    // Fused load: key (8B) + head (4B) + pad (4B). PTX doesn't have a
    // 16-byte vector load on global memory at all generations; we use
    // separate loads but pack them at adjacent offsets so the L1/L2
    // line-fill brings both into the same transaction. Equivalent fused-
    // load wins without needing ld.global.v4.
    writeln!(p, "\tld.global.s64 %rl5, [%rd6];").map_err(write_err)?;       // key
    writeln!(p, "\tld.global.u32 %r9, [%rd6 + 8];").map_err(write_err)?;   // head

    // Empty slot -> no match.
    writeln!(p, "\tsetp.eq.s64 %p1, %rl5, %rl4;").map_err(write_err)?;
    writeln!(p, "\t@%p1 bra DONE;").map_err(write_err)?;

    // Key match -> claim output slot.
    writeln!(p, "\tsetp.eq.s64 %p2, %rl5, %rl0;").map_err(write_err)?;
    writeln!(p, "\t@%p2 bra MATCH;").map_err(write_err)?;

    // Advance.
    writeln!(p, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(p, "\tbra PROBE_LOOP;").map_err(write_err)?;

    writeln!(p, "MATCH:").map_err(write_err)?;
    // atom.add on counter.
    writeln!(p, "\tld.param.u64 %rd8, [{entry}_param_4];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd8, %rd8;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r11, 1;").map_err(write_err)?;
    writeln!(p, "\tatom.global.add.u32 %r12, [%rd8], %r11;").map_err(write_err)?;
    writeln!(p, "\tsetp.ge.u32 %p4, %r12, %r22;").map_err(write_err)?;
    writeln!(p, "\t@%p4 bra DONE;").map_err(write_err)?;

    // out_probe_idx[claimed] = tid (%r3)
    writeln!(p, "\tld.param.u64 %rd9, [{entry}_param_2];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd9, %rd9;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd10, %r12, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd11, %rd9, %rd10;").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd11], %r3;").map_err(write_err)?;

    // out_build_idx[claimed] = head (%r9)
    writeln!(p, "\tld.param.u64 %rd12, [{entry}_param_3];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd12, %rd12;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd13, %rd12, %rd10;").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd13], %r9;").map_err(write_err)?;

    writeln!(p, "DONE:").map_err(write_err)?;
    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;
    Ok(p)
}

/// Compile the Stage-4 AoS-layout build kernel. Symmetric counterpart to
/// [`compile_probe_aos_kernel`]: inserts `(key, row_idx)` into the same
/// `[key:u64, head:u32, _pad:u32]` 16-byte slot layout.
///
/// ## Slot address math
///
/// `slot_addr = slots + slot_idx * 16` (golden:
/// `mul.lo.u32 ..., ..., 16`). Per-slot field offsets:
/// * Key (`i64`)      — `[slot_addr + 0]`. CAS target.
/// * Head (`u32`)     — `[slot_addr + 8]`. Row index of the inserting thread.
/// * Pad (`u32`)      — `[slot_addr + 12]`. Left untouched so the AoS
///                      probe's 16-byte fused load remains well-formed.
///
/// ## Why the CAS still works
///
/// The slot starts on a 16-byte boundary (host allocates `cap * 16` as a
/// single `GpuVec<u8>` and CUDA's allocator hands back 256-byte aligned
/// pointers), so the i64 at offset 0 is naturally 8-byte aligned —
/// `atom.cas.b64` works against it exactly as in the Stage-1 SoA build.
/// The atomic-CAS is the load-bearing instruction; flipping the surrounding
/// layout from SoA to AoS does not change its semantics.
///
/// ## Memory trade-off
///
/// SoA: `cap * 12` bytes. AoS: `cap * 16` bytes — **33% larger**, in
/// exchange for halving probe-side cache-line traffic. See
/// [`AOS_SLOT_BYTES`] for the full analysis. Worth it on probe-heavy
/// workloads, which is most joins.
///
/// ## ABI
///
/// ```text
/// .visible .entry bolt_hash_join_build_aos(
///     .param .u64 keys_col_ptr,      // i64, length n_rows (encoded keys)
///     .param .u64 slots_ptr,         // u8, length cap * 16
///     .param .u32 n_rows,
///     .param .u32 cap                // power-of-two
/// )
/// ```
///
/// Grid: 1D, one thread per build row, block size [`HASH_JOIN_BLOCK_SIZE`].
/// Slot bytes MUST be zero-initialised by the host so that:
/// * The i64 key word reads as `0` initially; the host stores `EMPTY_KEY =
///   i64::MIN` (`-9223372036854775808`) into every slot key BEFORE this
///   kernel launches. (See `gpu_join::execute_inner_join_on_gpu_aos`.)
/// * The u32 head word is `0` until a winner writes its tid — but the
///   reader must NOT treat 0 as "no row" because tid 0 is a legitimate
///   row index. Readers consult the key word's sentinel instead.
pub fn compile_build_aos_kernel() -> BoltResult<String> {
    let mut p = String::new();
    writeln!(p, "{PTX_VERSION}").map_err(write_err)?;
    writeln!(p, "{PTX_TARGET}").map_err(write_err)?;
    writeln!(p, "{PTX_ADDRESS_SIZE}").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    let entry = BUILD_AOS_KERNEL_ENTRY;
    writeln!(p, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_2,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_3").map_err(write_err)?;
    writeln!(p, ")").map_err(write_err)?;
    writeln!(p, "{{").map_err(write_err)?;

    // Register file — same shape as the Stage-1 SoA build kernel; slot math
    // changes (mul-by-16, not mul-by-8) but the rest is identical.
    writeln!(p, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32   %r<24>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64   %rd<16>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    // tid = ctaid * ntid + tid.x ; bail if tid >= n_rows.
    writeln!(p, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(p, "\tld.param.u32 %r4, [{entry}_param_2];").map_err(write_err)?;
    writeln!(p, "\tsetp.ge.s32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(p, "\t@%p0 bra DONE;").map_err(write_err)?;

    // cap, mask, max_probes.
    writeln!(p, "\tld.param.u32 %r5, [{entry}_param_3];").map_err(write_err)?;
    writeln!(p, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;
    writeln!(p, "\tmul.lo.u32 %r20, %r5, {MAX_PROBE_FACTOR};").map_err(write_err)?;

    // Load this row's key.
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

    // slots base pointer.
    writeln!(p, "\tld.param.u64 %rd3, [{entry}_param_1];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;

    // EMPTY_KEY sentinel + probe counter.
    writeln!(p, "\tmov.s64 %rl4, {EMPTY_KEY_LITERAL};").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r21, 0;").map_err(write_err)?;

    // PROBE_LOOP — AoS edition. slot_addr = slots + slot_idx * 16.
    writeln!(p, "PROBE_LOOP:").map_err(write_err)?;
    writeln!(p, "\tadd.u32 %r21, %r21, 1;").map_err(write_err)?;
    writeln!(p, "\tsetp.gt.u32 %p3, %r21, %r20;").map_err(write_err)?;
    writeln!(p, "\t@%p3 bra DONE;").map_err(write_err)?;

    // slot_offset = slot_idx * 16. Golden: `mul.lo.u32 ..., ..., 16`.
    // Matches the AoS probe's offset math byte-for-byte so the two kernels
    // address the same byte positions.
    writeln!(p, "\tmul.lo.u32 %r10, %r8, {AOS_SLOT_BYTES};").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd5, %r10, 1;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd6, %rd3, %rd5;").map_err(write_err)?;

    // atom.cas.b64 against the key word at offset 0. The slot is 16-byte
    // aligned (host allocates `cap * 16` as one buffer; CUDA hands back
    // 256-byte aligned pointers), so the i64 at offset 0 is naturally
    // 8-byte aligned and the CAS is well-defined.
    writeln!(p, "\tatom.global.cas.b64 %rl5, [%rd6], %rl4, %rl0;").map_err(write_err)?;

    // prev == EMPTY -> we own this slot.
    writeln!(p, "\tsetp.eq.s64 %p1, %rl5, %rl4;").map_err(write_err)?;
    writeln!(p, "\t@%p1 bra INSERTED;").map_err(write_err)?;

    // prev == key   -> another thread already inserted this key; bail.
    // Mirrors the Stage-1 SoA build's first-writer-wins contract. The host
    // enforces uniqueness before reaching here.
    writeln!(p, "\tsetp.eq.s64 %p2, %rl5, %rl0;").map_err(write_err)?;
    writeln!(p, "\t@%p2 bra DONE;").map_err(write_err)?;

    // Collision: advance slot.
    writeln!(p, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(p, "\tbra PROBE_LOOP;").map_err(write_err)?;

    // INSERTED: write row index at slot offset 8 (the `head` word in the
    // AoS slot tuple). The `_pad` word at offset 12 is left at its
    // zero-initialised state.
    writeln!(p, "INSERTED:").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd6 + 8], %r3;").map_err(write_err)?;

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
        // Stage 1.
        assert!(is_supported_key_dtype(DataType::Int32));
        assert!(is_supported_key_dtype(DataType::Int64));
        // Stage 2: bool + float keys flow through host-side encoding.
        assert!(is_supported_key_dtype(DataType::Float32));
        assert!(is_supported_key_dtype(DataType::Float64));
        assert!(is_supported_key_dtype(DataType::Bool));
        // Stage 3: Utf8 via dict interning to i32 indices on the host.
        assert!(is_supported_key_dtype(DataType::Utf8));
    }

    #[test]
    fn key_shape_exactness() {
        // Exact-in-i64 shapes: kernel-side comparison is exact.
        assert!(KeyShape::SingleI32.is_exact_in_i64());
        assert!(KeyShape::SingleI64.is_exact_in_i64());
        assert!(KeyShape::SingleBool.is_exact_in_i64());
        assert!(KeyShape::SingleF32.is_exact_in_i64());
        assert!(KeyShape::SingleF64.is_exact_in_i64());
        assert!(KeyShape::TwoI32.is_exact_in_i64());
        // Stage-3: Utf8 keys are exact once host-interned to i32 dict idx.
        assert!(KeyShape::SingleUtf8.is_exact_in_i64());
        // Lossy shapes: host folds to i64 by splitmix, false matches possible.
        assert!(!KeyShape::TwoI64.is_exact_in_i64());
        assert!(!KeyShape::MultiI32(3).is_exact_in_i64());
        assert!(!KeyShape::MultiI32(5).is_exact_in_i64());
    }

    /// Stage-3: only the lossy-fold shapes ask for host post-verify; every
    /// other shape produces exact matches from the kernel.
    #[test]
    fn key_shape_needs_post_verify() {
        assert!(!KeyShape::SingleI32.needs_host_post_verify());
        assert!(!KeyShape::SingleI64.needs_host_post_verify());
        assert!(!KeyShape::SingleBool.needs_host_post_verify());
        assert!(!KeyShape::SingleF64.needs_host_post_verify());
        assert!(!KeyShape::TwoI32.needs_host_post_verify());
        assert!(!KeyShape::SingleUtf8.needs_host_post_verify());
        assert!(KeyShape::TwoI64.needs_host_post_verify());
        assert!(KeyShape::MultiI32(3).needs_host_post_verify());
        assert!(KeyShape::MultiI32(8).needs_host_post_verify());
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

    // ----- Stage 2 PTX-shape goldens --------------------------------------

    /// Stage-2 entry-point names are also string-stable (gpu_join.rs picks
    /// each up by name at module load).
    #[test]
    fn stage2_entry_names_are_stable() {
        assert_eq!(
            BUILD_COLLISION_KERNEL_ENTRY,
            "bolt_hash_join_build_collision"
        );
        assert_eq!(
            PROBE_COLLISION_KERNEL_ENTRY,
            "bolt_hash_join_probe_collision"
        );
        assert_eq!(
            UNMATCHED_BUILD_KERNEL_ENTRY,
            "bolt_hash_join_emit_unmatched_build"
        );
    }

    /// The collision-list build kernel uses `atom.global.exch.b32` to
    /// atomically swap the slot's head pointer with the inserting row's tid;
    /// dropping it (e.g. falling back to a non-atomic store) would race
    /// concurrent inserters into the same chain.
    #[test]
    fn build_collision_ptx_uses_atom_exch_for_chain() {
        let ptx = compile_build_collision_kernel().unwrap();
        assert!(
            ptx.contains("atom.global.exch.b32"),
            "collision build kernel must use atom.global.exch.b32 to splice into the chain; got:\n{ptx}"
        );
        // Also still uses CAS on the keys table itself.
        assert!(ptx.contains("atom.global.cas.b64"));
    }

    /// The collision-list build kernel must accept SEVEN params (two more
    /// than Stage-1 build for `head` and `next_idx`). Catches accidental
    /// parameter-list drift.
    #[test]
    fn build_collision_ptx_has_seven_params() {
        let ptx = compile_build_collision_kernel().unwrap();
        for i in 0..7 {
            let needle = format!("bolt_hash_join_build_collision_param_{i}");
            assert!(
                ptx.contains(&needle),
                "collision build kernel missing param {i}\n{ptx}"
            );
        }
        // No param 7+.
        assert!(!ptx.contains("bolt_hash_join_build_collision_param_7"));
    }

    /// The collision-list probe kernel must contain a CHAIN_LOOP label and
    /// load from `next_idx` (the linked-list traversal). Dropping either
    /// turns multi-row matches into single-row matches.
    #[test]
    fn probe_collision_ptx_walks_chain() {
        let ptx = compile_probe_collision_kernel().unwrap();
        assert!(
            ptx.contains("WALK_CHAIN:"),
            "probe collision kernel must contain a chain walk entry point;\n{ptx}"
        );
        assert!(
            ptx.contains("CHAIN_LOOP:"),
            "probe collision kernel must contain a chain loop;\n{ptx}"
        );
        // The next-pointer load: `ld.global.u32 %r25, [%rd29];` — match
        // shape, not register naming, by checking the second u32 load
        // (head[slot] first, then next_idx[cursor]).
        let n_u32_loads = ptx.matches("ld.global.u32").count();
        assert!(
            n_u32_loads >= 2,
            "probe collision kernel must read head + next_idx as u32; saw {n_u32_loads}\n{ptx}"
        );
    }

    /// The matched bitmap must be set via `atom.global.or.b32` so concurrent
    /// matching probes don't race on the same word. The orchestrator
    /// passes a null pointer for INNER, so the OR is conditionally skipped;
    /// the kernel must still EMIT the instruction in its module text.
    #[test]
    fn probe_collision_ptx_uses_atom_or_for_matched_bitmap() {
        let ptx = compile_probe_collision_kernel().unwrap();
        assert!(
            ptx.contains("atom.global.or.b32"),
            "probe collision kernel must use atom.global.or.b32 to set matched bitmap; got:\n{ptx}"
        );
    }

    /// The collision probe must guard against the host passing a null
    /// matched_ptr (INNER variant). The guard is `setp.eq.u64 ..., 0`
    /// then a branch around the OR.
    #[test]
    fn probe_collision_ptx_skips_matched_set_when_null() {
        let ptx = compile_probe_collision_kernel().unwrap();
        assert!(
            ptx.contains("setp.eq.u64"),
            "probe collision must compare matched_ptr to 0; got:\n{ptx}"
        );
    }

    /// The unmatched-build kernel reads the matched bitmap, tests the per-row
    /// bit, and stores the row index on the unmatched path.
    #[test]
    fn unmatched_build_ptx_tests_bit_and_stores_row_idx() {
        let ptx = compile_unmatched_build_kernel().unwrap();
        // Reads the bitmap word and shifts down by (tid & 31).
        assert!(ptx.contains("ld.global.u32"));
        assert!(ptx.contains("shr.u32"));
        assert!(ptx.contains("and.b32"));
        // Branches around the store when matched != 0.
        assert!(ptx.contains("setp.ne.u32"));
        // Atomic counter for output slot claim.
        assert!(ptx.contains("atom.global.add.u32"));
        // Emits exactly one u32 store (the build row index).
        let n_st = ptx.matches("st.global.u32").count();
        assert_eq!(
            n_st, 1,
            "unmatched-build kernel must write exactly one u32 per unmatched row; saw {n_st}\n{ptx}"
        );
    }

    /// The unmatched-build kernel has five params: matched_ptr,
    /// out_build_idx_ptr, out_counter_ptr, build_n_rows, out_capacity.
    #[test]
    fn unmatched_build_ptx_has_five_params() {
        let ptx = compile_unmatched_build_kernel().unwrap();
        for i in 0..5 {
            let needle = format!("bolt_hash_join_emit_unmatched_build_param_{i}");
            assert!(
                ptx.contains(&needle),
                "unmatched-build kernel missing param {i}\n{ptx}"
            );
        }
        assert!(!ptx.contains("bolt_hash_join_emit_unmatched_build_param_5"));
    }

    /// All three Stage-2 kernels must share the Stage-1 hash function so
    /// host-side replays line up byte-for-byte (matters for the multi-key
    /// composite-pack convention).
    #[test]
    fn stage2_kernels_share_stage1_hash_function() {
        let collision_build = compile_build_collision_kernel().unwrap();
        let collision_probe = compile_probe_collision_kernel().unwrap();
        let mul_literal = format!("mov.s64 %rl1, {FX_MUL};");
        assert!(collision_build.contains(&mul_literal));
        assert!(collision_probe.contains(&mul_literal));
        assert!(collision_build.contains("shr.u64 %rl3, %rl2, 32;"));
        assert!(collision_probe.contains("shr.u64 %rl3, %rl2, 32;"));
    }

    // ----- Stage 3 PTX-shape goldens --------------------------------------

    /// Stage-3 entry-point names are also string-stable.
    #[test]
    fn stage3_entry_names_are_stable() {
        assert_eq!(CROSS_KERNEL_ENTRY, "bolt_cross_join");
        assert_eq!(PROBE_AOS_KERNEL_ENTRY, "bolt_hash_join_probe_aos");
    }

    /// CROSS kernel has the expected four-parameter ABI.
    #[test]
    fn cross_ptx_has_four_params() {
        let ptx = compile_cross_kernel().unwrap();
        for i in 0..4 {
            let needle = format!("bolt_cross_join_param_{i}");
            assert!(ptx.contains(&needle), "CROSS missing param {i}\n{ptx}");
        }
        assert!(!ptx.contains("bolt_cross_join_param_4"));
    }

    /// CROSS kernel computes `probe_idx = tid / n_build` and
    /// `build_idx = tid % n_build`. Catches accidental swap of divisor.
    #[test]
    fn cross_ptx_emits_div_and_rem_against_n_build() {
        let ptx = compile_cross_kernel().unwrap();
        // div + rem on tid by n_build.
        assert!(
            ptx.contains("div.u32 %r6, %r3, %r5;"),
            "CROSS must compute probe_idx = tid / n_build\n{ptx}"
        );
        assert!(
            ptx.contains("rem.u32 %r7, %r3, %r5;"),
            "CROSS must compute build_idx = tid % n_build\n{ptx}"
        );
    }

    /// CROSS kernel writes exactly two u32 stores per thread (probe + build).
    #[test]
    fn cross_ptx_writes_two_u32_stores() {
        let ptx = compile_cross_kernel().unwrap();
        let n_st = ptx.matches("st.global.u32").count();
        assert_eq!(
            n_st, 2,
            "CROSS must emit exactly two u32 stores (probe_idx + build_idx); saw {n_st}\n{ptx}"
        );
    }

    /// CROSS kernel has the OOB guard against `total`.
    #[test]
    fn cross_ptx_has_oob_guard_against_total() {
        let ptx = compile_cross_kernel().unwrap();
        assert!(ptx.contains("setp.ge.u32 %p0, %r3, %r4;"));
        assert!(ptx.contains("DONE:"));
    }

    /// AoS probe slot-offset math is `slot_idx * 16` — pinned because the
    /// scalar multiply chooses the slot footprint. If a future refactor
    /// switches AoS to a different slot size (e.g. drops the pad) this
    /// test must be updated intentionally.
    #[test]
    fn probe_aos_ptx_uses_16byte_slot_stride() {
        let ptx = compile_probe_aos_kernel().unwrap();
        assert!(
            ptx.contains(&format!("mul.lo.u32 %r10, %r8, {AOS_SLOT_BYTES};")),
            "AoS probe must compute slot_offset = slot_idx * 16 (got AOS_SLOT_BYTES={AOS_SLOT_BYTES})\n{ptx}"
        );
        // Sanity: AOS_SLOT_BYTES is the documented 16.
        assert_eq!(AOS_SLOT_BYTES, 16);
    }

    /// AoS probe loads the head u32 at slot_offset + 8 — i.e. immediately
    /// after the i64 key in the same slot. Catches accidental drift toward
    /// SoA-style separate-array indexing.
    #[test]
    fn probe_aos_ptx_loads_head_at_plus_eight() {
        let ptx = compile_probe_aos_kernel().unwrap();
        assert!(
            ptx.contains("ld.global.u32 %r9, [%rd6 + 8];"),
            "AoS probe must load head u32 at slot_addr + 8\n{ptx}"
        );
    }

    /// AoS probe shares the Stage-1 hash function (so reads of a SoA-built
    /// table by an AoS probe would line up if we ever cross-wire them).
    #[test]
    fn probe_aos_ptx_shares_stage1_hash_function() {
        let ptx = compile_probe_aos_kernel().unwrap();
        let mul_literal = format!("mov.s64 %rl1, {FX_MUL};");
        assert!(ptx.contains(&mul_literal));
        assert!(ptx.contains("shr.u64 %rl3, %rl2, 32;"));
    }

    /// AoS probe still uses atom.global.add for the output counter so
    /// concurrent threads claim disjoint slots.
    #[test]
    fn probe_aos_ptx_uses_atom_add_for_counter() {
        let ptx = compile_probe_aos_kernel().unwrap();
        assert!(
            ptx.contains("atom.global.add.u32"),
            "AoS probe must use atom.global.add.u32 for the output counter\n{ptx}"
        );
    }

    /// AoS slot bytes is exactly 16 (8B key + 4B head + 4B pad). The
    /// padding is what lets the AoS layout fit two reads into one
    /// 128-bit-aligned line; dropping it would defeat the bandwidth win
    /// (the loads would straddle a cache line for slot_idx > 0).
    #[test]
    fn aos_slot_bytes_includes_padding() {
        assert_eq!(AOS_SLOT_BYTES, 16);
        // 8 (key) + 4 (head) + 4 (pad) = 16.
        assert_eq!(8 + 4 + 4, AOS_SLOT_BYTES);
    }
}
