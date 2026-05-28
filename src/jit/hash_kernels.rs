// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for the single-pass open-addressing GPU hash-grouping kernels.
//!
//! Two kernels back the GROUP BY executor in `crate::exec::groupby`:
//!
//! 1. [`KEYS_KERNEL_ENTRY`] — `bolt_groupby_keys`. One thread per input row.
//!    Each thread hashes its key, then performs a linear-probe `atom.cas` loop
//!    on the keys table until either (a) it inserts the key into an empty slot
//!    or (b) it finds an existing slot containing the same key. No aggregate
//!    table is touched.
//!
//! 2. [`AGG_KERNEL_ENTRY`] — `bolt_groupby_agg`. Re-runs the same hash +
//!    probe sequence against an already-populated keys table to find the slot
//!    for this row, then issues a single `atom.global.<op>.<dtype>` on that
//!    slot in the accumulator table. The kernel handles ONE aggregate at a
//!    time; the host launches it once per aggregate.
//!
//! Splitting kernels this way keeps the parameter list small and the PTX
//! template manageable: every kernel takes pointers plus `n_rows` and `k`
//! (the table size, always a power of two so the probe can mask instead of
//! mod).
//!
//! ## ABIs
//!
//! Keys kernel (`with_validity = false`, classic):
//! ```text
//! .visible .entry bolt_groupby_keys(
//!     .param .u64 group_col_ptr,   // i64 group keys, length n_rows
//!     .param .u64 keys_table_ptr,  // i64, length k, init'd to EMPTY_KEY
//!     .param .u32 n_rows,
//!     .param .u32 k                // power-of-two table size
//! )
//! ```
//!
//! Keys kernel (`with_validity = true`, Stage C extension):
//! ```text
//! .visible .entry bolt_groupby_keys(
//!     .param .u64 group_col_ptr,
//!     .param .u64 keys_table_ptr,
//!     .param .u32 n_rows,
//!     .param .u32 k,
//!     .param .u64 key_validity_ptr, // packed-bit *u32 (ceil(n_rows/32) words)
//! )
//! ```
//! When the validity bit for this row is `0` the thread bails out
//! before issuing any `atom.cas` — NULL keys are dropped, matching SQL
//! semantics where NULL is not equal to itself and therefore does not
//! group.
//!
//! Agg kernel (`with_validity = false`, classic — input dtype `T`
//! parameterises the load + atomic instruction):
//! ```text
//! .visible .entry bolt_groupby_agg(
//!     .param .u64 group_col_ptr,   // i64 group keys, length n_rows
//!     .param .u64 keys_table_ptr,  // i64, length k, fully populated
//!     .param .u64 input_col_ptr,   // T, length n_rows
//!     .param .u64 acc_table_ptr,   // T, length k, init'd to identity(op)
//!     .param .u32 n_rows,
//!     .param .u32 k
//! )
//! ```
//!
//! Agg kernel (`with_validity = true`, Stage C extension):
//! ```text
//! .visible .entry bolt_groupby_agg(
//!     .param .u64 group_col_ptr,
//!     .param .u64 keys_table_ptr,
//!     .param .u64 input_col_ptr,
//!     .param .u64 acc_table_ptr,
//!     .param .u32 n_rows,
//!     .param .u32 k,
//!     .param .u64 value_validity_ptr, // packed-bit *u32
//! )
//! ```
//! When the value-validity bit for this row is `0` the thread does NOT
//! issue its atomic — the NULL contribution is dropped per SQL aggregate
//! semantics.
//!
//! ## Packed-bit validity layout (Stage C)
//!
//! Validity is **1 bit per row**, packed 32 rows per `u32` word, with
//! little-endian bit order inside each word: bit `0` describes the first
//! row of that 32-row chunk. This matches Arrow's standard null-buffer
//! convention.
//!
//! The kernel computes `word_idx = tid >> 5`, loads
//! `word = validity_ptr[word_idx]`, then extracts bit `tid & 31` with PTX
//! `bfe.u32 dst, word, off, 1` (returns 0 or 1). A nonzero result means
//! "row is valid".
//!
//! See [`packed_validity_word_count`] for the host-side word-count
//! helper used to size the `Vec<u32>`.
//!
//! ## Sentinel
//!
//! The keys table is initialised on the host to `i64::MIN` and that value is
//! reserved as the "empty" sentinel; the executor validates no input key
//! equals this before launching.
//!
//! ## Restrictions
//!
//! `MIN` / `MAX` over floating-point inputs is implemented via a CAS loop in
//! [`crate::jit::float_atomics`]; this module only emits integer atomic
//! kernels. PTX `atom.global.min/max.fXX` does not exist for `Float32` /
//! `Float64` on sm_70, so float MIN/MAX combinations are rejected here and
//! the executor dispatches them to the float-atomics path instead.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
use crate::jit::agg_kernels::ReduceOp;
use crate::plan::logical_plan::DataType;

/// Splitmix-style multiplier used by the per-row hash. Public so tests in the
/// executor can replay the hash on the host while building the expected
/// `(key -> slot)` mapping.
///
/// This is the canonical declaration of the constant — sibling kernel modules
/// (notably [`crate::jit::valid_flag_kernels`]) redeclare the same value so
/// they can be compiled / tested standalone, but the bit pattern must match
/// the one here byte-for-byte, otherwise host-side hash replay against a
/// classic-kernel-built table will disagree with a valid-flag-built one.
// NOTE: this value must match valid_flag_kernels::FX_MUL.
pub const FX_MUL: i64 = 0x9E3779B97F4A7C15u64 as i64;

/// Entry point of the keys-only kernel emitted by [`compile_groupby_keys_kernel`].
pub const KEYS_KERNEL_ENTRY: &str = "bolt_groupby_keys";

/// Entry point of the Robin Hood variant emitted by
/// [`compile_groupby_keys_kernel_robin_hood`]. Shares the same parameter
/// list and ABI as [`KEYS_KERNEL_ENTRY`] (4 params, no validity); the
/// executor swaps one for the other based on the `BOLT_HASH_ALGO`
/// environment variable.
pub const KEYS_KERNEL_RH_ENTRY: &str = "bolt_groupby_keys_rh";

/// Entry point of the aggregate-update kernel emitted by
/// [`compile_groupby_agg_kernel`].
pub const AGG_KERNEL_ENTRY: &str = "bolt_groupby_agg";

/// Threads per block for both grouping kernels.
const BLOCK_SIZE: u32 = 256;

/// PTX `i64::MIN` literal used as the "empty slot" sentinel.
const EMPTY_KEY_LITERAL: &str = "-9223372036854775808";

/// Host-side mirror of [`EMPTY_KEY_LITERAL`]: the i64 value that the
/// classic (non-validity) keys kernel reserves to mark empty slots in
/// the open-addressing hash table.
///
/// Exposed so Tier-1 dispatchers in `crate::exec::*` can pre-flight-scan
/// their key columns: if an Int64 input legitimately contains
/// `i64::MIN`, the row's key would collide with the empty-slot marker
/// and the kernel would silently drop (or overwrite) that group. Dispatch
/// is expected to fall back to the sentinel-free valid-flag executor in
/// [`crate::exec::groupby_valid`] when this collision is detected.
///
/// Must stay byte-identical to [`EMPTY_KEY_LITERAL`] (PTX) — review C7.
pub const I64_EMPTY_SENTINEL: i64 = i64::MIN;

/// Upper bound on the linear-probe loop, expressed as a multiple of `k`.
/// At load factor < 0.5 (enforced by the executor) the expected probe length
/// is well under `log2(k)`, so a full table sweep is generous. The bound
/// exists purely to prevent a runaway kernel — if the host's load-factor
/// invariant is honoured, the bound never triggers. Mirrors the
/// `MAX_PROBE_FACTOR` constant in [`crate::jit::valid_flag_kernels`].
///
/// TODO(perf): linear probing degrades into long clusters near the load-
/// factor cap; consider robin-hood hashing (steal-from-richer reduces
/// max-probe variance) or 2-way cuckoo hashing (worst-case O(1) probes
/// at the cost of an insert-time eviction loop). Either upgrade would
/// let us raise the load-factor ceiling and shrink the table — bigger
/// L2 residency win than the per-iter probe shave.
const MAX_PROBE_FACTOR: u32 = 2;

/// Maximum probe distance allowed by the Robin Hood variant emitted by
/// [`compile_groupby_keys_kernel_robin_hood`]. Pedro Celis (1986) showed
/// that Robin Hood probing bounds the variance of probe lengths very
/// tightly; at load factor < 0.5 the expected longest probe is roughly
/// `log2(k)` and the 99th-percentile probe is typically below 16 even on
/// adversarial inputs. Threads exceeding this give up silently — same
/// "silent drop" semantics as `MAX_PROBE_FACTOR` in the linear-probe
/// kernel. Future work: surface overflow via a spill counter (mirrors
/// `valid_flag_kernels::SPILL`).
///
/// The actual iteration cap emitted by the kernel is `MAX_RH_PROBE * 2`:
/// the doubled budget exists to absorb `RH_RETRY` re-probes of the same
/// slot under contention (the CAS-with-expected swap may legitimately
/// fail and re-enter the loop for the same slot a handful of times
/// before quiescence). On overflow the row is silently dropped.
const MAX_RH_PROBE: u32 = 16;

/// Number of `u32` words required to pack a `n_rows`-row validity bitmap
/// (1 bit per row, 32 rows per word). At least one word is allocated even
/// for `n_rows == 0` so kernels can safely read word 0 unconditionally —
/// in practice the kernel's `tid >= n_rows` bail-out short-circuits before
/// touching the bitmap.
pub fn packed_validity_word_count(n_rows: usize) -> usize {
    n_rows.div_ceil(32).max(1)
}

/// Generate PTX for the keys-building kernel. The kernel writes only to the
/// keys table; the accumulator tables are untouched.
///
/// `with_key_validity = false` is the historical entry point (`KEYS_KERNEL_ENTRY`,
/// 4 params). When `true` the kernel takes an extra trailing `*u64` pointing
/// at a packed-bit validity bitmap; rows whose validity bit is `0` skip
/// the insert entirely (matches SQL semantics: NULL keys form no group).
///
/// # Encoding contract
///
/// The kernel treats every entry of `group_col_ptr` as an opaque `i64` and
/// uses bitwise equality (via `atom.cas.b64`) to compare keys. The host is
/// responsible for ENCODING the user's group-by columns into i64s before
/// upload. The currently used encodings (see `exec::groupby::pack_keys`):
///
/// * Single Int32 → sign-extended to i64.
/// * Single Int64 → identity bitcast.
/// * Single Float32 → `(f.to_bits() as u32) as i64` (bitwise-equal floats
///   group together; `-0.0` vs `+0.0` and NaN bit patterns differ).
/// * Single Float64 → `f.to_bits() as i64`.
/// * Two columns whose combined width fits in 64 bits → high 32 bits = first
///   column, low 32 bits = second column, each using the same bit
///   representation as the single-column case.
///
/// Because every supported encoding is LOSSLESS (distinct tuples ↦ distinct
/// i64), this kernel needs no awareness of the per-row column count.
pub fn compile_groupby_keys_kernel() -> BoltResult<String> {
    compile_groupby_keys_kernel_inner(false)
}

/// Stage C: variant of [`compile_groupby_keys_kernel`] that consumes a
/// per-row validity bitmap. Rows whose validity bit is `0` skip the insert.
/// See the module-level ABI documentation for the parameter list.
pub fn compile_groupby_keys_kernel_with_validity() -> BoltResult<String> {
    compile_groupby_keys_kernel_inner(true)
}

fn compile_groupby_keys_kernel_inner(with_key_validity: bool) -> BoltResult<String> {
    let mut ptx = String::new();

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {}(", KEYS_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", KEYS_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", KEYS_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_2,", KEYS_KERNEL_ENTRY).map_err(write_err)?;
    if with_key_validity {
        writeln!(ptx, "\t.param .u32 {}_param_3,", KEYS_KERNEL_ENTRY).map_err(write_err)?;
        writeln!(ptx, "\t.param .u64 {}_param_4", KEYS_KERNEL_ENTRY).map_err(write_err)?;
    } else {
        writeln!(ptx, "\t.param .u32 {}_param_3", KEYS_KERNEL_ENTRY).map_err(write_err)?;
    }
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Generous `.reg` decls — only names, not real allocations.
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // tid = ctaid.x * ntid.x + tid.x ; bail if tid >= n_rows.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u32 %r4, [{}_param_2];",
        KEYS_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // Stage C: optional packed-bit key-validity gate. Skip the insert
    // entirely for rows whose validity bit is 0 (NULL keys are dropped
    // per SQL semantics).
    if with_key_validity {
        // word_idx = tid >> 5 ; bit_off = tid & 31
        writeln!(ptx, "\tshr.u32 %r10, %r3, 5;").map_err(write_err)?;
        writeln!(ptx, "\tand.b32 %r11, %r3, 31;").map_err(write_err)?;
        // base = key_validity_ptr (param_4)
        writeln!(
            ptx,
            "\tld.param.u64 %rd10, [{}_param_4];",
            KEYS_KERNEL_ENTRY
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd10, %rd10;").map_err(write_err)?;
        // addr = base + word_idx * 4
        writeln!(ptx, "\tmul.wide.u32 %rd11, %r10, 4;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd12, %rd10, %rd11;").map_err(write_err)?;
        // Validity bitmap is a read-only input — route through the read-only cache.
        writeln!(ptx, "\tld.global.nc.u32 %r12, [%rd12];").map_err(write_err)?;
        // bit = (word >> bit_off) & 1  via bfe.u32
        writeln!(ptx, "\tbfe.u32 %r13, %r12, %r11, 1;").map_err(write_err)?;
        writeln!(ptx, "\tsetp.eq.s32 %p4, %r13, 0;").map_err(write_err)?;
        writeln!(ptx, "\t@%p4 bra DONE;").map_err(write_err)?;
    }

    // Load k and compute k-1 (mask).
    writeln!(
        ptx,
        "\tld.param.u32 %r5, [{}_param_3];",
        KEYS_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;

    // max_probes = k * MAX_PROBE_FACTOR. Computed once at kernel entry so
    // the bounded PROBE_LOOP can compare against it cheaply. The host
    // enforces load factor < 0.5, so this bound is purely defensive — if it
    // ever triggers, the thread gives up silently rather than spin forever.
    writeln!(
        ptx,
        "\tmul.lo.u32 %r20, %r5, {factor};",
        factor = MAX_PROBE_FACTOR
    )
    .map_err(write_err)?;

    // Load this thread's key value from group_col. The group-by column is a
    // read-only input (the host allocates it as a distinct GpuVec from the
    // keys_table the kernel CASes into), so route through the read-only cache.
    writeln!(
        ptx,
        "\tld.param.u64 %rd0, [{}_param_0];",
        KEYS_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd1, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.s64 %rl0, [%rd2];").map_err(write_err)?;

    // Hash: h = (key * FX_MUL) >> 32 ; then & (k-1).
    writeln!(ptx, "\tmov.s64 %rl1, {};", FX_MUL).map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s64 %rl2, %rl0, %rl1;").map_err(write_err)?;
    writeln!(ptx, "\tshr.u64 %rl3, %rl2, 32;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u64 %r7, %rl3;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r7, %r6;").map_err(write_err)?;

    // Load keys_table base ptr.
    writeln!(
        ptx,
        "\tld.param.u64 %rd3, [{}_param_1];",
        KEYS_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;

    // %rl4 = EMPTY_KEY ; %rl0 still holds the key.
    writeln!(ptx, "\tmov.s64 %rl4, {};", EMPTY_KEY_LITERAL).map_err(write_err)?;

    // Bounded-probe counter. %r21 increments once per slot examined; on
    // overflow the thread bails to DONE rather than spinning indefinitely.
    writeln!(ptx, "\tmov.u32 %r21, 0;").map_err(write_err)?;

    // Probe loop. %r8 is the current slot; loops on collision.
    writeln!(ptx, "PROBE_LOOP:").map_err(write_err)?;
    // Bound check: probe_count += 1 ; if probe_count > max_probes -> DONE.
    // Give-up-silently semantics — the success path is unchanged. Host-side
    // post-launch detection of "did every key get placed?" is a separate
    // concern (see the valid-flag SPILL path for the version that surfaces
    // the overflow to the host).
    writeln!(ptx, "\tadd.u32 %r21, %r21, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.gt.u32 %p3, %r21, %r20;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra DONE;").map_err(write_err)?;
    // addr = keys_table + slot * 8
    writeln!(ptx, "\tmul.wide.u32 %rd4, %r8, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd5, %rd3, %rd4;").map_err(write_err)?;
    // atom.cas: try EMPTY -> key. Returns previous value.
    writeln!(
        ptx,
        "\tatom.global.cas.b64 %rl5, [%rd5], %rl4, %rl0;"
    )
    .map_err(write_err)?;
    // If old == EMPTY, we inserted.
    writeln!(ptx, "\tsetp.eq.s64 %p1, %rl5, %rl4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra DONE;").map_err(write_err)?;
    // If old == key, slot already holds our group.
    writeln!(ptx, "\tsetp.eq.s64 %p2, %rl5, %rl0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra DONE;").map_err(write_err)?;
    // Collision: advance slot (linear probe, masked to k).
    writeln!(ptx, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_LOOP;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Generate PTX for the **Robin Hood** variant of the keys-building kernel.
///
/// Swap via `atom.cas`-with-expected; race-free vs the earlier
/// `atom.exch` design — the displacement step now atomically replaces
/// the occupant **conditional on it still being the value the prior CAS
/// observed**, so no third-thread mutation can sneak in between the
/// observe and the swap and produce a duplicate slot for the same group
/// key.
///
/// This is the optional opt-in alternative to [`compile_groupby_keys_kernel`].
/// It shares the same ABI (4 params: group_col_ptr, keys_table_ptr, n_rows,
/// k) and the same EMPTY-slot sentinel (`I64_EMPTY_SENTINEL = i64::MIN`),
/// so it is a drop-in replacement at the PTX module level — only the entry
/// point name differs ([`KEYS_KERNEL_RH_ENTRY`] vs [`KEYS_KERNEL_ENTRY`]).
///
/// # Algorithm
///
/// Robin Hood hashing (Pedro Celis, 1986) caps the variance of probe
/// lengths by ensuring that the entry with the smaller "probe distance"
/// (distance from its hash home) always wins a slot contest. On
/// insertion of a new key K starting from `hash(K) & mask`, at each
/// occupied slot we compare OUR probe distance against the OCCUPANT's:
///
/// * If we are POORER (our_dist > occ_dist) we displace the occupant —
///   atomically write our key into the slot, then continue probing
///   forward carrying the displaced occupant's key.
/// * If we are RICHER OR EQUAL (our_dist <= occ_dist) we advance to the
///   next slot, incrementing our probe distance.
///
/// This bounds the worst-case probe distance dramatically: at load
/// factor < 0.5 the 99th percentile probe length is typically < 16,
/// regardless of key skew, vs linear probing where adversarial inputs
/// can produce arbitrarily long clusters.
///
/// # GPU concurrency model
///
/// Each thread maintains a `cur_key` register (initially its own input
/// key) and a `cur_dist` counter (initially 0). At each slot:
///
/// 1. `atom.cas.b64(slot, EMPTY, cur_key)` — try to claim the slot if
///    it is empty.
/// 2. If `old == EMPTY` we placed the key, done.
/// 3. If `old == cur_key` the slot already holds our group, done.
/// 4. Otherwise compute the OCCUPANT's probe distance:
///    `occ_dist = (slot - (hash(old) & mask)) & mask`.
/// 5. If `occ_dist >= cur_dist` (occupant richer-or-equal): advance.
/// 6. If `occ_dist <  cur_dist` (occupant poorer): we displace using a
///    second CAS conditional on the slot still equalling the previously
///    observed occupant:
///    `actual = atom.cas.b64(slot, observed_occupant, cur_key)`. If
///    `actual == observed_occupant` the swap landed: continue probing
///    with `cur_key := observed_occupant`,
///    `cur_dist := occ_dist + 1`. If not, the slot mutated under us — we
///    re-probe THIS slot from the top of the loop without changing
///    `cur_key` (the slot now holds either a new occupant we still need
///    to compare against, or, in the swap-back case, our own key).
///
/// # Concurrency notes
///
/// The two-CAS sequence (empty-claim CAS + expected-occupant swap CAS)
/// is the load-bearing fix for the swap race the earlier `atom.exch`
/// design exhibited: `atom.exch` carried no "expected" parameter, so
/// the value it returned under contention could differ from the one
/// the prior CAS observed, transiently producing two slots claiming
/// the same group key. The CAS-with-expected variant is race-free in
/// that single step.
///
/// The kernel remains **opt-in via `BOLT_HASH_ALGO=robin_hood`** —
/// the linear-probe kernel is still the default. PTX-shape tests below
/// exercise the emitter; end-to-end GPU correctness validation against
/// adversarial inputs is left as follow-up work.
///
/// # Probe cap
///
/// A thread that exceeds [`MAX_RH_PROBE`] slot examinations gives up
/// silently (no atomic update issued). Same defensive-bound contract as
/// the linear-probe kernel — at load factor < 0.5 it should never
/// trigger; if it does, the row's group is dropped.
///
/// An additional cap of `MAX_RH_PROBE * 2` total `RH_PROBE_LOOP`
/// iterations guards against the new `RH_RETRY` path spinning forever
/// under pathological contention: each retry consumes a unit of the
/// total iteration budget. On overflow we drop the row silently, same
/// as the linear path.
pub fn compile_groupby_keys_kernel_robin_hood() -> BoltResult<String> {
    let mut ptx = String::new();
    let entry = KEYS_KERNEL_RH_ENTRY;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Same 4-param ABI as the classic keys kernel.
    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_2,").map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {entry}_param_3").map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Generous register decls; only declared names, not real allocations.
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rl<24>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // tid = ctaid.x * ntid.x + tid.x ; bail if tid >= n_rows.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{entry}_param_2];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // Load k and compute mask = k - 1.
    writeln!(ptx, "\tld.param.u32 %r5, [{entry}_param_3];").map_err(write_err)?;
    writeln!(ptx, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;

    // Load this thread's key value from group_col into %rl0 (= cur_key).
    writeln!(ptx, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd1, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rl0, [%rd2];").map_err(write_err)?;

    // Hash cur_key: h = ((cur_key * FX_MUL) >> 32) & mask.
    writeln!(ptx, "\tmov.s64 %rl1, {};", FX_MUL).map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s64 %rl2, %rl0, %rl1;").map_err(write_err)?;
    writeln!(ptx, "\tshr.u64 %rl3, %rl2, 32;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u64 %r7, %rl3;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r7, %r6;").map_err(write_err)?;

    // Load keys_table base ptr.
    writeln!(ptx, "\tld.param.u64 %rd3, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;

    // EMPTY_KEY constant.
    writeln!(ptx, "\tmov.s64 %rl4, {};", EMPTY_KEY_LITERAL).map_err(write_err)?;

    // %r9  = cur_dist (starts at 0)
    // %r10 = probe_count (bounded-probe defensive counter; counts EVERY
    //        iteration of RH_PROBE_LOOP, including RH_RETRY re-probes
    //        of the same slot under contention).
    // %r11 = MAX_RH_PROBE * 2 constant. We use 2x the linear-probe cap
    //        because RH_RETRY can legitimately re-probe the same slot
    //        multiple times under contention; doubling the budget keeps
    //        the silent-failure semantics consistent with the linear
    //        path (rare under load factor < 0.5) while preventing the
    //        retry path from spinning forever.
    writeln!(ptx, "\tmov.u32 %r9, 0;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r10, 0;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r11, {};", MAX_RH_PROBE * 2).map_err(write_err)?;

    // ------------------------------------------------------------------
    // Robin Hood probe loop.
    // Loop registers:
    //   %rl0  = cur_key  (key currently being placed; may be original or displaced)
    //   %r8   = cur_slot (masked)
    //   %r9   = cur_dist (probe distance from cur_key's hash to cur_slot)
    //   %r10  = probe_count (bounded-probe counter)
    // ------------------------------------------------------------------
    writeln!(ptx, "RH_PROBE_LOOP:").map_err(write_err)?;
    // Bounded-probe defensive check: silent-drop on overflow.
    writeln!(ptx, "\tadd.u32 %r10, %r10, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.gt.u32 %p3, %r10, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra DONE;").map_err(write_err)?;

    // addr = keys_table + cur_slot * 8
    writeln!(ptx, "\tmul.wide.u32 %rd4, %r8, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd5, %rd3, %rd4;").map_err(write_err)?;

    // Try CAS: EMPTY -> cur_key. Returns previous value into %rl5.
    writeln!(
        ptx,
        "\tatom.global.cas.b64 %rl5, [%rd5], %rl4, %rl0;"
    )
    .map_err(write_err)?;

    // If old == EMPTY, we placed cur_key successfully; done.
    writeln!(ptx, "\tsetp.eq.s64 %p1, %rl5, %rl4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra DONE;").map_err(write_err)?;

    // If old == cur_key, the slot already holds our (or the displaced)
    // group; done. Note: when carrying a DISPLACED key, finding it
    // already present means our chase ends — the displaced key is
    // already in the table further along (the natural deduplication
    // path for the common low-contention case).
    writeln!(ptx, "\tsetp.eq.s64 %p2, %rl5, %rl0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra DONE;").map_err(write_err)?;

    // Slot was occupied by a different key — compute the occupant's
    // probe distance:
    //   occ_home = (occupant * FX_MUL) >> 32 & mask
    //   occ_dist = (cur_slot - occ_home) & mask
    //
    // %rl5 holds the occupant key.
    writeln!(ptx, "\tmul.lo.s64 %rl6, %rl5, %rl1;").map_err(write_err)?;
    writeln!(ptx, "\tshr.u64 %rl7, %rl6, 32;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u64 %r12, %rl7;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r13, %r12, %r6;").map_err(write_err)?;
    // occ_dist = (cur_slot - occ_home) & mask  -- mask handles wrap-around.
    writeln!(ptx, "\tsub.s32 %r14, %r8, %r13;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r14, %r14, %r6;").map_err(write_err)?;

    // Compare: are WE richer (cur_dist <= occ_dist) or POORER (cur_dist > occ_dist)?
    // If cur_dist <= occ_dist : occupant is richer-or-equal → advance.
    // If cur_dist >  occ_dist : occupant is poorer → SWAP.
    writeln!(ptx, "\tsetp.gt.u32 %p4, %r9, %r14;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra RH_SWAP;").map_err(write_err)?;

    // Richer-or-equal path: advance cur_slot, cur_dist += 1.
    writeln!(ptx, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r9, %r9, 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra RH_PROBE_LOOP;").map_err(write_err)?;

    // ------------------------------------------------------------------
    // Swap branch: atomically replace the slot's contents with cur_key
    // ONLY IF the slot still holds the occupant value %rl5 we observed
    // via the preceding `atom.cas`. If the CAS-with-expected succeeds
    // we pick up the displaced occupant as the new cur_key (it's just
    // the value we passed as `expected`, i.e. %rl5) and advance.
    //
    // If the CAS-with-expected FAILS the slot mutated under us
    // (concurrent thread inserted/displaced something else). We branch
    // to RH_RETRY, which re-enters the main loop at THIS slot without
    // touching cur_key — the next iteration will observe whatever the
    // slot currently holds and re-decide claim/swap/advance.
    //
    // This is the load-bearing fix vs the earlier `atom.exch` design:
    // exch had no "expected" parameter so under contention the value
    // it returned could differ from the one observed by the prior CAS,
    // producing duplicate slots for the same group key. CAS-with-
    // expected makes the swap race-free in that single step.
    // ------------------------------------------------------------------
    writeln!(ptx, "RH_SWAP:").map_err(write_err)?;
    // atom.cas(slot, expected=%rl5 occupant, new=%rl0 cur_key) → %rl8 actual
    writeln!(
        ptx,
        "\tatom.global.cas.b64 %rl8, [%rd5], %rl5, %rl0;"
    )
    .map_err(write_err)?;
    // If %rl8 != %rl5 (CAS failed), slot changed under us → RH_RETRY.
    writeln!(ptx, "\tsetp.eq.s64 %p5, %rl8, %rl5;").map_err(write_err)?;
    writeln!(ptx, "\t@!%p5 bra RH_RETRY;").map_err(write_err)?;

    // Swap succeeded. cur_key := the occupant we just displaced (we
    // already have it in %rl5; using it directly avoids a second load).
    writeln!(ptx, "\tmov.b64 %rl0, %rl5;").map_err(write_err)?;
    // cur_dist := occ_dist + 1; advance slot.
    writeln!(ptx, "\tadd.u32 %r9, %r14, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(ptx, "\tbra RH_PROBE_LOOP;").map_err(write_err)?;

    // ------------------------------------------------------------------
    // Retry branch: the CAS-with-expected failed because a sibling
    // thread mutated the slot between our observe and our swap. Do NOT
    // change cur_key — re-enter the main loop at the SAME slot. The
    // next iteration will read whatever the slot now holds (possibly a
    // swap-back of our own key, possibly someone else's key) and
    // re-decide. The bounded-probe counter at the top of the loop
    // limits how many times this can happen.
    // ------------------------------------------------------------------
    writeln!(ptx, "RH_RETRY:").map_err(write_err)?;
    writeln!(ptx, "\tbra RH_PROBE_LOOP;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Runtime dispatch helper for the keys-build kernel. Reads the
/// `BOLT_HASH_ALGO` environment variable on first call and routes to
/// either the linear-probe (default) or Robin Hood emitter.
///
/// Accepted values (case-insensitive):
///   * unset, empty, or `linear`  → [`compile_groupby_keys_kernel`]
///     (returns the classic [`KEYS_KERNEL_ENTRY`] entry point).
///   * `robin_hood` or `rh`       → [`compile_groupby_keys_kernel_robin_hood`]
///     (returns the [`KEYS_KERNEL_RH_ENTRY`] entry point).
///
/// Returns a `(ptx, entry_name)` pair so the executor knows which
/// `module.function(...)` name to look up. Unknown values fall back to
/// the linear path silently — Tier-1 dispatch must remain robust to
/// typo'd or stale env values.
///
/// This helper exists so the executor can flip kernels without a
/// recompile; it is intentionally a per-launch lookup (`std::env::var`
/// is cheap), keeping the opt-in surgical. Promoting the Robin Hood
/// kernel to the default is intentionally deferred to a follow-up
/// task — it still wants end-to-end GPU validation against adversarial
/// inputs before flipping the default here.
pub fn compile_groupby_keys_kernel_dispatched() -> BoltResult<(String, &'static str)> {
    let algo = std::env::var("BOLT_HASH_ALGO").unwrap_or_default();
    let algo_lc = algo.to_ascii_lowercase();
    if algo_lc == "robin_hood" || algo_lc == "rh" {
        let ptx = compile_groupby_keys_kernel_robin_hood()?;
        Ok((ptx, KEYS_KERNEL_RH_ENTRY))
    } else {
        // Default: linear probing (includes the "linear", "", and unknown
        // cases — robust to typos).
        let ptx = compile_groupby_keys_kernel()?;
        Ok((ptx, KEYS_KERNEL_ENTRY))
    }
}

/// Generate PTX for an aggregate-update kernel parameterised over `op` +
/// `input_dtype`. Assumes the keys table referenced by `keys_table_ptr` is
/// already fully populated by a prior [`compile_groupby_keys_kernel`] launch.
///
/// ## Cross-kernel synchronisation contract
///
/// The keys-kernel writes to `keys_table_ptr` and the agg-kernel reads from
/// it. The two kernels cooperate via the table but DO NOT synchronise
/// internally — the agg-kernel's probe loop assumes every slot that will
/// ever be written has already been written. The host is responsible for
/// enforcing that ordering, which means one of the following MUST hold
/// between the two launches:
///
/// * Both launches go on the same CUDA stream (CUDA's default in-order
///   semantics make this a memory-ordering no-op — the agg kernel's loads
///   are guaranteed to observe every store from the keys kernel), OR
/// * The host explicitly calls `cuStreamSynchronize` (or an equivalent
///   event-wait) between the two launches.
///
/// Cross-stream launches WITHOUT an explicit synchronise are a bug: the agg
/// kernel will see a partially-populated keys table, miss its slot during
/// linear probe, and either spin to the new bounded-probe limit and give up
/// silently OR (depending on probe path) atomically update the wrong slot.
/// Neither outcome is recoverable post-hoc. This invariant previously lived
/// only in scattered executor docstrings; it is restated here because the
/// agg-kernel PTX itself bakes it in as a pre-condition.
///
/// ## Probe-loop bound
///
/// The non-mutating probe loop here mirrors the bounded-probe pattern in
/// [`compile_groupby_keys_kernel`]: a per-thread counter increments once per
/// slot examined and the thread gives up silently (no atomic update issued)
/// after `MAX_PROBE_FACTOR * k` slots. Without this bound a thread whose key
/// is absent from the table — which can happen if the cross-kernel ordering
/// contract above is violated — would spin forever and hang the streaming
/// multiprocessor. Silent-drop matches the keys kernel's behaviour: the
/// kernel ABI is unchanged and the host's load-factor invariant ensures the
/// bound never triggers on a correctly-sequenced launch.
pub fn compile_groupby_agg_kernel(
    op: ReduceOp,
    input_dtype: DataType,
) -> BoltResult<String> {
    compile_groupby_agg_kernel_inner(op, input_dtype, false)
}

/// Stage C: variant of [`compile_groupby_agg_kernel`] that consumes a per-row
/// value-validity bitmap (packed-bit, `u32` words). Rows whose bit is `0`
/// skip the atomic — matches SQL semantics where NULL input rows do not
/// contribute to the aggregate.
pub fn compile_groupby_agg_kernel_with_validity(
    op: ReduceOp,
    input_dtype: DataType,
) -> BoltResult<String> {
    compile_groupby_agg_kernel_inner(op, input_dtype, true)
}

fn compile_groupby_agg_kernel_inner(
    op: ReduceOp,
    input_dtype: DataType,
    with_value_validity: bool,
) -> BoltResult<String> {
    // Reject unsupported (op, dtype) combinations up front with explicit errors.
    let atomic = atomic_for(op, input_dtype)?;

    let (load_suffix, reg_class) = ptx_type_info(input_dtype)?;
    let elem_bytes = input_dtype.byte_width().ok_or_else(|| {
        BoltError::Other(format!(
            "hash_kernels: variable-width dtype {:?} not supported",
            input_dtype
        ))
    })?;

    let mut ptx = String::new();

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {}(", AGG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", AGG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", AGG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_2,", AGG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_3,", AGG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_4,", AGG_KERNEL_ENTRY).map_err(write_err)?;
    if with_value_validity {
        writeln!(ptx, "\t.param .u32 {}_param_5,", AGG_KERNEL_ENTRY).map_err(write_err)?;
        writeln!(ptx, "\t.param .u64 {}_param_6", AGG_KERNEL_ENTRY).map_err(write_err)?;
    } else {
        writeln!(ptx, "\t.param .u32 {}_param_5", AGG_KERNEL_ENTRY).map_err(write_err)?;
    }
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    // A typed value register for the input column load. For Int64 inputs the
    // input value lives in the `%rl` namespace already used by the key path —
    // emitting a separate value register keeps the PTX uniform across dtypes.
    writeln!(
        ptx,
        "\t.reg .{ty}   %{rc}<4>;",
        ty = reg_decl_ty(input_dtype)?,
        rc = reg_class
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // tid = ctaid.x * ntid.x + tid.x ; bail if tid >= n_rows.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u32 %r4, [{}_param_4];",
        AGG_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // Stage C: optional packed-bit value-validity gate. Skip the atomic
    // update for this row when the validity bit is 0 (SQL: NULL inputs
    // do not contribute to SUM / MIN / MAX / COUNT(col) / AVG).
    if with_value_validity {
        // word_idx = tid >> 5 ; bit_off = tid & 31
        writeln!(ptx, "\tshr.u32 %r14, %r3, 5;").map_err(write_err)?;
        writeln!(ptx, "\tand.b32 %r15, %r3, 31;").map_err(write_err)?;
        // base = value_validity_ptr (param_6)
        writeln!(
            ptx,
            "\tld.param.u64 %rd16, [{}_param_6];",
            AGG_KERNEL_ENTRY
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd16, %rd16;").map_err(write_err)?;
        writeln!(ptx, "\tmul.wide.u32 %rd17, %r14, 4;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd18, %rd16, %rd17;").map_err(write_err)?;
        // value-validity bitmap is a read-only input — route through .nc.
        writeln!(ptx, "\tld.global.nc.u32 %r16, [%rd18];").map_err(write_err)?;
        writeln!(ptx, "\tbfe.u32 %r17, %r16, %r15, 1;").map_err(write_err)?;
        writeln!(ptx, "\tsetp.eq.s32 %p4, %r17, 0;").map_err(write_err)?;
        writeln!(ptx, "\t@%p4 bra DONE;").map_err(write_err)?;
    }

    // k and mask = k - 1.
    writeln!(
        ptx,
        "\tld.param.u32 %r5, [{}_param_5];",
        AGG_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;

    // max_probes = k * MAX_PROBE_FACTOR. Computed once at kernel entry so the
    // bounded PROBE_LOOP can compare against it cheaply. Mirrors the
    // identically-named computation in `compile_groupby_keys_kernel`; without
    // this bound a thread whose key is absent (which can only happen on a
    // partially-populated keys table — see the cross-kernel synchronisation
    // contract above) would spin forever and hang the SM.
    writeln!(
        ptx,
        "\tmul.lo.u32 %r20, %r5, {factor};",
        factor = MAX_PROBE_FACTOR
    )
    .map_err(write_err)?;

    // Load the key for this row. Key column is a read-only input.
    writeln!(
        ptx,
        "\tld.param.u64 %rd0, [{}_param_0];",
        AGG_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd1, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.s64 %rl0, [%rd2];").map_err(write_err)?;

    // Hash.
    writeln!(ptx, "\tmov.s64 %rl1, {};", FX_MUL).map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s64 %rl2, %rl0, %rl1;").map_err(write_err)?;
    writeln!(ptx, "\tshr.u64 %rl3, %rl2, 32;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u64 %r7, %rl3;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r7, %r6;").map_err(write_err)?;

    // Keys table base.
    writeln!(
        ptx,
        "\tld.param.u64 %rd3, [{}_param_1];",
        AGG_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;

    // Bounded-probe counter. %r21 increments once per slot examined; on
    // overflow the thread bails to DONE rather than spinning indefinitely.
    // Matches the same idiom in `compile_groupby_keys_kernel`.
    writeln!(ptx, "\tmov.u32 %r21, 0;").map_err(write_err)?;

    // Probe loop — non-mutating; the keys table is read-only here. We just
    // walk slots until we find the one whose key matches ours. The host's
    // cross-kernel synchronisation contract (see the doc comment above)
    // guarantees a matching slot exists; the bounded counter below is the
    // defensive fallback if that contract is violated.
    writeln!(ptx, "PROBE_LOOP:").map_err(write_err)?;
    // Bound check: probe_count += 1 ; if probe_count > max_probes -> DONE.
    // Give-up-silently semantics — no atomic update is issued for this row.
    // Same shape as the keys kernel's bound (setp.gt.u32 against %r20).
    writeln!(ptx, "\tadd.u32 %r21, %r21, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.gt.u32 %p3, %r21, %r20;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd4, %r8, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd5, %rd3, %rd4;").map_err(write_err)?;
    // The keys-table is non-mutating from this kernel's POV (it was populated
    // by the preceding keys kernel and is only READ here — no atom.cas, no
    // st.global to %rd3). Route through the read-only cache.
    writeln!(ptx, "\tld.global.nc.s64 %rl5, [%rd5];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s64 %p1, %rl5, %rl0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra FOUND;").map_err(write_err)?;
    // Otherwise advance.
    writeln!(ptx, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_LOOP;").map_err(write_err)?;
    writeln!(ptx, "FOUND:").map_err(write_err)?;

    // Load the input column value for this row (read-only input column).
    writeln!(
        ptx,
        "\tld.param.u64 %rd6, [{}_param_2];",
        AGG_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd6, %rd6;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd7, %r3, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd8, %rd6, %rd7;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.global.nc.{ld} %{rc}0, [%rd8];",
        ld = load_suffix,
        rc = reg_class
    )
    .map_err(write_err)?;

    // Compute the accumulator slot address (acc_table + slot * elem_bytes).
    writeln!(
        ptx,
        "\tld.param.u64 %rd9, [{}_param_3];",
        AGG_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd9, %rd9;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd10, %r8, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd11, %rd9, %rd10;").map_err(write_err)?;

    // Atomic update. PTX `atom` returns the old value into a destination
    // register; we don't need it, but the form requires one — reuse the
    // value register class with a fresh index.
    writeln!(
        ptx,
        "\t{atomic} %{rc}1, [%rd11], %{rc}0;",
        atomic = atomic,
        rc = reg_class
    )
    .map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Entry point of the fused multi-aggregate kernel emitted by
/// [`compile_groupby_agg_kernel_multi`].
pub const AGG_KERNEL_MULTI_ENTRY: &str = "bolt_groupby_agg_multi";

/// One aggregate's contribution to the fused multi-aggregate kernel.
///
/// The fused kernel hashes the group key exactly once, walks the probe loop
/// exactly once, and then issues `N` independent `atom.global.<op>.<dtype>`
/// instructions — one per `AggSpec` in the slice — against `N` per-aggregate
/// input columns and `N` per-aggregate accumulator tables.
///
/// Each spec contributes its own `(input_ptr, acc_ptr)` pointer pair through
/// the kernel's parameter list (see the ABI in
/// [`compile_groupby_agg_kernel_multi`]'s doc comment).
#[derive(Debug, Clone, Copy)]
pub struct AggSpec {
    /// Reduction operator (Sum / Min / Max / Count). MIN/MAX over floats is
    /// rejected by [`atomic_for`] just as in the single-agg path — Tier-1
    /// dispatch is expected to route those through `float_atomics`.
    pub op: ReduceOp,
    /// Element type of both the input column and the accumulator slot for
    /// this aggregate. Different specs may use different dtypes (e.g.
    /// `SUM(i32)` + `COUNT(*) -> i64` + `MIN(f64)`).
    pub input_dtype: DataType,
}

/// Generate PTX for the **fused multi-aggregate** kernel.
///
/// This is the multi-agg companion to [`compile_groupby_agg_kernel`]: where
/// the single-agg kernel re-hashes the group key (and re-walks the probe
/// chain) on every invocation, this kernel hashes ONCE and then issues `N`
/// atomic updates back-to-back. For `SELECT SUM(a), COUNT(*), MIN(b)
/// FROM t GROUP BY k` the dispatcher previously emitted three kernels each
/// repeating the hash + probe; this folds them into one launch.
///
/// Pattern lifted from
/// [`crate::jit::partition_reduce_kernel_multi::compile_partition_reduce_kernel_multi`]
/// (Tier-2's per-partition shared-mem reducer), adapted to Tier-1's global
/// open-addressing layout.
///
/// # ABI
///
/// `N = specs.len()`. Parameter ordering (all `.u64` except where noted):
///
/// ```text
/// .visible .entry bolt_groupby_agg_multi(
///     .param .u64 group_col_ptr,        // i64 group keys, length n_rows
///     .param .u64 keys_table_ptr,       // i64, length k, fully populated
///     .param .u64 input_col_ptr_0,
///     ...
///     .param .u64 input_col_ptr_{N-1},
///     .param .u64 acc_table_ptr_0,
///     ...
///     .param .u64 acc_table_ptr_{N-1},
///     .param .u32 n_rows,
///     .param .u32 k,                    // power-of-two table size
/// )
/// ```
///
/// Spec `j`'s input column elements are `specs[j].input_dtype.byte_width()`
/// bytes wide, matching its accumulator table's slot width. The host must
/// upload each input + accumulator buffer in spec order.
///
/// # Pre-conditions
///
/// Same cross-kernel synchronisation contract as the single-agg variant —
/// `keys_table_ptr` must reference a fully-populated keys table produced by
/// a prior [`compile_groupby_keys_kernel`] launch on the same stream (or
/// the host must explicitly synchronise between launches).
///
/// # Restrictions
///
/// * `specs` must be non-empty.
/// * Each spec is validated through [`atomic_for`] and [`ptx_type_info`],
///   so float MIN/MAX is rejected here — Tier-1 dispatch must keep float
///   MIN/MAX out of the fused path (route those through `float_atomics`
///   per-aggregate just as today). When the dispatch sees a fusable
///   homogeneous-key spec set with no float MIN/MAX, this is a strict win;
///   when it doesn't, the per-agg path keeps working.
///
/// # Note on validity
///
/// This first cut does NOT emit the Stage-C `_with_validity` gate. Adding
/// per-spec validity bitmaps multiplies the parameter list and forces a
/// per-spec bit-extract before each atomic; that's a follow-up. The Tier-1
/// dispatcher should keep the per-agg `_with_validity` path for queries
/// where ANY agg-input column has validity; fuse only the no-validity case.
pub fn compile_groupby_agg_kernel_multi(specs: &[AggSpec]) -> BoltResult<String> {
    if specs.is_empty() {
        return Err(BoltError::Other(
            "compile_groupby_agg_kernel_multi: specs must be non-empty"
                .into(),
        ));
    }
    let n = specs.len();

    // Validate every spec up front; collect per-spec PTX type info so the
    // body loop is allocation-free.
    struct PerSpec {
        atomic: &'static str,
        load_suffix: &'static str,
        reg_class: &'static str,
        reg_decl_ty: &'static str,
        elem_bytes: usize,
    }
    let mut per: Vec<PerSpec> = Vec::with_capacity(n);
    for s in specs {
        let atomic = atomic_for(s.op, s.input_dtype)?;
        let (load_suffix, reg_class) = ptx_type_info(s.input_dtype)?;
        let reg_decl_ty_s = reg_decl_ty(s.input_dtype)?;
        let elem_bytes = s.input_dtype.byte_width().ok_or_else(|| {
            BoltError::Other(format!(
                "hash_kernels: variable-width dtype {:?} not supported in fused multi-agg",
                s.input_dtype
            ))
        })?;
        per.push(PerSpec {
            atomic,
            load_suffix,
            reg_class,
            reg_decl_ty: reg_decl_ty_s,
            elem_bytes,
        });
    }

    let entry = AGG_KERNEL_MULTI_ENTRY;
    let mut ptx = String::new();

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Param layout (see ABI in doc comment):
    //   p0  = group_col_ptr
    //   p1  = keys_table_ptr
    //   p[2 .. 2+n)            = input_col_ptr_j
    //   p[2+n .. 2+2n)         = acc_table_ptr_j
    //   p[2+2n]                = n_rows (u32)
    //   p[3+2n]                = k      (u32)
    let total_u64_params = 2 + 2 * n;
    let n_rows_param = total_u64_params;
    let k_param = total_u64_params + 1;
    let total_params = total_u64_params + 2;

    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    for p in 0..total_params {
        let trailing = if p == total_params - 1 { "" } else { "," };
        let kind = if p < total_u64_params { "u64" } else { "u32" };
        writeln!(ptx, "\t.param .{kind} {entry}_param_{p}{trailing}")
            .map_err(write_err)?;
    }
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Register pool. Per-spec value registers are emitted as separate
    // `.reg` classes ("vr", "vl", "vf", "vd") so different dtypes don't
    // alias. Each class is sized large enough for the worst case of all
    // N specs sharing that class (4 vals per spec keeps headroom).
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<64>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rl<16>;").map_err(write_err)?;
    // Emit one `.reg` declaration per dtype actually used by any spec, so
    // we don't redeclare and don't waste names. Index `j` of a class is
    // assigned by the spec's position within that class's per-dtype group;
    // a fresh slot is allocated below when writing the atomic.
    let mut declared_classes: Vec<&'static str> = Vec::new();
    for p in &per {
        if !declared_classes.contains(&p.reg_class) {
            // Width per spec in this class: 4 names (loaded value + atomic
            // return + 2 spare). With at most n specs sharing a class the
            // declared range is 4*n which is a tight upper bound.
            writeln!(
                ptx,
                "\t.reg .{ty}   %{rc}<{w}>;",
                ty = p.reg_decl_ty,
                rc = p.reg_class,
                w = 4 * n.max(1),
            )
            .map_err(write_err)?;
            declared_classes.push(p.reg_class);
        }
    }
    writeln!(ptx).map_err(write_err)?;

    // tid = ctaid.x * ntid.x + tid.x ; bail if tid >= n_rows.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u32 %r4, [{entry}_param_{n_rows_param}];"
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // k and mask = k - 1.
    writeln!(
        ptx,
        "\tld.param.u32 %r5, [{entry}_param_{k_param}];"
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tsub.s32 %r6, %r5, 1;").map_err(write_err)?;

    // max_probes = k * MAX_PROBE_FACTOR.
    writeln!(
        ptx,
        "\tmul.lo.u32 %r20, %r5, {factor};",
        factor = MAX_PROBE_FACTOR
    )
    .map_err(write_err)?;

    // Load the key for this row (param 0 = group_col_ptr — read-only input).
    writeln!(ptx, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd1, %r3, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.s64 %rl0, [%rd2];").map_err(write_err)?;

    // Hash — computed exactly ONCE for the entire fused kernel. This is the
    // whole point: subsequent atomic updates reuse the resolved slot
    // without re-hashing.
    writeln!(ptx, "\tmov.s64 %rl1, {};", FX_MUL).map_err(write_err)?;
    writeln!(ptx, "\tmul.lo.s64 %rl2, %rl0, %rl1;").map_err(write_err)?;
    writeln!(ptx, "\tshr.u64 %rl3, %rl2, 32;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u64 %r7, %rl3;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r7, %r6;").map_err(write_err)?;

    // Keys table base (param 1).
    writeln!(ptx, "\tld.param.u64 %rd3, [{entry}_param_1];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;

    // Bounded-probe counter (matches single-agg `compile_groupby_agg_kernel`).
    writeln!(ptx, "\tmov.u32 %r21, 0;").map_err(write_err)?;

    // Probe loop — non-mutating; walks slots until the matching key is found
    // OR the bound trips (silent-drop, identical to the single-agg variant).
    writeln!(ptx, "PROBE_LOOP:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r21, %r21, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.gt.u32 %p3, %r21, %r20;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra DONE;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd4, %r8, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd5, %rd3, %rd4;").map_err(write_err)?;
    // Keys-table probe is non-mutating here — populated by the prior keys
    // kernel and READ from this multi-agg kernel. Route through .nc.
    writeln!(ptx, "\tld.global.nc.s64 %rl5, [%rd5];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s64 %p1, %rl5, %rl0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra FOUND;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r8, %r8, %r6;").map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_LOOP;").map_err(write_err)?;
    writeln!(ptx, "FOUND:").map_err(write_err)?;

    // ----------------- Phase: emit N atomic updates ----------------------
    //
    // Each spec j contributes:
    //   1. load input_j[tid] into a typed value register
    //   2. compute acc_j + slot * elem_bytes_j
    //   3. atom.global.<op_j>.<dtype_j> at that address
    //
    // Register-name bookkeeping: per dtype-class we hand out two fresh slot
    // indices per spec (one for the loaded value, one for the atomic's
    // ignored return register). The slot offset is tracked in a small
    // per-class counter so the names never collide across specs sharing a
    // class.
    let mut class_slot_counter: Vec<(&str, usize)> = Vec::new();
    fn take_two_slots<'a>(
        counter: &mut Vec<(&'a str, usize)>,
        rc: &'a str,
    ) -> (usize, usize) {
        if let Some(entry) = counter.iter_mut().find(|(c, _)| *c == rc) {
            let base = entry.1;
            entry.1 = base + 2;
            (base, base + 1)
        } else {
            counter.push((rc, 2));
            (0, 1)
        }
    }
    for (j, p) in per.iter().enumerate() {
        let input_param = 2 + j;
        let acc_param = 2 + n + j;
        // Scratch %rd index pool: reuse %rd10..%rd13 per j — each spec owns
        // them only between its load and its atom; nothing carries across.
        // Load input_j[tid].
        writeln!(
            ptx,
            "\tld.param.u64 %rd10, [{entry}_param_{input_param}];"
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd10, %rd10;").map_err(write_err)?;
        writeln!(
            ptx,
            "\tmul.wide.u32 %rd11, %r3, {bytes};",
            bytes = p.elem_bytes
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd12, %rd10, %rd11;").map_err(write_err)?;

        let (val_idx, ret_idx) = take_two_slots(&mut class_slot_counter, p.reg_class);
        // Per-spec input column is read-only (host upload-side guarantee).
        writeln!(
            ptx,
            "\tld.global.nc.{ld} %{rc}{vi}, [%rd12];",
            ld = p.load_suffix,
            rc = p.reg_class,
            vi = val_idx,
        )
        .map_err(write_err)?;

        // Accumulator slot address: acc_j + slot * elem_bytes_j.
        writeln!(
            ptx,
            "\tld.param.u64 %rd13, [{entry}_param_{acc_param}];"
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd13, %rd13;").map_err(write_err)?;
        writeln!(
            ptx,
            "\tmul.wide.u32 %rd14, %r8, {bytes};",
            bytes = p.elem_bytes
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd15, %rd13, %rd14;").map_err(write_err)?;

        writeln!(
            ptx,
            "\t{atomic} %{rc}{ri}, [%rd15], %{rc}{vi};",
            atomic = p.atomic,
            rc = p.reg_class,
            ri = ret_idx,
            vi = val_idx,
        )
        .map_err(write_err)?;
    }

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Block size accessor for the host-side launcher. Kept private to the module
/// for now; the executor reads it via [`groupby_block_size`].
pub fn groupby_block_size() -> u32 {
    BLOCK_SIZE
}

/// PTX `atom.global.*` mnemonic (with no operands) for the given op + dtype.
/// Returns an error for combinations the v1 implementation does not support
/// (most notably float MIN/MAX, which would need a CAS loop).
fn atomic_for(op: ReduceOp, dtype: DataType) -> BoltResult<&'static str> {
    use DataType::*;
    use ReduceOp::*;
    Ok(match (op, dtype) {
        (Sum, Int32) | (Count, Int32) => "atom.global.add.s32",
        // PTX has no `atom.add.s64` — only `.u64`. Two's-complement signed
        // addition is bit-identical to unsigned addition, so emitting `.u64`
        // for an `Int64` accumulator is sound. See PTX ISA, "atom" —
        // supported types are {u32, s32, u64, f16, f16x2, f32, f64, bf16, bf16x2}.
        (Sum, Int64) | (Count, Int64) => "atom.global.add.u64",
        (Sum, Float32) | (Count, Float32) => "atom.global.add.f32",
        (Sum, Float64) | (Count, Float64) => "atom.global.add.f64",

        (Min, Int32) => "atom.global.min.s32",
        (Min, Int64) => "atom.global.min.s64",
        (Max, Int32) => "atom.global.max.s32",
        (Max, Int64) => "atom.global.max.s64",

        (Min, Float32) | (Min, Float64) | (Max, Float32) | (Max, Float64) => {
            return Err(BoltError::Other(
                "MIN/MAX over float not yet supported in GROUP BY".into(),
            ))
        }

        (_, Bool) | (_, Utf8) | (_, Decimal128(_, _)) | (_, Date32) | (_, Timestamp(_, _)) => {
            return Err(BoltError::Type(format!(
                "hash_kernels: aggregate over dtype {:?} not supported",
                dtype
            )))
        }
    })
}

/// `(ld_suffix, reg_class)` for the input column / accumulator value type.
///
/// The register class is intentionally distinct from the `%r`, `%rl`, `%rd`
/// classes used for hashing/probing so the two namespaces don't collide.
fn ptx_type_info(dtype: DataType) -> BoltResult<(&'static str, &'static str)> {
    Ok(match dtype {
        DataType::Int32 => ("s32", "vr"),
        DataType::Int64 => ("s64", "vl"),
        DataType::Float32 => ("f32", "vf"),
        DataType::Float64 => ("f64", "vd"),
        DataType::Bool
        | DataType::Utf8
        | DataType::Decimal128(_, _)
        | DataType::Date32
        | DataType::Timestamp(_, _) => {
            return Err(BoltError::Type(format!(
                "hash_kernels: dtype {:?} not supported in aggregate kernel",
                dtype
            )))
        }
    })
}

/// PTX `.reg` declaration type for the input-value register class returned by
/// [`ptx_type_info`].
fn reg_decl_ty(dtype: DataType) -> BoltResult<&'static str> {
    Ok(match dtype {
        DataType::Int32 => "b32",
        DataType::Int64 => "b64",
        DataType::Float32 => "f32",
        DataType::Float64 => "f64",
        DataType::Bool
        | DataType::Utf8
        | DataType::Decimal128(_, _)
        | DataType::Date32
        | DataType::Timestamp(_, _) => {
            return Err(BoltError::Type(format!(
                "hash_kernels: dtype {:?} not supported in aggregate kernel",
                dtype
            )))
        }
    })
}

/// Adapt a `std::fmt::Error` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("hash_kernels: write failed: {}", e))
}

// ---------------------------------------------------------------------------
// PTX-shape golden tests for the Stage C validity wiring. These are host-only
// (no CUDA) — they assert that the emitted PTX text grows the expected param
// + `bfe.u32` + skip-on-null shape, not that it runs correctly. End-to-end
// numeric correctness is exercised by the GPU tests in `tests/e2e_tests.rs`.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod ptx_shape_tests {
    use super::*;

    /// The classic (no-validity) keys kernel keeps its historical 4-param ABI
    /// and emits no `bfe.u32` extraction.
    #[test]
    fn keys_kernel_classic_has_4_params_and_no_bfe() {
        let ptx = compile_groupby_keys_kernel().expect("classic keys ptx");
        // 4 params: 0..=3
        assert!(ptx.contains("bolt_groupby_keys_param_3"));
        assert!(!ptx.contains("bolt_groupby_keys_param_4"));
        assert!(!ptx.contains("bfe.u32"));
    }

    /// The Stage C keys kernel adds a 5th param and emits the packed-bit
    /// extract + branch-to-DONE shape.
    #[test]
    fn keys_kernel_with_validity_adds_param_and_bfe() {
        let ptx = compile_groupby_keys_kernel_with_validity()
            .expect("validity keys ptx");
        // 5 params: 0..=4
        assert!(ptx.contains("bolt_groupby_keys_param_4"));
        // word_idx = tid >> 5
        assert!(ptx.contains("shr.u32 %r10, %r3, 5;"));
        // bit_off = tid & 31
        assert!(ptx.contains("and.b32 %r11, %r3, 31;"));
        // bfe extracts the single bit
        assert!(ptx.contains("bfe.u32 %r13, %r12, %r11, 1;"));
        // setp + branch on zero
        assert!(ptx.contains("setp.eq.s32 %p4, %r13, 0;"));
        assert!(ptx.contains("@%p4 bra DONE;"));
    }

    /// The classic agg kernel keeps its historical 6-param ABI.
    #[test]
    fn agg_kernel_classic_has_6_params_and_no_bfe() {
        let ptx = compile_groupby_agg_kernel(ReduceOp::Sum, DataType::Int64)
            .expect("classic agg ptx");
        assert!(ptx.contains("bolt_groupby_agg_param_5"));
        assert!(!ptx.contains("bolt_groupby_agg_param_6"));
        assert!(!ptx.contains("bfe.u32"));
    }

    /// The Stage C agg kernel adds a 7th param (value validity) and emits
    /// the bit-extract + skip-on-null gate.
    #[test]
    fn agg_kernel_with_validity_adds_param_and_bfe() {
        let ptx = compile_groupby_agg_kernel_with_validity(
            ReduceOp::Sum,
            DataType::Int64,
        )
        .expect("validity agg ptx");
        assert!(ptx.contains("bolt_groupby_agg_param_6"));
        assert!(ptx.contains("shr.u32 %r14, %r3, 5;"));
        assert!(ptx.contains("and.b32 %r15, %r3, 31;"));
        assert!(ptx.contains("bfe.u32 %r17, %r16, %r15, 1;"));
        assert!(ptx.contains("setp.eq.s32 %p4, %r17, 0;"));
        assert!(ptx.contains("@%p4 bra DONE;"));
        // The atom.add must still be present after the gate.
        assert!(ptx.contains("atom.global.add.u64"));
    }

    /// The fused multi-aggregate kernel hashes the key ONCE and emits one
    /// atomic per spec — verifying both the fusion (single hash block) and
    /// the spec-count scaling (N atomic.add lines for N SUM specs).
    ///
    /// We build three Sum/Count specs that all lower to `atom.global.add`
    /// so a literal count of `atom.global.add` lines == 3, and we check
    /// that the canonical splitmix multiplier appears exactly once
    /// (a proxy for "the hash-mul-FNV block is emitted exactly once").
    #[test]
    fn agg_multi_kernel_emits_n_atomics_and_one_hash() {
        let specs = [
            AggSpec { op: ReduceOp::Sum,   input_dtype: DataType::Int64 },
            AggSpec { op: ReduceOp::Count, input_dtype: DataType::Int64 },
            AggSpec { op: ReduceOp::Sum,   input_dtype: DataType::Int32 },
        ];
        let ptx = compile_groupby_agg_kernel_multi(&specs)
            .expect("fused multi-agg ptx");

        // Three atomic updates, one per spec. Each Sum/Count over Int*
        // lowers to `atom.global.add.<u64|s32>` (see `atomic_for`).
        let n_atomics = ptx.matches("atom.global.add").count();
        assert_eq!(
            n_atomics, 3,
            "expected 3 atom.global.add lines for 3 specs, got {n_atomics}\n\
             --- emitted PTX ---\n{ptx}"
        );

        // The hash block writes the FNV/splitmix multiplier into %rl1
        // exactly once. If the loop body re-hashed per spec, we'd see this
        // literal appear N times.
        let mul_literal = format!("mov.s64 %rl1, {};", FX_MUL);
        let n_hash_blocks = ptx.matches(mul_literal.as_str()).count();
        assert_eq!(
            n_hash_blocks, 1,
            "expected exactly one hash-mul-FNV block (one `mov.s64 %rl1, FX_MUL`), \
             got {n_hash_blocks} — fusion isn't real\n\
             --- emitted PTX ---\n{ptx}"
        );

        // And the entry point should be the fused name.
        assert!(
            ptx.contains(&format!(".visible .entry {AGG_KERNEL_MULTI_ENTRY}(")),
            "fused entry-point name missing"
        );
    }

    /// `specs` must be non-empty.
    #[test]
    fn agg_multi_rejects_empty_specs() {
        assert!(compile_groupby_agg_kernel_multi(&[]).is_err());
    }

    /// The fused kernel's `.param .u64` count is `2 + 2 * n_specs`
    /// (group_col + keys_table + N input ptrs + N acc ptrs); the trailing
    /// `n_rows` and `k` are `.u32`.
    #[test]
    fn agg_multi_param_count_matches_signature() {
        for n_specs in 1..=4 {
            let specs: Vec<AggSpec> = (0..n_specs)
                .map(|_| AggSpec {
                    op: ReduceOp::Sum,
                    input_dtype: DataType::Int64,
                })
                .collect();
            let ptx = compile_groupby_agg_kernel_multi(&specs).unwrap();
            let expected_u64 = 2 + 2 * n_specs;
            let got_u64 = ptx.matches(".param .u64 ").count();
            assert_eq!(
                got_u64, expected_u64,
                "n_specs={n_specs}: expected {expected_u64} .u64 params, got {got_u64}"
            );
            let got_u32 = ptx.matches(".param .u32 ").count();
            assert_eq!(
                got_u32, 2,
                "n_specs={n_specs}: expected 2 .u32 params (n_rows + k), got {got_u32}"
            );
        }
    }

    /// Packed-bit word count rounds up.
    #[test]
    fn packed_validity_word_count_rounds_up() {
        assert_eq!(packed_validity_word_count(0), 1);
        assert_eq!(packed_validity_word_count(1), 1);
        assert_eq!(packed_validity_word_count(31), 1);
        assert_eq!(packed_validity_word_count(32), 1);
        assert_eq!(packed_validity_word_count(33), 2);
        assert_eq!(packed_validity_word_count(64), 2);
        assert_eq!(packed_validity_word_count(65), 3);
        assert_eq!(packed_validity_word_count(1_000_000), 31_250);
    }

    // -----------------------------------------------------------------
    // Robin Hood keys-kernel PTX-shape tests. Like the validity tests,
    // these are host-only — they assert the emitter produces the
    // expected PTX SHAPE (entry-point name, param count, presence of
    // the swap branch + atomic-cas instructions, bounded-probe cap).
    // End-to-end GPU correctness is intentionally NOT tested here.
    // -----------------------------------------------------------------

    /// The Robin Hood kernel exposes a distinct entry-point name so the
    /// executor can load it alongside (not in place of) the classic kernel.
    #[test]
    fn rh_kernel_uses_distinct_entry_name() {
        let ptx = compile_groupby_keys_kernel_robin_hood().expect("rh ptx");
        assert!(
            ptx.contains(&format!(".visible .entry {}(", KEYS_KERNEL_RH_ENTRY)),
            "RH entry-point name missing"
        );
        // Must NOT collide with the linear-probe entry point.
        assert!(
            !ptx.contains(&format!(".visible .entry {}(", KEYS_KERNEL_ENTRY)),
            "RH ptx should not declare the linear entry point"
        );
    }

    /// The Robin Hood kernel keeps the classic 4-param ABI (no validity).
    /// Validity is out of scope for this first cut; the dispatcher's
    /// fallback path keeps the classic-with-validity emitter for that case.
    #[test]
    fn rh_kernel_has_four_params() {
        let ptx = compile_groupby_keys_kernel_robin_hood().expect("rh ptx");
        // Four params: 0..=3
        assert!(ptx.contains(&format!("{}_param_3", KEYS_KERNEL_RH_ENTRY)));
        assert!(!ptx.contains(&format!("{}_param_4", KEYS_KERNEL_RH_ENTRY)));
    }

    /// The Robin Hood kernel must emit `atom.global.cas.b64` for both
    /// the empty-slot claim AND the swap (CAS-with-expected). The
    /// earlier `atom.global.exch.b64` swap design was racy under
    /// contention and has been removed.
    #[test]
    fn rh_kernel_emits_cas_for_claim_and_swap() {
        let ptx = compile_groupby_keys_kernel_robin_hood().expect("rh ptx");
        assert!(
            ptx.contains("atom.global.cas.b64"),
            "RH must use atom.cas for empty-slot claim and CAS-with-expected swap"
        );
        // The exch-based swap is gone; assert it's not regressed in.
        assert!(
            !ptx.contains("atom.global.exch.b64"),
            "RH must NOT use atom.exch — the swap is now CAS-with-expected to avoid the contention race"
        );
    }

    /// Race-free swap stress test: assert the emitted PTX carries both
    /// the CAS-with-expected swap and the RH_RETRY re-probe path, and
    /// does NOT contain the legacy `atom.global.exch.b64` swap form.
    /// This is a guard against accidental regressions of the
    /// contention-race fix.
    #[test]
    fn rh_kernel_swap_is_race_free_cas_with_expected() {
        let ptx = compile_groupby_keys_kernel_robin_hood().expect("rh ptx");

        // (1) RH path must contain atom.global.cas.b64 — used by both
        // the empty-slot claim AND the swap-with-expected step. The
        // emitter should produce at least two distinct CAS sites.
        let cas_count = ptx.matches("atom.global.cas.b64").count();
        assert!(
            cas_count >= 2,
            "RH must emit at least two atom.global.cas.b64 sites \
             (one for empty-claim, one for swap-with-expected); found {cas_count}"
        );

        // (2) The legacy exch-based swap must be gone.
        assert!(
            !ptx.contains("atom.global.exch.b64"),
            "RH PTX must not contain atom.global.exch.b64 (was racy under contention)"
        );

        // (3) Both control-flow labels for the new swap dance must
        // exist: RH_SWAP (entered on cur_dist > occ_dist) and
        // RH_RETRY (entered on CAS-with-expected failure).
        assert!(
            ptx.contains("RH_SWAP:"),
            "missing RH_SWAP label (swap-with-expected entry point)"
        );
        assert!(
            ptx.contains("RH_RETRY:"),
            "missing RH_RETRY label (CAS-failure re-probe path)"
        );

        // (4) The retry path must branch back to the main loop without
        // having mutated cur_key (we re-probe the same slot from the
        // top of the loop).
        let retry_pos = ptx.find("RH_RETRY:").expect("RH_RETRY present");
        let after_retry = &ptx[retry_pos..];
        assert!(
            after_retry.starts_with("RH_RETRY:\n\tbra RH_PROBE_LOOP;"),
            "RH_RETRY must immediately branch to RH_PROBE_LOOP without other side-effects"
        );
    }

    /// The Robin Hood kernel must emit the swap branch label (RH_SWAP)
    /// so the linear path can fall into it on richer-than-occupant
    /// comparison. Also asserts the bounded-probe cap is
    /// MAX_RH_PROBE * 2 (doubled to absorb RH_RETRY re-probes under
    /// contention without spinning forever).
    #[test]
    fn rh_kernel_emits_swap_branch_and_bound() {
        let ptx = compile_groupby_keys_kernel_robin_hood().expect("rh ptx");
        assert!(ptx.contains("RH_PROBE_LOOP:"), "missing RH_PROBE_LOOP label");
        assert!(ptx.contains("RH_SWAP:"), "missing RH_SWAP label");
        // The bounded-probe cap mov should reference MAX_RH_PROBE * 2
        // (doubled because RH_RETRY can legitimately re-probe the same
        // slot under contention; doubling the budget keeps the
        // silent-failure semantics consistent with the linear path).
        let expected_cap = MAX_RH_PROBE * 2;
        assert!(
            ptx.contains(&format!("mov.u32 %r11, {};", expected_cap)),
            "RH_PROBE bound must be MAX_RH_PROBE * 2 = {} (saw PTX: ...)",
            expected_cap
        );
    }

    /// The Robin Hood kernel must compute the occupant's probe distance
    /// (the load-bearing comparison for the swap decision). We assert
    /// the kernel uses the same splitmix multiplier on the occupant
    /// (i.e. `hash(occupant)` is computed for the dist comparison).
    #[test]
    fn rh_kernel_hashes_occupant_for_distance() {
        let ptx = compile_groupby_keys_kernel_robin_hood().expect("rh ptx");
        // The splitmix multiplier is loaded into %rl1 once and reused
        // for hashing both the input key (initial hash) and the
        // occupant (during the swap decision). We therefore expect a
        // SECOND multiply against %rl1 reading the occupant key from
        // %rl5 (the CAS return register).
        assert!(
            ptx.contains("mul.lo.s64 %rl6, %rl5, %rl1;"),
            "RH must re-multiply occupant key by FX_MUL to derive its home slot.\n\
             --- emitted PTX ---\n{}",
            ptx
        );
    }

    /// The dispatcher routes to the linear-probe kernel by default
    /// (BOLT_HASH_ALGO unset).
    ///
    /// Note: this test temporarily unsets the env var; if other tests
    /// in this binary set it concurrently the result is racy. The test
    /// module is intended to run with `--test-threads=1` for the env
    /// cases; the assertion is conservative (just verifies the entry
    /// name returned matches the default path).
    #[test]
    fn dispatcher_defaults_to_linear_probe() {
        // Save + clear the env so the dispatcher takes the default branch.
        let prev = std::env::var("BOLT_HASH_ALGO").ok();
        std::env::remove_var("BOLT_HASH_ALGO");
        let (_ptx, entry) = compile_groupby_keys_kernel_dispatched()
            .expect("dispatcher default");
        assert_eq!(entry, KEYS_KERNEL_ENTRY);
        // Restore.
        if let Some(v) = prev {
            std::env::set_var("BOLT_HASH_ALGO", v);
        }
    }

    /// The dispatcher routes to the Robin Hood kernel when
    /// `BOLT_HASH_ALGO=robin_hood` is set. Also accepts `rh` as a
    /// shorthand.
    #[test]
    fn dispatcher_opts_into_robin_hood() {
        let prev = std::env::var("BOLT_HASH_ALGO").ok();

        std::env::set_var("BOLT_HASH_ALGO", "robin_hood");
        let (_ptx, entry) = compile_groupby_keys_kernel_dispatched()
            .expect("rh long form");
        assert_eq!(entry, KEYS_KERNEL_RH_ENTRY);

        std::env::set_var("BOLT_HASH_ALGO", "RH");
        let (_ptx, entry) = compile_groupby_keys_kernel_dispatched()
            .expect("rh short upper");
        assert_eq!(entry, KEYS_KERNEL_RH_ENTRY);

        // Unknown values fall back to linear.
        std::env::set_var("BOLT_HASH_ALGO", "wibble");
        let (_ptx, entry) = compile_groupby_keys_kernel_dispatched()
            .expect("unknown fallback");
        assert_eq!(entry, KEYS_KERNEL_ENTRY);

        // Restore.
        std::env::remove_var("BOLT_HASH_ALGO");
        if let Some(v) = prev {
            std::env::set_var("BOLT_HASH_ALGO", v);
        }
    }
}
