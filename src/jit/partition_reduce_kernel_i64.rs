// SPDX-License-Identifier: Apache-2.0

//! Per-partition shared-memory GROUP BY SUM kernel — **Int64-key variant**
//! (Tier 2.1 for two-key GROUP BY).
//!
//! Sibling of [`crate::jit::partition_reduce_kernel`]. The Int32 sibling
//! handles single-Int32-key Tier-2 (q5); this file handles the two-Int32-
//! keys-packed-into-Int64 case used by the two-key Tier-2 path (q3).
//!
//! ## Why this exists
//!
//! Before this kernel landed, `groupby_tier2_twokey_orchestrator.rs`
//! reduced its scattered partitions on the **host** (download both keys
//! and vals, build N=4096 small `HashMap<i64, f64>`s, push the result).
//! That cost roughly the same as host-pass-2 did for the i32 path:
//! ~150 ms of D2H + HashMap work on a 10 M-row workload. Measured: q3
//! went from 807 ms baseline → 953 ms with twokey-Tier-2 enabled (host
//! pass-2), so the two-key path was disabled at integration time.
//!
//! This kernel is the i64-key analog of the i32 per-partition reduce
//! kernel that already won 3.7× on q5. Wiring it into the twokey
//! orchestrator restores Tier-2's structural advantage at the i64 key
//! width.
//!
//! ## Implementation note — shared with the i32 sibling
//!
//! The i32-key and i64-key SUM kernels share ~80% of their PTX scaffold.
//! Rather than duplicate that scaffold here, both delegate to the unified
//! [`crate::jit::partition_reduce_kernel::emit_sum_kernel`] generator,
//! which branches on a [`crate::jit::partition_reduce_kernel::KeyWidth`]
//! token only where the two genuinely differ (key load/store/compare
//! width, the i64 slot derivation via `cvt.u32.u64`, the key slot stride,
//! and a handful of register numbers). The four public `compile_*`
//! functions across the two files are thin wrappers selecting the right
//! `key_width`/`spill`/`entry`.
//!
//! ## Layout differences from the i32 variant
//!
//! | What                  | i32 variant            | i64 variant (here)      |
//! | --------------------- | ---------------------- | ----------------------- |
//! | `block_keys` slot     | 4 B                    | **8 B**                 |
//! | `block_keys_buf`      | 4 KiB (1024 × 4 B)     | **8 KiB** (1024 × 8 B)  |
//! | Key load              | `ld.global.s32`        | `ld.global.s64`         |
//! | Key compare           | `setp.eq.s32`          | `setp.eq.s64`           |
//! | Key store (shared)    | `st.shared.u32`        | `st.shared.u64`         |
//! | Output key store      | `st.global.s32`        | `st.global.s64`         |
//! | Output buffer / slot  | 4 B                    | **8 B**                 |
//! | Output buffer total   | 4 MiB (4096×1024×4 B)  | **8 MiB** (×8 B)        |
//! | Total per-block shmem | 16 KiB                 | **20 KiB**              |
//!
//! 20 KiB per block is comfortably under the 48 KiB sm_70 static budget.
//! Total output footprint is 8 + 8 + 4 = 20 B × 4096 × 1024 = 80 MiB
//! (vs 52 MiB for i32). The extra 28 MiB D2H is ~3 ms at PCIe Gen3 x16 —
//! negligible relative to the wins we expect.
//!
//! ## Slot mapping
//!
//! The partition kernel hashes i64 keys using Knuth's 64-bit Fibonacci
//! multiplier and takes the HIGH log₂(K) bits to select the partition.
//! Inside a partition the keys are "random" w.r.t. that selection, so we
//! can route them to shared-table slots using the **low 32 bits** of the
//! key as a direct slot index (`& (BLOCK_GROUPS - 1)`). For h2o.ai q3
//! (two i32 keys packed as `(id1 << 32) | id2`) this means the slot is
//! dictated by `id2 & 0x3FF` — id2 ranges 0..999 and is dense, giving an
//! evenly-distributed slot map.
//!
//! If a future workload arrives where the low 32 bits cluster, we can
//! drop in a `mul.lo.u32 %r_slot, %r_low_key, KNUTH_MULTIPLIER` between
//! the cast and the mask. That's a 2-line PTX change.
//!
//! ## Probe / collision / drop policy
//!
//! Identical to the i32 variant: open-addressing linear probe bounded
//! by `MAX_PROBES = BLOCK_GROUPS`, drop on overflow (deliberate v0).

use crate::error::BoltResult;

/// Number of slots in each block's shared-memory open-addressing table.
/// Must match [`crate::jit::partition_reduce_kernel::BLOCK_GROUPS`] so the
/// orchestrator can share the launch-geometry / output-sizing assumptions.
pub const BLOCK_GROUPS: u32 = 1024;

/// Threads per block. Matches the i32 sibling.
pub const BLOCK_THREADS: u32 = 256;

/// Number of partitions. Must match [`crate::jit::partition_kernel_i64::NUM_PARTITIONS`]
/// (which is 4096 after the Tier-2.1 NUM_PARTITIONS tuning).
pub const NUM_PARTITIONS: u32 = 4096;

/// Entry-point name embedded in the emitted PTX.
pub const KERNEL_ENTRY: &str = "bolt_partition_reduce_i64";

/// Entry-point name for the spill-counter variant. Distinct from
/// [`KERNEL_ENTRY`] so both kernels can coexist in the JIT module cache.
pub const KERNEL_ENTRY_WITH_SPILL: &str = "bolt_partition_reduce_i64_spill";

/// Probe bound. Same as i32 sibling.
const MAX_PROBES: u32 = BLOCK_GROUPS;

/// Per-iteration `nanosleep.u32` operand for the collision-advance path.
/// See `partition_reduce_kernel::SPIN_BACKOFF_NS` for full rationale.
/// TODO(perf): exponential back-off.
const SPIN_BACKOFF_NS: u32 = 32;

/// Generate PTX for the i64-key per-partition reduce kernel.
///
/// Kernel signature (PTX-level):
/// ```text
/// .visible .entry bolt_partition_reduce_i64(
///     .param .u64 partition_keys,    // const int64_t*  scatter_keys[n_rows]
///     .param .u64 partition_vals,    // const double*   scatter_vals[n_rows]
///     .param .u64 partition_offsets, // const uint32_t* offsets[NUM_PARTITIONS+1]
///     .param .u64 out_keys,          //       int64_t*  [NUM_PARTITIONS*BLOCK_GROUPS]
///     .param .u64 out_vals,          //       double*   [NUM_PARTITIONS*BLOCK_GROUPS]
///     .param .u64 out_set            //       uint8_t*  [NUM_PARTITIONS*BLOCK_GROUPS]
/// )
/// ```
///
/// Launch geometry: `grid = NUM_PARTITIONS, block = BLOCK_THREADS`. One
/// block per partition.
///
/// Delegates to the unified
/// [`crate::jit::partition_reduce_kernel::emit_sum_kernel`] with the
/// `I64` key width (see that function for the shared scaffold).
pub fn compile_partition_reduce_kernel_i64() -> BoltResult<String> {
    super::partition_reduce_kernel::emit_sum_kernel(
        super::partition_reduce_kernel::KeyWidth::I64,
        false,
        KERNEL_ENTRY,
    )
}

/// Spill-counter-aware sibling of [`compile_partition_reduce_kernel_i64`].
///
/// Identical algorithm and shared-memory layout, plus one extra kernel
/// parameter:
///
/// ```text
/// .param .u64 spill_counter   // uint32_t* &spill_counter[1]  (may be 0)
/// ```
///
/// On a probe overflow (MAX_PROBES exceeded without a free or matching
/// slot), the kernel issues `atom.global.add.u32 [spill_counter], 1`
/// before dropping the row — but only if the pointer is non-null. Host
/// orchestrators read the counter after launch+sync; any non-zero value
/// indicates the partition table overflowed and the per-group sums for
/// the spilled key would be silently incorrect.
///
/// Exports the distinct entry [`KERNEL_ENTRY_WITH_SPILL`] so both can
/// coexist in the same JIT cache. Delegates to the unified
/// [`crate::jit::partition_reduce_kernel::emit_sum_kernel`] with the `I64`
/// key width and `spill = true`.
pub fn compile_partition_reduce_kernel_i64_with_spill() -> BoltResult<String> {
    super::partition_reduce_kernel::emit_sum_kernel(
        super::partition_reduce_kernel::KeyWidth::I64,
        true,
        KERNEL_ENTRY_WITH_SPILL,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_returns_ok() {
        let ptx = compile_partition_reduce_kernel_i64().expect("compile");
        assert!(!ptx.is_empty());
    }

    #[test]
    fn has_correct_entry() {
        let ptx = compile_partition_reduce_kernel_i64().expect("compile");
        assert!(
            ptx.contains(".visible .entry bolt_partition_reduce_i64("),
            "entry point not found"
        );
    }

    #[test]
    fn uses_i64_key_loads() {
        let ptx = compile_partition_reduce_kernel_i64().expect("compile");
        assert!(
            ptx.contains("ld.global.s64") || ptx.contains("ld.shared.s64"),
            "i64 key loads not found:\n{ptx}"
        );
    }

    #[test]
    fn uses_i64_key_stores_in_output() {
        let ptx = compile_partition_reduce_kernel_i64().expect("compile");
        assert!(
            ptx.contains("st.global.s64"),
            "i64 key store to global not found:\n{ptx}"
        );
    }

    #[test]
    fn uses_atom_shared_cas() {
        let ptx = compile_partition_reduce_kernel_i64().expect("compile");
        assert!(
            ptx.contains("atom.shared.cas.b32"),
            "atom.shared.cas.b32 (slot-claim) not found"
        );
    }

    #[test]
    fn uses_atom_shared_add_f64() {
        let ptx = compile_partition_reduce_kernel_i64().expect("compile");
        assert!(
            ptx.contains("atom.shared.add.f64"),
            "atom.shared.add.f64 (sum) not found"
        );
    }

    #[test]
    fn has_two_syncthreads() {
        let ptx = compile_partition_reduce_kernel_i64().expect("compile");
        let count = ptx.matches("bar.sync 0").count();
        assert!(count >= 2, "expected ≥ 2 bar.sync 0, got {count}");
    }

    /// 1024 slots × 8 B keys + 1024 × 8 B vals + 1024 × 4 B set = 20 KiB.
    #[test]
    fn shared_mem_under_48k() {
        let ptx = compile_partition_reduce_kernel_i64().expect("compile");
        // Coarse check: shared arrays sized to expected bytes.
        assert!(ptx.contains("block_keys_buf[8192]"));
        assert!(ptx.contains("block_vals_buf[8192]"));
        assert!(ptx.contains("block_set_buf[4096]"));
    }

    // ----- _with_spill variant shape tests ---------------------------------

    /// Spill variant exposes a different entry name so both can live in
    /// the same JIT cache.
    #[test]
    fn with_spill_uses_distinct_entry_name() {
        let ptx = compile_partition_reduce_kernel_i64_with_spill().expect("compiles");
        assert_eq!(KERNEL_ENTRY_WITH_SPILL, "bolt_partition_reduce_i64_spill");
        let needle = format!(".visible .entry {}(", KERNEL_ENTRY_WITH_SPILL);
        assert!(ptx.contains(&needle), "PTX missing spill entry:\n{ptx}");
        assert!(
            !ptx.contains(".visible .entry bolt_partition_reduce_i64("),
            "spill variant must not also export base entry"
        );
    }

    /// Adds one .u64 param + the global atomic on overflow.
    #[test]
    fn with_spill_has_seven_params_and_global_atomic() {
        let ptx = compile_partition_reduce_kernel_i64_with_spill().expect("compiles");
        let n = ptx.matches(".param .u64 ").count();
        assert_eq!(n, 7, "spill variant must expose 7 .u64 params, got {n}");
        assert!(
            ptx.contains("atom.global.add.u32"),
            "spill kernel must bump the counter atomically:\n{ptx}"
        );
        assert!(
            ptx.contains("SPILL_BUMP:"),
            "spill kernel must label the overflow path"
        );
        assert!(
            ptx.contains("setp.eq.u64"),
            "spill kernel must null-check the spill_counter pointer"
        );
    }
}
