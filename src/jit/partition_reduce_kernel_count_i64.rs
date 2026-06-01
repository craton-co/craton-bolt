// SPDX-License-Identifier: Apache-2.0

//! Per-partition COUNT(*) reduce kernel — **i64 key variant**.
//!
//! Mirror of [`crate::jit::partition_reduce_kernel_count`] (i32 key)
//! adapted for the i64-packed-two-key Tier-2.1 path. Identical to its
//! i32 sibling except:
//!
//!   * Keys are loaded with `ld.global.s64` and stored with `st.global.s64`
//!   * Slot computation uses `cvt.u32.u64` on the low 32 bits then masks
//!   * `block_keys_buf` is 8 KiB (vs 4 KiB)
//!   * Output buffer's per-slot stride for keys is 8 B (vs 4 B)
//!
//! Used by the two-key COUNT(*) executor (`SELECT a, b, COUNT(*) FROM
//! x GROUP BY a, b`) and as the COUNT denominator for the two-key AVG
//! executor (when added).
//!
//! ## Shared emit helpers
//!
//! The non-spill [`compile_partition_reduce_kernel_count_i64`] and the
//! spill-aware [`compile_partition_reduce_kernel_count_i64_with_spill`]
//! are ~80% byte-identical. The shared phases are factored into the
//! `emit_*` private helpers below; each helper emits bytes that are
//! literally identical across both callers. The publish/probe protocol
//! is shared one level up via
//! `super::partition_reduce_kernel_spill_common::emit_publish_probe_protocol`.
//! Only the genuinely divergent bits (shared-buffer symbol suffix,
//! register-bank sizes, the PROBE_TOP overflow target, the spin-backoff,
//! the MATCH fall-through, the spill param/bump, and the two export
//! predicate numbers) remain per-emitter.

use crate::error::BoltResult;

pub const BLOCK_GROUPS: u32 = 1024;
pub const BLOCK_THREADS: u32 = 256;
pub const NUM_PARTITIONS: u32 = 4096;

pub const KERNEL_ENTRY: &str = "bolt_partition_reduce_count_i64";

/// Entry-point name for the spill-counter variant.
pub const KERNEL_ENTRY_WITH_SPILL: &str = "bolt_partition_reduce_count_i64_spill";

/// Generate PTX for the i64-key COUNT(*) per-partition reduce kernel.
///
/// Signature:
/// ```text
/// .visible .entry bolt_partition_reduce_count_i64(
///     .param .u64 partition_keys,    // const int64_t* scatter_keys[n_rows]
///     .param .u64 partition_offsets, // const uint32_t* offsets[K+1]
///     .param .u64 out_keys,          //       int64_t* [K*BG]
///     .param .u64 out_counts,        //       uint64_t* [K*BG]
///     .param .u64 out_set            //       uint8_t* [K*BG]
/// )
/// ```
///
/// Delegates to the unified
/// [`crate::jit::partition_reduce_kernel_count::emit_count_kernel`] with the
/// `I64` key width (see that function for the shared scaffold).
pub fn compile_partition_reduce_kernel_count_i64() -> BoltResult<String> {
    super::partition_reduce_kernel_count::emit_count_kernel(
        super::partition_reduce_kernel::KeyWidth::I64,
        false,
        KERNEL_ENTRY,
    )
}

/// Spill-counter-aware sibling of [`compile_partition_reduce_kernel_count_i64`].
///
/// Identical algorithm; adds one extra `.param .u64 spill_counter`
/// (uint32_t* &spill_counter[1]; may be 0/null). On MAX_PROBES overflow
/// the kernel issues `atom.global.add.u32 [spill_counter], 1` after a
/// null check, then drops the row.
///
/// Delegates to the unified
/// [`crate::jit::partition_reduce_kernel_count::emit_count_kernel`] with the
/// `I64` key width and `spill = true`.
pub fn compile_partition_reduce_kernel_count_i64_with_spill() -> BoltResult<String> {
    super::partition_reduce_kernel_count::emit_count_kernel(
        super::partition_reduce_kernel::KeyWidth::I64,
        true,
        KERNEL_ENTRY_WITH_SPILL,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_ok() {
        assert!(compile_partition_reduce_kernel_count_i64().is_ok());
    }

    #[test]
    fn has_correct_entry() {
        let ptx = compile_partition_reduce_kernel_count_i64().unwrap();
        assert!(ptx.contains(".visible .entry bolt_partition_reduce_count_i64("));
    }

    #[test]
    fn uses_i64_key_loads_and_stores() {
        let ptx = compile_partition_reduce_kernel_count_i64().unwrap();
        assert!(ptx.contains("ld.global.s64"));
        assert!(ptx.contains("st.global.s64"));
        assert!(ptx.contains("ld.shared.s64"));
        assert!(ptx.contains("st.shared.u64"));
    }

    #[test]
    fn uses_atom_shared_add_u64() {
        let ptx = compile_partition_reduce_kernel_count_i64().unwrap();
        assert!(ptx.matches("atom.shared.add.u64").count() >= 2);
    }

    // ----- _with_spill variant shape tests ---------------------------------

    #[test]
    fn with_spill_uses_distinct_entry_name() {
        let ptx = compile_partition_reduce_kernel_count_i64_with_spill().expect("compiles");
        assert_eq!(
            KERNEL_ENTRY_WITH_SPILL,
            "bolt_partition_reduce_count_i64_spill"
        );
        let needle = format!(".visible .entry {}(", KERNEL_ENTRY_WITH_SPILL);
        assert!(ptx.contains(&needle));
        assert!(!ptx.contains(".visible .entry bolt_partition_reduce_count_i64("));
    }

    #[test]
    fn with_spill_has_six_params_and_global_atomic() {
        let ptx = compile_partition_reduce_kernel_count_i64_with_spill().expect("compiles");
        let n = ptx.matches(".param .u64 ").count();
        assert_eq!(n, 6, "spill variant must add one .u64 param (got {n})");
        assert!(ptx.contains("atom.global.add.u32"));
        assert!(ptx.contains("SPILL_BUMP:"));
        assert!(ptx.contains("setp.eq.u64"));
    }
}
