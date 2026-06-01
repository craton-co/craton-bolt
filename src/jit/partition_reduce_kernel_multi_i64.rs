// SPDX-License-Identifier: Apache-2.0

//! Per-partition shared-memory reduce kernel — **i64-key, multi-value
//! SUM**. The intersection of `partition_reduce_kernel_multi` (i32-key,
//! N-value) and `partition_reduce_kernel_i64` (i64-key, single-value).
//!
//! Used by the two-key multi-aggregate Tier-2.1 path: pack two i32 keys
//! into an i64 (per `groupby.rs::pack_keys`), partition + scatter
//! through the i64 pipeline, then this kernel reduces each partition
//! into N parallel f64 SUMs keyed by i64.
//!
//! ## Shared-memory layout per block
//!
//! block_keys    : i64 × 1024 =  8 KiB
//! block_vals_0  : f64 × 1024 =  8 KiB
//! block_vals_1  : f64 × 1024 =  8 KiB   (only if n_vals ≥ 2)
//! block_vals_2  : f64 × 1024 =  8 KiB   (only if n_vals ≥ 3)
//! block_vals_3  : f64 × 1024 =  8 KiB   (only if n_vals ≥ 4)
//! block_set     : u32 × 1024 =  4 KiB
//!
//! Totals: N=1 → 20 KiB ; N=2 → 28 KiB ; N=3 → 36 KiB ; N=4 → 44 KiB.
//! All within sm_70's 48 KiB static-shared-mem budget.
//!
//! ## Code-generation note
//!
//! The PTX body is emitted by the unified, key-width-parameterised
//! [`crate::jit::partition_reduce_kernel_multi::emit_multi_kernel`] (the same
//! generator that backs the i32-key MULTI-SUM kernel). The two public
//! `compile_*` functions below validate `n_vals` and delegate with
//! `KeyWidth::I64`. See that generator's module docs for the exact i32-vs-i64
//! byte divergences (key align/width, the value-load row-offset reuse, the
//! slot/export offset-register order, the publish/probe + CLAIM key tokens).

use super::partition_reduce_kernel::KeyWidth;
use super::partition_reduce_kernel_multi::emit_multi_kernel;
use crate::error::{BoltError, BoltResult};

pub const BLOCK_GROUPS: u32 = 1024;
pub const BLOCK_THREADS: u32 = 256;
pub const MAX_VALS: u32 = 4;
pub const NUM_PARTITIONS: u32 = 4096;

pub fn kernel_entry(n_vals: u32) -> String {
    format!("bolt_partition_reduce_multi_sum_i64_{}", n_vals)
}

/// Entry-point name for the spill-counter variant.
pub fn kernel_entry_with_spill(n_vals: u32) -> String {
    format!("{}_spill", kernel_entry(n_vals))
}

pub fn compile_partition_reduce_kernel_multi_i64(n_vals: u32) -> BoltResult<String> {
    if n_vals == 0 || n_vals > MAX_VALS {
        return Err(BoltError::Other(format!(
            "partition_reduce_kernel_multi_i64: n_vals must be 1..={MAX_VALS}, got {n_vals}"
        )));
    }
    let entry = kernel_entry(n_vals);
    emit_multi_kernel(KeyWidth::I64, n_vals, false, entry.as_str())
}

/// Spill-counter-aware sibling of
/// [`compile_partition_reduce_kernel_multi_i64`]. Trailing
/// `.param .u64 spill_counter` (uint32_t*, may be null). On MAX_PROBES
/// overflow null-checks then `atom.global.add.u32` it.
pub fn compile_partition_reduce_kernel_multi_i64_with_spill(n_vals: u32) -> BoltResult<String> {
    if n_vals == 0 || n_vals > MAX_VALS {
        return Err(BoltError::Other(format!(
            "partition_reduce_kernel_multi_i64_with_spill: n_vals must be 1..={MAX_VALS}, got {n_vals}"
        )));
    }
    let entry = kernel_entry_with_spill(n_vals);
    emit_multi_kernel(KeyWidth::I64, n_vals, true, entry.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_for_all_n_vals() {
        for n in 1..=MAX_VALS {
            let ptx = compile_partition_reduce_kernel_multi_i64(n)
                .unwrap_or_else(|e| panic!("n_vals={n} should compile: {e}"));
            assert!(!ptx.is_empty());
        }
    }

    #[test]
    fn rejects_bad_n_vals() {
        assert!(compile_partition_reduce_kernel_multi_i64(0).is_err());
        assert!(compile_partition_reduce_kernel_multi_i64(MAX_VALS + 1).is_err());
    }

    #[test]
    fn uses_i64_key_loads_and_stores() {
        let ptx = compile_partition_reduce_kernel_multi_i64(2).unwrap();
        assert!(ptx.contains("ld.global.s64"));
        assert!(ptx.contains("st.global.s64"));
        assert!(ptx.contains("ld.shared.s64"));
        assert!(ptx.contains("st.shared.u64"));
    }

    #[test]
    fn emits_n_atomic_adds_per_path() {
        // CLAIM + MATCH each issue n_vals atom.shared.add.f64.
        for n in 1..=MAX_VALS {
            let ptx = compile_partition_reduce_kernel_multi_i64(n).unwrap();
            let c = ptx.matches("atom.shared.add.f64").count();
            assert_eq!(c, (2 * n) as usize, "n={n}: want {} atomics", 2 * n);
        }
    }

    // ----- _with_spill variant shape tests ---------------------------------

    #[test]
    fn with_spill_compiles_for_all_n_vals() {
        for n in 1..=MAX_VALS {
            let ptx = compile_partition_reduce_kernel_multi_i64_with_spill(n)
                .unwrap_or_else(|e| panic!("n_vals={n} should compile: {e}"));
            assert!(!ptx.is_empty());
        }
    }

    #[test]
    fn with_spill_rejects_bad_n_vals() {
        assert!(compile_partition_reduce_kernel_multi_i64_with_spill(0).is_err());
        assert!(compile_partition_reduce_kernel_multi_i64_with_spill(MAX_VALS + 1).is_err());
    }

    #[test]
    fn with_spill_has_extra_param_and_spill_atomic() {
        for n in 1..=MAX_VALS {
            let ptx = compile_partition_reduce_kernel_multi_i64_with_spill(n).unwrap();
            let expected = 4 + 2 * n as usize + 1;
            let c = ptx.matches(".param .u64 ").count();
            assert_eq!(c, expected, "n_vals={n}");
            assert!(ptx.contains("atom.global.add.u32"));
            assert!(ptx.contains("SPILL_BUMP:"));
            assert!(ptx.contains("setp.eq.u64"));
            let entry = kernel_entry_with_spill(n);
            assert!(ptx.contains(&format!(".visible .entry {entry}(")));
        }
    }
}
