// SPDX-License-Identifier: Apache-2.0

//! Per-partition shared-memory GROUP BY **MIN / MAX** kernel — i64-key
//! variant (Tier 2.1 for two-key MIN/MAX).
//!
//! Sibling of [`crate::jit::partition_reduce_kernel_minmax`] (i32 key).
//! The i32 variant handles single-Int32-key Tier-2 MIN/MAX; this file
//! handles the two-Int32-keys-packed-into-Int64 case used by the
//! two-key Tier-2.1 path.
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
//! | Slot mapping          | `& mask` on key        | `cvt.u32.u64` low + mask|
//!
//! ## Slot mapping
//!
//! The partition kernel hashes i64 keys via Knuth's 64-bit Fibonacci
//! multiplier and takes the HIGH bits to pick the partition. Inside a
//! partition we use the LOW 32 bits of the packed key — i.e. the
//! second Int32 column — as the open-addressing slot index, exactly
//! mirroring [`crate::jit::partition_reduce_kernel_i64`] and
//! [`crate::jit::partition_reduce_kernel_count_i64`].
//!
//! ## Code sharing
//!
//! The PTX-emitting body is no longer duplicated here: both the non-spill
//! and `_with_spill` emitters delegate to the unified, key-width-parameterised
//! [`crate::jit::partition_reduce_kernel_minmax::emit_minmax_kernel`] generator
//! with `KeyWidth::I64`. Only the i64-key entry-point naming (`_keyi64`
//! suffix) lives here. See that generator for the phase-by-phase breakdown of
//! the shared scaffold vs. the key-width-branched bytes.

use crate::error::BoltResult;
use crate::jit::partition_reduce_kernel::KeyWidth;
use crate::jit::partition_reduce_kernel_minmax::{emit_minmax_kernel, MinMaxDtype, MinMaxOp};

pub const BLOCK_GROUPS: u32 = 1024;
pub const BLOCK_THREADS: u32 = 256;
pub const NUM_PARTITIONS: u32 = 4096;

/// Entry-point name for the (op, dtype) combination. Distinct from the
/// i32-key sibling's entries via the `_keyi64` suffix so both can co-exist
/// in the same CUDA context.
pub fn kernel_entry(op: MinMaxOp, dtype: MinMaxDtype) -> String {
    let opn = match op {
        MinMaxOp::Min => "min",
        MinMaxOp::Max => "max",
    };
    let dt = match dtype {
        MinMaxDtype::Int32 => "i32",
        MinMaxDtype::Int64 => "i64",
    };
    format!("bolt_partition_reduce_{}_{}_keyi64", opn, dt)
}

/// Entry-point name for the spill-counter variant.
pub fn kernel_entry_with_spill(op: MinMaxOp, dtype: MinMaxDtype) -> String {
    format!("{}_spill", kernel_entry(op, dtype))
}

/// Emit PTX for the i64-key MIN/MAX per-partition reduce kernel.
///
/// Signature:
/// ```text
/// .visible .entry <entry>(
///     .param .u64 partition_keys,     // const int64_t*
///     .param .u64 partition_vals,     // const {int32_t|int64_t}*
///     .param .u64 partition_offsets,  // const uint32_t* [K+1]
///     .param .u64 out_keys,           //       int64_t* [K*BG]
///     .param .u64 out_vals,           //       {int32_t|int64_t}* [K*BG]
///     .param .u64 out_set             //       uint8_t* [K*BG]
/// )
/// ```
///
/// Delegates to the unified key-width-parameterised generator with
/// `KeyWidth::I64`.
pub fn compile_partition_reduce_kernel_minmax_i64(
    op: MinMaxOp,
    dtype: MinMaxDtype,
) -> BoltResult<String> {
    let entry = kernel_entry(op, dtype);
    emit_minmax_kernel(KeyWidth::I64, op, /* spill = */ false, &entry, dtype)
}

/// Spill-counter-aware sibling of
/// [`compile_partition_reduce_kernel_minmax_i64`]. Same algorithm with
/// one extra `.param .u64 spill_counter` and an `atom.global.add.u32`
/// (null-checked) on probe overflow.
///
/// Delegates to the unified generator with `KeyWidth::I64` and `spill = true`.
pub fn compile_partition_reduce_kernel_minmax_i64_with_spill(
    op: MinMaxOp,
    dtype: MinMaxDtype,
) -> BoltResult<String> {
    let entry = kernel_entry_with_spill(op, dtype);
    emit_minmax_kernel(KeyWidth::I64, op, /* spill = */ true, &entry, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op_name(op: MinMaxOp) -> &'static str {
        match op {
            MinMaxOp::Min => "min",
            MinMaxOp::Max => "max",
        }
    }

    fn atom_suffix(dtype: MinMaxDtype) -> &'static str {
        match dtype {
            MinMaxDtype::Int32 => "s32",
            MinMaxDtype::Int64 => "s64",
        }
    }

    #[test]
    fn compiles_all_four_combos() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [MinMaxDtype::Int32, MinMaxDtype::Int64] {
                let ptx = compile_partition_reduce_kernel_minmax_i64(op, dt)
                    .unwrap_or_else(|e| panic!("{:?}/{:?} should compile: {e}", op, dt));
                assert!(!ptx.is_empty());
            }
        }
    }

    #[test]
    fn has_keyi64_entry_name() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [MinMaxDtype::Int32, MinMaxDtype::Int64] {
                let ptx = compile_partition_reduce_kernel_minmax_i64(op, dt).unwrap();
                let entry = kernel_entry(op, dt);
                assert!(
                    entry.ends_with("_keyi64"),
                    "i64-key entry must end with `_keyi64`, got {entry}"
                );
                let needle = format!(".visible .entry {entry}(");
                assert!(
                    ptx.contains(&needle),
                    "{:?}/{:?}: PTX missing `{needle}`",
                    op,
                    dt
                );
            }
        }
    }

    #[test]
    fn uses_i64_key_load() {
        let ptx =
            compile_partition_reduce_kernel_minmax_i64(MinMaxOp::Min, MinMaxDtype::Int32).unwrap();
        assert!(
            ptx.contains("ld.global.s64"),
            "i64-key kernel must use `ld.global.s64` for keys"
        );
        assert!(
            ptx.contains("setp.eq.s64"),
            "i64-key kernel must compare keys with `setp.eq.s64`"
        );
        assert!(
            ptx.contains("st.global.s64"),
            "i64-key kernel must store keys with `st.global.s64`"
        );
    }

    #[test]
    fn emits_expected_atomic_op() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [MinMaxDtype::Int32, MinMaxDtype::Int64] {
                let ptx = compile_partition_reduce_kernel_minmax_i64(op, dt).unwrap();
                let want = format!("atom.shared.{}.{}", op_name(op), atom_suffix(dt));
                assert!(
                    ptx.contains(&want),
                    "{:?}/{:?}: missing `{want}` in emitted PTX",
                    op,
                    dt
                );
            }
        }
    }

    // ----- _with_spill variant shape tests ---------------------------------

    #[test]
    fn with_spill_compiles_all_four_combos() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [MinMaxDtype::Int32, MinMaxDtype::Int64] {
                let ptx = compile_partition_reduce_kernel_minmax_i64_with_spill(op, dt)
                    .unwrap_or_else(|e| panic!("{:?}/{:?}: {e}", op, dt));
                assert!(!ptx.is_empty());
            }
        }
    }

    #[test]
    fn with_spill_distinct_entry_and_spill_atomic() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [MinMaxDtype::Int32, MinMaxDtype::Int64] {
                let base = kernel_entry(op, dt);
                let spill = kernel_entry_with_spill(op, dt);
                assert_ne!(base, spill);
                assert!(spill.ends_with("_spill"));
                let ptx = compile_partition_reduce_kernel_minmax_i64_with_spill(op, dt).unwrap();
                let needle = format!(".visible .entry {spill}(");
                assert!(ptx.contains(&needle));
                assert!(ptx.contains("atom.global.add.u32"));
                assert!(ptx.contains("SPILL_BUMP:"));
                assert!(ptx.contains("setp.eq.u64"));
            }
        }
    }
}
