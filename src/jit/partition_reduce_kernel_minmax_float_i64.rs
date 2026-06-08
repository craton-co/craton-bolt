// SPDX-License-Identifier: Apache-2.0

//! Per-partition shared-memory **float MIN / MAX** kernel — i64-key
//! variant (Tier 2.1 for two-key float MIN/MAX).
//!
//! Sibling of [`crate::jit::partition_reduce_kernel_minmax_float`] (i32
//! key). Adapts the same CAS-loop atomic pattern (PTX has no native
//! `atom.shared.{min,max}.f{32,64}` on sm_70) to the i64-packed-two-key
//! layout used by the two-key Tier-2.1 pipeline.
//!
//! ## Layout differences from the i32-key variant
//!
//! | What                  | i32-key variant        | i64-key variant         |
//! | --------------------- | ---------------------- | ----------------------- |
//! | `block_keys` slot     | 4 B                    | **8 B**                 |
//! | `block_keys_buf`      | 4 KiB (1024 × 4 B)     | **8 KiB** (1024 × 8 B)  |
//! | Key load (rows)       | `ld.global.s32`        | `ld.global.s64`         |
//! | Key compare           | `setp.eq.s32`          | `setp.eq.s64`           |
//! | Key store (shared)    | `st.shared.u32`        | `st.shared.u64`         |
//! | Output key store      | `st.global.s32`        | `st.global.s64`         |
//! | Slot mapping          | `& mask` on i32 key    | `cvt.u32.u64` + mask    |
//!
//! The CAS-loop body itself operates on the float value bits and is
//! identical to the i32-key sibling.

use crate::error::BoltResult;
use crate::jit::partition_reduce_kernel::KeyWidth;
use crate::jit::partition_reduce_kernel_minmax::MinMaxOp;
use crate::jit::partition_reduce_kernel_minmax_float::FloatDtype;

pub const BLOCK_GROUPS: u32 = 1024;
pub const BLOCK_THREADS: u32 = 256;
pub const NUM_PARTITIONS: u32 = 4096;

/// Entry-point name for the (op, dtype) combination. Distinct from the
/// i32-key sibling via the `_keyi64` suffix.
pub fn kernel_entry(op: MinMaxOp, dtype: FloatDtype) -> String {
    let opn = match op {
        MinMaxOp::Min => "min",
        MinMaxOp::Max => "max",
    };
    let dt = match dtype {
        FloatDtype::Float32 => "f32",
        FloatDtype::Float64 => "f64",
    };
    format!("bolt_partition_reduce_{}_{}_keyi64", opn, dt)
}

/// Entry-point name for the spill-counter variant.
pub fn kernel_entry_with_spill(op: MinMaxOp, dtype: FloatDtype) -> String {
    format!("{}_spill", kernel_entry(op, dtype))
}

/// CAS-suffix token (`b32` / `b64`) for the float value dtype. Retained for
/// the in-file shape tests; the actual emission lives in the shared generator.
// Used only by the #[cfg(test)] shape tests below (see `uses_atom_shared_cas`);
// mirrors `partition_reduce_kernel_minmax_float::FloatDtype::ptx_cas_suffix`.
#[allow(dead_code)]
fn ptx_cas_suffix(dtype: FloatDtype) -> &'static str {
    match dtype {
        FloatDtype::Float32 => "b32",
        FloatDtype::Float64 => "b64",
    }
}

pub fn compile_partition_reduce_kernel_minmax_float_i64(
    op: MinMaxOp,
    dtype: FloatDtype,
) -> BoltResult<String> {
    let entry = kernel_entry(op, dtype);
    super::partition_reduce_kernel_minmax_float::emit_minmax_float_kernel(
        KeyWidth::I64,
        op,
        dtype,
        entry.as_str(),
        "",
        false,
    )
}

/// Spill-counter-aware sibling. Identical to
/// [`compile_partition_reduce_kernel_minmax_float_i64`] with one extra
/// `.param .u64 spill_counter` (uint32_t*, may be null). On MAX_PROBES
/// overflow null-checks then `atom.global.add.u32` it.
pub fn compile_partition_reduce_kernel_minmax_float_i64_with_spill(
    op: MinMaxOp,
    dtype: FloatDtype,
) -> BoltResult<String> {
    let entry = kernel_entry_with_spill(op, dtype);
    super::partition_reduce_kernel_minmax_float::emit_minmax_float_kernel(
        KeyWidth::I64,
        op,
        dtype,
        entry.as_str(),
        "_sp",
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_all_four_combos() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [FloatDtype::Float32, FloatDtype::Float64] {
                let ptx = compile_partition_reduce_kernel_minmax_float_i64(op, dt)
                    .unwrap_or_else(|e| panic!("{:?}/{:?}: {e}", op, dt));
                assert!(!ptx.is_empty());
            }
        }
    }

    #[test]
    fn has_keyi64_entry_name() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [FloatDtype::Float32, FloatDtype::Float64] {
                let entry = kernel_entry(op, dt);
                assert!(
                    entry.ends_with("_keyi64"),
                    "i64-key entry must end with `_keyi64`, got {entry}"
                );
                let ptx = compile_partition_reduce_kernel_minmax_float_i64(op, dt).unwrap();
                let needle = format!(".visible .entry {entry}(");
                assert!(ptx.contains(&needle), "PTX missing `{needle}`");
            }
        }
    }

    #[test]
    fn uses_i64_key_load_and_compare() {
        let ptx =
            compile_partition_reduce_kernel_minmax_float_i64(MinMaxOp::Min, FloatDtype::Float64)
                .unwrap();
        assert!(
            ptx.contains("ld.global.s64"),
            "i64-key float kernel must use `ld.global.s64` for keys"
        );
        assert!(
            ptx.contains("setp.eq.s64"),
            "i64-key float kernel must use `setp.eq.s64`"
        );
        assert!(
            ptx.contains("st.global.s64"),
            "i64-key float kernel must store keys as s64"
        );
    }

    #[test]
    fn uses_atom_shared_cas() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [FloatDtype::Float32, FloatDtype::Float64] {
                let ptx = compile_partition_reduce_kernel_minmax_float_i64(op, dt).unwrap();
                let want = format!("atom.shared.cas.{}", ptx_cas_suffix(dt));
                assert!(
                    ptx.contains(&want),
                    "{:?}/{:?}: PTX missing `{want}`",
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
            for dt in [FloatDtype::Float32, FloatDtype::Float64] {
                let ptx = compile_partition_reduce_kernel_minmax_float_i64_with_spill(op, dt)
                    .unwrap_or_else(|e| panic!("{:?}/{:?}: {e}", op, dt));
                assert!(!ptx.is_empty());
            }
        }
    }

    #[test]
    fn with_spill_has_distinct_entry_and_spill_atomic() {
        for op in [MinMaxOp::Min, MinMaxOp::Max] {
            for dt in [FloatDtype::Float32, FloatDtype::Float64] {
                let base = kernel_entry(op, dt);
                let spill = kernel_entry_with_spill(op, dt);
                assert_ne!(base, spill);
                assert!(spill.ends_with("_spill"));
                let ptx =
                    compile_partition_reduce_kernel_minmax_float_i64_with_spill(op, dt).unwrap();
                let needle = format!(".visible .entry {spill}(");
                assert!(ptx.contains(&needle));
                assert!(ptx.contains("atom.global.add.u32"));
                assert!(ptx.contains("SPILL_BUMP:"));
                assert!(ptx.contains("setp.eq.u64"));
            }
        }
    }
}
