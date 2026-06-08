// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for the **adjacent-distinct flag** kernel used by the
//! sort-based GPU DISTINCT path.
//!
//! Backs `crate::exec::distinct`'s GPU dedup strategy. The flow there is:
//!
//! ```text
//!  key column (host)
//!     │
//!     ▼  reuse crate::exec::gpu_sort (bitonic / radix)
//!  sorted key column (device, n_rows)   sorted indices (device)
//!     │
//!     ▼  THIS kernel: keep[i] = (i == 0) || (key[i] != key[i-1])
//!  keep mask (device, u8 per row)
//!     │
//!     ▼  reuse crate::exec::gpu_compact (prefix-scan + gather)
//!  deduped survivors
//! ```
//!
//! After a sort, equal keys are adjacent, so a single linear pass that
//! compares each row to its predecessor identifies the first occurrence of
//! every distinct value: a row is *kept* iff it is the first row (`tid == 0`)
//! or its key differs from the immediately preceding row's key. Every kept
//! row is the unique representative of its run of equal keys, exactly the SQL
//! `DISTINCT` semantics. The output mask plugs straight into the existing
//! prefix-scan + gather compaction kernels (`crate::jit::prefix_scan` /
//! `crate::exec::gpu_compact`), which is why we emit a `u8`-per-row mask in
//! the same `1 = keep / 0 = drop` convention the scan kernel consumes.
//!
//! ## NULL semantics (SQL DISTINCT: NULLs are equal to each other)
//!
//! The host sorts NULLs to one contiguous end of the column (via the sort
//! path's per-key validity bitmap + `nulls_first`), so all NULL rows form one
//! adjacent run. The kernel takes a packed-bit validity buffer (Arrow-style,
//! LSB-first; bit `i` of byte `i >> 3`, `1 = valid / 0 = NULL`) and applies:
//!
//!   * `tid == 0`                       → keep (first row, always a new run).
//!   * `self NULL` && `prev NULL`       → drop (two NULLs are equal → collapse).
//!   * `self NULL` xor `prev NULL`      → keep (NULL vs non-NULL is a new run).
//!   * both non-NULL && `key != prev`   → keep.
//!   * both non-NULL && `key == prev`   → drop.
//!
//! When the column is known NULL-free the host passes a null validity pointer
//! and we emit the value-only comparison (no validity loads). This mirrors the
//! `nullable` fast-path skip in `crate::jit::sort_kernel::emit_key_compare`.
//!
//! ## Float canonicalisation (deferred to the host)
//!
//! `-0.0 == +0.0` and "all NaN collapse to one DISTINCT row" are handled by
//! the host *before* upload, by canonicalising the key buffer
//! (`crate::exec::distinct::canonicalise_f32` / `canonicalise_f64`). After
//! canonicalisation the device comparison is a plain bit/value `setp.eq`, so
//! the float keys reuse the same `setp.eq.<ty>` the integer keys use and the
//! NaN/`-0.0` equivalence classes are already folded. Keeping the
//! canonicalisation host-side means the kernel stays branch-free of IEEE
//! special-case handling and the equivalence relation matches the host
//! fallback byte-for-byte.
//!
//! ## ABI
//!
//! ```text
//! .visible .entry bolt_distinct_flag_<dtype>[_v](
//!     .param .u64 keys_ptr,        // sorted key values, length n_rows
//!     .param .u64 validity_ptr,    // packed-bit validity (only if nullable)
//!     .param .u64 mask_ptr,        // u8 output, length n_rows (1 = keep)
//!     .param .u32 n_rows
//! )
//! ```
//!
//! Grid: 1D, one thread per row, block size 256 (matches the engine-wide
//! `BLOCK_SIZE` so occupancy tuning stays uniform).

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::DataType;

/// PTX target metadata baked into every emitted module. Matches the rest of
/// the JIT pipeline (see `sort_kernel.rs`, `scan_kernel.rs`).
const PTX_VERSION: &str = ".version 7.5";
/// Target SM architecture string.
const PTX_TARGET: &str = ".target sm_70";
/// Address size directive (we always use 64-bit pointers).
const PTX_ADDRESS_SIZE: &str = ".address_size 64";

/// Threads per block for the flag launch. Matches `BLOCK_SIZE` elsewhere so
/// occupancy tuning stays uniform across the engine's kernels.
pub const DISTINCT_FLAG_BLOCK_SIZE: u32 = 256;

/// Per-dtype PTX details for the adjacent-distinct comparison: the
/// `ld.global.<suffix>` mnemonic, the register class name, the `.reg`
/// declaration type, the `setp.eq.<ty>` mnemonic, and the element byte width.
///
/// Float keys are compared with `setp.eq.f32`/`f64` — after the host's
/// `-0.0`/NaN canonicalisation (see the module doc) the IEEE equality
/// `+0.0 == -0.0` is exactly the DISTINCT equivalence we want, and every NaN
/// has already been folded to one canonical bit pattern so two canonical NaNs
/// compare equal under `setp.eq.f`. Bool reuses the i32 path (`ld.global.u8`
/// zero-extends into a b32 register).
struct FlagFlavour {
    /// `ld.global.<suffix>` mnemonic for the key load.
    ld_suffix: &'static str,
    /// Register class prefix (e.g. `"rk"` → `%rk0`).
    reg_class: &'static str,
    /// `.reg .<ty>` declaration type for the value register pool.
    reg_decl_ty: &'static str,
    /// `setp.eq.<ty>` mnemonic for the self-vs-prev compare.
    setp_eq: &'static str,
    /// Element byte width on the device.
    byte_width: u32,
}

impl FlagFlavour {
    fn for_dtype(dtype: DataType) -> BoltResult<Self> {
        Ok(match dtype {
            DataType::Int32 => Self {
                ld_suffix: "s32",
                reg_class: "rk",
                reg_decl_ty: "b32",
                setp_eq: "setp.eq.s32",
                byte_width: 4,
            },
            DataType::Int64 => Self {
                ld_suffix: "s64",
                reg_class: "rkl",
                reg_decl_ty: "b64",
                setp_eq: "setp.eq.s64",
                byte_width: 8,
            },
            DataType::Float32 => Self {
                ld_suffix: "f32",
                reg_class: "rkf",
                reg_decl_ty: "f32",
                setp_eq: "setp.eq.f32",
                byte_width: 4,
            },
            DataType::Float64 => Self {
                ld_suffix: "f64",
                reg_class: "rkd",
                reg_decl_ty: "f64",
                setp_eq: "setp.eq.f64",
                byte_width: 8,
            },
            // Bool keys are widened to b32 via ld.global.u8 (zero-extend) and
            // compared as s32, mirroring sort_kernel's Bool handling.
            DataType::Bool => Self {
                ld_suffix: "u8",
                reg_class: "rk",
                reg_decl_ty: "b32",
                setp_eq: "setp.eq.s32",
                byte_width: 1,
            },
            other => {
                return Err(BoltError::Other(format!(
                    "distinct_kernel: dtype {:?} not supported \
                     (Int32/Int64/Float32/Float64/Bool — Utf8 / wide multi-key \
                     fall back to the host DISTINCT path)",
                    other
                )))
            }
        })
    }
}

/// Stable, content-addressed entry-point name for the flag kernel of a given
/// `(dtype, nullable)`. The `_v` suffix marks the validity-aware variant so
/// the module cache never confuses the two ABIs (one extra pointer param).
pub fn distinct_flag_entry(dtype: DataType, nullable: bool) -> BoltResult<String> {
    let dty = match dtype {
        DataType::Int32 => "i32",
        DataType::Int64 => "i64",
        DataType::Float32 => "f32",
        DataType::Float64 => "f64",
        DataType::Bool => "b",
        other => {
            return Err(BoltError::Other(format!(
                "distinct_kernel: dtype {:?} not supported",
                other
            )))
        }
    };
    Ok(if nullable {
        format!("bolt_distinct_flag_{dty}_v")
    } else {
        format!("bolt_distinct_flag_{dty}")
    })
}

/// Compile the adjacent-distinct flag kernel PTX for `(dtype, nullable)`.
///
/// See the module doc for the ABI and the keep/drop decision table. The
/// emitted kernel writes one `u8` (`1 = keep`, `0 = drop`) per row into the
/// mask buffer, which the prefix-scan + gather compaction consumes unchanged.
pub fn compile_distinct_flag_kernel(dtype: DataType, nullable: bool) -> BoltResult<String> {
    let flavour = FlagFlavour::for_dtype(dtype)?;
    let entry = distinct_flag_entry(dtype, nullable)?;
    let bw = flavour.byte_width;

    let mut p = String::new();
    writeln!(p, "{PTX_VERSION}").map_err(write_err)?;
    writeln!(p, "{PTX_TARGET}").map_err(write_err)?;
    writeln!(p, "{PTX_ADDRESS_SIZE}").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    // -- Signature ----------------------------------------------------
    // Param layout: keys, [validity], mask, n_rows.
    writeln!(p, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_0,").map_err(write_err)?; // keys
    let (mask_param, nrows_param) = if nullable {
        writeln!(p, "\t.param .u64 {entry}_param_1,").map_err(write_err)?; // validity
        writeln!(p, "\t.param .u64 {entry}_param_2,").map_err(write_err)?; // mask
        writeln!(p, "\t.param .u32 {entry}_param_3").map_err(write_err)?; // n_rows
        (2usize, 3usize)
    } else {
        writeln!(p, "\t.param .u64 {entry}_param_1,").map_err(write_err)?; // mask
        writeln!(p, "\t.param .u32 {entry}_param_2").map_err(write_err)?; // n_rows
        (1usize, 2usize)
    };
    writeln!(p, ")").map_err(write_err)?;
    writeln!(p, "{{").map_err(write_err)?;

    // -- Register declarations ---------------------------------------
    writeln!(p, "\t.reg .pred %p<8>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32 %r<24>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64 %rd<24>;").map_err(write_err)?;
    writeln!(
        p,
        "\t.reg .{ty} %{rc}<4>;",
        ty = flavour.reg_decl_ty,
        rc = flavour.reg_class
    )
    .map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    // -- tid -----------------------------------------------------------
    writeln!(p, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?; // %r3 = tid

    // OOB guard: if tid >= n_rows, return.
    writeln!(p, "\tld.param.u32 %r4, [{entry}_param_{nrows_param}];").map_err(write_err)?;
    writeln!(p, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(p, "\t@%p0 bra DONE;").map_err(write_err)?;

    // mask base pointer (global).
    writeln!(p, "\tld.param.u64 %rd0, [{entry}_param_{mask_param}];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    // mask address for this row: mask + tid (u8 stride).
    writeln!(p, "\tmul.wide.u32 %rd1, %r3, 1;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd1, %rd0, %rd1;").map_err(write_err)?;

    // tid == 0 -> always keep (first row of the sorted column begins a run).
    writeln!(p, "\tsetp.eq.s32 %p1, %r3, 0;").map_err(write_err)?;
    writeln!(p, "\t@%p1 bra KEEP;").map_err(write_err)?;

    // prev = tid - 1.
    writeln!(p, "\tsub.s32 %r5, %r3, 1;").map_err(write_err)?;

    if nullable {
        // Load self/prev validity bits from the packed-bit buffer.
        // self_valid -> %r10, prev_valid -> %r11.
        writeln!(p, "\tld.param.u64 %rd10, [{entry}_param_1];").map_err(write_err)?;
        writeln!(p, "\tcvta.to.global.u64 %rd10, %rd10;").map_err(write_err)?;
        // self bit: byte = tid>>3, bit = tid&7.
        writeln!(p, "\tshr.u32 %r12, %r3, 3;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r13, %r3, 7;").map_err(write_err)?;
        writeln!(p, "\tmul.wide.u32 %rd11, %r12, 1;").map_err(write_err)?;
        writeln!(p, "\tadd.s64 %rd11, %rd10, %rd11;").map_err(write_err)?;
        writeln!(p, "\tld.global.u8 %r10, [%rd11];").map_err(write_err)?;
        writeln!(p, "\tshr.u32 %r10, %r10, %r13;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r10, %r10, 1;").map_err(write_err)?; // %r10 = self_valid
                                                                     // prev bit: byte = prev>>3, bit = prev&7.
        writeln!(p, "\tshr.u32 %r14, %r5, 3;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r15, %r5, 7;").map_err(write_err)?;
        writeln!(p, "\tmul.wide.u32 %rd12, %r14, 1;").map_err(write_err)?;
        writeln!(p, "\tadd.s64 %rd12, %rd10, %rd12;").map_err(write_err)?;
        writeln!(p, "\tld.global.u8 %r11, [%rd12];").map_err(write_err)?;
        writeln!(p, "\tshr.u32 %r11, %r11, %r15;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r11, %r11, 1;").map_err(write_err)?; // %r11 = prev_valid

        // both NULL (self_valid==0 && prev_valid==0) -> equal -> DROP.
        writeln!(p, "\tor.b32 %r16, %r10, %r11;").map_err(write_err)?;
        writeln!(p, "\tsetp.eq.s32 %p2, %r16, 0;").map_err(write_err)?;
        writeln!(p, "\t@%p2 bra DROP;").map_err(write_err)?;
        // exactly one NULL (self_valid != prev_valid) -> new run -> KEEP.
        writeln!(p, "\tsetp.ne.s32 %p3, %r10, %r11;").map_err(write_err)?;
        writeln!(p, "\t@%p3 bra KEEP;").map_err(write_err)?;
        // else: both non-NULL -> fall through to value compare.
    }

    // Value compare: load key[tid] and key[prev].
    writeln!(p, "\tld.param.u64 %rd2, [{entry}_param_0];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;
    // self addr = keys + tid*bw.
    writeln!(p, "\tmul.wide.u32 %rd3, %r3, {bw};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd3, %rd2, %rd3;").map_err(write_err)?;
    // prev addr = keys + prev*bw.
    writeln!(p, "\tmul.wide.u32 %rd4, %r5, {bw};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd4, %rd2, %rd4;").map_err(write_err)?;
    writeln!(
        p,
        "\tld.global.{ld} %{rc}0, [%rd3];",
        ld = flavour.ld_suffix,
        rc = flavour.reg_class
    )
    .map_err(write_err)?;
    writeln!(
        p,
        "\tld.global.{ld} %{rc}1, [%rd4];",
        ld = flavour.ld_suffix,
        rc = flavour.reg_class
    )
    .map_err(write_err)?;
    // equal -> DROP, else fall through to KEEP.
    writeln!(
        p,
        "\t{eq} %p4, %{rc}0, %{rc}1;",
        eq = flavour.setp_eq,
        rc = flavour.reg_class
    )
    .map_err(write_err)?;
    writeln!(p, "\t@%p4 bra DROP;").map_err(write_err)?;

    // KEEP: store 1.
    writeln!(p, "KEEP:").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r20, 1;").map_err(write_err)?;
    writeln!(p, "\tst.global.u8 [%rd1], %r20;").map_err(write_err)?;
    writeln!(p, "\tbra DONE;").map_err(write_err)?;

    // DROP: store 0.
    writeln!(p, "DROP:").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r21, 0;").map_err(write_err)?;
    writeln!(p, "\tst.global.u8 [%rd1], %r21;").map_err(write_err)?;

    writeln!(p, "DONE:").map_err(write_err)?;
    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;

    Ok(p)
}

/// Adapt a `std::fmt::Error` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("distinct_kernel: write failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_names_are_distinct_per_dtype_and_nullability() {
        assert_eq!(
            distinct_flag_entry(DataType::Int32, false).unwrap(),
            "bolt_distinct_flag_i32"
        );
        assert_eq!(
            distinct_flag_entry(DataType::Int32, true).unwrap(),
            "bolt_distinct_flag_i32_v"
        );
        assert_eq!(
            distinct_flag_entry(DataType::Int64, false).unwrap(),
            "bolt_distinct_flag_i64"
        );
        assert_eq!(
            distinct_flag_entry(DataType::Float64, true).unwrap(),
            "bolt_distinct_flag_f64_v"
        );
        assert_eq!(
            distinct_flag_entry(DataType::Bool, false).unwrap(),
            "bolt_distinct_flag_b"
        );
    }

    #[test]
    fn unsupported_dtype_errors() {
        assert!(compile_distinct_flag_kernel(DataType::Utf8, false).is_err());
        assert!(distinct_flag_entry(DataType::Utf8, false).is_err());
    }

    #[test]
    fn i32_non_nullable_shape() {
        let ptx = compile_distinct_flag_kernel(DataType::Int32, false).unwrap();
        // Header + entry.
        assert!(ptx.contains(".version 7.5"), "{ptx}");
        assert!(ptx.contains(".target sm_70"), "{ptx}");
        assert!(
            ptx.contains(".visible .entry bolt_distinct_flag_i32("),
            "{ptx}"
        );
        // Non-nullable ABI: keys, mask, n_rows = 3 params, no _param_3.
        assert!(ptx.contains("bolt_distinct_flag_i32_param_2"), "{ptx}");
        assert!(!ptx.contains("bolt_distinct_flag_i32_param_3"), "{ptx}");
        // tid==0 always-keep branch + the adjacent compare.
        assert!(ptx.contains("setp.eq.s32 %p1, %r3, 0;"), "{ptx}");
        assert!(ptx.contains("setp.eq.s32 %p4"), "{ptx}");
        // Mask store of both 1 (keep) and 0 (drop).
        assert!(ptx.contains("st.global.u8 [%rd1], %r20;"), "{ptx}");
        assert!(ptx.contains("st.global.u8 [%rd1], %r21;"), "{ptx}");
        // i32 key load.
        assert!(ptx.contains("ld.global.s32"), "{ptx}");
        // No validity loads in the non-nullable variant.
        assert!(!ptx.contains("self_valid"), "{ptx}");
    }

    #[test]
    fn i64_nullable_shape_has_validity_loads() {
        let ptx = compile_distinct_flag_kernel(DataType::Int64, true).unwrap();
        assert!(
            ptx.contains(".visible .entry bolt_distinct_flag_i64_v("),
            "{ptx}"
        );
        // Nullable ABI has a 4th param (n_rows at _param_3).
        assert!(ptx.contains("bolt_distinct_flag_i64_v_param_3"), "{ptx}");
        // Packed-bit validity: byte load + shift + mask.
        assert!(ptx.contains("ld.global.u8 %r10"), "{ptx}");
        assert!(ptx.contains("ld.global.u8 %r11"), "{ptx}");
        // both-NULL collapse and one-NULL keep branches.
        assert!(ptx.contains("@%p2 bra DROP;"), "{ptx}");
        assert!(ptx.contains("@%p3 bra KEEP;"), "{ptx}");
        // i64 key compare.
        assert!(ptx.contains("setp.eq.s64"), "{ptx}");
        assert!(ptx.contains("ld.global.s64"), "{ptx}");
    }

    #[test]
    fn f64_uses_float_eq_after_host_canonicalisation() {
        let ptx = compile_distinct_flag_kernel(DataType::Float64, false).unwrap();
        // Float keys compare with setp.eq.f64 — the host has already folded
        // -0.0 and all-NaN into canonical bit patterns before upload.
        assert!(ptx.contains("setp.eq.f64"), "{ptx}");
        assert!(ptx.contains("ld.global.f64"), "{ptx}");
    }

    #[test]
    fn bool_widens_to_b32_via_u8_load() {
        let ptx = compile_distinct_flag_kernel(DataType::Bool, false).unwrap();
        assert!(
            ptx.contains(".visible .entry bolt_distinct_flag_b("),
            "{ptx}"
        );
        // Bool loads a byte and compares as s32.
        assert!(ptx.contains("ld.global.u8 %rk0"), "{ptx}");
        assert!(ptx.contains("setp.eq.s32 %p4"), "{ptx}");
    }
}
