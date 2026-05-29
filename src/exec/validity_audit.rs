// SPDX-License-Identifier: Apache-2.0

//! NULL / validity propagation audit + shared validity helpers.
//!
//! This module is the single place that documents, per executor / kernel,
//! HOW NULLs are handled today — whether validity bitmaps are propagated
//! natively through the GPU kernel (via a `*_with_validity` companion that
//! reads a packed-bit Arrow bitmap and guards accumulation), or whether the
//! host strips / drops NULL rows before the kernel runs. It also hosts the
//! small shared helper(s) the validity-aware launch paths reuse.
//!
//! # Why an audit lives in the tree
//!
//! Validity handling is spread across a dozen executors and three PTX
//! emitters. Without a central matrix it is easy for a new aggregate path to
//! silently regress to "drop nulls" (wrong for `COUNT(*)`) or to host-strip
//! (correct but defeats the point of GPU-native validity). Keeping the matrix
//! next to the shared helper makes the contract auditable in one read.
//!
//! # Validity bitmap wire format
//!
//! Every `*_with_validity` kernel consumes an **Arrow-LE packed-bit** bitmap:
//! bit `i % 8` of byte `i / 8` is row `i`'s validity flag (`1` = present,
//! `0` = NULL). This is bit-identical to `arrow_buffer::NullBuffer::buffer()`,
//! so a host that already holds an Arrow null buffer can upload it directly;
//! [`packed_validity_for`] (and
//! [`crate::jit::valid_flag_kernels::pack_validity_bits`]) build it from a
//! per-row bool view otherwise.
//!
//! # Propagation matrix (as of the validity-propagation completion pass)
//!
//! Legend:
//! * **native**     — validity bitmap uploaded; kernel guards accumulation
//!                    with a per-row null check (`*_with_validity` companion).
//! * **host-strip** — host filters NULL rows into a dense buffer before
//!                    upload; kernel never sees NULLs. Correct, but pays a
//!                    host pass and a copy.
//! * **host-count** — pure host computation from the Arrow null bitmap (no
//!                    GPU launch); used by COUNT(col) / AVG denominators.
//! * **drop-key**   — rows whose GROUP BY *key* is NULL are dropped (SQL
//!                    impl-defined; this engine drops them).
//! * **n/a**        — variant rejects this (op, dtype) up front.
//!
//! | Executor / path                               | SUM/MIN/MAX (int)            | SUM (float) | MIN/MAX (float)            | COUNT(expr)            | COUNT(*)   | AVG                          |
//! |-----------------------------------------------|-----------------------------|-------------|----------------------------|------------------------|------------|------------------------------|
//! | `aggregate.rs` (scalar, no GROUP BY)          | native (`bolt_reduce_with_validity`) | native      | native (tree-reduce, no atomics) | host-count            | row count  | host-strip values + GPU count |
//! | `agg_with_pre.rs` (scalar w/ pre-kernel)      | host-strip                  | host-strip  | host-strip                 | host-count (`non_null_count`) | row count | host-strip + fused count     |
//! | `groupby_valid.rs` (single-key, sentinel-free)| native (`*_valid_with_validity`) | native      | native (`valid_flag_float` `_with_validity`) | value-NULL mask + count | drop-key count | native SUM + value-NULL count |
//! | `groupby_with_pre.rs` (single-key w/ pre)     | native (`*_with_validity`)  | native      | host-strip (no float MIN/MAX companion) | host-strip count | drop-key count | host-strip + count          |
//! | `groupby.rs` (single-key, classic sentinel)   | native where companion exists; else host-strip | native | host-strip | value-NULL mask | drop-key count | mixed |
//! | `extended_agg.rs` (Bool / Utf8 scalar)        | n/a                         | n/a         | n/a                        | host-count            | row count  | n/a                          |
//! | `filter.rs` (predicate / projection)          | validity AND-tree per output via `KernelSpec::{input,output}_has_validity` (see `ptx_gen`) ||||||
//!
//! # Gaps deliberately left as host-strip
//!
//! * **Float MIN/MAX with a pre-kernel** (`groupby_with_pre.rs`): the
//!   sentinel-based classic GROUP BY routes float MIN/MAX through
//!   `float_atomics`, which has no `_with_validity` companion. The
//!   sentinel-FREE path (`groupby_valid.rs`) DOES (via
//!   [`crate::jit::valid_flag_float::compile_agg_valid_float_kernel_with_validity`]),
//!   so a plan that can use the sentinel-free executor gets native validity
//!   for float MIN/MAX; the pre-kernel classic path keeps host-strip as the
//!   correctness fallback.
//! * **Bool / Utf8 aggregates**: the primitive reduction / hash kernels
//!   reject these dtypes, so they always run the host `extended_agg` path,
//!   which is inherently NULL-aware (it walks the Arrow validity per element).
//!
//! # Invariants the matrix encodes
//!
//! 1. **COUNT(expr) is never "drop and re-count"** — it is always a count of
//!    NON-NULL rows of `expr`, computed either by a host bitmap walk
//!    (`non_null_count_for_input`) or by a validity-aware reduction over an
//!    all-ones value column (so each NULL contributes 0). `COUNT(*)` is the
//!    raw surviving-row count and ignores any column's null bitmap.
//! 2. **SUM/MIN/MAX skip NULL rows** — whether by host-strip or by the
//!    kernel folding a NULL row to the reduction identity (0 for SUM,
//!    `+inf`/`INT_MAX` for MIN, `-inf`/`INT_MIN` for MAX).
//! 3. **NULL GROUP BY keys are dropped** before the keys kernel runs, so a
//!    garbage key bit pattern never forms a spurious group.

use arrow_array::Array;

use crate::jit::valid_flag_kernels::pack_validity_bits;

/// Build the Arrow-LE packed-bit validity bitmap for `arr` (bit `i` of byte
/// `i / 8` is `1` iff row `i` is present / non-NULL), ready to upload as a
/// `GpuVec<u8>` for any `*_with_validity` kernel.
///
/// The output length is `ceil(arr.len() / 8)` bytes; trailing bits past
/// `arr.len()` in the final byte are `0` (those rows are out-of-range and the
/// kernel guards them with its `n_rows` bound regardless).
///
/// A fully-valid (`null_count() == 0`) array still produces an all-ones
/// bitmap here; callers that want to skip the upload entirely should branch
/// on `arr.null_count() > 0` BEFORE calling this (the validity-aware launch
/// paths do exactly that — the bitmap is only built and uploaded when the
/// column actually carries NULLs).
pub fn packed_validity_for(arr: &dyn Array) -> Vec<u8> {
    let n = arr.len();
    let mut bits: Vec<bool> = Vec::with_capacity(n);
    for i in 0..n {
        bits.push(!arr.is_null(i));
    }
    pack_validity_bits(&bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, Int64Array};
    use std::sync::Arc;

    /// `packed_validity_for` must match the Arrow LE bit convention: bit `i`
    /// of byte `i/8` is set iff row `i` is present.
    #[test]
    fn packed_validity_alternating_present_null() {
        // present, null, present, null, ... -> 0b0101_0101 = 0x55.
        let arr = Int32Array::from(vec![
            Some(1),
            None,
            Some(2),
            None,
            Some(3),
            None,
            Some(4),
            None,
        ]);
        let packed = packed_validity_for(&arr);
        assert_eq!(packed, vec![0x55u8]);
    }

    /// A NULL-free column packs to all-ones in every full byte, and the
    /// output length is `ceil(len/8)`.
    #[test]
    fn packed_validity_no_nulls_all_ones() {
        let arr = Int64Array::from((0..17i64).collect::<Vec<_>>());
        let packed = packed_validity_for(&arr);
        assert_eq!(packed.len(), 3); // ceil(17/8)
        assert_eq!(packed[0], 0xFF);
        assert_eq!(packed[1], 0xFF);
        assert_eq!(packed[2], 0x01); // only bit 0 (row 16) of the last byte.
    }

    /// An all-NULL column packs to all-zeros: every row is masked out.
    #[test]
    fn packed_validity_all_nulls_all_zeros() {
        let arr: Arc<Int32Array> =
            Arc::new(Int32Array::from(vec![Option::<i32>::None; 8]));
        let packed = packed_validity_for(arr.as_ref());
        assert_eq!(packed, vec![0x00u8]);
    }
}
