// SPDX-License-Identifier: Apache-2.0

//! Executor for the fully-GPU `SELECT LENGTH(<utf8_col>)` projection
//! ([`crate::plan::physical_plan::PhysicalPlan::StringLength`]).
//!
//! ## Why this is the lowest-risk GPU string path
//!
//! `LENGTH` on a dictionary-encoded `Utf8` column needs no variable-width
//! device writes. Each row already stores an `i32` dictionary key on the
//! device; the per-row byte length is a pure gather of a precomputed
//! per-dictionary-entry length table:
//!
//! ```text
//! out[row] = length_table[keys[row]]
//! ```
//!
//! The kernel is [`crate::jit::string_kernel::compile_length_gather_kernel`]
//! (fixed-width `Int32` output, no offset bookkeeping). This module owns the
//! host-side plumbing: building the length table that matches the device key
//! layout, launching the gather, downloading the `Int32` result, and widening
//! it to the `Int64` the SQL `LENGTH` contract declares.
//!
//! ## Layout-aware length tables
//!
//! Two GPU storage layouts back a `Utf8` column (see
//! [`crate::exec::gpu_table::GpuColumnData`]):
//!
//! * `Utf8 { indices, dictionary }` — engine-managed plain-string columns.
//!   Indices are 1-based: slot `0` is the NULL sentinel and `dictionary[i-1]`
//!   decodes index `i`. The length table is `[0, len(d[0]), len(d[1]), ...]`
//!   — i.e. slot `0` is `0` (NULL → 0, matching
//!   [`crate::exec::string_ops::length`]).
//!
//! * `DictUtf8 { keys, dict, valid_mask }` — native dictionary columns.
//!   Keys are 0-based into `dict` (no NULL offset); NULL rows live on
//!   `valid_mask`. The length table is `[len(d[0]), len(d[1]), ...]`.
//!
//! ## Fallback (no panic)
//!
//! The GPU gather only runs when the source column is dictionary-encoded on
//! the device AND (for the `DictUtf8` layout) carries no NULLs — a NULL row's
//! zeroed key would otherwise gather `dict[0]`'s length instead of SQL `NULL`
//! / `0`. Every other case (a `DictUtf8` column with NULLs, or any non-Utf8
//! storage) falls back to a host-side gather over downloaded keys, which
//! consults the NULL sentinel explicitly. Both paths produce the same
//! `Int64Array`.

use crate::error::{BoltError, BoltResult};
use crate::exec::gpu_table::GpuColumnData;

/// Layout of a dictionary-encoded `Utf8` column's device key buffer, which
/// determines how the per-dictionary-entry length table is indexed by the
/// gather kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyLayout {
    /// Engine-managed `Utf8` layout: 1-based keys, slot `0` is the NULL
    /// sentinel. Length table is `[0, len(d[0]), len(d[1]), ...]`.
    OneBasedNullSlot0,
    /// Native `DictUtf8` layout: 0-based keys into `dict` (no NULL offset).
    /// Length table is `[len(d[0]), len(d[1]), ...]`.
    ZeroBased,
}

/// Build the per-dictionary-entry `i32` byte-length table the gather kernel
/// indexes by device key.
///
/// For [`KeyLayout::OneBasedNullSlot0`] the table is
/// `[0, len(d[0]), len(d[1]), ...]` (slot `0` = NULL sentinel = `0` bytes,
/// matching [`crate::exec::string_ops::length`]). For [`KeyLayout::ZeroBased`]
/// the table is `[len(d[0]), len(d[1]), ...]` — one slot per dictionary entry,
/// no NULL offset.
///
/// Errors if any single string's byte length exceeds `i32::MAX` (an absurd 2
/// GiB value) — the gather kernel emits `Int32` lengths.
pub fn build_length_table(dict: &[String], layout: KeyLayout) -> BoltResult<Vec<i32>> {
    let extra = matches!(layout, KeyLayout::OneBasedNullSlot0) as usize;
    let mut table: Vec<i32> = Vec::with_capacity(dict.len() + extra);
    if extra == 1 {
        table.push(0); // NULL slot
    }
    for s in dict {
        let len = s.len();
        if len > i32::MAX as usize {
            return Err(BoltError::Other(format!(
                "LENGTH: string of {} bytes exceeds i32::MAX",
                len
            )));
        }
        table.push(len as i32);
    }
    Ok(table)
}

/// Pure host-side gather: `out[row] = length_table[keys[row]]`, returned as
/// `i64` (the SQL `LENGTH` output dtype). Mirrors the device gather so the
/// fallback path is byte-for-byte identical to a successful GPU launch on a
/// null-free input.
///
/// Bounds: a negative or out-of-range key means a kernel wrote something the
/// dictionary cannot decode — surfaced as an error rather than masked, same
/// strictness as [`crate::exec::string_ops::length`].
pub fn host_gather_lengths(keys: &[i32], length_table: &[i32]) -> BoltResult<Vec<i64>> {
    let mut out: Vec<i64> = Vec::with_capacity(keys.len());
    for &k in keys {
        if k < 0 {
            return Err(BoltError::Other(format!(
                "LENGTH: negative dictionary key {} (NULL is encoded as 0)",
                k
            )));
        }
        let len = *length_table.get(k as usize).ok_or_else(|| {
            BoltError::Other(format!(
                "LENGTH: key {} out of range (length table size {})",
                k,
                length_table.len()
            ))
        })?;
        out.push(len as i64);
    }
    Ok(out)
}

/// Decide the key layout (and whether the GPU gather is safe) for a source
/// column's device storage.
///
/// Returns `Some(layout)` when the column is dictionary-encoded and the GPU
/// gather will produce correct results, or `None` when the caller must use the
/// host fallback (a `DictUtf8` column with NULLs — whose zeroed keys would
/// gather the wrong slot — or any non-Utf8 storage).
pub fn gpu_gather_layout(data: &GpuColumnData) -> Option<KeyLayout> {
    match data {
        // Engine-managed plain-string columns: slot-0 NULL sentinel means the
        // gather is always correct (NULL rows have key 0 → length-table[0] = 0).
        GpuColumnData::Utf8 { .. } => Some(KeyLayout::OneBasedNullSlot0),
        // Native dict columns: only safe to gather on the GPU when there are no
        // NULLs. A NULL row's key is zeroed to 0, which would gather dict[0]'s
        // length instead of the NULL sentinel — route those to the host.
        GpuColumnData::DictUtf8 {
            valid_mask: None, ..
        } => Some(KeyLayout::ZeroBased),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owned(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn one_based_table_reserves_null_slot() {
        // Matches `exec::string_ops::lengths_table_pure`: slot 0 is the NULL
        // sentinel, then one slot per dict entry.
        let dict = owned(&["a", "bb", "ccc"]);
        let table = build_length_table(&dict, KeyLayout::OneBasedNullSlot0).unwrap();
        assert_eq!(table, vec![0, 1, 2, 3]);
    }

    #[test]
    fn zero_based_table_has_no_null_slot() {
        let dict = owned(&["a", "bb", "ccc"]);
        let table = build_length_table(&dict, KeyLayout::ZeroBased).unwrap();
        assert_eq!(table, vec![1, 2, 3]);
    }

    #[test]
    fn byte_length_not_char_length() {
        // "é" is two UTF-8 bytes (one character): byte semantics → 2.
        let dict = owned(&["é"]);
        assert_eq!(
            build_length_table(&dict, KeyLayout::ZeroBased).unwrap(),
            vec![2]
        );
        assert_eq!(
            build_length_table(&dict, KeyLayout::OneBasedNullSlot0).unwrap(),
            vec![0, 2]
        );
    }

    #[test]
    fn empty_string_distinct_from_null() {
        // An empty string is a real dict entry (length 0) distinct from the
        // NULL sentinel slot.
        let dict = owned(&["", "x"]);
        assert_eq!(
            build_length_table(&dict, KeyLayout::OneBasedNullSlot0).unwrap(),
            vec![0, 0, 1]
        );
        assert_eq!(
            build_length_table(&dict, KeyLayout::ZeroBased).unwrap(),
            vec![0, 1]
        );
    }

    #[test]
    fn empty_dictionary() {
        assert_eq!(
            build_length_table(&[], KeyLayout::OneBasedNullSlot0).unwrap(),
            vec![0]
        );
        assert!(build_length_table(&[], KeyLayout::ZeroBased)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn host_gather_one_based_matches_kernel_double_indirection() {
        // dict = ["us", "eu", "ca"], 1-based keys with NULL=0.
        let dict = owned(&["us", "eu", "ca"]);
        let table = build_length_table(&dict, KeyLayout::OneBasedNullSlot0).unwrap();
        // keys: NULL, "us", "ca", "eu", "us"
        let keys = vec![0, 1, 3, 2, 1];
        let got = host_gather_lengths(&keys, &table).unwrap();
        // NULL → 0, "us" → 2, "ca" → 2, "eu" → 2, "us" → 2
        assert_eq!(got, vec![0, 2, 2, 2, 2]);
    }

    #[test]
    fn host_gather_zero_based() {
        let dict = owned(&["alpha", "bb", "c"]);
        let table = build_length_table(&dict, KeyLayout::ZeroBased).unwrap();
        let keys = vec![0, 2, 1, 0];
        let got = host_gather_lengths(&keys, &table).unwrap();
        assert_eq!(got, vec![5, 1, 2, 5]);
    }

    #[test]
    fn host_gather_rejects_negative_key() {
        let table = vec![0, 1, 2];
        let err = host_gather_lengths(&[1, -1], &table).unwrap_err();
        assert!(format!("{err}").contains("negative dictionary key"));
    }

    #[test]
    fn host_gather_rejects_out_of_range_key() {
        let table = vec![0, 1];
        let err = host_gather_lengths(&[5], &table).unwrap_err();
        assert!(format!("{err}").contains("out of range"));
    }
}
