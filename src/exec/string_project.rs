// SPDX-License-Identifier: Apache-2.0

//! Executor helpers for the GPU variable-width string projection
//! ([`crate::plan::physical_plan::PhysicalPlan::StringProject`]) — currently
//! `UPPER` / `LOWER` over a `Utf8` column.
//!
//! ## Why this is the second GPU string path (after `LENGTH`)
//!
//! `LENGTH` (see [`crate::exec::string_length`]) is fixed-output-width: each
//! row's result is a 4-byte `Int32`, a pure dictionary-index gather. `UPPER` /
//! `LOWER` instead produce a brand-new `Utf8` array whose per-row byte widths
//! are data-dependent, so they need the classic GPU two-pass pattern wired in
//! [`crate::jit::string_kernel`]:
//!
//! 1. **Length pass** ([`crate::jit::string_kernel::compile_varwidth_len_pass`]):
//!    one thread per row reads its input slice and writes the output byte length
//!    into a `u32` `row_lens` buffer. For `UPPER`/`LOWER` ASCII case folding is
//!    length-preserving, so the output length equals the input length.
//! 2. **Exclusive scan** of `row_lens` → output `offsets` (`n_rows + 1`
//!    entries) and the grand total (= the output `bytes` buffer size). The
//!    two-pass kernel contract documents this as a host-side step; we keep it on
//!    the host (one d→h copy of `row_lens`, an O(n) scan, one h→d copy of the
//!    offsets) exactly like the LENGTH path's download → host-map → upload
//!    shape. A future revision can swap in the device prefix-scan from
//!    [`crate::jit::prefix_scan`].
//! 3. **Write pass** ([`crate::jit::string_kernel::compile_varwidth_write_pass`]):
//!    one thread per row copies / case-folds its input slice into
//!    `out_bytes[out_offsets[tid] ..]`.
//!
//! ## Source-slice layout
//!
//! The two-pass kernels consume Arrow-`Utf8`-shaped `src_offsets` (`i32`,
//! `n_rows + 1`) + `src_bytes` (`u8`) buffers — one contiguous slice per row.
//! GPU `Utf8` columns are dictionary-encoded (per-row `i32` keys + a host
//! dictionary), so this module materialises the row-aligned offsets+bytes from
//! the decoded column ([`build_row_aligned_input`]) and uploads them before the
//! length pass. NULL rows decode to an empty slice (the dictionary's NULL
//! sentinel), matching the host fallback.
//!
//! ## Fallback (no panic)
//!
//! The GPU two-pass path is taken only when the transform is ASCII-safe for the
//! column's dictionary (the kernels case-fold byte-wise, which is correct for
//! ASCII but NOT for arbitrary Unicode — e.g. `'ß'.to_uppercase()` is `"SS"`,
//! changing the byte length). When any dictionary entry contains a non-ASCII
//! byte the executor falls back to the host transform
//! ([`host_transform_strings`]), which uses full Unicode `to_uppercase` /
//! `to_lowercase` via [`crate::exec::string_ops`]. Both paths produce the same
//! `StringArray` for ASCII data.

use arrow_array::StringArray;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::ScalarFnKind;

/// The variable-width string transform a [`crate::plan::physical_plan::PhysicalPlan::StringProject`]
/// output applies. Mirrors the subset of [`ScalarFnKind`] that this executor
/// supports on the GPU two-pass path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringTransform {
    /// `UPPER(s)` — ASCII upper-case on the GPU; full Unicode on the host
    /// fallback.
    Upper,
    /// `LOWER(s)` — ASCII lower-case on the GPU; full Unicode on the host
    /// fallback.
    Lower,
}

impl StringTransform {
    /// Map to the JIT [`ScalarFnKind`] whose two-pass kernels implement this
    /// transform.
    pub fn scalar_fn_kind(self) -> ScalarFnKind {
        match self {
            StringTransform::Upper => ScalarFnKind::Upper,
            StringTransform::Lower => ScalarFnKind::Lower,
        }
    }

    /// Apply the **full-Unicode** host transform to a single string (the host
    /// fallback). Matches `str::to_uppercase` / `str::to_lowercase`.
    pub fn apply_host(self, s: &str) -> String {
        match self {
            StringTransform::Upper => s.to_uppercase(),
            StringTransform::Lower => s.to_lowercase(),
        }
    }

    /// Apply the **ASCII-only, byte-wise** transform to a single byte, exactly
    /// mirroring the per-byte fold the GPU write-pass emits
    /// ([`crate::jit::string_kernel::compile_varwidth_write_pass`]):
    ///
    /// * `UPPER`: `if b'a' <= b <= b'z' { b - 32 }`.
    /// * `LOWER`: `if b'A' <= b <= b'Z' { b + 32 }`.
    ///
    /// Used by the host mirror of the GPU path ([`gpu_path_transform_pure`]) so
    /// the fallback-vs-GPU equivalence is unit-testable without a CUDA runtime.
    pub fn apply_ascii_byte(self, b: u8) -> u8 {
        match self {
            StringTransform::Upper => {
                if b.is_ascii_lowercase() {
                    b - 32
                } else {
                    b
                }
            }
            StringTransform::Lower => {
                if b.is_ascii_uppercase() {
                    b + 32
                } else {
                    b
                }
            }
        }
    }
}

/// Layout of a dictionary-encoded `Utf8` column's device key buffer.
///
/// Re-declared here (rather than reused from [`crate::exec::string_length`]) to
/// keep the two string executors decoupled; the meaning is identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyLayout {
    /// Engine-managed `Utf8` layout: 1-based keys, slot `0` is the NULL
    /// sentinel (`dictionary[key - 1]` decodes a row; key `0` → empty/NULL).
    OneBasedNullSlot0,
    /// Native `DictUtf8` layout: 0-based keys into `dict` (no NULL offset).
    /// NULL rows are tracked on a separate validity bitmap, not in the key.
    ZeroBased,
}

/// Decode one row's source string from a dictionary-encoded column, honouring
/// the key layout and (for the `ZeroBased`/`DictUtf8` layout) an optional
/// validity bit.
///
/// Returns an empty string for NULL rows — the row-aligned input the GPU
/// two-pass kernels consume has no validity channel, so a NULL row produces an
/// empty output slice (matching the host fallback's treatment of NULL → empty).
fn decode_row<'a>(
    dict: &'a [String],
    key: i32,
    layout: KeyLayout,
    is_valid: bool,
) -> BoltResult<&'a str> {
    if !is_valid {
        return Ok("");
    }
    match layout {
        KeyLayout::OneBasedNullSlot0 => {
            if key == 0 {
                Ok("") // NULL sentinel
            } else if key < 0 {
                Err(BoltError::Other(format!(
                    "StringProject: negative dictionary key {key}"
                )))
            } else {
                dict.get((key - 1) as usize).map(String::as_str).ok_or_else(|| {
                    BoltError::Other(format!(
                        "StringProject: key {key} out of range (dict size {})",
                        dict.len()
                    ))
                })
            }
        }
        KeyLayout::ZeroBased => {
            if key < 0 {
                Err(BoltError::Other(format!(
                    "StringProject: negative dictionary key {key}"
                )))
            } else {
                dict.get(key as usize).map(String::as_str).ok_or_else(|| {
                    BoltError::Other(format!(
                        "StringProject: key {key} out of range (dict size {})",
                        dict.len()
                    ))
                })
            }
        }
    }
}

/// Materialise the row-aligned Arrow-`Utf8`-shaped (`offsets`, `bytes`) input
/// the GPU two-pass kernels consume, by decoding each row's string from the
/// dictionary-encoded column.
///
/// * `offsets` has `keys.len() + 1` `i32` entries; `offsets[r] .. offsets[r+1]`
///   is row `r`'s byte slice in `bytes`.
/// * `validity[r]` (when `Some`) gates NULL rows to an empty slice; `None`
///   means every row is valid.
///
/// Errors if the concatenated byte length would exceed `i32::MAX` (Arrow
/// `Utf8`, not `LargeUtf8`).
pub fn build_row_aligned_input(
    dict: &[String],
    keys: &[i32],
    layout: KeyLayout,
    validity: Option<&[bool]>,
) -> BoltResult<(Vec<i32>, Vec<u8>)> {
    let mut offsets: Vec<i32> = Vec::with_capacity(keys.len() + 1);
    let mut bytes: Vec<u8> = Vec::new();
    offsets.push(0);
    for (row, &key) in keys.iter().enumerate() {
        let is_valid = validity.map(|v| v.get(row).copied().unwrap_or(false)).unwrap_or(true);
        let s = decode_row(dict, key, layout, is_valid)?;
        bytes.extend_from_slice(s.as_bytes());
        if bytes.len() > i32::MAX as usize {
            return Err(BoltError::Other(format!(
                "StringProject: total output bytes {} exceeds i32::MAX (Utf8)",
                bytes.len()
            )));
        }
        offsets.push(bytes.len() as i32);
    }
    Ok((offsets, bytes))
}

/// Exclusive-scan the per-row `u32` output lengths produced by the length pass
/// into the output `offsets` array (`i32`, `n_rows + 1` entries) plus the grand
/// total (the output `bytes` buffer size).
///
/// This is the host stand-in for the device prefix scan; the contract matches
/// [`crate::jit::prefix_scan`]'s exclusive scan (`offsets[0] = 0`,
/// `offsets[r+1] = offsets[r] + row_lens[r]`).
///
/// Errors if the running total would exceed `i32::MAX`.
pub fn exclusive_scan_lens(row_lens: &[u32]) -> BoltResult<(Vec<i32>, usize)> {
    let mut offsets: Vec<i32> = Vec::with_capacity(row_lens.len() + 1);
    let mut acc: usize = 0;
    offsets.push(0);
    for &len in row_lens {
        acc = acc.checked_add(len as usize).ok_or_else(|| {
            BoltError::Other("StringProject: output offset overflow".into())
        })?;
        if acc > i32::MAX as usize {
            return Err(BoltError::Other(format!(
                "StringProject: total output bytes {acc} exceeds i32::MAX (Utf8)"
            )));
        }
        offsets.push(acc as i32);
    }
    Ok((offsets, acc))
}

/// Reconstruct an Arrow [`StringArray`] from a row-aligned (`offsets`, `bytes`)
/// pair downloaded from the GPU write pass.
///
/// `validity[r]` (when `Some`) marks NULL rows; a NULL row's slice is empty on
/// the device but surfaces as a true Arrow NULL here.
///
/// Errors if a slice is not valid UTF-8 (a kernel wrote bytes the dictionary
/// could not have produced) — surfaced rather than masked.
pub fn string_array_from_offsets(
    offsets: &[i32],
    bytes: &[u8],
    validity: Option<&[bool]>,
) -> BoltResult<StringArray> {
    if offsets.is_empty() {
        return Ok(StringArray::from(Vec::<Option<&str>>::new()));
    }
    let n_rows = offsets.len() - 1;
    let mut out: Vec<Option<String>> = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        let is_valid = validity.map(|v| v.get(row).copied().unwrap_or(false)).unwrap_or(true);
        if !is_valid {
            out.push(None);
            continue;
        }
        let begin = offsets[row] as usize;
        let end = offsets[row + 1] as usize;
        let slice = bytes.get(begin..end).ok_or_else(|| {
            BoltError::Other(format!(
                "StringProject: row {row} slice [{begin}..{end}] out of bytes range {}",
                bytes.len()
            ))
        })?;
        let s = std::str::from_utf8(slice).map_err(|e| {
            BoltError::Other(format!("StringProject: row {row} is not valid UTF-8: {e}"))
        })?;
        out.push(Some(s.to_string()));
    }
    Ok(StringArray::from(out))
}

/// Host implementation of the GPU two-pass path, byte-for-byte: decode the
/// column row-by-row, apply the **ASCII byte-wise** transform, and rebuild a
/// `StringArray`. The GPU launch must produce an identical result for ASCII
/// data; this is the reference used by unit tests (and is NOT the Unicode
/// fallback — see [`host_transform_strings`] for that).
pub fn gpu_path_transform_pure(
    dict: &[String],
    keys: &[i32],
    layout: KeyLayout,
    validity: Option<&[bool]>,
    transform: StringTransform,
) -> BoltResult<StringArray> {
    let mut out: Vec<Option<String>> = Vec::with_capacity(keys.len());
    for (row, &key) in keys.iter().enumerate() {
        let is_valid = validity.map(|v| v.get(row).copied().unwrap_or(false)).unwrap_or(true);
        if !is_valid {
            out.push(None);
            continue;
        }
        let s = decode_row(dict, key, layout, true)?;
        let folded: Vec<u8> = s.bytes().map(|b| transform.apply_ascii_byte(b)).collect();
        // ASCII byte folding keeps the string valid UTF-8 (folds only touch
        // a-z / A-Z, which are single-byte).
        let folded = String::from_utf8(folded).map_err(|e| {
            BoltError::Other(format!("StringProject: ASCII fold produced invalid UTF-8: {e}"))
        })?;
        out.push(Some(folded));
    }
    Ok(StringArray::from(out))
}

/// Full-Unicode host fallback: decode the column and apply
/// `to_uppercase` / `to_lowercase`, preserving NULLs as Arrow NULLs.
///
/// Used when the GPU ASCII path would be incorrect (any dictionary entry has a
/// non-ASCII byte) or when the column is not GPU-resident in a supported
/// layout. NULL rows surface as Arrow NULL (not empty string).
pub fn host_transform_strings(
    dict: &[String],
    keys: &[i32],
    layout: KeyLayout,
    validity: Option<&[bool]>,
    transform: StringTransform,
) -> BoltResult<StringArray> {
    let mut out: Vec<Option<String>> = Vec::with_capacity(keys.len());
    for (row, &key) in keys.iter().enumerate() {
        let is_valid = validity.map(|v| v.get(row).copied().unwrap_or(false)).unwrap_or(true);
        if !is_valid {
            out.push(None);
            continue;
        }
        let s = decode_row(dict, key, layout, true)?;
        out.push(Some(transform.apply_host(s)));
    }
    Ok(StringArray::from(out))
}

/// Whether the GPU ASCII case-fold path is correct for `dict`.
///
/// The GPU kernels fold byte-wise and assume the output byte length equals the
/// input byte length. That holds for ASCII; for non-ASCII it can be wrong both
/// in bytes-out (`'ß'` upper → `"SS"`) and in semantics. Route any dictionary
/// containing a non-ASCII byte to the host fallback.
pub fn dict_is_ascii(dict: &[String]) -> bool {
    dict.iter().all(|s| s.is_ascii())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;

    fn owned(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| s.to_string()).collect()
    }

    fn collect(arr: &StringArray) -> Vec<Option<String>> {
        (0..arr.len())
            .map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i).to_string())
                }
            })
            .collect()
    }

    #[test]
    fn ascii_byte_fold_matches_kernel_rule() {
        // UPPER: a-z subtract 32; everything else unchanged.
        assert_eq!(StringTransform::Upper.apply_ascii_byte(b'a'), b'A');
        assert_eq!(StringTransform::Upper.apply_ascii_byte(b'z'), b'Z');
        assert_eq!(StringTransform::Upper.apply_ascii_byte(b'A'), b'A');
        assert_eq!(StringTransform::Upper.apply_ascii_byte(b'5'), b'5');
        // LOWER: A-Z add 32.
        assert_eq!(StringTransform::Lower.apply_ascii_byte(b'A'), b'a');
        assert_eq!(StringTransform::Lower.apply_ascii_byte(b'Z'), b'z');
        assert_eq!(StringTransform::Lower.apply_ascii_byte(b'a'), b'a');
    }

    #[test]
    fn build_row_aligned_input_one_based() {
        // dict ["us","eu"]; 1-based keys, 0 = NULL.
        let dict = owned(&["us", "eu"]);
        let keys = vec![1, 2, 0, 1]; // us, eu, NULL, us
        let (offsets, bytes) =
            build_row_aligned_input(&dict, &keys, KeyLayout::OneBasedNullSlot0, None).unwrap();
        // slices: "us"(2), "eu"(2), ""(0), "us"(2)
        assert_eq!(offsets, vec![0, 2, 4, 4, 6]);
        assert_eq!(&bytes, b"useuus");
    }

    #[test]
    fn build_row_aligned_input_zero_based_with_validity() {
        let dict = owned(&["alpha", "bb"]);
        let keys = vec![0, 1, 0];
        let validity = vec![true, false, true];
        let (offsets, bytes) =
            build_row_aligned_input(&dict, &keys, KeyLayout::ZeroBased, Some(&validity)).unwrap();
        // "alpha"(5), NULL→""(0), "alpha"(5)
        assert_eq!(offsets, vec![0, 5, 5, 10]);
        assert_eq!(&bytes, b"alphaalpha");
    }

    #[test]
    fn exclusive_scan_matches_offsets() {
        let lens = vec![2u32, 0, 3, 1];
        let (offsets, total) = exclusive_scan_lens(&lens).unwrap();
        assert_eq!(offsets, vec![0, 2, 2, 5, 6]);
        assert_eq!(total, 6);
    }

    #[test]
    fn exclusive_scan_empty() {
        let (offsets, total) = exclusive_scan_lens(&[]).unwrap();
        assert_eq!(offsets, vec![0]);
        assert_eq!(total, 0);
    }

    #[test]
    fn string_array_roundtrip_with_nulls() {
        // Reconstruct from offsets+bytes the GPU write pass would produce for
        // UPPER over ["us","eu",NULL,"us"].
        let offsets = vec![0i32, 2, 4, 4, 6];
        let bytes = b"USEUUS".to_vec();
        let validity = vec![true, true, false, true];
        let arr = string_array_from_offsets(&offsets, &bytes, Some(&validity)).unwrap();
        assert_eq!(
            collect(&arr),
            vec![
                Some("US".to_string()),
                Some("EU".to_string()),
                None,
                Some("US".to_string()),
            ]
        );
    }

    #[test]
    fn string_array_rejects_invalid_utf8() {
        let offsets = vec![0i32, 1];
        let bytes = vec![0xFFu8];
        let err = string_array_from_offsets(&offsets, &bytes, None).unwrap_err();
        assert!(format!("{err}").contains("not valid UTF-8"), "{err}");
    }

    #[test]
    fn gpu_pure_equals_host_for_ascii() {
        // For ASCII data the GPU ASCII path and the Unicode host fallback must
        // agree.
        let dict = owned(&["Hello", "WORLD", "MixedCase"]);
        let keys = vec![1i32, 2, 3, 0]; // last row NULL
        for t in [StringTransform::Upper, StringTransform::Lower] {
            let gpu =
                gpu_path_transform_pure(&dict, &keys, KeyLayout::OneBasedNullSlot0, None, t)
                    .unwrap();
            let host =
                host_transform_strings(&dict, &keys, KeyLayout::OneBasedNullSlot0, None, t)
                    .unwrap();
            assert_eq!(collect(&gpu), collect(&host), "transform {t:?}");
        }
    }

    #[test]
    fn gpu_pure_upper_values() {
        // ZeroBased: key k → dict[k]. keys [0,1] → "abc","Z9z".
        let dict = owned(&["abc", "Z9z"]);
        let keys = vec![0i32, 1];
        let arr =
            gpu_path_transform_pure(&dict, &keys, KeyLayout::ZeroBased, None, StringTransform::Upper)
                .unwrap();
        // "abc"→"ABC" (folds a-z), "Z9z"→"Z9Z" (only the trailing 'z' folds).
        assert_eq!(
            collect(&arr),
            vec![Some("ABC".to_string()), Some("Z9Z".to_string())]
        );
    }

    #[test]
    fn non_ascii_dict_routes_to_host() {
        // 'ß' makes the dict non-ASCII; dict_is_ascii must be false so the
        // executor avoids the byte-wise GPU fold.
        assert!(dict_is_ascii(&owned(&["abc", "DEF"])));
        assert!(!dict_is_ascii(&owned(&["abc", "straße"])));
    }

    #[test]
    fn host_fallback_full_unicode() {
        // 'ß'.to_uppercase() == "SS" — a length change the GPU ASCII path can't
        // do; the host fallback must.
        let dict = owned(&["straße"]);
        let keys = vec![1i32];
        let arr = host_transform_strings(
            &dict,
            &keys,
            KeyLayout::OneBasedNullSlot0,
            None,
            StringTransform::Upper,
        )
        .unwrap();
        assert_eq!(collect(&arr), vec![Some("STRASSE".to_string())]);
    }
}
