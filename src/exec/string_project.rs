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
//! ## Fallback (no panic) and the v0.7.0 device-path gate
//!
//! **The GPU two-pass kernels have never been executed on GPU hardware as of
//! v0.7.0** (CI has no CUDA device). The host transform is therefore the
//! production correctness path: the executor must consult
//! [`gpu_string_enabled`] (`BOLT_GPU_STRING=1`, default OFF) before selecting
//! the device path, so the validated host helpers run by default. The device
//! launch is opt-in for hardware bring-up only.
//!
//! Even when the gate is ON, the GPU two-pass path is taken only when the
//! transform is ASCII-safe for the column's dictionary (the kernels case-fold
//! byte-wise, which is correct for ASCII but NOT for arbitrary Unicode — e.g.
//! `'ß'.to_uppercase()` is `"SS"`, changing the byte length). When any
//! dictionary entry contains a non-ASCII byte the executor falls back to the
//! host transform ([`host_transform_strings`]), which uses full Unicode
//! `to_uppercase` / `to_lowercase` via [`crate::exec::string_ops`]. Both paths
//! produce the same `StringArray` for ASCII data.

use arrow_array::StringArray;

use crate::error::{BoltError, BoltResult};
use crate::exec::expr_agg::{eval_expr, ColumnEnv, HostColumn};
use crate::plan::logical_plan::{DataType, Expr, ScalarFnKind};

// Re-export the shared GPU-string env gate so the executor and callers of this
// module can consult a single source of truth. See
// [`crate::exec::string_like::BOLT_GPU_STRING_ENV`].
pub use crate::exec::string_like::{gpu_string_enabled, BOLT_GPU_STRING_ENV};

/// Which end(s) a [`StringTransform::Trim`] strips ASCII/Unicode whitespace
/// from. Mirrors [`crate::exec::string_ops_extended::TrimSide`] but is declared
/// here so the [`StringTransform`] enum stays `Copy` and self-contained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimMode {
    /// Strip from BOTH ends (`TRIM`).
    Both,
    /// Strip from the START only (`LTRIM` / `TRIM(LEADING ...)`).
    Leading,
    /// Strip from the END only (`RTRIM` / `TRIM(TRAILING ...)`).
    Trailing,
}

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
    /// `SUBSTRING(s FROM start [FOR length])` — character-indexed (1-based)
    /// window. `length` is `None` when the SQL has no `FOR` clause (take to the
    /// end of the string). Realised via the byte-identical host mirror
    /// ([`crate::exec::string_ops_extended::substring_str`]); the GPU two-pass
    /// SUBSTRING producer exists in [`crate::jit::string_kernel`] but is
    /// unvalidated on hardware (like CONCAT), so this executor uses the host
    /// path for correctness.
    Substring {
        /// 1-based character start position (as written in the SQL).
        start: i32,
        /// Character count to take, or `None` for "to the end of the string".
        length: Option<i32>,
    },
    /// `TRIM`/`LTRIM`/`RTRIM(s)` — strip leading/trailing whitespace. Realised
    /// via the host mirror ([`crate::exec::string_ops_extended::trim_str`],
    /// full-Unicode whitespace). The GPU kernels trim only ASCII whitespace;
    /// the host realisation is the supported path.
    Trim {
        /// Which end(s) to strip.
        mode: TrimMode,
    },
}

impl StringTransform {
    /// Map to the JIT [`ScalarFnKind`] whose two-pass kernels implement this
    /// transform.
    pub fn scalar_fn_kind(self) -> ScalarFnKind {
        match self {
            StringTransform::Upper => ScalarFnKind::Upper,
            StringTransform::Lower => ScalarFnKind::Lower,
            StringTransform::Substring { .. } => ScalarFnKind::Substring,
            StringTransform::Trim { mode } => match mode {
                TrimMode::Both => ScalarFnKind::TrimBoth,
                TrimMode::Leading => ScalarFnKind::TrimLeading,
                TrimMode::Trailing => ScalarFnKind::TrimTrailing,
            },
        }
    }

    /// `true` when this transform is realised purely host-side (no GPU two-pass
    /// launch wired into the executor): `SUBSTRING` / `TRIM`. The case-fold
    /// transforms (`UPPER`/`LOWER`) drive the device kernels for ASCII data.
    pub fn is_host_realized(self) -> bool {
        matches!(
            self,
            StringTransform::Substring { .. } | StringTransform::Trim { .. }
        )
    }

    /// Apply the **full-Unicode** host transform to a single string (the host
    /// fallback / realisation). For `UPPER`/`LOWER` this matches
    /// `str::to_uppercase` / `str::to_lowercase`; for `SUBSTRING`/`TRIM` it
    /// matches the host helpers in [`crate::exec::string_ops_extended`].
    pub fn apply_host(self, s: &str) -> String {
        use crate::exec::string_ops_extended::{substring_str, trim_str, TrimSide};
        match self {
            StringTransform::Upper => s.to_uppercase(),
            StringTransform::Lower => s.to_lowercase(),
            StringTransform::Substring { start, length } => {
                // No `FOR` clause → take to the end. `substring_str` clamps a
                // huge length to the string's character count, so i32::MAX is a
                // safe "rest of string" sentinel matching the 2-arg SQL form.
                let len = length.unwrap_or(i32::MAX);
                substring_str(s, start, len)
            }
            StringTransform::Trim { mode } => {
                let side = match mode {
                    TrimMode::Both => TrimSide::Both,
                    TrimMode::Leading => TrimSide::Leading,
                    TrimMode::Trailing => TrimSide::Trailing,
                };
                trim_str(s, side, None)
            }
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
    ///
    /// Only defined for the byte-wise case folds (`UPPER`/`LOWER`); the
    /// variable-window transforms (`SUBSTRING`/`TRIM`) are realised whole-string
    /// host-side and never reach the per-byte fold, so they panic here (a
    /// programming error if ever called).
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
            StringTransform::Substring { .. } | StringTransform::Trim { .. } => {
                unreachable!("apply_ascii_byte is only valid for UPPER/LOWER")
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
                dict.get((key - 1) as usize)
                    .map(String::as_str)
                    .ok_or_else(|| {
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
//
// UNVALIDATED ON GPU HARDWARE as of v0.7.0 — host fallback is the correctness
// path; opt-in via BOLT_GPU_STRING for testing. This materialises the input the
// UPPER/LOWER device length pass consumes; the executor must only build/launch
// that device path when `gpu_string_enabled` is true (default OFF →
// host_transform_strings / gpu_path_transform_pure host mirror).
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
        let is_valid = validity
            .map(|v| v.get(row).copied().unwrap_or(false))
            .unwrap_or(true);
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
//
// UNVALIDATED ON GPU HARDWARE as of v0.7.0 — host fallback is the correctness
// path; opt-in via BOLT_GPU_STRING for testing. This is the host scan step
// wedged between the device length and write passes; it is only reached on the
// gated UPPER/LOWER device path.
pub fn exclusive_scan_lens(row_lens: &[u32]) -> BoltResult<(Vec<i32>, usize)> {
    let mut offsets: Vec<i32> = Vec::with_capacity(row_lens.len() + 1);
    let mut acc: usize = 0;
    offsets.push(0);
    for &len in row_lens {
        acc = acc
            .checked_add(len as usize)
            .ok_or_else(|| BoltError::Other("StringProject: output offset overflow".into()))?;
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
//
// UNVALIDATED ON GPU HARDWARE as of v0.7.0 — host fallback is the correctness
// path; opt-in via BOLT_GPU_STRING for testing. On the gated UPPER/LOWER device
// path this reconstructs the array from the device write-pass download; it is
// also reused by the pure-host gpu_path_concat_pure mirror.
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
        let is_valid = validity
            .map(|v| v.get(row).copied().unwrap_or(false))
            .unwrap_or(true);
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
//
// UNVALIDATED ON GPU HARDWARE as of v0.7.0 — host fallback is the correctness
// path; opt-in via BOLT_GPU_STRING for testing. NOTE: this function is itself a
// pure-HOST mirror (it never launches a kernel) and is therefore always safe to
// run; the comment flags that the *device* kernel it mirrors (jit::string_kernel
// UPPER/LOWER write pass, launched by Engine::string_transform_column) is the
// unvalidated path the executor must gate behind gpu_string_enabled.
pub fn gpu_path_transform_pure(
    dict: &[String],
    keys: &[i32],
    layout: KeyLayout,
    validity: Option<&[bool]>,
    transform: StringTransform,
) -> BoltResult<StringArray> {
    let mut out: Vec<Option<String>> = Vec::with_capacity(keys.len());
    for (row, &key) in keys.iter().enumerate() {
        let is_valid = validity
            .map(|v| v.get(row).copied().unwrap_or(false))
            .unwrap_or(true);
        // Honor the slot-0 NULL sentinel even without an explicit validity slice,
        // mirroring `host_transform_strings` so the two stay byte-for-byte equal.
        let is_null_sentinel = matches!(layout, KeyLayout::OneBasedNullSlot0) && key == 0;
        if !is_valid || is_null_sentinel {
            out.push(None);
            continue;
        }
        let s = decode_row(dict, key, layout, true)?;
        let folded: Vec<u8> = s.bytes().map(|b| transform.apply_ascii_byte(b)).collect();
        // ASCII byte folding keeps the string valid UTF-8 (folds only touch
        // a-z / A-Z, which are single-byte).
        let folded = String::from_utf8(folded).map_err(|e| {
            BoltError::Other(format!(
                "StringProject: ASCII fold produced invalid UTF-8: {e}"
            ))
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
        let is_valid = validity
            .map(|v| v.get(row).copied().unwrap_or(false))
            .unwrap_or(true);
        // A row is SQL NULL if the explicit validity bit says so, OR (for the
        // engine-managed Utf8 layout) the key is the slot-0 NULL sentinel. The
        // latter makes NULL surface as Arrow NULL even when no separate validity
        // slice is supplied (production always passes validity = key != 0, so
        // this only matters for callers that rely on the sentinel alone).
        let is_null_sentinel = matches!(layout, KeyLayout::OneBasedNullSlot0) && key == 0;
        if !is_valid || is_null_sentinel {
            out.push(None);
            continue;
        }
        let s = decode_row(dict, key, layout, true)?;
        out.push(Some(transform.apply_host(s)));
    }
    Ok(StringArray::from(out))
}

// ---------------------------------------------------------------------------
// N-input CONCAT two-pass executor helpers.
//
// `CONCAT(a_0, ..., a_{n-1})` over n dictionary-encoded Utf8 columns. The GPU
// producer (jit::string_kernel::compile_concat_{len,write}_pass) consumes n
// row-aligned (offsets, bytes) pairs and writes one new Utf8 array. The shape
// mirrors the UPPER/LOWER two-pass exactly: build n row-aligned inputs, sum the
// per-row byte lengths (length pass), exclusive-scan to out_offsets + total
// (reuse `exclusive_scan_lens`), copy each input slice in order (write pass),
// then reconstruct the StringArray. NULL is handled HOST-SIDE: per standard SQL
// CONCAT semantics the output row is NULL if ANY input row is NULL (matching
// `exec::string_ops_extended::concat`), re-applied below regardless of the empty
// slices NULL rows decode to on the device.
// ---------------------------------------------------------------------------

/// One CONCAT input column, already decoded into the row-aligned Arrow-`Utf8`
/// shape the GPU passes consume, plus its per-row validity (NULL channel).
///
/// `offsets` has `n_rows + 1` entries; `offsets[r]..offsets[r+1]` is row `r`'s
/// slice in `bytes`. `validity[r]` is `false` for a SQL NULL row (whose slice is
/// empty). `validity` always has `n_rows` entries (the row count is
/// `offsets.len() - 1`).
#[derive(Debug, Clone)]
pub struct ConcatInput {
    /// Row-aligned `i32` offsets (`n_rows + 1` entries).
    pub offsets: Vec<i32>,
    /// Row-aligned UTF-8 bytes.
    pub bytes: Vec<u8>,
    /// Per-row validity (`false` = SQL NULL).
    pub validity: Vec<bool>,
}

/// Sum the per-row output byte lengths across all `n` CONCAT inputs — the host
/// stand-in for the GPU length pass
/// ([`crate::jit::string_kernel::compile_concat_len_pass`]).
///
/// `row_lens[r] = sum_k (inputs[k].offsets[r+1] - inputs[k].offsets[r])`. NULL
/// rows contribute 0 (their device slice is empty); the SQL NULL-if-any-arg-NULL
/// rule is applied separately in [`concat_output_validity`] / the array rebuild.
///
/// Errors if `inputs` is empty or the inputs disagree on row count — the planner
/// should never emit those, but we surface rather than panic.
pub fn concat_row_lens(inputs: &[ConcatInput]) -> BoltResult<Vec<u32>> {
    if inputs.is_empty() {
        return Err(BoltError::Other(
            "CONCAT: at least one input is required".into(),
        ));
    }
    let n_rows = inputs[0].offsets.len().saturating_sub(1);
    for (k, inp) in inputs.iter().enumerate() {
        if inp.offsets.len().saturating_sub(1) != n_rows {
            return Err(BoltError::Other(format!(
                "CONCAT: n_rows mismatch (input 0 = {}, input {} = {})",
                n_rows,
                k,
                inp.offsets.len().saturating_sub(1)
            )));
        }
    }
    let mut row_lens = vec![0u32; n_rows];
    for inp in inputs {
        for (r, slot) in row_lens.iter_mut().enumerate() {
            let len = (inp.offsets[r + 1] - inp.offsets[r]) as u32;
            *slot = slot
                .checked_add(len)
                .ok_or_else(|| BoltError::Other("CONCAT: per-row length overflow".into()))?;
        }
    }
    Ok(row_lens)
}

/// Compute the output validity for CONCAT: row `r` is valid iff EVERY input row
/// `r` is valid (standard SQL — NULL if ANY arg is NULL). Matches
/// [`crate::exec::string_ops_extended::concat`].
pub fn concat_output_validity(inputs: &[ConcatInput]) -> Vec<bool> {
    if inputs.is_empty() {
        return Vec::new();
    }
    let n_rows = inputs[0].offsets.len().saturating_sub(1);
    (0..n_rows)
        .map(|r| {
            inputs
                .iter()
                .all(|inp| inp.validity.get(r).copied().unwrap_or(false))
        })
        .collect()
}

/// Host implementation of the GPU CONCAT two-pass path, byte-for-byte: sum the
/// per-row lengths, exclusive-scan to offsets, copy each input slice in order,
/// then reconstruct the `StringArray` re-applying NULL-if-any-arg-NULL. The GPU
/// launch must produce an identical `out_bytes`/`out_offsets`; this is the
/// reference used by unit tests (no CUDA runtime needed).
//
// UNVALIDATED ON GPU HARDWARE as of v0.7.0 — host fallback is the correctness
// path; opt-in via BOLT_GPU_STRING for testing. The device CONCAT two-pass
// kernels are NEVER selected by the executor today: Engine::execute_string_project
// always calls host_concat_strings. This pure-host mirror is retained only as
// the unit-test reference for the (currently unreachable) device producer.
pub fn gpu_path_concat_pure(inputs: &[ConcatInput]) -> BoltResult<StringArray> {
    let row_lens = concat_row_lens(inputs)?;
    let (out_offsets, total) = exclusive_scan_lens(&row_lens)?;
    let n_rows = row_lens.len();

    // Write pass: copy each input slice, in input order, into the row's region.
    let mut out_bytes = vec![0u8; total];
    for r in 0..n_rows {
        let mut cursor = out_offsets[r] as usize;
        for inp in inputs {
            let begin = inp.offsets[r] as usize;
            let end = inp.offsets[r + 1] as usize;
            let slice = inp.bytes.get(begin..end).ok_or_else(|| {
                BoltError::Other(format!(
                    "CONCAT: input slice [{begin}..{end}] out of bytes range {}",
                    inp.bytes.len()
                ))
            })?;
            out_bytes[cursor..cursor + slice.len()].copy_from_slice(slice);
            cursor += slice.len();
        }
    }

    let validity = concat_output_validity(inputs);
    string_array_from_offsets(&out_offsets, &out_bytes, Some(&validity))
}

/// Full host fallback for CONCAT when the GPU path is unavailable (column not
/// resident in a supported layout, arity beyond
/// [`crate::jit::string_kernel::CONCAT_MAX_INPUTS`], or any non-Utf8 arg).
///
/// Concatenates the decoded UTF-8 slices per row and re-applies
/// NULL-if-any-arg-NULL. Functionally identical to [`gpu_path_concat_pure`] for
/// any input (both honour the same NULL rule); kept as a distinct entry point so
/// the executor can call it without first materialising scan/offset buffers.
pub fn host_concat_strings(inputs: &[ConcatInput]) -> BoltResult<StringArray> {
    if inputs.is_empty() {
        return Err(BoltError::Other(
            "CONCAT: at least one input is required".into(),
        ));
    }
    let n_rows = inputs[0].offsets.len().saturating_sub(1);
    let validity = concat_output_validity(inputs);
    let mut out: Vec<Option<String>> = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        if !validity[r] {
            out.push(None);
            continue;
        }
        let mut s = String::new();
        for inp in inputs {
            let begin = inp.offsets[r] as usize;
            let end = inp.offsets[r + 1] as usize;
            let slice = inp.bytes.get(begin..end).ok_or_else(|| {
                BoltError::Other(format!(
                    "CONCAT: input slice [{begin}..{end}] out of bytes range {}",
                    inp.bytes.len()
                ))
            })?;
            let piece = std::str::from_utf8(slice).map_err(|e| {
                BoltError::Other(format!("CONCAT: input row {r} is not valid UTF-8: {e}"))
            })?;
            s.push_str(piece);
        }
        out.push(Some(s));
    }
    Ok(StringArray::from(out))
}

/// Build a [`ConcatInput`] for one CONCAT source column from its host-side
/// dictionary + keys + layout (+ optional validity), reusing the same row-
/// alignment decode the UPPER/LOWER path uses ([`build_row_aligned_input`]).
///
/// This is the per-column adapter the executor calls once per CONCAT argument
/// before handing the `Vec<ConcatInput>` to [`gpu_path_concat_pure`] (GPU) or
/// [`host_concat_strings`] (fallback). The validity it records is the row's SQL
/// validity (NULL channel), independent of the empty slice a NULL row decodes
/// to — so the NULL-if-any-arg-NULL rule can be applied downstream.
pub fn build_concat_input(
    dict: &[String],
    keys: &[i32],
    layout: KeyLayout,
    validity: Option<&[bool]>,
) -> BoltResult<ConcatInput> {
    let (offsets, bytes) = build_row_aligned_input(dict, keys, layout, validity)?;
    let row_validity: Vec<bool> = (0..keys.len())
        .map(|row| {
            // Honour an explicit validity slice; otherwise derive NULL from the
            // 1-based layout's slot-0 sentinel (matching `decode_row`).
            if let Some(v) = validity {
                v.get(row).copied().unwrap_or(false)
            } else {
                match layout {
                    KeyLayout::OneBasedNullSlot0 => keys[row] != 0,
                    KeyLayout::ZeroBased => true,
                }
            }
        })
        .collect();
    Ok(ConcatInput {
        offsets,
        bytes,
        validity: row_validity,
    })
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

// ---------------------------------------------------------------------------
// F11: host-evaluated CASE with a Utf8 result.
//
// A `CASE WHEN c1 THEN v1 [WHEN c2 THEN v2 ...] [ELSE ve] END` whose result
// type is `Utf8` selects a per-row string. Full on-device variable-width CASE
// is hard (no Utf8 register class / heap ABI), so we evaluate it host-side over
// the decoded source columns, honouring SQL CASE 3VL:
//
//   * Branches are tested in order; the FIRST branch whose condition is TRUE
//     wins. A condition that is FALSE *or* NULL (SQL UNKNOWN) does NOT fire —
//     the row falls through to the next branch (and ultimately to ELSE / NULL).
//   * With no ELSE, an unmatched row is SQL NULL.
//   * The selected THEN/ELSE value is itself an expression; its own value
//     (including NULL) is what the row takes once its branch is chosen.
//
// Each WHEN condition and each THEN/ELSE value is evaluated with the existing
// host evaluator (`expr_agg::eval_expr`), so any non-CASE sub-expression it
// supports (column refs, string/numeric comparisons, LIKE, literals, CONCAT/
// SUBSTRING/TRIM-free arithmetic, ...) works. A branch value that does not
// evaluate to Utf8, or a nested CASE the evaluator rejects, surfaces as an
// error (the lowering keeps such shapes off this path where it can).
// ---------------------------------------------------------------------------

/// Evaluate a `Utf8`-result `CASE` expression host-side into a [`StringArray`]
/// of `n_rows` rows, honouring SQL CASE 3VL (a NULL condition behaves like
/// FALSE — the row falls through to the next branch / ELSE / NULL).
///
/// `branches` are the `(WHEN condition, THEN value)` pairs in source order;
/// `else_branch` is the optional `ELSE value` (SQL NULL when `None`). `env` is a
/// [`ColumnEnv`] over the decoded source columns (built by the caller from the
/// host source batch). Each condition must evaluate to `Bool`; each THEN/ELSE
/// value must evaluate to `Utf8`.
pub fn eval_case_utf8(
    branches: &[(Expr, Expr)],
    else_branch: Option<&Expr>,
    env: &ColumnEnv<'_>,
    n_rows: usize,
) -> BoltResult<StringArray> {
    // Evaluate every WHEN condition (Bool) and THEN value (Utf8) up front, plus
    // the ELSE value if present. Column-major evaluation reuses the existing
    // vectorised host evaluator; we then combine row-by-row.
    let mut conds: Vec<Vec<Option<bool>>> = Vec::with_capacity(branches.len());
    let mut thens: Vec<Vec<Option<String>>> = Vec::with_capacity(branches.len());
    for (when, then) in branches {
        let cond = eval_expr(when, env, DataType::Bool, n_rows)?;
        let cond_vec = match cond {
            HostColumn::Bool(v) => v,
            other => {
                return Err(BoltError::Plan(format!(
                    "CASE(Utf8): WHEN condition must be Bool, got {:?}",
                    other.dtype()
                )))
            }
        };
        let then_col = eval_expr(then, env, DataType::Utf8, n_rows)?;
        let then_vec = match then_col {
            HostColumn::Utf8(v) => v,
            other => {
                return Err(BoltError::Plan(format!(
                    "CASE(Utf8): THEN value must be Utf8, got {:?}",
                    other.dtype()
                )))
            }
        };
        conds.push(cond_vec);
        thens.push(then_vec);
    }
    let else_vec: Option<Vec<Option<String>>> = match else_branch {
        None => None,
        Some(e) => {
            let col = eval_expr(e, env, DataType::Utf8, n_rows)?;
            match col {
                HostColumn::Utf8(v) => Some(v),
                other => {
                    return Err(BoltError::Plan(format!(
                        "CASE(Utf8): ELSE value must be Utf8, got {:?}",
                        other.dtype()
                    )))
                }
            }
        }
    };

    let mut out: Vec<Option<String>> = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        let mut chosen: Option<String> = None;
        let mut matched = false;
        for (b, cond) in conds.iter().enumerate() {
            // SQL 3VL: only a TRUE condition fires; FALSE and NULL fall through.
            if cond.get(row).copied().flatten() == Some(true) {
                chosen = thens[b].get(row).cloned().flatten();
                matched = true;
                break;
            }
        }
        if !matched {
            chosen = else_vec
                .as_ref()
                .and_then(|v| v.get(row).cloned().flatten());
        }
        out.push(chosen);
    }
    Ok(StringArray::from(out))
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
            let gpu = gpu_path_transform_pure(&dict, &keys, KeyLayout::OneBasedNullSlot0, None, t)
                .unwrap();
            let host = host_transform_strings(&dict, &keys, KeyLayout::OneBasedNullSlot0, None, t)
                .unwrap();
            assert_eq!(collect(&gpu), collect(&host), "transform {t:?}");
        }
    }

    #[test]
    fn gpu_pure_upper_values() {
        // ZeroBased: key k → dict[k]. keys [0,1] → "abc","Z9z".
        let dict = owned(&["abc", "Z9z"]);
        let keys = vec![0i32, 1];
        let arr = gpu_path_transform_pure(
            &dict,
            &keys,
            KeyLayout::ZeroBased,
            None,
            StringTransform::Upper,
        )
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

    // ---- CONCAT two-pass executor helpers --------------------------------

    fn concat_input(strs: &[Option<&str>]) -> ConcatInput {
        // Build a row-aligned input from a column of Option<&str>; None = NULL
        // row (empty slice, validity=false), matching the device decode.
        let mut offsets = vec![0i32];
        let mut bytes = Vec::new();
        let mut validity = Vec::new();
        for s in strs {
            match s {
                Some(v) => {
                    bytes.extend_from_slice(v.as_bytes());
                    validity.push(true);
                }
                None => validity.push(false),
            }
            offsets.push(bytes.len() as i32);
        }
        ConcatInput {
            offsets,
            bytes,
            validity,
        }
    }

    #[test]
    fn concat_row_lens_sums_inputs() {
        // a = ["us","eu","x"], b = ["NY","BERLIN","y"]
        let a = concat_input(&[Some("us"), Some("eu"), Some("x")]);
        let b = concat_input(&[Some("NY"), Some("BERLIN"), Some("y")]);
        let lens = concat_row_lens(&[a, b]).unwrap();
        // 2+2, 2+6, 1+1
        assert_eq!(lens, vec![4, 8, 2]);
    }

    #[test]
    fn concat_row_lens_null_contributes_zero() {
        // NULL rows decode to empty slices -> contribute 0 to the sum (the SQL
        // NULL propagation is separate, applied in concat_output_validity).
        let a = concat_input(&[Some("ab"), None]);
        let b = concat_input(&[None, Some("cd")]);
        let lens = concat_row_lens(&[a, b]).unwrap();
        assert_eq!(lens, vec![2, 2]);
    }

    #[test]
    fn concat_output_validity_is_and_of_inputs() {
        let a = concat_input(&[Some("a"), None, Some("c"), None]);
        let b = concat_input(&[Some("x"), Some("y"), None, None]);
        // Row valid iff BOTH inputs valid.
        assert_eq!(
            concat_output_validity(&[a, b]),
            vec![true, false, false, false]
        );
    }

    #[test]
    fn gpu_path_concat_basic_two_inputs() {
        // Mirrors the host concat_pure spec example.
        let a = concat_input(&[Some("us"), Some("eu"), Some("us")]);
        let b = concat_input(&[Some("NY"), Some("BERLIN"), Some("BERLIN")]);
        let arr = gpu_path_concat_pure(&[a, b]).unwrap();
        assert_eq!(
            collect(&arr),
            vec![
                Some("usNY".to_string()),
                Some("euBERLIN".to_string()),
                Some("usBERLIN".to_string()),
            ]
        );
    }

    #[test]
    fn gpu_path_concat_propagates_null_if_any_arg_null() {
        // Standard SQL: NULL on EITHER side -> NULL output. Must match
        // exec::string_ops_extended::concat_pure exactly.
        let a = concat_input(&[None, Some("a"), Some("a"), None]);
        let b = concat_input(&[Some("x"), None, Some("y"), None]);
        let arr = gpu_path_concat_pure(&[a, b]).unwrap();
        assert_eq!(
            collect(&arr),
            vec![None, None, Some("ay".to_string()), None]
        );
    }

    #[test]
    fn gpu_path_concat_three_inputs_in_order() {
        let a = concat_input(&[Some("a"), Some("1")]);
        let b = concat_input(&[Some("b"), Some("2")]);
        let c = concat_input(&[Some("c"), Some("3")]);
        let arr = gpu_path_concat_pure(&[a, b, c]).unwrap();
        assert_eq!(
            collect(&arr),
            vec![Some("abc".to_string()), Some("123".to_string())]
        );
    }

    #[test]
    fn gpu_path_concat_equals_host_concat() {
        // The GPU two-pass host mirror and the plain host fallback must agree on
        // every row, including NULL propagation.
        let a = concat_input(&[Some("foo"), None, Some(""), Some("z")]);
        let b = concat_input(&[Some("bar"), Some("q"), Some("w"), None]);
        let gpu = gpu_path_concat_pure(&[a.clone(), b.clone()]).unwrap();
        let host = host_concat_strings(&[a, b]).unwrap();
        assert_eq!(collect(&gpu), collect(&host));
    }

    #[test]
    fn gpu_path_concat_empty_strings_and_zero_rows() {
        // All-empty inputs: every row is "" (non-NULL).
        let a = concat_input(&[Some(""), Some("")]);
        let b = concat_input(&[Some(""), Some("")]);
        let arr = gpu_path_concat_pure(&[a, b]).unwrap();
        assert_eq!(
            collect(&arr),
            vec![Some("".to_string()), Some("".to_string())]
        );

        // Zero rows.
        let a0 = concat_input(&[]);
        let b0 = concat_input(&[]);
        let arr0 = gpu_path_concat_pure(&[a0, b0]).unwrap();
        assert_eq!(arr0.len(), 0);
    }

    #[test]
    fn build_concat_input_one_based_layout_decodes_nulls() {
        // dict ["us","eu"], 1-based keys, 0 = NULL.
        let dict = owned(&["us", "eu"]);
        let keys = vec![1i32, 0, 2]; // us, NULL, eu
        let inp = build_concat_input(&dict, &keys, KeyLayout::OneBasedNullSlot0, None).unwrap();
        assert_eq!(inp.offsets, vec![0, 2, 2, 4]); // "us","",  "eu"
        assert_eq!(&inp.bytes, b"useu");
        assert_eq!(inp.validity, vec![true, false, true]);
    }

    #[test]
    fn build_concat_input_round_trips_through_gpu_path() {
        // Two 1-based columns concatenate exactly like the host concat_pure spec.
        let a_dict = owned(&["us", "eu"]);
        let a_keys = vec![1i32, 2, 1]; // us, eu, us
        let b_dict = owned(&["NY", "BERLIN"]);
        let b_keys = vec![1i32, 2, 2]; // NY, BERLIN, BERLIN
        let a = build_concat_input(&a_dict, &a_keys, KeyLayout::OneBasedNullSlot0, None).unwrap();
        let b = build_concat_input(&b_dict, &b_keys, KeyLayout::OneBasedNullSlot0, None).unwrap();
        let arr = gpu_path_concat_pure(&[a, b]).unwrap();
        assert_eq!(
            collect(&arr),
            vec![
                Some("usNY".to_string()),
                Some("euBERLIN".to_string()),
                Some("usBERLIN".to_string()),
            ]
        );
    }

    #[test]
    fn concat_row_lens_rejects_row_count_mismatch() {
        let a = concat_input(&[Some("a"), Some("b")]);
        let b = concat_input(&[Some("x")]);
        let err = concat_row_lens(&[a, b]).unwrap_err();
        assert!(format!("{err}").contains("n_rows mismatch"), "{err}");
    }

    // ---- SUBSTRING / TRIM host realisation (F9) --------------------------

    #[test]
    fn substring_host_three_arg_over_dict() {
        // dict ["hello","world"]; 1-based keys, 0 = NULL.
        // SUBSTRING(s, 2, 3): "hello"->"ell", "world"->"orl", NULL->NULL.
        let dict = owned(&["hello", "world"]);
        let keys = vec![1i32, 2, 0];
        let t = StringTransform::Substring {
            start: 2,
            length: Some(3),
        };
        let arr =
            host_transform_strings(&dict, &keys, KeyLayout::OneBasedNullSlot0, None, t).unwrap();
        assert_eq!(
            collect(&arr),
            vec![Some("ell".to_string()), Some("orl".to_string()), None]
        );
    }

    #[test]
    fn substring_host_no_for_goes_to_end() {
        // SUBSTRING(s FROM 3) (length = None) takes to the end of the string.
        let dict = owned(&["hello"]);
        let keys = vec![1i32];
        let t = StringTransform::Substring {
            start: 3,
            length: None,
        };
        let arr =
            host_transform_strings(&dict, &keys, KeyLayout::OneBasedNullSlot0, None, t).unwrap();
        assert_eq!(collect(&arr), vec![Some("llo".to_string())]);
    }

    #[test]
    fn substring_host_unicode_char_indexed() {
        // "héllo" is 5 CHARACTERS; SUBSTRING(2,1) is the single char 'é'.
        let dict = owned(&["héllo"]);
        let keys = vec![1i32];
        let t = StringTransform::Substring {
            start: 2,
            length: Some(1),
        };
        let arr =
            host_transform_strings(&dict, &keys, KeyLayout::OneBasedNullSlot0, None, t).unwrap();
        assert_eq!(collect(&arr), vec![Some("é".to_string())]);
    }

    #[test]
    fn trim_host_modes_and_nulls() {
        // dict ["  hi  ", "x"]; 1-based keys, NULL row.
        let dict = owned(&["  hi  ", "x"]);
        let keys = vec![1i32, 2, 0];
        for (mode, expected_first) in [
            (TrimMode::Both, "hi"),
            (TrimMode::Leading, "hi  "),
            (TrimMode::Trailing, "  hi"),
        ] {
            let t = StringTransform::Trim { mode };
            let arr = host_transform_strings(&dict, &keys, KeyLayout::OneBasedNullSlot0, None, t)
                .unwrap();
            assert_eq!(
                collect(&arr),
                vec![
                    Some(expected_first.to_string()),
                    Some("x".to_string()),
                    None
                ],
                "trim mode {mode:?}"
            );
        }
    }

    #[test]
    fn trim_host_full_unicode_whitespace() {
        // The host realisation uses full-Unicode whitespace (str::trim), so a
        // non-ASCII whitespace codepoint is stripped — the supported behaviour
        // (the ASCII-only GPU kernels are not used on this path).
        let nbsp = "\u{2009}go\u{2009}"; // U+2009 THIN SPACE on both ends
        let dict = owned(&[nbsp]);
        let keys = vec![1i32];
        let t = StringTransform::Trim {
            mode: TrimMode::Both,
        };
        let arr =
            host_transform_strings(&dict, &keys, KeyLayout::OneBasedNullSlot0, None, t).unwrap();
        assert_eq!(collect(&arr), vec![Some("go".to_string())]);
    }

    #[test]
    fn substring_trim_scalar_fn_kind_mapping() {
        use crate::plan::logical_plan::ScalarFnKind;
        assert_eq!(
            StringTransform::Substring {
                start: 1,
                length: Some(2)
            }
            .scalar_fn_kind(),
            ScalarFnKind::Substring
        );
        assert_eq!(
            StringTransform::Trim {
                mode: TrimMode::Both
            }
            .scalar_fn_kind(),
            ScalarFnKind::TrimBoth
        );
        assert_eq!(
            StringTransform::Trim {
                mode: TrimMode::Leading
            }
            .scalar_fn_kind(),
            ScalarFnKind::TrimLeading
        );
        assert_eq!(
            StringTransform::Trim {
                mode: TrimMode::Trailing
            }
            .scalar_fn_kind(),
            ScalarFnKind::TrimTrailing
        );
        assert!(StringTransform::Substring {
            start: 1,
            length: None
        }
        .is_host_realized());
        assert!(StringTransform::Trim {
            mode: TrimMode::Both
        }
        .is_host_realized());
        assert!(!StringTransform::Upper.is_host_realized());
    }

    #[test]
    fn substring_trim_kinds_have_compilable_kernels() {
        // PTX-shape wiring check: every kind a SUBSTRING/TRIM StringTransform
        // maps to has a two-pass producer that compiles (the executor uses the
        // host mirror today, but the kernels back the same `ScalarFnKind` and
        // must stay in sync with the enum mapping).
        use crate::jit::string_kernel::{
            compile_varwidth_len_pass, compile_varwidth_write_pass, len_pass_entry,
            write_pass_entry,
        };
        for t in [
            StringTransform::Substring {
                start: 1,
                length: Some(2),
            },
            StringTransform::Substring {
                start: 2,
                length: None,
            },
            StringTransform::Trim {
                mode: TrimMode::Both,
            },
            StringTransform::Trim {
                mode: TrimMode::Leading,
            },
            StringTransform::Trim {
                mode: TrimMode::Trailing,
            },
        ] {
            let kind = t.scalar_fn_kind();
            let len_ptx = compile_varwidth_len_pass(kind).expect("len pass compiles");
            let write_ptx = compile_varwidth_write_pass(kind).expect("write pass compiles");
            assert!(len_ptx.contains(&len_pass_entry(kind).unwrap()));
            assert!(write_ptx.contains(&write_pass_entry(kind).unwrap()));
        }
    }

    // ---- CASE over Utf8 host evaluation (F11) ----------------------------

    fn env_of<'a>(cols: &'a [(&str, &'a HostColumn)]) -> crate::exec::expr_agg::ColumnEnv<'a> {
        cols.iter().map(|(n, c)| (n.to_string(), *c)).collect()
    }

    #[test]
    fn case_utf8_selects_then_else_with_nulls() {
        use crate::plan::logical_plan::{BinaryOp, Expr, Literal};
        // grade column: scores 90, 50, NULL.
        // CASE WHEN score >= 80 THEN 'A' ELSE 'B' END.
        let score = HostColumn::I64(vec![Some(90), Some(50), None]);
        let cols = [("score", &score)];
        let env = env_of(&cols);
        let when = Expr::Binary {
            op: BinaryOp::GtEq,
            left: Box::new(Expr::Column("score".into())),
            right: Box::new(Expr::Literal(Literal::Int64(80))),
        };
        let then = Expr::Literal(Literal::Utf8("A".into()));
        let els = Expr::Literal(Literal::Utf8("B".into()));
        let arr = eval_case_utf8(&[(when, then)], Some(&els), &env, 3).unwrap();
        // Row 0: 90>=80 → 'A'. Row 1: 50>=80 false → 'B'. Row 2: NULL>=80 is
        // UNKNOWN → falls through to ELSE → 'B' (the ELSE value is not NULL).
        assert_eq!(
            collect(&arr),
            vec![
                Some("A".to_string()),
                Some("B".to_string()),
                Some("B".to_string())
            ]
        );
    }

    #[test]
    fn case_utf8_no_else_unmatched_is_null() {
        use crate::plan::logical_plan::{BinaryOp, Expr, Literal};
        let x = HostColumn::I32(vec![Some(1), Some(2)]);
        let cols = [("x", &x)];
        let env = env_of(&cols);
        // CASE WHEN x = 1 THEN 'one' END  (no ELSE → NULL for unmatched).
        let when = Expr::Binary {
            op: BinaryOp::Eq,
            left: Box::new(Expr::Column("x".into())),
            right: Box::new(Expr::Literal(Literal::Int32(1))),
        };
        let then = Expr::Literal(Literal::Utf8("one".into()));
        let arr = eval_case_utf8(&[(when, then)], None, &env, 2).unwrap();
        assert_eq!(collect(&arr), vec![Some("one".to_string()), None]);
    }

    #[test]
    fn case_utf8_then_value_is_a_column() {
        use crate::plan::logical_plan::{Expr, Literal};
        // CASE WHEN flag THEN name ELSE 'none' END, name has a NULL row.
        let flag = HostColumn::Bool(vec![Some(true), Some(true), Some(false)]);
        let name = HostColumn::Utf8(vec![Some("ann".into()), None, Some("ignored".into())]);
        let cols = [("flag", &flag), ("name", &name)];
        let env = env_of(&cols);
        let when = Expr::Column("flag".into());
        let then = Expr::Column("name".into());
        let els = Expr::Literal(Literal::Utf8("none".into()));
        let arr = eval_case_utf8(&[(when, then)], Some(&els), &env, 3).unwrap();
        // Row0: flag true → name 'ann'. Row1: flag true → name is NULL → the
        // chosen value is that NULL (the branch fired). Row2: flag false → ELSE.
        assert_eq!(
            collect(&arr),
            vec![Some("ann".to_string()), None, Some("none".to_string())]
        );
    }

    #[test]
    fn case_utf8_first_true_branch_wins() {
        use crate::plan::logical_plan::{BinaryOp, Expr, Literal};
        // Two overlapping branches; the first TRUE one must win.
        let x = HostColumn::I32(vec![Some(5)]);
        let cols = [("x", &x)];
        let env = env_of(&cols);
        let b1 = (
            Expr::Binary {
                op: BinaryOp::Gt,
                left: Box::new(Expr::Column("x".into())),
                right: Box::new(Expr::Literal(Literal::Int32(0))),
            },
            Expr::Literal(Literal::Utf8("pos".into())),
        );
        let b2 = (
            Expr::Binary {
                op: BinaryOp::Gt,
                left: Box::new(Expr::Column("x".into())),
                right: Box::new(Expr::Literal(Literal::Int32(3))),
            },
            Expr::Literal(Literal::Utf8("big".into())),
        );
        let arr = eval_case_utf8(&[b1, b2], None, &env, 1).unwrap();
        assert_eq!(collect(&arr), vec![Some("pos".to_string())]);
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
