// SPDX-License-Identifier: Apache-2.0

//! Host-side string dictionary paired with on-device i32 indices.
//!
//! Variable-width strings are a poor fit for a fused-codegen GPU kernel: a
//! dynamic offset/byte layout forces every comparison to dereference, which
//! defeats coalesced loads. Instead we dictionary-encode strings on the host
//! and ship only fixed-width 32-bit indices to the device. Predicates like
//! `region = 'US'` reduce to integer equality against the index of `'US'` —
//! work the codegen path already does well.
//!
//! Layout convention:
//!   * Index `0` is reserved for SQL `NULL`. It NEVER appears in
//!     `dictionary[]` and is never returned by `index_of`.
//!   * Real strings start at index `1`. The i-th unique non-null string is
//!     stored at `dictionary[i - 1]`.
//!   * Indices are `i32`. Allowing > `i32::MAX` distinct strings would break
//!     downstream codegen; we surface that as a `JavelinError::Other`.

use std::collections::HashMap;

use arrow_array::{Array, StringArray};

use crate::cuda::GpuVec;
use crate::error::{JavelinError, JavelinResult};

/// On-host string dictionary + on-device i32 indices.
///
/// Strings are encoded as `i32` indices on the GPU; the host holds the
/// bidirectional mapping for decoding. Index `0` is reserved for NULL; real
/// strings occupy indices `1..=dictionary.len()`. The implementation never
/// emits a negative index.
pub struct DictionaryColumn {
    /// Host-side dictionary: position `i` → the `(i + 1)`-th index's string.
    /// Equivalently, `dictionary[k - 1]` is the string for GPU index `k`.
    pub dictionary: Vec<String>,
    /// GPU-side indices: one `i32` per source row. `0` means NULL.
    pub indices: GpuVec<i32>,
    /// Number of source rows.
    pub n_rows: usize,
}

impl DictionaryColumn {
    /// Encode an Arrow `StringArray` as a dictionary and upload the indices.
    ///
    /// Nulls in `arr` map to index `0`. Distinct non-null strings are
    /// deduplicated and assigned sequential indices starting at `1`, in
    /// first-occurrence order.
    pub fn from_string_array(arr: &StringArray) -> JavelinResult<Self> {
        let n_rows = arr.len();
        let mut dictionary: Vec<String> = Vec::new();
        let mut lookup: HashMap<String, i32> = HashMap::new();
        let mut indices: Vec<i32> = Vec::with_capacity(n_rows);

        for i in 0..n_rows {
            if arr.is_null(i) {
                indices.push(0);
                continue;
            }
            let s = arr.value(i);
            if let Some(&idx) = lookup.get(s) {
                indices.push(idx);
            } else {
                // Next index = current dictionary length + 1 (slot 0 reserved for NULL).
                let next_len = dictionary.len().checked_add(1).ok_or_else(|| {
                    JavelinError::Other(
                        "dictionary overflow: more than usize::MAX unique strings".into(),
                    )
                })?;
                if next_len > i32::MAX as usize {
                    return Err(JavelinError::Other(format!(
                        "dictionary overflow: more than {} unique strings (i32 index space)",
                        i32::MAX
                    )));
                }
                let idx = next_len as i32;
                let owned = s.to_string();
                dictionary.push(owned.clone());
                lookup.insert(owned, idx);
                indices.push(idx);
            }
        }

        let device_indices = GpuVec::<i32>::from_slice(&indices)?;
        Ok(Self {
            dictionary,
            indices: device_indices,
            n_rows,
        })
    }

    /// Lookup the index of a literal string in the dictionary.
    ///
    /// Returns `Some(index)` if `s` was seen during construction, or `None`
    /// otherwise. A `None` here is not an error: a predicate against an
    /// unknown literal trivially matches no rows.
    pub fn index_of(&self, s: &str) -> Option<i32> {
        // Linear scan keeps `index_of` O(dict) but avoids carrying the
        // construction-time HashMap. Literal lookups happen once per query, so
        // the asymptotic cost is dominated by row count, not dictionary size.
        self.dictionary
            .iter()
            .position(|d| d == s)
            // position is 0-based; real indices start at 1.
            .map(|p| (p as i32) + 1)
    }

    /// Download indices and reconstruct a `StringArray`.
    ///
    /// Index `0` becomes a SQL `NULL`. Indices outside `1..=dictionary.len()`
    /// surface as `JavelinError::Other` — that would indicate a kernel wrote
    /// something the host dictionary cannot decode.
    pub fn to_string_array(&self) -> JavelinResult<StringArray> {
        let host_indices: Vec<i32> = self.indices.to_vec()?;
        let mut out: Vec<Option<&str>> = Vec::with_capacity(host_indices.len());
        for &idx in &host_indices {
            if idx == 0 {
                out.push(None);
            } else if idx < 0 {
                return Err(JavelinError::Other(format!(
                    "dictionary decode: negative index {} (NULL is encoded as 0)",
                    idx
                )));
            } else {
                let pos = (idx as usize) - 1;
                let s = self.dictionary.get(pos).ok_or_else(|| {
                    JavelinError::Other(format!(
                        "dictionary decode: index {} out of range (dictionary size {})",
                        idx,
                        self.dictionary.len()
                    ))
                })?;
                out.push(Some(s.as_str()));
            }
        }
        Ok(StringArray::from(out))
    }
}
