// SPDX-License-Identifier: Apache-2.0

//! DISTINCT executor — host-side deduplication of a RecordBatch.
//!
//! Strategy: build a hash of each row's bytes per column, accumulate
//! a `HashSet<u64>` to identify duplicates, then use the dedup mask
//! to construct a new `BooleanArray` and apply `arrow::compute::filter`.
//!
//! Float semantics (review C12 alignment):
//!   * `+0.0` and `-0.0` are CANONICALISED to a single representation
//!     (`+0.0`) before hashing, so they dedupe to one row. This matches
//!     SQL/IEEE comparison semantics (`+0.0 == -0.0`) and what DuckDB
//!     does, and lines up with the `groupby` and host-side `join`
//!     executors which apply the same canonicalisation. See
//!     `canonicalise_f32` / `canonicalise_f64` below.
//!   * `NaN` bit patterns are LEFT AS-IS. The host-side canonicalisation
//!     uses `if x == 0.0 { 0.0 } else { x }` which evaluates `false` for
//!     every NaN (per IEEE) and therefore preserves NaN bit patterns
//!     verbatim. Documented SQL semantics: `NaN != NaN` (also DuckDB).
//!
//! GPU-side DISTINCT (via a sort + run-length encoding kernel) is a
//! 0.2 target — see ROADMAP.md.

use std::collections::HashSet;
use std::hash::Hasher;
use std::sync::Arc;

use arrow_array::{Array, BooleanArray, RecordBatch};

use crate::error::{BoltError, BoltResult};
use crate::exec::QueryHandle;

/// Apply DISTINCT to the input handle, returning a new handle whose
/// RecordBatch has duplicate rows removed (first-occurrence wins).
///
/// Host-side implementation. For wide schemas or large row counts this
/// is the slow path; the 0.2 release will add a GPU sort-based variant.
pub fn execute_distinct(input: QueryHandle) -> BoltResult<QueryHandle> {
    let batch = input.into_record_batch();
    let n_rows = batch.num_rows();
    if n_rows == 0 {
        // Trivial: re-wrap.
        return Ok(QueryHandle::from_record_batch(batch));
    }
    // Build per-row hashes by combining the hash of each column at row i.
    let mut seen: HashSet<u64> = HashSet::with_capacity(n_rows);
    let mut mask_bits: Vec<bool> = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for col in batch.columns() {
            hash_array_row(col.as_ref(), row, &mut h);
        }
        let digest = h.finish();
        mask_bits.push(seen.insert(digest));
    }
    let mask = BooleanArray::from(mask_bits);
    let filtered_cols: Vec<Arc<dyn Array>> = batch
        .columns()
        .iter()
        .map(|c| arrow::compute::filter(c.as_ref(), &mask).map_err(arrow_err))
        .collect::<BoltResult<Vec<_>>>()?;
    let out = RecordBatch::try_new(batch.schema(), filtered_cols).map_err(arrow_err)?;
    Ok(QueryHandle::from_record_batch(out))
}

/// Mix the byte representation of column `array` at row `row` into hasher `h`.
/// Handles every primitive Arrow type the engine produces.
fn hash_array_row(array: &dyn Array, row: usize, h: &mut impl Hasher) {
    use arrow_array::*;
    use arrow_schema::DataType;
    h.write_u8(if array.is_null(row) { 0 } else { 1 });
    if array.is_null(row) {
        return;
    }
    match array.data_type() {
        DataType::Int32 => {
            h.write_i32(array.as_any().downcast_ref::<Int32Array>().unwrap().value(row))
        }
        DataType::Int64 => {
            h.write_i64(array.as_any().downcast_ref::<Int64Array>().unwrap().value(row))
        }
        DataType::Float32 => h.write_u32(
            canonicalise_f32(
                array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .value(row),
            )
            .to_bits(),
        ),
        DataType::Float64 => h.write_u64(
            canonicalise_f64(
                array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .value(row),
            )
            .to_bits(),
        ),
        DataType::Boolean => h.write_u8(
            array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(row) as u8,
        ),
        DataType::Utf8 => h.write(
            array
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(row)
                .as_bytes(),
        ),
        other => panic!(
            "DISTINCT: unsupported dtype {:?} — should have been caught by the planner",
            other
        ),
    }
}

/// Collapse `-0.0` to `+0.0` so that signed-zero pairs hash identically
/// under DISTINCT. Preserves every other bit pattern, including the full
/// space of `NaN` payloads (the predicate `x == 0.0` is `false` for any
/// `NaN`). Mirrors the host-side canonicalisation applied in
/// `groupby::load_key_column_bits` and `join::extract_key` so that
/// DISTINCT, GROUP BY, and JOIN share one equivalence relation for
/// floats.
#[inline]
pub(crate) fn canonicalise_f64(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x }
}

/// `f32` analogue of [`canonicalise_f64`]; same shape, same rationale.
#[inline]
pub(crate) fn canonicalise_f32(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { x }
}

fn arrow_err(e: arrow::error::ArrowError) -> BoltError {
    BoltError::Other(format!("arrow: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    /// Build a one-column Int32 batch from the given values.
    fn int32_batch(values: Vec<Option<i32>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        let arr: Arc<dyn Array> = Arc::new(Int32Array::from(values));
        RecordBatch::try_new(schema, vec![arr]).unwrap()
    }

    /// Extract the Int32 column at index 0 as a Vec<Option<i32>>.
    fn col_to_vec(batch: &RecordBatch, col: usize) -> Vec<Option<i32>> {
        let arr = batch
            .column(col)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("expected Int32 column");
        (0..arr.len())
            .map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) })
            .collect()
    }

    #[test]
    fn distinct_int32_no_dups_returns_all_rows() {
        let batch = int32_batch(vec![Some(1), Some(2), Some(3), Some(4)]);
        let input = QueryHandle::from_record_batch(batch);
        let out = execute_distinct(input).unwrap();
        let out_batch = out.into_record_batch();
        assert_eq!(out_batch.num_rows(), 4);
        assert_eq!(
            col_to_vec(&out_batch, 0),
            vec![Some(1), Some(2), Some(3), Some(4)]
        );
    }

    #[test]
    fn distinct_int32_with_dups_drops_duplicates() {
        let batch = int32_batch(vec![Some(1), Some(2), Some(1), Some(3), Some(2)]);
        let input = QueryHandle::from_record_batch(batch);
        let out = execute_distinct(input).unwrap();
        let out_batch = out.into_record_batch();
        assert_eq!(out_batch.num_rows(), 3);
        // first-occurrence wins
        assert_eq!(
            col_to_vec(&out_batch, 0),
            vec![Some(1), Some(2), Some(3)]
        );
    }

    #[test]
    fn distinct_preserves_first_occurrence_order() {
        let batch = int32_batch(vec![Some(7), Some(3), Some(5), Some(3), Some(7), Some(9)]);
        let input = QueryHandle::from_record_batch(batch);
        let out = execute_distinct(input).unwrap();
        let out_batch = out.into_record_batch();
        assert_eq!(
            col_to_vec(&out_batch, 0),
            vec![Some(7), Some(3), Some(5), Some(9)]
        );
    }

    #[test]
    fn distinct_empty_input_is_empty() {
        let batch = int32_batch(vec![]);
        let input = QueryHandle::from_record_batch(batch);
        let out = execute_distinct(input).unwrap();
        assert_eq!(out.num_rows(), 0);
    }

    #[test]
    fn distinct_handles_nulls() {
        // Two NULLs in the same column should hash identically and dedupe to one.
        let batch = int32_batch(vec![None, Some(1), None, Some(1), Some(2)]);
        let input = QueryHandle::from_record_batch(batch);
        let out = execute_distinct(input).unwrap();
        let out_batch = out.into_record_batch();
        assert_eq!(out_batch.num_rows(), 3);
        assert_eq!(col_to_vec(&out_batch, 0), vec![None, Some(1), Some(2)]);
    }

    /// Review C12: `+0.0` and `-0.0` belong to the same equivalence
    /// class for DISTINCT (matches SQL/IEEE and DuckDB). Two rows
    /// holding signed-zero pairs must dedupe to one row.
    #[test]
    fn distinct_signed_zero_dedupes_to_one_row() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "f",
            DataType::Float64,
            false,
        )]));
        let arr: Arc<dyn Array> =
            Arc::new(Float64Array::from(vec![0.0_f64, -0.0_f64, 0.0_f64]));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        // {+0.0, -0.0, +0.0} all collapse to the canonical +0.0 key, so
        // only the first row survives.
        assert_eq!(out.into_record_batch().num_rows(), 1);
    }

    /// Review C12: `NaN` is left as-is — the canonicalisation only
    /// touches signed zeros, so two `NaN`s with the SAME bit pattern
    /// still hash equal (the row carries the raw bits) and dedupe, but
    /// the canonicalisation does NOT collapse NaN-vs-not-NaN.
    #[test]
    fn distinct_nan_canonicalisation_is_noop() {
        // canonicalise_f64 must preserve NaN bit-for-bit.
        let nan_in = f64::from_bits(0x7ff8_0000_0000_0001); // a quiet NaN
        let nan_out = canonicalise_f64(nan_in);
        assert!(nan_out.is_nan());
        assert_eq!(nan_in.to_bits(), nan_out.to_bits());
        // Signed-zero canonicalisation does happen.
        assert_eq!(canonicalise_f64(-0.0_f64).to_bits(), 0.0_f64.to_bits());
        assert_eq!(canonicalise_f32(-0.0_f32).to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn distinct_multi_column_utf8_and_int() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, false),
            Field::new("n", DataType::Int32, false),
        ]));
        let s: Arc<dyn Array> =
            Arc::new(StringArray::from(vec!["a", "b", "a", "a", "b"]));
        let n: Arc<dyn Array> = Arc::new(Int32Array::from(vec![1, 2, 1, 3, 2]));
        let batch = RecordBatch::try_new(schema, vec![s, n]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        let out_batch = out.into_record_batch();
        // ("a",1), ("b",2), ("a",3) — three uniques.
        assert_eq!(out_batch.num_rows(), 3);
    }
}
