// SPDX-License-Identifier: Apache-2.0

//! DISTINCT executor — host-side deduplication of a RecordBatch.
//!
//! Strategy: build a hash of each row's bytes per column, accumulate
//! a `HashSet<u64>` to identify duplicates, then use the dedup mask
//! to construct a new `BooleanArray` and apply `arrow::compute::filter`.
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
///
/// Stage 3 note: host-side only, no async opportunity here. The upstream
/// executor that produced `input` has already done its own pinned/async
/// D2H, so the `RecordBatch` we receive is already settled in host
/// memory. When the GPU-side DISTINCT lands it should pick up the same
/// async memcpy + pinned D2H pattern as the projection / aggregate paths.
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
            array
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(row)
                .to_bits(),
        ),
        DataType::Float64 => h.write_u64(
            array
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(row)
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

fn arrow_err(e: arrow::error::ArrowError) -> BoltError {
    BoltError::Other(format!("arrow: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
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
