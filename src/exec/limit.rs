// SPDX-License-Identifier: Apache-2.0

//! LIMIT / OFFSET executor — slice the input RecordBatch.
//!
//! `RecordBatch::slice(offset, length)` is a constant-time zero-copy
//! operation that produces a new batch sharing the same buffers.

use crate::error::BoltResult;
use crate::exec::QueryHandle;

/// Apply LIMIT (and optional OFFSET) to the input. `offset == 0` skips the
/// offset step; `limit` is interpreted as an upper bound on the result row
/// count (the returned batch has `min(limit, n_rows - offset)` rows).
pub fn execute_limit(input: QueryHandle, limit: usize, offset: usize) -> BoltResult<QueryHandle> {
    let batch = input.into_record_batch();
    let n_rows = batch.num_rows();

    if offset >= n_rows {
        // Empty result. RecordBatch::slice(n_rows, 0) is well-defined.
        let empty = batch.slice(n_rows, 0);
        return Ok(QueryHandle::from_record_batch(empty));
    }

    let take_n = std::cmp::min(limit, n_rows - offset);
    let sliced = batch.slice(offset, take_n);
    Ok(QueryHandle::from_record_batch(sliced))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_batch(values: Vec<i32>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let col = Arc::new(Int32Array::from(values));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    fn col_values(batch: &RecordBatch) -> Vec<i32> {
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("int32 column");
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }

    #[test]
    fn limit_three_of_ten_returns_three() {
        let batch = make_batch((0..10).collect());
        let handle = QueryHandle::from_record_batch(batch);
        let out = execute_limit(handle, 3, 0).unwrap();
        let result = out.into_record_batch();
        assert_eq!(result.num_rows(), 3);
        assert_eq!(col_values(&result), vec![0, 1, 2]);
    }

    #[test]
    fn limit_with_offset_two_returns_correct_window() {
        let batch = make_batch((0..10).collect());
        let handle = QueryHandle::from_record_batch(batch);
        let out = execute_limit(handle, 3, 2).unwrap();
        let result = out.into_record_batch();
        assert_eq!(result.num_rows(), 3);
        assert_eq!(col_values(&result), vec![2, 3, 4]);
    }

    #[test]
    fn limit_greater_than_n_rows_returns_all() {
        let batch = make_batch((0..5).collect());
        let handle = QueryHandle::from_record_batch(batch);
        let out = execute_limit(handle, 100, 0).unwrap();
        let result = out.into_record_batch();
        assert_eq!(result.num_rows(), 5);
        assert_eq!(col_values(&result), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn limit_with_offset_truncates_to_remaining() {
        // offset=7 of 10 rows leaves 3 remaining; limit=5 should clamp to 3.
        let batch = make_batch((0..10).collect());
        let handle = QueryHandle::from_record_batch(batch);
        let out = execute_limit(handle, 5, 7).unwrap();
        let result = out.into_record_batch();
        assert_eq!(result.num_rows(), 3);
        assert_eq!(col_values(&result), vec![7, 8, 9]);
    }

    #[test]
    fn offset_at_n_rows_returns_empty() {
        let batch = make_batch((0..4).collect());
        let handle = QueryHandle::from_record_batch(batch);
        let out = execute_limit(handle, 10, 4).unwrap();
        let result = out.into_record_batch();
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn offset_past_n_rows_returns_empty() {
        let batch = make_batch((0..4).collect());
        let handle = QueryHandle::from_record_batch(batch);
        let out = execute_limit(handle, 10, 999).unwrap();
        let result = out.into_record_batch();
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn limit_zero_returns_empty() {
        let batch = make_batch((0..10).collect());
        let handle = QueryHandle::from_record_batch(batch);
        let out = execute_limit(handle, 0, 0).unwrap();
        let result = out.into_record_batch();
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn empty_input_returns_empty() {
        let batch = make_batch(vec![]);
        let handle = QueryHandle::from_record_batch(batch);
        let out = execute_limit(handle, 10, 0).unwrap();
        let result = out.into_record_batch();
        assert_eq!(result.num_rows(), 0);
    }
}
