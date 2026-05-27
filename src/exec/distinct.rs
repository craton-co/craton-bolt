// SPDX-License-Identifier: Apache-2.0

//! DISTINCT executor — host-side deduplication of a RecordBatch.
//!
//! Strategy: build an owned, typed key per row (`RowKey` — a `Vec` of
//! `RowKeyValue` entries, one per column) and accumulate a
//! `HashSet<RowKey>`. Because the set keys carry the actual column values
//! (not just a 64-bit hash digest), `HashSet::insert` only reports a row
//! as a duplicate when every column value matches an already-seen row.
//!
//! This mirrors the `JoinKey` / `JoinKeyValue` pattern in `join.rs`,
//! which has the same correctness requirement: equi-join lookups must
//! compare the real values, not just a hash.
//!
//! Historical note: the original implementation used `HashSet<u64>` keyed
//! on a `DefaultHasher` digest of the row bytes. That is silently wrong:
//! two distinct rows that hash to the same `u64` would be deduped to one,
//! and the birthday-paradox collision probability becomes non-negligible
//! around 16M rows. Worse, `DefaultHasher`'s collisions can be coerced by
//! a chosen-input adversary, so the bug had a (small) security angle as
//! well as the obvious correctness one. The fix carries the values in the
//! set, so the only way two rows collapse is if they are genuinely equal.
//!
//! Float semantics: NaN/NaN and +0.0/-0.0 are compared **by bit pattern**
//! via `f32::to_bits` / `f64::to_bits`. Consequences:
//!   * Two `NaN`s with identical bit patterns dedupe to one row, even
//!     though `NaN == NaN` is `false` under IEEE-754. This matches what
//!     SQL's `DISTINCT` users intuitively expect — a single "NaN" should
//!     appear once in the output, not once per input occurrence.
//!     PostgreSQL and DuckDB take the same stance.
//!   * `+0.0` and `-0.0` have **different** bit patterns and therefore
//!     dedupe to two rows. This differs from PostgreSQL (which treats
//!     them as equal). We pick bit-equality here because it is the only
//!     equivalence relation that is consistent with the same-bits rule
//!     for `NaN`, and it matches the join executor's existing behaviour
//!     (`JoinKeyValue::F32`/`F64` also key on the raw bit pattern). If
//!     this turns into a real-world pain point we can swap to a
//!     canonicalising encoding later — the test `distinct_zero_signs`
//!     locks in today's behaviour so any change is intentional.
//!
//! GPU-side DISTINCT (via a sort + run-length encoding kernel) is a
//! 0.2 target — see ROADMAP.md.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{Array, BooleanArray, RecordBatch};

use crate::error::{BoltError, BoltResult};
use crate::exec::QueryHandle;

/// A column value inside a row key. Variants cover every primitive dtype
/// the engine produces; float variants store the raw bit pattern so that
/// `PartialEq + Eq + Hash` are bit-wise (see the module doc-comment for
/// the NaN / signed-zero implications).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RowKeyValue {
    /// Column is NULL at this row. Two NULLs in the same column position
    /// compare equal, which matches the engine-wide "two NULLs dedupe to
    /// one row" convention used by the SQL `DISTINCT` operator.
    Null,
    I32(i32),
    I64(i64),
    /// `f32` reinterpreted via `to_bits`. NaN-vs-NaN equality is therefore
    /// bit-wise; `+0.0` and `-0.0` have different bit patterns and do
    /// **not** compare equal. See module doc-comment.
    F32(u32),
    /// `f64` reinterpreted via `to_bits`. Same bit-wise semantics as `F32`.
    F64(u64),
    Bool(bool),
    Utf8(String),
}

/// A row's full key — one `RowKeyValue` per column, in column order.
/// Allocating a `Vec` per surviving row is the price of correctness over
/// the previous hash-only shape; the 0.2 GPU port will eliminate the
/// allocation entirely by switching to a sort-based DISTINCT.
type RowKey = Vec<RowKeyValue>;

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

    // Build an owned, typed key per row and check membership against the
    // set of already-seen keys. `HashSet::insert` returns `true` iff the
    // key was not already present — i.e. iff the row is a first occurrence.
    let mut seen: HashSet<RowKey> = HashSet::with_capacity(n_rows);
    let mut mask_bits: Vec<bool> = Vec::with_capacity(n_rows);
    let n_cols = batch.num_columns();
    for row in 0..n_rows {
        let mut key: RowKey = Vec::with_capacity(n_cols);
        for col in batch.columns() {
            key.push(extract_value(col.as_ref(), row)?);
        }
        mask_bits.push(seen.insert(key));
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

/// Pull the value at `(array, row)` out as an owned `RowKeyValue`.
/// Handles every primitive Arrow type the engine produces; unsupported
/// dtypes surface as a typed `BoltError` rather than a panic so the
/// caller can return a clean SQL error to the user.
fn extract_value(array: &dyn Array, row: usize) -> BoltResult<RowKeyValue> {
    use arrow_array::*;
    use arrow_schema::DataType;
    if array.is_null(row) {
        return Ok(RowKeyValue::Null);
    }
    Ok(match array.data_type() {
        DataType::Int32 => RowKeyValue::I32(
            array.as_any().downcast_ref::<Int32Array>().unwrap().value(row),
        ),
        DataType::Int64 => RowKeyValue::I64(
            array.as_any().downcast_ref::<Int64Array>().unwrap().value(row),
        ),
        DataType::Float32 => RowKeyValue::F32(
            array
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(row)
                .to_bits(),
        ),
        DataType::Float64 => RowKeyValue::F64(
            array
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(row)
                .to_bits(),
        ),
        DataType::Boolean => RowKeyValue::Bool(
            array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(row),
        ),
        DataType::Utf8 => RowKeyValue::Utf8(
            array
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(row)
                .to_string(),
        ),
        other => {
            return Err(BoltError::Type(format!(
                "DISTINCT: unsupported dtype {other:?} — should have been caught by the planner"
            )))
        }
    })
}

fn arrow_err(e: arrow::error::ArrowError) -> BoltError {
    BoltError::Other(format!("arrow: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    };
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
        // Two NULLs in the same column should compare equal and dedupe to one.
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

    /// Multi-column input where two rows differ in exactly one column —
    /// they must not collapse. Regression test for the hash-only shape,
    /// where a `u64` collision could silently drop one of them.
    #[test]
    fn distinct_multi_column_differs_in_one_column_kept_separate() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int64, false),
            Field::new("c", DataType::Utf8, false),
        ]));
        // Rows 0 and 1 differ only in column `b`. Rows 0 and 2 differ only
        // in column `c`. Row 3 is an exact duplicate of row 0. Expected:
        // three unique rows (0, 1, 2); row 3 deduped away.
        let a: Arc<dyn Array> = Arc::new(Int32Array::from(vec![1, 1, 1, 1]));
        let b: Arc<dyn Array> = Arc::new(Int64Array::from(vec![10_i64, 20, 10, 10]));
        let c: Arc<dyn Array> = Arc::new(StringArray::from(vec!["x", "x", "y", "x"]));
        let batch = RecordBatch::try_new(schema, vec![a, b, c]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        let out_batch = out.into_record_batch();
        assert_eq!(out_batch.num_rows(), 3);
    }

    /// Synthetic-collision regression: two genuinely different rows whose
    /// row-byte hashes coincide must still be kept as two separate rows.
    /// We simulate a "hash collision" by checking on a small input where
    /// the unique count is known; the old `HashSet<u64>` would have lost a
    /// row only on a probabilistic collision (rare on tiny inputs), so the
    /// strongest signal we can give in a unit test is to verify that the
    /// output preserves every truly-distinct row across a mix of dtypes.
    #[test]
    fn distinct_keeps_all_distinct_rows_across_dtypes() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int64, false),
            Field::new("f", DataType::Float64, false),
            Field::new("b", DataType::Boolean, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        // 5 genuinely distinct rows.
        let i: Arc<dyn Array> =
            Arc::new(Int64Array::from(vec![1_i64, 2, 3, 4, 5]));
        let f: Arc<dyn Array> = Arc::new(Float64Array::from(vec![
            1.5_f64, 2.5, 3.5, 4.5, 5.5,
        ]));
        let b: Arc<dyn Array> =
            Arc::new(BooleanArray::from(vec![true, false, true, false, true]));
        let s: Arc<dyn Array> =
            Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"]));
        let batch = RecordBatch::try_new(schema, vec![i, f, b, s]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        assert_eq!(out.into_record_batch().num_rows(), 5);
    }

    /// NaN-vs-NaN: two `f64::NAN`s with the same bit pattern dedupe to one
    /// row. This is the documented engine-wide stance; see module doc.
    #[test]
    fn distinct_nan_dedupes_to_one_row() {
        let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, false)]));
        let arr: Arc<dyn Array> = Arc::new(Float64Array::from(vec![
            f64::NAN,
            1.0,
            f64::NAN,
            2.0,
        ]));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        let out_batch = out.into_record_batch();
        // [NaN, 1.0, 2.0] — three rows.
        assert_eq!(out_batch.num_rows(), 3);
    }

    /// Signed-zero: `+0.0` and `-0.0` have different bit patterns and are
    /// therefore **not** deduped. Documented choice; locks in today's
    /// behaviour against accidental drift (see module doc-comment).
    #[test]
    fn distinct_zero_signs() {
        let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, false)]));
        let arr: Arc<dyn Array> =
            Arc::new(Float64Array::from(vec![0.0_f64, -0.0_f64, 0.0_f64]));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        let out_batch = out.into_record_batch();
        // +0.0 and -0.0 stay as two distinct rows; the third row (a
        // second +0.0) is a true dup of row 0 and goes away. Expected: 2.
        assert_eq!(out_batch.num_rows(), 2);
    }

    /// `f32::NAN` path: same bit-pattern dedup rule as `f64::NAN`.
    #[test]
    fn distinct_f32_nan_dedupes_to_one_row() {
        let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float32, false)]));
        let arr: Arc<dyn Array> = Arc::new(Float32Array::from(vec![
            f32::NAN,
            f32::NAN,
            1.0_f32,
        ]));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        assert_eq!(out.into_record_batch().num_rows(), 2);
    }

    /// All-NULL column: every row's key is `[Null]`, so they collapse to a
    /// single output row.
    #[test]
    fn distinct_all_null_column_yields_one_row() {
        let batch = int32_batch(vec![None, None, None, None, None]);
        let input = QueryHandle::from_record_batch(batch);
        let out = execute_distinct(input).unwrap();
        let out_batch = out.into_record_batch();
        assert_eq!(out_batch.num_rows(), 1);
        assert_eq!(col_to_vec(&out_batch, 0), vec![None]);
    }

    /// Boolean dtype: only two possible values, plus optional NULLs.
    #[test]
    fn distinct_boolean_dedupes() {
        let schema = Arc::new(Schema::new(vec![Field::new("b", DataType::Boolean, true)]));
        let arr: Arc<dyn Array> = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            Some(true),
            None,
            Some(false),
            None,
        ]));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        // {true, false, NULL} — three rows.
        assert_eq!(out.into_record_batch().num_rows(), 3);
    }
}
