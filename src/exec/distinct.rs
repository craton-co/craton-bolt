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
//!     Two `NaN`s with identical bit patterns DO dedupe to one row
//!     because the row key carries the raw bits and `Eq` is bit-wise.
//!
//! Allocation strategy (review H9): the inner loop pre-downcasts each
//! column ONCE into a typed `ColumnReader` enum (a struct-of-arrays view
//! of the batch), then walks rows pulling values through the readers.
//! This avoids the per-row `Array::as_any` + `downcast_ref` vtable
//! shuffle that the old `extract_value(&dyn Array, row)` shape paid on
//! every (row, column) pair — for an N-row × K-column batch that is N·K
//! vtable lookups dropped to K. The `Vec<RowKeyValue>` per row is
//! preallocated with `n_cols` capacity (no growth re-allocs); the freshly
//! built key is moved into `HashSet::insert`, so on a miss it lives in
//! the set and on a hit it is dropped — same allocation count as before
//! but the per-row dtype dispatch is now branch-predictor friendly
//! (constant variant per column) instead of an `Array::data_type()`
//! match in the inner loop.
//!
//! Dispatch: a single host-side path. The 0.2 target (sort-based DISTINCT
//! via `gpu_sort::sort_indices_on_gpu_multi`) is tracked in ROADMAP.md;
//! it requires the input columns to already be uploaded as `GpuVec`s,
//! which the distinct executor does not have hand — that restructure is
//! deferred to the GPU-side rework.

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow_schema::DataType;

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
    /// bit-wise; `+0.0` and `-0.0` are first canonicalised to `+0.0` via
    /// [`canonicalise_f32`]. See module doc-comment.
    F32(u32),
    /// `f64` reinterpreted via `to_bits`. Same bit-wise semantics as `F32`.
    F64(u64),
    Bool(bool),
    Utf8(String),
}

/// A row's full key — one `RowKeyValue` per column, in column order.
type RowKey = Vec<RowKeyValue>;

/// Pre-downcast column reader: a typed, zero-cost view into one column of
/// the input batch. Built once per column up-front so the inner row loop
/// no longer pays `Array::as_any` + `downcast_ref` per (row, column).
enum ColumnReader<'a> {
    I32(&'a Int32Array),
    I64(&'a Int64Array),
    F32(&'a Float32Array),
    F64(&'a Float64Array),
    Bool(&'a BooleanArray),
    Utf8(&'a StringArray),
}

impl<'a> ColumnReader<'a> {
    fn new(array: &'a dyn Array) -> BoltResult<Self> {
        Ok(match array.data_type() {
            DataType::Int32 => ColumnReader::I32(array.as_any().downcast_ref().unwrap()),
            DataType::Int64 => ColumnReader::I64(array.as_any().downcast_ref().unwrap()),
            DataType::Float32 => ColumnReader::F32(array.as_any().downcast_ref().unwrap()),
            DataType::Float64 => ColumnReader::F64(array.as_any().downcast_ref().unwrap()),
            DataType::Boolean => ColumnReader::Bool(array.as_any().downcast_ref().unwrap()),
            DataType::Utf8 => ColumnReader::Utf8(array.as_any().downcast_ref().unwrap()),
            other => {
                return Err(BoltError::Type(format!(
                    "DISTINCT: unsupported dtype {other:?} — should have been caught by the planner"
                )))
            }
        })
    }

    /// Pull the value at `row` out as an owned `RowKeyValue`. NULL handling
    /// is uniform: any column variant returns `RowKeyValue::Null` for a
    /// null row. The only path that allocates is `Utf8`, which clones the
    /// underlying `&str` into a `String`.
    #[inline]
    fn value_at(&self, row: usize) -> RowKeyValue {
        match self {
            ColumnReader::I32(a) => {
                if a.is_null(row) { RowKeyValue::Null } else { RowKeyValue::I32(a.value(row)) }
            }
            ColumnReader::I64(a) => {
                if a.is_null(row) { RowKeyValue::Null } else { RowKeyValue::I64(a.value(row)) }
            }
            ColumnReader::F32(a) => {
                if a.is_null(row) {
                    RowKeyValue::Null
                } else {
                    // Canonicalise +0.0/-0.0 → +0.0 (review C12).
                    RowKeyValue::F32(canonicalise_f32(a.value(row)).to_bits())
                }
            }
            ColumnReader::F64(a) => {
                if a.is_null(row) {
                    RowKeyValue::Null
                } else {
                    // Canonicalise +0.0/-0.0 → +0.0 (review C12).
                    RowKeyValue::F64(canonicalise_f64(a.value(row)).to_bits())
                }
            }
            ColumnReader::Bool(a) => {
                if a.is_null(row) { RowKeyValue::Null } else { RowKeyValue::Bool(a.value(row)) }
            }
            ColumnReader::Utf8(a) => {
                if a.is_null(row) {
                    RowKeyValue::Null
                } else {
                    // String allocation is unavoidable in the owned-key
                    // shape; the win from H9 is that we no longer redo
                    // the downcast per row, only the clone.
                    RowKeyValue::Utf8(a.value(row).to_string())
                }
            }
        }
    }
}

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

    // Pre-downcast every column ONCE (review H9). For an N-row × K-column
    // input this turns N·K vtable lookups into K.
    let n_cols = batch.num_columns();
    let readers: Vec<ColumnReader<'_>> = batch
        .columns()
        .iter()
        .map(|c| ColumnReader::new(c.as_ref()))
        .collect::<BoltResult<Vec<_>>>()?;

    // Build an owned, typed key per row and check membership against the
    // set of already-seen keys. `HashSet::insert` returns `true` iff the
    // key was not already present — i.e. iff the row is a first occurrence.
    // The freshly-built `key` is *moved* into `insert`, so on a miss it
    // lives in the set and on a hit it is dropped — exactly one
    // `Vec<RowKeyValue>` allocation per input row.
    let mut seen: HashSet<RowKey> = HashSet::with_capacity(n_rows);
    let mut mask_bits: Vec<bool> = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        let mut key: RowKey = Vec::with_capacity(n_cols);
        for reader in &readers {
            key.push(reader.value_at(row));
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
