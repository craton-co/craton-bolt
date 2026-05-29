// SPDX-License-Identifier: Apache-2.0

//! `EXCEPT` / `INTERSECT` executor — host-side multiset set operations.
//!
//! Lowered from [`crate::plan::logical_plan::LogicalPlan::SetOp`] /
//! `PhysicalPlan::SetOp`. The engine executes the left and right inputs into
//! [`QueryHandle`]s and hands them here; we then keep (or drop) each *left*
//! row based on its presence in the right input.
//!
//! # Row equality and NULL handling
//!
//! We deliberately reuse the `DISTINCT` executor's [`RowKey`] /
//! [`ColumnReader`] machinery (see [`crate::exec::distinct`]) so that
//! `EXCEPT` / `INTERSECT` share *exactly* one row-equality relation with
//! `DISTINCT`, `GROUP BY`, and `JOIN`:
//!   * two NULLs in the same column position compare **equal** (the
//!     engine-wide "NULLs are not distinct" convention — note this is
//!     *different* from SQL `=`, but is the standard rule for `DISTINCT` and
//!     the set operators), and
//!   * `+0.0` / `-0.0` are canonicalised to one key and `NaN` bit-patterns are
//!     compared verbatim (see the `distinct` module docs).
//!
//! # Multiset semantics (`ALL`)
//!
//! With `all == true` the result is a *multiset* and row multiplicities follow
//! the SQL standard, where `lc` / `rc` are the number of copies of a given row
//! in the left / right inputs:
//!   * `EXCEPT ALL`    → `max(0, lc - rc)` copies (drop the first `rc` left
//!     copies of each matched row),
//!   * `INTERSECT ALL` → `min(lc, rc)` copies (keep the first `rc` left copies
//!     of each matched row).
//!
//! With `all == false` the result is a *set* (each surviving row at most once):
//!   * `EXCEPT`    → distinct left rows whose key is absent from the right,
//!   * `INTERSECT` → distinct left rows whose key is present in the right.
//!
//! In every case the output preserves the left input's first-occurrence order
//! and the left input's schema (rows are filtered, never reshaped).
//!
//! Dispatch: a single host-side path, mirroring the `DISTINCT` executor (the
//! GPU sort-based variant is the same deferred follow-up — see
//! `crate::exec::distinct`).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, BooleanArray, RecordBatch};

use crate::error::{BoltError, BoltResult};
use crate::exec::distinct::{ColumnReader, RowKey};
use crate::exec::QueryHandle;
use crate::plan::logical_plan::SetOpKind;

/// Execute an `EXCEPT` / `INTERSECT` (optionally `ALL`) over the two already-
/// materialised inputs.
///
/// `left` / `right` are the executed child plans; `op` selects the operator
/// and `all` selects the multiset (`true`) vs set (`false`) variant. The
/// result handle carries the left input's schema with the surviving rows in
/// left first-occurrence order.
pub fn execute_setop(
    left: QueryHandle,
    right: QueryHandle,
    op: SetOpKind,
    all: bool,
) -> BoltResult<QueryHandle> {
    let left_batch = left.into_record_batch();
    let right_batch = right.into_record_batch();

    // The logical planner already enforced schema compatibility (same field
    // count + per-field dtypes); re-check the column count defensively so a
    // hand-built physical plan can't silently mis-key rows.
    if left_batch.num_columns() != right_batch.num_columns() {
        return Err(BoltError::Plan(format!(
            "{} inputs have a different number of columns ({} vs {})",
            op.keyword(),
            left_batch.num_columns(),
            right_batch.num_columns(),
        )));
    }

    let n_left = left_batch.num_rows();
    // Empty-left shortcut: the result is empty for both operators. Re-wrap the
    // (already-empty) left batch so the output schema is preserved.
    if n_left == 0 {
        return Ok(QueryHandle::from_record_batch(left_batch));
    }

    // Multiset of right-side row keys → count of occurrences. For the
    // set-returning (`all == false`) variants only presence matters, but a
    // count map serves both, so we build it once.
    let right_counts = build_key_counts(&right_batch)?;

    // Pre-downcast the left columns once (mirrors the DISTINCT executor's
    // `ColumnReader` allocation strategy: N·K vtable lookups → K).
    let left_readers: Vec<ColumnReader<'_>> = left_batch
        .columns()
        .iter()
        .map(|c| ColumnReader::new(c.as_ref()))
        .collect::<BoltResult<Vec<_>>>()?;

    // Decide, per left row, whether it survives into the output. The decision
    // depends on the operator + the `ALL` flag; see the module docs.
    let mut keep: Vec<bool> = Vec::with_capacity(n_left);
    // For the multiset (`ALL`) variants we consume right-side copies as we go,
    // so clone the counts into a mutable working map.
    let mut remaining = right_counts.clone();
    // For the set (`!all`) variants we must dedupe the left side ourselves so
    // each surviving key appears at most once.
    let mut emitted: HashMap<RowKey, ()> = HashMap::new();

    for row in 0..n_left {
        let key: RowKey = left_readers.iter().map(|r| r.value_at(row)).collect();
        let in_right = right_counts.get(&key).copied().unwrap_or(0) > 0;
        let survive = match (op, all) {
            // EXCEPT ALL: keep up to max(0, lc - rc) copies — drop a left copy
            // for each remaining right copy, keep the rest.
            (SetOpKind::Except, true) => {
                match remaining.get_mut(&key) {
                    Some(c) if *c > 0 => {
                        *c -= 1;
                        false
                    }
                    _ => true,
                }
            }
            // INTERSECT ALL: keep min(lc, rc) copies — keep a left copy while
            // right copies remain, drop the rest.
            (SetOpKind::Intersect, true) => match remaining.get_mut(&key) {
                Some(c) if *c > 0 => {
                    *c -= 1;
                    true
                }
                _ => false,
            },
            // EXCEPT (set): distinct left rows whose key is absent from right.
            (SetOpKind::Except, false) => {
                !in_right && emitted.insert(key.clone(), ()).is_none()
            }
            // INTERSECT (set): distinct left rows whose key is present in right.
            (SetOpKind::Intersect, false) => {
                in_right && emitted.insert(key.clone(), ()).is_none()
            }
        };
        keep.push(survive);
    }

    // Apply the keep-mask to every left column. `arrow::compute::filter`
    // preserves row order, so the output keeps left first-occurrence order.
    let mask = BooleanArray::from(keep);
    let filtered: Vec<Arc<dyn Array>> = left_batch
        .columns()
        .iter()
        .map(|c| arrow::compute::filter(c.as_ref(), &mask).map_err(arrow_err))
        .collect::<BoltResult<Vec<_>>>()?;
    let out = RecordBatch::try_new(left_batch.schema(), filtered).map_err(arrow_err)?;
    Ok(QueryHandle::from_record_batch(out))
}

/// Build a `RowKey -> occurrence count` multiset over every row of `batch`.
///
/// Uses the same [`ColumnReader`] / [`RowKey`] machinery as the `DISTINCT`
/// executor so the keys are byte-for-byte comparable with the ones built over
/// the left input in [`execute_setop`].
fn build_key_counts(batch: &RecordBatch) -> BoltResult<HashMap<RowKey, usize>> {
    let n_rows = batch.num_rows();
    let mut counts: HashMap<RowKey, usize> = HashMap::with_capacity(n_rows);
    if n_rows == 0 {
        return Ok(counts);
    }
    let readers: Vec<ColumnReader<'_>> = batch
        .columns()
        .iter()
        .map(|c| ColumnReader::new(c.as_ref()))
        .collect::<BoltResult<Vec<_>>>()?;
    for row in 0..n_rows {
        let key: RowKey = readers.iter().map(|r| r.value_at(row)).collect();
        *counts.entry(key).or_insert(0) += 1;
    }
    Ok(counts)
}

fn arrow_err(e: arrow::error::ArrowError) -> BoltError {
    BoltError::Other(format!("arrow: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    /// One-column Int32 handle from the given (nullable) values.
    fn int32_handle(values: Vec<Option<i32>>) -> QueryHandle {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        let arr: Arc<dyn Array> = Arc::new(Int32Array::from(values));
        QueryHandle::from_record_batch(RecordBatch::try_new(schema, vec![arr]).unwrap())
    }

    fn col_to_vec(h: &QueryHandle) -> Vec<Option<i32>> {
        let batch = h.record_batch();
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32 column");
        (0..arr.len())
            .map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) })
            .collect()
    }

    #[test]
    fn except_distinct_removes_right_keys_and_dedupes() {
        // left {1,2,2,3,4}, right {2,4} → EXCEPT (set) = {1,3}
        let l = int32_handle(vec![Some(1), Some(2), Some(2), Some(3), Some(4)]);
        let r = int32_handle(vec![Some(2), Some(4)]);
        let out = execute_setop(l, r, SetOpKind::Except, false).unwrap();
        assert_eq!(col_to_vec(&out), vec![Some(1), Some(3)]);
    }

    #[test]
    fn except_all_keeps_multiplicity_difference() {
        // left {2,2,2,3}, right {2} → EXCEPT ALL = {2,2,3}
        let l = int32_handle(vec![Some(2), Some(2), Some(2), Some(3)]);
        let r = int32_handle(vec![Some(2)]);
        let out = execute_setop(l, r, SetOpKind::Except, true).unwrap();
        assert_eq!(col_to_vec(&out), vec![Some(2), Some(2), Some(3)]);
    }

    #[test]
    fn intersect_distinct_keeps_common_keys_once() {
        // left {1,2,2,3}, right {2,3,3} → INTERSECT (set) = {2,3}
        let l = int32_handle(vec![Some(1), Some(2), Some(2), Some(3)]);
        let r = int32_handle(vec![Some(2), Some(3), Some(3)]);
        let out = execute_setop(l, r, SetOpKind::Intersect, false).unwrap();
        assert_eq!(col_to_vec(&out), vec![Some(2), Some(3)]);
    }

    #[test]
    fn intersect_all_keeps_min_multiplicity() {
        // left {2,2,2}, right {2,2} → INTERSECT ALL = {2,2}
        let l = int32_handle(vec![Some(2), Some(2), Some(2)]);
        let r = int32_handle(vec![Some(2), Some(2)]);
        let out = execute_setop(l, r, SetOpKind::Intersect, true).unwrap();
        assert_eq!(col_to_vec(&out), vec![Some(2), Some(2)]);
    }

    #[test]
    fn null_rows_compare_equal_across_inputs() {
        // EXCEPT: a NULL on the left is removed by a NULL on the right
        // (engine-wide "NULLs are not distinct" rule shared with DISTINCT).
        let l = int32_handle(vec![None, Some(1), None]);
        let r = int32_handle(vec![None]);
        // EXCEPT (set): distinct left {NULL, 1} minus right {NULL} = {1}.
        let out = execute_setop(l, r, SetOpKind::Except, false).unwrap();
        assert_eq!(col_to_vec(&out), vec![Some(1)]);
    }

    #[test]
    fn empty_left_yields_empty_result() {
        let l = int32_handle(vec![]);
        let r = int32_handle(vec![Some(1)]);
        let out = execute_setop(l, r, SetOpKind::Intersect, false).unwrap();
        assert_eq!(out.num_rows(), 0);
    }

    #[test]
    fn except_with_empty_right_is_distinct_left() {
        // EXCEPT against an empty right is just DISTINCT(left).
        let l = int32_handle(vec![Some(1), Some(1), Some(2)]);
        let r = int32_handle(vec![]);
        let out = execute_setop(l, r, SetOpKind::Except, false).unwrap();
        assert_eq!(col_to_vec(&out), vec![Some(1), Some(2)]);
    }

    #[test]
    fn multi_column_keys_match_whole_row() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, false),
            Field::new("n", DataType::Int32, false),
        ]));
        let l_s: Arc<dyn Array> = Arc::new(StringArray::from(vec!["a", "b", "a"]));
        let l_n: Arc<dyn Array> = Arc::new(Int32Array::from(vec![1, 2, 1]));
        let left = QueryHandle::from_record_batch(
            RecordBatch::try_new(schema.clone(), vec![l_s, l_n]).unwrap(),
        );
        // Right has ("a",1) only; ("a",2) is NOT in right.
        let r_s: Arc<dyn Array> = Arc::new(StringArray::from(vec!["a"]));
        let r_n: Arc<dyn Array> = Arc::new(Int32Array::from(vec![1]));
        let right = QueryHandle::from_record_batch(
            RecordBatch::try_new(schema, vec![r_s, r_n]).unwrap(),
        );
        // EXCEPT (set): distinct left {(a,1),(b,2)} minus {(a,1)} = {(b,2)}.
        let out = execute_setop(left, right, SetOpKind::Except, false).unwrap();
        assert_eq!(out.num_rows(), 1);
        let batch = out.record_batch();
        let s = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(s.value(0), "b");
    }
}
