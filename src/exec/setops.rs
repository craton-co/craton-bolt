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
use std::sync::{Arc, OnceLock};

use arrow_array::{Array, BooleanArray, RecordBatch};

use crate::error::{BoltError, BoltResult};
use crate::exec::distinct::{ColumnReader, RowKey};
use crate::exec::QueryHandle;
use crate::plan::logical_plan::SetOpKind;

/// Host-side cap on the number of distinct row keys the set-op key map may
/// hold, mirroring the `DISTINCT` executor's `DISTINCT_HOST_MAX_ROWS` guard
/// (`distinct.rs`).
///
/// Rationale: [`build_key_counts`] builds a `HashMap<RowKey, usize>` over the
/// *entire* right input (and [`execute_setop`] builds an `emitted` set over
/// distinct surviving left rows for the set variants). On a high-cardinality
/// input that map grows to `n_distinct × n_cols × ~24 B` of host RAM with no
/// upper limit — the same memory-DoS surface on user-controlled inputs that
/// the DISTINCT cap closes. The cap converts unbounded growth into a clean
/// [`BoltError::Other`] long before the OOM killer gets involved.
///
/// Overridable at runtime via [`SETOP_HOST_MAX_ROWS_ENV`] (parsed once on
/// first call; see [`setop_host_max_rows`]). The default matches
/// `DISTINCT_HOST_MAX_ROWS` (10M) so the two host set-building paths share a
/// single resource budget.
const SETOP_HOST_MAX_ROWS: usize = 10_000_000;

/// Environment variable that overrides [`SETOP_HOST_MAX_ROWS`] at runtime.
/// Parsed as a base-10 `usize`; `0` is rejected (it would disable the cap and
/// reintroduce the unbounded-growth bug). On any parse failure a `log::warn!`
/// is emitted and the default is used. Mirrors
/// `CRATON_DISTINCT_HOST_MAX_ROWS`.
const SETOP_HOST_MAX_ROWS_ENV: &str = "CRATON_SETOP_HOST_MAX_ROWS";

/// Latch for the per-process set-op host-row cap. First call resolves the env
/// var; subsequent calls hit the cached `usize`. Mirrors the DISTINCT cap's
/// `OnceLock` latch in `distinct.rs`.
static SETOP_HOST_MAX_ROWS_CACHE: OnceLock<usize> = OnceLock::new();

/// Resolve the per-process set-op host-row cap. First call performs the
/// env-var lookup; subsequent calls hit the latch. On any parse failure a
/// one-time `log::warn!` is emitted and the compile-time default
/// [`SETOP_HOST_MAX_ROWS`] is used.
fn setop_host_max_rows() -> usize {
    *SETOP_HOST_MAX_ROWS_CACHE.get_or_init(parse_setop_host_max_rows_env)
}

/// Pure parser for [`SETOP_HOST_MAX_ROWS_ENV`]. Extracted from the
/// `OnceLock` so callers (and tests) can exercise the parsing rules without
/// touching the latch. Returns the compile-time default on unset / empty /
/// unparseable / zero values, logging a warning in the unparseable / zero
/// cases. Mirrors `distinct::parse_distinct_host_max_rows_env`.
fn parse_setop_host_max_rows_env() -> usize {
    let raw = match std::env::var(SETOP_HOST_MAX_ROWS_ENV) {
        Ok(v) => v,
        Err(_) => return SETOP_HOST_MAX_ROWS,
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return SETOP_HOST_MAX_ROWS;
    }
    match trimmed.parse::<usize>() {
        Ok(0) => {
            log::warn!(
                "setops: {SETOP_HOST_MAX_ROWS_ENV}='0' would disable the host-side cap; \
                 using default of {SETOP_HOST_MAX_ROWS}"
            );
            SETOP_HOST_MAX_ROWS
        }
        Ok(v) => v,
        Err(e) => {
            log::warn!(
                "setops: {SETOP_HOST_MAX_ROWS_ENV}='{trimmed}' is not a valid usize ({e}); \
                 using default of {SETOP_HOST_MAX_ROWS}"
            );
            SETOP_HOST_MAX_ROWS
        }
    }
}

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
    let max_rows = setop_host_max_rows();
    execute_setop_with_cap(left, right, op, all, max_rows)
}

/// Internal entry point that lets callers (and tests) inject the host key-map
/// cap directly, bypassing the `OnceLock`-latched env-var resolution.
/// Production code goes through [`execute_setop`]; tests use this to exercise
/// the bound-exceeded path without poisoning the global latch for other tests
/// running in the same process. Mirrors `distinct::execute_distinct_with_cap`.
fn execute_setop_with_cap(
    left: QueryHandle,
    right: QueryHandle,
    op: SetOpKind,
    all: bool,
    max_rows: usize,
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
    //
    // `mut` so the multiset (`ALL`) variants can decrement copies IN PLACE
    // instead of cloning the whole map into a separate `remaining` working
    // copy (the old shape paid a full `HashMap` clone on every call, even
    // for the set variants that never decrement). `all` is constant for the
    // whole call, so the `ALL` (decrementing) and set (`get`-only) arms
    // never interleave: in the set arms the counts are read but never
    // mutated, so consuming `right_counts` as the working multiset for the
    // `ALL` arms is safe and preserves EXACT multiset semantics.
    let mut right_counts = build_key_counts(&right_batch, max_rows)?;

    // Pre-downcast the left columns once (mirrors the DISTINCT executor's
    // `ColumnReader` allocation strategy: N·K vtable lookups → K).
    let left_readers: Vec<ColumnReader<'_>> = left_batch
        .columns()
        .iter()
        .map(|c| ColumnReader::new(c.as_ref()))
        .collect::<BoltResult<Vec<_>>>()?;
    let n_cols = left_readers.len();

    // Decide, per left row, whether it survives into the output. The decision
    // depends on the operator + the `ALL` flag; see the module docs.
    let mut keep: Vec<bool> = Vec::with_capacity(n_left);
    // For the set (`!all`) variants we must dedupe the left side ourselves so
    // each surviving key appears at most once.
    let mut emitted: HashMap<RowKey, ()> = HashMap::new();

    for row in 0..n_left {
        let key = RowKey::from_values(n_cols, left_readers.iter().map(|r| r.value_at(row)));
        let survive = match (op, all) {
            // EXCEPT ALL: keep up to max(0, lc - rc) copies — drop a left copy
            // for each remaining right copy, keep the rest. Decrements the
            // right multiset in place (no per-call map clone).
            (SetOpKind::Except, true) => match right_counts.get_mut(&key) {
                Some(c) if *c > 0 => {
                    *c -= 1;
                    false
                }
                _ => true,
            },
            // INTERSECT ALL: keep min(lc, rc) copies — keep a left copy while
            // right copies remain, drop the rest. Decrements in place.
            (SetOpKind::Intersect, true) => match right_counts.get_mut(&key) {
                Some(c) if *c > 0 => {
                    *c -= 1;
                    true
                }
                _ => false,
            },
            // EXCEPT (set): distinct left rows whose key is absent from right.
            // The set arms only READ `right_counts` (never decrement), so the
            // multiset is intact for every row.
            (SetOpKind::Except, false) => {
                let in_right = right_counts.get(&key).copied().unwrap_or(0) > 0;
                !in_right && emitted.insert(key, ()).is_none()
            }
            // INTERSECT (set): distinct left rows whose key is present in right.
            (SetOpKind::Intersect, false) => {
                let in_right = right_counts.get(&key).copied().unwrap_or(0) > 0;
                in_right && emitted.insert(key, ()).is_none()
            }
        };
        keep.push(survive);
        // Resource bound on the left-side dedup map, mirroring the DISTINCT
        // cap (and the right-side cap in `build_key_counts`). Only the set
        // (`!all`) variants populate `emitted`; the `ALL` arms leave it empty
        // so this check is a no-op for them. Checking `emitted.len()` (the
        // distinct surviving-key count) rather than the row count lets a long
        // left input full of duplicates still complete.
        if emitted.len() > max_rows {
            return Err(BoltError::Other(format!(
                "{} exceeded host bound of {max_rows} distinct rows; \
                 LIMIT the input (override via {SETOP_HOST_MAX_ROWS_ENV})",
                op.keyword(),
            )));
        }
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
///
/// `max_rows` bounds the number of *distinct* keys the map may hold (the
/// resource cap mirroring DISTINCT — see [`SETOP_HOST_MAX_ROWS`]). The
/// initial capacity is clamped to `min(n_rows, max_rows)` so a giant `n_rows`
/// can't drive a multi-GiB up-front allocation, and the per-key check fires
/// when the *distinct* count crosses the cap (a right input full of
/// duplicates still completes). Exceeding the cap returns a clean
/// [`BoltError::Other`] rather than growing unboundedly.
fn build_key_counts(batch: &RecordBatch, max_rows: usize) -> BoltResult<HashMap<RowKey, usize>> {
    let n_rows = batch.num_rows();
    // Clamp the up-front allocation to the cap (DISTINCT's `initial_cap`
    // pattern): a huge `n_rows` must not pre-allocate past the budget.
    let initial_cap = n_rows.min(max_rows);
    let mut counts: HashMap<RowKey, usize> = HashMap::with_capacity(initial_cap);
    if n_rows == 0 {
        return Ok(counts);
    }
    let readers: Vec<ColumnReader<'_>> = batch
        .columns()
        .iter()
        .map(|c| ColumnReader::new(c.as_ref()))
        .collect::<BoltResult<Vec<_>>>()?;
    let n_cols = readers.len();
    for row in 0..n_rows {
        let key = RowKey::from_values(n_cols, readers.iter().map(|r| r.value_at(row)));
        *counts.entry(key).or_insert(0) += 1;
        // Resource bound: keep the multiset from growing without limit on a
        // high-cardinality right input. Checked on the distinct-key count
        // (`counts.len()`), not the row count, so a duplicate-heavy input
        // still completes; only the distinct cardinality is bounded.
        if counts.len() > max_rows {
            return Err(BoltError::Other(format!(
                "set operation exceeded host bound of {max_rows} distinct rows on the \
                 right input; LIMIT the input (override via {SETOP_HOST_MAX_ROWS_ENV})"
            )));
        }
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
            .map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i))
                }
            })
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
        let right =
            QueryHandle::from_record_batch(RecordBatch::try_new(schema, vec![r_s, r_n]).unwrap());
        // EXCEPT (set): distinct left {(a,1),(b,2)} minus {(a,1)} = {(b,2)}.
        let out = execute_setop(left, right, SetOpKind::Except, false).unwrap();
        assert_eq!(out.num_rows(), 1);
        let batch = out.record_batch();
        let s = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(s.value(0), "b");
    }

    // ─── Host key-map cap (memory-DoS guard, mirrors DISTINCT) ───────────

    /// A right input whose distinct cardinality exceeds the cap surfaces a
    /// clean error from `build_key_counts` rather than growing unboundedly.
    #[test]
    fn right_input_exceeding_cap_errors() {
        // 5 distinct right keys, cap of 3 → must error while building the
        // right multiset.
        let l = int32_handle(vec![Some(1)]);
        let r = int32_handle(vec![Some(10), Some(11), Some(12), Some(13), Some(14)]);
        let err = execute_setop_with_cap(l, r, SetOpKind::Intersect, false, 3)
            .expect_err("right-side cardinality over cap must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("exceeded host bound") && msg.contains("right input"),
            "expected right-side bound error, got: {msg}"
        );
    }

    /// The set (`!all`) left-side dedup map is also capped: a left input with
    /// many distinct surviving keys errors even when the right side is small.
    #[test]
    fn left_distinct_survivors_exceeding_cap_errors() {
        // Right is empty → EXCEPT (set) = DISTINCT(left). 5 distinct left
        // keys with cap 3 must trip the `emitted` cap.
        let l = int32_handle(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        let r = int32_handle(vec![]);
        let err = execute_setop_with_cap(l, r, SetOpKind::Except, false, 3)
            .expect_err("left distinct survivors over cap must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("exceeded host bound"),
            "expected host-bound error, got: {msg}"
        );
    }

    /// A duplicate-heavy input under the *distinct* cap still completes: the
    /// cap is on cardinality, not row count.
    #[test]
    fn duplicates_under_cap_complete() {
        // 100 rows but only 1 distinct key on each side; cap of 2 is fine.
        let l = int32_handle(vec![Some(7); 100]);
        let r = int32_handle(vec![Some(7); 100]);
        let out = execute_setop_with_cap(l, r, SetOpKind::Intersect, false, 2)
            .expect("duplicate-heavy input under the distinct cap must pass");
        assert_eq!(col_to_vec(&out), vec![Some(7)]);
    }

    /// Env-var parser: unset / empty / unparseable / zero all fall back to
    /// the compile-time default; a valid positive integer wins. Exercised
    /// against the pure parser so the `OnceLock` latch is not poisoned.
    /// Serialised on a local lock — `std::env` is process-global.
    #[test]
    fn setop_env_var_parser_handles_all_paths() {
        use std::sync::Mutex;
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap();

        let set = |v: Option<&str>| match v {
            Some(s) => std::env::set_var(SETOP_HOST_MAX_ROWS_ENV, s),
            None => std::env::remove_var(SETOP_HOST_MAX_ROWS_ENV),
        };

        set(None);
        assert_eq!(parse_setop_host_max_rows_env(), SETOP_HOST_MAX_ROWS);
        set(Some(""));
        assert_eq!(parse_setop_host_max_rows_env(), SETOP_HOST_MAX_ROWS);
        set(Some("0"));
        assert_eq!(parse_setop_host_max_rows_env(), SETOP_HOST_MAX_ROWS);
        set(Some("not-a-number"));
        assert_eq!(parse_setop_host_max_rows_env(), SETOP_HOST_MAX_ROWS);
        set(Some("42"));
        assert_eq!(parse_setop_host_max_rows_env(), 42);

        set(None);
    }
}
