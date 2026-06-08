// SPDX-License-Identifier: Apache-2.0

//! Planner-facing adapters over the engine's registered tables, lifted out
//! of `exec::engine` (pure reorganization; no behavior change).
//!
//! [`EngineProvider`] surfaces real per-column null counts to the planner;
//! [`EngineTableStats`] feeds base-table row counts to the cost-based
//! join-reorder pass.

use std::cell::Ref;
use std::collections::HashMap;

use arrow_array::RecordBatch;

use crate::error::BoltResult;
use crate::plan::{MemTableProvider, Schema};

// ---------------------------------------------------------------------------
// PV-stage-d: TableProvider adaptor that surfaces actual per-column null
// counts from the engine's registered `RecordBatch`es.
//
// `MemTableProvider` only knows the schema (column names + dtypes); the
// engine additionally holds the data, so the per-column `null_count()` is
// cheap to read here. We wrap the schema provider so the planner gets:
//   * Schema lookups via the underlying `MemTableProvider` (same as before).
//   * `has_nulls` answered by scanning the registered batches' bitmaps.
// ---------------------------------------------------------------------------

/// `TableProvider` adapter wrapping the engine's [`MemTableProvider`] schema
/// store and adding `has_nulls` / `null_count` answers backed by the actual
/// registered `RecordBatch`es.
pub(crate) struct EngineProvider<'a> {
    pub(crate) base: &'a MemTableProvider,
    pub(crate) tables: &'a HashMap<String, Vec<RecordBatch>>,
    /// Borrow of the lazy streaming overlay. By the time an `EngineProvider`
    /// is built, any streaming source referenced by the plan has already been
    /// collapsed to [`TableSource::Materialized`] (see
    /// [`Engine::ensure_streaming_materialized`]), so the null probes below
    /// can read its batches the same way they read `tables`.
    pub(crate) streaming: Ref<'a, HashMap<String, crate::exec::streaming::TableSource>>,
}

impl<'a> EngineProvider<'a> {
    /// Resolve a table name to its host-side batches, consulting `tables`
    /// first and the (already-materialised) streaming overlay second.
    fn batches_for(&self, table_name: &str) -> Option<&[RecordBatch]> {
        if let Some(b) = self.tables.get(table_name) {
            return Some(b.as_slice());
        }
        match self.streaming.get(table_name) {
            Some(crate::exec::streaming::TableSource::Materialized(b)) => Some(b.as_slice()),
            // A still-streaming source means the caller forgot to materialise
            // it before building the provider; treat as absent (safe-false).
            _ => None,
        }
    }
}

impl<'a> crate::plan::TableProvider for EngineProvider<'a> {
    fn schema(&self, name: &str) -> BoltResult<Schema> {
        self.base.schema(name)
    }

    fn has_nulls(&self, table_name: &str, col_idx: usize) -> bool {
        // PV-stage-f: returns `true` iff ANY registered batch for `table_name`
        // has at least one NULL on column ordinal `col_idx` (via
        // `RecordBatch::column(col_idx).null_count() > 0`). This is the
        // plan-time signal `populate_input_validity` /
        // `populate_aggregate_spec` (in `crate::plan::physical_plan`) read
        // to fill `KernelSpec::input_has_validity` and
        // `AggregateSpec::input_has_validity` respectively.
        //
        // Safe-false on any miss — the executor's host-strip fallback still
        // handles the row filtering, so an under-flag is correctness-safe.
        let batches = match self.batches_for(table_name) {
            Some(b) => b,
            None => return false,
        };
        for batch in batches {
            // Skip out-of-range column ordinals (e.g. dictionary-extended
            // `__idx_<col>` columns the dict registry mints; those have
            // their own null behaviour).
            if col_idx >= batch.num_columns() {
                continue;
            }
            if batch.column(col_idx).null_count() > 0 {
                return true;
            }
        }
        false
    }

    fn null_count(&self, table_name: &str, col_idx: usize) -> Option<usize> {
        let batches = self.batches_for(table_name)?;
        let mut total: usize = 0;
        for batch in batches {
            if col_idx >= batch.num_columns() {
                continue;
            }
            total = total.saturating_add(batch.column(col_idx).null_count());
        }
        Some(total)
    }
}

// ---------------------------------------------------------------------------
// Cost-based optimizer wiring: a `StatsProvider` backed by the engine's
// registered tables.
//
// The join-reorder pass (`crate::plan::optimizer::join_reorder`) is a no-op
// until it is handed a row estimator. We feed it one built from real data:
// each base table's `row_count` is the sum of its registered `RecordBatch`
// row counts. The engine takes a cheap *snapshot* of those counts at plan
// time (a `HashMap<String, usize>`), wraps it in `StatsEstimator`, and threads
// it into the default pass pipeline via `default_passes_with_estimator`.
// ---------------------------------------------------------------------------

/// Owned snapshot of base-table row counts used to drive cost-based join
/// reordering.
///
/// Built once per query in [`Engine::table_stats_snapshot`] from the engine's
/// registered tables. Owning the counts (rather than borrowing the engine)
/// makes this `Send + Sync + 'static`, so it can live behind the
/// `Arc<dyn RowEstimator>` the [`crate::plan::optimizer::JoinReorder`] pass
/// holds. A table absent from the snapshot has no entry and the estimator
/// returns `None` for it — which keeps reordering conservative (the pass only
/// fires when *every* leaf in a chain is costed).
#[derive(Debug, Default)]
pub(crate) struct EngineTableStats {
    /// Table name → total registered row count.
    pub(crate) row_counts: HashMap<String, usize>,
}

impl crate::plan::statistics::StatsProvider for EngineTableStats {
    fn table_stats(&self, name: &str) -> Option<crate::plan::statistics::TableStats> {
        self.row_counts
            .get(name)
            .map(|&n| crate::plan::statistics::TableStats::new(n))
    }
}

#[cfg(test)]
mod tests {
    //! Host-only tests for the planner-facing null probes. `EngineProvider`
    //! is pure `RecordBatch` host logic — no CUDA — so we build batches and
    //! the backing maps directly and drive `has_nulls` / `null_count`
    //! through the `TableProvider` trait.

    use super::*;
    use std::cell::RefCell;
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

    use crate::exec::streaming::TableSource;
    use crate::plan::{MemTableProvider, TableProvider};

    /// One-column Int32 `RecordBatch` from optional values (None => NULL row).
    fn int32_batch(name: &str, values: Vec<Option<i32>>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            name,
            ArrowDataType::Int32,
            true,
        )]));
        let arr = Arc::new(Int32Array::from(values)) as arrow_array::ArrayRef;
        RecordBatch::try_new(schema, vec![arr]).expect("batch")
    }

    /// Two-column Int32 batch so we can probe a second / out-of-range ordinal.
    fn two_col_batch(a: Vec<Option<i32>>, b: Vec<Option<i32>>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", ArrowDataType::Int32, true),
            ArrowField::new("b", ArrowDataType::Int32, true),
        ]));
        let ca = Arc::new(Int32Array::from(a)) as arrow_array::ArrayRef;
        let cb = Arc::new(Int32Array::from(b)) as arrow_array::ArrayRef;
        RecordBatch::try_new(schema, vec![ca, cb]).expect("batch")
    }

    /// Build an `EngineProvider` over the given `tables` map and an empty
    /// streaming overlay, run `f`, and return its result. The `RefCell` and
    /// `MemTableProvider` are owned by this scope so the `Ref` borrow the
    /// provider holds stays alive for the whole closure.
    fn with_provider<R>(
        tables: &HashMap<String, Vec<RecordBatch>>,
        streaming: &RefCell<HashMap<String, TableSource>>,
        f: impl FnOnce(&EngineProvider) -> R,
    ) -> R {
        let base = MemTableProvider::new();
        let provider = EngineProvider {
            base: &base,
            tables,
            streaming: streaming.borrow(),
        };
        f(&provider)
    }

    /// `has_nulls` returns true iff ANY batch has a NULL on the column —
    /// tested across a MULTI-batch table where only the second batch has one.
    #[test]
    fn has_nulls_scans_all_batches() {
        let mut tables = HashMap::new();
        tables.insert(
            "t".to_string(),
            vec![
                int32_batch("a", vec![Some(1), Some(2)]), // no nulls
                int32_batch("a", vec![Some(3), None]),    // null in batch 2
            ],
        );
        let streaming = RefCell::new(HashMap::new());
        with_provider(&tables, &streaming, |p| {
            assert!(p.has_nulls("t", 0), "null in any batch => true");
        });
    }

    /// A fully non-null multi-batch table reports no nulls.
    #[test]
    fn has_nulls_false_when_all_batches_clean() {
        let mut tables = HashMap::new();
        tables.insert(
            "t".to_string(),
            vec![
                int32_batch("a", vec![Some(1)]),
                int32_batch("a", vec![Some(2), Some(3)]),
            ],
        );
        let streaming = RefCell::new(HashMap::new());
        with_provider(&tables, &streaming, |p| {
            assert!(!p.has_nulls("t", 0));
        });
    }

    /// `null_count` sums across every batch (saturating add); skipping a
    /// per-batch out-of-range ordinal rather than counting it.
    #[test]
    fn null_count_sums_across_batches() {
        let mut tables = HashMap::new();
        tables.insert(
            "t".to_string(),
            vec![
                int32_batch("a", vec![Some(1), None]),       // 1 null
                int32_batch("a", vec![None, None, Some(4)]), // 2 nulls
            ],
        );
        let streaming = RefCell::new(HashMap::new());
        with_provider(&tables, &streaming, |p| {
            assert_eq!(p.null_count("t", 0), Some(3));
        });
    }

    /// An out-of-range `col_idx` is skipped per batch (e.g. a dict-extended
    /// `__idx_*` ordinal): `has_nulls` is safe-false and `null_count` counts
    /// nothing for the missing ordinal.
    #[test]
    fn out_of_range_col_idx_is_skipped() {
        let mut tables = HashMap::new();
        // Column 0 has a null; column 5 doesn't exist (only 2 columns).
        tables.insert(
            "t".to_string(),
            vec![two_col_batch(vec![Some(1), None], vec![Some(2), Some(3)])],
        );
        let streaming = RefCell::new(HashMap::new());
        with_provider(&tables, &streaming, |p| {
            // In-range column 0 sees the null; out-of-range column 5 is skipped.
            assert!(p.has_nulls("t", 0));
            assert!(!p.has_nulls("t", 5), "out-of-range ordinal => safe-false");
            assert_eq!(p.null_count("t", 5), Some(0), "skipped ordinal counts zero");
            // Column 1 (b) has no nulls.
            assert!(!p.has_nulls("t", 1));
        });
    }

    /// An unknown table: `has_nulls` is safe-false, `null_count` is `None`.
    #[test]
    fn unknown_table_is_safe_false_and_none() {
        let tables = HashMap::new();
        let streaming = RefCell::new(HashMap::new());
        with_provider(&tables, &streaming, |p| {
            assert!(!p.has_nulls("missing", 0));
            assert_eq!(p.null_count("missing", 0), None);
        });
    }

    /// A table present ONLY in the (materialised) streaming overlay resolves
    /// through `batches_for`'s second arm and answers identically.
    #[test]
    fn materialized_streaming_overlay_is_probed() {
        let tables = HashMap::new();
        let mut sm = HashMap::new();
        sm.insert(
            "s".to_string(),
            TableSource::Materialized(vec![int32_batch("a", vec![Some(1), None])]),
        );
        let streaming = RefCell::new(sm);
        with_provider(&tables, &streaming, |p| {
            assert!(p.has_nulls("s", 0), "overlay batch null must be seen");
            assert_eq!(p.null_count("s", 0), Some(1));
        });
    }

    /// A still-`Streaming` overlay entry (not yet materialised) is treated as
    /// absent by `batches_for` — `has_nulls` is safe-false and `null_count`
    /// is `None`. (A real query materialises the source before building the
    /// provider; this guards the defensive "caller forgot" path.)
    #[test]
    fn still_streaming_overlay_is_treated_as_absent() {
        let tables = HashMap::new();
        let mut sm = HashMap::new();
        // A trivial replayable producer that yields no batches. Its mere
        // presence as `Streaming` (not `Materialized`) is what we test.
        let producer: crate::exec::streaming::BatchProducer =
            Box::new(|| Box::new(std::iter::empty()));
        sm.insert("live".to_string(), TableSource::Streaming(producer));
        let streaming = RefCell::new(sm);
        with_provider(&tables, &streaming, |p| {
            assert!(!p.has_nulls("live", 0));
            assert_eq!(p.null_count("live", 0), None);
        });
    }

    /// `tables` takes precedence over the streaming overlay for the same name
    /// (the `batches_for` lookup order). A clean `tables` entry wins over a
    /// null-bearing overlay entry of the same name.
    #[test]
    fn tables_shadow_streaming_overlay() {
        let mut tables = HashMap::new();
        tables.insert("t".to_string(), vec![int32_batch("a", vec![Some(1)])]);
        let mut sm = HashMap::new();
        sm.insert(
            "t".to_string(),
            TableSource::Materialized(vec![int32_batch("a", vec![None])]),
        );
        let streaming = RefCell::new(sm);
        with_provider(&tables, &streaming, |p| {
            assert!(
                !p.has_nulls("t", 0),
                "the non-null `tables` entry must shadow the null overlay entry"
            );
        });
    }

    /// `EngineTableStats` returns a `TableStats` for known tables and `None`
    /// for absent ones (keeping join-reorder conservative).
    #[test]
    fn engine_table_stats_lookup() {
        use crate::plan::statistics::StatsProvider;
        let mut stats = EngineTableStats::default();
        stats.row_counts.insert("t".to_string(), 100);
        assert!(stats.table_stats("t").is_some());
        assert!(stats.table_stats("absent").is_none());
    }
}
