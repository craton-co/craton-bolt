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
