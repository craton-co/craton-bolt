// SPDX-License-Identifier: Apache-2.0
//! Engine glue for [`StorageTable`]s: registration, per-query pruned scans,
//! and table statistics.
//!
//! A storage table is scanned when a statement that reads it is prepared
//! (before compilation, like a lazy stream's drain): the statement's
//! optimized plan yields a [`ScanRequest`] per table, the table turns it into
//! a [`ScanSelection`] with its zone maps, and the selected chunks / columns
//! are loaded into the streaming overlay, where every executor reads them.
//! The plan keeps all its filters, so installing *more* data than a request
//! asked for is always correct; that is what lets a later statement reuse an
//! installed scan that covers its own request (no re-read, GPU residency
//! kept).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::RecordBatch;

use crate::error::{BoltError, BoltResult};
use crate::exec::engine::Engine;
use crate::plan::logical_plan::LogicalPlan;
use crate::storage::{ScanRequest, ScanSelection, StorageTable, TableStatistics};

/// A registered storage table and the scan currently installed for it.
pub(crate) struct StorageEntry {
    pub(crate) table: Arc<dyn StorageTable>,
    installed: Option<ScanSelection>,
    last_scan: Option<ScanReport>,
}

/// What the most recent scan of a storage table read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ScanReport {
    /// Chunks (row groups / batches) in the table.
    pub chunks_total: usize,
    /// Chunks read.
    pub chunks_read: usize,
    /// Rows read.
    pub rows_read: u64,
    /// Columns decoded (`None` = all).
    pub columns_read: Option<usize>,
    /// `true` when the statement reused an already-installed scan.
    pub reused: bool,
}

impl Engine {
    /// Register a [`StorageTable`] under `name`.
    ///
    /// Nothing is read at registration. Each statement that scans the table
    /// loads only the chunks whose statistics may match its filters and only
    /// the columns it reads (see the [`crate::storage`] module docs); a later
    /// statement whose needs are covered by the loaded data reuses it.
    ///
    /// # Errors
    /// A table named `name` is already registered.
    pub fn register_storage_table(
        &mut self,
        name: impl Into<String>,
        table: Arc<dyn StorageTable>,
    ) -> BoltResult<()> {
        let name = name.into();
        self.ensure_name_free(&name)?;
        self.touch_table(&name);
        self.provider.register(name.clone(), table.schema().clone());
        self.storage_tables.borrow_mut().insert(
            name,
            StorageEntry {
                table,
                installed: None,
                last_scan: None,
            },
        );
        Ok(())
    }

    /// Register the Parquet file — or directory of `*.parquet` files — at
    /// `path` as table `name`. Shorthand for
    /// [`register_storage_table`](Engine::register_storage_table) with a
    /// [`ParquetTable`](crate::storage::ParquetTable).
    #[cfg(feature = "storage")]
    pub fn register_parquet(
        &mut self,
        name: impl Into<String>,
        path: impl AsRef<std::path::Path>,
    ) -> BoltResult<()> {
        let table = crate::storage::ParquetTable::open(path)?;
        self.register_storage_table(name, Arc::new(table))
    }

    /// Attach a persistent catalog: register its tables and load its
    /// materialized views.
    #[cfg(feature = "storage")]
    pub(crate) fn attach_catalog(&mut self, catalog: crate::storage::Catalog) -> BoltResult<()> {
        let names: Vec<String> = catalog.manifest().tables.keys().cloned().collect();
        self.catalog = Some(catalog);
        for name in names {
            self.register_catalog_table(&name)?;
        }
        self.load_catalog_views();
        Ok(())
    }

    /// (Re-)register catalog table `name` from the catalog's current files.
    #[cfg(feature = "storage")]
    pub(crate) fn register_catalog_table(&mut self, name: &str) -> BoltResult<()> {
        let table = self
            .catalog
            .as_ref()
            .ok_or_else(|| BoltError::Plan("no catalog attached".into()))?
            .open_table(name)?;
        if self.has_table(name) {
            self.deregister_table(name)?;
        }
        self.register_storage_table(name, Arc::new(table))
    }

    /// The attached persistent catalog, if any.
    #[cfg(feature = "storage")]
    pub fn catalog(&self) -> Option<&crate::storage::Catalog> {
        self.catalog.as_ref()
    }

    /// The storage table registered as `name`, if any.
    pub fn storage_table(&self, name: &str) -> Option<Arc<dyn StorageTable>> {
        self.storage_tables
            .borrow()
            .get(name)
            .map(|e| Arc::clone(&e.table))
    }

    /// What the most recent statement that scanned storage table `name` read.
    pub fn last_scan(&self, name: &str) -> Option<ScanReport> {
        self.storage_tables
            .borrow()
            .get(name)
            .and_then(|e| e.last_scan)
    }

    /// Statistics of table `name`: row count, size, and per-column null
    /// counts, bounds and distinct estimates.
    ///
    /// A storage table answers from its chunk metadata without reading data
    /// (distinct estimates are then lower bounds, or unknown). For in-memory
    /// tables the statistics are computed over the data (exact counts and
    /// bounds, HyperLogLog distinct estimates); a lazily-registered stream is
    /// drained first.
    pub fn table_statistics(&self, name: &str) -> BoltResult<TableStatistics> {
        #[cfg(feature = "storage")]
        if let Some(s) = self
            .catalog
            .as_ref()
            .and_then(|c| c.table(name))
            .and_then(|t| t.analyzed.clone())
        {
            return Ok(s);
        }
        if let Some(t) = self.storage_table(name) {
            return Ok(t.statistics());
        }
        if let Some(batches) = self.tables.get(name) {
            return Ok(TableStatistics::from_batches(batches));
        }
        self.materialize_streaming_source(name)?;
        match self.streaming_sources.borrow().get(name) {
            Some(crate::exec::streaming::TableSource::Materialized(b)) => {
                Ok(TableStatistics::from_batches(b))
            }
            _ => Err(BoltError::Plan(format!("table '{name}' is not registered"))),
        }
    }

    /// Every registered table name (in-memory, lazy and storage), sorted.
    pub fn table_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tables.keys().cloned().collect();
        names.extend(self.streaming_sources.borrow().keys().cloned());
        names.extend(self.storage_tables.borrow().keys().cloned());
        names.sort();
        names.dedup();
        names
    }

    /// `true` when `name` is taken by any kind of table (in-memory, lazy or
    /// storage).
    pub(crate) fn has_table(&self, name: &str) -> bool {
        self.tables.contains_key(name)
            || self.streaming_sources.borrow().contains_key(name)
            || self.storage_tables.borrow().contains_key(name)
    }

    /// `Plan` error when `name` is taken by any kind of table.
    pub(crate) fn ensure_name_free(&self, name: &str) -> BoltResult<()> {
        if self.has_table(name) {
            return Err(BoltError::Plan(format!(
                "table '{name}' is already registered"
            )));
        }
        Ok(())
    }

    /// Row counts of the storage tables, for the optimizer's estimator.
    pub(crate) fn storage_row_counts(&self) -> Vec<(String, usize)> {
        self.storage_tables
            .borrow()
            .iter()
            .map(|(n, e)| (n.clone(), e.table.statistics().row_count as usize))
            .collect()
    }

    /// Load (or reuse) the scans of every storage table `plan` reads.
    ///
    /// `tree_tables` / `nested_tables` are the tables scanned by the plan's
    /// relational tree and by its subqueries. A table read inside a subquery
    /// is scanned whole: the subquery's own filters are compiled separately
    /// and must find every row. At top level the installed scan is replaced by
    /// exactly what this statement needs (unless it already covers it); a
    /// nested statement (subquery, component of a host-orchestrated shape)
    /// only ever widens the installed scan, so it can never take rows away
    /// from the statement that encloses it.
    pub(crate) fn scan_storage_tables(
        &self,
        plan: &LogicalPlan,
        tree_tables: &[String],
        nested_tables: &[String],
        top_level: bool,
    ) -> BoltResult<()> {
        let wanted: Vec<String> = {
            let st = self.storage_tables.borrow();
            let mut w: Vec<String> = tree_tables
                .iter()
                .chain(nested_tables)
                .filter(|t| st.contains_key(t.as_str()))
                .cloned()
                .collect();
            w.sort();
            w.dedup();
            w
        };
        if wanted.is_empty() {
            return Ok(());
        }
        let requests = self.storage_scan_requests(plan, &wanted, nested_tables);
        for name in wanted {
            let request = requests
                .get(&name)
                .cloned()
                .unwrap_or_else(ScanRequest::full);
            self.install_storage_scan(&name, &request, top_level)?;
        }
        Ok(())
    }

    /// Per-table requests from the statement's optimized plan. An optimizer
    /// failure falls back to full scans (the real compile reports it).
    fn storage_scan_requests(
        &self,
        plan: &LogicalPlan,
        tables: &[String],
        nested_tables: &[String],
    ) -> HashMap<String, ScanRequest> {
        let optimized = if self.optimize {
            let passes = crate::plan::default_passes_with_estimator(self.row_estimator());
            crate::plan::optimizer::run_to_fixpoint(&passes, plan.clone()).ok()
        } else {
            Some(plan.clone())
        };
        let Some(optimized) = optimized else {
            return HashMap::new();
        };
        let names: Vec<&str> = tables
            .iter()
            .filter(|t| !nested_tables.contains(t))
            .map(String::as_str)
            .collect();
        crate::storage::scan::scan_requests(&optimized, &names)
    }

    fn install_storage_scan(
        &self,
        name: &str,
        request: &ScanRequest,
        top_level: bool,
    ) -> BoltResult<()> {
        let (table, installed) = {
            let st = self.storage_tables.borrow();
            let e = &st[name];
            (Arc::clone(&e.table), e.installed.clone())
        };
        let needed = table.plan_scan(request);
        let selection = match &installed {
            Some(cur) if cur.covers(&needed) => {
                self.record_scan(name, cur, true);
                return Ok(());
            }
            Some(cur) if !top_level => cur.union(&needed),
            _ => needed,
        };
        let declared = table.schema().clone();
        let mut batches: Vec<RecordBatch> = Vec::new();
        for b in table.read(&selection)? {
            let b = b?;
            crate::exec::engine::validate_stream_batch_schema(
                "storage scan",
                &declared,
                &b,
                name,
                batches.len(),
            )?;
            if b.num_rows() > 0 {
                batches.push(b);
            }
        }
        let data = if batches.is_empty() {
            RecordBatch::new_empty(crate::exec::schema_convert::plan_schema_to_arrow_schema(
                &declared,
            )?)
        } else {
            crate::exec::engine_support::concat_table_batches(name, &batches)?
        };
        drop(batches);
        // Same dictionary policy as a lazy drain: whole-table dictionaries
        // only without a memory budget (morsels would not match them).
        // The registry is free here: the top-level pass installs every scan
        // (subquery tables whole) before compiling, so a nested statement
        // only ever reuses one. Refuse rather than leave stale dictionaries.
        let mut registry = self.dict_registry.try_borrow_mut().map_err(|_| {
            BoltError::Other(format!(
                "storage table '{name}': scan needed while the dictionary registry is in use"
            ))
        })?;
        registry.unregister_table(name);
        if self.memory_budget_bytes().is_none() {
            if let Err(e) = registry.register_table(name.to_string(), &data) {
                log::debug!("storage table '{name}': no string dictionaries ({e})");
            }
        }
        drop(registry);
        self.gpu_tables.borrow_mut().remove(name);
        self.streaming_sources.borrow_mut().insert(
            name.to_string(),
            crate::exec::streaming::TableSource::Materialized(vec![data]),
        );
        self.bump_data_epoch();
        let m = crate::metrics::metrics();
        m.add(
            crate::metrics::Counter::StorageChunksRead,
            selection.chunks.len() as u64,
        );
        m.add(
            crate::metrics::Counter::StorageChunksPruned,
            (selection.chunks_total - selection.chunks.len()) as u64,
        );
        self.record_scan(name, &selection, false);
        self.storage_tables
            .borrow_mut()
            .get_mut(name)
            .expect("storage entry")
            .installed = Some(selection);
        Ok(())
    }

    fn record_scan(&self, name: &str, s: &ScanSelection, reused: bool) {
        if let Some(e) = self.storage_tables.borrow_mut().get_mut(name) {
            e.last_scan = Some(ScanReport {
                chunks_total: s.chunks_total,
                chunks_read: s.chunks.len(),
                rows_read: s.rows,
                columns_read: s.columns.as_ref().map(|c| c.len()),
                reused,
            });
        }
    }
}
