// SPDX-License-Identifier: Apache-2.0
//! Multi-statement transactions: `BEGIN` / `COMMIT` / `ROLLBACK`.
//!
//! Between `BEGIN` and `COMMIT` the statements of [`Engine::execute_sql`]
//! change tables as usual and later statements (and queries) see those
//! changes, but a catalog publishes nothing: `COMMIT` makes every change
//! durable as **one** snapshot, `ROLLBACK` discards them — in the catalog and
//! in memory alike (in-memory tables, materialized views and table
//! registrations go back to their state at `BEGIN`).
//!
//! A statement that fails inside a transaction **aborts** it: later
//! statements fail until `ROLLBACK` (or `COMMIT`, which then rolls back and
//! reports it). Through a [`Database`](crate::Database), a transaction belongs
//! to the session that began it; other sessions' statements wait until it
//! ends (serializable, see the [`session`](crate::exec::session) docs).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::RecordBatch;

use crate::error::{BoltError, BoltResult};
use crate::exec::engine::Engine;
use crate::storage::StorageTable;

/// State captured at `BEGIN`.
pub(crate) struct Transaction {
    /// Session that began it (`None`: the engine's own caller).
    pub(crate) session: Option<u64>,
    /// A statement failed: only `ROLLBACK` / `COMMIT` are accepted.
    pub(crate) aborted: bool,
    tables: HashMap<String, Vec<RecordBatch>>,
    /// Registered storage tables that are not catalog tables.
    storage: Vec<(String, Arc<dyn StorageTable>)>,
    table_versions: HashMap<String, u64>,
    views: HashMap<String, crate::exec::engine_views::ViewDef>,
}

impl Engine {
    /// `true` while a transaction is open.
    pub fn in_transaction(&self) -> bool {
        self.txn.is_some()
    }

    /// Session holding the open transaction (`Some(None)` when it was begun
    /// outside a session).
    pub(crate) fn transaction_holder(&self) -> Option<Option<u64>> {
        self.txn.as_ref().map(|t| t.session)
    }

    /// Names of catalog tables (current manifest).
    fn catalog_table_names(&self) -> Vec<String> {
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_ref() {
            return cat.manifest().tables.keys().cloned().collect();
        }
        Vec::new()
    }

    /// `BEGIN`.
    pub fn begin_transaction(&mut self) -> BoltResult<()> {
        if self.txn.is_some() {
            return Err(BoltError::Plan("a transaction is already open".into()));
        }
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_mut() {
            if !cat.is_read_only() {
                cat.begin()?;
            }
        }
        let catalog_tables = self.catalog_table_names();
        let storage = self
            .storage_tables
            .borrow()
            .iter()
            .filter(|(n, _)| !catalog_tables.contains(n))
            .map(|(n, e)| (n.clone(), Arc::clone(&e.table)))
            .collect();
        self.txn = Some(Transaction {
            session: self.current_session,
            aborted: false,
            tables: self.tables.clone(),
            storage,
            table_versions: self.table_versions.borrow().clone(),
            views: self.views.borrow().clone(),
        });
        Ok(())
    }

    /// `COMMIT`: publish the transaction's changes. An aborted transaction
    /// is rolled back instead, and the error says so.
    pub fn commit_transaction(&mut self) -> BoltResult<()> {
        let Some(txn) = self.txn.as_ref() else {
            return Err(BoltError::Plan("no transaction is open".into()));
        };
        if txn.aborted {
            self.rollback_transaction()?;
            return Err(BoltError::Plan(
                "the transaction was aborted by an earlier error and has been rolled back".into(),
            ));
        }
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_mut() {
            if cat.in_transaction() {
                if let Err(e) = cat.commit_transaction() {
                    // Nothing was published; put the engine back in step
                    // with the catalog's last snapshot.
                    self.restore_snapshot(false)?;
                    return Err(e);
                }
            }
        }
        self.txn = None;
        Ok(())
    }

    /// `ROLLBACK`: discard the transaction's changes.
    pub fn rollback_transaction(&mut self) -> BoltResult<()> {
        if self.txn.is_none() {
            return Err(BoltError::Plan("no transaction is open".into()));
        }
        self.restore_snapshot(true)
    }

    /// Return every table and view to its state at `BEGIN` and close the
    /// transaction. `catalog_open`: the catalog still has the transaction
    /// open (and must roll it back).
    fn restore_snapshot(&mut self, catalog_open: bool) -> BoltResult<()> {
        let txn = self.txn.take().expect("transaction open");
        let mut catalog_tables = self.catalog_table_names();
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_mut() {
            if catalog_open && cat.in_transaction() {
                cat.rollback_transaction()?;
            }
        }
        let _ = catalog_open;
        // Catalog tables: re-open what the restored manifest holds, drop
        // what it does not.
        let restored = self.catalog_table_names();
        catalog_tables.extend(restored.iter().cloned());
        catalog_tables.sort();
        catalog_tables.dedup();
        for name in &catalog_tables {
            if restored.contains(name) {
                #[cfg(feature = "storage")]
                self.register_catalog_table(name)?;
            } else if self.has_table(name) {
                self.deregister_table(name)?;
            }
        }
        // In-memory tables.
        let current: Vec<String> = self.tables.keys().cloned().collect();
        for name in current {
            if !txn.tables.contains_key(&name) {
                self.deregister_table(&name)?;
            }
        }
        for (name, batches) in &txn.tables {
            let unchanged = self.tables.contains_key(name)
                && self.table_versions.borrow().get(name) == txn.table_versions.get(name);
            if unchanged {
                continue;
            }
            let Some(first) = batches.first() else {
                continue;
            };
            let rows = if batches.len() == 1 {
                first.clone()
            } else {
                arrow::compute::concat_batches(&first.schema(), batches)
                    .map_err(|e| BoltError::Other(format!("restoring '{name}': {e}")))?
            };
            if self.has_table(name) {
                self.deregister_table(name)?;
            }
            self.register_table(name.clone(), rows)?;
        }
        // Registered storage tables (non-catalog) and lazy streams added
        // inside the transaction are removed; dropped ones come back.
        let registered: Vec<String> = self
            .storage_tables
            .borrow()
            .keys()
            .filter(|n| !restored.contains(n))
            .cloned()
            .collect();
        for name in registered {
            if !txn.storage.iter().any(|(n, _)| *n == name) {
                self.deregister_table(&name)?;
            }
        }
        for (name, table) in &txn.storage {
            if !self.has_table(name) {
                self.register_storage_table(name, Arc::clone(table))?;
            }
        }
        *self.views.borrow_mut() = txn.views;
        *self.table_versions.borrow_mut() = txn.table_versions;
        self.bump_data_epoch();
        Ok(())
    }

    /// Error if the open transaction is aborted.
    pub(crate) fn ensure_transaction_usable(&self) -> BoltResult<()> {
        if self.txn.as_ref().is_some_and(|t| t.aborted) {
            return Err(BoltError::Plan(
                "the current transaction is aborted; ROLLBACK to end it".into(),
            ));
        }
        Ok(())
    }

    /// Record that a statement failed inside the open transaction.
    pub(crate) fn abort_transaction_on_error(&mut self) {
        if let Some(t) = self.txn.as_mut() {
            t.aborted = true;
        }
    }

    /// `DROP` of a lazy stream cannot be undone: refuse it inside a
    /// transaction.
    pub(crate) fn ensure_droppable_in_transaction(&self, name: &str) -> BoltResult<()> {
        if self.txn.is_some()
            && !self.tables.contains_key(name)
            && !self.storage_tables.borrow().contains_key(name)
            && self.streaming_sources.borrow().contains_key(name)
        {
            return Err(BoltError::Plan(format!(
                "lazy stream table '{name}' cannot be dropped inside a transaction"
            )));
        }
        Ok(())
    }
}
