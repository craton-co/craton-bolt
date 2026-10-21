// SPDX-License-Identifier: Apache-2.0
//! Multi-statement transactions: `BEGIN` / `COMMIT` / `ROLLBACK`,
//! `SAVEPOINT` / `ROLLBACK TO SAVEPOINT` / `RELEASE SAVEPOINT`.
//!
//! Between `BEGIN` and `COMMIT` the statements of [`Engine::execute_sql`]
//! change tables as usual and later statements (and queries) see those
//! changes, but a catalog publishes nothing: `COMMIT` makes every change
//! durable as **one** snapshot, `ROLLBACK` discards them — in the catalog and
//! in memory alike (in-memory tables, materialized views and table
//! registrations go back to their state at `BEGIN`). A savepoint captures the
//! same state inside the transaction; rolling back to it undoes only what
//! came after.
//!
//! A statement that fails inside a transaction **aborts** it: later
//! statements fail until `ROLLBACK` / `ROLLBACK TO SAVEPOINT` (or `COMMIT`,
//! which then rolls back and reports it).
//!
//! **Concurrency.** Through a [`Database`](crate::Database), each session
//! may hold its own transaction, and all of them run at once under
//! *snapshot isolation*: a transaction works on a private copy of the
//! table-and-view state taken at its `BEGIN` (the engine installs the
//! submitting session's state before each statement — see
//! [`Engine::activate_session`]), so it sees neither other transactions'
//! uncommitted changes nor commits made after it began. At `COMMIT` its
//! changes are merged into whatever has been committed since: the tables and
//! views it changed replace the committed ones, everything else keeps the
//! newer committed state. If another commit changed a table or view this
//! transaction also changed, the commit fails with a serialization error and
//! the transaction is rolled back (first committer wins). Write skew across
//! different tables is possible, as under any snapshot isolation;
//! [`IsolationLevel::Serializable`](crate::IsolationLevel) runs transactions
//! one at a time instead.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::RecordBatch;

use crate::error::{BoltError, BoltResult};
use crate::exec::engine::Engine;
use crate::exec::engine_views::ViewDef;
use crate::storage::StorageTable;

/// The table-and-view state statements see: the committed one, or an open
/// transaction's (at its `BEGIN`, at a savepoint, or its latest).
#[derive(Clone)]
pub(crate) struct Snapshot {
    tables: HashMap<String, Vec<RecordBatch>>,
    /// Registered storage tables that are not catalog tables.
    storage: Vec<(String, Arc<dyn StorageTable>)>,
    table_versions: HashMap<String, u64>,
    views: HashMap<String, ViewDef>,
    #[cfg(feature = "storage")]
    catalog: Option<crate::storage::catalog::CatalogState>,
}

/// An open transaction.
pub(crate) struct Transaction {
    /// Session that began it (`None`: the engine's own caller).
    pub(crate) session: Option<u64>,
    /// A statement failed: only `ROLLBACK` / `COMMIT` are accepted.
    pub(crate) aborted: bool,
    /// State at `BEGIN`: what the transaction changed is the difference
    /// between it and the transaction's latest state.
    base: Snapshot,
    /// Savepoints, oldest first.
    savepoints: Vec<(String, Snapshot)>,
}

/// What identifies a view's content for conflict detection.
fn view_key(
    v: &ViewDef,
) -> (
    &str,
    Option<std::time::SystemTime>,
    &std::collections::BTreeMap<String, u64>,
) {
    (v.sql.as_str(), v.refreshed_at, &v.base_versions)
}

fn conflict(what: &str, name: &str) -> BoltError {
    BoltError::Plan(format!(
        "could not serialize the transaction: {what} '{name}' was changed by a concurrent \
         transaction"
    ))
}

/// Merge the changes a transaction made (`base` → `work`) into `committed`.
/// Errors when something it changed was also changed by a later commit.
fn merge(base: &Snapshot, work: &Snapshot, committed: &Snapshot) -> BoltResult<Snapshot> {
    let mut next = committed.clone();
    // In-memory tables, identified by their data version.
    let version = |s: &Snapshot, n: &str| {
        s.tables
            .contains_key(n)
            .then(|| s.table_versions.get(n).copied())
    };
    let mut names: Vec<&String> = base.tables.keys().chain(work.tables.keys()).collect();
    names.sort();
    names.dedup();
    for n in names {
        let b = version(base, n);
        if b == version(work, n) {
            continue;
        }
        if version(committed, n) != b {
            return Err(conflict("table", n));
        }
        match work.tables.get(n) {
            Some(rows) => {
                next.tables.insert(n.clone(), rows.clone());
                if let Some(v) = work.table_versions.get(n) {
                    next.table_versions.insert(n.clone(), *v);
                }
            }
            None => {
                next.tables.remove(n);
                next.table_versions.remove(n);
            }
        }
    }
    // Versions of other tables (catalog tables carry their own).
    for (n, v) in &work.table_versions {
        if base.table_versions.get(n) != Some(v) && !work.tables.contains_key(n) {
            next.table_versions.insert(n.clone(), *v);
        }
    }
    // Registered storage tables, identified by the table object.
    let storage = |s: &Snapshot, n: &str| {
        s.storage
            .iter()
            .find(|(m, _)| m == n)
            .map(|(_, t)| Arc::as_ptr(t) as *const u8)
    };
    let mut names: Vec<&String> = base
        .storage
        .iter()
        .chain(&work.storage)
        .map(|(n, _)| n)
        .collect();
    names.sort();
    names.dedup();
    for n in names {
        let b = storage(base, n);
        if b == storage(work, n) {
            continue;
        }
        if storage(committed, n) != b {
            return Err(conflict("table", n));
        }
        next.storage.retain(|(m, _)| m != n);
        if let Some(entry) = work.storage.iter().find(|(m, _)| m == n) {
            next.storage.push(entry.clone());
        }
    }
    // Materialized views.
    let mut names: Vec<&String> = base.views.keys().chain(work.views.keys()).collect();
    names.sort();
    names.dedup();
    for n in names {
        let b = base.views.get(n).map(view_key);
        if b == work.views.get(n).map(view_key) {
            continue;
        }
        if committed.views.get(n).map(view_key) != b {
            return Err(conflict("materialized view", n));
        }
        match work.views.get(n) {
            Some(v) => next.views.insert(n.clone(), v.clone()),
            None => next.views.remove(n),
        };
    }
    Ok(next)
}

impl Engine {
    /// `true` while the current session (or the engine's caller) has a
    /// transaction open.
    pub fn in_transaction(&self) -> bool {
        self.txn.is_some()
    }

    /// Session holding the installed transaction (`Some(None)` when it was
    /// begun outside a session).
    pub(crate) fn transaction_holder(&self) -> Option<Option<u64>> {
        self.txn.as_ref().map(|t| t.session)
    }

    /// `true` when `session` has a transaction open (installed or not).
    pub(crate) fn has_transaction(&self, session: Option<u64>) -> bool {
        self.transaction_holder() == Some(session) || self.parked.contains_key(&session)
    }

    /// Number of open transactions, across sessions.
    pub fn open_transactions(&self) -> usize {
        usize::from(self.txn.is_some()) + self.parked.len()
    }

    /// Sessions with an open transaction.
    pub(crate) fn transaction_sessions(&self) -> Vec<Option<u64>> {
        self.transaction_holder()
            .into_iter()
            .chain(self.parked.keys().copied())
            .collect()
    }

    /// The current table-and-view state.
    fn capture(&self) -> Snapshot {
        let catalog_tables = self.catalog_table_names();
        Snapshot {
            tables: self.tables.clone(),
            storage: self
                .storage_tables
                .borrow()
                .iter()
                .filter(|(n, _)| !catalog_tables.contains(n))
                .map(|(n, e)| (n.clone(), Arc::clone(&e.table)))
                .collect(),
            table_versions: self.table_versions.borrow().clone(),
            views: self.views.borrow().clone(),
            #[cfg(feature = "storage")]
            catalog: self.catalog.as_ref().map(|c| c.state()),
        }
    }

    /// Names of catalog tables (current manifest).
    fn catalog_table_names(&self) -> Vec<String> {
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_ref() {
            return cat.manifest().tables.keys().cloned().collect();
        }
        Vec::new()
    }

    /// Make `s` the state statements see. Only what differs from the
    /// current state is re-registered (so device copies of unchanged tables
    /// survive a switch between sessions).
    fn install(&mut self, s: Snapshot) -> BoltResult<()> {
        // Catalog tables: re-open what changed, drop what is gone.
        #[cfg(feature = "storage")]
        let restored: Vec<String> = match (self.catalog.as_mut(), s.catalog) {
            (Some(cat), Some(state)) => {
                let before = cat.manifest().tables.clone();
                cat.set_state(state);
                let after = cat.manifest().tables.clone();
                let mut names: Vec<&String> = before.keys().chain(after.keys()).collect();
                names.sort();
                names.dedup();
                for name in names {
                    match after.get(name) {
                        Some(t) if before.get(name) != Some(t) || !self.has_table(name) => {
                            self.register_catalog_table(name)?
                        }
                        Some(_) => {}
                        None if self.has_table(name) && !s.tables.contains_key(name) => {
                            self.deregister_table(name)?
                        }
                        None => {}
                    }
                }
                after.keys().cloned().collect()
            }
            _ => Vec::new(),
        };
        #[cfg(not(feature = "storage"))]
        let restored: Vec<String> = Vec::new();
        // In-memory tables.
        let current: Vec<String> = self.tables.keys().cloned().collect();
        for name in current {
            if !s.tables.contains_key(&name) {
                self.deregister_table(&name)?;
            }
        }
        for (name, batches) in &s.tables {
            let unchanged = self.tables.contains_key(name)
                && self.table_versions.borrow().get(name) == s.table_versions.get(name);
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
        // Registered storage tables (non-catalog) and lazy streams added in
        // the other state are removed; missing ones come back.
        let registered: Vec<(String, *const u8)> = self
            .storage_tables
            .borrow()
            .iter()
            .filter(|(n, _)| !restored.contains(n))
            .map(|(n, e)| (n.clone(), Arc::as_ptr(&e.table) as *const u8))
            .collect();
        for (name, ptr) in registered {
            let same = s
                .storage
                .iter()
                .any(|(n, t)| *n == name && Arc::as_ptr(t) as *const u8 == ptr);
            if !same {
                self.deregister_table(&name)?;
            }
        }
        for (name, table) in &s.storage {
            if !self.has_table(name) {
                self.register_storage_table(name, Arc::clone(table))?;
            }
        }
        *self.views.borrow_mut() = s.views;
        *self.table_versions.borrow_mut() = s.table_versions;
        self.bump_data_epoch();
        Ok(())
    }

    /// Install the state of `session`: its open transaction's, or the
    /// committed state when it has none. The runtime calls this before each
    /// statement; switching parks the outgoing transaction's state.
    pub(crate) fn activate_session(&mut self, session: Option<u64>) -> BoltResult<()> {
        let installed = self.transaction_holder();
        let target = self.has_transaction(session).then_some(session);
        if installed == target {
            return Ok(());
        }
        match self.txn.take() {
            Some(t) => {
                let work = self.capture();
                self.parked.insert(t.session, (t, work));
            }
            None => self.committed = Some(self.capture()),
        }
        match target {
            None => {
                let c = self.committed.take().expect("committed state saved");
                self.install(c)
            }
            Some(s) => {
                let (t, work) = self.parked.remove(&s).expect("parked transaction");
                self.install(work)?;
                self.txn = Some(t);
                Ok(())
            }
        }
    }

    /// Install the committed state (parking an installed transaction).
    pub(crate) fn activate_committed(&mut self) -> BoltResult<()> {
        let Some(t) = self.txn.take() else {
            return Ok(());
        };
        let work = self.capture();
        self.parked.insert(t.session, (t, work));
        let c = self.committed.take().expect("committed state saved");
        self.install(c)
    }

    /// `BEGIN`.
    pub fn begin_transaction(&mut self) -> BoltResult<()> {
        if self.txn.is_some() {
            return Err(BoltError::Plan("a transaction is already open".into()));
        }
        let committed = self.capture();
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_mut() {
            if !cat.is_read_only() {
                cat.begin()?;
            }
        }
        self.committed = Some(committed);
        self.txn = Some(Transaction {
            session: self.current_session,
            aborted: false,
            base: self.capture(),
            savepoints: Vec::new(),
        });
        Ok(())
    }

    /// `COMMIT`: publish the transaction's changes, merged into what was
    /// committed since it began. An aborted transaction is rolled back
    /// instead, and the error says so; so is one whose changes conflict
    /// with a later commit.
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
        let committed = self.committed.as_ref().expect("committed state saved");
        let work = self.capture();
        let merged = match merge(&txn.base, &work, committed) {
            Ok(m) => m,
            Err(e) => {
                self.rollback_transaction()?;
                return Err(e);
            }
        };
        #[cfg(feature = "storage")]
        let mut merged = merged;
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_mut() {
            if cat.in_transaction() {
                let published = committed
                    .catalog
                    .as_ref()
                    .map(|c| c.manifest().clone())
                    .unwrap_or_else(|| cat.manifest().clone());
                let r = cat.commit_concurrent(published);
                merged.catalog = Some(cat.state());
                // `install` re-opens the catalog tables that differ from the
                // ones registered now, i.e. the transaction's.
                if let Some(state) = work.catalog.clone() {
                    cat.set_state(state);
                }
                if let Err(e) = r {
                    // Nothing was published.
                    self.rollback_transaction()?;
                    return Err(e);
                }
            } else {
                merged.catalog = Some(cat.state());
            }
        }
        self.txn = None;
        self.committed = None;
        self.install(merged)
    }

    /// `ROLLBACK`: discard the transaction's changes.
    pub fn rollback_transaction(&mut self) -> BoltResult<()> {
        if self.txn.take().is_none() {
            return Err(BoltError::Plan("no transaction is open".into()));
        }
        let c = self.committed.take().expect("committed state saved");
        self.install(c)
    }

    /// `SAVEPOINT name`: remember the transaction's current state.
    pub fn savepoint(&mut self, name: &str) -> BoltResult<()> {
        self.ensure_transaction_usable()?;
        let snapshot = self.capture();
        let t = self
            .txn
            .as_mut()
            .ok_or_else(|| BoltError::Plan("SAVEPOINT needs an open transaction".into()))?;
        t.savepoints.push((name.to_string(), snapshot));
        Ok(())
    }

    /// Index of the newest savepoint called `name`.
    fn savepoint_index(&self, name: &str) -> BoltResult<usize> {
        let t = self
            .txn
            .as_ref()
            .ok_or_else(|| BoltError::Plan("no transaction is open".into()))?;
        t.savepoints
            .iter()
            .rposition(|(n, _)| n.eq_ignore_ascii_case(name))
            .ok_or_else(|| BoltError::Plan(format!("savepoint '{name}' does not exist")))
    }

    /// `ROLLBACK TO SAVEPOINT name`: undo what came after the savepoint
    /// (which stays defined; later ones are released). Also ends an
    /// aborted state.
    pub fn rollback_to_savepoint(&mut self, name: &str) -> BoltResult<()> {
        let i = self.savepoint_index(name)?;
        let t = self.txn.as_mut().expect("checked");
        t.savepoints.truncate(i + 1);
        t.aborted = false;
        let snapshot = t.savepoints[i].1.clone();
        self.install(snapshot)
    }

    /// `RELEASE SAVEPOINT name`: forget the savepoint (and later ones),
    /// keeping their changes.
    pub fn release_savepoint(&mut self, name: &str) -> BoltResult<()> {
        self.ensure_transaction_usable()?;
        let i = self.savepoint_index(name)?;
        self.txn.as_mut().expect("checked").savepoints.truncate(i);
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
