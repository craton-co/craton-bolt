// SPDX-License-Identifier: Apache-2.0
//! Materialized views.
//!
//! A materialized view is a table holding the result of its defining query,
//! plus the metadata to know when that result is out of date: the base
//! tables the query reads and each one's data version at the last refresh.
//! Queries read the view like any table (no automatic query rewrite).
//!
//! * `REFRESH MATERIALIZED VIEW v` (or [`Engine::refresh_view`]) recomputes
//!   it in full.
//! * [`RefreshMode::OnCommit`] views are also refreshed automatically after
//!   any [`Engine::execute_sql`] statement that changes one of their base
//!   tables, and before each `execute_sql` statement if a change arrived
//!   through the Rust registration API meanwhile.
//! * [`Engine::view_status`] reports staleness.
//!
//! With a catalog attached, a view is durable: its rows are a managed
//! catalog table and its definition and base-table versions are part of the
//! catalog snapshot, so staleness survives a restart. A durable view may
//! read only catalog tables.

use std::collections::BTreeMap;

use crate::error::{BoltError, BoltResult};
use crate::exec::engine::Engine;

/// When a materialized view is refreshed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RefreshMode {
    /// Only by `REFRESH MATERIALIZED VIEW`.
    Manual,
    /// Also automatically after a statement that changes a base table.
    OnCommit,
}

/// Freshness of a materialized view.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ViewStatus {
    /// `true` when no base table changed since the last refresh.
    pub fresh: bool,
    /// Base tables changed (or dropped) since the last refresh.
    pub stale_dependencies: Vec<String>,
    /// Base tables the defining query reads.
    pub dependencies: Vec<String>,
    /// Refresh policy.
    pub refresh: RefreshMode,
}

/// In-engine definition of a materialized view.
#[derive(Debug, Clone)]
pub(crate) struct ViewDef {
    pub(crate) sql: String,
    pub(crate) dependencies: Vec<String>,
    pub(crate) base_versions: BTreeMap<String, u64>,
    pub(crate) refresh: RefreshMode,
}

impl Engine {
    /// Data version of table `name`: changes whenever its rows change.
    pub(crate) fn table_version(&self, name: &str) -> Option<u64> {
        #[cfg(feature = "storage")]
        if let Some(t) = self.catalog.as_ref().and_then(|c| c.table(name)) {
            return Some(t.table_version);
        }
        if !self.has_table(name) {
            return None;
        }
        Some(self.table_versions.borrow().get(name).copied().unwrap_or(0))
    }

    /// Record that table `name`'s data changed.
    pub(crate) fn touch_table(&self, name: &str) {
        self.bump_data_epoch();
        let v = self.data_epoch.get();
        self.table_versions.borrow_mut().insert(name.to_string(), v);
    }

    /// `true` when `name` is a materialized view.
    pub fn view_exists(&self, name: &str) -> bool {
        self.views.borrow().contains_key(name)
    }

    /// Names of the materialized views, sorted.
    pub fn view_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.views.borrow().keys().cloned().collect();
        v.sort();
        v
    }

    /// Freshness of materialized view `name`.
    pub fn view_status(&self, name: &str) -> BoltResult<ViewStatus> {
        let views = self.views.borrow();
        let def = views
            .get(name)
            .ok_or_else(|| BoltError::Plan(format!("materialized view '{name}' does not exist")))?;
        let stale: Vec<String> = def
            .dependencies
            .iter()
            .filter(|d| self.table_version(d) != def.base_versions.get(*d).copied())
            .cloned()
            .collect();
        Ok(ViewStatus {
            fresh: stale.is_empty(),
            stale_dependencies: stale,
            dependencies: def.dependencies.clone(),
            refresh: def.refresh,
        })
    }

    /// Base tables `sql` reads. Planned through the frontend when possible;
    /// a shape the generic planner does not take (recursive CTE, LATERAL, …)
    /// falls back to every registered table named in the text, which can
    /// only over-approximate.
    fn view_dependencies(&self, sql: &str) -> Vec<String> {
        let mut deps = match crate::plan::sql_frontend::parse(sql, &self.provider) {
            Ok(plan) => crate::exec::engine_query::deep_scan_tables(&plan),
            Err(_) => {
                let words: std::collections::HashSet<&str> = sql
                    .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.'))
                    .collect();
                self.table_names()
                    .into_iter()
                    .filter(|t| words.contains(t.as_str()))
                    .collect()
            }
        };
        deps.sort();
        deps.dedup();
        deps
    }

    fn snapshot_versions(&self, deps: &[String]) -> BTreeMap<String, u64> {
        deps.iter()
            .filter_map(|d| self.table_version(d).map(|v| (d.clone(), v)))
            .collect()
    }

    /// Create materialized view `name` over `sql` and compute it. Returns
    /// its row count.
    pub fn create_view(
        &mut self,
        name: &str,
        sql: &str,
        refresh: RefreshMode,
        or_replace: bool,
    ) -> BoltResult<u64> {
        if self.view_exists(name) {
            if !or_replace {
                return Err(BoltError::Plan(format!(
                    "materialized view '{name}' already exists"
                )));
            }
            self.drop_view(name)?;
        } else if self.has_table(name) {
            return Err(BoltError::Plan(format!(
                "a table named '{name}' already exists"
            )));
        }
        let deps = self.view_dependencies(sql);
        if deps.iter().any(|d| d == name) {
            return Err(BoltError::Plan(format!(
                "materialized view '{name}' cannot read itself"
            )));
        }
        #[cfg(feature = "storage")]
        let durable = self.catalog.as_ref().is_some_and(|c| !c.is_read_only());
        #[cfg(not(feature = "storage"))]
        let durable = false;
        #[cfg(feature = "storage")]
        if durable {
            let cat = self.catalog.as_ref().expect("durable");
            if let Some(d) = deps.iter().find(|d| cat.table(d).is_none()) {
                return Err(BoltError::Plan(format!(
                    "a durable materialized view can read only catalog tables; '{d}' is not one"
                )));
            }
        }
        let rows = self.sql(sql)?.into_record_batch();
        let n = rows.num_rows() as u64;
        self.create_table(name, rows, !durable)?;
        let def = ViewDef {
            sql: sql.to_string(),
            base_versions: self.snapshot_versions(&deps),
            dependencies: deps,
            refresh,
        };
        #[cfg(feature = "storage")]
        if durable {
            self.persist_view(name, &def)?;
        }
        self.views.borrow_mut().insert(name.to_string(), def);
        Ok(n)
    }

    /// Recompute materialized view `name` from its defining query. Returns
    /// its new row count.
    pub fn refresh_view(&mut self, name: &str) -> BoltResult<u64> {
        let def =
            self.views.borrow().get(name).cloned().ok_or_else(|| {
                BoltError::Plan(format!("materialized view '{name}' does not exist"))
            })?;
        let rows = self.sql(&def.sql)?.into_record_batch();
        let n = rows.num_rows() as u64;
        let target = self.table_arrow_schema(name)?;
        let rows = crate::exec::engine_ddl::align_batch(&rows, &target, &[])?;
        let updated = ViewDef {
            base_versions: self.snapshot_versions(&def.dependencies),
            ..def
        };
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_mut() {
            if cat.table(name).is_some() {
                cat.replace_contents(name, std::slice::from_ref(&rows))?;
                self.register_catalog_table(name)?;
                self.persist_view(name, &updated)?;
                self.views.borrow_mut().insert(name.to_string(), updated);
                return Ok(n);
            }
        }
        self.replace_table(name, rows)?;
        self.views.borrow_mut().insert(name.to_string(), updated);
        Ok(n)
    }

    /// Drop materialized view `name` and its rows.
    pub fn drop_view(&mut self, name: &str) -> BoltResult<()> {
        if self.views.borrow_mut().remove(name).is_none() {
            return Err(BoltError::Plan(format!(
                "materialized view '{name}' does not exist"
            )));
        }
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_mut() {
            if cat.manifest().views.contains_key(name) {
                cat.remove_view(name)?;
            }
        }
        if self.has_table(name) {
            self.drop_table(name)?;
        }
        Ok(())
    }

    /// Refresh every stale [`RefreshMode::OnCommit`] view (views over views
    /// in dependency order).
    pub(crate) fn refresh_stale_on_commit_views(&mut self) -> BoltResult<()> {
        let limit = self.views.borrow().len() + 1;
        for _ in 0..limit {
            let stale: Vec<String> = self
                .view_names()
                .into_iter()
                .filter(|v| {
                    self.views.borrow()[v].refresh == RefreshMode::OnCommit
                        && self.view_status(v).is_ok_and(|s| !s.fresh)
                })
                .collect();
            // Refresh views whose own dependencies are all fresh first.
            let ready: Vec<String> = stale
                .iter()
                .filter(|v| {
                    let deps = self.views.borrow()[*v].dependencies.clone();
                    !deps.iter().any(|d| stale.contains(d))
                })
                .cloned()
                .collect();
            if ready.is_empty() {
                return Ok(());
            }
            for v in ready {
                self.refresh_view(&v)?;
            }
        }
        Ok(())
    }

    /// Hook run after a statement changed table `name`.
    pub(crate) fn after_table_change(&mut self, name: &str) -> BoltResult<()> {
        self.touch_table(name);
        self.refresh_stale_on_commit_views()
    }

    #[cfg(feature = "storage")]
    fn persist_view(&mut self, name: &str, def: &ViewDef) -> BoltResult<()> {
        use crate::storage::catalog::{RefreshPolicy, ViewMeta};
        let meta = ViewMeta {
            sql: def.sql.clone(),
            dependencies: def.dependencies.clone(),
            base_versions: def.base_versions.clone(),
            refresh: match def.refresh {
                RefreshMode::Manual => RefreshPolicy::Manual,
                RefreshMode::OnCommit => RefreshPolicy::OnCommit,
            },
            backing_table: name.to_string(),
        };
        self.catalog
            .as_mut()
            .expect("catalog attached")
            .put_view(name, meta)
    }

    /// Load the catalog's view definitions into the engine.
    #[cfg(feature = "storage")]
    pub(crate) fn load_catalog_views(&mut self) {
        use crate::storage::catalog::RefreshPolicy;
        let Some(cat) = self.catalog.as_ref() else {
            return;
        };
        let defs: std::collections::HashMap<String, ViewDef> = cat
            .manifest()
            .views
            .iter()
            .map(|(n, m)| {
                (
                    n.clone(),
                    ViewDef {
                        sql: m.sql.clone(),
                        dependencies: m.dependencies.clone(),
                        base_versions: m.base_versions.clone(),
                        refresh: match m.refresh {
                            RefreshPolicy::Manual => RefreshMode::Manual,
                            RefreshPolicy::OnCommit => RefreshMode::OnCommit,
                        },
                    },
                )
            })
            .collect();
        *self.views.borrow_mut() = defs;
    }
}
