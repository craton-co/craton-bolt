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
//! * [`RefreshMode::Scheduled`] views are refreshed when stale and their
//!   interval has passed since the last refresh — before each
//!   [`Engine::execute_sql`] statement, by [`Engine::refresh_due_views`], and
//!   by an idle [`Database`](crate::Database) runtime.
//! * [`Engine::view_status`] reports staleness.
//!
//! **Incremental refresh.** A durable view of the shape `SELECT keys,
//! SUM / COUNT / MIN / MAX(...) FROM t [WHERE ...] GROUP BY keys` (or
//! without `GROUP BY`) over one catalog table refreshes from only the data
//! files the table gained since the last refresh, merging the new partial
//! aggregates into the view's rows, as long as the table changed by appends
//! only; any other change (a rewrite by `DELETE` / `UPDATE` / `COMPACT` /
//! `ALTER`) refreshes in full.
//!
//! **Query rewrite.** A query whose plan is the plan of a fresh view's
//! defining query is answered from the view's rows (unless disabled with
//! [`EngineBuilder::view_rewrite`](crate::EngineBuilder::view_rewrite)).
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
    /// Also automatically when stale, at most once per interval.
    Scheduled(std::time::Duration),
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
    /// The view's next refresh can be incremental (see the module docs).
    pub incremental: bool,
    /// The last refresh was incremental.
    pub last_refresh_incremental: bool,
    /// Wall-clock time of the last refresh.
    pub refreshed_at: Option<std::time::SystemTime>,
}

/// In-engine definition of a materialized view.
#[derive(Debug, Clone)]
pub(crate) struct ViewDef {
    pub(crate) sql: String,
    pub(crate) dependencies: Vec<String>,
    pub(crate) base_versions: BTreeMap<String, u64>,
    pub(crate) refresh: RefreshMode,
    pub(crate) refreshed_at: Option<std::time::SystemTime>,
    /// Catalog data files of each base table at the last refresh.
    #[cfg_attr(not(feature = "storage"), allow(dead_code))]
    pub(crate) base_files: BTreeMap<String, Vec<String>>,
    pub(crate) last_incremental: bool,
}

/// How each column of an incrementally maintainable view merges.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(not(feature = "storage"), allow(dead_code))]
enum MergeKind {
    Key,
    Sum,
    Min,
    Max,
}

/// An incrementally maintainable view: its one base table and, per output
/// column, how partial results merge.
#[cfg_attr(not(feature = "storage"), allow(dead_code))]
struct Incremental {
    base: String,
    columns: Vec<(String, MergeKind)>,
}

/// `Some` when `plan` (a view's defining query) is
/// `[Project] <- Aggregate(COUNT / SUM / MIN / MAX) <- Filter / Project <- Scan`
/// and every output column is a group key or an aggregate.
#[cfg_attr(not(feature = "storage"), allow(dead_code))]
fn incremental_shape(plan: &crate::plan::logical_plan::LogicalPlan) -> Option<Incremental> {
    use crate::plan::logical_plan::{AggregateExpr, Expr, LogicalPlan};
    let (agg, top) = match plan {
        LogicalPlan::Project { input, exprs } => (input.as_ref(), Some(exprs)),
        other => (other, None),
    };
    let LogicalPlan::Aggregate {
        input,
        group_by,
        aggregates,
    } = agg
    else {
        return None;
    };
    let base = crate::exec::engine_ooc::single_base_table(input)?.to_string();
    let agg_schema = agg.schema().ok()?;
    let mut kinds: Vec<(String, MergeKind)> = Vec::new();
    for (i, f) in agg_schema.fields.iter().enumerate() {
        let kind = if i < group_by.len() {
            MergeKind::Key
        } else {
            match &aggregates[i - group_by.len()] {
                AggregateExpr::Count(_) | AggregateExpr::Sum(_) => MergeKind::Sum,
                AggregateExpr::Min(_) => MergeKind::Min,
                AggregateExpr::Max(_) => MergeKind::Max,
                _ => return None,
            }
        };
        kinds.push((f.name.clone(), kind));
    }
    let columns = match top {
        None => kinds,
        Some(exprs) => exprs
            .iter()
            .map(|e| {
                let (inner, out) = match e {
                    Expr::Alias(x, n) => (x.as_ref(), n.clone()),
                    Expr::Column(c) => (e, c.clone()),
                    _ => return None,
                };
                let Expr::Column(c) = inner else {
                    return None;
                };
                let kind = kinds.iter().find(|(n, _)| n == c)?.1;
                Some((out, kind))
            })
            .collect::<Option<Vec<_>>>()?,
    };
    Some(Incremental { base, columns })
}

#[cfg_attr(not(feature = "storage"), allow(dead_code))]
fn now_ms(t: std::time::SystemTime) -> u64 {
    t.duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
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
        let incremental = self.incremental_delta(def).is_some();
        Ok(ViewStatus {
            fresh: stale.is_empty(),
            stale_dependencies: stale,
            dependencies: def.dependencies.clone(),
            refresh: def.refresh,
            incremental,
            last_refresh_incremental: def.last_incremental,
            refreshed_at: def.refreshed_at,
        })
    }

    /// Catalog data files of each of `deps` that is a catalog table.
    fn base_files(&self, deps: &[String]) -> BTreeMap<String, Vec<String>> {
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_ref() {
            return deps
                .iter()
                .filter_map(|d| {
                    cat.table(d)
                        .map(|t| (d.clone(), t.files.iter().map(|f| f.path.clone()).collect()))
                })
                .collect();
        }
        let _ = deps;
        BTreeMap::new()
    }

    /// For a view whose next refresh can be incremental: its shape and the
    /// base table's data files added since the last refresh.
    fn incremental_delta(&self, def: &ViewDef) -> Option<(Incremental, Vec<String>)> {
        #[cfg(feature = "storage")]
        {
            let [dep] = def.dependencies.as_slice() else {
                return None;
            };
            let cat = self.catalog.as_ref()?;
            let t = cat.table(dep)?;
            if !matches!(t.kind, crate::storage::catalog::TableKind::Managed) {
                return None;
            }
            let old = def.base_files.get(dep)?;
            let now: Vec<&String> = t.files.iter().map(|f| &f.path).collect();
            if !old.iter().all(|f| now.contains(&f)) {
                return None;
            }
            let plan = crate::plan::sql_frontend::parse(&def.sql, &self.provider).ok()?;
            let shape = incremental_shape(&plan)?;
            if shape.base != *dep {
                return None;
            }
            let added = now
                .into_iter()
                .filter(|f| !old.contains(f))
                .cloned()
                .collect();
            Some((shape, added))
        }
        #[cfg(not(feature = "storage"))]
        {
            let _ = def;
            None
        }
    }

    /// The view's new rows from its current rows plus the partial
    /// aggregates of `added` (the base table's new data files).
    #[cfg(feature = "storage")]
    fn incremental_rows(
        &mut self,
        name: &str,
        def: &ViewDef,
        shape: &Incremental,
        added: &[String],
    ) -> BoltResult<arrow_array::RecordBatch> {
        use crate::plan::logical_plan::{AggregateExpr, Expr, LogicalPlan};
        use crate::storage::StorageTable as _;
        const DELTA: &str = "__bolt_mv_delta";
        const OLD: &str = "__bolt_mv_rows";
        let view_schema = self.table_arrow_schema(name)?;
        let cat = self
            .catalog
            .as_ref()
            .expect("incremental views are durable");
        // The base table's new rows.
        let base_arrow = self.table_arrow_schema(&shape.base)?;
        let paths: Vec<std::path::PathBuf> = added.iter().map(|p| cat.root().join(p)).collect();
        let files = crate::storage::ParquetTable::with_schema(base_arrow.clone(), &paths)?;
        let sel = files.plan_scan(&crate::storage::ScanRequest::full());
        let delta_rows: Vec<arrow_array::RecordBatch> =
            files.read(&sel)?.collect::<BoltResult<_>>()?;
        let delta_rows = arrow::compute::concat_batches(&base_arrow, &delta_rows)
            .map_err(|e| BoltError::Other(format!("incremental refresh: {e}")))?;
        // The view's current rows.
        let current = {
            let t = cat.open_table(name)?;
            let sel = t.plan_scan(&crate::storage::ScanRequest::full());
            let rows: Vec<arrow_array::RecordBatch> = t.read(&sel)?.collect::<BoltResult<_>>()?;
            arrow::compute::concat_batches(&view_schema, &rows)
                .map_err(|e| BoltError::Other(format!("incremental refresh: {e}")))?
        };
        for t in [DELTA, OLD] {
            if self.has_table(t) {
                self.deregister_table(t)?;
            }
        }
        self.register_table(DELTA, delta_rows)?;
        let result = (|| -> BoltResult<arrow_array::RecordBatch> {
            // Partial results of the defining query over the new rows.
            let plan = crate::plan::sql_frontend::parse(&def.sql, &self.provider)?;
            let base_schema =
                crate::exec::engine_support::arrow_schema_to_plan_schema(&base_arrow)?;
            let delta_plan =
                crate::exec::engine_ooc::rename_scans(&plan, &shape.base, DELTA, &base_schema);
            let partial = self
                .execute_logical(&delta_plan, true, false)?
                .into_record_batch();
            let partial = crate::exec::engine_ddl::align_batch(&partial, &view_schema, &[])?;
            let merged_input =
                arrow::compute::concat_batches(&view_schema, [&current, &partial])
                    .map_err(|e| BoltError::Other(format!("incremental refresh: {e}")))?;
            self.register_table(OLD, merged_input)?;
            // Re-aggregate: group keys, SUM of sums and counts, MIN / MAX.
            let schema = crate::exec::engine_support::arrow_schema_to_plan_schema(&view_schema)?;
            let keys: Vec<Expr> = shape
                .columns
                .iter()
                .filter(|(_, k)| *k == MergeKind::Key)
                .map(|(c, _)| Expr::Column(c.clone()))
                .collect();
            let aggs: Vec<AggregateExpr> = shape
                .columns
                .iter()
                .filter(|(_, k)| *k != MergeKind::Key)
                .map(|(c, k)| {
                    let col = Expr::Column(c.clone());
                    match k {
                        MergeKind::Min => AggregateExpr::Min(col),
                        MergeKind::Max => AggregateExpr::Max(col),
                        _ => AggregateExpr::Sum(col),
                    }
                })
                .collect();
            let aggregate = LogicalPlan::Aggregate {
                input: Box::new(LogicalPlan::Scan {
                    table: OLD.to_string(),
                    projection: None,
                    schema,
                }),
                group_by: keys,
                aggregates: aggs.clone(),
            };
            let agg_schema = aggregate.schema()?;
            // Back to the view's column order and names.
            let mut next_agg = shape
                .columns
                .iter()
                .filter(|(_, k)| *k == MergeKind::Key)
                .count();
            let mut next_key = 0;
            let exprs: Vec<Expr> = shape
                .columns
                .iter()
                .map(|(c, k)| {
                    let src = if *k == MergeKind::Key {
                        next_key += 1;
                        agg_schema.fields[next_key - 1].name.clone()
                    } else {
                        next_agg += 1;
                        agg_schema.fields[next_agg - 1].name.clone()
                    };
                    Expr::Alias(Box::new(Expr::Column(src)), c.clone())
                })
                .collect();
            let merge = LogicalPlan::Project {
                input: Box::new(aggregate),
                exprs,
            };
            let rows = self
                .execute_logical(&merge, false, false)?
                .into_record_batch();
            crate::exec::engine_ddl::align_batch(&rows, &view_schema, &[])
        })();
        for t in [DELTA, OLD] {
            if self.has_table(t) {
                self.deregister_table(t)?;
            }
        }
        result
    }

    /// Refresh every stale [`RefreshMode::Scheduled`] view whose interval
    /// has passed. Returns the number refreshed.
    pub fn refresh_due_views(&mut self) -> BoltResult<usize> {
        let now = std::time::SystemTime::now();
        let due: Vec<String> = self
            .view_names()
            .into_iter()
            .filter(|v| {
                let views = self.views.borrow();
                let def = &views[v];
                let RefreshMode::Scheduled(every) = def.refresh else {
                    return false;
                };
                let elapsed = def
                    .refreshed_at
                    .and_then(|t| now.duration_since(t).ok())
                    .unwrap_or(every);
                elapsed >= every
            })
            .filter(|v| self.view_status(v).is_ok_and(|s| !s.fresh))
            .collect();
        for v in &due {
            self.refresh_view(v)?;
        }
        Ok(due.len())
    }

    /// `true` when some view refreshes on a schedule.
    pub(crate) fn has_scheduled_views(&self) -> bool {
        self.views
            .borrow()
            .values()
            .any(|d| matches!(d.refresh, RefreshMode::Scheduled(_)))
    }

    /// The fresh view whose defining query has `plan` as its plan, if any.
    pub(crate) fn matching_view(
        &self,
        plan: &crate::plan::logical_plan::LogicalPlan,
    ) -> Option<String> {
        if !self.view_rewrite {
            return None;
        }
        let views = self.views.borrow();
        if views.is_empty() {
            return None;
        }
        let key = format!("{plan:?}");
        let mut names: Vec<&String> = views.keys().collect();
        names.sort();
        names
            .into_iter()
            .find(|v| {
                let def = &views[*v];
                crate::plan::sql_frontend::parse(&def.sql, &self.provider)
                    .is_ok_and(|p| format!("{p:?}") == key)
                    && def
                        .dependencies
                        .iter()
                        .all(|d| self.table_version(d) == def.base_versions.get(d).copied())
            })
            .cloned()
    }

    /// Execute `plan` from the rows of a fresh view with the same plan.
    pub(crate) fn answer_from_view(
        &self,
        plan: &crate::plan::logical_plan::LogicalPlan,
    ) -> BoltResult<Option<crate::exec::engine::QueryHandle>> {
        let Some(view) = self.matching_view(plan) else {
            return Ok(None);
        };
        let schema = crate::plan::TableProvider::schema(&self.provider, &view)?;
        let scan = crate::plan::logical_plan::LogicalPlan::Scan {
            table: view,
            projection: None,
            schema,
        };
        crate::metrics::metrics().inc(crate::metrics::Counter::ViewRewrites);
        self.execute_logical(&scan, true, false).map(Some)
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
            base_files: self.base_files(&deps),
            dependencies: deps,
            refresh,
            refreshed_at: Some(std::time::SystemTime::now()),
            last_incremental: false,
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
        #[cfg(feature = "storage")]
        let incremental = if self.view_status(name).is_ok_and(|s| s.fresh) {
            None
        } else {
            self.incremental_delta(&def)
        };
        #[cfg(not(feature = "storage"))]
        let incremental: Option<(Incremental, Vec<String>)> = None;
        let target = self.table_arrow_schema(name)?;
        let (rows, last_incremental) = match incremental {
            #[cfg(feature = "storage")]
            Some((shape, added)) => (self.incremental_rows(name, &def, &shape, &added)?, true),
            #[cfg(not(feature = "storage"))]
            Some(_) => unreachable!("no catalog without the storage feature"),
            None => {
                // A query over this view's own definition must not be
                // answered from the stale rows being replaced.
                let saved = std::mem::replace(&mut self.view_rewrite, false);
                let rows = self.sql(&def.sql);
                self.view_rewrite = saved;
                (rows?.into_record_batch(), false)
            }
        };
        let n = rows.num_rows() as u64;
        let rows = crate::exec::engine_ddl::align_batch(&rows, &target, &[])?;
        let updated = ViewDef {
            base_versions: self.snapshot_versions(&def.dependencies),
            base_files: self.base_files(&def.dependencies),
            refreshed_at: Some(std::time::SystemTime::now()),
            last_incremental,
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
                RefreshMode::Scheduled(d) => RefreshPolicy::Scheduled {
                    seconds: d.as_secs().max(1),
                },
            },
            backing_table: name.to_string(),
            refreshed_at_ms: def.refreshed_at.map(now_ms),
            base_files: def.base_files.clone(),
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
                            RefreshPolicy::Scheduled { seconds } => {
                                RefreshMode::Scheduled(std::time::Duration::from_secs(seconds))
                            }
                        },
                        refreshed_at: m
                            .refreshed_at_ms
                            .map(|ms| std::time::UNIX_EPOCH + std::time::Duration::from_millis(ms)),
                        base_files: m.base_files.clone(),
                        last_incremental: false,
                    },
                )
            })
            .collect();
        *self.views.borrow_mut() = defs;
    }
}
