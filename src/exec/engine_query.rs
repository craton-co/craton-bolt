// SPDX-License-Identifier: Apache-2.0

//! The SQL / DataFrame query front door of [`Engine`]: `sql`, `explain_sql`,
//! `run_logical_plan` and the internal `run_subplan`, plus the single
//! [`QueryCompiler`] pipeline every one of them shares.
//!
//! ## One pipeline
//!
//! Every logical plan the engine executes — a top-level SQL query, a
//! DataFrame plan, a resolved subquery, or a component subplan of a
//! host-orchestrated shape (recursive CTE, LATERAL, correlated WHERE, ...) —
//! passes through the same stages in the same order:
//!
//! 1. dictionary rewrite (`Utf8` literal predicates → `__idx_<col>` folds),
//! 2. the built-in logical optimizer, driven to a bounded fixpoint,
//! 3. user [`PlanRewrite`](crate::plan::PlanRewrite) hooks,
//! 4. uncorrelated subquery resolution (or, for `EXPLAIN`, an explicit
//!    "deferred" marker),
//! 5. physical lowering and validity propagation.
//!
//! Before step 1, every lazily-registered table the statement reads —
//! including inside its subqueries — is drained, validated and (without a
//! memory budget) given string dictionaries, so the dictionary rewrite sees
//! them. Repeated top-level SQL reuses the prepared physical plan from the
//! per-engine [`CompiledPlanCache`] until the data changes.
//!
//! Before this module existed each entry point carried its own copy of that
//! sequence and they had drifted: `run_logical_plan` skipped the user hooks,
//! `explain_sql` skipped the optimizer, `sql` ran the dictionary rewrite
//! twice, and planning failures bypassed `QueriesFailed`. Routing everything
//! through [`QueryCompiler::compile`] and [`Engine::run_top_level`] makes
//! those classes of drift structurally impossible.
//!
//! ## Query shapes
//!
//! SQL text is classified exactly once by
//! [`classify_query`](crate::plan::sql_frontend::classify_query). Both `sql`
//! and `explain_sql` dispatch on the resulting
//! [`QueryShape`](crate::plan::sql_frontend::QueryShape), so the executed
//! and the explained path can never disagree about which shape a statement
//! has.

use std::sync::Arc;
use std::time::Instant;

use arrow_array::RecordBatch;

use crate::error::{BoltError, BoltResult};
use crate::exec::dict_registry::DictRegistry;
use crate::exec::engine::{Engine, QueryHandle};
use crate::exec::engine_provider::{EngineProvider, EngineTableStats};
use crate::plan::sql_frontend::{classify_query, QueryShape, RecursiveQueryPlan};
use crate::plan::{LogicalPlan, PhysicalPlan, PlanRewrite, RowEstimator};

/// How [`QueryCompiler::compile`] treats uncorrelated subqueries.
pub(crate) enum Subqueries<'e> {
    /// Execute each subquery through the supplied executor and fold its
    /// result into the plan as a constant (the execution path).
    Resolve(&'e mut dyn FnMut(LogicalPlan) -> BoltResult<RecordBatch>),
    /// Leave subquery nodes in place and report that they were deferred
    /// (the `EXPLAIN` path, which must not execute anything).
    Defer,
}

/// Output of [`QueryCompiler::compile`].
#[derive(Debug)]
pub(crate) struct CompiledLogical {
    /// The fully rewritten / optimized logical plan.
    pub(crate) plan: LogicalPlan,
    /// `true` when [`Subqueries::Defer`] left at least one subquery node in
    /// `plan` (so a physical rendering of it is *pre*-resolution).
    pub(crate) deferred_subqueries: bool,
}

/// The logical half of the engine's query pipeline, borrowed from an
/// [`Engine`] (or assembled directly in tests — it needs no CUDA context).
pub(crate) struct QueryCompiler<'a> {
    pub(crate) dict: &'a DictRegistry,
    pub(crate) rewrites: &'a [Box<dyn PlanRewrite>],
    pub(crate) optimize: bool,
    pub(crate) estimator: Arc<dyn RowEstimator>,
}

impl QueryCompiler<'_> {
    /// Run the logical stages over `plan`: dictionary rewrite → built-in
    /// optimizer (bounded fixpoint) → user rewrites (when `user_rewrites`)
    /// → subquery handling per `subqueries`.
    ///
    /// `user_rewrites` is `false` only for a subquery being resolved out of
    /// an enclosing plan: the enclosing plan already went through the user
    /// hooks with the subquery embedded in it, so re-running them on the
    /// extracted subplan would apply each hook to that subplan twice.
    pub(crate) fn compile(
        &self,
        plan: &LogicalPlan,
        user_rewrites: bool,
        subqueries: Subqueries<'_>,
    ) -> BoltResult<CompiledLogical> {
        // String-literal predicates against Utf8 columns are folded into
        // integer equality against the corresponding `__idx_<col>` column.
        let plan = tracing::info_span!("plan").in_scope(|| self.dict.rewrite_plan(plan))?;
        // Built-in optimizer (constant folding, predicate pushdown,
        // filter-into-join, statistics-driven join reorder, projection
        // pruning), re-run until the plan stabilises or the iteration cap is
        // hit. `optimize` is `false` only for the test-only
        // `EngineBuilder::without_optimizer()`.
        let plan = if self.optimize {
            let passes = crate::plan::default_passes_with_estimator(Arc::clone(&self.estimator));
            crate::plan::optimizer::run_to_fixpoint(&passes, plan)?
        } else {
            plan
        };
        // User hooks run AFTER the built-in optimizer (so they see the
        // engine's normalised form, `__idx_<col>` refs included) and BEFORE
        // subquery resolution and lowering. See `crate::plan::rewrite`.
        let plan = if user_rewrites {
            self.rewrites.iter().try_fold(plan, |p, r| r.rewrite(p))?
        } else {
            plan
        };
        match subqueries {
            Subqueries::Resolve(exec) => Ok(CompiledLogical {
                plan: crate::exec::subquery_resolve::resolve_plan(plan, &mut &mut *exec)?,
                deferred_subqueries: false,
            }),
            Subqueries::Defer => {
                let deferred_subqueries = plan_has_subqueries(&plan);
                Ok(CompiledLogical {
                    plan,
                    deferred_subqueries,
                })
            }
        }
    }
}

/// Every base table `plan` scans, including inside its (nested) uncorrelated
/// subqueries. Uses the subquery resolver's own walker with a recording
/// executor (returning an empty one-column batch), so what counts as a
/// subquery cannot drift from what the execution path resolves; the folded
/// plan is discarded.
pub(crate) fn deep_scan_tables(plan: &LogicalPlan) -> Vec<String> {
    let (mut tree, nested) = split_scan_tables(plan);
    tree.extend(nested);
    tree
}

/// [`deep_scan_tables`] split into the tables of `plan`'s relational tree
/// and those scanned (only or also) inside its subqueries.
fn split_scan_tables(plan: &LogicalPlan) -> (Vec<String>, Vec<String>) {
    let mut nested: Vec<String> = Vec::new();
    let mut record = |subplan: LogicalPlan| -> BoltResult<RecordBatch> {
        nested.extend(deep_scan_tables(&subplan));
        let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "probe",
            arrow_schema::DataType::Int64,
            true,
        )]));
        Ok(RecordBatch::new_empty(schema))
    };
    let _ = crate::exec::subquery_resolve::resolve_plan(plan.clone(), &mut record);
    (crate::plan::logical_plan::collect_scan_tables(plan), nested)
}

/// `Plan` error for a statement [`Engine::sql`] does not run (DDL / DML /
/// catalog maintenance), pointing at [`Engine::execute_sql`].
fn reject_table_statement(query: &str) -> BoltResult<()> {
    let first = query
        .trim_start()
        .split(|c: char| c.is_whitespace() || c == '(')
        .next()
        .unwrap_or("")
        .to_ascii_uppercase();
    const STATEMENTS: [&str; 20] = [
        "CREATE", "INSERT", "DROP", "TRUNCATE", "COPY", "ANALYZE", "SHOW", "DESCRIBE", "DESC",
        "COMPACT", "VACUUM", "REFRESH", "DELETE", "UPDATE", "ALTER", "BEGIN", "START", "COMMIT",
        "ROLLBACK", "END",
    ];
    if STATEMENTS.contains(&first.as_str()) {
        return Err(BoltError::Plan(format!(
            "{first} statements are run with Engine::execute_sql (Engine::sql runs queries)"
        )));
    }
    Ok(())
}

/// The generic logical plan of `query` for [`Engine::sql_stream`]
/// (`None` for host-orchestrated shapes, which run through `sql`).
pub(crate) fn classify_for_stream(engine: &Engine, query: &str) -> BoltResult<Option<LogicalPlan>> {
    reject_table_statement(query)?;
    Ok(match classify_query(query, &engine.provider)? {
        QueryShape::Generic(plan) => Some(plan),
        _ => None,
    })
}

/// `true` if `plan` still carries a scalar / IN subquery anywhere.
///
/// Reuses the resolver's own walker (so "what counts as a subquery" cannot
/// drift from what the execution path resolves) with an executor that aborts
/// on the first subquery it is handed.
fn plan_has_subqueries(plan: &LogicalPlan) -> bool {
    let mut seen = false;
    let mut probe = |_: LogicalPlan| -> BoltResult<RecordBatch> {
        seen = true;
        Err(BoltError::Other("subquery probe".into()))
    };
    let _ = crate::exec::subquery_resolve::resolve_plan(plan.clone(), &mut probe);
    seen
}

impl Engine {
    /// This engine's [`QueryCompiler`] over a borrowed dictionary registry
    /// (`let registry = self.dict_registry.borrow();` at the call site, held
    /// for the duration of the compile).
    pub(crate) fn compiler<'a>(&'a self, dict: &'a DictRegistry) -> QueryCompiler<'a> {
        QueryCompiler {
            dict,
            rewrites: &self.rewrites,
            optimize: self.optimize,
            estimator: self.row_estimator(),
        }
    }

    /// Snapshot the registered tables' row counts into an estimator the
    /// cost-based optimizer can consume.
    ///
    /// Each base table's row count is the sum of its registered `RecordBatch`
    /// row counts (eager `tables` store) plus any already-materialised
    /// streaming source. Lazily-registered streaming tables that haven't been
    /// drained yet are intentionally omitted — costing them would force
    /// materialisation at plan time. Their absence simply leaves any join chain
    /// touching them un-reordered (the conservative default).
    pub(crate) fn row_estimator(&self) -> Arc<dyn RowEstimator> {
        let mut row_counts = std::collections::HashMap::new();
        for (name, batches) in &self.tables {
            let rows = batches
                .iter()
                .map(RecordBatch::num_rows)
                .fold(0usize, |a, n| a.saturating_add(n));
            row_counts.insert(name.clone(), rows);
        }
        for (name, src) in self.streaming_sources.borrow().iter() {
            if let crate::exec::streaming::TableSource::Materialized(batches) = src {
                let rows = batches
                    .iter()
                    .map(RecordBatch::num_rows)
                    .fold(0usize, |a, n| a.saturating_add(n));
                row_counts.entry(name.clone()).or_insert(rows);
            }
        }
        for (name, rows) in self.storage_row_counts() {
            row_counts.entry(name).or_insert(rows);
        }
        let columns = self.planner_column_statistics();
        Arc::new(crate::plan::StatsEstimator::new(EngineTableStats {
            row_counts,
            columns,
        }))
    }

    /// Exactly-once accounting wrapper for a *top-level* query.
    ///
    /// Bumps `QueriesTotal` before running `f` and `QueriesFailed` iff `f`
    /// fails — at ANY stage: classification, parse, rewrite, optimize,
    /// subquery resolution, lowering, or execution — then performs the
    /// throttled pool-stats emit. `sql` and `run_logical_plan` are the only
    /// callers; nested subplans deliberately do not count (a statement with N
    /// subqueries is one query).
    fn run_top_level(
        &self,
        f: impl FnOnce() -> BoltResult<QueryHandle>,
    ) -> BoltResult<QueryHandle> {
        // Retry the pool-watcher's context capture (cheap atomic load once
        // captured; a no-op when no context is bound on this thread).
        crate::cuda::mem_pool::pool_watcher_retry_context_capture();
        crate::metrics::metrics().inc(crate::metrics::Counter::QueriesTotal);
        let result = f();
        if result.is_err() {
            crate::metrics::metrics().inc(crate::metrics::Counter::QueriesFailed);
        }
        // Runs whether the query succeeded or failed (an OOM-failed query is
        // itself a signal worth surfacing alongside the pool snapshot); never
        // escalates into the query result.
        self.maybe_emit_pool_stats(Instant::now());
        result
    }

    /// Compile one logical plan through the shared pipeline and execute it.
    ///
    /// `user_rewrites` — see [`QueryCompiler::compile`]. `top_level` controls
    /// whether the `Plan` / `Lower` / `Execute` phase histograms are observed
    /// (nested subplans are folded into their enclosing query's phases).
    pub(crate) fn execute_logical(
        &self,
        plan: &LogicalPlan,
        user_rewrites: bool,
        top_level: bool,
    ) -> BoltResult<QueryHandle> {
        if top_level {
            if let Some(h) = self.timed_ooc(plan)? {
                return Ok(h);
            }
            if let Some(h) = self.timed_host_tier(plan)? {
                return Ok(h);
            }
        }
        let phys = self.prepare_physical(plan, user_rewrites, top_level)?;
        if top_level {
            self.timed_execute(|| self.execute(&phys))
        } else {
            self.execute(&phys)
        }
    }

    /// Out-of-core execution of a top-level plan, if it needs it (see
    /// `engine_ooc`), timed as `Execute`.
    fn timed_ooc(&self, plan: &LogicalPlan) -> BoltResult<Option<QueryHandle>> {
        if self.memory_budget_bytes().is_none() && self.host_memory_budget_bytes().is_none() {
            return Ok(None);
        }
        let start = Instant::now();
        let result = self.try_out_of_core(plan);
        if matches!(result, Ok(Some(_)) | Err(_)) {
            crate::metrics::metrics()
                .observe_duration(crate::metrics::Phase::Execute, start.elapsed());
        }
        result
    }

    /// Host-tier execution of a top-level plan when the cost model prefers
    /// it (see `crate::exec::cost`), timed as `Execute`.
    fn timed_host_tier(&self, plan: &LogicalPlan) -> BoltResult<Option<QueryHandle>> {
        if self.tier_policy != crate::exec::cost::TierPolicy::CostBased {
            return Ok(None);
        }
        let start = Instant::now();
        let result = self.try_host_tier(plan, false);
        if matches!(result, Ok(Some(_)) | Err(_)) {
            crate::metrics::metrics()
                .observe_duration(crate::metrics::Phase::Execute, start.elapsed());
        }
        result
    }

    /// Run a top-level statement's execution step, feeding the `Execute`
    /// phase histogram (observed on failure too).
    fn timed_execute(
        &self,
        f: impl FnOnce() -> BoltResult<QueryHandle>,
    ) -> BoltResult<QueryHandle> {
        let start = Instant::now();
        let result = f();
        crate::metrics::metrics().observe_duration(crate::metrics::Phase::Execute, start.elapsed());
        result
    }

    /// Everything [`Engine::execute_logical`] does before execution: drain the
    /// lazy tables the statement reads, compile, lower, populate validity.
    fn prepare_physical(
        &self,
        plan: &LogicalPlan,
        user_rewrites: bool,
        top_level: bool,
    ) -> BoltResult<PhysicalPlan> {
        let plan_start = Instant::now();
        let mut resolve = |subplan: LogicalPlan| -> BoltResult<RecordBatch> {
            // A subquery was already seen by the user hooks as part of the
            // enclosing plan, so it is compiled without them here.
            self.execute_logical(&subplan, false, false)
                .map(QueryHandle::into_record_batch)
        };
        // Drain (and build dictionaries for) every lazy table the statement
        // reads — including inside its subqueries — BEFORE compiling, so the
        // dictionary rewrite already sees their dictionaries and no drain has
        // to mutate the registry while a compile holds it.
        let (tree_tables, nested_tables) = split_scan_tables(plan);
        if top_level {
            // Between statements, nothing holds a device table: evict the
            // least recently used resident tables this one does not read
            // until the device budget has room.
            let keep: Vec<String> = tree_tables.iter().chain(&nested_tables).cloned().collect();
            self.make_device_room(&keep);
        }
        // Storage tables: load this statement's pruned scan (or reuse one that
        // covers it) — also before compiling, for the same reason.
        self.scan_storage_tables(plan, &tree_tables, &nested_tables, top_level)?;
        for table in tree_tables.iter().chain(&nested_tables) {
            self.materialize_streaming_source(table)?;
        }
        let registry = self.dict_registry.borrow();
        let compiled = self.compiler(&registry).compile(
            plan,
            user_rewrites,
            Subqueries::Resolve(&mut resolve),
        );
        drop(registry);
        let compiled = compiled?;
        if top_level {
            crate::metrics::metrics()
                .observe_duration(crate::metrics::Phase::Plan, plan_start.elapsed());
        }
        let lower_start = Instant::now();
        let phys = self.lower_for_execution(&compiled.plan)?;
        if top_level {
            crate::metrics::metrics()
                .observe_duration(crate::metrics::Phase::Lower, lower_start.elapsed());
        }
        Ok(phys)
    }

    /// Lower a compiled logical plan and prepare it for [`Engine::execute`]:
    /// materialise exactly the lazily-registered sources the plan scans, then
    /// populate each kernel's `input_has_validity` from the registered data.
    fn lower_for_execution(&self, plan: &LogicalPlan) -> BoltResult<PhysicalPlan> {
        let mut phys = crate::plan::lower_physical(plan)?;
        // Only the tables this plan reads are drained — an unrelated lazy
        // source keeps its producer un-invoked (the `register_table_stream_lazy`
        // contract).
        self.ensure_streaming_materialized_for(plan)?;
        // PV-stage-d: the plan-time null signal that lets codegen emit
        // native-validity kernels instead of the run-time host-strip fallback.
        let nb = EngineProvider {
            base: &self.provider,
            tables: &self.tables,
            streaming: self.streaming_sources.borrow(),
        };
        crate::plan::physical_plan::populate_input_validity(&mut phys, &nb);
        // `nb` holds an overlay borrow; it drops here, before `execute`
        // (whose read paths may re-borrow the overlay).
        drop(nb);
        Ok(phys)
    }

    /// Compile and execute a SQL query string.
    ///
    /// The statement is classified once (`sql_frontend::classify_query`); an
    /// ordinary query runs through the shared compiler pipeline (see the
    /// module docs), and a host-orchestrated shape (recursive CTE, LATERAL,
    /// correlated WHERE, VALUES, generate_series, DISTINCT ON, COUNT(DISTINCT)
    /// GROUP BY) runs its descriptor, whose component subplans go through
    /// that same pipeline — user [`PlanRewrite`]s included.
    ///
    /// `QueriesTotal` counts the statement once and `QueriesFailed` counts
    /// any failure, whatever stage it comes from. After the query completes,
    /// the engine emits a periodic pool-stats log line at most once every
    /// `BOLT_POOL_STATS_INTERVAL_SECS` (default 60s), off the latency path of
    /// the returned result.
    pub fn sql(&self, query: &str) -> BoltResult<QueryHandle> {
        self.run_top_level(|| self.sql_inner(query))
    }

    fn sql_inner(&self, query: &str) -> BoltResult<QueryHandle> {
        reject_table_statement(query)?;
        // User hooks may be stateful or nondeterministic, so their presence
        // disables plan reuse (the contract runs them on every compile).
        let cacheable = self.rewrites.is_empty();
        if cacheable {
            let key = self.compiled_plan_key(query);
            let cached = self.compiled_plans.borrow_mut().get(&key);
            if let Some(phys) = cached {
                return self.timed_execute(|| self.execute(&phys));
            }
        }
        let parse_start = Instant::now();
        let shape = classify_query(query, &self.provider)?;
        crate::metrics::metrics()
            .observe_duration(crate::metrics::Phase::Parse, parse_start.elapsed());
        match shape {
            QueryShape::Generic(plan) => {
                // A fresh materialized view of this very query.
                if let Some(h) = self.answer_from_view(&plan)? {
                    return Ok(h);
                }
                // Out-of-core results depend on the budget path taken and
                // are not cached.
                if let Some(h) = self.timed_ooc(&plan)? {
                    return Ok(h);
                }
                // Neither are host-tier ones (the GPU plan is cached once the
                // cost model picks the GPU).
                if let Some(h) = self.timed_host_tier(&plan)? {
                    return Ok(h);
                }
                let phys = self.prepare_physical(&plan, true, true)?;
                if cacheable {
                    // Keyed AFTER preparing: a first query may drain lazy
                    // tables, which bumps the epoch the plan derives from.
                    let key = self.compiled_plan_key(query);
                    self.compiled_plans.borrow_mut().insert(key, phys.clone());
                }
                self.timed_execute(|| self.execute(&phys))
            }
            // Host-orchestrated shapes: their component subplans compile and
            // run inside the descriptor, so `Execute` covers both.
            shape => self.timed_execute(|| self.execute_shape(shape)),
        }
    }

    /// Run a host-orchestrated [`QueryShape`] descriptor.
    fn execute_shape(&self, shape: QueryShape) -> BoltResult<QueryHandle> {
        match shape {
            QueryShape::Generic(plan) => self.execute_logical(&plan, true, false),
            QueryShape::Recursive(RecursiveQueryPlan::Single(rec)) => {
                self.execute_recursive_cte(&rec)
            }
            QueryShape::Recursive(RecursiveQueryPlan::Mutual(rec)) => {
                self.execute_mutual_recursive_cte(&rec)
            }
            QueryShape::Lateral(la) => self.execute_lateral_apply(&la),
            QueryShape::CorrelatedWhere(cw) => self.execute_correlated_where(&cw),
            QueryShape::Values(vp) => self.execute_values_query(&vp),
            QueryShape::GenerateSeries(gp) => self.execute_generate_series_query(&gp),
            QueryShape::DistinctOn(dp) => self.execute_distinct_on(&dp),
            QueryShape::CountDistinctGroupBy(cd) => self.execute_count_distinct_groupby(&cd),
            QueryShape::MultiAggGroupBy(cd) => self.execute_multi_agg_groupby(&cd),
        }
    }

    /// Parse, plan, and *render* `query` without executing it — an
    /// `EXPLAIN`-style introspection hook.
    ///
    /// An ordinary query is compiled through exactly the pipeline
    /// [`Engine::sql`] uses — dictionary rewrite, built-in optimizer, user
    /// [`PlanRewrite`]s — so the rendered logical and physical plans are the
    /// ones that would execute. Two deliberate differences, both visible in
    /// the output:
    ///
    /// * Uncorrelated subqueries are NOT executed (explaining must not run
    ///   queries). When the plan carries any, the physical section is headed
    ///   `Physical plan (pre-subquery-resolution)` instead of silently
    ///   presenting a plan that differs from the executed one.
    /// * Lazily-registered stream producers are never invoked.
    ///
    /// Plans the GPU codegen cannot lower still return their logical plan,
    /// with the lowering error noted in place of the physical section. A
    /// host-orchestrated shape renders its descriptor; its component
    /// subplans are shown as planned, and each is compiled through the full
    /// pipeline when executed.
    pub fn explain_sql(&self, query: &str) -> BoltResult<String> {
        use crate::plan::explain as fmt;
        const COMPONENTS_NOTE: &str = "(host-orchestrated: each component subplan below is \
             compiled through the full rewrite/optimize/lower pipeline when executed)\n";
        let special = |title: &str, body: String| format!("{title}:\n{COMPONENTS_NOTE}{body}");
        let plan = match classify_query(query, &self.provider)? {
            QueryShape::Generic(plan) => plan,
            QueryShape::Recursive(RecursiveQueryPlan::Single(rec)) => {
                return Ok(special(
                    "Recursive CTE plan",
                    fmt::format_recursive_cte(&rec),
                ))
            }
            QueryShape::Recursive(RecursiveQueryPlan::Mutual(rec)) => {
                return Ok(special(
                    "Mutual recursive CTE plan",
                    fmt::format_mutual_recursive_cte(&rec),
                ))
            }
            QueryShape::Lateral(la) => {
                return Ok(special(
                    "LATERAL apply plan",
                    fmt::format_lateral_apply(&la),
                ))
            }
            QueryShape::CorrelatedWhere(cw) => {
                return Ok(special(
                    "Correlated WHERE plan",
                    fmt::format_correlated_where(&cw),
                ))
            }
            QueryShape::Values(vp) => {
                return Ok(special("VALUES plan", fmt::format_values_query(&vp)))
            }
            QueryShape::GenerateSeries(gp) => {
                return Ok(special(
                    "generate_series plan",
                    fmt::format_generate_series_query(&gp),
                ))
            }
            QueryShape::DistinctOn(dp) => {
                return Ok(special("DISTINCT ON plan", fmt::format_distinct_on(&dp)))
            }
            QueryShape::CountDistinctGroupBy(cd) => {
                return Ok(special(
                    "COUNT(DISTINCT) GROUP BY plan",
                    fmt::format_count_distinct_groupby(&cd),
                ))
            }
            QueryShape::MultiAggGroupBy(cd) => {
                return Ok(special(
                    "multi-agg GROUP BY plan",
                    fmt::format_multi_agg_groupby(&cd),
                ))
            }
        };
        let registry = self.dict_registry.borrow();
        let compiled = self
            .compiler(&registry)
            .compile(&plan, true, Subqueries::Defer)?;
        drop(registry);
        let mut out = render_explain(&compiled);
        out.push_str(&self.explain_resources(&plan)?);
        Ok(out)
    }

    /// The `EXPLAIN` sections about data access and memory: each storage
    /// table's pruned scan (decided from statistics; nothing is read) and
    /// whether the statement would run out of core.
    fn explain_resources(&self, plan: &LogicalPlan) -> BoltResult<String> {
        let mut out = String::new();
        let (tree, nested) = split_scan_tables(plan);
        let mut tables: Vec<String> = tree
            .iter()
            .chain(&nested)
            .filter(|t| self.storage_table(t).is_some())
            .cloned()
            .collect();
        tables.sort();
        tables.dedup();
        if !tables.is_empty() {
            let requests = self.storage_scan_requests(plan, &tables, &nested);
            out.push_str(
                "Storage scans:
",
            );
            for t in &tables {
                let table = self.storage_table(t).expect("filtered above");
                let req = requests.get(t).cloned().unwrap_or_default();
                let sel = table.plan_scan(&req);
                let cols = match &sel.columns {
                    Some(c) => format!("{} of {} columns", c.len(), table.schema().fields.len()),
                    None => "all columns".to_string(),
                };
                out.push_str(&format!(
                    "  {t}: {} of {} chunks (~{} rows), {cols}
",
                    sel.chunks.len(),
                    sel.chunks_total,
                    sel.rows
                ));
            }
        }
        if let Some(v) = self.matching_view(plan) {
            out.push_str(&format!(
                "Answered from materialized view '{v}' (fresh)
"
            ));
        }
        if let Some(t) = self.explain_tier(plan) {
            out.push_str(&t);
            out.push('\n');
        }
        if let Some(d) = self.describe_out_of_core(plan)? {
            out.push_str(&format!(
                "Execution: out of core — {d}
"
            ));
        }
        Ok(out)
    }

    /// Execute an already-built [`LogicalPlan`] and return a [`QueryHandle`].
    ///
    /// The post-parse half of [`Engine::sql`]: it skips SQL parsing (so the
    /// input plan can come from the [`DataFrame`](crate::DataFrame) builder,
    /// a test fixture, etc.) but runs the identical pipeline — dictionary
    /// rewrite → optimizer → user [`PlanRewrite`]s → subquery resolution →
    /// lower → lazy-source materialisation → validity propagation → execute
    /// — with the same `QueriesTotal` / `QueriesFailed` accounting and
    /// pool-stats emit.
    ///
    /// `&mut self` matches the [`DataFrame::collect`](crate::DataFrame::collect)
    /// signature; the engine state mutated here is bounded to the
    /// pool-stats throttle and the interior-mutable caches `sql()` touches.
    pub fn run_logical_plan(&mut self, plan: &LogicalPlan) -> BoltResult<QueryHandle> {
        self.run_top_level(|| self.execute_logical(plan, true, true))
    }

    /// Execute a self-contained component subplan of a host-orchestrated
    /// query shape and return its result batch.
    ///
    /// Runs the full pipeline (user hooks included — the enclosing statement
    /// never had a single plan for them to see). Does NOT bump
    /// `QueriesTotal` / `QueriesFailed`: the top-level `sql()` call already
    /// counts the statement once, and a failure here surfaces as that
    /// statement's failure.
    pub(crate) fn run_subplan(&self, plan: LogicalPlan) -> BoltResult<RecordBatch> {
        self.execute_logical(&plan, true, false)
            .map(QueryHandle::into_record_batch)
    }
}

/// Bound on cached compiled plans per engine (FIFO eviction).
const COMPILED_PLAN_CACHE_CAP: usize = 64;

/// `(SQL text, provider schema version, data epoch)`.
pub(crate) type CompiledPlanKey = (String, u64, u64);

/// Per-engine cache of fully prepared physical plans for repeated top-level
/// ordinary SQL. A hit skips classification, the dictionary rewrite, the
/// optimizer, subquery execution and lowering; the key's data epoch makes
/// any data or schema change a miss (see the engine's `data_epoch`).
#[derive(Default)]
pub(crate) struct CompiledPlanCache {
    map: std::collections::HashMap<CompiledPlanKey, PhysicalPlan>,
    order: std::collections::VecDeque<CompiledPlanKey>,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
}

impl CompiledPlanCache {
    /// Forget every cached plan (the tier decision changed).
    pub(crate) fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    fn get(&mut self, key: &CompiledPlanKey) -> Option<PhysicalPlan> {
        match self.map.get(key) {
            Some(p) => {
                self.hits += 1;
                Some(p.clone())
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    fn insert(&mut self, key: CompiledPlanKey, plan: PhysicalPlan) {
        if self.map.contains_key(&key) {
            return;
        }
        while self.map.len() >= COMPILED_PLAN_CACHE_CAP {
            match self.order.pop_front() {
                Some(old) => {
                    self.map.remove(&old);
                }
                None => break,
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, plan);
    }
}

impl Engine {
    fn compiled_plan_key(&self, sql: &str) -> CompiledPlanKey {
        (
            sql.to_string(),
            crate::plan::TableProvider::schema_version(&self.provider),
            self.data_epoch.get(),
        )
    }
}

/// Render a compiled plan for [`Engine::explain_sql`].
fn render_explain(compiled: &CompiledLogical) -> String {
    let mut out = String::from("Logical plan:\n");
    out.push_str(&crate::plan::format_logical(&compiled.plan));
    if compiled.deferred_subqueries {
        out.push_str(
            "\nPhysical plan (pre-subquery-resolution; at run time each uncorrelated \
             subquery is executed and folded to a constant before lowering):\n",
        );
    } else {
        out.push_str("\nPhysical plan:\n");
    }
    match crate::plan::lower_physical(&compiled.plan) {
        Ok(phys) => out.push_str(&crate::plan::format_physical(&phys)),
        Err(e) => out.push_str(&format!("  <not lowered: {e}>\n")),
    }
    out
}

#[cfg(test)]
mod tests {
    //! Host-only coverage of the shared compile pipeline. [`QueryCompiler`]
    //! needs no CUDA context, so the stage ordering and the
    //! execute-vs-explain parity contract are pinned here without a GPU; the
    //! engine-level wiring is covered by `tests/query_pipeline_parity_test.rs`.

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use super::*;
    use crate::exec::dict_registry::{rewrite_plan_calls, DictRegistry};
    use crate::plan::{DataType, Field, MemTableProvider, Schema};

    fn provider() -> MemTableProvider {
        MemTableProvider::new()
            .with_table(
                "t",
                Schema::new(vec![
                    Field::new("x", DataType::Int64, false),
                    Field::new("y", DataType::Int64, false),
                ]),
            )
            .with_table(
                "u",
                Schema::new(vec![Field::new("k", DataType::Int64, false)]),
            )
    }

    fn plan(sql: &str) -> LogicalPlan {
        crate::plan::parse_sql(sql, &provider()).expect("parse")
    }

    fn compiler<'a>(
        dict: &'a DictRegistry,
        rewrites: &'a [Box<dyn PlanRewrite>],
        optimize: bool,
    ) -> QueryCompiler<'a> {
        QueryCompiler {
            dict,
            rewrites,
            optimize,
            estimator: Arc::new(crate::plan::StatsEstimator::new(EngineTableStats::default())),
        }
    }

    /// Records every plan it is handed (rendered) and returns it unchanged.
    #[derive(Default)]
    struct Recorder {
        seen: Mutex<Vec<String>>,
        calls: AtomicUsize,
    }

    struct RecorderRef(Arc<Recorder>);

    impl PlanRewrite for RecorderRef {
        fn name(&self) -> &str {
            "recorder"
        }
        fn rewrite(&self, plan: LogicalPlan) -> BoltResult<LogicalPlan> {
            self.0.calls.fetch_add(1, Ordering::SeqCst);
            self.0
                .seen
                .lock()
                .unwrap()
                .push(crate::plan::format_logical(&plan));
            Ok(plan)
        }
    }

    fn no_subqueries() -> impl FnMut(LogicalPlan) -> BoltResult<RecordBatch> {
        |_| panic!("plan has no subqueries; the executor must not be called")
    }

    /// The pipeline runs the dictionary rewrite exactly once per compiled
    /// plan (it used to run twice back-to-back on the `sql()` path).
    #[test]
    fn compile_runs_dictionary_rewrite_once() {
        let dict = DictRegistry::new();
        let c = compiler(&dict, &[], true);
        let p = plan("SELECT x FROM t WHERE y > 3");
        let before = rewrite_plan_calls();
        let mut exec = no_subqueries();
        c.compile(&p, true, Subqueries::Resolve(&mut exec)).unwrap();
        assert_eq!(rewrite_plan_calls() - before, 1);
        let before = rewrite_plan_calls();
        c.compile(&p, true, Subqueries::Defer).unwrap();
        assert_eq!(rewrite_plan_calls() - before, 1);
    }

    /// User rewrites run after the built-in optimizer: the plan a hook sees
    /// is exactly the optimizer output.
    #[test]
    fn user_rewrites_see_optimized_plan() {
        let dict = DictRegistry::new();
        let rec = Arc::new(Recorder::default());
        let hooks: Vec<Box<dyn PlanRewrite>> = vec![Box::new(RecorderRef(Arc::clone(&rec)))];
        let p = plan("SELECT x FROM t WHERE 1 = 1 AND y > 3");

        let optimized_only = compiler(&dict, &[], true)
            .compile(&p, true, Subqueries::Defer)
            .unwrap();
        compiler(&dict, &hooks, true)
            .compile(&p, true, Subqueries::Defer)
            .unwrap();
        let seen = rec.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0], crate::plan::format_logical(&optimized_only.plan));
        assert_ne!(
            seen[0],
            crate::plan::format_logical(&p),
            "the constant conjunct must already be folded when the hook runs"
        );
    }

    /// `user_rewrites = false` (the extracted-subquery case) skips the hooks.
    #[test]
    fn user_rewrites_can_be_suppressed_for_extracted_subqueries() {
        let dict = DictRegistry::new();
        let rec = Arc::new(Recorder::default());
        let hooks: Vec<Box<dyn PlanRewrite>> = vec![Box::new(RecorderRef(Arc::clone(&rec)))];
        let c = compiler(&dict, &hooks, true);
        c.compile(&plan("SELECT x FROM t"), false, Subqueries::Defer)
            .unwrap();
        assert_eq!(rec.calls.load(Ordering::SeqCst), 0);
        c.compile(&plan("SELECT x FROM t"), true, Subqueries::Defer)
            .unwrap();
        assert_eq!(rec.calls.load(Ordering::SeqCst), 1);
    }

    /// EXPLAIN parity: for a subquery-free plan the explain mode (`Defer`)
    /// and the execution mode (`Resolve`) produce the identical plan.
    #[test]
    fn explain_and_execute_compile_identically_without_subqueries() {
        let dict = DictRegistry::new();
        let c = compiler(&dict, &[], true);
        for sql in [
            "SELECT x FROM t WHERE 1 = 1 AND y > 3",
            "SELECT x, SUM(y) FROM t GROUP BY x",
            "SELECT t.x FROM t JOIN u ON t.x = u.k WHERE t.y > 1",
        ] {
            let p = plan(sql);
            let mut exec = no_subqueries();
            let executed = c.compile(&p, true, Subqueries::Resolve(&mut exec)).unwrap();
            let explained = c.compile(&p, true, Subqueries::Defer).unwrap();
            assert!(!explained.deferred_subqueries, "{sql}");
            assert_eq!(
                crate::plan::format_logical(&executed.plan),
                crate::plan::format_logical(&explained.plan),
                "{sql}"
            );
        }
    }

    /// With a subquery, `Defer` leaves it in place and says so, while
    /// `Resolve` hands it to the executor and folds the result.
    #[test]
    fn subqueries_are_deferred_for_explain_and_resolved_for_execution() {
        let dict = DictRegistry::new();
        let c = compiler(&dict, &[], true);
        let p = plan("SELECT x FROM t WHERE y > (SELECT MAX(k) FROM u)");

        let explained = c.compile(&p, true, Subqueries::Defer).unwrap();
        assert!(explained.deferred_subqueries);
        assert!(render_explain(&explained).contains("pre-subquery-resolution"));

        let calls = AtomicUsize::new(0);
        let mut exec = |_: LogicalPlan| -> BoltResult<RecordBatch> {
            calls.fetch_add(1, Ordering::SeqCst);
            let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                "m",
                arrow_schema::DataType::Int64,
                true,
            )]));
            Ok(RecordBatch::try_new(
                schema,
                vec![Arc::new(arrow_array::Int64Array::from(vec![Some(7)]))],
            )
            .unwrap())
        };
        let executed = c.compile(&p, true, Subqueries::Resolve(&mut exec)).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(!executed.deferred_subqueries);
        assert!(!plan_has_subqueries(&executed.plan));
        assert!(!render_explain(&executed).contains("pre-subquery-resolution"));
    }

    /// A failing hook aborts compilation with its error.
    #[test]
    fn failing_user_rewrite_aborts_compile() {
        struct Fail;
        impl PlanRewrite for Fail {
            fn name(&self) -> &str {
                "fail"
            }
            fn rewrite(&self, _: LogicalPlan) -> BoltResult<LogicalPlan> {
                Err(BoltError::Plan("rewrite refused".into()))
            }
        }
        let dict = DictRegistry::new();
        let hooks: Vec<Box<dyn PlanRewrite>> = vec![Box::new(Fail)];
        let err = compiler(&dict, &hooks, true)
            .compile(&plan("SELECT x FROM t"), true, Subqueries::Defer)
            .unwrap_err();
        assert!(err.to_string().contains("rewrite refused"), "{err}");
    }
}
