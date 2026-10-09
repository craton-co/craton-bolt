// SPDX-License-Identifier: Apache-2.0
//! Out-of-core statement execution.
//!
//! When a statement reads a table larger than the memory budget
//! ([`Engine::memory_budget_bytes`], [`Engine::host_memory_budget_bytes`])
//! and its shape is not one the morsel streaming path already bounds, the
//! engine executes it in **hash partitions spilled to disk**, running the
//! ordinary (GPU) operators on one bounded partition at a time:
//!
//! * **GROUP BY / DISTINCT** — the large table is hash-partitioned on the
//!   grouping columns (through filters, projections and joins on which it is
//!   the preserved side). Groups never span partitions, so the per-partition
//!   results — for *any* aggregate, `COUNT(DISTINCT)` included — concatenate
//!   to the exact answer.
//! * **Equi-joins** (grace hash join) — both inputs are partitioned on their
//!   join keys with one type-normalised hash, so matching rows meet in the
//!   same partition pair; NULL keys share partition 0 and never match, so
//!   outer-join padding stays exact.
//! * **ORDER BY [LIMIT]** over a row-wise scan — sorted runs of bounded size
//!   (top-k runs under a LIMIT), merged k-way on the host.
//!
//! The executed node is replaced by a scan of its (host-resident) result and
//! the rest of the statement runs normally, so e.g. an aggregate over a
//! grace-joined result can itself go out of core. Partition inputs live in
//! Arrow IPC spill files under the spill directory; at most one partition
//! is in memory at a time. A partition that is still over budget (key skew)
//! is executed anyway.

use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::sync::Arc;

use arrow::compute::interleave;
use arrow::row::{RowConverter, SortField};
use arrow_array::RecordBatch;
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema, SchemaRef, SortOptions};

use crate::error::{BoltError, BoltResult};
use crate::exec::engine::{Engine, QueryHandle};
use crate::exec::spill::{key_family, partition_hash, split_by_partition, SpillFile, SpillWriter};
use crate::exec::streaming::TableSource;
use crate::plan::logical_plan::{AggregateExpr, Expr, JoinType, LogicalPlan, Schema};
use crate::storage::ScanRequest;

/// Rows per batch when slicing in-memory tables for partitioning.
const SLICE_ROWS: usize = 64 * 1024;
const MAX_PARTITIONS: usize = 256;

/// Ephemeral overlay tables of one out-of-core statement, removed on drop.
struct Ephemeral<'a> {
    engine: &'a Engine,
    names: RefCell<Vec<String>>,
}

impl<'a> Ephemeral<'a> {
    fn new(engine: &'a Engine) -> Self {
        Ephemeral {
            engine,
            names: RefCell::new(Vec::new()),
        }
    }

    fn install(&self, batch: RecordBatch) -> String {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let name = format!(
            "__ooc_{}",
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        self.engine.gpu_tables.borrow_mut().remove(&name);
        self.engine
            .streaming_sources
            .borrow_mut()
            .insert(name.clone(), TableSource::Materialized(vec![batch]));
        self.names.borrow_mut().push(name.clone());
        name
    }

    fn release(&self, name: &str) {
        self.engine.streaming_sources.borrow_mut().remove(name);
        self.engine.gpu_tables.borrow_mut().remove(name);
        self.engine.overlay_uploads.borrow_mut().remove(name);
        self.names.borrow_mut().retain(|n| n != name);
    }
}

impl Drop for Ephemeral<'_> {
    fn drop(&mut self) {
        for name in self.names.borrow_mut().drain(..) {
            self.engine.streaming_sources.borrow_mut().remove(&name);
            self.engine.gpu_tables.borrow_mut().remove(&name);
            self.engine.overlay_uploads.borrow_mut().remove(&name);
        }
    }
}

/// Rebuild `plan` with new children (same order as [`LogicalPlan::children`]).
fn with_children(plan: &LogicalPlan, mut children: Vec<LogicalPlan>) -> LogicalPlan {
    let mut next = || Box::new(children.remove(0));
    match plan {
        LogicalPlan::Scan { .. } => plan.clone(),
        LogicalPlan::Filter { predicate, .. } => LogicalPlan::Filter {
            input: next(),
            predicate: predicate.clone(),
        },
        LogicalPlan::Project { exprs, .. } => LogicalPlan::Project {
            input: next(),
            exprs: exprs.clone(),
        },
        LogicalPlan::Aggregate {
            group_by,
            aggregates,
            ..
        } => LogicalPlan::Aggregate {
            input: next(),
            group_by: group_by.clone(),
            aggregates: aggregates.clone(),
        },
        LogicalPlan::Distinct { .. } => LogicalPlan::Distinct { input: next() },
        LogicalPlan::Limit { limit, offset, .. } => LogicalPlan::Limit {
            input: next(),
            limit: *limit,
            offset: *offset,
        },
        LogicalPlan::Sort { sort_exprs, .. } => LogicalPlan::Sort {
            input: next(),
            sort_exprs: sort_exprs.clone(),
        },
        LogicalPlan::Window {
            window_exprs,
            partition_by,
            order_by,
            ..
        } => LogicalPlan::Window {
            input: next(),
            window_exprs: window_exprs.clone(),
            partition_by: partition_by.clone(),
            order_by: order_by.clone(),
        },
        LogicalPlan::Union { .. } => LogicalPlan::Union { inputs: children },
        LogicalPlan::SetOp { op, all, .. } => {
            let left = next();
            let right = next();
            LogicalPlan::SetOp {
                left,
                right,
                op: *op,
                all: *all,
            }
        }
        LogicalPlan::Join {
            join_type,
            on,
            filter,
            ..
        } => {
            let left = next();
            let right = next();
            LogicalPlan::Join {
                left,
                right,
                join_type: *join_type,
                on: on.clone(),
                filter: filter.clone(),
            }
        }
    }
}

/// Point every scan of `from` in `plan` at `to` (whose columns are `from`'s
/// base columns).
fn rename_scans(plan: &LogicalPlan, from: &str, to: &str, base: &Schema) -> LogicalPlan {
    match plan {
        LogicalPlan::Scan {
            table, projection, ..
        } if table == from => LogicalPlan::Scan {
            table: to.to_string(),
            projection: projection.clone(),
            schema: base.clone(),
        },
        other => {
            let kids = other
                .children()
                .into_iter()
                .map(|c| rename_scans(c, from, to, base))
                .collect();
            with_children(other, kids)
        }
    }
}

fn count_scans(plan: &LogicalPlan, table: &str) -> usize {
    crate::plan::logical_plan::collect_scan_tables(plan)
        .iter()
        .filter(|t| t.as_str() == table)
        .count()
}

/// Column name of a key expression (`col` or `col AS x`).
fn column_name(e: &Expr) -> Option<&str> {
    match e {
        Expr::Column(c) => Some(c),
        Expr::Alias(inner, _) => column_name(inner),
        _ => None,
    }
}

fn scan_column(schema: &Schema, table: &str, col: &str) -> Option<String> {
    if schema.index_of(col).is_ok() {
        return Some(col.to_string());
    }
    let (q, c) = col.rsplit_once('.')?;
    (q == table || q.ends_with(&format!(".{table}")))
        .then(|| schema.index_of(c).ok().map(|_| c.to_string()))
        .flatten()
}

/// Trace output column `col` of `plan` back to a base-table column through
/// filters, pass-through projections and (when `through_joins`) joins on
/// which that table is a preserved side.
fn resolve_base_column(
    plan: &LogicalPlan,
    col: &str,
    through_joins: bool,
) -> Option<(String, String)> {
    match plan {
        LogicalPlan::Scan { table, schema, .. } => {
            scan_column(schema, table, col).map(|c| (table.clone(), c))
        }
        LogicalPlan::Filter { input, .. } => resolve_base_column(input, col, through_joins),
        LogicalPlan::Project { input, exprs } => {
            let schema = plan.schema().ok()?;
            let i = schema.index_of(col).ok().or_else(|| {
                schema
                    .fields
                    .iter()
                    .position(|f| f.name.rsplit('.').next() == Some(col))
            })?;
            resolve_base_column(input, column_name(exprs.get(i)?)?, through_joins)
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            ..
        } if through_joins => {
            if let Some(r) = resolve_base_column(left, col, true) {
                return matches!(join_type, JoinType::Inner | JoinType::LeftOuter).then_some(r);
            }
            if let Some(r) = resolve_base_column(right, col, true) {
                return matches!(join_type, JoinType::Inner | JoinType::RightOuter).then_some(r);
            }
            None
        }
        _ => None,
    }
}

/// The one base table of a row-wise chain (filters / projections over a
/// single scan).
fn single_base_table(plan: &LogicalPlan) -> Option<&str> {
    match plan {
        LogicalPlan::Scan { table, .. } => Some(table),
        LogicalPlan::Filter { input, .. } | LogicalPlan::Project { input, .. } => {
            single_base_table(input)
        }
        _ => None,
    }
}

fn contains_join(plan: &LogicalPlan) -> bool {
    matches!(plan, LogicalPlan::Join { .. }) || plan.children().into_iter().any(contains_join)
}

/// Aggregates the morsel streaming path folds exactly.
fn streamable(a: &AggregateExpr) -> bool {
    matches!(
        a,
        AggregateExpr::Count(_)
            | AggregateExpr::Sum(_)
            | AggregateExpr::Min(_)
            | AggregateExpr::Max(_)
            | AggregateExpr::Avg(_)
            | AggregateExpr::VarPop(_)
            | AggregateExpr::VarSamp(_)
            | AggregateExpr::StddevPop(_)
            | AggregateExpr::StddevSamp(_)
    )
}

fn partitions_for(bytes: usize, budget: usize) -> usize {
    let per = (budget / 2).max(1);
    bytes.div_ceil(per).clamp(2, MAX_PARTITIONS)
}

/// Rename `batch`'s columns to `names` (same count).
fn renamed(batch: RecordBatch, schema: &Schema) -> BoltResult<RecordBatch> {
    if batch.num_columns() != schema.fields.len() {
        return Err(BoltError::Other(format!(
            "out-of-core: partition result has {} columns, plan expects {}",
            batch.num_columns(),
            schema.fields.len()
        )));
    }
    let fields: Vec<ArrowField> = batch
        .schema()
        .fields()
        .iter()
        .zip(&schema.fields)
        .map(|(f, want)| f.as_ref().clone().with_name(want.name.clone()))
        .collect();
    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), batch.columns().to_vec())
        .map_err(|e| BoltError::Other(format!("out-of-core: {e}")))
}

fn concat(schema: Option<SchemaRef>, batches: &[RecordBatch]) -> BoltResult<Option<RecordBatch>> {
    let Some(schema) = schema.or_else(|| batches.first().map(RecordBatch::schema)) else {
        return Ok(None);
    };
    arrow::compute::concat_batches(&schema, batches)
        .map(Some)
        .map_err(|e| BoltError::Other(format!("out-of-core: concatenating results: {e}")))
}

/// A qualifying partitioned GROUP BY / DISTINCT.
struct GroupPlan {
    table: String,
    keys: Vec<String>,
    est: usize,
}

/// A qualifying grace hash join.
struct JoinPlan {
    a: String,
    b: String,
    keys_a: Vec<String>,
    keys_b: Vec<String>,
    est: usize,
}

/// A qualifying external sort.
struct SortPlan<'p> {
    limit: Option<(usize, usize)>,
    input: &'p LogicalPlan,
    sort_exprs: &'p [crate::plan::logical_plan::SortExpr],
    table: String,
    est: usize,
    out_schema: Schema,
    keys: Vec<(usize, SortOptions)>,
    run_rows: usize,
}

impl Engine {
    /// Effective out-of-core threshold: the smaller configured budget.
    fn ooc_budget(&self) -> Option<usize> {
        match (self.memory_budget_bytes(), self.host_memory_budget_bytes()) {
            (Some(d), Some(h)) => Some(d.min(h)),
            (d, h) => d.or(h),
        }
    }

    /// Estimated in-memory size of table `name`, without reading data.
    pub(crate) fn estimated_table_bytes(&self, name: &str) -> Option<usize> {
        use crate::exec::streaming::estimate_batches_bytes;
        if let Some(b) = self.tables.get(name) {
            return Some(estimate_batches_bytes(b));
        }
        if let Some(t) = self.storage_table(name) {
            let width: usize = t
                .schema()
                .fields
                .iter()
                .map(|f| f.dtype.byte_width().unwrap_or(32) + 1)
                .sum();
            return Some((t.statistics().row_count as usize).saturating_mul(width));
        }
        match self.streaming_sources.borrow().get(name) {
            Some(TableSource::Materialized(b)) => Some(estimate_batches_bytes(b)),
            _ => None,
        }
    }

    /// Directory spill files are written to.
    pub(crate) fn spill_path(&self) -> std::path::PathBuf {
        self.spill_dir.clone().unwrap_or_else(|| {
            std::env::temp_dir().join(format!("bolt-spill-{}", std::process::id()))
        })
    }

    /// Execute `plan` out of core if it needs and supports it; `None` leaves
    /// it to the ordinary path.
    pub(crate) fn try_out_of_core(&self, plan: &LogicalPlan) -> BoltResult<Option<QueryHandle>> {
        let Some(budget) = self.ooc_budget() else {
            return Ok(None);
        };
        let eph = Ephemeral::new(self);
        if let Some(h) = self.try_external_sort(plan, budget, &eph)? {
            self.memory.record_out_of_core();
            return Ok(Some(h));
        }
        let mut current = plan.clone();
        let mut changed = false;
        while let Some(next) = self.rewrite_one(&current, budget, &eph, 0)? {
            current = next;
            changed = true;
        }
        if !changed {
            return Ok(None);
        }
        self.memory.record_out_of_core();
        let out = self.execute_logical(&current, true, false)?;
        Ok(Some(out))
    }

    /// Execute one out-of-core node of `plan` (aggregates before the joins
    /// beneath them, then joins bottom-up) and return the plan with it
    /// replaced by a scan of its result.
    fn rewrite_one(
        &self,
        plan: &LogicalPlan,
        budget: usize,
        eph: &Ephemeral<'_>,
        depth: usize,
    ) -> BoltResult<Option<LogicalPlan>> {
        if depth > crate::plan::sql_frontend::MAX_RECURSION_DEPTH {
            return Ok(None);
        }
        if matches!(
            plan,
            LogicalPlan::Aggregate { .. } | LogicalPlan::Distinct { .. }
        ) {
            if let Some(scan) = self.ooc_group(plan, budget, eph, depth)? {
                return Ok(Some(scan));
            }
        }
        let children = plan.children();
        for (i, child) in children.iter().enumerate() {
            if let Some(new_child) = self.rewrite_one(child, budget, eph, depth + 1)? {
                let mut kids: Vec<LogicalPlan> = children.iter().map(|c| (*c).clone()).collect();
                kids[i] = new_child;
                return Ok(Some(with_children(plan, kids)));
            }
        }
        if matches!(plan, LogicalPlan::Join { .. }) {
            return self.ooc_join(plan, budget, eph, depth);
        }
        Ok(None)
    }

    /// Replace a node whose executed form has `schema` by a scan of `result`
    /// (renamed to that schema). `schema` comes from the *executed* subplan:
    /// the original's scans may carry rewriter-synthesised columns the
    /// partition tables do not have.
    fn substitute(
        &self,
        schema: Schema,
        result: RecordBatch,
        eph: &Ephemeral<'_>,
    ) -> BoltResult<LogicalPlan> {
        let result = renamed(result, &schema)?;
        let typed = crate::exec::engine_support::arrow_schema_to_plan_schema(&result.schema())?;
        let schema = Schema::new(
            schema
                .fields
                .iter()
                .zip(typed.fields)
                .map(|(want, got)| {
                    crate::plan::logical_plan::Field::new(want.name.clone(), got.dtype, true)
                })
                .collect(),
        );
        let name = eph.install(result);
        Ok(LogicalPlan::Scan {
            table: name,
            projection: None,
            schema,
        })
    }

    /// Partitioned GROUP BY / DISTINCT.
    fn ooc_group(
        &self,
        node: &LogicalPlan,
        budget: usize,
        eph: &Ephemeral<'_>,
        depth: usize,
    ) -> BoltResult<Option<LogicalPlan>> {
        let Some(GroupPlan {
            table,
            keys: key_cols,
            est,
        }) = self.qualify_group(node, budget)?
        else {
            return Ok(None);
        };
        let parts = partitions_for(est, budget);
        let files = self.partition_table(node, &table, &key_cols, parts, 0x5eed + depth as u64)?;
        let base = self.base_schema(&table)?;
        let out_schema = rename_scans(node, &table, &table, &base).schema()?;
        let mut outputs = Vec::new();
        for file in files.into_iter().flatten() {
            let rows = concat(None, &file.read_all()?)?.expect("non-empty spill file");
            drop(file);
            let e = eph.install(rows);
            let sub = rename_scans(node, &table, &e, &base);
            let out = self.execute_logical(&sub, true, false)?.into_record_batch();
            eph.release(&e);
            if out.num_rows() > 0 {
                outputs.push(out);
            }
        }
        let Some(result) = concat(None, &outputs)? else {
            // Every partition was empty or produced nothing: the ordinary
            // path answers an empty input fine.
            return Ok(None);
        };
        Ok(Some(self.substitute(out_schema, result, eph)?))
    }

    /// Grace hash join.
    fn ooc_join(
        &self,
        node: &LogicalPlan,
        budget: usize,
        eph: &Ephemeral<'_>,
        depth: usize,
    ) -> BoltResult<Option<LogicalPlan>> {
        let LogicalPlan::Join {
            left,
            right,
            join_type,
            ..
        } = node
        else {
            return Ok(None);
        };
        let Some(JoinPlan {
            a,
            b,
            keys_a,
            keys_b,
            est,
        }) = self.qualify_join(node, budget)?
        else {
            return Ok(None);
        };
        let (sa, sb) = (self.table_arrow_schema(&a)?, self.table_arrow_schema(&b)?);
        let parts = partitions_for(est, budget);
        let seed = 0x9a7e + depth as u64;
        let files_a = self.partition_table(left, &a, &keys_a, parts, seed)?;
        let files_b = self.partition_table(right, &b, &keys_b, parts, seed)?;
        let (base_a, base_b) = (self.base_schema(&a)?, self.base_schema(&b)?);
        let out_schema = with_children(
            node,
            vec![
                rename_scans(left, &a, &a, &base_a),
                rename_scans(right, &b, &b, &base_b),
            ],
        )
        .schema()?;
        let mut outputs = Vec::new();
        for (fa, fb) in files_a.into_iter().zip(files_b) {
            // A partition pair can produce rows only if the sides the join
            // type preserves (or both, for INNER) are present.
            let (has_a, has_b) = (fa.is_some(), fb.is_some());
            let produces = match join_type {
                JoinType::Inner => has_a && has_b,
                JoinType::LeftOuter => has_a,
                JoinType::RightOuter => has_b,
                JoinType::FullOuter => has_a || has_b,
                JoinType::Cross => false,
            };
            if !produces {
                continue;
            }
            let load = |f: Option<SpillFile>, schema: &SchemaRef| -> BoltResult<RecordBatch> {
                match f {
                    Some(f) => Ok(concat(None, &f.read_all()?)?.expect("non-empty spill file")),
                    None => Ok(RecordBatch::new_empty(Arc::clone(schema))),
                }
            };
            let pa = load(fa, &sa)?;
            let pb = load(fb, &sb)?;
            let (na, nb) = (eph.install(pa), eph.install(pb));
            let sub = with_children(
                node,
                vec![
                    rename_scans(left, &a, &na, &base_a),
                    rename_scans(right, &b, &nb, &base_b),
                ],
            );
            let out = self.execute_logical(&sub, true, false)?.into_record_batch();
            eph.release(&na);
            eph.release(&nb);
            if out.num_rows() > 0 {
                outputs.push(out);
            }
        }
        let result = match concat(None, &outputs)? {
            Some(r) => r,
            None => {
                // No partition produced rows: an empty result of the join's
                // shape.
                let schema = crate::exec::schema_convert::plan_schema_to_arrow_schema(&out_schema)?;
                RecordBatch::new_empty(schema)
            }
        };
        Ok(Some(self.substitute(out_schema, result, eph)?))
    }

    /// External sort for `[LIMIT] ORDER BY` over a row-wise scan.
    fn try_external_sort(
        &self,
        plan: &LogicalPlan,
        budget: usize,
        eph: &Ephemeral<'_>,
    ) -> BoltResult<Option<QueryHandle>> {
        let Some(SortPlan {
            limit,
            input,
            sort_exprs,
            table,
            out_schema,
            keys,
            run_rows,
            ..
        }) = self.qualify_sort(plan, budget)?
        else {
            return Ok(None);
        };
        let base = self.base_schema(&table)?;
        let mut runs: Vec<RecordBatch> = Vec::new();
        let mut pending: Vec<RecordBatch> = Vec::new();
        let mut pending_rows = 0usize;
        let flush =
            |pending: &mut Vec<RecordBatch>, runs: &mut Vec<RecordBatch>| -> BoltResult<()> {
                let Some(rows) = concat(None, pending)? else {
                    return Ok(());
                };
                pending.clear();
                let e = eph.install(rows);
                let mut sub = LogicalPlan::Sort {
                    input: Box::new(rename_scans(input, &table, &e, &base)),
                    sort_exprs: sort_exprs.to_vec(),
                };
                if let Some((l, o)) = limit {
                    sub = LogicalPlan::Limit {
                        input: Box::new(sub),
                        limit: l.saturating_add(o),
                        offset: 0,
                    };
                }
                let out = self.execute_logical(&sub, true, false)?.into_record_batch();
                eph.release(&e);
                if out.num_rows() > 0 {
                    runs.push(out);
                }
                Ok(())
            };
        let request = self.request_for(plan, &table);
        for batch in self.table_batches(&table, &request)? {
            let batch = batch?;
            pending_rows += batch.num_rows();
            pending.push(batch);
            if pending_rows >= run_rows {
                flush(&mut pending, &mut runs)?;
                pending_rows = 0;
            }
        }
        flush(&mut pending, &mut runs)?;
        let merged = merge_sorted_runs(&runs, &keys, limit)?;
        let merged = match merged {
            Some(m) => renamed(m, &out_schema)?,
            None => RecordBatch::new_empty(
                crate::exec::schema_convert::plan_schema_to_arrow_schema(&out_schema)?,
            ),
        };
        Ok(Some(QueryHandle::from_record_batch(merged)))
    }

    /// Whether `node` (an `Aggregate` / `Distinct`) runs as a partitioned
    /// GROUP BY, and on what.
    fn qualify_group(&self, node: &LogicalPlan, budget: usize) -> BoltResult<Option<GroupPlan>> {
        let (input, keys, all_streamable): (&LogicalPlan, Vec<String>, bool) = match node {
            LogicalPlan::Aggregate {
                input,
                group_by,
                aggregates,
            } => {
                let keys: Option<Vec<String>> = group_by
                    .iter()
                    .map(|e| column_name(e).map(str::to_string))
                    .collect();
                match keys {
                    Some(k) if !k.is_empty() => (input, k, aggregates.iter().all(streamable)),
                    _ => return Ok(None),
                }
            }
            LogicalPlan::Distinct { input } => {
                let keys = input.schema()?.fields.into_iter().map(|f| f.name).collect();
                (input, keys, false)
            }
            _ => return Ok(None),
        };
        let Some(resolved) = keys
            .iter()
            .map(|k| resolve_base_column(input, k, true))
            .collect::<Option<Vec<(String, String)>>>()
        else {
            return Ok(None);
        };
        let table = resolved[0].0.clone();
        if resolved.iter().any(|(t, _)| *t != table) || count_scans(node, &table) != 1 {
            return Ok(None);
        }
        let Some(est) = self.estimated_table_bytes(&table) else {
            return Ok(None);
        };
        if est <= budget {
            return Ok(None);
        }
        // The morsel path already bounds device memory for these over a
        // lazily-held table (it keeps the table in host memory, so only when
        // that fits). In-memory registered tables have no morsel path.
        let morsel_ok = self.memory_budget_bytes().is_some()
            && !self.tables.contains_key(&table)
            && self.host_memory_budget_bytes().is_none_or(|h| est <= h);
        if all_streamable && !contains_join(input) && morsel_ok {
            return Ok(None);
        }
        let keys = resolved.into_iter().map(|(_, c)| c).collect();
        Ok(Some(GroupPlan { table, keys, est }))
    }

    /// Whether `node` (a `Join`) runs as a grace hash join, and on what.
    fn qualify_join(&self, node: &LogicalPlan, budget: usize) -> BoltResult<Option<JoinPlan>> {
        let LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            ..
        } = node
        else {
            return Ok(None);
        };
        if matches!(join_type, JoinType::Cross) || on.is_empty() {
            return Ok(None);
        }
        let (Some(a), Some(b)) = (single_base_table(left), single_base_table(right)) else {
            return Ok(None);
        };
        let (a, b) = (a.to_string(), b.to_string());
        let (Some(ea), Some(eb)) = (
            self.estimated_table_bytes(&a),
            self.estimated_table_bytes(&b),
        ) else {
            return Ok(None);
        };
        if ea.max(eb) <= budget {
            return Ok(None);
        }
        let mut keys_a = Vec::new();
        let mut keys_b = Vec::new();
        for (l, r) in on {
            let (Some(l), Some(r)) = (column_name(l), column_name(r)) else {
                return Ok(None);
            };
            let pair = match (
                resolve_base_column(left, l, false),
                resolve_base_column(right, r, false),
            ) {
                (Some(x), Some(y)) => (x.1, y.1),
                _ => match (
                    resolve_base_column(left, r, false),
                    resolve_base_column(right, l, false),
                ) {
                    (Some(x), Some(y)) => (x.1, y.1),
                    _ => return Ok(None),
                },
            };
            keys_a.push(pair.0);
            keys_b.push(pair.1);
        }
        let (sa, sb) = (self.table_arrow_schema(&a)?, self.table_arrow_schema(&b)?);
        for (ka, kb) in keys_a.iter().zip(&keys_b) {
            let fa = key_family(
                sa.field_with_name(ka)
                    .map_err(|e| BoltError::Plan(e.to_string()))?
                    .data_type(),
            );
            let fb = key_family(
                sb.field_with_name(kb)
                    .map_err(|e| BoltError::Plan(e.to_string()))?
                    .data_type(),
            );
            if fa.is_none() || fa != fb {
                return Ok(None);
            }
        }
        Ok(Some(JoinPlan {
            est: ea + eb,
            a,
            b,
            keys_a,
            keys_b,
        }))
    }

    /// Whether the root of `plan` runs as an external sort.
    fn qualify_sort<'p>(
        &self,
        plan: &'p LogicalPlan,
        budget: usize,
    ) -> BoltResult<Option<SortPlan<'p>>> {
        let (limit, sort) = match plan {
            LogicalPlan::Limit {
                input,
                limit,
                offset,
            } if matches!(input.as_ref(), LogicalPlan::Sort { .. }) => {
                (Some((*limit, *offset)), input.as_ref())
            }
            LogicalPlan::Sort { .. } => (None, plan),
            _ => return Ok(None),
        };
        let LogicalPlan::Sort { input, sort_exprs } = sort else {
            return Ok(None);
        };
        let Some(table) = single_base_table(input).map(str::to_string) else {
            return Ok(None);
        };
        let Some(est) = self.estimated_table_bytes(&table) else {
            return Ok(None);
        };
        if est <= budget {
            return Ok(None);
        }
        let out_schema = input.schema()?;
        let Some(keys) = sort_exprs
            .iter()
            .map(|s| {
                let c = column_name(&s.expr)?;
                let i = out_schema.index_of(c).ok()?;
                Some((
                    i,
                    SortOptions {
                        descending: s.descending,
                        nulls_first: s.nulls_first,
                    },
                ))
            })
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        let rows_total = self
            .storage_table(&table)
            .map(|t| t.statistics().row_count as usize)
            .or_else(|| {
                self.tables
                    .get(&table)
                    .map(|b| b.iter().map(RecordBatch::num_rows).sum())
            })
            .unwrap_or(0)
            .max(1);
        let row_bytes = (est / rows_total).max(1);
        let run_rows = (budget / 2 / row_bytes).max(1024);
        Ok(Some(SortPlan {
            limit,
            input,
            sort_exprs,
            table,
            est,
            out_schema,
            keys,
            run_rows,
        }))
    }

    /// How `plan` would run out of core, for `EXPLAIN` (`None` when it would
    /// not). Describes the first node the executor would take.
    pub(crate) fn describe_out_of_core(&self, plan: &LogicalPlan) -> BoltResult<Option<String>> {
        let Some(budget) = self.ooc_budget() else {
            return Ok(None);
        };
        if let Some(p) = self.qualify_sort(plan, budget)? {
            return Ok(Some(format!(
                "external sort of '{}' (~{} bytes, budget {budget}): sorted runs of {} rows, k-way merge{}",
                p.table,
                p.est,
                p.run_rows,
                if p.limit.is_some() { " (top-k runs)" } else { "" }
            )));
        }
        self.describe_first(plan, budget, 0)
    }

    fn describe_first(
        &self,
        plan: &LogicalPlan,
        budget: usize,
        depth: usize,
    ) -> BoltResult<Option<String>> {
        if depth > crate::plan::sql_frontend::MAX_RECURSION_DEPTH {
            return Ok(None);
        }
        if matches!(
            plan,
            LogicalPlan::Aggregate { .. } | LogicalPlan::Distinct { .. }
        ) {
            if let Some(g) = self.qualify_group(plan, budget)? {
                let what = if matches!(plan, LogicalPlan::Distinct { .. }) {
                    "DISTINCT"
                } else {
                    "GROUP BY"
                };
                return Ok(Some(format!(
                    "partitioned {what} on '{}' ({}) — ~{} bytes, budget {budget}, {} spill partitions",
                    g.table,
                    g.keys.join(", "),
                    g.est,
                    partitions_for(g.est, budget)
                )));
            }
        }
        for child in plan.children() {
            if let Some(d) = self.describe_first(child, budget, depth + 1)? {
                return Ok(Some(d));
            }
        }
        if let Some(j) = self.qualify_join(plan, budget)? {
            return Ok(Some(format!(
                "grace hash join of '{}' ({}) with '{}' ({}) — ~{} bytes, budget {budget}, {} spill partitions",
                j.a,
                j.keys_a.join(", "),
                j.b,
                j.keys_b.join(", "),
                j.est,
                partitions_for(j.est, budget)
            )));
        }
        Ok(None)
    }

    /// Plan-level base schema of `table` (no rewriter-synthesised columns).
    fn base_schema(&self, table: &str) -> BoltResult<Schema> {
        if let Some(t) = self.storage_table(table) {
            return Ok(t.schema().clone());
        }
        let arrow = self.table_arrow_schema(table)?;
        crate::exec::engine_support::arrow_schema_to_plan_schema(arrow.as_ref())
    }

    /// Pruned scan request of storage table `table` within `plan`.
    fn request_for(&self, plan: &LogicalPlan, table: &str) -> ScanRequest {
        if self.storage_table(table).is_none() {
            return ScanRequest::full();
        }
        let tables = vec![table.to_string()];
        self.storage_scan_requests(plan, &tables, &[])
            .remove(table)
            .unwrap_or_else(ScanRequest::full)
    }

    /// Stream `table`'s rows in bounded batches without materialising it.
    fn table_batches(
        &self,
        table: &str,
        request: &ScanRequest,
    ) -> BoltResult<Box<dyn Iterator<Item = BoltResult<RecordBatch>>>> {
        let slice =
            |batches: Vec<RecordBatch>| -> Box<dyn Iterator<Item = BoltResult<RecordBatch>>> {
                let mut out = Vec::new();
                for b in batches {
                    let mut off = 0;
                    while off < b.num_rows() {
                        let len = SLICE_ROWS.min(b.num_rows() - off);
                        out.push(Ok(b.slice(off, len)));
                        off += len;
                    }
                }
                Box::new(out.into_iter())
            };
        if let Some(b) = self.tables.get(table) {
            return Ok(slice(b.clone()));
        }
        if let Some(t) = self.storage_table(table) {
            let sel = t.plan_scan(request);
            return Ok(Box::new(t.read(&sel)?));
        }
        match self.streaming_sources.borrow().get(table) {
            Some(TableSource::Materialized(b)) => Ok(slice(b.clone())),
            Some(TableSource::Streaming(producer)) => Ok(producer()),
            None => Err(BoltError::Plan(format!(
                "table '{table}' is not registered"
            ))),
        }
    }

    /// Hash-partition `table`'s rows (as read for `plan`) on `key_cols` into
    /// spill files; `None` for an empty partition.
    fn partition_table(
        &self,
        plan: &LogicalPlan,
        table: &str,
        key_cols: &[String],
        partitions: usize,
        seed: u64,
    ) -> BoltResult<Vec<Option<SpillFile>>> {
        let dir = self.spill_path();
        let request = self.request_for(plan, table);
        let mut writers: Vec<Option<SpillWriter>> = (0..partitions).map(|_| None).collect();
        let mut keys: Option<Vec<usize>> = None;
        for batch in self.table_batches(table, &request)? {
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }
            let k = match &keys {
                Some(k) => k.clone(),
                None => {
                    let schema = batch.schema();
                    let k = key_cols
                        .iter()
                        .map(|c| {
                            schema.index_of(c).map_err(|_| {
                                BoltError::Plan(format!("out-of-core: no key column '{c}'"))
                            })
                        })
                        .collect::<BoltResult<Vec<_>>>()?;
                    keys = Some(k.clone());
                    k
                }
            };
            let assign = partition_hash(&batch, &k, partitions, seed)?;
            for (p, part) in split_by_partition(&batch, &assign, partitions)?
                .into_iter()
                .enumerate()
            {
                if part.num_rows() == 0 {
                    continue;
                }
                let w = match &mut writers[p] {
                    Some(w) => w,
                    slot => slot.insert(SpillWriter::create(&dir, &part.schema())?),
                };
                w.write(&part)?;
            }
        }
        let files: Vec<Option<SpillFile>> = writers
            .into_iter()
            .map(|w| w.map(SpillWriter::finish).transpose())
            .collect::<BoltResult<_>>()?;
        let (n, rows, bytes) = files
            .iter()
            .flatten()
            .fold((0u64, 0u64, 0u64), |(n, r, b), f| {
                (n + 1, r + f.rows, b + f.bytes)
            });
        log::debug!("out-of-core: '{table}' spilled {rows} rows into {n} of {partitions} partitions ({bytes} bytes)");
        self.memory.record_spill(n, bytes);
        let m = crate::metrics::metrics();
        m.add(crate::metrics::Counter::SpillFilesWritten, n);
        m.add(crate::metrics::Counter::SpillBytesWritten, bytes);
        Ok(files)
    }
}

/// k-way merge of sorted `runs` on `keys`, then `LIMIT limit OFFSET offset`.
fn merge_sorted_runs(
    runs: &[RecordBatch],
    keys: &[(usize, SortOptions)],
    limit: Option<(usize, usize)>,
) -> BoltResult<Option<RecordBatch>> {
    let Some(first) = runs.first() else {
        return Ok(None);
    };
    let err = |e: arrow_schema::ArrowError| BoltError::Other(format!("out-of-core merge: {e}"));
    let conv = RowConverter::new(
        keys.iter()
            .map(|(c, o)| SortField::new_with_options(first.column(*c).data_type().clone(), *o))
            .collect(),
    )
    .map_err(err)?;
    let rows = runs
        .iter()
        .map(|r| {
            let cols: Vec<_> = keys.iter().map(|(c, _)| Arc::clone(r.column(*c))).collect();
            conv.convert_columns(&cols)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    let mut heap = BinaryHeap::new();
    for (r, rs) in rows.iter().enumerate() {
        if rs.num_rows() > 0 {
            heap.push(std::cmp::Reverse((rs.row(0).owned(), r, 0usize)));
        }
    }
    let (take, skip) = match limit {
        Some((l, o)) => (l, o),
        None => (usize::MAX, 0),
    };
    let mut indices: Vec<(usize, usize)> = Vec::new();
    let mut seen = 0usize;
    while let Some(std::cmp::Reverse((_, r, i))) = heap.pop() {
        if seen >= skip {
            indices.push((r, i));
            if indices.len() >= take {
                break;
            }
        }
        seen += 1;
        if i + 1 < rows[r].num_rows() {
            heap.push(std::cmp::Reverse((rows[r].row(i + 1).owned(), r, i + 1)));
        }
    }
    let cols = (0..first.num_columns())
        .map(|c| {
            let arrays: Vec<&dyn arrow_array::Array> =
                runs.iter().map(|r| r.column(c).as_ref()).collect();
            interleave(&arrays, &indices)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    RecordBatch::try_new(first.schema(), cols)
        .map(Some)
        .map_err(err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::{Int64Array, StringArray};

    fn run(v: Vec<i64>, s: Vec<&str>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", arrow_schema::DataType::Int64, false),
            ArrowField::new("s", arrow_schema::DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(v)),
                Arc::new(StringArray::from(s)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn merge_interleaves_runs_and_applies_limit() {
        let runs = vec![
            run(vec![1, 4, 9], vec!["a", "d", "i"]),
            run(vec![2, 3, 10], vec!["b", "c", "j"]),
        ];
        let asc = [(0, SortOptions::default())];
        let all = merge_sorted_runs(&runs, &asc, None).unwrap().unwrap();
        let k: Vec<i64> = all.column(0).as_primitive::<Int64Type>().values().to_vec();
        assert_eq!(k, vec![1, 2, 3, 4, 9, 10]);
        assert_eq!(all.column(1).as_string::<i32>().value(4), "i");
        let page = merge_sorted_runs(&runs, &asc, Some((2, 3)))
            .unwrap()
            .unwrap();
        let k: Vec<i64> = page.column(0).as_primitive::<Int64Type>().values().to_vec();
        assert_eq!(k, vec![4, 9]);
        let desc = [(
            0,
            SortOptions {
                descending: true,
                nulls_first: false,
            },
        )];
        let runs_desc = vec![
            run(vec![9, 4, 1], vec!["i", "d", "a"]),
            run(vec![10, 3, 2], vec!["j", "c", "b"]),
        ];
        let d = merge_sorted_runs(&runs_desc, &desc, Some((3, 0)))
            .unwrap()
            .unwrap();
        let k: Vec<i64> = d.column(0).as_primitive::<Int64Type>().values().to_vec();
        assert_eq!(k, vec![10, 9, 4]);
        assert!(merge_sorted_runs(&[], &asc, None).unwrap().is_none());
    }

    #[test]
    fn partition_counts_scale_with_size() {
        assert_eq!(partitions_for(10, 1000), 2);
        assert_eq!(partitions_for(10_000, 1000), 20);
        assert_eq!(partitions_for(usize::MAX / 2, 1), MAX_PARTITIONS);
    }
}
