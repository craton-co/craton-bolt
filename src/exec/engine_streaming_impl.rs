// SPDX-License-Identifier: Apache-2.0

//! Streaming / morsel-driven execution methods for [`Engine`].
//!
//! Pure-reorg split of the former monolithic `engine.rs`: these
//! `impl Engine` methods were moved here verbatim to keep the parent file
//! navigable. No behaviour change — only the methods' visibility was widened
//! to `pub(crate)` so the top-level dispatch in `engine.rs` (which stays
//! there) can call into this sibling module.
//!
//! The cluster is the streaming gate predicates (`streamable_*`) and their
//! morsel-by-morsel drivers (`execute_streaming_*`). See the per-method docs
//! for the distributive-aggregate / row-wise-leaf invariants that make
//! morsel-by-morsel processing byte-for-byte identical to the whole-table
//! path.

use arrow_array::RecordBatch;

use crate::error::{BoltError, BoltResult};
use crate::exec::engine::{combine_scalar_aggregate_partials, Engine, QueryHandle};
use crate::exec::engine_support::plan_schema_to_arrow_schema;
use crate::plan::{DataType, PhysicalPlan, Schema};

impl Engine {
    /// If `phys` is a **streamable leaf scan** — a plan whose output rows are an
    /// independent, row-wise function of a single base table's scan rows, so
    /// that processing the table morsel-by-morsel and concatenating the
    /// per-morsel results yields the *byte-for-byte identical* batch as
    /// processing the whole table at once — return that table's name.
    ///
    /// The streamable shapes are exactly the three scan-leaf executors that take
    /// a `table: String` directly (no child sub-plan) and emit one output row
    /// per surviving input row:
    ///
    /// * [`PhysicalPlan::Projection`] — fused project/filter kernel. A `WHERE`
    ///   only *drops* rows; the surviving rows are unchanged and order-preserved
    ///   within each morsel, and morsels are produced in table order, so
    ///   `concat(project(morsel_i)) == project(concat(morsel_i))`.
    /// * [`PhysicalPlan::StringLength`] — `LENGTH(col)` + passthroughs.
    /// * [`PhysicalPlan::StringProject`] — `UPPER`/`LOWER`/passthroughs.
    ///
    /// Every *other* variant is **not** returned here. A scalar `Aggregate`
    /// over a distributive function (`SUM`/`COUNT`/`MIN`/`MAX`) has its OWN
    /// streaming hook ([`Engine::streamable_scalar_aggregate`] /
    /// [`Engine::execute_streaming_scalar_aggregate`], which folds per-morsel
    /// partials), so it does not need to appear here. Everything else drains the
    /// whole table (status quo): non-distributive / grouped `Aggregate`
    /// (`AVG`/variance, or any GROUP BY — a cross-row fold),
    /// `Sort`/`Distinct`/`SetOp`/`Union`/`Window` (cross-row ordering or
    /// dedup), `Join` (build side must be resident), `Limit`/`Filter`/`Project`/
    /// `CountRows`/`StringLikeFilter` (wrap a child sub-plan whose own scan would
    /// have to be threaded — out of scope for this minimal, correctness-first
    /// cut). Those keep the existing materialise-whole-table behaviour exactly.
    pub(crate) fn streamable_leaf_scan(phys: &PhysicalPlan) -> Option<&str> {
        match phys {
            PhysicalPlan::Projection { table, .. }
            | PhysicalPlan::StringLength { table, .. }
            | PhysicalPlan::StringProject { table, .. } => Some(table.as_str()),
            _ => None,
        }
    }

    /// Drive a streamable leaf scan morsel-by-morsel instead of materialising
    /// the whole table on the device at once.
    ///
    /// Precondition: `table` is a streaming-registered overlay table (present in
    /// `streaming_sources`, **not** in the eager `tables` store) — the only
    /// shape whose data this method can swap per-morsel under `&self`, because
    /// the overlay is interior-mutable (`RefCell`) while `tables` is not. The
    /// caller ([`Engine::execute`]) enforces this.
    ///
    /// For each morsel the table's overlay entry is temporarily replaced with a
    /// single-batch `Materialized` view of just that morsel, the per-morsel leaf
    /// executor runs (rebuilding a small, bounded GPU table for the morsel), and
    /// the result batch is collected. The original whole-table overlay entry is
    /// always restored afterwards — including on the error path — so a producer
    /// or kernel fault leaves no torn state. The per-morsel results are
    /// concatenated into the final batch, which is identical to the whole-table
    /// result because the leaf shapes are row-wise (see
    /// [`Engine::streamable_leaf_scan`]).
    ///
    /// `run_morsel` is the existing per-morsel dispatch (the same executor the
    /// whole-table path uses); it is invoked while the morsel is installed.
    pub(crate) fn execute_streaming_leaf(
        &self,
        table: &str,
        morsel_rows: usize,
        run_morsel: impl Fn() -> BoltResult<QueryHandle>,
        output_schema: &Schema,
    ) -> BoltResult<QueryHandle> {
        use crate::exec::streaming::{BatchStream, TableSource};

        // Snapshot the whole-table batches (cheap Arc clones) and the original
        // overlay entry so we can restore it. We must NOT hold the overlay
        // borrow across `run_morsel` (which re-borrows the overlay through
        // `materialize_table`/`ensure_gpu_table`), so we take owned batches.
        let whole: Vec<RecordBatch> = {
            let overlay = self.streaming_sources.borrow();
            match overlay.get(table) {
                Some(TableSource::Materialized(b)) => b.clone(),
                // The caller guarantees an overlay table; a still-`Streaming`
                // entry should have been collapsed by
                // `ensure_streaming_materialized_for` before `execute`.
                _ => {
                    return Err(BoltError::Other(format!(
                        "execute_streaming_leaf: table '{table}' is not a \
                         materialised streaming-overlay table"
                    )))
                }
            }
        };

        let mut results: Vec<RecordBatch> = Vec::new();

        // Run the morsels, restoring the whole-table overlay entry no matter
        // how the loop exits. The `BatchStream` (which borrows `whole`) is
        // built and fully consumed INSIDE this closure, so its borrow ends
        // before we move `whole` back into the overlay below.
        let loop_result: BoltResult<()> = (|| {
            let stream = BatchStream::new(&whole, morsel_rows)?;
            for morsel in stream.morsels() {
                // Install just this morsel as the table's data.
                self.streaming_sources
                    .borrow_mut()
                    .insert(table.to_string(), TableSource::Materialized(vec![morsel]));
                // A streaming-overlay table carries no `host_revisions` entry,
                // so `ensure_gpu_table` always rebuilds a fresh (small) GPU
                // table from the installed morsel — no stale-cache hazard.
                let handle = run_morsel()?;
                results.push(handle.batch);
            }
            Ok(())
        })();

        // Always restore the whole-table view (the source is re-iterable, so a
        // subsequent query sees the full table again).
        self.streaming_sources
            .borrow_mut()
            .insert(table.to_string(), TableSource::Materialized(whole));

        loop_result?;

        // Concatenate the per-morsel results. Zero morsels (an all-empty table)
        // yields an empty batch shaped by the output schema.
        let batch = if results.is_empty() {
            let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
            RecordBatch::new_empty(arrow_schema)
        } else {
            let schema = results[0].schema();
            arrow::compute::concat_batches(&schema, results.iter()).map_err(|e| {
                BoltError::Other(format!(
                    "execute_streaming_leaf: failed to concatenate {} morsel \
                     results for table '{table}': {e}",
                    results.len()
                ))
            })?
        };
        Ok(QueryHandle::from_record_batch(batch))
    }

    /// If `phys` is a **streamable scalar aggregate** — a no-GROUP-BY aggregate
    /// whose every aggregate function is *distributive* (combinable from
    /// per-morsel partials by re-applying a simple fold) — return the scanned
    /// table's name. Otherwise `None` (the caller drains the whole table).
    ///
    /// The distributive set is exactly `COUNT` / `SUM` / `MIN` / `MAX`:
    ///
    /// * `COUNT(e)` — partial is each morsel's non-NULL count; the whole-table
    ///   count is the **sum** of the partials.
    /// * `SUM(e)` — partial is each morsel's sum; the total is the **sum** of
    ///   the partials (NULL partials — an all-NULL/empty morsel — are ignored,
    ///   matching `SUM` SQL semantics).
    /// * `MIN(e)` / `MAX(e)` — partial is each morsel's extremum; the total is
    ///   the **min**/**max** of the partials (NULL partials ignored).
    ///
    /// `AVG`, `VAR_POP`/`VAR_SAMP`, `STDDEV_POP`/`STDDEV_SAMP` are **not**
    /// distributive over their *finalised* scalar output (an average of
    /// per-morsel averages is not the whole-table average), so they return
    /// `None` here and keep the legacy whole-table drain. A `WHERE` filter
    /// (carried in `pre`) is fine — it is row-wise, so per-morsel filtering
    /// then combining the survivors' partials is identical to filtering the
    /// whole table.
    ///
    /// `pre` is *not* inspected here beyond the aggregate-function check: the
    /// per-morsel executor (`execute_leaf_whole`) runs the exact same
    /// `pre`+reduce pipeline the whole-table path would, just on one morsel at a
    /// time, so any `pre` shape the whole-table aggregate accepts is accepted
    /// here too.
    pub(crate) fn streamable_scalar_aggregate(phys: &PhysicalPlan) -> Option<&str> {
        use crate::plan::logical_plan::AggregateExpr;
        let PhysicalPlan::Aggregate {
            table, aggregate, ..
        } = phys
        else {
            return None;
        };
        if !aggregate.group_by.is_empty() {
            return None;
        }
        if aggregate.aggregates.is_empty() {
            return None;
        }
        // Every aggregate must be distributive (combinable from finalised
        // per-morsel partials) AND emit one of the primitive numeric output
        // dtypes the host combiner ([`combine_scalar_aggregate_partials`])
        // folds. Decimal128 SUM/MIN/MAX are *also* distributive, but the host
        // combiner does not yet fold i128 partials, so a Decimal output keeps
        // the whole-table drain (see TODO in `combine_scalar_aggregate_partials`).
        if aggregate.output_schema.fields.len() != aggregate.aggregates.len() {
            return None;
        }
        let all_streamable = aggregate
            .aggregates
            .iter()
            .zip(aggregate.output_schema.fields.iter())
            .all(|(a, f)| {
                let distributive = matches!(
                    a,
                    AggregateExpr::Count(_)
                        | AggregateExpr::Sum(_)
                        | AggregateExpr::Min(_)
                        | AggregateExpr::Max(_)
                );
                let foldable_dtype = matches!(
                    f.dtype,
                    DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64
                );
                distributive && foldable_dtype
            });
        if all_streamable {
            Some(table.as_str())
        } else {
            None
        }
    }

    /// Drive a streamable **scalar aggregate** morsel-by-morsel instead of
    /// materialising the whole table on the device at once.
    ///
    /// Mirrors [`Engine::execute_streaming_leaf`]'s overlay-swap discipline: for
    /// each morsel the streaming-overlay entry for `table` is temporarily
    /// replaced with a single-batch view of that morsel, the *same* per-morsel
    /// aggregate executor the whole-table path uses (`run_morsel`) runs against
    /// it producing a single-row **partial** batch, and the whole-table overlay
    /// is always restored afterwards (including on the error path).
    ///
    /// The per-morsel partials are then folded into the final single-row result
    /// by [`combine_scalar_aggregate_partials`]. Because every aggregate is
    /// distributive (see [`Engine::streamable_scalar_aggregate`]) the combined
    /// result equals the whole-table aggregate exactly.
    ///
    /// Precondition (enforced by the caller in [`Engine::execute`]): `table` is
    /// a streaming-registered overlay table and `aggregate.group_by` is empty.
    pub(crate) fn execute_streaming_scalar_aggregate(
        &self,
        table: &str,
        morsel_rows: usize,
        run_morsel: impl Fn() -> BoltResult<QueryHandle>,
        aggregate: &crate::plan::physical_plan::AggregateSpec,
    ) -> BoltResult<QueryHandle> {
        use crate::exec::streaming::{BatchStream, TableSource};

        // Snapshot the whole-table batches and original overlay entry so we can
        // restore it. Owned (cheap Arc clones) so we never hold the overlay
        // borrow across `run_morsel` (which re-borrows the overlay).
        let whole: Vec<RecordBatch> = {
            let overlay = self.streaming_sources.borrow();
            match overlay.get(table) {
                Some(TableSource::Materialized(b)) => b.clone(),
                _ => {
                    return Err(BoltError::Other(format!(
                        "execute_streaming_scalar_aggregate: table '{table}' is not a \
                         materialised streaming-overlay table"
                    )))
                }
            }
        };

        let mut partials: Vec<RecordBatch> = Vec::new();

        let loop_result: BoltResult<()> = (|| {
            let stream = BatchStream::new(&whole, morsel_rows)?;
            for morsel in stream.morsels() {
                self.streaming_sources
                    .borrow_mut()
                    .insert(table.to_string(), TableSource::Materialized(vec![morsel]));
                let handle = run_morsel()?;
                partials.push(handle.batch);
            }
            Ok(())
        })();

        // Always restore the whole-table view.
        self.streaming_sources
            .borrow_mut()
            .insert(table.to_string(), TableSource::Materialized(whole));

        loop_result?;

        let batch = combine_scalar_aggregate_partials(aggregate, &partials)?;
        Ok(QueryHandle::from_record_batch(batch))
    }

    /// If `phys` is a **streamable grouped aggregate** — a `GROUP BY` whose
    /// every aggregate function is *distributive* (combinable from per-morsel
    /// partials by re-applying a keyed fold) and whose every aggregate output
    /// column is a primitive numeric dtype the host merge
    /// ([`crate::exec::streaming::merge_grouped_partials`]) can fold — return
    /// `(table, per-aggregate fold ops)`. Otherwise `None` (the caller drains
    /// the whole table).
    ///
    /// The distributive set is `COUNT` / `SUM` / `MIN` / `MAX`, exactly as for
    /// the scalar streaming gate ([`Engine::streamable_scalar_aggregate`]),
    /// lifted to a keyed merge:
    ///
    /// * `COUNT` / `SUM` → per-key **Add** across morsels.
    /// * `MIN` / `MAX` → per-key **min** / **max** across morsels.
    ///
    /// `AVG`, `VAR_*`, `STDDEV_*` are **not** distributive over their finalised
    /// per-morsel output (an average of per-morsel averages is not the
    /// whole-table average), so they return `None` and keep the whole-table
    /// drain. A grouped aggregate with **zero** aggregate functions (a pure
    /// `SELECT DISTINCT`-shaped `GROUP BY g`) also returns `None` — there is no
    /// distributive value to fold and the whole-table dedup path is correct as
    /// is. Decimal128 aggregate outputs are *also* distributive but the host
    /// merge does not yet fold i128 partials with scale, so a Decimal output
    /// keeps the whole-table drain (see the TODO on
    /// [`crate::exec::streaming::merge_grouped_partials`]).
    ///
    /// The group-key columns are inspected only for their **count** (they are
    /// rebuilt verbatim from the partials by the merge); their dtypes may be any
    /// type the per-batch executor and the row-key relation support
    /// (Int/Float/Bool/Utf8).
    pub(crate) fn streamable_grouped_aggregate(
        phys: &PhysicalPlan,
    ) -> Option<(&str, Vec<crate::exec::streaming::GroupedFold>)> {
        use crate::exec::streaming::GroupedFold;
        use crate::plan::logical_plan::AggregateExpr;
        let PhysicalPlan::Aggregate {
            table, aggregate, ..
        } = phys
        else {
            return None;
        };
        // Must be a GROUP BY (non-empty keys) with at least one aggregate.
        if aggregate.group_by.is_empty() || aggregate.aggregates.is_empty() {
            return None;
        }
        // Output schema is [group keys..., aggregate columns...]; the aggregate
        // result columns start after the group keys.
        let n_group = aggregate.group_by.len();
        if aggregate.output_schema.fields.len() != n_group + aggregate.aggregates.len() {
            return None;
        }
        let mut folds: Vec<GroupedFold> = Vec::with_capacity(aggregate.aggregates.len());
        for (i, agg) in aggregate.aggregates.iter().enumerate() {
            let field = &aggregate.output_schema.fields[n_group + i];
            let fold = match agg {
                AggregateExpr::Count(_) | AggregateExpr::Sum(_) => GroupedFold::Add,
                AggregateExpr::Min(_) => GroupedFold::Min,
                AggregateExpr::Max(_) => GroupedFold::Max,
                // AVG / VAR_* / STDDEV_* are not distributive over finalised
                // per-morsel partials — drain.
                _ => return None,
            };
            let foldable_dtype = matches!(
                field.dtype,
                DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64
            );
            if !foldable_dtype {
                return None;
            }
            folds.push(fold);
        }
        Some((table.as_str(), folds))
    }

    /// Drive a streamable **grouped aggregate** morsel-by-morsel instead of
    /// materialising the whole table on the device at once.
    ///
    /// Mirrors [`Engine::execute_streaming_scalar_aggregate`]'s overlay-swap
    /// discipline: for each morsel the streaming-overlay entry for `table` is
    /// temporarily replaced with a single-batch view of that morsel, the *same*
    /// per-batch grouped-aggregate executor the whole-table path uses
    /// (`run_morsel`) runs against it producing a **partial grouped batch**
    /// (group-key columns followed by per-aggregate result columns), and the
    /// whole-table overlay is always restored afterwards (including on the error
    /// path).
    ///
    /// The per-morsel partials are then hash-merged by key into the final
    /// grouped result by [`crate::exec::streaming::merge_grouped_partials`].
    /// Because every aggregate is distributive (see
    /// [`Engine::streamable_grouped_aggregate`]) the merged result equals the
    /// whole-table grouped aggregate exactly (modulo row order, which a GROUP BY
    /// without ORDER BY does not constrain).
    ///
    /// A zero-morsel table (zero rows) yields an empty batch shaped by the
    /// aggregate output schema — there is no partial to merge.
    ///
    /// Precondition (enforced by the caller in [`Engine::execute`]): `table` is
    /// a streaming-registered overlay table and `aggregate.group_by` is
    /// non-empty.
    pub(crate) fn execute_streaming_grouped_aggregate(
        &self,
        table: &str,
        morsel_rows: usize,
        run_morsel: impl Fn() -> BoltResult<QueryHandle>,
        aggregate: &crate::plan::physical_plan::AggregateSpec,
        folds: &[crate::exec::streaming::GroupedFold],
    ) -> BoltResult<QueryHandle> {
        use crate::exec::streaming::{merge_grouped_partials, BatchStream, TableSource};

        let whole: Vec<RecordBatch> = {
            let overlay = self.streaming_sources.borrow();
            match overlay.get(table) {
                Some(TableSource::Materialized(b)) => b.clone(),
                _ => {
                    return Err(BoltError::Other(format!(
                        "execute_streaming_grouped_aggregate: table '{table}' is not a \
                         materialised streaming-overlay table"
                    )))
                }
            }
        };

        let mut partials: Vec<RecordBatch> = Vec::new();

        let loop_result: BoltResult<()> = (|| {
            let stream = BatchStream::new(&whole, morsel_rows)?;
            for morsel in stream.morsels() {
                self.streaming_sources
                    .borrow_mut()
                    .insert(table.to_string(), TableSource::Materialized(vec![morsel]));
                let handle = run_morsel()?;
                partials.push(handle.batch);
            }
            Ok(())
        })();

        // Always restore the whole-table view.
        self.streaming_sources
            .borrow_mut()
            .insert(table.to_string(), TableSource::Materialized(whole));

        loop_result?;

        // Zero morsels (zero-row table): emit an empty batch from the output
        // schema — there is no partial to merge or to borrow a schema from.
        let batch = if partials.is_empty() {
            let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
            RecordBatch::new_empty(arrow_schema)
        } else {
            merge_grouped_partials(&partials, aggregate.group_by.len(), folds)?
        };
        Ok(QueryHandle::from_record_batch(batch))
    }

    /// If `phys` is an aggregate (scalar or grouped) that includes at least
    /// one **algebraic** aggregate — `AVG`, `VAR_*`, `STDDEV_*` — whose every
    /// aggregate can be rebuilt from distributive per-morsel partials, return
    /// `(table, partial plan, merges)`.
    ///
    /// The partial plan is `phys` with each algebraic aggregate decomposed into
    /// distributive ones the per-batch executors already compute:
    /// `AVG(e)` → `SUM(e), COUNT(e)`; `VAR_*/STDDEV_*(e)` → `COUNT(e),
    /// SUM(e), VAR_POP(e)`. [`crate::exec::streaming::merge_algebraic_partials`]
    /// recombines them exactly (AVG = Σsum/Σcount; variance via the two-pass
    /// parallel update). Plans with only distributive aggregates are left to
    /// the cheaper distributive hooks.
    pub(crate) fn streamable_algebraic_aggregate(
        phys: &PhysicalPlan,
    ) -> Option<(
        &str,
        PhysicalPlan,
        Vec<crate::exec::streaming::AlgebraicMerge>,
    )> {
        use crate::exec::streaming::{AlgebraicMerge, GroupedFold};
        use crate::plan::logical_plan::{AggregateExpr, Field};
        let PhysicalPlan::Aggregate {
            table, aggregate, ..
        } = phys
        else {
            return None;
        };
        let n_group = aggregate.group_by.len();
        if aggregate.aggregates.is_empty()
            || aggregate.output_schema.fields.len() != n_group + aggregate.aggregates.len()
        {
            return None;
        }
        let input_schema = Schema::new(
            aggregate
                .inputs
                .iter()
                .map(|io| Field::new(io.name.clone(), io.dtype, true))
                .collect(),
        );
        let numeric = |d: DataType| {
            matches!(
                d,
                DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64
            )
        };
        let mut partial_aggs: Vec<AggregateExpr> = Vec::new();
        let mut partial_fields: Vec<Field> = aggregate.output_schema.fields[..n_group].to_vec();
        let mut merges: Vec<AlgebraicMerge> = Vec::new();
        let mut any_algebraic = false;
        let push = |agg: AggregateExpr,
                    aggs: &mut Vec<AggregateExpr>,
                    fields: &mut Vec<Field>|
         -> Option<usize> {
            let dtype = agg.output_dtype(&input_schema).ok()?;
            if !numeric(dtype) {
                return None;
            }
            fields.push(Field::new(format!("__partial_{}", aggs.len()), dtype, true));
            aggs.push(agg);
            Some(aggs.len() - 1)
        };
        for (i, agg) in aggregate.aggregates.iter().enumerate() {
            if !numeric(aggregate.output_schema.fields[n_group + i].dtype) {
                return None;
            }
            let merge = match agg {
                AggregateExpr::Count(_) | AggregateExpr::Sum(_) => AlgebraicMerge::Distributive {
                    col: push(agg.clone(), &mut partial_aggs, &mut partial_fields)?,
                    fold: GroupedFold::Add,
                },
                AggregateExpr::Min(_) => AlgebraicMerge::Distributive {
                    col: push(agg.clone(), &mut partial_aggs, &mut partial_fields)?,
                    fold: GroupedFold::Min,
                },
                AggregateExpr::Max(_) => AlgebraicMerge::Distributive {
                    col: push(agg.clone(), &mut partial_aggs, &mut partial_fields)?,
                    fold: GroupedFold::Max,
                },
                AggregateExpr::Avg(e) => {
                    any_algebraic = true;
                    AlgebraicMerge::Avg {
                        sum: push(
                            AggregateExpr::Sum(e.clone()),
                            &mut partial_aggs,
                            &mut partial_fields,
                        )?,
                        count: push(
                            AggregateExpr::Count(e.clone()),
                            &mut partial_aggs,
                            &mut partial_fields,
                        )?,
                    }
                }
                AggregateExpr::VarPop(e)
                | AggregateExpr::VarSamp(e)
                | AggregateExpr::StddevPop(e)
                | AggregateExpr::StddevSamp(e) => {
                    any_algebraic = true;
                    let sample = matches!(
                        agg,
                        AggregateExpr::VarSamp(_) | AggregateExpr::StddevSamp(_)
                    );
                    let sqrt = matches!(
                        agg,
                        AggregateExpr::StddevPop(_) | AggregateExpr::StddevSamp(_)
                    );
                    let inner = (**e).clone();
                    AlgebraicMerge::Variance {
                        count: push(
                            AggregateExpr::Count(inner.clone()),
                            &mut partial_aggs,
                            &mut partial_fields,
                        )?,
                        sum: push(
                            AggregateExpr::Sum(inner.clone()),
                            &mut partial_aggs,
                            &mut partial_fields,
                        )?,
                        var_pop: push(
                            AggregateExpr::VarPop(Box::new(inner)),
                            &mut partial_aggs,
                            &mut partial_fields,
                        )?,
                        sample,
                        sqrt,
                    }
                }
            };
            merges.push(merge);
        }
        if !any_algebraic {
            return None;
        }
        let mut partial = phys.clone();
        if let PhysicalPlan::Aggregate {
            aggregate: spec, ..
        } = &mut partial
        {
            spec.aggregates = partial_aggs;
            spec.output_schema = Schema::new(partial_fields);
        }
        Some((table.as_str(), partial, merges))
    }

    /// Drive an aggregate admitted by [`Engine::streamable_algebraic_aggregate`]
    /// morsel-by-morsel: run the decomposed `partial` plan on each morsel
    /// (`run_partial`, same overlay-swap discipline as the other streaming
    /// drivers), then recombine with
    /// [`crate::exec::streaming::merge_algebraic_partials`] into the original
    /// `aggregate`'s output schema.
    pub(crate) fn execute_streaming_algebraic_aggregate(
        &self,
        table: &str,
        morsel_rows: usize,
        run_partial: impl Fn() -> BoltResult<QueryHandle>,
        aggregate: &crate::plan::physical_plan::AggregateSpec,
        merges: &[crate::exec::streaming::AlgebraicMerge],
    ) -> BoltResult<QueryHandle> {
        use crate::exec::streaming::{merge_algebraic_partials, BatchStream, TableSource};

        let whole: Vec<RecordBatch> = {
            let overlay = self.streaming_sources.borrow();
            match overlay.get(table) {
                Some(TableSource::Materialized(b)) => b.clone(),
                _ => {
                    return Err(BoltError::Other(format!(
                        "execute_streaming_algebraic_aggregate: table '{table}' is not a \
                         materialised streaming-overlay table"
                    )))
                }
            }
        };
        let mut partials: Vec<RecordBatch> = Vec::new();
        let mut run_all = || -> BoltResult<()> {
            let stream = BatchStream::new(&whole, morsel_rows)?;
            for morsel in stream.morsels() {
                self.streaming_sources
                    .borrow_mut()
                    .insert(table.to_string(), TableSource::Materialized(vec![morsel]));
                partials.push(run_partial()?.batch);
            }
            Ok(())
        };
        let loop_result = run_all();
        // Always restore the whole-table view.
        self.streaming_sources
            .borrow_mut()
            .insert(table.to_string(), TableSource::Materialized(whole));
        loop_result?;

        let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
        let batch = if partials.is_empty() {
            RecordBatch::new_empty(arrow_schema)
        } else {
            merge_algebraic_partials(&partials, aggregate.group_by.len(), merges, arrow_schema)?
        };
        Ok(QueryHandle::from_record_batch(batch))
    }
}
