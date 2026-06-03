// SPDX-License-Identifier: Apache-2.0

//! Host-side aggregate / row-source query executors for [`Engine`].
//!
//! Pure-reorg split of the former monolithic `engine.rs`: these
//! `impl Engine` methods were moved here verbatim to keep the parent file
//! navigable. No behaviour change — every method had its visibility widened
//! to `pub(crate)` so the SQL-frontend dispatch in `Engine::sql` (which stays
//! in `engine.rs`) can call them across the module boundary.
//!
//! The cluster covers the host-grouped descriptors and constant row sources:
//! `COUNT(DISTINCT) GROUP BY` (sole + generalized multi-agg), `VALUES`,
//! `generate_series`, and `DISTINCT ON`. Each materialises a base batch (via
//! `run_subplan`), folds it on the host, and runs any HAVING / ORDER BY /
//! LIMIT post-plan through the streaming overlay.

use arrow_array::{ArrayRef, RecordBatch};

use crate::error::{BoltError, BoltResult};
use crate::exec::engine::{
    host_count_distinct_groupby, host_multi_agg_groupby, materialize_generate_series_relation,
    materialize_values_relation, Engine, QueryHandle,
};
use crate::exec::engine_support::plan_schema_to_arrow_schema;
use crate::plan::LogicalPlan;

impl Engine {
    /// Execute a sole `COUNT(DISTINCT col)` with `GROUP BY` (feature
    /// F3-finish) as a host-orchestrated special-case.
    ///
    /// The per-group distinct count cannot be expressed as a single
    /// `LogicalPlan` that flows through the ordinary pipeline (see
    /// [`crate::plan::sql_frontend::CountDistinctGroupByPlan`]), so — exactly
    /// like [`Engine::execute_recursive_cte`] — the engine orchestrates it:
    ///
    /// 1. **Base.** Run `cd.base` (an ordinary subplan) through
    ///    [`Engine::run_subplan`] to materialise a `RecordBatch` of
    ///    `[group_keys..., distinct_col]` rows.
    /// 2. **Group + count (host).** Group rows by the leading group-key tuple
    ///    (using the same `RowKey` / `RowKeyValue` canonicalisation the
    ///    `DISTINCT` operator uses, so NULL group keys form their own group and
    ///    composite keys compose), and per group accumulate the set of DISTINCT
    ///    **non-NULL** values of `distinct_col`. The count is `set.len()`
    ///    (standard SQL: `COUNT(DISTINCT x)` ignores NULLs, so an all-NULL or
    ///    empty group yields 0).
    /// 3. **Build result.** One row per group (in first-occurrence order): the
    ///    group-key values (taken from the base batch at each group's first row
    ///    via `arrow::compute::take`, preserving exact dtype/value) followed by
    ///    the `Int64` count under `cd.count_alias`.
    /// 4. **Post.** If `cd.post` is present (HAVING / ORDER BY / LIMIT), bind
    ///    the result under the reserved ephemeral name in the streaming overlay
    ///    and run the post-plan through [`Engine::run_subplan`] — reusing the
    ///    ordinary Filter / Sort / Limit executors — then clear the overlay.
    pub(crate) fn execute_count_distinct_groupby(
        &self,
        cd: &crate::plan::sql_frontend::CountDistinctGroupByPlan,
    ) -> BoltResult<QueryHandle> {
        // --- 1. Run the base subplan: [group_keys..., distinct_col]. ---
        let base = self.run_subplan(cd.base.clone())?;

        // --- 2 + 3. Group host-side and build the count-result batch. ---
        let result =
            host_count_distinct_groupby(&base, cd.group_key_names.len(), &cd.result_schema)?;

        // --- 4. Apply the optional HAVING / ORDER BY / LIMIT post-plan over the
        // result, bound under the reserved ephemeral name in the overlay (same
        // pattern as the recursive-CTE main query). ---
        let post = match &cd.post {
            None => return Ok(QueryHandle::from_record_batch(result)),
            Some(p) => p,
        };
        use crate::exec::streaming::TableSource;
        let name = crate::plan::sql_frontend::COUNT_DISTINCT_GROUPBY_RESULT_TABLE;
        if self.tables.contains_key(name) || self.streaming_sources.borrow().contains_key(name) {
            return Err(BoltError::Plan(format!(
                "COUNT(DISTINCT) GROUP BY: reserved result table name '{name}' collides \
                 with a registered table"
            )));
        }
        self.gpu_tables.borrow_mut().remove(name);
        self.streaming_sources
            .borrow_mut()
            .insert(name.to_string(), TableSource::Materialized(vec![result]));
        let out = self.run_subplan(post.clone());
        self.streaming_sources.borrow_mut().remove(name);
        self.gpu_tables.borrow_mut().remove(name);
        Ok(QueryHandle::from_record_batch(out?))
    }

    /// Execute a `VALUES`-sourced query (feature VALUES).
    ///
    /// 1. **Materialise** the inferred relation into an Arrow `RecordBatch` via
    ///    [`materialize_values_relation`].
    /// 2. **Bare form** (`vp.post` is `None`): apply the optional ORDER BY /
    ///    LIMIT carried on the descriptor by binding the relation under the
    ///    reserved name and running a synthetic `Sort` / `Limit` subplan over it
    ///    (so the ordinary sort / limit executors are reused); when there is no
    ///    ORDER BY / LIMIT, return the materialised batch directly.
    /// 3. **FROM form** (`vp.post` is `Some`): bind the relation under the user
    ///    alias in the streaming overlay and run the outer query template through
    ///    [`Engine::run_subplan`], mirroring the WITH RECURSIVE / LATERAL overlay
    ///    pattern. The overlay entry is always cleared afterwards.
    pub(crate) fn execute_values_query(
        &self,
        vp: &crate::plan::sql_frontend::ValuesQueryPlan,
    ) -> BoltResult<QueryHandle> {
        use crate::exec::streaming::TableSource;

        let batch = materialize_values_relation(&vp.relation)?;

        // Refuse to shadow a real table under the bind name.
        let name = &vp.bind_name;
        if self.tables.contains_key(name) || self.streaming_sources.borrow().contains_key(name) {
            return Err(BoltError::Plan(format!(
                "VALUES: relation name '{name}' collides with a registered table — \
                 rename the VALUES alias"
            )));
        }

        // Build the plan to run over the bound relation: either the FROM-form
        // `post` template, or (bare form) a synthetic Sort/Limit over a Scan of
        // the relation. If the bare form has neither ORDER BY nor LIMIT, return
        // the batch directly without an overlay round-trip.
        let plan = match &vp.post {
            Some(post) => post.clone(),
            None => {
                if vp.order_by.is_empty() && vp.limit.is_none() {
                    return Ok(QueryHandle::from_record_batch(batch));
                }
                let mut p = LogicalPlan::Scan {
                    table: name.clone(),
                    projection: None,
                    schema: vp.relation.schema.clone(),
                };
                if !vp.order_by.is_empty() {
                    p = LogicalPlan::Sort {
                        input: Box::new(p),
                        sort_exprs: vp.order_by.clone(),
                    };
                }
                if let Some((limit, offset)) = vp.limit {
                    p = LogicalPlan::Limit {
                        input: Box::new(p),
                        limit,
                        offset,
                    };
                }
                p
            }
        };

        self.gpu_tables.borrow_mut().remove(name);
        self.streaming_sources
            .borrow_mut()
            .insert(name.clone(), TableSource::Materialized(vec![batch]));
        let out = self.run_subplan(plan);
        self.streaming_sources.borrow_mut().remove(name);
        self.gpu_tables.borrow_mut().remove(name);
        Ok(QueryHandle::from_record_batch(out?))
    }

    /// Execute a `generate_series`-sourced query (feature GENERATE_SERIES).
    ///
    /// Mirrors the FROM form of [`Engine::execute_values_query`]: materialise the
    /// series into a single-column Int64 batch, bind it under the relation name in
    /// the streaming overlay, run the outer query template through
    /// [`Engine::run_subplan`], and always clear the overlay entry afterwards.
    pub(crate) fn execute_generate_series_query(
        &self,
        gp: &crate::plan::sql_frontend::GenerateSeriesQueryPlan,
    ) -> BoltResult<QueryHandle> {
        use crate::exec::streaming::TableSource;

        let batch = materialize_generate_series_relation(&gp.relation)?;

        // Refuse to shadow a real table under the bind name.
        let name = &gp.bind_name;
        if self.tables.contains_key(name) || self.streaming_sources.borrow().contains_key(name) {
            return Err(BoltError::Plan(format!(
                "generate_series: relation name '{name}' collides with a registered table — \
                 add an alias (e.g. `AS gs`)"
            )));
        }

        self.gpu_tables.borrow_mut().remove(name);
        self.streaming_sources
            .borrow_mut()
            .insert(name.clone(), TableSource::Materialized(vec![batch]));
        let out = self.run_subplan(gp.post.clone());
        self.streaming_sources.borrow_mut().remove(name);
        self.gpu_tables.borrow_mut().remove(name);
        Ok(QueryHandle::from_record_batch(out?))
    }

    /// Execute a `SELECT DISTINCT ON (keys) ...` query (feature DISTINCT ON).
    ///
    /// 1. **Run the base** subplan — `[__diston_0..N, <user projection...>]` in
    ///    the query's ORDER BY order — through the ordinary pipeline.
    /// 2. **Dedup (host).** Keep the first row per leading-key tuple via
    ///    [`crate::exec::distinct::distinct_on_first_per_key`] (NULL keys form
    ///    their own group, matching GROUP BY).
    /// 3. **Project away the key columns** to restore the user projection.
    /// 4. **Apply the optional LIMIT / OFFSET** (Postgres applies LIMIT to the
    ///    DISTINCT ON result), reusing the overlay + synthetic `Limit` subplan.
    pub(crate) fn execute_distinct_on(
        &self,
        dp: &crate::plan::sql_frontend::DistinctOnPlan,
    ) -> BoltResult<QueryHandle> {
        use crate::exec::streaming::TableSource;

        // --- 1. Base batch: [keys..., user cols...] in ORDER BY order. ---
        let base = self.run_subplan(dp.base.clone())?;

        // --- 2. Keep the first row per key tuple. ---
        let deduped = crate::exec::distinct::distinct_on_first_per_key(&base, dp.n_keys)?;

        // --- 3. Drop the leading key columns to restore the user projection. ---
        let user_cols: Vec<ArrayRef> = deduped
            .columns()
            .iter()
            .skip(dp.n_keys)
            .cloned()
            .collect();
        let out_schema = plan_schema_to_arrow_schema(&dp.output_schema)?;
        let projected = RecordBatch::try_new_with_options(
            out_schema,
            user_cols,
            &arrow_array::RecordBatchOptions::new().with_row_count(Some(deduped.num_rows())),
        )
        .map_err(|e| BoltError::Plan(format!("DISTINCT ON projection: {e}")))?;

        // --- 4. Optional LIMIT / OFFSET via the overlay + a synthetic Limit. ---
        let (limit, offset) = match dp.limit {
            None => return Ok(QueryHandle::from_record_batch(projected)),
            Some(lo) => lo,
        };
        let name = crate::plan::sql_frontend::DISTINCT_ON_RESULT_TABLE;
        if self.tables.contains_key(name) || self.streaming_sources.borrow().contains_key(name) {
            return Err(BoltError::Plan(format!(
                "DISTINCT ON: reserved result table name '{name}' collides with a \
                 registered table"
            )));
        }
        let plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::Scan {
                table: name.to_string(),
                projection: None,
                schema: dp.output_schema.clone(),
            }),
            limit,
            offset,
        };
        self.gpu_tables.borrow_mut().remove(name);
        self.streaming_sources
            .borrow_mut()
            .insert(name.to_string(), TableSource::Materialized(vec![projected]));
        let out = self.run_subplan(plan);
        self.streaming_sources.borrow_mut().remove(name);
        self.gpu_tables.borrow_mut().remove(name);
        Ok(QueryHandle::from_record_batch(out?))
    }

    /// Execute the *generalized* COUNT(DISTINCT) + GROUP BY descriptor
    /// (feature F3-finish, generalized): multiple distinct counts and/or a mix
    /// with plain aggregates. Mirrors [`Engine::execute_count_distinct_groupby`]
    /// but delegates the per-group work to [`host_multi_agg_groupby`] (which
    /// computes every aggregate per group) and binds the result under
    /// [`crate::plan::sql_frontend::MULTI_AGG_GROUPBY_RESULT_TABLE`] for the
    /// optional HAVING / ORDER BY / LIMIT post-plan.
    pub(crate) fn execute_multi_agg_groupby(
        &self,
        cd: &crate::plan::sql_frontend::MultiAggGroupByPlan,
    ) -> BoltResult<QueryHandle> {
        // --- 1. Run the base subplan: [group_keys..., agg_inputs...]. ---
        let base = self.run_subplan(cd.base.clone())?;

        // --- 2 + 3. Group + compute every aggregate host-side. ---
        let result = host_multi_agg_groupby(
            &base,
            cd.n_keys,
            &cd.aggs,
            &cd.output_layout,
            &cd.result_schema,
        )?;

        // --- 4. Optional HAVING / ORDER BY / LIMIT post-plan over the result. ---
        let post = match &cd.post {
            None => return Ok(QueryHandle::from_record_batch(result)),
            Some(p) => p,
        };
        use crate::exec::streaming::TableSource;
        let name = crate::plan::sql_frontend::MULTI_AGG_GROUPBY_RESULT_TABLE;
        if self.tables.contains_key(name) || self.streaming_sources.borrow().contains_key(name) {
            return Err(BoltError::Plan(format!(
                "multi-agg GROUP BY: reserved result table name '{name}' collides \
                 with a registered table"
            )));
        }
        self.gpu_tables.borrow_mut().remove(name);
        self.streaming_sources
            .borrow_mut()
            .insert(name.to_string(), TableSource::Materialized(vec![result]));
        let out = self.run_subplan(post.clone());
        self.streaming_sources.borrow_mut().remove(name);
        self.gpu_tables.borrow_mut().remove(name);
        Ok(QueryHandle::from_record_batch(out?))
    }
}
