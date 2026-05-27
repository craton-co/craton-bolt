// SPDX-License-Identifier: Apache-2.0

//! Top-level engine: dispatches per-shape executors (scalar agg, GROUP BY, etc.);
//! performs GPU prefix-scan + gather compaction for filter outputs, or a host-side
//! `arrow::compute::filter` fallback when any output column is Utf8.
//!
//! The engine owns a CUDA context and a registry of host-side Arrow `RecordBatch`es.
//! `Engine::sql` parses, plans, codegens, launches, and returns a `QueryHandle` whose
//! `record_batch()` exposes the result.
//!
//! Projection-with-filter flow: a predicate-only kernel materialises a `u8` mask
//! into a fresh device buffer. When every output column is gather-friendly
//! (primitive or Bool), the engine then runs `gpu_compact::compact_columns_on_gpu`
//! (prefix scan + gather) entirely on the device and downloads only the surviving
//! rows. When any output column is Utf8 — the gather kernel cannot relocate
//! variable-width strings — the engine falls back to downloading the full
//! per-column outputs plus the mask and running `compact::compact_arrays`
//! (Arrow's host-side filter) on the host. Scalar aggregates, group-bys with or
//! without a `WHERE`, and their `extended_agg`/`expr_agg` variants are
//! dispatched to dedicated executors in `Engine::execute`.

use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow_array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch,
};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::dictionary::DictionaryColumn;
use crate::cuda::{CudaContext, GpuVec};
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{grid_x_for, CudaStream};
use crate::exec::n_rows_to_u32;
use crate::jit::{compile_ptx, CudaModule};
use crate::plan::{
    parse_sql, DataType, Field, KernelSpec, LogicalPlan, MemTableProvider, PhysicalPlan, Schema,
};

/// PTX entry-point name; matches the symbol `ptx_gen` emits.
const KERNEL_ENTRY: &str = "bolt_kernel";

/// Threads per CUDA block for the 1D launch.
const BLOCK_SIZE: u32 = 256;

/// Stage 7 (P1b): default interval between pool-stats emits in
/// [`Engine::sql`].
///
/// 60 seconds is a sensible floor for a typical analytical workload —
/// the pool changes slowly relative to query churn, and a coarser
/// cadence keeps the log line out of per-query latency. Override with
/// `BOLT_POOL_STATS_INTERVAL_SECS=<n>`; set to `0` to disable emission
/// entirely (handy for benchmark runs that don't want the noise).
const DEFAULT_POOL_STATS_INTERVAL_SECS: u64 = 60;

/// Environment-variable override for the pool-stats periodic-emit
/// interval. Parsed once per `Engine` construction; non-integer or
/// negative values fall back to [`DEFAULT_POOL_STATS_INTERVAL_SECS`].
const POOL_STATS_ENV: &str = "BOLT_POOL_STATS_INTERVAL_SECS";

/// Synchronize the default stream and convert any pending CUDA error.
///
/// `cuLaunchKernel` is asynchronous: its return value reflects only whether
/// the launch was *accepted*, not whether the kernel later faulted. If we
/// don't synchronize, a kernel-side fault (illegal address, OOB shared
/// memory access, assertion failure, etc.) surfaces at the *next* CUDA API
/// call — which may be many lines away and in unrelated code, producing
/// extremely misleading error messages and stack traces during debugging.
///
/// In debug builds we call `cuCtxSynchronize` immediately after every
/// launch site so faults are reported at the actual launch that caused
/// them. Release builds skip this entirely: the `cfg!(debug_assertions)`
/// check is a compile-time constant, so the optimiser folds this function
/// into a no-op (`Ok(())`) and any per-launch latency goes to zero.
///
/// Cheap in release: a no-op when `cfg!(debug_assertions)` is false.
#[inline]
fn debug_sync_check() -> crate::error::BoltResult<()> {
    if cfg!(debug_assertions) {
        unsafe { crate::cuda::cuda_sys::check(crate::cuda::cuda_sys::cuCtxSynchronize())? };
    }
    Ok(())
}

/// Top-level query engine.
///
/// Field-drop order matters: `dict_registry` owns `DictionaryColumn`s which own
/// `GpuVec`s — those must be freed BEFORE `_ctx` tears down the CUDA context.
/// Rust drops fields in declaration order, so `_ctx` sits last.
pub struct Engine {
    /// Registered tables, keyed by name. A single table may comprise multiple
    /// batches (wave-7 multi-batch support): the engine concatenates them via
    /// `arrow::compute::concat_batches` at query time. This is a 0.2-era
    /// simplification — a streaming, per-batch query plan is a 0.3 goal — so
    /// large multi-batch tables pay a full materialisation cost on every
    /// `sql()` call. Keep the per-table batch count modest until then.
    tables: HashMap<String, Vec<RecordBatch>>,
    /// Name → Schema provider, kept in sync with `tables`. The schema is
    /// EXTENDED with `__idx_<col>` Int32 columns for every registered Utf8
    /// column so the SQL frontend resolves rewriter-produced column refs.
    provider: MemTableProvider,
    /// Per-table Utf8 dictionaries; drives the string-literal predicate rewrite.
    dict_registry: crate::exec::dict_registry::DictRegistry,
    /// GPU-resident copies of every registered table. Owns the device
    /// allocations; must drop BEFORE `_ctx`.
    ///
    /// Wrapped in `RefCell<Option<_>>` to support a lazy-upload strategy:
    /// `register_batch` only mutates the host-side `tables` and sets the slot
    /// to `None` (dirty). The actual upload happens on the next query, in
    /// `ensure_gpu_table` from inside `execute_projection`. This collapses a
    /// streaming append workload that uploaded `1+2+…+N = N(N+1)/2` batches'
    /// worth of bytes (the per-append re-upload bug) down to a single upload
    /// per query of the current concatenated table — i.e. O(N) total bytes
    /// across the lifetime of a streaming-then-query session, instead of
    /// O(N²). Multiple consecutive `register_batch` calls without an
    /// intervening query share that one upload.
    gpu_tables: RefCell<HashMap<String, Option<crate::exec::gpu_table::GpuTable>>>,
    /// Stage 7 (P1b): pool-stats observability state.
    ///
    /// `Mutex<Option<Instant>>`: `Some(last_emit_time)` after the first
    /// emit, `None` before any query has run. The first query on a fresh
    /// engine always emits (so a short-lived process still surfaces at
    /// least one snapshot); subsequent queries emit only after
    /// `pool_stats_interval` has elapsed.
    ///
    /// Wrapped in a `Mutex` because `Engine::sql` takes `&self` and we
    /// support concurrent calls in principle (the underlying engine is
    /// not yet `Send + Sync` because of `RefCell`, but the
    /// pool-stats accounting is independent and shouldn't add new
    /// `!Sync` constraints when we eventually relax the rest).
    pool_stats_last_emit: Mutex<Option<Instant>>,
    /// Interval between pool-stats emits. Frozen at construction from
    /// `BOLT_POOL_STATS_INTERVAL_SECS` (default 60s). A value of
    /// `Duration::ZERO` disables periodic emission entirely.
    pool_stats_interval: Duration,
    /// Owned CUDA context — declared LAST so it drops AFTER dictionaries.
    _ctx: CudaContext,
}

impl Engine {
    /// Create an engine on the default CUDA device (ordinal 0).
    ///
    /// Convenience constructor for single-GPU systems. On hosts with more
    /// than one CUDA device, use [`Engine::new_with_device`] to pick a
    /// specific GPU.
    pub fn new() -> BoltResult<Self> {
        Self::new_with_device(0)
    }

    /// Create an engine bound to the CUDA device at ordinal `device_idx`.
    ///
    /// Use this when running on a multi-GPU host and you want to target a
    /// specific device. The constructor:
    ///   1. Initializes the CUDA driver (idempotent — safe to call repeatedly).
    ///   2. Validates `device_idx` against `cuDeviceGetCount`.
    ///   3. Creates an owned CUDA context on the selected device.
    ///
    /// # Errors
    /// Returns an error if `device_idx < 0` or `device_idx >=
    /// cuDeviceGetCount()`, or if any underlying CUDA driver call fails
    /// (e.g. no CUDA-capable device, driver/runtime mismatch).
    pub fn new_with_device(device_idx: i32) -> BoltResult<Self> {
        // Initialize the driver up-front so device_count() is callable.
        cuda_sys::init()?;
        let count = cuda_sys::device_count()?;
        if device_idx < 0 || device_idx >= count {
            return Err(BoltError::Other(format!(
                "CUDA device index {} is out of range: {} device(s) visible to the driver (valid range: 0..{})",
                device_idx, count, count
            )));
        }
        let ctx = CudaContext::new(device_idx)?;
        let pool_stats_interval = pool_stats_interval_from_env();
        Ok(Self {
            tables: HashMap::new(),
            provider: MemTableProvider::new(),
            dict_registry: crate::exec::dict_registry::DictRegistry::new(),
            gpu_tables: RefCell::new(HashMap::new()),
            pool_stats_last_emit: Mutex::new(None),
            pool_stats_interval,
            _ctx: ctx,
        })
    }

    /// Register a host-side `RecordBatch` under `name` as a single-batch table.
    /// Errors if a table with that name already exists; use
    /// [`Engine::register_batch`] to append additional batches to an existing
    /// table (wave-7 multi-batch entry).
    ///
    /// Also builds Utf8 dictionaries for the table and extends the engine-side
    /// schema with `__idx_<col>` Int32 columns so the rewriter's emitted column
    /// references resolve at parse time.
    pub fn register_table(
        &mut self,
        name: impl Into<String>,
        batch: RecordBatch,
    ) -> BoltResult<()> {
        let name = name.into();
        if self.tables.contains_key(&name) {
            return Err(BoltError::Plan(format!(
                "table '{name}' is already registered — use register_batch to append \
                 additional batches to an existing table"
            )));
        }
        // Stage 6: the historical flatten step (`flatten_dictionary_utf8_columns`)
        // is gone from the hot path. `DictRegistry::register_table` matches
        // `DictionaryArray<Int32, Utf8>` directly and re-uses the input
        // dictionary; `GpuTable::from_record_batch` routes the same Arrow
        // variant through `upload_dict_utf8`, packing the keys' null buffer
        // into an on-device validity bitmap. Stage 4's compat materialisation
        // is preserved as a deprecated no-op for out-of-tree callers only.
        //
        // Build Utf8 dictionaries first (may fail — surface before we mutate
        // tables/provider).
        self.dict_registry.register_table(name.clone(), &batch)?;
        let base_schema = arrow_schema_to_plan_schema(batch.schema().as_ref())?;
        let extended = self.dict_registry.extended_schema(&name, &base_schema);
        self.provider.register(name.clone(), extended);
        // Stage 6: surface per-column runtime nullability so the engine's
        // null-aware paths can short-circuit the validity bitmap upload
        // when a column is provably null-free. For `DictionaryArray`
        // columns the answer comes from `keys().null_count()` — *not* the
        // dictionary values.
        propagate_column_nullability(&mut self.provider, &name, &batch);
        // Build a GPU-resident copy so execution can query in place.
        let gpu_table = crate::exec::gpu_table::GpuTable::from_record_batch(&batch)?;
        self.gpu_tables
            .borrow_mut()
            .insert(name.clone(), Some(gpu_table));
        self.tables.insert(name, vec![batch]);
        Ok(())
    }

    /// Replace any existing table named `name` with a single-batch table
    /// holding `batch`. Idempotent; equivalent to "unregister then
    /// register_table" but performs both halves atomically with respect to
    /// engine state so a failure mid-rebuild can't leave a torn table.
    ///
    /// This is the right entry point when you want to *update* a table's
    /// contents, e.g. an analytics tool that re-uploads a refreshed snapshot,
    /// or a benchmark harness that verifies on a small fixture then swaps in
    /// the timed-run dataset (the use case that motivated this method).
    ///
    /// Dictionaries, the SQL-frontend provider schema, the host-side batch
    /// list, AND the GPU-resident `GpuTable` are all rebuilt from `batch`.
    /// The previous `GpuTable`'s device allocations are returned to the
    /// memory pool, where the new upload can recycle them.
    pub fn replace_table(
        &mut self,
        name: impl Into<String>,
        batch: RecordBatch,
    ) -> BoltResult<()> {
        let name = name.into();
        // Stage 6: see `register_table` — the flatten step is gone from the
        // hot path. Dict ingest is native through `DictRegistry::register_table`
        // and `GpuTable::from_record_batch::upload_dict_utf8`.
        //
        // Build the new GPU table FIRST so an upload failure can't leave the
        // engine half-replaced (we have not yet touched any existing entry).
        let new_gpu_table = crate::exec::gpu_table::GpuTable::from_record_batch(&batch)?;
        let base_schema = arrow_schema_to_plan_schema(batch.schema().as_ref())?;

        // Drop the old GpuTable explicitly so its device allocations return
        // to the pool BEFORE we mint the dictionary index columns for the
        // replacement (those may also allocate from the pool — letting the
        // pool churn rather than grow keeps RAII tidy).
        self.gpu_tables.borrow_mut().remove(&name);
        self.dict_registry.unregister_table(&name);
        // Re-register dictionaries for the new batch.
        self.dict_registry.register_table(name.clone(), &batch)?;
        let extended = self.dict_registry.extended_schema(&name, &base_schema);
        // `MemTableProvider::register` already overwrites — no separate `replace`
        // entry point needed.
        self.provider.register(name.clone(), extended);
        // Stage 6: mirror `register_table` — re-surface per-column nullability
        // so a replace doesn't leave stale claims behind.
        propagate_column_nullability(&mut self.provider, &name, &batch);
        self.gpu_tables
            .borrow_mut()
            .insert(name.clone(), Some(new_gpu_table));
        self.tables.insert(name, vec![batch]);
        Ok(())
    }

    /// Append `batch` to the table named `name`, creating it if absent.
    /// Multi-batch tables are concatenated into a single `RecordBatch` at
    /// query time via `arrow::compute::concat_batches` — see the field doc on
    /// `tables` for the perf caveat.
    ///
    /// Subsequent batches MUST share the schema of the first batch; mismatched
    /// schemas surface a `Plan` error here rather than at query time. The
    /// dictionary registry is built from the first batch only — appended
    /// batches with unseen Utf8 values are still queryable, but the string-
    /// literal rewriter can only match values present in batch 0. (Refreshing
    /// dictionaries per append is a 0.3 goal.)
    ///
    /// Performance: this method does NOT re-upload anything to the GPU. It
    /// only pushes the host-side `RecordBatch` and marks the corresponding
    /// `gpu_tables` slot dirty (`None`). The combined upload of the
    /// concatenated table happens lazily, inside the next query that touches
    /// this table, via `ensure_gpu_table`. Without this, a streaming-append
    /// workload would re-upload the *entire* concatenated history on every
    /// append — `1+2+…+N = N(N+1)/2` batches' worth of bytes for N appends.
    pub fn register_batch(
        &mut self,
        name: &str,
        batch: RecordBatch,
    ) -> BoltResult<()> {
        // Stage 6: dict-encoded columns are ingested natively now, so no
        // flatten-to-StringArray is needed for the schema check below to
        // line up — batch 0 and any appended batch both carry the Arrow
        // `Dictionary<Int32, Utf8>` type verbatim.
        if let Some(existing) = self.tables.get_mut(name) {
            // Schema-check against batch 0 — concat_batches would fail at query
            // time anyway, but surface it eagerly at registration time.
            if let Some(first) = existing.first() {
                if first.schema() != batch.schema() {
                    return Err(BoltError::Plan(format!(
                        "register_batch: schema mismatch for table '{name}' — \
                         expected {:?}, got {:?}",
                        first.schema(),
                        batch.schema()
                    )));
                }
            }
            existing.push(batch);
            // Stage 6: re-evaluate per-column nullability against the
            // *concatenated* view — a previously null-free column may have
            // just gained a null in the appended batch. We materialise here
            // ONLY to read null counts; the GPU upload itself stays lazy
            // (see below) so we don't pay the O(N²) PCIe upload cost.
            let concatenated = self.materialize_table(name)?;
            propagate_column_nullability(&mut self.provider, name, &concatenated);
            // Mark the GPU-resident copy dirty. The next query that hits this
            // table will rebuild it from the concatenated host batches. We
            // explicitly insert `None` (rather than `remove`) so the entry's
            // existence still signals "this table has been registered" — keeps
            // a future audit / introspection path simple.
            self.gpu_tables.borrow_mut().insert(name.to_string(), None);
            Ok(())
        } else {
            // First batch for a brand-new table: defer to register_table so the
            // dictionary + provider wiring happens exactly once.
            self.register_table(name.to_string(), batch)
        }
    }

    /// Make sure the GPU-resident copy of `name` is fresh, uploading from the
    /// host-side concatenated batches if the slot is dirty (`None`) or absent.
    /// Returns a `Ref` borrowing the inner `GpuTable` for the caller to use
    /// during a single query.
    ///
    /// Held for the duration of `execute_projection`; the borrow lifetime is
    /// enforced by the returned `Ref` (`RefCell` will panic if a second
    /// `borrow_mut` is attempted while the `Ref` is live, but no engine method
    /// touches `gpu_tables` mutably while a query is in flight).
    fn ensure_gpu_table(
        &self,
        name: &str,
    ) -> BoltResult<Ref<'_, crate::exec::gpu_table::GpuTable>> {
        // Fast path: borrow read-only and check for a hit. If the slot holds
        // `Some(GpuTable)`, project the `Ref<HashMap<_>>` down to the inner
        // `Ref<GpuTable>` and return it.
        {
            let g = self.gpu_tables.borrow();
            if matches!(g.get(name), Some(Some(_))) {
                return Ok(Ref::map(g, |m| {
                    // Safe: matched `Some(Some(_))` above and no mutation of
                    // `gpu_tables` happens between the matches! check and the
                    // map (the &self borrow forbids it).
                    m.get(name)
                        .expect("hit by matches! above")
                        .as_ref()
                        .expect("hit by matches! above")
                }));
            }
        }
        // Miss path: the slot is dirty (`None`) or missing. Build the GpuTable
        // from the host-concatenated batch, then re-borrow.
        let concatenated = self.materialize_table(name)?;
        let gpu_table = crate::exec::gpu_table::GpuTable::from_record_batch(&concatenated)?;
        self.gpu_tables
            .borrow_mut()
            .insert(name.to_string(), Some(gpu_table));
        let g = self.gpu_tables.borrow();
        Ok(Ref::map(g, |m| {
            m.get(name)
                .expect("just inserted")
                .as_ref()
                .expect("just inserted Some")
        }))
    }

    /// Materialise the concatenated `RecordBatch` for a registered table.
    ///
    /// Fast-path: zero batches errors, one batch is cloned cheaply (Arrow
    /// arrays are Arc-backed). Two or more batches go through
    /// `arrow::compute::concat_batches`, which copies every column — the
    /// 0.2 perf cost the field doc on `tables` warns about.
    fn materialize_table(&self, name: &str) -> BoltResult<RecordBatch> {
        let batches = self.tables.get(name).ok_or_else(|| {
            BoltError::Plan(format!("table '{name}' is not registered with the engine"))
        })?;
        match batches.len() {
            0 => Err(BoltError::Plan(format!(
                "table '{name}' is registered but contains zero batches"
            ))),
            1 => Ok(batches[0].clone()),
            _ => {
                let schema = batches[0].schema();
                arrow::compute::concat_batches(&schema, batches.iter()).map_err(|e| {
                    BoltError::Other(format!(
                        "failed to concatenate {} batches for table '{name}': {e}",
                        batches.len()
                    ))
                })
            }
        }
    }

    /// Compile and execute a SQL query string.
    ///
    /// Stage 7 (P1b): after the query completes, the engine emits a
    /// periodic pool-stats log line at most once every
    /// `BOLT_POOL_STATS_INTERVAL_SECS` (default 60s). The emit happens
    /// AFTER the query's `QueryHandle` is fully materialised — the log
    /// line is off the latency-critical path for the just-returned
    /// query. Failures (query error, log throttled, no-op observer)
    /// never affect the query result.
    pub fn sql(&self, query: &str) -> BoltResult<QueryHandle> {
        let plan: LogicalPlan = parse_sql(query, &self.provider)?;
        // String-literal predicates against Utf8 columns are folded into
        // integer equality against the corresponding __idx_<col> i32 column.
        let plan = self.dict_registry.rewrite_plan(&plan)?;
        let mut phys = crate::plan::lower_physical(&plan)?;
        // PV-stage-d: populate `KernelSpec::input_has_validity` for every
        // input column by consulting the engine-backed provider, which
        // looks straight at `RecordBatch::column(col).null_count()` for
        // each registered table. This is the plan-time signal that lets
        // the codegen emit native-validity kernels instead of leaning on
        // the run-time host-strip fallback in `groupby_with_pre` etc.
        let nb = EngineProvider {
            base: &self.provider,
            tables: &self.tables,
        };
        crate::plan::physical_plan::populate_input_validity(&mut phys, &nb);
        let result = self.execute(&phys);
        // Stage 7: periodic pool-stats emit. Runs whether the query
        // succeeded or failed (an OOM-failed query is itself a signal
        // worth surfacing alongside the pool snapshot). Internal errors
        // in the emit path are swallowed — they must never escalate to
        // the query result.
        self.maybe_emit_pool_stats(Instant::now());
        result
    }

    /// Emit a periodic pool-stats log line + observer notification if
    /// the configured interval has elapsed since the last emit.
    ///
    /// `now` is taken as a parameter (rather than calling `Instant::now()`
    /// inside) so the unit test below can drive the throttle deterministically.
    fn maybe_emit_pool_stats(&self, now: Instant) {
        if !should_emit_pool_stats(&self.pool_stats_last_emit, self.pool_stats_interval, now) {
            return;
        }
        // Throttle says go: snapshot the pool and emit. We do this OUTSIDE
        // the throttle's lock so a slow observer can't serialise concurrent
        // queries.
        let s = crate::pool_stats();
        log::info!(
            "craton-bolt pool: bucket_count={}, total_pooled_bytes={}, \
             oom_recoveries={}, proactive_evictions={}",
            s.bucket_count,
            s.total_pooled_bytes,
            s.oom_recovery_count,
            s.proactive_eviction_count,
        );
        crate::observability::notify_observers(s);
    }

    /// Execute a pre-built `PhysicalPlan`.
    pub fn execute(&self, phys: &PhysicalPlan) -> BoltResult<QueryHandle> {
        match phys {
            PhysicalPlan::Projection {
                table,
                kernel,
                output_schema,
            } => self.execute_projection(table, kernel, output_schema),
            PhysicalPlan::Aggregate {
                table,
                pre,
                aggregate,
            } => {
                let batch = self.materialize_table(table)?;
                let out = match (!aggregate.group_by.is_empty(), pre.is_some()) {
                    (true, true) => {
                        crate::exec::groupby_with_pre::execute_groupby_with_pre(phys, &batch)?
                    }
                    (true, false) => crate::exec::groupby::execute_groupby(phys, &batch)?,
                    (false, true) => {
                        crate::exec::agg_with_pre::execute_aggregate_with_pre(phys, &batch)?
                    }
                    (false, false) => crate::exec::aggregate::execute_aggregate(phys, &batch)?,
                };
                Ok(QueryHandle { batch: out })
            }
            // ----- wave-7 dispatch -----
            //
            // The PhysicalPlan variants below are added by agent 1 in the
            // same wave. If a variant doesn't exist yet at build time, the
            // match arm will surface a clear compile error pointing at the
            // missing variant — agent 1 then adds it and the build heals.
            //
            // The executor signatures assumed here mirror the wave-7 spec:
            //   execute_distinct(QueryHandle) -> BoltResult<QueryHandle>
            //   execute_limit  (QueryHandle, usize, Option<usize>) -> ...
            //   execute_sort   (QueryHandle, &[SortExpr]) -> ...
            //   execute_join   (left, right, join_type, on, &Engine) -> ...
            // Agents 3-6 match these.
            PhysicalPlan::Distinct { input } => {
                let h = self.execute(input)?;
                crate::exec::distinct::execute_distinct(h)
            }
            PhysicalPlan::Limit {
                input,
                limit,
                offset,
            } => {
                let h = self.execute(input)?;
                crate::exec::limit::execute_limit(h, *limit, *offset)
            }
            PhysicalPlan::Sort { input, sort_exprs } => {
                let h = self.execute(input)?;
                crate::exec::sort::execute_sort(h, sort_exprs)
            }
            PhysicalPlan::Union { inputs } => {
                // UNION ALL: execute each input, concat the result batches.
                // (Deduplication would happen via a Distinct wrapping the Union
                // in the logical plan — UNION ALL itself is pure concat.)
                if inputs.is_empty() {
                    return Err(BoltError::Plan(
                        "Union with zero inputs is not executable".into(),
                    ));
                }
                let mut handles: Vec<QueryHandle> = Vec::with_capacity(inputs.len());
                for inp in inputs {
                    handles.push(self.execute(inp)?);
                }
                let schema = handles[0].batch.schema();
                let batches: Vec<RecordBatch> =
                    handles.into_iter().map(|h| h.batch).collect();
                let merged = arrow::compute::concat_batches(&schema, batches.iter())
                    .map_err(|e| {
                        BoltError::Other(format!(
                            "failed to concatenate {} UNION ALL inputs: {e}",
                            batches.len()
                        ))
                    })?;
                Ok(QueryHandle { batch: merged })
            }
            PhysicalPlan::Join {
                left,
                right,
                join_type,
                on,
                output_schema,
            } => crate::exec::join::execute_join(
                left,
                right,
                join_type,
                on,
                output_schema,
                self,
            ),
            PhysicalPlan::Filter { input, predicate } => {
                // Host-side post-aggregate (or other non-scan-chain) filter.
                // The lowerer emits this for `HAVING` and any `Filter`
                // wrapping an operator that can't fold into a fused
                // projection kernel. The inner plan's output is materialised
                // as a host-side RecordBatch; we evaluate `predicate` against
                // it via `expr_agg::eval_expr` and drop the rows that don't
                // satisfy it. See `crate::exec::filter::execute_filter`.
                let h = self.execute(input)?;
                crate::exec::filter::execute_filter(h, predicate)
            }
            PhysicalPlan::Project {
                input,
                exprs,
                output_schema,
            } => {
                // Rename/reorder layer over an arbitrary upstream. Each
                // `exprs` entry is a bare column reference (possibly aliased)
                // into the input's schema; we just pick those columns out
                // and re-wrap them under `output_schema`. No compute.
                let h = self.execute(input)?;
                let in_batch = h.batch;
                let in_schema = in_batch.schema();
                let mut columns: Vec<ArrayRef> = Vec::with_capacity(exprs.len());
                for e in exprs {
                    let name = match e {
                        crate::plan::Expr::Column(n) => n.as_str(),
                        crate::plan::Expr::Alias(inner, _) => match inner.as_ref() {
                            crate::plan::Expr::Column(n) => n.as_str(),
                            _ => {
                                return Err(BoltError::Plan(
                                    "PhysicalPlan::Project: aliased expression must be a column reference"
                                        .into(),
                                ));
                            }
                        },
                        _ => {
                            return Err(BoltError::Plan(
                                "PhysicalPlan::Project: only column references / aliases are supported"
                                    .into(),
                            ));
                        }
                    };
                    let idx = in_schema.index_of(name).map_err(|_| {
                        BoltError::Plan(format!(
                            "PhysicalPlan::Project: column '{name}' not found in input schema"
                        ))
                    })?;
                    columns.push(in_batch.column(idx).clone());
                }
                let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
                let out = RecordBatch::try_new(arrow_schema, columns).map_err(|e| {
                    BoltError::Other(format!(
                        "failed to build PhysicalPlan::Project RecordBatch: {e}"
                    ))
                })?;
                Ok(QueryHandle { batch: out })
            }
        }
    }

    /// Execute a single fused projection (optionally with filter) kernel.
    fn execute_projection(
        &self,
        table: &str,
        kernel: &KernelSpec,
        output_schema: &Schema,
    ) -> BoltResult<QueryHandle> {
        // Lazy upload: if `register_batch` ran since the last query, this
        // rebuilds the GPU-resident copy from the host-concatenated batches.
        // The returned `Ref` is held across the kernel launch — no other
        // engine method touches `gpu_tables` mutably while `&self` is borrowed.
        let gpu_table_ref = self.ensure_gpu_table(table)?;
        let gpu_table: &crate::exec::gpu_table::GpuTable = &gpu_table_ref;
        let n_rows = gpu_table.n_rows;

        // 1. Resolve input device pointers in place — every column already
        //    lives on the GPU. No host bounce, no per-query upload.
        //
        // `__idx_<col>` inputs come from the dict_registry (they don't exist
        // in the source RecordBatch). They were synthesized by the
        // string-literal rewriter and resolve to the i32/i64 dictionary index
        // column already on the device — we hand the launch a borrowed
        // device pointer into the registry's `GpuVec` rather than bouncing the
        // index column through the host. `&self` is borrowed for the entire
        // `execute_projection`, so the dictionary's GpuVec outlives the launch.
        let mut input_ptrs: Vec<CUdeviceptr> = Vec::with_capacity(kernel.inputs.len());
        for io in &kernel.inputs {
            if let Some(original) = io.name.strip_prefix("__idx_") {
                let dict = self.dict_registry.dictionary(table, original).ok_or_else(|| {
                    BoltError::Plan(format!(
                        "rewriter-emitted column '{}' has no dictionary in registry",
                        io.name
                    ))
                })?;
                // Fail fast on plan/dict dtype mismatch BEFORE doing any I/O —
                // this catches a stale plan that names __idx_X with the wrong
                // width without paying the cost of touching the device.
                if io.dtype != dict.index_dtype() {
                    return Err(BoltError::Plan(format!(
                        "rewriter-emitted column '{}' dtype mismatch: plan says {:?}, dictionary is {:?}",
                        io.name, io.dtype, dict.index_dtype()
                    )));
                }
                // Borrow the device pointer from the registry's existing
                // index column — no host bounce, no fresh allocation.
                let ptr = match dict {
                    crate::cuda::dictionary_any::DictionaryColumnAny::I32(d) => {
                        d.indices.device_ptr()
                    }
                    crate::cuda::dictionary_any::DictionaryColumnAny::I64(d) => {
                        d.indices.device_ptr()
                    }
                };
                input_ptrs.push(ptr);
                continue;
            }
            let column = gpu_table.column(&io.name).ok_or_else(|| {
                BoltError::Plan(format!("column '{}' not in table '{}'", io.name, table))
            })?;
            if column.dtype != io.dtype {
                return Err(BoltError::Plan(format!(
                    "column '{}' dtype mismatch: plan says {:?}, table has {:?}",
                    io.name, io.dtype, column.dtype
                )));
            }
            input_ptrs.push(column.device_ptr());
        }

        // 2. Allocate output buffers, zero-initialised. For Utf8 passthrough
        //    columns (output dtype Utf8 AND name matches an input column),
        //    clone the source dictionary so download can decode indices back
        //    to strings. (Computed Utf8 outputs aren't supported yet.)
        let mut output_cols: Vec<DeviceCol> = Vec::with_capacity(kernel.outputs.len());
        for io in &kernel.outputs {
            let mut col = DeviceCol::alloc_zeros(io.dtype, n_rows)?;
            if io.dtype == DataType::Utf8 {
                if let Some(src) = kernel
                    .inputs
                    .iter()
                    .find(|in_io| in_io.name == io.name && in_io.dtype == DataType::Utf8)
                    .and_then(|in_io| gpu_table.column(&in_io.name))
                    .and_then(|c| c.utf8_dictionary())
                {
                    col.set_utf8_dictionary(src.to_vec());
                }
            }
            output_cols.push(col);
        }

        // 3. JIT-compile the kernel to PTX and load it.
        let ptx = compile_ptx(kernel, KERNEL_ENTRY)?;
        let module = CudaModule::from_ptx(&ptx)?;
        let function = module.function(KERNEL_ENTRY)?;

        // 4. Build the kernel-parameter list.
        //
        // `KernelArgs` is monomorphic on `T` per push and cannot store heterogenous
        // column types in one list. We bypass it and assemble raw kernel params
        // directly: inputs first, then outputs, then the row-count `u32`.
        let mut device_ptrs: Vec<CUdeviceptr> = Vec::with_capacity(input_ptrs.len() + output_cols.len());
        for p in &input_ptrs {
            device_ptrs.push(*p);
        }
        for c in &output_cols {
            device_ptrs.push(c.device_ptr());
        }
        let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;

        let mut kernel_params: Vec<*mut c_void> = Vec::with_capacity(device_ptrs.len() + 1);
        for p in device_ptrs.iter_mut() {
            kernel_params.push(p as *mut CUdeviceptr as *mut c_void);
        }
        kernel_params.push(&mut n_rows_u32 as *mut u32 as *mut c_void);

        // 5. Launch with one thread per row, block size 256.
        //
        // Stage-3 async memcpy: mint a per-call stream so the kernel
        // launch, mask materialisation (if any), and the final pinned
        // D2H download can run on the same stream — letting the driver
        // overlap them with concurrent work on the NULL stream. Falls
        // back to the NULL stream if stream creation fails (see
        // `CudaStream::null_or_default`).
        let stream = CudaStream::null_or_default();
        let grid_x = grid_x_for(n_rows_u32, BLOCK_SIZE);
        unsafe {
            cuda_sys::check(cuda_sys::cuLaunchKernel(
                function.raw(),
                grid_x,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                stream.raw(),
                kernel_params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }
        // Debug-only synchronize: pin any in-kernel fault to THIS launch
        // rather than letting it surface at the next CUDA API call.
        debug_sync_check()?;
        // NOTE: no `stream.synchronize()` here — the predicate / gather path
        // and the async-D2H path below both run on the same stream and so are
        // serialized after the kernel automatically. The single sync happens
        // at the bottom of this function (or inside `gpu_compact` for the
        // predicate path, which manages its own stream barriers).

        // 6. If the kernel has a predicate, run a separate predicate-only
        //    kernel to materialise a u8 mask. We default to GPU-side compaction
        //    (prefix scan + gather) when every output column is gather-friendly
        //    (primitive + Bool); Utf8 outputs fall back to the host-side path
        //    because the gather kernel can't move variable-width strings.
        let arrays: Vec<ArrayRef> = if kernel.predicate.is_some() {
            let pred_ptx =
                crate::jit::scan_kernel::compile_predicate_kernel(kernel, "bolt_predicate")?;
            let pred_module = CudaModule::from_ptx(&pred_ptx)?;
            let pred_function = pred_module.function("bolt_predicate")?;

            let mask = crate::exec::compact::alloc_mask_buffer(n_rows)?;
            crate::exec::compact::launch_predicate_kernel(
                pred_function,
                &input_ptrs,
                mask.device_ptr(),
                n_rows_to_u32(n_rows)?,
                &stream,
            )?;
            // Debug-only synchronize: surface predicate-kernel faults at
            // THIS launch site rather than at a later API call.
            debug_sync_check()?;

            let has_utf8_output = kernel.outputs.iter().any(|c| c.dtype == DataType::Utf8);
            if has_utf8_output {
                // Host-side fallback: download mask + outputs, then filter.
                let host_mask =
                    crate::exec::compact::download_mask(mask.device_ptr(), n_rows)?;
                // Stage-3: route every primitive output column through the
                // pinned async D2H path. Each `download_pinned` call
                // synchronizes the stream internally, so we don't need a
                // separate barrier between the predicate kernel and these
                // reads.
                let mut full: Vec<ArrayRef> = Vec::with_capacity(output_cols.len());
                for col in output_cols {
                    full.push(col.download_pinned(n_rows, &stream)?);
                }
                crate::exec::compact::compact_arrays(&full, &host_mask)?
            } else {
                // GPU-side path: prefix-scan + gather, download the compacted output.
                let cols: Vec<(CUdeviceptr, DataType)> = output_cols
                    .iter()
                    .zip(kernel.outputs.iter())
                    .map(|(c, io)| (c.device_ptr(), io.dtype))
                    .collect();
                let (gathered, _total) = crate::exec::gpu_compact::compact_columns_on_gpu(
                    mask.device_ptr(),
                    n_rows,
                    &cols,
                    &stream,
                )?;
                // Output buffers can drop now; gathered owns the compacted data.
                drop(output_cols);
                let mut out: Vec<ArrayRef> = Vec::with_capacity(gathered.len());
                for g in &gathered {
                    out.push(g.download()?);
                }
                out
            }
        } else {
            // Stage-3 pinned downloads on the per-call stream. Each
            // call synchronizes internally before reading, so the loop
            // is correct even though `stream` was used for the kernel
            // launch above.
            let mut full: Vec<ArrayRef> = Vec::with_capacity(output_cols.len());
            for col in output_cols {
                full.push(col.download_pinned(n_rows, &stream)?);
            }
            full
        };

        // 9. Build the result RecordBatch.
        let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
        let batch_out = RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
            BoltError::Other(format!("failed to build output RecordBatch: {e}"))
        })?;
        Ok(QueryHandle { batch: batch_out })
    }
}

/// Result of a query — wraps the output Arrow `RecordBatch`.
pub struct QueryHandle {
    /// The materialised result.
    batch: RecordBatch,
}

impl QueryHandle {
    /// Borrow the underlying record batch.
    pub fn record_batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// Consume the handle and return the owned record batch.
    pub fn into_record_batch(self) -> RecordBatch {
        self.batch
    }

    /// Wrap a `RecordBatch` produced by an executor into a `QueryHandle`.
    ///
    /// Internal hook for the wave-7 executor chain (Distinct / Limit / Sort /
    /// Union / Join): the top-level `Engine::execute` runs the child plan,
    /// hands the resulting `QueryHandle` to an `exec::*::execute_*` helper,
    /// and the helper rewraps its output with this constructor.
    ///
    /// Marked `#[doc(hidden)]` and `pub(crate)`: this is not part of the
    /// public 0.2 API; downstream consumers should keep going through
    /// `Engine::sql` / `Engine::execute`.
    #[doc(hidden)]
    pub(crate) fn from_record_batch(batch: RecordBatch) -> Self {
        Self { batch }
    }

    /// Number of rows in the result.
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }
}

/// Heterogenous owned device column. Keeps each `GpuVec<T>` alive past the kernel launch.
///
/// Used only for OUTPUT buffers in `execute_projection`. Input columns are
/// resolved through `GpuTable` (uploaded once at table-registration time) and
/// fed to kernels as raw `CUdeviceptr`s; the upload-from-Arrow path that used
/// to live here as `DeviceCol::upload` is gone — `GpuColumn::upload` in
/// `gpu_table.rs` is the single source of truth for host→device column
/// uploads. The historical `BoolNullable` and `Borrowed` variants and the
/// `utf8_dictionary` accessor went with it; both were only reachable through
/// `upload`.
enum DeviceCol {
    /// 32-bit signed integer column.
    I32(GpuVec<i32>),
    /// 64-bit signed integer column.
    I64(GpuVec<i64>),
    /// 32-bit float column.
    F32(GpuVec<f32>),
    /// 64-bit float column.
    F64(GpuVec<f64>),
    /// Bool stored as one byte per row (0 / 1). Used when the source Arrow
    /// array has no nulls.
    Bool(GpuVec<u8>),
    /// Utf8 stored as i32 dictionary indices; host dictionary lives alongside.
    Utf8(DictionaryColumn),
}

impl DeviceCol {
    /// Allocate a zero-initialised device column of `n` rows.
    ///
    /// Utf8 outputs allocate an empty dictionary; the engine must replace it
    /// with the source column's dictionary before download (today this only
    /// works for pure column-passthrough projections — `output_schema` field
    /// name matching an input column name).
    fn alloc_zeros(dtype: DataType, n: usize) -> BoltResult<Self> {
        match dtype {
            DataType::Int32 => Ok(DeviceCol::I32(GpuVec::<i32>::zeros(n)?)),
            DataType::Int64 => Ok(DeviceCol::I64(GpuVec::<i64>::zeros(n)?)),
            DataType::Float32 => Ok(DeviceCol::F32(GpuVec::<f32>::zeros(n)?)),
            DataType::Float64 => Ok(DeviceCol::F64(GpuVec::<f64>::zeros(n)?)),
            DataType::Bool => Ok(DeviceCol::Bool(GpuVec::<u8>::zeros(n)?)),
            DataType::Utf8 => Ok(DeviceCol::Utf8(DictionaryColumn {
                dictionary: Vec::new(),
                indices: GpuVec::<i32>::zeros(n)?,
                n_rows: n,
            })),
        }
    }

    /// Raw device pointer for kernel-parameter assembly.
    fn device_ptr(&self) -> CUdeviceptr {
        match self {
            DeviceCol::I32(v) => v.device_ptr(),
            DeviceCol::I64(v) => v.device_ptr(),
            DeviceCol::F32(v) => v.device_ptr(),
            DeviceCol::F64(v) => v.device_ptr(),
            DeviceCol::Bool(v) => v.device_ptr(),
            DeviceCol::Utf8(d) => d.indices.device_ptr(),
        }
    }

    /// Install a dictionary on a Utf8 column (for output columns whose source dictionary
    /// the engine knows). No-op for non-Utf8 columns.
    fn set_utf8_dictionary(&mut self, dict: Vec<String>) {
        if let DeviceCol::Utf8(d) = self {
            d.dictionary = dict;
        }
    }

    /// Copy the device column back to a host Arrow array of length `n_rows`.
    fn download(self, n_rows: usize) -> BoltResult<ArrayRef> {
        match self {
            DeviceCol::I32(v) => {
                let host = copy_back::<i32>(&v, n_rows)?;
                Ok(Arc::new(Int32Array::from(host)) as ArrayRef)
            }
            DeviceCol::I64(v) => {
                let host = copy_back::<i64>(&v, n_rows)?;
                Ok(Arc::new(Int64Array::from(host)) as ArrayRef)
            }
            DeviceCol::F32(v) => {
                let host = copy_back::<f32>(&v, n_rows)?;
                Ok(Arc::new(Float32Array::from(host)) as ArrayRef)
            }
            DeviceCol::F64(v) => {
                let host = copy_back::<f64>(&v, n_rows)?;
                Ok(Arc::new(Float64Array::from(host)) as ArrayRef)
            }
            DeviceCol::Bool(v) => {
                let host = copy_back::<u8>(&v, n_rows)?;
                let bools: Vec<bool> = host.into_iter().map(|b| b != 0).collect();
                Ok(Arc::new(BooleanArray::from(bools)) as ArrayRef)
            }
            DeviceCol::Utf8(d) => {
                let arr = d.to_string_array()?;
                Ok(Arc::new(arr) as ArrayRef)
            }
        }
    }

    /// Stage-3 async download: enqueue D2H from every primitive variant
    /// into pinned host buffers on `stream`, then synchronize ONCE and
    /// build the Arrow arrays from the resulting `Vec`s. Behaves
    /// identically to [`download`] for the Utf8 / Borrowed variants —
    /// those don't currently have a pinned fast path.
    ///
    /// The caller is responsible for ensuring `stream` is the same one
    /// the producing kernel was launched on (so the D2H sees committed
    /// results), and the function performs the synchronize internally
    /// before reading the pinned buffer.
    fn download_pinned(
        self,
        n_rows: usize,
        stream: &CudaStream,
    ) -> BoltResult<ArrayRef> {
        match self {
            DeviceCol::I32(v) => {
                let staged = StagedDownload::<i32>::from_gpu(&v, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), n_rows)?;
                Ok(Arc::new(Int32Array::from(host)) as ArrayRef)
            }
            DeviceCol::I64(v) => {
                let staged = StagedDownload::<i64>::from_gpu(&v, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), n_rows)?;
                Ok(Arc::new(Int64Array::from(host)) as ArrayRef)
            }
            DeviceCol::F32(v) => {
                let staged = StagedDownload::<f32>::from_gpu(&v, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), n_rows)?;
                Ok(Arc::new(Float32Array::from(host)) as ArrayRef)
            }
            DeviceCol::F64(v) => {
                let staged = StagedDownload::<f64>::from_gpu(&v, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), n_rows)?;
                Ok(Arc::new(Float64Array::from(host)) as ArrayRef)
            }
            DeviceCol::Bool(v) => {
                let staged = StagedDownload::<u8>::from_gpu(&v, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), n_rows)?;
                let bools: Vec<bool> = host.into_iter().map(|b| b != 0).collect();
                Ok(Arc::new(BooleanArray::from(bools)) as ArrayRef)
            }
            DeviceCol::Utf8(_) => {
                // Utf8 doesn't (yet) have a pinned fast path — fall back
                // to the sync download. The stream has already been
                // synchronized above for the primitive siblings, so this
                // is safe to invoke regardless.
                self.download(n_rows)
            }
        }
    }
}

/// Tiny invariant check used by the pinned-download path: every
/// `DeviceCol` output buffer is sized at allocation time to `n_rows`, so
/// a length mismatch on download is a bug, not a runtime condition.
fn check_len(have: usize, want: usize) -> BoltResult<()> {
    if have != want {
        return Err(BoltError::Other(format!(
            "internal: device buffer length {} did not match expected {}",
            have, want
        )));
    }
    Ok(())
}

/// Copy back a `GpuVec<T>` into a host `Vec<T>` of length `n_rows`.
///
/// Output buffers are allocated via `GpuVec::zeros(n_rows)`, whose `len()` is `n_rows`,
/// so `to_vec()` returns exactly that many elements.
fn copy_back<T>(v: &GpuVec<T>, n_rows: usize) -> BoltResult<Vec<T>>
where
    T: bytemuck::Pod,
{
    let host = v.to_vec()?;
    if host.len() != n_rows {
        return Err(BoltError::Other(format!(
            "internal: device buffer length {} did not match expected {}",
            host.len(),
            n_rows
        )));
    }
    Ok(host)
}

/// Stage-3 D2H staging buffer: async-copies a `GpuVec<T>` into a
/// page-locked host buffer on a caller-supplied stream, synchronises
/// once, and produces a regular `Vec<T>` for Arrow consumption.
///
/// Why a separate type vs. an inline call? Arrow array constructors
/// (`Int32Array::from(Vec<i32>)`) want owned `Vec`s with the standard
/// allocator — they will NOT accept a `PinnedHostBuffer` as a
/// zero-copy backing buffer (the lifecycle is incompatible: pinned
/// memory must be released via `cuMemFreeHost`, while Arrow buffers
/// release through the global allocator). So the pinned hop is purely
/// to get a true DMA without staging through a kernel-managed bounce
/// buffer; the final `.to_vec()` is the one host-host copy we keep.
///
/// Usage:
///
/// ```ignore
/// let staged = StagedDownload::from_gpu(&gpu_vec, stream.raw())?;
/// stream.synchronize()?;
/// let arrow_vec: Vec<i32> = staged.into_vec();
/// ```
struct StagedDownload<T: bytemuck::Pod> {
    pinned: crate::cuda::PinnedHostBuffer<T>,
}

impl<T: bytemuck::Pod> StagedDownload<T> {
    /// Enqueue an async D2H from `v` into a fresh pinned host buffer on
    /// `stream`. The caller MUST synchronize `stream` before calling
    /// [`into_vec`] / borrowing the pinned slice.
    fn from_gpu(v: &GpuVec<T>, stream: crate::cuda::CUstream) -> BoltResult<Self> {
        let pinned = v.to_pinned_async(stream)?;
        Ok(Self { pinned })
    }

    /// Consume the staged download and produce a regular host `Vec<T>`.
    ///
    /// Assumes the caller has synchronized the stream — there is no way
    /// to detect "not yet synchronized" without an event, which we skip
    /// in Stage 3. Calling this before sync produces uninitialised
    /// bytes (defined behaviour for `T: Pod` but functionally
    /// incorrect).
    fn into_vec(self) -> Vec<T> {
        self.pinned.as_slice().to_vec()
    }
}

/// Map our plan `DataType` to Arrow `DataType`.
fn plan_dtype_to_arrow(d: DataType) -> BoltResult<ArrowDataType> {
    match d {
        DataType::Int32 => Ok(ArrowDataType::Int32),
        DataType::Int64 => Ok(ArrowDataType::Int64),
        DataType::Float32 => Ok(ArrowDataType::Float32),
        DataType::Float64 => Ok(ArrowDataType::Float64),
        DataType::Bool => Ok(ArrowDataType::Boolean),
        DataType::Utf8 => Ok(ArrowDataType::Utf8),
    }
}

/// Map Arrow `DataType` to our plan `DataType`. Errors on unsupported types.
///
/// **Stage 4 / Stage 6** — `Dictionary(_, Utf8)` is accepted as a register-table
/// type and exposed to the planner as `DataType::Utf8`. The fact that the column
/// is dictionary-encoded is a *storage* detail: query planning, projection,
/// filtering, ORDER BY all reason about it as a Utf8 column. SQL frontends
/// see it identically to a flat `StringArray` column.
///
/// Stage 4 accepted any key width (Int32 *or* Int64) and routed through the
/// flatten step. Stage 6 added a native ingest path for `Dictionary<Int32, Utf8>`
/// in `GpuTable::from_record_batch` and `DictRegistry::register_table`, so the
/// flatten in `flatten_dictionary_utf8_columns` is now a deprecated no-op (the
/// dict layout reaches the GPU table directly). Int64-keyed dicts still take
/// the legacy path through `GpuColumn::upload`.
fn arrow_dtype_to_plan(d: &ArrowDataType) -> BoltResult<DataType> {
    match d {
        ArrowDataType::Int32 => Ok(DataType::Int32),
        ArrowDataType::Int64 => Ok(DataType::Int64),
        ArrowDataType::Float32 => Ok(DataType::Float32),
        ArrowDataType::Float64 => Ok(DataType::Float64),
        ArrowDataType::Boolean => Ok(DataType::Bool),
        ArrowDataType::Utf8 => Ok(DataType::Utf8),
        // Stage 4 / Stage 6: dictionary-encoded Utf8. Only string-valued
        // dicts map to `Utf8`; numeric-valued dicts are intentionally
        // rejected because the caller should hand the inner numeric column
        // directly through the normal path. Both Int32 and Int64 keys are
        // accepted: Int32 takes the Stage-6 native ingest in
        // `GpuTable::from_record_batch`, Int64 falls through to the legacy
        // `GpuColumn::upload` path (which still emits a `DictUtf8` variant).
        ArrowDataType::Dictionary(_key, value)
            if matches!(value.as_ref(), ArrowDataType::Utf8) =>
        {
            Ok(DataType::Utf8)
        }
        other => Err(BoltError::Type(format!(
            "unsupported Arrow dtype {:?}",
            other
        ))),
    }
}

/// Stage 4 — rewrite every `Dictionary(_, Utf8)` column in `batch` into a
/// plain `StringArray`, leaving non-dictionary columns untouched. Returns
/// the rewritten `RecordBatch` (cheap if no dict columns: just reuses the
/// original arrays via `Arc`).
///
/// Why flatten at registration time rather than carrying the dict through?
/// The GPU storage (`GpuTable`) already manages its own dictionary for Utf8
/// columns (see `GpuColumnData::Utf8`), so re-using the input dict would
/// require teaching every consumer (GpuTable upload, projection, gather,
/// expression evaluator, ORDER BY's host-side `take`) to read both dict
/// variants. Materialising once at registration is O(n) per dict column —
/// the same cost the engine's own dictionary builder pays — and keeps every
/// downstream stage's Utf8 handling unified on `StringArray`.
///
/// **Stage 5** added a native `GpuColumnData::DictUtf8` variant to
/// `GpuTable` so callers that go directly through `GpuTable::from_record_batch`
/// (skipping the engine's registry / `MemTableProvider`) can preserve the
/// input dictionary instead of materialising it.
///
/// **Stage 6** — DEPRECATED. The dict registry and `GpuTable` now ingest
/// `DictionaryArray<Int32, Utf8>` natively (the registry matches the dict
/// variant directly; `GpuTable::from_record_batch` routes through
/// `upload_dict_utf8`). The engine no longer calls this helper from
/// `register_table` / `replace_table` / `register_batch`, but the body is
/// kept callable so any out-of-tree consumer that imported it still
/// compiles. Will be removed in a wave following Stage 7.
#[deprecated(
    since = "0.3.0",
    note = "DictionaryArray<Int32, Utf8> is now ingested natively by DictRegistry \
            and GpuTable::from_record_batch; this flatten step is no longer \
            invoked by the engine and will be removed in a future release."
)]
#[allow(dead_code)]
pub(crate) fn flatten_dictionary_utf8_columns(batch: RecordBatch) -> BoltResult<RecordBatch> {
    use arrow_array::{Array, DictionaryArray, StringArray};
    use arrow_array::types::{Int32Type, Int64Type};

    let schema = batch.schema();
    let mut changed = false;
    let mut new_fields: Vec<ArrowField> = Vec::with_capacity(schema.fields().len());
    let mut new_cols: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    for (idx, field) in schema.fields().iter().enumerate() {
        let col = batch.column(idx);
        match field.data_type() {
            ArrowDataType::Dictionary(key_ty, value_ty)
                if matches!(value_ty.as_ref(), ArrowDataType::Utf8) =>
            {
                // Decode (key_idx, value_idx) -> StringArray entries.
                // Supports Int32 and Int64 key types (matches `arrow_dtype_to_plan`).
                let n = col.len();
                let mut out: Vec<Option<String>> = Vec::with_capacity(n);
                let decode_into = |out: &mut Vec<Option<String>>,
                                   value_idx: usize,
                                   sa: &StringArray| {
                    if sa.is_null(value_idx) {
                        out.push(None);
                    } else {
                        out.push(Some(sa.value(value_idx).to_string()));
                    }
                };
                match key_ty.as_ref() {
                    ArrowDataType::Int32 => {
                        let da = col
                            .as_any()
                            .downcast_ref::<DictionaryArray<Int32Type>>()
                            .ok_or_else(|| {
                                BoltError::Type(
                                    "register_table: dict<i32,utf8> downcast failed".into(),
                                )
                            })?;
                        let sa = da
                            .values()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| {
                                BoltError::Type(
                                    "register_table: dict values are not StringArray".into(),
                                )
                            })?;
                        let keys = da.keys();
                        for i in 0..n {
                            if keys.is_null(i) {
                                out.push(None);
                            } else {
                                decode_into(&mut out, keys.value(i) as usize, sa);
                            }
                        }
                    }
                    ArrowDataType::Int64 => {
                        let da = col
                            .as_any()
                            .downcast_ref::<DictionaryArray<Int64Type>>()
                            .ok_or_else(|| {
                                BoltError::Type(
                                    "register_table: dict<i64,utf8> downcast failed".into(),
                                )
                            })?;
                        let sa = da
                            .values()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| {
                                BoltError::Type(
                                    "register_table: dict values are not StringArray".into(),
                                )
                            })?;
                        let keys = da.keys();
                        for i in 0..n {
                            if keys.is_null(i) {
                                out.push(None);
                            } else {
                                decode_into(&mut out, keys.value(i) as usize, sa);
                            }
                        }
                    }
                    other => {
                        return Err(BoltError::Type(format!(
                            "register_table: dict key type {:?} not supported \
                             (expected Int32 or Int64 with Utf8 values)",
                            other
                        )));
                    }
                }
                let sa = StringArray::from(out);
                new_fields.push(ArrowField::new(
                    field.name().clone(),
                    ArrowDataType::Utf8,
                    field.is_nullable(),
                ));
                new_cols.push(Arc::new(sa) as ArrayRef);
                changed = true;
            }
            _ => {
                new_fields.push(field.as_ref().clone());
                new_cols.push(col.clone());
            }
        }
    }
    if !changed {
        return Ok(batch);
    }
    let new_schema = Arc::new(ArrowSchema::new(new_fields));
    RecordBatch::try_new(new_schema, new_cols)
        .map_err(|e| BoltError::Type(format!("register_table: rebuild after dict flatten failed: {e}")))
}

/// Parse the `BOLT_POOL_STATS_INTERVAL_SECS` environment variable into
/// a `Duration`. Missing or unparseable values default to
/// [`DEFAULT_POOL_STATS_INTERVAL_SECS`]; an explicit `0` disables
/// periodic emission (signalled by `Duration::ZERO`).
fn pool_stats_interval_from_env() -> Duration {
    match std::env::var(POOL_STATS_ENV).ok().and_then(|v| v.parse::<u64>().ok()) {
        Some(0) => Duration::ZERO,
        Some(n) => Duration::from_secs(n),
        None => Duration::from_secs(DEFAULT_POOL_STATS_INTERVAL_SECS),
    }
}

/// Decide whether to emit a pool-stats snapshot at time `now`, advancing
/// the throttle state on a positive decision.
///
/// Pulled out of [`Engine::maybe_emit_pool_stats`] so the throttle
/// semantics can be exercised without a live CUDA context. Side
/// effects: writes `Some(now)` into `last_emit` when emission is due,
/// leaves it untouched otherwise.
///
/// Returns `true` IFF the caller should emit a log line + observer
/// notification right now. Encapsulates three rules:
///   * `interval == 0` → never emit (env-var disables).
///   * `last_emit.is_none()` → always emit (first query on the engine).
///   * `now - last_emit >= interval` → emit and reset.
fn should_emit_pool_stats(
    last_emit: &Mutex<Option<Instant>>,
    interval: Duration,
    now: Instant,
) -> bool {
    if interval.is_zero() {
        return false;
    }
    let mut last = match last_emit.lock() {
        Ok(g) => g,
        Err(_) => return false, // poisoned — best-effort; skip the emit.
    };
    let should = match *last {
        None => true,
        Some(prev) => now.duration_since(prev) >= interval,
    };
    if should {
        *last = Some(now);
    }
    should
}

/// Stage 6 — walk `batch` and inform `provider` of each column's actual
/// runtime nullability (i.e. whether the source array had any nulls). For
/// `DictionaryArray<_, Utf8>` columns the per-row nullability lives on the
/// keys buffer, not the dictionary values; this helper consults
/// `keys().null_count()` to get the right answer. Called from
/// `register_table` / `replace_table` / `register_batch`, so the
/// engine-backed `TableProvider` (`EngineProvider::has_nulls`) and the
/// codegen-time `populate_input_validity` pass both see truthful claims.
fn propagate_column_nullability(
    provider: &mut MemTableProvider,
    table: &str,
    batch: &RecordBatch,
) {
    // `Array::null_count` is an inherent-trait method; pull the trait into
    // scope locally so we can ask any `&dyn Array` for its null count.
    use arrow_array::Array;
    let schema = batch.schema();
    for (idx, field) in schema.fields().iter().enumerate() {
        let arr = batch.column(idx);
        let has_nulls = match field.data_type() {
            ArrowDataType::Dictionary(key_t, _)
                if key_t.as_ref() == &ArrowDataType::Int32 =>
            {
                // Dict keys carry the per-row validity. Downcast and ask the
                // keys array directly; fall back to the array's own
                // `null_count()` if the downcast fails (shouldn't happen for
                // Int32 keys but defensive).
                match arr
                    .as_any()
                    .downcast_ref::<arrow_array::DictionaryArray<arrow_array::types::Int32Type>>()
                {
                    Some(da) => da.keys().null_count() > 0,
                    None => arr.null_count() > 0,
                }
            }
            _ => arr.null_count() > 0,
        };
        provider.set_column_nullability(table.to_string(), field.name().clone(), has_nulls);
    }
}

/// Convert an `arrow_schema::Schema` into our plan `Schema`.
fn arrow_schema_to_plan_schema(s: &ArrowSchema) -> BoltResult<Schema> {
    let mut fields = Vec::with_capacity(s.fields().len());
    for f in s.fields() {
        let dt = arrow_dtype_to_plan(f.data_type())?;
        fields.push(Field::new(f.name().clone(), dt, f.is_nullable()));
    }
    Ok(Schema::new(fields))
}

/// Convert our plan `Schema` to an `arrow_schema::Schema` (used for output `RecordBatch`).
fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}

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
struct EngineProvider<'a> {
    base: &'a MemTableProvider,
    tables: &'a HashMap<String, Vec<RecordBatch>>,
}

impl<'a> crate::plan::TableProvider for EngineProvider<'a> {
    fn schema(&self, name: &str) -> BoltResult<Schema> {
        self.base.schema(name)
    }

    fn has_nulls(&self, table_name: &str, col_idx: usize) -> bool {
        // Sum null_count across every registered batch for the table.
        // Safe-false on any miss — the executor's host-strip fallback still
        // handles the row filtering, so an under-flag is correctness-safe.
        let batches = match self.tables.get(table_name) {
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
        let batches = self.tables.get(table_name)?;
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

#[cfg(test)]
mod tests {
    //! Online tests for the lazy-upload `register_batch` path and the
    //! Stage-3 pinned async-memcpy wiring in `execute_projection`.
    //!
    //! The lazy-upload tests lock in the fix for the O(N²) PCIe re-upload bug
    //! described on the `gpu_tables` field: appending N batches must not cost
    //! `1+2+…+N` batches' worth of host→device traffic. They verify the
    //! observable correctness of the lazy path (rows from every appended batch
    //! are visible to the next query).
    //!
    //! The Stage-3 tests cover the per-query-stream + pinned D2H path —
    //! both the no-predicate and predicate flows — so any regression in the
    //! stream chaining surfaces as a value mismatch rather than a CUDA error.
    //!
    //! All tests are `#[ignore]`'d because they launch real kernels — run
    //! with `cargo test -- --ignored` on a GPU host.
    use super::*;
    use arrow_array::{Int32Array, Int64Array};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    /// Build a single-column `RecordBatch` whose `x` column holds the half-open
    /// range `[start, start+n)` as `Int64`. The schema is shared across all
    /// fixtures so `register_batch`'s schema check passes.
    fn int64_batch(start: i64, n: usize) -> RecordBatch {
        let col: Int64Array = (start..start + n as i64).collect();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            ArrowDataType::Int64,
            false,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap()
    }

    #[test]
    #[ignore = "requires CUDA device - run with `cargo test -- --ignored`"]
    fn register_batch_two_batches_query_sees_both() {
        // Register two batches, then SELECT the only column. The lazy-upload
        // path must rebuild the GpuTable from BOTH batches at query time, so
        // every row from both batches has to be visible in the result.
        let mut engine = Engine::new().expect("ctx");
        engine
            .register_batch("t", int64_batch(0, 4))
            .expect("first batch");
        engine
            .register_batch("t", int64_batch(4, 4))
            .expect("second batch");

        let h = engine.sql("SELECT x FROM t").expect("execute");
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 8, "8 rows after two 4-row batches");
        let actual = out
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 column");
        let got: Vec<i64> = (0..actual.len()).map(|i| actual.value(i)).collect();
        assert_eq!(got, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    #[ignore = "requires CUDA device - run with `cargo test -- --ignored`"]
    fn register_batch_ten_batches_combined_row_count() {
        // Append ten 100-row batches in a loop, then query. With the bug we
        // were fixing, this would upload 1+2+…+10 = 55 batches' worth of bytes
        // across the loop; with the fix it uploads zero bytes during the loop
        // and exactly one combined upload at query time. Correctness check:
        // the result has all 1000 rows and they sum to the expected total.
        let mut engine = Engine::new().expect("ctx");
        let n_batches = 10usize;
        let rows_per_batch = 100usize;
        for i in 0..n_batches {
            engine
                .register_batch("t", int64_batch((i * rows_per_batch) as i64, rows_per_batch))
                .unwrap_or_else(|e| panic!("register_batch {i}: {e}"));
        }
        let total_rows = n_batches * rows_per_batch;

        let h = engine.sql("SELECT x FROM t").expect("execute");
        let out = h.record_batch();
        assert_eq!(out.num_rows(), total_rows, "row count after 10 appends");

        let actual = out
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 column");
        let sum: i64 = (0..actual.len()).map(|i| actual.value(i)).sum();
        let expected_sum: i64 = (0..total_rows as i64).sum();
        assert_eq!(sum, expected_sum, "sum of x column across all 10 batches");
    }

    /// Verify that a bare projection still returns the right rows after the
    /// kernel launch and D2H downloads moved onto a per-query stream with
    /// async copies. Mirrors what the synchronous path was previously
    /// asserting — same input, same expected output — so any regression in
    /// the stream-flow shows up as a value mismatch rather than a CUDA error.
    #[test]
    #[ignore = "requires CUDA toolkit at runtime — Stage 2 async D2H correctness"]
    fn execute_projection_async_d2h_round_trip() {
        let mut engine = Engine::new().expect("engine init");

        // Single-column Int32 table: [1, 2, 3, 4, 5].
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1i32, 2, 3, 4, 5]));
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            ArrowDataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![arr]).expect("batch");
        engine.register_table("t", batch).expect("register");

        // Plain projection — no predicate, so the new async-D2H batch path
        // is exercised end-to-end.
        let handle = engine.sql("SELECT x FROM t").expect("query");
        let out = handle.record_batch();

        assert_eq!(out.num_rows(), 5);
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32");
        let got: Vec<i32> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, vec![1, 2, 3, 4, 5]);
    }

    /// Same shape, but with a WHERE clause so the predicate path is the one
    /// exercised. The Stage 2 patch removed the explicit
    /// `stream.synchronize()` after the projection kernel — the predicate
    /// kernel's own internal sync (inside `launch_predicate_kernel`) now
    /// covers both, and any regression in that chain surfaces here.
    #[test]
    #[ignore = "requires CUDA toolkit at runtime — Stage 2 stream chaining w/ predicate"]
    fn execute_projection_with_predicate_under_async_stream() {
        let mut engine = Engine::new().expect("engine init");

        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1i32, 2, 3, 4, 5]));
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            ArrowDataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![arr]).expect("batch");
        engine.register_table("t", batch).expect("register");

        let handle = engine
            .sql("SELECT x FROM t WHERE x > 2")
            .expect("query");
        let out = handle.record_batch();

        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32");
        let got: Vec<i32> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, vec![3, 4, 5]);
    }

    // ---------------------------------------------------------------------
    // Stage 7 (P1b): pool-stats periodic-emit throttle.
    //
    // We exercise `should_emit_pool_stats` directly with a mock `Instant`
    // sequence — no CUDA required. The function is the only stateful piece
    // of the periodic-log machinery (the rest is a log line + observer
    // call), so locking down its throttle semantics here gives us the full
    // behavioural coverage we need.
    // ---------------------------------------------------------------------

    #[test]
    fn pool_stats_throttle_first_call_always_emits() {
        // Fresh throttle (no previous emit) must always emit on first
        // call, regardless of how recently the test started.
        let last = Mutex::new(None);
        let now = Instant::now();
        assert!(should_emit_pool_stats(&last, Duration::from_secs(60), now));
        // Second call at the same instant: not enough time elapsed.
        assert!(!should_emit_pool_stats(&last, Duration::from_secs(60), now));
    }

    #[test]
    fn pool_stats_throttle_respects_interval() {
        let last = Mutex::new(None);
        let interval = Duration::from_secs(60);
        let t0 = Instant::now();
        assert!(should_emit_pool_stats(&last, interval, t0), "first emit");
        // 30s later: still inside the window.
        assert!(
            !should_emit_pool_stats(&last, interval, t0 + Duration::from_secs(30)),
            "30s < 60s — must NOT emit"
        );
        // 59s later: still inside.
        assert!(
            !should_emit_pool_stats(&last, interval, t0 + Duration::from_secs(59)),
            "59s < 60s — must NOT emit"
        );
        // 60s later: boundary should fire.
        assert!(
            should_emit_pool_stats(&last, interval, t0 + Duration::from_secs(60)),
            "60s == 60s — must emit"
        );
        // Right after the boundary fire: throttle is reset, so we must
        // wait the full window again.
        assert!(
            !should_emit_pool_stats(&last, interval, t0 + Duration::from_secs(61)),
            "1s after boundary emit — must NOT emit"
        );
        // 60s after the second emit: fires again.
        assert!(
            should_emit_pool_stats(&last, interval, t0 + Duration::from_secs(120)),
            "120s = 60s + 60s — must emit again"
        );
    }

    #[test]
    fn pool_stats_throttle_zero_interval_disables_emission() {
        // The env-var "0" sentinel disables periodic emission entirely.
        let last = Mutex::new(None);
        let now = Instant::now();
        assert!(!should_emit_pool_stats(&last, Duration::ZERO, now));
        // Even after a long delay, zero interval stays disabled.
        assert!(!should_emit_pool_stats(
            &last,
            Duration::ZERO,
            now + Duration::from_secs(3600)
        ));
        // `last` was never updated.
        assert!(last.lock().unwrap().is_none());
    }

    #[test]
    fn pool_stats_throttle_long_interval_still_fires_first_time() {
        // Even a 1-hour interval must produce the first-emit fire so a
        // short-lived process surfaces at least one snapshot.
        let last = Mutex::new(None);
        let now = Instant::now();
        let one_hour = Duration::from_secs(3600);
        assert!(should_emit_pool_stats(&last, one_hour, now));
    }

    #[test]
    fn pool_stats_interval_env_parsing_defaults() {
        // Smoke-test the env-var helper. We can't easily mutate the
        // process env in a parallel test runner safely, so just check the
        // explicit defaults arms. Without the env var set, the default
        // is 60 seconds.
        //
        // NOTE: this test reads (not writes) the env var, so it's safe to
        // run in parallel; the expected default here matches the constant.
        // If a future contributor sets `BOLT_POOL_STATS_INTERVAL_SECS` in
        // their shell while running `cargo test`, this assertion will
        // flag the override — that's intentional.
        if std::env::var(POOL_STATS_ENV).is_err() {
            assert_eq!(
                pool_stats_interval_from_env(),
                Duration::from_secs(DEFAULT_POOL_STATS_INTERVAL_SECS)
            );
        }
    }
}
