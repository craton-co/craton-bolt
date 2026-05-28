// SPDX-License-Identifier: Apache-2.0

//! SQL frontend: parses a SQL string into a `LogicalPlan` against a `TableProvider`.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use sqlparser::ast::{
    BinaryOperator, Distinct, Expr as SqlExpr, FunctionArg, FunctionArgExpr, FunctionArguments,
    GroupByExpr, Ident, JoinConstraint, JoinOperator, ObjectName, Offset, OrderByExpr, Query,
    Select, SelectItem, SetExpr, SetOperator, SetQuantifier, Statement, TableFactor,
    UnaryOperator, Value,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{
    AggregateExpr, BinaryOp, Expr, JoinType, Literal, LogicalPlan, Schema, SortExpr,
};

/// Resolves table names to their schemas; the SQL frontend cannot know table shapes otherwise.
///
/// # PV-stage-d: per-column null-bearing signal
///
/// The two extension hooks ([`has_nulls`](TableProvider::has_nulls) and
/// [`null_count`](TableProvider::null_count)) let the planner emit
/// validity-aware kernels for columns the provider knows carry a NULL
/// bitmap, and the simpler null-free path for everything else. Both
/// methods default to "safe-false" / `None`, so providers that haven't
/// been updated continue to work — the executor's run-time host-strip
/// fallback (see [`crate::exec::groupby_with_pre`],
/// [`crate::exec::groupby_valid`]) still handles row filtering for
/// columns that turn out to carry nulls at execution time.
///
/// # Run-time fallback (executor-time validity)
///
/// Even when a provider does not override `has_nulls`, the per-stage
/// upload helpers ([`crate::exec::agg_with_pre`],
/// [`crate::exec::groupby_with_pre`]) inspect `arr.null_count()` on each
/// input column at upload time and set
/// [`crate::plan::physical_plan::KernelSpec::input_has_validity`] from
/// `PreCol::has_validity()` before invoking
/// [`crate::jit::ptx_gen::compile`]. GROUP BY aggregate kernels likewise
/// dispatch to
/// [`crate::jit::hash_kernels::compile_groupby_agg_kernel_with_validity`]
/// when the registered Arrow batch carries nulls. The plan-time signal is
/// preferred when available because it lets the JIT cache key reflect
/// per-column nullability — but correctness does not depend on it.
pub trait TableProvider {
    /// Return the schema for `name`, or a `Plan` error if the table is unknown.
    fn schema(&self, name: &str) -> BoltResult<Schema>;

    /// Plan-time signal: does column `col_idx` of `table_name` carry a
    /// NULL bitmap that downstream kernels must consume?
    ///
    /// The default returns `false` for every column (safe — the executor
    /// still inspects `RecordBatch::null_count()` at run time and falls
    /// back to host-side row stripping if a null is found). Providers
    /// that already know their physical layout (e.g. backed by Arrow
    /// arrays whose `null_count()` is cheap to read) should override
    /// this so the planner can pick the native-validity kernel path.
    ///
    /// `col_idx` is the column ordinal in the schema returned by
    /// [`Self::schema`]. Out-of-range indices return `false`.
    fn has_nulls(&self, table_name: &str, col_idx: usize) -> bool {
        let _ = (table_name, col_idx);
        false
    }

    /// Optional richer signal: exact null count of column `col_idx`, or
    /// `None` if the provider can't (or won't) compute it. Defaults to
    /// `None`. Implementors that return `Some(_)` should keep
    /// [`Self::has_nulls`] consistent — i.e. `has_nulls(_, _)` should
    /// return `null_count(_, _).map_or(false, |n| n > 0)`. The split
    /// exists so cheap "is it dense?" checks don't pay for a full
    /// `null_count` materialisation.
    fn null_count(&self, table_name: &str, col_idx: usize) -> Option<usize> {
        let _ = (table_name, col_idx);
        None
    }

    /// Monotonically-increasing version token bumped whenever the provider's
    /// view of the schema universe changes (a table is registered, replaced,
    /// or dropped). The [`parse`] plan cache mixes this into its key so a
    /// schema change invalidates previously-cached plans automatically.
    ///
    /// **Default behaviour: opt-out of caching.** The default returns a
    /// fresh, never-recurring token on every call (pulled from a
    /// process-wide counter), so providers that haven't been audited for
    /// versioning correctness produce a cache miss every time. This is the
    /// safe choice: a stale-plan bug from a mis-implemented
    /// `schema_version` is harder to diagnose than the missing speed-up
    /// from a cache miss. Providers that intend to participate in the
    /// cache (like [`MemTableProvider`]) override this to return a token
    /// that is *stable* across calls *until* the schema state changes.
    fn schema_version(&self) -> u64 {
        // A fresh token on every call → no two `(sql, version)` keys ever
        // collide, → the cache stores entries that are never re-hit, → at
        // most one entry per (provider, parse call) ever accumulates and
        // FIFO eviction reclaims it long before it matters. Net effect:
        // the cache is functionally disabled for default-impl providers.
        next_provider_version()
    }
}

/// In-FROM-scope name resolver: maps `table.col` (and bare `col`) references in
/// the SELECT/WHERE clauses to the output column names produced by the
/// FROM-tree (a base Scan, possibly extended by one or more INNER JOINs).
///
/// Built incrementally as the planner walks the FROM tree so its rename
/// convention stays in lockstep with [`join_combined_schema`](crate::plan::logical_plan::join_combined_schema):
/// the leftmost table whose column name appears wins the bare name; every
/// later collision is renamed to `right.{col}` (with `__N` suffixes if even
/// that collides).
///
/// The resolver intentionally borrows nothing from the FROM-tree plan — it
/// owns its own (table_name, output_col) mapping so it remains valid after
/// the planner moves the plan into `LogicalPlan::Join` boxes.
#[derive(Debug, Default)]
struct NameResolver {
    /// One scope per table in FROM order (base first, then joined tables).
    tables: Vec<TableScope>,
}

/// One table's contribution to a [`NameResolver`].
#[derive(Debug)]
struct TableScope {
    /// Table name as it appears in FROM (no aliases — we don't support those yet).
    name: String,
    /// For each column of the table's original schema, the *output* column name
    /// it produces in the FROM-tree's combined schema. Indices align with the
    /// original [`Schema::fields`] order.
    cols: Vec<TableCol>,
}

/// One column in a [`TableScope`]: the user-typeable name (as it appeared in
/// the table's schema) plus the name that name maps to after JOIN renaming.
#[derive(Debug)]
struct TableCol {
    /// The original (qualifier-local) column name — what `table.col` matches.
    original: String,
    /// The output column name in the FROM-tree's combined schema.
    output: String,
}

impl NameResolver {
    /// Empty resolver (no tables in scope). Used by `lower_order_by`, where
    /// expressions run *after* projection so the FROM-tree's table qualifiers
    /// are no longer meaningful. With no tables, `Identifier` still lowers to
    /// a column ref (downstream type-checking validates the name), but
    /// `CompoundIdentifier` is rejected because no qualifier can match.
    fn empty() -> Self {
        Self::default()
    }

    /// Push the base table scope. Each column maps to its own original name —
    /// the base table is always the leftmost contributor, so nothing is
    /// renamed yet.
    fn push_base(&mut self, name: String, schema: &Schema) {
        let cols = schema
            .fields
            .iter()
            .map(|f| TableCol {
                original: f.name.clone(),
                output: f.name.clone(),
            })
            .collect();
        self.tables.push(TableScope { name, cols });
    }

    /// Push a joined table scope. Applies the same rename rule as
    /// [`join_combined_schema`]: a right-side column whose name already
    /// appears in the accumulated taken-set is renamed to `right.{col}`,
    /// with `__2`, `__3`, … suffixes appended as a last resort if even the
    /// qualified form clashes. Keeping this rule in lockstep with that
    /// function is the whole point of routing through both call sites.
    fn push_join(&mut self, name: String, schema: &Schema) {
        // Build the snapshot of names already taken across all previous
        // scopes' *output* names. This mirrors `join_combined_schema`'s
        // pass-by-pass accumulation: each new right side sees everything
        // produced so far on its left, not just the immediately preceding
        // table.
        let mut taken: std::collections::HashSet<String> = self
            .tables
            .iter()
            .flat_map(|t| t.cols.iter().map(|c| c.output.clone()))
            .collect();
        let mut cols = Vec::with_capacity(schema.fields.len());
        for f in &schema.fields {
            let mut out_name = if taken.contains(&f.name) {
                format!("right.{}", f.name)
            } else {
                f.name.clone()
            };
            // Final-resort uniqueness suffix; only triggers if even the
            // qualified form collides (e.g. an actual `right.x` column on
            // the left side).
            if taken.contains(&out_name) {
                let base = out_name.clone();
                let mut i = 2usize;
                loop {
                    let candidate = format!("{base}__{i}");
                    if !taken.contains(&candidate) {
                        out_name = candidate;
                        break;
                    }
                    i += 1;
                }
            }
            taken.insert(out_name.clone());
            cols.push(TableCol {
                original: f.name.clone(),
                output: out_name,
            });
        }
        self.tables.push(TableScope { name, cols });
    }

    /// Resolve `qualifier.col` to its output column name in the FROM-tree's
    /// combined schema.
    ///
    /// Errors with a clear message if the qualifier matches no in-scope table
    /// or the column doesn't exist in the qualified table's schema.
    fn resolve_compound(&self, qualifier: &str, col: &str) -> BoltResult<String> {
        let scope = self
            .tables
            .iter()
            .find(|t| t.name == qualifier)
            .ok_or_else(|| {
                BoltError::Sql(format!(
                    "unknown table qualifier '{qualifier}' in column reference '{qualifier}.{col}'"
                ))
            })?;
        let resolved = scope
            .cols
            .iter()
            .find(|c| c.original == col)
            .ok_or_else(|| {
                BoltError::Sql(format!(
                    "unknown column '{col}' in table '{qualifier}'"
                ))
            })?;
        Ok(resolved.output.clone())
    }
}

/// In-memory `name → Schema` provider; useful in tests and as a default.
///
/// Stage 6 addition: the provider also accepts a `nulls_by_column` side-table
/// declaring which `(table, column)` pairs are known to admit nulls. The
/// frontend doesn't otherwise care — nulls are a runtime concern — but the
/// engine's null-aware sort / aggregation paths consult [`Self::has_nulls`]
/// to skip the validity-bitmap upload when a column is provably null-free.
///
/// `Schema` already records nullability per [`Field`], so `has_nulls` first
/// consults the field-level flag. The `nulls_by_column` override exists for
/// cases where the schema is supplied externally with a pessimistic
/// `nullable: true` but the ingest path (e.g. a `DictionaryArray` whose key
/// buffer has `null_count() == 0`) has confirmed the column is null-free at
/// runtime.
#[derive(Debug)]
pub struct MemTableProvider {
    /// Registered tables, keyed by name.
    tables: HashMap<String, Schema>,
    /// Per-`(table, column)` runtime nullability override.
    /// `Some(true)`  → column is known to contain at least one NULL,
    /// `Some(false)` → column is known to be free of NULLs,
    /// `None` (absent) → fall back to the field's `nullable` flag.
    nulls_by_column: HashMap<(String, String), bool>,
    /// Process-wide-unique schema version. Bumped on every mutation
    /// ([`Self::register`], [`Self::unregister_table`],
    /// [`Self::set_column_nullability`]) by fetch-adding a single global
    /// counter ([`NEXT_PROVIDER_VERSION`]). Because every bump pulls from
    /// the global counter, no two `(provider, mutation_count)` states ever
    /// share a `version` value across the process — so the plan cache's
    /// `(sql, version)` key is unambiguous even when several
    /// `MemTableProvider` instances coexist (e.g. in parallel test
    /// threads sharing the process-wide `PLAN_CACHE`).
    version: AtomicU64,
}

/// Global monotonic counter feeding every [`MemTableProvider`] version bump.
/// `fetch_add(1, Relaxed)` is enough — we only need uniqueness, not any
/// happens-before ordering against other state.
static NEXT_PROVIDER_VERSION: AtomicU64 = AtomicU64::new(1);

/// Pull the next unique version token. Centralised so the constructor and
/// every mutation route through the same source of truth.
fn next_provider_version() -> u64 {
    NEXT_PROVIDER_VERSION.fetch_add(1, Ordering::Relaxed)
}

impl Default for MemTableProvider {
    fn default() -> Self {
        Self {
            tables: HashMap::new(),
            nulls_by_column: HashMap::new(),
            version: AtomicU64::new(next_provider_version()),
        }
    }
}

impl Clone for MemTableProvider {
    /// Cloning takes a fresh version token so the clone is treated as a
    /// distinct provider instance by the plan cache. Otherwise two clones
    /// that diverged via different `register_table` calls could collide on
    /// the same `(sql, version)` cache key for SQL referencing a table
    /// only present in one of them.
    fn clone(&self) -> Self {
        Self {
            tables: self.tables.clone(),
            nulls_by_column: self.nulls_by_column.clone(),
            version: AtomicU64::new(next_provider_version()),
        }
    }
}

impl MemTableProvider {
    /// Empty provider.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style: register a table and return `self`.
    pub fn with_table(mut self, name: impl Into<String>, schema: Schema) -> Self {
        self.register(name, schema);
        self
    }

    /// Mutating register; overwrites any existing entry with the same name.
    /// Bumps [`Self::schema_version`] so the parse cache invalidates plans
    /// that were lowered against the prior provider state.
    pub fn register(&mut self, name: impl Into<String>, schema: Schema) {
        let name = name.into();
        // Replace-not-merge: drop stale overrides for the prior schema of
        // this table so a re-registered table doesn't inherit a stale
        // nullability claim.
        self.nulls_by_column.retain(|(t, _), _| t != &name);
        self.tables.insert(name, schema);
        self.bump_version();
    }

    /// Remove a registered table (no-op if absent). Bumps the schema
    /// version regardless of whether the table existed, so callers don't
    /// have to second-guess whether a particular drop invalidated cached
    /// plans (this errs on the side of *more* invalidation, which is
    /// correctness-safe). Returns `true` iff the table was present.
    pub fn unregister_table(&mut self, name: &str) -> bool {
        let removed = self.tables.remove(name).is_some();
        if removed {
            self.nulls_by_column.retain(|(t, _), _| t != name);
        }
        self.bump_version();
        removed
    }

    /// Record runtime nullability for `(table, column)`. Used by the engine
    /// after it ingests a batch and learns the actual `null_count()` of
    /// each column — including `DictionaryArray` keys, where the
    /// nullability question is answered by the keys' null buffer, not the
    /// dictionary values. Bumps the schema version: the planner's
    /// `KernelSpec::input_has_validity` derivation reads this flag, so a
    /// change here must invalidate cached plans.
    pub fn set_column_nullability(
        &mut self,
        table: impl Into<String>,
        column: impl Into<String>,
        has_nulls: bool,
    ) {
        self.nulls_by_column
            .insert((table.into(), column.into()), has_nulls);
        self.bump_version();
    }

    /// Internal: assign a fresh process-wide-unique version token to this
    /// provider. Called from every mutating entry point.
    fn bump_version(&self) {
        // `Relaxed` is enough: we are the sole writer (mutating methods
        // take `&mut self`) and the only reader that cares about ordering
        // is the plan-cache lookup, which races against itself rather
        // than against any other piece of state.
        self.version
            .store(next_provider_version(), Ordering::Relaxed);
    }

    /// True if `(table, column)` is known to admit nulls.
    ///
    /// Resolution order:
    ///   1. Runtime override from [`Self::set_column_nullability`] (most precise).
    ///   2. Field-level `nullable` flag on the registered `Schema`.
    ///   3. `false` if the table or column is unknown.
    ///
    /// This is the entry point the engine's sort / aggregate paths consult
    /// when deciding whether to ship a validity bitmap to the device. For
    /// a `DictUtf8` column the answer comes from the keys array's
    /// `null_count()`, surfaced here via `set_column_nullability` at
    /// register-time — *not* from the dictionary values.
    pub fn has_nulls(&self, table: &str, column: &str) -> bool {
        if let Some(&v) = self
            .nulls_by_column
            .get(&(table.to_string(), column.to_string()))
        {
            return v;
        }
        match self.tables.get(table) {
            Some(s) => s
                .fields
                .iter()
                .find(|f| f.name == column)
                .map(|f| f.nullable)
                .unwrap_or(false),
            None => false,
        }
    }
}

impl TableProvider for MemTableProvider {
    fn schema(&self, name: &str) -> BoltResult<Schema> {
        self.tables
            .get(name)
            .cloned()
            .ok_or_else(|| BoltError::Plan(format!("unknown table '{name}'")))
    }

    fn schema_version(&self) -> u64 {
        // `Relaxed` matches the writer (`bump_version` uses `Relaxed`); we
        // do not need a happens-before edge against any other state.
        self.version.load(Ordering::Relaxed)
    }
}

/// Parse a SQL string into a single `LogicalPlan` using the given provider.
///
/// Repeated calls with identical `sql` against a provider whose
/// [`TableProvider::schema_version`] is unchanged are served from a small
/// process-wide LRU-ish (FIFO) plan cache (see [`PLAN_CACHE`]). Cache misses
/// fall through to the full tokenise + parse + lower pipeline. Failures
/// (`Err`) are *not* cached — a re-parse of bad SQL still pays the parser
/// cost, but that's a one-time hit at development time, not the dashboard
/// hot path the cache is here to accelerate.
pub fn parse(sql: &str, provider: &dyn TableProvider) -> BoltResult<LogicalPlan> {
    let version = provider.schema_version();
    if let Some(plan) = plan_cache_lookup(sql, version) {
        // Cheap deep-copy: every other lowered branch already pays a clone
        // somewhere downstream (e.g. through `lower(&logical)` borrowing).
        // `LogicalPlan` is `Clone` (verified in `logical_plan.rs`); the
        // cache stores the canonical `Arc<LogicalPlan>` so successive hits
        // pay only the clone of nested Strings / Boxes.
        return Ok((*plan).clone());
    }

    let plan = parse_uncached(sql, provider)?;
    plan_cache_insert(sql.to_string(), version, Arc::new(plan.clone()));
    Ok(plan)
}

/// Internal: do the real parse + lower, with no cache interaction at all.
/// Separated out so the cache layer in [`parse`] is a thin shell that's easy
/// to read and so tests / benches that want to bypass the cache can.
fn parse_uncached(sql: &str, provider: &dyn TableProvider) -> BoltResult<LogicalPlan> {
    let dialect = GenericDialect {};
    let mut stmts = Parser::parse_sql(&dialect, sql).map_err(|e| BoltError::Sql(e.to_string()))?;

    if stmts.len() != 1 {
        return Err(BoltError::Sql(format!(
            "expected exactly one statement, got {}",
            stmts.len()
        )));
    }
    let stmt = stmts.remove(0);
    let query = match stmt {
        Statement::Query(q) => q,
        other => {
            return Err(BoltError::Sql(format!(
                "only SELECT queries are supported, got: {other}"
            )));
        }
    };
    plan_query(&query, provider)
}

// ---------------------------------------------------------------------------
// Parse + plan cache
// ---------------------------------------------------------------------------
//
// Dashboard-style workloads send the same handful of SQL strings on every
// poll. Tokenising, parsing, and lowering each one from scratch is pure
// waste — the work is deterministic in `(sql, schema_universe)`, so memoise
// it. We follow the same pattern as `crate::jit::jit_compiler::PtxCache`:
//   * A `HashMap` keyed on the (sql, version) pair for O(1) lookup,
//     paired with a `VecDeque` of keys in insertion order for FIFO eviction
//     at the configured cap. The deque is the canonical eviction order —
//     `HashMap`'s iteration order would not survive rehashes.
//   * A `Lazy<Mutex<PlanCache>>` global, initialised on first `parse` call.
//   * Hit / miss / evict counters surfaced via [`plan_cache_stats`] for
//     orchestrator-side observability (no `tracing` dep yet).
//
// We store `Arc<LogicalPlan>` so the cache hit path's clone cost is bounded
// by the nested heap allocations inside the plan tree (a handful of String
// / Box::clone calls) and does NOT scan the lowered tree twice.

/// Environment variable that overrides the default cache capacity. Read
/// once on first access via [`plan_cache_cap`] and frozen for the rest of
/// the process lifetime; testing the eviction policy with a different cap
/// uses `PlanCache::with_capacity` directly instead of mutating env state.
const PLAN_CACHE_SIZE_ENV: &str = "CRATON_PLAN_CACHE_SIZE";

/// Default cache capacity if `CRATON_PLAN_CACHE_SIZE` is unset / unparsable.
/// Sized for "tens of dashboard tiles" — large enough to absorb the typical
/// repeated-query workload, small enough that the cache itself is a rounding
/// error against the lowered plan footprint.
const PLAN_CACHE_CAP_DEFAULT: usize = 64;

/// Parse a candidate capacity string (typically from `std::env::var`).
/// `None`, empty strings, zero, and unparseable values all map to
/// `default`. Factored out so the policy is unit-testable without touching
/// the process environment.
fn parse_plan_cache_cap(raw: Option<&str>, default: usize) -> usize {
    raw.and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// Resolve the effective cache cap. Memoised via an inner `OnceLock` so
/// the env var is only consulted once — a long-running process can't have
/// its plan cache resize partway through.
fn plan_cache_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let raw = std::env::var(PLAN_CACHE_SIZE_ENV).ok();
        parse_plan_cache_cap(raw.as_deref(), PLAN_CACHE_CAP_DEFAULT)
    })
}

/// Composite cache key: SQL text plus the provider's `schema_version`. A
/// schema-universe change bumps `version`, so previously-cached plans for
/// the same SQL become unreachable (and eventually FIFO-evicted) the next
/// time the same SQL is parsed against the new state.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct PlanCacheKey {
    sql: String,
    version: u64,
}

/// Cache state. Mirrors [`crate::jit::jit_compiler::PtxCache`]'s layout: a
/// hash map for lookup, a deque for FIFO eviction order, plus three
/// counters for `(hits, misses, evictions)`.
///
/// All mutation happens under a [`Mutex`]; the hot path is short (hash +
/// probe + Arc clone), so there is no concurrency hazard worth a more
/// elaborate scheme.
struct PlanCache {
    /// Maximum number of entries. Fixed at construction.
    capacity: usize,
    /// Cached plans keyed by `(sql, version)`. Values are `Arc<LogicalPlan>`
    /// so hits clone the cheap `Arc`, not the lowered tree.
    map: HashMap<PlanCacheKey, Arc<LogicalPlan>>,
    /// Keys in insertion order. Front is oldest; eviction pops the front.
    /// We do NOT touch this on hits — FIFO, not true LRU — which keeps the
    /// hot path lock-free of any deque manipulation. The terminology in
    /// the task is "LRU-ish": good enough for dashboard reuse patterns
    /// where the working set fits comfortably under `capacity`.
    order: VecDeque<PlanCacheKey>,
    /// Cumulative hits since process start.
    hits: usize,
    /// Cumulative misses since process start.
    misses: usize,
    /// Cumulative evictions since process start.
    evictions: usize,
}

impl PlanCache {
    /// Empty cache with the given (positive) capacity.
    fn with_capacity(capacity: usize) -> Self {
        // `capacity == 0` would mean "evict on every insert"; we treat it
        // as the default to keep callers (env-var-misconfigured users)
        // from accidentally disabling the cache entirely.
        let capacity = capacity.max(1);
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Look up a cached plan. On hit, returns the cached `Arc<LogicalPlan>`
    /// and bumps the hit counter; on miss, bumps the miss counter and
    /// returns `None`. We do NOT touch the eviction order on hits — see
    /// the docstring on `order`.
    fn lookup(&mut self, key: &PlanCacheKey) -> Option<Arc<LogicalPlan>> {
        match self.map.get(key) {
            Some(plan) => {
                self.hits += 1;
                Some(Arc::clone(plan))
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Insert a freshly-lowered plan. If a matching key is already present
    /// (a race between two misses that both did the parse work
    /// concurrently) the *existing* value is left in place and the new one
    /// is dropped — both threads return identical-looking plans, and we
    /// keep the insertion-order deque clean.
    ///
    /// Otherwise, if we are at capacity, FIFO-evict the oldest entry
    /// before inserting.
    fn insert(&mut self, key: PlanCacheKey, plan: Arc<LogicalPlan>) {
        if self.map.contains_key(&key) {
            return;
        }
        while self.map.len() >= self.capacity {
            match self.order.pop_front() {
                Some(old) => {
                    if self.map.remove(&old).is_some() {
                        self.evictions += 1;
                    }
                }
                None => break,
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, plan);
    }

    /// `(hits, misses, evictions)` snapshot. Cheap copy from the counters.
    fn stats(&self) -> (usize, usize, usize) {
        (self.hits, self.misses, self.evictions)
    }
}

/// Process-wide singleton. Initialised lazily on first `parse` call so
/// startup cost is `0` for binaries that never touch SQL.
static PLAN_CACHE: Lazy<Mutex<PlanCache>> =
    Lazy::new(|| Mutex::new(PlanCache::with_capacity(plan_cache_cap())));

/// Shared `AtomicUsize` snapshot of the most-recent `stats()` triple. Lets
/// observers read counters without taking the cache lock at all, which is
/// useful for periodic health-check probes. The triple is updated on every
/// `parse` call right before returning.
static PLAN_CACHE_HITS: AtomicUsize = AtomicUsize::new(0);
static PLAN_CACHE_MISSES: AtomicUsize = AtomicUsize::new(0);
static PLAN_CACHE_EVICTIONS: AtomicUsize = AtomicUsize::new(0);

/// Look up `(sql, version)` in the global plan cache, bumping the shared
/// `hits` / `misses` snapshot. Returns the cached `Arc<LogicalPlan>` on
/// hit, `None` on miss.
fn plan_cache_lookup(sql: &str, version: u64) -> Option<Arc<LogicalPlan>> {
    let key = PlanCacheKey {
        sql: sql.to_string(),
        version,
    };
    let mut cache = PLAN_CACHE.lock();
    let result = cache.lookup(&key);
    let (h, m, e) = cache.stats();
    drop(cache);
    PLAN_CACHE_HITS.store(h, Ordering::Relaxed);
    PLAN_CACHE_MISSES.store(m, Ordering::Relaxed);
    PLAN_CACHE_EVICTIONS.store(e, Ordering::Relaxed);
    result
}

/// Insert `plan` for `(sql, version)` in the global plan cache, then
/// refresh the shared `(hits, misses, evictions)` snapshot.
fn plan_cache_insert(sql: String, version: u64, plan: Arc<LogicalPlan>) {
    let key = PlanCacheKey { sql, version };
    let mut cache = PLAN_CACHE.lock();
    cache.insert(key, plan);
    let (h, m, e) = cache.stats();
    drop(cache);
    PLAN_CACHE_HITS.store(h, Ordering::Relaxed);
    PLAN_CACHE_MISSES.store(m, Ordering::Relaxed);
    PLAN_CACHE_EVICTIONS.store(e, Ordering::Relaxed);
}

/// Public observability hook: `(hits, misses, evictions)` since process
/// start. Read from atomic snapshots that are refreshed on every `parse`
/// call, so this never blocks behind the cache lock.
///
/// The returned triple is a point-in-time read; reading the three counters
/// is *not* atomic as a whole (each load is `Relaxed`), which is fine for
/// the intended dashboard-style sampling use. If you need a perfectly
/// consistent triple, take it inside a single `PLAN_CACHE.lock()` —
/// nothing else in the public API exposes the lock so that path is
/// internal only.
pub fn plan_cache_stats() -> (usize, usize, usize) {
    (
        PLAN_CACHE_HITS.load(Ordering::Relaxed),
        PLAN_CACHE_MISSES.load(Ordering::Relaxed),
        PLAN_CACHE_EVICTIONS.load(Ordering::Relaxed),
    )
}

/// Lower a top-level `Query`. Supports SELECT, UNION [ALL], ORDER BY, LIMIT,
/// and OFFSET. Rejects CTEs, FETCH, locks, EXCEPT/INTERSECT, and dialect
/// extensions outside our subset.
fn plan_query(query: &Query, provider: &dyn TableProvider) -> BoltResult<LogicalPlan> {
    if query.with.is_some() {
        return Err(BoltError::Sql("unsupported: WITH / CTEs".into()));
    }
    if !query.limit_by.is_empty() {
        return Err(BoltError::Sql("unsupported: LIMIT BY".into()));
    }
    if query.fetch.is_some() {
        return Err(BoltError::Sql("unsupported: FETCH".into()));
    }
    if !query.locks.is_empty() {
        return Err(BoltError::Sql("unsupported: FOR UPDATE/SHARE".into()));
    }
    if query.for_clause.is_some() {
        return Err(BoltError::Sql("unsupported: FOR clause".into()));
    }
    if query.settings.is_some() {
        return Err(BoltError::Sql("unsupported: SETTINGS clause".into()));
    }
    if query.format_clause.is_some() {
        return Err(BoltError::Sql("unsupported: FORMAT clause".into()));
    }

    // Lower the body into a base plan; UNION/UNION ALL builds a `Union` (and
    // optionally a `Distinct` wrapper) here, so the ORDER BY / LIMIT layers
    // below apply to the *combined* result, matching SQL semantics.
    let mut plan = lower_set_expr(query.body.as_ref(), provider)?;

    // ORDER BY: appended *outside* the body so it sees the final schema.
    if let Some(order_by) = &query.order_by {
        let sort_exprs = lower_order_by(&order_by.exprs)?;
        if !sort_exprs.is_empty() {
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                sort_exprs,
            };
        }
    }

    // LIMIT [OFFSET]: fold both into a single `Limit` node so a downstream
    // executor can implement the offset as a skip without needing a separate
    // operator. Either clause alone is legal; OFFSET without LIMIT is
    // represented as `Limit { limit: usize::MAX, offset }`.
    let limit_value = match &query.limit {
        Some(e) => Some(usize_from_literal(e, "LIMIT")?),
        None => None,
    };
    let offset_value = match &query.offset {
        Some(Offset { value, .. }) => Some(usize_from_literal(value, "OFFSET")?),
        None => None,
    };
    if limit_value.is_some() || offset_value.is_some() {
        plan = LogicalPlan::Limit {
            input: Box::new(plan),
            limit: limit_value.unwrap_or(usize::MAX),
            offset: offset_value.unwrap_or(0),
        };
    }

    Ok(plan)
}

/// Lower a `SetExpr` (SELECT body or UNION/EXCEPT/INTERSECT node) into a
/// `LogicalPlan`. UNION ALL becomes `Union { inputs }`; plain UNION becomes
/// `Distinct(Union { inputs })`. EXCEPT/INTERSECT are rejected.
fn lower_set_expr(expr: &SetExpr, provider: &dyn TableProvider) -> BoltResult<LogicalPlan> {
    match expr {
        SetExpr::Select(s) => plan_select(s.as_ref(), provider),
        SetExpr::Query(q) => plan_query(q.as_ref(), provider),
        SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } => {
            if *op != SetOperator::Union {
                return Err(BoltError::Sql(format!(
                    "unsupported set operator: {op}; only UNION / UNION ALL"
                )));
            }
            // Reject the BY NAME variants (non-standard, schema-rewriting).
            let dedup = match set_quantifier {
                SetQuantifier::All => false,
                SetQuantifier::Distinct | SetQuantifier::None => true,
                SetQuantifier::ByName
                | SetQuantifier::AllByName
                | SetQuantifier::DistinctByName => {
                    return Err(BoltError::Sql(
                        "unsupported: UNION BY NAME".into(),
                    ));
                }
            };
            // Flatten left-recursive UNION chains into a single Union node so
            // `q1 UNION ALL q2 UNION ALL q3` becomes one 3-input Union rather
            // than a nested binary tree. UNION (dedup) does NOT flatten across
            // UNION ALL boundaries: their semantics differ.
            let mut inputs: Vec<LogicalPlan> = Vec::new();
            collect_union_branches(left, provider, dedup, &mut inputs)?;
            collect_union_branches(right, provider, dedup, &mut inputs)?;
            let union = LogicalPlan::Union { inputs };
            Ok(if dedup {
                LogicalPlan::Distinct {
                    input: Box::new(union),
                }
            } else {
                union
            })
        }
        SetExpr::Values(_) => Err(BoltError::Sql("unsupported: VALUES".into())),
        SetExpr::Insert(_) | SetExpr::Update(_) => Err(BoltError::Sql(
            "unsupported: write statement in query body".into(),
        )),
        SetExpr::Table(_) => Err(BoltError::Sql("unsupported: TABLE statement".into())),
    }
}

/// Helper for `lower_set_expr`: if `expr` is itself a same-quantifier UNION,
/// recurse to collect its operands directly into `out`; otherwise lower it
/// as a single branch. `parent_dedup` indicates whether the enclosing UNION
/// is a dedup variant (so we only flatten matching-quantifier children).
fn collect_union_branches(
    expr: &SetExpr,
    provider: &dyn TableProvider,
    parent_dedup: bool,
    out: &mut Vec<LogicalPlan>,
) -> BoltResult<()> {
    if let SetExpr::SetOperation {
        op: SetOperator::Union,
        set_quantifier,
        left,
        right,
    } = expr
    {
        let child_dedup = match set_quantifier {
            SetQuantifier::All => false,
            SetQuantifier::Distinct | SetQuantifier::None => true,
            // Non-flattening cases — fall through to a non-flat lower.
            SetQuantifier::ByName
            | SetQuantifier::AllByName
            | SetQuantifier::DistinctByName => {
                out.push(lower_set_expr(expr, provider)?);
                return Ok(());
            }
        };
        if child_dedup == parent_dedup {
            collect_union_branches(left, provider, parent_dedup, out)?;
            collect_union_branches(right, provider, parent_dedup, out)?;
            return Ok(());
        }
    }
    out.push(lower_set_expr(expr, provider)?);
    Ok(())
}

/// Lower a list of `OrderByExpr` into our `SortExpr`s. The default sort
/// direction is ASC; the default NULL placement follows SQL convention
/// (NULLS FIRST for ASC, NULLS LAST for DESC) when the user omits it.
fn lower_order_by(exprs: &[OrderByExpr]) -> BoltResult<Vec<SortExpr>> {
    // ORDER BY runs outside the FROM-tree (after projection), so no table
    // qualifiers are in scope. We pass an empty resolver; bare identifiers
    // still lower as column refs against the post-projection schema, and
    // any stray `table.col` ref will fall through to a clean "unknown
    // table qualifier" error.
    let resolver = NameResolver::empty();
    let mut out = Vec::with_capacity(exprs.len());
    for OrderByExpr {
        expr,
        asc,
        nulls_first,
        with_fill,
    } in exprs
    {
        if with_fill.is_some() {
            return Err(BoltError::Sql(
                "unsupported: ORDER BY ... WITH FILL".into(),
            ));
        }
        let descending = match asc {
            Some(true) | None => false,
            Some(false) => true,
        };
        // Default NULL placement: NULLS FIRST for ASC, NULLS LAST for DESC.
        let nulls_first = match nulls_first {
            Some(b) => *b,
            None => !descending,
        };
        out.push(SortExpr {
            expr: lower_expr(expr, &resolver)?,
            descending,
            nulls_first,
        });
    }
    Ok(out)
}

/// Parse a SQL `LIMIT` / `OFFSET` clause value into a `usize`. The clause
/// must be a non-negative integer literal; anything else is rejected (no
/// dynamic LIMITs, no expressions). `kind` is used for error messages.
fn usize_from_literal(e: &SqlExpr, kind: &str) -> BoltResult<usize> {
    let value = match e {
        SqlExpr::Value(Value::Number(n, _)) => n,
        other => {
            return Err(BoltError::Sql(format!(
                "{kind} must be an integer literal, got: {other}"
            )));
        }
    };
    let parsed: i64 = value.parse().map_err(|_| {
        BoltError::Sql(format!("{kind} value '{value}' is not a valid integer"))
    })?;
    if parsed < 0 {
        return Err(BoltError::Sql(format!(
            "{kind} value must be non-negative, got {parsed}"
        )));
    }
    usize::try_from(parsed)
        .map_err(|_| BoltError::Sql(format!("{kind} value {parsed} exceeds usize range")))
}

/// Lower a `Select` into Scan [→ Filter] → (Project | Aggregate), optionally
/// wrapped in `Filter` (for HAVING) and/or `Distinct` (for SELECT DISTINCT).
/// Supports a single INNER JOIN in FROM.
fn plan_select(select: &Select, provider: &dyn TableProvider) -> BoltResult<LogicalPlan> {
    reject_unsupported_select(select)?;

    // FROM: exactly one base table reference. JOINs hang off `twj.joins`.
    if select.from.len() != 1 {
        return Err(BoltError::Sql(format!(
            "expected exactly one FROM table, got {}",
            select.from.len()
        )));
    }
    let twj = &select.from[0];

    // Build the base Scan from the first table reference.
    let (table_name, scan_schema) = lower_table_factor(&twj.relation, provider)?;
    let schema = scan_schema.clone();
    // The name resolver tracks the FROM-tree's `table.col` namespace so we
    // can resolve qualified references in WHERE / SELECT / GROUP BY / HAVING
    // (the ON-clause lowerer keeps its own simpler path — see `lower_join_side`).
    let mut resolver = NameResolver::empty();
    resolver.push_base(table_name.clone(), &scan_schema);
    let mut plan = LogicalPlan::Scan {
        table: table_name,
        projection: None,
        schema,
    };

    // JOIN handling. Supports INNER / LEFT / RIGHT / FULL with an equi-
    // conjunction ON predicate, plus CROSS (no ON clause). The join's
    // right side must itself be a bare table. Conjunctions of
    // `left.col = right.col` equalities; non-equi predicates remain
    // rejected. The host-side executor in `src/exec/join.rs` handles all
    // five join kinds.
    for join in &twj.joins {
        if join.global {
            return Err(BoltError::Sql(
                "unsupported: GLOBAL JOIN (ClickHouse extension)".into(),
            ));
        }
        // Pick out the (join_type, optional ON expr) pair. CROSS JOIN
        // has no ON clause — sqlparser models it with its own variant.
        let (join_type, on_expr) = match &join.join_operator {
            JoinOperator::Inner(c) => (JoinType::Inner, lower_join_constraint(c, "INNER")?),
            JoinOperator::LeftOuter(c) => (JoinType::LeftOuter, lower_join_constraint(c, "LEFT")?),
            JoinOperator::RightOuter(c) => {
                (JoinType::RightOuter, lower_join_constraint(c, "RIGHT")?)
            }
            JoinOperator::FullOuter(c) => (JoinType::FullOuter, lower_join_constraint(c, "FULL")?),
            JoinOperator::CrossJoin => (JoinType::Cross, None),
            other => {
                return Err(BoltError::Sql(format!(
                    "unsupported join kind: {other:?}; \
                     supported: INNER, LEFT, RIGHT, FULL OUTER, CROSS"
                )));
            }
        };
        let (rhs_table, rhs_schema) = lower_table_factor(&join.relation, provider)?;
        // Extend the resolver before we move `rhs_table` / `rhs_schema` into
        // the right-side Scan, so it sees the same rename rule as
        // `join_combined_schema` applies to the actual plan output.
        resolver.push_join(rhs_table.clone(), &rhs_schema);
        let right_plan = LogicalPlan::Scan {
            table: rhs_table,
            projection: None,
            schema: rhs_schema,
        };
        // CROSS has no ON predicate; everything else does. `lower_join_on`
        // is reused for the rest so non-equi forms keep a single
        // rejection path.
        let on_pairs = match on_expr {
            Some(e) => lower_join_on(e)?,
            None => Vec::new(),
        };
        plan = LogicalPlan::Join {
            left: Box::new(plan),
            right: Box::new(right_plan),
            join_type,
            on: on_pairs,
        };
    }
    // After a JOIN, the namespace for WHERE / SELECT items widens to the
    // join's output schema. The scan_schema below is still used for wildcard
    // expansion when no JOIN is present; when a JOIN *is* present, wildcard
    // expansion uses the join's full schema. We compute `scan_schema_for_wildcard`
    // for the wildcard-expansion branch below.
    let scan_schema_for_wildcard: Schema = if twj.joins.is_empty() {
        scan_schema.clone()
    } else {
        plan.schema()?
    };

    // WHERE
    if let Some(filter_sql) = &select.selection {
        let predicate = lower_expr(filter_sql, &resolver)?;
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate,
        };
    }

    // GROUP BY (must precede projection decision)
    let group_by_sql: Vec<&SqlExpr> = match &select.group_by {
        GroupByExpr::All(_) => {
            return Err(BoltError::Sql("unsupported: GROUP BY ALL".into()));
        }
        GroupByExpr::Expressions(exprs, modifiers) => {
            if !modifiers.is_empty() {
                return Err(BoltError::Sql(
                    "unsupported: GROUP BY modifiers (ROLLUP/CUBE/TOTALS)".into(),
                ));
            }
            exprs.iter().collect()
        }
    };

    // Expand SELECT items into (expr, optional alias). Wildcards expand to columns
    // of the scan's full schema.
    let mut items: Vec<(SqlExpr, Option<String>)> = Vec::new();
    for item in &select.projection {
        match item {
            SelectItem::UnnamedExpr(e) => items.push((e.clone(), None)),
            SelectItem::ExprWithAlias { expr, alias } => {
                items.push((expr.clone(), Some(alias.value.clone())))
            }
            SelectItem::Wildcard(_) => {
                for f in &scan_schema_for_wildcard.fields {
                    items.push((SqlExpr::Identifier(Ident::new(f.name.clone())), None));
                }
            }
            SelectItem::QualifiedWildcard(_, _) => {
                return Err(BoltError::Sql("unsupported: qualified wildcard".into()));
            }
        }
    }

    let has_agg_in_select = items
        .iter()
        .map(|(e, _)| try_aggregate(e, &resolver))
        .collect::<BoltResult<Vec<_>>>()?
        .iter()
        .any(|o| o.is_some());

    if has_agg_in_select || !group_by_sql.is_empty() {
        // Aggregate mode. Simplification: every selected item is either a bare
        // aggregate call or a bare group key already listed in GROUP BY. Mixed
        // post-aggregate scalar work (e.g. `SUM(a) + 1`) is rejected up front.
        let group_by: Vec<Expr> = group_by_sql
            .iter()
            .map(|e| lower_expr(e, &resolver))
            .collect::<BoltResult<_>>()?;

        let mut aggregates: Vec<AggregateExpr> = Vec::new();
        // For each SELECT item, remember how to pull it back out of the Aggregate
        // node's schema (group keys first, aggregates second per `Aggregate::schema()`).
        // Each entry is the *output* column name produced by the Aggregate, plus an
        // optional SELECT alias to rename it to in the final projection.
        enum SelectSource {
            /// SELECT references a group key; pull by the key's name in the Aggregate schema.
            GroupKey { key_name: String, alias: Option<String> },
            /// SELECT references the Nth aggregate in `aggregates`.
            Aggregate { index: usize },
        }
        let mut select_sources: Vec<SelectSource> = Vec::new();

        for (sql_expr, alias) in &items {
            if let Some(agg) = try_aggregate(sql_expr, &resolver)? {
                if alias.is_some() {
                    return Err(BoltError::Sql(
                        "unsupported: alias on aggregate expression".into(),
                    ));
                }
                let idx = aggregates.len();
                aggregates.push(agg);
                select_sources.push(SelectSource::Aggregate { index: idx });
                continue;
            }
            // Non-aggregate: must contain no nested aggregate (no post-aggregate exprs).
            if contains_aggregate(sql_expr, &resolver)? {
                return Err(BoltError::Sql(
                    "post-aggregate expressions not yet supported".into(),
                ));
            }
            let lowered = lower_expr(sql_expr, &resolver)?;
            // Must match some declared GROUP BY key by structural equality of the lowered form.
            if !group_by.iter().any(|g| expr_eq(g, &lowered)) {
                return Err(BoltError::Sql(
                    "non-aggregate SELECT expression must appear in GROUP BY".into(),
                ));
            }
            // Determine the *output* column name this key receives inside the Aggregate's
            // schema. Must mirror the naming rule in `LogicalPlan::schema()` for the
            // Aggregate arm: bare Column => its name, Alias => its name, else `__group_{i}`.
            // The aggregate plan's group_by list is the GROUP BY clause itself (not the
            // SELECT list), so we look up the matching key there to compute its position
            // and apply the same naming convention.
            let key_pos = group_by
                .iter()
                .position(|g| expr_eq(g, &lowered))
                .expect("matched above");
            let key_name = group_key_output_name(&group_by[key_pos], key_pos);
            select_sources.push(SelectSource::GroupKey {
                key_name,
                alias: alias.clone(),
            });
        }

        // The plan's `group_by` is the SQL GROUP BY list (not the SELECT keys);
        // this matches LogicalPlan::Aggregate's contract and types-checks the keys
        // even if SELECT names only a subset of them.
        let aggregate_plan = LogicalPlan::Aggregate {
            input: Box::new(plan),
            group_by,
            aggregates,
        };

        // Re-project the aggregate's output to honour SELECT-list column order
        // (Aggregate::schema places keys first, aggregates second — independent
        // of the user's SELECT order, which would silently swap columns).
        //
        // Aggregate output names follow `AggregateExpr::output_name()` in
        // `logical_plan.rs` (e.g. SUM(x) -> "sum_x", COUNT(*) -> "count"). That
        // method is private, so we mirror the same convention here in
        // `aggregate_output_name`. Group-key names mirror the rule in
        // `LogicalPlan::schema()` for the Aggregate arm.
        let aggregates_out: &[AggregateExpr] = match &aggregate_plan {
            LogicalPlan::Aggregate { aggregates, .. } => aggregates,
            _ => unreachable!("just constructed an Aggregate"),
        };
        let mut proj_exprs: Vec<Expr> = Vec::with_capacity(select_sources.len());
        for src in &select_sources {
            match src {
                SelectSource::GroupKey { key_name, alias } => {
                    let col = Expr::Column(key_name.clone());
                    proj_exprs.push(match alias {
                        Some(a) => col.alias(a.clone()),
                        None => col,
                    });
                }
                SelectSource::Aggregate { index } => {
                    let name = aggregate_output_name(&aggregates_out[*index]);
                    proj_exprs.push(Expr::Column(name));
                }
            }
        }

        plan = LogicalPlan::Project {
            input: Box::new(aggregate_plan),
            exprs: proj_exprs,
        };
    } else {
        // Scalar projection mode.
        if select.having.is_some() {
            return Err(BoltError::Sql(
                "HAVING requires GROUP BY or aggregate functions in SELECT".into(),
            ));
        }
        let mut exprs = Vec::with_capacity(items.len());
        for (sql_expr, alias) in items {
            let lowered = lower_expr(&sql_expr, &resolver)?;
            let lowered = match alias {
                Some(name) => lowered.alias(name),
                None => lowered,
            };
            exprs.push(lowered);
        }
        plan = LogicalPlan::Project {
            input: Box::new(plan),
            exprs,
        };
    }

    // HAVING: wrap the (SELECT-ordered) projection with a Filter. The
    // predicate references aggregate output column names by the names
    // generated in `AggregateExpr::output_name` (mirrored in
    // `aggregate_output_name` above), or group-key column names.
    //
    // SQL allows aggregate function *calls* inside HAVING (e.g.
    // `HAVING SUM(price) > 100`) — the GROUP BY has already established an
    // aggregation context. We rewrite each such call into a `Column`
    // reference using the same name the SELECT-order Project produced for
    // it. Non-aggregate sub-expressions go through the regular
    // `lower_expr`, which also handles bare group-key columns.
    if let Some(having_sql) = &select.having {
        let predicate = lower_expr_in_having(having_sql, &resolver)?;
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate,
        };
    }

    // SELECT DISTINCT: dedup the *output* rows (after projection, HAVING).
    if matches!(select.distinct, Some(Distinct::Distinct)) {
        plan = LogicalPlan::Distinct {
            input: Box::new(plan),
        };
    }

    Ok(plan)
}

/// Lower a single `TableFactor` into `(table_name, schema)`. Only bare
/// table references are accepted (no aliases, TVFs, version, hints, etc.).
fn lower_table_factor(
    tf: &TableFactor,
    provider: &dyn TableProvider,
) -> BoltResult<(String, Schema)> {
    match tf {
        TableFactor::Table {
            name,
            alias,
            args,
            with_hints,
            version,
            with_ordinality,
            partitions,
        } => {
            if alias.is_some() {
                return Err(BoltError::Sql("unsupported: table alias".into()));
            }
            if args.is_some() {
                return Err(BoltError::Sql("unsupported: table-valued function".into()));
            }
            if !with_hints.is_empty() {
                return Err(BoltError::Sql("unsupported: WITH hints".into()));
            }
            if version.is_some() {
                return Err(BoltError::Sql("unsupported: table version".into()));
            }
            if *with_ordinality {
                return Err(BoltError::Sql("unsupported: WITH ORDINALITY".into()));
            }
            if !partitions.is_empty() {
                return Err(BoltError::Sql("unsupported: PARTITION".into()));
            }
            let table_name = single_ident_from_object_name(name)?;
            let schema = provider.schema(&table_name)?;
            Ok((table_name, schema))
        }
        _ => Err(BoltError::Sql(
            "unsupported: only bare table references are allowed in FROM".into(),
        )),
    }
}

/// Pull the ON expression out of a `JoinConstraint` for non-CROSS joins.
/// USING / NATURAL get explicit rejections; an absent constraint is an
/// error for INNER/LEFT/RIGHT/FULL (all four require ON). CROSS doesn't
/// flow through this helper — the caller handles its `None` arm directly.
fn lower_join_constraint<'a>(
    c: &'a JoinConstraint,
    kind: &'static str,
) -> BoltResult<Option<&'a SqlExpr>> {
    match c {
        JoinConstraint::On(e) => Ok(Some(e)),
        JoinConstraint::Using(_) => Err(BoltError::Sql(format!(
            "unsupported: {kind} JOIN ... USING; rewrite as ON"
        ))),
        JoinConstraint::Natural => Err(BoltError::Sql(format!(
            "unsupported: NATURAL {kind} JOIN"
        ))),
        JoinConstraint::None => Err(BoltError::Sql(format!(
            "{kind} JOIN requires an ON clause"
        ))),
    }
}

/// Look up a join predicate expression as a conjunction of `left.col = right.col`
/// equalities. Reject non-equi joins and non-conjunctive forms with a clear
/// message; the executor scaffold only handles equi joins.
fn lower_join_on(e: &SqlExpr) -> BoltResult<Vec<(Expr, Expr)>> {
    let mut out = Vec::new();
    collect_join_eq(e, &mut out)?;
    if out.is_empty() {
        return Err(BoltError::Sql(
            "JOIN ON clause must contain at least one equality predicate".into(),
        ));
    }
    Ok(out)
}

/// Walk `e` flattening `AND` nodes; each leaf must be `<expr> = <expr>`.
fn collect_join_eq(e: &SqlExpr, out: &mut Vec<(Expr, Expr)>) -> BoltResult<()> {
    match e {
        SqlExpr::Nested(inner) => collect_join_eq(inner, out),
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            collect_join_eq(left, out)?;
            collect_join_eq(right, out)
        }
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } => {
            let l = lower_join_side(left)?;
            let r = lower_join_side(right)?;
            out.push((l, r));
            Ok(())
        }
        other => Err(BoltError::Sql(format!(
            "non-equi JOIN not yet supported (ON clause must be a conjunction of `a = b` predicates; got {other})"
        ))),
    }
}

/// Lower one side of an equi-join predicate. We accept either a bare
/// identifier or a `table.column` qualified identifier so users can
/// disambiguate same-named columns; both lower to a plain `Column` ref
/// (qualified column lookups beyond bare-name matching aren't supported
/// in 0.1.x but the parser accepts them so error messages stay friendly).
fn lower_join_side(e: &SqlExpr) -> BoltResult<Expr> {
    match e {
        SqlExpr::Identifier(ident) => Ok(Expr::Column(ident.value.clone())),
        SqlExpr::CompoundIdentifier(parts) => {
            // `table.col` — keep only the trailing column name. Cross-side
            // matching is the executor's job.
            let last = parts
                .last()
                .ok_or_else(|| BoltError::Sql("empty compound identifier in JOIN ON".into()))?;
            Ok(Expr::Column(last.value.clone()))
        }
        other => Err(BoltError::Sql(format!(
            "non-equi JOIN not yet supported (JOIN ON sides must be column references; got {other})"
        ))),
    }
}

/// Reject SELECT-level features outside our supported subset. `DISTINCT` and
/// `HAVING` are *not* rejected here — both are recognised by `plan_select`
/// and lowered into the plan.
fn reject_unsupported_select(select: &Select) -> BoltResult<()> {
    // DISTINCT ON (...) is a Postgres extension we don't support; plain
    // SELECT DISTINCT is handled by `plan_select`.
    if let Some(Distinct::On(_)) = &select.distinct {
        return Err(BoltError::Sql("unsupported: DISTINCT ON".into()));
    }
    if select.top.is_some() {
        return Err(BoltError::Sql("unsupported: TOP".into()));
    }
    if select.into.is_some() {
        return Err(BoltError::Sql("unsupported: SELECT INTO".into()));
    }
    if !select.lateral_views.is_empty() {
        return Err(BoltError::Sql("unsupported: LATERAL VIEW".into()));
    }
    if select.prewhere.is_some() {
        return Err(BoltError::Sql("unsupported: PREWHERE".into()));
    }
    if !select.cluster_by.is_empty() {
        return Err(BoltError::Sql("unsupported: CLUSTER BY".into()));
    }
    if !select.distribute_by.is_empty() {
        return Err(BoltError::Sql("unsupported: DISTRIBUTE BY".into()));
    }
    if !select.sort_by.is_empty() {
        return Err(BoltError::Sql("unsupported: SORT BY".into()));
    }
    if !select.named_window.is_empty() {
        return Err(BoltError::Sql("unsupported: WINDOW".into()));
    }
    if select.qualify.is_some() {
        return Err(BoltError::Sql("unsupported: QUALIFY".into()));
    }
    if select.value_table_mode.is_some() {
        return Err(BoltError::Sql("unsupported: SELECT AS STRUCT/VALUE".into()));
    }
    if select.connect_by.is_some() {
        return Err(BoltError::Sql("unsupported: CONNECT BY".into()));
    }
    Ok(())
}

/// Pull a single-part identifier out of an `ObjectName`, rejecting schema-qualified names.
fn single_ident_from_object_name(name: &ObjectName) -> BoltResult<String> {
    if name.0.len() != 1 {
        return Err(BoltError::Sql(format!(
            "qualified table names not supported: {name}"
        )));
    }
    Ok(name.0[0].value.clone())
}

/// Recognize a top-level aggregate function call. Returns `Ok(None)` for non-aggregates.
fn try_aggregate(e: &SqlExpr, resolver: &NameResolver) -> BoltResult<Option<AggregateExpr>> {
    let func = match e {
        SqlExpr::Function(f) => f,
        _ => return Ok(None),
    };
    if func.name.0.len() != 1 {
        return Ok(None);
    }
    let fname = func.name.0[0].value.to_ascii_uppercase();
    let kind = match fname.as_str() {
        "COUNT" | "SUM" | "MIN" | "MAX" | "AVG" => fname,
        _ => return Ok(None),
    };

    // Disallow OVER (window), FILTER, ORDER BY, WITHIN GROUP, parameters.
    if func.over.is_some() {
        return Err(BoltError::Sql(
            "unsupported: window functions (OVER)".into(),
        ));
    }
    if func.filter.is_some() {
        return Err(BoltError::Sql("unsupported: FILTER on aggregate".into()));
    }
    if func.null_treatment.is_some() {
        return Err(BoltError::Sql(
            "unsupported: IGNORE/RESPECT NULLS on aggregate".into(),
        ));
    }
    if !func.within_group.is_empty() {
        return Err(BoltError::Sql(
            "unsupported: WITHIN GROUP on aggregate".into(),
        ));
    }
    if !matches!(func.parameters, FunctionArguments::None) {
        return Err(BoltError::Sql(
            "unsupported: parametric aggregate function".into(),
        ));
    }

    let arg_list = match &func.args {
        FunctionArguments::List(list) => list,
        FunctionArguments::None => {
            return Err(BoltError::Sql(format!("{kind} requires arguments")));
        }
        FunctionArguments::Subquery(_) => {
            return Err(BoltError::Sql(format!(
                "unsupported: subquery argument to {kind}"
            )));
        }
    };
    if arg_list.duplicate_treatment.is_some() {
        return Err(BoltError::Sql(format!(
            "unsupported: DISTINCT/ALL inside {kind}"
        )));
    }
    if !arg_list.clauses.is_empty() {
        return Err(BoltError::Sql(format!(
            "unsupported: argument clauses on {kind}"
        )));
    }
    if arg_list.args.len() != 1 {
        return Err(BoltError::Sql(format!(
            "{kind} expects exactly one argument, got {}",
            arg_list.args.len()
        )));
    }

    let arg_expr = match &arg_list.args[0] {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => Some(e),
        FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => None,
        FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(_)) => {
            return Err(BoltError::Sql(format!(
                "unsupported: qualified wildcard in {kind}"
            )));
        }
        FunctionArg::Named { .. } => {
            return Err(BoltError::Sql(format!(
                "unsupported: named argument to {kind}"
            )));
        }
    };

    let inner = match arg_expr {
        Some(e) => lower_expr(e, resolver)?,
        None => {
            if kind != "COUNT" {
                return Err(BoltError::Sql(format!("{kind}(*) is not supported")));
            }
            // COUNT(*) sentinel: a literal 1; counted rows are independent of value.
            Expr::Literal(Literal::Int64(1))
        }
    };

    Ok(Some(match kind.as_str() {
        "COUNT" => AggregateExpr::Count(inner),
        "SUM" => AggregateExpr::Sum(inner),
        "MIN" => AggregateExpr::Min(inner),
        "MAX" => AggregateExpr::Max(inner),
        "AVG" => AggregateExpr::Avg(inner),
        _ => unreachable!("kind already filtered above"),
    }))
}

/// True if `e` contains any aggregate function call (anywhere in the tree).
fn contains_aggregate(e: &SqlExpr, resolver: &NameResolver) -> BoltResult<bool> {
    if try_aggregate(e, resolver)?.is_some() {
        return Ok(true);
    }
    match e {
        SqlExpr::BinaryOp { left, right, .. } => {
            Ok(contains_aggregate(left, resolver)? || contains_aggregate(right, resolver)?)
        }
        SqlExpr::UnaryOp { expr, .. } => contains_aggregate(expr, resolver),
        SqlExpr::Nested(inner) => contains_aggregate(inner, resolver),
        _ => Ok(false),
    }
}

/// Variant of `lower_expr` used inside a HAVING clause. Aggregate function
/// calls (anywhere in the tree) are rewritten into a bare `Column(name)`
/// where `name` is the column the post-aggregate Project produces for that
/// aggregate (per `aggregate_output_name`). Everything else delegates to
/// `lower_expr`, which keeps the usual rules — bare columns become column
/// refs, non-aggregate function calls are still rejected, etc.
fn lower_expr_in_having(e: &SqlExpr, resolver: &NameResolver) -> BoltResult<Expr> {
    if let Some(agg) = try_aggregate(e, resolver)? {
        return Ok(Expr::Column(aggregate_output_name(&agg)));
    }
    match e {
        SqlExpr::Nested(inner) => lower_expr_in_having(inner, resolver),
        SqlExpr::BinaryOp { left, op, right } => {
            let lop = lower_binary_op(op)?;
            let l = lower_expr_in_having(left, resolver)?;
            let r = lower_expr_in_having(right, resolver)?;
            Ok(Expr::Binary {
                op: lop,
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        SqlExpr::UnaryOp { op, expr } => match op {
            UnaryOperator::Plus => lower_expr_in_having(expr, resolver),
            UnaryOperator::Minus => {
                // Re-use the aggregate-aware lowerer for the operand, then
                // negate by hand (we can't fall through to `negate_expr`
                // because it would route through `lower_expr` and reject
                // any aggregate call nested under the unary minus).
                let inner = lower_expr_in_having(expr, resolver)?;
                Ok(Expr::Binary {
                    op: BinaryOp::Sub,
                    left: Box::new(Expr::Literal(Literal::Int64(0))),
                    right: Box::new(inner),
                })
            }
            other => Err(BoltError::Sql(format!(
                "unsupported unary operator: {other:?}"
            ))),
        },
        // Anything else is identical to a scalar HAVING fragment; defer to
        // the normal lowerer (which handles Identifier, Value, etc., and
        // still rejects bare non-aggregate Function calls).
        _ => lower_expr(e, resolver),
    }
}

/// Lower a scalar SQL expression into our `Expr`. Aggregates are rejected here —
/// callers must split them off via `try_aggregate` first.
///
/// Qualified column references (`table.col`) are resolved against `resolver`
/// to the output column name produced by the FROM-tree's combined schema;
/// see [`NameResolver`] for the rule. Bare `col` references pass through as
/// `Expr::Column(col)` — downstream type-checking validates that the name
/// exists in scope and (for JOINs) follows the leftmost-wins convention
/// enforced by [`join_combined_schema`](crate::plan::logical_plan::join_combined_schema).
fn lower_expr(e: &SqlExpr, resolver: &NameResolver) -> BoltResult<Expr> {
    match e {
        SqlExpr::Identifier(ident) => Ok(Expr::Column(ident.value.clone())),
        SqlExpr::CompoundIdentifier(parts) => {
            // We currently only support a single `table.column` qualifier
            // (no schema-qualified or struct-field forms). The frontend has
            // no schema/database concept, so anything deeper is meaningless.
            if parts.len() != 2 {
                return Err(BoltError::Sql(format!(
                    "unsupported: deeply qualified column reference '{}'",
                    parts
                        .iter()
                        .map(|p| p.value.as_str())
                        .collect::<Vec<_>>()
                        .join(".")
                )));
            }
            let qualifier = &parts[0].value;
            let col = &parts[1].value;
            let resolved = resolver.resolve_compound(qualifier, col)?;
            Ok(Expr::Column(resolved))
        }
        SqlExpr::Value(v) => lower_value(v),
        SqlExpr::Nested(inner) => lower_expr(inner, resolver),
        SqlExpr::BinaryOp { left, op, right } => {
            let lop = lower_binary_op(op)?;
            let l = lower_expr(left, resolver)?;
            let r = lower_expr(right, resolver)?;
            Ok(Expr::Binary {
                op: lop,
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        SqlExpr::UnaryOp { op, expr } => match op {
            UnaryOperator::Plus => lower_expr(expr, resolver),
            UnaryOperator::Minus => negate_expr(expr, resolver),
            other => Err(BoltError::Sql(format!(
                "unsupported unary operator: {other:?}"
            ))),
        },
        SqlExpr::Function(_) => Err(BoltError::Sql(
            "function calls are only allowed as top-level aggregates in SELECT".into(),
        )),
        other => Err(BoltError::Sql(format!(
            "unsupported expression: {other}"
        ))),
    }
}

/// Translate a SQL literal `Value` into our `Literal` expression.
fn lower_value(v: &Value) -> BoltResult<Expr> {
    match v {
        Value::Number(n, _long) => parse_number(n),
        Value::SingleQuotedString(s) => Ok(Expr::Literal(Literal::Utf8(s.clone()))),
        Value::Boolean(b) => Ok(Expr::Literal(Literal::Bool(*b))),
        Value::Null => Ok(Expr::Literal(Literal::Null)),
        other => Err(BoltError::Sql(format!("unsupported literal: {other}"))),
    }
}

/// True if `s` is written as a pure integer (no decimal point, no exponent).
/// Used to distinguish "user meant an integer that overflows" from
/// "user wrote a float that happens to round-trip through f64".
fn looks_like_pure_integer(s: &str) -> bool {
    !s.contains('.') && !s.contains('e') && !s.contains('E')
}

/// Parse a numeric literal string into `Int64` if it fits, otherwise `Float64`.
/// Integer-looking literals that overflow `i64` are *rejected* rather than silently
/// demoted to `Float64` (which would lose precision past 2^53).
fn parse_number(n: &str) -> BoltResult<Expr> {
    if let Ok(i) = n.parse::<i64>() {
        return Ok(Expr::Literal(Literal::Int64(i)));
    }
    if looks_like_pure_integer(n) {
        return Err(BoltError::Sql(format!(
            "integer literal {n} out of i64 range; use scientific notation or an explicit fractional part for Float64"
        )));
    }
    match n.parse::<f64>() {
        Ok(f) => Ok(Expr::Literal(Literal::Float64(f))),
        Err(_) => Err(BoltError::Sql(format!("invalid number literal '{n}'"))),
    }
}

/// Fold `-<number-literal>` into a single signed literal; otherwise lower as `0 - expr`.
/// The asymmetric `i64` range (`MIN = -2^63`, `MAX = 2^63 - 1`) is handled by
/// trying `i64::from_str` on the *negated* string, which succeeds at `i64::MIN`
/// even though `2^63` does not fit in a positive `i64`.
fn negate_expr(e: &SqlExpr, resolver: &NameResolver) -> BoltResult<Expr> {
    if let SqlExpr::Value(Value::Number(n, _)) = e {
        // Common case: positive literal fits in i64; just negate.
        if let Ok(i) = n.parse::<i64>() {
            return Ok(Expr::Literal(Literal::Int64(-i)));
        }
        // Edge case: -i64::MIN. The positive form "9223372036854775808" overflows
        // i64, but the negated literal "-9223372036854775808" parses cleanly.
        let negated = format!("-{n}");
        if let Ok(i) = negated.parse::<i64>() {
            return Ok(Expr::Literal(Literal::Int64(i)));
        }
        // Integer-looking but still out of range (e.g. -10^20): reject, do not
        // silently demote to Float64.
        if looks_like_pure_integer(n) {
            return Err(BoltError::Sql(format!(
                "integer literal -{n} out of i64 range; use scientific notation or an explicit fractional part for Float64"
            )));
        }
        if let Ok(f) = n.parse::<f64>() {
            return Ok(Expr::Literal(Literal::Float64(-f)));
        }
        return Err(BoltError::Sql(format!("invalid number literal '{n}'")));
    }
    let inner = lower_expr(e, resolver)?;
    Ok(Expr::Binary {
        op: BinaryOp::Sub,
        left: Box::new(Expr::Literal(Literal::Int64(0))),
        right: Box::new(inner),
    })
}

/// Mirror of the (private) `AggregateExpr::output_name` rule in
/// `logical_plan.rs`. Kept in sync by inspection — if that rule changes, this
/// must change with it. Used to re-project aggregate results in SELECT order.
fn aggregate_output_name(agg: &AggregateExpr) -> String {
    fn suffix(e: &Expr) -> String {
        match e {
            Expr::Column(n) => format!("_{n}"),
            Expr::Alias(_, n) => format!("_{n}"),
            _ => String::new(),
        }
    }
    match agg {
        AggregateExpr::Count(e) => format!("count{}", suffix(e)),
        AggregateExpr::Sum(e) => format!("sum{}", suffix(e)),
        AggregateExpr::Min(e) => format!("min{}", suffix(e)),
        AggregateExpr::Max(e) => format!("max{}", suffix(e)),
        AggregateExpr::Avg(e) => format!("avg{}", suffix(e)),
    }
}

/// Mirror of the group-key naming rule inside `LogicalPlan::schema()` for the
/// `Aggregate` arm in `logical_plan.rs`. Kept in sync by inspection.
fn group_key_output_name(key: &Expr, idx: usize) -> String {
    match key {
        Expr::Column(n) => n.clone(),
        Expr::Alias(_, n) => n.clone(),
        _ => format!("__group_{idx}"),
    }
}

/// Map a `sqlparser` `BinaryOperator` onto our small `BinaryOp` set; reject anything else.
fn lower_binary_op(op: &BinaryOperator) -> BoltResult<BinaryOp> {
    Ok(match op {
        BinaryOperator::Plus => BinaryOp::Add,
        BinaryOperator::Minus => BinaryOp::Sub,
        BinaryOperator::Multiply => BinaryOp::Mul,
        BinaryOperator::Divide => BinaryOp::Div,
        BinaryOperator::Eq => BinaryOp::Eq,
        BinaryOperator::NotEq => BinaryOp::NotEq,
        BinaryOperator::Lt => BinaryOp::Lt,
        BinaryOperator::LtEq => BinaryOp::LtEq,
        BinaryOperator::Gt => BinaryOp::Gt,
        BinaryOperator::GtEq => BinaryOp::GtEq,
        BinaryOperator::And => BinaryOp::And,
        BinaryOperator::Or => BinaryOp::Or,
        other => {
            return Err(BoltError::Sql(format!(
                "unsupported binary operator: {other}"
            )));
        }
    })
}

/// Structural equality of two lowered `Expr` trees (ignoring aliases at the root).
fn expr_eq(a: &Expr, b: &Expr) -> bool {
    let a = strip_alias(a);
    let b = strip_alias(b);
    match (a, b) {
        (Expr::Column(x), Expr::Column(y)) => x == y,
        (Expr::Literal(x), Expr::Literal(y)) => x == y,
        (
            Expr::Binary {
                op: ao,
                left: al,
                right: ar,
            },
            Expr::Binary {
                op: bo,
                left: bl,
                right: br,
            },
        ) => ao == bo && expr_eq(al, bl) && expr_eq(ar, br),
        _ => false,
    }
}

/// Peel one or more `Alias` wrappers off the root.
fn strip_alias(e: &Expr) -> &Expr {
    let mut cur = e;
    while let Expr::Alias(inner, _) = cur {
        cur = inner;
    }
    cur
}

#[cfg(test)]
mod plan_cache_tests {
    //! Unit tests for the parse + plan cache. These deliberately exercise
    //! `PlanCache` directly with explicit capacities instead of going
    //! through the env-var-frozen global cap, so the eviction case can be
    //! tested deterministically without depending on whether some other
    //! `#[test]` in this binary already initialised `PLAN_CACHE`. The
    //! env-var parsing is exercised separately via `parse_plan_cache_cap`.
    use super::*;
    use crate::plan::logical_plan::{DataType, Field};

    /// Two-table fixture matching the one in `wave7_tests::provider` but
    /// kept local so this module's tests don't depend on test-helper
    /// ordering between sibling modules.
    fn provider() -> MemTableProvider {
        let t1 = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int64, false),
        ]);
        let t2 = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("c", DataType::Float64, false),
        ]);
        MemTableProvider::new()
            .with_table("t1", t1)
            .with_table("t2", t2)
    }

    /// Build a tiny `Arc<LogicalPlan>` for direct cache tests. Using a
    /// hand-rolled `Scan` keeps the tests independent of the lowering
    /// pipeline (so a future change to `parse_uncached` doesn't break the
    /// pure cache-policy tests below).
    fn dummy_plan(table: &str) -> Arc<LogicalPlan> {
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, false)]);
        Arc::new(LogicalPlan::Scan {
            table: table.to_string(),
            projection: None,
            schema,
        })
    }

    fn key(sql: &str, version: u64) -> PlanCacheKey {
        PlanCacheKey {
            sql: sql.to_string(),
            version,
        }
    }

    #[test]
    fn parse_cache_cap_picks_up_env_var() {
        // End-to-end: the env-var parser handles set / unset / zero /
        // garbage. The global `plan_cache_cap()` is memoised, so we test
        // the pure helper instead — same pattern as
        // `parse_cap_picks_up_env_var` in `jit_compiler`.
        assert_eq!(parse_plan_cache_cap(Some("8"), 64), 8);
        assert_eq!(parse_plan_cache_cap(None, 64), 64);
        assert_eq!(parse_plan_cache_cap(Some(""), 64), 64);
        assert_eq!(parse_plan_cache_cap(Some("0"), 64), 64);
        assert_eq!(parse_plan_cache_cap(Some("not-a-number"), 64), 64);
    }

    #[test]
    fn cache_with_zero_capacity_promotes_to_one() {
        // Misconfigured env var -> cap 0 was rejected upstream by
        // `parse_plan_cache_cap`, but `PlanCache::with_capacity` is also
        // used directly (from this test module). Defensively bump 0 to 1
        // so inserts can succeed.
        let mut cache = PlanCache::with_capacity(0);
        cache.insert(key("SELECT 1", 0), dummy_plan("t1"));
        assert!(cache.lookup(&key("SELECT 1", 0)).is_some());
    }

    #[test]
    fn same_sql_twice_second_is_a_hit() {
        let mut cache = PlanCache::with_capacity(8);
        let plan = dummy_plan("t1");
        // First lookup: miss, no entry yet.
        assert!(cache.lookup(&key("SELECT a FROM t1", 1)).is_none());
        // Producer inserts after the miss.
        cache.insert(key("SELECT a FROM t1", 1), Arc::clone(&plan));
        // Second lookup: hit.
        let got = cache.lookup(&key("SELECT a FROM t1", 1));
        assert!(got.is_some(), "expected hit on identical (sql, version)");
        let (hits, misses, evictions) = cache.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(evictions, 0);
    }

    #[test]
    fn two_different_sql_strings_are_two_misses() {
        let mut cache = PlanCache::with_capacity(8);
        // Both lookups miss; counts increment independently.
        assert!(cache.lookup(&key("SELECT a FROM t1", 1)).is_none());
        assert!(cache.lookup(&key("SELECT b FROM t1", 1)).is_none());
        let (hits, misses, _) = cache.stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 2);
    }

    #[test]
    fn register_table_invalidates_previously_cached_sql() {
        // Same SQL string but a different schema version: the cache
        // treats them as distinct keys, so the post-mutation lookup is a
        // miss even though the SQL is byte-identical.
        let mut cache = PlanCache::with_capacity(8);
        let sql = "SELECT a FROM t1";
        let v0 = 1u64;
        cache.insert(key(sql, v0), dummy_plan("t1"));
        assert!(cache.lookup(&key(sql, v0)).is_some(), "pre-mutation hit");

        let v1 = v0 + 1; // simulate `register_table` bumping the version.
        assert!(
            cache.lookup(&key(sql, v1)).is_none(),
            "post-mutation must miss — version bump invalidates the cached plan"
        );
    }

    #[test]
    fn register_table_bumps_schema_version_end_to_end() {
        // End-to-end through the public `MemTableProvider` API: any
        // mutation must produce a strictly-greater `schema_version`.
        let mut provider = MemTableProvider::new();
        let v_empty = provider.schema_version();
        provider.register(
            "t1",
            Schema::new(vec![Field::new("a", DataType::Int32, false)]),
        );
        let v_after_register = provider.schema_version();
        assert!(
            v_after_register > v_empty,
            "register should bump version: {v_empty} -> {v_after_register}"
        );

        let removed = provider.unregister_table("t1");
        assert!(removed, "table was present");
        let v_after_unregister = provider.schema_version();
        assert!(
            v_after_unregister > v_after_register,
            "unregister should bump version: {v_after_register} -> {v_after_unregister}"
        );

        // Set column nullability also bumps so the planner's
        // `input_has_validity` derivation invalidates cleanly.
        provider.register(
            "t2",
            Schema::new(vec![Field::new("a", DataType::Int32, false)]),
        );
        let v_before_null = provider.schema_version();
        provider.set_column_nullability("t2", "a", true);
        let v_after_null = provider.schema_version();
        assert!(
            v_after_null > v_before_null,
            "set_column_nullability should bump version"
        );
    }

    #[test]
    fn parse_round_trip_hits_global_cache() {
        // Run twice through the *public* `parse` against the same provider
        // and SQL; the second call's plan must be equal to the first.
        // Whether or not other tests in this binary have already populated
        // `PLAN_CACHE` does not affect this assertion — the cache promise
        // is *correctness*, not "n misses became n-1".
        let p = provider();
        let sql = "SELECT a FROM t1";
        let plan1 = parse(sql, &p).expect("first parse ok");
        let plan2 = parse(sql, &p).expect("second parse ok");
        // Cheap structural check: same Debug rendering means same lowered
        // tree (Debug derive on LogicalPlan is structural).
        assert_eq!(format!("{plan1:?}"), format!("{plan2:?}"));

        // The shared stats counters must have moved by at least 1 hit OR
        // 1 miss between the two `parse` calls. We can't assert exact
        // values because other tests in this binary share the same
        // counters; instead we measure deltas around our two calls.
        let (h0, m0, _) = plan_cache_stats();
        let _ = parse(sql, &p).expect("third parse ok");
        let (h1, m1, _) = plan_cache_stats();
        // Either we hit or we missed-then-recovered (race against
        // eviction from an unrelated test in the same binary). Both are
        // valid outcomes — the cache promise is correctness, not strict
        // hit counts under concurrent test interleaving.
        assert!(
            h1 + m1 > h0 + m0,
            "stats counters must advance on every parse"
        );
    }

    #[test]
    fn parse_against_different_provider_versions_does_not_alias() {
        // Two `MemTableProvider` instances with the same registered table
        // get DIFFERENT version tokens (the global counter is bumped on
        // every mutation). So the same SQL against the two providers
        // hits two different cache keys and lowers cleanly against each.
        let p1 = provider();
        let p2 = provider();
        assert_ne!(
            p1.schema_version(),
            p2.schema_version(),
            "distinct provider instances must have distinct version tokens"
        );
        let plan1 = parse("SELECT a FROM t1", &p1).unwrap();
        let plan2 = parse("SELECT a FROM t1", &p2).unwrap();
        // Same lowered shape — providers describe the same schemas.
        assert_eq!(format!("{plan1:?}"), format!("{plan2:?}"));
    }

    #[test]
    fn cache_evicts_oldest_when_full() {
        // Mirrors the spec: with capacity=2, inserting 3 entries evicts
        // the first. Uses `PlanCache::with_capacity` directly so the test
        // is independent of `CRATON_PLAN_CACHE_SIZE` (which the global
        // cache memoises on first use, making mid-binary env-var
        // mutation unreliable).
        let mut cache = PlanCache::with_capacity(2);
        let p = dummy_plan("scratch");

        cache.insert(key("Q1", 0), Arc::clone(&p));
        cache.insert(key("Q2", 0), Arc::clone(&p));
        assert!(cache.lookup(&key("Q1", 0)).is_some(), "Q1 still cached");
        assert!(cache.lookup(&key("Q2", 0)).is_some(), "Q2 still cached");

        // Third insert evicts the oldest (Q1).
        cache.insert(key("Q3", 0), Arc::clone(&p));
        assert!(
            cache.lookup(&key("Q1", 0)).is_none(),
            "Q1 must be FIFO-evicted at capacity 2"
        );
        assert!(cache.lookup(&key("Q2", 0)).is_some(), "Q2 survives");
        assert!(cache.lookup(&key("Q3", 0)).is_some(), "Q3 just-inserted");
        let (_, _, evictions) = cache.stats();
        assert_eq!(evictions, 1, "exactly one entry was evicted");
    }

    #[test]
    fn cache_insert_does_not_double_count_existing_key() {
        // Re-inserting the same key (e.g. two threads racing through the
        // miss path) leaves the cache size unchanged — no spurious
        // eviction, no duplicated deque entry.
        let mut cache = PlanCache::with_capacity(2);
        let p = dummy_plan("scratch");
        cache.insert(key("Q1", 0), Arc::clone(&p));
        cache.insert(key("Q1", 0), Arc::clone(&p));
        cache.insert(key("Q1", 0), Arc::clone(&p));
        // map size 1, deque length 1, zero evictions.
        assert_eq!(cache.map.len(), 1);
        assert_eq!(cache.order.len(), 1);
        let (_, _, ev) = cache.stats();
        assert_eq!(ev, 0);
    }

    #[test]
    fn register_table_through_parse_returns_invalidated_plan() {
        // End-to-end: parse SQL, mutate the provider (unregister a
        // referenced table), re-parse the same SQL — the second call
        // must surface the NEW provider state (an "unknown table"
        // error here), not a stale cached plan from before the
        // mutation.
        let mut p = provider();
        let sql = "SELECT a FROM t1";
        let _plan_pre = parse(sql, &p).expect("plan against full provider");
        assert!(p.unregister_table("t1"), "t1 was present");
        let err = parse(sql, &p).expect_err(
            "post-mutation parse must reach the lowerer and surface the schema error",
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown table"),
            "expected 'unknown table' error from re-parse, got: {msg}"
        );
    }
}

#[cfg(test)]
mod wave7_tests {
    //! Parse-and-lower smoke tests for wave 7 features (DISTINCT, LIMIT,
    //! ORDER BY, HAVING, UNION [ALL], INNER JOIN). These tests check only
    //! the logical / physical plan *shape* — actual execution is covered by
    //! the e2e suite and is out of scope here.
    use super::*;
    use crate::plan::logical_plan::DataType;
    use crate::plan::physical_plan::{lower, PhysicalPlan};

    /// Minimal two-table fixture with stable column dtypes for plan tests.
    fn provider() -> MemTableProvider {
        use crate::plan::logical_plan::Field;
        let t1 = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int64, false),
        ]);
        let t2 = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("c", DataType::Float64, false),
        ]);
        MemTableProvider::new()
            .with_table("t1", t1)
            .with_table("t2", t2)
    }

    fn lp(sql: &str) -> LogicalPlan {
        parse(sql, &provider()).unwrap_or_else(|e| panic!("parse failed for {sql:?}: {e}"))
    }

    fn pp(sql: &str) -> PhysicalPlan {
        let logical = lp(sql);
        lower(&logical).unwrap_or_else(|e| panic!("lower failed for {sql:?}: {e}"))
    }

    #[test]
    fn select_distinct_wraps_in_distinct() {
        let plan = lp("SELECT DISTINCT a FROM t1");
        assert!(
            matches!(plan, LogicalPlan::Distinct { .. }),
            "expected Distinct at top, got {plan:?}"
        );
        let phys = lower(&plan).unwrap();
        assert!(matches!(phys, PhysicalPlan::Distinct { .. }));
    }

    #[test]
    fn limit_offset_parses() {
        let phys = pp("SELECT a FROM t1 LIMIT 10 OFFSET 5");
        match phys {
            PhysicalPlan::Limit {
                limit,
                offset,
                ref input,
                ..
            } => {
                assert_eq!(limit, 10);
                assert_eq!(offset, 5);
                assert!(matches!(**input, PhysicalPlan::Projection { .. }));
            }
            other => panic!("expected Limit, got {other:?}"),
        }
    }

    #[test]
    fn offset_without_limit_uses_usize_max() {
        let phys = pp("SELECT a FROM t1 OFFSET 3");
        match phys {
            PhysicalPlan::Limit { limit, offset, .. } => {
                assert_eq!(limit, usize::MAX);
                assert_eq!(offset, 3);
            }
            other => panic!("expected Limit, got {other:?}"),
        }
    }

    #[test]
    fn order_by_default_direction_and_nulls() {
        let plan = lp("SELECT a FROM t1 ORDER BY a");
        match plan {
            LogicalPlan::Sort { sort_exprs, .. } => {
                assert_eq!(sort_exprs.len(), 1);
                assert!(!sort_exprs[0].descending);
                assert!(sort_exprs[0].nulls_first, "ASC defaults to NULLS FIRST");
            }
            other => panic!("expected Sort, got {other:?}"),
        }
    }

    #[test]
    fn order_by_desc_defaults_to_nulls_last() {
        let plan = lp("SELECT a FROM t1 ORDER BY a DESC");
        match plan {
            LogicalPlan::Sort { sort_exprs, .. } => {
                assert!(sort_exprs[0].descending);
                assert!(!sort_exprs[0].nulls_first, "DESC defaults to NULLS LAST");
            }
            other => panic!("expected Sort, got {other:?}"),
        }
    }

    #[test]
    fn order_by_with_explicit_nulls_first() {
        let plan = lp("SELECT a FROM t1 ORDER BY a DESC NULLS FIRST");
        match plan {
            LogicalPlan::Sort { sort_exprs, .. } => {
                assert!(sort_exprs[0].descending);
                assert!(sort_exprs[0].nulls_first);
            }
            other => panic!("expected Sort, got {other:?}"),
        }
    }

    #[test]
    fn order_by_then_limit_layering() {
        // ORDER BY must sit *below* LIMIT in the tree (SQL semantics: sort
        // first, then truncate). The lowered physical plan mirrors this.
        let phys = pp("SELECT a FROM t1 ORDER BY a DESC LIMIT 5");
        match phys {
            PhysicalPlan::Limit { input, .. } => {
                assert!(matches!(*input, PhysicalPlan::Sort { .. }));
            }
            other => panic!("expected Limit, got {other:?}"),
        }
    }

    #[test]
    fn having_wraps_aggregate_in_filter() {
        let plan = lp("SELECT a, COUNT(b) FROM t1 GROUP BY a HAVING COUNT(b) > 1");
        // After the wave-1 SELECT-order Project on the aggregate, HAVING
        // appears as the outermost Filter.
        match plan {
            LogicalPlan::Filter { input, .. } => {
                // Below the Filter is the Project that fixes SELECT column order.
                assert!(
                    matches!(*input, LogicalPlan::Project { .. }),
                    "expected Project under HAVING Filter, got {input:?}"
                );
            }
            other => panic!("expected Filter (HAVING) at top, got {other:?}"),
        }
    }

    #[test]
    fn having_rejected_without_group_by_or_aggregate() {
        let err = parse("SELECT a FROM t1 HAVING a > 1", &provider())
            .expect_err("HAVING without aggregate must error");
        let msg = format!("{err}");
        assert!(
            msg.to_lowercase().contains("having"),
            "error message should mention HAVING, got: {msg}"
        );
    }

    #[test]
    fn union_all_builds_union() {
        let plan = lp("SELECT a FROM t1 UNION ALL SELECT a FROM t2");
        match plan {
            LogicalPlan::Union { inputs } => {
                assert_eq!(inputs.len(), 2);
            }
            other => panic!("expected Union, got {other:?}"),
        }
    }

    #[test]
    fn union_dedup_wraps_union_in_distinct() {
        let plan = lp("SELECT a FROM t1 UNION SELECT a FROM t2");
        match plan {
            LogicalPlan::Distinct { input } => {
                assert!(
                    matches!(*input, LogicalPlan::Union { .. }),
                    "expected Distinct(Union), got Distinct({input:?})"
                );
            }
            other => panic!("expected Distinct, got {other:?}"),
        }
    }

    #[test]
    fn union_all_is_flattened() {
        // Three-way UNION ALL should land as a single 3-input Union, not a
        // nested 2-tree, so executors can stream branches without recursion.
        let plan = lp("SELECT a FROM t1 UNION ALL SELECT a FROM t1 UNION ALL SELECT a FROM t2");
        match plan {
            LogicalPlan::Union { inputs } => {
                assert_eq!(inputs.len(), 3, "expected flattened 3-input Union");
            }
            other => panic!("expected Union, got {other:?}"),
        }
    }

    #[test]
    fn inner_join_parses_to_join_node() {
        let plan = lp("SELECT * FROM t1 INNER JOIN t2 ON t1.a = t2.a");
        match plan {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::Join {
                    join_type,
                    on,
                    ..
                } => {
                    assert_eq!(join_type, JoinType::Inner);
                    assert_eq!(on.len(), 1);
                }
                other => panic!("expected Join under Project, got {other:?}"),
            },
            other => panic!("expected Project, got {other:?}"),
        }
    }

    #[test]
    fn inner_join_conjunctive_on_collects_multiple_pairs() {
        let plan = lp("SELECT * FROM t1 INNER JOIN t2 ON t1.a = t2.a AND t1.b = t2.c");
        let join = match plan {
            LogicalPlan::Project { input, .. } => *input,
            other => panic!("expected Project, got {other:?}"),
        };
        match join {
            LogicalPlan::Join { on, .. } => assert_eq!(on.len(), 2),
            other => panic!("expected Join, got {other:?}"),
        }
    }

    #[test]
    fn non_equi_join_rejected_with_clear_message() {
        let err = parse(
            "SELECT * FROM t1 INNER JOIN t2 ON t1.a > t2.a",
            &provider(),
        )
        .expect_err("non-equi JOIN must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("non-equi JOIN not yet supported"),
            "error message should mention non-equi JOIN, got: {msg}"
        );
    }

    #[test]
    fn left_join_parses_with_nullable_right_schema() {
        // `LEFT JOIN` preserves the left side; right-side columns become
        // nullable (a left row with no match emits NULL right columns).
        let plan = lp("SELECT * FROM t1 LEFT JOIN t2 ON t1.a = t2.a");
        // Walk past the outer Project (wildcard expansion) to the Join.
        let join_plan = match &plan {
            LogicalPlan::Project { input, .. } => input.as_ref(),
            other => other,
        };
        let (join_type, schema) = match join_plan {
            LogicalPlan::Join {
                join_type, ..
            } => (*join_type, join_plan.schema().unwrap()),
            other => panic!("expected Join, got {other:?}"),
        };
        assert_eq!(join_type, JoinType::LeftOuter);
        // t1 fields are first (a, b); t2 fields (a, c) follow with collision
        // disambiguation. Left fields keep their original nullability;
        // right fields are now nullable.
        assert_eq!(schema.fields.len(), 4);
        assert_eq!(schema.fields[0].name, "a");
        assert!(!schema.fields[0].nullable, "left 'a' keeps nullable=false");
        assert_eq!(schema.fields[1].name, "b");
        assert_eq!(schema.fields[2].name, "right.a");
        assert!(schema.fields[2].nullable, "LEFT JOIN: right 'a' is nullable");
        assert_eq!(schema.fields[3].name, "c");
        assert!(schema.fields[3].nullable, "LEFT JOIN: right 'c' is nullable");
    }

    #[test]
    fn right_join_parses_with_nullable_left_schema() {
        let plan = lp("SELECT * FROM t1 RIGHT JOIN t2 ON t1.a = t2.a");
        let join_plan = match &plan {
            LogicalPlan::Project { input, .. } => input.as_ref(),
            other => other,
        };
        let (join_type, schema) = match join_plan {
            LogicalPlan::Join { join_type, .. } => {
                (*join_type, join_plan.schema().unwrap())
            }
            other => panic!("expected Join, got {other:?}"),
        };
        assert_eq!(join_type, JoinType::RightOuter);
        // Left fields become nullable; right fields keep their original.
        assert!(schema.fields[0].nullable, "RIGHT JOIN: left 'a' is nullable");
        assert!(schema.fields[1].nullable, "RIGHT JOIN: left 'b' is nullable");
        assert!(
            !schema.fields[2].nullable,
            "RIGHT JOIN: right 'a' keeps original nullable=false"
        );
    }

    #[test]
    fn full_outer_join_parses_with_both_sides_nullable() {
        let plan = lp("SELECT * FROM t1 FULL OUTER JOIN t2 ON t1.a = t2.a");
        let join_plan = match &plan {
            LogicalPlan::Project { input, .. } => input.as_ref(),
            other => other,
        };
        let (join_type, schema) = match join_plan {
            LogicalPlan::Join { join_type, .. } => {
                (*join_type, join_plan.schema().unwrap())
            }
            other => panic!("expected Join, got {other:?}"),
        };
        assert_eq!(join_type, JoinType::FullOuter);
        for (i, f) in schema.fields.iter().enumerate() {
            assert!(
                f.nullable,
                "FULL OUTER JOIN: field {i} '{}' must be nullable",
                f.name
            );
        }
    }

    #[test]
    fn cross_join_parses_with_no_on_predicate() {
        let plan = lp("SELECT * FROM t1 CROSS JOIN t2");
        let join_plan = match &plan {
            LogicalPlan::Project { input, .. } => input.as_ref(),
            other => other,
        };
        match join_plan {
            LogicalPlan::Join { join_type, on, .. } => {
                assert_eq!(*join_type, JoinType::Cross);
                assert!(on.is_empty(), "CROSS JOIN has no ON predicate");
            }
            other => panic!("expected Join, got {other:?}"),
        }
    }

    #[test]
    fn join_lowers_to_physical_join() {
        // `SELECT * FROM t1 INNER JOIN t2 ON ...` parses to
        // `Project { input: Join }`; our lowerer detects that the source
        // chain isn't a Scan-only chain and falls through to lowering the
        // inner Join directly. The wave-7 executor surfaces the actual
        // "JOIN not yet implemented" error at run time; the planner just
        // needs to produce a PhysicalPlan::Join here.
        let phys = pp("SELECT * FROM t1 INNER JOIN t2 ON t1.a = t2.a");
        assert!(
            matches!(phys, PhysicalPlan::Join { .. }),
            "expected PhysicalPlan::Join, got {phys:?}"
        );
    }

    #[test]
    fn join_select_qualified_columns_resolve_with_rename() {
        // `t1` and `t2` both have a column named `a`; per
        // `join_combined_schema`, the right-side `a` is renamed to
        // `right.a` in the join's output. The resolver must mirror that
        // rule so `t2.a` lowers to `Column("right.a")` (matching the
        // wildcard-expansion convention) while `t1.a` stays as
        // `Column("a")`.
        let plan = lp("SELECT t1.a, t2.a FROM t1 INNER JOIN t2 ON t1.a = t2.a");
        match plan {
            LogicalPlan::Project { exprs, .. } => {
                assert_eq!(exprs.len(), 2, "expected two columns, got {exprs:?}");
                match (&exprs[0], &exprs[1]) {
                    (Expr::Column(left), Expr::Column(right)) => {
                        assert_eq!(left, "a", "t1.a should keep bare name");
                        assert_eq!(
                            right, "right.a",
                            "t2.a should resolve to the post-rename `right.a`"
                        );
                    }
                    other => panic!("expected two Column refs, got {other:?}"),
                }
            }
            other => panic!("expected Project, got {other:?}"),
        }
    }

    #[test]
    fn join_select_unknown_qualifier_errors() {
        // `t3` isn't in FROM at all. The resolver must produce a clear
        // "unknown table qualifier" message; pre-fix the entire
        // CompoundIdentifier path errored generically.
        let err = parse(
            "SELECT t3.a FROM t1 INNER JOIN t2 ON t1.a = t2.a",
            &provider(),
        )
        .expect_err("unknown qualifier must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown table qualifier"),
            "expected 'unknown table qualifier', got: {msg}"
        );
    }
}
