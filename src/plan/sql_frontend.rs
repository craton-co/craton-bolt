// SPDX-License-Identifier: Apache-2.0

//! SQL frontend: parses a SQL string into a `LogicalPlan` against a `TableProvider`.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use sqlparser::ast::{
    BinaryOperator, CastKind, DataType as SqlDataType, Distinct,
    Expr as SqlExpr, FunctionArg, FunctionArgExpr, FunctionArguments, GroupByExpr, Ident,
    JoinConstraint, JoinOperator, ObjectName, Offset, OrderByExpr, Query, Select, SelectItem,
    SetExpr, SetOperator, SetQuantifier, Statement, TableFactor, UnaryOperator, Value,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::{Parser, ParserError};

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{
    aggregate_output_name, group_key_output_name, join_rename, AggregateExpr, BinaryOp, DataType,
    Expr, JoinType, Literal, LogicalPlan, ScalarFnKind, Schema, SetOpKind, SortExpr, TimeUnit,
    UnaryOp, WindowExpr, WindowFunc,
};
use sqlparser::ast::{WindowFrameBound, WindowFrameUnits, WindowType};

/// Maximum recursion depth allowed when walking attacker-controlled SQL
/// AST / `LogicalPlan` trees. Pathological inputs (e.g. `SELECT
/// (((((... 1 ...))))) FROM t` with 10^5 nested parens) would otherwise
/// overflow the host thread stack and abort the process; this bound
/// surfaces a `BoltError::Sql` long before that happens.
///
/// 256 is comfortably deeper than any realistic query (the largest hand-
/// written nesting in our test corpus tops out around 12 levels) but
/// shallow enough that no platform's default thread stack — including
/// Windows' 1 MiB default — comes close to its overflow point even with
/// large per-frame locals on the lowering call path.
pub(crate) const MAX_RECURSION_DEPTH: usize = 256;

/// SQL-standard identifier folding (v0.5 / M2).
///
/// Unquoted SQL identifiers are case-insensitive in the standard. We
/// implement the Postgres convention: unquoted identifiers fold to ASCII
/// lowercase at parse time; quoted identifiers (`"MyCol"`) keep their
/// case verbatim. The choice of *which* canonical case is somewhat
/// arbitrary — the standard mandates uppercase, but Postgres uses lower
/// and is by far the more common reference for application developers.
///
/// We pair this folding with a case-insensitive *fallback* in
/// [`crate::plan::logical_plan::Schema::index_of`] (and likewise in
/// [`MemTableProvider::schema`] below). The fallback means callers who
/// programmatically constructed plans against a verbatim-cased schema
/// continue to work unchanged; the new lowercase-by-default identifiers
/// produced by SQL parsing match the verbatim schema via the fallback.
///
/// Locale note: SQL identifier folding is defined over ASCII, and a
/// locale-aware lowercase would make lowered plans non-portable between
/// machines that differ in their Unicode tables. Use
/// `to_ascii_lowercase` everywhere.
///
/// Returns the canonical name to embed in the lowered IR for `ident`.
fn ident_to_name(ident: &Ident) -> String {
    if ident.quote_style.is_some() {
        ident.value.clone()
    } else {
        ident.value.to_ascii_lowercase()
    }
}

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
#[derive(Default)]
struct NameResolver<'a> {
    /// One scope per table in FROM order (base first, then joined tables).
    tables: Vec<TableScope>,
    /// Lowering context for nested subqueries: the table provider (to resolve
    /// a subquery's own FROM tables) and the in-scope CTE definitions. Both
    /// are `None` for resolvers built in contexts where a subquery cannot
    /// appear (e.g. [`NameResolver::empty`] used by ORDER BY lowering). When
    /// present, `lower_expr` uses them to lower `(SELECT ...)` /
    /// `IN (SELECT ...)` expressions and to reject correlated subqueries.
    ///
    /// The fields are borrowed for the resolver's (short) lifetime — the
    /// resolver never outlives the `plan_select` call that builds it.
    ctx: Option<SubqueryCtx<'a>>,
}

/// Borrowed lowering context threaded into [`NameResolver`] so the scalar /
/// IN subquery arms of [`lower_expr`] can lower their nested `LogicalPlan`
/// without widening every `lower_expr` call-site signature. Carries the
/// provider (for the subquery's own table-schema lookups) and the CTE scope
/// (so a subquery may reference a CTE just like the outer query).
#[derive(Clone, Copy)]
struct SubqueryCtx<'a> {
    /// Table provider used to resolve the subquery's own FROM tables.
    provider: &'a dyn TableProvider,
    /// CTE definitions in scope at the subquery's lexical position.
    ctes: &'a CteScope,
}

impl std::fmt::Debug for NameResolver<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `SubqueryCtx` holds a `&dyn TableProvider` which is not `Debug`;
        // elide it so `NameResolver` keeps a `Debug` impl for the existing
        // `{resolver:?}`-style diagnostics.
        f.debug_struct("NameResolver")
            .field("tables", &self.tables)
            .field("has_subquery_ctx", &self.ctx.is_some())
            .finish()
    }
}

/// One table's contribution to a [`NameResolver`].
#[derive(Debug)]
struct TableScope {
    /// Qualifier that user-typed `qualifier.col` references must match. For
    /// `FROM mytable` this is the bare table name; for `FROM mytable AS t`
    /// this is the alias `t`. Either way it is the *only* name a
    /// CompoundIdentifier in the SELECT / WHERE / ON tree is allowed to
    /// spell — the underlying table name (used for the `Scan` and provider
    /// lookup) is *not* in scope once an alias has shadowed it. This
    /// matches standard SQL aliasing semantics and is what callers such as
    /// `lower_join_side` and `resolve_compound` compare against.
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

impl<'a> NameResolver<'a> {
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
    /// [`join_combined_schema`](crate::plan::logical_plan::join_combined_schema):
    /// a right-side column whose name already appears in the accumulated
    /// taken-set is renamed to `right.{col}`, with `__2`, `__3`, …
    /// suffixes appended as a last resort if even the qualified form
    /// clashes.
    ///
    /// The rule itself lives in
    /// [`join_rename`](crate::plan::logical_plan::join_rename) so this
    /// call site and `join_combined_schema` cannot drift apart; do not
    /// duplicate the mangling logic here.
    fn push_join(&mut self, name: String, schema: &Schema) {
        // Build the snapshot of names already taken across all previous
        // scopes' *output* names. This mirrors `join_combined_schema`'s
        // pass-by-pass accumulation: each new right side sees everything
        // produced so far on its left, not just the immediately preceding
        // table. `join_rename` then mutates this set so each subsequent
        // right-side column sees the names produced by its predecessors.
        let mut taken: std::collections::HashSet<String> = self
            .tables
            .iter()
            .flat_map(|t| t.cols.iter().map(|c| c.output.clone()))
            .collect();
        let mut cols = Vec::with_capacity(schema.fields.len());
        for f in &schema.fields {
            let out_name = join_rename(&f.name, &mut taken);
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
    /// or the column doesn't exist in the qualified table's schema. When the
    /// qualifier is unknown, the message lists every in-scope qualifier so
    /// users can spot typos / missing aliases at a glance.
    ///
    /// SQL-standard case folding: qualifier and column lookups try an exact
    /// match first; if that misses and the lookup name is all-ASCII-lowercase,
    /// falls back to a case-insensitive search. Mixed-case (quoted) idents
    /// take the strict path.
    fn resolve_compound(&self, qualifier: &str, col: &str) -> BoltResult<String> {
        let qualifier_lc = !qualifier.chars().any(|c| c.is_ascii_uppercase());
        let col_lc = !col.chars().any(|c| c.is_ascii_uppercase());
        let candidates: Vec<&str> =
            self.tables.iter().map(|t| t.name.as_str()).collect();
        let candidate_msg = if candidates.is_empty() {
            "no tables in scope".to_string()
        } else {
            format!("in-scope: {}", candidates.join(", "))
        };
        let scope = self
            .tables
            .iter()
            .find(|t| t.name == qualifier)
            .or_else(|| {
                if !qualifier_lc {
                    return None;
                }
                self.tables
                    .iter()
                    .find(|t| t.name.eq_ignore_ascii_case(qualifier))
            })
            .ok_or_else(|| {
                // Suggest a close in-scope table qualifier if any.
                let suffix = crate::plan::suggest::did_you_mean_suffix(
                    qualifier,
                    self.tables.iter().map(|t| t.name.as_str()),
                );
                BoltError::Sql(format!(
                    "unknown table qualifier '{qualifier}' in column reference \
                     '{qualifier}.{col}' ({candidate_msg}){suffix}"
                ))
            })?;
        let resolved = scope
            .cols
            .iter()
            .find(|c| c.original == col)
            .or_else(|| {
                if !col_lc {
                    return None;
                }
                scope
                    .cols
                    .iter()
                    .find(|c| c.original.eq_ignore_ascii_case(col))
            })
            .ok_or_else(|| {
                let suffix = crate::plan::suggest::did_you_mean_suffix(
                    col,
                    scope.cols.iter().map(|c| c.original.as_str()),
                );
                BoltError::Sql(format!(
                    "unknown column '{col}' in table '{qualifier}'{suffix}"
                ))
            })?;
        Ok(resolved.output.clone())
    }

    /// The set of (ASCII-lowercased) column names available in this resolver's
    /// FROM scope, across every in-scope table. Used by the subquery
    /// correlation detector to recognise an outer-column reference inside a
    /// nested subquery (see [`crate::plan::subquery::reject_if_correlated`]).
    fn outer_column_names(&self) -> std::collections::HashSet<String> {
        self.tables
            .iter()
            .flat_map(|t| {
                t.cols
                    .iter()
                    .map(|c| c.original.to_ascii_lowercase())
            })
            .collect()
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
    /// Resolve a table name to its schema.
    ///
    /// # SQL-standard case folding (v0.5)
    ///
    /// Tries an exact match first. If that misses *and* `name` is
    /// all-ASCII-lowercase, falls back to a case-insensitive scan.
    /// Mixed-case lookup keys (verbatim programmatic calls, quoted SQL
    /// identifiers) take the strict path. Same rule as
    /// [`crate::plan::logical_plan::Schema::index_of`] — see that method
    /// for the rationale.
    fn schema(&self, name: &str) -> BoltResult<Schema> {
        if let Some(s) = self.tables.get(name) {
            return Ok(s.clone());
        }
        if name.chars().any(|c| c.is_ascii_uppercase()) {
            return Err(BoltError::Plan(format!("unknown table '{name}'")));
        }
        if let Some((_, s)) = self
            .tables
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
        {
            return Ok(s.clone());
        }
        Err(BoltError::Plan(format!("unknown table '{name}'")))
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
#[tracing::instrument(name = "parse", level = "info", skip_all, fields(sql_len = sql.len()))]
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
    let mut stmts =
        Parser::parse_sql(&dialect, sql).map_err(|e| parse_error_to_bolt_error(e, sql))?;

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
    let plan = plan_query(&query, provider, &CteScope::new(), 0)?;
    // A root UNION / EXCEPT / INTERSECT lowers to a `Union` / `SetOp` node
    // (UNION as `Distinct(Union { .. })`) WITHOUT validating that the branches
    // share a compatible schema — that check lives in `LogicalPlan::schema()`
    // and was never triggered for a root set-op, so an incompatible-arity
    // EXCEPT/INTERSECT/UNION parsed as `Ok`. We eagerly compute the schema
    // ONLY for a set-op root so the existing descriptive `BoltError::Plan`
    // (which names the offending op) surfaces at parse time.
    //
    // We deliberately do NOT force a full schema computation for non-set-op
    // roots: type-checking of ordinary SELECTs (CAST/CASE/string-fn/aggregate
    // type errors, ...) is contracted to happen lazily at `schema()` /
    // lowering, and callers/tests rely on `parse` succeeding for a plan that
    // only fails its later type-check. Scoping the eager check to set-ops adds
    // the missing arity/type rejection without changing that contract.
    if is_set_op_root(&plan) {
        let _ = plan.schema()?;
    }
    Ok(plan)
}

/// True if the root of `plan` is a set operation (`UNION` / `EXCEPT` /
/// `INTERSECT`), including a `UNION` lowered as `Distinct(Union { .. })`.
/// Used to scope the eager parse-time schema validation to set-ops, whose
/// branch-compatibility check would otherwise never run at parse time.
fn is_set_op_root(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Union { .. } | LogicalPlan::SetOp { .. } => true,
        LogicalPlan::Distinct { input } => is_set_op_root(input),
        _ => false,
    }
}

/// v0.6 / M5: convert a `sqlparser` [`ParserError`] into a [`BoltError`],
/// preserving the location information that sqlparser appends to its
/// `Display` output (`"... at Line: <L>, Column: <C>"`) as a byte-offset
/// span on [`BoltError::SqlWithSpan`].
///
/// sqlparser 0.52 does NOT expose `Location` structurally on the error
/// type itself — the location only appears as text in the formatted
/// message — so this helper does a small amount of string surgery to
/// recover the line/column pair, then maps it to a byte offset against
/// `sql`. When no location suffix is present (e.g.
/// [`ParserError::RecursionLimitExceeded`]) we fall back to the legacy
/// unspanned [`BoltError::Sql`] shape so the public API surface stays
/// uniform.
///
/// The half-open span is `[byte_offset .. byte_offset]` (zero-width)
/// because sqlparser only tells us *where* the error is, not how wide
/// the offending token is. A zero-width span is documented as legal on
/// [`BoltError::SqlWithSpan`] and editor consumers typically render it
/// as a single-character squiggle at the position.
pub(crate) fn parse_error_to_bolt_error(e: ParserError, sql: &str) -> BoltError {
    let rendered = e.to_string();
    match extract_location_suffix(&rendered) {
        Some((msg_without_loc, line, column)) => {
            match line_column_to_byte_offset(sql, line, column) {
                Some(offset) => BoltError::SqlWithSpan {
                    msg: msg_without_loc,
                    span: offset..offset,
                },
                // The location was syntactically valid but pointed past the
                // input (e.g. an off-by-one in sqlparser's column counter
                // for a multi-byte token). Fall back to unspanned rather
                // than guessing.
                None => BoltError::Sql(rendered),
            }
        }
        None => BoltError::Sql(rendered),
    }
}

/// Pull the `" at Line: <L>, Column: <C>"` suffix off a sqlparser-formatted
/// error message and return the trimmed message plus the parsed `(line,
/// column)` pair. Returns `None` if no recognisable suffix is present.
///
/// The exact format comes from `sqlparser::tokenizer::Location`'s `Display`
/// impl (a leading space, then `"at Line: N, Column: M"`). Centralised so
/// the parsing rule has a single point of maintenance if a future
/// sqlparser version drops the suffix or changes its shape.
fn extract_location_suffix(rendered: &str) -> Option<(String, u64, u64)> {
    // We search for the *last* " at Line: " marker because sqlparser
    // sometimes embeds its own internal location text inside the message
    // body (e.g. "Expected: a value, found: ... at Line: 1, Column: 7"),
    // and the trailing one is always the authoritative position.
    let marker = " at Line: ";
    let idx = rendered.rfind(marker)?;
    let (head, tail) = rendered.split_at(idx);
    // `tail` starts with " at Line: <L>, Column: <C>".
    let rest = tail.strip_prefix(marker)?;
    let comma_idx = rest.find(", Column: ")?;
    let (line_str, after_line) = rest.split_at(comma_idx);
    let col_str = after_line.strip_prefix(", Column: ")?;
    let line: u64 = line_str.trim().parse().ok()?;
    let column: u64 = col_str.trim().parse().ok()?;
    Some((head.to_string(), line, column))
}

/// Convert a 1-based `(line, column)` pair (sqlparser convention) into a
/// 0-based byte offset into `sql`. Returns `None` if the pair points
/// outside the input, which we treat as "no usable span" rather than as a
/// hard error — the caller falls back to the unspanned `Sql` shape.
///
/// `column` is interpreted as a character-column for ASCII input (the
/// common case in our test corpus and dashboard workloads). For
/// non-ASCII SQL the column count maps to chars, then we sum each char's
/// UTF-8 byte length to get the byte offset — same convention sqlparser
/// uses internally.
fn line_column_to_byte_offset(sql: &str, line: u64, column: u64) -> Option<usize> {
    if line == 0 || column == 0 {
        // sqlparser's `Location` Display elides "Line: 0" output entirely,
        // so a parsed `0` here means we mis-extracted; bail.
        return None;
    }
    let mut current_line: u64 = 1;
    let mut byte_offset: usize = 0;
    let bytes = sql.as_bytes();
    // Walk to the start of `line`. sqlparser counts a `\n` as the line
    // terminator; carriage returns are not consumed specially.
    while current_line < line {
        match bytes[byte_offset..].iter().position(|&b| b == b'\n') {
            Some(nl_rel) => {
                byte_offset += nl_rel + 1;
                current_line += 1;
            }
            // The line number is beyond the input.
            None => return None,
        }
        if byte_offset > bytes.len() {
            return None;
        }
    }
    // Now advance `column - 1` characters within this line. Use
    // `char_indices()` over the remainder so multi-byte chars contribute
    // their full UTF-8 width.
    let line_rest = sql.get(byte_offset..)?;
    let mut chars_consumed: u64 = 0;
    let target = column - 1;
    for (ch_byte_off, ch) in line_rest.char_indices() {
        if chars_consumed == target {
            return Some(byte_offset + ch_byte_off);
        }
        if ch == '\n' {
            // Column points past the end of this line.
            return None;
        }
        chars_consumed += 1;
    }
    // Column points exactly at end-of-line / end-of-input — still a valid
    // (zero-width) span.
    if chars_consumed == target {
        Some(byte_offset + line_rest.len())
    } else {
        None
    }
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

/// A `WITH` common-table-expression scope: maps a (case-folded) CTE name to
/// its already-lowered [`LogicalPlan`].
///
/// CTEs are **inlined**: when a later CTE or the main query references a CTE
/// by name in its FROM clause, the registered plan is substituted (cloned) at
/// the reference site. There is no shared/materialised CTE node in the IR.
///
/// Scoping rules implemented here mirror standard non-recursive SQL:
///   * CTEs are visible to *later* CTEs in the same `WITH` list and to the
///     main query — i.e. a CTE may reference any CTE declared before it.
///   * A CTE may **not** reference itself (no `WITH RECURSIVE`).
///   * Nested queries (subqueries, derived tables) inherit the enclosing CTE
///     scope, so they too may reference any in-scope CTE.
///
/// The scope is built incrementally as the `WITH` list is lowered (see
/// [`register_ctes`]); each CTE is lowered against the scope accumulated so
/// far, then added to it.
#[derive(Debug, Clone, Default)]
pub(crate) struct CteScope {
    /// Case-folded CTE name → its inlined logical plan.
    defs: HashMap<String, LogicalPlan>,
}

impl CteScope {
    /// Empty scope (no CTEs in scope).
    fn new() -> Self {
        Self::default()
    }

    /// Look up a CTE by (case-folded) name. The lookup tries an exact match
    /// first, then an ASCII case-insensitive fallback — the same folding
    /// convention the rest of the frontend uses for unquoted identifiers.
    fn get(&self, name: &str) -> Option<&LogicalPlan> {
        if let Some(p) = self.defs.get(name) {
            return Some(p);
        }
        if !name.chars().any(|c| c.is_ascii_uppercase()) {
            return self
                .defs
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v);
        }
        None
    }
}

/// Lower a `WITH` clause into a [`CteScope`], rejecting `WITH RECURSIVE`.
///
/// Each CTE in the list is lowered against the scope accumulated from the
/// CTEs that precede it (standard left-to-right visibility), then registered.
/// Column-list aliases on a CTE (`WITH c (a, b) AS (...)`) are rejected since
/// they would require renaming the lowered plan's output schema, which the
/// frontend does not implement. A duplicate CTE name is rejected.
///
/// `base` is the CTE scope inherited from any enclosing query (so a subquery's
/// own `WITH` can shadow / extend the outer scope).
fn register_ctes(
    with: &sqlparser::ast::With,
    provider: &dyn TableProvider,
    base: &CteScope,
    depth: usize,
) -> BoltResult<CteScope> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    if with.recursive {
        return Err(BoltError::Sql(
            "unsupported: WITH RECURSIVE (only non-recursive CTEs are supported)".into(),
        ));
    }
    let mut scope = base.clone();
    for cte in &with.cte_tables {
        // `cte.alias` is the CTE's name plus an optional column list.
        if !cte.alias.columns.is_empty() {
            return Err(BoltError::Sql(format!(
                "unsupported: CTE column-list alias on '{}' (WITH {} (..) AS ...)",
                cte.alias.name.value, cte.alias.name.value
            )));
        }
        if cte.from.is_some() {
            return Err(BoltError::Sql(
                "unsupported: CTE materialization hint (FROM)".into(),
            ));
        }
        let name = ident_to_name(&cte.alias.name);
        if scope.defs.contains_key(&name) {
            return Err(BoltError::Sql(format!(
                "duplicate CTE name '{name}' in WITH clause"
            )));
        }
        // Lower the CTE body against the scope built so far (earlier CTEs are
        // visible; the CTE itself is NOT yet in scope, so a self-reference
        // surfaces as an unknown-table error — there is no recursion).
        let plan = plan_query(&cte.query, provider, &scope, depth + 1)?;
        // Type-check eagerly so a malformed CTE is reported at its definition
        // site rather than at the (possibly distant) reference site.
        let _ = plan.schema()?;
        scope.defs.insert(name, plan);
    }
    Ok(scope)
}

/// Lower a top-level `Query`. Supports SELECT, UNION [ALL], ORDER BY, LIMIT,
/// OFFSET, and non-recursive `WITH` / CTEs. Rejects `WITH RECURSIVE`, FETCH,
/// locks, EXCEPT/INTERSECT, and dialect extensions outside our subset.
///
/// `ctes` is the CTE scope inherited from any enclosing query; a `WITH` clause
/// on `query` extends it for the duration of this query's lowering.
///
/// `depth` is the current recursion depth; returns Err if MAX_RECURSION_DEPTH is exceeded.
fn plan_query(
    query: &Query,
    provider: &dyn TableProvider,
    ctes: &CteScope,
    depth: usize,
) -> BoltResult<LogicalPlan> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    // WITH / CTEs: lower the CTE list into a scope (extending any inherited
    // scope), then lower the body against it. `register_ctes` rejects
    // WITH RECURSIVE cleanly.
    let local_ctes;
    let active_ctes = match &query.with {
        Some(with) => {
            local_ctes = register_ctes(with, provider, ctes, depth + 1)?;
            &local_ctes
        }
        None => ctes,
    };
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
    let mut plan = lower_set_expr(query.body.as_ref(), provider, active_ctes, depth + 1)?;

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
/// `Distinct(Union { inputs })`; EXCEPT / INTERSECT (with optional ALL) become
/// a binary `LogicalPlan::SetOp` node (executed host-side by
/// [`crate::exec::setops`]).
///
/// `depth` is the current recursion depth; returns Err if MAX_RECURSION_DEPTH is exceeded.
fn lower_set_expr(
    expr: &SetExpr,
    provider: &dyn TableProvider,
    ctes: &CteScope,
    depth: usize,
) -> BoltResult<LogicalPlan> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    match expr {
        SetExpr::Select(s) => plan_select(s.as_ref(), provider, ctes),
        SetExpr::Query(q) => plan_query(q.as_ref(), provider, ctes, depth + 1),
        SetExpr::SetOperation {
            op,
            set_quantifier,
            left,
            right,
        } => {
            // `BY NAME` variants are non-standard (schema-rewriting) and
            // rejected for every operator before we branch on the operator.
            let all = match set_quantifier {
                SetQuantifier::All => true,
                SetQuantifier::Distinct | SetQuantifier::None => false,
                SetQuantifier::ByName
                | SetQuantifier::AllByName
                | SetQuantifier::DistinctByName => {
                    return Err(BoltError::Sql(format!(
                        "unsupported: {op} BY NAME"
                    )));
                }
            };
            match op {
                SetOperator::Union => {
                    // `dedup` is the inverse of `all`: plain UNION dedups.
                    let dedup = !all;
                    // Flatten left-recursive UNION chains into a single Union
                    // node so `q1 UNION ALL q2 UNION ALL q3` becomes one
                    // 3-input Union rather than a nested binary tree. UNION
                    // (dedup) does NOT flatten across UNION ALL boundaries:
                    // their semantics differ.
                    let mut inputs: Vec<LogicalPlan> = Vec::new();
                    collect_union_branches(left, provider, ctes, dedup, &mut inputs, depth + 1)?;
                    collect_union_branches(right, provider, ctes, dedup, &mut inputs, depth + 1)?;
                    let union = LogicalPlan::Union { inputs };
                    Ok(if dedup {
                        LogicalPlan::Distinct {
                            input: Box::new(union),
                        }
                    } else {
                        union
                    })
                }
                // EXCEPT / INTERSECT (with optional ALL) lower to a binary
                // `SetOp` node executed host-side by `crate::exec::setops`.
                // We do NOT flatten chains here — the multiset semantics of
                // `a EXCEPT b EXCEPT c` are left-associative `(a EXCEPT b)
                // EXCEPT c`, which the nested-binary shape expresses directly.
                SetOperator::Except | SetOperator::Intersect => {
                    let set_op = match op {
                        SetOperator::Except => SetOpKind::Except,
                        SetOperator::Intersect => SetOpKind::Intersect,
                        SetOperator::Union => unreachable!("handled above"),
                    };
                    let left_plan = lower_set_expr(left, provider, ctes, depth + 1)?;
                    let right_plan = lower_set_expr(right, provider, ctes, depth + 1)?;
                    Ok(LogicalPlan::SetOp {
                        left: Box::new(left_plan),
                        right: Box::new(right_plan),
                        op: set_op,
                        all,
                    })
                }
            }
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
///
/// `depth` is the current recursion depth; returns Err if MAX_RECURSION_DEPTH is exceeded.
fn collect_union_branches(
    expr: &SetExpr,
    provider: &dyn TableProvider,
    ctes: &CteScope,
    parent_dedup: bool,
    out: &mut Vec<LogicalPlan>,
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
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
                out.push(lower_set_expr(expr, provider, ctes, depth + 1)?);
                return Ok(());
            }
        };
        if child_dedup == parent_dedup {
            collect_union_branches(left, provider, ctes, parent_dedup, out, depth + 1)?;
            collect_union_branches(right, provider, ctes, parent_dedup, out, depth + 1)?;
            return Ok(());
        }
    }
    out.push(lower_set_expr(expr, provider, ctes, depth + 1)?);
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
            expr: lower_expr(expr, &resolver, 0)?,
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
fn plan_select(
    select: &Select,
    provider: &dyn TableProvider,
    ctes: &CteScope,
) -> BoltResult<LogicalPlan> {
    reject_unsupported_select(select)?;

    // FROM: exactly one base table reference. JOINs hang off `twj.joins`.
    if select.from.len() != 1 {
        return Err(BoltError::Sql(format!(
            "expected exactly one FROM table, got {}",
            select.from.len()
        )));
    }
    let twj = &select.from[0];

    // Build the base plan from the first table reference. A plain table
    // reference lowers to a `Scan`; a reference that names an in-scope CTE
    // inlines (clones) the CTE's already-lowered plan. `base_qualifier` is the
    // alias (if any) the user-typed `qualifier.col` references must match.
    let (base_plan, base_qualifier, scan_schema) =
        lower_table_factor(&twj.relation, provider, ctes)?;
    // The name resolver tracks the FROM-tree's `table.col` namespace so we
    // can resolve qualified references in WHERE / SELECT / GROUP BY / HAVING.
    // The ON-clause lowerer also consults the resolver (see `lower_join_on`
    // / `lower_join_side`) to reject same-side and unknown-table qualifiers
    // before the executor ever runs.
    //
    // The resolver also carries the subquery lowering context (provider + CTE
    // scope) so a `(SELECT ...)` / `IN (SELECT ...)` in WHERE / SELECT can
    // lower its own nested plan; see [`SubqueryCtx`].
    let mut resolver = NameResolver::empty();
    resolver.ctx = Some(SubqueryCtx { provider, ctes });
    resolver.push_base(base_qualifier, &scan_schema);
    let mut plan = base_plan;

    // JOIN handling. Supports INNER / LEFT / RIGHT / FULL with an
    // ON predicate, plus CROSS (no ON clause). The join's right side must
    // itself be a bare table. Equi conjuncts (`left.col = right.col`)
    // route to the hash-join fast path; non-equi conjuncts (`<`, `>`,
    // `BETWEEN`, …) populate the residual `filter` slot and switch the
    // executor to the nested-loop fallback (v0.6). See
    // `crate::exec::join` for both paths.
    for join in &twj.joins {
        if join.global {
            return Err(BoltError::Sql(
                "unsupported: GLOBAL JOIN (ClickHouse extension)".into(),
            ));
        }
        // Pick out the (join_type, join constraint) pair. CROSS JOIN has no
        // constraint — sqlparser models it with its own variant. We keep the
        // raw `&JoinConstraint` (rather than eagerly extracting an ON expr) so
        // the `USING (...)` / `NATURAL` desugaring below can run *after* the
        // RHS schema is in scope; an ON predicate still routes through
        // `lower_join_on` exactly as before.
        let (join_type, constraint): (JoinType, Option<&JoinConstraint>) =
            match &join.join_operator {
                JoinOperator::Inner(c) => (JoinType::Inner, Some(c)),
                JoinOperator::LeftOuter(c) => (JoinType::LeftOuter, Some(c)),
                JoinOperator::RightOuter(c) => (JoinType::RightOuter, Some(c)),
                JoinOperator::FullOuter(c) => (JoinType::FullOuter, Some(c)),
                JoinOperator::CrossJoin => (JoinType::Cross, None),
                other => {
                    return Err(BoltError::Sql(format!(
                        "unsupported join kind: {other:?}; \
                         supported: INNER, LEFT, RIGHT, FULL OUTER, CROSS"
                    )));
                }
            };
        let (rhs_plan, rhs_qualifier, rhs_schema) =
            lower_table_factor(&join.relation, provider, ctes)?;
        // Extend the resolver before we move `rhs_*` into the right-side
        // plan, so it sees the same rename rule as `join_combined_schema`
        // applies to the actual plan output. The resolver records the
        // *qualifier* (alias if any) so user-typed `t.col` references in
        // WHERE / SELECT / ON resolve against the alias-name the user
        // wrote, not the underlying table name. The USING / NATURAL
        // desugaring below also consults the just-pushed RHS scope.
        resolver.push_join(rhs_qualifier.clone(), &rhs_schema);
        // Keep the qualifier available after the move into `right_plan` so
        // the ON-clause validator can distinguish "right side" (the
        // just-pushed qualifier) from "left side" (any earlier scope).
        let rhs_qualifier_for_on = rhs_qualifier;
        let right_plan = rhs_plan;
        // Resolve the join constraint into `(equi_pairs, residual_filter)`:
        //   * `ON <expr>`        — `lower_join_on` splits equi pairs from a
        //     non-equi residual (single rejection path for unsupported forms);
        //   * `USING (c1, ...)`  — each named column is desugared to an
        //     equi-pair `left.c = right.c` (see `desugar_using_columns`);
        //   * `NATURAL`          — every column common to both sides becomes
        //     such an equi-pair (see `desugar_natural_columns`);
        //   * CROSS / no clause  — no predicate.
        // USING / NATURAL never produce a residual filter — they are pure
        // equalities — so the nested-loop path is reserved for ON residuals.
        let (on_pairs, filter) = match constraint {
            Some(JoinConstraint::On(e)) => {
                let lowered = lower_join_on(e, &resolver, &rhs_qualifier_for_on)?;
                (lowered.equi_pairs, lowered.filter)
            }
            Some(JoinConstraint::Using(cols)) => {
                let pairs = desugar_using_columns(cols, &resolver)?;
                (pairs, None)
            }
            Some(JoinConstraint::Natural) => {
                let pairs = desugar_natural_columns(&resolver)?;
                (pairs, None)
            }
            Some(JoinConstraint::None) => {
                return Err(BoltError::Sql(
                    "JOIN requires an ON, USING, or NATURAL clause".into(),
                ));
            }
            None => (Vec::new(), None),
        };
        plan = LogicalPlan::Join {
            left: Box::new(plan),
            right: Box::new(right_plan),
            join_type,
            on: on_pairs,
            filter,
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
        let predicate = lower_expr(filter_sql, &resolver, 0)?;
        // Trigger type-checking on the predicate. `Expr::dtype` recurses
        // through the operand tree, so per-arm rules like the Utf8 check on
        // `Expr::Like` fire here. Without this call the lower-level type
        // rules are dormant during pure SQL parsing because nothing else
        // walks the predicate's dtype tree before execution.
        let input_schema = plan.schema()?;
        let predicate_dtype = predicate.dtype(&input_schema)?;
        if predicate_dtype != DataType::Bool {
            return Err(BoltError::Type(format!(
                "WHERE predicate must be Bool, got {predicate_dtype:?}"
            )));
        }
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
                // Column alias: same fold rule as any other unquoted identifier
                // (lowercase) — quoted aliases preserve case.
                items.push((expr.clone(), Some(ident_to_name(alias))))
            }
            SelectItem::Wildcard(_) => {
                // Expand `SELECT *` to the FROM-tree's registered field
                // names verbatim. We synthesize *quoted* `Ident`s so the
                // SQL-standard case folding ([`ident_to_name`]) preserves
                // the registered casing — anything else would silently
                // rename columns like `MyCol` to `mycol` in the lowered
                // plan, which is wrong: the user's host program registered
                // them with that casing and downstream code (executor,
                // physical planner) refers to them by that exact name.
                for f in &scan_schema_for_wildcard.fields {
                    items.push((
                        SqlExpr::Identifier(Ident::with_quote('"', f.name.clone())),
                        None,
                    ));
                }
            }
            SelectItem::QualifiedWildcard(_, _) => {
                return Err(BoltError::Sql("unsupported: qualified wildcard".into()));
            }
        }
    }

    // `COUNT(DISTINCT col)` — the one DISTINCT-quantified aggregate form we
    // support. It is only handled as the *sole* SELECT item with *no* GROUP BY;
    // anything richer (multiple SELECT items, a GROUP BY, or DISTINCT on a
    // non-COUNT aggregate) is rejected with a precise message so the user is
    // not left guessing. We lower it to
    //   Aggregate(COUNT(*)) ∘ Distinct ∘ Project([col]) ∘ Filter(col IS NOT NULL)
    // which gives the SQL-standard NULL-excluding distinct count: the
    // pre-Distinct Filter drops NULL rows, Distinct dedupes the surviving
    // values (reusing the row-key / NULL canonicalisation in
    // `crate::exec::distinct`), and COUNT(*) over a single-column projection
    // tallies them.
    {
        // Detect COUNT(DISTINCT ...) anywhere in the SELECT list so we can
        // either special-case the sole-item form or reject the unsupported
        // combinations clearly.
        let mut count_distinct_positions: Vec<usize> = Vec::new();
        for (i, (sql_expr, _)) in items.iter().enumerate() {
            if try_count_distinct(sql_expr, &resolver)?.is_some() {
                count_distinct_positions.push(i);
            }
        }
        if !count_distinct_positions.is_empty() {
            if items.len() != 1 {
                return Err(BoltError::Sql(
                    "COUNT(DISTINCT col) is only supported as the sole SELECT item \
                     (no other columns or aggregates alongside it)"
                        .into(),
                ));
            }
            if !group_by_sql.is_empty() {
                return Err(BoltError::Sql(
                    "COUNT(DISTINCT col) with GROUP BY is not supported".into(),
                ));
            }
            if select.having.is_some() {
                return Err(BoltError::Sql(
                    "COUNT(DISTINCT col) with HAVING is not supported".into(),
                ));
            }
            if matches!(select.distinct, Some(Distinct::Distinct)) {
                return Err(BoltError::Sql(
                    "SELECT DISTINCT COUNT(DISTINCT col) is not supported".into(),
                ));
            }
            let (sql_expr, alias) = &items[0];
            // Safe to unwrap: the position scan above already confirmed this.
            let inner = try_count_distinct(sql_expr, &resolver)?
                .expect("count_distinct_positions implies this item matches");
            let col = lower_expr(inner, &resolver, 0)?;
            // Type-check the argument against the current plan's schema so a
            // misnamed column surfaces here, not at execution time.
            let _ = col.dtype(&plan.schema()?)?;

            // Filter(col IS NOT NULL): exclude NULLs per SQL DISTINCT semantics.
            let not_null = Expr::Unary {
                op: UnaryOp::IsNotNull,
                operand: Box::new(col.clone()),
            };
            plan = LogicalPlan::Filter {
                input: Box::new(plan),
                predicate: not_null,
            };
            // Project([col]) — narrow to the single distinct-counted column.
            plan = LogicalPlan::Project {
                input: Box::new(plan),
                exprs: vec![col],
            };
            // Distinct — dedupe the surviving non-NULL values.
            plan = LogicalPlan::Distinct {
                input: Box::new(plan),
            };
            // Aggregate(COUNT(*)) — tally the distinct values. COUNT(*) uses
            // the literal-1 sentinel mirroring `try_aggregate`'s COUNT(*) path.
            let aggregates = vec![AggregateExpr::Count(Expr::Literal(Literal::Int64(1)))];
            let aggregate_plan = LogicalPlan::Aggregate {
                input: Box::new(plan),
                group_by: Vec::new(),
                aggregates,
            };
            // Re-project to honour any SELECT alias on the result column,
            // matching the post-aggregate Project the ordinary aggregate path
            // builds (so downstream stages see the user-friendly name).
            let out_name = aggregate_output_name(&AggregateExpr::Count(Expr::Literal(
                Literal::Int64(1),
            )));
            let col_ref = Expr::Column(out_name);
            let proj = match alias {
                Some(a) => col_ref.alias(a.clone()),
                None => col_ref,
            };
            plan = LogicalPlan::Project {
                input: Box::new(aggregate_plan),
                exprs: vec![proj],
            };
            return Ok(plan);
        }
    }

    // Use `contains_aggregate` (not `try_aggregate`) so that SELECT items
    // with aggregates *nested inside a scalar expression* — e.g.
    // `SUM(price) + 1` with no GROUP BY and no bare top-level aggregate —
    // still route into aggregate mode. `try_aggregate` only matches a bare
    // top-level aggregate call, which would leave such items to fall through
    // to plain projection where `lower_expr` rejects the aggregate as an
    // unsupported scalar function call. This aligns the gate with the
    // post-aggregate scalar-expression handling documented below.
    let has_agg_in_select = items
        .iter()
        .map(|(e, _)| contains_aggregate(e, &resolver, 0))
        .collect::<BoltResult<Vec<_>>>()?
        .iter()
        .any(|&b| b);

    if has_agg_in_select || !group_by_sql.is_empty() {
        // Aggregate mode. Each SELECT item is one of:
        //   * a bare aggregate call (`SUM(v)`),
        //   * a bare group key already listed in GROUP BY (`k`),
        //   * a scalar expression containing one or more aggregates
        //     (`SUM(price) + 1`, `AVG(qty) * 2`, `(SUM(a) + SUM(b)) / 2`) —
        //     the nested aggregates are extracted as feed inputs and the
        //     surface expression is rewritten with `Column("<agg_out>")` at
        //     each aggregate position. The rewritten expression goes into the
        //     post-Aggregate Project as a computed projection.
        let group_by: Vec<Expr> = group_by_sql
            .iter()
            .map(|e| lower_expr(e, &resolver, 0))
            .collect::<BoltResult<_>>()?;

        let mut aggregates: Vec<AggregateExpr> = Vec::new();
        // For each SELECT item, remember how to pull it back out of the Aggregate
        // node's schema (group keys first, aggregates second per `Aggregate::schema()`).
        // Each entry is the *output* column name produced by the Aggregate, plus an
        // optional SELECT alias to rename it to in the final projection.
        enum SelectSource {
            /// SELECT references a group key; pull by the key's name in the Aggregate schema.
            GroupKey { key_name: String, alias: Option<String> },
            /// SELECT references the Nth aggregate in `aggregates`. An optional
            /// alias renames the aggregate's output column in the SELECT-order
            /// Project (so HAVING and downstream stages can reference it by
            /// the user-friendly name).
            Aggregate { index: usize, alias: Option<String> },
            /// SELECT is a scalar expression with one or more aggregates
            /// inside. The expression has already been lowered with each
            /// aggregate position replaced by `Expr::Column(agg_out_name)`;
            /// each extracted aggregate has been pushed onto `aggregates`
            /// (deduplicated by output name). The Project step evaluates
            /// `expr` against the Aggregate's output schema.
            Computed { expr: Expr, alias: Option<String> },
        }
        let mut select_sources: Vec<SelectSource> = Vec::new();

        for (sql_expr, alias) in &items {
            if let Some(agg) = try_aggregate(sql_expr, &resolver, 0)? {
                // Bare aggregate path is unchanged: append the aggregate
                // without deduplication so the Aggregate plan node mirrors
                // the SELECT list 1:1 (pre-existing behaviour, exercised
                // by tier-2 tests).
                let idx = aggregates.len();
                aggregates.push(agg);
                select_sources.push(SelectSource::Aggregate {
                    index: idx,
                    alias: alias.clone(),
                });
                continue;
            }
            // Non-bare-aggregate item. Two sub-cases:
            //   (a) the item contains aggregates nested inside scalar
            //       operators (post-aggregate scalar expression). Extract
            //       each aggregate as a feed input and rewrite the surface
            //       expression with `Column("<agg_out>")` at each aggregate
            //       position. The rewritten expression becomes a `Computed`
            //       projection that the post-Aggregate Project evaluates.
            //   (b) the item has no aggregates at all. It must then resolve
            //       to a declared GROUP BY key (existing path).
            if contains_aggregate(sql_expr, &resolver, 0)? {
                let rewritten =
                    extract_and_rewrite_aggregates(sql_expr, &resolver, &mut aggregates, 0)?;
                select_sources.push(SelectSource::Computed {
                    expr: rewritten,
                    alias: alias.clone(),
                });
                continue;
            }
            let lowered = lower_expr(sql_expr, &resolver, 0)?;
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
        // Aggregate output names follow `aggregate_output_name` (a thin wrapper
        // over the `pub(crate)` `AggregateExpr::output_name` in
        // `logical_plan.rs`, e.g. SUM(x) -> "sum_x", COUNT(*) -> "count").
        // Group-key names follow `group_key_output_name`. Both live in
        // `logical_plan.rs` as the single source of truth; do not duplicate
        // the rule here.
        let aggregates_out: &[AggregateExpr] = match &aggregate_plan {
            LogicalPlan::Aggregate { aggregates, .. } => aggregates,
            _ => unreachable!("just constructed an Aggregate"),
        };
        let mut proj_exprs: Vec<Expr> = Vec::with_capacity(select_sources.len());
        // Build the (aggregate-output-name -> SELECT alias) map at the same
        // time so the HAVING lowerer below can rewrite e.g. `SUM(v)` into
        // the alias the Project exposed (otherwise it would lower to
        // `Column("sum_v")`, which no longer exists in the Project's
        // output once an alias renamed it).
        let mut agg_alias_for_having: HashMap<String, String> = HashMap::new();
        for src in &select_sources {
            match src {
                SelectSource::GroupKey { key_name, alias } => {
                    let col = Expr::Column(key_name.clone());
                    proj_exprs.push(match alias {
                        Some(a) => col.alias(a.clone()),
                        None => col,
                    });
                }
                SelectSource::Aggregate { index, alias } => {
                    let name = aggregate_output_name(&aggregates_out[*index]);
                    if let Some(a) = alias {
                        agg_alias_for_having.insert(name.clone(), a.clone());
                    }
                    let col = Expr::Column(name);
                    proj_exprs.push(match alias {
                        Some(a) => col.alias(a.clone()),
                        None => col,
                    });
                }
                SelectSource::Computed { expr, alias } => {
                    // The expression already references aggregate output
                    // columns (`Column("sum_price")`) and/or any other
                    // columns visible in the Aggregate's output schema. Wrap
                    // in `Alias` if the SELECT item carried `AS <name>`.
                    let e = expr.clone();
                    proj_exprs.push(match alias {
                        Some(a) => e.alias(a.clone()),
                        None => e,
                    });
                }
            }
        }
        // Stash for the HAVING block below. The empty-map case is fine —
        // `lower_expr_in_having` falls through to the unaliased name.
        let having_agg_aliases = agg_alias_for_having;

        plan = LogicalPlan::Project {
            input: Box::new(aggregate_plan),
            exprs: proj_exprs,
        };

        // HAVING (aggregate mode): the predicate may reference aggregates
        // by call (`SUM(v)`), by alias (`total`), or by group-key name. We
        // use the alias map built alongside the Project so an aggregate
        // call resolves to whichever name the Project actually exposed.
        if let Some(having_sql) = &select.having {
            let predicate = lower_expr_in_having(having_sql, &resolver, &having_agg_aliases, 0)?;
            let project_schema = plan.schema()?;
            validate_having_columns(&predicate, &project_schema)?;
            plan = LogicalPlan::Filter {
                input: Box::new(plan),
                predicate,
            };
        }
    } else {
        // Scalar projection mode.
        if select.having.is_some() {
            return Err(BoltError::Sql(
                "HAVING requires GROUP BY or aggregate functions in SELECT".into(),
            ));
        }

        // Window-function pass. A SELECT item may be a top-level window call
        // (optionally aliased): `ROW_NUMBER() OVER (...)`,
        // `SUM(x) OVER (...) AS s`. We collect every such item, build one
        // `LogicalPlan::Window` node per distinct window spec (stacked over
        // `plan`), and rewrite the projection to reference the generated
        // window output column. Window functions nested inside larger
        // expressions are rejected cleanly for now.
        //
        // `window_groups` keys a spec (partition_by + order_by) to the list
        // of (output_name, WindowFunc) it contributes; insertion order is
        // preserved so the lowered plan is deterministic.
        struct WindowGroup {
            partition_by: Vec<Expr>,
            order_by: Vec<SortExpr>,
            exprs: Vec<WindowExpr>,
        }
        let mut window_groups: Vec<WindowGroup> = Vec::new();
        let mut next_window_id: usize = 0;

        // First, lower each SELECT item to a projection expr. For window
        // items the expr becomes `Column("__window_N")` referencing the
        // appended window column.
        let mut proj_exprs: Vec<Expr> = Vec::with_capacity(items.len());
        for (sql_expr, alias) in &items {
            if let Some(pw) = try_window(sql_expr, &resolver, 0)? {
                let out_name = format!("__window_{next_window_id}");
                next_window_id += 1;
                let we = WindowExpr {
                    func: pw.func,
                    output_name: out_name.clone(),
                };
                // Find (or create) the group with a matching spec.
                let group_idx = window_groups.iter().position(|g| {
                    window_specs_eq(&g.partition_by, &g.order_by, &pw.partition_by, &pw.order_by)
                });
                match group_idx {
                    Some(i) => window_groups[i].exprs.push(we),
                    None => window_groups.push(WindowGroup {
                        partition_by: pw.partition_by,
                        order_by: pw.order_by,
                        exprs: vec![we],
                    }),
                }
                let col_ref = Expr::Column(out_name);
                proj_exprs.push(match alias {
                    Some(name) => col_ref.alias(name.clone()),
                    None => col_ref,
                });
                continue;
            }
            // Non-window item: reject any window function nested inside it so
            // the user gets a clear message rather than a silent miss.
            if sql_expr_contains_window(sql_expr, &resolver, 0)? {
                return Err(BoltError::Sql(
                    "window functions are only supported as a top-level SELECT item \
                     (optionally aliased), not nested inside a larger expression"
                        .into(),
                ));
            }
            let lowered = lower_expr(sql_expr, &resolver, 0)?;
            proj_exprs.push(match alias {
                Some(name) => lowered.alias(name.clone()),
                None => lowered,
            });
        }

        // Stack the Window nodes (if any) over the current plan.
        for g in window_groups {
            plan = LogicalPlan::Window {
                input: Box::new(plan),
                window_exprs: g.exprs,
                partition_by: g.partition_by,
                order_by: g.order_by,
            };
        }

        plan = LogicalPlan::Project {
            input: Box::new(plan),
            exprs: proj_exprs,
        };
    }

    // (HAVING is handled inside the aggregate-mode branch above, where the
    // alias map is in scope. The scalar-projection branch rejects HAVING
    // outright since there is no aggregation context.)

    // SELECT DISTINCT: dedup the *output* rows (after projection, HAVING).
    if matches!(select.distinct, Some(Distinct::Distinct)) {
        plan = LogicalPlan::Distinct {
            input: Box::new(plan),
        };
    }

    Ok(plan)
}

/// Lower a single `TableFactor` into `(base_plan, qualifier, schema)`.
///
///   * `base_plan` is the [`LogicalPlan`] that produces the table's rows: a
///     `LogicalPlan::Scan` for a registered base table, or the inlined
///     (cloned) plan of an in-scope CTE when the reference names one.
///   * `qualifier` is the name that user-typed `qualifier.col` references
///     must match in the SELECT / WHERE / ON tree. When the user wrote
///     `FROM mytable AS t` it is `t`; with no alias it equals the table /
///     CTE name. Once an alias shadows the underlying name, the bare name is
///     no longer in scope as a qualifier (standard SQL semantics).
///   * `schema` is the `base_plan`'s output schema.
///
/// Resolution order for a bare `TableFactor::Table` name: an in-scope CTE
/// shadows a registered base table of the same name (standard SQL — the
/// `WITH`-defined name wins within its scope). Only bare references are
/// accepted (TVFs, version, hints, derived-table subqueries, etc. remain
/// rejected). Column-list aliases (`AS t (c1, c2)`) are rejected because they
/// would also require renaming the schema fields, which the frontend does not
/// implement.
fn lower_table_factor(
    tf: &TableFactor,
    provider: &dyn TableProvider,
    ctes: &CteScope,
) -> BoltResult<(LogicalPlan, String, Schema)> {
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
            // Table aliases: accept `AS t` (or just `t`) but reject the
            // column-list form `AS t (c1, c2)` — we do not implement the
            // column renaming that would require.
            let qualifier = match alias {
                None => table_name.clone(),
                Some(a) => {
                    if !a.columns.is_empty() {
                        return Err(BoltError::Sql(
                            "unsupported: table alias with column list (AS t(c1, c2))".into(),
                        ));
                    }
                    a.name.value.clone()
                }
            };
            // CTE reference: inline the CTE's lowered plan. An in-scope CTE
            // shadows a registered base table of the same name.
            if let Some(cte_plan) = ctes.get(&table_name) {
                let schema = cte_plan.schema()?;
                return Ok((cte_plan.clone(), qualifier, schema));
            }
            let schema = provider.schema(&table_name)?;
            let base_plan = LogicalPlan::Scan {
                table: table_name,
                projection: None,
                schema: schema.clone(),
            };
            Ok((base_plan, qualifier, schema))
        }
        TableFactor::Derived { .. } => Err(BoltError::Sql(
            "unsupported: subquery in FROM (derived table); use a WITH/CTE instead".into(),
        )),
        _ => Err(BoltError::Sql(
            "unsupported: only bare table references are allowed in FROM".into(),
        )),
    }
}

/// Whether a column (by its qualifier-local `original` name) is present on
/// the *left* side of the current join — i.e. in any of `resolver.tables`
/// except the last, which is the just-pushed RHS.
///
/// SQL-standard case folding (mirrors [`NameResolver::resolve_compound`]): an
/// exact match is tried first; if the lookup name is all-ASCII-lowercase
/// (i.e. an unquoted identifier), a case-insensitive fallback is attempted.
///
/// * `Ok(true)`  — present in exactly one left table.
/// * `Ok(false)` — not present on the left side at all.
/// * `Err(_)`    — present in more than one left table (ambiguous), so a
///   USING / NATURAL column cannot be unambiguously resolved.
fn left_join_column_present(
    resolver: &NameResolver<'_>,
    col: &str,
) -> BoltResult<bool> {
    let col_lc = !col.chars().any(|c| c.is_ascii_uppercase());
    // All scopes left of the RHS (which is always the last pushed scope).
    let left_scopes = &resolver.tables[..resolver.tables.len() - 1];
    let mut hits = 0usize;
    for scope in left_scopes {
        let matched = scope
            .cols
            .iter()
            .find(|c| c.original == col)
            .or_else(|| {
                if !col_lc {
                    return None;
                }
                scope.cols.iter().find(|c| c.original.eq_ignore_ascii_case(col))
            });
        if matched.is_some() {
            hits += 1;
        }
    }
    if hits > 1 {
        return Err(BoltError::Sql(format!(
            "JOIN column '{col}' is ambiguous: it appears in more than one \
             table on the left side of the join"
        )));
    }
    Ok(hits == 1)
}

/// Whether a column (by `original` name) is present in the RHS scope (the
/// last pushed table). Same case-folding rule as [`left_join_column_present`].
fn rhs_join_column_present(resolver: &NameResolver<'_>, col: &str) -> bool {
    let col_lc = !col.chars().any(|c| c.is_ascii_uppercase());
    let rhs = resolver
        .tables
        .last()
        .expect("RHS scope was pushed before constraint resolution");
    rhs.cols
        .iter()
        .any(|c| c.original == col || (col_lc && c.original.eq_ignore_ascii_case(col)))
}

/// Build the equi-pair for a join column shared by both sides.
///
/// USING / NATURAL join columns have the *same* name on each side, so the
/// pair is `(Column(col), Column(col))` — exactly the shape [`lower_join_on`]
/// produces for an explicit `left.c = right.c` ON clause: a bare (original)
/// column name on each side. The join executor resolves the left name against
/// the left input batch and the right name against the (un-renamed) right
/// input batch (see [`crate::exec::join`]), so the un-renamed original name is
/// the correct key for both — the `join_combined_schema` rename only applies
/// to the *output* schema, never to the key lookups.
fn join_column_equi_pair(col: &str) -> (Expr, Expr) {
    (Expr::Column(col.to_string()), Expr::Column(col.to_string()))
}

/// Desugar a `JOIN ... USING (c1, c2, ...)` clause into equi-join pairs.
///
/// Each named column must exist on exactly one left table and on the RHS.
/// Missing, ambiguous, or duplicate USING columns are rejected with a clear
/// message.
fn desugar_using_columns(
    cols: &[Ident],
    resolver: &NameResolver<'_>,
) -> BoltResult<Vec<(Expr, Expr)>> {
    if cols.is_empty() {
        return Err(BoltError::Sql(
            "JOIN ... USING (...) requires at least one column".into(),
        ));
    }
    let mut pairs = Vec::with_capacity(cols.len());
    let mut seen: Vec<String> = Vec::with_capacity(cols.len());
    for ident in cols {
        let name = ident_to_name(ident);
        // A repeated column in the USING list is a user error.
        if seen.iter().any(|s| s == &name) {
            return Err(BoltError::Sql(format!(
                "JOIN ... USING lists column '{name}' more than once"
            )));
        }
        seen.push(name.clone());

        if !left_join_column_present(resolver, &name)? {
            return Err(BoltError::Sql(format!(
                "JOIN ... USING column '{name}' is not present on the left side of the join"
            )));
        }
        if !rhs_join_column_present(resolver, &name) {
            return Err(BoltError::Sql(format!(
                "JOIN ... USING column '{name}' is not present in the right table"
            )));
        }
        pairs.push(join_column_equi_pair(&name));
    }
    Ok(pairs)
}

/// Desugar a `NATURAL JOIN` into equi-join pairs over every column common to
/// both sides.
///
/// The common set is every RHS column (by `original` name) that also resolves
/// on the left side. A `NATURAL JOIN` with no common column is rejected (it
/// would silently degenerate to a CROSS join, which is almost never the
/// intent), as is one whose common column is ambiguous on the left (surfaced
/// by [`left_join_column_present`]).
fn desugar_natural_columns(
    resolver: &NameResolver<'_>,
) -> BoltResult<Vec<(Expr, Expr)>> {
    // Snapshot the RHS column names first so we don't hold a borrow of
    // `resolver` across the `left_join_column_present` calls below.
    let rhs_cols: Vec<String> = resolver
        .tables
        .last()
        .expect("RHS scope was pushed before constraint resolution")
        .cols
        .iter()
        .map(|tc| tc.original.clone())
        .collect();
    let mut pairs = Vec::new();
    for col in &rhs_cols {
        // Match RHS columns against the left side by their original name.
        if left_join_column_present(resolver, col)? {
            pairs.push(join_column_equi_pair(col));
        }
    }
    if pairs.is_empty() {
        return Err(BoltError::Sql(
            "NATURAL JOIN has no common column between the two sides".into(),
        ));
    }
    Ok(pairs)
}

/// Which side of the current join an ON-clause column reference belongs to.
///
/// `Right` means the column qualifier names the table just being added to
/// the join (`rhs_table` in the caller). `Left` means it names any other
/// table already in scope (the accumulated FROM-tree to the left of this
/// join). `Unknown` is reserved for bare `Identifier(col)` references, which
/// carry no qualifier and so cannot be classified by the planner — those
/// pass straight through to keep the executor's existing bare-name
/// behaviour intact (`t1 JOIN t2 ON a = b` is still accepted exactly as it
/// was pre-validation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JoinSide {
    Left,
    Right,
    /// Bare identifier (no qualifier) — side cannot be determined at plan
    /// time. Preserved as-is to avoid regressing queries that relied on the
    /// pre-validation lenient lowering.
    Unknown,
}

/// Result of lowering an ON-clause: zero-or-more equi pairs plus an
/// optional residual non-equi filter.
///
/// * `equi_pairs` holds the `left.col = right.col` equalities suitable for
///   the hash-join fast path.
/// * `filter` is the conjunction of every non-equi leaf (`<`, `>`,
///   `BETWEEN`, etc.) — `None` if the ON clause is purely equi. The filter
///   is lowered against the combined left ++ right schema (the resolver
///   already carries the renamed names), so the executor can evaluate it
///   directly against the join's output via [`crate::exec::filter`].
///
/// v0.6 routing rule: if `filter` is `Some(_)`, the planner emits a
/// `LogicalPlan::Join` with the residual carried in
/// [`LogicalPlan::Join::filter`], and the executor switches to the
/// nested-loop path (see [`crate::exec::join`]).
struct JoinOnLowered {
    equi_pairs: Vec<(Expr, Expr)>,
    filter: Option<Expr>,
}

/// Lower a join ON-clause into the `(equi_pairs, filter)` split.
///
/// Conjunctive AND nodes are flattened; each leaf is classified as:
///   * `left.col = right.col` (cross-table) → goes into `equi_pairs`.
///   * Anything else (non-equi comparison, BETWEEN, mixed expressions) →
///     lowered via [`lower_expr`] and AND-folded into `filter`.
///
/// The ON clause must produce at least one of the two (pure CROSS uses a
/// different code path); a fully empty result is rejected. Same-side
/// equalities like `t1.a = t1.b` are still rejected with a clear message
/// (route through WHERE instead) when both sides are qualified.
fn lower_join_on(
    e: &SqlExpr,
    resolver: &NameResolver<'_>,
    rhs_table: &str,
) -> BoltResult<JoinOnLowered> {
    let mut out = JoinOnLowered {
        equi_pairs: Vec::new(),
        filter: None,
    };
    collect_join_eq(e, &mut out, resolver, rhs_table, 0)?;
    if out.equi_pairs.is_empty() && out.filter.is_none() {
        return Err(BoltError::Sql(
            "JOIN ON clause must contain at least one predicate".into(),
        ));
    }
    Ok(out)
}

/// AND a new conjunct into the lowered ON-clause filter slot. The first
/// non-equi conjunct populates `filter`; subsequent ones are folded with
/// `AND` so the executor sees one composite predicate. Conjunctive
/// flattening matches the SQL standard's left-associative AND semantics.
fn and_into_filter(slot: &mut Option<Expr>, conjunct: Expr) {
    match slot.take() {
        None => *slot = Some(conjunct),
        Some(prev) => {
            *slot = Some(Expr::Binary {
                op: BinaryOp::And,
                left: Box::new(prev),
                right: Box::new(conjunct),
            });
        }
    }
}

/// Walk `e` flattening `AND` nodes. Each leaf is either an equi-join
/// equality (extracted into `out.equi_pairs`) or a non-equi conjunct
/// (lowered via [`lower_expr`] and AND-folded into `out.filter`).
///
/// `depth` is the current recursion depth; returns Err if MAX_RECURSION_DEPTH is exceeded.
fn collect_join_eq(
    e: &SqlExpr,
    out: &mut JoinOnLowered,
    resolver: &NameResolver<'_>,
    rhs_table: &str,
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    match e {
        SqlExpr::Nested(inner) => collect_join_eq(inner, out, resolver, rhs_table, depth + 1),
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            collect_join_eq(left, out, resolver, rhs_table, depth + 1)?;
            collect_join_eq(right, out, resolver, rhs_table, depth + 1)
        }
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } => {
            // Try the equi-join fast-path classification first. Both sides
            // must be simple column refs whose join-side is determinable.
            // If either side is a computed expression (e.g. `t1.a + 1`) the
            // equality falls through to the residual filter — but unknown
            // qualifiers (`t3.a` when t3 isn't in FROM) MUST hard-error so
            // typos surface immediately instead of being deferred to a
            // confusing runtime "column not found".
            let left_side = classify_join_side(left, resolver, rhs_table)?;
            let right_side = classify_join_side(right, resolver, rhs_table)?;
            match (left_side, right_side) {
                (Some((l, lside)), Some((r, rside))) => {
                    // Cross-table equi: same-side already rejected here for
                    // clarity (same-side equality on a JOIN ON is almost
                    // always a user typo and should be in WHERE). `Unknown`
                    // (bare identifier) is treated as cross-table per the
                    // lenient pre-v0.6 contract.
                    if lside != JoinSide::Unknown && lside == rside {
                        return Err(BoltError::Sql(format!(
                            "JOIN ON ... both sides reference the same table; \
                             expected a cross-table equality (got {left} = {right})"
                        )));
                    }
                    out.equi_pairs.push((l, r));
                    Ok(())
                }
                _ => {
                    // At least one side is a computed expression — route
                    // the whole equality into the residual filter so the
                    // nested-loop executor evaluates it host-side.
                    let lowered = lower_expr(e, resolver, depth + 1)?;
                    and_into_filter(&mut out.filter, lowered);
                    Ok(())
                }
            }
        }
        // Non-equi comparisons (<, <=, >, >=, <>) and the boolean operators
        // OR / NOT — the executor's nested-loop fallback handles these via
        // host-side predicate evaluation. We lower the whole leaf and push
        // it into the filter slot.
        SqlExpr::BinaryOp {
            op:
                BinaryOperator::Lt
                | BinaryOperator::LtEq
                | BinaryOperator::Gt
                | BinaryOperator::GtEq
                | BinaryOperator::NotEq
                | BinaryOperator::Or,
            ..
        } => {
            let lowered = lower_expr(e, resolver, depth + 1)?;
            and_into_filter(&mut out.filter, lowered);
            Ok(())
        }
        // `x BETWEEN low AND high` → lower as `low <= x AND x <= high` and
        // route to the residual filter. `negated` flips to
        // `x < low OR x > high`.
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => {
            let x = lower_expr(expr, resolver, depth + 1)?;
            let lo = lower_expr(low, resolver, depth + 1)?;
            let hi = lower_expr(high, resolver, depth + 1)?;
            let lowered = if *negated {
                // x < low OR x > high
                Expr::Binary {
                    op: BinaryOp::Or,
                    left: Box::new(Expr::Binary {
                        op: BinaryOp::Lt,
                        left: Box::new(x.clone()),
                        right: Box::new(lo),
                    }),
                    right: Box::new(Expr::Binary {
                        op: BinaryOp::Gt,
                        left: Box::new(x),
                        right: Box::new(hi),
                    }),
                }
            } else {
                // low <= x AND x <= high
                Expr::Binary {
                    op: BinaryOp::And,
                    left: Box::new(Expr::Binary {
                        op: BinaryOp::LtEq,
                        left: Box::new(lo),
                        right: Box::new(x.clone()),
                    }),
                    right: Box::new(Expr::Binary {
                        op: BinaryOp::LtEq,
                        left: Box::new(x),
                        right: Box::new(hi),
                    }),
                }
            };
            and_into_filter(&mut out.filter, lowered);
            Ok(())
        }
        other => Err(BoltError::Sql(format!(
            "unsupported JOIN ON predicate shape: {other}"
        ))),
    }
}

/// Lower one side of an equi-join predicate and report which side of the
/// join it lives on. We accept either a bare identifier or a `table.column`
/// qualified identifier so users can disambiguate same-named columns; both
/// lower to a plain `Column` ref (qualified column lookups beyond bare-name
/// matching aren't supported in 0.1.x but the parser accepts them so error
/// messages stay friendly).
///
/// Side classification:
///   - `CompoundIdentifier([qual, col])` with `qual == rhs_table` → `Right`.
///   - `CompoundIdentifier([qual, col])` with `qual` matching an earlier
///     scope in `resolver` → `Left`.
///   - `CompoundIdentifier([qual, col])` with `qual` unknown → hard error
///     (`JOIN ON ... references unknown table '{qual}'`).
///   - bare `Identifier` → `Unknown`. The caller treats this as "could be
///     either side" and so will not raise a same-side error; matches the
///     pre-validation lenient behaviour.
fn lower_join_side(
    e: &SqlExpr,
    resolver: &NameResolver<'_>,
    rhs_table: &str,
) -> BoltResult<(Expr, JoinSide)> {
    match e {
        SqlExpr::Identifier(ident) => {
            // Bare column reference — no qualifier to verify. Leave the
            // side undetermined so the caller doesn't accidentally treat
            // it as same-side. SQL-standard case folding (v0.5): the
            // lowered column name is the folded form (lowercase unless
            // the user quoted the identifier).
            Ok((Expr::Column(ident_to_name(ident)), JoinSide::Unknown))
        }
        SqlExpr::CompoundIdentifier(parts) => {
            // `table.col` — keep only the trailing column name in the
            // lowered Expr (the executor compares by output column name).
            // The qualifier is used purely for plan-time side validation.
            if parts.len() < 2 {
                let last = parts.last().ok_or_else(|| {
                    BoltError::Sql("empty compound identifier in JOIN ON".into())
                })?;
                return Ok((Expr::Column(ident_to_name(last)), JoinSide::Unknown));
            }
            if parts.len() > 2 {
                let full = parts
                    .iter()
                    .map(|p| p.value.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                if parts.len() == 3 {
                    return Err(BoltError::Sql(format!(
                        "schema-qualified names not supported: '{full}' in JOIN ON \
                         (only `table.col` / `alias.col` is accepted)"
                    )));
                }
                return Err(BoltError::Sql(format!(
                    "unsupported: deeply qualified column reference '{full}' in JOIN ON"
                )));
            }
            // SQL-standard case folding (v0.5): qualifier and column are
            // folded independently. The qualifier match against the
            // resolver is ASCII case-insensitive so a folded `t1` still
            // matches a table the host registered as `T1`.
            let qualifier = ident_to_name(&parts[0]);
            let col = ident_to_name(&parts[1]);
            // Verify the qualifier exists somewhere in the FROM-tree
            // (resolver already contains every in-scope table, including
            // the rhs we just pushed). This catches `t3.a` when only `t1`
            // and `t2` are joined.
            let in_scope = resolver
                .tables
                .iter()
                .any(|t| t.name == qualifier || t.name.eq_ignore_ascii_case(&qualifier));
            if !in_scope {
                return Err(BoltError::Sql(format!(
                    "JOIN ON ... references unknown table '{qualifier}' \
                     in column reference '{qualifier}.{col}'"
                )));
            }
            let side = if qualifier == rhs_table || qualifier.eq_ignore_ascii_case(rhs_table) {
                JoinSide::Right
            } else {
                JoinSide::Left
            };
            Ok((Expr::Column(col.clone()), side))
        }
        // Computed expressions (binary ops, function calls, literals, …)
        // are not valid sides for the equi-join fast path. The caller
        // routes them through the residual filter via
        // [`classify_join_side`]; this arm exists so a direct call still
        // produces a clear message rather than panicking.
        other => Err(BoltError::Sql(format!(
            "JOIN ON equi sides must be column references; got {other} (use the residual filter path for computed predicates)"
        ))),
    }
}

/// 3-valued classifier used by [`collect_join_eq`] to decide whether an
/// equality leaf can route through the equi-join fast path.
///
/// * `Ok(Some((expr, side)))` — `e` is a simple column reference and we
///   know which side of the join it belongs to. Caller pushes the pair
///   into `equi_pairs`.
/// * `Ok(None)`               — `e` is a computed expression (e.g.
///   `t1.a + 1`). Caller falls back to lowering the whole equality as a
///   residual filter.
/// * `Err(_)`                 — `e` is structurally malformed (deeply
///   qualified, unknown table). Caller propagates immediately so user
///   typos surface here rather than as a confusing runtime error.
///
/// This is a thin wrapper over [`lower_join_side`] that distinguishes
/// "this isn't a column ref" (recoverable, route to filter) from
/// "this is a column ref but it's broken" (hard error). The original
/// `lower_join_side` shape conflated both with `Err`.
fn classify_join_side(
    e: &SqlExpr,
    resolver: &NameResolver<'_>,
    rhs_table: &str,
) -> BoltResult<Option<(Expr, JoinSide)>> {
    match e {
        SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
            // Column-ref shapes: defer to lower_join_side. Its Err here is
            // load-bearing (unknown qualifier, deeply qualified ref) and
            // must propagate so the user sees a clear plan error.
            lower_join_side(e, resolver, rhs_table).map(Some)
        }
        // Anything else (BinaryOp, function call, literal, …) is a
        // computed expression — not a candidate for the equi-join fast
        // path. Route the caller to the residual-filter path.
        _ => Ok(None),
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

/// Pull the table name out of an `ObjectName`, accepting an optional single
/// schema/catalog qualifier.
///
/// The engine is single-catalog, so a leading qualifier carries no semantic
/// weight — `main.sales` and `sales` name the same table. We therefore accept
/// the common `schema.table` shape by treating the *trailing* identifier as
/// the table name and discarding the qualifier. A bare `table` is the
/// one-part case. Three or more parts (`catalog.schema.table`) are still
/// rejected: there is no second level of namespacing to collapse them into.
///
/// SQL-standard case folding (v0.5): unquoted table names are folded to
/// lowercase; quoted names (`"MyTable"`) keep their case verbatim. See
/// [`ident_to_name`] for the rule. Folding applies to the trailing (table)
/// part only — the discarded qualifier never reaches the lowered IR.
fn single_ident_from_object_name(name: &ObjectName) -> BoltResult<String> {
    match name.0.len() {
        // `table` — the common bare case.
        1 => Ok(ident_to_name(&name.0[0])),
        // `schema.table` / `catalog.table` — single-catalog engine, so the
        // qualifier is accepted and dropped; the table is resolved by its
        // base (trailing) name.
        2 => Ok(ident_to_name(&name.0[1])),
        // `catalog.schema.table` and deeper: no namespace to fold these into.
        _ => Err(BoltError::Sql(format!(
            "qualified table names support at most one schema/catalog \
             qualifier (e.g. `schema.table`); got: {name}"
        ))),
    }
}

/// Recognize a top-level aggregate function call. Returns `Ok(None)` for non-aggregates.
///
/// `depth` is the current recursion depth; returns Err if MAX_RECURSION_DEPTH is exceeded.
fn try_aggregate(
    e: &SqlExpr,
    resolver: &NameResolver<'_>,
    depth: usize,
) -> BoltResult<Option<AggregateExpr>> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    let func = match e {
        SqlExpr::Function(f) => f,
        _ => return Ok(None),
    };
    if func.name.0.len() != 1 {
        return Ok(None);
    }
    let fname = func.name.0[0].value.to_ascii_uppercase();
    // Plain `STDDEV(x)` is treated as `STDDEV_SAMP(x)` per the SQL standard
    // (the same convention DuckDB, Postgres, and Snowflake follow). Lower it
    // to the canonical `STDDEV_SAMP` spelling before the variant match below
    // so there is exactly one downstream representation per aggregate.
    let kind = match fname.as_str() {
        "COUNT" | "SUM" | "MIN" | "MAX" | "AVG"
        | "VAR_POP" | "VAR_SAMP" | "STDDEV_POP" | "STDDEV_SAMP" => fname,
        "VARIANCE" => "VAR_SAMP".to_string(),
        "STDDEV" => "STDDEV_SAMP".to_string(),
        _ => {
            // Surface "did you mean...?" hint for near-miss aggregate names
            // before falling through to scalar-function rejection downstream.
            const KNOWN_AGGREGATES: &[&str] = &[
                "COUNT", "SUM", "MIN", "MAX", "AVG",
                "VAR_POP", "VAR_SAMP", "VARIANCE",
                "STDDEV_POP", "STDDEV_SAMP", "STDDEV",
            ];
            if let Some(hint) = crate::plan::suggest::closest_match(
                &fname,
                KNOWN_AGGREGATES.iter().copied(),
            ) {
                let original = &func.name.0[0].value;
                return Err(BoltError::Sql(format!(
                    "unknown function '{original}' (did you mean '{hint}'?)"
                )));
            }
            return Ok(None);
        }
    };

    // An OVER clause makes this a *window* function, not a plain aggregate.
    // Defer to the window-lowering path (`try_window`) by reporting "not an
    // aggregate" here so the caller routes it correctly.
    if func.over.is_some() {
        return Ok(None);
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
        Some(e) => lower_expr(e, resolver, depth + 1)?,
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
        "VAR_POP" => AggregateExpr::VarPop(Box::new(inner)),
        "VAR_SAMP" => AggregateExpr::VarSamp(Box::new(inner)),
        "STDDEV_POP" => AggregateExpr::StddevPop(Box::new(inner)),
        "STDDEV_SAMP" => AggregateExpr::StddevSamp(Box::new(inner)),
        _ => unreachable!("kind already filtered above"),
    }))
}

/// Recognise the special `COUNT(DISTINCT <expr>)` SELECT item.
///
/// `COUNT(DISTINCT col)` is the one aggregate form where the `DISTINCT`
/// argument-quantifier is meaningful to us: it counts the number of distinct
/// *non-NULL* values of `col` (standard SQL — `DISTINCT` inside an aggregate
/// excludes NULLs just like the bare aggregate does). [`plan_select`] lowers a
/// sole, GROUP-BY-free `COUNT(DISTINCT col)` to
/// `Aggregate(COUNT) ∘ Distinct ∘ Project([col]) ∘ Filter(col IS NOT NULL)`;
/// see that call site for the wiring.
///
/// Returns:
///   * `Ok(Some(expr))` — `e` is `COUNT(DISTINCT <expr>)`; `expr` is the
///     *unlowered* SQL argument (the caller lowers it against the resolver).
///   * `Ok(None)`        — `e` is not a `COUNT(DISTINCT ...)` call (could be a
///     plain aggregate, a window, or a scalar expression).
///   * `Err(_)`          — `e` is a malformed / unsupported `COUNT(DISTINCT)`
///     shape that must surface a clear message rather than fall through:
///     `COUNT(DISTINCT *)`, `COUNT(DISTINCT a, b)`, `DISTINCT` on a windowed
///     `COUNT`, or `DISTINCT` on any aggregate other than `COUNT` (the latter
///     two are detected here so the error names the exact unsupported form).
fn try_count_distinct<'a>(
    e: &'a SqlExpr,
    _resolver: &NameResolver<'_>,
) -> BoltResult<Option<&'a SqlExpr>> {
    let func = match e {
        SqlExpr::Function(f) => f,
        _ => return Ok(None),
    };
    if func.name.0.len() != 1 {
        return Ok(None);
    }
    let fname = func.name.0[0].value.to_ascii_uppercase();
    let arg_list = match &func.args {
        FunctionArguments::List(list) => list,
        // No argument list at all (`COUNT`/`COUNT()`); not a DISTINCT form.
        FunctionArguments::None | FunctionArguments::Subquery(_) => return Ok(None),
    };
    // Only the `DISTINCT` quantifier concerns us here. `ALL` (or absent)
    // routes through the ordinary aggregate path.
    if !matches!(
        arg_list.duplicate_treatment,
        Some(sqlparser::ast::DuplicateTreatment::Distinct)
    ) {
        return Ok(None);
    }
    // From here on we KNOW the user wrote `<AGG>(DISTINCT ...)`. Anything we
    // can't lower must produce a precise error (rather than `Ok(None)`, which
    // would let `try_aggregate`'s generic "DISTINCT inside {kind}" rejection
    // fire and lose the specificity these messages provide).

    // DISTINCT is only supported inside COUNT.
    if fname != "COUNT" {
        return Err(BoltError::Sql(format!(
            "unsupported: DISTINCT inside {fname}; only COUNT(DISTINCT col) is supported"
        )));
    }
    // A windowed `COUNT(DISTINCT col) OVER (...)` is out of scope.
    if func.over.is_some() {
        return Err(BoltError::Sql(
            "unsupported: COUNT(DISTINCT ...) OVER (window)".into(),
        ));
    }
    if func.filter.is_some() {
        return Err(BoltError::Sql(
            "unsupported: FILTER on COUNT(DISTINCT ...)".into(),
        ));
    }
    if arg_list.args.len() != 1 {
        return Err(BoltError::Sql(format!(
            "COUNT(DISTINCT ...) expects exactly one argument, got {}",
            arg_list.args.len()
        )));
    }
    match &arg_list.args[0] {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(inner)) => Ok(Some(inner)),
        // `COUNT(DISTINCT *)` is meaningless (and ambiguous) — reject clearly.
        FunctionArg::Unnamed(FunctionArgExpr::Wildcard)
        | FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(_)) => Err(BoltError::Sql(
            "COUNT(DISTINCT *) is not supported; use COUNT(DISTINCT <column>)".into(),
        )),
        FunctionArg::Named { .. } => Err(BoltError::Sql(
            "unsupported: named argument to COUNT(DISTINCT ...)".into(),
        )),
    }
}

/// A window function call recognised in a SELECT item: the function plus its
/// parsed `OVER (...)` spec. Produced by [`try_window`] and consumed by the
/// window-lowering block in [`plan_select`].
struct ParsedWindow {
    /// The window function (ranking or aggregate).
    func: WindowFunc,
    /// `PARTITION BY` keys (lowered).
    partition_by: Vec<Expr>,
    /// `ORDER BY` keys within the partition (lowered).
    order_by: Vec<SortExpr>,
}

/// Recognise a top-level window-function call: `func(...) OVER (...)`.
///
/// Supports the ranking functions `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`
/// and the aggregate windows `SUM/AVG/MIN/MAX/COUNT(expr) OVER (...)`.
/// Returns `Ok(None)` if `e` is not a function call carrying an `OVER`
/// clause; returns an error for an OVER clause we recognise the *shape* of
/// but can't support (named windows, explicit non-default frames, etc.).
///
/// `depth` is the current recursion depth; returns Err if MAX_RECURSION_DEPTH
/// is exceeded.
fn try_window(
    e: &SqlExpr,
    resolver: &NameResolver,
    depth: usize,
) -> BoltResult<Option<ParsedWindow>> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    let func = match e {
        SqlExpr::Function(f) => f,
        _ => return Ok(None),
    };
    // No OVER clause => not a window function; let the aggregate / scalar
    // paths handle it.
    let over = match &func.over {
        Some(o) => o,
        None => return Ok(None),
    };
    if func.name.0.len() != 1 {
        return Ok(None);
    }
    let fname = func.name.0[0].value.to_ascii_uppercase();

    // Reject extension clauses we don't support on window functions, mirroring
    // the aggregate guard set.
    if func.filter.is_some() {
        return Err(BoltError::Sql(
            "unsupported: FILTER on window function".into(),
        ));
    }
    if func.null_treatment.is_some() {
        return Err(BoltError::Sql(
            "unsupported: IGNORE/RESPECT NULLS on window function".into(),
        ));
    }
    if !func.within_group.is_empty() {
        return Err(BoltError::Sql(
            "unsupported: WITHIN GROUP on window function".into(),
        ));
    }
    if !matches!(func.parameters, FunctionArguments::None) {
        return Err(BoltError::Sql(
            "unsupported: parametric window function".into(),
        ));
    }

    // Parse the OVER (...) spec.
    let spec = match over {
        WindowType::WindowSpec(s) => s,
        WindowType::NamedWindow(name) => {
            return Err(BoltError::Sql(format!(
                "unsupported: named window reference 'OVER {name}'"
            )));
        }
    };
    if spec.window_name.is_some() {
        return Err(BoltError::Sql(
            "unsupported: named window in OVER (...)".into(),
        ));
    }
    // Frame handling: only the SQL default frame is supported (RANGE/ROWS
    // UNBOUNDED PRECEDING [AND CURRENT ROW]). Anything else is rejected
    // cleanly so we never silently compute the wrong frame.
    if let Some(frame) = &spec.window_frame {
        reject_non_default_frame(frame, &fname)?;
    }

    let partition_by = spec
        .partition_by
        .iter()
        .map(|e| lower_expr(e, resolver, depth + 1))
        .collect::<BoltResult<Vec<_>>>()?;
    let order_by = lower_window_order_by(&spec.order_by, resolver, depth + 1)?;

    // Build the function. Ranking functions take no argument; aggregate
    // windows take exactly one.
    let func = match fname.as_str() {
        "ROW_NUMBER" => {
            reject_window_args(func, "ROW_NUMBER")?;
            WindowFunc::RowNumber
        }
        "RANK" => {
            reject_window_args(func, "RANK")?;
            WindowFunc::Rank
        }
        "DENSE_RANK" => {
            reject_window_args(func, "DENSE_RANK")?;
            WindowFunc::DenseRank
        }
        "SUM" | "AVG" | "MIN" | "MAX" | "COUNT" => {
            let inner = single_window_arg(func, &fname, resolver, depth + 1)?;
            match fname.as_str() {
                "SUM" => WindowFunc::Sum(inner),
                "AVG" => WindowFunc::Avg(inner),
                "MIN" => WindowFunc::Min(inner),
                "MAX" => WindowFunc::Max(inner),
                "COUNT" => WindowFunc::Count(inner),
                _ => unreachable!(),
            }
        }
        other => {
            return Err(BoltError::Sql(format!(
                "unsupported window function '{other}'; supported: ROW_NUMBER, RANK, \
                 DENSE_RANK, SUM, AVG, MIN, MAX, COUNT"
            )));
        }
    };

    Ok(Some(ParsedWindow {
        func,
        partition_by,
        order_by,
    }))
}

/// Structural equality of two lowered window specs (partition + ordering),
/// so multiple window functions sharing a spec collapse into a single
/// `LogicalPlan::Window` node. Compares partition keys by `expr_eq` and order
/// keys by `(expr_eq, descending, nulls_first)`.
fn window_specs_eq(
    a_part: &[Expr],
    a_order: &[SortExpr],
    b_part: &[Expr],
    b_order: &[SortExpr],
) -> bool {
    if a_part.len() != b_part.len() || a_order.len() != b_order.len() {
        return false;
    }
    if !a_part.iter().zip(b_part).all(|(x, y)| expr_eq(x, y)) {
        return false;
    }
    a_order.iter().zip(b_order).all(|(x, y)| {
        x.descending == y.descending
            && x.nulls_first == y.nulls_first
            && expr_eq(&x.expr, &y.expr)
    })
}

/// True if `e` contains a window function call (a `Function` with an `OVER`
/// clause) anywhere in its tree. Used to reject window functions nested
/// inside a larger SELECT expression, which the host executor does not lower
/// yet.
fn sql_expr_contains_window(
    e: &SqlExpr,
    resolver: &NameResolver,
    depth: usize,
) -> BoltResult<bool> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    // A function with an OVER clause is a window function regardless of name.
    if let SqlExpr::Function(f) = e {
        if f.over.is_some() {
            return Ok(true);
        }
    }
    // Recurse into the common composite expression shapes.
    let any = match e {
        SqlExpr::BinaryOp { left, right, .. } => {
            sql_expr_contains_window(left, resolver, depth + 1)?
                || sql_expr_contains_window(right, resolver, depth + 1)?
        }
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr)
        | SqlExpr::Nested(expr)
        | SqlExpr::Cast { expr, .. } => sql_expr_contains_window(expr, resolver, depth + 1)?,
        SqlExpr::Between {
            expr, low, high, ..
        } => {
            sql_expr_contains_window(expr, resolver, depth + 1)?
                || sql_expr_contains_window(low, resolver, depth + 1)?
                || sql_expr_contains_window(high, resolver, depth + 1)?
        }
        SqlExpr::Function(f) => {
            let mut found = false;
            if let FunctionArguments::List(list) = &f.args {
                for arg in &list.args {
                    if let FunctionArg::Unnamed(FunctionArgExpr::Expr(inner)) = arg {
                        if sql_expr_contains_window(inner, resolver, depth + 1)? {
                            found = true;
                            break;
                        }
                    }
                }
            }
            found
        }
        _ => false,
    };
    Ok(any)
}

/// Reject any window frame that isn't the SQL default
/// (`RANGE/ROWS [BETWEEN] UNBOUNDED PRECEDING [AND CURRENT ROW]`). The
/// executor only implements the default `RANGE UNBOUNDED PRECEDING AND
/// CURRENT ROW` frame, so an explicit exotic frame must error rather than be
/// silently mis-evaluated.
fn reject_non_default_frame(
    frame: &sqlparser::ast::WindowFrame,
    fname: &str,
) -> BoltResult<()> {
    // Units may be ROWS or RANGE; both collapse to the same default-frame
    // behaviour here because the only start bound we accept is UNBOUNDED
    // PRECEDING (under which ROWS and RANGE agree). GROUPS is rejected.
    if matches!(frame.units, WindowFrameUnits::Groups) {
        return Err(BoltError::Sql(format!(
            "unsupported: GROUPS frame on window function {fname}"
        )));
    }
    // start_bound must be UNBOUNDED PRECEDING (`Preceding(None)`).
    if !matches!(frame.start_bound, WindowFrameBound::Preceding(None)) {
        return Err(BoltError::Sql(format!(
            "unsupported window frame on {fname}: only the default \
             'UNBOUNDED PRECEDING [AND CURRENT ROW]' frame is supported"
        )));
    }
    // end_bound, if present, must be CURRENT ROW (the default).
    match &frame.end_bound {
        None => {}
        Some(WindowFrameBound::CurrentRow) => {}
        Some(_) => {
            return Err(BoltError::Sql(format!(
                "unsupported window frame on {fname}: only 'UNBOUNDED PRECEDING \
                 AND CURRENT ROW' is supported"
            )));
        }
    }
    Ok(())
}

/// Lower a window-spec ORDER BY list into our `SortExpr`s. Identical default
/// rules to [`lower_order_by`] but resolves against the in-scope resolver
/// (window specs are evaluated in the FROM-tree's namespace, not the
/// post-projection one).
fn lower_window_order_by(
    exprs: &[OrderByExpr],
    resolver: &NameResolver,
    depth: usize,
) -> BoltResult<Vec<SortExpr>> {
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
                "unsupported: window ORDER BY ... WITH FILL".into(),
            ));
        }
        let descending = matches!(asc, Some(false));
        let nulls_first = match nulls_first {
            Some(b) => *b,
            None => !descending,
        };
        out.push(SortExpr {
            expr: lower_expr(expr, resolver, depth + 1)?,
            descending,
            nulls_first,
        });
    }
    Ok(out)
}

/// Reject arguments on an argument-less ranking window function
/// (`ROW_NUMBER`, `RANK`, `DENSE_RANK`).
fn reject_window_args(
    func: &sqlparser::ast::Function,
    name: &str,
) -> BoltResult<()> {
    let empty = match &func.args {
        FunctionArguments::None => true,
        FunctionArguments::List(list) => list.args.is_empty(),
        FunctionArguments::Subquery(_) => false,
    };
    if !empty {
        return Err(BoltError::Sql(format!(
            "{name}() is a ranking window function and takes no arguments"
        )));
    }
    Ok(())
}

/// Extract the single argument expression of an aggregate window function,
/// lowering it. `COUNT(*)` is accepted as a count of all rows (lowered to a
/// constant `1` sentinel, matching the scalar-aggregate convention).
fn single_window_arg(
    func: &sqlparser::ast::Function,
    name: &str,
    resolver: &NameResolver,
    depth: usize,
) -> BoltResult<Expr> {
    let arg_list = match &func.args {
        FunctionArguments::List(list) => list,
        FunctionArguments::None => {
            return Err(BoltError::Sql(format!(
                "{name}(...) OVER (...) requires an argument"
            )));
        }
        FunctionArguments::Subquery(_) => {
            return Err(BoltError::Sql(format!(
                "unsupported: subquery argument to window {name}"
            )));
        }
    };
    if arg_list.duplicate_treatment.is_some() {
        return Err(BoltError::Sql(format!(
            "unsupported: DISTINCT/ALL inside window {name}"
        )));
    }
    if !arg_list.clauses.is_empty() {
        return Err(BoltError::Sql(format!(
            "unsupported: argument clauses on window {name}"
        )));
    }
    if arg_list.args.len() != 1 {
        return Err(BoltError::Sql(format!(
            "window {name} expects exactly one argument, got {}",
            arg_list.args.len()
        )));
    }
    match &arg_list.args[0] {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => lower_expr(e, resolver, depth + 1),
        FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => {
            if name != "COUNT" {
                return Err(BoltError::Sql(format!("{name}(*) is not supported")));
            }
            // COUNT(*) sentinel: a literal 1 (counts rows regardless of value).
            Ok(Expr::Literal(Literal::Int64(1)))
        }
        FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(_)) => Err(BoltError::Sql(
            format!("unsupported: qualified wildcard in window {name}"),
        )),
        FunctionArg::Named { .. } => Err(BoltError::Sql(format!(
            "unsupported: named argument to window {name}"
        ))),
    }
}

/// Recognise a string scalar function call (UPPER / LOWER / LENGTH / CONCAT)
/// at the SQL-frontend layer and lower it into an `Expr::ScalarFn`.
/// Returns `Ok(None)` for any other function name so the caller can produce
/// the catch-all "scalar function calls are not supported" rejection.
///
/// Aggregate names (COUNT/SUM/MIN/MAX/AVG) are intentionally NOT recognised
/// here — they're split off earlier by [`try_aggregate`] before `lower_expr`
/// gets a chance to see them. SUBSTRING is also NOT routed here because
/// sqlparser parses it as a dedicated `SqlExpr::Substring` variant rather
/// than a generic `Function` call; that variant is handled directly in
/// [`lower_expr`].
///
/// Argument lowering reuses the standard `lower_expr` path so column
/// resolution, NULL peer typing, and recursion-depth tracking all behave
/// identically inside function arguments. Type-checking (Utf8 / Int64
/// shape, arity bounds) lives in
/// [`crate::plan::logical_plan::scalar_fn_dtype`], which fires when the
/// plan's schema is queried; we don't pre-check here.
fn try_string_scalar_fn(
    func: &sqlparser::ast::Function,
    resolver: &NameResolver<'_>,
    depth: usize,
) -> BoltResult<Option<Expr>> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    if func.name.0.len() != 1 {
        return Ok(None);
    }
    let fname = func.name.0[0].value.to_ascii_uppercase();
    let kind = match fname.as_str() {
        "UPPER" => ScalarFnKind::Upper,
        "LOWER" => ScalarFnKind::Lower,
        "LENGTH" => ScalarFnKind::Length,
        "CONCAT" => ScalarFnKind::Concat,
        // Note: "SUBSTRING" is parsed by sqlparser as `SqlExpr::Substring`,
        // not `SqlExpr::Function`, so we never see it here. If we ever do
        // (e.g. via a non-standard dialect), explicitly reject so callers
        // route through the dedicated `Substring` arm and pick up the
        // FROM/FOR slot semantics correctly.
        "SUBSTRING" => {
            return Err(BoltError::Sql(
                "SUBSTRING must use SUBSTRING(s FROM i [FOR n]) or SUBSTRING(s, i [, n]) syntax"
                    .into(),
            ));
        }
        _ => return Ok(None),
    };

    let name = kind.sql_name();
    // Disallow OVER (window), FILTER, ORDER BY, WITHIN GROUP, parameters —
    // identical guard set to `try_aggregate` so any escape hatch the parser
    // produces lands as a `BoltError::Sql` here rather than silently
    // ignored.
    if func.over.is_some() {
        return Err(BoltError::Sql(format!(
            "unsupported: OVER clause on {name}"
        )));
    }
    if func.filter.is_some() {
        return Err(BoltError::Sql(format!(
            "unsupported: FILTER clause on {name}"
        )));
    }
    if func.null_treatment.is_some() {
        return Err(BoltError::Sql(format!(
            "unsupported: IGNORE/RESPECT NULLS on {name}"
        )));
    }
    if !func.within_group.is_empty() {
        return Err(BoltError::Sql(format!(
            "unsupported: WITHIN GROUP on {name}"
        )));
    }
    if !matches!(func.parameters, FunctionArguments::None) {
        return Err(BoltError::Sql(format!(
            "unsupported: parametric {name}"
        )));
    }

    let arg_list = match &func.args {
        FunctionArguments::List(list) => list,
        FunctionArguments::None => {
            return Err(BoltError::Sql(format!("{name} requires arguments")));
        }
        FunctionArguments::Subquery(_) => {
            return Err(BoltError::Sql(format!(
                "unsupported: subquery argument to {name}"
            )));
        }
    };
    if arg_list.duplicate_treatment.is_some() {
        return Err(BoltError::Sql(format!(
            "unsupported: DISTINCT/ALL inside {name}"
        )));
    }
    if !arg_list.clauses.is_empty() {
        return Err(BoltError::Sql(format!(
            "unsupported: argument clauses on {name}"
        )));
    }

    let mut args: Vec<Expr> = Vec::with_capacity(arg_list.args.len());
    for arg in &arg_list.args {
        let e = match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => e,
            FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => {
                return Err(BoltError::Sql(format!(
                    "unsupported: wildcard argument to {name}"
                )));
            }
            FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(_)) => {
                return Err(BoltError::Sql(format!(
                    "unsupported: qualified wildcard in {name}"
                )));
            }
            FunctionArg::Named { .. } => {
                return Err(BoltError::Sql(format!(
                    "unsupported: named argument to {name}"
                )));
            }
        };
        args.push(lower_expr(e, resolver, depth + 1)?);
    }

    Ok(Some(Expr::ScalarFn { kind, args }))
}

/// True if `e` contains any aggregate function call (anywhere in the tree).
///
/// `depth` is the current recursion depth; returns Err if MAX_RECURSION_DEPTH is exceeded.
fn contains_aggregate(e: &SqlExpr, resolver: &NameResolver<'_>, depth: usize) -> BoltResult<bool> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    if try_aggregate(e, resolver, depth + 1)?.is_some() {
        return Ok(true);
    }
    match e {
        SqlExpr::BinaryOp { left, right, .. } => Ok(
            contains_aggregate(left, resolver, depth + 1)?
                || contains_aggregate(right, resolver, depth + 1)?,
        ),
        SqlExpr::UnaryOp { expr, .. } => contains_aggregate(expr, resolver, depth + 1),
        // `IS NULL` / `IS NOT NULL` carry an operand expression that may
        // itself contain an aggregate (e.g. `HAVING SUM(v) IS NOT NULL`).
        SqlExpr::IsNull(inner) | SqlExpr::IsNotNull(inner) => {
            contains_aggregate(inner, resolver, depth + 1)
        }
        SqlExpr::InList { expr, list, .. } => {
            if contains_aggregate(expr, resolver, depth + 1)? {
                return Ok(true);
            }
            for item in list {
                if contains_aggregate(item, resolver, depth + 1)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        SqlExpr::Between {
            expr, low, high, ..
        } => Ok(contains_aggregate(expr, resolver, depth + 1)?
            || contains_aggregate(low, resolver, depth + 1)?
            || contains_aggregate(high, resolver, depth + 1)?),
        SqlExpr::Like { expr, .. } => contains_aggregate(expr, resolver, depth + 1),
        // `CAST(<expr> AS <type>)` is a transparent wrapper — recurse into
        // the inner expression so `HAVING CAST(SUM(v) AS Int64) > 0`
        // is recognised as referencing an aggregate.
        SqlExpr::Cast { expr, .. } => contains_aggregate(expr, resolver, depth + 1),
        SqlExpr::Nested(inner) => contains_aggregate(inner, resolver, depth + 1),
        // CASE: any nested subtree may contain an aggregate call (operand
        // of a simple CASE, any WHEN condition, any THEN value, the ELSE).
        SqlExpr::Case {
            operand,
            conditions,
            results,
            else_result,
        } => {
            if let Some(o) = operand {
                if contains_aggregate(o, resolver, depth + 1)? {
                    return Ok(true);
                }
            }
            for c in conditions {
                if contains_aggregate(c, resolver, depth + 1)? {
                    return Ok(true);
                }
            }
            for r in results {
                if contains_aggregate(r, resolver, depth + 1)? {
                    return Ok(true);
                }
            }
            if let Some(e) = else_result {
                if contains_aggregate(e, resolver, depth + 1)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            if contains_aggregate(expr, resolver, depth + 1)? {
                return Ok(true);
            }
            if let Some(e) = substring_from {
                if contains_aggregate(e, resolver, depth + 1)? {
                    return Ok(true);
                }
            }
            if let Some(e) = substring_for {
                if contains_aggregate(e, resolver, depth + 1)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        SqlExpr::Function(f) => {
            if let FunctionArguments::List(list) = &f.args {
                for arg in &list.args {
                    if let FunctionArg::Unnamed(FunctionArgExpr::Expr(inner)) = arg {
                        if contains_aggregate(inner, resolver, depth + 1)? {
                            return Ok(true);
                        }
                    }
                }
            }
            Ok(false)
        }
        _ => Ok(false),
    }
}

/// Append `agg` to `aggregates` unless an aggregate with the same
/// `aggregate_output_name` is already present, in which case return the
/// existing index. Used by the post-aggregate scalar rewriter so that
/// `SUM(x) + SUM(x)` (and `SELECT SUM(x), SUM(x) + 1` from the same
/// SELECT list) feed off a single physical aggregate, and the rewritten
/// surface expression references the same output column at both
/// positions.
fn push_aggregate_dedup(aggregates: &mut Vec<AggregateExpr>, agg: AggregateExpr) -> usize {
    let name = aggregate_output_name(&agg);
    if let Some(pos) = aggregates
        .iter()
        .position(|a| aggregate_output_name(a) == name)
    {
        return pos;
    }
    let idx = aggregates.len();
    aggregates.push(agg);
    idx
}

/// Walk a SQL expression that contains one or more aggregate calls,
/// extracting each aggregate into `aggregates` (deduplicated by output
/// name via [`push_aggregate_dedup`]) and returning a lowered `Expr`
/// where every aggregate position has been replaced by
/// `Expr::Column(aggregate_output_name)`.
///
/// Used by `plan_select` for post-aggregate scalar SELECT items like
/// `SUM(price) + 1`, `AVG(qty) * 2`, `(SUM(a) + SUM(b)) / 2`. The returned
/// `Expr` is evaluated by the post-Aggregate Project against the
/// Aggregate's output schema.
///
/// Sub-expressions that contain no aggregate (e.g. the `1` in
/// `SUM(price) + 1`, or a bare column reference resolved at scan time
/// but kept in the Aggregate's output via GROUP BY) fall through to
/// [`lower_expr`] unchanged. That keeps the rule for non-aggregate
/// fragments — name resolution, value lowering, literal parsing — in a
/// single place.
///
/// `depth` is the current recursion depth; returns Err if
/// `MAX_RECURSION_DEPTH` is exceeded.
fn extract_and_rewrite_aggregates(
    e: &SqlExpr,
    resolver: &NameResolver<'_>,
    aggregates: &mut Vec<AggregateExpr>,
    depth: usize,
) -> BoltResult<Expr> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    // Aggregate call at this node: lower it, dedup against the pool,
    // return a reference to its output column.
    if let Some(agg) = try_aggregate(e, resolver, depth + 1)? {
        // Compute the dedup key *before* moving `agg` into the pool.
        let name = aggregate_output_name(&agg);
        let _idx = push_aggregate_dedup(aggregates, agg);
        return Ok(Expr::Column(name));
    }
    match e {
        SqlExpr::Nested(inner) => {
            extract_and_rewrite_aggregates(inner, resolver, aggregates, depth + 1)
        }
        SqlExpr::BinaryOp { left, op, right } => {
            let lop = lower_binary_op(op)?;
            let l = extract_and_rewrite_aggregates(left, resolver, aggregates, depth + 1)?;
            let r = extract_and_rewrite_aggregates(right, resolver, aggregates, depth + 1)?;
            Ok(Expr::Binary {
                op: lop,
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        SqlExpr::UnaryOp { op, expr } => match op {
            UnaryOperator::Plus => {
                extract_and_rewrite_aggregates(expr, resolver, aggregates, depth + 1)
            }
            UnaryOperator::Minus => {
                // Mirror `lower_expr_in_having`'s unary-minus handling: we
                // cannot fall through to `negate_expr` because it would
                // route the operand through `lower_expr` and reject any
                // aggregate call nested under the unary minus
                // (`-SUM(v)`). Lower the operand recursively then negate
                // structurally as `0 - operand`.
                let inner =
                    extract_and_rewrite_aggregates(expr, resolver, aggregates, depth + 1)?;
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
        SqlExpr::IsNull(inner) => {
            let operand =
                extract_and_rewrite_aggregates(inner, resolver, aggregates, depth + 1)?;
            Ok(Expr::Unary {
                op: UnaryOp::IsNull,
                operand: Box::new(operand),
            })
        }
        SqlExpr::IsNotNull(inner) => {
            let operand =
                extract_and_rewrite_aggregates(inner, resolver, aggregates, depth + 1)?;
            Ok(Expr::Unary {
                op: UnaryOp::IsNotNull,
                operand: Box::new(operand),
            })
        }
        // Any other shape (Identifier, CompoundIdentifier, Value, ...)
        // is by construction aggregate-free at this node (otherwise the
        // `try_aggregate` branch above would have fired). Defer to the
        // normal lowerer for name resolution and literal parsing.
        _ => lower_expr(e, resolver, depth + 1),
    }
}

/// Variant of `lower_expr` used inside a HAVING clause. Aggregate function
/// calls (anywhere in the tree) are rewritten into a bare `Column(name)`
/// where `name` is the column the post-aggregate Project produces for that
/// aggregate (per `aggregate_output_name`), unless the SELECT renamed that
/// aggregate with an alias. `agg_aliases` maps raw aggregate output name to
/// the SELECT alias. `depth` enforces MAX_RECURSION_DEPTH against adversarial
/// SQL nesting.
fn lower_expr_in_having(
    e: &SqlExpr,
    resolver: &NameResolver<'_>,
    agg_aliases: &HashMap<String, String>,
    depth: usize,
) -> BoltResult<Expr> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    if let Some(agg) = try_aggregate(e, resolver, depth + 1)? {
        let raw = aggregate_output_name(&agg);
        let resolved = agg_aliases.get(&raw).cloned().unwrap_or(raw);
        return Ok(Expr::Column(resolved));
    }
    match e {
        SqlExpr::Nested(inner) => lower_expr_in_having(inner, resolver, agg_aliases, depth + 1),
        SqlExpr::BinaryOp { left, op, right } => {
            let lop = lower_binary_op(op)?;
            let l = lower_expr_in_having(left, resolver, agg_aliases, depth + 1)?;
            let r = lower_expr_in_having(right, resolver, agg_aliases, depth + 1)?;
            Ok(Expr::Binary {
                op: lop,
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        SqlExpr::UnaryOp { op, expr } => match op {
            UnaryOperator::Plus => lower_expr_in_having(expr, resolver, agg_aliases, depth + 1),
            UnaryOperator::Minus => {
                // Re-use the aggregate-aware lowerer for the operand, then
                // negate by hand (we can't fall through to `negate_expr`
                // because it would route through `lower_expr` and reject
                // any aggregate call nested under the unary minus).
                let inner = lower_expr_in_having(expr, resolver, agg_aliases, depth + 1)?;
                Ok(Expr::Binary {
                    op: BinaryOp::Sub,
                    left: Box::new(Expr::Literal(Literal::Int64(0))),
                    right: Box::new(inner),
                })
            }
            // `HAVING NOT (...)`. Route the operand back through this same
            // aggregate-aware lowerer so e.g. `HAVING NOT (SUM(v) > 0)`
            // rewrites the aggregate call to the projected column ref
            // before wrapping in the `UnaryOp::Not` node.
            UnaryOperator::Not => {
                let inner = lower_expr_in_having(expr, resolver, agg_aliases, depth + 1)?;
                Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(inner),
                })
            }
            other => Err(BoltError::Sql(format!(
                "unsupported unary operator: {other:?}"
            ))),
        },
        // HAVING ... IS [NOT] NULL: handle the same way as the scalar
        // surface but route the operand through `lower_expr_in_having` so
        // aggregate calls inside the operand (`HAVING SUM(v) IS NOT NULL`)
        // are rewritten to the aggregate-output column reference.
        SqlExpr::IsNull(inner) => {
            let operand = lower_expr_in_having(inner, resolver, agg_aliases, depth + 1)?;
            Ok(Expr::Unary {
                op: UnaryOp::IsNull,
                operand: Box::new(operand),
            })
        }
        SqlExpr::IsNotNull(inner) => {
            let operand = lower_expr_in_having(inner, resolver, agg_aliases, depth + 1)?;
            Ok(Expr::Unary {
                op: UnaryOp::IsNotNull,
                operand: Box::new(operand),
            })
        }
        // HAVING `<probe> [NOT] IN (...)`: lower the probe and each list
        // element via the aggregate-aware lowerer so aggregate calls
        // anywhere in the operand or the list (e.g.
        // `HAVING SUM(v) IN (10, 20)`) resolve to the aggregate-output
        // column reference, then desugar to the same OR/AND chain shape
        // as scalar `lower_in_list`.
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => {
            if list.len() > MAX_IN_LIST_VALUES {
                return Err(BoltError::Sql(format!(
                    "IN with > {MAX_IN_LIST_VALUES} values not yet supported; use a JOIN \
                     against a derived table (got {} values)",
                    list.len()
                )));
            }
            if list.is_empty() {
                return Ok(Expr::Literal(Literal::Bool(*negated)));
            }
            let lowered_expr = lower_expr_in_having(expr, resolver, agg_aliases, depth + 1)?;
            let (cmp_op, combine_op) = if *negated {
                (BinaryOp::NotEq, BinaryOp::And)
            } else {
                (BinaryOp::Eq, BinaryOp::Or)
            };
            let mut acc: Option<Expr> = None;
            for item in list {
                let item_lowered =
                    lower_expr_in_having(item, resolver, agg_aliases, depth + 1)?;
                let cmp = Expr::Binary {
                    op: cmp_op,
                    left: Box::new(lowered_expr.clone()),
                    right: Box::new(item_lowered),
                };
                acc = Some(match acc {
                    None => cmp,
                    Some(prev) => Expr::Binary {
                        op: combine_op,
                        left: Box::new(prev),
                        right: Box::new(cmp),
                    },
                });
            }
            Ok(acc.expect("non-empty IN list guarantees at least one chain element"))
        }
        // HAVING ... BETWEEN ...: desugar the same way as the scalar lowerer
        // but route each operand through `lower_expr_in_having` so aggregate
        // calls inside any of {expr, low, high} are rewritten to the
        // aggregate-output column reference.
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => {
            let operand = lower_expr_in_having(expr, resolver, agg_aliases, depth + 1)?;
            let lo = lower_expr_in_having(low, resolver, agg_aliases, depth + 1)?;
            let hi = lower_expr_in_having(high, resolver, agg_aliases, depth + 1)?;
            if *negated {
                let lt_low = Expr::Binary {
                    op: BinaryOp::Lt,
                    left: Box::new(operand.clone()),
                    right: Box::new(lo),
                };
                let gt_high = Expr::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(operand),
                    right: Box::new(hi),
                };
                Ok(Expr::Binary {
                    op: BinaryOp::Or,
                    left: Box::new(lt_low),
                    right: Box::new(gt_high),
                })
            } else {
                let ge_low = Expr::Binary {
                    op: BinaryOp::GtEq,
                    left: Box::new(operand.clone()),
                    right: Box::new(lo),
                };
                let le_high = Expr::Binary {
                    op: BinaryOp::LtEq,
                    left: Box::new(operand),
                    right: Box::new(hi),
                };
                Ok(Expr::Binary {
                    op: BinaryOp::And,
                    left: Box::new(ge_low),
                    right: Box::new(le_high),
                })
            }
        }
        // HAVING ... CASE: walk every subtree through the aggregate-aware
        // lowerer so an aggregate call buried in a CASE arm resolves to the
        // post-aggregate column reference.
        SqlExpr::Case {
            operand,
            conditions,
            results,
            else_result,
        } => {
            let lowered_operand = match operand {
                Some(o) => Some(lower_expr_in_having(o, resolver, agg_aliases, depth + 1)?),
                None => None,
            };
            if conditions.len() != results.len() {
                return Err(BoltError::Sql(format!(
                    "CASE expression has {} conditions and {} results; they must match",
                    conditions.len(),
                    results.len(),
                )));
            }
            if conditions.is_empty() {
                return Err(BoltError::Sql(
                    "CASE expression requires at least one WHEN/THEN branch".into(),
                ));
            }
            let mut branches: Vec<(Expr, Expr)> = Vec::with_capacity(conditions.len());
            for (w, t) in conditions.iter().zip(results.iter()) {
                let cond = match &lowered_operand {
                    Some(op_expr) => {
                        let v = lower_expr_in_having(w, resolver, agg_aliases, depth + 1)?;
                        Expr::Binary {
                            op: BinaryOp::Eq,
                            left: Box::new(op_expr.clone()),
                            right: Box::new(v),
                        }
                    }
                    None => lower_expr_in_having(w, resolver, agg_aliases, depth + 1)?,
                };
                let then = lower_expr_in_having(t, resolver, agg_aliases, depth + 1)?;
                branches.push((cond, then));
            }
            let else_expr = match else_result {
                Some(e) => Some(Box::new(lower_expr_in_having(
                    e,
                    resolver,
                    agg_aliases,
                    depth + 1,
                )?)),
                None => None,
            };
            Ok(Expr::Case {
                branches,
                else_branch: else_expr,
            })
        }
        // HAVING CAST(...) — descend into the inner with the aggregate-aware
        // lowerer so `HAVING CAST(SUM(v) AS Int64) > 0` rewrites correctly.
        SqlExpr::Cast {
            kind,
            expr,
            data_type,
            format,
        } => {
            if format.is_some() {
                return Err(BoltError::Sql(
                    "CAST with FORMAT clause not supported".into(),
                ));
            }
            match kind {
                CastKind::Cast | CastKind::DoubleColon => {}
                CastKind::TryCast => {
                    return Err(BoltError::Sql(
                        "TRY_CAST not supported; use CAST".into(),
                    ));
                }
                CastKind::SafeCast => {
                    return Err(BoltError::Sql(
                        "SAFE_CAST not supported; use CAST".into(),
                    ));
                }
            }
            let target = lower_cast_data_type(data_type)?;
            let inner = lower_expr_in_having(expr, resolver, agg_aliases, depth + 1)?;
            Ok(Expr::Cast {
                expr: Box::new(inner),
                target,
            })
        }
        // Anything else is identical to a scalar HAVING fragment; defer to
        // the normal lowerer (which handles Identifier, Value, etc., and
        // still rejects bare non-aggregate Function calls).
        _ => lower_expr(e, resolver, depth + 1),
    }
}

/// Walk a lowered HAVING predicate and verify that every `Expr::Column`
/// reference names a field present in `project_schema`.
///
/// The HAVING `Filter` sits *outside* the SELECT-order `Project`, so its
/// predicate is resolved against that Project's output. If a user typos a
/// column (or references a name that isn't in the SELECT list and isn't a
/// group key the SELECT exposed), this catches it with a friendly error
/// before the schema-checker further down produces a less obvious
/// `column not found` error. The check is also a structural guard: if a
/// future refactor changes the HAVING/Project ordering and silently breaks
/// alias passthrough, the integration tests for this function will fail
/// loudly rather than the queries silently misbehaving.
fn validate_having_columns(predicate: &Expr, project_schema: &Schema) -> BoltResult<()> {
    let mut stack: Vec<&Expr> = vec![predicate];
    while let Some(e) = stack.pop() {
        match e {
            Expr::Column(name) => {
                if !project_schema.fields.iter().any(|f| &f.name == name) {
                    return Err(BoltError::Sql(format!(
                        "HAVING references unknown column '{name}'"
                    )));
                }
            }
            Expr::Literal(_) => {}
            Expr::Binary { left, right, .. } => {
                stack.push(left);
                stack.push(right);
            }
            Expr::Alias(inner, _) => stack.push(inner),
            Expr::Unary { operand, .. } => stack.push(operand),
            Expr::Case {
                branches,
                else_branch,
            } => {
                for (when, then) in branches {
                    stack.push(when);
                    stack.push(then);
                }
                if let Some(e) = else_branch {
                    stack.push(e);
                }
            }
            Expr::Like { expr, .. } => stack.push(expr),
            Expr::Cast { expr, .. } => stack.push(expr),
            Expr::ScalarFn { args, .. } => {
                for a in args {
                    stack.push(a);
                }
            }
            Expr::Extract { expr, .. } | Expr::DateTrunc { expr, .. } => stack.push(expr),
            // A subquery references its own schema's columns, not HAVING's
            // post-aggregate projection schema, so do not walk into it. The
            // `InSubquery` probe is in HAVING's namespace, so check it.
            Expr::ScalarSubquery(_) => {}
            Expr::InSubquery { expr, .. } => stack.push(expr),
        }
    }
    Ok(())
}

/// Lower an uncorrelated subquery `query` into its own [`LogicalPlan`].
///
/// Pulls the subquery lowering context (provider + CTE scope) off `resolver`;
/// a `resolver` built without that context (e.g. ORDER BY lowering) rejects
/// subqueries with a clear message rather than panicking. Before lowering, the
/// query is checked for correlation against the outer scope's columns and
/// rejected if any outer column is referenced (see
/// [`crate::plan::subquery::reject_if_correlated`]).
fn lower_uncorrelated_subquery(
    query: &Query,
    resolver: &NameResolver<'_>,
    depth: usize,
) -> BoltResult<LogicalPlan> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    let ctx = resolver.ctx.ok_or_else(|| {
        BoltError::Sql(
            "subqueries are not supported in this position (only WHERE / SELECT)".into(),
        )
    })?;
    // Reject correlated subqueries before lowering. The outer scope's column
    // names let the detector name a precise offending reference.
    let outer_columns = resolver.outer_column_names();
    crate::plan::subquery::reject_if_correlated(query, &outer_columns, ctx.provider)?;
    // Lower the subquery as a standalone query against the same provider and
    // (inherited) CTE scope. It is uncorrelated, so it needs none of the outer
    // resolver's table scopes.
    plan_query(query, ctx.provider, ctx.ctes, depth + 1)
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
///
/// `depth` is the current recursion depth; returns Err if MAX_RECURSION_DEPTH is exceeded.
fn lower_expr(e: &SqlExpr, resolver: &NameResolver<'_>, depth: usize) -> BoltResult<Expr> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    match e {
        SqlExpr::Identifier(ident) => {
            // SQL-standard case folding (v0.5): unquoted column refs fold to
            // lowercase, quoted refs stay verbatim. The resolver and
            // downstream `Schema::index_of` apply a case-insensitive fallback
            // when the verbatim name is not present, so a folded `name`
            // still matches a schema field registered as `Name`.
            Ok(Expr::Column(ident_to_name(ident)))
        }
        SqlExpr::CompoundIdentifier(parts) => {
            // We support `table.col` and the schema-qualified `schema.table.col`
            // (or `alias`-qualified) form. The engine is single-catalog, so the
            // leading schema/catalog part of a three-segment reference carries
            // no meaning — the reference resolves by its trailing `table.col`
            // pair exactly as the two-part form does (and is rejected the same
            // way if that table/alias is not in scope). Four or more segments
            // have no namespace to collapse into and are rejected as "deeply
            // qualified".
            //
            // Cap the reflected fragment so adversarial input (very long
            // identifiers, deeply nested chains) can't flood logs. 200
            // chars is enough to identify the offending reference in
            // practice; anything past that gets an ellipsis.
            let display = || {
                let full = parts
                    .iter()
                    .map(|p| p.value.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                if full.chars().count() > 200 {
                    let truncated: String = full.chars().take(200).collect();
                    format!("{truncated}...")
                } else {
                    full
                }
            };
            match parts.len() {
                // `table.col` / `alias.col`. Per-part case folding via
                // ident_to_name: each segment is folded independently.
                // `T1.Name` and `t1.name` resolve to the same column;
                // `"T1"."Name"` (both quoted) is verbatim.
                2 => {
                    let qualifier = ident_to_name(&parts[0]);
                    let col = ident_to_name(&parts[1]);
                    let resolved = resolver.resolve_compound(&qualifier, &col)?;
                    Ok(Expr::Column(resolved))
                }
                // `schema.table.col`. Drop the leading single-catalog
                // qualifier and resolve the trailing `table.col` pair; an
                // unknown table/alias in the middle slot still produces the
                // standard "unknown table qualifier" error from
                // resolve_compound.
                3 => {
                    let qualifier = ident_to_name(&parts[1]);
                    let col = ident_to_name(&parts[2]);
                    let resolved = resolver.resolve_compound(&qualifier, &col)?;
                    Ok(Expr::Column(resolved))
                }
                _ => Err(BoltError::Sql(format!(
                    "unsupported: deeply qualified column reference '{}'",
                    display()
                ))),
            }
        }
        SqlExpr::Value(v) => lower_value(v),
        SqlExpr::Nested(inner) => lower_expr(inner, resolver, depth + 1),
        SqlExpr::BinaryOp { left, op, right } => {
            let lop = lower_binary_op(op)?;
            let l = lower_expr(left, resolver, depth + 1)?;
            let r = lower_expr(right, resolver, depth + 1)?;
            Ok(Expr::Binary {
                op: lop,
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        SqlExpr::UnaryOp { op, expr } => match op {
            UnaryOperator::Plus => lower_expr(expr, resolver, depth + 1),
            UnaryOperator::Minus => negate_expr(expr, resolver, depth + 1),
            // SQL `NOT <bool-expr>`. The operand must type-check to `Bool`
            // (enforced at the logical-plan layer in
            // `Expr::dtype_depth` — see the `UnaryOp::Not` arm there).
            UnaryOperator::Not => {
                let operand = lower_expr(expr, resolver, depth + 1)?;
                Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(operand),
                })
            }
            other => Err(BoltError::Sql(format!(
                "unsupported unary operator: {other:?}"
            ))),
        },
        // `<expr> IS NULL` and `<expr> IS NOT NULL`. We accept the syntax at
        // the planner; the physical-plan boundary rejects it cleanly with a
        // `BoltError::Plan` so callers get a useful message rather than the
        // executor being silently wrong. See `Expr::Unary` for the typing
        // contract (always Bool).
        SqlExpr::IsNull(inner) => {
            let operand = lower_expr(inner, resolver, depth + 1)?;
            Ok(Expr::Unary {
                op: UnaryOp::IsNull,
                operand: Box::new(operand),
            })
        }
        SqlExpr::IsNotNull(inner) => {
            let operand = lower_expr(inner, resolver, depth + 1)?;
            Ok(Expr::Unary {
                op: UnaryOp::IsNotNull,
                operand: Box::new(operand),
            })
        }
        // `<expr> [NOT] IN (v1, v2, ...)` — desugared into an OR/AND chain
        // of element-wise equalities/inequalities so existing comparison and
        // boolean codegen handles it without a new operator.
        SqlExpr::InList { expr, list, negated } => {
            lower_in_list(expr, list, *negated, resolver, depth + 1)
        }
        // Uncorrelated scalar subquery: `(SELECT max(y) FROM t2)`. Lower the
        // subquery to its own `LogicalPlan` (rejecting correlation) and wrap
        // it in `Expr::ScalarSubquery`. Type-checking (single output column)
        // happens later in `Expr::dtype`.
        SqlExpr::Subquery(query) => {
            let plan = lower_uncorrelated_subquery(query, resolver, depth + 1)?;
            Ok(Expr::ScalarSubquery(Box::new(plan)))
        }
        // Uncorrelated `x [NOT] IN (SELECT ...)`. The probe `expr` lowers
        // against the outer scope; the subquery lowers to its own plan.
        SqlExpr::InSubquery {
            expr,
            subquery,
            negated,
        } => {
            let probe = lower_expr(expr, resolver, depth + 1)?;
            let plan = lower_uncorrelated_subquery(subquery, resolver, depth + 1)?;
            Ok(Expr::InSubquery {
                expr: Box::new(probe),
                subquery: Box::new(plan),
                negated: *negated,
            })
        }
        // EXISTS / NOT EXISTS would need a dedicated boolean-existence node;
        // not yet supported. Reject cleanly rather than mis-lowering.
        SqlExpr::Exists { .. } => Err(BoltError::Sql(
            "unsupported: EXISTS subquery (only scalar and IN subqueries are supported)".into(),
        )),
        // `expr BETWEEN low AND high`  →  `(expr >= low) AND (expr <= high)`
        // `expr NOT BETWEEN low AND high`  →  `(expr <  low) OR  (expr >  high)`
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => {
            let operand = lower_expr(expr, resolver, depth + 1)?;
            let lo = lower_expr(low, resolver, depth + 1)?;
            let hi = lower_expr(high, resolver, depth + 1)?;
            if *negated {
                let lt_low = Expr::Binary {
                    op: BinaryOp::Lt,
                    left: Box::new(operand.clone()),
                    right: Box::new(lo),
                };
                let gt_high = Expr::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(operand),
                    right: Box::new(hi),
                };
                Ok(Expr::Binary {
                    op: BinaryOp::Or,
                    left: Box::new(lt_low),
                    right: Box::new(gt_high),
                })
            } else {
                let ge_low = Expr::Binary {
                    op: BinaryOp::GtEq,
                    left: Box::new(operand.clone()),
                    right: Box::new(lo),
                };
                let le_high = Expr::Binary {
                    op: BinaryOp::LtEq,
                    left: Box::new(operand),
                    right: Box::new(hi),
                };
                Ok(Expr::Binary {
                    op: BinaryOp::And,
                    left: Box::new(ge_low),
                    right: Box::new(le_high),
                })
            }
        }
        SqlExpr::Case {
            operand,
            conditions,
            results,
            else_result,
        } => lower_case(
            operand.as_deref(),
            conditions,
            results,
            else_result.as_deref(),
            resolver,
            depth + 1,
        ),
        SqlExpr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => {
            if *any {
                return Err(BoltError::Sql(
                    "unsupported: LIKE ANY (...)".into(),
                ));
            }
            // sqlparser hands us the ESCAPE clause as an `Option<String>`
            // (the literal text between the quotes). Standard SQL requires
            // exactly one character — reject anything else with a clear
            // message so the user sees the constraint up front.
            let escape: Option<char> = match escape_char.as_deref() {
                None => None,
                Some(s) => {
                    let mut iter = s.chars();
                    let first = iter.next().ok_or_else(|| {
                        BoltError::Sql(
                            "LIKE ESCAPE clause must be a single character, got an empty string"
                                .into(),
                        )
                    })?;
                    if iter.next().is_some() {
                        return Err(BoltError::Sql(format!(
                            "LIKE ESCAPE clause must be a single character, got {s:?}"
                        )));
                    }
                    Some(first)
                }
            };
            let pattern_str = match pattern.as_ref() {
                SqlExpr::Value(Value::SingleQuotedString(s)) => s.clone(),
                other => {
                    return Err(BoltError::Sql(format!(
                        "LIKE pattern must be a string literal constant, got: {other}"
                    )));
                }
            };
            let operand = lower_expr(expr, resolver, depth + 1)?;
            Ok(Expr::Like {
                expr: Box::new(operand),
                pattern: pattern_str,
                escape,
                negated: *negated,
            })
        }
        SqlExpr::Function(func) => {
            // Intercept COALESCE / NULLIF before the catch-all rejection.
            // Both desugar to CASE so the existing CASE codegen path (and
            // type-checker) handles them with no new IR node.
            if let Some(name) = scalar_function_name(func) {
                let upper = name.to_ascii_uppercase();
                match upper.as_str() {
                    "COALESCE" => {
                        let args = collect_scalar_function_args(func, &upper)?;
                        return lower_coalesce(&args, resolver, depth + 1);
                    }
                    "NULLIF" => {
                        let args = collect_scalar_function_args(func, &upper)?;
                        return lower_nullif(&args, resolver, depth + 1);
                    }
                    _ => {}
                }
            }
            // Then intercept the named string scalar functions UPPER / LOWER
            // / LENGTH / CONCAT. Aggregates are NOT routed here — callers
            // split those off via `try_aggregate` ahead of `lower_expr` — so
            // any `Function` we see here is genuinely a scalar call.
            if let Some(expr) = try_string_scalar_fn(func, resolver, depth)? {
                Ok(expr)
            } else {
                Err(BoltError::Sql(format!(
                    "scalar function calls are not supported: {}",
                    func.name
                )))
            }
        }
        // `CAST(<expr> AS <type>)`. v0.5 accepts the standard SQL spelling
        // and the Postgres `expr::type` shortcut (both surface as
        // `CastKind::Cast` / `CastKind::DoubleColon`). `TRY_CAST` and
        // `SAFE_CAST` carry NULL-on-failure semantics the planner can't
        // honour yet, so we reject them with a clear message rather than
        // silently treating them as a plain cast.
        SqlExpr::Cast {
            kind,
            expr,
            data_type,
            format,
        } => {
            if format.is_some() {
                return Err(BoltError::Sql(
                    "CAST with FORMAT clause not supported".into(),
                ));
            }
            match kind {
                CastKind::Cast | CastKind::DoubleColon => {}
                CastKind::TryCast => {
                    return Err(BoltError::Sql(
                        "TRY_CAST not supported; use CAST".into(),
                    ));
                }
                CastKind::SafeCast => {
                    return Err(BoltError::Sql(
                        "SAFE_CAST not supported; use CAST".into(),
                    ));
                }
            }
            let target = lower_cast_data_type(data_type)?;
            let inner = lower_expr(expr, resolver, depth + 1)?;
            Ok(Expr::Cast {
                expr: Box::new(inner),
                target,
            })
        }
        // `SUBSTRING(s FROM i [FOR n])` / `SUBSTRING(s, i [, n])`: sqlparser
        // surfaces both syntaxes as `SqlExpr::Substring`, not as a generic
        // `Function` call. Lower into the shared `Expr::ScalarFn` shape so
        // the physical-plan boundary can reject every string scalar function
        // uniformly. Type-checking (`Utf8`, `Int64`, `Int64`) lives in
        // [`crate::plan::logical_plan::scalar_fn_dtype`].
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            let s = lower_expr(expr, resolver, depth + 1)?;
            let start = match substring_from {
                Some(e) => lower_expr(e, resolver, depth + 1)?,
                None => {
                    return Err(BoltError::Sql(
                        "SUBSTRING requires a start position (use SUBSTRING(s FROM i [FOR n]) \
                         or SUBSTRING(s, i [, n]))"
                            .into(),
                    ));
                }
            };
            let mut args = vec![s, start];
            if let Some(e) = substring_for {
                args.push(lower_expr(e, resolver, depth + 1)?);
            }
            Ok(Expr::ScalarFn {
                kind: ScalarFnKind::Substring,
                args,
            })
        }
        // `TRIM([BOTH|LEADING|TRAILING] [chars] FROM s)` and `TRIM(s)`:
        // sqlparser surfaces these as the dedicated `SqlExpr::Trim` variant
        // (not a `Function`). We map the trim side to one of the three
        // `ScalarFnKind::Trim*` variants (default BOTH) and lower the source
        // plus optional trim-characters string into the `ScalarFn` args. The
        // comma form `TRIM(s, 'chars')` populates `trim_characters`; we accept
        // a single trim-string there for parity with the `... FROM s` form.
        SqlExpr::Trim {
            expr,
            trim_where,
            trim_what,
            trim_characters,
        } => {
            use sqlparser::ast::TrimWhereField;
            let kind = match trim_where {
                None | Some(TrimWhereField::Both) => ScalarFnKind::TrimBoth,
                Some(TrimWhereField::Leading) => ScalarFnKind::TrimLeading,
                Some(TrimWhereField::Trailing) => ScalarFnKind::TrimTrailing,
            };
            let s = lower_expr(expr, resolver, depth + 1)?;
            let mut args = vec![s];
            // `TRIM(chars FROM s)` form.
            if let Some(chars) = trim_what {
                args.push(lower_expr(chars, resolver, depth + 1)?);
            }
            // `TRIM(s, 'chars')` (BigQuery/Snowflake) form: a list of trim
            // strings. We support exactly one; more than one is rejected.
            if let Some(chars_list) = trim_characters {
                if trim_what.is_some() {
                    return Err(BoltError::Sql(
                        "TRIM: cannot combine `chars FROM s` with the comma `(s, chars)` form"
                            .into(),
                    ));
                }
                if chars_list.len() != 1 {
                    return Err(BoltError::Sql(format!(
                        "TRIM: expected exactly one trim-characters argument, got {}",
                        chars_list.len()
                    )));
                }
                args.push(lower_expr(&chars_list[0], resolver, depth + 1)?);
            }
            Ok(Expr::ScalarFn { kind, args })
        }
        // v0.6 / M4: typed-string literals — `DATE '2024-01-01'`,
        // `TIMESTAMP '2024-01-01 00:00:00'`. Other typed strings (TIME,
        // INTERVAL, ...) are not yet supported and surface the generic
        // "unsupported expression" error below.
        SqlExpr::TypedString { data_type, value } => {
            use sqlparser::ast::DataType as SqlDataType;
            match data_type {
                SqlDataType::Date => parse_date_literal(value),
                SqlDataType::Timestamp(_precision, _tz) => parse_timestamp_literal(value),
                other => Err(BoltError::Sql(format!(
                    "unsupported typed-string literal: {other} '{value}'"
                ))),
            }
        }
        other => Err(BoltError::Sql(format!(
            "unsupported expression: {other}"
        ))),
    }
}

/// Return the single-identifier function name if `func` is a bare
/// `NAME(args)` call with none of the SQL extensions (no OVER, no FILTER,
/// no WITHIN GROUP, etc.) the COALESCE / NULLIF desugar path is willing
/// to accept. Returns `None` for multi-part names or when any extension
/// clause is present — those fall through to the catch-all
/// "scalar function calls are not supported" rejection.
fn scalar_function_name(func: &sqlparser::ast::Function) -> Option<&str> {
    if func.name.0.len() != 1 {
        return None;
    }
    if func.over.is_some()
        || func.filter.is_some()
        || func.null_treatment.is_some()
        || !func.within_group.is_empty()
        || !matches!(func.parameters, FunctionArguments::None)
    {
        return None;
    }
    Some(func.name.0[0].value.as_str())
}

/// Pull out the unnamed positional argument expressions of a scalar
/// function call. Rejects wildcards, qualified wildcards, and named
/// arguments — none of which are meaningful for `COALESCE` / `NULLIF`.
/// `kind` is the upper-cased function name used in error messages.
fn collect_scalar_function_args<'a>(
    func: &'a sqlparser::ast::Function,
    kind: &str,
) -> BoltResult<Vec<&'a SqlExpr>> {
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
    let mut out: Vec<&SqlExpr> = Vec::with_capacity(arg_list.args.len());
    for arg in &arg_list.args {
        match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => out.push(e),
            FunctionArg::Unnamed(FunctionArgExpr::Wildcard) => {
                return Err(BoltError::Sql(format!(
                    "{kind} does not accept a `*` argument"
                )));
            }
            FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(_)) => {
                return Err(BoltError::Sql(format!(
                    "{kind} does not accept a qualified wildcard argument"
                )));
            }
            FunctionArg::Named { .. } => {
                return Err(BoltError::Sql(format!(
                    "unsupported: named argument to {kind}"
                )));
            }
        }
    }
    Ok(out)
}

/// Desugar `COALESCE(a, b, c, ..., last)` into the equivalent CASE:
///
/// ```text
/// CASE
///   WHEN a IS NOT NULL THEN a
///   WHEN b IS NOT NULL THEN b
///   ...
///   ELSE last
/// END
/// ```
///
/// Edge cases:
///   * `COALESCE()` (zero args)   → error.
///   * `COALESCE(a)` (one arg)    → lowers to `a` (no CASE wrapping); per
///     SQL semantics the value of a one-arg COALESCE is the argument
///     itself, and we want the IR to reflect that so downstream
///     optimisation isn't blocked on a trivially-collapsible CASE.
///   * `COALESCE(a, b)` (two args) → one WHEN/THEN branch + ELSE.
fn lower_coalesce(
    args: &[&SqlExpr],
    resolver: &NameResolver<'_>,
    depth: usize,
) -> BoltResult<Expr> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    if args.is_empty() {
        return Err(BoltError::Sql(
            "COALESCE requires at least one argument".into(),
        ));
    }
    if args.len() == 1 {
        // Single-arg COALESCE is identity. Lower the operand directly so
        // the IR doesn't carry a vestigial single-branch CASE.
        return lower_expr(args[0], resolver, depth + 1);
    }
    // Lower every argument once up front. We need each non-last argument
    // twice (once in the `IS NOT NULL` test, once as the THEN value), so
    // build the lowered list and clone the appropriate cells when
    // assembling the CASE.
    let mut lowered: Vec<Expr> = Vec::with_capacity(args.len());
    for a in args {
        lowered.push(lower_expr(a, resolver, depth + 1)?);
    }
    // Last argument becomes the ELSE. Everything before it becomes a
    // `WHEN arg IS NOT NULL THEN arg` branch in source order.
    let else_expr = lowered.pop().expect("len >= 2 checked above");
    let mut branches: Vec<(Expr, Expr)> = Vec::with_capacity(lowered.len());
    for arg in lowered {
        let cond = Expr::Unary {
            op: UnaryOp::IsNotNull,
            operand: Box::new(arg.clone()),
        };
        branches.push((cond, arg));
    }
    Ok(Expr::Case {
        branches,
        else_branch: Some(Box::new(else_expr)),
    })
}

/// Desugar `NULLIF(a, b)` into the equivalent CASE:
///
/// ```text
/// CASE WHEN a = b THEN NULL ELSE a END
/// ```
///
/// SQL `NULLIF` is strictly binary; any other arity is rejected at parse
/// time so the user gets a clean error instead of a confusing type-check
/// failure later.
fn lower_nullif(
    args: &[&SqlExpr],
    resolver: &NameResolver<'_>,
    depth: usize,
) -> BoltResult<Expr> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    if args.len() != 2 {
        return Err(BoltError::Sql(format!(
            "NULLIF expects exactly 2 arguments, got {}",
            args.len()
        )));
    }
    let a = lower_expr(args[0], resolver, depth + 1)?;
    let b = lower_expr(args[1], resolver, depth + 1)?;
    let cond = Expr::Binary {
        op: BinaryOp::Eq,
        left: Box::new(a.clone()),
        right: Box::new(b),
    };
    Ok(Expr::Case {
        branches: vec![(cond, Expr::Literal(Literal::Null))],
        else_branch: Some(Box::new(a)),
    })
}

/// Maximum number of values accepted on the right-hand side of a SQL
/// `IN (...)` list operator. Lists at or under this bound are desugared
/// to a balanced OR/AND chain of element-wise comparisons.
const MAX_IN_LIST_VALUES: usize = 64;

/// Desugar SQL `<expr> [NOT] IN (v1, v2, ..., vN)` into the equivalent
/// chain of element-wise comparisons:
///
///   * `IN`     → `(expr = v1) OR  (expr = v2) OR  ... OR  (expr = vN)`
///   * `NOT IN` → `(expr <> v1) AND (expr <> v2) AND ... AND (expr <> vN)`
fn lower_in_list(
    expr: &SqlExpr,
    list: &[SqlExpr],
    negated: bool,
    resolver: &NameResolver<'_>,
    depth: usize,
) -> BoltResult<Expr> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    if list.len() > MAX_IN_LIST_VALUES {
        return Err(BoltError::Sql(format!(
            "IN with > {MAX_IN_LIST_VALUES} values not yet supported; use a JOIN \
             against a derived table (got {} values)",
            list.len()
        )));
    }
    if list.is_empty() {
        return Ok(Expr::Literal(Literal::Bool(negated)));
    }
    let lowered_expr = lower_expr(expr, resolver, depth + 1)?;
    let (cmp_op, combine_op) = if negated {
        (BinaryOp::NotEq, BinaryOp::And)
    } else {
        (BinaryOp::Eq, BinaryOp::Or)
    };
    let mut acc: Option<Expr> = None;
    for item in list {
        let item_lowered = lower_expr(item, resolver, depth + 1)?;
        let cmp = Expr::Binary {
            op: cmp_op,
            left: Box::new(lowered_expr.clone()),
            right: Box::new(item_lowered),
        };
        acc = Some(match acc {
            None => cmp,
            Some(prev) => Expr::Binary {
                op: combine_op,
                left: Box::new(prev),
                right: Box::new(cmp),
            },
        });
    }
    Ok(acc.expect("non-empty IN list guarantees at least one chain element"))
}

/// Lower a SQL `CASE` expression — both the plain form (no operand) and
/// the simple form (with operand). The simple form is desugared into the
/// plain form by rewriting each WHEN `value` into `operand = value`.
fn lower_case(
    operand: Option<&SqlExpr>,
    conditions: &[SqlExpr],
    results: &[SqlExpr],
    else_result: Option<&SqlExpr>,
    resolver: &NameResolver<'_>,
    depth: usize,
) -> BoltResult<Expr> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    if conditions.len() != results.len() {
        return Err(BoltError::Sql(format!(
            "CASE expression has {} conditions and {} results; they must match",
            conditions.len(),
            results.len(),
        )));
    }
    if conditions.is_empty() {
        return Err(BoltError::Sql(
            "CASE expression requires at least one WHEN/THEN branch".into(),
        ));
    }
    let lowered_operand = match operand {
        Some(e) => Some(lower_expr(e, resolver, depth + 1)?),
        None => None,
    };
    let mut branches: Vec<(Expr, Expr)> = Vec::with_capacity(conditions.len());
    for (when_sql, then_sql) in conditions.iter().zip(results.iter()) {
        let cond_expr = match &lowered_operand {
            Some(op_expr) => {
                let v = lower_expr(when_sql, resolver, depth + 1)?;
                Expr::Binary {
                    op: BinaryOp::Eq,
                    left: Box::new(op_expr.clone()),
                    right: Box::new(v),
                }
            }
            None => lower_expr(when_sql, resolver, depth + 1)?,
        };
        let then_expr = lower_expr(then_sql, resolver, depth + 1)?;
        branches.push((cond_expr, then_expr));
    }
    let else_expr = match else_result {
        Some(e) => Some(Box::new(lower_expr(e, resolver, depth + 1)?)),
        None => None,
    };
    Ok(Expr::Case {
        branches,
        else_branch: else_expr,
    })
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

/// Map a parsed `sqlparser::ast::DataType` into the engine's internal
/// [`DataType`] for the v0.5 CAST surface.
///
/// Only the primitive types the executor will plausibly lower in v0.6 are
/// accepted — anything else (CHAR(n), DECIMAL, DATE/TIME, structured types,
/// etc.) surfaces a clear `BoltError::Sql` so the user knows the parser
/// understood the type name but the engine doesn't support converting to it
/// yet.
///
/// Recognised aliases follow common SQL spellings:
///   * `INT`, `INTEGER`, `INT4`, `INT32`   -> [`DataType::Int32`]
///   * `BIGINT`, `INT8`, `INT64`           -> [`DataType::Int64`]
///   * `REAL`, `FLOAT4`, `FLOAT32`         -> [`DataType::Float32`]
///   * `DOUBLE`, `DOUBLE PRECISION`,
///     `FLOAT8`, `FLOAT64`                 -> [`DataType::Float64`]
///   * `BOOL`, `BOOLEAN`                   -> [`DataType::Bool`]
fn lower_cast_data_type(t: &SqlDataType) -> BoltResult<DataType> {
    Ok(match t {
        SqlDataType::Int(_)
        | SqlDataType::Integer(_)
        | SqlDataType::Int4(_)
        | SqlDataType::Int32 => DataType::Int32,
        SqlDataType::BigInt(_) | SqlDataType::Int8(_) | SqlDataType::Int64 => DataType::Int64,
        SqlDataType::Real | SqlDataType::Float4 | SqlDataType::Float32 => DataType::Float32,
        SqlDataType::Double
        | SqlDataType::DoublePrecision
        | SqlDataType::Float8
        | SqlDataType::Float64 => DataType::Float64,
        SqlDataType::Bool | SqlDataType::Boolean => DataType::Bool,
        other => {
            return Err(BoltError::Sql(format!(
                "CAST target type not supported: {other}"
            )));
        }
    })
}

/// Parse a SQL `DATE 'YYYY-MM-DD'` literal into a [`Literal::Date32`]
/// counting days since the Unix epoch (1970-01-01). Rejects malformed
/// strings, out-of-range months/days, and dates outside the i32 day range.
///
/// The parser is intentionally minimal — exactly `YYYY-MM-DD`, no time
/// component, no timezone. A SQL `DATE` literal in the spec carries no
/// time component; if a caller wants time-of-day they should use
/// `TIMESTAMP`. This routine never panics: every failure surfaces a
/// `BoltError::Sql` with the offending text.
fn parse_date_literal(s: &str) -> BoltResult<Expr> {
    let s = s.trim();
    let (y, m, d) = parse_ymd(s).ok_or_else(|| {
        BoltError::Sql(format!(
            "DATE literal must be 'YYYY-MM-DD', got '{s}'"
        ))
    })?;
    let days = days_since_epoch(y, m, d).ok_or_else(|| {
        BoltError::Sql(format!(
            "DATE literal '{s}' is out of the supported range"
        ))
    })?;
    Ok(Expr::Literal(Literal::Date32(days)))
}

/// Parse a SQL `TIMESTAMP 'YYYY-MM-DD HH:MM:SS[.fff]'` literal into a
/// [`Literal::Timestamp`] of `TimeUnit::Nanosecond` resolution (matches
/// Arrow's default `TimestampNanosecondArray`). No timezone is currently
/// recognised here — the literal is interpreted as naive (TZ = `None`).
///
/// As with [`parse_date_literal`] this is intentionally minimal: the
/// optional fractional-seconds tail accepts 1..=9 digits (anything past 9
/// is truncated to nanoseconds), the date-time separator may be `' '` or
/// `'T'`. Anything else surfaces a `BoltError::Sql`.
fn parse_timestamp_literal(s: &str) -> BoltResult<Expr> {
    let s = s.trim();
    let (date_part, time_part) = split_date_time(s).ok_or_else(|| {
        BoltError::Sql(format!(
            "TIMESTAMP literal must be 'YYYY-MM-DD HH:MM:SS[.fff]', got '{s}'"
        ))
    })?;
    let (y, m, d) = parse_ymd(date_part).ok_or_else(|| {
        BoltError::Sql(format!(
            "TIMESTAMP literal '{s}' has malformed date component"
        ))
    })?;
    let (hh, mm, ss, nanos) = parse_hms_fraction(time_part).ok_or_else(|| {
        BoltError::Sql(format!(
            "TIMESTAMP literal '{s}' has malformed time component"
        ))
    })?;
    let days = days_since_epoch(y, m, d).ok_or_else(|| {
        BoltError::Sql(format!("TIMESTAMP literal '{s}' is out of range"))
    })?;
    let seconds_in_day = (hh as i64) * 3600 + (mm as i64) * 60 + (ss as i64);
    let total_seconds = (days as i64)
        .checked_mul(86_400)
        .and_then(|d| d.checked_add(seconds_in_day))
        .ok_or_else(|| {
            BoltError::Sql(format!("TIMESTAMP literal '{s}' overflows i64 seconds"))
        })?;
    let ticks = total_seconds
        .checked_mul(1_000_000_000)
        .and_then(|t| t.checked_add(nanos as i64))
        .ok_or_else(|| {
            BoltError::Sql(format!(
                "TIMESTAMP literal '{s}' overflows i64 nanoseconds since epoch"
            ))
        })?;
    Ok(Expr::Literal(Literal::Timestamp(
        ticks,
        TimeUnit::Nanosecond,
        None,
    )))
}

/// Parse a strict `YYYY-MM-DD` string into `(year, month, day)`. Returns
/// `None` on any structural mismatch. Years are accepted as any signed
/// `i32`-range integer (so `-0001-01-01` and `9999-12-31` both parse).
fn parse_ymd(s: &str) -> Option<(i32, u32, u32)> {
    // Allow a leading `-` for BC dates; the rest must be exactly digits and
    // dashes in `Y-M-D` form. We don't fix-width the year so e.g.
    // `2024-1-1` and `2024-01-01` both parse — DuckDB accepts both.
    // For BC dates `-0001-01-01` the first split yields an empty string and
    // the year is the second.
    let (year_sign, body) = if let Some(rest) = s.strip_prefix('-') {
        (-1i32, rest)
    } else {
        (1i32, s)
    };
    let parts: Vec<&str> = body.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let y_abs: i32 = parts[0].parse().ok()?;
    let m: u32 = parts[1].parse().ok()?;
    let d: u32 = parts[2].parse().ok()?;
    let y = y_abs.checked_mul(year_sign)?;
    if !(1..=12).contains(&m) || d < 1 {
        return None;
    }
    // Bound the accepted year range. `days_since_epoch` only carries an
    // `i32` day count anyway (roughly +/- 5.8M years), but we clamp to a
    // conservative window so the intermediate arithmetic in
    // `days_since_epoch` (and `era * 146_097`) can never wrap or panic in
    // debug builds for extreme but structurally-valid years. Dates outside
    // the `Date32` day range are still caught precisely by
    // `days_since_epoch` returning `None`.
    if !(MIN_SUPPORTED_YEAR..=MAX_SUPPORTED_YEAR).contains(&y) {
        return None;
    }
    // Validate the day against the real length of this month, accounting
    // for Gregorian leap years, so e.g. `2024-02-31`, `2024-04-31`, and
    // `2023-02-29` are rejected rather than silently producing a bogus
    // Date32.
    if d > days_in_month(y, m) {
        return None;
    }
    Some((y, m, d))
}

/// Smallest/largest proleptic-Gregorian year `parse_ymd` will accept.
/// Far wider than any `Date32`-representable date, but narrow enough that
/// the `era * 146_097` term in `days_since_epoch` stays well inside `i64`
/// and cannot overflow/panic in debug builds. Anything inside this window
/// that still falls outside the `i32` day range is rejected downstream by
/// `days_since_epoch`.
const MIN_SUPPORTED_YEAR: i32 = -999_999;
const MAX_SUPPORTED_YEAR: i32 = 999_999;

/// True for a proleptic-Gregorian leap year.
fn is_leap_year(y: i32) -> bool {
    (y % 4 == 0) && (y % 100 != 0 || y % 400 == 0)
}

/// Number of days in month `m` (1..=12) of year `y`. Caller must pass a
/// month already validated to be in `1..=12`.
fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(y) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Split a timestamp string into `(date, time)` on the first `' '` or `'T'`.
fn split_date_time(s: &str) -> Option<(&str, &str)> {
    if let Some(idx) = s.find([' ', 'T']) {
        let (a, b) = s.split_at(idx);
        // `b` includes the separator; drop it.
        Some((a, &b[1..]))
    } else {
        None
    }
}

/// Parse `HH:MM:SS[.fff]` (fractional seconds optional). Returns
/// `(hour, minute, second, nanos_in_second)`. Fractional digits past 9
/// are silently truncated; fewer than 9 are left-padded with zeros.
fn parse_hms_fraction(s: &str) -> Option<(u32, u32, u32, u32)> {
    let (hms, frac) = match s.find('.') {
        Some(i) => (&s[..i], &s[i + 1..]),
        None => (s, ""),
    };
    let parts: Vec<&str> = hms.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let hh: u32 = parts[0].parse().ok()?;
    let mm: u32 = parts[1].parse().ok()?;
    let ss: u32 = parts[2].parse().ok()?;
    // Reject `ss == 60`: we do not model leap seconds, and accepting 60
    // would silently roll over into the next minute when the literal is
    // converted to seconds-since-epoch. Strict rejection avoids surprising
    // round-trips. (No existing test pins the 60-accepting behaviour.)
    if hh > 23 || mm > 59 || ss > 59 {
        return None;
    }
    let nanos: u32 = if frac.is_empty() {
        0
    } else {
        if !frac.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let truncated = &frac[..frac.len().min(9)];
        // Left-pad with zeros after the digits to reach 9 (e.g. ".5" -> "500000000").
        let mut buf = String::with_capacity(9);
        buf.push_str(truncated);
        while buf.len() < 9 {
            buf.push('0');
        }
        buf.parse().ok()?
    };
    Some((hh, mm, ss, nanos))
}

/// Convert a `(year, month, day)` Gregorian calendar date into days since
/// the Unix epoch (1970-01-01), returning `None` if the value is outside
/// the `i32` day range that `DataType::Date32` carries. Uses Howard
/// Hinnant's `days_from_civil` algorithm (public-domain, branch-free).
fn days_since_epoch(y: i32, m: u32, d: u32) -> Option<i32> {
    // Hinnant's algorithm. See
    // https://howardhinnant.github.io/date_algorithms.html#days_from_civil
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe: u32 = (y - era * 400) as u32; // [0, 399]
    let doy: u32 = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe: u32 = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days: i64 = (era as i64) * 146_097 + (doe as i64) - 719_468;
    i32::try_from(days).ok()
}

#[cfg(test)]
mod date_validation_tests {
    //! Pure host-side tests for calendar validation in `parse_ymd` /
    //! `days_since_epoch` (PL-H1). No GPU involved.
    use super::*;

    #[test]
    fn rejects_impossible_days() {
        // 31 days in months that only have 30, and Feb overflow.
        assert!(parse_ymd("2024-02-31").is_none(), "Feb 31 must be rejected");
        assert!(parse_ymd("2024-04-31").is_none(), "Apr 31 must be rejected");
        assert!(parse_ymd("2024-06-31").is_none(), "Jun 31 must be rejected");
        // Feb 29 in a non-leap year.
        assert!(
            parse_ymd("2023-02-29").is_none(),
            "Feb 29 on a non-leap year must be rejected"
        );
        // Day zero is also invalid.
        assert!(parse_ymd("2024-01-00").is_none(), "day 0 must be rejected");
    }

    #[test]
    fn accepts_leap_day_on_leap_year() {
        // 2024 is divisible by 4 and not by 100 -> leap.
        assert_eq!(parse_ymd("2024-02-29"), Some((2024, 2, 29)));
        // 2000 is divisible by 400 -> leap.
        assert_eq!(parse_ymd("2000-02-29"), Some((2000, 2, 29)));
        // 1900 is divisible by 100 but not 400 -> NOT leap.
        assert!(
            parse_ymd("1900-02-29").is_none(),
            "1900 is not a leap year"
        );
    }

    #[test]
    fn valid_dates_pass_and_round_trip() {
        assert_eq!(parse_ymd("2024-04-30"), Some((2024, 4, 30)));
        assert_eq!(parse_ymd("2024-12-31"), Some((2024, 12, 31)));
        // The Unix epoch is day 0; one day later is day 1.
        assert_eq!(days_since_epoch(1970, 1, 1), Some(0));
        assert_eq!(days_since_epoch(1970, 1, 2), Some(1));
        assert_eq!(days_since_epoch(1969, 12, 31), Some(-1));
        // A known reference value: 2000-01-01 is 10957 days after epoch.
        assert_eq!(days_since_epoch(2000, 1, 1), Some(10_957));
    }

    #[test]
    fn year_range_is_bounded_without_panic() {
        // Inside the accepted year window: structurally fine and, since the
        // day count for year 999_999 (~365M days) still fits in i32, it yields
        // a valid `Some(_)` WITHOUT panicking (overflowing the i32 day range
        // would require a year past ~5.9M, which the parse step rejects first).
        assert_eq!(parse_ymd("999999-12-31"), Some((999_999, 12, 31)));
        assert!(
            days_since_epoch(999_999, 12, 31).is_some(),
            "extreme-but-bounded year must produce a valid day count without panicking"
        );
        // Beyond the accepted window: rejected at the parse step.
        assert!(
            parse_ymd("1000000-01-01").is_none(),
            "year past MAX_SUPPORTED_YEAR must be rejected"
        );
        // The boundary years themselves do not panic.
        assert_eq!(
            parse_ymd("-999999-01-01"),
            Some((MIN_SUPPORTED_YEAR, 1, 1))
        );
    }

    #[test]
    fn parse_date_literal_rejects_bad_calendar_dates() {
        assert!(parse_date_literal("2024-02-31").is_err());
        assert!(parse_date_literal("2023-02-29").is_err());
        assert!(parse_date_literal("2024-02-29").is_ok());
        assert!(parse_date_literal("2024-04-30").is_ok());
    }

    #[test]
    fn rejects_second_value_sixty() {
        // Leap second `:60` is rejected for strictness.
        assert!(parse_hms_fraction("00:00:60").is_none());
        assert!(parse_hms_fraction("23:59:59").is_some());
        assert!(
            parse_timestamp_literal("2024-01-01 00:00:60").is_err(),
            ":60 seconds must be rejected"
        );
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
///
/// `depth` is the current recursion depth; returns Err if MAX_RECURSION_DEPTH is exceeded.
fn negate_expr(e: &SqlExpr, resolver: &NameResolver<'_>, depth: usize) -> BoltResult<Expr> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
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
    let inner = lower_expr(e, resolver, depth + 1)?;
    Ok(Expr::Binary {
        op: BinaryOp::Sub,
        left: Box::new(Expr::Literal(Literal::Int64(0))),
        right: Box::new(inner),
    })
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
        BinaryOperator::StringConcat => BinaryOp::Concat,
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
mod parse_error_span_tests {
    //! v0.6 / M5: unit tests for `parse_error_to_bolt_error` and the
    //! supporting line/column → byte-offset helpers. We deliberately
    //! exercise the pure helpers in isolation so the location-extraction
    //! rule stays decoupled from the rest of the frontend pipeline; a
    //! future sqlparser upgrade that changes the message shape can
    //! re-target these tests without rewiring `parse_uncached`.
    use super::*;

    #[test]
    fn extract_location_suffix_basic_case() {
        // Shape mirrors sqlparser's `Location::Display`: leading space,
        // then `"at Line: <L>, Column: <C>"`.
        let s = "sql parser error: Expected: end of statement, found: FOO at Line: 1, Column: 14";
        let (msg, line, col) =
            extract_location_suffix(s).expect("recognised sqlparser location suffix");
        assert_eq!(
            msg,
            "sql parser error: Expected: end of statement, found: FOO"
        );
        assert_eq!(line, 1);
        assert_eq!(col, 14);
    }

    #[test]
    fn extract_location_suffix_missing_returns_none() {
        // Tokenizer errors don't carry a location in sqlparser 0.52, so
        // the helper must gracefully signal "no span".
        assert!(extract_location_suffix("sql parser error: recursion limit exceeded").is_none());
        assert!(extract_location_suffix("").is_none());
    }

    #[test]
    fn line_column_to_byte_offset_ascii_single_line() {
        // "SELECT * FROM"  →  col 8 (1-based) is the '*'  →  byte 7.
        let sql = "SELECT * FROM t";
        assert_eq!(line_column_to_byte_offset(sql, 1, 8), Some(7));
        // col 1 is the leading 'S'.
        assert_eq!(line_column_to_byte_offset(sql, 1, 1), Some(0));
    }

    #[test]
    fn line_column_to_byte_offset_multi_line() {
        // Each newline advances the line counter; columns reset to 1.
        let sql = "SELECT a\nFROM t\nWHERE x = 1";
        // Line 2, col 1 should land on 'F' of "FROM".
        let off = line_column_to_byte_offset(sql, 2, 1).expect("line 2 reachable");
        assert_eq!(&sql[off..off + 4], "FROM");
        // Line 3, col 7 should land on 'x'.
        let off = line_column_to_byte_offset(sql, 3, 7).expect("line 3 reachable");
        assert_eq!(&sql[off..off + 1], "x");
    }

    #[test]
    fn line_column_to_byte_offset_handles_multibyte_chars() {
        // The Greek 'α' is two bytes in UTF-8; the column counter must
        // tick by characters, not by bytes, when computing offsets.
        // "SELECT α" is 7 chars / 8 bytes (the α contributing two bytes).
        let sql = "SELECT αβ FROM t";
        // col 8 (1-based) lands on the 8th character — 'α' itself (the
        // space before it is char 7). The byte offset is 7 (one byte
        // per ASCII char up to and including the space).
        let off = line_column_to_byte_offset(sql, 1, 8).expect("multibyte column reachable");
        assert_eq!(off, 7);
        let next_char = sql[off..].chars().next().expect("at least one char left");
        assert_eq!(next_char, 'α');
        // col 9 lands on 'β'; α (2 bytes) pushes the byte offset to 9.
        let off = line_column_to_byte_offset(sql, 1, 9).expect("col 9 reachable");
        assert_eq!(off, 9);
        let next_char = sql[off..].chars().next().expect("char at col 9");
        assert_eq!(next_char, 'β');
    }

    #[test]
    fn line_column_to_byte_offset_out_of_range_returns_none() {
        let sql = "SELECT 1";
        // No line 99 in an 8-byte input.
        assert!(line_column_to_byte_offset(sql, 99, 1).is_none());
        // Line/column of zero are nonsensical under sqlparser's
        // 1-based convention.
        assert!(line_column_to_byte_offset(sql, 0, 1).is_none());
        assert!(line_column_to_byte_offset(sql, 1, 0).is_none());
    }

    #[test]
    fn parse_error_to_bolt_error_with_location_produces_sql_with_span() {
        // The exact PE construction here mirrors the message format
        // sqlparser would produce for a single-line query. We can't
        // easily provoke a real error through `Parser::parse_sql`
        // with a known canonical message text, so we synthesise the
        // shape directly.
        let sql = "SELECT FROM t";
        let pe = ParserError::ParserError(
            "Expected: an expression:, found: FROM at Line: 1, Column: 8".into(),
        );
        let be = parse_error_to_bolt_error(pe, sql);
        match &be {
            BoltError::SqlWithSpan { msg, span } => {
                assert!(
                    msg.contains("Expected: an expression"),
                    "msg should still carry the diagnostic text, got: {msg}"
                );
                // sqlparser column 8 (1-based) → byte 7 → 'F' of "FROM".
                assert_eq!(span.start, 7);
                assert_eq!(span.end, 7, "zero-width span (position-only)");
                assert_eq!(&sql[span.start..span.start + 4], "FROM");
            }
            other => panic!("expected SqlWithSpan, got {other:?}"),
        }
        // The new accessor agrees with the variant pattern-match.
        assert_eq!(be.span(), Some(7..7));
    }

    #[test]
    fn parse_error_to_bolt_error_without_location_falls_back_to_sql() {
        // `RecursionLimitExceeded` renders without any location suffix —
        // the helper must fall back to the legacy `Sql` shape rather
        // than fabricate a span.
        let sql = "SELECT 1";
        let be = parse_error_to_bolt_error(ParserError::RecursionLimitExceeded, sql);
        assert!(matches!(be, BoltError::Sql(_)));
        assert_eq!(be.span(), None);
    }

    #[test]
    fn parse_error_to_bolt_error_end_to_end_via_parser() {
        // End-to-end check: run a deliberately-broken SQL through the
        // real sqlparser-driven `parse_uncached` and confirm we get a
        // span-aware error out the other side. This is the only test
        // here that depends on sqlparser's actual error-formatting
        // behaviour; if a future upgrade changes the message shape and
        // breaks the regex above, this test is the canary.
        let p = MemTableProvider::new();
        // The trailing comma is a syntax error sqlparser reports with
        // a location attached.
        let err = parse_uncached("SELECT a, FROM t", &p).expect_err("parse must fail");
        let span = err.span();
        assert!(
            span.is_some(),
            "expected SqlWithSpan from a real parse error, got: {err:?}"
        );
        // The span must point somewhere inside the SQL (within bounds).
        let span = span.unwrap();
        assert!(
            span.start <= "SELECT a, FROM t".len(),
            "span start out of input bounds: {span:?}"
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
    fn non_equi_join_lowers_to_join_with_filter() {
        // v0.6: non-equi predicates no longer error at the planner. They
        // route through the residual `filter` slot on `LogicalPlan::Join`,
        // and the executor switches to the nested-loop fallback.
        let plan = parse(
            "SELECT * FROM t1 INNER JOIN t2 ON t1.a > t2.a",
            &provider(),
        )
        .expect("non-equi JOIN must parse in v0.6");
        // Walk past the outer wildcard Project to the Join.
        let join_plan = match &plan {
            LogicalPlan::Project { input, .. } => input.as_ref(),
            other => other,
        };
        match join_plan {
            LogicalPlan::Join { on, filter, .. } => {
                assert!(on.is_empty(), "pure non-equi ON yields no equi pairs");
                assert!(filter.is_some(), "non-equi ON populates the filter slot");
            }
            other => panic!("expected Join, got {other:?}"),
        }
    }

    #[test]
    fn between_join_lowers_to_join_with_filter() {
        // `BETWEEN` lowers to `low <= x AND x <= high` in the filter slot.
        let plan = parse(
            "SELECT * FROM t1 INNER JOIN t2 ON t1.a BETWEEN t2.a AND t2.c",
            &provider(),
        )
        .expect("BETWEEN JOIN must parse in v0.6");
        let join_plan = match &plan {
            LogicalPlan::Project { input, .. } => input.as_ref(),
            other => other,
        };
        match join_plan {
            LogicalPlan::Join { on, filter, .. } => {
                assert!(on.is_empty(), "BETWEEN yields no equi pairs");
                assert!(filter.is_some(), "BETWEEN populates the filter slot");
            }
            other => panic!("expected Join, got {other:?}"),
        }
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

    // ---- Schema-qualified names (single-catalog) ---------------------------

    #[test]
    fn schema_qualified_from_scans_base_table() {
        // `FROM main.t1` is the same table as `FROM t1`: the leading
        // schema/catalog qualifier is accepted and dropped, and the lowered
        // Scan names the *base* table.
        let plan = lp("SELECT a FROM main.t1");
        let scan = match plan {
            LogicalPlan::Project { input, .. } => *input,
            other => other,
        };
        match scan {
            LogicalPlan::Scan { table, .. } => {
                assert_eq!(table, "t1", "qualifier must be dropped, base scanned");
            }
            other => panic!("expected Scan, got {other:?}"),
        }
    }

    #[test]
    fn schema_qualified_unknown_base_table_errors() {
        // The qualifier is dropped, but the base table must still exist.
        let err = parse("SELECT a FROM main.nope", &provider())
            .expect_err("unknown base table must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown table 'nope'"),
            "expected unknown-base-table error, got: {msg}"
        );
    }

    #[test]
    fn three_part_column_ref_resolves_to_bare_column() {
        // `SELECT main.t1.a` drops `main` and resolves the trailing `t1.a`
        // pair to the same `Column("a")` the two-part form produces.
        let plan = lp("SELECT main.t1.a FROM main.t1");
        match plan {
            LogicalPlan::Project { exprs, .. } => {
                assert_eq!(exprs.len(), 1);
                match &exprs[0] {
                    Expr::Column(name) => assert_eq!(name, "a"),
                    other => panic!("expected Column, got {other:?}"),
                }
            }
            other => panic!("expected Project, got {other:?}"),
        }
    }

    #[test]
    fn three_part_column_ref_respects_join_rename() {
        // The middle slot is the table qualifier; the leading schema part is
        // dropped. `cat.t2.a` resolves to the post-join-rename `right.a`,
        // exactly as the two-part `t2.a` would.
        let plan = lp(
            "SELECT cat.t1.a, cat.t2.a FROM t1 INNER JOIN t2 ON t1.a = t2.a",
        );
        match plan {
            LogicalPlan::Project { exprs, .. } => match (&exprs[0], &exprs[1]) {
                (Expr::Column(left), Expr::Column(right)) => {
                    assert_eq!(left, "a", "cat.t1.a keeps bare name");
                    assert_eq!(right, "right.a", "cat.t2.a uses post-rename name");
                }
                other => panic!("expected two Column refs, got {other:?}"),
            },
            other => panic!("expected Project, got {other:?}"),
        }
    }

    #[test]
    fn three_part_column_ref_unknown_table_errors() {
        // The middle (table) slot must be in scope; the leading schema part
        // is irrelevant to resolution.
        let err = parse("SELECT cat.bogus.a FROM t1", &provider())
            .expect_err("unknown table qualifier must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown table qualifier"),
            "expected unknown-table-qualifier error, got: {msg}"
        );
    }

    #[test]
    fn four_part_column_ref_rejected_as_deeply_qualified() {
        // No second namespace level to fold a 4-part reference into.
        let err = parse("SELECT a.b.c.d FROM t1", &provider())
            .expect_err("4-part reference must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("deeply qualified"),
            "expected deeply-qualified error, got: {msg}"
        );
    }

    #[test]
    fn four_part_table_name_in_from_rejected() {
        // `cat.schema.table` (and deeper) has no namespace to collapse into.
        let err = parse("SELECT a FROM cat.schema.t1", &provider())
            .expect_err("3-part table name must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("at most one schema/catalog qualifier"),
            "expected one-qualifier-limit error, got: {msg}"
        );
    }

    /// Three-column fixture for the HAVING/alias stress tests: `t(k, v)`
    /// (group key + aggregable value), separate from the wave-7 join
    /// fixture above so the existing tests aren't perturbed.
    fn having_provider() -> MemTableProvider {
        use crate::plan::logical_plan::Field;
        let t = Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Float64, false),
        ]);
        MemTableProvider::new().with_table("t", t)
    }

    fn lp_with(sql: &str, prov: &MemTableProvider) -> LogicalPlan {
        parse(sql, prov).unwrap_or_else(|e| panic!("parse failed for {sql:?}: {e}"))
    }

    /// `HAVING total > 10` after `SUM(v) AS total`. The HAVING predicate
    /// references the SELECT alias; lowered as `Column("total")` and the
    /// outer Project (built in SELECT order) exposes a `total` column, so
    /// validation must succeed and the plan must end in
    /// `Filter(Project(Aggregate(...)))`.
    #[test]
    fn having_via_alias_passthrough_succeeds() {
        let prov = having_provider();
        let plan = lp_with(
            "SELECT SUM(v) AS total FROM t GROUP BY k HAVING total > 10",
            &prov,
        );
        match plan {
            LogicalPlan::Filter { input, predicate } => {
                // Below the HAVING Filter is the SELECT-order Project.
                assert!(
                    matches!(*input, LogicalPlan::Project { .. }),
                    "expected Project under HAVING Filter, got {input:?}"
                );
                // The predicate's left side must be the alias `total` (not
                // an unresolved aggregate output name like `sum_v`).
                match predicate {
                    Expr::Binary { left, .. } => match *left {
                        Expr::Column(name) => assert_eq!(
                            name, "total",
                            "HAVING alias should lower to Column(\"total\")"
                        ),
                        other => panic!("expected Column on HAVING LHS, got {other:?}"),
                    },
                    other => panic!("expected Binary HAVING predicate, got {other:?}"),
                }
            }
            other => panic!("expected Filter (HAVING) at top, got {other:?}"),
        }
    }

    /// `HAVING SUM(v) > 10` with no alias. The HAVING-aware lowerer
    /// rewrites `SUM(v)` into `Column("sum_v")` and the unaliased SELECT
    /// Project exposes that same name.
    #[test]
    fn having_no_alias_succeeds() {
        let prov = having_provider();
        let plan = lp_with(
            "SELECT SUM(v) FROM t GROUP BY k HAVING SUM(v) > 10",
            &prov,
        );
        match plan {
            LogicalPlan::Filter { input, predicate } => {
                assert!(
                    matches!(*input, LogicalPlan::Project { .. }),
                    "expected Project under HAVING Filter, got {input:?}"
                );
                match predicate {
                    Expr::Binary { left, .. } => match *left {
                        Expr::Column(name) => assert_eq!(
                            name, "sum_v",
                            "SUM(v) in HAVING should lower to Column(\"sum_v\")"
                        ),
                        other => panic!("expected Column on HAVING LHS, got {other:?}"),
                    },
                    other => panic!("expected Binary HAVING predicate, got {other:?}"),
                }
            }
            other => panic!("expected Filter (HAVING) at top, got {other:?}"),
        }
    }

    /// `SELECT SUM(v) AS total ... HAVING SUM(v) > 10`. The aggregate call
    /// in HAVING is rewritten to the underlying aggregate output name
    /// (`sum_v`), even though the SELECT names it `total`. Both names live
    /// in the Project's output (the alias renames one of them), so
    /// validation must pass.
    #[test]
    fn having_mix_alias_and_aggregate_call_succeeds() {
        let prov = having_provider();
        let plan = lp_with(
            "SELECT SUM(v) AS total FROM t GROUP BY k HAVING SUM(v) > 10",
            &prov,
        );
        // The plan must still wrap a Project in a Filter — alias renaming
        // doesn't change the layering.
        let (predicate, _) = match plan {
            LogicalPlan::Filter { predicate, input } => (predicate, input),
            other => panic!("expected Filter (HAVING) at top, got {other:?}"),
        };
        // The HAVING-aware lowerer rewrites the bare `SUM(v)` call into the
        // aggregate output name (`sum_v`). The SELECT alias `total` doesn't
        // change the predicate; it changes the Project. Either name would
        // be acceptable as long as the Project below carries it.
        match predicate {
            Expr::Binary { left, .. } => match *left {
                Expr::Column(name) => assert!(
                    name == "sum_v" || name == "total",
                    "HAVING SUM(v) should resolve to either the underlying \
                     aggregate name or its alias; got {name:?}"
                ),
                other => panic!("expected Column on HAVING LHS, got {other:?}"),
            },
            other => panic!("expected Binary HAVING predicate, got {other:?}"),
        }
    }

    /// `HAVING nonexistent > 10` — typo. Must fail with a clear "unknown
    /// column" error from the HAVING column validator, *not* a downstream
    /// type-checker error.
    #[test]
    fn having_unknown_column_errors() {
        let prov = having_provider();
        let err = parse(
            "SELECT SUM(v) AS total FROM t GROUP BY k HAVING nonexistent > 10",
            &prov,
        )
        .expect_err("HAVING with unknown column must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("HAVING references unknown column"),
            "expected HAVING validator error, got: {msg}"
        );
        assert!(
            msg.contains("nonexistent"),
            "error message should name the missing column, got: {msg}"
        );
    }

    /// `HAVING k > 0 AND s > 10` — combined group-key and aggregate alias
    /// references. Both names must be in the Project's output schema; the
    /// AND-node walker has to descend into both sides.
    #[test]
    fn having_group_key_and_aggregate_alias_succeeds() {
        let prov = having_provider();
        let plan = lp_with(
            "SELECT k, SUM(v) AS s FROM t GROUP BY k HAVING k > 0 AND s > 10",
            &prov,
        );
        match plan {
            LogicalPlan::Filter { input, predicate } => {
                assert!(
                    matches!(*input, LogicalPlan::Project { .. }),
                    "expected Project under HAVING Filter, got {input:?}"
                );
                // Walk the AND predicate, collect column names, and check
                // both `k` and `s` appear.
                let mut names: Vec<String> = Vec::new();
                let mut stack = vec![&predicate];
                while let Some(e) = stack.pop() {
                    match e {
                        Expr::Column(n) => names.push(n.clone()),
                        Expr::Binary { left, right, .. } => {
                            stack.push(left);
                            stack.push(right);
                        }
                        Expr::Alias(inner, _) => stack.push(inner),
                        Expr::Unary { operand, .. } => stack.push(operand),
                        Expr::Case {
                            branches,
                            else_branch,
                        } => {
                            for (w, t) in branches {
                                stack.push(w);
                                stack.push(t);
                            }
                            if let Some(e) = else_branch {
                                stack.push(e);
                            }
                        }
                        Expr::Like { expr, .. } => stack.push(expr),
                        Expr::Cast { expr, .. } => stack.push(expr),
                        Expr::ScalarFn { args, .. } => {
                            for a in args {
                                stack.push(a);
                            }
                        }
                        Expr::Extract { expr, .. } | Expr::DateTrunc { expr, .. } => {
                            stack.push(expr)
                        }
                        Expr::ScalarSubquery(_) => {}
                        Expr::InSubquery { expr, .. } => stack.push(expr),
                        Expr::Literal(_) => {}
                    }
                }
                assert!(
                    names.iter().any(|n| n == "k"),
                    "HAVING predicate must reference group key 'k', got {names:?}"
                );
                assert!(
                    names.iter().any(|n| n == "s"),
                    "HAVING predicate must reference aggregate alias 's', got {names:?}"
                );
            }
            other => panic!("expected Filter (HAVING) at top, got {other:?}"),
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

    // ---- ON-clause cross-side equality validation (MED fix) ----
    //
    // The lowerer previously stripped table qualifiers from each ON-clause
    // side without checking that the two sides came from different
    // tables. `JOIN t2 ON t1.a = t1.a` therefore reached the executor as
    // `(Column("a"), Column("a"))` and only blew up at run time; the four
    // tests below pin the new plan-time validation in place.

    #[test]
    fn join_on_cross_table_equality_passes_lowering() {
        // Sanity baseline: a well-formed `t1.a = t2.a` predicate is
        // accepted and lowers to a Join with one equality pair. The ON
        // clause keeps the original (pre-rename) bare column names — the
        // executor (`src/exec/join.rs::split_keys`) matches those against
        // its build/probe schemas directly. The rename rule applies only
        // to the JOIN's *output* schema, not to its ON pairs.
        let plan = lp("SELECT * FROM t1 JOIN t2 ON t1.a = t2.a");
        let join_plan = match &plan {
            LogicalPlan::Project { input, .. } => input.as_ref(),
            other => other,
        };
        match join_plan {
            LogicalPlan::Join { on, join_type, .. } => {
                assert_eq!(*join_type, JoinType::Inner);
                assert_eq!(on.len(), 1, "expected one equi pair, got {on:?}");
                match &on[0] {
                    (Expr::Column(l), Expr::Column(r)) => {
                        assert_eq!(l, "a", "left key uses bare column name");
                        assert_eq!(r, "a", "right key uses bare column name");
                    }
                    other => panic!("expected two Column refs, got {other:?}"),
                }
            }
            other => panic!("expected Join, got {other:?}"),
        }
    }

    #[test]
    fn join_on_same_side_equality_is_rejected() {
        // `t1.a = t1.a` is the bug case: both qualifiers name the left
        // side, so the predicate cannot constrain the join. Reject at
        // plan time with a clear message naming the failure mode.
        let err = parse("SELECT * FROM t1 JOIN t2 ON t1.a = t1.a", &provider())
            .expect_err("same-side ON equality must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("both sides reference the same table"),
            "expected same-side message, got: {msg}"
        );
    }

    #[test]
    fn join_on_unknown_table_is_rejected() {
        // `t3` is not part of FROM at all. The pre-fix lowerer dropped
        // the qualifier and quietly produced a useless `(a, a)` pair;
        // we now report the offending qualifier so the user can fix the
        // typo.
        let err = parse("SELECT * FROM t1 JOIN t2 ON t3.a = t2.a", &provider())
            .expect_err("ON clause referencing unknown table must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown table 't3'"),
            "expected 'unknown table t3' message, got: {msg}"
        );
    }

    #[test]
    fn join_on_second_conjunct_same_side_is_rejected() {
        // The first conjunct is well-formed (`t1.a = t2.a`); the second
        // is the bug shape (`t1.b = t1.b`). Validation must traverse
        // the full AND tree and reject the offending leaf so a partial
        // good prefix can't mask a bad tail.
        let err = parse(
            "SELECT * FROM t1 JOIN t2 ON t1.a = t2.a AND t1.b = t1.b",
            &provider(),
        )
        .expect_err("same-side conjunct anywhere in the AND tree must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("both sides reference the same table"),
            "expected same-side message, got: {msg}"
        );
    }

    // ----- Window function frontend lowering (v0.x) -----

    /// `ROW_NUMBER() OVER (PARTITION BY a ORDER BY b)` lowers to a Project
    /// over a single-expr `Window` node carrying the parsed spec.
    #[test]
    fn window_row_number_lowers_to_window_node() {
        let plan = lp("SELECT ROW_NUMBER() OVER (PARTITION BY a ORDER BY b) FROM t1");
        let input = match plan {
            LogicalPlan::Project { input, .. } => input,
            other => panic!("expected Project at top, got {other:?}"),
        };
        match *input {
            LogicalPlan::Window {
                window_exprs,
                partition_by,
                order_by,
                ..
            } => {
                assert_eq!(window_exprs.len(), 1);
                assert!(matches!(window_exprs[0].func, WindowFunc::RowNumber));
                assert_eq!(partition_by.len(), 1);
                assert_eq!(order_by.len(), 1);
            }
            other => panic!("expected Window under Project, got {other:?}"),
        }
    }

    /// `SUM(b) OVER (PARTITION BY a)` lowers to an aggregate window whose
    /// output schema appends a `sum` column to the input.
    #[test]
    fn window_sum_over_partition_schema() {
        let plan = lp("SELECT a, SUM(b) OVER (PARTITION BY a) AS rs FROM t1");
        let schema = plan.schema().expect("window plan must type-check");
        // Output is the projection: a, rs.
        assert_eq!(schema.fields.len(), 2);
        assert_eq!(schema.fields[1].name, "rs");
        // SUM(Int64) stays Int64.
        assert_eq!(schema.fields[1].dtype, DataType::Int64);
    }

    /// Multiple window functions sharing one spec collapse into a single
    /// `Window` node with two output exprs.
    #[test]
    fn window_shared_spec_collapses() {
        let plan = lp(
            "SELECT RANK() OVER (ORDER BY b), DENSE_RANK() OVER (ORDER BY b) FROM t1",
        );
        let input = match plan {
            LogicalPlan::Project { input, .. } => input,
            other => panic!("expected Project, got {other:?}"),
        };
        match *input {
            LogicalPlan::Window { window_exprs, .. } => {
                assert_eq!(window_exprs.len(), 2, "shared spec should produce one node");
            }
            other => panic!("expected single Window, got {other:?}"),
        }
    }

    /// An explicit non-default frame is rejected cleanly.
    #[test]
    fn window_explicit_frame_rejected() {
        let err = parse(
            "SELECT SUM(b) OVER (ORDER BY b ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t1",
            &provider(),
        )
        .expect_err("explicit non-default frame must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("frame"),
            "expected a frame-related rejection, got: {msg}"
        );
    }

    /// A window function nested inside a larger expression is rejected
    /// (top-level-only support for now).
    #[test]
    fn window_nested_in_expression_rejected() {
        let err = parse(
            "SELECT ROW_NUMBER() OVER (ORDER BY b) + 1 FROM t1",
            &provider(),
        )
        .expect_err("nested window must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("top-level SELECT item"),
            "expected top-level-only message, got: {msg}"
        );
    }

    /// The window path lowers through to a `PhysicalPlan::Window`.
    #[test]
    fn window_lowers_to_physical_window() {
        let phys = pp("SELECT ROW_NUMBER() OVER (PARTITION BY a ORDER BY b) FROM t1");
        // Project over Window over the scan-projection.
        match phys {
            PhysicalPlan::Project { input, .. } => {
                assert!(
                    matches!(*input, PhysicalPlan::Window { .. }),
                    "expected Window under Project"
                );
            }
            other => panic!("expected Project at top, got {other:?}"),
        }
    }
}

/// v0.5 LIKE parse-shape tests. Lock the SQL frontend surface for
/// `expr LIKE 'pat'` and `expr NOT LIKE 'pat'`: pattern must be a string
/// literal, ESCAPE is rejected as a follow-up, the typed Expr::Like
/// captures pattern verbatim. Execution is covered by
/// `tests/like_test.rs` against the host evaluator.
#[cfg(test)]
mod like_tests {
    use super::*;
    use crate::plan::logical_plan::{DataType, Field};

    /// `s` is a Utf8 column for LIKE tests; `v` is an unrelated Int64 so
    /// we can also assert LIKE on a non-Utf8 column rejects cleanly.
    fn s_provider() -> MemTableProvider {
        let t = Schema::new(vec![
            Field::new("s", DataType::Utf8, false),
            Field::new("v", DataType::Int64, false),
        ]);
        MemTableProvider::new().with_table("t", t)
    }

    /// Pull the Filter predicate out of a top-level
    /// `Project { input: Filter { .. } }` plan.
    fn predicate(plan: &LogicalPlan) -> &Expr {
        match plan {
            LogicalPlan::Project { input, .. } => match input.as_ref() {
                LogicalPlan::Filter { predicate, .. } => predicate,
                other => panic!("expected Filter under Project, got {other:?}"),
            },
            LogicalPlan::Filter { predicate, .. } => predicate,
            other => panic!("expected Project or Filter at top, got {other:?}"),
        }
    }

    #[test]
    fn parse_like_exact_pattern() {
        let plan = parse("SELECT s FROM t WHERE s LIKE 'foo'", &s_provider())
            .expect("LIKE 'foo' must parse");
        match predicate(&plan) {
            Expr::Like {
                pattern,
                negated,
                escape,
                ..
            } => {
                assert_eq!(pattern, "foo");
                assert!(!negated);
                assert!(escape.is_none());
            }
            other => panic!("expected Expr::Like, got {other:?}"),
        }
    }

    #[test]
    fn parse_like_prefix_pattern() {
        let plan = parse("SELECT s FROM t WHERE s LIKE 'foo%'", &s_provider())
            .expect("LIKE 'foo%' must parse");
        match predicate(&plan) {
            Expr::Like { pattern, .. } => assert_eq!(pattern, "foo%"),
            other => panic!("expected Expr::Like, got {other:?}"),
        }
    }

    #[test]
    fn parse_like_suffix_pattern() {
        let plan = parse("SELECT s FROM t WHERE s LIKE '%foo'", &s_provider()).unwrap();
        match predicate(&plan) {
            Expr::Like { pattern, .. } => assert_eq!(pattern, "%foo"),
            other => panic!("expected Expr::Like, got {other:?}"),
        }
    }

    #[test]
    fn parse_like_contains_pattern() {
        let plan = parse("SELECT s FROM t WHERE s LIKE '%foo%'", &s_provider()).unwrap();
        match predicate(&plan) {
            Expr::Like { pattern, .. } => assert_eq!(pattern, "%foo%"),
            other => panic!("expected Expr::Like, got {other:?}"),
        }
    }

    #[test]
    fn parse_not_like_sets_negated_flag() {
        let plan = parse(
            "SELECT s FROM t WHERE s NOT LIKE 'foo%'",
            &s_provider(),
        )
        .unwrap();
        match predicate(&plan) {
            Expr::Like {
                pattern, negated, ..
            } => {
                assert_eq!(pattern, "foo%");
                assert!(*negated, "NOT LIKE must set negated=true");
            }
            other => panic!("expected Expr::Like, got {other:?}"),
        }
    }

    #[test]
    fn parse_like_on_non_utf8_column_typeerrs() {
        // `v` is Int64; LIKE must require Utf8.
        let err = parse("SELECT v FROM t WHERE v LIKE 'foo'", &s_provider())
            .expect_err("LIKE on Int64 must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("LIKE requires a Utf8 operand"),
            "expected Utf8 type error, got: {msg}"
        );
    }

    #[test]
    fn parse_like_with_variable_pattern_rejected() {
        // Pattern must be a string literal — a column ref is not allowed.
        let err = parse(
            "SELECT s FROM t WHERE s LIKE s",
            &s_provider(),
        )
        .expect_err("variable LIKE pattern must reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("LIKE pattern must be a string literal constant"),
            "expected constant-pattern message, got: {msg}"
        );
    }

    #[test]
    fn parse_like_with_escape_captured_on_expr() {
        // v0.7: ESCAPE is now supported. The frontend captures both the
        // pattern (verbatim) and the escape character on Expr::Like.
        let plan = parse(
            r"SELECT s FROM t WHERE s LIKE 'a\_b' ESCAPE '\'",
            &s_provider(),
        )
        .expect("ESCAPE must parse");
        match predicate(&plan) {
            Expr::Like {
                pattern,
                escape,
                negated,
                ..
            } => {
                assert_eq!(pattern, r"a\_b");
                assert_eq!(*escape, Some('\\'));
                assert!(!negated);
            }
            other => panic!("expected Expr::Like, got {other:?}"),
        }
    }

    #[test]
    fn parse_like_with_multichar_escape_rejected() {
        // Standard SQL requires the ESCAPE clause to be a single
        // character — a two-char literal must reject cleanly.
        let err = parse(
            r"SELECT s FROM t WHERE s LIKE 'foo' ESCAPE 'ab'",
            &s_provider(),
        )
        .expect_err("multi-char ESCAPE must reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("single character"),
            "expected single-character message, got: {msg}"
        );
    }

    #[test]
    fn parse_not_like_with_escape_captures_negation_and_escape() {
        let plan = parse(
            r"SELECT s FROM t WHERE s NOT LIKE 'a!%b' ESCAPE '!'",
            &s_provider(),
        )
        .expect("NOT LIKE ... ESCAPE must parse");
        match predicate(&plan) {
            Expr::Like {
                pattern,
                escape,
                negated,
                ..
            } => {
                assert_eq!(pattern, "a!%b");
                assert_eq!(*escape, Some('!'));
                assert!(*negated, "NOT LIKE ... ESCAPE must set negated=true");
            }
            other => panic!("expected Expr::Like, got {other:?}"),
        }
    }
}

/// SUBSTRING / TRIM frontend parse + lower coverage. Execution is covered by
/// the host-eval unit tests in `exec::string_ops_extended` / `exec::expr_agg`
/// and the `#[ignore]` e2e tests in `tests/e2e_tests.rs`.
#[cfg(test)]
mod string_fn_tests {
    use super::*;
    use crate::plan::logical_plan::{DataType, Field, ScalarFnKind};
    use crate::plan::physical_plan::{lower, PhysicalPlan};

    fn s_provider() -> MemTableProvider {
        let t = Schema::new(vec![
            Field::new("s", DataType::Utf8, false),
            Field::new("v", DataType::Int64, false),
        ]);
        MemTableProvider::new().with_table("t", t)
    }

    /// First SELECT-list expr of a top-level `Project`, peeled of aliases.
    fn first_select_expr(plan: &LogicalPlan) -> Expr {
        let exprs = match plan {
            LogicalPlan::Project { exprs, .. } => exprs,
            other => panic!("expected Project at top, got {other:?}"),
        };
        let mut e = &exprs[0];
        while let Expr::Alias(inner, _) = e {
            e = inner;
        }
        e.clone()
    }

    #[test]
    fn substring_from_for_parses_to_scalar_fn() {
        let plan = parse("SELECT SUBSTRING(s FROM 2 FOR 3) FROM t", &s_provider())
            .expect("SUBSTRING ... FROM ... FOR must parse");
        match first_select_expr(&plan) {
            Expr::ScalarFn { kind, args } => {
                assert_eq!(kind, ScalarFnKind::Substring);
                assert_eq!(args.len(), 3);
            }
            other => panic!("expected ScalarFn(Substring), got {other:?}"),
        }
    }

    #[test]
    fn substring_comma_two_arg_parses() {
        let plan = parse("SELECT SUBSTRING(s, 2) FROM t", &s_provider())
            .expect("SUBSTRING(s, i) must parse");
        match first_select_expr(&plan) {
            Expr::ScalarFn { kind, args } => {
                assert_eq!(kind, ScalarFnKind::Substring);
                assert_eq!(args.len(), 2);
            }
            other => panic!("expected ScalarFn(Substring), got {other:?}"),
        }
    }

    #[test]
    fn trim_default_is_both() {
        let plan =
            parse("SELECT TRIM(s) FROM t", &s_provider()).expect("TRIM(s) must parse");
        match first_select_expr(&plan) {
            Expr::ScalarFn { kind, args } => {
                assert_eq!(kind, ScalarFnKind::TrimBoth);
                assert_eq!(args.len(), 1);
            }
            other => panic!("expected ScalarFn(TrimBoth), got {other:?}"),
        }
    }

    #[test]
    fn trim_leading_trailing_both_sides() {
        // NOTE: sqlparser 0.52's `parse_trim_expr` unconditionally parses an
        // expression after the `BOTH|LEADING|TRAILING` keyword *before* it
        // looks for `FROM`. That means the bare standard-SQL form
        // `TRIM(LEADING FROM s)` (a trim side with NO trim-characters between
        // the keyword and `FROM`) is NOT accepted by this version of the
        // parser, regardless of the configured `Dialect` (the relevant code
        // path is dialect-independent). It fails at the sqlparser stage with
        // an "Expected: ), found: s" parse error. The frontend's
        // `SqlExpr::Trim` -> `ScalarFnKind::Trim*` mapping is correct and
        // would handle this case fine; the limitation is purely upstream in
        // the pinned sqlparser version. See `trim_bare_side_from_unsupported`
        // below for the assertion that this form is currently rejected.
        //
        // We therefore exercise the three trim sides through the
        // `TRIM(<side> 'chars' FROM s)` form, which DOES parse and drives the
        // exact same `trim_where` -> `ScalarFnKind::Trim*` mapping branch.
        for (sql, expect) in [
            (
                "SELECT TRIM(LEADING 'x' FROM s) FROM t",
                ScalarFnKind::TrimLeading,
            ),
            (
                "SELECT TRIM(TRAILING 'x' FROM s) FROM t",
                ScalarFnKind::TrimTrailing,
            ),
            ("SELECT TRIM(BOTH 'x' FROM s) FROM t", ScalarFnKind::TrimBoth),
        ] {
            let plan = parse(sql, &s_provider()).unwrap_or_else(|e| panic!("{sql}: {e}"));
            match first_select_expr(&plan) {
                Expr::ScalarFn { kind, args } => {
                    assert_eq!(kind, expect, "for {sql}");
                    // source + trim-characters
                    assert_eq!(args.len(), 2, "for {sql}");
                }
                other => panic!("expected ScalarFn, got {other:?} for {sql}"),
            }
        }
    }

    #[test]
    fn trim_bare_side_from_unsupported() {
        // The bare `TRIM(<side> FROM s)` form (no trim-characters) is not
        // parseable by sqlparser 0.52 — see the NOTE on
        // `trim_leading_trailing_both_sides`. Assert it is currently rejected
        // with a clear SQL parse error rather than silently mis-parsing. If a
        // future sqlparser upgrade starts accepting this form, this test will
        // fail and is the signal to re-enable the bare-form assertions above.
        for sql in [
            "SELECT TRIM(LEADING FROM s) FROM t",
            "SELECT TRIM(TRAILING FROM s) FROM t",
            "SELECT TRIM(BOTH FROM s) FROM t",
        ] {
            let err = parse(sql, &s_provider())
                .expect_err("bare TRIM(<side> FROM s) is unsupported in sqlparser 0.52");
            let msg = format!("{err}");
            assert!(
                msg.contains("parse"),
                "expected a parse error for {sql}, got: {msg}"
            );
        }
    }

    #[test]
    fn trim_custom_chars_from_form() {
        let plan = parse("SELECT TRIM(LEADING 'xy' FROM s) FROM t", &s_provider())
            .expect("TRIM(LEADING 'xy' FROM s) must parse");
        match first_select_expr(&plan) {
            Expr::ScalarFn { kind, args } => {
                assert_eq!(kind, ScalarFnKind::TrimLeading);
                assert_eq!(args.len(), 2, "source + trim chars");
                assert!(matches!(args[1], Expr::Literal(Literal::Utf8(ref s)) if s == "xy"));
            }
            other => panic!("expected ScalarFn(TrimLeading), got {other:?}"),
        }
    }

    #[test]
    fn substring_and_trim_lower_to_host_project() {
        // Both functions have no GPU producer; they must lower to the
        // host-side PhysicalPlan::Project (not be rejected).
        for sql in [
            "SELECT SUBSTRING(s, 2, 3) FROM t",
            "SELECT TRIM(s) FROM t",
            "SELECT TRIM(TRAILING '-' FROM s) FROM t",
        ] {
            let plan = parse(sql, &s_provider()).unwrap_or_else(|e| panic!("{sql}: {e}"));
            let phys = lower(&plan).unwrap_or_else(|e| panic!("lower {sql}: {e}"));
            assert!(
                matches!(phys, PhysicalPlan::Project { .. }),
                "expected host PhysicalPlan::Project for {sql}, got {phys:?}"
            );
        }
    }

    #[test]
    fn trim_type_error_on_non_utf8() {
        // TRIM over an Int64 column must surface a Type error at schema check.
        // (The eager parse-time validation is scoped to set-op roots, so an
        // ordinary SELECT still type-checks lazily at `schema()`.)
        let plan = parse("SELECT TRIM(v) FROM t", &s_provider())
            .expect("TRIM(v) parses; type-check happens on schema()");
        let err = plan.schema().expect_err("TRIM(Int64) must type-error");
        let msg = format!("{err}");
        assert!(
            msg.contains("TRIM") && msg.contains("Utf8"),
            "expected TRIM Utf8 type error, got: {msg}"
        );
    }
}

#[cfg(test)]
mod root_setop_schema_tests {
    //! Regression tests for eager schema validation of the *top-level* plan in
    //! `parse_uncached`. A root UNION / EXCEPT / INTERSECT with incompatible
    //! branches (mismatched column count or per-field dtypes) must be rejected
    //! at parse time with the existing descriptive error that names the op,
    //! while compatible set-ops still parse cleanly.
    use super::*;
    use crate::plan::logical_plan::{DataType, Field};

    /// A table with a wide enough schema to slice differently-shaped SELECTs
    /// out of it: `region_id` (1 col) vs `region_id, qty` (2 cols).
    fn provider() -> MemTableProvider {
        let sales = Schema::new(vec![
            Field::new("region_id", DataType::Int64, false),
            Field::new("qty", DataType::Int64, false),
        ]);
        MemTableProvider::new().with_table("sales", sales)
    }

    #[test]
    fn except_incompatible_arity_rejected() {
        let p = provider();
        let err = parse(
            "SELECT region_id FROM sales EXCEPT SELECT region_id, qty FROM sales",
            &p,
        )
        .expect_err("incompatible-arity EXCEPT must be rejected at parse time");
        let msg = err.to_string();
        assert!(
            msg.contains("EXCEPT"),
            "expected error naming EXCEPT, got: {msg}"
        );
    }

    #[test]
    fn intersect_incompatible_arity_rejected() {
        let p = provider();
        let err = parse(
            "SELECT region_id FROM sales INTERSECT SELECT region_id, qty FROM sales",
            &p,
        )
        .expect_err("incompatible-arity INTERSECT must be rejected at parse time");
        assert!(
            err.to_string().contains("INTERSECT"),
            "expected error naming INTERSECT, got: {err}"
        );
    }

    #[test]
    fn union_incompatible_arity_rejected() {
        let p = provider();
        let err = parse(
            "SELECT region_id FROM sales UNION SELECT region_id, qty FROM sales",
            &p,
        )
        .expect_err("incompatible-arity UNION must be rejected at parse time");
        // The Union schema-check error mentions "UNION branch".
        assert!(
            err.to_string().contains("UNION"),
            "expected error naming UNION, got: {err}"
        );
    }

    #[test]
    fn compatible_setops_still_parse() {
        let p = provider();
        for sql in [
            "SELECT region_id, qty FROM sales EXCEPT SELECT region_id, qty FROM sales",
            "SELECT region_id, qty FROM sales INTERSECT SELECT region_id, qty FROM sales",
            "SELECT region_id, qty FROM sales UNION SELECT region_id, qty FROM sales",
            "SELECT region_id FROM sales", // a plain root SELECT is unaffected
        ] {
            let plan = parse(sql, &p).unwrap_or_else(|e| panic!("compatible {sql:?}: {e}"));
            // The eager check computed a schema; recomputing must agree.
            assert!(
                plan.schema().is_ok(),
                "valid plan schema must recompute cleanly for {sql:?}"
            );
        }
    }
}
