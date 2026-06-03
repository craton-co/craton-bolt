// SPDX-License-Identifier: Apache-2.0

//! SQL frontend: parses a SQL string into a `LogicalPlan` against a `TableProvider`.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use sqlparser::ast::{
    BinaryOperator, CastFormat as SqlCastFormat, CastKind, DataType as SqlDataType, Distinct,
    Expr as SqlExpr, Fetch, FunctionArg, FunctionArgExpr, FunctionArguments, GroupByExpr,
    GroupByWithModifier, Ident,
    JoinConstraint, JoinOperator, NamedWindowExpr, ObjectName, Offset, OrderByExpr, Query, Select,
    SelectItem, SetExpr, SetOperator, SetQuantifier, Statement, TableFactor, Top, TopQuantity,
    UnaryOperator, Value, WindowSpec,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::{Parser, ParserError};
use sqlparser::tokenizer::Tokenizer;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{
    aggregate_output_name, apply_column_aliases, group_key_output_name, join_rename, AggregateExpr,
    BinaryOp, DataType, Expr, Field, FormatToken, JoinType, Literal, LogicalPlan, ScalarFnKind,
    Schema, SetOpKind, SortExpr, TimeUnit, UnaryOp, WindowExpr, WindowFunc,
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

/// Hard ceiling on the *byte length* of an incoming SQL string, enforced
/// **before** the text is handed to `sqlparser`.
///
/// # Why this exists (do not remove)
///
/// sqlparser 0.52's own recursion guard (`ParserError::RecursionLimitExceeded`)
/// only counts *prefix* recursion. It does NOT bound the size of the AST built
/// from a flat, left-associative operator chain (`a + a + … + a`, tens of
/// thousands deep) or a long `OR`/`AND` chain, nor a deeply nested
/// `IN (SELECT …)` ladder. Such inputs parse into an enormous AST whose
/// *recursive `Drop`* blows the host thread stack and aborts the process
/// (observed: `STATUS_STACK_OVERFLOW` on ~20k `+`, ~200k `OR`, ~5k-deep
/// nested `IN`-subqueries). Our existing [`MAX_RECURSION_DEPTH`] lowering
/// guard runs far too late — by the time lowering walks the tree, the
/// dangerous AST already exists and will still crash on `Drop`.
///
/// The only robust mitigation is to refuse pathological inputs *before*
/// `sqlparser` allocates an AST at all. 1 MiB is generous — orders of
/// magnitude larger than any dashboard / hand-written query in our corpus —
/// yet small enough that even the densest valid SQL within it cannot build
/// an AST deep enough to overflow the stack on `Drop`.
const MAX_SQL_BYTES_DEFAULT: usize = 1 << 20; // 1 MiB

/// Hard ceiling on the *token count* of an incoming SQL string, enforced
/// **before** the full parse. See [`MAX_SQL_BYTES_DEFAULT`] for the crash
/// rationale: byte length alone does not bound AST depth (a 1 MiB blob of
/// `a+a+a+…` is short on bytes-per-token but long on AST nodes), so we also
/// cap how many tokens we will feed to the parser. Cheap to compute — the
/// tokenizer is a linear scan and allocates only a flat `Vec<Token>`, never
/// the recursive AST whose `Drop` is the hazard.
const MAX_SQL_TOKENS_DEFAULT: usize = 100_000;

/// Environment variable overriding [`MAX_SQL_BYTES_DEFAULT`]. Follows the
/// same `CRATON_*` convention as [`PLAN_CACHE_SIZE_ENV`]; read once and
/// frozen for the process lifetime.
const MAX_SQL_BYTES_ENV: &str = "CRATON_MAX_SQL_BYTES";

/// Environment variable overriding [`MAX_SQL_TOKENS_DEFAULT`]. See
/// [`MAX_SQL_BYTES_ENV`].
const MAX_SQL_TOKENS_ENV: &str = "CRATON_MAX_SQL_TOKENS";

/// Resolve the effective byte cap. Memoised via an inner `OnceLock` so the
/// env var is consulted only once (matching [`plan_cache_cap`]); an empty,
/// zero, or unparseable value falls back to the default.
fn max_sql_bytes() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var(MAX_SQL_BYTES_ENV)
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(MAX_SQL_BYTES_DEFAULT)
    })
}

/// Resolve the effective token cap. See [`max_sql_bytes`].
fn max_sql_tokens() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var(MAX_SQL_TOKENS_ENV)
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(MAX_SQL_TOKENS_DEFAULT)
    })
}

/// Pre-parse denial-of-service guard. Run at the very top of the parse
/// pipeline, *before* the SQL string reaches `sqlparser`'s `Parser`.
///
/// Rejects inputs that exceed [`max_sql_bytes`] (cheap, no allocation) or
/// [`max_sql_tokens`] (a linear tokenizer scan that allocates only a flat
/// token vector — never the recursive AST). See [`MAX_SQL_BYTES_DEFAULT`]
/// for why this must happen before parsing: an over-large AST crashes the
/// process during recursive `Drop`, long before the [`MAX_RECURSION_DEPTH`]
/// lowering guard could fire. Returns a descriptive [`BoltError::Sql`] so
/// the failure is a clean, recoverable error rather than a process abort.
fn guard_sql_size(sql: &str) -> BoltResult<()> {
    let max_bytes = max_sql_bytes();
    if sql.len() > max_bytes {
        return Err(BoltError::Sql(format!(
            "SQL input is {} bytes, exceeding the {max_bytes}-byte limit \
             (set {MAX_SQL_BYTES_ENV} to override)",
            sql.len()
        )));
    }
    // Tokenize cheaply to bound AST size. The tokenizer is a flat linear
    // scan; it never builds the recursive AST whose `Drop` is the crash
    // hazard, so counting tokens here is safe even for adversarial input.
    let max_tokens = max_sql_tokens();
    let dialect = GenericDialect {};
    let mut tokenizer = Tokenizer::new(&dialect, sql);
    let tokens = tokenizer
        .tokenize()
        .map_err(|e| BoltError::Sql(format!("tokenizer error: {e}")))?;
    if tokens.len() > max_tokens {
        return Err(BoltError::Sql(format!(
            "SQL input has {} tokens, exceeding the {max_tokens}-token limit \
             (set {MAX_SQL_TOKENS_ENV} to override)",
            tokens.len()
        )));
    }
    Ok(())
}

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

    /// Resolve a collected LATERAL correlation
    /// ([`crate::plan::subquery::CorrRef`]) to the index of the matching column
    /// in this (LEFT) resolver's *combined output schema* — i.e. the position
    /// of its `output` name across all scopes in FROM order (base columns
    /// first, then each joined table's). For a qualified `q.c` reference the
    /// qualifier picks the scope; a bare `c` must be unambiguous across the
    /// whole left relation.
    ///
    /// Uses the same case-folding convention as [`Self::resolve_compound`].
    fn resolve_correlation(
        &self,
        corr: &crate::plan::subquery::CorrRef,
    ) -> BoltResult<usize> {
        // The output index of a column = the count of all output columns in
        // earlier scopes + its position within its own scope.
        let scope_base = |scope_i: usize| -> usize {
            self.tables[..scope_i].iter().map(|t| t.cols.len()).sum()
        };
        match &corr.qualifier {
            Some(q) => {
                let q_lc = !q.chars().any(|c| c.is_ascii_uppercase());
                let scope_i = self
                    .tables
                    .iter()
                    .position(|t| t.name == *q)
                    .or_else(|| {
                        if !q_lc {
                            return None;
                        }
                        self.tables
                            .iter()
                            .position(|t| t.name.eq_ignore_ascii_case(q))
                    })
                    .ok_or_else(|| {
                        BoltError::Sql(format!("unknown table qualifier '{q}'"))
                    })?;
                let col_lc = !corr.column.chars().any(|c| c.is_ascii_uppercase());
                let scope = &self.tables[scope_i];
                let pos = scope
                    .cols
                    .iter()
                    .position(|c| c.original == corr.column)
                    .or_else(|| {
                        if !col_lc {
                            return None;
                        }
                        scope
                            .cols
                            .iter()
                            .position(|c| c.original.eq_ignore_ascii_case(&corr.column))
                    })
                    .ok_or_else(|| {
                        BoltError::Sql(format!(
                            "unknown column '{}' in table '{q}'",
                            corr.column
                        ))
                    })?;
                Ok(scope_base(scope_i) + pos)
            }
            None => {
                // Bare name: must be unambiguous across all scopes.
                let mut found: Option<usize> = None;
                for (scope_i, scope) in self.tables.iter().enumerate() {
                    for (pos, c) in scope.cols.iter().enumerate() {
                        if c.original.eq_ignore_ascii_case(&corr.column) {
                            if found.is_some() {
                                return Err(BoltError::Sql(format!(
                                    "ambiguous outer column '{}' (appears in more \
                                     than one left table)",
                                    corr.column
                                )));
                            }
                            found = Some(scope_base(scope_i) + pos);
                        }
                    }
                }
                found.ok_or_else(|| {
                    BoltError::Sql(format!("unknown outer column '{}'", corr.column))
                })
            }
        }
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
    // DoS guard: reject pathologically large / token-heavy SQL BEFORE
    // sqlparser builds an AST. An over-large flat AST (long `+`/`OR` chains,
    // deep `IN (SELECT …)`) crashes the process during recursive `Drop`,
    // which the late `MAX_RECURSION_DEPTH` lowering guard cannot prevent.
    // See [`guard_sql_size`]. This is the single chokepoint before
    // `Parser::parse_sql`; the cache in [`parse`] never holds an over-cap
    // entry because such input never reaches a successful parse + insert.
    guard_sql_size(sql)?;

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

/// A fully-planned `WITH RECURSIVE` query (feature F1).
///
/// The SQL frontend lowers the three logical pieces of a recursive CTE — the
/// anchor (seed) term, the recursive term, and the main query — into ordinary
/// [`LogicalPlan`]s, with the CTE name bound to a synthetic `Scan` (over the
/// anchor's output schema) wherever it is referenced. The host engine
/// orchestrates the fixpoint by repeatedly executing `recursive` with the CTE
/// table re-bound to the previous iteration's rows, then runs `main` over the
/// accumulated relation. Keeping these as separate, standalone `LogicalPlan`s
/// (rather than a new `LogicalPlan` variant) means each one flows through the
/// existing optimizer / lowering / execution pipeline unchanged.
///
/// See [`plan_recursive_cte`] for how this is built and which SQL shapes are
/// supported vs rejected.
#[derive(Debug, Clone)]
pub struct RecursiveCtePlan {
    /// Case-folded CTE name (the ephemeral relation the recursive term and the
    /// main query scan).
    pub name: String,
    /// The output schema of the accumulated CTE relation (anchor schema, with
    /// any column-list alias applied). The ephemeral table the engine
    /// registers per iteration must have exactly this schema.
    pub cte_schema: Schema,
    /// Non-recursive seed term. Does NOT reference `name`.
    pub anchor: LogicalPlan,
    /// Recursive term. References `name` as a `Scan` over [`Self::cte_schema`].
    pub recursive: LogicalPlan,
    /// `true` for `UNION ALL` (no dedup); `false` for `UNION` (dedup the
    /// accumulated result set; stop when no new rows are produced).
    pub all: bool,
    /// `true` when the recursive term references the CTE **more than once**
    /// (non-linear recursion, e.g. a self-join `FROM r AS r1, r AS r2`). All
    /// the self-reference scans resolve to the same ephemeral table the engine
    /// binds by name, so the only correct evaluation is **naive**: each
    /// iteration binds the FULL accumulated relation (not just the previous
    /// delta) and re-derives. The engine reads this flag to force the
    /// whole-result working set even for `UNION ALL` (where the linear path
    /// would otherwise feed only the delta — which is wrong for a self-join,
    /// and which semi-naive evaluation cannot rescue for a non-linear term).
    /// `false` for the linear single-self-reference case.
    pub naive: bool,
    /// Main query, planned with `name` resolvable as a table. Its schema is
    /// the overall query's output schema.
    pub main: LogicalPlan,
}

impl RecursiveCtePlan {
    /// Overall output schema (== the main query's schema).
    pub fn schema(&self) -> BoltResult<Schema> {
        self.main.schema()
    }
}

/// One relation of a mutually-recursive `WITH RECURSIVE` system (feature:
/// mutual recursion). Each CTE in the system has its own anchor (seed) and an
/// optional recursive term; the recursive term may reference **any** of the
/// system's CTE names (its own or a sibling's). The host engine advances all
/// of these in lockstep to a combined fixpoint (see
/// [`crate::exec::engine::Engine::execute_mutual_recursive_cte`]).
#[derive(Debug, Clone)]
pub struct RecursiveCteTerm {
    /// Case-folded CTE name (the ephemeral relation siblings + the main query
    /// scan).
    pub name: String,
    /// Accumulated-relation schema (anchor schema, optionally renamed by a
    /// column-list alias). The ephemeral table registered for this CTE each
    /// iteration has exactly this schema.
    pub cte_schema: Schema,
    /// Non-recursive seed term. Does NOT reference any CTE in the system.
    pub anchor: LogicalPlan,
    /// Recursive term, planned with every CTE name in the system bound to a
    /// synthetic `Scan` over that CTE's `cte_schema`. `None` for a
    /// non-recursive member of a `WITH RECURSIVE` list (a plain CTE that just
    /// happens to share the recursive `WITH`): such a member is seeded once
    /// and never re-derived.
    pub recursive: Option<LogicalPlan>,
    /// `true` for `UNION ALL` (append, no dedup); `false` for `UNION` (dedup
    /// the accumulated relation). Only meaningful when `recursive` is `Some`.
    pub all: bool,
}

/// A fully-planned mutually-recursive `WITH RECURSIVE` query: a *system* of
/// recursive relations advanced in lockstep, plus the main query.
///
/// This is the multi-relation generalisation of [`RecursiveCtePlan`]. It is
/// produced when a `WITH RECURSIVE` list declares more than one CTE (so a
/// recursive term may reference a sibling — mutual recursion). The host engine
/// materialises every anchor, then on each iteration binds ALL CTE names to
/// their current accumulated relations, re-evaluates every recursive term, and
/// unions the new rows into each CTE; it stops only when NO CTE grew (combined
/// fixpoint) or the shared iteration cap fires.
///
/// Kept as a standalone descriptor (not a `LogicalPlan` variant) for the same
/// reason as [`RecursiveCtePlan`]: it does not fit a single plan tree, and a
/// new enum variant would break exhaustive matches in shared files.
#[derive(Debug, Clone)]
pub struct MutualRecursiveCtePlan {
    /// The recursive system, in declaration order.
    pub ctes: Vec<RecursiveCteTerm>,
    /// Main query, planned with every CTE name resolvable as a table. Its
    /// schema is the overall query's output schema.
    pub main: LogicalPlan,
}

impl MutualRecursiveCtePlan {
    /// Overall output schema (== the main query's schema).
    pub fn schema(&self) -> BoltResult<Schema> {
        self.main.schema()
    }
}

/// Result of detecting a top-level `WITH RECURSIVE` query: either the
/// single-CTE fast path ([`RecursiveCtePlan`], possibly non-linear/naive) or a
/// multi-CTE lockstep system ([`MutualRecursiveCtePlan`]).
#[derive(Debug, Clone)]
pub enum RecursiveQueryPlan {
    /// Exactly one recursive CTE (linear or non-linear).
    Single(RecursiveCtePlan),
    /// Two or more CTEs in the `WITH RECURSIVE` list (mutual recursion).
    Mutual(MutualRecursiveCtePlan),
}

// ---------------------------------------------------------------------------
// LATERAL derived table — host nested-loop Apply (feature F3 — LATERAL)
// ---------------------------------------------------------------------------

/// Reserved ephemeral table name the per-left-row LATERAL subplan reads to pull
/// the current outer row's correlated values. Each correlated outer reference
/// in the LATERAL subquery is rewritten to a scalar subquery
/// `(SELECT __corr_<i> FROM __lateral_outer)`; the engine binds a **single-row**
/// relation under this name per left row, and `resolve_subqueries` folds those
/// scalar subqueries to the row's exact typed value. The `__` prefix keeps it
/// out of any user namespace.
pub const LATERAL_OUTER_TABLE: &str = "__lateral_outer";

/// Reserved ephemeral table name the OUTER query template
/// ([`LateralApplyPlan::post`]) scans — the host-built applied relation
/// (left columns ++ subquery columns). Mirrors
/// [`COUNT_DISTINCT_GROUPBY_RESULT_TABLE`].
pub const LATERAL_APPLY_RESULT_TABLE: &str = "__lateral_apply_result";

/// Column-name prefix for the synthetic per-row outer-value columns bound under
/// [`LATERAL_OUTER_TABLE`] (`__corr_0`, `__corr_1`, …), one per distinct
/// correlated reference.
pub const LATERAL_CORR_COL_PREFIX: &str = "__corr_";

/// Descriptor for a `FROM <left>, LATERAL (<subquery>) AS <alias>` query
/// (feature F3 — LATERAL), executed as a host-orchestrated nested-loop **Apply**
/// (dependent join) rather than a single `LogicalPlan`.
///
/// A LATERAL derived table is *correlated*: its subquery may reference columns
/// from the FROM items to its left, so it must be re-evaluated per left row with
/// those outer values bound in. There is no single plan tree for that, so —
/// exactly like [`RecursiveCtePlan`] and [`CountDistinctGroupByPlan`] — the
/// engine orchestrates it host-side (see
/// [`crate::exec::engine::Engine::execute_lateral_apply`]):
///
/// 1. Run [`Self::left`] → the left relation.
/// 2. For each left row (bounded by a mandatory safety cap), bind a single-row
///    [`LATERAL_OUTER_TABLE`] relation holding that row's correlated values
///    (columns `__corr_0..N`, in [`Self::corr_left_indices`] order) and run
///    [`Self::lateral_subplan`] — whose correlated references were rewritten to
///    `(SELECT __corr_<i> FROM __lateral_outer)` scalar subqueries that
///    `resolve_subqueries` folds to that row's exact typed value.
/// 3. Emit the per-row cross product: the left row's columns concatenated with
///    each produced subquery row. INNER LATERAL (plain comma / CROSS) drops a
///    left row with zero subquery rows; `left_join` keeps it once with the
///    subquery columns NULL-filled.
/// 4. Bind the concatenated applied relation under [`LATERAL_APPLY_RESULT_TABLE`]
///    and run [`Self::post`] (the OUTER query's projection / WHERE / GROUP BY /
///    ORDER BY / LIMIT), whose `left.col` / `alias.col` references were
///    pre-rewritten to the applied relation's bare column names.
///
/// Kept as a standalone descriptor (not a new `LogicalPlan`/`Expr` variant) so
/// no exhaustive match in a shared file breaks.
#[derive(Debug, Clone)]
pub struct LateralApplyPlan {
    /// The LEFT input: every FROM item before the LATERAL, lowered as an
    /// ordinary self-contained `LogicalPlan`.
    pub left: LogicalPlan,
    /// Output schema of [`Self::left`] (the leading columns of the applied
    /// relation).
    pub left_schema: Schema,
    /// The LATERAL subquery lowered to a `LogicalPlan` with every correlated
    /// outer reference rewritten to a `(SELECT __corr_<i> FROM __lateral_outer)`
    /// scalar subquery. Re-run per left row; its output is the subquery columns.
    pub lateral_subplan: LogicalPlan,
    /// Output schema of [`Self::lateral_subplan`] (the trailing columns of the
    /// applied relation). Computed once by running the subplan with the outer
    /// columns bound to typed NULLs.
    pub subquery_schema: Schema,
    /// Schema of the single-row [`LATERAL_OUTER_TABLE`] relation: one field
    /// `__corr_<i>` per distinct correlation, carrying the matched left
    /// column's dtype.
    pub outer_schema: Schema,
    /// For each `__corr_<i>` (in field order of [`Self::outer_schema`]), the
    /// index into [`Self::left_schema`] of the left column whose value fills it
    /// for the current row.
    pub corr_left_indices: Vec<usize>,
    /// Schema of the applied relation = [`Self::left_schema`] ++
    /// [`Self::subquery_schema`], with duplicate names disambiguated (the same
    /// `join_rename` rule the engine's JOIN output uses).
    pub combined_schema: Schema,
    /// OUTER query template: the user's projection / WHERE / GROUP BY / HAVING /
    /// ORDER BY / LIMIT, lowered over a synthetic `Scan` of
    /// [`LATERAL_APPLY_RESULT_TABLE`] (= [`Self::combined_schema`]).
    pub post: LogicalPlan,
    /// `true` for `LEFT JOIN LATERAL (...) ON true` (keep a left row with no
    /// subquery match, subquery columns NULL); `false` for INNER (plain comma /
    /// CROSS) LATERAL (drop such a left row).
    pub left_join: bool,
}

impl LateralApplyPlan {
    /// Overall output schema (== the OUTER template's schema).
    pub fn schema(&self) -> BoltResult<Schema> {
        self.post.schema()
    }
}

// ---------------------------------------------------------------------------
// Correlated WHERE subquery — host per-row semi/anti/scalar Apply
// (feature F4 — correlated EXISTS / NOT EXISTS / scalar subquery in WHERE)
// ---------------------------------------------------------------------------

/// Reserved ephemeral table name the OUTER projection template
/// ([`CorrelatedWherePlan::post`]) scans — the host-built relation of the
/// *surviving* outer rows (== [`CorrelatedWherePlan::left_schema`]). Mirrors
/// [`LATERAL_APPLY_RESULT_TABLE`]. The leading `__` keeps it out of any user
/// namespace.
pub const CORR_WHERE_RESULT_TABLE: &str = "__corr_where_result";

/// What per-outer-row test a [`CorrelatedWherePlan`] applies. Re-exported from
/// [`crate::plan::subquery`] so the engine and explain formatter can match on
/// it without depending on that module directly.
pub use crate::plan::subquery::CorrWhereKind;

/// Descriptor for a single `SELECT` whose WHERE contains exactly one
/// **correlated** subquery (feature F4), executed as a host-orchestrated
/// per-outer-row Apply (a correlated semi/anti-join, or a correlated
/// scalar-compare) rather than a single [`LogicalPlan`].
///
/// Like [`LateralApplyPlan`], this is a standalone descriptor (not a new
/// `LogicalPlan`/`Expr` variant) so no exhaustive match in a shared file
/// breaks. The engine (see
/// [`crate::exec::engine::Engine::execute_correlated_where`]):
///
/// 1. Runs [`Self::left`] → the outer relation (the FROM + every *ordinary*
///    (uncorrelated) WHERE conjunct applied as a normal `Filter`).
/// 2. For each outer row (bounded by the SAME mandatory cap as LATERAL —
///    [`crate::exec::engine::MAX_APPLY_LEFT_ROWS`] /
///    [`crate::exec::engine::MAX_APPLY_LEFT_ROWS_ENV`]) binds a single-row
///    [`LATERAL_OUTER_TABLE`] relation holding that row's correlated values
///    (`__corr_0..N`, in [`Self::corr_left_indices`] order) and:
///    * for [`CorrWhereKind::Exists`] / [`CorrWhereKind::NotExists`]: runs
///      [`Self::test_subplan`] and keeps / drops the row on its row count
///      (>= 1 for EXISTS, == 0 for NOT EXISTS — never NULL);
///    * for [`CorrWhereKind::Scalar`]: runs [`Self::test_subplan`] (a
///      `SELECT <the whole conjunct, outer refs and the inner scalar subquery
///      rewritten> FROM __lateral_outer`) which `resolve_subqueries` folds to a
///      single boolean — the row is kept iff that boolean is TRUE (a NULL /
///      FALSE drops it, matching WHERE 3VL; a scalar subquery returning >1 row
///      errors via the ordinary scalar-fold path).
/// 3. Concatenates the surviving outer rows, binds them under
///    [`CORR_WHERE_RESULT_TABLE`] and runs [`Self::post`] (the user's
///    projection / GROUP BY / HAVING / ORDER BY / LIMIT).
#[derive(Debug, Clone)]
pub struct CorrelatedWherePlan {
    /// Outer (LEFT) relation: the FROM tree with every ordinary uncorrelated
    /// WHERE conjunct already applied as a `Filter`.
    pub left: LogicalPlan,
    /// Output schema of [`Self::left`].
    pub left_schema: Schema,
    /// The per-row test subplan. For EXISTS/NOT EXISTS this is the correlated
    /// subquery with its outer references rewritten to
    /// `(SELECT __corr_<i> FROM __lateral_outer)`; its *row count* is the test.
    /// For the scalar kind this is `SELECT <conjunct> FROM __lateral_outer`
    /// with both the outer references AND the inner scalar subquery's own
    /// correlations rewritten away — its single boolean cell is the test.
    pub test_subplan: LogicalPlan,
    /// Schema of the single-row [`LATERAL_OUTER_TABLE`] relation: one
    /// `__corr_<i>` field per distinct correlation, carrying the matched outer
    /// column's dtype.
    pub outer_schema: Schema,
    /// For each `__corr_<i>` (in field order of [`Self::outer_schema`]), the
    /// index into [`Self::left_schema`] of the outer column whose value fills
    /// it for the current row.
    pub corr_left_indices: Vec<usize>,
    /// Which per-row test to apply.
    pub kind: CorrWhereKind,
    /// OUTER projection template: the user's projection / GROUP BY / HAVING /
    /// ORDER BY / LIMIT lowered over a synthetic `Scan` of
    /// [`CORR_WHERE_RESULT_TABLE`] (== [`Self::left_schema`]).
    pub post: LogicalPlan,
}

impl CorrelatedWherePlan {
    /// Overall output schema (== the OUTER template's schema).
    pub fn schema(&self) -> BoltResult<Schema> {
        self.post.schema()
    }
}

// ---------------------------------------------------------------------------
// VALUES as a row source (feature: VALUES)
// ---------------------------------------------------------------------------

/// Reserved ephemeral table name the OUTER query template
/// ([`ValuesQueryPlan::post`]) scans — the host-materialised VALUES relation
/// (for the `FROM (VALUES ...) AS t` form, this is *also* bound under the
/// user-supplied alias; the reserved name below is used only when the engine
/// needs a name and the user gave none, which cannot happen for the FROM form).
/// The leading `__` keeps it out of any user namespace.
pub const VALUES_RESULT_TABLE: &str = "__values_result";

/// Default per-process cap on the number of `VALUES` rows materialised, to keep
/// a giant literal blob (`VALUES (1),(2),...,(10^9)`) from allocating without
/// bound host-side. Overridable via [`VALUES_MAX_ROWS_ENV`].
pub const VALUES_MAX_ROWS: usize = 1_000_000;

/// Environment variable that overrides [`VALUES_MAX_ROWS`] at runtime. Parsed as
/// a base-10 `usize`; `0` / empty / unparseable fall back to the default.
pub const VALUES_MAX_ROWS_ENV: &str = "CRATON_VALUES_MAX_ROWS";

/// Resolve the `VALUES` row cap, honouring [`VALUES_MAX_ROWS_ENV`]. Re-parsed on
/// each call (cheap: one env lookup) — there is no hot loop here, and avoiding a
/// process-global latch keeps the cap test-overridable per process.
fn values_max_rows() -> usize {
    match std::env::var(VALUES_MAX_ROWS_ENV) {
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(0) | Err(_) => VALUES_MAX_ROWS,
            Ok(n) => n,
        },
        Err(_) => VALUES_MAX_ROWS,
    }
}

/// Enforce the `VALUES` row cap. Kept pure (the cap is passed in) so it is
/// testable without mutating the process-global [`VALUES_MAX_ROWS_ENV`] —
/// mutating that env var in a test would race the other parallel VALUES tests
/// that read the cap.
fn enforce_values_row_cap(n_rows: usize, cap: usize) -> BoltResult<()> {
    if n_rows > cap {
        return Err(BoltError::Sql(format!(
            "VALUES has {n_rows} rows, exceeding the {cap}-row cap (override via {VALUES_MAX_ROWS_ENV})"
        )));
    }
    Ok(())
}

/// A host-materialisable `VALUES` relation: an inferred [`Schema`] plus the
/// per-row constant values (one [`Literal`] per column, row-major). The engine
/// turns this into an Arrow `RecordBatch` (see
/// [`crate::exec::engine::Engine::execute_values_query`]).
///
/// Type inference (see [`lower_values_relation`]): a column's dtype is the
/// common type across all rows. Int32 promotes to Int64 when mixed; NULL takes
/// the column type from the other rows; an all-NULL column defaults to a
/// nullable `Int64` (documented choice — the widest integer is the least
/// surprising default and is GPU-friendly). Genuinely incompatible mixes (e.g.
/// Int vs Utf8) are a clean `BoltError`.
#[derive(Debug, Clone)]
pub struct ValuesRelation {
    /// Inferred output schema (column names + common-type dtypes + nullability).
    pub schema: Schema,
    /// Row-major literal values; `rows[r][c]` is row `r`, column `c`. Each row
    /// has exactly `schema.fields.len()` entries; `Literal::Null` marks a NULL.
    /// Every non-null value is already coerced to the column's common dtype.
    pub rows: Vec<Vec<Literal>>,
}

/// Descriptor for a query whose row source is a `VALUES` clause (feature
/// VALUES), executed as a host-orchestrated engine special-case rather than a
/// plan node (there is no `LogicalPlan`/`Expr` variant for an inline relation,
/// and adding one would touch shared files and break exhaustive matches).
///
/// Two shapes (see [`plan_values_query`]):
///   * **Bare** `VALUES (..),(..) [ORDER BY ..] [LIMIT ..]` — `post` is `None`;
///     the engine returns the materialised relation directly (after the
///     optional ORDER BY / LIMIT carried in `post_bare`).
///   * **FROM form** `SELECT ... FROM (VALUES ..) AS t(a,b) ...` — `post` is the
///     outer query lowered over a synthetic `Scan` of [`Self::bind_name`]; the
///     engine binds the materialised relation under that name in the streaming
///     overlay and runs `post` through the ordinary subplan executor (mirroring
///     the WITH RECURSIVE / LATERAL overlay pattern).
#[derive(Debug, Clone)]
pub struct ValuesQueryPlan {
    /// The materialisable VALUES relation (inferred schema + constant rows).
    pub relation: ValuesRelation,
    /// Name the relation is bound under for the FROM form (the user alias `t`);
    /// for the bare form this is [`VALUES_RESULT_TABLE`] (used only if a `post`
    /// scan is ever built, which the bare form does not do).
    pub bind_name: String,
    /// FROM form: the outer query lowered over a synthetic `Scan` of
    /// [`Self::bind_name`]. `None` for the bare top-level VALUES form.
    pub post: Option<LogicalPlan>,
    /// Bare form: optional ORDER BY applied to the materialised relation.
    pub order_by: Vec<SortExpr>,
    /// Bare form: optional `(limit, offset)` applied after the ORDER BY.
    pub limit: Option<(usize, usize)>,
}

impl ValuesQueryPlan {
    /// Overall output schema.
    pub fn schema(&self) -> BoltResult<Schema> {
        match &self.post {
            Some(p) => p.schema(),
            None => Ok(self.relation.schema.clone()),
        }
    }
}

/// Promote two column dtypes to their common type for VALUES inference. Returns
/// `None` when the two types are genuinely incompatible (e.g. Int vs Utf8).
///
/// Rules (intentionally conservative — only the promotions the executor can
/// build cheaply): identical types collapse to themselves; Int32↔Int64 →
/// Int64; Float32↔Float64 → Float64; Int32/Int64 ↔ Float32/Float64 → Float64
/// (integers widen into the float column). Everything else (Bool, Utf8,
/// temporal, decimal mixed with a different family) requires an exact match.
fn values_common_type(a: DataType, b: DataType) -> Option<DataType> {
    use DataType::*;
    if a == b {
        return Some(a);
    }
    Some(match (a, b) {
        (Int32, Int64) | (Int64, Int32) => Int64,
        (Float32, Float64) | (Float64, Float32) => Float64,
        (Int32, Float32) | (Float32, Int32) => Float32,
        (Int32, Float64)
        | (Float64, Int32)
        | (Int64, Float32)
        | (Float32, Int64)
        | (Int64, Float64)
        | (Float64, Int64) => Float64,
        _ => return None,
    })
}

/// Coerce a folded literal to the column's inferred common dtype. NULL passes
/// through unchanged (it carries the column type implicitly). Widening numeric
/// promotions (Int32→Int64, Int→Float, Float32→Float64) are applied so the
/// engine builds one typed array per column. A literal that does not match and
/// cannot widen is an error (should not happen — `values_common_type` already
/// validated compatibility — but kept as a clean guard).
fn coerce_values_literal(lit: Literal, target: DataType) -> BoltResult<Literal> {
    use DataType as D;
    Ok(match (&lit, target) {
        (Literal::Null, _) => Literal::Null,
        // Exact-type fast paths.
        (Literal::Int32(_), D::Int32)
        | (Literal::Int64(_), D::Int64)
        | (Literal::Float32(_), D::Float32)
        | (Literal::Float64(_), D::Float64)
        | (Literal::Bool(_), D::Bool)
        | (Literal::Utf8(_), D::Utf8) => lit,
        // Integer widening.
        (Literal::Int32(v), D::Int64) => Literal::Int64(*v as i64),
        // Integer → float.
        (Literal::Int32(v), D::Float32) => Literal::Float32(*v as f32),
        (Literal::Int32(v), D::Float64) => Literal::Float64(*v as f64),
        (Literal::Int64(v), D::Float64) => Literal::Float64(*v as f64),
        (Literal::Int64(v), D::Float32) => Literal::Float32(*v as f32),
        // Float widening.
        (Literal::Float32(v), D::Float64) => Literal::Float64(*v as f64),
        _ => {
            return Err(BoltError::Sql(format!(
                "VALUES: cannot coerce literal {lit:?} to inferred column type {target:?}"
            )))
        }
    })
}

/// Lower a parsed `VALUES` row list into a host-materialisable [`ValuesRelation`]
/// (feature VALUES).
///
/// Steps:
///   1. Reject an empty VALUES list and ragged rows (differing column counts).
///   2. Enforce the row cap ([`values_max_rows`]).
///   3. Fold each cell to a constant via [`lower_expr`] over an *empty* resolver
///      (so a column reference / non-constant errors cleanly) and require an
///      `Expr::Literal`.
///   4. Infer each column's common dtype across all rows via
///      [`values_common_type`]; NULLs do not constrain the type. An all-NULL
///      column defaults to nullable `Int64` (documented).
///   5. Coerce every non-NULL cell to its column's common dtype.
///
/// `col_names` supplies the output column names; when `None`, Postgres-style
/// `column1, column2, ...` defaults are used.
fn lower_values_relation(
    values: &sqlparser::ast::Values,
    col_names: Option<&[String]>,
) -> BoltResult<ValuesRelation> {
    if values.rows.is_empty() {
        return Err(BoltError::Sql("VALUES requires at least one row".into()));
    }
    let n_cols = values.rows[0].len();
    if n_cols == 0 {
        return Err(BoltError::Sql("VALUES row must have at least one column".into()));
    }
    enforce_values_row_cap(values.rows.len(), values_max_rows())?;

    // Fold every cell to a literal; reject ragged rows.
    let empty = NameResolver::empty();
    let mut lits: Vec<Vec<Literal>> = Vec::with_capacity(values.rows.len());
    for (ri, row) in values.rows.iter().enumerate() {
        if row.len() != n_cols {
            return Err(BoltError::Sql(format!(
                "VALUES row {} has {} columns but row 1 has {n_cols} (ragged VALUES)",
                ri + 1,
                row.len()
            )));
        }
        let mut row_lits = Vec::with_capacity(n_cols);
        for (ci, cell) in row.iter().enumerate() {
            let lowered = lower_expr(cell, &empty, 0)?;
            match lowered {
                Expr::Literal(l) => row_lits.push(l),
                _ => {
                    return Err(BoltError::Sql(format!(
                        "VALUES cell (row {}, column {}) is not a constant expression; \
                         only literals / constant-folding expressions are supported",
                        ri + 1,
                        ci + 1
                    )))
                }
            }
        }
        lits.push(row_lits);
    }

    // Infer each column's common dtype across all rows (NULLs don't constrain).
    let mut col_types: Vec<Option<DataType>> = vec![None; n_cols];
    for row in &lits {
        for (ci, lit) in row.iter().enumerate() {
            if let Some(dt) = lit.dtype() {
                col_types[ci] = match col_types[ci] {
                    None => Some(dt),
                    Some(prev) => Some(values_common_type(prev, dt).ok_or_else(|| {
                        BoltError::Sql(format!(
                            "VALUES column {} mixes incompatible types {prev:?} and {dt:?}",
                            ci + 1
                        ))
                    })?),
                };
            }
        }
    }

    // A column-list alias (`AS v(a, b, ...)`) must name exactly as many columns
    // as the VALUES rows have — otherwise reject cleanly (indexing `names[ci]`
    // below would otherwise panic on a short list).
    if let Some(names) = col_names {
        if names.len() != n_cols {
            return Err(BoltError::Sql(format!(
                "VALUES alias column list has {} name(s) but the rows have {} column(s)",
                names.len(),
                n_cols
            )));
        }
    }

    // Build the schema. An all-NULL column (col_types[ci] == None) defaults to
    // a nullable Int64 (documented). A column is nullable iff any row is NULL.
    let mut fields = Vec::with_capacity(n_cols);
    for ci in 0..n_cols {
        let dtype = col_types[ci].unwrap_or(DataType::Int64);
        let nullable = lits.iter().any(|r| matches!(r[ci], Literal::Null));
        let name = match col_names {
            Some(names) => names[ci].clone(),
            None => format!("column{}", ci + 1),
        };
        fields.push(Field::new(name, dtype, nullable));
    }
    let schema = Schema::new(fields);

    // Coerce every cell to its column's common dtype.
    for row in &mut lits {
        for ci in 0..n_cols {
            let target = schema.fields[ci].dtype;
            let coerced = coerce_values_literal(row[ci].clone(), target)?;
            row[ci] = coerced;
        }
    }

    Ok(ValuesRelation { schema, rows: lits })
}

/// Detect and plan a top-level query whose row source is a `VALUES` clause
/// (feature VALUES). Returns `Ok(Some(plan))` for the two supported shapes (bare
/// `VALUES ...` and `SELECT ... FROM (VALUES ...) AS t(a,b) ...`), `Ok(None)`
/// for everything else (so the engine falls through to the ordinary pipeline),
/// and `Err` for a VALUES of the right shape that is malformed (ragged rows,
/// incompatible types, non-constant cells, row-cap overflow).
pub fn plan_values_query(
    sql: &str,
    provider: &dyn TableProvider,
) -> BoltResult<Option<ValuesQueryPlan>> {
    guard_sql_size(sql)?;
    let dialect = GenericDialect {};
    let mut stmts =
        Parser::parse_sql(&dialect, sql).map_err(|e| parse_error_to_bolt_error(e, sql))?;
    if stmts.len() != 1 {
        return Ok(None);
    }
    let stmt = stmts.remove(0);
    let query = match stmt {
        Statement::Query(q) => q,
        _ => return Ok(None),
    };
    plan_values_query_inner(&query, provider)
}

fn plan_values_query_inner(
    query: &Query,
    provider: &dyn TableProvider,
) -> BoltResult<Option<ValuesQueryPlan>> {
    // No top-level WITH (a CTE wrapping VALUES is out of scope for this hook).
    if query.with.is_some() {
        return Ok(None);
    }

    // Shape 1: bare top-level `VALUES ...` (the body IS a Values node).
    if let SetExpr::Values(values) = query.body.as_ref() {
        let relation = lower_values_relation(values, None)?;
        let ctes = CteScope::new();
        // Optional ORDER BY / LIMIT over the bare relation's schema.
        let order_by = match &query.order_by {
            Some(ob) => lower_order_by(
                &ob.exprs,
                SubqueryCtx { provider, ctes: &ctes },
            )
            .and_then(|sorts| {
                // Type-check the sort refs against the relation schema.
                validate_sort_columns(&sorts, &relation.schema)?;
                Ok(sorts)
            })?,
            None => Vec::new(),
        };
        let limit = lower_bare_limit(query)?;
        return Ok(Some(ValuesQueryPlan {
            relation,
            bind_name: VALUES_RESULT_TABLE.to_string(),
            post: None,
            order_by,
            limit,
        }));
    }

    // Shape 2: `SELECT ... FROM (VALUES ...) AS t(cols) ...` — exactly one FROM
    // item, a non-lateral Derived whose subquery body is a bare VALUES, with an
    // alias. Anything more complex (joins onto the VALUES, multiple FROM items,
    // a WITH inside the derived subquery) declines to `Ok(None)` so the ordinary
    // pipeline handles / rejects it.
    let select = match query.body.as_ref() {
        SetExpr::Select(s) => s.as_ref(),
        _ => return Ok(None),
    };
    if select.from.len() != 1 || !select.from[0].joins.is_empty() {
        return Ok(None);
    }
    let (subquery, alias) = match &select.from[0].relation {
        TableFactor::Derived {
            lateral: false,
            subquery,
            alias: Some(alias),
        } => (subquery, alias),
        _ => return Ok(None),
    };
    let values = match subquery.body.as_ref() {
        SetExpr::Values(v) if subquery.with.is_none() && subquery.order_by.is_none() => v,
        _ => return Ok(None),
    };

    // Column names from the alias column list `AS t(a, b)` if present; else
    // Postgres-style `column1, column2, ...`.
    let alias_cols: Option<Vec<String>> = if alias.columns.is_empty() {
        None
    } else {
        Some(alias.columns.iter().map(ident_to_name).collect())
    };
    let relation = lower_values_relation(values, alias_cols.as_deref())?;
    // If an alias column list was supplied it must match the column count.
    if let Some(cols) = &alias_cols {
        if cols.len() != relation.schema.fields.len() {
            return Err(BoltError::Sql(format!(
                "VALUES alias column list has {} names but the relation has {} columns",
                cols.len(),
                relation.schema.fields.len()
            )));
        }
    }
    let bind_name = ident_to_name(&alias.name);

    // Build the outer query (projection / WHERE / GROUP BY / ORDER BY / LIMIT)
    // over a synthetic Scan of the bound relation. We rebuild the query with the
    // FROM rewritten to a bare table reference to `bind_name`, then lower it
    // through the ordinary `plan_query` so every SELECT feature is reused.
    let post = build_values_post_plan(query, &bind_name, &relation.schema, provider)?;
    Ok(Some(ValuesQueryPlan {
        relation,
        bind_name,
        post: Some(post),
        order_by: Vec::new(),
        limit: None,
    }))
}

/// Lower the outer query of a `FROM (VALUES ...) AS t` form into a `post`
/// [`LogicalPlan`] over a synthetic `Scan` of the bound relation. The synthetic
/// scan is registered under `bind_name` in the engine's streaming overlay before
/// the post-plan runs (mirroring [`build_count_distinct_post_plan`]).
fn build_values_post_plan(
    query: &Query,
    bind_name: &str,
    relation_schema: &Schema,
    provider: &dyn TableProvider,
) -> BoltResult<LogicalPlan> {
    // Rewrite the FROM so the single derived (VALUES) item becomes a bare table
    // reference to `bind_name`. We clone the query, swap the relation, and lower
    // the whole thing with the relation schema registered as a base table via a
    // schema-overlay provider — reusing the entire SELECT lowering path.
    let mut rewritten = query.clone();
    if let SetExpr::Select(select) = rewritten.body.as_mut() {
        let from0 = &mut select.from[0];
        from0.relation = TableFactor::Table {
            name: ObjectName(vec![Ident::new(bind_name.to_string())]),
            alias: None,
            args: None,
            with_hints: Vec::new(),
            version: None,
            with_ordinality: false,
            partitions: Vec::new(),
        };
    }
    // Provider overlay: the bound relation appears as a base table named
    // `bind_name` so `lower_table_factor` resolves the scan against it.
    let overlay = SchemaOverlayProvider {
        base: provider,
        name: bind_name.to_string(),
        schema: relation_schema.clone(),
    };
    let ctes = CteScope::new();
    let plan = plan_query(&rewritten, &overlay, &ctes, 0)?;
    let _ = plan.schema()?;
    Ok(plan)
}

/// A [`TableProvider`] that overlays a single named relation schema on top of a
/// base provider — used to lower a `post` plan over a host-materialised relation
/// (VALUES / DISTINCT ON) whose schema is known at plan time but whose batch is
/// bound at execution time. The base provider is consulted for every other name.
struct SchemaOverlayProvider<'a> {
    base: &'a dyn TableProvider,
    name: String,
    schema: Schema,
}

impl<'a> TableProvider for SchemaOverlayProvider<'a> {
    fn schema(&self, name: &str) -> BoltResult<Schema> {
        if name == self.name || name.eq_ignore_ascii_case(&self.name) {
            return Ok(self.schema.clone());
        }
        self.base.schema(name)
    }

    fn has_nulls(&self, table_name: &str, col_idx: usize) -> bool {
        if table_name == self.name || table_name.eq_ignore_ascii_case(&self.name) {
            return self
                .schema
                .fields
                .get(col_idx)
                .map(|f| f.nullable)
                .unwrap_or(false);
        }
        self.base.has_nulls(table_name, col_idx)
    }
}

// ---------------------------------------------------------------------------
// generate_series(start, stop[, step]) as a row source (feature GENERATE_SERIES)
// ---------------------------------------------------------------------------

/// Default PostgreSQL column name a `generate_series` relation exposes when the
/// query does not supply a column-list alias (`AS t(n)`). Also the default
/// relation name when no table alias is given (so `generate_series.generate_series`
/// and a bare `generate_series` both resolve).
pub const GENERATE_SERIES_DEFAULT_NAME: &str = "generate_series";

/// Default per-process cap on the number of rows a `generate_series` may
/// materialise. A series like `generate_series(1, 10^18)` is astronomically
/// large; without a cap it would attempt an unbounded host allocation. The cap
/// is checked with the *computed* row count (checked arithmetic) before any
/// allocation. Overridable via [`GENERATE_SERIES_MAX_ROWS_ENV`].
pub const GENERATE_SERIES_MAX_ROWS: usize = 10_000_000;

/// Environment variable overriding [`GENERATE_SERIES_MAX_ROWS`] at runtime.
/// Parsed as a base-10 `usize`; `0` / empty / unparseable fall back to default.
pub const GENERATE_SERIES_MAX_ROWS_ENV: &str = "CRATON_GENERATE_SERIES_MAX_ROWS";

/// Resolve the `generate_series` row cap, honouring
/// [`GENERATE_SERIES_MAX_ROWS_ENV`]. Re-parsed on each call (cheap env lookup,
/// no hot loop) so the cap stays per-process overridable without a global latch.
fn generate_series_max_rows() -> usize {
    match std::env::var(GENERATE_SERIES_MAX_ROWS_ENV) {
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(0) | Err(_) => GENERATE_SERIES_MAX_ROWS,
            Ok(n) => n,
        },
        Err(_) => GENERATE_SERIES_MAX_ROWS,
    }
}

/// Enforce the `generate_series` row cap. Kept PURE (the cap is passed in) so it
/// is testable without mutating the process-global [`GENERATE_SERIES_MAX_ROWS_ENV`]
/// — mutating that env var in a test would race the other parallel tests reading
/// the cap. Mirrors [`enforce_values_row_cap`].
fn enforce_generate_series_row_cap(n_rows: u64, cap: usize) -> BoltResult<()> {
    if n_rows > cap as u64 {
        return Err(BoltError::Sql(format!(
            "generate_series would produce {n_rows} rows, exceeding the {cap}-row cap \
             (override via {GENERATE_SERIES_MAX_ROWS_ENV})"
        )));
    }
    Ok(())
}

/// Number of rows an integer `generate_series(start, stop, step)` emits, computed
/// with checked arithmetic so endpoints / steps near `i64::MIN`/`i64::MAX` cannot
/// overflow. Returns:
///   * `Err` if `step == 0` (PostgreSQL: "step size cannot be zero").
///   * `Ok(0)` for an empty direction (start > stop with step > 0, or
///     start < stop with step < 0).
///   * `Ok(count)` otherwise, where the count is `(stop - start) / step + 1`
///     evaluated in `i128` (so the span `stop - start`, which can exceed
///     `i64::MAX` in magnitude, never overflows).
///
/// Pure: takes the three bounds, no I/O, no allocation — host-testable directly.
fn generate_series_row_count(start: i64, stop: i64, step: i64) -> BoltResult<u64> {
    if step == 0 {
        return Err(BoltError::Sql(
            "generate_series: step size cannot be zero".into(),
        ));
    }
    // Empty direction → 0 rows (not an error), matching PostgreSQL.
    if (step > 0 && start > stop) || (step < 0 && start < stop) {
        return Ok(0);
    }
    // span and the +1 are exact in i128 (i64 range is far inside i128).
    let span = (stop as i128) - (start as i128);
    let step128 = step as i128;
    // span and step have the same sign here, so the quotient is >= 0.
    let count = span / step128 + 1;
    // count is at most |i64::MIN..i64::MAX| + 1 = 2^64, which fits i128; clamp
    // into u64 for the cap comparison (anything beyond u64 trivially exceeds the
    // cap, but we never reach the cast unscathed because count <= 2^64).
    Ok(u64::try_from(count).unwrap_or(u64::MAX))
}

/// Build the `Vec<i64>` of series values for `generate_series(start, stop, step)`.
/// `n_rows` is the pre-computed (and cap-checked) row count from
/// [`generate_series_row_count`]. Values are produced with checked addition so a
/// final step that would overflow `i64` cannot wrap (it simply does not extend
/// past the last representable value — but `n_rows` already bounds the loop, so
/// overflow can only be the unreachable final increment). Pure / host-testable.
fn generate_series_values(start: i64, step: i64, n_rows: u64) -> Vec<i64> {
    let mut out = Vec::with_capacity(n_rows as usize);
    let mut cur = start;
    for _ in 0..n_rows {
        out.push(cur);
        // Checked: on the very last iteration `cur + step` may overflow i64
        // (e.g. start near i64::MAX). The loop ends after this push regardless,
        // so a saturating fallback is harmless and never observed.
        cur = cur.checked_add(step).unwrap_or(cur);
    }
    out
}

/// A host-materialisable integer `generate_series` relation: the resolved column
/// name plus the dense `Int64` series values (non-nullable). The engine turns
/// this into a single-column Arrow `RecordBatch`.
#[derive(Debug, Clone)]
pub struct GenerateSeriesRelation {
    /// Output column name (the column-list alias if present, else
    /// [`GENERATE_SERIES_DEFAULT_NAME`]).
    pub column_name: String,
    /// The series values, already bounded by the row cap. Non-nullable `Int64`.
    pub values: Vec<i64>,
}

impl GenerateSeriesRelation {
    /// Single-column, non-nullable `Int64` schema for this relation.
    pub fn schema(&self) -> Schema {
        Schema::new(vec![Field::new(
            self.column_name.clone(),
            DataType::Int64,
            false,
        )])
    }
}

/// Descriptor for a query whose row source is an integer `generate_series` TVF
/// (feature GENERATE_SERIES), executed as a host-orchestrated engine special-case
/// rather than a plan node — mirroring [`ValuesQueryPlan`] (an inline relation
/// has no `LogicalPlan`/`Expr` variant, and adding one would touch shared files
/// and break exhaustive matches).
///
/// The series is materialised host-side, bound under [`Self::bind_name`] in the
/// streaming overlay, and the outer query template [`Self::post`] (the SELECT /
/// WHERE / ORDER BY / LIMIT lowered over a synthetic `Scan` of that name) runs
/// through the ordinary subplan executor.
#[derive(Debug, Clone)]
pub struct GenerateSeriesQueryPlan {
    /// The materialisable series relation (column name + Int64 values).
    pub relation: GenerateSeriesRelation,
    /// Name the relation is bound under (the table alias `t`, or
    /// [`GENERATE_SERIES_DEFAULT_NAME`] when none was given).
    pub bind_name: String,
    /// The outer query lowered over a synthetic `Scan` of [`Self::bind_name`].
    pub post: LogicalPlan,
}

impl GenerateSeriesQueryPlan {
    /// Overall output schema (the post plan's schema).
    pub fn schema(&self) -> BoltResult<Schema> {
        self.post.schema()
    }
}

/// Fold a single `generate_series` argument to a constant `i64`. Reuses the
/// VALUES literal-lowering path (`lower_expr` over an empty resolver) so a
/// constant integer expression folds and any column reference / non-constant
/// errors cleanly. `Int32` widens to `Int64`. Non-integer constants (float,
/// string, bool, NULL) are rejected — only integer bounds are supported.
fn fold_generate_series_arg(expr: &SqlExpr, which: &str) -> BoltResult<i64> {
    let empty = NameResolver::empty();
    let lowered = lower_expr(expr, &empty, 0)?;
    match lowered {
        Expr::Literal(Literal::Int64(v)) => Ok(v),
        Expr::Literal(Literal::Int32(v)) => Ok(v as i64),
        Expr::Literal(other) => Err(BoltError::Sql(format!(
            "generate_series: {which} argument must be an integer constant, got {other:?}"
        ))),
        _ => Err(BoltError::Sql(format!(
            "generate_series: {which} argument must be a constant integer expression; \
             a column-correlated generate_series is not supported"
        ))),
    }
}

/// Extract the bare `Expr` of an unnamed positional function argument, rejecting
/// named args (`=> x`) and `*` / qualified-wildcard forms (`generate_series(*)`).
fn generate_series_arg_expr(arg: &FunctionArg) -> BoltResult<&SqlExpr> {
    match arg {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => Ok(e),
        _ => Err(BoltError::Sql(
            "generate_series: arguments must be positional integer expressions".into(),
        )),
    }
}

/// Detect and plan a top-level query whose single row source is an integer
/// `generate_series(start, stop[, step])` TVF (feature GENERATE_SERIES).
///
/// Returns `Ok(Some(plan))` for the supported shape
/// (`SELECT ... FROM generate_series(...) [AS t(n)] [WHERE/ORDER BY/LIMIT]`),
/// `Ok(None)` for every other query (so the engine falls through to the ordinary
/// pipeline, which still rejects an unhandled TVF in `lower_table_factor`), and
/// `Err` for a `generate_series` of the right shape that is malformed (wrong arg
/// count, non-integer / non-constant args, `step == 0`, or a series exceeding the
/// row cap).
pub fn plan_generate_series_query(
    sql: &str,
    provider: &dyn TableProvider,
) -> BoltResult<Option<GenerateSeriesQueryPlan>> {
    guard_sql_size(sql)?;
    let dialect = GenericDialect {};
    let mut stmts =
        Parser::parse_sql(&dialect, sql).map_err(|e| parse_error_to_bolt_error(e, sql))?;
    if stmts.len() != 1 {
        return Ok(None);
    }
    let query = match stmts.remove(0) {
        Statement::Query(q) => q,
        _ => return Ok(None),
    };
    plan_generate_series_query_inner(&query, provider)
}

fn plan_generate_series_query_inner(
    query: &Query,
    provider: &dyn TableProvider,
) -> BoltResult<Option<GenerateSeriesQueryPlan>> {
    // No top-level WITH (a CTE wrapping generate_series is out of scope here).
    if query.with.is_some() {
        return Ok(None);
    }
    // Exactly one SELECT body with a single, join-free FROM item.
    let select = match query.body.as_ref() {
        SetExpr::Select(s) => s.as_ref(),
        _ => return Ok(None),
    };
    if select.from.len() != 1 || !select.from[0].joins.is_empty() {
        return Ok(None);
    }
    // The FROM item must be a function-style table factor named generate_series.
    let (name, alias, fn_args) = match &select.from[0].relation {
        TableFactor::Table {
            name,
            alias,
            args: Some(args),
            with_hints,
            version,
            with_ordinality,
            partitions,
        } if with_hints.is_empty()
            && version.is_none()
            && !*with_ordinality
            && partitions.is_empty() =>
        {
            (name, alias, &args.args)
        }
        // A function-style factor with hints / version / WITH ORDINALITY /
        // PARTITION is out of scope here; decline so the ordinary pipeline
        // surfaces its own (more specific) rejection.
        _ => return Ok(None),
    };
    // Match the function name case-insensitively; decline other TVFs so they keep
    // their existing rejection path.
    let fn_name = match single_ident_from_object_name(name) {
        Ok(n) => n,
        Err(_) => return Ok(None),
    };
    if !fn_name.eq_ignore_ascii_case(GENERATE_SERIES_DEFAULT_NAME) {
        return Ok(None);
    }

    // From here the shape IS generate_series: malformed forms are clean errors.
    if fn_args.len() != 2 && fn_args.len() != 3 {
        return Err(BoltError::Sql(format!(
            "generate_series expects 2 or 3 integer arguments (start, stop[, step]), got {}",
            fn_args.len()
        )));
    }
    let start = fold_generate_series_arg(generate_series_arg_expr(&fn_args[0])?, "start")?;
    let stop = fold_generate_series_arg(generate_series_arg_expr(&fn_args[1])?, "stop")?;
    let step = if fn_args.len() == 3 {
        fold_generate_series_arg(generate_series_arg_expr(&fn_args[2])?, "step")?
    } else {
        1
    };

    // Checked row count + cap (pure helpers; cap passed in for test purity).
    let n_rows = generate_series_row_count(start, stop, step)?;
    enforce_generate_series_row_cap(n_rows, generate_series_max_rows())?;
    let values = generate_series_values(start, step, n_rows);

    // Naming: the table alias names the relation (else `generate_series`); the
    // column-list alias names the column (else the PostgreSQL default
    // `generate_series`). `AS t(n)` → relation `t`, column `n`.
    let (bind_name, column_name) = match alias {
        None => (
            GENERATE_SERIES_DEFAULT_NAME.to_string(),
            GENERATE_SERIES_DEFAULT_NAME.to_string(),
        ),
        Some(a) => {
            let rel = ident_to_name(&a.name);
            let col = match a.columns.len() {
                0 => GENERATE_SERIES_DEFAULT_NAME.to_string(),
                1 => ident_to_name(&a.columns[0]),
                n => {
                    return Err(BoltError::Sql(format!(
                        "generate_series produces a single column but the alias column list \
                         has {n} names"
                    )))
                }
            };
            (rel, col)
        }
    };

    let relation = GenerateSeriesRelation {
        column_name,
        values,
    };
    let schema = relation.schema();
    let post = build_generate_series_post_plan(query, &bind_name, &schema, provider)?;
    Ok(Some(GenerateSeriesQueryPlan {
        relation,
        bind_name,
        post,
    }))
}

/// Lower the outer query of a `FROM generate_series(...)` form into a `post`
/// [`LogicalPlan`] over a synthetic `Scan` of the bound relation, mirroring
/// [`build_values_post_plan`]: rewrite the function-style FROM item into a bare
/// table reference to `bind_name`, then lower the whole query through
/// [`plan_query`] over a [`SchemaOverlayProvider`] so every SELECT feature is
/// reused.
fn build_generate_series_post_plan(
    query: &Query,
    bind_name: &str,
    relation_schema: &Schema,
    provider: &dyn TableProvider,
) -> BoltResult<LogicalPlan> {
    let mut rewritten = query.clone();
    if let SetExpr::Select(select) = rewritten.body.as_mut() {
        let from0 = &mut select.from[0];
        from0.relation = TableFactor::Table {
            name: ObjectName(vec![Ident::new(bind_name.to_string())]),
            alias: None,
            args: None,
            with_hints: Vec::new(),
            version: None,
            with_ordinality: false,
            partitions: Vec::new(),
        };
    }
    let overlay = SchemaOverlayProvider {
        base: provider,
        name: bind_name.to_string(),
        schema: relation_schema.clone(),
    };
    let ctes = CteScope::new();
    let plan = plan_query(&rewritten, &overlay, &ctes, 0)?;
    let _ = plan.schema()?;
    Ok(plan)
}

/// Lower a bare query tail's `LIMIT [OFFSET]` into `Option<(limit, offset)>`.
/// Mirrors the LIMIT handling in [`plan_query`].
fn lower_bare_limit(query: &Query) -> BoltResult<Option<(usize, usize)>> {
    let limit_value = match &query.limit {
        Some(e) => Some(usize_from_literal(e, "LIMIT")?),
        None => None,
    };
    let offset_value = match &query.offset {
        Some(Offset { value, .. }) => Some(usize_from_literal(value, "OFFSET")?),
        None => None,
    };
    if limit_value.is_some() || offset_value.is_some() {
        Ok(Some((limit_value.unwrap_or(usize::MAX), offset_value.unwrap_or(0))))
    } else {
        Ok(None)
    }
}

/// Validate that every column referenced by `sorts` exists in `schema` (so a
/// bad ORDER BY over a bare VALUES surfaces at plan time, not execution).
fn validate_sort_columns(sorts: &[SortExpr], schema: &Schema) -> BoltResult<()> {
    for s in sorts {
        validate_expr_columns(&s.expr, schema)?;
    }
    Ok(())
}

/// Recursively check that every `Expr::Column` in `expr` resolves in `schema`.
fn validate_expr_columns(expr: &Expr, schema: &Schema) -> BoltResult<()> {
    match expr {
        Expr::Column(name) => {
            schema.index_of(name)?;
            Ok(())
        }
        Expr::Literal(_) => Ok(()),
        Expr::Binary { left, right, .. } => {
            validate_expr_columns(left, schema)?;
            validate_expr_columns(right, schema)
        }
        Expr::Unary { operand, .. } => validate_expr_columns(operand, schema),
        Expr::Cast { expr, .. } => validate_expr_columns(expr, schema),
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// DISTINCT ON (Postgres) — first row per key under ORDER BY
// ---------------------------------------------------------------------------

/// Reserved ephemeral table name the OUTER projection template
/// ([`DistinctOnPlan::post`]) is NOT needed for — DISTINCT ON post-processing is
/// done entirely host-side over the base batch — but kept for symmetry / future
/// use. The DISTINCT ON path binds nothing under a name.
pub const DISTINCT_ON_RESULT_TABLE: &str = "__distinct_on_result";

/// Descriptor for a `SELECT DISTINCT ON (key_exprs) ... [ORDER BY ...] [LIMIT ..]`
/// (Postgres extension), executed host-side as an engine special-case.
///
/// Semantics (matching Postgres): keep the FIRST row of each group of rows that
/// share the same `key_exprs` values, where "first" is the query's ORDER BY
/// order. ORDER BY should lead with the DISTINCT ON keys for a deterministic
/// representative; if it does not (or is absent) the chosen row is
/// arbitrary-but-one-per-group (we still pick deterministically — the first row
/// in the sorted-or-input order). A NULL key is its own group (like GROUP BY).
///
/// Execution (see [`crate::exec::engine::Engine::execute_distinct_on`]):
///   1. Run [`Self::base`] — the SELECT lowered as if DISTINCT ON were absent,
///      but with the DISTINCT ON key columns prepended to the projection (as
///      `__diston_0..N`) and the query's ORDER BY applied. So the base batch is
///      `[__diston_0..N, <user projection...>]` in ORDER BY order.
///   2. Host-side, keep the first row per `__diston_*` key tuple, drop the key
///      columns, and apply [`Self::limit`].
#[derive(Debug, Clone)]
pub struct DistinctOnPlan {
    /// Base subplan producing `[__diston_0..N, <user projection columns...>]`
    /// in the query's ORDER BY order. Flows through the ordinary pipeline.
    pub base: LogicalPlan,
    /// Number of leading DISTINCT ON key columns in `base`'s output.
    pub n_keys: usize,
    /// The user projection's output schema (== `base` schema minus the leading
    /// `n_keys` key columns).
    pub output_schema: Schema,
    /// Optional `(limit, offset)` applied after dedup.
    pub limit: Option<(usize, usize)>,
}

impl DistinctOnPlan {
    /// Overall output schema (the user projection).
    pub fn schema(&self) -> BoltResult<Schema> {
        Ok(self.output_schema.clone())
    }
}

/// Synthetic column-name prefix for the prepended DISTINCT ON key columns in
/// [`DistinctOnPlan::base`].
pub const DISTINCT_ON_KEY_PREFIX: &str = "__diston_";

/// Detect and plan a top-level `SELECT DISTINCT ON (...)` query (feature DISTINCT
/// ON). Returns `Ok(Some(plan))` for the supported shape, `Ok(None)` for any
/// query without `DISTINCT ON`, and `Err` for a `DISTINCT ON` query of an
/// unsupported shape (documented below).
pub fn plan_distinct_on(
    sql: &str,
    provider: &dyn TableProvider,
) -> BoltResult<Option<DistinctOnPlan>> {
    guard_sql_size(sql)?;
    let dialect = GenericDialect {};
    let mut stmts =
        Parser::parse_sql(&dialect, sql).map_err(|e| parse_error_to_bolt_error(e, sql))?;
    if stmts.len() != 1 {
        return Ok(None);
    }
    let stmt = stmts.remove(0);
    let query = match stmt {
        Statement::Query(q) => q,
        _ => return Ok(None),
    };
    plan_distinct_on_inner(&query, provider)
}

fn plan_distinct_on_inner(
    query: &Query,
    provider: &dyn TableProvider,
) -> BoltResult<Option<DistinctOnPlan>> {
    if query.with.is_some() {
        return Ok(None);
    }
    let select = match query.body.as_ref() {
        SetExpr::Select(s) => s.as_ref(),
        _ => return Ok(None),
    };
    let key_exprs = match &select.distinct {
        Some(Distinct::On(exprs)) => exprs,
        _ => return Ok(None), // plain SELECT / SELECT DISTINCT handled elsewhere.
    };
    if key_exprs.is_empty() {
        return Err(BoltError::Sql(
            "DISTINCT ON requires at least one key expression".into(),
        ));
    }

    // Supported shape: a single SELECT whose DISTINCT ON keys are simple column
    // references, and which has no GROUP BY / HAVING / window functions (those
    // interact with DISTINCT ON in ways we do not model). The keys must appear
    // in the SELECT projection by name so we can carry them through; rather than
    // require that, we PREPEND them to the projection as synthetic columns.
    let has_group_by = match &select.group_by {
        GroupByExpr::All(_) => true,
        GroupByExpr::Expressions(exprs, _) => !exprs.is_empty(),
    };
    if has_group_by {
        return Err(BoltError::Sql(
            "unsupported: DISTINCT ON combined with GROUP BY".into(),
        ));
    }
    if select.having.is_some() {
        return Err(BoltError::Sql(
            "unsupported: DISTINCT ON combined with HAVING".into(),
        ));
    }

    // The DISTINCT ON key expressions must be simple column identifiers (the
    // common Postgres usage). Computed-expression keys are rejected cleanly.
    let mut key_names: Vec<String> = Vec::with_capacity(key_exprs.len());
    for k in key_exprs {
        match k {
            SqlExpr::Identifier(id) => key_names.push(ident_to_name(id)),
            SqlExpr::CompoundIdentifier(parts) if !parts.is_empty() => {
                key_names.push(ident_to_name(parts.last().unwrap()))
            }
            _ => {
                return Err(BoltError::Sql(
                    "unsupported: DISTINCT ON key must be a simple column reference".into(),
                ))
            }
        }
    }

    // Build the base SELECT with DISTINCT ON stripped and the key columns
    // prepended to the projection. We clone the SELECT, drop the DISTINCT, and
    // prepend `key AS __diston_<i>` projection items, then lower it (with the
    // query's ORDER BY) through the ordinary `plan_query`.
    let mut base_select = select.clone();
    base_select.distinct = None;
    let mut new_projection: Vec<SelectItem> = Vec::with_capacity(
        key_names.len() + base_select.projection.len(),
    );
    for (i, k) in key_exprs.iter().enumerate() {
        new_projection.push(SelectItem::ExprWithAlias {
            expr: k.clone(),
            alias: Ident::new(format!("{DISTINCT_ON_KEY_PREFIX}{i}")),
        });
    }
    new_projection.extend(base_select.projection.iter().cloned());
    base_select.projection = new_projection;

    let mut base_query = query.clone();
    *base_query.body = SetExpr::Select(Box::new(base_select));
    // ORDER BY stays on `base_query` so the base batch arrives sorted. LIMIT is
    // applied AFTER dedup (Postgres applies LIMIT to the DISTINCT ON result), so
    // strip it from the base and carry it on the descriptor.
    let limit = lower_bare_limit(&base_query)?;
    base_query.limit = None;
    base_query.offset = None;
    base_query.fetch = None;

    let ctes = CteScope::new();
    let base = plan_query(&base_query, provider, &ctes, 0)?;
    let base_schema = base.schema()?;
    let n_keys = key_names.len();
    if base_schema.fields.len() < n_keys {
        return Err(BoltError::Sql(
            "DISTINCT ON: internal — base projection lost key columns".into(),
        ));
    }
    // The user output schema is the base schema minus the leading key columns.
    let output_schema = Schema::new(base_schema.fields[n_keys..].to_vec());

    Ok(Some(DistinctOnPlan {
        base,
        n_keys,
        output_schema,
        limit,
    }))
}

// ---------------------------------------------------------------------------
// COUNT(DISTINCT col) with GROUP BY (feature F3-finish)
// ---------------------------------------------------------------------------

/// Reserved ephemeral table name used by [`CountDistinctGroupByPlan::post`] to
/// reference the host-built count result. Mirrors the WITH RECURSIVE pattern,
/// where the engine binds an in-memory relation under a name the post-plan's
/// `Scan` reads. The leading `__` keeps it out of any user namespace (user
/// identifiers are case-folded but never synthesise this prefix).
pub const COUNT_DISTINCT_GROUPBY_RESULT_TABLE: &str = "__count_distinct_groupby_result";

/// Descriptor for a `COUNT(DISTINCT <col>)` query with a `GROUP BY` (feature
/// F3-finish), executed as a host-orchestrated engine special-case rather than
/// a plan node.
///
/// The plan-node route is blocked: `physical_plan.rs`'s `is_scan_chain`
/// requires a GROUP BY `Aggregate` to sit directly on a Scan/Filter/Project
/// chain (it rejects a `Distinct`/`Aggregate` child), and `AggregateExpr` has
/// no `CountDistinct` variant — adding either would touch shared files and
/// break exhaustive matches. So, exactly like [`RecursiveCtePlan`], this is a
/// standalone descriptor the engine orchestrates host-side: it runs [`Self::base`]
/// (an ordinary `LogicalPlan` that flows through the normal pipeline), groups
/// the resulting rows host-side by the leading group-key columns, counts the
/// DISTINCT non-NULL values of the trailing distinct column per group, and
/// (optionally) runs [`Self::post`] over the count result to apply HAVING /
/// ORDER BY / LIMIT.
///
/// See [`plan_count_distinct_groupby`] for the exact shape detected vs.
/// rejected.
#[derive(Debug, Clone)]
pub struct CountDistinctGroupByPlan {
    /// Base subplan: `SELECT <group_keys...>, <distinct_col> FROM <table>
    /// [WHERE ...]`. The group-key columns come first (in SELECT/GROUP-BY
    /// order), the distinct-counted column last. Flows through the ordinary
    /// optimize → lower → execute pipeline as a normal `LogicalPlan`.
    pub base: LogicalPlan,
    /// Output names of the group-key columns, in order — these are the leading
    /// columns of `base`'s output schema and of the result schema.
    pub group_key_names: Vec<String>,
    /// Output alias for the `Int64` count column appended after the group keys
    /// in the result (the SELECT alias, or the canonical aggregate name).
    pub count_alias: String,
    /// Schema of the host-built count result: the group-key fields (carried
    /// from `base`'s schema) followed by the `Int64` count column.
    pub result_schema: Schema,
    /// Optional post-processing plan (HAVING → `Filter`, ORDER BY → `Sort`,
    /// LIMIT → `Limit`) layered over a synthetic `Scan` of `result_schema`
    /// named [`COUNT_DISTINCT_GROUPBY_RESULT_TABLE`]. `None` when the query has
    /// none of those clauses (the count result is returned directly).
    pub post: Option<LogicalPlan>,
}

impl CountDistinctGroupByPlan {
    /// Overall output schema. With a `post` plan the schema is that plan's
    /// (HAVING/ORDER BY/LIMIT are schema-preserving here, but a future Project
    /// would not be); otherwise it is the count-result schema.
    pub fn schema(&self) -> BoltResult<Schema> {
        match &self.post {
            Some(p) => p.schema(),
            None => Ok(self.result_schema.clone()),
        }
    }
}

/// Detect and plan a top-level single-`SELECT` carrying a `GROUP BY` whose
/// aggregate list is *exactly one* `COUNT(DISTINCT <col>)` (feature
/// F3-finish).
///
/// Returns `Ok(Some(descriptor))` ONLY for that precise shape; `Ok(None)` for
/// everything else (the caller then falls through to the ordinary
/// [`parse`]/`plan_select` pipeline, which keeps producing the precise
/// rejections for the richer mixes this does not implement). A malformed query
/// of the right *shape* (e.g. a bad LIMIT literal) returns `Err`.
///
/// # Supported shape
///
/// A single `SELECT`:
/// * body is one `SELECT` (no UNION / set-op, no `WITH`);
/// * a plain `GROUP BY <keys>` (not ROLLUP / CUBE / GROUPING SETS / ALL);
/// * the SELECT list is the group keys (each a non-aggregate expression that is
///   a declared group key, and every declared key must be projected) followed
///   by exactly one `COUNT(DISTINCT <col>)` item as the **last** SELECT item,
///   optionally aliased (count-last keeps the result-column order == SELECT
///   order with no reprojection);
/// * optional `WHERE`, `HAVING` (over the count and/or group keys), `ORDER BY`,
///   `LIMIT`/`OFFSET`.
///
/// # Falls through (`Ok(None)`) — handled / rejected by the ordinary pipeline
///
/// * no `GROUP BY`, or no `COUNT(DISTINCT)` in the SELECT list (ordinary
///   queries);
/// * ROLLUP / CUBE / GROUPING SETS / `GROUP BY ALL`;
/// * `COUNT(DISTINCT)` alongside any *other* aggregate, or *more than one*
///   `COUNT(DISTINCT)`, or a non-group-key, non-`COUNT(DISTINCT)` SELECT item
///   (these reach `plan_select`, whose existing messages reject them);
/// * `SELECT DISTINCT` over this shape (kept to the existing path).
pub fn plan_count_distinct_groupby(
    sql: &str,
    provider: &dyn TableProvider,
) -> BoltResult<Option<CountDistinctGroupByPlan>> {
    guard_sql_size(sql)?;
    let dialect = GenericDialect {};
    let mut stmts =
        Parser::parse_sql(&dialect, sql).map_err(|e| parse_error_to_bolt_error(e, sql))?;
    if stmts.len() != 1 {
        // Let the ordinary pipeline produce the canonical multi-statement error.
        return Ok(None);
    }
    let stmt = stmts.remove(0);
    let query = match stmt {
        Statement::Query(q) => q,
        _ => return Ok(None),
    };
    // No CTEs / set-ops / dialect tail clauses here — those route through the
    // ordinary pipeline (or the recursive-CTE hook).
    if query.with.is_some() {
        return Ok(None);
    }
    let select = match query.body.as_ref() {
        SetExpr::Select(s) => s.as_ref(),
        _ => return Ok(None),
    };

    // Must be a plain GROUP BY (reject super-aggregates / GROUP BY ALL by
    // falling through, since `parse_group_by` rejects those and we only want
    // to special-case the plain form).
    let parsed_group_by = match parse_group_by(&select.group_by) {
        Ok(p) => p,
        // A super-aggregate / GROUP BY ALL shape: not ours — fall through so
        // the ordinary path emits its message.
        Err(_) => return Ok(None),
    };
    if parsed_group_by.is_super || parsed_group_by.all_cols.is_empty() {
        // No plain GROUP BY keys → not the shape we handle.
        return Ok(None);
    }

    // SELECT DISTINCT over this shape stays on the existing path.
    if select.distinct.is_some() {
        return Ok(None);
    }

    // We need a resolver to recognise COUNT(DISTINCT ...). `try_count_distinct`
    // does not actually consult the resolver, but the signature requires one;
    // an empty resolver is sufficient for detection.
    let resolver = NameResolver::empty();

    // Expand the SELECT list into (expr, alias). Wildcards can't be group-key
    // SELECT items in a well-formed GROUP BY query of our shape; if present we
    // fall through (the ordinary path will reject / expand as it sees fit).
    let mut items: Vec<(SqlExpr, Option<String>)> = Vec::new();
    for item in &select.projection {
        match item {
            SelectItem::UnnamedExpr(e) => items.push((e.clone(), None)),
            SelectItem::ExprWithAlias { expr, alias } => {
                items.push((expr.clone(), Some(ident_to_name(alias))))
            }
            // Wildcards / qualified wildcards in a COUNT(DISTINCT) GROUP BY
            // SELECT list are not our shape — defer to the ordinary path.
            SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(_, _) => return Ok(None),
        }
    }

    // Locate the COUNT(DISTINCT ...) items. `try_count_distinct` returns an
    // Err for malformed DISTINCT-quantified forms (e.g. `SUM(DISTINCT x)`,
    // `COUNT(DISTINCT *)`); since detection runs before the ordinary pipeline
    // and we want those precise messages to come from the ordinary path, we
    // treat such an Err as "not our shape" and fall through.
    let mut cd_positions: Vec<usize> = Vec::new();
    for (i, (e, _)) in items.iter().enumerate() {
        match try_count_distinct(e, &resolver) {
            Ok(Some(_)) => cd_positions.push(i),
            Ok(None) => {}
            Err(_) => return Ok(None),
        }
    }
    // Exactly one COUNT(DISTINCT) — zero means an ordinary query, more than one
    // is an unsupported mix (the ordinary path rejects it precisely).
    if cd_positions.len() != 1 {
        return Ok(None);
    }
    let cd_index = cd_positions[0];
    // We support the COUNT(DISTINCT) as the *last* SELECT item only (the common
    // `SELECT keys..., COUNT(DISTINCT x)` form). When it appears earlier, the
    // result-column order would diverge from the SELECT list; rather than emit
    // columns in the wrong order we fall through (the ordinary path rejects the
    // shape). Group keys then occupy SELECT positions `0..cd_index` and the
    // count is last — so result order == SELECT order with no reprojection.
    if cd_index != items.len() - 1 {
        return Ok(None);
    }

    // Every *other* SELECT item must be a non-aggregate expression that is a
    // declared GROUP BY key. We compare the *raw* SQL expressions structurally
    // (sqlparser `SqlExpr: PartialEq`): a group key is normally written
    // identically in SELECT and GROUP BY. Any item that
    //   * contains an aggregate (a second aggregate alongside COUNT(DISTINCT)),
    //     or
    //   * does not match a GROUP BY expression (e.g. a non-grouped column,
    //     which is not valid SQL for this shape),
    // is NOT our shape: fall through so the ordinary `plan_select` path emits
    // its precise message (the conservative structural compare also defers any
    // equivalent-but-differently-spelled key to that path rather than risk a
    // wrong host grouping).
    for (i, (e, _)) in items.iter().enumerate() {
        if i == cd_index {
            continue;
        }
        match contains_aggregate(e, &resolver, 0) {
            Ok(true) => return Ok(None),
            Ok(false) => {}
            Err(_) => return Ok(None),
        }
        if !parsed_group_by.all_cols.iter().any(|g| g == e) {
            return Ok(None);
        }
    }

    // Conversely, every declared GROUP BY key must appear in the SELECT list:
    // the host orchestrator builds its base from the SELECTed columns, so a key
    // that is grouped-by but not projected cannot be grouped on. If any GROUP
    // BY key is missing from the (non-COUNT-DISTINCT) SELECT items, this is not
    // our shape — fall through (the ordinary path rejects it precisely, rather
    // than the engine silently ignoring the missing key).
    let non_cd_items: Vec<&SqlExpr> = items
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != cd_index)
        .map(|(_, (e, _))| e)
        .collect();
    for g in &parsed_group_by.all_cols {
        if !non_cd_items.iter().any(|e| *e == g) {
            return Ok(None);
        }
    }

    // The distinct-counted argument expression and the count column's output
    // alias.
    let distinct_arg = match try_count_distinct(&items[cd_index].0, &resolver)? {
        Some(inner) => inner.clone(),
        None => return Ok(None),
    };
    let count_alias = match &items[cd_index].1 {
        Some(a) => a.clone(),
        None => aggregate_output_name(&AggregateExpr::Count(Expr::Literal(Literal::Int64(1)))),
    };

    // Build the base SELECT by cloning the original and rewriting the
    // projection so the COUNT(DISTINCT col) item becomes a bare `col`, and the
    // GROUP BY / HAVING / DISTINCT are stripped (the base just materialises the
    // (group_keys..., distinct_col) rows; grouping + counting happen
    // host-side). This reuses the full `plan_select` machinery (FROM, WHERE,
    // column resolution, type-checking) rather than re-deriving it.
    let mut base_select = select.clone();
    base_select.group_by = GroupByExpr::Expressions(Vec::new(), Vec::new());
    base_select.having = None;
    base_select.distinct = None;
    base_select.projection = {
        let mut p: Vec<SelectItem> = Vec::with_capacity(items.len());
        for (i, item) in select.projection.iter().enumerate() {
            if i == cd_index {
                // Replace COUNT(DISTINCT <arg>) with the bare distinct argument
                // (unaliased; we recover it positionally as the last column).
                p.push(SelectItem::UnnamedExpr(distinct_arg.clone()));
            } else {
                p.push(item.clone());
            }
        }
        p
    };

    // Lower the base SELECT. Since the COUNT(DISTINCT) item is the LAST SELECT
    // item, the base output is already `[group_keys..., distinct_col]` — exactly
    // the layout the host group-distinct-count step expects (leading keys,
    // trailing distinct value). No reprojection is needed.
    let base = plan_select(&base_select, provider, &CteScope::new(), 0)?;
    let base_schema = base.schema()?;
    if base_schema.fields.len() != items.len() || base_schema.fields.is_empty() {
        // Should not happen for our shape, but guard rather than panic.
        return Ok(None);
    }

    // Group keys are the leading `items.len() - 1` output columns; the distinct
    // column is the last (at `cd_index`).
    let group_key_fields: Vec<crate::plan::logical_plan::Field> =
        base_schema.fields[..cd_index].to_vec();
    let group_key_names: Vec<String> =
        group_key_fields.iter().map(|f| f.name.clone()).collect();

    // Result schema: group-key fields (carried through), then the Int64 count.
    // The count is non-nullable (COUNT never produces SQL NULL).
    let mut result_fields = group_key_fields;
    result_fields.push(crate::plan::logical_plan::Field::new(
        count_alias.clone(),
        DataType::Int64,
        false,
    ));
    let result_schema = Schema::new(result_fields);

    // Build the optional post-plan (HAVING / ORDER BY / LIMIT) over a synthetic
    // Scan of the result. Reuses the ordinary lowering helpers so HAVING /
    // ORDER BY can reference the count by its `COUNT(DISTINCT col)` call, by
    // alias, or a group key by name.
    let post = build_count_distinct_post_plan(
        &query,
        select,
        &distinct_arg,
        &count_alias,
        &result_schema,
        provider,
    )?;

    Ok(Some(CountDistinctGroupByPlan {
        base,
        group_key_names,
        count_alias,
        result_schema,
        post,
    }))
}

/// Build the optional HAVING/ORDER BY/LIMIT post-plan for a
/// [`CountDistinctGroupByPlan`], layered over a synthetic `Scan` of the count
/// result. Returns `Ok(None)` when the query has none of those clauses.
///
/// The synthetic scan is named [`COUNT_DISTINCT_GROUPBY_RESULT_TABLE`]; the
/// engine binds the host-built result batch under that name before running the
/// post-plan through the ordinary subplan executor (mirroring the WITH
/// RECURSIVE overlay). HAVING is lowered with
/// [`lower_having_over_count_distinct`] so `COUNT(DISTINCT col)` in HAVING maps
/// onto the result's count column; ORDER BY / LIMIT reuse the same helpers as
/// the ordinary query tail.
fn build_count_distinct_post_plan(
    query: &Query,
    select: &Select,
    distinct_arg: &SqlExpr,
    count_alias: &str,
    result_schema: &Schema,
    provider: &dyn TableProvider,
) -> BoltResult<Option<LogicalPlan>> {
    let has_having = select.having.is_some();
    let has_order = query
        .order_by
        .as_ref()
        .map(|o| !o.exprs.is_empty())
        .unwrap_or(false);
    let has_limit = query.limit.is_some() || query.offset.is_some();
    if !has_having && !has_order && !has_limit {
        return Ok(None);
    }

    // Synthetic scan over the result relation. A resolver scoped to this schema
    // lets HAVING / ORDER BY reference the result columns (group keys + count).
    let scan = LogicalPlan::Scan {
        table: COUNT_DISTINCT_GROUPBY_RESULT_TABLE.to_string(),
        projection: None,
        schema: result_schema.clone(),
    };
    let ctes = CteScope::new();
    let mut plan = scan;

    // HAVING → Filter over the count column. `lower_having_over_count_distinct`
    // resolves a `COUNT(DISTINCT col)` reference to `count_alias` (the result
    // column) when its argument matches the SELECT's distinct argument; a group
    // key referenced by name lowers as an ordinary column.
    if let Some(having_sql) = &select.having {
        let mut resolver = NameResolver::empty();
        resolver.ctx = Some(SubqueryCtx {
            provider,
            ctes: &ctes,
        });
        resolver.push_base(COUNT_DISTINCT_GROUPBY_RESULT_TABLE.to_string(), result_schema);
        let predicate = lower_having_over_count_distinct(
            having_sql,
            distinct_arg,
            &resolver,
            count_alias,
            0,
        )?;
        validate_having_columns(&predicate, result_schema)?;
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate,
        };
    }

    // ORDER BY → Sort over the result schema.
    if let Some(order_by) = &query.order_by {
        let sort_exprs = lower_order_by(
            &order_by.exprs,
            SubqueryCtx {
                provider,
                ctes: &ctes,
            },
        )?;
        if !sort_exprs.is_empty() {
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                sort_exprs,
            };
        }
    }

    // LIMIT [OFFSET] → Limit. Mirrors `plan_query`'s handling.
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

    // Force a type-check so a bad HAVING/ORDER BY reference surfaces at plan
    // time rather than execution.
    let _ = plan.schema()?;
    Ok(Some(plan))
}

// ---------------------------------------------------------------------------
// Multi / mixed COUNT(DISTINCT) with GROUP BY (feature F3-finish, generalized)
// ---------------------------------------------------------------------------

/// Reserved ephemeral table name used by [`MultiAggGroupByPlan::post`] to
/// reference the host-built aggregate result (sibling of
/// [`COUNT_DISTINCT_GROUPBY_RESULT_TABLE`] for the generalized path).
pub const MULTI_AGG_GROUPBY_RESULT_TABLE: &str = "__multi_agg_groupby_result";

/// One aggregate the host computes per group in the generalized
/// COUNT(DISTINCT) + GROUP BY path (feature F3-finish, generalized).
///
/// Each variant carries the index of its *input* column in the materialised
/// base batch (`[group_keys..., input_0, input_1, ...]`) — i.e. an index in
/// `n_keys..n_cols`. `CountStar` is a `COUNT(*)`: it counts rows regardless of
/// any column value, but still carries a (sentinel) input column so the base
/// projection stays uniform.
#[derive(Debug, Clone)]
pub enum CdAgg {
    /// `COUNT(DISTINCT col)` — distinct non-NULL values of the input column.
    CountDistinct { base_col: usize },
    /// `COUNT(col)` — non-NULL row count of the input column.
    Count { base_col: usize },
    /// `COUNT(*)` — total row count of the group.
    CountStar { base_col: usize },
    /// `SUM(col)` — output preserves the (numeric) input dtype.
    Sum { base_col: usize },
    /// `MIN(col)` — output preserves the input dtype.
    Min { base_col: usize },
    /// `MAX(col)` — output preserves the input dtype.
    Max { base_col: usize },
    /// `AVG(col)` — output `Float64`.
    Avg { base_col: usize },
}

impl CdAgg {
    /// Index of this aggregate's input column in the base batch.
    pub fn base_col(&self) -> usize {
        match self {
            CdAgg::CountDistinct { base_col }
            | CdAgg::Count { base_col }
            | CdAgg::CountStar { base_col }
            | CdAgg::Sum { base_col }
            | CdAgg::Min { base_col }
            | CdAgg::Max { base_col }
            | CdAgg::Avg { base_col } => *base_col,
        }
    }
}

/// One output column of the generalized result, in SELECT order: either a
/// projected group key (index into the group-key list) or a host-computed
/// aggregate (index into the `aggs` list).
#[derive(Debug, Clone, Copy)]
pub enum CdOutputCol {
    /// A group key: `group_key_names[idx]` / base column `idx`.
    GroupKey(usize),
    /// An aggregate: `aggs[idx]`.
    Agg(usize),
}

/// Descriptor for the *generalized* COUNT(DISTINCT) + GROUP BY shape — multiple
/// distinct counts and/or a mix with plain aggregates (feature F3-finish,
/// generalized). The single-sole-`COUNT(DISTINCT)` case stays on the dedicated
/// [`CountDistinctGroupByPlan`] path; this descriptor handles everything the
/// single-CD detector declines but which is still a plain `GROUP BY` whose
/// SELECT list is group keys plus host-computable aggregates including at
/// least one `COUNT(DISTINCT)`.
///
/// The engine ([`crate::exec::engine::Engine::execute_multi_agg_groupby`])
/// runs [`Self::base`] to materialise `[group_keys..., agg_inputs...]`, groups
/// host-side, computes every [`CdAgg`] per group, assembles the output columns
/// in [`Self::output_layout`] order, and (optionally) runs [`Self::post`] for
/// HAVING / ORDER BY / LIMIT.
#[derive(Debug, Clone)]
pub struct MultiAggGroupByPlan {
    /// Base subplan: `SELECT <group_keys...>, <agg_input_exprs...> FROM <table>
    /// [WHERE ...]`. Output columns are the group keys (first `n_keys`)
    /// followed by one input column per aggregate (in `aggs` order).
    pub base: LogicalPlan,
    /// Number of leading group-key columns in `base`.
    pub n_keys: usize,
    /// Output names of the group-key columns, in `base` order.
    pub group_key_names: Vec<String>,
    /// The per-group aggregates to compute, in input-column order.
    pub aggs: Vec<CdAgg>,
    /// The output columns in SELECT order (group keys + aggregates interleaved
    /// exactly as written).
    pub output_layout: Vec<CdOutputCol>,
    /// Schema of the host-built result (one field per `output_layout` entry).
    pub result_schema: Schema,
    /// Optional HAVING / ORDER BY / LIMIT post-plan over a synthetic `Scan` of
    /// `result_schema` named [`MULTI_AGG_GROUPBY_RESULT_TABLE`].
    pub post: Option<LogicalPlan>,
}

impl MultiAggGroupByPlan {
    /// Overall output schema.
    pub fn schema(&self) -> BoltResult<Schema> {
        match &self.post {
            Some(p) => p.schema(),
            None => Ok(self.result_schema.clone()),
        }
    }
}

/// A host-computable plain-aggregate kind for the generalized
/// COUNT(DISTINCT)+GROUP BY shape. Recognised purely by function NAME (no
/// argument lowering) so qualified arguments (`SUM(t.x)`) classify correctly;
/// the argument expression is lowered later, by `plan_select` over the
/// rewritten base, against a real resolver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlainAggTag {
    /// `COUNT(col)` (a present, non-`*` argument).
    Count,
    /// `COUNT(*)` (no argument).
    CountStar,
    Sum,
    Min,
    Max,
    Avg,
}

/// Classify a SELECT item for the generalized COUNT(DISTINCT)+GROUP BY shape.
enum CdSelectItem {
    /// A group-key expression (matched structurally against GROUP BY).
    GroupKey,
    /// `COUNT(DISTINCT <expr>)` carrying its unlowered argument.
    CountDistinct(SqlExpr),
    /// A plain aggregate the host path computes; carries its kind tag (the
    /// argument expression is recovered from the SELECT item later).
    Plain(PlainAggTag),
}

/// Recognise a *plain* (non-DISTINCT) aggregate SELECT item by function name,
/// WITHOUT lowering its argument. Returns:
///   * `Ok(Some(tag))` — `e` is a host-computable plain aggregate;
///   * `Ok(None)`       — `e` is not a (single-name) function call;
///   * `Err(_)`         — `e` is an aggregate the host path can't compute
///     under GROUP BY (e.g. `VAR_POP`, `STDDEV`), or a malformed aggregate
///     argument shape (multi-arg, qualified `*`).
///
/// A DISTINCT quantifier is left for [`try_count_distinct`]; this returns
/// `Ok(None)` for a DISTINCT-quantified call so the caller's COUNT(DISTINCT)
/// branch (which runs first) owns it.
fn try_plain_agg_tag(e: &SqlExpr) -> BoltResult<Option<PlainAggTag>> {
    let func = match e {
        SqlExpr::Function(f) => f,
        _ => return Ok(None),
    };
    if func.name.0.len() != 1 || func.over.is_some() {
        return Ok(None);
    }
    let fname = func.name.0[0].value.to_ascii_uppercase();
    let arg_list = match &func.args {
        FunctionArguments::List(list) => list,
        _ => return Ok(None),
    };
    // A DISTINCT-quantified call is handled by `try_count_distinct`.
    if arg_list.duplicate_treatment.is_some() {
        return Ok(None);
    }
    match fname.as_str() {
        "COUNT" | "SUM" | "MIN" | "MAX" | "AVG" => {}
        // Aggregates with no host GROUP BY implementation here.
        "VAR_POP" | "VAR_SAMP" | "VARIANCE" | "STDDEV" | "STDDEV_POP" | "STDDEV_SAMP" => {
            return Err(BoltError::Sql(format!(
                "unsupported: {fname} alongside COUNT(DISTINCT) under GROUP BY \
                 (host path computes only COUNT / SUM / MIN / MAX / AVG)"
            )));
        }
        _ => return Ok(None),
    }
    if arg_list.args.len() != 1 {
        return Err(BoltError::Sql(format!(
            "{fname} expects exactly one argument, got {}",
            arg_list.args.len()
        )));
    }
    match (&fname[..], &arg_list.args[0]) {
        ("COUNT", FunctionArg::Unnamed(FunctionArgExpr::Wildcard)) => {
            Ok(Some(PlainAggTag::CountStar))
        }
        (_, FunctionArg::Unnamed(FunctionArgExpr::Wildcard))
        | (_, FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(_))) => {
            Err(BoltError::Sql(format!("{fname}(*) is not supported")))
        }
        (_, FunctionArg::Unnamed(FunctionArgExpr::Expr(_))) => Ok(Some(match &fname[..] {
            "COUNT" => PlainAggTag::Count,
            "SUM" => PlainAggTag::Sum,
            "MIN" => PlainAggTag::Min,
            "MAX" => PlainAggTag::Max,
            "AVG" => PlainAggTag::Avg,
            _ => unreachable!("name already filtered"),
        })),
        (_, FunctionArg::Named { .. }) => Err(BoltError::Sql(format!(
            "unsupported: named argument to {fname}"
        ))),
    }
}

/// Detect and plan the *generalized* COUNT(DISTINCT) + GROUP BY shape (feature
/// F3-finish, generalized): multiple `COUNT(DISTINCT)` and/or a mix with plain
/// aggregates under a plain `GROUP BY`.
///
/// Returns `Ok(Some(descriptor))` ONLY when the query is a plain `GROUP BY`
/// whose SELECT list is, in any order, group keys plus aggregates where:
/// * at least one aggregate is a `COUNT(DISTINCT col)` (otherwise it is an
///   ordinary GROUP BY the normal pipeline already handles, so we fall
///   through); and
/// * the *non*-COUNT(DISTINCT) aggregates are all host-computable here:
///   `COUNT(*)`, `COUNT(col)`, `SUM(col)`, `MIN(col)`, `MAX(col)`, `AVG(col)`.
///
/// `Ok(None)` (fall through to the dedicated single-CD path / ordinary
/// pipeline) for: no GROUP BY; super-aggregates / GROUP BY ALL; `SELECT
/// DISTINCT`; CTE / set-op; a wildcard SELECT item; a non-group-key,
/// non-aggregate SELECT item; or a query the single-sole-COUNT(DISTINCT)
/// detector already handles (zero plain aggregates AND exactly one
/// COUNT(DISTINCT) — that stays on the proven path). An aggregate the host
/// path cannot compute here (e.g. `VAR_POP` / `STDDEV` under GROUP BY) yields
/// a precise `Err`.
pub fn plan_multi_agg_groupby(
    sql: &str,
    provider: &dyn TableProvider,
) -> BoltResult<Option<MultiAggGroupByPlan>> {
    guard_sql_size(sql)?;
    let dialect = GenericDialect {};
    let mut stmts =
        Parser::parse_sql(&dialect, sql).map_err(|e| parse_error_to_bolt_error(e, sql))?;
    if stmts.len() != 1 {
        return Ok(None);
    }
    let query = match stmts.remove(0) {
        Statement::Query(q) => q,
        _ => return Ok(None),
    };
    if query.with.is_some() {
        return Ok(None);
    }
    let select = match query.body.as_ref() {
        SetExpr::Select(s) => s.as_ref(),
        _ => return Ok(None),
    };

    // Plain GROUP BY only (reject super-aggregates / GROUP BY ALL by falling
    // through to the ordinary path's message).
    let parsed_group_by = match parse_group_by(&select.group_by) {
        Ok(p) => p,
        Err(_) => return Ok(None),
    };
    if parsed_group_by.is_super || parsed_group_by.all_cols.is_empty() {
        return Ok(None);
    }
    if select.distinct.is_some() {
        return Ok(None);
    }

    // Classification recognises aggregates by NAME only (never lowering their
    // arguments) so qualified args like `SUM(t.x)` classify correctly; the
    // real lowering happens later when `plan_select` lowers the rewritten base
    // against a proper resolver. `try_count_distinct` / `contains_aggregate`
    // take a resolver but don't consult it for detection, so an empty one
    // suffices here.
    let resolver = NameResolver::empty();

    // Expand the SELECT list into (expr, alias).
    let mut items: Vec<(SqlExpr, Option<String>)> = Vec::new();
    for item in &select.projection {
        match item {
            SelectItem::UnnamedExpr(e) => items.push((e.clone(), None)),
            SelectItem::ExprWithAlias { expr, alias } => {
                items.push((expr.clone(), Some(ident_to_name(alias))))
            }
            SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(_, _) => return Ok(None),
        }
    }

    // Classify each SELECT item. A genuinely-unsupported aggregate (e.g.
    // VAR_POP under GROUP BY) is a precise `Err`; a non-group-key,
    // non-aggregate item falls through (`Ok(None)`) so the ordinary path
    // emits its message.
    let mut classified: Vec<CdSelectItem> = Vec::with_capacity(items.len());
    let mut n_count_distinct = 0usize;
    let mut n_plain = 0usize;
    for (e, _alias) in &items {
        // COUNT(DISTINCT ...) takes priority over the plain-aggregate match.
        match try_count_distinct(e, &resolver) {
            Ok(Some(inner)) => {
                classified.push(CdSelectItem::CountDistinct(inner.clone()));
                n_count_distinct += 1;
                continue;
            }
            Ok(None) => {}
            // Malformed DISTINCT form: let the ordinary path emit the message.
            Err(_) => return Ok(None),
        }
        // Plain aggregate? Recognised by name (no argument lowering, so a
        // qualified argument classifies correctly). An aggregate the host path
        // can't compute under GROUP BY is a precise `Err`.
        match try_plain_agg_tag(e) {
            Ok(Some(tag)) => {
                classified.push(CdSelectItem::Plain(tag));
                n_plain += 1;
                continue;
            }
            Ok(None) => {}
            Err(err) => return Err(err),
        }
        // Not an aggregate: must be a declared group key, else not our shape.
        match contains_aggregate(e, &resolver, 0) {
            Ok(true) => return Ok(None),
            Ok(false) => {}
            Err(_) => return Ok(None),
        }
        if !parsed_group_by.all_cols.iter().any(|g| g == e) {
            return Ok(None);
        }
        classified.push(CdSelectItem::GroupKey);
    }

    // Need at least one COUNT(DISTINCT) to own this query at all.
    if n_count_distinct == 0 {
        return Ok(None);
    }
    // The single-sole-COUNT(DISTINCT) case (exactly one CD, no plain agg) stays
    // on the dedicated `plan_count_distinct_groupby` path — but ONLY when that
    // path actually accepts it (the CD must be the LAST SELECT item; otherwise
    // it declines and this generalized path must handle the query). Defer only
    // in the exact shape the dedicated path takes.
    if n_count_distinct == 1 && n_plain == 0 {
        let cd_is_last = matches!(classified.last(), Some(CdSelectItem::CountDistinct(_)));
        if cd_is_last {
            return Ok(None);
        }
    }

    // Every declared GROUP BY key must be projected (the host base is built
    // from the SELECTed group-key columns).
    for g in &parsed_group_by.all_cols {
        let projected = items.iter().zip(classified.iter()).any(|((e, _), c)| {
            matches!(c, CdSelectItem::GroupKey) && e == g
        });
        if !projected {
            return Ok(None);
        }
    }

    // Build the base SELECT: group keys (in GROUP BY order) followed by one
    // input column per aggregate (in SELECT order). We rewrite the projection
    // wholesale rather than in place so the base layout is exactly
    // `[keys..., agg_inputs...]` regardless of the SELECT interleaving.
    let n_keys = parsed_group_by.all_cols.len();
    let mut base_projection: Vec<SelectItem> = Vec::with_capacity(n_keys + items.len());
    for g in &parsed_group_by.all_cols {
        base_projection.push(SelectItem::UnnamedExpr(g.clone()));
    }
    // Aggregates in SELECT order; each contributes its input column. COUNT(*)
    // has no column argument, so we feed a literal `1` sentinel.
    let mut aggs: Vec<CdAgg> = Vec::new();
    let mut output_layout: Vec<CdOutputCol> = Vec::new();
    // group-key output index, assigned in SELECT order to the matching key.
    for ((e, _alias), c) in items.iter().zip(classified.iter()) {
        match c {
            CdSelectItem::GroupKey => {
                // Map this SELECT key to its GROUP BY position.
                let key_idx = parsed_group_by
                    .all_cols
                    .iter()
                    .position(|g| g == e)
                    .expect("group key was validated to be in GROUP BY");
                output_layout.push(CdOutputCol::GroupKey(key_idx));
            }
            CdSelectItem::CountDistinct(inner) => {
                let base_col = n_keys + aggs.len();
                base_projection.push(SelectItem::UnnamedExpr(inner.clone()));
                output_layout.push(CdOutputCol::Agg(aggs.len()));
                aggs.push(CdAgg::CountDistinct { base_col });
            }
            CdSelectItem::Plain(tag) => {
                let base_col = n_keys + aggs.len();
                let (arg_sql, kind) = plain_agg_input_and_kind(e, *tag, base_col)?;
                base_projection.push(SelectItem::UnnamedExpr(arg_sql));
                output_layout.push(CdOutputCol::Agg(aggs.len()));
                aggs.push(kind);
            }
        }
    }

    let mut base_select = select.clone();
    base_select.group_by = GroupByExpr::Expressions(Vec::new(), Vec::new());
    base_select.having = None;
    base_select.distinct = None;
    base_select.projection = base_projection;

    let base = plan_select(&base_select, provider, &CteScope::new(), 0)?;
    let base_schema = base.schema()?;
    let expected_cols = n_keys + aggs.len();
    if base_schema.fields.len() != expected_cols || base_schema.fields.is_empty() {
        return Ok(None);
    }

    // Group-key names / fields are the leading `n_keys` base fields.
    let group_key_fields: Vec<crate::plan::logical_plan::Field> =
        base_schema.fields[..n_keys].to_vec();
    let group_key_names: Vec<String> =
        group_key_fields.iter().map(|f| f.name.clone()).collect();

    // Result schema, in SELECT (output_layout) order. Group keys carry their
    // base dtype; aggregates carry the dtype the host produces.
    let mut result_fields: Vec<crate::plan::logical_plan::Field> =
        Vec::with_capacity(output_layout.len());
    // Per-aggregate output field, computed from its input column's dtype.
    for (out_idx, out) in output_layout.iter().enumerate() {
        match out {
            CdOutputCol::GroupKey(k) => result_fields.push(group_key_fields[*k].clone()),
            CdOutputCol::Agg(a) => {
                let agg = &aggs[*a];
                let input_field = &base_schema.fields[agg.base_col()];
                let (dtype, nullable) = multi_agg_output_type(agg, input_field)?;
                let name = multi_agg_output_name(&items, out_idx);
                result_fields.push(crate::plan::logical_plan::Field::new(name, dtype, nullable));
            }
        }
    }
    let result_schema = Schema::new(result_fields);

    let post = build_multi_agg_post_plan(&query, select, &items, &result_schema, provider)?;

    Ok(Some(MultiAggGroupByPlan {
        base,
        n_keys,
        group_key_names,
        aggs,
        output_layout,
        result_schema,
        post,
    }))
}

/// The output (dtype, nullable) for a host-computed [`CdAgg`] given its input
/// column field. Mirrors [`AggregateExpr`]'s documented output types.
fn multi_agg_output_type(
    agg: &CdAgg,
    input_field: &crate::plan::logical_plan::Field,
) -> BoltResult<(DataType, bool)> {
    Ok(match agg {
        // Counts are non-nullable Int64.
        CdAgg::CountDistinct { .. } | CdAgg::Count { .. } | CdAgg::CountStar { .. } => {
            (DataType::Int64, false)
        }
        // AVG is Float64, nullable (NULL for an empty / all-NULL group).
        CdAgg::Avg { .. } => (DataType::Float64, true),
        // SUM / MIN / MAX preserve the input dtype, nullable (empty / all-NULL
        // group yields NULL — matching the scalar host aggregate semantics).
        CdAgg::Sum { .. } | CdAgg::Min { .. } | CdAgg::Max { .. } => {
            (input_field.dtype, true)
        }
    })
}

/// Output column name for the aggregate at `items[out_idx]`: the SELECT alias
/// if present, else the canonical aggregate name. The canonical name follows
/// [`aggregate_output_name`]'s `<fn>_<col>` rule for a bare-column argument
/// (e.g. `count_a`, `sum_x`) and bare `<fn>` for `COUNT(*)` / a non-column
/// argument. Computed from the raw SQL (no resolver) so it works for
/// qualified arguments too.
fn multi_agg_output_name(items: &[(SqlExpr, Option<String>)], out_idx: usize) -> String {
    let (expr, alias) = &items[out_idx];
    if let Some(a) = alias {
        return a.clone();
    }
    let resolver = NameResolver::empty();
    // COUNT(DISTINCT col) → `count_<col>` (the canonical Count name).
    if let Ok(Some(inner)) = try_count_distinct(expr, &resolver) {
        return match simple_column_name(inner) {
            Some(col) => format!("count_{col}"),
            None => "count".to_string(),
        };
    }
    // Plain aggregate → `<fn>_<col>` (or bare `<fn>` for `*` / non-column).
    if let Ok(Some(tag)) = try_plain_agg_tag(expr) {
        let prefix = match tag {
            PlainAggTag::Count | PlainAggTag::CountStar => "count",
            PlainAggTag::Sum => "sum",
            PlainAggTag::Min => "min",
            PlainAggTag::Max => "max",
            PlainAggTag::Avg => "avg",
        };
        return match sql_function_single_arg(expr).and_then(simple_column_name) {
            Some(col) => format!("{prefix}_{col}"),
            None => prefix.to_string(),
        };
    }
    format!("__agg_{out_idx}")
}

/// The lowercased column name if `e` is a bare (possibly qualified) column
/// reference, else `None`. `a` → `a`; `t.a` → `a` (the trailing identifier,
/// matching how `aggregate_output_name` names a column argument).
fn simple_column_name(e: &SqlExpr) -> Option<String> {
    match e {
        SqlExpr::Identifier(id) => Some(ident_to_name(id)),
        SqlExpr::CompoundIdentifier(parts) => parts.last().map(ident_to_name),
        _ => None,
    }
}

/// Given a plain-aggregate SELECT item and its [`PlainAggTag`], produce
/// (input-column SQL expression, [`CdAgg`]). `COUNT(*)` feeds a literal `1`
/// sentinel column; every other aggregate feeds its single (unlowered)
/// argument expression so the base projection preserves exactly what the user
/// wrote — including a qualified `t.col` — for `plan_select` to lower.
fn plain_agg_input_and_kind(
    sql_item: &SqlExpr,
    tag: PlainAggTag,
    base_col: usize,
) -> BoltResult<(SqlExpr, CdAgg)> {
    let arg = sql_function_single_arg(sql_item);
    let one = SqlExpr::Value(Value::Number("1".to_string(), false));
    Ok(match tag {
        PlainAggTag::CountStar => (one, CdAgg::CountStar { base_col }),
        PlainAggTag::Count => (clone_arg_or_err(arg, "COUNT")?, CdAgg::Count { base_col }),
        PlainAggTag::Sum => (clone_arg_or_err(arg, "SUM")?, CdAgg::Sum { base_col }),
        PlainAggTag::Min => (clone_arg_or_err(arg, "MIN")?, CdAgg::Min { base_col }),
        PlainAggTag::Max => (clone_arg_or_err(arg, "MAX")?, CdAgg::Max { base_col }),
        PlainAggTag::Avg => (clone_arg_or_err(arg, "AVG")?, CdAgg::Avg { base_col }),
    })
}

/// The single function argument SQL expression of `e` if `e` is a one-arg
/// function call `f(<expr>)`; `None` for `f()` / `f(*)` / non-functions.
fn sql_function_single_arg(e: &SqlExpr) -> Option<&SqlExpr> {
    let func = match e {
        SqlExpr::Function(f) => f,
        _ => return None,
    };
    let list = match &func.args {
        FunctionArguments::List(l) => l,
        _ => return None,
    };
    if list.args.len() != 1 {
        return None;
    }
    match &list.args[0] {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(inner)) => Some(inner),
        _ => None,
    }
}

/// Clone the function argument or error (used for aggregates that require one).
fn clone_arg_or_err(arg: Option<&SqlExpr>, kind: &str) -> BoltResult<SqlExpr> {
    arg.cloned()
        .ok_or_else(|| BoltError::Sql(format!("{kind} requires a column argument")))
}

/// Build the optional HAVING / ORDER BY / LIMIT post-plan for a
/// [`MultiAggGroupByPlan`], over a synthetic `Scan` of the result schema named
/// [`MULTI_AGG_GROUPBY_RESULT_TABLE`]. Aggregate references in HAVING / ORDER
/// BY are rewritten to the result column whose SELECT-item aggregate they
/// structurally match.
fn build_multi_agg_post_plan(
    query: &Query,
    select: &Select,
    items: &[(SqlExpr, Option<String>)],
    result_schema: &Schema,
    provider: &dyn TableProvider,
) -> BoltResult<Option<LogicalPlan>> {
    let has_having = select.having.is_some();
    let has_order = query
        .order_by
        .as_ref()
        .map(|o| !o.exprs.is_empty())
        .unwrap_or(false);
    let has_limit = query.limit.is_some() || query.offset.is_some();
    if !has_having && !has_order && !has_limit {
        return Ok(None);
    }

    let scan = LogicalPlan::Scan {
        table: MULTI_AGG_GROUPBY_RESULT_TABLE.to_string(),
        projection: None,
        schema: result_schema.clone(),
    };
    let ctes = CteScope::new();
    let mut plan = scan;

    // HAVING → Filter. Rewrite aggregate calls to result columns by name.
    if let Some(having_sql) = &select.having {
        let mut resolver = NameResolver::empty();
        resolver.ctx = Some(SubqueryCtx {
            provider,
            ctes: &ctes,
        });
        resolver.push_base(MULTI_AGG_GROUPBY_RESULT_TABLE.to_string(), result_schema);
        let predicate = lower_multi_agg_having(having_sql, items, result_schema, &resolver, 0)?;
        validate_having_columns(&predicate, result_schema)?;
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate,
        };
    }

    // ORDER BY → Sort. ORDER BY references the result columns by name; an
    // ORDER BY over an aggregate *call* is rejected (use the alias or the
    // group-key name) to keep this path bounded.
    if let Some(order_by) = &query.order_by {
        let sort_exprs = lower_order_by(
            &order_by.exprs,
            SubqueryCtx {
                provider,
                ctes: &ctes,
            },
        )?;
        if !sort_exprs.is_empty() {
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                sort_exprs,
            };
        }
    }

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
    Ok(Some(plan))
}

/// Lower a HAVING predicate for the generalized COUNT(DISTINCT)+GROUP BY path.
///
/// Any aggregate call (`COUNT(DISTINCT col)`, `SUM(x)`, `COUNT(*)`, …) that
/// structurally matches one of the SELECT-list aggregate items is rewritten to
/// a reference to that result column (by its result-schema name). Group-key
/// references and literals lower ordinarily. An aggregate in HAVING that does
/// NOT appear in the SELECT list is rejected (the host path only materialises
/// the projected aggregates).
fn lower_multi_agg_having(
    e: &SqlExpr,
    items: &[(SqlExpr, Option<String>)],
    result_schema: &Schema,
    resolver: &NameResolver<'_>,
    depth: usize,
) -> BoltResult<Expr> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    // If this subexpression is itself an aggregate call, map it to a result
    // column. We detect aggregate-ness by NAME (no argument lowering — the
    // result-schema resolver doesn't carry the base columns) and match by
    // structural equality of the SQL aggregate expression against the SELECT
    // items (the same conservative compare the detector uses for group keys).
    let is_agg = matches!(try_count_distinct(e, resolver), Ok(Some(_)))
        || matches!(try_plain_agg_tag(e), Ok(Some(_)));
    if is_agg {
        // Find the SELECT-list output column whose aggregate item equals `e`.
        for (out_idx, (item_expr, _)) in items.iter().enumerate() {
            if item_expr == e {
                return Ok(Expr::Column(result_schema.fields[out_idx].name.clone()));
            }
        }
        return Err(BoltError::Sql(
            "HAVING references an aggregate that is not in the SELECT list; \
             only projected aggregates may be filtered in a COUNT(DISTINCT) \
             GROUP BY query"
                .into(),
        ));
    }
    match e {
        SqlExpr::Nested(inner) => {
            lower_multi_agg_having(inner, items, result_schema, resolver, depth + 1)
        }
        SqlExpr::BinaryOp { left, op, right } => {
            let lop = lower_binary_op(op)?;
            let l = lower_multi_agg_having(left, items, result_schema, resolver, depth + 1)?;
            let r = lower_multi_agg_having(right, items, result_schema, resolver, depth + 1)?;
            Ok(Expr::Binary {
                op: lop,
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        SqlExpr::UnaryOp { op, expr } => {
            let inner = lower_multi_agg_having(expr, items, result_schema, resolver, depth + 1)?;
            match op {
                UnaryOperator::Plus => Ok(inner),
                // Negate structurally as `0 - operand` (mirrors the other
                // HAVING lowerers; there is no `UnaryOp::Neg`).
                UnaryOperator::Minus => Ok(Expr::Binary {
                    op: BinaryOp::Sub,
                    left: Box::new(Expr::Literal(Literal::Int64(0))),
                    right: Box::new(inner),
                }),
                UnaryOperator::Not => Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(inner),
                }),
                other => Err(BoltError::Sql(format!(
                    "unsupported unary operator in HAVING: {other:?}"
                ))),
            }
        }
        // IS [NOT] NULL and everything else defers to the ordinary lowerer,
        // which resolves group-key column references against `result_schema`.
        _ => lower_expr(e, resolver, depth + 1),
    }
}

/// Detect and plan a top-level `WITH RECURSIVE` query (feature F1).
///
/// Returns `Ok(Some(plan))` when `sql` is a single `SELECT` statement carrying
/// a `WITH RECURSIVE` clause; `Ok(None)` when it is any other (non-recursive)
/// query — the caller should then fall back to the ordinary [`parse`] path.
/// A malformed / unsupported recursive shape returns `Err` with a precise
/// message.
///
/// # Supported shape (end-to-end)
///
/// The common linear case: exactly ONE CTE in the `WITH RECURSIVE` list, whose
/// body is `<anchor> UNION [ALL] <recursive term>` with a single
/// self-reference in the recursive term (linear recursion). An optional
/// column-list alias (`WITH RECURSIVE c(a, b) AS ...`) renames the relation's
/// columns. ORDER BY / LIMIT on the outer query are applied to the main query
/// (the same place the non-recursive path applies them).
///
/// Both NON-LINEAR recursion (a recursive term that scans the CTE more than
/// once — a self-join) and MUTUAL recursion (multiple recursive CTEs that
/// reference each other) are now supported and orchestrated host-side:
/// * a single-CTE query returns [`RecursiveQueryPlan::Single`] — linear when
///   there is exactly one self-reference, naive (`RecursiveCtePlan::naive`)
///   when there is more than one;
/// * a multi-CTE `WITH RECURSIVE` list returns [`RecursiveQueryPlan::Mutual`]
///   — a system advanced in lockstep to a combined fixpoint.
///
/// # Precise rejections (harder shapes)
///
/// * a body that is not a top-level `UNION` / `UNION ALL`;
/// * `UNION BY NAME`;
/// * an anchor term that references any CTE in the system (a recursive anchor
///   has no seed);
/// * a self-reference buried inside a scalar / `IN` subquery rather than a
///   top-level FROM/JOIN (our orchestrator only binds the CTE as a top-level
///   scan source);
/// * a multi-CTE list in which NO member references any CTE (it would never
///   recurse — that is an ordinary non-recursive `WITH`, handled elsewhere).
pub fn plan_recursive_cte(
    sql: &str,
    provider: &dyn TableProvider,
) -> BoltResult<Option<RecursiveQueryPlan>> {
    guard_sql_size(sql)?;
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
        // Not a query at all — let the normal path produce the canonical error.
        _ => return Ok(None),
    };
    // Only a `WITH RECURSIVE` clause is handled here.
    match &query.with {
        Some(with) if with.recursive => {}
        _ => return Ok(None),
    }
    let plan = plan_recursive_query(&query, provider, &CteScope::new(), 0)?;
    Ok(Some(plan))
}

/// Detect a top-level LATERAL-apply query and build its [`LateralApplyPlan`]
/// descriptor, mirroring [`plan_recursive_cte`]. Returns `Ok(None)` for every
/// query that is not a LATERAL apply (the caller then falls through to the
/// ordinary pipeline).
pub fn plan_lateral_apply(
    sql: &str,
    provider: &dyn TableProvider,
) -> BoltResult<Option<LateralApplyPlan>> {
    guard_sql_size(sql)?;
    let dialect = GenericDialect {};
    let mut stmts =
        Parser::parse_sql(&dialect, sql).map_err(|e| parse_error_to_bolt_error(e, sql))?;
    if stmts.len() != 1 {
        // Defer the canonical multi-statement error to the ordinary pipeline.
        return Ok(None);
    }
    let stmt = stmts.remove(0);
    let query = match stmt {
        Statement::Query(q) => q,
        _ => return Ok(None),
    };
    plan_lateral_apply_query(&query, provider)
}

/// Detect and plan a top-level single-`SELECT` whose FROM contains a LATERAL
/// derived table (feature F3 — LATERAL), returning a [`LateralApplyPlan`] the
/// engine executes as a host nested-loop Apply.
///
/// Returns `Ok(Some(descriptor))` ONLY for the supported shape; `Ok(None)` for
/// everything else (so [`Engine::sql`](crate::exec::engine::Engine::sql) falls
/// through to the ordinary pipeline — where a *non-LATERAL* derived table is
/// handled and a LATERAL one reached via the raw `parse()` API still hits the
/// precise rejection in [`lower_table_factor`]). A malformed query of the right
/// *shape* returns `Err`.
///
/// # Supported shape
///
/// A single `SELECT` (no top-level `WITH`, no set-op) whose FROM is
/// `<left items> [, | CROSS JOIN | [LEFT] JOIN ... ON true] LATERAL (<subq>) AS a`
/// where:
/// * exactly one LATERAL derived table appears, and it is the **last** FROM
///   item (a trailing comma item, or the relation of the last join of the last
///   FROM item);
/// * the join onto the LATERAL is a plain comma / `CROSS JOIN` (INNER apply) or
///   `[INNER|LEFT] JOIN LATERAL (...) ON true` (the `ON true` is required —
///   any other ON predicate is rejected, since the apply produces the
///   correlated cross product and a residual join predicate is not modelled);
/// * the LEFT items contain **no** LATERAL (nested LATERAL is rejected);
/// * the LATERAL subquery alias is present and column-list-free.
fn plan_lateral_apply_query(
    query: &Query,
    provider: &dyn TableProvider,
) -> BoltResult<Option<LateralApplyPlan>> {
    // Only a plain single SELECT (no WITH, no set-op) is in scope.
    if query.with.is_some() {
        return Ok(None);
    }
    let select = match query.body.as_ref() {
        SetExpr::Select(s) => s.as_ref(),
        _ => return Ok(None),
    };
    if select.from.is_empty() {
        return Ok(None);
    }

    // Locate the LATERAL derived table. We accept it only as the LAST FROM
    // item — either a trailing comma item whose `relation` is LATERAL, or the
    // `relation` of the last join of the last FROM item. Anything else
    // (LATERAL not last, more than one LATERAL, LATERAL nested in the LEFT)
    // is not our shape: `Ok(None)` for "no LATERAL at all", else `Err` with a
    // precise message.
    let is_lateral = |tf: &TableFactor| matches!(tf, TableFactor::Derived { lateral: true, .. });

    // Count LATERAL occurrences across the whole FROM to reject the multi /
    // not-last shapes precisely (and to short-circuit `Ok(None)` when none).
    let mut lateral_count = 0usize;
    for (i, twj) in select.from.iter().enumerate() {
        if is_lateral(&twj.relation) {
            // A LATERAL as a *base* FROM item (first comma item or any item's
            // relation) only makes sense as a non-first comma item; a leading
            // `FROM LATERAL (...)` has nothing to correlate against.
            lateral_count += 1;
            if i == 0 {
                return Err(BoltError::Sql(
                    "unsupported: leading LATERAL derived table (it has no \
                     left-hand FROM item to correlate against)"
                        .into(),
                ));
            }
        }
        for join in &twj.joins {
            if is_lateral(&join.relation) {
                lateral_count += 1;
            }
        }
    }
    if lateral_count == 0 {
        return Ok(None);
    }
    if lateral_count > 1 {
        return Err(BoltError::Sql(
            "unsupported: more than one LATERAL derived table in FROM \
             (only a single trailing LATERAL apply is supported)"
                .into(),
        ));
    }

    // Identify the single LATERAL and split the FROM into (left items, lateral
    // factor, alias, left_join). We rebuild a "left" Select carrying every FROM
    // item except the trailing LATERAL.
    let last_idx = select.from.len() - 1;
    let last_twj = &select.from[last_idx];

    // Determine where the lateral sits and capture its (subquery, alias) plus
    // the join kind (for INNER vs LEFT JOIN LATERAL and the `ON true` check).
    let (lateral_subquery, lateral_alias, left_join): (&Query, &sqlparser::ast::TableAlias, bool) = {
        if last_twj.joins.is_empty() {
            // The trailing LATERAL must be the last comma item's relation.
            if !is_lateral(&last_twj.relation) {
                // LATERAL is somewhere earlier (not last) — reject precisely.
                return Err(BoltError::Sql(
                    "unsupported: LATERAL derived table must be the last FROM \
                     item (apply runs the correlated subquery per left row)"
                        .into(),
                ));
            }
            match &last_twj.relation {
                TableFactor::Derived {
                    subquery, alias, ..
                } => {
                    let alias = alias.as_ref().ok_or_else(|| {
                        BoltError::Sql(
                            "LATERAL derived table requires an alias, e.g. \
                             `LATERAL (SELECT ...) AS t`"
                                .into(),
                        )
                    })?;
                    (subquery.as_ref(), alias, false)
                }
                _ => unreachable!("is_lateral matched a Derived factor"),
            }
        } else {
            // The trailing LATERAL is the relation of the LAST join of the last
            // FROM item.
            let last_join = last_twj.joins.last().unwrap();
            if !is_lateral(&last_join.relation) {
                return Err(BoltError::Sql(
                    "unsupported: LATERAL derived table must be the last FROM \
                     item (apply runs the correlated subquery per left row)"
                        .into(),
                ));
            }
            // The join onto the LATERAL must be CROSS / comma (INNER) or
            // [INNER|LEFT] JOIN ... ON true. Any other ON predicate / kind is
            // out of scope.
            let left_join = match &last_join.join_operator {
                JoinOperator::CrossJoin => false,
                JoinOperator::Inner(c) => {
                    require_on_true(c)?;
                    false
                }
                JoinOperator::LeftOuter(c) => {
                    require_on_true(c)?;
                    true
                }
                other => {
                    return Err(BoltError::Sql(format!(
                        "unsupported: JOIN LATERAL kind {other:?}; supported: \
                         comma / CROSS JOIN (INNER apply) or [INNER|LEFT] JOIN \
                         LATERAL (...) ON true"
                    )));
                }
            };
            match &last_join.relation {
                TableFactor::Derived {
                    subquery, alias, ..
                } => {
                    let alias = alias.as_ref().ok_or_else(|| {
                        BoltError::Sql(
                            "LATERAL derived table requires an alias, e.g. \
                             `LATERAL (SELECT ...) AS t`"
                                .into(),
                        )
                    })?;
                    (subquery.as_ref(), alias, left_join)
                }
                _ => unreachable!("is_lateral matched a Derived factor"),
            }
        }
    };
    if !lateral_alias.columns.is_empty() {
        return Err(BoltError::Sql(
            "unsupported: LATERAL alias with column list (AS t(c1, c2))".into(),
        ));
    }
    let lateral_qualifier = lateral_alias.name.value.clone();

    // --- Build the LEFT relation + a resolver describing its qualifiers /
    // columns, by walking the FROM items minus the trailing LATERAL. ---
    let ctes = CteScope::new();
    let (left_plan, left_resolver) =
        build_left_relation(select, last_idx, provider, &ctes, is_lateral)?;
    let left_schema = left_plan.schema()?;

    // --- Collect the LATERAL subquery's correlations against the LEFT scope. ---
    let outer_columns = left_resolver.outer_column_names();
    let corrs =
        crate::plan::subquery::collect_correlations(lateral_subquery, &outer_columns, provider)?;

    // Map each correlation to a LEFT output column (by qualifier+col, or by
    // bare col). Build the synthetic outer schema (`__corr_<i>` fields with the
    // matched left column's dtype) + the per-row source index list.
    let mut outer_fields: Vec<Field> = Vec::with_capacity(corrs.len());
    let mut corr_left_indices: Vec<usize> = Vec::with_capacity(corrs.len());
    // The rewrite map: (corr) -> the `__corr_<i>` name to substitute it with.
    let mut corr_to_synth: Vec<(crate::plan::subquery::CorrRef, String)> =
        Vec::with_capacity(corrs.len());
    for (i, corr) in corrs.iter().enumerate() {
        let left_idx = left_resolver.resolve_correlation(corr).map_err(|e| {
            BoltError::Sql(format!(
                "LATERAL subquery references outer column it cannot resolve: {e}"
            ))
        })?;
        let synth = format!("{LATERAL_CORR_COL_PREFIX}{i}");
        outer_fields.push(Field::new(
            synth.clone(),
            left_schema.fields[left_idx].dtype,
            true,
        ));
        corr_left_indices.push(left_idx);
        corr_to_synth.push((corr.clone(), synth));
    }
    let outer_schema = Schema::new(outer_fields);

    // --- Rewrite the LATERAL subquery AST: each correlated reference becomes a
    // scalar subquery `(SELECT __corr_<i> FROM __lateral_outer)`. The single-row
    // outer table the engine binds per row makes that fold (via
    // `resolve_subqueries`) to the row's exact typed value. ---
    let mut rewritten = lateral_subquery.clone();
    rewrite_correlations_in_query(&mut rewritten, &corr_to_synth, &left_resolver, 0)?;

    // Lower the rewritten subquery. The synthetic `__lateral_outer` scalar
    // subqueries resolve against the provider extended with the outer schema.
    let ext_provider = LateralProvider {
        base: provider,
        outer_schema: &outer_schema,
    };
    let lateral_subplan = plan_query(&rewritten, &ext_provider, &CteScope::new(), 1)?;
    let subquery_schema = lateral_subplan.schema()?;

    // --- Combined (applied) schema: left ++ subquery, with the same
    // duplicate-name disambiguation the engine's JOIN output uses, so the
    // OUTER template can reference both sides unambiguously. ---
    let (combined_schema, sub_output_names) =
        combine_lateral_schemas(&left_schema, &subquery_schema);

    // --- Build the OUTER query template: rewrite the user's SELECT so its FROM
    // is the single `__lateral_apply_result` table and its `left.col` /
    // `alias.col` references become the applied relation's bare column names,
    // then lower it normally. ---
    let post = build_lateral_post(
        select,
        query,
        &left_resolver,
        &lateral_qualifier,
        &subquery_schema,
        &sub_output_names,
        &combined_schema,
    )?;

    Ok(Some(LateralApplyPlan {
        left: left_plan,
        left_schema,
        lateral_subplan,
        subquery_schema,
        outer_schema,
        corr_left_indices,
        combined_schema,
        post,
        left_join,
    }))
}

/// Require that a JOIN constraint is exactly `ON true` (the only predicate a
/// `JOIN LATERAL` apply models — it produces the correlated cross product, so a
/// residual join predicate is out of scope). `USING` / `NATURAL` / a non-`true`
/// ON expression are rejected precisely.
fn require_on_true(c: &JoinConstraint) -> BoltResult<()> {
    match c {
        JoinConstraint::On(SqlExpr::Value(Value::Boolean(true))) => Ok(()),
        JoinConstraint::None => Ok(()),
        _ => Err(BoltError::Sql(
            "unsupported: JOIN LATERAL requires `ON true` (a residual join \
             predicate on a LATERAL apply is not supported — move it into the \
             subquery's WHERE)"
                .into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Correlated WHERE subquery detector (feature F4)
// ---------------------------------------------------------------------------

/// Detect a top-level `SELECT` whose WHERE holds a single **correlated**
/// subquery (correlated `EXISTS` / `NOT EXISTS`, or a scalar-compare conjunct
/// carrying a correlated scalar subquery) and build its
/// [`CorrelatedWherePlan`]. Mirrors [`plan_lateral_apply`]: returns `Ok(None)`
/// for every query that is not such a shape (so the caller falls through), and
/// `Err` only for a malformed query *of this shape*.
pub fn plan_correlated_where(
    sql: &str,
    provider: &dyn TableProvider,
) -> BoltResult<Option<CorrelatedWherePlan>> {
    guard_sql_size(sql)?;
    let dialect = GenericDialect {};
    let mut stmts =
        Parser::parse_sql(&dialect, sql).map_err(|e| parse_error_to_bolt_error(e, sql))?;
    if stmts.len() != 1 {
        return Ok(None);
    }
    let stmt = stmts.remove(0);
    let query = match stmt {
        Statement::Query(q) => q,
        _ => return Ok(None),
    };
    plan_correlated_where_query(&query, provider)
}

/// Inner detector: see [`plan_correlated_where`]. Operates on a parsed `Query`.
fn plan_correlated_where_query(
    query: &Query,
    provider: &dyn TableProvider,
) -> BoltResult<Option<CorrelatedWherePlan>> {
    use crate::plan::subquery::{
        as_exists, count_direct_subqueries, split_and_conjuncts, subquery_is_correlated,
        CorrWhereKind,
    };

    // Only a plain single SELECT (no WITH, no set-op) is in scope. A LATERAL
    // apply is handled by its own (earlier) detector.
    if query.with.is_some() {
        return Ok(None);
    }
    let select = match query.body.as_ref() {
        SetExpr::Select(s) => s.as_ref(),
        _ => return Ok(None),
    };
    if select.from.is_empty() {
        return Ok(None);
    }
    // A WHERE is required for there to be a correlated subquery to find.
    let where_expr = match &select.selection {
        Some(w) => w,
        None => return Ok(None),
    };
    // A LATERAL anywhere in FROM belongs to the LATERAL detector, not here.
    let is_lateral = |tf: &TableFactor| matches!(tf, TableFactor::Derived { lateral: true, .. });
    for twj in &select.from {
        if is_lateral(&twj.relation) || twj.joins.iter().any(|j| is_lateral(&j.relation)) {
            return Ok(None);
        }
    }

    // --- Build the OUTER (left) relation + a resolver for its scope, exactly
    // like the LATERAL path (`build_left_relation` with `last_idx` = the final
    // FROM item and *no* trailing LATERAL to strip). ---
    let ctes = CteScope::new();
    // Pass `last_idx = from.len()` so `build_left_relation` strips *no* trailing
    // LATERAL (there is none here) and includes every FROM item / join.
    let (mut left_plan, left_resolver) =
        build_left_relation(select, select.from.len(), provider, &ctes, is_lateral)?;
    let outer_columns = left_resolver.outer_column_names();

    // --- Split the WHERE into top-level AND conjuncts and find the
    // correlated-subquery conjunct(s). ---
    let conjuncts = split_and_conjuncts(where_expr);
    let mut corr_idx: Option<usize> = None;
    for (i, c) in conjuncts.iter().enumerate() {
        // A conjunct is "correlated" iff it directly carries a subquery whose
        // body references an outer column.
        let mut is_corr = false;
        if let Some((subq, _)) = as_exists(c) {
            is_corr = subquery_is_correlated(subq, &outer_columns, provider)?;
        } else {
            // Scan for a direct scalar `(SELECT ...)` that is correlated.
            for subq in direct_scalar_subqueries(c) {
                if subquery_is_correlated(subq, &outer_columns, provider)? {
                    is_corr = true;
                    break;
                }
            }
        }
        if is_corr {
            if corr_idx.is_some() {
                return Err(BoltError::Sql(
                    "unsupported: more than one correlated subquery in WHERE \
                     (only a single correlated EXISTS / NOT EXISTS / scalar \
                     subquery is supported)"
                        .into(),
                ));
            }
            corr_idx = Some(i);
        }
    }
    let corr_idx = match corr_idx {
        Some(i) => i,
        // No correlated subquery in WHERE — not our shape (an uncorrelated
        // subquery folds to a constant on the ordinary path).
        None => return Ok(None),
    };
    let corr_conjunct = conjuncts[corr_idx];

    // --- Classify the correlated conjunct + capture its template subquery. ---
    let (kind, template): (CorrWhereKind, &Query) = if let Some((subq, negated)) =
        as_exists(corr_conjunct)
    {
        (
            if negated { CorrWhereKind::NotExists } else { CorrWhereKind::Exists },
            subq,
        )
    } else {
        // Scalar-compare conjunct: it must carry EXACTLY ONE direct subquery
        // (the correlated scalar). More than one direct subquery in the same
        // conjunct (e.g. `(SELECT ..) > (SELECT ..)`) is out of scope.
        if count_direct_subqueries(corr_conjunct) != 1 {
            return Err(BoltError::Sql(
                "unsupported: correlated WHERE conjunct with more than one \
                 subquery (a single correlated scalar subquery per conjunct is \
                 supported)"
                    .into(),
            ));
        }
        let subq = direct_scalar_subqueries(corr_conjunct)
            .into_iter()
            .next()
            .ok_or_else(|| {
                BoltError::Sql(
                    "unsupported: correlated WHERE subquery shape (expected a \
                     correlated EXISTS / NOT EXISTS or a scalar comparison \
                     carrying a correlated scalar subquery)"
                        .into(),
                )
            })?;
        (CorrWhereKind::Scalar, subq)
    };

    // --- Apply every ORDINARY (non-correlated) conjunct as a normal Filter on
    // the outer plan. We rebuild a single AND of them and lower it against the
    // left resolver, then wrap the plan in a `Filter`. ---
    let ordinary: Vec<&SqlExpr> = conjuncts
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != corr_idx)
        .map(|(_, c)| *c)
        .collect();
    if !ordinary.is_empty() {
        let folded = and_fold_exprs(&ordinary);
        let predicate = lower_expr(&folded, &left_resolver, 1)?;
        left_plan = LogicalPlan::Filter {
            input: Box::new(left_plan),
            predicate,
        };
    }
    let left_schema = left_plan.schema()?;

    // --- Collect the correlations against the outer scope and build the
    // synthetic `__corr_<i>` outer schema + index list, exactly like the
    // LATERAL path. For EXISTS / NOT EXISTS the correlations live entirely
    // inside the subquery body. For the scalar kind the conjunct *itself* may
    // reference outer columns (e.g. `outer.a > (subquery)`) in addition to the
    // subquery's own correlations — so collect over the whole conjunct. ---
    let corrs = match kind {
        CorrWhereKind::Exists | CorrWhereKind::NotExists => {
            crate::plan::subquery::collect_correlations(template, &outer_columns, provider)?
        }
        CorrWhereKind::Scalar => crate::plan::subquery::collect_conjunct_correlations(
            corr_conjunct,
            &outer_columns,
            provider,
        )?,
    };
    if corrs.is_empty() {
        // Defensive: we only got here because the conjunct *was* correlated.
        return Ok(None);
    }
    let mut outer_fields: Vec<Field> = Vec::with_capacity(corrs.len());
    let mut corr_left_indices: Vec<usize> = Vec::with_capacity(corrs.len());
    let mut corr_to_synth: Vec<(crate::plan::subquery::CorrRef, String)> =
        Vec::with_capacity(corrs.len());
    for (i, corr) in corrs.iter().enumerate() {
        let li = left_resolver.resolve_correlation(corr).map_err(|e| {
            BoltError::Sql(format!(
                "correlated WHERE subquery references outer column it cannot resolve: {e}"
            ))
        })?;
        let synth = format!("{LATERAL_CORR_COL_PREFIX}{i}");
        outer_fields.push(Field::new(synth.clone(), left_schema.fields[li].dtype, true));
        corr_left_indices.push(li);
        corr_to_synth.push((corr.clone(), synth));
    }
    let outer_schema = Schema::new(outer_fields);
    let ext_provider = LateralProvider {
        base: provider,
        outer_schema: &outer_schema,
    };

    // --- Build the per-row test subplan. ---
    let test_subplan = match kind {
        CorrWhereKind::Exists | CorrWhereKind::NotExists => {
            // Re-run the correlated subquery per row; its row count is the test.
            let mut rewritten = template.clone();
            rewrite_correlations_in_query(&mut rewritten, &corr_to_synth, &left_resolver, 0)?;
            plan_query(&rewritten, &ext_provider, &CteScope::new(), 1)?
        }
        CorrWhereKind::Scalar => {
            // `SELECT (<conjunct>) FROM __lateral_outer`, with the conjunct's
            // outer references AND the inner correlated subquery's own
            // correlations rewritten to `(SELECT __corr_<i> FROM __lateral_outer)`.
            // After `resolve_subqueries` folds the (now-uncorrelated) inner
            // scalar subquery and the `__corr` cells, this yields one boolean
            // per row.
            let mut conj = corr_conjunct.clone();
            rewrite_corr_expr(&mut conj, &corr_to_synth, 0)?;
            rewrite_correlations_in_scalar_subqueries(&mut conj, &corr_to_synth, 0)?;
            let test_query = build_scalar_test_query(&conj)?;
            plan_query(&test_query, &ext_provider, &CteScope::new(), 1)?
        }
    };

    // --- Build the OUTER projection template over `__corr_where_result`
    // (== the surviving outer rows, same schema as `left`). The user's
    // qualified `t.col` references resolve against that single table. ---
    let post = build_corr_where_post(select, query, &left_resolver, &left_schema)?;

    Ok(Some(CorrelatedWherePlan {
        left: left_plan,
        left_schema,
        test_subplan,
        outer_schema,
        corr_left_indices,
        kind,
        post,
    }))
}

/// Collect every *direct* scalar `(SELECT ...)` subquery in `e` (not descending
/// into a subquery's own body, mirroring the correlation collector). Excludes
/// `EXISTS` / `IN (SELECT ...)` (handled separately).
fn direct_scalar_subqueries(e: &SqlExpr) -> Vec<&Query> {
    fn walk<'a>(e: &'a SqlExpr, out: &mut Vec<&'a Query>) {
        match e {
            SqlExpr::Subquery(q) => out.push(q.as_ref()),
            // Do not descend into EXISTS / IN subquery bodies.
            SqlExpr::Exists { .. } | SqlExpr::InSubquery { .. } => {}
            SqlExpr::Nested(inner)
            | SqlExpr::UnaryOp { expr: inner, .. }
            | SqlExpr::IsNull(inner)
            | SqlExpr::IsNotNull(inner)
            | SqlExpr::Cast { expr: inner, .. } => walk(inner, out),
            SqlExpr::BinaryOp { left, right, .. } => {
                walk(left, out);
                walk(right, out);
            }
            SqlExpr::Between {
                expr, low, high, ..
            } => {
                walk(expr, out);
                walk(low, out);
                walk(high, out);
            }
            SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
                walk(expr, out);
                walk(pattern, out);
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(e, &mut out);
    out
}

/// Fold a non-empty slice of conjuncts into a single left-deep `AND` AST.
fn and_fold_exprs(conjuncts: &[&SqlExpr]) -> SqlExpr {
    let mut iter = conjuncts.iter();
    let first = (*iter.next().expect("non-empty conjunct list")).clone();
    iter.fold(first, |acc, c| SqlExpr::BinaryOp {
        left: Box::new(acc),
        op: sqlparser::ast::BinaryOperator::And,
        right: Box::new((*c).clone()),
    })
}

/// Rewrite the correlations *inside* every direct scalar `(SELECT ...)`
/// subquery of `e` (the inner correlated scalar subquery), so that after the
/// rewrite the inner subquery is uncorrelated and folds normally. The outer
/// references in `e` itself are rewritten separately by [`rewrite_corr_expr`].
fn rewrite_correlations_in_scalar_subqueries(
    e: &mut SqlExpr,
    corr_to_synth: &[(crate::plan::subquery::CorrRef, String)],
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    match e {
        SqlExpr::Subquery(q) => {
            rewrite_correlations_in_query(q, corr_to_synth, &NameResolver::empty(), depth + 1)
        }
        // Do not descend into EXISTS / IN bodies here.
        SqlExpr::Exists { .. } | SqlExpr::InSubquery { .. } => Ok(()),
        SqlExpr::Nested(inner)
        | SqlExpr::UnaryOp { expr: inner, .. }
        | SqlExpr::IsNull(inner)
        | SqlExpr::IsNotNull(inner)
        | SqlExpr::Cast { expr: inner, .. } => {
            rewrite_correlations_in_scalar_subqueries(inner, corr_to_synth, depth + 1)
        }
        SqlExpr::BinaryOp { left, right, .. } => {
            rewrite_correlations_in_scalar_subqueries(left, corr_to_synth, depth + 1)?;
            rewrite_correlations_in_scalar_subqueries(right, corr_to_synth, depth + 1)
        }
        SqlExpr::Between {
            expr, low, high, ..
        } => {
            rewrite_correlations_in_scalar_subqueries(expr, corr_to_synth, depth + 1)?;
            rewrite_correlations_in_scalar_subqueries(low, corr_to_synth, depth + 1)?;
            rewrite_correlations_in_scalar_subqueries(high, corr_to_synth, depth + 1)
        }
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
            rewrite_correlations_in_scalar_subqueries(expr, corr_to_synth, depth + 1)?;
            rewrite_correlations_in_scalar_subqueries(pattern, corr_to_synth, depth + 1)
        }
        _ => Ok(()),
    }
}

/// Build the `SELECT (<conjunct>) FROM __lateral_outer` query whose single
/// boolean cell is the per-row scalar test. The conjunct has already had its
/// outer references + inner correlations rewritten to `__lateral_outer`
/// lookups, so the query is self-contained against the synthetic outer table.
fn build_scalar_test_query(conjunct: &SqlExpr) -> BoltResult<Query> {
    // Parse a skeleton and splice the rewritten conjunct in as the sole
    // projection — robust to sqlparser struct churn (same approach as
    // `lateral_corr_subquery`).
    let sql = format!("SELECT true AS __corr_test FROM {LATERAL_OUTER_TABLE}");
    let dialect = GenericDialect {};
    let mut stmts =
        Parser::parse_sql(&dialect, &sql).map_err(|e| parse_error_to_bolt_error(e, &sql))?;
    let stmt = stmts.remove(0);
    let mut query = match stmt {
        Statement::Query(q) => *q,
        _ => unreachable!("synthetic scalar-test query is always a SELECT"),
    };
    if let SetExpr::Select(select) = query.body.as_mut() {
        select.projection = vec![SelectItem::ExprWithAlias {
            expr: conjunct.clone(),
            alias: Ident::new("__corr_test"),
        }];
    }
    Ok(query)
}

/// Build the OUTER projection template for a correlated-WHERE plan: rewrite the
/// user's SELECT so its FROM is the single [`CORR_WHERE_RESULT_TABLE`] (== the
/// surviving outer rows) and its `qualifier.col` references resolve against
/// that relation, then lower it (projection / GROUP BY / HAVING / ORDER BY /
/// LIMIT). Reuses [`LateralRefMap`] + [`build_lateral_post`]-style rewriting,
/// but with NO subquery side (the correlated subquery has been consumed by the
/// per-row test) — so the ref map only carries the left (outer) columns.
fn build_corr_where_post(
    select: &Select,
    query: &Query,
    left_resolver: &NameResolver<'_>,
    left_schema: &Schema,
) -> BoltResult<LogicalPlan> {
    // A ref map over the left scope only (no lateral subquery columns): an
    // empty subquery schema makes `LateralRefMap::build` register just the
    // outer `qualifier.col` / bare-`col` → output-name entries.
    let empty_sub = Schema::new(Vec::new());
    let map = LateralRefMap::build(
        left_resolver,
        "__corr_where_no_alias",
        &empty_sub,
        &[],
        left_schema,
    );

    // Rewrite + repoint the SELECT at the result table, dropping the WHERE (it
    // has been fully applied by the Filter + per-row test).
    let mut out_select = select.clone();
    out_select.selection = None;
    out_select.from = vec![sqlparser::ast::TableWithJoins {
        relation: TableFactor::Table {
            name: ObjectName(vec![Ident::new(CORR_WHERE_RESULT_TABLE)]),
            alias: None,
            args: None,
            with_hints: Vec::new(),
            version: None,
            partitions: Vec::new(),
            with_ordinality: false,
        },
        joins: Vec::new(),
    }];
    rewrite_refs_in_select(&mut out_select, &map)?;

    let result_provider = CorrWhereResultProvider { schema: left_schema };
    let ctes = CteScope::new();
    let mut plan = plan_select(&out_select, &result_provider, &ctes, 1)?;

    // ORDER BY / LIMIT / OFFSET, mirroring `build_lateral_post`.
    if let Some(order_by) = &query.order_by {
        let mut order_exprs = order_by.exprs.clone();
        for ob in &mut order_exprs {
            rewrite_refs_in_expr(&mut ob.expr, &map, 0)?;
        }
        let sort_exprs = lower_order_by(
            &order_exprs,
            SubqueryCtx {
                provider: &result_provider,
                ctes: &ctes,
            },
        )?;
        if !sort_exprs.is_empty() {
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                sort_exprs,
            };
        }
    }
    if !query.limit_by.is_empty() {
        return Err(BoltError::Sql("unsupported: LIMIT BY".into()));
    }
    // FETCH folds into LIMIT, as in `plan_query`.
    let fetch_value = fetch_limit_value(query.fetch.as_ref(), query.limit.is_some())?;
    let limit_value = match &query.limit {
        Some(e) => Some(usize_from_literal(e, "LIMIT")?),
        None => fetch_value,
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
    let _ = plan.schema()?;
    Ok(plan)
}

/// A [`TableProvider`] exposing exactly the surviving-outer-rows schema under
/// [`CORR_WHERE_RESULT_TABLE`]. Mirrors [`LateralResultProvider`].
struct CorrWhereResultProvider<'a> {
    schema: &'a Schema,
}
impl TableProvider for CorrWhereResultProvider<'_> {
    fn schema(&self, name: &str) -> BoltResult<Schema> {
        if name == CORR_WHERE_RESULT_TABLE || name.eq_ignore_ascii_case(CORR_WHERE_RESULT_TABLE) {
            return Ok(self.schema.clone());
        }
        Err(BoltError::Plan(format!(
            "correlated-WHERE OUTER template referenced unknown table '{name}' \
             (only the surviving outer relation is in scope)"
        )))
    }
}

/// A [`TableProvider`] wrapper that additionally resolves the synthetic
/// per-row [`LATERAL_OUTER_TABLE`] to a fixed `outer_schema`, so the rewritten
/// LATERAL subquery (whose correlations became
/// `(SELECT __corr_<i> FROM __lateral_outer)`) lowers cleanly. Every other name
/// delegates to the base provider.
struct LateralProvider<'a> {
    base: &'a dyn TableProvider,
    outer_schema: &'a Schema,
}

impl TableProvider for LateralProvider<'_> {
    fn schema(&self, name: &str) -> BoltResult<Schema> {
        if name == LATERAL_OUTER_TABLE || name.eq_ignore_ascii_case(LATERAL_OUTER_TABLE) {
            return Ok(self.outer_schema.clone());
        }
        self.base.schema(name)
    }
    fn schema_version(&self) -> u64 {
        self.base.schema_version()
    }
}

/// Build the LEFT relation (all FROM items except the trailing LATERAL) and a
/// [`NameResolver`] describing its qualifier / column namespace.
///
/// This mirrors the base + join-step construction in [`plan_select`], but stops
/// before the trailing LATERAL item: for the last FROM item it includes every
/// join *except* the final one (which is the LATERAL). A LATERAL anywhere in
/// the LEFT portion is rejected (nested LATERAL is out of scope).
fn build_left_relation<'a>(
    select: &Select,
    last_idx: usize,
    provider: &'a dyn TableProvider,
    ctes: &'a CteScope,
    is_lateral: impl Fn(&TableFactor) -> bool,
) -> BoltResult<(LogicalPlan, NameResolver<'a>)> {
    let first = &select.from[0];
    if is_lateral(&first.relation) {
        return Err(BoltError::Sql(
            "unsupported: LATERAL as a base FROM item".into(),
        ));
    }
    let (base_plan, base_qualifier, scan_schema) =
        lower_table_factor(&first.relation, provider, ctes, 1)?;
    let mut resolver = NameResolver::empty();
    resolver.ctx = Some(SubqueryCtx { provider, ctes });
    resolver.push_base(base_qualifier, &scan_schema);
    let mut plan = base_plan;

    // Walk the join steps exactly as `plan_select`, but excluding the trailing
    // LATERAL (the last join of the last FROM item, or the last comma item).
    for (from_idx, item) in select.from.iter().enumerate() {
        // The trailing comma LATERAL item contributes no left step.
        if from_idx == last_idx && item.joins.is_empty() {
            // This is the LATERAL comma item — skip entirely.
            continue;
        }
        if from_idx > 0 {
            if is_lateral(&item.relation) {
                return Err(BoltError::Sql(
                    "unsupported: LATERAL as a non-trailing FROM item".into(),
                ));
            }
            let (rhs_plan, rhs_qualifier, rhs_schema) =
                lower_table_factor(&item.relation, provider, ctes, 1)?;
            resolver.push_join(rhs_qualifier, &rhs_schema);
            plan = LogicalPlan::Join {
                left: Box::new(plan),
                right: Box::new(rhs_plan),
                join_type: JoinType::Cross,
                on: Vec::new(),
                filter: None,
            };
        }
        // The number of joins to include: all of them, except — for the last
        // FROM item — drop the final one (it is the LATERAL).
        let join_limit = if from_idx == last_idx {
            item.joins.len() - 1
        } else {
            item.joins.len()
        };
        for join in &item.joins[..join_limit] {
            if is_lateral(&join.relation) {
                return Err(BoltError::Sql(
                    "unsupported: LATERAL as a non-trailing JOIN".into(),
                ));
            }
            let (join_type, constraint): (JoinType, Option<&JoinConstraint>) =
                match &join.join_operator {
                    JoinOperator::Inner(c) => (JoinType::Inner, Some(c)),
                    JoinOperator::LeftOuter(c) => (JoinType::LeftOuter, Some(c)),
                    JoinOperator::RightOuter(c) => (JoinType::RightOuter, Some(c)),
                    JoinOperator::FullOuter(c) => (JoinType::FullOuter, Some(c)),
                    JoinOperator::CrossJoin => (JoinType::Cross, None),
                    other => {
                        return Err(BoltError::Sql(format!(
                            "unsupported join kind: {other:?}; supported: \
                             INNER, LEFT, RIGHT, FULL OUTER, CROSS"
                        )));
                    }
                };
            let (rhs_plan, rhs_qualifier, rhs_schema) =
                lower_table_factor(&join.relation, provider, ctes, 1)?;
            resolver.push_join(rhs_qualifier.clone(), &rhs_schema);
            let rhs_qualifier_for_on = rhs_qualifier;
            let (on_pairs, filter) = match constraint {
                Some(JoinConstraint::On(e)) => {
                    let lowered = lower_join_on(e, &resolver, &rhs_qualifier_for_on)?;
                    (lowered.equi_pairs, lowered.filter)
                }
                Some(JoinConstraint::Using(cols)) => {
                    (desugar_using_columns(cols, &resolver)?, None)
                }
                Some(JoinConstraint::Natural) => (desugar_natural_columns(&resolver)?, None),
                Some(JoinConstraint::None) => {
                    return Err(BoltError::Sql(
                        "JOIN requires an ON, USING, or NATURAL clause".into(),
                    ));
                }
                None => (Vec::new(), None),
            };
            plan = LogicalPlan::Join {
                left: Box::new(plan),
                right: Box::new(rhs_plan),
                join_type,
                on: on_pairs,
                filter,
            };
        }
    }
    Ok((plan, resolver))
}

/// Combine the left + subquery schemas into the applied-relation schema using
/// the same `join_rename` disambiguation the engine's JOIN output uses. Returns
/// the combined schema and the *output* name each subquery column maps to (so
/// the OUTER template can rewrite `alias.col` references).
fn combine_lateral_schemas(left: &Schema, sub: &Schema) -> (Schema, Vec<String>) {
    let mut taken: std::collections::HashSet<String> =
        left.fields.iter().map(|f| f.name.clone()).collect();
    let mut fields: Vec<Field> = left.fields.clone();
    let mut sub_names: Vec<String> = Vec::with_capacity(sub.fields.len());
    for f in &sub.fields {
        let out = join_rename(&f.name, &mut taken);
        sub_names.push(out.clone());
        fields.push(Field::new(out, f.dtype, f.nullable));
    }
    (Schema::new(fields), sub_names)
}

/// Rewrite the OUTER query into a `post` [`LogicalPlan`]: replace its FROM with
/// the single applied-relation table and its `left.col` / `alias.col`
/// references with the applied relation's bare column names, then lower it via
/// [`plan_select`] against a provider exposing that table.
#[allow(clippy::too_many_arguments)]
fn build_lateral_post(
    select: &Select,
    query: &Query,
    left_resolver: &NameResolver<'_>,
    lateral_qualifier: &str,
    subquery_schema: &Schema,
    sub_output_names: &[String],
    combined_schema: &Schema,
) -> BoltResult<LogicalPlan> {
    // Rename map: every spelling the user may use for an applied-relation column
    // → its bare combined-schema name.
    //   * left `qual.col`  → left output name (already a combined name)
    //   * left bare `col`  → left output name (when unambiguous)
    //   * lateral `alias.col` / bare subquery `col` → its combined output name
    let map = LateralRefMap::build(
        left_resolver,
        lateral_qualifier,
        subquery_schema,
        sub_output_names,
        combined_schema,
    );

    // Clone + rewrite the SELECT so FROM is the single result table and all
    // column references are bare combined names.
    let mut out_select = select.clone();
    out_select.from = vec![sqlparser::ast::TableWithJoins {
        relation: TableFactor::Table {
            name: ObjectName(vec![Ident::new(LATERAL_APPLY_RESULT_TABLE)]),
            alias: None,
            args: None,
            with_hints: Vec::new(),
            version: None,
            partitions: Vec::new(),
            with_ordinality: false,
        },
        joins: Vec::new(),
    }];
    rewrite_refs_in_select(&mut out_select, &map)?;

    // Lower the rewritten SELECT against a provider exposing the result table.
    let result_provider = LateralResultProvider {
        schema: combined_schema,
    };
    let ctes = CteScope::new();
    let mut plan = plan_select(&out_select, &result_provider, &ctes, 1)?;

    // Layer the outer query's ORDER BY / LIMIT (mirrors `plan_query`). They run
    // over the projected result, so references are resolved post-projection;
    // rewrite ORDER BY exprs to bare names too.
    if let Some(order_by) = &query.order_by {
        let mut order_exprs = order_by.exprs.clone();
        for ob in &mut order_exprs {
            rewrite_refs_in_expr(&mut ob.expr, &map, 0)?;
        }
        let sort_exprs = lower_order_by(
            &order_exprs,
            SubqueryCtx {
                provider: &result_provider,
                ctes: &ctes,
            },
        )?;
        if !sort_exprs.is_empty() {
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                sort_exprs,
            };
        }
    }
    if !query.limit_by.is_empty() {
        return Err(BoltError::Sql("unsupported: LIMIT BY".into()));
    }
    // FETCH folds into LIMIT, as in `plan_query`.
    let fetch_value = fetch_limit_value(query.fetch.as_ref(), query.limit.is_some())?;
    let limit_value = match &query.limit {
        Some(e) => Some(usize_from_literal(e, "LIMIT")?),
        None => fetch_value,
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
    let _ = plan.schema()?;
    Ok(plan)
}

/// A [`TableProvider`] exposing exactly the applied-relation schema under
/// [`LATERAL_APPLY_RESULT_TABLE`].
struct LateralResultProvider<'a> {
    schema: &'a Schema,
}
impl TableProvider for LateralResultProvider<'_> {
    fn schema(&self, name: &str) -> BoltResult<Schema> {
        if name == LATERAL_APPLY_RESULT_TABLE || name.eq_ignore_ascii_case(LATERAL_APPLY_RESULT_TABLE)
        {
            return Ok(self.schema.clone());
        }
        Err(BoltError::Plan(format!(
            "LATERAL apply OUTER template referenced unknown table '{name}' \
             (only the applied relation is in scope)"
        )))
    }
}

/// Maps each user-spellable applied-relation reference to its bare
/// combined-schema column name, for rewriting the OUTER template.
struct LateralRefMap {
    /// `qualifier.col` (both lower-cased) → combined name.
    qualified: HashMap<(String, String), String>,
    /// bare `col` (lower-cased) → combined name, only for names unambiguous
    /// across the whole applied relation.
    bare: HashMap<String, String>,
}

impl LateralRefMap {
    fn build(
        left_resolver: &NameResolver<'_>,
        lateral_qualifier: &str,
        subquery_schema: &Schema,
        sub_output_names: &[String],
        combined_schema: &Schema,
    ) -> Self {
        let mut qualified: HashMap<(String, String), String> = HashMap::new();
        // bare_counts tracks ambiguity across the *combined* relation.
        let mut bare_counts: HashMap<String, usize> = HashMap::new();
        let mut bare: HashMap<String, String> = HashMap::new();

        // Left side: each (qualifier, original col) → its output (combined) name.
        for scope in &left_resolver.tables {
            let q = scope.name.to_ascii_lowercase();
            for c in &scope.cols {
                qualified.insert(
                    (q.clone(), c.original.to_ascii_lowercase()),
                    c.output.clone(),
                );
                *bare_counts.entry(c.original.to_ascii_lowercase()).or_insert(0) += 1;
                bare.insert(c.original.to_ascii_lowercase(), c.output.clone());
            }
        }
        // Lateral side: `alias.col` and bare `col` → its combined output name.
        let q = lateral_qualifier.to_ascii_lowercase();
        for (f, out) in subquery_schema.fields.iter().zip(sub_output_names) {
            qualified.insert((q.clone(), f.name.to_ascii_lowercase()), out.clone());
            *bare_counts.entry(f.name.to_ascii_lowercase()).or_insert(0) += 1;
            bare.insert(f.name.to_ascii_lowercase(), out.clone());
        }
        // Keep only unambiguous bare names.
        bare.retain(|k, _| bare_counts.get(k).copied().unwrap_or(0) == 1);
        // Sanity: every combined field name is a valid bare target too (so a
        // `SELECT *`-expanded reference resolves). Already covered by the loops.
        let _ = combined_schema;
        LateralRefMap { qualified, bare }
    }

    /// Resolve a bare identifier to its combined name (if unambiguous).
    fn resolve_bare(&self, col: &str) -> Option<String> {
        self.bare.get(&col.to_ascii_lowercase()).cloned()
    }
    /// Resolve a `qualifier.col` reference to its combined name.
    fn resolve_qualified(&self, qual: &str, col: &str) -> Option<String> {
        self.qualified
            .get(&(qual.to_ascii_lowercase(), col.to_ascii_lowercase()))
            .cloned()
    }
}

/// Build the scalar-subquery AST `(SELECT <synth> FROM __lateral_outer)` used
/// to replace a correlated reference. Parsed from SQL text (rather than
/// hand-built) so it stays robust to sqlparser struct churn.
fn lateral_corr_subquery(synth: &str) -> BoltResult<SqlExpr> {
    let sql = format!("SELECT {synth} FROM {LATERAL_OUTER_TABLE}");
    let dialect = GenericDialect {};
    let mut stmts =
        Parser::parse_sql(&dialect, &sql).map_err(|e| parse_error_to_bolt_error(e, &sql))?;
    let stmt = stmts.remove(0);
    let query = match stmt {
        Statement::Query(q) => q,
        _ => unreachable!("synthetic correlation subquery is always a SELECT"),
    };
    Ok(SqlExpr::Subquery(query))
}

/// Rewrite every correlated reference in a LATERAL subquery `Query` to its
/// `(SELECT __corr_<i> FROM __lateral_outer)` scalar subquery. Does NOT descend
/// into nested subqueries (each validates its own scope; the collector likewise
/// stops at subquery boundaries, so a deeper correlation is not in scope here).
fn rewrite_correlations_in_query(
    query: &mut Query,
    corr_to_synth: &[(crate::plan::subquery::CorrRef, String)],
    left_resolver: &NameResolver<'_>,
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    let _ = left_resolver;
    rewrite_correlations_in_setexpr(query.body.as_mut(), corr_to_synth, depth + 1)
}

fn rewrite_correlations_in_setexpr(
    set: &mut SetExpr,
    corr_to_synth: &[(crate::plan::subquery::CorrRef, String)],
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    match set {
        SetExpr::Select(s) => {
            let select = s.as_mut();
            for item in &mut select.projection {
                match item {
                    SelectItem::UnnamedExpr(e)
                    | SelectItem::ExprWithAlias { expr: e, .. } => {
                        rewrite_corr_expr(e, corr_to_synth, depth + 1)?;
                    }
                    _ => {}
                }
            }
            if let Some(w) = &mut select.selection {
                rewrite_corr_expr(w, corr_to_synth, depth + 1)?;
            }
            if let Some(h) = &mut select.having {
                rewrite_corr_expr(h, corr_to_synth, depth + 1)?;
            }
            if let GroupByExpr::Expressions(exprs, _) = &mut select.group_by {
                for e in exprs {
                    rewrite_corr_expr(e, corr_to_synth, depth + 1)?;
                }
            }
            for twj in &mut select.from {
                for join in &mut twj.joins {
                    if let Some(on) = join_on_expr_mut(&mut join.join_operator) {
                        rewrite_corr_expr(on, corr_to_synth, depth + 1)?;
                    }
                }
            }
            Ok(())
        }
        SetExpr::Query(q) => {
            rewrite_correlations_in_query(q, corr_to_synth, &NameResolver::empty(), depth + 1)
        }
        SetExpr::SetOperation { left, right, .. } => {
            rewrite_correlations_in_setexpr(left, corr_to_synth, depth + 1)?;
            rewrite_correlations_in_setexpr(right, corr_to_synth, depth + 1)
        }
        _ => Ok(()),
    }
}

/// Pull a mutable ON expression out of a `JoinOperator`, if it carries one.
fn join_on_expr_mut(op: &mut JoinOperator) -> Option<&mut SqlExpr> {
    let constraint = match op {
        JoinOperator::Inner(c)
        | JoinOperator::LeftOuter(c)
        | JoinOperator::RightOuter(c)
        | JoinOperator::FullOuter(c) => c,
        _ => return None,
    };
    match constraint {
        JoinConstraint::On(e) => Some(e),
        _ => None,
    }
}

/// Find the synthetic name for a (qualifier, column) correlation, if any.
fn corr_synth_for<'a>(
    corr_to_synth: &'a [(crate::plan::subquery::CorrRef, String)],
    qualifier: Option<&str>,
    column: &str,
) -> Option<&'a str> {
    let q = qualifier.map(|q| q.to_ascii_lowercase());
    let c = column.to_ascii_lowercase();
    corr_to_synth
        .iter()
        .find(|(r, _)| r.qualifier == q && r.column == c)
        .map(|(_, s)| s.as_str())
}

/// Recursively replace correlated `Identifier`/`CompoundIdentifier` nodes in an
/// expression. Subqueries are not descended into (matching the collector).
fn rewrite_corr_expr(
    e: &mut SqlExpr,
    corr_to_synth: &[(crate::plan::subquery::CorrRef, String)],
    depth: usize,
) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    let replacement: Option<SqlExpr> = match &*e {
        SqlExpr::Identifier(ident) => corr_synth_for(corr_to_synth, None, &ident.value)
            .map(lateral_corr_subquery)
            .transpose()?,
        SqlExpr::CompoundIdentifier(parts) if parts.len() >= 2 => {
            corr_synth_for(corr_to_synth, Some(&parts[0].value), &parts[1].value)
                .map(lateral_corr_subquery)
                .transpose()?
        }
        _ => None,
    };
    if let Some(r) = replacement {
        *e = r;
        return Ok(());
    }
    match e {
        SqlExpr::Nested(inner) => rewrite_corr_expr(inner, corr_to_synth, depth + 1),
        SqlExpr::BinaryOp { left, right, .. } => {
            rewrite_corr_expr(left, corr_to_synth, depth + 1)?;
            rewrite_corr_expr(right, corr_to_synth, depth + 1)
        }
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr)
        | SqlExpr::Cast { expr, .. } => rewrite_corr_expr(expr, corr_to_synth, depth + 1),
        SqlExpr::Between {
            expr, low, high, ..
        } => {
            rewrite_corr_expr(expr, corr_to_synth, depth + 1)?;
            rewrite_corr_expr(low, corr_to_synth, depth + 1)?;
            rewrite_corr_expr(high, corr_to_synth, depth + 1)
        }
        SqlExpr::InList { expr, list, .. } => {
            rewrite_corr_expr(expr, corr_to_synth, depth + 1)?;
            for v in list {
                rewrite_corr_expr(v, corr_to_synth, depth + 1)?;
            }
            Ok(())
        }
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
            rewrite_corr_expr(expr, corr_to_synth, depth + 1)?;
            rewrite_corr_expr(pattern, corr_to_synth, depth + 1)
        }
        SqlExpr::Case {
            operand,
            conditions,
            results,
            else_result,
        } => {
            if let Some(op) = operand {
                rewrite_corr_expr(op, corr_to_synth, depth + 1)?;
            }
            for c in conditions {
                rewrite_corr_expr(c, corr_to_synth, depth + 1)?;
            }
            for r in results {
                rewrite_corr_expr(r, corr_to_synth, depth + 1)?;
            }
            if let Some(er) = else_result {
                rewrite_corr_expr(er, corr_to_synth, depth + 1)?;
            }
            Ok(())
        }
        SqlExpr::Function(func) => {
            if let FunctionArguments::List(list) = &mut func.args {
                for arg in &mut list.args {
                    if let FunctionArg::Unnamed(FunctionArgExpr::Expr(ae))
                    | FunctionArg::Named {
                        arg: FunctionArgExpr::Expr(ae),
                        ..
                    } = arg
                    {
                        rewrite_corr_expr(ae, corr_to_synth, depth + 1)?;
                    }
                }
            }
            Ok(())
        }
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            rewrite_corr_expr(expr, corr_to_synth, depth + 1)?;
            if let Some(f) = substring_from {
                rewrite_corr_expr(f, corr_to_synth, depth + 1)?;
            }
            if let Some(f) = substring_for {
                rewrite_corr_expr(f, corr_to_synth, depth + 1)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Rewrite every applied-relation reference in the OUTER template SELECT to its
/// bare combined-schema column name (so it resolves against the single
/// `__lateral_apply_result` table). Wildcards expand naturally against that
/// table's schema, so they are left untouched.
fn rewrite_refs_in_select(select: &mut Select, map: &LateralRefMap) -> BoltResult<()> {
    for item in &mut select.projection {
        match item {
            SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => {
                rewrite_refs_in_expr(e, map, 0)?;
            }
            _ => {}
        }
    }
    if let Some(w) = &mut select.selection {
        rewrite_refs_in_expr(w, map, 0)?;
    }
    if let Some(h) = &mut select.having {
        rewrite_refs_in_expr(h, map, 0)?;
    }
    if let GroupByExpr::Expressions(exprs, _) = &mut select.group_by {
        for e in exprs {
            rewrite_refs_in_expr(e, map, 0)?;
        }
    }
    Ok(())
}

/// Recursively rewrite `qualifier.col` / bare `col` references to their bare
/// combined name. A reference that resolves to no applied-relation column is
/// left as-is (so `plan_select`'s ordinary resolution produces the precise
/// unknown-column error against the result schema).
fn rewrite_refs_in_expr(e: &mut SqlExpr, map: &LateralRefMap, depth: usize) -> BoltResult<()> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    let replacement: Option<String> = match &*e {
        SqlExpr::CompoundIdentifier(parts) if parts.len() >= 2 => {
            map.resolve_qualified(&parts[0].value, &parts[1].value)
        }
        SqlExpr::Identifier(ident) => map.resolve_bare(&ident.value),
        _ => None,
    };
    if let Some(name) = replacement {
        *e = SqlExpr::Identifier(Ident::new(name));
        return Ok(());
    }
    match e {
        SqlExpr::Nested(inner) => rewrite_refs_in_expr(inner, map, depth + 1),
        SqlExpr::BinaryOp { left, right, .. } => {
            rewrite_refs_in_expr(left, map, depth + 1)?;
            rewrite_refs_in_expr(right, map, depth + 1)
        }
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr)
        | SqlExpr::Cast { expr, .. } => rewrite_refs_in_expr(expr, map, depth + 1),
        SqlExpr::Between {
            expr, low, high, ..
        } => {
            rewrite_refs_in_expr(expr, map, depth + 1)?;
            rewrite_refs_in_expr(low, map, depth + 1)?;
            rewrite_refs_in_expr(high, map, depth + 1)
        }
        SqlExpr::InList { expr, list, .. } => {
            rewrite_refs_in_expr(expr, map, depth + 1)?;
            for v in list {
                rewrite_refs_in_expr(v, map, depth + 1)?;
            }
            Ok(())
        }
        SqlExpr::Like { expr, pattern, .. } | SqlExpr::ILike { expr, pattern, .. } => {
            rewrite_refs_in_expr(expr, map, depth + 1)?;
            rewrite_refs_in_expr(pattern, map, depth + 1)
        }
        SqlExpr::Case {
            operand,
            conditions,
            results,
            else_result,
        } => {
            if let Some(op) = operand {
                rewrite_refs_in_expr(op, map, depth + 1)?;
            }
            for c in conditions {
                rewrite_refs_in_expr(c, map, depth + 1)?;
            }
            for r in results {
                rewrite_refs_in_expr(r, map, depth + 1)?;
            }
            if let Some(er) = else_result {
                rewrite_refs_in_expr(er, map, depth + 1)?;
            }
            Ok(())
        }
        SqlExpr::Function(func) => {
            if let FunctionArguments::List(list) = &mut func.args {
                for arg in &mut list.args {
                    if let FunctionArg::Unnamed(FunctionArgExpr::Expr(ae))
                    | FunctionArg::Named {
                        arg: FunctionArgExpr::Expr(ae),
                        ..
                    } = arg
                    {
                        rewrite_refs_in_expr(ae, map, depth + 1)?;
                    }
                }
            }
            Ok(())
        }
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            rewrite_refs_in_expr(expr, map, depth + 1)?;
            if let Some(f) = substring_from {
                rewrite_refs_in_expr(f, map, depth + 1)?;
            }
            if let Some(f) = substring_for {
                rewrite_refs_in_expr(f, map, depth + 1)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Plan a `WITH RECURSIVE` [`Query`] into a [`RecursiveQueryPlan`].
///
/// Dispatches on the number of CTEs in the `WITH RECURSIVE` list:
/// * exactly one → [`plan_single_recursive_query`] →
///   [`RecursiveQueryPlan::Single`] (linear or non-linear/naive);
/// * two or more → [`plan_mutual_recursive_query`] →
///   [`RecursiveQueryPlan::Mutual`] (a lockstep system).
///
/// Assumes `query.with` is present and `recursive`. The outer query's
/// ORDER BY / LIMIT are layered onto the main query in each path.
fn plan_recursive_query(
    query: &Query,
    provider: &dyn TableProvider,
    base: &CteScope,
    depth: usize,
) -> BoltResult<RecursiveQueryPlan> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    let with = query
        .with
        .as_ref()
        .expect("plan_recursive_query called without a WITH clause");
    if with.cte_tables.len() == 1 {
        Ok(RecursiveQueryPlan::Single(plan_single_recursive_query(
            query, provider, base, depth,
        )?))
    } else {
        Ok(RecursiveQueryPlan::Mutual(plan_mutual_recursive_query(
            query, provider, base, depth,
        )?))
    }
}

/// Plan a single-CTE `WITH RECURSIVE` [`Query`] into a [`RecursiveCtePlan`].
///
/// The anchor / recursive / main subplans are lowered as ordinary
/// `LogicalPlan`s; the CTE name is bound (in a private scope used only for
/// lowering the recursive term and main query) to a synthetic `Scan` over the
/// anchor's output schema. The outer query's ORDER BY / LIMIT are layered onto
/// the main query.
///
/// Both linear (one self-reference) and non-linear (a self-join — more than
/// one self-reference) recursion are supported. A non-linear term sets
/// [`RecursiveCtePlan::naive`] so the engine binds the FULL accumulated
/// relation each iteration (the only correct evaluation when all the aliased
/// self-reference scans resolve to the one ephemeral table the engine binds by
/// name).
fn plan_single_recursive_query(
    query: &Query,
    provider: &dyn TableProvider,
    base: &CteScope,
    depth: usize,
) -> BoltResult<RecursiveCtePlan> {
    // `plan_recursive_query` already enforced the depth bound and the
    // single-CTE dispatch; this function handles exactly one recursive CTE.
    let with = query
        .with
        .as_ref()
        .expect("plan_single_recursive_query called without a WITH clause");
    debug_assert_eq!(with.cte_tables.len(), 1);
    let cte = &with.cte_tables[0];
    if cte.from.is_some() {
        return Err(BoltError::Sql(
            "unsupported: CTE materialization hint (FROM)".into(),
        ));
    }
    let name = ident_to_name(&cte.alias.name);

    // Optional column-list alias (`WITH RECURSIVE c(a, b) AS ...`).
    let column_aliases: Option<Vec<String>> = if cte.alias.columns.is_empty() {
        None
    } else {
        Some(
            cte.alias
                .columns
                .iter()
                .map(ident_to_name)
                .collect(),
        )
    };

    // The body must be a top-level UNION / UNION ALL: anchor on the left,
    // recursive term on the right.
    let (anchor_expr, recursive_expr, all) = match cte.query.body.as_ref() {
        SetExpr::SetOperation {
            op: SetOperator::Union,
            set_quantifier,
            left,
            right,
        } => {
            let all = match set_quantifier {
                SetQuantifier::All => true,
                SetQuantifier::Distinct | SetQuantifier::None => false,
                SetQuantifier::ByName
                | SetQuantifier::AllByName
                | SetQuantifier::DistinctByName => {
                    return Err(BoltError::Sql(
                        "unsupported: WITH RECURSIVE with UNION BY NAME".into(),
                    ));
                }
            };
            (left.as_ref(), right.as_ref(), all)
        }
        _ => {
            return Err(BoltError::Sql(
                "unsupported: WITH RECURSIVE body must be \
                 '<anchor> UNION [ALL] <recursive term>'"
                    .into(),
            ));
        }
    };

    // The anchor must NOT reference the CTE (a recursive anchor has no seed).
    if set_expr_references_table(anchor_expr, &name) {
        return Err(BoltError::Sql(format!(
            "unsupported: WITH RECURSIVE anchor term references '{name}' — \
             the anchor (left of UNION) must be non-recursive"
        )));
    }
    let anchor = lower_set_expr(anchor_expr, provider, base, depth + 1)?;
    let anchor_schema = anchor.schema()?;

    // Build the accumulated-relation schema (anchor schema, optionally renamed
    // by the column-list alias) and bind the CTE name to a synthetic Scan over
    // it. Both the recursive term and the main query reference the CTE through
    // this Scan.
    let cte_schema = match &column_aliases {
        Some(aliases) => {
            if aliases.len() != anchor_schema.fields.len() {
                return Err(BoltError::Sql(format!(
                    "WITH RECURSIVE column-list alias names {} columns but \
                     the CTE anchor produces {}",
                    aliases.len(),
                    anchor_schema.fields.len()
                )));
            }
            apply_column_aliases(&anchor_schema, aliases)
        }
        None => anchor_schema.clone(),
    };
    let cte_scan = LogicalPlan::Scan {
        table: name.clone(),
        projection: None,
        schema: cte_schema.clone(),
    };
    let mut scope = base.clone();
    scope.defs.insert(name.clone(), cte_scan);

    // Linearity guard: the recursive term must reference the CTE exactly once
    // at the top-level FROM/JOIN. A reference buried in a scalar/IN subquery is
    // NOT counted here; if that is the ONLY reference, `self_refs == 0` and we
    // reject with the "does not reference" message (the orchestrator cannot
    // bind the CTE inside a subquery).
    let self_refs = count_set_expr_table_refs(recursive_expr, &name);
    if self_refs == 0 {
        return Err(BoltError::Sql(format!(
            "unsupported: WITH RECURSIVE recursive term does not reference \
             '{name}' at the top level (a self-reference inside a subquery is \
             not supported, and a term that never references the CTE would \
             never recurse)"
        )));
    }
    // >1 top-level self-reference is *non-linear* recursion (e.g.
    // `FROM r AS r1, r AS r2` or `r JOIN r` in the recursive term). This is now
    // supported via NAIVE evaluation: every aliased self-reference scan
    // resolves to the SAME ephemeral table the engine binds by `name`, so the
    // engine must bind the FULL accumulated relation (not just the previous
    // delta) each iteration. That is exactly the standard naive semantics —
    // each self-reference evaluates against the full accumulation so far. The
    // semi-naive (delta-only) optimisation is unsafe for a non-linear term, so
    // we mark the plan `naive` and the engine forces the whole-result working
    // set even for `UNION ALL`. (For non-linear `UNION ALL` over cyclic data
    // naive evaluation can grow without bound; the engine's iteration cap is
    // the mandatory guard.) The binder resolves each aliased scan of the CTE
    // name through the single `scope.defs` binding below, which is correct here
    // because all references share one relation per iteration.
    let naive = self_refs > 1;

    // Lower the recursive term against the scope that now binds the CTE, and
    // validate its schema matches the anchor's (column count + per-position
    // dtype) — the two are unioned into the same accumulated relation.
    let recursive = lower_set_expr(recursive_expr, provider, &scope, depth + 1)?;
    let recursive_schema = recursive.schema()?;
    if !recursive_matches_anchor(&anchor_schema, &recursive_schema) {
        return Err(BoltError::Plan(format!(
            "WITH RECURSIVE: anchor and recursive terms have incompatible \
             schemas: anchor produces {} columns, recursive term produces {} \
             columns (per-position dtypes must also match)",
            anchor_schema.fields.len(),
            recursive_schema.fields.len(),
        )));
    }

    // Lower the MAIN query body against the scope that binds the CTE, then
    // layer the outer query's ORDER BY / LIMIT / OFFSET (same handling as the
    // non-recursive path). Other dialect clauses are rejected identically.
    if !query.limit_by.is_empty() {
        return Err(BoltError::Sql("unsupported: LIMIT BY".into()));
    }
    // FETCH folds into LIMIT and FOR UPDATE/SHARE is an accepted no-op, exactly
    // as in the non-recursive `plan_query` path.
    let fetch_value = fetch_limit_value(query.fetch.as_ref(), query.limit.is_some())?;
    let _ = &query.locks;
    let mut main = lower_set_expr(query.body.as_ref(), provider, &scope, depth + 1)?;
    if let Some(order_by) = &query.order_by {
        let sort_exprs = lower_order_by(
            &order_by.exprs,
            SubqueryCtx {
                provider,
                ctes: &scope,
            },
        )?;
        if !sort_exprs.is_empty() {
            main = LogicalPlan::Sort {
                input: Box::new(main),
                sort_exprs,
            };
        }
    }
    let limit_value = match &query.limit {
        Some(e) => Some(usize_from_literal(e, "LIMIT")?),
        None => fetch_value,
    };
    let offset_value = match &query.offset {
        Some(Offset { value, .. }) => Some(usize_from_literal(value, "OFFSET")?),
        None => None,
    };
    if limit_value.is_some() || offset_value.is_some() {
        main = LogicalPlan::Limit {
            input: Box::new(main),
            limit: limit_value.unwrap_or(usize::MAX),
            offset: offset_value.unwrap_or(0),
        };
    }
    // Force a type-check of the main query so a bad reference surfaces here.
    let _ = main.schema()?;

    Ok(RecursiveCtePlan {
        name,
        cte_schema,
        anchor,
        recursive,
        all,
        naive,
        main,
    })
}

/// Plan a multi-CTE `WITH RECURSIVE` [`Query`] into a
/// [`MutualRecursiveCtePlan`] (mutual recursion).
///
/// Each CTE in the list is parsed into an anchor + optional recursive term. A
/// CTE is *recursive* iff its body is a top-level `UNION [ALL]` whose right
/// (recursive) term references ANY CTE in the system; otherwise it is treated
/// as a plain (seeded-once) member. The anchors are lowered against the base
/// scope (anchors may not reference any system CTE). Every recursive term and
/// the main query are then lowered against a scope binding ALL CTE names to
/// synthetic `Scan`s over their accumulated schemas, so a recursive term may
/// reference its own CTE and/or a sibling. The engine advances the whole
/// system in lockstep to a combined fixpoint.
///
/// Schema validation: each recursive CTE's recursive term must be
/// union-compatible with its own anchor (column count + per-position dtype).
fn plan_mutual_recursive_query(
    query: &Query,
    provider: &dyn TableProvider,
    base: &CteScope,
    depth: usize,
) -> BoltResult<MutualRecursiveCtePlan> {
    let with = query
        .with
        .as_ref()
        .expect("plan_mutual_recursive_query called without a WITH clause");

    // --- Pass 1: extract each CTE's name, optional column-list alias, and
    // split its body into (anchor, optional recursive term, all). ---
    struct Parsed<'a> {
        name: String,
        column_aliases: Option<Vec<String>>,
        anchor_expr: &'a SetExpr,
        recursive_expr: Option<&'a SetExpr>,
        all: bool,
    }
    // The set of all CTE names in the system, used to classify a body's right
    // term as recursive (references any system CTE) vs. a plain UNION.
    let mut system_names: Vec<String> = Vec::with_capacity(with.cte_tables.len());
    for cte in &with.cte_tables {
        system_names.push(ident_to_name(&cte.alias.name));
    }
    let references_any_system = |expr: &SetExpr| -> bool {
        system_names
            .iter()
            .any(|n| set_expr_references_table(expr, n))
    };

    let mut parsed: Vec<Parsed> = Vec::with_capacity(with.cte_tables.len());
    for cte in &with.cte_tables {
        if cte.from.is_some() {
            return Err(BoltError::Sql(
                "unsupported: CTE materialization hint (FROM)".into(),
            ));
        }
        let name = ident_to_name(&cte.alias.name);
        let column_aliases: Option<Vec<String>> = if cte.alias.columns.is_empty() {
            None
        } else {
            Some(cte.alias.columns.iter().map(ident_to_name).collect())
        };
        // Detect a duplicate CTE name (the binder would otherwise silently
        // shadow one with another).
        if parsed.iter().any(|p| p.name == name) {
            return Err(BoltError::Sql(format!(
                "unsupported: WITH RECURSIVE declares CTE '{name}' more than once"
            )));
        }
        // Split the body. A top-level UNION whose right term references any
        // system CTE is recursive; anything else is a plain (seeded-once)
        // member of the recursive WITH list.
        let (anchor_expr, recursive_expr, all) = match cte.query.body.as_ref() {
            SetExpr::SetOperation {
                op: SetOperator::Union,
                set_quantifier,
                left,
                right,
            } if references_any_system(right) => {
                let all = match set_quantifier {
                    SetQuantifier::All => true,
                    SetQuantifier::Distinct | SetQuantifier::None => false,
                    SetQuantifier::ByName
                    | SetQuantifier::AllByName
                    | SetQuantifier::DistinctByName => {
                        return Err(BoltError::Sql(
                            "unsupported: WITH RECURSIVE with UNION BY NAME".into(),
                        ));
                    }
                };
                (left.as_ref(), Some(right.as_ref()), all)
            }
            // Not recursive: the whole body is the seed. (A plain CTE in a
            // recursive WITH list is legal SQL.)
            other => (other, None, false),
        };
        parsed.push(Parsed {
            name,
            column_aliases,
            anchor_expr,
            recursive_expr,
            all,
        });
    }

    // At least one member must actually recurse, otherwise this is an ordinary
    // non-recursive WITH (handled by the normal pipeline). The engine hook only
    // routes `recursive` WITH clauses here, but a `WITH RECURSIVE` list whose
    // members never self/cross-reference would loop zero times — reject so it
    // is not silently mistaken for recursion.
    if !parsed.iter().any(|p| p.recursive_expr.is_some()) {
        return Err(BoltError::Sql(
            "unsupported: WITH RECURSIVE list has no recursive member (no CTE's \
             term references any CTE in the list) — a recursive term must \
             reference a CTE in the system at the top-level FROM/JOIN"
                .into(),
        ));
    }

    // The set of *recursive* member names. An anchor may reference an earlier
    // NON-recursive member (a plain seeded CTE, lowered first) but must NOT
    // reference any recursive member (those have no seed value yet — that would
    // be a recursive anchor).
    let recursive_names: Vec<String> = parsed
        .iter()
        .filter(|p| p.recursive_expr.is_some())
        .map(|p| p.name.clone())
        .collect();
    let references_recursive = |expr: &SetExpr| -> bool {
        recursive_names
            .iter()
            .any(|n| set_expr_references_table(expr, n))
    };
    for p in &parsed {
        if p.recursive_expr.is_some() && references_recursive(p.anchor_expr) {
            return Err(BoltError::Sql(format!(
                "unsupported: WITH RECURSIVE anchor term of '{}' references a \
                 recursive CTE — the anchor (left of UNION) must be non-recursive",
                p.name
            )));
        }
    }

    // --- Pass 2a: lower the NON-recursive members first, left-to-right,
    // accumulating them into `nonrec_scope`. A recursive member's anchor (and
    // any later non-recursive member) may reference an earlier non-recursive
    // member through this scope; recursive members are NOT bound here (no seed
    // yet). Each member's accumulated schema + synthetic Scan binding is built
    // in declaration-order slots so passes below can index by position.
    let mut anchors: Vec<Option<LogicalPlan>> = vec![None; parsed.len()];
    let mut cte_schemas: Vec<Option<Schema>> = vec![None; parsed.len()];
    let mut nonrec_scope = base.clone();
    // Helper: given a lowered anchor plan, build the CTE's accumulated schema
    // (honouring any column-list alias) and the synthetic Scan over it.
    let build_schema_and_scan =
        |p: &Parsed<'_>, anchor: &LogicalPlan| -> BoltResult<(Schema, LogicalPlan)> {
            let anchor_schema = anchor.schema()?;
            let cte_schema = match &p.column_aliases {
                Some(aliases) => {
                    if aliases.len() != anchor_schema.fields.len() {
                        return Err(BoltError::Sql(format!(
                            "WITH RECURSIVE column-list alias names {} columns but \
                             the CTE '{}' anchor produces {}",
                            aliases.len(),
                            p.name,
                            anchor_schema.fields.len()
                        )));
                    }
                    apply_column_aliases(&anchor_schema, aliases)
                }
                None => anchor_schema.clone(),
            };
            let cte_scan = LogicalPlan::Scan {
                table: p.name.clone(),
                projection: None,
                schema: cte_schema.clone(),
            };
            Ok((cte_schema, cte_scan))
        };
    for (i, p) in parsed.iter().enumerate() {
        if p.recursive_expr.is_some() {
            continue; // recursive member — handled in pass 2b
        }
        let anchor = lower_set_expr(p.anchor_expr, provider, &nonrec_scope, depth + 1)?;
        let (cte_schema, cte_scan) = build_schema_and_scan(p, &anchor)?;
        nonrec_scope.defs.insert(p.name.clone(), cte_scan);
        anchors[i] = Some(anchor);
        cte_schemas[i] = Some(cte_schema);
    }

    // --- Pass 2b: lower each RECURSIVE member's anchor against the
    // non-recursive scope (it may seed from a non-recursive member but not from
    // a recursive one), and build its schema + binding. ---
    let mut scope = nonrec_scope.clone();
    for (i, p) in parsed.iter().enumerate() {
        if p.recursive_expr.is_none() {
            continue;
        }
        let anchor = lower_set_expr(p.anchor_expr, provider, &nonrec_scope, depth + 1)?;
        let (cte_schema, cte_scan) = build_schema_and_scan(p, &anchor)?;
        scope.defs.insert(p.name.clone(), cte_scan);
        anchors[i] = Some(anchor);
        cte_schemas[i] = Some(cte_schema);
    }
    // Every slot is now filled (every member is either recursive or not).
    let anchors: Vec<LogicalPlan> = anchors.into_iter().map(|a| a.unwrap()).collect();
    let cte_schemas: Vec<Schema> = cte_schemas.into_iter().map(|s| s.unwrap()).collect();

    // --- Pass 3: lower each recursive term against the full system scope and
    // validate it against its own anchor's schema. ---
    let mut terms: Vec<RecursiveCteTerm> = Vec::with_capacity(parsed.len());
    for (i, p) in parsed.iter().enumerate() {
        let recursive = match p.recursive_expr {
            None => None,
            Some(expr) => {
                let plan = lower_set_expr(expr, provider, &scope, depth + 1)?;
                let recursive_schema = plan.schema()?;
                let anchor_schema = anchors[i].schema()?;
                if !recursive_matches_anchor(&anchor_schema, &recursive_schema) {
                    return Err(BoltError::Plan(format!(
                        "WITH RECURSIVE: CTE '{}' anchor and recursive terms have \
                         incompatible schemas: anchor produces {} columns, \
                         recursive term produces {} columns (per-position dtypes \
                         must also match)",
                        p.name,
                        anchor_schema.fields.len(),
                        recursive_schema.fields.len(),
                    )));
                }
                Some(plan)
            }
        };
        terms.push(RecursiveCteTerm {
            name: p.name.clone(),
            cte_schema: cte_schemas[i].clone(),
            anchor: anchors[i].clone(),
            recursive,
            all: p.all,
        });
    }

    // --- Main query: lower against the full system scope, then layer the
    // outer ORDER BY / LIMIT / OFFSET (identical handling to the single path).
    if !query.limit_by.is_empty() {
        return Err(BoltError::Sql("unsupported: LIMIT BY".into()));
    }
    // FETCH folds into LIMIT and FOR UPDATE/SHARE is an accepted no-op, exactly
    // as in the non-recursive `plan_query` path.
    let fetch_value = fetch_limit_value(query.fetch.as_ref(), query.limit.is_some())?;
    let _ = &query.locks;
    let mut main = lower_set_expr(query.body.as_ref(), provider, &scope, depth + 1)?;
    if let Some(order_by) = &query.order_by {
        let sort_exprs = lower_order_by(
            &order_by.exprs,
            SubqueryCtx {
                provider,
                ctes: &scope,
            },
        )?;
        if !sort_exprs.is_empty() {
            main = LogicalPlan::Sort {
                input: Box::new(main),
                sort_exprs,
            };
        }
    }
    let limit_value = match &query.limit {
        Some(e) => Some(usize_from_literal(e, "LIMIT")?),
        None => fetch_value,
    };
    let offset_value = match &query.offset {
        Some(Offset { value, .. }) => Some(usize_from_literal(value, "OFFSET")?),
        None => None,
    };
    if limit_value.is_some() || offset_value.is_some() {
        main = LogicalPlan::Limit {
            input: Box::new(main),
            limit: limit_value.unwrap_or(usize::MAX),
            offset: offset_value.unwrap_or(0),
        };
    }
    // Force a type-check of the main query so a bad reference surfaces here.
    let _ = main.schema()?;

    Ok(MutualRecursiveCtePlan { ctes: terms, main })
}

/// True if a recursive term's schema is union-compatible with the anchor's
/// (same column count, same per-position dtype). Field names need not match —
/// the accumulated relation takes the anchor's names (SQL convention).
fn recursive_matches_anchor(anchor: &Schema, recursive: &Schema) -> bool {
    anchor.fields.len() == recursive.fields.len()
        && anchor
            .fields
            .iter()
            .zip(recursive.fields.iter())
            .all(|(a, r)| a.dtype == r.dtype)
}

/// Count the number of times a `SetExpr` references a table named `target`
/// in any of its (possibly UNION-chained) SELECTs' FROM / JOIN clauses.
///
/// Used as a linearity guard for `WITH RECURSIVE`: the recursive term must
/// name the CTE exactly once at the table level. References inside scalar /
/// IN subqueries are NOT counted here (they are rejected separately — our
/// orchestrator binds the CTE only as a top-level scan source).
fn count_set_expr_table_refs(expr: &SetExpr, target: &str) -> usize {
    match expr {
        SetExpr::Select(select) => select
            .from
            .iter()
            .map(|twj| {
                let mut n = table_factor_refs(&twj.relation, target);
                for join in &twj.joins {
                    n += table_factor_refs(&join.relation, target);
                }
                n
            })
            .sum(),
        SetExpr::Query(q) => count_set_expr_table_refs(q.body.as_ref(), target),
        SetExpr::SetOperation { left, right, .. } => {
            count_set_expr_table_refs(left, target)
                + count_set_expr_table_refs(right, target)
        }
        _ => 0,
    }
}

/// True if `expr` references the table `target` at the top-level FROM/JOIN
/// level (see [`count_set_expr_table_refs`]).
fn set_expr_references_table(expr: &SetExpr, target: &str) -> bool {
    count_set_expr_table_refs(expr, target) > 0
}

/// Count references to a table named `target` in a single `TableFactor`
/// (a bare table whose name folds to `target`).
fn table_factor_refs(tf: &TableFactor, target: &str) -> usize {
    if let TableFactor::Table { name, .. } = tf {
        if name.0.len() == 1 && ident_to_name(&name.0[0]).eq_ignore_ascii_case(target) {
            return 1;
        }
    }
    0
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
    // FETCH FIRST/NEXT n ROWS ONLY is standard-SQL LIMIT. We fold it into the
    // same `Limit` node the `LIMIT` keyword uses (see the LIMIT [OFFSET] block
    // below). `WITH TIES` / `PERCENT` are out of scope and rejected precisely
    // by `fetch_limit_value`; mixing FETCH with a `LIMIT` keyword is ambiguous
    // and rejected there too.
    let fetch_value = fetch_limit_value(query.fetch.as_ref(), query.limit.is_some())?;
    // T-SQL `SELECT TOP n ...` is a row limit too. The `TOP` keyword lives on
    // the SELECT body, but T-SQL applies it *after* ORDER BY, so we lower it
    // here at the query level (above the Sort node added below), folding it into
    // the same `Limit` node as LIMIT / FETCH. A body that is not a bare SELECT
    // (e.g. a UNION) carries no TOP, so this is `None` there. `PERCENT` /
    // `WITH TIES` and a TOP combined with LIMIT/FETCH are rejected precisely by
    // `top_limit_value`.
    let top_value = match query.body.as_ref() {
        SetExpr::Select(s) => {
            top_limit_value(s.top.as_ref(), query.limit.is_some() || query.fetch.is_some())?
        }
        _ => None,
    };
    // FOR UPDATE / FOR SHARE: row-level locking is a no-op for this read-only
    // OLAP engine — there is no concurrent writer to lock out and we never
    // mutate base data, so accepting and ignoring the clause leaves results
    // identical. (Standard SQL permits a lock clause to be a hint; here it has
    // no observable effect.)
    let _ = &query.locks;
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
        let sort_exprs = lower_order_by(
            &order_by.exprs,
            SubqueryCtx {
                provider,
                ctes: active_ctes,
            },
        )?;
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
    //
    // `query.limit` (the `LIMIT` keyword) and `query.fetch` (`FETCH FIRST n
    // ROWS ONLY`) are mutually exclusive (the ambiguous combination is rejected
    // by `fetch_limit_value`), so at most one of the two is `Some`; they share
    // the same `Limit` lowering and both compose with ORDER BY because this
    // block runs *after* the Sort node is layered on above.
    // At most one of LIMIT / FETCH / TOP is set (the ambiguous combinations are
    // rejected above), so `.or` collapses them into a single effective limit.
    let limit_value = match &query.limit {
        Some(e) => Some(usize_from_literal(e, "LIMIT")?),
        None => fetch_value.or(top_value),
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
        SetExpr::Select(s) => plan_select(s.as_ref(), provider, ctes, depth + 1),
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
                    // Reconcile branch column types to a common supertype
                    // (Int32 ∪ Int64 → Int64, Int ∪ Float → Float, …) so
                    // mismatched-but-compatible columns are coerced rather than
                    // rejected, mirroring how VALUES rows are unified.
                    coerce_set_op_branches("UNION", &mut inputs)?;
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
                    // Coerce the two branches to a common per-column supertype
                    // (same rule as UNION above). `coerce_set_op_branches`
                    // operates on a slice, so wrap the pair in an array,
                    // run it, then unpack back into left/right.
                    let mut pair = [left_plan, right_plan];
                    coerce_set_op_branches(set_op.keyword(), &mut pair)?;
                    let [left_plan, right_plan] = pair;
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

/// Reconcile the column types of set-operation branches to a common
/// per-column supertype, inserting casts on the branches that need them.
///
/// SQL UNION / EXCEPT / INTERSECT require that corresponding columns across
/// branches share a type. When the branches differ only in
/// numerically/temporally compatible types (e.g. `Int32` vs `Int64`, `Int`
/// vs `Float`), standard SQL coerces both branches to a common supertype
/// rather than rejecting the query. This mirrors how VALUES rows are unified
/// (see [`values_common_type`] / [`lower_values_relation`]).
///
/// Algorithm:
///   1. Validate every branch has the same column count (a genuine arity
///      mismatch stays an error — `schema_summary`-style messages are emitted
///      later by [`LogicalPlan::schema`], so here we only guard the indexing).
///   2. For each column position, fold the branches' dtypes through
///      [`values_common_type`]. An incompatible pair (e.g. `Int` vs `Utf8`)
///      returns a clear `Err`.
///   3. For each branch whose schema does not already match the computed
///      target dtypes, wrap it in a `Project` that casts each column to its
///      target (preserving the branch's own column names). Branches that
///      already match are left untouched (identical schemas are unchanged).
///
/// `op` is the operator keyword (`"UNION"` / `"EXCEPT"` / `"INTERSECT"`) used
/// in error messages.
fn coerce_set_op_branches(op: &str, branches: &mut [LogicalPlan]) -> BoltResult<()> {
    if branches.len() < 2 {
        // A single branch has nothing to reconcile.
        return Ok(());
    }
    // Compute every branch's schema once.
    let schemas: Vec<Schema> = branches
        .iter()
        .map(|b| b.schema())
        .collect::<BoltResult<Vec<_>>>()?;

    let n_cols = schemas[0].fields.len();
    // A genuine column-count mismatch is left for `LogicalPlan::schema` to
    // report with its richer message; bail out without coercing so we never
    // index past a shorter branch.
    if schemas.iter().any(|s| s.fields.len() != n_cols) {
        return Ok(());
    }

    // Fold each column's dtype across all branches into a common supertype.
    let mut targets: Vec<DataType> = Vec::with_capacity(n_cols);
    for ci in 0..n_cols {
        let mut common = schemas[0].fields[ci].dtype;
        for s in &schemas[1..] {
            let other = s.fields[ci].dtype;
            common = values_common_type(common, other).ok_or_else(|| {
                BoltError::Sql(format!(
                    "{op} branches have incompatible types for column {}: \
                     {common:?} and {other:?}",
                    ci + 1
                ))
            })?;
        }
        targets.push(common);
    }

    // If every branch already matches the targets exactly, there is nothing
    // to insert (identical-schema set-ops stay unchanged).
    let all_match = schemas
        .iter()
        .all(|s| s.fields.iter().zip(&targets).all(|(f, t)| f.dtype == *t));
    if all_match {
        return Ok(());
    }

    // Wrap each non-matching branch in a casting `Project`.
    for (branch, schema) in branches.iter_mut().zip(&schemas) {
        let needs_cast = schema
            .fields
            .iter()
            .zip(&targets)
            .any(|(f, t)| f.dtype != *t);
        if !needs_cast {
            continue;
        }
        let exprs: Vec<Expr> = schema
            .fields
            .iter()
            .zip(&targets)
            .map(|(f, t)| {
                let col = Expr::Column(f.name.clone());
                if f.dtype == *t {
                    // Already the target type: keep the bare column (its name
                    // flows through `Project`'s output-name rule unchanged).
                    col
                } else {
                    // Cast, then alias back to the original column name so the
                    // projected schema keeps the branch's column names rather
                    // than `Project`'s positional `__expr_{i}` placeholder.
                    Expr::Alias(Box::new(col.cast(*t)), f.name.clone())
                }
            })
            .collect();
        // Move the original branch out (replacing it with a cheap empty-Union
        // placeholder) so it can be re-boxed as the Project's input, then
        // overwrite the slot with the wrapping projection.
        let inner = std::mem::replace(branch, LogicalPlan::Union { inputs: Vec::new() });
        *branch = LogicalPlan::Project {
            input: Box::new(inner),
            exprs,
        };
    }
    Ok(())
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
fn lower_order_by(
    exprs: &[OrderByExpr],
    ctx: SubqueryCtx<'_>,
) -> BoltResult<Vec<SortExpr>> {
    // ORDER BY runs outside the FROM-tree (after projection), so no table
    // qualifiers are in scope. We pass a resolver with no table scopes; bare
    // identifiers still lower as column refs against the post-projection
    // schema, and any stray `table.col` ref will fall through to a clean
    // "unknown table qualifier" error.
    //
    // F12: the resolver carries the subquery lowering context (provider + CTE
    // scope) so an uncorrelated `(SELECT ...)` / `x IN (SELECT ...)` in ORDER
    // BY lowers its own nested plan. The exec-side subquery resolver already
    // walks `LogicalPlan::Sort`'s sort_exprs (see
    // `crate::exec::subquery_resolve::resolve_plan`), so these fold to
    // constants before physical lowering just like WHERE/SELECT subqueries.
    // With no outer table scope the correlation detector sees an empty
    // outer-column set, so a stray correlated reference is not silently
    // accepted — it surfaces as a normal unknown-column error during the
    // subquery's own lowering.
    let mut resolver = NameResolver::empty();
    resolver.ctx = Some(ctx);
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
/// Resolve a `FETCH FIRST/NEXT n ROWS ONLY` clause into an optional row limit,
/// to be folded into the same `LogicalPlan::Limit` the `LIMIT` keyword uses.
///
/// `FETCH` is the SQL-standard synonym for `LIMIT`; `FETCH FIRST n ROWS ONLY`
/// and `FETCH NEXT n ROWS ONLY` are identical (the `FIRST`/`NEXT` spelling is
/// cosmetic). Returns `None` when there is no FETCH clause.
///
/// Rejected precisely (genuinely out of scope, not oversights):
///   * `WITH TIES` — would require carrying the ORDER BY peer-group boundary
///     into the limit operator so tied rows at the cutoff are all kept; the
///     `Limit` operator is a plain row-count truncation and cannot express it.
///   * `PERCENT` — a fractional limit (`FETCH FIRST 10 PERCENT ...`) needs the
///     post-sort cardinality to size the cut, which the static `Limit` node
///     does not carry.
///   * a `FETCH` alongside a `LIMIT` keyword (`has_limit_keyword`) — two row
///     limits on one query are ambiguous, so we reject rather than silently
///     pick one.
fn fetch_limit_value(
    fetch: Option<&Fetch>,
    has_limit_keyword: bool,
) -> BoltResult<Option<usize>> {
    let fetch = match fetch {
        Some(f) => f,
        None => return Ok(None),
    };
    if has_limit_keyword {
        return Err(BoltError::Sql(
            "ambiguous row limit: a query may use LIMIT or FETCH FIRST, not both".into(),
        ));
    }
    if fetch.with_ties {
        return Err(BoltError::Sql(
            "unsupported: FETCH FIRST ... WITH TIES (the row-count Limit operator \
             cannot keep ORDER BY ties at the cutoff); use FETCH FIRST n ROWS ONLY"
                .into(),
        ));
    }
    if fetch.percent {
        return Err(BoltError::Sql(
            "unsupported: FETCH FIRST n PERCENT (a fractional limit needs the \
             post-sort row count, which the static Limit operator does not carry); \
             use an absolute FETCH FIRST n ROWS ONLY"
                .into(),
        ));
    }
    // `FETCH FIRST ROWS ONLY` with no quantity defaults to 1 row (SQL standard).
    match &fetch.quantity {
        Some(e) => Ok(Some(usize_from_literal(e, "FETCH FIRST")?)),
        None => Ok(Some(1)),
    }
}

/// Resolve a T-SQL `SELECT TOP n [PERCENT] [WITH TIES] ...` clause into an
/// optional row limit, folded into the same `LogicalPlan::Limit` as LIMIT /
/// FETCH. Returns `None` when there is no TOP clause.
///
/// Rejected precisely:
///   * `PERCENT` / `WITH TIES` — same limitations as FETCH (see
///     [`fetch_limit_value`]): the static row-count `Limit` operator cannot
///     express a fractional cut or keep ORDER BY ties at the cutoff.
///   * a `TOP` alongside a `LIMIT` / `FETCH` clause (`has_other_limit`) — two
///     row limits on one query are ambiguous, so we reject rather than guess.
fn top_limit_value(top: Option<&Top>, has_other_limit: bool) -> BoltResult<Option<usize>> {
    let top = match top {
        Some(t) => t,
        None => return Ok(None),
    };
    if has_other_limit {
        return Err(BoltError::Sql(
            "ambiguous row limit: a query may use TOP or LIMIT/FETCH, not both".into(),
        ));
    }
    if top.percent {
        return Err(BoltError::Sql(
            "unsupported: TOP n PERCENT (a fractional limit needs the post-sort \
             row count, which the static Limit operator does not carry); use an \
             absolute TOP n"
                .into(),
        ));
    }
    if top.with_ties {
        return Err(BoltError::Sql(
            "unsupported: TOP n WITH TIES (the row-count Limit operator cannot keep \
             ORDER BY ties at the cutoff); use a plain TOP n"
                .into(),
        ));
    }
    match &top.quantity {
        Some(TopQuantity::Constant(n)) => Ok(Some(
            usize::try_from(*n)
                .map_err(|_| BoltError::Sql(format!("TOP value {n} exceeds usize range")))?,
        )),
        Some(TopQuantity::Expr(e)) => Ok(Some(usize_from_literal(e, "TOP")?)),
        // `SELECT TOP ... ` with no quantity is not valid T-SQL; treat a
        // missing count defensively as "no limit" rather than erroring.
        None => Ok(None),
    }
}

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

/// Hard cap on the number of *grouping columns* allowed in a single
/// `CUBE` / `ROLLUP` (or `GROUPING SETS`) construct.
///
/// `CUBE(n)` expands to `2^n` grouping sets and each set becomes one full
/// `Aggregate` branch of a `UNION ALL`, so an unbounded column count would let
/// a short query explode into millions of branches (and an exponentially large
/// plan tree). 12 columns → at most 4096 branches, which is already far beyond
/// any realistic analytic query; anything larger is rejected cleanly rather
/// than allowed to blow up the planner. The cap counts *distinct* grouping
/// columns named anywhere in the construct (`ROLLUP(a,b)` and
/// `GROUPING SETS ((a),(b))` each count 2).
const MAX_GROUPING_SET_COLUMNS: usize = 12;

/// A parsed GROUP BY clause after expanding any ROLLUP / CUBE / GROUPING SETS
/// construct into an explicit list of grouping sets.
///
/// `sets` is the list of grouping sets; each set is the list of grouping
/// expressions that are *active* (real GROUP BY keys) for that set. The empty
/// set `[]` is the grand total (group the whole input). `all_cols` is the
/// ordered, de-duplicated union of every column that appears in any set — the
/// full set of grouping columns the result schema carries (the ones that are
/// NULL-filled in sets where they are inactive). `is_super` is true when the
/// clause came from a ROLLUP / CUBE / GROUPING SETS construct (i.e. more than a
/// plain single GROUP BY), which selects the UNION-ALL rewrite path.
///
/// `is_all` is true only for `GROUP BY ALL` (feature F1). Determining the
/// grouping keys for ALL requires the classified SELECT list (all non-aggregate
/// output expressions), which is not available where [`parse_group_by`] runs, so
/// `sets` / `all_cols` are left empty and [`plan_select`] fills them in after
/// SELECT-item classification — see the GROUP BY ALL handling there.
struct ParsedGroupBy {
    sets: Vec<Vec<SqlExpr>>,
    all_cols: Vec<SqlExpr>,
    is_super: bool,
    is_all: bool,
}

/// Parse `select.group_by` into a [`ParsedGroupBy`], expanding any
/// ROLLUP / CUBE / GROUPING SETS construct (feature F2) into an explicit list
/// of grouping sets. A plain `GROUP BY a, b` yields a single set `[[a, b]]`
/// with `is_super = false` so the caller can keep the existing fast path.
///
/// Features handled here:
/// * `GROUP BY ALL` (F1) — returns the `is_all` sentinel; the actual grouping
///   keys (every non-aggregated SELECT column) are filled in by [`plan_select`]
///   once the SELECT list is classified.
/// * `GROUP BY ... WITH TOTALS` (F3, ClickHouse) — rewritten to the grouping
///   sets `{(keys), ()}`, i.e. the normal result UNION ALL one grand-total row.
/// * `GROUP BY ... WITH ROLLUP` / `WITH CUBE` (MySQL trailing modifiers) —
///   rewritten to `ROLLUP(keys)` / `CUBE(keys)` over the listed columns.
///
/// Rejections (precise, documented): CUBE/ROLLUP/GROUPING-SETS columns exceeding
/// [`MAX_GROUPING_SET_COLUMNS`] (combinatorial-blowup guard); a trailing WITH
/// modifier combined with an explicit ROLLUP/CUBE/GROUPING SETS construct or
/// with GROUP BY ALL (ambiguous, ClickHouse-specific); and more than one
/// trailing WITH modifier. A construct may not be mixed with other top-level
/// GROUP BY items, and constructs may not be nested, both of which we reject
/// rather than silently mis-expand.
fn parse_group_by(group_by: &GroupByExpr) -> BoltResult<ParsedGroupBy> {
    let (exprs, modifiers) = match group_by {
        GroupByExpr::All(modifiers) => {
            // GROUP BY ALL = "group by every non-aggregated SELECT column"
            // (DuckDB/Snowflake/ClickHouse semantics; F1). The grouping set is
            // not knowable here (it needs the classified SELECT list), so we
            // return a sentinel and let `plan_select` fill `sets` / `all_cols`
            // in after classification. Trailing WITH modifiers on GROUP BY ALL
            // (a ClickHouse-only combination) are ambiguous to expand before we
            // know the key set, so we reject them precisely.
            if !modifiers.is_empty() {
                return Err(BoltError::Sql(
                    "unsupported: trailing WITH ROLLUP/CUBE/TOTALS modifier combined with \
                     GROUP BY ALL"
                        .into(),
                ));
            }
            return Ok(ParsedGroupBy {
                sets: Vec::new(),
                all_cols: Vec::new(),
                is_super: false,
                is_all: true,
            });
        }
        GroupByExpr::Expressions(exprs, modifiers) => (exprs, modifiers),
    };

    // `WITH ROLLUP` / `WITH CUBE` / `WITH TOTALS` trailing modifiers
    // (MySQL/ClickHouse style) are distinct from the SQL-standard
    // `ROLLUP(...)` / `CUBE(...)` expression constructs handled below.
    //   * WITH TOTALS (F3) ≡ grouping sets `{(keys), ()}` — the normal GROUP BY
    //     result plus a single grand-total row (all keys NULL, aggregates over
    //     the whole input). Reuses the F2 grouping-set machinery.
    //   * WITH ROLLUP ≡ `ROLLUP(keys)`; WITH CUBE ≡ `CUBE(keys)`.
    // A trailing modifier may not be combined with an explicit
    // ROLLUP/CUBE/GROUPING SETS construct (ambiguous), and only one trailing
    // modifier is allowed.
    let trailing_modifier = if modifiers.is_empty() {
        None
    } else {
        if modifiers.len() != 1 {
            return Err(BoltError::Sql(
                "unsupported: more than one trailing GROUP BY WITH modifier".into(),
            ));
        }
        Some(modifiers[0])
    };

    // Detect whether any top-level GROUP BY item is a super-aggregate
    // construct. If so it must be the *sole* item (mixing
    // `GROUP BY a, ROLLUP(b, c)` is valid SQL but adds combinatorial
    // bookkeeping we do not implement yet — reject precisely).
    let has_super = exprs
        .iter()
        .any(|e| matches!(e, SqlExpr::Rollup(_) | SqlExpr::Cube(_) | SqlExpr::GroupingSets(_)));

    if let Some(m) = trailing_modifier {
        if has_super {
            return Err(BoltError::Sql(
                "unsupported: trailing GROUP BY WITH modifier combined with an explicit \
                 ROLLUP/CUBE/GROUPING SETS construct"
                    .into(),
            ));
        }
        // Rewrite the trailing modifier into the equivalent grouping sets over
        // the listed columns, then fall into the shared super-aggregate path
        // below by synthesising the appropriate `sets`.
        return parse_trailing_modifier(exprs, m);
    }

    if !has_super {
        // Plain `GROUP BY a, b [, ...]` (possibly empty). Single grouping set;
        // every column is active, so no NULL-fill / UNION rewrite is needed.
        let cols: Vec<SqlExpr> = exprs.clone();
        return Ok(ParsedGroupBy {
            all_cols: cols.clone(),
            sets: vec![cols],
            is_super: false,
            is_all: false,
        });
    }

    if exprs.len() != 1 {
        return Err(BoltError::Sql(
            "unsupported: ROLLUP/CUBE/GROUPING SETS mixed with other GROUP BY items; \
             the construct must be the sole GROUP BY item"
                .into(),
        ));
    }

    // Reject nested constructs (`ROLLUP(CUBE(a), b)`): the grouping items must
    // be ordinary expressions or parenthesised tuples of them.
    let reject_nested = |item: &SqlExpr| -> BoltResult<()> {
        if matches!(
            item,
            SqlExpr::Rollup(_) | SqlExpr::Cube(_) | SqlExpr::GroupingSets(_)
        ) {
            return Err(BoltError::Sql(
                "unsupported: nested ROLLUP/CUBE/GROUPING SETS construct".into(),
            ));
        }
        Ok(())
    };

    // `items` is the list of *grouping items* the construct operates over.
    // Each item is itself a list of expressions: sqlparser models a composite
    // grouping item `(a, b)` as a multi-element inner Vec, and a simple item
    // `a` as a single-element Vec. ROLLUP/CUBE treat each item atomically.
    let sets: Vec<Vec<SqlExpr>> = match &exprs[0] {
        SqlExpr::Rollup(items) => {
            for grp in items {
                for e in grp {
                    reject_nested(e)?;
                }
            }
            // ROLLUP(i1, i2, ..., in) -> the n+1 prefixes:
            // {(i1..in), (i1..i_{n-1}), ..., (i1), ()}. Flatten each prefix's
            // items into the active-column list.
            let mut out: Vec<Vec<SqlExpr>> = Vec::with_capacity(items.len() + 1);
            for take in (0..=items.len()).rev() {
                out.push(items[..take].iter().flatten().cloned().collect());
            }
            out
        }
        SqlExpr::Cube(items) => {
            // CUBE(i1, ..., in) -> all 2^n subsets of the items.
            if items.len() > MAX_GROUPING_SET_COLUMNS {
                return Err(BoltError::Sql(format!(
                    "GROUP BY CUBE with {} grouping items exceeds the limit of {} \
                     (2^n grouping sets would explode the plan)",
                    items.len(),
                    MAX_GROUPING_SET_COLUMNS
                )));
            }
            for grp in items {
                for e in grp {
                    reject_nested(e)?;
                }
            }
            let n = items.len();
            let mut out: Vec<Vec<SqlExpr>> = Vec::with_capacity(1usize << n);
            // Enumerate subsets via a bitmask; bit `j` set => item `j` present.
            // Iterate the full set (all bits) down to () so the first branch
            // carries every grouping column (cosmetic; schema is name-stable).
            for mask in (0..(1u32 << n)).rev() {
                let mut active: Vec<SqlExpr> = Vec::new();
                for (j, grp) in items.iter().enumerate() {
                    if mask & (1 << j) != 0 {
                        active.extend(grp.iter().cloned());
                    }
                }
                out.push(active);
            }
            out
        }
        SqlExpr::GroupingSets(sets) => {
            for grp in sets {
                for e in grp {
                    reject_nested(e)?;
                }
            }
            // Each inner Vec is an explicit grouping set already.
            sets.clone()
        }
        _ => unreachable!("has_super implies exprs[0] is a super-aggregate construct"),
    };

    // Build the ordered, de-duplicated union of all grouping columns. This is
    // the full set of group keys the result schema carries; in any set where a
    // column is inactive it is emitted as a typed NULL.
    let mut all_cols: Vec<SqlExpr> = Vec::new();
    for set in &sets {
        for col in set {
            if !all_cols.iter().any(|c| sql_expr_struct_eq(c, col)) {
                all_cols.push(col.clone());
            }
        }
    }
    if all_cols.len() > MAX_GROUPING_SET_COLUMNS {
        return Err(BoltError::Sql(format!(
            "GROUP BY ROLLUP/CUBE/GROUPING SETS with {} distinct grouping columns \
             exceeds the limit of {}",
            all_cols.len(),
            MAX_GROUPING_SET_COLUMNS
        )));
    }

    // De-duplicate identical grouping sets (ROLLUP/CUBE/GROUPING SETS can
    // produce repeats — e.g. CUBE over a repeated column, or an explicit
    // GROUPING SETS list with a duplicate). Keeping duplicates would emit
    // duplicate result rows (UNION ALL does not dedup), which is wrong: SQL
    // treats repeated grouping sets as one. Compare by structural equality of
    // the (order-insensitive) column multiset.
    let mut deduped: Vec<Vec<SqlExpr>> = Vec::with_capacity(sets.len());
    for set in sets {
        if !deduped.iter().any(|existing| grouping_sets_eq(existing, &set)) {
            deduped.push(set);
        }
    }
    if deduped.is_empty() {
        // `GROUP BY GROUPING SETS ()` — zero grouping sets. Reject precisely
        // rather than build an empty UNION (which would surface as a less
        // helpful "UNION requires at least one input" plan error downstream).
        return Err(BoltError::Sql(
            "GROUP BY GROUPING SETS requires at least one grouping set".into(),
        ));
    }

    Ok(ParsedGroupBy {
        sets: deduped,
        all_cols,
        is_super: true,
        is_all: false,
    })
}

/// Build a [`ParsedGroupBy`] for a trailing `WITH ROLLUP` / `WITH CUBE` /
/// `WITH TOTALS` modifier over the listed GROUP BY columns `exprs` (features F3
/// and the MySQL trailing forms). The expansion reuses the exact F2 grouping-set
/// semantics:
///   * `WITH TOTALS` ≡ grouping sets `{(c1..cn), ()}` — the normal GROUP BY
///     result plus one grand-total row (all keys NULL, aggregates over the whole
///     input). With no GROUP BY columns this is just the grand total `{()}`.
///   * `WITH ROLLUP` ≡ `ROLLUP(c1, ..., cn)` — the n+1 prefixes.
///   * `WITH CUBE` ≡ `CUBE(c1, ..., cn)` — all `2^n` subsets.
/// The modifier columns must be plain expressions (no nested
/// ROLLUP/CUBE/GROUPING SETS); the column-count cap applies as for F2.
fn parse_trailing_modifier(
    exprs: &[SqlExpr],
    modifier: GroupByWithModifier,
) -> BoltResult<ParsedGroupBy> {
    // The trailing-modifier columns are the plain GROUP BY items; reject any
    // explicit construct mixed in (already excluded by `has_super` check at the
    // call site, but guard defensively for nested forms).
    for e in exprs {
        if matches!(
            e,
            SqlExpr::Rollup(_) | SqlExpr::Cube(_) | SqlExpr::GroupingSets(_)
        ) {
            return Err(BoltError::Sql(
                "unsupported: ROLLUP/CUBE/GROUPING SETS construct combined with a trailing \
                 GROUP BY WITH modifier"
                    .into(),
            ));
        }
    }
    if exprs.len() > MAX_GROUPING_SET_COLUMNS {
        return Err(BoltError::Sql(format!(
            "GROUP BY WITH modifier over {} columns exceeds the limit of {}",
            exprs.len(),
            MAX_GROUPING_SET_COLUMNS
        )));
    }

    let cols: Vec<SqlExpr> = exprs.to_vec();
    let sets: Vec<Vec<SqlExpr>> = match modifier {
        GroupByWithModifier::Totals => {
            // {(keys), ()}: the full set then the grand total.
            vec![cols.clone(), Vec::new()]
        }
        GroupByWithModifier::Rollup => {
            // n+1 prefixes, full set first down to ().
            let mut out: Vec<Vec<SqlExpr>> = Vec::with_capacity(cols.len() + 1);
            for take in (0..=cols.len()).rev() {
                out.push(cols[..take].to_vec());
            }
            out
        }
        GroupByWithModifier::Cube => {
            // All 2^n subsets (full set first).
            let n = cols.len();
            let mut out: Vec<Vec<SqlExpr>> = Vec::with_capacity(1usize << n);
            for mask in (0..(1u32 << n)).rev() {
                let mut active: Vec<SqlExpr> = Vec::new();
                for (j, c) in cols.iter().enumerate() {
                    if mask & (1 << j) != 0 {
                        active.push(c.clone());
                    }
                }
                out.push(active);
            }
            out
        }
    };

    // De-duplicate identical grouping sets (mirrors the F2 path: e.g. WITH
    // TOTALS over an empty GROUP BY collapses {(), ()} to {()}).
    let mut deduped: Vec<Vec<SqlExpr>> = Vec::with_capacity(sets.len());
    for set in sets {
        if !deduped.iter().any(|existing| grouping_sets_eq(existing, &set)) {
            deduped.push(set);
        }
    }

    Ok(ParsedGroupBy {
        all_cols: cols,
        sets: deduped,
        is_super: true,
        is_all: false,
    })
}

/// Structural equality of two grouping sets, treating each as an unordered
/// multiset of column expressions: `(a, b)` and `(b, a)` are the same set.
fn grouping_sets_eq(a: &[SqlExpr], b: &[SqlExpr]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    // O(n^2) match-and-consume; grouping sets are tiny (<= MAX_GROUPING_SET_COLUMNS).
    let mut used = vec![false; b.len()];
    for x in a {
        let mut matched = false;
        for (j, y) in b.iter().enumerate() {
            if !used[j] && sql_expr_struct_eq(x, y) {
                used[j] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            return false;
        }
    }
    true
}

/// Cheap structural equality on the small `SqlExpr` shapes that can appear as
/// grouping columns (identifiers, compound identifiers, and nested forms).
/// Falls back to the `Debug` representation for any other shape so we never
/// mis-merge distinct expressions; the only cost of a false negative here is a
/// harmless duplicate grouping set, which the dtype-identical UNION still
/// accepts.
fn sql_expr_struct_eq(a: &SqlExpr, b: &SqlExpr) -> bool {
    match (a, b) {
        (SqlExpr::Identifier(x), SqlExpr::Identifier(y)) => ident_to_name(x) == ident_to_name(y),
        (SqlExpr::CompoundIdentifier(x), SqlExpr::CompoundIdentifier(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|(p, q)| ident_to_name(p) == ident_to_name(q))
        }
        (SqlExpr::Nested(x), _) => sql_expr_struct_eq(x, b),
        (_, SqlExpr::Nested(y)) => sql_expr_struct_eq(a, y),
        _ => format!("{a:?}") == format!("{b:?}"),
    }
}

/// Reject the `GROUPING(...)` / `GROUPING_ID(...)` / `GROUP_ID()` indicator
/// functions when used WITHOUT a grouping-set construct. `GROUPING()` is only
/// defined relative to a ROLLUP/CUBE/GROUPING SETS (or `WITH TOTALS`)
/// super-aggregate; with a plain GROUP BY (or none) there is no super-aggregate
/// NULL to distinguish, so the indicator is meaningless and we reject it
/// cleanly. With a grouping-set construct present (feature F2) GROUPING() is
/// instead rewritten to a per-branch integer literal — see the
/// `SelectSource::Grouping` path — and this function is not called. We scan the
/// SELECT list and HAVING for any call to one of these names.
fn reject_grouping_indicator(select: &Select) -> BoltResult<()> {
    fn scan(e: &SqlExpr, depth: usize) -> BoltResult<()> {
        if depth > MAX_RECURSION_DEPTH {
            return Err(BoltError::Sql(format!(
                "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
            )));
        }
        if let SqlExpr::Function(f) = e {
            // Function name is the last segment of the object name.
            if let Some(last) = f.name.0.last() {
                let n = ident_to_name(last).to_ascii_lowercase();
                if n == "grouping" || n == "grouping_id" || n == "group_id" {
                    return Err(BoltError::Sql(
                        "GROUPING()/GROUPING_ID()/GROUP_ID() requires GROUP BY with a \
                         ROLLUP/CUBE/GROUPING SETS construct (or WITH TOTALS)"
                            .into(),
                    ));
                }
            }
        }
        // Walk the common nested-expression shapes so an indicator buried
        // inside an arithmetic / comparison expression is still caught.
        match e {
            SqlExpr::Nested(x) | SqlExpr::UnaryOp { expr: x, .. } => scan(x, depth + 1),
            SqlExpr::BinaryOp { left, right, .. } => {
                scan(left, depth + 1)?;
                scan(right, depth + 1)
            }
            SqlExpr::Cast { expr, .. } => scan(expr, depth + 1),
            _ => Ok(()),
        }
    }
    for item in &select.projection {
        match item {
            SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => {
                scan(e, 0)?;
            }
            _ => {}
        }
    }
    if let Some(h) = &select.having {
        scan(h, 0)?;
    }
    Ok(())
}

/// If `e` is a bare top-level `GROUPING(c1, ..., cn)` or `GROUPING_ID(...)` call
/// (feature F2), return its argument expressions; otherwise `None`. Both spell
/// the same indicator (`GROUPING_ID` is the multi-column alias of `GROUPING`),
/// so we treat them identically — each yields the integer bitmask of the
/// per-column grouping bits (MSB = first argument).
///
/// Only the bare top-level call form is recognised. A `GROUPING()` nested inside
/// a larger expression is not matched here; it falls through to the ordinary
/// lowering, where the unknown `grouping` scalar function is rejected with a
/// clear message (the per-branch literal substitution this rewrite performs is
/// only wired for the top-level SELECT-item form).
fn try_grouping_indicator(e: &SqlExpr) -> BoltResult<Option<Vec<&SqlExpr>>> {
    let f = match e {
        SqlExpr::Function(f) => f,
        _ => return Ok(None),
    };
    let name = match f.name.0.last() {
        Some(last) => ident_to_name(last).to_ascii_lowercase(),
        None => return Ok(None),
    };
    if name != "grouping" && name != "grouping_id" {
        return Ok(None);
    }
    // Reject the unusual call decorations the aggregate path also rejects, so a
    // malformed GROUPING(...) surfaces a precise message instead of being
    // silently mis-parsed.
    if !matches!(f.parameters, FunctionArguments::None) {
        return Err(BoltError::Sql(
            "unsupported: parametric GROUPING()".into(),
        ));
    }
    if f.over.is_some() {
        return Err(BoltError::Sql("unsupported: OVER on GROUPING()".into()));
    }
    let list = match &f.args {
        FunctionArguments::List(list) => list,
        FunctionArguments::None | FunctionArguments::Subquery(_) => {
            return Err(BoltError::Sql(
                "GROUPING()/GROUPING_ID() requires at least one column argument".into(),
            ));
        }
    };
    if list.duplicate_treatment.is_some() || !list.clauses.is_empty() {
        return Err(BoltError::Sql(
            "unsupported: DISTINCT/clauses inside GROUPING()".into(),
        ));
    }
    if list.args.is_empty() {
        return Err(BoltError::Sql(
            "GROUPING()/GROUPING_ID() requires at least one column argument".into(),
        ));
    }
    let mut out: Vec<&SqlExpr> = Vec::with_capacity(list.args.len());
    for arg in &list.args {
        match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(inner)) => out.push(inner),
            _ => {
                return Err(BoltError::Sql(
                    "unsupported: GROUPING()/GROUPING_ID() argument must be a plain column".into(),
                ));
            }
        }
    }
    Ok(Some(out))
}

/// Lower a `Select` into Scan [→ Filter] → (Project | Aggregate), optionally
/// wrapped in `Filter` (for HAVING) and/or `Distinct` (for SELECT DISTINCT).
/// Supports a single INNER JOIN in FROM.
fn plan_select(
    select: &Select,
    provider: &dyn TableProvider,
    ctes: &CteScope,
    depth: usize,
) -> BoltResult<LogicalPlan> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    reject_unsupported_select(select)?;

    // FROM: at least one base table reference. JOINs hang off each
    // `TableWithJoins.joins`. A comma-separated FROM list
    // (`FROM a, b, c`) is the SQL spelling of a cross product: it
    // desugars to a left-deep chain of CROSS JOINs
    // `((a CROSS JOIN b) CROSS JOIN c)` — see the cross-join step
    // synthesis after the base is built below. The optimizer's
    // `filter-into-join` pass then folds any `WHERE a.x = b.y` over such
    // a CROSS into the join's residual predicate (it is *not* promoted to
    // a hash-join equi-pair; it routes through the nested-loop join
    // executor), which is correct though not the hash-join fast path.
    if select.from.is_empty() {
        return Err(BoltError::Sql(
            "expected at least one FROM table, got 0".into(),
        ));
    }
    let twj = &select.from[0];

    // Build the base plan from the first table reference. A plain table
    // reference lowers to a `Scan`; a reference that names an in-scope CTE
    // inlines (clones) the CTE's already-lowered plan. `base_qualifier` is the
    // alias (if any) the user-typed `qualifier.col` references must match.
    let (base_plan, base_qualifier, scan_schema) =
        lower_table_factor(&twj.relation, provider, ctes, depth + 1)?;
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
    // Flatten the FROM list into an ordered sequence of join "steps". Each
    // step carries the right-side relation plus the join kind/constraint to
    // splice it onto the accumulated left-deep plan:
    //   * the first FROM item's explicit `JOIN`s (`twj.joins`) carry their
    //     own join kind/constraint, exactly as before;
    //   * every *subsequent* comma-separated FROM item desugars to a leading
    //     CROSS-JOIN step on its `relation` (no constraint — a cartesian
    //     product), followed by that item's own explicit `JOIN`s.
    // `global` is the ClickHouse `GLOBAL JOIN` flag, rejected below; the
    // synthetic CROSS steps are never global.
    struct JoinStep<'a> {
        join_type: JoinType,
        constraint: Option<&'a JoinConstraint>,
        relation: &'a TableFactor,
        global: bool,
    }
    let mut steps: Vec<JoinStep> = Vec::new();
    for (from_idx, item) in select.from.iter().enumerate() {
        // A comma-separated FROM item past the first contributes a leading
        // CROSS-JOIN step on its own `relation` (`a, b` ≡ `a CROSS JOIN b`);
        // the first item's relation is already the base plan.
        if from_idx > 0 {
            steps.push(JoinStep {
                join_type: JoinType::Cross,
                constraint: None,
                relation: &item.relation,
                global: false,
            });
        }
        // Then this item's explicit `JOIN`s, each carrying its own kind.
        for join in &item.joins {
            // Pick out the (join_type, join constraint) pair. CROSS JOIN has
            // no constraint — sqlparser models it with its own variant. We
            // keep the raw `&JoinConstraint` (rather than eagerly extracting
            // an ON expr) so the `USING (...)` / `NATURAL` desugaring below
            // can run *after* the RHS schema is in scope; an ON predicate
            // still routes through `lower_join_on` exactly as before.
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
            steps.push(JoinStep {
                join_type,
                constraint,
                relation: &join.relation,
                global: join.global,
            });
        }
    }

    for step in &steps {
        if step.global {
            return Err(BoltError::Sql(
                "unsupported: GLOBAL JOIN (ClickHouse extension)".into(),
            ));
        }
        let join_type = step.join_type;
        let constraint = step.constraint;
        let (rhs_plan, rhs_qualifier, rhs_schema) =
            lower_table_factor(step.relation, provider, ctes, depth + 1)?;
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
    let scan_schema_for_wildcard: Schema = if steps.is_empty() {
        scan_schema.clone()
    } else {
        plan.schema()?
    };

    // WHERE (+ PREWHERE). PREWHERE is ClickHouse's early-filter clause: it
    // restricts rows *before* the rest of the SELECT runs, but the result set
    // is identical to applying the same predicate in WHERE (only the evaluation
    // order — and thus performance — differs). We therefore fold PREWHERE into
    // the same single `Filter`, ANDing it ahead of WHERE
    // (`PREWHERE p1 WHERE p2` => filter on `p1 AND p2`); PREWHERE alone becomes
    // the filter predicate on its own. Both predicates resolve against the same
    // FROM-tree namespace, so this is a pure conjunction with no reordering of
    // semantics.
    let predicate = match (&select.prewhere, &select.selection) {
        (Some(pre), Some(whr)) => {
            Some(lower_expr(pre, &resolver, 0)?.and(lower_expr(whr, &resolver, 0)?))
        }
        (Some(pre), None) => Some(lower_expr(pre, &resolver, 0)?),
        (None, Some(whr)) => Some(lower_expr(whr, &resolver, 0)?),
        (None, None) => None,
    };
    if let Some(predicate) = predicate {
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

    // GROUP BY (must precede projection decision). Expand any
    // ROLLUP / CUBE / GROUPING SETS construct into an explicit list of
    // grouping sets (feature F2). A plain `GROUP BY a, b` yields a single set
    // with `is_super = false`, preserving the existing fast path below.
    let mut parsed_group_by = parse_group_by(&select.group_by)?;

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

    // GROUP BY ALL (feature F1). Semantics (DuckDB/Snowflake/ClickHouse): group
    // by every non-aggregated column in the SELECT list, then proceed exactly as
    // an explicit `GROUP BY <those keys>`. We can only enumerate that key set now
    // that the SELECT items are expanded, so `parse_group_by` returned an
    // `is_all` sentinel with empty `sets` and we fill them in here. A SELECT item
    // is a grouping key iff it contains no aggregate; the keys are the underlying
    // (pre-alias) SQL expressions, in SELECT order. When the SELECT has NO
    // aggregates at all, every item is a key — so GROUP BY ALL degenerates to
    // grouping by all selected columns (equivalent to SELECT DISTINCT over them),
    // matching DuckDB.
    if parsed_group_by.is_all {
        let mut keys: Vec<SqlExpr> = Vec::new();
        for (sql_expr, _) in &items {
            if !contains_aggregate(sql_expr, &resolver, 0)? {
                // De-duplicate structurally so `SELECT a, a, SUM(x) GROUP BY ALL`
                // does not group by `a` twice.
                if !keys.iter().any(|k| sql_expr_struct_eq(k, sql_expr)) {
                    keys.push(sql_expr.clone());
                }
            }
        }
        parsed_group_by = ParsedGroupBy {
            all_cols: keys.clone(),
            sets: vec![keys],
            is_super: false,
            is_all: true,
        };
    }

    // For the non-super (ordinary) path, `group_by_sql` is the single set's
    // column list — exactly what the original code expected. (For a super
    // aggregate, `sets[0]` is the first grouping set and `group_by_sql` is only
    // consulted on the non-super path below, so this is safe.)
    let group_by_sql: Vec<&SqlExpr> = parsed_group_by.sets[0].iter().collect();
    // GROUPING()/GROUPING_ID()/GROUP_ID() require a grouping-set construct
    // (ROLLUP/CUBE/GROUPING SETS or WITH TOTALS). When one is present (F2),
    // they are rewritten to per-branch integer literals below; reject them
    // cleanly only when used WITHOUT any such construct.
    if !parsed_group_by.is_super {
        reject_grouping_indicator(select)?;
    }

    // `COUNT(DISTINCT col)` — the one DISTINCT-quantified aggregate form we
    // support. It is handled as the *sole* SELECT item; GROUP BY is still
    // rejected (see below), but `HAVING` over the distinct-count and a
    // surrounding `SELECT DISTINCT` are now accepted (F3) and lowered on top of
    // the same base plan. Multiple SELECT items, a GROUP BY, or DISTINCT on a
    // non-COUNT aggregate remain rejected with a precise message so the user is
    // not left guessing. The base plan is
    //   Project ∘ Aggregate(COUNT(*)) ∘ Distinct ∘ Project([col]) ∘ Filter(col IS NOT NULL)
    // which gives the SQL-standard NULL-excluding distinct count: the
    // pre-Distinct Filter drops NULL rows, Distinct dedupes the surviving
    // values (reusing the row-key / NULL canonicalisation in
    // `crate::exec::distinct`), and COUNT(*) over a single-column projection
    // tallies them. With no GROUP BY the whole table is one implicit group, so
    // HAVING lowers to a `Filter` over the result Project and SELECT DISTINCT to
    // a `Distinct` on top — both reuse existing operators (no new plan nodes).
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
            // COUNT(DISTINCT col) with a *plain* GROUP BY — the common shape
            // where the group keys also appear in the SELECT list — is now
            // supported, but as a host-orchestrated engine special-case that is
            // intercepted *before* this parse path (see
            // [`plan_count_distinct_groupby`] and
            // `Engine::execute_count_distinct_groupby`, feature F3-finish). Any
            // query that reaches HERE with a GROUP BY is therefore one the
            // detector deliberately declined and which this parse path CANNOT
            // lower correctly, namely:
            //
            //   * a *super-aggregate* GROUP BY (ROLLUP / CUBE / GROUPING SETS) —
            //     the per-set grouping-set combinatorics are not implemented for
            //     the distinct count;
            //   * a *plain* GROUP BY whose keys are NOT all present in the SELECT
            //     list (e.g. `SELECT COUNT(DISTINCT x) FROM t GROUP BY k`) — the
            //     host orchestrator builds its base projection from the SELECTed
            //     group keys, so it cannot group by a key it cannot see;
            //   * the raw `parse()` API (which bypasses the engine detector
            //     entirely — like WITH RECURSIVE, this feature is only wired into
            //     `Engine::sql` / `Engine::explain_sql`).
            //
            // All of these must be REJECTED here rather than fall through to the
            // no-GROUP-BY lowering below (which would silently ignore the GROUP
            // BY and return a single whole-table distinct count). A general
            // plan-node lowering would still require a new
            // `AggregateExpr::CountDistinct` variant plus planner / executor
            // support in shared files outside this feature's edit set
            // (logical_plan.rs, physical_plan.rs).
            if !group_by_sql.is_empty() || parsed_group_by.is_super {
                return Err(BoltError::Sql(
                    "COUNT(DISTINCT col) with GROUP BY is only supported when the \
                     group-key columns also appear in the SELECT list (and not with \
                     ROLLUP/CUBE/GROUPING SETS); rewrite the query to project the \
                     grouping columns"
                        .into(),
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
            let col_ref = Expr::Column(out_name.clone());
            let proj = match alias {
                Some(a) => col_ref.alias(a.clone()),
                None => col_ref,
            };
            plan = LogicalPlan::Project {
                input: Box::new(aggregate_plan),
                exprs: vec![proj],
            };

            // HAVING over the bare COUNT(DISTINCT col) result. With no GROUP
            // BY the whole table is one implicit group, so HAVING filters that
            // single result row (standard SQL). The aggregate has already been
            // materialised into the Project's output column (`proj_name`); the
            // HAVING predicate references the distinct-count via the same
            // `COUNT(DISTINCT col)` call it was written with, which we resolve
            // to that output column. Reuses the existing `Filter` node — no new
            // execution machinery. Predicate forms beyond the comparison /
            // boolean / IS [NOT] NULL shapes are rejected cleanly inside
            // `lower_having_over_count_distinct`.
            if let Some(having_sql) = &select.having {
                let proj_name = match alias {
                    Some(a) => a.clone(),
                    None => out_name.clone(),
                };
                let predicate = lower_having_over_count_distinct(
                    having_sql,
                    inner,
                    &resolver,
                    &proj_name,
                    0,
                )?;
                // Type-check / column-validate against the Project's schema so a
                // malformed HAVING surfaces here rather than at execution.
                let project_schema = plan.schema()?;
                validate_having_columns(&predicate, &project_schema)?;
                plan = LogicalPlan::Filter {
                    input: Box::new(plan),
                    predicate,
                };
            }

            // SELECT DISTINCT over a bare COUNT(DISTINCT col): the result is a
            // single row, so DISTINCT is a (correct) no-op. We still lower it to
            // the existing `Distinct` node so the plan shape is uniform and the
            // operator's own semantics apply. Reuses `crate::exec::distinct`.
            if matches!(select.distinct, Some(Distinct::Distinct)) {
                plan = LogicalPlan::Distinct {
                    input: Box::new(plan),
                };
            }
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

    if has_agg_in_select || !group_by_sql.is_empty() || parsed_group_by.is_super {
        // Window functions, a named WINDOW clause, and QUALIFY are lowered only
        // in the scalar-projection branch below (which stacks the Window nodes).
        // Combining them with GROUP BY / aggregates (window-over-aggregate) is
        // out of scope, so we reject precisely here rather than silently drop
        // the clause.
        if !select.named_window.is_empty() {
            return Err(BoltError::Sql(
                "unsupported: a named WINDOW clause combined with GROUP BY / \
                 aggregates (window functions over an aggregate result are not lowered)"
                    .into(),
            ));
        }
        if select.qualify.is_some() {
            return Err(BoltError::Sql(
                "unsupported: QUALIFY combined with GROUP BY / aggregates (QUALIFY \
                 filters a window-function result, which is not lowered in aggregate \
                 mode); use HAVING to filter aggregates"
                    .into(),
            ));
        }
        // Aggregate mode. Each SELECT item is one of:
        //   * a bare aggregate call (`SUM(v)`),
        //   * a bare group key already listed in GROUP BY (`k`),
        //   * a scalar expression containing one or more aggregates
        //     (`SUM(price) + 1`, `AVG(qty) * 2`, `(SUM(a) + SUM(b)) / 2`) —
        //     the nested aggregates are extracted as feed inputs and the
        //     surface expression is rewritten with `Column("<agg_out>")` at
        //     each aggregate position. The rewritten expression goes into the
        //     post-Aggregate Project as a computed projection.
        //
        // For ROLLUP / CUBE / GROUPING SETS (`parsed_group_by.is_super`) we
        // build one such Aggregate→Project *branch per grouping set* and
        // combine them with a UNION ALL (feature F2). The SELECT-item
        // classification below is the same for every branch — only which group
        // columns are *active* (real keys) vs NULL-filled differs per set — so
        // we classify once and re-materialise a branch per set.

        // The full ordered list of grouping columns (lowered). For a plain
        // GROUP BY this equals the GROUP BY list; for a super-aggregate it is
        // the de-duplicated union over all sets. A SELECT item that is a group
        // key must match one of these.
        let all_group_by: Vec<Expr> = parsed_group_by
            .all_cols
            .iter()
            .map(|e| lower_expr(e, &resolver, 0))
            .collect::<BoltResult<_>>()?;
        // Type-check each grouping column against the base plan's schema once,
        // up front, so a misnamed group column surfaces here (and so the
        // typed-NULL fills below can use the resolved dtype).
        let base_schema = plan.schema()?;
        let all_group_dtypes: Vec<DataType> = all_group_by
            .iter()
            .map(|g| g.dtype(&base_schema))
            .collect::<BoltResult<_>>()?;

        let mut aggregates: Vec<AggregateExpr> = Vec::new();
        // For each SELECT item, remember how to pull it back out of the Aggregate
        // node's schema (group keys first, aggregates second per `Aggregate::schema()`).
        // Each entry is the *output* column name produced by the Aggregate, plus an
        // optional SELECT alias to rename it to in the final projection.
        enum SelectSource {
            /// SELECT references a group key. `col_idx` is its position in
            /// `all_group_by` / `all_group_dtypes`; per branch it is either a
            /// real key (pulled from the Aggregate by its output name) or, when
            /// inactive in that grouping set, a typed NULL fill.
            GroupKey {
                col_idx: usize,
                alias: Option<String>,
            },
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
            /// SELECT is a `GROUPING(c1, ..., cn)` / `GROUPING_ID(...)` indicator
            /// (feature F2). Each `arg_cols[k]` is the index into `all_group_by`
            /// of the k-th argument column. The value is a compile-time constant
            /// *per branch*: bit `n-1-k` (MSB = first arg) is 1 iff that column
            /// is NOT active in the branch's grouping set (i.e. NULL-filled /
            /// aggregated away). `build_branch` emits the resulting integer
            /// literal for each branch.
            Grouping {
                arg_cols: Vec<usize>,
                alias: Option<String>,
            },
        }
        let mut select_sources: Vec<SelectSource> = Vec::new();

        for (sql_expr, alias) in &items {
            // GROUPING()/GROUPING_ID() indicator (feature F2). Only meaningful
            // with a grouping-set construct (super-aggregate); resolve each
            // argument to its index in `all_group_by` so `build_branch` can emit
            // the per-branch bitmask literal. Rejected when not a super-aggregate
            // (handled above via `reject_grouping_indicator`) and when nested in
            // a larger expression (kept to the bare top-level form for now).
            if let Some(args) = try_grouping_indicator(sql_expr)? {
                if !parsed_group_by.is_super {
                    return Err(BoltError::Sql(
                        "GROUPING()/GROUPING_ID() requires GROUP BY with a ROLLUP/CUBE/\
                         GROUPING SETS construct (or WITH TOTALS)"
                            .into(),
                    ));
                }
                let mut arg_cols: Vec<usize> = Vec::with_capacity(args.len());
                for a in &args {
                    let lowered = lower_expr(*a, &resolver, 0)?;
                    let idx = all_group_by
                        .iter()
                        .position(|g| expr_eq(g, &lowered))
                        .ok_or_else(|| {
                            BoltError::Sql(
                                "GROUPING()/GROUPING_ID() argument must be a GROUP BY \
                                 grouping column"
                                    .into(),
                            )
                        })?;
                    arg_cols.push(idx);
                }
                select_sources.push(SelectSource::Grouping {
                    arg_cols,
                    alias: alias.clone(),
                });
                continue;
            }
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
            // Must match some declared grouping column by structural equality
            // of the lowered form. (For super-aggregates this is the union of
            // all sets — a column that is inactive in *some* set is still a
            // legal SELECT item; it is NULL in the sets where it is inactive.)
            let col_idx = match all_group_by.iter().position(|g| expr_eq(g, &lowered)) {
                Some(i) => i,
                None => {
                    return Err(BoltError::Sql(
                        "non-aggregate SELECT expression must appear in GROUP BY".into(),
                    ));
                }
            };
            select_sources.push(SelectSource::GroupKey {
                col_idx,
                alias: alias.clone(),
            });
        }

        // Build the (aggregate-output-name -> SELECT alias) map so the HAVING
        // lowerer below can rewrite e.g. `SUM(v)` into the alias the Project
        // exposed (otherwise it would lower to `Column("sum_v")`, which no
        // longer exists in the Project's output once an alias renamed it).
        // The aggregates and their aliases are identical across all branches,
        // so this map is computed once.
        let mut having_agg_aliases: HashMap<String, String> = HashMap::new();
        for src in &select_sources {
            if let SelectSource::Aggregate { index, alias: Some(a) } = src {
                having_agg_aliases.insert(aggregate_output_name(&aggregates[*index]), a.clone());
            }
        }

        // Materialise one Aggregate→Project branch for the grouping set
        // `active_cols` (indices into `all_group_by`). Group columns not in
        // `active_cols` are emitted as a typed NULL (`CAST(NULL AS <dtype>)`)
        // aliased to their output name so every branch shares one schema.
        //
        // `base` is this branch's input plan (a fresh clone of the post-WHERE
        // plan per branch). Aggregate output names follow `aggregate_output_name`
        // / `group_key_output_name` (the single source of truth in
        // `logical_plan.rs`); do not duplicate the rules here.
        let build_branch = |base: LogicalPlan, active_cols: &[usize]| -> BoltResult<LogicalPlan> {
            // The Aggregate's GROUP BY list is exactly the active columns, in
            // their order within `all_group_by` (so `group_key_output_name`'s
            // positional `__group_{i}` placeholders are stable per branch).
            let group_by: Vec<Expr> = active_cols.iter().map(|&i| all_group_by[i].clone()).collect();
            let aggregate_plan = LogicalPlan::Aggregate {
                input: Box::new(base),
                group_by,
                aggregates: aggregates.clone(),
            };
            // Validate the aggregate's output dtypes at parse time (this runs
            // `AggregateExpr::output_dtype` over every aggregate), so semantically
            // undefined aggregates — e.g. SUM/AVG over a temporal type — are
            // rejected here with a clear message rather than slipping through to
            // execution.
            let _ = aggregate_plan.schema()?;
            let group_by_out: &[Expr] = match &aggregate_plan {
                LogicalPlan::Aggregate { group_by, .. } => group_by,
                _ => unreachable!("just constructed an Aggregate"),
            };

            let mut proj_exprs: Vec<Expr> = Vec::with_capacity(select_sources.len());
            for (src_idx, src) in select_sources.iter().enumerate() {
                match src {
                    SelectSource::GroupKey { col_idx, alias } => {
                        let active_pos = active_cols.iter().position(|&i| i == *col_idx);
                        match active_pos {
                            Some(pos) => {
                                // Active key: pull from the Aggregate by the
                                // name it received at that position. Preserve
                                // the pre-F2 plan shape — a bare `Column` when
                                // the SELECT item carried no alias.
                                let col =
                                    Expr::Column(group_key_output_name(&group_by_out[pos], pos));
                                proj_exprs.push(match alias {
                                    Some(a) => col.alias(a.clone()),
                                    None => col,
                                });
                            }
                            None => {
                                // Inactive key in this grouping set (only
                                // possible for a super-aggregate): typed NULL so
                                // the branch schema's dtype matches the active
                                // branches (nullable, carrying the
                                // super-aggregate NULL). Alias it to the column's
                                // output name so this branch's schema column name
                                // matches the active branches'.
                                let out_name = alias.clone().unwrap_or_else(|| {
                                    group_key_output_name(&all_group_by[*col_idx], *col_idx)
                                });
                                let null_fill =
                                    Expr::Literal(Literal::Null).cast(all_group_dtypes[*col_idx]);
                                proj_exprs.push(null_fill.alias(out_name));
                            }
                        }
                    }
                    SelectSource::Aggregate { index, alias } => {
                        let name = aggregate_output_name(&aggregates[*index]);
                        let col = Expr::Column(name);
                        proj_exprs.push(match alias {
                            Some(a) => col.alias(a.clone()),
                            None => col,
                        });
                    }
                    SelectSource::Computed { expr, alias } => {
                        let e = expr.clone();
                        proj_exprs.push(match alias {
                            Some(a) => e.alias(a.clone()),
                            None => e,
                        });
                    }
                    SelectSource::Grouping { arg_cols, alias } => {
                        // Per-branch compile-time bitmask. For each argument
                        // column k (MSB = first arg), bit (n-1-k) is 1 iff the
                        // column is NOT active in this branch's grouping set
                        // (i.e. it is aggregated away / NULL-filled). This is the
                        // SQL-standard GROUPING/GROUPING_ID indicator value.
                        let n = arg_cols.len();
                        let mut bits: i64 = 0;
                        for (k, &col_idx) in arg_cols.iter().enumerate() {
                            let active = active_cols.contains(&col_idx);
                            if !active {
                                bits |= 1i64 << (n - 1 - k);
                            }
                        }
                        let lit = Expr::Literal(Literal::Int64(bits));
                        // Stable schema column name across branches: the SELECT
                        // alias if any, else a synthetic positional name (the
                        // value differs per branch but the *name* must match for
                        // the UNION schema check).
                        let out_name = alias
                            .clone()
                            .unwrap_or_else(|| format!("__grouping_{src_idx}"));
                        proj_exprs.push(lit.alias(out_name));
                    }
                }
            }
            Ok(LogicalPlan::Project {
                input: Box::new(aggregate_plan),
                exprs: proj_exprs,
            })
        };

        if parsed_group_by.is_super {
            // One branch per grouping set, combined with UNION ALL. Each branch
            // re-scans the same post-WHERE input (cloned). The branches share
            // an identical output schema (same names, dtypes, nullability) — the
            // group-key fields are nullable (Aggregate keys are nullable and the
            // NULL fills are nullable casts), so the Union schema-check passes.
            let mut branches: Vec<LogicalPlan> = Vec::with_capacity(parsed_group_by.sets.len());
            for set in &parsed_group_by.sets {
                // Map this set's SQL columns to indices into `all_group_by`.
                let mut active_cols: Vec<usize> = Vec::with_capacity(set.len());
                for col in set {
                    let lowered = lower_expr(col, &resolver, 0)?;
                    let idx = all_group_by
                        .iter()
                        .position(|g| expr_eq(g, &lowered))
                        .expect("set columns are a subset of all_cols by construction");
                    if !active_cols.contains(&idx) {
                        active_cols.push(idx);
                    }
                }
                active_cols.sort_unstable();
                branches.push(build_branch(plan.clone(), &active_cols)?);
            }
            plan = LogicalPlan::Union { inputs: branches };
        } else {
            // Plain GROUP BY: a single branch over all group columns. (When
            // there are no aggregates and no group columns this is the bare
            // implicit-group case, handled identically to before.)
            let active_cols: Vec<usize> = (0..all_group_by.len()).collect();
            plan = build_branch(plan, &active_cols)?;
        }

        // HAVING (aggregate mode): the predicate may reference aggregates by
        // call (`SUM(v)`), by alias (`total`), or by group-key name. It applies
        // over the (possibly UNION-ed) result, whose schema is the SELECT list.
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

        // Resolve the named WINDOW clause (`WINDOW w AS (...)`) so an `OVER w`
        // reference in any SELECT item — or in QUALIFY — lowers to the named
        // spec. Empty when there is no WINDOW clause.
        let named_windows = build_named_window_map(&select.named_window)?;

        // `add_window` registers a parsed window function into the (deduped)
        // group set and returns the synthesized output column name. Shared by
        // the SELECT-item pass and the QUALIFY pass so a window function used
        // only in QUALIFY materializes a hidden column the same way.
        let mut add_window =
            |pw: ParsedWindow, window_groups: &mut Vec<WindowGroup>, next_id: &mut usize| {
                let out_name = format!("__window_{next_id}");
                *next_id += 1;
                let we = WindowExpr {
                    func: pw.func,
                    output_name: out_name.clone(),
                };
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
                out_name
            };

        // First, lower each SELECT item to a projection expr. For window
        // items the expr becomes `Column("__window_N")` referencing the
        // appended window column.
        //
        // `window_aliases` records `SELECT-alias -> __window_N` so a QUALIFY
        // predicate may reference a window function by the alias it was given in
        // the SELECT list (`... AS rn ... QUALIFY rn = 1`). The alias does not
        // exist at the Window node's output (it is applied only by the final
        // Project, which sits *above* the QUALIFY Filter), so the QUALIFY lowerer
        // rewrites such references back to the underlying window column.
        let mut proj_exprs: Vec<Expr> = Vec::with_capacity(items.len());
        let mut window_aliases: HashMap<String, String> = HashMap::new();
        for (sql_expr, alias) in &items {
            if let Some(pw) = try_window(sql_expr, &resolver, &named_windows, 0)? {
                let out_name = add_window(pw, &mut window_groups, &mut next_window_id);
                if let Some(name) = alias {
                    window_aliases.insert(name.clone(), out_name.clone());
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

        // QUALIFY: filters rows on a window-function result, exactly as HAVING
        // filters on an aggregate. We lower it as a `Filter` applied *after* the
        // Window nodes (so the window columns exist) but *before* the final
        // Project, then project away any helper columns the predicate
        // materialized that are not part of the SELECT list.
        //
        // The predicate may reference a window function (`QUALIFY ROW_NUMBER()
        // OVER (...) = 1`) that is not in the SELECT list; such a call is
        // materialized as a hidden `__window_N` column (via `add_window`) which
        // the filter references and the trailing Project drops. A window call
        // already present in the SELECT list reuses its column through the same
        // spec-dedup in `add_window`.
        //
        // The number of SELECT-list output columns: the trailing Project must
        // restore exactly these (dropping any hidden window helpers) so the
        // output schema is unaffected by QUALIFY.
        let qualify_predicate = match &select.qualify {
            Some(q) => Some(lower_qualify_predicate(
                q,
                &resolver,
                &named_windows,
                &window_aliases,
                &mut |pw| add_window(pw, &mut window_groups, &mut next_window_id),
            )?),
            None => None,
        };

        // Stack the Window nodes (if any) over the current plan.
        for g in window_groups {
            plan = LogicalPlan::Window {
                input: Box::new(plan),
                window_exprs: g.exprs,
                partition_by: g.partition_by,
                order_by: g.order_by,
            };
        }

        // Apply QUALIFY as a Filter over the window projection (before the
        // SELECT-list Project, so the hidden window columns are still in scope).
        if let Some(predicate) = qualify_predicate {
            let input_schema = plan.schema()?;
            let predicate_dtype = predicate.dtype(&input_schema)?;
            if predicate_dtype != DataType::Bool {
                return Err(BoltError::Type(format!(
                    "QUALIFY predicate must be Bool, got {predicate_dtype:?}"
                )));
            }
            plan = LogicalPlan::Filter {
                input: Box::new(plan),
                predicate,
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
    depth: usize,
) -> BoltResult<(LogicalPlan, String, Schema)> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
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
                // A table-valued function (`FROM generate_series(1, 10)`,
                // `FROM unnest(arr)`, …) produces its rows from a function call
                // rather than a registered table. Supporting it would require a
                // function-as-table-source mechanism: a registry of set-returning
                // functions plus a logical "function scan" operator that
                // evaluates the call and exposes its output schema. The engine
                // scans only registered base tables / CTEs / derived tables, so
                // this is scoped out rather than missing by oversight.
                return Err(BoltError::Sql(
                    "unsupported: table-valued function in FROM (would require a \
                     function-as-table-source mechanism; only registered tables, \
                     CTEs, and derived tables are scannable)"
                        .into(),
                ));
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
        // F12: derived table `(SELECT ...) AS alias`. We recursively plan the
        // subquery as a child `LogicalPlan` and expose its output schema under
        // the alias — reusing the same pipeline a CTE reference uses (a CTE is
        // just a named, pre-lowered derived table). No new exec machinery is
        // needed: the produced plan is an ordinary self-contained subtree.
        //
        // Restrictions:
        //   * LATERAL derived tables are correlated (they may reference earlier
        //     FROM items) and the engine has no correlated-execution path, so we
        //     reject them with a precise message.
        //   * An alias is required (standard SQL — a derived table must be
        //     named) so qualified `alias.col` references can resolve.
        //   * Column-list aliases `AS t(c1, c2)` would require renaming the
        //     subquery's output fields, which we do not implement (mirrors the
        //     base-table arm above).
        TableFactor::Derived {
            lateral,
            subquery,
            alias,
        } => {
            // LATERAL (feature F3): the supported shapes
            // (`FROM <left>, LATERAL (subq) AS a`, plus CROSS / [INNER|LEFT]
            // JOIN LATERAL (...) ON true) are detected and orchestrated host-side
            // as a nested-loop Apply *before* this ordinary lowering path runs —
            // see [`plan_lateral_apply`] and `Engine::execute_lateral_apply`,
            // wired into `Engine::sql` / `Engine::explain_sql` exactly like the
            // WITH RECURSIVE hook. So a LATERAL reaching HERE is one the apply
            // detector deliberately declined, namely:
            //
            //   * the raw `parse()` API (which bypasses the engine detector
            //     entirely — like WITH RECURSIVE / COUNT(DISTINCT) GROUP BY, this
            //     feature is only wired into `Engine::sql` / `Engine::explain_sql`);
            //   * an out-of-scope LATERAL shape the detector returned `Err` for
            //     (more than one LATERAL, LATERAL not last, a non-`ON true` JOIN
            //     LATERAL predicate, …) — those surface the detector's precise
            //     message and never reach this arm.
            //
            // Either way this arm cannot run a correlated subquery as a plain
            // uncorrelated subplan, so it stays a precise rejection that points
            // at the engine path.
            if *lateral {
                return Err(BoltError::Sql(
                    "unsupported: LATERAL derived table on this code path — a \
                     correlated LATERAL apply is executed host-side by the engine \
                     (Engine::sql / Engine::explain_sql), not by the raw parse() \
                     API; supported shapes are `FROM <left>, LATERAL (subq) AS a` \
                     and `[INNER|LEFT] JOIN LATERAL (subq) AS a ON true`"
                        .into(),
                ));
            }
            let alias = alias.as_ref().ok_or_else(|| {
                BoltError::Sql(
                    "subquery in FROM (derived table) requires an alias, e.g. \
                     `(SELECT ...) AS t`"
                        .into(),
                )
            })?;
            if !alias.columns.is_empty() {
                return Err(BoltError::Sql(
                    "unsupported: derived-table alias with column list (AS t(c1, c2))".into(),
                ));
            }
            let qualifier = alias.name.value.clone();
            // Recursively plan the subquery. The MAX_RECURSION_DEPTH guard on
            // this function (checked at entry) and on `plan_query` bounds the
            // nesting depth of stacked derived tables.
            let sub_plan = plan_query(subquery, provider, ctes, depth + 1)?;
            let schema = sub_plan.schema()?;
            Ok((sub_plan, qualifier, schema))
        }
        TableFactor::NestedJoin { .. } => Err(BoltError::Sql(
            "unsupported: parenthesised / nested join in FROM".into(),
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
/// join it lives on. We accept a bare identifier, a `table.column` /
/// `alias.column` qualified identifier, or the schema-qualified
/// `schema.table.column` form (the leading single-catalog segment is dropped)
/// so users can disambiguate same-named columns; all lower to a plain `Column`
/// ref. Four-or-more-segment references have no namespace to collapse and are
/// rejected as "deeply qualified".
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
            if parts.len() > 3 {
                let full = parts
                    .iter()
                    .map(|p| p.value.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                return Err(BoltError::Sql(format!(
                    "unsupported: deeply qualified column reference '{full}' in JOIN ON"
                )));
            }
            // SQL-standard case folding (v0.5): qualifier and column are
            // folded independently. The qualifier match against the
            // resolver is ASCII case-insensitive so a folded `t1` still
            // matches a table the host registered as `T1`.
            //
            // F12: accept the schema-qualified `schema.table.col` spelling too.
            // The engine is single-catalog, so a leading schema/catalog segment
            // carries no meaning — we drop it and resolve by the trailing
            // `table.col` pair exactly as the two-part form does (mirrors the
            // 3-segment handling in `lower_expr`'s `CompoundIdentifier` arm). An
            // unknown table/alias in the middle slot still produces the standard
            // "unknown table" error below.
            let (qual_ident, col_ident) = if parts.len() == 3 {
                (&parts[1], &parts[2])
            } else {
                (&parts[0], &parts[1])
            };
            let qualifier = ident_to_name(qual_ident);
            let col = ident_to_name(col_ident);
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
    // T-SQL `SELECT TOP n` is a row limit, lowered to `LogicalPlan::Limit` by
    // [`plan_query`] *above* any top-level ORDER BY (T-SQL applies TOP after the
    // sort). It is therefore not rejected here. `PERCENT` / `WITH TIES` and a
    // TOP combined with LIMIT/FETCH are rejected precisely in `top_limit_value`.
    if select.into.is_some() {
        return Err(BoltError::Sql("unsupported: SELECT INTO".into()));
    }
    if !select.lateral_views.is_empty() {
        return Err(BoltError::Sql("unsupported: LATERAL VIEW".into()));
    }
    // PREWHERE (ClickHouse early-filter) is semantically equivalent to WHERE —
    // it only changes *when* the predicate is applied, not the result set — so
    // it is folded into the WHERE predicate by [`plan_select`] (PREWHERE AND
    // WHERE) rather than rejected here.
    if !select.cluster_by.is_empty() {
        return Err(BoltError::Sql("unsupported: CLUSTER BY".into()));
    }
    if !select.distribute_by.is_empty() {
        return Err(BoltError::Sql("unsupported: DISTRIBUTE BY".into()));
    }
    if !select.sort_by.is_empty() {
        return Err(BoltError::Sql("unsupported: SORT BY".into()));
    }
    // Named WINDOW clause (`... WINDOW w AS (PARTITION BY ...)`) is resolved at
    // lowering time: an `OVER w` reference is replaced by the named spec and
    // then lowered exactly like an inline `OVER (...)`. So it is not rejected
    // here; see [`build_named_window_map`] and `try_window`.
    //
    // QUALIFY (`... QUALIFY <window-fn predicate>`) is to window functions what
    // HAVING is to aggregates — lowered as a `Filter` over the window
    // projection by [`plan_select`]. Not rejected here.
    if select.value_table_mode.is_some() {
        return Err(BoltError::Sql("unsupported: SELECT AS STRUCT/VALUE".into()));
    }
    if select.connect_by.is_some() {
        // Oracle hierarchical queries (`START WITH ... CONNECT BY PRIOR ...`)
        // are an iterative tree-walk: they would require the same fixpoint /
        // recursive-CTE execution machinery as `WITH RECURSIVE` (seed the roots,
        // then repeatedly join children onto the frontier until it stops
        // growing), plus Oracle pseudo-columns (LEVEL, CONNECT_BY_ROOT). That
        // recursive execution path is not wired up for the CONNECT BY surface —
        // express the hierarchy as a `WITH RECURSIVE` CTE instead.
        return Err(BoltError::Sql(
            "unsupported: CONNECT BY hierarchical queries (would require \
             recursive-CTE-style fixpoint execution; rewrite as WITH RECURSIVE)"
                .into(),
        ));
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

/// Resolved named-window definitions for a single SELECT: maps a window name
/// (as written in `WINDOW w AS (...)`) to its concrete [`WindowSpec`]. An
/// `OVER w` reference is resolved against this map by [`resolve_named_window`].
/// Empty when the SELECT has no WINDOW clause (the common case).
type NamedWindowMap<'a> = HashMap<String, &'a WindowSpec>;

/// Build the [`NamedWindowMap`] for a SELECT's `WINDOW w AS (...)` clause.
///
/// Each definition is either a direct spec (`WINDOW w AS (PARTITION BY ...)`)
/// or a chained reference to another named window (`WINDOW w2 AS w1`, BigQuery).
/// Chained references are resolved transitively to the underlying spec; a cycle
/// or a dangling reference is rejected precisely. Definitions are processed in
/// source order, so a chain may only reference an *earlier* definition (matching
/// the parse-order scoping every supporting dialect uses).
fn build_named_window_map(
    defs: &[sqlparser::ast::NamedWindowDefinition],
) -> BoltResult<NamedWindowMap<'_>> {
    let mut map: NamedWindowMap = HashMap::with_capacity(defs.len());
    for def in defs {
        let name = ident_to_name(&def.0);
        let spec: &WindowSpec = match &def.1 {
            NamedWindowExpr::WindowSpec(s) => s,
            NamedWindowExpr::NamedWindow(other) => {
                // Chain reference: must resolve to an already-defined window.
                let other_name = ident_to_name(other);
                *map.get(&other_name).ok_or_else(|| {
                    BoltError::Sql(format!(
                        "WINDOW '{name}' references window '{other_name}', which is \
                         not defined earlier in the WINDOW clause"
                    ))
                })?
            }
        };
        if map.insert(name.clone(), spec).is_some() {
            return Err(BoltError::Sql(format!(
                "duplicate WINDOW definition for '{name}'"
            )));
        }
    }
    Ok(map)
}

/// Resolve an `OVER w` reference (and the `OVER (w ORDER BY ...)` extension
/// form) into a concrete owned [`WindowSpec`].
///
/// * `name` is the referenced window name.
/// * `inline` is `Some(spec)` for the extension form `OVER (w ...)` — an inline
///   spec whose `window_name` is `w`. The SQL rule is that the referenced
///   window supplies PARTITION BY (and ORDER BY); the inline part may *add* an
///   ORDER BY only when the base supplied none, and may not re-state PARTITION
///   BY or a frame. We support exactly that simple extension and reject anything
///   more involved precisely. `None` is the bare `OVER w` form.
fn resolve_named_window(
    name: &Ident,
    inline: Option<&WindowSpec>,
    named_windows: &NamedWindowMap,
) -> BoltResult<WindowSpec> {
    let key = ident_to_name(name);
    let base = named_windows.get(&key).ok_or_else(|| {
        BoltError::Sql(format!(
            "window '{key}' referenced by OVER is not defined in a WINDOW clause"
        ))
    })?;
    // A base window must itself be a plain spec, not a further name reference
    // (chains are already flattened by `build_named_window_map`).
    let mut resolved = (*base).clone();
    if let Some(inline) = inline {
        // Extension form `OVER (w ...)`. Only the simple ORDER-BY-extension is
        // supported.
        if !inline.partition_by.is_empty() {
            return Err(BoltError::Sql(format!(
                "OVER ({key} ...) may not re-state PARTITION BY (it is inherited \
                 from the named window)"
            )));
        }
        if inline.window_frame.is_some() {
            return Err(BoltError::Sql(format!(
                "unsupported: OVER ({key} ...) adding an explicit window frame"
            )));
        }
        if !inline.order_by.is_empty() {
            if !resolved.order_by.is_empty() {
                return Err(BoltError::Sql(format!(
                    "OVER ({key} ORDER BY ...) conflicts: the named window '{key}' \
                     already defines an ORDER BY"
                )));
            }
            resolved.order_by = inline.order_by.clone();
        }
    }
    // Clear the name so downstream lowering sees a fully-inlined spec.
    resolved.window_name = None;
    Ok(resolved)
}

/// Lower a QUALIFY predicate into an [`Expr`] over the window projection.
///
/// QUALIFY filters rows on window-function results (it is to window functions
/// what HAVING is to aggregates). Any window-function call in the predicate is
/// materialized as a `__window_N` column via `add_window` (the SELECT-item pass
/// already registered the SELECT-list windows under the same spec-dedup, so a
/// window call shared with the SELECT list reuses the existing column); the
/// call subtree is then replaced by a reference to that column. Non-window
/// parts of the predicate lower through the ordinary [`lower_expr`] path against
/// the FROM-tree namespace.
///
/// The rewrite walks the common boolean / comparison composite shapes; a
/// window function buried in an unsupported position surfaces through the
/// normal lowering (an `OVER` left in the tree would reach `lower_expr`, which
/// does not lower window calls, and error cleanly).
fn lower_qualify_predicate(
    e: &SqlExpr,
    resolver: &NameResolver,
    named_windows: &NamedWindowMap,
    window_aliases: &HashMap<String, String>,
    add_window: &mut dyn FnMut(ParsedWindow) -> String,
) -> BoltResult<Expr> {
    // Build a copy of the predicate with each window-function subtree replaced
    // by a synthetic quoted identifier naming its materialized column, then
    // lower the rewritten tree. `lower_expr` maps a bare identifier straight to
    // `Expr::Column(name)`, so the `__window_N` reference resolves against the
    // post-Window schema during type-checking.
    let rewritten =
        rewrite_qualify_windows(e, resolver, named_windows, window_aliases, add_window, 0)?;
    lower_expr(&rewritten, resolver, 0)
}

/// Recursively replace every window-function call in `e` with a synthetic
/// identifier naming the `__window_N` column produced by `add_window`. A bare
/// identifier that matches a SELECT-list window alias is likewise rewritten to
/// the underlying window column (the alias does not exist below the final
/// Project). Returns an owned, rewritten `SqlExpr` ready for [`lower_expr`].
fn rewrite_qualify_windows(
    e: &SqlExpr,
    resolver: &NameResolver,
    named_windows: &NamedWindowMap,
    window_aliases: &HashMap<String, String>,
    add_window: &mut dyn FnMut(ParsedWindow) -> String,
    depth: usize,
) -> BoltResult<SqlExpr> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    // A window-function call at this node: materialize it and substitute a
    // reference to the generated column.
    if let Some(pw) = try_window(e, resolver, named_windows, depth + 1)? {
        let out_name = add_window(pw);
        return Ok(SqlExpr::Identifier(Ident::with_quote('"', out_name)));
    }
    // A bare identifier naming a SELECT-list window alias: rewrite to the
    // underlying `__window_N` column so the filter sees the value the Window
    // node produced (the alias is applied only by the Project above the filter).
    if let SqlExpr::Identifier(id) = e {
        if let Some(col) = window_aliases.get(&ident_to_name(id)) {
            return Ok(SqlExpr::Identifier(Ident::with_quote('"', col.clone())));
        }
    }
    // Otherwise recurse through the boolean / comparison composite shapes a
    // QUALIFY predicate is built from. Leaf nodes (identifiers, literals,
    // non-window function calls) are returned verbatim for `lower_expr`.
    let rec = |sub: &SqlExpr,
               add_window: &mut dyn FnMut(ParsedWindow) -> String|
     -> BoltResult<Box<SqlExpr>> {
        Ok(Box::new(rewrite_qualify_windows(
            sub,
            resolver,
            named_windows,
            window_aliases,
            add_window,
            depth + 1,
        )?))
    };
    let out = match e {
        SqlExpr::BinaryOp { left, op, right } => SqlExpr::BinaryOp {
            left: rec(left, add_window)?,
            op: op.clone(),
            right: rec(right, add_window)?,
        },
        SqlExpr::UnaryOp { op, expr } => SqlExpr::UnaryOp {
            op: *op,
            expr: rec(expr, add_window)?,
        },
        SqlExpr::Nested(inner) => SqlExpr::Nested(rec(inner, add_window)?),
        SqlExpr::IsNull(inner) => SqlExpr::IsNull(rec(inner, add_window)?),
        SqlExpr::IsNotNull(inner) => SqlExpr::IsNotNull(rec(inner, add_window)?),
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => SqlExpr::Between {
            expr: rec(expr, add_window)?,
            negated: *negated,
            low: rec(low, add_window)?,
            high: rec(high, add_window)?,
        },
        // Any other shape is returned unchanged. If it nonetheless contains a
        // window call in a position we don't rewrite, the leftover `OVER` reaches
        // `lower_expr`, which rejects window calls cleanly.
        other => other.clone(),
    };
    Ok(out)
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
    named_windows: &NamedWindowMap,
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

    // Parse / resolve the OVER (...) spec into a concrete `WindowSpec`.
    //
    //   * `OVER (PARTITION BY ... ORDER BY ...)`  — an inline spec, used as-is.
    //   * `OVER w`                                — a bare named-window
    //     reference; resolved to the spec defined by `WINDOW w AS (...)`.
    //   * `OVER (w ORDER BY ...)`                 — an inline spec that *extends*
    //     a named window `w`; handled for the simple case where the inline part
    //     only adds an ORDER BY (and the named window supplied no ORDER BY), and
    //     rejected precisely otherwise.
    //
    // `resolve_named_window` returns an owned `WindowSpec` so both the bare
    // reference and the inline cases flow through the identical
    // partition/order/frame lowering below.
    let resolved;
    let spec: &WindowSpec = match over {
        WindowType::WindowSpec(s) => {
            if let Some(base_name) = &s.window_name {
                resolved = resolve_named_window(base_name, Some(s), named_windows)?;
                &resolved
            } else {
                s
            }
        }
        WindowType::NamedWindow(name) => {
            resolved = resolve_named_window(name, None, named_windows)?;
            &resolved
        }
    };
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
        // CHAR_LENGTH / CHARACTER_LENGTH are SQL-standard synonyms for the
        // character-based LENGTH; they lower to the same kind.
        "CHAR_LENGTH" | "CHARACTER_LENGTH" => ScalarFnKind::Length,
        "OCTET_LENGTH" => ScalarFnKind::OctetLength,
        "CONCAT" => ScalarFnKind::Concat,
        // STRPOS(s, substr) is the function-call spelling of POSITION; the
        // dedicated `POSITION(substr IN s)` syntax is handled in `lower_expr`
        // (sqlparser parses it as `SqlExpr::Position`, not a `Function`).
        "STRPOS" => ScalarFnKind::Position,
        "REPLACE" => ScalarFnKind::Replace,
        "LEFT" => ScalarFnKind::Left,
        "RIGHT" => ScalarFnKind::Right,
        "LPAD" => ScalarFnKind::Lpad,
        "RPAD" => ScalarFnKind::Rpad,
        "REVERSE" => ScalarFnKind::Reverse,
        "INITCAP" => ScalarFnKind::Initcap,
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
        SqlExpr::Like { expr, .. } | SqlExpr::ILike { expr, .. } => {
            contains_aggregate(expr, resolver, depth + 1)
        }
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
            // `CAST(... FORMAT '<pattern>')` is a host-only temporal ⇄ string
            // conversion (feature CAST FORMAT). It lowers to `Expr::CastFormat`
            // regardless of HAVING context; the inner is lowered with the
            // aggregate-aware lowerer so `HAVING CAST(ts AS VARCHAR FORMAT ...)`
            // still resolves any aggregate aliases beneath it.
            if let Some(fmt) = format {
                let inner = lower_expr_in_having(expr, resolver, agg_aliases, depth + 1)?;
                return lower_cast_format(kind, inner, data_type, fmt);
            }
            // `TRY_CAST` / `SAFE_CAST` are synonyms carrying NULL-on-failure
            // semantics (`safe = true`); plain `CAST` / `::` keep the
            // error-on-failure behaviour (`safe = false`).
            let safe = match kind {
                CastKind::Cast | CastKind::DoubleColon => false,
                CastKind::TryCast | CastKind::SafeCast => true,
            };
            let target = lower_cast_data_type(data_type)?;
            let inner = lower_expr_in_having(expr, resolver, agg_aliases, depth + 1)?;
            Ok(Expr::Cast {
                expr: Box::new(inner),
                target,
                safe,
            })
        }
        // Anything else is identical to a scalar HAVING fragment; defer to
        // the normal lowerer (which handles Identifier, Value, etc., and
        // still rejects bare non-aggregate Function calls).
        _ => lower_expr(e, resolver, depth + 1),
    }
}

/// Lower a HAVING predicate for the no-GROUP-BY, sole-`COUNT(DISTINCT col)`
/// path (see the bare COUNT(DISTINCT) block in [`plan_select`]).
///
/// With no GROUP BY the whole input is one implicit group, so HAVING simply
/// filters the single aggregate result row. The aggregate has already been
/// materialised into the Project's output column named `count_out_name`; any
/// `COUNT(DISTINCT <col>)` call in the HAVING predicate whose argument matches
/// the SELECT's distinct-counted column is rewritten to `Column(count_out_name)`.
/// Everything else is lowered structurally so the predicate references only
/// that output column (and any literals).
///
/// Supported predicate shapes: binary operators (comparisons / arithmetic /
/// AND / OR), unary `NOT` / unary `+` / unary `-`, parenthesised groups, and
/// `IS [NOT] NULL`. Any other shape — in particular a *different* aggregate,
/// or a raw column reference that is not the distinct-counted column — is
/// rejected with a precise message rather than silently mis-lowered.
///
/// `select_inner` is the SELECT's (un-lowered) `COUNT(DISTINCT <select_inner>)`
/// argument; a HAVING `COUNT(DISTINCT x)` only matches when `x` lowers to the
/// same expression (`expr_eq`).
fn lower_having_over_count_distinct(
    e: &SqlExpr,
    select_inner: &SqlExpr,
    resolver: &NameResolver<'_>,
    count_out_name: &str,
    depth: usize,
) -> BoltResult<Expr> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(BoltError::Sql(format!(
            "expression nesting exceeds depth limit ({MAX_RECURSION_DEPTH})"
        )));
    }
    // Does this subexpression name the distinct-counted aggregate? If so it
    // resolves to the already-projected count column.
    if let Some(inner) = try_count_distinct(e, resolver)? {
        let inner_lowered = lower_expr(inner, resolver, depth + 1)?;
        let select_lowered = lower_expr(select_inner, resolver, depth + 1)?;
        if expr_eq(&inner_lowered, &select_lowered) {
            return Ok(Expr::Column(count_out_name.to_string()));
        }
        return Err(BoltError::Sql(
            "HAVING references a COUNT(DISTINCT ...) over a different column than \
             the one in the SELECT list; only the projected distinct-count may \
             be filtered"
                .into(),
        ));
    }
    match e {
        SqlExpr::Nested(inner) => {
            lower_having_over_count_distinct(inner, select_inner, resolver, count_out_name, depth + 1)
        }
        SqlExpr::BinaryOp { left, op, right } => {
            let lop = lower_binary_op(op)?;
            let l = lower_having_over_count_distinct(
                left,
                select_inner,
                resolver,
                count_out_name,
                depth + 1,
            )?;
            let r = lower_having_over_count_distinct(
                right,
                select_inner,
                resolver,
                count_out_name,
                depth + 1,
            )?;
            Ok(Expr::Binary {
                op: lop,
                left: Box::new(l),
                right: Box::new(r),
            })
        }
        SqlExpr::UnaryOp { op, expr } => match op {
            UnaryOperator::Plus => lower_having_over_count_distinct(
                expr,
                select_inner,
                resolver,
                count_out_name,
                depth + 1,
            ),
            UnaryOperator::Minus => {
                let inner = lower_having_over_count_distinct(
                    expr,
                    select_inner,
                    resolver,
                    count_out_name,
                    depth + 1,
                )?;
                Ok(Expr::Binary {
                    op: BinaryOp::Sub,
                    left: Box::new(Expr::Literal(Literal::Int64(0))),
                    right: Box::new(inner),
                })
            }
            UnaryOperator::Not => {
                let inner = lower_having_over_count_distinct(
                    expr,
                    select_inner,
                    resolver,
                    count_out_name,
                    depth + 1,
                )?;
                Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(inner),
                })
            }
            other => Err(BoltError::Sql(format!(
                "unsupported unary operator in HAVING: {other:?}"
            ))),
        },
        SqlExpr::IsNull(inner) => {
            let operand = lower_having_over_count_distinct(
                inner,
                select_inner,
                resolver,
                count_out_name,
                depth + 1,
            )?;
            Ok(Expr::Unary {
                op: UnaryOp::IsNull,
                operand: Box::new(operand),
            })
        }
        SqlExpr::IsNotNull(inner) => {
            let operand = lower_having_over_count_distinct(
                inner,
                select_inner,
                resolver,
                count_out_name,
                depth + 1,
            )?;
            Ok(Expr::Unary {
                op: UnaryOp::IsNotNull,
                operand: Box::new(operand),
            })
        }
        // Literals (and any other purely-scalar fragment) lower normally. A
        // bare column reference or aggregate call that is *not* the projected
        // distinct-count would be lowered by `lower_expr`; for a column that
        // is not in the (single-column) Project output, `validate_having_columns`
        // rejects it afterwards with a friendly message.
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
            Expr::CastFormat { expr, .. } => stack.push(expr),
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
            "subqueries are not supported in this position (supported: WHERE, SELECT, \
             GROUP BY, HAVING, JOIN ON, ORDER BY)"
                .into(),
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
                case_insensitive: false,
            })
        }
        // `ILIKE` is the case-insensitive form of `LIKE`. sqlparser surfaces
        // it as a distinct AST node with identical fields; we mirror the
        // `Like` lowering above verbatim and only flip `case_insensitive`.
        SqlExpr::ILike {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => {
            if *any {
                return Err(BoltError::Sql(
                    "unsupported: ILIKE ANY (...)".into(),
                ));
            }
            let escape: Option<char> = match escape_char.as_deref() {
                None => None,
                Some(s) => {
                    let mut iter = s.chars();
                    let first = iter.next().ok_or_else(|| {
                        BoltError::Sql(
                            "ILIKE ESCAPE clause must be a single character, got an empty string"
                                .into(),
                        )
                    })?;
                    if iter.next().is_some() {
                        return Err(BoltError::Sql(format!(
                            "ILIKE ESCAPE clause must be a single character, got {s:?}"
                        )));
                    }
                    Some(first)
                }
            };
            let pattern_str = match pattern.as_ref() {
                SqlExpr::Value(Value::SingleQuotedString(s)) => s.clone(),
                other => {
                    return Err(BoltError::Sql(format!(
                        "ILIKE pattern must be a string literal constant, got: {other}"
                    )));
                }
            };
            let operand = lower_expr(expr, resolver, depth + 1)?;
            Ok(Expr::Like {
                expr: Box::new(operand),
                pattern: pattern_str,
                escape,
                negated: *negated,
                case_insensitive: true,
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
        // `SAFE_CAST` (synonyms) carry NULL-on-failure semantics: a
        // conversion failure on a non-null input yields SQL NULL instead of
        // an error. They lower to the same `Expr::Cast` shape with
        // `safe = true`. `CAST(... FORMAT '<pattern>')` is the bounded,
        // host-only temporal ⇄ string conversion (feature CAST FORMAT) and
        // lowers to `Expr::CastFormat` via `lower_cast_format`.
        SqlExpr::Cast {
            kind,
            expr,
            data_type,
            format,
        } => {
            if let Some(fmt) = format {
                let inner = lower_expr(expr, resolver, depth + 1)?;
                return lower_cast_format(kind, inner, data_type, fmt);
            }
            let safe = match kind {
                CastKind::Cast | CastKind::DoubleColon => false,
                CastKind::TryCast | CastKind::SafeCast => true,
            };
            let target = lower_cast_data_type(data_type)?;
            let inner = lower_expr(expr, resolver, depth + 1)?;
            Ok(Expr::Cast {
                expr: Box::new(inner),
                target,
                safe,
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
        // `POSITION(substr IN s)`: sqlparser surfaces this as the dedicated
        // `SqlExpr::Position { expr, r#in }` variant (`expr` is the substring,
        // `r#in` is the haystack). The `STRPOS(s, substr)` function spelling is
        // handled in `try_string_scalar_fn`. We normalise both into the
        // `ScalarFnKind::Position` argument order `[s, substr]`, so the haystack
        // is always arg 0 and the needle arg 1. Type-checking (`Utf8`, `Utf8` ->
        // `Int64`) lives in `scalar_fn_dtype`.
        SqlExpr::Position { expr, r#in } => {
            let substr = lower_expr(expr, resolver, depth + 1)?;
            let s = lower_expr(r#in, resolver, depth + 1)?;
            Ok(Expr::ScalarFn {
                kind: ScalarFnKind::Position,
                args: vec![s, substr],
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
/// Lower a `CAST(<inner> AS <data_type> FORMAT '<pattern>')` into an
/// [`Expr::CastFormat`] (feature CAST FORMAT).
///
/// This is a **host-only** conversion (no GPU codegen): the physical-plan
/// boundary routes any projection carrying a `CastFormat` to the host
/// `PhysicalPlan::Project`. Two directions are recognised, keyed off the
/// declared target type:
///   * target is a string type (`VARCHAR` / `TEXT` / `CHAR` / `STRING`):
///     format a `Date32` / `Timestamp` operand *to* text (`to_text = true`).
///   * target is `DATE` / `TIMESTAMP`: parse a `Utf8` operand *into* a
///     temporal value (`to_text = false`).
///
/// `TRY_CAST` / `SAFE_CAST` spellings are rejected with FORMAT (the
/// NULL-on-failure envelope is not defined for the format path); only plain
/// `CAST` / `::` carry FORMAT. `AT TIME ZONE` inside the FORMAT clause
/// (`CastFormat::ValueAtTimeZone`) is rejected. The pattern string is parsed
/// and validated by [`parse_cast_format_pattern`]; unknown tokens are rejected
/// there with a message naming the offending token.
fn lower_cast_format(
    kind: &CastKind,
    inner: Expr,
    data_type: &SqlDataType,
    fmt: &SqlCastFormat,
) -> BoltResult<Expr> {
    // Only plain CAST carries a FORMAT clause; TRY_CAST/SAFE_CAST + FORMAT has
    // no defined NULL-on-failure contract here, so reject it precisely.
    match kind {
        CastKind::Cast | CastKind::DoubleColon => {}
        CastKind::TryCast | CastKind::SafeCast => {
            return Err(BoltError::Sql(
                "TRY_CAST / SAFE_CAST with a FORMAT clause is not supported \
                 (FORMAT is only defined for plain CAST)"
                    .into(),
            ));
        }
    }
    // Extract the pattern string. `AT TIME ZONE` is not modelled.
    let pattern_str = match fmt {
        SqlCastFormat::Value(Value::SingleQuotedString(s))
        | SqlCastFormat::Value(Value::DoubleQuotedString(s)) => s.clone(),
        SqlCastFormat::Value(other) => {
            return Err(BoltError::Sql(format!(
                "CAST(... FORMAT ...) pattern must be a string literal, got {other}"
            )));
        }
        SqlCastFormat::ValueAtTimeZone(_, _) => {
            return Err(BoltError::Sql(
                "CAST(... FORMAT ... AT TIME ZONE ...) is not supported".into(),
            ));
        }
    };
    let pattern = parse_cast_format_pattern(&pattern_str)?;

    // Resolve the target dtype and the conversion direction from the declared
    // SQL type. String targets ⇒ format (to_text); temporal targets ⇒ parse.
    let (target, to_text) = match data_type {
        SqlDataType::Varchar(_)
        | SqlDataType::Nvarchar(_)
        | SqlDataType::Char(_)
        | SqlDataType::CharVarying(_)
        | SqlDataType::CharacterVarying(_)
        | SqlDataType::Character(_)
        | SqlDataType::Text
        | SqlDataType::String(_) => (DataType::Utf8, true),
        SqlDataType::Date => (DataType::Date32, false),
        // Timestamp FORMAT parses into a naive nanosecond timestamp (matching
        // the `TIMESTAMP '...'` literal path); timezone info on the type is
        // not modelled here.
        SqlDataType::Timestamp(_, _) => {
            (DataType::Timestamp(TimeUnit::Nanosecond, None), false)
        }
        other => {
            return Err(BoltError::Sql(format!(
                "CAST(... FORMAT ...) target type must be VARCHAR/TEXT (format) or \
                 DATE/TIMESTAMP (parse), got {other}"
            )));
        }
    };

    Ok(Expr::CastFormat {
        expr: Box::new(inner),
        target,
        pattern,
        to_text,
    })
}

/// Parse a `CAST(... FORMAT '<pattern>')` pattern string into a bounded,
/// validated [`FormatToken`] sequence (feature CAST FORMAT).
///
/// Supported vocabulary (case-insensitive for the field tokens):
///   * `YYYY` → 4-digit year
///   * `MM`   → 2-digit month
///   * `DD`   → 2-digit day
///   * `HH24` / `HH` → 2-digit 24-hour hour
///   * `MI`   → 2-digit minute
///   * `SS`   → 2-digit second
///   * literal separators: `-`, `/`, `:`, and space
///
/// The scan is greedy and longest-match (`HH24` before `HH`, `YYYY` before a
/// shorter year). Any unrecognised character or alphabetic run is rejected
/// with a `BoltError::Sql` that names the offending token, rather than being
/// silently mis-formatted.
fn parse_cast_format_pattern(pat: &str) -> BoltResult<Vec<FormatToken>> {
    if pat.is_empty() {
        return Err(BoltError::Sql(
            "CAST(... FORMAT ...) pattern must not be empty".into(),
        ));
    }
    let bytes = pat.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    // ASCII-only matching: every supported token and separator is ASCII, and a
    // non-ASCII byte can never be a continuation of one, so byte indexing is
    // safe here.
    let upper = pat.to_ascii_uppercase();
    let ub = upper.as_bytes();
    while i < bytes.len() {
        // Longest-match field tokens first.
        let matched = if ub[i..].starts_with(b"YYYY") {
            out.push(FormatToken::Year4);
            4
        } else if ub[i..].starts_with(b"HH24") {
            out.push(FormatToken::Hour24);
            4
        } else if ub[i..].starts_with(b"HH") {
            out.push(FormatToken::Hour24);
            2
        } else if ub[i..].starts_with(b"MM") {
            out.push(FormatToken::Month);
            2
        } else if ub[i..].starts_with(b"DD") {
            out.push(FormatToken::Day);
            2
        } else if ub[i..].starts_with(b"MI") {
            out.push(FormatToken::Minute);
            2
        } else if ub[i..].starts_with(b"SS") {
            out.push(FormatToken::Second);
            2
        } else {
            let c = bytes[i] as char;
            match c {
                '-' | '/' | ':' | ' ' => {
                    out.push(FormatToken::Literal(c));
                    1
                }
                // An alphabetic run that did not match a known token: surface
                // the whole run so the error names the offending token rather
                // than a single character.
                _ if c.is_ascii_alphabetic() => {
                    let start = i;
                    let mut j = i;
                    while j < bytes.len() && (bytes[j] as char).is_ascii_alphabetic() {
                        j += 1;
                    }
                    return Err(BoltError::Sql(format!(
                        "CAST(... FORMAT ...): unsupported pattern token '{}' \
                         (supported: YYYY, MM, DD, HH/HH24, MI, SS and the \
                         separators - / : space)",
                        &pat[start..j]
                    )));
                }
                _ => {
                    return Err(BoltError::Sql(format!(
                        "CAST(... FORMAT ...): unsupported separator '{c}' \
                         (only - / : and space are allowed)"
                    )));
                }
            }
        };
        i += matched;
    }
    Ok(out)
}

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

#[cfg(test)]
mod cast_format_pattern_tests {
    //! Host-side tests for the CAST FORMAT pattern vocabulary
    //! (`parse_cast_format_pattern`). No GPU / provider involved.
    use super::*;

    #[test]
    fn parses_full_datetime_pattern() {
        let toks = parse_cast_format_pattern("YYYY-MM-DD HH24:MI:SS").expect("valid pattern");
        assert_eq!(
            toks,
            vec![
                FormatToken::Year4,
                FormatToken::Literal('-'),
                FormatToken::Month,
                FormatToken::Literal('-'),
                FormatToken::Day,
                FormatToken::Literal(' '),
                FormatToken::Hour24,
                FormatToken::Literal(':'),
                FormatToken::Minute,
                FormatToken::Literal(':'),
                FormatToken::Second,
            ]
        );
    }

    #[test]
    fn hh_and_hh24_both_map_to_hour24() {
        assert_eq!(
            parse_cast_format_pattern("HH:MI").unwrap(),
            vec![FormatToken::Hour24, FormatToken::Literal(':'), FormatToken::Minute]
        );
        assert_eq!(
            parse_cast_format_pattern("HH24").unwrap(),
            vec![FormatToken::Hour24]
        );
    }

    #[test]
    fn is_case_insensitive_for_field_tokens() {
        assert_eq!(
            parse_cast_format_pattern("yyyy/mm/dd").unwrap(),
            vec![
                FormatToken::Year4,
                FormatToken::Literal('/'),
                FormatToken::Month,
                FormatToken::Literal('/'),
                FormatToken::Day,
            ]
        );
    }

    #[test]
    fn rejects_unknown_alphabetic_token_naming_it() {
        let err = parse_cast_format_pattern("YYYY-WW").expect_err("WW is not supported");
        let msg = format!("{err}");
        assert!(
            msg.contains("unsupported pattern token 'WW'"),
            "error must name the offending token, got: {msg}"
        );
    }

    #[test]
    fn rejects_unknown_separator() {
        let err = parse_cast_format_pattern("YYYY.MM").expect_err("'.' is not allowed");
        let msg = format!("{err}");
        assert!(
            msg.contains("unsupported separator '.'"),
            "error must name the bad separator, got: {msg}"
        );
    }

    #[test]
    fn rejects_empty_pattern() {
        assert!(parse_cast_format_pattern("").is_err());
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
        // Modulo / bitwise / shift — integer-only operators (Int32/Int64).
        // The logical-plan type-checker rejects float / decimal operands with
        // a clear message; the physical plan lowers these to guarded modulo /
        // bitwise / shift PTX. Both Postgres-flavoured `#` (PGBitwiseXor) and
        // ANSI `^` (BitwiseXor) spell XOR.
        BinaryOperator::Modulo => BinaryOp::Mod,
        BinaryOperator::BitwiseAnd => BinaryOp::BitAnd,
        BinaryOperator::BitwiseOr => BinaryOp::BitOr,
        BinaryOperator::BitwiseXor | BinaryOperator::PGBitwiseXor => BinaryOp::BitXor,
        BinaryOperator::PGBitwiseShiftLeft => BinaryOp::Shl,
        BinaryOperator::PGBitwiseShiftRight => BinaryOp::Shr,
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
    fn oversized_sql_is_rejected_not_panicked() {
        // DoS guard (V1): inputs over the byte / token caps must return a
        // clean `BoltError::Sql`, never panic or overflow the stack. We
        // assert on the pure `guard_sql_size` helper (independent of the
        // memoised global caps) for byte length, and synthesize a flat
        // operator chain that blows the token cap for the token path.
        // Byte cap: a string longer than the default 1 MiB.
        let big = "x".repeat(MAX_SQL_BYTES_DEFAULT + 1);
        assert!(
            matches!(guard_sql_size(&big), Err(BoltError::Sql(_))),
            "over-byte-cap SQL must be a clean Sql error"
        );

        // Token cap: a flat `1+1+1+…` chain that stays under the byte cap
        // but exceeds the token cap. This is exactly the AST-bloat shape
        // that crashes on recursive Drop if it reaches the parser.
        let mut adversarial = String::from("SELECT 1");
        // Each "+1" is 2 tokens; comfortably clear the 100k cap.
        for _ in 0..(MAX_SQL_TOKENS_DEFAULT) {
            adversarial.push_str("+1");
        }
        assert!(adversarial.len() <= MAX_SQL_BYTES_DEFAULT);
        assert!(
            matches!(guard_sql_size(&adversarial), Err(BoltError::Sql(_))),
            "over-token-cap SQL must be a clean Sql error"
        );

        // And end-to-end through the public entry point: still an Err,
        // still no panic.
        let p = provider();
        assert!(parse(&adversarial, &p).is_err());

        // A normal query passes the guard unharmed.
        assert!(guard_sql_size("SELECT a FROM t1").is_ok());
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

    // -------------------------------------------------------------------
    // F4: TRY_CAST / SAFE_CAST frontend acceptance + routing
    // -------------------------------------------------------------------

    /// Helper: pull the (unaliased) Expr at projection slot `n` from a
    /// top-level `Project`, panicking with the shape if it isn't one.
    fn nth_proj_expr(plan: &LogicalPlan, n: usize) -> Expr {
        match plan {
            LogicalPlan::Project { exprs, .. } => {
                let mut e = &exprs[n];
                while let Expr::Alias(inner, _) = e {
                    e = inner;
                }
                e.clone()
            }
            other => panic!("expected top-level Project, got {other:?}"),
        }
    }

    /// `TRY_CAST(<expr> AS <type>)` is now accepted (no longer rejected) and
    /// lowers to `Expr::Cast { safe: true, .. }`.
    #[test]
    fn try_cast_parses_to_safe_cast() {
        let plan = lp("SELECT TRY_CAST(a AS BIGINT) FROM t1");
        match nth_proj_expr(&plan, 0) {
            Expr::Cast { safe, target, .. } => {
                assert!(safe, "TRY_CAST must set safe = true");
                assert_eq!(target, DataType::Int64);
            }
            other => panic!("expected Expr::Cast, got {other:?}"),
        }
    }

    /// `SAFE_CAST` is a synonym for `TRY_CAST` — also `safe = true`.
    #[test]
    fn safe_cast_is_synonym_for_try_cast() {
        let plan = lp("SELECT SAFE_CAST(a AS BIGINT) FROM t1");
        match nth_proj_expr(&plan, 0) {
            Expr::Cast { safe, .. } => assert!(safe, "SAFE_CAST must set safe = true"),
            other => panic!("expected Expr::Cast, got {other:?}"),
        }
    }

    /// Plain `CAST` keeps `safe = false` (semantics preserved).
    #[test]
    fn plain_cast_stays_unsafe() {
        let plan = lp("SELECT CAST(a AS BIGINT) FROM t1");
        match nth_proj_expr(&plan, 0) {
            Expr::Cast { safe, .. } => assert!(!safe, "plain CAST must keep safe = false"),
            other => panic!("expected Expr::Cast, got {other:?}"),
        }
    }

    /// A projection carrying a safe cast routes to the host-side
    /// `PhysicalPlan::Project` (NULL-on-failure semantics live in the host
    /// evaluator). Plain CAST stays on the GPU `Projection` path.
    #[test]
    fn safe_cast_routes_to_host_project() {
        let safe = pp("SELECT TRY_CAST(a AS BIGINT) FROM t1");
        assert!(
            matches!(safe, PhysicalPlan::Project { .. }),
            "TRY_CAST projection must lower to host Project, got {safe:?}"
        );
        let plain = pp("SELECT CAST(a AS BIGINT) FROM t1");
        assert!(
            matches!(plain, PhysicalPlan::Projection { .. }),
            "plain CAST projection must stay on GPU Projection, got {plain:?}"
        );
    }

    // -------------------------------------------------------------------
    // Feature CAST FORMAT — frontend lowering + routing.
    // -------------------------------------------------------------------

    /// A provider with a temporal (`d` Date32, `ts` Timestamp) and a Utf8
    /// (`s`) column so both CAST FORMAT directions can be lowered + typed.
    fn temporal_provider() -> MemTableProvider {
        use crate::plan::logical_plan::Field;
        let t = Schema::new(vec![
            Field::new("d", DataType::Date32, false),
            Field::new("ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
            Field::new("s", DataType::Utf8, false),
        ]);
        MemTableProvider::new().with_table("tt", t)
    }

    fn tp_lp(sql: &str) -> LogicalPlan {
        parse(sql, &temporal_provider())
            .unwrap_or_else(|e| panic!("parse failed for {sql:?}: {e}"))
    }

    /// `CAST(d AS VARCHAR FORMAT 'YYYY-MM-DD')` lowers to `Expr::CastFormat`
    /// (to_text, target Utf8) and type-checks to Utf8.
    #[test]
    fn cast_format_date_to_string_lowers() {
        let plan = tp_lp("SELECT CAST(d AS VARCHAR FORMAT 'YYYY-MM-DD') FROM tt");
        match nth_proj_expr(&plan, 0) {
            Expr::CastFormat {
                target,
                to_text,
                pattern,
                ..
            } => {
                assert!(to_text, "temporal→string must set to_text = true");
                assert_eq!(target, DataType::Utf8);
                assert_eq!(pattern.first(), Some(&FormatToken::Year4));
            }
            other => panic!("expected Expr::CastFormat, got {other:?}"),
        }
        // Whole plan type-checks (the projected column is Utf8).
        assert!(plan.schema().is_ok(), "CAST FORMAT plan must type-check");
    }

    /// `CAST(s AS DATE FORMAT 'YYYY-MM-DD')` lowers to `Expr::CastFormat`
    /// (parse direction, target Date32).
    #[test]
    fn cast_format_string_to_date_lowers() {
        let plan = tp_lp("SELECT CAST(s AS DATE FORMAT 'YYYY-MM-DD') FROM tt");
        match nth_proj_expr(&plan, 0) {
            Expr::CastFormat { target, to_text, .. } => {
                assert!(!to_text, "string→temporal must set to_text = false");
                assert_eq!(target, DataType::Date32);
            }
            other => panic!("expected Expr::CastFormat, got {other:?}"),
        }
    }

    /// A CAST FORMAT projection routes to the host-side `PhysicalPlan::Project`
    /// (it is host-only; no GPU codegen).
    #[test]
    fn cast_format_routes_to_host_project() {
        let logical = tp_lp("SELECT CAST(d AS VARCHAR FORMAT 'YYYY-MM-DD') FROM tt");
        let phys = lower(&logical).expect("CAST FORMAT must lower without error");
        assert!(
            matches!(phys, PhysicalPlan::Project { .. }),
            "CAST FORMAT projection must lower to host Project, got {phys:?}"
        );
    }

    /// An unknown pattern token is rejected at lowering, naming the token.
    #[test]
    fn cast_format_unknown_token_rejected() {
        let err = parse(
            "SELECT CAST(d AS VARCHAR FORMAT 'YYYY-QQ') FROM tt",
            &temporal_provider(),
        )
        .expect_err("unknown token QQ must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("unsupported pattern token 'QQ'"),
            "error must name the bad token, got: {msg}"
        );
    }

    /// An intra-day token against a Date32 operand is an unsupported type/
    /// pattern combination — rejected cleanly at type-check (the projection
    /// lowers, but `schema()` type-checks the CAST FORMAT and rejects it).
    #[test]
    fn cast_format_intraday_token_on_date32_rejected() {
        let plan = tp_lp("SELECT CAST(d AS VARCHAR FORMAT 'YYYY-MM-DD HH24:MI') FROM tt");
        let err = plan
            .schema()
            .expect_err("HH/MI on a Date32 operand must type-error");
        let msg = format!("{err}");
        assert!(
            msg.contains("time-of-day token") && msg.contains("Date32"),
            "error must explain the Date32 / intra-day mismatch, got: {msg}"
        );
    }

    /// Formatting a non-temporal operand (Utf8) to string via FORMAT is an
    /// unsupported type combination — rejected cleanly at type-check.
    #[test]
    fn cast_format_nontemporal_operand_rejected() {
        // `s` is Utf8; formatting a Utf8 *to* VARCHAR via FORMAT is nonsense
        // (the to_text direction requires a temporal operand).
        let plan = tp_lp("SELECT CAST(s AS VARCHAR FORMAT 'YYYY-MM-DD') FROM tt");
        let err = plan
            .schema()
            .expect_err("formatting a Utf8 operand to text must type-error");
        let msg = format!("{err}");
        assert!(
            msg.contains("requires a Date32 or") && msg.contains("Timestamp"),
            "error must require a temporal operand, got: {msg}"
        );
    }

    /// `TRY_CAST(... FORMAT ...)` is rejected (FORMAT is plain-CAST only).
    #[test]
    fn try_cast_with_format_rejected() {
        let err = parse(
            "SELECT TRY_CAST(d AS VARCHAR FORMAT 'YYYY-MM-DD') FROM tt",
            &temporal_provider(),
        )
        .expect_err("TRY_CAST + FORMAT must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("TRY_CAST") && msg.contains("FORMAT"),
            "error must explain TRY_CAST/FORMAT is unsupported, got: {msg}"
        );
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

    /// `FROM t1, t2` desugars to a single CROSS JOIN (comma == cross product).
    #[test]
    fn comma_from_two_tables_desugars_to_cross_join() {
        let plan = lp("SELECT * FROM t1, t2");
        let join_plan = match &plan {
            LogicalPlan::Project { input, .. } => input.as_ref(),
            other => other,
        };
        match join_plan {
            LogicalPlan::Join {
                join_type, on, filter, ..
            } => {
                assert_eq!(*join_type, JoinType::Cross);
                assert!(on.is_empty(), "comma FROM has no ON predicate");
                assert!(filter.is_none(), "no WHERE -> no residual filter");
            }
            other => panic!("expected Cross Join, got {other:?}"),
        }
    }

    /// `FROM a, b, c` desugars to a LEFT-DEEP chain of CROSS JOINs:
    /// `((a CROSS JOIN b) CROSS JOIN c)`.
    #[test]
    fn comma_from_three_tables_is_left_deep_cross_chain() {
        use crate::plan::logical_plan::Field;
        // Third table with a distinct column name to avoid extra renames.
        let provider = MemTableProvider::new()
            .with_table(
                "t1",
                Schema::new(vec![Field::new("a", DataType::Int32, false)]),
            )
            .with_table(
                "t2",
                Schema::new(vec![Field::new("d", DataType::Int32, false)]),
            )
            .with_table(
                "t3",
                Schema::new(vec![Field::new("e", DataType::Int32, false)]),
            );
        let plan = parse("SELECT * FROM t1, t2, t3", &provider).unwrap();
        let top = match &plan {
            LogicalPlan::Project { input, .. } => input.as_ref(),
            other => other,
        };
        // Top join: (… ) CROSS JOIN t3.
        match top {
            LogicalPlan::Join { join_type, left, .. } => {
                assert_eq!(*join_type, JoinType::Cross);
                // Left child is itself a CROSS join (t1 CROSS JOIN t2).
                match left.as_ref() {
                    LogicalPlan::Join { join_type, .. } => {
                        assert_eq!(*join_type, JoinType::Cross)
                    }
                    other => panic!("expected nested Cross Join, got {other:?}"),
                }
            }
            other => panic!("expected top Cross Join, got {other:?}"),
        }
    }

    /// `FROM t1, t2 WHERE t1.a = t2.a` is correct as a cross-product + WHERE
    /// Filter. `parse` (the frontend) lowers it to `Project(Filter(Cross
    /// Join))`; the engine's `filter-into-join` optimizer later folds the
    /// equality into the Cross join's residual filter (it is NOT promoted to a
    /// hash equi-pair), which the nested-loop join evaluates. Here we assert
    /// the truthful pre-optimization shape produced by `parse`.
    #[test]
    fn comma_from_with_where_equality_is_filter_over_cross_join() {
        // The default `provider()` here gives t1(a,b) and t2(a,c); both have
        // an `a`, so qualify to disambiguate.
        let plan = lp("SELECT t1.b, t2.c FROM t1, t2 WHERE t1.a = t2.a");
        // Project -> Filter -> Cross Join.
        let filter = match &plan {
            LogicalPlan::Project { input, .. } => input.as_ref(),
            other => other,
        };
        let join = match filter {
            LogicalPlan::Filter { input, predicate } => {
                // The WHERE equality is preserved as the Filter predicate.
                assert!(
                    matches!(predicate, Expr::Binary { op: BinaryOp::Eq, .. }),
                    "expected an equality predicate, got {predicate:?}"
                );
                input.as_ref()
            }
            other => panic!("expected Filter over the join, got {other:?}"),
        };
        match join {
            LogicalPlan::Join { join_type, on, filter, .. } => {
                assert_eq!(*join_type, JoinType::Cross);
                assert!(on.is_empty() && filter.is_none(), "pre-optimization Cross");
            }
            other => panic!("expected Cross Join, got {other:?}"),
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
                        Expr::CastFormat { expr, .. } => stack.push(expr),
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
                assert!(matches!(join_type, JoinType::Inner));
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

    // ---- F3: COUNT(DISTINCT col) combined with SELECT DISTINCT / HAVING ----

    /// Bare `COUNT(DISTINCT col)` still lowers to the established shape:
    /// Project over Aggregate(COUNT*) over Distinct over Project over Filter.
    /// Locks the baseline the new combos build on.
    #[test]
    fn count_distinct_bare_lowers_to_project_over_aggregate() {
        let plan = lp("SELECT COUNT(DISTINCT a) FROM t1");
        match &plan {
            LogicalPlan::Project { input, .. } => {
                assert!(
                    matches!(**input, LogicalPlan::Aggregate { .. }),
                    "expected Aggregate under the result Project, got {input:?}"
                );
            }
            other => panic!("expected Project at top, got {other:?}"),
        }
        // And it lowers physically (no rejection) through the dedicated
        // CountRows-over-Distinct path. Assert the CountRows node is present
        // rather than over-fitting the exact wrapper nesting.
        let phys = lower(&plan).unwrap();
        assert!(
            format!("{phys:?}").contains("CountRows"),
            "expected a CountRows node in the physical plan, got {phys:?}"
        );
    }

    /// `SELECT DISTINCT COUNT(DISTINCT col)` (no GROUP BY) is now accepted and
    /// lowers to a `Distinct` wrapping the bare distinct-count plan. Reuses the
    /// existing `Distinct` operator — DISTINCT over the single result row is a
    /// correct no-op.
    #[test]
    fn select_distinct_over_count_distinct_is_accepted() {
        let plan = lp("SELECT DISTINCT COUNT(DISTINCT a) FROM t1");
        assert!(
            matches!(plan, LogicalPlan::Distinct { .. }),
            "expected Distinct at top, got {plan:?}"
        );
        // Physical lowering succeeds end-to-end (no rejection).
        let _ = lower(&plan).unwrap();
    }

    /// `HAVING` over a bare `COUNT(DISTINCT col)` (no GROUP BY) is now accepted:
    /// the whole table is one implicit group, so HAVING filters the single
    /// result row. Lowers to a `Filter` over the result `Project`, reusing the
    /// existing `Filter` operator.
    #[test]
    fn having_over_count_distinct_is_accepted() {
        let plan = lp("SELECT COUNT(DISTINCT a) FROM t1 HAVING COUNT(DISTINCT a) > 2");
        match &plan {
            LogicalPlan::Filter { input, .. } => {
                assert!(
                    matches!(**input, LogicalPlan::Project { .. }),
                    "expected Project under the HAVING Filter, got {input:?}"
                );
            }
            other => panic!("expected Filter at top, got {other:?}"),
        }
        let _ = lower(&plan).unwrap();
    }

    /// HAVING + SELECT DISTINCT stack together over the bare distinct-count:
    /// Distinct over Filter over Project.
    #[test]
    fn having_and_select_distinct_stack_over_count_distinct() {
        let plan =
            lp("SELECT DISTINCT COUNT(DISTINCT a) FROM t1 HAVING COUNT(DISTINCT a) >= 1");
        match &plan {
            LogicalPlan::Distinct { input } => {
                assert!(
                    matches!(**input, LogicalPlan::Filter { .. }),
                    "expected Filter under the top Distinct, got {input:?}"
                );
            }
            other => panic!("expected Distinct at top, got {other:?}"),
        }
        let _ = lower(&plan).unwrap();
    }

    /// HAVING referencing a COUNT(DISTINCT ...) over a *different* column than
    /// the projected one is rejected cleanly (no panic).
    #[test]
    fn having_over_mismatched_count_distinct_column_is_rejected() {
        let err = parse(
            "SELECT COUNT(DISTINCT a) FROM t1 HAVING COUNT(DISTINCT b) > 0",
            &provider(),
        )
        .expect_err("mismatched HAVING distinct-count column must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("different column"),
            "expected mismatched-column message, got: {msg}"
        );
    }

    /// COUNT(DISTINCT col) with GROUP BY through the *raw* `parse` API (which
    /// bypasses the engine's host-orchestrated F3-finish detector — the feature
    /// is wired into `Engine::sql` only, like WITH RECURSIVE) is rejected
    /// cleanly. The same is true for a plain GROUP BY whose key (`a` here) is
    /// not projected in the SELECT list, which the detector also declines.
    /// Must be a clean `Err`, no panic.
    #[test]
    fn count_distinct_with_group_by_is_rejected_cleanly() {
        let err = parse(
            "SELECT COUNT(DISTINCT b) FROM t1 GROUP BY a",
            &provider(),
        )
        .expect_err("COUNT(DISTINCT) with GROUP BY must be rejected on the parse path");
        let msg = format!("{err}");
        assert!(
            msg.contains("GROUP BY"),
            "expected a GROUP BY rejection message, got: {msg}"
        );
    }

    /// COUNT(DISTINCT col) alongside another SELECT item remains rejected
    /// cleanly (the sole-item restriction is unchanged).
    #[test]
    fn count_distinct_with_extra_select_item_is_rejected_cleanly() {
        let err = parse(
            "SELECT a, COUNT(DISTINCT b) FROM t1",
            &provider(),
        )
        .expect_err("COUNT(DISTINCT) alongside another item must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("sole SELECT item"),
            "expected sole-item rejection message, got: {msg}"
        );
    }

    // ===================================================================
    // F2 — GROUP BY ROLLUP / CUBE / GROUPING SETS
    //
    // The frontend rewrites these into a UNION ALL of ordinary GROUP BY
    // aggregates (one branch per grouping set), NULL-filling the group
    // columns that are inactive in a given set so all branches share one
    // schema. These tests lock the plan *shape*: the right number of
    // Aggregate branches and the typed-NULL fills. Execution is covered
    // elsewhere (host/GPU).
    // ===================================================================

    /// Three-column fixture: two group keys of different dtypes plus a
    /// numeric measure, so NULL-fill dtype correctness is exercised.
    fn rollup_provider() -> MemTableProvider {
        use crate::plan::logical_plan::Field;
        let sales = Schema::new(vec![
            Field::new("region", DataType::Utf8, false),
            Field::new("year", DataType::Int32, false),
            Field::new("amount", DataType::Float64, false),
        ]);
        MemTableProvider::new().with_table("sales", sales)
    }

    /// Pull the list of UNION branches out of a plan, accepting an optional
    /// outer Sort/Limit/Filter wrapping. Panics if the top isn't a Union.
    fn union_branches(plan: &LogicalPlan) -> &[LogicalPlan] {
        let mut p = plan;
        loop {
            match p {
                LogicalPlan::Union { inputs } => return inputs,
                LogicalPlan::Sort { input, .. }
                | LogicalPlan::Limit { input, .. }
                | LogicalPlan::Filter { input, .. }
                | LogicalPlan::Distinct { input } => p = input,
                other => panic!("expected Union (optionally wrapped), got {other:?}"),
            }
        }
    }

    /// Count the active GROUP BY keys of a branch's Aggregate node.
    fn branch_group_key_count(branch: &LogicalPlan) -> usize {
        match branch {
            LogicalPlan::Project { input, .. } => match input.as_ref() {
                LogicalPlan::Aggregate { group_by, .. } => group_by.len(),
                other => panic!("expected Aggregate under Project, got {other:?}"),
            },
            other => panic!("expected Project branch, got {other:?}"),
        }
    }

    /// Count typed-NULL fills (Alias(Cast(NULL))) in a branch's projection.
    fn branch_null_fill_count(branch: &LogicalPlan) -> usize {
        let exprs = match branch {
            LogicalPlan::Project { exprs, .. } => exprs,
            other => panic!("expected Project branch, got {other:?}"),
        };
        exprs
            .iter()
            .filter(|e| {
                let inner = match e {
                    Expr::Alias(inner, _) => inner.as_ref(),
                    other => other,
                };
                matches!(
                    inner,
                    Expr::Cast { expr, .. }
                        if matches!(expr.as_ref(), Expr::Literal(Literal::Null))
                )
            })
            .count()
    }

    #[test]
    fn rollup_expands_to_prefix_grouping_sets() {
        // ROLLUP(region, year) -> {(region,year),(region),()} = 3 branches.
        let plan = lp_with(
            "SELECT region, year, SUM(amount) FROM sales GROUP BY ROLLUP(region, year)",
            &rollup_provider(),
        );
        let branches = union_branches(&plan);
        assert_eq!(branches.len(), 3, "ROLLUP(2 cols) => 3 prefix sets");
        // Active group-key counts across branches: 2, 1, 0.
        let mut counts: Vec<usize> = branches.iter().map(branch_group_key_count).collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![0, 1, 2]);
        // NULL-fill counts complement the active counts (2 group cols total):
        // 0, 1, 2 fills respectively.
        let mut fills: Vec<usize> = branches.iter().map(branch_null_fill_count).collect();
        fills.sort_unstable();
        assert_eq!(fills, vec![0, 1, 2]);
        // The whole plan still type-checks (UNION schema-compat passes).
        assert!(plan.schema().is_ok(), "ROLLUP union schema must recompute");
        // Result schema: region(Utf8), year(Int32), sum(amount)(Float64),
        // group cols nullable.
        let schema = plan.schema().unwrap();
        assert_eq!(schema.fields.len(), 3);
        assert_eq!(schema.fields[0].dtype, DataType::Utf8);
        assert_eq!(schema.fields[1].dtype, DataType::Int32);
        assert_eq!(schema.fields[2].dtype, DataType::Float64);
        assert!(schema.fields[0].nullable && schema.fields[1].nullable);
    }

    #[test]
    fn cube_expands_to_all_subsets() {
        // CUBE(region, year) -> {(region,year),(region),(year),()} = 4 branches.
        let plan = lp_with(
            "SELECT region, year, SUM(amount) FROM sales GROUP BY CUBE(region, year)",
            &rollup_provider(),
        );
        let branches = union_branches(&plan);
        assert_eq!(branches.len(), 4, "CUBE(2 cols) => 2^2 = 4 sets");
        let mut counts: Vec<usize> = branches.iter().map(branch_group_key_count).collect();
        counts.sort_unstable();
        // Subset sizes for 2 columns: {2,1,1,0}.
        assert_eq!(counts, vec![0, 1, 1, 2]);
        assert!(plan.schema().is_ok(), "CUBE union schema must recompute");
    }

    #[test]
    fn grouping_sets_uses_explicit_list() {
        // Explicit GROUPING SETS ((region, year), (region), ()) = 3 branches.
        let plan = lp_with(
            "SELECT region, year, SUM(amount) FROM sales \
             GROUP BY GROUPING SETS ((region, year), (region), ())",
            &rollup_provider(),
        );
        let branches = union_branches(&plan);
        assert_eq!(branches.len(), 3);
        let mut counts: Vec<usize> = branches.iter().map(branch_group_key_count).collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![0, 1, 2]);
        assert!(plan.schema().is_ok());
    }

    #[test]
    fn grouping_sets_deduplicates_repeated_sets() {
        // A repeated grouping set is collapsed to one branch (UNION ALL would
        // otherwise duplicate rows).
        let plan = lp_with(
            "SELECT region, SUM(amount) FROM sales \
             GROUP BY GROUPING SETS ((region), (region), ())",
            &rollup_provider(),
        );
        let branches = union_branches(&plan);
        assert_eq!(branches.len(), 2, "duplicate (region) set must be deduped");
    }

    #[test]
    fn cube_column_cap_rejected() {
        use crate::plan::logical_plan::Field;
        // Build a 13-column table and CUBE over all 13 (> MAX_GROUPING_SET_COLUMNS).
        let mut fields: Vec<Field> = Vec::new();
        for i in 0..13 {
            fields.push(Field::new(format!("c{i}"), DataType::Int32, false));
        }
        let prov = MemTableProvider::new().with_table("wide", Schema::new(fields));
        let cols: Vec<String> = (0..13).map(|i| format!("c{i}")).collect();
        let sql = format!(
            "SELECT {}, COUNT(*) FROM wide GROUP BY CUBE({})",
            cols.join(", "),
            cols.join(", ")
        );
        let err = parse(&sql, &prov).expect_err("CUBE with 13 columns must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("exceeds the limit"),
            "expected blowup-cap rejection, got: {msg}"
        );
    }

    // ---- F1/F2/F3 helpers --------------------------------------------------

    /// Read the literal projected by a branch's Project at position `pos`,
    /// unwrapping an enclosing Alias. Returns the constant Int64 value (panics
    /// if the slot is not a (possibly aliased) integer literal).
    fn branch_int_literal_at(branch: &LogicalPlan, pos: usize) -> i64 {
        let exprs = match branch {
            LogicalPlan::Project { exprs, .. } => exprs,
            other => panic!("expected Project branch, got {other:?}"),
        };
        let mut e = &exprs[pos];
        if let Expr::Alias(inner, _) = e {
            e = inner.as_ref();
        }
        match e {
            Expr::Literal(Literal::Int64(v)) => *v,
            other => panic!("expected Int64 literal at {pos}, got {other:?}"),
        }
    }

    /// Active group-key count for a branch's Aggregate (alias of the F2 helper,
    /// kept local for the F1/F3 tests that read it).
    fn branch_keys(branch: &LogicalPlan) -> usize {
        branch_group_key_count(branch)
    }

    // ---- F2: GROUPING() / GROUPING_ID() indicator --------------------------

    #[test]
    fn grouping_indicator_emits_per_branch_bit() {
        // ROLLUP(region) -> branches {(region), ()}. GROUPING(region) is 0 in
        // the (region) branch (active) and 1 in the () branch (NULL-filled).
        let plan = lp_with(
            "SELECT region, GROUPING(region) AS g, SUM(amount) FROM sales \
             GROUP BY ROLLUP(region)",
            &rollup_provider(),
        );
        let branches = union_branches(&plan);
        assert_eq!(branches.len(), 2);
        // Position 1 in each branch's projection is GROUPING(region).
        // Pair (active-key-count, grouping-bit): (1,0) and (0,1).
        let mut pairs: Vec<(usize, i64)> = branches
            .iter()
            .map(|b| (branch_keys(b), branch_int_literal_at(b, 1)))
            .collect();
        pairs.sort_unstable();
        assert_eq!(pairs, vec![(0, 1), (1, 0)]);
        assert!(plan.schema().is_ok(), "GROUPING union schema must recompute");
    }

    #[test]
    fn grouping_id_multi_arg_bitmask() {
        // CUBE(region, year) -> 4 branches. GROUPING_ID(region, year) is the
        // 2-bit mask: bit1=region inactive, bit0=year inactive.
        //   (region,year): 0b00 = 0   (year):  0b10 = 2
        //   (region):      0b01 = 1   ():       0b11 = 3
        let plan = lp_with(
            "SELECT region, year, GROUPING_ID(region, year) AS gid, SUM(amount) \
             FROM sales GROUP BY CUBE(region, year)",
            &rollup_provider(),
        );
        let branches = union_branches(&plan);
        assert_eq!(branches.len(), 4);
        // GROUPING_ID is at projection position 2.
        let mut masks: Vec<i64> = branches
            .iter()
            .map(|b| branch_int_literal_at(b, 2))
            .collect();
        masks.sort_unstable();
        assert_eq!(masks, vec![0, 1, 2, 3], "all 2-bit masks must appear once");
        assert!(plan.schema().is_ok());
    }

    #[test]
    fn grouping_without_construct_rejected() {
        // Plain GROUP BY (no grouping-set construct): GROUPING() is meaningless.
        let err = parse(
            "SELECT region, GROUPING(region) FROM sales GROUP BY region",
            &rollup_provider(),
        )
        .expect_err("GROUPING() without a construct must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("GROUPING") && msg.contains("ROLLUP"),
            "expected GROUPING-requires-construct message, got: {msg}"
        );
    }

    #[test]
    fn grouping_arg_not_a_group_column_rejected() {
        // GROUPING(amount) where `amount` is not a grouping column.
        let err = parse(
            "SELECT region, GROUPING(amount), SUM(amount) FROM sales \
             GROUP BY ROLLUP(region)",
            &rollup_provider(),
        )
        .expect_err("GROUPING() over a non-group column must be rejected");
        assert!(format!("{err}").contains("grouping column"));
    }

    // ---- F3: WITH TOTALS ---------------------------------------------------

    #[test]
    fn with_totals_expands_to_keys_and_grand_total() {
        // `GROUP BY k WITH TOTALS` ≡ grouping sets {(k), ()} = 2 branches.
        let plan = lp_with(
            "SELECT k, SUM(v) FROM t GROUP BY k WITH TOTALS",
            &having_provider(),
        );
        let branches = union_branches(&plan);
        assert_eq!(branches.len(), 2, "WITH TOTALS => {{(k), ()}}");
        let mut counts: Vec<usize> = branches.iter().map(branch_keys).collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![0, 1], "one full branch + one grand-total");
        // The grand-total branch NULL-fills `k`.
        let fills: usize = branches.iter().map(branch_null_fill_count).sum();
        assert_eq!(fills, 1);
        assert!(plan.schema().is_ok());
    }

    #[test]
    fn with_totals_multi_key() {
        // Two group keys with WITH TOTALS -> {(a,b), ()} (NOT a rollup).
        let plan = lp_with(
            "SELECT region, year, SUM(amount) FROM sales \
             GROUP BY region, year WITH TOTALS",
            &rollup_provider(),
        );
        let branches = union_branches(&plan);
        assert_eq!(branches.len(), 2, "WITH TOTALS => {{(region,year), ()}}");
        let mut counts: Vec<usize> = branches.iter().map(branch_keys).collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![0, 2]);
    }

    #[test]
    fn with_rollup_modifier_expands_like_rollup() {
        // MySQL trailing `WITH ROLLUP` ≡ ROLLUP(region, year) -> 3 prefix sets.
        let plan = lp_with(
            "SELECT region, year, SUM(amount) FROM sales \
             GROUP BY region, year WITH ROLLUP",
            &rollup_provider(),
        );
        let branches = union_branches(&plan);
        assert_eq!(branches.len(), 3, "WITH ROLLUP(2 cols) => 3 prefix sets");
        let mut counts: Vec<usize> = branches.iter().map(branch_keys).collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![0, 1, 2]);
    }

    #[test]
    fn with_totals_combined_with_construct_rejected() {
        // A trailing modifier may not combine with an explicit construct.
        let err = parse(
            "SELECT region, SUM(amount) FROM sales \
             GROUP BY ROLLUP(region) WITH TOTALS",
            &rollup_provider(),
        )
        .expect_err("WITH TOTALS + ROLLUP(...) must be rejected");
        assert!(format!("{err}").contains("modifier"));
    }

    // ---- F1: GROUP BY ALL --------------------------------------------------

    #[test]
    fn group_by_all_collects_non_aggregate_keys() {
        // GROUP BY ALL with one aggregate -> group by the single non-aggregate
        // column `k`; plain (non-super) single-branch Aggregate.
        let plan = lp_with(
            "SELECT k, SUM(v) FROM t GROUP BY ALL",
            &having_provider(),
        );
        // Not a UNION (plain group-by fast path); end is Project(Aggregate(..)).
        match &plan {
            LogicalPlan::Project { input, .. } => match input.as_ref() {
                LogicalPlan::Aggregate { group_by, aggregates, .. } => {
                    assert_eq!(group_by.len(), 1, "GROUP BY ALL groups by `k`");
                    assert_eq!(aggregates.len(), 1, "one SUM aggregate");
                }
                other => panic!("expected Aggregate under Project, got {other:?}"),
            },
            other => panic!("expected Project at top, got {other:?}"),
        }
        assert!(plan.schema().is_ok());
    }

    #[test]
    fn group_by_all_two_keys() {
        // Two non-aggregate columns + one aggregate -> group by both.
        let plan = lp_with(
            "SELECT region, year, SUM(amount) FROM sales GROUP BY ALL",
            &rollup_provider(),
        );
        match &plan {
            LogicalPlan::Project { input, .. } => match input.as_ref() {
                LogicalPlan::Aggregate { group_by, .. } => {
                    assert_eq!(group_by.len(), 2, "GROUP BY ALL groups by region, year");
                }
                other => panic!("expected Aggregate, got {other:?}"),
            },
            other => panic!("expected Project, got {other:?}"),
        }
    }

    #[test]
    fn group_by_all_no_aggregates_is_distinct_over_columns() {
        // GROUP BY ALL with NO aggregates groups by every selected column
        // (DuckDB: equivalent to SELECT DISTINCT over them).
        let plan = lp_with(
            "SELECT region, year FROM sales GROUP BY ALL",
            &rollup_provider(),
        );
        match &plan {
            LogicalPlan::Project { input, .. } => match input.as_ref() {
                LogicalPlan::Aggregate { group_by, aggregates, .. } => {
                    assert_eq!(group_by.len(), 2, "groups by both selected columns");
                    assert!(aggregates.is_empty(), "no aggregates");
                }
                other => panic!("expected Aggregate, got {other:?}"),
            },
            other => panic!("expected Project, got {other:?}"),
        }
        assert!(plan.schema().is_ok());
    }

    #[test]
    fn rollup_with_having_and_order_by_wraps_union() {
        // HAVING applies over the UNION result (Filter on top); ORDER BY adds a
        // Sort outside that. Outer query layering must see the union schema.
        let plan = lp_with(
            "SELECT region, SUM(amount) AS s FROM sales \
             GROUP BY ROLLUP(region) HAVING SUM(amount) > 0 ORDER BY region",
            &rollup_provider(),
        );
        // Top: Sort(Filter(Union(...))).
        match &plan {
            LogicalPlan::Sort { input, .. } => match input.as_ref() {
                LogicalPlan::Filter { input, .. } => {
                    assert!(matches!(input.as_ref(), LogicalPlan::Union { .. }));
                }
                other => panic!("expected Filter under Sort, got {other:?}"),
            },
            other => panic!("expected Sort at top, got {other:?}"),
        }
        // Two branches: (region) and ().
        assert_eq!(union_branches(&plan).len(), 2);
        assert!(plan.schema().is_ok());
    }

    #[test]
    fn rollup_single_column_two_branches() {
        // ROLLUP(region) -> {(region), ()} = 2 branches; the () branch
        // NULL-fills region.
        let plan = lp_with(
            "SELECT region, SUM(amount) FROM sales GROUP BY ROLLUP(region)",
            &rollup_provider(),
        );
        let branches = union_branches(&plan);
        assert_eq!(branches.len(), 2);
        let fills: usize = branches.iter().map(branch_null_fill_count).sum();
        assert_eq!(fills, 1, "exactly the grand-total branch NULL-fills region");
    }

    #[test]
    fn plain_group_by_unaffected_by_f2_refactor() {
        // A plain GROUP BY must still lower to a single Project(Aggregate(...)),
        // NOT a Union, and a bare group key stays a bare Column (no alias).
        let plan = lp_with(
            "SELECT k, SUM(v) FROM t GROUP BY k",
            &having_provider(),
        );
        match &plan {
            LogicalPlan::Project { input, exprs } => {
                assert!(matches!(input.as_ref(), LogicalPlan::Aggregate { .. }));
                // First projection is the bare group key `k`.
                assert!(
                    matches!(&exprs[0], Expr::Column(n) if n == "k"),
                    "plain group key should remain a bare Column, got {:?}",
                    exprs[0]
                );
            }
            other => panic!("expected Project(Aggregate), got {other:?}"),
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

    /// `ILIKE` lowers to `Expr::Like` with `case_insensitive = true` while
    /// keeping the pattern verbatim and `negated = false`.
    #[test]
    fn parse_ilike_sets_case_insensitive_flag() {
        let plan = parse("SELECT s FROM t WHERE s ILIKE 'Foo%'", &s_provider())
            .expect("ILIKE 'Foo%' must parse");
        match predicate(&plan) {
            Expr::Like {
                pattern,
                negated,
                case_insensitive,
                ..
            } => {
                assert_eq!(pattern, "Foo%");
                assert!(!negated);
                assert!(*case_insensitive, "ILIKE must set case_insensitive=true");
            }
            other => panic!("expected Expr::Like, got {other:?}"),
        }
    }

    /// Plain `LIKE` must keep `case_insensitive = false` (regression guard:
    /// the new flag must not leak into the case-sensitive path).
    #[test]
    fn parse_plain_like_is_case_sensitive() {
        let plan = parse("SELECT s FROM t WHERE s LIKE 'Foo%'", &s_provider())
            .expect("LIKE 'Foo%' must parse");
        match predicate(&plan) {
            Expr::Like {
                case_insensitive, ..
            } => assert!(
                !case_insensitive,
                "plain LIKE must stay case-sensitive (case_insensitive=false)"
            ),
            other => panic!("expected Expr::Like, got {other:?}"),
        }
    }

    /// `NOT ILIKE` sets both `negated` and `case_insensitive`.
    #[test]
    fn parse_not_ilike_sets_negated_and_case_insensitive() {
        let plan = parse("SELECT s FROM t WHERE s NOT ILIKE '%bar'", &s_provider())
            .expect("NOT ILIKE must parse");
        match predicate(&plan) {
            Expr::Like {
                pattern,
                negated,
                case_insensitive,
                ..
            } => {
                assert_eq!(pattern, "%bar");
                assert!(*negated, "NOT ILIKE must set negated=true");
                assert!(*case_insensitive, "NOT ILIKE must set case_insensitive=true");
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
    fn substring_and_trim_lower_to_string_project() {
        // F9: SUBSTRING(col, lit, lit) and single-arg TRIM(col) over a bare
        // Utf8 scan now lower to PhysicalPlan::StringProject (host-realized
        // two-pass producer). A custom-chars TRIM (`TRIM(<chars> FROM col)`)
        // is still out of scope and falls back to the host PhysicalPlan::Project.
        let string_project_cases = [
            "SELECT SUBSTRING(s, 2, 3) FROM t",
            "SELECT TRIM(s) FROM t",
        ];
        for sql in string_project_cases {
            let plan = parse(sql, &s_provider()).unwrap_or_else(|e| panic!("{sql}: {e}"));
            let phys = lower(&plan).unwrap_or_else(|e| panic!("lower {sql}: {e}"));
            assert!(
                matches!(phys, PhysicalPlan::StringProject { .. }),
                "expected PhysicalPlan::StringProject for {sql}, got {phys:?}"
            );
        }
        // Custom-chars TRIM stays on the host Project fallback (not rejected).
        let sql = "SELECT TRIM(TRAILING '-' FROM s) FROM t";
        let plan = parse(sql, &s_provider()).unwrap_or_else(|e| panic!("{sql}: {e}"));
        let phys = lower(&plan).unwrap_or_else(|e| panic!("lower {sql}: {e}"));
        assert!(
            matches!(phys, PhysicalPlan::Project { .. }),
            "expected host PhysicalPlan::Project for {sql}, got {phys:?}"
        );
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

    // ===================================================================
    // New string functions (Agent M): OCTET_LENGTH, CHAR_LENGTH /
    // CHARACTER_LENGTH, POSITION / STRPOS, REPLACE, LEFT/RIGHT, LPAD/RPAD,
    // REVERSE, INITCAP. These lower to `Expr::ScalarFn`. Parse-shape and
    // type-check coverage lives here; the per-string host transforms are
    // unit-tested in `exec::string_ops_extended`. GPU lowering (`lower()`)
    // is intentionally NOT exercised: these functions are host-evaluated and
    // the executor/physical-plan wiring (`expr_agg::eval_scalar_fn`,
    // `physical_plan::all_scalar_fns_host_evaluable`) is applied by the
    // orchestrator — see `reviews/done_M.md`.
    // ===================================================================

    /// Assert `sql`'s first SELECT expr is `Expr::ScalarFn(kind)` with `nargs`.
    fn assert_scalar_fn(sql: &str, kind: ScalarFnKind, nargs: usize) {
        let plan = parse(sql, &s_provider()).unwrap_or_else(|e| panic!("{sql}: {e}"));
        match first_select_expr(&plan) {
            Expr::ScalarFn { kind: k, args } => {
                assert_eq!(k, kind, "kind for {sql}");
                assert_eq!(args.len(), nargs, "arity for {sql}");
            }
            other => panic!("expected ScalarFn({kind:?}) for {sql}, got {other:?}"),
        }
    }

    #[test]
    fn octet_length_parses() {
        assert_scalar_fn("SELECT OCTET_LENGTH(s) FROM t", ScalarFnKind::OctetLength, 1);
    }

    #[test]
    fn char_length_synonyms_lower_to_length() {
        // CHAR_LENGTH and CHARACTER_LENGTH are synonyms for character LENGTH.
        assert_scalar_fn("SELECT CHAR_LENGTH(s) FROM t", ScalarFnKind::Length, 1);
        assert_scalar_fn(
            "SELECT CHARACTER_LENGTH(s) FROM t",
            ScalarFnKind::Length,
            1,
        );
    }

    #[test]
    fn position_in_form_normalises_arg_order() {
        // POSITION(substr IN s) -> ScalarFn(Position, [s, substr]).
        let plan = parse("SELECT POSITION('lo' IN s) FROM t", &s_provider())
            .expect("POSITION(substr IN s) must parse");
        match first_select_expr(&plan) {
            Expr::ScalarFn { kind, args } => {
                assert_eq!(kind, ScalarFnKind::Position);
                assert_eq!(args.len(), 2);
                // arg 0 = haystack (the column), arg 1 = needle (the literal).
                assert!(matches!(args[0], Expr::Column(_)), "arg0 = haystack s");
                assert!(
                    matches!(args[1], Expr::Literal(Literal::Utf8(ref n)) if n == "lo"),
                    "arg1 = needle literal"
                );
            }
            other => panic!("expected ScalarFn(Position), got {other:?}"),
        }
    }

    #[test]
    fn strpos_function_form_parses() {
        // STRPOS(s, substr) is the function spelling — same kind, [s, substr].
        assert_scalar_fn("SELECT STRPOS(s, 'x') FROM t", ScalarFnKind::Position, 2);
    }

    #[test]
    fn replace_parses() {
        assert_scalar_fn("SELECT REPLACE(s, 'a', 'b') FROM t", ScalarFnKind::Replace, 3);
    }

    #[test]
    fn left_right_parse() {
        assert_scalar_fn("SELECT LEFT(s, 3) FROM t", ScalarFnKind::Left, 2);
        assert_scalar_fn("SELECT RIGHT(s, 3) FROM t", ScalarFnKind::Right, 2);
    }

    #[test]
    fn lpad_rpad_parse() {
        assert_scalar_fn("SELECT LPAD(s, 5, '0') FROM t", ScalarFnKind::Lpad, 3);
        assert_scalar_fn("SELECT RPAD(s, 5, ' ') FROM t", ScalarFnKind::Rpad, 3);
    }

    #[test]
    fn reverse_initcap_parse() {
        assert_scalar_fn("SELECT REVERSE(s) FROM t", ScalarFnKind::Reverse, 1);
        assert_scalar_fn("SELECT INITCAP(s) FROM t", ScalarFnKind::Initcap, 1);
    }

    #[test]
    fn case_insensitive_function_names() {
        // SQL identifiers are case-insensitive: lower-case spellings work.
        assert_scalar_fn("SELECT octet_length(s) FROM t", ScalarFnKind::OctetLength, 1);
        assert_scalar_fn("SELECT reverse(s) FROM t", ScalarFnKind::Reverse, 1);
    }

    // ----- type-check (via schema()) -------------------------------------

    #[test]
    fn octet_length_returns_int64() {
        let plan = parse("SELECT OCTET_LENGTH(s) FROM t", &s_provider()).unwrap();
        let schema = plan.schema().expect("OCTET_LENGTH(Utf8) type-checks");
        assert_eq!(schema.fields[0].dtype, DataType::Int64);
    }

    #[test]
    fn position_returns_int64() {
        let plan = parse("SELECT POSITION('x' IN s) FROM t", &s_provider()).unwrap();
        let schema = plan.schema().expect("POSITION type-checks");
        assert_eq!(schema.fields[0].dtype, DataType::Int64);
    }

    #[test]
    fn replace_returns_utf8() {
        let plan = parse("SELECT REPLACE(s, 'a', 'b') FROM t", &s_provider()).unwrap();
        let schema = plan.schema().expect("REPLACE type-checks");
        assert_eq!(schema.fields[0].dtype, DataType::Utf8);
    }

    #[test]
    fn left_returns_utf8_and_requires_int_count() {
        let plan = parse("SELECT LEFT(s, 2) FROM t", &s_provider()).unwrap();
        let schema = plan.schema().expect("LEFT type-checks");
        assert_eq!(schema.fields[0].dtype, DataType::Utf8);
        // LEFT(s, <utf8>) must type-error on the count argument.
        let bad = parse("SELECT LEFT(s, s) FROM t", &s_provider()).unwrap();
        let err = bad.schema().expect_err("LEFT(s, Utf8) must type-error");
        assert!(format!("{err}").contains("LEFT"), "{err}");
    }

    #[test]
    fn octet_length_type_error_on_non_utf8() {
        let plan = parse("SELECT OCTET_LENGTH(v) FROM t", &s_provider()).unwrap();
        let err = plan.schema().expect_err("OCTET_LENGTH(Int64) must type-error");
        let msg = format!("{err}");
        assert!(
            msg.contains("OCTET_LENGTH") && msg.contains("Utf8"),
            "expected OCTET_LENGTH Utf8 type error, got: {msg}"
        );
    }

    #[test]
    fn replace_arity_error() {
        // REPLACE needs exactly 3 args.
        let plan = parse("SELECT REPLACE(s, 'a') FROM t", &s_provider()).unwrap();
        let err = plan.schema().expect_err("REPLACE/2 must type-error");
        assert!(
            format!("{err}").contains("REPLACE"),
            "expected REPLACE arity error, got: {err}"
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

    /// Provider for the F12 frontend-acceptance tests below, which query
    /// `t1`/`t2` (not `sales`). Mirrors the `wave7_tests` fixture.
    fn f12_provider() -> MemTableProvider {
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

    /// Parse + unwrap helper for the positive F12 tests.
    fn lp(sql: &str) -> LogicalPlan {
        parse(sql, &f12_provider())
            .unwrap_or_else(|e| panic!("parse failed for {sql:?}: {e}"))
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

    // ===================================================================
    // F12 — frontend acceptance gaps
    // ===================================================================

    // ---- F12.1: schema-qualified names in JOIN ON ---------------------

    /// `schema.table.col` in a JOIN ON predicate now resolves: the leading
    /// single-catalog segment is dropped and the trailing `table.col` pair is
    /// used (same as the 3-segment handling in plain expression lowering). The
    /// ON pair keeps the bare column names.
    #[test]
    fn join_on_schema_qualified_resolves() {
        let plan = lp("SELECT * FROM t1 JOIN t2 ON public.t1.a = public.t2.a");
        let join_plan = match &plan {
            LogicalPlan::Project { input, .. } => input.as_ref(),
            other => other,
        };
        match join_plan {
            LogicalPlan::Join { on, join_type, .. } => {
                assert!(matches!(join_type, JoinType::Inner));
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

    /// A schema-qualified JOIN ON reference whose middle (table) segment names
    /// no in-scope table still errors with the unknown-table message — the
    /// dropped schema segment does not paper over a bad table name.
    #[test]
    fn join_on_schema_qualified_unknown_table_rejected() {
        let err = parse(
            "SELECT * FROM t1 JOIN t2 ON public.t3.a = public.t2.a",
            &f12_provider(),
        )
        .expect_err("schema-qualified ON ref to unknown table must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown table 't3'"),
            "expected 'unknown table t3' message, got: {msg}"
        );
    }

    /// Four-or-more-segment column references in JOIN ON have no namespace to
    /// collapse and are still rejected as "deeply qualified".
    #[test]
    fn join_on_deeply_qualified_still_rejected() {
        let err = parse(
            "SELECT * FROM t1 JOIN t2 ON cat.public.t1.a = t2.a",
            &f12_provider(),
        )
        .expect_err("4-segment ON ref must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("deeply qualified"),
            "expected 'deeply qualified' message, got: {msg}"
        );
    }

    // ---- F12.2: scalar function calls --------------------------------

    /// A genuinely-unsupported scalar function keeps a clean error naming the
    /// function (no panic). `SQRT` has no downstream support, so it must be
    /// rejected — verifying the catch-all path still guards the binder.
    #[test]
    fn unsupported_scalar_function_rejected_cleanly() {
        let err = parse("SELECT SQRT(b) FROM t1", &f12_provider())
            .expect_err("SQRT has no downstream support and must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("scalar function") && msg.to_ascii_uppercase().contains("SQRT"),
            "expected clean message naming SQRT, got: {msg}"
        );
    }

    // ---- F12.3: scalar subqueries in ORDER BY ------------------------

    /// An uncorrelated scalar subquery in ORDER BY now lowers: it produces a
    /// `Sort` whose sort-expr carries a `ScalarSubquery` (the exec-side
    /// resolver folds it to a constant before physical lowering).
    #[test]
    fn order_by_scalar_subquery_lowers() {
        let plan = lp("SELECT a FROM t1 ORDER BY (SELECT MAX(a) FROM t2)");
        match plan {
            LogicalPlan::Sort { sort_exprs, .. } => {
                assert_eq!(sort_exprs.len(), 1);
                assert!(
                    matches!(sort_exprs[0].expr, Expr::ScalarSubquery(_)),
                    "expected ScalarSubquery sort key, got {:?}",
                    sort_exprs[0].expr
                );
            }
            other => panic!("expected Sort, got {other:?}"),
        }
    }

    /// `ORDER BY x IN (SELECT ...)` also lowers, carrying an `InSubquery` sort
    /// key (resolved to a boolean constant fold by the exec-side pass).
    #[test]
    fn order_by_in_subquery_lowers() {
        let plan = lp("SELECT a FROM t1 ORDER BY a IN (SELECT a FROM t2)");
        match plan {
            LogicalPlan::Sort { sort_exprs, .. } => {
                assert_eq!(sort_exprs.len(), 1);
                assert!(
                    matches!(sort_exprs[0].expr, Expr::InSubquery { .. }),
                    "expected InSubquery sort key, got {:?}",
                    sort_exprs[0].expr
                );
            }
            other => panic!("expected Sort, got {other:?}"),
        }
    }

    // ---- F12.4: derived tables (subquery in FROM) --------------------

    /// A non-correlated derived table `(SELECT ...) AS d` lowers by recursively
    /// planning the subquery as a child plan and exposing it under the alias.
    /// The top-level Project sits over the recursively-planned subtree.
    #[test]
    fn derived_table_in_from_lowers() {
        let plan = lp("SELECT a FROM (SELECT a FROM t1) AS d");
        // Outer Project over the inner subquery's plan (itself a Project/Scan).
        match plan {
            LogicalPlan::Project { input, .. } => {
                // The recursively-planned subquery is the child; it must carry
                // column `a` in its output schema.
                let schema = input.schema().expect("derived subtree schema");
                assert!(
                    schema.fields.iter().any(|f| f.name == "a"),
                    "derived table must expose column 'a', got {schema:?}"
                );
            }
            other => panic!("expected Project over derived table, got {other:?}"),
        }
    }

    /// A derived table can be referenced through its alias in WHERE.
    #[test]
    fn derived_table_alias_resolves_in_where() {
        let plan = lp("SELECT d.a FROM (SELECT a, b FROM t1) AS d WHERE d.b > 0");
        assert!(
            plan.schema().is_ok(),
            "derived-table query must produce a recomputable schema"
        );
    }

    /// A derived table without an alias is rejected with a precise message
    /// (standard SQL requires the alias so `alias.col` can resolve).
    #[test]
    fn derived_table_without_alias_rejected() {
        let err = parse("SELECT a FROM (SELECT a FROM t1)", &provider())
            .expect_err("unaliased derived table must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("requires an alias"),
            "expected 'requires an alias' message, got: {msg}"
        );
    }

    /// On the raw `parse()` path (which bypasses the engine's LATERAL-apply
    /// detector) a LATERAL derived table stays rejected with a precise message
    /// that points at the engine path. Used as the sole FROM item so the
    /// rejection comes from the derived-table arm itself.
    #[test]
    fn lateral_derived_table_rejected() {
        let err = parse("SELECT a FROM LATERAL (SELECT a FROM t2) AS d", &provider())
            .expect_err("LATERAL derived table must be rejected on the parse() path");
        let msg = format!("{err}");
        assert!(
            msg.contains("LATERAL"),
            "expected LATERAL rejection, got: {msg}"
        );
        // The improved message explains *why* — a correlated LATERAL apply runs
        // host-side in the engine, not via the raw parse() API.
        assert!(
            msg.contains("correlated") && msg.contains("apply"),
            "improved LATERAL message should explain the correlated/apply limitation, got: {msg}"
        );
    }

    /// A derived-table alias with a column list `AS d(x, y)` requires field
    /// renaming we don't implement, so it stays rejected.
    #[test]
    fn derived_table_column_list_alias_rejected() {
        let err = parse("SELECT x FROM (SELECT a FROM t1) AS d(x)", &provider())
            .expect_err("derived-table column-list alias must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("column list"),
            "expected 'column list' message, got: {msg}"
        );
    }
}

#[cfg(test)]
mod setop_coercion_tests {
    //! Tests for common-supertype TYPE COERCION across set-operation branches
    //! (UNION / EXCEPT / INTERSECT). Mismatched-but-compatible column types
    //! (Int32 vs Int64, Int vs Float) are coerced to a common supertype
    //! (mirroring VALUES-row unification); identical schemas are left
    //! unchanged; genuinely incompatible types (Int vs Utf8) stay an error.
    use super::*;
    use crate::plan::logical_plan::{DataType, Field};

    /// Tables of a single column `x` in several types, so set-ops can mix
    /// branches whose only column differs in dtype.
    fn provider() -> MemTableProvider {
        let mk = |dt: DataType| Schema::new(vec![Field::new("x", dt, false)]);
        MemTableProvider::new()
            .with_table("i32t", mk(DataType::Int32))
            .with_table("i64t", mk(DataType::Int64))
            .with_table("f64t", mk(DataType::Float64))
            .with_table("strt", mk(DataType::Utf8))
    }

    /// dtype of the single output column of a successfully-parsed plan.
    fn col0_dtype(sql: &str) -> DataType {
        let plan = parse(sql, &provider())
            .unwrap_or_else(|e| panic!("parse failed for {sql:?}: {e}"));
        plan.schema()
            .unwrap_or_else(|e| panic!("schema recompute failed for {sql:?}: {e}"))
            .fields[0]
            .dtype
    }

    #[test]
    fn union_int32_int64_widens_to_int64() {
        // Int32 ∪ Int64 → Int64 (and the dedup-wrapping UNION still recomputes
        // its schema cleanly through the inserted casting Project).
        assert_eq!(
            col0_dtype("SELECT x FROM i32t UNION SELECT x FROM i64t"),
            DataType::Int64
        );
        assert_eq!(
            col0_dtype("SELECT x FROM i32t UNION ALL SELECT x FROM i64t"),
            DataType::Int64
        );
        // Order-independent: the wider branch on the left also coerces.
        assert_eq!(
            col0_dtype("SELECT x FROM i64t UNION ALL SELECT x FROM i32t"),
            DataType::Int64
        );
    }

    #[test]
    fn union_int_float_widens_to_float() {
        // Int ∪ Float → Float64.
        assert_eq!(
            col0_dtype("SELECT x FROM i32t UNION ALL SELECT x FROM f64t"),
            DataType::Float64
        );
        assert_eq!(
            col0_dtype("SELECT x FROM i64t UNION ALL SELECT x FROM f64t"),
            DataType::Float64
        );
    }

    #[test]
    fn except_intersect_coerce_to_supertype() {
        // EXCEPT / INTERSECT use the same coercion path.
        assert_eq!(
            col0_dtype("SELECT x FROM i32t EXCEPT SELECT x FROM i64t"),
            DataType::Int64
        );
        assert_eq!(
            col0_dtype("SELECT x FROM i32t INTERSECT SELECT x FROM f64t"),
            DataType::Float64
        );
    }

    #[test]
    fn three_way_union_folds_all_branches() {
        // A flattened 3-branch UNION ALL folds Int32, Int64, Float64 → Float64.
        assert_eq!(
            col0_dtype(
                "SELECT x FROM i32t UNION ALL SELECT x FROM i64t \
                 UNION ALL SELECT x FROM f64t"
            ),
            DataType::Float64
        );
    }

    #[test]
    fn identical_schemas_unchanged() {
        // Identical branch types need no coercion and the column type is kept.
        let plan = parse(
            "SELECT x FROM i64t UNION ALL SELECT x FROM i64t",
            &provider(),
        )
        .expect("identical-schema UNION must parse");
        // No casting Project is inserted. Each branch is `SELECT x`, which
        // naturally lowers to a `Project` over a scan; coercion would wrap
        // that in an ADDITIONAL `Project` (so `Project { input: Project }`).
        // For identical schemas, the branch's Project must sit directly on a
        // non-Project source — i.e. it was not re-wrapped.
        if let LogicalPlan::Union { inputs } = &plan {
            for inp in inputs {
                if let LogicalPlan::Project { input, .. } = inp {
                    assert!(
                        !matches!(input.as_ref(), LogicalPlan::Project { .. }),
                        "identical-schema branch must not be re-wrapped in a coercion Project"
                    );
                } else {
                    panic!("expected each UNION branch to be a Project, got {inp:?}");
                }
            }
        } else {
            panic!("expected a top-level Union, got {plan:?}");
        }
        assert_eq!(
            col0_dtype("SELECT x FROM i64t UNION ALL SELECT x FROM i64t"),
            DataType::Int64
        );
    }

    #[test]
    fn incompatible_types_rejected() {
        // Int vs Utf8 is genuinely incompatible: still an Err naming the op.
        for (sql, op) in [
            ("SELECT x FROM i32t UNION SELECT x FROM strt", "UNION"),
            ("SELECT x FROM i32t EXCEPT SELECT x FROM strt", "EXCEPT"),
            ("SELECT x FROM i32t INTERSECT SELECT x FROM strt", "INTERSECT"),
        ] {
            let err = parse(sql, &provider())
                .expect_err("incompatible-type set-op must be rejected");
            let msg = err.to_string();
            assert!(
                msg.contains(op),
                "expected error naming {op}, got: {msg}"
            );
            assert!(
                msg.contains("incompatible"),
                "expected 'incompatible' in error, got: {msg}"
            );
        }
    }

    #[test]
    fn modulo_and_bitwise_lower_for_integers() {
        // R1-modulo-bitwise: % and the bitwise/shift family now lower
        // end-to-end for integer columns. The single output column keeps an
        // integer dtype (Int64 here, since `x` is Int64).
        for sql in [
            "SELECT x % 2 FROM i64t",
            "SELECT x & 1 FROM i64t",
            "SELECT x | 1 FROM i64t",
            "SELECT x ^ 1 FROM i64t",
            "SELECT x << 1 FROM i64t",
            "SELECT x >> 1 FROM i64t",
        ] {
            assert_eq!(
                col0_dtype(sql),
                DataType::Int64,
                "expected Int64 result for {sql:?}"
            );
        }
    }

    #[test]
    fn modulo_and_bitwise_reject_float_operands() {
        // Float operands are rejected with a clear integer-only message. The
        // integer-only type-check fires during schema/dtype resolution, which
        // for a plain projection happens at `lower_physical`; accept the error
        // from whichever layer surfaces it (parse or lowering).
        for sql in [
            "SELECT x % 2 FROM f64t",
            "SELECT x & 1 FROM f64t",
            "SELECT x << 1 FROM f64t",
        ] {
            let err = match parse(sql, &provider()) {
                Ok(plan) => crate::plan::physical_plan::lower(&plan)
                    .expect_err("float operand must be rejected at parse or lowering"),
                Err(e) => e,
            };
            assert!(
                err.to_string().contains("requires integer"),
                "expected integer-operand error for {sql:?}, got: {err}"
            );
        }
    }
}

#[cfg(test)]
mod recursive_cte_tests {
    //! Frontend tests for feature F1 (`WITH RECURSIVE`). These exercise
    //! `plan_recursive_cte` purely host-side (parse → plan shape, schema /
    //! shape rejections); the end-to-end fixpoint *execution* lives in the
    //! engine and is tested there (gpu-gated).
    use super::*;
    use crate::plan::logical_plan::{DataType, Field};

    /// Provider with `edges(src Int64, dst Int64)` for graph-traversal CTEs.
    fn provider() -> MemTableProvider {
        let edges = Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ]);
        MemTableProvider::new().with_table("edges", edges)
    }

    /// Unwrap a planned single-CTE recursive query (the common fast path) or
    /// panic — used by the shape tests below.
    fn single(plan: RecursiveQueryPlan) -> RecursiveCtePlan {
        match plan {
            RecursiveQueryPlan::Single(s) => s,
            RecursiveQueryPlan::Mutual(_) => panic!("expected a single recursive CTE"),
        }
    }

    /// A non-recursive query returns `Ok(None)` so the engine falls through to
    /// the ordinary parse path.
    #[test]
    fn non_recursive_query_returns_none() {
        let got = plan_recursive_cte("SELECT src FROM edges", &provider())
            .expect("non-recursive query must not error here");
        assert!(got.is_none(), "expected None for a non-recursive query");
    }

    /// A simple integer-sequence recursive CTE planning shape. The anchor
    /// seeds from `edges` (the frontend requires a FROM clause, so a bare
    /// `SELECT 1` anchor is not available).
    #[test]
    fn integer_sequence_plan_shape() {
        let sql = "WITH RECURSIVE seq(n) AS (\
                       SELECT src FROM edges WHERE src = 1 \
                       UNION ALL \
                       SELECT n + 1 FROM seq WHERE n < 5\
                   ) SELECT n FROM seq";
        let rec = single(
            plan_recursive_cte(sql, &provider())
                .expect("planning must succeed")
                .expect("must be recognised as a recursive CTE"),
        );
        assert_eq!(rec.name, "seq");
        assert!(rec.all, "UNION ALL must set `all`");
        assert!(!rec.naive, "a single self-reference is linear (not naive)");
        assert_eq!(rec.cte_schema.fields.len(), 1);
        assert_eq!(rec.cte_schema.fields[0].name, "n");
        // The main query's output schema has a single column `n`.
        let schema = rec.schema().expect("schema must type-check");
        assert_eq!(schema.fields.len(), 1);
        assert_eq!(schema.fields[0].name, "n");
        // The recursive term references `seq` as a Scan over the CTE schema.
        let mentions_seq = matches!(&rec.recursive, LogicalPlan::Project { input, .. }
            if matches!(input.as_ref(), LogicalPlan::Filter { input, .. }
                if matches!(input.as_ref(), LogicalPlan::Scan { table, .. } if table == "seq")));
        assert!(mentions_seq, "recursive term must scan the CTE: {:?}", rec.recursive);
    }

    /// Plain `UNION` (distinct) is recognised and clears `all`.
    #[test]
    fn union_distinct_clears_all_flag() {
        let sql = "WITH RECURSIVE r(n) AS (\
                       SELECT src FROM edges WHERE src = 1 \
                       UNION \
                       SELECT n + 1 FROM r WHERE n < 3\
                   ) SELECT n FROM r";
        let rec = single(
            plan_recursive_cte(sql, &provider())
                .expect("planning must succeed")
                .expect("recursive CTE"),
        );
        assert!(!rec.all, "plain UNION must clear `all` (dedup)");
    }

    /// An anchor/recursive column-count mismatch is rejected cleanly.
    #[test]
    fn anchor_recursive_column_count_mismatch_rejected() {
        // Anchor produces 1 column; recursive term produces 2.
        let sql = "WITH RECURSIVE r(n) AS (\
                       SELECT src FROM edges WHERE src = 1 \
                       UNION ALL \
                       SELECT n + 1, n FROM r WHERE n < 3\
                   ) SELECT n FROM r";
        let err = plan_recursive_cte(sql, &provider())
            .expect_err("anchor/recursive arity mismatch must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("incompatible schemas") || msg.contains("columns"),
            "expected schema-mismatch message, got: {msg}"
        );
    }

    /// A column-list alias whose arity differs from the anchor's is rejected.
    #[test]
    fn column_list_alias_arity_mismatch_rejected() {
        // Anchor produces 1 column but the alias names 2.
        let sql = "WITH RECURSIVE r(a, b) AS (\
                       SELECT src FROM edges WHERE src = 1 \
                       UNION ALL \
                       SELECT a + 1 FROM r WHERE a < 3\
                   ) SELECT a FROM r";
        let err = plan_recursive_cte(sql, &provider())
            .expect_err("column-list alias arity mismatch must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("column-list alias"),
            "expected column-list alias message, got: {msg}"
        );
    }

    /// A recursive term that never references the CTE is rejected (it would
    /// never recurse).
    #[test]
    fn recursive_term_without_self_reference_rejected() {
        let sql = "WITH RECURSIVE r(n) AS (\
                       SELECT src FROM edges WHERE src = 1 \
                       UNION ALL \
                       SELECT dst FROM edges\
                   ) SELECT n FROM r";
        let err = plan_recursive_cte(sql, &provider())
            .expect_err("a term that never references the CTE must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("does not reference"),
            "expected 'does not reference' message, got: {msg}"
        );
    }

    /// Mutual recursion (two CTEs in one `WITH RECURSIVE`) now plans as a
    /// `Mutual` system rather than being rejected.
    #[test]
    fn mutual_recursion_plans_as_system() {
        let sql = "WITH RECURSIVE \
                     a(n) AS (SELECT src FROM edges WHERE src=1 UNION ALL SELECT m+1 FROM b WHERE m<3), \
                     b(m) AS (SELECT dst FROM edges WHERE dst=2 UNION ALL SELECT n+1 FROM a WHERE n<3) \
                   SELECT n FROM a";
        let plan = plan_recursive_cte(sql, &provider())
            .expect("planning must succeed")
            .expect("recursive CTE");
        let sys = match plan {
            RecursiveQueryPlan::Mutual(m) => m,
            RecursiveQueryPlan::Single(_) => panic!("expected a mutual system"),
        };
        assert_eq!(sys.ctes.len(), 2, "two CTEs in the system");
        assert_eq!(sys.ctes[0].name, "a");
        assert_eq!(sys.ctes[1].name, "b");
        // Both members are recursive (each references the other).
        assert!(sys.ctes[0].recursive.is_some());
        assert!(sys.ctes[1].recursive.is_some());
        // The main query type-checks.
        let schema = sys.schema().expect("main schema must type-check");
        assert_eq!(schema.fields.len(), 1);
    }

    /// A multi-CTE `WITH RECURSIVE` whose anchor of a recursive member
    /// references a recursive CTE is rejected (the anchor must be a seed).
    #[test]
    fn mutual_recursion_recursive_anchor_rejected() {
        let sql = "WITH RECURSIVE \
                     a(n) AS (SELECT n FROM b UNION ALL SELECT m+1 FROM b WHERE m<3), \
                     b(m) AS (SELECT dst FROM edges UNION ALL SELECT n+1 FROM a WHERE n<3) \
                   SELECT n FROM a";
        let err = plan_recursive_cte(sql, &provider())
            .expect_err("a recursive anchor in a mutual system must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("anchor") && msg.contains("non-recursive"),
            "expected recursive-anchor message, got: {msg}"
        );
    }

    /// A multi-CTE `WITH RECURSIVE` list where NO member references any CTE in
    /// the list is rejected (it would never recurse).
    #[test]
    fn mutual_recursion_no_recursive_member_rejected() {
        let sql = "WITH RECURSIVE \
                     a(n) AS (SELECT src FROM edges), \
                     b(m) AS (SELECT dst FROM edges) \
                   SELECT n FROM a";
        let err = plan_recursive_cte(sql, &provider())
            .expect_err("a non-recursive multi-CTE list must be rejected here");
        let msg = format!("{err}");
        assert!(
            msg.contains("no recursive member"),
            "expected 'no recursive member' message, got: {msg}"
        );
    }

    /// A body that is not a top-level UNION is rejected.
    #[test]
    fn non_union_body_rejected() {
        let sql = "WITH RECURSIVE r(n) AS (SELECT src FROM edges) SELECT n FROM r";
        let err = plan_recursive_cte(sql, &provider())
            .expect_err("a non-UNION recursive body must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("UNION"),
            "expected UNION-body message, got: {msg}"
        );
    }

    /// More than one self-reference (non-linear recursion, a self-join) now
    /// plans as a single CTE with the `naive` flag set (the engine binds the
    /// full accumulated relation each iteration).
    #[test]
    fn non_linear_self_join_sets_naive() {
        // Two references to `r` in the recursive term's FROM (self-join).
        let sql = "WITH RECURSIVE r(x, y) AS (\
                       SELECT src, dst FROM edges \
                       UNION ALL \
                       SELECT r1.x, r2.y FROM r AS r1 JOIN r AS r2 ON r1.y = r2.x\
                   ) SELECT x, y FROM r";
        let rec = single(
            plan_recursive_cte(sql, &provider())
                .expect("non-linear recursion must plan")
                .expect("recursive CTE"),
        );
        assert!(rec.naive, "a self-join recursive term must set `naive`");
        assert!(rec.all, "UNION ALL is preserved");
        assert_eq!(rec.cte_schema.fields.len(), 2);
    }

    /// A graph-traversal recursive CTE over a real base table plans cleanly and
    /// the CTE schema follows the anchor's column dtypes.
    #[test]
    fn graph_traversal_plan_shape() {
        // Recursive term joins the base table with the recursive relation via
        // explicit JOIN (the engine does not support comma/cross-join FROM).
        let sql = "WITH RECURSIVE reach(node) AS (\
                       SELECT src FROM edges WHERE src = 1 \
                       UNION \
                       SELECT edges.dst FROM edges JOIN reach ON edges.src = reach.node\
                   ) SELECT node FROM reach";
        let rec = single(
            plan_recursive_cte(sql, &provider())
                .expect("planning must succeed")
                .expect("recursive CTE"),
        );
        assert_eq!(rec.name, "reach");
        assert!(!rec.all);
        assert!(!rec.naive, "a single self-reference is linear");
        assert_eq!(rec.cte_schema.fields.len(), 1);
        assert_eq!(rec.cte_schema.fields[0].name, "node");
        assert_eq!(rec.cte_schema.fields[0].dtype, DataType::Int64);
    }

    /// A mutual system may mix a recursive member with a plain (seeded-once)
    /// member; the plain member has `recursive == None`.
    #[test]
    fn mutual_recursion_allows_plain_member() {
        let sql = "WITH RECURSIVE \
                     seed(s) AS (SELECT src FROM edges WHERE src = 1), \
                     walk(n) AS (SELECT s FROM seed UNION ALL SELECT n + 1 FROM walk WHERE n < 3) \
                   SELECT n FROM walk";
        let plan = plan_recursive_cte(sql, &provider())
            .expect("planning must succeed")
            .expect("recursive CTE");
        let sys = match plan {
            RecursiveQueryPlan::Mutual(m) => m,
            RecursiveQueryPlan::Single(_) => panic!("expected a mutual system"),
        };
        assert_eq!(sys.ctes.len(), 2);
        // `seed` is non-recursive; `walk` is recursive and references `seed`.
        let seed = sys.ctes.iter().find(|c| c.name == "seed").unwrap();
        let walk = sys.ctes.iter().find(|c| c.name == "walk").unwrap();
        assert!(seed.recursive.is_none(), "plain member has no recursive term");
        assert!(walk.recursive.is_some(), "walk recurses");
    }
}

#[cfg(test)]
mod count_distinct_groupby_tests {
    //! Frontend detection tests for feature F3-finish
    //! (`COUNT(DISTINCT col)` with `GROUP BY`). These exercise
    //! [`plan_count_distinct_groupby`] purely host-side: which shapes return
    //! `Some(descriptor)` and which fall through (`None`) to the ordinary
    //! pipeline. The host-execution correctness lives in the engine tests.
    use super::*;
    use crate::plan::logical_plan::{DataType, Field};

    /// `sales(region Int32, customer Int64, amount Float64)`.
    fn provider() -> MemTableProvider {
        let sales = Schema::new(vec![
            Field::new("region", DataType::Int32, true),
            Field::new("customer", DataType::Int64, true),
            Field::new("amount", DataType::Float64, true),
        ]);
        MemTableProvider::new().with_table("sales", sales)
    }

    fn detect(sql: &str) -> Option<CountDistinctGroupByPlan> {
        plan_count_distinct_groupby(sql, &provider())
            .unwrap_or_else(|e| panic!("detection errored for {sql:?}: {e}"))
    }

    #[test]
    fn supported_shape_returns_some() {
        let cd = detect("SELECT region, COUNT(DISTINCT customer) FROM sales GROUP BY region")
            .expect("sole COUNT(DISTINCT) + GROUP BY must be recognised");
        assert_eq!(cd.group_key_names, vec!["region".to_string()]);
        // Default alias is the canonical aggregate output name.
        assert_eq!(
            cd.count_alias,
            aggregate_output_name(&AggregateExpr::Count(Expr::Literal(Literal::Int64(1))))
        );
        // Result schema: group key + Int64 count (non-nullable).
        assert_eq!(cd.result_schema.fields.len(), 2);
        assert_eq!(cd.result_schema.fields[0].name, "region");
        assert_eq!(cd.result_schema.fields[1].dtype, DataType::Int64);
        assert!(!cd.result_schema.fields[1].nullable);
        // No HAVING/ORDER BY/LIMIT ⇒ no post-plan.
        assert!(cd.post.is_none());
        // Base output is [group_key, distinct_col].
        let base_schema = cd.base.schema().expect("base schema");
        assert_eq!(base_schema.fields.len(), 2);
    }

    #[test]
    fn supported_with_alias_uses_alias() {
        let cd = detect(
            "SELECT region, COUNT(DISTINCT customer) AS uniq FROM sales GROUP BY region",
        )
        .expect("aliased count must be recognised");
        assert_eq!(cd.count_alias, "uniq");
        assert_eq!(cd.result_schema.fields[1].name, "uniq");
    }

    #[test]
    fn supported_multi_key() {
        let cd = detect(
            "SELECT region, amount, COUNT(DISTINCT customer) FROM sales \
             GROUP BY region, amount",
        )
        .expect("multi-key shape must be recognised");
        assert_eq!(cd.group_key_names, vec!["region".to_string(), "amount".to_string()]);
        assert_eq!(cd.result_schema.fields.len(), 3);
    }

    #[test]
    fn supported_with_where_and_having_and_order_limit() {
        let cd = detect(
            "SELECT region, COUNT(DISTINCT customer) AS c FROM sales \
             WHERE amount > 0 GROUP BY region HAVING COUNT(DISTINCT customer) > 1 \
             ORDER BY c DESC LIMIT 5",
        )
        .expect("WHERE+HAVING+ORDER BY+LIMIT shape must be recognised");
        // WHERE is folded into the base plan (a Filter under the projection).
        // HAVING/ORDER BY/LIMIT produce a post-plan.
        let post = cd.post.expect("HAVING/ORDER BY/LIMIT ⇒ post-plan present");
        // Outermost post node is the Limit.
        assert!(matches!(post, LogicalPlan::Limit { .. }), "post root = Limit, got {post:?}");
    }

    #[test]
    fn count_distinct_no_groupby_falls_through() {
        // No GROUP BY ⇒ not our shape (the no-GROUP-BY COUNT(DISTINCT) path is
        // handled by the ordinary pipeline).
        assert!(detect("SELECT COUNT(DISTINCT customer) FROM sales").is_none());
    }

    #[test]
    fn ordinary_groupby_aggregate_falls_through() {
        // GROUP BY with a non-distinct aggregate ⇒ ordinary pipeline.
        assert!(detect("SELECT region, COUNT(customer) FROM sales GROUP BY region").is_none());
        assert!(detect("SELECT region, SUM(amount) FROM sales GROUP BY region").is_none());
    }

    #[test]
    fn count_distinct_with_other_aggregate_falls_through() {
        // COUNT(DISTINCT) alongside another aggregate is an unsupported mix:
        // fall through so the ordinary path rejects it precisely.
        assert!(detect(
            "SELECT region, COUNT(DISTINCT customer), SUM(amount) FROM sales GROUP BY region"
        )
        .is_none());
    }

    #[test]
    fn two_count_distincts_fall_through() {
        assert!(detect(
            "SELECT region, COUNT(DISTINCT customer), COUNT(DISTINCT amount) \
             FROM sales GROUP BY region"
        )
        .is_none());
    }

    #[test]
    fn super_aggregate_falls_through() {
        // ROLLUP/CUBE/GROUPING SETS with COUNT(DISTINCT) is not our shape; it
        // falls through to the ordinary path (which rejects it precisely).
        assert!(detect(
            "SELECT region, COUNT(DISTINCT customer) FROM sales GROUP BY ROLLUP(region)"
        )
        .is_none());
    }

    #[test]
    fn select_distinct_falls_through() {
        assert!(detect(
            "SELECT DISTINCT region, COUNT(DISTINCT customer) FROM sales GROUP BY region"
        )
        .is_none());
    }

    #[test]
    fn set_op_and_cte_fall_through() {
        assert!(detect(
            "SELECT region, COUNT(DISTINCT customer) FROM sales GROUP BY region \
             UNION ALL SELECT region, COUNT(DISTINCT customer) FROM sales GROUP BY region"
        )
        .is_none());
        assert!(detect(
            "WITH s AS (SELECT * FROM sales) \
             SELECT region, COUNT(DISTINCT customer) FROM s GROUP BY region"
        )
        .is_none());
    }
}

#[cfg(test)]
mod multi_agg_groupby_tests {
    //! Frontend detection tests for the *generalized* COUNT(DISTINCT) + GROUP
    //! BY shape (feature F3-finish, generalized): multiple distinct counts
    //! and/or a mix with plain aggregates. These exercise
    //! [`plan_multi_agg_groupby`] host-side (which shapes return
    //! `Some(descriptor)`, which fall through, which error). Host-execution
    //! correctness lives in the engine tests.
    use super::*;
    use crate::plan::logical_plan::{DataType, Field};

    fn provider() -> MemTableProvider {
        let sales = Schema::new(vec![
            Field::new("region", DataType::Int32, true),
            Field::new("customer", DataType::Int64, true),
            Field::new("product", DataType::Int64, true),
            Field::new("amount", DataType::Float64, true),
        ]);
        MemTableProvider::new().with_table("sales", sales)
    }

    fn detect(sql: &str) -> Option<MultiAggGroupByPlan> {
        plan_multi_agg_groupby(sql, &provider())
            .unwrap_or_else(|e| panic!("detection errored for {sql:?}: {e}"))
    }

    /// Two COUNT(DISTINCT) over different columns is recognised by the
    /// generalized detector (the single-CD path declines it).
    #[test]
    fn two_count_distincts_recognised() {
        let cd = detect(
            "SELECT region, COUNT(DISTINCT customer), COUNT(DISTINCT product) \
             FROM sales GROUP BY region",
        )
        .expect("two COUNT(DISTINCT) must be recognised");
        assert_eq!(cd.n_keys, 1);
        assert_eq!(cd.group_key_names, vec!["region".to_string()]);
        assert_eq!(cd.aggs.len(), 2);
        assert!(matches!(cd.aggs[0], CdAgg::CountDistinct { .. }));
        assert!(matches!(cd.aggs[1], CdAgg::CountDistinct { .. }));
        // Result: region + 2 Int64 counts.
        assert_eq!(cd.result_schema.fields.len(), 3);
        assert_eq!(cd.result_schema.fields[1].dtype, DataType::Int64);
        // Base output is [region, customer, product] (keys + agg inputs).
        assert_eq!(cd.base.schema().unwrap().fields.len(), 3);
    }

    /// COUNT(DISTINCT) mixed with plain aggregates (SUM / COUNT(*)).
    #[test]
    fn mixed_distinct_and_plain_recognised() {
        let cd = detect(
            "SELECT region, COUNT(DISTINCT customer), SUM(amount), COUNT(*) \
             FROM sales GROUP BY region",
        )
        .expect("mixed COUNT(DISTINCT)+SUM+COUNT(*) must be recognised");
        assert_eq!(cd.aggs.len(), 3);
        assert!(matches!(cd.aggs[0], CdAgg::CountDistinct { .. }));
        assert!(matches!(cd.aggs[1], CdAgg::Sum { .. }));
        assert!(matches!(cd.aggs[2], CdAgg::CountStar { .. }));
        // SUM(amount) is Float64 (preserves input dtype), nullable.
        let sum_field = &cd.result_schema.fields[2];
        assert_eq!(sum_field.dtype, DataType::Float64);
        assert!(sum_field.nullable);
        // COUNT(*) is non-nullable Int64.
        assert_eq!(cd.result_schema.fields[3].dtype, DataType::Int64);
        assert!(!cd.result_schema.fields[3].nullable);
    }

    /// The sole-single-COUNT(DISTINCT) case stays on the dedicated path: the
    /// generalized detector declines it (`None`).
    #[test]
    fn single_sole_count_distinct_defers() {
        assert!(detect(
            "SELECT region, COUNT(DISTINCT customer) FROM sales GROUP BY region"
        )
        .is_none());
    }

    /// An ordinary GROUP BY with no COUNT(DISTINCT) is not ours.
    #[test]
    fn no_count_distinct_falls_through() {
        assert!(detect("SELECT region, SUM(amount) FROM sales GROUP BY region").is_none());
        assert!(detect(
            "SELECT region, SUM(amount), COUNT(*) FROM sales GROUP BY region"
        )
        .is_none());
    }

    /// A non-host-computable aggregate alongside COUNT(DISTINCT) is a precise
    /// error (VAR_POP under GROUP BY is unsupported here).
    #[test]
    fn unsupported_aggregate_errors() {
        let err = plan_multi_agg_groupby(
            "SELECT region, COUNT(DISTINCT customer), VAR_POP(amount) \
             FROM sales GROUP BY region",
            &provider(),
        )
        .expect_err("VAR_POP under GROUP BY must be rejected precisely");
        let msg = format!("{err}");
        assert!(msg.contains("var_pop") || msg.contains("VAR_POP") || msg.contains("unsupported"),
            "expected an unsupported-aggregate message, got: {msg}");
    }

    /// HAVING / ORDER BY / LIMIT produce a post-plan (outermost = Limit).
    #[test]
    fn having_order_limit_builds_post_plan() {
        let cd = detect(
            "SELECT region, COUNT(DISTINCT customer) AS c, SUM(amount) AS s \
             FROM sales GROUP BY region HAVING COUNT(DISTINCT customer) > 1 \
             ORDER BY c DESC LIMIT 5",
        )
        .expect("must be recognised");
        let post = cd.post.expect("HAVING/ORDER BY/LIMIT ⇒ post-plan");
        assert!(matches!(post, LogicalPlan::Limit { .. }), "post root = Limit, got {post:?}");
    }

    /// A super-aggregate / SELECT DISTINCT / set-op / CTE falls through.
    #[test]
    fn non_plain_groupby_shapes_fall_through() {
        assert!(detect(
            "SELECT region, COUNT(DISTINCT customer), SUM(amount) \
             FROM sales GROUP BY ROLLUP(region)"
        )
        .is_none());
        assert!(detect(
            "SELECT DISTINCT region, COUNT(DISTINCT customer), SUM(amount) \
             FROM sales GROUP BY region"
        )
        .is_none());
    }
}

#[cfg(test)]
mod clause_acceptance_tests {
    //! Host-side lowering tests for the query-clause acceptance features:
    //! FETCH / TOP (→ Limit), FOR UPDATE/SHARE (accepted no-op), PREWHERE
    //! (folded into the WHERE Filter), named WINDOW (`OVER w` → inline spec),
    //! and QUALIFY (→ Filter over the Window projection). Plus the sharpened
    //! out-of-scope rejection messages (SUM(temporal) / CONNECT BY / TVF /
    //! PERCENT / WITH TIES). All pure-frontend — no GPU, no `Engine::new()`.
    use super::*;

    /// A provider with an int key, an int value, and a category column —
    /// enough to exercise WHERE / window / qualify lowering.
    fn provider() -> MemTableProvider {
        let t = Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new("v", DataType::Int64, false),
            Field::new("g", DataType::Int32, false),
        ]);
        MemTableProvider::new().with_table("t", t)
    }

    fn plan(sql: &str) -> LogicalPlan {
        parse(sql, &provider()).expect("plan must lower cleanly")
    }

    fn err(sql: &str) -> String {
        format!("{}", parse(sql, &provider()).expect_err("must be rejected"))
    }

    // ---- FETCH → Limit ----------------------------------------------------

    #[test]
    fn fetch_first_lowers_to_limit() {
        match plan("SELECT k FROM t FETCH FIRST 5 ROWS ONLY") {
            LogicalPlan::Limit { limit, offset, .. } => {
                assert_eq!(limit, 5);
                assert_eq!(offset, 0);
            }
            other => panic!("FETCH FIRST must lower to Limit, got {other:?}"),
        }
    }

    #[test]
    fn fetch_next_is_synonym_for_fetch_first() {
        match plan("SELECT k FROM t FETCH NEXT 3 ROWS ONLY") {
            LogicalPlan::Limit { limit, .. } => assert_eq!(limit, 3),
            other => panic!("FETCH NEXT must lower to Limit, got {other:?}"),
        }
    }

    #[test]
    fn fetch_composes_with_offset_and_order_by() {
        // ORDER BY must sit *below* the Limit (limit applies after the sort).
        match plan("SELECT k FROM t ORDER BY k OFFSET 2 ROWS FETCH NEXT 4 ROWS ONLY") {
            LogicalPlan::Limit {
                input,
                limit,
                offset,
            } => {
                assert_eq!(limit, 4);
                assert_eq!(offset, 2);
                assert!(
                    matches!(*input, LogicalPlan::Sort { .. }),
                    "Limit must wrap the Sort so the limit applies after ORDER BY"
                );
            }
            other => panic!("expected Limit over Sort, got {other:?}"),
        }
    }

    #[test]
    fn fetch_with_ties_rejected_cleanly() {
        let msg = err("SELECT k FROM t ORDER BY k FETCH FIRST 5 ROWS WITH TIES");
        assert!(msg.contains("WITH TIES"), "message must name WITH TIES: {msg}");
    }

    #[test]
    fn fetch_percent_rejected_cleanly() {
        let msg = err("SELECT k FROM t FETCH FIRST 5 PERCENT ROWS ONLY");
        assert!(msg.contains("PERCENT"), "message must name PERCENT: {msg}");
    }

    #[test]
    fn fetch_plus_limit_is_ambiguous() {
        let msg = err("SELECT k FROM t LIMIT 3 FETCH FIRST 5 ROWS ONLY");
        assert!(
            msg.contains("ambiguous") || msg.contains("not both"),
            "LIMIT + FETCH must be rejected as ambiguous: {msg}"
        );
    }

    // ---- TOP → Limit ------------------------------------------------------

    #[test]
    fn top_lowers_to_limit() {
        match plan("SELECT TOP 7 k FROM t") {
            LogicalPlan::Limit { limit, offset, .. } => {
                assert_eq!(limit, 7);
                assert_eq!(offset, 0);
            }
            other => panic!("TOP must lower to Limit, got {other:?}"),
        }
    }

    #[test]
    fn top_applies_after_order_by() {
        match plan("SELECT TOP 2 k FROM t ORDER BY k") {
            LogicalPlan::Limit { input, limit, .. } => {
                assert_eq!(limit, 2);
                assert!(
                    matches!(*input, LogicalPlan::Sort { .. }),
                    "TOP must wrap the Sort (T-SQL applies TOP after ORDER BY)"
                );
            }
            other => panic!("expected Limit over Sort, got {other:?}"),
        }
    }

    #[test]
    fn top_percent_rejected_cleanly() {
        let msg = err("SELECT TOP 10 PERCENT k FROM t");
        assert!(msg.contains("PERCENT"), "message must name PERCENT: {msg}");
    }

    #[test]
    fn top_with_ties_rejected_cleanly() {
        let msg = err("SELECT TOP 3 WITH TIES k FROM t ORDER BY k");
        assert!(msg.contains("WITH TIES"), "message must name WITH TIES: {msg}");
    }

    #[test]
    fn top_plus_limit_is_ambiguous() {
        let msg = err("SELECT TOP 3 k FROM t LIMIT 5");
        assert!(
            msg.contains("ambiguous") || msg.contains("not both"),
            "TOP + LIMIT must be rejected as ambiguous: {msg}"
        );
    }

    // ---- FOR UPDATE / FOR SHARE: accepted no-op ---------------------------

    #[test]
    fn for_update_is_accepted_and_changes_nothing() {
        // The locked query must lower to exactly the same plan as the unlocked
        // one — row locking is a no-op for this read-only engine.
        let bare = plan("SELECT k FROM t WHERE v > 1");
        let locked = plan("SELECT k FROM t WHERE v > 1 FOR UPDATE");
        assert_eq!(
            format!("{bare:?}"),
            format!("{locked:?}"),
            "FOR UPDATE must not alter the lowered plan"
        );
    }

    #[test]
    fn for_share_is_accepted_and_changes_nothing() {
        let bare = plan("SELECT k FROM t");
        let locked = plan("SELECT k FROM t FOR SHARE");
        assert_eq!(format!("{bare:?}"), format!("{locked:?}"));
    }

    // ---- PREWHERE folded into the WHERE Filter ----------------------------

    /// PREWHERE alone becomes the Filter predicate.
    #[test]
    fn prewhere_alone_becomes_the_filter() {
        match plan("SELECT k FROM t PREWHERE v > 1") {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::Filter { predicate, .. } => {
                    // A single comparison, not an AND.
                    assert!(
                        matches!(
                            predicate,
                            Expr::Binary { op: BinaryOp::Gt, .. }
                        ),
                        "PREWHERE-only predicate must be the comparison itself"
                    );
                }
                other => panic!("expected Filter under Project, got {other:?}"),
            },
            other => panic!("expected Project root, got {other:?}"),
        }
    }

    /// PREWHERE + WHERE AND together into a single Filter.
    #[test]
    fn prewhere_and_where_are_anded() {
        match plan("SELECT k FROM t PREWHERE v > 1 WHERE k < 10") {
            LogicalPlan::Project { input, .. } => match *input {
                LogicalPlan::Filter { predicate, .. } => {
                    assert!(
                        matches!(
                            predicate,
                            Expr::Binary { op: BinaryOp::And, .. }
                        ),
                        "PREWHERE AND WHERE must fold into a conjunction, got {predicate:?}"
                    );
                }
                other => panic!("expected Filter under Project, got {other:?}"),
            },
            other => panic!("expected Project root, got {other:?}"),
        }
    }

    // ---- Named WINDOW: `OVER w` resolves to the inline spec ---------------

    /// `OVER w` must lower to the *identical* plan as the equivalent inline
    /// `OVER (PARTITION BY ... ORDER BY ...)`.
    #[test]
    fn named_window_resolves_to_inline_spec() {
        let named = plan(
            "SELECT ROW_NUMBER() OVER w AS rn FROM t \
             WINDOW w AS (PARTITION BY g ORDER BY v)",
        );
        let inline = plan(
            "SELECT ROW_NUMBER() OVER (PARTITION BY g ORDER BY v) AS rn FROM t",
        );
        assert_eq!(
            format!("{named:?}"),
            format!("{inline:?}"),
            "OVER w must lower identically to the inline window spec"
        );
    }

    /// The `OVER (w ORDER BY ...)` extension form: base supplies PARTITION BY,
    /// inline adds the ORDER BY.
    #[test]
    fn named_window_order_by_extension() {
        let extended = plan(
            "SELECT ROW_NUMBER() OVER (w ORDER BY v) AS rn FROM t \
             WINDOW w AS (PARTITION BY g)",
        );
        let inline = plan(
            "SELECT ROW_NUMBER() OVER (PARTITION BY g ORDER BY v) AS rn FROM t",
        );
        assert_eq!(format!("{extended:?}"), format!("{inline:?}"));
    }

    /// Re-stating PARTITION BY in the extension form is rejected precisely.
    #[test]
    fn named_window_extension_repartition_rejected() {
        let msg = err(
            "SELECT ROW_NUMBER() OVER (w PARTITION BY k) AS rn FROM t \
             WINDOW w AS (PARTITION BY g)",
        );
        assert!(
            msg.contains("PARTITION BY"),
            "extension may not re-state PARTITION BY: {msg}"
        );
    }

    /// An `OVER w` naming an undefined window is a clean error.
    #[test]
    fn undefined_named_window_rejected() {
        let msg = err("SELECT ROW_NUMBER() OVER nope AS rn FROM t");
        assert!(
            msg.contains("not defined") || msg.contains("WINDOW"),
            "undefined window ref must error: {msg}"
        );
    }

    // ---- QUALIFY → Filter over the Window projection ----------------------

    /// QUALIFY referencing a window function already in the SELECT list:
    /// the Filter sits above the Window, below the final Project.
    #[test]
    fn qualify_lowers_to_filter_over_window() {
        // ROW_NUMBER is in the SELECT list; QUALIFY filters on its column.
        match plan(
            "SELECT k, ROW_NUMBER() OVER (PARTITION BY g ORDER BY v) AS rn \
             FROM t QUALIFY rn = 1",
        ) {
            LogicalPlan::Project { input, .. } => {
                assert!(
                    matches!(*input, LogicalPlan::Filter { .. }),
                    "QUALIFY must place a Filter directly under the Project"
                );
                if let LogicalPlan::Filter { input, .. } = *input {
                    assert!(
                        matches!(*input, LogicalPlan::Window { .. }),
                        "the QUALIFY Filter must sit over the Window node"
                    );
                }
            }
            other => panic!("expected Project root, got {other:?}"),
        }
    }

    /// QUALIFY referencing a window function NOT in the SELECT list: the
    /// helper window column is materialized for the filter and the final
    /// Project restores only the SELECT-list columns.
    #[test]
    fn qualify_materializes_hidden_window_then_filters() {
        match plan(
            "SELECT k FROM t \
             QUALIFY ROW_NUMBER() OVER (PARTITION BY g ORDER BY v) = 1",
        ) {
            LogicalPlan::Project { input, exprs } => {
                // The Project restores exactly the SELECT list (one column).
                assert_eq!(exprs.len(), 1, "output schema must be the SELECT list only");
                assert!(matches!(*input, LogicalPlan::Filter { .. }));
                if let LogicalPlan::Filter { input, .. } = *input {
                    assert!(
                        matches!(*input, LogicalPlan::Window { .. }),
                        "hidden window column must be produced by a Window node"
                    );
                }
            }
            other => panic!("expected Project root, got {other:?}"),
        }
    }

    // ---- Sharpened out-of-scope rejection messages ------------------------

    #[test]
    fn sum_over_temporal_is_rejected_by_design() {
        let p = MemTableProvider::new().with_table(
            "d",
            Schema::new(vec![Field::new("dt", DataType::Date32, false)]),
        );
        let msg = format!(
            "{}",
            parse("SELECT SUM(dt) FROM d", &p).expect_err("SUM(Date32) must be rejected")
        );
        assert!(
            msg.contains("undefined in SQL") && msg.contains("by design"),
            "SUM(temporal) message must read as an intentional limitation: {msg}"
        );
    }

    #[test]
    fn connect_by_rejection_explains_recursion() {
        let msg = err(
            "SELECT k FROM t START WITH k = 1 CONNECT BY PRIOR k = v",
        );
        assert!(
            msg.contains("CONNECT BY") && msg.contains("RECURSIVE"),
            "CONNECT BY message must point at WITH RECURSIVE: {msg}"
        );
    }

    #[test]
    fn tvf_rejection_explains_mechanism() {
        // A function-as-table-source in FROM.
        let msg = err("SELECT * FROM generate_series(1, 10)");
        assert!(
            msg.contains("table-valued function") && msg.contains("function-as-table-source"),
            "TVF message must explain the missing mechanism: {msg}"
        );
    }
}

#[cfg(test)]
mod values_and_distinct_on_tests {
    //! Host-side tests for the VALUES row source and DISTINCT ON detector
    //! (feature VALUES / DISTINCT ON). No GPU / engine; a standalone
    //! [`MemTableProvider`] is used where a provider is needed.
    use super::*;
    use crate::plan::logical_plan::{DataType, Field};

    fn provider() -> MemTableProvider {
        MemTableProvider::new().with_table(
            "t",
            Schema::new(vec![
                Field::new("a", DataType::Int64, false),
                Field::new("b", DataType::Utf8, true),
            ]),
        )
    }

    fn values_plan(sql: &str) -> ValuesQueryPlan {
        plan_values_query(sql, &provider())
            .expect("planning must not error")
            .expect("must be a VALUES query")
    }

    fn values_err(sql: &str) -> String {
        match plan_values_query(sql, &provider()) {
            Err(e) => e.to_string(),
            Ok(other) => panic!("expected a VALUES error, got {other:?}"),
        }
    }

    // ---- VALUES type inference -------------------------------------------

    #[test]
    fn bare_values_basic_schema_and_names() {
        let vp = values_plan("VALUES (1, 'a'), (2, 'b')");
        assert!(vp.post.is_none());
        let f = &vp.relation.schema.fields;
        assert_eq!(f.len(), 2);
        // Default Postgres-style names.
        assert_eq!(f[0].name, "column1");
        assert_eq!(f[1].name, "column2");
        assert_eq!(f[0].dtype, DataType::Int64);
        assert_eq!(f[1].dtype, DataType::Utf8);
        assert_eq!(vp.relation.rows.len(), 2);
    }

    #[test]
    fn int32_int64_promotion_widens_to_int64() {
        // 5_000_000_000 does not fit i32, so it is Int64; the other row's small
        // int folds to Int64 too — common type Int64.
        let vp = values_plan("VALUES (1), (5000000000)");
        assert_eq!(vp.relation.schema.fields[0].dtype, DataType::Int64);
        // Both cells coerced to Int64.
        assert!(matches!(vp.relation.rows[0][0], Literal::Int64(1)));
        assert!(matches!(vp.relation.rows[1][0], Literal::Int64(5_000_000_000)));
    }

    #[test]
    fn null_takes_type_from_other_rows_and_marks_nullable() {
        let vp = values_plan("VALUES (1), (NULL), (3)");
        let field = &vp.relation.schema.fields[0];
        assert_eq!(field.dtype, DataType::Int64);
        assert!(field.nullable, "a column with a NULL row must be nullable");
        assert!(matches!(vp.relation.rows[1][0], Literal::Null));
    }

    #[test]
    fn all_null_column_defaults_to_nullable_int64() {
        let vp = values_plan("VALUES (NULL), (NULL)");
        let field = &vp.relation.schema.fields[0];
        assert_eq!(field.dtype, DataType::Int64, "all-NULL defaults to Int64");
        assert!(field.nullable);
    }

    #[test]
    fn incompatible_types_rejected() {
        let msg = values_err("VALUES (1), ('a')");
        assert!(
            msg.contains("incompatible"),
            "int-vs-utf8 mix must be rejected: {msg}"
        );
    }

    #[test]
    fn ragged_rows_rejected() {
        let msg = values_err("VALUES (1, 2), (3)");
        assert!(msg.contains("ragged"), "ragged rows must be rejected: {msg}");
    }

    #[test]
    fn row_cap_enforced() {
        // Pure cap check — no process-global env mutation, so this never races
        // the other parallel VALUES tests that read the cap.
        let msg = enforce_values_row_cap(3, 2)
            .expect_err("cap must be exceeded")
            .to_string();
        assert!(msg.contains("cap"), "row-cap error must mention the cap: {msg}");
    }

    #[test]
    fn int_to_float_promotion() {
        let vp = values_plan("VALUES (1), (2.5)");
        assert_eq!(vp.relation.schema.fields[0].dtype, DataType::Float64);
        assert!(matches!(vp.relation.rows[0][0], Literal::Float64(_)));
    }

    // ---- FROM (VALUES ...) AS t(a, b) ------------------------------------

    #[test]
    fn from_values_with_column_aliases_resolves_columns() {
        let vp = values_plan("SELECT x, y FROM (VALUES (1, 'a'), (2, 'b')) AS v(x, y)");
        assert!(vp.post.is_some(), "FROM form has a post plan");
        assert_eq!(vp.bind_name, "v");
        // The relation schema carries the alias column names.
        assert_eq!(vp.relation.schema.fields[0].name, "x");
        assert_eq!(vp.relation.schema.fields[1].name, "y");
        // The outer query's output schema is [x, y].
        let out = vp.schema().expect("schema");
        assert_eq!(out.fields[0].name, "x");
        assert_eq!(out.fields[1].name, "y");
    }

    #[test]
    fn from_values_filter_over_relation() {
        // A WHERE over the VALUES relation lowers through the ordinary pipeline.
        let vp = values_plan("SELECT x FROM (VALUES (1), (2), (3)) AS v(x) WHERE x > 1");
        assert!(vp.post.is_some());
    }

    #[test]
    fn from_values_alias_count_mismatch_rejected() {
        let msg = values_err("SELECT * FROM (VALUES (1, 2)) AS v(only_one)");
        assert!(
            msg.contains("alias column list"),
            "mismatched alias column count must be rejected: {msg}"
        );
    }

    #[test]
    fn non_values_query_declines() {
        assert!(plan_values_query("SELECT a FROM t", &provider())
            .unwrap()
            .is_none());
    }

    // ---- generate_series TVF ---------------------------------------------

    fn gs_plan(sql: &str) -> GenerateSeriesQueryPlan {
        plan_generate_series_query(sql, &provider())
            .expect("planning must not error")
            .expect("must be a generate_series query")
    }

    fn gs_err(sql: &str) -> String {
        match plan_generate_series_query(sql, &provider()) {
            Err(e) => e.to_string(),
            Ok(other) => panic!("expected a generate_series error, got {other:?}"),
        }
    }

    #[test]
    fn gs_row_count_ascending_descending_and_step() {
        // Inclusive of both endpoints, step = 1.
        assert_eq!(generate_series_row_count(1, 5, 1).unwrap(), 5);
        // Step that lands exactly on stop.
        assert_eq!(generate_series_row_count(0, 10, 2).unwrap(), 6);
        // Step that overshoots stop (last value <= stop).
        assert_eq!(generate_series_row_count(1, 10, 3).unwrap(), 4); // 1,4,7,10
        assert_eq!(generate_series_row_count(1, 9, 3).unwrap(), 3); // 1,4,7
        // Descending.
        assert_eq!(generate_series_row_count(5, 1, -1).unwrap(), 5);
        assert_eq!(generate_series_row_count(10, 0, -2).unwrap(), 6);
        // Single element (start == stop).
        assert_eq!(generate_series_row_count(7, 7, 1).unwrap(), 1);
        assert_eq!(generate_series_row_count(7, 7, -1).unwrap(), 1);
    }

    #[test]
    fn gs_empty_direction_is_zero_rows() {
        // start > stop with positive step → empty (not an error).
        assert_eq!(generate_series_row_count(5, 1, 1).unwrap(), 0);
        // start < stop with negative step → empty.
        assert_eq!(generate_series_row_count(1, 5, -1).unwrap(), 0);
    }

    #[test]
    fn gs_step_zero_rejected() {
        let msg = generate_series_row_count(1, 5, 0)
            .expect_err("step 0 must be rejected")
            .to_string();
        assert!(
            msg.contains("step size cannot be zero"),
            "step=0 must be a clean error: {msg}"
        );
    }

    #[test]
    fn gs_row_count_near_extremes_does_not_overflow() {
        // Full i64 span with step 1 → count = 2^64 - 1 + 1 = 2^64, clamped to
        // u64::MAX. Must not panic / overflow.
        let n = generate_series_row_count(i64::MIN, i64::MAX, 1).unwrap();
        assert_eq!(n, u64::MAX);
        // Large descending span with a large step.
        assert_eq!(
            generate_series_row_count(i64::MAX, i64::MIN, i64::MIN).unwrap(),
            2
        );
    }

    #[test]
    fn gs_values_match_count_and_endpoints() {
        let n = generate_series_row_count(2, 8, 2).unwrap();
        let v = generate_series_values(2, 2, n);
        assert_eq!(v, vec![2, 4, 6, 8]);
        let n = generate_series_row_count(5, 1, -2).unwrap();
        let v = generate_series_values(5, -2, n);
        assert_eq!(v, vec![5, 3, 1]);
        // Values near i64::MAX: final increment must not wrap.
        let n = generate_series_row_count(i64::MAX - 2, i64::MAX, 1).unwrap();
        assert_eq!(n, 3);
        let v = generate_series_values(i64::MAX - 2, 1, n);
        assert_eq!(v, vec![i64::MAX - 2, i64::MAX - 1, i64::MAX]);
    }

    #[test]
    fn gs_row_cap_enforced_via_pure_helper() {
        // Pure cap check — no env mutation, so this never races parallel tests.
        let msg = enforce_generate_series_row_cap(11, 10)
            .expect_err("cap must be exceeded")
            .to_string();
        assert!(msg.contains("cap"), "row-cap error must mention the cap: {msg}");
        assert!(enforce_generate_series_row_cap(10, 10).is_ok());
    }

    #[test]
    fn gs_from_two_args_default_naming() {
        let gp = gs_plan("SELECT * FROM generate_series(1, 4)");
        // No alias → relation and column both default to `generate_series`.
        assert_eq!(gp.bind_name, "generate_series");
        assert_eq!(gp.relation.column_name, "generate_series");
        assert_eq!(gp.relation.values, vec![1, 2, 3, 4]);
        let schema = gp.relation.schema();
        assert_eq!(schema.fields.len(), 1);
        assert_eq!(schema.fields[0].dtype, DataType::Int64);
        assert!(!schema.fields[0].nullable);
    }

    #[test]
    fn gs_three_arg_descending() {
        let gp = gs_plan("SELECT * FROM generate_series(10, 0, -5)");
        assert_eq!(gp.relation.values, vec![10, 5, 0]);
    }

    #[test]
    fn gs_alias_names_relation_and_column_with_where() {
        // `AS t(n)`: relation `t`, column `n`; plus a WHERE over it.
        let gp = gs_plan("SELECT n FROM generate_series(1, 10) AS t(n) WHERE n > 5");
        assert_eq!(gp.bind_name, "t");
        assert_eq!(gp.relation.column_name, "n");
        assert_eq!(gp.relation.values, (1..=10).collect::<Vec<i64>>());
        // The outer query lowered to a Project over a Filter over a Scan of the
        // bound relation (no optimizer runs on the frontend `plan_query` output).
        assert!(
            matches!(&gp.post, LogicalPlan::Project { input, .. } if matches!(input.as_ref(), LogicalPlan::Filter { .. })),
            "post plan should be a Project over a Filter: {:?}",
            gp.post
        );
        // Output schema exposes `n`.
        let out = gp.schema().expect("schema");
        assert_eq!(out.fields[0].name, "n");
    }

    #[test]
    fn gs_table_alias_only_names_relation_column_defaults() {
        let gp = gs_plan("SELECT * FROM generate_series(1, 3) AS gs");
        assert_eq!(gp.bind_name, "gs");
        assert_eq!(gp.relation.column_name, "generate_series");
    }

    #[test]
    fn gs_empty_direction_plan_has_zero_rows() {
        let gp = gs_plan("SELECT * FROM generate_series(5, 1)");
        assert!(gp.relation.values.is_empty());
    }

    #[test]
    fn gs_step_zero_rejected_in_from() {
        let msg = gs_err("SELECT * FROM generate_series(1, 10, 0)");
        assert!(
            msg.contains("step size cannot be zero"),
            "step=0 must be rejected: {msg}"
        );
    }

    #[test]
    fn gs_non_constant_arg_rejected() {
        // A column reference is not a constant integer bound.
        let msg = gs_err("SELECT * FROM generate_series(a, 10)");
        assert!(
            msg.contains("constant integer") || msg.contains("column-correlated"),
            "non-constant arg must be rejected: {msg}"
        );
    }

    #[test]
    fn gs_non_integer_arg_rejected() {
        let msg = gs_err("SELECT * FROM generate_series(1, 10.5)");
        assert!(
            msg.contains("integer"),
            "non-integer arg must be rejected: {msg}"
        );
    }

    #[test]
    fn gs_wrong_arg_count_rejected() {
        let msg = gs_err("SELECT * FROM generate_series(1)");
        assert!(
            msg.contains("2 or 3"),
            "wrong arg count must be rejected: {msg}"
        );
    }

    #[test]
    fn gs_negative_literal_bounds_fold() {
        // Unary-minus literals fold to integer constants.
        let gp = gs_plan("SELECT * FROM generate_series(-3, 3, 3)");
        assert_eq!(gp.relation.values, vec![-3, 0, 3]);
    }

    #[test]
    fn gs_non_generate_series_query_declines() {
        // A non-TVF query is not a generate_series plan.
        assert!(plan_generate_series_query("SELECT a FROM t", &provider())
            .unwrap()
            .is_none());
    }

    #[test]
    fn gs_row_cap_rejected_in_from() {
        // The default cap is 10_000_000; a 20M-row series must be rejected
        // (this exercises the cap via the planner, with the default env cap).
        let msg = gs_err("SELECT * FROM generate_series(1, 20000000)");
        assert!(
            msg.contains("cap"),
            "an over-cap series must be rejected: {msg}"
        );
    }

    // ---- DISTINCT ON ------------------------------------------------------

    fn distinct_on_plan(sql: &str) -> DistinctOnPlan {
        plan_distinct_on(sql, &provider())
            .expect("planning must not error")
            .expect("must be a DISTINCT ON query")
    }

    fn distinct_on_err(sql: &str) -> String {
        match plan_distinct_on(sql, &provider()) {
            Err(e) => e.to_string(),
            Ok(other) => panic!("expected a DISTINCT ON error, got {other:?}"),
        }
    }

    #[test]
    fn distinct_on_basic_shape() {
        let dp = distinct_on_plan("SELECT DISTINCT ON (a) a, b FROM t ORDER BY a, b");
        assert_eq!(dp.n_keys, 1);
        // Output is the user projection [a, b].
        assert_eq!(dp.output_schema.fields.len(), 2);
        assert_eq!(dp.output_schema.fields[0].name, "a");
        assert_eq!(dp.output_schema.fields[1].name, "b");
    }

    #[test]
    fn distinct_on_multi_key() {
        let dp = distinct_on_plan("SELECT DISTINCT ON (a, b) a, b FROM t ORDER BY a, b");
        assert_eq!(dp.n_keys, 2);
    }

    #[test]
    fn distinct_on_with_limit_carries_limit() {
        let dp = distinct_on_plan("SELECT DISTINCT ON (a) a FROM t ORDER BY a LIMIT 5");
        assert_eq!(dp.limit, Some((5, 0)));
    }

    #[test]
    fn distinct_on_computed_key_rejected() {
        let msg = distinct_on_err("SELECT DISTINCT ON (a + 1) a FROM t ORDER BY a");
        assert!(
            msg.contains("simple column reference"),
            "computed DISTINCT ON key must be rejected: {msg}"
        );
    }

    #[test]
    fn distinct_on_with_group_by_rejected() {
        let msg = distinct_on_err("SELECT DISTINCT ON (a) a FROM t GROUP BY a");
        assert!(msg.contains("GROUP BY"), "DISTINCT ON + GROUP BY rejected: {msg}");
    }

    #[test]
    fn plain_select_declines_distinct_on() {
        assert!(plan_distinct_on("SELECT a FROM t", &provider())
            .unwrap()
            .is_none());
        assert!(plan_distinct_on("SELECT DISTINCT a FROM t", &provider())
            .unwrap()
            .is_none());
    }
}
