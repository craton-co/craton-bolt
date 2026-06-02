// SPDX-License-Identifier: Apache-2.0

//! String-literal predicate rewriting.
//!
//! The GPU codegen path has no Utf8 column support: variable-width strings
//! force pointer-chasing comparisons that defeat coalesced loads. Instead we
//! dictionary-encode Utf8 columns on the host (see [`crate::cuda::dictionary`]
//! and [`crate::cuda::dictionary_i64`]) and rewrite predicates of the form
//! `col = 'literal'` into an integer equality against the corresponding
//! dictionary-index column. The GPU then sees only integer equality, which it
//! already handles.
//!
//! Supported rewrites:
//!   * `col = 'lit'`  → `__idx_col = <dict index of lit>` (Int32 or Int64,
//!     matching the dictionary variant's index width)
//!   * `col <> 'lit'` → `__idx_col <> <dict index of lit>`
//!   * `col LIKE 'pattern'` (constant pattern, no `ESCAPE`, not negated) over
//!     a dict-encoded Utf8 column → an OR-of-equalities on `__idx_col` against
//!     the dictionary indices whose entries match the pattern. The pattern is
//!     evaluated HOST-side once against each dictionary entry (reusing
//!     [`crate::exec::like::PatternMatcher`]) to build the per-dict-entry match
//!     set — the "dictionary-precompute" — so the GPU only ever does integer-
//!     index compares, never device-side string scanning. An empty match set
//!     folds to `Bool(false)`. See [`LiteralResolver::like_match_indices`] and
//!     [`build_index_membership`]. `NOT LIKE`, `LIKE ... ESCAPE`, non-constant
//!     patterns, and non-dict Utf8 columns keep the host `LIKE` fallback.
//!   * Reversed shape (literal on the left) is normalised before rewrite.
//!   * If `'lit'` is not in the dictionary AND the dictionary is known-complete
//!     for that column (it observed every distinct value of the column at build
//!     time), the predicate is constant-folded: `=` → `Bool(false)`,
//!     `<>` → `Bool(true)`.
//!   * If `'lit'` is not in the dictionary and completeness is NOT guaranteed,
//!     the predicate is left as the original `col <op> 'lit'` string
//!     comparison (no dictionary-index rewrite, no constant fold) so the host
//!     path evaluates it against the actual decoded strings. See finding
//!     PL-M6 below.
//!
//! ## Completeness invariant (finding PL-M6)
//!
//! The "literal absent ⇒ `Bool(false)`" fold is only sound when the
//! dictionary observed *every distinct value* of the column. If the dictionary
//! was built from a sampled / partial batch — or the upload path coalesces
//! `""`→NULL differently from the source — a value that legitimately exists in
//! the column but was absent from the dictionary snapshot would make
//! `col = 'thatvalue'` fold to `false`: a silent wrong result. (The
//! closely-related union-dictionary bug is exercised by the C10 test below.)
//!
//! The resolver therefore exposes a [`LiteralResolver::is_complete`] signal,
//! which defaults to `false` (the safe assumption). The false-fold is gated on
//! it: only a column whose dictionary is *provably* complete folds an absent
//! literal to a constant; otherwise the rewriter falls back to the always-
//! correct host string comparison. A literal that IS in the dictionary always
//! folds to its index regardless of completeness — that fast path is exact.
//!
//! Ordering comparisons (finding F10):
//!   * `< <= > >=` on a Utf8 column against a Utf8 literal (the common
//!     `col OP 'lit'` case, either orientation) are lowered via a
//!     **byte-lexicographic collation** precompute. Dictionary indices reflect
//!     insertion order, not sort order, so instead of comparing indices the
//!     host partitions the dictionary entries by the literal under binary
//!     (UTF-8 byte) collation and emits an OR-of-equalities on the
//!     `__idx_<col>` integer column — the same form the LIKE precompute uses,
//!     so no collation-rank column or new kernel is needed. The literal need
//!     not be in the dictionary (half-open insertion partition). NULL rows
//!     (GPU index 0) are never in the match set, so a NULL value never passes
//!     an ordering predicate — the correct projection of SQL 3VL into a boolean
//!     filter. This is **binary** collation, NOT locale-aware / ICU collation,
//!     which is out of scope.
//!   * Column-vs-column Utf8 ordering (`col_a < col_b`, both dict columns) is
//!     lowered to a GPU rank comparison by finding F12 below: the rewriter emits
//!     a NULL-safe integer predicate over two materialised per-row rank columns.
//!     Any shape the rank path can't cover (a non-dict or protected column,
//!     non-`Column OP Column`) falls back to the always-correct host string
//!     comparison.
//!
//! Column-vs-column Utf8 ordering (finding F12):
//!   * `col_a OP col_b` (`OP` in `< <= > >=`) over two dict-encoded Utf8
//!     columns is order-equivalent to comparing the two rows' **collation
//!     ranks**: `rank(a_row) OP rank(b_row)`, where `rank` maps a row's
//!     dictionary index to the byte-sorted position of its string. Equality
//!     already works via index equality; ordering needs ranks because a
//!     dictionary's insertion-index order is not its lexicographic order.
//!   * CROSS-DICTIONARY CORRECTNESS: `col_a` and `col_b` may carry DIFFERENT
//!     dictionaries, so each column's *own* collation rank is meaningless to the
//!     other (rank 2 in dict A and rank 2 in dict B name unrelated strings).
//!     The dictionary layer therefore builds ONE shared byte-sorted universe —
//!     the de-duplicated union of both dictionaries — and ranks BOTH columns
//!     against it (`unified_rank_maps_of` / `DictionaryColumnAny::
//!     unified_rank_maps_with`). Then `rank_a(i) OP rank_b(j)` reproduces
//!     `string_a(i) OP string_b(j)` under byte collation. The same-dictionary
//!     case is the degenerate one where the union is a single copy and the two
//!     rank tables coincide. This is **binary collation**, NOT locale/ICU.
//!   * NULL handling (SQL 3VL): the NULL slot (GPU index 0) maps to the rank
//!     sentinel `-1` in both tables. Rather than wire validity pointers into the
//!     predicate kernel, the rewriter encodes 3VL *in the integer IR* — it emits
//!     `(__rank_a >= 0) AND (__rank_b >= 0) AND (__rank_a OP __rank_b)`. A NULL
//!     row materialises to `-1`, so its `>= 0` guard fails and the conjunction
//!     is false: the row is dropped from the filter, never satisfying the
//!     ordering — exactly the host string comparison's behaviour. See
//!     [`build_rank_comparison`].
//!   * EXEC WIRING (now LIVE): the rewriter records the cross-dictionary unified
//!     rank tables in a resolver side-channel ([`LiteralResolver::
//!     record_rank_plan`]); [`crate::exec::dict_registry::DictRegistry`] drains
//!     them keyed by the `__rank_<col>` names, and
//!     `src/exec/engine.rs::execute_projection` MATERIALISES each per-row rank
//!     column (`rank_table[index_column[row]]`) as an i64 device column the
//!     existing integer-comparison kernel consumes. The recorded plan and the
//!     emitted predicate come from the SAME `plan_col_vs_col_rank` call, so the
//!     executor can always back the comparison; shapes the rank path declines
//!     fall back to the host string comparison.
//!
//! Deferred:
//!   * `IN ('a','b','c')` — would lower to OR-of-equalities; not in scope.
//!   * Aggregate / group-by expressions over Utf8 columns — would need a
//!     separate dictionary-aware GROUP BY path.
//!
//! BREAKING CHANGE: the [`LiteralResolver`] trait's `resolve` method now
//! returns `Option<LiteralIndex>` instead of `Option<i32>`, so the rewriter
//! can emit either an `Int32` or `Int64` literal depending on the underlying
//! dictionary's index width. The only in-tree implementor is
//! [`StringPredicateRewriter`] itself (plus the test mock); external
//! implementors must migrate.

use std::collections::HashMap;

use crate::cuda::dictionary_any::DictionaryColumnAny;
use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{
    BinaryOp, DataType, Expr, Field, Literal, LogicalPlan, Schema,
};

/// Convention: the index column for a Utf8 column named `c` is `__idx_<c>`.
/// The engine uploads the dictionary indices under this name. The integer
/// width (Int32 vs Int64) is chosen per column based on cardinality; see
/// [`DictionaryColumnAny::index_dtype`].
pub fn index_column_name(original: &str) -> String {
    format!("__idx_{}", original)
}

/// Result of a literal lookup against a registered Utf8 column.
///
/// Carries both the index value and the integer width to use when emitting
/// the rewritten predicate's literal operand. The width must match the
/// `__idx_<col>` column's declared dtype on the scan side — i32-indexed
/// dictionaries produce [`LiteralIndex::I32`] and i64-indexed dictionaries
/// produce [`LiteralIndex::I64`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralIndex {
    /// The literal corresponds to an i32 index in an i32-indexed dictionary.
    I32(i32),
    /// The literal corresponds to an i64 index in an i64-indexed dictionary.
    I64(i64),
}

impl LiteralIndex {
    /// Build a [`Literal`] AST node carrying this index value with its dtype.
    pub fn into_literal(self) -> Literal {
        match self {
            LiteralIndex::I32(v) => Literal::Int32(v),
            LiteralIndex::I64(v) => Literal::Int64(v),
        }
    }

    /// The plan dtype of the wrapped index value.
    pub fn dtype(&self) -> DataType {
        match self {
            LiteralIndex::I32(_) => DataType::Int32,
            LiteralIndex::I64(_) => DataType::Int64,
        }
    }
}

/// The cross-dictionary rank lowering for a column-vs-column Utf8 ordering
/// comparison `col_a OP col_b` (finding F12).
///
/// Both `rank_a` and `rank_b` are per-GPU-index rank lookup tables computed
/// against ONE shared byte-sorted universe (the union of the two columns'
/// dictionaries). Each is indexed by the column's GPU dictionary index (the
/// same value stored in its `__idx_<col>` column):
///   * entry `0` is the NULL slot and holds
///     [`crate::cuda::dictionary::NULL_RANK_SENTINEL`] (`-1`);
///   * entry `k` (`k >= 1`) is the 0-based rank of that slot's string in the
///     shared universe.
///
/// Because both tables share the universe, `rank_a[idx_a] OP rank_b[idx_b]` is
/// order-equivalent to the byte-collation comparison of the two underlying
/// strings — correct even across different dictionaries. The degenerate
/// same-dictionary case (identical dictionaries) collapses the universe to one
/// copy and makes the two tables identical.
///
/// `rank_col_a` / `rank_col_b` are the names the executor would expose for the
/// materialised per-row rank columns (one rank value per row, gathered as
/// `rank_table[index_column[row]]`). They follow the `__rank_<col>` convention,
/// mirroring `__idx_<col>`. The rewriter would emit `rank_col_a OP rank_col_b`
/// once the executor can materialise those columns; see the exec-hook note in
/// the module docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColVsColRankPlan {
    /// Per-GPU-index rank table for `col_a` (slot 0 = NULL sentinel).
    pub rank_a: Vec<i64>,
    /// Per-GPU-index rank table for `col_b` (slot 0 = NULL sentinel).
    pub rank_b: Vec<i64>,
    /// Name the executor would give `col_a`'s materialised per-row rank column.
    pub rank_col_a: String,
    /// Name the executor would give `col_b`'s materialised per-row rank column.
    pub rank_col_b: String,
}

/// Convention: the per-row rank column for a Utf8 column named `c` (used only by
/// the deferred column-vs-column ordering lowering, finding F12) is
/// `__rank_<c>`. Mirrors [`index_column_name`]'s `__idx_<c>`.
pub fn rank_column_name(original: &str) -> String {
    format!("__rank_{}", original)
}

/// Abstracts dictionary lookups so the rewriter can be exercised without a
/// real [`DictionaryColumnAny`] (which requires CUDA to construct). The
/// production implementation is [`StringPredicateRewriter`]; tests provide
/// in-memory mocks.
pub trait LiteralResolver {
    /// Resolve a literal string against `column`'s dictionary. Returns
    /// `Some(LiteralIndex::I32 | I64)` matching the dictionary's index width,
    /// or `None` if the literal isn't in the dictionary.
    fn resolve(&self, column: &str, literal: &str) -> Option<LiteralIndex>;

    /// Mangled i32 OR i64 index-column name for an original Utf8 column.
    fn index_column_name(&self, original: &str) -> String;

    /// True if `column` is a registered Utf8 column with a dictionary.
    fn knows(&self, column: &str) -> bool;

    /// Evaluate a constant LIKE `pattern` against every entry of `column`'s
    /// dictionary, host-side, and return the [`LiteralIndex`]es (matching the
    /// dictionary's index width) of the entries that match.
    ///
    /// This is the dictionary-precompute step that lets a `col LIKE 'pat'`
    /// predicate over a dict-encoded Utf8 column be lowered to a GPU
    /// integer-index lookup: the host compiles the pattern once (reusing
    /// [`crate::exec::like::PatternMatcher`]), scans the dictionary, and the
    /// rewriter turns the returned index set into an OR-of-equalities on the
    /// `__idx_<col>` column. The GPU then only ever does integer-index
    /// compares — no device-side string scanning.
    ///
    /// The NULL slot (dictionary index `0`) is never included: a NULL row
    /// must yield SQL NULL, not a LIKE match, and the OR-of-equalities form
    /// the rewriter emits would turn a slot-0 hit into `true`. Excluding it
    /// keeps the lowered predicate correct for the non-negated `LIKE` shape
    /// the rewriter targets.
    ///
    /// Returns `None` when:
    ///   * `column` is not a registered dictionary column (caller keeps the
    ///     original host-evaluated `LIKE`), or
    ///   * the pattern fails to compile (e.g. a malformed ESCAPE sequence) —
    ///     the caller leaves the predicate intact so the host path surfaces
    ///     the same error / behaviour.
    ///
    /// `escape` mirrors [`crate::exec::like::PatternMatcher::compile`]'s
    /// escape parameter. The default implementation returns `None` (no
    /// dictionary), so non-dict resolvers transparently keep the host path.
    fn like_match_indices(
        &self,
        column: &str,
        pattern: &str,
        escape: Option<char>,
    ) -> Option<Vec<LiteralIndex>> {
        let _ = (column, pattern, escape);
        None
    }

    /// Byte-lexicographic ordering precompute (finding F10): the
    /// [`LiteralIndex`]es (matching the dictionary's index width) of every
    /// entry of `column`'s dictionary that satisfies `entry OP literal` under
    /// **binary** (UTF-8 byte) collation.
    ///
    /// `op` is one of the four ordering comparisons (`Lt`/`LtEq`/`Gt`/`GtEq`);
    /// the rewriter only calls this for those. This lets `col OP 'lit'` over a
    /// dict-encoded Utf8 column lower to an OR-of-equalities on the
    /// `__idx_<col>` integer column (the same form the LIKE precompute emits) —
    /// no collation rank column or new kernel is needed, because the host has
    /// already partitioned the dictionary by the literal.
    ///
    /// The NULL slot (GPU index `0`) is never included: a NULL string compares
    /// as SQL NULL, never satisfying an ordering predicate, and the
    /// OR-of-equalities form the rewriter emits would otherwise turn a slot-0
    /// hit into `true`. The probe literal need not be present in the
    /// dictionary — the per-entry comparison partitions the entries correctly
    /// (half-open insertion semantics), so strict (`<`) vs non-strict (`<=`)
    /// bounds are exact whether or not the literal is itself an entry.
    ///
    /// This is **binary collation**, NOT locale-aware / ICU collation, which is
    /// out of scope. Returns `None` when `column` is not a registered
    /// dictionary column (caller keeps the original host-evaluated comparison).
    /// The default returns `None` so non-dict resolvers keep the host path.
    fn ordering_match_indices(
        &self,
        column: &str,
        op: BinaryOp,
        literal: &str,
    ) -> Option<Vec<LiteralIndex>> {
        let _ = (column, op, literal);
        None
    }

    /// Cross-dictionary rank lowering for a column-vs-column Utf8 ordering
    /// comparison `col_a OP col_b` (finding F12).
    ///
    /// Returns the unified rank lookup tables for the two columns — both ranked
    /// against ONE shared byte-sorted universe (the union of the two
    /// dictionaries), so the ranks are directly comparable even when the two
    /// columns carry completely different dictionaries. See
    /// [`ColVsColRankPlan`] for the layout. Returns `None` when either column is
    /// not a registered dictionary column (the caller keeps the original
    /// host-evaluated string comparison).
    ///
    /// This produces the rank *tables* (one rank per dictionary slot); turning
    /// them into per-row rank columns on the device and comparing those is an
    /// executor responsibility that is NOT yet wired (see the module-level
    /// "Deferred" note and the exec-hook description there). Until that hook
    /// exists the rewriter does not emit a rank comparison — it preserves the
    /// host string comparison — but this method (and its production
    /// implementation) is complete and unit-tested so the wiring is a localised
    /// change. The default returns `None` so non-dict resolvers keep the host
    /// path.
    fn col_vs_col_rank_maps(
        &self,
        col_a: &str,
        col_b: &str,
    ) -> Option<ColVsColRankPlan> {
        let _ = (col_a, col_b);
        None
    }

    /// Side-channel hook (finding F12): record a [`ColVsColRankPlan`] that the
    /// live rewrite is about to lower to a `__rank_a OP __rank_b` integer
    /// comparison.
    ///
    /// The rank *tables* (`rank_a` / `rank_b`) are computed at rewrite time from
    /// both columns' dictionaries against ONE shared byte-sorted universe (so
    /// they are cross-dictionary-correct), but the executor — which only sees
    /// the rewritten `__rank_<col>` column references — needs those tables to
    /// MATERIALISE the per-row rank columns (`rank_table[index_column[row]]`) on
    /// the device. There is no way to recover the *pairing* (which two
    /// dictionaries share a universe) from a single `__rank_<col>` name, so the
    /// rewriter hands the precomputed tables to the resolver here, keyed by the
    /// rank-column name, and the executor reads them back from the same
    /// side-channel.
    ///
    /// The default is a no-op so test mocks and non-registry resolvers that
    /// never wire an executor stay unaffected (their emitted rank comparison is
    /// still exercised structurally). The production
    /// [`StringPredicateRewriter`] overrides this to buffer the plans for the
    /// [`crate::exec::dict_registry::DictRegistry`] to drain.
    fn record_rank_plan(&self, plan: &ColVsColRankPlan) {
        let _ = plan;
    }

    /// True iff a `col_a OP col_b` Utf8 ordering comparison somewhere in the
    /// plan ranks `column` — i.e. the scan schema must declare a `__rank_<col>`
    /// Int64 field so the physical planner can resolve the synthetic rank
    /// column the rewriter emits (finding F12).
    ///
    /// The scan-schema extension runs at the Scan leaf, BEFORE the predicate
    /// walk that emits the rank comparison, so a resolver that lowers col-vs-col
    /// ordering must compute this set up front (a pre-pass over the plan). The
    /// default is `false`, so resolvers that never lower col-vs-col ordering
    /// (test mocks, non-registry callers) add no rank columns and leave the
    /// common-case scan schema byte-for-byte unchanged. The production
    /// [`StringPredicateRewriter`] overrides this from a pre-pass it runs in
    /// [`Self::rewrite`].
    fn needs_rank_column(&self, column: &str) -> bool {
        let _ = column;
        false
    }

    /// True iff `column`'s dictionary is *known-complete*: it observed every
    /// distinct value the column can hold at build time (full scan, not a
    /// sample / partial batch, and with no `""`→NULL coalescing that would
    /// drop an observable value).
    ///
    /// This gates the "literal absent ⇒ constant fold" optimisation (finding
    /// PL-M6). When it returns `false`, an absent literal must NOT be folded to
    /// `Bool(false)` / `Bool(true)`, because the column may legitimately
    /// contain a value the dictionary never saw; the rewriter instead leaves
    /// the predicate as the original string comparison for correct host-side
    /// evaluation.
    ///
    /// Defaults to `false` — the safe assumption. Resolvers that can *prove*
    /// completeness (e.g. the dictionary was built from the full column in a
    /// single pass) override this to return `true` for those columns. A literal
    /// that IS present still folds to its index regardless of this signal.
    fn is_complete(&self, column: &str) -> bool {
        let _ = column;
        false
    }

    /// Plan dtype of `column`'s index column (`__idx_<col>`). Used by the
    /// scan-schema extension to declare the correct integer width when no
    /// upstream pass has already added the field. Defaults to
    /// [`DataType::Int32`] for back-compat with the historical i32 path;
    /// resolvers backed by [`DictionaryColumnAny`] must override to honour
    /// i64-indexed columns, otherwise a width-mismatch will surface at lower
    /// time when the rewriter emits a `LiteralIndex::I64` literal against an
    /// `Int32`-declared scan column.
    fn index_dtype(&self, column: &str) -> DataType {
        let _ = column;
        DataType::Int32
    }

    /// True iff a string predicate over `column` (`col LIKE 'pat'`,
    /// `col = 'lit'`, `col <> 'lit'`) may be folded into the GPU integer-index
    /// form (the dictionary precompute / index equality).
    ///
    /// Defaults to `true`. The engine's [`StringPredicateRewriter`] overrides
    /// this to return `false` for columns that the query also *projects as a
    /// bare Utf8 output*. For those, the integer-index filter cannot produce
    /// the surviving Utf8 rows (the fused GPU scan kernel has no Utf8 register
    /// class and does not compact), so the predicate must stay a real string
    /// comparison — the physical planner then routes it to the per-row GPU
    /// `StringLikeFilter` (LIKE) or a host `Filter` (Eq/Neq), both of which
    /// materialise and compact the Utf8 output correctly. `SELECT v FROM t
    /// WHERE s = 'x'` (string column NOT projected) is unaffected and keeps the
    /// faster integer fold.
    fn predicate_rewrite_allowed(&self, column: &str) -> bool {
        let _ = column;
        true
    }
}

/// Maps a Utf8 column name to the dictionary the engine loaded for it, plus
/// the name of the index column the engine has exposed (default convention:
/// `__idx_<col>`).
///
/// Dictionaries are stored as [`DictionaryColumnAny`] so the rewriter can
/// handle both i32- and i64-indexed variants uniformly.
pub struct StringPredicateRewriter<'a> {
    /// Dictionaries by original Utf8 column name.
    dicts: HashMap<String, &'a DictionaryColumnAny>,
    /// Optional override: original-name → mangled-index-column-name.
    /// If not set, defaults to `__idx_<original>`.
    name_map: HashMap<String, String>,
    /// Columns whose dictionary is known-complete (observed every distinct
    /// value of the column). Membership gates the "absent literal ⇒ constant
    /// fold" optimisation; a column not in this set is treated as possibly
    /// incomplete (the safe default — see finding PL-M6). Registration helpers
    /// do NOT add to this set, because a [`DictionaryColumnAny`] carries no
    /// completeness guarantee on its own; callers that built the dictionary
    /// from a full single-pass scan opt in via [`Self::mark_complete`].
    complete: std::collections::HashSet<String>,
    /// Columns whose string predicate (`LIKE`, `=`, `<>`) must NOT be folded
    /// into the integer-index form because the query projects the column as a
    /// bare Utf8 output (see [`LiteralResolver::predicate_rewrite_allowed`]).
    /// Populated by [`Self::protect_predicate`] from the plan before rewriting.
    predicate_protected: std::collections::HashSet<String>,
    /// Column-vs-column rank plans (finding F12) recorded during the rewrite,
    /// in emit order. Each [`ColVsColRankPlan`] carries the cross-dictionary
    /// unified rank tables for one `col_a OP col_b` ordering comparison the
    /// rewriter lowered to `__rank_a OP __rank_b`. The executor needs these
    /// tables to materialise the per-row rank columns, so they are buffered
    /// here (interior mutability — the rewrite walk borrows `&self`) and drained
    /// by [`Self::take_rank_plans`] after [`Self::rewrite`]. See
    /// [`LiteralResolver::record_rank_plan`].
    rank_plans: std::cell::RefCell<Vec<ColVsColRankPlan>>,
    /// Original column names that a `col_a OP col_b` Utf8 ordering comparison in
    /// the plan ranks (finding F12). Populated by a pre-pass in
    /// [`Self::rewrite`] *before* the Scan-schema extension, so the Scan leaf can
    /// declare the `__rank_<col>` Int64 fields the emitted rank comparison needs
    /// the physical planner to resolve. Empty unless the plan actually contains a
    /// rank-eligible col-vs-col ordering, so the common-case scan schema is
    /// unchanged. See [`LiteralResolver::needs_rank_column`].
    rank_needed: std::cell::RefCell<std::collections::HashSet<String>>,
}

impl<'a> Default for StringPredicateRewriter<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> StringPredicateRewriter<'a> {
    /// Empty rewriter; no Utf8 columns registered.
    pub fn new() -> Self {
        Self {
            dicts: HashMap::new(),
            name_map: HashMap::new(),
            complete: std::collections::HashSet::new(),
            predicate_protected: std::collections::HashSet::new(),
            rank_plans: std::cell::RefCell::new(Vec::new()),
            rank_needed: std::cell::RefCell::new(std::collections::HashSet::new()),
        }
    }

    /// Drain the column-vs-column rank plans (finding F12) recorded during the
    /// most recent [`Self::rewrite`]. Each [`ColVsColRankPlan`] carries the
    /// cross-dictionary unified rank tables for one `col_a OP col_b` ordering
    /// comparison that was lowered to `__rank_a OP __rank_b`. The caller (the
    /// dictionary registry) keys them by their `rank_col_a` / `rank_col_b`
    /// names into the executor side-channel so `execute_projection` can
    /// materialise the per-row rank columns. Returns an empty vec when no
    /// col-vs-col ordering comparison was rewritten. Leaves the buffer empty.
    pub fn take_rank_plans(&self) -> Vec<ColVsColRankPlan> {
        let mut guard = self.rank_plans.borrow_mut();
        std::mem::take(&mut *guard)
    }

    /// Protect `column`'s string predicates (`LIKE`, `=`, `<>`) from the
    /// integer-index fold (see [`LiteralResolver::predicate_rewrite_allowed`]).
    /// Called for every column the query projects as a bare Utf8 output, so the
    /// predicate stays a real string comparison and reaches the per-row GPU
    /// `StringLikeFilter` (LIKE) or a host `Filter` (Eq/Neq) — both of which can
    /// emit the surviving Utf8 rows — instead of an integer filter that cannot.
    pub fn protect_predicate(&mut self, column: impl Into<String>) {
        self.predicate_protected.insert(column.into());
    }

    /// Mark `column`'s dictionary as known-complete: it observed every distinct
    /// value the column can hold (built from a full single-pass scan, not a
    /// sample / partial batch). Only call this when completeness is provable —
    /// it re-enables the "absent literal ⇒ constant fold" fast path for the
    /// column (finding PL-M6). Columns left unmarked are treated as possibly
    /// incomplete and fall back to host string comparison for absent literals.
    pub fn mark_complete(&mut self, column: impl Into<String>) {
        self.complete.insert(column.into());
    }

    /// Register a Utf8 column's dictionary. The mangled index-column name
    /// defaults to `__idx_<original>`. Accepts either an i32- or i64-indexed
    /// dictionary via the [`DictionaryColumnAny`] wrapper.
    pub fn register(
        &mut self,
        original_name: impl Into<String>,
        dict: &'a DictionaryColumnAny,
    ) {
        let n = original_name.into();
        let mangled = index_column_name(&n);
        self.dicts.insert(n.clone(), dict);
        self.name_map.insert(n, mangled);
    }

    /// Register a Utf8 column with an explicit mangled-index-column name.
    /// Use this when the engine has uploaded indices under a non-default
    /// name.
    pub fn register_with_name(
        &mut self,
        original_name: impl Into<String>,
        mangled_index_name: impl Into<String>,
        dict: &'a DictionaryColumnAny,
    ) {
        let n = original_name.into();
        self.dicts.insert(n.clone(), dict);
        self.name_map.insert(n, mangled_index_name.into());
    }

    /// Walk `plan` and rewrite all string-eq / string-neq predicates against
    /// registered Utf8 columns into integer equality. Also extends the
    /// schema of any [`LogicalPlan::Scan`] to include the mangled index
    /// columns when they're not already present.
    ///
    /// Returns a new owned plan. Unsupported string ops (`Lt`/`Gt`/...)
    /// yield [`BoltError::Plan`].
    pub fn rewrite(&self, plan: &LogicalPlan) -> BoltResult<LogicalPlan> {
        // Finding F12 pre-pass: discover which columns a `col_a OP col_b` Utf8
        // ordering comparison in the plan will rank, BEFORE the Scan leaf is
        // rewritten — the Scan-schema extension needs this to declare the
        // `__rank_<col>` Int64 fields the emitted rank comparison references.
        // (The predicate walk that emits the rank comparison runs after the
        // Scan leaf, so it can't be discovered lazily.) The pre-pass is a pure,
        // side-effect-free shape match using the same `plan_col_vs_col_rank`
        // eligibility gate the emit path uses, so the declared columns and the
        // emitted comparisons stay in lockstep.
        {
            let mut needed = self.rank_needed.borrow_mut();
            needed.clear();
            collect_rank_needed_columns(plan, self, &mut needed, 0);
        }
        rewrite_plan_with(plan, self, 0)
    }
}

/// Walk `plan` collecting the original column names that a rank-eligible
/// `col_a OP col_b` Utf8 ordering comparison references (finding F12 pre-pass).
///
/// A column is collected only when the comparison would actually lower to a
/// rank comparison: both operands are registered, non-protected dict columns
/// (the exact gate `plan_col_vs_col_rank` applies). Bounded by
/// `MAX_RECURSION_DEPTH` like the rewrite walks; a deeper plan simply stops
/// collecting (the rewrite itself will surface the depth error).
fn collect_rank_needed_columns<R: LiteralResolver>(
    plan: &LogicalPlan,
    r: &R,
    out: &mut std::collections::HashSet<String>,
    depth: usize,
) {
    if depth > crate::plan::sql_frontend::MAX_RECURSION_DEPTH {
        return;
    }
    fn walk_expr<R: LiteralResolver>(
        e: &Expr,
        r: &R,
        out: &mut std::collections::HashSet<String>,
        depth: usize,
    ) {
        if depth > crate::plan::sql_frontend::MAX_RECURSION_DEPTH {
            return;
        }
        if let Expr::Binary { op, left, right } = e {
            if is_ordering(*op) {
                if let Some((a, b)) = extract_two_columns(left, right) {
                    if r.knows(&a)
                        && r.knows(&b)
                        && r.predicate_rewrite_allowed(&a)
                        && r.predicate_rewrite_allowed(&b)
                    {
                        out.insert(a);
                        out.insert(b);
                    }
                }
            }
        }
        // Recurse into every sub-expression so nested predicates (e.g. inside
        // an AND / OR / CASE) are covered.
        match e {
            Expr::Binary { left, right, .. } => {
                walk_expr(left, r, out, depth + 1);
                walk_expr(right, r, out, depth + 1);
            }
            Expr::Unary { operand, .. } => walk_expr(operand, r, out, depth + 1),
            Expr::Alias(inner, _) => walk_expr(inner, r, out, depth + 1),
            Expr::Cast { expr, .. }
            | Expr::CastFormat { expr, .. }
            | Expr::Like { expr, .. }
            | Expr::Extract { expr, .. }
            | Expr::DateTrunc { expr, .. }
            | Expr::InSubquery { expr, .. } => walk_expr(expr, r, out, depth + 1),
            Expr::ScalarFn { args, .. } => {
                for a in args {
                    walk_expr(a, r, out, depth + 1);
                }
            }
            Expr::Case { branches, else_branch } => {
                for (w, t) in branches {
                    walk_expr(w, r, out, depth + 1);
                    walk_expr(t, r, out, depth + 1);
                }
                if let Some(eb) = else_branch.as_deref() {
                    walk_expr(eb, r, out, depth + 1);
                }
            }
            Expr::Column(_) | Expr::Literal(_) | Expr::ScalarSubquery(_) => {}
        }
    }
    match plan {
        LogicalPlan::Filter { input, predicate } => {
            walk_expr(predicate, r, out, 0);
            collect_rank_needed_columns(input, r, out, depth + 1);
        }
        LogicalPlan::Project { input, exprs } => {
            for e in exprs {
                walk_expr(e, r, out, 0);
            }
            collect_rank_needed_columns(input, r, out, depth + 1);
        }
        LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Distinct { input, .. }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Sort { input, .. }
        | LogicalPlan::Window { input, .. } => {
            collect_rank_needed_columns(input, r, out, depth + 1);
        }
        LogicalPlan::Union { inputs } => {
            for inp in inputs {
                collect_rank_needed_columns(inp, r, out, depth + 1);
            }
        }
        LogicalPlan::Join { left, right, .. } | LogicalPlan::SetOp { left, right, .. } => {
            collect_rank_needed_columns(left, r, out, depth + 1);
            collect_rank_needed_columns(right, r, out, depth + 1);
        }
        LogicalPlan::Scan { .. } => {}
    }
}

impl<'a> LiteralResolver for StringPredicateRewriter<'a> {
    fn resolve(&self, column: &str, literal: &str) -> Option<LiteralIndex> {
        let dict = self.dicts.get(column)?;
        match dict {
            DictionaryColumnAny::I32(d) => d.index_of(literal).map(LiteralIndex::I32),
            DictionaryColumnAny::I64(d) => d.index_of(literal).map(LiteralIndex::I64),
        }
    }

    fn index_column_name(&self, original: &str) -> String {
        self.name_map
            .get(original)
            .cloned()
            .unwrap_or_else(|| index_column_name(original))
    }

    fn knows(&self, column: &str) -> bool {
        self.dicts.contains_key(column)
    }

    fn predicate_rewrite_allowed(&self, column: &str) -> bool {
        !self.predicate_protected.contains(column)
    }

    fn like_match_indices(
        &self,
        column: &str,
        pattern: &str,
        escape: Option<char>,
    ) -> Option<Vec<LiteralIndex>> {
        let dict = self.dicts.get(column)?;
        // Compile the pattern host-side once. A compile error (e.g. a
        // dangling ESCAPE) means we can't precompute the table — leave the
        // predicate to the host path, which surfaces the same error.
        let matcher = crate::exec::like::PatternMatcher::compile(pattern, escape).ok()?;
        // `dictionary()[p]` is the string for GPU index `p + 1` (slot 0 is
        // reserved for NULL and is intentionally never tested — see the
        // trait doc). Width of the emitted index follows the variant.
        let entries = dict.dictionary();
        let is_i32 = dict.is_i32();
        let mut out: Vec<LiteralIndex> = Vec::new();
        for (p, s) in entries.iter().enumerate() {
            if matcher.matches(s) {
                let idx = (p + 1) as i64;
                out.push(if is_i32 {
                    LiteralIndex::I32(idx as i32)
                } else {
                    LiteralIndex::I64(idx)
                });
            }
        }
        Some(out)
    }

    fn ordering_match_indices(
        &self,
        column: &str,
        op: BinaryOp,
        literal: &str,
    ) -> Option<Vec<LiteralIndex>> {
        let dict = self.dicts.get(column)?;
        // Map the BinaryOp into the dictionary layer (it owns the byte-wise
        // partition and the slot-0/NULL exclusion). `indices_satisfying_any`
        // returns 1-based GPU indices widened to i64; re-narrow per the
        // dictionary's index width so the emitted literals match the
        // `__idx_<col>` column.
        let is_i32 = dict.is_i32();
        let out = dict
            .indices_satisfying_any(op, literal)
            .into_iter()
            .map(|idx| {
                if is_i32 {
                    LiteralIndex::I32(idx as i32)
                } else {
                    LiteralIndex::I64(idx)
                }
            })
            .collect();
        Some(out)
    }

    fn col_vs_col_rank_maps(
        &self,
        col_a: &str,
        col_b: &str,
    ) -> Option<ColVsColRankPlan> {
        // Both sides must be registered dictionary columns; otherwise there is
        // no dictionary to rank against and the comparison stays host-side.
        let dict_a = self.dicts.get(col_a)?;
        let dict_b = self.dicts.get(col_b)?;
        // The dictionary layer owns the cross-dictionary correctness: it builds
        // ONE shared byte-sorted universe (the union of both dictionaries) and
        // ranks each column's slots against it, so the two rank tables are
        // directly comparable even when the dictionaries differ.
        let (rank_a, rank_b) = dict_a.unified_rank_maps_with(dict_b);
        Some(ColVsColRankPlan {
            rank_a,
            rank_b,
            rank_col_a: rank_column_name(col_a),
            rank_col_b: rank_column_name(col_b),
        })
    }

    fn record_rank_plan(&self, plan: &ColVsColRankPlan) {
        // Buffer the plan for the registry to drain after the rewrite. Cloning
        // is cheap relative to the query and keeps the trait method `&self`
        // (the rewrite walk holds `&self` throughout).
        self.rank_plans.borrow_mut().push(plan.clone());
    }

    fn needs_rank_column(&self, column: &str) -> bool {
        // Populated by the pre-pass in `rewrite` (`collect_rank_needed_columns`)
        // before the Scan leaf is extended.
        self.rank_needed.borrow().contains(column)
    }

    fn is_complete(&self, column: &str) -> bool {
        // Only columns explicitly opted in via `mark_complete` are trusted as
        // having observed every distinct value. A `DictionaryColumnAny` alone
        // carries no such guarantee (it may have been built from a sample or
        // an early batch), so the default is "incomplete" — see finding PL-M6.
        self.complete.contains(column)
    }

    fn index_dtype(&self, column: &str) -> DataType {
        // Mirror the dictionary's own index width so the scan-side schema
        // matches the literal width chosen in `resolve`. Falls back to the
        // trait default (`Int32`) for unregistered columns; the scan-extension
        // path only ever asks about columns it just confirmed via `knows`, so
        // the fallback is defensive only.
        self.dicts
            .get(column)
            .map(|d| d.index_dtype())
            .unwrap_or(DataType::Int32)
    }
}

/// True for the two ops we can reduce to integer (in)equality via dictionary
/// indices.
fn is_eq_or_neq(op: BinaryOp) -> bool {
    matches!(op, BinaryOp::Eq | BinaryOp::NotEq)
}

/// True for ordering comparisons.
fn is_ordering(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq
    )
}

/// Reflect an ordering op about its operands: the op `x` such that
/// `a OP b` ⇔ `b (reflect OP) a`. Used when the string literal sits on the
/// LEFT of an ordering comparison (`'lit' < col`): we always partition the
/// dictionary as `entry OP literal`, so a left-literal shape must reflect the
/// op first (`'lit' < col` ⇔ `col > 'lit'`). Non-ordering ops are returned
/// unchanged (the caller only reflects ordering ops).
fn reflect_ordering(op: BinaryOp) -> BinaryOp {
    match op {
        BinaryOp::Lt => BinaryOp::Gt,
        BinaryOp::Gt => BinaryOp::Lt,
        BinaryOp::LtEq => BinaryOp::GtEq,
        BinaryOp::GtEq => BinaryOp::LtEq,
        other => other,
    }
}

/// Peel `Alias(inner, _)` wrappers off `e`, returning the innermost non-alias
/// expression. Lets predicate-shape matching see through DataFrame-built
/// `(col AS x) = 'US'` and equivalent forms; the alias is irrelevant for
/// dictionary-lookup purposes.
fn strip_alias(e: &Expr) -> &Expr {
    let mut cur = e;
    while let Expr::Alias(inner, _) = cur {
        cur = inner;
    }
    cur
}

/// If exactly one side of `(left, right)` is `Column(c)` and the other is
/// `Literal(Utf8(s))`, return `(column_name, literal, swapped)` where
/// `swapped` is true if the literal was originally on the left. Returns
/// `None` for any other shape.
///
/// Both operands are first peeled of any `Alias` wrappers so that DataFrame
/// expressions like `(col("region") AS "r").eq(lit("US"))` are matched as
/// `col("region") = 'US'`.
fn extract_col_and_string_lit(left: &Expr, right: &Expr) -> Option<(String, String, bool)> {
    match (strip_alias(left), strip_alias(right)) {
        (Expr::Column(c), Expr::Literal(Literal::Utf8(s))) => Some((c.clone(), s.clone(), false)),
        (Expr::Literal(Literal::Utf8(s)), Expr::Column(c)) => Some((c.clone(), s.clone(), true)),
        _ => None,
    }
}

/// If both sides of `(left, right)` are `Column`s (after peeling any `Alias`
/// wrappers), return `(left_col, right_col)`; otherwise `None`. Used to detect
/// the column-vs-column ordering shape `col_a OP col_b` (finding F12). The order
/// is preserved (`left` first) so the caller can keep `OP`'s operand order.
fn extract_two_columns(left: &Expr, right: &Expr) -> Option<(String, String)> {
    match (strip_alias(left), strip_alias(right)) {
        (Expr::Column(a), Expr::Column(b)) => Some((a.clone(), b.clone())),
        _ => None,
    }
}

/// Analyse a binary expression for the column-vs-column Utf8 ordering shape
/// `col_a OP col_b` and, if it matches two registered (non-protected) dict
/// columns, return the cross-dictionary rank lowering plan (finding F12).
///
/// This is the single entry point the live rewrite calls on the col-vs-col
/// ordering shape: given the plan's [`ColVsColRankPlan`] the rewriter (1) records
/// the `rank_a` / `rank_b` tables in the resolver side-channel
/// ([`LiteralResolver::record_rank_plan`]) so the executor can materialise the
/// per-row rank columns `rank_table[index_column[row]]` under the `rank_col_a` /
/// `rank_col_b` names, and (2) emits the NULL-safe integer predicate
/// `(rank_a >= 0) AND (rank_b >= 0) AND (rank_a OP rank_b)` (see
/// [`build_rank_comparison`]) so the existing integer-comparison machinery
/// executes the ordering while a NULL on either side (rank `-1`) drops the row.
/// See the module-level F12 note for the exec-hook contract.
///
/// Returns `None` (caller keeps the host string comparison) when the expression
/// is not an ordering `Column OP Column` over two registered, non-protected dict
/// columns, or when the resolver declines to build the rank tables.
pub fn plan_col_vs_col_rank<R: LiteralResolver>(
    op: BinaryOp,
    left: &Expr,
    right: &Expr,
    r: &R,
) -> Option<(BinaryOp, ColVsColRankPlan)> {
    if !is_ordering(op) {
        return None;
    }
    let (col_a, col_b) = extract_two_columns(left, right)?;
    if !(r.knows(&col_a)
        && r.knows(&col_b)
        && r.predicate_rewrite_allowed(&col_a)
        && r.predicate_rewrite_allowed(&col_b))
    {
        return None;
    }
    let plan = r.col_vs_col_rank_maps(&col_a, &col_b)?;
    Some((op, plan))
}

/// Build a Bool predicate that is true iff `column` (an integer index
/// column, e.g. `__idx_<col>`) equals one of `indices`.
///
/// Emitted as a left-deep OR of equalities: `(col = i0) OR (col = i1) OR …`.
/// Every operand is a GPU-lowerable integer compare (`Op::Eq`) combined with
/// `Op::Or`, so the whole tree lowers to the fused predicate kernel — this is
/// the "integer-index → Bool match table" lookup expressed in the existing
/// IR, with no new op or kernel. (Mirrors the deferred `IN ('a','b',…)`
/// note in the module docs: a membership test reduces to OR-of-equalities.)
///
/// An empty index set means the pattern matched no dictionary entry, so the
/// predicate is unconditionally `false` for every non-NULL row (and NULL
/// rows, whose index is slot 0, are likewise excluded by construction):
/// fold straight to `Bool(false)`.
fn build_index_membership(column: &str, indices: &[LiteralIndex]) -> Expr {
    let mut iter = indices.iter().copied();
    let first = match iter.next() {
        Some(idx) => idx,
        None => return Expr::Literal(Literal::Bool(false)),
    };
    let eq = |idx: LiteralIndex| Expr::Binary {
        op: BinaryOp::Eq,
        left: Box::new(Expr::Column(column.to_string())),
        right: Box::new(Expr::Literal(idx.into_literal())),
    };
    let mut acc = eq(first);
    for idx in iter {
        acc = Expr::Binary {
            op: BinaryOp::Or,
            left: Box::new(acc),
            right: Box::new(eq(idx)),
        };
    }
    acc
}

/// Lower a column-vs-column Utf8 ordering `col_a OP col_b` (finding F12) into a
/// NULL-safe integer comparison over the two materialised per-row rank columns.
///
/// The executor materialises `__rank_<a>` and `__rank_<b>` as i64 columns whose
/// value for a row is `rank_table[index_column[row]]` — the row's byte-collation
/// rank within the shared universe of both dictionaries (see
/// [`ColVsColRankPlan`] / [`crate::cuda::dictionary::unified_rank_maps_of`]). A
/// NULL row (dict index 0) materialises to
/// [`crate::cuda::dictionary::NULL_RANK_SENTINEL`] (`-1`); every real rank is
/// `>= 0`.
///
/// SQL 3VL (either-side-NULL ⇒ the row never passes an ordering filter) is
/// encoded *in the integer IR itself*, with no validity-pointer wiring or new
/// kernel: the emitted predicate is
///
/// ```text
/// (__rank_a >= 0) AND (__rank_b >= 0) AND (__rank_a OP __rank_b)
/// ```
///
/// A NULL on either side makes its `>= 0` guard false, so the whole conjunction
/// is false (the row is dropped) — exactly the projection of SQL NULL into a
/// boolean filter, and identical to what the host string-comparison path
/// produces (a NULL string never satisfies an ordering comparison). Because
/// every operand is a plain integer compare combined with `AND`, the existing
/// fused GPU predicate kernel (and the host filter evaluator) execute it
/// unchanged — the same "no new op / no new kernel" property the F10
/// OR-of-equalities lowering relies on.
///
/// The ranks are i64 (the rank tables are `Vec<i64>` with an i64 sentinel), so
/// the guards and the comparison use `Literal::Int64(0)`; the executor allocates
/// the `__rank_<col>` columns as `Int64`.
fn build_rank_comparison(op: BinaryOp, plan: &ColVsColRankPlan) -> Expr {
    let rank_a = || Expr::Column(plan.rank_col_a.clone());
    let rank_b = || Expr::Column(plan.rank_col_b.clone());
    let non_null = |c: Expr| Expr::Binary {
        op: BinaryOp::GtEq,
        left: Box::new(c),
        right: Box::new(Expr::Literal(Literal::Int64(0))),
    };
    let ordering = Expr::Binary {
        op,
        left: Box::new(rank_a()),
        right: Box::new(rank_b()),
    };
    // `(rank_a >= 0) AND (rank_b >= 0) AND (rank_a OP rank_b)`, left-deep.
    Expr::Binary {
        op: BinaryOp::And,
        left: Box::new(Expr::Binary {
            op: BinaryOp::And,
            left: Box::new(non_null(rank_a())),
            right: Box::new(non_null(rank_b())),
        }),
        right: Box::new(ordering),
    }
}

/// Recursive expression rewrite, post-order: children first, then `self`.
///
/// `depth` is the current recursion depth; returns Err if MAX_RECURSION_DEPTH is exceeded.
fn rewrite_expr_with<R: LiteralResolver>(expr: &Expr, r: &R, depth: usize) -> BoltResult<Expr> {
    if depth > crate::plan::sql_frontend::MAX_RECURSION_DEPTH {
        return Err(BoltError::Plan(format!(
            "expression nesting exceeds depth limit ({})",
            crate::plan::sql_frontend::MAX_RECURSION_DEPTH
        )));
    }
    match expr {
        Expr::Column(_) | Expr::Literal(_) => Ok(expr.clone()),
        Expr::Alias(inner, name) => {
            let inner = rewrite_expr_with(inner, r, depth + 1)?;
            Ok(Expr::Alias(Box::new(inner), name.clone()))
        }
        Expr::Cast { expr: inner, target, safe } => {
            // CAST has no string-literal comparison surface to rewrite —
            // it converts a numeric / boolean expression into another
            // primitive type. Recurse so any rewritable sub-expression
            // (e.g. `CAST(col = 'lit' AS Int64)`) is still normalised.
            let new_inner = rewrite_expr_with(inner, r, depth + 1)?;
            Ok(Expr::Cast {
                expr: Box::new(new_inner),
                target: *target,
                safe: *safe,
            })
        }
        Expr::CastFormat { expr: inner, target, pattern, to_text } => {
            // Like CAST, recurse into the operand to normalise any rewritable
            // sub-expression; the FORMAT pattern itself has no rewrite surface.
            let new_inner = rewrite_expr_with(inner, r, depth + 1)?;
            Ok(Expr::CastFormat {
                expr: Box::new(new_inner),
                target: *target,
                pattern: pattern.clone(),
                to_text: *to_text,
            })
        }
        Expr::Unary { op, operand } => {
            // The `IS NULL` / `IS NOT NULL` surface has no string-literal
            // operand to dictionary-rewrite — the operand is the value whose
            // nullness we're testing, not a constant. Recurse so any nested
            // string-literal comparisons inside a typed operand still get
            // normalised.
            let new_operand = rewrite_expr_with(operand, r, depth + 1)?;
            Ok(Expr::Unary {
                op: *op,
                operand: Box::new(new_operand),
            })
        }
        Expr::Binary { op, left, right } => {
            // Rewrite children first so nested predicates are normalised.
            let new_left = rewrite_expr_with(left, r, depth + 1)?;
            let new_right = rewrite_expr_with(right, r, depth + 1)?;

            // Try to match a `col <op> 'lit'` (or reversed) shape against a
            // registered Utf8 column.
            if let Some((col_name, lit_str, swapped)) =
                extract_col_and_string_lit(&new_left, &new_right)
            {
                if r.knows(&col_name) {
                    // `predicate_rewrite_allowed` is false when the query
                    // projects this column as a bare Utf8 output: the
                    // integer-index filter can't emit the surviving Utf8 rows,
                    // so keep the real `col <op> 'lit'` string comparison for
                    // the host `Filter` path (routed by the physical planner).
                    // `SELECT v WHERE s = 'x'` (s not projected) stays eligible
                    // for the fold. The gate is scoped to the Eq/Neq fold only;
                    // ordering ops keep their explicit "not yet implemented"
                    // error below regardless of projection.
                    if is_eq_or_neq(*op) && r.predicate_rewrite_allowed(&col_name) {
                        let mangled = r.index_column_name(&col_name);
                        match r.resolve(&col_name, &lit_str) {
                            Some(idx) => {
                                // The Eq/NotEq is symmetric, so we don't need
                                // to preserve the original side order; emit
                                // the canonical `column <op> literal` form.
                                // The literal's dtype (Int32 vs Int64) is
                                // chosen by the dictionary's index width.
                                return Ok(Expr::Binary {
                                    op: *op,
                                    left: Box::new(Expr::Column(mangled)),
                                    right: Box::new(Expr::Literal(idx.into_literal())),
                                });
                            }
                            None => {
                                // Literal not in the dictionary. Folding `=`
                                // to `Bool(false)` / `<>` to `Bool(true)` is
                                // ONLY sound when the dictionary observed every
                                // distinct value of the column at build time —
                                // i.e. it is known-complete. See finding PL-M6
                                // and the "Completeness invariant" section in
                                // the module docs.
                                //
                                // If completeness is NOT guaranteed, the column
                                // may legitimately hold `lit_str` even though
                                // the dictionary snapshot missed it (sampled /
                                // partial batch, or `""`→NULL coalescing
                                // mismatch). Folding to a constant there is a
                                // silent wrong result. So we fall through and
                                // emit the ORIGINAL `col <op> 'lit'` string
                                // comparison — no index rewrite, no fold — and
                                // let the host path evaluate it against the
                                // actual decoded strings, which is always
                                // correct.
                                //
                                // Edge case: the empty string `""` is just
                                // another literal here. With a complete
                                // dictionary, `WHERE col = ''` folds to false
                                // iff no observed row held `""` (NULL lives at
                                // slot 0, not `""`). Without completeness it is
                                // left to the host path, so `''` still matches
                                // real empty-string rows even if the dictionary
                                // snapshot never saw one.
                                if r.is_complete(&col_name) {
                                    let folded = match op {
                                        BinaryOp::Eq => false,
                                        BinaryOp::NotEq => true,
                                        _ => unreachable!("is_eq_or_neq gated this branch"),
                                    };
                                    return Ok(Expr::Literal(Literal::Bool(folded)));
                                }
                                // Incomplete dictionary: fall through to the
                                // post-match `Ok(Expr::Binary { .. })` below,
                                // which reconstructs the original (recursively
                                // rewritten) `col <op> 'lit'` comparison. The
                                // string literal is preserved verbatim, so the
                                // host string-comparison path stays correct.
                            }
                        }
                    } else if is_ordering(*op) && r.predicate_rewrite_allowed(&col_name) {
                        // Finding F10: byte-lexicographic ordering. The
                        // dictionary precompute partitions the entries by the
                        // literal under binary (UTF-8 byte) collation and the
                        // rewriter lowers `col OP 'lit'` to an OR-of-equalities
                        // on the `__idx_<col>` integer column — the same form
                        // the LIKE precompute emits, so the existing GPU/host
                        // integer machinery executes it with no collation-rank
                        // column or new kernel.
                        //
                        // NULL handling: the matching set never contains GPU
                        // index 0, so a NULL row's index (slot 0) matches no
                        // equality and the predicate is false for it — which is
                        // the correct projection of SQL 3VL into a boolean
                        // filter (a NULL ordering compares as NULL, i.e. the row
                        // does not pass). Equality semantics with the existing
                        // index-membership LIKE path are preserved verbatim.
                        //
                        // Absent literal: handled inside the dictionary layer —
                        // the per-entry byte comparison gives the half-open
                        // insertion partition, so strict vs non-strict bounds
                        // are exact whether or not 'lit' is itself an entry. No
                        // completeness signal is needed: every entry is tested
                        // against the real literal, so a literal the dictionary
                        // never saw still partitions the known entries
                        // correctly. (Rows whose value is absent from a *partial*
                        // dictionary cannot occur here — the index column only
                        // ever holds slots the dictionary defines.)
                        //
                        // NOTE: this is binary collation, NOT locale-aware ICU
                        // collation, which is out of scope.
                        // The dictionary always partitions as `entry OP lit`.
                        // If the literal was on the LEFT (`'lit' OP col`), the
                        // predicate is `lit OP entry` ⇔ `entry (reflect OP) lit`,
                        // so reflect the op before asking the dictionary.
                        let probe_op = if swapped { reflect_ordering(*op) } else { *op };
                        if let Some(indices) =
                            r.ordering_match_indices(&col_name, probe_op, &lit_str)
                        {
                            let mangled = r.index_column_name(&col_name);
                            return Ok(build_index_membership(&mangled, &indices));
                        }
                        // Resolver declined (not a dict column it can partition):
                        // fall through and preserve the original comparison for
                        // the host path.
                    }
                    // Other ops (arithmetic, logical) against a Utf8 column
                    // are type errors elsewhere; fall through and let the
                    // standard type checker surface them.
                }
            }

            // Finding F12: column-vs-column Utf8 ordering (`col_a OP col_b`,
            // both registered dict columns). This is order-equivalent to
            // comparing the two rows' byte-collation ranks computed against a
            // SHARED universe (the union of both dictionaries) — see
            // `LiteralResolver::col_vs_col_rank_maps` and the module docs.
            //
            // GPU rank lowering (now LIVE): detect the `col_a OP col_b` ordering
            // shape over two registered, non-protected dict columns, build the
            // cross-dictionary-correct unified rank tables, record them in the
            // resolver's side-channel (so the executor can MATERIALISE the
            // per-row `__rank_<col>` columns as `rank_table[index[row]]`), and
            // emit a NULL-safe integer comparison
            //   `(__rank_a >= 0) AND (__rank_b >= 0) AND (__rank_a OP __rank_b)`
            // (see `build_rank_comparison`). A NULL row's index is 0, which
            // materialises to the `-1` rank sentinel, so its `>= 0` guard fails
            // and the conjunction is false — SQL 3VL projected into a boolean
            // filter, identical to the host string comparison and needing NO
            // new kernel or validity-pointer wiring.
            //
            // FALLBACK: `plan_col_vs_col_rank` returns `None` for any shape the
            // GPU rank path can't cover — a non-`Column OP Column` shape, a
            // non-ordering op, a column that isn't a registered dict column, or
            // a protected (bare-Utf8-projected) column. In every such case we
            // fall through to the preserved `col_a OP col_b` Utf8 comparison,
            // which the physical planner routes to the always-correct host
            // string path. We NEVER emit a rank comparison the executor can't
            // back with materialised rank columns: the recorded plan and the
            // emitted predicate come from the same `plan_col_vs_col_rank` call.
            if let Some((rank_op, plan)) =
                plan_col_vs_col_rank(*op, &new_left, &new_right, r)
            {
                // Hand the precomputed unified rank tables to the executor via
                // the resolver side-channel, keyed by the `__rank_<col>` names.
                r.record_rank_plan(&plan);
                return Ok(build_rank_comparison(rank_op, &plan));
            }

            Ok(Expr::Binary {
                op: *op,
                left: Box::new(new_left),
                right: Box::new(new_right),
            })
        }
        Expr::Case {
            branches,
            else_branch,
        } => {
            let new_branches = branches
                .iter()
                .map(|(w, t)| {
                    Ok::<_, BoltError>((
                        rewrite_expr_with(w, r, depth + 1)?,
                        rewrite_expr_with(t, r, depth + 1)?,
                    ))
                })
                .collect::<BoltResult<Vec<_>>>()?;
            let new_else = match else_branch {
                Some(e) => Some(Box::new(rewrite_expr_with(e, r, depth + 1)?)),
                None => None,
            };
            Ok(Expr::Case {
                branches: new_branches,
                else_branch: new_else,
            })
        }
        Expr::Like {
            expr: like_expr,
            pattern,
            escape,
            negated,
            case_insensitive,
        } => {
            let new_inner = rewrite_expr_with(like_expr, r, depth + 1)?;

            // Dictionary-precompute lowering: `col LIKE 'pat'` over a
            // dict-encoded Utf8 column becomes an OR-of-equalities on the
            // `__idx_<col>` integer index against the set of dictionary
            // entries that match the (constant) pattern. The match table is
            // built host-side via `PatternMatcher`; the GPU only ever sees
            // integer-index compares, so the predicate no longer forces the
            // whole filter onto the host. See `LiteralResolver::like_match_indices`.
            //
            // Gated to the lowest-risk shape:
            //   * non-negated `LIKE` (NOT LIKE has SQL-NULL 3VL semantics the
            //     OR-of-equalities form can't express, so it stays host-side);
            //   * no `ESCAPE` clause (kept host per the task — the host path
            //     already handles escape and the precompute would otherwise
            //     duplicate that logic at the rewrite boundary);
            //   * the operand is a bare `Column` of a registered dict column
            //     (after peeling any `Alias` wrappers).
            // Anything else falls through to the preserved `Expr::Like` below,
            // which the physical planner routes to the host filter.
            //
            // `ILIKE` (case_insensitive) is deliberately excluded: the
            // dictionary precompute uses the case-sensitive
            // `LiteralResolver::like_match_indices`, so applying it to an
            // ILIKE would silently produce case-sensitive results. ILIKE
            // therefore always falls through to the host `Expr::Like` path.
            if !*negated && escape.is_none() && !*case_insensitive {
                if let Expr::Column(col_name) = strip_alias(&new_inner) {
                    // `predicate_rewrite_allowed` is false when the query
                    // projects this column as a bare Utf8 output: the
                    // integer-index filter can't emit the surviving Utf8 rows,
                    // so keep the real `Expr::Like` for the per-row
                    // `StringLikeFilter` / host path. See the trait method doc.
                    if r.knows(col_name) && r.predicate_rewrite_allowed(col_name) {
                        if let Some(indices) =
                            r.like_match_indices(col_name, pattern, *escape)
                        {
                            let mangled = r.index_column_name(col_name);
                            return Ok(build_index_membership(&mangled, &indices));
                        }
                    }
                }
            }

            Ok(Expr::Like {
                expr: Box::new(new_inner),
                pattern: pattern.clone(),
                escape: *escape,
                negated: *negated,
                case_insensitive: *case_insensitive,
            })
        }
        Expr::ScalarFn { kind, args } => {
            // String scalar functions don't get folded by the dictionary
            // rewriter (their output isn't a registered Utf8 column), but
            // we still walk every argument so any nested `col = 'lit'`
            // shapes inside them are normalised before they reach the
            // physical-plan boundary (which currently rejects ScalarFn
            // outright, but the rewrite is structurally consistent).
            let mut new_args = Vec::with_capacity(args.len());
            for a in args {
                new_args.push(rewrite_expr_with(a, r, depth + 1)?);
            }
            Ok(Expr::ScalarFn {
                kind: *kind,
                args: new_args,
            })
        }
        Expr::Extract { field, expr: inner } => {
            // EXTRACT operates on a Date32 / Timestamp operand — never a
            // registered Utf8 column — so there is nothing for the dictionary
            // rewriter to fold here. Recurse for structural consistency.
            let new_inner = rewrite_expr_with(inner, r, depth + 1)?;
            Ok(Expr::Extract {
                field: *field,
                expr: Box::new(new_inner),
            })
        }
        Expr::DateTrunc { unit, expr: inner } => {
            let new_inner = rewrite_expr_with(inner, r, depth + 1)?;
            Ok(Expr::DateTrunc {
                unit: *unit,
                expr: Box::new(new_inner),
            })
        }
        // Subquery nodes carry a self-contained `LogicalPlan` against a
        // *different* schema; this resolver `r` is keyed to the enclosing
        // query's Utf8 columns, so descending into the subplan here would be
        // both incorrect and unnecessary. The subquery is lowered/rejected at
        // the physical-plan boundary as a unit. Pass through unchanged.
        Expr::ScalarSubquery(_) => Ok(expr.clone()),
        Expr::InSubquery {
            expr: probe,
            subquery,
            negated,
        } => {
            // The probe expression *is* in the enclosing query's schema, so
            // normalise it; the subquery plan is left untouched (see above).
            let new_probe = rewrite_expr_with(probe, r, depth + 1)?;
            Ok(Expr::InSubquery {
                expr: Box::new(new_probe),
                subquery: subquery.clone(),
                negated: *negated,
            })
        }
    }
}

/// Pass-through walker that does NOT rewrite aggregate / group-by
/// expressions but DOES rewrite filter / project expressions on the way
/// down.
///
/// `depth` is the current recursion depth; returns Err if MAX_RECURSION_DEPTH is exceeded.
fn rewrite_plan_with<R: LiteralResolver>(
    plan: &LogicalPlan,
    r: &R,
    depth: usize,
) -> BoltResult<LogicalPlan> {
    if depth > crate::plan::sql_frontend::MAX_RECURSION_DEPTH {
        return Err(BoltError::Plan(format!(
            "plan nesting exceeds depth limit ({})",
            crate::plan::sql_frontend::MAX_RECURSION_DEPTH
        )));
    }
    match plan {
        LogicalPlan::Scan {
            table,
            projection,
            schema,
        } => {
            // Extend the scan's schema with mangled index columns for every
            // registered Utf8 column that already appears in `schema`.
            // Leaves `projection` untouched: the user's SELECT list shouldn't
            // gain hidden columns, only the underlying scan does.
            let mut fields = schema.fields.clone();
            let existing: std::collections::HashSet<&str> =
                fields.iter().map(|f| f.name.as_str()).collect();
            // Collect the names of Utf8 columns present in the schema.
            let utf8_cols_present: Vec<String> = schema
                .fields
                .iter()
                .filter(|f| f.dtype == DataType::Utf8 && r.knows(&f.name))
                .map(|f| f.name.clone())
                .collect();
            // Index-column names + dtypes we need to append, in
            // deterministic order (schema order of their parent Utf8
            // columns). The dtype is queried per column from the resolver so
            // i32- and i64-indexed dictionaries get matching field widths.
            let mut to_add: Vec<(String, DataType)> = Vec::new();
            for orig in &utf8_cols_present {
                let mangled = r.index_column_name(orig);
                if !existing.contains(mangled.as_str())
                    && !to_add.iter().any(|(n, _)| n == &mangled)
                {
                    let dtype = r.index_dtype(orig);
                    to_add.push((mangled, dtype));
                }
                // Finding F12: declare the per-row rank column `__rank_<col>`
                // (always Int64 — the rank tables and the NULL sentinel are i64)
                // ONLY for columns a `col_a OP col_b` ordering comparison in this
                // plan actually ranks (see `needs_rank_column`, populated by the
                // pre-pass in `StringPredicateRewriter::rewrite`). Declaring it
                // lets the physical planner resolve the synthetic name (it
                // resolves every predicate column against this scan schema — see
                // `physical_plan.rs::Codegen::emit_column`); the executor
                // MATERIALISES it on demand as `rank_table[index[row]]`. It is
                // non-nullable: the value is always a defined i64 (real rank `>= 0`
                // or the `-1` NULL sentinel), so it carries no validity bitmap and
                // the rewriter's NULL-safe `>= 0` guard handles 3VL. Like
                // `__idx_<col>`, it never leaks into the query output (the scan
                // `projection` is untouched). Gating on `needs_rank_column` keeps
                // the common (no col-vs-col ordering) scan schema unchanged.
                if r.needs_rank_column(orig) {
                    let rank_name = rank_column_name(orig);
                    if !existing.contains(rank_name.as_str())
                        && !to_add.iter().any(|(n, _)| n == &rank_name)
                    {
                        to_add.push((rank_name, DataType::Int64));
                    }
                }
            }
            // Previously this branch hard-coded `Int32`, which silently
            // mismatched the rewritten predicate's literal width whenever the
            // resolver returned a `LiteralIndex::I64`. The registry-driven
            // path (`DictRegistry::extended_schema`) already declares the
            // correct per-column width upstream, so any pre-existing
            // `__idx_<col>` field is left untouched here. This branch only
            // fires when no upstream pass has added the field — and now
            // honours the resolver's per-column dtype so direct users of
            // `StringPredicateRewriter` (no registry) also get matching
            // widths on both sides of the predicate.
            for (mangled, dtype) in to_add {
                fields.push(Field::new(mangled, dtype, false));
            }
            Ok(LogicalPlan::Scan {
                table: table.clone(),
                projection: projection.clone(),
                schema: Schema::new(fields),
            })
        }
        LogicalPlan::Filter { input, predicate } => {
            let new_input = rewrite_plan_with(input, r, depth + 1)?;
            let new_predicate = rewrite_expr_with(predicate, r, 0)?;
            Ok(LogicalPlan::Filter {
                input: Box::new(new_input),
                predicate: new_predicate,
            })
        }
        LogicalPlan::Project { input, exprs } => {
            let new_input = rewrite_plan_with(input, r, depth + 1)?;
            let mut new_exprs = Vec::with_capacity(exprs.len());
            for e in exprs {
                new_exprs.push(rewrite_expr_with(e, r, 0)?);
            }
            Ok(LogicalPlan::Project {
                input: Box::new(new_input),
                exprs: new_exprs,
            })
        }
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
        } => {
            let new_input = rewrite_plan_with(input, r, depth + 1)?;
            // TODO: rewriting group_by / aggregate expressions over Utf8
            // columns would require a dictionary-aware GROUP BY codegen
            // path. For now we leave them untouched; callers that GROUP BY
            // a Utf8 column will hit the usual codegen restriction.
            Ok(LogicalPlan::Aggregate {
                input: Box::new(new_input),
                group_by: group_by.clone(),
                aggregates: aggregates.clone(),
            })
        }
        // Wave 7 variants: rewrite descendants, preserve structure. The
        // dictionary-encoded `__idx_<col>` extension is applied at every
        // Scan leaf — the wrappers are transparent for that purpose.
        LogicalPlan::Distinct { input } => {
            let new_input = rewrite_plan_with(input, r, depth + 1)?;
            Ok(LogicalPlan::Distinct {
                input: Box::new(new_input),
            })
        }
        LogicalPlan::Limit { input, limit, offset } => {
            let new_input = rewrite_plan_with(input, r, depth + 1)?;
            Ok(LogicalPlan::Limit {
                input: Box::new(new_input),
                limit: *limit,
                offset: *offset,
            })
        }
        LogicalPlan::Sort { input, sort_exprs } => {
            let new_input = rewrite_plan_with(input, r, depth + 1)?;
            Ok(LogicalPlan::Sort {
                input: Box::new(new_input),
                sort_exprs: sort_exprs.clone(),
            })
        }
        LogicalPlan::Window {
            input,
            window_exprs,
            partition_by,
            order_by,
        } => {
            // Transparent wrapper for the dictionary-rewrite pass: descend
            // into the input but leave the window spec untouched (window
            // functions over Utf8 columns aren't part of the dictionary
            // codegen path — this executor is host-side).
            let new_input = rewrite_plan_with(input, r, depth + 1)?;
            Ok(LogicalPlan::Window {
                input: Box::new(new_input),
                window_exprs: window_exprs.clone(),
                partition_by: partition_by.clone(),
                order_by: order_by.clone(),
            })
        }
        LogicalPlan::Union { inputs } => {
            let new_inputs = inputs
                .iter()
                .map(|inp| rewrite_plan_with(inp, r, depth + 1))
                .collect::<BoltResult<Vec<_>>>()?;
            Ok(LogicalPlan::Union { inputs: new_inputs })
        }
        LogicalPlan::Join { left, right, join_type, on, filter } => {
            let new_left = rewrite_plan_with(left, r, depth + 1)?;
            let new_right = rewrite_plan_with(right, r, depth + 1)?;
            Ok(LogicalPlan::Join {
                left: Box::new(new_left),
                right: Box::new(new_right),
                join_type: *join_type,
                on: on.clone(),
                filter: filter.clone(),
            })
        }
        LogicalPlan::SetOp { left, right, op, all } => {
            let new_left = rewrite_plan_with(left, r, depth + 1)?;
            let new_right = rewrite_plan_with(right, r, depth + 1)?;
            Ok(LogicalPlan::SetOp {
                left: Box::new(new_left),
                right: Box::new(new_right),
                op: *op,
                all: *all,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{col, lit};

    /// Index width hint for the mock resolver, so a single mock can fake
    /// either an i32- or i64-indexed dictionary without dragging CUDA in.
    #[derive(Debug, Clone, Copy)]
    enum MockWidth {
        I32,
        I64,
    }

    /// In-memory `LiteralResolver` for tests so we don't need to construct a
    /// real `DictionaryColumnAny` (which requires CUDA).
    ///
    /// Each entry is keyed by `(column, literal)` and stores the raw `i64`
    /// index plus the column's index width; on `resolve` we pack the result
    /// into the matching `LiteralIndex` variant.
    struct MockResolver {
        /// (column, literal) → dictionary index (raw i64; narrows to i32 if
        /// the column's width is `I32`).
        entries: HashMap<(String, String), i64>,
        /// columns the resolver "knows" (has a dictionary for), keyed to
        /// their index width.
        columns: HashMap<String, MockWidth>,
        /// columns whose dictionary is marked known-complete (gates the
        /// absent-literal constant fold — finding PL-M6). Empty by default, so
        /// the mock mirrors the production "incomplete unless proven" stance.
        complete: std::collections::HashSet<String>,
        /// Per-column dictionary entries (real strings only; slot 0 = NULL is
        /// implicit and never stored). `dict[column][p]` is the string for GPU
        /// index `p + 1`, mirroring the production `DictionaryColumnAny`
        /// layout. Used to exercise `like_match_indices` without CUDA.
        dict_entries: HashMap<String, Vec<String>>,
        /// columns whose string predicates are protected from the integer fold
        /// (mirrors `StringPredicateRewriter::protect_predicate`). Empty by
        /// default so the mock keeps folding unless a test opts a column in.
        protected: std::collections::HashSet<String>,
    }

    impl MockResolver {
        fn new() -> Self {
            Self {
                entries: HashMap::new(),
                columns: HashMap::new(),
                complete: std::collections::HashSet::new(),
                dict_entries: HashMap::new(),
                protected: std::collections::HashSet::new(),
            }
        }

        /// Attach a dictionary entry list to `col` so `like_match_indices` can
        /// scan it. Entries are real strings (slot 0 / NULL is implicit); the
        /// GPU index of `entries[p]` is `p + 1`. Registers the column as known
        /// (i32-indexed) if it isn't already.
        fn with_dict(mut self, col: &str, entries: &[&str]) -> Self {
            self.columns
                .entry(col.to_string())
                .or_insert(MockWidth::I32);
            self.dict_entries.insert(
                col.to_string(),
                entries.iter().map(|s| s.to_string()).collect(),
            );
            self
        }

        /// Mark `col`'s dictionary as known-complete so absent literals are
        /// allowed to constant-fold. Mirrors
        /// `StringPredicateRewriter::mark_complete`.
        fn complete(mut self, col: &str) -> Self {
            self.complete.insert(col.to_string());
            self
        }

        /// Register `col` as i32-indexed and map `lit` → `idx`.
        fn with_i32(mut self, col: &str, lit: &str, idx: i32) -> Self {
            self.columns.insert(col.to_string(), MockWidth::I32);
            self.entries
                .insert((col.to_string(), lit.to_string()), idx as i64);
            self
        }

        /// Register `col` as i64-indexed and map `lit` → `idx`.
        fn with_i64(mut self, col: &str, lit: &str, idx: i64) -> Self {
            self.columns.insert(col.to_string(), MockWidth::I64);
            self.entries
                .insert((col.to_string(), lit.to_string()), idx);
            self
        }

        /// Mark `col` as known (i32-indexed) but with no literal entries.
        fn known_i32(mut self, col: &str) -> Self {
            self.columns.insert(col.to_string(), MockWidth::I32);
            self
        }

        /// Mark `col` as known (i64-indexed) but with no literal entries.
        fn known_i64(mut self, col: &str) -> Self {
            self.columns.insert(col.to_string(), MockWidth::I64);
            self
        }
    }

    impl LiteralResolver for MockResolver {
        fn resolve(&self, column: &str, literal: &str) -> Option<LiteralIndex> {
            let raw = self
                .entries
                .get(&(column.to_string(), literal.to_string()))
                .copied()?;
            let width = self.columns.get(column).copied()?;
            Some(match width {
                MockWidth::I32 => LiteralIndex::I32(raw as i32),
                MockWidth::I64 => LiteralIndex::I64(raw),
            })
        }

        fn index_column_name(&self, original: &str) -> String {
            super::index_column_name(original)
        }

        fn knows(&self, column: &str) -> bool {
            self.columns.contains_key(column)
        }

        fn predicate_rewrite_allowed(&self, column: &str) -> bool {
            !self.protected.contains(column)
        }

        fn like_match_indices(
            &self,
            column: &str,
            pattern: &str,
            escape: Option<char>,
        ) -> Option<Vec<LiteralIndex>> {
            let entries = self.dict_entries.get(column)?;
            let width = self.columns.get(column).copied()?;
            let matcher = crate::exec::like::PatternMatcher::compile(pattern, escape).ok()?;
            let mut out = Vec::new();
            for (p, s) in entries.iter().enumerate() {
                if matcher.matches(s) {
                    let idx = (p + 1) as i64;
                    out.push(match width {
                        MockWidth::I32 => LiteralIndex::I32(idx as i32),
                        MockWidth::I64 => LiteralIndex::I64(idx),
                    });
                }
            }
            Some(out)
        }

        fn ordering_match_indices(
            &self,
            column: &str,
            op: BinaryOp,
            literal: &str,
        ) -> Option<Vec<LiteralIndex>> {
            let entries = self.dict_entries.get(column)?;
            let width = self.columns.get(column).copied()?;
            let keep = |s: &str| -> bool {
                match op {
                    BinaryOp::Lt => s < literal,
                    BinaryOp::LtEq => s <= literal,
                    BinaryOp::Gt => s > literal,
                    BinaryOp::GtEq => s >= literal,
                    _ => false,
                }
            };
            let mut out = Vec::new();
            for (p, s) in entries.iter().enumerate() {
                if keep(s) {
                    let idx = (p + 1) as i64;
                    out.push(match width {
                        MockWidth::I32 => LiteralIndex::I32(idx as i32),
                        MockWidth::I64 => LiteralIndex::I64(idx),
                    });
                }
            }
            Some(out)
        }

        fn col_vs_col_rank_maps(
            &self,
            col_a: &str,
            col_b: &str,
        ) -> Option<ColVsColRankPlan> {
            // Mirror the production path: both columns must have scannable
            // dictionary entries; rank both against the shared sorted union.
            let dict_a = self.dict_entries.get(col_a)?;
            let dict_b = self.dict_entries.get(col_b)?;
            let (rank_a, rank_b) =
                crate::cuda::dictionary::unified_rank_maps_of(dict_a, dict_b);
            Some(ColVsColRankPlan {
                rank_a,
                rank_b,
                rank_col_a: super::rank_column_name(col_a),
                rank_col_b: super::rank_column_name(col_b),
            })
        }

        fn is_complete(&self, column: &str) -> bool {
            self.complete.contains(column)
        }

        fn index_dtype(&self, column: &str) -> DataType {
            // Mirror the column's registered width so the mock behaves like
            // the real `StringPredicateRewriter`. Unknown columns fall back
            // to the trait default (`Int32`); the rewriter never asks about
            // an unknown column in practice, so this branch is defensive
            // only.
            match self.columns.get(column).copied() {
                Some(MockWidth::I32) => DataType::Int32,
                Some(MockWidth::I64) => DataType::Int64,
                None => DataType::Int32,
            }
        }
    }

    fn assert_int32_lit(e: &Expr, expected: i32) {
        match e {
            Expr::Literal(Literal::Int32(n)) => assert_eq!(*n, expected),
            other => panic!("expected Int32 literal {expected}, got {other:?}"),
        }
    }

    fn assert_int64_lit(e: &Expr, expected: i64) {
        match e {
            Expr::Literal(Literal::Int64(n)) => assert_eq!(*n, expected),
            other => panic!("expected Int64 literal {expected}, got {other:?}"),
        }
    }

    fn assert_column(e: &Expr, expected: &str) {
        match e {
            Expr::Column(n) => assert_eq!(n, expected),
            other => panic!("expected Column({expected}), got {other:?}"),
        }
    }

    #[test]
    fn literal_index_into_literal_round_trip() {
        // Sanity-check the LiteralIndex → Literal mapping.
        let i32_lit = LiteralIndex::I32(7).into_literal();
        assert_eq!(i32_lit, Literal::Int32(7));
        assert_eq!(LiteralIndex::I32(7).dtype(), DataType::Int32);

        let i64_lit = LiteralIndex::I64(9_000_000_000).into_literal();
        assert_eq!(i64_lit, Literal::Int64(9_000_000_000));
        assert_eq!(LiteralIndex::I64(9_000_000_000).dtype(), DataType::Int64);
    }

    #[test]
    fn rewrite_eq_with_known_literal() {
        let r = MockResolver::new().with_i32("region", "US", 5);
        let expr = col("region").eq(lit("US"));
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        match out {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                assert_column(&left, "__idx_region");
                assert_int32_lit(&right, 5);
            }
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    /// Regression: an i32-indexed dictionary still emits `Literal::Int32`.
    #[test]
    fn i32_dict_still_emits_int32_literal() {
        let r = MockResolver::new().with_i32("region", "US", 5);
        let expr = col("region").eq(lit("US"));
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        match out {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                assert_column(&left, "__idx_region");
                assert_int32_lit(&right, 5);
            }
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    /// New: an i64-indexed dictionary emits `Literal::Int64`.
    #[test]
    fn i64_dict_emits_int64_literal() {
        let r = MockResolver::new().with_i64("region", "US", 5);
        let expr = col("region").eq(lit("US"));
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        match out {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                assert_column(&left, "__idx_region");
                assert_int64_lit(&right, 5);
            }
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    /// New: two Utf8 columns with different index widths produce matching
    /// per-column literal dtypes in a single predicate tree.
    #[test]
    fn mixed_columns_each_get_own_dtype() {
        let r = MockResolver::new()
            .with_i32("region", "US", 5)
            .with_i64("user_id", "alice", 1_234_567_890_123);
        // `region = 'US' AND user_id = 'alice'`
        let expr = col("region")
            .eq(lit("US"))
            .and(col("user_id").eq(lit("alice")));
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        let Expr::Binary { op: and_op, left, right } = out else {
            panic!("expected top-level AND");
        };
        assert_eq!(and_op, BinaryOp::And);

        // Left: __idx_region = Int32(5).
        match *left {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                assert_column(&left, "__idx_region");
                assert_int32_lit(&right, 5);
            }
            other => panic!("expected Eq for region, got {other:?}"),
        }
        // Right: __idx_user_id = Int64(1_234_567_890_123).
        match *right {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                assert_column(&left, "__idx_user_id");
                assert_int64_lit(&right, 1_234_567_890_123);
            }
            other => panic!("expected Eq for user_id, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_eq_with_unknown_literal_folds_to_false() {
        // Column is known, literal not in the dictionary, AND the dictionary
        // is marked known-complete — so the absent-literal fold is sound.
        let r = MockResolver::new().known_i32("region").complete("region");
        let expr = col("region").eq(lit("ZZ"));
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        match out {
            Expr::Literal(Literal::Bool(false)) => {}
            other => panic!("expected Bool(false), got {other:?}"),
        }
    }

    /// New: the unknown-literal fold-to-Bool path is dictionary-width
    /// agnostic — an i64-indexed column folds the same way.
    #[test]
    fn unknown_literal_still_folds_to_bool() {
        // i32 side. Both columns are marked known-complete so the absent-
        // literal fold is sound.
        let r32 = MockResolver::new().known_i32("region").complete("region");
        let eq = rewrite_expr_with(&col("region").eq(lit("ZZ")), &r32, 0).unwrap();
        assert!(matches!(eq, Expr::Literal(Literal::Bool(false))));
        let neq = rewrite_expr_with(&col("region").neq(lit("ZZ")), &r32, 0).unwrap();
        assert!(matches!(neq, Expr::Literal(Literal::Bool(true))));

        // i64 side: same behaviour.
        let r64 = MockResolver::new().known_i64("user_id").complete("user_id");
        let eq64 = rewrite_expr_with(&col("user_id").eq(lit("ghost")), &r64, 0).unwrap();
        assert!(matches!(eq64, Expr::Literal(Literal::Bool(false))));
        let neq64 = rewrite_expr_with(&col("user_id").neq(lit("ghost")), &r64, 0).unwrap();
        assert!(matches!(neq64, Expr::Literal(Literal::Bool(true))));
    }

    #[test]
    fn rewrite_neq_with_unknown_literal_folds_to_true() {
        let r = MockResolver::new().known_i32("region").complete("region");
        let expr = col("region").neq(lit("ZZ"));
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        match out {
            Expr::Literal(Literal::Bool(true)) => {}
            other => panic!("expected Bool(true), got {other:?}"),
        }
    }

    #[test]
    fn rewrite_reversed_literal_on_left() {
        let r = MockResolver::new().with_i32("region", "US", 5);
        let expr = lit("US").eq(col("region"));
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        match out {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                assert_column(&left, "__idx_region");
                assert_int32_lit(&right, 5);
            }
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    #[test]
    fn pass_through_non_string_predicate() {
        let r = MockResolver::new().with_i32("region", "US", 5);
        let expr = col("price").gt(lit(100.0_f64));
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        match out {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Gt);
                assert_column(&left, "price");
                match *right {
                    Expr::Literal(Literal::Float64(v)) => assert_eq!(v, 100.0),
                    other => panic!("expected Float64 literal, got {other:?}"),
                }
            }
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    /// F10: ordering over a registered column with NO scannable dictionary
    /// entries (the resolver can't partition it) is no longer a hard error —
    /// the resolver declines and the original comparison is preserved for the
    /// host path. (The `with_i32` mock registers a literal→index map but no
    /// `dict_entries`, so `ordering_match_indices` returns `None`.)
    #[test]
    fn ordering_without_dict_entries_preserves_comparison() {
        let r = MockResolver::new().with_i32("region", "US", 5);
        let expr = col("region").lt(lit("US"));
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        match out {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Lt);
                assert_column(&left, "region");
                match *right {
                    Expr::Literal(Literal::Utf8(s)) => assert_eq!(s, "US"),
                    other => panic!("expected preserved Utf8 literal, got {other:?}"),
                }
            }
            other => panic!("expected preserved Binary, got {other:?}"),
        }
    }

    #[test]
    fn scan_schema_extended_with_index_column() {
        let r = MockResolver::new().with_i32("region", "US", 5);
        let schema = Schema::new(vec![
            Field::new("region", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
        ]);
        let plan = LogicalPlan::Scan {
            table: "orders".into(),
            projection: None,
            schema,
        };
        let out = rewrite_plan_with(&plan, &r, 0).unwrap();
        match out {
            LogicalPlan::Scan { schema, .. } => {
                let names: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
                assert_eq!(names, vec!["region", "price", "__idx_region"]);
                let idx_field = schema.field("__idx_region").unwrap();
                assert_eq!(idx_field.dtype, DataType::Int32);
                assert!(!idx_field.nullable);
            }
            other => panic!("expected Scan, got {other:?}"),
        }
    }

    #[test]
    fn scan_schema_not_double_extended() {
        // If the engine already added the index column, the rewriter
        // should leave the schema alone.
        let r = MockResolver::new().with_i32("region", "US", 5);
        let schema = Schema::new(vec![
            Field::new("region", DataType::Utf8, false),
            Field::new("__idx_region", DataType::Int32, false),
        ]);
        let plan = LogicalPlan::Scan {
            table: "orders".into(),
            projection: None,
            schema,
        };
        let out = rewrite_plan_with(&plan, &r, 0).unwrap();
        match out {
            LogicalPlan::Scan { schema, .. } => {
                assert_eq!(schema.fields.len(), 2);
            }
            other => panic!("expected Scan, got {other:?}"),
        }
    }

    #[test]
    fn rewrite_combined_predicate_and_scan() {
        // `WHERE region = 'US' AND price > 100.0` — worked example from
        // the design doc. The Utf8 half is rewritten; the numeric half is
        // left alone; the Scan schema picks up `__idx_region`.
        let r = MockResolver::new().with_i32("region", "US", 5);
        let schema = Schema::new(vec![
            Field::new("region", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
        ]);
        let scan = LogicalPlan::Scan {
            table: "orders".into(),
            projection: None,
            schema,
        };
        let predicate = col("region")
            .eq(lit("US"))
            .and(col("price").gt(lit(100.0_f64)));
        let plan = LogicalPlan::Filter {
            input: Box::new(scan),
            predicate,
        };
        let out = rewrite_plan_with(&plan, &r, 0).unwrap();
        let LogicalPlan::Filter { input, predicate } = out else {
            panic!("expected Filter at root");
        };
        // Scan schema extended.
        match *input {
            LogicalPlan::Scan { schema, .. } => {
                let names: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
                assert_eq!(names, vec!["region", "price", "__idx_region"]);
            }
            other => panic!("expected Scan under Filter, got {other:?}"),
        }
        // Top-level AND survives; left side rewritten, right side untouched.
        let Expr::Binary { op: and_op, left, right } = predicate else {
            panic!("expected Binary AND");
        };
        assert_eq!(and_op, BinaryOp::And);
        match *left {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                assert_column(&left, "__idx_region");
                assert_int32_lit(&right, 5);
            }
            other => panic!("expected rewritten Eq, got {other:?}"),
        }
        match *right {
            Expr::Binary { op, left, right: r2 } => {
                assert_eq!(op, BinaryOp::Gt);
                assert_column(&left, "price");
                match *r2 {
                    Expr::Literal(Literal::Float64(v)) => assert_eq!(v, 100.0),
                    other => panic!("expected Float64, got {other:?}"),
                }
            }
            other => panic!("expected Gt, got {other:?}"),
        }
    }

    #[test]
    fn unregistered_column_is_left_alone() {
        // Column not registered as Utf8 with a dictionary: the rewriter
        // shouldn't touch the predicate. (The downstream type checker will
        // reject Utf8 == Utf8 as appropriate.)
        let r = MockResolver::new(); // no columns known
        let expr = col("name").eq(lit("Alice"));
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        match out {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                assert_column(&left, "name");
                match *right {
                    Expr::Literal(Literal::Utf8(s)) => assert_eq!(s, "Alice"),
                    other => panic!("expected Utf8 literal, got {other:?}"),
                }
            }
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    #[test]
    fn nested_predicate_inside_alias() {
        // Make sure recursion descends through Alias.
        let r = MockResolver::new().with_i32("region", "US", 5);
        let expr = col("region").eq(lit("US")).alias("is_us");
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        match out {
            Expr::Alias(inner, name) => {
                assert_eq!(name, "is_us");
                match *inner {
                    Expr::Binary { op, left, right } => {
                        assert_eq!(op, BinaryOp::Eq);
                        assert_column(&left, "__idx_region");
                        assert_int32_lit(&right, 5);
                    }
                    other => panic!("expected rewritten Eq inside Alias, got {other:?}"),
                }
            }
            other => panic!("expected Alias, got {other:?}"),
        }
    }

    /// Review C10: when the engine rebuilds the dict registry across all
    /// registered batches (so the registry holds the union dict), the
    /// rewriter must resolve a literal that lives only in an appended
    /// batch — not constant-fold it to `Bool(false)`.
    ///
    /// This is the host-side counterpart to
    /// `c10_register_batch_unions_dictionaries_across_batches` in
    /// `engine.rs`. We can't construct a real `DictionaryColumnAny` without
    /// CUDA, but the rewriter sees the union dict purely through the
    /// `LiteralResolver` trait — so a `MockResolver` populated with the
    /// expected union (entries from both "batches") is the exact same input
    /// the post-fix engine produces. If the resolver knew only batch 0's
    /// values, the predicate would fold to `Bool(false)`; the assertion
    /// below would fail and surface the silent-wrong-result regression.
    #[test]
    fn c10_rewriter_resolves_literal_from_unioned_dict() {
        // Pre-fix engine state: registry holds only batch 0 → resolver
        // knows {"a": 1, "b": 2}. This dictionary is NOT complete (batch 1's
        // "c" was never observed). With the PL-M6 fix the rewriter no longer
        // folds the absent literal to `Bool(false)` — instead it preserves the
        // original `s = 'c'` string comparison so the host path can still match
        // real "c" rows. (Before PL-M6 this folded to `Bool(false)`: a silent
        // wrong result whenever batch 0 was a partial snapshot.)
        let pre_fix = MockResolver::new()
            .with_i32("s", "a", 1)
            .with_i32("s", "b", 2);
        let unfolded =
            rewrite_expr_with(&col("s").eq(lit("c")), &pre_fix, 0).unwrap();
        match unfolded {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                // Original Utf8 comparison preserved — NOT an index rewrite and
                // NOT a constant fold.
                assert_column(&left, "s");
                match *right {
                    Expr::Literal(Literal::Utf8(s)) => assert_eq!(s, "c"),
                    other => panic!("expected preserved Utf8 literal 'c', got {other:?}"),
                }
            }
            other => panic!(
                "incomplete dict must NOT fold absent literal to a constant, got {other:?}"
            ),
        }

        // Post-fix engine state: registry rebuilt from union of batch 0
        // (dict {"a","b"}) and batch 1 (dict {"a","b","c"}); resolver
        // knows {"a": 1, "b": 2, "c": 3}.
        let post_fix = MockResolver::new()
            .with_i32("s", "a", 1)
            .with_i32("s", "b", 2)
            .with_i32("s", "c", 3);
        let rewritten =
            rewrite_expr_with(&col("s").eq(lit("c")), &post_fix, 0).unwrap();
        match rewritten {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                assert_column(&left, "__idx_s");
                assert_int32_lit(&right, 3);
            }
            other => panic!(
                "post-fix: union dict must resolve 'c' to its index, got {other:?}"
            ),
        }
    }

    // ---- finding PL-M6: completeness-gated absent-literal fold ----

    /// PL-M6: a literal that IS in the dictionary folds to its index
    /// regardless of completeness. The fast path is exact and must be
    /// unaffected by the gating. (Here the column is deliberately left
    /// *incomplete*.)
    #[test]
    fn plm6_present_literal_still_folds_to_index_when_incomplete() {
        let r = MockResolver::new().with_i32("region", "US", 5); // not complete
        assert!(!r.is_complete("region"));
        let out = rewrite_expr_with(&col("region").eq(lit("US")), &r, 0).unwrap();
        match out {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                assert_column(&left, "__idx_region");
                assert_int32_lit(&right, 5);
            }
            other => panic!("expected index fold for present literal, got {other:?}"),
        }
    }

    /// PL-M6: a literal that is absent from a *known-complete* dictionary
    /// folds to a constant (`=`→false, `<>`→true). The optimisation is kept
    /// where it is provably sound.
    #[test]
    fn plm6_absent_literal_with_complete_dict_folds_to_constant() {
        let r = MockResolver::new()
            .with_i32("region", "US", 5)
            .complete("region");
        let eq = rewrite_expr_with(&col("region").eq(lit("ZZ")), &r, 0).unwrap();
        assert!(
            matches!(eq, Expr::Literal(Literal::Bool(false))),
            "absent literal on a complete dict: `=` folds to false, got {eq:?}"
        );
        let neq = rewrite_expr_with(&col("region").neq(lit("ZZ")), &r, 0).unwrap();
        assert!(
            matches!(neq, Expr::Literal(Literal::Bool(true))),
            "absent literal on a complete dict: `<>` folds to true, got {neq:?}"
        );
    }

    /// PL-M6 (the bug fix): a literal absent from a dictionary that is NOT
    /// known-complete must NOT fold to a constant — it could be a real value
    /// the partial/sampled dictionary never observed. The rewriter leaves the
    /// original `col <op> 'lit'` Utf8 comparison intact for correct host-side
    /// evaluation (no index rewrite, no fold).
    #[test]
    fn plm6_absent_literal_without_completeness_is_not_folded() {
        let r = MockResolver::new().with_i32("region", "US", 5); // not complete
        assert!(!r.is_complete("region"));

        // `=` is preserved as a Utf8 comparison.
        let eq = rewrite_expr_with(&col("region").eq(lit("ZZ")), &r, 0).unwrap();
        match eq {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                assert_column(&left, "region");
                match *right {
                    Expr::Literal(Literal::Utf8(s)) => assert_eq!(s, "ZZ"),
                    other => panic!("expected preserved Utf8 literal, got {other:?}"),
                }
            }
            other => panic!("absent literal w/o completeness must NOT fold, got {other:?}"),
        }

        // `<>` likewise preserved (it must not become Bool(true)).
        let neq = rewrite_expr_with(&col("region").neq(lit("ZZ")), &r, 0).unwrap();
        match neq {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::NotEq);
                assert_column(&left, "region");
                match *right {
                    Expr::Literal(Literal::Utf8(s)) => assert_eq!(s, "ZZ"),
                    other => panic!("expected preserved Utf8 literal, got {other:?}"),
                }
            }
            other => panic!("absent literal w/o completeness must NOT fold, got {other:?}"),
        }
    }

    /// PL-M6: the empty-string edge case. Without a completeness guarantee,
    /// `col = ''` must be left for the host path rather than folded away, so a
    /// real empty-string row the dictionary never saw still matches.
    #[test]
    fn plm6_empty_string_not_folded_without_completeness() {
        let r = MockResolver::new().with_i32("name", "Alice", 1); // "" absent, not complete
        let out = rewrite_expr_with(&col("name").eq(lit("")), &r, 0).unwrap();
        match out {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                assert_column(&left, "name");
                match *right {
                    Expr::Literal(Literal::Utf8(s)) => assert_eq!(s, ""),
                    other => panic!("expected preserved empty Utf8 literal, got {other:?}"),
                }
            }
            other => panic!("empty-string eq must not fold without completeness, got {other:?}"),
        }
    }

    /// Regression: `extract_col_and_string_lit` must peel `Alias` wrappers
    /// off either operand. Without this, DataFrame-built predicates like
    /// `(col("region") AS "r").eq(lit("US"))` fall through the match arm
    /// and never fold, leaving a `Utf8 = Utf8` predicate that downstream
    /// codegen can't handle.
    #[test]
    fn peels_alias_in_eq() {
        let r = MockResolver::new().with_i32("region", "US", 5);

        // Baseline: plain `col = 'US'`.
        let plain = rewrite_expr_with(&col("region").eq(lit("US")), &r, 0).unwrap();

        // Aliased: `(col AS x) = 'US'` — must fold identically.
        let aliased_left = rewrite_expr_with(
            &col("region").alias("r").eq(lit("US")),
            &r,
            0,
        )
        .unwrap();
        assert_eq!(
            format!("{:?}", aliased_left),
            format!("{:?}", plain),
            "Alias on the column side should not block rewrite",
        );

        // Aliased on the literal side too: `col = (lit('US') AS x)`.
        let aliased_right = rewrite_expr_with(
            &col("region").eq(lit("US").alias("v")),
            &r,
            0,
        )
        .unwrap();
        assert_eq!(
            format!("{:?}", aliased_right),
            format!("{:?}", plain),
            "Alias on the literal side should not block rewrite",
        );

        // Aliased on both sides, reversed shape: `(lit AS l) = (col AS c)`.
        let aliased_both_reversed = rewrite_expr_with(
            &lit("US").alias("l").eq(col("region").alias("c")),
            &r,
            0,
        )
        .unwrap();
        assert_eq!(
            format!("{:?}", aliased_both_reversed),
            format!("{:?}", plain),
            "Reversed + both-aliased should also fold to canonical form",
        );

        // And confirm the canonical shape itself.
        match plain {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                assert_column(&left, "__idx_region");
                assert_int32_lit(&right, 5);
            }
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    /// Regression: when the resolver hands back `LiteralIndex::I64`, the
    /// scan-extension path must declare the `__idx_<col>` field as `Int64`,
    /// not the historical hard-coded `Int32`. Otherwise the literal width
    /// (chosen by the resolver) silently mismatches the scan column width
    /// (chosen by this rewriter) and the type checker surfaces the error
    /// far downstream of the actual cause.
    #[test]
    fn i64_dict_emits_int64_column() {
        let r = MockResolver::new().with_i64("user_id", "alice", 1_234_567_890_123);
        let schema = Schema::new(vec![
            Field::new("user_id", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
        ]);
        let plan = LogicalPlan::Scan {
            table: "events".into(),
            projection: None,
            schema,
        };
        let out = rewrite_plan_with(&plan, &r, 0).unwrap();
        match out {
            LogicalPlan::Scan { schema, .. } => {
                let names: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
                assert_eq!(names, vec!["user_id", "price", "__idx_user_id"]);
                let idx_field = schema.field("__idx_user_id").unwrap();
                // The key assertion: dtype matches the resolver's i64 width,
                // not the legacy Int32 default.
                assert_eq!(idx_field.dtype, DataType::Int64);
                assert!(!idx_field.nullable);
            }
            other => panic!("expected Scan, got {other:?}"),
        }

        // Cross-check: the rewritten predicate against the same column
        // really does carry an Int64 literal, so the scan-side dtype above
        // genuinely matches the predicate-side dtype below.
        let pred = rewrite_expr_with(&col("user_id").eq(lit("alice")), &r, 0).unwrap();
        match pred {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                assert_column(&left, "__idx_user_id");
                assert_int64_lit(&right, 1_234_567_890_123);
            }
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    // ---- GPU LIKE via dictionary-precompute ----

    /// Collect the `Int32` index literals out of an OR-of-equalities tree
    /// produced by `build_index_membership`, asserting the column name and
    /// `Eq`/`Or` shape along the way. Returns the indices in left-to-right
    /// (emission) order.
    fn collect_membership_i32(e: &Expr, column: &str) -> Vec<i32> {
        match e {
            // Leaf: `col = Int32(n)`.
            Expr::Binary { op: BinaryOp::Eq, left, right } => {
                assert_column(left, column);
                match right.as_ref() {
                    Expr::Literal(Literal::Int32(n)) => vec![*n],
                    other => panic!("expected Int32 index literal, got {other:?}"),
                }
            }
            // Interior: `<acc> OR <eq>`.
            Expr::Binary { op: BinaryOp::Or, left, right } => {
                let mut v = collect_membership_i32(left, column);
                v.extend(collect_membership_i32(right, column));
                v
            }
            other => panic!("expected Eq/Or membership tree, got {other:?}"),
        }
    }

    /// A constant prefix `LIKE` over a dict column builds the match table
    /// host-side and lowers to an OR-of-equalities on `__idx_<col>` — the
    /// GPU-lowerable integer-index lookup. Dictionary: ["alpha","beta",
    /// "alps","gamma"] (GPU indices 1..4); `LIKE 'al%'` matches "alpha"(1)
    /// and "alps"(3).
    #[test]
    fn like_prefix_over_dict_lowers_to_index_membership() {
        let r = MockResolver::new().with_dict("region", &["alpha", "beta", "alps", "gamma"]);
        let expr = Expr::Like {
            expr: Box::new(col("region")),
            pattern: "al%".into(),
            escape: None,
            negated: false,
            case_insensitive: false,
        };
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        let idxs = collect_membership_i32(&out, "__idx_region");
        assert_eq!(idxs, vec![1, 3], "al% matches alpha(1) and alps(3)");
    }

    /// A single-match pattern collapses to one bare `Eq` (no surrounding OR).
    #[test]
    fn like_single_match_is_one_equality() {
        let r = MockResolver::new().with_dict("region", &["alpha", "beta", "gamma"]);
        let expr = Expr::Like {
            expr: Box::new(col("region")),
            pattern: "%eta".into(), // suffix: matches "beta"(2) only
            escape: None,
            negated: false,
            case_insensitive: false,
        };
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        match out {
            Expr::Binary { op: BinaryOp::Eq, left, right } => {
                assert_column(&left, "__idx_region");
                assert_int32_lit(&right, 2);
            }
            other => panic!("expected a single Eq, got {other:?}"),
        }
    }

    /// A pattern that matches no dictionary entry folds to `Bool(false)` —
    /// the predicate is unconditionally false, no GPU work needed.
    #[test]
    fn like_no_match_folds_to_false() {
        let r = MockResolver::new().with_dict("region", &["alpha", "beta"]);
        let expr = Expr::Like {
            expr: Box::new(col("region")),
            pattern: "zzz%".into(),
            escape: None,
            negated: false,
            case_insensitive: false,
        };
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        assert!(
            matches!(out, Expr::Literal(Literal::Bool(false))),
            "no dict entry matches → Bool(false), got {out:?}"
        );
    }

    /// `%` alone matches every (non-NULL) dictionary entry: the lowered
    /// membership set covers all real indices and excludes slot 0 (NULL).
    #[test]
    fn like_match_all_covers_every_entry_but_not_null() {
        let r = MockResolver::new().with_dict("region", &["a", "b", "c"]);
        let expr = Expr::Like {
            expr: Box::new(col("region")),
            pattern: "%".into(),
            escape: None,
            negated: false,
            case_insensitive: false,
        };
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        let idxs = collect_membership_i32(&out, "__idx_region");
        // Indices 1,2,3 — slot 0 (NULL) is intentionally never included.
        assert_eq!(idxs, vec![1, 2, 3]);
        assert!(!idxs.contains(&0), "NULL slot 0 must be excluded");
    }

    /// An i64-indexed dict column emits `Int64` index literals so the
    /// membership predicate matches the `__idx_<col>` column's width.
    #[test]
    fn like_over_i64_dict_emits_int64_indices() {
        let mut r = MockResolver::new();
        r.columns.insert("uid".into(), MockWidth::I64);
        r.dict_entries
            .insert("uid".into(), vec!["bob".into(), "bart".into(), "ann".into()]);
        let expr = Expr::Like {
            expr: Box::new(col("uid")),
            pattern: "b%".into(), // matches bob(1), bart(2)
            escape: None,
            negated: false,
            case_insensitive: false,
        };
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        // Walk the OR tree collecting Int64 literals.
        fn collect_i64(e: &Expr) -> Vec<i64> {
            match e {
                Expr::Binary { op: BinaryOp::Eq, left, right } => {
                    assert_column(left, "__idx_uid");
                    match right.as_ref() {
                        Expr::Literal(Literal::Int64(n)) => vec![*n],
                        other => panic!("expected Int64 index, got {other:?}"),
                    }
                }
                Expr::Binary { op: BinaryOp::Or, left, right } => {
                    let mut v = collect_i64(left);
                    v.extend(collect_i64(right));
                    v
                }
                other => panic!("expected Eq/Or, got {other:?}"),
            }
        }
        assert_eq!(collect_i64(&out), vec![1, 2]);
    }

    /// LIKE over a NON-dict (unregistered) column stays an `Expr::Like` —
    /// the physical planner then routes it to the host filter. No index
    /// rewrite, no fold.
    #[test]
    fn like_over_non_dict_column_stays_host() {
        let r = MockResolver::new(); // nothing registered
        let expr = Expr::Like {
            expr: Box::new(col("name")),
            pattern: "a%".into(),
            escape: None,
            negated: false,
            case_insensitive: false,
        };
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        match out {
            Expr::Like { expr: inner, pattern, escape, negated, case_insensitive } => {
                assert_column(&inner, "name");
                assert_eq!(pattern, "a%");
                assert!(escape.is_none());
                assert!(!negated);
                assert!(!case_insensitive);
            }
            other => panic!("non-dict LIKE must stay host-side, got {other:?}"),
        }
    }

    /// `NOT LIKE` stays host-side even over a dict column: the OR-of-
    /// equalities form can't express SQL three-valued NULL semantics for the
    /// negated case, so the predicate is preserved for the host filter.
    #[test]
    fn not_like_over_dict_stays_host() {
        let r = MockResolver::new().with_dict("region", &["alpha", "beta"]);
        let expr = Expr::Like {
            expr: Box::new(col("region")),
            pattern: "al%".into(),
            escape: None,
            negated: true,
            case_insensitive: false,
        };
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        match out {
            Expr::Like { negated, .. } => assert!(negated, "NOT LIKE preserved"),
            other => panic!("NOT LIKE must stay host-side, got {other:?}"),
        }
    }

    /// `LIKE ... ESCAPE` stays host-side over a dict column — the precompute
    /// path is gated to the no-escape shape (the host evaluator owns escape
    /// semantics).
    #[test]
    fn like_with_escape_over_dict_stays_host() {
        let r = MockResolver::new().with_dict("region", &["a%b", "axb"]);
        let expr = Expr::Like {
            expr: Box::new(col("region")),
            pattern: r"a\%b".into(),
            escape: Some('\\'),
            negated: false,
            case_insensitive: false,
        };
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        match out {
            Expr::Like { escape, .. } => {
                assert_eq!(escape, Some('\\'), "ESCAPE LIKE preserved for host path")
            }
            other => panic!("LIKE ESCAPE must stay host-side, got {other:?}"),
        }
    }

    /// The dict LIKE rewrite peels `Alias` wrappers off the operand, just
    /// like the eq-rewrite path, so `(col AS r) LIKE 'al%'` still lowers.
    #[test]
    fn like_peels_alias_on_operand() {
        let r = MockResolver::new().with_dict("region", &["alpha", "beta"]);
        let expr = Expr::Like {
            expr: Box::new(col("region").alias("r")),
            pattern: "al%".into(),
            escape: None,
            negated: false,
            case_insensitive: false,
        };
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        match out {
            Expr::Binary { op: BinaryOp::Eq, left, right } => {
                assert_column(&left, "__idx_region");
                assert_int32_lit(&right, 1); // alpha(1)
            }
            other => panic!("aliased operand should still lower, got {other:?}"),
        }
    }

    /// End-to-end through `rewrite_plan_with`: a `Filter { Scan }` whose
    /// predicate is `region LIKE 'al%'` rewrites the predicate to the index
    /// membership AND extends the scan schema with `__idx_region`. This is
    /// the shape the physical planner consumes — the rewritten Filter no
    /// longer carries an `Expr::Like`, so it is no longer forced to the host
    /// fallback.
    #[test]
    fn filter_like_over_dict_rewrites_predicate_and_extends_scan() {
        let r = MockResolver::new().with_dict("region", &["alpha", "beta", "alps"]);
        let schema = Schema::new(vec![
            Field::new("region", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
        ]);
        let scan = LogicalPlan::Scan {
            table: "orders".into(),
            projection: None,
            schema,
        };
        let predicate = Expr::Like {
            expr: Box::new(col("region")),
            pattern: "al%".into(),
            escape: None,
            negated: false,
            case_insensitive: false,
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(scan),
            predicate,
        };
        let out = rewrite_plan_with(&plan, &r, 0).unwrap();
        let LogicalPlan::Filter { input, predicate } = out else {
            panic!("expected Filter at root");
        };
        // Scan schema gained `__idx_region`.
        match *input {
            LogicalPlan::Scan { schema, .. } => {
                let names: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
                assert_eq!(names, vec!["region", "price", "__idx_region"]);
            }
            other => panic!("expected Scan under Filter, got {other:?}"),
        }
        // Predicate is now an integer-index membership — NOT an Expr::Like.
        assert!(
            !matches!(predicate, Expr::Like { .. }),
            "rewritten predicate must not be a LIKE (no host fallback)"
        );
        let idxs = collect_membership_i32(&predicate, "__idx_region");
        assert_eq!(idxs, vec![1, 3], "al% matches alpha(1) and alps(3)");
    }

    /// Exercise the *production* `StringPredicateRewriter::like_match_indices`
    /// (not the mock) against a host-only `DictionaryColumnAny`, confirming
    /// the real precompute scans the dictionary slice and builds the correct
    /// index set. `new_host_only` builds the wrapper without any GPU upload,
    /// so this runs on a CUDA-less machine.
    #[test]
    fn real_rewriter_builds_like_match_table_from_dictionary() {
        use crate::cuda::dictionary_any::DictionaryColumnAny;

        // Dictionary holds the real (non-NULL) strings; GPU index of
        // entry `p` is `p + 1`.  ["apple","banana","apricot"] → 1,2,3.
        let dict = DictionaryColumnAny::new_host_only(
            vec!["apple".into(), "banana".into(), "apricot".into()],
            3,
        )
        .expect("host-only dict");
        let mut rw = StringPredicateRewriter::new();
        rw.register("fruit", &dict);

        // `ap%` matches apple(1) and apricot(3) but not banana.
        let idxs = rw
            .like_match_indices("fruit", "ap%", None)
            .expect("dict column resolves a match table");
        assert_eq!(idxs, vec![LiteralIndex::I32(1), LiteralIndex::I32(3)]);

        // End-to-end: the rewriter lowers `fruit LIKE 'ap%'` to the index
        // membership predicate against `__idx_fruit`.
        let expr = Expr::Like {
            expr: Box::new(col("fruit")),
            pattern: "ap%".into(),
            escape: None,
            negated: false,
            case_insensitive: false,
        };
        let out = rewrite_expr_with(&expr, &rw, 0).unwrap();
        let collected = collect_membership_i32(&out, "__idx_fruit");
        assert_eq!(collected, vec![1, 3]);

        // A pattern absent from the dictionary yields an empty table → the
        // rewrite folds to Bool(false).
        let none = rw.like_match_indices("fruit", "zzz", None).unwrap();
        assert!(none.is_empty());

        // An unregistered column returns None (host fallback).
        assert!(rw.like_match_indices("unknown", "a%", None).is_none());
    }

    // ---- F10: byte-lexicographic ordering collation ----

    /// Collect the `Int32` index literals out of an ordering membership tree
    /// (or a single bare `Eq`) and return them sorted ascending. Shares the
    /// `Eq`/`Or` shape `build_index_membership` emits with the LIKE path.
    fn collect_membership_i32_sorted(e: &Expr, column: &str) -> Vec<i32> {
        let mut v = collect_membership_i32(e, column);
        v.sort_unstable();
        v
    }

    /// Direct (oracle) byte-lexicographic evaluation of `entry OP literal`
    /// over a dictionary, returning the GPU indices (1-based) that match —
    /// the ground truth the rewrite must reproduce.
    fn oracle_indices(entries: &[&str], op: BinaryOp, literal: &str) -> Vec<i32> {
        entries
            .iter()
            .enumerate()
            .filter(|(_, s)| match op {
                BinaryOp::Lt => **s < literal,
                BinaryOp::LtEq => **s <= literal,
                BinaryOp::Gt => **s > literal,
                BinaryOp::GtEq => **s >= literal,
                _ => unreachable!("oracle is ordering-only"),
            })
            .map(|(p, _)| (p as i32) + 1)
            .collect()
    }

    /// Rank computation over a known set of strings, via the real i32
    /// dictionary. Insertion order is deliberately NOT sorted, so the
    /// permutation is non-trivial. Byte order: "Zebra" < "apple" (uppercase
    /// 'Z' = 0x5A precedes lowercase 'a' = 0x61) — the binary-collation
    /// hallmark, distinct from a locale collation.
    #[test]
    fn f10_collation_ranks_known_set() {
        use crate::cuda::dictionary::DictionaryColumn;
        // Insertion slots:   0       1        2       3
        let dict = vec![
            "delta".to_string(),
            "apple".to_string(),
            "Zebra".to_string(),
            "mango".to_string(),
        ];
        let col = DictionaryColumn::new_host_only(dict, 0);
        // Sorted byte order: "Zebra"(slot2) < "apple"(slot1) < "delta"(slot0)
        // < "mango"(slot3). So ranks by insertion slot:
        //   slot0 "delta" -> rank 2
        //   slot1 "apple" -> rank 1
        //   slot2 "Zebra" -> rank 0
        //   slot3 "mango" -> rank 3
        assert_eq!(col.collation_ranks(), vec![2, 1, 0, 3]);
    }

    /// Insertion rank for present and absent literals (the half-open
    /// insertion point), via the real i32 dictionary.
    #[test]
    fn f10_insertion_rank_present_and_absent() {
        use crate::cuda::dictionary::DictionaryColumn;
        let dict = vec![
            "apple".to_string(),
            "delta".to_string(),
            "mango".to_string(),
        ]; // already sorted for clarity
        let col = DictionaryColumn::new_host_only(dict, 0);
        // Present literals: rank == count of strictly-smaller entries.
        assert_eq!(col.insertion_rank("apple"), 0);
        assert_eq!(col.insertion_rank("delta"), 1);
        assert_eq!(col.insertion_rank("mango"), 2);
        // Absent literals: half-open insertion point.
        assert_eq!(col.insertion_rank("aardvark"), 0); // before all
        assert_eq!(col.insertion_rank("cat"), 1); // between apple, delta
        assert_eq!(col.insertion_rank("zzz"), 3); // after all
    }

    /// `indices_satisfying` on the real i32 dictionary must agree with a
    /// direct byte comparison for every ordering op, for both a present and
    /// an absent literal.
    #[test]
    fn f10_indices_satisfying_matches_direct_comparison() {
        use crate::cuda::dictionary::DictionaryColumn;
        let entries = ["delta", "apple", "Zebra", "mango"];
        let dict: Vec<String> = entries.iter().map(|s| s.to_string()).collect();
        let col = DictionaryColumn::new_host_only(dict, 0);

        for &lit in &["mango", "cat", "Zebra", "zzz", "AAA"] {
            for op in [BinaryOp::Lt, BinaryOp::LtEq, BinaryOp::Gt, BinaryOp::GtEq] {
                let mut got = col.indices_satisfying(op, lit);
                got.sort_unstable();
                let mut want = oracle_indices(&entries, op, lit);
                want.sort_unstable();
                assert_eq!(
                    got, want,
                    "indices_satisfying({op:?}, {lit:?}) mismatch"
                );
            }
        }
    }

    /// End-to-end: `col < 'lit'` over a dict column lowers to the index
    /// membership of entries that sort before the literal, and the boolean
    /// result of the rewritten predicate matches a direct string comparison
    /// on sample data (each dictionary entry stands in for a sample row).
    #[test]
    fn f10_lt_rewrites_to_membership_and_matches_oracle() {
        let entries = ["delta", "apple", "Zebra", "mango"];
        let r = MockResolver::new().with_dict("region", &entries);
        let lit_val = "mango";
        let out = rewrite_expr_with(&col("region").lt(lit(lit_val)), &r, 0).unwrap();
        let got = collect_membership_i32_sorted(&out, "__idx_region");
        let mut want = oracle_indices(&entries, BinaryOp::Lt, lit_val);
        want.sort_unstable();
        assert_eq!(got, want);
        // "Zebra"(3) and "apple"(1) and "delta"(2) sort before "mango"; the
        // membership is exactly {1,2,3}. ("mango" itself excluded for strict.)
        assert_eq!(got, vec![1, 2, 3]);
    }

    /// All four ordering ops fold correctly through the rewriter, present and
    /// absent literal, and the resulting membership matches the oracle.
    #[test]
    fn f10_all_ops_present_and_absent_literal() {
        let entries = ["banana", "apple", "cherry"];
        let r = MockResolver::new().with_dict("fruit", &entries);
        for &lit_val in &["banana", "blueberry", "aaa", "zzz"] {
            for op in [BinaryOp::Lt, BinaryOp::LtEq, BinaryOp::Gt, BinaryOp::GtEq] {
                let expr = Expr::Binary {
                    op,
                    left: Box::new(col("fruit")),
                    right: Box::new(lit(lit_val)),
                };
                let out = rewrite_expr_with(&expr, &r, 0).unwrap();
                // Empty match set folds to Bool(false); otherwise a membership.
                let got = match &out {
                    Expr::Literal(Literal::Bool(false)) => Vec::new(),
                    other => collect_membership_i32_sorted(other, "__idx_fruit"),
                };
                let mut want = oracle_indices(&entries, op, lit_val);
                want.sort_unstable();
                assert_eq!(
                    got, want,
                    "op {op:?} lit {lit_val:?}: rewrite disagrees with oracle"
                );
            }
        }
    }

    /// Literal-on-the-left orientation (`'lit' < col`) must reflect the op so
    /// the partition is correct: `'mango' < col` ⇔ `col > 'mango'`.
    #[test]
    fn f10_reversed_literal_reflects_op() {
        let entries = ["delta", "apple", "Zebra", "mango", "pear"];
        let r = MockResolver::new().with_dict("region", &entries);
        // `'mango' < region`  ⇔  `region > 'mango'`.
        let expr = Expr::Binary {
            op: BinaryOp::Lt,
            left: Box::new(lit("mango")),
            right: Box::new(col("region")),
        };
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        let got = collect_membership_i32_sorted(&out, "__idx_region");
        let mut want = oracle_indices(&entries, BinaryOp::Gt, "mango");
        want.sort_unstable();
        assert_eq!(got, want, "reversed literal must reflect the op to '>'");
        // Only "pear" (slot 4, index 5) sorts strictly after "mango".
        assert_eq!(got, vec![5]);
    }

    /// A `>` predicate whose literal sorts after every entry yields an empty
    /// match set, which folds to Bool(false) — and crucially the NULL slot 0
    /// is never in any match set (a NULL row never passes an ordering pred).
    #[test]
    fn f10_no_match_folds_false_and_null_excluded() {
        let entries = ["apple", "banana"];
        let r = MockResolver::new().with_dict("region", &entries);
        let out = rewrite_expr_with(&col("region").gt(lit("zzz")), &r, 0).unwrap();
        assert!(
            matches!(out, Expr::Literal(Literal::Bool(false))),
            "nothing sorts after 'zzz' → Bool(false), got {out:?}"
        );

        // A predicate that matches everything still never includes slot 0.
        let all = rewrite_expr_with(&col("region").gt(lit("")), &r, 0).unwrap();
        let idxs = collect_membership_i32_sorted(&all, "__idx_region");
        assert_eq!(idxs, vec![1, 2]);
        assert!(!idxs.contains(&0), "NULL slot 0 must never be in the set");
    }

    /// Column-vs-column Utf8 ordering over two dictionary columns is now folded
    /// (finding F12) into a NULL-safe integer rank comparison on the synthetic
    /// `__rank_<col>` columns — no longer left verbatim for the host path. The
    /// `f12_*` tests below assert the full shape + cross-dictionary correctness;
    /// here we just confirm the col-vs-col case no longer stays a bare Utf8
    /// comparison on the original columns.
    #[test]
    fn f10_col_vs_col_is_folded_to_rank_comparison() {
        let r = MockResolver::new()
            .with_dict("a", &["x", "y"])
            .with_dict("b", &["x", "y"]);
        let expr = col("a").lt(col("b"));
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        let rendered = format!("{out:?}");
        assert!(
            rendered.contains("__rank_a") && rendered.contains("__rank_b"),
            "col-vs-col ordering must fold to a __rank_ comparison, got {out:?}"
        );
    }

    /// An ordering predicate over a column the query projects as a bare Utf8
    /// output is protected from the fold (the integer filter can't emit Utf8
    /// rows) — the comparison is preserved for the host path.
    #[test]
    fn f10_protected_column_is_not_folded() {
        let mut r = MockResolver::new().with_dict("region", &["apple", "mango"]);
        r.protected.insert("region".to_string());
        let out = rewrite_expr_with(&col("region").lt(lit("mango")), &r, 0).unwrap();
        match out {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Lt);
                assert_column(&left, "region");
                match *right {
                    Expr::Literal(Literal::Utf8(s)) => assert_eq!(s, "mango"),
                    other => panic!("expected preserved Utf8 literal, got {other:?}"),
                }
            }
            other => panic!("protected column must not fold, got {other:?}"),
        }
    }

    /// The production `StringPredicateRewriter::ordering_match_indices` over a
    /// host-only `DictionaryColumnAny` builds the correct set, and the
    /// end-to-end rewrite matches a direct comparison on the dictionary.
    #[test]
    fn f10_real_rewriter_orders_from_dictionary() {
        use crate::cuda::dictionary_any::DictionaryColumnAny;
        let entries = ["delta", "apple", "Zebra", "mango"];
        let dict = DictionaryColumnAny::new_host_only(
            entries.iter().map(|s| s.to_string()).collect(),
            4,
        )
        .expect("host-only dict");
        let mut rw = StringPredicateRewriter::new();
        rw.register("region", &dict);

        // `region <= 'mango'`: entries that sort <= "mango".
        let idxs = rw
            .ordering_match_indices("region", BinaryOp::LtEq, "mango")
            .expect("dict column resolves an ordering set");
        // Expect Int32 literals (i32 dict), sorted by GPU index.
        let mut got: Vec<i32> = idxs
            .iter()
            .map(|li| match li {
                LiteralIndex::I32(n) => *n,
                LiteralIndex::I64(n) => *n as i32,
            })
            .collect();
        got.sort_unstable();
        let mut want = oracle_indices(&entries, BinaryOp::LtEq, "mango");
        want.sort_unstable();
        assert_eq!(got, want);

        // End-to-end fold.
        let out = rewrite_expr_with(&col("region").lt_eq(lit("mango")), &rw, 0).unwrap();
        let collected = collect_membership_i32_sorted(&out, "__idx_region");
        assert_eq!(collected, want);

        // Unregistered column → None (host fallback).
        assert!(rw
            .ordering_match_indices("unknown", BinaryOp::Lt, "x")
            .is_none());
    }

    // ---- F12: column-vs-column Utf8 ordering (cross-dictionary ranks) ----

    /// Byte-string comparison oracle for two strings under all four ops.
    fn str_cmp_oracle(op: BinaryOp, a: &str, b: &str) -> bool {
        match op {
            BinaryOp::Lt => a < b,
            BinaryOp::LtEq => a <= b,
            BinaryOp::Gt => a > b,
            BinaryOp::GtEq => a >= b,
            _ => unreachable!("ordering-only oracle"),
        }
    }

    fn rank_cmp(op: BinaryOp, x: i64, y: i64) -> bool {
        match op {
            BinaryOp::Lt => x < y,
            BinaryOp::LtEq => x <= y,
            BinaryOp::Gt => x > y,
            BinaryOp::GtEq => x >= y,
            _ => unreachable!("ordering-only"),
        }
    }

    /// `plan_col_vs_col_rank` matches the `col_a OP col_b` shape over two dict
    /// columns and returns a cross-dictionary-correct rank plan: comparing the
    /// two columns' ranks reproduces the direct byte-string comparison for every
    /// row pairing and every ordering op, even though the dictionaries DIFFER.
    #[test]
    fn f12_plan_cross_dictionary_ranks_match_string_oracle() {
        let dict_a = ["delta", "apple", "mango"];
        let dict_b = ["cherry", "Zebra", "apple"];
        let r = MockResolver::new()
            .with_dict("a", &dict_a)
            .with_dict("b", &dict_b);

        for op in [BinaryOp::Lt, BinaryOp::LtEq, BinaryOp::Gt, BinaryOp::GtEq] {
            let (got_op, plan) =
                plan_col_vs_col_rank(op, &col("a"), &col("b"), &r).expect("plan built");
            assert_eq!(got_op, op);
            assert_eq!(plan.rank_col_a, "__rank_a");
            assert_eq!(plan.rank_col_b, "__rank_b");
            // Slot 0 is the NULL sentinel on both sides (SQL 3VL).
            assert_eq!(plan.rank_a[0], crate::cuda::dictionary::NULL_RANK_SENTINEL);
            assert_eq!(plan.rank_b[0], crate::cuda::dictionary::NULL_RANK_SENTINEL);
            // Every row pairing: rank comparison == string comparison.
            for (ai, a_s) in dict_a.iter().enumerate() {
                for (bi, b_s) in dict_b.iter().enumerate() {
                    let by_rank = rank_cmp(op, plan.rank_a[ai + 1], plan.rank_b[bi + 1]);
                    let by_str = str_cmp_oracle(op, a_s, b_s);
                    assert_eq!(
                        by_rank, by_str,
                        "op {op:?}: {a_s} vs {b_s} rank disagrees with string"
                    );
                }
            }
        }
    }

    /// Same-dictionary degenerate case still works: the two rank tables coincide
    /// and the comparison matches the string oracle.
    #[test]
    fn f12_plan_same_dictionary() {
        let dict = ["delta", "apple", "Zebra"];
        let r = MockResolver::new()
            .with_dict("a", &dict)
            .with_dict("b", &dict);
        let (_, plan) =
            plan_col_vs_col_rank(BinaryOp::Lt, &col("a"), &col("b"), &r).expect("plan");
        assert_eq!(plan.rank_a, plan.rank_b, "identical dicts → identical ranks");
        for (i, a_s) in dict.iter().enumerate() {
            for (j, b_s) in dict.iter().enumerate() {
                assert_eq!(
                    rank_cmp(BinaryOp::Lt, plan.rank_a[i + 1], plan.rank_b[j + 1]),
                    str_cmp_oracle(BinaryOp::Lt, a_s, b_s)
                );
            }
        }
    }

    /// Equality / non-ordering ops do NOT match the F12 ordering path (equality
    /// is already handled by index equality elsewhere).
    #[test]
    fn f12_plan_rejects_non_ordering_ops() {
        let r = MockResolver::new()
            .with_dict("a", &["x"])
            .with_dict("b", &["y"]);
        assert!(plan_col_vs_col_rank(BinaryOp::Eq, &col("a"), &col("b"), &r).is_none());
        assert!(plan_col_vs_col_rank(BinaryOp::NotEq, &col("a"), &col("b"), &r).is_none());
    }

    /// If EITHER column is not a registered dict column, the plan declines and
    /// the caller keeps the host string comparison.
    #[test]
    fn f12_plan_rejects_non_dict_column() {
        let r = MockResolver::new().with_dict("a", &["x", "y"]);
        // `b` is unknown.
        assert!(plan_col_vs_col_rank(BinaryOp::Lt, &col("a"), &col("b"), &r).is_none());
        // Both unknown.
        let empty = MockResolver::new();
        assert!(plan_col_vs_col_rank(BinaryOp::Lt, &col("a"), &col("b"), &empty).is_none());
    }

    /// A protected column (projected as bare Utf8 output) declines the rank
    /// plan, mirroring the F10 protection rule.
    #[test]
    fn f12_plan_rejects_protected_column() {
        let mut r = MockResolver::new()
            .with_dict("a", &["x", "y"])
            .with_dict("b", &["x", "y"]);
        r.protected.insert("a".to_string());
        assert!(plan_col_vs_col_rank(BinaryOp::Lt, &col("a"), &col("b"), &r).is_none());
    }

    /// `plan_col_vs_col_rank` peels Alias wrappers off both column operands.
    #[test]
    fn f12_plan_peels_alias() {
        let r = MockResolver::new()
            .with_dict("a", &["x", "y"])
            .with_dict("b", &["x", "y"]);
        let plan = plan_col_vs_col_rank(
            BinaryOp::Lt,
            &col("a").alias("l"),
            &col("b").alias("r"),
            &r,
        );
        assert!(plan.is_some(), "aliased columns must still match the shape");
    }

    /// Deconstruct the F12 rank comparison the live rewrite now emits:
    /// `(rank_a >= 0) AND (rank_b >= 0) AND (rank_a OP rank_b)`. Asserts the
    /// NULL-safe guard shape and returns `OP` so callers can check the op was
    /// preserved. Panics with a descriptive message on any structural mismatch.
    fn assert_rank_comparison(e: &Expr, rank_a: &str, rank_b: &str) -> BinaryOp {
        // Top: AND( AND(guard_a, guard_b), ordering ).
        let Expr::Binary { op: BinaryOp::And, left: guards, right: ordering } = e else {
            panic!("expected top-level AND, got {e:?}");
        };
        let Expr::Binary { op: BinaryOp::And, left: ga, right: gb } = &**guards else {
            panic!("expected AND of two NULL guards, got {guards:?}");
        };
        // guard_a: rank_a >= 0
        let check_guard = |g: &Expr, name: &str| {
            let Expr::Binary { op: BinaryOp::GtEq, left, right } = g else {
                panic!("expected `{name} >= 0` guard, got {g:?}");
            };
            assert_column(left, name);
            assert_int64_lit(right, 0);
        };
        check_guard(ga, rank_a);
        check_guard(gb, rank_b);
        // ordering: rank_a OP rank_b
        let Expr::Binary { op, left, right } = &**ordering else {
            panic!("expected ordering compare, got {ordering:?}");
        };
        assert_column(left, rank_a);
        assert_column(right, rank_b);
        *op
    }

    /// LIVE GPU rewrite (finding F12): `col_a OP col_b` over two dict columns now
    /// lowers to the NULL-safe rank comparison
    /// `(__rank_a >= 0) AND (__rank_b >= 0) AND (__rank_a OP __rank_b)` — not the
    /// preserved Utf8 comparison. The op is carried through for all four orders.
    #[test]
    fn f12_live_rewrite_emits_rank_comparison() {
        let r = MockResolver::new()
            .with_dict("a", &["x", "y"])
            .with_dict("b", &["x", "y"]);
        for op in [BinaryOp::Lt, BinaryOp::LtEq, BinaryOp::Gt, BinaryOp::GtEq] {
            let expr = Expr::Binary {
                op,
                left: Box::new(col("a")),
                right: Box::new(col("b")),
            };
            let out = rewrite_expr_with(&expr, &r, 0).unwrap();
            let got_op = assert_rank_comparison(&out, "__rank_a", "__rank_b");
            assert_eq!(got_op, op, "ordering op must be preserved");
        }
    }

    /// LIVE GPU rewrite across DIFFERENT dictionaries: the rank comparison is
    /// still emitted (cross-dictionary correctness is the rank tables' job; the
    /// emitted IR shape is identical — only the materialised values differ).
    #[test]
    fn f12_live_rewrite_emits_rank_comparison_cross_dictionary() {
        let r = MockResolver::new()
            .with_dict("a", &["delta", "apple", "mango"])
            .with_dict("b", &["cherry", "Zebra", "apple"]);
        let expr = Expr::Binary {
            op: BinaryOp::Lt,
            left: Box::new(col("a")),
            right: Box::new(col("b")),
        };
        let out = rewrite_expr_with(&expr, &r, 0).unwrap();
        let got_op = assert_rank_comparison(&out, "__rank_a", "__rank_b");
        assert_eq!(got_op, BinaryOp::Lt);
    }

    /// HOST FALLBACK: when a side is NOT a registered dict column the rank path
    /// declines, so the rewrite PRESERVES the original `col_a OP col_b` Utf8
    /// comparison for the always-correct host string path. Pins that the GPU
    /// rewrite never fires for an unbacked shape.
    #[test]
    fn f12_live_rewrite_falls_back_to_host_when_not_dict() {
        // `b` is unknown to the resolver.
        let r = MockResolver::new().with_dict("a", &["x", "y"]);
        for op in [BinaryOp::Lt, BinaryOp::LtEq, BinaryOp::Gt, BinaryOp::GtEq] {
            let expr = Expr::Binary {
                op,
                left: Box::new(col("a")),
                right: Box::new(col("b")),
            };
            let out = rewrite_expr_with(&expr, &r, 0).unwrap();
            match out {
                Expr::Binary { op: got_op, left, right } => {
                    assert_eq!(got_op, op);
                    assert_column(&left, "a");
                    assert_column(&right, "b");
                }
                other => panic!("unbacked col-vs-col must stay a Utf8 comparison, got {other:?}"),
            }
        }
    }

    /// END-TO-END host equivalence (finding F12): the emitted NULL-safe rank
    /// comparison, evaluated over materialised per-row ranks, yields exactly the
    /// same boolean as a direct byte-string comparison — including SQL 3VL for
    /// NULL rows (dict index 0 → rank `-1`, dropped by the `>= 0` guard) — across
    /// DIFFERENT dictionaries and all four ordering ops.
    #[test]
    fn f12_rank_comparison_matches_string_oracle_with_nulls() {
        let dict_a = ["delta", "apple", "mango"];
        let dict_b = ["cherry", "Zebra", "apple"];
        let r = MockResolver::new()
            .with_dict("a", &dict_a)
            .with_dict("b", &dict_b);

        // Sample rows as (Option<&str>, dict index) on each side. Index 0 is
        // NULL; index k>=1 is dict[k-1]. Include NULLs on each side and on both.
        // (string, gpu_index_a)
        let rows_a: &[(Option<&str>, i64)] = &[
            (Some("delta"), 1),
            (Some("apple"), 2),
            (Some("mango"), 3),
            (None, 0),
            (Some("apple"), 2),
        ];
        let rows_b: &[(Option<&str>, i64)] = &[
            (Some("cherry"), 1),
            (Some("Zebra"), 2),
            (Some("apple"), 3),
            (Some("apple"), 3),
            (None, 0),
        ];

        for op in [BinaryOp::Lt, BinaryOp::LtEq, BinaryOp::Gt, BinaryOp::GtEq] {
            // Rewrite once per op to get the emitted predicate.
            let expr = Expr::Binary {
                op,
                left: Box::new(col("a")),
                right: Box::new(col("b")),
            };
            let pred = rewrite_expr_with(&expr, &r, 0).unwrap();
            // Recover the unified rank tables the way the executor would.
            let plan = plan_col_vs_col_rank(op, &col("a"), &col("b"), &r).unwrap().1;

            for (&(sa, ia), &(sb, ib)) in rows_a.iter().zip(rows_b.iter()) {
                // Materialise per-row ranks: rank_table[gpu_index].
                let rank_a = plan.rank_a[ia as usize];
                let rank_b = plan.rank_b[ib as usize];
                // Evaluate the emitted integer predicate by hand (the same
                // semantics the GPU/host integer machinery applies).
                let by_pred = eval_rank_pred(&pred, rank_a, rank_b);
                // Oracle: SQL 3VL — NULL on either side ⇒ does not pass.
                let by_str = match (sa, sb) {
                    (Some(a), Some(b)) => str_cmp_oracle(op, a, b),
                    _ => false,
                };
                assert_eq!(
                    by_pred, by_str,
                    "op {op:?}: a={sa:?}(rank {rank_a}) b={sb:?}(rank {rank_b})"
                );
            }
        }
    }

    /// Evaluate the F12 emitted predicate shape
    /// `(rank_a >= 0) AND (rank_b >= 0) AND (rank_a OP rank_b)` for one row's
    /// materialised ranks. A tiny interpreter over exactly the IR
    /// `build_rank_comparison` produces — enough to prove host/GPU agreement
    /// without a CUDA device.
    fn eval_rank_pred(e: &Expr, rank_a: i64, rank_b: i64) -> bool {
        match e {
            Expr::Binary { op: BinaryOp::And, left, right } => {
                eval_rank_pred(left, rank_a, rank_b) && eval_rank_pred(right, rank_a, rank_b)
            }
            Expr::Binary { op, left, right } => {
                let lv = match &**left {
                    Expr::Column(n) if n == "__rank_a" => rank_a,
                    Expr::Column(n) if n == "__rank_b" => rank_b,
                    other => panic!("unexpected lhs {other:?}"),
                };
                let rv = match &**right {
                    Expr::Literal(Literal::Int64(v)) => *v,
                    Expr::Column(n) if n == "__rank_a" => rank_a,
                    Expr::Column(n) if n == "__rank_b" => rank_b,
                    other => panic!("unexpected rhs {other:?}"),
                };
                match op {
                    BinaryOp::Lt => lv < rv,
                    BinaryOp::LtEq => lv <= rv,
                    BinaryOp::Gt => lv > rv,
                    BinaryOp::GtEq => lv >= rv,
                    other => panic!("unexpected op {other:?}"),
                }
            }
            other => panic!("unexpected node {other:?}"),
        }
    }

    /// The production `StringPredicateRewriter::col_vs_col_rank_maps` over two
    /// host-only `DictionaryColumnAny`s with DIFFERENT dictionaries builds the
    /// cross-dictionary-correct rank tables (matches the string oracle).
    #[test]
    fn f12_real_rewriter_cross_dictionary_ranks() {
        use crate::cuda::dictionary_any::DictionaryColumnAny;
        let dict_a = ["delta", "apple", "mango"];
        let dict_b = ["cherry", "Zebra", "apple"];
        let da = DictionaryColumnAny::new_host_only(
            dict_a.iter().map(|s| s.to_string()).collect(),
            3,
        )
        .expect("da");
        let db = DictionaryColumnAny::new_host_only(
            dict_b.iter().map(|s| s.to_string()).collect(),
            3,
        )
        .expect("db");
        let mut rw = StringPredicateRewriter::new();
        rw.register("a", &da);
        rw.register("b", &db);

        let plan = rw.col_vs_col_rank_maps("a", "b").expect("rank plan");
        assert_eq!(plan.rank_col_a, "__rank_a");
        assert_eq!(plan.rank_col_b, "__rank_b");
        for op in [BinaryOp::Lt, BinaryOp::LtEq, BinaryOp::Gt, BinaryOp::GtEq] {
            for (ai, a_s) in dict_a.iter().enumerate() {
                for (bi, b_s) in dict_b.iter().enumerate() {
                    assert_eq!(
                        rank_cmp(op, plan.rank_a[ai + 1], plan.rank_b[bi + 1]),
                        str_cmp_oracle(op, a_s, b_s),
                        "op {op:?} {a_s} vs {b_s}"
                    );
                }
            }
        }
        // An unregistered column → None (host fallback).
        assert!(rw.col_vs_col_rank_maps("a", "unknown").is_none());
    }

    /// SIDE-CHANNEL (finding F12): the production `StringPredicateRewriter`
    /// records the unified rank tables when its live rewrite lowers a
    /// `col_a OP col_b` ordering, and `take_rank_plans` drains them keyed by the
    /// `__rank_<col>` names — exactly what `DictRegistry::rewrite_plan` relies on
    /// to feed the executor. Host-only via `new_host_only`.
    #[test]
    fn f12_rewriter_records_rank_plan_for_executor() {
        use crate::cuda::dictionary_any::DictionaryColumnAny;
        let dict_a = ["delta", "apple", "mango"];
        let dict_b = ["cherry", "Zebra", "apple"];
        let da = DictionaryColumnAny::new_host_only(
            dict_a.iter().map(|s| s.to_string()).collect(),
            3,
        )
        .expect("da");
        let db = DictionaryColumnAny::new_host_only(
            dict_b.iter().map(|s| s.to_string()).collect(),
            3,
        )
        .expect("db");
        let mut rw = StringPredicateRewriter::new();
        rw.register("a", &da);
        rw.register("b", &db);

        // Live rewrite of `a < b` must emit the rank comparison AND record the
        // plan in the side-channel.
        let out = rewrite_expr_with(&col("a").lt(col("b")), &rw, 0).unwrap();
        let op = assert_rank_comparison(&out, "__rank_a", "__rank_b");
        assert_eq!(op, BinaryOp::Lt);

        let plans = rw.take_rank_plans();
        assert_eq!(plans.len(), 1, "exactly one rank plan recorded");
        let plan = &plans[0];
        assert_eq!(plan.rank_col_a, "__rank_a");
        assert_eq!(plan.rank_col_b, "__rank_b");
        // Cross-dictionary correctness rides on the recorded tables.
        for (ai, a_s) in dict_a.iter().enumerate() {
            for (bi, b_s) in dict_b.iter().enumerate() {
                assert_eq!(
                    rank_cmp(BinaryOp::Lt, plan.rank_a[ai + 1], plan.rank_b[bi + 1]),
                    str_cmp_oracle(BinaryOp::Lt, a_s, b_s),
                );
            }
        }
        // Draining empties the buffer (scoped to one rewrite).
        assert!(rw.take_rank_plans().is_empty(), "buffer drained");
    }

    /// A col-vs-col ordering whose side is NOT a dict column records NO rank
    /// plan (it falls back to the host string comparison).
    #[test]
    fn f12_rewriter_records_nothing_on_host_fallback() {
        use crate::cuda::dictionary_any::DictionaryColumnAny;
        let da = DictionaryColumnAny::new_host_only(vec!["x".into(), "y".into()], 2)
            .expect("da");
        let mut rw = StringPredicateRewriter::new();
        rw.register("a", &da); // only `a` is a dict column
        let out = rewrite_expr_with(&col("a").lt(col("b")), &rw, 0).unwrap();
        // Preserved Utf8 comparison.
        match out {
            Expr::Binary { op: BinaryOp::Lt, left, right } => {
                assert_column(&left, "a");
                assert_column(&right, "b");
            }
            other => panic!("expected preserved `a < b`, got {other:?}"),
        }
        assert!(rw.take_rank_plans().is_empty(), "no rank plan on fallback");
    }

    /// NULL handling: the rank sentinel (-1) sits at slot 0 in both tables, and
    /// is strictly less than every real rank (>= 0). The executor must treat a
    /// row whose index is 0 on either side as SQL NULL; this test pins that the
    /// sentinel is distinguishable (negative) from every real rank so the exec
    /// hook can detect it.
    #[test]
    fn f12_null_sentinel_is_distinguishable() {
        let r = MockResolver::new()
            .with_dict("a", &["x", "y", "z"])
            .with_dict("b", &["x", "y", "z"]);
        let (_, plan) =
            plan_col_vs_col_rank(BinaryOp::Lt, &col("a"), &col("b"), &r).expect("plan");
        assert_eq!(plan.rank_a[0], crate::cuda::dictionary::NULL_RANK_SENTINEL);
        assert_eq!(plan.rank_b[0], crate::cuda::dictionary::NULL_RANK_SENTINEL);
        // Every real rank is >= 0, so the sentinel is strictly smaller.
        for &rk in &plan.rank_a[1..] {
            assert!(rk >= 0, "real ranks are non-negative");
            assert!(plan.rank_a[0] < rk, "sentinel must be below every real rank");
        }
    }
}
