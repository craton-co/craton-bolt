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
//!   * Reversed shape (literal on the left) is normalised before rewrite.
//!   * If `'lit'` is not in the dictionary, the predicate is constant-folded:
//!     `=` → `Bool(false)`, `<>` → `Bool(true)`.
//!
//! Unsupported (returns `BoltError::Plan`):
//!   * `< <= > >=` on Utf8 columns with Utf8 literals — dictionary indices
//!     reflect insertion order, not lexicographic order, so these can't be
//!     reduced to integer comparison without a collation pass.
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
        }
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
        rewrite_plan_with(plan, self, 0)
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

/// True for ordering comparisons we cannot reduce via dictionary indices.
fn is_ordering(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq
    )
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
            if let Some((col_name, lit_str, _swapped)) =
                extract_col_and_string_lit(&new_left, &new_right)
            {
                if r.knows(&col_name) {
                    if is_eq_or_neq(*op) {
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
                                // Literal not in the dictionary: `=` is
                                // trivially false, `<>` is trivially true.
                                //
                                // Edge case: the empty string `""` is treated
                                // like any other literal here. If the
                                // registered dictionary never observed `""`
                                // at build time, then `WHERE col = ''` folds
                                // to `Bool(false)` and `WHERE col <> ''`
                                // folds to `Bool(true)` — i.e. we assume no
                                // row holds an empty string. This matches the
                                // dictionary's `index_of` semantics (only
                                // observed literals are present; NULL lives
                                // at slot 0, not `""`), and is correct as
                                // long as the upload path never silently
                                // coalesces `""` into NULL. Callers that
                                // need `''` to match real empty-string rows
                                // must ensure `""` is in the source array
                                // when the dictionary is built.
                                let folded = match op {
                                    BinaryOp::Eq => false,
                                    BinaryOp::NotEq => true,
                                    _ => unreachable!("is_eq_or_neq gated this branch"),
                                };
                                return Ok(Expr::Literal(Literal::Bool(folded)));
                            }
                        }
                    } else if is_ordering(*op) {
                        return Err(BoltError::Plan(format!(
                            "ordering comparison {op:?} on Utf8 column '{col_name}' \
                             requires dictionary collation (not yet implemented)"
                        )));
                    }
                    // Other ops (arithmetic, logical) against a Utf8 column
                    // are type errors elsewhere; fall through and let the
                    // standard type checker surface them.
                }
            }

            Ok(Expr::Binary {
                op: *op,
                left: Box::new(new_left),
                right: Box::new(new_right),
            })
        }
        Expr::Unary { op, operand } => {
            // `IS [NOT] NULL` does not interact with the string-literal
            // rewriter: the rewriter folds `col = 'lit'` shapes into
            // integer-index comparisons against a registered dictionary,
            // and a unary validity test has no literal to resolve. We
            // still walk the operand so any rewritable sub-expression
            // (e.g. `(col = 'a') IS NULL`, however unusual) is normalised.
            let new_operand = rewrite_expr_with(operand, r)?;
            Ok(Expr::Unary {
                op: *op,
                operand: Box::new(new_operand),
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
        LogicalPlan::Union { inputs } => {
            let new_inputs = inputs
                .iter()
                .map(|inp| rewrite_plan_with(inp, r, depth + 1))
                .collect::<BoltResult<Vec<_>>>()?;
            Ok(LogicalPlan::Union { inputs: new_inputs })
        }
        LogicalPlan::Join { left, right, join_type, on } => {
            let new_left = rewrite_plan_with(left, r, depth + 1)?;
            let new_right = rewrite_plan_with(right, r, depth + 1)?;
            Ok(LogicalPlan::Join {
                left: Box::new(new_left),
                right: Box::new(new_right),
                join_type: *join_type,
                on: on.clone(),
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
    }

    impl MockResolver {
        fn new() -> Self {
            Self {
                entries: HashMap::new(),
                columns: HashMap::new(),
            }
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
        // Column is known but literal not in the dictionary.
        let r = MockResolver::new().known_i32("region");
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
        // i32 side.
        let r32 = MockResolver::new().known_i32("region");
        let eq = rewrite_expr_with(&col("region").eq(lit("ZZ")), &r32, 0).unwrap();
        assert!(matches!(eq, Expr::Literal(Literal::Bool(false))));
        let neq = rewrite_expr_with(&col("region").neq(lit("ZZ")), &r32, 0).unwrap();
        assert!(matches!(neq, Expr::Literal(Literal::Bool(true))));

        // i64 side: same behaviour.
        let r64 = MockResolver::new().known_i64("user_id");
        let eq64 = rewrite_expr_with(&col("user_id").eq(lit("ghost")), &r64, 0).unwrap();
        assert!(matches!(eq64, Expr::Literal(Literal::Bool(false))));
        let neq64 = rewrite_expr_with(&col("user_id").neq(lit("ghost")), &r64, 0).unwrap();
        assert!(matches!(neq64, Expr::Literal(Literal::Bool(true))));
    }

    #[test]
    fn rewrite_neq_with_unknown_literal_folds_to_true() {
        let r = MockResolver::new().known_i32("region");
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

    #[test]
    fn reject_lt_on_string_column() {
        let r = MockResolver::new().with_i32("region", "US", 5);
        let expr = col("region").lt(lit("US"));
        let err = rewrite_expr_with(&expr, &r, 0).unwrap_err();
        match err {
            BoltError::Plan(msg) => {
                assert!(
                    msg.contains("ordering comparison"),
                    "expected ordering message, got: {msg}"
                );
                assert!(msg.contains("region"), "expected column name in: {msg}");
            }
            other => panic!("expected BoltError::Plan, got {other:?}"),
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
        // knows {"a": 1, "b": 2}. Demonstrates the broken behaviour:
        // `s = 'c'` folds to `Bool(false)`.
        let pre_fix = MockResolver::new()
            .with_i32("s", "a", 1)
            .with_i32("s", "b", 2);
        let folded =
            rewrite_expr_with(&col("s").eq(lit("c")), &pre_fix, 0).unwrap();
        assert!(
            matches!(folded, Expr::Literal(Literal::Bool(false))),
            "pre-fix: missing union dict folds to Bool(false) — silent-wrong-result"
        );

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
}
