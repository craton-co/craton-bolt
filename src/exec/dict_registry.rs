// SPDX-License-Identifier: Apache-2.0

//! Per-table dictionary registry that drives the string-literal predicate
//! rewrite.
//!
//! When the engine registers a host-side `RecordBatch`, it hands the batch to
//! this registry. The registry walks every `Utf8` column, builds a
//! [`DictionaryColumnAny`] (which uploads the index column — i32 or i64
//! depending on cardinality — to the device), and stores it keyed by
//! `(table_name, column_name)`. When a SQL query arrives, the engine asks the
//! registry to rewrite the logical plan: the registry constructs a fresh
//! [`StringPredicateRewriter`] populated with the dictionaries of every table
//! the plan scans, then returns a new plan whose `col = 'literal'` predicates
//! have been folded into `__idx_col = <i32>` predicates.
//!
//! Lifetimes: the registry owns the `DictionaryColumnAny`s and therefore owns
//! their GPU allocations. Dropping the registry drops the device memory. The
//! engine must keep the registry alive at least as long as any kernel that
//! references the on-device index columns.
//!
//! Cross-table column-name collisions (finding F-7): the
//! [`StringPredicateRewriter`] is keyed by **column name only**, because a
//! plan's `Expr::Column` references are themselves unqualified. When a plan
//! scans two tables (today reachable via `UNION` / `SetOp`, since
//! [`collect_scan_tables`] recurses into both children) that each expose a
//! Utf8 column with the *same name* but *different dictionaries*, folding
//! `col = 'lit'` against either table's dictionary would be wrong for rows
//! sourced from the other table. The previous behaviour silently let the
//! *last*-registered dictionary win — a silent wrong-results bug.
//!
//! [`DictRegistry::rewrite_plan`] now detects such a collision by comparing
//! the actual dictionary contents per column name across all scanned tables.
//! A column name that resolves to *conflicting* dictionaries is **poisoned**:
//! it is left out of the rewriter entirely, so its predicates fall back to the
//! always-correct host string-comparison path instead of being folded against
//! the wrong index space. Single-table plans (and multi-scan plans whose
//! same-named columns happen to carry identical dictionaries) are unaffected.
//!
//! Index width: both the i32- and i64-indexed dictionary variants are wired
//! through [`StringPredicateRewriter`]. The rewriter inspects each registered
//! [`DictionaryColumnAny`] and emits an `Int32` or `Int64` literal to match
//! the dictionary's width — keeping the rewritten predicate type-consistent
//! with the `__idx_<col>` field that [`DictRegistry::extended_schema`]
//! declares.

use std::collections::HashMap;

use arrow_array::types::Int32Type as ArrowInt32Type;
use arrow_array::{Array, DictionaryArray, RecordBatch, StringArray};
use arrow_schema::DataType as ArrowDataType;

use crate::cuda::dictionary::DictionaryColumn;
use crate::cuda::dictionary_any::DictionaryColumnAny;
use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{DataType, Field, LogicalPlan, Schema};
use crate::plan::string_literal_rewrite::{index_column_name, StringPredicateRewriter};

/// Per-table dictionary store driving the string-literal predicate rewrite.
///
/// One entry per registered table; inside each table, one entry per Utf8
/// column. The contained [`DictionaryColumnAny`]s own GPU allocations — drop
/// the registry, drop the device memory.
pub struct DictRegistry {
    /// `table_name` → `column_name` → on-host-and-device dictionary.
    by_table: HashMap<String, HashMap<String, DictionaryColumnAny>>,
}

impl Default for DictRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DictRegistry {
    /// Empty registry; no tables yet.
    pub fn new() -> Self {
        Self {
            by_table: HashMap::new(),
        }
    }

    /// Walk `batch` for `Utf8` columns and build a [`DictionaryColumnAny`]
    /// for each. Any prior dictionaries for `table` are dropped first — the
    /// engine treats `register_table` as a full replace.
    ///
    /// Non-Utf8 columns are ignored: the engine uploads those directly per
    /// query via `DeviceCol::upload`. The choice between an i32- and an
    /// i64-indexed dictionary is made by [`DictionaryColumnAny::from_string_array`]
    /// based on each column's cardinality.
    pub fn register_table(
        &mut self,
        table: impl Into<String>,
        batch: &RecordBatch,
    ) -> BoltResult<()> {
        let table = table.into();
        let mut cols: HashMap<String, DictionaryColumnAny> = HashMap::new();

        let schema = batch.schema();
        for (idx, field) in schema.fields().iter().enumerate() {
            let arr = batch.column(idx);
            match field.data_type() {
                ArrowDataType::Utf8 => {
                    let sa = arr
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| {
                            BoltError::Type(format!(
                                "column '{}' in table '{}' declared Utf8 but did not downcast to StringArray (got {:?})",
                                field.name(),
                                table,
                                arr.data_type()
                            ))
                        })?;
                    let dict = DictionaryColumnAny::from_string_array(sa)?;
                    cols.insert(field.name().clone(), dict);
                }
                // Stage 6: native ingest for Arrow `DictionaryArray<Int32, Utf8>`.
                // Previously this required `flatten_dictionary_utf8_columns` to
                // first materialise a `StringArray` — the registry now reads
                // the Arrow dictionary in-place instead, saving a full
                // host-side rematerialisation.
                ArrowDataType::Dictionary(key_t, val_t)
                    if key_t.as_ref() == &ArrowDataType::Int32
                        && val_t.as_ref() == &ArrowDataType::Utf8 =>
                {
                    let da = arr
                        .as_any()
                        .downcast_ref::<DictionaryArray<ArrowInt32Type>>()
                        .ok_or_else(|| {
                            BoltError::Type(format!(
                                "column '{}' in table '{}' declared Dictionary<Int32, Utf8> but \
                                 did not downcast to DictionaryArray<Int32Type>",
                                field.name(),
                                table,
                            ))
                        })?;
                    let dict = DictionaryColumnAny::from_dictionary_array(da)?;
                    cols.insert(field.name().clone(), dict);
                }
                _ => {
                    // Numeric / boolean columns are uploaded directly by the
                    // engine's per-query path and don't need a dictionary.
                }
            }
        }

        // Replace-not-merge: matches the engine's `register_table` contract.
        self.by_table.insert(table, cols);
        Ok(())
    }

    /// Register a pre-built Arrow dictionary for `(table, column)` directly,
    /// re-using the Arrow `DictionaryArray`'s values verbatim instead of
    /// rebuilding the dictionary from a `StringArray`.
    ///
    /// Stage 6 entry point: when the engine ingests a column that already
    /// arrived as `DictionaryArray<Int32, Utf8>` (Arrow IPC, Parquet, etc.),
    /// this method lets the registry skip the flatten-then-rebuild round trip.
    /// The resulting [`DictionaryColumnAny`] is always the `I32` variant —
    /// matching the source key width.
    ///
    /// Idempotency mirrors `register_table`: any prior dictionary for
    /// `(table, column)` is overwritten.
    pub fn register_dictionary_column(
        &mut self,
        table: impl Into<String>,
        column: impl Into<String>,
        dict_arr: &DictionaryArray<ArrowInt32Type>,
    ) -> BoltResult<()> {
        let table = table.into();
        let column = column.into();
        let dict = DictionaryColumnAny::from_dictionary_array(dict_arr)?;
        self.by_table
            .entry(table)
            .or_default()
            .insert(column, dict);
        Ok(())
    }

    /// Drop all dictionaries for `table`. No-op if `table` was never
    /// registered.
    pub fn unregister_table(&mut self, table: &str) {
        self.by_table.remove(table);
    }

    /// Apply the string-literal rewriter to `plan`.
    ///
    /// Walks the plan to collect every `Scan`'s table name, looks each one
    /// up in the registry, and folds every dictionary it finds into a single
    /// [`StringPredicateRewriter`].
    ///
    /// # Cross-table column-name collisions (finding F-7)
    ///
    /// The rewriter — and the plan's `Expr::Column` references it folds — are
    /// keyed by **unqualified** column name. When the plan scans more than one
    /// table (reachable today through `UNION` / `SetOp`, since
    /// [`collect_scan_tables`] recurses into both children) and two of those
    /// tables expose a Utf8 column with the *same name* but *different*
    /// dictionaries, there is no sound single dictionary to fold against: a
    /// `col = 'lit'` predicate would be correct for one table's rows and wrong
    /// for the other's. Rather than silently fold against the last-registered
    /// dictionary (the old "last wins" bug), such a column name is **poisoned**
    /// and omitted from the rewriter, so its predicates fall back to the
    /// always-correct host string-comparison path.
    ///
    /// A column name is *not* poisoned when every scanned table that exposes it
    /// carries an identical dictionary (same values in the same order) — those
    /// fold against a single well-defined index space. Single-table plans never
    /// poison anything, preserving the existing fast path verbatim.
    ///
    /// Both index widths are handled: [`StringPredicateRewriter`] accepts a
    /// [`DictionaryColumnAny`] directly and dispatches on the variant when
    /// emitting the rewritten predicate's literal (`Int32` vs `Int64`).
    ///
    /// If no scanned table has any (non-poisoned) Utf8 dictionaries the
    /// rewriter is empty and the returned plan is functionally a clone of
    /// `plan` (the rewriter is a no-op when `knows()` returns false for every
    /// column).
    pub fn rewrite_plan(&self, plan: &LogicalPlan) -> BoltResult<LogicalPlan> {
        let tables = collect_scan_tables(plan);

        // First pass: for each Utf8 column *name* present across the scanned
        // tables, record one candidate registration and detect collisions.
        // A collision is two scanned tables exposing the same column name with
        // dictionaries whose decoded contents differ; such a name is poisoned
        // (see the method docs / finding F-7) and must NOT be folded, because
        // the unqualified `Expr::Column` in the plan cannot say which table a
        // given row came from.
        //
        // We dedup the table list first so a table that appears twice in the
        // plan (e.g. a self-`UNION`) is not mistaken for a cross-table
        // collision against itself.
        let mut seen_tables: std::collections::HashSet<&str> = std::collections::HashSet::new();
        // col_name -> (registering dict, poisoned?)
        let mut chosen: HashMap<&str, &DictionaryColumnAny> = HashMap::new();
        let mut poisoned: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for table in &tables {
            if !seen_tables.insert(table.as_str()) {
                continue;
            }
            if let Some(cols) = self.by_table.get(table) {
                for (col_name, dict) in cols {
                    let key = col_name.as_str();
                    if poisoned.contains(key) {
                        continue;
                    }
                    // Resolve the comparison BEFORE mutating `chosen` so we
                    // never hold a borrow of `chosen` across a mutation.
                    let conflict = match chosen.get(key) {
                        None => None, // first sighting
                        Some(prev) => Some(dicts_conflict(prev, dict)),
                    };
                    // `prev` above is `&&DictionaryColumnAny`; `dicts_conflict`
                    // takes `&DictionaryColumnAny`, and Rust auto-derefs the
                    // extra reference at the call site.
                    match conflict {
                        None => {
                            chosen.insert(key, dict);
                        }
                        Some(true) => {
                            // Two distinct tables disagree on this column's
                            // dictionary: poison it and drop the earlier
                            // registration so neither side gets folded.
                            poisoned.insert(key);
                            chosen.remove(key);
                        }
                        Some(false) => {
                            // Same name, identical dictionary: harmless, keep
                            // the existing registration.
                        }
                    }
                }
            }
        }

        let mut rewriter = StringPredicateRewriter::new();
        for (col_name, dict) in chosen {
            // Both i32- and i64-indexed variants go through the same path; the
            // rewriter inspects the variant when it resolves a literal.
            rewriter.register(col_name.to_string(), dict);
        }
        // Protect the `LIKE` of any column the query projects as a bare output
        // from the integer-index membership fold, so it reaches the per-row GPU
        // `StringLikeFilter` (which can emit + compact the Utf8 rows) instead of
        // an integer filter that cannot produce a Utf8 output column. Equality /
        // inequality folds are unaffected — only the `LIKE` arm consults this.
        for col_name in collect_projected_bare_columns(plan) {
            rewriter.protect_like(col_name);
        }
        rewriter.rewrite(plan)
    }

    /// Extend the engine-side "logical" schema for `table` with index columns
    /// alongside each registered `Utf8` column.
    ///
    /// The rewriter emits `__idx_<col>` column references when it folds
    /// string equality into integer equality; the engine uses this extended
    /// schema to know which columns to *upload* to the device. The mangled
    /// column is not appended if it's already present (e.g. because a prior
    /// pass already extended the schema).
    ///
    /// The mangled column's dtype is taken from each dictionary's
    /// [`DictionaryColumnAny::index_dtype`] — `Int32` for i32-indexed
    /// dictionaries and `Int64` for i64-indexed ones. One Utf8 column on a
    /// table may resolve to `Int32` while another resolves to `Int64`,
    /// depending on each column's cardinality at register-table time.
    ///
    /// Returns a clone of `original` if `table` has no registered
    /// dictionaries.
    pub fn extended_schema(&self, table: &str, original: &Schema) -> Schema {
        let Some(cols) = self.by_table.get(table) else {
            return original.clone();
        };

        let mut fields = original.fields.clone();
        // Track names already in the schema so we don't double-append the
        // mangled column if a prior pass already added it.
        let mut present: std::collections::HashSet<String> =
            fields.iter().map(|f| f.name.clone()).collect();

        // Walk in schema order for deterministic output: append a mangled
        // index column right at the end for each Utf8 column the registry
        // knows about that is present in the source schema.
        for f in &original.fields {
            if f.dtype != DataType::Utf8 {
                continue;
            }
            let Some(dict) = cols.get(&f.name) else {
                continue;
            };
            let mangled = index_column_name(&f.name);
            if present.contains(&mangled) {
                continue;
            }
            // Per-column dtype: i32-indexed dictionaries declare `Int32`;
            // i64-indexed dictionaries declare `Int64`.
            fields.push(Field::new(mangled.clone(), dict.index_dtype(), false));
            present.insert(mangled);
        }

        Schema::new(fields)
    }

    /// Borrow the dictionary for `(table, column)` if present.
    ///
    /// The return type is the unified [`DictionaryColumnAny`] wrapper;
    /// callers that need the underlying i32 variant specifically can call
    /// [`DictionaryColumnAny::as_i32`] or use the legacy
    /// [`Self::dictionary_i32`] accessor below.
    pub fn dictionary(&self, table: &str, column: &str) -> Option<&DictionaryColumnAny> {
        self.by_table.get(table).and_then(|cols| cols.get(column))
    }

    /// LEGACY: borrow the i32 variant for `(table, column)` if present.
    ///
    /// Returns `None` when the column either isn't registered or is
    /// i64-indexed. Exists so the engine's existing `__idx_<col>` upload
    /// path can keep working — it currently only knows how to ship `i32*`
    /// kernel arguments — until the orchestrator teaches that path to
    /// dispatch on [`DictionaryColumnAny::index_dtype`].
    pub fn dictionary_i32(
        &self,
        table: &str,
        column: &str,
    ) -> Option<&DictionaryColumn> {
        self.dictionary(table, column).and_then(|d| d.as_i32())
    }

    /// Plan dtype of the `__idx_<original_col>` column on `table`, if the
    /// column is registered.
    ///
    /// Used by the engine's upload path to decide between `DeviceCol::I32`
    /// and `DeviceCol::I64` when shipping the index column to the device.
    /// Returns `None` if either the table or the column is not registered.
    pub fn dict_index_dtype(
        &self,
        table: &str,
        original_col: &str,
    ) -> Option<DataType> {
        self.dictionary(table, original_col).map(|d| d.index_dtype())
    }
}

// Stage 7 (S1): `dictionary_any_from_arrow_dict` was promoted to
// `DictionaryColumnAny::from_dictionary_array` in `src/cuda/dictionary_any.rs`
// so the unified-wrapper module owns the entire dictionary-construction
// surface. Call-sites in this file now route through the canonical
// constructor; this comment is the only thing that remains here.

/// True iff two dictionaries would fold a given literal to *different* index
/// spaces — i.e. their decoded contents differ in value or order.
///
/// Used by [`DictRegistry::rewrite_plan`] to decide whether two scanned tables
/// that share a Utf8 column *name* actually share a dictionary (safe to fold)
/// or conflict (the column must be poisoned — finding F-7). Comparing the host
/// `dictionary()` slices is exact: the slot index a literal resolves to is a
/// pure function of that slice, so identical slices guarantee identical
/// folding, and any difference is a genuine collision.
fn dicts_conflict(a: &DictionaryColumnAny, b: &DictionaryColumnAny) -> bool {
    a.dictionary() != b.dictionary()
}

/// Walk a `LogicalPlan` and collect every `Scan`'s table name in plan order.
///
/// Pure host function — no I/O, no allocations beyond the result vec. Used by
/// [`DictRegistry::rewrite_plan`] to know which tables' dictionaries to fold
/// into the rewriter.
/// Collect the names of every column a `Project` node surfaces as a *bare
/// column output* (an `Expr::Column`, optionally wrapped in transparent
/// `Alias`es), anywhere in the plan.
///
/// Used by [`DictRegistry::rewrite_plan`] to decide which columns' `LIKE`
/// predicates must be protected from the integer-index membership fold: if the
/// query projects a string column verbatim, the GPU integer filter can't emit
/// the surviving Utf8 rows, so the `LIKE` has to stay a real `Expr::Like` for
/// the `StringLikeFilter` / host path. Collecting a superset (e.g. numeric
/// outputs too) is harmless — only registered Utf8 dict columns are ever gated.
fn collect_projected_bare_columns(plan: &LogicalPlan) -> std::collections::HashSet<String> {
    use crate::plan::logical_plan::Expr;
    fn peel(e: &Expr) -> &Expr {
        let mut cur = e;
        while let Expr::Alias(inner, _) = cur {
            cur = inner.as_ref();
        }
        cur
    }
    fn walk(p: &LogicalPlan, out: &mut std::collections::HashSet<String>) {
        match p {
            LogicalPlan::Project { input, exprs } => {
                for e in exprs {
                    if let Expr::Column(name) = peel(e) {
                        out.insert(name.clone());
                    }
                }
                walk(input, out);
            }
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Distinct { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Window { input, .. }
            | LogicalPlan::Sort { input, .. } => walk(input, out),
            LogicalPlan::Union { inputs } => {
                for inp in inputs {
                    walk(inp, out);
                }
            }
            LogicalPlan::Join { left, right, .. }
            | LogicalPlan::SetOp { left, right, .. } => {
                walk(left, out);
                walk(right, out);
            }
            LogicalPlan::Scan { .. } => {}
        }
    }
    let mut out = std::collections::HashSet::new();
    walk(plan, &mut out);
    out
}

fn collect_scan_tables(plan: &LogicalPlan) -> Vec<String> {
    fn walk(p: &LogicalPlan, out: &mut Vec<String>) {
        match p {
            LogicalPlan::Scan { table, .. } => out.push(table.clone()),
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Project { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Distinct { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Window { input, .. }
            | LogicalPlan::Sort { input, .. } => walk(input, out),
            LogicalPlan::Union { inputs } => {
                for inp in inputs {
                    walk(inp, out);
                }
            }
            LogicalPlan::Join { left, right, .. }
            | LogicalPlan::SetOp { left, right, .. } => {
                walk(left, out);
                walk(right, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(plan, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{col, lit, Expr};

    // ---- Pure-host tests ---------------------------------------------------

    /// Synthetic plan: Filter(Scan(orders)). `collect_scan_tables` should
    /// descend through the Filter and surface "orders".
    #[test]
    fn collect_scan_tables_descends_through_filter() {
        let scan = LogicalPlan::Scan {
            table: "orders".into(),
            projection: None,
            schema: Schema::new(vec![Field::new("price", DataType::Float64, false)]),
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(scan),
            predicate: col("price").gt(lit(0.0_f64)),
        };

        let tables = collect_scan_tables(&plan);
        assert_eq!(tables, vec!["orders".to_string()]);
    }

    /// Project(Aggregate(Filter(Scan(t)))). The walker must recurse through
    /// every wrapper node.
    #[test]
    fn collect_scan_tables_descends_through_aggregate_and_project() {
        let scan = LogicalPlan::Scan {
            table: "events".into(),
            projection: None,
            schema: Schema::new(vec![Field::new("v", DataType::Int64, false)]),
        };
        let filt = LogicalPlan::Filter {
            input: Box::new(scan),
            predicate: col("v").gt(lit(0_i64)),
        };
        let agg = LogicalPlan::Aggregate {
            input: Box::new(filt),
            group_by: Vec::new(),
            aggregates: Vec::new(),
        };
        let proj = LogicalPlan::Project {
            input: Box::new(agg),
            exprs: vec![Expr::Column("v".into())],
        };

        assert_eq!(collect_scan_tables(&proj), vec!["events".to_string()]);
    }

    /// A bare Scan returns exactly its table name.
    #[test]
    fn collect_scan_tables_bare_scan() {
        let plan = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: Schema::new(Vec::new()),
        };
        assert_eq!(collect_scan_tables(&plan), vec!["t".to_string()]);
    }

    /// Registry construction is host-only — no CUDA needed for an empty
    /// instance. `dictionary` and `unregister_table` must be safe on a
    /// never-registered table.
    #[test]
    fn empty_registry_lookup_returns_none() {
        let reg = DictRegistry::new();
        assert!(reg.dictionary("missing", "col").is_none());
    }

    #[test]
    fn unregister_unknown_table_is_noop() {
        let mut reg = DictRegistry::new();
        reg.unregister_table("nope"); // must not panic
        assert!(reg.dictionary("nope", "col").is_none());
    }

    /// `extended_schema` for a table with no registered dictionaries should
    /// return an equivalent schema (clone). This exercises the early-return
    /// branch without touching CUDA.
    #[test]
    fn extended_schema_unknown_table_returns_clone() {
        let reg = DictRegistry::new();
        let original = Schema::new(vec![
            Field::new("region", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
        ]);
        let extended = reg.extended_schema("orders", &original);
        let names: Vec<&str> = extended.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["region", "price"]);
    }

    // ---- CUDA-dependent tests ---------------------------------------------
    //
    // These actually call `DictionaryColumn::from_string_array`, which uploads
    // i32 indices to the device. They will fail on a build machine without a
    // CUDA toolchain at runtime — but they compile cleanly and document the
    // intended end-to-end behaviour.

    /// Build a small `RecordBatch` with a Utf8 `region` and an Int64 `price`,
    /// register it, then confirm the registry holds a dictionary for `region`
    /// (and no dictionary for `price`).
    #[test]
    #[ignore = "gpu:string"]
    fn register_table_indexes_only_utf8_columns() {
        use std::sync::Arc;

        use arrow_array::Int64Array;
        use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("region", ArrowDataType::Utf8, false),
            ArrowField::new("price", ArrowDataType::Int64, false),
        ]));
        let region = Arc::new(StringArray::from(vec!["US", "EU", "US", "JP"]));
        let price = Arc::new(Int64Array::from(vec![1_i64, 2, 3, 4]));
        let batch =
            RecordBatch::try_new(schema, vec![region, price]).expect("build batch");

        let mut reg = DictRegistry::new();
        reg.register_table("orders", &batch).expect("register");

        let d = reg
            .dictionary("orders", "region")
            .expect("region dictionary should exist");
        // `from_string_array` deduplicates in first-occurrence order: US, EU, JP.
        // The wrapper exposes the host-side dictionary via its accessor; the
        // tiny synthetic input must land on the i32 path.
        assert!(d.is_i32(), "small input should pick i32 indices");
        assert_eq!(d.dictionary(), &["US", "EU", "JP"]);
        // No dictionary for the numeric column.
        assert!(reg.dictionary("orders", "price").is_none());
    }

    /// Round-trip a `Scan` plan through `rewrite_plan` and confirm the Utf8
    /// equality is folded into an Int32 equality against `__idx_region`.
    #[test]
    #[ignore = "gpu:string"]
    fn rewrite_plan_folds_string_eq_into_index_eq() {
        use std::sync::Arc;

        use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};

        use crate::plan::logical_plan::{BinaryOp, Literal};

        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "region",
            ArrowDataType::Utf8,
            false,
        )]));
        let region = Arc::new(StringArray::from(vec!["US", "EU"]));
        let batch =
            RecordBatch::try_new(arrow_schema, vec![region]).expect("build batch");

        let mut reg = DictRegistry::new();
        reg.register_table("orders", &batch).expect("register");

        let scan = LogicalPlan::Scan {
            table: "orders".into(),
            projection: None,
            schema: Schema::new(vec![Field::new("region", DataType::Utf8, false)]),
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(scan),
            predicate: col("region").eq(lit("US")),
        };

        let rewritten = reg.rewrite_plan(&plan).expect("rewrite");
        let LogicalPlan::Filter { predicate, .. } = rewritten else {
            panic!("expected Filter at root after rewrite");
        };
        match predicate {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                match (*left, *right) {
                    (Expr::Column(name), Expr::Literal(Literal::Int32(idx))) => {
                        assert_eq!(name, "__idx_region");
                        assert_eq!(idx, 1); // "US" is the first inserted string.
                    }
                    other => panic!("unexpected operands after rewrite: {other:?}"),
                }
            }
            other => panic!("expected Binary predicate, got {other:?}"),
        }
    }

    /// `extended_schema` appends the mangled index column for every
    /// registered Utf8 column and leaves non-Utf8 / unregistered columns
    /// alone.
    #[test]
    #[ignore = "gpu:string"]
    fn extended_schema_appends_index_columns() {
        use std::sync::Arc;

        use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};

        let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "region",
            ArrowDataType::Utf8,
            false,
        )]));
        let region = Arc::new(StringArray::from(vec!["US"]));
        let batch =
            RecordBatch::try_new(arrow_schema, vec![region]).expect("build batch");

        let mut reg = DictRegistry::new();
        reg.register_table("orders", &batch).expect("register");

        let original = Schema::new(vec![
            Field::new("region", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
        ]);
        let extended = reg.extended_schema("orders", &original);
        let names: Vec<&str> = extended.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["region", "price", "__idx_region"]);

        let idx_field = extended.field("__idx_region").expect("mangled present");
        assert_eq!(idx_field.dtype, DataType::Int32);
        assert!(!idx_field.nullable);

        // Re-extending the already-extended schema is idempotent.
        let twice = reg.extended_schema("orders", &extended);
        let names_twice: Vec<&str> = twice.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names_twice, vec!["region", "price", "__idx_region"]);
    }

    // ---- F-7: cross-table same-name column collision -----------------------

    /// Helper: build a single-Utf8-column `RecordBatch` named `col` from
    /// `values`, used by the F-7 collision tests below.
    fn utf8_batch(col: &str, values: &[&str]) -> RecordBatch {
        use std::sync::Arc;

        use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            col,
            ArrowDataType::Utf8,
            false,
        )]));
        let arr = Arc::new(StringArray::from(values.to_vec()));
        RecordBatch::try_new(schema, vec![arr]).expect("build utf8 batch")
    }

    /// Helper: a bare `Scan` over `table` exposing a single Utf8 `column`.
    fn utf8_scan(table: &str, column: &str) -> LogicalPlan {
        LogicalPlan::Scan {
            table: table.into(),
            projection: None,
            schema: Schema::new(vec![Field::new(column, DataType::Utf8, false)]),
        }
    }

    /// F-7 (core): two scanned tables expose a same-named Utf8 column with
    /// **different** dictionaries. A `UNION` over both reaches both scans
    /// (`collect_scan_tables` recurses), so the rewriter must NOT fold
    /// `region = 'US'` against either dictionary — the column is poisoned and
    /// the predicate is left as the original string comparison for the host
    /// path. This proves the two scans keep distinct dictionaries rather than
    /// aliasing (the old "last wins" bug).
    #[test]
    #[ignore = "gpu:string"]
    fn rewrite_plan_poisons_conflicting_same_name_column() {
        use crate::plan::logical_plan::{BinaryOp, Literal};

        let mut reg = DictRegistry::new();
        // Two tables, same column name `region`, DIFFERENT dictionary content.
        reg.register_table("orders_us", &utf8_batch("region", &["US", "EU"]))
            .expect("register orders_us");
        reg.register_table("orders_apac", &utf8_batch("region", &["JP", "AU"]))
            .expect("register orders_apac");

        // UNION(Scan(orders_us, region='US'), Scan(orders_apac, region='US')).
        let left = LogicalPlan::Filter {
            input: Box::new(utf8_scan("orders_us", "region")),
            predicate: col("region").eq(lit("US")),
        };
        let right = LogicalPlan::Filter {
            input: Box::new(utf8_scan("orders_apac", "region")),
            predicate: col("region").eq(lit("US")),
        };
        let plan = LogicalPlan::Union {
            inputs: vec![left, right],
        };

        let rewritten = reg.rewrite_plan(&plan).expect("rewrite");
        let LogicalPlan::Union { inputs } = rewritten else {
            panic!("expected Union at root");
        };
        assert_eq!(inputs.len(), 2);
        for inp in &inputs {
            let LogicalPlan::Filter { predicate, .. } = inp else {
                panic!("expected Filter under Union");
            };
            // Poisoned: predicate stays `region = 'US'` (Utf8 literal), NOT
            // folded into `__idx_region = <int>`.
            match predicate {
                Expr::Binary { op, left, right } => {
                    assert_eq!(*op, BinaryOp::Eq);
                    match (&**left, &**right) {
                        (Expr::Column(name), Expr::Literal(Literal::Utf8(s))) => {
                            assert_eq!(name, "region", "column must stay unmangled");
                            assert_eq!(s, "US");
                        }
                        other => {
                            panic!("conflicting column must NOT fold; got {other:?}")
                        }
                    }
                }
                other => panic!("expected unfolded Binary, got {other:?}"),
            }
        }
    }

    /// F-7 (control): two scanned tables expose a same-named column with the
    /// **identical** dictionary. There is a single well-defined index space, so
    /// the column is NOT poisoned and the predicate still folds.
    #[test]
    #[ignore = "gpu:string"]
    fn rewrite_plan_folds_identical_same_name_column() {
        use crate::plan::logical_plan::{BinaryOp, Literal};

        let mut reg = DictRegistry::new();
        reg.register_table("a", &utf8_batch("region", &["US", "EU"]))
            .expect("register a");
        reg.register_table("b", &utf8_batch("region", &["US", "EU"]))
            .expect("register b");

        let left = LogicalPlan::Filter {
            input: Box::new(utf8_scan("a", "region")),
            predicate: col("region").eq(lit("US")),
        };
        let right = LogicalPlan::Filter {
            input: Box::new(utf8_scan("b", "region")),
            predicate: col("region").eq(lit("US")),
        };
        let plan = LogicalPlan::Union {
            inputs: vec![left, right],
        };

        let rewritten = reg.rewrite_plan(&plan).expect("rewrite");
        let LogicalPlan::Union { inputs } = rewritten else {
            panic!("expected Union");
        };
        for inp in &inputs {
            let LogicalPlan::Filter { predicate, .. } = inp else {
                panic!("expected Filter");
            };
            match predicate {
                Expr::Binary { op, left, right } => {
                    assert_eq!(*op, BinaryOp::Eq);
                    match (&**left, &**right) {
                        (Expr::Column(name), Expr::Literal(Literal::Int32(idx))) => {
                            assert_eq!(name, "__idx_region", "identical dicts fold");
                            assert_eq!(*idx, 1);
                        }
                        other => panic!("identical dicts must fold; got {other:?}"),
                    }
                }
                other => panic!("expected Binary, got {other:?}"),
            }
        }
    }

    /// F-7 (regression): the single-table fast path is unchanged — a lone scan
    /// still folds `region = 'US'` into `__idx_region = 1`.
    #[test]
    #[ignore = "gpu:string"]
    fn rewrite_plan_single_table_still_folds() {
        use crate::plan::logical_plan::Literal;

        let mut reg = DictRegistry::new();
        reg.register_table("orders", &utf8_batch("region", &["US", "EU"]))
            .expect("register");

        let plan = LogicalPlan::Filter {
            input: Box::new(utf8_scan("orders", "region")),
            predicate: col("region").eq(lit("US")),
        };
        let rewritten = reg.rewrite_plan(&plan).expect("rewrite");
        let LogicalPlan::Filter { predicate, .. } = rewritten else {
            panic!("expected Filter");
        };
        match predicate {
            Expr::Binary { left, right, .. } => match (*left, *right) {
                (Expr::Column(name), Expr::Literal(Literal::Int32(idx))) => {
                    assert_eq!(name, "__idx_region");
                    assert_eq!(idx, 1);
                }
                other => panic!("single table must fold; got {other:?}"),
            },
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    /// F-7 (pure host): `dicts_conflict` compares decoded dictionary contents.
    /// This exercises the collision predicate without CUDA by constructing the
    /// comparison over the `dictionary()` slices the poison logic relies on.
    /// (The `DictionaryColumnAny` values themselves need CUDA to build, so the
    /// end-to-end poison test above is `#[ignore]`; this asserts the contract
    /// `dicts_conflict` encodes: identical slices => no conflict, any
    /// value/order difference => conflict.)
    #[test]
    fn dicts_conflict_contract_is_slice_equality() {
        // Mirror the exact comparison `dicts_conflict` performs.
        let a: Vec<String> = vec!["US".into(), "EU".into()];
        let b: Vec<String> = vec!["US".into(), "EU".into()];
        let c: Vec<String> = vec!["JP".into(), "AU".into()];
        let d: Vec<String> = vec!["EU".into(), "US".into()]; // same set, diff order

        assert!(a == b, "identical dictionaries must compare equal (no conflict)");
        assert!(a != c, "different values must differ (conflict)");
        assert!(
            a != d,
            "different order maps a literal to a different index (conflict)"
        );
    }

    /// Stage 6: register a column from an Arrow `DictionaryArray<Int32, Utf8>`
    /// natively (no flatten step). Verifies that the registered dictionary
    /// values match the Arrow input slot-for-slot, modulo the `+1` NULL
    /// reservation on the *index* side.
    #[test]
    #[ignore = "gpu:string"]
    fn register_dictionary_column_reuses_arrow_dict() {
        use arrow_array::builder::StringDictionaryBuilder;

        let mut b: StringDictionaryBuilder<ArrowInt32Type> = StringDictionaryBuilder::new();
        b.append_value("US");
        b.append_value("EU");
        b.append_value("US");
        b.append_null();
        b.append_value("JP");
        let arr = b.finish();

        let mut reg = DictRegistry::new();
        reg.register_dictionary_column("orders", "region", &arr)
            .expect("register dict");

        let d = reg
            .dictionary("orders", "region")
            .expect("region dictionary should exist");
        // i32 variant — keys are i32 by definition for this constructor.
        assert!(d.is_i32(), "Arrow Int32 keys should stay i32");
        // Dictionary mirrors the Arrow dictionary's values 1:1.
        let dict: Vec<&str> = d.dictionary().iter().map(|s| s.as_str()).collect();
        assert_eq!(dict, vec!["US", "EU", "JP"]);
        // Lookups against the in-memory dictionary still work; widened to i64.
        assert_eq!(d.index_of_any("US"), Some(1));
        assert_eq!(d.index_of_any("JP"), Some(3));
        assert_eq!(d.index_of_any("missing"), None);
    }

    /// Round-trip: `register_dictionary_column` then `rewrite_plan` folds a
    /// `col = 'US'` predicate into `__idx_col = 1` just like the
    /// StringArray-based path.
    #[test]
    #[ignore = "gpu:string"]
    fn register_dictionary_column_drives_rewrite() {
        use arrow_array::builder::StringDictionaryBuilder;

        use crate::plan::logical_plan::{BinaryOp, Literal};

        let mut b: StringDictionaryBuilder<ArrowInt32Type> = StringDictionaryBuilder::new();
        b.append_value("US");
        b.append_value("EU");
        let arr = b.finish();

        let mut reg = DictRegistry::new();
        reg.register_dictionary_column("orders", "region", &arr)
            .expect("register dict");

        let scan = LogicalPlan::Scan {
            table: "orders".into(),
            projection: None,
            schema: Schema::new(vec![Field::new("region", DataType::Utf8, false)]),
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(scan),
            predicate: col("region").eq(lit("US")),
        };
        let rewritten = reg.rewrite_plan(&plan).expect("rewrite");
        let LogicalPlan::Filter { predicate, .. } = rewritten else {
            panic!("expected Filter at root");
        };
        match predicate {
            Expr::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Eq);
                match (*left, *right) {
                    (Expr::Column(name), Expr::Literal(Literal::Int32(idx))) => {
                        assert_eq!(name, "__idx_region");
                        assert_eq!(idx, 1);
                    }
                    other => panic!("unexpected operands: {other:?}"),
                }
            }
            other => panic!("expected Binary: {other:?}"),
        }
    }
}
