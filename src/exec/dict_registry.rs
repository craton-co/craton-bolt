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
//! Cross-table column-name collisions: the rewriter is keyed by column name
//! only, so if two scanned tables both expose a Utf8 column with the same
//! name but different dictionaries, the registry folds both into the same
//! rewriter and the *last* one wins. In practice the engine has no JOIN
//! support yet, so a single plan only ever references one table and this is a
//! moot point. The behaviour is documented on [`DictRegistry::rewrite_plan`]
//! for the day JOINs land.
//!
//! Index width: both the i32- and i64-indexed dictionary variants are wired
//! through [`StringPredicateRewriter`]. The rewriter inspects each registered
//! [`DictionaryColumnAny`] and emits an `Int32` or `Int64` literal to match
//! the dictionary's width — keeping the rewritten predicate type-consistent
//! with the `__idx_<col>` field that [`DictRegistry::extended_schema`]
//! declares.

use std::collections::HashMap;

use arrow_array::{Array, RecordBatch, StringArray};
use arrow_schema::DataType as ArrowDataType;

use crate::cuda::dictionary::DictionaryColumn;
use crate::cuda::dictionary_any::DictionaryColumnAny;
use crate::error::{PatinaError, PatinaResult};
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
    ) -> PatinaResult<()> {
        let table = table.into();
        let mut cols: HashMap<String, DictionaryColumnAny> = HashMap::new();

        let schema = batch.schema();
        for (idx, field) in schema.fields().iter().enumerate() {
            if field.data_type() != &ArrowDataType::Utf8 {
                continue;
            }
            let arr = batch.column(idx);
            let sa = arr
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    PatinaError::Type(format!(
                        "column '{}' in table '{}' declared Utf8 but did not downcast to StringArray (got {:?})",
                        field.name(),
                        table,
                        arr.data_type()
                    ))
                })?;
            let dict = DictionaryColumnAny::from_string_array(sa)?;
            cols.insert(field.name().clone(), dict);
        }

        // Replace-not-merge: matches the engine's `register_table` contract.
        self.by_table.insert(table, cols);
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
    /// [`StringPredicateRewriter`]. The rewriter is keyed by column name
    /// only — if two referenced tables both have a Utf8 column with the same
    /// name, the *last* one registered wins. This is acceptable until JOINs
    /// land, at which point the rewriter API will need a per-relation
    /// namespace.
    ///
    /// Both index widths are handled: [`StringPredicateRewriter`] accepts a
    /// [`DictionaryColumnAny`] directly and dispatches on the variant when
    /// emitting the rewritten predicate's literal (`Int32` vs `Int64`). No
    /// dictionary is skipped.
    ///
    /// If no scanned table has any Utf8 dictionaries the rewriter is empty
    /// and the returned plan is functionally a clone of `plan` (the
    /// rewriter is a no-op when `knows()` returns false for every column).
    pub fn rewrite_plan(&self, plan: &LogicalPlan) -> PatinaResult<LogicalPlan> {
        let tables = collect_scan_tables(plan);
        let mut rewriter = StringPredicateRewriter::new();
        for table in &tables {
            if let Some(cols) = self.by_table.get(table) {
                for (col_name, dict) in cols {
                    // Last-write-wins on cross-table column-name collisions;
                    // documented above. Both i32- and i64-indexed variants
                    // go through the same code path: the rewriter inspects
                    // the variant when it resolves a literal.
                    rewriter.register(col_name.clone(), dict);
                }
            }
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

/// Walk a `LogicalPlan` and collect every `Scan`'s table name in plan order.
///
/// Pure host function — no I/O, no allocations beyond the result vec. Used by
/// [`DictRegistry::rewrite_plan`] to know which tables' dictionaries to fold
/// into the rewriter.
fn collect_scan_tables(plan: &LogicalPlan) -> Vec<String> {
    fn walk(p: &LogicalPlan, out: &mut Vec<String>) {
        match p {
            LogicalPlan::Scan { table, .. } => out.push(table.clone()),
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Project { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Distinct { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Sort { input, .. } => walk(input, out),
            LogicalPlan::Union { inputs } => {
                for inp in inputs {
                    walk(inp, out);
                }
            }
            LogicalPlan::Join { left, right, .. } => {
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
    #[ignore = "requires CUDA device"]
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
    #[ignore = "requires CUDA device"]
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
    #[ignore = "requires CUDA device"]
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
}
