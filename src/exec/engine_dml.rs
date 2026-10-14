// SPDX-License-Identifier: Apache-2.0
//! Row and schema changes: `DELETE`, `UPDATE` and `ALTER TABLE`.
//!
//! Tables are columnar and their files immutable, so a change is a rewrite:
//!
//! * **`DELETE` / `UPDATE`** evaluate the `WHERE` predicate as an ordinary
//!   query (`CASE WHEN p THEN 1 ELSE 0 END`, so a NULL predicate matches no
//!   row) and rebuild the affected rows. For a catalog table only the data
//!   files whose zone maps may match the predicate are read and rewritten;
//!   the others are kept as they are, and the change commits one snapshot.
//!   Subqueries in the predicate or the new values see the table as it was
//!   before the statement.
//! * **`ALTER TABLE`** rewrites the whole table under its new schema
//!   (`ADD` / `DROP` / `RENAME COLUMN`, `ALTER COLUMN ... TYPE` / `SET NOT
//!   NULL` / `DROP NOT NULL`), or renames it (`RENAME TO`).
//!
//! In-memory tables are rebuilt in place; lazy streams, external Parquet
//! tables and registered storage tables are read-only.

use std::sync::Arc;

use arrow::compute::{cast_with_options, CastOptions};
use arrow_array::{new_null_array, Array, ArrayRef, RecordBatch};
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use sqlparser::ast::{
    AlterColumnOperation, AlterTableOperation, Assignment, AssignmentTarget, ColumnOption, Expr,
    FromTable, ObjectName, TableFactor, TableWithJoins,
};

use crate::error::{BoltError, BoltResult};
use crate::exec::engine::Engine;
use crate::exec::engine_ddl::{align_batch, ddl_type, object_name};

/// Engine table that holds the rows a statement rewrites while its
/// predicate runs (never visible to other statements).
const DML_SOURCE: &str = "__bolt_dml_source";

/// Name the predicate's column references may qualify: the table's alias,
/// or its name.
fn target_of(t: &TableWithJoins) -> BoltResult<(String, String)> {
    if !t.joins.is_empty() {
        return Err(BoltError::Sql(
            "DELETE / UPDATE on a join is not supported".into(),
        ));
    }
    match &t.relation {
        TableFactor::Table { name, alias, .. } => {
            let table = object_name(name);
            let qualifier = alias
                .as_ref()
                .map(|a| a.name.value.clone())
                .unwrap_or_else(|| name.0.last().map(|i| i.value.clone()).unwrap_or_default());
            Ok((table, qualifier))
        }
        other => Err(BoltError::Sql(format!(
            "DELETE / UPDATE target must be a table, not {other}"
        ))),
    }
}

fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn concat(schema: &SchemaRef, batches: &[RecordBatch]) -> BoltResult<RecordBatch> {
    let nonempty: Vec<&RecordBatch> = batches.iter().filter(|b| b.num_rows() > 0).collect();
    match nonempty.as_slice() {
        [] => Ok(RecordBatch::new_empty(Arc::clone(schema))),
        [one] => Ok((*one).clone()),
        many => arrow::compute::concat_batches(schema, many.iter().copied())
            .map_err(|e| BoltError::Other(format!("concatenating rows: {e}"))),
    }
}

fn cast_column(a: &ArrayRef, to: &arrow_schema::DataType, column: &str) -> BoltResult<ArrayRef> {
    if a.data_type() == to {
        return Ok(Arc::clone(a));
    }
    let opts = CastOptions {
        safe: false,
        ..Default::default()
    };
    cast_with_options(a, to, &opts)
        .map_err(|e| BoltError::Type(format!("cannot convert column '{column}': {e}")))
}

/// One step of an `ALTER TABLE` over the table's full rows.
enum Alteration {
    Add {
        field: ArrowField,
        default: Option<String>,
    },
    Drop(String),
    Rename {
        from: String,
        to: String,
    },
    SetType {
        column: String,
        dtype: arrow_schema::DataType,
    },
    SetNullable {
        column: String,
        nullable: bool,
    },
}

impl Engine {
    /// `DELETE FROM t [WHERE p]`. Returns the number of rows removed.
    pub(crate) fn execute_delete(&mut self, d: sqlparser::ast::Delete) -> BoltResult<u64> {
        if d.using.is_some() || d.returning.is_some() || d.limit.is_some() {
            return Err(BoltError::Sql(
                "DELETE supports only FROM t [WHERE ...] (no USING / RETURNING / LIMIT)".into(),
            ));
        }
        if !d.tables.is_empty() || !d.order_by.is_empty() {
            return Err(BoltError::Sql("multi-table DELETE is not supported".into()));
        }
        let from = match &d.from {
            FromTable::WithFromKeyword(t) | FromTable::WithoutKeyword(t) => t,
        };
        let [target] = from.as_slice() else {
            return Err(BoltError::Sql("DELETE reads exactly one table".into()));
        };
        let (table, qualifier) = target_of(target)?;
        let predicate = d.selection.as_ref().map(Expr::to_string);
        self.rewrite_rows(&table, &qualifier, predicate.as_deref(), None)
    }

    /// `UPDATE t SET c = e, ... [WHERE p]`. Returns the number of rows
    /// changed.
    pub(crate) fn execute_update(
        &mut self,
        table: TableWithJoins,
        assignments: Vec<Assignment>,
        from: Option<TableWithJoins>,
        selection: Option<Expr>,
        returning: Option<Vec<sqlparser::ast::SelectItem>>,
    ) -> BoltResult<u64> {
        if from.is_some() || returning.is_some() {
            return Err(BoltError::Sql(
                "UPDATE supports only SET ... [WHERE ...] (no FROM / RETURNING)".into(),
            ));
        }
        let (name, qualifier) = target_of(&table)?;
        let schema = self.writable_schema(&name)?;
        let mut set: Vec<(String, String)> = Vec::new();
        for a in assignments {
            let AssignmentTarget::ColumnName(col) = a.target else {
                return Err(BoltError::Sql(
                    "UPDATE SET (a, b) = ... is not supported".into(),
                ));
            };
            let col = col.0.last().map(|i| i.value.clone()).unwrap_or_default();
            if schema.index_of(&col).is_err() {
                return Err(BoltError::Plan(format!(
                    "column '{col}' does not exist in table '{name}'"
                )));
            }
            if set.iter().any(|(c, _)| *c == col) {
                return Err(BoltError::Plan(format!(
                    "column '{col}' is assigned more than once"
                )));
            }
            set.push((col, a.value.to_string()));
        }
        let predicate = selection.as_ref().map(Expr::to_string);
        self.rewrite_rows(&name, &qualifier, predicate.as_deref(), Some(&set))
    }

    /// Schema of table `name` if statements may change its rows.
    fn writable_schema(&self, name: &str) -> BoltResult<SchemaRef> {
        if self.view_exists(name) {
            return Err(BoltError::Plan(format!(
                "'{name}' is a materialized view; it changes only by REFRESH"
            )));
        }
        #[cfg(feature = "storage")]
        if let Some(t) = self.catalog.as_ref().and_then(|c| c.table(name)) {
            if !matches!(t.kind, crate::storage::catalog::TableKind::Managed) {
                return Err(BoltError::Plan(format!(
                    "table '{name}' is external and read-only"
                )));
            }
            return self.table_arrow_schema(name);
        }
        if self.tables.contains_key(name) {
            return self.table_arrow_schema(name);
        }
        if self.has_table(name) {
            return Err(BoltError::Plan(format!(
                "table '{name}' is read-only (lazy streams and registered storage tables \
                 cannot be changed)"
            )));
        }
        Err(BoltError::Plan(format!("table '{name}' does not exist")))
    }

    /// Delete (`set == None`) or update the rows of `name` matching
    /// `predicate` (all rows when `None`). Returns the rows affected.
    fn rewrite_rows(
        &mut self,
        name: &str,
        qualifier: &str,
        predicate: Option<&str>,
        set: Option<&[(String, String)]>,
    ) -> BoltResult<u64> {
        let schema = self.writable_schema(name)?;
        #[cfg(feature = "storage")]
        if self
            .catalog
            .as_ref()
            .is_some_and(|c| c.table(name).is_some())
        {
            return self.rewrite_catalog_rows(name, qualifier, predicate, set, &schema);
        }
        let (rows, affected) = self.rewritten(name, qualifier, predicate, set, &schema)?;
        if affected > 0 {
            self.replace_table(name, rows)?;
            self.after_table_change(name)?;
        }
        Ok(affected)
    }

    /// Rewrite only the catalog files the predicate may touch.
    #[cfg(feature = "storage")]
    fn rewrite_catalog_rows(
        &mut self,
        name: &str,
        qualifier: &str,
        predicate: Option<&str>,
        set: Option<&[(String, String)]>,
        schema: &SchemaRef,
    ) -> BoltResult<u64> {
        use crate::storage::StorageTable as _;
        // Zone maps of each file against the predicate's prunable conjuncts.
        let request = match predicate {
            Some(p) => crate::plan::sql_frontend::parse(
                &format!("SELECT * FROM {name} AS {} WHERE {p}", quote(qualifier)),
                &self.provider,
            )
            .ok()
            .and_then(|plan| {
                self.storage_scan_requests(&plan, &[name.to_string()], &[])
                    .remove(name)
            })
            .unwrap_or_else(crate::storage::ScanRequest::full),
            None => crate::storage::ScanRequest::full(),
        };
        let cat = self.catalog.as_ref().expect("checked");
        let meta = cat.table(name).expect("checked");
        let touched: Vec<String> = meta
            .files
            .iter()
            .filter(|f| request.chunk_may_match(&f.stats.columns, f.rows))
            .map(|f| f.path.clone())
            .collect();
        if touched.is_empty() {
            return Ok(0);
        }
        let paths: Vec<std::path::PathBuf> = touched.iter().map(|p| cat.root().join(p)).collect();
        let files = crate::storage::ParquetTable::with_schema(Arc::clone(schema), &paths)?;
        let all = files.plan_scan(&crate::storage::ScanRequest::full());
        let batches: Vec<RecordBatch> = files.read(&all)?.collect::<BoltResult<_>>()?;
        let source = concat(schema, &batches)?;
        let (rows, affected) = self.with_dml_source(source, |e| {
            e.rewritten(DML_SOURCE, qualifier, predicate, set, schema)
        })?;
        if affected == 0 {
            return Ok(0);
        }
        self.catalog.as_mut().expect("checked").rewrite_files(
            name,
            &touched,
            std::slice::from_ref(&rows),
        )?;
        self.register_catalog_table(name)?;
        self.after_table_change(name)?;
        Ok(affected)
    }

    /// Run `f` with `rows` registered as the [`DML_SOURCE`] table.
    fn with_dml_source<R>(
        &mut self,
        rows: RecordBatch,
        f: impl FnOnce(&mut Engine) -> BoltResult<R>,
    ) -> BoltResult<R> {
        if self.has_table(DML_SOURCE) {
            self.deregister_table(DML_SOURCE)?;
        }
        self.register_table(DML_SOURCE, rows)?;
        let r = f(self);
        self.deregister_table(DML_SOURCE)?;
        r
    }

    /// The rows of `source` after the change, and the number affected.
    fn rewritten(
        &mut self,
        source: &str,
        qualifier: &str,
        predicate: Option<&str>,
        set: Option<&[(String, String)]>,
        schema: &SchemaRef,
    ) -> BoltResult<(RecordBatch, u64)> {
        let from = format!("{source} AS {}", quote(qualifier));
        let matches = |hit: bool| {
            predicate.map(|p| {
                format!(
                    " WHERE (CASE WHEN ({p}) THEN 1 ELSE 0 END) = {}",
                    i32::from(hit)
                )
            })
        };
        let kept = match predicate {
            // Without a predicate every row is affected.
            None => RecordBatch::new_empty(Arc::clone(schema)),
            Some(_) => {
                let q = format!("SELECT * FROM {from}{}", matches(false).unwrap_or_default());
                align_batch(&self.sql(&q)?.into_record_batch(), schema, &[])?
            }
        };
        let total = self.table_row_count(source)?;
        let Some(set) = set else {
            let removed = total.saturating_sub(kept.num_rows() as u64);
            return Ok((kept, removed));
        };
        if kept.num_rows() as u64 == total {
            return Ok((kept, 0));
        }
        let list = schema
            .fields()
            .iter()
            .map(|f| match set.iter().find(|(c, _)| c == f.name()) {
                Some((_, e)) => format!("({e}) AS {}", quote(f.name())),
                None => format!("{}.{}", quote(qualifier), quote(f.name())),
            })
            .collect::<Vec<_>>()
            .join(", ");
        let q = format!(
            "SELECT {list} FROM {from}{}",
            matches(true).unwrap_or_default()
        );
        let changed = align_batch(&self.sql(&q)?.into_record_batch(), schema, &[])?;
        let n = changed.num_rows() as u64;
        Ok((concat(schema, &[kept, changed])?, n))
    }

    fn table_row_count(&self, name: &str) -> BoltResult<u64> {
        if let Some(b) = self.tables.get(name) {
            return Ok(b.iter().map(|b| b.num_rows() as u64).sum());
        }
        Ok(self.table_statistics(name)?.row_count)
    }

    /// Every row of a writable table, as one batch.
    fn full_rows(&self, name: &str, schema: &SchemaRef) -> BoltResult<RecordBatch> {
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_ref() {
            if cat.table(name).is_some() {
                use crate::storage::StorageTable as _;
                let t = cat.open_table(name)?;
                let all = t.plan_scan(&crate::storage::ScanRequest::full());
                let batches: Vec<RecordBatch> = t.read(&all)?.collect::<BoltResult<_>>()?;
                return concat(schema, &batches);
            }
        }
        concat(schema, self.tables.get(name).map_or(&[][..], Vec::as_slice))
    }

    /// `ALTER TABLE`.
    pub(crate) fn execute_alter_table(
        &mut self,
        name: &ObjectName,
        if_exists: bool,
        operations: Vec<AlterTableOperation>,
    ) -> BoltResult<()> {
        let table = object_name(name);
        if !self.has_table(&table) {
            if if_exists {
                return Ok(());
            }
            return Err(BoltError::Plan(format!("table '{table}' does not exist")));
        }
        let schema = self.writable_schema(&table)?;
        if let [AlterTableOperation::RenameTable { table_name }] = operations.as_slice() {
            return self.rename_table(&table, &object_name(table_name));
        }
        let mut steps = Vec::new();
        for op in operations {
            steps.push(match op {
                AlterTableOperation::AddColumn {
                    if_not_exists,
                    column_def,
                    ..
                } => {
                    let col = column_def.name.value.clone();
                    if schema.index_of(&col).is_ok() {
                        if if_not_exists {
                            continue;
                        }
                        return Err(BoltError::Plan(format!(
                            "column '{col}' already exists in table '{table}'"
                        )));
                    }
                    let dtype = crate::exec::schema_convert::plan_dtype_to_arrow(ddl_type(
                        &column_def.data_type.to_string(),
                    )?)?;
                    let mut not_null = false;
                    let mut default = None;
                    for o in &column_def.options {
                        match &o.option {
                            ColumnOption::NotNull => not_null = true,
                            ColumnOption::Null => {}
                            ColumnOption::Default(e) => default = Some(e.to_string()),
                            other => {
                                return Err(BoltError::Sql(format!(
                                    "ADD COLUMN option not supported: {other}"
                                )))
                            }
                        }
                    }
                    Alteration::Add {
                        field: ArrowField::new(col, dtype, !not_null),
                        default,
                    }
                }
                AlterTableOperation::DropColumn {
                    column_name,
                    if_exists,
                    ..
                } => {
                    if schema.index_of(&column_name.value).is_err() && if_exists {
                        continue;
                    }
                    Alteration::Drop(column_name.value)
                }
                AlterTableOperation::RenameColumn {
                    old_column_name,
                    new_column_name,
                } => Alteration::Rename {
                    from: old_column_name.value,
                    to: new_column_name.value,
                },
                AlterTableOperation::AlterColumn { column_name, op } => match op {
                    AlterColumnOperation::SetNotNull => Alteration::SetNullable {
                        column: column_name.value,
                        nullable: false,
                    },
                    AlterColumnOperation::DropNotNull => Alteration::SetNullable {
                        column: column_name.value,
                        nullable: true,
                    },
                    AlterColumnOperation::SetDataType {
                        data_type,
                        using: None,
                    } => Alteration::SetType {
                        column: column_name.value,
                        dtype: crate::exec::schema_convert::plan_dtype_to_arrow(ddl_type(
                            &data_type.to_string(),
                        )?)?,
                    },
                    other => {
                        return Err(BoltError::Sql(format!(
                            "ALTER COLUMN {other} is not supported"
                        )))
                    }
                },
                AlterTableOperation::RenameTable { .. } => {
                    return Err(BoltError::Sql(
                        "RENAME TO must be the only ALTER TABLE operation".into(),
                    ))
                }
                other => {
                    return Err(BoltError::Sql(format!(
                        "ALTER TABLE {other} is not supported"
                    )))
                }
            });
        }
        if steps.is_empty() {
            return Ok(());
        }
        let changes_columns = steps.iter().any(|s| !matches!(s, Alteration::Add { .. }));
        if changes_columns {
            self.ensure_no_dependent_views(&table)?;
        }
        let mut rows = self.full_rows(&table, &schema)?;
        for step in steps {
            rows = self.apply_alteration(&table, rows, step)?;
        }
        self.store_altered(&table, rows)?;
        self.after_table_change(&table)
    }

    fn ensure_no_dependent_views(&self, table: &str) -> BoltResult<()> {
        let views = self.views.borrow();
        let mut readers: Vec<&String> = views
            .iter()
            .filter(|(_, v)| v.dependencies.iter().any(|d| d == table))
            .map(|(n, _)| n)
            .collect();
        readers.sort();
        match readers.first() {
            Some(v) => Err(BoltError::Plan(format!(
                "materialized view '{v}' reads '{table}'; drop it before changing the table's \
                 columns"
            ))),
            None => Ok(()),
        }
    }

    fn apply_alteration(
        &mut self,
        table: &str,
        rows: RecordBatch,
        step: Alteration,
    ) -> BoltResult<RecordBatch> {
        let schema = rows.schema();
        let find = |c: &str| {
            schema.index_of(c).map_err(|_| {
                BoltError::Plan(format!("column '{c}' does not exist in table '{table}'"))
            })
        };
        let mut fields: Vec<ArrowField> =
            schema.fields().iter().map(|f| f.as_ref().clone()).collect();
        let mut cols: Vec<ArrayRef> = rows.columns().to_vec();
        match step {
            Alteration::Add { field, default } => {
                let n = rows.num_rows();
                let col = match default {
                    Some(e) if n > 0 => {
                        let v = self.with_dml_source(rows.clone(), |eng| {
                            eng.sql(&format!("SELECT ({e}) AS v FROM {DML_SOURCE}"))
                        })?;
                        let v = v.into_record_batch();
                        if v.num_rows() != n {
                            return Err(BoltError::Other(
                                "ADD COLUMN DEFAULT produced the wrong number of rows".into(),
                            ));
                        }
                        cast_column(v.column(0), field.data_type(), field.name())?
                    }
                    _ => new_null_array(field.data_type(), n),
                };
                if !field.is_nullable() && col.null_count() > 0 {
                    return Err(BoltError::Plan(format!(
                        "column '{}' is NOT NULL but existing rows would get NULL (add a DEFAULT)",
                        field.name()
                    )));
                }
                fields.push(field);
                cols.push(col);
            }
            Alteration::Drop(c) => {
                let i = find(&c)?;
                if fields.len() == 1 {
                    return Err(BoltError::Plan(format!(
                        "cannot drop '{c}', the only column of table '{table}'"
                    )));
                }
                fields.remove(i);
                cols.remove(i);
            }
            Alteration::Rename { from, to } => {
                let i = find(&from)?;
                if schema.index_of(&to).is_ok() {
                    return Err(BoltError::Plan(format!(
                        "column '{to}' already exists in table '{table}'"
                    )));
                }
                fields[i] = fields[i].clone().with_name(to);
            }
            Alteration::SetType { column, dtype } => {
                let i = find(&column)?;
                cols[i] = cast_column(&cols[i], &dtype, &column)?;
                fields[i] = fields[i].clone().with_data_type(dtype);
            }
            Alteration::SetNullable { column, nullable } => {
                let i = find(&column)?;
                if !nullable && cols[i].null_count() > 0 {
                    return Err(BoltError::Plan(format!(
                        "column '{column}' contains NULL values"
                    )));
                }
                fields[i] = fields[i].clone().with_nullable(nullable);
            }
        }
        RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), cols)
            .map_err(|e| BoltError::Other(format!("ALTER TABLE: {e}")))
    }

    /// Store `rows` (with their new schema) as table `name`.
    fn store_altered(&mut self, name: &str, rows: RecordBatch) -> BoltResult<()> {
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_mut() {
            if let Some(meta) = cat.table(name) {
                let schema =
                    crate::exec::engine_support::arrow_schema_to_plan_schema(&rows.schema())?;
                let cluster: Vec<String> = meta
                    .cluster_by
                    .iter()
                    .filter(|c| schema.fields.iter().any(|f| &f.name == *c))
                    .cloned()
                    .collect();
                let cluster = if cluster.len() == meta.cluster_by.len() {
                    cluster
                } else {
                    Vec::new()
                };
                cat.replace_table(name, &schema, std::slice::from_ref(&rows), cluster)?;
                return self.register_catalog_table(name);
            }
        }
        self.replace_table(name, rows)
    }

    /// `ALTER TABLE old RENAME TO new`.
    fn rename_table(&mut self, old: &str, new: &str) -> BoltResult<()> {
        self.ensure_name_free(new)?;
        self.ensure_no_dependent_views(old)?;
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_mut() {
            if cat.table(old).is_some() {
                cat.rename_table(old, new)?;
                self.deregister_table(old)?;
                self.register_catalog_table(new)?;
                self.touch_table(new);
                return Ok(());
            }
        }
        let schema = self.table_arrow_schema(old)?;
        let rows = self.full_rows(old, &schema)?;
        self.register_table(new, rows)?;
        self.deregister_table(old)?;
        self.touch_table(new);
        Ok(())
    }
}
