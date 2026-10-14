// SPDX-License-Identifier: Apache-2.0
//! SQL statements that change tables: `CREATE TABLE [AS]`, `CREATE EXTERNAL
//! TABLE`, `INSERT`, `DROP TABLE`, `TRUNCATE`, `COPY`, `ANALYZE`, plus the
//! read-only `SHOW TABLES` / `DESCRIBE` and the catalog-maintenance
//! statements `COMPACT TABLE` and `VACUUM`.
//!
//! [`Engine::execute_sql`] is the entry point (it takes `&mut self`;
//! [`Engine::sql`] stays query-only). With a persistent catalog attached
//! ([`EngineBuilder::database`](crate::EngineBuilder::database)) tables are
//! durable: `CREATE TABLE` / `INSERT` / `COPY ... FROM` write immutable
//! Parquet files and commit a new catalog snapshot, and catalog tables are
//! scanned through the storage layer (pruned per query). `CREATE TEMP
//! TABLE`, and every table of an engine without a catalog, lives in memory.

use std::sync::Arc;

use arrow::compute::{cast_with_options, CastOptions};
use arrow_array::{new_null_array, Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef,
};
use sqlparser::ast::{ColumnOption, CopySource, CopyTarget, ObjectName, ObjectType, Statement};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::error::{BoltError, BoltResult};
use crate::exec::engine::{Engine, QueryHandle};
use crate::plan::logical_plan::{DataType, Field, Schema, TimeUnit};
#[cfg(feature = "storage")]
use crate::storage::StorageTable;
use crate::storage::TableStatistics;

/// Flat engine table name of a (possibly qualified) SQL object name.
pub(crate) fn object_name(name: &ObjectName) -> String {
    name.0
        .iter()
        .map(|i| i.value.clone())
        .collect::<Vec<_>>()
        .join(".")
}

/// One-row `rows_affected` result of a statement.
fn status(rows: u64) -> QueryHandle {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "rows_affected",
        ArrowDataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![rows as i64]))])
        .expect("status batch");
    QueryHandle::from_record_batch(batch)
}

/// Lower a column type of a `CREATE TABLE` column list.
///
/// Matches on the type's SQL text so every spelling the parser accepts
/// (`INT`, `INTEGER`, `VARCHAR(20)`, `DECIMAL(10,2)`, `TIMESTAMP(3) WITH TIME
/// ZONE`, …) maps without depending on the parser's per-dialect variants.
pub(crate) fn ddl_type(text: &str) -> BoltResult<DataType> {
    let t = text.trim().to_ascii_uppercase();
    let (head, args) = match t.split_once('(') {
        Some((h, rest)) => {
            let (inside, _) = rest.split_once(')').unwrap_or((rest, ""));
            (h.trim().to_string(), Some(inside.to_string()))
        }
        None => (t.split_whitespace().next().unwrap_or("").to_string(), None),
    };
    let nums = |s: &Option<String>| -> Vec<u32> {
        s.as_deref()
            .map(|a| a.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_default()
    };
    Ok(match head.as_str() {
        "INT" | "INTEGER" | "INT4" | "INT32" | "SMALLINT" | "INT2" | "INT16" | "TINYINT"
        | "MEDIUMINT" => DataType::Int32,
        "BIGINT" | "INT8" | "INT64" | "LONG" => DataType::Int64,
        "REAL" | "FLOAT4" | "FLOAT32" => DataType::Float32,
        "DOUBLE" | "FLOAT" | "FLOAT8" | "FLOAT64" => DataType::Float64,
        "BOOL" | "BOOLEAN" => DataType::Bool,
        "VARCHAR" | "CHAR" | "CHARACTER" | "TEXT" | "STRING" | "NVARCHAR" | "NCHAR" | "CLOB" => {
            DataType::Utf8
        }
        "DATE" => DataType::Date32,
        "TIMESTAMP" | "DATETIME" | "TIMESTAMPTZ" => {
            let unit = match nums(&args).first() {
                Some(0) => TimeUnit::Second,
                Some(1..=3) => TimeUnit::Millisecond,
                Some(7..=9) => TimeUnit::Nanosecond,
                _ => TimeUnit::Microsecond,
            };
            let tz = (t.contains("WITH TIME ZONE") || head == "TIMESTAMPTZ")
                .then(|| crate::plan::logical_plan::intern_timezone("UTC"));
            DataType::Timestamp(unit, tz)
        }
        "DECIMAL" | "NUMERIC" | "DEC" => {
            let n = nums(&args);
            let p = n.first().copied().unwrap_or(38).clamp(1, 38) as u8;
            let s = n
                .get(1)
                .copied()
                .unwrap_or(if args.is_some() { 0 } else { 10 }) as i8;
            DataType::Decimal128(p, s)
        }
        _ => {
            // `CHARACTER VARYING` / `DOUBLE PRECISION` / `TIMESTAMP WITH ...`.
            if t.starts_with("CHARACTER VARYING") || t.starts_with("CHAR VARYING") {
                DataType::Utf8
            } else if t.starts_with("DOUBLE PRECISION") {
                DataType::Float64
            } else {
                return Err(BoltError::Sql(format!("column type not supported: {text}")));
            }
        }
    })
}

/// Cast / reorder `src` into `target`. `columns` names the target columns
/// `src`'s columns fill, positionally; empty means all of them in order.
/// Target columns not named are NULL (an error if they are non-nullable).
pub(crate) fn align_batch(
    src: &RecordBatch,
    target: &SchemaRef,
    columns: &[String],
) -> BoltResult<RecordBatch> {
    let positions: Vec<usize> = if columns.is_empty() {
        (0..target.fields().len()).collect()
    } else {
        columns
            .iter()
            .map(|c| {
                target.index_of(c).map_err(|_| {
                    BoltError::Plan(format!("column '{c}' does not exist in the target table"))
                })
            })
            .collect::<BoltResult<_>>()?
    };
    if src.num_columns() != positions.len() {
        return Err(BoltError::Plan(format!(
            "INSERT supplies {} columns but {} are expected",
            src.num_columns(),
            positions.len()
        )));
    }
    let opts = CastOptions {
        safe: false,
        ..Default::default()
    };
    let n = src.num_rows();
    let mut out: Vec<ArrayRef> = Vec::with_capacity(target.fields().len());
    for (i, field) in target.fields().iter().enumerate() {
        let col = match positions.iter().position(|&p| p == i) {
            Some(j) => {
                let a = src.column(j);
                if a.data_type() == field.data_type() {
                    Arc::clone(a)
                } else {
                    cast_with_options(a, field.data_type(), &opts).map_err(|e| {
                        BoltError::Type(format!(
                            "cannot store a value in column '{}': {e}",
                            field.name()
                        ))
                    })?
                }
            }
            None => new_null_array(field.data_type(), n),
        };
        if !field.is_nullable() && col.null_count() > 0 {
            return Err(BoltError::Plan(format!(
                "column '{}' is NOT NULL but would receive NULL values",
                field.name()
            )));
        }
        out.push(col);
    }
    RecordBatch::try_new(Arc::clone(target), out)
        .map_err(|e| BoltError::Other(format!("aligning rows: {e}")))
}

fn concat_or_empty(schema: SchemaRef, batches: &[RecordBatch]) -> BoltResult<RecordBatch> {
    let nonempty: Vec<&RecordBatch> = batches.iter().filter(|b| b.num_rows() > 0).collect();
    match nonempty.as_slice() {
        [] => Ok(RecordBatch::new_empty(schema)),
        [one] => Ok((*one).clone()),
        many => arrow::compute::concat_batches(&schema, many.iter().copied())
            .map_err(|e| BoltError::Other(format!("concatenating batches: {e}"))),
    }
}

/// `true` for the statements [`Engine::try_maintenance_statement`] handles.
fn is_maintenance_statement(sql: &str) -> bool {
    let upper: Vec<String> = sql
        .split_whitespace()
        .take(6)
        .map(|w| w.to_ascii_uppercase())
        .collect();
    let w: Vec<&str> = upper.iter().map(String::as_str).collect();
    let n = sql.split_whitespace().count();
    matches!(
        (w.as_slice(), n),
        (["COMPACT", "TABLE", ..], _)
            | (["VACUUM", ..], 1..=3)
            | (["REFRESH", "MATERIALIZED", "VIEW", ..], 4)
            | (["ANALYZE", ..], 2)
            | (["ANALYZE", "TABLE", ..], 3)
            | (["DESCRIBE" | "DESC", ..], 2)
            | (["DESCRIBE" | "DESC", "TABLE", ..], 3)
            | (["DROP", "MATERIALIZED", "VIEW", ..], 4)
            | (["DROP", "MATERIALIZED", "VIEW", "IF", "EXISTS", ..], 6)
    )
}

/// File format of a `COPY` target.
#[derive(Debug, Clone, Copy, PartialEq)]
enum CopyFormat {
    Parquet,
    Csv { header: bool, delimiter: u8 },
}

impl Engine {
    /// Execute one or more `;`-separated SQL statements and return the result
    /// of the last one.
    ///
    /// Queries behave exactly as through [`Engine::sql`]. Statements that
    /// change tables return a one-row `rows_affected` result. Supported:
    ///
    /// | Statement | Notes |
    /// |---|---|
    /// | `CREATE [TEMP] TABLE [IF NOT EXISTS] t (col type [NOT NULL], ...)` | empty table |
    /// | `CREATE [OR REPLACE] [TEMP] TABLE t [(cols)] AS SELECT ...` | |
    /// | `CREATE EXTERNAL TABLE t STORED AS PARQUET LOCATION 'path'` | file or directory |
    /// | `INSERT INTO t [(cols)] SELECT ... \| VALUES (...)` | |
    /// | `DROP TABLE [IF EXISTS] t, ...` / `TRUNCATE [TABLE] t` | |
    /// | `COPY t [(cols)] \| (query) TO 'file' [(FORMAT parquet\|csv, HEADER b)]` | |
    /// | `COPY t [(cols)] FROM 'file' [(FORMAT ...)]` | appends |
    /// | `ANALYZE [TABLE] t` | per-column statistics |
    /// | `SHOW TABLES` / `DESCRIBE t` | |
    /// | `COMPACT TABLE t` / `VACUUM [KEEP n]` | catalog tables only |
    /// | `CREATE [OR REPLACE] MATERIALIZED VIEW v [WITH (refresh = 'on_commit')] AS SELECT ...`, `REFRESH MATERIALIZED VIEW v`, `DROP MATERIALIZED VIEW v` | see [`Engine::view_status`] |
    /// | `DELETE FROM t [WHERE ...]` / `UPDATE t SET c = expr, ... [WHERE ...]` | rewrites the affected files |
    /// | `ALTER TABLE t ADD [COLUMN] c type [DEFAULT e]`, `DROP [COLUMN] c`, `RENAME [COLUMN] a TO b`, `ALTER [COLUMN] c TYPE t \| SET NOT NULL \| DROP NOT NULL`, `RENAME TO u` | |
    /// | `BEGIN` / `START TRANSACTION`, `COMMIT` / `END`, `ROLLBACK` | see [`Engine::begin_transaction`] |
    ///
    /// Without a file-format option, `COPY` infers it from the file
    /// extension (`.csv` → CSV with a header, anything else → Parquet).
    ///
    /// Outside a transaction each statement commits on its own; a failing
    /// statement leaves the tables as the previous statement left them, and
    /// later statements do not run. Inside one (see the
    /// [`engine_txn`](crate::exec::engine_txn) docs) a failure aborts the
    /// transaction.
    pub fn execute_sql(&mut self, sql: &str) -> BoltResult<QueryHandle> {
        let trimmed = sql.trim().trim_end_matches(';').trim();
        if is_maintenance_statement(trimmed) {
            self.ensure_transaction_usable()?;
            let r = self.count_statement(|e| {
                e.try_maintenance_statement(trimmed).unwrap_or_else(|| {
                    Err(BoltError::Sql(format!("malformed statement: {trimmed}")))
                })
            });
            if r.is_err() {
                self.abort_transaction_on_error();
            }
            return r;
        }
        crate::plan::sql_frontend::guard_sql_size(sql)?;
        let stmts = Parser::parse_sql(&GenericDialect {}, sql)
            .map_err(|e| crate::plan::sql_frontend::parse_error_to_bolt_error(e, sql))?;
        if stmts.is_empty() {
            return Err(BoltError::Sql("no SQL statement to execute".into()));
        }
        let mut last = None;
        for stmt in stmts {
            let control = matches!(
                stmt,
                Statement::StartTransaction { .. }
                    | Statement::Commit { .. }
                    | Statement::Rollback { .. }
            );
            if control {
                last = Some(self.count_statement(|e| e.transaction_statement(stmt))?);
                continue;
            }
            self.ensure_transaction_usable()?;
            let r = (|| {
                // Catch up on-commit views with changes made through the
                // Rust registration API since the last statement.
                self.refresh_stale_on_commit_views()?;
                match stmt {
                    // `sql()` does its own query accounting.
                    Statement::Query(_) => self.sql(&stmt.to_string()),
                    other => self.count_statement(|e| e.execute_statement(other)),
                }
            })();
            match r {
                Ok(h) => last = Some(h),
                Err(e) => {
                    self.abort_transaction_on_error();
                    return Err(e);
                }
            }
        }
        Ok(last.expect("at least one statement"))
    }

    /// `BEGIN` / `COMMIT` / `ROLLBACK`.
    fn transaction_statement(&mut self, stmt: Statement) -> BoltResult<QueryHandle> {
        match stmt {
            Statement::StartTransaction { modifier: None, .. } => self.begin_transaction()?,
            Statement::Commit { chain: false } => self.commit_transaction()?,
            Statement::Rollback {
                chain: false,
                savepoint: None,
            } => self.rollback_transaction()?,
            other => {
                return Err(BoltError::Sql(format!(
                    "transaction statement not supported: {other}"
                )))
            }
        }
        Ok(status(0))
    }

    /// `QueriesTotal` / `QueriesFailed` accounting for a non-query statement.
    fn count_statement(
        &mut self,
        f: impl FnOnce(&mut Engine) -> BoltResult<QueryHandle>,
    ) -> BoltResult<QueryHandle> {
        let m = crate::metrics::metrics();
        m.inc(crate::metrics::Counter::QueriesTotal);
        let r = f(self);
        if r.is_err() {
            m.inc(crate::metrics::Counter::QueriesFailed);
        }
        r
    }

    /// Statements the SQL parser does not know: `COMPACT TABLE t`,
    /// `VACUUM [KEEP n]`, `REFRESH MATERIALIZED VIEW v`.
    fn try_maintenance_statement(&mut self, sql: &str) -> Option<BoltResult<QueryHandle>> {
        let words: Vec<&str> = sql.split_whitespace().collect();
        let upper: Vec<String> = words.iter().map(|w| w.to_ascii_uppercase()).collect();
        let strip = |w: &str| w.trim_matches('"').to_string();
        match upper
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice()
        {
            ["COMPACT", "TABLE", _] => Some(
                self.compact_table(&strip(words[2]), None)
                    .map(|r| status(r as u64)),
            ),
            ["COMPACT", "TABLE", _, "ORDER", "BY", ..] => {
                let cols: Vec<String> = words[5..]
                    .join(" ")
                    .trim_matches(|c| c == '(' || c == ')')
                    .split(',')
                    .map(|c| c.trim().trim_matches('"').to_string())
                    .filter(|c| !c.is_empty())
                    .collect();
                Some(if cols.is_empty() {
                    Err(BoltError::Sql(
                        "COMPACT TABLE ... ORDER BY needs columns".into(),
                    ))
                } else {
                    self.compact_table(&strip(words[2]), Some(&cols))
                        .map(|r| status(r as u64))
                })
            }
            ["VACUUM"] => Some(self.vacuum(1).map(|r| status(r as u64))),
            ["VACUUM", "KEEP", n] => Some(match n.parse::<usize>() {
                Ok(n) => self.vacuum(n).map(|r| status(r as u64)),
                Err(_) => Err(BoltError::Sql(format!(
                    "VACUUM KEEP expects a number, got '{n}'"
                ))),
            }),
            ["REFRESH", "MATERIALIZED", "VIEW", _] => {
                Some(self.refresh_view(&strip(words[3])).map(status))
            }
            ["ANALYZE", _] | ["ANALYZE", "TABLE", _] => {
                Some(self.analyze_table(&strip(words[words.len() - 1])))
            }
            ["DESCRIBE" | "DESC", _] | ["DESCRIBE" | "DESC", "TABLE", _] => {
                Some(self.describe_table(&strip(words[words.len() - 1])))
            }
            // The SQL parser has no MATERIALIZED VIEW object type for DROP.
            ["DROP", "MATERIALIZED", "VIEW", _] => {
                Some(self.drop_view(&strip(words[3])).map(|_| status(0)))
            }
            ["DROP", "MATERIALIZED", "VIEW", "IF", "EXISTS", _] => {
                let name = strip(words[5]);
                Some(if self.view_exists(&name) {
                    self.drop_view(&name).map(|_| status(0))
                } else {
                    Ok(status(0))
                })
            }
            _ => None,
        }
    }

    fn execute_statement(&mut self, stmt: Statement) -> BoltResult<QueryHandle> {
        match stmt {
            Statement::CreateTable(ct) => {
                let name = object_name(&ct.name);
                if ct.external || ct.location.is_some() {
                    let format_ok = ct
                        .file_format
                        .as_ref()
                        .is_none_or(|f| matches!(f, sqlparser::ast::FileFormat::PARQUET));
                    let location = ct.location.clone().ok_or_else(|| {
                        BoltError::Sql("CREATE EXTERNAL TABLE needs a LOCATION".into())
                    })?;
                    if !format_ok {
                        return Err(BoltError::Sql(
                            "external tables must be STORED AS PARQUET".into(),
                        ));
                    }
                    if self.has_table(&name) {
                        if ct.if_not_exists {
                            return Ok(status(0));
                        }
                        if !ct.or_replace {
                            return Err(BoltError::Plan(format!("table '{name}' already exists")));
                        }
                        self.drop_table(&name)?;
                    }
                    self.create_external_table(&name, &location)?;
                    return Ok(status(0));
                }
                if self.has_table(&name) {
                    if ct.if_not_exists {
                        return Ok(status(0));
                    }
                    if !ct.or_replace {
                        return Err(BoltError::Plan(format!("table '{name}' already exists")));
                    }
                }
                let declared: Option<Schema> = if ct.columns.is_empty() {
                    None
                } else {
                    let fields = ct
                        .columns
                        .iter()
                        .map(|c| {
                            let not_null = c
                                .options
                                .iter()
                                .any(|o| matches!(o.option, ColumnOption::NotNull));
                            Ok(Field::new(
                                c.name.value.clone(),
                                ddl_type(&c.data_type.to_string())?,
                                !not_null,
                            ))
                        })
                        .collect::<BoltResult<Vec<_>>>()?;
                    Some(Schema::new(fields))
                };
                let batch = match (&ct.query, &declared) {
                    (Some(q), decl) => {
                        let rows = self.sql(&q.to_string())?.into_record_batch();
                        match decl {
                            Some(schema) => {
                                let target =
                                    crate::exec::schema_convert::plan_schema_to_arrow_schema(
                                        schema,
                                    )?;
                                align_batch(&rows, &target, &[])?
                            }
                            None => rows,
                        }
                    }
                    (None, Some(schema)) => RecordBatch::new_empty(
                        crate::exec::schema_convert::plan_schema_to_arrow_schema(schema)?,
                    ),
                    (None, None) => {
                        return Err(BoltError::Sql(
                            "CREATE TABLE needs a column list or AS SELECT".into(),
                        ))
                    }
                };
                if self.has_table(&name) {
                    self.drop_table(&name)?;
                }
                let rows = batch.num_rows() as u64;
                self.create_table(&name, batch, ct.temporary)?;
                Ok(status(rows))
            }
            Statement::Insert(ins) => {
                let name = object_name(&ins.table_name);
                let source = ins.source.ok_or_else(|| {
                    BoltError::Sql("INSERT needs a SELECT or VALUES source".into())
                })?;
                let rows = self.sql(&source.to_string())?.into_record_batch();
                let columns: Vec<String> = ins.columns.iter().map(|c| c.value.clone()).collect();
                let n = self.insert_rows(&name, &rows, &columns)?;
                Ok(status(n))
            }
            Statement::Drop {
                object_type,
                if_exists,
                names,
                ..
            } => {
                for n in &names {
                    let name = object_name(n);
                    match object_type {
                        ObjectType::Table => {
                            if self.view_exists(&name) {
                                return Err(BoltError::Plan(format!(
                                    "'{name}' is a materialized view; use DROP MATERIALIZED VIEW"
                                )));
                            }
                            if !self.has_table(&name) {
                                if if_exists {
                                    continue;
                                }
                                return Err(BoltError::Plan(format!(
                                    "table '{name}' does not exist"
                                )));
                            }
                            self.ensure_droppable_in_transaction(&name)?;
                            self.drop_table(&name)?;
                        }
                        ObjectType::View => {
                            if !self.view_exists(&name) {
                                if if_exists {
                                    continue;
                                }
                                return Err(BoltError::Plan(format!(
                                    "materialized view '{name}' does not exist"
                                )));
                            }
                            self.drop_view(&name)?;
                        }
                        other => {
                            return Err(BoltError::Sql(format!("DROP {other} is not supported")));
                        }
                    }
                }
                Ok(status(0))
            }
            Statement::Delete(d) => self.execute_delete(d).map(status),
            Statement::Update {
                table,
                assignments,
                from,
                selection,
                returning,
            } => self
                .execute_update(table, assignments, from, selection, returning)
                .map(status),
            Statement::AlterTable {
                name,
                if_exists,
                operations,
                location: None,
                on_cluster: None,
                only: false,
            } => self
                .execute_alter_table(&name, if_exists, operations)
                .map(|()| status(0)),
            Statement::Truncate { table_names, .. } => {
                for t in &table_names {
                    self.truncate_table(&object_name(&t.name))?;
                }
                Ok(status(0))
            }
            Statement::Copy {
                source,
                to,
                target,
                options,
                legacy_options,
                ..
            } => {
                let path = match target {
                    CopyTarget::File { filename } => filename,
                    other => {
                        return Err(BoltError::Sql(format!(
                            "COPY supports only file targets, not {other}"
                        )))
                    }
                };
                let format = copy_format(&path, &options, &legacy_options)?;
                if to {
                    let query = match &source {
                        CopySource::Table {
                            table_name,
                            columns,
                        } => {
                            let cols = if columns.is_empty() {
                                "*".to_string()
                            } else {
                                columns
                                    .iter()
                                    .map(|c| format!("\"{}\"", c.value))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            };
                            format!("SELECT {cols} FROM {table_name}")
                        }
                        CopySource::Query(q) => q.to_string(),
                    };
                    let rows = self.sql(&query)?.into_record_batch();
                    write_copy_file(std::path::Path::new(&path), &rows, format)?;
                    Ok(status(rows.num_rows() as u64))
                } else {
                    let (name, columns) = match source {
                        CopySource::Table {
                            table_name,
                            columns,
                        } => (
                            object_name(&table_name),
                            columns.into_iter().map(|c| c.value).collect::<Vec<_>>(),
                        ),
                        CopySource::Query(_) => {
                            return Err(BoltError::Sql("COPY (query) FROM is not valid".into()))
                        }
                    };
                    let target = self.table_arrow_schema(&name)?;
                    let read_schema: SchemaRef = if columns.is_empty() {
                        Arc::clone(&target)
                    } else {
                        let fields = columns
                            .iter()
                            .map(|c| {
                                target.field_with_name(c).cloned().map_err(|_| {
                                    BoltError::Plan(format!("column '{c}' does not exist"))
                                })
                            })
                            .collect::<BoltResult<Vec<_>>>()?;
                        Arc::new(ArrowSchema::new(fields))
                    };
                    let rows = read_copy_file(std::path::Path::new(&path), &read_schema, format)?;
                    let n = self.insert_rows(&name, &rows, &columns)?;
                    Ok(status(n))
                }
            }
            Statement::Analyze { table_name, .. } => self.analyze_table(&object_name(&table_name)),
            Statement::ShowTables { .. } => {
                let names = self.table_names();
                let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                    "table_name",
                    ArrowDataType::Utf8,
                    false,
                )]));
                let batch = RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(names))])
                    .map_err(|e| BoltError::Other(e.to_string()))?;
                Ok(QueryHandle::from_record_batch(batch))
            }
            Statement::ExplainTable { table_name, .. } => {
                self.describe_table(&object_name(&table_name))
            }
            Statement::CreateView {
                materialized: true,
                or_replace,
                name,
                query,
                options,
                ..
            } => {
                let policy = view_refresh_policy(&options)?;
                self.create_view(&object_name(&name), &query.to_string(), policy, or_replace)
                    .map(status)
            }
            Statement::CreateView { .. } => Err(BoltError::Sql(
                "only CREATE MATERIALIZED VIEW is supported (plain views are not)".into(),
            )),
            other => Err(BoltError::Sql(format!("statement not supported: {other}"))),
        }
    }

    /// Arrow schema rows must have to be stored in table `name`.
    pub(crate) fn table_arrow_schema(&self, name: &str) -> BoltResult<SchemaRef> {
        if let Some(b) = self.tables.get(name).and_then(|b| b.first()) {
            return Ok(b.schema());
        }
        if let Some(t) = self.storage_table(name) {
            return crate::exec::schema_convert::plan_schema_to_arrow_schema(t.schema());
        }
        let schema = crate::plan::TableProvider::schema(&self.provider, name)?;
        crate::exec::schema_convert::plan_schema_to_arrow_schema(&schema)
    }

    /// Create table `name` holding `batch` — durable in the attached catalog
    /// unless `temporary` (or no catalog is attached).
    pub(crate) fn create_table(
        &mut self,
        name: &str,
        batch: RecordBatch,
        temporary: bool,
    ) -> BoltResult<()> {
        self.ensure_name_free(name)?;
        #[cfg(feature = "storage")]
        if !temporary {
            if let Some(cat) = self.catalog.as_mut() {
                let schema =
                    crate::exec::engine_support::arrow_schema_to_plan_schema(&batch.schema())?;
                cat.create_table(name, &schema, std::slice::from_ref(&batch))?;
                return self.register_catalog_table(name);
            }
        }
        let _ = temporary;
        self.register_table(name, batch)
    }

    /// Register external Parquet data as table `name` (recorded in the
    /// catalog when one is attached).
    fn create_external_table(&mut self, name: &str, location: &str) -> BoltResult<()> {
        #[cfg(feature = "storage")]
        {
            if let Some(cat) = self.catalog.as_mut() {
                cat.create_external_table(name, location)?;
                return self.register_catalog_table(name);
            }
            self.register_parquet(name, location)
        }
        #[cfg(not(feature = "storage"))]
        {
            let _ = (name, location);
            Err(BoltError::Plan(
                "external tables need the `storage` feature".into(),
            ))
        }
    }

    /// Append `rows` (filling `columns`, or all columns) to table `name`.
    /// Returns the number of rows stored.
    pub(crate) fn insert_rows(
        &mut self,
        name: &str,
        rows: &RecordBatch,
        columns: &[String],
    ) -> BoltResult<u64> {
        if self.view_exists(name) {
            return Err(BoltError::Plan(format!(
                "'{name}' is a materialized view; it changes only by REFRESH"
            )));
        }
        let target = self.table_arrow_schema(name)?;
        let aligned = align_batch(rows, &target, columns)?;
        let n = aligned.num_rows() as u64;
        if n == 0 {
            return Ok(0);
        }
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_mut() {
            if cat.table(name).is_some() {
                cat.append(name, std::slice::from_ref(&aligned))?;
                // Compaction policy: merge the table's files (keeping its
                // clustering key) once appends have produced too many.
                if let Some(max) = self.auto_compact_files {
                    if cat.table(name).is_some_and(|t| t.files.len() > max) {
                        cat.compact(name, None)?;
                    }
                }
                self.register_catalog_table(name)?;
                self.after_table_change(name)?;
                return Ok(n);
            }
        }
        if self.tables.contains_key(name) {
            self.register_batch(name, aligned)?;
            self.after_table_change(name)?;
            return Ok(n);
        }
        Err(BoltError::Plan(format!(
            "table '{name}' is read-only (lazy streams and registered storage tables \
             cannot be inserted into)"
        )))
    }

    /// Append `batch` to table `name`, matching columns by position and
    /// casting to the column types. An in-memory table grows in place; a
    /// catalog table gets one new immutable data file and a new snapshot.
    /// Returns the number of rows stored.
    ///
    /// # Errors
    /// The column count differs, a value cannot be cast, a `NOT NULL` column
    /// would receive NULLs, or the table is read-only (lazy stream, external
    /// or registered storage table, materialized view).
    pub fn insert_batch(&mut self, name: &str, batch: RecordBatch) -> BoltResult<u64> {
        self.insert_rows(name, &batch, &[])
    }

    /// Append a stream of batches to table `name`, buffering up to
    /// `flush_rows` rows per commit: a catalog table gets one data file (and
    /// one snapshot) per flush instead of one per batch. Returns the number
    /// of rows stored.
    ///
    /// Each flush commits on its own: if the stream yields an error, the
    /// rows already flushed stay in the table and the error is returned.
    pub fn ingest_stream<I>(&mut self, name: &str, batches: I, flush_rows: usize) -> BoltResult<u64>
    where
        I: IntoIterator<Item = BoltResult<RecordBatch>>,
    {
        let flush_rows = flush_rows.max(1);
        let mut pending: Vec<RecordBatch> = Vec::new();
        let mut pending_rows = 0usize;
        let mut stored = 0u64;
        let flush = |engine: &mut Engine, pending: &mut Vec<RecordBatch>| -> BoltResult<u64> {
            if pending.is_empty() {
                return Ok(0);
            }
            let schema = pending[0].schema();
            let rows = concat_or_empty(schema, pending)?;
            pending.clear();
            engine.insert_rows(name, &rows, &[])
        };
        for batch in batches {
            let batch = batch?;
            if let Some(first) = pending.first() {
                if first.schema() != batch.schema() {
                    // A schema change starts a new buffer (each is aligned
                    // to the table on its own).
                    stored += flush(self, &mut pending)?;
                    pending_rows = 0;
                }
            }
            pending_rows += batch.num_rows();
            pending.push(batch);
            if pending_rows >= flush_rows {
                stored += flush(self, &mut pending)?;
                pending_rows = 0;
            }
        }
        stored += flush(self, &mut pending)?;
        Ok(stored)
    }

    /// Remove table `name` from the engine, whatever kind it is. With a
    /// catalog attached, a catalog table is dropped from the catalog too.
    pub fn drop_table(&mut self, name: &str) -> BoltResult<()> {
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_mut() {
            if cat.table(name).is_some() {
                cat.drop_table(name)?;
            }
        }
        self.deregister_table(name)
    }

    /// Unregister table `name` from this engine (in-memory, lazy or storage)
    /// without touching any catalog. Device and host copies are released.
    pub fn deregister_table(&mut self, name: &str) -> BoltResult<()> {
        if !self.has_table(name) {
            return Err(BoltError::Plan(format!("table '{name}' is not registered")));
        }
        self.bump_data_epoch();
        self.tables.remove(name);
        self.streaming_sources.borrow_mut().remove(name);
        self.storage_tables.borrow_mut().remove(name);
        self.gpu_tables.borrow_mut().remove(name);
        self.overlay_uploads.borrow_mut().remove(name);
        self.gpu_residency.borrow_mut().remove(name);
        self.analyzed.borrow_mut().remove(name);
        self.dict_registry.get_mut().unregister_table(name);
        self.provider.unregister_table(name);
        self.forget_host_revisions(name);
        Ok(())
    }

    /// Remove every row of table `name`.
    pub fn truncate_table(&mut self, name: &str) -> BoltResult<()> {
        if self.view_exists(name) {
            return Err(BoltError::Plan(format!("'{name}' is a materialized view")));
        }
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_mut() {
            if cat.table(name).is_some() {
                cat.truncate(name)?;
                self.register_catalog_table(name)?;
                return self.after_table_change(name);
            }
        }
        if self.tables.contains_key(name) {
            let schema = self.table_arrow_schema(name)?;
            self.replace_table(name, RecordBatch::new_empty(schema))?;
            return self.after_table_change(name);
        }
        Err(BoltError::Plan(format!(
            "table '{name}' is read-only or does not exist"
        )))
    }

    /// Compute exact statistics of table `name` over all its rows (recorded
    /// in the catalog for a catalog table) and return them, one row per
    /// column.
    pub fn analyze_table(&mut self, name: &str) -> BoltResult<QueryHandle> {
        let stats = self.full_statistics(name)?;
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_mut() {
            if cat.table(name).is_some() && !cat.is_read_only() {
                cat.set_statistics(name, stats.clone())?;
            }
        }
        // Valid until the table's data next changes; feeds the optimizer.
        if let Some(v) = self.table_version(name) {
            self.analyzed
                .borrow_mut()
                .insert(name.to_string(), (v, stats.clone()));
        }
        let schema = self.table_arrow_schema(name)?;
        let render =
            |l: &Option<crate::plan::logical_plan::Literal>| l.as_ref().map(render_literal);
        let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
        let cols = &stats.columns;
        let out_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("column_name", ArrowDataType::Utf8, false),
            ArrowField::new("row_count", ArrowDataType::Int64, false),
            ArrowField::new("null_count", ArrowDataType::Int64, true),
            ArrowField::new("min", ArrowDataType::Utf8, true),
            ArrowField::new("max", ArrowDataType::Utf8, true),
            ArrowField::new("distinct_estimate", ArrowDataType::Int64, true),
        ]));
        let n = names.len();
        let get = |i: usize| cols.get(i).cloned().unwrap_or_default();
        let batch = RecordBatch::try_new(
            out_schema,
            vec![
                Arc::new(StringArray::from(names)),
                Arc::new(Int64Array::from(vec![stats.row_count as i64; n])),
                Arc::new(Int64Array::from(
                    (0..n)
                        .map(|i| get(i).null_count.map(|v| v as i64))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    (0..n).map(|i| render(&get(i).min)).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    (0..n).map(|i| render(&get(i).max)).collect::<Vec<_>>(),
                )),
                Arc::new(Int64Array::from(
                    (0..n)
                        .map(|i| get(i).distinct_estimate.map(|v| v as i64))
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .map_err(|e| BoltError::Other(e.to_string()))?;
        Ok(QueryHandle::from_record_batch(batch))
    }

    /// Exact statistics over every row of `name` (storage tables are read in
    /// full).
    fn full_statistics(&self, name: &str) -> BoltResult<TableStatistics> {
        if let Some(t) = self.storage_table(name) {
            let all = t.plan_scan(&crate::storage::ScanRequest::full());
            let batches: Vec<RecordBatch> = t.read(&all)?.collect::<BoltResult<_>>()?;
            if batches.is_empty() {
                return Ok(TableStatistics {
                    columns: vec![Default::default(); t.schema().fields.len()],
                    ..Default::default()
                });
            }
            return Ok(TableStatistics::from_batches(&batches));
        }
        let schema_len = self.table_arrow_schema(name)?.fields().len();
        let mut s = self.table_statistics(name)?;
        if s.columns.is_empty() {
            s.columns = vec![Default::default(); schema_len];
        }
        Ok(s)
    }

    fn describe_table(&self, name: &str) -> BoltResult<QueryHandle> {
        if !self.has_table(name) {
            return Err(BoltError::Plan(format!("table '{name}' does not exist")));
        }
        let schema = if let Some(t) = self.storage_table(name) {
            t.schema().clone()
        } else {
            let arrow = self.table_arrow_schema(name)?;
            crate::exec::engine_support::arrow_schema_to_plan_schema(arrow.as_ref())?
        };
        let out = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("column_name", ArrowDataType::Utf8, false),
            ArrowField::new("data_type", ArrowDataType::Utf8, false),
            ArrowField::new("nullable", ArrowDataType::Boolean, false),
        ]));
        let batch = RecordBatch::try_new(
            out,
            vec![
                Arc::new(StringArray::from(
                    schema
                        .fields
                        .iter()
                        .map(|f| f.name.clone())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    schema
                        .fields
                        .iter()
                        .map(|f| sql_type_name(f.dtype))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(arrow_array::BooleanArray::from(
                    schema.fields.iter().map(|f| f.nullable).collect::<Vec<_>>(),
                )),
            ],
        )
        .map_err(|e| BoltError::Other(e.to_string()))?;
        Ok(QueryHandle::from_record_batch(batch))
    }

    /// Rewrite a catalog table's data files into as few as possible —
    /// sorted on `cluster_by` (recorded as the table's clustering key), or on
    /// its recorded key when `None`. Returns the file count afterwards.
    pub fn compact_table(
        &mut self,
        name: &str,
        cluster_by: Option<&[String]>,
    ) -> BoltResult<usize> {
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_mut() {
            if cat.table(name).is_some() {
                let r = cat.compact(name, cluster_by)?;
                self.register_catalog_table(name)?;
                return Ok(r.files_after);
            }
        }
        let _ = cluster_by;
        Err(BoltError::Plan(format!(
            "COMPACT: '{name}' is not a catalog table"
        )))
    }

    /// Delete catalog snapshots older than the newest `keep_snapshots` and
    /// every data file no remaining snapshot references. Returns the number
    /// of files deleted.
    pub fn vacuum(&mut self, keep_snapshots: usize) -> BoltResult<usize> {
        #[cfg(feature = "storage")]
        if let Some(cat) = self.catalog.as_mut() {
            let r = cat.vacuum(keep_snapshots)?;
            return Ok(r.files_removed);
        }
        let _ = keep_snapshots;
        Err(BoltError::Plan("VACUUM needs an attached catalog".into()))
    }
}

/// `SQL`-style name of a plan type, for `DESCRIBE`.
fn sql_type_name(d: DataType) -> String {
    match d {
        DataType::Bool => "BOOLEAN".into(),
        DataType::Int32 => "INTEGER".into(),
        DataType::Int64 => "BIGINT".into(),
        DataType::Float32 => "REAL".into(),
        DataType::Float64 => "DOUBLE".into(),
        DataType::Utf8 => "VARCHAR".into(),
        DataType::Date32 => "DATE".into(),
        DataType::Decimal128(p, s) => format!("DECIMAL({p},{s})"),
        DataType::Timestamp(u, tz) => {
            let p = match u {
                TimeUnit::Second => 0,
                TimeUnit::Millisecond => 3,
                TimeUnit::Microsecond => 6,
                TimeUnit::Nanosecond => 9,
            };
            match tz {
                Some(tz) => format!("TIMESTAMP({p}) WITH TIME ZONE '{tz}'"),
                None => format!("TIMESTAMP({p})"),
            }
        }
        #[allow(unreachable_patterns)]
        other => format!("{other:?}"),
    }
}

/// Human-readable statistic bound.
fn render_literal(l: &crate::plan::logical_plan::Literal) -> String {
    use crate::plan::logical_plan::Literal as L;
    match l {
        L::Null => "NULL".into(),
        L::Bool(b) => b.to_string(),
        L::Int32(v) => v.to_string(),
        L::Int64(v) => v.to_string(),
        L::Float32(v) => v.to_string(),
        L::Float64(v) => v.to_string(),
        L::Utf8(s) => s.clone(),
        L::Date32(d) => d.to_string(),
        L::Decimal128(v, _, s) => {
            let s = *s as u32;
            if s == 0 {
                v.to_string()
            } else {
                let p = 10i128.pow(s);
                let sign = if *v < 0 { "-" } else { "" };
                format!(
                    "{sign}{}.{:0width$}",
                    (v / p).abs(),
                    (v % p).abs(),
                    width = s as usize
                )
            }
        }
        L::Timestamp(v, _, _) => v.to_string(),
    }
}

/// `WITH (refresh = 'manual' | 'on_commit')` of a materialized view.
fn view_refresh_policy(
    options: &sqlparser::ast::CreateTableOptions,
) -> BoltResult<crate::exec::engine_views::RefreshMode> {
    use crate::exec::engine_views::RefreshMode;
    let opts = match options {
        sqlparser::ast::CreateTableOptions::With(o)
        | sqlparser::ast::CreateTableOptions::Options(o) => o.as_slice(),
        sqlparser::ast::CreateTableOptions::None => &[],
    };
    let mut mode = RefreshMode::Manual;
    for o in opts {
        let text = o.to_string();
        let (k, v) = text
            .split_once('=')
            .ok_or_else(|| BoltError::Sql(format!("unsupported view option {text}")))?;
        if !k.trim().eq_ignore_ascii_case("refresh") {
            return Err(BoltError::Sql(format!("unsupported view option {text}")));
        }
        mode = match v.trim().trim_matches('\'').to_ascii_lowercase().as_str() {
            "manual" => RefreshMode::Manual,
            "on_commit" => RefreshMode::OnCommit,
            other => return Err(BoltError::Sql(format!("unknown refresh policy '{other}'"))),
        };
    }
    Ok(mode)
}

fn copy_format(
    path: &str,
    options: &[sqlparser::ast::CopyOption],
    legacy: &[sqlparser::ast::CopyLegacyOption],
) -> BoltResult<CopyFormat> {
    use sqlparser::ast::{CopyLegacyOption, CopyOption};
    let mut format: Option<String> = None;
    let mut header = true;
    let mut delimiter = b',';
    for o in options {
        match o {
            CopyOption::Format(f) => format = Some(f.value.to_ascii_lowercase()),
            CopyOption::Header(h) => header = *h,
            CopyOption::Delimiter(c) => delimiter = *c as u8,
            other => {
                return Err(BoltError::Sql(format!(
                    "COPY option not supported: {other}"
                )))
            }
        }
    }
    for o in legacy {
        match o {
            CopyLegacyOption::Csv(_) => format = Some("csv".into()),
            CopyLegacyOption::Delimiter(c) => delimiter = *c as u8,
            other => {
                return Err(BoltError::Sql(format!(
                    "COPY option not supported: {other}"
                )))
            }
        }
    }
    let format = format.unwrap_or_else(|| {
        if path.to_ascii_lowercase().ends_with(".csv") {
            "csv".into()
        } else {
            "parquet".into()
        }
    });
    match format.as_str() {
        "parquet" => Ok(CopyFormat::Parquet),
        "csv" => Ok(CopyFormat::Csv { header, delimiter }),
        other => Err(BoltError::Sql(format!(
            "COPY format not supported: {other}"
        ))),
    }
}

fn write_copy_file(
    path: &std::path::Path,
    rows: &RecordBatch,
    format: CopyFormat,
) -> BoltResult<()> {
    match format {
        CopyFormat::Parquet => {
            #[cfg(feature = "storage")]
            {
                crate::storage::write_parquet(
                    path,
                    std::slice::from_ref(rows),
                    &Default::default(),
                )?;
                Ok(())
            }
            #[cfg(not(feature = "storage"))]
            {
                let _ = (path, rows);
                Err(BoltError::Plan(
                    "COPY ... TO Parquet needs the `storage` feature".into(),
                ))
            }
        }
        CopyFormat::Csv { header, delimiter } => {
            let f = std::fs::File::create(path)
                .map_err(|e| BoltError::Other(format!("COPY: creating {}: {e}", path.display())))?;
            let mut w = arrow::csv::WriterBuilder::new()
                .with_header(header)
                .with_delimiter(delimiter)
                .build(f);
            w.write(rows)
                .map_err(|e| BoltError::Other(format!("COPY: writing CSV: {e}")))
        }
    }
}

fn read_copy_file(
    path: &std::path::Path,
    schema: &SchemaRef,
    format: CopyFormat,
) -> BoltResult<RecordBatch> {
    match format {
        CopyFormat::Parquet => {
            #[cfg(feature = "storage")]
            {
                let t = crate::storage::ParquetTable::open(path)?;
                let all = t.plan_scan(&crate::storage::ScanRequest::full());
                let batches: Vec<RecordBatch> = t.read(&all)?.collect::<BoltResult<_>>()?;
                let src_schema = Arc::clone(t.arrow_schema());
                let src = concat_or_empty(src_schema, &batches)?;
                // Match the file's columns to the target by name.
                let mut cols = Vec::with_capacity(schema.fields().len());
                for f in schema.fields() {
                    let c = src.column_by_name(f.name()).ok_or_else(|| {
                        BoltError::Plan(format!(
                            "COPY: file {} has no column '{}'",
                            path.display(),
                            f.name()
                        ))
                    })?;
                    cols.push(Arc::clone(c));
                }
                let fields: Vec<ArrowField> = schema
                    .fields()
                    .iter()
                    .zip(&cols)
                    .map(|(f, c)| ArrowField::new(f.name(), c.data_type().clone(), true))
                    .collect();
                RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), cols)
                    .map_err(|e| BoltError::Other(e.to_string()))
            }
            #[cfg(not(feature = "storage"))]
            {
                let _ = (path, schema);
                Err(BoltError::Plan(
                    "COPY ... FROM Parquet needs the `storage` feature".into(),
                ))
            }
        }
        CopyFormat::Csv { header, delimiter } => {
            let f = std::fs::File::open(path)
                .map_err(|e| BoltError::Other(format!("COPY: opening {}: {e}", path.display())))?;
            // Parse as the target types, every column nullable (empty → NULL).
            let read_schema = Arc::new(ArrowSchema::new(
                schema
                    .fields()
                    .iter()
                    .map(|f| {
                        let dt = match f.data_type() {
                            ArrowDataType::Dictionary(_, v) => v.as_ref().clone(),
                            other => other.clone(),
                        };
                        ArrowField::new(f.name(), dt, true)
                    })
                    .collect::<Vec<_>>(),
            ));
            let reader = arrow::csv::ReaderBuilder::new(Arc::clone(&read_schema))
                .with_header(header)
                .with_delimiter(delimiter)
                .build(f)
                .map_err(|e| BoltError::Other(format!("COPY: reading CSV: {e}")))?;
            let batches: Vec<RecordBatch> = reader
                .collect::<Result<_, _>>()
                .map_err(|e| BoltError::Other(format!("COPY: parsing CSV: {e}")))?;
            concat_or_empty(read_schema, &batches)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ddl_types_cover_common_spellings() {
        assert_eq!(ddl_type("INT").unwrap(), DataType::Int32);
        assert_eq!(ddl_type("integer").unwrap(), DataType::Int32);
        assert_eq!(ddl_type("BIGINT").unwrap(), DataType::Int64);
        assert_eq!(ddl_type("DOUBLE PRECISION").unwrap(), DataType::Float64);
        assert_eq!(ddl_type("VARCHAR(20)").unwrap(), DataType::Utf8);
        assert_eq!(ddl_type("CHARACTER VARYING(5)").unwrap(), DataType::Utf8);
        assert_eq!(
            ddl_type("DECIMAL(10,2)").unwrap(),
            DataType::Decimal128(10, 2)
        );
        assert_eq!(ddl_type("NUMERIC").unwrap(), DataType::Decimal128(38, 10));
        assert_eq!(
            ddl_type("TIMESTAMP(3)").unwrap(),
            DataType::Timestamp(TimeUnit::Millisecond, None)
        );
        assert!(matches!(
            ddl_type("TIMESTAMP WITH TIME ZONE").unwrap(),
            DataType::Timestamp(TimeUnit::Microsecond, Some(_))
        ));
        assert_eq!(ddl_type("date").unwrap(), DataType::Date32);
        assert!(ddl_type("BLOB").is_err());
    }

    fn batch(cols: Vec<(&str, ArrayRef)>) -> RecordBatch {
        let fields: Vec<ArrowField> = cols
            .iter()
            .map(|(n, a)| ArrowField::new(*n, a.data_type().clone(), true))
            .collect();
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(fields)),
            cols.into_iter().map(|(_, a)| a).collect(),
        )
        .unwrap()
    }

    #[test]
    fn align_casts_reorders_and_fills_nulls() {
        let target: SchemaRef = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", ArrowDataType::Int64, false),
            ArrowField::new("b", ArrowDataType::Utf8, true),
            ArrowField::new("c", ArrowDataType::Float64, true),
        ]));
        let src = batch(vec![
            (
                "x",
                Arc::new(arrow_array::Int32Array::from(vec![1, 2])) as ArrayRef,
            ),
            (
                "y",
                Arc::new(arrow_array::Int32Array::from(vec![7, 8])) as ArrayRef,
            ),
        ]);
        let out = align_batch(&src, &target, &["c".into(), "a".into()]).unwrap();
        assert_eq!(out.column(0).data_type(), &ArrowDataType::Int64);
        assert_eq!(out.column(1).null_count(), 2);
        assert_eq!(out.column(2).data_type(), &ArrowDataType::Float64);
        // Missing NOT NULL column → error.
        assert!(align_batch(&src, &target, &["b".into(), "c".into()]).is_err());
        // Arity mismatch → error.
        assert!(align_batch(&src, &target, &[]).is_err());
        // Unknown column → error.
        assert!(align_batch(&src, &target, &["zz".into(), "a".into()]).is_err());
    }
}
