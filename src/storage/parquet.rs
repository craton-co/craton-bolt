// SPDX-License-Identifier: Apache-2.0
//! Parquet-backed tables and the Parquet writer.
//!
//! A [`ParquetTable`] is one or more Parquet files with a common schema. Each
//! **row group** is a scan chunk: its footer statistics (min / max / null
//! count) are the zone map [`crate::storage::scan`] prunes with, and only the
//! columns a query reads are decoded. Footers are read once, at open.
//!
//! A directory laid out Hive-style (`year=2024/month=7/part-0.parquet`)
//! yields **partition columns** after the file columns, typed `Int64` when
//! every value is an integer and `Utf8` otherwise
//! (`__HIVE_DEFAULT_PARTITION__` is NULL). A file's partition values are
//! exact statistics of each of its row groups (min = max = value), so a
//! filter on a partition column prunes whole partitions without opening
//! their files.

use std::collections::BTreeSet;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::compute::cast;
use arrow_array::{Array, RecordBatch};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef,
};
use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::{EnabledStatistics, WriterProperties};

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::Literal;
use crate::plan::logical_plan::Schema;
use crate::storage::stats::{array_value, ColumnStatistics, TableStatistics};
use crate::storage::table::{widen_to_schema, BatchIter, ScanSelection, StorageTable};

/// Rows per decoded batch.
const READ_BATCH_ROWS: usize = 64 * 1024;

fn perr(ctx: &str, e: impl std::fmt::Display) -> BoltError {
    BoltError::Other(format!("{ctx}: {e}"))
}

/// One file of a [`ParquetTable`].
#[derive(Debug, Clone)]
struct ParquetFile {
    path: PathBuf,
    metadata: ArrowReaderMetadata,
    /// Table column ordinal → root column ordinal in this file (file columns
    /// only; partition columns follow them in the table schema).
    column_map: Vec<usize>,
    /// Value of each partition column for this file.
    partition_values: Vec<Option<Literal>>,
}

/// A table backed by Parquet files. See the [module docs](self).
#[derive(Debug, Clone)]
pub struct ParquetTable {
    schema: Schema,
    arrow_schema: SchemaRef,
    files: Vec<ParquetFile>,
    /// Chunk ordinal → (file, row group).
    chunks: Vec<(usize, usize)>,
    stats: Vec<TableStatistics>,
    skipped_columns: Vec<String>,
    /// Trailing schema columns that come from Hive-style directory names.
    partition_columns: usize,
}

/// Hive's marker for a NULL partition value.
const HIVE_NULL: &str = "__HIVE_DEFAULT_PARTITION__";

/// Decode Hive's `%XX` escapes in a partition value.
fn hive_unescape(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A Parquet file found under a table directory, with the `key=value`
/// directory segments of its relative path.
type FoundFile = (PathBuf, Vec<(String, String)>);

/// Every `*.parquet` file under `dir` (recursively, sorted), skipping hidden
/// and `_`-prefixed entries.
fn walk_parquet(dir: &Path) -> BoltResult<Vec<FoundFile>> {
    fn rec(dir: &Path, segs: &[(String, String)], out: &mut Vec<FoundFile>) -> BoltResult<()> {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| perr(&format!("reading directory {}", dir.display()), e))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        entries.sort();
        for p in entries {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with('.') || name.starts_with('_') {
                continue;
            }
            if p.is_dir() {
                let mut next = segs.to_vec();
                if let Some((k, v)) = name.split_once('=') {
                    next.push((k.to_string(), hive_unescape(v)));
                }
                rec(&p, &next, out)?;
            } else if p
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("parquet"))
            {
                out.push((p, segs.to_vec()));
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    rec(dir, &[], &mut out)?;
    Ok(out)
}

/// Map a file column type onto the engine's type, if it has one.
fn normalize_type(dt: &ArrowDataType) -> Option<ArrowDataType> {
    match dt {
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => {
            Some(ArrowDataType::Utf8)
        }
        ArrowDataType::Dictionary(_, v)
            if matches!(
                v.as_ref(),
                ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View
            ) =>
        {
            Some(ArrowDataType::Utf8)
        }
        ArrowDataType::Boolean
        | ArrowDataType::Int32
        | ArrowDataType::Int64
        | ArrowDataType::Float32
        | ArrowDataType::Float64
        | ArrowDataType::Date32
        | ArrowDataType::Decimal128(_, _)
        | ArrowDataType::Timestamp(_, _) => Some(dt.clone()),
        // Narrow integers widen losslessly.
        ArrowDataType::Int8
        | ArrowDataType::Int16
        | ArrowDataType::UInt8
        | ArrowDataType::UInt16 => Some(ArrowDataType::Int32),
        ArrowDataType::UInt32 => Some(ArrowDataType::Int64),
        _ => None,
    }
}

impl ParquetTable {
    /// Open a Parquet file, or every `*.parquet` file under a directory
    /// (recursively, sorted; hidden and `_`-prefixed entries skipped). A
    /// Hive-style `key=value` directory layout adds partition columns (see
    /// the [module docs](self)).
    pub fn open(path: impl AsRef<Path>) -> BoltResult<Self> {
        let path = path.as_ref();
        if !path.is_dir() {
            return Self::from_files(&[path.to_path_buf()]);
        }
        let found = walk_parquet(path)?;
        if found.is_empty() {
            return Err(BoltError::Plan(format!(
                "no .parquet files under directory {}",
                path.display()
            )));
        }
        let keys: Vec<String> = found[0].1.iter().map(|(k, _)| k.clone()).collect();
        for (p, segs) in &found {
            if segs.len() != keys.len() || segs.iter().zip(&keys).any(|((k, _), want)| k != want) {
                return Err(BoltError::Plan(format!(
                    "inconsistent partition layout: {} does not have partition keys {keys:?}",
                    p.display()
                )));
            }
        }
        let files: Vec<PathBuf> = found.iter().map(|(p, _)| p.clone()).collect();
        let values: Vec<Vec<Option<String>>> = found
            .into_iter()
            .map(|(_, segs)| {
                segs.into_iter()
                    .map(|(_, v)| (v != HIVE_NULL).then_some(v))
                    .collect()
            })
            .collect();
        Self::open_partitioned(&files, &keys, &values)
    }

    /// Open a table over `files`, which must share column names and
    /// compatible types. The table schema comes from the first file; columns
    /// whose type the engine cannot represent (nested, binary, unsigned
    /// 64-bit, …) are left out and listed by [`ParquetTable::skipped_columns`].
    pub fn from_files(files: &[PathBuf]) -> BoltResult<Self> {
        Self::open_partitioned(files, &[], &vec![Vec::new(); files.len()])
    }

    fn open_partitioned(
        files: &[PathBuf],
        keys: &[String],
        values: &[Vec<Option<String>>],
    ) -> BoltResult<Self> {
        if files.is_empty() {
            return Err(BoltError::Plan(
                "a Parquet table needs at least one file".into(),
            ));
        }
        let mut metas = Vec::with_capacity(files.len());
        for p in files {
            let f = File::open(p).map_err(|e| perr(&format!("opening {}", p.display()), e))?;
            let meta = ArrowReaderMetadata::load(&f, ArrowReaderOptions::new())
                .map_err(|e| perr(&format!("reading Parquet footer of {}", p.display()), e))?;
            metas.push((p.clone(), meta));
        }
        // Table schema from the first file.
        let first = metas[0].1.schema().clone();
        let mut fields = Vec::new();
        let mut skipped_columns = Vec::new();
        for f in first.fields() {
            match normalize_type(f.data_type()) {
                Some(dt) => fields.push(ArrowField::new(f.name(), dt, f.is_nullable())),
                None => skipped_columns.push(f.name().clone()),
            }
        }
        if fields.is_empty() {
            return Err(BoltError::Plan(
                "no Parquet column has a type the engine supports".into(),
            ));
        }
        for name in &skipped_columns {
            log::warn!("Parquet table: column '{name}' has an unsupported type and is skipped");
        }
        // Partition columns: Int64 when every value is an integer.
        let mut part_values: Vec<Vec<Option<Literal>>> = vec![Vec::new(); files.len()];
        for (ki, key) in keys.iter().enumerate() {
            if fields.iter().any(|f| f.name() == key) {
                return Err(BoltError::Plan(format!(
                    "partition column '{key}' also exists in the Parquet files"
                )));
            }
            let raw: Vec<&Option<String>> = values.iter().map(|v| &v[ki]).collect();
            let ints = raw
                .iter()
                .all(|v| v.as_ref().is_none_or(|s| s.parse::<i64>().is_ok()));
            let nullable = raw.iter().any(|v| v.is_none());
            let dt = if ints {
                ArrowDataType::Int64
            } else {
                ArrowDataType::Utf8
            };
            fields.push(ArrowField::new(key, dt, nullable));
            for (fi, v) in raw.into_iter().enumerate() {
                part_values[fi].push(v.as_ref().map(|s| match s.parse::<i64>() {
                    Ok(n) if ints => Literal::Int64(n),
                    _ => Literal::Utf8(s.clone()),
                }));
            }
        }
        let mut t = Self::build(Arc::new(ArrowSchema::new(fields)), metas, part_values)?;
        t.skipped_columns = skipped_columns;
        Ok(t)
    }

    /// Open a table with a known `schema` (engine types) over `files`, which
    /// may be empty. Every file must hold every schema column, with a type
    /// that normalises to the declared one.
    pub fn with_schema(schema: SchemaRef, files: &[PathBuf]) -> BoltResult<Self> {
        let mut metas = Vec::with_capacity(files.len());
        for p in files {
            let f = File::open(p).map_err(|e| perr(&format!("opening {}", p.display()), e))?;
            let meta = ArrowReaderMetadata::load(&f, ArrowReaderOptions::new())
                .map_err(|e| perr(&format!("reading Parquet footer of {}", p.display()), e))?;
            metas.push((p.clone(), meta));
        }
        let n = metas.len();
        Self::build(schema, metas, vec![Vec::new(); n])
    }

    /// `part_values[file]` holds that file's values of the trailing
    /// partition columns of `arrow_schema`.
    fn build(
        arrow_schema: SchemaRef,
        metas: Vec<(PathBuf, ArrowReaderMetadata)>,
        part_values: Vec<Vec<Option<Literal>>>,
    ) -> BoltResult<Self> {
        let schema = crate::exec::engine_support::arrow_schema_to_plan_schema(&arrow_schema)?;
        let n_part = part_values.first().map_or(0, Vec::len);
        let n_file = arrow_schema.fields().len() - n_part;
        let mut out_files = Vec::with_capacity(metas.len());
        let mut chunks = Vec::new();
        let mut stats = Vec::new();
        for (fi, ((path, meta), partition_values)) in metas.into_iter().zip(part_values).enumerate()
        {
            let file_schema = meta.schema().clone();
            let mut column_map = Vec::with_capacity(n_file);
            for f in arrow_schema.fields().iter().take(n_file) {
                let idx = file_schema.index_of(f.name()).map_err(|_| {
                    BoltError::Plan(format!(
                        "Parquet file {} has no column '{}'",
                        path.display(),
                        f.name()
                    ))
                })?;
                let file_dt = file_schema.field(idx).data_type();
                if normalize_type(file_dt).as_ref() != Some(f.data_type()) {
                    return Err(BoltError::Plan(format!(
                        "Parquet file {}: column '{}' is {file_dt:?}, expected {:?}",
                        path.display(),
                        f.name(),
                        f.data_type()
                    )));
                }
                column_map.push(idx);
            }
            let file_stats = row_group_statistics(&meta, &arrow_schema, &column_map)
                .map_err(|e| perr(&format!("Parquet statistics of {}", path.display()), e))?;
            for (rg, mut s) in file_stats.into_iter().enumerate() {
                // A partition value is exact for every row of the file.
                for v in &partition_values {
                    s.columns.push(ColumnStatistics {
                        null_count: Some(if v.is_some() { 0 } else { s.row_count }),
                        min: v.clone(),
                        max: v.clone(),
                        distinct_estimate: Some(u64::from(v.is_some())),
                    });
                }
                chunks.push((fi, rg));
                stats.push(s);
            }
            out_files.push(ParquetFile {
                path,
                metadata: meta,
                column_map,
                partition_values,
            });
        }
        Ok(ParquetTable {
            schema,
            arrow_schema,
            files: out_files,
            chunks,
            stats,
            skipped_columns: Vec::new(),
            partition_columns: n_part,
        })
    }

    /// Names of the Hive partition columns (the schema's trailing columns).
    pub fn partition_columns(&self) -> Vec<&str> {
        let n = self.arrow_schema.fields().len();
        self.arrow_schema.fields()[n - self.partition_columns..]
            .iter()
            .map(|f| f.name().as_str())
            .collect()
    }

    /// Columns of the files that the table leaves out (unsupported types).
    pub fn skipped_columns(&self) -> &[String] {
        &self.skipped_columns
    }

    /// The table's files, in chunk order.
    pub fn files(&self) -> Vec<&Path> {
        self.files.iter().map(|f| f.path.as_path()).collect()
    }

    /// Arrow form of [`StorageTable::schema`].
    pub fn arrow_schema(&self) -> &SchemaRef {
        &self.arrow_schema
    }
}

/// Per-row-group zone maps of one file, in table column order.
fn row_group_statistics(
    meta: &ArrowReaderMetadata,
    table_schema: &ArrowSchema,
    column_map: &[usize],
) -> parquet::errors::Result<Vec<TableStatistics>> {
    let pq = meta.metadata();
    let file_schema = meta.schema();
    let n_rg = pq.num_row_groups();
    let mut out: Vec<TableStatistics> = pq
        .row_groups()
        .iter()
        .map(|rg| TableStatistics {
            row_count: rg.num_rows().max(0) as u64,
            byte_size: rg.compressed_size().max(0) as u64,
            columns: Vec::with_capacity(column_map.len()),
        })
        .collect();
    for (ti, &fi) in column_map.iter().enumerate() {
        let name = file_schema.field(fi).name();
        let _ = table_schema.field(ti);
        let conv =
            StatisticsConverter::try_new(name, file_schema, pq.file_metadata().schema_descr());
        let (mins, maxes, nulls) = match conv {
            Ok(c) => (
                c.row_group_mins(pq.row_groups().iter()).ok(),
                c.row_group_maxes(pq.row_groups().iter()).ok(),
                c.row_group_null_counts(pq.row_groups().iter()).ok(),
            ),
            Err(_) => (None, None, None),
        };
        for (rg, s) in out.iter_mut().enumerate().take(n_rg) {
            let bound = |a: &Option<arrow_array::ArrayRef>| {
                a.as_ref()
                    .filter(|a| rg < a.len())
                    .and_then(|a| array_value(a.as_ref(), rg))
            };
            let null_count = nulls
                .as_ref()
                .filter(|n| rg < n.len() && n.is_valid(rg))
                .map(|n| n.value(rg));
            s.columns.push(ColumnStatistics {
                null_count,
                min: bound(&mins),
                max: bound(&maxes),
                distinct_estimate: None,
            });
        }
    }
    Ok(out)
}

impl StorageTable for ParquetTable {
    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    fn chunk_statistics(&self, chunk: usize) -> &TableStatistics {
        &self.stats[chunk]
    }

    fn read(&self, selection: &ScanSelection) -> BoltResult<BatchIter> {
        let read_columns: Vec<usize> = match &selection.columns {
            Some(c) => c
                .iter()
                .copied()
                .filter(|&c| c < self.arrow_schema.fields().len())
                .collect(),
            None => (0..self.arrow_schema.fields().len()).collect(),
        };
        // Group the selected row groups by file, keeping chunk order.
        let mut per_file: Vec<(usize, Vec<usize>)> = Vec::new();
        for &c in &selection.chunks {
            let (f, rg) = self.chunks[c];
            match per_file.last_mut() {
                Some((lf, rgs)) if *lf == f => rgs.push(rg),
                _ => per_file.push((f, vec![rg])),
            }
        }
        let files = self.files.clone();
        let schema = Arc::clone(&self.arrow_schema);
        let n_file = schema.fields().len() - self.partition_columns;
        let iter = per_file
            .into_iter()
            .flat_map(move |(fi, rgs)| -> BatchIter {
                match open_reader(&files[fi], &rgs, &read_columns, &schema, n_file) {
                    Ok(it) => it,
                    Err(e) => Box::new(std::iter::once(Err(e))),
                }
            });
        Ok(Box::new(iter))
    }
}

/// A column holding `v` in each of `n` rows.
fn constant_column(dt: &ArrowDataType, v: &Option<Literal>, n: usize) -> arrow_array::ArrayRef {
    match v {
        Some(Literal::Int64(x)) => Arc::new(arrow_array::Int64Array::from(vec![*x; n])),
        Some(Literal::Utf8(s)) => Arc::new(arrow_array::StringArray::from(vec![s.as_str(); n])),
        _ => arrow_array::new_null_array(dt, n),
    }
}

fn open_reader(
    file: &ParquetFile,
    row_groups: &[usize],
    read_columns: &[usize],
    schema: &SchemaRef,
    n_file: usize,
) -> BoltResult<BatchIter> {
    let f =
        File::open(&file.path).map_err(|e| perr(&format!("opening {}", file.path.display()), e))?;
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(f, file.metadata.clone());
    let mut roots: BTreeSet<usize> = read_columns
        .iter()
        .filter(|&&c| c < n_file)
        .map(|&c| file.column_map[c])
        .collect();
    if roots.is_empty() {
        // Only partition columns were asked for; decode one file column for
        // the row counts.
        if let Some(&first) = file.column_map.first() {
            roots.insert(first);
        }
    }
    let mask = ProjectionMask::roots(builder.parquet_schema(), roots.iter().copied());
    let reader = builder
        .with_row_groups(row_groups.to_vec())
        .with_projection(mask)
        .with_batch_size(READ_BATCH_ROWS)
        .build()
        .map_err(|e| perr(&format!("reading {}", file.path.display()), e))?;
    let schema = Arc::clone(schema);
    let read_columns = read_columns.to_vec();
    let path = file.path.clone();
    let partition_values = file.partition_values.clone();
    Ok(Box::new(reader.map(move |batch| {
        let batch = batch.map_err(|e| perr(&format!("decoding {}", path.display()), e))?;
        let mut cols = Vec::with_capacity(read_columns.len());
        let mut fields = Vec::with_capacity(read_columns.len());
        for &c in &read_columns {
            let field = schema.field(c);
            if c >= n_file {
                let n = batch.num_rows();
                cols.push(constant_column(
                    field.data_type(),
                    &partition_values[c - n_file],
                    n,
                ));
                fields.push(field.clone());
                continue;
            }
            let src = batch.column_by_name(field.name()).ok_or_else(|| {
                BoltError::Other(format!("Parquet reader lost column '{}'", field.name()))
            })?;
            let col = if src.data_type() == field.data_type() {
                Arc::clone(src)
            } else {
                cast(src, field.data_type())
                    .map_err(|e| perr(&format!("casting '{}'", field.name()), e))?
            };
            cols.push(col);
            fields.push(field.clone());
        }
        let projected = RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), cols)
            .map_err(|e| perr("assembling Parquet batch", e))?;
        widen_to_schema(&projected, &schema, &read_columns)
    })))
}

/// Options for [`write_parquet`].
#[derive(Debug, Clone)]
pub struct ParquetWriteOptions {
    /// Maximum rows per row group — the pruning granularity. Default 128 Ki.
    pub row_group_rows: usize,
    /// zstd level (1–22). Default 3.
    pub zstd_level: i32,
}

impl Default for ParquetWriteOptions {
    fn default() -> Self {
        ParquetWriteOptions {
            row_group_rows: 128 * 1024,
            zstd_level: 3,
        }
    }
}

/// Write `batches` (one schema) to a Parquet file at `path` with chunk
/// statistics, atomically: the data goes to a temporary sibling first and is
/// renamed into place only once complete. Returns the row count.
pub fn write_parquet(
    path: &Path,
    batches: &[RecordBatch],
    options: &ParquetWriteOptions,
) -> BoltResult<u64> {
    let first = batches
        .first()
        .ok_or_else(|| BoltError::Plan("write_parquet: no batches".into()))?;
    let schema = first.schema();
    let level = ZstdLevel::try_new(options.zstd_level).map_err(|e| perr("zstd level", e))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(level))
        .set_max_row_group_size(options.row_group_rows.max(1))
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .build();
    let tmp = path.with_extension("parquet.tmp");
    let result = (|| -> BoltResult<u64> {
        let f = File::create(&tmp).map_err(|e| perr(&format!("creating {}", tmp.display()), e))?;
        let mut w = ArrowWriter::try_new(f, Arc::clone(&schema), Some(props))
            .map_err(|e| perr("starting Parquet writer", e))?;
        let mut rows = 0u64;
        for b in batches {
            if b.schema() != schema {
                return Err(BoltError::Plan(
                    "write_parquet: batches must share one schema".into(),
                ));
            }
            w.write(b).map_err(|e| perr("writing Parquet", e))?;
            rows += b.num_rows() as u64;
        }
        let f = w
            .into_inner()
            .map_err(|e| perr("finishing Parquet file", e))?;
        f.sync_all().map_err(|e| perr("syncing Parquet file", e))?;
        Ok(rows)
    })();
    match result {
        Ok(rows) => {
            std::fs::rename(&tmp, path)
                .map_err(|e| perr(&format!("renaming into {}", path.display()), e))?;
            Ok(rows)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::Literal;
    use crate::storage::scan::{PruneOp, PrunePredicate, ScanRequest};
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::{BinaryArray, Int16Array, Int64Array, LargeStringArray};

    pub(crate) fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "bolt-pq-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn batch(lo: i64, n: usize) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int64, false),
            ArrowField::new("s", ArrowDataType::LargeUtf8, true),
            ArrowField::new("small", ArrowDataType::Int16, false),
            ArrowField::new("blob", ArrowDataType::Binary, false),
        ]));
        let k: Vec<i64> = (lo..lo + n as i64).collect();
        let s: Vec<Option<String>> = k
            .iter()
            .map(|v| {
                if v % 10 == 0 {
                    None
                } else {
                    Some(format!("v{v}"))
                }
            })
            .collect();
        let small: Vec<i16> = k.iter().map(|v| (v % 100) as i16).collect();
        let blob: Vec<&[u8]> = k.iter().map(|_| b"x".as_slice()).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(k)),
                Arc::new(LargeStringArray::from(s)),
                Arc::new(Int16Array::from(small)),
                Arc::new(BinaryArray::from(blob)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn write_open_prune_and_project() {
        let dir = temp_dir("rt");
        let opts = ParquetWriteOptions {
            row_group_rows: 1000,
            ..Default::default()
        };
        write_parquet(&dir.join("a.parquet"), &[batch(0, 3000)], &opts).unwrap();
        write_parquet(&dir.join("b.parquet"), &[batch(3000, 2000)], &opts).unwrap();
        let t = ParquetTable::open(&dir).unwrap();
        assert_eq!(t.skipped_columns(), ["blob".to_string()]);
        assert_eq!(t.schema().fields.len(), 3);
        assert_eq!(t.chunk_count(), 5);
        let s = t.statistics();
        assert_eq!(s.row_count, 5000);
        assert_eq!(s.columns[0].min, Some(Literal::Int64(0)));
        assert_eq!(s.columns[0].max, Some(Literal::Int64(4999)));
        assert_eq!(s.columns[1].null_count, Some(500));

        let req = ScanRequest {
            columns: Some([0].into_iter().collect()),
            filters: vec![vec![PrunePredicate::Compare {
                column: 0,
                op: PruneOp::Gt,
                value: Literal::Int64(3500),
            }]],
        };
        let sel = t.plan_scan(&req);
        assert_eq!(sel.chunks, vec![3, 4]);
        let batches: Vec<RecordBatch> = t.read(&sel).unwrap().map(Result::unwrap).collect();
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(rows, 2000);
        let b = &batches[0];
        assert_eq!(b.schema(), *t.arrow_schema());
        assert_eq!(b.column(0).as_primitive::<Int64Type>().value(0), 3000);
        // Unread column: placeholder, not decoded data.
        assert_eq!(b.column(1).as_string::<i32>().value(1), "");
        // Full read: types normalised (LargeUtf8 → Utf8, Int16 → Int32).
        let all = t.plan_scan(&ScanRequest::full());
        let b = t.read(&all).unwrap().next().unwrap().unwrap();
        assert_eq!(b.column(1).as_string::<i32>().value(1), "v1");
        assert_eq!(b.column(2).data_type(), &ArrowDataType::Int32);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn hive_partitions_become_prunable_columns() {
        let dir = temp_dir("hive");
        for (y, m) in [
            ("2024", "1"),
            ("2024", "2"),
            ("2025", "1"),
            ("__HIVE_DEFAULT_PARTITION__", "1"),
        ] {
            let d = dir.join(format!("year={y}")).join(format!("month={m}"));
            std::fs::create_dir_all(&d).unwrap();
            write_parquet(
                &d.join("part-0.parquet"),
                &[batch(0, 10)],
                &Default::default(),
            )
            .unwrap();
        }
        std::fs::write(dir.join("_SUCCESS"), b"").unwrap();
        let t = ParquetTable::open(&dir).unwrap();
        assert_eq!(t.partition_columns(), vec!["year", "month"]);
        assert_eq!(t.chunk_count(), 4);
        let year = t.schema().index_of("year").unwrap();
        let req = ScanRequest {
            columns: Some([0, year].into_iter().collect()),
            filters: vec![vec![PrunePredicate::Compare {
                column: year,
                op: PruneOp::Eq,
                value: Literal::Int64(2024),
            }]],
        };
        let sel = t.plan_scan(&req);
        assert_eq!(sel.chunks.len(), 2, "only the year=2024 partitions");
        let batches: Vec<RecordBatch> = t.read(&sel).unwrap().map(Result::unwrap).collect();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 20);
        let y = batches[0].column(year).as_primitive::<Int64Type>();
        assert!(y.iter().all(|v| v == Some(2024)));
        // The NULL partition reads as NULL.
        let null_sel = t.plan_scan(&ScanRequest {
            columns: None,
            filters: vec![vec![PrunePredicate::IsNull { column: year }]],
        });
        assert_eq!(null_sel.chunks.len(), 1);
        let b = t.read(&null_sel).unwrap().next().unwrap().unwrap();
        assert_eq!(b.column(year).null_count(), 10);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn mismatched_files_are_rejected() {
        let dir = temp_dir("mm");
        write_parquet(&dir.join("a.parquet"), &[batch(0, 10)], &Default::default()).unwrap();
        let other = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "k",
                ArrowDataType::Utf8,
                false,
            )])),
            vec![Arc::new(arrow_array::StringArray::from(vec!["x"]))],
        )
        .unwrap();
        write_parquet(&dir.join("b.parquet"), &[other], &Default::default()).unwrap();
        let err = ParquetTable::open(&dir).unwrap_err().to_string();
        assert!(err.contains("b.parquet"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn failed_write_leaves_no_file() {
        let dir = temp_dir("fw");
        let a = batch(0, 5);
        let b = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "z",
                ArrowDataType::Int64,
                false,
            )])),
            vec![Arc::new(Int64Array::from(vec![1]))],
        )
        .unwrap();
        let p = dir.join("x.parquet");
        assert!(write_parquet(&p, &[a, b], &Default::default()).is_err());
        assert!(!p.exists());
        assert!(!p.with_extension("parquet.tmp").exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
