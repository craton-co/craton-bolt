// SPDX-License-Identifier: Apache-2.0
//! Persistent, snapshot-versioned table catalog.
//!
//! A catalog is a directory:
//!
//! ```text
//! <root>/
//!   _bolt/CURRENT                    name of the current manifest
//!   _bolt/LOCK                       held by the one writer process
//!   _bolt/manifests/manifest-<N>.json
//!   data/<table-id>/part-<N>-<k>.parquet
//! ```
//!
//! Data files are immutable Parquet. Every change — create, append, drop,
//! truncate, compaction, statistics — writes its new data files first and
//! then commits a new **manifest** (the complete table list with each
//! table's files and per-file statistics) by writing `manifest-<N>.json` and
//! atomically renaming a new `CURRENT` into place. A crash at any point
//! leaves the previous snapshot intact; files written by an uncommitted
//! change are unreferenced and removed by [`Catalog::vacuum`]. Each manifest
//! number is a **snapshot**: [`Catalog::open_snapshot`] opens an older one
//! read-only (time travel) until vacuum discards it.
//!
//! One process writes a catalog at a time (`_bolt/LOCK`, created
//! exclusively and removed when the writer closes).

use std::collections::BTreeMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use arrow_array::RecordBatch;
use serde::{Deserialize, Serialize};

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{DataType, Field, Schema, TimeUnit};
use crate::storage::parquet::{write_parquet, ParquetTable, ParquetWriteOptions};
use crate::storage::stats::TableStatistics;
use crate::storage::table::StorageTable;

const FORMAT_VERSION: u32 = 1;

fn io_err(ctx: impl std::fmt::Display, e: impl std::fmt::Display) -> BoltError {
    BoltError::Other(format!("catalog: {ctx}: {e}"))
}

/// One column of a catalog table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnMeta {
    /// Column name.
    pub name: String,
    /// Type, in the catalog's text form (see [`dtype_to_text`]).
    #[serde(rename = "type")]
    pub dtype: String,
    /// Whether NULLs are admitted.
    pub nullable: bool,
}

/// One immutable data file of a table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataFile {
    /// Path relative to the catalog root.
    pub path: String,
    /// Rows in the file.
    pub rows: u64,
    /// Size on disk.
    pub bytes: u64,
    /// Statistics of the file (merged over its row groups).
    pub stats: TableStatistics,
}

/// How a table's data is held.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TableKind {
    /// Files owned by the catalog under `data/`.
    Managed,
    /// Parquet data at an external location the catalog only references.
    External {
        /// File or directory path (absolute, or relative to the root).
        location: String,
    },
}

/// Catalog entry for one table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableMeta {
    /// Stable id (names the data directory).
    pub id: u64,
    /// Storage kind.
    pub kind: TableKind,
    /// Columns.
    pub columns: Vec<ColumnMeta>,
    /// Data files (managed tables).
    pub files: Vec<DataFile>,
    /// Bumped on every change to the table's data.
    pub table_version: u64,
    /// Statistics recorded by `ANALYZE` (exact distinct estimates).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analyzed: Option<TableStatistics>,
}

impl TableMeta {
    /// Plan schema of the table.
    pub fn schema(&self) -> BoltResult<Schema> {
        let fields = self
            .columns
            .iter()
            .map(|c| {
                Ok(Field::new(
                    c.name.clone(),
                    dtype_from_text(&c.dtype)?,
                    c.nullable,
                ))
            })
            .collect::<BoltResult<Vec<_>>>()?;
        Ok(Schema::new(fields))
    }

    /// Total rows over the data files.
    pub fn row_count(&self) -> u64 {
        self.files.iter().map(|f| f.rows).sum()
    }
}

/// Refresh policy of a materialized view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefreshPolicy {
    /// Only `REFRESH MATERIALIZED VIEW` refreshes it.
    Manual,
    /// Refreshed automatically after a statement changes a base table.
    OnCommit,
}

/// Catalog entry for a materialized view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewMeta {
    /// Defining query.
    pub sql: String,
    /// Base tables the query reads.
    pub dependencies: Vec<String>,
    /// `table_version` of each dependency at the last refresh.
    pub base_versions: BTreeMap<String, u64>,
    /// Refresh policy.
    pub refresh: RefreshPolicy,
    /// Managed table holding the view's rows.
    pub backing_table: String,
}

/// A complete catalog snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Manifest format version.
    pub format: u32,
    /// Snapshot number (monotone).
    pub version: u64,
    /// Next table id to assign.
    pub next_table_id: u64,
    /// Tables by name.
    pub tables: BTreeMap<String, TableMeta>,
    /// Materialized views by name.
    #[serde(default)]
    pub views: BTreeMap<String, ViewMeta>,
}

impl Manifest {
    fn empty() -> Self {
        Manifest {
            format: FORMAT_VERSION,
            version: 0,
            next_table_id: 1,
            tables: BTreeMap::new(),
            views: BTreeMap::new(),
        }
    }
}

/// Exclusive writer lock; removes `_bolt/LOCK` on drop.
#[derive(Debug)]
struct WriterLock(PathBuf);

impl Drop for WriterLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// What [`Catalog::compact`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CompactReport {
    /// Files before.
    pub files_before: usize,
    /// Files after.
    pub files_after: usize,
}

/// What [`Catalog::vacuum`] removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct VacuumReport {
    /// Old manifests deleted.
    pub manifests_removed: usize,
    /// Unreferenced data files deleted.
    pub files_removed: usize,
}

/// A persistent catalog. See the [module docs](self).
#[derive(Debug)]
pub struct Catalog {
    root: PathBuf,
    manifest: Manifest,
    lock: Option<WriterLock>,
    write_options: ParquetWriteOptions,
}

impl Catalog {
    /// Open (creating if needed) the catalog at `root` for writing.
    ///
    /// # Errors
    /// Another process holds the writer lock, or the current manifest is
    /// unreadable.
    pub fn open(root: impl AsRef<Path>) -> BoltResult<Self> {
        let root = root.as_ref().to_path_buf();
        let meta = root.join("_bolt");
        fs::create_dir_all(meta.join("manifests")).map_err(|e| io_err(meta.display(), e))?;
        fs::create_dir_all(root.join("data")).map_err(|e| io_err("creating data/", e))?;
        let lock_path = meta.join("LOCK");
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    BoltError::Plan(format!(
                        "catalog {} is locked by another writer (remove {} if no \
                         process is using it)",
                        root.display(),
                        lock_path.display()
                    ))
                } else {
                    io_err(lock_path.display(), e)
                }
            })?;
        let _ = writeln!(f, "{}", std::process::id());
        let lock = WriterLock(lock_path);
        let manifest = match read_current(&root)? {
            Some(n) => read_manifest(&root, n)?,
            None => Manifest::empty(),
        };
        Ok(Catalog {
            root,
            manifest,
            lock: Some(lock),
            write_options: ParquetWriteOptions::default(),
        })
    }

    /// Open snapshot `version` of the catalog at `root`, read-only.
    pub fn open_snapshot(root: impl AsRef<Path>, version: u64) -> BoltResult<Self> {
        let root = root.as_ref().to_path_buf();
        let manifest = if version == 0 {
            Manifest::empty()
        } else {
            read_manifest(&root, version)?
        };
        Ok(Catalog {
            root,
            manifest,
            lock: None,
            write_options: ParquetWriteOptions::default(),
        })
    }

    /// Catalog directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Current snapshot number (0 for a new, empty catalog).
    pub fn version(&self) -> u64 {
        self.manifest.version
    }

    /// `true` for a snapshot opened with [`Catalog::open_snapshot`].
    pub fn is_read_only(&self) -> bool {
        self.lock.is_none()
    }

    /// The current manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Snapshot numbers still on disk, ascending.
    pub fn snapshots(&self) -> BoltResult<Vec<u64>> {
        let dir = self.root.join("_bolt").join("manifests");
        let mut out: Vec<u64> = fs::read_dir(&dir)
            .map_err(|e| io_err(dir.display(), e))?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                name.strip_prefix("manifest-")?
                    .strip_suffix(".json")?
                    .parse()
                    .ok()
            })
            .collect();
        out.sort_unstable();
        Ok(out)
    }

    /// Set the Parquet layout used for new data files.
    pub fn set_write_options(&mut self, options: ParquetWriteOptions) {
        self.write_options = options;
    }

    /// The entry for table `name`.
    pub fn table(&self, name: &str) -> Option<&TableMeta> {
        self.manifest.tables.get(name)
    }

    /// Open table `name` as a [`ParquetTable`].
    pub fn open_table(&self, name: &str) -> BoltResult<ParquetTable> {
        let meta = self.require(name)?;
        match &meta.kind {
            TableKind::Managed => {
                let files: Vec<PathBuf> =
                    meta.files.iter().map(|f| self.root.join(&f.path)).collect();
                let schema =
                    crate::exec::schema_convert::plan_schema_to_arrow_schema(&meta.schema()?)?;
                ParquetTable::with_schema(schema, &files)
            }
            TableKind::External { location } => ParquetTable::open(self.resolve(location)),
        }
    }

    fn resolve(&self, location: &str) -> PathBuf {
        let p = Path::new(location);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.root.join(p)
        }
    }

    fn require(&self, name: &str) -> BoltResult<&TableMeta> {
        self.manifest
            .tables
            .get(name)
            .ok_or_else(|| BoltError::Plan(format!("catalog has no table '{name}'")))
    }

    fn writable(&self) -> BoltResult<()> {
        if self.lock.is_none() {
            return Err(BoltError::Plan(format!(
                "catalog snapshot {} is read-only",
                self.manifest.version
            )));
        }
        Ok(())
    }

    /// Create managed table `name` with `schema`, holding `batches` (may be
    /// empty).
    pub fn create_table(
        &mut self,
        name: &str,
        schema: &Schema,
        batches: &[RecordBatch],
    ) -> BoltResult<()> {
        self.writable()?;
        if self.manifest.tables.contains_key(name) || self.manifest.views.contains_key(name) {
            return Err(BoltError::Plan(format!(
                "catalog already has a table named '{name}'"
            )));
        }
        let mut next = self.manifest.clone();
        let id = next.next_table_id;
        next.next_table_id += 1;
        let columns = schema
            .fields
            .iter()
            .map(|f| {
                Ok(ColumnMeta {
                    name: f.name.clone(),
                    dtype: dtype_to_text(f.dtype)?,
                    nullable: f.nullable,
                })
            })
            .collect::<BoltResult<Vec<_>>>()?;
        let version = next.version + 1;
        let files = self.write_files(id, version, batches)?;
        next.tables.insert(
            name.to_string(),
            TableMeta {
                id,
                kind: TableKind::Managed,
                columns,
                files,
                table_version: 1,
                analyzed: None,
            },
        );
        self.commit(next)
    }

    /// Register external Parquet data at `location` as table `name`.
    pub fn create_external_table(&mut self, name: &str, location: &str) -> BoltResult<()> {
        self.writable()?;
        if self.manifest.tables.contains_key(name) || self.manifest.views.contains_key(name) {
            return Err(BoltError::Plan(format!(
                "catalog already has a table named '{name}'"
            )));
        }
        let table = ParquetTable::open(self.resolve(location))?;
        let mut next = self.manifest.clone();
        let id = next.next_table_id;
        next.next_table_id += 1;
        let columns = table
            .schema()
            .fields
            .iter()
            .map(|f| {
                Ok(ColumnMeta {
                    name: f.name.clone(),
                    dtype: dtype_to_text(f.dtype)?,
                    nullable: f.nullable,
                })
            })
            .collect::<BoltResult<Vec<_>>>()?;
        next.tables.insert(
            name.to_string(),
            TableMeta {
                id,
                kind: TableKind::External {
                    location: location.to_string(),
                },
                columns,
                files: Vec::new(),
                table_version: 1,
                analyzed: None,
            },
        );
        self.commit(next)
    }

    /// Append `batches` to managed table `name` as one new data file.
    pub fn append(&mut self, name: &str, batches: &[RecordBatch]) -> BoltResult<()> {
        self.writable()?;
        let meta = self.require(name)?;
        if !matches!(meta.kind, TableKind::Managed) {
            return Err(BoltError::Plan(format!(
                "table '{name}' is external and read-only"
            )));
        }
        let id = meta.id;
        let mut next = self.manifest.clone();
        let version = next.version + 1;
        let files = self.write_files(id, version, batches)?;
        if files.is_empty() {
            return Ok(());
        }
        let t = next.tables.get_mut(name).expect("checked");
        t.files.extend(files);
        t.table_version += 1;
        t.analyzed = None;
        self.commit(next)
    }

    /// Remove every row of managed table `name`.
    pub fn truncate(&mut self, name: &str) -> BoltResult<()> {
        self.writable()?;
        let meta = self.require(name)?;
        if !matches!(meta.kind, TableKind::Managed) {
            return Err(BoltError::Plan(format!(
                "table '{name}' is external and read-only"
            )));
        }
        let mut next = self.manifest.clone();
        let t = next.tables.get_mut(name).expect("checked");
        t.files.clear();
        t.table_version += 1;
        t.analyzed = None;
        self.commit(next)
    }

    /// Drop table `name` (its files are removed by [`Catalog::vacuum`], so
    /// older snapshots stay readable until then).
    pub fn drop_table(&mut self, name: &str) -> BoltResult<()> {
        self.writable()?;
        self.require(name)?;
        let mut next = self.manifest.clone();
        next.tables.remove(name);
        self.commit(next)
    }

    /// Record `ANALYZE` statistics for table `name`.
    pub fn set_statistics(&mut self, name: &str, stats: TableStatistics) -> BoltResult<()> {
        self.writable()?;
        self.require(name)?;
        let mut next = self.manifest.clone();
        next.tables.get_mut(name).expect("checked").analyzed = Some(stats);
        self.commit(next)
    }

    /// Rewrite managed table `name`'s data files into as few files as the
    /// write options allow (sorted reading order preserved). The row data is
    /// unchanged, so `table_version` is not bumped.
    pub fn compact(&mut self, name: &str) -> BoltResult<CompactReport> {
        self.writable()?;
        let meta = self.require(name)?.clone();
        if !matches!(meta.kind, TableKind::Managed) {
            return Err(BoltError::Plan(format!(
                "table '{name}' is external and read-only"
            )));
        }
        let before = meta.files.len();
        if before <= 1 {
            return Ok(CompactReport {
                files_before: before,
                files_after: before,
            });
        }
        let table = self.open_table(name)?;
        let all = table.plan_scan(&crate::storage::ScanRequest::full());
        let batches: Vec<RecordBatch> = table.read(&all)?.collect::<BoltResult<_>>()?;
        let mut next = self.manifest.clone();
        let version = next.version + 1;
        let files = self.write_files(meta.id, version, &batches)?;
        let after = files.len();
        next.tables.get_mut(name).expect("checked").files = files;
        self.commit(next)?;
        Ok(CompactReport {
            files_before: before,
            files_after: after,
        })
    }

    /// Create or replace materialized view `name`.
    pub fn put_view(&mut self, name: &str, view: ViewMeta) -> BoltResult<()> {
        self.writable()?;
        let mut next = self.manifest.clone();
        next.views.insert(name.to_string(), view);
        self.commit(next)
    }

    /// Remove materialized view `name` (not its backing table).
    pub fn remove_view(&mut self, name: &str) -> BoltResult<()> {
        self.writable()?;
        let mut next = self.manifest.clone();
        next.views.remove(name);
        self.commit(next)
    }

    /// Replace the rows of managed table `name` with `batches` in one
    /// snapshot (used to refresh materialized views).
    pub fn replace_contents(&mut self, name: &str, batches: &[RecordBatch]) -> BoltResult<()> {
        self.writable()?;
        let id = self.require(name)?.id;
        let mut next = self.manifest.clone();
        let version = next.version + 1;
        let files = self.write_files(id, version, batches)?;
        let t = next.tables.get_mut(name).expect("checked");
        t.files = files;
        t.table_version += 1;
        t.analyzed = None;
        self.commit(next)
    }

    /// Delete all but the newest `keep_snapshots` manifests (at least the
    /// current one is kept), then every data file no remaining manifest
    /// references.
    pub fn vacuum(&mut self, keep_snapshots: usize) -> BoltResult<VacuumReport> {
        self.writable()?;
        let keep = keep_snapshots.max(1);
        let snaps = self.snapshots()?;
        let mut report = VacuumReport::default();
        let cut = snaps.len().saturating_sub(keep);
        for &v in &snaps[..cut] {
            if v == self.manifest.version {
                continue;
            }
            if fs::remove_file(manifest_path(&self.root, v)).is_ok() {
                report.manifests_removed += 1;
            }
        }
        let mut referenced = std::collections::HashSet::new();
        for v in self.snapshots()? {
            let m = if v == self.manifest.version {
                self.manifest.clone()
            } else {
                read_manifest(&self.root, v)?
            };
            for t in m.tables.values() {
                for f in &t.files {
                    referenced.insert(self.root.join(&f.path));
                }
            }
        }
        let data = self.root.join("data");
        for dir in fs::read_dir(&data)
            .map_err(|e| io_err(data.display(), e))?
            .flatten()
        {
            if !dir.path().is_dir() {
                continue;
            }
            let mut left = 0;
            for f in fs::read_dir(dir.path())
                .map_err(|e| io_err(dir.path().display(), e))?
                .flatten()
            {
                let p = f.path();
                if referenced.contains(&p) {
                    left += 1;
                } else if fs::remove_file(&p).is_ok() {
                    report.files_removed += 1;
                }
            }
            if left == 0 {
                let _ = fs::remove_dir(dir.path());
            }
        }
        Ok(report)
    }

    /// Write `batches` as data files of table `id` for snapshot `version`.
    fn write_files(
        &self,
        id: u64,
        version: u64,
        batches: &[RecordBatch],
    ) -> BoltResult<Vec<DataFile>> {
        let batches: Vec<RecordBatch> = batches
            .iter()
            .filter(|b| b.num_rows() > 0)
            .cloned()
            .collect();
        if batches.is_empty() {
            return Ok(Vec::new());
        }
        let rel_dir = format!("data/{id}");
        fs::create_dir_all(self.root.join(&rel_dir)).map_err(|e| io_err(&rel_dir, e))?;
        // One file per append keeps snapshots cheap; the writer splits it
        // into row groups for pruning. (`compact` rewrites many into few.)
        let rel = format!("{rel_dir}/part-{version:09}-0.parquet");
        let path = self.root.join(&rel);
        let rows = write_parquet(&path, &batches, &self.write_options)?;
        let bytes = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let stats = ParquetTable::open(&path)?.statistics();
        Ok(vec![DataFile {
            path: rel,
            rows,
            bytes,
            stats,
        }])
    }

    /// Durably publish `next` as the new current snapshot.
    fn commit(&mut self, mut next: Manifest) -> BoltResult<()> {
        next.version = self.manifest.version + 1;
        next.format = FORMAT_VERSION;
        let path = manifest_path(&self.root, next.version);
        let text =
            serde_json::to_vec_pretty(&next).map_err(|e| io_err("serialising manifest", e))?;
        write_atomic(&path, &text)?;
        let current = self.root.join("_bolt").join("CURRENT");
        write_atomic(
            &current,
            format!("manifest-{:09}.json\n", next.version).as_bytes(),
        )?;
        self.manifest = next;
        Ok(())
    }
}

fn manifest_path(root: &Path, version: u64) -> PathBuf {
    root.join("_bolt")
        .join("manifests")
        .join(format!("manifest-{version:09}.json"))
}

fn read_current(root: &Path) -> BoltResult<Option<u64>> {
    let p = root.join("_bolt").join("CURRENT");
    match fs::read_to_string(&p) {
        Ok(s) => {
            let v = s
                .trim()
                .strip_prefix("manifest-")
                .and_then(|r| r.strip_suffix(".json"))
                .and_then(|n| n.parse().ok())
                .ok_or_else(|| {
                    BoltError::Other(format!("catalog: malformed {}: {s:?}", p.display()))
                })?;
            Ok(Some(v))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(io_err(p.display(), e)),
    }
}

fn read_manifest(root: &Path, version: u64) -> BoltResult<Manifest> {
    let p = manifest_path(root, version);
    let text = fs::read(&p).map_err(|e| io_err(p.display(), e))?;
    let m: Manifest = serde_json::from_slice(&text).map_err(|e| io_err(p.display(), e))?;
    if m.format > FORMAT_VERSION {
        return Err(BoltError::Plan(format!(
            "catalog manifest {} has format {}, newer than this build supports ({FORMAT_VERSION})",
            p.display(),
            m.format
        )));
    }
    Ok(m)
}

/// Write `bytes` to `path` via a synced temporary sibling and a rename.
fn write_atomic(path: &Path, bytes: &[u8]) -> BoltResult<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = fs::File::create(&tmp).map_err(|e| io_err(tmp.display(), e))?;
        f.write_all(bytes).map_err(|e| io_err(tmp.display(), e))?;
        f.sync_all().map_err(|e| io_err(tmp.display(), e))?;
    }
    fs::rename(&tmp, path).map_err(|e| io_err(path.display(), e))
}

/// Catalog text form of a plan type (`int64`, `decimal128(10,2)`,
/// `timestamp(us,UTC)`, …).
pub fn dtype_to_text(d: DataType) -> BoltResult<String> {
    Ok(match d {
        DataType::Bool => "bool".into(),
        DataType::Int32 => "int32".into(),
        DataType::Int64 => "int64".into(),
        DataType::Float32 => "float32".into(),
        DataType::Float64 => "float64".into(),
        DataType::Utf8 => "utf8".into(),
        DataType::Date32 => "date32".into(),
        DataType::Decimal128(p, s) => format!("decimal128({p},{s})"),
        DataType::Timestamp(u, tz) => {
            let u = match u {
                TimeUnit::Second => "s",
                TimeUnit::Millisecond => "ms",
                TimeUnit::Microsecond => "us",
                TimeUnit::Nanosecond => "ns",
            };
            match tz {
                Some(tz) => format!("timestamp({u},{tz})"),
                None => format!("timestamp({u})"),
            }
        }
        #[allow(unreachable_patterns)]
        other => {
            return Err(BoltError::Type(format!(
                "catalog cannot store type {other:?}"
            )))
        }
    })
}

/// Inverse of [`dtype_to_text`].
pub fn dtype_from_text(s: &str) -> BoltResult<DataType> {
    let bad = || BoltError::Other(format!("catalog: unknown column type '{s}'"));
    Ok(match s {
        "bool" => DataType::Bool,
        "int32" => DataType::Int32,
        "int64" => DataType::Int64,
        "float32" => DataType::Float32,
        "float64" => DataType::Float64,
        "utf8" => DataType::Utf8,
        "date32" => DataType::Date32,
        _ => {
            let (head, args) = s
                .strip_suffix(')')
                .and_then(|r| r.split_once('('))
                .ok_or_else(bad)?;
            let mut parts = args.splitn(2, ',');
            let a = parts.next().ok_or_else(bad)?;
            let b = parts.next();
            match head {
                "decimal128" => DataType::Decimal128(
                    a.parse().map_err(|_| bad())?,
                    b.ok_or_else(bad)?.parse().map_err(|_| bad())?,
                ),
                "timestamp" => {
                    let u = match a {
                        "s" => TimeUnit::Second,
                        "ms" => TimeUnit::Millisecond,
                        "us" => TimeUnit::Microsecond,
                        "ns" => TimeUnit::Nanosecond,
                        _ => return Err(bad()),
                    };
                    DataType::Timestamp(u, b.map(crate::plan::logical_plan::intern_timezone))
                }
                _ => return Err(bad()),
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, StringArray};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    fn root(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "bolt-cat-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        d
    }

    fn batch(lo: i64, n: usize) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int64, false),
            ArrowField::new("s", ArrowDataType::Utf8, true),
        ]));
        let k: Vec<i64> = (lo..lo + n as i64).collect();
        let s: Vec<String> = k.iter().map(|v| format!("s{v}")).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(k)),
                Arc::new(StringArray::from(s)),
            ],
        )
        .unwrap()
    }

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("s", DataType::Utf8, true),
        ])
    }

    fn rows(c: &Catalog, t: &str) -> usize {
        let tbl = c.open_table(t).unwrap();
        let sel = tbl.plan_scan(&crate::storage::ScanRequest::full());
        tbl.read(&sel).unwrap().map(|b| b.unwrap().num_rows()).sum()
    }

    #[test]
    fn create_append_reopen_and_time_travel() {
        let r = root("rt");
        {
            let mut c = Catalog::open(&r).unwrap();
            assert_eq!(c.version(), 0);
            c.create_table("t", &schema(), &[batch(0, 10)]).unwrap();
            c.append("t", &[batch(10, 5)]).unwrap();
            assert_eq!(c.version(), 2);
            assert_eq!(rows(&c, "t"), 15);
            assert_eq!(c.table("t").unwrap().table_version, 2);
            // A second writer is refused while this one is open.
            assert!(Catalog::open(&r).is_err());
        }
        let mut c = Catalog::open(&r).unwrap();
        assert_eq!(c.version(), 2);
        assert_eq!(rows(&c, "t"), 15);
        assert_eq!(c.table("t").unwrap().files.len(), 2);
        let old = Catalog::open_snapshot(&r, 1).unwrap();
        assert!(old.is_read_only());
        assert_eq!(rows(&old, "t"), 10);
        let rep = c.compact("t").unwrap();
        assert_eq!((rep.files_before, rep.files_after), (2, 1));
        assert_eq!(rows(&c, "t"), 15);
        c.drop_table("t").unwrap();
        assert!(c.table("t").is_none());
        // Dropped data stays readable from an old snapshot until vacuum.
        assert_eq!(rows(&Catalog::open_snapshot(&r, 2).unwrap(), "t"), 15);
        let v = c.vacuum(1).unwrap();
        assert_eq!(v.manifests_removed, 3);
        assert_eq!(v.files_removed, 3);
        assert_eq!(c.snapshots().unwrap(), vec![c.version()]);
        drop(c);
        fs::remove_dir_all(&r).ok();
    }

    #[test]
    fn empty_tables_truncate_and_stats() {
        let r = root("empty");
        let mut c = Catalog::open(&r).unwrap();
        c.create_table("e", &schema(), &[]).unwrap();
        assert_eq!(rows(&c, "e"), 0);
        assert_eq!(c.open_table("e").unwrap().schema().fields.len(), 2);
        c.append("e", &[batch(0, 3)]).unwrap();
        c.truncate("e").unwrap();
        assert_eq!(rows(&c, "e"), 0);
        c.set_statistics("e", TableStatistics::default()).unwrap();
        assert!(c.table("e").unwrap().analyzed.is_some());
        assert!(c.create_table("e", &schema(), &[]).is_err());
        drop(c);
        fs::remove_dir_all(&r).ok();
    }

    #[test]
    fn dtype_text_round_trips() {
        for d in [
            DataType::Bool,
            DataType::Int32,
            DataType::Int64,
            DataType::Float32,
            DataType::Float64,
            DataType::Utf8,
            DataType::Date32,
            DataType::Decimal128(38, 10),
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Timestamp(TimeUnit::Nanosecond, Some("Europe/Paris")),
        ] {
            assert_eq!(dtype_from_text(&dtype_to_text(d).unwrap()).unwrap(), d);
        }
        assert!(dtype_from_text("blob").is_err());
    }
}
