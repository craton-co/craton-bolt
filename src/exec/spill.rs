// SPDX-License-Identifier: Apache-2.0
//! Spill files and hash partitioning for out-of-core execution.
//!
//! Spill files are Arrow IPC files in the engine's spill directory, deleted
//! when their [`SpillFile`] handle drops. [`partition_hash`] assigns rows to
//! partitions by a hash of key columns that is normalised across types — an
//! `Int32` key and an `Int64` key with the same value land in the same
//! partition — so two sides of an equi-join partitioned independently stay
//! aligned. NULL keys go to partition 0 (they never match an equi-join, and
//! all-NULL groups stay together).

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float64Type, Int32Type, Int64Type, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType,
};
use arrow_array::{Array, ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::{DataType as ArrowDataType, SchemaRef, TimeUnit as ArrowTimeUnit};

use crate::error::{BoltError, BoltResult};

static SPILL_SEQ: AtomicU64 = AtomicU64::new(0);

fn ioe(ctx: impl std::fmt::Display, e: impl std::fmt::Display) -> BoltError {
    BoltError::Other(format!("spill: {ctx}: {e}"))
}

/// A finished spill file; deleted on drop.
#[derive(Debug)]
pub(crate) struct SpillFile {
    path: PathBuf,
    pub(crate) rows: u64,
    pub(crate) bytes: u64,
}

impl SpillFile {
    /// Read every batch back.
    pub(crate) fn read_all(&self) -> BoltResult<Vec<RecordBatch>> {
        let f = File::open(&self.path).map_err(|e| ioe(self.path.display(), e))?;
        let reader = FileReader::try_new(f, None).map_err(|e| ioe(self.path.display(), e))?;
        reader
            .map(|b| b.map_err(|e| ioe(self.path.display(), e)))
            .collect()
    }
}

impl Drop for SpillFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// An open spill file being appended to.
pub(crate) struct SpillWriter {
    path: PathBuf,
    writer: FileWriter<BufWriter<File>>,
    rows: u64,
}

impl SpillWriter {
    pub(crate) fn create(dir: &Path, schema: &SchemaRef) -> BoltResult<Self> {
        std::fs::create_dir_all(dir).map_err(|e| ioe(dir.display(), e))?;
        let path = dir.join(format!(
            "spill-{}-{}.arrow",
            std::process::id(),
            SPILL_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let f = File::create(&path).map_err(|e| ioe(path.display(), e))?;
        let writer =
            FileWriter::try_new(BufWriter::new(f), schema).map_err(|e| ioe(path.display(), e))?;
        Ok(SpillWriter {
            path,
            writer,
            rows: 0,
        })
    }

    pub(crate) fn write(&mut self, batch: &RecordBatch) -> BoltResult<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        self.rows += batch.num_rows() as u64;
        self.writer
            .write(batch)
            .map_err(|e| ioe(self.path.display(), e))
    }

    pub(crate) fn finish(mut self) -> BoltResult<SpillFile> {
        self.writer
            .finish()
            .map_err(|e| ioe(self.path.display(), e))?;
        let bytes = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);
        let path = std::mem::take(&mut self.path);
        Ok(SpillFile {
            path,
            rows: self.rows,
            bytes,
        })
    }
}

impl Drop for SpillWriter {
    fn drop(&mut self) {
        // An unfinished writer (error path) leaves no file behind.
        if !self.path.as_os_str().is_empty() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn mix(mut h: u64) -> u64 {
    // splitmix64 finaliser.
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(FNV_OFFSET, |h, b| (h ^ *b as u64).wrapping_mul(FNV_PRIME))
}

/// Type family a key hashes as; two columns are join-compatible for
/// partitioning only when their families are equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeyFamily {
    Integer,
    Float,
    Text,
    Bool,
    Decimal(i8),
    Timestamp(ArrowTimeUnit),
}

/// Hash family of an Arrow type, or `None` when it cannot be a partition
/// key.
pub(crate) fn key_family(dt: &ArrowDataType) -> Option<KeyFamily> {
    Some(match dt {
        ArrowDataType::Int8
        | ArrowDataType::Int16
        | ArrowDataType::Int32
        | ArrowDataType::Int64
        | ArrowDataType::Date32 => KeyFamily::Integer,
        ArrowDataType::Float32 | ArrowDataType::Float64 => KeyFamily::Float,
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => KeyFamily::Text,
        ArrowDataType::Dictionary(_, v) if matches!(v.as_ref(), ArrowDataType::Utf8) => {
            KeyFamily::Text
        }
        ArrowDataType::Boolean => KeyFamily::Bool,
        ArrowDataType::Decimal128(_, s) => KeyFamily::Decimal(*s),
        ArrowDataType::Timestamp(u, _) => KeyFamily::Timestamp(*u),
        _ => return None,
    })
}

/// Fold one value per row into `h`; rows `valid` rejects are flagged NULL.
fn fold_rows(
    h: &mut [u64],
    null: &mut [bool],
    valid: impl Fn(usize) -> bool,
    value: impl Fn(usize) -> u64,
) {
    for (i, (hv, nv)) in h.iter_mut().zip(null.iter_mut()).enumerate() {
        if valid(i) {
            *hv = mix(*hv ^ value(i));
        } else {
            *nv = true;
        }
    }
}

/// Fold column `col` into the per-row hashes `h`; rows where it is NULL are
/// flagged in `null`.
fn hash_column(col: &ArrayRef, h: &mut [u64], null: &mut [bool]) -> BoltResult<()> {
    macro_rules! ints {
        ($t:ty) => {{
            let a = col.as_primitive::<$t>();
            fold_rows(h, null, |i| a.is_valid(i), |i| a.value(i) as i64 as u64)
        }};
    }
    let widen = |to: &ArrowDataType| arrow::compute::cast(col, to).map_err(|e| ioe("key cast", e));
    match col.data_type() {
        ArrowDataType::Int32 => ints!(Int32Type),
        ArrowDataType::Int64 => ints!(Int64Type),
        ArrowDataType::Date32 => ints!(Date32Type),
        ArrowDataType::Int8 | ArrowDataType::Int16 => {
            return hash_column(&widen(&ArrowDataType::Int64)?, h, null)
        }
        ArrowDataType::Timestamp(u, _) => match u {
            ArrowTimeUnit::Second => ints!(TimestampSecondType),
            ArrowTimeUnit::Millisecond => ints!(TimestampMillisecondType),
            ArrowTimeUnit::Microsecond => ints!(TimestampMicrosecondType),
            ArrowTimeUnit::Nanosecond => ints!(TimestampNanosecondType),
        },
        ArrowDataType::Float32 => return hash_column(&widen(&ArrowDataType::Float64)?, h, null),
        ArrowDataType::Float64 => {
            let a = col.as_primitive::<Float64Type>();
            // -0.0 == 0.0 and every NaN alike.
            let bits = |v: f64| {
                if v == 0.0 {
                    0
                } else if v.is_nan() {
                    f64::NAN.to_bits()
                } else {
                    v.to_bits()
                }
            };
            fold_rows(h, null, |i| a.is_valid(i), |i| bits(a.value(i)))
        }
        ArrowDataType::Decimal128(_, _) => {
            let a = col.as_primitive::<Decimal128Type>();
            fold_rows(
                h,
                null,
                |i| a.is_valid(i),
                |i| {
                    let v = a.value(i);
                    (v as u64) ^ ((v >> 64) as u64).rotate_left(17)
                },
            )
        }
        ArrowDataType::Boolean => {
            let a = col.as_boolean();
            fold_rows(h, null, |i| a.is_valid(i), |i| a.value(i) as u64 + 1)
        }
        ArrowDataType::Utf8 => {
            let a = col.as_string::<i32>();
            fold_rows(h, null, |i| a.is_valid(i), |i| fnv(a.value(i).as_bytes()))
        }
        ArrowDataType::LargeUtf8 => {
            let a = col.as_string::<i64>();
            fold_rows(h, null, |i| a.is_valid(i), |i| fnv(a.value(i).as_bytes()))
        }
        ArrowDataType::Dictionary(_, _) => {
            return hash_column(&widen(&ArrowDataType::Utf8)?, h, null)
        }
        other => {
            return Err(BoltError::Unsupported(format!(
                "cannot partition on a {other:?} key"
            )))
        }
    }
    Ok(())
}

/// Partition index of every row of `batch` by its `keys` columns.
pub(crate) fn partition_hash(
    batch: &RecordBatch,
    keys: &[usize],
    partitions: usize,
    seed: u64,
) -> BoltResult<Vec<u32>> {
    let n = batch.num_rows();
    let mut h = vec![mix(seed); n];
    let mut null = vec![false; n];
    for &k in keys {
        hash_column(batch.column(k), &mut h, &mut null)?;
    }
    Ok(h.iter()
        .zip(&null)
        .map(|(&v, &is_null)| {
            if is_null {
                0
            } else {
                (v % partitions as u64) as u32
            }
        })
        .collect())
}

/// Split `batch` into one (possibly empty) batch per partition.
pub(crate) fn split_by_partition(
    batch: &RecordBatch,
    assignment: &[u32],
    partitions: usize,
) -> BoltResult<Vec<RecordBatch>> {
    let mut idx: Vec<Vec<u32>> = vec![Vec::new(); partitions];
    for (row, &p) in assignment.iter().enumerate() {
        idx[p as usize].push(row as u32);
    }
    idx.into_iter()
        .map(|rows| {
            let take = UInt32Array::from(rows);
            let cols = batch
                .columns()
                .iter()
                .map(|c| arrow::compute::take(c.as_ref(), &take, None))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| ioe("partition take", e))?;
            RecordBatch::try_new(batch.schema(), cols).map_err(|e| ioe("partition batch", e))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use arrow_array::{Float32Array, Float64Array, Int32Array, Int64Array, StringArray};
    use arrow_schema::{Field, Schema};

    fn one(col: ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "k",
            col.data_type().clone(),
            true,
        )]));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    #[test]
    fn hashing_is_normalised_across_widths() {
        let a = one(Arc::new(Int32Array::from(vec![
            Some(1),
            Some(-7),
            None,
            Some(40),
        ])));
        let b = one(Arc::new(Int64Array::from(vec![
            Some(1),
            Some(-7),
            None,
            Some(40),
        ])));
        assert_eq!(
            partition_hash(&a, &[0], 8, 3).unwrap(),
            partition_hash(&b, &[0], 8, 3).unwrap()
        );
        let f = one(Arc::new(Float32Array::from(vec![0.0f32, -0.0, 2.5])));
        let d = one(Arc::new(Float64Array::from(vec![-0.0f64, 0.0, 2.5])));
        assert_eq!(
            partition_hash(&f, &[0], 8, 3).unwrap(),
            partition_hash(&d, &[0], 8, 3).unwrap()
        );
        let s = one(Arc::new(StringArray::from(vec![Some("x"), None])));
        assert_eq!(
            partition_hash(&s, &[0], 8, 3).unwrap()[1],
            0,
            "NULL keys go to partition 0"
        );
        assert_eq!(
            key_family(&ArrowDataType::Int32),
            key_family(&ArrowDataType::Int64)
        );
        assert_ne!(
            key_family(&ArrowDataType::Int32),
            key_family(&ArrowDataType::Float64)
        );
    }

    #[test]
    fn split_and_spill_round_trip() {
        let b = one(Arc::new(Int64Array::from((0..1000).collect::<Vec<i64>>())));
        let assign = partition_hash(&b, &[0], 4, 0).unwrap();
        let parts = split_by_partition(&b, &assign, 4).unwrap();
        assert_eq!(parts.iter().map(RecordBatch::num_rows).sum::<usize>(), 1000);
        assert!(
            parts.iter().all(|p| p.num_rows() > 150),
            "hash spreads rows"
        );
        let dir = std::env::temp_dir().join(format!("bolt-spill-test-{}", std::process::id()));
        let mut w = SpillWriter::create(&dir, &b.schema()).unwrap();
        w.write(&parts[1]).unwrap();
        w.write(&parts[2]).unwrap();
        let f = w.finish().unwrap();
        assert_eq!(f.rows as usize, parts[1].num_rows() + parts[2].num_rows());
        let back = f.read_all().unwrap();
        assert_eq!(
            back.iter().map(RecordBatch::num_rows).sum::<usize>(),
            f.rows as usize
        );
        let path = f.path.clone();
        drop(f);
        assert!(!path.exists(), "spill files are deleted on drop");
        let _ = std::fs::remove_dir(&dir);
    }
}
