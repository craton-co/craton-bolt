// SPDX-License-Identifier: Apache-2.0
//! The storage-table contract and the in-memory implementation.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::array::ArrayData;
use arrow_array::{make_array, ArrayRef, RecordBatch};
use arrow_buffer::{Buffer, MutableBuffer};
use arrow_schema::{DataType as ArrowDataType, SchemaRef};

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::Schema;
use crate::storage::scan::ScanRequest;
use crate::storage::stats::TableStatistics;

/// Fallible batch iterator returned by [`StorageTable::read`].
pub type BatchIter = Box<dyn Iterator<Item = BoltResult<RecordBatch>> + Send>;

/// What a scan will read, decided from chunk statistics before any data is
/// touched.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanSelection {
    /// Chunk ordinals to read, ascending.
    pub chunks: Vec<usize>,
    /// Columns to read (`None` = all). Other columns come back as
    /// placeholders.
    pub columns: Option<BTreeSet<usize>>,
    /// Total chunks in the table.
    pub chunks_total: usize,
    /// Rows in the selected chunks.
    pub rows: u64,
    /// Approximate bytes of the selected chunks.
    pub bytes: u64,
}

impl ScanSelection {
    /// `true` when this selection reads at least everything `other` reads.
    pub fn covers(&self, other: &ScanSelection) -> bool {
        let cols = match (&self.columns, &other.columns) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(a), Some(b)) => b.is_subset(a),
        };
        cols && other
            .chunks
            .iter()
            .all(|c| self.chunks.binary_search(c).is_ok())
    }

    /// Selection reading everything either side reads.
    pub fn union(&self, other: &ScanSelection) -> ScanSelection {
        let mut chunks: Vec<usize> = self.chunks.iter().chain(&other.chunks).copied().collect();
        chunks.sort_unstable();
        chunks.dedup();
        let columns = match (&self.columns, &other.columns) {
            (Some(a), Some(b)) => Some(a.union(b).copied().collect()),
            _ => None,
        };
        // Row / byte totals of the union are not derivable without the chunk
        // metadata; keep the larger side as an estimate.
        ScanSelection {
            chunks,
            columns,
            chunks_total: self.chunks_total,
            rows: self.rows.max(other.rows),
            bytes: self.bytes.max(other.bytes),
        }
    }
}

/// A table whose data lives outside the engine's in-memory store — Parquet
/// files, a database catalog table, a user-supplied source.
///
/// The table is divided into **chunks** (row groups, files, batches), each
/// with [`TableStatistics`]. The engine asks for a [`ScanSelection`] per query
/// ([`StorageTable::plan_scan`], which prunes chunks with their zone maps and
/// the query's [`ScanRequest`]), then reads it ([`StorageTable::read`]).
///
/// # Contract
///
/// * Every batch [`StorageTable::read`] yields has exactly the Arrow form of
///   [`StorageTable::schema`] (same field count, names and types). Columns
///   outside [`ScanSelection::columns`] may hold placeholder values
///   ([`placeholder_array`]); the query plan never reads them.
/// * Reads are repeatable: the same selection yields the same rows.
/// * Chunk statistics must be *sound*: a chunk whose min / max / null count
///   would let [`ScanRequest::chunk_may_match`] reject it must really contain
///   no matching row. Unknown statistics are always sound.
///
/// # Stability
///
/// Experimental until 1.0.
pub trait StorageTable: Send + Sync + std::fmt::Debug {
    /// Plan-level schema of the table.
    fn schema(&self) -> &Schema;

    /// Number of chunks.
    fn chunk_count(&self) -> usize;

    /// Statistics of chunk `chunk` (`row_count` must be exact).
    fn chunk_statistics(&self, chunk: usize) -> &TableStatistics;

    /// Read the selected chunks / columns.
    fn read(&self, selection: &ScanSelection) -> BoltResult<BatchIter>;

    /// Table-level statistics. The default merges the chunk statistics
    /// (exact counts and bounds; distinct estimates are a lower bound).
    fn statistics(&self) -> TableStatistics {
        let mut total = TableStatistics::default();
        for c in 0..self.chunk_count() {
            total.merge(self.chunk_statistics(c));
        }
        if total.columns.is_empty() {
            total.columns = vec![Default::default(); self.schema().fields.len()];
        }
        total
    }

    /// Decide which chunks and columns a query needs.
    fn plan_scan(&self, request: &ScanRequest) -> ScanSelection {
        let mut chunks = Vec::new();
        let (mut rows, mut bytes) = (0u64, 0u64);
        for c in 0..self.chunk_count() {
            let s = self.chunk_statistics(c);
            if request.chunk_may_match(&s.columns, s.row_count) {
                chunks.push(c);
                rows += s.row_count;
                bytes += s.byte_size;
            }
        }
        ScanSelection {
            chunks,
            columns: request.columns.clone(),
            chunks_total: self.chunk_count(),
            rows,
            bytes,
        }
    }
}

/// A zero-filled, NULL-free array of `dtype` with `len` rows, used for
/// columns a scan did not read. Zero values (not NULLs) keep a non-nullable
/// field valid and leave the engine's validity probes on the dense path.
pub fn placeholder_array(dtype: &ArrowDataType, len: usize) -> BoltResult<ArrayRef> {
    let zeroed = |bytes: usize| Buffer::from(MutableBuffer::from_len_zeroed(bytes));
    let data = match dtype {
        ArrowDataType::Boolean => ArrayData::builder(dtype.clone())
            .len(len)
            .add_buffer(zeroed(len.div_ceil(8))),
        ArrowDataType::Utf8 => ArrayData::builder(dtype.clone())
            .len(len)
            .add_buffer(zeroed((len + 1) * 4))
            .add_buffer(zeroed(0)),
        other => match other.primitive_width() {
            Some(w) => ArrayData::builder(other.clone())
                .len(len)
                .add_buffer(zeroed(len * w)),
            None => {
                return Err(BoltError::Type(format!(
                    "no scan placeholder for Arrow type {other:?}"
                )))
            }
        },
    };
    let data = data
        .build()
        .map_err(|e| BoltError::Other(format!("placeholder column: {e}")))?;
    Ok(make_array(data))
}

/// Re-assemble a batch holding only `read_columns` (in ascending ordinal
/// order) into the full `schema`, filling the rest with placeholders.
pub fn widen_to_schema(
    projected: &RecordBatch,
    schema: &SchemaRef,
    read_columns: &[usize],
) -> BoltResult<RecordBatch> {
    let n = projected.num_rows();
    let mut cols = Vec::with_capacity(schema.fields().len());
    let mut next = 0;
    for (i, field) in schema.fields().iter().enumerate() {
        if read_columns.get(next) == Some(&i) {
            cols.push(Arc::clone(projected.column(next)));
            next += 1;
        } else {
            cols.push(placeholder_array(field.data_type(), n)?);
        }
    }
    RecordBatch::try_new(Arc::clone(schema), cols)
        .map_err(|e| BoltError::Other(format!("scan batch assembly failed: {e}")))
}

/// An in-memory [`StorageTable`]: batches split into fixed-size chunks with
/// zone maps. Useful as a reference implementation, and for data that should
/// get chunk pruning without being written to files.
#[derive(Debug, Clone)]
pub struct MemoryTable {
    schema: Schema,
    chunks: Vec<RecordBatch>,
    stats: Vec<TableStatistics>,
}

impl MemoryTable {
    /// Build from `batches` (all with one schema), re-chunked to at most
    /// `chunk_rows` rows per chunk (`None` keeps the batch boundaries).
    pub fn try_new(batches: Vec<RecordBatch>, chunk_rows: Option<usize>) -> BoltResult<Self> {
        let first = batches
            .first()
            .ok_or_else(|| BoltError::Plan("MemoryTable needs at least one batch".into()))?;
        let arrow_schema = first.schema();
        let schema = crate::exec::engine_support::arrow_schema_to_plan_schema(&arrow_schema)?;
        let mut chunks = Vec::new();
        for b in batches {
            if b.schema() != arrow_schema {
                return Err(BoltError::Plan(
                    "MemoryTable batches must share one schema".into(),
                ));
            }
            match chunk_rows {
                Some(n) if n > 0 && b.num_rows() > n => {
                    let mut off = 0;
                    while off < b.num_rows() {
                        let len = n.min(b.num_rows() - off);
                        chunks.push(b.slice(off, len));
                        off += len;
                    }
                }
                _ => chunks.push(b),
            }
        }
        let stats = chunks
            .iter()
            .map(|c| TableStatistics::from_batches(std::slice::from_ref(c)))
            .collect();
        Ok(MemoryTable {
            schema,
            chunks,
            stats,
        })
    }
}

impl StorageTable for MemoryTable {
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
        let batches: Vec<RecordBatch> = selection
            .chunks
            .iter()
            .map(|&c| self.chunks[c].clone())
            .collect();
        Ok(Box::new(batches.into_iter().map(Ok)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::Literal;
    use crate::storage::scan::{PruneOp, PrunePredicate};
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::{Array, Int64Array, StringArray};
    use arrow_schema::{Field, Schema as ArrowSchema};

    fn batch(lo: i64, n: usize) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("k", ArrowDataType::Int64, false),
            Field::new("s", ArrowDataType::Utf8, false),
        ]));
        let k: Vec<i64> = (lo..lo + n as i64).collect();
        let s: Vec<String> = k.iter().map(|v| format!("v{v}")).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(k)),
                Arc::new(StringArray::from(s)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn memory_table_prunes_chunks_by_zone_map() {
        let t = MemoryTable::try_new(vec![batch(0, 100)], Some(25)).unwrap();
        assert_eq!(t.chunk_count(), 4);
        let req = ScanRequest {
            columns: None,
            filters: vec![vec![PrunePredicate::Compare {
                column: 0,
                op: PruneOp::GtEq,
                value: Literal::Int64(60),
            }]],
        };
        let sel = t.plan_scan(&req);
        assert_eq!(sel.chunks, vec![2, 3]);
        assert_eq!(sel.rows, 50);
        let rows: usize = t.read(&sel).unwrap().map(|b| b.unwrap().num_rows()).sum();
        assert_eq!(rows, 50);
        let stats = t.statistics();
        assert_eq!(stats.row_count, 100);
        assert_eq!(stats.columns[0].max, Some(Literal::Int64(99)));
    }

    #[test]
    fn selection_cover_and_union() {
        let a = ScanSelection {
            chunks: vec![0, 2],
            columns: Some([0].into_iter().collect()),
            chunks_total: 4,
            rows: 2,
            bytes: 2,
        };
        let b = ScanSelection {
            chunks: vec![2],
            columns: Some([0, 1].into_iter().collect()),
            ..a.clone()
        };
        assert!(!a.covers(&b));
        let u = a.union(&b);
        assert!(u.covers(&a) && u.covers(&b));
        assert_eq!(u.chunks, vec![0, 2]);
    }

    #[test]
    fn widen_fills_unread_columns_with_placeholders() {
        let full = batch(5, 3);
        let projected = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![full.schema().field(0).clone()])),
            vec![Arc::clone(full.column(0))],
        )
        .unwrap();
        let wide = widen_to_schema(&projected, &full.schema(), &[0]).unwrap();
        assert_eq!(wide.num_columns(), 2);
        assert_eq!(wide.column(0).as_primitive::<Int64Type>().value(2), 7);
        let s = wide.column(1).as_string::<i32>();
        assert_eq!(s.null_count(), 0);
        assert_eq!(s.value(1), "");
    }

    #[test]
    fn placeholders_cover_engine_types() {
        for dt in [
            ArrowDataType::Boolean,
            ArrowDataType::Int32,
            ArrowDataType::Int64,
            ArrowDataType::Float32,
            ArrowDataType::Float64,
            ArrowDataType::Utf8,
            ArrowDataType::Date32,
            ArrowDataType::Decimal128(20, 2),
            ArrowDataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
        ] {
            let a = placeholder_array(&dt, 17).unwrap();
            assert_eq!(a.len(), 17);
            assert_eq!(a.null_count(), 0);
            assert_eq!(a.data_type(), &dt);
        }
    }
}
