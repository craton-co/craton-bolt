// SPDX-License-Identifier: Apache-2.0

//! Storage form of extended columns (see [`crate::plan::extended_types`]).
//!
//! [`ExtendedStore::ingest`] converts a registered batch to the columns the
//! executors run on: time and duration columns become their `Int32` /
//! `Int64` tick arrays (zero-copy), binary columns lowercase hex `Utf8`,
//! `LargeUtf8` / `Utf8View` plain `Utf8`, and an opaque column an `Int64`
//! handle column (`(source << 40) | row`, NULL exactly where the value is)
//! while the store keeps the original arrays. [`ExtendedStore::restore`]
//! turns a result's storage columns back into the query's logical output
//! types; opaque values are gathered from the kept arrays by handle.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow_array::{
    make_array, new_null_array, Array, ArrayRef, BinaryArray, BinaryViewArray,
    FixedSizeBinaryArray, Int64Array, LargeBinaryArray, RecordBatch, StringArray,
};
use arrow_schema::{DataType as A, Field as ArrowField, Schema as ArrowSchema, SchemaRef};

use crate::error::{BoltError, BoltResult};
use crate::plan::extended_types::{family, hex, storage_dtype, Family};
use crate::plan::logical_plan::{DataType, Schema};

/// Bits of a handle that hold the row; the rest hold the source id.
const ROW_BITS: u32 = 40;

/// The kept arrays of one opaque column, in row order.
#[derive(Debug, Default)]
struct Source {
    arrays: Vec<ArrayRef>,
    /// Row number of each array's first row.
    starts: Vec<usize>,
    rows: usize,
}

/// A table's extended columns.
#[derive(Debug, Clone)]
struct TableEntry {
    /// The registered (logical) Arrow schema.
    schema: SchemaRef,
    /// Source id of each opaque column, by column index.
    sources: Vec<Option<u32>>,
}

/// Per-engine store of opaque column values and logical table schemas.
#[derive(Debug, Default)]
pub(crate) struct ExtendedStore {
    sources: HashMap<u32, Source>,
    tables: HashMap<String, TableEntry>,
    next_source: u32,
}

/// The result of [`ExtendedStore::ingest`], applied by
/// [`ExtendedStore::commit`] once the table is installed.
#[derive(Debug)]
pub(crate) struct Staged {
    table: String,
    entry: Option<TableEntry>,
    append: bool,
    /// Opaque arrays to keep: `(source, array)`.
    kept: Vec<(u32, ArrayRef)>,
    next_source: u32,
    /// The logical engine type of each column whose storage differs.
    logical: Vec<(usize, DataType)>,
}

impl Staged {
    /// `schema` (the storage plan schema the engine derived, possibly with
    /// appended `__idx_` columns) with every converted column given its
    /// logical type, for the SQL frontend.
    pub(crate) fn logical_schema(&self, mut schema: Schema) -> Schema {
        for (i, dt) in &self.logical {
            schema.fields[*i].dtype = *dt;
        }
        schema
    }
}

fn arrow_err(e: arrow_schema::ArrowError) -> BoltError {
    BoltError::Other(format!("extended types: {e}"))
}

fn reinterpret(a: &dyn Array, to: A) -> BoltResult<ArrayRef> {
    let data = a
        .to_data()
        .into_builder()
        .data_type(to)
        .build()
        .map_err(arrow_err)?;
    Ok(make_array(data))
}

fn storage_arrow(dt: DataType) -> A {
    match storage_dtype(dt) {
        DataType::Int32 => A::Int32,
        DataType::Utf8 => A::Utf8,
        _ => A::Int64,
    }
}

impl ExtendedStore {
    /// The storage form of `batch`, registered (`append == false`) or
    /// appended to `table`. Nothing changes until [`Self::commit`].
    pub(crate) fn ingest(
        &self,
        table: &str,
        batch: &RecordBatch,
        append: bool,
    ) -> BoltResult<(RecordBatch, Staged)> {
        let schema = batch.schema();
        let previous = self.tables.get(table).filter(|_| append);
        if let Some(prev) = previous {
            if prev.schema.fields().len() != schema.fields().len()
                || prev
                    .schema
                    .fields()
                    .iter()
                    .zip(schema.fields())
                    .any(|(a, b)| a.name() != b.name() || a.data_type() != b.data_type())
            {
                return Err(BoltError::Plan(format!(
                    "register_batch: schema mismatch for table '{table}' — expected {:?}, \
                     got {:?}",
                    prev.schema, schema
                )));
            }
        }
        let mut next_source = self.next_source;
        let mut staged = Staged {
            table: table.to_string(),
            entry: None,
            append,
            kept: Vec::new(),
            next_source,
            logical: Vec::new(),
        };
        let mut sources = Vec::with_capacity(schema.fields().len());
        let mut fields = Vec::with_capacity(schema.fields().len());
        let mut columns = Vec::with_capacity(schema.fields().len());
        for (i, (field, col)) in schema.fields().iter().zip(batch.columns()).enumerate() {
            let dt = DataType::from_arrow(col.data_type())
                .map_err(|e| BoltError::Type(format!("column '{}': {e}", field.name())))?;
            let mut source = None;
            let stored: ArrayRef = match (dt, col.data_type()) {
                (DataType::Utf8, A::LargeUtf8 | A::Utf8View) => {
                    arrow::compute::cast(col, &A::Utf8).map_err(arrow_err)?
                }
                (DataType::Extended(t), _) => {
                    // A dictionary of an extended value type is unpacked.
                    let col = if col.data_type() == t {
                        Arc::clone(col)
                    } else {
                        arrow::compute::cast(col, t).map_err(arrow_err)?
                    };
                    staged.logical.push((i, dt));
                    match family(t) {
                        Family::Time(_) | Family::Duration(_) => {
                            reinterpret(col.as_ref(), storage_arrow(dt))?
                        }
                        Family::Binary => Arc::new(to_hex(col.as_ref())?),
                        Family::Opaque => {
                            let (sid, base) = match previous.and_then(|p| p.sources[i]) {
                                Some(sid) => (sid, self.sources.get(&sid).map_or(0, |s| s.rows)),
                                None => {
                                    let sid = next_source;
                                    next_source += 1;
                                    (sid, 0)
                                }
                            };
                            if sid >= 1 << (63 - ROW_BITS)
                                || (base + col.len()) as u64 >= 1 << ROW_BITS
                            {
                                return Err(BoltError::Memory(format!(
                                    "column '{}': too many rows or columns of extended types",
                                    field.name()
                                )));
                            }
                            source = Some(sid);
                            staged.kept.push((sid, Arc::clone(&col)));
                            Arc::new(handles(col.as_ref(), sid, base))
                        }
                    }
                }
                _ => Arc::clone(col),
            };
            sources.push(source);
            fields.push(ArrowField::new(
                field.name(),
                stored.data_type().clone(),
                field.is_nullable() || stored.data_type() != col.data_type(),
            ));
            columns.push(stored);
        }
        staged.next_source = next_source;
        if !staged.logical.is_empty() || previous.is_some() {
            staged.entry = Some(TableEntry {
                schema: Arc::clone(&schema),
                sources,
            });
        }
        let out =
            RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns).map_err(arrow_err)?;
        Ok((out, staged))
    }

    /// Apply `staged`: keep its opaque arrays and record the table's
    /// logical schema (a registration replaces the table's previous entry
    /// and drops its kept arrays).
    pub(crate) fn commit(&mut self, staged: Staged) {
        if !staged.append {
            self.forget(&staged.table);
        }
        for (sid, array) in staged.kept {
            let s = self.sources.entry(sid).or_default();
            s.starts.push(s.rows);
            s.rows += array.len();
            s.arrays.push(array);
        }
        self.next_source = staged.next_source;
        if let Some(entry) = staged.entry {
            self.tables.insert(staged.table, entry);
        }
    }

    /// The registered (logical) Arrow schema of `table`, when it has an
    /// extended column.
    pub(crate) fn logical_schema(&self, table: &str) -> Option<SchemaRef> {
        self.tables.get(table).map(|e| Arc::clone(&e.schema))
    }

    /// Drop table `table`'s entry and kept arrays.
    pub(crate) fn forget(&mut self, table: &str) {
        if let Some(entry) = self.tables.remove(table) {
            for sid in entry.sources.into_iter().flatten() {
                self.sources.remove(&sid);
            }
        }
    }

    /// `batch`'s storage columns as the types of `logical` (the query's
    /// output schema; columns match by position).
    pub(crate) fn restore(&self, batch: RecordBatch, logical: &Schema) -> BoltResult<RecordBatch> {
        if batch.num_columns() != logical.fields.len() {
            return Err(BoltError::Other(format!(
                "extended types: result has {} columns, the plan {}",
                batch.num_columns(),
                logical.fields.len()
            )));
        }
        let schema = batch.schema();
        let mut fields = Vec::with_capacity(batch.num_columns());
        let mut columns = Vec::with_capacity(batch.num_columns());
        for ((field, col), lf) in schema
            .fields()
            .iter()
            .zip(batch.columns())
            .zip(&logical.fields)
        {
            let restored = match lf.dtype {
                DataType::Extended(t) => self.restore_column(col, lf.dtype, t)?,
                _ => Arc::clone(col),
            };
            fields.push(ArrowField::new(
                field.name(),
                restored.data_type().clone(),
                field.is_nullable() || restored.null_count() > 0,
            ));
            columns.push(restored);
        }
        RecordBatch::try_new_with_options(
            Arc::new(ArrowSchema::new(fields)),
            columns,
            &arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )
        .map_err(arrow_err)
    }

    fn restore_column(&self, col: &ArrayRef, dt: DataType, t: &'static A) -> BoltResult<ArrayRef> {
        match family(t) {
            Family::Opaque => {
                let col = arrow::compute::cast(col, &A::Int64).map_err(arrow_err)?;
                self.gather(col.as_primitive::<arrow_array::types::Int64Type>(), t)
            }
            _ => from_storage(col, dt),
        }
    }

    /// The opaque values `handles` point at.
    fn gather(&self, handles: &Int64Array, t: &A) -> BoltResult<ArrayRef> {
        let mut arrays: Vec<&dyn Array> = Vec::new();
        let mut first: HashMap<u32, usize> = HashMap::new();
        let null = new_null_array(t, 1);
        let mut indices = Vec::with_capacity(handles.len());
        for i in 0..handles.len() {
            if handles.is_null(i) {
                indices.push((usize::MAX, 0));
                continue;
            }
            let h = handles.value(i) as u64;
            let sid = (h >> ROW_BITS) as u32;
            let row = (h & ((1 << ROW_BITS) - 1)) as usize;
            let source = self.sources.get(&sid).ok_or_else(|| {
                BoltError::Other(format!("extended types: unknown value source {sid}"))
            })?;
            let base = *first.entry(sid).or_insert_with(|| {
                let at = arrays.len();
                arrays.extend(source.arrays.iter().map(|a| a.as_ref()));
                at
            });
            let chunk = source.starts.partition_point(|&s| s <= row) - 1;
            indices.push((base + chunk, row - source.starts[chunk]));
        }
        let null_at = arrays.len();
        arrays.push(null.as_ref());
        for ix in &mut indices {
            if ix.0 == usize::MAX {
                *ix = (null_at, 0);
            }
        }
        arrow::compute::interleave(&arrays, &indices).map_err(arrow_err)
    }
}

/// A time, duration or binary column of type `dt` from its storage form
/// (ticks, hex). Opaque values have no storage form outside a table.
pub(crate) fn from_storage(col: &ArrayRef, dt: DataType) -> BoltResult<ArrayRef> {
    let DataType::Extended(t) = dt else {
        return Ok(Arc::clone(col));
    };
    let storage = storage_arrow(dt);
    let col = if col.data_type() == &storage {
        Arc::clone(col)
    } else {
        arrow::compute::cast(col, &storage).map_err(arrow_err)?
    };
    match family(t) {
        Family::Time(_) | Family::Duration(_) => reinterpret(col.as_ref(), t.clone()),
        Family::Binary => from_hex(col.as_string::<i32>(), t),
        Family::Opaque => Err(BoltError::Type(format!(
            "{t} values cannot be built from constants"
        ))),
    }
}

/// `(sid << ROW_BITS) | (base + i)`, NULL where `col` is.
fn handles(col: &dyn Array, sid: u32, base: usize) -> Int64Array {
    let high = (sid as i64) << ROW_BITS;
    let values: Vec<i64> = (0..col.len()).map(|i| high | (base + i) as i64).collect();
    Int64Array::new(values.into(), col.logical_nulls())
}

fn to_hex(col: &dyn Array) -> BoltResult<StringArray> {
    Ok(match col.data_type() {
        A::Binary => col.as_binary::<i32>().iter().map(|v| v.map(hex)).collect(),
        A::LargeBinary => col.as_binary::<i64>().iter().map(|v| v.map(hex)).collect(),
        A::BinaryView => col.as_binary_view().iter().map(|v| v.map(hex)).collect(),
        A::FixedSizeBinary(_) => col
            .as_fixed_size_binary()
            .iter()
            .map(|v| v.map(hex))
            .collect(),
        other => {
            return Err(BoltError::Type(format!("{other} is not a binary type")));
        }
    })
}

fn unhex(s: &str) -> BoltResult<Vec<u8>> {
    crate::plan::extended_types::unhex(s)
        .ok_or_else(|| BoltError::Other(format!("extended types: '{s}' is not hex")))
}

fn from_hex(col: &StringArray, t: &A) -> BoltResult<ArrayRef> {
    let bytes: Vec<Option<Vec<u8>>> = col
        .iter()
        .map(|v| v.map(unhex).transpose())
        .collect::<BoltResult<_>>()?;
    Ok(match t {
        A::Binary => Arc::new(BinaryArray::from_iter(bytes)),
        A::LargeBinary => Arc::new(LargeBinaryArray::from_iter(bytes)),
        A::BinaryView => Arc::new(BinaryViewArray::from_iter(bytes)),
        A::FixedSizeBinary(n) => Arc::new(
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(bytes.into_iter(), *n)
                .map_err(arrow_err)?,
        ),
        other => return Err(BoltError::Type(format!("{other} is not a binary type"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::types::Int32Type;
    use arrow_array::{ListArray, Time32SecondArray};
    use arrow_buffer::{NullBuffer, ScalarBuffer};

    fn batch(cols: Vec<(&str, ArrayRef)>) -> RecordBatch {
        let fields: Vec<ArrowField> = cols
            .iter()
            .map(|(n, c)| ArrowField::new(*n, c.data_type().clone(), true))
            .collect();
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new(fields)),
            cols.into_iter().map(|(_, c)| c).collect(),
        )
        .unwrap()
    }

    fn logical(b: &RecordBatch) -> Schema {
        crate::exec::engine_support::arrow_schema_to_plan_schema(b.schema().as_ref()).unwrap()
    }

    fn list(rows: &[Option<Vec<Option<i32>>>]) -> ArrayRef {
        Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(
            rows.to_vec(),
        ))
    }

    #[test]
    fn ingest_then_restore_round_trips_every_family() {
        let tm = Time32SecondArray::new(
            ScalarBuffer::from(vec![10, 99, 30]),
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let b = BinaryArray::from_iter(vec![Some(vec![0u8, 0xff]), None, Some(vec![])]);
        let fb = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
            vec![Some([1u8, 2]), None, Some([3, 4])].into_iter(),
            2,
        )
        .unwrap();
        let l = list(&[Some(vec![Some(1)]), None, Some(vec![])]);
        let input = batch(vec![
            ("tm", Arc::new(tm)),
            ("b", Arc::new(b)),
            ("fb", Arc::new(fb)),
            ("l", l),
        ]);
        let mut store = ExtendedStore::default();
        let (stored, staged) = store.ingest("t", &input, false).unwrap();
        assert_eq!(stored.column(0).data_type(), &A::Int32);
        assert_eq!(stored.column(1).data_type(), &A::Utf8);
        assert_eq!(stored.column(1).as_string::<i32>().value(0), "00ff");
        assert_eq!(stored.column(3).data_type(), &A::Int64);
        assert_eq!(stored.column(3).null_count(), 1);
        store.commit(staged);
        assert!(store.tables.contains_key("t"));
        let back = store.restore(stored, &logical(&input)).unwrap();
        for c in 0..input.num_columns() {
            assert_eq!(
                back.column(c).as_ref(),
                input.column(c).as_ref(),
                "column {c}"
            );
        }
    }

    #[test]
    fn appended_rows_get_handles_after_the_existing_ones() {
        let mut store = ExtendedStore::default();
        let a = batch(vec![(
            "l",
            list(&[Some(vec![Some(1)]), Some(vec![Some(2)])]),
        )]);
        let b = batch(vec![("l", list(&[None, Some(vec![Some(3), Some(4)])]))]);
        let (sa, st) = store.ingest("t", &a, false).unwrap();
        store.commit(st);
        let (sb, st) = store.ingest("t", &b, true).unwrap();
        store.commit(st);
        let both = arrow::compute::concat_batches(&sa.schema(), &[sa, sb]).unwrap();
        // Reverse the rows, as a sort would.
        let idx = arrow_array::UInt32Array::from(vec![3, 2, 1, 0]);
        let rev = arrow::compute::take_record_batch(&both, &idx).unwrap();
        let got = store.restore(rev, &logical(&a)).unwrap();
        let want = list(&[
            Some(vec![Some(3), Some(4)]),
            None,
            Some(vec![Some(2)]),
            Some(vec![Some(1)]),
        ]);
        assert_eq!(got.column(0).as_ref(), want.as_ref());
        // A mismatched append is refused.
        let c = batch(vec![(
            "l",
            Arc::new(arrow_array::Int32Array::from(vec![1])) as ArrayRef,
        )]);
        assert!(store.ingest("t", &c, true).is_err());
        // Re-registering drops the old values.
        let (_, st) = store.ingest("t", &a, false).unwrap();
        store.commit(st);
        assert_eq!(store.sources.len(), 1);
    }

    #[test]
    fn plain_tables_are_not_tracked() {
        let store = ExtendedStore::default();
        let input = batch(vec![(
            "s",
            Arc::new(arrow_array::LargeStringArray::from(vec!["a"])) as ArrayRef,
        )]);
        let (stored, staged) = store.ingest("t", &input, false).unwrap();
        assert_eq!(stored.column(0).data_type(), &A::Utf8);
        assert!(staged.entry.is_none() && staged.logical.is_empty());
    }
}
