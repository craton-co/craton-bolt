// SPDX-License-Identifier: Apache-2.0
//! Table and column statistics.
//!
//! Statistics serve two consumers:
//!
//! * **Scan pruning.** Per-chunk (row group / batch) min / max / null counts
//!   are zone maps: [`crate::storage::scan`] skips a chunk whose ranges prove
//!   a predicate cannot match.
//! * **The optimizer.** Table-level row counts feed the join-reorder row
//!   estimator without scanning the table.
//!
//! Values are carried as plan [`Literal`]s, so a statistic compares directly
//! against a predicate constant (see [`compare_literals`]).

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int32Type, Int64Type,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType,
};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, TimeUnit as ArrowTimeUnit};

use crate::plan::logical_plan::{Literal, TimeUnit};

/// Statistics of one column over a chunk or a whole table.
///
/// Every field is optional: `None` means "unknown", which pruning and
/// estimation treat conservatively.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "storage", derive(serde::Serialize, serde::Deserialize))]
pub struct ColumnStatistics {
    /// Number of NULL values.
    pub null_count: Option<u64>,
    /// Smallest non-NULL value.
    #[cfg_attr(feature = "storage", serde(with = "literal_serde"))]
    pub min: Option<Literal>,
    /// Largest non-NULL value.
    #[cfg_attr(feature = "storage", serde(with = "literal_serde"))]
    pub max: Option<Literal>,
    /// Estimated number of distinct non-NULL values (HyperLogLog, ~2% error).
    pub distinct_estimate: Option<u64>,
}

/// Statistics of a table (or of one chunk of it).
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "storage", derive(serde::Serialize, serde::Deserialize))]
pub struct TableStatistics {
    /// Number of rows.
    pub row_count: u64,
    /// Approximate in-memory (or on-disk, for file-backed chunks) size.
    pub byte_size: u64,
    /// One entry per schema column, in schema order.
    pub columns: Vec<ColumnStatistics>,
}

impl TableStatistics {
    /// Compute exact min / max / null counts, and HyperLogLog distinct
    /// estimates, over `batches`. `columns` has one entry per batch column.
    pub fn from_batches(batches: &[RecordBatch]) -> Self {
        let n_cols = batches.first().map_or(0, RecordBatch::num_columns);
        let mut acc: Vec<ColumnAccumulator> =
            (0..n_cols).map(|_| ColumnAccumulator::new()).collect();
        let mut row_count = 0u64;
        let mut byte_size = 0u64;
        for batch in batches {
            row_count += batch.num_rows() as u64;
            byte_size += crate::exec::streaming::estimate_batch_bytes(batch) as u64;
            for (a, col) in acc.iter_mut().zip(batch.columns()) {
                a.update(col);
            }
        }
        TableStatistics {
            row_count,
            byte_size,
            columns: acc.into_iter().map(ColumnAccumulator::finish).collect(),
        }
    }

    /// Fold another chunk's statistics into `self` (min of mins, max of
    /// maxes, summed counts). Distinct estimates are not mergeable from
    /// their finished values, so the merge keeps the larger one as a lower
    /// bound — [`TableStatistics::from_batches`] (`ANALYZE`) recomputes them
    /// exactly-mergeably over the data.
    pub fn merge(&mut self, other: &TableStatistics) {
        if self.columns.is_empty() && self.row_count == 0 {
            *self = other.clone();
            return;
        }
        self.row_count += other.row_count;
        self.byte_size += other.byte_size;
        for (a, b) in self.columns.iter_mut().zip(&other.columns) {
            a.null_count = a.null_count.zip(b.null_count).map(|(x, y)| x + y);
            a.min = merge_bound(a.min.take(), b.min.clone(), Ordering::Less);
            a.max = merge_bound(a.max.take(), b.max.clone(), Ordering::Greater);
            a.distinct_estimate = a
                .distinct_estimate
                .zip(b.distinct_estimate)
                .map(|(x, y)| x.max(y));
        }
    }
}

/// Keep the bound that wins `want` (`Less` for min, `Greater` for max). An
/// unknown side makes the merged bound unknown; incomparable values too.
fn merge_bound(a: Option<Literal>, b: Option<Literal>, want: Ordering) -> Option<Literal> {
    match (a, b) {
        (Some(a), Some(b)) => match compare_literals(&b, &a) {
            Some(o) if o == want => Some(b),
            Some(_) => Some(a),
            None => None,
        },
        _ => None,
    }
}

/// Order two literals of compatible types, or `None` when they are not
/// comparable (different type families, NaN, `Null`).
///
/// Integers compare exactly (as `i128`); an integer against a float, and
/// float against float, compare as `f64`. Decimals are rescaled to a common
/// scale. Timestamps compare only at equal units.
pub fn compare_literals(a: &Literal, b: &Literal) -> Option<Ordering> {
    use Literal as L;
    fn int(l: &Literal) -> Option<i128> {
        match l {
            L::Int32(v) => Some(*v as i128),
            L::Int64(v) => Some(*v as i128),
            _ => None,
        }
    }
    fn float(l: &Literal) -> Option<f64> {
        match l {
            L::Int32(v) => Some(*v as f64),
            L::Int64(v) => Some(*v as f64),
            L::Float32(v) => Some(*v as f64),
            L::Float64(v) => Some(*v),
            _ => None,
        }
    }
    if let (Some(x), Some(y)) = (int(a), int(b)) {
        return Some(x.cmp(&y));
    }
    if let (Some(x), Some(y)) = (float(a), float(b)) {
        return x.partial_cmp(&y);
    }
    match (a, b) {
        (L::Bool(x), L::Bool(y)) => Some(x.cmp(y)),
        (L::Utf8(x), L::Utf8(y)) => Some(x.as_bytes().cmp(y.as_bytes())),
        (L::Date32(x), L::Date32(y)) => Some(x.cmp(y)),
        (L::Timestamp(x, ux, _), L::Timestamp(y, uy, _)) if ux == uy => Some(x.cmp(y)),
        (L::Decimal128(x, _, sx), L::Decimal128(y, _, sy)) => {
            let (x, y) = rescale_pair(*x, *sx, *y, *sy)?;
            Some(x.cmp(&y))
        }
        (L::Decimal128(x, _, s), other) | (other, L::Decimal128(x, _, s))
            if int(other).is_some() =>
        {
            let v = int(other)?;
            let (d, i) = rescale_pair(*x, *s, v, 0)?;
            let ord = d.cmp(&i);
            Some(if matches!(a, L::Decimal128(..)) {
                ord
            } else {
                ord.reverse()
            })
        }
        _ => None,
    }
}

fn rescale_pair(x: i128, sx: i8, y: i128, sy: i8) -> Option<(i128, i128)> {
    let up = |v: i128, by: i8| -> Option<i128> { v.checked_mul(10i128.checked_pow(by as u32)?) };
    match sx.cmp(&sy) {
        Ordering::Equal => Some((x, y)),
        Ordering::Less => Some((up(x, sy - sx)?, y)),
        Ordering::Greater => Some((x, up(y, sx - sy)?)),
    }
}

/// Read element `i` of `array` as a plan literal (`None` for NULL or an
/// unsupported type).
pub fn array_value(array: &dyn Array, i: usize) -> Option<Literal> {
    if array.is_null(i) {
        return None;
    }
    Some(match array.data_type() {
        ArrowDataType::Boolean => Literal::Bool(array.as_boolean().value(i)),
        ArrowDataType::Int32 => Literal::Int32(array.as_primitive::<Int32Type>().value(i)),
        ArrowDataType::Int64 => Literal::Int64(array.as_primitive::<Int64Type>().value(i)),
        ArrowDataType::Float32 => Literal::Float32(array.as_primitive::<Float32Type>().value(i)),
        ArrowDataType::Float64 => Literal::Float64(array.as_primitive::<Float64Type>().value(i)),
        ArrowDataType::Utf8 => Literal::Utf8(array.as_string::<i32>().value(i).to_string()),
        ArrowDataType::LargeUtf8 => Literal::Utf8(array.as_string::<i64>().value(i).to_string()),
        ArrowDataType::Date32 => Literal::Date32(array.as_primitive::<Date32Type>().value(i)),
        ArrowDataType::Decimal128(p, s) => {
            Literal::Decimal128(array.as_primitive::<Decimal128Type>().value(i), *p, *s)
        }
        ArrowDataType::Timestamp(unit, tz) => {
            let tz = tz
                .as_deref()
                .map(crate::plan::logical_plan::intern_timezone);
            let (v, u) = match unit {
                ArrowTimeUnit::Second => (
                    array.as_primitive::<TimestampSecondType>().value(i),
                    TimeUnit::Second,
                ),
                ArrowTimeUnit::Millisecond => (
                    array.as_primitive::<TimestampMillisecondType>().value(i),
                    TimeUnit::Millisecond,
                ),
                ArrowTimeUnit::Microsecond => (
                    array.as_primitive::<TimestampMicrosecondType>().value(i),
                    TimeUnit::Microsecond,
                ),
                ArrowTimeUnit::Nanosecond => (
                    array.as_primitive::<TimestampNanosecondType>().value(i),
                    TimeUnit::Nanosecond,
                ),
            };
            Literal::Timestamp(v, u, tz)
        }
        ArrowDataType::Dictionary(_, _) => {
            let dict = array.as_any_dictionary();
            let key = dict.normalized_keys()[i];
            return array_value(dict.values().as_ref(), key);
        }
        _ => return None,
    })
}

/// Streaming min / max / null-count / distinct accumulator for one column.
struct ColumnAccumulator {
    nulls: u64,
    min: Option<Literal>,
    max: Option<Literal>,
    supported: bool,
    hll: Hll,
}

impl ColumnAccumulator {
    fn new() -> Self {
        ColumnAccumulator {
            nulls: 0,
            min: None,
            max: None,
            supported: true,
            hll: Hll::new(),
        }
    }

    fn update(&mut self, col: &ArrayRef) {
        self.nulls += col.null_count() as u64;
        // Value-wise scan: one pass computes the bounds and feeds the sketch.
        // Statistics run at ANALYZE / ingest time, not on the query path.
        for i in 0..col.len() {
            let Some(v) = array_value(col.as_ref(), i) else {
                if !col.is_null(i) {
                    self.supported = false;
                    return;
                }
                continue;
            };
            if matches!(&v, Literal::Float32(f) if f.is_nan())
                || matches!(&v, Literal::Float64(f) if f.is_nan())
            {
                // NaN carries no ordering; keep it out of the bounds (IEEE
                // comparisons against NaN are false, so pruning stays sound).
                self.hll.insert(&v);
                continue;
            }
            self.hll.insert(&v);
            if self
                .min
                .as_ref()
                .is_none_or(|m| compare_literals(&v, m) == Some(Ordering::Less))
            {
                self.min = Some(v.clone());
            }
            if self
                .max
                .as_ref()
                .is_none_or(|m| compare_literals(&v, m) == Some(Ordering::Greater))
            {
                self.max = Some(v);
            }
        }
    }

    fn finish(self) -> ColumnStatistics {
        if !self.supported {
            return ColumnStatistics {
                null_count: Some(self.nulls),
                ..Default::default()
            };
        }
        ColumnStatistics {
            null_count: Some(self.nulls),
            min: self.min,
            max: self.max,
            distinct_estimate: Some(self.hll.estimate()),
        }
    }
}

/// HyperLogLog distinct-count sketch (precision 12: 4096 registers, ~1.6%
/// standard error). Mergeable, so per-partition sketches combine exactly.
#[derive(Debug, Clone)]
pub struct Hll {
    registers: Vec<u8>,
}

const HLL_P: u32 = 12;
const HLL_M: usize = 1 << HLL_P;

impl Default for Hll {
    fn default() -> Self {
        Self::new()
    }
}

impl Hll {
    /// Empty sketch.
    pub fn new() -> Self {
        Hll {
            registers: vec![0; HLL_M],
        }
    }

    /// Add one value.
    pub fn insert(&mut self, v: &Literal) {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        hash_literal(v, &mut h);
        self.insert_hash(h.finish());
    }

    fn insert_hash(&mut self, hash: u64) {
        let idx = (hash >> (64 - HLL_P)) as usize;
        let rest = (hash << HLL_P) | (1 << (HLL_P - 1));
        let rank = rest.leading_zeros() as u8 + 1;
        if rank > self.registers[idx] {
            self.registers[idx] = rank;
        }
    }

    /// Union another sketch into this one.
    pub fn merge(&mut self, other: &Hll) {
        for (a, b) in self.registers.iter_mut().zip(&other.registers) {
            *a = (*a).max(*b);
        }
    }

    /// Current distinct-count estimate.
    pub fn estimate(&self) -> u64 {
        let m = HLL_M as f64;
        let alpha = 0.7213 / (1.0 + 1.079 / m);
        let sum: f64 = self.registers.iter().map(|&r| 2f64.powi(-(r as i32))).sum();
        let raw = alpha * m * m / sum;
        let zeros = self.registers.iter().filter(|&&r| r == 0).count();
        let est = if raw <= 2.5 * m && zeros > 0 {
            // Small-range correction: linear counting.
            m * (m / zeros as f64).ln()
        } else {
            raw
        };
        est.round() as u64
    }
}

/// Hash a literal so equal SQL values hash equally across integer widths.
fn hash_literal<H: Hasher>(v: &Literal, h: &mut H) {
    match v {
        Literal::Int32(x) => (*x as i64).hash(h),
        Literal::Int64(x) => x.hash(h),
        Literal::Float32(x) => (*x as f64).to_bits().hash(h),
        Literal::Float64(x) => x.to_bits().hash(h),
        Literal::Bool(x) => x.hash(h),
        Literal::Utf8(s) => s.hash(h),
        Literal::Date32(x) => x.hash(h),
        Literal::Decimal128(x, _, s) => (x, s).hash(h),
        Literal::Timestamp(x, _, _) => x.hash(h),
        Literal::Null => 0u8.hash(h),
    }
}

/// Serde adapter for `Option<Literal>` statistics bounds: a tagged
/// `{"t": <type>, "v": <value>}` object, so the persisted catalog does not
/// depend on the plan enum's in-memory layout.
#[cfg(feature = "storage")]
pub(crate) mod literal_serde {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct Tagged {
        t: String,
        v: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tz: Option<String>,
    }

    pub(crate) fn to_tagged(l: &Literal) -> Option<serde_json::Value> {
        use serde_json::json;
        let (t, v, tz): (&str, serde_json::Value, Option<String>) = match l {
            Literal::Null => return None,
            Literal::Bool(b) => ("bool", json!(b), None),
            Literal::Int32(x) => ("i32", json!(x), None),
            Literal::Int64(x) => ("i64", json!(x.to_string()), None),
            Literal::Float32(x) => ("f32", json!(x), None),
            Literal::Float64(x) => ("f64", json!(x), None),
            Literal::Utf8(s) => ("utf8", json!(s), None),
            Literal::Date32(x) => ("date32", json!(x), None),
            Literal::Decimal128(x, p, s) => ("decimal", json!([x.to_string(), p, s]), None),
            Literal::Timestamp(x, u, tz) => {
                let unit = match u {
                    TimeUnit::Second => "ts_s",
                    TimeUnit::Millisecond => "ts_ms",
                    TimeUnit::Microsecond => "ts_us",
                    TimeUnit::Nanosecond => "ts_ns",
                };
                (unit, json!(x.to_string()), tz.map(str::to_string))
            }
        };
        serde_json::to_value(Tagged { t: t.into(), v, tz }).ok()
    }

    pub(crate) fn from_tagged(value: serde_json::Value) -> Option<Literal> {
        let tagged: Tagged = serde_json::from_value(value).ok()?;
        let s = |v: &serde_json::Value| v.as_str().map(str::to_string);
        let tz = tagged
            .tz
            .as_deref()
            .map(crate::plan::logical_plan::intern_timezone);
        let ts = |u| Some(Literal::Timestamp(s(&tagged.v)?.parse().ok()?, u, tz));
        match tagged.t.as_str() {
            "bool" => tagged.v.as_bool().map(Literal::Bool),
            "i32" => tagged.v.as_i64().map(|x| Literal::Int32(x as i32)),
            "i64" => s(&tagged.v)?.parse().ok().map(Literal::Int64),
            "f32" => tagged.v.as_f64().map(|x| Literal::Float32(x as f32)),
            "f64" => tagged.v.as_f64().map(Literal::Float64),
            "utf8" => s(&tagged.v).map(Literal::Utf8),
            "date32" => tagged.v.as_i64().map(|x| Literal::Date32(x as i32)),
            "decimal" => {
                let a = tagged.v.as_array()?;
                Some(Literal::Decimal128(
                    s(a.first()?)?.parse().ok()?,
                    a.get(1)?.as_u64()? as u8,
                    a.get(2)?.as_i64()? as i8,
                ))
            }
            "ts_s" => ts(TimeUnit::Second),
            "ts_ms" => ts(TimeUnit::Millisecond),
            "ts_us" => ts(TimeUnit::Microsecond),
            "ts_ns" => ts(TimeUnit::Nanosecond),
            _ => None,
        }
    }

    pub(crate) fn serialize<S: Serializer>(v: &Option<Literal>, s: S) -> Result<S::Ok, S::Error> {
        v.as_ref().and_then(to_tagged).serialize(s)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Literal>, D::Error> {
        let v: Option<serde_json::Value> = Option::deserialize(d)?;
        Ok(v.and_then(from_tagged))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, Int32Array, StringArray};
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;

    fn batch(ints: Vec<Option<i32>>, strs: Vec<&str>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", ArrowDataType::Int32, true),
            Field::new("s", ArrowDataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ints)),
                Arc::new(StringArray::from(strs)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn from_batches_computes_bounds_nulls_and_distinct() {
        let b1 = batch(vec![Some(5), None, Some(-3)], vec!["m", "a", "z"]);
        let b2 = batch(vec![Some(9), Some(5)], vec!["b", "a"]);
        let s = TableStatistics::from_batches(&[b1, b2]);
        assert_eq!(s.row_count, 5);
        let i = &s.columns[0];
        assert_eq!(i.null_count, Some(1));
        assert_eq!(i.min, Some(Literal::Int32(-3)));
        assert_eq!(i.max, Some(Literal::Int32(9)));
        assert_eq!(i.distinct_estimate, Some(3));
        let st = &s.columns[1];
        assert_eq!(st.min, Some(Literal::Utf8("a".into())));
        assert_eq!(st.max, Some(Literal::Utf8("z".into())));
        assert_eq!(st.distinct_estimate, Some(4));
    }

    #[test]
    fn merge_combines_chunks() {
        let mut a = TableStatistics::from_batches(&[batch(vec![Some(1)], vec!["x"])]);
        let b = TableStatistics::from_batches(&[batch(vec![Some(7), None], vec!["a", "b"])]);
        a.merge(&b);
        assert_eq!(a.row_count, 3);
        assert_eq!(a.columns[0].min, Some(Literal::Int32(1)));
        assert_eq!(a.columns[0].max, Some(Literal::Int32(7)));
        assert_eq!(a.columns[0].null_count, Some(1));
        assert_eq!(a.columns[1].min, Some(Literal::Utf8("a".into())));
    }

    #[test]
    fn nan_stays_out_of_bounds() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "f",
            ArrowDataType::Float64,
            false,
        )]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1.0, f64::NAN, 4.0]))],
        )
        .unwrap();
        let s = TableStatistics::from_batches(&[b]);
        assert_eq!(s.columns[0].min, Some(Literal::Float64(1.0)));
        assert_eq!(s.columns[0].max, Some(Literal::Float64(4.0)));
    }

    #[test]
    fn literal_ordering_across_types() {
        use Literal as L;
        assert_eq!(
            compare_literals(&L::Int32(3), &L::Int64(3)),
            Some(Ordering::Equal)
        );
        assert_eq!(
            compare_literals(&L::Int64(i64::MAX), &L::Int64(i64::MAX - 1)),
            Some(Ordering::Greater)
        );
        assert_eq!(
            compare_literals(&L::Int32(2), &L::Float64(2.5)),
            Some(Ordering::Less)
        );
        assert_eq!(
            compare_literals(&L::Decimal128(150, 10, 2), &L::Int64(1)),
            Some(Ordering::Greater)
        );
        assert_eq!(
            compare_literals(&L::Int64(2), &L::Decimal128(150, 10, 2)),
            Some(Ordering::Greater)
        );
        assert_eq!(
            compare_literals(&L::Decimal128(150, 10, 2), &L::Decimal128(15, 10, 1)),
            Some(Ordering::Equal)
        );
        assert_eq!(compare_literals(&L::Utf8("a".into()), &L::Int32(1)), None);
        assert_eq!(
            compare_literals(&L::Float64(f64::NAN), &L::Float64(1.0)),
            None
        );
    }

    #[test]
    fn hll_estimates_and_merges() {
        let mut a = Hll::new();
        let mut b = Hll::new();
        for i in 0..50_000i64 {
            a.insert(&Literal::Int64(i));
            b.insert(&Literal::Int64(i + 25_000));
        }
        let ea = a.estimate() as f64;
        assert!((ea - 50_000.0).abs() / 50_000.0 < 0.05, "{ea}");
        a.merge(&b);
        let eu = a.estimate() as f64;
        assert!((eu - 75_000.0).abs() / 75_000.0 < 0.05, "{eu}");
    }

    #[cfg(feature = "storage")]
    #[test]
    fn statistics_round_trip_through_json() {
        let s = TableStatistics {
            row_count: 3,
            byte_size: 10,
            columns: vec![
                ColumnStatistics {
                    null_count: Some(1),
                    min: Some(Literal::Int64(i64::MIN)),
                    max: Some(Literal::Utf8("q\"x".into())),
                    distinct_estimate: Some(2),
                },
                ColumnStatistics {
                    null_count: None,
                    min: Some(Literal::Decimal128(-12345678901234567890, 30, 4)),
                    max: Some(Literal::Timestamp(7, TimeUnit::Microsecond, Some("UTC"))),
                    distinct_estimate: None,
                },
            ],
        };
        let text = serde_json::to_string(&s).unwrap();
        let back: TableStatistics = serde_json::from_str(&text).unwrap();
        assert_eq!(back, s);
    }
}
