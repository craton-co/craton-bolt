// SPDX-License-Identifier: Apache-2.0

//! Morsel / chunk streaming abstractions for bounded, larger-than-VRAM
//! execution.
//!
//! Today the engine materialises a table's `Vec<RecordBatch>` into one
//! concatenated `RecordBatch` and uploads the whole thing to the device in a
//! single shot (see [`crate::exec::engine::Engine::materialize_table`] and
//! [`crate::exec::gpu_table::GpuTable::from_record_batch`]). That caps the
//! engine's working set at VRAM size: a table that does not fit on the device
//! cannot be queried at all.
//!
//! This module is the host-side scaffolding for the bounded-chunk
//! alternative. It provides three things, all pure host logic that compiles
//! and is unit-testable under `cuda-stub` (no device, no CUDA calls):
//!
//! 1. [`BatchStream`] — a re-iterable morsel iterator over a slice of
//!    `RecordBatch`es. It yields fixed-row-count *morsels* (each itself a
//!    `RecordBatch`, produced by zero-copy Arrow slicing) so an executor can
//!    process a table chunk-by-chunk instead of all-at-once. Morsel
//!    boundaries are exact: every morsel except possibly the last has exactly
//!    `morsel_rows` rows; the last carries the remainder.
//!
//! 2. [`TableSource`] — the table-representation enum that lets the engine
//!    store either an eagerly-materialised `Vec<RecordBatch>`
//!    ([`TableSource::Materialized`]) or a lazily-drained producer
//!    ([`TableSource::Streaming`]). The streaming variant defers pulling the
//!    producer's batches until the table is first queried.
//!
//! 3. [`MorselPlan`] / [`plan_upload`] — the spill/budget hook. Given a
//!    table's estimated byte size and the engine's
//!    [`memory_budget`](crate::exec::engine::EngineBuilder::memory_budget),
//!    it decides whether a whole-table upload fits or whether the table must
//!    be processed in morsels, and computes a morsel row count that keeps each
//!    chunk under budget. Intermediates are conceptually kept in host-pinned
//!    memory; the device-pinned allocation is left as a `cuda`-feature TODO
//!    (see [`PinnedBudget`]).

use arrow_array::{Array, RecordBatch};

use crate::error::{BoltError, BoltResult};

/// A batch producer that can be re-iterated.
///
/// `register_table_stream` accepts an `IntoIterator` of `BoltResult<RecordBatch>`,
/// which is single-use. To store a table lazily we need a source that can be
/// *replayed* — once to validate/materialise on first query, and again if the
/// engine ever re-derives the table. A boxed factory closure is the simplest
/// re-iterable shape that does not constrain the caller to a concrete iterator
/// type.
///
/// The factory returns a fresh boxed iterator each time it is called. Producer
/// errors surface as `Err` items from that iterator, exactly like the eager
/// `register_table_stream` contract.
pub type BatchProducer =
    Box<dyn Fn() -> Box<dyn Iterator<Item = BoltResult<RecordBatch>>> + Send + Sync>;

/// How a registered table's host-side data is stored.
///
/// The engine's canonical read paths (`materialize_table`, the provider's
/// null-count probes) operate on a `Vec<RecordBatch>`. `TableSource` lets the
/// engine keep that eager representation for tables registered through
/// `register_table` / `register_batch` while *additionally* supporting a
/// lazily-drained producer for tables registered through the streaming path.
///
/// A `Streaming` source is collapsed to `Materialized` the first time the
/// table is queried (see
/// [`crate::exec::engine::Engine`]'s streaming-materialisation hook). Keeping
/// the variant around after collapse is harmless; the engine treats a
/// collapsed source as materialised.
pub enum TableSource {
    /// Eagerly materialised batches — the legacy representation. Every batch
    /// already lives in host memory.
    Materialized(Vec<RecordBatch>),
    /// A lazily-drained producer. The batches are NOT pulled until the table
    /// is first queried; the producer is replayable (see [`BatchProducer`]).
    Streaming(BatchProducer),
}

impl std::fmt::Debug for TableSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TableSource::Materialized(b) => f
                .debug_struct("Materialized")
                .field("batches", &b.len())
                .finish(),
            TableSource::Streaming(_) => f.debug_struct("Streaming").finish_non_exhaustive(),
        }
    }
}

impl TableSource {
    /// Drain the source into an owned `Vec<RecordBatch>`.
    ///
    /// For [`TableSource::Materialized`] this clones the batch vector (Arrow
    /// arrays are `Arc`-backed, so the clone is cheap — pointer bumps, no
    /// column copies). For [`TableSource::Streaming`] this invokes the
    /// producer factory and collects every yielded batch, propagating the
    /// first producer error.
    ///
    /// An empty stream is rejected with a `Plan` error — a registered table
    /// must contain at least one batch, matching the eager
    /// `register_table_stream` contract.
    pub fn drain_to_batches(&self, table_name: &str) -> BoltResult<Vec<RecordBatch>> {
        match self {
            TableSource::Materialized(batches) => {
                if batches.is_empty() {
                    return Err(BoltError::Plan(format!(
                        "table '{table_name}' is registered but contains zero batches"
                    )));
                }
                Ok(batches.clone())
            }
            TableSource::Streaming(producer) => {
                let mut out = Vec::new();
                for (i, item) in producer().enumerate() {
                    match item {
                        Ok(b) => out.push(b),
                        Err(e) => return Err(e),
                    }
                    let _ = i;
                }
                if out.is_empty() {
                    return Err(BoltError::Plan(format!(
                        "streaming source for table '{table_name}' yielded zero \
                         batches — a registered table must contain at least one batch"
                    )));
                }
                Ok(out)
            }
        }
    }

    /// `true` if this source still needs to be drained from a producer.
    pub fn is_streaming(&self) -> bool {
        matches!(self, TableSource::Streaming(_))
    }
}

/// Estimate the host-memory footprint of a single `RecordBatch` in bytes.
///
/// Sums [`Array::get_array_memory_size`] across columns. This includes Arrow's
/// buffer overhead (validity bitmaps, offset buffers for variable-width types)
/// so it is a conservative *upper* bound on the bytes that would be uploaded —
/// which is what we want for a budget guard that errs toward smaller morsels.
pub fn estimate_batch_bytes(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|c| c.get_array_memory_size())
        .fold(0usize, |acc, n| acc.saturating_add(n))
}

/// Estimate the total host-memory footprint of a slice of batches.
pub fn estimate_batches_bytes(batches: &[RecordBatch]) -> usize {
    batches
        .iter()
        .map(estimate_batch_bytes)
        .fold(0usize, |acc, n| acc.saturating_add(n))
}

/// The decision produced by [`plan_upload`]: either the table fits the budget
/// and can be uploaded whole, or it must be processed in morsels of a bounded
/// row count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorselPlan {
    /// The whole table fits under the memory budget (or no budget is set).
    /// Upload it in one shot, as today.
    Whole,
    /// The table exceeds the budget. Process it in morsels of `morsel_rows`
    /// rows each so every chunk's working set stays under budget. The last
    /// morsel carries the remainder.
    Morsels {
        /// Target rows per morsel. Always `>= 1`.
        morsel_rows: usize,
    },
}

impl MorselPlan {
    /// The morsel row count if this plan calls for chunking, else `None`.
    pub fn morsel_rows(&self) -> Option<usize> {
        match self {
            MorselPlan::Whole => None,
            MorselPlan::Morsels { morsel_rows } => Some(*morsel_rows),
        }
    }
}

/// Decide how to upload a table given its size and the engine's budget.
///
/// * `total_bytes` — estimated host/device footprint of the whole table (see
///   [`estimate_batches_bytes`]).
/// * `total_rows` — the table's row count (used to derive a morsel size that
///   keeps each chunk's byte footprint under budget).
/// * `budget_bytes` — the engine's soft memory budget; `None` means uncapped,
///   in which case the answer is always [`MorselPlan::Whole`].
///
/// When the table exceeds the budget we compute the largest morsel row count
/// whose estimated byte size still fits: `morsel_rows = budget / bytes_per_row`,
/// clamped to at least one row (a single oversized row cannot be split — it is
/// processed alone and the caller's device path must cope, which it can because
/// one-row working sets are the smallest possible). A zero-row or zero-byte
/// table trivially fits.
pub fn plan_upload(
    total_bytes: usize,
    total_rows: usize,
    budget_bytes: Option<usize>,
) -> MorselPlan {
    let budget = match budget_bytes {
        None => return MorselPlan::Whole,
        Some(b) => b,
    };
    if total_bytes <= budget || total_rows == 0 {
        return MorselPlan::Whole;
    }
    // bytes_per_row, rounded up so we never under-estimate a morsel's cost.
    let bytes_per_row = total_bytes.div_ceil(total_rows).max(1);
    let rows_that_fit = budget / bytes_per_row;
    let morsel_rows = rows_that_fit.max(1);
    MorselPlan::Morsels { morsel_rows }
}

/// A re-iterable morsel iterator over a table's batches.
///
/// Borrows a slice of `RecordBatch`es and, on iteration, yields fixed-row-count
/// morsels via zero-copy Arrow slicing ([`RecordBatch::slice`]). Morsels never
/// span a batch boundary — a morsel is always a contiguous slice of a single
/// source batch — so a 10-row batch with `morsel_rows = 4` yields morsels of
/// 4, 4, 2 rows.
///
/// Re-iterable: call [`BatchStream::morsels`] as many times as needed; each
/// call returns a fresh [`Morsels`] iterator over the same borrowed batches.
///
/// Empty source batches (zero rows) are skipped — they would otherwise yield a
/// spurious zero-row morsel.
pub struct BatchStream<'a> {
    batches: &'a [RecordBatch],
    morsel_rows: usize,
}

impl<'a> BatchStream<'a> {
    /// Build a morsel stream over `batches`, yielding morsels of at most
    /// `morsel_rows` rows.
    ///
    /// # Errors
    /// Returns a `Plan` error if `morsel_rows` is zero — a zero-row morsel
    /// would make the iterator spin forever without consuming input.
    pub fn new(batches: &'a [RecordBatch], morsel_rows: usize) -> BoltResult<Self> {
        if morsel_rows == 0 {
            return Err(BoltError::Plan(
                "BatchStream: morsel_rows must be >= 1 (zero-row morsels are not allowed)"
                    .to_string(),
            ));
        }
        Ok(Self {
            batches,
            morsel_rows,
        })
    }

    /// The configured morsel row count.
    pub fn morsel_rows(&self) -> usize {
        self.morsel_rows
    }

    /// Total rows across all source batches.
    pub fn total_rows(&self) -> usize {
        self.batches
            .iter()
            .map(RecordBatch::num_rows)
            .fold(0usize, |a, n| a.saturating_add(n))
    }

    /// Number of morsels this stream will yield (ceil over each non-empty
    /// batch, summed). Useful for pre-sizing collectors and for tests.
    pub fn num_morsels(&self) -> usize {
        self.batches
            .iter()
            .map(|b| {
                let n = b.num_rows();
                if n == 0 {
                    0
                } else {
                    n.div_ceil(self.morsel_rows)
                }
            })
            .fold(0usize, |a, n| a.saturating_add(n))
    }

    /// Fresh iterator over the morsels. Re-iterable: the underlying batches
    /// are only borrowed, so this may be called repeatedly.
    pub fn morsels(&self) -> Morsels<'a> {
        Morsels {
            batches: self.batches,
            morsel_rows: self.morsel_rows,
            batch_idx: 0,
            row_offset: 0,
        }
    }
}

/// Iterator returned by [`BatchStream::morsels`]. Yields one `RecordBatch`
/// morsel per `next`, walking batches in order and slicing each into
/// `morsel_rows`-sized chunks.
pub struct Morsels<'a> {
    batches: &'a [RecordBatch],
    morsel_rows: usize,
    /// Index of the batch currently being sliced.
    batch_idx: usize,
    /// Row offset within `batches[batch_idx]` of the next morsel.
    row_offset: usize,
}

impl Iterator for Morsels<'_> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let batch = self.batches.get(self.batch_idx)?;
            let n = batch.num_rows();
            if self.row_offset >= n {
                // Exhausted this batch (or it was empty) — advance.
                self.batch_idx += 1;
                self.row_offset = 0;
                continue;
            }
            let remaining = n - self.row_offset;
            let take = remaining.min(self.morsel_rows);
            // Zero-copy slice: shares the underlying Arrow buffers.
            let morsel = batch.slice(self.row_offset, take);
            self.row_offset += take;
            return Some(morsel);
        }
    }
}

/// Host-side budget bookkeeping for morsel execution.
///
/// Tracks how many bytes of intermediate results are "live" while a table is
/// processed morsel-by-morsel, so the orchestrator can keep the working set
/// under the engine's budget. Intermediates are conceptually held in
/// **host-pinned** memory (page-locked, so HtoD/DtoH transfers can overlap
/// compute). The host-side accounting here is the source of truth; the actual
/// pinned-buffer allocation is gated behind the `cuda` feature.
#[derive(Debug, Clone)]
pub struct PinnedBudget {
    /// Soft cap, in bytes. `None` is uncapped.
    budget_bytes: Option<usize>,
    /// Bytes currently accounted as live host-pinned intermediates.
    live_bytes: usize,
}

impl PinnedBudget {
    /// New budget tracker. `budget_bytes == None` means uncapped.
    pub fn new(budget_bytes: Option<usize>) -> Self {
        Self {
            budget_bytes,
            live_bytes: 0,
        }
    }

    /// Bytes currently accounted as live.
    pub fn live_bytes(&self) -> usize {
        self.live_bytes
    }

    /// `true` if adding `bytes` more would stay within budget (always `true`
    /// when uncapped).
    pub fn fits(&self, bytes: usize) -> bool {
        match self.budget_bytes {
            None => true,
            Some(b) => self.live_bytes.saturating_add(bytes) <= b,
        }
    }

    /// Reserve `bytes` of host-pinned intermediate space.
    ///
    /// On the host side this only bumps the accounting counter. The
    /// device-pinned allocation (`cuMemHostAlloc` with the portable /
    /// write-combined flags, registered for overlapped transfer) is a
    /// follow-up:
    ///
    /// TODO(cuda): when the `cuda` feature is active, back each reservation
    /// with a real page-locked host buffer from `crate::cuda::mem_pool` so
    /// the morsel pipeline can issue async HtoD copies that overlap kernel
    /// execution. The host accounting below stays the budget source of truth.
    ///
    /// # Errors
    /// Returns a `Plan` error if the reservation would exceed the budget.
    pub fn reserve(&mut self, bytes: usize) -> BoltResult<()> {
        if !self.fits(bytes) {
            return Err(BoltError::Plan(format!(
                "host-pinned intermediate budget exceeded: {} live + {} requested > {} budget",
                self.live_bytes,
                bytes,
                self.budget_bytes.unwrap_or(usize::MAX)
            )));
        }
        self.live_bytes = self.live_bytes.saturating_add(bytes);
        Ok(())
    }

    /// Release `bytes` previously reserved. Saturates at zero so a
    /// double-release cannot underflow.
    pub fn release(&mut self, bytes: usize) {
        self.live_bytes = self.live_bytes.saturating_sub(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    /// Build a single-column Int32 batch with `n` rows valued `0..n`.
    fn int_batch(n: usize) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "v",
            ArrowDataType::Int32,
            false,
        )]));
        let arr = Int32Array::from((0..n as i32).collect::<Vec<_>>());
        RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap()
    }

    /// Collect the row counts of every morsel a stream yields.
    fn morsel_row_counts(batches: &[RecordBatch], morsel_rows: usize) -> Vec<usize> {
        let stream = BatchStream::new(batches, morsel_rows).unwrap();
        stream.morsels().map(|m| m.num_rows()).collect()
    }

    #[test]
    fn morsels_exact_multiple() {
        // 8 rows, morsel 4 → two full morsels, no remainder.
        let batches = vec![int_batch(8)];
        assert_eq!(morsel_row_counts(&batches, 4), vec![4, 4]);
        let stream = BatchStream::new(&batches, 4).unwrap();
        assert_eq!(stream.num_morsels(), 2);
        assert_eq!(stream.total_rows(), 8);
    }

    #[test]
    fn morsels_with_remainder() {
        // 10 rows, morsel 4 → 4, 4, 2.
        let batches = vec![int_batch(10)];
        assert_eq!(morsel_row_counts(&batches, 4), vec![4, 4, 2]);
        let stream = BatchStream::new(&batches, 4).unwrap();
        assert_eq!(stream.num_morsels(), 3);
    }

    #[test]
    fn morsels_morsel_larger_than_table() {
        // morsel bigger than the whole table → single morsel of all rows.
        let batches = vec![int_batch(3)];
        assert_eq!(morsel_row_counts(&batches, 100), vec![3]);
        assert_eq!(BatchStream::new(&batches, 100).unwrap().num_morsels(), 1);
    }

    #[test]
    fn morsels_single_row_morsel() {
        // morsel of 1 → one morsel per row.
        let batches = vec![int_batch(3)];
        assert_eq!(morsel_row_counts(&batches, 1), vec![1, 1, 1]);
    }

    #[test]
    fn morsels_empty_table() {
        // Zero rows → zero morsels.
        let batches = vec![int_batch(0)];
        assert_eq!(morsel_row_counts(&batches, 4), Vec::<usize>::new());
        assert_eq!(BatchStream::new(&batches, 4).unwrap().num_morsels(), 0);
        assert_eq!(BatchStream::new(&batches, 4).unwrap().total_rows(), 0);
    }

    #[test]
    fn morsels_no_batches_at_all() {
        // No batches → zero morsels (does not panic / spin).
        let batches: Vec<RecordBatch> = vec![];
        assert_eq!(morsel_row_counts(&batches, 4), Vec::<usize>::new());
    }

    #[test]
    fn morsels_do_not_span_batch_boundary() {
        // Two batches of 3 and 5 rows, morsel 4. Morsels never cross a batch
        // boundary, so batch0 → [3], batch1 → [4, 1].
        let batches = vec![int_batch(3), int_batch(5)];
        assert_eq!(morsel_row_counts(&batches, 4), vec![3, 4, 1]);
        let stream = BatchStream::new(&batches, 4).unwrap();
        assert_eq!(stream.num_morsels(), 3);
        assert_eq!(stream.total_rows(), 8);
    }

    #[test]
    fn morsels_skip_empty_batches_between_full_ones() {
        // Empty batch in the middle is skipped, not yielded as a 0-row morsel.
        let batches = vec![int_batch(4), int_batch(0), int_batch(2)];
        assert_eq!(morsel_row_counts(&batches, 4), vec![4, 2]);
    }

    #[test]
    fn morsels_preserve_values() {
        // The sliced morsels must carry the right row values (zero-copy slice
        // offset is applied correctly).
        let batches = vec![int_batch(5)];
        let stream = BatchStream::new(&batches, 2).unwrap();
        let collected: Vec<Vec<i32>> = stream
            .morsels()
            .map(|m| {
                let a = m
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                (0..a.len()).map(|i| a.value(i)).collect()
            })
            .collect();
        assert_eq!(collected, vec![vec![0, 1], vec![2, 3], vec![4]]);
    }

    #[test]
    fn morsels_re_iterable() {
        // morsels() can be called twice with identical results.
        let batches = vec![int_batch(7)];
        let stream = BatchStream::new(&batches, 3).unwrap();
        let first: Vec<usize> = stream.morsels().map(|m| m.num_rows()).collect();
        let second: Vec<usize> = stream.morsels().map(|m| m.num_rows()).collect();
        assert_eq!(first, second);
        assert_eq!(first, vec![3, 3, 1]);
    }

    #[test]
    fn batchstream_rejects_zero_morsel_rows() {
        let batches = vec![int_batch(4)];
        assert!(BatchStream::new(&batches, 0).is_err());
    }

    // ---- budget / spill hook ------------------------------------------

    #[test]
    fn plan_upload_uncapped_is_whole() {
        assert_eq!(plan_upload(1 << 40, 1_000, None), MorselPlan::Whole);
    }

    #[test]
    fn plan_upload_under_budget_is_whole() {
        // 100 bytes, budget 1000 → fits whole.
        assert_eq!(plan_upload(100, 10, Some(1000)), MorselPlan::Whole);
    }

    #[test]
    fn plan_upload_exactly_at_budget_is_whole() {
        assert_eq!(plan_upload(1000, 10, Some(1000)), MorselPlan::Whole);
    }

    #[test]
    fn plan_upload_over_budget_chunks() {
        // 1000 bytes over 100 rows → 10 bytes/row. Budget 250 → 25 rows/morsel.
        match plan_upload(1000, 100, Some(250)) {
            MorselPlan::Morsels { morsel_rows } => assert_eq!(morsel_rows, 25),
            other => panic!("expected Morsels, got {other:?}"),
        }
    }

    #[test]
    fn plan_upload_oversized_row_clamps_to_one() {
        // Budget smaller than a single row's footprint → morsel of 1 row.
        // 1000 bytes over 2 rows → 500 bytes/row, budget 100 → clamp to 1.
        assert_eq!(
            plan_upload(1000, 2, Some(100)),
            MorselPlan::Morsels { morsel_rows: 1 }
        );
    }

    #[test]
    fn plan_upload_zero_rows_is_whole() {
        // Degenerate: a zero-row table always fits.
        assert_eq!(plan_upload(0, 0, Some(10)), MorselPlan::Whole);
    }

    #[test]
    fn morsel_plan_accessor() {
        assert_eq!(MorselPlan::Whole.morsel_rows(), None);
        assert_eq!(
            MorselPlan::Morsels { morsel_rows: 7 }.morsel_rows(),
            Some(7)
        );
    }

    // ---- TableSource --------------------------------------------------

    #[test]
    fn table_source_materialized_drains_clone() {
        let src = TableSource::Materialized(vec![int_batch(3), int_batch(2)]);
        assert!(!src.is_streaming());
        let drained = src.drain_to_batches("t").unwrap();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].num_rows(), 3);
        assert_eq!(drained[1].num_rows(), 2);
    }

    #[test]
    fn table_source_materialized_empty_errors() {
        let src = TableSource::Materialized(vec![]);
        assert!(src.drain_to_batches("t").is_err());
    }

    #[test]
    fn table_source_streaming_drains_lazily_and_replays() {
        let producer: BatchProducer = Box::new(|| {
            Box::new(
                vec![Ok(int_batch(4)), Ok(int_batch(1))].into_iter(),
            )
        });
        let src = TableSource::Streaming(producer);
        assert!(src.is_streaming());
        // First drain.
        let a = src.drain_to_batches("t").unwrap();
        assert_eq!(a.iter().map(|b| b.num_rows()).sum::<usize>(), 5);
        // Re-iterable: a second drain yields the same shape.
        let b = src.drain_to_batches("t").unwrap();
        assert_eq!(b.iter().map(|x| x.num_rows()).sum::<usize>(), 5);
    }

    #[test]
    fn table_source_streaming_propagates_error() {
        let producer: BatchProducer = Box::new(|| {
            Box::new(
                vec![
                    Ok(int_batch(2)),
                    Err(BoltError::Plan("boom".to_string())),
                ]
                .into_iter(),
            )
        });
        let src = TableSource::Streaming(producer);
        let err = src.drain_to_batches("t").unwrap_err();
        assert!(matches!(err, BoltError::Plan(m) if m == "boom"));
    }

    #[test]
    fn table_source_streaming_empty_errors() {
        let producer: BatchProducer = Box::new(|| Box::new(std::iter::empty()));
        let src = TableSource::Streaming(producer);
        assert!(src.drain_to_batches("t").is_err());
    }

    // ---- PinnedBudget -------------------------------------------------

    #[test]
    fn pinned_budget_uncapped_always_fits() {
        let mut b = PinnedBudget::new(None);
        assert!(b.fits(usize::MAX));
        b.reserve(1 << 30).unwrap();
        assert_eq!(b.live_bytes(), 1 << 30);
    }

    #[test]
    fn pinned_budget_reserve_release_roundtrip() {
        let mut b = PinnedBudget::new(Some(100));
        b.reserve(60).unwrap();
        assert_eq!(b.live_bytes(), 60);
        assert!(b.fits(40));
        assert!(!b.fits(41));
        b.release(60);
        assert_eq!(b.live_bytes(), 0);
        assert!(b.fits(100));
    }

    #[test]
    fn pinned_budget_over_reserve_errors() {
        let mut b = PinnedBudget::new(Some(100));
        b.reserve(80).unwrap();
        assert!(b.reserve(30).is_err());
        // Failed reservation does not change live bytes.
        assert_eq!(b.live_bytes(), 80);
    }

    #[test]
    fn pinned_budget_release_saturates() {
        let mut b = PinnedBudget::new(Some(100));
        b.reserve(10).unwrap();
        b.release(1000); // over-release
        assert_eq!(b.live_bytes(), 0);
    }

    #[test]
    fn estimate_batch_bytes_nonzero() {
        // A 100-row Int32 batch must report at least 400 bytes (4 bytes/row).
        let b = int_batch(100);
        assert!(estimate_batch_bytes(&b) >= 400);
        assert!(estimate_batches_bytes(&[int_batch(50), int_batch(50)]) >= 400);
    }
}
