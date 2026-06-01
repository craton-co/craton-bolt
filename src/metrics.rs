// SPDX-License-Identifier: Apache-2.0

//! Lightweight, dependency-free metrics registry (M5).
//!
//! The [`observability`](crate::observability) module wraps each query phase
//! in a `tracing` span so a subscriber can attribute *latency* to the right
//! slice of the pipeline. Spans are great for per-query, sampled traces but
//! awkward for the other half of observability: monotone process-wide
//! *aggregates* — "how many queries have we run", "what's the PTX cache hit
//! rate", "how many bytes have we shipped over PCIe". Those want cheap atomic
//! counters and pre-aggregated latency histograms that a scraper can poll on
//! its own schedule, not a span per event.
//!
//! This module provides exactly that, with no external metrics crate:
//!
//! * A fixed set of named [`Counter`]s (monotone `u64`, [`Relaxed`] atomics).
//! * A fixed set of named latency [`Histogram`]s with power-of-two micro-second
//!   buckets, one per query phase.
//! * A process-wide [`Metrics`] singleton reached via [`metrics()`].
//! * A plain-data [`MetricsSnapshot`] returned by [`snapshot()`], mirroring the
//!   way [`pool_stats`](crate::pool_stats) returns a [`PoolStats`](crate::PoolStats)
//!   value the caller owns.
//!
//! # Scraping
//!
//! The registry is *self-hosting*: callers across the engine bump counters and
//! observe latencies inline; an exporter polls [`snapshot()`] whenever it likes.
//! There is no callback to install (unlike
//! [`install_pool_stats_observer`](crate::install_pool_stats_observer)) because
//! the data is already aggregated — pull, don't push.
//!
//! ## Prometheus
//!
//! A textfile / `/metrics` handler maps the snapshot to exposition format. Each
//! counter is a Prometheus *counter*; each histogram bucket is a cumulative
//! `_bucket{le=...}` series. Cheap pseudocode:
//!
//! ```ignore
//! use craton_bolt::metrics_snapshot; // = craton_bolt::snapshot(), re-exported
//!
//! let s = metrics_snapshot();
//! let mut out = String::new();
//! for (name, value) in s.counters() {
//!     out += &format!("# TYPE bolt_{name} counter\nbolt_{name} {value}\n");
//! }
//! for h in s.histograms() {
//!     out += &format!("# TYPE bolt_phase_latseconds_{} histogram\n", h.phase.as_str());
//!     let mut cumulative = 0u64;
//!     for (upper_us, count) in h.buckets() {
//!         cumulative += count;
//!         let le = upper_us as f64 / 1e6; // bucket upper bound, seconds
//!         out += &format!(
//!             "bolt_phase_latseconds_{}_bucket{{le=\"{le}\"}} {cumulative}\n",
//!             h.phase.as_str()
//!         );
//!     }
//!     out += &format!(
//!         "bolt_phase_latseconds_{}_bucket{{le=\"+Inf\"}} {}\n",
//!         h.phase.as_str(), h.count
//!     );
//!     out += &format!("bolt_phase_latseconds_{}_count {}\n", h.phase.as_str(), h.count);
//!     out += &format!("bolt_phase_latseconds_{}_sum {}\n",
//!         h.phase.as_str(), h.sum_micros as f64 / 1e6);
//! }
//! ```
//!
//! ## Structured logs
//!
//! For stacks without a pull collector, periodically log the snapshot — it is
//! `Debug` and `Clone`, so `log::info!("metrics: {:?}", metrics_snapshot())`
//! emits one structured line you can index downstream.
//!
//! # Concurrency & cost
//!
//! Every counter and every histogram bucket is a single [`AtomicU64`]. Bumps
//! use [`Relaxed`] ordering: these are independent statistics, not
//! synchronisation flags, so no happens-before relationship between distinct
//! metrics is needed or implied. A snapshot reads each atom with [`Relaxed`]
//! too, so it is *not* a globally-consistent instant — counters may be observed
//! mid-flight relative to one another. That is the standard, intended contract
//! for metrics scraping and keeps the hot path lock-free and allocation-free.
//!
//! [`Relaxed`]: std::sync::atomic::Ordering::Relaxed

use std::sync::atomic::{AtomicU64, Ordering};

use once_cell::sync::Lazy;

/// Number of latency histogram buckets.
///
/// Buckets are power-of-two micro-second upper bounds: bucket `i` counts
/// observations in `(2^(i-1), 2^i]` µs, with bucket `0` catching everything
/// `<= 1 µs`. With [`HISTOGRAM_BUCKETS`] = 24 the top finite bound is `2^23`
/// µs ≈ 8.4 s; an observation larger than that lands in the final overflow
/// bucket (index `HISTOGRAM_BUCKETS - 1`), which therefore means "≥ 2^22 µs".
pub const HISTOGRAM_BUCKETS: usize = 24;

/// Named monotone counters exposed by the registry.
///
/// The set is fixed and ordered; [`Counter::ALL`] iterates it in declaration
/// order, which is also the order [`MetricsSnapshot::counters`] yields. Adding a
/// variant is non-breaking; reordering / removing is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Counter {
    /// Total queries accepted for execution (success or failure).
    QueriesTotal = 0,
    /// Queries that returned an error from any phase.
    QueriesFailed = 1,
    /// PTX module cache lookups that hit a warm entry.
    PtxCacheHits = 2,
    /// PTX module cache lookups that had to compile / load.
    PtxCacheMisses = 3,
    /// Kernel grid launches issued to the driver.
    GpuLaunchesTotal = 4,
    /// Queries (or sub-plans) that fell back to the host execution path.
    HostFallbacksTotal = 5,
    /// Cumulative bytes copied host→device (H2D).
    BytesUploaded = 6,
    /// Cumulative bytes copied device→host (D2H).
    BytesDownloaded = 7,
}

impl Counter {
    /// All counters in declaration / snapshot order.
    pub const ALL: [Counter; 8] = [
        Counter::QueriesTotal,
        Counter::QueriesFailed,
        Counter::PtxCacheHits,
        Counter::PtxCacheMisses,
        Counter::GpuLaunchesTotal,
        Counter::HostFallbacksTotal,
        Counter::BytesUploaded,
        Counter::BytesDownloaded,
    ];

    /// Stable snake_case name, suitable as a metric key (no `bolt_` prefix —
    /// add your own namespace at export time).
    pub const fn as_str(self) -> &'static str {
        match self {
            Counter::QueriesTotal => "queries_total",
            Counter::QueriesFailed => "queries_failed",
            Counter::PtxCacheHits => "ptx_cache_hits",
            Counter::PtxCacheMisses => "ptx_cache_misses",
            Counter::GpuLaunchesTotal => "gpu_launches_total",
            Counter::HostFallbacksTotal => "host_fallbacks_total",
            Counter::BytesUploaded => "bytes_uploaded",
            Counter::BytesDownloaded => "bytes_downloaded",
        }
    }
}

/// Named query phases that own a latency histogram.
///
/// Names mirror the tracing span catalogue in
/// [`observability`](crate::observability) so a span name and its histogram
/// line up one-to-one. [`Phase::ALL`] iterates in declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Phase {
    /// SQL text → `LogicalPlan`.
    Parse = 0,
    /// Logical-plan rewrite pipeline.
    Plan = 1,
    /// `LogicalPlan` → `PhysicalPlan`.
    Lower = 2,
    /// PhysicalPlan → PTX text.
    Codegen = 3,
    /// PTX → loaded driver module.
    PtxLoad = 4,
    /// Kernel grid launch + sync.
    Launch = 5,
    /// H2D / D2H memcpy.
    Transfer = 6,
    /// Arrow array packing of results.
    Materialize = 7,
}

impl Phase {
    /// All phases in declaration / snapshot order.
    pub const ALL: [Phase; 8] = [
        Phase::Parse,
        Phase::Plan,
        Phase::Lower,
        Phase::Codegen,
        Phase::PtxLoad,
        Phase::Launch,
        Phase::Transfer,
        Phase::Materialize,
    ];

    /// Stable snake_case name, matching the corresponding tracing span.
    pub const fn as_str(self) -> &'static str {
        match self {
            Phase::Parse => "parse",
            Phase::Plan => "plan",
            Phase::Lower => "lower",
            Phase::Codegen => "codegen",
            Phase::PtxLoad => "ptx_load",
            Phase::Launch => "launch",
            Phase::Transfer => "transfer",
            Phase::Materialize => "materialize",
        }
    }
}

/// Map a latency in micro-seconds to a power-of-two bucket index.
///
/// Bucket `i` (for `0 < i < HISTOGRAM_BUCKETS - 1`) covers `(2^(i-1), 2^i]` µs.
/// Bucket `0` covers `[0, 1]` µs (i.e. `<= 1`). The final bucket
/// (`HISTOGRAM_BUCKETS - 1`) is the overflow class: anything beyond the
/// last finite bound lands there.
///
/// Boundary semantics are *inclusive upper bound*: exactly `2^k` µs maps to
/// bucket `k`, and `2^k + 1` maps to bucket `k + 1`. This matches the
/// Prometheus `le` ("less-than-or-equal") convention.
#[inline]
fn bucket_index(micros: u64) -> usize {
    if micros <= 1 {
        return 0;
    }
    // `micros >= 2` here. The target bucket is `ceil(log2(micros))`, which is
    // also the exponent of the smallest power of two `>= micros`. Computing it
    // via leading-zeros keeps the inclusive-upper-bound rule (exact `2^k` maps
    // to bucket `k`) crisp and branch-light:
    //   ceil_log2(micros) == u64::BITS - (micros - 1).leading_zeros()
    let ceil_log2 = (u64::BITS - (micros - 1).leading_zeros()) as usize;
    ceil_log2.min(HISTOGRAM_BUCKETS - 1)
}

/// Inclusive upper bound (in µs) of histogram bucket `i`.
///
/// For the overflow bucket (`HISTOGRAM_BUCKETS - 1`) this returns the upper
/// bound of the last *finite* class, i.e. `2^(HISTOGRAM_BUCKETS - 1)`; callers
/// rendering exposition format should treat that final bucket as `+Inf`.
const fn bucket_upper_micros(i: usize) -> u64 {
    if i == 0 {
        1
    } else {
        1u64 << i
    }
}

/// A single monotone counter backed by one [`AtomicU64`].
#[derive(Debug, Default)]
pub struct AtomicCounter(AtomicU64);

impl AtomicCounter {
    /// Add `n` to the counter. [`Relaxed`](Ordering::Relaxed) — counters are
    /// independent statistics, not synchronisation flags.
    #[inline]
    pub fn add(&self, n: u64) {
        self.0.fetch_add(n, Ordering::Relaxed);
    }

    /// Add one.
    #[inline]
    pub fn inc(&self) {
        self.add(1);
    }

    /// Current value (Relaxed load).
    #[inline]
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// A latency histogram: per-bucket counts plus running observation count and
/// micro-second sum (so a scraper can compute averages and `_sum` series).
#[derive(Debug)]
pub struct Histogram {
    buckets: [AtomicU64; HISTOGRAM_BUCKETS],
    count: AtomicU64,
    sum_micros: AtomicU64,
}

impl Default for Histogram {
    fn default() -> Self {
        // `AtomicU64` is not `Copy`, so we can't use array-repeat syntax.
        Histogram {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            count: AtomicU64::new(0),
            sum_micros: AtomicU64::new(0),
        }
    }
}

impl Histogram {
    /// Record one observation of `micros` micro-seconds.
    #[inline]
    pub fn observe(&self, micros: u64) {
        let idx = bucket_index(micros);
        self.buckets[idx].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_micros.fetch_add(micros, Ordering::Relaxed);
    }

    /// Record an observation given a [`std::time::Duration`].
    ///
    /// Saturates to `u64::MAX` µs for absurdly long durations rather than
    /// wrapping — they land in the overflow bucket either way.
    #[inline]
    pub fn observe_duration(&self, d: std::time::Duration) {
        let micros = u64::try_from(d.as_micros()).unwrap_or(u64::MAX);
        self.observe(micros);
    }

    /// Total observations recorded (Relaxed load).
    #[inline]
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Sum of all observed latencies in µs (Relaxed load).
    #[inline]
    pub fn sum_micros(&self) -> u64 {
        self.sum_micros.load(Ordering::Relaxed)
    }
}

/// The process-wide metrics registry.
///
/// Reach the singleton via [`metrics()`]. All fields are atomic, so the
/// registry is shared by `&'static` reference with no locking.
#[derive(Debug)]
pub struct Metrics {
    counters: [AtomicCounter; Counter::ALL.len()],
    histograms: [Histogram; Phase::ALL.len()],
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics {
            counters: std::array::from_fn(|_| AtomicCounter::default()),
            histograms: std::array::from_fn(|_| Histogram::default()),
        }
    }
}

impl Metrics {
    /// Borrow the [`AtomicCounter`] for `c`.
    #[inline]
    pub fn counter(&self, c: Counter) -> &AtomicCounter {
        &self.counters[c as usize]
    }

    /// Increment counter `c` by one.
    #[inline]
    pub fn inc(&self, c: Counter) {
        self.counter(c).inc();
    }

    /// Add `n` to counter `c`.
    #[inline]
    pub fn add(&self, c: Counter, n: u64) {
        self.counter(c).add(n);
    }

    /// Borrow the [`Histogram`] for phase `p`.
    #[inline]
    pub fn histogram(&self, p: Phase) -> &Histogram {
        &self.histograms[p as usize]
    }

    /// Record a `micros`-µs latency for phase `p`.
    #[inline]
    pub fn observe(&self, p: Phase, micros: u64) {
        self.histogram(p).observe(micros);
    }

    /// Record a [`Duration`](std::time::Duration) latency for phase `p`.
    #[inline]
    pub fn observe_duration(&self, p: Phase, d: std::time::Duration) {
        self.histogram(p).observe_duration(d);
    }

    /// Take a plain-data [`MetricsSnapshot`] of the whole registry.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            counters: std::array::from_fn(|i| self.counters[i].get()),
            histograms: std::array::from_fn(|i| {
                let h = &self.histograms[i];
                HistogramSnapshot {
                    phase: Phase::ALL[i],
                    buckets: std::array::from_fn(|b| h.buckets[b].load(Ordering::Relaxed)),
                    count: h.count(),
                    sum_micros: h.sum_micros(),
                }
            }),
        }
    }
}

/// Process-wide metrics singleton.
static METRICS: Lazy<Metrics> = Lazy::new(Metrics::default);

/// Borrow the process-wide [`Metrics`] registry.
///
/// Cheap and lock-free; safe to call from any thread on the hot path. The
/// returned reference is `'static`.
///
/// ```ignore
/// use craton_bolt::{metrics, Counter, Phase};
/// metrics().inc(Counter::QueriesTotal);
/// metrics().observe(Phase::Parse, 42); // 42 µs
/// ```
pub fn metrics() -> &'static Metrics {
    &METRICS
}

/// Convenience wrapper for [`metrics()`]`.snapshot()`.
///
/// Mirrors [`pool_stats`](crate::pool_stats): one function returns an owned,
/// plain-data value an exporter can serialise without touching engine
/// internals.
pub fn snapshot() -> MetricsSnapshot {
    metrics().snapshot()
}

/// Owned, plain-data snapshot of all counters and histograms.
///
/// Returned by [`snapshot()`]. Not a globally-consistent instant — individual
/// atomics are read with [`Relaxed`](Ordering::Relaxed) ordering, the standard
/// metrics-scraping contract (see module docs).
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    counters: [u64; Counter::ALL.len()],
    histograms: [HistogramSnapshot; Phase::ALL.len()],
}

impl MetricsSnapshot {
    /// Value of counter `c` at snapshot time.
    pub fn counter(&self, c: Counter) -> u64 {
        self.counters[c as usize]
    }

    /// Iterate `(name, value)` for every counter, in declaration order.
    pub fn counters(&self) -> impl Iterator<Item = (&'static str, u64)> + '_ {
        Counter::ALL
            .iter()
            .map(move |&c| (c.as_str(), self.counters[c as usize]))
    }

    /// Histogram snapshot for phase `p`.
    pub fn histogram(&self, p: Phase) -> &HistogramSnapshot {
        &self.histograms[p as usize]
    }

    /// Iterate every [`HistogramSnapshot`], in declaration order.
    pub fn histograms(&self) -> impl Iterator<Item = &HistogramSnapshot> + '_ {
        self.histograms.iter()
    }
}

/// Owned snapshot of one phase histogram.
#[derive(Debug, Clone)]
pub struct HistogramSnapshot {
    /// The phase this histogram belongs to.
    pub phase: Phase,
    /// Per-bucket observation counts (non-cumulative). Bucket `i`'s inclusive
    /// upper bound in µs is [`bucket_upper_micros`]; the last bucket is the
    /// overflow class.
    pub buckets: [u64; HISTOGRAM_BUCKETS],
    /// Total observations.
    pub count: u64,
    /// Sum of observed latencies in µs.
    pub sum_micros: u64,
}

impl HistogramSnapshot {
    /// Iterate `(upper_bound_micros, count)` per bucket, non-cumulative, in
    /// ascending-bound order. The final tuple is the overflow bucket; its
    /// "upper bound" is the last finite bound (treat as `+Inf` when exporting).
    pub fn buckets(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.buckets
            .iter()
            .enumerate()
            .map(|(i, &n)| (bucket_upper_micros(i), n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_names_match_variants() {
        // Spot-check a couple and the full count.
        assert_eq!(Counter::ALL.len(), 8);
        assert_eq!(Counter::QueriesTotal.as_str(), "queries_total");
        assert_eq!(Counter::BytesDownloaded.as_str(), "bytes_downloaded");
        // `as usize` discriminants line up with `ALL` ordering.
        for (i, c) in Counter::ALL.iter().enumerate() {
            assert_eq!(*c as usize, i);
        }
    }

    #[test]
    fn phase_names_match_variants() {
        assert_eq!(Phase::ALL.len(), 8);
        assert_eq!(Phase::Parse.as_str(), "parse");
        assert_eq!(Phase::Materialize.as_str(), "materialize");
        for (i, p) in Phase::ALL.iter().enumerate() {
            assert_eq!(*p as usize, i);
        }
    }

    #[test]
    fn counter_increment_and_read() {
        let m = Metrics::default();
        assert_eq!(m.counter(Counter::QueriesTotal).get(), 0);
        m.inc(Counter::QueriesTotal);
        m.inc(Counter::QueriesTotal);
        assert_eq!(m.counter(Counter::QueriesTotal).get(), 2);

        m.add(Counter::BytesUploaded, 4096);
        m.add(Counter::BytesUploaded, 100);
        assert_eq!(m.counter(Counter::BytesUploaded).get(), 4196);

        // Independent counters do not bleed into each other.
        assert_eq!(m.counter(Counter::BytesDownloaded).get(), 0);
    }

    #[test]
    fn bucket_index_zero_and_one_go_to_bucket_zero() {
        assert_eq!(bucket_index(0), 0);
        assert_eq!(bucket_index(1), 0);
    }

    #[test]
    fn bucket_index_inclusive_upper_bound() {
        // 2 µs -> bucket 1 (upper bound 2^1 = 2, inclusive).
        assert_eq!(bucket_index(2), 1);
        // 3 µs -> bucket 2 (covers (2, 4]).
        assert_eq!(bucket_index(3), 2);
        assert_eq!(bucket_index(4), 2);
        // 5 µs -> bucket 3 (covers (4, 8]).
        assert_eq!(bucket_index(5), 3);
        assert_eq!(bucket_index(8), 3);
        assert_eq!(bucket_index(9), 4);
        // Exact powers of two map to their own exponent's bucket.
        assert_eq!(bucket_index(1024), 10);
        assert_eq!(bucket_index(1025), 11);
    }

    #[test]
    fn bucket_index_overflow_saturates() {
        let last = HISTOGRAM_BUCKETS - 1;
        // Last finite bound is 2^(HISTOGRAM_BUCKETS-1).
        let big = 1u64 << (HISTOGRAM_BUCKETS - 1);
        assert_eq!(bucket_index(big), last);
        assert_eq!(bucket_index(big + 1), last);
        assert_eq!(bucket_index(u64::MAX), last);
    }

    #[test]
    fn bucket_upper_micros_matches_index() {
        assert_eq!(bucket_upper_micros(0), 1);
        assert_eq!(bucket_upper_micros(1), 2);
        assert_eq!(bucket_upper_micros(10), 1024);
        // Round-trip: an observation exactly at a bucket's upper bound lands
        // in that bucket.
        for i in 1..HISTOGRAM_BUCKETS - 1 {
            let ub = bucket_upper_micros(i);
            assert_eq!(bucket_index(ub), i, "upper bound {ub} should map to bucket {i}");
        }
    }

    #[test]
    fn histogram_observe_counts_and_sums() {
        let h = Histogram::default();
        h.observe(1); // bucket 0
        h.observe(2); // bucket 1
        h.observe(3); // bucket 2
        h.observe(4); // bucket 2
        assert_eq!(h.count(), 4);
        assert_eq!(h.sum_micros(), 1 + 2 + 3 + 4);

        let snap = {
            let m = Metrics::default();
            m.histogram(Phase::Parse).observe(3);
            m.histogram(Phase::Parse).observe(4);
            m.snapshot()
        };
        let ph = snap.histogram(Phase::Parse);
        assert_eq!(ph.count, 2);
        assert_eq!(ph.sum_micros, 7);
        assert_eq!(ph.buckets[2], 2);
    }

    #[test]
    fn histogram_observe_duration() {
        let h = Histogram::default();
        h.observe_duration(std::time::Duration::from_micros(5)); // bucket 3
        h.observe_duration(std::time::Duration::from_millis(1)); // 1000 µs -> bucket 10
        assert_eq!(h.count(), 2);
        assert_eq!(h.sum_micros(), 1005);
        assert_eq!(bucket_index(1000), 10);
    }

    #[test]
    fn snapshot_fidelity() {
        let m = Metrics::default();
        m.add(Counter::QueriesTotal, 7);
        m.inc(Counter::PtxCacheHits);
        m.observe(Phase::Launch, 16); // bucket 4 (covers (8,16])
        m.observe(Phase::Launch, 17); // bucket 5
        m.observe(Phase::Transfer, 2); // bucket 1

        let snap = m.snapshot();

        // Counters mirror the registry exactly.
        assert_eq!(snap.counter(Counter::QueriesTotal), 7);
        assert_eq!(snap.counter(Counter::PtxCacheHits), 1);
        assert_eq!(snap.counter(Counter::QueriesFailed), 0);

        // counters() iterator yields all names in order.
        let pairs: Vec<_> = snap.counters().collect();
        assert_eq!(pairs.len(), Counter::ALL.len());
        assert_eq!(pairs[0], ("queries_total", 7));

        // Histograms mirror per-bucket counts.
        let launch = snap.histogram(Phase::Launch);
        assert_eq!(launch.count, 2);
        assert_eq!(launch.sum_micros, 33);
        assert_eq!(launch.buckets[4], 1);
        assert_eq!(launch.buckets[5], 1);

        let transfer = snap.histogram(Phase::Transfer);
        assert_eq!(transfer.count, 1);
        assert_eq!(transfer.buckets[1], 1);

        // Untouched phase is all zeros.
        let parse = snap.histogram(Phase::Parse);
        assert_eq!(parse.count, 0);
        assert!(parse.buckets.iter().all(|&b| b == 0));

        // buckets() iterator: cumulative reconstruction equals count.
        let total: u64 = launch.buckets().map(|(_, n)| n).sum();
        assert_eq!(total, launch.count);
    }

    #[test]
    fn snapshot_is_a_copy_not_a_view() {
        let m = Metrics::default();
        m.inc(Counter::QueriesTotal);
        let snap = m.snapshot();
        // Mutating the live registry does not change a prior snapshot.
        m.add(Counter::QueriesTotal, 100);
        assert_eq!(snap.counter(Counter::QueriesTotal), 1);
        assert_eq!(m.counter(Counter::QueriesTotal).get(), 101);
    }

    #[test]
    fn global_singleton_is_stable() {
        let a = metrics() as *const Metrics;
        let b = metrics() as *const Metrics;
        assert_eq!(a, b, "metrics() must return the same singleton");
        // snapshot() convenience matches metrics().snapshot() shape.
        let s = snapshot();
        assert_eq!(s.histograms().count(), Phase::ALL.len());
    }
}
