// SPDX-License-Identifier: Apache-2.0

//! Lightweight, dependency-free metrics registry (M5).
//!
//! The [`observability`](crate::observability) module wraps each query phase
//! in a `tracing` span so a subscriber can attribute *latency* to the right
//! slice of the pipeline. Spans are great for per-query, sampled traces but
//! awkward for the other half of observability: monotone process-wide
//! *aggregates* — "how many queries have we run", "what's the PTX cache hit
//! rate", "how often do we fall back to the host path". Those want cheap atomic
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
//! A textfile / `/metrics` handler maps the snapshot to exposition format via
//! [`render_prometheus`] — no external metrics crate, no new dependency. Each
//! counter becomes a Prometheus *counter* (`craton_<name>`); each phase
//! histogram becomes a Prometheus *histogram* with cumulative
//! `craton_phase_latency_seconds_bucket{phase="…",le="…"}` series plus the
//! conventional `_count` / `_sum` companions. The whole `/metrics` body is one
//! call:
//!
//! ```ignore
//! use craton_bolt::{metrics_snapshot, render_prometheus};
//!
//! // GET /metrics
//! let body = render_prometheus(&metrics_snapshot());
//! // -> Content-Type: text/plain; version=0.0.4
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
///
/// Every variant here is *actually recorded* somewhere on the engine hot path
/// (see `exec::engine` for the query/launch/fallback bumps and
/// `exec::module_cache` for the PTX-cache hit/miss bumps). We deliberately do
/// **not** advertise a counter we never increment: a shipped, documented metric
/// that is permanently zero is worse than absent, because an operator wiring a
/// dashboard cannot tell "feature is idle" from "instrumentation is missing".
/// The H2D/D2H PCIe byte totals once lived here but were never wired at any
/// `cuda::GpuBuffer` copy site, so they were removed rather than ship a
/// guaranteed-zero series; add them back only together with the call-site bumps.
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
}

impl Counter {
    /// All counters in declaration / snapshot order.
    pub const ALL: [Counter; 6] = [
        Counter::QueriesTotal,
        Counter::QueriesFailed,
        Counter::PtxCacheHits,
        Counter::PtxCacheMisses,
        Counter::GpuLaunchesTotal,
        Counter::HostFallbacksTotal,
    ];

    /// Stable snake_case name, suitable as a metric key (no `craton_` prefix —
    /// add your own namespace at export time, e.g. [`render_prometheus`]).
    pub const fn as_str(self) -> &'static str {
        match self {
            Counter::QueriesTotal => "queries_total",
            Counter::QueriesFailed => "queries_failed",
            Counter::PtxCacheHits => "ptx_cache_hits",
            Counter::PtxCacheMisses => "ptx_cache_misses",
            Counter::GpuLaunchesTotal => "gpu_launches_total",
            Counter::HostFallbacksTotal => "host_fallbacks_total",
        }
    }

    /// One-line, human-readable description for the Prometheus `# HELP` line.
    ///
    /// Stable prose; safe to surface in a dashboard. Kept free of `\n` and `\`
    /// so it needs no exposition-format escaping.
    pub const fn help(self) -> &'static str {
        match self {
            Counter::QueriesTotal => "Total queries accepted for execution (success or failure).",
            Counter::QueriesFailed => "Queries that returned an error from any phase.",
            Counter::PtxCacheHits => "PTX module cache lookups that hit a warm entry.",
            Counter::PtxCacheMisses => "PTX module cache lookups that had to compile or load.",
            Counter::GpuLaunchesTotal => "Kernel grid launches issued to the driver.",
            Counter::HostFallbacksTotal => {
                "Queries or sub-plans that fell back to the host execution path."
            }
        }
    }
}

/// Named query phases that own a latency histogram.
///
/// Each name matches the corresponding tracing span in
/// [`observability`](crate::observability), but the two surfaces are *not* a
/// one-to-one mapping: the span catalogue is broader (it also covers the
/// device-side phases), whereas this enum lists only the phases that feed a
/// histogram. [`Phase::ALL`] iterates in declaration order.
///
/// Only phases whose latency is *actually observed* on the hot path live here
/// (`exec::engine` times `Parse` / `Plan` / `Lower` / `Materialize`). The
/// device-side phases — `codegen`, `ptx_load`, `launch`, `transfer` — still
/// appear as tracing spans (see the catalogue in
/// [`observability`](crate::observability)) but never had a histogram
/// observation wired at their kernel call sites, so they are intentionally
/// absent from this enum rather than exposing a permanently empty histogram via
/// [`snapshot()`]. Re-add a variant only together with the
/// `observe_duration(Phase::_, …)` call at its source site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Phase {
    /// SQL text → `LogicalPlan`.
    Parse = 0,
    /// Logical-plan rewrite pipeline.
    Plan = 1,
    /// `LogicalPlan` → `PhysicalPlan`.
    Lower = 2,
    /// Arrow array packing of results.
    Materialize = 3,
}

impl Phase {
    /// All phases in declaration / snapshot order.
    pub const ALL: [Phase; 4] = [Phase::Parse, Phase::Plan, Phase::Lower, Phase::Materialize];

    /// Stable snake_case name, matching the corresponding tracing span.
    pub const fn as_str(self) -> &'static str {
        match self {
            Phase::Parse => "parse",
            Phase::Plan => "plan",
            Phase::Lower => "lower",
            Phase::Materialize => "materialize",
        }
    }

    /// One-line description for the Prometheus `# HELP` line.
    pub const fn help(self) -> &'static str {
        match self {
            Phase::Parse => "SQL text to LogicalPlan latency.",
            Phase::Plan => "Logical-plan rewrite pipeline latency.",
            Phase::Lower => "LogicalPlan to PhysicalPlan lowering latency.",
            Phase::Materialize => "Arrow array packing (result materialization) latency.",
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

/// Metric-name prefix applied to every series rendered by
/// [`render_prometheus`]. Kept as one constant so the namespace is changed in
/// exactly one place.
const PROM_PREFIX: &str = "craton_";

/// Fully-qualified name of the per-phase latency histogram (without the
/// `_bucket` / `_count` / `_sum` suffixes Prometheus appends to a histogram).
/// Latency is reported in **seconds** per Prometheus base-unit convention,
/// converted from the registry's internal micro-second accounting.
const PROM_PHASE_HISTOGRAM: &str = "craton_phase_latency_seconds";

/// Render `snap` as a Prometheus text-exposition (`text/plain; version=0.0.4`)
/// document.
///
/// This is the production replacement for the old hand-rolled scraping
/// pseudocode: pure host code, zero new dependencies, always available (no
/// cargo feature gate). It renders **exactly** what the snapshot holds — every
/// live [`Counter`] and every [`Phase`] histogram — and invents nothing that is
/// not recorded.
///
/// ## Shape
///
/// Each counter emits the canonical three lines:
///
/// ```text
/// # HELP craton_queries_total Total queries accepted for execution (success or failure).
/// # TYPE craton_queries_total counter
/// craton_queries_total 7
/// ```
///
/// Each phase histogram emits a single `# HELP`/`# TYPE … histogram` header
/// (the metric is shared across phases, distinguished by a `phase` label),
/// followed, for every phase, by cumulative `_bucket{phase,le}` series — bucket
/// upper bounds converted µs → seconds, terminating in `le="+Inf"` — plus the
/// conventional `_sum` (seconds) and `_count` companions:
///
/// ```text
/// # HELP craton_phase_latency_seconds Per-phase pipeline latency in seconds.
/// # TYPE craton_phase_latency_seconds histogram
/// craton_phase_latency_seconds_bucket{phase="parse",le="0.000001"} 0
/// ...
/// craton_phase_latency_seconds_bucket{phase="parse",le="+Inf"} 0
/// craton_phase_latency_seconds_sum{phase="parse"} 0
/// craton_phase_latency_seconds_count{phase="parse"} 0
/// ```
///
/// The output always ends with a trailing newline, as required by the
/// exposition format.
pub fn render_prometheus(snap: &MetricsSnapshot) -> String {
    let mut out = String::new();

    // --- Counters -----------------------------------------------------------
    for &c in Counter::ALL.iter() {
        let name = c.as_str();
        let value = snap.counter(c);
        out.push_str("# HELP ");
        out.push_str(PROM_PREFIX);
        out.push_str(name);
        out.push(' ');
        out.push_str(c.help());
        out.push('\n');

        out.push_str("# TYPE ");
        out.push_str(PROM_PREFIX);
        out.push_str(name);
        out.push_str(" counter\n");

        out.push_str(PROM_PREFIX);
        out.push_str(name);
        out.push(' ');
        push_u64(&mut out, value);
        out.push('\n');
    }

    // --- Phase histograms ---------------------------------------------------
    //
    // One metric (`craton_phase_latency_seconds`) keyed by a `phase` label, so
    // the HELP/TYPE header is emitted once and every phase contributes its own
    // labelled bucket/sum/count series. This is the idiomatic way to expose a
    // family of same-shape histograms in Prometheus.
    out.push_str("# HELP ");
    out.push_str(PROM_PHASE_HISTOGRAM);
    out.push_str(" Per-phase pipeline latency in seconds.\n");
    out.push_str("# TYPE ");
    out.push_str(PROM_PHASE_HISTOGRAM);
    out.push_str(" histogram\n");

    for h in snap.histograms() {
        let phase = h.phase.as_str();

        // Cumulative buckets: Prometheus `_bucket{le}` series are monotone
        // non-decreasing, so we accumulate the per-bucket (non-cumulative)
        // counts the snapshot stores.
        let mut cumulative: u64 = 0;
        for (upper_us, count) in h.buckets() {
            cumulative += count;
            out.push_str(PROM_PHASE_HISTOGRAM);
            out.push_str("_bucket{phase=\"");
            out.push_str(phase);
            out.push_str("\",le=\"");
            push_seconds(&mut out, upper_us);
            out.push_str("\"} ");
            push_u64(&mut out, cumulative);
            out.push('\n');
        }
        // The final, mandatory `+Inf` bucket equals the total observation
        // count. The last finite bucket is the overflow class, so its
        // cumulative total already equals `h.count`; emitting `+Inf`
        // explicitly is required by the format regardless.
        out.push_str(PROM_PHASE_HISTOGRAM);
        out.push_str("_bucket{phase=\"");
        out.push_str(phase);
        out.push_str("\",le=\"+Inf\"} ");
        push_u64(&mut out, h.count);
        out.push('\n');

        // `_sum` in seconds, `_count` as the raw observation total.
        out.push_str(PROM_PHASE_HISTOGRAM);
        out.push_str("_sum{phase=\"");
        out.push_str(phase);
        out.push_str("\"} ");
        push_seconds(&mut out, h.sum_micros);
        out.push('\n');

        out.push_str(PROM_PHASE_HISTOGRAM);
        out.push_str("_count{phase=\"");
        out.push_str(phase);
        out.push_str("\"} ");
        push_u64(&mut out, h.count);
        out.push('\n');
    }

    out
}

/// Append the decimal form of `v` to `out` without an intermediate allocation
/// beyond the formatter's own.
#[inline]
fn push_u64(out: &mut String, v: u64) {
    use std::fmt::Write as _;
    // Writing to a `String` is infallible.
    let _ = write!(out, "{v}");
}

/// Append `micros` micro-seconds as a Prometheus-friendly seconds value.
///
/// Uses Rust's `f64` `Display` (`{}`), which emits fixed-point decimal — never
/// scientific notation — so values render as `0.000001`, `0.001048576`,
/// `8.388608`, all of which the Prometheus text parser accepts. Zero renders as
/// `0`.
#[inline]
fn push_seconds(out: &mut String, micros: u64) {
    use std::fmt::Write as _;
    let seconds = micros as f64 / 1e6;
    let _ = write!(out, "{seconds}");
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
        assert_eq!(Counter::ALL.len(), 6);
        assert_eq!(Counter::QueriesTotal.as_str(), "queries_total");
        assert_eq!(Counter::HostFallbacksTotal.as_str(), "host_fallbacks_total");
        // `as usize` discriminants line up with `ALL` ordering.
        for (i, c) in Counter::ALL.iter().enumerate() {
            assert_eq!(*c as usize, i);
        }
    }

    #[test]
    fn phase_names_match_variants() {
        assert_eq!(Phase::ALL.len(), 4);
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

        m.add(Counter::GpuLaunchesTotal, 4096);
        m.add(Counter::GpuLaunchesTotal, 100);
        assert_eq!(m.counter(Counter::GpuLaunchesTotal).get(), 4196);

        // Independent counters do not bleed into each other.
        assert_eq!(m.counter(Counter::HostFallbacksTotal).get(), 0);
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
            assert_eq!(
                bucket_index(ub),
                i,
                "upper bound {ub} should map to bucket {i}"
            );
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
        m.observe(Phase::Lower, 16); // bucket 4 (covers (8,16])
        m.observe(Phase::Lower, 17); // bucket 5
        m.observe(Phase::Materialize, 2); // bucket 1

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
        let lower = snap.histogram(Phase::Lower);
        assert_eq!(lower.count, 2);
        assert_eq!(lower.sum_micros, 33);
        assert_eq!(lower.buckets[4], 1);
        assert_eq!(lower.buckets[5], 1);

        let materialize = snap.histogram(Phase::Materialize);
        assert_eq!(materialize.count, 1);
        assert_eq!(materialize.buckets[1], 1);

        // Untouched phase is all zeros.
        let parse = snap.histogram(Phase::Parse);
        assert_eq!(parse.count, 0);
        assert!(parse.buckets.iter().all(|&b| b == 0));

        // buckets() iterator: cumulative reconstruction equals count.
        let total: u64 = lower.buckets().map(|(_, n)| n).sum();
        assert_eq!(total, lower.count);
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

    // ---- Prometheus exposition rendering ----------------------------------

    /// Helper: collect non-comment metric sample lines (drop `# HELP`/`# TYPE`).
    fn sample_lines(text: &str) -> Vec<&str> {
        text.lines().filter(|l| !l.starts_with('#')).collect()
    }

    #[test]
    fn render_prometheus_empty_snapshot_is_well_formed() {
        let snap = Metrics::default().snapshot();
        let out = render_prometheus(&snap);

        // Trailing newline is mandatory in the exposition format.
        assert!(
            out.ends_with('\n'),
            "exposition body must end with a newline"
        );
        assert!(!out.is_empty());

        // Every counter is present as a fully-qualified three-line block, all
        // at value 0.
        for c in Counter::ALL {
            let name = format!("craton_{}", c.as_str());
            assert!(
                out.contains(&format!("# HELP {name} ")),
                "missing HELP for {name}"
            );
            assert!(
                out.contains(&format!("# TYPE {name} counter\n")),
                "missing TYPE for {name}"
            );
            assert!(
                out.contains(&format!("\n{name} 0\n")) || out.starts_with(&format!("{name} 0\n")),
                "missing zero sample line for {name}\n---\n{out}"
            );
        }

        // The histogram header appears exactly once.
        assert_eq!(
            out.matches("# TYPE craton_phase_latency_seconds histogram\n")
                .count(),
            1,
            "histogram TYPE header must be emitted exactly once"
        );

        // Each phase contributes a +Inf bucket, a sum and a count, all 0.
        for p in Phase::ALL {
            let ph = p.as_str();
            assert!(out.contains(&format!(
                "craton_phase_latency_seconds_bucket{{phase=\"{ph}\",le=\"+Inf\"}} 0\n"
            )));
            assert!(out.contains(&format!(
                "craton_phase_latency_seconds_count{{phase=\"{ph}\"}} 0\n"
            )));
            assert!(out.contains(&format!(
                "craton_phase_latency_seconds_sum{{phase=\"{ph}\"}} 0\n"
            )));
        }
    }

    #[test]
    fn render_prometheus_counter_values_and_help() {
        let m = Metrics::default();
        m.add(Counter::QueriesTotal, 7);
        m.add(Counter::QueriesFailed, 2);
        m.inc(Counter::PtxCacheHits);
        m.add(Counter::GpuLaunchesTotal, 4196);

        let out = render_prometheus(&m.snapshot());

        // Exact HELP + TYPE + sample triple for a representative counter.
        assert!(out.contains(
            "# HELP craton_queries_total Total queries accepted for execution (success or failure).\n\
             # TYPE craton_queries_total counter\n\
             craton_queries_total 7\n"
        ));
        // Other recorded values render verbatim.
        assert!(out.contains("craton_queries_failed 2\n"));
        assert!(out.contains("craton_ptx_cache_hits 1\n"));
        assert!(out.contains("craton_gpu_launches_total 4196\n"));
        // Untouched counter still appears at 0.
        assert!(out.contains("craton_host_fallbacks_total 0\n"));

        // Counters are emitted in declaration order, before any histogram.
        let q_total = out.find("craton_queries_total 7").unwrap();
        let hist = out.find("craton_phase_latency_seconds").unwrap();
        assert!(q_total < hist, "counters must precede histograms");
    }

    #[test]
    fn render_prometheus_histogram_cumulative_buckets_and_seconds() {
        let m = Metrics::default();
        // 16 µs -> bucket 4 (covers (8,16]); 17 µs -> bucket 5.
        m.observe(Phase::Lower, 16);
        m.observe(Phase::Lower, 17);

        let out = render_prometheus(&m.snapshot());

        // `_count` and `_sum` (seconds) for the touched phase.
        assert!(out.contains("craton_phase_latency_seconds_count{phase=\"lower\"} 2\n"));
        // 33 µs = 0.000033 s.
        assert!(out.contains("craton_phase_latency_seconds_sum{phase=\"lower\"} 0.000033\n"));

        // Buckets are cumulative and monotone non-decreasing. Reconstruct the
        // (le, cumulative-count) series for the `lower` phase and check it.
        let lower_buckets: Vec<(String, u64)> = out
            .lines()
            .filter(|l| l.starts_with("craton_phase_latency_seconds_bucket{phase=\"lower\","))
            .map(|l| {
                let le_start = l.find("le=\"").unwrap() + 4;
                let le_end = l[le_start..].find('"').unwrap() + le_start;
                let le = l[le_start..le_end].to_string();
                let count: u64 = l.rsplit(' ').next().unwrap().parse().unwrap();
                (le, count)
            })
            .collect();

        // Monotone non-decreasing cumulative counts.
        let mut prev = 0u64;
        for (_, c) in &lower_buckets {
            assert!(*c >= prev, "bucket counts must be cumulative/monotone");
            prev = *c;
        }
        // Final +Inf bucket equals total count (2).
        let inf = lower_buckets.last().unwrap();
        assert_eq!(inf.0, "+Inf");
        assert_eq!(inf.1, 2);

        // The first observation (16µs, bucket 4 -> le 16µs = 0.000016 s) is the
        // first le at which the cumulative count reaches 1.
        let first_nonzero = lower_buckets.iter().find(|(_, c)| *c >= 1).unwrap();
        assert_eq!(first_nonzero.0, "0.000016");
        assert_eq!(first_nonzero.1, 1);

        // Every finite bucket has 24 entries + 1 for +Inf.
        assert_eq!(lower_buckets.len(), HISTOGRAM_BUCKETS + 1);
    }

    #[test]
    fn render_prometheus_names_are_valid() {
        // Render a fully-populated snapshot and validate every metric name and
        // label against the Prometheus name grammar.
        let m = Metrics::default();
        for c in Counter::ALL {
            m.add(c, 3);
        }
        for p in Phase::ALL {
            m.observe(p, 5);
        }
        let out = render_prometheus(&m.snapshot());

        // Metric name: [a-zA-Z_][a-zA-Z0-9_]* (suffixes/labels stripped).
        fn valid_metric_name(name: &str) -> bool {
            let mut chars = name.chars();
            match chars.next() {
                Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
                _ => return false,
            }
            chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        }

        for line in sample_lines(&out) {
            assert!(!line.is_empty());
            // Series := <name>[{labels}] <value>
            let (series, value) = line.rsplit_once(' ').expect("sample line has a value");
            assert!(!value.is_empty());

            let metric = match series.split_once('{') {
                Some((name, labels)) => {
                    assert!(labels.ends_with('}'), "label set must close: {line}");
                    name
                }
                None => series,
            };
            assert!(
                valid_metric_name(metric),
                "invalid metric name {metric:?} in line {line:?}"
            );
            assert!(
                metric.starts_with("craton_"),
                "metric {metric:?} must carry the craton_ prefix"
            );
        }
    }

    #[test]
    fn render_prometheus_renders_only_recorded_metrics() {
        // No surprise series: the only metric families are the six counters and
        // the single phase-latency histogram. Count distinct `# TYPE` headers.
        let out = render_prometheus(&Metrics::default().snapshot());
        let type_lines: Vec<&str> = out.lines().filter(|l| l.starts_with("# TYPE ")).collect();
        // 6 counters + 1 histogram family.
        assert_eq!(type_lines.len(), Counter::ALL.len() + 1);
        assert_eq!(
            type_lines
                .iter()
                .filter(|l| l.ends_with(" histogram"))
                .count(),
            1
        );
        assert_eq!(
            type_lines
                .iter()
                .filter(|l| l.ends_with(" counter"))
                .count(),
            Counter::ALL.len()
        );
    }
}
