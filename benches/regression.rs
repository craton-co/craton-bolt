// SPDX-License-Identifier: Apache-2.0

//! Craton Bolt — baseline regression benchmark suite.
//!
//! v0.6 / M6 ("production discipline") introduces a small, *stable* set of
//! Criterion benchmarks whose median wall-time we track release-over-release
//! as a regression tripwire. This file is intentionally **narrow** and
//! **headless**: only the CPU-bound stages of the query pipeline
//! (parse + logical plan + physical lowering + PTX code generation) are
//! exercised, and *no* CUDA driver call is ever issued. That property is what
//! lets the suite run in CI on a vanilla Linux runner with **no GPU
//! attached**: the crate is built with the `cuda-stub` feature, the FFI
//! shims in `src/cuda/cuda_sys.rs` are stubbed to return `CUDA_ERROR_STUB`,
//! and we deliberately never hit them.
//!
//! Run locally
//! -----------
//! ```text
//! cargo bench --bench regression --features cuda-stub
//! ```
//!
//! Run in CI
//! ---------
//! The CI job mirrors the local invocation. Because `cuda-stub` strips the
//! `#[link(name = "cuda")]` block, no CUDA toolkit / driver is required on
//! the runner — the bench builds and runs on any host with a stable Rust
//! toolchain.
//!
//! Regression threshold convention
//! -------------------------------
//! We treat a **>5% slowdown** in the median wall-time (vs. the committed
//! baseline for the same bench id) as a regression that must be either
//! justified or fixed before merge. Criterion's own change-detection at
//! default sample-size already flags shifts of this magnitude with ample
//! statistical confidence on a quiet runner.
//!
//! Opt-in regression guard (self-contained, no CI / data-file deps)
//! ----------------------------------------------------------------
//! Criterion benches are a poor place to *hard-fail* CI (noisy, and
//! `cargo bench` isn't wired into CI here), so the guard below is **opt-in**
//! and **self-contained**: it lives entirely in this file, is driven by env
//! vars, and is a complete no-op unless one of them is set. The default
//! `cargo bench --bench regression` run is byte-for-byte unchanged — it
//! still just runs the three Criterion groups.
//!
//! The guard does NOT reuse Criterion's measurement loop (that would couple
//! us to `target/criterion/.../estimates.json` internals). Instead it runs a
//! small, independent timing pass: each `(stage, query)` operation is invoked
//! over a fixed iteration budget with `std::time::Instant`, and the **median**
//! per-iteration nanosecond cost is recorded under the *same* bench id
//! Criterion uses (`regression/<stage>/<query>`). That keeps a single key
//! diff-able across both the Criterion HTML report and this guard.
//!
//! Env vars
//! ~~~~~~~~
//! * `BOLT_REGRESSION_WRITE_BASELINE=<path>` — run the timing pass and write
//!   the current medians to `<path>` as JSON (`{ "<bench_id>": <median_ns>, ... }`).
//!   Use this once on a known-good commit / quiet machine to mint a baseline.
//! * `BOLT_REGRESSION_BASELINE=<path>` — run the timing pass, read the JSON
//!   baseline from `<path>`, and print a `PASS` / `REGRESSION` line per bench
//!   id plus a summary. A missing or garbled baseline prints a clear message
//!   and is skipped — it never panics or aborts the run.
//! * `BOLT_REGRESSION_THRESHOLD_PCT=<f>` — regression threshold in percent
//!   (default `5`). A bench is flagged when
//!   `current_ns > baseline_ns * (1 + pct/100)`.
//!
//! If both write- and compare- vars are set, the write happens first, then the
//! compare (typically you'd set only one). When neither is set, none of this
//! runs.
//!
//! Generating then enforcing a baseline (maintainer workflow)
//! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//! ```text
//! # 1. Mint a baseline on a known-good commit / quiet machine:
//! BOLT_REGRESSION_WRITE_BASELINE=regression-baseline.json \
//!     cargo bench --bench regression --features cuda-stub
//!
//! # 2. Later, enforce it (e.g. locally before merge, or in an opt-in job):
//! BOLT_REGRESSION_BASELINE=regression-baseline.json \
//! BOLT_REGRESSION_THRESHOLD_PCT=5 \
//!     cargo bench --bench regression --features cuda-stub
//! ```
//! The compare pass only *reports* (prints `REGRESSION` lines and a non-zero
//! count in the summary); it deliberately does not call `process::exit`, so
//! the surrounding Criterion run still completes. A maintainer / wrapper
//! script that wants a hard gate can grep the summary line for a non-zero
//! regression count.
//!
//! JSON handling: the baseline format is a flat object of string → number, so
//! this file hand-rolls a tiny tolerant parser / formatter rather than pulling
//! in `serde_json` (which is not a dependency and would need a Cargo.toml
//! edit). The format is fully controlled by this file.
//!
//! Workload
//! --------
//! All three queries run against an in-memory table of 100_000 rows with
//! three columns (`region`, `price`, `tax`). The chosen queries are the
//! smallest representatives of three distinct execution shapes:
//!
//! 1. **Scalar aggregate** — `SELECT COUNT(*), SUM(price), AVG(price) FROM t`
//! 2. **GROUP BY**         — `SELECT region, SUM(price) FROM t GROUP BY region`
//!                           (10 distinct groups)
//! 3. **Filter**           — `SELECT price FROM t WHERE price > 50`
//!
//! Each query is timed at three pipeline stages: `parse`, `lower`, and
//! `ptx_gen`. The bench *ids* are stable (`regression/<stage>/<query>` —
//! the Criterion group is `regression/<stage>` and the per-query parameter
//! is the query name) so future tooling — including the opt-in guard below —
//! can diff a single key over time.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use craton_bolt::jit::compile_ptx;
use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Field, MemTableProvider, PhysicalPlan, Schema,
};

/// Row count for the in-memory fixture. Small enough that the bench is
/// dominated by codegen / planning cost (the bits we actually exercise on a
/// `cuda-stub` host) and finishes quickly in CI; large enough to stay
/// representative of the constants planners care about (cardinality
/// estimation, group count, etc.).
const BENCH_ROWS: usize = 100_000;

/// Distinct group cardinality for the GROUP BY workload.
const NUM_GROUPS: i32 = 10;

// --- Queries ----------------------------------------------------------------
//
// These three SQL strings ARE the regression contract. Changing them changes
// the bench id and resets the baseline — only do so deliberately, with a
// commit message that calls it out.

/// (1) Scalar aggregate — one row out, three aggregates over `BENCH_ROWS` rows.
const Q_SCALAR_AGG: &str = "SELECT COUNT(*), SUM(price), AVG(price) FROM t";

/// (2) GROUP BY — `NUM_GROUPS` rows out, one SUM per group.
const Q_GROUP_BY: &str = "SELECT region, SUM(price) FROM t GROUP BY region";

/// (3) Filter — selective projection; ~half the input rows survive.
const Q_FILTER: &str = "SELECT price FROM t WHERE price > 50";

// --- Fixture ---------------------------------------------------------------

fn schema() -> Schema {
    Schema::new(vec![
        Field {
            name: "region".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "price".into(),
            dtype: DataType::Float64,
            nullable: false,
        },
        Field {
            name: "tax".into(),
            dtype: DataType::Float64,
            nullable: false,
        },
    ])
}

fn provider() -> MemTableProvider {
    // Note: the planner only needs the schema (and a row-count hint, where
    // applicable); we deliberately do not register an Arrow `RecordBatch`
    // payload because the bench never reaches the execution stage on a
    // `cuda-stub` host. `BENCH_ROWS` / `NUM_GROUPS` are documented above
    // for the day someone wires an Arrow-backed provider in.
    let _ = BENCH_ROWS;
    let _ = NUM_GROUPS;
    MemTableProvider::new().with_table("t", schema())
}

// --- Benches ---------------------------------------------------------------

/// Time SQL → `LogicalPlan` (parse + bind + validate) for each of the three
/// regression queries.
fn bench_parse(c: &mut Criterion) {
    let p = provider();
    let mut g = c.benchmark_group("regression/parse");
    for (name, sql) in [
        ("scalar_agg", Q_SCALAR_AGG),
        ("group_by", Q_GROUP_BY),
        ("filter", Q_FILTER),
    ] {
        g.bench_with_input(BenchmarkId::from_parameter(name), &sql, |b, &sql| {
            b.iter(|| {
                let plan = parse_sql(black_box(sql), &p).unwrap();
                black_box(plan)
            })
        });
    }
    g.finish();
}

/// Time `LogicalPlan` → `PhysicalPlan` (lowering) for each query.
fn bench_lower(c: &mut Criterion) {
    let p = provider();
    let mut g = c.benchmark_group("regression/lower");
    for (name, sql) in [
        ("scalar_agg", Q_SCALAR_AGG),
        ("group_by", Q_GROUP_BY),
        ("filter", Q_FILTER),
    ] {
        let plan = parse_sql(sql, &p).unwrap();
        g.bench_with_input(BenchmarkId::from_parameter(name), &plan, |b, plan| {
            b.iter(|| {
                let phys = lower_physical(black_box(plan)).unwrap();
                black_box(phys)
            })
        });
    }
    g.finish();
}

/// Time `PhysicalPlan` → PTX (codegen) for the filter query, which is the
/// only one of the three whose physical plan is a leaf-`Projection` and
/// therefore exposes a single `kernel` we can hand to `compile_ptx`.
/// Aggregate/GROUP BY plans are multi-stage and don't fit a single-kernel
/// codegen call; for those, the regression signal lives in `bench_parse`
/// and `bench_lower` above.
fn bench_ptx_gen(c: &mut Criterion) {
    let p = provider();
    let mut g = c.benchmark_group("regression/ptx_gen");
    for (name, sql) in [("filter", Q_FILTER)] {
        let plan = parse_sql(sql, &p).unwrap();
        let phys = lower_physical(&plan).unwrap();
        let kernel = match phys {
            PhysicalPlan::Projection { kernel, .. } => kernel,
            // Defensive: if a future planner change reshapes the filter
            // plan into something other than a leaf Projection, the bench
            // should be re-thought rather than silently skipped.
            other => panic!(
                "regression bench expected a leaf Projection plan for `{}`, got {:?}",
                sql, other
            ),
        };
        g.bench_with_input(BenchmarkId::from_parameter(name), &kernel, |b, kernel| {
            b.iter(|| {
                let ptx = compile_ptx(black_box(kernel), "bolt_regression_kernel").unwrap();
                black_box(ptx)
            })
        });
    }
    g.finish();
}

// --- Opt-in regression guard ------------------------------------------------
//
// Everything below is a no-op unless `BOLT_REGRESSION_WRITE_BASELINE` or
// `BOLT_REGRESSION_BASELINE` is set in the environment. It is intentionally
// independent of Criterion's measurement loop (see the module header): a small
// `Instant`-based timing pass recorded under the same `regression/<stage>/<query>`
// bench ids, then compared against / written to a flat JSON baseline.

use std::collections::BTreeMap;
use std::time::Instant;

/// Env var: write the current medians to this path as JSON.
const ENV_WRITE_BASELINE: &str = "BOLT_REGRESSION_WRITE_BASELINE";
/// Env var: read a JSON baseline from this path and compare against it.
const ENV_COMPARE_BASELINE: &str = "BOLT_REGRESSION_BASELINE";
/// Env var: regression threshold, in percent. Defaults to `DEFAULT_THRESHOLD_PCT`.
const ENV_THRESHOLD_PCT: &str = "BOLT_REGRESSION_THRESHOLD_PCT";

/// Default regression threshold: a >5% slowdown is flagged.
const DEFAULT_THRESHOLD_PCT: f64 = 5.0;

/// Iterations per `(stage, query)` operation in the manual timing pass. Large
/// enough to get a stable median on the sub-microsecond-to-tens-of-microseconds
/// operations this suite exercises, small enough to stay well under a second of
/// total wall-time across all bench ids.
const GUARD_ITERS: usize = 2_000;

/// Build the `regression/<stage>/<query>` bench id used as the JSON key. This
/// MUST match the Criterion id (`group_name` + `BenchmarkId::from_parameter`)
/// so a single key is diff-able across the Criterion report and this guard.
fn bench_id(stage: &str, query: &str) -> String {
    format!("regression/{stage}/{query}")
}

/// Run a closure `GUARD_ITERS` times and return the **median** per-iteration
/// cost in nanoseconds. We time each iteration individually and take the
/// median so a single scheduler hiccup does not skew the recorded number the
/// way a mean would. `black_box` keeps the optimiser from hoisting the work.
fn median_ns<T>(mut op: impl FnMut() -> T) -> f64 {
    let mut samples: Vec<u128> = Vec::with_capacity(GUARD_ITERS);
    for _ in 0..GUARD_ITERS {
        let start = Instant::now();
        let out = op();
        let elapsed = start.elapsed().as_nanos();
        criterion::black_box(out);
        samples.push(elapsed);
    }
    samples.sort_unstable();
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        samples[n / 2] as f64
    } else {
        (samples[n / 2 - 1] as f64 + samples[n / 2] as f64) / 2.0
    }
}

/// Measure every `(stage, query)` bench id with the manual timing pass and
/// return an ordered `bench_id -> median_ns` map. Mirrors exactly the work
/// the three Criterion `bench_*` functions time, so the keys line up.
fn collect_medians() -> BTreeMap<String, f64> {
    let p = provider();
    let queries = [
        ("scalar_agg", Q_SCALAR_AGG),
        ("group_by", Q_GROUP_BY),
        ("filter", Q_FILTER),
    ];

    let mut out: BTreeMap<String, f64> = BTreeMap::new();

    // parse: SQL -> LogicalPlan
    for (name, sql) in queries {
        let ns = median_ns(|| parse_sql(criterion::black_box(sql), &p).unwrap());
        out.insert(bench_id("parse", name), ns);
    }

    // lower: LogicalPlan -> PhysicalPlan
    for (name, sql) in queries {
        let plan = parse_sql(sql, &p).unwrap();
        let ns = median_ns(|| lower_physical(criterion::black_box(&plan)).unwrap());
        out.insert(bench_id("lower", name), ns);
    }

    // ptx_gen: PhysicalPlan -> PTX (filter only — see `bench_ptx_gen`).
    {
        let plan = parse_sql(Q_FILTER, &p).unwrap();
        let phys = lower_physical(&plan).unwrap();
        if let PhysicalPlan::Projection { kernel, .. } = phys {
            let ns = median_ns(|| {
                compile_ptx(criterion::black_box(&kernel), "bolt_regression_kernel").unwrap()
            });
            out.insert(bench_id("ptx_gen", "filter"), ns);
        } else {
            // Match `bench_ptx_gen`'s expectation but don't abort the guard;
            // simply omit the key (the compare pass reports it as missing).
            eprintln!(
                "[regression-guard] WARN: filter plan is not a leaf Projection; \
                 skipping ptx_gen/filter"
            );
        }
    }

    out
}

// --- Minimal JSON for a flat string -> number object ------------------------
//
// The baseline format is fully controlled by this file: a flat JSON object
// mapping bench-id strings to numbers, e.g.
//   { "regression/parse/filter": 1234.5, "regression/lower/filter": 67.0 }
// That is trivial enough to format and parse by hand, which avoids taking a
// `serde_json` dependency (not in Cargo.toml, and out of scope to add).

/// Serialise an ordered `bench_id -> median_ns` map to a pretty JSON object.
fn to_json(medians: &BTreeMap<String, f64>) -> String {
    let mut s = String::from("{\n");
    for (i, (k, v)) in medians.iter().enumerate() {
        let comma = if i + 1 < medians.len() { "," } else { "" };
        // Keys are our own ASCII bench ids ('/', alnum, '_') — no escaping
        // needed, but go through the escaper for safety/future-proofing.
        s.push_str(&format!(
            "  \"{}\": {}{}\n",
            json_escape(k),
            fmt_num(*v),
            comma
        ));
    }
    s.push_str("}\n");
    s
}

/// Format a number without scientific notation and without a trailing `.0`
/// noise tail, so the JSON stays human-diffable.
fn fmt_num(v: f64) -> String {
    // Medians are nanoseconds; one decimal place of precision is plenty and
    // keeps the file stable. Round to 0.1 ns.
    let rounded = (v * 10.0).round() / 10.0;
    if rounded.fract() == 0.0 {
        format!("{}", rounded as i64)
    } else {
        format!("{rounded}")
    }
}

/// Escape a string for embedding in a JSON double-quoted scalar. Only the
/// minimal set we could ever emit is handled (bench ids never contain these,
/// but be defensive).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

/// Tolerant hand-rolled parser for a flat JSON object of string -> number.
/// Returns `Err(msg)` on anything it can't make sense of; the caller treats
/// that as "no usable baseline" and skips the compare (never panics).
///
/// Accepts insignificant whitespace, `"key": <num>` pairs, and trailing
/// commas. It does NOT support nested objects/arrays, string values, or
/// escapes inside keys other than the ones `json_escape` emits — which is all
/// this format ever produces.
fn parse_flat_json(input: &str) -> Result<BTreeMap<String, f64>, String> {
    let bytes = input.as_bytes();
    let mut i = 0usize;
    let n = bytes.len();
    let mut map = BTreeMap::new();

    fn skip_ws(b: &[u8], i: &mut usize) {
        while *i < b.len() && (b[*i] as char).is_whitespace() {
            *i += 1;
        }
    }

    skip_ws(bytes, &mut i);
    if i >= n || bytes[i] != b'{' {
        return Err("expected '{' at start of object".into());
    }
    i += 1;

    loop {
        skip_ws(bytes, &mut i);
        if i >= n {
            return Err("unexpected end of input (unterminated object)".into());
        }
        if bytes[i] == b'}' {
            i += 1;
            break;
        }

        // Parse a JSON string key.
        if bytes[i] != b'"' {
            return Err(format!("expected '\"' to start key at byte {i}"));
        }
        i += 1;
        let mut key = String::new();
        loop {
            if i >= n {
                return Err("unterminated string key".into());
            }
            let c = bytes[i];
            if c == b'"' {
                i += 1;
                break;
            }
            if c == b'\\' {
                i += 1;
                if i >= n {
                    return Err("dangling escape in key".into());
                }
                match bytes[i] {
                    b'"' => key.push('"'),
                    b'\\' => key.push('\\'),
                    b'/' => key.push('/'),
                    b'n' => key.push('\n'),
                    b'r' => key.push('\r'),
                    b't' => key.push('\t'),
                    other => key.push(other as char),
                }
                i += 1;
            } else {
                // Keys are ASCII in our format; pass bytes through.
                key.push(c as char);
                i += 1;
            }
        }

        skip_ws(bytes, &mut i);
        if i >= n || bytes[i] != b':' {
            return Err(format!("expected ':' after key '{key}'"));
        }
        i += 1;
        skip_ws(bytes, &mut i);

        // Parse a number value: [-+]?digits[.digits][eE[-+]?digits]
        let start = i;
        while i < n {
            let c = bytes[i] as char;
            if c.is_ascii_digit() || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' {
                i += 1;
            } else {
                break;
            }
        }
        if i == start {
            return Err(format!("expected a number value for key '{key}'"));
        }
        let num_str = &input[start..i];
        let value: f64 = num_str
            .parse()
            .map_err(|_| format!("invalid number '{num_str}' for key '{key}'"))?;
        map.insert(key, value);

        skip_ws(bytes, &mut i);
        if i < n && bytes[i] == b',' {
            i += 1;
            continue;
        }
        // Either a closing brace next, or (tolerantly) end — loop will catch.
    }

    Ok(map)
}

/// Read the threshold percentage from the env, falling back to the default on
/// an unset or unparseable value.
fn threshold_pct() -> f64 {
    match std::env::var(ENV_THRESHOLD_PCT) {
        Ok(s) => match s.trim().parse::<f64>() {
            Ok(v) if v >= 0.0 => v,
            _ => {
                eprintln!(
                    "[regression-guard] WARN: {ENV_THRESHOLD_PCT}={s:?} is not a \
                     non-negative number; using default {DEFAULT_THRESHOLD_PCT}%"
                );
                DEFAULT_THRESHOLD_PCT
            }
        },
        Err(_) => DEFAULT_THRESHOLD_PCT,
    }
}

/// Write `medians` to `path` as JSON, logging the outcome.
fn write_baseline(path: &str, medians: &BTreeMap<String, f64>) {
    match std::fs::write(path, to_json(medians)) {
        Ok(()) => eprintln!(
            "[regression-guard] wrote baseline ({} ids) to {path}",
            medians.len()
        ),
        Err(e) => eprintln!("[regression-guard] ERROR: could not write baseline to {path}: {e}"),
    }
}

/// Compare `current` medians against the baseline JSON at `path`. Prints a
/// PASS / REGRESSION line per bench id and a summary. Never panics: a missing
/// or garbled baseline prints a message and returns.
fn compare_baseline(path: &str, current: &BTreeMap<String, f64>) {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "[regression-guard] could not read baseline {path}: {e} — skipping compare. \
                 (Generate one with {ENV_WRITE_BASELINE}={path}.)"
            );
            return;
        }
    };

    let baseline = match parse_flat_json(&raw) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "[regression-guard] baseline {path} is not valid baseline JSON: {e} — \
                 skipping compare."
            );
            return;
        }
    };

    if baseline.is_empty() {
        eprintln!("[regression-guard] baseline {path} has no entries — skipping compare.");
        return;
    }

    let pct = threshold_pct();
    let factor = 1.0 + pct / 100.0;

    eprintln!("[regression-guard] comparing against {path} (threshold {pct}%)");

    let mut regressions = 0usize;
    let mut compared = 0usize;
    let mut missing_in_baseline = 0usize;

    for (id, &cur) in current {
        match baseline.get(id) {
            Some(&base) if base > 0.0 => {
                compared += 1;
                let limit = base * factor;
                let delta_pct = (cur - base) / base * 100.0;
                if cur > limit {
                    regressions += 1;
                    eprintln!(
                        "[regression-guard] REGRESSION {id}: {cur:.1} ns vs baseline {base:.1} ns \
                         ({delta_pct:+.1}%, > {pct:.1}% threshold)"
                    );
                } else {
                    eprintln!(
                        "[regression-guard] PASS       {id}: {cur:.1} ns vs baseline {base:.1} ns \
                         ({delta_pct:+.1}%)"
                    );
                }
            }
            Some(_) => {
                eprintln!("[regression-guard] SKIP       {id}: baseline value is non-positive");
            }
            None => {
                missing_in_baseline += 1;
                eprintln!("[regression-guard] NEW        {id}: {cur:.1} ns (not in baseline)");
            }
        }
    }

    // Note baseline ids that no longer exist in the current run.
    let mut stale = 0usize;
    for id in baseline.keys() {
        if !current.contains_key(id) {
            stale += 1;
            eprintln!("[regression-guard] STALE      {id}: in baseline but not measured");
        }
    }

    eprintln!(
        "[regression-guard] summary: {compared} compared, {regressions} regression(s), \
         {missing_in_baseline} new, {stale} stale (threshold {pct}%)"
    );
    if regressions > 0 {
        eprintln!(
            "[regression-guard] RESULT: FAIL — {regressions} bench id(s) exceeded the \
             {pct}% threshold. (Guard reports only; it does not abort the run.)"
        );
    } else {
        eprintln!("[regression-guard] RESULT: OK — no bench id exceeded the {pct}% threshold.");
    }
}

/// Entry point for the opt-in guard. A complete no-op (returns immediately,
/// runs zero timing) unless a write- or compare- baseline env var is set, so
/// the default `cargo bench` behaviour is unchanged.
///
/// Registered as a Criterion group so it runs within the normal harness, but
/// it does not define any Criterion benchmarks — it takes `&mut Criterion`
/// only to fit the `criterion_group!` signature.
fn regression_guard(_c: &mut Criterion) {
    let write_path = std::env::var(ENV_WRITE_BASELINE)
        .ok()
        .filter(|s| !s.is_empty());
    let compare_path = std::env::var(ENV_COMPARE_BASELINE)
        .ok()
        .filter(|s| !s.is_empty());

    if write_path.is_none() && compare_path.is_none() {
        // Default path: no guard env vars set — behave exactly as the
        // original scaffold (do nothing here).
        return;
    }

    eprintln!("[regression-guard] running manual timing pass ({GUARD_ITERS} iters/id)...");
    let medians = collect_medians();

    if let Some(path) = write_path.as_deref() {
        write_baseline(path, &medians);
    }
    if let Some(path) = compare_path.as_deref() {
        compare_baseline(path, &medians);
    }
}

criterion_group!(
    benches,
    bench_parse,
    bench_lower,
    bench_ptx_gen,
    regression_guard
);
criterion_main!(benches);
