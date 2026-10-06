// SPDX-License-Identifier: Apache-2.0

//! Cost-model benchmark suite: turns the engine's runtime decisions into
//! measurable response surfaces instead of a handful of fixed workloads.
//!
//! Each group sweeps ONE independent variable with everything else held
//! fixed, so a regression (or a tier-selection heuristic) can be read off as
//! a curve. The axes follow `docs/BENCHMARKS.md` ("Cost-model matrix"):
//!
//! | Group | Axis | Needs GPU |
//! |---|---|---|
//! | `plan_pipeline` | query shape × stage (parse / optimize / lower), optimizer on/off | no |
//! | `dict_rewrite` | predicate count × dictionary cardinality | CUDA build (registration uploads dictionaries) |
//! | `fragmentation` | batch count at constant rows: per-query concat (old) vs coalesced read (new) | no |
//! | `batch_ingest` | batch count at constant rows: ingest + first query | yes |
//! | `filter_selectivity` | 0 / 1 / 10 / 50 / 90 / 100 % | yes |
//! | `groupby_cardinality` | 1 → 1M distinct keys | yes |
//! | `groupby_skew` | uniform vs Zipf-like hot keys | yes |
//! | `null_density` | 0 / 10 / 50 / 90 % NULL | yes |
//! | `jit_cache` | cold codegen+load vs warm module | yes |
//! | `residency` | re-uploaded table vs resident `GpuTable` | yes |
//! | `join_shape` | build:probe ratio | yes |
//! | `string_width` | average Utf8 value width | yes |
//! | `morsel_size` | memory budget → morsel rows (lazy stream) | yes |
//! | `recursive_cte` | recursion depth | yes |
//!
//! GPU groups run only with `BOLT_BENCH_GPU=1` (and a CUDA build); the host
//! groups run everywhere, including `--no-default-features --features
//! cuda-stub`. Row count for GPU groups: `BOLT_COST_ROWS` (default
//! 4,000,000). Every run writes a reproducibility manifest (see
//! `benches/support/manifest.rs`).
//!
//! ```text
//! cargo bench --bench cost_model --no-default-features --features cuda-stub   # host groups
//! BOLT_BENCH_GPU=1 cargo bench --bench cost_model                              # everything
//! ```

#[path = "support/manifest.rs"]
mod manifest;

use std::sync::Arc;
use std::time::Duration;

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use craton_bolt::exec::dict_registry::DictRegistry;
use craton_bolt::plan::{
    default_passes, lower_physical, parse_sql, DataType, Field, MemTableProvider, Schema,
};

// ---- configuration --------------------------------------------------------

fn gpu_enabled() -> bool {
    std::env::var("BOLT_BENCH_GPU").as_deref() == Ok("1")
}

#[cfg_attr(feature = "cuda-stub", allow(dead_code))]
fn gpu_rows() -> usize {
    std::env::var("BOLT_COST_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(4_000_000)
}

/// Deterministic xorshift PRNG — benches must not depend on `rand` and must
/// produce identical data run to run.
#[cfg_attr(feature = "cuda-stub", allow(dead_code))]
struct Rng(u64);

#[cfg_attr(feature = "cuda-stub", allow(dead_code))]
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---- fixtures ---------------------------------------------------------------

fn arrow_schema(fields: &[(&str, ArrowDataType, bool)]) -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(
        fields
            .iter()
            .map(|(n, t, nullable)| ArrowField::new(*n, t.clone(), *nullable))
            .collect::<Vec<_>>(),
    ))
}

/// `(k Int64, v Int64)` with keys from `key_of(row)`.
#[cfg_attr(feature = "cuda-stub", allow(dead_code))]
fn kv_batch(rows: usize, mut key_of: impl FnMut(usize) -> i64) -> RecordBatch {
    let k: Int64Array = (0..rows).map(&mut key_of).collect();
    let v: Int64Array = (0..rows as i64).collect();
    RecordBatch::try_new(
        arrow_schema(&[
            ("k", ArrowDataType::Int64, false),
            ("v", ArrowDataType::Int64, false),
        ]),
        vec![Arc::new(k) as ArrayRef, Arc::new(v)],
    )
    .unwrap()
}

/// `(x Int64)` holding `[start, start + rows)`.
fn x_batch(start: i64, rows: usize) -> RecordBatch {
    let x: Int64Array = (start..start + rows as i64).collect();
    RecordBatch::try_new(
        arrow_schema(&[("x", ArrowDataType::Int64, false)]),
        vec![Arc::new(x) as ArrayRef],
    )
    .unwrap()
}

/// `(x Float64)` with `null_pct`% NULLs, deterministic.
#[cfg_attr(feature = "cuda-stub", allow(dead_code))]
fn nullable_batch(rows: usize, null_pct: u64) -> RecordBatch {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let x: Float64Array = (0..rows)
        .map(|i| (rng.below(100) >= null_pct).then_some(i as f64))
        .collect();
    RecordBatch::try_new(
        arrow_schema(&[("x", ArrowDataType::Float64, true)]),
        vec![Arc::new(x) as ArrayRef],
    )
    .unwrap()
}

/// `(s Utf8)` with values of `width` bytes drawn from `distinct` values.
fn string_batch(rows: usize, width: usize, distinct: usize) -> RecordBatch {
    let pool: Vec<String> = (0..distinct)
        .map(|i| {
            let tag = format!("{i:08}");
            let mut s = tag.repeat(width / tag.len() + 1);
            s.truncate(width.max(1));
            s
        })
        .collect();
    let s: StringArray = (0..rows)
        .map(|i| Some(pool[i % distinct].as_str()))
        .collect();
    RecordBatch::try_new(
        arrow_schema(&[("s", ArrowDataType::Utf8, false)]),
        vec![Arc::new(s) as ArrayRef],
    )
    .unwrap()
}

#[cfg_attr(feature = "cuda-stub", allow(dead_code))]
fn plan_schema_x() -> Schema {
    Schema::new(vec![Field::new("x", DataType::Int64, false)])
}

// ---- host-only groups ------------------------------------------------------

fn plan_provider() -> MemTableProvider {
    let wide = Schema::new(
        (0..8)
            .map(|i| Field::new(format!("c{i}"), DataType::Int64, false))
            .collect(),
    );
    MemTableProvider::new()
        .with_table("a", wide.clone())
        .with_table("b", wide.clone())
        .with_table("c", wide.clone())
        .with_table("d", wide)
}

/// Queries of increasing AST / plan size.
fn plan_queries() -> Vec<(&'static str, String)> {
    let wide_expr = (0..64)
        .map(|i| format!("c{} * {}", i % 8, i + 1))
        .collect::<Vec<_>>()
        .join(" + ");
    vec![
        (
            "filter",
            "SELECT c0 FROM a WHERE c1 > 10 AND c2 < 99".into(),
        ),
        ("wide_expr_64", format!("SELECT {wide_expr} AS e FROM a")),
        (
            "join_3",
            "SELECT a.c0, d.c1 FROM a JOIN b ON a.c0 = b.c0 JOIN c ON b.c1 = c.c1 \
             JOIN d ON c.c2 = d.c2 WHERE a.c3 > 5"
                .into(),
        ),
        (
            "subquery",
            "SELECT c0 FROM a WHERE c1 > (SELECT MAX(c1) FROM b) AND c2 IN (SELECT c2 FROM c)"
                .into(),
        ),
        (
            "groupby_having",
            "SELECT c0, SUM(c1), AVG(c2) FROM a GROUP BY c0 HAVING SUM(c1) > 10 ORDER BY c0".into(),
        ),
    ]
}

fn bench_plan_pipeline(c: &mut Criterion) {
    let mut g = c.benchmark_group("plan_pipeline");
    g.sample_size(30).measurement_time(Duration::from_secs(3));
    for (name, sql) in plan_queries() {
        // Fresh provider per iteration: its unique schema version makes the
        // process-wide plan cache miss, so this measures a real parse.
        g.bench_function(format!("{name}/parse"), |b| {
            b.iter_batched(
                plan_provider,
                |p| black_box(parse_sql(&sql, &p).unwrap()),
                BatchSize::SmallInput,
            )
        });
        let plan = parse_sql(&sql, &plan_provider()).unwrap();
        g.bench_function(format!("{name}/optimize"), |b| {
            let passes = default_passes();
            b.iter_batched(
                || plan.clone(),
                |p| black_box(craton_bolt::plan::optimizer::run_to_fixpoint(&passes, p).unwrap()),
                BatchSize::SmallInput,
            )
        });
        let optimized =
            craton_bolt::plan::optimizer::run_to_fixpoint(&default_passes(), plan.clone()).unwrap();
        for (label, p) in [
            ("lower_unoptimized", &plan),
            ("lower_optimized", &optimized),
        ] {
            // Subquery plans do not lower until resolved; skip honestly.
            if lower_physical(p).is_ok() {
                g.bench_function(format!("{name}/{label}"), |b| {
                    b.iter(|| black_box(lower_physical(p).unwrap()))
                });
            }
        }
    }
    g.finish();
}

fn bench_dict_rewrite(c: &mut Criterion) {
    let mut g = c.benchmark_group("dict_rewrite");
    g.sample_size(30).measurement_time(Duration::from_secs(3));
    for cardinality in [16usize, 1_024, 65_536] {
        let batch = string_batch(cardinality * 2, 12, cardinality);
        let mut registry = DictRegistry::new();
        // Dictionary registration builds device-side dictionary columns, so
        // this group needs a CUDA build even though the rewrite is host work.
        if let Err(e) = registry.register_table("t".to_string(), &batch) {
            eprintln!("dict_rewrite: skipped ({e})");
            break;
        }
        let provider = MemTableProvider::new().with_table(
            "t",
            Schema::new(vec![Field::new("s", DataType::Utf8, false)]),
        );
        for predicates in [1usize, 8, 64] {
            let ors = (0..predicates)
                .map(|i| format!("s = '{:08}'", i % cardinality))
                .collect::<Vec<_>>()
                .join(" OR ");
            let plan = parse_sql(&format!("SELECT s FROM t WHERE {ors}"), &provider).unwrap();
            g.bench_function(format!("card_{cardinality}/preds_{predicates}"), |b| {
                b.iter(|| black_box(registry.rewrite_plan(&plan).unwrap()))
            });
        }
    }
    g.finish();
}

/// The removed per-query cost vs. the new one. Before ingest-time
/// coalescing, every query over a k-batch table ran
/// `concat_batches` (O(table bytes)); now the table is one batch and a
/// query reads it with an Arc clone. Bytes copied per query are reported as
/// throughput so the report shows the copy volume directly.
fn bench_fragmentation(c: &mut Criterion) {
    const ROWS: usize = 2_000_000;
    let mut concat = c.benchmark_group("fragmentation/per_query_concat");
    concat
        .sample_size(15)
        .measurement_time(Duration::from_secs(4));
    let mut fixtures = Vec::new();
    for batches in [1usize, 2, 8, 64, 1024] {
        let per = ROWS / batches;
        let parts: Vec<RecordBatch> = (0..batches)
            .map(|i| x_batch((i * per) as i64, per))
            .collect();
        let bytes: usize = parts.iter().map(|b| b.get_array_memory_size()).sum();
        let schema = parts[0].schema();
        concat.throughput(Throughput::Bytes(bytes as u64));
        concat.bench_function(format!("{batches}_batches"), |b| {
            b.iter(|| black_box(arrow::compute::concat_batches(&schema, parts.iter()).unwrap()))
        });
        let coalesced = arrow::compute::concat_batches(&schema, parts.iter()).unwrap();
        fixtures.push((batches, coalesced));
    }
    concat.finish();
    let mut g = c.benchmark_group("fragmentation/coalesced_read");
    g.sample_size(15).measurement_time(Duration::from_secs(2));
    for (batches, coalesced) in &fixtures {
        g.bench_function(format!("{batches}_batches"), |b| {
            b.iter(|| black_box(coalesced.clone()))
        });
    }
    g.finish();
}

// ---- GPU groups -------------------------------------------------------------

#[cfg(not(feature = "cuda-stub"))]
mod gpu {
    use super::*;
    use craton_bolt::Engine;

    fn group<'a>(
        c: &'a mut Criterion,
        name: &str,
    ) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
        let mut g = c.benchmark_group(name);
        g.sample_size(10).measurement_time(Duration::from_secs(5));
        g
    }

    fn run(engine: &Engine, sql: &str) -> RecordBatch {
        engine
            .sql(sql)
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .into_record_batch()
    }

    pub fn batch_ingest(c: &mut Criterion) {
        let rows = gpu_rows();
        let mut g = group(c, "batch_ingest");
        for batches in [1usize, 8, 64, 512] {
            let per = rows / batches;
            g.bench_function(format!("{batches}_batches"), |b| {
                b.iter_batched(
                    || {
                        (0..batches)
                            .map(|i| x_batch((i * per) as i64, per))
                            .collect::<Vec<_>>()
                    },
                    |parts| {
                        let mut e = Engine::new().unwrap();
                        for p in parts {
                            e.register_batch("t", p).unwrap();
                        }
                        black_box(run(&e, "SELECT SUM(x) FROM t"));
                    },
                    BatchSize::LargeInput,
                )
            });
        }
        g.finish();
    }

    pub fn filter_selectivity(c: &mut Criterion) {
        let rows = gpu_rows();
        let mut e = Engine::new().unwrap();
        e.register_table("t", x_batch(0, rows)).unwrap();
        let mut g = group(c, "filter_selectivity");
        g.throughput(Throughput::Elements(rows as u64));
        for pct in [0usize, 1, 10, 50, 90, 100] {
            let cut = rows * pct / 100;
            let sql = format!("SELECT x FROM t WHERE x < {cut}");
            run(&e, &sql);
            g.bench_function(format!("{pct}pct"), |b| b.iter(|| black_box(run(&e, &sql))));
        }
        g.finish();
    }

    pub fn groupby_cardinality(c: &mut Criterion) {
        let rows = gpu_rows();
        let mut g = group(c, "groupby_cardinality");
        g.throughput(Throughput::Elements(rows as u64));
        for keys in [1usize, 100, 10_000, 1_000_000] {
            let mut e = Engine::new().unwrap();
            e.register_table("t", kv_batch(rows, |i| (i % keys) as i64))
                .unwrap();
            let sql = "SELECT k, SUM(v) FROM t GROUP BY k";
            run(&e, sql);
            g.bench_function(format!("{keys}_keys"), |b| {
                b.iter(|| black_box(run(&e, sql)))
            });
        }
        g.finish();
    }

    pub fn groupby_skew(c: &mut Criterion) {
        let rows = gpu_rows();
        const KEYS: u64 = 10_000;
        let mut g = group(c, "groupby_skew");
        g.throughput(Throughput::Elements(rows as u64));
        // Uniform vs. a Zipf-like distribution (inverse-power sampling,
        // exponent ~1.1): a few hot keys take most rows, which serialises
        // atomics on those keys.
        let mut rng = Rng(42);
        let mut zipf = move |_: usize| {
            let u = rng.unit().max(1e-12);
            ((u.powf(-1.0 / 0.1) as u64) % KEYS) as i64
        };
        let mut uniform_rng = Rng(7);
        for (label, batch) in [
            (
                "uniform",
                kv_batch(rows, |_| uniform_rng.below(KEYS) as i64),
            ),
            ("zipf", kv_batch(rows, &mut zipf)),
        ] {
            let mut e = Engine::new().unwrap();
            e.register_table("t", batch).unwrap();
            let sql = "SELECT k, SUM(v) FROM t GROUP BY k";
            run(&e, sql);
            g.bench_function(label, |b| b.iter(|| black_box(run(&e, sql))));
        }
        g.finish();
    }

    pub fn null_density(c: &mut Criterion) {
        let rows = gpu_rows();
        let mut g = group(c, "null_density");
        g.throughput(Throughput::Elements(rows as u64));
        for pct in [0u64, 10, 50, 90] {
            let mut e = Engine::new().unwrap();
            e.register_table("t", nullable_batch(rows, pct)).unwrap();
            let sql = "SELECT SUM(x), COUNT(x) FROM t";
            run(&e, sql);
            g.bench_function(format!("{pct}pct_null"), |b| {
                b.iter(|| black_box(run(&e, sql)))
            });
        }
        g.finish();
    }

    pub fn jit_cache(c: &mut Criterion) {
        let mut e = Engine::new().unwrap();
        e.register_table("t", x_batch(0, 65_536)).unwrap();
        let mut g = group(c, "jit_cache");
        // Cold: a distinct constant per iteration yields a distinct kernel,
        // so every iteration pays codegen + module load (the in-memory and
        // disk PTX caches can only miss).
        let mut n = 0u64;
        g.bench_function("cold_codegen_and_load", |b| {
            b.iter(|| {
                n += 1;
                black_box(run(
                    &e,
                    &format!("SELECT x * {n} + {} AS y FROM t", n * 7 + 3),
                ))
            })
        });
        let warm = "SELECT x * 3 + 5 AS y FROM t";
        run(&e, warm);
        g.bench_function("warm_module", |b| b.iter(|| black_box(run(&e, warm))));
        g.finish();
    }

    pub fn residency(c: &mut Criterion) {
        let rows = gpu_rows();
        let mut e = Engine::new().unwrap();
        e.register_table("t", x_batch(0, rows)).unwrap();
        let sql = "SELECT SUM(x) FROM t";
        run(&e, sql);
        let mut g = group(c, "residency");
        g.throughput(Throughput::Bytes((rows * 8) as u64));
        g.bench_function("resident_gpu_table", |b| b.iter(|| black_box(run(&e, sql))));
        let fresh = x_batch(0, rows);
        g.bench_function("cold_host_table_reupload", |b| {
            b.iter(|| {
                e.replace_table("t", fresh.clone()).unwrap();
                black_box(run(&e, sql))
            })
        });
        g.finish();
    }

    pub fn join_shape(c: &mut Criterion) {
        let probe_rows = gpu_rows();
        let mut g = group(c, "join_shape");
        for ratio in [1usize, 10, 100] {
            let build_rows = (probe_rows / ratio).max(1);
            let mut e = Engine::new().unwrap();
            e.register_table("build", kv_batch(build_rows, |i| i as i64))
                .unwrap();
            let mut rng = Rng(11);
            e.register_table(
                "probe",
                kv_batch(probe_rows, |_| rng.below(build_rows as u64) as i64),
            )
            .unwrap();
            let sql = "SELECT COUNT(*) FROM probe JOIN build ON probe.k = build.k";
            run(&e, sql);
            g.bench_function(format!("build_1_to_probe_{ratio}"), |b| {
                b.iter(|| black_box(run(&e, sql)))
            });
        }
        g.finish();
    }

    pub fn string_width(c: &mut Criterion) {
        let rows = gpu_rows() / 4;
        let mut g = group(c, "string_width");
        for width in [4usize, 64, 1024] {
            let mut e = Engine::new().unwrap();
            e.register_table("t", string_batch(rows, width, 256))
                .unwrap();
            let needle = {
                let tag = format!("{:08}", 17);
                let mut s = tag.repeat(width / tag.len() + 1);
                s.truncate(width.max(1));
                s
            };
            let sql = format!("SELECT COUNT(*) FROM t WHERE s = '{needle}'");
            run(&e, &sql);
            g.bench_function(format!("{width}B"), |b| b.iter(|| black_box(run(&e, &sql))));
        }
        g.finish();
    }

    pub fn morsel_size(c: &mut Criterion) {
        let rows = gpu_rows();
        let total_bytes = rows * 8;
        let mut g = group(c, "morsel_size");
        // Budget = whole table (no morsels) down to 1/64 of it.
        for divisor in [1usize, 4, 16, 64] {
            let mut e = Engine::builder()
                .memory_budget((total_bytes / divisor).max(4096))
                .build()
                .unwrap();
            e.register_table_stream_lazy(
                "t",
                plan_schema_x(),
                Box::new(move || Box::new(std::iter::once(Ok(x_batch(0, rows))))),
            )
            .unwrap();
            let sql = "SELECT x FROM t WHERE x > 10";
            run(&e, sql);
            g.bench_function(format!("budget_1_over_{divisor}"), |b| {
                b.iter(|| black_box(run(&e, sql)))
            });
        }
        g.finish();
    }

    pub fn recursive_cte(c: &mut Criterion) {
        let mut e = Engine::new().unwrap();
        e.register_table("seed", x_batch(1, 1)).unwrap();
        let mut g = group(c, "recursive_cte");
        for depth in [10usize, 100, 500] {
            let sql = format!(
                "WITH RECURSIVE r(n) AS (SELECT x FROM seed UNION ALL \
                 SELECT n + 1 FROM r WHERE n < {depth}) SELECT COUNT(*) FROM r"
            );
            run(&e, &sql);
            g.bench_function(format!("depth_{depth}"), |b| {
                b.iter(|| black_box(run(&e, &sql)))
            });
        }
        g.finish();
    }
}

fn bench_gpu(c: &mut Criterion) {
    if !gpu_enabled() {
        eprintln!("cost_model: GPU groups skipped (set BOLT_BENCH_GPU=1 on a CUDA build)");
        return;
    }
    #[cfg(not(feature = "cuda-stub"))]
    {
        gpu::batch_ingest(c);
        gpu::filter_selectivity(c);
        gpu::groupby_cardinality(c);
        gpu::groupby_skew(c);
        gpu::null_density(c);
        gpu::jit_cache(c);
        gpu::residency(c);
        gpu::join_shape(c);
        gpu::string_width(c);
        gpu::morsel_size(c);
        gpu::recursive_cte(c);
    }
    #[cfg(feature = "cuda-stub")]
    {
        let _ = c;
        eprintln!("cost_model: GPU groups unavailable in a cuda-stub build");
    }
}

fn write_manifest(_: &mut Criterion) {
    manifest::write(
        "cost_model",
        &[
            manifest::Axis {
                name: "gpu_rows",
                value: gpu_rows().to_string(),
            },
            manifest::Axis {
                name: "gpu_groups",
                value: gpu_enabled().to_string(),
            },
            manifest::Axis {
                name: "transfer_included",
                value: "true (end-to-end Engine::sql, H2D/D2H inside the timing)".into(),
            },
            manifest::Axis {
                name: "cold_or_warm",
                value: "warm (one untimed run per configuration) except jit_cache/cold".into(),
            },
        ],
    );
}

criterion_group!(
    benches,
    write_manifest,
    bench_plan_pipeline,
    bench_dict_rewrite,
    bench_fragmentation,
    bench_gpu
);
criterion_main!(benches);
