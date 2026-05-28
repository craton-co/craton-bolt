// SPDX-License-Identifier: Apache-2.0

//! End-to-end check that the v0.6 / M5 tracing instrumentation actually
//! emits the documented spans across the parse → lower → codegen → ptx_load
//! pipeline.
//!
//! Strategy: install a custom `tracing-subscriber` layer that records every
//! `on_new_span` name into a shared `Vec`, drive the offline portion of the
//! engine pipeline (no GPU required), then assert the captured set contains
//! the spans listed in `craton_bolt::observability`.
//!
//! GPU-dependent spans (`launch`, `transfer`, `materialize`) are not
//! covered here — those are exercised by the ignored on-GPU tests under
//! `tests/e2e_tests.rs` and the broader integration suite. The offline
//! spans suffice to lock in the instrumentation contract for the part of
//! the pipeline every CI run touches.

use std::sync::{Arc, Mutex};

use arrow_array::{Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use std::sync::Arc as StdArc;

use craton_bolt::jit::{compile_ptx, CudaModule};
use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Field, MemTableProvider, PhysicalPlan, Schema,
};
use craton_bolt::tracing as bolt_tracing;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

/// Shared, lock-protected list of every span name seen during a test.
#[derive(Default, Clone)]
struct CapturedSpans(Arc<Mutex<Vec<String>>>);

impl CapturedSpans {
    fn names(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
}

/// Custom `tracing-subscriber` layer that records the name of every new
/// span into the shared `CapturedSpans`. Indifferent to span hierarchy /
/// fields — we only assert on names here.
struct CaptureLayer {
    spans: CapturedSpans,
}

impl<S> Layer<S> for CaptureLayer
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        _id: &tracing::span::Id,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if let Ok(mut v) = self.spans.0.lock() {
            v.push(attrs.metadata().name().to_string());
        }
    }
}

fn sales_schema() -> Schema {
    Schema::new(vec![
        Field {
            name: "region_id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "price".into(),
            dtype: DataType::Float64,
            nullable: false,
        },
    ])
}

fn sales_provider() -> MemTableProvider {
    MemTableProvider::new().with_table("sales", sales_schema())
}

#[allow(dead_code)]
fn sales_batch(n: usize) -> RecordBatch {
    let region: Int32Array = (0..n as i32).map(|i| i % 4).collect();
    let price: Float64Array = (0..n).map(|i| (i + 1) as f64).collect();
    let schema = StdArc::new(ArrowSchema::new(vec![
        ArrowField::new("region_id", ArrowDataType::Int32, false),
        ArrowField::new("price", ArrowDataType::Float64, false),
    ]));
    RecordBatch::try_new(schema, vec![StdArc::new(region), StdArc::new(price)]).unwrap()
}

/// Drive the offline portion of the pipeline (parse + lower + codegen +
/// ptx_load) under a capturing subscriber, then assert that each documented
/// span fired at least once.
///
/// We `cuModuleLoadDataEx` the PTX too — that crosses into the CUDA driver,
/// but the `from_ptx` cache layer is the part we want to instrument; the
/// underlying `cuModuleLoadDataEx` call gracefully no-ops in cuda-stub
/// builds, which is the CI baseline.
#[test]
fn spans_fire_across_offline_pipeline() {
    let captured = CapturedSpans::default();
    let layer = CaptureLayer {
        spans: captured.clone(),
    };

    // Scoped subscriber so we don't leak global state into sibling tests
    // run in the same process by `cargo test`.
    let subscriber = tracing_subscriber::registry().with(layer);
    bolt_tracing::subscriber::with_default(subscriber, || {
        let provider = sales_provider();
        let plan = parse_sql("SELECT price FROM sales", &provider).expect("parse");
        let phys = lower_physical(&plan).expect("lower");
        let PhysicalPlan::Projection { kernel, .. } = &phys else {
            panic!("expected Projection");
        };
        let ptx = compile_ptx(kernel, "bolt_kernel").expect("codegen");
        // The `ptx_load` span fires inside `CudaModule::from_ptx`. On hosts
        // without a real CUDA driver this will error — that's fine for the
        // span-capture assertion; the span is entered before the driver
        // call, so a downstream failure does not remove it from `captured`.
        let _ = CudaModule::from_ptx(&ptx);
    });

    let names = captured.names();
    let expect_present = ["parse", "lower", "codegen", "ptx_load"];
    for want in expect_present {
        assert!(
            names.iter().any(|n| n == want),
            "expected span `{}` to fire; captured = {:?}",
            want,
            names,
        );
    }
}

/// `pub use ::tracing` at the crate root must round-trip — downstream code
/// is supposed to be able to call into the re-export without adding a
/// direct `tracing` dependency. If this stops compiling, the re-export was
/// broken.
#[test]
fn crate_reexports_tracing() {
    // Reach a stable item via the re-export path. `Level` is part of
    // `tracing`'s public surface and won't change across patch releases.
    let _lvl: bolt_tracing::Level = bolt_tracing::Level::INFO;
}
