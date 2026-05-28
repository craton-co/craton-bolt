//! Craton Bolt quickstart: build a RecordBatch, register it, run a SELECT.
//!
//! Run with:
//!     cargo run --example quickstart
//!
//! (requires a CUDA-capable GPU; build with `--features cuda-stub` to
//! compile-only.)

use std::sync::Arc;
use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use craton_bolt::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("score", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec!["alice", "bob", "carol", "dave"])),
            Arc::new(Int32Array::from(vec![95, 87, 92, 78])),
        ],
    )?;

    let mut engine = match Engine::new() {
        Ok(e) => e,
        Err(err) => {
            eprintln!(
                "Engine::new() failed (no CUDA?): {err}\n\
                 This example requires a CUDA-capable GPU. Recompile with\n\
                 `--features cuda-stub --no-default-features` to build-only."
            );
            return Ok(());
        }
    };

    engine.register_table("users", batch)?;
    let result = engine.sql(
        "SELECT name, score FROM users WHERE score >= 90 ORDER BY score DESC",
    )?;
    println!("rows = {}", result.num_rows());
    println!("{:?}", result.record_batch());
    Ok(())
}
