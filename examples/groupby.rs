// SPDX-License-Identifier: Apache-2.0
//! Craton Bolt groupby example: synthesise 100K rows, run a SUM/COUNT.
//!
//! Run with:
//!     cargo run --example groupby --release
//!
//! (requires a CUDA-capable GPU.)

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use craton_bolt::Engine;
use std::sync::Arc;

const N_ROWS: usize = 100_000;
const N_GROUPS: i32 = 64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("group_id", DataType::Int32, false),
        Field::new("amount", DataType::Int32, false),
    ]));

    let group_ids: Vec<i32> = (0..N_ROWS).map(|i| (i as i32) % N_GROUPS).collect();
    let amounts: Vec<i32> = (0..N_ROWS).map(|i| ((i * 7919) % 1000) as i32).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(group_ids)),
            Arc::new(Int32Array::from(amounts)),
        ],
    )?;

    let mut engine = match Engine::new() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Engine::new() failed: {err}\n(requires CUDA)");
            return Ok(());
        }
    };

    engine.register_table("sales", batch)?;
    let t0 = std::time::Instant::now();
    let result = engine.sql(
        "SELECT group_id, SUM(amount), COUNT(*) \
         FROM sales \
         GROUP BY group_id \
         ORDER BY group_id",
    )?;
    let elapsed = t0.elapsed();
    println!("groupby on {N_ROWS} rows / {N_GROUPS} groups: {elapsed:?}");
    println!("result rows = {}", result.num_rows());
    Ok(())
}
