// SPDX-License-Identifier: AGPL-3.0-only
//! Benchmark Craton Bolt (GPU) on prebuilt `sales_{1m,10m,100m}.parquet`
//! across the 5 realistic analytical queries (Q1 from postgres-bolt-speedup.mdx + Q2..Q5).

use std::fs::File;
use std::time::Instant;

use arrow::compute::concat_batches;
use arrow_array::RecordBatch;
use craton_bolt::Engine;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const QUERIES: &[(&str, &str)] = &[
    (
        "q1_article_region_net",
        "SELECT region_id, SUM((price * (1.0 + tax)) - discount) AS net_total \
         FROM sales WHERE region_id < 50 GROUP BY region_id ORDER BY region_id",
    ),
    (
        "q2_region_category_rev",
        "SELECT region_id, category_id, \
                SUM(quantity * price * (1.0 - discount / 100.0)) AS gross_revenue, \
                AVG(price) AS avg_price, \
                COUNT(*) AS order_lines \
         FROM sales WHERE order_day >= 365 AND order_day < 1460 \
         GROUP BY region_id, category_id ORDER BY region_id, category_id",
    ),
    (
        "q3_store_profit",
        "SELECT store_id, \
                SUM((quantity * price * (1.0 + tax)) - discount - shipping_cost) AS net_profit, \
                SUM(quantity) AS total_units \
         FROM sales WHERE quantity >= 2 AND price > 20.0 \
         GROUP BY store_id ORDER BY store_id",
    ),
    (
        "q4_scalar_audit",
        "SELECT SUM(((quantity * price * (1.0 + tax)) - discount) * 0.97 - shipping_cost * 1.05) AS audited_net \
         FROM sales WHERE (quantity * price) > 250.0 AND region_id < 80",
    ),
    (
        "q5_customer_ltv",
        "SELECT customer_id, SUM(quantity * price - discount) AS lifetime_spend \
         FROM sales WHERE order_day >= 730 GROUP BY customer_id",
    ),
];

fn load_parquet(path: &str) -> RecordBatch {
    let file = File::open(path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("parquet reader builder")
        .with_batch_size(1_048_576);
    let schema = builder.schema().clone();
    let reader = builder.build().expect("build parquet reader");
    let batches: Vec<RecordBatch> = reader.map(|r| r.expect("read batch")).collect();
    concat_batches(&schema, &batches).expect("concat batches")
}

fn main() {
    let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| "./data".into());
    let scales_raw = std::env::var("SCALES").unwrap_or_else(|_| "1m,10m,100m".into());
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    for scale in scales_raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let pq_path = format!("{data_dir}/sales_{scale}.parquet");
        println!("\n================ BOLT GPU SCALE: {scale} ({pq_path}) ================");
        let t_load = Instant::now();
        let batch = load_parquet(&pq_path);
        let n_rows = batch.num_rows();
        println!("  loaded {n_rows} rows in {:.2?}", t_load.elapsed());

        let mut engine = Engine::new().expect("CUDA Engine::new");
        let t_reg = Instant::now();
        engine.register_table("sales", batch).expect("register_table");
        println!("  registered in VRAM in {:.2?}", t_reg.elapsed());

        for &(qname, sql) in QUERIES {
            // Cold run (incl. PTX JIT)
            let t_cold = Instant::now();
            match engine.sql(sql) {
                Ok(res) => {
                    let cold_ms = t_cold.elapsed().as_secs_f64() * 1000.0;
                    let out_rows = res.num_rows();
                    let mut warm_ms = Vec::with_capacity(iters);
                    for _ in 0..iters {
                        let t0 = Instant::now();
                        let r = engine.sql(sql).expect("warm query");
                        warm_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
                        assert_eq!(r.num_rows(), out_rows);
                    }
                    warm_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let min_ms = warm_ms[0];
                    let med_ms = warm_ms[iters / 2];
                    let avg_ms = warm_ms.iter().sum::<f64>() / iters as f64;
                    println!(
                        "  [bolt_gpu] {qname:24} cold={cold_ms:8.2} ms  median={med_ms:8.2} ms  min={min_ms:8.2} ms  avg={avg_ms:8.2} ms  (rows={out_rows})"
                    );
                }
                Err(err) => {
                    println!("  [bolt_gpu] {qname:24} ERROR: {err}");
                }
            }
        }
    }
}
