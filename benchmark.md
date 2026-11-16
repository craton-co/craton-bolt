# Craton Bolt Benchmarks (NVIDIA L40S)

All measurements below were run on an **NVIDIA L40S (46,068 MiB VRAM)** host with **AMD EPYC 9224 (12 vCPUs)**, **70 GiB RAM**, **CUDA 13.2** (Driver 595.91.07), and **PostgreSQL 14.24** (`UNLOGGED` tables, `VACUUM ANALYZE`, `shared_buffers=16GB`, `work_mem=256MB`). `BOLT_BENCH_GPU=1` was enabled for all Criterion suites.

---

## 1. `olap_benchmarks` — h2o.ai GROUP BY, 10 M rows

| Query | DuckDB 1.2 | Polars 0.42 | Craton Bolt (L40S) | PG (1-core) | PG (4-core) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `q1_low_card_sum` | 5.48 ms | 16.53 ms | **3.79 ms** | 1561.46 ms | 383.76 ms |
| `q2_med_card_2sum` | **19.26 ms** | 46.63 ms | 117.72 ms | 1437.19 ms | 387.74 ms |
| `q3_two_key_sum` | 122.27 ms | 229.31 ms | **110.94 ms** | 3275.61 ms | 1463.08 ms |
| `q4_low_card_3avg` | **9.74 ms** | 51.22 ms | 24.69 ms | 1707.03 ms | 405.38 ms |
| `q5_high_card_sum` | 170.35 ms | 253.49 ms | **156.16 ms** | 7915.51 ms | 2709.39 ms |

---

## 2. `compute_benchmarks` — Resident-VRAM / In-Memory Compute, 10 M rows

| Kernel | Polars 0.42 | DuckDB 1.2 | Craton Bolt (L40S) | PG (1-core) | PG (4-core) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `c1_fma_sum` | 132.98 ms | 7.72 ms | **867.68 µs** | 628.39 ms | 169.79 ms |
| `c2_poly4_sum` | 305.04 ms | 9.60 ms | **426.42 µs** | 815.39 ms | 194.48 ms |
| `c3_sphere_filter_sum` | 237.64 ms | 15.38 ms | **1.60 ms** | 770.43 ms | 190.84 ms |
| `c4_weighted_sum` | 361.58 ms | 14.94 ms | **875.93 µs** | 746.64 ms | 184.99 ms |
| `c5_filtered_fma_sum` | 134.37 ms | 14.39 ms | **1.57 ms** | 770.36 ms | 168.02 ms |
| `c6_poly12_sum` | 1.225 s | 42.08 ms | **454.33 µs** | 1248.28 ms | 336.60 ms |

---

## 3. `query_benchmarks` — Heavy Arithmetic E2E, 50 M rows

*(PostgreSQL measured via server-side `EXPLAIN (ANALYZE, FORMAT JSON)` execution time)*

| Query | Polars (CPU MT) | Craton Bolt E2E (L40S) | CPU Ref (1T) | PG (1-core) | PG (4-core) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `proj` | **3.12 µs** | 625.55 ms | — | 4671.11 ms | 4701.35 ms |
| `arith` | 5.336 s | 595.12 ms | **450.51 ms** | 7116.68 ms | 7162.24 ms |
| `filtered` | 589.10 ms | **156.52 ms** | — | 2862.69 ms | 2955.93 ms |

---

## 4. `scaling_benchmarks` — `fused_filter_projection` (1 M / 10 M / 100 M)

*(DuckDB materialized via `.arrow().read_all()`; PostgreSQL via `EXPLAIN ANALYZE` execution time)*

| Scale | Craton Bolt (L40S) | DuckDB | Polars CPU | Polars GPU | PG (1-core) | PG (4-core) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **1 M** | 5.47 ms | 12.01 ms | **2.39 ms** | 7.65 ms | 114.56 ms | 120.81 ms |
| **10 M** | 112.01 ms | 87.05 ms | **9.92 ms** | 77.36 ms | 1030.25 ms | 1028.61 ms |
| **100 M** | 1.008 s | 1.063 s | **74.63 ms** | 646.52 ms | 9334.83 ms | 9342.44 ms |
