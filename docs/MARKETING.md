# Javelin — a JIT-compiled GPU SQL engine in pure Rust

*Marketing one-pager. For technical depth see [`README.md`](../README.md),
[`docs/ARCHITECTURE.md`](ARCHITECTURE.md), and [`docs/BENCHMARKS.md`](BENCHMARKS.md).*

---

## The pitch in one sentence

**SQL goes in, NVIDIA PTX comes out at runtime, the GPU does the rest —
with Rust's borrow checker enforcing memory safety all the way down to the
device pointer.**

## The pitch in three sentences

Most GPU SQL engines are C++ kernel libraries with a Python or Rust wrapper.
Javelin is the inverse: a Rust crate that compiles every query into a fresh
NVIDIA PTX kernel at execution time, fuses the whole expression tree into one
register-resident pipeline, and loads it via the raw CUDA driver — no C++,
no NVRTC, no third-party query engine. Borrow-checked GPU memory primitives
(`GpuVec<T>`, `GpuView<'a, T>`) make use-after-free, double-free, and shared
mutable aliasing across kernel boundaries compile-time errors.

---

## Why this matters

### The problem with the status quo

| Approach | What it actually is | The pain |
| --- | --- | --- |
| RAPIDS / cuDF | Pre-compiled CUDA kernel library, Python frontend | Intermediates round-trip through global memory between operators; kernel-fusion story is incomplete; no SQL frontend; Python deployment baggage. |
| HeavyDB | C++ JIT via LLVM | LLVM dependency is heavy (~1 GB); single-vendor build; not embeddable from Rust without an FFI dance. |
| BlazingSQL | RAPIDS + Apache Calcite | Three runtimes glued together; project largely dormant since 2022. |
| Polars / DuckDB on CPU | Excellent CPU engines, no GPU path | Will not benefit from the GPU you already paid for. |
| DataFusion + cudf adapters | Active but adapter-heavy | Each operator boundary is a copy; you get the worst of both worlds for fused arithmetic. |

### What Javelin does differently

1. **Whole-query kernel fusion.** A query like `SELECT (a * b + c * d) * 1.0825 FROM t WHERE id > 100` becomes **one** PTX kernel. All seven operations stay in registers — no intermediate materialisation, no D2D round-trips. The compiler-level analog to what Polars / DataFusion do for the CPU, applied to the GPU.

2. **Borrow-checked GPU memory ("CUDA-Oxide").** GPU buffers are typed handles. The borrow checker prevents:
   - Use-after-free (the buffer outlives the kernel launch borrow)
   - Double-free (drops route through a pool with refcount safety)
   - Concurrent mutable aliasing across kernel boundaries
   - Reading a buffer being written by another in-flight kernel
   Other engines push these guarantees to runtime asserts (at best) or rely on convention (at worst). Javelin enforces them at compile time, in pure-safe Rust.

3. **Sub-25 µs CPU-side overhead.** Parse → plan → codegen → PTX-string in under 25 microseconds end-to-end. JIT-compiling every query is a viable execution model, not a performance footgun.

4. **Pure-Rust, single-crate deployable.** No LLVM. No NVRTC. No C++ runtime to ship. Just `cargo add javelin` and a CUDA driver.

5. **Multi-tier GROUP BY.** Three-tier execution strategy gated on cardinality:
   - **Tier 1** (≤ 1024 groups): per-block shared-memory pre-aggregation
   - **Tier 2** (medium-to-high): hash-partitioned two-pass with on-device pass-2 reduce
   - **Tier 2.1** (high-card): 4096-way partition + per-partition shared-mem reduce
   Covers SUM, MULTI-SUM (1..=4 cols), AVG, COUNT, MIN, MAX, two-key packed-i64, and float MIN/MAX via CAS-loop atomic.

---

## Performance highlights

*All numbers from `cargo bench` on RTX 2060, CUDA 12.6, 10 M-row datasets,
verified end-to-end equivalent against Polars and DuckDB before timing.
Full table in [`docs/BENCHMARKS.md`](BENCHMARKS.md).*

### h2o.ai db-benchmark groupby subset (community standard)

| Query                          | Polars (CPU MT) | DuckDB (CPU MT) | **Javelin (GPU)** | Δ vs cold baseline |
| ------------------------------ | --------------- | --------------- | ----------------- | ------------------ |
| q1 low-card SUM (100 grps)     | 19.0 ms         | 6.9 ms          | **51.4 ms**       | **5.5× faster**    |
| q2 med-card 2-SUM (10 K)       | 99.4 ms         | 46.4 ms         | **384 ms**        | **1.15× faster**   |
| q3 two-key SUM (1 M grps)      | 385 ms          | 498 ms          | **219 ms** ⭐      | **3.7× faster**    |
| q4 low-card 3-AVG (100 grps)   | 97.0 ms         | 12.9 ms         | **70.5 ms**       | **6.4× faster**    |
| q5 high-card SUM (1 M grps)    | 358 ms          | 623 ms          | **237 ms** ⭐      | **3.7× faster**    |

⭐ Javelin wins outright vs both Polars and DuckDB.

### Heavy-arithmetic OLAP (50 M rows, fused multi-operator)

| Query                                   | Polars     | **Javelin (GPU)**     | Speedup |
| --------------------------------------- | ---------- | --------------------- | ------- |
| 11-op arithmetic chain (10× *, 10× ± )  | 4.05 s     | **124.8 ms**          | **32.4× faster** |
| Filter + 4-op arithmetic                | 369 ms     | **41.8 ms**           | **8.8× faster**  |

The arithmetic win is the structural story: GPU register fusion vs CPU
intermediate materialisation. On idiomatic OLAP groupby (the table above)
the wins are smaller but on the **two hardest workloads** — q3 two-key
high-cardinality, q5 1 M-group high-cardinality — **Javelin is currently
the fastest of the three**.

---

## Where Javelin fits

### Good fit

- **Embedded analytical workloads** in Rust services that already have a GPU available (game servers, ML preprocessing, scientific instruments, real-time fraud, observability backends).
- **High-cardinality GROUP BY at scale** where multi-threaded CPU engines hit hash-table contention walls (q3 / q5 territory).
- **Fused-arithmetic ETL** — anything where the per-row work is more than a single multiply and the data is GPU-resident or can be.
- **Memory-safety-critical analytics** — finance, healthcare, defense — where Rust's compile-time guarantees + Apache 2.0 licensing pencil out vs the C++ alternatives.

### Honest non-fit

- **Pure column projection / passthrough.** Polars's zero-copy LazyFrame select is unbeatable; no GPU pipeline can match a metadata clone.
- **Production-critical SQL today.** Public API is unstable pre-1.0. Subset of SQL supported. Use for evaluation, prototyping, and benchmarking — not for paying customers' OLTP backends.
- **Hosts without an NVIDIA GPU.** macOS is unsupported (Apple killed CUDA in 2019). AMD / Intel GPUs: no plan yet.

---

## What's under the hood (one-paragraph version)

`sqlparser` → logical plan AST → physical plan with SSA-shaped IR → hand-written
PTX codegen (`src/jit/`) → `cuModuleLoadDataEx` → `cuLaunchKernel` →
Arrow `RecordBatch` back to host. A 256-entry FIFO PTX cache eliminates
recompilation on hot kernel shapes; GPU-resident tables eliminate H2D re-uploads
on every query; a process-wide power-of-two memory pool eliminates
`cuMemAlloc`/`cuMemFree` traffic in the hot path. The CUDA-Oxide layer makes
all of this safe by construction.

Architecture diagram, layer-by-layer file walkthrough, and the SQL-supported
subset are in [`docs/ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Differentiators at a glance

| Dimension                | RAPIDS / cuDF         | HeavyDB            | DataFusion+adapter      | **Javelin**             |
| ------------------------ | --------------------- | ------------------ | ----------------------- | ----------------------- |
| Language                 | C++ + Python wrapper  | C++                | Rust + C++ kernels      | **Pure Rust**           |
| Kernel strategy          | Pre-compiled library  | LLVM JIT           | Pre-compiled adapters   | **Runtime PTX codegen** |
| Whole-query fusion       | Partial               | Yes (LLVM)         | Limited                 | **Yes (per-query PTX)** |
| Memory safety            | C++ conventions       | C++ conventions    | Mixed                   | **Compile-time (CUDA-Oxide)** |
| Deps weight              | RAPIDS (~hundreds MB) | LLVM (~GB)         | Adapter glue            | **Single crate + CUDA driver** |
| Embeddable from Rust     | Awkward FFI           | Awkward FFI        | Native (but heavy)      | **Native, idiomatic**   |
| License                  | Apache 2.0            | Apache 2.0         | Apache 2.0              | **Apache 2.0**          |

---

## What's next (0.3 → 0.4)

- **cudarc adoption** (Stage 1.5 landed): replace hand-rolled `cuda_sys` FFI with the `cudarc` crate while keeping the CUDA-Oxide layer on top. Reduces unsafe-block surface area, gains community-maintained driver bindings. See [`docs/CUDARC_ADOPTION.md`](CUDARC_ADOPTION.md).
- **rust-cuda for kernel codegen** (Wave A spike landed): explore writing kernels in Rust itself via `rustc_codegen_nvvm`, compiled to PTX at build time. Two-track plan (rust-cuda *vs* extended hand-emit) documented in [`docs/MILESTONE_0_4.md`](MILESTONE_0_4.md) and [`docs/rust_cuda/`](rust_cuda/).
- **JOIN, ORDER BY, LIMIT, DISTINCT** — scaffolding in place (`src/exec/join.rs`, `sort.rs`, `limit.rs`, `distinct.rs`); execution paths land in 0.3 / 0.4.
- **String GROUP BY keys.** Dictionary encoding is plumbed; the hash kernel needs the dictionary-index path wired through.
- **Streaming ingest.** Right now every table lives in VRAM. A spill-to-host path for tables larger than VRAM is the obvious 0.5 feature.

The full roadmap with dates and decision points is in [`ROADMAP.md`](../ROADMAP.md).

---

## Try it

```bash
git clone <repo-url> javelin
cd javelin
cargo build --release                                       # default
cargo build --release --features cudarc                     # cudarc backend
JAVELIN_BENCH_GPU=1 cargo bench --bench olap_benchmarks     # h2o.ai vs Polars vs DuckDB
```

Sample usage in 12 lines in [`README.md#run-a-query`](../README.md#run-a-query).

---

## Status & licensing

- **Status:** Active development. Apache-2.0. Pre-1.0; expect API churn.
- **Target hardware:** NVIDIA GPU, compute capability ≥ 7.0 (Volta or newer). Driver matching CUDA Toolkit ≥ 12.
- **Platforms:** Linux x86_64, Windows x86_64 MSVC. macOS unsupported by upstream CUDA.
- **Security:** see [`SECURITY.md`](../SECURITY.md). Private disclosure preferred.
- **Contributing:** [`CONTRIBUTING.md`](../CONTRIBUTING.md). DCO sign-off required.

---

*Javelin stands on `arrow-rs`, `sqlparser-rs`, and NVIDIA's CUDA driver.
Everything above the driver is ours; everything in this document is
reproducible from the source tree.*
