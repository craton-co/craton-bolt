# GROUP BY performance — analysis and proposed optimizations

> [!NOTE]
> **Status: Implemented** (Tiers 1–2 landed; see [BENCHMARKS.md](./BENCHMARKS.md) for results)

## What the measurements say

From the h2o.ai db-benchmark groupby subset, three-engine table (10 M rows,
RTX 2060) in [`BENCHMARKS.md`](./BENCHMARKS.md):

| Query                   | DuckDB    | Polars     | Craton Bolt    | Craton Bolt gap |
| ----------------------- | --------- | ---------- | ---------- | ----------- |
| q1: SUM by id1 (100)    | 6.12 ms   | 16.4 ms    | 269 ms     | **44× / 16×** |
| q2: 2-SUM by id2 (10 K) | 43.7 ms   | 79.5 ms    | 406 ms     | **9.3× / 5.1×** |
| q3: SUM by (id1, id2)   | 448 ms    | **296 ms** | 704 ms     | **1.6× / 2.4×** |
| q5: SUM by id3 (1 M)    | 549 ms    | **241 ms** | 770 ms     | **1.4× / 3.2×** |

Craton Bolt loses on every query. The gap is widest at low cardinality (q1, q2)
and narrowest at high cardinality (q3, q5) — the opposite of what a naive
"GPUs are good at parallelism" intuition predicts.

## Why we lose

### Diagnosis at low cardinality (q1, q2)

For q1 with 100 groups, the GPU kernel issues ≈ 10 M `atom.global.add.s32`
instructions, all targeting one of 100 slots in the global hash table.
Every warp's 32 threads contend for the same handful of cache lines, and
the hardware serializes the atomics through the L2 atomic unit. Effective
throughput collapses from "10 M-row scan" to "100 atomic-conflict chains".

DuckDB and Polars **partition input across CPU cores** and keep a
**per-thread hash table in L2/L3**. With 100 groups, each per-thread table
is 100 × (8 + 8) ≈ 1.6 KB — fits trivially in L1. Final merge of N
per-thread tables is one short scan. No serialization, no atomic
contention.

This is why DuckDB hits 1.6 Gelem/s on q1: it is approximately doing
SIMD-vectorised reads into a private hash table per core. The GPU has to
do something fundamentally different to compete.

### Diagnosis at high cardinality (q5)

For q5 with 1 M groups, the global hash table is 1 M slots × 16 B ≈ 16 MB —
larger than L2 on a 2060. Every `atom.global.add` is now mostly an L2 miss
plus an atomic, scattered across DRAM. We're memory-bound on a random-
access pattern. The CPU engines win here too because:

- They build per-thread hash tables in cache and **merge** at the end.
- The merge step happens once over the much smaller set of N × K cells
  (N = thread count, K = group count), and most of those merges are L2 hits.

The GPU's wide-DRAM bandwidth (336 GB/s on a 2060) doesn't help when the
access pattern is purely random.

### Diagnosis at multi-key (q3)

q3 has ~1 M distinct (id1, id2) tuples. The current `groupby_with_pre`
path emits a pre-aggregation projection that materialises a packed key
column, then runs the same single-global-hash-table reduction as q5. Same
problem: random DRAM accesses across a 16 MB table.

## What to do about it

The numbers above are not a bug — they're the cost of the current kernel
design. Closing the gap needs an algorithmic change, not micro-optimisation.

Below is a ranked proposal: tier-1 attacks the **block-level contention**
that dominates low / medium cardinality, tier-2 attacks the **DRAM-random-
access** pattern that dominates high cardinality, tier-3 is polish.

---

### Tier 1 — Per-block shared-memory pre-aggregation

**Idea.** Each CUDA block builds a **shared-memory hash table** over its
slice of the input rows, then performs **one atomic merge per non-empty
slot** into the global table. Atomic traffic into the global table drops
by a factor proportional to the block size (typically 128 – 256).

Pseudocode for a SUM-by-key kernel, sketched at the PTX level:

```
__global__ void gby_sum(const int32_t* keys,
                        const double*  vals,
                        double*        out,
                        uint32_t       n_rows,
                        uint32_t       n_groups) {
    __shared__ double  block_acc[BLOCK_GROUPS];   // BLOCK_GROUPS = 128
    __shared__ uint8_t block_set[BLOCK_GROUPS];

    // 1. Zero the block-local table cooperatively.
    if (threadIdx.x < BLOCK_GROUPS) {
        block_acc[threadIdx.x] = 0.0;
        block_set[threadIdx.x] = 0;
    }
    __syncthreads();

    // 2. Each thread scans a strided chunk of input rows and accumulates
    //    into the SHARED table. If the key doesn't fit, fall back to a
    //    direct global atom.add (the "overflow" path).
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_rows;
         i += gridDim.x * blockDim.x) {
        uint32_t slot = keys[i] % BLOCK_GROUPS;   // mod is cheap, key is bounded
        atomicAdd(&block_acc[slot], vals[i]);
        block_set[slot] = 1;
    }
    __syncthreads();

    // 3. Merge non-empty block slots into the global table.
    if (threadIdx.x < BLOCK_GROUPS && block_set[threadIdx.x]) {
        atomicAdd(&out[threadIdx.x], block_acc[threadIdx.x]);
    }
}
```

**Why it works.** Shared-memory atomics on the same address run through
the shared-memory atomic unit at multi-banks-per-cycle, NOT through the L2.
A 256-thread block doing 10 M / num_blocks = ~40 K rows funnels them into
128 slots locally, then issues only `min(128, distinct_keys_in_block)`
global atomics. At 256 blocks (≈ what we'd launch for 10 M rows at 1024
threads/block), global atomic traffic drops from 10 M to ~32 K — a 300×
reduction.

**Expected impact on the bench.**

- q1 (100 groups): 269 ms → ~10–15 ms. Should beat Polars.
- q2 (10 K groups): the block-local table can't hold 10 K — see Tier 2 below.

**Caveats.**

- `BLOCK_GROUPS` must equal or exceed the actual group count for this
  variant. For unknown / large cardinality, we need to either size the
  block-local table dynamically (impossible in shared memory) or fall
  through to the global atomic on overflow. The right move is a **two-
  variant kernel**: emit the block-local variant when `n_groups ≤ 1024`
  (a single block-shared table fits in 32 KB), else fall through to
  Tier 2.
- We need `n_groups` known at codegen time, or as a launch parameter.
  Currently `groupby_valid.rs` already computes the key table size — we
  can plumb that through.

**Engineering scope.** 1 new PTX kernel template in `jit/`, dispatch tweak
in `groupby_valid.rs`. ~2–3 days.

---

### Tier 2 — Hash-partitioned two-pass aggregation

**Idea.** When the group count exceeds what fits in shared memory, partition
input rows into K disjoint device buffers by `hash(key) % K`, then run a
smaller per-partition GROUP BY in a second pass. Each per-partition
hashtable is small enough to live in L2; cross-partition results are
disjoint by construction so the final concatenation is a single memcpy
per output column.

Pseudocode (two kernels):

```
// Pass 1 — count partition sizes, then scatter rows into partitions.
//   partition_counts[k]  = number of rows in partition k
//   partition_offsets[k] = prefix-sum(partition_counts)[k]
//   partition_keys[off+i] / partition_vals[off+i] = scattered row i in part k
gby_partition(keys, vals, partition_keys, partition_vals,
              partition_offsets, n_rows, K);

// Pass 2 — for each partition, run the Tier-1 block-local groupby.
//   Each partition's groups all hash to k, so there are at most n_groups / K
//   distinct keys per partition. Choose K so this fits in shared mem.
for (uint32_t k = 0; k < K; ++k) {
    gby_block_local<<< … >>>(
        partition_keys + partition_offsets[k],
        partition_vals + partition_offsets[k],
        out_partial[k],
        partition_counts[k],
        n_groups / K);          // ≤ BLOCK_GROUPS
}

// Pass 3 — concat per-partition outputs.
memcpy_concat(out_final, out_partial, K);
```

**Why it works.** For q5 (1 M groups) and K = 4096 partitions, each
partition holds ~250 keys — comfortably under `BLOCK_GROUPS = 1024`. The
scatter is one DRAM read + one DRAM write per row, totally coalesced.
Pass 2 then runs at Tier-1 speed.

**Expected impact.**

- q5 (1 M groups): 770 ms → ~80–120 ms. Should beat DuckDB.
- q3 (~1 M two-key groups): 704 ms → ~80–120 ms. Should comfortably beat
  both competitors.

**Caveats.**

- Adds two device-side scratch buffers sized to the input (16 + 80 MB at
  N = 10 M). The memory pool absorbs this gracefully.
- The partition kernel itself uses `atomicAdd` on the per-partition counter
  vector — but counters are only K = 4096 slots, so it's strictly cheaper
  than the current bottleneck. Putting the counters in shared memory makes
  it cheaper still.
- Scatter pattern is write-coalesced within a partition but
  random across partitions. That's fine: writes are pure bandwidth.

**Engineering scope.** 2 new PTX kernels, scatter logic in `exec/groupby`,
new dispatch heuristic ("if n_groups > 1024, two-pass"). ~1 week.

---

### Tier 3 — Drop-in: cuCo `static_map`

NVIDIA maintains
[**cuCollections (cuCo)**](https://github.com/NVIDIA/cuCollections), a
header-only C++ library of GPU-optimised concurrent containers. Its
`cuco::static_map` is the workhorse hash table behind RAPIDS cuDF's
groupby. If we're willing to take a C++ dependency, dropping it in
end-to-end probably matches or beats Tier 2 with less custom kernel code.

**Pros.** Battle-tested, multi-probing, hardware-aware. Almost certainly
faster than anything we'd write ourselves.

**Cons.** Adds a C++ build dependency (cuCo is header-only but uses CUB
internally; CUB is also header-only but adds compile time). We'd lose
"all the kernels are generated by our JIT" as a clean architectural story.

Verdict: **keep Tier 3 as a Plan B** if Tier 1 + Tier 2 don't close the gap
to within 1.5× of DuckDB on all four queries.

---

### Tier 4 — Polish (after Tier 1 + 2 land)

These are 1.1×–1.5× wins each, worth chasing only after the algorithmic
fixes are in.

| Optimization                                          | Where                          | Estimated gain |
| ----------------------------------------------------- | ------------------------------ | -------------- |
| Vectorised i32 key loads (4× per `ld.global.v4.u32`)  | `groupby_valid.rs` kernel emit | 1.1–1.3×       |
| Async D2H of the result while host decodes prior      | `exec/engine.rs`               | 1.05–1.1×      |
| Block size sweep (256 vs 512 vs 1024)                 | `BLOCK_SIZE` in `engine.rs`    | 1.05–1.2×      |
| Precompute key hashes at `register_table` time        | `gpu_table.rs`                 | 1.1×           |
| Use `atom.add.noftz.f64` where lower precision is OK  | `jit/valid_flag_kernels.rs`    | 1.05× for f64  |

---

## Recommended order of work

1. **Tier 1** (per-block shared-mem). One new kernel template. Should
   immediately close the gap on q1 / q2 and may already be sufficient for q4
   (AVG sees the same low-cardinality pattern, now that the `atom.add.s64`
   → `.u64` codegen bug is fixed and AVG runs at all).
2. **Measure.** If Tier 1 alone gets all four queries within 2× of DuckDB,
   stop and rebench publicly. If q3 / q5 still lose, proceed to Tier 2.
3. **Tier 2** (two-pass partitioning). Required for q5 (1 M groups) and
   any future workload above ~1 K distinct groups.
4. **Re-measure with the full bench.** This is also the natural moment to
   add q4 / q7 / q8 from the h2o.ai spec, since they all share the
   groupby kernel.
5. **Tier 4** polish in whichever order the profiler points at.

## What to read while implementing

- NVIDIA, *GPU Hash Tables for Graph Analytics* (Awad et al., 2020) —
  partitioning + warp-cooperative probing.
- DuckDB blog, *The Vectorised Volcano in C++* — the per-core hashtable
  + merge pattern we're trying to match on the GPU.
- cuDF docs, *Hash Aggregation on the GPU* — production reference for
  Tier 1 + Tier 2 combined.
- NVIDIA, *cuCo static_map* README — the Plan B drop-in.
