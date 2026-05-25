# Kernel Inventory & Complexity Assessment (0.3 rust-cuda Migration)

Every PTX-emitting module in `src/jit/` characterised for the rust-cuda port. "PTX lines" counts `writeln!(ptx,...)` calls (actual PTX ≈ 1.0-1.2×). `jit_compiler.rs` is the PTX→`CUmodule` loader — covered at the end. Complexity bins: **Trivial** = pure compute. **Moderate** = shared mem OR atomics, single phase. **Hard** = open-addressing CAS or multi-phase sync. **Research** = CAS loop on float bit patterns / MIN-MAX without native atomic.

## Per-kernel breakdown

### 1. `src/jit/ptx_gen.rs` — projection / arithmetic row kernel
- **Entry name:** dynamic (caller-supplied). **PTX lines:** ~30 prologue + 3-8 per `Op`.
- **Shape:** One thread per row. Compute lineage of loads/casts/binary ops, optional `setp`-gated stores. No cross-thread coordination.
- **Shared mem:** No. **Atomics:** None. **Complexity: Trivial** — pure SIMT compute; the hard work is the *codegen layer* (lowering `KernelSpec::Op` to Rust source instead of PTX strings).
- **`cuda_std` needs:** `#[kernel]`/`extern "ptx-kernel"`, `thread::index_1d`, raw global-pointer arithmetic.
- **Cost:** **3-5 days** (codegen rewrite via proc-macro / quote!). The runtime kernel itself is trivial.

### 2. `src/jit/scan_kernel.rs` — predicate / filter mask
- **Entry name:** dynamic, typically `patina_predicate`. **PTX lines:** ~25 prologue + 3-8 per `Op` + 6 mask-store tail.
- **Shape:** One thread per row. Re-runs projection compute for predicate only, narrows `b32`-bool to `u8`, stores one mask byte/row.
- **Shared mem:** No. **Atomics:** None. **Complexity: Trivial** — shares everything with #1; only tail differs.
- **`cuda_std` needs:** Same as #1.
- **Cost:** Folds into #1. Marginal **0.5-1 day**.

### 3. `src/jit/agg_kernels.rs` — scalar reduction
- **Entry name:** `patina_reduce`. **PTX lines:** ~130 (53 fixed + 3 unrolled inter-warp strides + 5 unrolled warp-shuffle strides).
- **Shape:** Two-phase per-block reduction — (1) inter-warp tree on shared mem with `bar.sync` at strides 128/64/32; (2) intra-warp `shfl.sync.down.b32` for strides 16/8/4/2/1. Block partials go to host for cross-block final. Parameterised over ReduceOp × DataType (16 combinations).
- **Shared mem:** Yes (`BLOCK_SIZE × acc_elem_bytes` ≤ 2 KiB). **Atomics:** None (host does cross-block).
- **Complexity: Moderate** — shared mem + warp shuffle; 64-bit shuffle needs manual hi/lo b32 split on sm_70.
- **`cuda_std` needs:** `shared_array!`, `thread::sync_threads`, `cuda_std::warp::shfl_down`. For b64 split may need `core::arch::asm!`.
- **Cost:** **2-3 days** including f64/i64 split path.

### 4. `src/jit/hash_kernels.rs` — legacy sentinel-based GROUP BY (keys + agg)
- **Entry names:** `patina_groupby_keys`, `patina_groupby_agg`. **PTX lines:** 103 (~45 + ~58).
- **Shape:** Open-addressing hash, linear probe, `i64::MIN` empty sentinel. **Keys:** splitmix-hash, `atom.global.cas.b64` claim, bounded probe (`2*k`). **Agg:** re-hash, non-mutating probe, single `atom.global.<op>.<dtype>` on slot. Two kernels host-sequenced.
- **Shared mem:** No. **Atomics:** `atom.global.cas.b64`; `atom.global.add.{s32,u64,f32,f64}`, `atom.global.{min,max}.{s32,s64}`.
- **Complexity: Hard** — CAS open-addressing + cross-kernel ordering contract + wide op×dtype matrix.
- **`cuda_std` needs:** `AtomicI64::compare_exchange`, dtype-generic `AtomicT::fetch_{add,min,max}`. May need `asm!` for `atom.add.u64` on signed i64 (no `.s64` variant).
- **Cost:** **4-5 days** — two kernels, 8-way matrix, cross-kernel sequencing tests.

### 5. `src/jit/float_atomics.rs` — float MIN/MAX via CAS loop (legacy path)
- **Entry name:** `patina_groupby_agg` (overloads #4 agg; host selects). **PTX lines:** ~59.
- **Shape:** Probe lookup (as #4 agg) + CAS retry on accumulator raw bits: reinterpret via `mov.bXX/fXX`, `setp.lt/gt.fXX`, `selp.fXX`, `atom.global.cas.bXX`, retry on race-loss. Skip-on-no-change handles NaN inputs silently.
- **Shared mem:** No. **Atomics:** `atom.global.cas.b32`, `atom.global.cas.b64`.
- **Complexity: Research** — CAS loop on float bit patterns; PTX lacks `atom.{min,max}.f*` on sm_70.
- **`cuda_std` needs:** `AtomicU32/U64::compare_exchange`, `f32::to_bits`/`f64::to_bits`. May need `asm!` if cuda_std CAS doesn't preserve "return-old".
- **Cost:** **3-4 days** including NaN tests.

### 6. `src/jit/valid_flag_kernels.rs` — sentinel-free GROUP BY (keys + agg)
- **Entry names:** `patina_groupby_keys_valid`, `patina_groupby_agg_valid`. **PTX lines:** 187 across both.
- **Shape:** Three-state slot lifecycle (0=empty/1=claimed/2=committed) via parallel `slot_valid: u32[k]`. **Keys:** `atom.global.cas.b32` on slot_valid (0→1), `st.global.s64` key, `membar.gl`, `atom.global.exch.b32` to publish (→2). Losers SPIN on `slot_valid==2` (`SPIN_LIMIT=1024`). Bounded outer PROBE (`2*k`). Both bounds fall through to host-allocated SPILL via `atom.global.add.u32` on a counter.
- **Shared mem:** No. **Atomics:** `atom.global.cas.b32`, `atom.global.exch.b32`, `atom.global.add.u32`, `atom.global.<op>.<dtype>`; `membar.gl`.
- **Complexity: Hard** — multi-phase sync with memory barrier + spill fallback + deadlock-safety reasoning.
- **`cuda_std` needs:** `AtomicU32::compare_exchange/swap`, `cuda_std::mem::fence_gl` or `asm!("membar.gl")`, 8-way op×dtype agg matrix.
- **Cost:** **1 week** — three-state protocol, spill ABI, op×dtype matrix.

### 7. `src/jit/valid_flag_float.rs` — valid-flag float MIN/MAX
- **Entry name:** `patina_groupby_agg_valid` (overloads #6 agg). **PTX lines:** 105.
- **Shape:** SPIN-on-(slot_valid==2) probe from #6 plus float-CAS accumulator loop from #5. 11-parameter ABI including spill triplet.
- **Shared mem:** No. **Atomics:** `atom.global.cas.b{32,64}`, `atom.global.add.u32` (spill); reads `slot_valid`.
- **Complexity: Research** — composes #5 + #6.
- **`cuda_std` needs:** Union of #5 + #6.
- **Cost:** **3-4 days** if #5+#6 already ported, else 1 week.

### 8. `src/jit/prefix_scan.rs` — Hillis-Steele scan + per-dtype gather
- **Entry names:** `patina_prefix_scan`; `patina_gather_{bool,i32,i64,f32,f64}`. **PTX lines:** 119.
- **Shape:** **Scan:** load u8 mask → coerce 0/1 → stash in ping `sdata`, Hillis-Steele unroll (log₂(256)=8 rounds) ping-ponging two `sdata` buffers with `bar.sync`/round, store exclusive scan; last thread stores block_sum. **Gather:** read mask, look up `local_indices[gid]` + `block_bases[blockIdx.x]`, copy `input[gid]` → `output[base+local_idx]`.
- **Shared mem:** Scan: 2×(`BLOCK_SIZE × 4`) = 2 KiB. Gather: none. **Atomics:** None.
- **Complexity:** Scan **Moderate**, gather **Trivial**.
- **`cuda_std` needs:** `shared_array!`, `sync_threads`. Optionally `warp::shfl_up` to reduce barrier count.
- **Cost:** **2-3 days** scan, **1 day** gather family (generic over T:Copy).

### 9. `src/jit/prefix_scan_multipass.rs` — recursive scan + add-bases
- **Entry names:** `patina_prefix_scan_u32`, `patina_add_block_bases`. **PTX lines:** 94.
- **Shape:** **scan_u32:** identical to #8 scan but reads u32 input (no u8 coercion); used recursively against deeper-level block_sums. **add_block_bases:** `indices[i] += block_bases[i / BLOCK_SIZE]` — pure compute.
- **Shared mem:** scan_u32 2 KiB, add_bases none. **Atomics:** None.
- **Complexity:** scan_u32 **Moderate**, add_bases **Trivial**.
- **`cuda_std` needs:** Same as #8.
- **Cost:** **1-2 days** if #8 parameterised over input type.

### 10. `src/jit/shmem_sum_kernel.rs` — Tier-1 GROUP BY SUM (f64, direct-mapped)
- **Entry name:** `patina_groupby_shmem_sum_f64`. **PTX lines:** 93.
- **Shape:** Three-phase. (1) Cooperative zero of `block_acc[1024]` + `block_set[1024]`, `__syncthreads`. (2) Grid-stride accumulate — direct-map `key & (BLOCK_GROUPS-1)` slot, `atom.shared.add.f64`, non-atomic `block_set[key]=1`; overflow (`key ≥ BLOCK_GROUPS`) goes straight to `atom.global.add.f64`. (3) `__syncthreads`, merge non-empty slots via `atom.global.add.f64`.
- **Shared mem:** Yes (`8 KiB acc + 1 KiB set` = 9 KiB). **Atomics:** `atom.shared.add.f64`, `atom.global.add.f64`.
- **Complexity: Moderate** — shared mem + dual-tier atomics + three-phase barriers; no CAS.
- **`cuda_std` needs:** `shared_array!`, `sync_threads`, `AtomicF64::fetch_add` in **shared** and global address spaces. Shared-state atomics may need `asm!` fallback.
- **Cost:** **3-4 days** — first shared-atomic encounter; templates the shmem_* family.

### 11. `src/jit/shmem_multi_sum_kernel.rs` — Tier-1 multi-SUM
- **Entry name:** `patina_groupby_shmem_multi_sum_f64_{n_vals}` (1..=4). **PTX lines:** ~70 scaffolding + per-aggregate unroll (~600 lines @ n_vals=4).
- **Shape:** Same three-phase as #10 with N parallel f64 accumulators per slot. Per-aggregate work unrolled at emit time. One `block_set` shared across N.
- **Shared mem:** Up to 40 KiB @ n_vals=4 (`4×8 KiB vals + 4 KiB set`). **Atomics:** N × `atom.shared.add.f64`/row; N × `atom.global.add.f64`/non-empty slot.
- **Complexity: Moderate** — #10 generic over N (const-generic `<const N: u32>`).
- **`cuda_std` needs:** Same as #10 + confirm const-generic kernel params.
- **Cost:** **2 days** after #10.

### 12. `src/jit/shmem_count_kernel.rs` — Tier-1 COUNT(*)
- **Entry name:** `patina_groupby_shmem_count_u64`. **PTX lines:** 88.
- **Shape:** Same three-phase as #10 with u64 accumulator; per-row update `atom.shared.add.u64(slot, 1)`. No value column.
- **Shared mem:** Yes (`8 + 1` = 9 KiB). **Atomics:** `atom.shared.add.u64`, `atom.global.add.u64`.
- **Complexity: Moderate** — #10 with u64 element.
- **`cuda_std` needs:** `AtomicU64::fetch_add` in both address spaces.
- **Cost:** **1-2 days** after #10.

### 13. `src/jit/shmem_minmax_kernel.rs` — Tier-1 MIN/MAX (int)
- **Entry names:** `patina_groupby_shmem_{min,max}_{i32,i64}` (4 variants). **PTX lines:** 92.
- **Shape:** Same three-phase as #10. Slot init: `iN::MAX` (MIN) / `iN::MIN` (MAX). Per-row: `atom.shared.{min,max}.{s32,s64}`.
- **Shared mem:** ~8-12 KiB. **Atomics:** `atom.shared.{min,max}.{s32,s64}`, global mirror at export.
- **Complexity: Moderate** — #10 family, no CAS. Float deferred to #24.
- **`cuda_std` needs:** `AtomicI32/I64::fetch_min/fetch_max` in shared + global.
- **Cost:** **1-2 days** after #10.

### 14. `src/jit/partition_kernel.rs` — Tier-2 hash partition (i32 key)
- **Entry name:** `patina_partition`. **PTX lines:** 49.
- **Shape:** Grid-stride loop. Per row: `mul.lo.u32` Knuth-hash i32 key, mask to `NUM_PARTITIONS=4096`, `atom.global.add.u32(&counts[pid], 1)` (return discarded), `partition_ids[i]=pid`.
- **Shared mem:** No. **Atomics:** `atom.global.add.u32`.
- **Complexity: Moderate** — single global atomic, no shared mem, no CAS. Simplest atomic kernel in the codebase.
- **`cuda_std` needs:** `AtomicU32::fetch_add` (global).
- **Cost:** **1 day**.

### 15. `src/jit/partition_kernel_i64.rs` — Tier-2 hash partition (i64 key)
- **Entry name:** `patina_partition_i64`. **PTX lines:** 50.
- **Shape:** Same as #14 with i64 key: `mul.lo.u64` × 64-bit Knuth constant, `shr.u64 ..., 54` for top-10-bit partition id.
- **Shared mem:** No. **Atomics:** `atom.global.add.u32`.
- **Complexity: Moderate** — sibling of #14.
- **`cuda_std` needs:** Same as #14.
- **Cost:** **0.5 day** after #14.

### 16. `src/jit/scatter_kernel.rs` — Tier-2 scatter (i32 key)
- **Entry name:** `patina_scatter`. **PTX lines:** 70.
- **Shape:** Per row: `local_idx = atomicAdd(&cursors[pid], 1)` (OLD value), `out_pos = offsets[pid] + local_idx`, copy `(key, val)` to `out[out_pos]`. No barriers.
- **Shared mem:** No. **Atomics:** `atom.global.add.u32` (OLD-value-as-index pattern).
- **Complexity: Moderate** — atomic-fetch-add + scatter store.
- **`cuda_std` needs:** `AtomicU32::fetch_add` (standard return-prev semantics).
- **Cost:** **1 day**.

### 17. `src/jit/scatter_kernel_i64.rs` — Tier-2 scatter (i64 key)
- **Entry name:** `patina_scatter_i64`. **PTX lines:** 69.
- **Shape:** Same as #16 with `ld.global.s64`/`st.global.u64` for the key.
- **Shared mem:** No. **Atomics:** `atom.global.add.u32`.
- **Complexity: Moderate** — sibling of #16.
- **`cuda_std` needs:** Same as #16.
- **Cost:** **0.5 day** after #16.

### 18. `src/jit/partition_reduce_kernel.rs` — Tier-2.1 per-partition reduce (i32 key, f64 SUM)
- **Entry name:** `patina_partition_reduce`. **PTX lines:** 125.
- **Shape:** One block per partition. Cooperatively zero `block_keys[1024]:i32`, `block_vals[1024]:f64`, `block_set[1024]:u32`. Walk slice `[offsets[pid]..offsets[pid+1])` grid-stride. Linear probe: `atom.shared.cas.b32` on `block_set[slot]` (0→1); winner writes key + `atom.shared.add.f64`; key-match → `atom.shared.add.f64`; collision → `(slot+1) & mask` up to `MAX_PROBES=1024`. After `__syncthreads`, first BLOCK_GROUPS threads export to global indexed `pid*BLOCK_GROUPS+slot`.
- **Shared mem:** Yes (`4 + 8 + 4` = 16 KiB). **Atomics:** `atom.shared.cas.b32`, `atom.shared.add.f64`; global mirror at export.
- **Complexity: Hard** — open-addressing CAS in shared mem + multi-phase sync.
- **`cuda_std` needs:** `AtomicU32::compare_exchange` in **shared** address space, `AtomicF64::fetch_add` in shared. Likely needs `asm!` if cuda_std lacks typed shared-state CAS.
- **Cost:** **1 week** — templates the #19-#24 family.

### 19. `src/jit/partition_reduce_kernel_i64.rs` — Tier-2.1 reduce (i64 key, f64 SUM)
- **Entry name:** `patina_partition_reduce_i64`. **PTX lines:** 124.
- **Shape:** Same as #18 with `block_keys[1024]:i64`; slot index from low 32 bits of key.
- **Shared mem:** Yes (`8 + 8 + 4` = 20 KiB). **Atomics:** `atom.shared.cas.b32`, `atom.shared.add.f64`.
- **Complexity: Hard** — sibling of #18.
- **`cuda_std` needs:** Same as #18.
- **Cost:** **2 days** after #18.

### 20. `src/jit/partition_reduce_kernel_multi.rs` — Tier-2.1 reduce (i32 key, N×f64 SUM)
- **Entry name:** `patina_partition_reduce_multi_sum_{n_vals}` (1..=4). **PTX lines:** 117 scaffolding + N-unrolled body.
- **Shape:** Same as #18 with N parallel f64 accumulators per slot. N × `ld.global.f64` + N × `atom.shared.add.f64`/row; N × `atom.global.add.f64`/non-empty slot at export.
- **Shared mem:** 16-40 KiB across n_vals=1..4. **Atomics:** `atom.shared.cas.b32`, N × `atom.shared.add.f64`, N × `atom.global.add.f64`.
- **Complexity: Hard** — #18 generic over N.
- **`cuda_std` needs:** Same as #18 + const-generic `<const N: u32>`.
- **Cost:** **2 days** after #18.

### 21. `src/jit/partition_reduce_kernel_multi_i64.rs` — Tier-2.1 reduce (i64 key, N×f64 SUM)
- **Entry name:** `patina_partition_reduce_multi_sum_i64_{n_vals}` (1..=4). **PTX lines:** 118.
- **Shape:** Intersection of #19 (i64 key) and #20 (N values).
- **Shared mem:** 20-44 KiB across n_vals=1..4. **Atomics:** Same as #20.
- **Complexity: Hard** — composes #19 + #20.
- **`cuda_std` needs:** Same as #18-#20.
- **Cost:** **1-2 days** after #19 + #20.

### 22. `src/jit/partition_reduce_kernel_count.rs` — Tier-2.1 reduce (i32 key, COUNT(*))
- **Entry name:** `patina_partition_reduce_count`. **PTX lines:** 118.
- **Shape:** Same as #18 with u64 accumulator; per-row update `atom.shared.add.u64(slot, 1)`. No value column.
- **Shared mem:** Yes (`4 + 8 + 4` = 16 KiB). **Atomics:** `atom.shared.cas.b32`, `atom.shared.add.u64`, `atom.global.add.u64`.
- **Complexity: Hard** — #18 with u64 accumulator.
- **`cuda_std` needs:** `AtomicU32::compare_exchange` + `AtomicU64::fetch_add` in shared.
- **Cost:** **1-2 days** after #18.

### 23. `src/jit/partition_reduce_kernel_minmax.rs` — Tier-2.1 reduce (int MIN/MAX)
- **Entry names:** `patina_partition_reduce_{min,max}_{i32,i64}` (4 variants). **PTX lines:** 113.
- **Shape:** Same as #18; per-row update `atom.shared.{min,max}.{s32,s64}`. Slot init: `i*::MAX` (MIN) / `i*::MIN` (MAX). `block_set` distinguishes "untouched" from "identity is the answer".
- **Shared mem:** 12-20 KiB. **Atomics:** `atom.shared.cas.b32`, `atom.shared.{min,max}.{s32,s64}`; global mirror.
- **Complexity: Hard** — #18 with `fetch_min`/`fetch_max`.
- **`cuda_std` needs:** `AtomicI32/I64::fetch_min/fetch_max` in shared.
- **Cost:** **2 days** after #18.

### 24. `src/jit/partition_reduce_kernel_minmax_float.rs` — Tier-2.1 reduce (float MIN/MAX)
- **Entry names:** `patina_partition_reduce_{min,max}_{f32,f64}`. **PTX lines:** 120.
- **Shape:** Open-addressing slot claim (as #18) PLUS per-row CAS retry loop on slot's raw bits — PTX lacks `atom.shared.{min,max}.f*` on sm_70. `setp.lt/gt.fXX` + `selp.fXX` + `atom.shared.cas.b{32,64}`.
- **Shared mem:** 12-20 KiB. **Atomics:** `atom.shared.cas.b32` (slot claim) layered with `atom.shared.cas.b{32,64}` (value path).
- **Complexity: Research** — float-CAS in shared mem on top of slot-claim CAS. Most complex kernel in the codebase.
- **`cuda_std` needs:** `AtomicU32/U64::compare_exchange` in shared address space; NaN-aware compare contract. Likely bespoke `asm!`.
- **Cost:** **1 week** even after #18 + #5.

## Non-kernel infrastructure

**`src/jit/jit_compiler.rs`** — owns `CudaModule::from_ptx` (a `cuModuleLoadDataEx` wrapper with PTXAS log capture), a process-wide `OnceLock<Mutex<PtxCache>>` keyed by 64-bit hash of PTX text (FIFO eviction @ 256 entries, hash-collision-safe via stored-text re-compare), and `CudaFunction<'a>`. Under rust-cuda most kernels become pre-built artifacts compiled at *craton-patina* build time; the PTX→CUmodule boundary survives only for dynamic codegen (#1, #2 remain string-emitting because they're parameterised by user query shape), the rest move to static-symbol resolution. **Cost: 2-3 days** to retain the cache for the codegen path plus a static-symbol path for pre-built kernels.

## Summary by complexity

| Bin | Kernels | Count |
| --- | --- | --- |
| **Trivial** | ptx_gen (#1), scan_kernel (#2), gather tail of #8, add_bases tail of #9 | 4 |
| **Moderate** | agg_kernels (#3), scan head of #8, scan_u32 head of #9, shmem_sum (#10), shmem_multi_sum (#11), shmem_count (#12), shmem_minmax (#13), partition (#14), partition_i64 (#15), scatter (#16), scatter_i64 (#17) | 11 |
| **Hard** | hash_kernels (#4), valid_flag_kernels (#6), partition_reduce (#18), partition_reduce_i64 (#19), partition_reduce_multi (#20), partition_reduce_multi_i64 (#21), partition_reduce_count (#22), partition_reduce_minmax (#23) | 8 |
| **Research** | float_atomics (#5), valid_flag_float (#7), partition_reduce_minmax_float (#24) | 3 |

26 distinct kernel entries across 26 files. Total `writeln!(ptx, ...)` calls ≈ 2 136.

## Ordering recommendation — feasibility spike first

**First: `src/jit/partition_kernel.rs` (#14).** No shared memory — defers how `cuda_std` exposes `__shared__` and shared-state atomics. Single global atomic (`atom.global.add.u32`) exercises `AtomicU32::fetch_add`, the bedrock primitive every Tier-2 kernel above #14 also needs — de-risks #15-#23. Tiny: ~49 `writeln!`s. Full port (Rust → PTX → existing `jit_compiler.rs` cache → existing `exec/partition_offsets.rs` launcher) fits in 1-2 days. Already integration-tested via `q5`/`q3` benchmarks.

**Second: `scatter_kernel.rs` (#16)** — same atomic profile, no shared mem; validates the "atomic-fetch-add returning OLD as index" pattern.

**Third: `ptx_gen.rs` (#1) + `scan_kernel.rs` (#2) as a pair** — once kernels can be written in Rust, the next question is whether dataflow-parameterised codegen stays in PTX strings or moves to a Rust IR + `rustc_codegen_nvvm`. Largest code impact, lowest runtime risk.

**Fourth: `agg_kernels.rs` (#3)** — first shared-mem + warp-shuffle. Establishes the pattern for the shmem_* family.

**Fifth: `shmem_sum_kernel.rs` (#10)** — first shared-memory atomic. Once it works, #11-#13 follow by mechanical adaptation; #18 and siblings (#19-#23) are predominantly more of the same.

**Defer until last:** the three **Research** kernels (#5, #7, #24) — all float-CAS-loop variants. The cuda_std design decision (inline `asm!`? `AtomicU{32,64}::compare_exchange` + `f{32,64}::from_bits`?) needs to land once across all three, after the rest of the migration has shaken out the surface area.
