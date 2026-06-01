# Section-5 Direction Features (2026-05-30)

Four dedicated Opus agents, one per direction, disjoint files. Library compiles
clean (`cargo build --features cuda-stub`); 1801 host lib tests pass / 0 fail /
121 ignored (GPU-gated); all test+bench targets compile. GPU paths are
compile-verified + reasoned only (no CUDA device here) — runnable via
`BOLT_BENCH_GPU=1 cargo test -- --ignored`.

## 1. Cost-based optimizer (CBO)
- New `src/plan/optimizer/cost.rs`: Selinger-style cardinality model
  (`|A||B|/max(ndv)`, containment fallback) + cost = Σ intermediate cardinalities;
  bitmask-DP enumeration producing **bushy** plans (≤10 relations, greedy fallback
  beyond), connected-only (never invents cross products), smaller side → build side.
- Wired into `join_reorder.rs` (replaces left-deep-only heuristic); semantics-preserving,
  no-op when stats/shape unknown. `estimate_equijoin_rows` added to `statistics.rs`.
- `pub mod cost;` added to `optimizer/mod.rs`. Predicate-preservation + bushy + no-op tests.
- Remaining: operator costs beyond joins, NDV/histograms plumbed end-to-end, outer/semi reorder.

## 2. GPU DISTINCT
- `src/exec/distinct.rs`: GPU sort-based dedup (`BOLT_GPU_DISTINCT=1`, default off) —
  GPU sort → adjacent-distinct flag → prefix-scan+gather; host fallback for
  multi-key/Utf8/unsupported dtype/`GpuCapacity`.
- New `src/jit/distinct_kernel.rs`: adjacent-distinct flag kernel (i32/i64/f32/f64,
  nullable), SQL NULL-collapse + float canonicalization. `pub mod distinct_kernel;` added.
- Host masking core unit-tested; golden PTX tests; device round-trip `#[ignore="gpu:distinct"]`.
- Remaining: multi-key, Utf8, hash-based alternative, true on-device mask (avoid D2H of sorted col).

## 3. GPU window functions
- `src/exec/window.rs`: GPU dispatch for ROW_NUMBER / RANK / DENSE_RANK / running SUM
  via sort + segmented scan; host fallback for other fns/frames/types. Device launch
  currently returns `Ok(None)` (host fallback) until GPU-validated.
- New `src/jit/window_kernel.rs`: segmented rank / running-sum / partition-boundary
  emitters. `pub mod window_kernel;` added. Host derivation math unit-tested; golden PTX tests.
- Remaining: actual device launch wiring, RANGE/ROWS frames, LAG/LEAD, more aggregates.

## 4. Streaming-to-device
- `src/exec/streaming.rs`: real `MorselDriver` (morsel-at-a-time, `PinnedBudget`-bounded
  async H2D via pinned buffers, double-buffer-ready), honest `classify_operator`
  (projection/filter/partial-agg stream; sort/distinct/window/agg-final/join-build/setop
  materialize), corrected docstrings (removed the overselling + stale `TODO(cuda)`).
- Self-contained — no `engine.rs` edit. Engine activation snippet provided by the agent as a
  follow-up (NOT applied: it references engine internals that must be verified first).
- Host-only logic fully unit-tested; device round-trips `#[ignore="gpu:stream"]`.
- Note: per-morsel budget accounting is conservative for *sliced* morsels
  (`estimate_batch_bytes` counts shared parent buffers) — refine to logical-row sizing
  when wiring into the engine.

## Follow-ups
- GPU validation on a device for all four (`BOLT_BENCH_GPU=1 ... -- --ignored`).
- Apply the streaming engine.rs activation snippet after verifying engine internals.
- Wire GPU window device launch (currently host-fallback).
