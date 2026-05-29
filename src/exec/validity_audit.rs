// SPDX-License-Identifier: Apache-2.0

//! NULL / validity-consumption audit matrix for the GROUP BY executors.
//!
//! This module is **documentation-only** — it carries no runtime code. It
//! records, per GROUP BY path, whether NULL inputs are consumed *on the
//! device* (a validity-aware `_with_validity` kernel reads a packed-bit
//! bitmap and skips null rows, folding each aggregate to its identity) or
//! *on the host* (rows whose input is NULL are stripped before upload). The
//! goal is a single auditable place that answers "does path X honour nulls
//! natively, and if not, why not?".
//!
//! ## Packed-bit validity convention
//!
//! Every native path uses the same layout as
//! [`crate::jit::hash_kernels`]: 1 bit per row, 32 rows per `u32` word,
//! little-endian bit order (bit `0` = first row of the word). A kernel
//! computes `word = tid >> 5`, `bit = tid & 31`, loads the word through the
//! read-only cache, and extracts the bit with `bfe.u32`. A `0` bit means
//! "row is NULL" and the thread skips its contribution. This matches Arrow's
//! null-buffer convention so the host can upload the Arrow validity buffer
//! (re-packed) directly.
//!
//! ## Matrix
//!
//! | Path                                   | Kernel / dispatch                                                                 | NULL consumption |
//! |----------------------------------------|----------------------------------------------------------------------------------|------------------|
//! | Scalar aggregate (no GROUP BY)         | [`crate::jit::agg_kernels`] reduction kernels                                     | host-strip (legacy) |
//! | Single-key, single-agg                 | [`crate::jit::hash_kernels::compile_groupby_agg_kernel_with_validity`]            | **native** (per-row `bfe.u32` gate) |
//! | Single-key keys-build                  | [`crate::jit::hash_kernels::compile_groupby_keys_kernel_with_validity`]           | **native** (NULL keys dropped) |
//! | Single-key, multi-agg (per-agg launch) | N× [`crate::jit::hash_kernels::compile_groupby_agg_kernel_with_validity`]         | **native** (each agg launch gated) |
//! | Single-key, multi-agg (fused)          | [`crate::jit::hash_kernels::compile_groupby_agg_kernel_multi_with_validity`]      | **native** (per-spec `bfe.u32` guard) |
//! | Single-key float MIN/MAX               | [`crate::jit::float_atomics`] CAS-loop kernel                                     | host-strip (no `_with_validity` companion) |
//! | Two-key (tier2) integer aggs           | `groupby_tier2_twokey_*_exec`                                                     | host-strip (see below) |
//! | Two-key (tier2) multi-agg              | [`crate::exec::groupby_tier2_twokey_multi_exec`]                                  | host-strip (see below) |
//! | Tier-2 partitioned single-key multi    | [`crate::exec::groupby_tier2_multi_exec`]                                         | host-strip (see below) |
//! | Shared-mem direct-mapped (`shmem_*`)   | [`crate::jit::shmem_multi_sum_kernel`] et al.                                     | host-strip (see below) |
//!
//! ## Native multi-aggregate (fused) — completed
//!
//! The fused single-key multi-aggregate kernel now has a validity-aware
//! companion,
//! [`crate::jit::hash_kernels::compile_groupby_agg_kernel_multi_with_validity`].
//! It appends one packed-bit validity pointer per validity-carrying spec
//! (after the `n_rows` / `k` scalars, so the no-validity ABI is unchanged)
//! and emits a per-spec `bfe.u32` null-guard before each aggregate's atomic.
//! A NULL row branches past that spec's atomic to a `SPEC_SKIP_j` landing
//! pad, folding the row to that aggregate's identity (SUM/COUNT contribute
//! nothing; MIN/MAX leave the slot untouched) while the *other* fused
//! aggregates for the same row proceed normally. This is the multi-column
//! generalisation of the single-agg gate: each fused aggregate reads its own
//! input column, so each needs its own bitmap and its own guard.
//!
//! The fused dispatch flip in [`crate::exec::groupby`] (TODO L3) is still a
//! separate plumbing change; once it lands, the validity-carrying case can
//! route to the fused `_with_validity` emitter instead of N per-agg launches.
//! Until then the per-agg path already consumes nulls natively, so there is
//! no correctness gap — only the not-yet-realised fusion *perf* win.
//!
//! ## Why the remaining paths still host-strip
//!
//! The following paths intentionally retain host-side NULL stripping rather
//! than a half-implemented device gate. Each is a deliberate scope boundary,
//! not an oversight:
//!
//! * **Two-key (tier2) paths.** The packed two-column key is built host-side
//!   and the per-aggregate kernels are a larger family (count / minmax /
//!   minmax-float / avg / multi), each with its own ABI. Threading a per-agg
//!   validity bitmap through the tier2 orchestrator + merge stages touches
//!   every member of that family at once; doing it piecemeal risks an ABI
//!   skew between the orchestrator and an individual kernel. Left as a
//!   follow-up so the two-key family can be converted as one coherent unit.
//!
//! * **Tier-2 partitioned single-key multi-SUM.** Validity would have to
//!   survive the partition → scatter → reduce pipeline (the bitmap must be
//!   permuted alongside the values during scatter). That is a cross-stage
//!   change to the partition/scatter kernels, out of scope for the
//!   single-pass open-addressing work here.
//!
//! * **Shared-mem direct-mapped (`shmem_*`).** Float64 SUM only; the
//!   block-local accumulate + merge structure would need the validity bit
//!   checked in the grid-stride row loop before each `atom.shared.add`. A
//!   reasonable follow-up, but a different kernel family from the global
//!   open-addressing kernel this work extends.
//!
//! * **Float MIN/MAX.** Implemented as a CAS loop in
//!   [`crate::jit::float_atomics`], which has no `_with_validity` companion.
//!   The single-key dispatch predicate already excludes this combo from the
//!   native path and routes it through host-strip.
//!
//! When any of these is later converted, move its row in the matrix above to
//! **native** and delete the corresponding bullet here.
