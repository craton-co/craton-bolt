// SPDX-License-Identifier: Apache-2.0

//! Module-cache key + host-revision snapshot types lifted out of
//! `exec::engine` (pure reorganization; no behavior change).
//!
//! Holds the [`ModuleCacheKey`] content-hash used by `Engine::module_cache`
//! and the per-table host-revision tracking types
//! ([`HostTableRevision`], [`ClonedHostRevision`], [`HostRevisionSnapshot`])
//! used by the incremental GpuTable cache.

use std::collections::HashMap;
use std::hash::Hasher;

use crate::plan::KernelSpec;

/// Cache key for [`Engine::module_cache`]: a 128-bit content hash of the
/// `KernelSpec` plus the PTX entry-point name. The entry name distinguishes
/// the two different PTX shapes the projection path can emit for the same
/// spec — the full projection kernel (`KERNEL_ENTRY`) and the
/// predicate-only mask kernel (`PREDICATE_ENTRY`).
///
/// # Why not `#[derive(Hash)]` on `KernelSpec`?
///
/// `KernelSpec` transitively contains `Op::Const { lit: Literal }`, and
/// `Literal` carries `f32`/`f64` constants. Floats do not implement `Hash`
/// (NaN inequality is the canonical reason), so deriving `Hash` on the
/// planner IR would require either a hand-rolled `Hash` over the raw bit
/// patterns of every numeric literal (and matching `PartialEq` so the
/// `Hash`/`Eq` contract holds) or a from-scratch traversal type. Either
/// route reaches far outside this file's blast radius.
///
/// # Hashing strategy
///
/// We keep the "format the IR via `Debug` then hash the bytes" pattern but
/// upgrade two things:
///
/// 1. **128-bit fingerprint.** A single 64-bit `DefaultHasher` exposes a
///    birthday-paradox collision probability of ~1 in 2^32 across all
///    distinct kernels seen during a process's lifetime; on a collision the
///    cache would silently serve the WRONG `CudaModule` for a colliding
///    spec — a silent-wrong-result failure mode. We instead hash with two
///    independent `DefaultHasher` instances domain-separated by a leading
///    byte and concatenate the 64-bit results into a `(u64, u64)`. The
///    birthday bound is now ~1 in 2^64 — unreachable for any realistic
///    workload.
///
/// 2. **No per-lookup allocation.** The previous implementation called
///    `format!("{:?}", spec)` on every cache lookup, allocating (and
///    then dropping) the entire `Debug` string just to feed it to the
///    hasher. We instead use a tiny `fmt::Write` adapter ([`HasherWrite`])
///    that streams the `Debug` output directly into the hasher as the
///    formatter emits it — zero heap allocation, identical hash input.
///
/// `DefaultHasher` is internally SipHash-1-3 with a fixed zero key, which
/// is *not* cryptographic but is more than adequate here: we are defending
/// against accidental collisions in our own deterministic IR, not against
/// an adversarial preimage attack. The two-hash domain-separation byte
/// (`0x01` vs `0x02`) makes the two streams independent enough that a
/// 128-bit collision requires a simultaneous collision in both halves.
///
/// # Correctness invariant (finding V-15)
///
/// This key derives entirely from `format!("{:?}", spec)` (see
/// [`ModuleCacheKey::new`]). Its correctness therefore rests on a single
/// invariant:
///
/// > **distinct specs => distinct `Debug` output.**
///
/// The default, `#[derive(Debug)]`-generated formatting on the `KernelSpec`
/// IR satisfies this because the derive emits every field and enum
/// discriminant. **Do not** add a hand-written `Debug` impl to `KernelSpec`
/// or any type it transitively contains that elides, abbreviates, or
/// otherwise collapses a discriminating field (e.g. printing only a summary,
/// hiding a "default" variant, or rounding a numeric literal). Two specs that
/// differ only in an elided field would then format identically, hash to the
/// same key, and the cache would silently serve the WRONG compiled
/// `CudaModule` for one of them — a silent-wrong-result failure mode that no
/// test of this module would catch. If a custom `Debug` is ever required for
/// readability, route this cache key through a dedicated, exhaustive
/// fingerprint instead of reusing `Debug`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ModuleCacheKey {
    /// Upper 64 bits of the 128-bit content hash (domain byte `0x01`).
    pub(crate) spec_hash_hi: u64,
    /// Lower 64 bits of the 128-bit content hash (domain byte `0x02`).
    pub(crate) spec_hash_lo: u64,
    /// PTX entry-point name (`KERNEL_ENTRY` vs `PREDICATE_ENTRY`).
    entry: &'static str,
}

/// `fmt::Write` → `Hasher` adapter. Lets us run `write!(adapter, "{:?}",
/// spec)` and have the formatter's emitted bytes go directly into the
/// underlying hasher without ever materialising a `String`. Saves an
/// allocation per cache lookup on the hot path.
struct HasherWrite<'a, H: Hasher>(&'a mut H);

impl<H: Hasher> std::fmt::Write for HasherWrite<'_, H> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.write(s.as_bytes());
        Ok(())
    }
}

impl ModuleCacheKey {
    /// Compute the cache key for `(spec, entry)`.
    ///
    /// Streams `format!("{:?}", spec)` into two domain-separated
    /// `DefaultHasher` instances and packs the resulting 128 bits into the
    /// key. See the type-level docstring for the rationale.
    pub(crate) fn new(spec: &KernelSpec, entry: &'static str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::fmt::Write as _;

        // Domain separation: write a distinguishing byte first so the two
        // hashers consume different prefixes and produce independent
        // streams over the same spec text. The actual byte values are
        // arbitrary; only the fact that they differ matters.
        let mut hi = DefaultHasher::new();
        hi.write_u8(0x01);
        // `Debug` formatting is infallible for the IR types, and
        // `HasherWrite::write_str` itself never returns an error — both
        // arms below are unreachable. Use `let _ =` rather than `unwrap`
        // so a hypothetical future failure mode degrades to a benign
        // cache miss rather than a panic in `Engine::sql`.
        let _ = write!(HasherWrite(&mut hi), "{:?}", spec);

        let mut lo = DefaultHasher::new();
        lo.write_u8(0x02);
        let _ = write!(HasherWrite(&mut lo), "{:?}", spec);

        Self {
            spec_hash_hi: hi.finish(),
            spec_hash_lo: lo.finish(),
            entry,
        }
    }
}

/// Per-table host-side revision tracker for the incremental GpuTable cache
/// (batch 5).
///
/// `table_revision` bumps on every host-side mutation that touches the
/// table — `register_table` (start at 1), `replace_table` (bump),
/// `register_batch` (bump). `column_revisions` bumps for every column
/// whose host data changed at that mutation; `column_n_rows` records the
/// total host rows that column has at the current revision (used by the
/// prefix-preserving extension path in `ensure_gpu_table`).
///
/// Mirrors the planner-cache batch 3 mechanism in spirit but stays
/// engine-local — the planner cache's invalidation is keyed off
/// `KernelSpec` content, not host data revisions.
#[derive(Debug, Default)]
pub(crate) struct HostTableRevision {
    /// Bumped on every host-side mutation. The GpuTable's
    /// `last_uploaded_revision` is compared against this on cache lookup.
    pub(crate) table_revision: u64,
    /// Per-column revision counter. Bumped for every column whose host
    /// data changed at the latest mutation. For `register_batch`
    /// (append), every column's host data changes (more rows) so every
    /// column's revision bumps.
    pub(crate) column_revisions: HashMap<String, u64>,
    /// Total host-row count per column at the current revision.
    /// `register_batch` records this so `ensure_gpu_table` can size the
    /// new GpuVec correctly and identify the previously-uploaded prefix.
    pub(crate) column_n_rows: HashMap<String, usize>,
    /// Total host-row count for the table.
    pub(crate) n_rows: usize,
}

/// Owned snapshot of a [`HostTableRevision`] taken under the `&self`
/// borrow before mutating `gpu_tables`. We can't keep a `&HostTableRevision`
/// across the `gpu_tables.borrow_mut()` because both live on `&self` and
/// the borrow-checker won't let us hold a reference into one engine field
/// while mutably reborrowing through a `RefCell` on another. Cloning the
/// few values we actually need is cheaper than refactoring the borrow
/// graph.
#[derive(Debug)]
pub(crate) struct ClonedHostRevision {
    pub(crate) table_revision: u64,
    pub(crate) column_revisions: HashMap<String, u64>,
}

/// Extension trait helper — clones a [`HostTableRevision`] reference (if
/// any) into the standalone owned form used by the incremental rebuild
/// path.
pub(crate) trait HostRevisionSnapshot {
    fn cloned_revision_owned(self) -> Option<ClonedHostRevision>;
}

impl HostRevisionSnapshot for Option<&HostTableRevision> {
    fn cloned_revision_owned(self) -> Option<ClonedHostRevision> {
        self.map(|h| ClonedHostRevision {
            table_revision: h.table_revision,
            column_revisions: h.column_revisions.clone(),
        })
    }
}
