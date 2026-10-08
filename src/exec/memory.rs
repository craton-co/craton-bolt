// SPDX-License-Identifier: Apache-2.0
//! Query-level memory governance.
//!
//! The device memory pool recycles allocations; it does not decide *what*
//! may occupy memory. The [`MemoryGovernor`] does, at operator level:
//!
//! * **Budgets.** A device budget ([`EngineBuilder::memory_budget`]) and a
//!   host budget ([`EngineBuilder::host_memory_budget`]).
//! * **Reservations.** Long-lived host data the engine can rebuild — an
//!   installed storage-table scan — holds a [`MemoryReservation`]. When a
//!   new reservation would exceed the budget, the engine first evicts its
//!   own rebuildable data (least recently used scans; resident GPU table
//!   copies on the device side), then asks registered
//!   [`MemoryConsumer`]s to spill.
//! * **Out-of-core execution.** A statement whose input exceeds the budget
//!   is executed in hash partitions spilled to disk (see `engine_ooc`); the
//!   governor records the spill volume ([`MemoryStats`]).
//!
//! [`EngineBuilder::memory_budget`]: crate::EngineBuilder::memory_budget
//! [`EngineBuilder::host_memory_budget`]: crate::EngineBuilder::host_memory_budget

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use crate::error::{BoltError, BoltResult};

/// Which memory a reservation counts against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    /// Host RAM.
    Host,
    /// GPU device memory.
    Device,
}

impl MemoryKind {
    fn idx(self) -> usize {
        match self {
            MemoryKind::Host => 0,
            MemoryKind::Device => 1,
        }
    }
}

/// Something holding memory it can give back on request — a cache, a
/// buffer that can be written to disk.
///
/// Register with [`MemoryGovernor::register_consumer`]; the governor keeps a
/// weak reference, so dropping the consumer unregisters it.
pub trait MemoryConsumer: Send + Sync {
    /// Name for diagnostics.
    fn name(&self) -> &str;
    /// Bytes currently held.
    fn bytes_reserved(&self) -> usize;
    /// Release at least `target_bytes` if possible; return the bytes
    /// actually released.
    fn spill(&self, target_bytes: usize) -> BoltResult<usize>;
}

/// Snapshot of the governor's accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct MemoryStats {
    /// Host budget, if any.
    pub host_budget: Option<usize>,
    /// Device budget, if any.
    pub device_budget: Option<usize>,
    /// Host bytes currently reserved.
    pub host_reserved: usize,
    /// Highest host reservation total seen.
    pub host_peak: usize,
    /// Spill files written by out-of-core execution.
    pub spill_files: u64,
    /// Bytes written to spill files.
    pub spilled_bytes: u64,
    /// Rebuildable entries evicted to make room (scans, GPU table copies).
    pub evictions: u64,
    /// Statements executed out of core.
    pub out_of_core_statements: u64,
}

/// See the [module docs](self).
pub struct MemoryGovernor {
    budgets: [Option<usize>; 2],
    reserved: [AtomicUsize; 2],
    peak: [AtomicUsize; 2],
    consumers: Mutex<Vec<Weak<dyn MemoryConsumer>>>,
    spill_files: AtomicU64,
    spilled_bytes: AtomicU64,
    evictions: AtomicU64,
    ooc_statements: AtomicU64,
}

impl std::fmt::Debug for MemoryGovernor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryGovernor")
            .field("stats", &self.stats())
            .finish()
    }
}

/// Bytes counted against a [`MemoryGovernor`] budget until dropped.
#[derive(Debug)]
pub struct MemoryReservation {
    governor: Arc<MemoryGovernor>,
    kind: MemoryKind,
    bytes: usize,
}

impl MemoryReservation {
    /// Reserved bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.governor.reserved[self.kind.idx()].fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

impl MemoryGovernor {
    /// Governor with the given budgets (`None` = unlimited).
    pub fn new(host_budget: Option<usize>, device_budget: Option<usize>) -> Arc<Self> {
        Arc::new(MemoryGovernor {
            budgets: [host_budget, device_budget],
            reserved: [AtomicUsize::new(0), AtomicUsize::new(0)],
            peak: [AtomicUsize::new(0), AtomicUsize::new(0)],
            consumers: Mutex::new(Vec::new()),
            spill_files: AtomicU64::new(0),
            spilled_bytes: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            ooc_statements: AtomicU64::new(0),
        })
    }

    /// Budget of `kind`.
    pub fn budget(&self, kind: MemoryKind) -> Option<usize> {
        self.budgets[kind.idx()]
    }

    /// Bytes currently reserved against `kind`.
    pub fn reserved(&self, kind: MemoryKind) -> usize {
        self.reserved[kind.idx()].load(Ordering::Acquire)
    }

    /// Bytes still available under the `kind` budget (`usize::MAX` when
    /// unlimited).
    pub fn available(&self, kind: MemoryKind) -> usize {
        match self.budget(kind) {
            Some(b) => b.saturating_sub(self.reserved(kind)),
            None => usize::MAX,
        }
    }

    /// Register a consumer to be asked to spill under pressure.
    pub fn register_consumer(&self, consumer: &Arc<dyn MemoryConsumer>) {
        let mut c = self.consumers.lock().unwrap_or_else(|p| p.into_inner());
        c.retain(|w| w.strong_count() > 0);
        c.push(Arc::downgrade(consumer));
    }

    /// Reserve `bytes` of `kind`, asking registered consumers (largest
    /// first) to spill if the budget would be exceeded.
    ///
    /// # Errors
    /// [`BoltError::Memory`] when the budget cannot accommodate the request
    /// even after spilling.
    pub fn try_reserve(
        self: &Arc<Self>,
        kind: MemoryKind,
        bytes: usize,
    ) -> BoltResult<MemoryReservation> {
        if let Some(budget) = self.budget(kind) {
            let over = (self.reserved(kind) + bytes).saturating_sub(budget);
            if over > 0 {
                self.spill_consumers(over)?;
            }
            let now = self.reserved(kind);
            if now + bytes > budget {
                return Err(BoltError::Memory(format!(
                    "{kind:?} memory budget of {budget} bytes cannot fit {bytes} more bytes \
                     ({now} reserved)"
                )));
            }
        }
        Ok(self.reserve_unchecked(kind, bytes))
    }

    /// Reserve without a budget check (for data that must be held anyway;
    /// it still counts against later requests).
    pub fn reserve_unchecked(
        self: &Arc<Self>,
        kind: MemoryKind,
        bytes: usize,
    ) -> MemoryReservation {
        let now = self.reserved[kind.idx()].fetch_add(bytes, Ordering::AcqRel) + bytes;
        self.peak[kind.idx()].fetch_max(now, Ordering::AcqRel);
        MemoryReservation {
            governor: Arc::clone(self),
            kind,
            bytes,
        }
    }

    fn spill_consumers(&self, mut need: usize) -> BoltResult<()> {
        let mut live: Vec<Arc<dyn MemoryConsumer>> = {
            let c = self.consumers.lock().unwrap_or_else(|p| p.into_inner());
            c.iter().filter_map(Weak::upgrade).collect()
        };
        live.sort_by_key(|c| std::cmp::Reverse(c.bytes_reserved()));
        for c in live {
            if need == 0 {
                break;
            }
            let freed = c.spill(need)?;
            log::debug!(
                "memory governor: consumer '{}' released {freed} bytes",
                c.name()
            );
            need = need.saturating_sub(freed);
        }
        Ok(())
    }

    pub(crate) fn record_spill(&self, files: u64, bytes: u64) {
        self.spill_files.fetch_add(files, Ordering::Relaxed);
        self.spilled_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub(crate) fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_out_of_core(&self) {
        self.ooc_statements.fetch_add(1, Ordering::Relaxed);
    }

    /// Current accounting.
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            host_budget: self.budgets[0],
            device_budget: self.budgets[1],
            host_reserved: self.reserved(MemoryKind::Host),
            host_peak: self.peak[0].load(Ordering::Acquire),
            spill_files: self.spill_files.load(Ordering::Relaxed),
            spilled_bytes: self.spilled_bytes.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            out_of_core_statements: self.ooc_statements.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Cache {
        held: AtomicUsize,
        reservation: Mutex<Option<MemoryReservation>>,
    }

    impl MemoryConsumer for Cache {
        fn name(&self) -> &str {
            "cache"
        }
        fn bytes_reserved(&self) -> usize {
            self.held.load(Ordering::Acquire)
        }
        fn spill(&self, _target: usize) -> BoltResult<usize> {
            let freed = self.held.swap(0, Ordering::AcqRel);
            self.reservation.lock().unwrap().take();
            Ok(freed)
        }
    }

    #[test]
    fn reservations_count_and_release() {
        let g = MemoryGovernor::new(Some(100), None);
        let a = g.try_reserve(MemoryKind::Host, 60).unwrap();
        assert_eq!(g.reserved(MemoryKind::Host), 60);
        assert!(g.try_reserve(MemoryKind::Host, 50).is_err());
        drop(a);
        assert_eq!(g.reserved(MemoryKind::Host), 0);
        let _b = g.try_reserve(MemoryKind::Host, 100).unwrap();
        assert_eq!(g.stats().host_peak, 100);
        // Unlimited kinds never refuse.
        let _c = g.try_reserve(MemoryKind::Device, usize::MAX / 2).unwrap();
    }

    #[test]
    fn pressure_spills_registered_consumers() {
        let g = MemoryGovernor::new(Some(100), None);
        let cache = Arc::new(Cache {
            held: AtomicUsize::new(80),
            reservation: Mutex::new(None),
        });
        *cache.reservation.lock().unwrap() = Some(g.reserve_unchecked(MemoryKind::Host, 80));
        let dyn_cache: Arc<dyn MemoryConsumer> = cache.clone();
        g.register_consumer(&dyn_cache);
        let r = g.try_reserve(MemoryKind::Host, 50).unwrap();
        assert_eq!(cache.bytes_reserved(), 0);
        assert_eq!(g.reserved(MemoryKind::Host), 50);
        drop(r);
        drop(dyn_cache);
        drop(cache);
        // A dropped consumer is skipped, not called.
        let _x = g.try_reserve(MemoryKind::Host, 100).unwrap();
    }
}
