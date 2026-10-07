// SPDX-License-Identifier: Apache-2.0
//! Storage layer: tables whose data lives outside the engine's in-memory
//! store, their statistics, and scan pruning.
//!
//! * [`StorageTable`] — the scan contract (chunked data + zone maps).
//!   [`MemoryTable`] is the in-memory implementation; [`ParquetTable`]
//!   (feature `storage`) reads Parquet files.
//! * [`TableStatistics`] / [`ColumnStatistics`] — row counts, bounds, null
//!   counts and distinct estimates, used for pruning and by the optimizer.
//! * [`ScanRequest`] — the columns and prunable predicates a query needs,
//!   derived from its optimized plan.
//!
//! The engine scans a storage table per query
//! ([`Engine::register_storage_table`](crate::Engine::register_storage_table)):
//! only the chunks whose statistics may match the query's filters, and only
//! the columns it reads, are loaded.
//!
//! # Stability
//!
//! Experimental until 1.0.

pub mod scan;
pub mod stats;
pub mod table;

#[cfg(feature = "storage")]
pub mod catalog;
#[cfg(feature = "storage")]
pub mod parquet;

pub use scan::{PruneOp, PrunePredicate, ScanRequest};
pub use stats::{ColumnStatistics, TableStatistics};
pub use table::{BatchIter, MemoryTable, ScanSelection, StorageTable};

pub use crate::exec::engine_storage::ScanReport;

#[cfg(feature = "storage")]
pub use self::catalog::{Catalog, CompactReport, RefreshPolicy, VacuumReport};
#[cfg(feature = "storage")]
pub use self::parquet::{write_parquet, ParquetTable, ParquetWriteOptions};
