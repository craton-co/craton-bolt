// SPDX-License-Identifier: Apache-2.0

//! Reserved sibling module for the SQL-frontend-facing query methods of
//! [`crate::exec::engine::Engine`] (`sql` / `explain_sql` / `run_logical_plan`
//! / `run_subplan` / table registration).
//!
//! Part of the pure-reorg split of the formerly-monolithic `engine.rs`. The
//! streaming, string-projection, and host-aggregate clusters were moved out
//! into their own sibling modules (`engine_streaming_impl`,
//! `engine_string_impl`, `engine_agg_impl`).
//!
//! The query / dispatch entry points were intentionally **left in
//! `engine.rs`** for this pass: `Engine::sql` is the top-level dispatch the
//! split is explicitly meant to keep alongside the struct/builder, and
//! `run_logical_plan` / `run_subplan` are tightly coupled to the recursive-CTE,
//! LATERAL, and correlated-WHERE executors (which also remain in `engine.rs`),
//! so relocating them would mean either widening a large fan-out of private
//! methods or moving those executors too — out of scope for a behaviour-
//! preserving move. This file is reserved for that follow-up.
//!
//! TODO(engine-split): if/when the recursive / LATERAL / correlated-WHERE
//! executors are also extracted, move the query-frontend methods here as an
//! `impl Engine` block.
