// SPDX-License-Identifier: Apache-2.0

//! Reserved sibling module for the join execution / dispatch methods of
//! [`crate::exec::engine::Engine`].
//!
//! Part of the pure-reorg split of the formerly-monolithic `engine.rs`. The
//! streaming, string-projection, and host-aggregate clusters have been moved
//! out into their own sibling modules (`engine_streaming_impl`,
//! `engine_string_impl`, `engine_agg_impl`). The join-dispatch cluster has
//! **not** yet been extracted: in the current tree join execution is dispatched
//! inline from `Engine::execute_leaf_whole` (via `crate::exec::join` /
//! `crate::exec::gpu_join`) rather than living in a self-contained set of
//! `impl Engine` methods, so there is no coherent verbatim block to move
//! without restructuring call sites (which would not be a behaviour-preserving
//! pure move). This file is intentionally empty pending that follow-up.
//!
//! TODO(engine-split): once the join path is refactored into dedicated
//! `Engine::execute_join_*` methods, move them here as an `impl Engine` block
//! and widen the minimal field/method visibility, exactly as the other three
//! clusters were split.
