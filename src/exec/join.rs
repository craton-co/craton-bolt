// SPDX-License-Identifier: Apache-2.0

//! JOIN executor — type scaffold and 0.2 placeholder.
//!
//! INNER JOIN parses and lowers in 0.1.x but execution returns a clear
//! error. The 0.2 release will land a hash-join implementation:
//! - Build a host-side `HashMap<JoinKey, Vec<row_idx>>` over the smaller
//!   side.
//! - Probe the larger side row-by-row, emit matching (left, right)
//!   row pairs.
//! - Construct a RecordBatch from the gathered rows using
//!   `arrow::compute::take`.
//!
//! See ROADMAP.md → 0.2 milestones.

use crate::error::{JavelinError, JavelinResult};
use crate::exec::{Engine, QueryHandle};
use crate::plan::logical_plan::{Expr, JoinType};
use crate::plan::physical_plan::PhysicalPlan;

/// Execute an INNER JOIN. 0.1.x placeholder: returns
/// `JavelinError::Other("JOIN not yet implemented")` while keeping the
/// dispatch surface stable for 0.2.
///
/// Signature mirrors the destructured `PhysicalPlan::Join` dispatch in
/// `Engine::execute`, which borrows the plan: `left` / `right` are sub-plans
/// the 0.2 implementation will recursively `engine.execute(...)` to obtain
/// build- and probe-side `QueryHandle`s before running the hash-join body.
pub fn execute_join(
    _left: &PhysicalPlan,
    _right: &PhysicalPlan,
    join_type: &JoinType,
    _on: &[(Expr, Expr)],
    _engine: &Engine,
) -> JavelinResult<QueryHandle> {
    Err(JavelinError::Other(format!(
        "JOIN ({:?}) not yet implemented; the parser, AST and physical-plan \
         lowering land in 0.1.x as scaffold for the 0.2 hash-join executor. \
         See ROADMAP.md.",
        join_type,
    )))
}

#[cfg(test)]
mod tests {
    // Constructing a `PhysicalPlan::Join` (and the `Engine` the dispatch
    // signature requires) is non-trivial from a unit test — both normally
    // flow out of `lower(...)` and a real `Engine::new(...)` against
    // registered tables. The hash-join body lands in 0.2 alongside an
    // end-to-end test that drives `Engine::query("SELECT ... JOIN ...")`
    // and asserts result rows, which exercises this dispatch path far
    // more meaningfully than a synthetic call here would.
    #[test]
    #[ignore = "0.2 follow-up: real test arrives with the hash-join body"]
    fn join_errors_clearly_in_0_1_x() {}
}
