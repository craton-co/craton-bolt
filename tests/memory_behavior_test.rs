// SPDX-License-Identifier: Apache-2.0

//! `PhysicalPlan::memory_behavior` / `memory_profile`: how each operator's
//! state grows with its input (host-only: plans are lowered, not run).

use craton_bolt::plan::{lower_physical, parse_sql, DataType, Field, MemTableProvider, Schema};
use craton_bolt::MemoryBehavior;

fn profile(sql: &str) -> MemoryBehavior {
    let provider = MemTableProvider::new()
        .with_table(
            "t",
            Schema::new(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("v", DataType::Float64, false),
            ]),
        )
        .with_table(
            "u",
            Schema::new(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("w", DataType::Float64, false),
            ]),
        );
    let plan = parse_sql(sql, &provider).unwrap_or_else(|e| panic!("{sql}: {e}"));
    lower_physical(&plan)
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .memory_profile()
}

#[test]
fn operators_are_classified() {
    assert_eq!(
        profile("SELECT k, v * 2 AS d FROM t WHERE v > 1"),
        MemoryBehavior::Streaming
    );
    assert!(matches!(
        profile("SELECT SUM(v) FROM t WHERE k > 3"),
        MemoryBehavior::BoundedState { .. }
    ));
    assert_eq!(
        profile("SELECT k, SUM(v) FROM t GROUP BY k"),
        MemoryBehavior::Partitionable
    );
    assert_eq!(
        profile("SELECT t.k, u.w FROM t JOIN u ON t.k = u.k"),
        MemoryBehavior::Partitionable
    );
    assert_eq!(
        profile("SELECT k FROM t ORDER BY v"),
        MemoryBehavior::Spillable
    );
    assert_eq!(
        profile("SELECT k, SUM(v) OVER (PARTITION BY k) AS s FROM t"),
        MemoryBehavior::FullyMaterializing
    );
    // The most demanding node decides.
    assert_eq!(
        MemoryBehavior::Streaming.max(MemoryBehavior::Spillable),
        MemoryBehavior::Spillable
    );
}
