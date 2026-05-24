// SPDX-License-Identifier: Apache-2.0

//! Lazy DataFrame builder over `LogicalPlan`.

use crate::error::JavelinResult;
use crate::plan::logical_plan::{AggregateExpr, Expr, LogicalPlan, Schema};

/// Lazy DataFrame — wraps a `LogicalPlan` and offers a builder API.
#[derive(Debug, Clone)]
pub struct DataFrame {
    plan: LogicalPlan,
}

impl DataFrame {
    /// Start a query against a registered table.
    pub fn scan(table: impl Into<String>, schema: Schema) -> Self {
        Self {
            plan: LogicalPlan::Scan {
                table: table.into(),
                projection: None,
                schema,
            },
        }
    }

    /// Wrap an already-built `LogicalPlan` as a `DataFrame`.
    pub fn from_plan(plan: LogicalPlan) -> Self {
        Self { plan }
    }

    /// SELECT — replace the projection list.
    pub fn select<I: IntoIterator<Item = Expr>>(self, exprs: I) -> Self {
        Self {
            plan: LogicalPlan::Project {
                input: Box::new(self.plan),
                exprs: exprs.into_iter().collect(),
            },
        }
    }

    /// WHERE — narrow rows by a boolean predicate.
    pub fn filter(self, predicate: Expr) -> Self {
        Self {
            plan: LogicalPlan::Filter {
                input: Box::new(self.plan),
                predicate,
            },
        }
    }

    /// GROUP BY — returns a `GroupedDataFrame` awaiting `.agg(...)`.
    pub fn group_by<I: IntoIterator<Item = Expr>>(self, keys: I) -> GroupedDataFrame {
        GroupedDataFrame {
            plan: self.plan,
            keys: keys.into_iter().collect(),
        }
    }

    /// Inspect the current plan.
    pub fn logical_plan(&self) -> &LogicalPlan {
        &self.plan
    }

    /// Type-check the plan and return its output schema.
    pub fn schema(&self) -> JavelinResult<Schema> {
        self.plan.schema()
    }

    /// Hand the plan off to the engine.
    // TODO(1.0): introduce a real `collect()` that materializes the plan to a
    // `RecordBatch` via `Engine`. The current `collect` alias below is a
    // doc-hidden tombstone kept only so older internal call sites compile; it
    // should be removed once that materializing API lands.
    pub fn into_plan(self) -> LogicalPlan {
        self.plan
    }

    /// Deprecated alias for [`DataFrame::into_plan`]. Hidden from rustdoc
    /// because the name `collect` is reserved for a future materializing API
    /// (Polars-style) in 1.0; today this is a no-op rename.
    #[doc(hidden)]
    pub fn collect(self) -> LogicalPlan {
        self.into_plan()
    }
}

/// Intermediate produced by `DataFrame::group_by`.
#[derive(Debug, Clone)]
pub struct GroupedDataFrame {
    plan: LogicalPlan,
    keys: Vec<Expr>,
}

impl GroupedDataFrame {
    /// Attach aggregate expressions and return a `DataFrame`.
    pub fn agg<I: IntoIterator<Item = AggregateExpr>>(self, aggs: I) -> DataFrame {
        DataFrame {
            plan: LogicalPlan::Aggregate {
                input: Box::new(self.plan),
                group_by: self.keys,
                aggregates: aggs.into_iter().collect(),
            },
        }
    }
}

/// `COUNT(expr)` aggregate.
pub fn count(e: Expr) -> AggregateExpr {
    AggregateExpr::Count(e)
}

/// `SUM(expr)` aggregate.
pub fn sum(e: Expr) -> AggregateExpr {
    AggregateExpr::Sum(e)
}

/// `MIN(expr)` aggregate.
pub fn min(e: Expr) -> AggregateExpr {
    AggregateExpr::Min(e)
}

/// `MAX(expr)` aggregate.
pub fn max(e: Expr) -> AggregateExpr {
    AggregateExpr::Max(e)
}

/// `AVG(expr)` aggregate.
pub fn avg(e: Expr) -> AggregateExpr {
    AggregateExpr::Avg(e)
}
