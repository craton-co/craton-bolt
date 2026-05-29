// SPDX-License-Identifier: Apache-2.0

//! Logical plan IR, DataFrame builder, SQL frontend, and physical plan.

pub mod logical_plan;
pub mod dataframe;
pub mod sql_frontend;
pub mod physical_plan;
pub mod explain;
pub mod string_literal_rewrite;
pub mod suggest;
pub mod rewrite;
pub mod optimizer;
pub mod statistics;
pub mod subquery;

pub use logical_plan::{
    AggregateExpr, BinaryOp, DataType, Expr, Field, Literal, LogicalPlan, ScalarFnKind, Schema,
    TimeUnit, UnaryOp, col, lit,
};
pub use dataframe::{
    DataFrame, GroupedDataFrame, avg, count, max, min, stddev_pop, stddev_samp, sum, var_pop,
    var_samp,
};
pub use sql_frontend::{parse as parse_sql, MemTableProvider, TableProvider};
pub use physical_plan::{
    lower as lower_physical, ColumnIO, KernelSpec, Op, PhysicalPlan, Reg, StringLengthOutput,
    Value,
};
pub use rewrite::PlanRewrite;
pub use optimizer::{default_passes, default_passes_with_estimator, RowEstimator, StatsEstimator};
pub use explain::{format_logical, format_physical};
