// SPDX-License-Identifier: Apache-2.0

//! Logical plan IR, DataFrame builder, SQL frontend, and physical plan.

pub mod logical_plan;
pub mod dataframe;
pub mod sql_frontend;
pub mod physical_plan;
pub mod string_literal_rewrite;

pub use logical_plan::{
    AggregateExpr, BinaryOp, DataType, Expr, Field, Literal, LogicalPlan, ScalarFnKind, Schema,
    UnaryOp, col, lit,
};
pub use dataframe::{DataFrame, GroupedDataFrame, avg, count, max, min, sum};
pub use sql_frontend::{parse as parse_sql, MemTableProvider, TableProvider};
pub use physical_plan::{
    lower as lower_physical, ColumnIO, KernelSpec, Op, PhysicalPlan, Reg, Value,
};
