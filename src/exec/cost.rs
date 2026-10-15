// SPDX-License-Identifier: Apache-2.0
//! Cost-based CPU / GPU tier selection.
//!
//! A GPU operator can be faster in isolation and still lose end to end: a
//! small statement over a table that is not resident on the device pays a
//! host-to-device copy, kernel launches and (on first use) JIT compilation
//! that the host would not. A [`CostModel`] estimates both sides of one
//! operator from its input statistics and the table's residency, as an
//! [`OperatorCost`]; under [`TierPolicy::CostBased`] the engine runs the
//! statement on the tier the estimate prefers.
//!
//! The host tier covers row-wise statements (filters, projections, `ORDER
//! BY` on columns, `LIMIT`) and scalar `COUNT` / `SUM` / `MIN` / `MAX` /
//! `AVG` over one table whose columns and expressions the host evaluator
//! supports (numeric, boolean and string columns; arithmetic, comparisons,
//! boolean logic, `CASE`, `IS [NOT] NULL`, numeric casts). Anything else,
//! and any statement the host evaluator declines at run time, runs on the
//! GPU as before. `EXPLAIN` reports the estimate and the chosen tier.
//!
//! [`DefaultCostModel`] is a linear model with constants for a PCIe 3 ×16
//! card; [`Engine::calibrate_cost_model`](crate::Engine::calibrate_cost_model)
//! measures them on the current device and installs the result.
//!
//! # Stability
//!
//! Experimental until 1.0.

use crate::plan::ExecutionTier;

/// When the engine considers running a statement on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum TierPolicy {
    /// Every statement the GPU path supports runs there (the default).
    #[default]
    GpuFirst,
    /// Statements the host tier supports run there when the
    /// [`CostModel`] estimates the host to be cheaper.
    CostBased,
}

/// Shape of the operator being costed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OperatorKind {
    /// Filters / projections / sort / limit over a scan: rows in, rows out.
    RowWise,
    /// Aggregates without `GROUP BY`: rows in, one row out.
    ScalarAggregate,
}

/// What a [`CostModel`] knows about one operator.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CostInputs {
    /// Operator shape.
    pub kind: OperatorKind,
    /// Input rows (from table statistics).
    pub rows: u64,
    /// Bytes of the input columns the operator reads.
    pub input_bytes: u64,
    /// Estimated bytes of the result.
    pub output_bytes: u64,
    /// Expressions evaluated per row (predicates, projections, aggregates).
    pub expressions: usize,
    /// The input table is already resident on the device.
    pub resident: bool,
    /// The statement's GPU kernels are already compiled (warm plan cache).
    pub compiled: bool,
}

/// Estimated cost of one operator on each tier.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub struct OperatorCost {
    /// Host execution time.
    pub cpu_ns: f64,
    /// Device execution time, transfers and launches included (compilation
    /// excluded).
    pub gpu_ns: f64,
    /// Bytes copied host → device.
    pub h2d_bytes: u64,
    /// Bytes copied device → host.
    pub d2h_bytes: u64,
    /// Device memory the operator needs at peak.
    pub peak_device_bytes: u64,
    /// One-off JIT compilation time if the kernels are not cached.
    pub compilation_ns: f64,
}

impl OperatorCost {
    /// The tier with the lower estimated total time (GPU on a tie).
    pub fn preferred_tier(&self) -> ExecutionTier {
        if self.cpu_ns < self.gpu_ns + self.compilation_ns {
            ExecutionTier::Host
        } else {
            ExecutionTier::Gpu
        }
    }
}

/// Estimates [`OperatorCost`]s. Install one with
/// [`EngineBuilder::cost_model`](crate::EngineBuilder::cost_model).
pub trait CostModel: Send + Sync + std::fmt::Debug {
    /// Cost of the operator described by `inputs` on each tier.
    fn estimate(&self, inputs: &CostInputs) -> OperatorCost;
}

/// Linear cost model: transfer bytes over measured bandwidths, fixed launch
/// overhead, per-row device and host throughput, and a one-off compile cost.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct DefaultCostModel {
    /// Host → device copy bandwidth.
    pub h2d_bytes_per_sec: f64,
    /// Device → host copy bandwidth.
    pub d2h_bytes_per_sec: f64,
    /// Fixed device overhead of one statement (launches, synchronisation,
    /// result packaging).
    pub launch_ns: f64,
    /// Device time per input row per expression.
    pub gpu_ns_per_row: f64,
    /// Host time per input row per expression.
    pub cpu_ns_per_row: f64,
    /// JIT compilation and module load of a statement's kernels.
    pub compile_ns: f64,
}

impl Default for DefaultCostModel {
    fn default() -> Self {
        DefaultCostModel {
            h2d_bytes_per_sec: 10e9,
            d2h_bytes_per_sec: 10e9,
            launch_ns: 60_000.0,
            gpu_ns_per_row: 0.02,
            cpu_ns_per_row: 8.0,
            compile_ns: 25e6,
        }
    }
}

impl CostModel for DefaultCostModel {
    fn estimate(&self, i: &CostInputs) -> OperatorCost {
        let rows = i.rows as f64;
        let exprs = i.expressions.max(1) as f64;
        let h2d = if i.resident { 0 } else { i.input_bytes };
        let d2h = match i.kind {
            OperatorKind::RowWise => i.output_bytes,
            OperatorKind::ScalarAggregate => 64,
        };
        let gpu_ns = self.launch_ns
            + h2d as f64 / self.h2d_bytes_per_sec * 1e9
            + d2h as f64 / self.d2h_bytes_per_sec * 1e9
            + rows * exprs * self.gpu_ns_per_row;
        OperatorCost {
            cpu_ns: rows * exprs * self.cpu_ns_per_row,
            gpu_ns,
            h2d_bytes: h2d,
            d2h_bytes: d2h,
            peak_device_bytes: i.input_bytes + i.output_bytes,
            compilation_ns: if i.compiled { 0.0 } else { self.compile_ns },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(rows: u64, resident: bool, compiled: bool) -> CostInputs {
        CostInputs {
            kind: OperatorKind::RowWise,
            rows,
            input_bytes: rows * 16,
            output_bytes: rows * 8,
            expressions: 2,
            resident,
            compiled,
        }
    }

    #[test]
    fn small_cold_inputs_prefer_the_host() {
        let m = DefaultCostModel::default();
        assert_eq!(
            m.estimate(&inputs(1_000, false, false)).preferred_tier(),
            ExecutionTier::Host
        );
        // Even warm, a thousand rows do not amortise a launch.
        assert_eq!(
            m.estimate(&inputs(1_000, true, true)).preferred_tier(),
            ExecutionTier::Host
        );
    }

    #[test]
    fn large_resident_inputs_prefer_the_gpu() {
        let m = DefaultCostModel::default();
        let c = m.estimate(&inputs(100_000_000, true, true));
        assert_eq!(c.preferred_tier(), ExecutionTier::Gpu);
        assert_eq!(c.h2d_bytes, 0);
        // Residency removes the upload from the estimate.
        let cold = m.estimate(&inputs(100_000_000, false, true));
        assert!(cold.gpu_ns > c.gpu_ns);
        assert_eq!(cold.h2d_bytes, 1_600_000_000);
    }

    #[test]
    fn compilation_counts_only_when_cold() {
        let m = DefaultCostModel::default();
        assert_eq!(m.estimate(&inputs(10, true, true)).compilation_ns, 0.0);
        assert!(m.estimate(&inputs(10, true, false)).compilation_ns > 0.0);
    }
}
