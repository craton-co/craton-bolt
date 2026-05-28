// SPDX-License-Identifier: Apache-2.0

//! Welford's one-pass algorithm for numerically-stable mean / variance.
//!
//! Used by the `VAR_POP` / `VAR_SAMP` and `STDDEV_POP` / `STDDEV_SAMP`
//! scalar aggregates. The state is the canonical triple
//! `(count, mean, M2)`. At any point, population variance is
//! `M2 / count`, sample variance is `M2 / (count - 1)`. Standard
//! deviations are the square roots of those.

use crate::error::{BoltError, BoltResult};

/// Running `(count, mean, M2)` Welford state, accumulated in `f64`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WelfordState {
    /// Number of values folded in so far.
    pub count: u64,
    /// Running mean.
    pub mean: f64,
    /// Sum of squared deviations from the mean.
    pub m2: f64,
}

impl Default for WelfordState {
    fn default() -> Self {
        Self::empty()
    }
}

impl WelfordState {
    /// Empty state — identity for [`combine`](Self::combine).
    pub const fn empty() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Fresh empty state (alias for `empty`).
    pub fn new() -> Self {
        Self::empty()
    }

    /// Fold one value into `self` (Welford's classic update).
    #[inline]
    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / (self.count as f64);
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Fold an entire slice in source order.
    pub fn push_slice_f64(&mut self, xs: &[f64]) {
        for &x in xs {
            self.push(x);
        }
    }

    /// Fold a slice of `i64` values (promoted to `f64`).
    pub fn push_slice_i64(&mut self, xs: &[i64]) {
        for &x in xs {
            self.push(x as f64);
        }
    }

    /// Fold a slice of `i32` values.
    pub fn push_slice_i32(&mut self, xs: &[i32]) {
        for &x in xs {
            self.push(x as f64);
        }
    }

    /// Fold a slice of `f32` values (promoted to `f64`).
    pub fn push_slice_f32(&mut self, xs: &[f32]) {
        for &x in xs {
            self.push(x as f64);
        }
    }

    /// Merge two partial Welford states (Chan-Golub-LeVeque parallel update).
    pub fn combine(a: WelfordState, b: WelfordState) -> WelfordState {
        if a.count == 0 {
            return b;
        }
        if b.count == 0 {
            return a;
        }
        let n_a = a.count as f64;
        let n_b = b.count as f64;
        let n = n_a + n_b;
        let delta = b.mean - a.mean;
        let mean = a.mean + delta * (n_b / n);
        let m2 = a.m2 + b.m2 + delta * delta * (n_a * n_b / n);
        WelfordState {
            count: a.count + b.count,
            mean,
            m2,
        }
    }

    /// Population variance: `M2 / count`, or `None` for an empty state.
    pub fn var_pop(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.m2 / (self.count as f64))
        }
    }

    /// Sample variance: `M2 / (count - 1)`, or `None` when `count <= 1`.
    pub fn var_samp(&self) -> Option<f64> {
        if self.count <= 1 {
            None
        } else {
            Some(self.m2 / ((self.count - 1) as f64))
        }
    }

    /// Population standard deviation: `sqrt(var_pop)`. `None` for empty.
    pub fn stddev_pop(&self) -> Option<f64> {
        let v = self.var_pop()?;
        Some(v.max(0.0).sqrt())
    }

    /// Sample standard deviation: `sqrt(var_samp)`. `None` when `count <= 1`.
    pub fn stddev_samp(&self) -> Option<f64> {
        let v = self.var_samp()?;
        Some(v.max(0.0).sqrt())
    }
}

/// Compute population variance of `xs` in one host-side Welford pass.
pub fn var_pop_f64(xs: &[f64]) -> Option<f64> {
    let mut s = WelfordState::new();
    s.push_slice_f64(xs);
    s.var_pop()
}

/// Compute sample variance of `xs` in one host-side Welford pass.
pub fn var_samp_f64(xs: &[f64]) -> Option<f64> {
    let mut s = WelfordState::new();
    s.push_slice_f64(xs);
    s.var_samp()
}

/// Whether to finalize a Welford state into a population or sample stddev.
#[derive(Debug, Clone, Copy)]
pub enum StddevKind {
    /// `STDDEV_POP` — divisor is `count`.
    Pop,
    /// `STDDEV_SAMP` — divisor is `count - 1`; result is NULL when `count <= 1`.
    Samp,
}

/// Finalize a Welford state to a scalar standard deviation.
pub fn finalize(state: &WelfordState, kind: StddevKind) -> Option<f64> {
    match kind {
        StddevKind::Pop => state.stddev_pop(),
        StddevKind::Samp => state.stddev_samp(),
    }
}

/// Error helper for dtype dispatch failures in Welford paths.
pub fn err_unsupported_dtype(dtype_dbg: &str, op: &str) -> BoltError {
    BoltError::Type(format!(
        "{op} over dtype {dtype_dbg} not supported in scalar Welford path"
    ))
}

/// Ensure `dtype` is one of the four numeric primitives the Welford push
/// helpers accept.
pub fn ensure_numeric_dtype(dtype: crate::plan::logical_plan::DataType, op: &str) -> BoltResult<()> {
    use crate::plan::logical_plan::DataType::*;
    match dtype {
        Int32 | Int64 | Float32 | Float64 => Ok(()),
        other => Err(err_unsupported_dtype(&format!("{:?}", other), op)),
    }
}

/// Build a state and fold an `f64` slice. Convenience for callers.
pub fn reduce_f64_slice(xs: &[f64]) -> WelfordState {
    let mut s = WelfordState::empty();
    s.push_slice_f64(xs);
    s
}

/// Build a state and fold an `i64` slice.
pub fn reduce_i64_slice(xs: &[i64]) -> WelfordState {
    let mut s = WelfordState::empty();
    s.push_slice_i64(xs);
    s
}

/// Build a state and fold an `i32` slice.
pub fn reduce_i32_slice(xs: &[i32]) -> WelfordState {
    let mut s = WelfordState::empty();
    s.push_slice_i32(xs);
    s
}

/// Build a state and fold an `f32` slice.
pub fn reduce_f32_slice(xs: &[f32]) -> WelfordState {
    let mut s = WelfordState::empty();
    s.push_slice_f32(xs);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_returns_none() {
        let s = WelfordState::empty();
        assert_eq!(s.var_pop(), None);
        assert_eq!(s.var_samp(), None);
        assert_eq!(s.stddev_pop(), None);
        assert_eq!(s.stddev_samp(), None);
    }

    #[test]
    fn single_observation() {
        let mut s = WelfordState::empty();
        s.push(5.0);
        assert_eq!(s.var_pop(), Some(0.0));
        assert_eq!(s.var_samp(), None);
        assert_eq!(s.stddev_pop(), Some(0.0));
        assert_eq!(s.stddev_samp(), None);
    }

    #[test]
    fn three_observations_match_closed_form() {
        let xs = [1.0, 2.0, 3.0];
        let vp = var_pop_f64(&xs).unwrap();
        let vs = var_samp_f64(&xs).unwrap();
        assert!((vp - (2.0 / 3.0)).abs() < 1e-12);
        assert!((vs - 1.0).abs() < 1e-12);
    }

    #[test]
    fn constant_input_has_zero_variance() {
        let xs = [7.5_f64; 50];
        assert_eq!(var_pop_f64(&xs), Some(0.0));
        assert_eq!(var_samp_f64(&xs), Some(0.0));
    }

    #[test]
    fn combine_identity_with_empty() {
        let s = reduce_f64_slice(&[1.0, 2.0, 3.0]);
        let empty = WelfordState::empty();
        assert_eq!(WelfordState::combine(empty, s).var_pop(), s.var_pop());
        assert_eq!(WelfordState::combine(s, empty).var_pop(), s.var_pop());
    }

    #[test]
    fn combine_matches_concatenated_push() {
        let a = reduce_f64_slice(&[1.0, 2.0, 3.0]);
        let b = reduce_f64_slice(&[4.0, 5.0, 6.0]);
        let merged = WelfordState::combine(a, b);
        let full = reduce_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!((merged.var_pop().unwrap() - full.var_pop().unwrap()).abs() < 1e-12);
    }
}
