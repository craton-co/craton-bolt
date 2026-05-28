// SPDX-License-Identifier: Apache-2.0

//! Welford's online algorithm for variance, used by the `VAR_POP` /
//! `VAR_SAMP` scalar aggregates.
//!
//! For v0.5 the GPU codegen path does not yet emit Welford-on-device
//! kernels; the scalar (no GROUP BY) aggregate path downloads the column
//! to the host and reduces it here in `f64`. This module is the single
//! source of truth for the numerical recipe so a future device-side
//! implementation can compare bit-for-bit against the host fallback.
//!
//! The algorithm carries three running quantities: `count`, `mean`, and
//! `M2` (the sum of squared deviations from the running mean). For each
//! new observation `x`:
//!
//! ```text
//!   count <- count + 1
//!   delta  = x - mean
//!   mean  <- mean + delta / count
//!   delta2 = x - mean
//!   M2    <- M2 + delta * delta2
//! ```
//!
//! After streaming all observations:
//!   * `var_pop  = M2 / count`           (population variance)
//!   * `var_samp = M2 / (count - 1)`     (sample variance; NULL when count <= 1)
//!
//! The host reduction works on `f64` regardless of input dtype — narrow
//! integers and Float32 are widened during the source-column upcast.

/// Running Welford state. `count` is non-negative; `mean` and `m2` are
/// only meaningful when `count > 0`.
///
/// Construct with [`WelfordState::new`] and feed observations through
/// [`WelfordState::push`]. Finalise with [`WelfordState::var_pop`] /
/// [`WelfordState::var_samp`].
#[derive(Debug, Clone, Copy, Default)]
pub struct WelfordState {
    /// Number of observations accumulated so far.
    pub count: u64,
    /// Running mean.
    pub mean: f64,
    /// Running sum of squared deviations from the mean.
    pub m2: f64,
}

impl WelfordState {
    /// Fresh empty state. Equivalent to `Default::default()` but spelled
    /// out so the call site reads as deliberate initialisation.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Fold a single observation `x` into the running state. Numerically
    /// stable — uses the mean-update form (Welford 1962) rather than the
    /// naive `sum_of_squares - sum*sum/n` formulation, which catastrophically
    /// cancels on inputs whose mean is much larger than their spread.
    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        // count >= 1 here, so the divide is well-defined.
        self.mean += delta / (self.count as f64);
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Population variance: `M2 / count`, or `None` when no observations
    /// have been folded in. The SQL standard says `VAR_POP` of an empty /
    /// all-NULL group is NULL — surface that via `Option<f64>` so the
    /// caller can pack a nullable Arrow cell directly.
    pub fn var_pop(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.m2 / (self.count as f64))
        }
    }

    /// Sample variance: `M2 / (count - 1)`, or `None` when fewer than two
    /// observations have been folded in. SQL standard returns NULL when
    /// `count <= 1`; this mirrors that contract.
    pub fn var_samp(&self) -> Option<f64> {
        if self.count <= 1 {
            None
        } else {
            Some(self.m2 / ((self.count - 1) as f64))
        }
    }
}

/// Compute the population variance of `xs` in one host-side Welford pass.
/// Returns `None` for an empty slice (SQL NULL semantics).
pub fn var_pop_f64(xs: &[f64]) -> Option<f64> {
    let mut s = WelfordState::new();
    for &x in xs {
        s.push(x);
    }
    s.var_pop()
}

/// Compute the sample variance of `xs` in one host-side Welford pass.
/// Returns `None` when `xs.len() <= 1` (SQL NULL semantics).
pub fn var_samp_f64(xs: &[f64]) -> Option<f64> {
    let mut s = WelfordState::new();
    for &x in xs {
        s.push(x);
    }
    s.var_samp()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty input → both variants return `None`.
    #[test]
    fn empty_returns_none() {
        assert_eq!(var_pop_f64(&[]), None);
        assert_eq!(var_samp_f64(&[]), None);
    }

    /// One observation: VAR_POP is defined (== 0), VAR_SAMP is NULL.
    #[test]
    fn single_observation() {
        assert_eq!(var_pop_f64(&[5.0]), Some(0.0));
        assert_eq!(var_samp_f64(&[5.0]), None);
    }

    /// Three observations: spot-check against the closed-form result.
    /// `[1, 2, 3]` -> mean 2, deviations `[-1, 0, 1]`, M2 = 2.
    /// VAR_POP = 2/3, VAR_SAMP = 2/2 = 1.
    #[test]
    fn three_observations_match_closed_form() {
        let xs = [1.0, 2.0, 3.0];
        let vp = var_pop_f64(&xs).unwrap();
        let vs = var_samp_f64(&xs).unwrap();
        assert!((vp - (2.0 / 3.0)).abs() < 1e-12, "VAR_POP = {vp}");
        assert!((vs - 1.0).abs() < 1e-12, "VAR_SAMP = {vs}");
    }

    /// Order-independence: streaming in reverse must yield the same result
    /// up to floating-point round-off.
    #[test]
    fn order_independence_within_tolerance() {
        let xs: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let mut rev: Vec<f64> = xs.clone();
        rev.reverse();
        let a = var_pop_f64(&xs).unwrap();
        let b = var_pop_f64(&rev).unwrap();
        assert!((a - b).abs() < 1e-9, "order: {a} vs {b}");
    }

    /// Constant input: variance is exactly 0 regardless of length.
    #[test]
    fn constant_input_has_zero_variance() {
        let xs = [7.5_f64; 50];
        assert_eq!(var_pop_f64(&xs), Some(0.0));
        let vs = var_samp_f64(&xs).unwrap();
        assert_eq!(vs, 0.0);
    }

    /// Welford's stability win: a stream with a huge mean and small spread
    /// must still produce a small variance. The naive
    /// `E[X^2] - E[X]^2` formulation would catastrophically cancel here.
    #[test]
    fn welford_is_stable_for_high_mean_small_spread() {
        let base = 1e9_f64;
        let xs: Vec<f64> = (0..1000).map(|i| base + (i as f64) * 1e-3).collect();
        // Closed-form: a sequence base, base+d, base+2d, ..., base+(n-1)d
        // has var_pop = d^2 * (n^2 - 1) / 12.
        let n = xs.len() as f64;
        let d = 1e-3_f64;
        let expected = d * d * (n * n - 1.0) / 12.0;
        let got = var_pop_f64(&xs).unwrap();
        let rel = (got - expected).abs() / expected;
        assert!(
            rel < 1e-6,
            "Welford lost precision on high-mean input: got {got}, expected {expected}, \
             rel error {rel}"
        );
    }
}
