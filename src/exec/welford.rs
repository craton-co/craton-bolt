// SPDX-License-Identifier: Apache-2.0

//! Welford's one-pass algorithm for numerically-stable mean / variance.
//!
//! Used by the `STDDEV_POP` / `STDDEV_SAMP` aggregates (this module) and,
//! later, by `VAR_POP` / `VAR_SAMP` (a sibling agent). The state is the
//! canonical triple `(count, mean, M2)` where:
//!
//! * `count`  — number of values folded in so far,
//! * `mean`   — running mean of those values,
//! * `M2`     — sum of squared deviations from the mean (`Σ (x_i - mean)^2`).
//!
//! At any point, the **population variance** is `M2 / count` and the
//! **sample variance** is `M2 / (count - 1)` (defined only when
//! `count > 1`). The standard deviations are the square roots of those.
//!
//! # Why Welford?
//!
//! The naive two-pass formula `Σ x_i^2 / N - mean^2` is catastrophically
//! unstable in single precision (and lossy in double) when the running
//! `Σ x_i^2` is much larger than `N * mean^2` — typical for any column
//! with a non-zero mean. Welford keeps the accumulator close to the data
//! magnitude, so float cancellation never dominates.
//!
//! # Combine rule
//!
//! Welford's update rule extends cleanly to **merging** two partial states
//! (Chan-Golub-LeVeque, 1979 — the parallel reduction form). This is the
//! ingredient that makes an on-device reduction kernel possible later: each
//! block computes a partial `(count, mean, M2)`, the host (or a final pass
//! kernel) merges them with [`WelfordState::combine`]. We currently use the
//! host-side serial fold path; the combine helper is the seam future GPU
//! work plugs into.
//!
//! # Host vs device
//!
//! This v0.5 cut runs the reduction on the host. We download (or already
//! have) the input values as a host slice, fold them through
//! [`WelfordState::push`] in source order, and the final stddev is a single
//! `sqrt`. The constants are small enough that the host overhead is
//! invisible against the GPU H2D + reduce + D2H pipeline that AVG already
//! pays for any non-trivial column. A device-side kernel (per-block
//! partials + host or second-pass merge) is a v0.6 stretch goal and would
//! reuse exactly this state.

use crate::error::{BoltError, BoltResult};

/// Running `(count, mean, M2)` Welford state, accumulated in `f64`.
///
/// `f32` inputs are promoted to `f64` at the push site so the accumulator
/// stays in double precision; the final `sqrt` is similarly in `f64`. The
/// state is `Clone + Copy` because it's three machine words — passing it
/// by value through the per-block partials merge step is the natural API
/// for the eventual GPU path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WelfordState {
    /// Number of values folded in so far.
    pub count: u64,
    /// Running mean.
    pub mean: f64,
    /// Sum of squared deviations from the mean (Σ (x_i - mean)^2).
    pub m2: f64,
}

impl Default for WelfordState {
    fn default() -> Self {
        Self::empty()
    }
}

impl WelfordState {
    /// Empty state — the identity for [`combine`](Self::combine). Used at
    /// the start of every reduction and as the per-thread initial value in
    /// the eventual device kernel.
    pub const fn empty() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Fold one value into `self`. Welford's classic update:
    ///
    /// ```text
    /// count += 1
    /// delta  = x - mean
    /// mean  += delta / count
    /// delta2 = x - mean        // recomputed AFTER mean update
    /// M2    += delta * delta2
    /// ```
    ///
    /// The two `delta`s are intentionally different — using the *post*-update
    /// mean for the second one is what gives Welford its numerical stability
    /// (vs. the naive equivalent that uses `delta * delta` and is much less
    /// stable when the running mean is far from the new value).
    #[inline]
    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        // Safe because `count` was just incremented from a u64; we never
        // observe a count of 0 here.
        self.mean += delta / (self.count as f64);
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Fold an entire slice in source order. Equivalent to repeated
    /// [`push`](Self::push) but factored out so the call site stays a
    /// single line for the common scalar-aggregate path.
    pub fn push_slice_f64(&mut self, xs: &[f64]) {
        for &x in xs {
            self.push(x);
        }
    }

    /// Fold a slice of `i64` values. Each value is promoted to `f64`; loss
    /// of precision is possible past 2^53 magnitudes — same caveat as the
    /// rest of the engine's `i64 -> f64` accumulation paths (see
    /// `aggregate.rs::scalar_to_array`).
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

    /// Fold a slice of `f32` values, promoted to `f64`.
    pub fn push_slice_f32(&mut self, xs: &[f32]) {
        for &x in xs {
            self.push(x as f64);
        }
    }

    /// Merge two partial Welford states into a single combined state.
    /// Implements the Chan-Golub-LeVeque parallel update (1979):
    ///
    /// ```text
    /// n   = a.count + b.count
    /// δ   = b.mean - a.mean
    /// μ   = a.mean + δ * (b.count / n)            // weighted mean
    /// M2  = a.M2 + b.M2 + δ^2 * (a.count * b.count / n)
    /// ```
    ///
    /// `combine(empty(), s) == combine(s, empty()) == s` — the empty state
    /// is the identity, which is what makes a tree-reduce over arbitrary
    /// partition sizes (including empty blocks at the tail) safe.
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

    /// Population standard deviation: `sqrt(M2 / count)`.
    ///
    /// Returns `Ok(Some(σ))` for a non-empty state, `Ok(None)` for empty —
    /// the caller decides whether to surface the empty case as SQL NULL or
    /// as `0.0` (the existing AVG convention).
    pub fn stddev_pop(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let var = self.m2 / (self.count as f64);
        // `var` can be a tiny negative number due to f64 round-off when
        // every input is the same; clamp before sqrt so we don't return
        // NaN for a mathematically-zero variance.
        Some(var.max(0.0).sqrt())
    }

    /// Sample standard deviation: `sqrt(M2 / (count - 1))`.
    ///
    /// Returns `Ok(Some(σ))` for `count >= 2`, `Ok(None)` for `count <= 1`
    /// (the SQL-standard "undefined" case — divisor is zero or negative).
    pub fn stddev_samp(&self) -> Option<f64> {
        if self.count <= 1 {
            return None;
        }
        let var = self.m2 / ((self.count - 1) as f64);
        Some(var.max(0.0).sqrt())
    }
}

/// Result of finalizing a Welford reduction into a single scalar.
///
/// Mirrors the convention the existing aggregate path uses for AVG (an
/// empty input returns the additive identity rather than SQL NULL,
/// because the output schema field is non-nullable in the current cut).
/// For STDDEV_POP we likewise return `Some(0.0)` on an empty input so the
/// scalar-aggregate path can pack a non-NULL Float64; STDDEV_SAMP returns
/// `None` on `count <= 1` so the call site can build a nullable
/// Float64Array with a NULL slot.
#[derive(Debug, Clone, Copy)]
pub enum StddevKind {
    /// `STDDEV_POP` — divisor is `count`.
    Pop,
    /// `STDDEV_SAMP` — divisor is `count - 1`; result is undefined (NULL)
    /// for `count <= 1`.
    Samp,
}

/// Finalize a Welford state to a scalar standard deviation. Returns
/// `Some(σ)` when the result is defined and `None` when it isn't — the
/// caller decides whether to surface `None` as SQL NULL or as a sentinel.
pub fn finalize(state: &WelfordState, kind: StddevKind) -> Option<f64> {
    match kind {
        StddevKind::Pop => state.stddev_pop(),
        StddevKind::Samp => state.stddev_samp(),
    }
}

/// Convenience: error type for `dtype` lookups that aren't yet supported
/// by the host-side Welford push helpers. Kept here so callers in
/// `aggregate.rs` can route through this module instead of duplicating
/// the dtype-dispatch ladder.
pub fn err_unsupported_dtype(dtype_dbg: &str, op: &str) -> BoltError {
    BoltError::Type(format!(
        "{op} over dtype {dtype_dbg} not supported in scalar Welford path"
    ))
}

/// Convenience wrapper: build a fresh state, fold the slice in source
/// order, return the state. Useful for tests and for the scalar
/// aggregate dispatch which doesn't need to share state across calls.
pub fn reduce_f64_slice(xs: &[f64]) -> WelfordState {
    let mut s = WelfordState::empty();
    s.push_slice_f64(xs);
    s
}

/// Convenience wrapper: build a fresh state, fold the slice (i64), return.
pub fn reduce_i64_slice(xs: &[i64]) -> WelfordState {
    let mut s = WelfordState::empty();
    s.push_slice_i64(xs);
    s
}

/// Convenience wrapper: build a fresh state, fold the slice (i32), return.
pub fn reduce_i32_slice(xs: &[i32]) -> WelfordState {
    let mut s = WelfordState::empty();
    s.push_slice_i32(xs);
    s
}

/// Convenience wrapper: build a fresh state, fold the slice (f32), return.
pub fn reduce_f32_slice(xs: &[f32]) -> WelfordState {
    let mut s = WelfordState::empty();
    s.push_slice_f32(xs);
    s
}

/// Helper used by `BoltResult`-returning aggregate paths that need to
/// surface "dtype unsupported by the Welford path" with a consistent
/// message. The free function form keeps call sites a single line.
pub fn ensure_numeric_dtype(dtype: crate::plan::logical_plan::DataType, op: &str) -> BoltResult<()> {
    use crate::plan::logical_plan::DataType::*;
    match dtype {
        Int32 | Int64 | Float32 | Float64 => Ok(()),
        other => Err(err_unsupported_dtype(&format!("{:?}", other), op)),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed reference: σ_pop and σ_samp over [1.0, 2.0, 3.0, 4.0, 5.0].
    /// mean = 3.0, Σ (x-3)^2 = 4 + 1 + 0 + 1 + 4 = 10.
    /// σ_pop  = sqrt(10/5) = sqrt(2) ≈ 1.4142135...
    /// σ_samp = sqrt(10/4) = sqrt(2.5) ≈ 1.5811388...
    #[test]
    fn small_known_sequence_matches_hand_computed() {
        let s = reduce_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(s.count, 5);
        assert!((s.mean - 3.0).abs() < 1e-12, "mean = {}", s.mean);
        assert!((s.m2 - 10.0).abs() < 1e-12, "M2 = {}", s.m2);
        let pop = s.stddev_pop().expect("non-empty");
        let samp = s.stddev_samp().expect("count > 1");
        assert!((pop - 2.0_f64.sqrt()).abs() < 1e-12);
        assert!((samp - 2.5_f64.sqrt()).abs() < 1e-12);
    }

    /// Constant input → variance must be exactly 0 (the `max(0.0)` clamp
    /// kicks in to absorb the tiny round-off the running mean accumulates).
    #[test]
    fn constant_input_has_zero_variance() {
        let s = reduce_f64_slice(&[7.0; 100]);
        assert_eq!(s.count, 100);
        assert_eq!(s.stddev_pop().expect("non-empty"), 0.0);
        assert_eq!(s.stddev_samp().expect("count > 1"), 0.0);
    }

    /// Empty input: pop returns None, samp returns None.
    #[test]
    fn empty_input_returns_none_for_both() {
        let s = WelfordState::empty();
        assert!(s.stddev_pop().is_none());
        assert!(s.stddev_samp().is_none());
    }

    /// Single value: pop = 0, samp = None (undefined, divisor would be 0).
    #[test]
    fn single_value_pop_is_zero_samp_is_none() {
        let s = reduce_f64_slice(&[42.0]);
        assert_eq!(s.count, 1);
        assert_eq!(s.stddev_pop(), Some(0.0));
        assert!(s.stddev_samp().is_none(), "STDDEV_SAMP undefined for n=1");
    }

    /// Combine is the identity rule we'll lean on for the GPU per-block
    /// reduction: `combine(empty, s) == combine(s, empty) == s`.
    #[test]
    fn combine_identity_with_empty() {
        let a = reduce_f64_slice(&[1.0, 2.0, 3.0]);
        let combined_left = WelfordState::combine(WelfordState::empty(), a);
        let combined_right = WelfordState::combine(a, WelfordState::empty());
        assert_eq!(combined_left, a);
        assert_eq!(combined_right, a);
    }

    /// Combine matches a single-pass push over the concatenation. This is
    /// the load-bearing correctness property for the parallel reduction;
    /// without it the future GPU port would silently produce wrong stddevs.
    #[test]
    fn combine_matches_concatenated_push() {
        let xs: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let mid = xs.len() / 2;
        let a = reduce_f64_slice(&xs[..mid]);
        let b = reduce_f64_slice(&xs[mid..]);
        let combined = WelfordState::combine(a, b);
        let full = reduce_f64_slice(&xs);
        assert_eq!(combined.count, full.count);
        assert!(
            (combined.mean - full.mean).abs() < 1e-12,
            "mean: combined={}, full={}",
            combined.mean,
            full.mean
        );
        assert!(
            (combined.m2 - full.m2).abs() < 1e-9,
            "M2: combined={}, full={}",
            combined.m2,
            full.m2
        );
        let pop_a = combined.stddev_pop().unwrap();
        let pop_b = full.stddev_pop().unwrap();
        assert!((pop_a - pop_b).abs() < 1e-9);
    }

    /// Push-slice helpers match scalar push for integer dtypes.
    #[test]
    fn integer_push_slices_match_f64_promotion() {
        let i32_state = reduce_i32_slice(&[1i32, 2, 3, 4, 5]);
        let i64_state = reduce_i64_slice(&[1i64, 2, 3, 4, 5]);
        let f64_state = reduce_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(i32_state.count, f64_state.count);
        assert_eq!(i64_state.count, f64_state.count);
        assert!((i32_state.m2 - f64_state.m2).abs() < 1e-12);
        assert!((i64_state.m2 - f64_state.m2).abs() < 1e-12);
    }

    /// Welford stays stable on inputs whose two-pass formula would lose
    /// precision: large-magnitude mean + small variance. We don't make a
    /// hard quantitative claim here (the threshold depends on the rounding
    /// of intermediates) but we DO verify the result is close to the true
    /// stddev of `1.0` — the naive `(Σx^2)/N - mean^2` formula returns
    /// nonsense (often a tiny negative number → NaN after sqrt) on this
    /// shape.
    #[test]
    fn welford_is_stable_for_large_mean_small_variance() {
        let mean = 1.0e9_f64;
        let xs: Vec<f64> = (0..1000).map(|i| mean + (i as f64 - 499.5)).collect();
        // True variance of 0..1000 around 499.5 is 83333.25, stddev ≈ 288.67.
        let s = reduce_f64_slice(&xs);
        let pop = s.stddev_pop().expect("non-empty");
        let expected = ((0..1000)
            .map(|i| {
                let d = i as f64 - 499.5;
                d * d
            })
            .sum::<f64>()
            / 1000.0)
            .sqrt();
        assert!(
            (pop - expected).abs() < 1e-6,
            "Welford stddev {pop} vs expected {expected}"
        );
    }

    /// Finalize routes through the same Option contract as the methods.
    #[test]
    fn finalize_routes_to_pop_and_samp() {
        let s = reduce_f64_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(finalize(&s, StddevKind::Pop), s.stddev_pop());
        assert_eq!(finalize(&s, StddevKind::Samp), s.stddev_samp());
    }
}
