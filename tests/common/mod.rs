// SPDX-License-Identifier: Apache-2.0

//! Shared test helpers. Integration tests under `tests/<name>.rs` include
//! this via `mod common;`. Not part of the published crate.
//!
//! Rust's integration test runner treats every `tests/*.rs` as its own
//! binary, so factoring shared helpers requires the `tests/common/mod.rs`
//! sub-module pattern (the `mod.rs` suffix is recognised; `tests/common.rs`
//! would itself be compiled as another test binary).
//!
//! # Standard `#[ignore]` categories (review L5)
//!
//! - `gpu:tier1` — Tier-1 GROUP BY / aggregate
//! - `gpu:tier2` — Tier-2 hash-partitioned GROUP BY
//! - `gpu:join` — GPU hash join
//! - `gpu:sort` — GPU sort
//! - `gpu:mempool` — Memory pool / VRAM tests
//! - `gpu:string` — Utf8 / dictionary tests
//! - `gpu:e2e` — Generic e2e SQL needing GPU
//! - `gpu:proptest-semantic` — Property-test semantic diff vs DuckDB
//! - `bootstrap` — Snapshot bootstrap (rare)
//!
//! Run a subset via: `cargo test -- --ignored --filter <bucket>`.

/// Default relative-tolerance constant for numerical equality across
/// the test + bench suite. Floating-point arithmetic is non-associative
/// and chunking / SIMD / fast-math can introduce per-batch drift of
/// ~1e-12; 1e-9 is conservative for f64 sums up to ~1e6.
// Each tests/*.rs is its own integration-test crate that only imports the
// helpers it needs, so per-binary dead_code warnings here are spurious.
#[allow(dead_code)]
pub const REL_TOL: f64 = 1e-9;

/// Deterministic xorshift64* PRNG used by the integration test fixtures.
///
/// All call-sites must use the same seed-mixing convention for
/// reproducibility across machines and rust versions. `new` does the
/// golden-ratio mix + non-zero guard so callers can pass the raw user seed
/// directly (e.g. a per-test literal).
///
/// Output quality is far more than adequate for fixture generation; do not
/// use it for anything security-sensitive.
// Each tests/*.rs is its own integration-test crate that only imports the
// helpers it needs, so per-binary dead_code warnings here are spurious.
#[allow(dead_code)]
pub struct Xorshift64Star {
    state: u64,
}

#[allow(dead_code)]
impl Xorshift64Star {
    /// Construct from an arbitrary `u64` seed. The seed is mixed with the
    /// golden-ratio constant (so adjacent integer seeds produce
    /// well-separated streams) and clamped away from the `0` fixed point of
    /// xorshift.
    pub fn new(seed: u64) -> Self {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        if state == 0 {
            state = 0xDEAD_BEEF_CAFE_BABE;
        }
        Self { state }
    }

    /// One step of xorshift64*: a 64-bit xorshift followed by a multiply with
    /// a high-entropy odd constant.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform `f64` in `[0.0, 1.0)`. Takes the top 53 bits of a fresh draw
    /// — the standard "fill the f64 mantissa" trick. Matches what the
    /// fixture builders relied on before consolidation.
    #[inline]
    pub fn next_unit_f64(&mut self) -> f64 {
        let r = self.next_u64();
        (r >> 11) as f64 * (1.0_f64 / ((1_u64 << 53) as f64))
    }

    /// Uniform `f64` in `[-1.0, 1.0)`. Equivalent to
    /// `next_unit_f64() * 2.0 - 1.0`.
    #[inline]
    pub fn next_signed_unit_f64(&mut self) -> f64 {
        self.next_unit_f64() * 2.0 - 1.0
    }
}

/// Deterministic Fisher-Yates shuffle so tests are reproducible without
/// pulling a `rand` dev-dep. The LCG constants are Knuth's.
///
/// This is a distinct PRNG from [`Xorshift64Star`] — kept separate because
/// existing sort fixtures hash-coded the LCG output across many tests and
/// changing the PRNG would shift every expected value.
// Each tests/*.rs is its own integration-test crate that only imports the
// helpers it needs, so per-binary dead_code warnings here are spurious.
#[allow(dead_code)]
pub fn shuffle_deterministic<T: Copy>(xs: &mut [T], seed: u64) {
    let mut s = seed;
    for i in (1..xs.len()).rev() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (s as usize) % (i + 1);
        xs.swap(i, j);
    }
}
