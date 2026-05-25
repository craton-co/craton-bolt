// SPDX-License-Identifier: Apache-2.0

//! Launch-parameter auto-tuner for the per-block shared-memory GROUP BY
//! kernel (Tier 1 of the GROUP BY performance plan).
//!
//! The Tier 1 kernel builds a small per-block accumulator table in shared
//! memory, then merges into the global table at block end. That cuts global
//! atomic traffic by `~grid_blocks` x compared to the GlobalAtomic baseline,
//! but only when the launch shape is sane:
//!
//!   * Block too small => more blocks => more final-merge atomics.
//!   * Block too large => more shared-mem pressure, fewer resident blocks per SM.
//!   * Grid too small  => SMs sit idle.
//!   * Grid too large  => per-block setup (shared-mem zeroing + final merge)
//!                        dominates the grid-stride loop body.
//!
//! This module is intentionally pure logic: it knows nothing about CUDA
//! drivers or PTX. It produces a `(grid_x, block_x, shared_bytes)` tuple
//! that the dispatcher hands to `cuLaunchKernel`.

use thiserror::Error;

/// Threads per block for the shared-mem GROUP BY kernel.
///
/// 256 = 8 warps. Empirically a good sweet spot for the shared-mem accumulate
/// pattern: enough warps to hide shared-mem-atomic latency, few enough that
/// the per-block final-merge stays cheap. Multiple of the warp size (32).
pub const BLOCK_THREADS: u32 = 256;

/// Target average number of rows processed by each thread via the grid-stride
/// loop. Picked so the loop body (input load + shared-mem atomic) runs long
/// enough to amortise the per-block fixed cost (shared-mem zero-out + final
/// merge into the global table).
pub const ROWS_PER_THREAD_TARGET: u32 = 256;

/// Conservative sm_70 (Volta) per-block shared-mem floor. Every NVIDIA GPU
/// from Volta onward supports at least this much without needing the
/// `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
/// dance. 48 KiB.
const DEFAULT_MAX_SHARED_PER_BLOCK: u32 = 49_152;

/// Grid-dim.x ceiling. CUDA itself allows ~2^31-1 in grid_dim.x on modern
/// devices, but capping at 65535 keeps us inside the conservative compute
/// capability 2.x limit and dodges a handful of historical driver quirks.
/// For this kernel that's more than enough — at 65535 blocks * 256 threads
/// = ~16.7M threads, each thread already processes ~60 rows at 1 B input.
const MAX_GRID_BLOCKS: u32 = 65_535;

/// Hardware-aware launch parameters for the per-block shared-mem
/// GROUP BY kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShmemLaunchParams {
    /// Threads per block (`block_dim.x`). Always a multiple of 32 (warp size).
    pub block_threads: u32,
    /// Number of blocks in the grid (`grid_dim.x`). Sized so each thread
    /// processes ~`rows_per_thread()` rows via a grid-stride loop.
    pub grid_blocks: u32,
    /// Shared-memory bytes per block. Covers BOTH the f64 accumulator and
    /// the u8 "set" flag, with 8-byte alignment for the accumulator.
    pub shared_bytes: u32,
}

impl ShmemLaunchParams {
    /// Average rows processed per thread in the grid-stride loop.
    pub fn rows_per_thread(&self, n_rows: u32) -> u32 {
        let total_threads = self.block_threads.saturating_mul(self.grid_blocks).max(1);
        n_rows.div_ceil(total_threads)
    }
}

/// Tuner inputs.
#[derive(Debug, Clone, Copy)]
pub struct TuneInputs {
    pub n_rows: u32,
    pub n_groups: u32,
    /// Bytes per group slot (e.g. 8 for f64 accumulator + 1 for the set
    /// flag; the tuner adds alignment padding internally).
    pub bytes_per_acc_slot: u32,
    /// Optional: max shared-mem-per-block reported by the device. Pass `None`
    /// to use a portable sm_70 floor of 48 KB.
    pub max_shared_per_block: Option<u32>,
}

/// Pick launch params for the shared-mem groupby kernel.
///
/// Policy (v0):
/// * `block_threads = 256` (8 warps; balances occupancy vs final-merge cost)
/// * `shared_bytes  = round_up_to_8(n_groups * bytes_per_acc_slot) + n_groups`
///   (accumulator table + per-slot set flag, accumulator is 8-byte aligned)
/// * `grid_blocks   = clamp(n_rows / (block_threads * ROWS_PER_THREAD_TARGET),
///                          min = 1, max = 65535)`
///
/// `ROWS_PER_THREAD_TARGET = 256` keeps the grid-stride loop body running
/// long enough to amortise the per-block shared-mem zero-out and final-merge.
///
/// # Errors
///
/// Returns [`TuneError::SharedMemTooLarge`] if the requested shared-mem
/// exceeds the device's per-block limit. The caller is expected to fall back
/// to the GlobalAtomic strategy in that case.
pub fn tune(inputs: TuneInputs) -> Result<ShmemLaunchParams, TuneError> {
    let block_threads = BLOCK_THREADS;

    // ---- Shared-mem layout ------------------------------------------------
    //
    // Layout (per block):
    //
    //   [ f64 accumulator table : n_groups * bytes_per_acc_slot bytes ]
    //   [ padding to multiple of 8                                    ]
    //   [ u8 set-flag table     : n_groups bytes                      ]
    //
    // The accumulator starts at offset 0 of dynamic shared mem, which CUDA
    // already 16-byte-aligns, so f64 alignment is satisfied. The set-flag
    // table comes after, and we explicitly round the accumulator region up
    // to a multiple of 8 so the flag table doesn't disturb f64 alignment if
    // a future revision flips the layout.
    let acc_bytes_raw = (inputs.n_groups as u64) * (inputs.bytes_per_acc_slot as u64);
    let acc_bytes_aligned = round_up_to_multiple_of_8(acc_bytes_raw);
    let flag_bytes = inputs.n_groups as u64;
    let shared_total = acc_bytes_aligned.saturating_add(flag_bytes);

    let limit = inputs
        .max_shared_per_block
        .unwrap_or(DEFAULT_MAX_SHARED_PER_BLOCK);

    if shared_total > limit as u64 {
        // Clamp the reported `requested` to u32::MAX so the error stays
        // representable; the magnitude is the diagnostic, not the exact bytes.
        let requested = shared_total.min(u32::MAX as u64) as u32;
        return Err(TuneError::SharedMemTooLarge { requested, limit });
    }
    // Safe: shared_total <= limit <= u32::MAX.
    let shared_bytes = shared_total as u32;

    // ---- Grid size --------------------------------------------------------
    //
    // We want each thread to process ~ROWS_PER_THREAD_TARGET rows on average,
    // so:
    //   grid_blocks ~= n_rows / (block_threads * ROWS_PER_THREAD_TARGET)
    //
    // Floor + clamp to [1, MAX_GRID_BLOCKS]. Floor (not ceil) is intentional:
    // if the division leaves remainder, the grid-stride loop happily picks up
    // the tail — over-shooting block count just dilutes occupancy.
    let work_per_block = (block_threads as u64) * (ROWS_PER_THREAD_TARGET as u64);
    let raw = (inputs.n_rows as u64) / work_per_block;
    let grid_blocks = raw.clamp(1, MAX_GRID_BLOCKS as u64) as u32;

    Ok(ShmemLaunchParams {
        block_threads,
        grid_blocks,
        shared_bytes,
    })
}

/// Round `x` up to the next multiple of 8. Explicit, not clever.
#[inline]
fn round_up_to_multiple_of_8(x: u64) -> u64 {
    let rem = x % 8;
    if rem == 0 {
        x
    } else {
        x + (8 - rem)
    }
}

#[derive(Debug, Error)]
pub enum TuneError {
    #[error("shared-mem requirement {requested} bytes exceeds device limit {limit}")]
    SharedMemTooLarge { requested: u32, limit: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f64 accumulator (8 bytes) + 1-byte set flag => 9 bytes/slot from the
    /// caller's point of view. The tuner adds alignment padding internally.
    const FACTS_BYTES_PER_SLOT: u32 = 8;

    #[test]
    fn picks_sensible_params_for_low_card() {
        let p = tune(TuneInputs {
            n_rows: 10_000_000,
            n_groups: 100,
            bytes_per_acc_slot: FACTS_BYTES_PER_SLOT,
            max_shared_per_block: None,
        })
        .expect("low-card tune should succeed");

        assert_eq!(p.block_threads, 256, "block size policy is 256 threads");
        assert!(p.grid_blocks > 0, "grid must be non-empty");
        assert!(
            p.shared_bytes < 49_152,
            "low-card shared bytes should be well under 48K, got {}",
            p.shared_bytes
        );
        // 100 groups * 8 bytes = 800, rounded up to 800 (already /8) + 100
        // bytes of flags = 900.
        assert_eq!(p.shared_bytes, 900, "expected exact layout 800 + 100 = 900");
    }

    #[test]
    fn respects_shared_mem_ceiling() {
        // 8 KB groups * 8 bytes = 64 KB accumulator alone, already over 48 KB.
        let err = tune(TuneInputs {
            n_rows: 1_000_000,
            n_groups: 8 * 1024,
            bytes_per_acc_slot: FACTS_BYTES_PER_SLOT,
            max_shared_per_block: None,
        })
        .expect_err("should reject over-budget shared mem");

        match err {
            TuneError::SharedMemTooLarge { requested, limit } => {
                assert_eq!(limit, 49_152, "default limit should be 48 KiB");
                assert!(
                    requested > limit,
                    "requested ({}) must exceed limit ({})",
                    requested,
                    limit
                );
            }
        }
    }

    #[test]
    fn rows_per_thread_amortizes() {
        let p = tune(TuneInputs {
            n_rows: 10_000_000,
            n_groups: 100,
            bytes_per_acc_slot: FACTS_BYTES_PER_SLOT,
            max_shared_per_block: None,
        })
        .unwrap();

        let rpt = p.rows_per_thread(10_000_000);
        assert!(
            rpt >= 32,
            "stride loop should process >= 32 rows/thread (got {})",
            rpt
        );
    }

    #[test]
    fn grid_capped_at_65535() {
        let p = tune(TuneInputs {
            n_rows: 1_000_000_000,
            n_groups: 64,
            bytes_per_acc_slot: FACTS_BYTES_PER_SLOT,
            max_shared_per_block: None,
        })
        .unwrap();

        assert!(
            p.grid_blocks <= 65_535,
            "grid must be capped at 65535, got {}",
            p.grid_blocks
        );
    }

    #[test]
    fn tiny_input_still_launches() {
        let p = tune(TuneInputs {
            n_rows: 1024,
            n_groups: 4,
            bytes_per_acc_slot: FACTS_BYTES_PER_SLOT,
            max_shared_per_block: None,
        })
        .unwrap();

        assert!(
            p.grid_blocks >= 1,
            "even tiny inputs must launch at least one block (got {})",
            p.grid_blocks
        );
        assert_eq!(p.block_threads, 256);
    }

    #[test]
    fn shared_mem_layout_rounds_to_8() {
        // n_groups = 3 * bytes_per_acc_slot = 8  => acc = 24 (already /8),
        // flags = 3, total = 27. No padding between acc and flags needed
        // because 24 is already aligned, so total is 27.
        let p = tune(TuneInputs {
            n_rows: 1000,
            n_groups: 3,
            bytes_per_acc_slot: 8,
            max_shared_per_block: None,
        })
        .unwrap();
        assert_eq!(p.shared_bytes, 24 + 3);

        // n_groups = 5, bytes_per_acc_slot = 1 => acc raw = 5, rounded to 8,
        // + flags 5 = 13.
        let p2 = tune(TuneInputs {
            n_rows: 1000,
            n_groups: 5,
            bytes_per_acc_slot: 1,
            max_shared_per_block: None,
        })
        .unwrap();
        assert_eq!(p2.shared_bytes, 8 + 5);
    }

    #[test]
    fn round_up_helper() {
        assert_eq!(round_up_to_multiple_of_8(0), 0);
        assert_eq!(round_up_to_multiple_of_8(1), 8);
        assert_eq!(round_up_to_multiple_of_8(7), 8);
        assert_eq!(round_up_to_multiple_of_8(8), 8);
        assert_eq!(round_up_to_multiple_of_8(9), 16);
        assert_eq!(round_up_to_multiple_of_8(800), 800);
    }
}
