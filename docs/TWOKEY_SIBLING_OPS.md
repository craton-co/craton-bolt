# Two-key Tier-2.1 sibling ops — implementation status

The Tier-2.1 path (`partition + scatter + GPU per-partition reduce`)
is now parameterised over key dtype, value dtype, and aggregate count.
The cell coverage matrix for two-key (i64-packed) workloads:

| Op family       | Kernel                                            | Executor                                          | Status     |
| --------------- | ------------------------------------------------- | ------------------------------------------------- | ---------- |
| Single SUM      | `partition_reduce_kernel_i64`                     | `groupby_tier2_twokey_exec`                       | ✅ shipped (q3 = 219 ms) |
| Multi-SUM (1–4) | `partition_reduce_kernel_multi_i64`               | `groupby_tier2_twokey_multi_exec`                 | ✅ shipped |
| COUNT(*)        | **`partition_reduce_kernel_count_i64`** ← NEW     | TODO                                              | Kernel ✅, executor TODO |
| AVG (1–4)       | reuse `partition_reduce_kernel_multi_i64` + count | TODO                                              | Kernels ✅, executor TODO |
| MIN/MAX int     | TODO: clone `partition_reduce_kernel_minmax`      | TODO                                              | Both TODO  |
| MIN/MAX float   | TODO: clone `partition_reduce_kernel_minmax_float`| TODO                                              | Both TODO  |

## What this session delivered for two-key sibling ops

- `src/jit/partition_reduce_kernel_count_i64.rs` — new PTX kernel for
  i64-key COUNT(*) reduce. 4 unit tests pass. Mirror of the i32 sibling
  with 8-byte slot stride for keys and `cvt.u32.u64` slot mapping. The
  i64-key two-key COUNT(*) and the COUNT denominator for two-key AVG
  both consume this.

## Pattern for remaining sibling executors

Each missing executor is a near-mechanical adaptation of one i32-key
executor in the existing tree. The recipe:

1. **Copy** the matching i32-key executor file:
   - `groupby_tier2_count_exec.rs` → `groupby_tier2_twokey_count_exec.rs`
   - `groupby_tier2_avg_exec.rs` → `groupby_tier2_twokey_avg_exec.rs`
   - `groupby_tier2_minmax_exec.rs` → `groupby_tier2_twokey_minmax_exec.rs`
   - `groupby_tier2_minmax_float_exec.rs` → `groupby_tier2_twokey_minmax_float_exec.rs`
2. **Substitute imports**: swap `partition_kernel` → `partition_kernel_i64`,
   `scatter_kernel` → `scatter_kernel_i64`, and the reduce-kernel module
   to the i64 sibling (e.g. `partition_reduce_kernel_count` →
   `partition_reduce_kernel_count_i64`).
3. **Adjust types**: `GpuVec<i32>` → `GpuVec<i64>` for the key column;
   `Int32Array::from(...)` → `Int64Array::from(...)` in output build;
   shift+pack/unpack on host for the GROUP BY columns mirror what's
   done in `groupby_tier2_twokey_exec.rs::execute_inner`.
4. **Eligibility check**: accept `aggregate.group_by.len() == 2` (not 1)
   with both keys Int32. Pack to i64 via `(a << 32) | (b & 0xFFFF_FFFF)`
   on the host before uploading.
5. **Register** the new module in `src/exec/mod.rs` and wire a
   `try_execute` call into `groupby.rs` immediately after the
   matching single-key sibling.

Each follow-up is ~250–400 LOC. Estimated effort: **1 engineer-day** to
complete all four (count / avg / int-minmax / float-minmax) once a
benchmark workload exercising any of them lands.

## Why we stopped here

No h2o.ai bench query exercises two-key + AVG / COUNT / MIN / MAX. The
kernels are now structurally ready (i64-key partition + scatter +
reduce family is complete); the executor wiring is a clone-and-adapt
mechanical follow-up that lands when there's a measurable workload to
validate it against.

Don't add the executor files speculatively — they'd be dead code with
no test coverage. Add them on demand.
