# FAQ

Frequently asked questions about Craton Bolt. For the supported SQL surface
see [`SQL_REFERENCE.md`](SQL_REFERENCE.md); for planned work see
[`../ROADMAP.md`](../ROADMAP.md).

## Q1. Why no NVRTC or runtime LLVM?

NVRTC compiles CUDA C++ to PTX; Craton Bolt emits PTX directly from Rust, so
NVRTC would buy us nothing. To go PTX → SASS we call the driver's
`cuModuleLoadDataEx`, which assembles internally. The result is that the
only runtime dependency is the CUDA driver itself — no `libnvrtc`, no
bundled `ptxas`, no LLVM. NVRTC alone would add roughly 80 MB of shared
libraries to a deployment.

## Q2. Why no macOS support?

Apple removed CUDA from macOS in 2019; there is no NVIDIA driver and no
recent CUDA toolkit for any Apple-shipped GPU. You can still type-check
the crate on macOS with `cargo check --features cuda-stub`, but you
cannot run a kernel.

## Q3. What's the status of async memcpy?

It's shipped. The FFI bindings for `cuMemcpyAsync` and pinned host
allocation landed in 0.3.0, and the safe wrappers (`memcpy_h2d_async`,
`memcpy_d2h_async`, `memset_d8_async`, `PinnedHostBuffer<T>`,
`GpuBuffer::copy_{from,to}_async`) followed. The scalar-aggregate
executor was the 0.6 pilot
(`upload_primitive_values_async`); 0.7 rolled async memcpy out to the
remaining `GROUP BY` variants (tier2 / shmem / wide / valid) and added
async D2H for `compact::download_mask` (the `WHERE` filter path). See
`docs/JIT_PIPELINE.md` for the staging history.

## Q4. Can I share a `GpuVec` between threads?

`GpuVec<T>` is `Send` but not `Sync`. You can move ownership of a vec to
another thread, but only one thread at a time can take a `GpuViewMut`.
`GpuView<'a, T>` is `Send + !Sync` for the same reason — sharing an
immutable view across threads would let a sibling thread holding the
parent vec construct a `GpuViewMut` and race a writer kernel against
your reader.

## Q5. What happens if I call `Engine::sql` on a batch with too many rows?

The `n_rows_to_u32` helper errors when `n_rows > u32::MAX`. The PTX
kernels use `.u32` for the row count and `.s32` for the thread index, so
the engine refuses to launch with a row count that would overflow.

## Q6. How do I run on a specific GPU?

`Engine::new_with_device(idx)`. `Engine::new()` picks device 0. There is
one CUDA context per engine, so a multi-GPU workload runs one engine per
device.

## Q7. Why does `SUM(int_col)` return `Int64` even when the column is `Int32`?

Widening. `SUM(Int32) -> Int64` gives the accumulator headroom on long
columns; `SUM(Int64)` and `SUM(Float32|Float64)` are unchanged. The
widening is applied consistently in the scalar and GROUP BY paths via
`crate::plan::logical_plan::sum_output_dtype`. Note that widening is not a
substitute for overflow safety: if the `i64` accumulator does overflow, the
query fails loudly with a `BoltError::Type("SUM(integer) overflow")` rather
than wrapping silently (the same applies to `SUM(Decimal128)`). See
[`SQL_REFERENCE.md`](SQL_REFERENCE.md) and [`LIMITATIONS.md`](LIMITATIONS.md)
for the full overflow semantics, including the grouped-`SUM` streaming caveat.

## Q8. Are `SELECT t.col FROM t` and `SELECT COL FROM t` (uppercase) accepted?

Yes — both work as of 0.5. Single-level qualified column references
(`t.col`, `alias.col`) resolve against the FROM-tree, including JOIN
aliases, in SELECT / WHERE / GROUP BY / HAVING / `JOIN ... ON`; only the
resolved column name survives lowering. Deeper qualifications
(`db.t.col`, struct-field access) are still rejected.

Identifiers are also case-insensitive: an unquoted SQL ident folds to
lowercase at parse time, and schema lookup falls back to a
case-insensitive match, so `SELECT COL` resolves to a column named
`col`. Quoted identifiers (`"MyCol"`) preserve case and match verbatim.

## Q9. Why is the CHANGELOG / NOTICE attribution "Craton Software Company"?

The project is licensed Apache-2.0 with org-level copyright attribution
to Craton Software Company. Individual contributors are covered by DCO
sign-off on commits (see `CONTRIBUTING.md`). The copyright line is a
formality of the Apache-2.0 NOTICE convention, not a CLA.

## Q10. How do I report a security issue?

Email `security@cratonsoftware.com`. Do not file public GitHub issues
for vulnerabilities. See [`../SECURITY.md`](../SECURITY.md) for the full
disclosure policy.

## Q11. Why does the codegen always target `sm_70`?

sm_70 (Volta, V100) is the realistic floor for serious GPU compute and
covers every instruction the JIT emits — `atom.global.add.f64`,
`atom.global.cas.b64`, `shfl.sync.*`, `cvta.to.global.u64`. Targeting
lower would lose `shfl.sync` and the f64 atomic add; targeting higher
would shrink the deployment surface for no codegen benefit. If you
change the floor, audit `src/jit/float_atomics.rs` first — the CAS-loop
pattern there exists because `atom.global.{min,max}.f*` is still
unavailable through sm_90.

## Q12. Why does `cargo bench` show `engine_execute` as skipped?

The bench file gates GPU benches on the `BOLT_BENCH_GPU=1`
environment variable so that contributors without a GPU can still run
`cargo bench` for the planner / codegen / CPU-reference / Polars
comparisons. Set the variable to include the GPU path. See
[`DEVELOPMENT.md`](DEVELOPMENT.md) for the bench commands.
