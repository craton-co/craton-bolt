# cudarc backend

`cudarc` is Craton Bolt's supported, default CUDA context/driver adapter
(`default = ["cudarc"]`). It owns the device primary context and is exercised
by the blocking real-GPU test lane. The public CUDA-Oxide layer (`GpuVec`,
`GpuView`, `GpuViewMut`, and `GpuBuffer`) is unchanged.

## Supported boundary

The adapter in `src/cuda/cudarc_backend.rs` owns:

- primary-context selection and per-thread binding;
- allocation/free;
- synchronous and asynchronous host/device copies; and
- asynchronous memset.

Module loading, launches, events, streams, and several less common Driver API
calls still use the small ABI declarations in `src/cuda/cuda_sys.rs`. Those
calls execute against the cudarc-owned current context. This is one supported
hybrid adapter, not two competing backends.

The dependency selects cudarc's CUDA 12.6 API surface and has been validated
against CUDA 12.x and CUDA 13.3 drivers/toolkits.

## Context and engine invariant

The process-wide memory pool still owns context-bound allocations. Stream,
module, and graph caches are context-tagged, but the memory-pool registry is not
yet per-context. Consequently Craton Bolt supports exactly one live `Engine`
per process. `EngineBuilder::build` enforces that boundary atomically and
returns a deterministic error for a second live engine; it no longer permits
cross-context handles to reach CUDA.

Multi-engine or multi-GPU applications should use one process per device until
the memory-pool registry becomes per-context. Dropping the live engine releases
the permit after the CUDA context and its resources are torn down.

## Validation

The canonical GPU lane sets `BOLT_BENCH_GPU=1` and runs ignored tests with
`--test-threads=1`. This is required because the CUDA context and process-global
resource registries are deliberately serialized. The lane covers allocation,
copy, module, launch, stream, sort, string, aggregation, and query-level
behavior on actual hardware.

`cuda-stub` remains the explicit no-device configuration:

```bash
cargo test --no-default-features --features cuda-stub
```

It validates planning, host fallbacks, and PTX shape only; it is not evidence
for device execution.
