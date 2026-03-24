# GPU & Multithreading

!!! warning "Work in progress"
    GPU acceleration is **experimental**. The GPU path uses a different time integration
    scheme (operator-split RK4 + exponential integrator) from the default adaptive Tsit5.
    Results are close but not bit-for-bit identical to the sequential CPU path.
    **Always cross-check GPU results against the CPU reference** before using them in
    production or publication.

## Multi-core CPU

DRex parallelises across tracers using `Threads.@threads`. Launch Julia with multiple
threads to take advantage of all available cores:

```bash
julia -t auto script.jl
```

Each tracer (pathline) runs on its own thread. The per-grain inner loop is
allocation-free, so scaling is nearly linear with core count.

## GPU via KernelAbstractions.jl

The per-grain orientation kernel (`_batch_grain_kernel!`) is written with
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl),
which runs on any supported backend — CPU, Metal (Apple Silicon), CUDA, ROCm — without
changing source code.

Pass a `backend` keyword argument to use a GPU:

```julia
using Metal          # Apple Silicon; or: using CUDA / using AMDGPU
backend = Metal.MetalBackend()

# Single pathline step (low-level)
F = update_all!(minerals, params, F, get_vg, pathline; backend = backend)

# Batch mode — one kernel launch per step over all tracers × grains (recommended for GPU)
run_pathlines_batch!(minerals_per_tracer, params, pathlines_data; backend = backend)
```

The batch path launches a single kernel that covers all grains and all tracers at once,
amortising GPU launch overhead. This is the most efficient mode when running many tracers
with large grain counts (e.g. 4+ tracers × 5000 grains on Apple Silicon M-series).

## Integration schemes

| Path | Orientation | Fractions | Notes |
|---|---|---|---|
| CPU sequential (`update_orientations!`) | Adaptive Tsit5, coupled | Tsit5, coupled | Reference; most accurate |
| CPU batch (`run_pathlines_batch!` + `CPU()`) | Adaptive Tsit5, coupled | Tsit5, coupled | Same accuracy as sequential |
| GPU batch (`run_pathlines_batch!` + non-CPU) | Fixed-step RK4 | Exponential integrator (operator-split) | Fast; see caveat above |

The GPU path uses a fixed-step RK4 for orientations and an operator-split exponential
integrator for grain volume fractions. Strain energies are averaged over the four RK4
stages to reduce splitting error. The number of RK4 sub-steps per outer timestep is
determined automatically from a stability criterion. Increasing `N_TIMESTEPS_BATCH`
(number of outer timesteps) improves accuracy.

## Corner flow example with Metal

```bash
cd examples/standalone
julia --project=. cornerflow_simple.jl --metal
```

The `--metal` flag selects the Metal GPU backend and automatically enables batch mode.
Results can be compared against the CPU reference image
`cornerflow2d_simple_example.png`.
