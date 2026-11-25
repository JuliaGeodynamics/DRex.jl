# DRex.jl

[![CI](https://github.com/JuliaGeodynamics/DRex.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaGeodynamics/DRex.jl/actions/workflows/CI.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19110048.svg)](https://doi.org/10.5281/zenodo.19110048)

A Julia package for simulating **crystallographic preferred orientation (CPO)** evolution in olivine and enstatite polycrystals during plastic deformation. It implements two models: the discrete-grain **D-Rex** model and the continuum **ODFTEX** model.

DRex.jl is a Julia translation of [PyDRex](https://github.com/seismic-anisotropy/PyDRex) (Bilton et al., 2025), which itself is based on the original FORTRAN D-Rex model by Kaminski & Ribe (2001, 2004). The Julia implementation is significantly faster (>3 orders of magnitude) thanks to allocation-free inner loops via `StaticArrays.jl`, multi-core threading, and optional GPU offloading via `KernelAbstractions.jl`.

The ODFTEX model of Ribe, Faccenda & VanderBeek (2026) is a port of the authors' Fortran code, extended to time-dependent flow. See [D-Rex and ODFTEX](@ref) for how the two models differ and how to use each.

## Features

- **Core D-Rex solver** — allocation-free per-grain rotation and volume fraction derivatives
- **ODFTEX solver** — continuum texture model with a single recrystallisation parameter; faster than D-Rex and free of sampling noise
- **CPO integration** — ODE-based tracking along pathlines (adaptive Tsit5), with grain boundary sliding and recrystallisation
- **Voigt averaging** — elastic tensor averaging over polycrystal orientations
- **Diagnostics** — M-index, PGR symmetry, elasticity decomposition, pole figures, Bingham average
- **Velocity fields** — analytical 2D fields for simple shear, convection cells, and corner flow
- **Multithreading** — `Threads.@threads` parallelisation across tracers; run with `julia -t auto`
- **GPU support** — KernelAbstractions backend (Metal, CUDA, ROCm); see [GPU & Multithreading](@ref)
- **LaMEM integration** — optional extension for coupling to 3D geodynamic simulations; see [LaMEM Integration](@ref)

## Getting Started

- [Installation](@ref) — how to install the package
- [Quick Start](@ref) — a minimal working example
- [D-Rex and ODFTEX](@ref) — the two texture models, their differences and how to use them
- [Examples](@ref) — scripts reproducing the figures of Bilton et al. (2025) and of Ribe et al. (2026)
- [GPU & Multithreading](@ref) — parallel and GPU-accelerated computation
- [LaMEM Integration](@ref) — coupling to LaMEM 3D geodynamic simulations
- [API Reference](@ref) — full function and type documentation

## Citing

If you use DRex.jl, please cite the papers of the model you use and the Zenodo archive:

- Bilton, L., Duvernay, T., Davies, D.R., Eakin, C.M., 2025. *PyDRex: predicting crystallographic preferred orientation in peridotites under steady-state and time-dependent strain*. Geophysical Journal International 241, 35–57. <https://doi.org/10.1093/gji/ggaf026>
- Ribe, N.M., Faccenda, M., VanderBeek, B., 2026. *ODFTEX: a continuum model for texture evolution with dynamic recrystallization*. Geophysical Journal International 246, 1–19. <https://doi.org/10.1093/gji/ggag170> (if you use the ODFTEX model)
- Kaus, B.J.P., 2025. *DRex.jl: A Julia package for simulating crystallographic preferred orientation (CPO) evolution* (v0.1.0). Zenodo. <https://doi.org/10.5281/zenodo.19110048>
