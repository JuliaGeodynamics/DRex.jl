# DRex.jl

[![CI](https://github.com/JuliaGeodynamics/DRex.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaGeodynamics/DRex.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGeodynamics.github.io/DRex.jl/dev)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19110048.svg)](https://doi.org/10.5281/zenodo.19110048)

A Julia implementation of the D-Rex model for simulating **crystallographic preferred orientation (CPO)** evolution in olivine and enstatite polycrystals.

Based on [PyDRex](https://github.com/seismic-anisotropy/PyDRex) (Bilton et al., 2025) and the original FORTRAN D-Rex by Kaminski & Ribe (2001, 2004). Runs >3 orders of magnitude faster than the Python version through allocation-free inner loops, multi-core threading, and (experimental) optional GPU offloading.

📖 **[Full documentation](https://JuliaGeodynamics.github.io/DRex.jl)**

## Installation

```julia
] add https://github.com/JuliaGeodynamics/DRex.jl
```

## Quick example

```julia
using LinearAlgebra, DRex

mineral = Mineral(phase=olivine, fabric=olivine_A, regime=matrix_dislocation, n_grains=3500)
_, get_L = simple_shear_2d("X", "Z", 1e-15)
params = default_params()
F = Matrix{Float64}(I, 3, 3)
for t in eachindex(range(0, 1e15, length=26))[2:end]
    ts = range(0, 1e15, length=26)
    global F = update_orientations!(mineral, params, F, get_L, (ts[t-1], ts[t], _ -> zeros(3)))
end
```

Run with multiple threads: `julia -t auto script.jl`

## Citing

DRex.jl was developed by translating PyDRex (Bilton et al., 2025) to Julia, including all tests. If you find this package useful, please give credit to the original authors by citing their work:

- Bilton, L., Duvernay, T., Davies, D.R., Eakin, C.M., 2025. PyDRex: predicting crystallographic preferred orientation in peridotites under steady-state and time-dependent strain. *Geophysical Journal International* 241, 35–57. <https://doi.org/10.1093/gji/ggaf026>

As there are no new scientific features compared to the Python version, we do not plan a separate publication. You can cite the Julia package itself from Zenodo:

- Kaus, B.J.P., 2025. DRex.jl: A Julia package for simulating crystallographic preferred orientation (CPO) evolution (v0.1.0). Zenodo. <https://doi.org/10.5281/zenodo.19110048>

## References

- Kaminski, É. & Ribe, N.M. (2001). A kinematic model for recrystallization and texture development in olivine polycrystals. *Earth and Planetary Science Letters*, 189(3-4), 253–267.
- Kaminski, É. & Ribe, N.M. (2004). Timescales for the evolution of seismic anisotropy in mantle flow. *Geochemistry, Geophysics, Geosystems*, 3(1).
- Browaeys, J.T. & Chevrot, S. (2004). Decomposition of the elastic tensor and geophysical applications. *Geophysical Journal International*, 159(2), 667–678.
