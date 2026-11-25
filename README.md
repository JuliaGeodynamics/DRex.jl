# DRex.jl

[![CI](https://github.com/JuliaGeodynamics/DRex.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaGeodynamics/DRex.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGeodynamics.github.io/DRex.jl/dev)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19110048.svg)](https://doi.org/10.5281/zenodo.19110048)

A Julia package for simulating **crystallographic preferred orientation (CPO)** evolution in olivine and enstatite polycrystals. It implements two texture models:

- **D-Rex** — follows a set of discrete grains (Kaminski & Ribe, 2001; Bilton et al., 2025).
- **ODFTEX** — follows a continuous orientation distribution function (Ribe, Faccenda & VanderBeek, 2026).

The D-Rex part is based on [PyDRex](https://github.com/seismic-anisotropy/PyDRex) (Bilton et al., 2025) and the original FORTRAN D-Rex by Kaminski & Ribe (2001, 2004). It runs >3 orders of magnitude faster than the Python version through allocation-free inner loops, multi-core threading, and (experimental) optional GPU offloading. The ODFTEX part is a port of the Fortran code of Ribe et al. (2026), extended to time-dependent flow.

📖 **[Full documentation](https://JuliaGeodynamics.github.io/DRex.jl)**

## Installation

```julia
] add https://github.com/JuliaGeodynamics/DRex.jl
```

## D-Rex or ODFTEX?

| | D-Rex | ODFTEX |
|---|---|---|
| Texture | A few thousand grains with orientations and volume fractions | Continuous ODF on a grid in Euler-angle space |
| Slip systems | Several per mineral; olivine fabrics A–E | One, (010)[100]: A-type olivine only |
| Recrystallisation | Nucleation, grain-boundary migration and sliding; three parameters (M\*, λ\*, χ) | One parameter, λ = 2.6 |
| Results | Depend on the random initial grains | Deterministic, no sampling noise |
| Speed | Slower | Faster (about 6× in the paper; 2–20× in the examples here) |

The two models predict similar fast directions; texture strength can differ at large strain. The [documentation](https://JuliaGeodynamics.github.io/DRex.jl) has the details and reproduces the figures of both papers.

## Quick example

Both models are driven by a velocity gradient along a pathline:

```julia
using LinearAlgebra, DRex

# Simple shear in the X–Z plane at 1e-15 s⁻¹, followed for 1e15 s
_, get_L = simple_shear_2d("X", "Z", 1e-15)
ts = range(0, 1e15, length=26)

# D-Rex: discrete grains
mineral = Mineral(phase=olivine, fabric=olivine_A, regime=matrix_dislocation, n_grains=3500)
params  = default_params()
F = Matrix{Float64}(I, 3, 3)
for i in 2:length(ts)
    global F = update_orientations!(mineral, params, F, get_L, (ts[i-1], ts[i], _ -> zeros(3)))
end
ori, _ = resample_orientations(mineral.orientations, mineral.fractions)
println("D-Rex  M-index: ", misorientation_index(ori[end], orthorhombic))

# ODFTEX: continuous orientation distribution on a 30³ Euler grid
odf = init_odf(30; λ=2.6)
F = Matrix{Float64}(I, 3, 3)
for i in 2:length(ts)
    global F = odftex_update!(odf, F, get_L, (ts[i-1], ts[i], _ -> zeros(3)))
end
println("ODFTEX M-index: ", odf_misorientation_index(odf))

# Elastic tensor of a 70:30 olivine–enstatite aggregate from the ODF
C = voigt_averages(odf_to_minerals(odf), [olivine, enstatite], [0.7, 0.3])
```

Run with multiple threads: `julia -t auto script.jl`

When post-processing [LaMEM](https://github.com/JuliaGeodynamics/LaMEM.jl) simulations the model is selected with one keyword, `method = :drex` or `method = :odftex`.

## Citing

DRex.jl was developed by translating PyDRex (Bilton et al., 2025) to Julia, including all tests. If you find this package useful, please give credit to the original authors by citing their work:

- Bilton, L., Duvernay, T., Davies, D.R., Eakin, C.M., 2025. PyDRex: predicting crystallographic preferred orientation in peridotites under steady-state and time-dependent strain. *Geophysical Journal International* 241, 35–57. <https://doi.org/10.1093/gji/ggaf026>

If you use the ODFTEX model, please cite:

- Ribe, N.M., Faccenda, M., VanderBeek, B., 2026. ODFTEX: a continuum model for texture evolution with dynamic recrystallization. *Geophysical Journal International* 246, 1–19. <https://doi.org/10.1093/gji/ggag170>

As there are no new scientific features compared to the Python and Fortran version, we do not plan a separate publication. You can cite the Julia package itself from Zenodo:

- Kaus, B.J.P., 2025. DRex.jl: A Julia package for simulating crystallographic preferred orientation (CPO) evolution (v0.1.0). Zenodo. <https://doi.org/10.5281/zenodo.19110048>

## References

- Ribe, N.M., Faccenda, M. & VanderBeek, B. (2026). ODFTEX: a continuum model for texture evolution with dynamic recrystallization. *Geophysical Journal International*, 246, 1–19.
- Kaminski, É. & Ribe, N.M. (2001). A kinematic model for recrystallization and texture development in olivine polycrystals. *Earth and Planetary Science Letters*, 189(3-4), 253–267.
- Kaminski, É. & Ribe, N.M. (2004). Timescales for the evolution of seismic anisotropy in mantle flow. *Geochemistry, Geophysics, Geosystems*, 3(1).
- Browaeys, J.T. & Chevrot, S. (2004). Decomposition of the elastic tensor and geophysical applications. *Geophysical Journal International*, 159(2), 667–678.
