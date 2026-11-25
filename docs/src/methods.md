# D-Rex and ODFTEX

DRex.jl contains two models for the evolution of crystallographic preferred orientation (CPO) in olivine–enstatite aggregates. Both take the same input — a velocity gradient along a pathline — and both give orientations from which the usual diagnostics and elastic tensors are computed. They differ in how the texture is represented and in how recrystallisation is described.

- **D-Rex** (Kaminski & Ribe 2001; Kaminski et al. 2004; Bilton et al. 2025) follows a set of discrete grains.
- **ODFTEX** (Ribe, Faccenda & VanderBeek 2026) follows a continuous orientation distribution function (ODF).

## How the two models differ

| | D-Rex | ODFTEX |
|---|---|---|
| Texture representation | A few thousand grains, each with an orientation and a volume fraction | A continuous ODF ``f(\varphi,\theta,\psi)`` on a grid of `nbox`³ boxes in Euler-angle space |
| Slip systems | Several per mineral, weighted by their critical resolved shear stresses; olivine fabrics A–E | One: (010)[100] for olivine (A-type). Orthopyroxene is obtained from the same ODF by a fixed rotation, so adding it costs nothing |
| Recrystallisation | Grain nucleation and grain-boundary migration driven by strain energy, plus grain-boundary sliding of small grains | One term in the evolution equation: grains that are poorly oriented for slip are consumed faster |
| Free parameters | Three: `gbm_mobility` (M\*), `nucleation_efficiency` (λ\*), `gbs_threshold` (χ) | One: `λ`, calibrated to 2.6 against a simple-shear experiment and confirmed against uniaxial compression |
| Numerical method | Adaptive ODE integration of every grain | Method of characteristics with second-order midpoint steps, at most 0.025 strain per step |
| Randomness | Random initial grain orientations (`seed`), so results carry sampling noise | Deterministic; the ODF itself has no sampling noise |
| Large strain | Texture strength set by the balance of the three parameters | Texture grows without bound unless the optional limiter `rmax` freezes it beyond a given finite strain |
| Cost | Grows with the number of grains | Grows with `nbox`³; `nbox = 12` is enough for seismic anisotropy |

In practice:

- **Speed.** ODFTEX is faster. The paper reports a factor of about 6 against D-Rex_M; with this package the 2-D convection example below ran 21 times faster with ODFTEX (12³ boxes against 2 × 1000 grains), and the LaMEM subduction example about 1.8 times faster (30³ boxes).
- **Fast directions agree.** In both examples the fast axes of the two models differ by only a few degrees on average (3° for the P-wave fast direction in the convection example, a median of 10° in the 3-D subduction example).
- **Texture strength can differ.** Without the limiter ODFTEX textures keep sharpening with strain. In the subduction example ODFTEX reached an M-index of 0.85 at a strain of 2.5 where D-Rex with M\* = 10 gave 0.30. Use `rmax` (the paper uses 5) if this matters; the tracer pipeline does not set it.
- **What ODFTEX does not have here.** Only A-type olivine is supported, the D-Rex parameters are ignored, and there is no batch or GPU path.

## Using them

Both models are driven the same way: a function `get_velocity_gradient(t, x)` and a pathline `(t_start, t_end, get_position)`.

```julia
using LinearAlgebra, DRex

# Simple shear in the X–Z plane at 1e-15 s⁻¹, followed for 1e15 s
_, get_L = simple_shear_2d("X", "Z", 1e-15)
ts = range(0, 1e15, length = 26)
```

**D-Rex** evolves one or more [`Mineral`](@ref)s:

```julia
mineral = Mineral(phase = olivine, fabric = olivine_A, regime = matrix_dislocation, n_grains = 3500)
params  = default_params()            # M*, λ*, χ, phase fractions, …

F = Matrix{Float64}(I, 3, 3)
for i in 2:length(ts)
    global F = update_orientations!(mineral, params, F, get_L, (ts[i-1], ts[i], _ -> zeros(3)))
end

ori, _ = resample_orientations(mineral.orientations, mineral.fractions)   # weight grains by volume
misorientation_index(ori[end], orthorhombic)
```

**ODFTEX** evolves an [`ODFState`](@ref):

```julia
odf = init_odf(30; λ = 2.6)           # 30³ Euler boxes, isotropic start

F = Matrix{Float64}(I, 3, 3)
for i in 2:length(ts)
    global F = odftex_update!(odf, F, get_L, (ts[i-1], ts[i], _ -> zeros(3)))   # add rmax = 5 for the limiter
end

odf_misorientation_index(odf)                   # texture strength
odf_mean_axis(odf, "a")                         # mean [100] direction
x, y, density = odf_polefigure_density(odf, 1)  # [100] pole figure

# Elastic tensor of a 70:30 olivine–enstatite aggregate
minerals = odf_to_minerals(odf)
C = voigt_averages(minerals, [olivine, enstatite], [0.7, 0.3])
```

[`odftex_update!`](@ref) works in any consistent units (for example Myr and 1/Myr) and handles time-dependent flow. To reuse the grain-based diagnostics, [`sample_orientations`](@ref) draws discrete orientations from the ODF and [`odf_to_minerals`](@ref) converts it exactly into weighted grains.

With LaMEM output the choice is a single keyword, `method = :drex` or `method = :odftex`; see [LaMEM Integration](@ref).

## Which one to use

- Use **ODFTEX** when you need seismic anisotropy for many tracers in an A-type olivine mantle and want a fast, noise-free result with one calibrated parameter.
- Use **D-Rex** when you need other olivine fabrics (B–E), want control over the recrystallisation parameters, or want to compare with the large body of existing D-Rex results.

## References

- Ribe, N.M., Faccenda, M. & VanderBeek, B., 2026. ODFTEX: a continuum model for texture evolution with dynamic recrystallization. *Geophysical Journal International* 246, 1–19. <https://doi.org/10.1093/gji/ggag170>
- Bilton, L., Duvernay, T., Davies, D.R. & Eakin, C.M., 2025. PyDRex: predicting crystallographic preferred orientation in peridotites under steady-state and time-dependent strain. *Geophysical Journal International* 241, 35–57. <https://doi.org/10.1093/gji/ggaf026>
- Kaminski, É. & Ribe, N.M., 2001. A kinematic model for recrystallization and texture development in olivine polycrystals. *Earth and Planetary Science Letters* 189, 253–267.
