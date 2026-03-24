# Quick Start

## Minimal example: simple shear

```julia
using LinearAlgebra
using DRex

# Create an olivine mineral with 3500 grains
mineral = Mineral(
    phase   = olivine,
    fabric  = olivine_A,
    regime  = matrix_dislocation,
    n_grains = 3500,
    seed    = 8816,
)

# Analytical simple-shear velocity gradient (strain rate 1e-15 s⁻¹)
_, get_velocity_gradient = simple_shear_2d("X", "Z", 1e-15)

# Integrate CPO forward along a (stationary) pathline
params = default_params()
timestamps = range(0, 1e15, length=26)   # 25 steps

deformation_gradient = Matrix{Float64}(I, 3, 3)
for t in eachindex(timestamps)[2:end]
    global deformation_gradient = update_orientations!(
        mineral, params, deformation_gradient,
        get_velocity_gradient,
        (timestamps[t-1], timestamps[t], _ -> zeros(3)),
    )
end

# Post-process
orientations_resampled, _ = resample_orientations(mineral.orientations, mineral.fractions)
m_index = misorientation_index(orientations_resampled[end], orthorhombic)
println("Final M-index: ", round(m_index; sigdigits=3))
```

## Key types

| Type | Description |
|---|---|
| [`Mineral`](@ref) | Stores orientation and volume-fraction histories for one mineral phase |
| [`MineralPhase`](@ref) | Enum: `olivine`, `enstatite` |
| [`MineralFabric`](@ref) | Enum: `olivine_A` through `olivine_E`, `enstatite_AB` |
| [`DeformationRegime`](@ref) | Enum: `matrix_dislocation`, `frictional_yielding`, … |

## Key functions

| Function | Description |
|---|---|
| [`update_orientations!`](@ref) | Integrate one mineral phase along one pathline step |
| [`update_all!`](@ref) | Integrate all phases for one pathline step |
| [`run_pathlines_batch!`](@ref) | Integrate multiple tracers simultaneously (GPU-capable) |
| [`misorientation_index`](@ref) | Compute M-index (CPO strength 0–1) |
| [`bingham_average`](@ref) | Mean crystallographic axis direction |
| [`resample_orientations`](@ref) | Resample orientations weighted by volume fractions |

See the [API Reference](@ref) for the full list.
