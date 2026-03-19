# LaMEM coupling types and function stubs.
#
# The actual implementations live in ext/DRexLaMEMExt.jl and are loaded
# automatically when both LaMEM and GeophysicalModelGenerator are imported
# alongside DRex:
#
#   using DRex, LaMEM, GeophysicalModelGenerator
#
# Without those packages the stubs below throw a descriptive MethodError.

# --------------------------------------------------------------------------- #
# Types (pure Julia — no LaMEM dependency)
# --------------------------------------------------------------------------- #

"""
    LaMEMSnapshot

One timestep of LaMEM output: velocity gradient tensor and (optionally) velocity
on a structured rectilinear grid.

| Field      | Units   | Description                          |
|------------|---------|--------------------------------------|
| `time`     | Myr     |                                      |
| `x,y,z`    | km      | 1-D coordinate vectors               |
| `vel_grad` | 1/Myr   | 3×3×nx×ny×nz array (converted from LaMEM's 1/s) |
| `velocity` | km/Myr  | 3×nx×ny×nz array or `nothing` (converted from cm/yr) |
"""
struct LaMEMSnapshot
    time::Float64
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    vel_grad::Array{Float64,5}                     # 3×3×nx×ny×nz
    velocity::Union{Array{Float64,4}, Nothing}     # 3×nx×ny×nz  or  nothing
end

"""
    CPOTracer

A Lagrangian particle that carries crystallographic preferred orientation (CPO)
through a LaMEM simulation.

| Field                  | Description                              |
|------------------------|------------------------------------------|
| `position`             | Current [x,y,z] in km                   |
| `positions`            | Position history (one entry per step)   |
| `minerals`             | One `Mineral` per phase                 |
| `deformation_gradient` | 3×3 deformation gradient tensor F       |
"""
mutable struct CPOTracer
    position::Vector{Float64}
    positions::Vector{Vector{Float64}}
    minerals::Vector{Mineral}
    deformation_gradient::Matrix{Float64}
end

# --------------------------------------------------------------------------- #
# Function stubs (implemented in ext/DRexLaMEMExt.jl)
# --------------------------------------------------------------------------- #

"""
    load_snapshots(sim_name, dir=""; vel_grad_field=:vel_gr_tensor, load_velocity=true)
    → Vector{LaMEMSnapshot}

Read all timesteps of a LaMEM simulation.
Requires `LaMEM` and `GeophysicalModelGenerator` to be loaded.
"""
function load_snapshots end

"""
    create_tracers(positions; n_grains=1000, seed=42,
                   phase_assemblage=[olivine,enstatite], fabric=olivine_A)
    → Vector{CPOTracer}

Create CPO tracers with random initial orientations at the given positions.
Requires `LaMEM` and `GeophysicalModelGenerator` to be loaded.
"""
function create_tracers end

"""
    evolve_cpo!(tracers, params, snapshots; advect=true, n_substeps=5)

Evolve CPO on all tracers through the LaMEM snapshot sequence.
Requires `LaMEM` and `GeophysicalModelGenerator` to be loaded.
"""
function evolve_cpo! end

"""
    backtrack_positions(target_positions, snapshots; n_substeps=5)
    → Vector{Vector{Float64}}

Advect positions backward through the snapshot sequence to find source positions
at `snapshots[1]`.
Requires `LaMEM` and `GeophysicalModelGenerator` to be loaded.
"""
function backtrack_positions end

"""
    run_cpo_at_locations(target_positions, snapshots, drex_params;
                         n_grains=1000, seed=42, n_substeps=5, fabric=olivine_A)
    → Vector{CPOTracer}

High-level function: compute CPO at known target locations at the final snapshot.
Backtracks positions to find material source, then runs CPO forward.
Requires `LaMEM` and `GeophysicalModelGenerator` to be loaded.
"""
function run_cpo_at_locations end

"""
    compute_cpo_from_lamem(sim_name, sim_dir;
                           target_positions=nothing, initial_positions=nothing,
                           output_dir="\$(sim_name)_tracers",
                           skip_initial_steps=5, start_step=nothing, end_step=nothing,
                           steady_state_step=nothing, steady_state_duration=nothing,
                           steady_state_n_steps=100,
                           drex_params=default_params(), n_grains=1000, seed=42,
                           n_substeps=5, fabric=olivine_A, vel_grad_field=:vel_gr_tensor)

All-in-one function: load LaMEM output, (optionally backtrack positions), evolve
CPO forward, and write a Paraview time-series (.pvd + .vtp files).

Returns `(tracers, snapshots_used)`.

Exactly one of `target_positions` or `initial_positions` must be provided:
- `target_positions` — `[x,y,z]` km where you want CPO at the **last used snapshot**;
  positions are backtracked to find the material source.
- `initial_positions` — `[x,y,z]` km of material at the **first used snapshot**;
  CPO is evolved forward from here directly.

# Time-dependent mode (default)
Reads velocity gradients from every snapshot in the usable window.
- `skip_initial_steps` — drop the first N snapshots (spin-up transients)
- `start_step` / `end_step` — optional sub-window within the usable range

# Steady-state mode
Set `steady_state_step` and `steady_state_duration` to use a single snapshot's
velocity field for the entire integration (constant flow assumption).
- `steady_state_step`     — index into the usable snapshot window for the reference field
- `steady_state_duration` — total integration time (Myr)
- `steady_state_n_steps`  — number of output steps (default 100)

Requires `LaMEM` and `GeophysicalModelGenerator` to be loaded.
"""
function compute_cpo_from_lamem end

"""
    trilinear_interpolate(x_grid, y_grid, z_grid, field_3d, point) → Float64

Trilinear interpolation of a scalar 3D field at a point. Boundary-clamped.
Requires `LaMEM` and `GeophysicalModelGenerator` to be loaded.
"""
function trilinear_interpolate end

"""
    interpolate_vel_grad(snap, point) → Matrix{Float64}(3,3)

Interpolate the 3×3 velocity gradient at an arbitrary point from a snapshot.
Requires `LaMEM` and `GeophysicalModelGenerator` to be loaded.
"""
function interpolate_vel_grad end

"""
    interpolate_velocity(snap, point) → Vector{Float64}(3)

Interpolate the velocity vector at an arbitrary point from a snapshot.
Requires `LaMEM` and `GeophysicalModelGenerator` to be loaded.
"""
function interpolate_velocity end

"""
    make_velocity_gradient_func(snap_prev, snap_next) → Function

Return a `(t, x) → 3×3 Matrix` callable that linearly interpolates the
velocity gradient in time and trilinearly in space between two snapshots.
Requires `LaMEM` and `GeophysicalModelGenerator` to be loaded.
"""
function make_velocity_gradient_func end

"""
    make_velocity_func(snap_prev, snap_next) → Function

Return a `(t, x) → Vector` callable that linearly interpolates velocity
in time and trilinearly in space between two snapshots.
Requires `LaMEM` and `GeophysicalModelGenerator` to be loaded.
"""
function make_velocity_func end
