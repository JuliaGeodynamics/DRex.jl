# DRexLaMEMExt — LaMEM coupling extension for DRex.jl
#
# Activated automatically when both LaMEM and GeophysicalModelGenerator are
# loaded alongside DRex:
#
#   using DRex, LaMEM, GeophysicalModelGenerator
#
# Implements all functions declared as stubs in src/lamem_coupling.jl.

module DRexLaMEMExt

using DRex
using LaMEM, GeophysicalModelGenerator
using LinearAlgebra

using WriteVTK

import DRex:
    LaMEMSnapshot, CPOTracer,
    load_snapshots, create_tracers, evolve_cpo!,
    backtrack_positions, run_cpo_at_locations, compute_cpo_from_lamem,
    trilinear_interpolate, interpolate_vel_grad, interpolate_velocity,
    make_velocity_gradient_func, make_velocity_func

# --------------------------------------------------------------------------- #
# Unit conversion constant
# --------------------------------------------------------------------------- #

const SECONDS_PER_MYR = 1e6 * 365.25 * 24 * 3600   # ≈ 3.15576×10¹³ s/Myr

# --------------------------------------------------------------------------- #
# Loading LaMEM output
# --------------------------------------------------------------------------- #

"""
    load_snapshots(sim_name, dir=""; vel_grad_field=:vel_gr_tensor, load_velocity=true)
    → Vector{LaMEMSnapshot}

Read all LaMEM output timesteps from a completed simulation.

# Units stored in the returned snapshots
| Quantity   | Unit   | Conversion from LaMEM output       |
|------------|--------|------------------------------------|
| time       | Myr    | as-is                              |
| vel_grad   | 1/Myr  | LaMEM 1/s × SECONDS_PER_MYR       |
| velocity   | km/Myr | LaMEM cm/yr × 10                  |
| positions  | km     | as-is                              |
"""
function load_snapshots(
    sim_name::String,
    dir::String = "";
    vel_grad_field::Symbol = :vel_gr_tensor,
    load_velocity::Bool = true,
)
    timesteps, _, times = read_LaMEM_simulation(sim_name, dir)
    snapshots = LaMEMSnapshot[]
    for ts in timesteps
        data, t = read_LaMEM_timestep(sim_name, ts, dir)
        push!(snapshots, _extract_snapshot(data, t[1], vel_grad_field, load_velocity))
    end
    @info "Loaded $(length(snapshots)) snapshots (t = $(times[1]) → $(times[end]))"
    return snapshots
end

function _extract_snapshot(data, time, vel_grad_field, load_velocity)
    x  = Float64.(data.x.val[:, 1, 1])
    y  = Float64.(data.y.val[1, :, 1])
    z  = Float64.(data.z.val[1, 1, :])
    nx, ny, nz = length(x), length(y), length(z)

    # Velocity gradient: LaMEM 1/s → 1/Myr
    vg       = data.fields[vel_grad_field]
    vel_grad = Array{Float64,5}(undef, 3, 3, nx, ny, nz)
    k = 1
    for i in 1:3, j in 1:3
        vel_grad[i, j, :, :, :] .= Float64.(vg[k]) .* SECONDS_PER_MYR
        k += 1
    end

    # Velocity: LaMEM cm/yr → km/Myr  (1 cm/yr = 10 km/Myr)
    velocity = nothing
    if load_velocity && haskey(data.fields, :velocity)
        vel      = data.fields[:velocity]
        velocity = Array{Float64,4}(undef, 3, nx, ny, nz)
        for c in 1:3
            velocity[c, :, :, :] .= Float64.(vel[c]) .* 10.0
        end
    end

    return LaMEMSnapshot(time, x, y, z, vel_grad, velocity)
end

# --------------------------------------------------------------------------- #
# Trilinear interpolation
# --------------------------------------------------------------------------- #

function trilinear_interpolate(
    x_grid::Vector{Float64}, y_grid::Vector{Float64}, z_grid::Vector{Float64},
    field_3d::AbstractArray{Float64,3}, point,
)
    px, py, pz = point[1], point[2], point[3]
    nx, ny, nz = length(x_grid), length(y_grid), length(z_grid)

    ix = clamp(searchsortedlast(x_grid, px), 1, nx - 1)
    iy = clamp(searchsortedlast(y_grid, py), 1, ny - 1)
    iz = clamp(searchsortedlast(z_grid, pz), 1, nz - 1)

    tx = clamp((px - x_grid[ix]) / (x_grid[ix+1] - x_grid[ix]), 0.0, 1.0)
    ty = clamp((py - y_grid[iy]) / (y_grid[iy+1] - y_grid[iy]), 0.0, 1.0)
    tz = clamp((pz - z_grid[iz]) / (z_grid[iz+1] - z_grid[iz]), 0.0, 1.0)

    return (field_3d[ix,   iy,   iz  ] * (1-tx)*(1-ty)*(1-tz) +
            field_3d[ix+1, iy,   iz  ] * tx    *(1-ty)*(1-tz) +
            field_3d[ix,   iy+1, iz  ] * (1-tx)*ty    *(1-tz) +
            field_3d[ix+1, iy+1, iz  ] * tx    *ty    *(1-tz) +
            field_3d[ix,   iy,   iz+1] * (1-tx)*(1-ty)*tz     +
            field_3d[ix+1, iy,   iz+1] * tx    *(1-ty)*tz     +
            field_3d[ix,   iy+1, iz+1] * (1-tx)*ty    *tz     +
            field_3d[ix+1, iy+1, iz+1] * tx    *ty    *tz)
end

function interpolate_vel_grad(snap::LaMEMSnapshot, point)
    L = Matrix{Float64}(undef, 3, 3)
    for i in 1:3, j in 1:3
        L[i, j] = trilinear_interpolate(snap.x, snap.y, snap.z,
                                         @view(snap.vel_grad[i, j, :, :, :]), point)
    end
    return L
end

function interpolate_velocity(snap::LaMEMSnapshot, point)
    v = Vector{Float64}(undef, 3)
    for c in 1:3
        v[c] = trilinear_interpolate(snap.x, snap.y, snap.z,
                                      @view(snap.velocity[c, :, :, :]), point)
    end
    return v
end

# --------------------------------------------------------------------------- #
# Time-interpolated callables
# --------------------------------------------------------------------------- #

function make_velocity_gradient_func(snap_prev::LaMEMSnapshot, snap_next::LaMEMSnapshot)
    t0, t1 = snap_prev.time, snap_next.time
    dt = t1 - t0
    function get_velocity_gradient(t, x)
        α  = dt ≈ 0 ? 0.0 : clamp((t - t0) / dt, 0.0, 1.0)
        L0 = interpolate_vel_grad(snap_prev, x)
        L1 = interpolate_vel_grad(snap_next, x)
        return (1 - α) .* L0 .+ α .* L1
    end
    return get_velocity_gradient
end

function make_velocity_func(snap_prev::LaMEMSnapshot, snap_next::LaMEMSnapshot)
    t0, t1 = snap_prev.time, snap_next.time
    dt = t1 - t0
    function get_velocity(t, x)
        α  = dt ≈ 0 ? 0.0 : clamp((t - t0) / dt, 0.0, 1.0)
        v0 = interpolate_velocity(snap_prev, x)
        v1 = interpolate_velocity(snap_next, x)
        return (1 - α) .* v0 .+ α .* v1
    end
    return get_velocity
end

# --------------------------------------------------------------------------- #
# Particle advection helpers
# --------------------------------------------------------------------------- #

function _advect_particle!(tracer::CPOTracer, get_velocity, t_start, t_end;
                           n_substeps::Int = 10)
    dt_sub = (t_end - t_start) / n_substeps
    t = t_start
    for _ in 1:n_substeps
        v = get_velocity(t, tracer.position)
        tracer.position .+= v .* dt_sub
        t += dt_sub
    end
end

# --------------------------------------------------------------------------- #
# Time-dependent CPO evolution
# --------------------------------------------------------------------------- #

function evolve_cpo!(
    tracers::Vector{CPOTracer},
    params::Dict{Symbol,Any},
    snapshots::Vector{LaMEMSnapshot};
    advect::Bool    = true,
    n_substeps::Int = 10,
)
    n_steps = length(snapshots) - 1
    for step in 1:n_steps
        snap_prev = snapshots[step]
        snap_next = snapshots[step + 1]
        t_start   = snap_prev.time
        t_end     = snap_next.time

        get_vel_grad = make_velocity_gradient_func(snap_prev, snap_next)

        has_velocity = snap_prev.velocity !== nothing && snap_next.velocity !== nothing
        get_velocity = (advect && has_velocity) ?
                       make_velocity_func(snap_prev, snap_next) : nothing

        Threads.@threads for tracer in tracers
            pos_start = copy(tracer.position)

            if get_velocity !== nothing
                _advect_particle!(tracer, get_velocity, t_start, t_end;
                                  n_substeps = n_substeps)
            end

            pos_end = copy(tracer.position)
            push!(tracer.positions, pos_end)

            get_position = let p0 = pos_start, p1 = pos_end, t0 = t_start, t1 = t_end
                function(t)
                    α = t1 ≈ t0 ? 0.0 : clamp((t - t0) / (t1 - t0), 0.0, 1.0)
                    return (1 - α) .* p0 .+ α .* p1
                end
            end

            pathline = (t_start, t_end, get_position)
            tracer.deformation_gradient = update_all!(
                tracer.minerals, params, tracer.deformation_gradient,
                get_vel_grad, pathline,
            )
        end

        @info "Step $step/$n_steps (t = $(round(t_start; sigdigits=4)) → $(round(t_end; sigdigits=4)))"
    end
end

# --------------------------------------------------------------------------- #
# Steady-state CPO evolution (single snapshot, constant velocity field)
# --------------------------------------------------------------------------- #

"""
    _evolve_cpo_steady!(tracers, params, snap, duration, n_steps; n_substeps, advect)

Evolve CPO assuming a fixed (steady-state) velocity field from `snap` for
`duration` Myr.  The interval is split into `n_steps` output steps, each
subdivided into `n_substeps` sub-steps for integration.
"""
function _evolve_cpo_steady!(
    tracers::Vector{CPOTracer},
    params::Dict{Symbol,Any},
    snap::LaMEMSnapshot,
    duration::Float64,
    n_steps::Int;
    n_substeps::Int = 5,
    advect::Bool    = true,
)
    has_velocity = snap.velocity !== nothing
    dt_step      = duration / n_steps
    t            = 0.0

    for step in 1:n_steps
        t_start = t
        t_end   = t + dt_step

        Threads.@threads for tracer in tracers
            pos_start = copy(tracer.position)

            if advect && has_velocity
                dt_sub = dt_step / n_substeps
                for _ in 1:n_substeps
                    v = interpolate_velocity(snap, tracer.position)
                    tracer.position .+= v .* dt_sub
                end
            end

            pos_end = copy(tracer.position)
            push!(tracer.positions, pos_end)

            get_position = let p0 = pos_start, p1 = pos_end, t0 = t_start, t1 = t_end
                (t__) -> begin
                    α = (t1 - t0) ≈ 0.0 ? 0.0 : clamp((t__ - t0) / (t1 - t0), 0.0, 1.0)
                    (1 - α) .* p0 .+ α .* p1
                end
            end

            get_vel_grad = (_, x) -> interpolate_vel_grad(snap, x)

            pathline = (t_start, t_end, get_position)
            tracer.deformation_gradient = update_all!(
                tracer.minerals, params, tracer.deformation_gradient,
                get_vel_grad, pathline,
            )
        end

        t += dt_step
        @info "Steady-state step $step/$n_steps (t = $(round(t_start; sigdigits=4)) → $(round(t_end; sigdigits=4)) Myr)"
    end
end

"""
    _backtrack_steady(positions, snap, duration, n_substeps) → Vector{Vector{Float64}}

Backtrack positions backward in time for `duration` Myr using the constant
velocity field from a single snapshot.
"""
function _backtrack_steady(
    positions::Vector{Vector{Float64}},
    snap::LaMEMSnapshot,
    duration::Float64,
    n_substeps::Int,
)
    if snap.velocity === nothing
        @warn "No velocity data in snapshot — cannot backtrack; returning target positions"
        return [copy(p) for p in positions]
    end
    dt     = duration / n_substeps
    result = [copy(p) for p in positions]
    Threads.@threads for i in eachindex(result)
        pos = result[i]
        for _ in 1:n_substeps
            v    = interpolate_velocity(snap, pos)
            pos .-= v .* dt
        end
    end
    return result
end

# --------------------------------------------------------------------------- #
# Tracer creation
# --------------------------------------------------------------------------- #

function create_tracers(
    positions::Vector{Vector{Float64}};
    n_grains::Int                          = 1000,
    seed::Union{Int,Nothing}               = nothing,
    phase_assemblage::Vector{MineralPhase} = [olivine, enstatite],
    fabric::MineralFabric                  = olivine_A,
)
    tracers = CPOTracer[]
    for (i, pos) in enumerate(positions)
        minerals = Mineral[]
        for (j, phase) in enumerate(phase_assemblage)
            s            = isnothing(seed) ? nothing : seed + i * 1000 + j
            phase_fabric = phase == enstatite ? enstatite_AB : fabric
            push!(minerals, Mineral(; phase = phase, fabric = phase_fabric,
                                      n_grains = n_grains, seed = s))
        end
        push!(tracers, CPOTracer(copy(pos), [copy(pos)], minerals,
                                 Matrix{Float64}(I, 3, 3)))
    end
    return tracers
end

# --------------------------------------------------------------------------- #
# Backward advection (time-dependent)
# --------------------------------------------------------------------------- #

function backtrack_positions(
    target_positions::Vector{Vector{Float64}},
    snapshots::Vector{LaMEMSnapshot};
    n_substeps::Int = 5,
)
    if !all(s -> s.velocity !== nothing, snapshots)
        @warn "No velocity data — backtracking unavailable; returning copies of target positions"
        return [copy(p) for p in target_positions]
    end

    positions = [copy(p) for p in target_positions]
    n_steps   = length(snapshots) - 1

    Threads.@threads for i in eachindex(positions)
        pos = positions[i]
        for step in n_steps:-1:1
            snap_prev    = snapshots[step]
            snap_next    = snapshots[step + 1]
            t_start      = snap_prev.time
            t_end        = snap_next.time
            dt_sub       = (t_end - t_start) / n_substeps
            get_velocity = make_velocity_func(snap_prev, snap_next)
            t = t_end
            for _ in 1:n_substeps
                v    = get_velocity(t, pos)
                pos .-= v .* dt_sub
                t   -= dt_sub
            end
        end
    end

    return positions
end

# --------------------------------------------------------------------------- #
# High-level entry point (time-dependent mode with target positions)
# --------------------------------------------------------------------------- #

function run_cpo_at_locations(
    target_positions::Vector{Vector{Float64}},
    snapshots::Vector{LaMEMSnapshot},
    drex_params::Dict{Symbol,Any};
    n_grains::Int            = 1000,
    seed::Union{Int,Nothing} = 42,
    n_substeps::Int          = 5,
    fabric::MineralFabric    = olivine_A,
)
    @info "Backtracking $(length(target_positions)) positions through $(length(snapshots)-1) intervals..."
    source_positions = backtrack_positions(target_positions, snapshots; n_substeps)

    @info "Creating CPO tracers at source positions..."
    tracers = create_tracers(source_positions;
        n_grains, seed,
        phase_assemblage = drex_params[:phase_assemblage],
        fabric,
    )

    @info "Evolving CPO forward..."
    evolve_cpo!(tracers, drex_params, snapshots; advect = true, n_substeps)

    return tracers
end

# --------------------------------------------------------------------------- #
# Paraview output helper
# --------------------------------------------------------------------------- #

function _write_paraview(tracers, vtk_prefix, n_saved, get_time)
    n_tracers = length(tracers)
    pvd = paraview_collection(vtk_prefix)

    for k in 1:n_saved
        t_k           = get_time(k)
        positions_out = zeros(3, n_tracers)
        fast_axes     = zeros(3, n_tracers)
        m_indices     = zeros(n_tracers)
        strains       = zeros(n_tracers)

        for (i, tr) in enumerate(tracers)
            positions_out[:, i] .= tr.positions[k]
            ol                   = tr.minerals[1]
            ori                  = ol.orientations[k]
            fast_axes[:, i]     .= bingham_average(ori; axis = "a")
            m_indices[i]         = misorientation_index(ori, orthorhombic)
            F                    = tr.deformation_gradient
            strains[i]           = sqrt(maximum(svdvals(F * F'))) - 1
        end

        fname = "$(vtk_prefix)_$(lpad(k-1, 4, '0'))"
        vtk   = vtk_grid(fname, positions_out,
                         MeshCell[MeshCell(VTKCellTypes.VTK_VERTEX, [i])
                                  for i in 1:n_tracers])
        vtk["fast_axis"]     = fast_axes
        vtk["m_index"]       = m_indices
        vtk["finite_strain"] = strains
        pvd[t_k] = vtk
        vtk_save(vtk)
    end
    vtk_save(pvd)
end

# --------------------------------------------------------------------------- #
# All-in-one entry point
# --------------------------------------------------------------------------- #

"""
    compute_cpo_from_lamem(sim_name, sim_dir; kwargs...)
    → (tracers, snapshots_used)

Load a LaMEM simulation, (optionally backtrack positions), evolve CPO, and
write a Paraview time-series — all in one call.

# Positional arguments
- `sim_name` — LaMEM `out_file_name`
- `sim_dir`  — directory containing the LaMEM `.pvd` file

# Position specification (exactly one required)
| Keyword              | Description                                                          |
|----------------------|----------------------------------------------------------------------|
| `target_positions`   | [x,y,z] km where CPO is desired at the **last used snapshot**; positions are backtracked to find the material source at the first used snapshot |
| `initial_positions`  | [x,y,z] km of material at the **first used snapshot**; CPO is evolved forward from here directly without backtracking |

# Mode
- **Time-dependent** (default): reads the full snapshot sequence; velocity gradients are interpolated in time between consecutive snapshots.
- **Steady-state**: set `steady_state_step` and `steady_state_duration` to use a single snapshot's velocity field for the entire integration, assuming the flow is constant.

# Common keyword arguments
| Keyword                 | Default                   | Description                                        |
|-------------------------|---------------------------|----------------------------------------------------|
| `output_dir`            | `"\$(sim_name)_tracers"` | Directory for Paraview output                      |
| `skip_initial_steps`    | `5`                       | Drop first N snapshots (spin-up transients)        |
| `start_step`            | `nothing` (= 1)           | First snapshot index within usable window          |
| `end_step`              | `nothing` (= last)        | Last  snapshot index within usable window          |
| `drex_params`           | `default_params()`        | DRex parameter dict                                |
| `n_grains`              | `1000`                    | Grains per mineral per tracer                      |
| `seed`                  | `42`                      | RNG seed                                           |
| `n_substeps`            | `5`                       | Advection / CPO sub-steps per output step          |
| `fabric`                | `olivine_A`               | Olivine fabric type                                |
| `vel_grad_field`        | `:vel_gr_tensor`          | Field name in LaMEM output                         |

# Steady-state-only keyword arguments
| Keyword                 | Default    | Description                                               |
|-------------------------|------------|-----------------------------------------------------------|
| `steady_state_step`     | `nothing`  | Index into the usable snapshot window to use as reference |
| `steady_state_duration` | `nothing`  | Total integration time (Myr)                              |
| `steady_state_n_steps`  | `100`      | Number of output steps                                    |

# Returns
`(tracers::Vector{CPOTracer}, snaps::Vector{LaMEMSnapshot})`
"""
function compute_cpo_from_lamem(
    sim_name::String,
    sim_dir::String;
    # Position specification (exactly one required)
    target_positions::Union{Vector{Vector{Float64}}, Nothing}  = nothing,
    initial_positions::Union{Vector{Vector{Float64}}, Nothing} = nothing,
    # Snapshot window
    output_dir::String              = "$(sim_name)_tracers",
    skip_initial_steps::Int         = 5,
    start_step::Union{Int,Nothing}  = nothing,
    end_step::Union{Int,Nothing}    = nothing,
    # Steady-state mode
    steady_state_step::Union{Int,Nothing}         = nothing,
    steady_state_duration::Union{Float64,Nothing} = nothing,
    steady_state_n_steps::Int                     = 100,
    # DRex / tracer settings
    drex_params::Dict{Symbol,Any}   = default_params(),
    n_grains::Int                   = 1000,
    seed::Union{Int,Nothing}        = 42,
    n_substeps::Int                 = 5,
    fabric::MineralFabric           = olivine_A,
    vel_grad_field::Symbol          = :vel_gr_tensor,
)
    # ── Validate inputs ──────────────────────────────────────────────────────
    if isnothing(target_positions) == isnothing(initial_positions)
        error("Exactly one of `target_positions` or `initial_positions` must be provided.")
    end

    steady_state = !isnothing(steady_state_step) || !isnothing(steady_state_duration)
    if steady_state && (isnothing(steady_state_step) || isnothing(steady_state_duration))
        error("Both `steady_state_step` and `steady_state_duration` must be provided for steady-state mode.")
    end

    positions        = something(target_positions, initial_positions)
    use_backtracking = !isnothing(target_positions)

    # ── 1. Load snapshots ────────────────────────────────────────────────────
    @info "Loading LaMEM snapshots from $(sim_dir)/…"
    all_snaps = load_snapshots(sim_name, sim_dir;
                               vel_grad_field, load_velocity = true)
    @info "  $(length(all_snaps)) snapshots, t = $(all_snaps[1].time) – $(all_snaps[end].time) Myr"

    usable  = all_snaps[(skip_initial_steps + 1):end]
    i_start = something(start_step, 1)
    i_end   = something(end_step,   length(usable))
    snaps   = usable[i_start:i_end]
    @info "  Using $(length(snaps)) snapshots (indices $(skip_initial_steps + i_start)–$(skip_initial_steps + i_end) of $(length(all_snaps)))"

    # ── 2. Backtrack / seed / evolve ─────────────────────────────────────────
    if steady_state
        ref_snap = snaps[steady_state_step]
        dur      = Float64(steady_state_duration)

        if use_backtracking
            @info "Backtracking $(length(positions)) positions (steady-state, $(n_substeps) substeps, $(dur) Myr)..."
            source_positions = _backtrack_steady(positions, ref_snap, dur, n_substeps)
        else
            source_positions = positions
        end

        @info "Creating CPO tracers..."
        tracers = create_tracers(source_positions;
            n_grains, seed,
            phase_assemblage = drex_params[:phase_assemblage],
            fabric,
        )

        @info "Evolving CPO (steady-state, $(steady_state_n_steps) steps, $(dur) Myr)..."
        _evolve_cpo_steady!(tracers, drex_params, ref_snap, dur, steady_state_n_steps;
                            n_substeps, advect = true)

        # Time axis for Paraview: starts at the reference snapshot time
        t_base  = ref_snap.time
        dt_step = dur / steady_state_n_steps
        get_time = k -> t_base + (k - 1) * dt_step

    else
        # Time-dependent mode
        if use_backtracking
            @info "Backtracking $(length(positions)) positions through $(length(snaps)-1) intervals..."
            source_positions = backtrack_positions(positions, snaps; n_substeps)
        else
            source_positions = positions
        end

        @info "Creating CPO tracers..."
        tracers = create_tracers(source_positions;
            n_grains, seed,
            phase_assemblage = drex_params[:phase_assemblage],
            fabric,
        )

        @info "Evolving CPO forward..."
        evolve_cpo!(tracers, drex_params, snaps; advect = true, n_substeps)

        # Time axis: snapshot times
        get_time = k -> snaps[k].time
    end

    @info "CPO evolution complete."

    # ── 3. Paraview time-series output ───────────────────────────────────────
    mkpath(output_dir)
    vtk_prefix = joinpath(output_dir, "cpo_tracers")
    @info "Writing Paraview time-series → $(output_dir)/ …"

    n_saved = length(tracers[1].minerals[1].orientations)
    _write_paraview(tracers, vtk_prefix, n_saved, get_time)
    @info "Paraview output written: $(vtk_prefix).pvd + $(n_saved) .vtp files"

    m_final = [misorientation_index(tr.minerals[1].orientations[end], orthorhombic)
               for tr in tracers]
    @info "M-index range: $(round(minimum(m_final); sigdigits=3)) – $(round(maximum(m_final); sigdigits=3))"

    return tracers, snaps
end

end # module DRexLaMEMExt
