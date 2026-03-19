#=
# postprocess_cpo.jl
#
# CPO post-processing for any LaMEM simulation run with:
#   out_vel_gr_tensor = 1
#   out_velocity      = 1
#
# Usage
# -----
#   cd examples/LaMEM
#   julia --project=. -t auto postprocess_cpo.jl
#
# '-t auto' uses all available CPU cores (one thread per tracer — scales linearly).
#
# Adapt the USER SETTINGS block; the rest of the script needs no editing.
#
# Paraview output: File → Open → <output_dir>/cpo_tracers.pvd
#   Useful filters: Glyph (Arrow by 'fast_axis', scale by 'm_index'),
#                   Threshold on 'm_index'
#
# Units: DRexLaMEMExt converts LaMEM output to consistent km / Myr / (1/Myr).
=#

using LaMEM, GeophysicalModelGenerator   # triggers DRexLaMEMExt automatically
using DRex
using CairoMakie
using LinearAlgebra

# ============================================================================
# USER SETTINGS
# ============================================================================

const SIM_NAME = "Subduction_3D_CPO"
const SIM_DIR  = "Subduction_3D_CPO"

# Target positions [x, y, z] in km — where you want CPO at the END of the run.
# The script backtracks these through the velocity field, then evolves CPO forward.
target_positions = let
    pts = Vector{Float64}[]
    for x in -1500.0:150.0:-300.0        # km along-trench  (step 150 → 9 values)
        for y in 0.0:250.0:1500.0         # km across-strike (step 250 → 7 values)
            for z in [-150.0, -300.0]     # km depth         (2 values)
                push!(pts, Float64[x, y, z])
            end
        end
    end
    pts
end

# DRex parameters (edit as needed)
drex_params = let p = default_params()
    p[:phase_assemblage] = [olivine, enstatite]
    p[:phase_fractions]  = [0.7, 0.3]
    p[:gbm_mobility]     = 10.0
    p
end

# ============================================================================
# Run CPO (loads snapshots, backtracks, evolves, writes Paraview output)
# ============================================================================

tracers, snaps = compute_cpo_from_lamem(
    SIM_NAME, SIM_DIR;
    target_positions   = target_positions,   # backtrack → forward (use initial_positions to skip backtracking)
    output_dir         = "$(SIM_NAME)_tracers",
    skip_initial_steps = 5,
    # start_step = 50,   # optional: start from a later snapshot
    # end_step   = 150,  # optional: stop at an earlier snapshot
    # --- steady-state mode (comment out the two lines above and uncomment below) ---
    # steady_state_step     = 100,   # snapshot index to use as reference velocity field
    # steady_state_duration = 10.0,  # Myr to integrate
    # steady_state_n_steps  = 200,   # output steps
    drex_params        = drex_params,
    n_grains           = 1000,
    seed               = 42,
    n_substeps         = 5,
    fabric             = olivine_A,
)

# ============================================================================
# Pole-figure plots (final state)  — optional, edit or remove as needed
# ============================================================================

@info "Plotting olivine [100] pole figures…"

m_indices_final = [misorientation_index(tr.minerals[1].orientations[end], orthorhombic)
                   for tr in tracers]

n_tracers = length(tracers)
n_cols    = min(n_tracers, 6)
n_rows    = cld(n_tracers, n_cols)
fig       = Figure(size = (n_cols * 220, n_rows * 260))

Label(fig[0, :], "Olivine [100] pole figures at target positions (x km, z km)";
      fontsize = 13, font = :bold)

θ_circle = range(0, 2π; length = 200)

for (idx, tracer) in enumerate(tracers)
    row = cld(idx, n_cols)
    col = mod1(idx, n_cols)

    ol   = tracer.minerals[1]
    ori  = ol.orientations[end]
    frac = ol.fractions[end]

    ori_r_list, _ = resample_orientations([ori], [frac])
    ori_r = ori_r_list[1]

    xvals, yvals, zvals = poles(ori_r; hkl = [1, 0, 0], ref_axes = "xz")
    x2d, y2d = lambert_equal_area(xvals, yvals, zvals)

    mi    = round(m_indices_final[idx]; sigdigits = 2)
    x_km  = round(Int, target_positions[idx][1])
    z_km  = round(Int, target_positions[idx][3])

    ax = Axis(fig[row, col];
              aspect    = DataAspect(),
              title     = "(x=$x_km, z=$z_km)\nM=$(mi)",
              titlesize = 10)
    hidedecorations!(ax)
    lines!(ax, cos.(θ_circle), sin.(θ_circle); color = :black, linewidth = 0.8)
    scatter!(ax, x2d, y2d; markersize = 2, color = (:steelblue, 0.4))
end

outpng = joinpath("$(SIM_NAME)_tracers", "cpo_polefigures.png")
save(outpng, fig; px_per_unit = 2)
@info "Pole figures saved → $outpng"
