#!/usr/bin/env julia
#
# Corner-flow CPO example reproducing Fig. 10 of
# Bilton et al. (2025), Geophys. J. Int., 241(1), 35–57.
#
# Four panels:
#   (a) M*=10  – CPO bars (Bingham avg of olivine a-axis) scaled by M-index
#   (b) M*=125 – same
#   (c) M*=10  – M-index vs accumulated strain
#   (d) M*=125 – same
#
# Uses GLMakie for plotting.  Run from the examples/standalone directory:
#   julia --project=. cornerflow_simple.jl

using LinearAlgebra
using DRex
using GLMakie

# ── Simulation parameters ────────────────────────────────────────────────────

const PLATE_SPEED     = 2.0 / (100 * 365 * 86400)   # 2 cm/yr → m/s
const DOMAIN_HEIGHT   = 2e5                           # 200 km (olivine-spinel transition)
const DOMAIN_WIDTH    = 1e6                           # 1000 km
const N_TIMESTEPS     = 50
const PATHLINE_ENDS   = (-0.1, -0.3, -0.54, -0.78)   # fraction of domain height
const DOMAIN_COORDS   = ("X", "Z")
const MAX_STRAIN      = 10.0
const GBM_MOBILITIES  = (10, 125)                     # M* values for the two columns
const OUT_FIGURE      = "cornerflow2d_simple_example.png"

const MIN_COORDS = [0.0, 0.0, -DOMAIN_HEIGHT]
const MAX_COORDS = [DOMAIN_WIDTH, 0.0, 0.0]

# ── Velocity field ───────────────────────────────────────────────────────────

f_velocity, f_velocity_grad = corner_2d(DOMAIN_COORDS..., PLATE_SPEED)

final_locations = [
    [DOMAIN_WIDTH, 0.0, z * DOMAIN_HEIGHT] for z in PATHLINE_ENDS
]

# ── Single-pathline CPO solver ───────────────────────────────────────────────

function run_pathline(params, f_velocity, f_velocity_grad,
                      min_coords, max_coords, final_location)
    olA = Mineral(
        phase   = olivine,
        fabric  = olivine_A,
        regime  = matrix_dislocation,
        n_grains = params[:number_of_grains],
    )
    ens = Mineral(
        phase   = enstatite,
        fabric  = enstatite_AB,
        regime  = matrix_dislocation,
        n_grains = params[:number_of_grains],
    )

    timestamps, f_position = get_pathline(
        final_location, f_velocity, f_velocity_grad,
        min_coords, max_coords;
        max_strain = MAX_STRAIN,
        regular_steps = N_TIMESTEPS,
    )

    positions = [collect(f_position(t)) for t in timestamps]
    velocity_gradients = [f_velocity_grad(NaN, x) for x in positions]

    deformation_gradient = Matrix{Float64}(I, 3, 3)
    strains = zeros(length(timestamps))

    Mstar = params[:gbm_mobility]
    for i in 2:length(timestamps)
        strains[i] = strains[i-1] + strain_increment(
            timestamps[i] - timestamps[i-1], velocity_gradients[i]
        )
        println("  M*=$Mstar path z=$(round(final_location[3]/DOMAIN_HEIGHT; digits=2)); " *
                "step $(i-1)/$N_TIMESTEPS (ε = $(round(strains[i]; digits=2)))")
        deformation_gradient = update_all!(
            [olA, ens], params, deformation_gradient,
            f_velocity_grad,
            (timestamps[i-1], timestamps[i], f_position),
        )
    end
    return timestamps, positions, strains, olA, ens
end

# ── Post-processing: M-index and Bingham average directions ──────────────────

function compute_diagnostics(olA::Mineral, seed::Int=1)
    orient_resampled, _ = resample_orientations(
        olA.orientations, olA.fractions; seed=seed,
    )
    n = length(orient_resampled)
    m_indices = zeros(n)
    directions = zeros(n, 3)
    primary_axis = OLIVINE_PRIMARY_AXIS[olA.fabric]
    for idx in 1:n
        m_indices[idx] = misorientation_index(orient_resampled[idx], orthorhombic)
        directions[idx, :] .= bingham_average(orient_resampled[idx]; axis=primary_axis)
    end
    return m_indices, directions
end

# ── Run all pathlines for both M* values ──────────────────────────────────────

cases = Dict{Int, Dict{Symbol, Vector}}()

for (mi, Mstar) in enumerate(GBM_MOBILITIES)
    params = default_params()
    params[:phase_assemblage] = [olivine, enstatite]
    params[:phase_fractions]  = [0.7, 0.3]
    params[:gbm_mobility]     = Float64(Mstar)
    params[:number_of_grains] = 5000

    case = Dict{Symbol,Vector}(
        :strains    => Vector{Vector{Float64}}(),
        :positions  => Vector{Vector{Vector{Float64}}}(),
        :m_indices  => Vector{Vector{Float64}}(),
        :directions => Vector{Matrix{Float64}}(),
    )

    for (i, final_loc) in enumerate(final_locations)
        println("M*=$Mstar  Pathline $i / $(length(final_locations))")
        _, positions, strains, olA, _ = run_pathline(
            params, f_velocity, f_velocity_grad, MIN_COORDS, MAX_COORDS, final_loc,
        )
        m_indices, directions = compute_diagnostics(olA)
        push!(case[:strains], strains)
        push!(case[:positions], positions)
        push!(case[:m_indices], m_indices)
        push!(case[:directions], directions)
    end
    cases[mi] = case
end

# ── Plotting with GLMakie (4-panel layout matching Fig. 10) ──────────────────

fig = Figure(size = (1200, 1100))

# Shared strain color range
strain_min = 0.0
strain_max = maximum(vcat(maximum.(cases[1][:strains]), maximum.(cases[2][:strains])))
cmap = :batlow
cmap_rev = Reverse(:batlow)
cgrad_rev = cgrad(:batlow, rev=true)

markers_list = [:rect, :circle, :utriangle, :star5]
pathline_labels = [
    "zf = $(round(z * DOMAIN_HEIGHT / 1e3; digits=0)) km"
    for z in PATHLINE_ENDS
]
to_km(x) = x / 1e3

# ── Helper: draw domain panel (a) or (b) ─────────────────────────────────────

function draw_domain_panel!(fig_pos, panel_label, case)
    ax = Axis(fig_pos;
        xlabel = "x (km)",
        ylabel = "z (km)",
        title  = panel_label,
        aspect = DataAspect(),
    )

    # Velocity arrows on a coarse grid
    nx, nz = 50, 10
    xs = range(1e3, DOMAIN_WIDTH, length=nx)
    zs = range(-DOMAIN_HEIGHT, -1e3, length=nz)
    ux = zeros(nx, nz)
    uz = zeros(nx, nz)
    for (ix, x) in enumerate(xs), (iz, z) in enumerate(zs)
        v = f_velocity(NaN, [x, 0.0, z])
        ux[ix, iz] = v[1]
        uz[ix, iz] = v[3]
    end
    speed = sqrt.(ux .^ 2 .+ uz .^ 2)
    speed[speed .== 0] .= NaN
    ux_norm = ux ./ speed
    uz_norm = uz ./ speed
    arrows!(ax,
        vec(collect(xs) * ones(nz)') ./ 1e3,
        vec(ones(nx) * collect(zs)') ./ 1e3,
        vec(ux_norm), vec(uz_norm);
        arrowsize = 8, lengthscale = 12,
        color = (:black, 0.5))

    for i in eachindex(final_locations)
        pos = case[:positions][i]
        px = [to_km(p[1]) for p in pos]
        pz = [to_km(p[3]) for p in pos]
        strains   = case[:strains][i]
        m_idx     = case[:m_indices][i]
        dirs      = case[:directions][i]

        # Pathline curve
        lines!(ax, px, pz; color = :grey40, linewidth = 0.5)

        # CPO direction bars scaled by M-index
        scale = 100.0  # km half-length at M=1
        for j in eachindex(px)
            m = m_idx[j]
            m < 0.01 && continue
            hx = dirs[j, 1] * m * scale / 2
            hz = dirs[j, 3] * m * scale / 2
            t = clamp((strains[j] - strain_min) / (strain_max - strain_min), 0.0, 1.0)
            c = cgrad_rev[t]
            lines!(ax, [px[j] - hx, px[j] + hx], [pz[j] - hz, pz[j] + hz];
                color = c, linewidth = 1.5)
        end

        # Small markers colored by strain
        scatter!(ax, px, pz;
            color      = strains,
            colormap   = cmap_rev,
            colorrange = (strain_min, strain_max),
            marker     = markers_list[i],
            markersize = 5,
        )
    end

    xlims!(ax, 0, to_km(DOMAIN_WIDTH) + 50)
    ylims!(ax, to_km(-DOMAIN_HEIGHT) - 10, 10)
    return ax
end

# ── Helper: draw M-index vs strain panel (c) or (d) ──────────────────────────

function draw_strength_panel!(fig_pos, panel_label, case)
    ax = Axis(fig_pos;
        xlabel = "Strain (ε)",
        ylabel = "CPO strength (M-index)",
        title  = panel_label,
    )

    for i in eachindex(final_locations)
        strains = case[:strains][i]
        m_idx   = case[:m_indices][i]
        scatter!(ax, strains, m_idx;
            color      = strains,
            colormap   = cmap_rev,
            colorrange = (strain_min, strain_max),
            marker     = markers_list[i],
            markersize = 8,
            label      = pathline_labels[i],
        )
    end

    ylims!(ax, 0, 0.6)
    axislegend(ax; position = :lt, labelsize = 10, framevisible = false)
    return ax
end

# ── Assemble figure ──────────────────────────────────────────────────────────
# Row 1: (a) M*=10, Row 2: (b) M*=125, Row 3: (c) and (d) side by side

draw_domain_panel!(fig[1, 1:2], "(a) M* = 10",  cases[1])
draw_domain_panel!(fig[2, 1:2], "(b) M* = 125", cases[2])
draw_strength_panel!(fig[3, 1], "(c) M* = 10",  cases[1])
draw_strength_panel!(fig[3, 2], "(d) M* = 125", cases[2])

Colorbar(fig[0, 1:2]; colormap = cmap_rev, colorrange = (strain_min, strain_max),
    label = "Strain (ε)", vertical = false, flipaxis = false,
    width = Relative(0.5), height = 15)

save(OUT_FIGURE, fig; px_per_unit = 2)
println("\nFigure saved to $OUT_FIGURE")
