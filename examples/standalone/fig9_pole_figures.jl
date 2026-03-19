# Reproduce Figure 8 of Bilton et al. (2025):
#
# [100] pole figures (Lambert equal-area projections) for A-type olivine in
# simple shear at different strains and grain-boundary mobility (M*) values.
#
# Layout: 3 rows (M* = 0, 50, 200) × 4 columns (ε = 0.2, 0.4, 0.6, 0.8).
# Each panel shows the [100] axis orientations of 5000 grains projected onto
# the X-Y plane, with marker opacity scaled by the grain volume fraction.
#
# Run with: julia +1.12 --project=. --threads=auto examples/standalone/fig9_pole_figures.jl

push!(LOAD_PATH, "@v#.#")  # stacked env for CairoMakie

using DRex
using LinearAlgebra
using CairoMakie
using Base.Threads: @threads, nthreads

# ── Parameters ───────────────────────────────────────────────────────────────
const N_GRAINS       = 5000
const STRAIN_RATE    = 1e-4
const GBM_MOBILITIES = [0, 50, 200]         # M* values (rows)
const TARGET_STRAINS = [0.2, 0.4, 0.6, 0.8] # ε values (columns)
const SEED           = 8816                  # single representative seed

# Derived: timestamps that hit the target strains exactly
const T_MAX    = maximum(TARGET_STRAINS) / STRAIN_RATE   # ε = D₀ t
const N_STEPS  = 80   # enough resolution; we pick the closest timestep
const timestamps = range(0, T_MAX, length=N_STEPS + 1)
const strains    = collect(timestamps) .* STRAIN_RATE

# Map each target strain to the closest timestep index
const target_indices = [argmin(abs.(strains .- ε)) for ε in TARGET_STRAINS]

# ── Plotting helper: draw a single pole figure ──────────────────────────────
"""
    plot_polefigure!(ax, orientations, fractions; hkl, ref_axes)

Draw a [hkl] pole figure on `ax` using Lambert equal-area projection.

Orientations are resampled weighted by volume fractions (as in PyDRex),
so grains with larger volume appear proportionally more often.
`orientations` is (n_grains, 3, 3), `fractions` is (n_grains,).
"""
function plot_polefigure!(ax::Axis, orientations::Array{Float64,3},
                          fractions::Vector{Float64};
                          hkl::Vector{Int}=[1,0,0],
                          ref_axes::String="yx")
    # Resample orientations weighted by volume fractions
    resampled, _ = resample_orientations(
        [orientations], [fractions]; n_samples=size(orientations, 1), seed=1,
    )
    ori_r = resampled[1]

    x3, y3, z3 = poles(ori_r; ref_axes=ref_axes, hkl=hkl)
    px, py = lambert_equal_area(x3, y3, z3)

    # Circle outline
    θs = range(0, 2π, length=361)
    lines!(ax, cos.(θs), sin.(θs), color=:black, linewidth=1.2)

    # Cross-hairs
    lines!(ax, [-1, 1], [0, 0], color=:grey75, linewidth=0.8)
    lines!(ax, [0, 0], [-1, 1], color=:grey75, linewidth=0.8)

    # Scatter pole points (uniform alpha, density shows texture strength)
    scatter!(ax, px, py, color=(:black, 0.33), markersize=3, markerspace=:pixel)

    # Axis labels: horizontal=Y (shear direction), vertical=X (gradient direction)
    text!(ax, -0.85, 0.80, text="x", fontsize=12, color=:grey50)
    text!(ax, 0.60, -0.85, text="y", fontsize=12, color=:grey50)
end

# ── Run simulations ─────────────────────────────────────────────────────────
n_M = length(GBM_MOBILITIES)
n_cols = length(TARGET_STRAINS)

# Store the Mineral objects: minerals[m] for each M* value
minerals = Vector{Mineral}(undef, n_M)

println("Running simulations: $(n_M) M* values × $(N_STEPS) steps × $(N_GRAINS) grains  [$(nthreads()) threads]")
println("=" ^ 60)

@threads for m in 1:n_M
    Mstar = GBM_MOBILITIES[m]
    t0 = time()

    params = default_params()
    params[:number_of_grains]      = N_GRAINS
    params[:gbm_mobility]          = Float64(Mstar)
    params[:gbs_threshold]         = 0.0
    params[:nucleation_efficiency] = 5.0
    params[:stress_exponent]       = 1.5
    params[:deformation_exponent]  = 3.5

    _, get_L = simple_shear_2d("Y", "X", STRAIN_RATE)

    mineral = Mineral(
        phase    = olivine,
        fabric   = olivine_A,
        regime   = matrix_dislocation,
        n_grains = N_GRAINS,
        seed     = SEED,
    )
    deformation_gradient = Matrix{Float64}(I, 3, 3)

    for t in 2:(N_STEPS + 1)
        deformation_gradient = update_orientations!(
            mineral, params, deformation_gradient, get_L,
            (timestamps[t-1], timestamps[t], t -> zeros(3)),
        )
    end

    minerals[m] = mineral
    elapsed = time() - t0
    println("M* = $(lpad(Mstar, 3))  done in $(round(elapsed, digits=1))s")
end

println("=" ^ 60)

# ── Plot the figure ─────────────────────────────────────────────────────────
fig = Figure(size=(1000, 800), backgroundcolor=:white)

# Top label
Label(fig[0, 1:n_cols], "strain (ε)", fontsize=22, halign=:center)

for (col, ti) in enumerate(target_indices)
    # Column header: strain value
    Label(fig[1, col], "$(TARGET_STRAINS[col])", fontsize=20, halign=:center)
end

for (row, Mstar) in enumerate(GBM_MOBILITIES)
    # Row header: M* value
    Label(fig[row + 1, 0], "M* = $Mstar", fontsize=20, rotation=0, halign=:right)

    for (col, ti) in enumerate(target_indices)
        ax = Axis(fig[row + 1, col],
            aspect=DataAspect(),
            width=180, height=180,
        )
        hidedecorations!(ax)
        hidespines!(ax)
        xlims!(ax, -1.15, 1.15)
        ylims!(ax, -1.15, 1.15)

        ori  = minerals[row].orientations[ti]
        frac = minerals[row].fractions[ti]
        plot_polefigure!(ax, ori, frac)
    end
end

# Tighten layout
colgap!(fig.layout, 5)
rowgap!(fig.layout, 5)

outpath = joinpath(@__DIR__, "fig9_pole_figures.png")
save(outpath, fig, px_per_unit=3)
println("Saved: $outpath")
