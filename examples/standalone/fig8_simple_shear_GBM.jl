# Reproduce Figure 8 of Bilton et al. (2025):
#
# A-type olivine test demonstrating the effect of the grain boundary mobility
# (M*) parameter on CPO evolution in simple shear (L₂₁ = 2D₀).
#
# An ensemble of N_SEEDS runs is performed for each M* value, each with a
# different random initial grain orientation state.  The CPO direction is
# calculated at every timestep using the hexagonal symmetry axis of the
# Voigt-averaged elastic tensor (via Browaeys & Chevrot 2004 decomposition).
# Mean angles (±1 std. dev.) are plotted against accumulated strain ε.
#
# The FSE long-axis angle is computed analytically from eq. (29).
#
# Experimental data from Skemer et al. (2016) compilation are overlaid.
#
# Run with: julia +1.12 --project=. --threads=auto examples/standalone/fig8_simple_shear_GBM.jl

push!(LOAD_PATH, "@v#.#")  # stacked env for CairoMakie

using DRex
using LinearAlgebra
using Statistics: mean, std
using CairoMakie
using Base.Threads: @threads, nthreads

# ── Tuneable constants ───────────────────────────────────────────────────────
const N_GRAINS       = 5000
const N_TIMESTAMPS   = 61      # 60 strain intervals → Δε = 0.05
const STRAIN_RATE    = 1e-4
const T_END          = 3e4     # D₀ t_max = 3  →  ε_max = 3.0
const GBM_MOBILITIES = [0, 10, 50, 125]  # M* values shown in the figure
const N_SEEDS        = 41      # number of ensemble members (paper uses 41)

# 41 seeds that produce initial SCCS ≈ 45° from X (from hexaxis_nearX45_seeds.scsv)
const SEEDS = [
    120,  722, 1436, 1540, 2094, 2206, 2238, 2756, 3232, 3809,
    3868, 3941, 3992, 4140, 4192, 4283, 4584, 4952, 5062, 5346,
    6880, 6925, 7018, 7035, 7229, 7433, 7439, 7451, 7468, 8020,
    8110, 8194, 8369, 8392, 8427, 8543, 8564, 8729, 8930, 9041, 9922,
]

# ── Experimental data (Skemer et al. 2016 compilation, A-type olivine) ───────
# shear_strain stored in percent → tensorial strain ε = shear_strain / 200
const EXP_ZK_1473 = (;  # Zhang & Karato, 1995 (1473 K)
    strain = [17, 30, 45, 65, 110] ./ 200,
    angle  = [43.0, 37.0, 38.0, 24.0, 20.0],
)
const EXP_ZK_1573 = (;  # Zhang & Karato, 1995 (1573 K)
    strain = [11, 7, 65, 58, 100, 115, 150] ./ 200,
    angle  = [36.0, 28.0, 18.0, 10.0, 5.0, 10.0, 0.0],
)
const EXP_SKEMER_2011 = (;  # Skemer et al., 2011 (1500 K)
    strain = [120, 180, 350] ./ 200,
    angle  = [55.0, 53.0, 45.0],
)
const EXP_HANSEN_2014 = (;  # Hansen et al., 2014 (1473 K) — A-type only
    strain = [590, 680, 680, 760, 820, 870, 880, 1020, 1060, 1090] ./ 200,
    angle  = abs.([0.6, -5.5, -4.3, -8.4, -1.9, -0.9, -8.9, 2.8, -2.0, -1.5]),
)
const EXP_WARREN_2008 = (;  # Warren et al., 2008
    strain = [0, 65, 118, 131, 258, 386, 386, 525, 168] ./ 200,
    angle  = [62.0, 37.0, 49.0, 61.0, 4.0, 11.0, 0.0, 1.0, 33.0],
)
const EXP_WEBBER_2010 = (;  # Webber et al., 2010
    strain = [0, 25, 100, 130, 168, 330, 330] ./ 200,
    angle  = [55.0, 35.0, 47.0, 29.0, 37.0, 47.0, 45.0],
)
const EXP_HW_2015 = (;  # Hansen & Warren, 2015
    strain = [32, 32, 81, 81, 118, 118, 258, 258, 286, 286, 337, 337, 386, 386, 525, 525] ./ 200,
    angle  = [48.0, 51.0, 44.0, 35.0, 35.0, 40.0, 1.0, 15.0, 25.0, 28.0, 28.0, 39.0, 1.0, 8.0, 4.0, 11.0],
)

# ── Derived constants ────────────────────────────────────────────────────────
const timestamps = range(0, T_END, length=N_TIMESTAMPS)
const strains    = collect(timestamps) .* STRAIN_RATE   # ε = D₀ t
const shear_direction = Float64[0, 1, 0]

# ── Helper: Voigt average at a specific stored timestep ──────────────────────
function voigt_average_step(mineral::Mineral, step::Int, C_tensor)
    orientations = mineral.orientations[step]
    fractions = mineral.fractions[step]
    n = mineral.n_grains
    C_avg = zeros(3, 3, 3, 3)
    @inbounds for g in 1:n
        R = zeros(3, 3)
        for i in 1:3, j in 1:3
            R[i,j] = orientations[g, i, j]
        end
        C_rot = rotate_tensor(C_tensor, R')
        for i in 1:3, j in 1:3, k in 1:3, l in 1:3
            C_avg[i,j,k,l] += fractions[g] * C_rot[i,j,k,l]
        end
    end
    return elastic_tensor_to_voigt(C_avg)
end

# ── Helper: extract CPO angles (hex axis vs shear direction) per timestep ────
function cpo_angles(mineral::Mineral)
    st = StiffnessTensors()
    C_ref = DRex.get_stiffness(st, mineral.phase)
    C_tensor = voigt_to_elastic_tensor(C_ref)
    n_steps = length(mineral.orientations)
    voigt_mats = [voigt_average_step(mineral, s, C_tensor) for s in 1:n_steps]
    hex_axes = elasticity_components(voigt_mats)["hexagonal_axis"]
    return [smallest_angle(hex_axes[s, :], shear_direction) for s in 1:n_steps]
end

# ── FSE angle (analytical, eq. 29) ──────────────────────────────────────────
θ_fse = [rad2deg(atan(sqrt(ε^2 + 1) + ε)) for ε in strains]
θ_fse_from_shear = 90.0 .- θ_fse   # angle between FSE long axis and shear direction

# ── Main simulation loop (multithreaded over seeds) ─────────────────────────
n_M = length(GBM_MOBILITIES)
seeds_to_use = SEEDS[1:min(N_SEEDS, length(SEEDS))]
n_seeds = length(seeds_to_use)

# angles[m, s, t] = angle for M* index m, seed index s, timestep t
all_angles = zeros(n_M, n_seeds, N_TIMESTAMPS)

println("Running ensemble: $(n_M) M* values × $(n_seeds) seeds × " *
        "$(N_TIMESTAMPS-1) time steps × $(N_GRAINS) grains  [$(nthreads()) threads]")
println("=" ^ 70)

t_total = time()
print_lock = ReentrantLock()

for (m, Mstar) in enumerate(GBM_MOBILITIES)
    params = default_params()
    params[:number_of_grains]      = N_GRAINS
    params[:gbm_mobility]          = Float64(Mstar)
    params[:gbs_threshold]         = 0.0
    params[:nucleation_efficiency] = 5.0
    params[:stress_exponent]       = 1.5
    params[:deformation_exponent]  = 3.5

    _, get_L = simple_shear_2d("Y", "X", STRAIN_RATE)

    @threads for s in 1:n_seeds
        seed = seeds_to_use[s]
        t0 = time()

        mineral = Mineral(
            phase   = olivine,
            fabric  = olivine_A,
            regime  = matrix_dislocation,
            n_grains = N_GRAINS,
            seed    = seed,
        )
        deformation_gradient = Matrix{Float64}(I, 3, 3)

        for t in 2:N_TIMESTAMPS
            deformation_gradient = update_orientations!(
                mineral, params, deformation_gradient, get_L,
                (timestamps[t-1], timestamps[t], t -> zeros(3)),
            )
        end

        all_angles[m, s, :] .= cpo_angles(mineral)

        elapsed = time() - t0
        Base.@lock print_lock println(
            "M*=$(lpad(Mstar, 3))  seed $(lpad(s, 2))/$(n_seeds) " *
            "(#$(seed))  $(round(elapsed, digits=1))s")
    end
end

elapsed_total = time() - t_total
println("=" ^ 70)
println("Total elapsed: $(round(elapsed_total, digits=1))s")

# ── Ensemble statistics ──────────────────────────────────────────────────────
mean_angles = dropdims(mean(all_angles, dims=2), dims=2)   # (n_M, N_TIMESTAMPS)
std_angles  = dropdims(std(all_angles, dims=2), dims=2)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig = Figure(size=(900, 450))
ax = Axis(fig[1, 1],
    xlabel = "Strain (ε)",
    ylabel = "θ̄ ∈ [0, 90]°",
    xticks = 0:0.5:3.0,
    yticks = 0:10:80,
)
xlims!(ax, 0, 3)
ylims!(ax, 0, 80)

# Colour palette & markers for M* curves
Mstar_colors  = [:royalblue, :orange, :green3, :firebrick4]
Mstar_markers = [:xcross, :star5, :star4, :circle]

for (m, Mstar) in enumerate(GBM_MOBILITIES)
    c = Mstar_colors[m]
    mk = Mstar_markers[m]
    μ = mean_angles[m, :]
    σ = std_angles[m, :]

    # Shaded ±1σ band
    band!(ax, strains, max.(μ .- σ, 0.0), min.(μ .+ σ, 90.0), color=(c, 0.15))
    # Mean line with markers (dashed)
    scatterlines!(ax, strains, μ, color=c,
        markersize=7, marker=mk, linewidth=1.5, linestyle=:dash,
        label="M* = $Mstar")
end

# FSE long-axis angle (dashed, purple)
lines!(ax, strains, θ_fse_from_shear, linestyle=:dash, color=:mediumpurple, linewidth=2,
    label="FSE")

# ── Experimental data (black markers, A-type olivine) ────────────────────────
scatter!(ax, EXP_ZK_1473.strain, EXP_ZK_1473.angle,
    marker=:dtriangle, markersize=10, color=:transparent,
    strokecolor=:black, strokewidth=1.5,
    label="Zhang & Karato, 1995 (1473 K)")
scatter!(ax, EXP_ZK_1573.strain, EXP_ZK_1573.angle,
    marker=:utriangle, markersize=10, color=:transparent,
    strokecolor=:black, strokewidth=1.5,
    label="Zhang & Karato, 1995 (1573 K)")
scatter!(ax, EXP_SKEMER_2011.strain, EXP_SKEMER_2011.angle,
    marker=:circle, markersize=9, color=:transparent,
    strokecolor=:black, strokewidth=1.5,
    label="Skemer et al., 2011 (1500 K)")
scatter!(ax, EXP_HANSEN_2014.strain, EXP_HANSEN_2014.angle,
    marker=:rect, markersize=9, color=:transparent,
    strokecolor=:black, strokewidth=1.5,
    label="Hansen et al., 2014 (1473 K)")
scatter!(ax, EXP_WARREN_2008.strain, EXP_WARREN_2008.angle,
    marker=:dtriangle, markersize=10, color=:black,
    label="Warren et al., 2008")
scatter!(ax, EXP_WEBBER_2010.strain, EXP_WEBBER_2010.angle,
    marker=:circle, markersize=9, color=:black,
    label="Webber et al., 2010")
scatter!(ax, EXP_HW_2015.strain, EXP_HW_2015.angle,
    marker=:rect, markersize=9, color=:black,
    label="Hansen & Warren, 2015")

Legend(fig[1, 2], ax, framevisible=true, padding=(5, 5, 5, 5), rowgap=2, labelsize=11)

outpath = joinpath(@__DIR__, "fig8_simple_shear_GBM.png")
save(outpath, fig, px_per_unit=2)
println("Saved: $outpath")
