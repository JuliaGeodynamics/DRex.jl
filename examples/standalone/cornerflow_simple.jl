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
#   julia --project=. -t auto cornerflow_simple.jl
#
# '-t auto' enables multi-threading: pathlines run in parallel across cores,
# and the per-grain D-Rex loop is also threaded within each pathline.
#
# Pass '--metal' to offload the per-grain kernel to the Metal GPU (Apple Silicon).
# Metal requires Float32, so only the Float32 run is executed in that mode.
#   julia --project=. cornerflow_simple.jl --metal

using LinearAlgebra
using DRex
using GLMakie
using KernelAbstractions

# ── Backend selection ─────────────────────────────────────────────────────────
#
# '--metal'  offloads the per-grain kernel to the Metal GPU (Apple Silicon).
# '--batch'  uses the batch integrator (run_pathlines_batch!) regardless of backend.
#            Metal automatically implies batch mode (single kernel call per step
#            over all tracers is much more efficient than one launch per tracer).

const USE_METAL = "--metal" in ARGS
const USE_BATCH = "--batch" in ARGS || USE_METAL

if USE_METAL
    using Metal
    get_backend() = Metal.MetalBackend()
else
    get_backend() = CPU()
end

# ── Simulation parameters ────────────────────────────────────────────────────

const PLATE_SPEED     = 2.0 / (100 * 365 * 86400)   # 2 cm/yr → m/s
const DOMAIN_HEIGHT   = 2e5                           # 200 km (olivine-spinel transition)
const DOMAIN_WIDTH    = 1e6                           # 1000 km
const N_TIMESTEPS       = 50    # ODE path (adaptive Tsit5 sub-steps internally)
const N_TIMESTEPS_BATCH = 50    # batch path (Tsit5 handles accuracy per interval)
const PATHLINE_ENDS   = (-0.1, -0.3, -0.54, -0.78)   # fraction of domain height
const DOMAIN_COORDS   = ("X", "Z")
const MAX_STRAIN      = 10.0
const GBM_MOBILITIES  = (10, 125)                     # M* values for the two columns
const _SUFFIX = USE_BATCH ? "_batch" : ""
const OUT_FIGURE_F64  = "cornerflow2d_simple_example$(_SUFFIX).png"
const OUT_FIGURE_F32  = "cornerflow2d_simple_example_float32$(_SUFFIX).png"

# Runs to execute: Metal forces Float32-only; CPU runs both precisions.
const RUNS = USE_METAL ?
    ((Float32, "cornerflow2d_simple_example_metal$(_SUFFIX).png"),) :
    ((Float64, OUT_FIGURE_F64), (Float32, OUT_FIGURE_F32))

const MIN_COORDS = [0.0, 0.0, -DOMAIN_HEIGHT]
const MAX_COORDS = [DOMAIN_WIDTH, 0.0, 0.0]

# ── Velocity field ───────────────────────────────────────────────────────────

f_velocity, f_velocity_grad = corner_2d(DOMAIN_COORDS..., PLATE_SPEED)

final_locations = [
    [DOMAIN_WIDTH, 0.0, z * DOMAIN_HEIGHT] for z in PATHLINE_ENDS
]

# ── Single-pathline CPO solver ───────────────────────────────────────────────

function run_pathline(params, f_velocity, f_velocity_grad,
                      min_coords, max_coords, final_location;
                      float_type::Type{T}=Float64,
                      backend::KernelAbstractions.Backend=CPU()) where T<:AbstractFloat
    olA = Mineral(
        float_type = T,
        phase   = olivine,
        fabric  = olivine_A,
        regime  = matrix_dislocation,
        n_grains = params[:number_of_grains],
    )
    ens = Mineral(
        float_type = T,
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

    deformation_gradient = Matrix{T}(I, 3, 3)
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
            (timestamps[i-1], timestamps[i], f_position);
            backend = backend,
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

function run_all_cases(float_type::Type{T};
                       backend::KernelAbstractions.Backend=CPU()) where T<:AbstractFloat
    cases = Dict{Int, Dict{Symbol, Vector}}()
    for (mi, Mstar) in enumerate(GBM_MOBILITIES)
        params = default_params()
        params[:phase_assemblage] = [olivine, enstatite]
        params[:phase_fractions]  = [0.7, 0.3]
        params[:gbm_mobility]     = Float64(Mstar)
        params[:number_of_grains] = 5000

        n_paths = length(final_locations)
        path_results = Vector{Any}(undef, n_paths)

        Threads.@threads for i in eachindex(final_locations)
            println("[$T] M*=$Mstar  Pathline $i / $n_paths  (thread $(Threads.threadid()))")
            _, positions, strains, olA, _ = run_pathline(
                params, f_velocity, f_velocity_grad, MIN_COORDS, MAX_COORDS, final_locations[i];
                float_type=T, backend=backend,
            )
            path_results[i] = (positions, strains, compute_diagnostics(olA)...)
        end

        case = Dict{Symbol,Vector}(
            :strains    => Vector{Vector{Float64}}(),
            :positions  => Vector{Vector{Vector{Float64}}}(),
            :m_indices  => Vector{Vector{Float64}}(),
            :directions => Vector{Matrix{Float64}}(),
        )
        for i in 1:n_paths
            positions, strains, m_indices, directions = path_results[i]
            push!(case[:strains], strains)
            push!(case[:positions], positions)
            push!(case[:m_indices], m_indices)
            push!(case[:directions], directions)
        end
        cases[mi] = case
    end
    return cases
end

# ── Batch GPU variant: all pathlines processed in one kernel call per step ─────
#
# Pre-computes pathlines and VGs on CPU, then calls run_pathlines_batch! which
# issues a single _batch_grain_kernel! call per outer step covering
# n_paths × n_grains work items simultaneously.
#
# Uses forward-Euler integration (vs. adaptive Tsit5 in run_all_cases), so
# results are slightly less accurate but computation is much faster for many
# tracers.  The difference is negligible at N_TIMESTEPS=50 for the M-index plot.

function run_all_cases_batch(float_type::Type{T};
                              backend::KernelAbstractions.Backend=CPU()) where T<:AbstractFloat
    cases = Dict{Int, Dict{Symbol, Vector}}()
    for (mi, Mstar) in enumerate(GBM_MOBILITIES)
        params = default_params()
        params[:phase_assemblage] = [olivine, enstatite]
        params[:phase_fractions]  = [0.7, 0.3]
        params[:gbm_mobility]     = Float64(Mstar)
        params[:number_of_grains] = 5000

        n_paths = length(final_locations)

        # ── Pre-compute pathlines (CPU, one per tracer) ──────────────────────
        pathlines_raw = Vector{Any}(undef, n_paths)
        Threads.@threads for i in eachindex(final_locations)
            timestamps, f_pos = get_pathline(
                final_locations[i], f_velocity, f_velocity_grad,
                MIN_COORDS, MAX_COORDS;
                max_strain = MAX_STRAIN,
                regular_steps = N_TIMESTEPS_BATCH,
            )
            positions          = [collect(f_pos(t)) for t in timestamps]
            velocity_gradients = [f_velocity_grad(NaN, x) for x in positions]
            pathlines_raw[i]   = (timestamps, positions, velocity_gradients)
        end

        # ── Build per-tracer Mineral sets ────────────────────────────────────
        minerals_per_tracer = Vector{Vector{Mineral{T}}}(undef, n_paths)
        for i in 1:n_paths
            minerals_per_tracer[i] = [
                Mineral(float_type=T, phase=olivine,   fabric=olivine_A,   regime=matrix_dislocation, n_grains=params[:number_of_grains]),
                Mineral(float_type=T, phase=enstatite, fabric=enstatite_AB, regime=matrix_dislocation, n_grains=params[:number_of_grains]),
            ]
        end

        println("[$T batch] M*=$Mstar — running $(n_paths) tracers × $(params[:number_of_grains]) grains on $(backend)")
        batch_strains = run_pathlines_batch!(
            minerals_per_tracer, params, pathlines_raw;
            backend = backend,
            snapshot_stride = N_TIMESTEPS_BATCH ÷ N_TIMESTEPS,
        )

        # ── Collect results in the same format as run_all_cases ─────────────
        case = Dict{Symbol,Vector}(
            :strains    => Vector{Vector{Float64}}(),
            :positions  => Vector{Vector{Vector{Float64}}}(),
            :m_indices  => Vector{Vector{Float64}}(),
            :directions => Vector{Matrix{Float64}}(),
        )
        stride = N_TIMESTEPS_BATCH ÷ N_TIMESTEPS
        for i in 1:n_paths
            olA = minerals_per_tracer[i][1]
            m_indices, directions = compute_diagnostics(olA)
            push!(case[:strains],    batch_strains[i])
            push!(case[:positions],  pathlines_raw[i][2][1:stride:end])
            push!(case[:m_indices],  m_indices)
            push!(case[:directions], directions)
        end
        cases[mi] = case
    end
    return cases
end

const markers_list = [:rect, :circle, :utriangle, :star5]
const pathline_labels = [
    "zf = $(round(z * DOMAIN_HEIGHT / 1e3; digits=0)) km"
    for z in PATHLINE_ENDS
]
to_km(x) = x / 1e3

# ── Helper: draw domain panel (a) or (b) ─────────────────────────────────────

function draw_domain_panel!(fig_pos, panel_label, case, strain_min, strain_max, cgrad_rev, cmap_rev)
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

function draw_strength_panel!(fig_pos, panel_label, case, strain_min, strain_max, cmap_rev)
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

# ── Run and plot for Float64 and Float32 ─────────────────────────────────────

println("Running on $(Threads.nthreads()) thread(s)$(USE_METAL ? " (Metal GPU)" : "")$(USE_BATCH ? " [batch mode]" : "")")

for (float_type, out_figure) in RUNS
    println("\n=== Float type: $float_type, backend: $(get_backend()) ===")
    t_total = @elapsed cases = USE_BATCH ?
        run_all_cases_batch(float_type; backend=get_backend()) :
        run_all_cases(float_type; backend=get_backend())
    println("Total computation time: $(round(t_total; digits=1)) s")

    # ── Plotting with GLMakie (4-panel layout matching Fig. 10) ────────────────

    strain_min = 0.0
    strain_max = maximum(vcat(maximum.(cases[1][:strains]), maximum.(cases[2][:strains])))
    cmap_rev  = Reverse(:batlow)
    cgrad_rev = cgrad(:batlow, rev=true)

    fig = Figure(size = (1200, 1100))

    # ── Assemble figure ────────────────────────────────────────────────────────
    # Row 1: (a) M*=10, Row 2: (b) M*=125, Row 3: (c) and (d) side by side

    draw_domain_panel!(fig[1, 1:2], "(a) M* = 10",  cases[1], strain_min, strain_max, cgrad_rev, cmap_rev)
    draw_domain_panel!(fig[2, 1:2], "(b) M* = 125", cases[2], strain_min, strain_max, cgrad_rev, cmap_rev)
    draw_strength_panel!(fig[3, 1], "(c) M* = 10",  cases[1], strain_min, strain_max, cmap_rev)
    draw_strength_panel!(fig[3, 2], "(d) M* = 125", cases[2], strain_min, strain_max, cmap_rev)

    Colorbar(fig[0, 1:2]; colormap = cmap_rev, colorrange = (strain_min, strain_max),
        label = "Strain (ε)", vertical = false, flipaxis = false,
        width = Relative(0.5), height = 15)

    save(out_figure, fig; px_per_unit = 2)
    println("Figure saved to $out_figure")
end  # for (float_type, out_figure)
