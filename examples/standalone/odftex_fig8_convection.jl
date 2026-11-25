# Reproduce Figure 8 of Ribe, Faccenda & VanderBeek (2026, GJI) — ODFTEX paper.
#
# Seismic anisotropy in a kinematically prescribed, steady 2-D flow made of two
# adjacent convection cells (600 km × 400 km). Texture is evolved for 20 Myr from
# an isotropic start for harzburgite aggregates (olivine:orthopyroxene = 70:30)
# on a regular grid (5 km spacing ⇒ 9600 aggregates), once with D-Rex and once
# with ODFTEX:
#   (a) streamlines
#   (b) D-Rex   — M* = 10, χ = 0.3, λ* = 5
#   (c) ODFTEX  — slip system (010)[100], λ = 2.6, 12³ Euler boxes, r_max = 5
# Colours: S-wave radial anisotropy ξ = N/L. White bars: P-wave fast direction in
# the plane of the flow, with length proportional to the P-wave anisotropy.
#
# Each aggregate is traced back along its pathline for 20 Myr and then evolved
# forward with `update_all!` (D-Rex) or `odftex_update!` (ODFTEX).
#
# ASSUMPTIONS — the paper shows the streamlines but does not give the flow:
#   • stream function ψ ∝ sin(πx/300 km)·sin(πz/400 km), downwelling in the centre;
#   • peak horizontal surface velocity U = 50 km/Myr (5 cm/yr).
# The paper's D-Rex panel was computed with D-Rex_M (ECOMAN), which differs in
# detail from the D-Rex implemented here.
#
# Run with (threads strongly recommended; ~9600 aggregates at the default spacing):
#   julia -t auto --project=examples/standalone examples/standalone/odftex_fig8_convection.jl
# A coarser grid for a quick look:  ODFTEX_FIG8_SPACING=20 julia -t auto …

push!(LOAD_PATH, "@v#.#")

using MantleFabric
using LinearAlgebra
using Printf

const MAKIE_BACKEND = try
    @eval using CairoMakie
    :cairo
catch
    @eval using GLMakie
    :gl
end

using MantleFabric: init_odf, odftex_update!, odf_to_minerals

# ── Parameters ────────────────────────────────────────────────────────────────
const WIDTH    = 600.0         # km, two cells of 300 km
const DEPTH    = 400.0         # km
const CELL     = WIDTH / 2
const U_SURF   = 50.0          # km/Myr, peak horizontal surface velocity (assumed)
const T_END    = 20.0          # Myr
const N_INT    = 200           # pathline intervals
const SPACING  = parse(Float64, get(ENV, "ODFTEX_FIG8_SPACING", "5"))   # km
const NBOX     = 12
const LAMBDA   = 2.6
const RMAX     = 5.0
const N_GRAINS = 1000          # D-Rex grains per phase
const PHASES   = [olivine, enstatite]
const FRACS    = [0.7, 0.3]

# ── Flow (lab frame: x horizontal, y out of plane, z up; depth d = −z) ────────
function velocity(p)
    sx, cx = sincos(π * p[1] / CELL)
    sd, cd = sincos(-π * p[3] / DEPTH)
    return [U_SURF * sx * cd, 0.0, U_SURF * (DEPTH / CELL) * cx * sd]
end

function velocity_gradient(t, p)
    sx, cx = sincos(π * p[1] / CELL)
    sd, cd = sincos(-π * p[3] / DEPTH)
    L = zeros(3, 3)
    L[1, 1] =  U_SURF * π / CELL * cx * cd
    L[3, 3] = -L[1, 1]
    L[1, 3] =  U_SURF * π / DEPTH * sx * sd
    L[3, 1] = -U_SURF * (DEPTH / CELL) * π / CELL * sx * sd
    return L
end

stream_function(x, d) = U_SURF * DEPTH / π * sin(π * x / CELL) * sin(π * d / DEPTH)

"Positions at the N_INT+1 interval boundaries of the pathline ending at `target` (RK4, backwards)."
function pathline(target; n_sub = 4)
    path = Vector{Vector{Float64}}(undef, N_INT + 1)
    path[end] = copy(target)
    h = -T_END / N_INT / n_sub
    p = copy(target)
    for i in N_INT:-1:1
        for _ in 1:n_sub
            k1 = velocity(p); k2 = velocity(p .+ h / 2 .* k1)
            k3 = velocity(p .+ h / 2 .* k2); k4 = velocity(p .+ h .* k3)
            p = p .+ h / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        end
        path[i] = p
    end
    return path
end

# ── Texture evolution along one pathline → Voigt-averaged elastic tensor ──────
function evolve(method::Symbol, path, seed)
    dt = T_END / N_INT
    F  = Matrix(1.0I, 3, 3)
    if method == :odftex
        state = init_odf(NBOX; λ = LAMBDA)
    else
        minerals = Mineral[Mineral(; phase = olivine, fabric = olivine_A, n_grains = N_GRAINS, seed),
                           Mineral(; phase = enstatite, fabric = enstatite_AB, n_grains = N_GRAINS, seed = seed + 1)]
    end
    for i in 1:N_INT
        t0, t1 = (i - 1) * dt, i * dt
        p0, p1 = path[i], path[i + 1]
        getpos = t -> p0 .+ (t - t0) / dt .* (p1 .- p0)
        if method == :odftex
            F = odftex_update!(state, F, velocity_gradient, (t0, t1, getpos); rmax = RMAX)
        else
            F = update_all!(minerals, DREX_PARAMS, F, velocity_gradient, (t0, t1, getpos);
                            push_snapshot = false)
        end
    end
    method == :odftex && (minerals = odf_to_minerals(state; phases = PHASES))
    return voigt_averages(minerals, PHASES, FRACS)
end

const DREX_PARAMS = let p = default_params()
    p[:phase_assemblage]      = PHASES
    p[:phase_fractions]       = FRACS
    p[:gbm_mobility]          = 10.0      # M*
    p[:gbs_threshold]         = 0.3       # χ
    p[:nucleation_efficiency] = 5.0       # λ*
    p[:number_of_grains]      = N_GRAINS
    p
end

# ── Seismic anisotropy from a 6×6 Voigt matrix (axis 3 vertical) ──────────────
radial_anisotropy(C) = ((C[1,1] + C[2,2]) / 8 - C[1,2] / 4 + C[6,6] / 2) / ((C[4,4] + C[5,5]) / 2)

"P-wave fast direction in the x–z plane (angle from x, degrees) and anisotropy (%)."
function p_wave_anisotropy(C)
    c = voigt_to_elastic_tensor(C)
    vp = map(0:179) do α
        n = (cosd(α), 0.0, sind(α))
        Γ = [sum(c[i,j,k,l] * n[j] * n[l] for j in 1:3, l in 1:3) for i in 1:3, k in 1:3]
        sqrt(maximum(eigvals(Symmetric(Γ))))
    end
    vmax, imax = findmax(vp)
    return Float64(imax - 1), 200 * (vmax - minimum(vp)) / (vmax + minimum(vp))
end

# ── Run ───────────────────────────────────────────────────────────────────────
xs = collect(SPACING / 2 : SPACING : WIDTH)
ds = collect(SPACING / 2 : SPACING : DEPTH)
targets = [[x, 0.0, -d] for x in xs, d in ds]
println("Fig. 8: $(length(targets)) aggregates ($(SPACING) km spacing), $(Threads.nthreads()) thread(s)")

paths = Array{Vector{Vector{Float64}}}(undef, size(targets))
Threads.@threads for I in eachindex(targets)
    paths[I] = pathline(targets[I])
end

results = Dict{Symbol,Any}()
for method in (:odftex, :drex)
    C = Array{Matrix{Float64}}(undef, size(targets))
    t = @elapsed Threads.@threads for I in eachindex(targets)
        C[I] = evolve(method, paths[I], 2 * I)
    end
    pw = p_wave_anisotropy.(C)
    results[method] = (ξ = radial_anisotropy.(C), angle = first.(pw), aniso = last.(pw), time = t)
    @printf("  %-6s: %.0f s   ξ range %.3f–%.3f   max P anisotropy %.1f%%\n",
            method, t, extrema(results[method].ξ)..., maximum(results[method].aniso))
end

o, d = results[:odftex], results[:drex]
Δang = [min(abs(a - b), 180 - abs(a - b)) for (a, b) in zip(o.angle, d.angle)]
@printf("  ODFTEX − D-Rex: mean |Δξ| = %.3f, mean P-direction difference = %.1f°, mean |ΔP anisotropy| = %.2f%%\n",
        sum(abs, o.ξ .- d.ξ) / length(o.ξ), sum(Δang) / length(Δang),
        sum(abs, o.aniso .- d.aniso) / length(o.aniso))
@printf("  ODFTEX is %.1f× faster than D-Rex\n", d.time / o.time)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = Figure(size = (760, 1250))
common = (; xlabel = "km", ylabel = "Depth (km)", yreversed = true, aspect = DataAspect(),
            limits = (0, WIDTH, 0, DEPTH))

ax = Axis(fig[1, 1]; title = "(a) Streamlines", common...)
xf = range(0, WIDTH; length = 241); df = range(0, DEPTH; length = 161)
contour!(ax, xf, df, [abs(stream_function(x, dd)) for x in xf, dd in df];
         levels = 10, color = :black, linewidth = 1)

stride = max(1, round(Int, 20 / SPACING))            # bars every ~20 km
bar_scale = 0.9 * stride * SPACING / max(maximum(o.aniso), maximum(d.aniso))
hm = nothing
for (row, (res, name)) in enumerate(((d, "(b) D-Rex  (M* = 10, χ = 0.3, λ* = 5)"),
                                     (o, "(c) ODFTEX  (λ = 2.6, r_max = 5)")))
    local ax = Axis(fig[row + 1, 1]; title = name, common...)
    global hm = heatmap!(ax, xs, ds, res.ξ; colormap = :RdBu, colorrange = (0.9, 1.1))
    for i in stride ÷ 2 + 1 : stride : length(xs), j in stride ÷ 2 + 1 : stride : length(ds)
        h = bar_scale * res.aniso[i, j] / 2
        dx, dd = h * cosd(res.angle[i, j]), -h * sind(res.angle[i, j])   # depth axis points down
        lines!(ax, [xs[i] - dx, xs[i] + dx], [ds[j] - dd, ds[j] + dd]; color = :white, linewidth = 1.2)
    end
end
Colorbar(fig[4, 1], hm; vertical = false, label = "Radial anisotropy ξ", width = Relative(0.6))

outfile = joinpath(@__DIR__, "odftex_fig8_convection.png")
save(outfile, fig; px_per_unit = 2)
println("Saved: ", outfile, "  (backend: ", MAKIE_BACKEND, ")")
