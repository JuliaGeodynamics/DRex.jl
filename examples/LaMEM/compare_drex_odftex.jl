#=
# compare_drex_odftex.jl
#
# Run the CPO post-processing of the 3D subduction example twice — once with the
# discrete D-Rex model and once with the continuum ODFTEX model — on identical
# tracers, and plot the two textures side by side.
#
# Usage (after run_lamem_subduction3d.jl has produced Subduction_3D_CPO/):
#   cd examples/LaMEM
#   julia --project=. -t auto compare_drex_odftex.jl
#
# Output: Subduction_3D_CPO_tracers/drex_vs_odftex.png
=#

using LaMEM, GeophysicalModelGenerator
using DRex
using CairoMakie
using LinearAlgebra

const SIM_NAME = "Subduction_3D_CPO"
const SIM_DIR  = "Subduction_3D_CPO"

# Same target positions and D-Rex parameters as postprocess_cpo.jl
target_positions = [Float64[x, y, z] for x in -1500.0:150.0:-300.0
                                     for y in 0.0:250.0:1500.0
                                     for z in (-150.0, -300.0)]

drex_params = let p = default_params()
    p[:phase_assemblage] = [olivine, enstatite]
    p[:phase_fractions]  = [0.7, 0.3]
    p[:gbm_mobility]     = 10.0
    p
end

# ── Load once, backtrack once, evolve with both methods ──────────────────────
snaps = load_snapshots(SIM_NAME, SIM_DIR)[6:end]          # skip 5 spin-up steps
src   = backtrack_positions(target_positions, snaps; n_substeps = 5)

results = Dict{Symbol,Any}()
timing  = Dict{Symbol,Float64}()
for method in (:drex, :odftex)
    tracers = create_tracers(src; method, n_grains = 1000, seed = 42,
                             phase_assemblage = drex_params[:phase_assemblage])
    timing[method]  = @elapsed evolve_cpo!(tracers, drex_params, snaps; n_substeps = 5)
    results[method] = tracers
    @info "$method: $(round(timing[method]; digits = 1)) s for $(length(tracers)) tracers"
end

# ── Diagnostics (olivine; D-Rex grains weighted by their volume fractions) ───
function final_olivine(tracer)
    ol = tracer.minerals[1]
    return resample_orientations([ol.orientations[end]], [ol.fractions[end]]; seed = 1)[1][1]
end

strain(tr) = sqrt(maximum(svdvals(tr.deformation_gradient * tr.deformation_gradient'))) - 1

ori   = Dict(m => final_olivine.(results[m]) for m in (:drex, :odftex))
mind  = Dict(m => [misorientation_index(o, orthorhombic) for o in ori[m]] for m in (:drex, :odftex))
fast  = Dict(m => [bingham_average(o; axis = "a") for o in ori[m]] for m in (:drex, :odftex))
eps   = strain.(results[:drex])
angle = [acosd(clamp(abs(dot(a, b)), 0, 1)) for (a, b) in zip(fast[:drex], fast[:odftex])]

@info "Median angle between fast axes: $(round(sort(angle)[end ÷ 2]; digits = 1))°"

# ── Figure ───────────────────────────────────────────────────────────────────
# Six tracers spanning the range of finite strain
shown  = sortperm(eps)[round.(Int, range(1, length(eps); length = 6))]
labels = Dict(:drex => "D-Rex", :odftex => "ODFTEX")
colors = Dict(:drex => :steelblue, :odftex => :firebrick)
circ   = range(0, 2π; length = 200)

fig = Figure()
Label(fig[0, 1:6], "3D subduction: olivine [100] pole figures at the end of the run";
      fontsize = 15, font = :bold)

for (r, method) in enumerate((:drex, :odftex)), (c, i) in enumerate(shown)
    xv, yv, zv = poles(ori[method][i]; hkl = [1, 0, 0], ref_axes = "xz")
    x2d, y2d   = lambert_equal_area(xv, yv, zv)
    title = r == 1 ? "x=$(round(Int, target_positions[i][1])), y=$(round(Int, target_positions[i][2])), " *
                     "z=$(round(Int, target_positions[i][3])) km\nstrain $(round(eps[i]; digits = 1))" : ""
    ax = Axis(fig[r, c]; width = 170, height = 170, title, titlesize = 11,
              xlabel = "M = $(round(mind[method][i]; digits = 2))", xlabelsize = 11)
    hidedecorations!(ax; label = false)
    hidespines!(ax)
    lines!(ax, cos.(circ), sin.(circ); color = :black, linewidth = 0.8)
    scatter!(ax, x2d, y2d; markersize = 2.5, color = (colors[method], 0.4))
    c == 1 && Label(fig[r, 0], labels[method]; rotation = π / 2, font = :bold, fontsize = 14)
end

bottom = fig[3, 0:6] = GridLayout()
ax1 = Axis(bottom[1, 1]; height = 280, xlabel = "Finite strain (largest stretch − 1)", ylabel = "M-index",
           title = "Texture strength")
for method in (:drex, :odftex)
    scatter!(ax1, eps, mind[method]; color = colors[method], markersize = 7, label = labels[method])
end
axislegend(ax1; position = :lt)

ax2 = Axis(bottom[1, 2]; height = 280, xlabel = "M-index, D-Rex", ylabel = "M-index, ODFTEX",
           title = "Texture strength compared\n(one point per tracer)", aspect = DataAspect())
mmax = 1.05 * max(maximum(mind[:drex]), maximum(mind[:odftex]))
lines!(ax2, [0, mmax], [0, mmax]; color = :gray, linestyle = :dash)
scatter!(ax2, mind[:drex], mind[:odftex]; color = :black, markersize = 7)

ax3 = Axis(bottom[1, 3]; height = 280, xlabel = "Finite strain (largest stretch − 1)",
           ylabel = "Angle between the two fast axes (°)",
           title = "Fast-axis mismatch between D-Rex and ODFTEX\n(one point per tracer)")
scatter!(ax3, eps, angle; color = :black, markersize = 7)
ylims!(ax3, 0, 90)

resize_to_layout!(fig)

outpng = joinpath("$(SIM_NAME)_tracers", "drex_vs_odftex.png")
mkpath(dirname(outpng))
save(outpng, fig; px_per_unit = 2)
@info "Comparison figure saved → $outpng"
