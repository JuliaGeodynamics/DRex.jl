# Reproduce Figure 6 of Ribe, Faccenda & VanderBeek (2026, GJI) — the ODFTEX paper.
#
# Mean orientation of the olivine [100] axis relative to the shear plane, as a
# function of shear strain, in SIMPLE SHEAR (Γ = 1). Three model curves:
#   • FSE   (red)   — long axis of the finite strain ellipsoid
#   • λ=2.6 (blue)  — ODFTEX mean [100] orientation
#   • λ=10  (green) — ODFTEX mean [100] orientation
#
# The mean orientation is the leading eigenvector of the ODF-weighted orientation
# tensor ⟨a⊗a⟩ of the [100] axes (`odf_mean_axis`). The paper does not spell out
# its definition of the "expectation value of the [100] orientation"; this one
# reproduces its curves to within a few degrees (≈6° at γ = 2 for λ = 2.6, and the
# shear plane reached by γ ≈ 1.3 for λ = 10). The alternative `odf_expectation_angle`
# (mean of the folded in-plane angle) is biased towards 45° and does not.
# The experimental symbols in the paper are laboratory data and are not shown.
#
# Sense of shear: sinistral in the X–Y plane, L[1,2] = ∂vx/∂y = −2 (ε̇₀ = 1), so
# the shear strain is γ = 2τ where τ = ε̇₀ t is the dimensionless time.
#
# Run with:
#   julia --project=examples/standalone examples/standalone/odftex_fig6_orientation_vs_strain.jl

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

using MantleFabric: init_odf, odftex_step!, fse_long_axis, odf_mean_axis

# ── Parameters ────────────────────────────────────────────────────────────────
const NBOX      = 60
const GAMMA_MAX = 2.0
const N_GAMMA   = 41                  # strain samples along the curve
const N_STEPS   = 60                  # ODFTEX steps to reach each strain
const LAMBDAS   = [(2.6, :blue, "λ = 2.6"), (10.0, :green, "λ = 10")]

# Sinistral simple shear in the X–Y plane.
const VG = [0.0 -2.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]

# ── Helpers ───────────────────────────────────────────────────────────────────
"""Mean [100] orientation (deg from shear plane) at strain γ."""
function a_axis_angle(λ, γ)
    τ = γ / 2
    st = init_odf(NBOX; λ = λ)
    dt = τ / N_STEPS
    for _ in 1:N_STEPS
        odftex_step!(st, VG, dt)
    end
    # Deformation plane is X–Y; X is the shear direction.
    v = odf_mean_axis(st, "a")
    a = abs(atand(v[2], v[1]))
    return a > 90 ? 180 - a : a
end

"""FSE long-axis orientation (deg from shear plane) at strain γ."""
function fse_angle(γ)
    v = fse_long_axis(exp(VG * (γ / 2)))
    a = abs(atand(v[2], v[1]))
    return a > 90 ? 180 - a : a
end

# ── Compute curves ────────────────────────────────────────────────────────────
# Sample from a small non-zero strain (the mean axis is undefined at γ=0; for
# infinitesimal strain it lies along the extension direction, at 45°).
println("Computing ODFTEX mean [100] orientation vs shear strain (nbox=$(NBOX)) …")
γs = collect(range(0.02, GAMMA_MAX; length = N_GAMMA))

fse_curve = [fse_angle(γ) for γ in γs]
λ_curves  = map(LAMBDAS) do (λ, _, _)
    @printf("  λ = %.1f …\n", λ)
    [a_axis_angle(λ, γ) for γ in γs]
end

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = Figure(; size = (640, 520))
ax = Axis(fig[1, 1];
          xlabel = "shear strain", ylabel = "orientation relative to shear plane (degrees)",
          title = "ODFTEX — mean [100] orientation vs shear strain (simple shear)",
          xticks = 0:0.5:2.0, yticks = 0:15:45)
limits!(ax, 0, GAMMA_MAX, -2, 47)

lines!(ax, γs, fse_curve; color = :red, linewidth = 2.5, label = "FSE")
for ((λ, col, lab), curve) in zip(LAMBDAS, λ_curves)
    lines!(ax, γs, curve; color = col, linewidth = 2.5, label = lab)
end
hlines!(ax, [0.0]; color = :black, linestyle = :dash, linewidth = 1)
axislegend(ax; position = :rt)

outfile = joinpath(@__DIR__, "odftex_fig6_orientation_vs_strain.png")
save(outfile, fig)
println("Saved: ", outfile, "  (backend: ", MAKIE_BACKEND, ")")
@printf("  FSE @ γ=2: %.1f°,  λ=2.6 @ γ=2: %.1f°,  λ=10 @ γ=2: %.1f°\n",
        fse_curve[end], λ_curves[1][end], λ_curves[2][end])
