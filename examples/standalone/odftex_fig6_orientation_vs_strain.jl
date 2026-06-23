# Reproduce Figure 6 of Ribe, Faccenda & VanderBeek (2026, GJI) — the ODFTEX paper.
#
# Expectation value of the olivine [100] orientation relative to the shear plane,
# as a function of shear strain, in SIMPLE SHEAR (Γ = 1). Three model curves:
#   • FSE   (red)   — long axis of the finite strain ellipsoid
#   • λ=2.6 (blue)  — ODFTEX [100] expectation-value orientation
#   • λ=10  (green) — ODFTEX [100] expectation-value orientation
#
# The paper plots the "expectation value of the [100] orientation"
# (∫|angle| f dg / ∫ f dg, via `odf_expectation_angle`) — the ODF-weighted mean of
# the in-plane angle, NOT the dominant axis (`odf_mean_axis`). For an isotropic
# ODF this is 45°, so all curves start at 45°. At low strain the recrystallising
# curves sit slightly ABOVE the FSE (the [100] distribution stays spread near the
# instantaneous extension direction), then cross below it and rotate to the shear
# plane faster than the FSE — the signature of dynamic recrystallisation (Zhang &
# Karato 1995). The experimental symbols in the paper are laboratory data and are
# not reproduced here.
#
# Caveat: the low-strain behaviour and the FSE > λ=2.6 > λ=10 crossover are
# reproduced; the λ=2.6 curve saturates higher at large strain (≈20° at γ=2) than
# the paper's (~5°), reflecting a difference in the exact expectation definition.
# The underlying ODF matches Neil Ribe's reference Fortran element-wise (see the
# validation test), so this is a diagnostic-definition difference, not a method
# difference.
#
# Sense of shear: sinistral in the X–Y plane, L[1,2] = ∂vx/∂y = −2 (ε̇₀ = 1), so
# the shear strain is γ = 2τ where τ = ε̇₀ t is the dimensionless time.
#
# Run with:
#   julia --project=examples/standalone examples/standalone/odftex_fig6_orientation_vs_strain.jl

push!(LOAD_PATH, "@v#.#")

using DRex
using LinearAlgebra
using Printf

const MAKIE_BACKEND = try
    @eval using CairoMakie
    :cairo
catch
    @eval using GLMakie
    :gl
end

using DRex: init_odf, odftex_step!, fse_long_axis, odf_expectation_angle

# ── Parameters ────────────────────────────────────────────────────────────────
const NBOX      = 60
const GAMMA_MAX = 2.0
const N_GAMMA   = 41                  # strain samples along the curve
const N_STEPS   = 60                  # ODFTEX steps to reach each strain
const LAMBDAS   = [(2.6, :blue, "λ = 2.6"), (10.0, :green, "λ = 10")]

# Sinistral simple shear in the X–Y plane.
const VG = [0.0 -2.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]

# ── Helpers ───────────────────────────────────────────────────────────────────
"""[100] expectation-value orientation (deg from shear plane) at strain γ."""
function a_axis_angle(λ, γ)
    τ = γ / 2
    st = init_odf(NBOX; λ = λ)
    dt = τ / N_STEPS
    for _ in 1:N_STEPS
        odftex_step!(st, VG, dt)
    end
    # Deformation plane is X–Y; X is the shear direction. Expectation value of the
    # in-plane [100] angle relative to the shear plane.
    return odf_expectation_angle(st, 1; plane_dir1 = 1, plane_dir2 = 2)
end

"""FSE long-axis orientation (deg from shear plane) at strain γ."""
function fse_angle(γ)
    v = fse_long_axis(exp(VG * (γ / 2)))
    a = abs(atand(v[2], v[1]))
    return a > 90 ? 180 - a : a
end

# ── Compute curves ────────────────────────────────────────────────────────────
# Sample from a small non-zero strain (the texture is undefined at γ=0; the
# expectation value starts at 45° = the isotropic in-plane mean angle).
println("Computing ODFTEX [100] expectation orientation vs shear strain (nbox=$(NBOX)) …")
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
          title = "ODFTEX — [100] expectation orientation vs shear strain (simple shear)",
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
