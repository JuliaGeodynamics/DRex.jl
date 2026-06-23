# Reproduce Figure 5(b) of Ribe, Faccenda & VanderBeek (2026, GJI) — ODFTEX paper.
#
# ODFTEX prediction for an olivine aggregate deformed in SIMPLE SHEAR (Γ = 1) to
# 140% shear strain (γ = 1.4, τ = ε̇₀ t = 0.7) WITH recrystallisation (λ = 2.6) —
# the model panel compared against the MIT23 experiment of Lee et al. (2002) in
# the paper. Panel (a) of the paper is the experimental data and is not
# reproduced here; this script reproduces the ODFTEX model panel (b).
#
# [100], [010] and [001] pole figures are shown. On the [100] panel three
# reference directions are overlaid (as in the paper):
#   • shear plane                 — black
#   • max instantaneous extension — yellow  (eigenvector of the strain rate)
#   • finite strain ellipsoid     — red     (long axis of F Fᵀ)
#
# Colour scale 0–6 MUD (as in the paper Fig. 5), Crameri "berlin" map.
# Sense of shear: sinistral in the X–Y plane, L[1,2] = ∂vx/∂y = −2 (ε̇₀ = 1).
#
# Run with:
#   julia --project=examples/standalone examples/standalone/odftex_fig5_mit23.jl

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

using DRex: init_odf, odftex_step!, odftex_kinematics,
            odf_polefigure_density, fse_long_axis, odf_misorientation_index
using Random: MersenneTwister

include(joinpath(@__DIR__, "odftex_polefigure_utils.jl"))

# ── Parameters ────────────────────────────────────────────────────────────────
const NBOX     = 90
const N_STEPS  = 40
const LAMBDA   = 2.6           # recrystallisation rate (paper recommendation)
const TAU      = 0.7           # γ = 1.4 (140% shear strain)
const AXES     = [("[100]", 1), ("[010]", 2), ("[001]", 3)]

# Colour scale for Fig. 5 (0–6 MUD).
const F5_CMAX   = 6.0
const F5_NBANDS = 12
const F5_TICKS  = 0:1:6

# Sinistral simple shear in the X–Y plane.
const VG = [0.0 -2.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]

# ── Reference directions for the [100] panel ─────────────────────────────────
# Shear plane: for v_x = L₁₂ y the plane normal is Y, so the plane projects to
# the horizontal (X) axis in the X–Y pole figure.
const SHEAR_PLANE_DIR = [1.0, 0.0, 0.0]

# Max instantaneous extension: leading eigenvector of the strain-rate tensor.
const MAX_EXT_DIR = let
    sr, _, _ = odftex_kinematics(VG)
    _, vecs = eigen(Symmetric(Matrix(sr)))
    vecs[:, end]
end

# FSE long axis at this strain (F = exp(L τ), ε̇₀ = 1 ⇒ t = τ).
const FSE_DIR = fse_long_axis(exp(VG * TAU))

# ── Run ODFTEX ────────────────────────────────────────────────────────────────
@printf("Running ODFTEX simple shear (Γ=1, λ=%.1f, γ=%d%%, nbox=%d) …\n",
        LAMBDA, round(Int, 200TAU), NBOX)
state = init_odf(NBOX; λ = LAMBDA, frac_opx = 0.0, s_ol = 1, ilimit = 0)
for _ in 1:N_STEPS
    odftex_step!(state, VG, TAU / N_STEPS)
end
m_index = odf_misorientation_index(state; n = 40000, rng = MersenneTwister(1))
@printf("  ODF integral = %.4f,  M-index (sampled) = %.4f\n",
        DRex.integral_drx(state), m_index)
@printf("  shear plane @ %.1f°,  max-extension @ %.1f°,  FSE @ %.1f° (from X)\n",
        atand(SHEAR_PLANE_DIR[2], SHEAR_PLANE_DIR[1]),
        atand(MAX_EXT_DIR[2], MAX_EXT_DIR[1]),
        atand(FSE_DIR[2], FSE_DIR[1]))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = Figure(; size = (1040, 380))
for (ci, (label, row)) in enumerate(AXES)
    xs, ys, dens = odf_polefigure_density(state, row)
    ax = Axis(fig[1, ci]; title = label)
    cmax = draw_polefigure!(ax, xs, ys, dens;
                            cmin = 0.0, cmax = F5_CMAX, nbands = F5_NBANDS)
    if ci == 1
        # Reference lines on the [100] panel (same order as the paper).
        draw_axis_line!(ax, SHEAR_PLANE_DIR; color = :black,  linewidth = 2.5)
        draw_axis_line!(ax, MAX_EXT_DIR;     color = :yellow, linewidth = 2.5)
        draw_axis_line!(ax, FSE_DIR;         color = :red,    linewidth = 2.5)
    end
    @printf("  panel %s : max density = %.2f MUD\n", label, cmax)
end
add_row_colorbar!(fig, 1, length(AXES) + 1; cmin = 0.0, cmax = F5_CMAX, ticks = F5_TICKS)
Label(fig[0, 1:length(AXES)],
      @sprintf("ODFTEX simple shear, γ=140%%, λ=2.6  (M-index %.3f) — black: shear plane, yellow: max extension, red: FSE",
               m_index);
      fontsize = 13, font = :bold)

outfile = joinpath(@__DIR__, "odftex_fig5_mit23.png")
save(outfile, fig)
println("Saved: ", outfile, "  (backend: ", MAKIE_BACKEND, ")")
