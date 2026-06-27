# Reproduce Figure 7(b) of Ribe, Faccenda & VanderBeek (2026, GJI) — ODFTEX paper.
#
# ODFTEX prediction for an olivine aggregate deformed in UNIAXIAL COMPRESSION by
# 58% shortening WITH recrystallisation (λ = 2.6), single slip system (010)[100].
# [100], [010] and [001] pole figures are shown, viewed DOWN the compression axis
# (the compression axis is at the centre of each pole figure, as in the paper).
#
# Panel (a) of the paper's Fig. 7 is the experimental data of Nicolas et al.
# (1973) and is not reproduced here; this script reproduces the model panel (b).
#
# Kinematics: axisymmetric compression along Z with equal extension in X,Y:
#   L = diag(ε̇, ε̇, −2ε̇)   (vorticity 0 ⇒ Γ = 0; strain-rate scale ė₀ = √3 ε̇)
# 58% shortening ⇒ axial stretch exp(−2ε̇ t) = 0.42 ⇒ ε̇ t = −ln(0.42)/2, so the
# dimensionless time is τ = ė₀ t = √3 · ε̇ t ≈ 0.751.
#
# Result (matches the paper): [010] forms a strong CENTRAL peak along the
# compression axis; [100] and [001] form girdles in the plane perpendicular to
# compression, with the [001] girdle stronger/more concentrated than [100].
# Colour scale 0–5 MUD (as in the paper Fig. 7), Crameri "berlin" map.
#
# Run with:
#   julia --project=examples/standalone examples/standalone/odftex_fig7_uniaxial.jl

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

using DRex: init_odf, odftex_step!, odftex_kinematics, strain_rate_scale,
            odf_polefigure_density, odf_misorientation_index
using Random: MersenneTwister

include(joinpath(@__DIR__, "odftex_polefigure_utils.jl"))

# ── Parameters ────────────────────────────────────────────────────────────────
const NBOX       = 90
const N_STEPS    = 40
const LAMBDA     = 2.6
const SHORTENING = 0.58                    # 58% shortening
const AXES       = [("[100]", 1), ("[010]", 2), ("[001]", 3)]

# Colour scale for Fig. 7 (0–5 MUD).
const F7_CMAX   = 5.0
const F7_NBANDS = 12
const F7_TICKS  = 0:1:5

# Axisymmetric uniaxial compression along Z (compression axis = view axis).
const VG = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 -2.0]

# Dimensionless time τ for the target shortening (ε̇₀ = √3 ε̇ with ε̇ = 1 here).
const E0      = strain_rate_scale(first(odftex_kinematics(VG)))
const EPS_T   = -log(1 - SHORTENING) / 2          # ε̇ t  (axial log strain / 2)
const TAU     = E0 * EPS_T
# Nondimensionalise the velocity gradient so ė₀ = 1, then step to τ.
const VG_HAT  = VG ./ E0

# ── Run ODFTEX ────────────────────────────────────────────────────────────────
@printf("Running ODFTEX uniaxial compression (%.0f%% shortening, λ=%.1f, τ=%.3f, nbox=%d) …\n",
        100SHORTENING, LAMBDA, TAU, NBOX)
state = init_odf(NBOX; λ = LAMBDA, frac_opx = 0.0, s_ol = 1, ilimit = 0)
for _ in 1:N_STEPS
    odftex_step!(state, VG_HAT, TAU / N_STEPS)
end
m_index = odf_misorientation_index(state; n = 40000, rng = MersenneTwister(1))
@printf("  ODF integral = %.4f,  M-index (sampled) = %.4f  (paper native 12³ = 0.0984)\n",
        DRex.integral_drx(state), m_index)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = Figure(; size = (1040, 380))
for (ci, (label, row)) in enumerate(AXES)
    xs, ys, dens = odf_polefigure_density(state, row)
    ax = Axis(fig[1, ci]; title = label)
    cmax = draw_polefigure!(ax, xs, ys, dens;
                            cmin = 0.0, cmax = F7_CMAX, nbands = F7_NBANDS)
    @printf("  panel %s : max density = %.2f MUD\n", label, cmax)
end
add_row_colorbar!(fig, 1, length(AXES) + 1; cmin = 0.0, cmax = F7_CMAX, ticks = F7_TICKS)
Label(fig[0, 1:length(AXES)],
      @sprintf("ODFTEX uniaxial compression, 58%% shortening, λ=2.6  (M-index %.3f) — compression axis at centre",
               m_index);
      fontsize = 13, font = :bold)

outfile = joinpath(@__DIR__, "odftex_fig7_uniaxial.png")
save(outfile, fig)
println("Saved: ", outfile, "  (backend: ", MAKIE_BACKEND, ")")
