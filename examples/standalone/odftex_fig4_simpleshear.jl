# Reproduce Figure 4 of Ribe, Faccenda & VanderBeek (2026, GJI) — the ODFTEX paper.
#
# Texture of a pure olivine aggregate deformed in SIMPLE SHEAR (vorticity number
# Γ = 1) WITHOUT recrystallisation (λ = 0), single slip system (010)[100].
# [100], [010] and [001] pole figures are shown at two shear strains:
#   (a) γ = 100%  (τ = ε̇₀ t = 0.5)
#   (b) γ = 200%  (τ = 1.0)
#
# The white line on each [100] pole figure marks the long axis of the finite
# strain ellipsoid (FSE). The key result of the paper's Fig. 4 is that the [100]
# peaks follow the FSE long axis exactly when recrystallisation is inactive —
# McKenzie's (1979) finite-strain hypothesis.
#
# Densities are multiples of a uniform distribution (MUD), Crameri "berlin" map.
# Sense of shear: sinistral in the X–Y plane (top moves toward −X), i.e.
# velocity gradient L[1,2] = ∂vx/∂y = −2 (so ε̇₀ = 1 and γ = 2τ).
#
# Run with:
#   julia --project=examples/standalone examples/standalone/odftex_fig4_simpleshear.jl

push!(LOAD_PATH, "@v#.#")   # allow CairoMakie from a stacked env

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

using DRex: init_odf, odftex_step!, odf_polefigure_density, fse_long_axis

# Shared pole-figure drawing helpers (load after the Makie backend).
include(joinpath(@__DIR__, "odftex_polefigure_utils.jl"))

# ── Parameters ────────────────────────────────────────────────────────────────
const NBOX     = 90            # Euler-grid resolution per angle
const N_STEPS  = 40            # ODFTEX time steps per panel
const LAMBDA   = 0.0           # no recrystallisation (Fig. 4)
const TAUS     = [0.5, 1.0]    # rows: (a) γ=100%, (b) γ=200%   (γ = 2τ)
const AXES     = [("[100]", 1), ("[010]", 2), ("[001]", 3)]

# Sinistral simple shear in the X–Y plane: L[1,2] = ∂vx/∂y = −2.
const VG = [0.0 -2.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]

# Finite strain ellipsoid: with a constant velocity gradient L, F = exp(L t);
# `fse_long_axis(F)` (from DRex) returns the FSE long axis. Here t = τ (ε̇₀=1).

# ── Run ODFTEX at each strain ─────────────────────────────────────────────────
println("Running ODFTEX simple shear (Γ=1, λ=$(LAMBDA), nbox=$(NBOX)) …")
states = map(TAUS) do τ
    st = init_odf(NBOX; λ = LAMBDA, frac_opx = 0.0, s_ol = 1, ilimit = 0)
    dt = τ / N_STEPS
    for _ in 1:N_STEPS
        odftex_step!(st, VG, dt)
    end
    @printf("  τ=%.1f (γ=%d%%) : ODF integral = %.4f\n",
            τ, round(Int, 200τ), DRex.integral_drx(st))
    st
end

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = Figure(; size = (1040, 640))
rowlabels = ["(a) γ = 100%", "(b) γ = 200%"]
for (ri, st) in enumerate(states)
    fse = fse_long_axis(exp(VG * TAUS[ri]))   # F = exp(L t), t = τ (ε̇₀ = 1)
    fse_ang = atand(fse[2], fse[1])
    for (ci, (label, row)) in enumerate(AXES)
        xs, ys, dens = odf_polefigure_density(st, row)
        ax = Axis(fig[ri, ci]; title = ri == 1 ? label : "")
        cmax = draw_polefigure!(ax, xs, ys, dens)
        ci == 1 && draw_fse_line!(ax, fse)   # FSE long axis on the [100] panel
        if ci == 1
            Label(fig[ri, 0], rowlabels[ri]; rotation = π/2, tellheight = false,
                  fontsize = 14, font = :bold)
        end
        @printf("  panel τ=%.1f %s : max density = %.2f MUD%s\n",
                TAUS[ri], label, cmax,
                ci == 1 ? @sprintf("  (FSE long axis @ %.1f° from X)", fse_ang) : "")
    end
    add_row_colorbar!(fig, ri, length(AXES) + 1)
end
Label(fig[0, 1:length(AXES)],
      "ODFTEX simple shear (Γ=1), λ=0 — white line = FSE long axis";
      fontsize = 16, font = :bold)

outfile = joinpath(@__DIR__, "odftex_fig4_simpleshear.png")
save(outfile, fig)
println("Saved: ", outfile, "  (backend: ", MAKIE_BACKEND, ")")
