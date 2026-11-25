# Reproduce Figure 3 of Ribe, Faccenda & VanderBeek (2026, GJI) — the ODFTEX paper.
#
# Texture of a pure olivine aggregate deformed in PURE SHEAR (vorticity number
# Γ = 0) to a dimensionless strain τ = ε̇₀ t = 0.7, with the single active slip
# system (010)[100]. [100], [010] and [001] pole figures are shown for:
#   (a) without recrystallisation   (λ = 0)
#   (b) with    recrystallisation   (λ = 2.6)
#
# The extensional axis is horizontal. Densities are plotted as multiples of a
# uniform distribution (MUD), using Crameri's "berlin" scientific colour map.
#
# Diagnostic features (paper Fig. 3):
#   • λ=0: [001] is exactly isotropic (single active slip system ⇒ isotropic
#     b×n axis — a mathematically exact result).
#   • λ=2.6: [001] becomes anisotropic, and the [100]/[010] peaks split.
#
# Run with:
#   julia --project=examples/standalone examples/standalone/odftex_fig3_pureshear.jl

push!(LOAD_PATH, "@v#.#")   # allow CairoMakie from a stacked env

using MantleFabric
using LinearAlgebra
using Printf

# Prefer CairoMakie (headless PNG); fall back to GLMakie.
const MAKIE_BACKEND = try
    @eval using CairoMakie
    :cairo
catch
    @eval using GLMakie
    :gl
end

using MantleFabric: init_odf, odftex_step!

# Shared pole-figure density + drawing utilities (load after the Makie backend).
include(joinpath(@__DIR__, "odftex_polefigure_utils.jl"))

# ── Parameters ────────────────────────────────────────────────────────────────
const NBOX      = 90          # Euler-grid resolution per angle (denser ⇒ cleaner poles)
const TAU_MAX   = 0.7         # dimensionless strain ε̇₀ t
const N_STEPS   = 40
const LAMBDAS   = [0.0, 2.6]  # rows: (a) no drx, (b) drx
const AXES      = [("[100]", 1), ("[010]", 2), ("[001]", 3)]   # (label, row)

# Pure shear in the x–y plane: extension along x (horizontal), compression along
# y (vertical), z neutral. Symmetric ⇒ zero vorticity ⇒ Γ = 0, and ε̇₀ = 1.
const VG = [1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 0.0]

# ── Run ODFTEX for each λ ─────────────────────────────────────────────────────
println("Running ODFTEX pure shear (Γ=0, τ=$(TAU_MAX), nbox=$(NBOX)) …")
states = map(LAMBDAS) do λ
    st = init_odf(NBOX; λ = λ, frac_opx = 0.0, s_ol = 1, ilimit = 0)
    dt = TAU_MAX / N_STEPS
    for _ in 1:N_STEPS
        odftex_step!(st, VG, dt)
    end
    @printf("  λ=%.1f : ODF integral = %.4f\n", λ, MantleFabric.integral_drx(st))
    st
end

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = Figure(; size = (1040, 640))
rowlabels = ["(a) λ = 0", "(b) λ = 2.6"]
for (ri, st) in enumerate(states)
    for (ci, (label, row)) in enumerate(AXES)
        xs, ys, dens = odf_polefigure_density(st, row)
        ax = Axis(fig[ri, ci]; title = ri == 1 ? label : "")
        cmax = draw_polefigure!(ax, xs, ys, dens)
        if ci == 1
            Label(fig[ri, 0], rowlabels[ri]; rotation = π/2, tellheight = false,
                  fontsize = 14, font = :bold)
        end
        @printf("  panel λ=%.1f %s : max density = %.2f MUD\n", LAMBDAS[ri], label, cmax)
    end
    add_row_colorbar!(fig, ri, length(AXES) + 1)
end
Label(fig[0, 1:length(AXES)], "ODFTEX pure shear, τ = $(TAU_MAX) — extension axis horizontal";
      fontsize = 16, font = :bold)

outfile = joinpath(@__DIR__, "odftex_fig3_pureshear.png")
save(outfile, fig)
println("Saved: ", outfile, "  (backend: ", MAKIE_BACKEND, ")")
