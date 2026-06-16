# Reproduce Figure 3 of Ribe, Faccenda & VanderBeek (2026, GJI) — the ODFTEX paper.
#
# Texture of a pure olivine aggregate deformed in PURE SHEAR (vorticity number
# Γ = 0) to a dimensionless strain τ = ε̇₀ t = 0.7, with the single active slip
# system (010)[100]. [100], [010] and [001] pole figures are shown for:
#   (a) without recrystallisation   (λ = 0)
#   (b) with    recrystallisation   (λ = 2.6)
#
# The extensional axis is horizontal. Densities are plotted as multiples of a
# uniform distribution (MUD), using Crameri's "berlin" scientific colour map
# (as in the paper).
#
# Expected, diagnostic features (paper Fig. 3):
#   • λ=0: [001] is exactly isotropic (a single active slip system gives an
#     isotropic b×n axis — a mathematically exact result).
#   • λ=2.6: [001] becomes anisotropic, and the [100]/[010] peaks split.
#
# Run with:
#   julia --project=examples/standalone examples/standalone/odftex_fig3_pureshear.jl

push!(LOAD_PATH, "@v#.#")   # allow CairoMakie/ColorSchemes from a stacked env

using DRex
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

using DRex: init_odf, odftex_step!, odftex_kinematics, strain_rate_scale,
            odf_box_weights, eulerian_transform, euler_to_dircos

# ── Parameters ────────────────────────────────────────────────────────────────
const NBOX      = 60          # Euler-grid resolution per angle
const TAU_MAX   = 0.7         # dimensionless strain ε̇₀ t
const N_STEPS   = 40
const LAMBDAS   = [0.0, 2.6]  # rows: (a) no drx, (b) drx
const AXES      = [("[100]", 1), ("[010]", 2), ("[001]", 3)]   # (label, row)

# Pure shear in the x–y plane: extension along x (horizontal), compression along
# y (vertical), z neutral. Symmetric ⇒ zero vorticity ⇒ Γ = 0, and ε̇₀ = 1.
const VG = [1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 0.0]

# ── ODF → smooth pole-figure density (multiples of uniform) ──────────────────
"""
    odf_polefigure_density(state, row; ngrid=161, σ=0.045)

Build a smooth Lambert equal-area pole-figure density for the crystallographic
axis in `row` (1=a/[100], 2=b/[010], 3=c/[001]). Each ODF box's axis direction
is projected to the disk (antipodally folded to the upper hemisphere) and
forward-splatted with a 2-D Gaussian of width `σ`; the accumulated field is then
normalised to multiples of a uniform distribution (MUD).

Returns `(xs, ys, density)` on a square grid; points outside the disk are NaN.
This is O(nbox³ + ngrid²·support) — fast enough for nbox≈60.
"""
function odf_polefigure_density(state, row::Int; ngrid::Int = 161, σ::Float64 = 0.065)
    w = vec(odf_box_weights(state))        # box probabilities (sum = 1)

    xs = range(-1.0, 1.0; length = ngrid)
    ys = range(-1.0, 1.0; length = ngrid)
    h  = step(xs)
    acc = zeros(ngrid, ngrid)
    half = ceil(Int, 3σ / h)               # 3σ Gaussian support in cells
    inv2σ2 = 1 / (2σ^2)
    gnorm = 1 / (2π * σ^2)                 # 2-D Gaussian normalisation

    # Each box carries probability mass w[k] AND represents an equal patch dg of
    # Euler space. To get a *density per unit area on the disk* we must weight by
    # the box mass only; the patch areas are equal, so the projected density of a
    # uniform ODF (w≡const) is uniform on the equal-area disk by construction.
    k = 1
    @inbounds for I in eachindex(state.pha)
        a = euler_to_dircos(state.pha[I], state.tha[I], state.psa[I])
        vx, vy, vz = a[row, 1], a[row, 2], a[row, 3]
        if vz < 0                          # antipodal fold to upper hemisphere
            vx, vy, vz = -vx, -vy, -vz
        end
        # Lambert equal-area projection of a unit vector (vx,vy,vz≥0):
        #   (X,Y) = sqrt((1-vz)/(vx²+vy²)) · (vx,vy)
        r2 = vx^2 + vy^2
        if r2 < 1e-14
            X, Y = 0.0, 0.0
        else
            pf = sqrt((1 - vz) / r2)
            X, Y = pf * vx, pf * vy
        end
        wk = w[k]; k += 1

        # nearest grid cell, then splat a normalised Gaussian over its neighbourhood
        ix = clamp(round(Int, (X + 1) / h) + 1, 1, ngrid)
        iy = clamp(round(Int, (Y + 1) / h) + 1, 1, ngrid)
        for jx in max(1, ix-half):min(ngrid, ix+half)
            dx = xs[jx] - X
            for jy in max(1, iy-half):min(ngrid, iy+half)
                dy = ys[jy] - Y
                acc[jx, jy] += wk * gnorm * exp(-(dx^2 + dy^2) * inv2σ2)
            end
        end
    end

    # Normalise to multiples of a uniform distribution: divide by the mean disk
    # density (a uniform sphere distribution projects to uniform disk density,
    # so its MUD is 1 everywhere). Reference = total mass / disk area (π).
    dens = fill(NaN, ngrid, ngrid)
    ref = 1.0 / π          # total projected mass is 1 (Σ w = 1), disk area π
    @inbounds for jx in 1:ngrid, jy in 1:ngrid
        (xs[jx]^2 + ys[jy]^2) > 1.0 && continue
        dens[jx, jy] = acc[jx, jy] / ref
    end
    return collect(xs), collect(ys), dens
end

# ── Run ODFTEX for each λ ─────────────────────────────────────────────────────
println("Running ODFTEX pure shear (Γ=0, τ=$(TAU_MAX), nbox=$(NBOX)) …")
states = map(LAMBDAS) do λ
    st = init_odf(NBOX; λ = λ, frac_opx = 0.0, s_ol = 1, ilimit = 0)
    dt = TAU_MAX / N_STEPS
    for _ in 1:N_STEPS
        odftex_step!(st, VG, dt)
    end
    @printf("  λ=%.1f : ODF integral = %.4f\n", λ, DRex.integral_drx(st))
    st
end

# ── Plot ──────────────────────────────────────────────────────────────────────
# Crameri's "berlin" scientific colour map (as used in the paper). Makie ships
# the Crameri maps, accessible by name.
const BERLIN = :berlin

function draw_polefigure!(ax, xs, ys, dens)
    cmax = maximum(filter(isfinite, dens))
    levels = range(0, max(cmax, 1.0); length = 24)
    contourf!(ax, xs, ys, dens; levels = levels, colormap = BERLIN, extendhigh = :auto)
    θs = range(0, 2π; length = 361)
    lines!(ax, cos.(θs), sin.(θs); color = :black, linewidth = 1.2)
    lines!(ax, [-1, 1], [0, 0]; color = (:white, 0.5), linewidth = 0.6)
    lines!(ax, [0, 0], [-1, 1]; color = (:white, 0.5), linewidth = 0.6)
    hidedecorations!(ax); hidespines!(ax)
    ax.aspect = DataAspect()
    limits!(ax, -1.15, 1.15, -1.15, 1.15)
    return cmax
end

fig = Figure(; size = (920, 640))
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
end
Label(fig[0, 1:3], "ODFTEX pure shear, τ = $(TAU_MAX) — extension axis horizontal";
      fontsize = 16, font = :bold)

outfile = joinpath(@__DIR__, "odftex_fig3_pureshear.png")
save(outfile, fig)
println("Saved: ", outfile, "  (backend: ", MAKIE_BACKEND, ")")
