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
const NBOX      = 90          # Euler-grid resolution per angle (denser ⇒ cleaner poles)
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

    # Splat each box's axis as a normalised Gaussian on the disk. For axial data
    # on an equal-area pole figure the disk rim is the equator, a real great
    # circle: kernel mass from a near-rim point that would fall *outside* the disk
    # belongs to the antipodal azimuth on the rim. We therefore splat each box at
    # its disk point AND at the rim-mirror image (X,Y) → -(X,Y)·(2-R²)/R² of the
    # part that spills out, which keeps the projected density continuous across
    # the equator (so a uniform ODF stays flat right out to the edge).
    #
    # Each box represents an equal patch dg of Euler space, so weighting by the
    # box mass w[k] alone gives density per unit disk area; a uniform ODF then
    # projects to uniform disk density by construction.
    function splat!(X, Y, wk)
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
        wk = w[k]; k += 1

        if r2 < 1e-10
            # Near the pole the azimuth is ill-defined; spread the splat evenly
            # around a tiny ring at the projected radius so the centre stays flat
            # (avoids a single-pixel artifact at the origin).
            Rp = sqrt(max(0.0, 1 - vz))      # Lambert radius at this colatitude
            nring = 12
            for t in 0:(nring-1)
                ang = 2π * t / nring
                splat!(Rp * cos(ang), Rp * sin(ang), wk / nring)
            end
            continue
        end

        pf = sqrt((1 - vz) / r2)
        X, Y = pf * vx, pf * vy

        splat!(X, Y, wk)

        # Rim-continuity ghost: if the point is near the equator (disk radius
        # close to 1), also splat its antipode reflected across the rim so the
        # kernel mass that spills past the edge re-enters at the opposite side.
        R2 = X^2 + Y^2
        if R2 > (1 - 3σ)^2          # within ~3σ of the rim
            # antipodal direction (vx,vy,-vz) → upper-hemisphere fold (−vx,−vy,vz):
            # its Lambert point is the rim reflection of (X,Y).
            rscale = sqrt(max(0.0, 2 - R2)) / sqrt(max(R2, 1e-30))
            splat!(-X * rscale, -Y * rscale, wk)
        end
    end

    # Normalise to multiples of a uniform distribution. A uniform sphere
    # distribution projects to uniform disk density, MUD = 1 everywhere; the
    # reference is total projected mass / disk area (π). The ghost splats double
    # only the near-rim mass, which the same construction in the uniform case
    # also doubles, so the MUD reference is unchanged in the disk interior.
    dens = fill(NaN, ngrid, ngrid)
    ref = 1.0 / π          # total mass ≈ 1 (Σ w = 1), disk area π
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
const BERLIN  = :berlin
const CMIN     = 0.0       # fixed colour range, as in the paper
const CMAX     = 3.5
const NBANDS   = 11        # filled-contour bands counted in the paper's Fig. 3
const CLEVELS  = range(CMIN, CMAX; length = NBANDS + 1)   # band edges
const CTICKS   = 0:0.5:CMAX   # colorbar labelled every 0.5 MUD, as in the paper

function draw_polefigure!(ax, xs, ys, dens)
    # Fixed level set 0–CMAX fixes the colour range for filled contours;
    # densities above CMAX are clamped to the top colour (extendhigh).
    contourf!(ax, xs, ys, dens;
              levels = CLEVELS, colormap = BERLIN, extendhigh = :auto)
    θs = range(0, 2π; length = 361)
    lines!(ax, cos.(θs), sin.(θs); color = :black, linewidth = 1.2)
    lines!(ax, [-1, 1], [0, 0]; color = (:white, 0.5), linewidth = 0.6)
    lines!(ax, [0, 0], [-1, 1]; color = (:white, 0.5), linewidth = 0.6)
    hidedecorations!(ax); hidespines!(ax)
    ax.aspect = DataAspect()
    limits!(ax, -1.15, 1.15, -1.15, 1.15)
    return maximum(filter(isfinite, dens))
end

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
    # One colorbar per row (fixed 0–3.5 range), matching the paper layout.
    Colorbar(fig[ri, length(AXES) + 1];
             colormap = BERLIN, colorrange = (CMIN, CMAX),
             label = "density (mud)",
             ticks = CTICKS, height = Relative(0.85))
end
Label(fig[0, 1:length(AXES)], "ODFTEX pure shear, τ = $(TAU_MAX) — extension axis horizontal";
      fontsize = 16, font = :bold)

outfile = joinpath(@__DIR__, "odftex_fig3_pureshear.png")
save(outfile, fig)
println("Saved: ", outfile, "  (backend: ", MAKIE_BACKEND, ")")
