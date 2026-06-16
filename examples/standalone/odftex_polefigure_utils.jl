# Shared pole-figure PLOTTING helpers for the ODFTEX paper-reproduction examples
# (Ribe, Faccenda & VanderBeek 2026, GJI).
#
# `include` this from a figure script *after* a Makie backend is loaded. The
# numeric pole-figure density and FSE long axis live in DRex itself
# (odf_polefigure_density, fse_long_axis); this file only adds the Makie drawing
# helpers and the paper's fixed colour scale.
#
# Conventions match the paper: equal-area (Lambert) projection, axial/antipodal
# symmetry, density in multiples of a uniform distribution (MUD), Crameri's
# "berlin" colour map, fixed 0–3.5 colour range with 12 filled-contour bands.

using DRex: odf_polefigure_density, fse_long_axis

# ── Plotting constants (paper Fig. 3/4 style) ────────────────────────────────
const BERLIN  = :berlin
const CMIN    = 0.0
const CMAX    = 3.5
const NBANDS  = 12                              # filled-contour bands (counted in paper Fig. 3/4)
const CLEVELS = range(CMIN, CMAX; length = NBANDS + 1)
const CTICKS  = 0:0.5:CMAX                      # colorbar labelled every 0.5 MUD

"""
    draw_polefigure!(ax, xs, ys, dens) -> max density

Draw a filled-contour pole figure on `ax` with the fixed paper colour scale.
Returns the panel's maximum density (MUD).
"""
function draw_polefigure!(ax, xs, ys, dens)
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

"""
    draw_fse_line!(ax, fse)

Draw the finite-strain-ellipsoid long axis `fse` as a white diameter on a pole
figure (X horizontal, Y vertical), projected into the X–Y plane.
"""
function draw_fse_line!(ax, fse)
    dx, dy = fse[1], fse[2]
    r = hypot(dx, dy)
    r < 1e-9 && return
    dx /= r; dy /= r
    lines!(ax, [-dx, dx], [-dy, dy]; color = :white, linewidth = 3)
end

"""
    add_row_colorbar!(fig, row, col) — fixed 0–3.5 colorbar at fig[row, col].
"""
function add_row_colorbar!(fig, row::Int, col::Int)
    Colorbar(fig[row, col];
             colormap = BERLIN, colorrange = (CMIN, CMAX),
             label = "density (mud)", ticks = CTICKS, height = Relative(0.85))
end
