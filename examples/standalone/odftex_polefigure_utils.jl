# Shared pole-figure PLOTTING helpers for the ODFTEX paper-reproduction examples
# (Ribe, Faccenda & VanderBeek 2026, GJI).
#
# `include` this from a figure script *after* a Makie backend is loaded. The
# numeric pole-figure density and FSE long axis live in MantleFabric itself
# (odf_polefigure_density, fse_long_axis); this file only adds the Makie drawing
# helpers and the paper's fixed colour scale.
#
# Conventions match the paper: equal-area (Lambert) projection, axial/antipodal
# symmetry, density in multiples of a uniform distribution (MUD), Crameri's
# "berlin" colour map, fixed 0–3.5 colour range with 12 filled-contour bands.

using MantleFabric: odf_polefigure_density, fse_long_axis

# ── Plotting constants (paper style) ─────────────────────────────────────────
# Crameri's "berlin" scientific colour map. Most figures use a fixed 0–3.5 MUD
# range with 12 filled-contour bands (counted in the paper Fig. 3/4); Fig. 5
# uses 0–6, so the range is a keyword on the drawing helpers.
const BERLIN  = :berlin
const CMIN    = 0.0
const CMAX    = 3.5
const NBANDS  = 12

"""
    draw_polefigure!(ax, xs, ys, dens; cmin=0.0, cmax=3.5, nbands=12) -> max density

Draw a filled-contour pole figure on `ax` with a fixed colour scale spanning
`cmin`–`cmax` in `nbands` bands. Returns the panel's maximum density (MUD).
"""
function draw_polefigure!(ax, xs, ys, dens; cmin = CMIN, cmax = CMAX, nbands = NBANDS)
    levels = range(cmin, cmax; length = nbands + 1)
    contourf!(ax, xs, ys, dens;
              levels = levels, colormap = BERLIN, extendhigh = :auto)
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
    draw_axis_line!(ax, dir; color, linewidth=3)

Draw an in-plane axis `dir` (3-vector) as a full diameter (±direction) on a pole
figure (X horizontal, Y vertical). Used for the shear-plane / max-extension / FSE
reference lines of paper Fig. 5.
"""
function draw_axis_line!(ax, dir; color, linewidth = 3)
    dx, dy = dir[1], dir[2]
    r = hypot(dx, dy)
    r < 1e-9 && return
    dx /= r; dy /= r
    lines!(ax, [-dx, dx], [-dy, dy]; color = color, linewidth = linewidth)
end

"""
    draw_fse_line!(ax, fse) — white FSE long-axis diameter (Fig. 4 convenience).
"""
draw_fse_line!(ax, fse) = draw_axis_line!(ax, fse; color = :white)

"""
    add_row_colorbar!(fig, row, col; cmin=0.0, cmax=3.5, ticks=0:0.5:3.5)

Add a colorbar at `fig[row, col]` over the fixed `cmin`–`cmax` range.
"""
function add_row_colorbar!(fig, row::Int, col::Int;
                           cmin = CMIN, cmax = CMAX, ticks = 0:0.5:CMAX)
    Colorbar(fig[row, col];
             colormap = BERLIN, colorrange = (cmin, cmax),
             label = "density (mud)", ticks = ticks, height = Relative(0.85))
end
