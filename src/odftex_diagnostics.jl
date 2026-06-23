# Diagnostics for the ODFTEX continuum model.
#
# Two complementary paths (see project plan):
#  1. Native ODF diagnostics that integrate directly over the Euler-angle grid
#     (M-index, pole-figure density, mean axes), in the spirit of the paper.
#  2. A sampler that draws discrete orientation matrices distributed according to
#     the ODF, so the full suite of existing grain-based diagnostics
#     (bingham_average, misorientation_index, symmetry_pgr, poles, …) and the
#     VTK/LaMEM output plumbing can be reused unchanged.
#
# Both operate on the angle set of a chosen slip system. Pass `s=1` for the
# master/olivine frame or `s=4` for orthopyroxene (via eulerian_transform).

# ──────────────────────────────────────────────────────────────────────────────
# Box weights (consistent with integral_drx)
# ──────────────────────────────────────────────────────────────────────────────

"""
    odf_box_weights(state) -> Array{T,3}

Per-box probability weights w(i,j,k) = f · (dg / f_nodrx), normalised to sum to 1.
This is the discrete measure that `integral_drx` integrates, so sampling or
averaging with these weights is consistent with the ODF normalisation.
"""
function odf_box_weights(state::ODFState{T}) where {T<:AbstractFloat}
    nbox = state.nbox
    dcosth = T(2) / nbox
    dph    = T(π) / nbox
    dps    = T(π) / nbox
    dg     = dcosth * dph * dps

    w = Array{T,3}(undef, nbox, nbox, nbox)
    total = zero(T)
    @inbounds for k in 1:nbox, j in 1:nbox, i in 1:nbox
        wijk = dg * state.f[i,j,k] / state.f_nodrx[i,j,k]
        w[i,j,k] = wijk
        total += wijk
    end
    if total > 0
        @inbounds w ./= total
    end
    return w
end

# ──────────────────────────────────────────────────────────────────────────────
# Sampler: ODF -> discrete orientation matrices (reuse existing diagnostics/IO)
# ──────────────────────────────────────────────────────────────────────────────

"""
    sample_orientations(state, n; s=1, rng=Random.default_rng()) -> Array{T,3}

Draw `n` discrete orientation matrices distributed according to the ODF, returned
as an `n×3×3` array compatible with the existing grain-based diagnostics and the
VTK/LaMEM output path.

Each draw selects a box with probability proportional to its ODF weight (see
`odf_box_weights`) and returns the direction-cosine matrix for that box's
(advected) Euler angles, transformed to the slip-system-`s` frame.
"""
function sample_orientations(state::ODFState{T}, n::Int;
                             s::Int = 1,
                             rng::AbstractRNG = Random.default_rng()) where {T<:AbstractFloat}
    _, ph2, th2, ps2 = eulerian_transform(state, s)
    # weights follow the master grid; the transform only relabels angles, not f
    w = odf_box_weights(state)
    wflat = vec(w)
    cdf = cumsum(wflat)
    @inbounds cdf[end] = one(T)   # guard against round-off so a draw of 1.0 lands

    ph_flat = vec(ph2); th_flat = vec(th2); ps_flat = vec(ps2)

    out = Array{T,3}(undef, n, 3, 3)
    @inbounds for g in 1:n
        r = rand(rng, T)
        idx = searchsortedfirst(cdf, r)
        idx = clamp(idx, 1, length(cdf))
        a = euler_to_dircos(ph_flat[idx], th_flat[idx], ps_flat[idx])
        for i in 1:3, j in 1:3
            out[g,i,j] = a[i,j]
        end
    end
    return out
end

# ──────────────────────────────────────────────────────────────────────────────
# Native ODF diagnostics (integrate over the Euler grid)
# ──────────────────────────────────────────────────────────────────────────────

"""
    odf_mean_axis(state, axis; s=1) -> SVector{3}

ODF-weighted mean crystallographic axis ("a"=[100], "b"=[010], "c"=[001]) as a
unit vector, computed from the leading eigenvector of the weighted, antipodally
symmetric scatter matrix Σ w·(uuᵀ). `s` selects the slip-system frame.
"""
function odf_mean_axis(state::ODFState{T}, axis::String = "a"; s::Int = 1) where {T<:AbstractFloat}
    row = axis == "a" ? 1 : axis == "b" ? 2 : axis == "c" ? 3 :
        throw(ArgumentError("axis must be 'a', 'b', or 'c'"))
    _, ph2, th2, ps2 = eulerian_transform(state, s)
    w = odf_box_weights(state)

    scatter = zero(SMatrix{3,3,T,9})
    @inbounds for I in eachindex(w)
        a = euler_to_dircos(ph2[I], th2[I], ps2[I])
        u = SVector{3,T}(a[row,1], a[row,2], a[row,3])
        scatter += w[I] * (u * u')
    end
    _, vecs = eigen(Symmetric(Matrix(scatter)))
    v = vecs[:, end]
    return v ./ norm(v)
end

"""
    odf_pole_figure(state, axis; s=1, nθ=181, nφ=361) -> (θ, φ, density)

Pole-figure density for the given crystallographic axis, accumulated on a
(colatitude θ ∈ [0,π], azimuth φ ∈ [0,2π]) grid by binning each box's axis
direction with its ODF weight. Returns the bin-center vectors and a density
matrix (multiples of a uniform distribution, area-weighted).

Lightweight histogram intended for plotting; for publication-quality smoothed
pole figures sample orientations and use the existing `poles`/`lambert_equal_area`
machinery instead.
"""
function odf_pole_figure(state::ODFState{T}, axis::String = "a";
                         s::Int = 1, nθ::Int = 181, nφ::Int = 361) where {T<:AbstractFloat}
    row = axis == "a" ? 1 : axis == "b" ? 2 : axis == "c" ? 3 :
        throw(ArgumentError("axis must be 'a', 'b', or 'c'"))
    _, ph2, th2, ps2 = eulerian_transform(state, s)
    w = odf_box_weights(state)

    density = zeros(T, nθ, nφ)
    dθ = T(π) / (nθ - 1)
    dφ = T(2π) / (nφ - 1)
    @inbounds for I in eachindex(w)
        a = euler_to_dircos(ph2[I], th2[I], ps2[I])
        # antipodal symmetry: fold to the upper hemisphere
        z = a[row,3]
        ux, uy, uz = z < 0 ? (-a[row,1], -a[row,2], -z) : (a[row,1], a[row,2], a[row,3])
        θ = acos(clamp(uz, -one(T), one(T)))
        φ = mod(atan(uy, ux), T(2π))
        iθ = clamp(round(Int, θ / dθ) + 1, 1, nθ)
        iφ = clamp(round(Int, φ / dφ) + 1, 1, nφ)
        density[iθ, iφ] += w[I]
    end

    θ = collect(range(zero(T), T(π); length = nθ))
    φ = collect(range(zero(T), T(2π); length = nφ))
    return θ, φ, density
end

"""
    odf_misorientation_index(state; s=1, n=10000, system=orthorhombic, rng=...) -> Float64

M-index for the ODF. Computed by sampling `n` orientations from the ODF and
reusing the existing grain-based `misorientation_index`. (A fully native
quadrature over the Euler grid is possible but the sampled estimate matches the
discrete-grain diagnostic used elsewhere in DRex, which is the point of the
"both" approach.)
"""
function odf_misorientation_index(state::ODFState{T};
                                  s::Int = 1, n::Int = 10000,
                                  system::LatticeSystem = orthorhombic,
                                  rng::AbstractRNG = Random.default_rng()) where {T<:AbstractFloat}
    ori = sample_orientations(state, n; s = s, rng = rng)
    return misorientation_index(ori, system)
end

# ──────────────────────────────────────────────────────────────────────────────
# Pole-figure density (Lambert equal-area, multiples of uniform) — plotting-free
# ──────────────────────────────────────────────────────────────────────────────

"""
    odf_polefigure_density(state, row; s=1, ngrid=161, σ=0.065)
        -> (xs, ys, density)

Smooth Lambert equal-area pole-figure density for the crystallographic axis in
`row` (1=a/[100], 2=b/[010], 3=c/[001]), in multiples of a uniform distribution
(MUD). Each ODF box's axis direction is projected to the disk (antipodally folded
to the upper hemisphere) and forward-splatted with a 2-D Gaussian of width `σ`.

Two pole-figure singularities are handled so a uniform ODF projects to a flat
field everywhere: near-rim (equator) kernel mass that would spill outside the
disk is re-entered at the antipodal azimuth, and near-pole axes are spread around
a small azimuthal ring. `density` is `NaN` outside the unit disk. `s` selects the
slip-system frame (1=olivine master, 4=orthopyroxene).

This is a plotting-free numeric routine; pass the result to any contour plotter.
"""
function odf_polefigure_density(state::ODFState{T}, row::Int;
                                s::Int = 1, ngrid::Int = 161,
                                σ::Real = 0.065) where {T<:AbstractFloat}
    1 <= row <= 3 || throw(ArgumentError("row must be 1, 2 or 3"))
    _, ph2, th2, ps2 = eulerian_transform(state, s)
    w = vec(odf_box_weights(state))

    σf = float(σ)
    xs = range(-1.0, 1.0; length = ngrid)
    ys = range(-1.0, 1.0; length = ngrid)
    h  = step(xs)
    acc = zeros(ngrid, ngrid)
    half = ceil(Int, 3σf / h)
    inv2σ2 = 1 / (2σf^2)
    gnorm  = 1 / (2π * σf^2)

    splat! = (X, Y, wk) -> begin
        ix = clamp(round(Int, (X + 1) / h) + 1, 1, ngrid)
        iy = clamp(round(Int, (Y + 1) / h) + 1, 1, ngrid)
        @inbounds for jx in max(1, ix-half):min(ngrid, ix+half)
            dx = xs[jx] - X
            for jy in max(1, iy-half):min(ngrid, iy+half)
                dy = ys[jy] - Y
                acc[jx, jy] += wk * gnorm * exp(-(dx^2 + dy^2) * inv2σ2)
            end
        end
    end

    @inbounds for I in eachindex(ph2)
        a = euler_to_dircos(ph2[I], th2[I], ps2[I])
        vx, vy, vz = a[row, 1], a[row, 2], a[row, 3]
        if vz < 0
            vx, vy, vz = -vx, -vy, -vz
        end
        r2 = vx^2 + vy^2
        wk = w[I]

        if r2 < 1e-10
            Rp = sqrt(max(0.0, 1 - vz))
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

        R2 = X^2 + Y^2
        if R2 > (1 - 3σf)^2
            rscale = sqrt(max(0.0, 2 - R2)) / sqrt(max(R2, 1e-30))
            splat!(-X * rscale, -Y * rscale, wk)
        end
    end

    dens = fill(NaN, ngrid, ngrid)
    ref = 1.0 / π                  # total mass ≈ 1 (Σ w = 1), disk area π
    @inbounds for jx in 1:ngrid, jy in 1:ngrid
        (xs[jx]^2 + ys[jy]^2) > 1.0 && continue
        dens[jx, jy] = acc[jx, jy] / ref
    end
    return collect(xs), collect(ys), dens
end

# ──────────────────────────────────────────────────────────────────────────────
# Finite strain ellipsoid
# ──────────────────────────────────────────────────────────────────────────────

"""
    fse_long_axis(deformation_gradient) -> SVector{3}

Unit long axis of the finite strain ellipsoid for a deformation gradient `F`:
the leading eigenvector of the left Cauchy–Green tensor B = F Fᵀ.
"""
function fse_long_axis(F::AbstractMatrix{T}) where {T<:AbstractFloat}
    B = F * transpose(F)
    _, vecs = eigen(Symmetric(Matrix(B)))
    v = vecs[:, end]
    return SVector{3,T}(v[1], v[2], v[3]) ./ norm(v)
end

"""
    odf_expectation_angle(state, axis_row; plane_dir1=1, plane_dir2=2, s=1) -> Float64

ODF-weighted **expectation value** of the in-plane orientation angle (degrees, in
[0,90]) of a crystallographic axis relative to a reference direction within a
deformation plane. This is the quantity plotted by the paper's Fig. 6
("expectation values of the [100] orientation"): the axis is projected into the
plane spanned by Cartesian axes `plane_dir1`/`plane_dir2`, and the acute angle of
that projection from `plane_dir1` is averaged over the ODF —
`∫ |angle(g)| f(g) dg / ∫ f(g) dg` — rather than taking the dominant axis
(`odf_mean_axis`).

For the paper's simple-shear setup the deformation plane is X–Y (`plane_dir1=1`,
`plane_dir2=2`) and `plane_dir1` (X) is the shear direction, so the reported
angle is the [100] orientation relative to the shear plane.

- `axis_row` — 1=a/[100], 2=b/[010], 3=c/[001]
- `s`        — slip-system frame (1=olivine master)

For an isotropic ODF the in-plane expectation angle is 45° (the mean acute angle
of uniformly distributed in-plane axes from a reference direction).
"""
function odf_expectation_angle(state::ODFState{T}, axis_row::Int;
                               plane_dir1::Int = 1, plane_dir2::Int = 2,
                               s::Int = 1) where {T<:AbstractFloat}
    1 <= axis_row <= 3 || throw(ArgumentError("axis_row must be 1, 2 or 3"))
    _, ph2, th2, ps2 = eulerian_transform(state, s)
    w = vec(odf_box_weights(state))

    num = zero(T); den = zero(T)
    @inbounds for I in eachindex(ph2)
        a = euler_to_dircos(ph2[I], th2[I], ps2[I])
        # in-plane projection components (axial: fold to acute angle in [0,90])
        p1 = a[axis_row, plane_dir1]
        p2 = a[axis_row, plane_dir2]
        ang = abs(atand(p2, p1))
        ang > 90 && (ang = 180 - ang)
        num += w[I] * ang
        den += w[I]
    end
    return den > 0 ? num / den : T(NaN)
end
