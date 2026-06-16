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
