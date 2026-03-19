# Statistical methods for orientation and elasticity data.

"""
    resample_orientations(orientations, fractions; n_samples=nothing, seed=nothing)

Return new samples from `orientations` weighted by the volume distribution.

Accepts either:
- `orientations::Vector{Array{Float64,3}}`, `fractions::Vector{Vector{Float64}}`
  (list of N×3×3 arrays, list of N-vectors — matches Python API)
- `orientations::AbstractArray{Float64,4}`, `fractions::AbstractArray{Float64,2}`
  (P×M×3×3 and P×M arrays)
"""
function resample_orientations(
    orientations::Vector{<:AbstractArray{Float64,3}},
    fractions::Vector{<:AbstractVector{Float64}};
    n_samples::Union{Int,Nothing}=nothing,
    seed::Union{Int,Nothing}=nothing,
)
    P = length(orientations)
    @assert length(fractions) == P "orientation and fraction lists must have the same length"
    for p in 1:P
        @assert size(orientations[p], 1) == length(fractions[p]) "orientation/fraction size mismatch"
    end

    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    M = size(orientations[1], 1)
    ns = n_samples === nothing ? M : n_samples

    out_orientations = Vector{Array{Float64,3}}(undef, P)
    out_fractions = Vector{Vector{Float64}}(undef, P)

    for i in 1:P
        frac = fractions[i]
        orient = orientations[i]
        M_i = length(frac)
        sort_idx = sortperm(frac)
        frac_sorted = frac[sort_idx]
        cumfrac = cumsum(frac_sorted)
        cumfrac[end] = 1.0
        samples = rand(rng, ns)
        count_less = searchsortedlast.(Ref(cumfrac), samples)
        count_less = clamp.(count_less, 1, M_i)
        out_o = Array{Float64,3}(undef, ns, 3, 3)
        out_f = Vector{Float64}(undef, ns)
        for s in 1:ns
            idx = sort_idx[count_less[s]]
            out_o[s, :, :] .= orient[idx, :, :]
            out_f[s] = frac[idx]
        end
        out_orientations[i] = out_o
        out_fractions[i] = out_f
    end
    return out_orientations, out_fractions
end

"""
    _scatter_matrix(orientations, row)

Lower-triangular scatter (inertia) matrix for orientation data.
`orientations` is Nx3x3, `row` is 1, 2, or 3 (for a, b, c axis).
"""
function _scatter_matrix(orientations::AbstractArray{Float64,3}, row::Int)
    scatter = zeros(3, 3)
    n = size(orientations, 1)
    @inbounds for g in 1:n
        scatter[1,1] += orientations[g, row, 1]^2
        scatter[2,2] += orientations[g, row, 2]^2
        scatter[3,3] += orientations[g, row, 3]^2
        scatter[2,1] += orientations[g, row, 1] * orientations[g, row, 2]
        scatter[3,1] += orientations[g, row, 1] * orientations[g, row, 3]
        scatter[3,2] += orientations[g, row, 2] * orientations[g, row, 3]
    end
    return scatter
end

"""
    misorientation_hist(orientations, system; bins=nothing)

Calculate misorientation histogram for polycrystal orientations.
Returns (counts, bin_edges) as from a normalised histogram.

Uses only proper rotation symmetry operations (not reflections) to match the
theoretical random distribution of Skemer et al. (2005).  Since proper rotations
form a group, the minimum over cross-combinations reduces to iterating only nsym
per pair (fix one side, vary the other).

The inner loop is allocation-free: symmetry-applied quaternions are precomputed
per grain (n × nsym tuples) and the histogram is built on-the-fly.
"""
function misorientation_hist(orientations::AbstractArray{Float64,3}, system::LatticeSystem;
                             bins::Union{Int,Nothing}=nothing)
    all_ops = symmetry_operations(system)
    # Keep only proper rotation ops (quaternion vectors); discard reflections (matrices).
    rot_ops = [op for op in all_ops if !(op isa AbstractMatrix)]
    n = size(orientations, 1)
    nsym = length(rot_ops)

    # Convert orientations to quaternions (scalar-last [x,y,z,w])
    quats = Vector{NTuple{4,Float64}}(undef, n)
    for i in 1:n
        R = @view orientations[i, :, :]
        q = _rotation_matrix_to_quat(R)
        quats[i] = (q[1], q[2], q[3], q[4])
    end

    # Precompute symmetry-applied quaternions for every grain.
    # sym_quats[g, s] = rot_ops[s] ∘ quats[g] via quaternion product.
    sym_quats = Matrix{NTuple{4,Float64}}(undef, n, nsym)
    for (s, qs) in enumerate(rot_ops)
        x1, y1, z1, w1 = qs
        @inbounds for g in 1:n
            x2, y2, z2, w2 = quats[g]
            sym_quats[g, s] = (
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
            )
        end
    end

    θmax = _max_misorientation(system)
    nbins = bins === nothing ? θmax : bins
    bin_width = θmax / nbins
    counts = zeros(nbins)
    npairs = 0

    # Compute minimum misorientation angle per pair and bin on-the-fly.
    # Because proper rotations form a group, min_{a,b} angle(S_a q_i, S_b q_j)
    # = min_s angle(S_s q_i, q_j), so we fix q_j and vary only the sym op on q_i.
    @inbounds for i in 1:n
        for j in (i+1):n
            q2 = quats[j]
            min_angle = Inf
            for s in 1:nsym
                q1 = sym_quats[i, s]
                dot_val = q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3] + q1[4]*q2[4]
                angle = 2 * rad2deg(acos(abs(clamp(dot_val, -1.0, 1.0))))
                if angle < min_angle
                    min_angle = angle
                end
            end
            bin = clamp(Int(floor(min_angle / bin_width)) + 1, 1, nbins)
            counts[bin] += 1
            npairs += 1
        end
    end

    total = npairs * bin_width
    counts ./= total
    return counts, collect(range(0, θmax, length=nbins+1))
end

"""Quaternion from rotation matrix (scalar-last [x,y,z,w] convention)."""
function _rotation_matrix_to_quat(R::AbstractMatrix)
    # Use the Shepperd method
    t = tr(R)
    if t > 0
        s = 0.5 / sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[3,2] - R[2,3]) * s
        y = (R[1,3] - R[3,1]) * s
        z = (R[2,1] - R[1,2]) * s
    elseif R[1,1] > R[2,2] && R[1,1] > R[3,3]
        s = 2.0 * sqrt(1.0 + R[1,1] - R[2,2] - R[3,3])
        w = (R[3,2] - R[2,3]) / s
        x = 0.25 * s
        y = (R[1,2] + R[2,1]) / s
        z = (R[1,3] + R[3,1]) / s
    elseif R[2,2] > R[3,3]
        s = 2.0 * sqrt(1.0 + R[2,2] - R[1,1] - R[3,3])
        w = (R[1,3] - R[3,1]) / s
        x = (R[1,2] + R[2,1]) / s
        y = 0.25 * s
        z = (R[2,3] + R[3,2]) / s
    else
        s = 2.0 * sqrt(1.0 + R[3,3] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = (R[1,3] + R[3,1]) / s
        y = (R[2,3] + R[3,2]) / s
        z = 0.25 * s
    end
    q = [x, y, z, w]
    return q ./ norm(q)
end

"""
    misorientations_random(low, high, system)

Expected count of random misorientation angles in (low, high) degrees.
"""
function misorientations_random(low::Real, high::Real, system::LatticeSystem)
    M, N = lattice_system_values(system)
    max_θ = _max_misorientation(system)
    (0 <= low <= high <= max_θ) || throw(ArgumentError(
        "bounds must obey low ∈ [0, high) and high ≤ $max_θ. Got ($low, $high)."))

    a = tan(deg2rad(90 / M))
    b = 2 * rad2deg(atan(sqrt(1 + a^2)))
    c = round(2 * rad2deg(atan(sqrt(1 + 2 * a^2))))

    counts = zeros(2)
    for (idx, edgeval) in enumerate([low, high])
        d = deg2rad(edgeval)
        if 0 <= edgeval <= 180 / M
            counts[idx] += (N / 180) * (1 - cos(d))
        elseif 180 / M <= edgeval <= 180 * M / N
            counts[idx] += (N / 180) * a * sin(d)
        elseif 90 <= edgeval <= b
            counts[idx] += (M / 90) * ((M + a) * sin(d) - M * (1 - cos(d)))
        elseif b <= edgeval <= c
            ν = tan(deg2rad(edgeval / 2))^2
            counts[idx] = (M / 90) * (
                (M + a) * sin(d) - M * (1 - cos(d)) +
                (M / 180) * (
                    (1 - cos(d)) * (
                        rad2deg(acos((1 - ν * cos(deg2rad(180 / M))) / (ν - 1))) +
                        2 * rad2deg(acos(a / (sqrt(ν - a^2) * sqrt(ν - 1))))
                    ) -
                    2 * sin(d) * (
                        2 * rad2deg(acos(a / sqrt(ν - 1))) +
                        a * rad2deg(acos(1 / sqrt(ν - a^2)))
                    )
                )
            )
        end
    end
    return sum(counts) / 2
end

"""Maximum misorientation angle for the given lattice system."""
function _max_misorientation(system::LatticeSystem)
    if system == orthorhombic || system == rhombohedral
        return 120
    elseif system == tetragonal || system == hexagonal
        return 90
    elseif system == triclinic || system == monoclinic
        return 180
    end
    throw(ArgumentError("unsupported lattice system: $system"))
end
