# Utility functions for DRex.

"""
    strain_increment(dt, velocity_gradient)

Calculate strain increment for a given time increment and velocity gradient.
Returns tensorial strain increment ε = γ/2.
"""
function strain_increment(dt::Real, velocity_gradient::AbstractMatrix{Float64})
    sym = (velocity_gradient .+ velocity_gradient') ./ 2
    return abs(dt) * maximum(abs.(eigvals(Symmetric(sym))))
end

"""
    apply_gbs!(orientations, fractions, gbs_threshold, orientations_prev, n_grains)

Apply grain boundary sliding for small grains. Modifies orientations and fractions in-place.
"""
function apply_gbs!(
    orientations::AbstractArray{Float64,3},
    fractions::AbstractVector{Float64},
    gbs_threshold::Float64,
    orientations_prev::AbstractArray{Float64,3},
    n_grains::Int
)
    threshold = gbs_threshold / n_grains
    @inbounds for g in 1:n_grains
        if fractions[g] < threshold
            for i in 1:3, j in 1:3
                orientations[g,i,j] = orientations_prev[g,i,j]
            end
            fractions[g] = threshold
        end
    end
    s = sum(fractions)
    fractions ./= s
    return orientations, fractions
end

"""
    extract_vars(y, n_grains)

Extract deformation gradient, orientation matrices and grain fractions from flat ODE state vector.
"""
function extract_vars(y::AbstractVector{Float64}, n_grains::Int)
    deformation_gradient = reshape(y[1:9], 3, 3)
    orientations_flat = @view y[10:n_grains*9+9]
    orientations = reshape(copy(orientations_flat), n_grains, 3, 3)
    clamp!(orientations, -1.0, 1.0)
    fractions = copy(@view y[n_grains*9+10:n_grains*10+9])
    clamp!(fractions, 0.0, Inf)
    s = sum(fractions)
    fractions ./= s
    return deformation_gradient, orientations, fractions
end

"""
    quat_product(q1, q2)

Hamilton product of two quaternions in [x,y,z,w] (scalar-last) convention.
"""
function quat_product(q1::AbstractVector, q2::AbstractVector)
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ]
end

"""
    remove_nans(a)

Remove NaN values from an array.
"""
function remove_nans(a::AbstractArray)
    return filter(!isnan, a)
end

"""
    pad_with(a; x=NaN)

Pad a vector of vectors with `x` to form a regular matrix.
"""
function pad_with(a; x=NaN)
    longest = maximum(length.(a))
    out = fill(x, length(a), longest)
    for (i, d) in enumerate(a)
        out[i, 1:length(d)] .= d
    end
    return out
end

"""
    add_dim(a, dim; val=0.0)

Insert `val` at 1-based position `dim` in a vector, extending it by one element.
"""
function add_dim(a::AbstractVector, dim::Int; val=0.0)
    return vcat(a[1:dim-1], [oftype(float(first(a)), val)], a[dim:end])
end

"""
    default_ncpus()

Return number of available CPU threads.
"""
default_ncpus() = Threads.nthreads()
