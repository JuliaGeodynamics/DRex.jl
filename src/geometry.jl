# Geometric coordinate conversions and projections.

"""Crystallographic lattice systems supported by postprocessing methods."""
@enum LatticeSystem begin
    triclinic    # (1, 1), θmax = 180
    monoclinic   # (2, 2), θmax = 180
    orthorhombic # (2, 4), θmax = 120
    rhombohedral # (3, 6), θmax = 120
    tetragonal   # (4, 8), θmax = 90
    hexagonal    # (6, 12), θmax = 90
end

"""Return (a, b) pair for a lattice system as in Grimmer 1979 Table 1."""
function lattice_system_values(s::LatticeSystem)
    s == triclinic    && return (1, 1)
    s == monoclinic   && return (2, 2)
    s == orthorhombic && return (2, 4)
    s == rhombohedral && return (3, 6)
    s == tetragonal   && return (4, 8)
    s == hexagonal    && return (6, 12)
    throw(ArgumentError("unsupported lattice system: $s"))
end

"""
    to_cartesian(ϕ, θ; r=1.0)

Convert spherical to Cartesian coordinates.
ϕ = longitude (azimuth), θ = colatitude (inclination).
"""
function to_cartesian(ϕ::AbstractVector, θ::AbstractVector; r=ones(length(ϕ)))
    r = r isa Number ? fill(Float64(r), length(ϕ)) : Float64.(r)
    x = r .* sin.(θ) .* cos.(ϕ)
    y = r .* sin.(θ) .* sin.(ϕ)
    z = r .* cos.(θ)
    return x, y, z
end

function to_cartesian(ϕ::Real, θ::Real; r::Real=1.0)
    x = r * sin(θ) * cos(ϕ)
    y = r * sin(θ) * sin(ϕ)
    z = r * cos(θ)
    return x, y, z
end

"""
    to_spherical(x, y, z)

Convert Cartesian to spherical coordinates. Returns (r, ϕ, θ).
"""
function to_spherical(x, y, z)
    r = sqrt.(x.^2 .+ y.^2 .+ z.^2)
    ϕ = atan.(y, x)
    θ = sign.(y) .* acos.(x ./ sqrt.(x.^2 .+ y.^2))
    return r, ϕ, θ
end

"""
    misorientation_angles(q1_array, q2_array)

Calculate minimum misorientation angles for collections of rotation quaternions.
q1_array has shape (N, A, 4), q2_array has shape (N, B, 4).
Returns N-element vector of minimum misorientation angles in degrees.
"""
function misorientation_angles(q1_array::AbstractArray{T,3}, q2_array::AbstractArray{T,3}) where {T}
    N = size(q1_array, 1)
    A = size(q1_array, 2)
    B = size(q2_array, 2)
    @assert size(q2_array, 1) == N
    angles = Matrix{T}(undef, N, A * B)
    k = 0
    for i in 1:A, j in 1:B
        k += 1
        @inbounds for n in 1:N
            dot_val = zero(T)
            for d in 1:4
                dot_val += q1_array[n, i, d] * q2_array[n, j, d]
            end
            angles[n, k] = 2 * rad2deg(acos(abs(clamp(dot_val, -one(T), one(T)))))
        end
    end
    return [minimum(@view angles[n, :]) for n in 1:N]
end

"""
    poles(orientations, ref_axes="xz", hkl=[1,0,0])

Extract 3D vectors of crystallographic directions from orientation matrices.
`orientations` is an N×3×3 array.
"""
function poles(orientations::AbstractArray{Float64,3};
               ref_axes::String="xz", hkl::Vector{Int}=[1,0,0])
    _ref = lowercase(ref_axes)
    up_char = first(setdiff(Set(['x','y','z']), Set(collect(_ref))))
    axes_map = Dict('x' => 1, 'y' => 2, 'z' => 3)

    n = size(orientations, 1)
    # directions = orientationsᵀ * hkl  (for each grain)
    hkl_f = Float64.(hkl)
    directions = zeros(n, 3)
    @inbounds for g in 1:n
        for i in 1:3
            val = 0.0
            for j in 1:3
                val += orientations[g, j, i] * hkl_f[j]
            end
            directions[g, i] = val
        end
    end
    # Normalise
    @inbounds for g in 1:n
        nrm = sqrt(directions[g,1]^2 + directions[g,2]^2 + directions[g,3]^2)
        directions[g,1] /= nrm
        directions[g,2] /= nrm
        directions[g,3] /= nrm
    end

    xvals = directions[:, axes_map[_ref[1]]]
    yvals = directions[:, axes_map[_ref[2]]]
    zvals = directions[:, axes_map[up_char]]
    return xvals, yvals, zvals
end

"""
    lambert_equal_area(xvals, yvals, zvals)

Project axial data from a 3D sphere onto a 2D disk using Lambert equal-area projection.
"""
function lambert_equal_area(xvals, yvals, zvals)
    x = Float64.(xvals)
    y = Float64.(yvals)
    z = abs.(Float64.(zvals))  # Map to upper hemisphere
    n = length(x)
    x_out = similar(x)
    y_out = similar(y)
    @inbounds for i in 1:n
        if abs(x[i]) < 1e-16 && abs(y[i]) < 1e-16
            x_out[i] = 0.0
            y_out[i] = 0.0
        else
            pf = sqrt((1 - z[i]) / (x[i]^2 + y[i]^2))
            x_out[i] = pf * x[i]
            y_out[i] = pf * y[i]
        end
    end
    return x_out, y_out
end

"""
    shirley_concentric_squaredisk(xvals, yvals)

Project points from a square onto a disk (Shirley & Chiu 1997 concentric method).
"""
function shirley_concentric_squaredisk(xvals, yvals)
    x = Float64.(xvals)
    y = Float64.(yvals)
    n = length(x)
    xd = similar(x)
    yd = similar(y)
    @inbounds for i in 1:n
        if abs(x[i]) >= abs(y[i])
            r = x[i]
            θ = (π/4) * (y[i] / (x[i] + 1e-12))
            xd[i] = r * cos(θ)
            yd[i] = r * sin(θ)
        else
            r = y[i]
            θ = (π/4) * (x[i] / (y[i] + 1e-12))
            xd[i] = r * sin(θ)
            yd[i] = r * cos(θ)
        end
    end
    return xd, yd
end

"""
    to_indices2d(horizontal, vertical)

Convert axis labels ("X","Y","Z") to 1-based index pair.
"""
function to_indices2d(horizontal::String, vertical::String)
    axes_map = Dict("X" => 1, "Y" => 2, "Z" => 3)
    h = uppercase(horizontal)
    v = uppercase(vertical)
    haskey(axes_map, h) && haskey(axes_map, v) || throw(ArgumentError("invalid axes"))
    return (axes_map[h], axes_map[v])
end

"""
    symmetry_operations(system::LatticeSystem)

Return symmetry operations (as quaternion 4-vectors or 4×4 reflection matrices).
"""
function symmetry_operations(system::LatticeSystem)
    identity_q = [0.0, 0.0, 0.0, 1.0]  # scalar-last convention
    if system == triclinic
        return [identity_q]
    elseif system == monoclinic || system == orthorhombic
        rotations = []
        for axis in ([0.0,0.0,1.0], [0.0,1.0,0.0], [1.0,0.0,0.0])
            angle = π
            # Quaternion for rotation by angle around axis (scalar-last)
            s = sin(angle/2)
            c = cos(angle/2)
            push!(rotations, [s*axis[1], s*axis[2], s*axis[3], c])
        end
        reflections = [
            diagm([1.0, -1.0, -1.0, 1.0]),
            diagm([1.0, -1.0, 1.0, -1.0]),
            diagm([1.0, 1.0, -1.0, -1.0]),
        ]
        return [[identity_q]; rotations; reflections]
    elseif system == rhombohedral
        rotations = []
        for axis in ([0.0,0.0,1.0], [0.0,1.0,0.0], [1.0,0.0,0.0])
            for n in (1, 2)
                angle = n * π / 3
                s = sin(angle/2); c = cos(angle/2)
                push!(rotations, [s*axis[1], s*axis[2], s*axis[3], c])
            end
        end
        return [[identity_q]; rotations]
    elseif system == tetragonal
        rotations = []
        for axis in ([0.0,0.0,1.0], [0.0,1.0,0.0], [1.0,0.0,0.0])
            for n in (1, 2, 3)
                angle = n * π / 2
                s = sin(angle/2); c = cos(angle/2)
                push!(rotations, [s*axis[1], s*axis[2], s*axis[3], c])
            end
        end
        return [[identity_q]; rotations]
    elseif system == hexagonal
        ops = []
        for axis in ([0.0,0.0,1.0], [0.0,1.0,0.0], [1.0,0.0,0.0])
            for n in (1, 2)
                angle = n * π / 3
                s = sin(angle/2); c = cos(angle/2)
                push!(ops, [s*axis[1], s*axis[2], s*axis[3], c])
            end
        end
        for axis in ([0.0,0.0,1.0], [0.0,1.0,0.0], [1.0,0.0,0.0])
            for n in (1, 3, 5)
                angle = n * π / 6
                s = sin(angle/2); c = cos(angle/2)
                push!(ops, [s*axis[1], s*axis[2], s*axis[3], c])
            end
        end
        return [[identity_q]; ops]
    end
    throw(ArgumentError("unsupported lattice system: $system"))
end
