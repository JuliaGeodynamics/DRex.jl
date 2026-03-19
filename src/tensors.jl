# Tensor operations and Voigt notation helpers.
#
# For Voigt notation, the symmetric 6×6 matrix representation is used.
# The vectorial notation uses 21 components (independent symmetric 6×6 entries).
# All inner routines use StaticArrays where possible for allocation-free operation.

"""
    polar_decompose(M; left=true)

Compute polar decomposition. If `left=true`, returns (R, V) with M = V*R.
If `left=false`, returns (R, U) with M = R*U.
"""
function polar_decompose(M::AbstractMatrix{Float64}; left::Bool=true)
    F = svd(M)
    R = F.U * F.Vt
    if left
        V = F.U * Diagonal(F.S) * F.U'
        return R, V
    else
        U = F.Vt' * Diagonal(F.S) * F.Vt
        return R, U
    end
end

"""
    invariants_second_order(tensor)

Calculate invariants of a second-order tensor. Returns (I₁, I₂, I₃).
"""
function invariants_second_order(tensor::AbstractMatrix{Float64})
    I1 = tr(tensor)
    I2 = (tensor[1,1]*tensor[2,2] + tensor[2,2]*tensor[3,3] + tensor[3,3]*tensor[1,1]
         - tensor[1,2]*tensor[2,1] - tensor[2,3]*tensor[3,2] - tensor[3,1]*tensor[1,3])
    I3 = det(tensor)
    return (I1, I2, I3)
end

"""
    upper_tri_to_symmetric(A)

Create symmetric matrix using the upper triangle of the input matrix.
"""
function upper_tri_to_symmetric(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    S = similar(A)
    @inbounds for j in 1:n, i in 1:n
        if i <= j
            S[i,j] = A[i,j]
        else
            S[i,j] = A[j,i]
        end
    end
    return S
end

"""
    voigt_decompose(matrix)

Decompose elastic tensor (6×6 Voigt matrix) into dilatational and deviatoric
stiffness tensors. Returns (d_ij, v_ij) where d_ij = C_ijkk and v_ij = C_ikjk.
"""
function voigt_decompose(matrix::AbstractMatrix{Float64})
    tensor = voigt_to_elastic_tensor(matrix)
    # d_ij = C_ijkk (contract last two indices)
    sd = zeros(3, 3)
    @inbounds for i in 1:3, j in 1:3
        for k in 1:3
            sd[i,j] += tensor[i,j,k,k]
        end
    end
    # v_ij = C_ikjk (contract 2nd and 4th indices)
    sv = zeros(3, 3)
    @inbounds for i in 1:3, j in 1:3
        for k in 1:3
            sv[i,j] += tensor[i,k,j,k]
        end
    end
    return sd, sv
end

"""
    voigt_to_elastic_tensor(matrix)

Create 4th-order elastic tensor from a 6×6 Voigt matrix.
"""
function voigt_to_elastic_tensor(matrix::AbstractMatrix{Float64})
    tensor = Array{Float64,4}(undef, 3, 3, 3, 3)
    @inbounds for p in 1:3, q in 1:3
        δpq = (p == q) ? 1 : 0
        i = (p) * δpq + (1 - δpq) * (9 - p - q)   # 1-based Voigt index
        for r in 1:3, s in 1:3
            δrs = (r == s) ? 1 : 0
            j = (r) * δrs + (1 - δrs) * (9 - r - s)
            tensor[p,q,r,s] = matrix[i,j]
        end
    end
    return tensor
end

"""
    elastic_tensor_to_voigt(tensor)

Create a 6×6 Voigt matrix from a 4th-order elastic tensor.
"""
function elastic_tensor_to_voigt(tensor::AbstractArray{Float64,4})
    matrix = zeros(6, 6)
    counts = zeros(6, 6)
    @inbounds for p in 1:3, q in 1:3
        δpq = (p == q) ? 1 : 0
        i = (p) * δpq + (1 - δpq) * (9 - p - q)
        for r in 1:3, s in 1:3
            δrs = (r == s) ? 1 : 0
            j = (r) * δrs + (1 - δrs) * (9 - r - s)
            matrix[i,j] += tensor[p,q,r,s]
            counts[i,j] += 1.0
        end
    end
    matrix ./= counts
    return (matrix .+ matrix') ./ 2
end

"""
    voigt_matrix_to_vector(matrix)

Create the 21-component Voigt vector from the 6×6 Voigt matrix.
"""
function voigt_matrix_to_vector(matrix::AbstractMatrix{Float64})
    v = zeros(21)
    for i in 1:3
        # Julia uses 1-based indexing; the cyclic modular arithmetic is adjusted
        i1 = mod1(i + 1, 3)
        i2 = mod1(i + 2, 3)
        v[i]      = matrix[i, i]
        v[i + 3]  = sqrt(2) * matrix[i1, i2]
        v[i + 6]  = 2 * matrix[i + 3, i + 3]
        v[i + 9]  = 2 * matrix[i, i + 3]
        v[i + 12] = 2 * matrix[i2, i + 3]
        v[i + 15] = 2 * matrix[i1, i + 3]
        v[i + 18] = 2 * sqrt(2) * matrix[i1 + 3, i2 + 3]
    end
    return v
end

"""
    voigt_vector_to_matrix(vector)

Create the 6×6 Voigt matrix from the 21-component Voigt vector.
"""
function voigt_vector_to_matrix(vector::AbstractVector{Float64})
    m = zeros(6, 6)
    for i in 1:3
        m[i, i] = vector[i]
        m[i + 3, i + 3] = 0.5 * vector[i + 6]
    end

    m[2, 3] = 1 / sqrt(2) * vector[4]
    m[1, 3] = 1 / sqrt(2) * vector[5]
    m[1, 2] = 1 / sqrt(2) * vector[6]

    m[1, 4] = 0.5 * vector[10]
    m[2, 5] = 0.5 * vector[11]
    m[3, 6] = 0.5 * vector[12]
    m[3, 4] = 0.5 * vector[13]

    m[1, 5] = 0.5 * vector[14]
    m[2, 6] = 0.5 * vector[15]
    m[2, 4] = 0.5 * vector[16]

    m[3, 5] = 0.5 * vector[17]
    m[1, 6] = 0.5 * vector[18]
    m[5, 6] = 0.5 / sqrt(2) * vector[19]
    m[4, 6] = 0.5 / sqrt(2) * vector[20]
    m[4, 5] = 0.5 / sqrt(2) * vector[21]
    return upper_tri_to_symmetric(m)
end

"""
    rotate_tensor(tensor, rotation)

Rotate a 4th-order tensor using a 3×3 rotation matrix.
"""
function rotate_tensor(tensor::AbstractArray{Float64,4}, rotation::AbstractMatrix{Float64})
    rotated = zeros(3, 3, 3, 3)
    @inbounds for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        val = 0.0
        for a in 1:3, b in 1:3, c in 1:3, d in 1:3
            val += rotation[i,a] * rotation[j,b] *
                   rotation[k,c] * rotation[l,d] * tensor[a,b,c,d]
        end
        rotated[i,j,k,l] = val
    end
    return rotated
end

# ──────────────────────────────────────────────────────────────────────────────
# Symmetry projections onto Voigt vector subspaces
# ──────────────────────────────────────────────────────────────────────────────

"""Project 21-component Voigt vector onto monoclinic symmetry subspace (13 components)."""
function mono_project(v::AbstractVector{Float64})
    out = copy(v)
    for i in (10, 11, 13, 14, 16, 17, 19, 20)
        out[i] = 0.0
    end
    return out
end

"""Project 21-component Voigt vector onto orthorhombic symmetry subspace (9 components)."""
function ortho_project(v::AbstractVector{Float64})
    out = copy(v)
    out[10:21] .= 0.0
    return out
end

"""Project 21-component Voigt vector onto tetragonal symmetry subspace (6 components)."""
function tetr_project(v::AbstractVector{Float64})
    out = ortho_project(v)
    for (i, j) in ((1, 2), (4, 5), (7, 8))
        avg = 0.5 * (v[i] + v[j])
        out[i] = avg
        out[j] = avg
    end
    return out
end

"""Project 21-component Voigt vector onto hexagonal symmetry subspace (5 components)."""
function hex_project(v::AbstractVector{Float64})
    out = zeros(21)
    out[1] = out[2] = 3/8 * (v[1] + v[2]) + v[6]/(4*sqrt(2)) + v[9]/4
    out[3] = v[3]
    out[4] = out[5] = (v[4] + v[5]) / 2
    out[6] = (v[1] + v[2]) / (4*sqrt(2)) + 3/4 * v[6] - v[9]/(2*sqrt(2))
    out[7] = out[8] = (v[7] + v[8]) / 2
    out[9] = (v[1] + v[2]) / 4 - v[6]/(2*sqrt(2)) + v[9]/2
    return out
end
