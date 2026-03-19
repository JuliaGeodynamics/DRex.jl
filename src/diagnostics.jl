# Texture and strain diagnostics.

"""
    elasticity_components(voigt_matrices)

Calculate elasticity decompositions for N×6×6 array of Voigt matrices.
Returns a Dict with keys: bulk_modulus, shear_modulus, percent_anisotropy,
percent_hexagonal, percent_tetragonal, percent_orthorhombic, percent_monoclinic,
percent_triclinic, hexagonal_axis.
"""
function elasticity_components(voigt_matrices::Vector{Matrix{Float64}})
    n = length(voigt_matrices)
    out = Dict{String,Any}(
        "bulk_modulus"         => zeros(n),
        "shear_modulus"        => zeros(n),
        "percent_anisotropy"   => zeros(n),
        "percent_hexagonal"    => zeros(n),
        "percent_tetragonal"   => zeros(n),
        "percent_orthorhombic" => zeros(n),
        "percent_monoclinic"   => zeros(n),
        "percent_triclinic"    => zeros(n),
        "hexagonal_axis"       => zeros(n, 3),
    )

    for (m, mat) in enumerate(voigt_matrices)
        voigt_matrix = upper_tri_to_symmetric(mat)
        sd, sv = voigt_decompose(voigt_matrix)
        K = tr(sd) / 9
        G = (tr(sv) - 3K) / 10
        out["bulk_modulus"][m] = K
        out["shear_modulus"][m] = G

        # Isotropic vector
        iso = vcat(
            fill(K + 4G/3, 3),
            fill(sqrt(2) * (K - 2G/3), 3),
            fill(2G, 3),
            zeros(12),
        )
        vv = voigt_matrix_to_vector(voigt_matrix)
        out["percent_anisotropy"][m] = norm(vv - iso) / norm(vv) * 100

        # SCCS search
        unperm_SCCS = zeros(3, 3)
        eigv_d = eigvecs(Symmetric(sd))
        eigv_v = eigvecs(Symmetric(sv))

        for i in 1:3
            index_vij = 0
            angle = 10.0
            for j in 1:3
                dot_val = dot(eigv_d[:, i], eigv_v[:, j])
                angle_val = smallest_angle(eigv_d[:, i], eigv_v[:, j])
                if angle_val < angle
                    angle = angle_val
                    index_vij = dot_val != 0 ? sign(dot_val) * j : j
                end
            end
            if index_vij == 0
                vec_SCCS = eigv_d[:, i]
            else
                vec_SCCS = (eigv_d[:, i] .+ index_vij .* eigv_v[:, Int(abs(index_vij))]) ./ 2
            end
            vec_SCCS ./= norm(vec_SCCS)
            unperm_SCCS[:, i] = vec_SCCS
        end

        # Best hexagonal approximation
        elastic_tensor = voigt_to_elastic_tensor(voigt_matrix)
        distance = norm(vv)
        for i in 1:3
            perm = [mod1(i + j - 1, 3) for j in 1:3]
            perm_SCCS = unperm_SCCS[:, perm]
            rot_voigt = elastic_tensor_to_voigt(rotate_tensor(elastic_tensor, perm_SCCS'))
            rot_vv = voigt_matrix_to_vector(rot_voigt)
            mono_h = mono_project(rot_vv)
            tric_v = rot_vv .- mono_h
            ortho_h = ortho_project(mono_h)
            tetr_h = tetr_project(ortho_h)
            hex_h = hex_project(tetr_h)
            mono_v = mono_h .- ortho_h
            ortho_v = ortho_h .- tetr_h
            tetr_v = tetr_h .- hex_h
            hex_v = hex_h .- iso

            δ = norm(rot_vv .- hex_h)
            if δ < distance
                distance = δ
                pct = 100 / norm(vv)
                out["percent_triclinic"][m]    = norm(tric_v) * pct
                out["percent_monoclinic"][m]   = norm(mono_v) * pct
                out["percent_orthorhombic"][m] = norm(ortho_v) * pct
                out["percent_tetragonal"][m]   = norm(tetr_v) * pct
                out["percent_hexagonal"][m]    = norm(hex_v) * pct
                out["hexagonal_axis"][m, :]    = perm_SCCS[:, 3]
            end
        end
    end
    return out
end

"""
    bingham_average(orientations; axis="a")

Compute Bingham average (antipodally symmetric mean) of orientation matrices.
`orientations` is N×3×3. Returns the mean direction as a 3-vector.
"""
function bingham_average(orientations::AbstractArray{Float64,3}; axis::String="a")
    row = axis == "a" ? 1 : axis == "b" ? 2 : axis == "c" ? 3 :
        throw(ArgumentError("axis must be 'a', 'b', or 'c'"))
    scatter = _scatter_matrix(orientations, row)
    vals, vecs = eigen(Symmetric(scatter, :L))
    # Largest eigenvalue is last (ascending order)
    mean_vec = vecs[:, end]
    return mean_vec ./ norm(mean_vec)
end

"""
    finite_strain(deformation_gradient)

Extract largest principal strain value and maximum extension axis from the 3×3
deformation gradient. Returns (strain, axis).
"""
function finite_strain(deformation_gradient::AbstractMatrix{Float64})
    B = deformation_gradient * deformation_gradient'
    vals, vecs = eigen(Symmetric(B))
    return sqrt(vals[end]) - 1, vecs[:, end]
end

"""
    symmetry_pgr(orientations; axis="a")

Compute Point, Girdle, Random symmetry diagnostics. Returns (P, G, R).
"""
function symmetry_pgr(orientations::AbstractArray{Float64,3}; axis::String="a")
    row = axis == "a" ? 1 : axis == "b" ? 2 : axis == "c" ? 3 :
        throw(ArgumentError("axis must be 'a', 'b', or 'c'"))
    scatter = _scatter_matrix(orientations, row)
    eigvals_desc = reverse(eigvals(Symmetric(scatter, :L)))
    s = sum(eigvals_desc)
    P = (eigvals_desc[1] - eigvals_desc[2]) / s
    G = 2 * (eigvals_desc[2] - eigvals_desc[3]) / s
    R = 3 * eigvals_desc[3] / s
    return (P, G, R)
end

"""
    misorientation_index(orientations, system; bins=nothing)

Calculate M-index for polycrystal orientations.
"""
function misorientation_index(orientations::AbstractArray{Float64,3},
                              system::LatticeSystem; bins::Union{Int,Nothing}=nothing)
    θmax = _max_misorientation(system)
    mis_count, bin_edges = misorientation_hist(orientations, system; bins=bins)
    nbins = length(mis_count)
    mis_theory = [misorientations_random(bin_edges[i], bin_edges[i+1], system) for i in 1:nbins]
    return (θmax / (2 * nbins)) * sum(abs.(mis_theory .- mis_count))
end

"""
    coaxial_index(orientations; axis1="b", axis2="a")

Calculate coaxial "BA" index. Returns value in [0, 1].
"""
function coaxial_index(orientations::AbstractArray{Float64,3};
                       axis1::String="b", axis2::String="a")
    P1, G1, _ = symmetry_pgr(orientations; axis=axis1)
    P2, G2, _ = symmetry_pgr(orientations; axis=axis2)
    return 0.5 * (2 - P1 / (G1 + P1) - G2 / (G2 + P2))
end

"""
    smallest_angle(vector, axis; plane=nothing)

Get smallest angle (degrees) between a unit vector and a bidirectional axis.
"""
function smallest_angle(vector::AbstractVector{Float64}, axis::AbstractVector{Float64};
                        plane::Union{AbstractVector{Float64},Nothing}=nothing)
    v = plane !== nothing ? vector .- plane .* dot(vector, plane) : vector
    cosval = clamp(dot(v, axis) / (norm(v) * norm(axis)), -1.0, 1.0)
    angle = rad2deg(acos(cosval))
    return angle > 90 ? 180 - angle : angle
end
