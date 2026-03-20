using Test
using LinearAlgebra
using Random
using DRex
using DRex: elasticity_components, symmetry_pgr, upper_tri_to_symmetric
using DRex: resample_orientations

"""Rotation matrix from axis-angle (Rodrigues formula)."""
function _rotation_from_rotvec(rotvec::Vector{Float64})
    θ = norm(rotvec)
    if θ < 1e-30
        return Matrix{Float64}(I, 3, 3)
    end
    k = rotvec ./ θ
    K = [0 -k[3] k[2]; k[3] 0 -k[1]; -k[2] k[1] 0]
    return I + sin(θ) * K + (1 - cos(θ)) * K^2
end

"""Generate random rotation matrices (Haar-distributed)."""
function _random_rotations(n::Int; rng=Random.default_rng(), float_type::Type{T}=Float64) where T
    orientations = Array{T,3}(undef, n, 3, 3)
    for g in 1:n
        A = randn(rng, 3, 3)
        Q, R = qr(A)
        Qm = Matrix(Q) * Diagonal(sign.(diag(R)))
        if det(Qm) < 0
            Qm[:, 1] .*= -1
        end
        for i in 1:3, j in 1:3
            orientations[g, i, j] = T(Qm[i, j])
        end
    end
    return orientations
end

# Elasticity tests use Float64 only (elasticity_components requires Matrix{Float64}).
@testset "Elasticity Components" begin
    @testset "olivine_Browaeys2004" begin
        C = Float64[
            192  66  60  0  0  0;
             66 160  56  0  0  0;
             60  56 272  0  0  0;
              0   0   0 60  0  0;
              0   0   0  0 62  0;
              0   0   0  0  0 49
        ]
        out = elasticity_components([C])
        @test isapprox(out["bulk_modulus"][1], 109.8, atol=0.1)
        @test isapprox(out["shear_modulus"][1], 63.7, atol=0.1)
        @test isapprox(out["percent_anisotropy"][1], 20.7, atol=0.1)
        @test isapprox(out["percent_monoclinic"][1], 0.0, atol=0.1)
        @test isapprox(out["percent_triclinic"][1], 0.0, atol=0.1)
        @test isapprox(out["hexagonal_axis"][1, :], [0, 0, 1], atol=1e-10)
    end

    @testset "enstatite_Browaeys2004" begin
        C = Float64[
            225  54  72  0  0  0;
             54 214  53  0  0  0;
             72  53 178  0  0  0;
              0   0   0 78  0  0;
              0   0   0  0 82  0;
              0   0   0  0  0 76
        ]
        out = elasticity_components([C])
        @test isapprox(out["bulk_modulus"][1], 108.3, atol=0.1)
        @test isapprox(out["shear_modulus"][1], 76.4, atol=0.1)
        @test isapprox(out["percent_anisotropy"][1], 9.2, atol=0.1)
        @test isapprox(out["percent_monoclinic"][1], 0.0, atol=0.1)
        @test isapprox(out["percent_triclinic"][1], 0.0, atol=0.1)
        @test isapprox(out["hexagonal_axis"][1, :], [0, 0, 1], atol=1e-10)
    end
end

@testset "Symmetry PGR" begin
    for T in (Float64, Float32)
        @testset "pointX T=$T" begin
            rng = MersenneTwister(42)
            n = 100
            orientations = Array{T,3}(undef, n, 3, 3)
            for i in 1:n
                x = rand(rng)
                rv = [0.0, x * π/18 - π/36, x * π/18 - π/36]
                R = _rotation_from_rotvec(rv)
                Rinv = inv(R)
                for ii in 1:3, jj in 1:3
                    orientations[i, ii, jj] = T(Rinv[ii, jj])
                end
            end
            P, G, R = symmetry_pgr(orientations; axis="a")
            @test isapprox(P, 1.0, atol=0.05)
            @test isapprox(G, 0.0, atol=0.05)
            @test isapprox(R, 0.0, atol=0.05)
        end

        @testset "random T=$T" begin
            orientations = _random_rotations(1000; rng=MersenneTwister(123), float_type=T)
            P, G, R = symmetry_pgr(orientations; axis="a")
            @test isapprox(P, 0.0, atol=0.15)
            @test isapprox(G, 0.0, atol=0.15)
            @test isapprox(R, 1.0, atol=0.15)
        end

        @testset "girdle T=$T" begin
            rng = MersenneTwister(99)
            n = 1000
            quats = zeros(n, 4)
            quats[:, 3] = randn(rng, n)
            quats[:, 4] = randn(rng, n)
            for i in 1:n
                q = quats[i, :]
                q ./= norm(q)
                quats[i, :] = q
            end
            orientations = Array{T,3}(undef, n, 3, 3)
            for i in 1:n
                x, y, z, w = quats[i, :]
                R = [
                    1-2*(y^2+z^2)  2*(x*y-z*w)    2*(x*z+y*w);
                    2*(x*y+z*w)    1-2*(x^2+z^2)  2*(y*z-x*w);
                    2*(x*z-y*w)    2*(y*z+x*w)    1-2*(x^2+y^2)
                ]
                for ii in 1:3, jj in 1:3
                    orientations[i, ii, jj] = T(R[ii, jj])
                end
            end
            P, G, R = symmetry_pgr(orientations; axis="a")
            @test isapprox(P, 0.0, atol=0.1)
            @test isapprox(G, 1.0, atol=0.1)
            @test isapprox(R, 0.0, atol=0.1)
        end
    end
end

@testset "Volume Weighting" begin
    for T in (Float64, Float32)
        @testset "output_shape T=$T" begin
            o1 = _random_rotations(1000; rng=MersenneTwister(1), float_type=T)
            o2 = _random_rotations(1000; rng=MersenneTwister(2), float_type=T)
            f1 = fill(T(1)/T(1000), 1000)
            f2 = fill(T(1)/T(1000), 1000)
            new_o, new_f = resample_orientations([o1, o2], [f1, f2])
            @test size(new_o) == (2,)
            @test size(new_o[1]) == (1000, 3, 3)
            @test length(new_f[1]) == 1000

            new_o2, new_f2 = resample_orientations([o1, o2], [f1, f2]; n_samples=500)
            @test size(new_o2[1]) == (500, 3, 3)
            @test length(new_f2[1]) == 500
        end
    end
end
