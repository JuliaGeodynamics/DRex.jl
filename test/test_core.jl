using Test
using LinearAlgebra
using DRex
using DRex: _get_slip_invariants, _get_deformation_rate, _get_slip_rates_olivine,
              _argsort4, _get_rotation_and_strain
using StaticArrays

"""Rotation matrix from axis-angle (Rodrigues formula)."""
function rotation_from_rotvec(rotvec::Vector{Float64})
    θ = norm(rotvec)
    if θ < 1e-30
        return Matrix{Float64}(I, 3, 3)
    end
    k = rotvec ./ θ
    K = [0 -k[3] k[2]; k[3] 0 -k[1]; -k[2] k[1] 0]
    return I + sin(θ) * K + (1 - cos(θ)) * K^2
end

@testset "Dislocation Creep OPX" begin
    @testset "shear_dudz" begin
        for θ in range(0, 2π, length=360)
            R = rotation_from_rotvec([0.0, θ, 0.0])
            orientations = reshape(R, 1, 3, 3)
            orientations_diff, fractions_diff = DRex.derivatives(
                matrix_dislocation,
                enstatite,
                enstatite_AB,
                1,
                orientations,
                [1.0],
                Float64[0 0 1; 0 0 0; 1 0 0],
                Float64[0 0 2; 0 0 0; 0 0 0],
                fill(NaN, 3, 3),
                1.5,
                3.5,
                5.0,
                0.0,
                1.0,
            )
            cosθ = cos(θ); cos2θ = cos(2θ); sinθ = sin(θ)
            target = (1 + cos2θ) * [sinθ 0 -cosθ; 0 0 0; cosθ 0 sinθ]
            @test isapprox(orientations_diff[1,:,:], target, atol=1e-12)
            @test isapprox(sum(fractions_diff), 0.0, atol=1e-15)
        end
    end

    @testset "shear_dvdx" begin
        for θ in range(0, 2π, length=361)
            R = rotation_from_rotvec([0.0, 0.0, θ])
            orientations = reshape(R, 1, 3, 3)

            # Check that deformation rate is zero for enstatite with this shear
            ori_s = SMatrix{3,3,Float64,9}(R')  # transpose as orientation
            slip_rates = SVector{4,Float64}(0.0, 0.0, 0.0, 0.0)
            deformation_rate = _get_deformation_rate(enstatite, ori_s, slip_rates)
            @test isapprox(Array(deformation_rate), zeros(3,3), atol=1e-15)

            orientations_diff, fractions_diff = DRex.derivatives(
                matrix_dislocation,
                enstatite,
                enstatite_AB,
                1,
                orientations,
                [1.0],
                Float64[0 1 0; 1 0 0; 0 0 0],
                Float64[0 0 0; 2 0 0; 0 0 0],
                fill(NaN, 3, 3),
                1.5,
                3.5,
                5.0,
                0.0,
                1.0,
            )
            sinθ = sin(θ); cosθ = cos(θ)
            target = [sinθ cosθ 0; -cosθ sinθ 0; 0 0 0]
            @test isapprox(orientations_diff[1,:,:], target, atol=1e-15)
            @test isapprox(sum(fractions_diff), 0.0, atol=1e-15)
        end
    end
end

@testset "Dislocation Creep Olivine A" begin
    @testset "shear_dvdx_slip_010_100" begin
        L = Float64[0 0 0; 2 0 0; 0 0 0]
        sr = Float64[0 1 0; 1 0 0; 0 0 0]

        for θ in range(0, 2π, length=3600)
            R = rotation_from_rotvec([0.0, 0.0, θ])
            orientations = reshape(R, 1, 3, 3)

            # Check slip system is (010)[100]
            crss = get_crss(olivine, olivine_A)
            sr_s = SMatrix{3,3,Float64,9}(sr)
            ori_s = SMatrix{3,3,Float64,9}(R)
            slip_invariants = _get_slip_invariants(sr_s, ori_s)
            ratios = SVector{4,Float64}(abs(slip_invariants[1]/crss[1]),
                                        abs(slip_invariants[2]/crss[2]),
                                        abs(slip_invariants[3]/crss[3]),
                                        abs(slip_invariants[4]/crss[4]))
            slip_indices = _argsort4(ratios)
            slip_system = DRex.OLIVINE_SLIP_SYSTEMS[slip_indices[4]]
            @test slip_system == ([0,1,0], [1,0,0])

            orientations_diff, fractions_diff = DRex.derivatives(
                matrix_dislocation,
                olivine,
                olivine_A,
                1,
                orientations,
                [1.0],
                sr,
                L,
                fill(NaN, 3, 3),
                1.5,
                3.5,
                5.0,
                0.0,
                1.0,
            )
            cosθ = cos(θ); cos2θ = cos(2θ); sinθ = sin(θ)
            target = [
                sinθ*(1+cos2θ)   cosθ*(1+cos2θ)  0;
                -cosθ*(1+cos2θ)  sinθ*(1+cos2θ)  0;
                0                0               0
            ]
            @test isapprox(orientations_diff[1,:,:], target, atol=1e-12)
            @test isapprox(sum(fractions_diff), 0.0, atol=1e-15)
        end
    end

    @testset "shear_dudz_slip_001_100" begin
        L = Float64[0 0 2; 0 0 0; 0 0 0]
        sr = Float64[0 0 1; 0 0 0; 1 0 0]

        for θ in range(0, 2π, length=360)
            R = rotation_from_rotvec([0.0, θ, 0.0])
            orientations = reshape(R, 1, 3, 3)

            # Check slip system is (001)[100]
            crss = get_crss(olivine, olivine_A)
            sr_s = SMatrix{3,3,Float64,9}(sr)
            ori_s = SMatrix{3,3,Float64,9}(R)
            slip_invariants = _get_slip_invariants(sr_s, ori_s)
            ratios = SVector{4,Float64}(abs(slip_invariants[1]/crss[1]),
                                        abs(slip_invariants[2]/crss[2]),
                                        abs(slip_invariants[3]/crss[3]),
                                        abs(slip_invariants[4]/crss[4]))
            slip_indices = _argsort4(ratios)
            slip_system = DRex.OLIVINE_SLIP_SYSTEMS[slip_indices[4]]
            @test slip_system == ([0,0,1], [1,0,0])

            orientations_diff, fractions_diff = DRex.derivatives(
                matrix_dislocation,
                olivine,
                olivine_A,
                1,
                orientations,
                [1.0],
                sr,
                L,
                fill(NaN, 3, 3),
                1.5,
                3.5,
                5.0,
                0.0,
                1.0,
            )
            cosθ = cos(θ); cos2θ = cos(2θ); sinθ = sin(θ)
            target = [
                -sinθ*(cos2θ-1)  0  cosθ*(cos2θ-1);
                 0               0  0;
                 cosθ*(1-cos2θ)  0  sinθ*(1-cos2θ)
            ]
            @test isapprox(orientations_diff[1,:,:], target, atol=1e-12)
            @test isapprox(sum(fractions_diff), 0.0, atol=1e-15)
        end
    end
end
