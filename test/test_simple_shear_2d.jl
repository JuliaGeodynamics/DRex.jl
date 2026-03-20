using Test
using LinearAlgebra
using Statistics
using DRex
using DRex: simple_shear_2d, strain_increment, finite_strain, smallest_angle

@testset "Simple Shear 2D Preliminaries" begin
    @testset "strain_increment" begin
        # Test strain accumulation with constant velocity gradient
        _, get_L = simple_shear_2d("X", "Z", 1.0)
        timestamps = range(0, 1, length=10)
        strains_inc = zeros(length(timestamps))
        L = get_L(NaN, [0.0, 0.0, 0.0])
        dt = timestamps[2] - timestamps[1]
        for i in 2:length(timestamps)
            strains_inc[i] = strains_inc[i-1] + strain_increment(dt, L)
        end
        @test isapprox(strains_inc, collect(timestamps), atol=6e-16)

        # Same for experimental-like strain rate
        _, get_L2 = simple_shear_2d("Y", "X", 1e-5)
        timestamps2 = range(0, 1e6, length=10)
        strains_inc2 = zeros(length(timestamps2))
        L2 = get_L2(NaN, [0.0, 0.0, 0.0])
        dt2 = timestamps2[2] - timestamps2[1]
        for i in 2:length(timestamps2)
            strains_inc2[i] = strains_inc2[i-1] + strain_increment(dt2, L2)
        end
        @test isapprox(strains_inc2, collect(timestamps2) .* 1e-5, atol=5e-15)
    end
end

@testset "Simple Shear 2D OlivineA" begin
    for T in (Float64, Float32)
        atol_frac = T == Float32 ? 1e-5 : 1e-15

        @testset "zero_recrystallisation T=$T" begin
            seed = 8816
            params = DRex.default_params()
            params[:gbm_mobility] = 0.0
            strain_rate = 1.0
            timestamps = range(0, 1, length=25)
            _, get_L = simple_shear_2d("Y", "X", strain_rate)

            mineral = Mineral(
                float_type=T,
                phase=olivine,
                fabric=olivine_A,
                regime=matrix_dislocation,
                n_grains=params[:number_of_grains],
                seed=seed,
            )
            deformation_gradient = Matrix{T}(I, 3, 3)
            for t in 2:length(timestamps)
                deformation_gradient = update_orientations!(
                    mineral,
                    params,
                    deformation_gradient,
                    get_L,
                    (timestamps[t-1], timestamps[t], t -> zeros(3)),
                )
            end
            # With M*=0, fractions should not change
            initial_fractions = mineral.fractions[1]
            for frac in mineral.fractions[2:end]
                @test isapprox(frac, initial_fractions, atol=atol_frac)
            end
        end

        @testset "grainsize_median T=$T" begin
            for gbm_mobility in [50, 100, 150]
                seed = 8816
                params = DRex.default_params()
                params[:gbm_mobility] = Float64(gbm_mobility)
                params[:nucleation_efficiency] = 5.0
                strain_rate = 1.0
                timestamps = range(0, 1, length=25)
                _, get_L = simple_shear_2d("Y", "X", strain_rate)

                mineral = Mineral(
                    float_type=T,
                    phase=olivine,
                    fabric=olivine_A,
                    regime=matrix_dislocation,
                    n_grains=params[:number_of_grains],
                    seed=seed,
                )
                deformation_gradient = Matrix{T}(I, 3, 3)
                for t in 2:length(timestamps)
                    deformation_gradient = update_orientations!(
                        mineral,
                        params,
                        deformation_gradient,
                        get_L,
                        (timestamps[t-1], timestamps[t], t -> zeros(3)),
                    )
                end
                n_steps = length(timestamps)
                medians = [median(mineral.fractions[i]) for i in 1:n_steps]
                # After first step (nucleation sets in), medians should decrease
                diffs = diff(medians)[2:end]
                @test all(d -> d < 0, diffs)
            end
        end
    end
end

# Helper: median that doesn't require Statistics.jl
function median_val(v::AbstractVector)
    s = sort(v)
    n = length(s)
    if isodd(n)
        return s[div(n+1, 2)]
    else
        return (s[div(n, 2)] + s[div(n, 2)+1]) / 2
    end
end
