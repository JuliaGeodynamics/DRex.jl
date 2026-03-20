using Test
using LinearAlgebra
using Statistics
using Random
using DRex
using DRex: cell_2d, get_pathline, strain_increment, to_indices2d, add_dim,
              bingham_average, smallest_angle, symmetry_pgr,
              resample_orientations, update_orientations!

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run a 2D Stokes cell simulation for a single mineral phase.
# Ported from Python run_singlephase (without visualisation).
# ─────────────────────────────────────────────────────────────────────────────
function run_singlephase(
    params, seed;
    horizontal="X", vertical="Z",
    velocity_edge=6.342e-10,
    edge_length=2e5,
    max_strain=nothing,
    phase=olivine, fabric=olivine_A,
    regime=matrix_dislocation,
    orientations_init=nothing,
    assert_each=nothing,
    float_type::Type{T}=Float64,
) where T<:AbstractFloat
    get_velocity, get_velocity_gradient = cell_2d(
        horizontal, vertical, velocity_edge; edge_length=edge_length,
    )
    if max_strain === nothing
        max_strain = Int(ceil(velocity_edge * (edge_length / 2)^2))
    end

    mineral = Mineral(
        float_type=T,
        phase=phase, fabric=fabric, regime=regime,
        n_grains=params[:number_of_grains], seed=seed,
        orientations_init=orientations_init === nothing ? nothing : T.(orientations_init),
    )

    sz = edge_length / 2
    h, v = to_indices2d(horizontal, vertical)
    dummy_dim = setdiff(1:3, [h, v])[1]

    final_loc = add_dim([0.5, -0.75], dummy_dim) .* sz
    min_c = add_dim([-sz, -sz], dummy_dim)
    max_c = add_dim([sz, sz], dummy_dim)

    timestamps, get_position = get_pathline(
        final_loc, get_velocity, get_velocity_gradient,
        min_c, max_c;
        max_strain=Float64(max_strain),
        regular_steps=max_strain * 10,
    )
    positions = [get_position(t) for t in timestamps]
    velocity_gradients = [get_velocity_gradient(NaN, x) for x in positions]

    strains = zeros(length(timestamps))
    deformation_gradient = Matrix{T}(I, 3, 3)

    for t in 2:length(timestamps)
        strains[t] = strains[t-1] + strain_increment(
            timestamps[t] - timestamps[t-1], velocity_gradients[t],
        )
        deformation_gradient = update_orientations!(
            mineral, params, deformation_gradient, get_velocity_gradient,
            (timestamps[t-1], timestamps[t], get_position),
        )
        if assert_each !== nothing
            assert_each(mineral, deformation_gradient)
        end
    end
    return mineral, strains
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run TestCellOlivineA.run equivalent
# ─────────────────────────────────────────────────────────────────────────────
function run_cell_olivine_a(
    params, final_location, get_velocity, get_velocity_gradient,
    min_coords, max_coords, max_strain; seed=nothing,
    float_type::Type{T}=Float64,
) where T<:AbstractFloat
    mineral = Mineral(
        float_type=T,
        phase=olivine, fabric=olivine_A, regime=matrix_dislocation,
        n_grains=params[:number_of_grains], seed=seed,
    )
    deformation_gradient = Matrix{T}(I, 3, 3)

    timestamps, get_position = get_pathline(
        final_location, get_velocity, get_velocity_gradient,
        min_coords, max_coords;
        max_strain=Float64(max_strain),
        regular_steps=Int(max_strain * 10),
    )
    positions = [get_position(t) for t in timestamps]
    velocity_gradients = [get_velocity_gradient(NaN, x) for x in positions]

    strains = zeros(length(timestamps))
    for t in 2:length(timestamps)
        strains[t] = strains[t-1] + strain_increment(
            timestamps[t] - timestamps[t-1], velocity_gradients[t],
        )
        deformation_gradient = update_orientations!(
            mineral, params, deformation_gradient, get_velocity_gradient,
            (timestamps[t-1], timestamps[t], get_position),
        )
    end
    return timestamps, positions, strains, mineral, deformation_gradient
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Y-axis rotation matrix
# ─────────────────────────────────────────────────────────────────────────────
function _rotation_y(θ)
    c, s = cos(θ), sin(θ)
    return [c 0 s; 0 1 0; -s 0 c]
end

function _orientations_girdle_y(n_grains, rng; float_type::Type{T}=Float64) where T
    orientations = Array{T,3}(undef, n_grains, 3, 3)
    for g in 1:n_grains
        R = _rotation_y(rand(rng) * 2π)
        for i in 1:3, j in 1:3
            orientations[g, i, j] = T(R[i, j])
        end
    end
    return orientations
end

function _orientations_clustered_y(n_grains, rng; float_type::Type{T}=Float64) where T
    orientations = Array{T,3}(undef, n_grains, 3, 3)
    for g in 1:n_grains
        R = _rotation_y(rand(rng) * π / 8)
        for i in 1:3, j in 1:3
            orientations[g, i, j] = T(R[i, j])
        end
    end
    return orientations
end

# ═════════════════════════════════════════════════════════════════════════════
# TestCellOlivineA — A-type olivine polycrystals in a 2D Stokes cell.
# Non-dimensional cell: edge_length=2.0 (default), velocity_edge=1.
# ═════════════════════════════════════════════════════════════════════════════
@testset "Vortex 2D — CellOlivineA" begin
    seed = 8816

    # Reduced values for fast testing:
    n_grains_list = [100, 500]

    for T in (Float64, Float32)
        atol_frac = T == Float32 ? 1e-4 : 1e-10

        for n_grains in n_grains_list
            @testset "test_xz N=$n_grains T=$T" begin
                params = DRex.default_params()
                params[:number_of_grains] = n_grains

                get_velocity, get_velocity_gradient = cell_2d("X", "Z", 1.0)

                timestamps, positions, strains, mineral, deformation_gradient =
                    run_cell_olivine_a(
                        params,
                        [0.5, 0.0, -0.75],
                        get_velocity,
                        get_velocity_gradient,
                        [-1.0, 0.0, -1.0],
                        [1.0, 0.0, 1.0],
                        7;
                        seed=seed, float_type=T,
                    )

                # Compute angles between Bingham average a-axis and velocity at each step.
                angles = Float64[]
                for (ori, pos) in zip(mineral.orientations, positions)
                    v = bingham_average(ori; axis="a")
                    vel = get_velocity(NaN, pos)
                    push!(angles, smallest_angle(v, vel))
                end

                # Sanity: all angles in [0, 90].
                @test all(0 .<= angles .<= 90)

                # Grain size fractions sum to 1 at every step.
                for frac in mineral.fractions
                    @test isapprox(sum(frac), 1.0; atol=atol_frac)
                end

                # Assertions for "enough" grains (ε ≈ 3.75 dip is least sensitive to seed).
                if n_grains >= 5000
                    mean_θ_in_dip = mean(angles[35:43])
                    @test mean_θ_in_dip < 12

                    mean_size_in_dip = log10(
                        mean(maximum.(mineral.fractions[35:43])) * n_grains
                    )
                    @test 2 < mean_size_in_dip < 3

                    max_size_post_dip = log10(
                        maximum(maximum.(mineral.fractions[44:end])) * n_grains
                    )
                    @test max_size_post_dip > 3
                end
            end
        end
    end
end

# ═════════════════════════════════════════════════════════════════════════════
# TestDiffusionCreep — diffusion creep regime keeps symmetry stable.
# Three initial orientation types: random, girdle around Y, clustered.
# ═════════════════════════════════════════════════════════════════════════════
@testset "Vortex 2D — DiffusionCreep" begin
    seed = 8816

    params = DRex.default_params()
    params[:gbm_mobility] = 10.0

    n_grains = params[:number_of_grains]

    for T in (Float64, Float32)
        # Build the three initial orientations with the same RNG sequence.
        init_rng = MersenneTwister(8816)
        orientations_init_list = [
            nothing,
            _orientations_girdle_y(n_grains, init_rng; float_type=T),
            _orientations_clustered_y(n_grains, init_rng; float_type=T),
        ]
        labels = ["random", "girdle", "clustered"]

        atol_frac_dc = T == Float32 ? 1e-6 : 1e-15

        for (i, (label, ori_init)) in enumerate(zip(labels, orientations_init_list))
            @testset "cell_olA $label T=$T" begin
                function assert_each(mineral, deformation_gradient)
                    n_steps = length(mineral.fractions)
                    n_steps < 2 && return
                    # Fractions should not change in diffusion creep.
                    @test mineral.fractions[end] ≈ mineral.fractions[end-1] atol=atol_frac_dc
                    # PGR should be approximately stable.
                    p_now, g_now, r_now = symmetry_pgr(mineral.orientations[end])
                    p_prev, g_prev, r_prev = symmetry_pgr(mineral.orientations[end-1])
                    @test isapprox(p_now, p_prev; atol=0.25)
                    @test isapprox(g_now, g_prev; atol=0.25)
                    @test isapprox(r_now, r_prev; atol=0.25)
                    # Symmetry type must remain dominant.
                    if i == 1      # random → R dominates
                        @test r_now > 0.9
                    elseif i == 2  # girdle → G dominates
                        @test g_now > 0.9
                    elseif i == 3  # clustered → P dominates
                        @test p_now > 0.9
                    end
                end

                mineral, strains = run_singlephase(
                    params, seed;
                    regime=matrix_diffusion,
                    orientations_init=ori_init,
                    assert_each=assert_each,
                    float_type=T,
                )

                # Final sanity: strains should be monotonically increasing.
                @test issorted(strains)
            end
        end
    end
end
