using Test
using LinearAlgebra
using Statistics
using DRex
using DRex: corner_2d, get_pathline, strain_increment, update_orientations!,
              resample_orientations, bingham_average, smallest_angle,
              misorientation_index, OLIVINE_PRIMARY_AXIS

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run a single pathline in 2D corner flow (olivine A).
# Ported from TestOlivineA.run() in Python test_corner_flow_2d.py.
# ─────────────────────────────────────────────────────────────────────────────
function run_corner_olivine_a(
    params, seed, get_velocity, get_velocity_gradient,
    min_coords, max_coords, max_strain, n_timesteps, final_location,
)
    mineral = Mineral(
        phase=olivine, fabric=olivine_A, regime=matrix_dislocation,
        n_grains=params[:number_of_grains], seed=seed,
    )
    deformation_gradient = Matrix{Float64}(I, 3, 3)

    timestamps, get_position = get_pathline(
        final_location, get_velocity, get_velocity_gradient,
        min_coords, max_coords;
        max_strain=Float64(max_strain),
        regular_steps=n_timesteps,
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

# ═════════════════════════════════════════════════════════════════════════════
# TestOlivineA — 2D corner flow tests for A-type olivine.
#
# The Python test (test_steady4) is marked @slow (11 CPU hrs, 60GB RAM) and
# has NO assertions — it only computes CPO and saves figures/data.
# The full simulation is also available in examples/standalone/cornerflow_simple.jl.
#
# Here we run a reduced version with actual sanity-check assertions.
# Original paper values: 5000 grains, 4 pathlines, n_timesteps=50
# Reduced values:        500 grains,  2 pathlines, n_timesteps=20
# ═════════════════════════════════════════════════════════════════════════════
@testset "Corner Flow 2D — OlivineA" begin
    seed = 8816

    # Plate speed: 2 cm/yr → m/s.
    plate_speed = 2.0 / (100.0 * 365.0 * 86400.0)
    domain_height = 2.0e5
    domain_width  = 1.0e6

    # Original paper values (commented out for fast testing):
    # n_grains = 5000
    # n_timesteps = 50
    # z_fracs = (-0.1, -0.3, -0.54, -0.78)

    # Reduced values for fast testing:
    n_grains = 500
    n_timesteps = 20
    z_fracs = (-0.3, -0.54)  # 2 of the 4 pathlines
    max_strain = 10.0

    params = DRex.default_params()
    params[:number_of_grains] = n_grains

    get_velocity, get_velocity_gradient = corner_2d("X", "Z", plate_speed)

    min_coords = [0.0, 0.0, -domain_height]
    max_coords = [domain_width, 0.0, 0.0]

    for z_frac in z_fracs
        final_location = [domain_width, 0.0, z_frac * domain_height]

        @testset "z_exit=$(round(z_frac * domain_height / 1e3; digits=0)) km" begin
            timestamps, positions, strains, mineral, deformation_gradient =
                run_corner_olivine_a(
                    params, seed, get_velocity, get_velocity_gradient,
                    min_coords, max_coords, max_strain, n_timesteps, final_location,
                )

            n_steps = length(timestamps)

            # Strains should be monotonically increasing.
            @test issorted(strains)
            @test strains[end] > 0

            # Correct number of orientation/fraction snapshots.
            @test length(mineral.orientations) == n_steps
            @test length(mineral.fractions) == n_steps

            # Grain fractions sum to 1 at every step.
            for frac in mineral.fractions
                @test isapprox(sum(frac), 1.0; atol=1e-10)
            end

            # Post-process: resample, compute M-index and Bingham angles.
            orientations_resampled, _ = resample_orientations(
                mineral.orientations, mineral.fractions; seed=seed,
            )
            primary_axis = OLIVINE_PRIMARY_AXIS[mineral.fabric]

            m_indices = Float64[]
            angles = Float64[]
            for idx in 1:n_steps
                push!(m_indices, misorientation_index(
                    orientations_resampled[idx], orthorhombic,
                ))
                dir_mean = bingham_average(
                    orientations_resampled[idx]; axis=primary_axis,
                )
                push!(angles, smallest_angle(dir_mean, [1.0, 0.0, 0.0]))
            end

            # M-index in [0, 1].
            @test all(0 .<= m_indices .<= 1)
            # Texture should develop: final M-index > 0.
            @test m_indices[end] > 0.01

            # Bingham angles in [0, 90].
            @test all(0 .<= angles .<= 90)

            # Deformation gradient should remain a proper 3×3 matrix.
            @test size(deformation_gradient) == (3, 3)
            @test det(deformation_gradient) > 0
        end
    end
end
