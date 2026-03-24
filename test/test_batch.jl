using Test
using LinearAlgebra
using Statistics
using KernelAbstractions
using DRex
using DRex: corner_2d, get_pathline, strain_increment, update_orientations!,
              run_pathlines_batch!, resample_orientations, misorientation_index

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

function _corner_pathline_data(final_location, f_velocity, f_velocity_grad,
                                min_coords, max_coords, ::Type{T}=Float64;
                                n_timesteps=20, max_strain=10.0) where T
    timestamps, f_position = get_pathline(
        final_location, f_velocity, f_velocity_grad,
        min_coords, max_coords;
        max_strain=max_strain, regular_steps=n_timesteps,
    )
    positions    = [collect(f_position(t)) for t in timestamps]
    vgs          = [Matrix{T}(f_velocity_grad(NaN, x)) for x in positions]
    timestamps_T = Vector{T}(timestamps)
    return (timestamps_T, positions, vgs)
end

# Run the sequential path (update_orientations! per step) and return minerals + strains.
function _run_sequential(pathline_data, params, ::Type{T}=Float64) where T
    timestamps, positions, vgs = pathline_data
    n_steps = length(timestamps)

    olA = Mineral(float_type=T, phase=olivine, fabric=olivine_A,
                  regime=matrix_dislocation, n_grains=params[:number_of_grains], seed=42)
    ens = Mineral(float_type=T, phase=enstatite, fabric=enstatite_AB,
                  regime=matrix_dislocation, n_grains=params[:number_of_grains], seed=43)
    F = Matrix{T}(I, 3, 3)

    strains = zeros(T, n_steps)
    for i in 2:n_steps
        t0, t1 = Float64(timestamps[i-1]), Float64(timestamps[i])
        vg0, vg1 = Matrix{T}(vgs[i-1]), Matrix{T}(vgs[i])
        get_pos = let p0=positions[i-1], p1=positions[i]
            t -> (1.0 - (t-t0)/(t1-t0)) .* p0 .+ ((t-t0)/(t1-t0)) .* p1
        end
        get_vg = let t0=t0, t1=t1, vg0=vg0, vg1=vg1
            (t, _) -> let α=T((t-t0)/(t1-t0)); (1-α).*vg0 .+ α.*vg1 end
        end
        F = update_all!([olA, ens], params, F, get_vg, (t0, t1, get_pos))
        strains[i] = strains[i-1] + T(strain_increment(t1-t0, Float64.(vg1)))
    end
    return [olA, ens], strains, F
end

# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

@testset "Batch CPU path — run_pathlines_batch!" begin
    plate_speed   = 2.0 / (100.0 * 365.0 * 86400.0)
    domain_height = 2.0e5
    domain_width  = 1.0e6
    min_coords    = [0.0, 0.0, -domain_height]
    max_coords    = [domain_width, 0.0, 0.0]

    f_velocity, f_velocity_grad = corner_2d("X", "Z", plate_speed)

    params = DRex.default_params()
    params[:number_of_grains] = 300   # small for speed
    params[:phase_assemblage] = [olivine, enstatite]
    params[:phase_fractions]  = [0.7, 0.3]

    final_location = [domain_width, 0.0, -0.3 * domain_height]

    for T in (Float64, Float32)
        atol_frac = T == Float32 ? 1e-4 : 1e-10

        @testset "CPU batch path T=$T" begin
            pathline_data = _corner_pathline_data(
                final_location, f_velocity, f_velocity_grad,
                min_coords, max_coords, T; n_timesteps=20,
            )
            timestamps, positions, vgs = pathline_data
            n_steps = length(timestamps)

            # ── Build per-tracer mineral lists ────────────────────────────
            minerals = [
                Mineral(float_type=T, phase=olivine, fabric=olivine_A,
                        regime=matrix_dislocation, n_grains=params[:number_of_grains], seed=42),
                Mineral(float_type=T, phase=enstatite, fabric=enstatite_AB,
                        regime=matrix_dislocation, n_grains=params[:number_of_grains], seed=43),
            ]
            minerals_per_tracer = [minerals]

            # ── Run batch path ────────────────────────────────────────────
            strains_batch = run_pathlines_batch!(
                minerals_per_tracer, params, [pathline_data]; backend=CPU(),
            )

            olA_batch = minerals_per_tracer[1][1]

            # Correct number of snapshots
            @test length(olA_batch.orientations) == n_steps
            @test length(olA_batch.fractions) == n_steps

            # Grain fractions sum to 1 at every snapshot
            for frac in olA_batch.fractions
                @test isapprox(sum(frac), 1; atol=atol_frac)
            end

            # Strains are monotonically non-decreasing and positive at the end
            @test issorted(strains_batch[1])
            @test strains_batch[1][end] > 0

            # M-index is in [0, 1]
            m_final = misorientation_index(olA_batch.orientations[end], orthorhombic)
            @test 0 <= m_final <= 1

            # Texture should develop: M-index > 0 at the end
            @test m_final > 0.01

            # ── Compare batch vs sequential (CPU batch uses same Tsit5) ──
            minerals_seq, strains_seq, F_seq = _run_sequential(pathline_data, params, T)
            olA_seq = minerals_seq[1]

            m_seq = misorientation_index(olA_seq.orientations[end], orthorhombic)

            # On CPU, batch and sequential use the same Tsit5 integrator,
            # so M-indices should match closely.
            @test isapprox(m_final, m_seq; atol=0.02)

            # Accumulated strains should also match
            @test isapprox(strains_batch[1][end], strains_seq[end]; atol=atol_frac)
        end
    end
end

@testset "Batch — two tracers, multi-phase" begin
    plate_speed   = 2.0 / (100.0 * 365.0 * 86400.0)
    domain_height = 2.0e5
    domain_width  = 1.0e6
    min_coords    = [0.0, 0.0, -domain_height]
    max_coords    = [domain_width, 0.0, 0.0]

    f_velocity, f_velocity_grad = corner_2d("X", "Z", plate_speed)

    params = DRex.default_params()
    params[:number_of_grains] = 200
    params[:phase_assemblage] = [olivine, enstatite]
    params[:phase_fractions]  = [0.7, 0.3]

    final_locs = [
        [domain_width, 0.0, -0.3 * domain_height],
        [domain_width, 0.0, -0.54 * domain_height],
    ]

    pathlines = [
        _corner_pathline_data(loc, f_velocity, f_velocity_grad, min_coords, max_coords,
                               Float64; n_timesteps=15)
        for loc in final_locs
    ]

    # All tracers must share the same n_steps — verified by run_pathlines_batch!
    n_steps = length(pathlines[1][1])
    @test length(pathlines[2][1]) == n_steps

    minerals_per_tracer = [
        [
            Mineral(float_type=Float64, phase=olivine, fabric=olivine_A,
                    regime=matrix_dislocation, n_grains=params[:number_of_grains],
                    seed=10+i),
            Mineral(float_type=Float64, phase=enstatite, fabric=enstatite_AB,
                    regime=matrix_dislocation, n_grains=params[:number_of_grains],
                    seed=20+i),
        ]
        for i in 1:2
    ]

    strains_all = run_pathlines_batch!(
        minerals_per_tracer, params, pathlines; backend=CPU(),
    )

    @test length(strains_all) == 2

    for ti in 1:2
        olA = minerals_per_tracer[ti][1]
        ens = minerals_per_tracer[ti][2]

        @test length(olA.orientations) == n_steps
        @test length(ens.orientations) == n_steps

        for frac in olA.fractions
            @test isapprox(sum(frac), 1; atol=1e-10)
        end

        @test issorted(strains_all[ti])
        @test strains_all[ti][end] > 0
    end
end
