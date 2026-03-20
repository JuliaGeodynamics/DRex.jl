using Test
using LinearAlgebra
using Statistics
using DRex
using DRex: simple_shear_2d, smallest_angle, bingham_average,
              misorientation_index, resample_orientations

# Parameters matching ParamsFraters2021 from the Python PyDRex mock module.
# Reference: Fraters & Billen, 2021 (https://doi.org/10.1029/2021gc009846)
function params_fraters2021(; number_of_grains=5000)
    params = DRex.default_params()
    params[:phase_assemblage] = [olivine, enstatite]
    params[:phase_fractions] = [0.7, 0.3]
    params[:stress_exponent] = 1.5
    params[:deformation_exponent] = 3.5
    params[:gbm_mobility] = 125.0
    params[:gbs_threshold] = 0.3
    params[:nucleation_efficiency] = 5.0
    params[:number_of_grains] = number_of_grains
    return params
end

# 1000 unique seeds from PyDRex data/rng/seeds.scsv (first 500 used by the paper).
const SEEDS_1000 = [
    101,121,1218,3052,4232,4557,5675,8236,8711,10113,10325,12198,12467,12762,
    13178,13515,14756,15181,16535,16608,16913,19182,20342,20442,20519,20692,
    20893,20909,21232,21400,21590,24094,26169,26974,28635,31173,31719,33355,
    34472,34851,36199,43410,43879,44099,44409,44886,45928,46708,46908,47611,
    48561,49754,50018,50036,50296,50569,50669,51406,51623,52535,53478,54061,
    54882,55276,56346,56394,57961,59776,60397,60411,60955,63519,63566,64142,
    64206,67640,70280,70618,71666,71838,72127,74830,75257,75269,75509,76437,
    76509,77002,77468,79002,79073,80782,80962,83466,84510,85192,86036,86446,
    87099,87126,90646,90949,91002,91827,92409,92420,94664,95791,97951,98873,
    100710,101899,104164,104963,105813,106577,107646,107784,108242,109640,
    110784,111304,111478,112182,113363,113535,113663,114704,115470,116042,
    116335,117661,118109,120051,120765,121729,124533,126017,126657,129519,
    129878,130272,130950,131319,132147,132333,133142,134203,134561,134679,
    136585,136866,137027,138114,138160,139623,141670,142614,142630,142677,
    143678,144881,144972,145400,145478,145637,146239,147742,148686,149713,
    150432,152078,152272,152827,153859,154469,154522,154661,157064,159107,
    161425,161575,162130,162194,163472,163863,165201,165590,166203,167019,
    168486,170180,171159,172981,173861,174256,175802,176530,176625,176661,
    176762,177884,177902,178679,179030,179426,179772,180748,182886,183850,
    184169,185118,185422,185920,186416,187608,188228,191526,191953,193226,
    194848,195211,199081,199740,200064,200969,201228,201897,202641,203371,
    205585,206823,207006,207651,207694,209018,212606,213511,213662,214205,
    215936,216028,217889,218486,218898,219503,220767,221923,224888,227310,
    227658,227774,229028,229228,229427,232214,233523,234135,234590,235239,
    236275,236990,237639,240855,241246,241308,241519,242014,244336,245199,
    247561,247885,250933,251730,254091,254312,254421,254663,255256,255659,
    255767,255813,255850,257203,257382,259837,260294,260583,261521,261725,
    261746,262517,262591,262643,263132,263585,263664,264112,266552,267007,
    267096,267664,269002,271054,271783,271879,272072,272103,273326,275228,
    275445,276563,278368,278855,278877,280582,280807,283301,284463,284933,
    287994,289416,291901,292819,295332,295940,296427,297562,298877,300906,
    301643,302043,302942,304463,305273,306387,310707,311713,313520,314883,
    315141,317174,320123,320152,320651,322318,323683,324677,325364,326698,
    327262,327333,327578,329023,329785,332494,332962,334043,335832,336802,
    336919,338138,338987,343049,345528,346444,346585,347563,349501,352775,
    355260,356715,357839,358841,358865,358948,359893,360175,360396,361291,
    362095,362290,363809,364865,365604,370152,371380,372149,373716,373928,
    374838,374909,378247,379464,379554,381361,382202,382449,382599,382866,
    383357,383533,383853,385205,385469,386029,388625,389027,390437,391075,
    391332,391520,393397,393906,395389,398151,398261,398460,399329,399601,
    402816,403551,403990,404522,405987,412442,412630,412656,413271,413467,
    415170,415344,415773,415936,418239,418388,418701,418778,419633,420532,
    421043,421263,423522,423915,425898,427400,427946,428117,429464,429593,
    433933,433983,434484,435212,435348,435604,437148,437283,438641,439831,
    441009,441581,441979,442138,442804,444584,447019,447986,449235,449525,
    450820,450822,452296,452971,454420,454661,455744,455876,456950,458972,
    459948,462615,463811,464004,465753,466349,467139,467531,467584,467710,
    468484,470558,470682,470934,471145,471826,471955,472864,473529,475202,
]

"""
    run_simple_shear_3d(params, timestamps, get_L_initial, get_L_final, switch_time;
                        seed=nothing, float_type=Float64)

Run simulation with stationary particle in a velocity gradient that switches direction.

Initial velocity gradient is applied until `switch_time`, then `get_L_final` is used.
Returns `(olivine_mineral, enstatite_mineral)` where the second may be `nothing`
if enstatite is not in the phase assemblage.
"""
function run_simple_shear_3d(
    params, timestamps, get_L_initial, get_L_final, switch_time;
    seed=nothing, float_type::Type{T}=Float64,
) where T<:AbstractFloat
    get_position = t -> fill(NaN, 3)
    n_grains = params[:number_of_grains]

    ol = Mineral(
        float_type=T,
        phase=olivine,
        fabric=olivine_A,
        regime=matrix_dislocation,
        n_grains=n_grains,
        seed=seed,
    )

    # Check if enstatite is in the assemblage
    has_enstatite = enstatite in params[:phase_assemblage]
    ens = if has_enstatite
        Mineral(
            float_type=T,
            phase=enstatite,
            fabric=enstatite_AB,
            regime=matrix_dislocation,
            n_grains=n_grains,
            seed=seed,
        )
    else
        nothing
    end

    deformation_gradient = Matrix{T}(I, 3, 3)

    for t in 2:length(timestamps)
        time = timestamps[t-1]
        get_L = time > switch_time ? get_L_final : get_L_initial

        if ens !== nothing
            update_orientations!(
                ens, params, deformation_gradient, get_L,
                (time, timestamps[t], get_position),
            )
        end
        deformation_gradient = update_orientations!(
            ol, params, deformation_gradient, get_L,
            (time, timestamps[t], get_position),
        )
    end
    return ol, ens
end

@testset "Simple Shear 3D — Fraters2021 direction change" begin
    # Original values from Fraters & Billen, 2021 (Figure 5):
    # n_grains = 5000
    # n_timestamps = 500
    # seeds = SEEDS_1000[1:500]  # 500 runs as per the paper.
    # n_samples_mindex = 1000

    # Reduced values for fast testing:
    n_grains = 500
    n_timestamps = 50
    n_samples_mindex = 200
    strain_rate = 5e-7  # per year
    timestamps = range(0, 5e6, length=n_timestamps)

    _, get_L_initial = simple_shear_2d("X", "Z", strain_rate)  # du/dz
    _, get_L_final   = simple_shear_2d("Y", "X", strain_rate)  # dv/dx

    seeds = SEEDS_1000[1:3]  # Reduced from 500 for fast testing.
    n_seeds = length(seeds)

    for T in (Float64, Float32)
        for switch_time_Ma in [0.0, 1.0, 2.5, Inf]
            @testset "switch_time=$(switch_time_Ma) Ma T=$T" begin
                params = params_fraters2021(; number_of_grains=n_grains)
                switch_time = switch_time_Ma * 1e6
                has_enstatite = enstatite in params[:phase_assemblage]

                # Output arrays for [100] mean direction angles (per seed, per timestep).
                olA_from_proj_XZ = zeros(n_seeds, n_timestamps)
                olA_from_proj_YX = zeros(n_seeds, n_timestamps)
                ens_from_proj_XZ = zeros(n_seeds, n_timestamps)
                ens_from_proj_YX = zeros(n_seeds, n_timestamps)
                # Output arrays for M-index (CPO strength).
                olA_strength = zeros(n_seeds, n_timestamps)
                ens_strength = zeros(n_seeds, n_timestamps)

                # Run seeds in parallel — each seed creates independent Mineral objects.
                Threads.@threads for s in 1:n_seeds
                    seed = seeds[s]
                    ol, ens_mineral = run_simple_shear_3d(
                        params, timestamps, get_L_initial, get_L_final, switch_time;
                        seed=seed, float_type=T,
                    )

                    # Post-process olivine: Bingham average a-axis at each timestep.
                    olA_resampled, _ = resample_orientations(
                        ol.orientations, ol.fractions; seed=seed,
                    )
                    for i in 1:n_timestamps
                        v = bingham_average(olA_resampled[i]; axis="a")
                        olA_from_proj_XZ[s, i] = smallest_angle(v, v .- v .* [0, 1, 0])
                        olA_from_proj_YX[s, i] = smallest_angle(v, v .- v .* [0, 0, 1])
                    end

                    # M-index at each timestep.
                    olA_down, _ = resample_orientations(
                        ol.orientations, ol.fractions; seed=seed, n_samples=n_samples_mindex,
                    )
                    for i in 1:n_timestamps
                        olA_strength[s, i] = misorientation_index(olA_down[i], orthorhombic)
                    end

                    # Post-process enstatite.
                    if has_enstatite && ens_mineral !== nothing
                        ens_resampled, _ = resample_orientations(
                            ens_mineral.orientations, ens_mineral.fractions; seed=seed,
                        )
                        for i in 1:n_timestamps
                            v = bingham_average(ens_resampled[i]; axis="a")
                            ens_from_proj_XZ[s, i] = smallest_angle(v, v .- v .* [0, 1, 0])
                            ens_from_proj_YX[s, i] = smallest_angle(v, v .- v .* [0, 0, 1])
                        end
                        ens_down, _ = resample_orientations(
                            ens_mineral.orientations, ens_mineral.fractions; seed=seed,
                            n_samples=n_samples_mindex,
                        )
                        for i in 1:n_timestamps
                            ens_strength[s, i] = misorientation_index(
                                ens_down[i], orthorhombic,
                            )
                        end
                    end
                end

                # Ensemble averages over seeds.
                olA_from_proj_XZ_mean = mean(olA_from_proj_XZ, dims=1)[1, :]
                olA_from_proj_YX_mean = mean(olA_from_proj_YX, dims=1)[1, :]
                olA_strength_mean     = mean(olA_strength, dims=1)[1, :]

                # Sanity checks: angles in [0, 90].
                @test all(0 .<= olA_from_proj_XZ_mean .<= 90)
                @test all(0 .<= olA_from_proj_YX_mean .<= 90)

                # M-index in [0, 1] and texture develops.
                @test all(0 .<= olA_strength_mean .<= 1)
                @test olA_strength_mean[end] > 0.01

                if switch_time_Ma == Inf
                    # Pure initial shear (XZ): a-axis aligns near XZ plane.
                    @test olA_from_proj_XZ_mean[end] < 30
                elseif switch_time_Ma == 0.0
                    # Pure final shear (YX): a-axis aligns near YX plane.
                    @test olA_from_proj_YX_mean[end] < 30
                end

                if has_enstatite
                    ens_from_proj_XZ_mean = mean(ens_from_proj_XZ, dims=1)[1, :]
                    ens_from_proj_YX_mean = mean(ens_from_proj_YX, dims=1)[1, :]
                    ens_strength_mean     = mean(ens_strength, dims=1)[1, :]
                    @test all(0 .<= ens_from_proj_XZ_mean .<= 90)
                    @test all(0 .<= ens_from_proj_YX_mean .<= 90)
                    @test all(0 .<= ens_strength_mean .<= 1)
                end
            end
        end
    end
end
