# Mineral texture and elasticity computations.

using OrdinaryDiffEq

"""Primary slip axis name for olivine fabrics."""
const OLIVINE_PRIMARY_AXIS = Dict(
    olivine_A => "a",
    olivine_B => "c",
    olivine_C => "c",
    olivine_D => "a",
    olivine_E => "a",
)

"""Slip systems for olivine in conventional order: (plane_normal, slip_direction)."""
const OLIVINE_SLIP_SYSTEMS = (
    ([0,1,0], [1,0,0]),
    ([0,0,1], [1,0,0]),
    ([0,1,0], [0,0,1]),
    ([1,0,0], [0,0,1]),
)

"""Stiffness tensors (Voigt 6×6), units of GPa."""
struct StiffnessTensors
    olivine::Matrix{Float64}
    enstatite::Matrix{Float64}
end

function StiffnessTensors()
    ol = [
        320.71  69.84  71.22  0.0   0.0   0.0;
         69.84 197.25  74.8   0.0   0.0   0.0;
         71.22  74.8  234.32  0.0   0.0   0.0;
          0.0    0.0    0.0  63.77  0.0   0.0;
          0.0    0.0    0.0   0.0  77.67  0.0;
          0.0    0.0    0.0   0.0   0.0  78.36
    ]
    en = [
        236.9  79.6  63.2  0.0  0.0  0.0;
         79.6 180.5  56.8  0.0  0.0  0.0;
         63.2  56.8 230.4  0.0  0.0  0.0;
          0.0   0.0   0.0 84.3  0.0  0.0;
          0.0   0.0   0.0  0.0 79.4  0.0;
          0.0   0.0   0.0  0.0  0.0 80.1
    ]
    return StiffnessTensors(ol, en)
end

"""Get stiffness tensor for a mineral phase."""
function get_stiffness(st::StiffnessTensors, phase::MineralPhase)
    phase == olivine && return st.olivine
    phase == enstatite && return st.enstatite
    throw(ArgumentError("unknown phase: $phase"))
end

"""
    Mineral

Store polycrystal texture data for a single mineral phase.

# Fields
- `phase`: MineralPhase
- `fabric`: MineralFabric
- `regime`: DeformationRegime
- `n_grains`: number of grains
- `fractions`: list of volume fraction snapshots  (Vector{Vector{Float64}})
- `orientations`: list of orientation snapshots (Vector{Array{Float64,3}})
- `seed`: RNG seed for initial random orientations
"""
mutable struct Mineral{T<:AbstractFloat}
    phase::MineralPhase
    fabric::MineralFabric
    regime::DeformationRegime
    n_grains::Int
    fractions::Vector{Vector{T}}
    orientations::Vector{Array{T,3}}
    seed::Union{Int,Nothing}
    lband::Union{Int,Nothing}
    uband::Union{Int,Nothing}
end

function Base.show(io::IO, ::MIME"text/plain", m::Mineral{T}) where T
    n_steps = length(m.fractions)
    frac = m.fractions[end]
    print(io, "Mineral{", T, "} (", m.phase, ", ", m.fabric, ", ", m.regime, ")\n")
    print(io, "  grains    : ", m.n_grains, "\n")
    print(io, "  seed      : ", something(m.seed, "none"), "\n")
    print(io, "  timesteps : ", n_steps, "\n")
    if !isempty(frac)
        mn, mx, md = minimum(frac), maximum(frac), frac[div(length(frac)+1, 2)]
        sorted = sort(frac)
        md = sorted[div(length(sorted)+1, 2)]
        print(io, "  fractions : min=", round(mn; sigdigits=4),
              "  median=", round(md; sigdigits=4),
              "  max=", round(mx; sigdigits=4))
    end
end

function Base.show(io::IO, m::Mineral{T}) where T
    print(io, "Mineral{", T, "}(", m.phase, ", ", m.fabric, ", n=", m.n_grains,
          ", steps=", length(m.fractions), ")")
end

"""
    Mineral(; phase=olivine, fabric=olivine_A, regime=matrix_dislocation,
              n_grains=3500, fractions_init=nothing, orientations_init=nothing, seed=nothing)

Construct a Mineral with optional initial texture.
"""
function Mineral(;
    float_type::Type{T}=Float64,
    phase::MineralPhase=olivine,
    fabric::MineralFabric=olivine_A,
    regime::DeformationRegime=matrix_dislocation,
    n_grains::Int=DefaultParams().number_of_grains,
    fractions_init=nothing,
    orientations_init=nothing,
    seed::Union{Int,Nothing}=nothing,
    lband::Union{Int,Nothing}=nothing,
    uband::Union{Int,Nothing}=nothing,
) where T<:AbstractFloat
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    fractions_T = fractions_init === nothing ?
        fill(one(T) / T(n_grains), n_grains) :
        convert(Vector{T}, fractions_init)
    orientations_T = orientations_init === nothing ?
        _random_orientations(n_grains, rng, T) :
        convert(Array{T,3}, orientations_init)
    if lband === nothing && uband === nothing && n_grains > 4632
        lband = 6000
        uband = 6000
    end
    return Mineral{T}(phase, fabric, regime, n_grains,
                      [fractions_T], [orientations_T],
                      seed, lband, uband)
end

"""Generate random orientation matrices as n×3×3 array."""
function _random_orientations(n::Int, rng::AbstractRNG, ::Type{T}=Float64) where T<:AbstractFloat
    orientations = Array{T,3}(undef, n, 3, 3)
    for g in 1:n
        R = _random_rotation(rng)
        for i in 1:3, j in 1:3
            orientations[g,i,j] = T(R[i,j])
        end
    end
    return orientations
end

"""Generate random rotation matrix using QR decomposition of random Gaussian matrix."""
function _random_rotation(rng::AbstractRNG)
    A = randn(rng, 3, 3)
    Q, R = qr(A)
    # Ensure proper rotation (det = +1)
    Q = Matrix(Q) * Diagonal(sign.(diag(R)))
    if det(Q) < 0
        Q[:, 1] .*= -1
    end
    return Q
end

"""
    update_orientations!(mineral, params, deformation_gradient,
                         get_velocity_gradient, pathline; get_regime=nothing)

Update crystalline orientations and grain volume distribution.
Returns the updated deformation gradient.

- `params` — Dict with parameter keys matching `default_params()`
- `deformation_gradient` — 3×3 matrix
- `get_velocity_gradient(t, x)` — callable returning 3×3 velocity gradient
- `pathline` — (t_start, t_end, get_position) where get_position(t) returns 3D position
"""
function update_orientations!(
    mineral::Mineral{T},
    params::Dict{Symbol,Any},
    deformation_gradient::AbstractMatrix,
    get_velocity_gradient,
    pathline::Tuple;
    get_regime=nothing,
) where T<:AbstractFloat
    time_start, time_end, get_position = pathline
    n_grains = mineral.n_grains

    phase_idx = findfirst(==(mineral.phase), params[:phase_assemblage])
    if phase_idx === nothing
        @warn "phase $(mineral.phase) not in assemblage, skipping"
        return deformation_gradient
    end
    volume_fraction = params[:phase_fractions][phase_idx]

    orientations_prev = mineral.orientations[end]
    fractions_prev = mineral.fractions[end]

    # Flatten state: [F(9), orientations(n*9), fractions(n)] — all promoted to T
    y0 = T.(vcat(
        vec(deformation_gradient),
        vec(orientations_prev),
        fractions_prev,
    ))

    function rhs!(dy, y, p, t)
        position = get_position(t)
        vel_grad = T.(get_velocity_gradient(t, position))

        if get_regime !== nothing
            mineral.regime = get_regime(t, position)
        end

        sr = (vel_grad .+ vel_grad') ./ 2
        sr_max = maximum(abs.(eigvals(Symmetric(sr))))

        F, ori, frac = extract_vars(y, n_grains)
        F_diff = vel_grad * F
        _, V = polar_decompose(F_diff)

        ori_diff = Array{T,3}(undef, n_grains, 3, 3)
        frac_diff = Vector{T}(undef, n_grains)

        derivatives!(ori_diff, frac_diff,
                     mineral.regime, mineral.phase, mineral.fabric, n_grains,
                     ori, frac,
                     sr ./ sr_max, vel_grad ./ sr_max, V,
                     params[:stress_exponent],
                     params[:deformation_exponent],
                     params[:nucleation_efficiency],
                     params[:gbm_mobility],
                     volume_fraction)

        dy[1:9] .= vec(F_diff)
        dy[10:n_grains*9+9] .= vec(ori_diff) .* sr_max
        dy[n_grains*9+10:n_grains*10+9] .= frac_diff .* sr_max
    end

    # Callback to apply GBS after each step
    function gbs_callback!(integrator)
        y = integrator.u
        _, ori, frac = extract_vars(y, n_grains)
        ori, frac = apply_gbs!(ori, frac, params[:gbs_threshold],
                                orientations_prev, n_grains)
        y[10:n_grains*9+9] .= vec(ori)
        y[n_grains*9+10:n_grains*10+9] .= frac
        u_modified!(integrator, true)
    end

    tspan = (Float64(time_start), Float64(time_end))
    prob = ODEProblem(rhs!, y0, tspan)

    # Use a callback that fires after every accepted step
    cb = DiscreteCallback((u, t, integrator) -> true, gbs_callback!)

    sol = solve(prob, Tsit5();
                callback=cb,
                abstol=1e-4,
                reltol=1e-6,
                dtmin=abs(time_end - time_start) * 1e-12)

    y_final = sol.u[end]
    F_new, ori_new, frac_new = extract_vars(y_final, n_grains)
    push!(mineral.orientations, ori_new)
    push!(mineral.fractions, frac_new)
    return F_new
end

"""
    update_all!(minerals, params, deformation_gradient,
                get_velocity_gradient, pathline; get_regime=nothing)

Update orientations and volume distributions for all mineral phases.
Returns the updated deformation gradient tensor.
"""
function update_all!(
    minerals,
    params::Dict{Symbol,Any},
    deformation_gradient::AbstractMatrix,
    get_velocity_gradient,
    pathline::Tuple;
    get_regime=nothing,
)
    new_F = deformation_gradient
    for mineral in minerals
        new_F = update_orientations!(
            mineral, params, deformation_gradient,
            get_velocity_gradient, pathline;
            get_regime=get_regime,
        )
    end
    return new_F
end

"""
    voigt_averages(minerals, phase_assemblage, phase_fractions; elastic_tensors=StiffnessTensors())

Calculate elastic tensors as Voigt averages of a collection of minerals.
Returns a 6×6 Voigt matrix.
"""
function voigt_averages(
    minerals::Vector{Mineral},
    phase_assemblage::Vector{MineralPhase},
    phase_fractions::Vector{Float64};
    elastic_tensors::StiffnessTensors=StiffnessTensors(),
)
    voigt_sum = zeros(6, 6)
    for mineral in minerals
        p_idx = findfirst(==(mineral.phase), phase_assemblage)
        p_idx === nothing && continue
        C_ref = get_stiffness(elastic_tensors, mineral.phase)
        C_tensor = voigt_to_elastic_tensor(C_ref)
        orientations = mineral.orientations[end]
        fractions = mineral.fractions[end]
        n = mineral.n_grains
        C_avg = zeros(3, 3, 3, 3)
        @inbounds for g in 1:n
            R = zeros(3, 3)
            for i in 1:3, j in 1:3
                R[i,j] = orientations[g, i, j]
            end
            # Transpose: orientation matrices are passive (lab→grain),
            # but we need to rotate the tensor from grain frame to lab frame.
            C_rot = rotate_tensor(C_tensor, R')
            for i in 1:3, j in 1:3, k in 1:3, l in 1:3
                C_avg[i,j,k,l] += fractions[g] * C_rot[i,j,k,l]
            end
        end
        voigt_sum .+= phase_fractions[p_idx] .* elastic_tensor_to_voigt(C_avg)
    end
    return voigt_sum
end

"""
    peridotite_solidus(pressure; fit="Hirschmann2000")

Get peridotite solidus temperature (K) based on experimental fits. Pressure in GPa.
"""
function peridotite_solidus(pressure::Real; fit::String="Hirschmann2000")
    if fit == "Herzberg2000"
        return 1086 - 5.7 * pressure + 390 * log(pressure)
    elseif fit == "Hirschmann2000"
        return -5.104 * pressure^2 + 132.899 * pressure + 1120.661
    elseif fit == "Duvernay2024"
        return -6.8 * pressure^2 + 141.4 * pressure + 1101.3
    end
    throw(ArgumentError("unsupported fit '$fit'"))
end
