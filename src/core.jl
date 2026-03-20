# Core D-Rex functions, enums and parameter types.
#
# The `derivatives!` function implements the core D-Rex solver, which computes the
# crystallographic rotation rate and changes in fractional grain volumes.
# All inner-loop routines operate on StaticArrays for zero-allocation performance.

# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

"""Supported mineral phases."""
@enum MineralPhase begin
    olivine   = 0
    enstatite = 1
end

"""Supported mineral fabrics / CRSS distributions."""
@enum MineralFabric begin
    olivine_A    = 0
    olivine_B    = 1
    olivine_C    = 2
    olivine_D    = 3
    olivine_E    = 4
    enstatite_AB = 5
end

"""Deformation mechanism regimes."""
@enum DeformationRegime begin
    min_viscosity        = 0
    matrix_diffusion     = 1
    boundary_diffusion   = 2
    sliding_diffusion    = 3
    matrix_dislocation   = 4
    sliding_dislocation  = 5
    frictional_yielding  = 6
    max_viscosity        = 7
end

# ──────────────────────────────────────────────────────────────────────────────
# Levi-Civita symbol (compile-time constant)
# ──────────────────────────────────────────────────────────────────────────────

const PERMUTATION_SYMBOL = SArray{Tuple{3,3,3},Int8}(
    # [i,j,k] with column-major ordering
    # ε_{1,j,k}
     0,  0, 0,   0, 0,-1,   0, 1, 0,
    # ε_{2,j,k}
     0,  0, 1,   0, 0, 0,  -1, 0, 0,
    # ε_{3,j,k}
     0, -1, 0,   1, 0, 0,   0, 0, 0,
)

# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────

"""Default parameters for DRex simulations."""
Base.@kwdef struct DefaultParams
    phase_assemblage::NTuple{1,MineralPhase} = (olivine,)
    phase_fractions::NTuple{1,Float64} = (1.0,)
    stress_exponent::Float64 = 1.5
    deformation_exponent::Float64 = 3.5
    gbm_mobility::Float64 = 125.0
    gbs_threshold::Float64 = 0.3
    nucleation_efficiency::Float64 = 5.0
    number_of_grains::Int = 3500
    initial_olivine_fabric::MineralFabric = olivine_A
end

"""Convert DefaultParams to a NamedTuple (mutable-like usage)."""
function as_dict(p::DefaultParams)
    return (
        phase_assemblage = collect(p.phase_assemblage),
        phase_fractions = collect(p.phase_fractions),
        stress_exponent = p.stress_exponent,
        deformation_exponent = p.deformation_exponent,
        gbm_mobility = p.gbm_mobility,
        gbs_threshold = p.gbs_threshold,
        nucleation_efficiency = p.nucleation_efficiency,
        number_of_grains = p.number_of_grains,
        initial_olivine_fabric = p.initial_olivine_fabric,
    )
end

"""Return default parameters as a Dict{Symbol,Any} for easy mutation."""
function default_params()
    p = DefaultParams()
    return Dict{Symbol,Any}(
        :phase_assemblage      => collect(p.phase_assemblage),
        :phase_fractions       => collect(p.phase_fractions),
        :stress_exponent       => p.stress_exponent,
        :deformation_exponent  => p.deformation_exponent,
        :gbm_mobility          => p.gbm_mobility,
        :gbs_threshold         => p.gbs_threshold,
        :nucleation_efficiency => p.nucleation_efficiency,
        :number_of_grains      => p.number_of_grains,
        :initial_olivine_fabric => p.initial_olivine_fabric,
    )
end

# ──────────────────────────────────────────────────────────────────────────────
# CRSS (Critical Resolved Shear Stress)
# ──────────────────────────────────────────────────────────────────────────────

"""
    get_crss(phase, fabric) -> SVector{4,Float64}

Get normalised Critical Resolved Shear Stress for the given mineral phase and fabric.
Returns a static vector (allocation-free).
"""
function get_crss(phase::MineralPhase, fabric::MineralFabric,
                  ::Type{T}=Float64)::SVector{4,T} where T<:AbstractFloat
    if phase == olivine
        if fabric == olivine_A
            return SVector{4,T}(1, 2, 3, Inf)
        elseif fabric == olivine_B
            return SVector{4,T}(3, 2, 1, Inf)
        elseif fabric == olivine_C
            return SVector{4,T}(3, 2, Inf, 1)
        elseif fabric == olivine_D
            return SVector{4,T}(1, 1, 3, Inf)
        else  # olivine_E
            return SVector{4,T}(3, 1, 2, Inf)
        end
    else  # enstatite — enstatite_AB
        return SVector{4,T}(Inf, Inf, Inf, 1)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Allocation-free inner routines (all operate on SMatrix / SVector)
# ──────────────────────────────────────────────────────────────────────────────

"""
    _get_slip_invariants(strain_rate, orientation) -> SVector{4,Float64}

Calculate strain rate invariants for the four slip systems. Allocation-free.
"""
@inline function _get_slip_invariants(
    strain_rate::SMatrix{3,3,T,9},
    orientation::SMatrix{3,3,T,9}
)::SVector{4,T} where T<:AbstractFloat
    inv1 = zero(T)
    inv2 = zero(T)
    inv3 = zero(T)
    inv4 = zero(T)
    @inbounds for j in 1:3, i in 1:3
        s = strain_rate[i,j]
        # (010)[100]
        inv1 += s * orientation[1,i] * orientation[2,j]
        # (001)[100]
        inv2 += s * orientation[1,i] * orientation[3,j]
        # (010)[001]
        inv3 += s * orientation[3,i] * orientation[2,j]
        # (100)[001]
        inv4 += s * orientation[3,i] * orientation[1,j]
    end
    return SVector(inv1, inv2, inv3, inv4)
end

"""
    _get_deformation_rate(phase, orientation, slip_rates) -> SMatrix{3,3}

Calculate deformation rate tensor (Schmid tensor). Allocation-free.
"""
@inline function _get_deformation_rate(
    ::MineralPhase,
    orientation::SMatrix{3,3,T,9},
    slip_rates::SVector{4,T}
)::SMatrix{3,3,T,9} where T<:AbstractFloat
    r1 = slip_rates[1]; r2 = slip_rates[2]; r3 = slip_rates[3]; r4 = slip_rates[4]
    return SMatrix{3,3,T,9}(ntuple(Val(9)) do idx
        j, i = fldmod1(idx, 3)  # column-major: (col, row)
        2 * (r1 * orientation[1,i] * orientation[2,j] +
             r2 * orientation[1,i] * orientation[3,j] +
             r3 * orientation[3,i] * orientation[2,j] +
             r4 * orientation[3,i] * orientation[1,j])
    end)
end

"""
    _get_slip_rate_softest(deformation_rate, velocity_gradient) -> Float64

Calculate dimensionless slip rate on the softest slip system. Allocation-free.
"""
@inline function _get_slip_rate_softest(
    deformation_rate::SMatrix{3,3,T,9},
    velocity_gradient::SMatrix{3,3,T,9}
)::T where T<:AbstractFloat
    enumerator = zero(T)
    denominator = zero(T)
    @inbounds for j in 1:3
        k = mod1(j + 1, 3)
        enumerator -= (velocity_gradient[j,k] - velocity_gradient[k,j]) *
                      (deformation_rate[j,k] - deformation_rate[k,j])
        denominator -= (deformation_rate[j,k] - deformation_rate[k,j])^2
        for l in 1:3
            enumerator += 2 * deformation_rate[j,l] * velocity_gradient[j,l]
            denominator += 2 * deformation_rate[j,l]^2
        end
    end
    if abs(denominator) < eps(T) * T(10)
        return zero(T)
    end
    return enumerator / denominator
end

"""
    _get_slip_rates_olivine(invariants, slip_indices, crss, deformation_exponent) -> SVector{4}

Calculate relative slip rates of the active slip systems for olivine. Allocation-free.
"""
@inline function _get_slip_rates_olivine(
    invariants::SVector{4,T},
    slip_indices::SVector{4,Int32},
    crss::SVector{4,T},
    deformation_exponent::T
)::SVector{4,T} where T<:AbstractFloat
    i_inac = slip_indices[1]
    i_min  = slip_indices[2]
    i_int  = slip_indices[3]
    i_max  = slip_indices[4]

    prefactor = crss[i_max] / invariants[i_max]
    ratio_min = prefactor * invariants[i_min] / crss[i_min]
    ratio_int = prefactor * invariants[i_int] / crss[i_int]

    r_min = ratio_min * abs(ratio_min)^(deformation_exponent - one(T))
    r_int = ratio_int * abs(ratio_int)^(deformation_exponent - one(T))

    # Build without MVector: select each slot by comparing against known indices.
    # Fully unrolled by the compiler (GPU-safe, no heap allocation).
    v1 = Int32(1) == i_inac ? zero(T) : Int32(1) == i_min ? r_min : Int32(1) == i_int ? r_int : one(T)
    v2 = Int32(2) == i_inac ? zero(T) : Int32(2) == i_min ? r_min : Int32(2) == i_int ? r_int : one(T)
    v3 = Int32(3) == i_inac ? zero(T) : Int32(3) == i_min ? r_min : Int32(3) == i_int ? r_int : one(T)
    v4 = Int32(4) == i_inac ? zero(T) : Int32(4) == i_min ? r_min : Int32(4) == i_int ? r_int : one(T)
    return SVector{4,T}(v1, v2, v3, v4)
end

"""
    _get_orientation_change(orientation, velocity_gradient, deformation_rate, slip_rate_softest)

Calculate the rotation rate for a grain. Allocation-free.
"""
@inline function _get_orientation_change(
    orientation::SMatrix{3,3,T,9},
    velocity_gradient::SMatrix{3,3,T,9},
    deformation_rate::SMatrix{3,3,T,9},
    slip_rate_softest::T
)::SMatrix{3,3,T,9} where T<:AbstractFloat
    # Spin vector
    spin = MVector{3,T}(undef)
    @inbounds for j in 1:3
        r = mod1(j + 1, 3)
        s = mod1(j + 2, 3)
        spin[j] = ((velocity_gradient[s,r] - velocity_gradient[r,s]) -
                    (deformation_rate[s,r] - deformation_rate[r,s]) * slip_rate_softest) / 2
    end

    # orientation_change[p,q] = Σ_rs ε[q,r,s] * orientation[p,s] * spin[r]
    return SMatrix{3,3,T,9}(ntuple(Val(9)) do idx
        q, p = fldmod1(idx, 3)  # column-major: (col, row)
        val = zero(T)
        @inbounds for s in 1:3, r in 1:3
            val += PERMUTATION_SYMBOL[q,r,s] * orientation[p,s] * spin[r]
        end
        val
    end)
end

"""
    _get_strain_energy(crss, slip_rates, slip_indices, slip_rate_softest,
                       stress_exponent, deformation_exponent, nucleation_efficiency)

Calculate strain energy due to dislocations for a grain. Allocation-free.
"""
@inline function _get_strain_energy(
    crss::SVector{4,T},
    slip_rates::SVector{4,T},
    slip_rate_softest::T,
    stress_exponent::T,
    deformation_exponent::T,
    nucleation_efficiency::T
)::T where T<:AbstractFloat
    strain_energy = zero(T)
    @inbounds for i in 1:3
        dislocation_density = (1 / crss[i])^(deformation_exponent - stress_exponent) *
            abs(slip_rates[i] * slip_rate_softest)^(stress_exponent / deformation_exponent)
        strain_energy += dislocation_density *
            exp(-nucleation_efficiency * dislocation_density^2)
    end
    return strain_energy
end

"""
    _get_rotation_and_strain(phase, fabric, orientation, strain_rate, velocity_gradient,
                             stress_exponent, deformation_exponent, nucleation_efficiency)

Get crystal axes rotation rate and strain energy for a single grain. Allocation-free.
Returns (orientation_change::SMatrix{3,3}, strain_energy::Float64).
"""
@inline function _get_rotation_and_strain(
    phase::MineralPhase,
    fabric::MineralFabric,
    orientation::SMatrix{3,3,T,9},
    strain_rate::SMatrix{3,3,T,9},
    velocity_gradient::SMatrix{3,3,T,9},
    stress_exponent::T,
    deformation_exponent::T,
    nucleation_efficiency::T
) where T<:AbstractFloat
    crss = get_crss(phase, fabric, T)
    slip_invariants = _get_slip_invariants(strain_rate, orientation)

    if phase == olivine
        # Guard against division by zero in _get_slip_rates_olivine.
        if all(==(zero(T)), slip_invariants)
            return SMatrix{3,3,T}(0,0,0, 0,0,0, 0,0,0), zero(T)
        end
        # argsort by |invariant / crss|
        ratios = SVector{4,T}(
            abs(slip_invariants[1] / crss[1]),
            abs(slip_invariants[2] / crss[2]),
            abs(slip_invariants[3] / crss[3]),
            abs(slip_invariants[4] / crss[4])
        )
        slip_indices = _argsort4(ratios)
        slip_rates = _get_slip_rates_olivine(slip_invariants, slip_indices, crss, deformation_exponent)
    elseif phase == enstatite
        slip_indices = _argsort4(SVector{4,T}(1/crss[1], 1/crss[2], 1/crss[3], 1/crss[4]))
        slip_rates = SVector{4,T}(zero(T), zero(T), zero(T),
            abs(slip_invariants[4]) > eps(T) ? one(T) : zero(T))
    else
        # Unreachable for valid MineralPhase values; returns zeros to stay GPU-safe.
        return SMatrix{3,3,T,9}(0,0,0, 0,0,0, 0,0,0), zero(T)
    end

    deformation_rate = _get_deformation_rate(phase, orientation, slip_rates)
    slip_rate_softest = _get_slip_rate_softest(deformation_rate, velocity_gradient)
    orientation_change = _get_orientation_change(orientation, velocity_gradient,
                                                  deformation_rate, slip_rate_softest)
    strain_energy = _get_strain_energy(crss, slip_rates, slip_rate_softest,
                                        stress_exponent, deformation_exponent,
                                        nucleation_efficiency)
    return orientation_change, strain_energy
end

"""Allocation-free argsort for 4-element SVector. Returns 1-based Int32 indices sorted ascending."""
@inline function _argsort4(v::SVector{4,T})::SVector{4,Int32} where T<:AbstractFloat
    # Simple insertion sort for 4 elements (Int32 for GPU compatibility)
    idx = MVector{4,Int32}(Int32(1), Int32(2), Int32(3), Int32(4))
    @inbounds for i in 2:4
        key = v[idx[i]]
        ki  = idx[i]
        j   = i - 1
        while j >= 1 && v[idx[j]] > key
            idx[j+1] = idx[j]
            j -= 1
        end
        idx[j+1] = ki
    end
    return SVector(idx)
end

# ──────────────────────────────────────────────────────────────────────────────
# GPU kernel: per-grain orientation change + strain energy (Pass 1)
# ──────────────────────────────────────────────────────────────────────────────

"""
Per-grain KernelAbstractions kernel.  Each work item processes one grain independently.
Writes orientation change into `orientations_diff` and scalar strain energy into
`strain_energies`.  The fraction-update (Pass 2) requires a global reduction and is
done on CPU after synchronisation.
"""
@kernel function _grain_kernel!(
    orientations_diff,          # AbstractArray{T,3}  — output
    strain_energies,            # AbstractVector{T}   — output
    @Const(orientations),       # AbstractArray{T,3}  — read-only input
    sr_vec,                     # SVector{9,T}  — flattened strain-rate matrix
    vg_vec,                     # SVector{9,T}  — flattened velocity-gradient matrix
    phase_int::Int32,
    fabric_int::Int32,
    smoothing,
    stress_exponent,
    deformation_exponent,
    nucleation_efficiency,
)
    T   = eltype(orientations_diff)
    g   = @index(Global, Linear)

    sr  = SMatrix{3,3,T,9}(sr_vec[1], sr_vec[2], sr_vec[3],
                            sr_vec[4], sr_vec[5], sr_vec[6],
                            sr_vec[7], sr_vec[8], sr_vec[9])
    vg  = SMatrix{3,3,T,9}(vg_vec[1], vg_vec[2], vg_vec[3],
                            vg_vec[4], vg_vec[5], vg_vec[6],
                            vg_vec[7], vg_vec[8], vg_vec[9])

    phase  = MineralPhase(phase_int)
    fabric = MineralFabric(fabric_int)

    @inbounds ori = SMatrix{3,3,T,9}(
        orientations[g,1,1], orientations[g,2,1], orientations[g,3,1],
        orientations[g,1,2], orientations[g,2,2], orientations[g,3,2],
        orientations[g,1,3], orientations[g,2,3], orientations[g,3,3],
    )

    _n = T(stress_exponent)
    _p = T(deformation_exponent)
    _ν = T(nucleation_efficiency)
    _s = T(smoothing)

    orientation_change, s_energy = _get_rotation_and_strain(
        phase, fabric, ori, sr, vg, _n, _p, _ν,
    )

    oc = orientation_change * _s
    @inbounds for i in 1:3, j in 1:3
        orientations_diff[g,i,j] = oc[i,j]
    end
    @inbounds strain_energies[g] = s_energy
end

# ──────────────────────────────────────────────────────────────────────────────
# Main derivatives function
# ──────────────────────────────────────────────────────────────────────────────

"""
    derivatives!(orientations_diff, fractions_diff,
                 regime, phase, fabric, n_grains,
                 orientations, fractions,
                 strain_rate, velocity_gradient, deformation_gradient_spin,
                 stress_exponent, deformation_exponent,
                 nucleation_efficiency, gbm_mobility, volume_fraction)

Compute derivatives of orientation and volume distribution (in-place, minimally allocating).

- `orientations_diff` — pre-allocated n_grains×3×3 output array for rotation rates
- `fractions_diff` — pre-allocated n_grains output vector for volume fraction changes
- `orientations` — n_grains×3×3 input orientation matrices
- `fractions` — n_grains input volume fractions

The inner grain-level computations are fully allocation-free using StaticArrays.
"""
function derivatives!(
    orientations_diff::AbstractArray{T,3},
    fractions_diff::AbstractVector{T},
    regime::DeformationRegime,
    phase::MineralPhase,
    fabric::MineralFabric,
    n_grains::Int,
    orientations::AbstractArray{T,3},
    fractions::AbstractVector{T},
    strain_rate::AbstractMatrix{T},
    velocity_gradient::AbstractMatrix{T},
    deformation_gradient_spin::AbstractMatrix{T},
    stress_exponent::Real,
    deformation_exponent::Real,
    nucleation_efficiency::Real,
    gbm_mobility::Real,
    volume_fraction::Real;
    backend::KernelAbstractions.Backend = CPU(),
) where T<:AbstractFloat
    # Convert scalar parameters to T once
    _n  = T(stress_exponent)
    _p  = T(deformation_exponent)
    _ν  = T(nucleation_efficiency)
    _M  = T(gbm_mobility)
    _vf = T(volume_fraction)

    if regime == min_viscosity || regime == max_viscosity
        # Zero derivatives: identity rotation, no volume change.
        @inbounds for g in 1:n_grains
            for i in 1:3, j in 1:3
                orientations_diff[g,i,j] = (i == j) ? one(T) : zero(T)
            end
            fractions_diff[g] = zero(T)
        end
        return nothing
    elseif regime == matrix_diffusion
        # Passive rotation based on spin of deformation gradient.
        @inbounds for g in 1:n_grains
            for i in 1:3, j in 1:3
                orientations_diff[g,i,j] = deformation_gradient_spin[i,j]
            end
            fractions_diff[g] = zero(T)
        end
        return nothing
    elseif regime == boundary_diffusion || regime == sliding_diffusion ||
           regime == sliding_dislocation
        throw(ArgumentError("this deformation mechanism is not yet supported"))
    end

    # matrix_dislocation or frictional_yielding
    smoothing = regime == frictional_yielding ? T(0.3) : one(T)

    # Flatten 3×3 matrices to SVector{9} for kernel argument passing (value types, no
    # device transfer needed).
    sr_vec = SVector{9,T}(
        strain_rate[1,1], strain_rate[2,1], strain_rate[3,1],
        strain_rate[1,2], strain_rate[2,2], strain_rate[3,2],
        strain_rate[1,3], strain_rate[2,3], strain_rate[3,3],
    )
    vg_vec = SVector{9,T}(
        velocity_gradient[1,1], velocity_gradient[2,1], velocity_gradient[3,1],
        velocity_gradient[1,2], velocity_gradient[2,2], velocity_gradient[3,2],
        velocity_gradient[1,3], velocity_gradient[2,3], velocity_gradient[3,3],
    )

    # ── Pass 1: per-grain orientation change + strain energy (GPU kernel) ─────
    # Allocate strain_energies on the backend device (no-op copy on CPU backend).
    strain_energies = KernelAbstractions.allocate(backend, T, (n_grains,))

    grain_kern = _grain_kernel!(backend, 64)
    grain_kern(
        orientations_diff, strain_energies, orientations,
        sr_vec, vg_vec,
        Int32(Int(phase)), Int32(Int(fabric)),
        smoothing, _n, _p, _ν;
        ndrange = n_grains,
    )
    KernelAbstractions.synchronize(backend)

    # ── Pass 2: mean energy (global reduction) + fraction derivatives (CPU) ──
    # fractions / fractions_diff are always CPU-side (ODE state vector).
    strain_energies_cpu = Array(strain_energies)   # no-op on CPU backend

    mean_energy = zero(T)
    @inbounds for g in 1:n_grains
        mean_energy += fractions[g] * strain_energies_cpu[g]
    end

    @inbounds for g in 1:n_grains
        residual = smoothing * (mean_energy - strain_energies_cpu[g])
        fractions_diff[g] = _vf * _M * fractions[g] * residual
    end
    return nothing
end

"""
    derivatives(regime, phase, fabric, n_grains, orientations, fractions,
                strain_rate, velocity_gradient, deformation_gradient_spin,
                stress_exponent, deformation_exponent,
                nucleation_efficiency, gbm_mobility, volume_fraction)

Allocating wrapper that returns (orientations_diff, fractions_diff).
Compatible with the Python API.
"""
function derivatives(
    regime::DeformationRegime,
    phase::MineralPhase,
    fabric::MineralFabric,
    n_grains::Int,
    orientations::AbstractArray{T,3},
    fractions::AbstractVector{T},
    strain_rate::AbstractMatrix{T},
    velocity_gradient::AbstractMatrix{T},
    deformation_gradient_spin::AbstractMatrix{T},
    stress_exponent::Real,
    deformation_exponent::Real,
    nucleation_efficiency::Real,
    gbm_mobility::Real,
    volume_fraction::Real,
) where T<:AbstractFloat
    orientations_diff = Array{T,3}(undef, n_grains, 3, 3)
    fractions_diff = Vector{T}(undef, n_grains)
    derivatives!(orientations_diff, fractions_diff,
                 regime, phase, fabric, n_grains,
                 orientations, fractions,
                 strain_rate, velocity_gradient, deformation_gradient_spin,
                 stress_exponent, deformation_exponent,
                 nucleation_efficiency, gbm_mobility, volume_fraction)
    return orientations_diff, fractions_diff
end
