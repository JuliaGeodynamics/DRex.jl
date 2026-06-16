# ODFTEX — continuum texture-evolution solver (Ribe, Faccenda & VanderBeek 2026, GJI).
#
# An alternative to the discrete D-Rex model. Instead of tracking many discrete
# grain orientations, ODFTEX evolves a continuous orientation distribution
# function (ODF) f(φ,θ,ψ) sampled at the centers of an nbox³ grid in Euler space.
# Each step advects the grid points along characteristics (extrinsic + intrinsic
# spin) using second-order midpoint time stepping, multiplies f by an exponential
# recrystallization/compressibility factor, and renormalises ∫f dg = 1.
#
# This is a Julia port of Neil Ribe's reference Fortran (odftex_2orderstep.f90),
# generalised to time-dependent flow: the Fortran nondimensionalises time by the
# strain-rate scale √(EᵢⱼEᵢⱼ/2), which is only valid for a constant velocity
# gradient. Here the strain-rate tensor, extrinsic spin and the recrystallisation
# constant c1 are kept dimensional and supplied per step, so the same stepper can
# be driven along a pathline through a time-dependent flow.
#
# State layout: angle/ODF arrays are nbox×nbox×nbox with index (i,j,k) mapping to
#   φ  = (i-1/2)·dφ            φ  ∈ [0,π]
#   θ  = acos(-1 + (j-1/2)·dcosθ)   cosθ ∈ [-1,1]
#   ψ  = (k-1/2)·dψ            ψ  ∈ [0,π]
# matching the Fortran grid (centers of boxes, avoiding pole singularities).

"""
    ODFState{T}

Continuous orientation distribution function on a fixed nbox³ Euler-angle grid.

# Fields
- `nbox`     — number of intervals per Euler angle (total boxes = nbox³)
- `pha,tha,psa` — current Eulerian angle grids (advected along characteristics)
- `f`        — ODF values at the (advected) grid points
- `f_nodrx`  — ODF without recrystallisation; tracks the elemental Euler-space
               volume so the integral can be normalised on a distorted grid
- `λ`        — dimensionless recrystallisation rate (paper recommends 2.6)
- `frac_opx` — volume fraction of orthopyroxene
- `s_ol`     — active olivine slip system index (1 ⇒ (010)[100])
- `ilimit`   — 1 to passively limit texture (FSE limiter; see notes)
"""
mutable struct ODFState{T<:AbstractFloat}
    nbox::Int
    pha::Array{T,3}
    tha::Array{T,3}
    psa::Array{T,3}
    f::Array{T,3}
    f_nodrx::Array{T,3}
    λ::T
    frac_opx::T
    s_ol::Int
    ilimit::Int
end

"""
    init_odf(nbox; λ=2.6, frac_opx=0.0, s_ol=1, ilimit=0, float_type=Float64)

Create an `ODFState` with a uniform (isotropic) ODF on a fresh nbox³ Euler grid.
Grid points sit at box centers to avoid the θ=0,π singularities.
"""
function init_odf(nbox::Int;
                  λ::Real = 2.6,
                  frac_opx::Real = 0.0,
                  s_ol::Int = 1,
                  ilimit::Int = 0,
                  float_type::Type{T} = Float64) where {T<:AbstractFloat}
    pha = Array{T,3}(undef, nbox, nbox, nbox)
    tha = Array{T,3}(undef, nbox, nbox, nbox)
    psa = Array{T,3}(undef, nbox, nbox, nbox)
    f       = ones(T, nbox, nbox, nbox)
    f_nodrx = ones(T, nbox, nbox, nbox)

    dph    = T(π) / nbox
    dcosth = T(2) / nbox
    dps    = T(π) / nbox

    @inbounds for k in 1:nbox, j in 1:nbox, i in 1:nbox
        pha[i,j,k] = (i - T(0.5)) * dph
        costh      = -one(T) + (j - T(0.5)) * dcosth
        tha[i,j,k] = acos(costh)
        psa[i,j,k] = (k - T(0.5)) * dps
    end

    return ODFState{T}(nbox, pha, tha, psa, f, f_nodrx,
                       T(λ), T(frac_opx), s_ol, ilimit)
end

# ──────────────────────────────────────────────────────────────────────────────
# Euler ↔ direction-cosine conversions (port of EULER_TO_DIRCOS / DIRCOS_TO_EULER)
# ──────────────────────────────────────────────────────────────────────────────

"""
    euler_to_dircos(φ, θ, ψ) -> SMatrix{3,3}

Direction-cosine matrix for the given Eulerian angles (Fortran EULER_TO_DIRCOS).
"""
@inline function euler_to_dircos(φ::T, θ::T, ψ::T) where {T<:AbstractFloat}
    cph, sph = cos(φ), sin(φ)
    cth, sth = cos(θ), sin(θ)
    cps, sps = cos(ψ), sin(ψ)
    return SMatrix{3,3,T,9}(
        # column 1 (o[1,1], o[2,1], o[3,1])
         cps*cph - cth*sph*sps,
        -sps*cph - cth*sph*cps,
         sph*sth,
        # column 2
         cps*sph + cth*cph*sps,
        -sps*sph + cth*cph*cps,
        -cph*sth,
        # column 3
         sps*sth,
         cps*sth,
         cth,
    )
end

"""
    dircos_to_euler(o) -> (φ, θ, ψ)

Eulerian angles from a direction-cosine matrix (Fortran DIRCOS_TO_EULER).
"""
@inline function dircos_to_euler(o::SMatrix{3,3,T,9}) where {T<:AbstractFloat}
    θ = acos(clamp(o[3,3], -one(T), one(T)))
    φ = atan(o[3,1], -o[3,2])
    ψ = atan(o[1,3], o[2,3])
    return φ, θ, ψ
end

# ──────────────────────────────────────────────────────────────────────────────
# Intrinsic spin ψ̇ and its ψ-derivative (port of PSDOT_CALC_GENERAL)
# ──────────────────────────────────────────────────────────────────────────────

"""
    psdot_calc_general(φ, θ, ψ, sr) -> (ψ̇, divψ̇)

Intrinsic crystallographic spin `ψ̇ = Sᵢⱼ Eᵢⱼ` for the s=1 slip system and its
derivative `divψ̇ = ∂ψ̇/∂ψ` (the compressibility of the characteristic flow used
to evolve f). `sr` is the (dimensional) strain-rate tensor.
"""
@inline function psdot_calc_general(φ::T, θ::T, ψ::T,
                                    sr::AbstractMatrix) where {T<:AbstractFloat}
    a = euler_to_dircos(φ, θ, ψ)

    cph, sph = cos(φ), sin(φ)
    cth, sth = cos(θ), sin(θ)
    cps, sps = cos(ψ), sin(ψ)

    # ∂a(i,j)/∂ψ  (row 3 is independent of ψ ⇒ zero)
    daps11 = -(cps*cth*sph) - cph*sps
    daps12 =  cph*cps*cth - sph*sps
    daps13 =  cps*sth
    daps21 = -(cph*cps) + cth*sph*sps
    daps22 = -(cps*sph) - cph*cth*sps
    daps23 = -(sps*sth)

    # Note: only these six components enter ψ̇ (matching the Fortran reference,
    # which omits sr[1,3] and uses sr[3,1] for the symmetric off-diagonal).
    s11, s12 = sr[1,1], sr[1,2]
    s22, s23 = sr[2,2], sr[2,3]
    s31, s33 = sr[3,1], sr[3,3]

    ψdot = a[1,2]*(a[2,1]*s12 + a[2,2]*s22 + a[2,3]*s23) +
           a[1,1]*(a[2,1]*s11 + a[2,2]*s12 + a[2,3]*s31) +
           a[1,3]*(a[2,2]*s23 + a[2,1]*s31 + a[2,3]*s33)

    divψdot = daps12*(a[2,1]*s12 + a[2,2]*s22 + a[2,3]*s23) +
              a[1,2]*(daps21*s12 + daps22*s22 + daps23*s23) +
              daps11*(a[2,1]*s11 + a[2,2]*s12 + a[2,3]*s31) +
              a[1,1]*(daps21*s11 + daps22*s12 + daps23*s31) +
              daps13*(a[2,2]*s23 + a[2,1]*s31 + a[2,3]*s33) +
              a[1,3]*(daps22*s23 + daps21*s31 + daps23*s33)

    return ψdot, divψdot
end

# ──────────────────────────────────────────────────────────────────────────────
# Strain-rate scale, extrinsic spin and recrystallisation constant c1
# ──────────────────────────────────────────────────────────────────────────────

"""
    strain_rate_scale(sr) -> √(Σ sr²/2)

The characteristic strain-rate scale ė₀ = √(EᵢⱼEᵢⱼ/2).
"""
@inline function strain_rate_scale(sr::AbstractMatrix{T}) where {T<:AbstractFloat}
    acc = zero(T)
    @inbounds for j in 1:3, i in 1:3
        acc += sr[i,j]^2
    end
    return sqrt(acc / 2)
end

"""
    odftex_kinematics(velocity_gradient) -> (sr, omdot, c1)

From a 3×3 velocity gradient compute the (dimensional) strain-rate tensor `sr`,
the extrinsic spin `omdot` (one-half the vorticity) and the recrystallisation
constant `c1` (paper eq. 17). All quantities are dimensional; for a time-
dependent flow they are recomputed each step from the local VGT.
"""
function odftex_kinematics(vg::AbstractMatrix{T}) where {T<:AbstractFloat}
    sr = SMatrix{3,3,T,9}(ntuple(Val(9)) do idx
        j, i = fldmod1(idx, 3)
        (vg[j,i] + vg[i,j]) / 2
    end)
    omdot = SVector{3,T}(
        (vg[3,2] - vg[2,3]) / 2,
        (vg[1,3] - vg[3,1]) / 2,
        (vg[2,1] - vg[1,2]) / 2,
    )
    c1 = -2 * (sr[1,1]^2 + sr[1,2]^2 + sr[1,1]*sr[2,2] +
               sr[2,2]^2 + sr[2,3]^2 + sr[3,1]^2) / 5
    return sr, omdot, c1
end

# ──────────────────────────────────────────────────────────────────────────────
# Single characteristic step (port of subroutine STEP)
# ──────────────────────────────────────────────────────────────────────────────

# isteptype = 1: half-step. Advance angles from (ph1,th1,ps1) by dt using the RHS
#                evaluated at the midpoint angles (phm,thm,psm); write into
#                (ph2,th2,ps2). Does NOT touch f.
# isteptype = 2: full step. Same advance but additionally evolve f and f_nodrx.
#
# sr / omdot / c1 are dimensional and correspond to the time at which the RHS is
# evaluated (τ for the half-step, τ+dτ/2 for the full step).
function _odftex_substep!(isteptype::Int,
                          ph1, th1, ps1,          # initial angles
                          phm, thm, psm,          # angles where RHS is evaluated
                          ph2, th2, ps2,          # output angles
                          f, f_nodrx, nbox,
                          omdot, sr, λ::T, c1::T, ilimit::Int, dt::T) where {T}
    @inbounds for k in 1:nbox, j in 1:nbox, i in 1:nbox
        cph = cos(phm[i,j,k])
        sph = sin(phm[i,j,k])
        tanth = tan(thm[i,j,k])
        sinth = sin(thm[i,j,k])

        phdot_ext = omdot[3] + (omdot[2]*cph - omdot[1]*sph) / tanth
        thdot_ext = omdot[1]*cph + omdot[2]*sph
        psdot_ext = (omdot[1]*sph - omdot[2]*cph) / sinth

        if ilimit != 1
            ψdot_int, divψdot = psdot_calc_general(phm[i,j,k], thm[i,j,k], psm[i,j,k], sr)
            drx = λ * (c1 + 2*ψdot_int^2)
            phdot = phdot_ext
            thdot = thdot_ext
            psdot = psdot_ext + ψdot_int
        else
            drx     = zero(T)
            divψdot = zero(T)
            phdot   = phdot_ext
            thdot   = thdot_ext
            psdot   = psdot_ext
        end

        ph2[i,j,k] = ph1[i,j,k] + dt*phdot
        th2[i,j,k] = th1[i,j,k] + dt*thdot
        ps2[i,j,k] = ps1[i,j,k] + dt*psdot

        if isteptype == 2
            f[i,j,k]       *= exp(dt * (drx - divψdot))
            f_nodrx[i,j,k] *= exp(-dt * divψdot)
        end
    end
    return nothing
end

"""
    odftex_step!(state, sr0, omdot0, c10, sr1, omdot1, c11, dt)

Advance the ODF by one second-order midpoint step of size `dt`.

The `*0` kinematics (strain rate, extrinsic spin, recrystallisation constant) are
evaluated at the start time τ and used for the half-step to the midpoint; the `*1`
kinematics are evaluated at the midpoint time τ+dt/2 and used for the full step.
For a steady flow pass the same kinematics for both. After the step the ODF is
renormalised to unit integral.

Mutates `state` in place. Uses three scratch midpoint-angle arrays.
"""
function odftex_step!(state::ODFState{T},
                      sr0, omdot0, c10::T,
                      sr1, omdot1, c11::T,
                      dt::T) where {T<:AbstractFloat}
    nbox = state.nbox
    dt2  = dt / 2

    # scratch midpoint angle buffers
    phm = similar(state.pha)
    thm = similar(state.tha)
    psm = similar(state.psa)

    # Half step to the midpoint: RHS evaluated at the start angles (pha,tha,psa).
    _odftex_substep!(1,
                     state.pha, state.tha, state.psa,
                     state.pha, state.tha, state.psa,
                     phm, thm, psm,
                     state.f, state.f_nodrx, nbox,
                     omdot0, sr0, state.λ, c10, state.ilimit, dt2)

    # Full step: RHS evaluated at midpoint angles (phm,thm,psm); also evolves f.
    _odftex_substep!(2,
                     state.pha, state.tha, state.psa,
                     phm, thm, psm,
                     state.pha, state.tha, state.psa,
                     state.f, state.f_nodrx, nbox,
                     omdot1, sr1, state.λ, c11, state.ilimit, dt)

    # Normalise ∫f dg = 1.
    integral = integral_drx(state)
    if integral > 0
        @inbounds state.f ./= integral
    end
    return state
end

"""
    odftex_step!(state, velocity_gradient, dt)

Convenience steady-flow step: derive the kinematics from a single velocity
gradient and use them for both half- and full-step.
"""
function odftex_step!(state::ODFState{T}, vg::AbstractMatrix, dt::Real) where {T}
    sr, omdot, c1 = odftex_kinematics(T.(vg))
    return odftex_step!(state, sr, omdot, T(c1), sr, omdot, T(c1), T(dt))
end

# ──────────────────────────────────────────────────────────────────────────────
# ODF integral / normalisation (port of INTEGRAL_DRX)
# ──────────────────────────────────────────────────────────────────────────────

"""
    integral_drx(state) -> (1/2π²) ∫ f dg

Approximate Euler-space integral of the ODF. `f_nodrx` estimates the local
elemental volume on the (distorted) advected grid; the ratio f/f_nodrx weights
each box. Equals 1 exactly when recrystallisation is off.
"""
function integral_drx(state::ODFState{T}) where {T<:AbstractFloat}
    nbox = state.nbox
    dcosth = T(2) / nbox
    dph    = T(π) / nbox
    dps    = T(π) / nbox
    dg     = dcosth * dph * dps

    acc = zero(T)
    @inbounds for k in 1:nbox, j in 1:nbox, i in 1:nbox
        acc += dg * state.f[i,j,k] / state.f_nodrx[i,j,k]
    end
    return acc / (2 * T(π)^2)
end

# ──────────────────────────────────────────────────────────────────────────────
# Eulerian transform between slip-system reference frames (port of EULERIAN_TRANSFORM)
# ──────────────────────────────────────────────────────────────────────────────

"""
    bmat_for_slip(s, T) -> SMatrix{3,3}

Rotation matrix M(s) relating slip system `s` to the s=1 reference frame
(paper eq. 2). s=1 is the identity; s=4 maps olivine → orthopyroxene.
"""
@inline function bmat_for_slip(s::Int, ::Type{T}) where {T}
    if s == 2
        return SMatrix{3,3,T,9}(one(T),zero(T),zero(T),  zero(T),zero(T),-one(T),  zero(T),one(T),zero(T))
    elseif s == 3
        return SMatrix{3,3,T,9}(zero(T),zero(T),one(T),  zero(T),-one(T),zero(T),  one(T),zero(T),zero(T))
    elseif s == 4
        return SMatrix{3,3,T,9}(zero(T),zero(T),one(T),  one(T),zero(T),zero(T),  zero(T),one(T),zero(T))
    else
        return SMatrix{3,3,T,9}(one(T),zero(T),zero(T),  zero(T),one(T),zero(T),  zero(T),zero(T),one(T))
    end
end

"""
    eulerian_transform(state, s) -> (f2, ph2, th2, ps2)

Transform the master ODF (in s=1 reference angles) to the angle set for slip
system `s` (paper Section 2). ODF *values* are unchanged; only the angles they
are associated with change. For s=1 returns copies of the master grid.
"""
function eulerian_transform(state::ODFState{T}, s::Int) where {T<:AbstractFloat}
    nbox = state.nbox
    ph2 = similar(state.pha); th2 = similar(state.tha); ps2 = similar(state.psa)
    f2  = copy(state.f)

    if s == 1
        copyto!(ph2, state.pha); copyto!(th2, state.tha); copyto!(ps2, state.psa)
        return f2, ph2, th2, ps2
    end

    bmat = bmat_for_slip(s, T)
    @inbounds for k in 1:nbox, j in 1:nbox, i in 1:nbox
        amat  = euler_to_dircos(state.pha[i,j,k], state.tha[i,j,k], state.psa[i,j,k])
        bamat = bmat * amat
        φ, θ, ψ = dircos_to_euler(bamat)
        ph2[i,j,k] = φ; th2[i,j,k] = θ; ps2[i,j,k] = ψ
    end
    return f2, ph2, th2, ps2
end
