using Test
using LinearAlgebra
using StaticArrays
using Random
using DRex
using DRex: ODFState, init_odf, odftex_step!, odftex_kinematics,
            euler_to_dircos, dircos_to_euler, psdot_calc_general,
            integral_drx, strain_rate_scale,
            odf_box_weights, sample_orientations, odf_mean_axis,
            odf_misorientation_index, misorientation_index, orthorhombic

# ──────────────────────────────────────────────────────────────────────────────
# Euler ↔ direction-cosine conversions
# ──────────────────────────────────────────────────────────────────────────────

@testset "ODFTEX euler/dircos" begin
    # Direction-cosine matrices must be proper rotations.
    for (φ, θ, ψ) in ((0.3, 0.7, 1.1), (1.7, 2.2, 0.4), (2.9, 1.5, 2.5))
        o = euler_to_dircos(φ, θ, ψ)
        @test isapprox(o * o', I; atol = 1e-12)
        @test isapprox(det(o), 1.0; atol = 1e-12)
    end

    # Round-trip φ→o→φ for angles in the interior of the orthorhombic ranges
    # φ,ψ ∈ (0,π), θ ∈ (0,π) where the atan2 inversion is single-valued.
    for (φ, θ, ψ) in ((0.3, 0.7, 1.1), (1.7, 2.2, 0.4), (2.9, 1.5, 2.5))
        o = euler_to_dircos(φ, θ, ψ)
        φ2, θ2, ψ2 = dircos_to_euler(o)
        @test isapprox(θ2, θ; atol = 1e-10)
        # φ,ψ are recovered modulo the branch; compare the reconstructed matrix.
        @test isapprox(euler_to_dircos(φ2, θ2, ψ2), o; atol = 1e-10)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Intrinsic spin: divψ̇ must equal ∂ψ̇/∂ψ (finite-difference check)
# ──────────────────────────────────────────────────────────────────────────────

@testset "ODFTEX psdot derivative" begin
    sr = SMatrix{3,3,Float64,9}(0.0, -1.0, 0.0,
                                -1.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0)  # symmetric part of simple shear
    h = 1e-6
    for (φ, θ, ψ) in ((0.5, 0.9, 1.3), (2.1, 1.8, 0.6), (1.0, 1.2, 2.4))
        ψdot, divψdot = psdot_calc_general(φ, θ, ψ, sr)
        ψp, _ = psdot_calc_general(φ, θ, ψ + h, sr)
        ψm, _ = psdot_calc_general(φ, θ, ψ - h, sr)
        fd = (ψp - ψm) / (2h)
        @test isapprox(divψdot, fd; atol = 1e-6, rtol = 1e-5)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Conservation: with λ=0 the normalised integral stays ≈ 1 every step
# ──────────────────────────────────────────────────────────────────────────────

@testset "ODFTEX conservation (λ=0)" begin
    nbox = 30
    state = init_odf(nbox; λ = 0.0)
    @test isapprox(integral_drx(state), 1.0; atol = 1e-12)

    # dimensionless simple shear, same scaling as the Fortran reference
    vg = [0.0 -2.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    sr, _, _ = odftex_kinematics(vg)
    e0 = strain_rate_scale(sr)
    vg ./= e0
    dt = 1.0 / 40
    for _ in 1:40
        odftex_step!(state, vg, dt)
        # f is renormalised each step, so post-step integral is ~1.
        @test isapprox(integral_drx(state), 1.0; atol = 1e-10)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Validation against Neil Ribe's reference Fortran (Fig. 5b simple shear case)
# ──────────────────────────────────────────────────────────────────────────────

"""Read the (φ,θ,ψ,f) columns from a Fortran f_ol.out reference file (degrees)."""
function _read_fortran_odf(path)
    fvals = Float64[]
    for line in eachline(path)
        cols = split(strip(line))
        isempty(cols) && continue
        push!(fvals, parse(Float64, cols[4]))
    end
    return fvals
end

@testset "ODFTEX vs Fortran reference (simple shear, λ=2.6)" begin
    ref_path = joinpath(@__DIR__, "data", "odftex", "f_ol_n30.out")
    @test isfile(ref_path)
    f_ref = _read_fortran_odf(ref_path)

    nbox = 30
    @test length(f_ref) == nbox^3

    # Reproduce the exact reference run: vg from the .inp (row i, cols j),
    # nondimensionalised by ė₀, λ=2.6, taumax=1.0, 40 steps, s_ol=1, frac_opx=0.
    state = init_odf(nbox; λ = 2.6, frac_opx = 0.0, s_ol = 1, ilimit = 0)
    vg = [0.0 -2.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    sr, _, _ = odftex_kinematics(vg)
    vg ./= strain_rate_scale(sr)
    dt = 1.0 / 40
    for _ in 1:40
        odftex_step!(state, vg, dt)
    end

    # The Fortran writes with i outermost, k innermost (DO i; DO j; DO k),
    # so the n-th line corresponds to f(i,j,k) with k varying fastest. Build a
    # comparison vector from state.f in that same order.
    f_jl = Vector{Float64}(undef, nbox^3)
    n = 1
    for i in 1:nbox, j in 1:nbox, k in 1:nbox
        f_jl[n] = state.f[i,j,k]
        n += 1
    end
    @test length(f_jl) == length(f_ref)

    # Element-wise agreement. The reference is printed to 4 sig figs, so use a
    # modest tolerance; the dominant peaks must line up closely.
    maxabs = maximum(abs, f_ref)
    rel_err = maximum(abs.(f_jl .- f_ref)) / maxabs
    @test rel_err < 1e-2

    # Peak ODF value must match. The simple-shear texture has several exactly
    # degenerate peaks (a symmetry of the deformation), so argmax can land on a
    # different member of the degenerate set in each code; instead require that
    # the Julia peak location is itself a near-maximum of the reference field.
    @test isapprox(maximum(f_jl), maximum(f_ref); rtol = 2e-2)
    @test f_ref[argmax(f_jl)] ≥ 0.99 * maximum(f_ref)
    @test f_jl[argmax(f_ref)] ≥ 0.99 * maximum(f_jl)
end

# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics: box weights, sampler, native mean axis / M-index
# ──────────────────────────────────────────────────────────────────────────────

@testset "ODFTEX box weights" begin
    # Uniform ODF ⇒ all box weights equal and summing to 1.
    state = init_odf(20; λ = 0.0)
    w = odf_box_weights(state)
    @test isapprox(sum(w), 1.0; atol = 1e-12)
    @test isapprox(minimum(w), maximum(w); rtol = 1e-12)
end

@testset "ODFTEX sampler" begin
    state = init_odf(20; λ = 0.0)
    rng = MersenneTwister(123)
    ori = sample_orientations(state, 2000; rng = rng)
    @test size(ori) == (2000, 3, 3)
    # Every sampled matrix is a proper rotation.
    for g in (1, 500, 1234, 2000)
        a = ori[g, :, :]
        @test isapprox(a * a', I; atol = 1e-10)
        @test isapprox(det(a), 1.0; atol = 1e-10)
    end
    # A uniform ODF should give a near-isotropic discrete sample (small M-index).
    @test misorientation_index(ori, orthorhombic) < 0.1
end

@testset "ODFTEX native diagnostics on simple-shear texture" begin
    nbox = 30
    state = init_odf(nbox; λ = 2.6, frac_opx = 0.0, s_ol = 1, ilimit = 0)
    vg = [0.0 -2.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    sr, _, _ = odftex_kinematics(vg)
    vg ./= strain_rate_scale(sr)
    dt = 1.0 / 40
    for _ in 1:40
        odftex_step!(state, vg, dt)
    end

    # The deformed texture is anisotropic: M-index well above the isotropic case.
    rng = MersenneTwister(7)
    m_def = odf_misorientation_index(state; n = 20000, rng = rng)
    m_iso = odf_misorientation_index(init_odf(nbox; λ = 0.0); n = 20000, rng = MersenneTwister(7))
    @test m_def > m_iso
    @test m_def > 0.05

    # The a-axis mean direction is a unit vector lying in the shear (x–y) plane.
    â = odf_mean_axis(state, "a")
    @test isapprox(norm(â), 1.0; atol = 1e-10)
    @test abs(â[3]) < 0.2   # [100] concentrates near the shear plane, not out of it
end
