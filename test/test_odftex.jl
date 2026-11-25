using Test
using LinearAlgebra
using StaticArrays
using Random
using MantleFabric
using MantleFabric: ODFState, init_odf, odftex_step!, odftex_kinematics,
            euler_to_dircos, dircos_to_euler, psdot_calc_general,
            integral_drx, strain_rate_scale,
            odf_box_weights, sample_orientations, odf_mean_axis,
            odf_misorientation_index, misorientation_index, orthorhombic,
            odf_polefigure_density, fse_long_axis

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

# ──────────────────────────────────────────────────────────────────────────────
# Pole-figure density helper (used by the paper-figure examples)
# ──────────────────────────────────────────────────────────────────────────────

@testset "ODFTEX pole-figure density" begin
    # A uniform ODF must project to a flat field at ~1 MUD everywhere.
    st = init_odf(40; λ = 0.0)
    xs, ys, dens = odf_polefigure_density(st, 3; ngrid = 121)
    finite = filter(isfinite, dens)
    @test isapprox(sum(finite) / length(finite), 1.0; atol = 0.02)   # mean ≈ 1 MUD
    @test maximum(finite) < 1.4                                       # no spurious peaks
    # Out-of-disk corners are NaN.
    @test isnan(dens[1, 1])
    @test_throws ArgumentError odf_polefigure_density(st, 4)
end

# ──────────────────────────────────────────────────────────────────────────────
# Figure-reproduction physics (asserts what the example scripts plot)
# ──────────────────────────────────────────────────────────────────────────────

"""Scatter-matrix eigenvalues of the `row`-th crystallographic axis (a/b/c)."""
function _axis_scatter_eigs(state, row; rng = MersenneTwister(1), n = 40000)
    ori = sample_orientations(state, n; rng = rng)
    S = zeros(3, 3)
    for g in 1:size(ori, 1)
        u = ori[g, row, :]
        S .+= u * transpose(u)
    end
    return sort(eigvals(Symmetric(S ./ size(ori, 1))))
end

@testset "Fig. 3 physics — pure shear, [001] isotropy vs drx" begin
    # Pure shear in x–y plane (Γ=0, ė₀=1), τ=0.7.
    vg = [1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 0.0]
    runto(λ) = begin
        st = init_odf(60; λ = λ)
        for _ in 1:40
            odftex_step!(st, vg, 0.7 / 40)
        end
        st
    end

    # λ=0: [001] (b×n axis) is isotropic ⇒ scatter eigenvalues ≈ [1/3,1/3,1/3];
    # [100]/[010] are anisotropic. This is the paper's exact single-slip result.
    ev_c0 = _axis_scatter_eigs(runto(0.0), 3)
    @test all(isapprox.(ev_c0, 1/3; atol = 0.02))
    ev_a0 = _axis_scatter_eigs(runto(0.0), 1)
    @test (ev_a0[3] - ev_a0[1]) > 0.15            # clearly anisotropic

    # λ=2.6: recrystallisation makes [001] anisotropic.
    ev_c1 = _axis_scatter_eigs(runto(2.6), 3)
    @test (ev_c1[3] - ev_c1[1]) > 0.1
end

@testset "Fig. 4 physics — simple shear, [100] follows FSE" begin
    # Sinistral simple shear (Γ=1, ė₀=1), λ=0. L[1,2] = -2.
    vg = [0.0 -2.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    for τ in (0.5, 1.0)
        st = init_odf(60; λ = 0.0)
        for _ in 1:40
            odftex_step!(st, vg, τ / 40)
        end

        # FSE long axis from F = exp(L τ).
        fse = fse_long_axis(exp(vg * τ))
        fse_ang = atand(fse[2], fse[1])

        # [100] mean direction (sampled).
        â = odf_mean_axis(st, "a")
        a_ang = atand(â[2], â[1])

        # The [100] peak follows the FSE long axis (McKenzie finite-strain
        # hypothesis when recrystallisation is off): within a few degrees.
        @test abs(a_ang - fse_ang) < 6.0

        # FSE rotates toward the shear plane (decreasing |angle|) with strain;
        # both lie in the fourth quadrant (negative angle) for sinistral shear.
        @test fse_ang < 0
        @test a_ang < 0

        # λ=0 ⇒ [001] stays isotropic.
        ev_c = _axis_scatter_eigs(st, 3)
        @test all(isapprox.(ev_c, 1/3; atol = 0.03))
    end
end

@testset "Fig. 5 physics — MIT23 simple shear, λ=2.6, γ=140%" begin
    # Sinistral simple shear (Γ=1, ė₀=1), λ=2.6, τ=0.7 (γ=1.4). This is the
    # Fortran reference case validated element-wise elsewhere; here we assert the
    # geometry of the three reference directions and the M-index range.
    vg = [0.0 -2.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    τ  = 0.7

    # Shear plane normal is Y ⇒ the plane is the horizontal (X) axis: 0°.
    # Max instantaneous extension = leading strain-rate eigenvector at ±45°.
    sr, _, _ = odftex_kinematics(vg)
    _, vecs = eigen(Symmetric(Matrix(sr)))
    ext = vecs[:, end]
    ext_ang = mod(atand(ext[2], ext[1]), 180.0)        # axis ⇒ mod 180
    @test isapprox(ext_ang, 135.0; atol = 1e-6)         # i.e. −45° as an axis

    # FSE long axis lies strictly between the shear plane (0°) and the
    # max-extension direction (−45°), rotated toward the shear plane.
    fse = fse_long_axis(exp(vg * τ))
    fse_ang = atand(fse[2], fse[1])
    @test -45.0 < fse_ang < 0.0
    @test fse_ang ≈ -27.5 atol = 1.0

    # M-index in the recrystallised A-type range (paper reports ≈0.115 native /
    # our sampled estimator runs a touch higher; both clearly anisotropic).
    st = init_odf(60; λ = 2.6)
    for _ in 1:40
        odftex_step!(st, vg, τ / 40)
    end
    m = odf_misorientation_index(st; n = 40000, rng = MersenneTwister(1))
    @test 0.08 < m < 0.20

    # Recrystallisation makes [001] anisotropic (vs the isotropic λ=0 case).
    ev_c = _axis_scatter_eigs(st, 3)
    @test (ev_c[3] - ev_c[1]) > 0.05
end

@testset "Fig. 6 physics — [100] expectation orientation vs FSE" begin
    # Simple shear (Γ=1). Mean folded in-plane [100] angle relative to the shear
    # plane (odf_expectation_angle). For an isotropic ODF this is 45°.
    vg = [0.0 -2.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]

    fse_angle(γ) = (v = fse_long_axis(exp(vg * (γ / 2)));
                    a = abs(atand(v[2], v[1])); a > 90 ? 180 - a : a)

    function a_angle(λ, γ)
        τ = γ / 2
        st = init_odf(40; λ = λ)
        for _ in 1:40
            odftex_step!(st, vg, τ / 40)
        end
        return odf_expectation_angle(st, 1; plane_dir1 = 1, plane_dir2 = 2)
    end

    # Isotropic in-plane expectation angle is exactly 45°.
    @test isapprox(odf_expectation_angle(init_odf(30; λ = 0.0), 1), 45.0; atol = 1e-6)

    # Low strain: both recrystallising curves sit ABOVE the FSE (the [100]
    # distribution stays spread near the instantaneous extension direction while
    # the FSE has already begun rotating toward the shear plane).
    for γ in (0.1, 0.2, 0.3)
        f = fse_angle(γ)
        @test a_angle(2.6, γ) > f
        @test a_angle(10.0, γ) > f
    end

    # The λ=10 curve crosses below the FSE (rotates faster) by moderate strain…
    @test a_angle(10.0, 1.0) < fse_angle(1.0)
    # …and the ordering at large strain is FSE > λ=2.6 > λ=10.
    @test fse_angle(2.0) > a_angle(2.6, 2.0) > a_angle(10.0, 2.0)
    # λ=10 reaches close to the shear plane at large strain.
    @test a_angle(10.0, 2.0) < 8.0
end

@testset "Fig. 7 physics — uniaxial compression, λ=2.6, 58% shortening" begin
    # Axisymmetric compression along Z: L = diag(ε̇, ε̇, −2ε̇). Vorticity 0, and
    # the strain-rate scale is √3 ε̇. 58% shortening ⇒ ε̇ t = −ln(0.42)/2, so
    # τ = ė₀ t = √3 · ε̇ t ≈ 0.751.
    vg = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 -2.0]
    sr, om, _ = odftex_kinematics(vg)
    e0 = strain_rate_scale(sr)
    @test isapprox(e0, sqrt(3); atol = 1e-12)
    @test all(isapprox.(om, 0.0; atol = 1e-12))        # Γ = 0 (no vorticity)

    τ = e0 * (-log(1 - 0.58) / 2)
    st = init_odf(60; λ = 2.6)
    for _ in 1:40
        odftex_step!(st, vg ./ e0, τ / 40)
    end

    # [010] concentrates along the compression axis (Z): its dominant scatter
    # eigenvector is ±Z and it is the strongest single-axis concentration.
    function scatter(row)
        ori = sample_orientations(st, 40000; rng = MersenneTwister(1))
        S = zeros(3, 3)
        for g in 1:size(ori, 1)
            u = ori[g, row, :]
            S .+= u * transpose(u)
        end
        return Symmetric(S ./ size(ori, 1))
    end

    S010 = scatter(2)
    ev010 = sort(eigvals(S010))
    _, vv = eigen(S010)
    dom010 = abs.(vv[:, end])
    @test dom010[3] > 0.95                  # [010] dominant axis ≈ Z (compression)
    @test ev010[3] > 0.5                    # strong central concentration

    # [100] and [001] form girdles in the plane ⟂ compression: their dominant
    # axes lie in X–Y (small Z component), and the [001] girdle is at least as
    # concentrated as the [100] girdle (paper: [001] stronger).
    ev100 = sort(eigvals(scatter(1)))
    ev001 = sort(eigvals(scatter(3)))
    @test ev100[1] < 0.30 && ev001[1] < 0.30          # girdle ⇒ small minimum eig
    @test ev001[3] ≥ ev100[3] - 0.02                  # [001] ≳ [100]

    # M-index in the recrystallised range (paper native 12³ = 0.0984).
    m = odf_misorientation_index(st; n = 40000, rng = MersenneTwister(1))
    @test 0.06 < m < 0.16
end

# ──────────────────────────────────────────────────────────────────────────────
# Pathline driver for time-dependent flow (odftex_update!)
# ──────────────────────────────────────────────────────────────────────────────

@testset "ODFTEX pathline update (time-dependent flow)" begin
    nbox = 12
    L0 = [0.0 -2.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]     # ė₀ = 1
    getpos = t -> [0.0, 0.0, 0.0]

    # Reference: 40 manual steady steps to τ = 1.
    ref = init_odf(nbox)
    for _ in 1:40
        odftex_step!(ref, L0, 1.0 / 40)
    end

    # Steady flow through the driver picks the same 40 steps and F = I + L·t.
    st = init_odf(nbox)
    F = MantleFabric.odftex_update!(st, Matrix(1.0I, 3, 3), (t, x) -> L0, (0.0, 1.0, getpos))
    @test st.f ≈ ref.f
    @test F ≈ I + L0

    # Same strain accrued 1000× faster (dimensional rates): identical texture.
    st = init_odf(nbox)
    F = MantleFabric.odftex_update!(st, Matrix(1.0I, 3, 3), (t, x) -> 1000 .* L0, (0.0, 1e-3, getpos))
    @test st.f ≈ ref.f
    @test F ≈ I + L0

    # Time-dependent rate g(t) = 2t on [0,1] (∫g = 1): same strain path, so the
    # texture must converge to the steady result.
    st = init_odf(nbox)
    F = MantleFabric.odftex_update!(st, Matrix(1.0I, 3, 3), (t, x) -> (2t) .* L0, (0.0, 1.0, getpos);
                            max_strain_step = 0.005)
    @test F ≈ I + L0
    @test maximum(abs.(st.f .- ref.f)) / maximum(ref.f) < 2e-2

    # No deformation: nothing changes.
    st = init_odf(nbox)
    MantleFabric.odftex_update!(st, Matrix(1.0I, 3, 3), (t, x) -> zeros(3, 3), (0.0, 1.0, getpos))
    @test all(st.f .≈ 1)
end

@testset "ODFTEX texture limiter (rmax) and exact Mineral export" begin
    L0 = [0.0 -2.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    getpos = t -> zeros(3)
    I3 = Matrix(1.0I, 3, 3)

    # Limiter engages once ln(c1/c2) > rmax and the texture then stops sharpening.
    st = init_odf(12)
    F = MantleFabric.odftex_update!(st, I3, (t, x) -> L0, (0.0, 1.0, getpos); rmax = 0.5)
    @test st.ilimit == 1
    fmax = maximum(st.f)
    MantleFabric.odftex_update!(st, F, (t, x) -> L0, (1.0, 2.0, getpos); rmax = 0.5)
    @test maximum(st.f) ≈ fmax
    free = init_odf(12)
    MantleFabric.odftex_update!(free, I3, (t, x) -> L0, (0.0, 2.0, getpos))
    @test free.ilimit == 0 && maximum(free.f) > 2 * fmax

    # Isotropic ODF → nearly isotropic Voigt average; textured ODF → anisotropic.
    ms = MantleFabric.odf_to_minerals(init_odf(12))
    @test length(ms) == 2 && sum(ms[1].fractions[end]) ≈ 1
    C = voigt_averages(ms, [olivine, enstatite], [0.7, 0.3])
    @test abs(C[1,1] - C[3,3]) / C[1,1] < 0.01
    C = voigt_averages(MantleFabric.odf_to_minerals(free), [olivine, enstatite], [0.7, 0.3])
    @test abs(C[1,1] - C[3,3]) / C[1,1] > 0.03
end

@testset "Fig. 6 — mean [100] axis rotates to the shear plane faster than the FSE" begin
    # The dominant [100] axis (odf_mean_axis) reproduces the curves of the paper's
    # Fig. 6: ≈6° from the shear plane at γ = 2 for λ = 2.6 (FSE: 22.5°), and
    # within a few degrees of the shear plane by γ = 1.25 for λ = 10.
    vg = [0.0 -2.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    function mean_angle(λ, γ)
        st = init_odf(24; λ = λ)
        for _ in 1:40
            odftex_step!(st, vg, γ / 2 / 40)
        end
        v = odf_mean_axis(st, "a")
        a = abs(atand(v[2], v[1]))
        return a > 90 ? 180 - a : a
    end
    fse(γ) = atand(2 / γ) / 2
    @test 4 < mean_angle(2.6, 2.0) < 8
    @test 22 < mean_angle(2.6, 1.0) < 27
    @test mean_angle(10.0, 1.25) < 3
    @test mean_angle(2.6, 1.0) < fse(1.0) && mean_angle(10.0, 0.5) < fse(0.5)
end
