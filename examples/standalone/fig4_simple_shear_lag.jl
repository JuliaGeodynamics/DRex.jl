# Reproduce Figure 4 of Bilton et al. (2025):
#
# Theoretical misfit θ − Φ between single crystal olivine a-axis angle θ
# and the FSE long axis orientation Φ in simple shear, as a function of
# strain ε = D₀t.  Dislocation glide is active on the (010)[100] system.
#
# Analytical solutions (from the paper):
#   θ(ε) = arctan(2ε + 1)               — eq. (24)
#   Φ(ε) = arctan(√(ε² + 1) + ε)       — eq. (29)
#
# At ε ≈ 0.5, the difference reaches a maximum of θ − Φ ≈ 5°.
# Both angles are measured from the X-axis and converge to π/2 as ε → ∞.
#
# Also overlays numerical DRex single-crystal result (1 grain, M* = 0).

push!(LOAD_PATH, "@v#.#")  # stacked env for CairoMakie

using DRex
using DRex: simple_shear_2d, update_orientations!, finite_strain, smallest_angle
using LinearAlgebra
using CairoMakie

# ── Analytical solutions (eqs. 24, 29) ──────────────────────────────────────
θ_theory(ε) = atand(2ε + 1)
Φ_theory(ε) = atand(√(ε^2 + 1) + ε)

# ── Numerical single-crystal simulation ──────────────────────────────────────
function run_single_crystal(; max_strain=10.0, n_steps=500)
    strain_rate = 1.0
    _, get_L = simple_shear_2d("Y", "X", strain_rate)

    # Initial [100] at 45° from X-axis in X-Y plane.
    s = √2 / 2
    R0 = [s s 0; -s s 0; 0 0 1]
    orientations_init = reshape(R0, 1, 3, 3)

    mineral = Mineral(
        phase=olivine, fabric=olivine_A, regime=matrix_dislocation,
        n_grains=1, fractions_init=[1.0], orientations_init=orientations_init,
    )

    params = default_params()
    params[:number_of_grains] = 1
    params[:gbm_mobility] = 0.0
    params[:gbs_threshold] = 0.0

    timestamps = range(0, max_strain / strain_rate, length=n_steps + 1)
    F = Matrix{Float64}(I, 3, 3)

    x_hat = [1.0, 0.0, 0.0]
    strains = Float64[0.0]
    θ_num = Float64[45.0]
    Φ_num = Float64[45.0]

    for t in 2:length(timestamps)
        F = update_orientations!(mineral, params, F, get_L,
                (timestamps[t-1], timestamps[t], _ -> zeros(3)))

        push!(strains, strain_rate * timestamps[t])

        # [100] angle from X-axis
        a_axis = mineral.orientations[end][1, 1, :]
        push!(θ_num, smallest_angle(a_axis, x_hat))

        # FSE long axis angle from X-axis
        _, fse_v = finite_strain(F)
        push!(Φ_num, smallest_angle(fse_v, x_hat))
    end

    return strains, θ_num, Φ_num
end

# ── Compute ──────────────────────────────────────────────────────────────────
# Analytical (fine grid)
ε_fine = range(0, 10, length=1000)
lag_theory = [θ_theory(ε) - Φ_theory(ε) for ε in ε_fine]

# Numerical
println("Running single-crystal DRex simulation (1 grain, 500 steps, ε_max=10)...")
t0 = time()
strains, θ_num, Φ_num = run_single_crystal()
lag_num = θ_num .- Φ_num
elapsed = time() - t0
println("Done in $(round(elapsed, digits=2))s")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig = Figure(size=(700, 450))
ax = Axis(fig[1, 1],
    xlabel="Strain (ε)",
    ylabel="θ − Φ (°)",
    title="Fig. 4 (Bilton et al. 2025): Single-crystal a-axis / FSE misfit in simple shear",
)
lines!(ax, collect(ε_fine), lag_theory, linewidth=2.5, color=:black,
    label="Analytical (eqs. 24, 29)")
scatter!(ax, strains[1:25:end], lag_num[1:25:end], markersize=8,
    color=:steelblue, marker=:circle, label="DRex (1 grain, M*=0)")

xlims!(ax, 0, 10)
ylims!(ax, -0.5, 6)
axislegend(ax, position=:rt)

outpath = joinpath(@__DIR__, "fig4_lag_angle.png")
save(outpath, fig, px_per_unit=2)
println("Saved: $outpath")
