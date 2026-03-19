# Reproduce Figure 5 of Bilton et al. (2025):
#
# Numerical rotation rates for a single crystal of A-type olivine deforming
# via dislocation creep in simple shear with L₂₁ = 2D₀.
#
# The initial [100] angle θ is measured from the X-axis in the X–Y plane.
# At θ = 90° and 270°, the [100] vector is aligned to the shear axis (Y)
# and the crystal does not experience rotation.
#
# Theoretical relationship (eq. 23 / eq. S23):
#   ∂θ/∂t = D₀(1 + cos 2θ)
#
# The plot shows the theoretical curve (solid line) and DRex numerical
# rotation rates (open circles) sampled every 25th angle.

push!(LOAD_PATH, "@v#.#")  # stacked env for CairoMakie

using DRex
using LinearAlgebra
using CairoMakie

"""Rotation matrix from axis-angle (Rodrigues formula)."""
function rotation_from_rotvec(rotvec)
    θ = norm(rotvec)
    if θ < 1e-30
        return Matrix{Float64}(I, 3, 3)
    end
    k = rotvec ./ θ
    K = [0 -k[3] k[2]; k[3] 0 -k[1]; -k[2] k[1] 0]
    return I + sin(θ) * K + (1 - cos(θ)) * K^2
end

# ── Compute numerical rotation rates ────────────────────────────────────────
L  = Float64[0 0 0; 2 0 0; 0 0 0]   # L₂₁ = 2D₀
sr = Float64[0 1 0; 1 0 0; 0 0 0]   # strain rate = ½(L + Lᵀ)

angles_deg = Float64[]
numerical_rates = Float64[]
target_rates = Float64[]

println("Computing rotation rates for 3600 initial orientations...")
t0 = time()

for θ in range(0, 2π, length=3600)
    R = rotation_from_rotvec([0.0, 0.0, θ])
    orientations = reshape(R, 1, 3, 3)

    orientations_diff, _ = DRex.derivatives(
        matrix_dislocation, olivine, olivine_A, 1,
        orientations, [1.0], sr, L, fill(NaN, 3, 3),
        1.5, 3.5, 5.0, 0.0, 1.0,
    )

    rate = sqrt(orientations_diff[1, 1, 1]^2 + orientations_diff[1, 1, 2]^2)
    push!(angles_deg, rad2deg(θ))
    push!(numerical_rates, rate)
    push!(target_rates, 1 + cos(2θ))
end

elapsed = time() - t0
println("Done in $(round(elapsed, digits=2))s")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig = Figure(size=(700, 450))
ax = Axis(fig[1, 1],
    xlabel="Initial [100] angle from X-axis (°)",
    ylabel="Rotation rate (∂θ/∂t) / D₀",
    title="Fig. 5 (Bilton et al. 2025): Single-crystal olivine rotation rate in simple shear",
    xticks=0:45:360,
)

# Theoretical curve (solid line)
lines!(ax, angles_deg, target_rates, linewidth=2, color=:black,
    label="D₀(1 + cos 2θ)")

# Numerical (open circles, every 25th point)
idx = 1:25:length(angles_deg)
scatter!(ax, angles_deg[idx], numerical_rates[idx], markersize=8,
    color=(:steelblue, 0.0), strokecolor=:steelblue, strokewidth=1.5,
    marker=:circle, label="DRex numerical")

# Dashed lines at 90° and 270° (shear-aligned, zero rotation)
vlines!(ax, [90, 270], linestyle=:dash, color=:gray60, linewidth=1)

xlims!(ax, 0, 360)
ylims!(ax, -0.1, 2.1)
axislegend(ax, position=:rt)

outpath = joinpath(@__DIR__, "fig5_rotation_rates.png")
save(outpath, fig, px_per_unit=2)
println("Saved: $outpath")
