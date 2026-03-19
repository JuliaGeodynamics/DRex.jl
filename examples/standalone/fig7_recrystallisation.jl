# Reproduce Figure 7 of Bilton et al. (2025):
#
# Numerical rotation rates (a) and grain growth rates (b) for an A-type olivine
# polycrystal, with an ideal girdle texture in the X–Y plane (deformation plane),
# deforming via dislocation creep in simple shear with L₂₁ = 2D₀.
#
# The initial [100] angles are measured from the X-axis in the X–Y plane, so
# that at 90° and 270° (dashed lines), the [100] vector is aligned to the shear
# axis and the grain does not experience rotation.
#
# Theoretical relationships:
#   Rotation rate (identical to single crystal case, Fig. 5):
#     ∂θ/∂t = D₀(1 + cos 2θ)                                 [eq. 23]
#
#   Dislocation density on (010)[100]:
#     ρ* = |cos 2θ|^(p/n)                                     [eq. 30]
#
#   Strain energy per grain:
#     E = ρ* exp(-λ* (ρ*)²)                                   [eq. 31]
#
#   Grain growth rate (from GBM):
#     ∂f/∂ε = -M* f (E - Ē)                                   [eq. 18]

push!(LOAD_PATH, "@v#.#")  # stacked env for CairoMakie

using DRex
using LinearAlgebra
using CairoMakie
using Statistics: mean

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

# ── Parameters ───────────────────────────────────────────────────────────────
N = 360000           # number of surrogate grains (ideal girdle)
p = 1.5              # stress exponent
n = 3.5              # deformation exponent
λ = 5.0              # nucleation efficiency
M = 125.0            # grain boundary mobility

L  = Float64[0 0 0; 2 0 0; 0 0 0]   # L₂₁ = 2D₀
sr = Float64[0 1 0; 1 0 0; 0 0 0]   # strain rate = ½(L + Lᵀ)

# ── Build ideal girdle in the X–Y plane ──────────────────────────────────────
initial_angles = range(0, 2π, length=N)
orientations = Array{Float64,3}(undef, N, 3, 3)
for (i, θ) in enumerate(initial_angles)
    orientations[i, :, :] .= rotation_from_rotvec([0.0, 0.0, θ])
end
fractions = fill(1.0 / N, N)

# ── Call DRex derivatives (single step) ────────────────────────────────────
println("Computing derivatives for $N grains (ideal girdle)...")
t0 = time()

orientations_diff, fractions_diff = DRex.derivatives(
    matrix_dislocation, olivine, olivine_A, N,
    orientations, fractions, sr, L, fill(NaN, 3, 3),
    p, n, λ, M, 1.0,
)

elapsed = time() - t0
println("Done in $(round(elapsed, digits=2))s")

# ── Extract numerical rotation rates ────────────────────────────────────────
numerical_rotation = [
    sqrt(orientations_diff[i, 1, 1]^2 + orientations_diff[i, 1, 2]^2)
    for i in 1:N
]

# ── Analytical targets ───────────────────────────────────────────────────────
angles_rad = collect(initial_angles)
angles_deg = rad2deg.(angles_rad)
cos2θ = cos.(2 .* angles_rad)

# Rotation rate: ∂θ/∂t = D₀(1 + cos 2θ)
target_rotation = 1 .+ cos2θ

# Grain growth: ρ* = |cos 2θ|^(p/n),  E = ρ* exp(-λ ρ*²),  ∂f/∂ε = -M f(E - Ē)
ρ = abs.(cos2θ) .^ (p / n)
E = ρ .* exp.(-λ .* ρ .^ 2)
Ē = mean(E)
target_fractions_diff = -M .* (1.0 / N) .* (E .- Ē)

# ── Verify numerical accuracy ────────────────────────────────────────────────
max_rot_err = maximum(abs.(numerical_rotation .- target_rotation))
max_frac_err = maximum(abs.(fractions_diff .- target_fractions_diff))
println("Max rotation rate error: $max_rot_err")
println("Max grain growth rate error: $max_frac_err")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig = Figure(size=(700, 800))

# Panel (a): Rotation rates
ax_a = Axis(fig[1, 1],
    ylabel="Rotation rate (∂θ/∂t) / D₀",
    title="Fig. 7 (Bilton et al. 2025): Recrystallisation for ideal girdle in deformation plane",
    xticks=0:45:360,
)

lines!(ax_a, angles_deg, target_rotation, linewidth=2, color=:black,
    label="D₀(1 + cos 2θ)")
idx = 1:250:N   # subsample for readability
scatter!(ax_a, angles_deg[idx], numerical_rotation[idx], markersize=8,
    color=(:steelblue, 0.0), strokecolor=:steelblue, strokewidth=1.5,
    marker=:circle, label="DRex numerical")
vlines!(ax_a, [90, 270], linestyle=:dash, color=:gray60, linewidth=1)
xlims!(ax_a, 0, 360)
ylims!(ax_a, -0.1, 2.1)
axislegend(ax_a, position=:rt)
hidexdecorations!(ax_a, grid=false)

# Panel (b): Grain growth rates (∂f/∂ε)
ax_b = Axis(fig[2, 1],
    xlabel="Initial [100] angle from X-axis (°)",
    ylabel="Grain growth rate (∂f/∂ε)",
    xticks=0:45:360,
)

lines!(ax_b, angles_deg, target_fractions_diff, linewidth=2, color=:black,
    label="−M*f(E − Ē)")
scatter!(ax_b, angles_deg[idx], fractions_diff[idx], markersize=8,
    color=(:steelblue, 0.0), strokecolor=:steelblue, strokewidth=1.5,
    marker=:circle, label="DRex numerical")
vlines!(ax_b, [90, 270], linestyle=:dash, color=:gray60, linewidth=1)
xlims!(ax_b, 0, 360)
axislegend(ax_b, position=:rt)

# Add (a) and (b) labels
Label(fig[1, 1, TopLeft()], "(a)", fontsize=16, font=:bold, padding=(0, 0, 5, 0))
Label(fig[2, 1, TopLeft()], "(b)", fontsize=16, font=:bold, padding=(0, 0, 5, 0))

outpath = joinpath(@__DIR__, "fig7_recrystallisation.png")
save(outpath, fig, px_per_unit=2)
println("Saved: $outpath")
