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
    backend::KernelAbstractions.Backend = CPU(),
    push_snapshot::Bool = true,
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

    # Pre-allocate buffers once and reuse across every ODE step.
    # On CPU backend these are plain Arrays (no overhead); on GPU they are
    # device arrays so the kernel can operate on them directly.
    ori_device      = KernelAbstractions.allocate(backend, T, (n_grains, 3, 3))
    ori_diff_device = KernelAbstractions.allocate(backend, T, (n_grains, 3, 3))
    frac_diff_buf   = Vector{T}(undef, n_grains)

    function rhs!(dy, y, _, t)
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

        # Upload orientations to the device (no-op memcopy on CPU backend).
        copyto!(ori_device, ori)

        derivatives!(ori_diff_device, frac_diff_buf,
                     mineral.regime, mineral.phase, mineral.fabric, n_grains,
                     ori_device, frac,
                     sr ./ sr_max, vel_grad ./ sr_max, V,
                     params[:stress_exponent],
                     params[:deformation_exponent],
                     params[:nucleation_efficiency],
                     params[:gbm_mobility],
                     volume_fraction;
                     backend = backend)

        dy[1:9] .= vec(F_diff)
        # Download orientation derivatives back to CPU (no-op on CPU backend).
        dy[10:n_grains*9+9] .= vec(Array(ori_diff_device)) .* sr_max
        dy[n_grains*9+10:n_grains*10+9] .= frac_diff_buf .* sr_max
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
    if push_snapshot
        push!(mineral.orientations, ori_new)
        push!(mineral.fractions, frac_new)
    else
        # Overwrite the last snapshot in-place so the next call starts from the
        # correct evolved state without growing the snapshot vectors.
        mineral.orientations[end] = ori_new
        mineral.fractions[end] = frac_new
    end
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
    backend::KernelAbstractions.Backend = CPU(),
    push_snapshot::Bool = true,
)
    new_F = deformation_gradient
    for mineral in minerals
        new_F = update_orientations!(
            mineral, params, deformation_gradient,
            get_velocity_gradient, pathline;
            get_regime=get_regime,
            backend=backend,
            push_snapshot=push_snapshot,
        )
    end
    return new_F
end

"""
    run_pathlines_batch!(minerals_per_tracer, params, pathlines_data; backend=CPU())

Integrate CPO for `n_tracers` pathlines using Tsit5 (adaptive Runge-Kutta).

Each outer timestep is integrated with `update_all!` (Tsit5), which internally uses
the same GPU kernel as `update_orientations!`.  Velocity gradients are linearly
interpolated between consecutive pathline samples within each interval.

# Arguments
- `minerals_per_tracer` — `Vector{Vector{Mineral{T}}}` of length `n_tracers`;
  each inner vector holds one `Mineral` per phase in the same order for every tracer.
- `params` — parameter dict from `default_params()`, identical for all tracers.
- `pathlines_data` — `Vector` of length `n_tracers`; each element is a named tuple or
  tuple `(timestamps, positions, velocity_gradients)`:
  - `timestamps` — `Vector{Float64}` of length `n_steps`
  - `positions` — `Vector{Vector}` of 3-D positions at each timestamp
  - `velocity_gradients` — `Vector{Matrix}` of 3×3 VG matrices at each timestamp
- `backend` — KernelAbstractions backend (default `CPU()`).
- `snapshot_stride` — save an orientation/fraction snapshot every this many steps
  (default `1` = every step).  Use e.g. `10` to reduce snapshot count 10×,
  keeping only ~51 output frames for 500 inner steps.  The returned strains
  vector is subsampled to match.

Returns `Vector{Vector{T}}` of per-tracer accumulated strain values
(length `1 + cld(n_steps-1, snapshot_stride)`).
"""
function run_pathlines_batch!(
    minerals_per_tracer::Vector{<:Vector{<:Mineral{T}}},
    params::Dict{Symbol,Any},
    pathlines_data::AbstractVector;
    backend::KernelAbstractions.Backend = CPU(),
    snapshot_stride::Int = 1,
) where T<:AbstractFloat
    n_tracers = length(minerals_per_tracer)
    n_steps   = length(pathlines_data[1][1])

    for ti in 2:n_tracers
        length(pathlines_data[ti][1]) == n_steps ||
            error("all pathlines must have the same number of timesteps for batch processing")
    end

    # Number of output snapshots (initial state + one per stride).
    n_snapshots = 1 + cld(n_steps - 1, snapshot_stride)

    # Per-tracer strain at each saved snapshot (length = n_snapshots).
    strains     = [zeros(T, n_snapshots) for _ in 1:n_tracers]
    # Running cumulative strain (updated every step, flushed to strains at snapshots).
    cum_strains = zeros(Float64, n_tracers)

    # Per-tracer macroscopic deformation gradient (identity initially).
    def_grads = [Matrix{T}(I, 3, 3) for _ in 1:n_tracers]

    if backend isa KernelAbstractions.CPU
        # ── CPU path: Tsit5 per tracer, parallelised with Threads.@threads ──────
        for step in 2:n_steps
            save_snapshot = (step - 1) % snapshot_stride == 0 || step == n_steps
            snap_idx      = save_snapshot ? cld(step - 1, snapshot_stride) + 1 : 0

            inner! = let step=step, save_snapshot=save_snapshot, snap_idx=snap_idx
                ti -> begin
                    timestamps = pathlines_data[ti][1]
                    positions  = pathlines_data[ti][2]
                    vgs        = pathlines_data[ti][3]

                    t0   = Float64(timestamps[step-1])
                    t1   = Float64(timestamps[step])
                    pos0 = positions[step-1]
                    pos1 = positions[step]
                    vg0  = Matrix{T}(vgs[step-1])
                    vg1  = Matrix{T}(vgs[step])

                    get_position = let t0=t0, t1=t1, pos0=pos0, pos1=pos1
                        t -> begin
                            α = (t - t0) / (t1 - t0)
                            (1.0-α) .* pos0 .+ α .* pos1
                        end
                    end
                    get_vg = let t0=t0, t1=t1, vg0=vg0, vg1=vg1
                        (t, _) -> begin
                            α = T((t - t0) / (t1 - t0))
                            (1-α) .* vg0 .+ α .* vg1
                        end
                    end

                    def_grads[ti] = update_all!(
                        minerals_per_tracer[ti], params, def_grads[ti],
                        get_vg, (t0, t1, get_position);
                        backend=backend,
                        push_snapshot=save_snapshot,
                    )

                    cum_strains[ti] += strain_increment(t1 - t0, Float64.(vg1))

                    if save_snapshot
                        strains[ti][snap_idx] = T(cum_strains[ti])
                    end
                end
            end

            Threads.@threads for ti in 1:n_tracers
                inner!(ti)
            end
        end

    else
        # ── GPU batch path: adaptive-substep RK4 with _batch_grain_kernel! ────────
        # All tracers are processed simultaneously per kernel launch.
        # The number of RK4 sub-steps per outer timestep is determined from two
        # stability bounds:
        #   orientation: srm·h_sub ≤ 0.3  →  n_sub ≥ srm·|h_outer| / 0.3
        #   fraction:    M_eff·srm·h_sub ≤ 2.79  →  n_sub ≥ M_eff·srm·|h_outer| / 2.79
        # where srm = sr_max = max absolute strain-rate eigenvalue.

        n_phases = length(minerals_per_tracer[1])
        n_grains  = minerals_per_tracer[1][1].n_grains

        # ── Device buffers ───────────────────────────────────────────────────────
        ori_device   = KernelAbstractions.allocate(backend, T, (n_grains, n_tracers, 3, 3))
        ori_diff_dev = KernelAbstractions.allocate(backend, T, (n_grains, n_tracers, 3, 3))
        se_device    = KernelAbstractions.allocate(backend, T, (n_grains, n_tracers))
        vg_dev       = KernelAbstractions.allocate(backend, T, (9, n_tracers))
        sr_dev       = KernelAbstractions.allocate(backend, T, (9, n_tracers))

        # ── CPU working buffers ──────────────────────────────────────────────────
        ori_cpu    = Array{T,4}(undef, n_grains, n_tracers, 3, 3)
        fracs_cpu  = Array{T,2}(undef, n_grains, n_tracers)
        ori_start  = Array{T,4}(undef, n_grains, n_tracers, 3, 3)
        ori_tmp    = Array{T,4}(undef, n_grains, n_tracers, 3, 3)
        fracs_tmp  = Array{T,2}(undef, n_grains, n_tracers)
        k1_ori     = Array{T,4}(undef, n_grains, n_tracers, 3, 3)
        k2_ori     = Array{T,4}(undef, n_grains, n_tracers, 3, 3)
        k3_ori     = Array{T,4}(undef, n_grains, n_tracers, 3, 3)
        k4_ori     = Array{T,4}(undef, n_grains, n_tracers, 3, 3)
        k1_frac    = Array{T,2}(undef, n_grains, n_tracers)
        k2_frac    = Array{T,2}(undef, n_grains, n_tracers)
        k3_frac    = Array{T,2}(undef, n_grains, n_tracers)
        k4_frac    = Array{T,2}(undef, n_grains, n_tracers)
        ori_diff   = Array{T,4}(undef, n_grains, n_tracers, 3, 3)
        se_cpu     = Array{T,2}(undef, n_grains, n_tracers)
        se_k1      = Array{T,2}(undef, n_grains, n_tracers)
        se_k2      = Array{T,2}(undef, n_grains, n_tracers)
        se_k3      = Array{T,2}(undef, n_grains, n_tracers)
        se_k4      = Array{T,2}(undef, n_grains, n_tracers)
        se_avg     = Array{T,2}(undef, n_grains, n_tracers)
        vg_cpu     = Array{T,2}(undef, 9, n_tracers)
        sr_cpu     = Array{T,2}(undef, 9, n_tracers)
        sr_max_arr = Vector{T}(undef, n_tracers)
        sr_max_k1  = Vector{T}(undef, n_tracers)

        vg0_all = [Matrix{T}(undef, 3, 3) for _ in 1:n_tracers]
        vg1_all = [Matrix{T}(undef, 3, 3) for _ in 1:n_tracers]

        batch_kern = _batch_grain_kernel!(backend, 64)

        # ── Inner helper: evaluate batch RHS at (ori_in, fracs_in, α) ───────────
        # α ∈ [0,1] interpolates VG within the current outer step.
        # Writes scaled time derivatives into dest_ori / dest_frac.
        function eval_batch!(dest_ori, dest_frac,
                             ori_in, fracs_in, α,
                             phase_int, fabric_int, smoothing,
                             vol_frac, gbm_M, s_exp, d_exp, n_eff)
            for ti in 1:n_tracers
                vg_ti  = (1-α) .* vg0_all[ti] .+ α .* vg1_all[ti]
                sr_ti  = (vg_ti .+ vg_ti') ./ 2
                sr_max = T(maximum(abs, eigvals(Symmetric(Matrix{Float64}(sr_ti)))))
                # Use actual sr_max (no eps clamp) — eps(T) ≈ 1.19e-7 for Float32,
                # which is >> actual corner-flow strain rates (~1e-14 s⁻¹) and would
                # artificially inflate the stiffness of the fraction ODE by ~10^7×.
                # Use a tiny floor only to guard against true-zero sr_max.
                sr_max = max(sr_max, T(1e-30))
                sr_max_arr[ti] = sr_max
                @inbounds for j in 1:3, i in 1:3
                    k = (j-1)*3 + i
                    vg_cpu[k,ti] = vg_ti[i,j] / sr_max
                    sr_cpu[k,ti] = sr_ti[i,j]  / sr_max
                end
            end
            copyto!(vg_dev,    vg_cpu)
            copyto!(sr_dev,    sr_cpu)
            copyto!(ori_device, ori_in)
            batch_kern(
                ori_diff_dev, se_device, ori_device, vg_dev, sr_dev,
                phase_int, fabric_int, smoothing, s_exp, d_exp, n_eff;
                ndrange = n_grains * n_tracers,
            )
            KernelAbstractions.synchronize(backend)
            copyto!(ori_diff, ori_diff_dev)
            copyto!(se_cpu,   se_device)
            @inbounds for ti in 1:n_tracers
                sr_max = sr_max_arr[ti]
                mean_E = zero(T)
                for g in 1:n_grains
                    mean_E += fracs_in[g,ti] * se_cpu[g,ti]
                end
                for i in 1:3, j in 1:3, g in 1:n_grains
                    dest_ori[g,ti,i,j] = ori_diff[g,ti,i,j] * sr_max
                end
                for g in 1:n_grains
                    dest_frac[g,ti] = vol_frac * gbm_M * fracs_in[g,ti] *
                        smoothing * (mean_E - se_cpu[g,ti]) * sr_max
                end
            end
        end

        for step in 2:n_steps
            save_snapshot = (step - 1) % snapshot_stride == 0 || step == n_steps
            snap_idx      = save_snapshot ? cld(step - 1, snapshot_stride) + 1 : 0
            h_outer       = T(pathlines_data[1][1][step] - pathlines_data[1][1][step-1])

            # Accumulate strain + Euler-update deformation gradients
            for ti in 1:n_tracers
                t0  = Float64(pathlines_data[ti][1][step-1])
                t1  = Float64(pathlines_data[ti][1][step])
                vg1 = pathlines_data[ti][3][step]
                cum_strains[ti] += strain_increment(t1 - t0, Float64.(vg1))
                if save_snapshot
                    strains[ti][snap_idx] = T(cum_strains[ti])
                end
                def_grads[ti] = (I + Matrix{T}(vg1) * T(t1 - t0)) * def_grads[ti]
            end

            # Cache endpoint VGs for interpolation inside eval_batch!
            for ti in 1:n_tracers
                vg0_all[ti] .= pathlines_data[ti][3][step-1]
                vg1_all[ti] .= pathlines_data[ti][3][step]
            end

            for phase_idx in 1:n_phases
                phase_int  = Int32(Int(minerals_per_tracer[1][phase_idx].phase))
                fabric_int = Int32(Int(minerals_per_tracer[1][phase_idx].fabric))
                vol_frac   = T(params[:phase_fractions][phase_idx])
                gbm_M      = T(params[:gbm_mobility])
                smoothing  = T(minerals_per_tracer[1][phase_idx].regime == frictional_yielding ? 0.3 : 1.0)
                gbs_th_g   = T(params[:gbs_threshold]) / n_grains
                s_exp      = T(params[:stress_exponent])
                d_exp      = T(params[:deformation_exponent])
                n_eff      = T(params[:nucleation_efficiency])
                # Pack current orientations/fractions; save outer-step start for GBS
                for ti in 1:n_tracers
                    m      = minerals_per_tracer[ti][phase_idx]
                    ori_s  = m.orientations[end]
                    frac_s = m.fractions[end]
                    @inbounds for i in 1:3, j in 1:3, g in 1:n_grains
                        v = ori_s[g,i,j]
                        ori_cpu[g,ti,i,j]   = v
                        ori_start[g,ti,i,j] = v
                    end
                    @inbounds for g in 1:n_grains
                        fracs_cpu[g,ti] = frac_s[g]
                    end
                end

                # ── Determine number of RK4 sub-steps from orientation stability ─
                # The fraction ODE is extremely stiff (α*h_outer >> 1) so it is
                # integrated with an exact exponential integrator instead of RK4.
                # Only the orientation stability bound determines n_sub.
                max_srm_h = T(0)
                for ti in 1:n_tracers
                    for α_end in (T(0), T(1))
                        vg_ti  = (1-α_end) .* vg0_all[ti] .+ α_end .* vg1_all[ti]
                        sr_ti  = (vg_ti .+ vg_ti') ./ 2
                        srm    = T(maximum(abs, eigvals(Symmetric(Matrix{Float64}(sr_ti)))))
                        max_srm_h = max(max_srm_h, srm * abs(h_outer))
                    end
                end
                n_sub = max(1, ceil(Int, Float64(max_srm_h) / 0.3))
                h_sub = h_outer / T(n_sub)

                # ── RK4 sub-steps (orientations) + exponential integrator (fracs) ─
                for s in 0:n_sub-1
                    α0 = T(s)   / T(n_sub)   # start of sub-step
                    α1 = T(s+1) / T(n_sub)   # end of sub-step
                    αm = (α0 + α1) / 2        # midpoint

                    # k1 = f(ori, α0) — also captures se_cpu and sr_max for frac update
                    eval_batch!(k1_ori, k1_frac, ori_cpu, fracs_cpu, α0,
                                phase_int, fabric_int, smoothing,
                                vol_frac, gbm_M, s_exp, d_exp, n_eff)
                    # Save strain energies + sr_max from k1 for the exponential frac update.
                    # Orientation derivatives do NOT depend on fracs, so se_cpu is fully
                    # determined by orientations and VG alone.
                    se_k1     .= se_cpu
                    sr_max_k1 .= sr_max_arr

                    # k2 = f(ori + h/2·k1, αm)  — fracs constant (don't affect ori deriv)
                    @inbounds for ti in 1:n_tracers, i in 1:3, j in 1:3, g in 1:n_grains
                        ori_tmp[g,ti,i,j] = ori_cpu[g,ti,i,j] + (h_sub/2) * k1_ori[g,ti,i,j]
                    end
                    eval_batch!(k2_ori, k2_frac, ori_tmp, fracs_cpu, αm,
                                phase_int, fabric_int, smoothing,
                                vol_frac, gbm_M, s_exp, d_exp, n_eff)
                    se_k2 .= se_cpu

                    # k3 = f(ori + h/2·k2, αm)
                    @inbounds for ti in 1:n_tracers, i in 1:3, j in 1:3, g in 1:n_grains
                        ori_tmp[g,ti,i,j] = ori_cpu[g,ti,i,j] + (h_sub/2) * k2_ori[g,ti,i,j]
                    end
                    eval_batch!(k3_ori, k3_frac, ori_tmp, fracs_cpu, αm,
                                phase_int, fabric_int, smoothing,
                                vol_frac, gbm_M, s_exp, d_exp, n_eff)
                    se_k3 .= se_cpu

                    # k4 = f(ori + h·k3, α1)
                    @inbounds for ti in 1:n_tracers, i in 1:3, j in 1:3, g in 1:n_grains
                        ori_tmp[g,ti,i,j] = ori_cpu[g,ti,i,j] + h_sub * k3_ori[g,ti,i,j]
                    end
                    eval_batch!(k4_ori, k4_frac, ori_tmp, fracs_cpu, α1,
                                phase_int, fabric_int, smoothing,
                                vol_frac, gbm_M, s_exp, d_exp, n_eff)
                    se_k4 .= se_cpu

                    # ── Orientation RK4 update ────────────────────────────────────
                    @inbounds for ti in 1:n_tracers, i in 1:3, j in 1:3, g in 1:n_grains
                        ori_cpu[g,ti,i,j] += (h_sub / 6) * (
                            k1_ori[g,ti,i,j] + 2*k2_ori[g,ti,i,j] +
                            2*k3_ori[g,ti,i,j] + k4_ori[g,ti,i,j])
                    end

                    # ── Exponential fraction integrator with adaptive sub-stepping ─
                    # The fraction ODE dfrac/dt = M·vf·s·frac·(mean_E - E_g)·sr has
                    # exact solution frac(t+h) ∝ frac(t)·exp(α_g·h).
                    #
                    # Use RK4-weighted average of SE across k1..k4 to account for
                    # orientation evolution during the sub-step (reduces operator-
                    # splitting error vs using SE frozen at k1 alone).
                    @inbounds for ti in 1:n_tracers
                        for g in 1:n_grains
                            se_avg[g,ti] = (se_k1[g,ti] + 2*se_k2[g,ti] +
                                            2*se_k3[g,ti] + se_k4[g,ti]) / 6
                        end
                    end

                    α_frac_max = zero(T)
                    @inbounds for ti in 1:n_tracers
                        sr = sr_max_k1[ti]
                        mean_E = zero(T)
                        for g in 1:n_grains
                            mean_E += fracs_cpu[g,ti] * se_avg[g,ti]
                        end
                        for g in 1:n_grains
                            ag = abs(vol_frac * gbm_M * smoothing * (mean_E - se_avg[g,ti]) * sr)
                            α_frac_max = max(α_frac_max, ag)
                        end
                    end
                    n_frac_sub = max(1, ceil(Int, Float64(α_frac_max) * Float64(h_sub) / 2.79))
                    h_frac = h_sub / T(n_frac_sub)

                    for _ in 1:n_frac_sub
                        @inbounds for ti in 1:n_tracers
                            sr = sr_max_k1[ti]
                            mean_E = zero(T)
                            for g in 1:n_grains
                                mean_E += fracs_cpu[g,ti] * se_avg[g,ti]
                            end
                            # Log-space update, shift by max to avoid Float32 overflow
                            max_log_w = T(-Inf32)
                            for g in 1:n_grains
                                α_g = vol_frac * gbm_M * smoothing * (mean_E - se_avg[g,ti]) * sr
                                lw = log(max(fracs_cpu[g,ti], eps(T))) + α_g * h_frac
                                fracs_tmp[g,ti] = lw
                                max_log_w = max(max_log_w, lw)
                            end
                            s_frac = zero(T)
                            for g in 1:n_grains
                                fracs_cpu[g,ti] = exp(fracs_tmp[g,ti] - max_log_w)
                                s_frac += fracs_cpu[g,ti]
                            end
                            for g in 1:n_grains
                                fracs_cpu[g,ti] /= s_frac
                            end
                        end
                    end

                    # Apply GBS after each sub-step (compare against outer-step start)
                    @inbounds for ti in 1:n_tracers
                        s_frac = zero(T)
                        for g in 1:n_grains
                            if fracs_cpu[g,ti] < gbs_th_g
                                for i in 1:3, j in 1:3
                                    ori_cpu[g,ti,i,j] = ori_start[g,ti,i,j]
                                end
                                fracs_cpu[g,ti] = gbs_th_g
                            end
                            s_frac += fracs_cpu[g,ti]
                        end
                        for g in 1:n_grains
                            fracs_cpu[g,ti] /= s_frac
                        end
                    end
                end  # for s in sub-steps

                # ── Store snapshot from final state ───────────────────────────
                for ti in 1:n_tracers
                    m         = minerals_per_tracer[ti][phase_idx]
                    ori_snap  = Array{T,3}(undef, n_grains, 3, 3)
                    frac_snap = Vector{T}(undef, n_grains)
                    @inbounds for g in 1:n_grains
                        frac_snap[g] = fracs_cpu[g,ti]
                        for i in 1:3, j in 1:3
                            ori_snap[g,i,j] = ori_cpu[g,ti,i,j]
                        end
                    end
                    if save_snapshot
                        push!(m.orientations, ori_snap)
                        push!(m.fractions, frac_snap)
                    else
                        m.orientations[end] = ori_snap
                        m.fractions[end] = frac_snap
                    end
                end
            end  # for phase_idx
        end  # for step
    end  # if CPU / else GPU

    return strains
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
