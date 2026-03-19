# Pathline construction for steady-state flows.

using OrdinaryDiffEq

"""
    get_pathline(final_location, get_velocity, get_velocity_gradient,
                 min_coords, max_coords; max_strain=10.0, regular_steps=nothing)

Determine the pathline for a particle in a steady-state flow by integrating
backwards in time from `final_location`.

Returns `(timestamps, position_fn)` where `position_fn(t)` gives the 3D
position at time `t` (negative times, since integration is backwards).

- `final_location` — 3D coordinates of the final (exit) location
- `get_velocity(t, x)` — velocity callable
- `get_velocity_gradient(t, x)` — velocity gradient callable
- `min_coords` — lower-bound coordinates of the domain
- `max_coords` — upper-bound coordinates of the domain
- `max_strain` — target strain at the final location (default 10)
- `regular_steps` — if set, return regularly-spaced timestamps (default: solver timestamps)
"""
function get_pathline(
    final_location::AbstractVector{Float64},
    get_velocity,
    get_velocity_gradient,
    min_coords::AbstractVector{Float64},
    max_coords::AbstractVector{Float64};
    max_strain::Float64=10.0,
    regular_steps::Union{Int,Nothing}=nothing,
)
    # Accumulated strain, tracked via the callback.
    strain_acc = Ref(max_strain)
    time_prev = Ref(0.0)

    function _is_inside(point)
        @inbounds for i in eachindex(point)
            (point[i] < min_coords[i] || point[i] > max_coords[i]) && return false
        end
        return true
    end

    # RHS: dx/dt = v(t, x), but only inside the domain.
    function rhs!(dx, x, p, t)
        if _is_inside(x)
            v = get_velocity(t, x)
            dx .= v
        else
            dx .= 0.0
        end
    end

    # Termination condition: stop when accumulated strain reaches 0 or leaves domain.
    function condition(u, t, integrator)
        if !_is_inside(u)
            return 0.0  # Terminate
        end
        vel_grad = get_velocity_gradient(NaN, u)
        dε = strain_increment(abs(t - time_prev[]), vel_grad)
        if t < time_prev[]  # Going backwards in time
            strain_acc[] -= dε
        else
            strain_acc[] += dε
        end
        time_prev[] = t
        return strain_acc[]
    end

    # Integrate backwards to t = -100 Myr (in seconds).
    tspan = (0.0, -100e6 * 365.25 * 86400.0)
    prob = ODEProblem(rhs!, copy(final_location), tspan)

    cb = ContinuousCallback(condition, terminate!)

    sol = solve(prob, Tsit5();
                callback=cb,
                abstol=1e-8,
                reltol=1e-5,
                save_everystep=true)

    # Build an interpolator from the solution.
    position_fn = t -> sol(t)

    if regular_steps === nothing
        timestamps = reverse(sol.t)
        return timestamps, position_fn
    else
        timestamps = range(sol.t[end], sol.t[1], length=regular_steps + 1)
        return collect(timestamps), position_fn
    end
end
