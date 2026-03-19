# Steady-state solutions of velocity (gradients) for various flows.
#
# All callables returned from functions in this module have signature f(t, x)
# where t is a time scalar and x is a 3D position vector. They return 3D tensors
# so they can be directly used as arguments to `update_orientations!`.

"""
    simple_shear_2d(direction, deformation_plane, strain_rate)

Return `(velocity, velocity_gradient)` callables for 2D simple shear.

- `direction` — "X", "Y", or "Z": velocity vector direction
- `deformation_plane` — "X", "Y", or "Z": direction of velocity gradient
- `strain_rate` — 1/2 × magnitude of the largest eigenvalue of the velocity gradient
"""
function simple_shear_2d(direction::String, deformation_plane::String, strain_rate::Float64)
    d, p = to_indices2d(direction, deformation_plane)

    velocity = let d=d, p=p, sr=strain_rate
        function(t, x)
            v = zeros(3)
            v[d] = x[p] * sr
            return v
        end
    end

    velocity_gradient = let d=d, p=p, sr=strain_rate
        function(t, x)
            L = zeros(3, 3)
            L[d, p] = 2 * sr
            return L
        end
    end

    return (velocity, velocity_gradient)
end

"""
    cell_2d(horizontal, vertical, velocity_edge; edge_length=2.0)

Return `(velocity, velocity_gradient)` callables for a steady-state 2D Stokes convection cell.

The cell is centered at (0,0) with velocity:
  u = U cos(πx/d) sin(πz/d) ĥ - U sin(πx/d) cos(πz/d) v̂

- `horizontal` — "X", "Y", or "Z"
- `vertical` — "X", "Y", or "Z"
- `velocity_edge` — velocity magnitude at cell edge center
- `edge_length` — cell edge length (default 2.0)
"""
function cell_2d(horizontal::String, vertical::String, velocity_edge::Float64;
                 edge_length::Float64=2.0)
    edge_length < 0 && throw(ArgumentError("edge length must be positive, got $edge_length"))
    h, v = to_indices2d(horizontal, vertical)

    velocity = let h=h, v=v, U=velocity_edge, d=edge_length
        function(t, x)
            _lim = d / 2
            (abs(x[h]) > _lim || abs(x[v]) > _lim) &&
                throw(DomainError(x, "position outside domain with xᵢ ∈ [-$_lim, $_lim]"))
            out = zeros(3)
            out[h] =  U * cos(π * x[h] / d) * sin(π * x[v] / d)
            out[v] = -U * sin(π * x[h] / d) * cos(π * x[v] / d)
            return out
        end
    end

    velocity_gradient = let h=h, v=v, U=velocity_edge, d=edge_length
        function(t, x)
            _lim = d / 2
            (abs(x[h]) > _lim || abs(x[v]) > _lim) &&
                throw(DomainError(x, "position outside domain with xᵢ ∈ [-$_lim, $_lim]"))
            s_x = sin(π * x[h] / d)
            c_x = cos(π * x[h] / d)
            s_z = sin(π * x[v] / d)
            c_z = cos(π * x[v] / d)
            L = zeros(3, 3)
            L[h, h] = -π / d * s_z * s_x
            L[h, v] =  π / d * c_z * c_x
            L[v, v] = -π / d * c_z * c_x
            L[v, h] =  π / d * s_z * s_x
            return U * L
        end
    end

    return (velocity, velocity_gradient)
end

"""
    corner_2d(horizontal, vertical, plate_speed)

Return `(velocity, velocity_gradient)` callables for steady-state 2D corner flow.

Velocity field:
  u = (2U/π)(arctan(x/(-z)) + xz/(x²+z²)) x̂ + (2U/π)(z²/(x²+z²)) ẑ

- `horizontal` — "X", "Y", or "Z"
- `vertical` — "X", "Y", or "Z"
- `plate_speed` — upper boundary ("plate") speed
"""
function corner_2d(horizontal::String, vertical::String, plate_speed::Float64)
    hi, vi = to_indices2d(horizontal, vertical)

    velocity = let h=hi, v=vi, U=plate_speed
        function(t, x)
            xh = x[h]
            xv = x[v]
            if abs(xh) < 1e-15 && abs(xv) < 1e-15
                return fill(NaN, 3)
            end
            out = zeros(3)
            pf = 2U / π
            out[h] = pf * (atan(xh, -xv) + xh * xv / (xh^2 + xv^2))
            out[v] = pf * xv^2 / (xh^2 + xv^2)
            return out
        end
    end

    velocity_gradient = let h=hi, v=vi, U=plate_speed
        function(t, x)
            xh = x[h]
            xv = x[v]
            if abs(xh) < 1e-15 && abs(xv) < 1e-15
                return fill(NaN, 3, 3)
            end
            L = zeros(3, 3)
            pf = 4U / (π * (xh^2 + xv^2)^2)
            L[h, h] = -xh^2 * xv
            L[h, v] =  xh^3
            L[v, h] = -xh * xv^2
            L[v, v] =  xh^2 * xv
            return pf * L
        end
    end

    return (velocity, velocity_gradient)
end
