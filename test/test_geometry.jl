using Test
using DRex
using DRex: lambert_equal_area, shirley_concentric_squaredisk
using Random

@testset "Lambert Equal Area" begin
    seed = 8816
    x_range = range(-1, 1, length=11)
    y_range = range(-1, 1, length=11)
    x_flat = Float64[]
    y_flat = Float64[]
    for xi in x_range, yi in y_range
        push!(x_flat, xi)
        push!(y_flat, yi)
    end
    x_disk, y_disk = shirley_concentric_squaredisk(x_flat, y_flat)
    rng = MersenneTwister(seed)
    sign_arr = rand(rng, [1, -1], length(x_disk))
    z_vals = [s * (1 - (xd^2 + yd^2)) for (s, xd, yd) in zip(sign_arr, x_disk, y_disk)]
    x_laea, y_laea = lambert_equal_area(x_disk, y_disk, z_vals)
    @test isapprox(x_disk, x_laea, atol=1e-15)
    @test isapprox(y_disk, y_laea, atol=1e-15)
end
