using Test
using LinearAlgebra
using DRex
using DRex: voigt_decompose, voigt_to_elastic_tensor, elastic_tensor_to_voigt,
              voigt_matrix_to_vector, voigt_vector_to_matrix, upper_tri_to_symmetric

@testset "Voigt Decompose" begin
    st = StiffnessTensors()
    olivine_tensor = voigt_to_elastic_tensor(st.olivine)
    dilat, voigt = voigt_decompose(st.olivine)
    # einsum("ijkk") → contract last two indices
    dilat_ref = zeros(3, 3)
    for i in 1:3, j in 1:3, k in 1:3
        dilat_ref[i, j] += olivine_tensor[i, j, k, k]
    end
    @test isapprox(dilat, dilat_ref)
    # einsum("ijkj") → contract 2nd and 4th indices
    voigt_ref = zeros(3, 3)
    for i in 1:3, j in 1:3, k in 1:3
        voigt_ref[i, k] += olivine_tensor[i, j, k, j]
    end
    @test isapprox(voigt, voigt_ref)
end

@testset "Voigt Tensor" begin
    st = StiffnessTensors()

    olivine_tensor = zeros(3, 3, 3, 3)
    olivine_tensor[1,1,1,1] = 320.71; olivine_tensor[1,1,2,2] = 69.84;  olivine_tensor[1,1,3,3] = 71.22
    olivine_tensor[1,1,2,3] = 0.0;    olivine_tensor[1,1,3,2] = 0.0
    olivine_tensor[1,2,1,2] = 78.36;  olivine_tensor[1,2,2,1] = 78.36
    olivine_tensor[1,3,1,3] = 77.67;  olivine_tensor[1,3,3,1] = 77.67
    olivine_tensor[2,1,2,1] = 78.36;  olivine_tensor[2,1,1,2] = 78.36
    olivine_tensor[2,2,1,1] = 69.84;  olivine_tensor[2,2,2,2] = 197.25; olivine_tensor[2,2,3,3] = 74.8
    olivine_tensor[2,3,2,3] = 63.77;  olivine_tensor[2,3,3,2] = 63.77
    olivine_tensor[3,1,3,1] = 77.67;  olivine_tensor[3,1,1,3] = 77.67
    olivine_tensor[3,2,3,2] = 63.77;  olivine_tensor[3,2,2,3] = 63.77
    olivine_tensor[3,3,1,1] = 71.22;  olivine_tensor[3,3,2,2] = 74.8;  olivine_tensor[3,3,3,3] = 234.32

    @test isapprox(voigt_to_elastic_tensor(st.olivine), olivine_tensor)
    @test isapprox(elastic_tensor_to_voigt(voigt_to_elastic_tensor(st.olivine)), st.olivine)
    @test isapprox(voigt_to_elastic_tensor(elastic_tensor_to_voigt(olivine_tensor)), olivine_tensor)
    @test isapprox(elastic_tensor_to_voigt(voigt_to_elastic_tensor(st.enstatite)), st.enstatite)
end

@testset "Voigt to Vector" begin
    st = StiffnessTensors()
    expected = [
        236.9, 180.5, 230.4,
        sqrt(2)*56.8, sqrt(2)*63.2, sqrt(2)*79.6,
        168.6, 158.8, 160.2,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ]
    @test isapprox(voigt_matrix_to_vector(st.enstatite), expected, atol=1e-12)
    @test isapprox(st.olivine, voigt_vector_to_matrix(voigt_matrix_to_vector(st.olivine)))

    # Test with random matrix
    r = rand(6, 6)
    v = voigt_matrix_to_vector(r)
    expected_r = [
        r[1,1], r[2,2], r[3,3],
        sqrt(2)*r[2,3], sqrt(2)*r[3,1], sqrt(2)*r[1,2],
        2*r[4,4], 2*r[5,5], 2*r[6,6],
        2*r[1,4], 2*r[2,5], 2*r[3,6],
        2*r[3,4], 2*r[1,5], 2*r[2,6],
        2*r[2,4], 2*r[3,5], 2*r[1,6],
        2*sqrt(2)*r[5,6], 2*sqrt(2)*r[6,4], 2*sqrt(2)*r[4,5],
    ]
    @test isapprox(v, expected_r)
end
