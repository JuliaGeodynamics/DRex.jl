using Test
using DRex
using DRex: SCSVError, read_scsv, save_scsv, write_scsv_header, scsv_data

# ═════════════════════════════════════════════════════════════════════════════
# Schema validation
# ═════════════════════════════════════════════════════════════════════════════
@testset "SCSV Schema Validation" begin
    @test_throws SCSVError save_scsv(
        tempname(), Dict("delimiter"=>",", "missing"=>"-",
            "fields"=>[Dict("name"=>"nofill", "type"=>"float")]),
        [[0.1]],
    )
    @test_throws SCSVError save_scsv(
        tempname(), Dict("delimiter"=>",",
            "fields"=>[Dict("name"=>"nomissing", "type"=>"float", "fill"=>"NaN")]),
        [[0.1]],
    )
    @test_throws SCSVError save_scsv(
        tempname(), Dict("delimiter"=>",", "missing"=>"-"),
        [[0.1]],
    )
    @test_throws SCSVError save_scsv(
        tempname(), Dict("delimiter"=>",", "missing"=>"-",
            "fields"=>[Dict("name"=>"bad name", "type"=>"float", "fill"=>"NaN")]),
        [[0.1]],
    )
    @test_throws SCSVError save_scsv(
        tempname(), Dict("delimiter"=>",", "missing"=>",",
            "fields"=>[Dict("name"=>"baddelim", "type"=>"float", "fill"=>"NaN")]),
        [[0.1]],
    )
    @test_throws SCSVError save_scsv(
        tempname(), Dict("delimiter"=>",", "missing"=>"-,",
            "fields"=>[Dict("name"=>"baddelim", "type"=>"float", "fill"=>"NaN")]),
        [[0.1]],
    )
    @test_throws SCSVError save_scsv(
        tempname(), Dict("delimiter"=>",,", "missing"=>"-",
            "fields"=>[Dict("name"=>"baddelim", "type"=>"float", "fill"=>"NaN")]),
        [[0.1]],
    )
end

# ═════════════════════════════════════════════════════════════════════════════
# Read spec file
# ═════════════════════════════════════════════════════════════════════════════
@testset "SCSV Read Spec File" begin
    data = read_scsv(joinpath(scsv_data("specs"), "spec.scsv"))

    @test keys(data) == (
        :first_column, :second_column, :third_column,
        :float_column, :bool_column, :complex_column,
    )
    @test data.first_column  == ["s1", "MISSING", "s3"]
    @test data.second_column == ["A", "B, b", ""]
    @test data.third_column  == [999999, 10, 1]
    # Float column: second entry is NaN — test with isequal for NaN equality.
    @test data.float_column[1] == 1.1
    @test isnan(data.float_column[2])
    @test data.float_column[3] == 1.0
    @test data.bool_column == [true, false, true]
    # Complex column: second entry is NaN+0im.
    @test data.complex_column[1] == 0.1 + 0.0im
    @test isnan(real(data.complex_column[2])) && imag(data.complex_column[2]) == 0.0
    @test data.complex_column[3] == 1.0 + 1.0im
end

# ═════════════════════════════════════════════════════════════════════════════
# Save and re-read spec file (roundtrip)
# ═════════════════════════════════════════════════════════════════════════════
@testset "SCSV Save Spec File" begin
    schema = Dict(
        "delimiter" => ",",
        "missing" => "-",
        "fields" => [
            Dict("name"=>"first_column", "type"=>"string", "fill"=>"MISSING", "unit"=>"percent"),
            Dict("name"=>"second_column"),
            Dict("name"=>"third_column", "type"=>"integer", "fill"=>"999999"),
            Dict("name"=>"float_column", "type"=>"float", "fill"=>"NaN"),
            Dict("name"=>"bool_column", "type"=>"boolean"),
            Dict("name"=>"complex_column", "type"=>"complex", "fill"=>"NaN"),
        ],
    )
    schema_alt = Dict(
        "delimiter" => ",",
        "missing" => "-",
        "fields" => [
            Dict("name"=>"first_column", "type"=>"string", "unit"=>"percent"),
            Dict("name"=>"second_column"),
            Dict("name"=>"third_column", "type"=>"integer", "fill"=>"999991"),
            Dict("name"=>"float_column", "type"=>"float", "fill"=>"0.0"),
            Dict("name"=>"bool_column", "type"=>"boolean"),
            Dict("name"=>"complex_column", "type"=>"complex", "fill"=>"NaN"),
        ],
    )

    data = [
        ["s1", "MISSING", "s3"],
        ["A", "B, b", ""],
        [999999, 10, 1],
        [1.1, NaN, 1.0],
        [true, false, true],
        [0.1+0im, NaN+0im, 1.0+1.0im],
    ]
    data_alt = [
        ["s1", "", "s3"],
        ["A", "B, b", ""],
        [999991, 10, 1],
        [1.1, 0.0, 1.0],
        [true, false, true],
        [0.1+0im, NaN+0im, 1.0+1.0im],
    ]

    temp1 = tempname() * ".scsv"
    temp2 = tempname() * ".scsv"
    save_scsv(temp1, schema, data)
    save_scsv(temp2, schema_alt, data_alt)

    # Read raw lines after the YAML header — CSV content must match.
    function csv_lines_after_header(filepath)
        lines = readlines(filepath)
        # Find the second "---" which ends the header.
        header_end = 0
        count = 0
        for (i, l) in enumerate(lines)
            if strip(l) == "---"
                count += 1
                if count == 2
                    header_end = i
                    break
                end
            end
        end
        return filter(!isempty, lines[header_end+1:end])
    end

    raw1 = csv_lines_after_header(temp1)
    raw2 = csv_lines_after_header(temp2)
    @test raw1 == raw2

    # Roundtrip: re-read and verify.
    reread = read_scsv(temp1)
    @test reread.first_column  == ["s1", "MISSING", "s3"]
    @test reread.second_column == ["A", "B, b", ""]
    @test reread.third_column  == [999999, 10, 1]
    @test reread.float_column[1] == 1.1
    @test isnan(reread.float_column[2])
    @test reread.float_column[3] == 1.0
    @test reread.bool_column == [true, false, true]
    @test reread.complex_column[1] == 0.1 + 0im
    @test isnan(real(reread.complex_column[2]))
    @test reread.complex_column[3] == 1.0 + 1.0im

    rm(temp1; force=true)
    rm(temp2; force=true)
end

# ═════════════════════════════════════════════════════════════════════════════
# Save errors
# ═════════════════════════════════════════════════════════════════════════════
@testset "SCSV Save Errors" begin
    schema = Dict(
        "delimiter" => ",",
        "missing" => "-",
        "fields" => [Dict("name"=>"foo", "type"=>"integer", "fill"=>"999999")],
    )
    # A column with a float among integers should error.
    @test_throws SCSVError save_scsv(tempname(), schema, [[1, 5, 0.2]])
end

# ═════════════════════════════════════════════════════════════════════════════
# Read Kaminski 2002
# ═════════════════════════════════════════════════════════════════════════════
@testset "SCSV Read Kaminski2002" begin
    data = read_scsv(joinpath(scsv_data("thirdparty"), "Kaminski2002_ISAtime.scsv"))
    @test keys(data) == (:time_ISA, :vorticity)
    @test data.time_ISA ≈ [
        2.48, 2.50, 2.55, 2.78, 3.07, 3.58, 4.00, 4.88, 4.01, 3.79,
        3.72, 3.66, 3.71, 4.22, 4.73, 3.45, 1.77, 0.51,
    ]
    @test data.vorticity ≈ [
        0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
        0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
    ]
end

# ═════════════════════════════════════════════════════════════════════════════
# Read Kaminski 2004
# ═════════════════════════════════════════════════════════════════════════════
@testset "SCSV Read Kaminski2004" begin
    data = read_scsv(joinpath(scsv_data("thirdparty"), "Kaminski2004_AaxisDynamicShear.scsv"))
    @test keys(data) == (:time, :meanA_X0, :meanA_X02, :meanA_X04)
    @test data.time ≈ [
        -0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
        2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1,
        3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2,
        4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0,
    ]
    @test data.meanA_X02 ≈ [
        -0.54, -0.54, -0.27, 0.13, 0.94, 2.82, 5.37, 9.53, 14.77, 20.40,
        26.58, 32.89, 39.73, 47.25, 53.69, 58.66, 60.81, 60.81, 59.73, 58.52, 58.12,
        56.64, 54.09, 53.69, 55.57, 57.05, 58.66, 60.54, 60.81, 61.21, 61.21, 61.61,
        61.48, 61.61, 61.61, 61.61, 61.21, 61.21, 61.07, 60.81, 60.81, 60.54, 60.00,
        59.60, 59.33, 58.52, 58.12, 57.85, 57.45, 57.05, 57.05,
    ]
end

# ═════════════════════════════════════════════════════════════════════════════
# Read Skemer 2016
# ═════════════════════════════════════════════════════════════════════════════
@testset "SCSV Read Skemer2016" begin
    data = read_scsv(joinpath(scsv_data("thirdparty"), "Skemer2016_ShearStrainAngles.scsv"))
    @test keys(data) == (:study, :sample_id, :shear_strain, :angle, :fabric, :M_index)

    @test data.study[1:5] == [
        "Z&K 1200 C", "Z&K 1200 C", "Z&K 1200 C", "Z&K 1200 C", "Z&K 1200 C",
    ]
    @test data.study[end-3:end] == ["H&W 2015", "H&W 2015", "H&W 2015", "H&W 2015"]

    @test data.sample_id[1:5] == ["PI-148", "PI-150", "PI-154", "PI-158", "PI-284"]

    @test data.shear_strain[1:10] == [17, 30, 45, 65, 110, 11, 7, 65, 58, 100]
    @test data.shear_strain[end-3:end] == [386, 386, 525, 525]

    @test data.angle[1:5] ≈ [43.0, 37.0, 38.0, 24.0, 20.0]
    @test data.angle[end-3:end] ≈ [1.0, 8.0, 4.0, 11.0]

    @test data.fabric[1:5] == ["A", "A", "A", "A", "A"]
    @test data.fabric[end-3:end] == ["A", "A", "A", "A"]

    # M_index has NaN entries.
    @test isnan(data.M_index[1])
    @test data.M_index[5] ≈ 0.09
    @test data.M_index[end] ≈ 0.26
    @test length(data.study) == 94
end
