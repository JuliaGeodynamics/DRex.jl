# SCSV file I/O — CSV files with a YAML schema header.

"""SCSV format error."""
struct SCSVError <: Exception
    msg::String
end
Base.showerror(io::IO, e::SCSVError) = print(io, "SCSVError: ", e.msg)

const SCSV_TYPEMAP = Dict(
    "string"  => String,
    "integer" => Int,
    "float"   => Float64,
    "boolean" => Bool,
    "complex" => ComplexF64,
)

const _SCSV_DEFAULT_TYPE = "string"
const _SCSV_DEFAULT_FILL = ""

# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

const _DATA_DIR = normpath(joinpath(@__DIR__, "..", "data"))

"""
    scsv_data(directory)

Return resolved path to a data subdirectory (e.g. `"specs"`, `"thirdparty"`).
"""
function scsv_data(directory::AbstractString)
    path = joinpath(_DATA_DIR, directory)
    isdir(path) || throw(ErrorException("$path is not a directory"))
    return path
end

# ─────────────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────────────

function _is_identifier(s::AbstractString)
    m = match(r"^[A-Za-z_][A-Za-z0-9_]*$", s)
    return m !== nothing
end

function _validate_scsv_schema(schema::Dict)
    haskey(schema, "delimiter") || throw(SCSVError("schema missing 'delimiter'"))
    haskey(schema, "missing")   || throw(SCSVError("schema missing 'missing'"))
    haskey(schema, "fields")    || throw(SCSVError("schema missing 'fields'"))
    length(schema["fields"]) > 0 || throw(SCSVError("schema must have at least one field"))
    delim = schema["delimiter"]
    miss  = schema["missing"]
    length(delim) == 1 || throw(SCSVError("delimiter must be a single character, got '$delim'"))
    delim == miss && throw(SCSVError("delimiter and missing string must differ"))
    occursin(delim, miss) && throw(SCSVError("delimiter must not appear in missing string"))
    for field in schema["fields"]
        name = field["name"]
        _is_identifier(name) || throw(SCSVError("field name '$name' is not a valid identifier"))
        ftype = get(field, "type", _SCSV_DEFAULT_TYPE)
        haskey(SCSV_TYPEMAP, ftype) || throw(SCSVError("unsupported field type '$ftype'"))
        if ftype ∉ (_SCSV_DEFAULT_TYPE, "boolean") && !haskey(field, "fill")
            throw(SCSVError("field '$name' of type '$ftype' requires a fill value"))
        end
    end
    return true
end

# ─────────────────────────────────────────────────────────────────────────────
# YAML header parser (handles the SCSV-specific subset)
# ─────────────────────────────────────────────────────────────────────────────

"""Strip inline YAML comment (not inside quotes)."""
function _strip_yaml_comment(line::AbstractString)
    in_single = false
    in_double = false
    for (i, c) in enumerate(line)
        if c == '\'' && !in_double
            in_single = !in_single
        elseif c == '"' && !in_single
            in_double = !in_double
        elseif c == '#' && !in_single && !in_double
            return rstrip(line[1:i-1])
        end
    end
    return line
end

"""Unquote a YAML value string."""
function _unquote(s::AbstractString)
    s = strip(s)
    if (startswith(s, "'") && endswith(s, "'")) ||
       (startswith(s, "\"") && endswith(s, "\""))
        return s[2:end-1]
    end
    return s
end

function _parse_scsv_yaml(yaml_text::AbstractString)
    lines = split(yaml_text, '\n')
    schema = Dict{String,Any}()
    fields = Dict{String,Any}[]
    current_field = nothing

    for raw_line in lines
        line = _strip_yaml_comment(String(raw_line))
        stripped = strip(line)
        isempty(stripped) && continue

        # Top-level "schema:" — skip
        if stripped == "schema:"
            continue
        end

        # Determine indentation level
        indent = length(line) - length(lstrip(line))

        if indent <= 4 && startswith(stripped, "delimiter:")
            schema["delimiter"] = _unquote(strip(split(stripped, ":", limit=2)[2]))
        elseif indent <= 4 && startswith(stripped, "missing:")
            schema["missing"] = _unquote(strip(split(stripped, ":", limit=2)[2]))
        elseif stripped == "fields:"
            continue
        elseif startswith(stripped, "- name:")
            # New field
            current_field = Dict{String,Any}()
            current_field["name"] = _unquote(strip(split(stripped, ":", limit=2)[2]))
            push!(fields, current_field)
        elseif current_field !== nothing && startswith(stripped, "type:")
            current_field["type"] = strip(split(stripped, ":", limit=2)[2])
        elseif current_field !== nothing && startswith(stripped, "fill:")
            current_field["fill"] = _unquote(strip(split(stripped, ":", limit=2)[2]))
        elseif current_field !== nothing && startswith(stripped, "unit:")
            current_field["unit"] = _unquote(strip(split(stripped, ":", limit=2)[2]))
        elseif current_field !== nothing && startswith(stripped, "- name:")
            # Shouldn't hit this due to earlier branch, but just in case
        end
    end

    schema["fields"] = fields
    return schema
end

# ─────────────────────────────────────────────────────────────────────────────
# CSV line parser (handles double-quoted fields)
# ─────────────────────────────────────────────────────────────────────────────

function _parse_csv_line(line::AbstractString, delimiter::Char)
    fields = String[]
    current = IOBuffer()
    in_quotes = false
    i = 1
    chars = collect(line)
    n = length(chars)
    while i <= n
        c = chars[i]
        if in_quotes
            if c == '"'
                if i < n && chars[i+1] == '"'
                    write(current, '"')
                    i += 1
                else
                    in_quotes = false
                end
            else
                write(current, c)
            end
        else
            if c == '"'
                in_quotes = true
            elseif c == delimiter
                push!(fields, String(take!(current)))
                current = IOBuffer()
            else
                write(current, c)
            end
        end
        i += 1
    end
    push!(fields, String(take!(current)))
    return fields
end

# ─────────────────────────────────────────────────────────────────────────────
# Cell parsing
# ─────────────────────────────────────────────────────────────────────────────

function _parse_scsv_bool(x::AbstractString)
    return lowercase(strip(x)) in ("yes", "true", "t", "1")
end

function _parse_scsv_cell(T::Type, data::AbstractString, missingstr::AbstractString, fillval::AbstractString)
    s = strip(data)
    if s == missingstr
        if fillval == "NaN"
            if T == Float64
                return NaN
            elseif T == ComplexF64
                return complex(NaN, 0.0)
            end
        end
        if T == String
            return isempty(fillval) ? "" : String(fillval)
        elseif T == Int
            return parse(Int, fillval)
        elseif T == Float64
            return parse(Float64, fillval)
        elseif T == Bool
            return _parse_scsv_bool(fillval)
        elseif T == ComplexF64
            return parse(ComplexF64, fillval)
        end
    end
    if T == Bool
        return _parse_scsv_bool(s)
    elseif T == String
        return s
    elseif T == Int
        return parse(Int, s)
    elseif T == Float64
        return parse(Float64, s)
    elseif T == ComplexF64
        # Handle "1+j", "0.1", "1+1j", etc.
        return _parse_complex(s)
    end
    throw(SCSVError("cannot parse '$s' as $T"))
end

function _parse_complex(s::AbstractString)
    s = strip(s)
    # Replace 'j' with 'im' for Julia, handling bare 'j' as '1im'
    s = replace(s, "j" => "im")
    # Handle cases like "1+im" → "1+1im", "-im" → "-1im"
    s = replace(s, "+im" => "+1im")
    s = replace(s, "-im" => "-1im")
    if s == "im"
        s = "1im"
    end
    val = eval(Meta.parse(s))
    return ComplexF64(val)
end

# ─────────────────────────────────────────────────────────────────────────────
# read_scsv
# ─────────────────────────────────────────────────────────────────────────────

"""
    read_scsv(filepath) -> NamedTuple

Read an SCSV file and return a NamedTuple whose field names match the column names.
Each field is a Vector of the appropriate type.
"""
function read_scsv(filepath::AbstractString)
    text = read(filepath, String)

    # Split into YAML and CSV sections on "---" delimiters.
    yaml_parts = String[]
    csv_lines = String[]
    in_yaml = false
    for line in split(text, '\n')
        sline = strip(line)
        if sline == "---"
            if in_yaml
                in_yaml = false
                continue
            else
                in_yaml = true
                continue
            end
        end
        if in_yaml
            push!(yaml_parts, String(line))
        else
            if !isempty(sline)
                push!(csv_lines, String(line))
            end
        end
    end

    schema = _parse_scsv_yaml(join(yaml_parts, '\n'))
    _validate_scsv_schema(schema)

    delim = schema["delimiter"][1]  # single char
    miss  = schema["missing"]

    field_defs = schema["fields"]
    col_names  = [f["name"] for f in field_defs]
    col_types  = [SCSV_TYPEMAP[get(f, "type", _SCSV_DEFAULT_TYPE)] for f in field_defs]
    col_fills  = [get(f, "fill", _SCSV_DEFAULT_FILL) for f in field_defs]

    # First CSV line is the header.
    isempty(csv_lines) && throw(SCSVError("no CSV data lines in file"))
    header = [strip(h) for h in _parse_csv_line(csv_lines[1], delim)]
    header == col_names || throw(SCSVError(
        "schema field names must match column headers. Schema: $col_names, Header: $header"
    ))

    # Parse data rows.
    n_cols = length(col_names)
    columns = [Vector{col_types[j]}() for j in 1:n_cols]

    for row_line in csv_lines[2:end]
        cells = _parse_csv_line(row_line, delim)
        length(cells) == n_cols || throw(SCSVError(
            "row has $(length(cells)) fields, expected $n_cols"
        ))
        for j in 1:n_cols
            val = _parse_scsv_cell(col_types[j], cells[j], miss, col_fills[j])
            push!(columns[j], val)
        end
    end

    # Build NamedTuple dynamically.
    syms = Tuple(Symbol.(col_names))
    return NamedTuple{syms}(Tuple(columns))
end

# ─────────────────────────────────────────────────────────────────────────────
# save_scsv
# ─────────────────────────────────────────────────────────────────────────────

"""
    write_scsv_header(io, schema; comments=nothing)

Write the YAML header of an SCSV file to `io`.
"""
function write_scsv_header(out::IO, schema::Dict; comments=nothing)
    _validate_scsv_schema(schema)
    println(out, "---")
    if comments !== nothing
        for c in comments
            println(out, "# ", c)
        end
    end
    println(out, "schema:")
    println(out, "  delimiter: '", schema["delimiter"], "'")
    println(out, "  missing: '", schema["missing"], "'")
    println(out, "  fields:")
    for field in schema["fields"]
        println(out, "    - name: ", field["name"])
        println(out, "      type: ", get(field, "type", _SCSV_DEFAULT_TYPE))
        if haskey(field, "unit")
            println(out, "      unit: ", field["unit"])
        end
        if haskey(field, "fill")
            println(out, "      fill: ", field["fill"])
        end
    end
    println(out, "---")
end

"""
    save_scsv(filepath, schema, data; comments=nothing)

Write data columns to an SCSV file.

- `schema` — Dict with keys `"delimiter"`, `"missing"`, `"fields"`
- `data` — Vector of column vectors, one per field
"""
function save_scsv(filepath::AbstractString, schema::Dict, data::AbstractVector;
                   comments=nothing)
    _validate_scsv_schema(schema)

    fields = schema["fields"]
    length(data) == length(fields) || throw(SCSVError(
        "number of data columns ($(length(data))) does not match schema fields ($(length(fields)))"
    ))
    n_rows = length(data[1])
    for col in data[2:end]
        length(col) == n_rows || throw(SCSVError(
            "data columns have unequal lengths"
        ))
    end

    delim = schema["delimiter"]
    miss  = schema["missing"]
    col_types = [SCSV_TYPEMAP[get(f, "type", _SCSV_DEFAULT_TYPE)] for f in fields]
    col_fills = [get(f, "fill", _SCSV_DEFAULT_FILL) for f in fields]
    col_names = [f["name"] for f in fields]

    # Validate data types before writing.
    for (i, (col, T, name)) in enumerate(zip(data, col_types, col_names))
        for (r, val) in enumerate(col)
            if T == Int
                val isa Integer || throw(SCSVError(
                    "invalid data for column '$name': cannot parse $val as Int"
                ))
            elseif T == Float64
                val isa Real || throw(SCSVError(
                    "invalid data for column '$name': cannot parse $val as Float64"
                ))
            elseif T == Bool
                val isa Bool || throw(SCSVError(
                    "invalid data for column '$name': cannot parse $val as Bool"
                ))
            elseif T == ComplexF64
                val isa Number || throw(SCSVError(
                    "invalid data for column '$name': cannot parse $val as Complex"
                ))
            end
        end
    end

    open(filepath, "w") do io
        write_scsv_header(io, schema; comments=comments)
        println(io, join(col_names, delim * " "))
        for r in 1:n_rows
            cells = String[]
            for (j, (T, fill)) in enumerate(zip(col_types, col_fills))
                val = data[j][r]
                if _is_fill_value(val, T, fill, miss)
                    push!(cells, miss)
                else
                    push!(cells, _format_cell(val, T, delim))
                end
            end
            println(io, join(cells, delim * " "))
        end
    end
end

function _is_fill_value(val, T, fill, miss)
    if T in (Float64, ComplexF64)
        if fill == "NaN"
            if val isa AbstractFloat && isnan(val)
                return true
            elseif val isa Complex && isnan(real(val))
                return true
            end
        end
        fillval = fill == "NaN" ? NaN : parse(Float64, string(fill))
        if val isa Real && !isnan(val) && val == fillval
            return true
        end
    elseif T == Int
        if val isa Integer && val == parse(Int, string(fill))
            return true
        end
    elseif T == String
        fillstr = string(fill)
        if string(val) == fillstr
            return true
        end
    end
    return false
end

function _format_cell(val, T, delim)
    if T == Bool
        return val ? "True" : "False"
    elseif T == ComplexF64
        v = ComplexF64(val)
        r, im = real(v), imag(v)
        if im == 0
            return string(r)
        elseif r == 0
            return string(im) * "j"
        elseif im > 0
            return string(r) * "+" * string(im) * "j"
        else
            return string(r) * string(im) * "j"
        end
    elseif T == String
        s = string(val)
        if occursin(delim, s)
            return "\"" * s * "\""
        end
        return s
    else
        return string(val)
    end
end
