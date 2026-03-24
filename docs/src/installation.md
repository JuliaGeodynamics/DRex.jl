# Installation

DRex.jl is hosted on GitHub and not yet registered in the Julia General registry. Install it directly from the repository:

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaGeodynamics/DRex.jl")
```

Or from the package REPL (press `]`):

```
] add https://github.com/JuliaGeodynamics/DRex.jl
```

For local development (clones the repo and uses it in development mode):

```julia
using Pkg
Pkg.develop(url="https://github.com/JuliaGeodynamics/DRex.jl")
```

## Dependencies

The core package requires:

| Package | Purpose |
|---|---|
| `StaticArrays.jl` | Allocation-free inner-loop grain computations |
| `OrdinaryDiffEq.jl` | Adaptive ODE integration (Tsit5) for CPO evolution |
| `Rotations.jl` | Rotation utilities |
| `KernelAbstractions.jl` | GPU-agnostic kernel programming |
| `LinearAlgebra` (stdlib) | |
| `Random` (stdlib) | |

## Optional extensions

The **LaMEM extension** auto-loads when three additional packages are present in your environment:

```julia
using LaMEM, GeophysicalModelGenerator, WriteVTK
using DRex   # DRexLaMEMExt loads automatically
```

The **Metal GPU backend** requires Apple Silicon and the `Metal.jl` package:

```julia
using Metal
using DRex
```

## Running the tests

```julia
using Pkg
Pkg.test("DRex")
```
