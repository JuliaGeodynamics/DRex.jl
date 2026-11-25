using Documenter
using MantleFabric

makedocs(;
    modules  = [MantleFabric],
    sitename = "MantleFabric.jl",
    authors  = "Boris Kaus and contributors",
    repo     = Remotes.GitHub("JuliaGeodynamics", "MantleFabric.jl"),
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://JuliaGeodynamics.github.io/MantleFabric.jl",
        edit_link  = "main",
        assets     = String[],
    ),
    pages = [
        "Home"                  => "index.md",
        "Installation"          => "installation.md",
        "Quick Start"           => "quickstart.md",
        "D-Rex and ODFTEX"      => "methods.md",
        "Examples"              => "examples.md",
        "GPU & Multithreading"  => "gpu.md",
        "LaMEM Integration"     => "lamem.md",
        "API Reference"         => "api.md",
    ],
    warnonly = true,
)

deploydocs(;
    repo   = "github.com/JuliaGeodynamics/MantleFabric.jl",
    devbranch = "main",
)
