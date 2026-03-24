using Documenter
using DRex

makedocs(;
    modules  = [DRex],
    sitename = "DRex.jl",
    authors  = "Boris Kaus and contributors",
    repo     = Remotes.GitHub("JuliaGeodynamics", "DRex.jl"),
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://JuliaGeodynamics.github.io/DRex.jl",
        edit_link  = "main",
        assets     = String[],
    ),
    pages = [
        "Home"                  => "index.md",
        "Installation"          => "installation.md",
        "Quick Start"           => "quickstart.md",
        "Examples"              => "examples.md",
        "GPU & Multithreading"  => "gpu.md",
        "LaMEM Integration"     => "lamem.md",
        "API Reference"         => "api.md",
    ],
    warnonly = true,
)

deploydocs(;
    repo   = "github.com/JuliaGeodynamics/DRex.jl",
    devbranch = "main",
)
