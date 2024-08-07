using DifferentiableHarmonics
using Documenter

DocMeta.setdocmeta!(DifferentiableHarmonics, :DocTestSetup, :(using DifferentiableHarmonics); recursive=true)

makedocs(;
    modules=[DifferentiableHarmonics],
    authors="Maximilian Gelbrecht <maximilian.gelbrecht@posteo.de> and contributors",
    sitename="DifferentiableHarmonics.jl",
    format=Documenter.HTML(;
        canonical="https://maximilian-gelbrecht.github.io/DifferentiableHarmonics.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/maximilian-gelbrecht/DifferentiableHarmonics.jl",
    devbranch="main",
)
