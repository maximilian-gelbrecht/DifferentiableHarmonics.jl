using DifferentiableHarmonics
using Test, Zygote

@testset "DifferentiableHarmonics.jl" begin
    include("r2r_transform.jl")
    include("transforms.jl")
    include("derivatives.jl")
end
