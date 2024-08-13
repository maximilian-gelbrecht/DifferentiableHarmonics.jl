using DifferentiableHarmonics
using Test

@testset "DifferentiableHarmonics.jl" begin
    include("r2r_transform.jl")
    include("transforms.jl")
end
