module DifferentiableHarmonics

# Write your package code here.

using DocStringExtensions, FastGaussQuadrature, GSL, Tullio, KernelAbstractions, CUDA

include("gpu.jl")
include("parameters.jl")
include("r2r_transform.jl")
include("transform.jl")
include("utils.jl")

export HarmonicsParameters
export SHtoGaussianGridTransform, GaussianGridtoSHTransform
export Derivative_dλ, GaussianGrid_dμ, Laplacian
export zeros_grid, zeros_SH, rand_grid, rand_SH
export transform_grid, transform_SH

end
