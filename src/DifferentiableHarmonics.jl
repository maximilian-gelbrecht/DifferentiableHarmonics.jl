module DifferentiableHarmonics

using DocStringExtensions, FastGaussQuadrature, GSL, Tullio, KernelAbstractions, CUDA

include("abstracttypes.jl")
include("gpu.jl")
include("parameters.jl")
include("utils.jl")
include("r2r_transform.jl")
include("transform.jl")
include("derivatives.jl")

export HarmonicsParameters
export SHtoGaussianGridTransform, GaussianGridtoSHTransform
export Derivative_dλ, GaussianGrid_dμ, Laplacian
export zeros_grid, zeros_SH, rand_grid, rand_SH
export transform_grid, transform_SH

end
