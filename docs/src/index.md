```@meta
CurrentModule = DifferentiableHarmonics
```

# DifferentiableHarmonics.jl

This package is spun of [QG3.jl](https://github.com/maximilian-gelbrecht/QG3.jl) to give a stand-alone implementation of reverse-mode differentiable spherical harmonics transforms and derivative operators that work on CUDA GPUs and CPU. All implementations are checked again finite difference implenations and the transforms of SpeedyWeather.jl.

# How-to 

Each of the transforms and derivatives needs to be pre-planned. The common parameters of these plans are hold in [`HarmonicsParameters`](@ref). First, we can plan transforms for a quadratic FullGaussianGrid with truncation at the 21st wave number

```julia 
L_max = 21
p = HarmonicsParameters(L_max, GPU=false)
```

Then, we generate some random fields in both the spectral and grid domain:: 
```julia 
A = rand_grid(p)
B = rand_SH(p)

# batched matrices with trailing dimensions can be declared by:
A_b = rand_grid(p, N_channels, N_batch) # N_lat x N_lon x N_channels x N_batch
B_b = rand_SH(p,  N_channels, N_batch)
```

Now we can preplan all derivatives and transforms:
```julia 
dλ = Derivative_dλ(p, N_batch=N_batch)
dμ = GaussianGrid_dμ(p, N_channels, N_batch)
L = Laplacian(p, init_inverse=true, N_batch=N_batch)
analysis_plan = SHtoGaussianGridTransform(p, N_channels, N_batch)
synthesis_plan = GaussianGridtoSHTransform(p, N_channels, N_batch)
```

And apply to plans like this:
```julia
transform_grid(B_b, analysis_plan) # SH -> grid 
transform_grid(A_b, synthesis_plan) # grid -> SH 
Δ(B_b, L) # SH -> Laplacian(SH)
SHtoGrid_dθ(B_b, dμ) # SH -> d B / dθ   colatitude
SHtoGrid_dμ(B_b, dμ) # SH -> d B / dμ   (μ = sin(lat))
SHtoGrid_dλ(B_b, dλ) # SH -> d B / dλ   longitutde
```

# Harmonics Coefficients

The harmonics coefficents are saved in the following format: 

```julia 
show(sph_modes(p))
```

# Work in Progress

In the future, this will be replaced by SpeedyWeather.jl's SpeedyTransform, but so far it is not differentiable yet. 

```@index
```

```@autodocs
Modules = [DifferentiableHarmonics]
```
