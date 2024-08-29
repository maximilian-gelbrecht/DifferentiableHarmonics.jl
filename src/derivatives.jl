import Base.show 
using Zygote

# this file contains all the code to take derivatives

# derivate functions follow the naming scheme: "Domain1Input"to"Domain2Output"_d"derivativeby"

# there are variants for GPU and CPU, the different grids and 3d and 2d fields

# longitude derivates are computed via SH relation 
# latitude derivatives are computed pseudo-spectral with pre-computed ass. legendre polynomials

abstract type AbstractDerivative{onGPU} end 

abstract type AbstractλDerivative{onGPU} <: AbstractDerivative{onGPU} end

"""
$(TYPEDSIGNATURES)

Required fields: 

* `msinθ`: To change between μ and latitude derivative (-sin(colats))
* `msinθ_3d`: To change between μ and latitude derivative (-sin(colats))

"""
abstract type AbstractμDerivative{onGPU} <: AbstractDerivative{onGPU} end


struct Derivative_dλ{R,S,A4,T,onGPU} <: AbstractλDerivative{onGPU}
    mm::R
    mm_3d::S
    mm_4d::A4
    swap_m_sign_array::T
end

"""
$(TYPEDSIGNATURES)
Pre-computes Derivatives by longitude. Uses the SH relation, is therefore independ from the grid.
"""
function Derivative_dλ(p::HarmonicsParameters{T}; N_batch::Int=0) where {T}

    mm = -(mMatrix(p))
    mm_3d = repeat(mm, 1,1,1)

    swap_m_sign_array = DeviceArray(p.device, [1; Int((p.N_lons)/2)+3 : p.N_lons + 2; 1:Int((p.N_lons)/2)+1;])
     
    if N_batch > 0 
        mm_4d = repeat(mm_3d, 1,1,1,1)
    else 
        mm_4d = nothing
    end

    Derivative_dλ{typeof(mm), typeof(mm_3d), typeof(mm_4d), typeof(swap_m_sign_array), isgpu(p.device)}(mm, mm_3d, mm_4d, swap_m_sign_array)
end

"""
$(TYPEDSIGNATURES)
Derivative of input after φ (polar angle) or λ (longtitude) in SH to Grid
"""
SHtoGrid_dφ(ψ, dλ::Derivative_dλ, sh2g::AbstractSHtoGridTransform) = transform_grid(SHtoSH_dφ(ψ,dλ), sh2g)

"""
$(TYPEDSIGNATURES)
Derivative of input after φ (polar angle) or λ (longtitude) in SH to Grid
"""
SHtoGrid_dλ(ψ, dl, sh2g) = SHtoGrid_dφ(ψ, dl, sh2g)

"""
$(TYPEDSIGNATURES)
Derivative of input after φ (polar angle/longtitude) in SH, output in SH
"""
SHtoSH_dλ(ψ, m) = SHtoSH_dφ(ψ, m)

# 2d field variant
SHtoSH_dφ(ψ::AbstractArray{T,2}, g::Derivative_dλ) where {T} = _SHtoSH_dφ(ψ, g.mm, g.swap_m_sign_array)

# 3d field variant
SHtoSH_dφ(ψ::AbstractArray{T,3}, g::Derivative_dλ) where {T} = _SHtoSH_dφ(ψ, g.mm_3d, g.swap_m_sign_array)

# 4d field variant 
SHtoSH_dφ(ψ::AbstractArray{T,4}, g::Derivative_dλ) where {T} = _SHtoSH_dφ(ψ, g.mm_4d, g.swap_m_sign_array)

_SHtoSH_dφ(ψ::AbstractArray{T,N}, mm::AbstractArray{S,N}, swap_arr) where {T,S,N} = mm .* change_msign(ψ, swap_arr)


"""
$(TYPEDSIGNATURES)
Change the sign of the m in SH. This version returns a view

there is currently a bug or at least missing feature in Zygote, the AD library, that stops views from always working flawlessly when a view is mixed with prior indexing of an array. We need a view for the derivative after φ to change the sign of m, so here is a differentiable variant of the SHtoSH_dφ function for the 2d field
"""
change_msign(A::AbstractArray{T,2}, swap_array::AbstractArray{Int,1}) where {T} = @inbounds view(A,:,swap_array)

# 3d field version
change_msign(A::AbstractArray{T,3}, swap_array::AbstractArray{Int,1}) where {T} = @inbounds view(A,:,swap_array,:)

# 4d field version 
change_msign(A::AbstractArray{T,4}, swap_array::AbstractArray{Int,1}) where {T} = @inbounds view(A,:,swap_array,:,:)

Zygote.@adjoint function change_msign(A::AbstractArray{T,N}, swap_array::AbstractArray{Int,1}) where {T,N}
    return (change_msign(A,swap_array), Δ->(change_msign(Δ,swap_array),nothing))
end

change_msign(A::AbstractArray{T,3}, i::Integer, swap_array::AbstractArray{Int,1}) where T<:Number = @inbounds view(A,:,swap_array,i)

struct GaussianGrid_dμ{onGPU, S<:SHtoGaussianGridTransform, M, A, A4} <: AbstractμDerivative{onGPU}
    t::S
    msinθ::M
    msinθ_3d::A
    msinθ_4d::A4
end

show(io::IO, t::GaussianGrid_dμ{true}) = print(io, "Pre-computed SH to Gaussian Grid Derivative on GPU")
show(io::IO, t::GaussianGrid_dμ{false}) = print(io, "Pre-computed SH to Gaussian Grid Derivative on CPU")

"""
$(TYPEDSIGNATURES)
Pre-computes Pseudo-spectral approach to computing derivatives with repsect to μ = sin(lats). Derivatives are called with following the naming scheme: "Domain1Input"to"Domain2Output"_d"derivativeby"
"""
function GaussianGrid_dμ(p::HarmonicsParameters{T}, N_channels::Int=3, N_batch::Int=0) where T
    dPμdμ, __ = compute_P(p)
    A_real = zeros_grid(p, N_channels)

    if N_batch > 0 
        A_real4d = zeros_grid(p, N_channels, N_batch) 
    else 
        iFT_4d = nothing 
    end  

    FT_2d = plan_r2r_AD(A_real[:,:,1], 2)
    iFT_2d = plan_ir2r_AD(FT_2d*(A_real[:,:,1]), p.N_lons, 2)

    FT_3d = plan_r2r_AD(A_real, 2)
    iFT_3d = plan_ir2r_AD(FT_3d*A_real, p.N_lons, 2)

    if N_batch > 0 
        FT_4d = plan_r2r_AD(A_real4d, 2)
        iFT_4d = plan_ir2r_AD(FT_4d*A_real4d, p.N_lons, 2)
    end 
    
    outputsize = (p.N_lats, p.N_lons)

    msinθ = DeviceArray(p.device, T.(reshape(-sin.(p.θ),p.N_lats, 1)))
    msinθ_3d = repeat(msinθ, 1,1,1)
    
    if N_batch > 0 
        msinθ_4d = repeat(msinθ_3d, 1,1,1,1)
    else 
        msinθ_4d = nothing 
    end

    transform = SHtoGaussianGridTransform{T, typeof(iFT_2d), typeof(iFT_3d), typeof(iFT_4d), typeof(dPμdμ), typeof(outputsize), typeof(p.N_lats), isgpu(p.device)}(iFT_2d, iFT_3d, iFT_4d, dPμdμ, outputsize, p.N_lats, p.N_lons, p.M)
    
    GaussianGrid_dμ{isgpu(p.device), typeof(transform), typeof(msinθ), typeof(msinθ_3d), typeof(msinθ_4d)}(transform, msinθ, msinθ_3d, msinθ_4d)
end

"""
$(TYPEDSIGNATURES)
Derivative of input after μ = sinϕ in SH, uses pre computed SH evaluations
"""
SHtoGrid_dμ(ψ, d::GaussianGrid_dμ) = transform_grid(ψ, d.t)

SHtoSH_dθ(ψ, m) = transform_SH(SHtoGrid_dθ(ψ,m), m)
SHtoSH_dϕ(ψ, m) = eltype(ψ)(-1) .* SHtoSH_dθ(ψ, m)

"""
$(TYPEDSIGNATURES)
Derivative of input after ϕ - latitude in SH, uses pre computed SH evaluations
"""
SHtoGrid_dϕ(ψ, m) = eltype(ψ)(-1) .* SHtoGrid_dθ(ψ, m)
SHtoGrid_dϕ(ψ, d::GaussianGrid_dμ) = eltype(ψ)(-1) .* SHtoGrid_dθ(ψ, d)

"""
$(TYPEDSIGNATURES)
derivative of input after θ (azimutal angle/colatitude) in SH, uses pre computed SH evaluations (dependend on the grid type)
"""
SHtoGrid_dθ(ψ::AbstractArray{T,2}, d::AbstractμDerivative) where {T} = d.msinθ .* SHtoGrid_dμ(ψ, d)
SHtoGrid_dθ(ψ::AbstractArray{T,3}, d::AbstractμDerivative) where {T} = d.msinθ_3d .* SHtoGrid_dμ(ψ, d)
SHtoGrid_dθ(ψ::AbstractArray{T,4}, d::AbstractμDerivative) where {T} = d.msinθ_4d .* SHtoGrid_dμ(ψ, d)

struct Laplacian{T,M<:AbstractArray{T,2},A1<:AbstractArray{T,3},A2<:AbstractArray{T,3},A4,A5,onGPU} <: AbstractDerivative{onGPU}
    Δ::M
    Δ_3d::A1
    Δ_4d::A4
    Δ⁻¹::M
    Δ⁻¹_3d::A2
    Δ⁻¹_4d::A5
end 

"""
$(TYPEDSIGNATURES)
Initializes the `Laplacian` in spherical harmonics and if `init_inverse==true` also its inverse

Apply the Laplacian with the functions (@ref)[`Δ`] and (@ref)[`Δ⁻¹`]
"""
function Laplacian(p::HarmonicsParameters{T}; init_inverse=false, R::T=T(1), N_batch::Int=0, kwargs...) where T
    
    Δ = compute_Δ(p)
    Δ ./= (R*R)
    Δ_3d = repeat(Δ,1,1,1)

    if init_inverse
        Δ⁻¹ = compute_Δ⁻¹(p)
        Δ⁻¹ .*= (R*R)
        Δ⁻¹_3d = repeat(Δ⁻¹,1,1,1)
    else 
        Δ⁻¹ = Array{T,2}(undef,0,0)
        Δ⁻¹_3d = Array{T,3}(undef,0,0,0)
    end 
        
    if N_batch > 0 
        Δ_4d = repeat(Δ_3d, 1,1,1,1)
        Δ⁻¹_4d = repeat(Δ⁻¹_3d, 1,1,1,1)  
    else 
        Δ_4d = nothing
        Δ⁻¹_4d = nothing  
    end

    Laplacian{T, typeof(Δ), typeof(Δ_3d), typeof(Δ⁻¹_3d), typeof(Δ_4d), typeof(Δ⁻¹_4d), isgpu(p.device)}(Δ, Δ_3d, Δ_4d, Δ⁻¹, Δ⁻¹_3d, Δ⁻¹_4d)
end

"""
$(TYPEDSIGNATURES)
Pre-compute the Laplacian in Spherical Harmonics, follows the matrix convention of FastTransforms.jl
"""
function compute_Δ(p::HarmonicsParameters{T}) where T<:Number
    l = T.(lMatrix(p))
    return -l .* (l .+ 1)
end

"""
$(TYPEDSIGNATURES)
Pre-compute the inverse Laplacian in Spherical Harmonics, follows the matrix convention of FastTransforms.jl
"""
function compute_Δ⁻¹(p::HarmonicsParameters{T}) where T<:Number
    Δ⁻¹ = inv.(compute_Δ(p))
    Δ⁻¹[isinf.(Δ⁻¹)] .= T(0) # set integration constant and spurious elements zero 
    return Δ⁻¹
end

"""
$(TYPEDSIGNATURES)
Apply the Laplacian. Also serves to convert regular vorticity (not the quasigeostrophic one) to streamfunction) 
"""
Δ(ψ::AbstractArray{T,4}, L::Laplacian{T}) where T = L.Δ_4d .* ψ
Δ(ψ::AbstractArray{T,3}, L::Laplacian{T}) where T = L.Δ_3d .* ψ
Δ(ψ::AbstractArray{T,2}, L::Laplacian{T}) where T = L.Δ .* ψ

"""
$(TYPEDSIGNATURES)
Apply the inverse Laplacian. Also serves to convert the streamfunction to regular vorticity 
"""
Δ⁻¹(ψ::AbstractArray{T,4}, L::Laplacian{T}) where T = L.Δ⁻¹_4d .* ψ
Δ⁻¹(ψ::AbstractArray{T,3}, L::Laplacian{T}) where T = L.Δ⁻¹_3d .* ψ
Δ⁻¹(ψ::AbstractArray{T,2}, L::Laplacian{T}) where T = L.Δ⁻¹ .* ψ 
