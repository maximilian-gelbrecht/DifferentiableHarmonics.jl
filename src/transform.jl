import Base.show
using Tullio
using GSL
import GSL.sf_legendre_deriv_array_e
import GSL.sf_legendre_deriv_alt_array_e
import GSL.sf_legendre_array_index

"""
    abstract type AbstractSHTransform{onGPU} 

Required fields for all subtypes: 

* outputsize

"""
abstract type AbstractSHTransform{onGPU} end

abstract type AbstractSHtoGridTransform{onGPU} <: AbstractSHTransform{onGPU} end 

abstract type AbstractGridtoSHTransform{onGPU} <: AbstractSHTransform{onGPU} end 


"""
    transform_SH(data::AbstractArray{T,N}, t::GaussianGridtoSHTransform) 

Transforms `data` into the spherical harmonics domain. The coefficents are ordered in a matrix in coloumns of the m value. On CPU the convention of FastTransform.jl is used (0, -1, 1, -2, 2, ...), on GPU the convention (0, 1, 2, 3, ...., (nothing, -1, -2, -3, ...)). Watch out, in future proabaly this might be standardized. 
"""
transform_SH


"""
    transform_grid(data::AbstractArray{T,N}, t::SHtoGaussianGridTransform) 

Transforms `data` from the spherical harmonics domain to a Gaussian Grid. The coefficents are ordered in a matrix in coloumns of the m value. On CPU the convention of FastTransform.jl is used (0, -1, 1, -2, 2, ...), on GPU the convention (0, 1, 2, 3, ...., (nothing, -1, -2, -3, ...)). Watch out, in future proabaly this might be standardized. 

"""
transform_grid 


"""
    GaussianGridtoSHTransform(p::QG3ModelParameters{T}, N_level::Int=3; N_batch::Int=0)

"""
struct GaussianGridtoSHTransform{P,S,T,FT,U,TU,onGPU} <: AbstractGridtoSHTransform{onGPU}
    FT_2d::S
    FT_3d::T
    FT_4d::FT
    Pw::U
    output_size::TU
end

show(io::IO, t::GaussianGridtoSHTransform{P,S,T,FT,U,TU,true}) where {P,S,T,FT,U,TU} = print(io, "Pre-computed Gaussian Grid to SH Transform{",P,"} on GPU")
show(io::IO, t::GaussianGridtoSHTransform{P,S,T,FT,U,TU,false}) where {P,S,T,FT,U,TU} = print(io, "Pre-computed Gaussian Grid to SH Transform{",P,"} on CPU")

"""
($TYPEDSIGNATURES)

Returns transform struct, that can be used with [`transform_SH`](@ref). Transforms Gaussian Grid data to real spherical harmonics coefficients that follow the coefficient logic explained in the main documenation.

## Additional input arguments: 
    
* `N_channels`: defines the transform for `N_channel` horizontal/channel levels. Has to be equal to three for the QG3 model itself, but might be different for other applications. 
* `N_batch`: defines the transforms with an additional batch dimension for ML tasks, if `N_batch==0` this is omitted
"""
function GaussianGridtoSHTransform(p::HarmonicsParameters{T}, N_channels::Int=3, N_batch::Int=0) where {T}
    (; device) = p
    __, P = compute_P(p)
    Pw = compute_LegendreGauss(p, P)
    A_real = rand_grid(p, N_channels) 

    if N_batch > 0 
        A_real4d = rand_grid(p, N_channels, N_batch) 
    else 
        FT_4d = nothing 
    end  

    FT_2d = plan_r2r_AD(A_real[:,:,1], 2)
    FT_3d = plan_r2r_AD(A_real, 2)

    if N_batch > 0 
        FT_4d = plan_r2r_AD(A_real4d, 2)
    end

    outputsize = (p.N_lats, p.N_lons+2)
   
    GaussianGridtoSHTransform{T,typeof(FT_2d),typeof(FT_3d),typeof(FT_4d),typeof(Pw), typeof(outputsize), isgpu(device)}(FT_2d, FT_3d, FT_4d, Pw, outputsize)
end

# 2D version 
function transform_SH(A::AbstractArray{P,2}, t::GaussianGridtoSHTransform{P,S,T,FT,U,V,TU}) where {P,S,T,FT,U,V,TU}
    FTA = t.FT_2d * A

    # truncation is performed in this step as Pw has 0s where the expansion is truncated
    @tullio out[il,im] := t.Pw[i,il,im] * FTA[i,im]
end

# 3D version
function transform_SH(A::AbstractArray{P,3}, t::GaussianGridtoSHTransform{P,S,T,FT,U,V,TU}) where {P,S,T,FT,U,V,TU}
    FTA = t.FT_3d * A

    # truncation is performed in this step as Pw has 0s where the expansion is truncated
    @tullio out[il,im,ilvl] := t.Pw[ilat,il,im] * FTA[ilat,im,ilvl]
end

# 4D version 
function transform_SH(A::AbstractArray{P,4}, t::GaussianGridtoSHTransform{P,S,T,FT,U,V,TU}) where {P<:Number,S,T,FT<:Union{AbstractFFTs.Plan,AbstractDifferentiableR2RPlan},U,V,TU}
    FTA = t.FT_4d * A

    # truncation is performed in this step as Pw has 0s where the expansion is truncated
    @tullio out[il,im,ilvl,ib] := t.Pw[ilat,il,im] * FTA[ilat,im,ilvl,ib]
end

struct SHtoGaussianGridTransform{R,S,T,FT,U,TU,I<:Integer,onGPU} <: AbstractSHtoGridTransform{onGPU}
    iFT_2d::S
    iFT_3d::T
    iFT_4d::FT
    P::U
    output_size::TU
    N_lats::I
    N_lons::I
    M::I
end

show(io::IO, t::SHtoGaussianGridTransform{P,S,T,FT,U,TU,I,true}) where {P,S,T,FT,U,TU,I} = print(io, "Pre-computed SH to Gaussian Grid Transform{",P,"} on GPU")
show(io::IO, t::SHtoGaussianGridTransform{P,S,T,FT,U,TU,I,false}) where {P,S,T,FT,U,TU,I} = print(io, "Pre-computed SH to Gaussian Grid Transform{",P,"} on CPU")

"""
$(TYPEDSIGNATURES)
Returns transform struct, that can be used with [`transform_grid`](@ref). Transforms real spherical harmonics coefficients to Gaussian grid data, follows the coefficient logic explained in the main documenation.

* `N_channel`: defines the transform for `N_channel` horizontal levels. 
* `N_batch`: defines the transforms with an additional batch dimension for ML tasks, if `N_batch==0` this is omitted and only 2D and 3D input will work
"""
function SHtoGaussianGridTransform(p::HarmonicsParameters{T}, N_channels::Int=3, N_batch::Int=0) where T
    (;N_lats, N_lons, device) = p
    __, P = compute_P(p)

    A_real = rand_grid(p, N_channels) 

    if N_batch > 0 
        A_real4d = rand_grid(p, N_channels, N_batch) 
    else 
        iFT_4d = nothing 
    end  

    FT_2d = plan_r2r_AD(A_real[:,:,1], 2)
    iFT_2d = plan_ir2r_AD(FT_2d*(A_real[:,:,1]), N_lons, 2)

    FT_3d = plan_r2r_AD(A_real, 2)
    iFT_3d = plan_ir2r_AD(FT_3d*A_real, N_lons, 2)

    if N_batch > 0 
        FT_4d = plan_r2r_AD(A_real4d, 2)
        iFT_4d = plan_ir2r_AD(FT_4d*A_real4d, N_lons, 2)
    end 
    
    outputsize = (N_lats, N_lons)

    SHtoGaussianGridTransform{T,typeof(iFT_2d),typeof(iFT_3d),typeof(iFT_4d), typeof(P), typeof(outputsize), typeof(p.N_lats), isgpu(device)}(iFT_2d, iFT_3d, iFT_4d, P, outputsize, N_lats, N_lons, p.M)
end

# 2D Version
function transform_grid(A::AbstractArray{P,2}, t::SHtoGaussianGridTransform{P,S,T,FT,U,TU,I}) where {P,S,T,FT,U,TU,I}

    out = batched_vec(t.P,A)
    
    (t.iFT_2d * out) ./ t.N_lons # has to be normalized as this is not done by the FFT
end

# 3D Version
function transform_grid(A::AbstractArray{P,3}, t::SHtoGaussianGridTransform{P,S,T,FT,U,TU,I}) where {P,S,T,FT,U,TU,I}
    @tullio out[ilat, im, lvl] := t.P[ilat, il, im] * A[il, im, lvl]

    (t.iFT_3d * out) ./ t.N_lons # has to be normalized as this is not done by the FFT
end

# 4D Version 
function transform_grid(A::AbstractArray{P,4}, t::SHtoGaussianGridTransform{P,S,T,FT,U,TU,I}) where {P<:Number,S,T,FT<:Union{AbstractFFTs.Plan,AbstractDifferentiableR2RPlan},U,TU,I}
    @tullio out[ilat, im, lvl, ib] := t.P[ilat, il, im] * A[il, im, lvl, ib]

    (t.iFT_4d * out) ./ t.N_lons # has to be normalized as this is not done by the FFT
end

"""
$(TYPEDSIGNATURES)
Pre-compute ass. Legendre Polynomials and dP/dx (derivative of ass. Legendre Polynomial) at the grid points and also the remainder of the Spherical Harmonics at the grid points using GSL

m values are stored 0,1,2,3,4,5,6,7, ...l_max, 0 (nothing),-1, -2, -3, (on GPU)  (the second 0 is the Imanigary part / sin part of the fourier transform which is always identical to zero, it is kept here to have equal matrix sizes)

# so far only |m| is used, as I assume real SPH.
"""
function compute_P(p::HarmonicsParameters{T}; sh_norm=GSL.GSL_SF_LEGENDRE_SPHARM, CSPhase::Integer=-1,prefactor=false) where T<:Number
    (; μ, N_lats, L, M, device) = p 

    N_lats = length(μ)
    P = zeros(T, N_lats, L, M+1) # one more because we have one redudant column
    dPμdμ = zeros(T, N_lats, L, M+1)

    gsl_legendre_index(l,m) = m > l ? error("m > l, not defined") : sf_legendre_array_index(l,m)+1 # +1 because of c indexing vs julia indexing

    # normalization pre-factor for real SPH    
    pre_factor(m) = prefactor ? (m==0 ? T(1) : sqrt(T(2))) : T(1)

    for ilat ∈ 1:N_lats
        temp = sf_legendre_deriv_array_e(sh_norm, L - 1, μ[ilat], CSPhase)

        for m ∈ -(L-1):(L-1)
            for il ∈ 1:(L - abs(m)) # l = abs(m):l_max
                l = il + abs(m) - 1
                if m<0 # the ass. LP are actually the same for m<0 for our application as only |m| is needed, but I do this here in this way to have everything related to SH in the same matrix format
                    P[ilat, il, L+abs(m)+1] = pre_factor(m) * temp[1][gsl_legendre_index(l,abs(m))]
                    dPμdμ[ilat, il, L+abs(m)+1] = pre_factor(m) * temp[2][gsl_legendre_index(l,abs(m))]
                else
                    P[ilat, il, m+1] = pre_factor(m) * temp[1][gsl_legendre_index(l,m)]
                    dPμdμ[ilat, il, m+1] = pre_factor(m) * temp[2][gsl_legendre_index(l,m)]
                end
            end
        end
    end
    return DeviceArray(device, dPμdμ), DeviceArray(device, P)
end

"""
($TYPEDSIGNATURES)
Pre-computes gaussian weights for Legendre Transform, also checks if we really have the correct gaussian latitudes
"""
function compute_GaussWeights(p::HarmonicsParameters{T}, reltol=1e-2) where T<:Number
     nodes, weights = gausslegendre(p.N_lats)
     N_lats2 = Int(p.N_lats/2)
     μ = Array(p.μ)

     # get the order right, nodes is counting up
     nodes = μ[1] > 0 ? reverse(nodes) : nodes
     weights =  μ[1] > 0 ? reverse(weights) : weights

     # check if the Gaussian latitudes are correct
     check = (nodes[1:N_lats2] .- (reltol .* nodes[1:N_lats2])) .<= μ[1:N_lats2] .<= (nodes[1:N_lats2] .+ (reltol .* nodes[1:N_lats2]))

     if sum(check)!=N_lats2
         error("Gaussian Latitudes not set currently")
     end

     T.(weights)
end

"""
($TYPEDSIGNATURES)
"""
function compute_LegendreGauss(p::HarmonicsParameters{T}, P::AbstractArray{T,3}, w::AbstractArray{T,1}) where T<:Number
    # P in format lat x L x M
    for i=1:p.N_lats
        P[i,:,:] *= (2π*w[i]) # 4π from integral norm 
    end
    P
end

"""
($TYPEDSIGNATURES)
"""
function compute_LegendreGauss(p::HarmonicsParameters{T}, P::AbstractArray{T,3}; reltol::Number=1e-2) where T<:Number
    w = compute_GaussWeights(p, reltol)
    return compute_LegendreGauss(p, P, w)
end