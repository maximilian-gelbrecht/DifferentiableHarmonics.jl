"""
$(TYPEDSIGNATURES)
Holds all parameters for the spherical harmonics. `L` and `M` are the number of harmonics numbers. `L_max` is the largest value of `l`.
"""
struct HarmonicsParameters{T,D,TU}
    L::Int
    M::Int 
    L_max::Int
    N_lats::Int
    N_lons::Int
    lats::AbstractVector{T}
    lons::AbstractVector{T}
    θ::AbstractVector{T} # colatitudes 
    μ::AbstractVector{T} # sin(lats) == cos(colats)
    size_SH::TU # size of the Array in SH domain 
    size_grid::TU # size of the Array in grid domain
    neg_m_offset::Int # offset where in the SH array the negative m start = N_lons // 2 + 1
    device::D
end 

"""
$(TYPEDSIGNATURES)
Initializes the [`HarmonicsParameters`](@ref) based on a standard triangular truncation and the grid handed over. 
"""
function HarmonicsParameters(Lmax::Integer, lats::AbstractArray{T,1}, lons::AbstractArray{T,1}; GPU::Bool=false) where T
    dev = get_device(GPU)

    L = Lmax + 1 
    M = 2*L - 1

    lats = DeviceArray(dev, lats)
    lons = DeviceArray(dev, lons)

    N_lats = size(lats,1)

    M = 2*L - 1
    N_lons = size(lons,1)

    colats = lat_to_colat.(lats)
    μ = sin.(lats)

    size_SH = (p.N_lats, p.N_lons+2)
    
    HarmonicsParameters{eltype{lats},typeof(dev),typeof(size_SH)}(L, M, Lmax, N_lats, N_lons, lats, lons, colats, μ, (p.N_lats, p.N_lons+2), (p.N_lats, p.N_lons),  N_lons // 2 + 1, dev)
end 

"""
$(TYPEDSIGNATURES)
Initializes the [`HarmonicsParameters`](@ref) based on a standard triangular truncation, `:quadratic` or `:cubic` antialiasing and Gaussian latitudes. 
"""
function HarmonicsParameters(Lmax::Integer, antialiasing::Symbol=:quadratic; eltype=Float32, GPU::Bool=false) 
    @assert antialiasing in [:linear, :quadratic, :cubic]
    
    L = Lmax + 1 
    M = 2*L - 1

    if antialiasing == :quadratic
        N_lats = Int(floor(3/2*L))
        N_lats = iseven(N_lats) ? N_lats : N_lats - 1
    elseif antialiasing == :cubic 
        N_lats = 2*L
    else # linear 
        N_lats = L
        N_lats = iseven(N_lats) ? N_lats : N_lats - 1
    end 


    # compute Gaussian latitudes 
    gaussian_nodes, __ = gausslegendre(N_lats) # TODO
    μ = -gaussian_nodes 
    lats = asin.(μ)

    # compute Longittudes 
    N_lons = 2*N_lats
    lons = eltype.(range(0, length=N_lons, step=2π/N_lons))
    
    dev = get_device(GPU)
    lats = DeviceArray(dev, lats)
    lons = DeviceArray(dev, lons)
    μ = DeviceArray(dev, μ)

    colats = lat_to_colat.(lats)
    size_SH = (N_lats, N_lons+2)

    HarmonicsParameters{eltype,typeof(dev),typeof(size_SH)}(L, M, Lmax, N_lats, N_lons, lats, lons, colats, μ, size_SH, (N_lats, N_lons), N_lons // 2 + 1, dev)
end 

