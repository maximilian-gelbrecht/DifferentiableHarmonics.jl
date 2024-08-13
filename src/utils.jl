lat_to_colat(lat::T) where {T} = T(π/2) - lat

"""
($TYPEDSIGNATURES)

Returns a matrix with the SPH modes (l, m) arranged as they are in all SH domain matrices in DifferentiableHarmonics.jl
"""
function sph_modes(p::HarmonicsParameters)
    (; size_SH, L, M, L_max) = p
    modes = Array{NTuple{2},2}(undef, size_SH...)

    l = lMatrix(p)
    m = mMatrix(p)

    for i ∈ eachindex(modes)
        modes[i] = (l[i],m[i])
    end 
    
    return modes 
end 

"""
($TYPEDSIGNATURES)

Returns the order numbers m of the SPH coefficents. 
"""
function mMatrix(p::HarmonicsParameters)
    (; size_SH, L, M, L_max) = p
    m = Array{Int,2}(undef, size_SH...)

    return m
end 

"""
($TYPEDSIGNATURES)

Returns the order numbers l of the SPH coefficents. 
"""
function lMatrix(p::HarmonicsParameters)
    (; size_SH, L, M, L_max) = p
    l = Array{Int,2}(undef, size_SH...)

    return l
end 

"""
($TYPEDSIGNATURES)

Generate a zero array with the correct shape to represend the SH domain and on the correct device and eltype. Additional input arguments are appended after the first two leading dimensions
"""
function zeros_SH(p::HarmonicsParameters{T}, size...) where T
    return DeviceArray(p.device, zeros(T, p.size_SH..., size...))
end 

"""
($TYPEDSIGNATURES)

Generate a random array with the correct shape to represend the SH domain and on the correct device and eltype. Additional input arguments are appended after the first two leading dimensions
"""
function rand_SH(p::HarmonicsParameters{T}, size...) where T
    return DeviceArray(p.device, rand(T, p.size_SH..., size...))
end 

"""
($TYPEDSIGNATURES)

Generate a zero array with the correct shape to represend the grid domain and on the correct device and eltype. Additional input arguments are appended after the first two leading dimensions
"""
function zeros_grid(p::HarmonicsParameters{T}, size...) where T
    return DeviceArray(p.device, zeros(T, p.size_grid..., size...))
end 

"""
($TYPEDSIGNATURES)

Generate a zero array with the correct shape to represend the grid domain and on the correct device and eltype. Additional input arguments are appended after the first two leading dimensions
"""
function rand_grid(p::HarmonicsParameters{T}, size...) where T
    return DeviceArray(p.device, rand(T, p.size_grid..., size...))
end 