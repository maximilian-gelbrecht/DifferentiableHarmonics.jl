lat_to_colat(lat::T) where {T} = T(π/2) - lat

"""
($TYPEDSIGNATURES)

Returns a matrix with the SPH modes (l, m) arranged as they are in all SH domain matrices in DifferentiableHarmonics.jl
"""
function sph_modes(p::HarmonicsParameters)
    (; size_SH) = p
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
    (; size_SH, L, M, L_max, neg_m_offset, device) = p
    m = zeros(Int, size_SH...)

    for (i_m, m_i) in enumerate(0:L_max) # positive m 
        m[1:L-m_i, i_m] .= m_i
    end 

    for (i_m, m_i) in enumerate(-1:-1:(-L_max)) # negative m
        m[1:L-abs(m_i), i_m + neg_m_offset] .= m_i
    end 

    return DeviceArray(device, m)
end 

"""
($TYPEDSIGNATURES)

Returns the order numbers l of the SPH coefficents. 
"""
function lMatrix(p::HarmonicsParameters)
    (; size_SH, L, M, L_max, neg_m_offset, device) = p
    l = zeros(Int, size_SH...)

    for m ∈ (-L_max):L_max
        im = m<0 ? abs(m)+neg_m_offset : m + 1
        l[1:L-abs(m),im] = abs(m):(L-1)
    end

    return DeviceArray(device, l)
end 

"""
($TYPEDSIGNATURES)
Returns a Boolean array with all entries that represent SPH coefficents `true`
"""
function SH_mask(p::HarmonicsParameters)
    mask = (lMatrix(p) .!= 0)
    mask[1,1] = true 
    return mask 
end 

"""
($TYPEDSIGNATURES)
Returns a Boolean array with all entries that don't represent SPH coefficents `true`
"""
function no_SH_mask(p::HarmonicsParameters)
    mask = (lMatrix(p) .== 0)
    mask[1,1] = false 
    return mask 
end 

"""
($TYPEDSIGNATURES)

Generate a zero array with the correct shape to represend the SH domain and on the correct device and eltype. Additional input arguments are appended after the first two leading dimensions
"""
function zeros_SH(p::HarmonicsParameters{T}, size...) where T
    return isgpu(p.device) ? CUDA.zeros(T, p.size_SH..., size...) : zeros(T, p.size_SH..., size...) 
end 

"""
($TYPEDSIGNATURES)

Generate a random array with the correct shape to represend the SH domain and on the correct device and eltype. Additional input arguments are appended after the first two leading dimensions
"""
function rand_SH(p::HarmonicsParameters{T}, size...) where T
    out = isgpu(p.device) ? CUDA.rand(T, p.size_SH..., size...) : rand(T, p.size_SH..., size...)
    out[no_SH_mask(p), [Colon() for i=1:length(size)]...] .= zero(T)
    return out
end 

"""
($TYPEDSIGNATURES)

Generate a zero array with the correct shape to represend the grid domain and on the correct device and eltype. Additional input arguments are appended after the first two leading dimensions
"""
function zeros_grid(p::HarmonicsParameters{T}, size...) where T
    return isgpu(p.device) ? CUDA.zeros(T, p.size_grid..., size...) : zeros(T, p.size_grid..., size...)
end 

"""
($TYPEDSIGNATURES)

Generate a zero array with the correct shape to represend the grid domain and on the correct device and eltype. Additional input arguments are appended after the first two leading dimensions
"""
function rand_grid(p::HarmonicsParameters{T}, size...) where T
    return isgpu(p.device) ? CUDA.rand(T, p.size_grid..., size...) : rand(T, p.size_grid..., size...)
end 