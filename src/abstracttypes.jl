"""
    abstract type AbstractSHTransform{onGPU} 

Required fields for all subtypes: 

* outputsize

"""
abstract type AbstractSHTransform{onGPU} end

abstract type AbstractSHtoGridTransform{onGPU} <: AbstractSHTransform{onGPU} end 

abstract type AbstractGridtoSHTransform{onGPU} <: AbstractSHTransform{onGPU} end 
