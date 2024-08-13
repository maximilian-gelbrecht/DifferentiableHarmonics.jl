using CUDA 

abstract type AbstractDevice end 

struct CUDADevice <: AbstractDevice
end 

struct CPUDevice <: AbstractDevice
end 

get_device(GPU::Bool) = GPU ? CUDADevice() : CPUDevice()

DeviceArray(dev::CUDADevice, arr::AbstractArray) = CuArray(arr)
DeviceArray(dev::CPUDevice, arr::AbstractArray) = Array(arr)

isgpu(dev::CUDADevice) = true 
isgpu(dev::CPUDevice) = false 

iscpu(dev::CUDADevice) = false 
iscpu(dev::CPUDevice) = true