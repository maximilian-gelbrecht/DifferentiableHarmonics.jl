using SpeedyWeather.RingGrids
using SpeedyWeather.LowerTriangularMatrices
using SpeedyWeather.SpeedyTransforms
using SpeedyWeather
using StatsBase

# this is a very basic to test if the transform and derivite work sort of correct, it just checks if it can correctly transform and take derivatives of cosθ
@testset "Transforms" begin
    
    # load forcing and model parameters
    L_max = 21
    p = HarmonicsParameters(L_max)

    synthesis_plan = GaussianGridtoSHTransform(p, 3, 2)
    analysis_plan = SHtoGaussianGridTransform(p, 3, 2)
    
    A = DifferentiableHarmonics.rand_grid(p, 3, 2)
    B = DifferentiableHarmonics.rand_SH(p, 3, 2)
    
    # test that the transform is unitary ()
    @test maximum(abs.(transform_SH(transform_grid(B, analysis_plan),synthesis_plan) - B)) < 1e-2
    @test maximum(abs.(transform_grid(transform_SH(A, synthesis_plan),analysis_plan) - A)) < 1e-2
    @test isapprox(transform_SH(transform_grid(B, analysis_plan),synthesis_plan),B,rtol=1e-3)

    cosθ = zeros_grid(p) 
    msinθ = zeros_grid(p)
    for i ∈ 1:p.N_lats
        cosθ[i,:] .= cos(p.θ[i])
        msinθ[i,:] .= -sin(p.θ[i])
    end

    # 2D transforms
    cSH_2d = transform_SH(cosθ, synthesis_plan)
    @test sum(abs.(cSH_2d) .> 1) == 1
    @test abs.(cSH_2d[2,1]) > 20.
    
    cg = transform_grid(cSH_2d, analysis_plan)
    @test mean(abs.(cg - cosθ) ./ abs.(cg)) < 1e-3
    
    # 3D transforms
    cosθ = repeat(cosθ, 1, 1, 3)
    msinθ = repeat(msinθ, 1, 1, 3)
    
    cSH = transform_SH(cosθ, synthesis_plan)
    
    @test mean(abs.(cSH[:,:,1] - cSH_2d)) < 1e-5
    @test sum(abs.(cSH) .> 1.) == 3
    @test abs.(cSH[2,1,1]) > 20.
    @test abs.(cSH[2,1,2]) > 20.
    @test abs.(cSH[2,1,3]) > 20.
    
    cg = transform_grid(cSH, analysis_plan)
    @test mean(abs.(cg - cosθ) ./ abs.(cg)) < 1e-3
    
    # batched transforms & derivs 
    cosθ = repeat(cosθ, 1,1,1, 2)
    msinθ = repeat(msinθ, 1,1,1, 2)
    C = rand_grid(p, 3, 2)
    
    @test !isnothing(analysis_plan.iFT_4d)
    @test !isnothing(synthesis_plan.FT_4d)
    
    cSH = transform_SH(cosθ, synthesis_plan)
    
    @test transform_SH(cosθ[:,:,:,1], synthesis_plan) ≈ cSH[:,:,:,1]
    
    ASH = transform_SH(A, synthesis_plan)

    @test transform_SH(A[:,:,:,1], synthesis_plan) ≈ ASH[:,:,:,1]


    # gradient correctness 

    # test that the AD of the transform are doing what they are supposed to do
    y, back = Zygote.pullback(x -> transform_grid(x, analysis_plan), B)
    fd_jvp = j′vp(central_fdm(5,1), x -> transform_grid(x, analysis_plan), y, B)
    diff = (fd_jvp[1] - back(y)[1])
    @test maximum(abs.(diff)) < 1e-4 

    y, back = Zygote.pullback(x -> transform_SH(x, synthesis_plan), A)
    fd_jvp = j′vp(central_fdm(5,1), x -> transform_SH(x, synthesis_plan), y, A)
    diff = (fd_jvp[1] - back(y)[1])
    @test maximum(abs.(diff)) < 1e-4

    if CUDA.functional() 
        
        p_gpu = HarmonicsParameters(L_max, GPU=true)
        synthesis_plan_gpu = GaussianGridtoSHTransform(p_gpu, 3, 2)
        analysis_plan_gpu = SHtoGaussianGridTransform(p_gpu, 3, 2)

        A_gpu = cu(A)
        B_gpu = cu(B)

        y_cpu, back_cpu = Zygote.pullback(x -> transform_grid(x, analysis_plan), A)
        y_gpu, back_gpu = Zygote.pullback(x -> transform_grid(x, analysis_plan_gpu), A_gpu);
        @test maximum(Array(back_gpu(y_gpu)[1]) - back_cpu(y_cpu)[1]) < 1e-4

        y_cpu, back_cpu = Zygote.pullback(x -> transform_SH(x, synthesis_plan), B)
        y_gpu, back_gpu = Zygote.pullback(x -> transform_SH(x, synthesis_plan_gpu), B_gpu);
        @test Array(back_gpu(y_gpu)[1]) ≈ back_cpu(y_cpu)[1]
    end 

    # compare with SpeedyTransforms 

    #spectral_grid = SpeedyWeather.SpectralGrid(Float32, trunc=L_max, Grid=FullGaussianGrid, dealiasing=2)
    #S = SpectralTransform(spectral_grid)

    #grid = FullGaussianGrid(A)

    # convert SPH funciton


end