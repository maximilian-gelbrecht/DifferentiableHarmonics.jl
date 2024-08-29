import DifferentiableHarmonics: SHtoGrid_dλ, SHtoGrid_dμ, SHtoGrid_dθ, Δ

@testset "Derivatives" begin 

    L_max = 21 
    N_batch = 2
    p = HarmonicsParameters(L_max)

    dλ = Derivative_dλ(p, N_batch=N_batch)
    dμ = GaussianGrid_dμ(p, 3, N_batch)
    L = Laplacian(p, init_inverse=true, N_batch=N_batch)
    analysis_plan = SHtoGaussianGridTransform(p, 3, N_batch)
    synthesis_plan = GaussianGridtoSHTransform(p, 3, N_batch)

    # 2D deriv
    cosθ = zeros_grid(p) 
    msinθ = zeros_grid(p)
    for i ∈ 1:p.N_lats
        cosθ[i,:] .= cos(p.θ[i])
        msinθ[i,:] .= -sin(p.θ[i])
    end
    cSH_2d = transform_SH(cosθ, synthesis_plan)

    # very close to zero
    @test abs.(mean(SHtoGrid_dλ(cSH_2d, dλ, analysis_plan))) < 1e-5
    
    # near constant 1
    @test mean(abs.(SHtoGrid_dμ(cSH_2d, dμ) .- 1)) < 1e-2
    
    # near -sinθ
    @test mean(abs.(SHtoGrid_dθ(cSH_2d, dμ) - msinθ)) < 1e-2

    cosθ = repeat(cosθ, 1, 1, 3)
    msinθ = repeat(msinθ, 1, 1, 3)
    cSH = transform_SH(cosθ, synthesis_plan)

    # 3D deriv
    # very close to zero
    @test abs.(mean(SHtoGrid_dλ(cSH, dλ, analysis_plan))) < 1e-5
    
    # near constant 1
    @test mean(abs.(SHtoGrid_dμ(cSH, dμ) .- 1)) < 1e-2
    
    # near -sinθ
    @test mean(abs.(SHtoGrid_dθ(cSH, dμ) - msinθ)) < 1e-2

    #there's a bias close to the poles)
    A = rand_SH(p, 2, 2)
    msinθ = msinθ[:,:,:,1]

    # test Laplacian (there's a bias close to the poles)
    L1 = SHtoGrid_dθ(transform_SH((-msinθ).*SHtoGrid_dθ(A, dμ),synthesis_plan),dμ) ./ (-msinθ) + SHtoGrid_dφ(SHtoSH_dφ(ψ_0, dλ), dλ, analysis_plan) ./ (msinθ .* msinθ)
    L2 = transform_grid(Δ(A, L), analysis_plan)
    @test mean(abs.(L1-L2)[4:end-4,:,:]) < 0.05

    B = rand_SH(p, 2, 2)
    # gradient correctness 
    y, back = Zygote.pullback(x -> SHtoGrid_dθ(x,  dμ), B)
    fd_jvp = j′vp(central_fdm(5,1), x -> SHtoGrid_dθ(x,  dμ), y, B)
    diff = (fd_jvp[1] - back(y)[1])
    @test maximum(abs.(diff)) < 1e-4 

    y, back = Zygote.pullback(x -> SHtoSH_dφ(x,  dλ), B)
    fd_jvp = j′vp(central_fdm(5,1), x -> SHtoSH_dφ(x,  dλ), y, B)
    diff = (fd_jvp[1] - back(y)[1])
    @test maximum(abs.(diff)) < 1e-4 

    y, back = Zygote.pullback(x -> Δ(x, L), B)
    fd_jvp = j′vp(central_fdm(5,1), x -> Δ(x, L), y, B)
    diff = (fd_jvp[1] - back(y)[1])
    @test maximum(abs.(diff)) < 1e-4 

    if CUDA.functional()

        p_gpu = HarmonicsParameters(L_max, GPU=true)
        analysis_plan_gpu = GaussianGridtoSHTransform(p_gpu)
        synthesis_plan_gpu = SHtoGaussianGridTransform(p_gpu)

        dλ_gpu = Derivative_dλ(p_gpu, N_batch=N_batch)
        dμ_gpu = GaussianGrid_dμ(p_gpu, 3, N_batch)
        L_gpu = Laplacian(p_gpu, init_inverse=true, N_batch=N_batch)
        analysis_plan = SHtoGaussianGridTransform(p_gpu, 3, N_batch)
        synthesis_plan = GaussianGridtoSHTransform(p_gpu, 3, N_batch)

        A_gpu = cu(A)

        y_cpu, back_cpu = Zygote.pullback(x -> SHtoGrid_dθ(x, dμ), A)
        y_gpu, back_gpu = Zygote.pullback(x -> SHtoGrid_dθ(x, dμ_gpu), A_gpu);
        @test y_cpu ≈ Array(y_gpu)
        @test maximum(Array(back_gpu(y_gpu)[1]) - back_cpu(y_cpu)[1]) < 1e-4

        y_cpu, back_cpu = Zygote.pullback(x -> SHtoSH_dφ(x, dλ), A)
        y_gpu, back_gpu = Zygote.pullback(x -> SHtoSH_dφ(x, dλ_gpu), A_gpu);
        @test y_cpu ≈ Array(y_gpu)
        @test Array(back_gpu(y_gpu)[1]) ≈ back_cpu(y_cpu)[1]
    end 
end 