import DifferentiableHarmonics: plan_ir2r_AD, plan_r2r_AD
using FiniteDifferences, FFTW, CUDA, Zygote

@testset "R2R FFT Wrapper" begin 

    A = rand(100)
    A2 = rand(100, 10)

    fft_plan = plan_r2r_AD(A, 1)
    ifft_plan = plan_ir2r_AD(fft_plan * A, 100, 1)

    # test if the CPU wrappers are functional, compare to FFTW.r2r HC2R and R2HC 
    A_out = fft_plan * A
    A_out_r2r = FFTW.r2r(A, FFTW.R2HC, 1)

    @test A_out[1:51] ≈ A_out_r2r[1:51]
    @test A_out[52] ≈ 0 # thats the Im(f_0) = 0 (always)
    @test A_out[53:end-1] ≈ A_out_r2r[end:-1:52]
    @test A_out[end] ≈ 0 # thats the Im(f_end) = 0 (always)

    @test (ifft_plan * A_out)./100 ≈ A # plan not normalized 

    fft_plan_2 = plan_r2r_AD(A2, 1)
    ifft_plan_2 = plan_ir2r_AD(fft_plan_2 * A2, 100, 1)

    A_out_2 = fft_plan_2 * A2
    A_out_r2r_2 = FFTW.r2r(A2, FFTW.R2HC, 1)

    @test A_out_2[1:51,:] ≈ A_out_r2r_2[1:51,:]
    @test all(A_out_2[52,:] .≈ 0) # thats the Im(f_0) = 0 (always)
    @test A_out_2[53:end-1,:] ≈ A_out_r2r_2[end:-1:52,:]
    @test all(A_out_2[end] .≈ 0) # thats the Im(f_end) = 0 (always)

    @test (ifft_plan_2 * A_out_2)./100 ≈ A2 # plan not normalized 

    if CUDA.functional()
        # test if GPU and CPU are identical 
        A2_gpu = cu(A2)

        fft_plan_2_gpu = plan_r2r_AD(A2_gpu, 1)
        ifft_plan_2_gpu = plan_ir2r_AD(fft_plan_2 * A2_gpu, 100, 1)

        A_out_2_gpu = fft_plan_2_gpu * A2_gpu

        @test typeof(A_out_2_gpu) <: CuArray 
        @test Array(A_out_2_gpu) ≈ A_out_2
        @test (ifft_plan_2_gpu * A_out_2_gpu)./100 ≈ A2_gpu # plan not normalized 
    end 

    # test Differentiability + correctness of gradients against finite differences 
    y, back = Zygote.pullback(x -> fft_plan_2*x, A2)
    fd_jvp = j′vp(central_fdm(5,1), x -> fft_plan_2*x, y, A2)
    diff_val = (fd_jvp[1] - back(y)[1]) 
    @test maximum(abs.(diff_val)) < 1e-7

    yi, backi = Zygote.pullback(x -> ifft_plan_2*x, A_out_2)
    fd_jvpi = j′vp(central_fdm(5,1), x -> ifft_plan_2*x, yi, A_out_2)
    diff_val = (fd_jvpi[1] - backi(yi)[1]) 
    @test maximum(abs.(diff_val)) < 1e-7

    if CUDA.functional()
        A2_gpu = CUDA.CuArray(A2)
        A_out_2_gpu = CUDA.CuArray(A_out_2)

        r2r_plan_gpu = QG3.plan_r2r_AD(A2_gpu, 1)
        ir2r_plan_gpu = QG3.plan_ir2r_AD(A_out_2_gpu, size(A2_gpu,1), 1)

        cpudiv = (fft_plan_2 \ A_out_2_gpu);
        gpudiv = (r2r_plan_gpu \ A_out_2_gpu);
        @test cpudiv ≈ Array(gpudiv)

        gpudiv = Array(ir2r_plan_gpu \ A2_gpu)
        cpudiv = ifft_plan_2 \ A2
        @test gpudiv ≈ cpudiv

        y_gpu, back_gpu = Zygote.pullback(x -> r2r_plan_gpu*x, A2_gpu)
        diff_val = (fd_jvp[1] - Array(back_gpu(y_gpu)[1])) 
        @test maximum(abs.(diff_val)) < 1e-4

        y_gpu, back_gpu = Zygote.pullback(x -> ir2r_plan_gpu*x, A_out_2_gpu)
        iback_gpu = back_gpu(y_gpu)[1]; 
        diff_val = Array(iback_gpu) - fd_jvpi[1]
        @test maximum(abs.(diff_val)) < 1e-4

        diff_val = Array(iback_gpu) - fd_jvpi[1]
        @test maximum(abs.(diff_val)) < 1e-4
    end
end 