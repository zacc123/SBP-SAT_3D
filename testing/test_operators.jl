include("../ops_BP5.jl")
include("../utils.jl")
# Useful imports
using LinearAlgebra
using SparseArrays
using DifferentialEquations
using NaNMath
using DelimitedFiles
using Printf

using Plots
using BenchmarkTools
using Dates

using CUDA
using CUDA.CUSPARSE
using CUDSS

# Most of these not helpful ughhh
using Arpack
# using IncompleteLU
using LinearSolve
# using CUSOLVERRF
using Krylov, KrylovPreconditioners, LinearOperators

function test_uMask()
    print("\nTesting U Mask:")
    # Build Test U
    fail_flag = false
    u = zeros(99, 1)
    
    u1 = zeros(33, 1)
    u2 = zeros(33, 1)
    u3 = zeros(33, 1)

    # Set it to be the natual numbers 1 - 99
    u1_idx = 1
    u2_idx = 1
    u3_idx = 1
    for i in 1:99
        u[i] = i
        if mod(i, 3) == 1
            u1[u1_idx] = i
            u1_idx += 1
        elseif mod(i, 3) == 2
            u2[u2_idx] = i
            u2_idx += 1
        elseif mod(i, 3) == 0
            u3[u3_idx] = i
            u3_idx += 1
        end
    end

    mask_1 = uMask(u, 1)
    mask_2 = uMask(u, 2)
    mask_3 = uMask(u, 3)
    
    # Test Extremem Cases
    try
        print("\n\tTest Index out of bounds...")
        mask_4 = uMask(u, 4)
        print("FAILED")
        
    catch
        print("PASSED")
    end

    try
        print("\n\tTest U wrong num elems...")
        mask_4 = uMask(zeros(100), 1)
        print("FAILED")
        fail_flag = true
        
    catch
        print("PASSED")
    end

    # normal cases
    try
        print("\n\tTest Mask....")
        
        @assert abs(norm(u1 .- (mask_1' * u))) == 0.0
        @assert abs(norm(u2 .- (mask_2' * u))) == 0.0
        @assert abs(norm(u3 .- (mask_3' * u))) == 0.0
        print("PASSED")
        
    catch
        fail_flag = true
        print("FAILED")
        print("\nExpected u1: $(typeof(u1))\n$(u1)")
        print("\nReceived: $(typeof(mask_1' * u))\n$(mask_1' * u)")

        print("\nExpected u2: $(typeof(u2))\n$(u2)")
        print("\nReceived: $(typeof(mask_2' * u))\n$(mask_2' * u)")

        print("\nExpected u3: $(typeof(u3))\n$(u3)")
        print("\nReceived: $(typeof(mask_3' * u))\n$(mask_3' * u)")
        
    end
    if fail_flag
        print("\n[FAILED] : Testing U Mask")
    else
        print("\n[PASSED] : Testing U Mask\n")
    end
end

function test_mMask()
    print("\nTesting Mat Mask:")
    # Build Test U
    fail_flag = false
    M = zeros(33, 99)
    
    u1 = ones(33, 33)
    u2 = ones(33, 33) .* 2
    u3 = ones(33, 33) .* 3

    for i in 1:99
        if mod(i, 3) == 1
            M[:, i] .= 1 
            
        elseif mod(i, 3) == 2
            M[:, i] .= 2 
        elseif mod(i, 3) == 0
            M[:, i] .= 3 
        end
    end

    mask_1 = mMask(M, 1)
    mask_2 = mMask(M, 2)
    mask_3 = mMask(M, 3)
    
    # Test Extreme Cases
    try
        print("\n\tTest Index out of bounds...")
        mask_4 = mMask(u, 4)
        print("FAILED")
        
    catch
        print("PASSED")
    end

    try
        print("\n\tTest M wrong num elems...")
        mask_4 = uMask(zeros(100, 100), 1)
        print("FAILED")
        fail_flag = true
        
    catch
        print("PASSED")
    end

    # normal cases
    try
        print("\n\tTest Mask....")
        
        m1 = M * mask_1
        @assert norm(m1 .- u1) == 0
        m2 = M * mask_2
        @assert norm(m2 .- u2) == 0
        m3 = M * mask_3
        @assert norm(m3 .- u3) == 0
        print("PASSED")
        
    catch
        fail_flag = true
        print("FAILED")
        print("\nExpected m1: $(typeof(u1))\n$(u1)")
        print("\nReceived: $(typeof(m1))\n$(m1)")

    end
    if fail_flag
        print("\n[FAILED] : Testing M Mask")
    else
        print("\n[PASSED] : Testing M Mask\n")
    end
end

function test_uShift()
    print("\nTesting U Shift:")
    # Build Test U
    fail_flag = false
    u = zeros(99, 1)
    
    u1 = zeros(33, 1)
    u2 = zeros(33, 1)
    u3 = zeros(33, 1)

    # Set it to be the natual numbers 1 - 99
    u1_idx = 1
    u2_idx = 1
    u3_idx = 1
    for i in 1:99
        u[i] = i
        if mod(i, 3) == 1
            u1[u1_idx] = i
            u1_idx += 1
        elseif mod(i, 3) == 2
            u2[u2_idx] = i
            u2_idx += 1
        elseif mod(i, 3) == 0
            u3[u3_idx] = i
            u3_idx += 1
        end
    end

    shift1 = uShift(u1, 1)
    shift2 = uShift(u2, 2)
    shift3 = uShift(u3, 3)
    
    # Test Extremem Cases
    try
        print("\n\tTest Index out of bounds...")
        mask_4 = uShift(u, 4)
        print("FAILED")
        
    catch
        print("PASSED")
    end

    # normal cases
    try
        print("\n\tTest Shift....")
         
        uTest = shift1 * u1 .+ shift2 * u2 .+ shift3 * u3
        @assert abs(norm(uTest .- u)) == 0
        print("PASSED")
        
    catch
        fail_flag = true
        print("FAILED")
        print("\nExpected u: $(typeof(u))\n$(u)")
        print("\nReceived: $(typeof(uTest))\n$(uTest)")

        
    end
    if fail_flag
        print("\n[FAILED] : Testing U Shift")
    else
        print("\n[PASSED] : Testing U Shift\n")
    end
end

function test_mShift()
    print("\nTesting M Shift:")
    # Build Test U
    fail_flag = false
    # Set it to be the natual numbers 1 - 99
    M = zeros(33, 99)
    
    u1 = ones(33, 33)
    u2 = ones(33, 33) .* 2
    u3 = ones(33, 33) .* 3

    for i in 1:99
        if mod(i, 3) == 1
            M[:, i] .= 1 
            
        elseif mod(i, 3) == 2
            M[:, i] .= 2 
        elseif mod(i, 3) == 0
            M[:, i] .= 3 
        end
    end

    shift1 = mShift(u1, 1)
    shift2 = mShift(u2, 2)
    shift3 = mShift(u3, 3)
    
    # Test Extremem Cases
    try
        print("\n\tTest Index out of bounds...")
        mask_4 = mShift(u1, 4)
        print("FAILED")
        
    catch
        print("PASSED")
    end

    # normal cases
    try
        print("\n\tTest Shift....")
         
        mTest = (u1 * shift1') .+ (u2 * shift2') .+ (u3 * shift3')
        @assert abs(norm(mTest .- M)) == 0
        print("PASSED")
        
    catch
        fail_flag = true
        print("FAILED")
        print("\nExpected u: $(typeof(u))\n$(u)")
        print("\nReceived: $(typeof(uTest))\n$(uTest)")

        
    end
    if fail_flag
        print("\n[FAILED] : Testing M Shift")
    else
        print("\n[PASSED] : Testing M Shift\n")
    end
end

function test_multiply()
    
    A11, _, _, _ = diagonal_sbp_D2(2, 10)
    A12, _, _, _ = diagonal_sbp_D1(2, 10)
    A13, _, _, _ = diagonal_sbp_D1(2, 10)

    big_shift = shift_operator(A11)

    for i in 1:33
        print(norm(big_shift[i, :] .- big_shift[:, i]))
    end

    A_shift = A_test * big_shift
 
    res = A_test * x
    res_shift = A_shift * big_shift' * x

    @assert A_test * big_shift  ≈ A_shift
    @assert res≈res_shift

    
    @assert Array(big_shift) == Array(big_shift)'

    b = A_test \ res
    bs = big_shift * (A_shift \ res)
    #print("\nB: $(b[1:11])")
    #print("\nB_S$(bs[1:3:33])")
    #print("\nx:$(x[1:11])")

    

end

"""
Testing to confirm that my A's get formed right after shifting for the new stacking order
"""
function testLocoShift(grid_size::Int64)
    print("\nTesting Shift Operator on grid size: $(grid_size)^3:")
    fail_flag = false
    print("\n\tCreating Metrics...")

    Nq = Nr = Ns = grid_size

    Nqp = Nq + 1
    Nrp = Nr + 1 
    Nsp = Ns + 1

    Np = Nqp * Nrp * Nsp # total size of 1 comp of operator (i.e xx part)

    xc = (0, 100)
    yc = (-50, 50)
    zc = (0, 100)

    # Get stretch factors to move between 
    α_x = (xc[2] - xc[1]) / 2
    α_y = (yc[2] - yc[1]) / 2
    α_z = (zc[2] - zc[1]) / 2

    β_x = (xc[2] + xc[1]) / 2
    β_y = (yc[2] + yc[1]) / 2
    β_z = (zc[2] + zc[1]) / 2


    # TODO Fix these with what brittany wants for Coordinate Tranform. Start with trivial 0, Lz -> (-1, 1), etc
    xt=(q,r,s) -> ((q .* α_x) .+ β_x, ones(size(q)) .* α_x, zeros(size(r)),       zeros(size(s)))
    yt=(q,r,s) -> ((r .* α_y) .+ β_y, zeros(size(q)),       ones(size(r)) .* α_y, zeros(size(s)))
    zt=(q,r,s) -> ((s .* α_z) .+ β_z, zeros(size(q)),       zeros(size(r)),       ones(size(s)) .* α_z)

    # TODO: Run these functions by Brittany to set correctly * prob just the normal constant ρ / cs or something
        # Answer is that these exist in mms.jl : )
        # Should be fine for now though
    λ_f(x, y, z, B_p) = 1
    μ_f(x, y, z, B_p) = 2 
    K = 1 # Doesnt get used in metrics, but is asked as input
    B_p = 1

    
    metrics = create_metrics(2, Nq, Nr, Ns, λ_f, μ_f, K, B_p, xt, yt, zt)
    # metrics = create_metrics(SBPp, Nq, Nr, Ns, λ_f, μ_f, K, B_p, xt, yt, zt)
    print("Done")

    print("\n\tCreating Operators...")
    @time (M, B, JH, A, S, HqI, HrI, HsI, T, e, H, HM) = locoperator(2, Nq, Nr, Ns, metrics, metrics.C) # TODO: extraneaous C from metrics in there
    print("Done") 

    print("\n\tCreating SHIFTED Operators...")
    @time (M_shift, B, JH, A_shift, S_shift, HqI, HrI, HsI, T, e, H, HM_shift) = locoperator_shift(2, Nq, Nr, Ns, metrics, metrics.C) # TODO: extraneaous C from metrics in there
    print("Done") 

    # Test that these work

    # A
    Amask1 = mMask(A, 1)
    Amask2 = mMask(A, 2)
    Amask3 = mMask(A, 3)

    A1 = A_shift * Amask1 
    A2 = A_shift * Amask2 
    A3 = A_shift * Amask3 

    # S
    Smask1 = mMask(S, 1)
    Smask2 = mMask(S, 2)
    Smask3 = mMask(S, 3)

    S1 = S_shift * Smask1 
    S2 = S_shift * Smask2
    S3 = S_shift * Smask3  

    # M
    Mmask1 = mMask(M, 1)
    Mmask2 = mMask(M, 2)
    Mmask3 = mMask(M, 3)

    M1 = M_shift * Mmask1 
    M2 = M_shift * Mmask2 
    M3 = M_shift * Mmask3 

     # M
    HMmask1 = mMask(HM, 1)
    HMmask2 = mMask(HM, 2)
    HMmask3 = mMask(HM, 3)

    HM1 = HM_shift * HMmask1 
    HM2 = HM_shift * HMmask2 
    HM3 = HM_shift * HMmask3 

    A_test = [A1 A2 A3]
    S_test = [S1 S2 S3]
    M_test = [M1 M2 M3]
    HM_test = [HM1 HM2 HM3]

    @assert norm(HM .- HM')  == 0
    @assert norm(HM_shift .- HM_shift')  == 0

    print("\n\nDEBUG: HM MASK1:$(Array(HM_test[1:9, 1:9]))")
    print("\n\nDEBUG: HM shift:$(Array(HM_shift[1:9, 1:9]))")
    print("\n\nDEBUG: HM :$(Array(HM[1:9, 1:9]))")
    try
        print("\n\tTesting A shift == A...")
        @assert norm(A_test .- A) == 0
        print("PASSED")

        print("\n\tTesting S shift == S...")
        @assert norm(S_test .- S) == 0
        print("PASSED")

        print("\n\tTesting M shift == M...")
        @assert norm(M_test .- M) == 0
        print("PASSED")

        print("\n\tTesting HM shift == HM...")
        @assert norm(HM_test .- HM) == 0
        print("PASSED")
    catch 
        print("FAILED")
        print("\n Expected first 10x10 elements of A to be:\n$(A[1:10, 1:10])")
        print("\n Received first 10x10 elements of A_shift to be:\n$(A_test[1:10, 1:10])")
        print("\nL2 Norm of A - unshift(A_shift) =$(norm(A_test .- A))")
        fail_flag = true
    end

    if fail_flag
        print("\n[FAILED] : Locoperator Shift")
    else
        print("\n[PASSED] : Locoperator Shift\n")
    end
   
end


"""
Testing to confirm that my A's get formed right after shifting for the new stacking order
"""
function testBP5()

    localARGS = ["../BP5.dat"]
    # Read in params from DAT file for problem
    (pth, stride_space, stride_time, SBPp,
     xc, yc, zc,
     Hx, Hy, Hz, 
     Nx, Ny, Nz, 
     cg_flag, gpu_flag,
     ρ, cs, ν, 
     RSamin, RSamax, RSb,
     σn, RSDc, Vp, RSVinit,
     RSV0, RSf0, RShs,
     RSht, RSH, RSl, RSWf,
     RSlf, RSw, Δz,
     sim_years) = read_params_BP5(localARGS[1])

    # Setup the ouptut dir, try / catch to cover lazy users like me lol
    try
      mkdir(pth)
    catch
       print("\nContinuing")
    end

    # parameter house keeping and setting up the problem domain
    year_seconds = 31556926
    μ = cs^2 * ρ 
    μshear = cs^2 * ρ
    η = μshear / (2 * cs)
    λ = 2*μ*ν / (1 - 2*ν)
    Κ = 2*μ*(ν + 1) / (3* (1 - 2*ν))

    ################################## COORDINATE TRANSFORM ###################################
    # Physical Domain: (x, y, z) in (0, Lx) x (0, Ly) x (0, Lz)

    dx = (xc[2] - xc[1]) / Nx
    dy = (yc[2] - yc[1]) / Ny
    dz = (zc[2] - zc[1]) / Nz

    x = xc[1]:dx:xc[2]
    y = yc[1]:dy:yc[2]
    z = zc[1]:dz:zc[2]

    if dz < 2*Hz/Nz # TODO Ask Brittany about these checks in 3D
      #for bp1-qd, need dz to be greater than 2Hz/Ns or will get errors
      print("need more grid points or increase dz\n")
      return
    end

    # Move to Logical Names for the rest of simulation
    Nq = Nx
    Nr = Ny
    Ns = Nz

    Nqp = Nq + 1
    Nrp = Nr + 1 
    Nsp = Ns + 1

    Np = Nqp * Nrp * Nsp # total size of 1 comp of operator (i.e xx part)

    # Get stretch factors to move between 
    α_x = (xc[2] - xc[1]) / 2
    α_y = (yc[2] - yc[1]) / 2
    α_z = (zc[2] - zc[1]) / 2

    β_x = (xc[2] + xc[1]) / 2
    β_y = (yc[2] + yc[1]) / 2
    β_z = (zc[2] + zc[1]) / 2


    # TODO Fix these with what brittany wants for Coordinate Tranform. Start with trivial 0, Lz -> (-1, 1), etc
    xt=(q,r,s) -> ((q .* α_x) .+ β_x, ones(size(q)) .* α_x, zeros(size(r)),       zeros(size(s)))
    yt=(q,r,s) -> ((r .* α_y) .+ β_y, zeros(size(q)),       ones(size(r)) .* α_y, zeros(size(s)))
    zt=(q,r,s) -> ((s .* α_z) .+ β_z, zeros(size(q)),       zeros(size(r)),       ones(size(s)) .* α_z)

    # TODO: Run these functions by Brittany to set correctly * prob just the normal constant ρ / cs or something
        # Answer is that these exist in mms.jl : )
        # Should be fine for now though
    λ_f(x, y, z, B_p) = λ
    μ_f(x, y, z, B_p) = μ 
    K = Κ # Doesnt get used in metrics, but is asked as input
    B_p = 1

    print("\nCreating metrics....\n")
    @time metrics = create_metrics(SBPp, Nq, Nr, Ns, λ_f, μ_f, K, B_p, xt, yt, zt)
    # metrics = create_metrics(SBPp, Nq, Nr, Ns, λ_f, μ_f, K, B_p, xt, yt, zt)
    print("\nCreating metrics Done\n")

    ###################################################################### 
    # create finite difference operators on computational domain:
    # Notation: 
        # M == D2 + SAT terms for RHS, 
        # B == Boundary Coefs,
        # JH == Det of the Jacobian x H tilde,
        # A == D2, 
        # S == SAT Coefs
    print("\nCreating Operators....\n")
    # @time (M_tmp, B_tmp, JH_tmp, A_tmp, S_tmp, HqI_tmp, HrI_tmp, HsI_tmp, T_tmp, e_tmp, H_tmp, HM_tmp) = locoperator(SBPp, Nq, Nr, Ns, metrics, metrics.C) # TODO: extraneaous C from metrics in there
    @time (M, B, JH, A, S, HqI, HrI, HsI, T, e, H, HM) = locoperator(SBPp, Nq, Nr, Ns, metrics, metrics.C) # TODO: extraneaous C from metrics in there
    print("\nCreating Operators Done\n") 

    print("\nCreating Operators....\n")
     @time (M_tmp, B_tmp, JH_tmp, A_tmp, S_tmp, HqI_tmp, HrI_tmp, HsI_tmp, T_tmp, e_tmp, H_tmp, HM_tmp) = locoperator(SBPp, Nq, Nr, Ns, metrics, metrics.C) # TODO: extraneaous C from metrics in there
    #@time (M_tmp, B_tmp, JH_tmp, A_, S, HqI, HrI, HsI, T, e, H, HM) = locoperator(SBPp, Nq, Nr, Ns, metrics, metrics.C) # TODO: extraneaous C from metrics in there
    print("\nCreating Operators Done\n") 



     # initialize time and vector b that stores boundary data (linear system will be Au = b, where b = B*g)
    t = 0
    b = zeros(3 * Nqp * Nrp * Nsp) # this sucker is bigggggg 
    b_tmp = zeros(3 * Nqp * Nrp * Nsp) # this sucker is bigggggg 

    # initial slip vector
    δ = zeros(2 * Nrp * Nsp) # 2D Plane aghhhh 3 components :|
    δ_tmp = zeros(2 * Nrp * Nsp) # 2D Plane aghhhh 3 components :|

    # get grid size for setting b
    params = (Nqp, Nrp, Nsp)
    
    # Set face two
    remote_boundary = zeros(Nrp * Nsp * 3)
    remote_boundary[Nrp*Nsp+1: 2*Nrp*Nsp] += (t * Vp/2) .* ones(Nrp*Nsp)

    N = Nqp * Nrp * Nsp

    print("\nBuilding Mask and Shift Operators:\n\tTime:")
    @time begin
        shifty = shift_operator(HM)
    end

    # set b for inital displacement calc
    bdry_vec_strip!(b, B, δ ./ 2, remote_boundary, H, params)
    bdry_vec_strip!(b_tmp, B_tmp, δ_tmp ./ 2, remote_boundary,  H_tmp, params)
    
    try
        print("\n\tTesting bndry_vec_strip_shift at t=0...")
        res = norm(b .- b_tmp)
        @assert res == 0.0
        print("PASSED")
    catch
        print("FAILED")
        print("\nExpected: b = $(b_tmp[1:10])")
        print("\nGot: b = $(b1[1:10])")
    end
     # these are testing flags and results will get added into other things
        # First Get M to look right and be SPD
    print("\nMaking HM SPD...")
    HM .*= -1
    
    # Now we are going to take HM, shift it and multiply by the transpose
    HM = HM * shifty
    M = HM' * HM


    HM_tmp .*= -1
    M_tmp = HM_tmp
    print("DONE")

    print("\nCopying HM to Device in CSR format...")
    M_cu = CuSparseMatrixCSR(M) # Move PD matrix to GPU
    M_cu_tmp = CuSparseMatrixCSR(M_tmp) # Move PD matrix to GPU
    print("Done\n")
        
    u = zeros(size(b)) # need to initialize u
    b_cu = CuArray(HM' * b)
    workspace = Krylov.CgWorkspace(M_cu, b_cu)

    u_tmp = zeros(size(b_tmp)) # need to initialize u
    b_cu_tmp = CuArray(b_tmp)
    workspace_tmp = Krylov.CgWorkspace(M_cu_tmp, b_cu_tmp)
        
    print("\nTime for 1st solve:")
    Krylov.cg!(workspace, M_cu, b_cu)
    @time Krylov.cg!(workspace, M_cu, b_cu) # warmup
    u .= Array(workspace.x)

    Krylov.cg!(workspace_tmp, M_cu_tmp, b_cu_tmp)
    print("\n")
    @time Krylov.cg!(workspace_tmp, M_cu_tmp, b_cu_tmp) # warmup
    u_tmp .= Array(workspace_tmp.x)

    u_test = shifty * u
    res = norm(u_test .- u_tmp)
    print("\nRes of u vs u: $(res)")
    try
        
        @assert res == 0.0
        print("PASSED")
    catch
        print("FAILED")

    end
    # Following vectors, τ, RSa, θ will only apply to Face 1, and are size 1x(NspxNrp)
    # initialize change in shear stress due to quasi-static deformation
   
    # Set friction coefficients for rate and state
    RS_params = RSht, RSl, RSlf, RSw, RSWf, RShs, RSH, RSamin, RSamax, RSDc, RSVinit
    grid_params = (xc[1]:dx:xc[2], yc[1]:dy:yc[2], zc[1]:dz:zc[2],
                    Nqp, Nrp, Nsp)

    remote_boundary = zeros(3 * Nqp * Nrp * Nsp)
    remote_boundary[1+ (Nqp * Nrp * Nsp): 2* Nqp * Nrp * Nsp] += (Vp .* 10 ./ 2) .* ones(Nqp * Nrp * Nsp) # Slow creep at face 2 

    δ  .= 1
    δ_tmp .= 1

    bdry_vec_strip!(b, B, δ ./ 2, remote_boundary, H, params)
    bdry_vec_strip!(b_tmp, B_tmp, δ_tmp ./ 2, remote_boundary,  H_tmp, params)
    
    # solve for displacements everywhere in domain
    b .*= -1 
    b = HM' * b
    b_tmp .*= -1

    b_cu = CuArray(b)
    b_cu_tmp = CuArray(b_tmp)

    atol_0 = norm(b) * eps(Float64)

    res_tmp, stats_tmp = Krylov.cg(M_cu_tmp, b_cu_tmp, atol=atol_0, rtol=1e-6)
    
    tmp_res_tmp = Array(res_tmp)
    
    res, stats = Krylov.cg(M_cu, b_cu,  atol=atol_0, rtol=1e-6)

    u .= Array(res)
    u_tmp .= Array(res_tmp)

    N = Nrp * Nqp * Nsp
    u = shifty * u
    print("\nCG 2: U - u: $(norm(u .- u_tmp, Inf))")

    # Set initial state variable according to benchmark
    θ, RS_indices, Nθ = set_theta(RS_params, grid_params)
    RSDc = RSDc .* ones(length(θ))
    # Initialize psi version of state variable
    ψ = RSf0 .+ RSb .* log.(RSV0 .* θ ./ RSDc)

    # Update friction coefficients based on RS zone
    RSa = initialize_friction_params_vec(RS_params, grid_params, Nθ, RS_indices)
    # Set pre-stress according to benchmark

    # A bit tricky, τ has y and z comp.  scalar pres stress initialized according to BP5 eq 22
    τ0 = σn .* RSa .* asinh.((RSVinit / (2 * RSV0)) .* exp.((RSf0 + RSb * log.(RSV0 / RSVinit)) ./ RSa)) .+ (η * RSVinit)
    

    Δτ_vec = zeros(2 * length(τ0)) # this will be how stresses change through sim
    τ0_vec = zeros(length(Δτ_vec))

    RSVzero = 1e-20 # TODO move this into DAT file
    V = [RSVinit, RSVzero]
    V_mag = norm(V, 2)

    τ0_vec[1:Nθ] .= (τ0 .* V[1] ./ V_mag) # set y and z comps
    τ0_vec[1+Nθ:2*Nθ] .= (τ0 .* V[2] ./ V_mag)

    # Quick sanity checks
    @assert length(τ0) == length(RSa)
    @assert length(τ0) ==  (RS_indices[1, 2] - RS_indices[1, 1] + 1) * (RS_indices[2, 2] - RS_indices[2, 1] + 1)

    # For QD Setup, reset tau0 in nucleation zone
    # TODO move this to the .dat file
    Vi = 0.03
    τ_params = Vi, RSV0, RSVinit, σn, η, RSb, RSf0
    set_prestress_QD!(τ0_vec, RS_params, grid_params, τ_params, Nθ, RS_indices, RSDc)

    # Set initial condition for index 1 DAE - this is a stacked vector of psi, followed by slip
    # Can ask brittany if this is ok but I think it should work
    # TODO
    ψδ = zeros(Nθ + (2* Nrp * Nsp))  #because length(ψ) = 1 * Nrp * Nsp,  length(δ) = 2 * Nrp * Nsp 
    ψδ[1:Nθ] .= ψ[:]
    ψδ[Nθ+1:end] .= δ[:]

    # Set up stations on fault using Y, Z indices
    stations = [(0.0, 0.0), (0.0, 10.0), (0.0, 22.0), (16.0, 0.0), (16.0, 10.0), (36.0, 0.0), (-16.0, 0.0), (-16.0, 10.0), (-24.0, 10.0), (-36.0, 0.0)] # km
    station_indices = find_station_index(stations, y, z)
    station_strings = [ "0000", "0010", "0022", "1600", "1610", "3600", "-1600", "-1610", "-2410", "-3600"] # str names "$(x_digits)$(y_digits)" where each gets 2 digits e.g y=16,z=10 = "1610"

    # Start here
    Δτ_tmp = computetraction_stripped_shift(T, u, e, metrics.sJ, masks)     # calc Traction on whole face
    Δτ_tmp_tmp = computetraction_stripped(T_tmp, u_tmp, e_tmp, metrics.sJ) 

    res = norm(Δτ_tmp .- Δτ_tmp_tmp)
    print("\nResidual of traction= $(res)")

    V = zeros((2* Nrp * Nsp))
    Δτ_2, Δτ_3, V2, V3 = update_tau_v_vec(Δτ_tmp, V, RS_params, grid_params, Nθ, RS_indices)
    Δτ_2_tmp, Δτ_3_tmp, V2_tmp, V3_tmp = update_tau_v_vec(Δτ_tmp_tmp, V, RS_params, grid_params, Nθ, RS_indices)

    res = norm(Δτ_2 .- Δτ_2_tmp)
    print("\nResidual of t2= $(res)")
    res = norm(Δτ_3 .- Δτ_3_tmp)
    print("\nResidual of t3= $(res)")
    res = norm(V2 .- V2_tmp)
    print("\nResidual of v2= $(res)")
    res = norm(V3 .- V3_tmp)
    print("\nResidual of v3= $(res)")

    # Sanity Check, make sure delta tau is set correctly
    Δτ_vec[1:Nθ] .=  Δτ_2[:]
    Δτ_vec[1+Nθ:end] .=  Δτ_3[:]

    Δτ_vec_tmp = zeros(2*Nθ)
   
    Δτ_vec_tmp[1:Nθ] .=  Δτ_2_tmp[:]
    Δτ_vec_tmp[1+Nθ:end] .=  Δτ_3_tmp[:]
    
    τf = Δτ_vec .+ τ0_vec # Set final stress on RS fault
    τf_tmp = Δτ_vec_tmp .+ τ0_vec # Set final stress on RS fault

    # break into comp for easier reading
    τf_2 = τf[1:Nθ]
    τf_3 = τf[1+Nθ:2*Nθ]

    τf_2_tmp = τf_tmp[1:Nθ]
    τf_3_tmp = τf_tmp[1+Nθ:2*Nθ]

    # This is just a 0 vector lol
    V_v = hypot.(V2, V3)
    τ_magnitudes = hypot.(τf_2, τf_3) # get these for newton method

    V_v_tmp = hypot.(V2_tmp, V3_tmp)
    τ_magnitudes_tmp = hypot.(τf_2_tmp, τf_3_tmp)

    res = norm(V_v .- V_v_tmp)
    print("\nResidual of VV= $(res)")
    res = norm(τ_magnitudes .- τ_magnitudes_tmp)
    print("\nResidual of t3= $(res)")
   
end




function runTests()
    # Vector Stacking Operations
    #test_uMask()
    #test_uShift()
    # Matrix Stacking Operations
    #test_mMask()
    #test_mShift()
    #testLocoShift(10)
    #testLocoShift(25)
    testBP5()
    #test_multiply()
end

runTests()