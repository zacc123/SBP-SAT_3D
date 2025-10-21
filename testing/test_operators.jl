include("./operators.jl")

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
    @time (M_shift, B, JH, A_shift, S_shift, HqI, HrI, HsI, T, e, H, HM) = locoperator_shift(2, Nq, Nr, Ns, metrics, metrics.C) # TODO: extraneaous C from metrics in there
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
    

    A_test = [A1 A2 A3]
    S_test = [S1 S2 S3]
    M_test = [M1 M2 M3]

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

function runTests()
    # Vector Stacking Operations
    test_uMask()
    test_uShift()
    # Matrix Stacking Operations
    test_mMask()
    test_mShift()
    testLocoShift(10)
    testLocoShift(25)
end

runTests()