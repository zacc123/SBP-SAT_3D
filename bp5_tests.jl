#= 
Unit Tests for BP5 Code
=#

using SparseArrays
using LinearAlgebra

include("./BP5_ops.jl")
include("../utils.jl")

function Test_Slice_Helpers()
    print("Testing Slicing Operators:\n")

    # Setup grid
    x = 1:1:10
    y = 1:1:5
    z = 1:1:20
    comp = 1:1:3

    Nx = length(x)
    Ny = length(y)
    Nz = length(z)
    Nc = length(comp)

    @assert Nx == 10
    @assert Ny == 5
    @assert Nz == 20
    @assert Nc == 3


    u = zeros(Nx * Ny * Nz * Nc)

    num_vals = (Nx, Ny, Nz)

    # set vals the old school way to make sure my understanding on stacking order is right
    cnt = 1
    for c in comp
        for i in x
            for j in y
                for k in z
                    u[cnt] = cnt
                    cnt += 1
                end
            end
        end
    end

    # Test 1:
    print("\tTesting get_value...")
    indices_1 = (1, 1, 1, 1)
    indices_2 = (3, 2, 6, 3)
    indices_3 = (10, 5, 20, 3)

    try
        @assert get_value(u, indices_1, num_vals) == 1
        @assert get_value(u, indices_2, num_vals) == 2 * (Nx * Ny * Nz) + 2 * (Ny * Nz) + 1 * (Nz) + 6
        @assert get_value(u, indices_3, num_vals) == 3000
        print("[OK]\n")
    catch 
        print("[ERROR]\n")
        print("Expected: [1, $(2 * (Nx * Ny * Nz) + 2 * (Ny * Nz) + 1 * (Nz) + 6), $(3000)]\n")
        print("Received: [$(get_value(u, indices_1, num_vals)), $(get_value(u, indices_2, num_vals)), $(get_value(u, indices_3, num_vals))]\n")
    end

    # Test 2:
    print("\tTesting get_vector...")
    indices_1 = (1:1, 1, 1, 1)
    indices_2 = (3, 2, 6:19, 3)
    indices_3 = (1:10, 1:5, 1:20, 1:3)

    try
        @assert get_vector(u, indices_1, num_vals) == [1]
        #@assert get_value(u, indices_2, num_vals) == 2 * (Nx * Ny * Nz) + 2 * (Ny * Nz) + 1 * (Nz) + 6
        #@assert get_value(u, indices_3, num_vals) == 3000
        print("[OK]\n")
    catch 
        print("[ERROR]\n")
        print("Expected: $([1.0])\n")
        print("Received: $(get_vector(u, indices_1, num_vals))\n")
    end
    
end

function Test_Fault_Finders()

    print("Running Tests for Fault Helper Functions:\n")
    # Setup Fault
    # Get stretch factors to move between 
    xc = (0, 80)
    yc = (-40, 40)
    zc = (0, 80)

    dx = 8
    dy = 8
    dz = 8

    x_grid = 0:dx:80
    y_grid = -40:dy:40
    z_grid = 0:dz:80

    
    α_x = (xc[2] - xc[1]) / 2
    α_y = (yc[2] - yc[1]) / 2
    α_z = (zc[2] - zc[1]) / 2

    β_x = (xc[2] + xc[1]) / 2
    β_y = (yc[2] + yc[1]) / 2
    β_z = (zc[2] + zc[1]) / 2

    Nq = Nr = Ns = 10

    # TODO Fix these with what brittany wants for Coordinate Tranform. Start with trivial 0, Lz -> (-1, 1), etc
    xt=(q,r,s) -> ((q .* α_x) .+ β_x, ones(size(q)) .* α_x, zeros(size(r)),       zeros(size(s)))
    yt=(q,r,s) -> ((r .* α_y) .+ β_y, zeros(size(q)),       ones(size(r)) .* α_y, zeros(size(s)))
    zt=(q,r,s) -> ((s .* α_z) .+ β_z, zeros(size(q)),       zeros(size(r)),       ones(size(s)) .* α_z)

    # TODO: Run these functions by Brittany to set correctly * prob just the normal constant ρ / cs or something
        # Answer is that these exist in mms.jl : )
        # Should be fine for now though
    λ_f(x, y, z, B_p) = 1
    μ_f(x, y, z, B_p) = 1
    K = 0 # Doesnt get used in metrics, but is asked as input
    B_p = 1

    
    # @time metrics = create_metrics(SBPp, Nq, Nr, Ns, λ_f, μ_f, K, B_p, xt, yt, zt)
    metrics = create_metrics(2, Nq, Nr, Ns, λ_f, μ_f, K, B_p, xt, yt, zt)
    fault_y = metrics.facecoord[2][1] 
    RS_params = 8, 32, 64, 8, 64, 4, 30, 0.004, 0.04, 0.14, 1e-9
                #RSht, RSl, RSlf, RSw, RSWf, RShs, RSH, RSamin, RSamax
                # (y, z, hs, ht, H, l)
    
    grid_params = (xc[1]:dx:xc[2], yc[1]:dy:yc[2], zc[1]:dz:zc[2],
                    Nq+1, Nr+1, Ns+1)
    θ, indices, Nθ = set_theta(RS_params, grid_params)
    a = initialize_friction_params_mat(RS_params, grid_params, Nθ, indices)

    res = zeros(9, 9)
    
    res[1, 1:9] .= 0.04
    res[2, 1:9] .= 0.04

    res[3, 1] = 0.04
    res[3, 2] = RS_r(y_grid[4], z_grid[2], 4, 8, 30, 32) * (0.04 - 0.004) + 0.004
    res[3, 3:6] .= 0.004
    res[3, 7] = RS_r(y_grid[4], z_grid[7], 4, 8, 30, 32) * (0.04 - 0.004) + 0.004
    res[3, 8:9] .= 0.04

    res[4, 1] = 0.04
    res[4, 2] = RS_r(y_grid[5], z_grid[2], 4, 8, 30, 32) * (0.04 - 0.004) + 0.004
    res[4, 3:6] .= 0.004
    res[4, 7] = RS_r(y_grid[5], z_grid[7], 4, 8, 30, 32) * (0.04 - 0.004) + 0.004
    res[4, 8:9] .= 0.04

    res[5, 1] = 0.04
    res[5, 2] = RS_r(y_grid[6], z_grid[2], 4, 8, 30, 32) * (0.04 - 0.004) + 0.004
    res[5, 3:6] .= 0.004
    res[5, 7] = RS_r(y_grid[6], z_grid[7], 4, 8, 30, 32) * (0.04 - 0.004) + 0.004
    res[5, 8:9] .= 0.04
    
    res[6, 1] = 0.04
    res[6, 2] = RS_r(y_grid[7], z_grid[2], 4, 8, 30, 32) * (0.04 - 0.004) + 0.004
    res[6, 3:6] .= 0.004
    res[6, 7] = RS_r(y_grid[7], z_grid[7], 4, 8, 30, 32) * (0.04 - 0.004) + 0.004
    res[6, 8:9] .= 0.04

    res[7, 1] = 0.04
    res[7, 2] = RS_r(y_grid[8], z_grid[2], 4, 8, 30, 32) * (0.04 - 0.004) + 0.004
    res[7, 3:6] .= 0.004
    res[7, 7] = RS_r(y_grid[8], z_grid[7], 4, 8, 30, 32) * (0.04 - 0.004) + 0.004
    res[7, 8:9] .= 0.04

    res[8, 1:9] .= 0.04
    res[9, 1:9] .= 0.04

    try
        print("\tTesting initialize_friction_params_mat...")
        @assert a == res
        print("[OK]\n")
    catch 
        print("[ERROR]\n")
        is, j = size(a)
        print(RS_params)
        for i in 1:is
            print("\nRow $(i): \nActual$(a[i, :])\nExpected:$(res[i, :])\n")
        end
    end

    b = initialize_friction_params_vec(RS_params, grid_params, Nθ, indices) 
    try
        print("\tTesting initialize_friction_params_vec...")
        res_vec = zeros(Nθ)
        rows, cols = size(res)
        for row in 1:rows
            res_vec[(row - 1)*cols + 1: row*cols] .= res[row, 1:cols]
        end
        @assert b[:] == res_vec[:]
        print("[OK]\n")
    catch 
        print("[ERROR]\n")
        print("\nActual$(b[:])\nExpected:$(res[:])\n")
    end

end

function main()
    Test_Slice_Helpers()
    Test_Fault_Finders()
    return nothing
end

main()
