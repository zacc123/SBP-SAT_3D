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
using AlgebraicMultigrid

include("./ops_BP5.jl")
include("./odefun_BP5.jl")
include("./utils.jl") # get 3d metrics and ops

const global localARGS = ["./BP5.dat"]

function main()

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
        # folder already exists and data will be overwritten.
        print("Directory: $(pth) already exists \nDo you want to overwrite the existing data? (Y or N)\n")
        response = readline()
        if (response[1] == 'Y' || response[1] == 'y')
            print("Continuing...\n")
        
        else
            print("aborting...\n")
            return nothing
        end
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
    @time (M, B, JH, A, S, HqI, HrI, HsI, T, e, H, HM) = locoperator(SBPp, Nq, Nr, Ns, metrics, metrics.C) # TODO: extraneaous C from metrics in there
    print("\nCreating Operators Done\n") 

   
   
     # initialize time and vector b that stores boundary data (linear system will be Au = b, where b = B*g)
    t = 0
    b = zeros(3 * Nqp * Nrp * Nsp) # this sucker is bigggggg 
    
    # initial slip vector
    δ = zeros(2 * Nrp * Nsp) # 2D Plane aghhhh 3 components :|

    # get grid size for setting b
    params = (Nqp, Nrp, Nsp)
    
    # Set face two
    remote_boundary = zeros(Nrp * Nsp * 3)
    remote_boundary[Nrp*Nsp+1: 2*Nrp*Nsp] += (t * Vp/2) .* ones(Nrp*Nsp)

    print("\nBuilding Mask and Shift Operators:\n\tTime:")
    @time begin
        shift = shift_operator(M)
    end

    bdry_vec_strip!(b, B, δ ./ 2, remote_boundary, params)
    # if doing backslash, do this up front
    # Everything will be done on CPU with Backslash
   
    # these are testing flags and results will get added into other things
    # First Get M to look right and be SPD
    print("\nMaking HM SPD...")
    M_shift = (M * shift)'
    M2 = M_shift * M_shift'
    print("Done\n")

    print("\nCopying HM to Device in CSR format...")
    M_cu = CuSparseMatrixCSR(M2) # Move PD matrix to GPU
    print("Done\n")
    b = M_shift * b

    u = zeros(size(b)) # need to initialize u

    b_cu = CuArray(b)

    workspace = Krylov.CgWorkspace(M_cu, b_cu)
    

    print("\nTime for 1st solve:")
    Krylov.cg!(workspace, M_cu, b_cu)
    @time Krylov.cg!(workspace, M_cu, b_cu) # warmup
    u .= shift * Array(workspace.x)
    
    # Following vectors, τ, RSa, θ will only apply to Face 1, and are size 1x(NspxNrp)
    # initialize change in shear stress due to quasi-static deformation
   
    # Set friction coefficients for rate and state
    RS_params = RSht, RSl, RSlf, RSw, RSWf, RShs, RSH, RSamin, RSamax, RSDc, RSVinit
    grid_params = (xc[1]:dx:xc[2], yc[1]:dy:yc[2], zc[1]:dz:zc[2],
                    Nqp, Nrp, Nsp)

    
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
    
   
    # TEST 1, NO PC
    odeparam = (reject_step = [false], 
                sim_years =  sim_years,
                Vp=Vp,
                M = M_shift,
                M2 = M2,
                u=u,
                Δτ = Δτ_vec,
                τf = τ0_vec.*ones(length(τ0_vec)),
                b = b,
                μshear=μshear,
                RSa=RSa,
                RSb=RSb,
                σn=σn,
                η=η,
                RSV0=RSV0,
                Nθ = Nθ,
                τ0=τ0_vec,
                RSDc=RSDc,
                RSf0=RSf0,
                B = B,
                T = T,
                x = x,
                y = y, 
                z = z,
                e = e,
                sJ = metrics.sJ,
                save_stride_fields = stride_time, # save every save_stride_fields time steps
                RS_params = RS_params,
                RS_indices = RS_indices,
                t_prv = [0.0],
                M_cu = M_cu,
                Pc = nothing,
                workspace=workspace,
                counter=[0],
                num_iters=[0],
                num_nm_iters=[0],
                num_aprods = [0],
                name = "none",
                shift=shift
                )
    # Set time span over which to solve:
    tspan = (0, sim_years * year_seconds)
    # Set up ODE problem corresponding to DAE
    
    flt_loc_y = y[RS_indices[1, 1]:stride_space:RS_indices[1, 2]]
    flt_loc_z = z[RS_indices[2, 1]:stride_space:RS_indices[2, 2]] 
               
    flt_loc_indices = RS_indices
    
    # Set call-back function so that files are written to after successful time steps only.
    # cb_fun = SavingCallback((ψδ, t, i) -> write_to_file_BP5(pth, ψδ, t, i, y, z, flt_loc_y, flt_loc_z, flt_loc_indices,station_strings, station_indices, odeparam, "BP5_", 0.1 * year_seconds), SavedValues(Float64, Float64))

    # Testing block
    max_its = 50
    ### 
    ###
    #=
    # Start with NO P_C
    prob = ODEProblem(odefun_cg_testing_shift, ψδ, tspan, odeparam)
    
    # Solve DAE using Tsit5()
    print("\nTime for No PC Solve for 20 iterations:")
    @time sol = solve(prob, Tsit5(); dt=0.2,
            abstol = 1e-5, reltol = 1e-5, save_everystep=true, gamma = 0.2,
            internalnorm=(x, _)->norm(x, Inf), maxiters=max_its)
    n = odeparam.counter[1]
    print("\nAvg Num Iters: $(odeparam.num_iters[1]/n)")
    print("\nAvg Num NM Iters: $(odeparam.num_nm_iters[1]/n)")
    print("\nAvg Num A Prods: $(odeparam.num_aprods[1]/n)")
    end
    =#
    ### 
    # 3 BWJ
    ###
    # Start with ILU preconditioner 
    print("\nGetting Blockwise Jacobi PC on Device...")
    BJ_cu = KrylovPreconditioners.kp_block_jacobi(M_cu) # Get incomplete cholesky decomp
    print("Done\n")
    for i in 1:4
        
        

        odeparam3 = (reject_step = [false], 
                    sim_years =  sim_years,
                    Vp=Vp,
                    M = M_shift,
                    M2 = M2,
                    u=u,
                    Δτ = Δτ_vec,
                    τf = τ0_vec.*ones(length(τ0_vec)),
                    b = b,
                    μshear=μshear,
                    RSa=RSa,
                    RSb=RSb,
                    σn=σn,
                    η=η,
                    RSV0=RSV0,
                    Nθ = Nθ,
                    τ0=τ0_vec,
                    RSDc=RSDc,
                    RSf0=RSf0,
                    B = B,
                    T = T,
                    x = x,
                    y = y, 
                    z = z,
                    e = e,
                    sJ = metrics.sJ,
                    save_stride_fields = stride_time, # save every save_stride_fields time steps
                    RS_params = RS_params,
                    RS_indices = RS_indices,
                    t_prv = [0.0],
                    M_cu = M_cu,
                    Pc = BJ_cu,
                    workspace=workspace,
                    counter=[0],
                    num_iters=[0],
                    num_nm_iters=[0],
                    num_aprods = [0],
                    name = "BWJ",
                    shift=shift
                    )
        prob = ODEProblem(odefun_cg_testing_shift, ψδ, tspan, odeparam3)
        
        # Solve DAE using Tsit5()
        print("\nTime for BWJ Solve for 20 iterations:")
        @time sol = solve(prob, Tsit5(); dt=0.2,
                abstol = 1e-5, reltol = 1e-5, save_everystep=true, gamma = 0.2,
                internalnorm=(x, _)->norm(x, Inf), maxiters=max_its)

        n = odeparam3.counter[1]
        print("\nAvg Num Iters: $(odeparam3.num_iters[1]/n)")
        print("\nAvg Num NM Iters: $(odeparam3.num_nm_iters[1]/n)")
        print("\nAvg Num A Prods: $(odeparam3.num_aprods[1]/n)")
    end
    
    ### 
    # 5 CHolesky
    ###
    #=
    Ic_cu = KrylovPreconditioners.kp_ic0(M_cu)
    # Start with ILU preconditioner 
    # TEST 1, NO PC
    odeparam5 = (reject_step = [false], 
                sim_years =  sim_years,
                Vp=Vp,
                M = M_shift,
                M2 = M2,
                u=u,
                Δτ = Δτ_vec,
                τf = τ0_vec.*ones(length(τ0_vec)),
                b = b,
                μshear=μshear,
                RSa=RSa,
                RSb=RSb,
                σn=σn,
                η=η,
                RSV0=RSV0,
                Nθ = Nθ,
                τ0=τ0_vec,
                RSDc=RSDc,
                RSf0=RSf0,
                B = B,
                T = T,
                x = x,
                y = y, 
                z = z,
                e = e,
                sJ = metrics.sJ,
                save_stride_fields = stride_time, # save every save_stride_fields time steps
                RS_params = RS_params,
                RS_indices = RS_indices,
                t_prv = [0.0],
                M_cu = M_cu,
                Pc = Ic_cu,
                workspace=workspace,
                counter=[0],
                num_iters=[0],
                num_nm_iters=[0],
                num_aprods = [0],
                name = "IC0",
                shift=shift
                )
    prob = ODEProblem(odefun_cg_testing_shift, ψδ, tspan, odeparam5)
    
    # Solve DAE using Tsit5()
    print("\nTime for no PC Solve for 20 iterations:")
    @time sol = solve(prob, Tsit5(); dt=0.2,
            abstol = 1e-5, reltol = 1e-5, save_everystep=true, gamma = 0.2,
            internalnorm=(x, _)->norm(x, Inf),  maxiters=max_its)

    n = odeparam5.counter[1]
    print("\nAvg Num Iters: $(odeparam5.num_iters[1]/n)")
    print("\nAvg Num NM Iters: $(odeparam5.num_nm_iters[1]/n)")
    print("\nAvg Num A Prods: $(odeparam5.num_aprods[1]/n)")

   =#
end

main()


