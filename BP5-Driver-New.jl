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

include("./BP5_ops.jl")
include("./odefun_BP5.jl")
include("../utils.jl") # get 3d metrics and ops
include("../utils_3D.jl") # actual utils file

# TODO: Remove this eventually
global const localARGS = ["./bp5_3D.dat"]
#b global const pth_glob = "./bp5_3D.dat"


function main()


    # Read in params from DAT file for problem
    (pth, stride_space, stride_time, SBPp,
     xc, yc, zc,
     Hx, Hy, Hz, 
     Nx, Ny, Nz, 
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
    λ_f(x, y, z, B_p) = cs^2 * ρ
    μ_f(x, y, z, B_p) = cs^2 * ρ 
    K = 0 # Doesnt get used in metrics, but is asked as input
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
    @time (M, B, JH, A, S, HqI, HrI, HsI, T, e) = locoperator(SBPp, Nq, Nr, Ns, metrics, metrics.C) # TODO: extraneaous C from metrics in there
    print("\nCreating Operators Done\n") 

    print("\nGetting LU Factorization of M\n")
    @time M = lu(M) # matrix factorization
    print("\nLU Factorization of M done\n")

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

    # set b for inital displacement calc
    bdry_vec_strip!(b, B, δ ./ 2, remote_boundary, params)
    # print(b)
    # Calculate initial displacement t = 0
    u = M \ b

    # Following vectors, τ, RSa, θ will only apply to Face 1, and are size 1x(NspxNrp)
    # initialize change in shear stress due to quasi-static deformation
   

    # Set friction coefficients for rate and state
    fault_y = metrics.facecoord[2][1] 
    RS_params = RSht, RSl, RSlf, RSw, RSWf, RShs, RSH, RSamin, RSamax, RSDc, RSVinit
    grid_params = (xc[1]:dx:xc[2], yc[1]:dy:yc[2], zc[1]:dz:zc[2],
                    Nqp, Nrp, Nsp)

    # Set initial state variable according to benchmark
    θ, RS_indices, Nθ = set_theta(RS_params, grid_params)
    # Initialize psi version of state variable
    ψ = RSf0 .+ RSb .* log.(RSV0 .* θ ./ RSDc)

    # Update friction coefficients based on RS zone
    RSa = initialize_friction_params_vec(RS_params, grid_params, Nθ, RS_indices)
    
    # Set pre-stress according to benchmark

    # A bit tricky, τ has y and z comp.  scalar pres stress initialized according to BP5 eq 22
    τ0 = σn .* RSa .* asinh.((RSVinit / (2 * RSV0)) .* exp.((RSf0 + RSb * log.(RSV0 / RSVinit)) ./ RSa)) .+ (η * RSVinit)
    
    
    Δτ_vec = zeros(2 * length(τ0)) # this will be how stresses change through sim
    τ0_vec = zeros(length(Δτ_vec))

    RSVzero = 10e-20 # TODO move this into DAT file
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
    set_prestress_QD!(τ0_vec, RS_params, grid_params, τ_params, Nθ, RS_indices)

    # Set initial condition for index 1 DAE - this is a stacked vector of psi, followed by slip
    # Can ask brittany if this is ok but I think it should work
    # TODO
    ψδ = zeros(Nθ + (2* Nrp * Nsp))  #because length(ψ) = 1 * Nrp * Nsp,  length(δ) = 2 * Nrp * Nsp 
    ψδ[1:Nθ] .= ψ
    ψδ[Nθ+1:end] .= δ

    # Set fault station locations (depths) specified in benchmark
    # TODO
    # I think these are all at x = 0
    stations = [(-16.0, 0.0), (0.0, 0.0), (16.0, 0.0)] # km
    station_indices = find_station_index(stations, y, z)
    station_strings = [ "025", "005", "075"] # # TODO fix these :/ 
    # print(station_indices,"\n")

    # TODO Setup the fault location per BP outline

    # set up parameters sent to the right hand side of the DAE:
    odeparam = (reject_step = [false], 
                sim_years =  sim_years,
                Vp=Vp,
                M = M,
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
                RS_indices = RS_indices
                )
    # Set time span over which to solve:
    tspan = (0, sim_years * year_seconds)
    # print(u)
    # Set up ODE problem corresponding to DAE
    prob = ODEProblem(odefun, ψδ, tspan, odeparam)

    print(station_indices)
    print(RS_indices)

    flt_loc = [y[RS_indices[1, 1]:stride_space:RS_indices[1, 2]]; z[RS_indices[2, 1]:stride_space:RS_indices[2, 2]]]
    flt_loc_indices = RS_indices

    
    # Set call-back function so that files are written to after successful time steps only.
    cb_fun = SavingCallback((ψδ, t, i) -> write_to_file_BP5(pth, ψδ, t, i, y, z, flt_loc, flt_loc_indices,station_strings, station_indices, odeparam, "BP5_", 0.1 * year_seconds), SavedValues(Float64, Float64))

    # Start here getting all this machinery working : ()
    # Make text files to store on-fault time series and slip data,
    # Also initialize with initial data:
    create_text_files(pth, flt_loc, flt_loc_indices, stations, station_strings, station_indices, 0, RSVinit, δ, τ0[1], θ, y, z)
    
    # Solve DAE using Tsit5()
    sol = solve(prob, Tsit5(); dt=0.2,
            abstol = 1e-5, reltol = 1e-5, save_everystep=true, gamma = 0.2,
            internalnorm=(x, _)->norm(x, Inf), callback=cb_fun)        
    # (sol, z, pth)

    # READ THIS PLZ
    # I think we're good on the driver script.... now onto the ODE function

    print("size of T1x_1: ", size([T[1] T[2] T[3]]), "\n")
end

main()


