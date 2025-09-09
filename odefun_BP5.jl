const year_seconds = 31556926
global const ctr = Ref{Int64}(1) 

using DifferentialEquations
using Printf

using DelimitedFiles

function odefun(dψV, ψδ, p, t)
  
    Vp = p.Vp
    M = p.M
    u = p.u
    Δτ = p.Δτ
    τf = p.τf
    b = p.b
    μshear = p.μshear
    RSa = p.RSa
    RSb = p.RSb
    σn = p.σn
    η = p.η
    RSV0 = p.RSV0
    τ0 = p.τ0
    RSDc = p.RSDc
    RSf0 = p.RSf0
    Nθ = p.Nθ
    τf = p.τf
    x = p.x 
    y = p.y
    z = p.z
    T = p.T
    e = p.e
    sJ = p.sJ
    RS_params = p.RS_params
    RS_indices = p.RS_indices
    B = p.B
    
    current_time = t ./ 31556926
    print("TIME [YRS] = $(current_time).\n")

    Nqp = length(x)
    Nrp = length(y)
    Nsp = length(z)

    grid_params = (x, y, z, Nqp, Nrp, Nsp)

    ψ  = @view ψδ[(1:Nθ)]
    δ  = ψδ[Nθ .+ (1:2*Nrp*Nsp)]

    b .= 0

    params = (Nqp, Nrp, Nsp)
    remote_boundary = (Vp .* t ./ 2) .* ones(3 * Nqp * Nrp * Nsp) # Slow creep at face 2 

    bdry_vec_strip!(b, B, δ ./ 2, remote_boundary, params)

    # solve for displacements everywhere in domain
    u[:] = M \ b

    # set up rates of change for  state and slip
    dψ  = @view dψV[(1:Nθ)]
    V  = dψV[Nθ .+ (1:2*Nrp*Nsp)]

    dψ .= 0 # initialize values to 0
    V  .= 0 # initialize values to 0

    # Update the fault data
    Δτ .= 0
    lf1 = 1  # fault is at face 1

    # Start here
    Δτ_tmp = computetraction_stripped(T, u, e, sJ) # calc Traction on whole face
    Δτ_2, Δτ_3, V2, V3 = update_tau_v_vec(Δτ_tmp, V, RS_params, grid_params, Nθ, RS_indices)
    
    τf[1:Nθ] .= Δτ_2 .+ τ0[1:Nθ]# restrict to only the RS zone
    τf[1+Nθ:2*Nθ] .= Δτ_3 .+ τ0[1+Nθ:2*Nθ]
    
    print(Δτ_tmp[1:3], "\n")

    # break into comp for easier reading
    τf_2 = τf[1:Nθ]
    print(τf_2[1:3])
    τf_3 = τf[1+Nθ:2*Nθ]

    V_v = hypot.(V2, V3)

    τ_magnitudes = hypot.(τf_2, τf_3) # get these for newton method

    # Newton Bndry method

    # bisection guarded newton's method (pretty much all from Alex's code from here until)
    xL = fill(0.0, length(τ_magnitudes))
    xR = τ_magnitudes ./ η

    # All this comes from Alex's code and is in ops.jl
        # it really shouldnt need tweaks as its all down in the stacked vector
    (V_v_tmp, f_v, iter) = newtbndv_vectorized(rateandstate_vectorized, xL, xR, V_v, ψ, σn, τ_magnitudes, η,
                                    RSa, RSV0; ftol=1e-6, maxiter=500, minchange=0, atolx = 1e-6, rtolx=1e-6)

    # end of bisection guarded newton's method

    # calculating V2_v and V3_v from V_v
    V_v .= V_v_tmp[:]
    V2 .= V_v .* τf_2 ./ τ_magnitudes
    V3 .= V_v .* τf_3 ./ τ_magnitudes
    # end of calculating V2_v and V3_v from V_v

    # rejecting if V2 or V3 has infinite entries
    if !all(isfinite.(V2)) || !all(isfinite.(V3))
        println("V reject")
        reject_step[1] = true
        return
    end

    # or newton's method does not converge
    if iter < 0
        println("iter reject")
        reject_step[1] = true
        return
    end
    # end of rejecting from V2, V3, or iter

    if iter > 1
        @show iter
    end

    # Set Vs
    # Remember that V is [Vy, Vz] since Vx = 0
    V[1:Nrp * Nsp] .= Vp # set all of the region to Vp to start for V2
    V[Nrp * Nsp + 1: end] .= 0  # Set all v3 to 0

    V_updates = (V2, V3)
    
    # Now updated Velocity:
    update_V_RS_zone!(V, V_updates, RS_params, grid_params, Nθ, RS_indices)
    print("\nDEBUG: V2 max:", maximum(abs.(extrema(V2))))
    print("\nDEBUG: V3 max:", maximum(abs.(extrema(V3))))

    
    # Updating ψ based on iteration convergence
    # dψ[n] = (RSb * RSV0 / RSDc) * (exp((RSf0 - ψn) / RSb) - abs(Vn) / RSV0) # BP1
    # dψ .= (RSb * RSV0 / RSL) .* (exp.((RSf0 .- ψ) ./ RSb) .- sqrt.(V2_v.^2 .+ V3_v.^2) ./ RSV0)
    if iter > 0
        dψ .= (RSb * RSV0 ./ RSDc) .* (exp.((RSf0 .- ψ) ./ RSb) .- sqrt.(V2.^2 .+ V3.^2) ./ RSV0)
    else
        dψ .= 0
    end

    

  nothing
end



export odefun