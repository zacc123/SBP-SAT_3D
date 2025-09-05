# Helper functions for BP5-QD and BP5-FD simulations
using DifferentialEquations
using Printf
using Plots
using CUDA
using DelimitedFiles
using JSON

function _unpack(x::NamedTuple)
    kk = keys(x)
    vv = values(x)
    i = 1
    for k in kk
        @eval $k = $vv[$i]
        i += 1
    end
end


macro unpack_namedtuple(arg)
    quote
        _unpack($arg)
    end |> esc
end


struct coefficients
    # Table 1: Parameter values used in this benchmark problem 
    #        # variable names                        # recommended values                
    ρ       # density                               2670 kg/m³
    cs      # shear wave speed                      3.464 km/s
    ν       # Poisson's ratio                       0.25
    a0      # rate-and-state parameter              0.004
    amax    # rate-and-state parameter              0.04
    b0      # rate-and-state parameter              0.03
    σn      # effective normal stress               25 MPa
    L       # critical slip distance                0.14 m/0.13 m 
    Vp      # plate rate                            10⁻⁹ m/s
    Vinit   # initial slip rate                     10⁻⁹ m/s
    V0      # reference slip rate                   10⁻⁶ m/s 
    f0      # reference friction coefficient        0.6
    hs      # width of shallow VS zone              2 km
    ht      # width of VW-VS transition zone        2 km
    H       # width of uniform VW region            12 km
    l       # length of uniform VW region           60 km
    Wf      # width of rate-and-state fault         40 km
    lf      # length of rate-and-state fault        100 km
    w       # width of favorible nucleation zone    12 km
    Δz      # suggested cell size                   1000m
    tf      # final simulation time                 1800 years
end

# default constructor
coefficients() = coefficients(
    2670,                   # ρ
    3.464,                  # cs
    0.25,                   # ν
    0.004,                  # a0
    0.04,                   # amax
    0.03,                   # b0            value for b in this problem
    25,                     # σn
    0.14,                   # L
    1E-9,                   # Vp
    1E-9,                   # Vinit
    1E-6,                   # V0
    0.6,                    # f0
    2,                      # hs
    2,                      # ht
    12,                     # H
    60,                     # l
    40,                     # Wf
    100,                    # lf
    12,                     # w
    1000,                   # Δz in meter, 
    1800                    # tf
)

# initial state variable over the entire fault
# x2 
function θ0_func(Nx, Ny, BP5_coeff::coefficients)
    return fill(BP5_coeff.L / BP5_coeff.Vinit, Nx * Ny)
end

# for BP5, b is set to be the constant value b0
# using b in the variables of the functions below 
# to write more modular code easier to maintain

function f_func(V, θ, a, b, BP5_coeff::coefficients)
    return a * asinh(
        V / (2 * BP5_coeff.V0) * exp((BP5_coeff.f0 + b * ln(BP5_coeff.V0 / BP5_coeff.L)) / a) # is b the value b0 in coefficients?
    )
end

function a_func(x2, x3, BP5_coeff::coefficients)
    if (BP5_coeff.hs + BP5_coeff.ht ≤ x3 ≤ BP5_coeff.hs + BP5_coeff.ht + BP5_coeff.H) && (abs(x2) ≤ BP5_coeff.l / 2)
        return BP5_coeff.a0
    elseif (0 ≤ x3 ≤ BP5_coeff.hs) || (BP5_coeff.hs + 2 * BP5_coeff.ht + BP5_coeff.H ≤ x3 ≤ BP5_coeff.Wf) || (BP5_coeff.l / 2 + BP5_coeff.ht ≤ abs(x2) ≤ BP5_coeff.lf / 2)
        return BP5_coeff.amax
    else
        r = max(abs(x3 - BP5_coeff.hs - BP5_coeff.ht - BP5_coeff.H / 2) - BP5_coeff.H / 2, abs(x2) - BP5_coeff.l / 2) / BP5_coeff.ht
        return BP5_coeff.a0 + r * (BP5_coeff.amax - BP5_coeff.a0)
    end
end

# auxiliary function to determine which region a belongs to
function a_func_region(x2, x3, BP5_coeff::coefficients)
    if (BP5_coeff.hs + BP5_coeff.ht ≤ x3 ≤ BP5_coeff.hs + BP5_coeff.ht + BP5_coeff.H) && (abs(x2) ≤ BP5_coeff.l / 2)
        return 0
    elseif (0 ≤ x3 ≤ BP5_coeff.hs) || (BP5_coeff.hs + 2 * BP5_coeff.ht + BP5_coeff.H ≤ x3 ≤ BP5_coeff.Wf) || (BP5_coeff.l / 2 + BP5_coeff.ht ≤ abs(x2) ≤ BP5_coeff.lf / 2)
        return 2
    else
        r = max(abs(x3 - BP5_coeff.hs - BP5_coeff.ht - BP5_coeff.H / 2) - BP5_coeff.H / 2, abs(x2) - BP5_coeff.l / 2) / BP5_coeff.ht
        return 1
    end
end

# For BP5-QD, the scalar pre-stress τ⁰ is chosen as the steady-state stress
function τ0_QD_func(a, b, η, BP5_coeff::coefficients)
    return BP5_coeff.σn * a * asinh(BP5_coeff.Vinit / (2 * BP5_coeff.V0) * exp((BP5_coeff.f0 + b) / a)) + η * BP5_coeff.Vinit
end

# For BP5-QD, the scalar pre-stress τ⁰ is chosen as the steady-state stress
function τ0_FD_func(a, b, BP5_coeff::coefficients)
    return BP5_coeff.σn * asinh(BP5_coeff.Vinit / (2 * BP5_coeff.V0) * exp((BP5_coeff.f0 + b) / a))
end

# a higher pre-stress along the x2-direction
function τi0_FD_func(a, b, δ, τ, BP5_coeff::coefficients)
    return BP5_coeff.σn * asinh(BP5_coeff.Vinit / (2 * BP5_coeff.V0) * exp((BP5_coeff.f0 + b) / a)) + δ * τ
end


# quasi-static process zone
function Λ0_func(C, b, μ, BP5_coeff::coefficients)
    return C * μ * BP5_coeff.L / (b * BP5_coeff.σn)
end

# nucleation zone
function h_func(a, b, μ, BP5_coeff::coefficients)
    return π / 2 * (μ * b * BP5_coeff.L) / ((b - a)^2 * BP5_coeff.σn)^2
end

# fault strength
function F_func(f, Vbold, BP5_coeff::coefficients)
    return BP5_coeff.σn * f * Vbold / norm(V)
end

# Zac's Addition to make this more like the rest of thrase
function bdry_vec_strip!(g, B, x, z, slip_data, remote_data, free_surface_data, Lx, Lz)
 
    # reset boundary results
    g[:] .= 0

    # fault (Dirichlet):
    vf = slip_data
    g[:] += B[1] * vf

    # FACE 2 (Dirichlet):
    vf = remote_data
    g[:] += B[2] * vf

    # FACE 3 (Neumann):
    gN = free_surface_data
    vf = gN
    g[:] += B[3] * sJ[3] * vf  #TODO: confirm

    # FACE 4 (Neumann):
    gN = free_surface_data
    vf = gN
    g[:] += B[4] * sJ[4] * vf #TODO: prob error

     # FACE 5 (Neumann):
    gN = free_surface_data
    vf = gN
    g[:] += B[5] * sJ[5] * vf #TODO: prob error

     # FACE 6 (Neumann):
    gN = free_surface_data
    vf = gN
    g[:] += B[6] * sJ[6] * vf #TODO: prob error
  

end


# boundary functions
# Dirichlet boundary conditions
function bc_Dirichlet(face, Vp, δ, x, t)
    if face == 1 # check this
        return (δ ./ 2)
    elseif face == 2 # check the 
        return fill(t * Vp / 2, size(x))
    end
end

# Neumann boundary conditions
function bc_Neumann(face, Vp, δ, x, t)
    return zeros(size(x))
end


function rateandstate(V2, V3, psi, σn, τ2, τ3, η, a, V0)

    V = sqrt(V2^2 + V3^2)
    dV_dV2 = 0.5*(V2^2 + V3^2)^(-0.5) .* 2 .* V2
    dV_dV3 = 0.5*(V2^2 + V3^2)^(-0.5) .* 2 .* V3
    
    Y = (1 ./ (2 .* V0)) .* exp.(psi ./ a)
    f = a .* asinh.(V .* Y)  # compute friction coefficient 
    df_dV2  = a .* (1 ./ sqrt.(1 + (V .* Y).^2)) .* (dV_dV2 .* Y)  # derivative wrt V_2
    df_dV3  = a .* (1 ./ sqrt.(1 + (V .* Y).^2)) .* (dV_dV3 .* Y)  # derivative wrt V_2

    g1 = σn .* f .* V2 / V   + η .* V2 - τ2
    g2 = σn .* f .* V3 / V   + η .* V3 - τ3
   

    A2 = V2/V
    A3 = V3/V

    dA2_dV2 = (V - V2*dV_dV2)/V^2
    dA2_dV3 = (-V2*dV_dV3)/V^2
    dA3_dV2 = (-V3*dV_dV2)/V^2
    dA3_dV3 = (V - V3*dV_dV3)/V^2

    dg1_dV2 = σn .* (df_dV2 .* V2 / V + f .* dA2_dV2) + η
    dg1_dV3 = σn .* (df_dV3 .* V2 / V + f .* dA2_dV3)

    dg2_dV2 = σn .* (df_dV2 .* V3 / V + f .* dA3_dV2) 
    dg2_dV3 = σn .* (df_dV3 .* V3 / V + f .* dA3_dV3) + η 

    return (g1, g2, dg1_dV2, dg1_dV3, dg2_dV2, dg2_dV3)
end
  

function newtbndv(func, x, y; ftol = 1e-12, maxiter = 500, 
                    atolx = 1e-4, rtolx = 1e-4)

    (f, g, dfx, dfy, dgx, dgy) = func(x, y)
    for iter = 1:maxiter

        z = [x; y] 
        (f, g, dfx, dfy, dgx, dgy) = func(x, y)
      
        J = [dfx dfy; dgx dgy] 
        dx, dy = -J\[f; g]
      
        x = x + dx
        y = y + dy
  
       if abs(f) < ftol && abs(dx) < atolx + rtolx * (abs(dx) + abs(x)) && abs(g) < ftol && abs(dy) < atolx + rtolx * (abs(dy) + abs(y))
            
            return (x, y, f, g, iter)
       end
    end
    return (x, y, f, g, -maxiter)
end


function rateandstate_vectorized(V_v, ψ, σn, τ_v, η, RSas, RSV0)
    # V and τ both stand for absolute value of slip rate and traction vecxtors. 
    Y_v = (1 ./ (2 .* RSV0)) .* exp.(ψ ./ RSas)
    f_v = RSas .* asinh.(V_v .* Y_v)
    dfdV_v = RSas .* (1 ./ sqrt.(1 .+ (V_v .* Y_v) .^ 2)) .* Y_v
  
    g_v = σn .* f_v .+ η .* V_v .- τ_v
    dgdV_v = σn .* dfdV_v .+ η
    return (g_v, dgdV_v)
end

function newtbndv_vectorized(rateandstate_vectorized, xL, xR, V_v, ψ, σn, τ_v, η, 
                        RSas, RSV0; ftol=1e-6, maxiter = 500, minchange = 0, atolx = 1e-4, rtolx=1e-4)
    fL_v = rateandstate_vectorized(xL, ψ, σn, τ_v, η, RSas, RSV0)[1]
    fR_v = rateandstate_vectorized(xR, ψ, σn, τ_v, η, RSas, RSV0)[1]

    if any(x -> x > 0, fL_v .* fR_v)
        return (fill(typeof(V_v)(NaN), length(V_v)), fill(typeof(V_v)(NaN), length(V_v)), -maxiter)
    end

    f_v = rateandstate_vectorized(V_v, ψ, σn, τ_v, η, RSas, RSV0)[1]
    df_v = rateandstate_vectorized(V_v, ψ, σn, τ_v, η, RSas, RSV0)[2]
    dxlr_v = xR .- xL

 

    for iter = 1:maxiter
        dV_v = -f_v ./ df_v
        V_v = V_v .+ dV_v
        
        mask = (V_v .< xL) .| (V_v .> xR) .| (abs.(dV_v) ./ dxlr_v .< minchange)
        V_v[mask] .= (xR[mask] .+ xL[mask]) ./ 2
        dV_v[mask] .= (xR[mask] .- xL[mask]) ./ 2

        f_v = rateandstate_vectorized(V_v, ψ, σn, τ_v, η, RSas, RSV0)[1]
        df_v = rateandstate_vectorized(V_v, ψ, σn, τ_v, η, RSas, RSV0)[2]
        
        mask_2 = f_v .* fL_v .> 0
        fL_v[mask_2] .= f_v[mask_2]
        xL[mask_2] .= V_v[mask_2]
        fR_v[.!mask_2] .= f_v[.!mask_2]
        xR[.!mask_2] .= V_v[.!mask_2]

        dxlr_v .= xR .- xL

        if all(abs.(f_v) .< ftol) && all(abs.(dV_v .< atolx .+ rtolx .* (abs.(dV_v) .+ abs.(V_v))))
            return (V_v, f_v, iter)
        end
    end
    return (V_v, f_v, -maxiter)

end


function map_jacobian(x, y, a, b)
    return [x y; a b]
end

function map_jacobian_inv(x, y, a, b)
    return inv([x y; a b])
end

function map_cat(x,y)
    return [x;y]
end

function get_first(a)
    return a[1]
end

function get_second(a)
    return a[2]
end

# Auxiliary functions for debugging
# Plut RS region
function plot_RS(vector)
    spy(reshape(vector, fN2, fN3)')
end
# Plot favorible region
function plot_f(vector)
    spy(reshape(vector, fN2_VW_favorable, fN3_VW)')
end


# Plot the slip in 2D from BP1 problem
function plot_slip(S, δNp, yf, stride_time)

    m = length(yf)
    no_time_steps = size(S.t)
    slip_final = S.u[end][end]

    for i = 1:stride_time:no_time_steps[1]

        slip_t = S.u[i][δNp+1:end] # slip at time t
        #pyplot()
        display(plot(slip_t, -yf, xtickfont=font(18),
            ytickfont=font(18),
            guidefont=font(18),
            legendfont=font(18), ylabel="Depth (km)", xlabel="Slip (m)", xlims=(0, slip_final)))
        sleep(0.1)
    end

    #nothing
end

# Plot slipt against time 
function plot_time_series(path, filenm, region, fieldname, col; headerlines=16, figuresize=(800,600))
    default(fontfamily = "Times", xtickfont = font(20), ytickfont = font(20), legendfont = font(20), guidefont = font(20))

    filename = string(path, filenm)
    A = readdlm(filename, skipstart=headerlines)

    if region == "fault"
        T = A[:, 1] ./ (60*60*24*30) # seconds to years
        slip = A[:, 2]
        V = A[:, 3]
        tau = A[:, 4]
        state = A[:, 5]

        if fieldname == "slip"
            plot(T, slip, color=col, linewidth=2, size=figuresize)
        elseif fieldname == "V"
            plot(T, V, color=col, linewidth=2, size=figuresize)
        elseif fieldname == "10toV"
            plot(T, 10 .^ V, color=col, linewidth=2, size=figuresize)
        elseif fieldname == "tau"
            plot(T, tau, color=col, linewidth=2, size=figuresize)
        else
            plot(T, state, color=col, linewidth=2, size=figuresize)
        end

    elseif region == "surface"
        T = A[:, 1] ./ (60*60*24*30) # seconds to years
        horiz = A[:, 2]
        vert = A[:, 3]
        Vhoriz = A[:, 4]
        Vvert = A[:, 5]

        if fieldname == "horiz"
            plot(T, horiz, color=col)
        elseif fieldname == "vert"
            plot(T, vert, color=col)
        elseif fieldname == "Vhoriz"
            plot(T, Vhoriz, color=col)
        else
            plot(T, Vvert, color=col)
        end
    end

    xlabel!("Time (months)")
    ylabel!(fieldname)
    output_figure_name = split(filename,'.')[1]
    savefig("$output_figure_name.png")
    return
end


########################## Test functions, subjected to changes ################################


function test_unpack(p)
    @unpack_namedtuple p
    @show Vp
    @show RHS
end


function create_text_files(path, station_strings, station_indices, δ, τb, θ, t)
    Vzero = 1e-20
    for n = 1:length(station_strings)
        XXX = path * "fltst_strk" * station_strings[n] * ".txt"
        ww = Array{Float64}(undef, 1, 8)
        ww[1] = t
        RS_index = RS_filter_2D_nzind[station_indices[n]]
        ww[2] = δ[2 * RS_index-1]
        ww[3] = δ[2 * RS_index]
        # ww[4] = log10(BP5_coeff.Vinit)
        ww[4] = log10(abs(V2_v[station_indices[n]]))
        ww[5] = log10(Vzero)
        ww[6] = τb[2 * RS_index - 1] # need to define this
        ww[7] = τb[2 * RS_index]
        ww[8] = log10(θ[station_indices[n]])  # 
        open(XXX, "w") do io
            write(io, "# This is the file header")
            write(io, "# problem=SEAS Benchmark BP5-QD\n")  # 
            write(io, "# code=Thrase\n")
            write(io, "# modeler=B. A. Erickson\n")
            write(io, "# date=2023/01/09\n")
            write(io, "# element size=1000 m\n")
            write(io, "#location=on fault, 0km along strike, 8km away from the fault, 0km depth")
            write(io, "# minimum_time_step=0.1\n")
            write(io, "# maximum_time_step=3.157e6\n")
            write(io, "# num_time_steps=2400\n")
            write(io, "# Column #1 = Time(s)\n")
            write(io, "# Column #2 = Slip_2(m)\n")
            write(io, "# Column #3 = Slip_3(m)\n")
            write(io, "# Column #4 = Slip_rate_2(log10 m/s)\n")
            write(io, "# Column #5 = Slip_rate_3(log10 m/s)\n")
            write(io, "# Column #6 = Shear_stress_2 (MPa)\n")
            write(io, "# Column #7 = Shear_stress_3 (MPa)\n")
            write(io, "# Column #8 = State (log10 s)\n")
            write(io, "t slip_2 slip_3 slip_rate_2 slip_rate_3 shear_stress_2 shear_stress_3 state\n")
            write(io, "# Here is the time-series data.\n")
            writedlm(io, ww)
        end
    end

end



function write_to_file(path, ψδ, t, i, odeparam, station_strings, station_indices)
    @unpack_namedtuple odeparam;
    Vmax = 0.0
    # if t == (sim_years ./ 31556926)
   
    if isdefined(i, :fsallast)
        dψV = i.fsallast
        dψ, V, ψ, δ = create_view(dψV, ψδ)
        if mod(ctr[], odeparam.stride_time) == 0
            for n = 1:length(station_strings)
                XXX = path * "fltst_strk" * station_strings[n] * ".txt"
                ww = Array{Float64}(undef, 1, 8)
                ww[1] = t
                RS_index = RS_filter_2D_nzind[station_indices[n]] # Check if should be just station_indices[n]
                ww[2] = δ[2 * RS_index-1]
                ww[3] = δ[2 * RS_index]
                V2_real = abs((V[2 * RS_index-1]))
                V3_real = abs((V[2 * RS_index]))
                ww[4] = log10(V2_real)
                ww[5] = log10(V3_real)
                ww[6] = τfb[2 * RS_index - 1] # need to define this
                ww[7] = τfb[2 * RS_index]
                θ = BP5_coeff.L / BP5_coeff.V0 * exp((ψ[station_indices[n]] - BP5_coeff.f0)/BP5_coeff.b0)
                ww[8] = log10(θ)
                open(XXX, "a") do io
                    writedlm(io, ww)
                end
            end
        end
        if mod(ctr[], 1000) == 0
            restart_values = (t = t, 
            ψ = ψ,
            δ = δ,
            V = V)
            open(path * "restart_values.json", "w") do file
                JSON.print(file, restart_values)
            end
        end
        global ctr[] += 1
        @show ctr[]
    end
    Vmax
end

function read_params_BP5(f_name)
  f = open(f_name, "r")
  tmp_params = []
  while ! eof(f)
      s = readline(f)
      if s[1] != '#'
          push!(tmp_params, split(s, '=')[2])
          flush(stdout)
      end
  end
  close(f)

 
    #=  (pth, stride_space, stride_time, SBPp
        xc, yc, zc
        Hx, Hy, Hz, 
        Nx, Ny, Nz, 
        ρ, cs, ν, 
        RSamin, RSamax, RSb
        σn, RSDc, Vp,
        RSV0, RSf0, RShs
        RSht, RSH, RSl
        RSlf, W, Δz,
        sim years) = read_params(localARGS[1])
    =#
    params = Vector{Any}(undef, 34)
    params[1] = strip(tmp_params[1]) # pth
    params[2] = parse(Int64, tmp_params[2]) # stride_space
    params[3] = parse(Int64, tmp_params[3]) # stride_time
    params[4] = parse(Int64, tmp_params[4]) # SBPp 
    params[5] = (parse(Float64, tmp_params[5]), parse(Float64, tmp_params[6])) # xc
    params[6] = (parse(Float64, tmp_params[7]), parse(Float64, tmp_params[8])) # yc
    params[7] = (parse(Float64, tmp_params[9]), parse(Float64, tmp_params[10])) # zc
    params[8] = parse(Int64, tmp_params[11]) # Hx
    params[9] = parse(Int64, tmp_params[12]) # Hy
    params[10] = parse(Int64, tmp_params[13]) # Hz
    params[11] = parse(Int64, tmp_params[14]) # Nx
    params[12] = parse(Int64, tmp_params[15]) # Ny
    params[13] = parse(Int64, tmp_params[16]) # Nz
    for i = 17:length(tmp_params)
      params[i-3] = parse(Float64, tmp_params[i])
    end
    
  return params
end

# Helper function to pull out the different fault zone indices to set DIR params
# According to BP5 Problem, description
    # VS = Velocity Strengthening Region (Blue)
    # VW = Velocity Weaking Zone (Green)
    # TZ = Transition zone between VS and VW (Yellow)
    # NZ = Nucleation Zone (Dark Green)
# For more info on actual problem meaning see figure 1 in benchmark: https://strike.scec.org/cvws/seas/download/SEAS_BP5.pdf
#= 
Assumes x->
            y-> 
                z-> stacking where Y is split symmetrically over the origin

Input:
    Fault is at X = LX1, 2D array stacked like above
            We need LY1, LY2, and LZ2 (we assume all the width measurements in z are with respect to LZ1, so we can just do relative comparisons, 
                                                                                                                i.e. z[i] > Wf == z[i] > Wf + LZ1)
    RS_params is a struct or named tuple following:
        RS_Params = (
                        )

Output:
        named tuple of indices assuming a 1D stacked vector

        Will be a bit tricky to implement, so will handle things based on rows where they happen in Y
            For standardization, still write out idx = (row - 1) * Nzp + col, even tho col should always = 1 when setting
=#
function get_fault_indices(fault, RS_params, grid_params)
   
    _, y_grid, z_grid,
    Nxp, Nyp, Nzp = grid_params
    h_t, l, lf, w, W_f, h_s, H, a0, a_max = RS_params

    print(y_grid)
    # Initialize pairs
    VS = [-1, -1]
    TZ = [-1, -1]
    VW = [-1, -1]
    NZ = [-1, -1]

    rows, cols = size(fault)
    for row in 1:rows
        for col in 1:cols

            # Set start of VS Zone
            if abs(y_grid[row]) <= lf / 2 && VS[1] == -1
                VS[1] = (row - 1) * Nzp + col  
                break
            end

            # Set End of VS Zone
            if abs(y_grid[row]) >= lf / 2 && VS[1] != -1 && VS[2] == -1
                VS[2] = (row - 1) * Nzp + col 
                break
            end
    
            # Set start of TZ Zone
            if abs(y_grid[row]) <= l/2 + h_t && TZ[1] == -1
                TZ[1] = (row - 1) * Nzp + col  
                break
            end

            # Set End of TZ Zone
            if abs(y_grid[row]) >= l / 2 + h_t && TZ[1] != -1 && TZ[2] == -1
                TZ[2] = (row - 1) * Nzp + col 
                break
            end
            
            # Set start of VW Zone
            if abs(y_grid[row]) <= l / 2 && VW[1] == -1
                VW[1] = (row - 1) * Nzp + col  
                NZ[1] = (row - 1) * Nzp + col 
                break 
            end
            
            # Set End of VW Zone
            if abs(y_grid[row]) >= l / 2 && VW[1] != -1 && VW[2] == -1
                VW[2] = (row - 1) * Nzp + col
                break
            end

            # Set End of VW Zone
            if abs(y_grid[row]) >= l / 2 - w && NZ[1] != -1 && NZ[2] == -1
                NZ[2] = (row - 1) * Nzp + col
                break
            end

            
        end
    end
    return (VS_indices = VS,
            TZ_indices = TZ,
            VW_indices = VW,
            NZ_indices = NZ)
    
    return nothing
end

"""
Helper function to get the rate a state parameter correct
"""
function RS_r(y, z, hs, ht, H, l)
    return max(abs(z - hs - ht - H/2) - H/2, abs(y) - l/2) / ht 
end

"""
Function to set the rate and state parameter for the fault face in BP5
"""
function initialize_friction_params(fault, RS_params, grid_params)

    _, y_grid, z_grid,
    Nxp, Nyp, Nzp = grid_params
    ht, l, lf, w, Wf, hs, H, a_min, a_max = RS_params

    # Setup result matrix where values with be a_min, a_max, or r of a
    res = zeros(size(fault))

    rows, cols = size(fault)
    
    for row in 1:rows

        for col in 1:cols
            

            if abs(y_grid[row]) > lf / 2 || z_grid[col] > Wf
                # Non RS zone
                # print("$((row, col)),  $(y_grid[row]), $(z_grid[col])\n")
               res[row, col] = 0.0

            elseif abs(y_grid[row]) <= lf / 2 && z_grid[col] <= Wf
                
                if (z_grid[col] <= hs) || (z_grid[col] >= hs + H + 2*ht) || (abs(y_grid[row]) >= l/2 + ht)
                    # Get entire VS region
                    res[row, col] = a_max
                elseif (z_grid[col] >= hs + ht && z_grid[col] <= hs + ht + H) && (abs(y_grid[row]) < l/2)
                    # VW and NZ
                    res[row, col] = a_min
                else
                    # Transition Region
                    res[row, col] = (RS_r(y_grid[row], z_grid[col], hs, ht, H, l) * (a_max - a_min)) + a_min

                end
            else
                print("Error in VS Index setup")
            end
        end
    end

    return res

end
            





