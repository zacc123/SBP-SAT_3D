#=
Helper functions for use in BP5 Benchmarks
=#

# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
# 
#
# Domain and Operator Helpers
#
#
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #

# U is stacked so U = [U1; U2; U3], U1 = [U_111, U_112, U_113, U_121, U_122, ....] for U_xyz

# Start with the Basics, pull out a given constant slice

"""
Helper function to grab a scalar value at position [x, y, z] for U_dir
    Inputs:
        u: stacked vector in x and y and z
        index: tuple (x_in, y_in, z_in, dir)
        num_vals: (Nx, Ny, Nz)
    Output:
        Num 
"""
function get_value(u, indexes, num_vals)
    # Unpack indices
    x, y, z, comp = indexes
   
    Nx, Ny, Nz = num_vals

    # Calc Index
    N = Nx * Ny * Nz
    index = ((comp - 1) * N) + ((x - 1) * (Ny * Nz)) + ((y - 1) * (Nz)) + z

    return u[index]
end

"""
LOL Im re inventing view 
Helper function to grab a vector value at position [x, y, z] for U_dir
    Inputs:
        u: stacked vector in x and y and z
        index: tuple (x_in, y_in, z_in, dir)
            1 set on indices will be a range i.e 1:N
        num_vals: (Nx, Ny, Nz)
    Output:
        Num 
"""

function get_vector(u, indexes, num_vals)
    
    x, y, z, comp = indexes
    Nx, Ny, Nz = num_vals
    
    # Calc Index
    N = Nx .* Ny .* Nz
    index = ((comp .- 1) .* N) .+ ((x .- 1) .* (Ny .* Nz)) .+ ((y .- 1) .* (Nz)) .+ z

    return u[index]
end

# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
# 
#
# Friction and Fault Helpers
#
#
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #




"""
Helper function to get the rate a state parameter correct
"""
function RS_r(y, z, hs, ht, H, l)
    return max(abs(z - hs - ht - H/2) - H/2, abs(y) - l/2) / ht 
end

"""
Function to set the rate and state parameter for the fault face in BP5
"""
function initialize_friction_params_mat(RS_params, grid_params, Nθ, indices)

    _, y_grid, z_grid,
    Nxp, Nyp, Nzp = grid_params
    ht, l, lf, w, Wf, hs, H, a_min, a_max, RSDc, Vinit = RS_params

    # Setup result matrix where values with be a_min, a_max, or r of a
    rows = indices[1, 2] - indices[1, 1] + 1
    cols = indices[2, 2] - indices[2, 1] + 1
    res = zeros(rows, cols)
    Nzp = length(z_grid)

    for row in 1:rows

        for col in 1:cols

            # adjust the indices for the actual coefficient calc, kill myself
            y_idx = row + indices[1, 1] - 1
            z_idx = col + indices[2, 1] - 1

            if abs(y_grid[y_idx]) > lf / 2 || z_grid[z_idx] > Wf
                # Non RS zone
                # print("$((row, col)),  $(y_grid[row]), $(z_grid[col])\n")
               res[row, col] = 0.0

            elseif abs(y_grid[y_idx]) <= lf / 2 && z_grid[z_idx] <= Wf
                
                if (z_grid[z_idx] <= hs) || (z_grid[z_idx] >= hs + H + 2*ht) || (abs(y_grid[y_idx]) >= l/2 + ht)
                    # Get entire VS region
                    res[row, col] = a_max
                elseif (z_grid[z_idx] >= hs + ht && z_grid[z_idx] <= hs + ht + H) && (abs(y_grid[y_idx]) < l/2)
                    # VW and NZ
                    res[row, col] = a_min
                else
                    # Transition Region
                    res[row, col] = (RS_r(y_grid[y_idx], z_grid[z_idx], hs, ht, H, l) * (a_max - a_min)) + a_min

                end
            else
                print("Error in VS Index setup")
            end
        end
    end

    return res

end


"""
Function to set the rate and state parameter for the fault face in BP5
"""
function initialize_friction_params_vec(RS_params, grid_params, Nθ, indices)

    _, y_grid, z_grid,
    Nxp, Nyp, Nzp = grid_params
    ht, l, lf, w, Wf, hs, H, a_min, a_max, RSDc, Vinit = RS_params

    # Setup result matrix where values with be a_min, a_max, or r of a
    rows = indices[1, 2] - indices[1, 1] + 1
    cols = indices[2, 2] - indices[2, 1] + 1
    res = zeros(rows*cols)

    for row in 1:rows

        for col in 1:cols

            # adjust the indices for the actual coefficient calc, kill myself
            y_idx = row + indices[1, 1] - 1
            z_idx = col + indices[2, 1] - 1

            idx = (row - 1) * cols + col

            if abs(y_grid[y_idx]) > lf / 2 || z_grid[z_idx] > Wf
                # Non RS zone
                # print("$((row, col)),  $(y_grid[row]), $(z_grid[col])\n")
               res[idx] = 0.0

            elseif abs(y_grid[y_idx]) <= lf / 2 && z_grid[z_idx] <= Wf
                
                if (z_grid[z_idx] <= hs) || (z_grid[z_idx] >= hs + H + 2*ht) || (abs(y_grid[y_idx]) >= l/2 + ht)
                    # Get entire VS region
                    res[idx] = a_max
                elseif (z_grid[z_idx] >= hs + ht && z_grid[z_idx] <= hs + ht + H) && (abs(y_grid[y_idx]) < l/2)
                    # VW and NZ
                    res[idx] = a_min
                else
                    # Transition Region
                    res[idx] = (RS_r(y_grid[y_idx], z_grid[z_idx], hs, ht, H, l) * (a_max - a_min)) + a_min

                end
            else
                print("Error in VS Index setup")
            end
        end
    end

    return res

end


"""
Use a function to set the theta values according to BP5 description on the fault

    Inputs: 
        - RS_params and grid params per the previous functions to get fault data
        - A coefficients to set them correctly
    Output:
        - Theta: 1 x num_nodes in fault where num_nodes will be the rate and state area (VS) area < Nyp x Nzp
        - Indices: [y1, y2;   To keep track of where the area goes from
                    z1, z2]   Kind of funky, will be 1:N_z dir .+ 1:Nzp:NypxNzp (sub rect at z=0) 
                    
        - Num nodes in theta 

"""
function set_theta(RS_params, grid_params)
    _, y_grid, z_grid,
    Nxp, Nyp, Nzp = grid_params
    ht, l, lf, w, Wf, hs, H, a_min, a_max, RSDc, RSVinit = RS_params

    # STEP 1: Find the size of the RS zone

    # Initialize stoppers
    ny_start = 0
    ny_end = 0
    nz_end = 0

    # Get Y nodes
    for i in eachindex(y_grid)
        if abs(y_grid[i]) <= lf/2 && ny_start == 0 
            ny_start = i
        end
        if abs(y_grid[i]) > lf/2 && ny_end == 0 && ny_start != 0
            ny_end = i-1
            break
        end
    end

    # get z nodes
    for i in eachindex(z_grid)
        if z_grid[i] > Wf
            nz_end = i - 1
            break
        end
    end

    # initialize theta
    θ = RSDc ./ RSVinit .* ones((nz_end) * (ny_end - ny_start + 1))
    
    return (θ, 
            [ny_start ny_end; 1 nz_end;], 
            (nz_end) * (ny_end - ny_start + 1))
end
            
"""
Modify τ term for BP5 Problem setup in Nucleation zone of Rate and State fault

τ is a stacked vector [τy, τz] , but only τy is affected here
"""
function set_prestress_QD!(τ0, RS_params, grid_params, τ_params, Nθ, indices)
    _, y_grid, z_grid,
    Nxp, Nyp, Nzp = grid_params
    ht, l, lf, w, Wf, hs, H, a_min, a_max, RSDc, Vinit = RS_params
    Vi, V0, Vinit, σn, η, RSb, RSf0 = τ_params 

    rows = indices[1, 2] - indices[1, 1] + 1
    cols = indices[2, 2] - indices[2, 1] + 1
    
    for row in 1:rows

        for col in 1:cols
            
            y_idx = row + indices[1, 1] - 1
            z_idx = col + indices[2, 1] - 1
            idx = (row - 1) * cols + col

            if abs(y_grid[y_idx]) > lf / 2 || z_grid[z_idx] > Wf
                # in non rs zone 
                nothing

            elseif abs(y_grid[y_idx]) <= lf / 2 && z_grid[z_idx] <= Wf
                
                if (z_grid[z_idx] <= hs) || (z_grid[z_idx] >= hs + H + 2*ht) || (abs(y_grid[y_idx]) >= l/2 + ht)
                    # Get entire VS region
                    nothing
                # GET Nucleation Zone here: First check Z requirements, then 1sided NZ
                # Check with Brittany about this too 
                #TODO
                elseif (z_grid[z_idx] >= hs + ht && z_grid[z_idx] <= hs + ht + H) && (y_grid[y_idx] >= -l/2 && y_grid[y_idx] <= -l/2 + w)
                    # Update τ0
                    τ0[idx] = σn * a_min * asinh( (Vi / (2*V0)) * exp((RSf0 + RSb * log(V0 / Vinit)) / a_min) ) + (η * Vi)
                    
                else
                    # Transition Region + VW not in W
                    nothing

                end
            else
                print("Error in VS Index setup")
            end
        end
    end

end


"""
Function to set the rate and state τ for the fault face in BP5 after a traction update
"""
function update_tau_v_vec(τ_full, v_full, RS_params, grid_params, Nθ, indices)

    _, y_grid, z_grid,
    Nxp, Nyp, Nzp = grid_params
    ht, l, lf, w, Wf, hs, H, a_min, a_max, RSDc, Vinit = RS_params

    # Setup result matrix where values with be a_min, a_max, or r of a
    rows = indices[1, 2] - indices[1, 1] + 1
    cols = indices[2, 2] - indices[2, 1] + 1
    N = rows * cols
    res_t2 = zeros(N)
    res_t3 = zeros(N)
    res_v2 = zeros(N)
    res_v3 = zeros(N)
    

    for row in 1:rows

        for col in 1:cols

            # adjust the indices for the actual coefficient calc, kill myself
            y_idx = row + indices[1, 1] - 1
            z_idx = col + indices[2, 1] - 1
            actual_idx = (y_idx - 1) * Nzp + z_idx
            idx = (row - 1) * cols + col

            if abs(y_grid[y_idx]) > lf / 2 || z_grid[z_idx] > Wf
                # Non RS zone
                # print("$((row, col)),  $(y_grid[row]), $(z_grid[col])\n")
               print("Error in VS Index setup")

            elseif abs(y_grid[y_idx]) <= lf / 2 && z_grid[z_idx] <= Wf
        
                res_t2[idx] = τ_full[actual_idx] # set τy
                res_t3[idx] = τ_full[actual_idx + (Nyp * Nzp)] # set τz since they're stacked
                res_v2[idx] = v_full[actual_idx] # set τy
                res_v3[idx] = v_full[actual_idx + (Nyp * Nzp)]
              
            else
                print("Error in VS Index setup")
            end
        end
    end

    return res_t2, res_t3, res_v2, res_v3

end


"""
Function to set the rate and state τ for the fault face in BP5 after a traction update
"""
function set_v_vec(v_full, RS_params, grid_params, Nθ, indices)

    _, y_grid, z_grid,
    Nxp, Nyp, Nzp = grid_params
    ht, l, lf, w, Wf, hs, H, a_min, a_max, RSDc, Vinit = RS_params

    # Setup result matrix where values with be a_min, a_max, or r of a
    rows = indices[1, 2] - indices[1, 1] + 1
    cols = indices[2, 2] - indices[2, 1] + 1
    N = rows * cols
    res = zeros(2 * N)
    

    for row in 1:rows

        for col in 1:cols

            # adjust the indices for the actual coefficient calc, kill myself
            y_idx = row + indices[1, 1] - 1
            z_idx = col + indices[2, 1] - 1
            actual_idx = (y_idx - 1) * Nzp + z_idx
            idx = (row - 1) * cols + col

            if abs(y_grid[y_idx]) > lf / 2 || z_grid[z_idx] > Wf
                # Non RS zone
                # print("$((row, col)),  $(y_grid[row]), $(z_grid[col])\n")
               print("Error in VS Index setup")

            elseif abs(y_grid[y_idx]) <= lf / 2 && z_grid[z_idx] <= Wf
        
                res[idx] = τ_full[actual_idx] # set τy
                res[idx + N] = τ_full[actual_idx + (Nyp * Nzp)] # set τz since they're stacked
              
            else
                print("Error in VS Index setup")
            end
        end
    end

    return res

end
"""
Taken straight outta Alex's code lets goooo
"""

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

    f_v, df_v = rateandstate_vectorized(V_v, ψ, σn, τ_v, η, RSas, RSV0)
    #print("\nDEBUG: Type of df_v", df_v)
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

"""
Update the actual RS velocity
"""
function update_V_RS_zone!(V, V_updates, RS_params, grid_params, Nθ, indices)

    _, y_grid, z_grid,
    Nxp, Nyp, Nzp = grid_params
    ht, l, lf, w, Wf, hs, H, a_min, a_max, RSDc, Vinit = RS_params
    Vy, Vz = V_updates

    # Setup result matrix where values with be a_min, a_max, or r of a
    rows = indices[1, 2] - indices[1, 1] + 1
    cols = indices[2, 2] - indices[2, 1] + 1
    res_y = zeros(rows*cols)
    res_z = zeros(rows*cols)

    for row in 1:rows

        for col in 1:cols

            # adjust the indices for the actual coefficient calc, kill myself
            y_idx = row + indices[1, 1] - 1
            z_idx = col + indices[2, 1] - 1
            actual_idx = (y_idx - 1) * Nzp + z_idx
            idx = (row - 1) * cols + col

            if abs(y_grid[y_idx]) > lf / 2 || z_grid[z_idx] > Wf
                # Non RS zone
                # print("$((row, col)),  $(y_grid[row]), $(z_grid[col])\n")
               nothing

            elseif abs(y_grid[y_idx]) <= lf / 2 && z_grid[z_idx] <= Wf
        
                V[actual_idx] = Vy[idx] # set τy
                V[actual_idx + (Nyp * Nzp)] = Vz[idx]
                
            else
                print("Error in VS Index setup")
            end
        end
    end

    return [res_y res_z]
end

 # Function that finds the depth-index corresponding to a station location
function find_station_index(stations, y_grid, z_grid)
      numstations = length(stations)
      station_ind = zeros(numstations, 2)
      for i in range(1, stop=numstations)
        station_ind[i, 1] = argmin(abs.(y_grid .- stations[i][1])) # get y indx
        station_ind[i, 2] = argmin(abs.(z_grid .- stations[i][2])) # get z indx
      end
    return Integer.(station_ind)
end
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
# 
#
# File and IO Helpers
#
#
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #

            
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