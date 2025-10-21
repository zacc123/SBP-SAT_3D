include("diagonal_sbp.jl")

using SparseArrays
using LinearAlgebra
using SparseArrayKit
using Interpolations

# Zac's Addition
using Base.Threads


function interp1(xpt, ypt, x)

  knots = (xpt,) 
  itp = interpolate(knots, ypt, Gridded(Linear()))
  #itp[x]  # endpoints of x must be between xpt[1] and xpt[end]
end

# havent adjusted for 3d yet
# big Adjustment we'll write it like this 0 0 Ys
                                        # 0 0 Zs
function create_text_files(pth, flt_loc_y, flt_loc_z, flt_loc_indices, stations, station_strings, station_indices, t, V0_2, V0_3, δ,τ0, θ, yf, zf)

  
    path_to_slip = pth * "slip.dat"
    # slip.dat is a file that stores time, max(V) and slip at all the stations:
    open(path_to_slip, "w") do io
        write(io,"0.0 0.0 ")
        for i in eachindex(flt_loc_y)
            for j in eachindex(flt_loc_z)
                write(io,"$(flt_loc_y[i]) ")
            end
        end
            write(io,"\n")
    
        write(io,"0.0 0.0 ")
        for i in eachindex(flt_loc_y)
            for j in eachindex(flt_loc_z)
                write(io,"$(flt_loc_z[j]) ")
            end
        end
            write(io,"\n")
    end
    
    #write out initial data into devol.txt:
    y_indices = flt_loc_indices[1,1]:flt_loc_indices[1,2]
    z_indices = flt_loc_indices[2,1]:flt_loc_indices[2,2]
    Nzp_virtual = length(flt_loc_z)

    Nyp = length(yf)
    Nzp = length(zf)

    ny_fault = length(y_indices)
    nz_fault = length(z_indices)

    vv = Array{Float64}(undef, 1, 2+ 1*(ny_fault * nz_fault))
        vv[1] = t
        vv[2] = log10(V0_2)

        y_offset = flt_loc_indices[1, 1]
        z_offset = flt_loc_indices[2, 1]
        for i in eachindex(flt_loc_y)
            for j in eachindex(flt_loc_z)
                virtual_idx = 2 + (i - 1) * Nzp_virtual + j
                real_idx =  2 + (i + y_offset - 1) * Nzp + j + z_offset
                vv[virtual_idx] = δ[real_idx]
            end
        end
     
        open(path_to_slip, "a") do io
            writedlm(io, vv)
        end

  # write out initial data into station files:

  # fltst_dpXXX.txt is a file that stores time and time-series of slip, log10(slip_rate), 
  # shear_stress and log10(state) at depth of z = XXX km, where XXX is each of the fault station depths.
  # First we write out initial data into each fltst_dpXXX.txt:

  for n = 1:length(station_strings)
        y_idx = station_indices[n, 1]
        z_idx = station_indices[n, 2]
        real_idx = (y_idx - 1) * Nzp + z_idx
        virtual_idx = (y_idx - 1 - y_offset) * Nzp_virtual + z_idx - z_offset

        XXX = pth * "fltst_strk"*station_strings[n]*".txt"
        ww = Array{Float64}(undef, 1, 8)
        ww[1] = t
        ww[2] = δ[real_idx] #slip y
        ww[3] = δ[real_idx + (Nzp * Nyp)] # slip z
        ww[4] = log10(abs(V0_2))
        ww[5] = log10(abs(V0_3))
        ww[6] = τ0[virtual_idx]
        ww[7] = τ0[virtual_idx + length(θ)]
        ww[8] = log10(abs(θ[virtual_idx]))  # state
        open(XXX, "w") do io
        write(io, "# problem=SEAS Benchmark BP5-QD\n")  # 
        write(io, "# code=Thrase\n")
        write(io, "# modeler=B. A. Erickson & Z. A. Cross\n")
        write(io, "# date=2023/01/09\n")
        write(io, "# element size=xx m\n")
        write(io, "# location=on fault, z = "*string(parse(Int64, station_strings[n])/10)*" km\n")
        write(io, "# Lz = 128 km\n")
        write(io, "t slip_2 slip_3 slip_rate_2 slip_rate_3 shear_stress_2 shear_stress_3  state\n")
        writedlm(io, ww)
    end
  end

end

# havent adjusted for 3d yet
function write_to_file_BP5(pth, ψδ, t, i, yf, zf, flt_loc_y, flt_loc_z, flt_loc_indices, station_strings, station_indices, p, base_name="", tdump=100)
  
  path_to_slip = pth * "slip.dat"
  Vmax = 0.0

  # All of this is to get the right indices to work out agh
    Nyp = length(yf)
    Nzp_virtual = length(flt_loc_z)
    Nzp = length(zf)

    N = Nyp * Nzp


  if isdefined(i,:fsallast) 
    Nθ = p.Nθ
    dψV = i.fsallast
    dψ = @view dψV[1:Nθ]
    V = @view dψV[Nθ .+ (1:2*N)]
    Vmax = maximum(abs.(extrema(V)))
    δ = @view ψδ[Nθ .+ (1:2*N)]
    ψ = @view ψδ[1:Nθ ]
    τf = p.τf
  
 
    θ = (p.RSDc .* exp.((ψ .- p.RSf0) ./ p.RSb)) / p.RSV0  # Invert ψ for θ.
  
    

    if mod(ctr[], p.save_stride_fields) == 0 || t == (p.sim_years ./ 31556926)
      vv = Array{Float64}(undef, 1, 2+(length(flt_loc_y) * length(flt_loc_z)))
      vv[1] = t
      vv[2] = (Vmax)

      # a bit tricky in 3d
      # Might regret this but lets store these indices as 1, 2 -> t, log10(vmax), 
      # then 3, 4 -> (y1, z1), 5, 6 -> (y1, z2) ... etc
      #
        y_offset = flt_loc_indices[1, 1]
        z_offset = flt_loc_indices[2, 1]
        for i in eachindex(flt_loc_y)
            for j in eachindex(flt_loc_z)
                virtual_idx = 2 + (i - 1) * Nzp_virtual + j
                real_idx =  2 + (i + y_offset - 1) * Nzp + j + z_offset
                vv[virtual_idx] = δ[real_idx]
                # vv[virtual_idx + 1] = δ[real_idx]
            end
        end

        open(path_to_slip, "a") do io
            writedlm(io, vv)
        end


        for i = 1:length(station_strings)
            y_idx = station_indices[i, 1]
            z_idx = station_indices[i, 2]
            real_idx = (y_idx - 1) * Nzp + z_idx
            virtual_idx = (y_idx - 1 - y_offset) * Nzp_virtual + z_idx - z_offset
            
            ww = Array{Float64}(undef, 1, 8)
            ww[1] = t

            ww[2] = δ[real_idx] # y comp
            ww[3] = δ[real_idx + N] # z comp

            ww[4] = log10(abs(V[real_idx])) # y comp
            ww[5] = log10(abs(V[real_idx + N])) # z comp

            ww[6] = τf[virtual_idx]
            ww[7] = τf[virtual_idx + Nθ]
            ww[8] = log10(abs(θ[virtual_idx]))

            XXX = pth * "fltst_strk"*station_strings[i]*".txt"
            open(XXX, "a") do io
                writedlm(io, ww)
            end
        end
      
    end
  
    global ctr[] += 1
  end

  Vmax
end

# find_ind() differentiates b/t phases by defining
# interseismic when max slip rate < 10^-3 m/s
# mv is maximum slip rate (log10 m/s) 
function find_ind(mv)
  ind = [1]
  int = 1
  cos = 0
  for i = 2:length(mv)
    if mv[i] > -3 && int == 1 && cos == 0
      append!(ind, i);
      int = 0;
      cos = 1;
    end
  
    if mv[i] < -3 && int == 0 && cos == 1
      append!(ind, i-1)
      int = 1
      cos = 0
    end
  end


  ind = append!(ind, length(mv));  #tack on for plotting any part of an incomplete coseismic/interseismic phase
  
  return ind
end


# plot_slip will plot slip contours from devol.txt - every 5 years in blue during interseismic, 
# every 1 second in red during coseismic
function plot_slip_3D(filename)

    grid = readdlm(filename, Float64)
    sz = size(grid)
    flt_loc_y = grid[1,3:end]
    flt_loc_z = grid[2,3:end]
    T = grid[3:sz[1],1]
    maxV = grid[3:end, 2]
    slip = grid[3:sz[1], 3:sz[2]]
    N = size(slip)[2]


    ind = find_ind(maxV);        #finds indices for inter/co-seismic phases
    interval = [5*31556926 1]   #plot every 5 years and every 1 second
    
    ct = 0   #this counts the number of events


    #Assumes an initial interseismic period
    #This for-loop only plots completed phases
    for i = 1:2:length(ind)-2
        
        T1 = T[ind[i]]:interval[1]:T[ind[i+1]];

        W1 = interp1(T,slip[:,1],T1)';
        
        for j = 2:N 
        w1 = interp1(T,slip[:,j],T1)';
        W1 = [W1; w1]
        end

        if i == 1
        plot(W1, flt_loc_y, linecolor = :blue, legend = false) #interseismic phase
        else
        plot!(W1, flt_loc_y, linecolor = :blue, legend = false) #interseismic phase
        end

    
        T1 = T[ind[i+1]]:interval[2]:T[ind[i+2]];


        W1 = interp1(T,slip[:,1],T1)';
        for j = 2:N 
        w1 = interp1(T,slip[:,j],T1)';
        W1 = [W1; w1]
        end

        plot!(W1, flt_loc_y, linecolor = :red, legend = false) #interseismic phase

        ct = ct+1;
    end

    
    # plot remainder of an incomplete interseismic period:
    i = length(ind)-1;
    T1 = T[ind[i]]:interval[1]:T[ind[i+1]];
    W1 = interp1(T,slip[:,1],T1)';
    print(W1)
    # Quick pull out Ys
    #=
    for i in eachindex(flt_loc_y)
        W1_y[i] = slip[W1[(i-1)*length(flt_loc_z) + 1]]
    end
    =#
        for j = 2:N 
            w1_y = interp1(T,slip[:,j],T1)';
            W1 = [W1; w1_y]
        end
        if i == 1
            plot(W1, flt_loc_y, linecolor = :blue, legend = false) #interseismic phase
        else
            plot!(W1, flt_loc_y, linecolor = :blue, legend = false) #interseismic phase
        end

        xlabel!("Cumulative Slip (m)")
        ylabel!("Fault Position (Y) (km)")
        title!("Slip at Depth z = 0km")
        png("./output/slip.png")
end

# plot_slip will plot slip contours from devol.txt - every 5 years in blue during interseismic, 
# every 1 second in red during coseismic
function plot_traction_3D(filename)

    # read in the grid
    grid = readdlm(filename, Float64; comments=true)
    sz = size(grid)


    time = grid[2:end, 1] ./ 31556926 # grab time in years
    δy =   grid[2:end, 2] 
    δz =   grid[2:end, 3] 
    vy =   grid[2:end, 4]
    vz =   grid[2:end, 5]
    τy =   grid[2:end, 6]
    τz =   grid[2:end, 7]
    θ  =   grid[2:end, 8]

    filename_prefix = split(filename, ".t")[1]
    
    plot(time, δy)
    xlabel!("time (yrs)")
    ylabel!("slip y (m)")
    title!("Slip in Y Direction vs Time at $(split(filename_prefix, "_")[2])")
    png("$(filename_prefix)_slip_y.png")

    plot(time, δz)
    xlabel!("time (yrs)")
    ylabel!("slip z (m)")
    title!("Slip in Z Direction vs Time at $(split(filename_prefix, "_")[2])")
    png("$(filename_prefix)_slip_z.png")

    plot(time, vy)
    xlabel!("time (yrs)")
    ylabel!("slip rate y (m/s)")
    title!("Slip Rate in Y Direction vs Time at $(split(filename_prefix, "_")[2])")
    png("$(filename_prefix)_sliprate_y.png")

    plot(time, vz)
    xlabel!("time (yrs)")
    ylabel!("slip rate z (m/s)")
    title!("Slip Rate in Z Direction vs Time at $(split(filename_prefix, "_")[2])")
    png("$(filename_prefix)_sliprate_z.png")

    plot(time, τy)
    xlabel!("time (yrs)")
    ylabel!("τ-y")
    title!("Stress in Y Direction vs Time at $(split(filename_prefix, "_")[2])")
    png("$(filename_prefix)_stress_y.png")

    plot(time, τz)
    xlabel!("time (yrs)")
    ylabel!("τ-z")
    title!("Stress in Z Direction vs Time at $(split(filename_prefix, "_")[2])")
    png("$(filename_prefix)_stress_z.png")

    plot(time, θ)
    xlabel!("time (yrs)")
    ylabel!("θ")
    title!("RS State vs Time at $(split(filename_prefix, "_")[2])")
    png("$(filename_prefix)_state.png")
end

    

    

