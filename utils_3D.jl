#=
TODO fix naming
A Little Confusing RN but this will be the utility functions like write to text file
following Thrase format
=#

using Plots
using SparseArrays
using LinearAlgebra
using DelimitedFiles
using DifferentialEquations
using Interpolations



# havent adjusted for 3d yet
function create_text_files(pth, flt_loc, flt_loc_indices, stations, station_strings, station_indices, t, RSVinit, δ, τz0, θ, yf, zf)

    path_to_slip = pth * "slip.dat"
    # slip.dat is a file that stores time, max(V) and slip at all the stations:
    open(path_to_slip, "w") do io
        write(io,"0.0 0.0 ")
        for i in 1:length(flt_loc)
            write(io,"$(flt_loc[i]) ")
        end
            write(io,"\n")
        end
    
    #write out initial data into devol.txt:
    y_indices = flt_loc_indices[1,1]:flt_loc_indices[1,2]
    z_indices = flt_loc_indices[2,1]:flt_loc_indices[2,2]
    Nzp_virtual = length(flt_loc[2, :])

    Nyp = length(yf)
    Nzp = length(zf)

    ny_fault = length(y_indices)
    nz_fault = length(z_indices)

    vv = Array{Float64}(undef, 1, 2+ 2*(ny_fault * nz_fault))
        vv[1] = t
        vv[2] = log10(RSVinit)

        y_offset = flt_loc_indices[1, 1]
        z_offset = flt_loc_indices[2, 1]
        for i in eachindex(flt_loc[1, :])
            for j in eachindex(flt_loc[2, :])
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
        ww = Array{Float64}(undef, 1, 6)
        ww[1] = t
        ww[2] = δ[real_idx]
        ww[3] = δ[real_idx + (Nzp * Nyp)]
        ww[4] = log10(RSVinit)
        ww[5] = τz0
        ww[6] = log10(θ[virtual_idx])  # 
        open(XXX, "w") do io
        write(io, "# problem=SEAS Benchmark BP5-QD\n")  # 
        write(io, "# code=Thrase\n")
        write(io, "# modeler=B. A. Erickson\n")
        write(io, "# date=2023/01/09\n")
        write(io, "# element size=xx m\n")
        write(io, "# location=on fault, z = "*string(parse(Int64, station_strings[n])/10)*" km\n")
        write(io, "# Lz = 80 km\n")
        write(io, "t slip slip_rate shear_stress state\n")

        writedlm(io, ww)
    end
  end

end

# havent adjusted for 3d yet
function write_to_file_BP5(pth, ψδ, t, i, yf, zf, flt_loc, flt_loc_indices, station_strings, station_indices, p, base_name="", tdump=100)
  
  path_to_slip = pth * "slip.dat"
  Vmax = 0.0

  # All of this is to get the right indices to work out agh
    Nyp = length(yf)
    Nzp_virtual = length(flt_loc[2, :])
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
  
 
    θ = (p.RSDc * exp.((ψ .- p.RSf0) ./ p.RSb)) / p.RSV0  # Invert ψ for θ.
  
    

    if mod(ctr[], p.save_stride_fields) == 0 || t == (p.sim_years ./ 31556926)
      vv = Array{Float64}(undef, 1, 2+(length(flt_loc[1, :]) * length(flt_loc[2, :])))
      vv[1] = t
      vv[2] = log10(Vmax)

      # a bit tricky in 3d
      # Might regret this but lets store these indices as 1, 2 -> t, log10(vmax), 
      # then 3, 4 -> (y1, z1), 5, 6 -> (y1, z2) ... etc
      #
        y_offset = flt_loc_indices[1, 1]
        z_offset = flt_loc_indices[2, 1]
        for i in eachindex(flt_loc[1, :])
            for j in eachindex(flt_loc[2, :])
                virtual_idx = 2 + (i - 1) * Nzp_virtual + j
                real_idx =  2 + (i + y_offset - 1) * Nzp + j + z_offset
                vv[virtual_idx] = δ[real_idx]
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
            
            ww = Array{Float64}(undef, 1, 7)
            ww[1] = t

            ww[2] = δ[real_idx] # y comp
            ww[3] = δ[real_idx + N] # z comp

            ww[4] = log10(V[real_idx]) # y comp
            ww[5] = log10(V[real_idx + N]) # z comp

            ww[6] = τf[virtual_idx]
            ww[7] = log10(θ[virtual_idx])

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