using IterativeSolvers
include("utils.jl")

p = 2
k = 5
Nq = 2^k
Nr = 2^k
Ns = 2^k

# BASIN STRUCTURE
B_p = (λ = 1.0, μ_in = 2, μ_out = 5, c = 1, r̄ = 144, r_w = 20, on = true)

# K + 4μ/3 = λ + 2μ 
# K - 2μ/3 = 

# somewhat basic coordinate transform option
# xf=(q,r,s)->(1 .+ .1 .* sin.(q), 0.1 * cos.(q), zeros(size(r)), zeros(size(s)))
# yf=(q,r,s)->(.5 .* r, zeros(size(q)), .5 * ones(size(r)), zeros(size(s)))
# zf=(q,r,s)->(2 .* s, zeros(size(q)), zeros(size(r)), 2 * ones(size(s)))

# another basic coordinate transform option
xf, yf, zf = transforms_ne(24, 1e12, 1e12, 1e12)

# TODO: FINISH coordinate transform option using transfinite interpolation between corners of a general hexahedron - this is not done yet. 
#  include("transfinite.jl")
#  xcorners = [0   1  0 1  0  1 0 1] #[0  2 0 3 0 2 0 3]
#  ycorners = [-1 -1  1 1 -1 -1 1 1] #[-2 -2  2 2 -2 -2 2 2]
#  zcorners = [-1 -1 -1 -1 1 1 1 1] #[-2 -3 -2 -2 2 2 4 2]
# xf, yf, zf = transfinite_blend_3D(xcorners, ycorners, zcorners)



function μ(x, y, z, B_p)

    c = B_p.c
    μ_in = B_p.μ_in
    μ_out = B_p.μ_out
    r̄ = B_p.r̄
    r_w = B_p.r_w
    on = B_p.on

    if on == false
        return repeat([μ_out], outer=size(x))
    else
        return (μ_out - μ_in)/2 *
            (tanh.((x .^ 2 .+ c^2 * y .^ 2 .- r̄) ./ r_w) .+ 1) .+ μ_in
    end
end

function μ_x(x, y, z, B_p)
    
    c = B_p.c
    μ_in = B_p.μ_in
    μ_out = B_p.μ_out
    r̄ = B_p.r̄
    r_w = B_p.r_w
    on = B_p.on

    if on == false
        return zeros(size(x))  
    else
        return ((μ_out - μ_in) .* x .*
                sech.((x .^ 2 .+ c^2 * y .^ 2 .- r̄) ./ r_w) .^ 2) ./ r_w
    end
end

function μ_y(x, y, z, B_p)

    c = B_p.c
    μ_in = B_p.μ_in
    μ_out = B_p.μ_out
    r̄ = B_p.r̄
    r_w = B_p.r_w
    on = B_p.on

    if on == false
        
        return zeros(size(x))

    else    
        return ((μ_out - μ_in) .* (c^2 * y) .*
            sech.((x .^ 2 + c^2 * y .^ 2 .- r̄) ./ r_w) .^ 2) ./ r_w
    end
end

function μ_z(x, y, z, B_p)

    c = B_p.c
    μ_in = B_p.μ_in
    μ_out = B_p.μ_out
    r̄ = B_p.r̄
    r_w = B_p.r_w
    on = B_p.on

    if on == false
        
        return zeros(size(x))

    else    
        return 0 .* ((μ_out - μ_in) .* (c^2 * y) .*
            sech.((x .^ 2 + c^2 * y .^ 2 .- r̄) ./ r_w) .^ 2) ./ r_w   #zeros!
    end
end

function λ(x, y, z, B_p)
    return B_p.λ .+ zeros(size(x))
end

function λ_x(x, y, z, B_p)
    return zeros(size(x))
end

function λ_y(x, y, z, B_p)
    return zeros(size(x))
end

function λ_z(x, y, z, B_p)
    return zeros(size(x))
end

function K(x, y, z, B_p)
    return λ(x, y, z, B_p) .+ 2 .* μ(x, y, z, B_p) ./ 3
end

function K_x(x, y, z, B_p)
    return λ_x(x, y, z, B_p) .+ 2 .* μ_x(x, y, z, B_p) ./ 3
end

function K_y(x, y, z, B_p)
    return λ_y(x, y, z, B_p) .+ 2 .* μ_y(x, y, z, B_p) ./ 3
end

function K_z(x, y, z, B_p)
    return λ_z(x, y, z, B_p) .+ 2 .* μ_z(x, y, z, B_p) ./ 3
end


BC_DIRICHLET = 1
metrics = create_metrics(1, Nq, Nr, Ns, λ, μ, K, B_p, xf, yf, zf);
nx1 = spdiagm(0 => metrics.nx[1][:])
nx2 = spdiagm(0 => metrics.nx[2][:])
nx3 = spdiagm(0 => metrics.nx[3][:])
nx4 = spdiagm(0 => metrics.nx[4][:])
nx5 = spdiagm(0 => metrics.nx[5][:])
nx6 = spdiagm(0 => metrics.nx[6][:])

ny1 = spdiagm(0 => metrics.ny[1][:])
ny2 = spdiagm(0 => metrics.ny[2][:])
ny3 = spdiagm(0 => metrics.ny[3][:])
ny4 = spdiagm(0 => metrics.ny[4][:])
ny5 = spdiagm(0 => metrics.ny[5][:])
ny6 = spdiagm(0 => metrics.ny[6][:])


nz1 = spdiagm(0 => metrics.nz[1][:])
nz2 = spdiagm(0 => metrics.nz[2][:])
nz3 = spdiagm(0 => metrics.nz[3][:])
nz4 = spdiagm(0 => metrics.nz[4][:])
nz5 = spdiagm(0 => metrics.nz[5][:])
nz6 = spdiagm(0 => metrics.nz[6][:])

  
# SET FACES 1 AND 2 TO BE DIRICHLET
# ALL OTHER FACES ARE TRACTION

(x, y, z) = metrics.coord

((xf1, xf2, xf3, xf4, xf5, xf6), (yf1, yf2, yf3, yf4, yf5, yf6), (zf1, zf2, zf3, zf4, zf5, zf6)) = metrics.facecoord


function exact_u1(x, y, z)
    a = 1
    b = 1
    u1 =    a .+ b .* sin.(x .+ y .+  z)
    u1_x =   b * cos.(x .+ y .+  z)
    u1_y =   b * cos.(x .+ y .+  z)
    u1_z =   b * cos.(x .+ y .+  z)
    u1_xx = -b * sin.(x .+ y .+  z)
    u1_xy = -b * sin.(x .+ y .+  z)
    u1_xz = -b * sin.(x .+ y .+  z)
    u1_yy = -b * sin.(x .+ y .+  z)
    u1_yz = -b * sin.(x .+ y .+  z)
    u1_zz = -b * sin.(x .+ y .+  z)

    return (u1, u1_x, u1_y, u1_z, u1_xx, u1_xy, u1_xz, u1_yy, u1_yz, u1_zz)
end

function exact_u2(x, y, z)
    
    a = 1
    b = 1
    u2 =    a .+ b .* cos.(x .+ y .+  z)
    u2_x =   -b * sin.(x .+ y .+  z)
    u2_y =   -b * sin.(x .+ y .+  z)
    u2_z =   -b * sin.(x .+ y .+  z)
    u2_xx = -b * cos.(x .+ y .+  z)
    u2_xy = -b * cos.(x .+ y .+  z)
    u2_xz = -b * cos.(x .+ y .+  z)
    u2_yy = -b * cos.(x .+ y .+  z)
    u2_yz = -b * cos.(x .+ y .+  z)
    u2_zz = -b * cos.(x .+ y .+  z)


    
    return (u2, u2_x, u2_y, u2_z, u2_xx, u2_xy, u2_xz, u2_yy, u2_yz, u2_zz)
end

function exact_u3(x, y, z)
    
    a = 1
    b = 1
    u3 =    a .+ b .* sin.(0.2x .+ 0.3y .+  0.4z)
    u3_x =   0.2b  * cos.(0.2x .+ 0.3y .+  0.4z)
    u3_y =   0.3b  * cos.(0.2x .+ 0.3y .+  0.4z)
    u3_z =   0.4b  * cos.(0.2x .+ 0.3y .+  0.4z)
    u3_xx = -.04b  * sin.(0.2x .+ 0.3y .+  0.4z)
    u3_xy = -.06b  * sin.(0.2x .+ 0.3y .+  0.4z)
    u3_xz = -.08b * sin.(0.2x .+ 0.3y .+  0.4z)
    u3_yy = -.09b  * sin.(0.2x .+ 0.3y .+  0.4z)
    u3_yz = -.12b * sin.(0.2x .+ 0.3y .+  0.4z)
    u3_zz = -.16b * sin.(0.2x .+ 0.3y .+  0.4z)

   
    return (u3, u3_x, u3_y, u3_z, u3_xx, u3_xy, u3_xz, u3_yy, u3_yz, u3_zz)

end

function source(x, y, z)
    (u1, u1_x, u1_y, u1_z, u1_xx, u1_xy, u1_xz, u1_yy, u1_yz, u1_zz) = exact_u1(x, y, z)
    (u2, u2_x, u2_y, u2_z, u2_xx, u2_xy, u2_xz, u2_yy, u2_yz, u2_zz) = exact_u2(x, y, z)
    (u3, u3_x, u3_y, u3_z, u3_xx, u3_xy, u3_xz, u3_yy, u3_yz, u3_zz) = exact_u3(x, y, z)

    # σxx = λ(x, y, z, B_p)*(u1_x + u2_y + u3_z) + 2μ(x, y, z, B_p)*u1_x
    # σxy = μ(x, y, z, B_p)*(u1_y + u2_x)
    # σxz = μ(x, y, z, B_p)*(u1_z + u3_x)
    # σyy = λ(x, y, z, B_p)*(u1_x + u2_y + u3_z) + 2μ(x, y, z, B_p)*u2_y
    # σyz = μ(x, y, z, B_p)*(u2_z + u3_y)
    # σzz = λ(x, y, z, B_p)*(u1_x + u2_y + u3_z) + 2μ(x, y, z, B_p)*u3_z
    
    
    σxx_x = λ_x(x, y, z, B_p) .* (u1_x + u2_y .+ u3_z) + 2μ_x(x, y, z, B_p) .* u1_x +  λ(x, y, z, B_p) .* (u1_xx .+ u2_xy .+ u3_xz) .+ 2μ(x, y, z, B_p) .* u1_xx
    σxy_y = μ_y(x, y, z, B_p) .* (u1_y + u2_x) .+ μ(x, y, z, B_p) .* (u1_yy .+ u2_xy)
    σxz_z = μ_z(x, y, z, B_p) .* (u1_z + u3_x) .+ μ(x, y, z, B_p) .* (u1_zz .+ u3_xz)

    σxy_x = μ_x(x, y, z, B_p) .* (u1_y + u2_x) .+ μ(x, y, z, B_p) .* (u1_xy .+ u2_xx)
    σyy_y = λ_y(x, y, z, B_p) .* (u1_x + u2_y .+ u3_z) + 2μ_y(x, y, z, B_p) .* u2_y + λ(x, y, z, B_p) .* (u1_xy .+ u2_yy .+ u3_yz) + 2μ(x, y, z, B_p) .* u2_yy
    σyz_z = μ_z(x, y, z, B_p) .* (u2_z + u3_y) .+ μ(x, y, z, B_p) .* (u2_zz .+ u3_yz)

    σxz_x = μ_x(x, y, z, B_p) .* (u1_z + u3_x) .+ μ(x, y, z, B_p) .* (u1_xz .+ u3_xx)
    σyz_y = μ_y(x, y, z, B_p) .* (u2_z + u3_y) .+ μ(x, y, z, B_p) .* (u2_yz .+ u3_yy)
    σzz_z = λ_z(x, y, z, B_p) .* (u1_x + u2_y + u3_z) .+ 2μ_z(x, y, z, B_p) .* u3_z + λ(x, y, z, B_p) .* (u1_xz .+ u2_yz + u3_zz) + 2μ(x, y, z, B_p) .* u3_zz

    
    s1 = σxx_x .+ σxy_y .+ σxz_z
    s2 = σxy_x .+ σyy_y .+ σyz_z
    s3 = σxz_x .+ σyz_y .+ σzz_z
    
    
    return [s1[:];s2[:];s3[:]]
end


function stresses(x, y, z)
  

    (u1, u1_x, u1_y, u1_z, u1_xx, u1_xy, u1_xz, u1_yy, u1_yz, u1_zz) = exact_u1(x, y, z)
    (u2, u2_x, u2_y, u2_z, u2_xx, u2_xy, u2_xz, u2_yy, u2_yz, u2_zz) = exact_u2(x, y, z)
    (u3, u3_x, u3_y, u3_z, u3_xx, u3_xy, u3_xz, u3_yy, u3_yz, u3_zz) = exact_u3(x, y, z)

    σxx = λ(x, y, z, B_p) .* (u1_x + u2_y + u3_z) + 2 .* μ(x, y, z, B_p) .* u1_x
    σxy = μ(x, y, z, B_p) .* (u1_y + u2_x)
    σxz = μ(x, y, z, B_p) .* (u1_z + u3_x)
    σyy = λ(x, y, z, B_p) .* (u1_x + u2_y + u3_z) + 2 .* μ(x, y, z, B_p) .* u2_y
    σyz = μ(x, y, z, B_p) .* (u2_z + u3_y)
    σzz = λ(x, y, z, B_p) .* (u1_x + u2_y + u3_z) + 2 .* μ(x, y, z, B_p) .* u3_z

    
    return (σxx, σxy, σxz, σyy, σyz, σzz)
end

# # IF ALL DIRICHLET BCs 
# g1_1 = exact_u1(xf1, yf1, zf1)[1][:]
# g1_2 = exact_u1(xf2, yf2, zf2)[1][:]
# g1_3 = exact_u1(xf3, yf3, zf3)[1][:]
# g1_4 = exact_u1(xf4, yf4, zf4)[1][:]
# g1_5 = exact_u1(xf5, yf5, zf5)[1][:]
# g1_6 = exact_u1(xf6, yf6, zf6)[1][:]
# g2_1 = exact_u2(xf1, yf1, zf1)[1][:]
# g2_2 = exact_u2(xf2, yf2, zf2)[1][:]
# g2_3 = exact_u2(xf3, yf3, zf3)[1][:]
# g2_4 = exact_u2(xf4, yf4, zf4)[1][:]
# g2_5 = exact_u2(xf5, yf5, zf5)[1][:]
# g2_6 = exact_u2(xf6, yf6, zf6)[1][:]
# g3_1 = exact_u3(xf1, yf1, zf1)[1][:]
# g3_2 = exact_u3(xf2, yf2, zf2)[1][:]
# g3_3 = exact_u3(xf3, yf3, zf3)[1][:]
# g3_4 = exact_u3(xf4, yf4, zf4)[1][:]
# g3_5 = exact_u3(xf5, yf5, zf5)[1][:]
# g3_6 = exact_u3(xf6, yf6, zf6)[1][:]

# DIRICHLET ON FACES 1 and 2, else TRACTION
(σxx3, σxy3, σxz3, σyy3, σyz3, σzz3) = stresses(xf3, yf3,  zf3)
(σxx4, σxy4, σxz4, σyy4, σyz4, σzz4) = stresses(xf4, yf4,  zf4)
(σxx5, σxy5, σxz5, σyy5, σyz5, σzz5) = stresses(xf5, yf5,  zf5)
(σxx6, σxy6, σxz6, σyy6, σyz6, σzz6) = stresses(xf6, yf6,  zf6)

g1_1 = exact_u1(xf1, yf1,  zf1)[1][:]
g1_2 = exact_u1(xf2, yf2,  zf2)[1][:]
g1_3 = nx3*σxx3[:] + ny3*σxy3[:] + nz3*σxz3[:]
g1_4 = nx4*σxx4[:] + ny4*σxy4[:] + nz4*σxz4[:]
g1_5 = nx5*σxx5[:] + ny5*σxy5[:] + nz5*σxz5[:]
g1_6 = nx6*σxx6[:] + ny6*σxy6[:] + + nz6*σxz6[:]

g2_1 = exact_u2(xf1, yf1,  zf1)[1][:]
g2_2 = exact_u2(xf2, yf2,  zf2)[1][:]
g2_3 = nx3*σxy3[:] + ny3*σyy3[:] + nz3*σyz3[:]
g2_4 = nx4*σxy4[:] + ny4*σyy4[:] + nz4*σyz4[:]
g2_5 = nx5*σxy5[:] + ny5*σyy5[:] + nz5*σyz5[:]
g2_6 = nx6*σxy6[:] + ny6*σyy6[:] + nz6*σyz6[:]

g3_1 = exact_u3(xf1, yf1,  zf1)[1][:]
g3_2 = exact_u3(xf2, yf2,  zf2)[1][:]
g3_3 = nx3*σxz3[:] + ny3*σyz3[:] + nz3*σzz3[:]
g3_4 = nx4*σxz4[:] + ny4*σyz4[:] + nz4*σzz4[:]
g3_5 = nx5*σxz5[:] + ny5*σyz5[:] + nz5*σzz5[:]
g3_6 = nx6*σxz6[:] + ny6*σyz6[:] + nz6*σzz6[:]

u1 = exact_u1(x, y, z)[1][:]
u2 = exact_u2(x, y, z)[1][:]
u3 = exact_u3(x, y, z)[1][:]

Uexact = [u1;u2;u3]

Source = source(x, y, z)
(M, B, JH, A, S, HIq, HIr, HIs) = locoperator(p, Nq, Nr, Ns, metrics, metrics.C)


(b11, b12, b13, b21, b22, b23, b31, b32, b33) = B

# ALL DIRICHLET: 
# b = [b11[1]*g1_1 + b11[2]*g1_2 + b11[3]*g1_3 + b11[4]*g1_4 + b11[5]*g1_5 + b11[6]*g1_6 + 
#      b12[1]*g2_1 + b12[2]*g2_2 + b12[3]*g2_3 + b12[4]*g2_4 + b12[5]*g2_5 + b12[6]*g2_6 + 
#      b13[1]*g3_1 + b13[2]*g3_2 + b13[3]*g3_3 + b13[4]*g3_4 + b13[5]*g3_5 + b13[6]*g3_6;

#      b21[1]*g1_1 + b21[2]*g1_2 + b21[3]*g1_3 + b21[4]*g1_4 + b21[5]*g1_5 + b21[6]*g1_6 + 
#      b22[1]*g2_1 + b22[2]*g2_2 + b22[3]*g2_3 + b22[4]*g2_4 + b22[5]*g2_5 + b22[6]*g2_6 + 
#      b23[1]*g3_1 + b23[2]*g3_2 + b23[3]*g3_3 + b23[4]*g3_4 + b23[5]*g3_5 + b23[6]*g3_6;

#      b31[1]*g1_1 + b31[2]*g1_2 + b31[3]*g1_3 + b31[4]*g1_4 + b31[5]*g1_5 + b31[6]*g1_6 + 
#      b32[1]*g2_1 + b32[2]*g2_2 + b32[3]*g2_3 + b32[4]*g2_4 + b32[5]*g2_5 + b32[6]*g2_6 + 
#      b33[1]*g3_1 + b33[2]*g3_2 + b33[3]*g3_3 + b33[4]*g3_4 + b33[5]*g3_5 + b33[6]*g3_6]

# DIRICHLET on faces 1 and 2, else traction
b = [b11[1]*g1_1 + b11[2]*g1_2 + b11[3]*g1_3 + b11[4]*g1_4 + b11[5]*g1_5 + b11[6]*g1_6 + 
      b12[1]*g2_1 + b12[2]*g2_2 +  
      b13[1]*g3_1 + b13[2]*g3_2;

      b21[1]*g1_1 + b21[2]*g1_2  + 
      b22[1]*g2_1 + b22[2]*g2_2 + b22[3]*g2_3 + b22[4]*g2_4 + b22[5]*g2_5 + b22[6]*g2_6 + 
      b23[1]*g3_1 + b23[2]*g3_2;

      b31[1]*g1_1 + b31[2]*g1_2 + 
      b32[1]*g2_1 + b32[2]*g2_2 + 
      b33[1]*g3_1 + b33[2]*g3_2 + b33[3]*g3_3 + b33[4]*g3_4 + b33[5]*g3_5 + b33[6]*g3_6]


#U = M\(b + Source)

JH3 = kron([1 0 0;0 1 0;0 0 1],JH)
fM = JH3*M
fb = JH3*(b + Source)
#e = eigen(-Matrix(JH3*M)).values
#extrema(fM'-fM)

U = cg(fM, fb)

tot_idx = (Nq+1)*(Nr+1)*(Ns+1)

U1 = U[1:tot_idx]
U2 = U[tot_idx+1:2*tot_idx]
U3 = U[2*tot_idx+1:end]

err = sqrt((u1 - U1)'*JH*(u1-U1) + (u2 - U2)'*JH*(u2-U2) + (u3 - U3)'*JH*(u3-U3))
