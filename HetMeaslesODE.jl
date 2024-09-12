using DifferentialEquations
using Statistics
using Plots
using ProgressMeter
using LaTeXStrings

function final_size(Nc, Ncs, H, β, ν, I0)
    # H : Heterogenaity parameter [0,1]
    # Nc : Number of clusters
    # Ncs : Number of suceptible clusters
    # β : Rate of infection 
    # ν : Rate of recovery
    # I0 . Intital condition vector Nc-dim. fraction of infectious 


    #Suceptible fraction
    Sc = Ncs/Nc

    #Intital S0 indexed by cluster
    S0 = zeros(Nc)  
    S0[1:Ncs] .= 1-H*(1-Sc)
    S0[Ncs+1:end] .= H*Sc

    # I0 are given in fractions of susceptibles so compute this to avoid getting more than 100% of infected
    I0 = S0 .* I0
    #Remove I0 from the suceptibles
    S0 = S0 .- I0
    #Remove all from R0
    R0 = -I0.-S0.+1

    #Collect in inital-condition matrix
    u0 = transpose(hcat(S0, I0, R0))
    
    p = [β,ν]
    #Vector u of solution is 3xNc 1 = S and 2 = I, 3 = R

    function SIR_cluster(du, u, p ,t)
        #unpack values
        β = p[1]
        ν = p[2]
        #Modified SIR-model for each cluster
        for i in 1:Nc
            du[1,i] = -0.5*β*u[1,i]*(u[2,i]+ 1/(Nc-1) * (sum(u[2,1:i-1])+sum(u[2,i+1:Nc])))
            du[2,i] = 0.5*β*u[2,i]*(u[1,i]+ 1/(Nc-1) * (sum(u[1,1:i-1])+sum(u[1,i+1:Nc])))- ν*u[2,i]
            du[3,i] = ν*u[2,i]
        end
    end 

    cluster_prob = ODEProblem(SIR_cluster, u0, (0.0,1000.0),p)

    # Define condition: terminate when all elements in the second row (I) are close to zero
    function condition(u, t, integrator)
        return all(abs.(u[2, :]) .< 1e-6)  # Use a small threshold like 1e-6
    end

    # Define affect! to terminate the integrator
    affect!(integrator) = terminate!(integrator)

    # Create the callback
    cb = ContinuousCallback(condition, affect!)

    # Solve the problem with the callback
    sol = solve(cluster_prob,callback=cb)

    #Compute and return the final size
    return sum(sol.u[size(sol.u)[1]][3,:]-sol.u[1][3,:])/(Nc-sum(sol.u[1][3,:]))
end

H_res = 50
H_range = (0:1/H_res:1)
Nc = 50
Ncs_range = 1:Nc

I0_vec = zeros(Nc)
I0_vec[1:2] .= 0.001
#And the rest are connected to 1/(Nc-1) of the initally infected 
I0_vec[3:end] .= 2*0.001/(Nc-1)

final_sizes =zeros(H_res,Nc)


β = 2*1.6
ν = 0.8
@showprogress for i in 1:H_res
    for j in Ncs_range
        final_sizes[i,j] = final_size(Nc, j, H_range[i], β, ν, I0_vec)
    end 
end 


heatmap(final_sizes, title = "R0 = $(β/ν)")

function analytic_R0(H, Ns, Nc, β, ν)
    Sc = Ns/Nc
    R0s = β/ν * (Ns+(Nc-2)*(1-H+H*Sc))/(2*(Nc-1))
    R0i = β/ν * (Ns+(Nc-2)*Sc*H)/(2*(Nc-1))

    return max(R0s,R0i)
end 

r0_calc = zeros(H_res,Nc)

@showprogress for i in 1:H_res
    for j in Ncs_range
        r0_calc[i,j] = analytic_R0(H_range[i], j,Nc, β, ν)
    end 
end 

function sq(ns)
    #return 1-sqrt(1-ns)
    return 2*ns-ns^2
end

function h_parm(H,sc)
    return (H+sc-1)/(2*sc-1)
end

heatmap(range(0,1,length = 50),range(0,1,length = 50),final_sizes, title = "β/ν = $(β/ν)",xlabel=L"s", ylabel=L"H")
contour!(range(0,1,length = 50),range(0,1,length = 50),r0_calc/1000, levels = 1/1000:1/1000, label = "R0(s,H)",c =:red)
vline!([ν/β], color=:blue, linewidth=2,label=L"ν/β")

heatmap(r0_calc./10 .>1/10, title = "R0 = $(β/ν)")

plot(analytic_R0.((0:1/10:1),50,100,β,ν))

