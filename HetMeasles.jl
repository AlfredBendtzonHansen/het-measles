using Random
using Distributions
using DataStructures
using Statistics
using Plots
using ProgressMeter
using JLD2
using LaTeXStrings
#Load a simulation output dataset
#@load "R15N1000M200nr100m025.jld2" mean_fraction_infected R0_mean_matrix Dispersion_matrix prob_of_outbreak final_sizes N M mean_R0 I0 num_simulations

# Parameters
N = 500  # Number of agents
M = 200   # Number of clusters
mean_R0 = 15.0  # Mean of the Poisson distribution for R0
I0 = 1    # Initial number of infected agents
num_simulations = 100  # Number of simulations per parameter combination

# Function to run a single simulation
function run_simulation(N, M, m, p, mean_R0, I0)
    # Initialize agents and clusters
    agents = 1:N
    clusters = [Set{Int}() for _ in 1:M]
    agent_contacts = zeros(Int, N)  # Array to track the number of contacts for each agent
    cluster_sizes = [0 for _ in clusters]
    while !all(cluster_sizes.>0) #make sure no cluster is empty
        # Assign each agent to two randomly chosen clusters
        for agent in agents
            cluster_indices = sample(1:M, 2, replace = false)
            push!(clusters[cluster_indices[1]], agent)
            push!(clusters[cluster_indices[2]], agent)
        end

        # Count contacts of each agent
        for cluster in clusters
            for agent in cluster
                agent_contacts[agent] += (length(cluster) - 1)  # Each agent in a cluster contacts others in the same cluster
            end
        end

        # Extract cluster sizes
        cluster_sizes = [length(cluster) for cluster in clusters]
    end
    # Determine which clusters are susceptible

    susceptible_clusters = randperm(M)[1:m]
    susceptible_agents = Set{Int}()

    for cluster in susceptible_clusters
        union!(susceptible_agents, clusters[cluster])
    end

    # Redistribute a fraction p of susceptible agents
    num_agents_to_redistribute = Int(floor(p * length(susceptible_agents)))
    if num_agents_to_redistribute > length(susceptible_agents)
        num_agents_to_redistribute = length(susceptible_agents)
    end

    # Step 1: Make num_agents_to_redistribute susceptible agents recovered
    redistributed_agents = sample(collect(susceptible_agents), num_agents_to_redistribute, replace=false)
    remaining_susceptible_agents = setdiff(susceptible_agents, redistributed_agents)

    # Step 2: Pick the same number of recovered agents and make them susceptible
    all_agents = Set(agents)
    remaining_recovered_agents = setdiff(all_agents, remaining_susceptible_agents)
    new_susceptible_agents = sample(collect(remaining_recovered_agents), num_agents_to_redistribute, replace=false)
    union!(remaining_susceptible_agents, new_susceptible_agents)
    initial_recovered = setdiff(remaining_recovered_agents, remaining_susceptible_agents)
    recovered_agents = setdiff(remaining_recovered_agents, remaining_susceptible_agents)
    initial_nr_susceptible = length(remaining_susceptible_agents)

    # Assign an R0 value to each agent
    R0_distribution = Poisson(mean_R0)
    agent_R0 = [rand(R0_distribution) for _ in agents]

    # Simulate the infection process
    if length(remaining_susceptible_agents) < 1
       println("STOP!")
    end
    infected_agents = sample(collect(remaining_susceptible_agents), I0)

    function simulate_infection(agent, clusters, agent_R0, recovered_agents)
        contacts = Set{Int}()
        for cluster in clusters
            if agent in cluster
                union!(contacts, cluster)
            end
        end
        num_to_infect = min(agent_R0[agent], length(contacts))
        if num_to_infect > 0
            new_infections = sample(collect(contacts), num_to_infect, replace=false)
            # Track potential secondary infections in S population 
            
            #Remove contacts that are not susceptible
            new_infections = setdiff(new_infections, recovered_agents)

            potential_secondary_infections = length(new_infections)
            return new_infections, potential_secondary_infections
        else
            potential_secondary_infections = 0
            return Set{Int}(), potential_secondary_infections
        end
    end

    step = 0
    R0_series = []
    while !isempty(infected_agents)
        step += 1
        new_infected_agents = Set{Int}()
        est_R0 = Set{Int}()
        for agent in infected_agents
            new_infections, potential_secondary = simulate_infection(agent, clusters, agent_R0, recovered_agents)
            union!(new_infected_agents, new_infections)
            push!(recovered_agents, agent)
            push!(est_R0, potential_secondary)
        end
        R0_step = mean(collect(est_R0))
        push!(R0_series, R0_step)
        infected_agents = new_infected_agents
    end

    infected_susceptible_count = length(setdiff(recovered_agents, initial_recovered))
    fraction_infected = infected_susceptible_count / initial_nr_susceptible

    return fraction_infected, R0_series, cluster_sizes, agent_contacts
end

# Define parameter ranges
p_values = range(0, 1, length=100)
m_ratios = range(1/M, 1, length=100)

# Collect results
mean_fraction_infected = zeros(length(p_values), length(m_ratios))
mean_R0_measure = Any[]
R0_mean_matrix = zeros(length(p_values), length(m_ratios))
Dispersion_matrix = zeros(length(p_values), length(m_ratios))
all_cluster_sizes = Any[]
all_agent_contacts = Int[]
prob_of_outbreak = zeros(length(p_values), length(m_ratios))
final_sizes = zeros(length(p_values), length(m_ratios), num_simulations)

@showprogress for i in 1:length(p_values)
    for j in 1:length(m_ratios)
        p = p_values[i]
        m = Int(floor(m_ratios[j] * M))
        fraction_infected_list = []
        R0_meta = []
        for k in 1:num_simulations
            result, R0_run, cluster_sizes, agent_contacts = run_simulation(N, M, m, p, mean_R0, I0)
            push!(fraction_infected_list, result)
            push!(R0_meta, R0_run)
            append!(all_cluster_sizes, cluster_sizes)
            append!(all_agent_contacts, agent_contacts)
            final_sizes[i,j,k] = result
        end
        # Define probability of outbreak as outbreaks where at least 5% of the susceptible population gets infected
        prob_of_outbreak[i,j] = mean(fraction_infected_list .> 0.05)

        mean_fraction_infected[i, j] = mean(fraction_infected_list)

        # Take mean of uneven length R0 time-series
        max_length = maximum(length.(R0_meta))
        padded_time_series_R0_list = [vcat(r0, fill(missing, max_length - length(r0))) for r0 in R0_meta]
        avg_time_series_R0 = [mean(skipmissing([padded_time_series_R0_list[k][i] for k in 1:num_simulations])) for i in 1:max_length]
        var_time_series_R0 = [var(skipmissing([padded_time_series_R0_list[k][i] for k in 1:num_simulations])) for i in 1:max_length]

        push!(mean_R0_measure, avg_time_series_R0)

        if length(avg_time_series_R0) > 3
            R0_mean_matrix[i,j] = mean(avg_time_series_R0[2:3])
            Dispersion_matrix[i,j] = mean(avg_time_series_R0[2:3].^2/var_time_series_R0[2:3])
        end 
    end
end

#Save output if needed af "name.jld2" file followed by spaced variable names
@save "R15N500M200nr100res100.jld2" mean_fraction_infected R0_mean_matrix Dispersion_matrix prob_of_outbreak final_sizes N M mean_R0 I0 num_simulations


p_values = range(0, 1, length=size(mean_fraction_infected)[1])
m_ratios = range(1/M, 0.25, length=size(mean_fraction_infected)[2])
# Plot results
heatmap(m_ratios, p_values, mean_fraction_infected, xlabel="s (Fraction Susceptible Clusters)", ylabel="p (Heterogeneity)", title="Fraction infected of Susceptible Agents: R0 = $(mean_R0), N = $(N), M = $(M)", color=:viridis)
heatmap(m_ratios, p_values, R0_mean_matrix, xlabel="s (Fraction Susceptible Clusters)", ylabel="p (Heterogeneity)", title="Est. R0*: R0 = $(mean_R0), N = $(N), M = $(M)", color=:viridis)
heatmap(m_ratios, p_values, log10.(Dispersion_matrix), xlabel="s (Fraction Susceptible Clusters)", ylabel="p (Heterogeneity)", title="Log Dipsersion param.: R0 = $(mean_R0), N = $(N), M = $(M)", color=:viridis)



plot(mean_R0_measure[10])

# Plot cluster sizes
histogram(all_cluster_sizes, bins=100, xlabel="Cluster Size", ylabel="Frequency", title="Distribution of Cluster Sizes", legend=false)

# Plot the distribution of number of contacts per agent
histogram(all_agent_contacts, bins=200, xlabel="Number of Contacts", ylabel="Frequency", title="Distribution of Number of Contacts per Agent", legend=false)

#Cluster size distributon given N and M binomial
function cluster_dist(N, M, K)
    p = 1/M + 1/(M-1)
    distri = binomial.(N,K)*p^K*(1-p)^(N-K)
    return distri
end

plot(cluster_dist.(2000,200,0:20))







# Plotting!
@load "R2N1000M200nr100.jld2" mean_fraction_infected R0_mean_matrix Dispersion_matrix prob_of_outbreak final_sizes N M mean_R0 I0 num_simulations
p_values = range(0, 1, length=size(mean_fraction_infected)[1])
m_ratios = range(1/M, 1, length=size(mean_fraction_infected)[2])

heatmap(m_ratios, p_values, R0_mean_matrix, xlabel=L"s", ylabel=L"H", title=L"\textrm{Measured\,\,\,R_0:} C_0 = 2, N = 1000", color=:viridis, dpi = 500)
vline!([1/mean_R0], color=:red, linewidth=2,label=L"1/C_0")
savefig("R2N1000_r0.pdf")

heatmap(m_ratios, p_values, log10.(Dispersion_matrix), xlabel=L"s", ylabel=L"H", title=L"\textrm{Dispersion\,\,\, Parameter:} C_0 = 2, N = 1000", color=:viridis, dpi = 500)
vline!([1/mean_R0], color=:red, linewidth=2,label=L"1/C_0")
savefig("R2N1000_disp.pdf")






heatmap(m_ratios, p_values, prob_of_outbreak, xlabel=L"s", ylabel=L"H", title=L"\textrm{Outbreak\,\,\, Probability:} C_0 = 15, N = 500", color=:viridis, dpi = 500)
vline!([1/mean_R0], color=:red, linewidth=2,label=L"1/C_0")
savefig("R15N500_prob_outbreak.pdf")

#Nice final size, cluster distribution and contact number distribution plots
final_sizes_single = zeros(100000)

@showprogress for k in 1:100000
    result, R0_run, cluster_sizes, agent_contacts = run_simulation(1000, 200, 5, 0.01, 15, 1)
    final_sizes_single[k] = result
end


exp_law(x,c) =  c .* exp.(-c .* x)
x_vals = range(0.001, 0.3, length=100)


histogram(final_sizes_single, bins=100, xlabel=L"\textrm{Final\,\,\,Size}", ylabel=L"\textrm{Frequency}", title=L"H = 0.01, s = 0.025 ", legend=false, dpi = 500, normalize=:pdf)
#plot!(x_vals,exp_law.(x_vals, 27), yscale = :log10)
#savefig("R15m5p001fs.pdf")
#plot!(LinRange(0.1,0.5,100),LinRange(0.1,0.5,100).^(-0.5))



all_cluster_sizes_single = Any[]
all_agent_contacts_single = Int[]
@showprogress for k in 1:100000
    result, R0_run, cluster_sizes, agent_contacts = run_simulation(1000, 200, 5, 0, 0, 1)
    append!(all_cluster_sizes_single, cluster_sizes)
    append!(all_agent_contacts_single, agent_contacts)
end
stephist(all_agent_contacts_single, dpi = 500, normalize=:probability, label = L"\textrm{Sim.}")
d = Binomial(499, 1/100 + 2/(200-1))
plot!(title = L"\textrm{Contacts, N = 1000, M = 200}")
plot!(xlims =(0,40), xlabel = L"\textrm{Nr.\,\,\, of\,\,\, Contacts}", ylabel = L"\textrm{Frequency}")
plot!([i for i in 1:40], pdf.(d,[i for i in 1:40]), label = L"\textrm{Model}")
savefig("n1000m200contacts.pdf")


stephist(all_cluster_sizes_single, normalize=:probability, dpi = 500, label = L"\textrm{Sim.}")
plot!(title = L"\textrm{Cluster\,\,\, Sizes, N = 1000, M = 200}")
plot!(xlims =(0,30), xlabel = L"\textrm{Cluster\,\,\, Size}", ylabel = L"\textrm{Frequency}")
d = Binomial(1000, 1/200 + 1/(200-1))
plot!([i for i in 1:30], pdf.(d,[i for i in 1:30]), label = L"\textrm{Model}")
savefig("n1000m200cluster.pdf")

