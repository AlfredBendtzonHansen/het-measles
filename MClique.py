import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from scipy.interpolate import interp1d
from tqdm import tqdm

def generate_clustered_network(N, M, lambda_edges=2.5, lambda_cliques=2.0, nr_cliques=False):
    """
    Generates a random clustered network with cliques of size M.

    Parameters:
        N (int): Total number of nodes in the network.
        M (int): Size of the cliques to be formed (M >= 3).
        lambda_edges (float): Average number of single edges per node.
        lambda_cliques (float): Average number of cliques per node.
        seed (int): Random seed for reproducibility.
        nr_cliques (bool): If True, returns the number of cliques in the network.

    Returns:
        If nr_cliques is True:
            int: Number of cliques in the network.
        Else:
            adj_matrix (numpy.ndarray): Adjacency matrix of the generated network.
            cliques (list of lists): List containing all the cliques (each clique is a list of node indices).
    """

    # Generate degrees for edges and cliques
    # For simplicity, we assume independence between edges and cliques
    edge_degrees = np.random.poisson(lambda_edges, N)
    clique_degrees = np.random.poisson(lambda_cliques, N)

    # Adjust degrees to ensure total stubs are compatible
    # For edges
    if sum(edge_degrees) % 2 != 0:
        idx = np.random.randint(0, N)
        edge_degrees[idx] += 1

    # For cliques
    remainder = sum(clique_degrees) % M
    if remainder != 0:
        for _ in range(M - remainder):
            idx = np.random.randint(0, N)
            clique_degrees[idx] += 1

    # Create edge stubs and clique stubs
    edge_stubs = []
    for node, degree in enumerate(edge_degrees):
        edge_stubs.extend([node] * degree)

    clique_stubs = []
    for node, degree in enumerate(clique_degrees):
        clique_stubs.extend([node] * degree)

    # Shuffle stubs
    np.random.shuffle(edge_stubs)
    np.random.shuffle(clique_stubs)

    # Construct edges by pairing edge stubs
    edges = []
    for i in range(0, len(edge_stubs), 2):
        if i + 1 < len(edge_stubs):
            n1 = edge_stubs[i]
            n2 = edge_stubs[i + 1]
            if n1 != n2:
                edges.append((n1, n2))

    # Construct cliques by grouping clique stubs into groups of size M
    cliques = []
    for i in range(0, len(clique_stubs), M):
        if i + M - 1 < len(clique_stubs):
            nodes = clique_stubs[i:i + M]
            unique_nodes = set(nodes)
            if len(unique_nodes) == M:
                cliques.append(nodes)

    # Combine edges and cliques into adjacency list
    adjacency = {i: set() for i in range(N)}

    # Add edges to adjacency list
    for n1, n2 in edges:
        adjacency[n1].add(n2)
        adjacency[n2].add(n1)

    # Add cliques to adjacency list
    for clique_nodes in cliques:
        for n1 in clique_nodes:
            others = set(clique_nodes) - {n1}
            adjacency[n1].update(others)

    # Reindex nodes to have consecutive indices starting from 0
    remaining_nodes = sorted(adjacency.keys())
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_nodes)}
    num_nodes = len(remaining_nodes)

    # Create adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for old_node, neighbors in adjacency.items():
        new_node = node_mapping[old_node]
        for neighbor in neighbors:
            if neighbor in node_mapping:
                new_neighbor = node_mapping[neighbor]
                adj_matrix[new_node, new_neighbor] = 1
                # For undirected graph, ensure symmetry
                adj_matrix[new_neighbor, new_node] = 1

    if nr_cliques:
        return adj_matrix, cliques, len(cliques)
    else:
        return adj_matrix, cliques


import igraph as ig
from itertools import combinations

def generate_clustered_network_discrete_M(N, M, lambda_edges=2.5, m_cliques=2, nr_cliques=False, no_clique=False):
    """
    Generates a random clustered network with cliques of size M.

    Parameters:
        N (int): Total number of nodes in the network.
        M (int): Size of the cliques to be formed (M >= 3).
        lambda_edges (float): Average number of single edges per node.
        m_cliques (float): Number of cliques per node.
        nr_cliques (bool): If True, returns the number of cliques.
        no_clique (bool): If True, no cliques are added to the network.

    Returns:
        adj_matrix (numpy.ndarray): Adjacency matrix of the generated network.
        cliques (list of lists): List containing all the cliques (each clique is a list of node indices).
        (Optional) number_of_cliques (int): Returned if nr_cliques is True.
    """

    # Generate degrees for edges and cliques
    if no_clique:
        edge_degrees = np.random.poisson(lambda_edges, N) + (M - 1) * nr_cliques
        clique_degrees = np.zeros(N, dtype=int)
    else:
        edge_degrees = np.random.poisson(lambda_edges, N)
        clique_degrees = np.full(N, m_cliques, dtype=int)

    # Adjust edge degrees to ensure the sum is even
    edge_degree_sum = edge_degrees.sum()
    if edge_degree_sum % 2 != 0:
        idx = np.random.randint(N)
        edge_degrees[idx] += 1

    # Adjust clique degrees to ensure total stubs are divisible by M
    clique_degree_sum = clique_degrees.sum()
    remainder = clique_degree_sum % M
    if remainder != 0:
        indices = np.random.choice(N, M - remainder, replace=False)
        clique_degrees[indices] += 1

    # Generate edge graph using the configuration model
    g_edges = ig.Graph.Degree_Sequence(edge_degrees.tolist(), method='configuration')
    g_edges.simplify(multiple=True, loops=True)

    # Initialize the final graph with edges from g_edges
    g = ig.Graph(n=N)
    g.add_edges(g_edges.get_edgelist())

    cliques = []

    if not no_clique:
        # Create clique stubs
        clique_stubs = np.repeat(np.arange(N), clique_degrees)
        np.random.shuffle(clique_stubs)

        # Truncate stubs to a multiple of M
        num_cliques = len(clique_stubs) // M
        clique_stubs = clique_stubs[:num_cliques * M]
        clique_groups = clique_stubs.reshape(-1, M)

        # Build list of clique edges
        clique_edges_list = []
        for group in clique_groups:
            unique_nodes = np.unique(group)
            if len(unique_nodes) == M:
                # Generate all possible edges within the clique
                clique_edges = list(combinations(unique_nodes, 2))
                clique_edges_list.extend(clique_edges)
                cliques.append(unique_nodes.tolist())

        # Add clique edges to the graph
        g.add_edges(clique_edges_list)
        g.simplify(multiple=True, loops=True)

    # Generate the adjacency matrix
    adj_matrix = np.array(g.get_adjacency().data)

    if nr_cliques:
        return adj_matrix, cliques, len(cliques)
    else:
        return adj_matrix, cliques


# def generate_clustered_network_discrete_M(N, M, lambda_edges=2.5, m_cliques=2, nr_cliques=False, no_clique = False):
#     """
#     Generates a random clustered network with cliques of size M.

#     Parameters:
#         N (int): Total number of nodes in the network.
#         M (int): Size of the cliques to be formed (M >= 3).
#         lambda_edges (float): Average number of single edges per node.
#         m_cliques (float): Number of cliques per node.
#         seed (int): Random seed for reproducibility.

#     Returns:
#         adj_matrix (numpy.ndarray): Adjacency matrix of the generated network.
#         cliques (list of lists): List containing all the cliques (each clique is a list of node indices).
#     """

#     # Generate degrees for edges and cliques
#     # For simplicity, we assume independence between edges and cliques
#     if no_clique:
#         edge_degrees = np.random.poisson(lambda_edges, N) + (M-1)*nr_cliques
#         clique_degrees = np.zeros(N, dtype = int)
#     else:
#         edge_degrees = np.random.poisson(lambda_edges, N)
#         clique_degrees = np.ones(N, dtype = int)*m_cliques

#     # Adjust degrees to ensure total stubs are compatible
#     # For edges
#     if sum(edge_degrees) % 2 != 0:
#         idx = np.random.randint(0, N)
#         edge_degrees[idx] += 1

#     # For cliques
#     remainder = sum(clique_degrees) % M
#     if remainder != 0:
#         for _ in range(M - remainder):
#             idx = np.random.randint(0, N)
#             clique_degrees[idx] += 1

#     # Create edge stubs and clique stubs
#     edge_stubs = []
#     for node, degree in enumerate(edge_degrees):
#         edge_stubs.extend([node] * degree)

#     clique_stubs = []
#     for node, degree in enumerate(clique_degrees):
#         clique_stubs.extend([node] * degree)

#     # Shuffle stubs
#     np.random.shuffle(edge_stubs)
#     np.random.shuffle(clique_stubs)

#     # Construct edges by pairing edge stubs
#     edges = []
#     for i in range(0, len(edge_stubs), 2):
#         if i + 1 < len(edge_stubs):
#             n1 = edge_stubs[i]
#             n2 = edge_stubs[i + 1]
#             if n1 != n2:
#                 edges.append((n1, n2))

#     # Construct cliques by grouping clique stubs into groups of size M
#     cliques = []
#     for i in range(0, len(clique_stubs), M):
#         if i + M - 1 < len(clique_stubs):
#             nodes = clique_stubs[i:i + M]
#             unique_nodes = set(nodes)
#             if len(unique_nodes) == M:
#                 cliques.append(nodes)

#     # Combine edges and cliques into adjacency list
#     adjacency = {i: set() for i in range(N)}

#     # Add edges to adjacency list
#     for n1, n2 in edges:
#         adjacency[n1].add(n2)
#         adjacency[n2].add(n1)

#     # Add cliques to adjacency list
#     for clique_nodes in cliques:
#         for n1 in clique_nodes:
#             others = set(clique_nodes) - {n1}
#             adjacency[n1].update(others)

#     # Remove isolated nodes (nodes with no neighbors)
#     #isolated_nodes = [node for node, neighbors in adjacency.items() if not neighbors]
#     #for node in isolated_nodes:
#     #    del adjacency[node]

#     # Reindex nodes to have consecutive indices starting from 0
#     remaining_nodes = sorted(adjacency.keys())
#     node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_nodes)}
#     num_nodes = len(remaining_nodes)

#     # Create adjacency matrix
#     adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
#     for old_node, neighbors in adjacency.items():
#         new_node = node_mapping[old_node]
#         for neighbor in neighbors:
#             if neighbor in node_mapping:
#                 new_neighbor = node_mapping[neighbor]
#                 adj_matrix[new_node, new_neighbor] = 1
#                 # For undirected graph, ensure symmetry
#                 adj_matrix[new_neighbor, new_node] = 1
#     if nr_cliques:
#         return adj_matrix, cliques, len(cliques)
#     else:
#         return adj_matrix, cliques


import numpy as np
import igraph as ig

def keep_clusters_with_node_swap(adj_matrix, cliques, s_f=1.0, H=0.0):
    """
    Optimized function to keep a fraction s_f of clusters (cliques) and remove or swap nodes.

    Parameters:
        adj_matrix (numpy.ndarray): Adjacency matrix of the network.
        cliques (list of lists): List containing all the cliques (each clique is a list of node indices).
        s_f (float): Fraction of cliques to keep (0 <= s_f <= 1).
        H (float): Fraction of nodes in the selected cliques to swap with nodes from the rest of the network (0 <= H <= 1).

    Returns:
        new_adj_matrix (numpy.ndarray): Adjacency matrix after the modifications.
    """
    if not (0.0 <= s_f <= 1.0):
        raise ValueError("Fraction s_f must be between 0 and 1.")
    if not (0.0 <= H <= 1.0):
        raise ValueError("Fraction H must be between 0 and 1.")

    num_cliques = len(cliques)
    if s_f == 0.0 or num_cliques == 0:
        # Keep no cliques; return an empty adjacency matrix
        return np.array([[]], dtype=adj_matrix.dtype)

    # Step 1: Keep a fraction s_f of cliques
    num_cliques_to_keep = max(int(round(s_f * num_cliques)), 1)  # Ensure at least one clique is kept
    indices = np.random.choice(num_cliques, size=num_cliques_to_keep, replace=False)
    cliques_to_keep = [cliques[i] for i in indices]

    # Collect all nodes to keep (as a set for fast membership testing)
    nodes_to_keep = set().union(*cliques_to_keep)

    # Step 2: Randomly discard a fraction H of the nodes_to_keep
    num_nodes_to_discard = int(round(H * len(nodes_to_keep)))
    if num_nodes_to_discard > 0:
        nodes_to_keep_array = np.array(list(nodes_to_keep))
        nodes_to_discard = np.random.choice(nodes_to_keep_array, size=num_nodes_to_discard, replace=False)
        nodes_to_keep.difference_update(nodes_to_discard)
    else:
        nodes_to_discard = np.array([], dtype=int)

    # Step 3: Randomly select the same number of nodes from the rest of the network to add
    if num_nodes_to_discard > 0:
        all_nodes = np.arange(adj_matrix.shape[0])
        nodes_not_kept = np.setdiff1d(all_nodes, list(nodes_to_keep), assume_unique=True)
        nodes_to_add = np.random.choice(nodes_not_kept, size=num_nodes_to_discard, replace=False)
        nodes_to_keep.update(nodes_to_add)

    # If no nodes to keep after swapping, return an empty adjacency matrix
    if not nodes_to_keep:
        return np.array([[]], dtype=adj_matrix.dtype)

    # Convert the set to a sorted list to maintain consistent ordering
    nodes_to_keep_sorted = sorted(nodes_to_keep)

    # Build the new graph using igraph for efficiency
    # Convert the adjacency matrix to an igraph Graph
    G = ig.Graph.Adjacency((adj_matrix > 0).tolist(), mode=ig.ADJ_UNDIRECTED)
    # Extract the subgraph containing the nodes to keep
    subgraph = G.subgraph(nodes_to_keep_sorted)

    # Get the adjacency matrix of the subgraph
    new_adj_matrix = np.array(subgraph.get_adjacency().data)

    return new_adj_matrix


# def keep_clusters_with_node_swap(adj_matrix, cliques, s_f=1.0, H=0.0):
#     """
#     Keeps a fraction s_f of clusters (cliques) and removes all nodes not in the selected cliques.
#     Then, after selecting nodes to keep, randomly discard a fraction H of these nodes,
#     and randomly choose the same number of nodes from the remaining nodes (which includes the ones just discarded) to keep instead.

#     Parameters:
#         adj_matrix (numpy.ndarray): Adjacency matrix of the network.
#         cliques (list of lists): List containing all the cliques (each clique is a list of node indices).
#         s_f (float): Fraction of cliques to keep (0 <= s_f <= 1).
#         H (float): Fraction of nodes in the selected cliques to swap with nodes from the rest of the network (0 <= H <= 1).
#         seed (int): Random seed for reproducibility.

#     Returns:
#         new_adj_matrix (numpy.ndarray): Adjacency matrix after the modifications.
#     """
#     if not (0.0 <= s_f <= 1.0):
#         raise ValueError("Fraction s_f must be between 0 and 1.")
#     if not (0.0 <= H <= 1.0):
#         raise ValueError("Fraction H must be between 0 and 1.")

#     if s_f == 0.0:
#         # Keep no cliques, hence remove all nodes
#         return np.array([[]], dtype=int)

#     num_cliques = len(cliques)
#     if num_cliques == 0:
#         #print("No cliques to keep.")
#         return np.array([[]], dtype=int)

#     # Step 1: Keep a fraction s_f of cliques
#     num_cliques_to_keep = int(s_f * num_cliques)
#     num_cliques_to_keep = max(num_cliques_to_keep, 1)  # Ensure at least one clique is kept
#     cliques_to_keep = random.sample(cliques, num_cliques_to_keep)

#     # Collect all nodes to keep (unique set)
#     nodes_to_keep = set()
#     for clique in cliques_to_keep:
#         nodes_to_keep.update(clique)

#     if not nodes_to_keep:
#         # No nodes to keep
#         return np.array([[]], dtype=int)

#     # Step 2: Randomly discard a fraction H of the nodes_to_keep
#     num_nodes_to_discard = int(H * len(nodes_to_keep))
#     num_nodes_to_discard = min(num_nodes_to_discard, len(nodes_to_keep))
#     nodes_to_keep_list = list(nodes_to_keep)  # Convert to list for sampling
#     nodes_to_discard = set(random.sample(nodes_to_keep_list, num_nodes_to_discard))

#     # Update nodes_to_keep by removing nodes_to_discard
#     nodes_to_keep -= nodes_to_discard

#     # Step 3: Randomly select the same number of nodes from the rest of the network to add to nodes_to_keep
#     all_nodes = set(range(adj_matrix.shape[0]))
#     nodes_not_kept = all_nodes - nodes_to_keep
#     nodes_to_add = set(random.sample(sorted(nodes_not_kept), num_nodes_to_discard))

#     # Update nodes_to_keep by adding nodes_to_add
#     nodes_to_keep.update(nodes_to_add)

#     # Now nodes_to_keep contains the nodes after swapping
#     if not nodes_to_keep:
#         # No nodes to keep after swapping
#         return np.array([[]], dtype=int)

#     # Build the new adjacency matrix
#     # Map node indices to consecutive indices starting from 0
#     nodes_to_keep_sorted = sorted(nodes_to_keep)
#     node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes_to_keep_sorted)}
#     num_nodes_new = len(nodes_to_keep_sorted)

#     # Create new adjacency matrix
#     new_adj_matrix = np.zeros((num_nodes_new, num_nodes_new), dtype=adj_matrix.dtype)
#     for old_i in nodes_to_keep_sorted:
#         new_i = node_mapping[old_i]
#         neighbors = np.nonzero(adj_matrix[old_i])[0]
#         for old_j in neighbors:
#             if old_j in nodes_to_keep:
#                 new_j = node_mapping[old_j]
#                 new_adj_matrix[new_i, new_j] = adj_matrix[old_i, old_j]

#     return new_adj_matrix

def perform_bond_percolation(adj_matrix, N_p, p_values):
    """
    Performs bond percolation on a network defined by adj_matrix over a range of edge retention probabilities.
    
    Parameters:
        adj_matrix (numpy.ndarray): Adjacency matrix of the network.
        N_p (int): Number of percolation trials per edge retention probability.
        p_values (numpy.ndarray): Array of edge retention probabilities (between 0 and 1).
    
    Returns:
        percolation_results (dict): Dictionary containing percolation analysis results.
    """
    num_nodes = adj_matrix.shape[0]
    percolation_results = {
        'p_values': p_values,
        'giant_component_sizes': [],
        'cluster_size_distributions': [],
    }
    
    for p in tqdm(p_values, desc='Percolation Progress'):
        largest_components = []
        cluster_sizes = []
        
        for _ in range(N_p):
            # Perform bond percolation by retaining edges with probability p
            percolated_adj = adj_matrix.copy()
            # Get indices of the upper triangle (since adj_matrix is symmetric)
            upper_tri_indices = np.triu_indices(num_nodes, k=1)
            edges = np.vstack(upper_tri_indices).T
            existing_edges = edges[percolated_adj[upper_tri_indices] > 0]
            
            # Randomly remove edges
            retain_mask = np.random.rand(len(existing_edges)) < p
            retained_edges = existing_edges[retain_mask]
            
            # Create a new adjacency matrix for the percolated graph
            percolated_adj = np.zeros_like(adj_matrix)
            percolated_adj[retained_edges[:, 0], retained_edges[:, 1]] = 1
            percolated_adj[retained_edges[:, 1], retained_edges[:, 0]] = 1  # Symmetric
            
            # Create a NetworkX graph from the percolated adjacency matrix
            G_perc = nx.from_numpy_array(percolated_adj)
            
            # Get connected components
            connected_components = [len(cc) for cc in nx.connected_components(G_perc)]
            
            if connected_components:
                # Largest connected component size
                largest_components.append(max(connected_components) / num_nodes)
                # Record all cluster sizes
                cluster_sizes.extend(connected_components)
            else:
                largest_components.append(0)
        
        # Average size of the largest connected component for this p
        avg_largest_component = np.mean(largest_components)
        percolation_results['giant_component_sizes'].append(avg_largest_component)
        percolation_results['cluster_size_distributions'].append(cluster_sizes)
    
    # Convert lists to numpy arrays
    percolation_results['giant_component_sizes'] = np.array(percolation_results['giant_component_sizes'])
    
    # Estimate the percolation threshold
    percolation_threshold = estimate_percolation_threshold(
        percolation_results['p_values'],
        percolation_results['giant_component_sizes']
    )
    percolation_results['percolation_threshold'] = percolation_threshold
    
    return percolation_results

def estimate_percolation_threshold(p_values, giant_component_sizes):
    """
    Estimates the percolation threshold as the point where the size of the giant component
    increases rapidly (inflection point).
    
    Parameters:
        p_values (numpy.ndarray): Array of edge retention probabilities.
        giant_component_sizes (numpy.ndarray): Corresponding sizes of the largest connected component.
    
    Returns:
        p_c (float): Estimated percolation threshold.
    """
    # Smooth the giant component sizes
    f = interp1d(p_values, giant_component_sizes, kind='cubic')
    p_smooth = np.linspace(p_values[0], p_values[-1], 500)
    gc_smooth = f(p_smooth)
    
    # Compute the derivative of the giant component sizes
    derivative = np.gradient(gc_smooth, p_smooth)
    
    # Find the p value where the derivative is maximized
    max_derivative_index = np.argmax(derivative)
    p_c = p_smooth[max_derivative_index]
    
    return p_c

def perform_percolation(adj_matrix, p, N_p, epi = False):
    """
    Performs percolation starting from randomly selected node(s).

    Parameters:
        adj_matrix (numpy.ndarray): Adjacency matrix of the network.
        p_i (float): Percolation probability (beta / (beta + gamma)).
        N_p (int): Number of percolation trials per edge retention probability.

    Returns:
        fraction_infected (float): Fraction of nodes reached by percolation.
    """
    num_nodes = adj_matrix.shape[0]
    if num_nodes == 0:
        return 0.0  # Empty network

    largest_components = []
    cluster_sizes = []
    
    for _ in range(N_p):
        # Perform bond percolation by retaining edges with probability p
        percolated_adj = adj_matrix.copy()
        # Get indices of the upper triangle (since adj_matrix is symmetric)
        upper_tri_indices = np.triu_indices(num_nodes, k=1)
        edges = np.vstack(upper_tri_indices).T
        existing_edges = edges[percolated_adj[upper_tri_indices] > 0]
        
        # Randomly remove edges
        retain_mask = np.random.rand(len(existing_edges)) <= p
        retained_edges = existing_edges[retain_mask]
        
        # Create a new adjacency matrix for the percolated graph
        percolated_adj = np.zeros_like(adj_matrix)
        percolated_adj[retained_edges[:, 0], retained_edges[:, 1]] = 1
        percolated_adj[retained_edges[:, 1], retained_edges[:, 0]] = 1  # Symmetric


        num_nodes = percolated_adj.shape[0]
        if np.shape(percolated_adj) == (1,0):
            largest_components.append(0)
            cluster_sizes.append(0)
        else:
            # Create a NetworkX graph from the percolated adjacency matrix
            G_perc = nx.from_numpy_array(percolated_adj)
            
            # Get connected components
            connected_components = [len(cc)/num_nodes for cc in nx.connected_components(G_perc)]
            
            if connected_components:
                # Largest connected component size
                largest_components.append(max(connected_components))
                # Record all cluster sizes
                cluster_sizes.extend(connected_components)
            else:
                largest_components.append(0)
                cluster_sizes.extend(0)
    
    # Average size of the largest connected component for this p
    avg_largest_component = np.mean(largest_components)
    mean_cluster_size = np.mean(cluster_sizes)#/num_nodes
    if np.sum(cluster_sizes) == 0:
        return 0
    else: 
        prob_outbreak = np.average(cluster_sizes, weights = cluster_sizes)
        return prob_outbreak

# def gillespie_SIR(adj_matrix, beta, gamma, initial_infected, max_time):
#     """
#     Simulates the SIR epidemic using the Gillespie algorithm on a given network.

#     Parameters:
#         adj_matrix (numpy.ndarray): Adjacency matrix of the network.
#         beta (float): Infection rate per contact.
#         gamma (float): Recovery rate.
#         initial_infected (list or set): Indices of initially infected nodes.
#         max_time (float): Maximum simulation time.

#     Returns:
#         times (list): Times at which events occur.
#         num_infected (list): Number of infected individuals over time.
#         final_size (int): Total number of recovered individuals at the end.
#         events (list): List of events that occurred (infection or recovery).
#     """
#     N = adj_matrix.shape[0]
#     S = set(range(N)) - set(initial_infected)
#     I = set(initial_infected)
#     R = set()

#     t = 0.0
#     times = [t]
#     num_infected = [len(I)]
#     events = []

#     while t < max_time and len(I) > 0:
#         rates = []
#         events_list = []

#         for node in I:
#             # Recovery event for this node
#             recovery_rate = gamma
#             rates.append(recovery_rate)
#             events_list.append(('recovery', node))

#             # Infection events to susceptible neighbors
#             neighbors = np.where(adj_matrix[node, :] > 0)[0]
#             susceptible_neighbors = S.intersection(neighbors)
#             infection_rate = beta * len(susceptible_neighbors)
#             if infection_rate > 0:
#                 rates.append(infection_rate)
#                 events_list.append(('infection', node, susceptible_neighbors))

#         total_rate = sum(rates)

#         if total_rate == 0:
#             break  # No more possible events

#         dt = -np.log(np.random.rand()) / total_rate
#         t += dt

#         # Choose event
#         cumulative_rates = np.cumsum(rates)
#         r = np.random.rand() * total_rate
#         event_index = np.searchsorted(cumulative_rates, r)
#         event = events_list[event_index]

#         if event[0] == 'recovery':
#             node = event[1]
#             I.remove(node)
#             R.add(node)
#         elif event[0] == 'infection':
#             infector = event[1]
#             susceptible_neighbors = event[2]
#             if susceptible_neighbors:
#                 infectee = random.choice(list(susceptible_neighbors))
#                 S.remove(infectee)
#                 I.add(infectee)
#             else:
#                 continue  # No susceptible neighbors to infect

#         times.append(t)
#         num_infected.append(len(I))
#         events.append(event)

#     final_size = len(R)

#     return times, num_infected, final_size, events


import numpy as np
import numba
from numba import njit

@njit
def gillespie_SIR(adj_matrix, beta, gamma, initial_infected, max_time):
    """
    Optimized Gillespie SIR simulation using NumPy arrays and Numba JIT compilation.

    Parameters:
        adj_matrix (numpy.ndarray): Adjacency matrix of the network.
        beta (float): Infection rate per contact.
        gamma (float): Recovery rate.
        initial_infected (list or numpy.ndarray): Indices of initially infected nodes.
        max_time (float): Maximum simulation time.

    Returns:
        times (numpy.ndarray): Times at which events occur.
        num_infected (numpy.ndarray): Number of infected individuals over time.
        final_size (int): Total number of recovered individuals at the end.
    """
    N = adj_matrix.shape[0]
    
    # Convert initial_infected to NumPy array
    initial_infected = np.array(initial_infected, dtype=np.int32)
    
    status = np.zeros(N, dtype=np.int8)  # 0: Susceptible, 1: Infected, 2: Recovered
    status[initial_infected] = 1  # Set initial infected nodes

    t = 0.0
    times = [t]
    num_infected = [len(initial_infected)]

    # List of infected nodes
    infected_nodes = initial_infected.copy()
    num_infected_nodes = len(infected_nodes)

    # Precompute neighbors for each node
    neighbor_indices = []
    for i in range(N):
        neighbors = np.where(adj_matrix[i] > 0)[0]
        neighbor_indices.append(neighbors.astype(np.int32))

    while t < max_time and num_infected_nodes > 0:
        total_rate = 0.0
        rates = []
        cumulative_rates = []

        # Initialize lists to store events and their rates
        event_nodes = []
        event_types = []  # 0: recovery, 1: infection

        # Calculate rates for recovery and infection events
        for idx in range(num_infected_nodes):
            node = infected_nodes[idx]

            # Recovery rate
            rec_rate = gamma
            total_rate += rec_rate
            rates.append(rec_rate)
            cumulative_rates.append(total_rate)
            event_nodes.append(node)
            event_types.append(0)  # Recovery event

            # Infection rate
            neighbors = neighbor_indices[node]
            susceptible_neighbors = neighbors[status[neighbors] == 0]
            inf_rate = beta * len(susceptible_neighbors)

            if inf_rate > 0:
                total_rate += inf_rate
                rates.append(inf_rate)
                cumulative_rates.append(total_rate)
                event_nodes.append(node)
                event_types.append(1)  # Infection event

        if total_rate == 0.0:
            break  # No more events can occur

        # Time to next event
        dt = -np.log(np.random.random()) / total_rate
        t += dt

        # Determine which event occurs
        r = np.random.random() * total_rate
        event_index = np.searchsorted(cumulative_rates, r)
        event_node = event_nodes[event_index]
        event_type = event_types[event_index]

        if event_type == 0:
            # Recovery event
            status[event_node] = 2  # Recovered
            idx = np.where(infected_nodes == event_node)[0][0]
            infected_nodes = np.delete(infected_nodes, idx)
            num_infected_nodes -= 1
        else:
            # Infection event
            neighbors = neighbor_indices[event_node]
            susceptible_neighbors = neighbors[status[neighbors] == 0]
            if len(susceptible_neighbors) > 0:
                # Randomly infect one susceptible neighbor
                infectee = susceptible_neighbors[np.random.randint(len(susceptible_neighbors))]
                status[infectee] = 1  # Infected
                infected_nodes = np.append(infected_nodes, infectee)
                num_infected_nodes += 1

        times.append(t)
        num_infected.append(num_infected_nodes)

    times = np.array(times)
    num_infected = np.array(num_infected)
    final_size = np.sum(status == 2)

    return times, num_infected, final_size
