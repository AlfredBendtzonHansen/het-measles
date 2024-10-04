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

def generate_clustered_network_discrete_M(N, M, lambda_edges=2.5, m_cliques=2, nr_cliques=False, no_clique = False):
    """
    Generates a random clustered network with cliques of size M.

    Parameters:
        N (int): Total number of nodes in the network.
        M (int): Size of the cliques to be formed (M >= 3).
        lambda_edges (float): Average number of single edges per node.
        m_cliques (float): Number of cliques per node.
        seed (int): Random seed for reproducibility.

    Returns:
        adj_matrix (numpy.ndarray): Adjacency matrix of the generated network.
        cliques (list of lists): List containing all the cliques (each clique is a list of node indices).
    """

    # Generate degrees for edges and cliques
    # For simplicity, we assume independence between edges and cliques
    if no_clique:
        edge_degrees = np.random.poisson(lambda_edges, N) + (M-1)*nr_cliques
        clique_degrees = np.zeros(N, dtype = int)
    else:
        edge_degrees = np.random.poisson(lambda_edges, N)
        clique_degrees = np.ones(N, dtype = int)*m_cliques

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

    # Remove isolated nodes (nodes with no neighbors)
    #isolated_nodes = [node for node, neighbors in adjacency.items() if not neighbors]
    #for node in isolated_nodes:
    #    del adjacency[node]

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


def keep_clusters_with_node_swap(adj_matrix, cliques, s_f=1.0, H=0.0):
    """
    Keeps a fraction s_f of clusters (cliques) and removes all nodes not in the selected cliques.
    Then, after selecting nodes to keep, randomly discard a fraction H of these nodes,
    and randomly choose the same number of nodes from the remaining nodes (which includes the ones just discarded) to keep instead.

    Parameters:
        adj_matrix (numpy.ndarray): Adjacency matrix of the network.
        cliques (list of lists): List containing all the cliques (each clique is a list of node indices).
        s_f (float): Fraction of cliques to keep (0 <= s_f <= 1).
        H (float): Fraction of nodes in the selected cliques to swap with nodes from the rest of the network (0 <= H <= 1).
        seed (int): Random seed for reproducibility.

    Returns:
        new_adj_matrix (numpy.ndarray): Adjacency matrix after the modifications.
    """
    if not (0.0 <= s_f <= 1.0):
        raise ValueError("Fraction s_f must be between 0 and 1.")
    if not (0.0 <= H <= 1.0):
        raise ValueError("Fraction H must be between 0 and 1.")

    if s_f == 0.0:
        # Keep no cliques, hence remove all nodes
        return np.array([[]], dtype=int)

    num_cliques = len(cliques)
    if num_cliques == 0:
        #print("No cliques to keep.")
        return np.array([[]], dtype=int)

    # Step 1: Keep a fraction s_f of cliques
    num_cliques_to_keep = int(s_f * num_cliques)
    num_cliques_to_keep = max(num_cliques_to_keep, 1)  # Ensure at least one clique is kept
    cliques_to_keep = random.sample(cliques, num_cliques_to_keep)

    # Collect all nodes to keep (unique set)
    nodes_to_keep = set()
    for clique in cliques_to_keep:
        nodes_to_keep.update(clique)

    if not nodes_to_keep:
        # No nodes to keep
        return np.array([[]], dtype=int)

    # Step 2: Randomly discard a fraction H of the nodes_to_keep
    num_nodes_to_discard = int(H * len(nodes_to_keep))
    num_nodes_to_discard = min(num_nodes_to_discard, len(nodes_to_keep))
    nodes_to_keep_list = list(nodes_to_keep)  # Convert to list for sampling
    nodes_to_discard = set(random.sample(nodes_to_keep_list, num_nodes_to_discard))

    # Update nodes_to_keep by removing nodes_to_discard
    nodes_to_keep -= nodes_to_discard

    # Step 3: Randomly select the same number of nodes from the rest of the network to add to nodes_to_keep
    all_nodes = set(range(adj_matrix.shape[0]))
    nodes_not_kept = all_nodes - nodes_to_keep
    nodes_to_add = set(random.sample(sorted(nodes_not_kept), num_nodes_to_discard))

    # Update nodes_to_keep by adding nodes_to_add
    nodes_to_keep.update(nodes_to_add)

    # Now nodes_to_keep contains the nodes after swapping
    if not nodes_to_keep:
        # No nodes to keep after swapping
        return np.array([[]], dtype=int)

    # Build the new adjacency matrix
    # Map node indices to consecutive indices starting from 0
    nodes_to_keep_sorted = sorted(nodes_to_keep)
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes_to_keep_sorted)}
    num_nodes_new = len(nodes_to_keep_sorted)

    # Create new adjacency matrix
    new_adj_matrix = np.zeros((num_nodes_new, num_nodes_new), dtype=adj_matrix.dtype)
    for old_i in nodes_to_keep_sorted:
        new_i = node_mapping[old_i]
        neighbors = np.nonzero(adj_matrix[old_i])[0]
        for old_j in neighbors:
            if old_j in nodes_to_keep:
                new_j = node_mapping[old_j]
                new_adj_matrix[new_i, new_j] = adj_matrix[old_i, old_j]

    return new_adj_matrix

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
