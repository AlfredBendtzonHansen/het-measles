import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from scipy.interpolate import interp1d

def generate_clustered_network(N, M, lambda_edges=2.5, lambda_cliques=2.0, seed=42):
    """
    Generates a random clustered network with cliques of size M.

    Parameters:
        N (int): Total number of nodes in the network.
        M (int): Size of the cliques to be formed (M >= 3).
        lambda_edges (float): Average number of single edges per node.
        lambda_cliques (float): Average number of cliques per node.
        seed (int): Random seed for reproducibility.

    Returns:
        adj_matrix (numpy.ndarray): Adjacency matrix of the generated network.
        cliques (list of lists): List containing all the cliques (each clique is a list of node indices).
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

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

    # Remove isolated nodes (nodes with no neighbors)
    isolated_nodes = [node for node, neighbors in adjacency.items() if not neighbors]
    for node in isolated_nodes:
        del adjacency[node]

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

    return adj_matrix, cliques#, node_mapping



def remove_clusters(adj_matrix, cliques, s_f=0.0, seed=42):
    """
    Removes a fraction s_f of clusters (cliques) and their nodes from the network.

    Parameters:
        adj_matrix (numpy.ndarray): Adjacency matrix of the network.
        cliques (list of lists): List containing all the cliques (each clique is a list of node indices).
        s_f (float): Fraction of cliques to remove (0 <= s_f <= 1).
        seed (int): Random seed for reproducibility.

    Returns:
        new_adj_matrix (numpy.ndarray): Adjacency matrix after removing specified cliques and their nodes.
        remaining_cliques (list of lists): List of cliques remaining after removal.
        removed_cliques (list of lists): List of cliques that were removed.
    """
    if not (0.0 <= s_f <= 1.0):
        raise ValueError("Fraction s_f must be between 0 and 1.")

    if s_f == 0.0:
        # No removal needed
        return adj_matrix.copy()#, cliques.copy(), []

    np.random.seed(seed)
    random.seed(seed)

    num_cliques = len(cliques)
    if num_cliques == 0:
        print("No cliques to remove.")
        return adj_matrix.copy(), cliques.copy(), []

    num_cliques_to_remove = int(s_f * num_cliques)
    num_cliques_to_remove = min(num_cliques_to_remove, num_cliques)  # Ensure we don't exceed available cliques

    # Randomly select cliques to remove
    cliques_to_remove = random.sample(cliques, num_cliques_to_remove)

    # Collect all nodes to remove (unique set)
    nodes_to_remove = set()
    for clique in cliques_to_remove:
        nodes_to_remove.update(clique)

    # Convert adjacency matrix to a NetworkX graph for easier manipulation
    G = nx.from_numpy_array(adj_matrix)

    # Remove the nodes
    G.remove_nodes_from(nodes_to_remove)

    # Generate new adjacency matrix
    new_adj_matrix = nx.to_numpy_array(G, dtype=int)

    # Update the cliques list by removing the cliques that were removed
    remaining_cliques = [clique for clique in cliques if clique not in cliques_to_remove]

    return new_adj_matrix#, remaining_cliques, cliques_to_remove




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


