import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import re
from gunfolds.utils import graphkit as gk
from gunfolds import conversions as cv
from os import listdir
from gunfolds.utils import zickle  as zkl
import pandas as pd


def remove_bidir_edges(input_dict):
    result_dict = {}
    for key, inner_dict in input_dict.items():
        result_dict[key] = {}
        for inner_key, value in inner_dict.items():
            if value == 1:
                result_dict[key][inner_key] = 1
            elif value == 2:
                pass  # Skip adding this key-value pair
            elif value == 3:
                result_dict[key][inner_key] = 1
            else:
                raise ValueError("Invalid value encountered: {}".format(value))
    return result_dict

def read_gimme_to_graph(beta_file_path, std_error_file_path):
    """
    Reads beta coefficients and standard errors from files and converts them to a simplified NetworkX graph.
    Variables and their lagged versions are treated as the same entity.
    Args:
        beta_file_path (str): Path to the file containing beta coefficients.
        std_error_file_path (str): Path to the file containing standard errors.
    Returns:
        nx.DiGraph: A simplified directed graph.
    """
    # Read beta and standard error files
    betas = pd.read_csv(beta_file_path)
    std_errors = pd.read_csv(std_error_file_path)
    def simplify_label(label):
        """
        Simplifies a variable name by removing 'lag' to treat variables and their lags as the same.
        """
        if "lag" in label:
            return label.replace("lag", "")
        return label
    # Initialize a directed graph
    G = nx.DiGraph()
    # Iterate through the beta coefficients to add edges
    for i, row in betas.iterrows():
        source = simplify_label(row.iloc[0])  # First column is the source variable (row name)
        for target, beta_value in row.iloc[1:].items():  # Remaining columns are target variables
            if beta_value != 0:  # Only add edges with non-zero beta coefficients
                simplified_target = simplify_label(target)
                std_error = std_errors.iloc[i].loc[target]  # Find the standard error for the same path
                # Combine weights and standard errors if the edge already exists
                if G.has_edge(source, simplified_target):
                    existing_weight = G[source][simplified_target]['weight']
                    existing_std_error = G[source][simplified_target]['std_error']
                    G[source][simplified_target]['weight'] = (existing_weight + beta_value) / 2
                    G[source][simplified_target]['std_error'] = (existing_std_error + std_error) / 2
                else:
                    G.add_edge(source, simplified_target, weight=beta_value, std_error=std_error)
    return G

def convert_nodes_to_numbers(graph):
    mapping = {f"X{i}": i for i in range(1, len(graph.nodes)+1)}  # Map 'X1', 'X2', ..., 'X10' to 1, 2, ..., 10
    return nx.relabel_nodes(graph, mapping)

def precision_recall(answer, network_GT_selfloop, include_selfloop=True):
    # Precision = True Positives / (True Positives + False Positives)
    # Recall = True Positives /  (True Positives + False Negatives)
    res_graph = answer
    GT_nx = gk.graph2nx(network_GT_selfloop)
    res_nx = gk.graph2nx(res_graph)

    #######precision and recall (orientation)
    TP, FP, FN = 0, 0, 0
    for edge in GT_nx.edges():
        if include_selfloop or edge[1] != edge[0]:
            if edge in res_nx.edges():
                TP += 1
            else:
                FN += 1
    for edge in res_nx.edges():
        if edge not in GT_nx.edges():
            if include_selfloop or edge[1] != edge[0]:
                FP += 1
    p_O = (TP / (TP + FP)) if (TP + FP) else 0
    r_O = (TP / (TP + FN)) if (TP + FN) else 0
    f1_O = (2 * TP) / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0

    #######precision and recall (adjacency)
    TP, FP, FN = 0, 0, 0
    for edge in GT_nx.edges():
        if include_selfloop or edge[1] != edge[0]:
            if edge in res_nx.edges() or (edge[1], edge[0]) in res_nx.edges():
                if ((edge[1], edge[0]) in GT_nx.edges()) and (edge[1] != edge[0]):
                    TP += 0.5
                else:
                    TP += 1
            else:
                if (edge[1], edge[0]) in GT_nx.edges() and (edge[1] != edge[0]):
                    FN += 0.5
                else:
                    FN += 1
    for edge in res_nx.edges():
        if include_selfloop or edge[1] != edge[0]:
            if not (edge in GT_nx.edges() or (edge[1], edge[0]) in GT_nx.edges()):
                if ((edge[1], edge[0]) in res_nx.edges()) and (edge[1] != edge[0]):
                    FP += 0.5
                else:
                    FP += 1
    p_A = (TP / (TP + FP)) if (TP + FP) else 0
    r_A = (TP / (TP + FN)) if (TP + FN) else 0
    f1_A = (2 * TP) / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0

    #######precision and recall (2-cycle)

    TP, FP, FN = 0, 0, 0
    for edge in GT_nx.edges():
        if include_selfloop or edge[1] != edge[0]:
            if not edge[1] == edge[0]:
                if (edge[1], edge[0]) in GT_nx.edges():
                    if edge in res_nx.edges() and (edge[1], edge[0]) in res_nx.edges():
                        TP += 1
                    else:
                        FN += 1
    for edge in res_nx.edges():
        if include_selfloop or edge[1] != edge[0]:
            if not edge[1] == edge[0]:
                if (edge[1], edge[0]) in res_nx.edges():
                    if not (edge in GT_nx.edges() and (edge[1], edge[0]) in GT_nx.edges()):
                        FP += 1
    p_C = (TP / (TP + FP)) if (TP + FP) else 0
    r_C = (TP / (TP + FN)) if (TP + FN) else 0
    f1_C = (2 * TP) / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0

    prf = {'orientation': {'precision': p_O, 'recall': r_O, 'F1': f1_O},
           'adjacency': {'precision': p_A, 'recall': r_A, 'F1': f1_A},
           'cycle': {'precision': p_C, 'recall': r_C, 'F1': f1_C}}

    return prf

def precision_recall_all_cycle(answer, network_GT_selfloop, include_selfloop=True):
    # Precision = True Positives / (True Positives + False Positives)
    # Recall = True Positives /  (True Positives + False Negatives)
    res_graph = answer
    GT_nx = gk.graph2nx(network_GT_selfloop)
    res_nx = gk.graph2nx(res_graph)

    #######precision and recall (orientation)
    TP, FP, FN = 0, 0, 0
    for edge in GT_nx.edges():
        if include_selfloop or edge[1] != edge[0]:
            if edge in res_nx.edges():
                TP += 1
            else:
                FN += 1
    for edge in res_nx.edges():
        if edge not in GT_nx.edges():
            if include_selfloop or edge[1] != edge[0]:
                FP += 1
    p_O = (TP / (TP + FP)) if (TP + FP) else 0
    r_O = (TP / (TP + FN)) if (TP + FN) else 0
    f1_O = (2 * TP) / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0

    #######precision and recall (adjacency)
    TP, FP, FN = 0, 0, 0
    for edge in GT_nx.edges():
        if include_selfloop or edge[1] != edge[0]:
            if edge in res_nx.edges() or (edge[1], edge[0]) in res_nx.edges():
                if ((edge[1], edge[0]) in GT_nx.edges()) and (edge[1] != edge[0]):
                    TP += 0.5
                else:
                    TP += 1
            else:
                if (edge[1], edge[0]) in GT_nx.edges() and (edge[1] != edge[0]):
                    FN += 0.5
                else:
                    FN += 1
    for edge in res_nx.edges():
        if include_selfloop or edge[1] != edge[0]:
            if not (edge in GT_nx.edges() or (edge[1], edge[0]) in GT_nx.edges()):
                if ((edge[1], edge[0]) in res_nx.edges()) and (edge[1] != edge[0]):
                    FP += 0.5
                else:
                    FP += 1
    p_A = (TP / (TP + FP)) if (TP + FP) else 0
    r_A = (TP / (TP + FN)) if (TP + FN) else 0
    f1_A = (2 * TP) / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0

    #######precision and recall (all cycle)

    GT_cycles = list(nx.simple_cycles(GT_nx))
    res_cycles = list(nx.simple_cycles(res_nx))

    # Convert cycles to sets of tuples for easier comparison
    GT_cycles_set = set(frozenset(cycle) for cycle in GT_cycles)
    res_cycles_set = set(frozenset(cycle) for cycle in res_cycles)

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = len(GT_cycles_set & res_cycles_set)  # Cycles present in both GT and res
    FP = len(res_cycles_set - GT_cycles_set)  # Cycles present in res but not in GT
    FN = len(GT_cycles_set - res_cycles_set)  # Cycles present in GT but not in res

    # Calculate precision, recall, and F1 score
    p_C = (TP / (TP + FP)) if (TP + FP) else 0
    r_C = (TP / (TP + FN)) if (TP + FN) else 0
    f1_C = (2 * TP) / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0

    prf = {'orientation': {'precision': round(p_O, 2), 'recall': round(r_O, 2), 'F1': round(f1_O, 2)},
           'adjacency': {'precision': round(p_A, 2), 'recall': round(r_A, 2), 'F1': round(f1_A, 2)},
           'cycle': {'precision': round(p_C, 2), 'recall': round(r_C, 2), 'F1': round(f1_C, 2)}}

    return prf

def precision_recall_no_cycle(answer, network_GT_selfloop, include_selfloop=True):
    # Precision = True Positives / (True Positives + False Positives)
    # Recall = True Positives /  (True Positives + False Negatives)
    res_graph = answer
    GT_nx = gk.graph2nx(network_GT_selfloop)
    res_nx = gk.graph2nx(res_graph)

    #######precision and recall (orientation)
    TP, FP, FN = 0, 0, 0
    for edge in GT_nx.edges():
        if include_selfloop or edge[1] != edge[0]:
            if edge in res_nx.edges():
                TP += 1
            else:
                FN += 1
    for edge in res_nx.edges():
        if edge not in GT_nx.edges():
            if include_selfloop or edge[1] != edge[0]:
                FP += 1
    p_O = (TP / (TP + FP)) if (TP + FP) else 0
    r_O = (TP / (TP + FN)) if (TP + FN) else 0
    f1_O = (2 * TP) / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0

    #######precision and recall (adjacency)
    TP, FP, FN = 0, 0, 0
    for edge in GT_nx.edges():
        if include_selfloop or edge[1] != edge[0]:
            if edge in res_nx.edges() or (edge[1], edge[0]) in res_nx.edges():
                if ((edge[1], edge[0]) in GT_nx.edges()) and (edge[1] != edge[0]):
                    TP += 0.5
                else:
                    TP += 1
            else:
                if (edge[1], edge[0]) in GT_nx.edges() and (edge[1] != edge[0]):
                    FN += 0.5
                else:
                    FN += 1
    for edge in res_nx.edges():
        if include_selfloop or edge[1] != edge[0]:
            if not (edge in GT_nx.edges() or (edge[1], edge[0]) in GT_nx.edges()):
                if ((edge[1], edge[0]) in res_nx.edges()) and (edge[1] != edge[0]):
                    FP += 0.5
                else:
                    FP += 1
    p_A = (TP / (TP + FP)) if (TP + FP) else 0
    r_A = (TP / (TP + FN)) if (TP + FN) else 0
    f1_A = (2 * TP) / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0



    prf = {'orientation': {'precision': round(p_O, 2), 'recall': round(r_O, 2), 'F1': round(f1_O, 2)},
           'adjacency': {'precision': round(p_A, 2), 'recall': round(r_A, 2), 'F1': round(f1_A, 2)}}

    return prf

def round_tuple_elements(input_tuple, decimal_points=3):
    return tuple(round(elem, decimal_points) if isinstance(elem, (int, float)) else elem for elem in input_tuple)

def find_two_cycles(graph):
    # Find all 2-cycles in the graph
    two_cycles = set()
    visited = set()
    for node in graph.nodes():
        for neighbor in graph.neighbors(node):
            # Check for a directed edge in both directions and ensure it's not a self-loop
            if node != neighbor and graph.has_edge(node, neighbor) and graph.has_edge(neighbor, node):
                # Ensure we count each 2-cycle only once
                edge_pair = tuple(sorted([node, neighbor]))
                if edge_pair not in visited:
                    two_cycles.add(edge_pair)
                    visited.add(edge_pair)
    return two_cycles


def divide_into_batches(lst, n):
    # Calculate the size of each batch
    batch_size = len(lst) // n
    batches = []

    for i in range(n):
        # Calculate the start and end indices for each batch
        start = i * batch_size
        if i == n - 1:  # Last batch takes the remaining elements
            end = len(lst)
        else:
            end = start + batch_size
        batches.append(lst[start:end])

    return batches

def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def check_matrix_powers(W, A, powers, threshold):
    for n in powers:
        W_n = np.linalg.matrix_power(W, n)
        non_zero_indices = np.nonzero(W_n)
        if (np.abs(W_n[non_zero_indices]) < threshold).any():
            return False
    return True


def create_stable_weighted_matrix(
        A,
        threshold=0.1,
        powers=[1, 2, 3, 4],
        max_attempts=10000000,
        damping_factor=0.99,
        random_state=None,
):
    np.random.seed(
        random_state
    )  # Set random seed for reproducibility if provided
    attempts = 0

    while attempts < max_attempts:
        # Generate a random matrix with the same sparsity pattern as A
        random_weights = np.random.randn(*A.shape)
        weighted_matrix = A * random_weights

        # Convert to sparse format for efficient eigenvalue computation
        weighted_sparse = sp.csr_matrix(weighted_matrix)

        # Compute the largest eigenvalue in magnitude
        eigenvalues, _ = eigs(weighted_sparse, k=1, which="LM")
        max_eigenvalue = np.abs(eigenvalues[0])

        # Scale the matrix so that the spectral radius is slightly less than 1
        if max_eigenvalue > 0:
            weighted_matrix *= damping_factor / max_eigenvalue
            # Check if the powers of the matrix preserve the threshold for non-zero entries of A
            if check_matrix_powers(weighted_matrix, A, powers, threshold):
                return weighted_matrix

        attempts += 1

    raise ValueError(
        f"Unable to create a matrix satisfying the condition after {max_attempts} attempts."
    )

def genData(A, rate=2, burnin=100, ssize=5000, noise=0.1, dist='normal'):
    Agt = A
    data = drawsamplesLG(Agt, samples=burnin + (ssize * rate), nstd=noise)
    data = data[:, burnin:]
    return data[:, ::rate]


def drawsamplesLG(A, nstd=0.1, samples=100):
    n = A.shape[0]
    data = np.zeros([n, samples])
    data[:, 0] = nstd * np.random.randn(A.shape[0])
    for i in range(1, samples):
        data[:, i] = A @ data[:, i - 1] + nstd * np.random.randn(A.shape[0])
    return data

def parse_nodes(graph_str):
    pattern = r'Graph Nodes:\n([\w;]+)'
    match = re.search(pattern, graph_str)
    if match:
        nodes = match.group(1).split(';')
        return nodes
    return []


def parse_edges(graph_str):
    pattern = r'Graph Edges:\n((?:\d+\.\s\w+\s-->\s\w+\n)+)'
    match = re.search(pattern, graph_str)
    if match:
        edges_str = match.group(1)
        edge_pattern = r'\d+\.\s(\w+)\s-->\s(\w+)'
        edges = re.findall(edge_pattern, edges_str)
        return edges
    return []

def create_adjacency_matrix(edges, nodes):
    node_index = {node: idx for idx, node in enumerate(nodes)}
    size = len(nodes)
    matrix = np.zeros((size, size), dtype=int)

    for start, end in edges:
        start_idx = node_index[start]
        end_idx = node_index[end]
        matrix[start_idx, end_idx] = 1

    return matrix


def update_base_graph(base_graph, new_graph):
    # Convert the current base graph to adjacency matrices
    base_adj = cv.graph2adj(base_graph)
    base_badj = cv.graph2badj(base_graph)

    # Convert the new graph to adjacency matrices
    new_adj = cv.graph2adj(new_graph)
    new_badj = cv.graph2badj(new_graph)

    # Update the base adjacency matrices using logical OR
    updated_adj = np.logical_or(base_adj, new_adj).astype(int)
    updated_badj = np.logical_or(base_badj, new_badj).astype(int)

    # Convert the updated adjacency matrices back to the graph structure
    updated_graph = cv.adjs2graph(updated_adj, updated_badj)

    return updated_graph

def update_DD_BD(g_estimated, DD, BD, base_DD, base_BD, base_g):
    N = len(g_estimated)  # Assuming A and B are N x N matrices
    g_adj = cv.graph2adj(g_estimated)
    g_badj = cv.graph2badj(g_estimated)
    base_g_adj = cv.graph2adj(base_g)
    base_g_adjb = cv.graph2badj(base_g)
    g_adj = np.where(g_adj != base_g_adj, -1, 1)
    g_badj = np.where(g_badj != base_g_adjb, -1, 1)
    base_DD += np.multiply(g_adj,DD)
    base_BD += np.multiply(g_badj,BD)

    return base_DD, base_BD

def concat_dataset_batches(path=None):
    def sort_key(filename):
        # Use regex to match '_batch_' followed by one or more digits, ending with '.zkl'
        match = re.search(r'_batch(\d+)\.zkl$', filename)
        if match:
            # Extract the batch number and convert it to an integer
            return int(match.group(1))
        return float('inf')  # Return a large number if no match (place those at the end)

    items = listdir(path)
    items = [item for item in items if not item.startswith('.')]
    # Sort the items based on the extracted number

    items.sort(key=sort_key)
    dataset = []
    for item in items:
        curr = zkl.load(path + '/' + item)
        dataset.append(curr)
    zkl.save(dataset,'/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/datasets/VAR_BOLD_standatd_Gis_extended_end_time_ringmore_V_ruben_undersampled_by_100.zkl')