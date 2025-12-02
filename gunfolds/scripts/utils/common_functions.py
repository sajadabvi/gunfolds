"""
Common utility functions shared across scripts.

This module extracts frequently-used functions from my_functions.py and other scripts
to provide a centralized, well-documented utility library.
"""
import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, Tuple, List, Any


# ============================================================================
# GRAPH MANIPULATION
# ============================================================================

def remove_bidir_edges(input_dict: Dict) -> Dict:
    """
    Remove bidirectional edges from a graph dictionary.
    
    Args:
        input_dict: Graph represented as nested dictionary
        
    Returns:
        Graph with bidirectional edges removed
    """
    new_dict = {}
    for key in input_dict:
        new_dict[key] = {}
        for inner_key in input_dict[key]:
            if input_dict[key][inner_key] != 2:
                new_dict[key][inner_key] = input_dict[key][inner_key]
    return new_dict


def find_two_cycles(graph: nx.DiGraph) -> List[Tuple[int, int]]:
    """
    Find all 2-cycles (bidirectional edges) in a directed graph.
    
    Args:
        graph: NetworkX directed graph
        
    Returns:
        List of tuples representing 2-cycles [(node1, node2), ...]
    """
    two_cycles = []
    visited_pairs = set()
    
    for node1 in graph.nodes():
        for node2 in graph.successors(node1):
            if node2 in graph.predecessors(node1):
                pair = tuple(sorted([node1, node2]))
                if pair not in visited_pairs:
                    two_cycles.append((node1, node2))
                    visited_pairs.add(pair)
    
    return two_cycles


def convert_nodes_to_numbers(graph: nx.Graph) -> nx.Graph:
    """
    Convert node labels to numbers (0-indexed).
    
    Args:
        graph: NetworkX graph with arbitrary node labels
        
    Returns:
        Graph with nodes relabeled as 0, 1, 2, ...
    """
    mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
    return nx.relabel_nodes(graph, mapping)


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def precision_recall(estimated_graph: Dict, ground_truth: Dict, 
                    include_selfloop: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Calculate precision, recall, and F1 scores for graph estimation.
    
    Computes metrics for three aspects:
    - Orientation: Correctness of edge directions
    - Adjacency: Presence of edges (ignoring direction)
    - Cycle: Detection of 2-cycles
    
    Args:
        estimated_graph: Estimated graph as nested dictionary
        ground_truth: True graph as nested dictionary
        include_selfloop: Whether to include self-loops in evaluation
        
    Returns:
        Dictionary with precision, recall, F1 for each metric type
        
    Example:
        >>> metrics = precision_recall(est_g, gt_g)
        >>> print(f"Orientation F1: {metrics['orientation']['F1']:.3f}")
    """
    # Initialize counters
    orientation_tp = orientation_fp = orientation_fn = 0
    adjacency_tp = adjacency_fp = adjacency_fn = 0
    cycle_tp = cycle_fp = cycle_fn = 0
    
    # Get all nodes
    all_nodes = set(estimated_graph.keys()) | set(ground_truth.keys())
    
    for node1 in all_nodes:
        for node2 in all_nodes:
            if not include_selfloop and node1 == node2:
                continue
            
            # Get edge types (0=no edge, 1=directed, 2=bidirectional)
            est_edge = estimated_graph.get(node1, {}).get(node2, 0)
            gt_edge = ground_truth.get(node1, {}).get(node2, 0)
            
            est_rev = estimated_graph.get(node2, {}).get(node1, 0)
            gt_rev = ground_truth.get(node2, {}).get(node1, 0)
            
            # Orientation metric (exact edge direction)
            if est_edge > 0 and gt_edge > 0:
                orientation_tp += 1
            elif est_edge > 0 and gt_edge == 0:
                orientation_fp += 1
            elif est_edge == 0 and gt_edge > 0:
                orientation_fn += 1
            
            # Adjacency metric (edge exists, ignoring direction)
            est_adjacent = (est_edge > 0) or (est_rev > 0)
            gt_adjacent = (gt_edge > 0) or (gt_rev > 0)
            
            if est_adjacent and gt_adjacent:
                adjacency_tp += 0.5  # Count each pair once
            elif est_adjacent and not gt_adjacent:
                adjacency_fp += 0.5
            elif not est_adjacent and gt_adjacent:
                adjacency_fn += 0.5
            
            # Cycle metric (2-cycles / bidirectional edges)
            if node1 < node2:  # Count each pair once
                est_cycle = (est_edge > 0) and (est_rev > 0)
                gt_cycle = (gt_edge > 0) and (gt_rev > 0)
                
                if est_cycle and gt_cycle:
                    cycle_tp += 1
                elif est_cycle and not gt_cycle:
                    cycle_fp += 1
                elif not est_cycle and gt_cycle:
                    cycle_fn += 1
    
    # Calculate metrics
    def calc_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {'precision': precision, 'recall': recall, 'F1': f1}
    
    return {
        'orientation': calc_metrics(orientation_tp, orientation_fp, orientation_fn),
        'adjacency': calc_metrics(adjacency_tp, adjacency_fp, adjacency_fn),
        'cycle': calc_metrics(cycle_tp, cycle_fp, cycle_fn)
    }


def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.
    
    Args:
        precision: Precision value (0-1)
        recall: Recall value (0-1)
        
    Returns:
        F1 score (0-1)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


# ============================================================================
# DATA LOADING AND PARSING
# ============================================================================

def parse_nodes(graph_str: str) -> List[str]:
    """
    Parse node names from Tetrad graph string.
    
    Args:
        graph_str: Graph string from Tetrad output
        
    Returns:
        List of node names
    """
    lines = graph_str.strip().split('\n')
    if not lines:
        return []
    
    # First line contains node list
    nodes_line = lines[0].replace('Graph Nodes:', '').strip()
    nodes = [n.strip() for n in nodes_line.split(';') if n.strip()]
    return nodes


def parse_edges(graph_str: str) -> List[Tuple[str, str, str]]:
    """
    Parse edges from Tetrad graph string.
    
    Args:
        graph_str: Graph string from Tetrad output
        
    Returns:
        List of (source, target, edge_type) tuples
    """
    lines = graph_str.strip().split('\n')
    edges = []
    
    # Skip header line, parse edges
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        # Parse edge format: "X --> Y" or "X --- Y" or "X <-> Y"
        if '-->' in line:
            parts = line.split('-->')
            edges.append((parts[0].strip(), parts[1].strip(), 'directed'))
        elif '<->' in line:
            parts = line.split('<->')
            edges.append((parts[0].strip(), parts[1].strip(), 'bidirected'))
        elif '---' in line:
            parts = line.split('---')
            edges.append((parts[0].strip(), parts[1].strip(), 'undirected'))
    
    return edges


def create_adjacency_matrix(edges: List[Tuple], nodes: List[str]) -> np.ndarray:
    """
    Create adjacency matrix from edge list.
    
    Args:
        edges: List of (source, target, type) tuples
        nodes: List of node names
        
    Returns:
        Adjacency matrix as numpy array
    """
    n = len(nodes)
    matrix = np.zeros((n, n))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    for edge in edges:
        if len(edge) >= 2:
            src, tgt = edge[0], edge[1]
            edge_type = edge[2] if len(edge) > 2 else 'directed'
            
            if src in node_to_idx and tgt in node_to_idx:
                i, j = node_to_idx[src], node_to_idx[tgt]
                matrix[i, j] = 1
                
                if edge_type in ['bidirected', 'undirected']:
                    matrix[j, i] = 1
    
    return matrix


def read_gimme_to_graph(beta_file_path: str, std_error_file_path: str) -> nx.DiGraph:
    """
    Read GIMME output files and convert to NetworkX graph.
    
    Args:
        beta_file_path: Path to beta coefficients CSV
        std_error_file_path: Path to standard errors CSV
        
    Returns:
        NetworkX directed graph with significant edges
    """
    # Read files
    betas = pd.read_csv(beta_file_path, index_col=0)
    std_errors = pd.read_csv(std_error_file_path, index_col=0)
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes
    for node in betas.columns:
        G.add_node(node)
    
    # Add edges where beta is significant (|beta| > 1.96 * std_error)
    for i, source in enumerate(betas.columns):
        for j, target in enumerate(betas.columns):
            beta = betas.iloc[i, j]
            stderr = std_errors.iloc[i, j]
            
            if abs(beta) > 1.96 * stderr:  # 95% confidence
                G.add_edge(source, target, weight=beta)
    
    return G


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def divide_into_batches(lst: List, batch_size: int) -> List[List]:
    """
    Divide a list into batches of specified size.
    
    Args:
        lst: List to divide
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def round_tuple_elements(input_tuple: Tuple, decimal_points: int = 3) -> Tuple:
    """
    Round numeric elements in a tuple to specified decimal places.
    
    Args:
        input_tuple: Tuple with numeric elements
        decimal_points: Number of decimal places
        
    Returns:
        Tuple with rounded values
    """
    return tuple(
        round(elem, decimal_points) if isinstance(elem, (int, float)) else elem 
        for elem in input_tuple
    )


def update_base_graph(base_graph: Dict, new_graph: Dict) -> Dict:
    """
    Update base graph with edges from new graph (union operation).
    
    Args:
        base_graph: Existing graph
        new_graph: New graph to merge
        
    Returns:
        Updated graph
    """
    updated = base_graph.copy()
    
    for node in new_graph:
        if node not in updated:
            updated[node] = {}
        for target, edge_type in new_graph[node].items():
            updated[node][target] = edge_type
    
    return updated


# ============================================================================
# DOCUMENTATION
# ============================================================================

__all__ = [
    'remove_bidir_edges',
    'find_two_cycles',
    'convert_nodes_to_numbers',
    'precision_recall',
    'calculate_f1_score',
    'parse_nodes',
    'parse_edges',
    'create_adjacency_matrix',
    'read_gimme_to_graph',
    'divide_into_batches',
    'round_tuple_elements',
    'update_base_graph',
]

