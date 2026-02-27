from collections import defaultdict
from gunfolds.viz import gtool as gt

def merge_graphs(graphs, threshold_density):
    # Dictionary to store edge frequencies
    edge_frequencies = defaultdict(int)

    # Set to store all nodes in the input graphs
    all_nodes = set()

    # Count the frequencies of each edge in the list of graphs
    for graph in graphs:
        all_nodes.update(graph.keys())
        for node, connections in graph.items():
            for neighbor, edge_type in connections.items():
                edge_frequencies[(node, neighbor, edge_type)] += 1

    # Calculate the threshold for inclusion based on the provided density
    threshold = threshold_density * len(graphs)

    # Create the merged graph based on edges that meet the threshold
    merged_graph = {}
    for (node, neighbor, edge_type), frequency in edge_frequencies.items():
        if frequency >= threshold:
            if node not in merged_graph:
                merged_graph[node] = {}
            merged_graph[node][neighbor] = edge_type

    # Ensure all nodes are present in the merged graph
    for graph in graphs:
        for node in all_nodes:
            if node not in merged_graph:
                merged_graph[node] = {}

    return merged_graph

for i in range(100):
    gt.plotg(gis_u4[i], output='./figs/g_u4_'+str(i)+'.pdf')