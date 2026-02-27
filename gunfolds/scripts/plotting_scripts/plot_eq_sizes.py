from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from gunfolds.utils import zickle as zkl
from os import listdir
from gunfolds.viz import gtool as gt
from collections import defaultdict
from gunfolds.utils import graphkit as gk


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


############################################################

folder14 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/stable_trans_mat/PCMCI/prioriti42531_alpha1_my_weights/u2/'

file_list14 = listdir(folder14)
file_list14.sort()
if file_list14[0].startswith('.'):
    file_list14.pop(0)

res14 = [zkl.load(folder14 + file) for file in file_list14]

############################################################

folder15 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/stable_trans_mat/PCMCI/prioriti42531_alpha1_my_weights/u3/'

file_list15 = listdir(folder15)
file_list15.sort()
if file_list15[0].startswith('.'):
    file_list15.pop(0)

res15 = [zkl.load(folder15 + file) for file in file_list15]

############################################################

folder16 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/stable_trans_mat/PCMCI/prioriti42531_alpha1_my_weights/u4/'

file_list16 = listdir(folder16)
file_list16.sort()
if file_list16[0].startswith('.'):
    file_list16.pop(0)

res16 = [zkl.load(folder16 + file) for file in file_list16]

############################################################



if __name__ == '__main__':
    df = pd.DataFrame()
    u = []
    eq_size = []

    for item in res14:
        eq_size.append(len(item['general']['full_sols']))
        u.append('U2')
    for item in res15:
        eq_size.append(len(item['general']['full_sols']))
        u.append('U3')
    for item in res16:
        eq_size.append(len(item['general']['full_sols']))
        u.append('U4')

    df['eq_size'] = eq_size
    df['u'] = u

    sns.set({"xtick.minor.size": 0.2})
    pal = dict(U2="gold", U3="blue",
               U4="maroon", U5="green", U6="red", U4_New="yellow")
    g = sns.FacetGrid(df, height=4, aspect=1.5, margin_titles=True)


    def custom_boxplot(*args, **kwargs):
        sns.boxplot(*args, **kwargs, palette='Set1')

    g.map_dataframe(custom_boxplot, x='u', y='eq_size')
    g.add_legend()
    g.set_axis_labels("undersampling rate", "equivalence class size")
    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            ax = g.facet_axis(i, j)
            ax.set_ylim(0, max(eq_size))
    plt.suptitle("equivalence class size of optimized solutions per undersampling ", x=0.45, y=1, fontsize=12)
    # plt.show()
    plt.savefig("figs/PCMCI_my_weights_pr42531_eq_size.png")
