from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from gunfolds.utils import zickle as zkl
from os import listdir
from matplotlib.ticker import MultipleLocator
from gunfolds.viz import gtool as gt
from collections import defaultdict
import re

def extract_numbers(input_string):
    threshold_match = re.search(r'_threshold_(\d+)', input_string)
    noise_match = re.search(r'_noise_(\d+)', input_string)
    gmin_match = re.search(r'_GMIN(\d+)', input_string)
    gmax_match = re.search(r'_GMAX_(\d+)', input_string)

    threshold = int(threshold_match.group(1)) if threshold_match else None
    noise = int(noise_match.group(1)) if noise_match else None
    gmin = int(gmin_match.group(1)) if gmin_match else None
    gmax = int(gmax_match.group(1)) if gmax_match else None

    return threshold, noise, gmin, gmax

def compare_elements(input_tuple, template=(5, 10, 3, 7)):

    if input_tuple[0] != template[0]:
        return ("threshold",0)

    if input_tuple[1] != template[1]:
        return ("noise",1)

    if input_tuple[2] != template[2] or input_tuple[3] != template[3]:
        return ("Clip_min_max",2)

    return ("default", -1)

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

folder1 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/' \
          'same_num_samples_gtdensdensity_2priority/8nodes/all/'


file_list1 = listdir(folder1)
file_list1.sort()
if file_list1[0].startswith('.'):
    file_list1.pop(0)

folder2 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/hyperpram_tuning/'


file_list2 = listdir(folder2)
file_list2.sort()
if file_list2[0].startswith('.'):
    file_list2.pop(0)


# res1 = [zkl.load(folder1 + file) for file in file_list1]





if __name__ == '__main__':
    df = pd.DataFrame()
    Err = []
    method = []
    ErrType = []
    u = []
    deg = []
    node = []
    WRT = []
    ErrVs = []
    weights_scheme = []

    param = 2

    for file in file_list2:
        if extract_numbers(file) == (5, 10, 3, 7):
            print('stop')
        if compare_elements(extract_numbers(file))[1] == param:
            res = zkl.load(folder2 + file)
            WRT.append(compare_elements(extract_numbers(file))[0])
            if compare_elements(extract_numbers(file))[1] == 0 or compare_elements(extract_numbers(file))[1] == 1:
                ErrVs.append(extract_numbers(file)[compare_elements(extract_numbers(file))[1]])
            elif compare_elements(extract_numbers(file))[1] == 2:
                ErrVs.append( str(extract_numbers(file)[2]) + '-' +str(extract_numbers(file)[3]))
            Err.append(res['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0])
            ErrType.append('omm')
            u.append('U' + str(res['general']['intended_u_rate']))

            WRT.append(compare_elements(extract_numbers(file))[0])
            if compare_elements(extract_numbers(file))[1] == 0 or compare_elements(extract_numbers(file))[1] == 1:
                ErrVs.append(extract_numbers(file)[compare_elements(extract_numbers(file))[1]])
            elif compare_elements(extract_numbers(file))[1] == 2:
                ErrVs.append(str(extract_numbers(file)[2]) + '-' + str(extract_numbers(file)[3]))
            Err.append(res['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1])
            ErrType.append('comm')
            u.append('U' + str(res['general']['intended_u_rate']))


    for file in file_list1:
        if param == 0:
            code = 'threshold'
        elif param == 1:
            code = 'noise'
        else:
            code = 'Clip_min_max'

        res = zkl.load(folder1 + file)
        WRT.append(code)
        if param == 0:
            ErrVs.append(5)
        elif param == 1:
            ErrVs.append(10)
        elif param == 2:
            ErrVs.append( str(3) + '-' +str(7))
        Err.append(res['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0])
        ErrType.append('omm')
        u.append('U' + str(res['general']['intended_u_rate']))

        WRT.append(code)
        if param == 0:
            ErrVs.append(5)
        elif param == 1:
            ErrVs.append(10)
        elif param == 2:
            ErrVs.append(str(3) + '-' + str(7))
        Err.append(res['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1])
        ErrType.append('comm')
        u.append('U' + str(res['general']['intended_u_rate']))



    df['Err'] = Err
    df['ErrVs'] = ErrVs
    df['ErrType'] = ErrType
    df['WRT'] = WRT
    df['u'] = u

    sns.set({"xtick.minor.size": 0.2})
    pal = dict(U2="gold", U3="blue",
               U4="maroon", U5="green",U6="red",U4_2000samlpes="yellow")
    g = sns.FacetGrid(df, col="WRT", row="ErrType", height=4, aspect=1.5, margin_titles=True)


    def custom_boxplot(*args, **kwargs):
        sns.boxplot(*args, **kwargs, palette='Set1')


    g.map_dataframe(custom_boxplot, x='ErrVs', y='Err', hue='u')
    g.add_legend()
    g.set_axis_labels("error type", "normalized error")

    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            ax = g.facet_axis(i, j)
            # ax.xaxis.grid(True, "minor", linewidth=.75)
            # ax.xaxis.grid(True, "major", linewidth=3)
            # ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.set_ylim(0, 1)

    plt.show()
    # plt.savefig("figs/VAR_sim_upto_4_undersampling_effect_number_samples.svg")
