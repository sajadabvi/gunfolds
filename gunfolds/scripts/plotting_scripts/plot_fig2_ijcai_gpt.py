from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from gunfolds.utils import zickle as zkl
from os import listdir
from collections import defaultdict
from gunfolds.utils import graphkit as gk
from gunfolds.utils import bfutils

# Set global font size for the plot
sns.set_context("notebook", font_scale=5)  # Adjust font_scale for desired size
sns.set_style("whitegrid")

# Function to load data from a given folder
def load_data_from_folder(folder_path):
    file_list = [f for f in sorted(listdir(folder_path)) if not f.startswith('.')]
    data = [zkl.load(folder_path + file) for file in file_list]
    return data

# Function to merge multiple graphs based on edge frequency
def merge_graphs(graphs, threshold_density):
    edge_frequencies = defaultdict(int)
    all_nodes = set()

    for graph in graphs:
        all_nodes.update(graph.keys())
        for node, connections in graph.items():
            for neighbor, edge_type in connections.items():
                edge_frequencies[(node, neighbor, edge_type)] += 1

    threshold = threshold_density * len(graphs)
    merged_graph = {}
    for (node, neighbor, edge_type), frequency in edge_frequencies.items():
        if frequency >= threshold:
            merged_graph.setdefault(node, {})[neighbor] = edge_type

    for node in all_nodes:
        merged_graph.setdefault(node, {})

    return merged_graph

# Function to calculate mean errors
def cal_mean_error(item, err_criteria, error_type):
    GuVsGTu, GuVsGest, G1VsGT = [], [], []
    gt = item['general']['GT']
    gtu = item['general']['GT_at_actual_U']
    g_est = item['general']['g_estimated']

    for sol in item['general']['full_sols']:
        g1 = bfutils.num2CG(sol[0][0], len(gt))
        gu = bfutils.undersample(g1, sol[0][1][0])
        GuVsGTu.append(gk.OCE(gu, gtu, undirected=False, normalized=True)[err_criteria][error_type])
        GuVsGest.append(gk.OCE(gu, g_est, undirected=False, normalized=True)[err_criteria][error_type])
        G1VsGT.append(gk.OCE(g1, gt, undirected=False, normalized=True)[err_criteria][error_type])

    return [
        sum(GuVsGTu) / len(GuVsGTu),
        sum(GuVsGest) / len(GuVsGest),
        sum(G1VsGT) / len(G1VsGT)
    ]

# Function to process items with traditional error calculation for G1VsGT and G1OptVsGT
def process_item_traditional_error(item, weight_scheme, err_criteria):
    data_list = []
    key = 'G1OptVsGT'
    # Omission Error
    err_value_omm = gk.OCE(item[key]['G1_opt_WRT_' + key], item['general']['GT'],
                           undirected=False, normalized=True)[err_criteria][0]
    data_list.append({
        'Err': err_value_omm,
        'ErrVs': 'G1VsGT',
        'ErrType': 'omm',
        'WRT': key,
        'weights_scheme': weight_scheme,
    })
    # Commission Error
    err_value_comm = gk.OCE(item[key]['G1_opt_WRT_' + key], item['general']['GT'],
                            undirected=False, normalized=True)[err_criteria][1]
    data_list.append({
        'Err': err_value_comm,
        'ErrVs': 'G1VsGT',
        'ErrType': 'comm',
        'WRT': key,
        'weights_scheme': weight_scheme,
    })
    return data_list

# Function to process items with mean error calculation for GuVsGTu and GuOptVsGest
def process_item_mean_error(item, weight_scheme, err_criteria):
    data_list = []
    WRT_value = 'GuOptVsGest'
    # Omission Error
    err_values_omm = cal_mean_error(item, err_criteria, 0)
    err_value_omm = err_values_omm[0]  # Index 0 corresponds to 'GuVsGTu'
    data_list.append({
        'Err': err_value_omm,
        'ErrVs': 'GuVsGTu',
        'ErrType': 'omm',
        'WRT': WRT_value,
        'weights_scheme': weight_scheme,
    })
    # Commission Error
    err_values_comm = cal_mean_error(item, err_criteria, 1)
    err_value_comm = err_values_comm[0]  # Index 0 corresponds to 'GuVsGTu'
    data_list.append({
        'Err': err_value_comm,
        'ErrVs': 'GuVsGTu',
        'ErrType': 'comm',
        'WRT': WRT_value,
        'weights_scheme': weight_scheme,
    })
    return data_list

# Main code
if __name__ == '__main__':
    err_criteria = 'total'
    base_folder = '/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/stable_trans_mat/PCMCI/prioriti42531/'

    # Load data once per folder
    data_dict = {
        'u2': load_data_from_folder(base_folder + 'u2/'),
        'u3': load_data_from_folder(base_folder + 'u3/'),
        'u4': load_data_from_folder(base_folder + 'u4/'),
    }

    # Define dataset information
    datasets_info = [
        {'data': data_dict['u2'], 'weight_scheme': 'U2_traditional_error', 'process_function': process_item_traditional_error},
        {'data': data_dict['u3'], 'weight_scheme': 'U3_traditional_error', 'process_function': process_item_traditional_error},
        {'data': data_dict['u4'], 'weight_scheme': 'U4_traditional_error', 'process_function': process_item_traditional_error},
        {'data': data_dict['u2'], 'weight_scheme': 'U2_mean_error', 'process_function': process_item_mean_error},
        {'data': data_dict['u3'], 'weight_scheme': 'U3_mean_error', 'process_function': process_item_mean_error},
        {'data': data_dict['u4'], 'weight_scheme': 'U4_mean_error', 'process_function': process_item_mean_error},
    ]

    data_records = []

    for dataset_info in datasets_info:
        data = dataset_info['data']
        weight_scheme = dataset_info['weight_scheme']
        process_function = dataset_info['process_function']
        for item in data:
            data_list = process_function(item, weight_scheme, err_criteria)
            data_records.extend(data_list)

    # Filter data according to the specific conditions
    df = pd.DataFrame(data_records)

    # Keep only traditional error for "G1VsGT" and "G1OptVsGT"
    traditional_error_df = df[
        (df['WRT'] == 'G1OptVsGT') &
        (df['ErrVs'] == 'G1VsGT') &
        (df['weights_scheme'].str.contains('traditional_error'))
    ]

    # Keep only mean error for "GuVsGTu" and "GuOptVsGest"
    mean_error_df = df[
        (df['WRT'] == 'GuOptVsGest') &
        (df['ErrVs'] == 'GuVsGTu') &
        (df['weights_scheme'].str.contains('mean_error'))
    ]

    # Combine the filtered data
    df_filtered = pd.concat([traditional_error_df, mean_error_df])

    # Plotting
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    sns.set({"xtick.minor.size": 0.2})
    g = sns.FacetGrid(df_filtered, col="WRT", row="ErrType", height=8, aspect=0.7, margin_titles=True)

    # def custom_boxplot_with_points(data, x, y, hue, **kwargs):
    #     sns.boxplot(data=data, x=x, y=y, hue=hue, palette='Set1', showfliers=False, **kwargs)
    #     sns.stripplot(data=data, x=x, y=y, hue=hue, size=10, jitter=True, dodge=True, **kwargs)
    def custom_boxplot_with_points(data, x, y, hue, **kwargs):
        # Plot the boxplot
        sns.boxplot(
            data=data, x=x, y=y, hue=hue, palette='Set1', showfliers=False, **kwargs
        )
        # Overlay the strip plot
        sns.stripplot(
            data=data,
            x=x,
            y=y,
            hue=hue,  # Respect grouping for positioning
            dodge=True,
            jitter=True,
            size=10,  # Adjust size for visibility
            marker='o',  # Circular marker
            edgecolor='black',  # Solid black border
            linewidth=1,  # Border width
            alpha=0.45,  # Transparency for the fill
            palette=None,  # Ignore palette
            **{k: v for k, v in kwargs.items() if k != 'color'}  # Avoid passing 'color' redundantly
        )
        # Remove duplicate hue legends caused by the strip plot
        plt.legend([], [], frameon=False)
    g.map_dataframe(custom_boxplot_with_points, x='ErrVs', y='Err', hue='weights_scheme')
    # g.add_legend()
    g.set_axis_labels("Error Type", "Normalized Error")

    # Custom titles for columns
    column_titles = ["G1OptVsGT", "GuOptVsGest"]
    for ax, title in zip(g.axes[0], column_titles):
        ax.set_title(title)
        ax.tick_params(labelsize=12)

    # Adjust y-axis limits
    for ax in g.axes.flatten():
        ax.set_ylim(0, 1)

    plt.suptitle(f"Calculating {err_criteria.capitalize()} Errors", x=0.45, y=1.02, fontsize=20)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"figs/selected_errors_{err_criteria}_error_with_points.png")