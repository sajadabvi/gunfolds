import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from gunfolds.utils import zickle as zkl
from os import listdir
from gunfolds.utils import graphkit as gk

# Function to load data from a given folder
def load_data_from_folder(folder_path):
    file_list = [f for f in sorted(listdir(folder_path)) if not f.startswith('.')]
    data = [zkl.load(folder_path + file) for file in file_list]
    return data

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

# Main script
if __name__ == '__main__':
    err_criteria = 'total'
    base_folder = '/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/stable_trans_mat/PCMCI/prioriti42531/'

    # Load data once per folder
    data_dict = {
        'u2': load_data_from_folder(base_folder + 'u2/'),
        'u3': load_data_from_folder(base_folder + 'u3/'),
        'u4': load_data_from_folder(base_folder + 'u4/'),
    }

    datasets_info = [
        {'data': data_dict['u2'], 'weight_scheme': 'U2_traditional_error'},
        {'data': data_dict['u3'], 'weight_scheme': 'U3_traditional_error'},
        {'data': data_dict['u4'], 'weight_scheme': 'U4_traditional_error'},
    ]

    data_records = []

    for dataset_info in datasets_info:
        data = dataset_info['data']
        weight_scheme = dataset_info['weight_scheme']
        for item in data:
            data_list = process_item_traditional_error(item, weight_scheme, err_criteria)
            data_records.extend(data_list)

    # Create DataFrame
    df = pd.DataFrame(data_records)

    # Filter data for "G1VsGT" and "G1OptVsGT"
    df_filtered = df[
        (df['WRT'] == 'G1OptVsGT') &
        (df['ErrVs'] == 'G1VsGT')
    ]

    # Set poster-style context
    sns.set_context("poster", font_scale=1.5)
    sns.set_style("darkgrid")

    # Plotting
    g = sns.FacetGrid(df_filtered, row="ErrType", height=10, aspect=0.7, margin_titles=True)

    def custom_boxplot_with_points(data, x, y, hue, **kwargs):
        sns.boxplot(data=data, x=x, y=y, hue=hue, palette='Set1', showfliers=False, **kwargs)
        sns.stripplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            dodge=True,
            jitter=True,
            size=14,
            marker='o',
            edgecolor='black',
            linewidth=1,
            alpha=0.45,
            palette='dark:black',
            **{k: v for k, v in kwargs.items() if k != 'color'}
        )
        # plt.legend([], [], frameon=False)

    g.map_dataframe(custom_boxplot_with_points, x='ErrVs', y='Err', hue='weights_scheme')
    # g.add_legend()
    g.set_axis_labels("Error Type", "Normalized Error")
    g.set_titles(size=20)

    # Set y-axis limits for all plots
    for ax in g.axes.flat:
        ax.set_ylim(0, 1)

    plt.suptitle("G1OptVsGT: Traditional Error", x=0.45, y=1.08, fontsize=24)
    plt.tight_layout()
    plt.show()