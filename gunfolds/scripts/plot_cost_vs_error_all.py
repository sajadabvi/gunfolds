import matplotlib.pyplot as plt
import pandas as pd
from gunfolds.utils import zickle as zkl
from os import listdir, makedirs
from os.path import exists
from gunfolds.utils import graphkit as gk
from gunfolds.utils import bfutils

# Function to load data from a given folder
def load_data_from_folder(folder_path):
    file_list = [f for f in sorted(listdir(folder_path)) if not f.startswith('.')]
    data = [zkl.load(folder_path + file) for file in file_list]
    return data

# Function to process traditional error
def process_item_traditional_error(item, weight_scheme, err_criteria):
    key = 'G1OptVsGT'
    # Omission Error
    err_value_omm = gk.OCE(item[key]['G1_opt_WRT_' + key], item['general']['GT'],
                           undirected=False, normalized=True)[err_criteria][0]
    return err_value_omm

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
        {'data': data_dict['u2'], 'weight_scheme': 'U2_mean_error'},
        {'data': data_dict['u3'], 'weight_scheme': 'U3_mean_error'},
        {'data': data_dict['u4'], 'weight_scheme': 'U4_mean_error'},
    ]

    x_values = []
    y_values = []

    # Combine data for all items
    for dataset_info in datasets_info:
        data = dataset_info['data']
        weight_scheme = dataset_info['weight_scheme']
        for item in data:
            for sol in item['general']['full_sols']:
                x = gk.OCE(
                    bfutils.num2CG(sol[0][0], len(item['general']['GT'])),
                    item['general']['GT'],
                    undirected=False,
                    normalized=True
                )['total'][1]
                y = sol[1]  # Extracting the y-value as described
                x_values.append(x)
                y_values.append(y)

    # Plotting all data in one figure
    plt.figure(figsize=(10, 6))
    plt.scatter(y_values, x_values, c='blue', alpha=0.6, edgecolors='k')  # Reversed axes
    plt.title('Combined Scatter Plot: Commission Error vs Optimization Cost')
    plt.xlabel('Optimization Cost')  # Reversed axis labels
    plt.ylabel('Commission Error')
    plt.grid(True)
    plt.tight_layout()

    # Save the combined plot
    combined_plot_filename = './pic/combined_plot.svg'
    if not exists('./pic'):
        makedirs('./pic')
    plt.savefig(combined_plot_filename)
    plt.show()