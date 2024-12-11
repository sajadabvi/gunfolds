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

    # Create a directory for saving the plots
    output_folder = './pic'
    if not exists(output_folder):
        makedirs(output_folder)

    # Generate plots for each item
    item_count = 0  # Counter to ensure unique filenames
    for dataset_info in datasets_info:
        data = dataset_info['data']
        weight_scheme = dataset_info['weight_scheme']
        for item in data:
            x_values = []
            y_values = []
            # x = process_item_traditional_error(item, weight_scheme, err_criteria)
            for sol in item['general']['full_sols']:
                x = gk.OCE(bfutils.num2CG(sol[0][0],len(item['general']['GT'])), item['general']['GT'], undirected=False, normalized=True)['total'][1]
                y = sol[1]  # Extracting the y-value as described
                x_values.append(x)
                y_values.append(y)

            # Plotting the data
            plt.figure(figsize=(10, 6))
            plt.scatter(y_values, x_values, c='blue', alpha=0.6, edgecolors='k')  # Reversed axes
            plt.title(f'Plot for Solution {item_count}: Commission Error vs Optimization Cost')
            plt.xlabel('Optimization Cost')  # Reversed axis labels
            plt.ylabel('Commission Error')
            plt.xlim(0, 700000)  # Set X-axis range
            plt.ylim(0, 0.67)
            plt.grid(True)
            plt.tight_layout()

            # Save the plot
            plot_filename = f'{output_folder}/item_{item_count}.svg'
            plt.savefig(plot_filename)
            plt.close()  # Close the figure to save memory
            item_count += 1