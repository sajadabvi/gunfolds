import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from gunfolds.utils import zickle as zkl
from os import listdir
from gunfolds.utils import graphkit as gk
from gunfolds.utils import bfutils

# Function to load data from a given folder
def load_data_from_folder(folder_path):
    file_list = [f for f in sorted(listdir(folder_path)) if not f.startswith('.')]
    data = [zkl.load(folder_path + file) for file in file_list]
    return data

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
    data_list.append({
        'Err': err_values_omm[0],  # Mean error for GuVsGTu (omm)
        'ErrVs': 'GuVsGTu',
        'ErrType': 'omm',
        'WRT': WRT_value,
        'num_sols': item['general']['num_sols'],
        'weights_scheme': weight_scheme,
    })
    # Commission Error
    err_values_comm = cal_mean_error(item, err_criteria, 1)
    data_list.append({
        'Err': err_values_comm[0],  # Mean error for GuVsGTu (comm)
        'ErrVs': 'GuVsGTu',
        'ErrType': 'comm',
        'WRT': WRT_value,
        'num_sols': item['general']['num_sols'],
        'weights_scheme': weight_scheme,
    })
    return data_list

# Main script
if __name__ == '__main__':
    plt.style.use('Solarize_Light2')
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

    data_records = []

    for dataset_info in datasets_info:
        data = dataset_info['data']
        weight_scheme = dataset_info['weight_scheme']
        for item in data:
            data_list = process_item_mean_error(item, weight_scheme, err_criteria)
            data_records.extend(data_list)

    # Create DataFrame
    df = pd.DataFrame(data_records)

    # Filter data for "GuVsGTu" and "GuOptVsGest"
    df_filtered = df[
        (df['WRT'] == 'GuOptVsGest') &
        (df['ErrVs'] == 'GuVsGTu')
    ]

    # Set poster-style context
    sns.set_context("poster", font_scale=1.5)
    sns.set_style("whitegrid")

    # Plotting
    g = sns.FacetGrid(df_filtered,
                      # row="ErrType",
                      height=10, aspect=0.85, margin_titles=True)

    def custom_boxplot_with_points(data, x, y, hue, **kwargs):
        sns.boxplot(data=data, x=x, y=y, hue=hue, palette='Set1', showfliers=False, **kwargs)
        sns.stripplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            dodge=True,
            jitter=0.25,
            size=14,
            marker='o',
            edgecolor='black',
            linewidth=1,
            alpha=0.45,
            color='black',
            **{k: v for k, v in kwargs.items() if k != 'color'}
        )
        # plt.legend([], [], frameon=False)

    g.map_dataframe(custom_boxplot_with_points, x='ErrVs', y='num_sols', hue='weights_scheme')
    # g.add_legend()
    g.set_axis_labels("Undersampling", "Size of Solution Set")
    g.set_titles(size=20)

    # Set y-axis limits for all plots
    for ax in g.axes.flat:
        ax.set_ylim(30, 100)

    plt.suptitle("Size of Solution Set across Undersampling", x=0.45, y=1.08, fontsize=24)
    plt.tight_layout()
    plt.show()
    # plt.savefig('eq_size_3Uss.svg')