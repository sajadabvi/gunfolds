from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from gunfolds.utils import zickle as zkl
from os import listdir
from matplotlib.ticker import MultipleLocator

CAPSIZE = 0
num_nodes = 30
density_dict = [0.2, 0.25, 0.3]
degree_lsit = [0.9, 2, 3, 5]
undersampling_dict = [2, 3, 4]

undersampling = 4

node_directories = ['6 nodes', '7 nodes', '8 nodes', '9 nodes']
list_of_lists = []

node_directories2 = ['6', '7', '8', '9']
list_of_lists2 = []

for directory in node_directories:
    file_list = listdir(f'./results/weighted_zero_cap/{directory}')
    file_list.sort()
    if file_list[0].startswith('.'):
        file_list.pop(0)
    item_list = [zkl.load(f'./results/weighted_zero_cap/{directory}/{name}') for name in file_list]
    list_of_lists.append(item_list)

for directory in node_directories2:
    file_list = listdir(f'./res_drasl_after_optim/{directory}')
    file_list.sort()
    if file_list[0].startswith('.'):
        file_list.pop(0)
    item_list2 = [zkl.load(f'./res_drasl_after_optim/{directory}/{name}') for name in file_list]
    list_of_lists2.append(item_list2)

if __name__ == '__main__':
    df = pd.DataFrame()
    Err = []
    method = []
    ErrType = []
    u = []
    deg = []
    node = []

    for index, item_list in enumerate(list_of_lists, start=6):
        for item in item_list:
            if item['u'] == undersampling:
                Err.extend([item['normed_errors']['total'][0], item['normed_errors']['total'][1]])
                ErrType.extend(['omm', 'comm'])
                method.extend(['optim'] * 2)
                u.extend([item['u'], item['u']])
                deg.extend([item['density'], item['density']])
                node.extend([str(index)] * 2)

    for index, item_list2 in enumerate(list_of_lists2, start=6):
        for item in item_list2:
            if item['u'] == undersampling:
                Err.extend([item['normed_errors']['total'][0], item['normed_errors']['total'][1]])
                ErrType.extend(['omm', 'comm'])
                method.extend(['optim_then_sRASL'] * 2)
                u.extend([item['u'], item['u']])
                deg.extend([item['density'], item['density']])
                node.extend([str(index)] * 2)

    df['Err'] = Err
    df['method'] = method
    df['ErrType'] = ErrType
    df['deg'] = deg
    df['node'] = node
    df['u'] = u

    sns.set({"xtick.minor.size": 0.2})
    g = sns.FacetGrid(df, col="deg", row="ErrType", hue="method",height=4, aspect=0.4, margin_titles=True)
    g.map_dataframe(sns.boxplot, x='node', y='Err')
    g.add_legend()
    g.set_axis_labels("Number of node", "normalized error")

    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            ax = g.facet_axis(i, j)
            ax.xaxis.grid(True, "minor", linewidth=.75)
            ax.xaxis.grid(True, "major", linewidth=3)
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.set_ylim(0, 1)

    plt.show()
    # plt.savefig("figs/4.svg")
