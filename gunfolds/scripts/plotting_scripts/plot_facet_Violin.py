'''
Hint: since violin plt Split=True only works for two hue levels at the same time,
we need to process data two by two and save figure as .svg . Later we can merge
the two different svg files in Inkscape to get a violin plot with 4 colors.
'''




from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from gunfolds.utils import zickle as zkl
from os import listdir
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MultipleLocator

undersampling = 3
deg_list = [1.5,2,2.5]
node_directories = ['6 nodes', '7 nodes', '8 nodes', '9 nodes']
list_of_lists = []
list_of_lists3 = []
list_of_lists4 = []
node_directories2 = ['6', '7', '8', '9']
list_of_lists2 = []

# for directory in node_directories:
#     file_list = listdir(f'./results/weighted_zero_cap/{directory}')
#     file_list.sort()
#     if file_list[0].startswith('.'):
#         file_list.pop(0)
#     item_list = [zkl.load(f'./results/weighted_zero_cap/{directory}/{name}') for name in file_list]
#     list_of_lists.append(item_list)

for directory in node_directories:
    file_list = listdir(f'./res/optimization_then_drasl/{directory}')
    file_list.sort()
    if file_list[0].startswith('.'):
        file_list.pop(0)
    item_list = [zkl.load(f'./res/optimization_then_drasl/{directory}/{name}') for name in file_list]
    list_of_lists3.append(item_list)

# for directory in node_directories:
#     file_list = listdir(f'./res/weighted_experiment_capped/{directory}')
#     file_list.sort()
#     if file_list[0].startswith('.'):
#         file_list.pop(0)
#     item_list = [zkl.load(f'./res/weighted_experiment_capped/{directory}/{name}') for name in file_list]
#     list_of_lists4.append(item_list)

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

    for index, item_list in enumerate(list_of_lists3, start=6):
        for item in item_list:
            if item['u'] == undersampling and item['density'] in deg_list:
                Err.extend([item['normed_errors']['total'][0], item['normed_errors']['total'][1]])
                ErrType.extend(['omm', 'comm'])
                method.extend(['Capped_optim_the_sRASL'] * 2)
                u.extend([item['u'], item['u']])
                deg.extend([item['density'], item['density']])
                node.extend([str(index)] * 2)

    # for index, item_list in enumerate(list_of_lists4, start=6):
    #     for item in item_list:
    #         if item['u'] == undersampling and item['density'] in deg_list:
    #             Err.extend([item['normed_errors']['total'][0], item['normed_errors']['total'][1]])
    #             ErrType.extend(['omm', 'comm'])
    #             method.extend(['Capped_optim'] * 2)
    #             u.extend([item['u'], item['u']])
    #             deg.extend([item['density'], item['density']])
    #             node.extend([str(index)] * 2)
    #
    # for index, item_list in enumerate(list_of_lists, start=6):
    #     for item in item_list:
    #         if item['u'] == undersampling and item['density'] in deg_list:
    #             Err.extend([item['normed_errors']['total'][0], item['normed_errors']['total'][1]])
    #             ErrType.extend(['omm', 'comm'])
    #             method.extend(['UNCapped_optim'] * 2)
    #             u.extend([item['u'], item['u']])
    #             deg.extend([item['density'], item['density']])
    #             node.extend([str(index)] * 2)

    for index, item_list2 in enumerate(list_of_lists2, start=6):
        for item in item_list2:
            if item[0]['u'] == undersampling:
                Err.extend([item[1]['min_norm_err'][0], item[1]['min_norm_err'][1]])
                ErrType.extend(['omm', 'comm'])
                method.extend(['UNCapped_optim_then_sRASL'] * 2)
                u.extend([item[0]['u'], item[0]['u']])
                deg.extend([item[0]['density'], item[0]['density']])
                node.extend([str(index)] * 2)

    df['Err'] = Err
    df['method'] = method
    df['ErrType'] = ErrType
    df['deg'] = deg
    df['node'] = node
    df['u'] = u

    sns.set({"xtick.minor.size": 0.2})
    pal = dict(Capped_optim="gold", Capped_optim_the_sRASL="orangered",
               UNCapped_optim="blue", UNCapped_optim_then_sRASL="limegreen")
    g = sns.FacetGrid(df, col="deg", row="ErrType", height=4, aspect=1, margin_titles=True)


    def custom_violinplot(*args, **kwargs):
        sns.violinplot(*args, **kwargs, palette=pal)


    g.map_dataframe(custom_violinplot, x='node', y='Err', hue="method", split=True, scale="count", inner="point")
    g.add_legend()
    g.set_axis_labels("Number of nodes", "normalized error")
    g.set(ylim=(-0.1, 1))
    method_value_all = df["method"].unique()
    node_value_all = df["node"].unique()
    deg_value_all = df["deg"].unique()
    ErrType_value_all = df["ErrType"].unique()
    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            for node_value in node_value_all.tolist():
                for k in range(len(method_value_all.tolist())):
                    ax = g.facet_axis(i, j)
                    # Get the column values for the current facet
                    ErrType_value = ErrType_value_all[i]
                    deg_value = deg_value_all[j]
                    method_value = method_value_all[k]
                    # Filter the data for the current facet
                    data = df[(df["method"] == method_value) & (df["node"] == node_value) & (df["ErrType"] == ErrType_value) & (df["deg"] == deg_value)]
                    # Calculate counts for each violin half
                    counts = len(data)
                    # Add annotations
                    outline_effect = withStroke(linewidth=2, foreground="black")
                    outline_effect2 = withStroke(linewidth=4, foreground="white")
                    annotation_x = node_value_all.tolist().index(node_value) + (0.05 if k == 0 else -0.4)
                    annotation_y = 0
                    ax.text(annotation_x, annotation_y, str(counts), color=pal[method_value], fontsize=18, path_effects=[outline_effect,outline_effect])
    plt.savefig("figs/undersampling3v2.svg")

