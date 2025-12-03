from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from gunfolds.utils import zickle as zkl
from os import listdir

undersampling = 4

deg_list = [1.5, 2.0, 2.5]
node_directories = ['6 nodes', '7 nodes', '8 nodes', '9 nodes']
list_of_lists = []
list_of_lists4 = []
list_of_lists5 = []

# Load data
# for directory in node_directories:
#     file_list = listdir(f'./results/edge_breaking_optN/res_CAP_0_/{directory}')
#     file_list.sort()
#     if file_list[0].startswith('.'):
#         file_list.pop(0)
#     item_list = [zkl.load(f'./results/edge_breaking_optN/res_CAP_0_/{directory}/{name}') for name in file_list]
#     list_of_lists.append(item_list)

for directory in node_directories:
    file_list = listdir(f'./res/weighted_experiment_capped/{directory}')
    file_list.sort()
    if file_list[0].startswith('.'):
        file_list.pop(0)
    item_list = [zkl.load(f'./res/weighted_experiment_capped/{directory}/{name}') for name in file_list]
    list_of_lists4.append(item_list)

for directory in node_directories:
    file_list = listdir(f'./results/edge_breaking_optN_new/{directory}')
    file_list.sort()
    if file_list[0].startswith('.'):
        file_list.pop(0)
    item_list = [zkl.load(f'./results/edge_breaking_optN_new/{directory}/{name}') for name in file_list]
    list_of_lists5.append(item_list)

# Create DataFrame
df = pd.DataFrame()
Err, method, ErrType, u, deg, node = [], [], [], [], [], []

for index, item_list in enumerate(list_of_lists4, start=6):
    for item in item_list:
        if item['u'] == undersampling and item['density'] in deg_list:
            Err.extend([item['normed_errors']['total'][0], item['normed_errors']['total'][1]])
            ErrType.extend(['omm', 'comm'])
            method.extend(['sRASL'] * 2)
            u.extend([item['u'], item['u']])
            deg.extend([item['density'], item['density']])
            node.extend([str(index)] * 2)

# for index, item_list in enumerate(list_of_lists, start=6):
#     for item in item_list:
#         if item['u'] == undersampling and item['density'] in deg_list:
#             Err.extend([item['normed_errors']['total'][0], item['normed_errors']['total'][1]])
#             ErrType.extend(['omm', 'comm'])
#             method.extend(['new_optN'] * 2)
#             u.extend([item['u'], item['u']])
#             deg.extend([item['density'], item['density']])
#             node.extend([str(index)] * 2)

for index, item_list in enumerate(list_of_lists5, start=6):
    for item in item_list:
        if item['u'] == undersampling and item['density'] in deg_list:
            Err.extend([item['normed_errors']['total'][0], item['normed_errors']['total'][1]])
            ErrType.extend(['omm', 'comm'])
            method.extend(['Ours'] * 2)
            u.extend([item['u'], item['u']])
            deg.extend([item['density'], item['density']])
            node.extend([str(index)] * 2)

df['Err'] = Err
df['method'] = method
df['ErrType'] = ErrType
df['deg'] = deg
df['node'] = node
df['u'] = u

# Set theme and style
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
pal = dict(sRASL="gold", Ours="blue", new_optN_update="red")

# Create the facet grid
g = sns.FacetGrid(df, col="deg", row="ErrType", height=4, aspect=0.5, margin_titles=True)

# Custom boxplot function
def custom_boxplot(*args, **kwargs):
    sns.boxplot(*args, **kwargs, palette=pal, linewidth=1)

g.map_dataframe(custom_boxplot, x='node', y='Err', hue="method")
g.add_legend(title="Method")
g.set_axis_labels("Number of Nodes", "Normalized Error")

# Customize axes and add titles
for i in range(g.axes.shape[0]):
    for j in range(g.axes.shape[1]):
        ax = g.facet_axis(i, j)
        ax.set_ylim(0, 1)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)

# Adjust spacing and layout
g.fig.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)
g.fig.suptitle("Comparison of Methods Across Densities and Error Types", fontsize=16)
# plt.style.use('Solarize_Light2')
# Save the figure
plt.savefig(f"paper_quality_figure_u_{undersampling}.svg")
# plt.show()