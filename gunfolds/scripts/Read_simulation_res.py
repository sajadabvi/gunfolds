from gunfolds.utils import zickle as zkl
import seaborn as sns
from os import listdir
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

folder = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/res_simulation' \
         '/7nodes/zkl/'
cats = [1.49, 47.566, 180.738, 20.914, 132.105, 31.86, 55.764, 39.024, 6.292, 26.122, 0.979, 0.727, 4.468, 20.192,
        53.597, 0.938, 0.477, 30.932, 0.148, 0.577, 0.094, 19.898, 54.733, 65.322, 23.107, 26.225, 48.217, 0.802,
        45.076, 0.763]

min_value = min(cats)
max_value = max(cats)

normalized_cats = [(x - min_value) / (max_value - min_value) for x in cats]

file_list = listdir(folder)
file_list.sort()
if file_list[0].startswith('.'):
    file_list.pop(0)

# activate this part if time information is not in .zkl files and manually add them through [cats]
'''# for i in range(30):
#     item = zkl.load(folder + file_list[i])
#     item['mins'] = cats[i]
#     zkl.save(item, folder + file_list[i])'''
res = [zkl.load(folder + file) for file in file_list]

G1_opt_error_GT_om = []
G1_opt_error_GT_com = []
Gu_opt_errors_network_GT_U_om = []
Gu_opt_errors_network_GT_U_com = []
Gu_opt_errors_g_estimated_om = []
Gu_opt_errors_g_estimated_com = []

for item in res:
    G1_opt_error_GT_om.append(item['G1_opt_error_GT'][0])
    G1_opt_error_GT_com.append(item['G1_opt_error_GT'][1])

    Gu_opt_errors_network_GT_U_om.append(item['Gu_opt_errors_network_GT_U'][0])
    Gu_opt_errors_network_GT_U_com.append(item['Gu_opt_errors_network_GT_U'][1])

    Gu_opt_errors_g_estimated_om.append(item['Gu_opt_errors_g_estimated'][0])
    Gu_opt_errors_g_estimated_com.append(item['Gu_opt_errors_g_estimated'][1])

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.75})
sns.set_style("darkgrid")
# Create a figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 8))  # 2 rows, 1 column

# First subplot
sns.lineplot(data=G1_opt_error_GT_om, label='G1_opt error GT', ax=axes[0])
sns.lineplot(data=Gu_opt_errors_network_GT_U_om, label='Gu_opt errors network_GT_U', ax=axes[0])
sns.lineplot(data=Gu_opt_errors_g_estimated_om, label='Gu_opt errors g_estimated', ax=axes[0])
sns.lineplot(data=normalized_cats, label='normalized time', ax=axes[0])
axes[0].set_title('VAR Simulation on 7 node random graphs error for different SCC options')
axes[0].set_xlim(0, 29)
axes[0].set_ylim(0, 1)

# Second subplot
sns.lineplot(data=G1_opt_error_GT_com, label='G1 opt error GT', ax=axes[1])
sns.lineplot(data=Gu_opt_errors_network_GT_U_com, label='Gu_opt errors network_GT_U', ax=axes[1])
sns.lineplot(data=Gu_opt_errors_g_estimated_com, label='Gu_opt errors g_estimated', ax=axes[1])
axes[1].set_xlim(0, 29)
axes[1].set_ylim(0, 1)

for ax in axes:
    ax.xaxis.grid(True, "minor", linewidth=.75)
    ax.xaxis.grid(True, "major", linewidth=5)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(10))

plt.show()
# plt.savefig("simulation_results_7nodes.svg")
