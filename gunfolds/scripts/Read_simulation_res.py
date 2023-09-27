from gunfolds.utils import zickle as zkl
import seaborn as sns
from os import listdir
from matplotlib import pyplot as plt
import numpy as np

folder = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/res_simulation' \
         '/8nodes/n8stmt14/'
folder2 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/res_simulation' \
         '/8nodes/n8stmf14/'
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

file_list2 = listdir(folder2)
file_list2.sort()
if file_list2[0].startswith('.'):
    file_list2.pop(0)

#activate this part if you want to run simulation on  results of previous simulation
not_n8stmt14 = []
for i in range(len(file_list)):
    item = zkl.load(folder + file_list[i])
    not_n8stmt14.append(item['GT'])
    print(item['intended_u_rate'])

zkl.save(not_n8stmt14, './datasets/graphs_not_in_n8stmt14.zkl')

not_n8stmf14 = []
for i in range(len(file_list2)):
    item = zkl.load(folder2 + file_list2[i])
    not_n8stmf14.append(item['GT'])
    print(item['intended_u_rate'])

zkl.save(not_n8stmf14, './datasets/graphs_not_in_n8stmf14.zkl')

# activate this part if time information is not in .zkl files and manually add them through [cats]
'''# for i in range(30):
#     item = zkl.load(folder + file_list[i])
#     item['mins'] = cats[i]
#     zkl.save(item, folder + file_list[i])'''
res = [zkl.load(folder + file) for file in file_list]
res2 = [zkl.load(folder2 + file) for file in file_list2]

G1_opt_error_GT_om = []
G1_opt_error_GT_com = []
Gu_opt_errors_network_GT_U_om = []
Gu_opt_errors_network_GT_U_com = []
Gu_opt_errors_g_estimated_om = []
Gu_opt_errors_g_estimated_com = []

times = []
for item in res:
    G1_opt_error_GT_om.append(item['G1_opt_error_GT'][0])
    G1_opt_error_GT_com.append(item['G1_opt_error_GT'][1])

    Gu_opt_errors_network_GT_U_om.append(item['Gu_opt_errors_network_GT_U'][0])
    Gu_opt_errors_network_GT_U_com.append(item['Gu_opt_errors_network_GT_U'][1])

    Gu_opt_errors_g_estimated_om.append(item['Gu_opt_errors_g_estimated'][0])
    Gu_opt_errors_g_estimated_com.append(item['Gu_opt_errors_g_estimated'][1])

    times.append(item['total_time'])

for item in res2:
    G1_opt_error_GT_om.append(item['G1_opt_error_GT'][0])
    G1_opt_error_GT_com.append(item['G1_opt_error_GT'][1])

    Gu_opt_errors_network_GT_U_om.append(item['Gu_opt_errors_network_GT_U'][0])
    Gu_opt_errors_network_GT_U_com.append(item['Gu_opt_errors_network_GT_U'][1])

    Gu_opt_errors_g_estimated_om.append(item['Gu_opt_errors_g_estimated'][0])
    Gu_opt_errors_g_estimated_com.append(item['Gu_opt_errors_g_estimated'][1])

    times.append(item['total_time'])

min_value = min(times)
max_value = max(times)

normalized_times = [(x - min_value) / (max_value - min_value) for x in times]

med1om = np.mean(Gu_opt_errors_network_GT_U_om[:35])
med2om = np.mean(Gu_opt_errors_network_GT_U_om[35:])

med1com = np.mean(Gu_opt_errors_network_GT_U_com[:35])
med2com = np.mean(Gu_opt_errors_network_GT_U_com[35:])

med1time = np.mean(normalized_times[:35])
med2time = np.mean(normalized_times[35:])

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.75})
sns.set_style("darkgrid")
fig, axes = plt.subplots(2, 1, figsize=(10, 8))  # 2 rows, 1 column


# First subplot (scatter plot)
sns.scatterplot(data=Gu_opt_errors_network_GT_U_om, ax=axes[0], s=50, alpha=.8, label='Gu_opt errors network_GT_U')
sns.scatterplot(data=normalized_times, ax=axes[0], s=50, alpha=.8, color='red', label='normalized_times')
sns.lineplot(data=Gu_opt_errors_network_GT_U_om, ax=axes[0], alpha=.5, linewidth=1)
sns.lineplot(data=normalized_times, ax=axes[0], alpha=.5, linewidth=1)
axes[0].axhline(y=med1om, color='blue', linestyle='-', xmin=0/73, xmax=35/73, label=f'mean Error = {med1om}')
axes[0].axhline(y=med2om, color='blue', linestyle='-', xmin=35/73, xmax=73/73, label=f'mean Error = {med2om}')
axes[0].axhline(y=med1time, color='red', linestyle='--', xmin=0/73, xmax=35/73, label=f'mean Time = {med1time}')
axes[0].axhline(y=med2time, color='red', linestyle='--', xmin=35/73, xmax=73/73, label=f'mean Time = {med2time}')
axes[0].set_title('Plot of optimization error and time-to-solve for random 8-node graphs')
axes[0].set_xlim(0, len(Gu_opt_errors_network_GT_U_om))
axes[0].set_ylim(0, 1)
axes[0].set_ylabel('Normalized Omission\n Error')
axes[0].yaxis.label.set_color('blue')
axes[0].tick_params(axis='y', colors='blue')

# Add a secondary y-axis (right-side y-axis) for the first subplot
axes2_0 = axes[0].twinx()
axes2_0.set_ylim(0, 100)
axes2_0.set_ylabel('Hours')
axes2_0.yaxis.label.set_color('red')
axes2_0.tick_params(axis='y', colors='red')
axes2_0.grid(False)


# Second subplot (scatter plot)
sns.scatterplot(data=Gu_opt_errors_network_GT_U_com, ax=axes[1], s=50, alpha=.8)
sns.lineplot(data=Gu_opt_errors_network_GT_U_com, ax=axes[1], alpha=.5, linewidth=1)
axes[1].set_xlim(0, len(Gu_opt_errors_network_GT_U_com))
axes[1].axhline(y=med1com, color='blue', linestyle='-', xmin=0/73, xmax=35/73, label=f'mean Error = {med1com}')
axes[1].axhline(y=med2com, color='blue', linestyle='-', xmin=35/73, xmax=73/73, label=f'mean Error = {med2com}')
axes[1].set_ylim(0, 1)
axes[1].set_ylabel('Normalized Commission\n Error')
axes[1].yaxis.label.set_color('blue')
axes[1].tick_params(axis='y', colors='blue')

# Add a secondary y-axis (right-side y-axis) for the second subplot
axes2_1 = axes[1].twinx()
axes2_1.set_ylim(0, 100)
axes2_1.set_ylabel('Hours')
axes2_1.yaxis.label.set_color('red')
axes2_1.tick_params(axis='y', colors='red')
axes2_1.grid(False)

for ax in axes:
    ax.xaxis.grid(True, "minor", linewidth=.75)
    ax.xaxis.grid(True, "major", linewidth=5)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.xaxis.set_major_locator(plt.MultipleLocator(35))

# plt.show()
plt.savefig("simulation_results_8nodes_with_mean.svg")

