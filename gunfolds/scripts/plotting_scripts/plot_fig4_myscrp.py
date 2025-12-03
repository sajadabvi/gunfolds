import random
from gunfolds.utils import zickle as zkl
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.patches as mpatches
import numpy as np

PNUM = 4

PreFix = 'MVAR'
concat = False
POSTFIX = 'Ruben_data' + 'concat' if concat else 'individual'

now = str(datetime.now())
now = now[:-7].replace(' ', '_')

###saving files
filename =  now + '_'

#MVAR
# Data for group 1
data_group1 = zkl.load("/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/RASL_fig4.zkl")

# Data for group 2
data_group2 = zkl.load("/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/FASK_fig4.zkl")

data_group3 = zkl.load("/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/PCMCI_fig4.zkl")

data_group4 = zkl.load("/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/GIMME_fig4.zkl")

data_group5 = zkl.load("/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/MVAR_fig4.zkl")

data_group6 = zkl.load("/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/MVGC_fig4.zkl")

# Labels and titles for subplots
titles = ['Orientation', 'Adjacency', '2 cycles']
colors = ['blue', 'orange', 'red', 'yellow', 'green','purple']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

for i, (data1, data2
        , data3, data4, data5, data6
        , title) in enumerate(zip(data_group1, data_group2
                                  , data_group3, data_group4, data_group5, data_group6
                                     , titles)):
    ax1 = axes[i]

    bplots = []
    bplots.append(
        ax1.boxplot(data1, positions=np.array(range(len(data1))) * 2.0 - 0.4, patch_artist=True, showmeans=True,
                    widths=0.2))
    bplots.append(
        ax1.boxplot(data2, positions=np.array(range(len(data2))) * 2.0 - 0.2, patch_artist=True, showmeans=True,
                    widths=0.2))
    bplots.append(
        ax1.boxplot(data3, positions=np.array(range(len(data3))) * 2.0, patch_artist=True, showmeans=True, widths=0.2))
    bplots.append(
        ax1.boxplot(data4, positions=np.array(range(len(data4))) * 2.0 + 0.2, patch_artist=True, showmeans=True,
                    widths=0.2))
    bplots.append(
        ax1.boxplot(data5, positions=np.array(range(len(data5))) * 2.0 + 0.4, patch_artist=True, showmeans=True,
                    widths=0.2))
    bplots.append(
        ax1.boxplot(data6, positions=np.array(range(len(data6))) * 2.0 + 0.6, patch_artist=True, showmeans=True,
                    widths=0.2))

    # Set the face colors and transparency for each box
    for bplot, color in zip(bplots, colors):
        for patch in bplot['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    # Plot individual data points for group 1

    for j in range(len(data1)):
        ax1.plot(np.ones_like(data1[j]) * (j * 2.0 - 0.4)+ np.random.uniform(-0.05, 0.05, size=len(data1[j]))
                 , data1[j], 'o', color='black', alpha=0.5, markersize=1)

    for j in range(len(data2)):
        ax1.plot(np.ones_like(data2[j]) * (j * 2.0 - 0.2)+ np.random.uniform(-0.05, 0.05, size=len(data2[j]))
                 , data2[j], 'o', color='black', alpha=0.5, markersize=1)

    for j in range(len(data3)):
        ax1.plot(np.ones_like(data3[j]) * (j * 2.0 )+ np.random.uniform(-0.05, 0.05, size=len(data3[j]))
                 , data3[j], 'o', color='black', alpha=0.5, markersize=1)

    for j in range(len(data4)):
        ax1.plot(np.ones_like(data4[j]) * (j * 2.0 + 0.2)+ np.random.uniform(-0.05, 0.05, size=len(data4[j]))
                 , data4[j], 'o', color='black', alpha=0.5, markersize=1)

    for j in range(len(data5)):
        ax1.plot(np.ones_like(data5[j]) * (j * 2.0 + 0.4)+ np.random.uniform(-0.05, 0.05, size=len(data5[j]))
                 , data5[j], 'o', color='black', alpha=0.5, markersize=1)

    for j in range(len(data6)):
        ax1.plot(np.ones_like(data6[j]) * (j * 2.0 + 0.6)+ np.random.uniform(-0.05, 0.05, size=len(data6[j]))
                 , data6[j], 'o', color='black', alpha=0.5, markersize=1)


    ax1.set_xticks(range(0, len(data1) * 2, 2))
    ax1.set_xticklabels(['Precision', 'Recall', 'F1-score'])
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Value')
    ax1.set_title(f'({title})')
    ax1.grid(True)
    ax1.set_ylim(0, 1)

# Add super title
# plt.suptitle('Comparison of methods on data from simple Networks of R')
# Legend
blue_patch = mpatches.Patch(color='blue', label='RASL')
orange_patch = mpatches.Patch(color='orange', label='FASK')
red_patch = mpatches.Patch(color='red', label='PCMCI')
yellow_patch = mpatches.Patch(color='yellow', label='GIMME')
green_patch = mpatches.Patch(color='green', label='MVAR')
purple_patch = mpatches.Patch(color='purple', label='MVGC')
plt.legend(handles=[blue_patch, orange_patch
    , red_patch, yellow_patch, green_patch, purple_patch
                    ], loc='upper right')

plt.tight_layout()

# Save the figure
plt.savefig(filename + 'fig4.svg')
# plt.show()
plt.close()
