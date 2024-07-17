import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.patches as mpatches
import argparse
import distutils.util
from gunfolds.utils import zickle  as zkl
from os import listdir
import glob

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-p", "--PNUM", default=4, help="number of CPUs in machine.", type=int)
parser.add_argument("-c", "--CONCAT", default="t", help="true to use concat data", type=str)
parser.add_argument("-u", "--UNDERSAMPLED", default="f", help="true to use tr 3 time scale", type=str)
args = parser.parse_args()
PNUM = args.PNUM
UNDERSAMPLED = bool(distutils.util.strtobool(args.UNDERSAMPLED))
for TR in['3s', '1.20s']:
    for concat in [True, False]:

        POSTFIX = 'tepmporal_undetsampling_data' + 'concat' if concat else 'individual'

        save_results = []


        edge_weights = [1, 3, 1, 3, 2]
        method = 'FASK'
        PreFix = method + TR
        folder = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/data_group/' + method + '/'

        groups = listdir(folder)
        groups.sort()
        if groups[0].startswith('.'):
            groups.pop(0)

        # Initialize the nested list structure


        for i in groups:
            concatenated_data = [[[] for _ in range(3)] for _ in range(3)]
            pattern = folder  + str(i) + f'/*_{TR}_{concat}.zkl'
            # files = listdir(folder + '/' + str(i) + '/')
            files = glob.glob(pattern) #listdir(folder + '/' + str(i) + '/')

            files.sort()
            if files[0].startswith('.'):
                files.pop(0)

            for file in files:

                curr = zkl.load(file)
                for j in range(3):
                    for k in range(3):
                        concatenated_data[j][k].append(curr[j][k][0])

            exec(f"data_group{i} = concatenated_data")


        now = str(datetime.now())
        now = now[:-7].replace(' ', '_')

        ###saving files
        filename = PreFix + '_prior_'+''.join(map(str, edge_weights))+'_with_selfloop_net_' + str('all') + '_amp_' + now + '_' + (
            'concat' if concat else 'individual')


        # Labels and titles for subplots
        titles = ['Orientation', 'Adjacency', '2 cycles']
        colors = ['blue', 'orange', 'red', 'yellow', 'green','pink']

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

        for i, (
                # data0,
                data1, data2
                , data3, data4, data5,
                # data6,
                data7
                , title) in enumerate(zip(
            # data_group0,
            data_group1, data_group2
                                          , data_group3, data_group4, data_group5,
            # data_group6,
            data_group7
                                             , titles)):
            ax1 = axes[i]
            bplots = []

            # bplots.append(ax1.boxplot(data0, positions=np.array(range(len(data0))) * 2.0 - 0.6, patch_artist=True, showmeans=True,
            #             boxprops=dict(facecolor=colors[0]), widths=0.2))
            bplots.append(ax1.boxplot(data1, positions=np.array(range(len(data1))) * 2.0 - 0.4, patch_artist=True, showmeans=True,
                        boxprops=dict(facecolor=colors[0]), widths=0.2))
            bplots.append(ax1.boxplot(data2, positions=np.array(range(len(data2))) * 2.0 - 0.2, patch_artist=True, showmeans=True,
                        boxprops=dict(facecolor=colors[1]), widths=0.2))
            bplots.append(ax1.boxplot(data3, positions=np.array(range(len(data3))) * 2.0 , patch_artist=True, showmeans=True,
                        boxprops=dict(facecolor=colors[2]), widths=0.2))
            bplots.append(ax1.boxplot(data4, positions=np.array(range(len(data4))) * 2.0 + 0.2, patch_artist=True, showmeans=True,
                        boxprops=dict(facecolor=colors[3]), widths=0.2))
            bplots.append(ax1.boxplot(data5, positions=np.array(range(len(data5))) * 2.0 + 0.4, patch_artist=True, showmeans=True,
                        boxprops=dict(facecolor=colors[4]), widths=0.2))
            # bplots.append(ax1.boxplot(data6, positions=np.array(range(len(data6))) * 2.0 + 0.6, patch_artist=True, showmeans=True,
            #             boxprops=dict(facecolor=colors[6]), widths=0.2))
            bplots.append(ax1.boxplot(data7, positions=np.array(range(len(data7))) * 2.0 + 0.6, patch_artist=True, showmeans=True,
                        boxprops=dict(facecolor=colors[5]), widths=0.2))


            for bplot, color in zip(bplots, colors):
                for patch in bplot['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)

            # Plot individual data points for group 1
            # for j in range(len(data0)):
            #     ax1.plot(np.ones_like(data0[j]) * (j * 2.0 - 0.6) + np.random.uniform(-0.05, 0.05, size=len(data0[j]))
            #              , data0[j], 'o', color='black', alpha=0.5, markersize=1)

            for j in range(len(data1)):
                ax1.plot(np.ones_like(data1[j]) * (j * 2.0 - 0.4) + np.random.uniform(-0.05, 0.05, size=len(data1[j]))
                         , data1[j], 'o', color='black', alpha=0.5, markersize=1)

            for j in range(len(data2)):
                ax1.plot(np.ones_like(data2[j]) * (j * 2.0 - 0.2) + np.random.uniform(-0.05, 0.05, size=len(data2[j]))
                         , data2[j], 'o', color='black', alpha=0.5, markersize=1)

            for j in range(len(data3)):
                ax1.plot(np.ones_like(data3[j]) * (j * 2.0) + np.random.uniform(-0.05, 0.05, size=len(data3[j]))
                         , data3[j], 'o', color='black', alpha=0.5, markersize=1)

            for j in range(len(data4)):
                ax1.plot(np.ones_like(data4[j]) * (j * 2.0 + 0.2) + np.random.uniform(-0.05, 0.05, size=len(data4[j]))
                         , data4[j], 'o', color='black', alpha=0.5, markersize=1)

            for j in range(len(data5)):
                ax1.plot(np.ones_like(data5[j]) * (j * 2.0 + 0.4) + np.random.uniform(-0.05, 0.05, size=len(data5[j]))
                         , data5[j], 'o', color='black', alpha=0.5, markersize=1)

            # for j in range(len(data6)):
            #     ax1.plot(np.ones_like(data6[j]) * (j * 2.0 + 0.6)+ np.random.uniform(-0.05, 0.05, size=len(data6[j]))
            #              , data6[j], 'o', color='black', alpha=0.5, markersize=1)

            for j in range(len(data7)):
                ax1.plot(np.ones_like(data7[j]) * (j * 2.0 + 0.6) + np.random.uniform(-0.05, 0.05, size=len(data7[j]))
                         , data7[j], 'o', color='black', alpha=0.5, markersize=1)

            ax1.set_xticks(range(0, len(data1) * 2, 2))
            ax1.set_xticklabels(['Precision', 'Recall', 'F1-score'])
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Value')
            ax1.set_title(f'({title})')
            ax1.grid(True)
            ax1.set_ylim(0, 1)
            # ax2.set_ylim(0, 1)
            # ax3.set_ylim(0, 1)
        # Add super title
        plt.suptitle(method + ', TR= ' + TR + ' ' + ('concat' if concat else 'individual') + ' data')
        # Legend
        gray_patch = mpatches.Patch(color='gray', label='Ruben reported')
        blue_patch = mpatches.Patch(color='blue', label='ORG. GT')
        orange_patch = mpatches.Patch(color='orange', label='GT^2')
        red_patch = mpatches.Patch(color='red', label=method +'+bi+sRASL')
        yellow_patch = mpatches.Patch(color='yellow', label='mean error')
        green_patch = mpatches.Patch(color='green', label='least cost sol')
        purple_patch = mpatches.Patch(color='purple', label='multi indiv rasl')
        pink_patch = mpatches.Patch(color='pink', label='PCMCI+sRASL')
        plt.legend(handles=[
            # gray_patch,
            blue_patch, orange_patch
            , red_patch, yellow_patch, green_patch,
            # purple_patch,
            pink_patch
                            ], loc='upper right')

        plt.tight_layout()

        # Save the figure
        plt.savefig(filename + '_grouped_boxplot.png')
        plt.close()
