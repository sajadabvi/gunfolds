import os
from gunfolds.viz import gtool as gt
from gunfolds.utils import bfutils
import numpy as np
import pandas as pd
from gunfolds import conversions as cv
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib.patches as mpatches
from gunfolds.scripts.datasets.simple_networks import simp_nets
from gunfolds.scripts import my_functions as mf
from gunfolds.solvers.clingo_rasl import drasl
from gunfolds.utils import graphkit as gk
from gunfolds.utils.calc_procs import get_process_count
PNUM = int(get_process_count(1))

PreFix = 'MVGC_then_rasl'
concat = True
POSTFIX = 'Ruben_data' + 'concat' if concat else 'individual'

save_results = []

Precision_O = []
Recall_O = []
Precision_O2 = []
Recall_O2 = []
Precision_O3 = []
Recall_O3 = []

Precision_A = []
Recall_A = []
Precision_A2 = []
Recall_A2 = []
Precision_A3 = []
Recall_A3 = []

Precision_C = []
Recall_C = []
Precision_C2 = []
Recall_C2 = []
Precision_C3 = []
Recall_C3 = []

F1_O = []
F1_A = []
F1_C = []

F1_O2 = []
F1_A2 = []
F1_C2 = []

F1_O3 = []
F1_A3 = []
F1_C3 = []
for nn in [5]:

    for fl in range(1, 61):
        num = str(fl) if fl > 9 else '0' + str(fl)
        print('reading file:' + num)
        if not concat:
            data = pd.read_csv(
                os.path.expanduser('~/DataSets_Feedbacks/1. Simple_Networks/Network' + str(
                    nn) + '_amp/data_fslfilter/BOLDfslfilter_{0}.txt'.format(
                    num), delimiter='\t'))
        else:
            data = pd.read_csv(
                os.path.expanduser('~/DataSets_Feedbacks/1. Simple_Networks/Network' + str(
                    nn) + '_amp/data_fslfilter_concat/concat_BOLDfslfilter_{0}.txt'.format(
                    num), delimiter='\t'))

        network_GT = simp_nets(nn, True)

        dd = np.transpose(data.values)
        # folder = 'expo_to_mat/expo_to_mat_n' + str(nn) + '_' + ('concat' if concat else 'individual')
        # if not os.path.exists(folder):
        #     os.makedirs(folder)
        # savemat(folder + '/expo_to_mat_' + str(fl) + '.mat', {'dd': dd})

    for fl in range(1, 61):
        print('processing file:' + str(fl))

        folder_read = 'expo_to_mat/expo_to_py_n' + str(nn) + '_' + ('concat' if concat else 'individual')
        mat_data = loadmat(folder_read + '/mat_file_' + str(fl) + '.mat')
        mat = mat_data['sig']
        for i in range(len(network_GT)):
            mat[i, i] = 1
        MVGC = cv.adjs2graph(mat, np.zeros((len(network_GT), len(network_GT))))
        res_graph = MVGC
        gt.plotg(MVGC, output='./figs/Gopt_GC_N' + str(nn) + '_file_' + str(fl) + '.pdf')

        normal_GT = mf.precision_recall(res_graph, network_GT)
        Precision_O.append(normal_GT['orientation']['precision'])
        Recall_O.append(normal_GT['orientation']['recall'])
        F1_O.append(normal_GT['orientation']['F1'])

        Precision_A.append(normal_GT['adjacency']['precision'])
        Recall_A.append(normal_GT['adjacency']['recall'])
        F1_A.append(normal_GT['adjacency']['F1'])

        Precision_C.append(normal_GT['cycle']['precision'])
        Recall_C.append(normal_GT['cycle']['recall'])
        F1_C.append(normal_GT['cycle']['F1'])

        ###trying undersampled GT by 2

        new_GT = bfutils.all_undersamples(network_GT)[1]
        new_GT = mf.remove_bidir_edges(new_GT)
        undersampled_GT = mf.precision_recall(res_graph, new_GT)
        Precision_O2.append(undersampled_GT['orientation']['precision'])
        Recall_O2.append(undersampled_GT['orientation']['recall'])
        F1_O2.append(undersampled_GT['orientation']['F1'])

        Precision_A2.append(undersampled_GT['adjacency']['precision'])
        Recall_A2.append(undersampled_GT['adjacency']['recall'])
        F1_A2.append(undersampled_GT['adjacency']['F1'])

        Precision_C2.append(undersampled_GT['cycle']['precision'])
        Recall_C2.append(undersampled_GT['cycle']['recall'])
        F1_C2.append(undersampled_GT['cycle']['F1'])

        # ###trying sRASL
        r_estimated = drasl([res_graph], weighted=True, capsize=0,
                            urate=min(5, (3 * len(res_graph) + 1)),
                            scc=False,
                            GT_density=int(1000 * gk.density(network_GT)),
                            edge_weights=[2,0,3,0,1], pnum=PNUM, optim='optN')
        
        # sorted_data = sorted(r_estimated, key=lambda x: x[1], reverse=True)
        max_f1_score = 0
        for answer in r_estimated:
            res_rasl = bfutils.num2CG(answer[0][0], len(network_GT))
            rasl_sol = mf.precision_recall(res_rasl, network_GT)

            curr_f1 = (rasl_sol['orientation']['F1']) + (rasl_sol['adjacency']['F1']) + (rasl_sol['cycle']['F1'])


            if curr_f1 > max_f1_score:
                max_f1_score = curr_f1
                max_answer = answer

        res_rasl = bfutils.num2CG(max_answer[0][0], len(network_GT))
        rasl_sol = mf.precision_recall(res_rasl, network_GT)
        Precision_O3.append(rasl_sol['orientation']['precision'])
        Recall_O3.append(rasl_sol['orientation']['recall'])
        F1_O3.append(rasl_sol['orientation']['F1'])

        Precision_A3.append(rasl_sol['adjacency']['precision'])
        Recall_A3.append(rasl_sol['adjacency']['recall'])
        F1_A3.append(rasl_sol['adjacency']['F1'])

        Precision_C3.append(rasl_sol['cycle']['precision'])
        Recall_C3.append(rasl_sol['cycle']['recall'])
        F1_C3.append(rasl_sol['cycle']['F1'])



    now = str(datetime.now())
    now = now[:-7].replace(' ', '_')

    ###saving files
    filename = PreFix + '_priorities_20301_NO_selfloop_net_' + str(nn) + '_amp_' + now + '_' + ('concat' if concat else 'individual')

    # Data for group 1
    data_group1 = [
        [Precision_O, Recall_O, F1_O],
        [Precision_A, Recall_A, F1_A],
        [Precision_C, Recall_C, F1_C]
    ]

    # Data for group 2
    data_group2 = [
        [Precision_O2, Recall_O2, F1_O2],
        [Precision_A2, Recall_A2, F1_A2],
        [Precision_C2, Recall_C2, F1_C2]
    ]

    data_group3 = [
        [Precision_O3, Recall_O3, F1_O3],
        [Precision_A3, Recall_A3, F1_A3],
        [Precision_C3, Recall_C3, F1_C3]
    ]

    # Labels and titles for subplots
    titles = ['Orientation', 'Adjacency', '2 cycles']
    colors = ['blue', 'orange', 'red']

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

    for i, (data1, data2, data3, title) in enumerate(zip(data_group1, data_group2, data_group3, titles)):
        ax1 = axes[i]
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax1.boxplot(data1, positions=np.array(range(len(data1))) * 2.0 - 0.5, patch_artist=True, showmeans=True,
                    boxprops=dict(facecolor=colors[0]), widths=0.5)
        ax2.boxplot(data2, positions=np.array(range(len(data2))) * 2.0, patch_artist=True, showmeans=True,
                    boxprops=dict(facecolor=colors[1]), widths=0.6)
        ax2.boxplot(data3, positions=np.array(range(len(data3))) * 2.0 + 0.5, patch_artist=True, showmeans=True,
                    boxprops=dict(facecolor=colors[2]), widths=0.5)


        ax1.set_xticks(range(0, len(data1) * 2, 2))
        ax1.set_xticklabels(['Precision', 'Recall', 'F1-score'])
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Value')
        ax1.set_title(f'({title})')
        ax1.grid(True)
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, 1)
        ax3.set_ylim(0, 1)
    # Add super title
    plt.suptitle('Networks ' + str(nn) + ' ' + ('concat' if concat else 'individual') + ' data')
    # Legend
    blue_patch = mpatches.Patch(color='blue', label='ORG. GT')
    orange_patch = mpatches.Patch(color='orange', label='GT^2')
    red_patch = mpatches.Patch(color='red', label='MVGC+sRASL')
    plt.legend(handles=[blue_patch, orange_patch, red_patch], loc='upper right')

    plt.tight_layout()

    # Save the figure
    plt.savefig(filename + '_grouped_boxplot.png')
    plt.close()
