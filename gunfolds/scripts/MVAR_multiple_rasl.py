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

PNUM = 4

PreFix = 'MVAR_hardcode_selfloop_add_bidir_rasl'
concat = False
POSTFIX = 'Ruben_data' + 'concat' if concat else 'individual'

save_results = []

Precision_O = []
Recall_O = []
Precision_O2 = []
Recall_O2 = []
Precision_O3 = []
Recall_O3 = []
Precision_O4 = []
Recall_O4 = []
Precision_O5 = []
Recall_O5 = []

Precision_A = []
Recall_A = []
Precision_A2 = []
Recall_A2 = []
Precision_A3 = []
Recall_A3 = []
Precision_A4 = []
Recall_A4 = []
Precision_A5 = []
Recall_A5 = []

Precision_C = []
Recall_C = []
Precision_C2 = []
Recall_C2 = []
Precision_C3 = []
Recall_C3 = []
Precision_C4 = []
Recall_C4 = []
Precision_C5 = []
Recall_C5 = []

F1_O = []
F1_A = []
F1_C = []

F1_O2 = []
F1_A2 = []
F1_C2 = []

F1_O3 = []
F1_A3 = []
F1_C3 = []

F1_O4 = []
F1_A4 = []
F1_C4 = []

F1_O5 = []
F1_A5 = []
F1_C5 = []


for nn in [1,2,3,4,5,6]:

    # for fl in range(1, 61):
        # num = str(fl) if fl > 9 else '0' + str(fl)
        # print('reading file:' + num)
        # if not concat:
        #     data = pd.read_csv(
        #         './DataSets_Feedbacks/1. Simple_Networks/Network' + str(
        #             nn) + '_amp/data_fslfilter/BOLDfslfilter_{0}.txt'.format(
        #             num), delimiter='\t')
        # else:
        #     data = pd.read_csv(
        #         './DataSets_Feedbacks/1. Simple_Networks/Network' + str(
        #             nn) + '_amp/data_fslfilter_concat/concat_BOLDfslfilter_{0}.txt'.format(
        #             num), delimiter='\t')
        #
        # network_GT = simp_nets(nn, True)
        #
        # dd = np.transpose(data.values)
        # folder = 'expo_to_mat/MVAR_expo_to_mat_n' + str(nn) + '_' + ('concat' if concat else 'individual')
        # if not os.path.exists(folder):
        #     os.makedirs(folder)
        # savemat(folder + '/expo_to_mat_' + str(fl) + '.mat', {'dd': dd})


    network_GT = simp_nets(nn, True)
    for fl in range(1, 61):
        print('processing file:' + str(fl))

        folder_read = 'expo_to_mat/MVAR_expo_to_py_n' + str(nn) + '_' + ('concat' if concat else 'individual') + '_new1'
        mat_data = loadmat(folder_read + '/mat_file_' + str(fl) + '.mat')
        mat = mat_data['sig']
        for i in range(len(network_GT)):
            mat[i, i] = 1
        B = np.zeros((len(network_GT), len(network_GT))).astype(int)
        MVGC = cv.adjs2graph(mat, np.zeros((len(network_GT), len(network_GT))))
        normal_GT = mf.precision_recall(MVGC, network_GT)
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
        undersampled_GT = mf.precision_recall(MVGC, new_GT)
        Precision_O2.append(undersampled_GT['orientation']['precision'])
        Recall_O2.append(undersampled_GT['orientation']['recall'])
        F1_O2.append(undersampled_GT['orientation']['F1'])

        Precision_A2.append(undersampled_GT['adjacency']['precision'])
        Recall_A2.append(undersampled_GT['adjacency']['recall'])
        F1_A2.append(undersampled_GT['adjacency']['F1'])

        Precision_C2.append(undersampled_GT['cycle']['precision'])
        Recall_C2.append(undersampled_GT['cycle']['recall'])
        F1_C2.append(undersampled_GT['cycle']['F1'])

        edge_weights = [1, 3, 1, 3, 2]

        # ###trying sRASL

'''        nx_MVGC = gk.graph2nx(MVGC)
        two_cycle = mf.find_two_cycles(nx_MVGC)
        DD = np.ones((len(network_GT), len(network_GT))) * 5000
        BD = np.ones((len(network_GT), len(network_GT))) * 10000
        for cycle in two_cycle:
            DD[cycle[0]-1][cycle[1]-1] = 2500
            DD[cycle[1]-1][cycle[0]-1] = 2500
            B[cycle[0] - 1][cycle[1] - 1] = 1
            B[cycle[1] - 1][cycle[0] - 1] = 1

        for i in range(len(network_GT)):
            DD[i][i] = 10000
        MVGC_bi = cv.adjs2graph(mat, B)
        # gt.plotg(MVGC_bi, output='./figs/cycle_removed/Gopt_GC_' + str(nn*1000+fl) + '.pdf')
        
        r_estimated = drasl([MVGC_bi], weighted=True, capsize=0,
                            urate=min(5, (3 * len(MVGC_bi) + 1)),
                            scc=False,
                            dm=[DD],
                            bdm=[BD],
                            GT_density=int(1000 * gk.density(network_GT)),
                            edge_weights=edge_weights, pnum=PNUM, optim='optN')

        max_f1_score = 0
        for answer in r_estimated:
            res_rasl = bfutils.num2CG(answer[0][0], len(network_GT))
            rasl_sol = mf.precision_recall(res_rasl, network_GT)

            curr_f1 = ((rasl_sol['orientation']['F1']))
            # curr_f1 = (rasl_sol['orientation']['F1']) + (rasl_sol['adjacency']['F1']) + (rasl_sol['cycle']['F1'])

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

        sorted_data = sorted(r_estimated, key=lambda x: x[1], reverse=True)
        sorted_data = sorted_data[int(3*len(sorted_data)/4):-1]

        curr_po= 0
        curr_ro= 0
        curr_fo= 0
        curr_pa= 0
        curr_ra= 0
        curr_fa= 0
        curr_pc= 0
        curr_rc= 0
        curr_fc= 0
        ### mean cost answers
        for ans in sorted_data:
            mean_err = bfutils.num2CG(ans[0][0], len(network_GT))
            mean_err_sol = mf.precision_recall(mean_err, network_GT)
            curr_po += mean_err_sol['orientation']['precision']
            curr_ro += mean_err_sol['orientation']['recall']
            curr_fo += mean_err_sol['orientation']['F1']

            curr_pa += mean_err_sol['adjacency']['precision']
            curr_ra += mean_err_sol['adjacency']['recall']
            curr_fa += mean_err_sol['adjacency']['F1']

            curr_pc += mean_err_sol['cycle']['precision']
            curr_rc += mean_err_sol['cycle']['recall']
            curr_fc += mean_err_sol['cycle']['F1']

        Precision_O4.append(curr_po/len(sorted_data))
        Recall_O4.append(curr_ro/len(sorted_data))
        F1_O4.append(curr_fo/len(sorted_data))

        Precision_A4.append(curr_pa/len(sorted_data))
        Recall_A4.append(curr_ra/len(sorted_data))
        F1_A4.append(curr_fa/len(sorted_data))

        Precision_C4.append(curr_pc/len(sorted_data))
        Recall_C4.append(curr_rc/len(sorted_data))
        F1_C4.append(curr_fc/len(sorted_data))

        ### least cost answer
        least_err = bfutils.num2CG(sorted_data[-1][0][0], len(network_GT))
        least_err_sol = mf.precision_recall(least_err, network_GT)
        Precision_O5.append(least_err_sol['orientation']['precision'])
        Recall_O5.append(least_err_sol['orientation']['recall'])
        F1_O5.append(least_err_sol['orientation']['F1'])

        Precision_A5.append(least_err_sol['adjacency']['precision'])
        Recall_A5.append(least_err_sol['adjacency']['recall'])
        F1_A5.append(least_err_sol['adjacency']['F1'])

        Precision_C5.append(least_err_sol['cycle']['precision'])
        Recall_C5.append(least_err_sol['cycle']['recall'])
        F1_C5.append(least_err_sol['cycle']['F1'])'''


now = str(datetime.now())
now = now[:-7].replace(' ', '_')

###saving files
filename = PreFix + '_prior_'+''.join(map(str, edge_weights))+'_with_selfloop_net_' + str('all') + '_amp_' + now + '_' + (
    'concat' if concat else 'individual')

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

# data_group3 = [
#     [Precision_O3, Recall_O3, F1_O3],
#     [Precision_A3, Recall_A3, F1_A3],
#     [Precision_C3, Recall_C3, F1_C3]
# ]
#
# data_group4 = [
#     [Precision_O4, Recall_O4, F1_O4],
#     [Precision_A4, Recall_A4, F1_A4],
#     [Precision_C4, Recall_C4, F1_C4]
# ]
#
# data_group5 = [
#     [Precision_O5, Recall_O5, F1_O5],
#     [Precision_A5, Recall_A5, F1_A5],
#     [Precision_C5, Recall_C5, F1_C5]
# ]

# Labels and titles for subplots
titles = ['Orientation', 'Adjacency', '2 cycles']
colors = ['blue', 'orange', 'red'
    # , 'yellow', 'green'
          ]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

for i, (data1, data2
        # , data3
        # , data4, data5
        , title) in enumerate(zip(data_group1, data_group2
                                                            # , data_group3,
                                                            #        data_group4, data_group5
                                                                        , titles)):
    ax1 = axes[i]
    # ax2 = ax1.twinx()
    # ax3 = ax1.twinx()
    # ax4 = ax1.twinx()
    # ax5 = ax1.twinx()
    ax1.boxplot(data1, positions=np.array(range(len(data1))) * 2.0 - 0.6, patch_artist=True, showmeans=True,
                boxprops=dict(facecolor=colors[0]), widths=0.3)
    ax1.boxplot(data2, positions=np.array(range(len(data2))) * 2.0 - 0.3, patch_artist=True, showmeans=True,
                boxprops=dict(facecolor=colors[1]), widths=0.3)
    # ax1.boxplot(data3, positions=np.array(range(len(data3))) * 2.0 , patch_artist=True, showmeans=True,
    #             boxprops=dict(facecolor=colors[2]), widths=0.3)
    # ax1.boxplot(data4, positions=np.array(range(len(data4))) * 2.0 + 0.3, patch_artist=True, showmeans=True,
    #             boxprops=dict(facecolor=colors[3]), widths=0.3)
    # ax1.boxplot(data5, positions=np.array(range(len(data5))) * 2.0 + 0.6, patch_artist=True, showmeans=True,
    #             boxprops=dict(facecolor=colors[4]), widths=0.3)

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
plt.suptitle('Networks ' + str('all') + ' ' + ('concat' if concat else 'individual') + ' data')
# Legend
blue_patch = mpatches.Patch(color='blue', label='ORG. GT')
orange_patch = mpatches.Patch(color='orange', label='GT^2')
red_patch = mpatches.Patch(color='red', label='MVGC+bi+sRASL')
yellow_patch = mpatches.Patch(color='yellow', label='mean error')
green_patch = mpatches.Patch(color='green', label='least cost sol')
plt.legend(handles=[blue_patch, orange_patch
    # , red_patch, yellow_patch, green_patch
                    ], loc='upper right')

plt.tight_layout()

# Save the figure
plt.savefig(filename + '_grouped_boxplot.png')
plt.close()
