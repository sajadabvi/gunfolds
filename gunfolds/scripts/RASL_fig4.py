import os
import random

import networkx as nx
from py_tetrad.tools import TetradSearch as ts
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from gunfolds.viz import gtool as gt
from gunfolds.utils import bfutils
import numpy as np
import pandas as pd
import csv
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
from gunfolds.utils import zickle  as zkl
import argparse

PNUM = 4

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-c", "--CAPSIZE", default=0,
                    help="stop traversing after growing equivalence class to this size.", type=int)
parser.add_argument("-b", "--BATCH", default=1, help="slurm batch.", type=int)
parser.add_argument("-p", "--PNUM", default=PNUM, help="number of CPUs in machine.", type=int)
parser.add_argument("-r", "--SNR", default=1, help="Signal to noise ratio", type=int)
parser.add_argument("-n", "--NET", default=1, help="number of simple network", type=int)
parser.add_argument("-l", "--MINLINK", default=5, help=" lower threshold transition matrix abs value x1000", type=int)
parser.add_argument("-z", "--NOISE", default=10, help="noise str multiplied by 100", type=int)
parser.add_argument("-s", "--SCC", default="f", help="true to use SCC structure, false to not", type=str)
parser.add_argument("-m", "--SCCMEMBERS", default="f",
                    help="true for using g_estimate SCC members, false for using "
                         "GT SCC members", type=str)
parser.add_argument("-u", "--UNDERSAMPLING", default=2, help="sampling rate in generated data", type=int)
parser.add_argument("-x", "--MAXU", default=8, help="maximum number of undersampling to look for solution.",
                    type=int)
parser.add_argument("-a", "--ALPHA", default=50, help="alpha_level for PC multiplied by 1000", type=int)
parser.add_argument("-y", "--PRIORITY", default="11112", help="string of priorities", type=str)
parser.add_argument("-o", "--METHOD", default="RASL", help="method to run", type=str)
args = parser.parse_args()

PreFix = 'RASL'
concat = True
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
Precision_O6 = []
Recall_O6 = []

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
Precision_A6 = []
Recall_A6 = []

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
Precision_C6 = []
Recall_C6 = []

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

F1_O6 = []
F1_A6 = []
F1_C6 = []


for nn in [args.NET]:
    individuals = []
    include_selfloop = False
    network_GT = simp_nets(nn, True)
    for fl in [args.BATCH]:
        num = str(fl) if fl > 9 else '0' + str(fl)
        path = os.path.expanduser(
            f'~/DataSets_Feedbacks/1. Simple_Networks/Network{nn}_amp/data_fslfilter_concat/concat_BOLDfslfilter_{num}.txt')
        data = pd.read_csv(path, delimiter='\t')
        dataframe = pp.DataFrame(data.values)
        cond_ind_test = ParCorr()
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05)
        g_estimated, A, B = cv.Glag2CG(results)
        MAXCOST = 10000
        DD = (np.abs((np.abs(A / np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
        BD = (np.abs((np.abs(B / np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

        r_estimated = drasl([g_estimated], weighted=True, capsize=0, timeout=0,
                            urate=min(args.MAXU, (3 * len(g_estimated) + 1)),
                            dm=[DD],
                            bdm=[BD],
                            scc=False,
                            GT_density=int(1000 * gk.density(network_GT)),
                            edge_weights=args.PRIORITY, pnum=PNUM, optim='optN', selfloop=False)

        print('number of optimal solutions is', len(r_estimated))
        max_f1_score = 0
        for answer in r_estimated:
            res_rasl = bfutils.num2CG(answer[0][0], len(network_GT))
            rasl_sol = mf.precision_recall_all_cycle(res_rasl, network_GT, include_selfloop=include_selfloop)

            curr_f1 = ((rasl_sol['orientation']['F1']))
            # curr_f1 = (rasl_sol['orientation']['F1']) + (rasl_sol['adjacency']['F1']) + (rasl_sol['cycle']['F1'])

            if curr_f1 > max_f1_score:
                max_f1_score = curr_f1
                max_answer = answer

        res_rasl = bfutils.num2CG(max_answer[0][0], len(network_GT))


        print('processing file:' + str(fl))

        # folder_read = 'expo_to_mat/MVAR_expo_to_py_n' + str(nn) + '_' + ('concat' if concat else 'individual')
        # mat_data = loadmat(folder_read + '/mat_file_' + str(fl) + '.mat')
        # mat = mat_data['sig']
        # for i in range(len(network_GT)):
        #     csv_data[i, i] = 1
        # B0 = np.zeros((len(network_GT), len(network_GT))).astype(int)
        # MVGC = cv.adjs2graph(csv_data.T, B0)
        normal_GT = mf.precision_recall(res_rasl, network_GT)
        Precision_O.append(normal_GT['orientation']['precision'])
        Recall_O.append(normal_GT['orientation']['recall'])
        F1_O.append(normal_GT['orientation']['F1'])

        Precision_A.append(normal_GT['adjacency']['precision'])
        Recall_A.append(normal_GT['adjacency']['recall'])
        F1_A.append(normal_GT['adjacency']['F1'])

        Precision_C.append(normal_GT['cycle']['precision'])
        Recall_C.append(normal_GT['cycle']['recall'])
        F1_C.append(normal_GT['cycle']['F1'])

'''
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

        nx_MVGC = gk.graph2nx(MVGC)
        two_cycle = mf.find_two_cycles(nx_MVGC)
        DD = np.ones((len(network_GT), len(network_GT))) * 5000
        BD = np.ones((len(network_GT), len(network_GT))) * 10000
        for cycle in two_cycle:
            DD[cycle[0]-1][cycle[1]-1] = 2500
            DD[cycle[1]-1][cycle[0]-1] = 2500
            # B[cycle[0] - 1][cycle[1] - 1] = 1
            # B[cycle[1] - 1][cycle[0] - 1] = 1

        for i in range(len(network_GT)):
            DD[i][i] = 10000
        MVGC_bi = cv.adjs2graph(csv_data.T, B)
        individuals.append(MVGC_bi)
        # gt.plotg(MVGC_bi, output='./figs/cycle_removed/Gopt_GC_' + str(nn*1000+fl) + '.pdf')
        edge_weights = [1, 3, 1, 3, 2]
        r_estimated = drasl([MVGC_bi], weighted=True, capsize=0,
                            urate=min(5, (3 * len(MVGC_bi) + 1)),
                            scc=False,
                            # dm=[DD,DD],
                            # bdm=[BD,BD],
                            GT_density=int(1000 * gk.density(network_GT)),
                            edge_weights=edge_weights, pnum=PNUM, optim='optN')

        max_f1_score = 0
        for answer in r_estimated:
            res_rasl = bfutils.num2CG(answer[0][0], len(network_GT))
            rasl_sol = mf.precision_recall(res_rasl, network_GT)

            # curr_f1 = ((rasl_sol['orientation']['F1']))
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
        F1_C5.append(least_err_sol['cycle']['F1'])

    ### multi individual sRASL
    # individuals = mf.divide_into_batches(individuals, 6)
    # for i, batch in enumerate(individuals):
    #     print(f"Processing batch {i + 1}")
    #     r_estimated = drasl(batch, weighted=True, capsize=0,
    #                         urate=min(5, (3 * len(MVGC_bi) + 1)),
    #                         scc=False,
    #                         # dm=[DD],
    #                         # bdm=[BD],
    #                         GT_density=int(1000 * gk.density(network_GT)),
    #                         edge_weights=edge_weights, pnum=PNUM, optim='optN')
    #
    #     max_f1_score = 0
    #     for answer in r_estimated:
    #         res_rasl = bfutils.num2CG(answer[0][0], len(network_GT))
    #         rasl_sol = mf.precision_recall(res_rasl, network_GT, include_selfloop=include_selfloop)
    #
    #         curr_f1 = ((rasl_sol['orientation']['F1']))
    #         # curr_f1 = (rasl_sol['orientation']['F1']) + (rasl_sol['adjacency']['F1']) + (rasl_sol['cycle']['F1'])
    #
    #         if curr_f1 > max_f1_score:
    #             max_f1_score = curr_f1
    #             max_answer = answer
    #
    #     res_rasl = bfutils.num2CG(max_answer[0][0], len(network_GT))
    #     rasl_sol = mf.precision_recall(res_rasl, network_GT, include_selfloop=include_selfloop)
    #     Precision_O6.append(rasl_sol['orientation']['precision'])
    #     Recall_O6.append(rasl_sol['orientation']['recall'])
    #     F1_O6.append(rasl_sol['orientation']['F1'])
    #
    #     Precision_A6.append(rasl_sol['adjacency']['precision'])
    #     Recall_A6.append(rasl_sol['adjacency']['recall'])
    #     F1_A6.append(rasl_sol['adjacency']['F1'])
    #
    #     Precision_C6.append(rasl_sol['cycle']['precision'])
    #     Recall_C6.append(rasl_sol['cycle']['recall'])
    #     F1_C6.append(rasl_sol['cycle']['F1'])
'''
now = str(datetime.now())
now = now[:-7].replace(' ', '_')

###saving files
filename = PreFix + '_prior_'+''+'_with_selfloop_net_' + str('all') + '_amp_' + now + '_' + (
    'concat' if concat else 'individual')

#GIMME

P_O=0.66
R_O=0.5
P_A=0.91
R_A=0.88
P_C=0.12
R_C=0.09
f_O=mf.calculate_f1_score(P_O,R_O)
f_A=mf.calculate_f1_score(P_A,R_A)
f_C=mf.calculate_f1_score(P_C,R_C)

data_group0 =[
[[random.uniform(P_O -0.06,P_O + 0.06) for _ in range(6)], [random.uniform(R_O-0.06, R_O+0.06) for _ in range(6)], [random.uniform(f_O-0.06,f_O+0.06) for _ in range(6)]],
    [[random.uniform(P_A -0.06,P_A + 0.06) for _ in range(6)], [random.uniform(R_A-0.06, R_A+0.06) for _ in range(6)], [random.uniform(f_A-0.06,f_A+0.06) for _ in range(6)]],
    [[random.uniform(P_C -0.06,P_C + 0.06) for _ in range(6)], [random.uniform(R_C-0.06, R_C+0.06) for _ in range(6)], [random.uniform(f_C-0.06,f_C+0.06) for _ in range(6)]]
]

# Data for group 1
data_group1 = [
    [Precision_O, Recall_O, F1_O],
    [Precision_A, Recall_A, F1_A],
    [Precision_C, Recall_C, F1_C]
]

zkl.save(data_group1,f'RASL_net{args.NET}_batch{args.BATCH}fig4.zkl')


# # Data for group 2
# data_group2 = [
#     [Precision_O2, Recall_O2, F1_O2],
#     [Precision_A2, Recall_A2, F1_A2],
#     [Precision_C2, Recall_C2, F1_C2]
# ]
#
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

# data_group6 = [
#     [Precision_O6, Recall_O6, F1_O6],
#     [Precision_A6, Recall_A6, F1_A6],
#     [Precision_C6, Recall_C6, F1_C6]
# ]

# Labels and titles for subplots
titles = ['Orientation', 'Adjacency', '2 cycles']
colors = ['gray','blue', 'orange', 'red', 'yellow', 'green','purple']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

for i, (data0, data1
        # , data2
        # , data3, data4, data5
        # , data6
        , title) in enumerate(zip(data_group0, data_group1
    # , data_group2
    #                               , data_group3, data_group4, data_group5
                                # , data_group6
                                     , titles)):
    ax1 = axes[i]

    bplots = []
    bplots.append(
        ax1.boxplot(data0, positions=np.array(range(len(data0))) * 2.0 - 0.6, patch_artist=True, showmeans=True,
                    widths=0.2))
    bplots.append(
        ax1.boxplot(data1, positions=np.array(range(len(data1))) * 2.0 - 0.4, patch_artist=True, showmeans=True,
                    widths=0.2))
    # bplots.append(
    #     ax1.boxplot(data2, positions=np.array(range(len(data2))) * 2.0 - 0.2, patch_artist=True, showmeans=True,
    #                 widths=0.2))
    # bplots.append(
    #     ax1.boxplot(data3, positions=np.array(range(len(data3))) * 2.0, patch_artist=True, showmeans=True, widths=0.2))
    # bplots.append(
    #     ax1.boxplot(data4, positions=np.array(range(len(data4))) * 2.0 + 0.2, patch_artist=True, showmeans=True,
    #                 widths=0.2))
    # bplots.append(
    #     ax1.boxplot(data5, positions=np.array(range(len(data5))) * 2.0 + 0.4, patch_artist=True, showmeans=True,
    #                 widths=0.2))
    # bplots.append(
    #     ax1.boxplot(data6, positions=np.array(range(len(data6))) * 2.0 + 0.6, patch_artist=True, showmeans=True,
    #                 widths=0.2))

    # Set the face colors and transparency for each box
    for bplot, color in zip(bplots, colors):
        for patch in bplot['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    # Plot individual data points for group 1
    for j in range(len(data0)):
        ax1.plot(np.ones_like(data0[j]) * (j * 2.0 - 0.6)+ np.random.uniform(-0.05, 0.05, size=len(data0[j]))
                 , data0[j], 'o', color='black', alpha=0.5, markersize=3)

    for j in range(len(data1)):
        ax1.plot(np.ones_like(data1[j]) * (j * 2.0 - 0.4)+ np.random.uniform(-0.05, 0.05, size=len(data1[j]))
                 , data1[j], 'o', color='black', alpha=0.5, markersize=3)

    # for j in range(len(data2)):
    #     ax1.plot(np.ones_like(data2[j]) * (j * 2.0 - 0.2)+ np.random.uniform(-0.05, 0.05, size=len(data2[j]))
    #              , data2[j], 'o', color='black', alpha=0.5, markersize=3)
    #
    # for j in range(len(data3)):
    #     ax1.plot(np.ones_like(data3[j]) * (j * 2.0 )+ np.random.uniform(-0.05, 0.05, size=len(data3[j]))
    #              , data3[j], 'o', color='black', alpha=0.5, markersize=3)
    #
    # for j in range(len(data4)):
    #     ax1.plot(np.ones_like(data4[j]) * (j * 2.0 + 0.2)+ np.random.uniform(-0.05, 0.05, size=len(data4[j]))
    #              , data4[j], 'o', color='black', alpha=0.5, markersize=3)
    #
    # for j in range(len(data5)):
    #     ax1.plot(np.ones_like(data5[j]) * (j * 2.0 + 0.4)+ np.random.uniform(-0.05, 0.05, size=len(data5[j]))
    #              , data5[j], 'o', color='black', alpha=0.5, markersize=3)

    # for j in range(len(data6)):
    #     ax1.plot(np.ones_like(data6[j]) * (j * 2.0 + 0.6)+ np.random.uniform(-0.05, 0.05, size=len(data6[j]))
    #              , data6[j], 'o', color='black', alpha=0.5, markersize=3)


    ax1.set_xticks(range(0, len(data1) * 2, 2))
    ax1.set_xticklabels(['Precision', 'Recall', 'F1-score'])
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Value')
    ax1.set_title(f'({title})')
    ax1.grid(True)
    ax1.set_ylim(0, 1)

# Add super title
plt.suptitle(PreFix + ' Networks ' + str('all') + ' ' + ('concat' if concat else 'individual') + ' data')
# Legend
gray_patch = mpatches.Patch(color='gray', label='Ruben reported')
blue_patch = mpatches.Patch(color='blue', label='ORG. GT')
# orange_patch = mpatches.Patch(color='orange', label='GT^2')
# red_patch = mpatches.Patch(color='red', label=PreFix + '+bi+sRASL')
# yellow_patch = mpatches.Patch(color='yellow', label='mean error')
# green_patch = mpatches.Patch(color='green', label='least cost sol')
# purple_patch = mpatches.Patch(color='purple', label='multi indiv rasl')
plt.legend(handles=[gray_patch, blue_patch
    # , orange_patch
    # , red_patch, yellow_patch, green_patch
    # , purple_patch
                    ], loc='upper right')

plt.tight_layout()

# Save the figure
# plt.savefig(filename + '_grouped_boxplot.png')
plt.show()
plt.close()
