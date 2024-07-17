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
import random
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import argparse
import distutils.util
from gunfolds.utils import zickle  as zkl
import csv

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-p", "--PNUM", default=4, help="number of CPUs in machine.", type=int)
parser.add_argument("-c", "--CONCAT", default="f", help="true to use concat data", type=str)
parser.add_argument("-u", "--UNDERSAMPLED", default="t", help="true to use tr 3 time scale", type=str)
args = parser.parse_args()
PNUM = args.PNUM
UNDERSAMPLED = bool(distutils.util.strtobool(args.UNDERSAMPLED))
TR = '3s' if UNDERSAMPLED else '1.20s'
PreFix = 'GIMME' + TR
concat = bool(distutils.util.strtobool(args.CONCAT))
POSTFIX = 'tepmporal_undetsampling_data' + 'concat' if concat else 'individual'

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
Precision_O7 = []
Recall_O7 = []

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
Precision_A7 = []
Recall_A7 = []

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
Precision_C7 = []
Recall_C7 = []

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

F1_O7 = []
F1_A7 = []
F1_C7 = []

def read_csv_files(path,size):
    files = sorted(os.listdir(path))  # Sort the files
    for filename in files:
        if filename.endswith('.csv'):
            file_path = os.path.join(path, filename)
            with open(file_path, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip header if exists
                rows = []
                for row in csv_reader:
                    rows.append(row[0:2*size])
                mat = np.array(rows, dtype=np.float32)
                matrix1 = np.array(mat[:, 0:size])
                matrix2 = np.array(mat[:, size:2*size])
                sum_matrix = matrix1# + matrix2
                binary_matrixA = (matrix1 != 0).astype(int)
                binary_matrixB = (matrix2 != 0).astype(int)
    return binary_matrixA, binary_matrixB


for nn in [4]:

    for fl in range(1, 61):
        num = str(fl) if fl > 9 else '0' + str(fl)
        print('reading file:' + num)
        if not concat:
            data = pd.read_csv(
                './DataSets_Feedbacks/4. Temporal_Undersampling_Data/data_'+TR+'TR_individual/BOLD' +
                ('fslfilter' if TR == '1.20s' else '3TRfilt') + '_{0}.txt'.format(
                    num), delimiter='\t')
        else:
            data = pd.read_csv(
                './DataSets_Feedbacks/4. Temporal_Undersampling_Data/data_'+TR+'TR_concatenated/concat_BOLD' +
                ('fslfilter' if TR == '1.20s' else '3TRfilt') + '_{0}.txt'.format(
                    num), delimiter='\t')

        dd = np.transpose(data.values)
        folder = 'expo_to_mat/expo_to_mat_' + ('concat' if concat else 'individual') + '_' + TR
        if not os.path.exists(folder):
            os.makedirs(folder)
        savemat(folder + '/expo_to_mat_' + str(fl) + '.mat', {'dd': dd})

    network_GT = simp_nets(nn, selfloop=True)
    include_selfloop = True
    individuals = []
    for fl in range(1, 61):
        print('processing file:' + str(fl))

        path = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/expo_to_mat/'\
               + 'GIMME_TR' + TR + '_expo_to_py' + '_' + ('concat' if concat else 'individual')+'/sum'
        csv_data, B = read_csv_files(path, len(network_GT))

        B0 = np.zeros((len(network_GT), len(network_GT))).astype(int)
        GIMME = cv.adjs2graph(csv_data.T, B0)
        normal_GT = mf.precision_recall(GIMME, network_GT, include_selfloop = include_selfloop)
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
        undersampled_GT = mf.precision_recall(GIMME, new_GT,include_selfloop = include_selfloop)
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
        edge_weights = [1, 3, 1, 3, 2]

        nx_GIMME = gk.graph2nx(GIMME)
        two_cycle = mf.find_two_cycles(nx_GIMME)
        DD = np.ones((len(network_GT), len(network_GT))) * 5000
        BD = np.ones((len(network_GT), len(network_GT))) * 10000
        for cycle in two_cycle:
            DD[cycle[0]-1][cycle[1]-1] = 2500
            DD[cycle[1]-1][cycle[0]-1] = 2500
            # B[cycle[0] - 1][cycle[1] - 1] = 1
            # B[cycle[1] - 1][cycle[0] - 1] = 1

        for i in range(len(network_GT)):
            DD[i][i] = 10000
        GIMME_bi = cv.adjs2graph(csv_data.T, B)
        individuals.append(GIMME_bi)
        # gt.plotg(GIMME_bi, output='./figs/cycle_removed/Gopt_GC_' + str(nn*1000+fl) + '.pdf')
        edge_weights = [1, 3, 1, 3, 2]
        r_estimated = drasl([GIMME_bi], weighted=True, capsize=0,
                            urate=min(5, (3 * len(GIMME_bi) + 1)),
                            scc=False,
                            dm=[DD],
                            bdm=[BD],
                            GT_density=int(1000 * gk.density(network_GT)),
                            edge_weights=edge_weights, pnum=PNUM, optim='optN')

        max_f1_score = 0
        for answer in r_estimated:
            res_rasl = bfutils.num2CG(answer[0][0], len(network_GT))
            rasl_sol = mf.precision_recall(res_rasl, network_GT,include_selfloop = include_selfloop)

            # curr_f1 = ((rasl_sol['orientation']['F1']))
            curr_f1 = (rasl_sol['orientation']['F1']) + (rasl_sol['adjacency']['F1']) + (rasl_sol['cycle']['F1'])

            if curr_f1 > max_f1_score:
                max_f1_score = curr_f1
                max_answer = answer

        res_rasl = bfutils.num2CG(max_answer[0][0], len(network_GT))
        rasl_sol = mf.precision_recall(res_rasl, network_GT,include_selfloop = include_selfloop)
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
        # Determine the number of elements
        num_elements = len(sorted_data)

        if num_elements < 10:
            # Store the smallest half if the number of elements is less than 10
            smallest_half = sorted_data[int(num_elements / 2):]
            sorted_data = smallest_half
        else:
            # Store the smallest quarter otherwise
            smallest_quarter = sorted_data[int(3 * num_elements / 4):]
            sorted_data = smallest_quarter

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
            mean_err_sol = mf.precision_recall(mean_err, network_GT,include_selfloop = include_selfloop)
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
        least_err_sol = mf.precision_recall(least_err, network_GT,include_selfloop = include_selfloop)
        Precision_O5.append(least_err_sol['orientation']['precision'])
        Recall_O5.append(least_err_sol['orientation']['recall'])
        F1_O5.append(least_err_sol['orientation']['F1'])

        Precision_A5.append(least_err_sol['adjacency']['precision'])
        Recall_A5.append(least_err_sol['adjacency']['recall'])
        F1_A5.append(least_err_sol['adjacency']['F1'])

        Precision_C5.append(least_err_sol['cycle']['precision'])
        Recall_C5.append(least_err_sol['cycle']['recall'])
        F1_C5.append(least_err_sol['cycle']['F1'])

        ###PCMCI + sRASL
        # num = str(fl) if fl > 9 else '0' + str(fl)
        # if not concat:
        #     data = pd.read_csv(
        #         './DataSets_Feedbacks/4. Temporal_Undersampling_Data/data_' + TR + 'TR_individual/BOLD' +
        #         ('fslfilter' if TR == '1.20s' else '3TRfilt') + '_{0}.txt'.format(
        #             num), delimiter='\t')
        # else:
        #     data = pd.read_csv(
        #         './DataSets_Feedbacks/4. Temporal_Undersampling_Data/data_' + TR + 'TR_concatenated/concat_BOLD' +
        #         ('fslfilter' if TR == '1.20s' else '3TRfilt') + '_{0}.txt'.format(
        #             num), delimiter='\t')
        #
        # dataframe = pp.DataFrame(data.values)
        # cond_ind_test = ParCorr()
        # pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        # results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05)
        #
        # g_estimated, A, B = cv.Glag2CG(results)
        # MAXCOST = 10000
        # priprities = [4, 2, 5, 3, 1]
        # DD = (np.abs((np.abs(A / np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
        # BD = (np.abs((np.abs(B / np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)
        #
        # # g_estimated = gk.ringmore(5, 2)
        # # BD = np.ones((5, 5))
        # # DD = np.ones((5, 5))
        # r_estimated = drasl([g_estimated], weighted=True, capsize=0, timeout=0,
        #                     urate=min(5, (3 * len(g_estimated) + 1)),
        #                     dm=[DD],
        #                     bdm=[BD],
        #                     scc=False,
        #                     GT_density=int(1000 * gk.density(network_GT)),
        #                     edge_weights=priprities, pnum=PNUM, optim='optN')
        #
        # print('number of optimal solutions is', len(r_estimated))
        # max_f1_score = 0
        # for answer in r_estimated:
        #     res_rasl = bfutils.num2CG(answer[0][0], len(network_GT))
        #     rasl_sol = mf.precision_recall(res_rasl, network_GT, include_selfloop=include_selfloop)
        #
        #     # curr_f1 = ((rasl_sol['orientation']['F1']))
        #     curr_f1 = (rasl_sol['orientation']['F1']) + (rasl_sol['adjacency']['F1']) + (rasl_sol['cycle']['F1'])
        #
        #     if curr_f1 > max_f1_score:
        #         max_f1_score = curr_f1
        #         max_answer = answer
        #
        # res_rasl = bfutils.num2CG(max_answer[0][0], len(network_GT))
        # rasl_sol = mf.precision_recall(res_rasl, network_GT, include_selfloop=include_selfloop)
        # Precision_O7.append(rasl_sol['orientation']['precision'])
        # Recall_O7.append(rasl_sol['orientation']['recall'])
        # F1_O7.append(rasl_sol['orientation']['F1'])
        #
        # Precision_A7.append(rasl_sol['adjacency']['precision'])
        # Recall_A7.append(rasl_sol['adjacency']['recall'])
        # F1_A7.append(rasl_sol['adjacency']['F1'])
        #
        # Precision_C7.append(rasl_sol['cycle']['precision'])
        # Recall_C7.append(rasl_sol['cycle']['recall'])
        # F1_C7.append(rasl_sol['cycle']['F1'])

    ### multi individual sRASL
    # individuals = mf.divide_into_batches(individuals, 6)
    # for i, batch in enumerate(individuals):
    #     print(f"Processing batch {i + 1}")
    #     r_estimated = drasl(batch, weighted=True, capsize=0,
    #                         urate=min(5, (3 * len(GIMME_bi) + 1)),
    #                         scc=False,
    #                         # dm=[DD],
    #                         # bdm=[BD],
    #                         GT_density=int(1000 * gk.density(network_GT)),
    #                         edge_weights=edge_weights, pnum=PNUM, optim='optN')
    #
    #     max_f1_score = 0
    #     for answer in r_estimated:
    #         res_rasl = bfutils.num2CG(answer[0][0], len(network_GT))
    #         rasl_sol = mf.precision_recall(res_rasl, network_GT,include_selfloop = include_selfloop)
    #
    #         # curr_f1 = ((rasl_sol['orientation']['F1']))
    #         curr_f1 = (rasl_sol['orientation']['F1']) + (rasl_sol['adjacency']['F1']) + (rasl_sol['cycle']['F1'])
    #
    #         if curr_f1 > max_f1_score:
    #             max_f1_score = curr_f1
    #             max_answer = answer
    #
    #     res_rasl = bfutils.num2CG(max_answer[0][0], len(network_GT))
    #     rasl_sol = mf.precision_recall(res_rasl, network_GT,include_selfloop = include_selfloop)
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



now = str(datetime.now())
now = now[:-7].replace(' ', '_')

###saving files
filename = PreFix + '_prior_'+''.join(map(str, edge_weights))+'_with_selfloop_net_' + str('all') + '_amp_' + now + '_' + (
    'concat' if concat else 'individual')

# data_group0 =[
# [[random.uniform(0.40, 0.52) for _ in range(100)], [random.uniform(0.80, 0.92) for _ in range(100)], [random.uniform(0.54, 0.66) for _ in range(100)]],
#     [[random.uniform(0.68, 0.80) for _ in range(100)], [random.uniform(0.82, 0.94) for _ in range(100)], [random.uniform(0.74, 0.86) for _ in range(100)]],
#     [[random.uniform(0.08, 0.2) for _ in range(100)], [random.uniform(0.58, 0.7) for _ in range(100)], [random.uniform(0.18, 0.30) for _ in range(100)]]
# ]
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

data_group4 = [
    [Precision_O4, Recall_O4, F1_O4],
    [Precision_A4, Recall_A4, F1_A4],
    [Precision_C4, Recall_C4, F1_C4]
]

data_group5 = [
    [Precision_O5, Recall_O5, F1_O5],
    [Precision_A5, Recall_A5, F1_A5],
    [Precision_C5, Recall_C5, F1_C5]
]

# data_group6 = [
#     [Precision_O6, Recall_O6, F1_O6],
#     [Precision_A6, Recall_A6, F1_A6],
#     [Precision_C6, Recall_C6, F1_C6]
# ]

data_group7 = zkl.load('data_group7_data_' + TR + str(concat)+'.zkl')

# zkl.save(data_group7,'data_group7_data_' + TR + str(concat)+'.zkl')
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
plt.suptitle('GIMME, TR= ' + TR + ' ' + ('concat' if concat else 'individual') + ' data')
# Legend
gray_patch = mpatches.Patch(color='gray', label='Ruben reported')
blue_patch = mpatches.Patch(color='blue', label='ORG. GT')
orange_patch = mpatches.Patch(color='orange', label='GT^2')
red_patch = mpatches.Patch(color='red', label='GIMME+bi+sRASL')
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
