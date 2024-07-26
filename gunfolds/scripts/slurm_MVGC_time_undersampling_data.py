import os
from gunfolds.utils import bfutils
import numpy as np
import pandas as pd
from gunfolds import conversions as cv
from datetime import datetime
from scipy.io import loadmat
from gunfolds.scripts.datasets.simple_networks import simp_nets
from gunfolds.scripts import my_functions as mf
from gunfolds.solvers.clingo_rasl import drasl
from gunfolds.utils import graphkit as gk
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import argparse
import distutils.util
from gunfolds.utils import zickle as zkl
import glob

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-b", "--BATCH", default=1, help="slurm batch.", type=int)
parser.add_argument("-p", "--PNUM", default=4, help="number of CPUs in machine.", type=int)
parser.add_argument("-c", "--CONCAT", default="t", help="true to use concat data", type=str)
parser.add_argument("-u", "--UNDERSAMPLED", default="t", help="true to use tr 3 time scale", type=str)
parser.add_argument("-m", "--MANUAL", default="t", help="true to manually undersample 3 time scale", type=str)
args = parser.parse_args()
PNUM = args.PNUM
UNDERSAMPLED = bool(distutils.util.strtobool(args.UNDERSAMPLED))
MANUAL = bool(distutils.util.strtobool(args.MANUAL))
TR = '3s' if UNDERSAMPLED else '1.20s'
PreFix = 'MVGC' + TR
concat = bool(distutils.util.strtobool(args.CONCAT))
POSTFIX = 'temporal_undersampling_data' + 'concat' if concat else 'individual'

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

for nn in [4]:

    network_GT = simp_nets(nn, selfloop=True)
    include_selfloop = True
    for fl in [args.BATCH]:
        print('processing file:' + str(fl))

        folder_read = 'expo_to_mat/' + ('manual/' if MANUAL else '') + 'MVGC_TR' + TR + '_expo_to_py' + '_' + (
            'concat' if concat else 'individual')  # + '_new1'
        mat_data = loadmat(folder_read + '/mat_file_' + str(fl) + '.mat')
        mat = mat_data['sig']
        for i in range(len(network_GT)):
            mat[i, i] = 1
        B = np.zeros((len(network_GT), len(network_GT))).astype(int)
        MVGC = cv.adjs2graph(mat, np.zeros((len(network_GT), len(network_GT))))
        normal_GT = mf.precision_recall(MVGC, network_GT, include_selfloop=include_selfloop)
        Precision_O.append(normal_GT['orientation']['precision'])
        Recall_O.append(normal_GT['orientation']['recall'])
        F1_O.append(normal_GT['orientation']['F1'])

        Precision_A.append(normal_GT['adjacency']['precision'])
        Recall_A.append(normal_GT['adjacency']['recall'])
        F1_A.append(normal_GT['adjacency']['F1'])

        Precision_C.append(normal_GT['cycle']['precision'])
        Recall_C.append(normal_GT['cycle']['recall'])
        F1_C.append(normal_GT['cycle']['F1'])

        # trying undersampled GT by 2

        new_GT = bfutils.all_undersamples(network_GT)[1]
        new_GT = mf.remove_bidir_edges(new_GT)
        undersampled_GT = mf.precision_recall(MVGC, new_GT, include_selfloop=include_selfloop)
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

        nx_MVGC = gk.graph2nx(MVGC)
        two_cycle = mf.find_two_cycles(nx_MVGC)
        DD = np.ones((len(network_GT), len(network_GT))) * 5000
        BD = np.ones((len(network_GT), len(network_GT))) * 10000
        for cycle in two_cycle:
            DD[cycle[0] - 1][cycle[1] - 1] = 2500
            DD[cycle[1] - 1][cycle[0] - 1] = 2500
            B[cycle[0] - 1][cycle[1] - 1] = 1
            B[cycle[1] - 1][cycle[0] - 1] = 1

        for i in range(len(network_GT)):
            DD[i][i] = 10000
        MVGC_bi = cv.adjs2graph(mat, B)
        edge_weights = [1, 3, 1, 3, 2]
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
            rasl_sol = mf.precision_recall(res_rasl, network_GT, include_selfloop=include_selfloop)

            # curr_f1 = ((rasl_sol['orientation']['F1']))
            curr_f1 = (rasl_sol['orientation']['F1']) + (rasl_sol['adjacency']['F1']) + (rasl_sol['cycle']['F1'])

            if curr_f1 > max_f1_score:
                max_f1_score = curr_f1
                max_answer = answer

        res_rasl = bfutils.num2CG(max_answer[0][0], len(network_GT))
        rasl_sol = mf.precision_recall(res_rasl, network_GT, include_selfloop=include_selfloop)
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

        curr_po = 0
        curr_ro = 0
        curr_fo = 0
        curr_pa = 0
        curr_ra = 0
        curr_fa = 0
        curr_pc = 0
        curr_rc = 0
        curr_fc = 0
        # mean cost answers
        for ans in sorted_data:
            mean_err = bfutils.num2CG(ans[0][0], len(network_GT))
            mean_err_sol = mf.precision_recall(mean_err, network_GT, include_selfloop=include_selfloop)
            curr_po += mean_err_sol['orientation']['precision']
            curr_ro += mean_err_sol['orientation']['recall']
            curr_fo += mean_err_sol['orientation']['F1']

            curr_pa += mean_err_sol['adjacency']['precision']
            curr_ra += mean_err_sol['adjacency']['recall']
            curr_fa += mean_err_sol['adjacency']['F1']

            curr_pc += mean_err_sol['cycle']['precision']
            curr_rc += mean_err_sol['cycle']['recall']
            curr_fc += mean_err_sol['cycle']['F1']

        Precision_O4.append(curr_po / len(sorted_data))
        Recall_O4.append(curr_ro / len(sorted_data))
        F1_O4.append(curr_fo / len(sorted_data))

        Precision_A4.append(curr_pa / len(sorted_data))
        Recall_A4.append(curr_ra / len(sorted_data))
        F1_A4.append(curr_fa / len(sorted_data))

        Precision_C4.append(curr_pc / len(sorted_data))
        Recall_C4.append(curr_rc / len(sorted_data))
        F1_C4.append(curr_fc / len(sorted_data))

        # least cost answer
        least_err = bfutils.num2CG(sorted_data[-1][0][0], len(network_GT))
        least_err_sol = mf.precision_recall(least_err, network_GT, include_selfloop=include_selfloop)
        Precision_O5.append(least_err_sol['orientation']['precision'])
        Recall_O5.append(least_err_sol['orientation']['recall'])
        F1_O5.append(least_err_sol['orientation']['F1'])

        Precision_A5.append(least_err_sol['adjacency']['precision'])
        Recall_A5.append(least_err_sol['adjacency']['recall'])
        F1_A5.append(least_err_sol['adjacency']['F1'])

        Precision_C5.append(least_err_sol['cycle']['precision'])
        Recall_C5.append(least_err_sol['cycle']['recall'])
        F1_C5.append(least_err_sol['cycle']['F1'])

        # PCMCI + sRASL
        pattern = f'./data_group/*data_group7_data_batch{args.BATCH}_{TR}_{concat}_MANUAL_{MANUAL}.zkl'

        if glob.glob(pattern):
            data_group7 = zkl.load(glob.glob(pattern)[0])
            print('using previous results:')
            print(glob.glob(pattern)[0])
        else:
            num = str(fl) if fl > 9 else '0' + str(fl)
            if not concat:
                data = pd.read_csv(
                    './DataSets_Feedbacks/4. Temporal_Undersampling_Data/' + (
                        'manual/' if MANUAL else '') + 'data_' + TR + 'TR_individual/BOLD' +
                    ('fslfilter' if TR == '1.20s' else '3TRfilt') + '_{0}.txt'.format(
                        num), delimiter='\t')
            else:
                data = pd.read_csv(
                    './DataSets_Feedbacks/4. Temporal_Undersampling_Data/' + (
                        'manual/' if MANUAL else '') + 'data_' + TR + 'TR_concatenated/concat_BOLD' +
                    ('fslfilter' if TR == '1.20s' else '3TRfilt') + '_{0}.txt'.format(
                        num), delimiter='\t')

            dataframe = pp.DataFrame(data.values)
            cond_ind_test = ParCorr()
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
            results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05)

            g_estimated, A, B = cv.Glag2CG(results)
            MAXCOST = 10000
            priorities = [4, 2, 5, 3, 1]
            DD = (np.abs((np.abs(A / np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
            BD = (np.abs((np.abs(B / np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

            r_estimated = drasl([g_estimated], weighted=True, capsize=0, timeout=0,
                                urate=min(5, (3 * len(g_estimated) + 1)),
                                dm=[DD],
                                bdm=[BD],
                                scc=False,
                                GT_density=int(1000 * gk.density(network_GT)),
                                edge_weights=priorities, pnum=PNUM, optim='optN')

            print('number of optimal solutions is', len(r_estimated))
            max_f1_score = 0
            for answer in r_estimated:
                res_rasl = bfutils.num2CG(answer[0][0], len(network_GT))
                rasl_sol = mf.precision_recall(res_rasl, network_GT, include_selfloop=include_selfloop)

                # curr_f1 = ((rasl_sol['orientation']['F1']))
                curr_f1 = (rasl_sol['orientation']['F1']) + (rasl_sol['adjacency']['F1']) + (rasl_sol['cycle']['F1'])

                if curr_f1 > max_f1_score:
                    max_f1_score = curr_f1
                    max_answer = answer

            res_rasl = bfutils.num2CG(max_answer[0][0], len(network_GT))
            rasl_sol = mf.precision_recall(res_rasl, network_GT, include_selfloop=include_selfloop)
            Precision_O7.append(rasl_sol['orientation']['precision'])
            Recall_O7.append(rasl_sol['orientation']['recall'])
            F1_O7.append(rasl_sol['orientation']['F1'])

            Precision_A7.append(rasl_sol['adjacency']['precision'])
            Recall_A7.append(rasl_sol['adjacency']['recall'])
            F1_A7.append(rasl_sol['adjacency']['F1'])

            Precision_C7.append(rasl_sol['cycle']['precision'])
            Recall_C7.append(rasl_sol['cycle']['recall'])
            F1_C7.append(rasl_sol['cycle']['F1'])

            data_group7 = [
                [Precision_O7, Recall_O7, F1_O7],
                [Precision_A7, Recall_A7, F1_A7],
                [Precision_C7, Recall_C7, F1_C7]
            ]

now = str(datetime.now())
now = now[:-7].replace(' ', '_')

# saving files
filename = PreFix + '_prior_' + ''.join(map(str, edge_weights)) + '_with_selfloop_net_' + str(
    'all') + '_amp_' + now + '_' + (
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

if not os.path.exists('data_group'):
    os.makedirs('data_group')
zkl.save(data_group1, 'data_group/' + filename + 'data_group1_data_batch' + str(args.BATCH) + TR + '_' + str(
    concat) + '_MANUAL_' + str(MANUAL) + '.zkl')
zkl.save(data_group2, 'data_group/' + filename + 'data_group2_data_batch' + str(args.BATCH) + TR + '_' + str(
    concat) + '_MANUAL_' + str(MANUAL) + '.zkl')
zkl.save(data_group3, 'data_group/' + filename + 'data_group3_data_batch' + str(args.BATCH) + TR + '_' + str(
    concat) + '_MANUAL_' + str(MANUAL) + '.zkl')
zkl.save(data_group4, 'data_group/' + filename + 'data_group4_data_batch' + str(args.BATCH) + TR + '_' + str(
    concat) + '_MANUAL_' + str(MANUAL) + '.zkl')
zkl.save(data_group5, 'data_group/' + filename + 'data_group5_data_batch' + str(args.BATCH) + TR + '_' + str(
    concat) + '_MANUAL_' + str(MANUAL) + '.zkl')

zkl.save(data_group7, 'data_group/' + filename + 'data_group7_data_batch' + str(args.BATCH) + TR + '_' + str(
    concat) + '_MANUAL_' + str(MANUAL) + '.zkl')
