import os
# from gunfolds.viz import gtool as gt
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
from distutils.util import strtobool
from gunfolds.estimation import linear_model as lm
import glob
from gunfolds.viz import gtool as gt
from gunfolds.utils import zickle as zkl
import time
import sys
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import csv
from gunfolds.scripts import bold_function as hrf
sys.path.append('~/tread/py-tetrad')
from py_tetrad.tools import TetradSearch as ts

def parse_arguments(PNUM):
    parser = argparse.ArgumentParser(description='Run settings.')
    parser.add_argument("-c", "--CAPSIZE", default=0,
                        help="stop traversing after growing equivalence class to this size.", type=int)
    parser.add_argument("-b", "--BATCH", default=24, help="slurm batch.", type=int)
    parser.add_argument("-p", "--PNUM", default=PNUM, help="number of CPUs in machine.", type=int)
    parser.add_argument("-n", "--NET", default=1, help="number of simple network", type=int)
    parser.add_argument("-l", "--MINLINK", default=5, help=" lower threshold transition matrix abs value x1000", type=int)
    parser.add_argument("-z", "--NOISE", default=10, help="noise str multiplied by 100", type=int)
    parser.add_argument("-s", "--SCC", default="f", help="true to use SCC structure, false to not", type=str)
    parser.add_argument("-m", "--SCCMEMBERS", default="f",
                        help="true for using g_estimate SCC members, false for using "
                             "GT SCC members", type=str)
    parser.add_argument("-u", "--UNDERSAMPLING", default=2, help="sampling rate in generated data", type=int)
    parser.add_argument("-x", "--MAXU", default=6, help="maximum number of undersampling to look for solution.",
                        type=int)
    parser.add_argument("-a", "--ALPHA", default=50, help="alpha_level for PC multiplied by 1000", type=int)
    parser.add_argument("-y", "--PRIORITY", default="11112", help="string of priorities", type=str)
    parser.add_argument("-o", "--METHOD", default="RASL", help="method to run", type=str)
    return parser.parse_args()

def convert_str_to_bool(args):
    args.SCC = bool(strtobool(args.SCC))
    args.SCCMEMBERS = bool(strtobool(args.SCCMEMBERS))
    args.NOISE = args.NOISE / 100
    args.ALPHA = args.ALPHA / 1000
    priprities = []
    for char in args.PRIORITY:
        priprities.append(int(char))
    args.PRIORITY = priprities
    return args


# Define the functions
def MVGC(args, network_GT):
    path = (f'/Users/sajad/GSU Dropbox Dropbox/Mohammadsajad '
            f'Abavisani/Mac/Documents/PhD/Research/code/DataSets_Feedbacks/8_VAR_simulation/ri'
            f'ngmore/u{args.UNDERSAMPLING}/MVGC')
    mat_data = loadmat(path + f'/mat_file_{args.BATCH}.mat')['sig']
    for i in range(len(network_GT)):
        mat_data[i, i] = 0
    B = np.zeros((len(network_GT), len(network_GT))).astype(int)
    MVGC = cv.adjs2graph(mat_data, np.zeros((len(network_GT), len(network_GT))))
    return MVGC

def MVAR(args, network_GT):
    path = (f'/Users/sajad/GSU Dropbox Dropbox/Mohammadsajad '
            f'Abavisani/Mac/Documents/PhD/Research/code/DataSets_Feedbacks/8_VAR_simulation/ri'
            f'ngmore/u{args.UNDERSAMPLING}/MVAR')
    mat_data = loadmat(path + f'/mat_file_{args.BATCH}.mat')['sig']
    for i in range(len(network_GT)):
        mat_data[i, i] = 0
    B = np.zeros((len(network_GT), len(network_GT))).astype(int)
    MVAR = cv.adjs2graph(mat_data, np.zeros((len(network_GT), len(network_GT))))
    return MVAR

def GIMME(args, network_GT):
    size = len(network_GT)
    path = (f'/Users/sajad/GSU Dropbox Dropbox/Mohammadsajad '
            f'Abavisani/Mac/Documents/PhD/Research/code/DataSets_Feedbacks/8_VAR_simulation/'
            f'ringmore/u{args.UNDERSAMPLING}/GIMMESTD'
           f'/data{args.BATCH}/individual/StdErrors/data{args.BATCH}StdErrors.csv')
    with open(path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header if exists
        rows = []
        for row in csv_reader:
            rows.append(row[1:2*size+1])
        mat = np.array(rows, dtype=np.float32)
        matrix1 = np.array(mat[:, 0:size])
        for i in range(len(network_GT)):
            matrix1[i, i] = 0
        matrix2 = np.array(mat[:, size:2*size])
        binary_matrixA = (matrix1 != 0).astype(int)
        binary_matrixB = (matrix2 != 0).astype(int)
    B0 = np.zeros((len(network_GT), len(network_GT))).astype(int)
    GIMME = cv.adjs2graph(binary_matrixA.T, B0)
    return GIMME

def FASK(args, network_GT):
    path = (f'~'
            f'/DataSets_Feedbacks/8_VAR_simulation/ringmore/u{args.UNDERSAMPLING}/txtSTD/data{args.BATCH}.txt')
    data = pd.read_csv(path, delimiter='\t')
    search = ts.TetradSearch(data)
    search.set_verbose(False)
    search.use_sem_bic()
    search.use_fisher_z(alpha=0.05)

    search.run_fask(alpha=0.05, left_right_rule=1)

    graph_string = str(search.get_string())
    # Parse nodes and edges from the input string
    nodes = mf.parse_nodes(graph_string)
    edges = mf.parse_edges(graph_string)

    # Create adjacency matrix
    adj_matrix = mf.create_adjacency_matrix(edges, nodes)
    for i in range(len(network_GT)):
        adj_matrix[i, i] = 0
    B = np.zeros((len(network_GT), len(network_GT))).astype(int)
    FASK = cv.adjs2graph(adj_matrix, np.zeros((len(network_GT), len(network_GT))))
    return FASK

def RASL(args, network_GT):
    data = zkl.load(f'datasets/VAR_ringmore_V_ruben_undersampled_by_{args.UNDERSAMPLING}_link10.zkl')
    network_GT = data[args.BATCH][0]

    ut = data[args.BATCH][1]
    ut_scaled = ut / ut.max()

    bold_out, _ = hrf.compute_bold_signals(ut_scaled)
    bold_out = bold_out[:, 1000:] # drop initial states
    dataframe = pp.DataFrame(bold_out.T)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05)
    g_estimated, A, B = cv.Glag2CG(results)
    MAXCOST = 10000
    DD = (np.abs((np.abs(A / np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
    BD = (np.abs((np.abs(B / np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

    r_estimated = drasl([g_estimated], weighted=True, capsize=0, timeout=0,
                        urate=min(5, (3 * len(g_estimated) + 1)),
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
    return res_rasl

def mRASL(args, network_GT):
    BATCH = args.BATCH*6
    network_GT = zkl.load(os.path.expanduser(f'~/DataSets_Feedbacks/8_VAR_simulation/ringmore/u{args.UNDERSAMPLING}/GT/GT{BATCH}.zkl'))
    MAXCOST = 1000
    N = len(network_GT)
    base_g = {i: {} for i in range(1, N + 1)}
    base_DD = np.zeros((N,N)).astype(int)
    base_BD = np.zeros((N,N)).astype(int)
    g_est_list = []
    DD_list = []
    BD_list = []
    for i in range(6):
        path = os.path.expanduser(f'~/DataSets_Feedbacks/8_VAR_simulation/ringmore/u{args.UNDERSAMPLING}/txtSTD/data{BATCH-i}.txt')
        data = pd.read_csv(path, delimiter='\t')
        # dataframe = pp.DataFrame(data.values)
        # cond_ind_test = ParCorr()
        # pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        # results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05)
        # g_estimated, A, B = cv.Glag2CG(results)
        bold_out, _ = hrf.compute_bold_signals(data.values)
        g_estimated, A, B = lm.data2graph(bold_out.T, th=0.03)
        DD = (np.abs((np.abs(A / np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
        BD = (np.abs((np.abs(B / np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

        g_est_list.append(g_estimated)
        DD_list.append(DD)
        BD_list.append(BD)

        base_g = mf.update_base_graph(base_g, g_estimated)
        base_DD, base_BD = mf.update_DD_BD(g_estimated, DD, BD, base_DD, base_BD,base_g)


    base_DD = np.where(base_DD < 0, 6000 + base_DD, base_DD)
    base_BD = np.where(base_BD < 0, 6000 + base_BD, base_BD)
    r_estimated = drasl(g_est_list, weighted=True, capsize=0, timeout=0,
                        urate=min(5, (3 * len(g_estimated) + 1)),
                        dm=DD_list,
                        bdm=BD_list,
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
    return res_rasl

def initialize_metrics():
    return {
        'Precision_O': [], 'Recall_O': [], 'F1_O': [],
        'Precision_A': [], 'Recall_A': [], 'F1_A': [],
        'Precision_C': [], 'Recall_C': [], 'F1_C': []
    }

def convert_to_mat(args):
    data = zkl.load(f'datasets/VAR_ringmore_V_ruben_undersampled_by_{args.UNDERSAMPLING}_link10.zkl')
    for i, dd in enumerate(data, start=1):
        folder = os.path.expanduser(f'~/DataSets_Feedbacks/8_VAR_simulation/ringmore/u{args.UNDERSAMPLING}/mat')
        if not os.path.exists(folder):
            os.makedirs(folder)
        savemat(folder + '/expo_to_mat_' + str(i) + '.mat', {'dd': dd[1]})

def convert_to_txt(args):
    data = zkl.load(f'datasets/VAR_ringmore_V_ruben_undersampled_by_{args.UNDERSAMPLING}_link10.zkl')
    for i, dd in enumerate(data, start=1):
        folder = os.path.expanduser(f'~/DataSets_Feedbacks/8_VAR_simulation/ringmore/u{args.UNDERSAMPLING}/txtSTD')
        if not os.path.exists(folder):
            os.makedirs(folder)
        zkl.save(dd[0], f'{folder}/GT{i}.zkl')
        variances = np.var(dd[1], axis=1, ddof=0)  # ddof=0 for population variance

        # Calculate the standard deviation (sqrt of variance) for each row
        std_devs = np.sqrt(variances)

        # Normalize each row by dividing by its standard deviation
        normalized_array = dd[1] / std_devs[:, np.newaxis]

        # Calculate the mean of each row in the normalized matrix
        means = np.mean(normalized_array, axis=1)

        # Zero-mean each row by subtracting the mean from each element
        zero_mean_array = normalized_array - means[:, np.newaxis]

        header = '\t'.join([f'X{j + 1}' for j in range(dd[1].shape[0])])

        with open(f'{folder}/data{i}.txt', 'w') as f:
            # Write the header
            f.write(header + '\n')

            # Write the data, one column per line
            for col in range(zero_mean_array.shape[1]):
                line = '\t'.join(map(str, zero_mean_array[:, col]))
                f.write(line + '\n')

def run_analysis(args,network_GT,include_selfloop):
    metrics = {key: {args.UNDERSAMPLING: initialize_metrics()} for key in [args.METHOD]}

    for method in metrics.keys():
        # loading = f'datasets/{method}/net{args.NET}' \
        #           f'_undersampled_by_{args.UNDERSAMPLING}_batch{args.BATCH}.*'
        # if not glob.glob(loading):
        #     save_dataset(args)

        result = globals()[method](args, network_GT)
        print(f"Result from {method}: {result}")
        normal_GT = mf.precision_recall_all_cycle(result, network_GT, include_selfloop=include_selfloop)
        metrics[method][args.UNDERSAMPLING]['Precision_O'].append(normal_GT['orientation']['precision'])
        metrics[method][args.UNDERSAMPLING]['Recall_O'].append(normal_GT['orientation']['recall'])
        metrics[method][args.UNDERSAMPLING]['F1_O'].append(normal_GT['orientation']['F1'])

        metrics[method][args.UNDERSAMPLING]['Precision_A'].append(normal_GT['adjacency']['precision'])
        metrics[method][args.UNDERSAMPLING]['Recall_A'].append(normal_GT['adjacency']['recall'])
        metrics[method][args.UNDERSAMPLING]['F1_A'].append(normal_GT['adjacency']['F1'])

        metrics[method][args.UNDERSAMPLING]['Precision_C'].append(normal_GT['cycle']['precision'])
        metrics[method][args.UNDERSAMPLING]['Recall_C'].append(normal_GT['cycle']['recall'])
        metrics[method][args.UNDERSAMPLING]['F1_C'].append(normal_GT['cycle']['F1'])

    print(metrics)
    if not os.path.exists('VAR_ringmore_v3'):
        os.makedirs('VAR_ringmore_v3')
    filename = f'VAR_ringmore_v3/VAR_{args.METHOD}_BOLD_ruben_ringmore_undersampled_by_{args.UNDERSAMPLING}_batch_{args.BATCH}.zkl'
    zkl.save(metrics,filename)
    print('file saved to :' + filename)

def save_dataset(args):
    dataset = []
    #graph size and number of edges to match Ruben simple networks

    x_values = np.array([5, 6, 7, 8, 9, 10])
    y_values = np.array([7, 8, 10, 10, 11, 19])
    # Fit a polynomial of degree 5
    coefficients = np.polyfit(x_values, y_values, 6)
    size = 5 + ((args.BATCH-1) % 6)
    i = 1 + ((args.BATCH-1) / 6)
    # for size in range(5,11):
    #     for i in range(10):
    GT = gk.ringmore(size, int(round(np.polyval(coefficients, size) - size)))
    A = cv.graph2adj(GT)


    for j in range(6):
        W = mf.create_stable_weighted_matrix(A,
                                             threshold=int((args.MINLINK)*(3**-args.UNDERSAMPLING)*(3**args.MAXU))/ 1000,
                                             powers=[t for t in range(1,args.UNDERSAMPLING + 1)])
        data = (GT,mf.genData(W, rate=1, ssize=5000* args.UNDERSAMPLING, noise=args.NOISE)) # we undersample after hrf function
        data_scaled = data[1] / data[1].max()

        bold_out, _ = hrf.compute_bold_signals(data_scaled)
        bold_out = bold_out[:, int((bold_out.shape[1])/5):]  # drop initial states
        data_undersampled = bold_out[:, ::args.UNDERSAMPLING] #undersample
        dataset.append((GT,data_undersampled))
    zkl.save(dataset,f'datasets/VAR_BOLD_ringmore_V_ruben_undersampled_by_{args.UNDERSAMPLING}_link_version2{args.MINLINK}_batch{args.BATCH}.zkl')

if __name__ == "__main__":
    error_normalization = True
    CLINGO_LIMIT = 64
    PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))
    POSTFIX = 'VAR_ruben_nets'
    PreFix = 'PCMCI'

    args = parse_arguments(PNUM)
    args = convert_str_to_bool(args)
    omp_num_threads = args.PNUM
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    # network_GT = zkl.load(os.path.expanduser(f'~/DataSets_Feedbacks/8_VAR_simulation/ringmore/u{args.UNDERSAMPLING}/GT/GT{args.BATCH}.zkl'))
    include_selfloop = False
    # pattern = f'datasets/VAR_sim_ruben_simple_net{args.NET}_undersampled_by_{args.UNDERSAMPLING}.zkl'

    # if not glob.glob(pattern):
    # for i in range(1,4):
    #     args.UNDERSAMPLING = i
    save_dataset(args)
    # for i in range(1,10):
    # for j in range(1,4):
    #     args.UNDERSAMPLING = j
    #     print(f' u {j}')
    #     convert_to_mat(args)
    #         args.NET = i
    #         args.UNDERSAMPLING = j
    # convert_to_txt(args)

    # run_analysis(args,network_GT,include_selfloop)