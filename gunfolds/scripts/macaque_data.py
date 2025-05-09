import os
from brainiak.utils import fmrisim
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
from gunfolds.scripts.datasets.simple_networks import macaque_net
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
    parser.add_argument("-b", "--BATCH", default=1, help="slurm batch.", type=int)
    parser.add_argument("-p", "--PNUM", default=PNUM, help="number of CPUs in machine.", type=int)
    parser.add_argument("-r", "--SNR", default=1, help="Signal to noise ratio", type=int)
    parser.add_argument("-z", "--NOISE", default=10, help="noise str multiplied by 100", type=int)
    parser.add_argument("-s", "--SCC", default="t", help="true to use SCC structure, false to not", type=str)
    parser.add_argument("-m", "--SCCMEMBERS", default="f",
                        help="true for using g_estimate SCC members, false for using "
                             "GT SCC members", type=str)
    parser.add_argument("-u", "--UNDERSAMPLING", default=75, help="sampling rate in generated data", type=int)
    parser.add_argument("-x", "--MAXU", default=4, help="maximum number of undersampling to look for solution.",
                        type=int)
    parser.add_argument("-t", "--CONCAT", default="t", help="true to use concat data, false to not", type=str)

    parser.add_argument("-y", "--PRIORITY", default="11112", help="string of priorities", type=str)
    parser.add_argument("-o", "--METHOD", default="RASL", help="method to run", type=str)
    parser.add_argument("-v", "--VERSION", default="LongRange", help="version of macaque data", type=str)
    return parser.parse_args()

def convert_str_to_bool(args):
    args.SCC = bool(strtobool(args.SCC))
    args.SCCMEMBERS = bool(strtobool(args.SCCMEMBERS))
    args.CONCAT = bool(strtobool(args.CONCAT))
    args.NOISE = args.NOISE / 100
    priprities = []
    for char in args.PRIORITY:
        priprities.append(int(char))
    args.PRIORITY = priprities
    return args


# Define the functions
def MVGC(args, network_GT):
    path = os.path.expanduser(f'~/DataSets_Feedbacks/9_VAR_BOLD_simulation/ri'
            f'ngmore/u{args.UNDERSAMPLING}/MVGC')
    mat_data = loadmat(path + f'/mat_file_{args.BATCH}.mat')['sig']
    for i in range(len(network_GT)):
        mat_data[i, i] = 0
    B = np.zeros((len(network_GT), len(network_GT))).astype(int)
    MVGC = cv.adjs2graph(mat_data, np.zeros((len(network_GT), len(network_GT))))
    return MVGC

def MVAR(args, network_GT):
    path = os.path.expanduser(f'~/DataSets_Feedbacks/9_VAR_BOLD_simulation/ri'
            f'ngmore/u{args.UNDERSAMPLING}/MVAR')
    mat_data = loadmat(path + f'/mat_file_{args.BATCH}.mat')['sig']
    for i in range(len(network_GT)):
        mat_data[i, i] = 0
    B = np.zeros((len(network_GT), len(network_GT))).astype(int)
    MVAR = cv.adjs2graph(mat_data, np.zeros((len(network_GT), len(network_GT))))
    return MVAR

def GIMME(args, network_GT):
    size = len(network_GT)
    path = os.path.expanduser(f'~/DataSets_Feedbacks/9_VAR_BOLD_simulation/'
            f'ringmore/u{args.UNDERSAMPLING}/GIMME'
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
    num = str(args.BATCH) if args.BATCH > 9 else '0' + str(args.BATCH)
    print('reading file:' + num)
    if not args.CONCAT:
        data = pd.read_csv(
            os.path.expanduser(
                f"~/DataSets_Feedbacks/2. Macaque_Networks/{args.VERSION}/data_fslfilter/BOLDfslfilter_{num}.txt"), delimiter='\t')
    else:
        data = pd.read_csv(
            os.path.expanduser(
                f'~/DataSets_Feedbacks/2. Macaque_Networks/{args.VERSION}/data_fslfilter_concat/concat_BOLDfslfilter_{num}.txt'), delimiter='\t')


    # path = os.path.expanduser(f'~/DataSets_Feedbacks/2. Macaque_Networks/Full/data_fslfilter_concat/concat_BOLDfslfilter_{args.BATCH}.txt')
    # data = pd.read_csv(path, delimiter='\t')
    search = ts.TetradSearch(data)
    search.set_verbose(False)
    search.use_sem_bic()
    search.use_fisher_z(alpha=0.0000001)

    search.run_fask(alpha=0.00001, left_right_rule=1)

    graph_string = str(search.get_string())
    # Parse nodes and edges from the input string
    nodes = mf.parse_nodes(graph_string)
    edges = mf.parse_edges(graph_string)

    # Create adjacency matrix
    adj_matrix = mf.create_adjacency_matrix(edges, nodes)
    B = np.zeros((len(network_GT), len(network_GT))).astype(int)
    FASK = cv.adjs2graph(adj_matrix, np.zeros((len(network_GT), len(network_GT))))
    return FASK

def RASL(args, network_GT):
    num = str(args.BATCH) if args.BATCH > 9 else '0' + str(args.BATCH)
    print('reading file:' + num)
    if not args.CONCAT:
        data = pd.read_csv(
            os.path.expanduser(
                f"~/DataSets_Feedbacks/2. Macaque_Networks/{args.VERSION}/data_fslfilter/BOLDfslfilter_{num}.txt"),
            delimiter='\t')
    else:
        data = pd.read_csv(
            os.path.expanduser(
                f'~/DataSets_Feedbacks/2. Macaque_Networks/{args.VERSION}/data_fslfilter_concat/concat_BOLDfslfilter_{num}.txt'),
            delimiter='\t')


    dataframe = pp.DataFrame(data.values)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.01)
    g_estimated, A, B = cv.Glag2CG(results)
    # members = nx.strongly_connected_components(gk.graph2nx(g_estimated))
    members = [s for s in nx.strongly_connected_components(gk.graph2nx(g_estimated))]
    MAXCOST = 10000
    DD = (np.abs((np.abs(A / np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
    BD = (np.abs((np.abs(B / np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

    r_estimated = drasl([g_estimated], weighted=True, capsize=0, timeout=0,
                        urate=min(args.MAXU, (3 * len(g_estimated) + 1)),
                        dm=[DD],
                        bdm=[BD],
                        scc=True,
                        scc_members=members,
                        GT_density=int(1000 * gk.density(network_GT)),
                        edge_weights=args.PRIORITY, pnum=PNUM, optim='optN', selfloop=True)

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
    network_GT = zkl.load(os.path.expanduser(f'~/DataSets_Feedbacks/9_VAR_BOLD_simulation/ringmore/u{args.UNDERSAMPLING}/GT/GT{BATCH}.zkl'))
    MAXCOST = 1000
    N = len(network_GT)
    base_g = {i: {} for i in range(1, N + 1)}
    base_DD = np.zeros((N,N)).astype(int)
    base_BD = np.zeros((N,N)).astype(int)
    g_est_list = []
    DD_list = []
    BD_list = []
    for i in range(6):
        path = os.path.expanduser(f'~/DataSets_Feedbacks/9_VAR_BOLD_simulation/ringmore/u{args.UNDERSAMPLING}/txtSTD/data{BATCH-i}.txt')
        data = pd.read_csv(path, delimiter='\t')
        # dataframe = pp.DataFrame(data.values)
        # cond_ind_test = ParCorr()
        # pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        # results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05)
        # g_estimated, A, B = cv.Glag2CG(results)
        # bold_out, _ = hrf.compute_bold_signals(data.values)
        g_estimated, A, B = lm.data2graph(data.values.T, th=0.03)
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
                        urate=min(args.MAXU, (3 * len(g_estimated) + 1)),
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
        'Precision_A': [], 'Recall_A': [], 'F1_A': []
    }


def run_analysis(args,network_GT,include_selfloop):
    metrics = {key: {args.UNDERSAMPLING: initialize_metrics()} for key in [args.METHOD]}

    for method in metrics.keys():
        # loading = f'datasets/{method}/net{args.NET}' \
        #           f'_undersampled_by_{args.UNDERSAMPLING}_batch{args.BATCH}.*'
        # if not glob.glob(loading):
        #     save_dataset(args)

        result = globals()[method](args, network_GT)
        print(f"Result from {method}: {result}")
        normal_GT = mf.precision_recall_no_cycle(result, network_GT, include_selfloop=include_selfloop)
        metrics[method][args.UNDERSAMPLING]['Precision_O'].append(normal_GT['orientation']['precision'])
        metrics[method][args.UNDERSAMPLING]['Recall_O'].append(normal_GT['orientation']['recall'])
        metrics[method][args.UNDERSAMPLING]['F1_O'].append(normal_GT['orientation']['F1'])

        metrics[method][args.UNDERSAMPLING]['Precision_A'].append(normal_GT['adjacency']['precision'])
        metrics[method][args.UNDERSAMPLING]['Recall_A'].append(normal_GT['adjacency']['recall'])
        metrics[method][args.UNDERSAMPLING]['F1_A'].append(normal_GT['adjacency']['F1'])


    print(metrics)
    if not os.path.exists('macaque_results'):
        os.makedirs('macaque_results')
    filename = f'macaque_results/macaque_{args.VERSION}_{args.METHOD}_batch_{args.BATCH}.zkl'
    zkl.save(metrics,filename)
    print('file saved to :' + filename)



if __name__ == "__main__":
    error_normalization = True
    CLINGO_LIMIT = 64
    PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))
    POSTFIX = 'Macaque_data'
    PreFix = 'ruben_sim'

    args = parse_arguments(PNUM)
    args = convert_str_to_bool(args)
    omp_num_threads = args.PNUM
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    include_selfloop = True


    network_GT = macaque_net(args.VERSION)
    # for i in range(1,61):
    #     args.BATCH = i
    run_analysis(args,network_GT,include_selfloop)