from datetime import datetime
import matplotlib.patches as mpatches
import os
from brainiak.utils import fmrisim
from gunfolds.viz import gtool as gt
from gunfolds.utils import bfutils
import numpy as np
import pandas as pd
from gunfolds import conversions as cv
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.io import loadmat
from scipy.io import savemat
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
    parser.add_argument("-b", "--BATCH", default=2, help="slurm batch.", type=int)
    parser.add_argument("-p", "--PNUM", default=PNUM, help="number of CPUs in machine.", type=int)
    parser.add_argument("-r", "--SNR", default=1, help="Signal to noise ratio", type=int)
    parser.add_argument("-n", "--NET", default=1, help="number of simple network", type=int)
    parser.add_argument("-l", "--MINLINK", default=5, help=" lower threshold transition matrix abs value x1000", type=int)
    parser.add_argument("-z", "--NOISE", default=10, help="noise str multiplied by 100", type=int)
    parser.add_argument("-s", "--SCC", default="f", help="true to use SCC structure, false to not", type=str)
    parser.add_argument("-t", "--CONCAT", default="t", help="true to use concat data, false to not", type=str)
    parser.add_argument("-m", "--SCCMEMBERS", default="f",
                        help="true for using g_estimate SCC members, false for using "
                             "GT SCC members", type=str)
    parser.add_argument("-u", "--UNDERSAMPLING", default=2, help="sampling rate in generated data", type=int)
    parser.add_argument("-x", "--MAXU", default=8, help="maximum number of undersampling to look for solution.",
                        type=int)
    parser.add_argument("-a", "--ALPHA", default=50, help="alpha_level for PC multiplied by 1000", type=int)
    parser.add_argument("-y", "--PRIORITY", default="11112", help="string of priorities", type=str)
    parser.add_argument("-o", "--METHOD", default="FASK", help="method to run", type=str)
    return parser.parse_args()

def convert_str_to_bool(args):
    args.SCC = bool(strtobool(args.SCC))
    args.CONCAT = bool(strtobool(args.CONCAT))
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
    path = 'expo_to_mat/MVGC_expo_to_py_n' + str(args.NET) + '_' + ('concat' if args.CONCAT else 'individual')
    mat_data = loadmat(path + f'/mat_file_{args.BATCH}.mat')['sig']
    for i in range(len(network_GT)):
        mat_data[i, i] = 0
    MVGC = cv.adjs2graph(mat_data, np.zeros((len(network_GT), len(network_GT))))
    return MVGC

def MVAR(args, network_GT):
    path = 'expo_to_mat/MVAR_expo_to_py_n' + str(args.NET) + '_' + ('concat' if args.CONCAT else 'individual') + '_new1'
    mat_data = loadmat(path + f'/mat_file_{args.BATCH}.mat')['sig']
    for i in range(len(network_GT)):
        mat_data[i, i] = 0
    MVAR = cv.adjs2graph(mat_data, np.zeros((len(network_GT), len(network_GT))))
    return MVAR

def GIMME(args, network_GT):
    num = str(args.BATCH) if args.BATCH > 9 else '0' + str(args.BATCH)
    path = os.path.expanduser(f'~/DataSets_Feedbacks/1. Simple_Networks/GIMME_results/{args.NET}_05VARfalse/individual')
    beta_file = f'{path}/concat_BOLDfslfilter_{num}BetasStd.csv'
    std_error_file = f'{path}/StdErrors/concat_BOLDfslfilter_{num}StdErrors.csv'
    graph = mf.read_gimme_to_graph(beta_file, std_error_file)
    numeric_graph = mf.convert_nodes_to_numbers(graph)
    # numeric_graph_no_selfloops = numeric_graph.copy()
    # numeric_graph_no_selfloops.remove_edges_from(nx.selfloop_edges(numeric_graph_no_selfloops))

    GIMME = gk.nx2graph(numeric_graph)
    return GIMME

def FASK(args, network_GT):
    num = str(args.BATCH) if args.BATCH > 9 else '0' + str(args.BATCH)
    print('reading file:' + num)
    if not args.CONCAT:
        data = pd.read_csv(
            os.path.expanduser(
                f"~/DataSets_Feedbacks/1. Simple_Networks/Network{args.NET}_amp/data_fslfilter"
                f"/BOLDfslfilter_{num}.txt"), delimiter='\t')
    else:
        data = pd.read_csv(
            os.path.expanduser(
                f"~/DataSets_Feedbacks/1. Simple_Networks/Network{args.NET}_amp/data_fslfilter_concat"
                f"/concat_BOLDfslfilter_{num}.txt"), delimiter='\t')

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
    B = np.zeros((len(network_GT), len(network_GT))).astype(int)
    FASK = cv.adjs2graph(adj_matrix, np.zeros((len(network_GT), len(network_GT))))
    return FASK


def PC(args, network_GT):
    path = os.path.expanduser(f'~'
            f'/DataSets_Feedbacks/9_VAR_BOLD_simulation/ringmore/u{args.UNDERSAMPLING}/txt/data{args.BATCH}.txt')
    data = pd.read_csv(path, delimiter='\t')
    dataframe = pp.DataFrame(data.values)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05)
    g_estimated, _, _ = cv.Glag2CG(results)
    PC = mf.remove_bidir_edges(g_estimated)
    return PC

def RASL_meta(result, args, network_GT):
    B = np.zeros((len(network_GT), len(network_GT))).astype(int)
    nx_result = gk.graph2nx(result)
    two_cycle = mf.find_two_cycles(nx_result)
    DD = np.ones((len(network_GT), len(network_GT))) * 5000
    BD = np.ones((len(network_GT), len(network_GT))) * 10000
    for cycle in two_cycle:
        DD[cycle[0] - 1][cycle[1] - 1] = 2500
        DD[cycle[1] - 1][cycle[0] - 1] = 2500
        B[cycle[0] - 1][cycle[1] - 1] = 1
        B[cycle[1] - 1][cycle[0] - 1] = 1

    for i in range(len(network_GT)):
        DD[i][i] = 10000
    result_bi = cv.adjs2graph(cv.graph2adj(result), B)
    r_estimated = drasl([result_bi], weighted=True, capsize=0, timeout=0,
                        urate=min(args.MAXU, (3 * len(result_bi) + 1)),
                        dm=[DD],
                        bdm=[BD],
                        scc=False,
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


def PCMCI_RASL(args, network_GT):
    num = str(args.BATCH) if args.BATCH > 9 else '0' + str(args.BATCH)
    print('reading file:' + num)
    if not args.CONCAT:
        data = pd.read_csv(
            os.path.expanduser(
                f"~/DataSets_Feedbacks/1. Simple_Networks/Network{args.NET}_amp/data_fslfilter"
                f"/BOLDfslfilter_{num}.txt"), delimiter='\t')
    else:
        data = pd.read_csv(
            os.path.expanduser(
                f"~/DataSets_Feedbacks/1. Simple_Networks/Network{args.NET}_amp/data_fslfilter_concat"
                f"/concat_BOLDfslfilter_{num}.txt"), delimiter='\t')

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
        'Precision_A': [], 'Recall_A': [], 'F1_A': [],
        'Precision_C': [], 'Recall_C': [], 'F1_C': []
    }




def run_analysis(args,network_GT,include_selfloop):
    metrics = {key: {approach:initialize_metrics() for approach in ['org', 'gt2','meta','pc']} for key in [args.METHOD]}

    for method in metrics.keys():

        result = globals()[method](args, network_GT)
        print(f"Result from {method}: {result}")
        normal_GT = mf.precision_recall_all_cycle(result, network_GT, include_selfloop=include_selfloop)
        for category, prefix in zip(['orientation', 'adjacency', 'cycle'], ['O', 'A', 'C']):
            for metric in ['precision', 'recall', 'F1']:
                metrics[method]['org'][f"{metric.capitalize()}_{prefix}"].append(normal_GT[category][metric])


        new_GT = mf.remove_bidir_edges(bfutils.all_undersamples(network_GT)[1])
        undersampled_GT = mf.precision_recall(result, new_GT, include_selfloop=include_selfloop)
        for category, prefix in zip(['orientation', 'adjacency', 'cycle'], ['O', 'A', 'C']):
            for metric in ['precision', 'recall', 'F1']:
                metrics[method]['gt2'][f"{metric.capitalize()}_{prefix}"].append(undersampled_GT[category][metric])


        meta_rasl_result = RASL_meta(result, args, network_GT)
        meta_rasl = mf.precision_recall(meta_rasl_result, network_GT, include_selfloop=include_selfloop)
        for category, prefix in zip(['orientation', 'adjacency', 'cycle'], ['O', 'A', 'C']):
            for metric in ['precision', 'recall', 'F1']:
                metrics[method]['meta'][f"{metric.capitalize()}_{prefix}"].append(meta_rasl[category][metric])

        pattern = f'ruben_runs/*/rubenNets_*_net_{args.NET}_batch_{args.BATCH}.zkl'
        matching_files = glob.glob(pattern)
        if matching_files:
            file_to_load = matching_files[0]
            data = zkl.load(file_to_load)
            pc_value = None
            for key, sub_dict in data.items():
                if isinstance(sub_dict, dict) and 'pc' in sub_dict:
                    pc_value = sub_dict['pc']
                    break  # Stop at the first match (if needed)

            if pc_value is not None:
                metrics[method]['pc'] = pc_value
            else:
                print(f"No 'pc' key found in any sub-dictionaries. Available keys: {list(data.keys())}")

        else:
            pcmci_rasl_result = PCMCI_RASL(args, network_GT)
            pcmci_rasl = mf.precision_recall(pcmci_rasl_result, network_GT, include_selfloop=include_selfloop)
            for category, prefix in zip(['orientation', 'adjacency', 'cycle'], ['O', 'A', 'C']):
                for metric in ['precision', 'recall', 'F1']:
                    metrics[method]['pc'][f"{metric.capitalize()}_{prefix}"].append(pcmci_rasl[category][metric])

        if not os.path.exists(f'ruben_runs/{args.METHOD}'):
            os.makedirs(f'ruben_runs/{args.METHOD}')
        filename = f'ruben_runs/{args.METHOD}/rubenNets_{args.METHOD}_net_{args.NET}_batch_{args.BATCH}.zkl'
        zkl.save(metrics,filename)
        print(metrics)
        print('file saved to :' + filename)
        print('-----------------------------------------------------')



if __name__ == "__main__":
    error_normalization = True
    CLINGO_LIMIT = 64
    PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))
    POSTFIX = 'ruben_nets'
    PreFix = 'all'

    args = parse_arguments(PNUM)
    args = convert_str_to_bool(args)
    omp_num_threads = args.PNUM
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    include_selfloop = True
    for i in range(4,5):
        for j in range(1,61):
            args.BATCH = j
            args.NET = i
            pattern = f'ruben_runs/{args.METHOD}/rubenNets_{args.METHOD}_net_{args.NET}_batch_{args.BATCH}.zkl'

            if not glob.glob(pattern):
                network_GT = simp_nets(args.NET, True)
                run_analysis(args, network_GT, include_selfloop)
