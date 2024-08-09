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
sys.path.append('/Users/sajad/GSU Dropbox Dropbox/Mohammadsajad Abavisani/Mac/Documents/PhD/Research/code/tread/py-tetrad')
from py_tetrad.tools import TetradSearch as ts

def parse_arguments(PNUM):
    parser = argparse.ArgumentParser(description='Run settings.')
    parser.add_argument("-c", "--CAPSIZE", default=0,
                        help="stop traversing after growing equivalence class to this size.", type=int)
    parser.add_argument("-b", "--BATCH", default=1, help="slurm batch.", type=int)
    parser.add_argument("-p", "--PNUM", default=PNUM, help="number of CPUs in machine.", type=int)
    parser.add_argument("-n", "--NET", default=1, help="number of simple network", type=int)
    parser.add_argument("-l", "--MINLINK", default=2, help=" lower threshold transition matrix abs value x10", type=int)
    parser.add_argument("-z", "--NOISE", default=10, help="noise str multiplied by 100", type=int)
    parser.add_argument("-s", "--SCC", default="f", help="true to use SCC structure, false to not", type=str)
    parser.add_argument("-m", "--SCCMEMBERS", default="f",
                        help="true for using g_estimate SCC members, false for using "
                             "GT SCC members", type=str)
    parser.add_argument("-u", "--UNDERSAMPLING", default=3, help="sampling rate in generated data", type=int)
    parser.add_argument("-x", "--MAXU", default=5, help="maximum number of undersampling to look for solution.",
                        type=int)
    parser.add_argument("-a", "--ALPHA", default=50, help="alpha_level for PC multiplied by 1000", type=int)
    parser.add_argument("-y", "--PRIORITY", default="43521", help="string of priorities", type=str)
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
    path = f'./DataSets_Feedbacks/8_VAR_simulation/net{args.NET}/u{args.UNDERSAMPLING}/MVGC'
    mat_data = loadmat(path + f'/mat_file_{args.BATCH}.mat')['sig']
    for i in range(len(network_GT)):
        mat_data[i, i] = 1
    B = np.zeros((len(network_GT), len(network_GT))).astype(int)
    MVGC = cv.adjs2graph(mat_data, np.zeros((len(network_GT), len(network_GT))))
    return MVGC

def MVAR(args, network_GT):
    path = f'./DataSets_Feedbacks/8_VAR_simulation/net{args.NET}/u{args.UNDERSAMPLING}/MVAR'
    mat_data = loadmat(path + f'/mat_file_{args.BATCH}.mat')['sig']
    for i in range(len(network_GT)):
        mat_data[i, i] = 1
    B = np.zeros((len(network_GT), len(network_GT))).astype(int)
    MVAR = cv.adjs2graph(mat_data, np.zeros((len(network_GT), len(network_GT))))
    return MVAR

def GIMME(args, network_GT):
    size = len(network_GT)
    path = f'./DataSets_Feedbacks/8_VAR_simulation/net{args.NET}/u{args.UNDERSAMPLING}/GIMMESTD' \
           f'/individual/StdErrors/data{args.BATCH}StdErrors.csv'
    with open(path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header if exists
        rows = []
        for row in csv_reader:
            rows.append(row[1:2*size+1])
        mat = np.array(rows, dtype=np.float32)
        matrix1 = np.array(mat[:, 0:size])
        matrix2 = np.array(mat[:, size:2*size])
        binary_matrixA = (matrix1 != 0).astype(int)
        binary_matrixB = (matrix2 != 0).astype(int)
    B0 = np.zeros((len(network_GT), len(network_GT))).astype(int)
    GIMME = cv.adjs2graph(binary_matrixA.T, B0)
    return GIMME

def FASK(args, network_GT):
    path = f'./DataSets_Feedbacks/8_VAR_simulation/net{args.NET}/u{args.UNDERSAMPLING}/txtSTD/data{args.BATCH}.txt'
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
        adj_matrix[i, i] = 1
    B = np.zeros((len(network_GT), len(network_GT))).astype(int)
    FASK = cv.adjs2graph(adj_matrix, np.zeros((len(network_GT), len(network_GT))))
    return FASK

def RASL(args, network_GT):
    return "RASL result"

def mRASL(args, network_GT):
    return "mRASL result"

def initialize_metrics():
    return {
        'Precision_O': [], 'Recall_O': [], 'F1_O': [],
        'Precision_A': [], 'Recall_A': [], 'F1_A': [],
        'Precision_C': [], 'Recall_C': [], 'F1_C': []
    }

def convert_to_mat(args):
    data = zkl.load(f'datasets/VAR_sim_ruben_simple_net{args.NET}_undersampled_by_{args.UNDERSAMPLING}.zkl')
    for i, dd in enumerate(data, start=1):
        folder = f'./DataSets_Feedbacks/8_VAR_simulation/net{args.NET}/u{args.UNDERSAMPLING}/mat'
        if not os.path.exists(folder):
            os.makedirs(folder)
        savemat(folder + '/expo_to_mat_' + str(i) + '.mat', {'dd': dd})

def convert_to_txt(args):
    data = zkl.load(f'datasets/VAR_sim_ruben_simple_net{args.NET}_undersampled_by_{args.UNDERSAMPLING}.zkl')
    for i, dd in enumerate(data, start=1):
        folder = f'./DataSets_Feedbacks/8_VAR_simulation/net{args.NET}/u{args.UNDERSAMPLING}/txt'
        if not os.path.exists(folder):
            os.makedirs(folder)
        variances = np.var(dd, axis=1, ddof=0)  # ddof=0 for population variance

        # Calculate the standard deviation (sqrt of variance) for each row
        std_devs = np.sqrt(variances)

        # Normalize each row by dividing by its standard deviation
        normalized_array = dd / std_devs[:, np.newaxis]

        # Calculate the mean of each row in the normalized matrix
        means = np.mean(normalized_array, axis=1)

        # Zero-mean each row by subtracting the mean from each element
        zero_mean_array = normalized_array - means[:, np.newaxis]

        header = '\t'.join([f'X{j + 1}' for j in range(dd.shape[0])])

        with open(f'{folder}/data{i}.txt', 'w') as f:
            # Write the header
            f.write(header + '\n')

            # Write the data, one column per line
            for col in range(zero_mean_array.shape[1]):
                line = '\t'.join(map(str, zero_mean_array[:, col]))
                f.write(line + '\n')

def run_analysis(args,network_GT):
    metrics = {key: initialize_metrics() for key in ['MVGC', 'MVAR', 'GIMME', 'FASK', 'RASL', 'mRASL']}
    for method in metrics.keys():
        # loading = f'datasets/{method}/net{args.NET}' \
        #           f'_undersampled_by_{args.UNDERSAMPLING}_batch{args.BATCH}.*'
        # if not glob.glob(loading):
        #     save_dataset(args)

        result = globals()[method](args, network_GT)
        print(f"Result from {method}: {result}")


    GT = simp_nets(args.NET, selfloop=True)
    A = cv.graph2adj(GT)
    W = mf.create_stable_weighted_matrix(A, threshold=args.MINLINK / 10, powers=[2, 3, 4])
    data = lm.genData(W, rate=args.UNDERSAMPLING, ssize=5000, noise=args.NOISE)
    MAXCOST = 10000

    dataframe = pp.DataFrame(np.transpose(data))
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=args.ALPHA)

    g_estimated, A, B = mf.Glag2CG(results)
    DD = (np.abs((np.abs(A / np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
    BD = (np.abs((np.abs(B / np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

def save_dataset(args):
    dataset = []
    GT = simp_nets(args.NET, selfloop=True)
    A = cv.graph2adj(GT)
    W = mf.create_stable_weighted_matrix(A, threshold=args.MINLINK / 10, powers=[2, 3])
    for i in range(60):
        data = mf.genData(W, rate=args.UNDERSAMPLING, ssize=5000, noise=args.NOISE)
        dataset.append(data)
        print(f'generating batch:{i}')
    zkl.save(dataset,f'datasets/VAR_sim_ruben_simple_net{args.NET}_undersampled_by_{args.UNDERSAMPLING}.zkl')

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
    network_GT = simp_nets(args.NET, selfloop=True)
    pattern = f'datasets/VAR_sim_ruben_simple_net{args.NET}_undersampled_by_{args.UNDERSAMPLING}.zkl'

    # if not glob.glob(pattern):
    #     save_dataset(args)
    # for i in range(1,10):
    #     for j in range(1,4):
    #         print(f'net {i}, u {j}')
    #         args.NET = i
    #         args.UNDERSAMPLING = j
    #         convert_to_txt(args)
    run_analysis(args,network_GT)