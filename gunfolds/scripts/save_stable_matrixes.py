import os
import copy
import distutils.util
from gunfolds.utils import bfutils
from gunfolds.utils import zickle as zkl
import numpy as np
import time, socket
import scipy
from gunfolds.solvers.clingo_rasl import drasl, drasl_command
import argparse
from gunfolds.utils import graphkit as gk
from gunfolds.utils.calc_procs import get_process_count
from gunfolds.estimation import linear_model as lm
from gunfolds import conversions as cv
from numpy import linalg as la
import networkx as nx
from math import log
from gunfolds.viz import gtool as gt
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from matplotlib import pyplot as plt
from matplotlib import cm

CLINGO_LIMIT = 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))
POSTFIX = 'VAR_stable_trans_mat'
Using_SVAR = True
PreFix = 'SVAR' if Using_SVAR else 'GC'
parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-c", "--CAPSIZE", default=0,
                    help="stop traversing after growing equivalence class to this size.", type=int)
parser.add_argument("-b", "--BATCH", default=11, help="slurm batch.", type=int)
parser.add_argument("-p", "--PNUM", default=PNUM, help="number of CPUs in machine.", type=int)
parser.add_argument("-n", "--NODE", default=8, help="number of nodes in graph", type=int)
parser.add_argument("-d", "--DEN", default=0.14, help="density of graph", type=str)
parser.add_argument("-g", "--GTYPE", default="t", help="true for ringmore graph, false for random graph", type=str)
parser.add_argument("-t", "--TIMEOUT", default=120, help="timeout in hours", type=int)
parser.add_argument("-r", "--THRESHOLD", default=5, help="threshold for SVAR", type=int)
parser.add_argument("-l", "--MINLINK", default=5, help=" lower threshold transition matrix abs value x10", type=int)
parser.add_argument("-z", "--NOISE", default=10, help="noise str multiplied by 100", type=int)
parser.add_argument("-s", "--SCC", default="f", help="true to use SCC structure, false to not", type=str)
parser.add_argument("-m", "--SCCMEMBERS", default="f", help="true for using g_estimate SCC members, false for using "
                                                            "GT SCC members", type=str)
parser.add_argument("-u", "--UNDERSAMPLING", default=2, help="sampling rate in generated data", type=int)
parser.add_argument("-x", "--MAXU", default=15, help="maximum number of undersampling to look for solution.", type=int)
args = parser.parse_args()
TIMEOUT = args.TIMEOUT * 60 * 60
GRAPHTYPE = bool(distutils.util.strtobool(args.GTYPE))
DENSITY = float(args.DEN)
graphType = 'ringmore' if GRAPHTYPE else 'bp_mean'
SCC = bool(distutils.util.strtobool(args.SCC))
SCC_members = bool(distutils.util.strtobool(args.SCCMEMBERS))
SCC = True if SCC_members else SCC
u_rate = args.UNDERSAMPLING
k_threshold = args.THRESHOLD
EDGE_CUTOFF = 0.01
noise_svar = args.NOISE / 100

error_normalization = True

def round_tuple_elements(input_tuple, decimal_points=3):
    return tuple(round(elem, decimal_points) if isinstance(elem, (int, float)) else elem for elem in input_tuple)



def check_matrix_powers(W, A, powers, threshold):
    for n in powers:
        W_n = np.linalg.matrix_power(W, n)
        non_zero_indices = np.nonzero(W_n)
        if (np.abs(W_n[non_zero_indices]) < threshold).any():
            return False
    return True


def create_stable_weighted_matrix(
    A,
    threshold=0.1,
    powers=[1, 2, 3, 4],
    max_attempts=10000000,
    damping_factor=0.99,
    random_state=None,
):
    np.random.seed(
        random_state
    )  # Set random seed for reproducibility if provided
    attempts = 0

    while attempts < max_attempts:
        # Generate a random matrix with the same sparsity pattern as A
        random_weights = np.random.randn(*A.shape)
        weighted_matrix = A * random_weights

        # Convert to sparse format for efficient eigenvalue computation
        weighted_sparse = sp.csr_matrix(weighted_matrix)

        # Compute the largest eigenvalue in magnitude
        eigenvalues, _ = eigs(weighted_sparse, k=1, which="LM")
        max_eigenvalue = np.abs(eigenvalues[0])

        # Scale the matrix so that the spectral radius is slightly less than 1
        if max_eigenvalue > 0:
            weighted_matrix *= damping_factor / max_eigenvalue
            # Check if the powers of the matrix preserve the threshold for non-zero entries of A
            if check_matrix_powers(weighted_matrix, A, powers, threshold):
                return weighted_matrix

        attempts += 1

    raise ValueError(
        f"Unable to create a matrix satisfying the condition after {max_attempts} attempts."
    )

def get_strongly_connected_components(graph):
    return [c for c in nx.strongly_connected_components(graph)]


def calculate_jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def quantify_graph_difference(graph1, graph2):
    # Step 1: Find the strongly connected components in both graphs
    scc1 = get_strongly_connected_components(graph1)
    scc2 = get_strongly_connected_components(graph2)

    # Step 2: Represent SCCs as sets of vertices
    scc_sets1 = set([frozenset(component) for component in scc1])
    scc_sets2 = set([frozenset(component) for component in scc2])

    # Step 3: Calculate Jaccard similarity between SCC sets
    intersection = len(scc_sets1.intersection(scc_sets2))
    union = len(scc_sets1.union(scc_sets2))
    jaccard_similarity = intersection / union if union > 0 else 0

    return jaccard_similarity








# def genData(A, rate=2, burnin=100, ssize=2000, noise=0.1, dist='beta'):
def genData(A, rate=2, burnin=100, ssize=5000, noise=0.1, dist='normal'):

    Agt = A
    data = drawsamplesLG(Agt, samples=burnin + (ssize * rate), nstd=noise)
    data = data[:, burnin:]
    return data[:, ::rate]


def drawsamplesLG(A, nstd=0.1, samples=100):

    n = A.shape[0]
    data = np.zeros([n, samples])
    data[:, 0] = nstd * np.random.randn(A.shape[0])
    for i in range(1, samples):
        data[:, i] = A @ data[:, i - 1] + nstd * np.random.randn(A.shape[0])
    return data







'''
if you get a graph G_1 that is a DAG (directed acyclic graph), i.e. all SCCs are singletons (each node is its own SCC)
 then this is a very non-biological case and you need to discard it and try to generate another/better graph
there are a few things to consider though
the case you describe: if any of these singleton nodes have a self-loop, we can use this graph OK. Not ideal,
but better than a simple DAG. Also, in this case you do not risk getting a Gu with no directed edges
2. since you are using scc=true  it is worth making sure all sccs are gcd=1 . I suspect there is already 
a function to ensure this in graphkit
New
https://neuroneural.github.io/gunfolds/utils/graphkit.html#ensure-gcd1
3:44
but this function will be bad for singleton nodes - it will add a self loop to all of them, which is not a good idea
 :slightly_smiling_face: Please do not apply it to singleton SCCs
'''
print('_____________________________________________')
if GRAPHTYPE:
    dataset = zkl.load('datasets/ringmore_n8d14.zkl')
    GT = dataset[args.BATCH-1]
    A = cv.graph2adj(GT)
    W = create_stable_weighted_matrix(A, threshold=args.MINLINK/10, powers=[2, 3, 4])
else:
    GT = gk.bp_mean_degree_graph(8, 1.4)
    A = cv.graph2adj(GT)
    W = create_stable_weighted_matrix(A, threshold=args.MINLINK / 10, powers=[2, 3, 4])



'''saving files'''
filename = 'Stable_matrix_ringmore_'+str(GRAPHTYPE) + '_batch_' + str(args.BATCH)
folder = 'stable_matrix'
if not os.path.exists(folder):
    os.makedirs(folder)
zkl.save(W, folder + '/' + filename + '.zkl')
print('_____________________________________________')
