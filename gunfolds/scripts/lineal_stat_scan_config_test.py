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
from progressbar import ProgressBar, Percentage
from numpy import linalg as la
import networkx as nx
from math import log


CLINGO_LIMIT = 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))
POSTFIX = 'linear_simu'
Using_SVAR = True
Using_VAR = False
PreFix = 'SVAR' if Using_SVAR else 'GC'
parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-c", "--CAPSIZE", default=0,
                    help="stop traversing after growing equivalence class to this size.", type=int)
parser.add_argument("-b", "--BATCH", default=1, help="slurm batch.", type=int)
parser.add_argument("-p", "--PNUM", default=PNUM, help="number of CPUs in machine.", type=int)
parser.add_argument("-n", "--NODE", default=6, help="number of nodes in graph", type=int)
parser.add_argument("-d", "--DEN", default=0.15, help="density of graph", type=str)
parser.add_argument("-o", "--CONFIG", default="auto", help="configuration for solving", type=str)
parser.add_argument("-g", "--GTYPE", default="f", help="true for ringmore graph, false for random graph", type=str)
parser.add_argument("-t", "--TIMEOUT", default=120, help="timeout in hours", type=int)
parser.add_argument("-r", "--THRESHOLD", default=5, help="threshold for SVAR", type=int)
parser.add_argument("-s", "--SCC", default="t", help="true to use SCC structure, false to not", type=str)
parser.add_argument("-m", "--SCCMEMBERS", default="t", help="true for using g_estimate SCC members, false for using "
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
noise_svar = 0.1

drop_bd_normed_errors_comm = []
drop_bd_normed_errors_omm = []
dir_errors_omm = []
dir_errors_comm = []
opt_dir_errors_omm = []
opt_dir_errors_comm = []
g_dir_errors_omm = []
g_dir_errors_comm = []
Gu_opt_dir_errors_omm = []
Gu_opt_dir_errors_comm = []
error_normalization = True

def round_tuple_elements(input_tuple, decimal_points=3):
    return tuple(round(elem, decimal_points) if isinstance(elem, (int, float)) else elem for elem in input_tuple)


def partition_distance(G1, G2):
    # Get strongly connected components of the graphs
    scc1 = list(nx.strongly_connected_components(G1))
    scc2 = list(nx.strongly_connected_components(G2))

    # Calculate variation of information
    vi = 0
    for s1 in scc1:
        for s2 in scc2:
            intersection = len(s1.intersection(s2))
            if intersection > 0:
                vi += intersection * (log(intersection) - log(len(s1)) - log(len(s2)))

    # Multiply by 2 because we considered each pair twice
    vi *= 2

    # Normalize by log of the total number of nodes (assuming the graphs have the same nodes)
    vi /= log(len(G1.nodes))

    # Return partition distance as the variation of information
    return vi


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


def makeConnected(G):
    # Get weakly connected components of the graph
    weakly_connected_components = list(nx.weakly_connected_components(G))

    # Connect weakly connected components
    for i in range(len(weakly_connected_components) - 1):
        # Add edge between first nodes of consecutive components
        G.add_edge(list(weakly_connected_components[i])[0], list(weakly_connected_components[i + 1])[0])

    return G


def rmBidirected(gu):
    g = copy.deepcopy(gu)
    for v in g:
        for w in list(g[v]):
            if g[v][w] == 2:
                del g[v][w]
            elif g[v][w] == 3:
                g[v][w] = 1
    return g


def transitionMatrix4(g, minstrength=0.1, distribution='normal', maxtries=1000):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)

    :param minstrength:
    :type minstrength: float

    :param distribution: (GUESS)distribution from which to sample the weights. Available
     options are flat, flatsigned, beta, normal, uniform
    :type distribution: string

    :param maxtries:
    :type maxtries: (guess)integer

    :returns:
    :rtype:
    """
    A = cv.graph2adj(g)
    edges = np.where(A == 1)
    s = 2.0
    c = 0
    pbar = ProgressBar(widgets=['Searching for weights: ',
                                Percentage(), ' '],
                       maxval=maxtries).start()
    while s > 1.0:
        minstrength -= 0.001
        A = lm.initRandomMatrix(A, edges, distribution=distribution)
        x = A[edges]
        delta = minstrength / np.min(np.abs(x))
        A[edges] = delta * x
        l = lm.linalg.eig(A)[0]
        s = np.max(np.real(l * scipy.conj(l)))
        c += 1
        if c > maxtries:
            return None
        pbar.update(c)
    pbar.finish()

    return A


# def genData(A, rate=2, burnin=100, ssize=2000, noise=0.1, dist='beta'):
def genData(A, rate=2, burnin=100, ssize=5000, noise=0.1, dist='normal'):
    """
    Given a number of nodes this function randomly generates a ring
    SCC and the corresponding stable transition matrix. It tries until
    succeeds and for some graph densities and parameters of the
    distribution of transition matrix values it may take
    forever. Please play with the dist parameter to stableVAR. Then
    using this transition matrix it generates `ssize` samples of data
    and undersamples them by `rate` discarding the `burnin` number of
    samples at the beginning.

    :param n: number of nodes in the desired graph
    :type n: (guess)integer

    :param rate: undersampling rate (1 - no undersampling)
    :type rate: integer

    :param density: density of the graph to be generted
    :type density: (guess) float

    :param burnin: number of samples to discard since the beginning of VAR sampling
    :type burnin: integer

    :param ssize: how many samples to keep at the causal sampling rate
    :type ssize: (guess)integer

    :param noise: noise standard deviation for the VAR model
    :type noise: (guess)float

    :param dist: (GUESS)distribution from which to sample the weights. Available
     options are flat, flatsigned, beta, normal, uniform
    :type dist: (guess)string

    :returns:
    :rtype:
    """
    Agt = A
    data = drawsamplesLG(Agt, samples=burnin + (ssize * rate), nstd=noise)
    data = data[:, burnin:]
    return data[:, ::rate]


def drawsamplesLG(A, nstd=0.1, samples=100):
    """
    :param A:
    :type A:

    :param nstd:
    :type nstd: float

    :param samples:
    :type samples: integer

    :returns:
    :rtype:
    """
    n = A.shape[0]
    data = np.zeros([n, samples])
    data[:, 0] = nstd * np.random.randn(A.shape[0])
    for i in range(1, samples):
        data[:, i] = A @ data[:, i - 1] + nstd * np.random.randn(A.shape[0])
    return data


def drawsamplesMA(A, nstd=0.1, samples=100, order=5):
    """
    :param A:
    :type A:

    :param nstd:
    :type nstd: float

    :param samples:
    :type samples: integer

    :param order:
    :type order: integer

    :returns:
    :rtype:
    """
    n = A.shape[0]
    data = scipy.zeros([n, samples])
    data[:, 0] = nstd * scipy.random.randn(A.shape[0])
    for i in range(1, samples):
        if i > order:
            result = 0
            for j in range(order):
                result += np.dot(1 / (j + 1) * A, data[:, i - 1 - j]) \
                          + nstd * np.dot(1 / (j + 1) * A, scipy.random.randn(A.shape[0]))
            data[:, i] = result
        else:
            data[:, i] = scipy.dot(A, data[:, i - 1]) \
                         + nstd * scipy.random.randn(A.shape[0])
    return data


def AB2intAB_1(A, B, th=0.09):
    """
    :param A:
    :type A:

    :param B:
    :type B:

    :param th: (GUESS)threshold for discarding edges in A and B
    :type th: float

    :returns:
    :rtype:
    """

    A[amap(lambda x: abs(x) > th, A)] = 1
    A[amap(lambda x: abs(x) < 1, A)] = 0
    B[amap(lambda x: abs(x) > th, B)] = 1
    B[amap(lambda x: np.abs(x) < 1, B)] = 0
    np.fill_diagonal(B, 0)
    return A, B


def amap(f, a):
    """
    :param f:
    :type f:

    :param a:
    :type a:

    :returns:
    :rtype:
    """
    v = np.vectorize(f)
    return v(a)


print('_____________________________________________')



G_test_set = \
    [{1: {3: 1, 7: 1, 8: 1},3: {},7: {6: 1}, 2: {7: 1}, 4: {1: 1, 2: 1}, 5: {5: 1, 8: 1}, 8: {}, 6: {}},
              {1: {2: 1}, 2: {6: 1}, 3: {2: 1}, 4: {}, 5: {3: 1}, 6: {}, 7: {4: 1}, 8: {3: 1, 5: 1, 7: 1}},
              {1: {4: 1, 7: 1, 2: 1}, 4: {}, 7: {}, 2: {3: 1}, 3: {8: 1}, 5: {2: 1}, 6: {1: 1, 4: 1}, 8: {}},
              {1: {3: 1, 4: 1}, 3: {5: 1}, 2: {}, 5: {2: 1}, 4: {6: 1}, 6: {7: 1}, 7: {}, 8: {2: 1}},
              {1: {2: 1}, 2: {4: 1}, 3: {6: 1, 8: 1}, 6: {6: 1, 7: 1}, 8: {3: 1, 8: 1}, 4: {}, 5: {4: 1}, 7: {1: 1, 7: 1}},
              {1: {4: 1}, 2: {5: 1, 8: 1}, 3: {2: 1, 7: 1}, 4: {5: 1}, 5: {}, 6: {1: 1}, 7: {3: 1, 7: 1}, 8: {5: 1}},
              {1: {4: 1, 7: 1}, 2: {3: 1}, 3: {4: 1}, 4: {}, 5: {5: 1},6: {}, 7: {1: 1, 5: 1, 6: 1}, 8: {1: 1, 5: 1}}]

# mask = [cv.graph2adj(GT) for GT in G_test_set]
#
#
# G = [np.clip(np.random.randn(*maski.shape) * 0.2 + 0.5, 0.3, 0.7) for maski in mask]
#
# Con_mat = [Gi * maski for Gi, maski in zip(G, mask)]
#
# for i in range(len(Con_mat)):
#     w, v = la.eig(Con_mat[i])
#     res = all(ele <= 1 for ele in abs(w))
#
#     while not res:
#         G = np.clip(np.random.randn(*mask[i].shape) * 0.2 + 0.5, 0.3, 0.7)
#         Con_mat[i] = G * mask
#         w, v = la.eig(Con_mat[i])
#         res = all(ele <= 1 for ele in abs(w))
#
# '''SVAR'''
# dd = [genData(Con_mati, rate=u_rate, ssize=2000, noise=noise_svar)  for Con_mati in Con_mat]
#
#
# esimations = [(lm.data2graph(ddi, th=EDGE_CUTOFF * k_threshold)) for ddi in dd]
# DDs = []
# BDs = []
# for element in esimations:
#     DD = (np.abs(cv.graph2adj(element[0]) * element[1]) * 10000).astype(int)
#     DD[np.where(DD == 0)] = DD.max()
#     DDs.append(DD)
#     BD = (np.abs(cv.graph2badj(element[0]) * element[2]) * 10000).astype(int)
#     BD[np.where(BD == 0)] = BD.max()
#     BDs.append(BD)
# g_estimateds = [item[0] for item in esimations]
#
# test_set = {'g_estimateds':g_estimateds,
#             'DDs':DDs,
#             'BDs':BDs}

dataset = zkl.load('datasets/test_set.zkl')
GT = G_test_set[args.BATCH-1]
GT_at_actual_U = bfutils.undersample(GT, u_rate)
g_estimated = dataset['g_estimateds'][args.BATCH-1]
DD = dataset['DDs'][args.BATCH-1]
BD = dataset['BDs'][args.BATCH-1]

'''task optimization'''
if SCC_members:
    members = nx.strongly_connected_components(gk.graph2nx(g_estimated))
else:
    members = nx.strongly_connected_components(gk.graph2nx(GT_at_actual_U))
startTime = int(round(time.time() * 1000))
# if Using_SVAR:
r_estimated = drasl([g_estimated], weighted=True, capsize=0, timeout=TIMEOUT,
                    urate=min(args.MAXU, (3 * len(g_estimated) + 1)),
                    dm=[DD],
                    bdm=[BD],
                    scc=SCC,
                    scc_members=members,
                    edge_weights=(1, 1),
                    pnum=args.PNUM,
                    configuration=args.CONFIG)

endTime = int(round(time.time() * 1000))
sat_time = endTime - startTime
'''G1_opt - the solution of optimization problem (r_estimated from g_estimated) in causal time scale'''
G1_opt = bfutils.num2CG(r_estimated[0][0], len(g_estimated))

'''Gu_opt - the solution of optimization problem (r_estimated from g_estimated) in measured time scale'''
Gu_opt = bfutils.undersample(G1_opt, r_estimated[0][1][0])
'''network_GT_U - the GT  in measured time scale'''
network_GT_U = bfutils.undersample(GT, r_estimated[0][1][0])

Gu_opt_errors_network_GT_U = gk.OCE(Gu_opt, network_GT_U, undirected=False, normalized=error_normalization)['total']

Gu_opt_errors_g_estimated = gk.OCE(Gu_opt, g_estimated, undirected=False, normalized=error_normalization)['total']

G1_opt_error_GT = gk.OCE(G1_opt, GT, undirected=False, normalized=error_normalization)['total']

print('batch: ' + str(args.BATCH))
print('configuration: ' + args.CONFIG)
print('U rate found to be:' + str(r_estimated[0][1][0]))
print('Gu_opt_errors_network_GT_U = ', round_tuple_elements(Gu_opt_errors_network_GT_U))
print('Gu_opt_errors_g_estimated', round_tuple_elements(Gu_opt_errors_g_estimated))
print('G1_opt_error_GT', round_tuple_elements(G1_opt_error_GT))

'''task optimization then sRASL to find min error'''
print('*******************************************')

if SCC_members:
    members2 = nx.strongly_connected_components(gk.graph2nx(G1_opt))
else:
    members2 = nx.strongly_connected_components(gk.graph2nx(GT))
startTime2 = int(round(time.time() * 1000))
c = drasl(glist=[Gu_opt], capsize=args.CAPSIZE, weighted=False, urate=min(args.MAXU, (3 * len(g_estimated) + 1)),
          timeout=TIMEOUT, scc=SCC, scc_members=members2)
endTime2 = int(round(time.time() * 1000))
sat_time2 = endTime2 - startTime2
min_err = {'total': (0, 0)}
min_norm_err = {'total': (0, 0)}
min_val = 1000000
min_error_graph = {}
print('cap size is', len(c))
for answer in c:
    curr_errors = gk.OCE(bfutils.num2CG(answer[0], len(g_estimated)), GT, undirected=False, normalized=False)
    '''compute the normalized/relative error and save it'''
    curr_normed_errors = gk.OCE(bfutils.num2CG(answer[0], len(g_estimated)), GT,
                                normalized=error_normalization,
                                undirected=False)
    if 0.5 * (curr_errors['total'][0] + curr_errors['total'][1]) < min_val:
        min_err = curr_errors
        min_norm_err = curr_normed_errors
        min_val = 0.5 * (curr_errors['total'][0] + curr_errors['total'][1])
        min_error_graph = answer

print('min G1_err_GT after opt + sRASL = ', round_tuple_elements(min_norm_err['total']))
print('took ' + str(round(((sat_time + sat_time2)/60000), 3)) + ' mins to solve')

'''saving results'''
F = 2 * (gk.density(GT) * len(GT) * len(GT) - min_norm_err['total'][0]) / (
        2 * gk.density(GT) * len(GT) * len(GT) - min_norm_err['total'][0] + min_norm_err['total'][1])
results = {'method': PreFix,
           'g_estimated': g_estimated,
           'dm': DD,
           'bdm': BD,
           'after_optim': G1_opt,
           'U_found': r_estimated[0][1][0],
           'optim_cost': r_estimated[1][0],
           'GU_optimized': Gu_opt,
           'network_GT_U': network_GT_U,
           'GT': GT,
           'eq_class': c,
           'threshold': EDGE_CUTOFF * k_threshold,
           'timeout': TIMEOUT,
           'graphType': graphType,
           'intended_u_rate': u_rate,
           'noise_svar': noise_svar,
           'Gu_opt_errors_network_GT_U': Gu_opt_errors_network_GT_U,
           'Gu_opt_errors_g_estimated': Gu_opt_errors_g_estimated,
           'G1_opt_error_GT': G1_opt_error_GT,
           'min_error_graph': min_error_graph,
           'min_norm_err': min_norm_err,
           'num_edges': gk.density(GT) * len(GT) * len(GT),
           'F_score': F,
           'total_time': round(((sat_time + sat_time2)/60000), 3),
           'config':args.CONFIG}

'''saving files'''
filename = 'nodes_' + str(args.NODE) + '_density_' + str(DENSITY) + '_undersampling_' + str(args.UNDERSAMPLING) + \
           '_' + PreFix + '_dataset_' + POSTFIX + '_' + graphType + '_CAPSIZE_' + str(args.CAPSIZE) + '_batch_' + \
           str(args.BATCH) + '_pnum_' + str(args.PNUM) + '_timeout_' + str(args.TIMEOUT) + '_threshold_' + \
           str(args.THRESHOLD) + '_maxu_' + str(args.MAXU) + '_sccMember_' + str(SCC_members) + '_SCC_' + str(SCC) + \
           "_config_" + str(args.CONFIG)
folder = 'res_config_test'
if not os.path.exists(folder):
    os.makedirs(folder)
zkl.save(results, folder + '/' + filename + '.zkl')
print('_____________________________________________')
