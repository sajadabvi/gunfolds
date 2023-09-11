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
if graphType == 'ringmore':
    e = bfutils.dens2edgenum(DENSITY, n=args.NODE)
    GT = gk.ringmore(args.NODE, e)
    mask = cv.graph2adj(GT)
else:
    deg = (((args.NODE ** 2) + args.NODE) * DENSITY) / args.NODE
    GT = gk.bp_mean_degree_graph(args.NODE, deg)
    GG = GG = gk.graph2nx(GT)
    if not nx.is_weakly_connected(GG):
        GGC = makeConnected(GG)
        GT = gk.nx2graph(GGC)
    mask = cv.graph2adj(GT)
    print('density {0:} in {1:} nodes is average degree {2:}'.format(DENSITY, args.NODE, deg))

G = np.clip(np.random.randn(*mask.shape) * 0.2 + 0.5, 0.3, 0.7)
Con_mat = G * mask

w, v = la.eig(Con_mat)
res = all(ele <= 1 for ele in abs(w))

while not res:
    G = np.clip(np.random.randn(*mask.shape) * 0.2 + 0.5, 0.3, 0.7)
    Con_mat = G * mask
    w, v = la.eig(Con_mat)
    res = all(ele <= 1 for ele in abs(w))

'''SVAR'''
dd = genData(Con_mat, rate=u_rate, ssize=2000, noise=noise_svar)  # data.values

# if Using_SVAR:

g_estimated, A, B = lm.data2graph(dd, th=EDGE_CUTOFF * k_threshold)
DD = (np.abs(cv.graph2adj(g_estimated) * A) * 10000).astype(int)
DD[np.where(DD == 0)] = DD.max()
BD = (np.abs(cv.graph2badj(g_estimated) * B) * 10000).astype(int)
BD[np.where(BD == 0)] = BD.max()
# else:
#     g_estimated = gc.gc(dd.T, pval=0.005)

GT_at_actual_U = bfutils.undersample(GT, u_rate)
# g = {1: {2: 1}, 2: {3: 1}, 3: {4: 1, 1: 1}, 4: {5: 1}, 5: {6: 1, 1: 1}, 6: {1: 1, 5: 1}, 7: {8: 1}, 8: {9: 1, 12: 1}, 9: {10: 1, 8: 1}, 10: {11: 1}, 11: {12: 1}, 12: {7: 1, 12: 1}, 13: {14: 1, 17: 1}, 14: {15: 1, 18: 1}, 15: {16: 1, 17: 1}, 16: {17: 1}, 17: {18: 1}, 18: {13: 1}, 19: {20: 1}, 20: {21: 1, 8: 1}, 21: {22: 1, 23: 1, 40: 1}, 22: {23: 1}, 23: {24: 1, 22: 1, 20: 1, 39: 1}, 24: {19: 1}, 25: {26: 1}, 26: {27: 1}, 27: {28: 1, 25: 1}, 28: {29: 1}, 29: {30: 1, 25: 1, 28: 1}, 30: {25: 1}, 31: {32: 1}, 32: {33: 1, 36: 1}, 33: {34: 1, 1: 1, 30: 1}, 34: {35: 1, 28: 1}, 35: {36: 1, 31: 1}, 36: {31: 1, 35: 1}, 37: {38: 1, 41: 1}, 38: {39: 1, 42: 1}, 39: {40: 1, 38: 1}, 40: {41: 1}, 41: {42: 1}, 42: {37: 1}, 43: {44: 1}, 44: {45: 1, 48: 1, 13: 1, 37: 1}, 45: {46: 1, 47: 1}, 46: {47: 1, 17: 1, 34: 1}, 47: {48: 1, 46: 1, 36: 1, 42: 1}, 48: {43: 1}, 49: {50: 1}, 50: {51: 1, 49: 1}, 51: {52: 1}, 52: {53: 1, 54: 1}, 53: {54: 1, 50: 1, 28: 1}, 54: {49: 1, 25: 1}}
# XX = gk.graph2nx(g)
# partition_distance(XX, XX)
jaccard_similarity = quantify_graph_difference(gk.graph2nx(g_estimated), gk.graph2nx(GT_at_actual_U))
g_estimated_errors_GT_at_actual_U = \
    gk.OCE(g_estimated, GT_at_actual_U, undirected=False, normalized=error_normalization)['total']

print("Gtype : {0:}, intended sampling rate : {1:} Num nodes  "
      ": {2:}, dens : {3:}\nBatch : {4:}\n"
      "g_estimated error with GT at intended U: {5:}\n"
      "using estimated SCC: {6:}".format(graphType, u_rate, args.NODE, DENSITY, args.BATCH,
                                         round_tuple_elements(g_estimated_errors_GT_at_actual_U), SCC_members))

print('jaccard similarity is: ' +str(jaccard_similarity))
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
                    edge_weights=(1, 1), pnum=args.PNUM)
# else:
#     r_estimated = drasl([g_estimated], weighted=True, capsize=0, timeout=TIMEOUT,
#                         urate=args.MAXU, edge_weights=(1, 1))
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
           'jaccard_similarity': jaccard_similarity,
           'g_estimated_errors_GT_at_actual_U': g_estimated_errors_GT_at_actual_U,
           'Gu_opt_errors_network_GT_U': Gu_opt_errors_network_GT_U,
           'Gu_opt_errors_g_estimated': Gu_opt_errors_g_estimated,
           'G1_opt_error_GT': G1_opt_error_GT,
           'min_error_graph': min_error_graph,
           'min_norm_err': min_norm_err,
           'num_edges': gk.density(GT) * len(GT) * len(GT),
           'F_score': F,
           'total_time': round(((sat_time + sat_time2)/60000), 3)}

'''saving files'''
filename = 'nodes_' + str(args.NODE) + '_density_' + str(DENSITY) + '_undersampling_' + str(args.UNDERSAMPLING) + \
           '_' + PreFix + '_' + POSTFIX + '_' + graphType + '_CAPSIZE_' + str(args.CAPSIZE) + '_batch_' + \
           str(args.BATCH) + '_pnum_' + str(args.PNUM) + '_timeout_' + str(args.TIMEOUT) + '_threshold_' + \
           str(args.THRESHOLD) + '_maxu_' + str(args.MAXU) + '_sccMember_' + str(SCC_members) + '_SCC_' + str(SCC)
folder = 'res_simulation'
if not os.path.exists(folder):
    os.makedirs(folder)
zkl.save(results, folder + '/' + filename + '.zkl')
print('_____________________________________________')
