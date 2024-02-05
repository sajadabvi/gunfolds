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
parser.add_argument("-b", "--BATCH", default=1, help="slurm batch.", type=int)
parser.add_argument("-p", "--PNUM", default=PNUM, help="number of CPUs in machine.", type=int)
parser.add_argument("-n", "--NODE", default=8, help="number of nodes in graph", type=int)
parser.add_argument("-d", "--DEN", default=0.14, help="density of graph", type=str)
parser.add_argument("-g", "--GTYPE", default="f", help="true for ringmore graph, false for random graph", type=str)
parser.add_argument("-t", "--TIMEOUT", default=120, help="timeout in hours", type=int)
parser.add_argument("-r", "--THRESHOLD", default=5, help="threshold for SVAR", type=int)
parser.add_argument("-l", "--MINLINK", default=2, help=" lower threshold transition matrix abs value x10", type=int)
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
    max_attempts=100000,
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
dataset = zkl.load('datasets/ringmore_n8d14.zkl')
GT = dataset[args.BATCH-1]
A = cv.graph2adj(GT)
W = create_stable_weighted_matrix(A, threshold=args.MINLINK/10, powers=[2, 3, 4])

# for i in range(1,10):
#     plt.subplot(3,3,i)
#     M = np.linalg.matrix_power(W,i)
#     plt.imshow(M,interpolation="none",cmap=cm.seismic)
#     plt.colorbar()
#     plt.axis('off')
#     plt.clim([-np.abs(M).max(),np.abs(M).max()])
#     plt.title('u='+str(i))
# #

'''SVAR'''
dd = genData(W, rate=u_rate, ssize=8000, noise=noise_svar)  # data.values


MAXCOST = 10000
g_estimated, A, B = lm.data2graph(dd, th=EDGE_CUTOFF * k_threshold)
DD = (np.abs((np.abs(A/np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1))*MAXCOST)).astype(int)
BD = (np.abs((np.abs(B/np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1))*MAXCOST)).astype(int)

GT_at_actual_U = bfutils.undersample(GT, u_rate)

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

r_estimated = drasl([g_estimated], weighted=True, capsize=0, timeout=TIMEOUT,
                    urate=min(args.MAXU, (3 * len(g_estimated) + 1)),
                    dm=[DD],
                    bdm=[BD],
                    scc=SCC,
                    scc_members=members,
                    GT_density=int(1000*gk.density(GT)),
                    edge_weights=(1, 1), pnum=args.PNUM, optim='optN')

endTime = int(round(time.time() * 1000))
sat_time = endTime - startTime

print('number of optimal solutions is', len(r_estimated))

### minimizing with respect to Gu_opt Vs. G_estimate
min_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
min_norm_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
min_val = 1000000
min_cost = 10000000
for answer in r_estimated:
    curr_errors = gk.OCE(bfutils.undersample(bfutils.num2CG(answer[0][0], len(GT)),answer[0][1][0]), g_estimated)
    curr_normed_errors = gk.OCE(bfutils.undersample(bfutils.num2CG(answer[0][0], len(GT)),answer[0][1][0]), g_estimated, normalized=True)
    curr_cost = answer[1]
    if  (curr_errors['total'][0] + curr_errors['total'][1]) < min_val:
        min_err = curr_errors
        min_norm_err = curr_normed_errors
        min_cost = curr_cost
        min_val =  (curr_errors['total'][0] + curr_errors['total'][1])
        min_answer_WRT_GuOptVsGest = answer
    elif (curr_errors['total'][0] + curr_errors['total'][1]) == min_val:
        if curr_cost < min_cost:
            min_err = curr_errors
            min_norm_err = curr_normed_errors
            min_cost = curr_cost
            min_val = (curr_errors['total'][0] + curr_errors['total'][1])
            min_answer_WRT_GuOptVsGest = answer

'''G1_opt - the solution of optimization problem (r_estimated from g_estimated) in causal time scale'''
G1_opt_WRT_GuOptVsGest = bfutils.num2CG(min_answer_WRT_GuOptVsGest[0][0], len(g_estimated))

'''Gu_opt - the solution of optimization problem (r_estimated from g_estimated) in measured time scale'''
Gu_opt_WRT_GuOptVsGest = bfutils.undersample(G1_opt_WRT_GuOptVsGest, min_answer_WRT_GuOptVsGest[0][1][0])
'''network_GT_U - the GT  in measured time scale'''
network_GT_U_WRT_GuOptVsGest = bfutils.undersample(GT, min_answer_WRT_GuOptVsGest[0][1][0])

Gu_opt_errors_network_GT_U_WRT_GuOptVsGest = gk.OCE(Gu_opt_WRT_GuOptVsGest, network_GT_U_WRT_GuOptVsGest, undirected=False, normalized=error_normalization)['total']
Gu_opt_errors_g_estimated_WRT_GuOptVsGest = gk.OCE(Gu_opt_WRT_GuOptVsGest, g_estimated, undirected=False, normalized=error_normalization)['total']
G1_opt_error_GT_WRT_GuOptVsGest = gk.OCE(G1_opt_WRT_GuOptVsGest, GT, undirected=False, normalized=error_normalization)['total']
print('*******************************************')
print('results with respect to Gu_opt Vs. G_estimate ')
print('U rate found to be:' + str(min_answer_WRT_GuOptVsGest[0][1][0]))
print('Gu_opt_errors_network_GT_U = ', round_tuple_elements(Gu_opt_errors_network_GT_U_WRT_GuOptVsGest))
print('Gu_opt_errors_g_estimated', round_tuple_elements(Gu_opt_errors_g_estimated_WRT_GuOptVsGest))
print('G1_opt_error_GT', round_tuple_elements(G1_opt_error_GT_WRT_GuOptVsGest))


### minimizing with respect to Gu_opt Vs. GTu
min_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
min_norm_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
min_val = 1000000
min_cost = 10000000
for answer in r_estimated:
    curr_errors = gk.OCE(bfutils.undersample(bfutils.num2CG(answer[0][0], len(GT)),answer[0][1][0]), bfutils.undersample(GT, answer[0][1][0]))
    curr_normed_errors = gk.OCE(bfutils.undersample(bfutils.num2CG(answer[0][0], len(GT)),answer[0][1][0]), bfutils.undersample(GT, answer[0][1][0]), normalized=True)
    curr_cost = answer[1]
    if  (curr_errors['total'][0] + curr_errors['total'][1]) < min_val:
        min_err = curr_errors
        min_norm_err = curr_normed_errors
        min_cost = curr_cost
        min_val =  (curr_errors['total'][0] + curr_errors['total'][1])
        min_answer_WRT_GuOptVsGTu = answer
    elif (curr_errors['total'][0] + curr_errors['total'][1]) == min_val:
        if curr_cost < min_cost:
            min_err = curr_errors
            min_norm_err = curr_normed_errors
            min_cost = curr_cost
            min_val = (curr_errors['total'][0] + curr_errors['total'][1])
            min_answer_WRT_GuOptVsGTu = answer

'''G1_opt - the solution of optimization problem (r_estimated from g_estimated) in causal time scale'''
G1_opt_WRT_GuOptVsGTu = bfutils.num2CG(min_answer_WRT_GuOptVsGTu[0][0], len(g_estimated))

'''Gu_opt - the solution of optimization problem (r_estimated from g_estimated) in measured time scale'''
Gu_opt_WRT_GuOptVsGTu = bfutils.undersample(G1_opt_WRT_GuOptVsGTu, min_answer_WRT_GuOptVsGTu[0][1][0])
'''network_GT_U - the GT  in measured time scale'''
network_GT_U_WRT_GuOptVsGTu = bfutils.undersample(GT, min_answer_WRT_GuOptVsGTu[0][1][0])

Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu = gk.OCE(Gu_opt_WRT_GuOptVsGTu, network_GT_U_WRT_GuOptVsGTu, undirected=False, normalized=error_normalization)['total']
Gu_opt_errors_g_estimated_WRT_GuOptVsGTu = gk.OCE(Gu_opt_WRT_GuOptVsGTu, g_estimated, undirected=False, normalized=error_normalization)['total']
G1_opt_error_GT_WRT_GuOptVsGTu = gk.OCE(G1_opt_WRT_GuOptVsGTu, GT, undirected=False, normalized=error_normalization)['total']
print('*******************************************')
print('results of minimizing with respect to Gu_opt Vs. GTu')
print('U rate found to be:' + str(min_answer_WRT_GuOptVsGTu[0][1][0]))
print('Gu_opt_errors_network_GT_U = ', round_tuple_elements(Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu))
print('Gu_opt_errors_g_estimated', round_tuple_elements(Gu_opt_errors_g_estimated_WRT_GuOptVsGTu))
print('G1_opt_error_GT', round_tuple_elements(G1_opt_error_GT_WRT_GuOptVsGTu))

### minimizing with respect to G1_opt Vs. GT
min_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
min_norm_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
min_val = 1000000
min_cost = 10000000
for answer in r_estimated:
    curr_errors = gk.OCE(bfutils.num2CG(answer[0][0], len(GT)),GT)
    curr_normed_errors = gk.OCE(bfutils.num2CG(answer[0][0], len(GT)), GT, normalized=True)
    curr_cost = answer[1]
    if (curr_errors['total'][0] + curr_errors['total'][1]) < min_val:
        min_err = curr_errors
        min_norm_err = curr_normed_errors
        min_cost = curr_cost
        min_val = (curr_errors['total'][0] + curr_errors['total'][1])
        min_answer_WRT_G1OptVsGT = answer
    elif (curr_errors['total'][0] + curr_errors['total'][1]) == min_val:
        if curr_cost < min_cost:
            min_err = curr_errors
            min_norm_err = curr_normed_errors
            min_cost = curr_cost
            min_val = (curr_errors['total'][0] + curr_errors['total'][1])
            min_answer_WRT_G1OptVsGT = answer

'''G1_opt - the solution of optimization problem (r_estimated from g_estimated) in causal time scale'''
G1_opt_WRT_G1OptVsGT = bfutils.num2CG(min_answer_WRT_G1OptVsGT[0][0], len(g_estimated))

'''Gu_opt - the solution of optimization problem (r_estimated from g_estimated) in measured time scale'''
Gu_opt_WRT_G1OptVsGT = bfutils.undersample(G1_opt_WRT_G1OptVsGT, min_answer_WRT_G1OptVsGT[0][1][0])
'''network_GT_U - the GT  in measured time scale'''
network_GT_U_WRT_G1OptVsGT = bfutils.undersample(GT, min_answer_WRT_G1OptVsGT[0][1][0])

Gu_opt_errors_network_GT_U_WRT_G1OptVsGT = gk.OCE(Gu_opt_WRT_G1OptVsGT, network_GT_U_WRT_G1OptVsGT, undirected=False, normalized=error_normalization)['total']
Gu_opt_errors_g_estimated_WRT_G1OptVsGT = gk.OCE(Gu_opt_WRT_G1OptVsGT, g_estimated, undirected=False, normalized=error_normalization)['total']
G1_opt_error_GT_WRT_G1OptVsGT = gk.OCE(G1_opt_WRT_G1OptVsGT, GT, undirected=False, normalized=error_normalization)['total']
print('*******************************************')
print('results of minimizing with respect to G1_opt Vs. GT')
print('U rate found to be:' + str(min_answer_WRT_G1OptVsGT[0][1][0]))
print('Gu_opt_errors_network_GT_U = ', round_tuple_elements(Gu_opt_errors_network_GT_U_WRT_G1OptVsGT))
print('Gu_opt_errors_g_estimated', round_tuple_elements(Gu_opt_errors_g_estimated_WRT_G1OptVsGT))
print('G1_opt_error_GT', round_tuple_elements(G1_opt_error_GT_WRT_G1OptVsGT))


'''saving results'''
sorted_data = sorted(r_estimated, key=lambda x: x[1], reverse=True)
F = 2 * (gk.density(GT) * len(GT) * len(GT) - min_norm_err['total'][0]) / (
        2 * gk.density(GT) * len(GT) * len(GT) - min_norm_err['total'][0] + min_norm_err['total'][1])
results = {'general':{'method': PreFix,
           'g_estimated': g_estimated,
           'W': W,
           'dm': DD,
           'bdm': BD,
           'optim_cost': sorted_data[-1][1],
           'num_sols': len(r_estimated),
           'GT': GT,
           'GT_at_actual_U': GT_at_actual_U,
           'threshold': EDGE_CUTOFF * k_threshold,
           'timeout': TIMEOUT,
           'graphType': graphType,
           'intended_u_rate': u_rate,
           'noise_svar': noise_svar,
           'full_sols':r_estimated,
           'jaccard_similarity': jaccard_similarity,
           'g_estimated_errors_GT_at_actual_U': g_estimated_errors_GT_at_actual_U,
           'num_edges': gk.density(GT) * len(GT) * len(GT),
           'F_score': F,
           'total_time': round(((sat_time)/60000), 3)},
           'GuOptVsGest':{
               'min_answer_WRT_GuOptVsGest':min_answer_WRT_GuOptVsGest,
               'G1_opt_WRT_GuOptVsGest':G1_opt_WRT_GuOptVsGest,
               'Gu_opt_WRT_GuOptVsGest':Gu_opt_WRT_GuOptVsGest,
               'network_GT_U_WRT_GuOptVsGest':network_GT_U_WRT_GuOptVsGest,
               'Gu_opt_errors_network_GT_U_WRT_GuOptVsGest':Gu_opt_errors_network_GT_U_WRT_GuOptVsGest,
               'Gu_opt_errors_g_estimated_WRT_GuOptVsGest':Gu_opt_errors_g_estimated_WRT_GuOptVsGest,
               'G1_opt_error_GT_WRT_GuOptVsGest':G1_opt_error_GT_WRT_GuOptVsGest
           },
           'GuOptVsGTu':{
               'min_answer_WRT_GuOptVsGTu':min_answer_WRT_GuOptVsGTu,
               'G1_opt_WRT_GuOptVsGTu':G1_opt_WRT_GuOptVsGTu,
               'Gu_opt_WRT_GuOptVsGTu':Gu_opt_WRT_GuOptVsGTu,
               'network_GT_U_WRT_GuOptVsGTu':network_GT_U_WRT_GuOptVsGTu,
               'Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu':Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu,
               'Gu_opt_errors_g_estimated_WRT_GuOptVsGTu':Gu_opt_errors_g_estimated_WRT_GuOptVsGTu,
               'G1_opt_error_GT_WRT_GuOptVsGTu':G1_opt_error_GT_WRT_GuOptVsGTu
           },
           'G1OptVsGT':{
               'min_answer_WRT_G1OptVsGT':min_answer_WRT_G1OptVsGT,
               'G1_opt_WRT_G1OptVsGT':G1_opt_WRT_G1OptVsGT,
               'Gu_opt_WRT_G1OptVsGT':Gu_opt_WRT_G1OptVsGT,
               'network_GT_U_WRT_G1OptVsGT':network_GT_U_WRT_G1OptVsGT,
               'Gu_opt_errors_network_GT_U_WRT_G1OptVsGT':Gu_opt_errors_network_GT_U_WRT_G1OptVsGT,
               'Gu_opt_errors_g_estimated_WRT_G1OptVsGT':Gu_opt_errors_g_estimated_WRT_G1OptVsGT,
               'G1_opt_error_GT_WRT_G1OptVsGT':G1_opt_error_GT_WRT_G1OptVsGT
           }}

'''saving files'''
filename = 'full_sols_nodes_' + str(args.NODE) + '_density_' + str(DENSITY) + '_undersampling_' + str(args.UNDERSAMPLING) + \
           '_' + PreFix + '_optN_gt_den_priority2_dataset_' + POSTFIX + '_' + graphType + '_CAPSIZE_' + str(args.CAPSIZE) + '_batch_' + \
           str(args.BATCH) + '_pnum_' + str(args.PNUM) + '_timeout_' + str(args.TIMEOUT) + '_threshold_' + \
           str(args.THRESHOLD) + '_noise_' + str(args.NOISE)  + '_MINLINK_' +str(args.MINLINK) + \
           '_maxu_' + str(args.MAXU) + '_sccMember_' + str(SCC_members) + '_SCC_' + str(SCC)
folder = 'res_simulation'
if not os.path.exists(folder):
    os.makedirs(folder)
zkl.save(results, folder + '/' + filename + '.zkl')
print('_____________________________________________')
