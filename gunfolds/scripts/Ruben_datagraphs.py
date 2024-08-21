from gunfolds.viz import gtool as gt
import pickle
import distutils.util
import copy
from gunfolds.utils import bfutils
from gunfolds.utils import zickle as zkl
from gunfolds.estimation import grangercausality as gc
import numpy as np
import time, socket
import scipy
from gunfolds.solvers.clingo_rasl import drasl
import networkx as nx
import argparse
from gunfolds.utils import graphkit as gk
from gunfolds.utils.calc_procs import get_process_count
import pandas as pd
from gunfolds.estimation import linear_model as lm
from gunfolds.viz.dbn2latex import output_graph_figure
from gunfolds import conversions as cv
import matplotlib.pyplot as plt
from datetime import datetime

CLINGO_LIMIT = 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))
POSTFIX = 'Ruben_data_concat'
Using_SVAR = True
PreFix = 'SVAR' if Using_SVAR else 'GC'
parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-c", "--CAPSIZE", default=10000,
                    help="stop traversing after growing equivalence class to this size.", type=int)
parser.add_argument("-b", "--BATCH", default=10, help="slurm batch.", type=int)
parser.add_argument("-n", "--NUMBER", default=1, help="simple network index.", type=int)
parser.add_argument("-p", "--PNUM", default=PNUM, help="number of CPUs in machine.", type=int)
parser.add_argument("-t", "--TIMEOUT", default=12, help="timeout in hours", type=int)
parser.add_argument("-x", "--MAXU", default=15, help="maximum number of undersampling to look for solution.", type=int)
parser.add_argument("-r", "--THRESHOLD", default=5, help="threshold for SVAR", type=int)
parser.add_argument("-s", "--SCC", default="f", help="true to use SCC structure, false to not", type=str)
parser.add_argument("-m", "--SCCMEMBERS", default="f", help="true for using g_estimate SCC members, false for using "
                                                            "GT SCC members", type=str)
parser.add_argument("-y", "--PRIORITY", default="42531", help="string of priorities", type=str)
args = parser.parse_args()
TIMEOUT = args.TIMEOUT * 60 * 60
fl = args.BATCH
k_threshold = args.THRESHOLD
EDGE_CUTOFF = 0.01
SCC = bool(distutils.util.strtobool(args.SCC))
SCC_members = bool(distutils.util.strtobool(args.SCCMEMBERS))
SCC = True if SCC_members else SCC
priprities = []
for char in args.PRIORITY:
    priprities.append(int(char))

SL_drop_bd_normed_errors_comm = []
SL_drop_bd_normed_errors_omm = []
drop_bd_normed_errors_comm = []
drop_bd_normed_errors_omm = []

SL_undir_normed_errors_omm = []
SL_undir_normed_errors_comm = []
undir_errors_omm = []
undir_errors_comm = []

opt_SL_undir_normed_errors_omm = []
opt_SL_undir_normed_errors_comm = []
opt_undir_errors_omm = []
opt_undir_errors_comm = []

def round_tuple_elements(input_tuple, decimal_points=3):
    return tuple(round(elem, decimal_points) if isinstance(elem, (int, float)) else elem for elem in input_tuple)
def rmBidirected(gu):
    g = copy.deepcopy(gu)
    for v in g:
        for w in list(g[v]):
            if g[v][w] == 2:
                del g[v][w]
            elif g[v][w] == 3:
                g[v][w] = 1
    return g

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

def Glag2CG(results):
    """Converts lag graph format to gunfolds graph format,
   and A and B matrices representing directed and bidirected edges weights.

   Args:
       results (dict): A dictionary containing:
           - 'graph': A 3D NumPy array of shape [N, N, 2] representing the graph structure.
           - 'val_matrix': A NumPy array of shape [N, N, 2] storing edge weights.

   Returns:
       tuple: (graph_dict, A_matrix, B_matrix)
   """

    graph_array = results['graph']
    bidirected_edges = np.where(graph_array == 'o-o', 1, 0).astype(int)
    directed_edges = np.where(graph_array == '-->', 1, 0).astype(int)

    graph_dict = cv.adjs2graph(directed_edges[:, :, 1], bidirected_edges[:, :, 0])
    A_matrix = results['val_matrix'][:, :, 1]
    B_matrix = results['val_matrix'][:, :, 0]

    return graph_dict, A_matrix, B_matrix

def precision_recall(graph1, graph2):
    # Convert both graphs to undirected
    graph1_undirected = graph1.to_undirected()
    graph2_undirected = graph2.to_undirected()

    # Get adjacency matrices
    adj_matrix1 = nx.adjacency_matrix(graph1_undirected).todense()
    adj_matrix2 = nx.adjacency_matrix(graph2_undirected).todense()

    # Calculate true positives (intersection of edges)
    true_positives = np.sum(np.logical_and(adj_matrix1, adj_matrix2))

    # Calculate false positives (edges in graph2 but not in graph1)
    false_positives = np.sum(np.logical_and(adj_matrix2, np.logical_not(adj_matrix1)))

    # Calculate false negatives (edges in graph1 but not in graph2)
    false_negatives = np.sum(np.logical_and(adj_matrix1, np.logical_not(adj_matrix2)))

    # Calculate precision
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0

    # Calculate recall
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

    return precision, recall


for nn in [2]:
    save_results = {}
    args.NUMBER = nn
    SL_undir_normed_errors_omm = []
    SL_undir_normed_errors_comm = []
    undir_errors_omm = []
    undir_errors_comm = []
    opt_SL_undir_normed_errors_omm = []
    opt_SL_undir_normed_errors_comm = []
    opt_undir_errors_omm = []
    opt_undir_errors_comm = []
    save_results = []
    Precision_O = []
    Recall_O = []
    Precision_A = []
    Recall_A = []
    Precision_C = []
    Recall_C = []
    for fl in range(1, 61):
        num = str(fl) if fl > 9 else '0' + str(fl)
        print('reading file:' + num)
        data = pd.read_csv(
            '~/DataSets_Feedbacks/1. Simple_Networks/Network' + str(
                args.NUMBER) + '_amp/data_fslfilter/BOLDfslfilter_{0}.txt'.format(
                num), delimiter='\t')

        selfloops = []
        no_selfloops = []

        network1_GT_selfloop = {1: {1: 1, 2: 1, 5: 1}, 2: {2: 1, 3: 1, 1: 1}, 3: {3: 1, 4: 1}, 4: {4: 1, 5: 1},
                                5: {5: 1}}
        network1_GT = {1: {2: 1, 5: 1}, 2: {3: 1, 1: 1}, 3: {4: 1}, 4: {5: 1}, 5: {}}
        selfloops.append(network1_GT_selfloop)
        no_selfloops.append(network1_GT)

        network2_GT_selfloop = {1: {1: 1, 2: 1, 5: 1}, 2: {2: 1, 3: 1, 1: 1}, 3: {2: 1, 3: 1, 4: 1}, 4: {4: 1, 5: 1},
                                5: {5: 1}}
        network2_GT = {1: {2: 1, 5: 1}, 2: {3: 1, 1: 1}, 3: {2: 1, 4: 1}, 4: {5: 1}, 5: {}}
        selfloops.append(network2_GT_selfloop)
        no_selfloops.append(network2_GT)

        network3_GT_selfloop = {1: {1: 1, 2: 1, 5: 1}, 2: {2: 1, 3: 1, 1: 1}, 3: {3: 1, 4: 1}, 4: {3: 1, 4: 1, 5: 1},
                                5: {5: 1}}
        network3_GT = {1: {2: 1, 5: 1}, 2: {3: 1, 1: 1}, 3: {4: 1}, 4: {3: 1, 5: 1}, 5: {}}
        selfloops.append(network3_GT_selfloop)
        no_selfloops.append(network3_GT)

        network4_GT_selfloop = {1: {4: 1, 8: 1, 6: 1, 1: 1}, 2: {2: 1, 3: 1}, 3: {2: 1, 3: 1},
                                4: {4: 1, 2: 1, 7: 1, 9: 1, 5: 1}, 5: {5: 1, 4: 1, 6: 1},
                                6: {6: 1, 10: 1}, 7: {7: 1, 3: 1, 10: 1}, 8: {8: 1, 2: 1, 9: 1}, 9: {9: 1, 8: 1, 6: 1},
                                10: {10: 1, 6: 1}}
        network4_GT = {1: {4: 1, 8: 1, 6: 1}, 2: {3: 1}, 3: {2: 1}, 4: {2: 1, 7: 1, 9: 1, 5: 1}, 5: {4: 1, 6: 1},
                       6: {10: 1}, 7: {3: 1, 10: 1}, 8: {2: 1, 9: 1}, 9: {8: 1, 6: 1}, 10: {6: 1}}
        selfloops.append(network4_GT_selfloop)
        no_selfloops.append(network4_GT)

        network5_GT_selfloop = {1: {1: 1, 3: 1}, 2: {2: 1, 4: 1}, 3: {3: 1, 4: 1, 5: 1}, 4: {4: 1, 3: 1}, 5: {5: 1}}
        network5_GT = {1: {3: 1}, 2: {4: 1}, 3: {4: 1, 5: 1}, 4: {3: 1}, 5: {}}
        selfloops.append(network5_GT_selfloop)
        no_selfloops.append(network5_GT)

        network6_GT_selfloop = {1: {1: 1, 3: 1}, 2: {2: 1, 3: 1}, 3: {3: 1, 4: 1}, 4: {4: 1, 3: 1, 5: 1},
                                5: {5: 1, 7: 1, 8: 1, 6: 1},
                                6: {6: 1}, 7: {7: 1}, 8: {8: 1}}
        network6_GT = {1: {3: 1}, 2: {3: 1}, 3: {4: 1}, 4: {3: 1, 5: 1}, 5: {7: 1, 8: 1, 6: 1}, 6: {}, 7: {}, 8: {}}
        selfloops.append(network6_GT_selfloop)
        no_selfloops.append(network6_GT)

        network7_GT_selfloop = {1: {1: 1, 2: 1}, 2: {2: 1, 3: 1}, 3: {3: 1, 4: 1}, 4: {4: 1, 5: 1},
                                5: {5: 1, 2: 1, 6: 1},
                                6: {6: 1}}
        network7_GT = {1: {2: 1}, 2: {3: 1}, 3: {4: 1}, 4: {5: 1}, 5: {2: 1, 6: 1}, 6: {}}
        selfloops.append(network7_GT_selfloop)
        no_selfloops.append(network7_GT)

        network8_GT_selfloop = {1: {1: 1, 2: 1}, 2: {2: 1, 3: 1}, 3: {3: 1, 4: 1, 8: 1}, 4: {4: 1, 5: 1, 6: 1},
                                5: {5: 1, 2: 1}, 6: {6: 1, 7: 1}, 7: {7: 1, 5: 1}, 8: {8: 1}}
        network8_GT = {1: {2: 1}, 2: {3: 1}, 3: {4: 1, 8: 1}, 4: {5: 1, 6: 1}, 5: {5: 1, 2: 1}, 6: {7: 1},
                       7: {5: 1}, 8: {}}
        selfloops.append(network8_GT_selfloop)
        no_selfloops.append(network8_GT)

        network9_GT_selfloop = {1: {1: 1, 2: 1}, 2: {2: 1, 3: 1}, 3: {3: 1, 4: 1}, 4: {4: 1, 5: 1}, 5: {5: 1, 2: 1},
                                6: {6: 1, 7: 1, 9: 1}, 7: {7: 1, 8: 1}, 8: {8: 1, 4: 1}, 9: {9: 1}}
        network9_GT = {1: {2: 1}, 2: {3: 1}, 3: {4: 1}, 4: {5: 1}, 5: {2: 1}, 6: {7: 1, 9: 1}, 7: {8: 1}, 8: {4: 1},
                       9: {}}
        selfloops.append(network9_GT_selfloop)
        no_selfloops.append(network9_GT)

        network_GT_selfloop = selfloops[args.NUMBER - 1]
        network_GT = no_selfloops[args.NUMBER - 1]
        '''SVAR'''
        dd = np.transpose(data.values)
        if Using_SVAR:
            g_estimated, A, B = lm.data2graph(dd, th=EDGE_CUTOFF * k_threshold)
        else:
            g_estimated = gc.gc(dd.T, pval=0.005)
        '''task 1'''  # Compare with Networks as is, dropping bidirected in H_*
        '''
        g_noBidirect = rmBidirected(g_gc)
        drop_bd_errors = gk.OCE(network_GT, g_noBidirect)
        drop_bd_normed_errors = gk.OCE(network_GT, g_noBidirect, normalized=True)

        SL_drop_bd_errors = gk.OCE(network_GT_selfloop, g_noBidirect)
        SL_drop_bd_normed_errors = gk.OCE(network_GT_selfloop, g_noBidirect, normalized=True)
        print(drop_bd_errors, drop_bd_normed_errors, SL_drop_bd_errors,SL_drop_bd_normed_errors)

        SL_drop_bd_normed_errors_omm.append(SL_drop_bd_normed_errors['total'][0])
        SL_drop_bd_normed_errors_comm.append(SL_drop_bd_normed_errors['total'][1])
        drop_bd_normed_errors_omm.append(drop_bd_normed_errors['total'][0])
        drop_bd_normed_errors_comm.append(drop_bd_normed_errors['total'][1])
        '''
        '''task 2'''  # Compare with Networks skeleton turning all including bidirected to just undirected edges in H_*
        '''

        undir_errors = gk.OCE(g_gc, network_GT, undirected=True, normalized=True)
        SL_undir_errors =gk.OCE(g_gc, network_GT_selfloop, undirected=True, normalized=True)

        SL_undir_normed_errors_omm.append(SL_undir_errors['undirected'][0])
        SL_undir_normed_errors_comm.append(SL_undir_errors['undirected'][1])
        undir_errors_omm.append(undir_errors['undirected'][0])
        undir_errors_comm.append(undir_errors['undirected'][1])
        '''
        '''task 3'''  # Do this after applying sRASL to H_*
        '''
        A, B = lm.data2AB(dd.T)
        DD = (np.abs(cv.graph2adj(g_gc) * A) * 100).astype(int)
        BD = (np.abs(cv.graph2badj(g_gc) * B) * 100).astype(int)
        startTime = int(round(time.time() * 1000))
        r_estimated = drasl([g_gc], weighted=True, dm=[DD], bdm=[BD], capsize=args.CAPSIZE, timeout=TIMEOUT, urate=args.MAXU)
        endTime = int(round(time.time() * 1000))
        G1_opt = bfutils.num2CG(r_estimated[0], len(g_gc))
        Gu_opt = bfutils.undersample(G1_opt, r_estimated[1][0])
        network_GT_U = bfutils.undersample(network_GT, r_estimated[1][0])
        network_GT_U_SL = bfutils.undersample(network_GT_selfloop, r_estimated[1][0])
        print('U rate found to be:' + str(r_estimated[1][0]))
        undir_error = gk.OCE(Gu_opt, network_GT_U, undirected=True, normalized=True)
        undir_error_SL = gk.OCE(Gu_opt, network_GT_U_SL, undirected=True, normalized=True)

        SL_undir_normed_errors_omm.append(undir_error_SL['undirected'][0])
        SL_undir_normed_errors_comm.append(undir_error_SL['undirected'][1])
        undir_errors_omm.append(undir_error['undirected'][0])
        undir_errors_comm.append(undir_error['undirected'][1])
        '''
        '''task 4'''  # Compute all undersampled versions of Networks
        # (version with and without self loops everywhere separately)
        # and compare each with H_* using both procedures above

        '''all_network = bfutils.all_undersamples(network_GT)
        all_network_SL = bfutils.all_undersamples(network_GT_selfloop)
        g_noBidirect = rmBidirected(g_estimated)
        min_error = 10
        for i in range(len(all_network)):
            curr_no_bi = rmBidirected(all_network[i])
            drop_bd_normed_errors = gk.OCE(g_noBidirect, curr_no_bi, normalized=True, undirected=False)
            if (drop_bd_normed_errors['total'][0] + drop_bd_normed_errors['total'][1]) < min_error:
                min_error = (drop_bd_normed_errors['total'][0] + drop_bd_normed_errors['total'][1])
                error_comp = drop_bd_normed_errors
                min_err_index = i
        undir_errors_omm.append(error_comp['total'][0])
        undir_errors_comm.append(error_comp['total'][1])

        min_error2 = 10
        for j in range(len(all_network_SL)):
            curr_no_bi = rmBidirected(all_network_SL[j])
            SL_drop_bd_normed_errors = gk.OCE(g_noBidirect, curr_no_bi, normalized=True, undirected=False)
            if (SL_drop_bd_normed_errors['total'][0] + SL_drop_bd_normed_errors['total'][1]) < min_error2:
                min_error2 = (SL_drop_bd_normed_errors['total'][0] + SL_drop_bd_normed_errors['total'][1])
                error_comp2 = SL_drop_bd_normed_errors
                min_err_index = j
        SL_undir_normed_errors_omm.append(error_comp2['total'][0])
        SL_undir_normed_errors_comm.append(error_comp2['total'][1])'''

        '''task 5''' # full sRASL optimization
        MAXCOST = 10000
        DD = (np.abs((np.abs(A / np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
        BD = (np.abs((np.abs(B / np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

        if SCC_members:
            members = nx.strongly_connected_components(gk.graph2nx(g_estimated))
        else:
            members = nx.strongly_connected_components(gk.graph2nx(network_GT_selfloop))
        startTime = int(round(time.time() * 1000))

        startTime = int(round(time.time() * 1000))
        r_estimated = drasl([g_estimated], weighted=True, capsize=0, timeout=TIMEOUT,
                            urate=min(args.MAXU, (3 * len(g_estimated) + 1)),
                            dm=[DD],
                            bdm=[BD],
                            scc=SCC,
                            scc_members=members,
                            GT_density=int(1000 * gk.density(network_GT_selfloop)),
                            edge_weights=priprities, pnum=args.PNUM, optim='optN')
        endTime = int(round(time.time() * 1000))
        sat_time = endTime - startTime
        print('number of optimal solutions is', len(r_estimated))
        ### minimizing with respect to Gu_opt Vs. G_estimate
        min_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
        min_norm_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
        min_val = 1000000
        min_cost = 10000000
        for answer in r_estimated:
            curr_errors = gk.OCE(bfutils.undersample(bfutils.num2CG(answer[0][0], len(network_GT_selfloop)), answer[0][1][0]),
                                 g_estimated)
            curr_normed_errors = gk.OCE(bfutils.undersample(bfutils.num2CG(answer[0][0], len(network_GT_selfloop)), answer[0][1][0]),
                                        g_estimated, normalized=True)
            curr_cost = answer[1]
            if (curr_errors['total'][0] + curr_errors['total'][1]) < min_val:
                min_err = curr_errors
                min_norm_err = curr_normed_errors
                min_cost = curr_cost
                min_val = (curr_errors['total'][0] + curr_errors['total'][1])
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
        network_GT_U_WRT_GuOptVsGest = bfutils.undersample(network_GT_selfloop, min_answer_WRT_GuOptVsGest[0][1][0])

        Gu_opt_errors_network_GT_U_WRT_GuOptVsGest = \
        gk.OCE(Gu_opt_WRT_GuOptVsGest, network_GT_U_WRT_GuOptVsGest, undirected=False, normalized=True)[
            'total']
        Gu_opt_errors_g_estimated_WRT_GuOptVsGest = \
        gk.OCE(Gu_opt_WRT_GuOptVsGest, g_estimated, undirected=False, normalized=True)['total']
        G1_opt_error_GT_WRT_GuOptVsGest = \
        gk.OCE(G1_opt_WRT_GuOptVsGest, network_GT_selfloop, undirected=False, normalized=True)['total']
        print('*******************************************')
        print('results with respect to Gu_opt Vs. G_estimate ')
        print('U rate found to be:' + str(min_answer_WRT_GuOptVsGest[0][1][0]))
        print('Gu_opt_errors_network_GT_U = ', round_tuple_elements(Gu_opt_errors_network_GT_U_WRT_GuOptVsGest))
        print('Gu_opt_errors_g_estimated', round_tuple_elements(Gu_opt_errors_g_estimated_WRT_GuOptVsGest))
        print('G1_opt_error_GT', round_tuple_elements(G1_opt_error_GT_WRT_GuOptVsGest))


        SL_undir_normed_errors_omm.append(G1_opt_error_GT_WRT_GuOptVsGest[0])
        SL_undir_normed_errors_comm.append(G1_opt_error_GT_WRT_GuOptVsGest[1])

        ### minimizing with respect to G1_opt Vs. GT
        min_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
        min_norm_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
        min_val = 1000000
        min_cost = 10000000
        for answer in r_estimated:
            curr_errors = gk.OCE(bfutils.num2CG(answer[0][0], len(network_GT_selfloop)), network_GT_selfloop)
            curr_normed_errors = gk.OCE(bfutils.num2CG(answer[0][0], len(network_GT_selfloop)), network_GT_selfloop, normalized=True)
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
        network_GT_U_WRT_G1OptVsGT = bfutils.undersample(network_GT_selfloop, min_answer_WRT_G1OptVsGT[0][1][0])

        Gu_opt_errors_network_GT_U_WRT_G1OptVsGT = \
        gk.OCE(Gu_opt_WRT_G1OptVsGT, network_GT_U_WRT_G1OptVsGT, undirected=False, normalized=True)[
            'total']
        Gu_opt_errors_g_estimated_WRT_G1OptVsGT = \
        gk.OCE(Gu_opt_WRT_G1OptVsGT, g_estimated, undirected=False, normalized=True)['total']
        G1_opt_error_GT_WRT_G1OptVsGT = \
        gk.OCE(G1_opt_WRT_G1OptVsGT, network_GT_selfloop, undirected=False, normalized=True)['total']
        print('*******************************************')
        print('results of minimizing with respect to G1_opt Vs. GT')
        print('U rate found to be:' + str(min_answer_WRT_G1OptVsGT[0][1][0]))
        print('Gu_opt_errors_network_GT_U = ', round_tuple_elements(Gu_opt_errors_network_GT_U_WRT_G1OptVsGT))
        print('Gu_opt_errors_g_estimated', round_tuple_elements(Gu_opt_errors_g_estimated_WRT_G1OptVsGT))
        print('G1_opt_error_GT', round_tuple_elements(G1_opt_error_GT_WRT_G1OptVsGT))

        opt_SL_undir_normed_errors_omm.append(G1_opt_error_GT_WRT_G1OptVsGT[0])
        opt_SL_undir_normed_errors_comm.append(G1_opt_error_GT_WRT_G1OptVsGT[1])
        #######presision and recall (orintattion)
        # Precision = True Positives / (True Positives + False Positives)
        # Recall = True Positives /  (True Positives + False Negatives)

        res_graph = bfutils.num2CG(min_answer_WRT_G1OptVsGT[0][0], len(g_estimated))
        GT_nx = gk.graph2nx(network_GT_selfloop)
        res_nx = gk.graph2nx(res_graph)
        TP, FP, FN = 0, 0, 0
        for edge in GT_nx.edges():
                if edge in res_nx.edges():
                    TP += 1
                else:
                    FN += 1
        for edge in res_nx.edges():
                if edge not in GT_nx.edges():
                    FP += 1
        if (TP + FP) != 0 and (TP + FN) != 0:
            Precision_O.append(TP / (TP + FP))
            Recall_O.append(TP / (TP + FN))

        #######presision and recall (adjacency)
        TP, FP, FN = 0, 0, 0
        for edge in GT_nx.edges():
                if edge in res_nx.edges() or (edge[1], edge[0]) in res_nx.edges():
                    if ((edge[1], edge[0]) in GT_nx.edges()) and (edge[1] != edge[0]):
                        TP += 0.5
                    else:
                        TP += 1
                else:
                    if (edge[1], edge[0]) in GT_nx.edges():
                        FN += 0.5
                    else:
                        FN += 1
        for edge in res_nx.edges():
                if not (edge in GT_nx.edges() or (edge[1], edge[0]) in GT_nx.edges()):
                    if ((edge[1], edge[0]) in res_nx.edges()) and (edge[1] != edge[0]):
                        FP += 0.5
                    else:
                        FP += 1
        if not (FP % 1 == 0 and TP % 1 == 0 and FN % 1 == 0):
            print('see why')
        if (TP + FP) != 0 and (TP + FN) != 0:
            Precision_A.append(TP / (TP + FP))
            Recall_A.append(TP / (TP + FN))
        else:
            print('see how')


        #######presision and recall (2-cycle)

        TP, FP, FN = 0, 0, 0
        for edge in GT_nx.edges():
            if not edge[1] == edge[0]:
                if (edge[1], edge[0]) in GT_nx.edges():
                    if edge in res_nx.edges() and (edge[1], edge[0]) in res_nx.edges():
                        TP += 1
                    else:
                        FN += 1
        for edge in res_nx.edges():
            if not edge[1] == edge[0]:
                if (edge[1], edge[0]) in res_nx.edges():
                    if not (edge in GT_nx.edges() and (edge[1], edge[0]) in GT_nx.edges()):
                        FP += 1
        if (TP + FP) != 0 and (TP + FN) != 0:
            Precision_C.append(TP / (TP + FP))
            Recall_C.append(TP / (TP + FN))

        sorted_data = sorted(r_estimated, key=lambda x: x[1], reverse=True)
        # add to lists for dict
        results = {'general': {'method': PreFix,
                               'g_estimated': g_estimated,
                               'sample': fl,
                               'dm': DD,
                               'bdm': BD,
                               'optim_cost': sorted_data[-1][1],
                               'num_sols': len(r_estimated),
                               'GT': network_GT_selfloop,
                               'threshold': EDGE_CUTOFF * k_threshold,
                               'timeout': TIMEOUT,
                               'full_sols': r_estimated,
                               'total_time': round(((sat_time) / 60000), 3)},
                   'GuOptVsGest': {
                       'min_answer_WRT_GuOptVsGest': min_answer_WRT_GuOptVsGest,
                       'G1_opt_WRT_GuOptVsGest': G1_opt_WRT_GuOptVsGest,
                       'Gu_opt_WRT_GuOptVsGest': Gu_opt_WRT_GuOptVsGest,
                       'network_GT_U_WRT_GuOptVsGest': network_GT_U_WRT_GuOptVsGest,
                       'Gu_opt_errors_network_GT_U_WRT_GuOptVsGest': Gu_opt_errors_network_GT_U_WRT_GuOptVsGest,
                       'Gu_opt_errors_g_estimated_WRT_GuOptVsGest': Gu_opt_errors_g_estimated_WRT_GuOptVsGest,
                       'G1_opt_error_GT_WRT_GuOptVsGest': G1_opt_error_GT_WRT_GuOptVsGest,
                       'U_found':min_answer_WRT_GuOptVsGest[0][1][0]
                   },
                   'G1OptVsGT': {
                       'min_answer_WRT_G1OptVsGT': min_answer_WRT_G1OptVsGT,
                       'G1_opt_WRT_G1OptVsGT': G1_opt_WRT_G1OptVsGT,
                       'Gu_opt_WRT_G1OptVsGT': Gu_opt_WRT_G1OptVsGT,
                       'network_GT_U_WRT_G1OptVsGT': network_GT_U_WRT_G1OptVsGT,
                       'Gu_opt_errors_network_GT_U_WRT_G1OptVsGT': Gu_opt_errors_network_GT_U_WRT_G1OptVsGT,
                       'Gu_opt_errors_g_estimated_WRT_G1OptVsGT': Gu_opt_errors_g_estimated_WRT_G1OptVsGT,
                       'G1_opt_error_GT_WRT_G1OptVsGT': G1_opt_error_GT_WRT_G1OptVsGT,
                       'U_found': min_answer_WRT_G1OptVsGT[0][1][0]
                   }}
        save_results.append(results)

    now = str(datetime.now())
    now = now[:-7].replace(' ', '_')
    # plotting
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 32}
    plt.rc('font', **font)
    fig, axs = plt.subplots(2)
    for ax in axs:
        ax.set_ylim(0, 1)
    fig.set_figwidth(35)
    fig.set_figheight(20)
    fig.suptitle('Network' + str(
        args.NUMBER) + ' on ' + PreFix + ':Om and comm err after comparing G1 to H1. individual data. no selfloop')
    # axs[0].plot(errors_omm,label="ommition error")
    axs[0].plot(SL_undir_normed_errors_omm, c='red', label="not knowing GT", linewidth=5)
    # axs[0].plot(undir_errors_omm, c='green', label="omm: sRASL", linewidth=5)
    axs[0].plot(opt_SL_undir_normed_errors_omm, c='orange', label="knowing GT", linewidth=5)
    # axs[0].plot(opt_undir_errors_omm, c='blue', label="omm: sRASL + Opt", linewidth=5)
    axs[0].legend()
    # axs[1].plot(errors_comm, label="commition error")
    # SL_drop_bd_normed_errors_comm= [0.003+i for i in SL_drop_bd_normed_errors_comm]all_network_SL
    axs[1].plot(SL_undir_normed_errors_comm, c='red', label="not knowing GT", linewidth=5)
    # axs[1].plot(undir_errors_comm, c='green', label="comm: sRASL", linewidth=5)
    axs[1].plot(opt_SL_undir_normed_errors_comm, c='orange', label="knowing GT", linewidth=5)
    # axs[1].plot(opt_undir_errors_comm, c='blue', label="comm: sRASL + Opt", linewidth=5)
    axs[1].legend()
    # plt.show()

    ###saving files
    filename = PreFix + '_results_with_directions_Optim_vs_OPTsRASL_error_on_G1_H1' + str(args.NUMBER) + '_' + now
    plt.savefig(filename + '_.png')
    plt.close()
    zkl.save(save_results, filename + '_.zkl')
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 18}
    plt.rc('font', **font)

    precision_mean = sum(Precision_O) / len(Precision_O)
    recall_mean = sum(Recall_O) / len(Recall_O)
    # Names for the bars
    names = ['Precision', 'Recall']
    # Mean values for the bars
    means = [precision_mean, recall_mean]
    # Plotting the bar plot
    plt.bar(names, means, color=['blue', 'green'])
    plt.ylim(0, 1)
    plt.xlabel('Metrics')
    plt.ylabel('Mean')
    plt.title('Mean of Precision and Recall (orientation)')
    plt.savefig(filename + '_orintation.png')
    plt.close()

    precision_mean = sum(Precision_A) / len(Precision_A)
    recall_mean = sum(Recall_A) / len(Recall_A)
    # Names for the bars
    names = ['Precision', 'Recall']
    # Mean values for the bars
    means = [precision_mean, recall_mean]
    # Plotting the bar plot
    plt.bar(names, means, color=['blue', 'green'])
    plt.ylim(0, 1)
    plt.xlabel('Metrics')
    plt.ylabel('Mean')
    plt.title('Mean of Precision and Recall (adjacency)')
    plt.savefig(filename + '_adjacency.png')
    plt.close()

    precision_mean = sum(Precision_C) / len(Precision_C)
    recall_mean = sum(Recall_C) / len(Recall_C)
    # Names for the bars
    names = ['Precision', 'Recall']
    # Mean values for the bars
    means = [precision_mean, recall_mean]
    # Plotting the bar plot
    plt.bar(names, means, color=['blue', 'green'])
    plt.ylim(0, 1)
    plt.xlabel('Metrics')
    plt.ylabel('Mean')
    plt.title('Mean of Precision and Recall (2 cycle)')
    plt.savefig(filename + '_2_cycle.png')