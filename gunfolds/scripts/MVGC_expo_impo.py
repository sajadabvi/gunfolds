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
from scipy.io import loadmat
from scipy.io import savemat

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
    true_positives += np.trace(np.logical_and(adj_matrix1, adj_matrix2))
    # Calculate false positives (edges in graph2 but not in graph1)
    false_positives = np.sum(np.logical_and(adj_matrix2, np.logical_not(adj_matrix1)))
    false_positives += np.trace(np.logical_and(adj_matrix2, np.logical_not(adj_matrix1)))

    # Calculate false negatives (edges in graph1 but not in graph2)
    false_negatives = np.sum(np.logical_and(adj_matrix1, np.logical_not(adj_matrix2)))
    false_negatives += np.trace(np.logical_and(adj_matrix1, np.logical_not(adj_matrix2)))

    true_positives = true_positives/2
    false_positives= false_positives/2
    false_negatives = false_negatives/2

    # Calculate precision
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0

    # Calculate recall
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

    return precision, recall


def find_two_cycles(graph):
    # Find all 2-cycles in the graph
    two_cycles = set()
    visited = set()
    for node in graph.nodes():
        for neighbor in graph.neighbors(node):
            # Check for a directed edge in both directions and ensure it's not a self-loop
            if node != neighbor and graph.has_edge(node, neighbor) and graph.has_edge(neighbor, node):
                # Ensure we count each 2-cycle only once
                edge_pair = tuple(sorted([node, neighbor]))
                if edge_pair not in visited:
                    two_cycles.add(edge_pair)
                    visited.add(edge_pair)
    return two_cycles

def precision_recall_2cycles(graph1, graph2):


    # Find 2-cycles in both graphs
    two_cycles_graph1 = find_two_cycles(graph1)
    two_cycles_graph2 = find_two_cycles(graph2)

    # Calculate true positives (intersection of 2-cycles)
    true_positives = len(two_cycles_graph1.intersection(two_cycles_graph2))

    # Calculate false positives (2-cycles in graph2 but not in graph1)
    false_positives = len(two_cycles_graph2 - two_cycles_graph1)

    # Calculate false negatives (2-cycles in graph1 but not in graph2)
    false_negatives = len(two_cycles_graph1 - two_cycles_graph2)

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
    Precision_A2 = []
    Recall_A2 = []
    Precision_C = []
    Recall_C = []
    Precision_C2 = []
    Recall_C2 = []
    for fl in range(1, 61):
        num = str(fl) if fl > 9 else '0' + str(fl)
        print('reading file:' + num)
        data = pd.read_csv(
            './DataSets_Feedbacks/1. Simple_Networks/Network' + str(
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
        savemat('expo_to_mat/expo_to_mat_'+str(fl)+'.mat', {'dd': dd})

    for fl in range(1, 61):
        #######presision and recall (orintattion)
        # Precision = True Positives / (True Positives + False Positives)
        # Recall = True Positives /  (True Positives + False Negatives)
        mat_data = loadmat('expo_to_py/mat_file_'+str(fl)+'.mat')
        mat = mat_data['sig']
        for i in range(5):
            mat[i, i] = 1
        MVGC = cv.adjs2graph(mat, np.zeros((5, 5)))
        res_graph = MVGC
        gt.plotg(MVGC, output='./figs/Gopt_GC_' + str(fl) + '.pdf')
        GT_nx = gk.graph2nx(network_GT_selfloop)
        res_nx = gk.graph2nx(res_graph)
        TP, FP, FN = 0, 0, 0
        for edge in GT_nx.edges():
            if (edge[1] != edge[0]):
                if edge in res_nx.edges():
                    TP += 1
                else:
                    FN += 1
        for edge in res_nx.edges():
            if (edge[1] != edge[0]):
                if edge not in GT_nx.edges():
                    FP += 1
        if (TP + FP) != 0 and (TP + FN) != 0:
            Precision_O.append(TP / (TP + FP))
            Recall_O.append(TP / (TP + FN))

        #######presision and recall (adjacency)
        TP, FP, FN = 0, 0, 0
        for edge in GT_nx.edges() :
            if (edge[1] != edge[0]):
                if edge in res_nx.edges() or (edge[1], edge[0]) in res_nx.edges():
                    if ( (edge[1], edge[0]) in GT_nx.edges()) and (edge[1] != edge[0]):
                        TP += 0.5
                    else:
                        TP += 1
                else:
                    if (edge[1], edge[0]) in GT_nx.edges():
                        FN += 0.5
                    else:
                        FN += 1
        for edge in res_nx.edges():
            if (edge[1] != edge[0]):
                if not(edge in GT_nx.edges() or (edge[1], edge[0]) in GT_nx.edges()):
                    if  ((edge[1], edge[0]) in res_nx.edges()) and (edge[1] != edge[0]):
                        FP += 0.5
                    else:
                        FP += 1
        if not (FP%1 == 0 and TP%1 == 0 and FN%1 == 0):
            print('see why')
        if (TP + FP) != 0 and (TP + FN) != 0:
            Precision_A.append(TP / (TP + FP))
            Recall_A.append(TP / (TP + FN))
        else:
            print('see how')

        #WRONG CALCULATIONS
        # precision_val, recall_val = precision_recall(GT_nx,res_nx)
        # Precision_A2.append(precision_val)
        # Recall_A2.append(recall_val)
        #######presision and recall (2-cycle)

        TP, FP, FN = 0, 0, 0
        for edge in GT_nx.edges():
            if not edge[1]== edge[0]:
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
        if  (TP + FP) !=0 and  (TP + FN) !=0:
            Precision_C.append(TP / (TP + FP))
            Recall_C.append(TP / (TP + FN))

        # precision_val, recall_val = precision_recall_2cycles(GT_nx, res_nx)
        # Precision_C2.append(precision_val)
        # Recall_C2.append(recall_val)

    now = str(datetime.now())
    now = now[:-7].replace(' ', '_')

    ###saving files
    filename = PreFix + '_res_MVGC_no_selfloop' + str(args.NUMBER) + '_' + now


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

    #####
    # precision_mean = sum(Precision_A2) / len(Precision_A2)
    # recall_mean = sum(Recall_A2) / len(Recall_A2)
    # # Names for the bars
    # names = ['Precision', 'Recall']
    # # Mean values for the bars
    # means = [precision_mean, recall_mean]
    # # Plotting the bar plot
    # plt.bar(names, means, color=['blue', 'green'])
    # plt.ylim(0, 1)
    # plt.xlabel('Metrics')
    # plt.ylabel('Mean')
    # plt.title('Mean of Precision and Recall (adjacency22)')
    # plt.savefig(filename + '_adjacency22.png')
    # plt.close()
    #######

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

    # precision_mean = sum(Precision_C2) / len(Precision_C2)
    # recall_mean = sum(Recall_C2) / len(Recall_C2)
    # # Names for the bars
    # names = ['Precision', 'Recall']
    # # Mean values for the bars
    # means = [precision_mean, recall_mean]
    # # Plotting the bar plot
    # plt.bar(names, means, color=['blue', 'green'])
    # plt.ylim(0, 1)
    # plt.xlabel('Metrics')
    # plt.ylabel('Mean')
    # plt.title('Mean of Precision and Recall (2 cycle22)')
    # plt.savefig(filename + '_2_cycle22.png')