import networkx as nx
from gunfolds.utils import graphkit as gk


def remove_bidir_edges(input_dict):
    result_dict = {}
    for key, inner_dict in input_dict.items():
        result_dict[key] = {}
        for inner_key, value in inner_dict.items():
            if value == 1:
                result_dict[key][inner_key] = 1
            elif value == 2:
                pass  # Skip adding this key-value pair
            elif value == 3:
                result_dict[key][inner_key] = 1
            else:
                raise ValueError("Invalid value encountered: {}".format(value))
    return result_dict


def precision_recall(answer, network_GT_selfloop):
    # Precision = True Positives / (True Positives + False Positives)
    # Recall = True Positives /  (True Positives + False Negatives)
    res_graph = answer
    GT_nx = gk.graph2nx(network_GT_selfloop)
    res_nx = gk.graph2nx(res_graph)

    #######precision and recall (orientation)
    TP, FP, FN = 0, 0, 0
    for edge in GT_nx.edges():
        if edge in res_nx.edges():
            TP += 1
        else:
            FN += 1
    for edge in res_nx.edges():
        if edge not in GT_nx.edges():
            FP += 1
    p_O = (TP / (TP + FP)) if (TP + FP) else 0
    r_O = (TP / (TP + FN)) if (TP + FN) else 0
    f1_O = (2 * TP) / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0

    #######precision and recall (adjacency)
    TP, FP, FN = 0, 0, 0
    for edge in GT_nx.edges():
        if edge in res_nx.edges() or (edge[1], edge[0]) in res_nx.edges():
            if ((edge[1], edge[0]) in GT_nx.edges()) and (edge[1] != edge[0]):
                TP += 0.5
            else:
                TP += 1
        else:
            if (edge[1], edge[0]) in GT_nx.edges() and (edge[1] != edge[0]):
                FN += 0.5
            else:
                FN += 1
    for edge in res_nx.edges():
        if not (edge in GT_nx.edges() or (edge[1], edge[0]) in GT_nx.edges()):
            if ((edge[1], edge[0]) in res_nx.edges()) and (edge[1] != edge[0]):
                FP += 0.5
            else:
                FP += 1
    p_A = (TP / (TP + FP)) if (TP + FP) else 0
    r_A = (TP / (TP + FN)) if (TP + FN) else 0
    f1_A = (2 * TP) / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0

    #######precision and recall (2-cycle)

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
    p_C = (TP / (TP + FP)) if (TP + FP) else 0
    r_C = (TP / (TP + FN)) if (TP + FN) else 0
    f1_C = (2 * TP) / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0

    prf = {'orientation': {'precision': p_O, 'recall': r_O, 'F1': f1_O},
           'adjacency': {'precision': p_A, 'recall': r_A, 'F1': f1_A},
           'cycle': {'precision': p_C, 'recall': r_C, 'F1': f1_C}}

    return prf


def round_tuple_elements(input_tuple, decimal_points=3):
    return tuple(round(elem, decimal_points) if isinstance(elem, (int, float)) else elem for elem in input_tuple)
