from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from gunfolds.utils import zickle as zkl
from os import listdir
from gunfolds.viz import gtool as gt
from collections import defaultdict
from gunfolds.utils import graphkit as gk
from gunfolds.utils import bfutils


############################################################

folder14 = '/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/stable_trans_mat/PCMCI/prioriti42531/u2/'

file_list14 = listdir(folder14)
file_list14.sort()
if file_list14[0].startswith('.'):
    file_list14.pop(0)

res14 = [zkl.load(folder14 + file) for file in file_list14]

############################################################

folder15 = '/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/stable_trans_mat/PCMCI/prioriti42531/u3/'

file_list15 = listdir(folder15)
file_list15.sort()
if file_list15[0].startswith('.'):
    file_list15.pop(0)

res15 = [zkl.load(folder15 + file) for file in file_list15]

############################################################

folder16 = '/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/stable_trans_mat/PCMCI/prioriti42531/u4/'

file_list16 = listdir(folder16)
file_list16.sort()
if file_list16[0].startswith('.'):
    file_list16.pop(0)

res16 = [zkl.load(folder16 + file) for file in file_list16]

############################################################

folder17 = '/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/stable_trans_mat/PCMCI/prioriti42531/u2/'

file_list17 = listdir(folder17)
file_list17.sort()
if file_list17[0].startswith('.'):
    file_list17.pop(0)

res17 = [zkl.load(folder17 + file) for file in file_list17]

############################################################

folder18 = '/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/stable_trans_mat/PCMCI/prioriti42531/u3/'

file_list18 = listdir(folder18)
file_list18.sort()
if file_list18[0].startswith('.'):
    file_list18.pop(0)

res18 = [zkl.load(folder18 + file) for file in file_list18]

############################################################

folder19 = '/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/stable_trans_mat/PCMCI/prioriti42531/u4/'

file_list19 = listdir(folder19)
file_list19.sort()
if file_list19[0].startswith('.'):
    file_list19.pop(0)
res19 = [zkl.load(folder19 + file) for file in file_list19]
############################################################


G1_opt_error_GT_om = []
G1_opt_error_GT_com = []
Gu_opt_errors_network_GT_U_om = []
Gu_opt_errors_network_GT_U_com = []
Gu_opt_errors_g_estimated_om = []
Gu_opt_errors_g_estimated_com = []

err_criteria = 'total'
def merge_graphs(graphs, threshold_density):
    # Dictionary to store edge frequencies
    edge_frequencies = defaultdict(int)

    # Set to store all nodes in the input graphs
    all_nodes = set()

    # Count the frequencies of each edge in the list of graphs
    for graph in graphs:
        all_nodes.update(graph.keys())
        for node, connections in graph.items():
            for neighbor, edge_type in connections.items():
                edge_frequencies[(node, neighbor, edge_type)] += 1

    # Calculate the threshold for inclusion based on the provided density
    threshold = threshold_density * len(graphs)

    # Create the merged graph based on edges that meet the threshold
    merged_graph = {}
    for (node, neighbor, edge_type), frequency in edge_frequencies.items():
        if frequency >= threshold:
            if node not in merged_graph:
                merged_graph[node] = {}
            merged_graph[node][neighbor] = edge_type

    # Ensure all nodes are present in the merged graph
    for graph in graphs:
        for node in all_nodes:
            if node not in merged_graph:
                merged_graph[node] = {}

    return merged_graph

def cal_freq_error(item, err_criteria, type):
    g1s = []
    gus = []
    gt = item['general']['GT']
    gtu = item['general']['GT_at_actual_U']
    g_est = item['general']['g_estimated']
    for sol in item['general']['full_sols']:
        g1s.append(bfutils.num2CG(sol[0][0], len(item['general']['GT'])))
        gus.append(bfutils.undersample(bfutils.num2CG(sol[0][0], len(item['general']['GT'])), sol[0][1][0]))
    th1 = 0.1
    g1 = merge_graphs(g1s,th1)
    while gk.density(g1) > gk.density(gt):
        th1 += 0.01
        g1 = merge_graphs(g1s, th1)

    thu = 0.1
    gu = merge_graphs(gus, thu)
    while gk.density(gu) > gk.density(gtu):
        thu += 0.01
        gu = merge_graphs(gus, thu)

    err_GuVsGTu = gk.OCE(gu, gtu, undirected=False, normalized=True)[err_criteria][type]
    err_GuVsGest = gk.OCE(gu, g_est, undirected=False, normalized=True)[err_criteria][type]
    err_G1VsGT = gk.OCE(g1, gt, undirected=False, normalized=True)[err_criteria][type]
    return [err_GuVsGTu, err_GuVsGest, err_G1VsGT]
def cal_mean_error(item, err_criteria, type):
    GuVsGTu = []
    GuVsGest = []
    G1VsGT = []
    gt = item['general']['GT']
    gtu = item['general']['GT_at_actual_U']
    g_est = item['general']['g_estimated']
    for sol in item['general']['full_sols']:
        g1 = bfutils.num2CG(sol[0][0], len(item['general']['GT']))
        gu = bfutils.undersample(bfutils.num2CG(sol[0][0], len(item['general']['GT'])), sol[0][1][0])
        GuVsGTu.append(gk.OCE(gu, gtu, undirected=False, normalized=True)[err_criteria][type])
        GuVsGest.append(gk.OCE(gu, g_est, undirected=False, normalized=True)[err_criteria][type])
        G1VsGT.append(gk.OCE(g1, gt, undirected=False, normalized=True)[err_criteria][type])
    err_GuVsGTu = sum(GuVsGTu) / len(GuVsGTu)
    err_GuVsGest = sum(GuVsGest) / len(GuVsGest)
    err_G1VsGT = sum(G1VsGT) / len(G1VsGT)
    return [err_GuVsGTu, err_GuVsGest, err_G1VsGT]

if __name__ == '__main__':
    df = pd.DataFrame()
    Err = []
    method = []
    ErrType = []
    u = []
    deg = []
    node = []
    WRT = []
    ErrVs = []
    weights_scheme = []

    for item in res14:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([gk.OCE(item['GuOptVsGest']['Gu_opt_WRT_GuOptVsGest'],
                           item['GuOptVsGest']['network_GT_U_WRT_GuOptVsGest'], undirected=False, normalized=True)[
                        err_criteria][0],
                    gk.OCE(item['GuOptVsGest']['Gu_opt_WRT_GuOptVsGest'], item['general']['g_estimated'],
                           undirected=False, normalized=True)[err_criteria][0],
                    gk.OCE(item['GuOptVsGest']['G1_opt_WRT_GuOptVsGest'], item['general']['GT'], undirected=False,
                           normalized=True)[err_criteria][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U2_traditional_error', 'U2_traditional_error', 'U2_traditional_error'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([gk.OCE(item['GuOptVsGest']['Gu_opt_WRT_GuOptVsGest'],
                           item['GuOptVsGest']['network_GT_U_WRT_GuOptVsGest'], undirected=False, normalized=True)[
                        err_criteria][1],
                    gk.OCE(item['GuOptVsGest']['Gu_opt_WRT_GuOptVsGest'], item['general']['g_estimated'],
                           undirected=False, normalized=True)[err_criteria][1],
                    gk.OCE(item['GuOptVsGest']['G1_opt_WRT_GuOptVsGest'], item['general']['GT'], undirected=False,
                           normalized=True)[err_criteria][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U2_traditional_error', 'U2_traditional_error', 'U2_traditional_error'])
        #################################################################################################
        # ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        # WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        # Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
        #             item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
        #             item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        # ErrType.extend(['omm', 'omm', 'omm'])
        # weights_scheme.extend(['U2_traditional_error', 'U2_traditional_error', 'U2_traditional_error'])
        # ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        # WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        # Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
        #             item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
        #             item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        # ErrType.extend(['comm', 'comm', 'comm'])
        # weights_scheme.extend(['U2_traditional_error', 'U2_traditional_error', 'U2_traditional_error'])
        #################################################################################################
        # Err.extend([gk.OCE(item['G1OptVsGT']['Gu_opt_WRT_G1OptVsGT'], item['G1OptVsGT']['network_GT_U_WRT_G1OptVsGT'], undirected=False, normalized=True)[err_criteria][0],
        #             gk.OCE(item['G1OptVsGT']['Gu_opt_WRT_G1OptVsGT'], item['general']['g_estimated'], undirected=False, normalized=True)[err_criteria][0],
        #             gk.OCE(item['G1OptVsGT']['G1_opt_WRT_G1OptVsGT'], item['general']['GT'], undirected=False, normalized=True)[err_criteria][0]])

        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([gk.OCE(item['G1OptVsGT']['Gu_opt_WRT_G1OptVsGT'], item['G1OptVsGT']['network_GT_U_WRT_G1OptVsGT'],
                           undirected=False, normalized=True)[err_criteria][0],
                    gk.OCE(item['G1OptVsGT']['Gu_opt_WRT_G1OptVsGT'], item['general']['g_estimated'], undirected=False,
                           normalized=True)[err_criteria][0],
                    gk.OCE(item['G1OptVsGT']['G1_opt_WRT_G1OptVsGT'], item['general']['GT'], undirected=False,
                           normalized=True)[err_criteria][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U2_traditional_error', 'U2_traditional_error', 'U2_traditional_error'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([gk.OCE(item['G1OptVsGT']['Gu_opt_WRT_G1OptVsGT'], item['G1OptVsGT']['network_GT_U_WRT_G1OptVsGT'],
                           undirected=False, normalized=True)[err_criteria][1],
                    gk.OCE(item['G1OptVsGT']['Gu_opt_WRT_G1OptVsGT'], item['general']['g_estimated'], undirected=False,
                           normalized=True)[err_criteria][1],
                    gk.OCE(item['G1OptVsGT']['G1_opt_WRT_G1OptVsGT'], item['general']['GT'], undirected=False,
                           normalized=True)[err_criteria][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U2_traditional_error', 'U2_traditional_error', 'U2_traditional_error'])

    for item in res17:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend(cal_mean_error(item,err_criteria,0))
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U2_mean_error', 'U2_mean_error', 'U2_mean_error'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend(cal_mean_error(item,err_criteria,1))
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U2_mean_error', 'U2_mean_error', 'U2_mean_error'])
        #################################################################################################
        # ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        # WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        # Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
        #             item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
        #             item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        # ErrType.extend(['omm', 'omm', 'omm'])
        # weights_scheme.extend(['U2_mean_error', 'U2_mean_error', 'U2_mean_error'])
        # ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        # WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        # Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
        #             item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
        #             item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        # ErrType.extend(['comm', 'comm', 'comm'])
        # weights_scheme.extend(['U2_mean_error', 'U2_mean_error', 'U2_mean_error'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend(cal_mean_error(item,err_criteria,0))
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U2_mean_error', 'U2_mean_error', 'U2_mean_error'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend(cal_mean_error(item,err_criteria,1))
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U2_mean_error', 'U2_mean_error', 'U2_mean_error'])

    for item in res15:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([gk.OCE(item['GuOptVsGest']['Gu_opt_WRT_GuOptVsGest'],
                           item['GuOptVsGest']['network_GT_U_WRT_GuOptVsGest'], undirected=False, normalized=True)[
                        err_criteria][0],
                    gk.OCE(item['GuOptVsGest']['Gu_opt_WRT_GuOptVsGest'], item['general']['g_estimated'],
                           undirected=False, normalized=True)[err_criteria][0],
                    gk.OCE(item['GuOptVsGest']['G1_opt_WRT_GuOptVsGest'], item['general']['GT'], undirected=False,
                           normalized=True)[err_criteria][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U3_traditional_error', 'U3_traditional_error', 'U3_traditional_error'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([gk.OCE(item['GuOptVsGest']['Gu_opt_WRT_GuOptVsGest'],
                           item['GuOptVsGest']['network_GT_U_WRT_GuOptVsGest'], undirected=False, normalized=True)[
                        err_criteria][1],
                    gk.OCE(item['GuOptVsGest']['Gu_opt_WRT_GuOptVsGest'], item['general']['g_estimated'],
                           undirected=False, normalized=True)[err_criteria][1],
                    gk.OCE(item['GuOptVsGest']['G1_opt_WRT_GuOptVsGest'], item['general']['GT'], undirected=False,
                           normalized=True)[err_criteria][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U3_traditional_error', 'U3_traditional_error', 'U3_traditional_error'])
        #################################################################################################
        # ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        # WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        # Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
        #             item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
        #             item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        # ErrType.extend(['omm', 'omm', 'omm'])
        # weights_scheme.extend(['U3_traditional_error', 'U3_traditional_error', 'U3_traditional_error'])
        # ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        # WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        # Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
        #             item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
        #             item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        # ErrType.extend(['comm', 'comm', 'comm'])
        # weights_scheme.extend(['U3_traditional_error', 'U3_traditional_error', 'U3_traditional_error'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([gk.OCE(item['G1OptVsGT']['Gu_opt_WRT_G1OptVsGT'], item['G1OptVsGT']['network_GT_U_WRT_G1OptVsGT'],
                           undirected=False, normalized=True)[err_criteria][0],
                    gk.OCE(item['G1OptVsGT']['Gu_opt_WRT_G1OptVsGT'], item['general']['g_estimated'], undirected=False,
                           normalized=True)[err_criteria][0],
                    gk.OCE(item['G1OptVsGT']['G1_opt_WRT_G1OptVsGT'], item['general']['GT'], undirected=False,
                           normalized=True)[err_criteria][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U3_traditional_error', 'U3_traditional_error', 'U3_traditional_error'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([gk.OCE(item['G1OptVsGT']['Gu_opt_WRT_G1OptVsGT'], item['G1OptVsGT']['network_GT_U_WRT_G1OptVsGT'],
                           undirected=False, normalized=True)[err_criteria][1],
                    gk.OCE(item['G1OptVsGT']['Gu_opt_WRT_G1OptVsGT'], item['general']['g_estimated'], undirected=False,
                           normalized=True)[err_criteria][1],
                    gk.OCE(item['G1OptVsGT']['G1_opt_WRT_G1OptVsGT'], item['general']['GT'], undirected=False,
                           normalized=True)[err_criteria][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U3_traditional_error', 'U3_traditional_error', 'U3_traditional_error'])

    for item in res18:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend(cal_mean_error(item,err_criteria,0))
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U3_mean_error', 'U3_mean_error', 'U3_mean_error'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend(cal_mean_error(item,err_criteria,1))
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U3_mean_error', 'U3_mean_error', 'U3_mean_error'])
        #################################################################################################
        # ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        # WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        # Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
        #             item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
        #             item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        # ErrType.extend(['omm', 'omm', 'omm'])
        # weights_scheme.extend(['U3_mean_error', 'U3_mean_error', 'U3_mean_error'])
        # ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        # WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        # Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
        #             item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
        #             item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        # ErrType.extend(['comm', 'comm', 'comm'])
        # weights_scheme.extend(['U3_mean_error', 'U3_mean_error', 'U3_mean_error'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend(cal_mean_error(item,err_criteria,0))
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U3_mean_error', 'U3_mean_error', 'U3_mean_error'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend(cal_mean_error(item,err_criteria,1))
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U3_mean_error', 'U3_mean_error', 'U3_mean_error'])

    for item in res16:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([gk.OCE(item['GuOptVsGest']['Gu_opt_WRT_GuOptVsGest'],
                           item['GuOptVsGest']['network_GT_U_WRT_GuOptVsGest'], undirected=False, normalized=True)[
                        err_criteria][0],
                    gk.OCE(item['GuOptVsGest']['Gu_opt_WRT_GuOptVsGest'], item['general']['g_estimated'],
                           undirected=False, normalized=True)[err_criteria][0],
                    gk.OCE(item['GuOptVsGest']['G1_opt_WRT_GuOptVsGest'], item['general']['GT'], undirected=False,
                           normalized=True)[err_criteria][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U4_traditional_error', 'U4_traditional_error', 'U4_traditional_error'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([gk.OCE(item['GuOptVsGest']['Gu_opt_WRT_GuOptVsGest'],
                           item['GuOptVsGest']['network_GT_U_WRT_GuOptVsGest'], undirected=False, normalized=True)[
                        err_criteria][1],
                    gk.OCE(item['GuOptVsGest']['Gu_opt_WRT_GuOptVsGest'], item['general']['g_estimated'],
                           undirected=False, normalized=True)[err_criteria][1],
                    gk.OCE(item['GuOptVsGest']['G1_opt_WRT_GuOptVsGest'], item['general']['GT'], undirected=False,
                           normalized=True)[err_criteria][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U4_traditional_error', 'U4_traditional_error', 'U4_traditional_error'])
        #################################################################################################
        # ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        # WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        # Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
        #             item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
        #             item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        # ErrType.extend(['omm', 'omm', 'omm'])
        # weights_scheme.extend(['U4_traditional_error', 'U4_traditional_error', 'U4_traditional_error'])
        # ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        # WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        # Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
        #             item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
        #             item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        # ErrType.extend(['comm', 'comm', 'comm'])
        # weights_scheme.extend(['U4_traditional_error', 'U4_traditional_error', 'U4_traditional_error'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([gk.OCE(item['G1OptVsGT']['Gu_opt_WRT_G1OptVsGT'], item['G1OptVsGT']['network_GT_U_WRT_G1OptVsGT'],
                           undirected=False, normalized=True)[err_criteria][0],
                    gk.OCE(item['G1OptVsGT']['Gu_opt_WRT_G1OptVsGT'], item['general']['g_estimated'], undirected=False,
                           normalized=True)[err_criteria][0],
                    gk.OCE(item['G1OptVsGT']['G1_opt_WRT_G1OptVsGT'], item['general']['GT'], undirected=False,
                           normalized=True)[err_criteria][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U4_traditional_error', 'U4_traditional_error', 'U4_traditional_error'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([gk.OCE(item['G1OptVsGT']['Gu_opt_WRT_G1OptVsGT'], item['G1OptVsGT']['network_GT_U_WRT_G1OptVsGT'],
                           undirected=False, normalized=True)[err_criteria][1],
                    gk.OCE(item['G1OptVsGT']['Gu_opt_WRT_G1OptVsGT'], item['general']['g_estimated'], undirected=False,
                           normalized=True)[err_criteria][1],
                    gk.OCE(item['G1OptVsGT']['G1_opt_WRT_G1OptVsGT'], item['general']['GT'], undirected=False,
                           normalized=True)[err_criteria][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U4_traditional_error', 'U4_traditional_error', 'U4_traditional_error'])

    for item in res19:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend(cal_mean_error(item,err_criteria,0))
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U4_mean_error', 'U4_mean_error', 'U4_mean_error'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend(cal_mean_error(item,err_criteria,1))
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U4_mean_error', 'U4_mean_error', 'U4_mean_error'])
        #################################################################################################
        # ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        # WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        # Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
        #             item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
        #             item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        # ErrType.extend(['omm', 'omm', 'omm'])
        # weights_scheme.extend(['U4_mean_error', 'U4_mean_error', 'U4_mean_error'])
        # ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        # WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        # Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
        #             item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
        #             item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        # ErrType.extend(['comm', 'comm', 'comm'])
        # weights_scheme.extend(['U4_mean_error', 'U4_mean_error', 'U4_mean_error'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend(cal_mean_error(item,err_criteria,0))
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U4_mean_error', 'U4_mean_error', 'U4_mean_error'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend(cal_mean_error(item,err_criteria,1))
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U4_mean_error', 'U4_mean_error', 'U4_mean_error'])

    df['Err'] = Err
    df['ErrVs'] = ErrVs
    df['ErrType'] = ErrType
    df['WRT'] = WRT
    df['weights_scheme'] = weights_scheme

    sns.set({"xtick.minor.size": 0.2})
    pal = dict(U2="gold", U3="blue",
               U4="maroon", U5="green", U6="red", U4_New="yellow")
    g = sns.FacetGrid(df, col="WRT", row="ErrType", height=4, aspect=1.5, margin_titles=True)


    def custom_boxplot(*args, **kwargs):
        sns.boxplot(*args, **kwargs, palette='Set1')


    g.map_dataframe(custom_boxplot, x='ErrVs', y='Err', hue='weights_scheme')
    g.add_legend()
    g.set_axis_labels("error type", "normalized error")
    column_titles = ["Fair errors, not knowing the GT", "optimal case knowing the GT"]  # Add your custom titles here
    for i, title in enumerate(column_titles):
        g.axes[0, i].set_title(title, x=(0.4 + (i * 0.2)))
    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            ax = g.facet_axis(i, j)
            # ax.xaxis.grid(True, "minor", linewidth=.75)
            # ax.xaxis.grid(True, "major", linewidth=3)
            # ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.set_ylim(0, 1)
    plt.suptitle("calculating " + err_criteria + " errors ", x=0.45, y=1, fontsize=20)
    plt.show()
    # plt.savefig("figs/old_errror_cal_VS_meanuency_error_" + err_criteria + "_error.png")
