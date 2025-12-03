from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from gunfolds.utils import zickle as zkl
from os import listdir
from matplotlib.ticker import MultipleLocator
from gunfolds.viz import gtool as gt
from collections import defaultdict
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
folder3 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/res_simulation' \
         '/8nodes/n8sfmf14/'

folder = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/res_simulation' \
         '/8nodes/n8stmt14/'
folder2 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/res_simulation' \
         '/8nodes/n8stmf14/'


file_list3 = listdir(folder3)
file_list3.sort()
if file_list3[0].startswith('.'):
    file_list3.pop(0)

file_list = listdir(folder)
file_list.sort()
if file_list[0].startswith('.'):
    file_list.pop(0)

file_list2 = listdir(folder2)
file_list2.sort()
if file_list2[0].startswith('.'):
    file_list2.pop(0)

res3 = [zkl.load(folder3 + file) for file in file_list3]
res = [zkl.load(folder + file) for file in file_list]
res2 = [zkl.load(folder2 + file) for file in file_list2]
############################################################

folder6 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/soft_weights' \
          '/res_simulation/8nodes/n8sfmf14/'
folder4 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/soft_weights' \
          '/res_simulation/8nodes/n8stmt14/'
folder5 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/soft_weights' \
          '/res_simulation/8nodes/n8stmf14/'

file_list6 = listdir(folder6)
file_list6.sort()
if file_list6[0].startswith('.'):
    file_list6.pop(0)

file_list4 = listdir(folder4)
file_list4.sort()
if file_list4[0].startswith('.'):
    file_list4.pop(0)

file_list5 = listdir(folder5)
file_list5.sort()
if file_list5[0].startswith('.'):
    file_list5.pop(0)

res6 = [zkl.load(folder6 + file) for file in file_list6]
res4 = [zkl.load(folder4 + file) for file in file_list4]
res5 = [zkl.load(folder5 + file) for file in file_list5]


############################################################

folder9 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/soft_weight_no_priority' \
          '/res_simulation/8nodes/n8sfmf14/'
folder7 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/soft_weight_no_priority' \
          '/res_simulation/8nodes/n8stmt14/'
folder8 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/soft_weight_no_priority' \
          '/res_simulation/8nodes/n8stmf14/'

file_list9 = listdir(folder9)
file_list9.sort()
if file_list9[0].startswith('.'):
    file_list9.pop(0)

file_list7 = listdir(folder7)
file_list7.sort()
if file_list7[0].startswith('.'):
    file_list7.pop(0)

file_list8 = listdir(folder8)
file_list8.sort()
if file_list8[0].startswith('.'):
    file_list8.pop(0)

res9 = [zkl.load(folder9 + file) for file in file_list9]
res7 = [zkl.load(folder7 + file) for file in file_list7]
res8 = [zkl.load(folder8 + file) for file in file_list8]

############################################################

folder12 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/ringmore' \
          '/res_simulation/8nodes/'
# folder10 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/soft_weight_no_priority' \
#           '/res_simulation/8nodes/n8stmt14/'
# folder11 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/soft_weight_no_priority' \
#           '/res_simulation/8nodes/n8stmf14/'

file_list12 = listdir(folder12)
file_list12.sort()
if file_list12[0].startswith('.'):
    file_list12.pop(0)

res12 = [zkl.load(folder12 + file) for file in file_list12]
# res10 = [zkl.load(folder10 + file) for file in file_list10]
# res11 = [zkl.load(folder11 + file) for file in file_list11]
############################################################

folder13 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/res_simulation/8nodes/'


file_list13 = listdir(folder13)
file_list13.sort()
if file_list13[0].startswith('.'):
    file_list13.pop(0)



res13 = [zkl.load(folder13 + file) for file in file_list13]

############################################################

folder14 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u2/'


file_list14 = listdir(folder14)
file_list14.sort()
if file_list14[0].startswith('.'):
    file_list14.pop(0)



res14 = [zkl.load(folder14 + file) for file in file_list14]

############################################################

folder15 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u3/'


file_list15 = listdir(folder15)
file_list15.sort()
if file_list15[0].startswith('.'):
    file_list15.pop(0)



res15 = [zkl.load(folder15 + file) for file in file_list15]

############################################################

folder16 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u4/'


file_list16 = listdir(folder16)
file_list16.sort()
if file_list16[0].startswith('.'):
    file_list16.pop(0)



res16 = [zkl.load(folder16 + file) for file in file_list16]

############################################################

folder17 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u5/'


file_list17 = listdir(folder17)
file_list17.sort()
if file_list17[0].startswith('.'):
    file_list17.pop(0)



res17 = [zkl.load(folder17 + file) for file in file_list17]

############################################################

folder18 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u6/'


file_list18 = listdir(folder18)
file_list18.sort()
if file_list18[0].startswith('.'):
    file_list18.pop(0)

res18 = [zkl.load(folder18 + file) for file in file_list18]

############################################################

folder19 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u7/'


file_list19 = listdir(folder19)
file_list19.sort()
if file_list19[0].startswith('.'):
    file_list19.pop(0)
res19 = [zkl.load(folder19 + file) for file in file_list19]
############################################################

folder20 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u8/'


file_list20 = listdir(folder20)
file_list20.sort()
if file_list20[0].startswith('.'):
    file_list20.pop(0)


res20 = [zkl.load(folder20 + file) for file in file_list20]
############################################################

folder21 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u9/'


file_list21 = listdir(folder21)
file_list21.sort()
if file_list21[0].startswith('.'):
    file_list21.pop(0)
res21 = [zkl.load(folder21 + file) for file in file_list21]
############################################################

folder22 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u10/'


file_list22 = listdir(folder22)
file_list22.sort()
if file_list22[0].startswith('.'):
    file_list22.pop(0)
res22 = [zkl.load(folder22 + file) for file in file_list22]
############################################################

folder23 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u11/'


file_list23 = listdir(folder23)
file_list23.sort()
if file_list23[0].startswith('.'):
    file_list23.pop(0)
res23 = [zkl.load(folder23 + file) for file in file_list23]
############################################################

folder24 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u12/'


file_list24 = listdir(folder24)
file_list24.sort()
if file_list24[0].startswith('.'):
    file_list24.pop(0)
res24 = [zkl.load(folder24 + file) for file in file_list24]
############################################################
folder25 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u13/'


file_list25 = listdir(folder25)
file_list25.sort()
if file_list25[0].startswith('.'):
    file_list25.pop(0)
res25 = [zkl.load(folder25 + file) for file in file_list25]
############################################################
folder26 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u14/'


file_list26 = listdir(folder26)
file_list26.sort()
if file_list26[0].startswith('.'):
    file_list26.pop(0)
res26 = [zkl.load(folder26 + file) for file in file_list26]
############################################################
folder27 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u15/'


file_list27 = listdir(folder27)
file_list27.sort()
if file_list27[0].startswith('.'):
    file_list27.pop(0)
res27 = [zkl.load(folder27 + file) for file in file_list27]
############################################################
folder28 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u16/'


file_list28 = listdir(folder28)
file_list28.sort()
if file_list28[0].startswith('.'):
    file_list28.pop(0)
res28 = [zkl.load(folder28 + file) for file in file_list28]
############################################################
folder29 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u17/'


file_list29 = listdir(folder29)
file_list29.sort()
if file_list29[0].startswith('.'):
    file_list29.pop(0)
res29 = [zkl.load(folder29 + file) for file in file_list29]
############################################################
folder30 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u18/'


file_list30 = listdir(folder30)
file_list30.sort()
if file_list30[0].startswith('.'):
    file_list30.pop(0)
res30 = [zkl.load(folder30 + file) for file in file_list30]
############################################################
folder31 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u19/'


file_list31 = listdir(folder31)
file_list31.sort()
if file_list31[0].startswith('.'):
    file_list31.pop(0)
res31 = [zkl.load(folder31 + file) for file in file_list31]
############################################################
folder32 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/gt_density' \
          '/priority2/8nodes/u20/'


file_list32 = listdir(folder32)
file_list32.sort()
if file_list32[0].startswith('.'):
    file_list32.pop(0)
res32 = [zkl.load(folder32 + file) for file in file_list32]




G1_opt_error_GT_om = []
G1_opt_error_GT_com = []
Gu_opt_errors_network_GT_U_om = []
Gu_opt_errors_network_GT_U_com = []
Gu_opt_errors_g_estimated_om = []
Gu_opt_errors_g_estimated_com = []


# G1_opt_error_GT_om_WRT_GuOptVsGTu = []
# G1_opt_error_GT_com_WRT_GuOptVsGTu = []
# Gu_opt_errors_network_GT_U_om_WRT_GuOptVsGTu = []
# Gu_opt_errors_network_GT_U_com_WRT_GuOptVsGTu = []
# Gu_opt_errors_g_estimated_om_WRT_GuOptVsGTu = []
# Gu_opt_errors_g_estimated_com_WRT_GuOptVsGTu = []
#
#
# G1_opt_error_GT_om_WRT_G1OptVsGT = []
# G1_opt_error_GT_com_WRT_G1OptVsGT = []
# Gu_opt_errors_network_GT_U_om_WRT_G1OptVsGT = []
# Gu_opt_errors_network_GT_U_com_WRT_G1OptVsGT = []
# Gu_opt_errors_g_estimated_om_WRT_G1OptVsGT = []
# Gu_opt_errors_g_estimated_com_WRT_G1OptVsGT = []

'''
    for item in res3+res+res2:
        ErrVs.extend([ 'GuVsGTu', 'GuVsGest','G1VsGT'])
        WRT.extend(['GuOptVsGest','GuOptVsGest','GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm','omm'])
        weights_scheme.extend(['hard','hard','hard'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['hard', 'hard', 'hard'])
#################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['hard', 'hard', 'hard'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['hard', 'hard', 'hard'])
#################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['hard', 'hard', 'hard'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['hard', 'hard', 'hard'])

    for item in res6 + res4 + res5:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['soft', 'soft', 'soft'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['soft', 'soft', 'soft'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['soft', 'soft', 'soft'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['soft', 'soft', 'soft'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['soft', 'soft', 'soft'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['soft', 'soft', 'soft'])


    for item in res9 + res7 + res8:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['same_priority', 'same_priority', 'same_priority'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['same_priority', 'same_priority', 'same_priority'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['same_priority', 'same_priority', 'same_priority'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['same_priority', 'same_priority', 'same_priority'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['same_priority', 'same_priority', 'same_priority'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['same_priority', 'same_priority', 'same_priority'])


    for item in res12:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['ringmore', 'ringmore', 'ringmore'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['ringmore', 'ringmore', 'ringmore'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['ringmore', 'ringmore', 'ringmore'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['ringmore', 'ringmore', 'ringmore'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['ringmore', 'ringmore', 'ringmore'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['ringmore', 'ringmore', 'ringmore'])


    for item in res13:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['gt_density', 'gt_density', 'gt_density'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['gt_density', 'gt_density', 'gt_density'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['gt_density', 'gt_density', 'gt_density'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['gt_density', 'gt_density', 'gt_density'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['gt_density', 'gt_density', 'gt_density'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['gt_density', 'gt_density', 'gt_density'])'''
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


    gis_u2 = []
    for item in res14:
        gis_u2.append(item['general']['g_estimated'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U2', 'U2', 'U2'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U2', 'U2', 'U2'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U2', 'U2', 'U2'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U2', 'U2', 'U2'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U2', 'U2', 'U2'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U2', 'U2', 'U2'])
    gis_u3 = []
    for item in res15:
        gis_u3.append(item['general']['g_estimated'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U3', 'U3', 'U3'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U3', 'U3', 'U3'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U3', 'U3', 'U3'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U3', 'U3', 'U3'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U3', 'U3', 'U3'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U3', 'U3', 'U3'])

    gis_u4 = []
    for item in res16:
        gis_u4.append(item['general']['g_estimated'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U4', 'U4', 'U4'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U4', 'U4', 'U4'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U4', 'U4', 'U4'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U4', 'U4', 'U4'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U4', 'U4', 'U4'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U4', 'U4', 'U4'])
    gis_u5 = []
    for item in res17:
        gis_u5.append(item['general']['g_estimated'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U5', 'U5', 'U5'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U5', 'U5', 'U5'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U5', 'U5', 'U5'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U5', 'U5', 'U5'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U5', 'U5', 'U5'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U5', 'U5', 'U5'])

    gis_u6 = []
    for item in res18:
        gis_u6.append(item['general']['g_estimated'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U6', 'U6', 'U6'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U6', 'U6', 'U6'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U6', 'U6', 'U6'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U6', 'U6', 'U6'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U6', 'U6', 'U6'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U6', 'U6', 'U6'])


    for item in res19:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U7', 'U7', 'U7'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U7', 'U7', 'U7'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U7', 'U7', 'U7'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U7', 'U7', 'U7'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U7', 'U7', 'U7'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U7', 'U7', 'U7'])
    gis = []
    for item in res20:
        gis.append(item['general']['g_estimated'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U8', 'U8', 'U8'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U8', 'U8', 'U8'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U8', 'U8', 'U8'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U8', 'U8', 'U8'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U8', 'U8', 'U8'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U8', 'U8', 'U8'])

    for item in res21:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U9', 'U9', 'U9'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U9', 'U9', 'U9'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U9', 'U9', 'U9'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U9', 'U9', 'U9'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U9', 'U9', 'U9'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U9', 'U9', 'U9'])

    for item in res22:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U10', 'U10', 'U10'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U10', 'U10', 'U10'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U10', 'U10', 'U10'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U10', 'U10', 'U10'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U10', 'U10', 'U10'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U10', 'U10', 'U10'])

    for item in res23:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U11', 'U11', 'U11'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U11', 'U11', 'U11'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U11', 'U11', 'U11'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U11', 'U11', 'U11'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U11', 'U11', 'U11'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U11', 'U11', 'U11'])

    for item in res24:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U12', 'U12', 'U12'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U12', 'U12', 'U12'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U12', 'U12', 'U12'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U12', 'U12', 'U12'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U12', 'U12', 'U12'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U12', 'U12', 'U12'])

    for item in res25:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U13', 'U13', 'U13'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U13', 'U13', 'U13'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U13', 'U13', 'U13'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U13', 'U13', 'U13'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U13', 'U13', 'U13'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U13', 'U13', 'U13'])

    for item in res26:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U14', 'U14', 'U14'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U14', 'U14', 'U14'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U14', 'U14', 'U14'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U14', 'U14', 'U14'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U14', 'U14', 'U14'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U14', 'U14', 'U14'])

    for item in res27:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U15', 'U15', 'U15'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U15', 'U15', 'U15'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U15', 'U15', 'U15'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U15', 'U15', 'U15'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U15', 'U15', 'U15'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U15', 'U15', 'U15'])

    for item in res28:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U16', 'U16', 'U16'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U16', 'U16', 'U16'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U16', 'U16', 'U16'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U16', 'U16', 'U16'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U16', 'U16', 'U16'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U16', 'U16', 'U16'])

    for item in res29:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U17', 'U17', 'U17'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U17', 'U17', 'U17'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U17', 'U17', 'U17'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U17', 'U17', 'U17'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U17', 'U17', 'U17'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U17', 'U17', 'U17'])

    for item in res30:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U18', 'U18', 'U18'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U18', 'U18', 'U18'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U18', 'U18', 'U18'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U18', 'U18', 'U18'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U18', 'U18', 'U18'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U18', 'U18', 'U18'])

    for item in res31:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U19', 'U19', 'U19'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U19', 'U19', 'U19'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U19', 'U19', 'U19'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U19', 'U19', 'U19'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U19', 'U19', 'U19'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U19', 'U19', 'U19'])

    for item in res32:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U20', 'U20', 'U20'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U20', 'U20', 'U20'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U20', 'U20', 'U20'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U20', 'U20', 'U20'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['U20', 'U20', 'U20'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['U20', 'U20', 'U20'])




    df['Err'] = Err
    df['ErrVs'] = ErrVs
    df['ErrType'] = ErrType
    df['WRT'] = WRT
    df['weights_scheme'] = weights_scheme

    sns.set({"xtick.minor.size": 0.2})
    pal = dict(U2="gold", U3="blue",
               U4="maroon", U5="green",U6="red",U7="yellow")
    g = sns.FacetGrid(df, col="WRT", row="ErrType", height=4, aspect=1.5, margin_titles=True)


    def custom_boxplot(*args, **kwargs):
        sns.boxplot(*args, **kwargs, palette='Set1')


    g.map_dataframe(custom_boxplot, x='ErrVs', y='Err', hue='weights_scheme')
    g.add_legend()
    g.set_axis_labels("error type", "normalized error")

    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            ax = g.facet_axis(i, j)
            # ax.xaxis.grid(True, "minor", linewidth=.75)
            # ax.xaxis.grid(True, "major", linewidth=3)
            # ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.set_ylim(0, 1)

    plt.show()
    # plt.savefig("figs/VAR_sim_upto_20_undersampling.png")
