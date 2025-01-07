import os
from brainiak.utils import fmrisim
# from gunfolds.viz import gtool as gt
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
from gunfolds.scripts import bold_function as hrf
sys.path.append('~/tread/py-tetrad')
from py_tetrad.tools import TetradSearch as ts

def parse_arguments(PNUM):
    parser = argparse.ArgumentParser(description='Run settings.')
    parser.add_argument("-c", "--CAPSIZE", default=0,
                        help="stop traversing after growing equivalence class to this size.", type=int)
    parser.add_argument("-b", "--BATCH", default=1, help="slurm batch.", type=int)
    parser.add_argument("-p", "--PNUM", default=PNUM, help="number of CPUs in machine.", type=int)
    parser.add_argument("-r", "--SNR", default=1, help="Signal to noise ratio", type=int)
    parser.add_argument("-n", "--NET", default=1, help="number of simple network", type=int)
    parser.add_argument("-l", "--MINLINK", default=5, help=" lower threshold transition matrix abs value x1000", type=int)
    parser.add_argument("-z", "--NOISE", default=10, help="noise str multiplied by 100", type=int)
    parser.add_argument("-s", "--SCC", default="f", help="true to use SCC structure, false to not", type=str)
    parser.add_argument("-m", "--SCCMEMBERS", default="f",
                        help="true for using g_estimate SCC members, false for using "
                             "GT SCC members", type=str)
    parser.add_argument("-u", "--UNDERSAMPLING", default=75, help="sampling rate in generated data", type=int)
    parser.add_argument("-x", "--MAXU", default=8, help="maximum number of undersampling to look for solution.",
                        type=int)
    parser.add_argument("-t", "--CONCAT", default="t", help="true to use concat data, false to not", type=str)
    parser.add_argument("-a", "--ALPHA", default=50, help="alpha_level for PC multiplied by 1000", type=int)
    parser.add_argument("-y", "--PRIORITY", default="11112", help="string of priorities", type=str)
    parser.add_argument("-o", "--METHOD", default="FASK", help="method to run", type=str)
    return parser.parse_args()

def convert_str_to_bool(args):
    args.SCC = bool(strtobool(args.SCC))
    args.SCCMEMBERS = bool(strtobool(args.SCCMEMBERS))
    args.CONCAT = bool(strtobool(args.CONCAT))
    args.NOISE = args.NOISE / 100
    args.ALPHA = args.ALPHA / 1000
    priprities = []
    for char in args.PRIORITY:
        priprities.append(int(char))
    args.PRIORITY = priprities
    return args


# Define the functions
def MVGC(args, network_GT):
    path = os.path.expanduser(f'~/DataSets_Feedbacks/9_VAR_BOLD_simulation/ri'
            f'ngmore/u{args.UNDERSAMPLING}/MVGC')
    mat_data = loadmat(path + f'/mat_file_{args.BATCH}.mat')['sig']
    for i in range(len(network_GT)):
        mat_data[i, i] = 0
    B = np.zeros((len(network_GT), len(network_GT))).astype(int)
    MVGC = cv.adjs2graph(mat_data, np.zeros((len(network_GT), len(network_GT))))
    return MVGC

def MVAR(args, network_GT):
    path = os.path.expanduser(f'~/DataSets_Feedbacks/9_VAR_BOLD_simulation/ri'
            f'ngmore/u{args.UNDERSAMPLING}/MVAR')
    mat_data = loadmat(path + f'/mat_file_{args.BATCH}.mat')['sig']
    for i in range(len(network_GT)):
        mat_data[i, i] = 0
    B = np.zeros((len(network_GT), len(network_GT))).astype(int)
    MVAR = cv.adjs2graph(mat_data, np.zeros((len(network_GT), len(network_GT))))
    return MVAR

def GIMME(args, network_GT):
    size = len(network_GT)
    path = os.path.expanduser(f'~/DataSets_Feedbacks/9_VAR_BOLD_simulation/'
            f'ringmore/u{args.UNDERSAMPLING}/GIMME'
           f'/data{args.BATCH}/individual/StdErrors/data{args.BATCH}StdErrors.csv')
    with open(path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header if exists
        rows = []
        for row in csv_reader:
            rows.append(row[1:2*size+1])
        mat = np.array(rows, dtype=np.float32)
        matrix1 = np.array(mat[:, 0:size])
        for i in range(len(network_GT)):
            matrix1[i, i] = 0
        matrix2 = np.array(mat[:, size:2*size])
        binary_matrixA = (matrix1 != 0).astype(int)
        binary_matrixB = (matrix2 != 0).astype(int)
    B0 = np.zeros((len(network_GT), len(network_GT))).astype(int)
    GIMME = cv.adjs2graph(binary_matrixA.T, B0)
    return GIMME

def FASK(args, network_GT):
    num = str(args.BATCH) if args.BATCH > 9 else '0' + str(args.BATCH)
    print('reading file:' + num)
    if not args.CONCAT:
        data = pd.read_csv(
            os.path.expanduser(
                f"~/DataSets_Feedbacks/2. Macaque_Networks/Full/data_fslfilter/BOLDfslfilter_{num}.txt"), delimiter='\t')
    else:
        data = pd.read_csv(
            os.path.expanduser(
                f'~/DataSets_Feedbacks/2. Macaque_Networks/Full/data_fslfilter_concat/concat_BOLDfslfilter_{num}.txt'), delimiter='\t')


    # path = os.path.expanduser(f'~/DataSets_Feedbacks/2. Macaque_Networks/Full/data_fslfilter_concat/concat_BOLDfslfilter_{args.BATCH}.txt')
    # data = pd.read_csv(path, delimiter='\t')
    search = ts.TetradSearch(data)
    search.set_verbose(False)
    search.use_sem_bic()
    search.use_fisher_z(alpha=0.0000001)

    search.run_fask(alpha=0.00001, left_right_rule=1)

    graph_string = str(search.get_string())
    # Parse nodes and edges from the input string
    nodes = mf.parse_nodes(graph_string)
    edges = mf.parse_edges(graph_string)

    # Create adjacency matrix
    adj_matrix = mf.create_adjacency_matrix(edges, nodes)
    B = np.zeros((len(network_GT), len(network_GT))).astype(int)
    FASK = cv.adjs2graph(adj_matrix, np.zeros((len(network_GT), len(network_GT))))
    return FASK

def RASL(args, network_GT):
    num = str(args.BATCH) if args.BATCH > 9 else '0' + str(args.BATCH)
    print('reading file:' + num)
    if not args.CONCAT:
        data = pd.read_csv(
            os.path.expanduser(
                f"~/DataSets_Feedbacks/2. Macaque_Networks/Full/data_fslfilter/BOLDfslfilter_{num}.txt"),
            delimiter='\t')
    else:
        data = pd.read_csv(
            os.path.expanduser(
                f'~/DataSets_Feedbacks/2. Macaque_Networks/Full/data_fslfilter_concat/concat_BOLDfslfilter_{num}.txt'),
            delimiter='\t')


    dataframe = pp.DataFrame(data.values)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.000001)
    g_estimated, A, B = cv.Glag2CG(results)
    MAXCOST = 10000
    DD = (np.abs((np.abs(A / np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
    BD = (np.abs((np.abs(B / np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

    r_estimated = drasl([g_estimated], weighted=True, capsize=0, timeout=0,
                        urate=min(args.MAXU, (3 * len(g_estimated) + 1)),
                        dm=[DD],
                        bdm=[BD],
                        scc=True,
                        GT_density=int(1000 * gk.density(network_GT)),
                        edge_weights=args.PRIORITY, pnum=PNUM, optim='optN', selfloop=True)

    print('number of optimal solutions is', len(r_estimated))
    max_f1_score = 0
    for answer in r_estimated:
        res_rasl = bfutils.num2CG(answer[0][0], len(network_GT))
        rasl_sol = mf.precision_recall_all_cycle(res_rasl, network_GT, include_selfloop=include_selfloop)

        curr_f1 = ((rasl_sol['orientation']['F1']))
        # curr_f1 = (rasl_sol['orientation']['F1']) + (rasl_sol['adjacency']['F1']) + (rasl_sol['cycle']['F1'])

        if curr_f1 > max_f1_score:
            max_f1_score = curr_f1
            max_answer = answer

    res_rasl = bfutils.num2CG(max_answer[0][0], len(network_GT))
    return res_rasl

def mRASL(args, network_GT):
    BATCH = args.BATCH*6
    network_GT = zkl.load(os.path.expanduser(f'~/DataSets_Feedbacks/9_VAR_BOLD_simulation/ringmore/u{args.UNDERSAMPLING}/GT/GT{BATCH}.zkl'))
    MAXCOST = 1000
    N = len(network_GT)
    base_g = {i: {} for i in range(1, N + 1)}
    base_DD = np.zeros((N,N)).astype(int)
    base_BD = np.zeros((N,N)).astype(int)
    g_est_list = []
    DD_list = []
    BD_list = []
    for i in range(6):
        path = os.path.expanduser(f'~/DataSets_Feedbacks/9_VAR_BOLD_simulation/ringmore/u{args.UNDERSAMPLING}/txtSTD/data{BATCH-i}.txt')
        data = pd.read_csv(path, delimiter='\t')
        # dataframe = pp.DataFrame(data.values)
        # cond_ind_test = ParCorr()
        # pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        # results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05)
        # g_estimated, A, B = cv.Glag2CG(results)
        # bold_out, _ = hrf.compute_bold_signals(data.values)
        g_estimated, A, B = lm.data2graph(data.values.T, th=0.03)
        DD = (np.abs((np.abs(A / np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
        BD = (np.abs((np.abs(B / np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

        g_est_list.append(g_estimated)
        DD_list.append(DD)
        BD_list.append(BD)

        base_g = mf.update_base_graph(base_g, g_estimated)
        base_DD, base_BD = mf.update_DD_BD(g_estimated, DD, BD, base_DD, base_BD,base_g)


    base_DD = np.where(base_DD < 0, 6000 + base_DD, base_DD)
    base_BD = np.where(base_BD < 0, 6000 + base_BD, base_BD)
    r_estimated = drasl(g_est_list, weighted=True, capsize=0, timeout=0,
                        urate=min(args.MAXU, (3 * len(g_estimated) + 1)),
                        dm=DD_list,
                        bdm=BD_list,
                        scc=False,
                        GT_density=int(1000 * gk.density(network_GT)),
                        edge_weights=args.PRIORITY, pnum=PNUM, optim='optN', selfloop=False)

    print('number of optimal solutions is', len(r_estimated))
    max_f1_score = 0
    for answer in r_estimated:
        res_rasl = bfutils.num2CG(answer[0][0], len(network_GT))
        rasl_sol = mf.precision_recall_all_cycle(res_rasl, network_GT, include_selfloop=include_selfloop)

        curr_f1 = ((rasl_sol['orientation']['F1']))
        # curr_f1 = (rasl_sol['orientation']['F1']) + (rasl_sol['adjacency']['F1']) + (rasl_sol['cycle']['F1'])

        if curr_f1 > max_f1_score:
            max_f1_score = curr_f1
            max_answer = answer

    res_rasl = bfutils.num2CG(max_answer[0][0], len(network_GT))
    return res_rasl

def initialize_metrics():
    return {
        'Precision_O': [], 'Recall_O': [], 'F1_O': [],
        'Precision_A': [], 'Recall_A': [], 'F1_A': [],
        'Precision_C': [], 'Recall_C': [], 'F1_C': []
    }

def convert_to_mat(args):
    data = zkl.load(f'datasets/VAR_BOLD_standatd_Gis_ringmore_V_ruben_undersampled_by_{args.UNDERSAMPLING}.zkl')
    for i, dd in enumerate(data, start=1):
        folder = os.path.expanduser(f'~/DataSets_Feedbacks/9_VAR_BOLD_simulation/ringmore/u{args.UNDERSAMPLING}/mat')
        if not os.path.exists(folder):
            os.makedirs(folder)
        savemat(folder + '/expo_to_mat_' + str(i) + '.mat', {'dd': dd['data']})

        print('file saved to :' + folder + '/expo_to_mat_' + str(i) + '.mat')

def convert_to_txt(args):
    data = zkl.load(f'datasets/VAR_BOLD_standatd_Gis_ringmore_V_ruben_undersampled_by_{args.UNDERSAMPLING}.zkl')
    for i, dd in enumerate(data, start=1):
        folder = os.path.expanduser(f'~/DataSets_Feedbacks/9_VAR_BOLD_simulation/ringmore/u{args.UNDERSAMPLING}/txt')
        if not os.path.exists(folder):
            os.makedirs(folder)
        gt_folder = f'{folder}/../GT'
        if not os.path.exists(gt_folder):
            os.makedirs(gt_folder)
        zkl.save(dd['GT'], f'{gt_folder}/GT{i}.zkl')

        data_scaled = dd['data'] / dd['data'].max()

        ### zero mean and std = 1

        # variances = np.var(dd[1], axis=1, ddof=0)
        # std_devs = np.sqrt(variances)
        # normalized_array = dd[1] / std_devs[:, np.newaxis]
        # means = np.mean(normalized_array, axis=1)
        # zero_mean_array = normalized_array - means[:, np.newaxis]

        header = '\t'.join([f'X{j + 1}' for j in range(data_scaled.shape[0])])

        with open(f'{folder}/data{i}.txt', 'w') as f:
            # Write the header
            f.write(header + '\n')

            # Write the data, one column per line
            for col in range(data_scaled.shape[1]):
                line = '\t'.join(map(str, data_scaled[:, col]))
                f.write(line + '\n')

        print('file saved to :' + f'{folder}/data{i}.txt')

def run_analysis(args,network_GT,include_selfloop):
    metrics = {key: {args.UNDERSAMPLING: initialize_metrics()} for key in [args.METHOD]}

    for method in metrics.keys():
        # loading = f'datasets/{method}/net{args.NET}' \
        #           f'_undersampled_by_{args.UNDERSAMPLING}_batch{args.BATCH}.*'
        # if not glob.glob(loading):
        #     save_dataset(args)

        result = globals()[method](args, network_GT)
        print(f"Result from {method}: {result}")
        normal_GT = mf.precision_recall_all_cycle(result, network_GT, include_selfloop=include_selfloop)
        metrics[method][args.UNDERSAMPLING]['Precision_O'].append(normal_GT['orientation']['precision'])
        metrics[method][args.UNDERSAMPLING]['Recall_O'].append(normal_GT['orientation']['recall'])
        metrics[method][args.UNDERSAMPLING]['F1_O'].append(normal_GT['orientation']['F1'])

        metrics[method][args.UNDERSAMPLING]['Precision_A'].append(normal_GT['adjacency']['precision'])
        metrics[method][args.UNDERSAMPLING]['Recall_A'].append(normal_GT['adjacency']['recall'])
        metrics[method][args.UNDERSAMPLING]['F1_A'].append(normal_GT['adjacency']['F1'])

        metrics[method][args.UNDERSAMPLING]['Precision_C'].append(normal_GT['cycle']['precision'])
        metrics[method][args.UNDERSAMPLING]['Recall_C'].append(normal_GT['cycle']['recall'])
        metrics[method][args.UNDERSAMPLING]['F1_C'].append(normal_GT['cycle']['F1'])

    print(metrics)
    if not os.path.exists('VAR_ringmore_v3'):
        os.makedirs('VAR_ringmore_v3')
    filename = f'VAR_ringmore_v3/VAR_{args.METHOD}_BOLD_ruben_ringmore_undersampled_by_{args.UNDERSAMPLING}_batch_{args.BATCH}.zkl'
    zkl.save(metrics,filename)
    print('file saved to :' + filename)


def parse_text_to_adjacency(text):
    """
    Parses a textual representation of nodes and edges to create a 2D binary adjacency array.

    Args:
        text (str): The input text containing node definitions and edges.

    Returns:
        np.ndarray: A binary adjacency matrix.
    """
    # Split the text into lines
    lines = text.strip().split('\n')

    # Extract node names from the "Graph Nodes" line
    nodes = None
    for line in lines:
        if line.startswith("Graph Nodes:"):
            nodes_line = line.split(':', 1)[1].strip()
            nodes = [node.strip() for node in nodes_line.split(',')]
            break
    if nodes is None:
        raise ValueError("No nodes found in the input text.")

    # Map node names to indices
    node_to_index = {node: i for i, node in enumerate(nodes)}

    # Initialize an adjacency matrix
    n = len(nodes)
    adjacency_matrix = np.zeros((n, n), dtype=int)

    # Process edges
    edges_start = False
    for line in lines:
        if line.startswith("Graph Edges:"):
            edges_start = True
            continue

        if edges_start and '-->' in line:
            # Parse edge, e.g., "1. X2 --> X1"
            try:
                source, target = line.split('-->')
                source = source.split('.')[-1].strip()  # Extract after the number and period
                target = target.strip()

                # Update adjacency matrix
                adjacency_matrix[node_to_index[source], node_to_index[target]] = 1
            except KeyError as e:
                print(f"Warning: Edge references unknown node '{e.args[0]}'.")

    return adjacency_matrix

# Example usage
text = """
Graph Nodes:X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23,X24,X25,X26,X27,X28,X29,X30,X31,X32,X33,X34,X35,X36,X37,X38,X39,X40,X41,X42,X43,X44,X45,X46,X47,X48,X49,X50,X51,X52,X53,X54,X55,X56,X57,X58,X59,X60,X61,X62,X63,X64,X65,X66,X67,X68,X69,X70,X71,X72,X73,X74,X75,X76,X77,X78,X79,X80,X81,X82,X83,X84,X85,X86,X87,X88,X89,X90,X91

Graph Edges:
1. X2 --> X1
2. X3 --> X1
3. X4 --> X1
4. X5 --> X1
5. X6 --> X1
6. X7 --> X1
7. X8 --> X1
8. X9 --> X1
9. X10 --> X1
10. X11 --> X1
11. X12 --> X1
12. X13 --> X1
13. X14 --> X1
14. X15 --> X1
15. X16 --> X1
16. X17 --> X1
17. X18 --> X1
18. X19 --> X1
19. X20 --> X1
20. X21 --> X1
21. X22 --> X1
22. X23 --> X1
23. X24 --> X1
24. X25 --> X1
25. X26 --> X1
26. X27 --> X1
27. X28 --> X1
28. X30 --> X1
29. X31 --> X1
30. X35 --> X1
31. X36 --> X1
32. X37 --> X1
33. X38 --> X1
34. X39 --> X1
35. X1 --> X2
36. X3 --> X2
37. X4 --> X2
38. X5 --> X2
39. X6 --> X2
40. X7 --> X2
41. X8 --> X2
42. X9 --> X2
43. X10 --> X2
44. X11 --> X2
45. X12 --> X2
46. X13 --> X2
47. X14 --> X2
48. X15 --> X2
49. X16 --> X2
50. X17 --> X2
51. X18 --> X2
52. X19 --> X2
53. X20 --> X2
54. X21 --> X2
55. X22 --> X2
56. X23 --> X2
57. X24 --> X2
58. X25 --> X2
59. X26 --> X2
60. X27 --> X2
61. X28 --> X2
62. X29 --> X2
63. X30 --> X2
64. X31 --> X2
65. X32 --> X2
66. X33 --> X2
67. X34 --> X2
68. X35 --> X2
69. X1 --> X5
70. X2 --> X5
71. X3 --> X5
72. X4 --> X5
73. X6 --> X5
74. X7 --> X5
75. X8 --> X5
76. X9 --> X5
77. X10 --> X5
78. X11 --> X5
79. X12 --> X5
80. X13 --> X5
81. X14 --> X5
82. X15 --> X5
83. X16 --> X5
84. X17 --> X5
85. X18 --> X5
86. X19 --> X5
87. X20 --> X5
88. X21 --> X5
89. X22 --> X5
90. X23 --> X5
91. X24 --> X5
92. X25 --> X5
93. X26 --> X5
94. X27 --> X5
95. X29 --> X5
96. X31 --> X5
97. X32 --> X5
98. X40 --> X5
99. X41 --> X5
100. X42 --> X5
101. X43 --> X5
102. X44 --> X5
103. X45 --> X5
104. X1 --> X9
105. X2 --> X9
106. X3 --> X9
107. X4 --> X9
108. X5 --> X9
109. X6 --> X9
110. X7 --> X9
111. X8 --> X9
112. X10 --> X9
113. X11 --> X9
114. X12 --> X9
115. X13 --> X9
116. X14 --> X9
117. X15 --> X9
118. X16 --> X9
119. X17 --> X9
120. X18 --> X9
121. X19 --> X9
122. X20 --> X9
123. X21 --> X9
124. X22 --> X9
125. X23 --> X9
126. X24 --> X9
127. X25 --> X9
128. X26 --> X9
129. X27 --> X9
130. X30 --> X9
131. X32 --> X9
132. X35 --> X9
133. X36 --> X9
134. X37 --> X9
135. X38 --> X9
136. X39 --> X9
137. X40 --> X9
138. X41 --> X9
139. X42 --> X9
140. X43 --> X9
141. X44 --> X9
142. X47 --> X9
143. X49 --> X9
144. X56 --> X9
145. X58 --> X9
146. X60 --> X9
147. X61 --> X9
148. X62 --> X9
149. X64 --> X9
150. X65 --> X9
151. X66 --> X9
152. X67 --> X9
153. X70 --> X9
154. X71 --> X9
155. X74 --> X9
156. X75 --> X9
157. X76 --> X9
158. X81 --> X9
159. X83 --> X9
160. X89 --> X9
161. X1 --> X10
162. X5 --> X10
163. X7 --> X10
164. X9 --> X10
165. X11 --> X10
166. X12 --> X10
167. X13 --> X10
168. X14 --> X10
169. X15 --> X10
170. X16 --> X10
171. X17 --> X10
172. X18 --> X10
173. X19 --> X10
174. X20 --> X10
175. X21 --> X10
176. X22 --> X10
177. X23 --> X10
178. X24 --> X10
179. X25 --> X10
180. X26 --> X10
181. X27 --> X10
182. X28 --> X10
183. X29 --> X10
184. X30 --> X10
185. X31 --> X10
186. X32 --> X10
187. X33 --> X10
188. X35 --> X10
189. X38 --> X10
190. X40 --> X10
191. X41 --> X10
192. X42 --> X10
193. X43 --> X10
194. X44 --> X10
195. X45 --> X10
196. X50 --> X10
197. X51 --> X10
198. X52 --> X10
199. X56 --> X10
200. X57 --> X10
201. X58 --> X10
202. X60 --> X10
203. X61 --> X10
204. X64 --> X10
205. X65 --> X10
206. X66 --> X10
207. X67 --> X10
208. X68 --> X10
209. X69 --> X10
210. X70 --> X10
211. X71 --> X10
212. X72 --> X10
213. X74 --> X10
214. X75 --> X10
215. X77 --> X10
216. X78 --> X10
217. X81 --> X10
218. X82 --> X10
219. X83 --> X10
220. X84 --> X10
221. X85 --> X10
222. X86 --> X10
223. X87 --> X10
224. X89 --> X10
225. X90 --> X10
226. X91 --> X10
227. X6 --> X11
228. X7 --> X11
229. X10 --> X11
230. X12 --> X11
231. X13 --> X11
232. X14 --> X11
233. X16 --> X11
234. X17 --> X11
235. X20 --> X11
236. X22 --> X11
237. X23 --> X11
238. X25 --> X11
239. X26 --> X11
240. X27 --> X11
241. X28 --> X11
242. X29 --> X11
243. X30 --> X11
244. X33 --> X11
245. X35 --> X11
246. X38 --> X11
247. X42 --> X11
248. X43 --> X11
249. X50 --> X11
250. X51 --> X11
251. X52 --> X11
252. X56 --> X11
253. X59 --> X11
254. X61 --> X11
255. X64 --> X11
256. X65 --> X11
257. X66 --> X11
258. X67 --> X11
259. X69 --> X11
260. X70 --> X11
261. X71 --> X11
262. X72 --> X11
263. X74 --> X11
264. X75 --> X11
265. X77 --> X11
266. X80 --> X11
267. X82 --> X11
268. X83 --> X11
269. X84 --> X11
270. X86 --> X11
271. X87 --> X11
272. X90 --> X11
273. X1 --> X12
274. X4 --> X12
275. X5 --> X12
276. X7 --> X12
277. X8 --> X12
278. X9 --> X12
279. X10 --> X12
280. X11 --> X12
281. X13 --> X12
282. X14 --> X12
283. X15 --> X12
284. X16 --> X12
285. X17 --> X12
286. X18 --> X12
287. X19 --> X12
288. X20 --> X12
289. X21 --> X12
290. X22 --> X12
291. X23 --> X12
292. X24 --> X12
293. X25 --> X12
294. X26 --> X12
295. X27 --> X12
296. X28 --> X12
297. X29 --> X12
298. X30 --> X12
299. X31 --> X12
300. X32 --> X12
301. X33 --> X12
302. X34 --> X12
303. X35 --> X12
304. X36 --> X12
305. X37 --> X12
306. X38 --> X12
307. X40 --> X12
308. X41 --> X12
309. X42 --> X12
310. X43 --> X12
311. X44 --> X12
312. X46 --> X12
313. X48 --> X12
314. X49 --> X12
315. X50 --> X12
316. X51 --> X12
317. X52 --> X12
318. X54 --> X12
319. X56 --> X12
320. X58 --> X12
321. X61 --> X12
322. X62 --> X12
323. X64 --> X12
324. X65 --> X12
325. X66 --> X12
326. X67 --> X12
327. X70 --> X12
328. X71 --> X12
329. X74 --> X12
330. X75 --> X12
331. X76 --> X12
332. X80 --> X12
333. X82 --> X12
334. X83 --> X12
335. X87 --> X12
336. X89 --> X12
337. X1 --> X17
338. X2 --> X17
339. X3 --> X17
340. X4 --> X17
341. X5 --> X17
342. X6 --> X17
343. X7 --> X17
344. X8 --> X17
345. X9 --> X17
346. X10 --> X17
347. X11 --> X17
348. X12 --> X17
349. X13 --> X17
350. X14 --> X17
351. X15 --> X17
352. X16 --> X17
353. X18 --> X17
354. X19 --> X17
355. X20 --> X17
356. X21 --> X17
357. X22 --> X17
358. X23 --> X17
359. X24 --> X17
360. X25 --> X17
361. X26 --> X17
362. X27 --> X17
363. X28 --> X17
364. X29 --> X17
365. X31 --> X17
366. X32 --> X17
367. X35 --> X17
368. X36 --> X17
369. X37 --> X17
370. X38 --> X17
371. X40 --> X17
372. X42 --> X17
373. X43 --> X17
374. X44 --> X17
375. X45 --> X17
376. X47 --> X17
377. X50 --> X17
378. X51 --> X17
379. X53 --> X17
380. X54 --> X17
381. X56 --> X17
382. X57 --> X17
383. X58 --> X17
384. X60 --> X17
385. X62 --> X17
386. X63 --> X17
387. X67 --> X17
388. X71 --> X17
389. X73 --> X17
390. X75 --> X17
391. X77 --> X17
392. X78 --> X17
393. X81 --> X17
394. X84 --> X17
395. X89 --> X17
396. X1 --> X18
397. X3 --> X18
398. X4 --> X18
399. X5 --> X18
400. X6 --> X18
401. X7 --> X18
402. X8 --> X18
403. X9 --> X18
404. X10 --> X18
405. X11 --> X18
406. X12 --> X18
407. X13 --> X18
408. X14 --> X18
409. X15 --> X18
410. X16 --> X18
411. X17 --> X18
412. X19 --> X18
413. X20 --> X18
414. X21 --> X18
415. X22 --> X18
416. X23 --> X18
417. X24 --> X18
418. X25 --> X18
419. X26 --> X18
420. X27 --> X18
421. X28 --> X18
422. X29 --> X18
423. X30 --> X18
424. X31 --> X18
425. X32 --> X18
426. X40 --> X18
427. X41 --> X18
428. X42 --> X18
429. X43 --> X18
430. X44 --> X18
431. X45 --> X18
432. X46 --> X18
433. X47 --> X18
434. X48 --> X18
435. X49 --> X18
436. X50 --> X18
437. X51 --> X18
438. X52 --> X18
439. X53 --> X18
440. X54 --> X18
441. X55 --> X18
442. X56 --> X18
443. X57 --> X18
444. X58 --> X18
445. X59 --> X18
446. X60 --> X18
447. X61 --> X18
448. X62 --> X18
449. X63 --> X18
450. X64 --> X18
451. X65 --> X18
452. X66 --> X18
453. X67 --> X18
454. X68 --> X18
455. X69 --> X18
456. X70 --> X18
457. X1 --> X23
458. X3 --> X23
459. X4 --> X23
460. X5 --> X23
461. X6 --> X23
462. X7 --> X23
463. X8 --> X23
464. X9 --> X23
465. X10 --> X23
466. X11 --> X23
467. X12 --> X23
468. X13 --> X23
469. X14 --> X23
470. X15 --> X23
471. X16 --> X23
472. X17 --> X23
473. X18 --> X23
474. X19 --> X23
475. X20 --> X23
476. X21 --> X23
477. X22 --> X23
478. X24 --> X23
479. X25 --> X23
480. X26 --> X23
481. X27 --> X23
482. X28 --> X23
483. X29 --> X23
484. X31 --> X23
485. X32 --> X23
486. X33 --> X23
487. X38 --> X23
488. X40 --> X23
489. X41 --> X23
490. X42 --> X23
491. X44 --> X23
492. X45 --> X23
493. X50 --> X23
494. X51 --> X23
495. X52 --> X23
496. X55 --> X23
497. X56 --> X23
498. X57 --> X23
499. X58 --> X23
500. X62 --> X23
501. X64 --> X23
502. X65 --> X23
503. X67 --> X23
504. X68 --> X23
505. X69 --> X23
506. X70 --> X23
507. X71 --> X23
508. X73 --> X23
509. X80 --> X23
510. X84 --> X23
511. X85 --> X23
512. X87 --> X23
513. X91 --> X23
514. X1 --> X31
515. X2 --> X31
516. X3 --> X31
517. X4 --> X31
518. X5 --> X31
519. X6 --> X31
520. X7 --> X31
521. X8 --> X31
522. X9 --> X31
523. X10 --> X31
524. X11 --> X31
525. X12 --> X31
526. X13 --> X31
527. X14 --> X31
528. X15 --> X31
529. X16 --> X31
530. X17 --> X31
531. X18 --> X31
532. X19 --> X31
533. X20 --> X31
534. X21 --> X31
535. X22 --> X31
536. X23 --> X31
537. X24 --> X31
538. X25 --> X31
539. X26 --> X31
540. X27 --> X31
541. X28 --> X31
542. X29 --> X31
543. X30 --> X31
544. X32 --> X31
545. X33 --> X31
546. X34 --> X31
547. X35 --> X31
548. X36 --> X31
549. X37 --> X31
550. X38 --> X31
551. X40 --> X31
552. X41 --> X31
553. X42 --> X31
554. X43 --> X31
555. X44 --> X31
556. X45 --> X31
557. X46 --> X31
558. X47 --> X31
559. X48 --> X31
560. X49 --> X31
561. X50 --> X31
562. X51 --> X31
563. X52 --> X31
564. X54 --> X31
565. X55 --> X31
566. X56 --> X31
567. X57 --> X31
568. X58 --> X31
569. X59 --> X31
570. X60 --> X31
571. X61 --> X31
572. X62 --> X31
573. X63 --> X31
574. X64 --> X31
575. X65 --> X31
576. X66 --> X31
577. X67 --> X31
578. X68 --> X31
579. X69 --> X31
580. X70 --> X31
581. X72 --> X31
582. X73 --> X31
583. X74 --> X31
584. X75 --> X31
585. X76 --> X31
586. X77 --> X31
587. X78 --> X31
588. X79 --> X31
589. X80 --> X31
590. X81 --> X31
591. X82 --> X31
592. X83 --> X31
593. X84 --> X31
594. X85 --> X31
595. X86 --> X31
596. X87 --> X31
597. X88 --> X31
598. X89 --> X31
599. X90 --> X31
600. X91 --> X31
601. X1 --> X38
602. X3 --> X38
603. X4 --> X38
604. X5 --> X38
605. X6 --> X38
606. X7 --> X38
607. X8 --> X38
608. X9 --> X38
609. X10 --> X38
610. X11 --> X38
611. X12 --> X38
612. X13 --> X38
613. X14 --> X38
614. X15 --> X38
615. X16 --> X38
616. X17 --> X38
617. X19 --> X38
618. X20 --> X38
619. X23 --> X38
620. X24 --> X38
621. X25 --> X38
622. X26 --> X38
623. X27 --> X38
624. X28 --> X38
625. X29 --> X38
626. X30 --> X38
627. X31 --> X38
628. X32 --> X38
629. X33 --> X38
630. X34 --> X38
631. X35 --> X38
632. X36 --> X38
633. X37 --> X38
634. X40 --> X38
635. X41 --> X38
636. X42 --> X38
637. X43 --> X38
638. X44 --> X38
639. X45 --> X38
640. X46 --> X38
641. X47 --> X38
642. X48 --> X38
643. X49 --> X38
644. X50 --> X38
645. X51 --> X38
646. X52 --> X38
647. X54 --> X38
648. X55 --> X38
649. X56 --> X38
650. X57 --> X38
651. X58 --> X38
652. X59 --> X38
653. X60 --> X38
654. X61 --> X38
655. X62 --> X38
656. X63 --> X38
657. X64 --> X38
658. X65 --> X38
659. X66 --> X38
660. X67 --> X38
661. X68 --> X38
662. X69 --> X38
663. X70 --> X38
664. X72 --> X38
665. X73 --> X38
666. X74 --> X38
667. X75 --> X38
668. X76 --> X38
669. X77 --> X38
670. X78 --> X38
671. X79 --> X38
672. X80 --> X38
673. X81 --> X38
674. X82 --> X38
675. X83 --> X38
676. X84 --> X38
677. X85 --> X38
678. X86 --> X38
679. X87 --> X38
680. X88 --> X38
681. X1 --> X40
682. X2 --> X40
683. X4 --> X40
684. X5 --> X40
685. X7 --> X40
686. X8 --> X40
687. X9 --> X40
688. X10 --> X40
689. X11 --> X40
690. X12 --> X40
691. X13 --> X40
692. X14 --> X40
693. X15 --> X40
694. X16 --> X40
695. X17 --> X40
696. X18 --> X40
697. X19 --> X40
698. X20 --> X40
699. X22 --> X40
700. X23 --> X40
701. X24 --> X40
702. X25 --> X40
703. X26 --> X40
704. X27 --> X40
705. X28 --> X40
706. X29 --> X40
707. X31 --> X40
708. X35 --> X40
709. X36 --> X40
710. X37 --> X40
711. X38 --> X40
712. X39 --> X40
713. X41 --> X40
714. X42 --> X40
715. X43 --> X40
716. X44 --> X40
717. X45 --> X40
718. X46 --> X40
719. X47 --> X40
720. X48 --> X40
721. X49 --> X40
722. X50 --> X40
723. X54 --> X40
724. X56 --> X40
725. X57 --> X40
726. X58 --> X40
727. X60 --> X40
728. X61 --> X40
729. X62 --> X40
730. X64 --> X40
731. X65 --> X40
732. X66 --> X40
733. X67 --> X40
734. X70 --> X40
735. X75 --> X40
736. X76 --> X40
737. X78 --> X40
738. X79 --> X40
739. X83 --> X40
740. X87 --> X40
741. X1 --> X43
742. X3 --> X43
743. X5 --> X43
744. X7 --> X43
745. X8 --> X43
746. X9 --> X43
747. X10 --> X43
748. X11 --> X43
749. X12 --> X43
750. X13 --> X43
751. X15 --> X43
752. X16 --> X43
753. X17 --> X43
754. X18 --> X43
755. X20 --> X43
756. X21 --> X43
757. X22 --> X43
758. X25 --> X43
759. X26 --> X43
760. X28 --> X43
761. X29 --> X43
762. X30 --> X43
763. X31 --> X43
764. X32 --> X43
765. X33 --> X43
766. X34 --> X43
767. X35 --> X43
768. X36 --> X43
769. X37 --> X43
770. X38 --> X43
771. X39 --> X43
772. X40 --> X43
773. X42 --> X43
774. X44 --> X43
775. X45 --> X43
776. X46 --> X43
777. X47 --> X43
778. X48 --> X43
779. X49 --> X43
780. X50 --> X43
781. X51 --> X43
782. X52 --> X43
783. X54 --> X43
784. X56 --> X43
785. X57 --> X43
786. X58 --> X43
787. X59 --> X43
788. X60 --> X43
789. X61 --> X43
790. X62 --> X43
791. X63 --> X43
792. X64 --> X43
793. X65 --> X43
794. X66 --> X43
795. X67 --> X43
796. X68 --> X43
797. X69 --> X43
798. X70 --> X43
799. X71 --> X43
800. X72 --> X43
801. X73 --> X43
802. X74 --> X43
803. X75 --> X43
804. X76 --> X43
805. X77 --> X43
806. X78 --> X43
807. X79 --> X43
808. X80 --> X43
809. X81 --> X43
810. X82 --> X43
811. X83 --> X43
812. X84 --> X43
813. X85 --> X43
814. X86 --> X43
815. X87 --> X43
816. X3 --> X45
817. X7 --> X45
818. X10 --> X45
819. X11 --> X45
820. X12 --> X45
821. X13 --> X45
822. X14 --> X45
823. X15 --> X45
824. X16 --> X45
825. X19 --> X45
826. X20 --> X45
827. X21 --> X45
828. X22 --> X45
829. X23 --> X45
830. X24 --> X45
831. X25 --> X45
832. X26 --> X45
833. X27 --> X45
834. X28 --> X45
835. X30 --> X45
836. X31 --> X45
837. X32 --> X45
838. X34 --> X45
839. X35 --> X45
840. X36 --> X45
841. X37 --> X45
842. X38 --> X45
843. X40 --> X45
844. X41 --> X45
845. X42 --> X45
846. X43 --> X45
847. X44 --> X45
848. X46 --> X45
849. X48 --> X45
850. X49 --> X45
851. X50 --> X45
852. X51 --> X45
853. X52 --> X45
854. X54 --> X45
855. X55 --> X45
856. X56 --> X45
857. X57 --> X45
858. X58 --> X45
859. X59 --> X45
860. X60 --> X45
861. X61 --> X45
862. X62 --> X45
863. X63 --> X45
864. X64 --> X45
865. X65 --> X45
866. X66 --> X45
867. X67 --> X45
868. X68 --> X45
869. X69 --> X45
870. X70 --> X45
871. X72 --> X45
872. X73 --> X45
873. X74 --> X45
874. X75 --> X45
875. X76 --> X45
876. X77 --> X45
877. X78 --> X45
878. X79 --> X45
879. X80 --> X45
880. X81 --> X45
881. X82 --> X45
882. X83 --> X45
883. X84 --> X45
884. X85 --> X45
885. X86 --> X45
886. X87 --> X45
887. X88 --> X45
888. X1 --> X46
889. X7 --> X46
890. X10 --> X46
891. X11 --> X46
892. X12 --> X46
893. X13 --> X46
894. X16 --> X46
895. X17 --> X46
896. X20 --> X46
897. X24 --> X46
898. X25 --> X46
899. X27 --> X46
900. X28 --> X46
901. X29 --> X46
902. X30 --> X46
903. X34 --> X46
904. X35 --> X46
905. X36 --> X46
906. X37 --> X46
907. X38 --> X46
908. X40 --> X46
909. X42 --> X46
910. X43 --> X46
911. X44 --> X46
912. X45 --> X46
913. X47 --> X46
914. X48 --> X46
915. X49 --> X46
916. X54 --> X46
917. X56 --> X46
918. X58 --> X46
919. X59 --> X46
920. X60 --> X46
921. X62 --> X46
922. X63 --> X46
923. X65 --> X46
924. X68 --> X46
925. X70 --> X46
926. X73 --> X46
927. X74 --> X46
928. X78 --> X46
929. X79 --> X46
930. X81 --> X46
931. X88 --> X46
932. X7 --> X48
933. X11 --> X48
934. X16 --> X48
935. X20 --> X48
936. X25 --> X48
937. X28 --> X48
938. X34 --> X48
939. X35 --> X48
940. X36 --> X48
941. X40 --> X48
942. X41 --> X48
943. X42 --> X48
944. X46 --> X48
945. X47 --> X48
946. X49 --> X48
947. X50 --> X48
948. X53 --> X48
949. X54 --> X48
950. X55 --> X48
951. X56 --> X48
952. X57 --> X48
953. X58 --> X48
954. X59 --> X48
955. X60 --> X48
956. X61 --> X48
957. X62 --> X48
958. X64 --> X48
959. X70 --> X48
960. X75 --> X48
961. X76 --> X48
962. X78 --> X48
963. X79 --> X48
964. X80 --> X48
965. X88 --> X48
966. X1 --> X49
967. X2 --> X49
968. X3 --> X49
969. X4 --> X49
970. X5 --> X49
971. X7 --> X49
972. X8 --> X49
973. X9 --> X49
974. X10 --> X49
975. X11 --> X49
976. X12 --> X49
977. X13 --> X49
978. X14 --> X49
979. X15 --> X49
980. X16 --> X49
981. X17 --> X49
982. X20 --> X49
983. X23 --> X49
984. X24 --> X49
985. X26 --> X49
986. X27 --> X49
987. X28 --> X49
988. X29 --> X49
989. X30 --> X49
990. X31 --> X49
991. X32 --> X49
992. X34 --> X49
993. X35 --> X49
994. X36 --> X49
995. X37 --> X49
996. X38 --> X49
997. X39 --> X49
998. X40 --> X49
999. X41 --> X49
1000. X42 --> X49
1001. X43 --> X49
1002. X44 --> X49
1003. X45 --> X49
1004. X46 --> X49
1005. X47 --> X49
1006. X48 --> X49
1007. X51 --> X49
1008. X53 --> X49
1009. X54 --> X49
1010. X55 --> X49
1011. X56 --> X49
1012. X57 --> X49
1013. X58 --> X49
1014. X59 --> X49
1015. X60 --> X49
1016. X61 --> X49
1017. X62 --> X49
1018. X64 --> X49
1019. X65 --> X49
1020. X66 --> X49
1021. X68 --> X49
1022. X69 --> X49
1023. X70 --> X49
1024. X71 --> X49
1025. X74 --> X49
1026. X75 --> X49
1027. X76 --> X49
1028. X77 --> X49
1029. X78 --> X49
1030. X79 --> X49
1031. X80 --> X49
1032. X83 --> X49
1033. X86 --> X49
1034. X87 --> X49
1035. X88 --> X49
1036. X90 --> X49
1037. X1 --> X51
1038. X5 --> X51
1039. X7 --> X51
1040. X9 --> X51
1041. X10 --> X51
1042. X11 --> X51
1043. X12 --> X51
1044. X13 --> X51
1045. X14 --> X51
1046. X15 --> X51
1047. X16 --> X51
1048. X20 --> X51
1049. X27 --> X51
1050. X28 --> X51
1051. X29 --> X51
1052. X30 --> X51
1053. X31 --> X51
1054. X32 --> X51
1055. X33 --> X51
1056. X35 --> X51
1057. X38 --> X51
1058. X40 --> X51
1059. X41 --> X51
1060. X42 --> X51
1061. X43 --> X51
1062. X44 --> X51
1063. X49 --> X51
1064. X50 --> X51
1065. X52 --> X51
1066. X56 --> X51
1067. X57 --> X51
1068. X58 --> X51
1069. X59 --> X51
1070. X61 --> X51
1071. X63 --> X51
1072. X64 --> X51
1073. X65 --> X51
1074. X66 --> X51
1075. X67 --> X51
1076. X69 --> X51
1077. X70 --> X51
1078. X71 --> X51
1079. X74 --> X51
1080. X75 --> X51
1081. X76 --> X51
1082. X77 --> X51
1083. X80 --> X51
1084. X82 --> X51
1085. X83 --> X51
1086. X84 --> X51
1087. X85 --> X51
1088. X86 --> X51
1089. X87 --> X51
1090. X89 --> X51
1091. X90 --> X51
1092. X91 --> X51
1093. X7 --> X54
1094. X20 --> X54
1095. X22 --> X54
1096. X25 --> X54
1097. X28 --> X54
1098. X29 --> X54
1099. X34 --> X54
1100. X36 --> X54
1101. X40 --> X54
1102. X41 --> X54
1103. X42 --> X54
1104. X45 --> X54
1105. X46 --> X54
1106. X50 --> X54
1107. X53 --> X54
1108. X55 --> X54
1109. X58 --> X54
1110. X59 --> X54
1111. X62 --> X54
1112. X63 --> X54
1113. X64 --> X54
1114. X68 --> X54
1115. X70 --> X54
1116. X72 --> X54
1117. X73 --> X54
1118. X74 --> X54
1119. X76 --> X54
1120. X78 --> X54
1121. X79 --> X54
1122. X81 --> X54
1123. X84 --> X54
1124. X85 --> X54
1125. X87 --> X54
1126. X88 --> X54
1127. X91 --> X54
1128. X7 --> X60
1129. X10 --> X60
1130. X11 --> X60
1131. X12 --> X60
1132. X13 --> X60
1133. X14 --> X60
1134. X16 --> X60
1135. X17 --> X60
1136. X20 --> X60
1137. X25 --> X60
1138. X28 --> X60
1139. X31 --> X60
1140. X32 --> X60
1141. X34 --> X60
1142. X35 --> X60
1143. X36 --> X60
1144. X38 --> X60
1145. X40 --> X60
1146. X42 --> X60
1147. X43 --> X60
1148. X44 --> X60
1149. X45 --> X60
1150. X46 --> X60
1151. X47 --> X60
1152. X48 --> X60
1153. X49 --> X60
1154. X50 --> X60
1155. X51 --> X60
1156. X52 --> X60
1157. X55 --> X60
1158. X56 --> X60
1159. X57 --> X60
1160. X58 --> X60
1161. X59 --> X60
1162. X61 --> X60
1163. X62 --> X60
1164. X63 --> X60
1165. X65 --> X60
1166. X66 --> X60
1167. X68 --> X60
1168. X70 --> X60
1169. X73 --> X60
1170. X74 --> X60
1171. X76 --> X60
1172. X78 --> X60
1173. X79 --> X60
1174. X80 --> X60
1175. X81 --> X60
1176. X84 --> X60
1177. X87 --> X60
1178. X88 --> X60
1179. X1 --> X61
1180. X7 --> X61
1181. X10 --> X61
1182. X11 --> X61
1183. X12 --> X61
1184. X13 --> X61
1185. X14 --> X61
1186. X15 --> X61
1187. X16 --> X61
1188. X18 --> X61
1189. X20 --> X61
1190. X22 --> X61
1191. X25 --> X61
1192. X26 --> X61
1193. X28 --> X61
1194. X30 --> X61
1195. X31 --> X61
1196. X32 --> X61
1197. X34 --> X61
1198. X35 --> X61
1199. X36 --> X61
1200. X38 --> X61
1201. X40 --> X61
1202. X42 --> X61
1203. X43 --> X61
1204. X44 --> X61
1205. X45 --> X61
1206. X46 --> X61
1207. X48 --> X61
1208. X49 --> X61
1209. X50 --> X61
1210. X51 --> X61
1211. X52 --> X61
1212. X53 --> X61
1213. X55 --> X61
1214. X56 --> X61
1215. X57 --> X61
1216. X58 --> X61
1217. X59 --> X61
1218. X60 --> X61
1219. X62 --> X61
1220. X63 --> X61
1221. X64 --> X61
1222. X65 --> X61
1223. X66 --> X61
1224. X67 --> X61
1225. X68 --> X61
1226. X69 --> X61
1227. X70 --> X61
1228. X73 --> X61
1229. X74 --> X61
1230. X76 --> X61
1231. X77 --> X61
1232. X78 --> X61
1233. X79 --> X61
1234. X80 --> X61
1235. X81 --> X61
1236. X82 --> X61
1237. X83 --> X61
1238. X84 --> X61
1239. X86 --> X61
1240. X87 --> X61
1241. X1 --> X62
1242. X2 --> X62
1243. X3 --> X62
1244. X7 --> X62
1245. X10 --> X62
1246. X11 --> X62
1247. X12 --> X62
1248. X19 --> X62
1249. X20 --> X62
1250. X22 --> X62
1251. X23 --> X62
1252. X24 --> X62
1253. X25 --> X62
1254. X26 --> X62
1255. X27 --> X62
1256. X28 --> X62
1257. X29 --> X62
1258. X31 --> X62
1259. X32 --> X62
1260. X33 --> X62
1261. X34 --> X62
1262. X37 --> X62
1263. X38 --> X62
1264. X39 --> X62
1265. X40 --> X62
1266. X41 --> X62
1267. X42 --> X62
1268. X43 --> X62
1269. X44 --> X62
1270. X45 --> X62
1271. X46 --> X62
1272. X47 --> X62
1273. X48 --> X62
1274. X49 --> X62
1275. X50 --> X62
1276. X51 --> X62
1277. X52 --> X62
1278. X53 --> X62
1279. X54 --> X62
1280. X55 --> X62
1281. X56 --> X62
1282. X57 --> X62
1283. X58 --> X62
1284. X59 --> X62
1285. X60 --> X62
1286. X61 --> X62
1287. X63 --> X62
1288. X64 --> X62
1289. X65 --> X62
1290. X66 --> X62
1291. X67 --> X62
1292. X68 --> X62
1293. X69 --> X62
1294. X70 --> X62
1295. X72 --> X62
1296. X73 --> X62
1297. X74 --> X62
1298. X75 --> X62
1299. X77 --> X62
1300. X78 --> X62
1301. X79 --> X62
1302. X80 --> X62
1303. X81 --> X62
1304. X82 --> X62
1305. X83 --> X62
1306. X84 --> X62
1307. X85 --> X62
1308. X86 --> X62
1309. X87 --> X62
1310. X88 --> X62
1311. X89 --> X62
1312. X90 --> X62
1313. X7 --> X63
1314. X13 --> X63
1315. X20 --> X63
1316. X22 --> X63
1317. X25 --> X63
1318. X28 --> X63
1319. X34 --> X63
1320. X40 --> X63
1321. X41 --> X63
1322. X42 --> X63
1323. X45 --> X63
1324. X46 --> X63
1325. X50 --> X63
1326. X53 --> X63
1327. X54 --> X63
1328. X55 --> X63
1329. X57 --> X63
1330. X58 --> X63
1331. X59 --> X63
1332. X60 --> X63
1333. X62 --> X63
1334. X64 --> X63
1335. X65 --> X63
1336. X68 --> X63
1337. X69 --> X63
1338. X70 --> X63
1339. X72 --> X63
1340. X73 --> X63
1341. X74 --> X63
1342. X78 --> X63
1343. X79 --> X63
1344. X81 --> X63
1345. X83 --> X63
1346. X84 --> X63
1347. X85 --> X63
1348. X87 --> X63
1349. X88 --> X63
1350. X1 --> X64
1351. X3 --> X64
1352. X7 --> X64
1353. X9 --> X64
1354. X10 --> X64
1355. X11 --> X64
1356. X12 --> X64
1357. X13 --> X64
1358. X16 --> X64
1359. X17 --> X64
1360. X18 --> X64
1361. X19 --> X64
1362. X20 --> X64
1363. X23 --> X64
1364. X24 --> X64
1365. X27 --> X64
1366. X30 --> X64
1367. X31 --> X64
1368. X32 --> X64
1369. X34 --> X64
1370. X36 --> X64
1371. X37 --> X64
1372. X38 --> X64
1373. X40 --> X64
1374. X41 --> X64
1375. X42 --> X64
1376. X43 --> X64
1377. X45 --> X64
1378. X47 --> X64
1379. X49 --> X64
1380. X50 --> X64
1381. X51 --> X64
1382. X52 --> X64
1383. X54 --> X64
1384. X55 --> X64
1385. X56 --> X64
1386. X57 --> X64
1387. X58 --> X64
1388. X59 --> X64
1389. X60 --> X64
1390. X61 --> X64
1391. X62 --> X64
1392. X65 --> X64
1393. X66 --> X64
1394. X67 --> X64
1395. X68 --> X64
1396. X69 --> X64
1397. X70 --> X64
1398. X74 --> X64
1399. X75 --> X64
1400. X76 --> X64
1401. X77 --> X64
1402. X78 --> X64
1403. X79 --> X64
1404. X80 --> X64
1405. X81 --> X64
1406. X82 --> X64
1407. X83 --> X64
1408. X84 --> X64
1409. X86 --> X64
1410. X87 --> X64
1411. X5 --> X66
1412. X7 --> X66
1413. X9 --> X66
1414. X10 --> X66
1415. X11 --> X66
1416. X12 --> X66
1417. X13 --> X66
1418. X14 --> X66
1419. X15 --> X66
1420. X20 --> X66
1421. X21 --> X66
1422. X25 --> X66
1423. X26 --> X66
1424. X27 --> X66
1425. X28 --> X66
1426. X29 --> X66
1427. X31 --> X66
1428. X32 --> X66
1429. X33 --> X66
1430. X38 --> X66
1431. X40 --> X66
1432. X41 --> X66
1433. X42 --> X66
1434. X43 --> X66
1435. X44 --> X66
1436. X45 --> X66
1437. X49 --> X66
1438. X50 --> X66
1439. X51 --> X66
1440. X52 --> X66
1441. X56 --> X66
1442. X57 --> X66
1443. X58 --> X66
1444. X59 --> X66
1445. X60 --> X66
1446. X61 --> X66
1447. X62 --> X66
1448. X64 --> X66
1449. X65 --> X66
1450. X67 --> X66
1451. X68 --> X66
1452. X69 --> X66
1453. X70 --> X66
1454. X73 --> X66
1455. X74 --> X66
1456. X75 --> X66
1457. X76 --> X66
1458. X77 --> X66
1459. X78 --> X66
1460. X80 --> X66
1461. X81 --> X66
1462. X82 --> X66
1463. X83 --> X66
1464. X84 --> X66
1465. X85 --> X66
1466. X86 --> X66
1467. X87 --> X66
1468. X90 --> X66
1469. X91 --> X66
1470. X7 --> X74
1471. X10 --> X74
1472. X11 --> X74
1473. X12 --> X74
1474. X14 --> X74
1475. X16 --> X74
1476. X24 --> X74
1477. X29 --> X74
1478. X31 --> X74
1479. X32 --> X74
1480. X33 --> X74
1481. X34 --> X74
1482. X36 --> X74
1483. X37 --> X74
1484. X38 --> X74
1485. X39 --> X74
1486. X40 --> X74
1487. X42 --> X74
1488. X43 --> X74
1489. X44 --> X74
1490. X45 --> X74
1491. X46 --> X74
1492. X47 --> X74
1493. X48 --> X74
1494. X49 --> X74
1495. X53 --> X74
1496. X54 --> X74
1497. X55 --> X74
1498. X56 --> X74
1499. X57 --> X74
1500. X58 --> X74
1501. X59 --> X74
1502. X60 --> X74
1503. X61 --> X74
1504. X62 --> X74
1505. X63 --> X74
1506. X64 --> X74
1507. X65 --> X74
1508. X66 --> X74
1509. X68 --> X74
1510. X69 --> X74
1511. X70 --> X74
1512. X73 --> X74
1513. X75 --> X74
1514. X76 --> X74
1515. X77 --> X74
1516. X78 --> X74
1517. X79 --> X74
1518. X80 --> X74
1519. X81 --> X74
1520. X82 --> X74
1521. X83 --> X74
1522. X84 --> X74
1523. X12 --> X78
1524. X16 --> X78
1525. X28 --> X78
1526. X34 --> X78
1527. X38 --> X78
1528. X40 --> X78
1529. X42 --> X78
1530. X48 --> X78
1531. X53 --> X78
1532. X54 --> X78
1533. X55 --> X78
1534. X56 --> X78
1535. X58 --> X78
1536. X59 --> X78
1537. X60 --> X78
1538. X61 --> X78
1539. X62 --> X78
1540. X63 --> X78
1541. X66 --> X78
1542. X68 --> X78
1543. X74 --> X78
1544. X75 --> X78
1545. X79 --> X78
1546. X80 --> X78
1547. X81 --> X78
1548. X83 --> X78
1549. X1 --> X82
1550. X7 --> X82
1551. X8 --> X82
1552. X9 --> X82
1553. X10 --> X82
1554. X11 --> X82
1555. X12 --> X82
1556. X13 --> X82
1557. X14 --> X82
1558. X15 --> X82
1559. X16 --> X82
1560. X20 --> X82
1561. X21 --> X82
1562. X22 --> X82
1563. X23 --> X82
1564. X24 --> X82
1565. X25 --> X82
1566. X26 --> X82
1567. X27 --> X82
1568. X28 --> X82
1569. X29 --> X82
1570. X30 --> X82
1571. X31 --> X82
1572. X32 --> X82
1573. X33 --> X82
1574. X34 --> X82
1575. X38 --> X82
1576. X40 --> X82
1577. X41 --> X82
1578. X42 --> X82
1579. X43 --> X82
1580. X44 --> X82
1581. X45 --> X82
1582. X47 --> X82
1583. X50 --> X82
1584. X51 --> X82
1585. X52 --> X82
1586. X56 --> X82
1587. X57 --> X82
1588. X58 --> X82
1589. X59 --> X82
1590. X60 --> X82
1591. X61 --> X82
1592. X62 --> X82
1593. X64 --> X82
1594. X65 --> X82
1595. X66 --> X82
1596. X67 --> X82
1597. X68 --> X82
1598. X69 --> X82
1599. X70 --> X82
1600. X71 --> X82
1601. X74 --> X82
1602. X75 --> X82
1603. X76 --> X82
1604. X77 --> X82
1605. X78 --> X82
1606. X79 --> X82
1607. X80 --> X82
1608. X81 --> X82
1609. X83 --> X82
1610. X84 --> X82
1611. X85 --> X82
1612. X86 --> X82
1613. X87 --> X82
1614. X89 --> X82
1615. X90 --> X82

"""






if __name__ == "__main__":
    error_normalization = True
    CLINGO_LIMIT = 64
    PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))
    POSTFIX = 'Macaque_data'
    PreFix = 'PCMCI'

    args = parse_arguments(PNUM)
    args = convert_str_to_bool(args)
    omp_num_threads = args.PNUM
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    include_selfloop = True

    adjacency_matrix = parse_text_to_adjacency(text)
    for i in range(adjacency_matrix.shape[0]):
        adjacency_matrix[i, i] = 1
    network_GT = cv.adjs2graph(adjacency_matrix, np.zeros(adjacency_matrix.shape))
    run_analysis(args,network_GT,include_selfloop)