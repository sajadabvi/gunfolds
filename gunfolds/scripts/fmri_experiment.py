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
from gunfolds.scripts.datasets.simple_networks import macaque_net
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
    parser.add_argument("-z", "--NOISE", default=10, help="noise str multiplied by 100", type=int)
    parser.add_argument("-s", "--SCC", default="f", help="true to use SCC structure, false to not", type=str)
    parser.add_argument("-m", "--SCCMEMBERS", default="f",
                        help="true for using g_estimate SCC members, false for using "
                             "GT SCC members", type=str)
    parser.add_argument("-u", "--UNDERSAMPLING", default=75, help="sampling rate in generated data", type=int)
    parser.add_argument("-x", "--MAXU", default=4, help="maximum number of undersampling to look for solution.",
                        type=int)
    parser.add_argument("-t", "--CONCAT", default="t", help="true to use concat data, false to not", type=str)

    parser.add_argument("-y", "--PRIORITY", default="11112", help="string of priorities", type=str)
    parser.add_argument("-o", "--METHOD", default="RASL", help="method to run", type=str)
    parser.add_argument("-v", "--VERSION", default="SmallDegree", help="version of macaque data", type=str)
    return parser.parse_args()

def convert_str_to_bool(args):
    args.SCC = bool(strtobool(args.SCC))
    args.SCCMEMBERS = bool(strtobool(args.SCCMEMBERS))
    args.CONCAT = bool(strtobool(args.CONCAT))
    args.NOISE = args.NOISE / 100
    priprities = []
    for char in args.PRIORITY:
        priprities.append(int(char))
    args.PRIORITY = priprities
    return args

# ---- Configuration for component selection and naming ----
COMP_IDX = [25, 29, 35, 44, 45, 46]  # 0-based indices for rPPC, rFIC, rDPFC, ACC, PCC, VMPFC
COMP_NAMES = ["rPPC", "rFIC", "rDPFC", "ACC", "PCC", "VMPFC"]

def run_pcmci_to_cg(ts_2d):
    """
    ts_2d: ndarray [T, n_nodes] for one subject
    Returns: (g_estimated, A, B) where g_estimated is CG, A forward lag, B backward lag
    """
    dataframe = pp.DataFrame(ts_2d.T)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.1)
    g_estimated, A, B = cv.Glag2CG(results)
    return g_estimated, A, B

def RASL_subject(ts_2d, args, network_GT, include_selfloop):
    """
    Run RASL pipeline for a single subject given its time series.
    """
    g_estimated, A, B = run_pcmci_to_cg(ts_2d)

    if args.SCCMEMBERS:
        members = [s for s in nx.strongly_connected_components(gk.graph2nx(g_estimated))]
    else:
        members = [s for s in nx.strongly_connected_components(gk.graph2nx(network_GT))]

    MAXCOST = 10000
    # Normalize and build distance penalties
    DD = (np.abs((np.abs(A / np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
    BD = (np.abs((np.abs(B / np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

    r_estimated = drasl(
        [g_estimated],
        weighted=True,
        capsize=0,
        timeout=0,
        urate=min(args.MAXU, (3 * len(g_estimated) + 1)),
        dm=[DD],
        bdm=[BD],
        scc=True,
        scc_members=members,
        GT_density=int(1000 * gk.density(network_GT)),
        edge_weights=args.PRIORITY,
        pnum=PNUM,
        optim='optN',
        selfloop=False,
    )

    # Rank all solutions by F1 and keep top 75%
    kept = select_top_solutions(r_estimated, network_GT, include_selfloop)
    # Return only the CGs in descending F1 order
    return [res for _, res in kept]

def select_top_solutions(r_estimated, network_GT, include_selfloop):
    """
    Score each candidate CG with F1 (orientation). Drop the worst 25%.
    Return a list of (f1, res_cg) sorted by descending F1 for the retained set.
    """
    scored = []
    for answer in r_estimated:
        res = bfutils.num2CG(answer[0][0], len(network_GT))
        scores = mf.precision_recall_all_cycle(res, network_GT, include_selfloop=include_selfloop)
        f1 = scores['orientation']['F1']
        scored.append((f1, res))
    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    k = max(1, int(np.ceil(0.75 * len(scored))))
    return scored[:k]

def cg_to_adj_binary(cg):
    """Return binary adjacency (NxN) from a CG, ignoring weights."""
    A = cv.graph2adj(cg)
    A = (A > 0).astype(int)
    np.fill_diagonal(A, 0)
    return A

def plot_group_graph(counts, names, outpath):
    """
    Draw a weighted directed group graph with edge widths proportional to counts.
    counts: NxN integer matrix of edge frequencies (directed)
    """
    N = len(names)
    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i, label=names[i])
    maxcount = int(counts.max()) if counts.size else 0
    for i in range(N):
        for j in range(N):
            c = int(counts[i, j])
            if c > 0:
                G.add_edge(i, j, weight=c)

    # Fixed circular layout for consistent positions across runs
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pos = {i: (np.cos(theta[i]), np.sin(theta[i])) for i in range(N)}

    plt.figure(figsize=(6, 6))
    NODE_SIZE = 900
    nx.draw_networkx_nodes(G, pos, node_size=NODE_SIZE)
    nx.draw_networkx_labels(G, pos, labels={i: names[i] for i in range(N)}, font_size=10)

    # Extreme-contrast mapping: exponential on normalized weights
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    if weights:
        w_min, w_max = min(weights), max(weights)
        if w_min == w_max:
            widths = [10.0] * len(weights)
        else:
            min_thick, max_thick = 0.3, 12.0  # very wide dynamic range
            k = 5  # larger => more extreme separation
            denom = np.exp(k) - 1.0
            widths = []
            for w in weights:
                t = (w - w_min) / (w_max - w_min)
                s = (np.exp(k * t) - 1.0) / denom
                widths.append(min_thick + (max_thick - min_thick) * s)
    else:
        widths = []

    # --- Make arrowheads sharp and ensure they reach the node border ---
    if widths:
        max_w = max(widths)
    else:
        max_w = 1.0
    # Arrowsize must dominate shaft width to avoid deformation
    arrowsize = int(max(22, np.ceil(2.5 * max_w)))
    # Small positive margins in points: head stops at node border with no gap
    margin = 2.0

    # Draw directed edges. If both directions exist, draw two arcs with opposite curvature.
    if widths:
        # Map each edge to its computed width
        widths_map = {edge: w for edge, w in zip(G.edges(), widths)}
        drawn = set()
        for (u, v) in G.edges():
            if (u, v) in drawn:
                continue
            has_back = G.has_edge(v, u)
            if has_back:
                # Opposite curvatures to separate the pair visually
                w_uv = widths_map.get((u, v), 1.0)
                w_vu = widths_map.get((v, u), 1.0)
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[(u, v)],
                    width=w_uv,
                    arrows=True,
                    arrowstyle='-|>',
                    arrowsize=arrowsize,
                    min_source_margin=margin,
                    min_target_margin=margin,
                    connectionstyle='arc3,rad=0.25',
                )
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[(v, u)],
                    width=w_vu,
                    arrows=True,
                    arrowstyle='-|>',
                    arrowsize=arrowsize,
                    min_source_margin=margin,
                    min_target_margin=margin,
                    connectionstyle='arc3,rad=0.25',
                )
                drawn.add((u, v))
                drawn.add((v, u))
            else:
                # Single directed edge, straight line
                w_uv = widths_map.get((u, v), 1.0)
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[(u, v)],
                    width=w_uv,
                    arrows=True,
                    arrowstyle='-|>',
                    arrowsize=arrowsize,
                    min_source_margin=margin,
                    min_target_margin=margin,
                    connectionstyle='arc3,rad=0.0',
                )

    plt.axis('off')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()

def get_labels(npz):
    # Support both 'labels' and 'label' keys
    if 'labels' in npz.files:
        return npz['labels']
    if 'label' in npz.files:
        return npz['label']
    raise KeyError("Labels not found in NPZ. Expected key 'labels' or 'label'.")

def run_all_subjects(args, network_GT, include_selfloop):
    """
    Run RASL for every subject in fbirn_sz_data.npz. Save per-subject plots and group graphs
    separately for each label group, and also keep a combined set as before.
    Directory layout:
      fbirn_results/
        combined/ ... (original behavior)
        group_<LABEL>/
          subjects/*.pdf
          group_edge_counts_{LABEL}.npz
          group_edge_counts_{LABEL}.csv
          group_graph_gt_{LABEL}.pdf
          group_graph_weighted_{LABEL}.pdf
    """
    npzfile = np.load("./fbirn/fbirn_sz_data.npz")
    data = npzfile['data']  # [n_subjects, T, F]
    labels = get_labels(npzfile)  # shape [n_subjects]

    n_subj = data.shape[0]
    N = len(COMP_IDX)

    # --- Prepare output roots ---
    root_dir = "fbirn_results"
    combined_dir = os.path.join(root_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    # --- Set up containers for combined outputs ---
    combined_counts = np.zeros((N, N), dtype=int)

    # --- Identify unique label groups ---
    unique_labels = list(pd.unique(labels))

    # For each label, process subjects and generate artifacts
    for grp in unique_labels:
        grp_mask = (labels == grp)
        subj_indices = np.where(grp_mask)[0]
        if subj_indices.size == 0:
            continue

        # Per-group output dirs
        out_dir = os.path.join(root_dir, f"group_{grp}")
        per_subj_dir = os.path.join(out_dir, "subjects")
        os.makedirs(per_subj_dir, exist_ok=True)

        counts = np.zeros((N, N), dtype=int)

        for s in subj_indices:
            ts_2d = data[s, :, COMP_IDX]  # [T, N]
            res_list = RASL_subject(ts_2d, args, network_GT, include_selfloop)

            # Save plots for all kept solutions and accumulate counts
            for r_idx, res_rasl in enumerate(res_list, start=1):
                gt.plotg(
                    res_rasl,
                    names=COMP_NAMES,
                    output=os.path.join(per_subj_dir, f"rasl_subj_{s:04d}_sol_{r_idx:02d}_grp_{grp}.pdf"),
                )
                A = cg_to_adj_binary(res_rasl)
                counts += A
                combined_counts += A

        # Persist per-group counts
        np.savez(os.path.join(out_dir, f"group_edge_counts_{grp}.npz"), counts=counts, names=np.array(COMP_NAMES))

        # CSV table for the group
        with open(os.path.join(out_dir, f"group_edge_counts_{grp}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["src", "dst", "count"])
            for i in range(N):
                for j in range(N):
                    if counts[i, j] > 0:
                        w.writerow([COMP_NAMES[i], COMP_NAMES[j], int(counts[i, j])])

        # Build a weighted CG dict for gt.plotg (edge values = 1 to indicate presence; weights are visualized via the weighted plot)
        group_cg = {i + 1: {} for i in range(N)}
        for i in range(N):
            for j in range(N):
                c = int(counts[i, j])
                if c > 0:
                    group_cg[i + 1][j + 1] = 1

        # Plot with gt.plotg
        gt.plotg(group_cg, names=COMP_NAMES, output=os.path.join(out_dir, f"group_graph_gt_{grp}.pdf"))
        # Also plot a guaranteed weighted version
        plot_group_graph(counts, COMP_NAMES, os.path.join(out_dir, f"group_graph_weighted_{grp}.pdf"))

    # --- Preserve original combined outputs for compatibility ---
    # Persist combined counts
    np.savez(os.path.join(combined_dir, "group_edge_counts_combined.npz"), counts=combined_counts, names=np.array(COMP_NAMES))

    # CSV table for combined
    with open(os.path.join(combined_dir, "group_edge_counts_combined.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "count"])
        for i in range(N):
            for j in range(N):
                if combined_counts[i, j] > 0:
                    w.writerow([COMP_NAMES[i], COMP_NAMES[j], int(combined_counts[i, j])])

    # Build combined graph for gt.plotg
    group_cg = {i + 1: {} for i in range(N)}
    for i in range(N):
        for j in range(N):
            c = int(combined_counts[i, j])
            if c > 0:
                group_cg[i + 1][j + 1] = 1

    gt.plotg(group_cg, names=COMP_NAMES, output=os.path.join(combined_dir, "group_graph_gt_combined.pdf"))
    plot_group_graph(combined_counts, COMP_NAMES, os.path.join(combined_dir, "group_graph_weighted_combined.pdf"))

def RASL(args, network_GT):
    npzfile = np.load("./fbirn/fbirn_sz_data.npz")

    data = npzfile['data']
    labels = npzfile['labels']

    dataframe = pp.DataFrame(npzfile['data'][0,:,[25,29,35,44,45,46]].T)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.1)
    g_estimated, A, B = cv.Glag2CG(results)
    # members = nx.strongly_connected_components(gk.graph2nx(g_estimated))
    if args.SCCMEMBERS:
        members = [s for s in nx.strongly_connected_components(gk.graph2nx(g_estimated))]
    else:
        members = [s for s in nx.strongly_connected_components(gk.graph2nx(network_GT))]

    MAXCOST = 10000
    DD = (np.abs((np.abs(A / np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
    BD = (np.abs((np.abs(B / np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

    r_estimated = drasl([g_estimated], weighted=True, capsize=0, timeout=0,
                        urate=min(args.MAXU, (3 * len(g_estimated) + 1)),
                        dm=[DD],
                        bdm=[BD],
                        scc=True,
                        scc_members=members,
                        GT_density=int(1000 * gk.density(network_GT)),
                        edge_weights=args.PRIORITY, pnum=PNUM, optim='optN', selfloop=False)

    kept = select_top_solutions(r_estimated, network_GT, include_selfloop)
    # Optional: plot all kept solutions for subject 0 when using RASL() directly
    out_dir = "fbirn_results"
    os.makedirs(out_dir, exist_ok=True)
    for r_idx, (_, res_keep) in enumerate(kept, start=1):
        gt.plotg(res_keep, names=COMP_NAMES, output=os.path.join(out_dir, f"rasl_single_run_sol_{r_idx:02d}.pdf"))

    # Use the best of the kept set as the return value
    res_rasl = kept[0][1] if kept else bfutils.num2CG(r_estimated[0][0][0], len(network_GT))
    return res_rasl



def initialize_metrics():
    return {
        'Precision_O': [], 'Recall_O': [], 'F1_O': [],
        'Precision_A': [], 'Recall_A': [], 'F1_A': []
    }


def run_analysis(args,network_GT,include_selfloop):
    metrics = {key: {args.UNDERSAMPLING: initialize_metrics()} for key in [args.METHOD]}

    for method in metrics.keys():

        result = globals()[method](args, network_GT)
        print(f"Result from {method}: {result}")
        normal_GT = mf.precision_recall_no_cycle(result, network_GT, include_selfloop=include_selfloop)
        metrics[method][args.UNDERSAMPLING]['Precision_O'].append(normal_GT['orientation']['precision'])
        metrics[method][args.UNDERSAMPLING]['Recall_O'].append(normal_GT['orientation']['recall'])
        metrics[method][args.UNDERSAMPLING]['F1_O'].append(normal_GT['orientation']['F1'])

        metrics[method][args.UNDERSAMPLING]['Precision_A'].append(normal_GT['adjacency']['precision'])
        metrics[method][args.UNDERSAMPLING]['Recall_A'].append(normal_GT['adjacency']['recall'])
        metrics[method][args.UNDERSAMPLING]['F1_A'].append(normal_GT['adjacency']['F1'])


    print(metrics)
    if not os.path.exists('fbirin_results'):
        os.makedirs('fbirin_results')
    filename = f'fbirin_results/fbirin_{args.METHOD}_batch_{args.BATCH}.zkl'
    zkl.save(metrics,filename)
    print('file saved to :' + filename)
    #gt.plotg(res_rasl,names=["rPPC","rFIC","rDPFC","ACC","PCC","VMPFC"],output='see_out.pdf')


if __name__ == "__main__":
    error_normalization = True
    CLINGO_LIMIT = 64
    PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))
    POSTFIX = 'fbrirn_data'
    PreFix = 'RASL_sim'

    args = parse_arguments(PNUM)
    args = convert_str_to_bool(args)
    omp_num_threads = args.PNUM
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    include_selfloop = True

    network_GT = {1: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1}, 2: {1: 1, 3: 1, 4: 1, 5: 1, 6: 1},
                  3: {1: 1, 2: 1, 4: 1, 5: 1, 6: 1}, 4: {1: 1, 2: 1, 3: 1, 5: 1, 6: 1},
                  5: {1: 1, 2: 1, 3: 1, 4: 1, 6: 1}, 6: {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}}

    run_all_subjects(args, network_GT, include_selfloop)