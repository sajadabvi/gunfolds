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
from gunfolds.scripts.utils import my_functions as mf
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
from gunfolds.scripts.simulation import bold_function as hrf
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
    
    # Solution selection parameters
    parser.add_argument("--selection_mode", default="top_k", 
                        choices=['top_k', 'delta_threshold'],
                        help="Solution selection mode: 'top_k' (select top k by cost) or 'delta_threshold' (select all within delta * min_cost)")
    parser.add_argument("--top_k", default=10, type=int,
                        help="Number of top solutions to select (for 'top_k' mode)")
    parser.add_argument("--delta_multiplier", default=1.9, type=float,
                        help="Delta multiplier for threshold selection (for 'delta_threshold' mode). E.g., 1.9 means select solutions with cost <= 1.9 * min_cost")
    
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
COMP_IDX = [25,26,35,44,45,46]  # 0-based indices for rPPC, rFIC, rDLPFC, ACC, PCC, VMPFC
COMP_NAMES = ["rPPC", "rFIC", "rDLPFC", "ACC", "PCC", "VMPFC"]

def run_pcmci_to_cg(ts_2d):
    """
    ts_2d: ndarray [T, n_nodes] for one subject
    Returns: (g_estimated, A, B) where g_estimated is CG, A forward lag, B backward lag
    """
    dataframe = pp.DataFrame(ts_2d)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
    results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.1)
    g_estimated, A, B = cv.Glag2CG(results)
    return g_estimated, A, B

def RASL_subject(ts_2d, args, network_GT, include_selfloop, selection_mode='top_k', top_k=10, delta_multiplier=1.9):
    """
    Run RASL pipeline for a single subject given its time series.
    
    Args:
        ts_2d: Time series data [T, n_nodes]
        args: Arguments object
        network_GT: Ground truth network (used only for GT_density and SCC members if needed)
        include_selfloop: Whether to include self-loops
        selection_mode: 'top_k' or 'delta_threshold'
        top_k: Number of top solutions to keep (for 'top_k' mode)
        delta_multiplier: Delta multiplier (for 'delta_threshold' mode)
    
    Returns:
        Tuple: (list of selected CGs, list of full solution info with costs)
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

    # Select solutions based on cost (no ground truth scoring)
    n_nodes = len(g_estimated)
    kept = select_top_solutions(
        r_estimated, 
        n_nodes, 
        selection_mode=selection_mode,
        k=top_k,
        delta_multiplier=delta_multiplier
    )
    
    # Return both the CGs and the full info (with costs) for analysis
    res_cgs = [res for _, res, _ in kept]
    return res_cgs, kept

def select_top_solutions(r_estimated, n_nodes, selection_mode='top_k', k=10, delta_multiplier=1.9):
    """
    Select solutions based on cost (no ground truth needed).
    
    Args:
        r_estimated: Set of solutions from RASL/DRASL
        n_nodes: Number of nodes in the graph
        selection_mode: Either 'top_k' or 'delta_threshold'
            - 'top_k': Select top k solutions by lowest cost
            - 'delta_threshold': Select all solutions where cost <= min_cost * delta_multiplier
        k: Number of top solutions to select (for 'top_k' mode)
        delta_multiplier: Multiplier for delta threshold (for 'delta_threshold' mode)
    
    Returns:
        List of tuples: [(cost, res_cg, undersampling), ...] sorted by ascending cost
    """
    if not r_estimated:
        return []
    
    # Extract solutions with their costs and undersampling
    # r_estimated is a set of tuples: ((graph_number, (undersampling,)), cost)
    solutions_with_costs = []
    for answer in r_estimated:
        graph_num = answer[0][0]  # Graph number
        undersampling = answer[0][1]  # Undersampling tuple
        cost = answer[1]  # Cost
        res_cg = bfutils.num2CG(graph_num, n_nodes)
        solutions_with_costs.append((cost, res_cg, undersampling))
    
    # Sort by cost (ascending - lower cost is better)
    solutions_with_costs.sort(key=lambda x: x[0])
    
    if selection_mode == 'top_k':
        # Select top k solutions by lowest cost
        selected = solutions_with_costs[:min(k, len(solutions_with_costs))]
    elif selection_mode == 'delta_threshold':
        # Select all solutions within delta threshold
        min_cost = solutions_with_costs[0][0]
        threshold = min_cost * delta_multiplier
        selected = [s for s in solutions_with_costs if s[0] <= threshold]
        if not selected:
            # Fallback: at least include the best solution
            selected = [solutions_with_costs[0]]
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}. Use 'top_k' or 'delta_threshold'.")
    
    return selected

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

def run_all_subjects(args, network_GT, include_selfloop, selection_mode='top_k', top_k=10, delta_multiplier=1.9):
    """
    Run RASL for every subject in fbirn_sz_data.npz. Save per-subject plots and group graphs
    separately for each label group, and also keep a combined set as before.
    
    Args:
        args: Arguments object
        network_GT: Ground truth network
        include_selfloop: Whether to include self-loops
        selection_mode: 'top_k' or 'delta_threshold'
        top_k: Number of top solutions to keep (for 'top_k' mode)
        delta_multiplier: Delta multiplier (for 'delta_threshold' mode)
    
    Directory layout:
      fbirn_results/
        combined/ ... (original behavior)
        group_<LABEL>/
          subjects/*.pdf
          solutions/ ... (saved solution data)
          group_edge_counts_{LABEL}.npz
          group_edge_counts_{LABEL}.csv
          group_graph_gt_{LABEL}.pdf
          group_graph_weighted_{LABEL}.pdf
    """
    npzfile = np.load("../fbirn/fbirn_sz_data.npz")
    data = npzfile['data']  # [n_subjects, T, F]
    labels = get_labels(npzfile)  # shape [n_subjects]

    n_subj = data.shape[0]
    N = len(COMP_IDX)

    # --- Prepare output roots with timestamp ---
    timestamp = datetime.now().strftime('%m%d%Y%H%M%S')
    root_dir = os.path.join("fbirn_results", timestamp)
    combined_dir = os.path.join(root_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    
    print(f"Saving results to: {root_dir}")
    print(f"Timestamp: {timestamp}")

    # --- Set up containers for combined outputs ---
    combined_counts = np.zeros((N, N), dtype=int)
    
    # Storage for all solution info (for analysis)
    all_solutions_info = []

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
        solutions_dir = os.path.join(out_dir, "solutions")
        os.makedirs(per_subj_dir, exist_ok=True)
        os.makedirs(solutions_dir, exist_ok=True)

        counts = np.zeros((N, N), dtype=int)
        group_solutions_info = []

        for s in subj_indices:
            ts_2d = data[s][:, COMP_IDX]  # [T, N]
            res_list, solution_info = RASL_subject(
                ts_2d, args, network_GT, include_selfloop,
                selection_mode=selection_mode,
                top_k=top_k,
                delta_multiplier=delta_multiplier
            )

            # Store solution info for this subject
            subject_info = {
                'subject_id': int(s),
                'group': grp,
                'num_solutions': len(solution_info),
                'solutions': []
            }
            
            # Save plots for all kept solutions and accumulate counts
            for r_idx, (res_rasl, sol_detail) in enumerate(zip(res_list, solution_info), start=1):
                cost, res_cg, undersampling = sol_detail
                
                gt.plotg(
                    res_rasl,
                    names=COMP_NAMES,
                    output=os.path.join(per_subj_dir, f"rasl_subj_{s:04d}_sol_{r_idx:02d}_grp_{grp}.pdf"),
                )
                A = cg_to_adj_binary(res_rasl)
                counts += A
                combined_counts += A
                
                # Save solution details
                subject_info['solutions'].append({
                    'solution_idx': r_idx,
                    'cost': float(cost),
                    'undersampling': undersampling,
                    'graph': res_cg
                })
            
            group_solutions_info.append(subject_info)
            all_solutions_info.append(subject_info)
        
        # Save solutions info for this group
        solutions_file = os.path.join(solutions_dir, f"solutions_info_{grp}.zkl")
        zkl.save(group_solutions_info, solutions_file)
        print(f"Saved solutions info for group {grp} to: {solutions_file}")

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
    # Save all solutions info (combined)
    combined_solutions_file = os.path.join(combined_dir, "all_solutions_info.zkl")
    zkl.save(all_solutions_info, combined_solutions_file)
    print(f"Saved all solutions info to: {combined_solutions_file}")
    
    # Also save selection parameters used
    selection_params = {
        'selection_mode': selection_mode,
        'top_k': top_k if selection_mode == 'top_k' else None,
        'delta_multiplier': delta_multiplier if selection_mode == 'delta_threshold' else None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    params_file = os.path.join(combined_dir, "selection_params.zkl")
    zkl.save(selection_params, params_file)
    print(f"Saved selection parameters to: {params_file}")
    
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


def initialize_metrics():
    return {
        'Precision_O': [], 'Recall_O': [], 'F1_O': [],
        'Precision_A': [], 'Recall_A': [], 'F1_A': []
    }



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

    # Print selection parameters
    print("=" * 80)
    print("FMRI EXPERIMENT - SOLUTION SELECTION CONFIGURATION")
    print("=" * 80)
    print(f"Selection mode: {args.selection_mode}")
    if args.selection_mode == 'top_k':
        print(f"  Top K: {args.top_k} (selecting top {args.top_k} solutions by lowest cost)")
    elif args.selection_mode == 'delta_threshold':
        print(f"  Delta multiplier: {args.delta_multiplier} (selecting solutions with cost <= {args.delta_multiplier} * min_cost)")
    print("=" * 80)
    print()

    run_all_subjects(
        args, 
        network_GT, 
        include_selfloop,
        selection_mode=args.selection_mode,
        top_k=args.top_k,
        delta_multiplier=args.delta_multiplier
    )