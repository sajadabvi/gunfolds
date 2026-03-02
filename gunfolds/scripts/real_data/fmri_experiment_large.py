"""
fMRI experiment: RASL / PCMCI / GCM causal discovery on FBIRN ICA data.

Supports:
  - Variable component counts (10, 20, 53) via NeuroMark subsets
  - Multiple SCC strategies for RASL (domain, correlation, estimated, none)
  - Three methods: RASL (undersampling-aware), PCMCI-only, GCM (baselines)
  - Single-subject mode for SLURM array jobs (--subject_idx)
  - All-subjects sequential mode for local testing

Usage examples:
  # Single subject on SLURM
  python fmri_experiment_large.py --subject_idx 42 --n_components 20 \\
      --scc_strategy domain --method RASL --timestamp 03012026120000

  # All subjects locally (small N)
  python fmri_experiment_large.py --n_components 10 --scc_strategy domain --method PCMCI
"""

import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from gunfolds.utils import bfutils
from gunfolds import conversions as cv
from gunfolds.solvers.clingo_rasl import drasl
from gunfolds.utils import graphkit as gk
from gunfolds.utils.calc_procs import get_process_count
from gunfolds.viz import gtool as gt
from gunfolds.utils import zickle as zkl

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from component_config import (
    get_comp_indices, get_comp_names, get_scc_members,
)

sys.path.append(os.path.expanduser("~/tread/py-tetrad"))

CLINGO_LIMIT = 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments():
    p = argparse.ArgumentParser(
        description="fMRI causal discovery experiment (RASL / PCMCI / GCM)."
    )
    # Experiment configuration
    p.add_argument("--n_components", type=int, default=10, choices=[10, 20, 53],
                   help="Number of ICA components to use")
    p.add_argument("--scc_strategy", type=str, default="domain",
                   choices=["domain", "correlation", "estimated", "none"],
                   help="SCC grouping strategy for RASL")
    p.add_argument("--method", type=str, default="RASL",
                   choices=["RASL", "PCMCI", "GCM"],
                   help="Causal discovery method")
    p.add_argument("--subject_idx", type=int, default=None,
                   help="Single subject index (for SLURM). Omit to run all.")
    p.add_argument("--timestamp", type=str, default=None,
                   help="Shared timestamp for grouping results")
    p.add_argument("--data_path", type=str, default="../fbirn/fbirn_sz_data.npz",
                   help="Path to fbirn_sz_data.npz")

    # RASL parameters
    p.add_argument("-p", "--PNUM", default=PNUM, type=int,
                   help="Number of CPUs for clingo")
    p.add_argument("-x", "--MAXU", default=4, type=int,
                   help="Max undersampling rate to search")
    p.add_argument("-y", "--PRIORITY", default="11112", type=str,
                   help="Edge weight priorities (string of digits)")
    p.add_argument("--selection_mode", default="top_k",
                   choices=["top_k", "delta_threshold"],
                   help="RASL solution selection mode")
    p.add_argument("--top_k", default=10, type=int,
                   help="Top k solutions to keep")
    p.add_argument("--delta_multiplier", default=1.9, type=float,
                   help="Delta multiplier for threshold selection")

    # GCM parameters
    p.add_argument("--gcm_alpha", default=0.01, type=float,
                   help="GCM significance level")
    p.add_argument("--gcm_pmax", default=8, type=int,
                   help="GCM max VAR lag order")
    p.add_argument("--gcm_nboot", default=200, type=int,
                   help="GCM number of bootstrap surrogates")

    # Correlation SCC parameters
    p.add_argument("--corr_max_cluster", default=8, type=int,
                   help="Max cluster size for correlation-based SCC")

    return p.parse_args()


# ---------------------------------------------------------------------------
# PCMCI helpers
# ---------------------------------------------------------------------------

def run_pcmci_to_cg(ts_2d):
    """
    Run PCMCI on time series and return gunfolds causal graph + lag matrices.

    Parameters
    ----------
    ts_2d : ndarray [T, n_nodes]

    Returns
    -------
    g_estimated : dict  (gunfolds CG)
    A : ndarray         (forward lag coefficients)
    B : ndarray         (backward lag coefficients)
    """
    dataframe = pp.DataFrame(ts_2d.T)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.1)
    g_estimated, A, B = cv.Glag2CG(results)
    return g_estimated, A, B


# ---------------------------------------------------------------------------
# GCM method
# ---------------------------------------------------------------------------

def run_gcm_subject(ts_2d, names, alpha=0.01, pmax=8, n_boot=200):
    """
    Run Roebroeck GCM on one subject and return the graph as a gunfolds CG.

    Parameters
    ----------
    ts_2d : ndarray [T, N]
    names : list[str]
    alpha, pmax, n_boot : GCM hyperparameters

    Returns
    -------
    cg : dict  (gunfolds CG, 1-based keys)
    adj : ndarray  (binary adjacency matrix)
    """
    from roebroeck_gcm import run_roebroeck_gcm

    res = run_roebroeck_gcm(
        ts_2d, tr_sec=2.0,
        n_cycles=0, remove_linear=False,
        alpha=alpha, pmax=pmax, n_boot=n_boot,
        surr_mode="block", block_len=16, seed=0, fdr=False,
        names=names, make_plot=False,
    )
    adj = res["Adj"].astype(int)
    N = adj.shape[0]
    cg = {i + 1: {} for i in range(N)}
    for i in range(N):
        for j in range(N):
            if adj[i, j]:
                cg[i + 1][j + 1] = 1
    return cg, adj


# ---------------------------------------------------------------------------
# RASL method
# ---------------------------------------------------------------------------

def run_rasl_subject(ts_2d, args, comp_indices, scc_members_override=None,
                     selection_mode="top_k", top_k=10, delta_multiplier=1.9):
    """
    Run the full RASL pipeline for one subject.

    Parameters
    ----------
    ts_2d : ndarray [T, N]
    args : Namespace
    comp_indices : list[int]  (0-based component indices, for density calc)
    scc_members_override : list[set] or None
        Pre-computed SCC members (domain or correlation). If None, use
        estimated SCCs from the PCMCI graph.
    selection_mode, top_k, delta_multiplier : solution selection params

    Returns
    -------
    res_cgs : list[dict]       (selected CG solutions)
    kept : list[tuple]         (cost, cg, undersampling) sorted by cost
    g_estimated : dict         (PCMCI-estimated graph, for reference)
    """
    g_estimated, A, B = run_pcmci_to_cg(ts_2d)
    n_nodes = len(g_estimated)

    # SCC members
    if scc_members_override is not None:
        members = scc_members_override
        use_scc = True
    elif args.scc_strategy == "estimated":
        members = list(nx.strongly_connected_components(gk.graph2nx(g_estimated)))
        use_scc = True
    else:
        members = None
        use_scc = False

    # Distance penalty matrices
    MAXCOST = 10000
    a_max = np.abs(A).max()
    b_max = np.abs(B).max()
    if a_max > 0:
        DD = (np.abs((np.abs(A / a_max) + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
    else:
        DD = (np.abs((cv.graph2adj(g_estimated) - 1) * MAXCOST)).astype(int)
    if b_max > 0:
        BD = (np.abs((np.abs(B / b_max) + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)
    else:
        BD = (np.abs((cv.graph2badj(g_estimated) - 1) * MAXCOST)).astype(int)

    gt_density = int(1000 * gk.density(g_estimated))

    priority = [int(c) for c in args.PRIORITY]

    r_estimated = drasl(
        [g_estimated],
        weighted=True,
        capsize=0,
        timeout=0,
        urate=min(args.MAXU, (3 * n_nodes + 1)),
        dm=[DD],
        bdm=[BD],
        scc=use_scc,
        scc_members=members,
        GT_density=gt_density,
        edge_weights=priority,
        pnum=args.PNUM,
        optim="optN",
        selfloop=False,
    )

    kept = select_top_solutions(
        r_estimated, n_nodes,
        selection_mode=selection_mode,
        k=top_k,
        delta_multiplier=delta_multiplier,
    )
    res_cgs = [cg for _, cg, _ in kept]
    return res_cgs, kept, g_estimated


def select_top_solutions(r_estimated, n_nodes, selection_mode="top_k",
                         k=10, delta_multiplier=1.9):
    """Select top solutions by cost from the RASL answer set."""
    if not r_estimated:
        return []

    solutions = []
    for answer in r_estimated:
        graph_num = answer[0][0]
        undersampling = answer[0][1]
        cost = answer[1]
        res_cg = bfutils.num2CG(graph_num, n_nodes)
        solutions.append((cost, res_cg, undersampling))

    solutions.sort(key=lambda x: x[0])

    if selection_mode == "top_k":
        return solutions[:min(k, len(solutions))]
    elif selection_mode == "delta_threshold":
        min_cost = solutions[0][0]
        threshold = min_cost * delta_multiplier
        selected = [s for s in solutions if s[0] <= threshold]
        return selected if selected else [solutions[0]]
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def cg_to_adj_binary(cg):
    """Return binary adjacency (NxN) from a gunfolds CG, no self-loops."""
    A = cv.graph2adj(cg)
    A = (A > 0).astype(int)
    np.fill_diagonal(A, 0)
    return A


def get_labels(npz):
    if "labels" in npz.files:
        return npz["labels"]
    if "label" in npz.files:
        return npz["label"]
    raise KeyError("Labels not found in NPZ. Expected 'labels' or 'label'.")


# ---------------------------------------------------------------------------
# Visualization (group-level)
# ---------------------------------------------------------------------------

def plot_group_graph(counts, names, outpath):
    """Draw weighted directed group graph with edge widths ~ counts."""
    N = len(names)
    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i, label=names[i])
    for i in range(N):
        for j in range(N):
            c = int(counts[i, j])
            if c > 0:
                G.add_edge(i, j, weight=c)

    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pos = {i: (np.cos(theta[i]), np.sin(theta[i])) for i in range(N)}

    fig_w = max(8, N * 0.4)
    plt.figure(figsize=(fig_w, fig_w))
    node_size = max(300, 900 - N * 10)
    font_size = max(5, 10 - N // 10)
    nx.draw_networkx_nodes(G, pos, node_size=node_size)
    nx.draw_networkx_labels(G, pos, labels={i: names[i] for i in range(N)},
                            font_size=font_size)

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    if weights:
        w_min, w_max = min(weights), max(weights)
        if w_min == w_max:
            widths = [4.0] * len(weights)
        else:
            min_thick, max_thick = 0.3, 8.0
            k_exp = 5
            denom = np.exp(k_exp) - 1.0
            widths = []
            for w in weights:
                t = (w - w_min) / (w_max - w_min)
                s = (np.exp(k_exp * t) - 1.0) / denom
                widths.append(min_thick + (max_thick - min_thick) * s)
    else:
        widths = []

    max_w = max(widths) if widths else 1.0
    arrowsize = int(max(18, np.ceil(2.5 * max_w)))
    margin = 2.0

    if widths:
        widths_map = dict(zip(G.edges(), widths))
        drawn = set()
        for (u, v) in G.edges():
            if (u, v) in drawn:
                continue
            has_back = G.has_edge(v, u)
            rad = 0.25 if has_back else 0.0
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], width=widths_map.get((u, v), 1.0),
                arrows=True, arrowstyle="-|>", arrowsize=arrowsize,
                min_source_margin=margin, min_target_margin=margin,
                connectionstyle=f"arc3,rad={rad}",
            )
            if has_back:
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(v, u)], width=widths_map.get((v, u), 1.0),
                    arrows=True, arrowstyle="-|>", arrowsize=arrowsize,
                    min_source_margin=margin, min_target_margin=margin,
                    connectionstyle=f"arc3,rad={rad}",
                )
                drawn.add((v, u))
            drawn.add((u, v))

    plt.axis("off")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Config tag
# ---------------------------------------------------------------------------

def make_config_tag(args):
    return f"N{args.n_components}_{args.scc_strategy}_{args.method}"


# ---------------------------------------------------------------------------
# Single-subject runner (SLURM mode)
# ---------------------------------------------------------------------------

def run_single_subject(args, data, labels, comp_indices, comp_names):
    """
    Process a single subject and save results to disk.
    Called when --subject_idx is provided (SLURM array task).
    """
    s = args.subject_idx
    ts_2d = data[s, :, comp_indices]  # [T, N]
    n_nodes = len(comp_indices)
    label = int(labels[s])
    config_tag = make_config_tag(args)
    timestamp = args.timestamp or datetime.now().strftime("%m%d%Y%H%M%S")

    out_dir = os.path.join("fbirn_results", timestamp, config_tag,
                           f"subject_{s:04d}")
    os.makedirs(out_dir, exist_ok=True)

    subject_info = {
        "subject_id": int(s),
        "group": label,
        "method": args.method,
        "config_tag": config_tag,
        "n_components": args.n_components,
        "scc_strategy": args.scc_strategy,
        "comp_indices": comp_indices,
        "comp_names": comp_names,
        "solutions": [],
    }

    if args.method == "RASL":
        scc_members = get_scc_members(
            args.scc_strategy, comp_indices, ts_2d,
            max_cluster_size=args.corr_max_cluster,
        )
        res_cgs, kept, g_est = run_rasl_subject(
            ts_2d, args, comp_indices,
            scc_members_override=scc_members,
            selection_mode=args.selection_mode,
            top_k=args.top_k,
            delta_multiplier=args.delta_multiplier,
        )
        for r_idx, (cost, cg, usamp) in enumerate(kept, start=1):
            adj = cg_to_adj_binary(cg)
            subject_info["solutions"].append({
                "solution_idx": r_idx,
                "cost": float(cost),
                "undersampling": usamp,
                "graph": cg,
                "adj": adj.tolist(),
            })
        subject_info["g_estimated"] = g_est
        subject_info["num_solutions"] = len(kept)

    elif args.method == "PCMCI":
        g_est, A, B = run_pcmci_to_cg(ts_2d)
        adj = cg_to_adj_binary(g_est)
        subject_info["solutions"].append({
            "solution_idx": 1,
            "cost": 0.0,
            "undersampling": None,
            "graph": g_est,
            "adj": adj.tolist(),
        })
        subject_info["g_estimated"] = g_est
        subject_info["num_solutions"] = 1

    elif args.method == "GCM":
        cg, adj = run_gcm_subject(
            ts_2d, comp_names,
            alpha=args.gcm_alpha, pmax=args.gcm_pmax, n_boot=args.gcm_nboot,
        )
        subject_info["solutions"].append({
            "solution_idx": 1,
            "cost": 0.0,
            "undersampling": None,
            "graph": cg,
            "adj": adj.tolist(),
        })
        subject_info["num_solutions"] = 1

    zkl.save(subject_info, os.path.join(out_dir, "result.zkl"))
    print(f"[{config_tag}] Subject {s} (group {label}): "
          f"{subject_info['num_solutions']} solution(s) saved to {out_dir}")


# ---------------------------------------------------------------------------
# All-subjects runner (local sequential mode)
# ---------------------------------------------------------------------------

def run_all_subjects(args, data, labels, comp_indices, comp_names):
    """
    Process all subjects sequentially.
    Used when --subject_idx is omitted (local testing / small runs).
    """
    n_subj = data.shape[0]
    N = len(comp_indices)
    config_tag = make_config_tag(args)
    timestamp = args.timestamp or datetime.now().strftime("%m%d%Y%H%M%S")

    root_dir = os.path.join("fbirn_results", timestamp, config_tag)
    os.makedirs(root_dir, exist_ok=True)

    print(f"Config: {config_tag}")
    print(f"Saving results to: {root_dir}")

    combined_counts = np.zeros((N, N), dtype=int)
    all_solutions_info = []
    unique_labels = list(pd.unique(labels))

    # Per-group storage
    group_counts = {grp: np.zeros((N, N), dtype=int) for grp in unique_labels}

    for s in range(n_subj):
        ts_2d = data[s, :, comp_indices]
        label = int(labels[s])

        if args.method == "RASL":
            scc_members = get_scc_members(
                args.scc_strategy, comp_indices, ts_2d,
                max_cluster_size=args.corr_max_cluster,
            )
            res_cgs, kept, g_est = run_rasl_subject(
                ts_2d, args, comp_indices,
                scc_members_override=scc_members,
                selection_mode=args.selection_mode,
                top_k=args.top_k,
                delta_multiplier=args.delta_multiplier,
            )
            subj_info = {
                "subject_id": int(s), "group": label,
                "num_solutions": len(kept), "solutions": [],
            }
            for r_idx, (cost, cg, usamp) in enumerate(kept, start=1):
                adj = cg_to_adj_binary(cg)
                combined_counts += adj
                group_counts[label] += adj
                subj_info["solutions"].append({
                    "solution_idx": r_idx, "cost": float(cost),
                    "undersampling": usamp, "graph": cg,
                })

        elif args.method == "PCMCI":
            g_est, A, B = run_pcmci_to_cg(ts_2d)
            adj = cg_to_adj_binary(g_est)
            combined_counts += adj
            group_counts[label] += adj
            subj_info = {
                "subject_id": int(s), "group": label,
                "num_solutions": 1,
                "solutions": [{"solution_idx": 1, "cost": 0.0,
                               "undersampling": None, "graph": g_est}],
            }

        elif args.method == "GCM":
            cg, adj_mat = run_gcm_subject(
                ts_2d, comp_names,
                alpha=args.gcm_alpha, pmax=args.gcm_pmax, n_boot=args.gcm_nboot,
            )
            combined_counts += adj_mat
            group_counts[label] += adj_mat
            subj_info = {
                "subject_id": int(s), "group": label,
                "num_solutions": 1,
                "solutions": [{"solution_idx": 1, "cost": 0.0,
                               "undersampling": None, "graph": cg}],
            }

        all_solutions_info.append(subj_info)
        print(f"  Subject {s}/{n_subj - 1} (group {label}): "
              f"{subj_info['num_solutions']} solution(s)")

    # Save combined results
    combined_dir = os.path.join(root_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    zkl.save(all_solutions_info,
             os.path.join(combined_dir, "all_solutions_info.zkl"))
    np.savez(os.path.join(combined_dir, "group_edge_counts_combined.npz"),
             counts=combined_counts, names=np.array(comp_names))
    _save_edge_csv(combined_counts, comp_names,
                   os.path.join(combined_dir, "group_edge_counts_combined.csv"))
    plot_group_graph(combined_counts, comp_names,
                     os.path.join(combined_dir, "group_graph_weighted_combined.pdf"))

    # Per-group results
    for grp in unique_labels:
        grp_dir = os.path.join(root_dir, f"group_{grp}")
        os.makedirs(grp_dir, exist_ok=True)
        counts = group_counts[grp]
        np.savez(os.path.join(grp_dir, f"group_edge_counts_{grp}.npz"),
                 counts=counts, names=np.array(comp_names))
        _save_edge_csv(counts, comp_names,
                       os.path.join(grp_dir, f"group_edge_counts_{grp}.csv"))
        plot_group_graph(counts, comp_names,
                         os.path.join(grp_dir, f"group_graph_weighted_{grp}.pdf"))

    # Save run parameters
    params = {
        "config_tag": config_tag,
        "method": args.method,
        "n_components": args.n_components,
        "scc_strategy": args.scc_strategy,
        "selection_mode": args.selection_mode,
        "top_k": args.top_k,
        "delta_multiplier": args.delta_multiplier,
        "timestamp": timestamp,
    }
    zkl.save(params, os.path.join(combined_dir, "run_params.zkl"))
    print(f"\nDone. Results in {root_dir}")


def _save_edge_csv(counts, names, path):
    N = len(names)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "count"])
        for i in range(N):
            for j in range(N):
                if counts[i, j] > 0:
                    w.writerow([names[i], names[j], int(counts[i, j])])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_arguments()
    os.environ["OMP_NUM_THREADS"] = str(args.PNUM)

    comp_indices = get_comp_indices(args.n_components)
    comp_names = get_comp_names(comp_indices)

    print("=" * 80)
    print("FMRI EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"  Method:          {args.method}")
    print(f"  N components:    {args.n_components}")
    print(f"  SCC strategy:    {args.scc_strategy}")
    print(f"  Config tag:      {make_config_tag(args)}")
    if args.subject_idx is not None:
        print(f"  Subject index:   {args.subject_idx}")
    else:
        print(f"  Mode:            all subjects (sequential)")
    if args.method == "RASL":
        print(f"  Selection:       {args.selection_mode}"
              f" (top_k={args.top_k}, delta={args.delta_multiplier})")
        print(f"  Max undersamp:   {args.MAXU}")
        print(f"  Priority:        {args.PRIORITY}")
    print("=" * 80)

    npzfile = np.load(args.data_path)
    data = npzfile["data"]          # [n_subjects, T, F]
    labels = get_labels(npzfile)    # [n_subjects]
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")

    if args.subject_idx is not None:
        if args.subject_idx < 0 or args.subject_idx >= data.shape[0]:
            raise ValueError(
                f"subject_idx {args.subject_idx} out of range [0, {data.shape[0] - 1}]"
            )
        run_single_subject(args, data, labels, comp_indices, comp_names)
    else:
        run_all_subjects(args, data, labels, comp_indices, comp_names)
