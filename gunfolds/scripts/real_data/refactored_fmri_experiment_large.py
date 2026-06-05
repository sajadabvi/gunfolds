"""
refactored_fmri_experiment_large.py
===================================

Refactored fMRI causal-discovery experiment implementing the cost-weighted
Bayesian-model-averaging pipeline, replacing the legacy "top-k + pooled
edge-frequency" approach.  The legacy script (`fmri_experiment_large.py`) is
kept unchanged for exact reproduction of prior results.

WHY (three weaknesses of the legacy pipeline this fixes):
  1. Legacy ignores the clingo cost (all kept solutions weighted equally).
  2. Legacy pools k solutions/subject and tests with #solutions as the unit ->
     pseudo-replication (effective N inflated ~k-fold, anti-conservative p).
  3. Fixed top-k truncates each subject's solution landscape arbitrarily.

WHAT THIS SCRIPT CHANGES (the experiment / save side):
  1. COST-BAND retention (`--selection_mode cost_band`, default): keep ALL
     solutions within a relative cost band of the per-subject optimum
     (`cost <= c_min*(1+--delta_band)`), capped at `--max_keep`.  This captures
     the full near-optimal set the posterior needs, instead of a truncated 10.
  2. Saves the full retained set WITH per-solution cost + undersampling, plus a
     precomputed per-subject cost-weighted edge POSTERIOR (Boltzmann, `--tau`)
     and underdetermination biomarkers (ESS, edge entropy, penumbra, MAP-u).
  3. Optional BOOTSTRAP stability selection (`--bootstrap B`): block-bootstrap
     the time series B times, re-run PCMCI->RASL, average the posteriors into a
     per-edge stability map (the gold-standard robustness layer).
  4. Explicit undersampling handling: per-solution u saved; posterior
     marginalises over u by default, or conditions on the MAP u (`--map_u_only`).

The PCMCI / GCM / DD-BD front-end is imported verbatim from the legacy module,
so only retention + saved payload + bootstrap differ.

Downstream analysis: analysis/refactored_analyze_fmri_experiment.py.

Usage:
  python refactored_fmri_experiment_large.py --subject_idx 0 --n_components 13 \
      --scc_strategy domain --method RASL --delta_band 0.5 --max_keep 200 \
      --tau 1.0 --bootstrap 0 --timestamp 06042026120000
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx

# Make the repo importable when invoked as a script from real_data/ (local
# debugging); on the cluster gunfolds is conda-installed so this is a no-op.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

from gunfolds.utils import bfutils
from gunfolds import conversions as cv
from gunfolds.solvers.clingo_rasl import drasl
from gunfolds.utils import graphkit as gk
from gunfolds.utils.calc_procs import get_process_count
from gunfolds.utils import zickle as zkl

# Reuse the legacy front-end verbatim (PCMCI, GCM, density/alpha resolution).
from gunfolds.scripts.real_data.fmri_experiment_large import (
    run_pcmci_to_cg, run_gcm_subject, get_labels, cg_to_adj_binary,
    resolve_fixed_gt_density, resolve_pcmci_alpha,
)
from gunfolds.scripts.real_data.component_config import (
    get_comp_indices, get_comp_names, get_scc_members, INDEX_TO_DOMAIN,
)

CLINGO_LIMIT = 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))
MAXCOST = 20


# ---------------------------------------------------------------------------
# Posterior + biomarkers (per subject)
# ---------------------------------------------------------------------------

def _adj_of(cg, n):
    a = (cv.graph2adj(cg) > 0).astype(float)
    np.fill_diagonal(a, 0.0)
    return a


def cost_weighted_posterior(adjs, costs, tau):
    """Boltzmann posterior over a per-subject solution set.

    w_i ~ exp(-(c_i - c_min) / (tau * scale)),  scale = mean positive cost gap.
    Returns (P[NxN] in [0,1], weights w).
    """
    costs = np.asarray(costs, dtype=float)
    dc = costs - costs.min()
    pos = dc[dc > 0]
    scale = pos.mean() if pos.size else 1.0
    if scale <= 0:
        scale = 1.0
    w = np.exp(-dc / (tau * scale))
    w = w / w.sum()
    P = np.tensordot(w, adjs, axes=(0, 0))
    return P, w


def underdetermination_features(adjs, costs, us, w):
    """Per-subject epistemic-uncertainty biomarkers from the solution set."""
    n = adjs.shape[1]
    K = adjs.shape[0]
    freq = adjs.mean(axis=0)                      # unweighted edge frequency
    f = freq[~np.eye(n, dtype=bool)]
    fe = np.clip(f, 1e-9, 1 - 1e-9)
    binent = -(fe * np.log2(fe) + (1 - fe) * np.log2(1 - fe))
    costs = np.asarray(costs, float)
    cmin = costs.min()
    us = np.asarray(us, float)
    best = int(np.argmin(costs))
    return dict(
        n_solutions=int(K),
        ess=float(1.0 / np.sum(w ** 2)),
        cost_spread=float((costs.max() - cmin) / cmin) if cmin > 0 else 0.0,
        mean_edge_entropy=float(binent.mean()),
        penumbra=float(np.mean((f > 0.1) & (f < 0.9))),
        stable_core=float(np.mean(f >= 0.9)),
        map_u=int(us[best]),
        mean_u=float(np.dot(w, us)),
    )


# ---------------------------------------------------------------------------
# RASL with cost-band retention
# ---------------------------------------------------------------------------

def _build_DD_BD(g_estimated, A, B):
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
    return DD, BD


def run_rasl_band(ts_2d, args, comp_indices, scc_members_override):
    """
    Run PCMCI -> drasl and retain the full near-optimal cost band.

    Returns (g_estimated, band) where band is a list of (cost, cg, undersampling)
    sorted ascending by cost, retained by `cost <= c_min*(1+delta_band)` and
    capped at `max_keep`.
    """
    g_estimated, A, B = run_pcmci_to_cg(
        ts_2d, pcmci_method=args.pcmci_method, tau_max=args.pcmci_tau_max,
        alpha_level=args.pcmci_alpha, pc_alpha=args.pcmci_pc_alpha,
        fdr_method=args.pcmci_fdr,
    )
    n_nodes = len(g_estimated)

    if scc_members_override is not None:
        members, use_scc = scc_members_override, True
    elif args.scc_strategy == "estimated":
        members = list(nx.strongly_connected_components(gk.graph2nx(g_estimated)))
        use_scc = True
    else:
        members, use_scc = None, False

    DD, BD = _build_DD_BD(g_estimated, A, B)

    if args.gt_density_mode == "none":
        gt_density = None
    elif args.gt_density_mode == "fixed":
        gt_density = resolve_fixed_gt_density(len(comp_indices), args.gt_density)
    else:
        est = gk.density(g_estimated)
        gt_density = int(100 * est * max(0.0, min(1.0, args.gt_density_fraction)))

    priority = [int(c) for c in args.PRIORITY]
    r = drasl(
        [g_estimated], weighted=True, capsize=0, timeout=0,
        urate=min(args.MAXU, (3 * n_nodes + 1)),
        dm=[DD], bdm=[BD], scc=use_scc, scc_members=members,
        GT_density=gt_density, edge_weights=priority, pnum=args.PNUM,
        optim="optN", selfloop=None,
    )

    sols = []
    for answer in (r or []):
        graph_num, usamp = answer[0][0], answer[0][1]
        cost = answer[1]
        sols.append((cost, bfutils.num2CG(graph_num, n_nodes), usamp))
    sols.sort(key=lambda t: t[0])
    if not sols:
        return g_estimated, []

    cmin = sols[0][0]
    thresh = cmin * (1.0 + args.delta_band) if cmin > 0 else (cmin + args.delta_band_abs)
    band = [s for s in sols if s[0] <= thresh][:args.max_keep]

    if args.map_u_only and band:                  # condition on MAP undersampling
        map_u = band[0][2]
        band = [s for s in band if s[2] == map_u]
    return g_estimated, band


def _u_int(u):
    return int(u[0]) if isinstance(u, (tuple, list)) else int(u)


def band_to_payload(band, n, tau):
    """Turn a retained band into solutions list + posterior + features."""
    adjs = np.stack([_adj_of(cg, n) for _, cg, _ in band])
    costs = np.array([c for c, _, _ in band], dtype=float)
    us = np.array([_u_int(u) for _, _, u in band], dtype=float)
    P, w = cost_weighted_posterior(adjs, costs, tau)
    feats = underdetermination_features(adjs, costs, us, w)
    solutions = []
    for idx, (cost, cg, usamp) in enumerate(band, start=1):
        a = _adj_of(cg, n).astype(int)
        solutions.append({
            "solution_idx": idx, "cost": float(cost), "undersampling": usamp,
            "graph": cg, "adj": a.tolist(),
        })
    return solutions, P, feats


# ---------------------------------------------------------------------------
# Bootstrap stability
# ---------------------------------------------------------------------------

def block_bootstrap(ts_2d, block_len, rng):
    T, N = ts_2d.shape
    out = np.empty_like(ts_2d)
    i = 0
    while i < T:
        start = rng.randint(0, max(1, T - block_len + 1))
        L = min(block_len, T - i)
        out[i:i + L] = ts_2d[start:start + L]
        i += L
    return out


def bootstrap_stability(ts_2d, args, comp_indices, scc_members, B, tau, seed=0):
    """Average the cost-weighted posterior over B block-bootstraps."""
    n = len(comp_indices)
    rng = np.random.RandomState(seed)
    acc = np.zeros((n, n))
    done = 0
    for b in range(B):
        try:
            tsb = block_bootstrap(ts_2d, args.block_len, rng)
            _, band = run_rasl_band(tsb, args, comp_indices, scc_members)
            if not band:
                continue
            adjs = np.stack([_adj_of(cg, n) for _, cg, _ in band])
            costs = np.array([c for c, _, _ in band], dtype=float)
            P, _ = cost_weighted_posterior(adjs, costs, tau)
            acc += P
            done += 1
        except Exception as e:
            print(f"    [bootstrap {b}] failed: {e}", flush=True)
    return (acc / done) if done else acc, done


# ---------------------------------------------------------------------------
# Single subject
# ---------------------------------------------------------------------------

def make_config_tag(args):
    return f"N{args.n_components}_{args.scc_strategy}_{args.method}"


def run_single_subject(args, data, labels, comp_indices, comp_names):
    s = args.subject_idx
    ts_2d = data[s][:, comp_indices]
    n = len(comp_indices)
    label = int(labels[s])
    config_tag = make_config_tag(args)
    timestamp = args.timestamp or datetime.now().strftime("%m%d%Y%H%M%S")
    out_dir = os.path.join("fbirn_results_refactored", timestamp, config_tag,
                           f"subject_{s:04d}")
    os.makedirs(out_dir, exist_ok=True)

    info = {
        "subject_id": int(s), "group": label, "method": args.method,
        "config_tag": config_tag, "n_components": args.n_components,
        "scc_strategy": args.scc_strategy, "comp_indices": comp_indices,
        "comp_names": comp_names, "domains": [INDEX_TO_DOMAIN.get(c, "?") for c in comp_indices],
        "selection_mode": args.selection_mode, "delta_band": args.delta_band,
        "max_keep": args.max_keep, "temperature": args.tau,
        "map_u_only": args.map_u_only,
        "gt_density_mode": args.gt_density_mode,
        "gt_density": (resolve_fixed_gt_density(n, args.gt_density)
                       if args.gt_density_mode == "fixed" else None),
        "solutions": [], "posterior": None, "features": None,
        "bootstrap_stability": None, "bootstrap_B": args.bootstrap,
    }

    if args.method == "RASL":
        scc_members = get_scc_members(args.scc_strategy, comp_indices, ts_2d,
                                      max_cluster_size=args.corr_max_cluster)
        g_est, band = run_rasl_band(ts_2d, args, comp_indices, scc_members)
        if not band:
            print(f"  WARN subject {s}: no RASL solutions.");
        else:
            sols, P, feats = band_to_payload(band, n, args.tau)
            info["solutions"] = sols
            info["posterior"] = P.tolist()
            info["features"] = feats
            info["num_solutions"] = len(sols)
            info["g_estimated"] = g_est
            if args.bootstrap > 0:
                stab, done = bootstrap_stability(
                    ts_2d, args, comp_indices, scc_members,
                    args.bootstrap, args.tau, seed=s)
                info["bootstrap_stability"] = stab.tolist()
                info["bootstrap_done"] = done

    elif args.method == "PCMCI":
        g_est, A, B = run_pcmci_to_cg(
            ts_2d, pcmci_method=args.pcmci_method, tau_max=args.pcmci_tau_max,
            alpha_level=args.pcmci_alpha, pc_alpha=args.pcmci_pc_alpha,
            fdr_method=args.pcmci_fdr)
        adj = cg_to_adj_binary(g_est)
        info["solutions"] = [{"solution_idx": 1, "cost": 0.0, "undersampling": None,
                              "graph": g_est, "adj": adj.tolist()}]
        info["posterior"] = adj.astype(float).tolist()
        info["features"] = dict(n_solutions=1, ess=1.0, cost_spread=0.0,
                                mean_edge_entropy=0.0, penumbra=0.0,
                                stable_core=float((adj > 0).mean()), map_u=1, mean_u=1.0)
        info["num_solutions"] = 1
        info["g_estimated"] = g_est

    elif args.method == "GCM":
        cg, adj = run_gcm_subject(ts_2d, comp_names, alpha=args.gcm_alpha,
                                  pmax=args.gcm_pmax, n_boot=args.gcm_nboot)
        info["solutions"] = [{"solution_idx": 1, "cost": 0.0, "undersampling": None,
                              "graph": cg, "adj": adj.tolist()}]
        info["posterior"] = adj.astype(float).tolist()
        info["features"] = dict(n_solutions=1, ess=1.0, cost_spread=0.0,
                                mean_edge_entropy=0.0, penumbra=0.0,
                                stable_core=float((adj > 0).mean()), map_u=1, mean_u=1.0)
        info["num_solutions"] = 1

    zkl.save(info, os.path.join(out_dir, "result.zkl"))
    nb = info.get("num_solutions", 0)
    print(f"[{config_tag}] subject {s} (group {label}): {nb} solution(s) "
          f"band(delta<={args.delta_band}) saved to {out_dir}", flush=True)


def run_all_subjects(args, data, labels, comp_indices, comp_names):
    for s in range(data.shape[0]):
        args.subject_idx = s
        run_single_subject(args, data, labels, comp_indices, comp_names)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_arguments():
    p = argparse.ArgumentParser(description="Refactored fMRI causal-discovery "
                                "experiment (cost-band retention + posterior).")
    p.add_argument("--n_components", type=int, default=10, choices=[10, 13, 20, 53])
    p.add_argument("--scc_strategy", default="domain",
                   choices=["domain", "correlation", "estimated", "none"])
    p.add_argument("--method", default="RASL", choices=["RASL", "PCMCI", "GCM"])
    p.add_argument("--subject_idx", type=int, default=0)
    p.add_argument("--timestamp", default=None)
    p.add_argument("--data_path", default="../fbirn/fbirn_sz_data.npz")
    p.add_argument("--all_subjects", action="store_true",
                   help="Run every subject sequentially (omit --subject_idx use).")

    # RASL
    p.add_argument("-p", "--PNUM", default=PNUM, type=int)
    p.add_argument("-x", "--MAXU", default=5, type=int)
    p.add_argument("-y", "--PRIORITY", default="11112", type=str)

    # NEW: cost-band retention + posterior
    p.add_argument("--selection_mode", default="cost_band",
                   choices=["cost_band", "top_k"],
                   help="cost_band (default): keep all within delta of optimum.")
    p.add_argument("--delta_band", default=0.5, type=float,
                   help="Relative cost band: keep cost <= c_min*(1+delta_band).")
    p.add_argument("--delta_band_abs", default=1.0, type=float,
                   help="Absolute band fallback when c_min<=0.")
    p.add_argument("--max_keep", default=200, type=int,
                   help="Cap on retained solutions per subject.")
    p.add_argument("--tau", default=1.0, type=float,
                   help="Boltzmann temperature for the cost-weighted posterior.")
    p.add_argument("--map_u_only", action="store_true",
                   help="Condition posterior on the MAP undersampling rate.")

    # NEW: bootstrap stability
    p.add_argument("--bootstrap", default=0, type=int,
                   help="Block-bootstrap reps for stability selection (0=off).")
    p.add_argument("--block_len", default=20, type=int,
                   help="Block length for the moving-block time-series bootstrap.")

    # GT density
    p.add_argument("--gt_density_mode", default="fixed",
                   choices=["none", "fixed", "fraction"])
    p.add_argument("--gt_density", default=None, type=int)
    p.add_argument("--gt_density_fraction", default=1.0, type=float)

    # PCMCI
    p.add_argument("--pcmci_method", default="pcmci", choices=["pcmci", "pcmciplus"])
    p.add_argument("--pcmci_tau_max", default=1, type=int)
    p.add_argument("--pcmci_alpha", default=None, type=float)
    p.add_argument("--pcmci_pc_alpha", default=0.01, type=float)
    p.add_argument("--pcmci_fdr", default="none", choices=["none", "fdr_bh"])

    # GCM
    p.add_argument("--gcm_alpha", default=0.01, type=float)
    p.add_argument("--gcm_pmax", default=8, type=int)
    p.add_argument("--gcm_nboot", default=200, type=int)

    # correlation SCC
    p.add_argument("--corr_max_cluster", default=8, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    os.environ["OMP_NUM_THREADS"] = str(args.PNUM)

    comp_indices = get_comp_indices(args.n_components)
    comp_names = get_comp_names(comp_indices)
    args.pcmci_alpha = resolve_pcmci_alpha(args.n_components, args.pcmci_alpha)

    print("=" * 80)
    print("REFACTORED FMRI EXPERIMENT (cost-band retention + posterior)")
    print("=" * 80)
    print(f"  Method:        {args.method}   N={args.n_components}   "
          f"SCC={args.scc_strategy}")
    print(f"  Retention:     cost_band delta<={args.delta_band} "
          f"(max_keep={args.max_keep}), tau={args.tau}, map_u_only={args.map_u_only}")
    print(f"  Bootstrap:     {args.bootstrap} reps (block_len={args.block_len})")
    if args.method in ("RASL", "PCMCI"):
        print(f"  PCMCI:         {args.pcmci_method} tau_max={args.pcmci_tau_max} "
              f"alpha={args.pcmci_alpha} pc_alpha={args.pcmci_pc_alpha} fdr={args.pcmci_fdr}")
    print("=" * 80)

    npz = np.load(args.data_path)
    data = npz["data"]
    labels = get_labels(npz)
    print(f"Data: {data.shape}, labels: {labels.shape}")

    if args.all_subjects:
        run_all_subjects(args, data, labels, comp_indices, comp_names)
    else:
        if args.subject_idx < 0 or args.subject_idx >= data.shape[0]:
            raise ValueError(f"subject_idx {args.subject_idx} out of range")
        run_single_subject(args, data, labels, comp_indices, comp_names)
