"""
Per-u (undersampling-rate) split of the fMRI RASL experiment
(fmri_experiment_large.py) for cluster parallelisation across all subjects.

Same architecture as runtime_scaling_per_u.py — prep-once / solve-per-u /
aggregate — applied to the FBIRN fMRI pipeline:

    prep   — for ONE subject: load its ICA time series, run PCMCI → g_estimated,
             build the DD/BD penalty matrices, resolve the SCC members and
             GT_density EXACTLY as fmri_experiment_large.run_rasl_subject does,
             and zkl-save that drasl input.  Running PCMCI once per subject (not
             once per u) keeps all the subject's u-jobs on bit-for-bit identical
             input so their costs are comparable.

    solve  — load that input and run drasl(..., fix_urate=u, optim='optN'),
             saving every returned solution (graph, undersampling rate, cost)
             for that single u to u<u>.json.

Aggregation (separate script aggregate_fmri_per_u.py) pools a subject's per-u
JSONs, re-selects the top solutions by cost, and writes result.zkl in the exact
format of fmri_experiment_large.run_single_subject so the existing group-level
analysis is unchanged.

This script is RASL-only — undersampling splitting is meaningless for the
PCMCI / GCM baselines, which the original single-subject mode already handles.

Output layout (per subject), rooted at --results_root (default fbirn_results):
    <root>/<timestamp>/<config_tag>/subject_<idx:04d>/
        input.zkl        (prep)
        u2.json u3.json …(solve, one per rate)
        result.zkl       (aggregate — original format)
        per_u_summary.json

The (timestamp, config_tag, subject_idx) triple is the unique tag that groups a
subject's prep + u-solves + aggregate; the directory path encodes it.

Usage:
    python fmri_experiment_per_u.py prep  --subject_idx 42 --timestamp 06032026 \
        --n_components 20 --scc_strategy domain
    python fmri_experiment_per_u.py solve --subject_idx 42 --timestamp 06032026 \
        --n_components 20 --scc_strategy domain --u_value 3
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np

# Make the repo importable when invoked by path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gunfolds.utils import zickle as zkl
from gunfolds.utils.calc_procs import get_process_count

CLINGO_LIMIT = 64
PNUM_DEFAULT = int(min(CLINGO_LIMIT, get_process_count(1)))
MAXCOST = 20            # mirrors fmri_experiment_large.run_rasl_subject


def make_config_tag(n_components, scc_strategy, method="RASL"):
    return f"N{n_components}_{scc_strategy}_{method}"


def subject_dir(results_root, timestamp, config_tag, subject_idx):
    return os.path.join(results_root, timestamp, config_tag,
                        f"subject_{subject_idx:04d}")


def input_path(sdir):
    return os.path.join(sdir, "input.zkl")


def solve_path(sdir, u_value):
    return os.path.join(sdir, f"u{u_value}.json")


# ═════════════════════════════════════════════════════════════════════════════
# PREP — PCMCI + DD/BD + SCC + GT_density, once per subject
# ═════════════════════════════════════════════════════════════════════════════

def cmd_prep(args):
    # Heavy pipeline imports deferred here so `solve` never needs tigramite.
    import networkx as nx
    from gunfolds import conversions as cv
    from gunfolds.utils import graphkit as gk
    from gunfolds.scripts.real_data.component_config import (
        get_comp_indices, get_comp_names, get_scc_members,
    )
    from gunfolds.scripts.real_data.fmri_experiment_large import (
        run_pcmci_to_cg, get_labels, resolve_fixed_gt_density, resolve_pcmci_alpha,
    )

    config_tag = make_config_tag(args.n_components, args.scc_strategy, "RASL")
    comp_indices = get_comp_indices(args.n_components)
    comp_names = get_comp_names(comp_indices)
    pcmci_alpha = resolve_pcmci_alpha(args.n_components, args.pcmci_alpha)

    sdir = subject_dir(args.results_root, args.timestamp, config_tag, args.subject_idx)
    os.makedirs(sdir, exist_ok=True)
    out = input_path(sdir)

    t0 = time.perf_counter()
    print("=" * 70, flush=True)
    print(f"PREP  {config_tag}  subject={args.subject_idx}  ts={args.timestamp}",
          flush=True)
    print("=" * 70, flush=True)

    npzfile = np.load(args.data_path)
    data = npzfile["data"]            # [n_subjects, T, F]
    labels = get_labels(npzfile)
    if args.subject_idx < 0 or args.subject_idx >= data.shape[0]:
        raise ValueError(f"subject_idx {args.subject_idx} out of range "
                         f"[0, {data.shape[0]-1}]")
    ts_2d = data[args.subject_idx][:, comp_indices]   # [T, N]
    label = int(labels[args.subject_idx])
    n_nodes = len(comp_indices)

    # ── PCMCI → g_estimated, lag matrices ────────────────────────────────────
    g_estimated, A, B = run_pcmci_to_cg(
        ts_2d, pcmci_method=args.pcmci_method, tau_max=args.pcmci_tau_max,
        alpha_level=pcmci_alpha, pc_alpha=args.pcmci_pc_alpha,
        fdr_method=args.pcmci_fdr,
    )

    # ── SCC members (identical dispatch to run_single_subject/run_rasl_subject)
    scc_override = get_scc_members(
        args.scc_strategy, comp_indices, ts_2d, max_cluster_size=args.corr_max_cluster)
    if scc_override is not None:
        members = scc_override
        use_scc = True
    elif args.scc_strategy == "estimated":
        members = list(nx.strongly_connected_components(gk.graph2nx(g_estimated)))
        use_scc = True
    else:
        members = None
        use_scc = False

    # ── DD / BD penalty matrices (canonical, MAXCOST=20) ─────────────────────
    a_max = np.abs(A).max()
    b_max = np.abs(B).max()
    if a_max > 0:
        DD = (np.abs((np.abs(A / a_max) +
                      (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
    else:
        DD = (np.abs((cv.graph2adj(g_estimated) - 1) * MAXCOST)).astype(int)
    if b_max > 0:
        BD = (np.abs((np.abs(B / b_max) +
                      (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)
    else:
        BD = (np.abs((cv.graph2badj(g_estimated) - 1) * MAXCOST)).astype(int)

    # ── GT_density: none / fixed / fraction (mirror run_rasl_subject) ────────
    if args.gt_density_mode == "none":
        gt_density = None
    elif args.gt_density_mode == "fixed":
        gt_density = resolve_fixed_gt_density(len(comp_indices), args.gt_density)
    else:  # fraction
        est_density = gk.density(g_estimated)
        frac = max(0.0, min(1.0, args.gt_density_fraction))
        gt_density = int(100 * est_density * frac)

    priority = [int(c) for c in args.PRIORITY]
    urate = min(args.MAXU, 3 * n_nodes + 1)

    payload = {
        # identity
        "subject_idx": int(args.subject_idx),
        "label": label,
        "timestamp": args.timestamp,
        "config_tag": config_tag,
        "n_components": args.n_components,
        "scc_strategy": args.scc_strategy,
        "method": "RASL",
        "comp_indices": comp_indices,
        "comp_names": comp_names,
        # drasl inputs (rebuild the IDENTICAL call, varying only fix_urate)
        "g_estimated": g_estimated,
        "DD": DD, "BD": BD,
        "use_scc": use_scc,
        "scc_members": members,
        "gt_density": gt_density,
        "gt_density_mode": args.gt_density_mode,
        "gt_density_explicit": (args.gt_density if args.gt_density_mode == "fixed" else None),
        "gt_density_fraction": (args.gt_density_fraction if args.gt_density_mode == "fraction" else None),
        "n_nodes": n_nodes,
        "urate": urate,
        "MAXU": args.MAXU,
        "priority": priority,
        "optim": "optN",
        "selfloop": None,
        # selection (used by the aggregator to match the original output)
        "selection_mode": args.selection_mode,
        "top_k": args.top_k,
        "delta_multiplier": args.delta_multiplier,
        # provenance
        "pcmci_method": args.pcmci_method,
        "pcmci_tau_max": args.pcmci_tau_max,
        "pcmci_alpha": pcmci_alpha,
        "pcmci_pc_alpha": args.pcmci_pc_alpha,
        "pcmci_fdr": args.pcmci_fdr,
        "prep_time_sec": round(time.perf_counter() - t0, 3),
    }
    zkl.save(payload, out)
    print(f"  g_est nodes={len(g_estimated)}  use_scc={use_scc}  "
          f"gt_density={gt_density}  urate={urate}  "
          f"DD[{DD.min()},{DD.max()}]  BD[{BD.min()},{BD.max()}]", flush=True)
    print(f"Wrote {out}  ({payload['prep_time_sec']:.1f}s)", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# SOLVE — one drasl run for a single fixed undersampling rate
# ═════════════════════════════════════════════════════════════════════════════

def cmd_solve(args):
    from gunfolds.solvers.clingo_rasl import drasl

    config_tag = make_config_tag(args.n_components, args.scc_strategy, "RASL")
    sdir = subject_dir(args.results_root, args.timestamp, config_tag, args.subject_idx)
    in_zkl = input_path(sdir)
    if not os.path.exists(in_zkl):
        print(f"!! SOLVE ERROR: prepped input not found: {in_zkl}\n"
              f"   Run `prep` for subject {args.subject_idx} first.", flush=True)
        sys.exit(1)

    P = zkl.load(in_zkl)
    u = args.u_value
    out_json = solve_path(sdir, u)

    result = {
        "subject_idx": P["subject_idx"],
        "label": P["label"],
        "config_tag": config_tag,
        "timestamp": P["timestamp"],
        "u": u,
        "n_nodes": P["n_nodes"],
        "status": "",
        "drasl_time_sec": float("nan"),
        "n_solutions": 0,
        "solutions": [],
    }

    t0 = time.perf_counter()
    print("=" * 70, flush=True)
    print(f"SOLVE {config_tag}  subject={P['subject_idx']}  u={u}  "
          f"pnum={args.PNUM}", flush=True)
    print("=" * 70, flush=True)

    try:
        # Rebuild the IDENTICAL drasl call as fmri_experiment_large.run_rasl_subject,
        # except the undersampling rate is forced to exactly u via fix_urate.
        # density_mode/tol are left at drasl()'s defaults (adaptive ladder), as in
        # the original.
        r = drasl(
            [P["g_estimated"]],
            weighted=True,
            capsize=0,
            timeout=0,
            urate=P["urate"],
            dm=[P["DD"]],
            bdm=[P["BD"]],
            scc=P["use_scc"],
            scc_members=P["scc_members"],
            GT_density=P["gt_density"],
            edge_weights=P["priority"],
            pnum=args.PNUM,
            optim="optN",
            selfloop=None,
            fix_urate=u,
        )
        result["drasl_time_sec"] = round(time.perf_counter() - t0, 4)

        # drasl() (exact=False path) returns a set of ((graph_num, u_tuple), cost)
        # where cost is already sum(cost_vector) — see gunfolds.utils.clingo.clingo.
        sols = []
        for answer, cost in (r or set()):
            graph_num = int(answer[0])
            u_tuple = [int(x) for x in answer[1]]
            sols.append({
                "graph_num": graph_num,
                "u": u,
                "u_parsed": u_tuple,
                "cost": float(cost),
            })
        sols.sort(key=lambda s: s["cost"])
        result["solutions"] = sols
        result["n_solutions"] = len(sols)
        result["status"] = "completed"
        best = sols[0]["cost"] if sols else None
        print(f"  done: {result['drasl_time_sec']:.2f}s  n_solutions={len(sols)}  "
              f"best_cost={best}", flush=True)

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n!! SOLVE ERROR: {exc}\n{tb}", flush=True)
        result["status"] = f"error: {str(exc)[:200]}"

    result["total_time_sec"] = round(time.perf_counter() - t0, 4)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_json}  status={result['status']}  "
          f"n_solutions={result['n_solutions']}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _add_common(p):
    p.add_argument("--subject_idx", type=int, required=True)
    p.add_argument("--timestamp", type=str, required=True,
                   help="Shared timestamp grouping a whole submission.")
    p.add_argument("--n_components", type=int, default=20, choices=[10, 20, 53])
    p.add_argument("--scc_strategy", type=str, default="domain",
                   choices=["domain", "correlation", "estimated", "none"])
    p.add_argument("--results_root", type=str, default="fbirn_results")
    p.add_argument("--PNUM", type=int, default=PNUM_DEFAULT)


def _add_rasl_params(p):
    """RASL/PCMCI params that affect the prepped input — defaults mirror the
    N=20 production config in slurm_fmri_large.sh."""
    p.add_argument("--MAXU", type=int, default=5)
    p.add_argument("--PRIORITY", type=str, default="11112")
    p.add_argument("--selection_mode", default="top_k",
                   choices=["top_k", "delta_threshold"])
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--delta_multiplier", type=float, default=1.9)
    p.add_argument("--gt_density_mode", default="fixed",
                   choices=["none", "fixed", "fraction"])
    p.add_argument("--gt_density", type=int, default=None,
                   help="Fixed GT_density×100. When omitted, the N-specific "
                        "literature default is used (22 for N=20, 13 for N=53).")
    p.add_argument("--gt_density_fraction", type=float, default=1.0)
    p.add_argument("--pcmci_method", default="pcmci", choices=["pcmci", "pcmciplus"])
    p.add_argument("--pcmci_tau_max", type=int, default=1)
    p.add_argument("--pcmci_alpha", type=float, default=0.05)
    p.add_argument("--pcmci_pc_alpha", type=float, default=0.01)
    p.add_argument("--pcmci_fdr", default="none", choices=["none", "fdr_bh"])
    p.add_argument("--corr_max_cluster", type=int, default=8)
    p.add_argument("--data_path", type=str, default="../fbirn/fbirn_sz_data.npz")


def main():
    parser = argparse.ArgumentParser(
        description="Per-u split of the fMRI RASL experiment (prep / solve).")
    sub = parser.add_subparsers(dest="mode", required=True)

    pp_ = sub.add_parser("prep", help="PCMCI + DD/BD + SCC, once per subject.")
    _add_common(pp_)
    _add_rasl_params(pp_)
    pp_.set_defaults(func=cmd_prep)

    sp = sub.add_parser("solve", help="drasl for a single fixed u.")
    _add_common(sp)
    sp.add_argument("--u_value", type=int, required=True,
                    help="The single undersampling rate this job fixes.")
    sp.set_defaults(func=cmd_solve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
