"""
Aggregate the per-u fMRI RASL runs produced by fmri_experiment_per_u.py.

For each subject this pools every solution across the per-u JSONs
(u2.json, u3.json, …), re-selects the top solutions by cost (the SAME
selection as fmri_experiment_large — top_k or delta_threshold), and writes
``result.zkl`` in the EXACT format of
fmri_experiment_large.run_single_subject, so the existing group-level analysis
(analyze_fmri_experiment.py etc.) works unchanged.

Because clingo's per-u searches partition the combined search space by minimal
undersampling rate, pooling the per-u solutions and re-selecting by cost
reproduces what a single combined drasl() run would have selected — only the
search was parallelised across one job per rate.

Usage:
    # one subject
    python aggregate_fmri_per_u.py --timestamp 06032026 --subject_idx 42 \
        --n_components 20 --scc_strategy domain

    # every subject of this (timestamp, config)
    python aggregate_fmri_per_u.py --timestamp 06032026 \
        --n_components 20 --scc_strategy domain
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gunfolds.utils import bfutils
from gunfolds import conversions as cv
from gunfolds.utils import zickle as zkl


def make_config_tag(n_components, scc_strategy, method="RASL"):
    return f"N{n_components}_{scc_strategy}_{method}"


# ── tiny local copies (avoid importing fmri_experiment_large → tigramite) ────

def cg_to_adj_binary(cg):
    A = cv.graph2adj(cg)
    A = (A > 0).astype(int)
    np.fill_diagonal(A, 0)
    return A


def select_top(solutions, selection_mode="top_k", k=10, delta_multiplier=1.9):
    """solutions: list of (cost, cg, undersampling) sorted ascending by cost.
    Mirrors fmri_experiment_large.select_top_solutions."""
    if not solutions:
        return []
    solutions = sorted(solutions, key=lambda x: x[0])
    if selection_mode == "top_k":
        return solutions[:min(k, len(solutions))]
    elif selection_mode == "delta_threshold":
        min_cost = solutions[0][0]
        threshold = min_cost * delta_multiplier
        selected = [s for s in solutions if s[0] <= threshold]
        return selected if selected else [solutions[0]]
    raise ValueError(f"Unknown selection_mode: {selection_mode}")


# ── aggregation ──────────────────────────────────────────────────────────────

def load_pooled(sdir):
    """Pool all u*.json in a subject dir.  Returns (records, per_u_counts,
    u_status, files).  Each record: dict graph_num/u/cost."""
    files = sorted(glob.glob(os.path.join(sdir, "u*.json")),
                   key=lambda p: int(re.search(r"u(\d+)\.json$", p).group(1)))
    records, per_u_counts, u_status = [], {}, {}
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception as e:
            print(f"    [WARN] could not read {f}: {e}")
            continue
        u = d.get("u")
        u_status[u] = d.get("status", "")
        sols = d.get("solutions", [])
        per_u_counts[u] = len(sols)
        for s in sols:
            records.append({"graph_num": s["graph_num"],
                            "u": s.get("u", u),
                            "cost": s["cost"]})
    return records, per_u_counts, u_status, files


def dedup(records):
    """Collapse identical (graph_num, u) to the lowest cost."""
    best = {}
    for r in records:
        key = (r["graph_num"], r["u"])
        if key not in best or r["cost"] < best[key]["cost"]:
            best[key] = r
    return list(best.values())


def aggregate_subject(sdir, sel_override):
    in_zkl = os.path.join(sdir, "input.zkl")
    if not os.path.exists(in_zkl):
        print(f"  [skip] {sdir}: no input.zkl")
        return None
    records, per_u_counts, u_status, files = load_pooled(sdir)
    if not files:
        print(f"  [skip] {os.path.basename(sdir)}: no u*.json")
        return None

    P = zkl.load(in_zkl)
    n_nodes = P["n_nodes"]
    selection_mode = sel_override.get("selection_mode") or P.get("selection_mode", "top_k")
    top_k = sel_override.get("top_k") or P.get("top_k", 10)
    delta_multiplier = sel_override.get("delta_multiplier") or P.get("delta_multiplier", 1.9)

    records = dedup(records)
    # Build (cost, cg, undersampling) triples and select exactly as the original.
    triples = [(r["cost"], bfutils.num2CG(r["graph_num"], n_nodes), (r["u"],))
               for r in records]
    kept = select_top(triples, selection_mode=selection_mode,
                       k=top_k, delta_multiplier=delta_multiplier)

    # ── result.zkl in the EXACT run_single_subject format ────────────────────
    subject_info = {
        "subject_id": int(P["subject_idx"]),
        "group": P["label"],
        "method": "RASL",
        "config_tag": P["config_tag"],
        "n_components": P["n_components"],
        "scc_strategy": P["scc_strategy"],
        "comp_indices": P["comp_indices"],
        "comp_names": P["comp_names"],
        "gt_density_mode": P.get("gt_density_mode"),
        "gt_density": P.get("gt_density"),
        "gt_density_explicit": P.get("gt_density_explicit"),
        "gt_density_fraction": P.get("gt_density_fraction"),
        "selection_mode": selection_mode,
        "top_k": top_k,
        "solutions": [],
    }
    for r_idx, (cost, cg, usamp) in enumerate(kept, start=1):
        adj = cg_to_adj_binary(cg)
        subject_info["solutions"].append({
            "solution_idx": r_idx,
            "cost": float(cost),
            "undersampling": usamp,
            "graph": cg,
            "adj": adj.tolist(),
        })
    subject_info["g_estimated"] = P["g_estimated"]
    subject_info["num_solutions"] = len(kept)
    # per-u provenance (extra vs the original format; harmless to downstream)
    subject_info["per_u"] = {
        "per_u_counts": per_u_counts,
        "u_status": u_status,
        "n_pooled": len(records),
    }

    zkl.save(subject_info, os.path.join(sdir, "result.zkl"))

    summary = {
        "subject_idx": int(P["subject_idx"]),
        "label": P["label"],
        "config_tag": P["config_tag"],
        "timestamp": P["timestamp"],
        "n_nodes": n_nodes,
        "per_u_counts": per_u_counts,
        "u_status": u_status,
        "n_pooled": len(records),
        "selection_mode": selection_mode,
        "top_k": top_k,
        "num_selected": len(kept),
        "best": ({"graph_num": dedup_min(records)["graph_num"],
                  "u": dedup_min(records)["u"],
                  "cost": dedup_min(records)["cost"]} if records else None),
    }
    with open(os.path.join(sdir, "per_u_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    best_c = summary["best"]["cost"] if summary["best"] else None
    print(f"  subject {P['subject_idx']:>4}: pooled={len(records):4d}  "
          f"selected={len(kept):2d}  per_u={per_u_counts}  best_cost={best_c}")
    return summary


def dedup_min(records):
    return min(records, key=lambda r: r["cost"])


def main():
    p = argparse.ArgumentParser(
        description="Pool per-u fMRI RASL runs and write the standard result.zkl.")
    p.add_argument("--timestamp", required=True)
    p.add_argument("--n_components", type=int, default=20, choices=[10, 20, 53])
    p.add_argument("--scc_strategy", default="domain",
                   choices=["domain", "correlation", "estimated", "none"])
    p.add_argument("--results_root", default="fbirn_results")
    p.add_argument("--subject_idx", type=int, default=None,
                   help="Aggregate one subject; omit to aggregate all found.")
    # Optional selection overrides (default: whatever prep recorded).
    p.add_argument("--selection_mode", default=None,
                   choices=["top_k", "delta_threshold"])
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--delta_multiplier", type=float, default=None)
    args = p.parse_args()

    config_tag = make_config_tag(args.n_components, args.scc_strategy, "RASL")
    config_dir = os.path.join(args.results_root, args.timestamp, config_tag)
    sel_override = {"selection_mode": args.selection_mode,
                    "top_k": args.top_k, "delta_multiplier": args.delta_multiplier}

    if args.subject_idx is not None:
        sdirs = [os.path.join(config_dir, f"subject_{args.subject_idx:04d}")]
    else:
        sdirs = sorted(glob.glob(os.path.join(config_dir, "subject_*")))
        if not sdirs:
            print(f"No subject_* dirs in {config_dir}")
            return

    print(f"Aggregating {len(sdirs)} subject(s) under {config_dir}\n")
    summaries = []
    for sdir in sdirs:
        s = aggregate_subject(sdir, sel_override)
        if s:
            summaries.append(s)

    if summaries:
        master = os.path.join(config_dir, "per_u_aggregated_master.json")
        with open(master, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nWrote {master}  ({len(summaries)} subject(s))")


if __name__ == "__main__":
    main()
