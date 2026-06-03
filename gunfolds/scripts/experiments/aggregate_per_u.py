"""
Aggregate the per-u (undersampling-rate) clingo runs produced by
``runtime_scaling_per_u.py`` and keep the top fraction of solutions by cost.

For each run_tag, this script:
  1. globs ``<tag>__u*.json`` (one file per fixed undersampling rate),
  2. pools EVERY solution from every u-rate into one list — each solution is a
     (graph, undersampling-rate, clingo-cost) triple,
  3. sorts the pooled list by least cost (``cost_total`` = sum of the clingo
     cost vector, matching runtime_scaling.py's convention),
  4. keeps the top fraction (default 30 %) — i.e. the lowest-cost solutions,
  5. writes:
       <tag>__top<pct>.csv          — the kept solutions, ranked
       <tag>__aggregated.json       — summary (per-u counts, best solution,
                                       and F1 vs ground truth if the prep
                                       pickle <tag>__input.pkl is present).

The per-u JSONs are an exact partition of the combined-search SPACE (each
causal graph has a unique minimal undersampling rate), so pooling them loses
no candidate solution.

Cost ranking (``--rank``) — IMPORTANT
-------------------------------------
clingo optimises the cost vector LEXICOGRAPHICALLY (priority level by priority
level), but runtime_scaling.py ranks solutions by ``sum(cost)``.  These can
disagree: a model may have a smaller sum yet a worse lexicographic cost (e.g.
a lower priority-0 density term but a higher top-priority edge term).  So:

  --rank sum  (default) reproduces runtime_scaling.py's ``sum(cost)`` ordering
              and may surface a lower-sum solution than a single combined run
              reported as "optimal".
  --rank lex  uses clingo's native objective; taking the lex-min across the
              pooled per-u solutions reproduces the combined single-run optimum
              exactly.

Both CSVs/summaries carry the full cost vector, so you can always re-rank.

Usage:
    # one tag
    python aggregate_per_u.py --input_dir results/runtime_scaling_per_u/ \
        --run_tag rtpu_n14_inst3

    # every tag found in the directory
    python aggregate_per_u.py --input_dir results/runtime_scaling_per_u/

    # different fraction
    python aggregate_per_u.py --input_dir results/... --top_frac 0.5
"""

import argparse
import csv
import glob
import json
import math
import os
import pickle
import sys

import numpy as np

# Make the repo importable when invoked by path (so the optional F1 step can
# `import gunfolds.utils.bfutils` regardless of the caller's cwd).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# F1 helpers — mirror runtime_scaling.py exactly (kept local so aggregation
# stays dependency-light: numpy + gunfolds.bfutils only, no tigramite/BOLD).
# ─────────────────────────────────────────────────────────────────────────────

def graph_to_dir_adj(g, n):
    adj = np.zeros((n, n), dtype=int)
    for i, nbrs in g.items():
        for j, v in nbrs.items():
            if v in (1, 3):
                adj[i - 1, j - 1] = 1
    return adj


def f1_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_solution(pred_g, gt_g, n):
    pred = graph_to_dir_adj(pred_g, n)
    true = graph_to_dir_adj(gt_g, n)
    np.fill_diagonal(pred, 0)
    np.fill_diagonal(true, 0)
    tp_o = int(np.sum((pred == 1) & (true == 1)))
    fp_o = int(np.sum((pred == 1) & (true == 0)))
    fn_o = int(np.sum((pred == 0) & (true == 1)))
    orientation_f1 = f1_from_counts(tp_o, fp_o, fn_o)
    pred_u = ((pred + pred.T) > 0).astype(int)
    true_u = ((true + true.T) > 0).astype(int)
    iu = np.triu_indices(n, k=1)
    tp_a = int(np.sum((pred_u[iu] == 1) & (true_u[iu] == 1)))
    fp_a = int(np.sum((pred_u[iu] == 1) & (true_u[iu] == 0)))
    fn_a = int(np.sum((pred_u[iu] == 0) & (true_u[iu] == 1)))
    adjacency_f1 = f1_from_counts(tp_a, fp_a, fn_a)
    return adjacency_f1, orientation_f1


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def discover_run_tags(input_dir):
    """Tags are the stems of the per-u JSONs: ``<tag>__u<k>.json``.  We derive
    tags from those (not just the prep pickles) so a run is still aggregatable
    even if its input pickle was cleaned up."""
    tags = set()
    for f in glob.glob(os.path.join(input_dir, "*__u*.json")):
        base = os.path.basename(f)
        idx = base.rfind("__u")
        if idx > 0:
            tags.add(base[:idx])
    return sorted(tags)


def load_solutions(input_dir, run_tag):
    """Pool every solution across all <tag>__u*.json files.

    Returns (pooled, per_u_counts, files, n_timeout, n_error).
    """
    pooled = []
    per_u_counts = {}
    files = sorted(glob.glob(os.path.join(input_dir, f"{run_tag}__u*.json")))
    n_timeout = n_error = 0
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"  [WARN] could not read {f}: {e}")
            continue
        status = data.get("status", "")
        if status == "timeout":
            n_timeout += 1
        elif status.startswith("error"):
            n_error += 1
        u = data.get("u")
        sols = data.get("solutions", [])
        per_u_counts[u] = len(sols)
        for s in sols:
            pooled.append({
                "graph_num": s["graph_num"],
                "u": s.get("u", u),
                "cost_total": s["cost_total"],
                "cost_vector": s.get("cost_vector", []),
            })
    return pooled, per_u_counts, files, n_timeout, n_error


def dedup(pooled):
    """Defensive: collapse identical (graph_num, u) to the lowest cost_total."""
    best = {}
    for s in pooled:
        key = (s["graph_num"], s["u"])
        if key not in best or s["cost_total"] < best[key]["cost_total"]:
            best[key] = s
    return list(best.values())


def _sort_key(rank):
    """Return the sort key fn for the chosen ranking mode.

    rank='sum' : by cost_total = sum(cost vector).  Matches the
                 runtime_scaling.py convention (``scored.sort(key=sum(cost))``).
                 NOTE: this is NOT what clingo optimises.
    rank='lex' : by the full cost vector (clingo orders it high→low priority,
                 so tuple comparison == clingo's lexicographic objective).
                 Ranking this way reproduces the combined single-run optimum.

    The two can disagree: a model can have a lower priority-0 (e.g. density)
    term but a higher top-priority term, giving it a smaller sum yet a worse
    lexicographic cost.  See the module note / wiki.
    """
    if rank == "lex":
        return lambda s: (list(s["cost_vector"]), s["cost_total"], s["graph_num"])
    return lambda s: (s["cost_total"], list(s["cost_vector"]), s["graph_num"])


def rank_and_keep(pooled, top_frac, rank="sum"):
    """Sort by least cost (per ``rank``); keep the top fraction."""
    pooled_sorted = sorted(pooled, key=_sort_key(rank))
    n = len(pooled_sorted)
    n_keep = max(1, math.ceil(top_frac * n)) if n else 0
    return pooled_sorted, pooled_sorted[:n_keep], n_keep


def aggregate_one(input_dir, run_tag, top_frac, rank="sum"):
    pooled, per_u_counts, files, n_timeout, n_error = load_solutions(
        input_dir, run_tag)
    if not files:
        print(f"[{run_tag}] no <tag>__u*.json files found — skipping")
        return None
    pooled = dedup(pooled)
    pooled_sorted, kept, n_keep = rank_and_keep(pooled, top_frac, rank=rank)

    summary = {
        "run_tag": run_tag,
        "n_u_files": len(files),
        "per_u_counts": per_u_counts,
        "n_pooled": len(pooled),
        "top_frac": top_frac,
        "rank": rank,
        "n_kept": n_keep,
        "n_timeout_files": n_timeout,
        "n_error_files": n_error,
        "best": kept[0] if kept else None,
        "u_rates_present": sorted(per_u_counts.keys()),
    }

    # Optional ground-truth F1 for the single best pooled solution.
    in_pkl = os.path.join(input_dir, f"{run_tag}__input.pkl")
    if kept and os.path.exists(in_pkl):
        try:
            from gunfolds.utils import bfutils
            with open(in_pkl, "rb") as f:
                P = pickle.load(f)
            n_nodes = P["n_nodes"]
            best_pred_g = bfutils.num2CG(kept[0]["graph_num"], n_nodes)
            adj_f1, orient_f1 = evaluate_solution(best_pred_g, P["gt_g"], n_nodes)
            summary["best_adjacency_f1"] = round(adj_f1, 4)
            summary["best_orientation_f1"] = round(orient_f1, 4)
            summary["best_u"] = kept[0]["u"]
            summary["n_nodes"] = n_nodes
            summary["instance_id"] = P.get("instance_id")
        except Exception as e:
            print(f"  [WARN] F1 computation failed for {run_tag}: {e}")

    # Write top-N CSV.
    pct = int(round(top_frac * 100))
    csv_path = os.path.join(input_dir, f"{run_tag}__top{pct}_{rank}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "graph_num", "u", "cost_total", "cost_vector"])
        for i, s in enumerate(kept, start=1):
            w.writerow([i, s["graph_num"], s["u"], s["cost_total"],
                        json.dumps(s["cost_vector"]).replace(",", ";")])

    # Write summary JSON.
    json_path = os.path.join(input_dir, f"{run_tag}__aggregated.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    f1s = ""
    if "best_orientation_f1" in summary:
        f1s = (f"  best_F1(adj/orient)={summary['best_adjacency_f1']}/"
               f"{summary['best_orientation_f1']}  best_u={summary.get('best_u')}")
    print(f"[{run_tag}] pooled={len(pooled):4d}  kept(top{pct}%)={n_keep:4d}  "
          f"per_u={per_u_counts}"
          f"{('  best_cost=' + str(summary['best']['cost_total'])) if summary['best'] else ''}"
          f"{f1s}")
    return summary


def main():
    p = argparse.ArgumentParser(
        description="Pool per-u clingo runs and keep the top fraction by cost.")
    p.add_argument("--input_dir", default="results/runtime_scaling_per_u/")
    p.add_argument("--run_tag", default=None,
                   help="Aggregate just this tag. If omitted, every tag found.")
    p.add_argument("--top_frac", type=float, default=0.30,
                   help="Fraction of lowest-cost solutions to keep (default 0.30).")
    p.add_argument("--rank", choices=["sum", "lex"], default="sum",
                   help="Cost ranking. 'sum' = sum(cost vector) "
                        "(runtime_scaling.py convention, default). "
                        "'lex' = clingo's native lexicographic cost "
                        "(reproduces the combined single-run optimum).")
    args = p.parse_args()

    if not 0 < args.top_frac <= 1:
        p.error("--top_frac must be in (0, 1].")

    if args.run_tag:
        tags = [args.run_tag]
    else:
        tags = discover_run_tags(args.input_dir)
        if not tags:
            print(f"No <tag>__u*.json files found in {args.input_dir}")
            return

    print(f"Aggregating {len(tags)} run_tag(s) from {args.input_dir} "
          f"(top {int(round(args.top_frac*100))}% by {args.rank}-cost)\n")
    summaries = []
    for tag in tags:
        s = aggregate_one(args.input_dir, tag, args.top_frac, rank=args.rank)
        if s:
            summaries.append(s)

    # Master index of all aggregated runs.
    if summaries:
        master = os.path.join(args.input_dir,
                              f"per_u_aggregated_master_{args.rank}.json")
        with open(master, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nWrote {master}  ({len(summaries)} run_tag(s))")


if __name__ == "__main__":
    main()
