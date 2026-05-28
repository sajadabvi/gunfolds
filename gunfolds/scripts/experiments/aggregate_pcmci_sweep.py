"""
Aggregate pcmci_alpha_sweep.py outputs and pick the best alpha.

Score (per Exp 4, fmri_experiment_large.py):
    composite = 0.6 * median(cross-subject Jaccard) + 0.4 * density_proximity

where:
    density_proximity = max(0, 1 - |median_density - target_density| / target_density)

Target density per N matches DEFAULT_GT_DENSITY_BY_N in fmri_experiment_large.py
(divided by 100 to get a 0–1 fraction):
    N=10 → 0.35
    N=20 → 0.22
    N=53 → 0.13

Usage:
    python aggregate_pcmci_sweep.py --input_dir results/pcmci_alpha_sweep_N53 \\
        --n_components 53
"""

import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np


TARGET_DENSITY = {10: 0.35, 20: 0.22, 53: 0.13}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", required=True,
                   help="Directory containing alpha_*.json files")
    p.add_argument("--n_components", type=int, required=True,
                   choices=[10, 20, 53])
    p.add_argument("--max_pairs", type=int, default=10000,
                   help="Cap on Jaccard pairs (full N=311 → 48k pairs; sample if larger)")
    p.add_argument("--output_csv", type=str, default=None,
                   help="Optional path to write a CSV summary")
    return p.parse_args()


def jaccard(a_edges, b_edges):
    A = set(map(tuple, a_edges))
    B = set(map(tuple, b_edges))
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def score_one_file(path, max_pairs):
    with open(path) as f:
        d = json.load(f)
    rows = [r for r in d["rows"] if "edges" in r]   # drop failures
    if len(rows) < 2:
        return None
    densities = np.array([r["density"] for r in rows])
    pairs = list(combinations(range(len(rows)), 2))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    jaccs = [jaccard(rows[i]["edges"], rows[j]["edges"]) for i, j in pairs]
    return {
        "alpha": d["meta"]["alpha"],
        "n_subjects": len(rows),
        "n_failures": d["meta"]["n_failures"],
        "median_density": float(np.median(densities)),
        "iqr_density": float(np.percentile(densities, 75) - np.percentile(densities, 25)),
        "median_jaccard": float(np.median(jaccs)),
        "median_n_edges": float(np.median([r["n_edges"] for r in rows])),
        "total_time_sec": d["meta"]["total_time_sec"],
    }


def main():
    args = parse_args()
    target = TARGET_DENSITY[args.n_components]
    paths = sorted(glob.glob(os.path.join(args.input_dir, "alpha_*.json")))
    if not paths:
        raise SystemExit(f"No alpha_*.json files in {args.input_dir}")

    results = []
    for p in paths:
        r = score_one_file(p, args.max_pairs)
        if r is None:
            print(f"  [skip] {os.path.basename(p)} — fewer than 2 successful subjects")
            continue
        r["target_density"] = target
        r["density_proximity"] = max(
            0.0, 1.0 - abs(r["median_density"] - target) / target
        )
        r["composite"] = 0.6 * r["median_jaccard"] + 0.4 * r["density_proximity"]
        results.append(r)

    results.sort(key=lambda r: -r["composite"])

    hdr = f"{'alpha':>7}  {'jacc':>5}  {'dens':>5}  {'prox':>5}  {'compos':>6}  {'edges':>5}  {'fails':>5}  {'time(s)':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['alpha']:>7.4f}  {r['median_jaccard']:>5.3f}  "
              f"{r['median_density']:>5.3f}  {r['density_proximity']:>5.3f}  "
              f"{r['composite']:>6.3f}  {r['median_n_edges']:>5.0f}  "
              f"{r['n_failures']:>5d}  {r['total_time_sec']:>7.0f}")
    print()
    best = results[0]
    print(f"BEST: alpha={best['alpha']}  composite={best['composite']:.3f}  "
          f"(N={args.n_components}, target_density={target})")

    if args.output_csv:
        import csv
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
