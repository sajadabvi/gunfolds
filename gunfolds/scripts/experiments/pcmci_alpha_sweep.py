"""
PCMCI-only alpha sweep — no drasl, no ground truth.

One run = one alpha value over all subjects. Writes a JSON file with per-subject
PCMCI graph density + edge list, suitable for the Exp 4 composite-score recipe
(0.6 × cross-subject Jaccard + 0.4 × density proximity to N-specific target).

Usage:
    python pcmci_alpha_sweep.py --alpha 0.05 --n_components 53 \\
        --data_path ../fbirn/fbirn_sz_data.npz \\
        --output_dir results/pcmci_alpha_sweep_N53/

Designed to be submitted as a SLURM array (one task per alpha).
PCMCI at N=53 is ~10–60 s per subject, so 311 subjects ≈ 1–6 hours per alpha.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

# Make the repo importable when invoked as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..", "..")
sys.path.insert(0, _ROOT)

from gunfolds.utils import graphkit as gk
from gunfolds.scripts.real_data.component_config import get_comp_indices
from gunfolds.scripts.real_data.fmri_experiment_large import (
    run_pcmci_to_cg, get_labels,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, required=True,
                   help="PCMCI alpha (used for both alpha_level and pc_alpha)")
    p.add_argument("--n_components", type=int, required=True,
                   choices=[10, 13, 20, 53])
    p.add_argument("--data_path", type=str,
                   default="../fbirn/fbirn_sz_data.npz")
    p.add_argument("--output_dir", type=str,
                   default="results/pcmci_alpha_sweep")
    p.add_argument("--pcmci_method", type=str, default="pcmci",
                   choices=["pcmci", "pcmciplus"])
    p.add_argument("--tau_max", type=int, default=1)
    p.add_argument("--fdr", type=str, default="none",
                   choices=["none", "fdr_bh"])
    p.add_argument("--subject_start", type=int, default=0,
                   help="First subject index (inclusive)")
    p.add_argument("--subject_end", type=int, default=-1,
                   help="Last subject index (exclusive); -1 = all")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"alpha_{args.alpha:.4f}.json")

    print("=" * 70, flush=True)
    print(f"pcmci_alpha_sweep — alpha={args.alpha}  N={args.n_components}",
          flush=True)
    print(f"  method={args.pcmci_method} tau_max={args.tau_max} fdr={args.fdr}",
          flush=True)
    print(f"  output: {out_path}", flush=True)
    print(f"  start  {datetime.now()}", flush=True)
    print("=" * 70, flush=True)

    comp_indices = get_comp_indices(args.n_components)
    npz = np.load(args.data_path)
    data = npz["data"]
    labels = get_labels(npz)
    n_subj = data.shape[0]
    end = n_subj if args.subject_end < 0 else min(n_subj, args.subject_end)

    rows = []
    fail_count = 0
    overall_t0 = time.perf_counter()

    for s in range(args.subject_start, end):
        ts_2d = data[s][:, comp_indices]
        t0 = time.perf_counter()
        try:
            g_est, _, _ = run_pcmci_to_cg(
                ts_2d,
                pcmci_method=args.pcmci_method,
                tau_max=args.tau_max,
                alpha_level=args.alpha,
                pc_alpha=args.alpha,
                fdr_method=args.fdr,
            )
            density = float(gk.density(g_est))
            edges = [[int(i), int(j)] for i in g_est for j in g_est[i]]
            rows.append({
                "subject": int(s),
                "group": int(labels[s]),
                "density": density,
                "n_edges": len(edges),
                "edges": edges,
                "pcmci_time_sec": round(time.perf_counter() - t0, 3),
            })
        except Exception as e:
            fail_count += 1
            rows.append({
                "subject": int(s),
                "group": int(labels[s]),
                "error": f"{type(e).__name__}: {e}",
                "pcmci_time_sec": round(time.perf_counter() - t0, 3),
            })
            print(f"  [s={s:03d}] FAILED: {type(e).__name__}: {e}", flush=True)

        if (s + 1) % 25 == 0 or s == end - 1:
            elapsed = time.perf_counter() - overall_t0
            print(f"  progress: {s + 1 - args.subject_start}/{end - args.subject_start}  "
                  f"elapsed={elapsed:.0f}s  fails={fail_count}", flush=True)

    meta = {
        "alpha": args.alpha,
        "n_components": args.n_components,
        "pcmci_method": args.pcmci_method,
        "tau_max": args.tau_max,
        "fdr": args.fdr,
        "data_path": args.data_path,
        "subject_range": [args.subject_start, end],
        "n_subjects": end - args.subject_start,
        "n_failures": fail_count,
        "total_time_sec": round(time.perf_counter() - overall_t0, 2),
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f)
    print(f"\nwrote {out_path}  ({meta['total_time_sec']:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
