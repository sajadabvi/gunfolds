"""
Diagnose why some subjects solve much faster than others under the production
DRASL encoding.

Empirically observed (2026-05-04):
  N=10, --optim opt, density_mode='hard_soft0', tol_low=15, tol_high=5
    subject 0: baseline times out (>>1 min, cost still descending past 421)
    subject 1: baseline solves to proven optimum [608] in 1.43 s

Subject-level features that *should* correlate with solve time:

  1. PCMCI prior sharpness (DD/BD weight distribution).
     Bimodal at {0, MAXCOST}  -> most edges have an obvious choice -> easy.
     Concentrated mid-range   -> most edges are ambiguous           -> hard.

  2. Hard density window position.
     g_estimated density inside [GT-tol_low, GT+tol_high]  -> PCMCI prior
     is feasible -> solver doesn't fight the cardinality constraint.
     Outside the window                                    -> harder.

  3. SCC size distribution.
     Determined by --scc_strategy and which components are picked, *not* by
     subject data. If --scc_strategy='domain', expected to be identical
     across subjects at fixed N. Reported for confirmation only.

  4. Cost-landscape flatness.
     # of distinct DD weight values, std of weights. Flat (many equal weights)
     means many cost-tied solutions -> enumeration plateau in proof phase.

  5. Edge-count match.
     Distance between |edges(g_estimated)| and the *centre* of the hard
     window. Larger distance -> bigger surgery the solver must perform.

Usage:
  python diagnose_subject_difficulty.py --n_components 10 --subjects 0,1
  python diagnose_subject_difficulty.py --n_components 14 --subjects 0,1,2,3
"""

import os
import sys
import argparse
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "real_data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from gunfolds import conversions as cv
from gunfolds.utils import graphkit as gk

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from gunfolds.scripts.real_data.component_config import (
    get_comp_indices, get_scc_members,
)

MAXCOST = 20
DEFAULT_GT_DENSITY_BY_N = {10: 35, 13: 32, 14: 30, 20: 22, 53: 13}
DEFAULT_PCMCI_ALPHA_BY_N = {10: 0.08, 20: 0.05, 53: 0.05}  # swept per-N (pcmci_alpha_sweep.py); N not in table -> 0.05


def histogram_str(values, bins=10, lo=0, hi=20, width=40):
    """ASCII histogram for a list/array of integers."""
    if len(values) == 0:
        return "  (no values)"
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    peak = max(counts.max(), 1)
    out = []
    for i, c in enumerate(counts):
        bar = "#" * int(width * c / peak)
        out.append(f"  [{edges[i]:5.1f}-{edges[i+1]:5.1f}]  "
                   f"{c:>5d} {bar}")
    return "\n".join(out)


def analyze_subject(subject_idx, data, comp_indices, n_nodes, gt_density,
                    pcmci_method, pcmci_tau_max, pcmci_alpha, pcmci_fdr,
                    scc_strategy, tol_low, tol_high):
    ts_2d = data[subject_idx][:, comp_indices]

    # --- PCMCI ---
    dataframe = pp.DataFrame(ts_2d)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    if pcmci_method == "pcmciplus":
        results = pcmci.run_pcmciplus(
            tau_max=pcmci_tau_max, pc_alpha=0.01, fdr_method=pcmci_fdr)
    else:
        results = pcmci.run_pcmci(
            tau_max=pcmci_tau_max, pc_alpha=None,
            alpha_level=pcmci_alpha, fdr_method=pcmci_fdr)
    g_estimated, A, B = cv.Glag2CG(results)

    density = gk.density(g_estimated)
    n_dir = sum(1 for n in g_estimated for t in g_estimated[n]
                if g_estimated[n][t] in (1, 3))
    n_bidir = sum(1 for n in g_estimated for t in g_estimated[n]
                  if g_estimated[n][t] in (2, 3) and n < t)

    # --- DD / BD matrices (same recipe as the benchmark) ---
    a_max = np.abs(A).max()
    b_max = np.abs(B).max()
    if a_max > 0:
        DD = (np.abs((np.abs(A / a_max) + (cv.graph2adj(g_estimated) - 1))
                     * MAXCOST)).astype(int)
    else:
        DD = (np.abs((cv.graph2adj(g_estimated) - 1) * MAXCOST)).astype(int)
    if b_max > 0:
        BD = (np.abs((np.abs(B / b_max) + (cv.graph2badj(g_estimated) - 1))
                     * MAXCOST)).astype(int)
    else:
        BD = (np.abs((cv.graph2badj(g_estimated) - 1) * MAXCOST)).astype(int)

    # Off-diagonal only (self-loops disabled in the encoding)
    mask = ~np.eye(n_nodes, dtype=bool)
    DD_off = DD[mask]
    BD_off = BD[mask]

    # --- Hard density window math ---
    n_sq = n_nodes * n_nodes
    d_lo_pct = max(0, gt_density - tol_low)
    d_hi_pct = min(100, gt_density + tol_high)
    d_lo_edges = int(d_lo_pct * n_sq / 100)
    d_hi_edges = int(d_hi_pct * n_sq / 100) + 1
    cur_edges = n_dir
    in_window = d_lo_edges <= cur_edges <= d_hi_edges
    window_centre = (d_lo_edges + d_hi_edges) / 2
    edges_to_centre = abs(cur_edges - window_centre)
    feasible_cardinalities = max(0, d_hi_edges - d_lo_edges + 1)

    # --- Sharpness metrics ---
    # "Sharp" = weights are close to 0 or close to MAXCOST.
    # ambiguous_count: # of off-diag DD entries strictly between 4 and 16
    DD_low = int(np.sum(DD_off <= 4))
    DD_high = int(np.sum(DD_off >= 16))
    DD_mid = int(np.sum((DD_off > 4) & (DD_off < 16)))
    sharp_ratio = (DD_low + DD_high) / max(len(DD_off), 1)

    DD_unique = len(np.unique(DD_off))

    # --- SCC structure ---
    scc_members = get_scc_members(scc_strategy, comp_indices, ts_2d,
                                  max_cluster_size=8)
    scc_sizes = sorted([len(s) for s in scc_members],
                       reverse=True) if scc_members else []

    # --- Total weak-constraint cost we'd see if PCMCI were exactly right ---
    # Ideal cost: edges PCMCI says present that solver also says present have
    # cost = DD weight * 0 (matched). So this is a measure of how much
    # *unavoidable* mismatch the prior implies (irrelevant edges in DD already
    # contribute 0 if matched).
    # Simpler proxy: total |DD| over off-diagonal — sum of all edge weights.
    # Higher  -> more "evidence mass" the solver must shape.
    total_dd_mass = int(DD_off.sum())
    total_bd_mass = int(BD_off.sum())

    return {
        "subject": subject_idx,
        "density": density,
        "n_dir": n_dir,
        "n_bidir": n_bidir,
        "DD_off": DD_off,
        "BD_off": BD_off,
        "DD_min": int(DD_off.min()),
        "DD_max": int(DD_off.max()),
        "DD_mean": float(DD_off.mean()),
        "DD_std": float(DD_off.std()),
        "DD_low": DD_low,
        "DD_mid": DD_mid,
        "DD_high": DD_high,
        "DD_unique": DD_unique,
        "sharp_ratio": sharp_ratio,
        "BD_mean": float(BD_off.mean()),
        "BD_std": float(BD_off.std()),
        "scc_sizes": scc_sizes,
        "n_sq": n_sq,
        "cur_edges": cur_edges,
        "d_lo_edges": d_lo_edges,
        "d_hi_edges": d_hi_edges,
        "in_window": in_window,
        "window_centre": window_centre,
        "edges_to_centre": edges_to_centre,
        "feasible_cardinalities": feasible_cardinalities,
        "total_dd_mass": total_dd_mass,
        "total_bd_mass": total_bd_mass,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_components", type=int, default=10,
                   choices=[10, 13, 14, 20, 53])
    p.add_argument("--subjects", type=str, default="0,1",
                   help="Comma-separated subject indices to compare")
    p.add_argument("--data_path", type=str, default="../fbirn/fbirn_sz_data.npz")
    p.add_argument("--scc_strategy", type=str, default="domain",
                   choices=["domain", "correlation", "estimated", "none"])
    p.add_argument("--gt_density", type=int, default=None)
    p.add_argument("--tol_low", type=int, default=15)
    p.add_argument("--tol_high", type=int, default=5)
    p.add_argument("--pcmci_method", default="pcmci")
    p.add_argument("--pcmci_tau_max", type=int, default=1)
    p.add_argument("--pcmci_alpha", type=float, default=None,
                   help="PCMCI significance level. Omit to use the swept "
                        "per-N default (DEFAULT_PCMCI_ALPHA_BY_N).")
    p.add_argument("--pcmci_fdr", default="none")
    args = p.parse_args()
    if args.pcmci_alpha is None:
        args.pcmci_alpha = DEFAULT_PCMCI_ALPHA_BY_N.get(args.n_components, 0.05)

    subject_idxs = [int(x) for x in args.subjects.split(",") if x.strip()]
    gt_density = args.gt_density or DEFAULT_GT_DENSITY_BY_N[args.n_components]

    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "..", "real_data",
                                 data_path)
    npzfile = np.load(data_path)
    data = npzfile["data"]
    labels_key = "labels" if "labels" in npzfile.files else "label"
    labels = npzfile[labels_key]

    comp_indices = get_comp_indices(args.n_components)
    n_nodes = len(comp_indices)

    print("=" * 80)
    print(f"SUBJECT-DIFFICULTY DIAGNOSTIC — N={args.n_components}, "
          f"subjects={subject_idxs}")
    print("=" * 80)
    print(f"  GT_density={gt_density}  tol_low={args.tol_low}  "
          f"tol_high={args.tol_high}")
    print(f"  Hard window: [{max(0, gt_density-args.tol_low)}%, "
          f"{min(100, gt_density+args.tol_high)}%] of N²={n_nodes**2}")
    print()

    results = []
    for s in subject_idxs:
        print(f"[Subject {s}, label={int(labels[s])}] running PCMCI...",
              flush=True)
        r = analyze_subject(
            s, data, comp_indices, n_nodes, gt_density,
            args.pcmci_method, args.pcmci_tau_max,
            args.pcmci_alpha, args.pcmci_fdr,
            args.scc_strategy, args.tol_low, args.tol_high)
        results.append(r)

    # --- Side-by-side table ---
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    fields = [
        ("group label",           lambda r, s: int(labels[s])),
        ("density",               lambda r, _: f"{r['density']:.3f}"),
        ("# directed edges",      lambda r, _: r['n_dir']),
        ("# bidirected edges",    lambda r, _: r['n_bidir']),
        ("",                      None),
        ("DD min/mean/max",       lambda r, _: f"{r['DD_min']}/"
                                              f"{r['DD_mean']:.1f}/"
                                              f"{r['DD_max']}"),
        ("DD std",                lambda r, _: f"{r['DD_std']:.2f}"),
        ("DD unique values",      lambda r, _: r['DD_unique']),
        ("DD low (≤4)",           lambda r, _: r['DD_low']),
        ("DD mid (5..15)",        lambda r, _: r['DD_mid']),
        ("DD high (≥16)",         lambda r, _: r['DD_high']),
        ("Sharp-edge ratio",      lambda r, _: f"{r['sharp_ratio']:.2%}"),
        ("",                      None),
        ("Edges in g_est",        lambda r, _: r['cur_edges']),
        ("Window edge range",     lambda r, _: f"[{r['d_lo_edges']}, "
                                              f"{r['d_hi_edges']}]"),
        ("In window?",            lambda r, _: "YES" if r['in_window']
                                              else "NO"),
        ("|edges - centre|",      lambda r, _: f"{r['edges_to_centre']:.1f}"),
        ("Feasible cardinalities",lambda r, _: r['feasible_cardinalities']),
        ("",                      None),
        ("SCC sizes",             lambda r, _: r['scc_sizes']),
        ("",                      None),
        ("Total DD mass",         lambda r, _: r['total_dd_mass']),
        ("Total BD mass",         lambda r, _: r['total_bd_mass']),
    ]

    col_w = max(20, max(len(name) for name, _ in fields))
    header = f"{'Metric':<{col_w}s} " + " ".join(
        f"{'subj '+str(r['subject']):>14s}" for r in results)
    print(header)
    print("-" * len(header))
    for name, fn in fields:
        if fn is None:
            print()
            continue
        row = f"{name:<{col_w}s} " + " ".join(
            f"{str(fn(r, r['subject'])):>14s}" for r in results)
        print(row)

    # --- DD weight histograms per subject ---
    print()
    print("=" * 80)
    print("DD WEIGHT HISTOGRAMS (off-diagonal)")
    print("=" * 80)
    for r in results:
        print(f"\nSubject {r['subject']}:  N={len(r['DD_off'])}  "
              f"mean={r['DD_mean']:.2f}  std={r['DD_std']:.2f}")
        print(histogram_str(r['DD_off'], bins=10, lo=0, hi=MAXCOST))

    # --- Interpretive narrative ---
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if len(results) >= 2:
        a, b = results[0], results[1]
        print(f"\nSubject {a['subject']} vs subject {b['subject']}:\n")

        # Sharpness
        if abs(a['sharp_ratio'] - b['sharp_ratio']) > 0.05:
            sharper = a if a['sharp_ratio'] > b['sharp_ratio'] else b
            duller = b if sharper is a else a
            print(f"  Sharpness: subject {sharper['subject']} has a SHARPER "
                  f"PCMCI prior ({sharper['sharp_ratio']:.1%} of edges have "
                  f"DD ≤4 or ≥16, vs {duller['sharp_ratio']:.1%}). "
                  f"This means the @1 cost objective has fewer ambiguous "
                  f"middle-ground edges -> typically faster optimum proof.")
        else:
            print(f"  Sharpness: similar ({a['sharp_ratio']:.1%} vs "
                  f"{b['sharp_ratio']:.1%}). Not a likely explanation.")

        # In-window
        if a['in_window'] != b['in_window']:
            ins, outs = (a, b) if a['in_window'] else (b, a)
            print(f"  Density window: subject {ins['subject']} is INSIDE the "
                  f"hard window (PCMCI prior is feasible by cardinality), "
                  f"subject {outs['subject']} is OUTSIDE. The outside subject "
                  f"forces the solver to add/remove edges against the prior.")
        else:
            d_a = a['edges_to_centre']
            d_b = b['edges_to_centre']
            if abs(d_a - d_b) > 1:
                closer = a if d_a < d_b else b
                farther = b if closer is a else a
                print(f"  Density window: both inside, but subject "
                      f"{closer['subject']} is closer to centre "
                      f"({closer['edges_to_centre']:.1f} vs "
                      f"{farther['edges_to_centre']:.1f} edges away). "
                      f"Less surgery needed.")

        # Plateau risk
        if abs(a['DD_unique'] - b['DD_unique']) > 2:
            flatter = a if a['DD_unique'] < b['DD_unique'] else b
            sharper2 = b if flatter is a else a
            print(f"  Cost-tie risk: subject {flatter['subject']} has only "
                  f"{flatter['DD_unique']} distinct DD weights vs "
                  f"{sharper2['DD_unique']}. Fewer unique weights -> more "
                  f"cost-tied solutions -> longer enumeration plateau.")

        # SCC
        if a['scc_sizes'] != b['scc_sizes']:
            print(f"  WARNING: SCC sizes differ "
                  f"({a['scc_sizes']} vs {b['scc_sizes']}) — unexpected "
                  f"under deterministic --scc_strategy. Check setup.")
        else:
            print(f"  SCC structure: identical ({a['scc_sizes']}). "
                  f"Not an explanation.")

    print()


if __name__ == "__main__":
    main()
