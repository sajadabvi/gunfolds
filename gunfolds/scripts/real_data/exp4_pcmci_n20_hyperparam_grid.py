"""
Experiment 4: PCMCI Hyperparameter Grid Search for N=20 Components.

Extends Experiment 2 (N=10 grid search) to 20 ICA components and adds:
  1. pc_alpha as a proper hyperparameter for PCMCIplus (was fixed at 0.01).
  2. tau_max extended to 4 (from 3).
  3. Composite score balancing Jaccard stability and proximity to the 22%
     ground-truth density (literature moderate tier for N=20).

Ground-truth density reference (ground_truth_connectivity_estimates.md):
  N=20 moderate tier: 20-25%  ->  TARGET_DENSITY = 0.22

---------------------------------------------------------------------------
THREE OPERATING MODES
---------------------------------------------------------------------------

1) Sequential (all 48 configs, one machine):
       python exp4_pcmci_n20_hyperparam_grid.py

2) Single-config mode (SLURM array task):
       python exp4_pcmci_n20_hyperparam_grid.py \\
           --task_id $SLURM_ARRAY_TASK_ID \\
           --run_dir results_exp4_<TIMESTAMP>

   Writes: results_exp4_<TIMESTAMP>/cfg_<task_id:03d>.json

3) Merge / summary (run after all array tasks finish):
       python exp4_pcmci_n20_hyperparam_grid.py \\
           --merge --run_dir results_exp4_<TIMESTAMP>

   Reads all cfg_*.json files, prints tables, writes final JSON.
---------------------------------------------------------------------------

Run from: gunfolds/scripts/real_data/
    conda activate gunfolds
    python exp4_pcmci_n20_hyperparam_grid.py [--n_subjects 10]
"""

import sys
import os
import json
import itertools
import time
import argparse
import glob as _glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import numpy as np
from datetime import datetime

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from gunfolds import conversions as cv
from component_config import get_comp_indices

# ---------------------------------------------------------------------------
# Target density (from ground_truth_connectivity_estimates.md, N=20 moderate)
# ---------------------------------------------------------------------------
TARGET_DENSITY = 0.22

# ---------------------------------------------------------------------------
# Hyperparameter grid — order must be deterministic (task_id maps to config)
# ---------------------------------------------------------------------------
PCMCI_GRID = {
    "tau_max":     [1, 2, 3, 4],
    "alpha_level": [0.005, 0.01, 0.05, 0.1],
    "fdr_method":  ["none", "fdr_bh"],
}

PCMCIPLUS_GRID = {
    "tau_max":  [1, 2, 3, 4],
    "pc_alpha": [0.001, 0.005, 0.01, 0.05],
}

# Composite score weights
W_JACCARD         = 0.6
W_DENSITY         = 0.4
DENSITY_BANDWIDTH = 0.10   # |density - target| >= bw => density score = 0


# ---------------------------------------------------------------------------
# Build the full, deterministic config list (index = task_id)
# ---------------------------------------------------------------------------

def build_config_list():
    pcmci_configs = [
        {
            "method":      "run_pcmci",
            "tau_max":     tau_max,
            "alpha_level": alpha,
            "fdr_method":  fdr,
            "pc_alpha":    None,
            "tag": f"pcmci_tau{tau_max}_a{alpha}_fdr{fdr}",
        }
        for tau_max, alpha, fdr in itertools.product(
            PCMCI_GRID["tau_max"],
            PCMCI_GRID["alpha_level"],
            PCMCI_GRID["fdr_method"],
        )
    ]
    pcmciplus_configs = [
        {
            "method":      "run_pcmciplus",
            "tau_max":     tau_max,
            "alpha_level": None,
            "fdr_method":  "none",
            "pc_alpha":    pc_alpha,
            "tag": f"pcmciplus_tau{tau_max}_pca{pc_alpha}",
        }
        for tau_max, pc_alpha in itertools.product(
            PCMCIPLUS_GRID["tau_max"],
            PCMCIPLUS_GRID["pc_alpha"],
        )
    ]
    return pcmci_configs + pcmciplus_configs


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def cg_to_adj(g_est):
    """Gunfolds CG dict → binary adjacency (0-based rows/cols)."""
    n = len(g_est)
    adj = np.zeros((n, n), dtype=int)
    for i, nbrs in g_est.items():
        for j, v in nbrs.items():
            if v in (1, 3):
                adj[i - 1, j - 1] = 1
    return adj


def run_one_config(ts_2d, method, tau_max, alpha_level, fdr_method, pc_alpha):
    """Run PCMCI / PCMCIplus, return binary adjacency matrix."""
    df = pp.DataFrame(ts_2d)   # (T, N) confirmed in Exp 0
    pcmci_obj = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)

    if method == "run_pcmci":
        res = pcmci_obj.run_pcmci(
            tau_max=tau_max,
            pc_alpha=None,
            alpha_level=alpha_level,
            fdr_method=fdr_method,
        )
    else:
        res = pcmci_obj.run_pcmciplus(
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            fdr_method=fdr_method,
        )

    g_est, _A, _B = cv.Glag2CG(res)
    return cg_to_adj(g_est)


def jaccard(a, b):
    union = np.sum(a | b)
    return 1.0 if union == 0 else float(np.sum(a & b)) / float(union)


def pairwise_jaccards(adjs, groups, max_pairs=2000, rng_seed=0):
    pairs = [(i, j) for i in range(len(adjs)) for j in range(i + 1, len(adjs))]
    if len(pairs) > max_pairs:
        rng = np.random.RandomState(rng_seed)
        pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]

    jac = {"all": [], "HC": [], "SZ": []}
    for i, j in pairs:
        v = jaccard(adjs[i].astype(bool), adjs[j].astype(bool))
        jac["all"].append(v)
        if groups[i] == groups[j]:
            key = "HC" if groups[i] == 0 else "SZ"
            jac[key].append(v)

    return {k: float(np.mean(v)) if v else -1.0 for k, v in jac.items()}


def density_score(density, target=TARGET_DENSITY, bw=DENSITY_BANDWIDTH):
    return max(0.0, 1.0 - abs(density - target) / bw)


def composite_score(jaccard_all, dens):
    return W_JACCARD * jaccard_all + W_DENSITY * density_score(dens)


def get_labels(npz):
    for key in ("labels", "label"):
        if key in npz.files:
            return npz[key].flatten()
    raise KeyError("Labels not found in NPZ (tried 'labels', 'label')")


# ---------------------------------------------------------------------------
# Run a single config over all subjects
# ---------------------------------------------------------------------------

def run_config(cfg, data, labels, comp_indices, n_subj, max_edges, target):
    tag      = cfg["tag"]
    method   = cfg["method"]
    tau_max  = cfg["tau_max"]
    alpha    = cfg["alpha_level"]
    fdr      = cfg["fdr_method"]
    pc_alpha = cfg["pc_alpha"]

    t0     = time.time()
    adjs   = []
    groups = []
    errors = 0

    for s in range(n_subj):
        ts_2d = data[s][:, comp_indices]   # (T, N)
        try:
            adj = run_one_config(ts_2d, method, tau_max, alpha, fdr, pc_alpha)
            adjs.append(adj)
            groups.append(int(labels[s]))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    Subject {s} failed: {e}")

    if not adjs:
        print(f"    ALL subjects failed — skipping {tag}.")
        return None

    adjs   = np.array(adjs)
    groups = np.array(groups)
    elapsed = time.time() - t0

    per_subj_dens = adjs.sum(axis=(1, 2)) / max_edges
    mean_density  = float(per_subj_dens.mean())
    std_density   = float(per_subj_dens.std())
    density_dev   = float(abs(mean_density - target))

    jac        = pairwise_jaccards(adjs, groups)
    comp       = composite_score(jac["all"], mean_density)

    hc_mask = groups == 0
    sz_mask = groups == 1
    if hc_mask.any() and sz_mask.any():
        hc_cons    = (adjs[hc_mask].mean(axis=0) > 0.5).astype(int)
        sz_cons    = (adjs[sz_mask].mean(axis=0) > 0.5).astype(int)
        group_diff = int(np.sum(hc_cons != sz_cons))
    else:
        group_diff = -1

    return {
        "config":           tag,
        "method":           method,
        "tau_max":          tau_max,
        "alpha_level":      alpha,
        "fdr_method":       fdr,
        "pc_alpha":         pc_alpha,
        "density":          mean_density,
        "density_std":      std_density,
        "density_dev":      density_dev,
        "jaccard_all":      jac["all"],
        "jaccard_HC":       jac["HC"],
        "jaccard_SZ":       jac["SZ"],
        "group_diff_edges": group_diff,
        "composite_score":  comp,
        "elapsed_sec":      round(elapsed, 1),
        "n_subjects":       len(adjs),
        "n_errors":         errors,
    }


# ---------------------------------------------------------------------------
# Summary / print helpers
# ---------------------------------------------------------------------------

def print_summary(results_all, target):
    ranked = sorted(results_all, key=lambda x: -x["composite_score"])
    close  = [r for r in results_all if r["density_dev"] <= 0.05]
    close_ranked = sorted(close, key=lambda x: -x["jaccard_all"])

    hdr = (f"{'Config':<46} {'Dens':>6} {'Dev':>6} "
           f"{'J_all':>6} {'J_HC':>6} {'J_SZ':>6} "
           f"{'GDiff':>6} {'Score':>6}")
    sep = "-" * len(hdr)

    def row(r):
        return (
            f"{r['config']:<46} "
            f"{r['density']:>6.3f} "
            f"{r['density_dev']:>6.3f} "
            f"{r['jaccard_all']:>6.3f} "
            f"{r['jaccard_HC']:>6.3f} "
            f"{r['jaccard_SZ']:>6.3f} "
            f"{r['group_diff_edges']:>6d} "
            f"{r['composite_score']:>6.3f}"
        )

    print("\nTOP 20 configurations (sorted by composite score):")
    print(hdr); print(sep)
    for r in ranked[:20]:
        print(row(r))

    print(f"\nConfigs within ±5 pp of {target:.0%} target ({len(close)} total), "
          f"sorted by Jaccard:")
    print(hdr); print(sep)
    for r in close_ranked[:15]:
        print(row(r))

    if ranked:
        best = ranked[0]
        print(f"\n{'='*80}")
        print("RECOMMENDATION")
        print(f"  Best config : {best['config']}")
        print(f"  Density     : {best['density']:.3f}  (target {target:.3f}, "
              f"dev {best['density_dev']:.3f})")
        print(f"  Jaccard     : all={best['jaccard_all']:.3f}  "
              f"HC={best['jaccard_HC']:.3f}  SZ={best['jaccard_SZ']:.3f}")
        print(f"  Group diff  : {best['group_diff_edges']} edges")
        print(f"  Composite   : {best['composite_score']:.3f}")

        print(f"\n  --> fmri_experiment_large.py flags:")
        print(f"       --n_components 20 \\")
        if best["method"] == "run_pcmciplus":
            print(f"       --pcmci_method run_pcmciplus \\")
            print(f"       --pcmci_tau_max {best['tau_max']} \\")
            print(f"       --pcmci_pc_alpha {best['pc_alpha']} \\")
        else:
            print(f"       --pcmci_method run_pcmci \\")
            print(f"       --pcmci_tau_max {best['tau_max']} \\")
            print(f"       --pcmci_alpha {best['alpha_level']} \\")
            print(f"       --pcmci_fdr {best['fdr_method']} \\")
        print(f"       --gt_density_mode fixed --gt_density 220")
        print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp 4: PCMCI hyperparameter grid, N=20 components.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n_subjects", type=int, default=None,
                        help="Limit number of subjects (default: all)")
    parser.add_argument("--data_path", type=str,
                        default="../fbirn/fbirn_sz_data.npz")
    parser.add_argument("--target_density", type=float, default=TARGET_DENSITY,
                        help=f"Target graph density (default: {TARGET_DENSITY})")

    # --- SLURM array mode ---
    parser.add_argument("--task_id", type=int, default=None,
                        help="SLURM array task index (0-based). "
                             "Run only that single config and write a partial JSON.")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Directory for partial / merged results. "
                             "Required with --task_id and --merge.")

    # --- Merge mode ---
    parser.add_argument("--merge", action="store_true",
                        help="Aggregate all partial cfg_*.json files in --run_dir "
                             "and print the summary table.")

    args = parser.parse_args()
    target = args.target_density

    all_configs = build_config_list()
    n_total     = len(all_configs)

    # =========================================================================
    # MODE 1: merge / summary
    # =========================================================================
    if args.merge:
        if not args.run_dir:
            parser.error("--run_dir is required with --merge")

        pattern = os.path.join(args.run_dir, "cfg_*.json")
        partial_files = sorted(_glob.glob(pattern))
        if not partial_files:
            print(f"No partial result files found matching: {pattern}")
            sys.exit(1)

        results_all = []
        missing = []
        for i in range(n_total):
            path = os.path.join(args.run_dir, f"cfg_{i:03d}.json")
            if os.path.exists(path):
                with open(path) as f:
                    results_all.append(json.load(f))
            else:
                missing.append(i)

        print(f"Loaded {len(results_all)} / {n_total} configs from {args.run_dir}")
        if missing:
            print(f"  Missing task IDs: {missing}")

        ts  = datetime.now().strftime("%m%d%Y%H%M%S")
        out = os.path.join(args.run_dir, f"exp4_n20_results_{ts}.json")
        with open(out, "w") as f:
            json.dump(results_all, f, indent=2)
        print(f"  Final JSON saved -> {out}\n")

        print_summary(results_all, target)
        return

    # =========================================================================
    # Load data (needed for modes 2 and 3)
    # =========================================================================
    npzfile     = np.load(args.data_path)
    data        = npzfile["data"]       # [n_subjects, T, 53]
    labels      = get_labels(npzfile)
    comp_indices = get_comp_indices(20)
    n_subj      = args.n_subjects or data.shape[0]
    N_nodes     = len(comp_indices)     # 20
    max_edges   = N_nodes * (N_nodes - 1)

    print(f"Data shape   : {data.shape}")
    print(f"N=20 indices : {comp_indices}")
    print(f"Subjects     : {n_subj} / {data.shape[0]}")
    print(f"Target density: {target:.1%}  (literature: 20-25% for N=20)")
    print(f"Composite score = {W_JACCARD}*Jaccard + {W_DENSITY}*DensityProximity\n")

    # =========================================================================
    # MODE 2: single config (SLURM array task)
    # =========================================================================
    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= n_total:
            print(f"task_id {args.task_id} out of range [0, {n_total-1}]")
            sys.exit(1)
        if not args.run_dir:
            parser.error("--run_dir is required with --task_id")

        os.makedirs(args.run_dir, exist_ok=True)

        cfg = all_configs[args.task_id]
        print(f"[task {args.task_id}/{n_total-1}] {cfg['tag']}")

        rec = run_config(cfg, data, labels, comp_indices, n_subj, max_edges, target)
        if rec is None:
            sys.exit(1)

        print(f"  dens={rec['density']:.3f}+/-{rec['density_std']:.3f}  "
              f"dev={rec['density_dev']:.3f}  "
              f"J_all={rec['jaccard_all']:.3f}  "
              f"J_HC={rec['jaccard_HC']:.3f}  "
              f"J_SZ={rec['jaccard_SZ']:.3f}  "
              f"GDiff={rec['group_diff_edges']}  "
              f"score={rec['composite_score']:.3f}  "
              f"t={rec['elapsed_sec']:.1f}s")

        out = os.path.join(args.run_dir, f"cfg_{args.task_id:03d}.json")
        with open(out, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"  Saved -> {out}")
        return

    # =========================================================================
    # MODE 3: sequential (all configs, single machine)
    # =========================================================================
    print(f"Total configurations: {n_total}  "
          f"(32 PCMCI + 16 PCMCIplus)\n")

    results_all = []
    for ci, cfg in enumerate(all_configs):
        print(f"[{ci+1:3d}/{n_total}] {cfg['tag']}")
        rec = run_config(cfg, data, labels, comp_indices, n_subj, max_edges, target)
        if rec is None:
            continue
        results_all.append(rec)
        print(f"    dens={rec['density']:.3f}+/-{rec['density_std']:.3f}  "
              f"dev={rec['density_dev']:.3f}  "
              f"J_all={rec['jaccard_all']:.3f}  "
              f"J_HC={rec['jaccard_HC']:.3f}  "
              f"J_SZ={rec['jaccard_SZ']:.3f}  "
              f"GDiff={rec['group_diff_edges']}  "
              f"score={rec['composite_score']:.3f}  "
              f"t={rec['elapsed_sec']:.1f}s")

    ts  = datetime.now().strftime("%m%d%Y%H%M%S")
    out = f"exp4_n20_results_{ts}.json"
    with open(out, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\n{'='*80}")
    print(f"Results saved -> {out}")
    print(f"{'='*80}\n")

    print_summary(results_all, target)


if __name__ == "__main__":
    main()
