"""
Experiment 2: PCMCI Hyperparameter Grid Search on fMRI data.

Measures stability (within-group Jaccard) and discriminability (group diff)
across a grid of PCMCI configurations.

Run from: gunfolds/scripts/real_data/
    conda activate gunfolds
    python exp2_pcmci_hyperparam_grid.py

For a quick test with fewer subjects:
    python exp2_pcmci_hyperparam_grid.py --n_subjects 10

NOTE: Adjust the pp.DataFrame() call based on Experiment 0 results.
      If Exp 0 shows pp.DataFrame(ts_2d) is correct, change the .T below.
"""
import sys, os, json, itertools, time, argparse
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
# Config
# ---------------------------------------------------------------------------
CONFIGS = {
    "method":      ["run_pcmci", "run_pcmciplus"],
    "tau_max":     [1, 2, 3],
    "alpha_level": [0.01, 0.05, 0.1],
    "fdr_method":  ["none", "fdr_bh"],
}


def cg_to_adj(g_est):
    """Convert gunfolds CG to binary adjacency matrix."""
    n = len(g_est)
    adj = np.zeros((n, n), dtype=int)
    for i, nbrs in g_est.items():
        for j, v in nbrs.items():
            if v in (1, 3):
                adj[i - 1, j - 1] = 1
    return adj


def run_one_config(ts_2d, method, tau_max, alpha_level, fdr_method):
    """Run PCMCI with given config, return binary adjacency and graph."""
    # Exp 0 confirmed: pp.DataFrame expects (T, N), ts_2d is already [T, N]
    df = pp.DataFrame(ts_2d)
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)

    if method == "run_pcmci":
        res = pcmci.run_pcmci(
            tau_max=tau_max, pc_alpha=None,
            alpha_level=alpha_level, fdr_method=fdr_method,
        )
    else:
        res = pcmci.run_pcmciplus(
            tau_max=tau_max, pc_alpha=0.01,
            fdr_method=fdr_method,
        )

    g_est, A, B = cv.Glag2CG(res)
    adj = cg_to_adj(g_est)
    return adj, g_est, res


def jaccard(a, b):
    union = np.sum(a | b)
    if union == 0:
        return 1.0
    return float(np.sum(a & b)) / float(union)


def get_labels(npz):
    if "labels" in npz.files:
        return npz["labels"].flatten()
    if "label" in npz.files:
        return npz["label"].flatten()
    raise KeyError("Labels not found in NPZ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int, default=10, choices=[10, 20, 53])
    parser.add_argument("--n_subjects", type=int, default=None,
                        help="Limit subjects (None=all)")
    parser.add_argument("--data_path", type=str, default="../fbirn/fbirn_sz_data.npz")
    args = parser.parse_args()

    comp_indices = get_comp_indices(args.n_components)
    npzfile = np.load(args.data_path)
    data = npzfile["data"]           # [n_subjects, T, F]
    labels = get_labels(npzfile)     # [n_subjects]
    print(f"Data shape: {data.shape}, Labels: {labels.shape}")
    print(f"Using {args.n_components} components: indices {comp_indices}")

    n_subj = args.n_subjects or data.shape[0]
    print(f"Running on {n_subj} subjects\n")

    configs = list(itertools.product(
        CONFIGS["method"], CONFIGS["tau_max"],
        CONFIGS["alpha_level"], CONFIGS["fdr_method"],
    ))
    print(f"Total configurations: {len(configs)}")

    results_all = []

    for ci, (method, tau_max, alpha_level, fdr_method) in enumerate(configs):
        tag = f"{method}_tau{tau_max}_a{alpha_level}_fdr{fdr_method}"
        print(f"\n[{ci+1}/{len(configs)}] {tag}")
        t0 = time.time()

        adjs = []
        groups = []
        errors = 0
        for s in range(n_subj):
            ts_2d = data[s][:, comp_indices]  # [T, N]
            try:
                adj, _, _ = run_one_config(
                    ts_2d, method, tau_max, alpha_level, fdr_method)
                adjs.append(adj)
                groups.append(int(labels[s]))
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  Subject {s} failed: {e}")

        if len(adjs) == 0:
            print("  ALL subjects failed, skipping.")
            continue

        adjs = np.array(adjs)
        groups = np.array(groups)
        elapsed = time.time() - t0

        # Edge density
        N = adjs.shape[1]
        density = float(np.mean([a.sum() / (N * N) for a in adjs]))

        # Within-group Jaccard (sample up to 1000 pairs for speed)
        jaccards = {"all": [], "HC": [], "SZ": []}
        pairs = []
        for i in range(len(adjs)):
            for j in range(i + 1, len(adjs)):
                pairs.append((i, j))
        if len(pairs) > 2000:
            rng = np.random.RandomState(0)
            pairs = [pairs[k] for k in rng.choice(len(pairs), 2000, replace=False)]

        for i, j in pairs:
            jac = jaccard(adjs[i].astype(bool), adjs[j].astype(bool))
            jaccards["all"].append(jac)
            if groups[i] == groups[j]:
                key = "HC" if groups[i] == 0 else "SZ"
                jaccards[key].append(jac)

        # Group discriminability
        hc_mask = groups == 0
        sz_mask = groups == 1
        if hc_mask.any() and sz_mask.any():
            hc_consensus = (adjs[hc_mask].mean(axis=0) > 0.5).astype(int)
            sz_consensus = (adjs[sz_mask].mean(axis=0) > 0.5).astype(int)
            group_diff = int(np.sum(hc_consensus != sz_consensus))
        else:
            group_diff = -1

        rec = {
            "config": tag,
            "method": method,
            "tau_max": tau_max,
            "alpha_level": alpha_level,
            "fdr_method": fdr_method,
            "density": density,
            "jaccard_all": float(np.mean(jaccards["all"])) if jaccards["all"] else -1,
            "jaccard_HC": float(np.mean(jaccards["HC"])) if jaccards["HC"] else -1,
            "jaccard_SZ": float(np.mean(jaccards["SZ"])) if jaccards["SZ"] else -1,
            "group_diff_edges": group_diff,
            "elapsed_sec": round(elapsed, 1),
            "n_subjects": len(adjs),
            "n_errors": errors,
        }
        results_all.append(rec)
        print(f"  density={density:.3f}  jaccard_all={rec['jaccard_all']:.3f}  "
              f"jaccard_HC={rec['jaccard_HC']:.3f}  jaccard_SZ={rec['jaccard_SZ']:.3f}  "
              f"group_diff={group_diff}  time={elapsed:.1f}s")

    # Save
    ts = datetime.now().strftime("%m%d%Y%H%M%S")
    out = f"pcmci_hyperparam_results_{ts}.json"
    with open(out, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Results saved to {out}")
    print(f"{'='*70}")

    # Summary table
    print(f"\n{'Config':<50} {'Dens':>6} {'J_all':>6} {'J_HC':>6} {'J_SZ':>6} {'GDiff':>6}")
    print("-" * 86)
    for r in sorted(results_all, key=lambda x: -x['jaccard_all']):
        print(f"{r['config']:<50} {r['density']:>6.3f} {r['jaccard_all']:>6.3f} "
              f"{r['jaccard_HC']:>6.3f} {r['jaccard_SZ']:>6.3f} {r['group_diff_edges']:>6d}")


if __name__ == "__main__":
    main()
