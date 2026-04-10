"""
Tier 1: Classical ML classification of HC vs SZ from causal-graph features.

Loads per-subject result.zkl files produced by fmri_experiment_large.py,
extracts vectorised graph features (edge frequencies, topology metrics,
RASL-specific features), and trains SVM / Random Forest / Logistic Regression
classifiers with stratified nested cross-validation.

Usage:
    python classify_hc_sz.py --timestamp 03232026175230
    python classify_hc_sz.py --timestamp 03232026175230 --configs N10_domain_RASL N10_none_PCMCI
"""

import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_validate,
    permutation_test_score,
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, make_scorer,
    classification_report,
)

from gunfolds.utils import zickle as zkl
from gunfolds import conversions as cv

_REAL_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "real_data"
)
if _REAL_DATA_DIR not in sys.path:
    sys.path.insert(0, _REAL_DATA_DIR)
from component_config import get_comp_indices

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Tier-1 classical ML HC vs SZ.")
    p.add_argument("--timestamp", required=True)
    p.add_argument("--results_root", default="fbirn_results")
    p.add_argument("--configs", nargs="*", default=None,
                   help="Specific config tags to evaluate (default: all)")
    p.add_argument("--outer_folds", type=int, default=5)
    p.add_argument("--inner_folds", type=int, default=5)
    p.add_argument("--n_permutations", type=int, default=100,
                   help="Permutation test iterations (0 to skip)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=None,
                   help="Override output directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_configs(root_dir):
    configs = []
    if not os.path.isdir(root_dir):
        return configs
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path):
            continue
        parts = name.split("_")
        if len(parts) >= 3 and parts[0].startswith("N"):
            configs.append(name)
    return configs


def load_config_subjects(config_dir):
    pattern = os.path.join(config_dir, "subject_*", "result.zkl")
    results = []
    for f in sorted(glob.glob(pattern)):
        try:
            results.append(zkl.load(f))
        except Exception as e:
            print(f"  Warning: {f}: {e}")
    return results


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _adj_from_sol(sol):
    if "adj" in sol:
        return np.array(sol["adj"], dtype=float)
    return (cv.graph2adj(sol["graph"]) > 0).astype(float)


def extract_features(subject_results):
    """
    Build a feature matrix X [n_subjects, n_features] and label vector y.

    Features per subject:
        - Mean adjacency vector (across solutions), flattened off-diagonal
        - Edge frequency std across solutions (RASL: captures agreement)
        - Graph density (mean)
        - In-degree / out-degree stats (mean, std of each)
        - RASL-specific: mean cost, std cost, mean undersampling rate,
          solution-set entropy (per edge)
    """
    X_rows = []
    y = []
    feature_names = None

    for info in subject_results:
        n_sol = len(info["solutions"])
        adjs = np.stack([_adj_from_sol(s) for s in info["solutions"]])
        N = adjs.shape[1]

        mean_adj = adjs.mean(axis=0)
        np.fill_diagonal(mean_adj, 0)

        off_diag_mask = ~np.eye(N, dtype=bool)
        edge_vec = mean_adj[off_diag_mask]

        if n_sol > 1:
            std_adj = adjs.std(axis=0)
            np.fill_diagonal(std_adj, 0)
            edge_std_vec = std_adj[off_diag_mask]
        else:
            edge_std_vec = np.zeros_like(edge_vec)

        density = mean_adj.sum() / (N * (N - 1)) if N > 1 else 0.0
        in_deg = mean_adj.sum(axis=0)
        out_deg = mean_adj.sum(axis=1)

        topo_feats = [
            density,
            in_deg.mean(), in_deg.std(),
            out_deg.mean(), out_deg.std(),
        ]

        rasl_feats = [0.0, 0.0, 0.0, 0.0]
        if info.get("method") == "RASL" and n_sol > 0:
            costs = [s["cost"] for s in info["solutions"]]
            rasl_feats[0] = np.mean(costs)
            rasl_feats[1] = np.std(costs) if len(costs) > 1 else 0.0
            u_rates = []
            for s in info["solutions"]:
                u = s.get("undersampling")
                if u is not None:
                    u_rates.append(u[0] if isinstance(u, (list, tuple)) else int(u))
            rasl_feats[2] = np.mean(u_rates) if u_rates else 0.0
            # Per-edge entropy across solutions
            eps = 1e-10
            p = np.clip(mean_adj[off_diag_mask], eps, 1 - eps)
            entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
            rasl_feats[3] = entropy.mean()

        feat = np.concatenate([edge_vec, edge_std_vec, topo_feats, rasl_feats])
        X_rows.append(feat)
        y.append(info["group"])

        if feature_names is None:
            n_edges = edge_vec.shape[0]
            names = [f"edge_freq_{i}" for i in range(n_edges)]
            names += [f"edge_std_{i}" for i in range(n_edges)]
            names += ["density", "in_deg_mean", "in_deg_std",
                       "out_deg_mean", "out_deg_std"]
            names += ["rasl_mean_cost", "rasl_std_cost",
                       "rasl_mean_usamp", "rasl_edge_entropy"]
            feature_names = names

    return np.array(X_rows), np.array(y), feature_names


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def get_classifiers():
    return {
        "SVM_linear": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear", probability=True, max_iter=5000)),
            ]),
            "param_grid": {"clf__C": [0.01, 0.1, 1, 10]},
        },
        "SVM_rbf": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, max_iter=5000)),
            ]),
            "param_grid": {
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale", 0.01, 0.001],
            },
        },
        "RandomForest": {
            "pipeline": Pipeline([
                ("clf", RandomForestClassifier(random_state=42)),
            ]),
            "param_grid": {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [5, 10, None],
                "clf__min_samples_leaf": [2, 5],
            },
        },
        "LogReg_L1": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l1", solver="saga", max_iter=5000)),
            ]),
            "param_grid": {"clf__C": [0.01, 0.1, 1, 10]},
        },
        "LogReg_L2": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l2", solver="lbfgs", max_iter=5000)),
            ]),
            "param_grid": {"clf__C": [0.01, 0.1, 1, 10]},
        },
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_config(X, y, args):
    """Run nested CV for all classifiers. Return a list of result dicts."""
    outer_cv = StratifiedKFold(
        n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    inner_cv = StratifiedKFold(
        n_splits=args.inner_folds, shuffle=True, random_state=args.seed)

    classifiers = get_classifiers()
    results = []

    for clf_name, spec in classifiers.items():
        print(f"    {clf_name} ... ", end="", flush=True)

        grid = GridSearchCV(
            spec["pipeline"], spec["param_grid"],
            cv=inner_cv, scoring="accuracy", n_jobs=-1, refit=True,
        )

        scoring = {
            "accuracy": "accuracy",
            "roc_auc": "roc_auc",
        }
        cv_out = cross_validate(
            grid, X, y, cv=outer_cv, scoring=scoring,
            return_estimator=True, n_jobs=1,
        )

        acc = cv_out["test_accuracy"]
        auc = cv_out["test_roc_auc"]

        row = {
            "classifier": clf_name,
            "accuracy_mean": acc.mean(),
            "accuracy_std": acc.std(),
            "auc_mean": auc.mean(),
            "auc_std": auc.std(),
        }

        if args.n_permutations > 0:
            _, _, perm_pval = permutation_test_score(
                grid, X, y, cv=outer_cv, scoring="accuracy",
                n_permutations=args.n_permutations, random_state=args.seed,
                n_jobs=-1,
            )
            row["perm_p_value"] = perm_pval
        else:
            row["perm_p_value"] = None

        best_params = []
        for est in cv_out["estimator"]:
            best_params.append(est.best_params_)
        row["best_params"] = best_params

        print(f"acc={acc.mean():.3f}±{acc.std():.3f}  "
              f"auc={auc.mean():.3f}±{auc.std():.3f}")
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    root_dir = os.path.join(args.results_root, args.timestamp)
    if not os.path.isdir(root_dir):
        # Try from analysis dir (script might be run from gunfolds/scripts/analysis/)
        alt = os.path.join("..", "real_data", args.results_root, args.timestamp)
        if os.path.isdir(alt):
            root_dir = alt
        else:
            print(f"Error: {root_dir} not found")
            sys.exit(1)

    configs = args.configs or discover_configs(root_dir)
    if not configs:
        print("No configs found.")
        sys.exit(1)

    out_dir = args.output_dir or os.path.join(root_dir, "ml_classification")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("TIER 1: CLASSICAL ML — HC vs SZ CLASSIFICATION")
    print("=" * 80)
    print(f"Timestamp:   {args.timestamp}")
    print(f"Configs:     {configs}")
    print(f"Outer folds: {args.outer_folds}  Inner folds: {args.inner_folds}")
    print(f"Output:      {out_dir}")
    print()

    all_rows = []

    for cfg in configs:
        cfg_dir = os.path.join(root_dir, cfg)
        if not os.path.isdir(cfg_dir):
            print(f"  {cfg}: directory not found, skipping")
            continue

        print(f"  Loading {cfg} ...")
        subjects = load_config_subjects(cfg_dir)
        if len(subjects) < 10:
            print(f"    Only {len(subjects)} subjects, skipping")
            continue

        X, y, feat_names = extract_features(subjects)
        n0 = (y == 0).sum()
        n1 = (y == 1).sum()
        print(f"    {len(subjects)} subjects, {X.shape[1]} features, "
              f"group0={n0} group1={n1}")

        clf_results = evaluate_config(X, y, args)
        for row in clf_results:
            row["config"] = cfg
            parts = cfg.split("_")
            row["method"] = parts[2] if len(parts) >= 3 else cfg
            row["n_components"] = int(parts[0][1:]) if parts[0].startswith("N") else 0
            row["scc_strategy"] = parts[1] if len(parts) >= 2 else ""
            row["n_subjects"] = len(subjects)
            row["n_features"] = X.shape[1]
            all_rows.append(row)

    if not all_rows:
        print("No results.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    display_cols = [
        "config", "classifier", "accuracy_mean", "accuracy_std",
        "auc_mean", "auc_std", "perm_p_value",
    ]
    show = df[[c for c in display_cols if c in df.columns]].copy()
    for c in ["accuracy_mean", "accuracy_std", "auc_mean", "auc_std"]:
        if c in show.columns:
            show[c] = show[c].round(4)
    if "perm_p_value" in show.columns:
        show["perm_p_value"] = show["perm_p_value"].apply(
            lambda v: f"{v:.4f}" if v is not None else "")

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(show.to_string(index=False))

    csv_path = os.path.join(out_dir, "tier1_results.csv")
    df.drop(columns=["best_params"], errors="ignore").to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Best per config
    print("\n" + "=" * 80)
    print("BEST CLASSIFIER PER CONFIG")
    print("=" * 80)
    best = df.loc[df.groupby("config")["accuracy_mean"].idxmax()]
    for _, r in best.iterrows():
        print(f"  {r['config']:30s}  {r['classifier']:15s}  "
              f"acc={r['accuracy_mean']:.3f}  auc={r['auc_mean']:.3f}")

    # Cross-method comparison
    print("\n" + "=" * 80)
    print("RASL vs BASELINES (best classifier per method)")
    print("=" * 80)
    if "method" in df.columns:
        for n_comp in sorted(df["n_components"].unique()):
            sub = df[df["n_components"] == n_comp]
            best_per_method = sub.loc[sub.groupby("method")["accuracy_mean"].idxmax()]
            print(f"\n  N = {n_comp}:")
            for _, r in best_per_method.iterrows():
                print(f"    {r['method']:6s} ({r['scc_strategy']:12s})  "
                      f"{r['classifier']:15s}  "
                      f"acc={r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f}  "
                      f"auc={r['auc_mean']:.3f}±{r['auc_std']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
