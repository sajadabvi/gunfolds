"""
refactored_analyze_fmri_experiment.py
======================================

Refactored cross-configuration analysis implementing the full
"cost-weighted posterior -> subject-as-unit -> block-level -> omnibus ->
underdetermination biomarkers (+ bootstrap stability)" pipeline, with a
head-to-head contrast against the legacy pooled edge-frequency test.

It mirrors the CLI of the legacy `analyze_fmri_experiment.py` (--timestamp,
--results_root, --plot, --alpha, --correction) and discovers every config dir
(`N<k>_<scc>_<method>`) under the timestamp.  For each RASL config it reports:

  LEGACY (reproduced) : pooled edge frequency, unit = SOLUTION  (pseudo-replicated)
  NEW  (1) per-subject cost-weighted edge posterior (Boltzmann, --tau)
       (2) subject-as-unit edge tests (Mann-Whitney; uniform + weighted + MAP)
       (3) network-block (NeuroMark domain) tests
       (4) omnibus two-sample tests over graph space (CV-classifier + RBF-MMD)
       (5) underdetermination biomarkers (entropy / ESS / penumbra / MAP-u)
       (6) bootstrap-stability edge test, when the run saved `bootstrap_stability`
  HEAD-TO-HEAD: which legacy "significant" edges survive subject-as-unit testing.

Works on BOTH refactored runs (fbirn_results_refactored, full cost band saved)
AND legacy top-k runs (fbirn_results) -- the posterior is recomputed from
whatever solutions were saved, so the A/B is apples-to-apples.

The statistical engine is imported from posterior_group_analysis.py (same dir).

Usage:
  python refactored_analyze_fmri_experiment.py --timestamp 06042026120000 \
      --results_root fbirn_results_refactored --tau 1.0 --correction fdr --plot
"""

import os
import sys
import csv
import json
import glob
import argparse

import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))   # repo root
sys.path.insert(0, _HERE)                                    # for the engine import

from gunfolds.utils import zickle as zkl
from gunfolds.scripts.real_data.component_config import INDEX_TO_DOMAIN, DOMAIN_ORDER

# Statistical engine (shared with the standalone posterior_group_analysis.py)
from posterior_group_analysis import (
    subject_posteriors, legacy_edge_tests, subject_unit_edge_tests,
    block_level_tests, classifier_two_sample, mmd_two_sample,
    _correct, edge_list, _edge_idx,
)

FEAT_KEYS = ["n_solutions", "ess", "cost_spread", "mean_edge_entropy",
             "penumbra", "stable_core", "map_u", "mean_u"]


# ---------------------------------------------------------------------------
# Loading / discovery
# ---------------------------------------------------------------------------

def discover_configs(root):
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and name.startswith("N") and len(name.split("_")) >= 3:
            out.append(name)
    return out


def load_config(config_dir):
    files = sorted(glob.glob(os.path.join(config_dir, "subject_*", "result.zkl")))
    subs = []
    for f in files:
        try:
            subs.append(zkl.load(f))
        except Exception as e:
            print(f"  WARN load {f}: {e}")
    return subs


# ---------------------------------------------------------------------------
# Per-config analysis
# ---------------------------------------------------------------------------

def analyze_config(subjects, tau, alpha, correction, n_perm):
    """Run the full new pipeline + legacy A/B on one config's subjects."""
    names = subjects[0]["comp_names"]
    comp_idx = subjects[0].get("comp_indices", list(range(len(names))))
    domains = [INDEX_TO_DOMAIN.get(ci, "??") for ci in comp_idx]
    groups = sorted({s["group"] for s in subjects})
    if len(groups) < 2:
        return None
    g0, g1 = groups[0], groups[1]
    gcount = {g: sum(1 for s in subjects if s["group"] == g) for g in groups}

    # per-subject representations + biomarkers
    reps = {g: {"uniform": [], "weighted": [], "map": []} for g in groups}
    stabs = {g: [] for g in groups}
    has_boot = False
    feat_rows = []
    for info in subjects:
        r = subject_posteriors(info, tau=tau)
        g = info["group"]
        reps[g]["uniform"].append(r["P_uniform"])
        reps[g]["weighted"].append(r["P_weighted"])
        reps[g]["map"].append(r["P_map"])
        feat_rows.append({"subject": info.get("subject_id"), "group": g, **r["feats"]})
        bs = info.get("bootstrap_stability")
        if bs is not None:
            has_boot = True
            stabs[g].append(np.array(bs, dtype=float))
        else:
            stabs[g].append(None)
    P_by = {rep: {g: np.stack(reps[g][rep]) for g in groups}
            for rep in ["uniform", "weighted", "map"]}

    out = {"groups": [g0, g1], "group_counts": gcount, "names": names,
           "domains": domains, "tau": tau, "alpha": alpha, "correction": correction}

    # LEGACY (pooled, unit=solution)
    lp, lsig, n0, n1 = legacy_edge_tests(subjects, alpha, correction)
    out["legacy"] = {"unit": "solution", "n0": n0, "n1": n1,
                     "n_sig": int(lsig.sum()), "edges": edge_list(names, lsig, lp)}

    # NEW subject-as-unit, three weightings
    out["new"] = {}
    new_sig = {}
    for rep in ["uniform", "weighted", "map"]:
        p, sig, eff = subject_unit_edge_tests(P_by[rep], alpha, correction)
        new_sig[rep] = (p, sig)
        out["new"][rep] = {"n_sig": int(sig.sum()),
                           "edges": edge_list(names, sig, p, eff)}

    # block-level (weighted)
    dom_names, bp, bsig, beff = block_level_tests(P_by["weighted"], domains,
                                                  alpha, correction)
    blocks = []
    for i, da in enumerate(dom_names):
        for j, db in enumerate(dom_names):
            if not np.isnan(bp[i, j]) and bsig[i, j]:
                blocks.append({"block": f"{da}->{db}", "p": float(bp[i, j]),
                               "effect_SZminusHC": float(beff[i, j])})
    out["block_level"] = {"domains": dom_names, "n_sig": len(blocks),
                          "blocks": sorted(blocks, key=lambda d: d["p"])}

    # omnibus
    def flat(P):
        n = P.shape[1]
        m = ~np.eye(n, dtype=bool)
        return P.reshape(P.shape[0], -1)[:, m.ravel()]
    X = np.vstack([flat(P_by["weighted"][g0]), flat(P_by["weighted"][g1])])
    yv = np.array([0] * gcount[g0] + [1] * gcount[g1])
    if n_perm > 0:
        auc, auc_p, chance = classifier_two_sample(X, yv, n_perm=n_perm)
        mmd, mmd_p = mmd_two_sample(X, yv, n_perm=max(n_perm * 5, 1000))
    else:
        auc = auc_p = chance = mmd = mmd_p = None
    out["omnibus"] = {"cv_auc": auc, "cv_auc_perm_p": auc_p, "chance": chance,
                      "mmd2": mmd, "mmd_perm_p": mmd_p}

    # underdetermination biomarkers (subject-as-unit)
    udet = {}
    for k in FEAT_KEYS:
        x = np.array([f[k] for f in feat_rows if f["group"] == g0], float)
        y = np.array([f[k] for f in feat_rows if f["group"] == g1], float)
        try:
            _, p = mannwhitneyu(x, y, alternative="two-sided")
        except Exception:
            p = np.nan
        udet[k] = {"HC_mean": float(x.mean()), "SZ_mean": float(y.mean()),
                   "p": float(p)}
    out["underdetermination"] = udet
    out["_feat_rows"] = feat_rows

    # bootstrap-stability edge test (only if the run saved it)
    if has_boot and all(all(v is not None for v in stabs[g]) for g in groups):
        S_by = {g: np.stack(stabs[g]) for g in groups}
        sp, ssig, seff = subject_unit_edge_tests(S_by, alpha, correction)
        out["bootstrap_edges"] = {"n_sig": int(ssig.sum()),
                                  "edges": edge_list(names, ssig, sp, seff)}
    else:
        out["bootstrap_edges"] = None

    # HEAD-TO-HEAD: do legacy edges survive subject-as-unit (weighted)?
    pw, sw = new_sig["weighted"]
    h2h, survive = [], 0
    for e in out["legacy"]["edges"]:
        i, j = _edge_idx(names, e["edge"])
        s = bool(sw[i, j])
        survive += int(s)
        h2h.append({"edge": e["edge"], "legacy_p": e["p"],
                    "subj_weighted_p": (None if np.isnan(pw[i, j]) else float(pw[i, j])),
                    "survives": s})
    out["head_to_head"] = {"legacy_n": out["legacy"]["n_sig"],
                           "survive": survive, "detail": h2h}
    out["_P_by"] = P_by
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_posterior_diff(res, names, domains, outpath):
    P = res["_P_by"]["weighted"]
    g0, g1 = res["groups"]
    diff = P[g1].mean(0) - P[g0].mean(0)        # SZ - HC posterior edge prob
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.6)))
    vmax = np.abs(diff).max() or 1.0
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("target"); ax.set_ylabel("source")
    ax.set_title("SZ - HC  cost-weighted edge posterior")
    # mark subject-as-unit significant edges
    for e in res["new"]["weighted"]["edges"]:
        i, j = _edge_idx(names, e["edge"])
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                   edgecolor="black", lw=2))
    # domain dividers
    for k in range(1, n):
        if domains[k] != domains[k - 1]:
            ax.axhline(k - .5, color="gray", lw=.5); ax.axvline(k - .5, color="gray", lw=.5)
    fig.colorbar(im, ax=ax, fraction=0.046)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Refactored fMRI analysis "
                                "(posterior / subject-as-unit / block / omnibus).")
    p.add_argument("--timestamp", required=True)
    p.add_argument("--results_root", default="fbirn_results_refactored")
    p.add_argument("--tau", default=1.0, type=float)
    p.add_argument("--alpha", default=0.05, type=float)
    p.add_argument("--correction", default="bonferroni", choices=["bonferroni", "fdr"])
    p.add_argument("--n_perm", default=200, type=int,
                   help="Permutations for omnibus tests (0 to skip).")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", default=None, help="Output dir (default <root>/<ts>/analysis_refactored)")
    return p.parse_args()


def main():
    a = parse_args()
    root = os.path.join(a.results_root, a.timestamp)
    out_dir = a.out or os.path.join(root, "analysis_refactored")
    os.makedirs(out_dir, exist_ok=True)
    configs = discover_configs(root)
    print("=" * 80)
    print("REFACTORED FMRI ANALYSIS")
    print(f"  root={root}  tau={a.tau}  alpha={a.alpha}  correction={a.correction}")
    print(f"  configs: {configs}")
    print("=" * 80)
    if not configs:
        print("No config dirs found."); sys.exit(1)

    summary_rows = []
    for cfg in configs:
        subs = load_config(os.path.join(root, cfg))
        if not subs:
            print(f"[{cfg}] no subjects, skipping."); continue
        res = analyze_config(subs, a.tau, a.alpha, a.correction, a.n_perm)
        if res is None:
            print(f"[{cfg}] <2 groups, skipping."); continue

        leg = res["legacy"]["n_sig"]
        nw = res["new"]["weighted"]["n_sig"]
        nu = res["new"]["uniform"]["n_sig"]
        bl = res["block_level"]["n_sig"]
        om = res["omnibus"]
        h2h = res["head_to_head"]
        print(f"\n[{cfg}]  subjects={sum(res['group_counts'].values())} "
              f"({res['group_counts']})")
        print(f"  LEGACY pooled (unit=solution): {leg} sig edges "
              f"(n0={res['legacy']['n0']}, n1={res['legacy']['n1']})")
        print(f"  NEW subject-as-unit: uniform={nu}  weighted={nw}  "
              f"map={res['new']['map']['n_sig']} sig edges")
        print(f"  HEAD-TO-HEAD: {h2h['survive']}/{h2h['legacy_n']} legacy edges "
              f"survive subject-as-unit (weighted)")
        print(f"  block-level: {bl} sig block-pairs")
        if om["cv_auc"] is not None:
            print(f"  OMNIBUS: CV-AUC={om['cv_auc']:.3f} p={om['cv_auc_perm_p']:.4f} "
                  f"(chance {om['chance']:.3f}) | MMD p={om['mmd_perm_p']:.4f}")
        sig_feats = [k for k in FEAT_KEYS if res["underdetermination"][k]["p"] < a.alpha]
        print(f"  underdetermination biomarkers sig (p<{a.alpha}): {sig_feats}")
        if res["bootstrap_edges"] is not None:
            print(f"  bootstrap-stability edges sig: {res['bootstrap_edges']['n_sig']}")

        # write per-config json (+ per-subject features csv)
        cdir = os.path.join(out_dir, cfg)
        os.makedirs(cdir, exist_ok=True)
        feat_rows = res.pop("_feat_rows")
        P_by = res.pop("_P_by")
        with open(os.path.join(cdir, "result.json"), "w") as jf:
            json.dump(res, jf, indent=2)
        with open(os.path.join(cdir, "subject_features.csv"), "w", newline="") as fc:
            w = csv.DictWriter(fc, fieldnames=["subject", "group"] + FEAT_KEYS)
            w.writeheader()
            for fr in feat_rows:
                w.writerow(fr)
        if a.plot:
            res["_P_by"] = P_by
            plot_posterior_diff(res, res["names"], res["domains"],
                                os.path.join(cdir, "posterior_diff.png"))

        summary_rows.append({
            "config": cfg, "n_subjects": sum(res["group_counts"].values()),
            "legacy_sig_edges": leg, "new_sig_uniform": nu, "new_sig_weighted": nw,
            "new_sig_map": res["new"]["map"]["n_sig"],
            "legacy_survive_subject_unit": h2h["survive"],
            "block_sig": bl,
            "cv_auc": om["cv_auc"], "cv_auc_p": om["cv_auc_perm_p"],
            "mmd_p": om["mmd_perm_p"],
            "biomarkers_sig": ";".join(sig_feats),
            "bootstrap_sig_edges": (res["bootstrap_edges"]["n_sig"]
                                    if res["bootstrap_edges"] else ""),
        })

    # combined comparison CSV
    comp_csv = os.path.join(out_dir, "legacy_vs_refactored_comparison.csv")
    with open(comp_csv, "w", newline="") as fc:
        fields = ["config", "n_subjects", "legacy_sig_edges", "new_sig_uniform",
                  "new_sig_weighted", "new_sig_map", "legacy_survive_subject_unit",
                  "block_sig", "cv_auc", "cv_auc_p", "mmd_p", "biomarkers_sig",
                  "bootstrap_sig_edges"]
        w = csv.DictWriter(fc, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"\nWrote comparison: {comp_csv}")
    print(f"Per-config JSON/CSV{'+plots' if a.plot else ''} under: {out_dir}")


if __name__ == "__main__":
    main()
