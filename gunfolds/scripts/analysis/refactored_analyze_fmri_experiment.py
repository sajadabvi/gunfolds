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


def load_config(config_dir, restrict=None):
    """Load every subject's result.zkl under config_dir.

    If `restrict` is a set of subject indices, only subject_<idx> dirs whose
    index is in the set are loaded -- used to match cohorts across configs/runs
    for a fair comparison. (Pure read-only stats aggregation; no gunfolds
    solver code is touched -- checklist items 1-17 are about the generation
    path and do not apply here.)
    """
    files = sorted(glob.glob(os.path.join(config_dir, "subject_*", "result.zkl")))
    subs = []
    for f in files:
        if restrict is not None:
            try:
                sidx = int(os.path.basename(os.path.dirname(f)).split("_")[-1])
            except ValueError:
                sidx = None
            if sidx not in restrict:
                continue
        try:
            subs.append(zkl.load(f))
        except Exception as e:
            print(f"  WARN load {f}: {e}")
    return subs


# ---------------------------------------------------------------------------
# "Middle ground": prevalence pre-filter + graded confidence tiers
# ---------------------------------------------------------------------------

def _bh_mask(pv_flat, q):
    """Benjamini-Hochberg mask at level q over a 1-D p-vector."""
    m = pv_flat.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pv_flat)
    thr = q * np.arange(1, m + 1) / m
    below = np.where(pv_flat[order] <= thr)[0]
    flat = np.zeros(m, dtype=bool)
    if len(below):
        flat[order[:below[-1] + 1]] = True
    return flat


def weighted_sig_mask(subjects, tau, p_thresh=0.05):
    """NxN bool mask of edges with subject-as-unit MWU p < p_thresh (weighted).

    Used to derive an independent hypothesis set from one config (e.g. PCMCI)
    to restrict another config's testing (legitimate multiplicity reduction).
    """
    groups = sorted({s["group"] for s in subjects})[:2]
    reps = {g: [] for g in groups}
    for info in subjects:
        if info["group"] in reps:
            reps[info["group"]].append(subject_posteriors(info, tau=tau)["P_weighted"])
    A, B = np.stack(reps[groups[0]]), np.stack(reps[groups[1]])
    n = A.shape[1]
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x, y = A[:, i, j], B[:, i, j]
            if np.allclose(x, x[0]) and np.allclose(y, y[0]) and x[0] == y[0]:
                continue
            try:
                if mannwhitneyu(x, y, alternative="two-sided")[1] < p_thresh:
                    mask[i, j] = True
            except Exception:
                pass
    return mask


def tiered_edge_analysis(P_by, names, present_thresh, min_prevalence, effect_min,
                         restrict_mask=None):
    """
    Subject-as-unit edge testing with (1) a LABEL-BLIND prevalence pre-filter
    that only tests edges present in >= min_prevalence of ALL subjects (pooling
    both groups, so it is independent of the group-difference statistic -> not
    double-dipping), and (2) graded confidence tiers instead of a single cutoff.

    Tiers (all subject-as-unit; correction only over the pre-filtered edges):
      confirmatory : BH-FDR q = 0.05
      exploratory  : BH-FDR q = 0.10
      suggestive   : uncorrected p < 0.05 AND |mean(SZ)-mean(HC)| >= effect_min
    """
    groups = sorted(P_by.keys())
    g0, g1 = groups
    A, B = P_by[g0], P_by[g1]                 # (n_subj, N, N) posteriors
    n = A.shape[1]
    allP = np.concatenate([A, B], axis=0)

    # label-blind prevalence: fraction of ALL subjects with posterior >= thresh
    prevalence = (allP >= present_thresh).mean(axis=0)
    testable = (prevalence >= min_prevalence) & (~np.eye(n, dtype=bool))
    if restrict_mask is not None:                # independent hypothesis set
        testable = testable & restrict_mask

    pv = np.full((n, n), np.nan)
    eff = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if not testable[i, j]:
                continue
            x, y = A[:, i, j], B[:, i, j]
            if np.allclose(x, x[0]) and np.allclose(y, y[0]) and x[0] == y[0]:
                continue
            try:
                _, p = mannwhitneyu(x, y, alternative="two-sided")
                pv[i, j] = p
                eff[i, j] = y.mean() - x.mean()
            except Exception:
                pass

    vmask = ~np.isnan(pv)
    vp = pv[vmask]

    def mask_from_flat(flat):
        out = np.zeros((n, n), dtype=bool)
        out[vmask] = flat
        return out

    conf = mask_from_flat(_bh_mask(vp, 0.05))
    expl = mask_from_flat(_bh_mask(vp, 0.10))
    sugg = vmask & (pv < 0.05) & (np.abs(eff) >= effect_min)

    return {
        "present_thresh": present_thresh, "min_prevalence": min_prevalence,
        "effect_min": effect_min,
        "n_edges_total": int(n * (n - 1)),
        "n_testable": int(testable.sum()), "n_tested": int(vmask.sum()),
        "confirmatory_fdr05": edge_list(names, conf, pv, eff),
        "exploratory_fdr10": edge_list(names, expl, pv, eff),
        "suggestive_uncorr05": edge_list(names, sugg, pv, eff),
    }


# ---------------------------------------------------------------------------
# Per-config analysis
# ---------------------------------------------------------------------------

def analyze_config(subjects, tau, alpha, correction, n_perm,
                   present_thresh=0.1, min_prevalence=0.05, effect_min=0.0,
                   restrict_mask=None):
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
    effs = {}
    for rep in ["uniform", "weighted", "map"]:
        p, sig, eff = subject_unit_edge_tests(P_by[rep], alpha, correction)
        new_sig[rep] = (p, sig)
        effs[rep] = eff
        out["new"][rep] = {"n_sig": int(sig.sum()),
                           "edges": edge_list(names, sig, p, eff)}
    out["_pmat_w"] = new_sig["weighted"][0]      # full p-matrix (for plots)
    out["_effmat_w"] = effs["weighted"]

    # MIDDLE-GROUND: prevalence pre-filter + graded tiers (weighted, subject-unit)
    out["tiered"] = tiered_edge_analysis(
        P_by["weighted"], names, present_thresh, min_prevalence, effect_min,
        restrict_mask=restrict_mask)
    out["tiered"]["restricted_to_hypothesis_set"] = restrict_mask is not None

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
# Plots  (each figure is self-explanatory: titles/labels carry the takeaway)
# ---------------------------------------------------------------------------

def _glabel(g):
    return {0: "HC (0)", 1: "SZ (1)"}.get(g, f"group {g}")


def _domain_dividers(ax, domains):
    for k in range(1, len(domains)):
        if domains[k] != domains[k - 1]:
            ax.axhline(k - .5, color="0.4", lw=.6)
            ax.axvline(k - .5, color="0.4", lw=.6)


def plot_posterior_heatmap(res, P_by, names, domains, cfg, outpath):
    """SZ-HC cost-weighted edge posterior; exploratory edges boxed; domain blocks."""
    g0, g1 = res["groups"]
    Pw = P_by["weighted"]
    diff = Pw[g1].mean(0) - Pw[g0].mean(0)
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.62), max(6, n * 0.62)))
    vmax = max(float(np.abs(diff).max()), 1e-6)
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("target node   (arrow: source row → target col)")
    ax.set_ylabel("source node")
    for e in res["tiered"]["exploratory_fdr10"]:
        i, j = _edge_idx(names, e["edge"])
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                   edgecolor="k", lw=2.4))
    _domain_dividers(ax, domains)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"{_glabel(g1)} − {_glabel(g0)}   edge-posterior prob.")
    rtag = " (restricted to PCMCI hypothesis set)" if \
        res["tiered"].get("restricted_to_hypothesis_set") else ""
    ax.set_title(f"{cfg} — group difference in causal edge posterior\n"
                 f"red = stronger in SZ, blue = stronger in HC; "
                 f"black box = exploratory edge (FDR.10{rtag})", fontsize=9)
    fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_story(res, feat_rows, cfg, outpath):
    """One 1x3 'story' figure: (A) why edges shrink, (B) the edges, (C) the biomarker."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    g0, g1 = res["groups"]

    # ---- A: unit-of-analysis effect on edge count ----
    ax = axes[0]
    leg = res["legacy"]["n_sig"]
    new_all = res["new"]["weighted"]["n_sig"]
    expl = len(res["tiered"]["exploratory_fdr10"])
    bars = ["legacy\n(per-solution,\ninflated N)", "subject-unit\n(all 90 edges)",
            "subject-unit\n(hypothesis set,\nFDR.10)"]
    vals = [leg, new_all, expl]
    cols = ["#c0392b", "#7f8c8d", "#27ae60"]
    ax.bar(bars, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02 + 0.1, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("# significant directed edges")
    ax.set_title("A. Why the count shrinks\n(pseudo-replication removed)", fontsize=10)

    # ---- B: exploratory edges forest (effect + p) ----
    ax = axes[1]
    edges = res["tiered"]["exploratory_fdr10"]
    if edges:
        edges = sorted(edges, key=lambda e: e["effect_SZminusHC"])
        ys = np.arange(len(edges))
        effs = [e["effect_SZminusHC"] for e in edges]
        cols = ["#27ae60" if e > 0 else "#2980b9" for e in effs]
        ax.barh(ys, effs, color=cols)
        ax.set_yticks(ys)
        ax.set_yticklabels([e["edge"] for e in edges], fontsize=8)
        for y, e in zip(ys, edges):
            ax.text(e["effect_SZminusHC"], y, f"  p={e['p']:.3f}",
                    va="center", fontsize=7,
                    ha="left" if e["effect_SZminusHC"] >= 0 else "right")
        ax.axvline(0, color="k", lw=.8)
        ax.set_xlabel(f"posterior diff  ({_glabel(g1)} − {_glabel(g0)})")
    else:
        ax.text(0.5, 0.5, "no exploratory edges", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("B. Surviving edges (green=SZ>HC,\nblue=HC>SZ)", fontsize=10)

    # ---- C: the most significant biomarker, HC vs SZ ----
    ax = axes[2]
    udet = res["underdetermination"]
    best_k = min(udet, key=lambda k: udet[k]["p"])
    x0 = [f[best_k] for f in feat_rows if f["group"] == g0]
    x1 = [f[best_k] for f in feat_rows if f["group"] == g1]
    parts = ax.violinplot([x0, x1], showmeans=True, showextrema=False)
    for b, c in zip(parts["bodies"], ["#3498db", "#e74c3c"]):
        b.set_facecolor(c); b.set_alpha(.6)
    ax.set_xticks([1, 2]); ax.set_xticklabels([_glabel(g0), _glabel(g1)])
    ax.set_ylabel(best_k)
    ax.set_title(f"C. Where RASL's signal lives:\n{best_k}  (p={udet[best_k]['p']:.4f})",
                 fontsize=10)

    fig.suptitle(f"{cfg}  —  subjects: {sum(res['group_counts'].values())} "
                 f"({_glabel(g0)}={res['group_counts'][g0]}, "
                 f"{_glabel(g1)}={res['group_counts'][g1]})", fontsize=11, y=1.02)
    fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_pvalue_qq(res, pmat_w, cfg, outpath):
    """QQ of subject-unit edge p-values vs uniform null: points above line = signal."""
    n = pmat_w.shape[0]
    p = pmat_w[~np.eye(n, dtype=bool)]
    p = np.sort(p[~np.isnan(p)])
    if p.size == 0:
        return
    m = p.size
    expected = (np.arange(1, m + 1) - 0.5) / m
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(-np.log10(expected), -np.log10(p), s=14, color="#34495e")
    lim = max(-np.log10(p).max(), -np.log10(expected).min(), 1) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="null (no group difference)")
    ax.set_xlabel("expected  −log10(p)")
    ax.set_ylabel("observed  −log10(p)")
    ax.set_title(f"{cfg} — edge p-value enrichment\n"
                 f"points above the red line = real (but possibly weak) signal",
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_cross_config_summary(summary_rows, outpath):
    """Grouped bars: legacy vs subject-unit(all) vs exploratory, per config."""
    if not summary_rows:
        return
    cfgs = [r["config"] for r in summary_rows]
    leg = [int(r["legacy_sig_edges"]) for r in summary_rows]
    newall = [int(r["new_allEdges_FDR"]) for r in summary_rows]
    expl = [int(r["tier_exploratory_FDR10"]) for r in summary_rows]
    x = np.arange(len(cfgs)); w = 0.26
    fig, ax = plt.subplots(figsize=(max(6, len(cfgs) * 2.2), 4.5))
    ax.bar(x - w, leg, w, label="legacy (per-solution, inflated)", color="#c0392b")
    ax.bar(x, newall, w, label="subject-unit, all edges (FDR.05)", color="#7f8c8d")
    ax.bar(x + w, expl, w, label="subject-unit, hypothesis set (FDR.10)", color="#27ae60")
    for xs, vs in [(x - w, leg), (x, newall), (x + w, expl)]:
        for xi, v in zip(xs, vs):
            ax.text(xi, v + 0.1, str(v), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(cfgs, fontsize=8)
    ax.set_ylabel("# significant directed edges")
    ax.set_title("Significant edges by method & analysis unit\n"
                 "(legacy counts are inflated by pseudo-replication)", fontsize=10)
    ax.legend(fontsize=8)
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
    # "middle ground": prevalence pre-filter + graded confidence tiers
    p.add_argument("--present_thresh", default=0.1, type=float,
                   help="Posterior >= this counts an edge 'present' in a subject "
                        "(for the prevalence pre-filter).")
    p.add_argument("--min_prevalence", default=0.05, type=float,
                   help="Only test edges present in >= this fraction of ALL "
                        "subjects (label-blind filter; lowers correction burden).")
    p.add_argument("--effect_min", default=0.0, type=float,
                   help="Min |mean(SZ)-mean(HC)| posterior diff for the "
                        "'suggestive' (uncorrected p<0.05) tier.")
    p.add_argument("--hypothesis_from", default=None,
                   help="Config name (e.g. N10_domain_PCMCI) whose subject-unit "
                        "p<0.05 edges form an INDEPENDENT hypothesis set; other "
                        "configs' edge tiers are restricted to it (legitimate "
                        "multiplicity reduction -> more power, less p-hacking).")
    p.add_argument("--restrict_subjects", default=None,
                   help="Path to a file of subject indices (one per line, or "
                        "comma/space separated). If given, EVERY config (incl. "
                        "the --hypothesis_from set) is restricted to exactly "
                        "these subject_<idx> dirs -- use to match cohorts across "
                        "configs/runs for a fair comparison (e.g. the 231 "
                        "subjects an incomplete RASL config actually finished).")
    return p.parse_args()


def main():
    a = parse_args()
    root = os.path.join(a.results_root, a.timestamp)
    out_dir = a.out or os.path.join(root, "analysis_refactored")
    os.makedirs(out_dir, exist_ok=True)
    configs = discover_configs(root)

    # Optional cohort restriction: load only these subject_<idx> dirs in every
    # config (matched-cohort comparison across runs).
    restrict = None
    if a.restrict_subjects:
        with open(a.restrict_subjects) as fh:
            restrict = {int(t) for t in fh.read().replace(",", " ").split()}

    print("=" * 80)
    print("REFACTORED FMRI ANALYSIS")
    print(f"  root={root}  tau={a.tau}  alpha={a.alpha}  correction={a.correction}")
    print(f"  configs: {configs}")
    if restrict is not None:
        print(f"  RESTRICT: {len(restrict)} subject indices from {a.restrict_subjects}")
    print(f"  out: {out_dir}")
    print("=" * 80)
    if not configs:
        print("No config dirs found."); sys.exit(1)

    # Optional independent hypothesis set from another config (e.g. PCMCI)
    hyp_mask = None
    if a.hypothesis_from:
        hsubs = load_config(os.path.join(root, a.hypothesis_from), restrict=restrict)
        if hsubs:
            hyp_mask = weighted_sig_mask(hsubs, a.tau)
            print(f"Hypothesis set from {a.hypothesis_from}: "
                  f"{int(hyp_mask.sum())} edges (subject-unit p<0.05); "
                  f"other configs' tiers restricted to these.\n")
        else:
            print(f"WARN: --hypothesis_from {a.hypothesis_from} has no subjects.\n")

    summary_rows = []
    for cfg in configs:
        subs = load_config(os.path.join(root, cfg), restrict=restrict)
        if not subs:
            print(f"[{cfg}] no subjects, skipping."); continue
        # don't restrict the hypothesis config by its own edges (circular)
        rmask = hyp_mask if (hyp_mask is not None and cfg != a.hypothesis_from) else None
        res = analyze_config(subs, a.tau, a.alpha, a.correction, a.n_perm,
                             present_thresh=a.present_thresh,
                             min_prevalence=a.min_prevalence,
                             effect_min=a.effect_min, restrict_mask=rmask)
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
        ti = res["tiered"]
        rtag = " [restricted to hypothesis set]" if ti.get("restricted_to_hypothesis_set") else ""
        print(f"  MIDDLE-GROUND{rtag} (prevalence-filtered, {ti['n_testable']}/"
              f"{ti['n_edges_total']} edges tested): "
              f"confirmatory(FDR.05)={len(ti['confirmatory_fdr05'])}  "
              f"exploratory(FDR.10)={len(ti['exploratory_fdr10'])}  "
              f"suggestive(p<.05)={len(ti['suggestive_uncorr05'])}")
        if ti["exploratory_fdr10"]:
            print("    exploratory edges: " +
                  ", ".join(f"{e['edge']}(p={e['p']:.1e})"
                            for e in ti["exploratory_fdr10"][:12]))
        if om["cv_auc"] is not None:
            print(f"  OMNIBUS: CV-AUC={om['cv_auc']:.3f} p={om['cv_auc_perm_p']:.4f} "
                  f"(chance {om['chance']:.3f}) | MMD p={om['mmd_perm_p']:.4f}")
        elif om["mmd_perm_p"] is not None:
            print(f"  OMNIBUS: MMD p={om['mmd_perm_p']:.4f} "
                  f"(classifier test skipped — sklearn not installed)")
        sig_feats = [k for k in FEAT_KEYS if res["underdetermination"][k]["p"] < a.alpha]
        print(f"  underdetermination biomarkers sig (p<{a.alpha}): {sig_feats}")
        if res["bootstrap_edges"] is not None:
            print(f"  bootstrap-stability edges sig: {res['bootstrap_edges']['n_sig']}")

        # write per-config json (+ per-subject features csv)
        cdir = os.path.join(out_dir, cfg)
        os.makedirs(cdir, exist_ok=True)
        feat_rows = res.pop("_feat_rows")
        P_by = res.pop("_P_by")
        pmat_w = res.pop("_pmat_w")
        res.pop("_effmat_w", None)
        with open(os.path.join(cdir, "result.json"), "w") as jf:
            json.dump(res, jf, indent=2)
        with open(os.path.join(cdir, "subject_features.csv"), "w", newline="") as fc:
            w = csv.DictWriter(fc, fieldnames=["subject", "group"] + FEAT_KEYS)
            w.writeheader()
            for fr in feat_rows:
                w.writerow(fr)
        if a.plot:
            plot_story(res, feat_rows, cfg, os.path.join(cdir, "story.png"))
            plot_posterior_heatmap(res, P_by, res["names"], res["domains"], cfg,
                                   os.path.join(cdir, "posterior_diff.png"))
            plot_pvalue_qq(res, pmat_w, cfg, os.path.join(cdir, "pvalue_qq.png"))

        summary_rows.append({
            "config": cfg, "n_subjects": sum(res["group_counts"].values()),
            "legacy_sig_edges": leg,
            "new_allEdges_FDR": nw,                 # strict: FDR over all 90 edges
            "restricted_to_hypothesis": ti.get("restricted_to_hypothesis_set", False),
            "edges_tested": ti["n_testable"],
            "tier_confirmatory_FDR05": len(ti["confirmatory_fdr05"]),
            "tier_exploratory_FDR10": len(ti["exploratory_fdr10"]),
            "tier_suggestive_p05": len(ti["suggestive_uncorr05"]),
            "exploratory_edges": ";".join(e["edge"] for e in ti["exploratory_fdr10"]),
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
        fields = ["config", "n_subjects", "legacy_sig_edges", "new_allEdges_FDR",
                  "restricted_to_hypothesis", "edges_tested",
                  "tier_confirmatory_FDR05", "tier_exploratory_FDR10",
                  "tier_suggestive_p05", "exploratory_edges",
                  "block_sig", "cv_auc", "cv_auc_p", "mmd_p", "biomarkers_sig",
                  "bootstrap_sig_edges"]
        w = csv.DictWriter(fc, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    if a.plot:
        plot_cross_config_summary(
            summary_rows, os.path.join(out_dir, "summary_edge_counts.png"))
    print(f"\nWrote comparison: {comp_csv}")
    print(f"Per-config JSON/CSV{'+plots' if a.plot else ''} under: {out_dir}")


if __name__ == "__main__":
    main()
