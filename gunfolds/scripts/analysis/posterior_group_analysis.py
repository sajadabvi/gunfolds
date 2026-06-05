"""
posterior_group_analysis.py
============================

Re-analysis of RASL per-subject solution sets that implements the
"cost-weighted posterior -> subject-as-unit -> block-level -> omnibus" pipeline
and contrasts it head-to-head against the legacy pooled edge-frequency analysis
(`analyze_fmri_experiment.py`).

Motivation (three weaknesses of the legacy approach):
  1. It ignores the clingo cost — all kept solutions count equally.
  2. It pools k solutions per subject and tests with #solutions as the unit,
     conflating within-subject solution multiplicity (epistemic, RASL is an
     underdetermined inverse problem) with between-subject biology. The legacy
     2x2 table uses n0,n1 = number of *solutions* (~1510/1590), not subjects
     (151/159) -> effective N inflated ~k-fold -> anti-conservative p-values.
  3. Fixed top-k + marginal edge counting ignores cost geometry and the joint
     graph structure.

WHAT RUNS ON THE EXISTING result.zkl FILES (no rerun) -- implemented here:
  - cost-weighting of the *saved* solutions (per-subject Boltzmann posterior)
  - subject-as-unit group testing (subject is the replication unit, N=311)
  - network-block-level testing (NeuroMark domains)
  - omnibus two-sample tests over graph space (CV classifier + kernel MMD)
  - underdetermination biomarkers (solution-set entropy, ESS, MAP undersampling)

WHAT NEEDS A RERUN (NOT done here; flagged in the report):
  - *full* cost-band retention: the saved sets are top-k truncated (here ~98%
    of subjects saved exactly 10 solutions with a median 31% cost spread), so
    the weighted posterior is over a truncated set. A `delta_threshold` rerun
    would capture "all graphs within Delta of optimum".
  - bootstrap stability selection (resample time series -> rerun PCMCI->RASL).

Usage:
  python posterior_group_analysis.py \
      --run_dir /Users/.../03052026215245/N10_domain_RASL \
      --tau 1.0 --alpha 0.05 --out posterior_analysis_out
"""

import os
import sys
import glob
import json
import argparse
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

# Make the repo importable when invoked as a script (script dir, not cwd, is on
# sys.path for `python file.py`).  analysis -> scripts -> gunfolds -> repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

from gunfolds.utils import zickle as zkl

# NeuroMark index -> domain map (works for any N subset via saved comp_indices)
try:
    from gunfolds.scripts.real_data.component_config import INDEX_TO_DOMAIN, DOMAIN_ORDER
except Exception:
    INDEX_TO_DOMAIN, DOMAIN_ORDER = {}, ["SC", "AU", "SM", "VI", "CC", "DM", "CB"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(run_dir):
    """Load every subject_*/result.zkl under a config dir."""
    files = sorted(glob.glob(os.path.join(run_dir, "subject_*", "result.zkl")))
    subjects = []
    for f in files:
        try:
            subjects.append(zkl.load(f))
        except Exception as e:
            print(f"  WARN: failed to load {f}: {e}")
    return subjects


def _sol_adj(sol, n):
    if "adj" in sol and sol["adj"] is not None:
        a = np.array(sol["adj"], dtype=float)
    else:
        from gunfolds import conversions as cv
        a = (cv.graph2adj(sol["graph"]) > 0).astype(float)
    np.fill_diagonal(a, 0.0)
    return a


# ---------------------------------------------------------------------------
# Per-subject representations
# ---------------------------------------------------------------------------

def subject_posteriors(info, tau=1.0):
    """
    Build per-subject edge representations from the saved solution set.

    Returns dict with:
      P_uniform   : NxN  mean over solutions (each solution weight 1/K)
      P_weighted  : NxN  Boltzmann cost-weighted posterior  w_i ~ exp(-dc_i/(tau*scale))
      P_map       : NxN  the single lowest-cost solution
      freq        : NxN  raw solution-set edge frequency (== P_uniform here)
      feats       : dict of underdetermination biomarkers
    """
    sols = info["solutions"]
    n = len(np.array(sols[0]["adj"])) if "adj" in sols[0] else len(sols[0]["graph"])
    adjs = np.stack([_sol_adj(s, n) for s in sols])          # (K, N, N)
    costs = np.array([float(s["cost"]) for s in sols], dtype=float)
    K = len(sols)

    # cost-weighted (Boltzmann) posterior over the SAVED set
    dc = costs - costs.min()
    pos = dc[dc > 0]
    scale = pos.mean() if pos.size else 1.0                   # per-subject energy scale
    if scale <= 0:
        scale = 1.0
    w = np.exp(-dc / (tau * scale))
    w = w / w.sum()

    P_uniform = adjs.mean(axis=0)
    P_weighted = np.tensordot(w, adjs, axes=(0, 0))
    best = int(np.argmin(costs))
    P_map = adjs[best]

    # ---- underdetermination biomarkers ----
    ess = 1.0 / np.sum(w ** 2)                                # effective # solutions
    cmin = costs.min()
    cost_spread = float((costs.max() - cmin) / cmin) if cmin > 0 else 0.0
    # solution-set edge entropy: how much edges flip across the (unweighted) set
    f = P_uniform[~np.eye(n, dtype=bool)]
    fe = np.clip(f, 1e-9, 1 - 1e-9)
    binent = -(fe * np.log2(fe) + (1 - fe) * np.log2(1 - fe))
    mean_edge_entropy = float(binent.mean())
    penumbra = float(np.mean((f > 0.1) & (f < 0.9)))          # ambiguous edges
    stable_core = float(np.mean(f >= 0.9))                    # edges in ~all solutions
    # undersampling of the MAP solution (None for PCMCI/GCM -> treat as u=1)
    def _u(x):
        if x is None:
            return 1
        return int(x[0]) if isinstance(x, (tuple, list)) else int(x)
    map_u = _u(sols[best]["undersampling"])
    us = np.array([_u(s["undersampling"]) for s in sols], dtype=float)
    mean_u = float(np.dot(w, us))

    feats = dict(n_solutions=K, ess=float(ess), cost_spread=cost_spread,
                 mean_edge_entropy=mean_edge_entropy, penumbra=penumbra,
                 stable_core=stable_core, map_u=map_u, mean_u=mean_u)
    return dict(P_uniform=P_uniform, P_weighted=P_weighted, P_map=P_map,
                freq=P_uniform, feats=feats, n=n)


# ---------------------------------------------------------------------------
# LEGACY method (reproduced exactly) -- pooled, solution-as-unit
# ---------------------------------------------------------------------------

def legacy_edge_tests(subjects, alpha=0.05, correction="bonferroni"):
    """Reproduce analyze_fmri_experiment.edge_level_tests on this run."""
    groups = sorted({s["group"] for s in subjects})
    g0, g1 = groups[0], groups[1]
    n = len(np.array(subjects[0]["solutions"][0]["adj"]))
    counts = {g0: np.zeros((n, n)), g1: np.zeros((n, n))}
    nsol = {g0: 0, g1: 0}
    for info in subjects:
        g = info["group"]
        for s in info["solutions"]:
            counts[g] += _sol_adj(s, n)
            nsol[g] += 1
    c0, c1 = counts[g0], counts[g1]
    n0, n1 = nsol[g0], nsol[g1]                # <-- SOLUTIONS, not subjects (the bug)
    pvals = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = int(c0[i, j]), int(c1[i, j])
            tbl = np.array([[a, b], [n0 - a, n1 - b]])
            if tbl.sum() == 0:
                continue
            try:
                if tbl.min() < 5:
                    _, p = fisher_exact(tbl)
                else:
                    _, p, _, _ = chi2_contingency(tbl, correction=True)
                pvals[i, j] = p
            except Exception:
                pass
    sig = _correct(pvals, alpha, correction)
    return pvals, sig, n0, n1


# ---------------------------------------------------------------------------
# NEW method -- subject-as-unit
# ---------------------------------------------------------------------------

def subject_unit_edge_tests(P_by_group, alpha=0.05, correction="bonferroni"):
    """
    Per-edge Mann-Whitney U on the per-subject edge probabilities.
    P_by_group[g] : (n_subjects_g, N, N) stack of per-subject matrices.
    Subject is the unit -> N = #subjects.
    """
    groups = sorted(P_by_group.keys())
    g0, g1 = groups[0], groups[1]
    A, B = P_by_group[g0], P_by_group[g1]
    n = A.shape[1]
    pvals = np.full((n, n), np.nan)
    eff = np.full((n, n), np.nan)              # effect = mean(SZ) - mean(HC)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x, y = A[:, i, j], B[:, i, j]
            if np.allclose(x, x[0]) and np.allclose(y, y[0]) and x[0] == y[0]:
                continue                        # constant -> untestable
            try:
                _, p = mannwhitneyu(x, y, alternative="two-sided")
                pvals[i, j] = p
                eff[i, j] = y.mean() - x.mean()
            except Exception:
                pass
    sig = _correct(pvals, alpha, correction)
    return pvals, sig, eff


def block_level_tests(P_by_group, domains, alpha=0.05, correction="bonferroni"):
    """
    Aggregate edges into NeuroMark domain blocks and test block->block directed
    influence between groups (subject as unit). Far fewer tests than edges.
    """
    groups = sorted(P_by_group.keys())
    g0, g1 = groups[0], groups[1]
    dom_names = [d for d in DOMAIN_ORDER if d in set(domains)]
    idx = {d: [k for k, dd in enumerate(domains) if dd == d] for d in dom_names}

    def block_tensor(P):                        # (n_subj, n_blocks, n_blocks)
        nb = len(dom_names)
        out = np.zeros((P.shape[0], nb, nb))
        for bi, da in enumerate(dom_names):
            for bj, db in enumerate(dom_names):
                rows, cols = idx[da], idx[db]
                sub = P[:, rows, :][:, :, cols]      # (n_subj, |rows|, |cols|)
                # mean over off-diagonal cells within the block-pair
                mask = np.ones((len(rows), len(cols)), dtype=bool)
                if da == db:
                    for r, rr in enumerate(rows):
                        for c, cc in enumerate(cols):
                            if rr == cc:
                                mask[r, c] = False
                if mask.sum() == 0:
                    out[:, bi, bj] = np.nan
                else:
                    out[:, bi, bj] = sub[:, mask].mean(axis=1)
        return out

    BA, BB = block_tensor(P_by_group[g0]), block_tensor(P_by_group[g1])
    nb = len(dom_names)
    pvals = np.full((nb, nb), np.nan)
    eff = np.full((nb, nb), np.nan)
    for i in range(nb):
        for j in range(nb):
            x, y = BA[:, i, j], BB[:, i, j]
            if np.all(np.isnan(x)) or np.all(np.isnan(y)):
                continue
            if np.allclose(x, x[0]) and np.allclose(y, y[0]) and x[0] == y[0]:
                continue
            try:
                _, p = mannwhitneyu(x, y, alternative="two-sided")
                pvals[i, j] = p
                eff[i, j] = np.nanmean(y) - np.nanmean(x)
            except Exception:
                pass
    sig = _correct(pvals, alpha, correction)
    return dom_names, pvals, sig, eff


# ---------------------------------------------------------------------------
# Omnibus two-sample tests over graph space
# ---------------------------------------------------------------------------

def classifier_two_sample(X, y, n_perm=200, seed=0):
    """CV-AUC of HC-vs-SZ classifier + label-permutation p-value."""
    rng = np.random.RandomState(seed)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0))

    def cv_auc(yy):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        proba = cross_val_predict(clf, X, yy, cv=skf, method="predict_proba")[:, 1]
        return roc_auc_score(yy, proba)

    obs = cv_auc(y)
    null = np.array([cv_auc(rng.permutation(y)) for _ in range(n_perm)])
    p = (1 + np.sum(null >= obs)) / (1 + n_perm)
    return float(obs), float(p), float(null.mean())


def mmd_two_sample(X, y, n_perm=2000, seed=0):
    """Unbiased RBF-MMD^2 between groups + permutation p-value (median heuristic)."""
    rng = np.random.RandomState(seed)
    g = sorted(np.unique(y))
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X, "sqeuclidean"))
    med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
    Km = np.exp(-D / (med + 1e-12))

    def mmd2(mask):
        a = np.where(mask)[0]
        b = np.where(~mask)[0]
        Kaa = Km[np.ix_(a, a)]
        Kbb = Km[np.ix_(b, b)]
        Kab = Km[np.ix_(a, b)]
        na, nb = len(a), len(b)
        ta = (Kaa.sum() - np.trace(Kaa)) / (na * (na - 1))
        tb = (Kbb.sum() - np.trace(Kbb)) / (nb * (nb - 1))
        tab = Kab.mean()
        return ta + tb - 2 * tab

    mask = (y == g[0])
    obs = mmd2(mask)
    null = np.array([mmd2(rng.permutation(mask)) for _ in range(n_perm)])
    p = (1 + np.sum(null >= obs)) / (1 + n_perm)
    return float(obs), float(p)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _correct(pvals, alpha, correction):
    n = pvals.shape[0]
    sig = np.zeros_like(pvals, dtype=bool)
    valid = ~np.isnan(pvals) & (np.eye(n) == 0)
    vp = pvals[valid]
    if vp.size == 0:
        return sig
    if correction == "fdr":
        order = np.argsort(vp)
        thr = alpha * np.arange(1, vp.size + 1) / vp.size
        below = np.where(vp[order] <= thr)[0]
        flat = np.zeros(vp.size, dtype=bool)
        if len(below):
            flat[order[:below[-1] + 1]] = True
    else:
        flat = vp < (alpha / vp.size)
    sig[valid] = flat
    return sig


def edge_list(names, mask, pvals, eff=None):
    out = []
    n = len(names)
    for i in range(n):
        for j in range(n):
            if i != j and mask[i, j]:
                e = {"edge": f"{names[i]}->{names[j]}", "p": float(pvals[i, j])}
                if eff is not None:
                    e["effect_SZminusHC"] = float(eff[i, j])
                out.append(e)
    return sorted(out, key=lambda d: d["p"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="config dir containing subject_*/result.zkl")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="Boltzmann temperature for cost weighting")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--correction", default="bonferroni",
                    choices=["bonferroni", "fdr"])
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--out", default="posterior_analysis_out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("=" * 78)
    print("POSTERIOR / SUBJECT-AS-UNIT RE-ANALYSIS")
    print("=" * 78)
    print(f"Run: {args.run_dir}")
    subjects = load_run(args.run_dir)
    if not subjects:
        print("No subjects found."); sys.exit(1)

    names = subjects[0]["comp_names"]
    comp_idx = subjects[0].get("comp_indices", list(range(len(names))))
    domains = [INDEX_TO_DOMAIN.get(ci, "??") for ci in comp_idx]
    groups = sorted({s["group"] for s in subjects})
    g0, g1 = groups[0], groups[1]
    gcount = {g: sum(1 for s in subjects if s["group"] == g) for g in groups}
    print(f"Subjects: {len(subjects)}  | group {g0}(HC?)={gcount[g0]}  "
          f"group {g1}(SZ?)={gcount[g1]}")
    print(f"Nodes ({len(names)}): {list(zip(names, domains))}")
    print(f"tau={args.tau}  alpha={args.alpha}  correction={args.correction}")

    # ---- per-subject representations ----
    reps = {g: {"uniform": [], "weighted": [], "map": []} for g in groups}
    feat_rows = []
    for info in subjects:
        r = subject_posteriors(info, tau=args.tau)
        g = info["group"]
        reps[g]["uniform"].append(r["P_uniform"])
        reps[g]["weighted"].append(r["P_weighted"])
        reps[g]["map"].append(r["P_map"])
        feat_rows.append({"subject": info["subject_id"], "group": g, **r["feats"]})
    P_by = {rep: {g: np.stack(reps[g][rep]) for g in groups}
            for rep in ["uniform", "weighted", "map"]}

    # ---- LEGACY (pooled, solution-as-unit) ----
    leg_p, leg_sig, n0, n1 = legacy_edge_tests(subjects, args.alpha, args.correction)
    leg_edges = edge_list(names, leg_sig, leg_p)
    print("\n" + "-" * 78)
    print(f"[LEGACY] pooled edge-frequency, unit = SOLUTION (n0={n0}, n1={n1})")
    print(f"  significant edges ({args.correction}): {len(leg_edges)}")

    # ---- NEW subject-as-unit, three weightings ----
    new = {}
    for rep in ["uniform", "weighted", "map"]:
        p, sig, eff = subject_unit_edge_tests(P_by[rep], args.alpha, args.correction)
        new[rep] = dict(p=p, sig=sig, eff=eff, edges=edge_list(names, sig, p, eff))
        print(f"[NEW] subject-as-unit ({rep:8s}): "
              f"{len(new[rep]['edges'])} significant edges "
              f"(unit = SUBJECT, n={gcount[g0]}/{gcount[g1]})")

    # ---- block-level (cost-weighted) ----
    dom_names, bp, bsig, beff = block_level_tests(
        P_by["weighted"], domains, args.alpha, args.correction)
    block_edges = []
    for i, da in enumerate(dom_names):
        for j, db in enumerate(dom_names):
            if not np.isnan(bp[i, j]) and bsig[i, j]:
                block_edges.append({"block": f"{da}->{db}", "p": float(bp[i, j]),
                                    "effect_SZminusHC": float(beff[i, j])})
    block_edges.sort(key=lambda d: d["p"])
    print(f"[NEW] block-level (weighted, {len(dom_names)} domains -> "
          f"{len(dom_names)**2} tests): {len(block_edges)} significant block-pairs")

    # ---- omnibus ----
    def flat(P):  # per-subject off-diagonal edge vector
        n = P.shape[1]
        m = ~np.eye(n, dtype=bool)
        return P.reshape(P.shape[0], -1)[:, m.ravel()]
    X = np.vstack([flat(P_by["weighted"][g0]), flat(P_by["weighted"][g1])])
    yv = np.array([0] * gcount[g0] + [1] * gcount[g1])
    auc, auc_p, auc_null = classifier_two_sample(X, yv, n_perm=args.n_perm)
    mmd, mmd_p = mmd_two_sample(X, yv, n_perm=max(args.n_perm * 5, 1000))
    print(f"[OMNIBUS] CV-AUC (HC vs SZ on weighted graphs): "
          f"{auc:.3f}  perm-p={auc_p:.4f}  (chance≈{auc_null:.3f})")
    print(f"[OMNIBUS] RBF-MMD^2: {mmd:.4e}  perm-p={mmd_p:.4f}")

    # ---- underdetermination biomarkers: group difference ----
    import numpy as _np
    feat_keys = ["n_solutions", "ess", "cost_spread", "mean_edge_entropy",
                 "penumbra", "stable_core", "map_u", "mean_u"]
    udet = {}
    for k in feat_keys:
        x = _np.array([f[k] for f in feat_rows if f["group"] == g0], float)
        y = _np.array([f[k] for f in feat_rows if f["group"] == g1], float)
        try:
            _, p = mannwhitneyu(x, y, alternative="two-sided")
        except Exception:
            p = _np.nan
        udet[k] = dict(HC_mean=float(x.mean()), SZ_mean=float(y.mean()), p=float(p))
    print("\n[UNDERDETERMINATION biomarkers]  (Mann-Whitney HC vs SZ)")
    for k in feat_keys:
        star = " *" if udet[k]["p"] < 0.05 else ""
        print(f"  {k:18s} HC={udet[k]['HC_mean']:.3f}  SZ={udet[k]['SZ_mean']:.3f}"
              f"  p={udet[k]['p']:.4f}{star}")

    # ---- HEAD-TO-HEAD: do the legacy 'significant' edges survive? ----
    print("\n" + "=" * 78)
    print("HEAD-TO-HEAD: legacy significant edges under subject-as-unit testing")
    print("=" * 78)
    survive = 0
    h2h = []
    for e in leg_edges:
        i, j = _edge_idx(names, e["edge"])
        pw = new["weighted"]["p"][i, j]
        sw = new["weighted"]["sig"][i, j]
        pu = new["uniform"]["p"][i, j]
        survive += int(bool(sw))
        h2h.append({"edge": e["edge"], "legacy_p": e["p"],
                    "subj_uniform_p": float(pu) if not np.isnan(pu) else None,
                    "subj_weighted_p": float(pw) if not np.isnan(pw) else None,
                    "survives_subject_unit": bool(sw)})
        print(f"  {e['edge']:22s} legacy_p={e['p']:.2e}  "
              f"subj_w_p={pw:.3e}  -> {'SURVIVES' if sw else 'drops'}")
    print(f"\n  {survive}/{len(leg_edges)} legacy edges survive subject-as-unit "
          f"({args.correction}).")

    # ---- write outputs ----
    import csv
    with open(os.path.join(args.out, "subject_features.csv"), "w", newline="") as fcsv:
        wtr = csv.DictWriter(fcsv, fieldnames=["subject", "group"] + feat_keys)
        wtr.writeheader()
        for f in feat_rows:
            wtr.writerow(f)
    summary = dict(
        run_dir=args.run_dir, n_subjects=len(subjects),
        group_counts={str(k): v for k, v in gcount.items()},
        tau=args.tau, alpha=args.alpha, correction=args.correction,
        legacy=dict(unit="solution", n0=n0, n1=n1,
                    n_sig_edges=len(leg_edges), edges=leg_edges),
        new_subject_unit={rep: dict(n_sig_edges=len(new[rep]["edges"]),
                                    edges=new[rep]["edges"])
                          for rep in ["uniform", "weighted", "map"]},
        block_level=dict(domains=dom_names, n_sig=len(block_edges), blocks=block_edges),
        omnibus=dict(cv_auc=auc, cv_auc_perm_p=auc_p, chance=auc_null,
                     mmd2=mmd, mmd_perm_p=mmd_p),
        underdetermination=udet,
        head_to_head=dict(legacy_n=len(leg_edges),
                          survive_subject_unit=survive, detail=h2h),
        note=("Cost-weighting uses the SAVED top-k solutions (median spread ~31%, "
              "~98% of subjects truncated at 10). FULL cost-band retention and "
              "bootstrap stability selection require a rerun and are NOT included."),
    )
    with open(os.path.join(args.out, "comparison_summary.json"), "w") as jf:
        json.dump(summary, jf, indent=2)
    print(f"\nWrote {args.out}/comparison_summary.json and subject_features.csv")


def _edge_idx(names, edge):
    a, b = edge.split("->")
    return names.index(a), names.index(b)


if __name__ == "__main__":
    main()
