#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roebroeck–Formisano–Goebel (2005) Granger Causality Mapping for ROI time series

Implements bivariate GCM with:
- Preprocessing: regress out linear trend + up to K low-frequency DCT cycles; z-score
- Joint VAR(p) estimated by OLS; p chosen by Schwarz Criterion (SC/BIC)
- Geweke measures: F_{x->y}, F_{y->x}, F_inst, F_total; direction via F_diff = F_{x->y} - F_{y->x}
- Empirical p-values from surrogate nulls (block bootstrap or circular shift)
- Pairwise mapping over all node pairs; optional FDR; optional graph plot

Input: data of shape (T, N) = timepoints × nodes

Usage:
    import numpy as np
    from roebroeck_gcm import run_roebroeck_gcm
    res = run_roebroeck_gcm(X, tr_sec=1.0, n_cycles=4, alpha=0.05, pmax=8, n_boot=200)
    Adj = res["Adj"]  # boolean matrix of significant directions by p_diff
"""

from typing import Optional, Tuple, Dict, Any, List
from numpy.linalg import lstsq, det
import os, datetime as _dt
import numpy as np
import pandas as pd
import itertools
import networkx as nx
import matplotlib.pyplot as plt

__all__ = ["run_roebroeck_gcm", "gcm_pair", "remove_lowfreq_dct"]

# -----------------------------
# Preprocessing
# -----------------------------


def _mk_dirs(base="./gcm_roebroeck"):
    fig_dir = os.path.join(base, "figures")
    csv_dir = os.path.join(base, "csv")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    return fig_dir, csv_dir

def _fmt_bool(b): return "1" if b else "0"

def dct_matrix(T: int) -> np.ndarray:
    """DCT-II orthonormal basis (size TxT)."""
    n = np.arange(T)
    D = np.zeros((T, T))
    D[:, 0] = 1.0 / np.sqrt(T)
    for k in range(1, T):
        D[:, k] = np.sqrt(2.0 / T) * np.cos(np.pi * (n + 0.5) * k / T)
    return D


def remove_lowfreq_dct(data: np.ndarray, n_cycles: int = 4, remove_linear: bool = True) -> np.ndarray:
    """
    Regress out DC + first n_cycles DCT components; optionally linear trend.
    Returns residuals, z-scored per column.
    """
    T = data.shape[0]
    # Basis: DC + cycles 1..n_cycles
    X = dct_matrix(T)[:, : (n_cycles + 1)]
    if remove_linear:
        t = np.linspace(-1, 1, T)[:, None]
        X = np.hstack([X, t])
    # Orthonormal projector via QR to avoid explicit inverse
    Q, _ = np.linalg.qr(X)
    beta = Q.T @ data
    fitted = Q @ beta
    resid = data - fitted
    # Z-score columns
    resid = (resid - resid.mean(axis=0, keepdims=True)) / (resid.std(axis=0, keepdims=True) + 1e-12)
    return resid


# -----------------------------
# VAR construction and fitting
# -----------------------------

def _lag_stack_xy(x: np.ndarray, y: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Joint design matrix for 2-var VAR(p): predictors [x_{t-1..p}, y_{t-1..p}, 1], targets [x_t, y_t]."""
    T = len(x)
    if len(y) != T:
        raise ValueError("x and y must have same length")
    rows = T - p
    if rows <= 2 * p + 1:
        raise ValueError("time series too short for chosen p")
    Z = np.ones((rows, 2 * p + 1))
    for t in range(p, T):
        Z[t - p, :p] = [x[t - k] for k in range(1, p + 1)]
        Z[t - p, p:2 * p] = [y[t - k] for k in range(1, p + 1)]
    Y = np.column_stack([x[p:], y[p:]])
    return Z, Y


def _lag_stack_uni(s: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Univariate AR(p) design for residual variance terms in Geweke measures."""
    T = len(s)
    rows = T - p
    Z = np.ones((rows, p + 1))
    for t in range(p, T):
        Z[t - p, :p] = [s[t - k] for k in range(1, p + 1)]
    y = s[p:]
    return Z, y


def fit_var_ols(x: np.ndarray, y: np.ndarray, p: int) -> Dict[str, Any]:
    """Fit joint VAR(p) via OLS; return residual covariance and univariate AR residual variances."""
    Z, Y = _lag_stack_xy(x, y, p)
    B, *_ = lstsq(Z, Y, rcond=None)                        # (2p+1)×2
    E = Y - Z @ B                                          # residuals (rows×2)
    N = E.shape[0]
    Sigma = (E.T @ E) / N                                  # 2×2 residual covariance

    # Univariate AR(p) residual variances
    Zx, yx = _lag_stack_uni(x, p)
    bx, *_ = lstsq(Zx, yx, rcond=None)
    ex = yx - Zx @ bx
    sig1 = float((ex @ ex) / len(ex))

    Zy, yy = _lag_stack_uni(y, p)
    by, *_ = lstsq(Zy, yy, rcond=None)
    ey = yy - Zy @ by
    t1 = float((ey @ ey) / len(ey))

    sigma2 = float(Sigma[0, 0])
    t2 = float(Sigma[1, 1])
    c12 = float(Sigma[0, 1])

    return {"B": B, "Sigma": Sigma, "sig1": sig1, "t1": t1, "sigma2": sigma2, "t2": t2, "c12": c12, "N": N}


def schwarz_criterion(Sigma: np.ndarray, N: int, p: int, D: int = 2) -> float:
    """SC(p) = ln|Σ| + (ln N / N) * p * D^2   (Luetkepohl 1991; used in Roebroeck et al.)."""
    return float(np.log(det(Sigma) + 1e-18) + (np.log(N) / N) * (p * (D ** 2)))


def choose_p_by_sc(x: np.ndarray, y: np.ndarray, pmax: int = 8) -> Tuple[int, Dict[str, Any]]:
    """Pick p in 1..pmax minimizing SC."""
    best = None
    for p in range(1, pmax + 1):
        try:
            fit = fit_var_ols(x, y, p)
            sc = schwarz_criterion(fit["Sigma"], fit["N"], p, D=2)
        except Exception:
            continue
        if best is None or sc < best[0]:
            best = (sc, p, fit)
    if best is None:
        raise ValueError("failed to fit VAR for any p")
    return best[1], best[2]


# -----------------------------
# Geweke measures
# -----------------------------

def geweke_measures_from_fit(fit: Dict[str, Any]) -> Dict[str, float]:
    """Compute Geweke's F measures from OLS fit (scalar x,y)."""
    sig1, t1 = fit["sig1"], fit["t1"]               # univariate AR residual variances
    sigma2, t2, c12 = fit["sigma2"], fit["t2"], fit["c12"]  # joint residual cov terms
    detY = sigma2 * t2 - c12 * c12

    Fx_y = float(np.log((t1 + 1e-18) / (t2 + 1e-18)))                          # x→y
    Fy_x = float(np.log((sig1 + 1e-18) / (sigma2 + 1e-18)))                    # y→x
    F_inst = float(np.log(((sigma2 + 1e-18) * (t2 + 1e-18)) / (detY + 1e-18))) # instantaneous
    F_total = float(np.log(((sig1 + 1e-18) * (t1 + 1e-18)) / (detY + 1e-18)))  # total
    return {"Fx_to_y": Fx_y, "Fy_to_x": Fy_x, "F_inst": F_inst, "F_total": F_total, "F_diff": Fx_y - Fy_x}


# -----------------------------
# Surrogate nulls and pairwise test
# -----------------------------

def surrogate(series: np.ndarray,
              mode: str = "block",
              block_len: int = 16,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Build surrogate preserving autocorrelation: 'block' bootstrap or 'cshift' circular shift."""
    if rng is None:
        rng = np.random.default_rng()
    T = len(series)
    if mode == "cshift":
        k = int(rng.integers(0, T))
        return np.roll(series, k)

    # block bootstrap: sample random contiguous blocks and concatenate
    idx = []
    i = 0
    while i < T:
        start = int(rng.integers(0, max(1, T - block_len + 1)))
        idx.extend(range(start, min(start + block_len, T)))
        i += block_len
    return series[np.array(idx[:T])]


def gcm_pair(x: np.ndarray,
             y: np.ndarray,
             pmax: int = 8,
             n_boot: int = 200,
             surr_mode: str = "block",
             block_len: int = 16,
             rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """Compute GCM for a pair with SC order selection and surrogate p-values."""
    if rng is None:
        rng = np.random.default_rng()

    p_opt, fit = choose_p_by_sc(x, y, pmax=pmax)
    obs = geweke_measures_from_fit(fit)

    null_diff = np.empty(n_boot)
    null_fx = np.empty(n_boot)
    null_fy = np.empty(n_boot)

    for b in range(n_boot):
        xb = surrogate(x, mode=surr_mode, block_len=block_len, rng=rng)
        yb = surrogate(y, mode=surr_mode, block_len=block_len, rng=rng)
        _, fitb = choose_p_by_sc(xb, yb, pmax=pmax)
        mb = geweke_measures_from_fit(fitb)
        null_diff[b] = mb["F_diff"]
        null_fx[b] = mb["Fx_to_y"]
        null_fy[b] = mb["Fy_to_x"]

    # one-sided for directed terms; two-sided for difference
    p_fx = float((np.sum(null_fx >= obs["Fx_to_y"]) + 1) / (n_boot + 1))
    p_fy = float((np.sum(null_fy >= obs["Fy_to_x"]) + 1) / (n_boot + 1))
    p_diff = float((np.sum(np.abs(null_diff) >= abs(obs["F_diff"])) + 1) / (n_boot + 1))

    return {"p_opt": p_opt, **obs, "p_fx": p_fx, "p_fy": p_fy, "p_diff": p_diff}


# -----------------------------
# Whole-matrix run
# -----------------------------

def _bh_fdr(pvals: np.ndarray, alpha: float) -> float:
    """Benjamini–Hochberg threshold for a 1-D array of p-values. Returns cutoff; 0 if none pass."""
    m = pvals.size
    order = np.argsort(pvals)
    ranks = np.arange(1, m + 1)
    thresh = (ranks / m) * alpha
    hits = np.where(pvals[order] <= thresh)[0]
    return float(pvals[order][hits[-1]]) if hits.size else 0.0

def fixed_circle_positions(names):
    # names order defines placement. First node at top, then clockwise.
    N = len(names)
    angles = np.linspace(-np.pi/2, 3*np.pi/2, N, endpoint=False)
    return {i: (float(np.cos(a)), float(np.sin(a))) for i, a in enumerate(angles)}

def run_roebroeck_gcm(
    data: np.ndarray,
    tr_sec: float = 1.0,
    n_cycles: int = 4,
    remove_linear: bool = True,
    alpha: float = 0.05,
    pmax: int = 8,
    n_boot: int = 200,
    surr_mode: str = "block",
    block_len: int = 16,
    seed: int = 0,
    fdr: bool = False,
    names: Optional[List[str]] = None,   # instead of list | None
    make_plot: bool = False,
    out_base: str = "./gcm_roebroeck",
    run_tag: Optional[str] = None        # instead of str | None
):
    """
    Run pairwise Roebroeck–Formisano–Goebel GCM over columns of `data` (shape T×N).
    Saves CSVs to ./gcm_roebroeck/csv and figure to ./gcm_roebroeck/figures with a unique tag.

    Returns dict with Fx, Fy, Finst, Fdiff, Pdiff, Pfx, Pfy, Popt, Adj, fig_path, tag, csv_prefix.
    """
    rng = np.random.default_rng(seed)
    X = remove_lowfreq_dct(np.asarray(data, float), n_cycles=n_cycles, remove_linear=remove_linear)
    T, N = X.shape

    # Unique run tag (hyperparameters + timestamp + optional run_tag)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = (
        f"T{T}_N{N}"
        f"_TR{tr_sec:g}"
        f"_nc{n_cycles}"
        f"_lin{_fmt_bool(remove_linear)}"
        f"_a{alpha:g}"
        f"_pmax{pmax}"
        f"_boot{n_boot}"
        f"_{surr_mode}"
        f"_blk{block_len}"
        f"_seed{seed}"
        f"_fdr{_fmt_bool(fdr)}"
        f"_{ts}"
    )
    if run_tag:
        safe = "".join(c for c in str(run_tag) if c.isalnum() or c in ("-", "_", ","))
        tag = f"{safe}__{tag}"

    fig_dir, csv_dir = _mk_dirs(out_base)
    prefix = os.path.join(csv_dir, f"gcm_{tag}_")

    Fx = np.zeros((N, N)); Fy = np.zeros((N, N)); Finst = np.zeros((N, N)); Fdiff = np.zeros((N, N))
    Pdiff = np.ones((N, N)); Pfx = np.ones((N, N)); Pfy = np.ones((N, N)); Popt = np.zeros((N, N), dtype=int)

    for i, j in itertools.permutations(range(N), 2):
        r = gcm_pair(
            X[:, i], X[:, j],
            pmax=pmax, n_boot=n_boot,
            surr_mode=surr_mode, block_len=block_len, rng=rng
        )
        Fx[i, j] = r["Fx_to_y"]
        Fy[i, j] = r["Fy_to_x"]
        Finst[i, j] = r["F_inst"]
        Fdiff[i, j] = r["F_diff"]
        Pdiff[i, j] = r["p_diff"]
        Pfx[i, j] = r["p_fx"]
        Pfy[i, j] = r["p_fy"]
        Popt[i, j] = r["p_opt"]

    # Adjacency: by p_diff, optionally BH-FDR
    if fdr:
        mask = ~np.eye(N, dtype=bool)
        pvals = Pdiff[mask].ravel()
        m = pvals.size
        order = np.argsort(pvals)
        ranks = np.arange(1, m + 1)
        thr = (ranks / m) * alpha
        hits = np.where(pvals[order] <= thr)[0]
        cutoff = float(pvals[order][hits[-1]]) if hits.size else 0.0
        Adj = (Pdiff < cutoff) & mask if cutoff > 0 else np.zeros_like(Pdiff, dtype=bool)
    else:
        Adj = (Pdiff < alpha) & (~np.eye(N, dtype=bool))

    # Save CSVs
    pd.DataFrame(Fx).to_csv(prefix + "Fx.csv", index=False, header=False)
    pd.DataFrame(Fy).to_csv(prefix + "Fy.csv", index=False, header=False)
    pd.DataFrame(Finst).to_csv(prefix + "Finst.csv", index=False, header=False)
    pd.DataFrame(Fdiff).to_csv(prefix + "Fdiff.csv", index=False, header=False)
    pd.DataFrame(Pdiff).to_csv(prefix + "Pdiff.csv", index=False, header=False)
    pd.DataFrame(Pfx).to_csv(prefix + "Pfx.csv", index=False, header=False)
    pd.DataFrame(Pfy).to_csv(prefix + "Pfy.csv", index=False, header=False)
    pd.DataFrame(Popt).to_csv(prefix + "Popt.csv", index=False, header=False)
    pd.DataFrame(Adj.astype(int)).to_csv(prefix + "Adj.csv", index=False, header=False)

    # Plot
    fig_path = None
    if make_plot:
        if names is None:
            names = [f"N{i}" for i in range(N)]
        G = nx.DiGraph()
        for k in range(N):
            G.add_node(k, label=names[k])
        W = np.abs(Fdiff); W = W / (W.max() + 1e-12)
        for i in range(N):
            for j in range(N):
                if i != j and Adj[i, j]:
                    G.add_edge(i, j, weight=float(W[i, j]), D=float(Fdiff[i, j]))
        pos = fixed_circle_positions(names)
        plt.figure(figsize=(6, 5))
        nx.draw_networkx_nodes(G, pos, node_size=700)
        nx.draw_networkx_labels(G, pos, labels={k: names[k] for k in range(N)}, font_size=9)
        widths = [1 + 4 * G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=widths, arrows=True, arrowstyle='-|>', arrowsize=12)
        plt.title("Roebroeck GCM: edges by p_diff")
        plt.axis('off')
        fig_path = os.path.join(fig_dir, f"gcm_{tag}.png")
        plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()

    return {
        "Fx": Fx, "Fy": Fy, "Finst": Finst, "Fdiff": Fdiff,
        "Pdiff": Pdiff, "Pfx": Pfx, "Pfy": Pfy, "Popt": Popt,
        "Adj": Adj, "fig_path": fig_path, "tag": tag, "csv_prefix": prefix
    }

# Optional CLI for quick checks
if __name__ == "__main__":
    # Small synthetic test: X -> Y and X -> Z
    rng = np.random.default_rng(4)
    T = 220
    x = np.zeros(T); y = np.zeros(T); z = np.zeros(T)
    e = rng.normal(size=(T, 3))
    for t in range(1, T):
        x[t] = 0.7 * x[t - 1] + e[t, 0]
        y[t] = 0.6 * y[t - 1] + 0.30 * x[t - 1] + e[t, 1]  # x -> y
        z[t] = 0.5 * z[t - 1] + 0.20 * x[t - 1] + e[t, 2]  # x -> z
    X = np.column_stack([x, y, z])
    X = (X - X.mean(0)) / X.std(0)

    out = run_roebroeck_gcm(X, n_cycles=4, alpha=0.05, pmax=6, n_boot=50, block_len=20, make_plot=True, names=["X","Y","Z"])
    print("Adjacency (p_diff < 0.05):")
    print(out["Adj"].astype(int))
    if out["fig_path"]:
        print("Saved graph:", out["fig_path"])