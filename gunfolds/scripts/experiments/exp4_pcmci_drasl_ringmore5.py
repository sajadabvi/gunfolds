"""
Experiment 4: PCMCI Hyperparameter Effect on DRASL Performance
             (Ring-5 + 2 Extra Edges, VAR → BOLD Simulation)

This is a natural extension of Experiment 2 (PCMCI hyperparameter grid search
on fMRI data).  Instead of measuring PCMCI stability on real fMRI subjects, we
use synthetic data with a *known* ground truth so we can directly measure how
each PCMCI configuration affects the final DRASL recovery accuracy.

Pipeline per batch:
  1. Generate a ring graph with 5 nodes + 2 additional random directed edges
     (GT = gk.ringmore(5, 2))
  2. Build a stable weighted VAR transition matrix from GT
  3. Simulate high-resolution VAR time series
  4. Convolve with a haemodynamic response function to obtain BOLD signals
  5. Undersample BOLD to simulate fMRI TR
  6. Sweep all PCMCI configurations (method × tau_max × alpha × fdr_method)
     on the *same* BOLD data
  7. Feed each PCMCI graph estimate into DRASL
  8. Evaluate every DRASL solution set against GT using orientation/adjacency F1

Output metrics per config:
  - pcmci_density:       fraction of possible edges detected by PCMCI
  - pcmci_precision/recall/f1: PCMCI edge accuracy vs GT (directed adj only)
  - drasl_n_solutions:   number of optimal DRASL solutions
  - drasl_orient_f1/p/r: best orientation F1 across DRASL solutions
  - drasl_adj_f1/p/r:    best adjacency F1 across DRASL solutions

Execution modes
---------------
  local      — run all batches sequentially in a single process
  run        — run a single batch (SLURM array element), write .zkl task file
  aggregate  — combine all task .zkl files and produce comparison plots
  plot_only  — re-plot from a previously saved aggregated .zkl

Usage examples
--------------
  # Quick local run (3 batches, 10-min timeout per DRASL call):
  python exp4_pcmci_drasl_ringmore5.py local --batches 3 --timeout 1

  # SLURM array (e.g. 20 batches → task IDs 0..19):
  python exp4_pcmci_drasl_ringmore5.py run --task_id $SLURM_ARRAY_TASK_ID \\
         --batches 20 --timestamp 04102026120000

  # Aggregate after all tasks finish:
  python exp4_pcmci_drasl_ringmore5.py aggregate --timestamp 04102026120000

  # Re-plot from saved aggregated results:
  python exp4_pcmci_drasl_ringmore5.py plot_only results_exp4/04102026120000/exp4_results.zkl
"""
import os
import sys
import glob
import itertools
import argparse
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# Ensure the project root is on sys.path however the script is invoked
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..', '..')
sys.path.insert(0, _ROOT)

from gunfolds.utils import bfutils
from gunfolds.utils import zickle as zkl
from gunfolds.utils import graphkit as gk
from gunfolds.solvers.clingo_rasl import drasl
from gunfolds.utils.calc_procs import get_process_count
from gunfolds import conversions as cv
from gunfolds.scripts.simulation import bold_function as hrf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLINGO_LIMIT = 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))

N_NODES = 5       # ring graph size
EXTRA_EDGES = 2   # additional directed edges beyond the ring

# PCMCI configuration grid (matches Experiment 2)
PCMCI_GRID = {
    "method":      ["run_pcmci", "run_pcmciplus"],
    "tau_max":     [1, 2, 3],
    "alpha_level": [0.01, 0.05, 0.1],
    "fdr_method":  ["none", "fdr_bh"],
}

DRASL_PRIORITIES = [1, 2, 1, 2, 3]
MAXCOST = 10000


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description='Exp 4: PCMCI hyperparameter effect on DRASL — ring-5 + 2 edges',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='mode')

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--batches', type=int, default=10,
                        help='Number of random graph/data batches')
    shared.add_argument('--u_rate', type=int, default=2,
                        help='BOLD undersampling rate (fMRI TR equivalent)')
    shared.add_argument('--ssize', type=int, default=2500,
                        help='VAR time-series length before BOLD (per rate unit)')
    shared.add_argument('--noise', type=float, default=0.1,
                        help='VAR noise standard deviation')
    shared.add_argument('--timeout', type=float, default=2.0,
                        help='DRASL timeout per call in hours')
    shared.add_argument('--pnum', type=int, default=PNUM,
                        help='Number of parallel clingo workers')
    shared.add_argument('--output_dir', default='results_exp4',
                        help='Base output directory')
    shared.add_argument('--timestamp', default=None,
                        help='Shared timestamp string for grouping outputs')
    shared.add_argument('--seed', type=int, default=None,
                        help='Global NumPy random seed (None = random)')

    # local mode
    local_p = sub.add_parser('local', parents=[shared],
                              help='Run all batches sequentially')

    # run mode (SLURM array element)
    run_p = sub.add_parser('run', parents=[shared],
                           help='Run one batch (SLURM array element)')
    run_p.add_argument('--task_id', type=int, required=True,
                       help='0-based SLURM array task ID == batch index')

    # aggregate mode
    agg_p = sub.add_parser('aggregate', parents=[shared],
                            help='Aggregate task .zkl files and plot')

    # plot_only mode
    plot_p = sub.add_parser('plot_only',
                             help='Re-plot from a saved aggregated .zkl')
    plot_p.add_argument('path', help='Path to aggregated .zkl file')

    return parser


# ---------------------------------------------------------------------------
# Graph + data generation
# ---------------------------------------------------------------------------

def generate_graph():
    """Return a fresh ringmore(5, 2) ground-truth graph."""
    return gk.ringmore(N_NODES, EXTRA_EDGES)


def create_stable_weighted_matrix(A, threshold=0.1, powers=(1, 2, 3, 4),
                                  max_attempts=1_000_000, damping=0.99):
    """Repeatedly draw random weights until the VAR system is stable."""
    for _ in range(max_attempts):
        W = A * np.random.randn(*A.shape)
        Ws = sp.csr_matrix(W)
        evals, _ = eigs(Ws, k=1, which='LM')
        rho = np.abs(evals[0])
        if rho == 0:
            continue
        W *= damping / rho
        ok = True
        for n in powers:
            Wn = np.linalg.matrix_power(W, n)
            nz = np.nonzero(Wn)
            if len(nz[0]) > 0 and (np.abs(Wn[nz]) < threshold).any():
                ok = False
                break
        if ok:
            return W
    raise ValueError(f'Could not find stable matrix after {max_attempts} tries.')


def simulate_var(W, ssize, noise):
    """Simulate a VAR(1) process.  Returns (n_nodes, ssize) array."""
    n = W.shape[0]
    data = np.zeros((n, ssize))
    data[:, 0] = noise * np.random.randn(n)
    for t in range(1, ssize):
        data[:, t] = W @ data[:, t - 1] + noise * np.random.randn(n)
    return data


def simulate_bold(var_data, u_rate):
    """
    Apply haemodynamic response to VAR data and undersample.

    var_data : (n_nodes, T_var)  — raw VAR time series
    Returns   : (n_nodes, T_obs) — BOLD, scaled, undersampled
    """
    data_scaled = var_data / (np.abs(var_data).max() + 1e-12)
    bold_out, _ = hrf.compute_bold_signals(data_scaled)
    # Drop first 1/5 to remove initial transient
    drop = bold_out.shape[1] // 5
    bold_out = bold_out[:, drop:]
    return bold_out[:, ::u_rate]          # (n_nodes, T_obs)


def generate_batch_data(ssize, noise, u_rate):
    """
    Generate one (GT, W, BOLD_data) batch.

    Returns
    -------
    GT   : gunfolds graph dict
    W    : stable transition matrix
    data : (n_nodes, T_obs)  — BOLD-convolved, undersampled
    """
    GT = generate_graph()
    A = cv.graph2adj(GT)

    try:
        W = create_stable_weighted_matrix(A, threshold=0.1,
                                          powers=[2, 3, 4])
    except ValueError:
        return None, None, None

    var_data = simulate_var(W, ssize=ssize * u_rate, noise=noise)
    data = simulate_bold(var_data, u_rate=u_rate)
    return GT, W, data


# ---------------------------------------------------------------------------
# PCMCI helpers
# ---------------------------------------------------------------------------

def all_pcmci_configs():
    """Return list of (method, tau_max, alpha_level, fdr_method) tuples."""
    return list(itertools.product(
        PCMCI_GRID['method'],
        PCMCI_GRID['tau_max'],
        PCMCI_GRID['alpha_level'],
        PCMCI_GRID['fdr_method'],
    ))


def config_tag(method, tau_max, alpha_level, fdr_method):
    return f'{method}_tau{tau_max}_a{alpha_level}_fdr{fdr_method}'


def run_pcmci_config(data_TN, method, tau_max, alpha_level, fdr_method):
    """
    Run one PCMCI configuration.

    data_TN : (T, N) array — tigramite convention (time × variables)
    Returns (g_est, A_mat, B_mat, results_dict)
    """
    df = pp.DataFrame(data_TN)
    pcmci_obj = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)

    if method == 'run_pcmci':
        res = pcmci_obj.run_pcmci(
            tau_max=tau_max,
            pc_alpha=None,
            alpha_level=alpha_level,
            fdr_method=fdr_method,
        )
    else:  # run_pcmciplus
        res = pcmci_obj.run_pcmciplus(
            tau_max=tau_max,
            pc_alpha=0.01,
            fdr_method=fdr_method,
        )

    g_est, A_mat, B_mat = cv.Glag2CG(res)
    return g_est, A_mat, B_mat, res


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def graph_to_dir_adj(g, n):
    """Convert gunfolds graph to binary (n×n) directed adjacency matrix."""
    adj = np.zeros((n, n), dtype=int)
    for i, nbrs in g.items():
        for j, v in nbrs.items():
            if v in (1, 3):
                adj[i - 1, j - 1] = 1
    return adj


def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def evaluate_graph_vs_gt(g_est, GT, n):
    """Return (precision, recall, F1) for directed edges only."""
    pred = graph_to_dir_adj(g_est, n)
    true = graph_to_dir_adj(GT, n)
    # Exclude self-loops
    np.fill_diagonal(pred, 0)
    np.fill_diagonal(true, 0)
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true.astype(bool)))
    fn = int(np.sum(~pred.astype(bool) & true))
    p, r, f = precision_recall_f1(tp, fp, fn)
    density = float(pred.sum()) / (n * n - n)  # excl. diagonal
    return {'precision': p, 'recall': r, 'f1': f, 'density': density}


def evaluate_drasl_solutions(solutions, GT, n):
    """
    Find the best solution (lowest OCE error) and return its metrics.
    Uses gk.OCE for a consistent, detailed breakdown.
    """
    if not solutions:
        return {
            'orient_precision': 0.0, 'orient_recall': 0.0, 'orient_f1': 0.0,
            'adj_precision': 0.0, 'adj_recall': 0.0, 'adj_f1': 0.0,
            'n_solutions': 0,
        }

    gt_dir_edges = len(gk.edgelist(GT))
    gt_bidir_edges = len(set(tuple(sorted(e)) for e in gk.bedgelist(GT)))
    n_gt_total = gt_dir_edges + gt_bidir_edges

    best_err = float('inf')
    best = None

    for answer in solutions:
        g1 = bfutils.num2CG(answer[0][0], n)
        oce = gk.OCE(g1, GT, normalized=False)
        total_err = oce['total'][0] + oce['total'][1]
        if total_err < best_err:
            best_err = total_err
            best = oce

    # Orientation (directed edges)
    om, oc = best['directed']
    tp_o = max(0, gt_dir_edges - om)
    op, orr, of1 = precision_recall_f1(tp_o, oc, om)

    # Total (adjacency — directed + bidirected)
    tm, tc = best['total']
    tp_t = max(0, n_gt_total - tm)
    ap, ar, af1 = precision_recall_f1(tp_t, tc, tm)

    return {
        'orient_precision': op, 'orient_recall': orr, 'orient_f1': of1,
        'adj_precision': ap, 'adj_recall': ar, 'adj_f1': af1,
        'n_solutions': len(solutions),
    }


# ---------------------------------------------------------------------------
# Single-batch experiment
# ---------------------------------------------------------------------------

def run_single_batch(batch_idx, ssize, noise, u_rate, timeout_hours, pnum):
    """
    Run one full batch:  graph → BOLD → all PCMCI configs → DRASL → evaluate.

    Returns a dict keyed by config_tag with per-config metrics, plus GT info.
    """
    print(f'\n[Batch {batch_idx}] Generating graph and BOLD data...')
    GT, W, data = generate_batch_data(ssize, noise, u_rate)
    if GT is None:
        print(f'  [Batch {batch_idx}] Skipped — could not create stable matrix')
        return None

    n = len(GT)
    actual_density = gk.density(GT)
    n_edges = len(gk.edgelist(GT))
    print(f'  GT: n={n}, extra={EXTRA_EDGES}, '
          f'edges={n_edges}, density={actual_density:.3f}, '
          f'BOLD shape={data.shape}')

    # tigramite wants (T, N)
    data_TN = data.T

    configs = all_pcmci_configs()
    timeout_sec = int(timeout_hours * 3600)
    batch_results = {}

    for ci, (method, tau_max, alpha_level, fdr_method) in enumerate(configs):
        tag = config_tag(method, tau_max, alpha_level, fdr_method)
        print(f'  [{ci + 1}/{len(configs)}] {tag}', flush=True)
        t0 = time.time()

        # --- PCMCI ---
        try:
            g_est, A_mat, B_mat, _ = run_pcmci_config(
                data_TN, method, tau_max, alpha_level, fdr_method)
        except Exception as exc:
            print(f'    PCMCI failed: {exc}')
            continue

        pcmci_metrics = evaluate_graph_vs_gt(g_est, GT, n)

        # --- Distance matrices for DRASL ---
        A_norm = np.abs(A_mat) / (np.abs(A_mat).max() + 1e-12)
        B_norm = np.abs(B_mat) / (np.abs(B_mat).max() + 1e-12)
        DD = (np.abs((A_norm + (cv.graph2adj(g_est) - 1)) * MAXCOST)).astype(int)
        BD = (np.abs((B_norm + (cv.graph2badj(g_est) - 1)) * MAXCOST)).astype(int)

        # --- DRASL ---
        try:
            solutions = drasl(
                [g_est],
                weighted=True,
                capsize=0,
                timeout=timeout_sec,
                urate=min(4, 3 * n + 1),
                dm=[DD],
                bdm=[BD],
                GT_density=int(1000 * actual_density),
                edge_weights=DRASL_PRIORITIES,
                pnum=pnum,
                optim='optN',
                selfloop=False,
            )
        except Exception as exc:
            print(f'    DRASL failed: {exc}')
            solutions = []

        drasl_metrics = evaluate_drasl_solutions(solutions, GT, n)
        elapsed = time.time() - t0

        batch_results[tag] = {
            'method': method,
            'tau_max': tau_max,
            'alpha_level': alpha_level,
            'fdr_method': fdr_method,
            'pcmci_density': pcmci_metrics['density'],
            'pcmci_precision': pcmci_metrics['precision'],
            'pcmci_recall': pcmci_metrics['recall'],
            'pcmci_f1': pcmci_metrics['f1'],
            **drasl_metrics,
            'elapsed_sec': round(elapsed, 1),
        }
        print(f'    PCMCI→ density={pcmci_metrics["density"]:.2f} '
              f'f1={pcmci_metrics["f1"]:.3f} | '
              f'DRASL→ orient_f1={drasl_metrics["orient_f1"]:.3f} '
              f'adj_f1={drasl_metrics["adj_f1"]:.3f} '
              f'n_sol={drasl_metrics["n_solutions"]} '
              f'({elapsed:.1f}s)')

    return {
        'batch': batch_idx,
        'n_nodes': n,
        'n_edges': n_edges,
        'actual_density': actual_density,
        'configs': batch_results,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_batch_results(all_batch_results):
    """
    Gather per-config metrics across all batches.

    Returns a dict:  config_tag → {metric: list_of_values, ...}
    """
    aggregated = defaultdict(lambda: defaultdict(list))
    scalar_fields = [
        'pcmci_density', 'pcmci_precision', 'pcmci_recall', 'pcmci_f1',
        'orient_precision', 'orient_recall', 'orient_f1',
        'adj_precision', 'adj_recall', 'adj_f1',
        'n_solutions', 'elapsed_sec',
    ]
    meta_fields = ['method', 'tau_max', 'alpha_level', 'fdr_method']

    for batch in all_batch_results:
        if batch is None:
            continue
        for tag, rec in batch['configs'].items():
            for f in scalar_fields:
                if f in rec:
                    aggregated[tag][f].append(rec[f])
            for f in meta_fields:
                aggregated[tag][f] = rec[f]   # same for all batches

    return dict(aggregated)


def summary_table(agg):
    """Print a ranked summary table."""
    rows = []
    for tag, metrics in agg.items():
        if not metrics.get('orient_f1'):
            continue
        rows.append({
            'config': tag,
            'method': metrics['method'],
            'tau_max': metrics['tau_max'],
            'alpha': metrics['alpha_level'],
            'fdr': metrics['fdr_method'],
            'pcmci_d': np.mean(metrics['pcmci_density']),
            'pcmci_f1': np.mean(metrics['pcmci_f1']),
            'drasl_ori_f1': np.mean(metrics['orient_f1']),
            'drasl_adj_f1': np.mean(metrics['adj_f1']),
            'n_sol': np.mean(metrics['n_solutions']),
            'n': len(metrics['orient_f1']),
        })

    rows.sort(key=lambda r: -r['drasl_ori_f1'])

    hdr = (f"{'Config':<55} {'PCMCI-d':>8} {'PCMCI-F1':>9} "
           f"{'Ori-F1':>8} {'Adj-F1':>8} {'nSol':>6} {'N':>4}")
    print('\n' + '=' * len(hdr))
    print('EXPERIMENT 4 RESULTS — sorted by DRASL orientation F1')
    print('=' * len(hdr))
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['config']:<55} {r['pcmci_d']:>8.3f} {r['pcmci_f1']:>9.3f} "
              f"{r['drasl_ori_f1']:>8.3f} {r['drasl_adj_f1']:>8.3f} "
              f"{r['n_sol']:>6.1f} {r['n']:>4}")
    print('=' * len(hdr))

    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(rows, output_path):
    """
    Two-panel comparison plot:
      Left  — DRASL orientation F1 per config
      Right — PCMCI F1 vs DRASL orientation F1 scatter
    """
    if not rows:
        print('No data to plot.')
        return

    df = pd.DataFrame(rows).sort_values('drasl_ori_f1', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(df) * 0.3 + 1)))

    # Panel 1: horizontal bar of DRASL orientation F1 coloured by method
    ax = axes[0]
    colours = {'run_pcmci': '#2196F3', 'run_pcmciplus': '#FF5722'}
    for i, row in df.iterrows():
        c = colours.get(row['method'], 'grey')
        ax.barh(row['config'], row['drasl_ori_f1'], color=c, alpha=0.8)
    ax.set_xlabel('DRASL Orientation F1', fontsize=11)
    ax.set_title('DRASL Orientation F1 by PCMCI Config', fontsize=12)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color='k', linestyle='--', linewidth=0.7, alpha=0.4)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colours['run_pcmci'],
                       label='run_pcmci'),
        plt.Rectangle((0, 0), 1, 1, color=colours['run_pcmciplus'],
                       label='run_pcmciplus'),
    ]
    ax.legend(handles=handles, fontsize=9)
    ax.tick_params(axis='y', labelsize=7)

    # Panel 2: PCMCI F1 vs DRASL orientation F1 scatter
    ax2 = axes[1]
    for method, grp in df.groupby('method'):
        c = colours.get(method, 'grey')
        ax2.scatter(grp['pcmci_f1'], grp['drasl_ori_f1'],
                    c=c, label=method, alpha=0.7, s=60)
        for _, row in grp.iterrows():
            ax2.annotate(f"τ{row['tau_max']}",
                         (row['pcmci_f1'], row['drasl_ori_f1']),
                         fontsize=6, alpha=0.6,
                         textcoords='offset points', xytext=(3, 3))

    ax2.set_xlabel('PCMCI Directed F1 vs GT', fontsize=11)
    ax2.set_ylabel('DRASL Orientation F1 vs GT', fontsize=11)
    ax2.set_title('PCMCI Input Quality → DRASL Accuracy', fontsize=12)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='diagonal')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {output_path}')


def plot_tau_sweep(rows, output_path):
    """
    Show how tau_max and method interact with DRASL F1.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    tau_vals = sorted(df['tau_max'].unique())
    methods = sorted(df['method'].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = {'run_pcmci': '#2196F3', 'run_pcmciplus': '#FF5722'}
    width = 0.35
    x = np.arange(len(tau_vals))

    for i, method in enumerate(methods):
        grp = df[df['method'] == method]
        means = [grp[grp['tau_max'] == t]['drasl_ori_f1'].mean() for t in tau_vals]
        stds  = [grp[grp['tau_max'] == t]['drasl_ori_f1'].std() for t in tau_vals]
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               color=colours.get(method, 'grey'), label=method, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'tau={t}' for t in tau_vals], fontsize=10)
    ax.set_ylabel('DRASL Orientation F1', fontsize=11)
    ax.set_title('Effect of tau_max and PCMCI Method on DRASL Accuracy\n'
                 f'(Ring-{N_NODES} + {EXTRA_EDGES} extra edges, VAR→BOLD)',
                 fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Tau-sweep plot saved to {output_path}')


def plot_alpha_fdr_heatmap(rows, output_path):
    """
    For run_pcmci configs only: heatmap of DRASL F1 across alpha × fdr combos.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    pcmci_df = df[df['method'] == 'run_pcmci'].copy()
    if pcmci_df.empty:
        return

    alpha_vals = sorted(pcmci_df['alpha'].unique())
    fdr_vals   = sorted(pcmci_df['fdr'].unique())
    tau_vals   = sorted(pcmci_df['tau_max'].unique())

    n_tau = len(tau_vals)
    fig, axes = plt.subplots(1, n_tau, figsize=(5 * n_tau, 4), squeeze=False)

    for col, tau in enumerate(tau_vals):
        ax = axes[0][col]
        mat = np.zeros((len(alpha_vals), len(fdr_vals)))
        for ri, alpha in enumerate(alpha_vals):
            for ci, fdr in enumerate(fdr_vals):
                sub = pcmci_df[
                    (pcmci_df['tau_max'] == tau) &
                    (pcmci_df['alpha'] == alpha) &
                    (pcmci_df['fdr'] == fdr)
                ]
                mat[ri, ci] = sub['drasl_ori_f1'].mean() if not sub.empty else 0.0

        im = ax.imshow(mat, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(fdr_vals)))
        ax.set_xticklabels(fdr_vals, fontsize=9)
        ax.set_yticks(range(len(alpha_vals)))
        ax.set_yticklabels([str(a) for a in alpha_vals], fontsize=9)
        ax.set_xlabel('FDR method', fontsize=10)
        ax.set_ylabel('alpha_level', fontsize=10)
        ax.set_title(f'run_pcmci, tau_max={tau}', fontsize=11)
        for ri in range(len(alpha_vals)):
            for ci in range(len(fdr_vals)):
                ax.text(ci, ri, f'{mat[ri, ci]:.2f}',
                        ha='center', va='center', fontsize=9,
                        color='black')
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('DRASL Orientation F1 — PCMCI alpha × FDR heatmap', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Alpha/FDR heatmap saved to {output_path}')


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------

def mode_local(args):
    if args.seed is not None:
        np.random.seed(args.seed)

    ts = args.timestamp or datetime.now().strftime('%m%d%Y%H%M%S')
    out_dir = os.path.join(args.output_dir, ts)
    os.makedirs(out_dir, exist_ok=True)

    all_batches = []
    for b in range(1, args.batches + 1):
        result = run_single_batch(
            batch_idx=b,
            ssize=args.ssize,
            noise=args.noise,
            u_rate=args.u_rate,
            timeout_hours=args.timeout,
            pnum=args.pnum,
        )
        if result is not None:
            all_batches.append(result)

    _save_and_plot(all_batches, out_dir)


def mode_run(args):
    """Run a single batch identified by task_id (== batch index - 1)."""
    batch_idx = args.task_id + 1  # 1-based for readability

    if args.seed is not None:
        np.random.seed(args.seed + args.task_id)

    result = run_single_batch(
        batch_idx=batch_idx,
        ssize=args.ssize,
        noise=args.noise,
        u_rate=args.u_rate,
        timeout_hours=args.timeout,
        pnum=args.pnum,
    )

    ts = args.timestamp or 'notimestamp'
    task_dir = os.path.join(args.output_dir, ts, 'tasks')
    os.makedirs(task_dir, exist_ok=True)

    out_path = os.path.join(task_dir, f'task_{args.task_id:04d}.zkl')
    zkl.save({'task_id': args.task_id, 'result': result}, out_path)
    print(f'Task {args.task_id} result saved to {out_path}')


def mode_aggregate(args):
    ts = args.timestamp or 'notimestamp'
    task_dir = os.path.join(args.output_dir, ts, 'tasks')

    if not os.path.isdir(task_dir):
        print(f'ERROR: task directory not found: {task_dir}')
        sys.exit(1)

    task_files = sorted(glob.glob(os.path.join(task_dir, 'task_*.zkl')))
    print(f'Found {len(task_files)} task file(s) in {task_dir}')

    all_batches = []
    for fpath in task_files:
        payload = zkl.load(fpath)
        if payload.get('result') is not None:
            all_batches.append(payload['result'])

    out_dir = os.path.join(args.output_dir, ts)
    _save_and_plot(all_batches, out_dir)


def mode_plot_only(args):
    print(f'Loading results from {args.path}')
    saved = zkl.load(args.path)
    agg = saved.get('aggregated', {})
    rows = summary_table(agg)
    out_base = args.path.replace('.zkl', '')
    plot_results(rows, out_base + '_comparison.pdf')
    plot_tau_sweep(rows, out_base + '_tau_sweep.pdf')
    plot_alpha_fdr_heatmap(rows, out_base + '_alpha_fdr_heatmap.pdf')


def _save_and_plot(all_batches, out_dir):
    """Shared save + plot logic used by local and aggregate modes."""
    os.makedirs(out_dir, exist_ok=True)
    agg = aggregate_batch_results(all_batches)
    rows = summary_table(agg)

    save_path = os.path.join(out_dir, 'exp4_results.zkl')
    zkl.save({'all_batches': all_batches, 'aggregated': agg, 'rows': rows},
             save_path)
    print(f'\nResults saved to {save_path}')

    plot_results(rows, os.path.join(out_dir, 'exp4_comparison.pdf'))
    plot_tau_sweep(rows, os.path.join(out_dir, 'exp4_tau_sweep.pdf'))
    plot_alpha_fdr_heatmap(rows, os.path.join(out_dir, 'exp4_alpha_fdr_heatmap.pdf'))

    # Also dump a CSV for easy inspection
    csv_path = os.path.join(out_dir, 'exp4_summary.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'Summary CSV saved to {csv_path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        'local':     mode_local,
        'run':       mode_run,
        'aggregate': mode_aggregate,
        'plot_only': mode_plot_only,
    }
    dispatch[args.mode](args)


if __name__ == '__main__':
    main()
