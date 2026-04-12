"""
Experiment 4 (saved-data variant): PCMCI Hyperparameter Effect on DRASL
       using pre-generated VAR→BOLD ringmore data files.

Loads existing data and ground-truth files from disk (no simulation needed),
runs all 36 PCMCI configurations on each batch, feeds each graph estimate
into DRASL, and compares recovery accuracy across configurations.

Data expected at:
  <data_dir>/data{B}.txt   — tab-separated, header X1..XN, (T × N)
  <gt_dir>/GT{B}.zkl       — gunfolds graph dict (ground truth)

Usage
-----
  conda activate gunfolds
  python exp4_saved_data.py \\
      --data_dir /path/to/u100/txt \\
      --gt_dir   /path/to/u100/GT  \\
      --batches  1 2 3            \\
      --timeout  1.0              \\
      --output_dir results_exp4_saved

The script writes:
  <output_dir>/exp4_saved_results.json   — full per-config metrics (JSON)
  <output_dir>/exp4_saved_summary.csv    — aggregated summary table
  <output_dir>/exp4_saved_comparison.pdf — horizontal-bar F1 chart
  <output_dir>/exp4_saved_tau_sweep.pdf  — tau_max × method bar chart
  <output_dir>/exp4_saved_heatmap.pdf    — alpha × FDR heatmap (run_pcmci only)
"""
import os
import sys
import json
import time
import argparse
import itertools
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _ROOT)

from gunfolds.utils import bfutils, graphkit as gk, zickle as zkl
from gunfolds.solvers.clingo_rasl import drasl
from gunfolds.utils.calc_procs import get_process_count
from gunfolds import conversions as cv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLINGO_LIMIT = 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))
MAXCOST = 10000
DRASL_PRIORITIES = [1, 1, 1, 1, 2]

PCMCI_GRID = {
    'method':      ['run_pcmci'],
    'tau_max':     [1],
    'alpha_level': [0.05],
    'fdr_method':  ['none'],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Exp 4 on saved data: 36 PCMCI configs × DRASL comparison',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data_dir',
        default='/Users/mabavisani/DataSets_Feedbacks/'
                '9_VAR_BOLD_simulation/ringmore/u100/txt',
        help='Directory containing data{B}.txt files',
    )
    parser.add_argument(
        '--gt_dir',
        default='/Users/mabavisani/DataSets_Feedbacks/'
                '9_VAR_BOLD_simulation/ringmore/u100/GT',
        help='Directory containing GT{B}.zkl files',
    )
    parser.add_argument(
        '--batches', nargs='+', type=int, default=[1],
        help='Batch indices to process (space-separated)',
    )
    parser.add_argument(
        '--timeout', type=float, default=1.0,
        help='DRASL timeout per call (hours)',
    )
    parser.add_argument(
        '--pnum', type=int, default=PNUM,
        help='Number of parallel clingo workers',
    )
    parser.add_argument(
        '--output_dir', default='results_exp4_saved',
        help='Output directory for results and plots',
    )
    parser.add_argument(
        '--selfloop', action='store_true', default=False,
        help='Include self-loops in DRASL and evaluation',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_batch(batch_idx, data_dir, gt_dir):
    """Return (data_TN, GT) for a batch, or (None, None) on failure."""
    data_path = os.path.join(data_dir, f'data{batch_idx}.txt')
    gt_path   = os.path.join(gt_dir,   f'GT{batch_idx}.zkl')

    if not os.path.isfile(data_path):
        print(f'  [Batch {batch_idx}] Data file not found: {data_path}')
        return None, None
    if not os.path.isfile(gt_path):
        print(f'  [Batch {batch_idx}] GT file not found: {gt_path}')
        return None, None

    df   = pd.read_csv(data_path, delimiter='\t')
    data_TN = df.values          # shape (T, N) — correct for tigramite
    GT   = zkl.load(gt_path)

    T, N = data_TN.shape
    print(f'  [Batch {batch_idx}] Loaded: T={T}, N={N}, GT={GT}')
    return data_TN, GT


# ---------------------------------------------------------------------------
# PCMCI
# ---------------------------------------------------------------------------

def run_pcmci_config(data_TN, method, tau_max, alpha_level, fdr_method):
    """Run one PCMCI configuration; return (g_est, A_mat, B_mat)."""
    df_tg = pp.DataFrame(data_TN)
    pcmci_obj = PCMCI(dataframe=df_tg, cond_ind_test=ParCorr(), verbosity=0)

    if method == 'run_pcmci':
        res = pcmci_obj.run_pcmci(
            tau_max=tau_max, pc_alpha=None,
            alpha_level=alpha_level, fdr_method=fdr_method,
        )
    else:
        res = pcmci_obj.run_pcmciplus(
            tau_max=tau_max, pc_alpha=0.01,
            fdr_method=fdr_method,
        )

    g_est, A_mat, B_mat = cv.Glag2CG(res)
    return g_est, A_mat, B_mat


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def graph_to_dir_adj(g, n, exclude_selfloops=True):
    adj = np.zeros((n, n), dtype=bool)
    for i, nbrs in g.items():
        for j, v in nbrs.items():
            if exclude_selfloops and i == j:
                continue
            if v in (1, 3):
                adj[i - 1, j - 1] = True
    return adj


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def eval_pcmci(g_est, GT, n, exclude_sl=True):
    """Directed-edge precision/recall/F1 of PCMCI estimate vs GT."""
    pred = graph_to_dir_adj(g_est, n, exclude_sl)
    true = graph_to_dir_adj(GT,    n, exclude_sl)
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    p, r, f = prf(tp, fp, fn)
    denom = n * n - n if exclude_sl else n * n
    density = float(pred.sum()) / denom if denom > 0 else 0.0
    return {'precision': p, 'recall': r, 'f1': f, 'density': round(density, 4),
            'tp': tp, 'fp': fp, 'fn': fn}


def eval_drasl(solutions, GT, n, exclude_sl=True):
    """Find best DRASL solution (lowest OCE error) and return F1 metrics."""
    if not solutions:
        return {
            'orient_p': 0.0, 'orient_r': 0.0, 'orient_f1': 0.0,
            'adj_p': 0.0, 'adj_r': 0.0, 'adj_f1': 0.0,
            'n_solutions': 0,
        }

    gt_dir  = len([e for e in gk.edgelist(GT) if e[0] != e[1]]) if exclude_sl else len(gk.edgelist(GT))
    gt_bidir = len(set(tuple(sorted(e)) for e in gk.bedgelist(GT)))
    n_gt_total = gt_dir + gt_bidir

    best_err = float('inf')
    best_oce = None
    for answer in solutions:
        g1  = bfutils.num2CG(answer[0][0], n)
        oce = gk.OCE(g1, GT, normalized=False)
        err = oce['total'][0] + oce['total'][1]
        if err < best_err:
            best_err = err
            best_oce = oce

    om, oc = best_oce['directed']
    tp_o = max(0, gt_dir - om)
    op, orr, of1 = prf(tp_o, oc, om)

    tm, tc = best_oce['total']
    tp_t = max(0, n_gt_total - tm)
    ap, ar, af1 = prf(tp_t, tc, tm)

    return {
        'orient_p': op, 'orient_r': orr, 'orient_f1': of1,
        'adj_p': ap, 'adj_r': ar, 'adj_f1': af1,
        'n_solutions': len(solutions),
    }


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_experiment(args):
    os.makedirs(args.output_dir, exist_ok=True)
    timeout_sec = int(args.timeout * 3600)
    configs = list(itertools.product(
        PCMCI_GRID['method'], PCMCI_GRID['tau_max'],
        PCMCI_GRID['alpha_level'], PCMCI_GRID['fdr_method'],
    ))
    n_configs = len(configs)
    print(f'Grid: {n_configs} PCMCI configurations × {len(args.batches)} batches '
          f'= {n_configs * len(args.batches)} runs\n')

    # results[tag][batch_idx] = metric_dict
    all_results = {}   # tag → list of per-batch dicts
    meta = {}          # tag → {method, tau_max, alpha_level, fdr_method}

    for method, tau_max, alpha_level, fdr_method in configs:
        tag = f'{method}_tau{tau_max}_a{alpha_level}_fdr{fdr_method}'
        all_results[tag] = []
        meta[tag] = dict(method=method, tau_max=tau_max,
                         alpha_level=alpha_level, fdr_method=fdr_method)

    for batch_idx in args.batches:
        print(f'\n{"="*70}')
        print(f'BATCH {batch_idx}')
        print('='*70)
        data_TN, GT = load_batch(batch_idx, args.data_dir, args.gt_dir)
        if data_TN is None:
            continue

        n = data_TN.shape[1]
        actual_density = gk.density(GT)
        exclude_sl = not args.selfloop

        for ci, (method, tau_max, alpha_level, fdr_method) in enumerate(configs, 1):
            tag = f'{method}_tau{tau_max}_a{alpha_level}_fdr{fdr_method}'
            print(f'  [{ci:02d}/{n_configs}] {tag}', flush=True)
            t0 = time.time()

            # --- PCMCI ---
            try:
                g_est, A_mat, B_mat = run_pcmci_config(
                    data_TN, method, tau_max, alpha_level, fdr_method)
            except Exception as exc:
                print(f'    PCMCI FAILED: {exc}')
                all_results[tag].append({'batch': batch_idx, 'error': str(exc)})
                continue

            pcmci_m = eval_pcmci(g_est, GT, n, exclude_sl)

            # --- Distance matrices ---
            A_scale = np.abs(A_mat) / (np.abs(A_mat).max() + 1e-12)
            B_scale = np.abs(B_mat) / (np.abs(B_mat).max() + 1e-12)
            DD = (np.abs((A_scale + (cv.graph2adj(g_est)  - 1)) * MAXCOST)).astype(int)
            BD = (np.abs((B_scale + (cv.graph2badj(g_est) - 1)) * MAXCOST)).astype(int)

            # --- DRASL ---
            try:
                solutions = drasl(
                    [g_est],
                    weighted=True, capsize=0,
                    timeout=timeout_sec,
                    urate=min(8, 3 * n + 1),
                    dm=[DD], bdm=[BD],
                    GT_density=int(1000 * actual_density),
                    edge_weights=DRASL_PRIORITIES,
                    pnum=args.pnum, optim='optN',
                    selfloop=args.selfloop,
                )
            except Exception as exc:
                print(f'    DRASL FAILED: {exc}')
                solutions = []

            drasl_m = eval_drasl(solutions, GT, n, exclude_sl)
            elapsed = time.time() - t0

            rec = {
                'batch': batch_idx,
                **pcmci_m,
                **drasl_m,
                'elapsed_sec': round(elapsed, 1),
            }
            all_results[tag].append(rec)

            print(f'    PCMCI→ d={pcmci_m["density"]:.2f} '
                  f'f1={pcmci_m["f1"]:.3f}  |  '
                  f'DRASL→ ori_f1={drasl_m["orient_f1"]:.3f} '
                  f'adj_f1={drasl_m["adj_f1"]:.3f} '
                  f'n_sol={drasl_m["n_solutions"]} '
                  f'({elapsed:.1f}s)')

    return all_results, meta


# ---------------------------------------------------------------------------
# Aggregation + summary
# ---------------------------------------------------------------------------

def aggregate(all_results, meta):
    """Return list of summary dicts (one per config), sorted by orient_f1 desc."""
    rows = []
    scalar_keys = [
        'density', 'f1', 'precision', 'recall',
        'orient_f1', 'orient_p', 'orient_r',
        'adj_f1', 'adj_p', 'adj_r',
        'n_solutions', 'elapsed_sec',
    ]
    for tag, records in all_results.items():
        valid = [r for r in records if 'error' not in r]
        if not valid:
            continue
        row = {'config': tag, **meta[tag]}
        for k in scalar_keys:
            vals = [r[k] for r in valid if k in r]
            row[f'{k}_mean'] = round(float(np.mean(vals)), 4) if vals else None
            row[f'{k}_std']  = round(float(np.std(vals)),  4) if vals else None
        row['n_batches'] = len(valid)
        rows.append(row)

    rows.sort(key=lambda r: -(r['orient_f1_mean'] or 0))
    return rows


def print_summary(rows):
    w = 56
    hdr = (f"{'Config':<{w}} {'PCMCI-d':>8} {'PCMCI-F1':>9} "
           f"{'Ori-F1':>8} {'Adj-F1':>8} {'nSol':>6} {'N':>3}")
    sep = '=' * len(hdr)
    print(f'\n{sep}')
    print('EXPERIMENT 4 (saved data) — ranked by DRASL orientation F1')
    print(sep)
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['config']:<{w}} "
              f"{r['density_mean'] or 0:>8.3f} "
              f"{r['f1_mean'] or 0:>9.3f} "
              f"{r['orient_f1_mean'] or 0:>8.3f} "
              f"{r['adj_f1_mean'] or 0:>8.3f} "
              f"{r['n_solutions_mean'] or 0:>6.1f} "
              f"{r['n_batches']:>3}")
    print(sep)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(rows, path):
    """Horizontal bar chart of DRASL orientation F1 per config."""
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values('orient_f1_mean', ascending=True)

    fig_h = max(6, len(df) * 0.32 + 1)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    colours = {'run_pcmci': '#2196F3', 'run_pcmciplus': '#FF5722'}
    for _, row in df.iterrows():
        c = colours.get(row['method'], 'grey')
        ax.barh(row['config'], row['orient_f1_mean'] or 0, color=c, alpha=0.8,
                xerr=row['orient_f1_std'] or 0, capsize=2)

    ax.set_xlabel('DRASL Orientation F1 (mean ± std)', fontsize=11)
    ax.set_title('DRASL Orientation F1 by PCMCI Config\n'
                 '(Ring-5 + 2 extra edges, VAR→BOLD, u=2)', fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color='k', linestyle='--', linewidth=0.7, alpha=0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=m)
               for m, c in colours.items()]
    ax.legend(handles=handles, fontsize=9, loc='lower right')
    ax.tick_params(axis='y', labelsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Comparison chart → {path}')


def plot_pcmci_vs_drasl(rows, path):
    """Scatter: PCMCI F1 vs DRASL orientation F1, coloured by method."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    colours = {'run_pcmci': '#2196F3', 'run_pcmciplus': '#FF5722'}

    fig, ax = plt.subplots(figsize=(7, 6))
    for method, grp in df.groupby('method'):
        c = colours.get(method, 'grey')
        ax.errorbar(grp['f1_mean'], grp['orient_f1_mean'],
                    xerr=grp['f1_std'], yerr=grp['orient_f1_std'],
                    fmt='o', color=c, label=method, alpha=0.75,
                    markersize=7, capsize=3, linewidth=1)
        for _, row in grp.iterrows():
            ax.annotate(f"τ{row['tau_max']}",
                        (row['f1_mean'], row['orient_f1_mean']),
                        fontsize=7, alpha=0.65,
                        textcoords='offset points', xytext=(4, 3))

    ax.set_xlabel('PCMCI Directed F1 vs GT', fontsize=11)
    ax.set_ylabel('DRASL Orientation F1 vs GT', fontsize=11)
    ax.set_title('PCMCI Input Quality → DRASL Accuracy\n'
                 '(VAR→BOLD ringmore-5, u=2)', fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.25, label='diagonal')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Scatter plot → {path}')


def plot_tau_sweep(rows, path):
    """Grouped bar: DRASL orientation F1 by tau_max × method."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    tau_vals  = sorted(df['tau_max'].unique())
    methods   = sorted(df['method'].unique())
    colours   = {'run_pcmci': '#2196F3', 'run_pcmciplus': '#FF5722'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    metrics_panels = [
        ('orient_f1_mean', 'orient_f1_std', 'DRASL Orientation F1'),
        ('f1_mean',        'f1_std',        'PCMCI Directed F1 vs GT'),
    ]

    for ax, (mean_col, std_col, title) in zip(axes, metrics_panels):
        x     = np.arange(len(tau_vals))
        width = 0.35
        for i, method in enumerate(methods):
            grp   = df[df['method'] == method]
            means = [grp[grp['tau_max'] == t][mean_col].mean() for t in tau_vals]
            stds  = [grp[grp['tau_max'] == t][std_col].mean()  for t in tau_vals]
            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                   color=colours.get(method, 'grey'), label=method, alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels([f'tau_max={t}' for t in tau_vals], fontsize=10)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.2)

    fig.suptitle('Effect of tau_max and PCMCI Method\n'
                 '(VAR→BOLD ringmore-5, u=2)', fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Tau-sweep chart → {path}')


def plot_alpha_fdr_heatmap(rows, path):
    """
    For run_pcmci only: 1 × tau_max grid of heatmaps showing
    DRASL orientation F1 as a function of alpha_level × fdr_method.
    """
    if not rows:
        return
    df      = pd.DataFrame(rows)
    pcmci_df = df[df['method'] == 'run_pcmci'].copy()
    if pcmci_df.empty:
        return

    alpha_vals = sorted(pcmci_df['alpha_level'].unique())
    fdr_vals   = sorted(pcmci_df['fdr_method'].unique())
    tau_vals   = sorted(pcmci_df['tau_max'].unique())

    fig, axes = plt.subplots(1, len(tau_vals),
                             figsize=(5 * len(tau_vals), 4.2), squeeze=False)

    for col, tau in enumerate(tau_vals):
        ax = axes[0][col]
        mat = np.zeros((len(alpha_vals), len(fdr_vals)))
        for ri, alpha in enumerate(alpha_vals):
            for ci, fdr in enumerate(fdr_vals):
                sub = pcmci_df[
                    (pcmci_df['tau_max']     == tau) &
                    (pcmci_df['alpha_level'] == alpha) &
                    (pcmci_df['fdr_method']  == fdr)
                ]
                mat[ri, ci] = sub['orient_f1_mean'].mean() if not sub.empty else 0.0

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
                ax.text(ci, ri, f'{mat[ri, ci]:.3f}',
                        ha='center', va='center', fontsize=10, color='black')
        plt.colorbar(im, ax=ax, shrink=0.82)

    fig.suptitle('DRASL Orientation F1 — alpha × FDR heatmap (run_pcmci)\n'
                 '(VAR→BOLD ringmore-5, u=2)', fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Alpha/FDR heatmap → {path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    print(f'Data dir : {args.data_dir}')
    print(f'GT dir   : {args.gt_dir}')
    print(f'Batches  : {args.batches}')
    print(f'Timeout  : {args.timeout} h per DRASL call')
    print(f'Output   : {args.output_dir}')
    print(f'selfloop : {args.selfloop}\n')

    t_start = time.time()
    all_results, meta = run_experiment(args)
    rows = aggregate(all_results, meta)
    print_summary(rows)

    # --- Save ---
    os.makedirs(args.output_dir, exist_ok=True)

    json_path = os.path.join(args.output_dir, 'exp4_saved_results.json')
    with open(json_path, 'w') as fh:
        json.dump({'all_results': all_results, 'meta': meta, 'rows': rows},
                  fh, indent=2)
    print(f'\nFull results → {json_path}')

    csv_path = os.path.join(args.output_dir, 'exp4_saved_summary.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'Summary CSV → {csv_path}')

    # --- Plots ---
    plot_comparison(rows,
        os.path.join(args.output_dir, 'exp4_saved_comparison.pdf'))
    plot_pcmci_vs_drasl(rows,
        os.path.join(args.output_dir, 'exp4_saved_scatter.pdf'))
    plot_tau_sweep(rows,
        os.path.join(args.output_dir, 'exp4_saved_tau_sweep.pdf'))
    plot_alpha_fdr_heatmap(rows,
        os.path.join(args.output_dir, 'exp4_saved_heatmap.pdf'))

    total = time.time() - t_start
    print(f'\nTotal elapsed: {total/60:.1f} min')


if __name__ == '__main__':
    main()
