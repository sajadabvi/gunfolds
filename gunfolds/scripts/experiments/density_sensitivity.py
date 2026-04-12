"""
Density Sensitivity Analysis

Tests how RnR performance degrades when the assumed density deviates from
the true ground-truth density by -30%, -10%, 0% (correct), +10%, +30%.

This addresses Reviewer rp9v Q2 and Reviewer cuMG's concern: "How sensitive
is RnR to misspecification of the target density d?"

Modes:
  run        - Execute a single task identified by --task_id (for SLURM array jobs)
  aggregate  - Combine individual task results and produce summary + plots
  local      - Run all tasks sequentially in a single process (original behavior)
  plot_only  - Re-plot from saved aggregated .zkl file
"""
import os
import sys
import glob
import numpy as np
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gunfolds.utils import bfutils
from gunfolds.utils import zickle as zkl
from gunfolds.utils import graphkit as gk
from gunfolds.solvers.clingo_rasl import drasl
from gunfolds.utils.calc_procs import get_process_count
from gunfolds import conversions as cv

import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import tigramite.data_processing as pp

CLINGO_LIMIT = 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))

DENSITY_SCALES = [0.7, 0.9, 1.0, 1.1, 1.3]


def build_task_list(node_sizes, extra_edges_list, undersampling_rates, batches):
    """Build an ordered list of (n_nodes, extra_edges, u_rate, batch) tuples.

    The SLURM array task ID indexes into this list.
    """
    tasks = []
    for n in node_sizes:
        for e in extra_edges_list:
            for u in undersampling_rates:
                for batch in range(1, batches + 1):
                    tasks.append((n, e, u, batch))
    return tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description='Density sensitivity analysis (SLURM array compatible).')

    sub = parser.add_subparsers(dest='mode', help='Execution mode')

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("-n", "--NODE_SIZES", nargs='+', default=[5, 6],
                        type=int, help="graph sizes (number of nodes)")
    shared.add_argument("-e", "--EXTRA_EDGES", nargs='+', default=[1, 2, 3],
                        type=int, help="number of extra edges beyond the ring")
    shared.add_argument("-u", "--UNDERSAMPLING", nargs='+', default=[2, 3],
                        type=int, help="undersampling rates")
    shared.add_argument("-b", "--BATCHES", default=10, type=int,
                        help="number of batches per configuration")
    shared.add_argument("--ssize", default=5000, type=int, help="sample size")
    shared.add_argument("--noise", default=0.1, type=float, help="noise level")
    shared.add_argument("--output_dir", default="results_density_sensitivity",
                        help="output directory")
    shared.add_argument("--timestamp", default=None,
                        help="shared timestamp for grouping results")

    run_parser = sub.add_parser('run', parents=[shared],
                                help='Run a single task (SLURM array element)')
    run_parser.add_argument("--task_id", required=True, type=int,
                            help="SLURM_ARRAY_TASK_ID (0-based index into task list)")
    run_parser.add_argument("-p", "--PNUM", default=PNUM, type=int,
                            help="number of CPUs for clingo")
    run_parser.add_argument("-t", "--TIMEOUT", default=2, type=int,
                            help="timeout in hours")

    agg_parser = sub.add_parser('aggregate', parents=[shared],
                                help='Aggregate task results and produce plots')

    local_parser = sub.add_parser('local', parents=[shared],
                                  help='Run all tasks sequentially in one process')
    local_parser.add_argument("-p", "--PNUM", default=PNUM, type=int,
                              help="number of CPUs for clingo")
    local_parser.add_argument("-t", "--TIMEOUT", default=2, type=int,
                              help="timeout in hours")

    plot_parser = sub.add_parser('plot_only',
                                 help='Re-plot from saved aggregated .zkl file')
    plot_parser.add_argument("path", help="path to saved results .zkl")

    return parser.parse_args()


# =========================================================================
# Simulation helpers
# =========================================================================

def create_stable_weighted_matrix(A, threshold=0.1, powers=[1, 2, 3, 4],
                                  max_attempts=10000000, damping_factor=0.99):
    attempts = 0
    while attempts < max_attempts:
        random_weights = np.random.randn(*A.shape)
        weighted_matrix = A * random_weights
        weighted_sparse = sp.csr_matrix(weighted_matrix)
        eigenvalues, _ = eigs(weighted_sparse, k=1, which="LM")
        max_eigenvalue = np.abs(eigenvalues[0])
        if max_eigenvalue > 0:
            weighted_matrix *= damping_factor / max_eigenvalue
            ok = True
            for n in powers:
                W_n = np.linalg.matrix_power(weighted_matrix, n)
                non_zero_indices = np.nonzero(W_n)
                if (np.abs(W_n[non_zero_indices]) < threshold).any():
                    ok = False
                    break
            if ok:
                return weighted_matrix
        attempts += 1
    raise ValueError(f"Unable to create a stable matrix after {max_attempts} attempts.")


def genData(A, rate=2, burnin=100, ssize=5000, noise=0.1):
    n = A.shape[0]
    data = np.zeros([n, burnin + (ssize * rate)])
    data[:, 0] = noise * np.random.randn(n)
    for i in range(1, data.shape[1]):
        data[:, i] = A @ data[:, i - 1] + noise * np.random.randn(n)
    data = data[:, burnin:]
    return data[:, ::rate]


# =========================================================================
# Metrics
# =========================================================================

def compute_f1(omission, commission, n_gt_edges):
    tp = n_gt_edges - omission
    fp = commission
    fn = omission
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall


def evaluate_best_solution(solutions, GT, n_nodes):
    """Find the best solution by total error and return its metrics."""
    gt_dir_edges = len(gk.edgelist(GT))
    gt_bidir_edges = len(set(tuple(sorted(e)) for e in gk.bedgelist(GT)))
    n_gt_total = gt_dir_edges + gt_bidir_edges

    best_err = float('inf')
    best_metrics = None

    for answer in solutions:
        g1 = bfutils.num2CG(answer[0][0], n_nodes)
        oce = gk.OCE(g1, GT, normalized=False)
        total_err = oce['total'][0] + oce['total'][1]

        if total_err < best_err:
            best_err = total_err
            orient_f1, _, _ = compute_f1(oce['directed'][0], oce['directed'][1], gt_dir_edges)
            total_f1, _, _ = compute_f1(oce['total'][0], oce['total'][1], n_gt_total)

            best_metrics = {
                'orientation_f1': orient_f1,
                'adjacency_f1': total_f1,
                'dir_omission': oce['directed'][0],
                'dir_commission': oce['directed'][1],
                'total_error': best_err,
                'n_solutions': len(solutions),
            }

    return best_metrics


# =========================================================================
# Single experiment (one task = one n_nodes x extra_edges x u_rate x batch)
# =========================================================================

def run_density_sweep(n_nodes, extra_edges, u_rate, batch_idx, ssize, noise,
                      timeout_hours, pnum):
    """Run the solver at multiple density scales for one data generation."""
    GT = gk.ringmore(n_nodes, extra_edges)
    print(f"  Generated ringmore graph: n={n_nodes}, extra={extra_edges}, "
          f"density={gk.density(GT):.3f}, edges={len(gk.edgelist(GT))}")
    A = cv.graph2adj(GT)
    MAXCOST = 10000
    timeout_sec = timeout_hours * 60 * 60

    try:
        W = create_stable_weighted_matrix(A, threshold=0.2, powers=[2, 3, 4])
    except ValueError:
        print(f"  Skipping: could not create stable matrix")
        return None

    dd = genData(W, rate=u_rate, ssize=ssize, noise=noise)

    dataframe = pp.DataFrame(np.transpose(dd))
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results_pcmci = pcmci.run_pcmci(tau_max=2, pc_alpha=None)
    g_estimated, A_mat, B_mat = cv.Glag2CG(results_pcmci)

    DD = (np.abs((np.abs(A_mat / np.abs(A_mat).max()) +
                  (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
    BD = (np.abs((np.abs(B_mat / np.abs(B_mat).max()) +
                  (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

    true_density = int(1000 * gk.density(GT))
    priorities = [1, 2, 1, 2, 3]

    sweep_results = {}
    for scale in DENSITY_SCALES:
        scaled_density = max(1, int(true_density * scale))
        label = f"{int((scale - 1) * 100):+d}%"
        print(f"    Density scale={scale} ({label}), "
              f"true={true_density}, assumed={scaled_density}")

        start = time.time()
        r = drasl([g_estimated], weighted=True, capsize=0,
                  timeout=timeout_sec,
                  urate=min(15, 3 * n_nodes + 1),
                  dm=[DD], bdm=[BD],
                  GT_density=scaled_density,
                  edge_weights=priorities, pnum=pnum, optim='optN',
                  selfloop=False)
        elapsed = time.time() - start

        if len(r) == 0:
            print(f"      No solutions")
            continue

        metrics = evaluate_best_solution(r, GT, n_nodes)
        if metrics:
            metrics['density_scale'] = scale
            metrics['density_label'] = label
            metrics['true_density'] = true_density
            metrics['assumed_density'] = scaled_density
            metrics['runtime_sec'] = elapsed
            sweep_results[label] = metrics
            print(f"      Orient F1={metrics['orientation_f1']:.3f}, "
                  f"Adj F1={metrics['adjacency_f1']:.3f}")

    return sweep_results


# =========================================================================
# Aggregation & reporting
# =========================================================================

def aggregate_density_results(all_results):
    """Aggregate across all runs, grouped by density scale."""
    aggregated = {}
    for scale in DENSITY_SCALES:
        label = f"{int((scale - 1) * 100):+d}%"
        metrics_list = []
        for run in all_results:
            if run and label in run:
                metrics_list.append(run[label])

        if not metrics_list:
            continue

        agg = {}
        for key in metrics_list[0]:
            if isinstance(metrics_list[0][key], (int, float)):
                vals = [m[key] for m in metrics_list]
                agg[key] = {'mean': np.mean(vals), 'std': np.std(vals), 'n': len(vals)}
        aggregated[label] = agg

    return aggregated


def print_summary(aggregated):
    print("\n" + "=" * 70)
    print("DENSITY SENSITIVITY ANALYSIS RESULTS")
    print("=" * 70)
    print(f"{'Density Offset':>15} {'Orient F1':>15} {'Adj F1':>15} {'#Solutions':>12}")
    print("-" * 60)

    for scale in DENSITY_SCALES:
        label = f"{int((scale - 1) * 100):+d}%"
        if label not in aggregated:
            continue
        a = aggregated[label]
        print(f"{label:>15} "
              f"{a['orientation_f1']['mean']:>8.3f}±{a['orientation_f1']['std']:.3f}"
              f"{a['adjacency_f1']['mean']:>10.3f}±{a['adjacency_f1']['std']:.3f}"
              f"{a['n_solutions']['mean']:>9.0f}±{a['n_solutions']['std']:.0f}")

    print("=" * 70)


def plot_density_sensitivity(aggregated, output_path):
    """Plot F1 vs. density misspecification."""
    labels_ordered = [f"{int((s - 1) * 100):+d}%" for s in DENSITY_SCALES]
    labels_present = [l for l in labels_ordered if l in aggregated]

    if not labels_present:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    offsets = [int(l.replace('%', '')) for l in labels_present]

    orient_means = [aggregated[l]['orientation_f1']['mean'] for l in labels_present]
    orient_stds = [aggregated[l]['orientation_f1']['std'] for l in labels_present]
    adj_means = [aggregated[l]['adjacency_f1']['mean'] for l in labels_present]
    adj_stds = [aggregated[l]['adjacency_f1']['std'] for l in labels_present]

    ax.errorbar(offsets, orient_means, yerr=orient_stds, marker='o', capsize=5,
                linewidth=2, markersize=8, label='Orientation F1', color='#2C3E50')
    ax.errorbar(offsets, adj_means, yerr=adj_stds, marker='s', capsize=5,
                linewidth=2, markersize=8, label='Adjacency F1', color='#E74C3C')

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='True density')
    ax.set_xlabel('Density Misspecification (%)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Sensitivity to Density Prior Misspecification', fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Density sensitivity plot saved to {output_path}")


# =========================================================================
# Mode: run (single SLURM array task)
# =========================================================================

def mode_run(args):
    tasks = build_task_list(args.NODE_SIZES, args.EXTRA_EDGES,
                            args.UNDERSAMPLING, args.BATCHES)
    total_tasks = len(tasks)

    if args.task_id < 0 or args.task_id >= total_tasks:
        print(f"ERROR: task_id {args.task_id} out of range [0, {total_tasks - 1}]")
        sys.exit(1)

    n_nodes, extra_edges, u_rate, batch = tasks[args.task_id]
    print(f"[Task {args.task_id}/{total_tasks - 1}] "
          f"n={n_nodes}, extra={extra_edges}, u={u_rate}, batch={batch}")

    result = run_density_sweep(n_nodes, extra_edges, u_rate, batch,
                               args.ssize, args.noise,
                               args.TIMEOUT, args.PNUM)

    ts = args.timestamp or "notimestamp"
    out_dir = os.path.join(args.output_dir, ts, "tasks")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"task_{args.task_id:04d}.zkl")
    save_payload = {
        'task_id': args.task_id,
        'n_nodes': n_nodes,
        'extra_edges': extra_edges,
        'u_rate': u_rate,
        'batch': batch,
        'result': result,
    }
    zkl.save(save_payload, out_path)
    print(f"Task result saved to {out_path}")

    if result is not None:
        for label, metrics in result.items():
            print(f"  {label}: Orient F1={metrics['orientation_f1']:.3f}, "
                  f"Adj F1={metrics['adjacency_f1']:.3f}")
    else:
        print("  Task returned no result (skipped or no solutions).")


# =========================================================================
# Mode: aggregate
# =========================================================================

def mode_aggregate(args):
    ts = args.timestamp or "notimestamp"
    task_dir = os.path.join(args.output_dir, ts, "tasks")

    if not os.path.isdir(task_dir):
        print(f"ERROR: task directory not found: {task_dir}")
        sys.exit(1)

    task_files = sorted(glob.glob(os.path.join(task_dir, "task_*.zkl")))
    tasks = build_task_list(args.NODE_SIZES, args.EXTRA_EDGES,
                            args.UNDERSAMPLING, args.BATCHES)
    total_expected = len(tasks)

    print(f"Found {len(task_files)} / {total_expected} task result files in {task_dir}")

    all_results = []
    n_skipped = 0

    for fpath in task_files:
        payload = zkl.load(fpath)
        result = payload['result']
        if result is not None:
            all_results.append(result)
        else:
            n_skipped += 1

    print(f"Loaded {len(all_results)} successful results, {n_skipped} skipped/empty")

    agg_dir = os.path.join(args.output_dir, ts)
    save_data = {
        'all_results': all_results,
        'args': {k: v for k, v in vars(args).items() if k != 'mode'},
        'density_scales': DENSITY_SCALES,
    }
    save_path = os.path.join(agg_dir, 'density_sensitivity_results.zkl')
    zkl.save(save_data, save_path)
    print(f"Aggregated results saved to {save_path}")

    aggregated = aggregate_density_results(all_results)
    print_summary(aggregated)
    plot_path = os.path.join(agg_dir, 'density_sensitivity.pdf')
    plot_density_sensitivity(aggregated, plot_path)


# =========================================================================
# Mode: local (original sequential behavior)
# =========================================================================

def mode_local(args):
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    tasks = build_task_list(args.NODE_SIZES, args.EXTRA_EDGES,
                            args.UNDERSAMPLING, args.BATCHES)
    total = len(tasks)

    for idx, (n_nodes, extra_edges, u_rate, batch) in enumerate(tasks):
        print(f"\n[{idx + 1}/{total}] n={n_nodes}, extra={extra_edges}, "
              f"u={u_rate}, batch={batch}")
        result = run_density_sweep(n_nodes, extra_edges, u_rate, batch,
                                   args.ssize, args.noise,
                                   args.TIMEOUT, args.PNUM)
        all_results.append(result)

    save_data = {
        'all_results': all_results,
        'args': {k: v for k, v in vars(args).items() if k != 'mode'},
        'density_scales': DENSITY_SCALES,
    }
    save_path = os.path.join(args.output_dir, 'density_sensitivity_results.zkl')
    zkl.save(save_data, save_path)
    print(f"\nResults saved to {save_path}")

    aggregated = aggregate_density_results(all_results)
    print_summary(aggregated)
    plot_path = os.path.join(args.output_dir, 'density_sensitivity.pdf')
    plot_density_sensitivity(aggregated, plot_path)


# =========================================================================
# Mode: plot_only
# =========================================================================

def mode_plot_only(args):
    print(f"Loading saved results from {args.path}")
    saved = zkl.load(args.path)
    aggregated = aggregate_density_results(saved['all_results'])
    print_summary(aggregated)
    plot_density_sensitivity(
        aggregated,
        os.path.join(os.path.dirname(os.path.abspath(args.path)),
                     'density_sensitivity.pdf'))


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()

    if args.mode is None:
        print("ERROR: specify a mode: run | aggregate | local | plot_only")
        print("  run        - single SLURM array task (use with --task_id)")
        print("  aggregate  - combine task results and produce plots")
        print("  local      - run all tasks sequentially (original behavior)")
        print("  plot_only  - re-plot from saved .zkl file")
        sys.exit(1)

    dispatch = {
        'run': mode_run,
        'aggregate': mode_aggregate,
        'local': mode_local,
        'plot_only': mode_plot_only,
    }
    dispatch[args.mode](args)


if __name__ == '__main__':
    main()
