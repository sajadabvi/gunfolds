"""
Edge-Level Frequency vs. Correctness Calibration — Ruben Simulation Data

Same calibration analysis as edge_frequency_calibration.py but using Ruben's
simulated fMRI data (from ~/DataSets_Feedbacks/1. Simple_Networks/) instead of
generating synthetic data on the fly.

For each data file with known ground truth, this script:
1. Loads the pre-existing BOLD time-series from Ruben's simulations
2. Runs PCMCI to get an initial graph estimate
3. Runs DRASL (with self-loops enabled) to get the solution set
4. Computes per-edge frequency across solutions
5. Bins edges by frequency and computes precision per bin
6. Produces a calibration curve: edge agreement frequency vs. correctness

Modes:
  run        - Execute a single task identified by --task_id (for SLURM array jobs)
  aggregate  - Combine individual task results and produce calibration plots
  local      - Run all tasks sequentially in a single process
  plot_only  - Re-plot from saved aggregated results

By default, self-loops are excluded from the precision calculation.
Use --include-selfloops to include them.
"""
import os
import sys
import json
import glob
import numpy as np
import argparse
from collections import defaultdict
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from gunfolds.utils import bfutils
from gunfolds.utils import zickle as zkl
from gunfolds.utils import graphkit as gk
from gunfolds.solvers.clingo_rasl import drasl
from gunfolds.utils.calc_procs import get_process_count
from gunfolds import conversions as cv
from gunfolds.scripts.datasets.simple_networks import simp_nets

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

CLINGO_LIMIT = 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))


def build_task_list(networks, batches):
    """Build an ordered list of (network, batch) tuples.

    The SLURM array task ID indexes into this list.
    """
    tasks = []
    for net_num in networks:
        for batch in range(1, batches + 1):
            tasks.append((net_num, batch))
    return tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description='Edge frequency calibration using Ruben simulation data '
                    '(SLURM array compatible).')

    sub = parser.add_subparsers(dest='mode', help='Execution mode')

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("-n", "--NETWORKS", nargs='+', default=[1, 2, 3, 5],
                        type=int, help="Sanchez-Romero network numbers to use")
    shared.add_argument("-b", "--BATCHES", default=60, type=int,
                        help="number of data files per network")
    shared.add_argument("--data_root",
                        default=os.path.expanduser(
                            "~/DataSets_Feedbacks/1. Simple_Networks"),
                        help="root directory for Ruben simulation data")
    shared.add_argument("--concat", action="store_true", default=True,
                        help="use concatenated data files (default)")
    shared.add_argument("--no-concat", dest="concat", action="store_false",
                        help="use individual (non-concatenated) data files")
    shared.add_argument("--tau_max", default=1, type=int,
                        help="maximum lag for PCMCI")
    shared.add_argument("--maxu", default=4, type=int,
                        help="maximum undersampling rate to search")
    shared.add_argument("--output_dir",
                        default="results_edge_calibration_ruben",
                        help="output directory")
    shared.add_argument("--timestamp", default=None,
                        help="shared timestamp for grouping results")
    shared.add_argument("--include-selfloops", action="store_true",
                        default=False,
                        help="include self-loops in precision calculation "
                             "(ignored by default)")

    # --- 'run' mode ---
    run_parser = sub.add_parser('run', parents=[shared],
                                help='Run a single task (SLURM array element)')
    run_parser.add_argument("--task_id", required=True, type=int,
                            help="SLURM_ARRAY_TASK_ID (0-based)")
    run_parser.add_argument("-p", "--PNUM", default=PNUM, type=int,
                            help="number of CPUs for clingo")
    run_parser.add_argument("-t", "--TIMEOUT", default=2, type=int,
                            help="timeout in hours (0 = no timeout)")

    # --- 'aggregate' mode ---
    sub.add_parser('aggregate', parents=[shared],
                   help='Aggregate task results and produce plots')

    # --- 'local' mode ---
    local_parser = sub.add_parser('local', parents=[shared],
                                  help='Run all tasks sequentially')
    local_parser.add_argument("-p", "--PNUM", default=PNUM, type=int,
                              help="number of CPUs for clingo")
    local_parser.add_argument("-t", "--TIMEOUT", default=2, type=int,
                              help="timeout in hours (0 = no timeout)")

    # --- 'plot_only' mode ---
    plot_parser = sub.add_parser('plot_only',
                                 help='Re-plot from saved aggregated .zkl file')
    plot_parser.add_argument("path", help="path to saved results .zkl")

    return parser.parse_args()


# =========================================================================
# Data loading
# =========================================================================

def load_ruben_data(data_root, network_num, batch, concat=True):
    """Load a single Ruben BOLD time-series file.

    Returns a pandas DataFrame (rows=time, columns=nodes).
    """
    num = str(batch) if batch > 9 else '0' + str(batch)
    if concat:
        path = os.path.join(
            data_root,
            f'Network{network_num}_amp',
            'data_fslfilter_concat',
            f'concat_BOLDfslfilter_{num}.txt')
    else:
        path = os.path.join(
            data_root,
            f'Network{network_num}_amp',
            'data_fslfilter',
            f'BOLDfslfilter_{num}.txt')

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    data = pd.read_csv(path, delimiter='\t')
    return data


# =========================================================================
# Edge frequency analysis (identical to edge_frequency_calibration.py)
# =========================================================================

def compute_edge_frequencies(solutions, n_nodes, ignore_selfloops=True):
    """Compute frequency of each directed and bidirected edge across solutions."""
    directed_counts = defaultdict(int)
    bidirected_counts = defaultdict(int)
    n_solutions = len(solutions)

    for answer in solutions:
        g = bfutils.num2CG(answer[0][0], n_nodes)
        for node in g:
            for neighbor, edge_type in g[node].items():
                if ignore_selfloops and node == neighbor:
                    continue
                if edge_type in (1, 3):
                    directed_counts[(node, neighbor)] += 1
                if edge_type in (2, 3):
                    key = tuple(sorted((node, neighbor)))
                    bidirected_counts[key] += 1

    directed_freq = {e: c / n_solutions for e, c in directed_counts.items()}
    bidirected_freq = {e: c / n_solutions for e, c in bidirected_counts.items()}
    return directed_freq, bidirected_freq


def compute_calibration_data(solutions, gt, n_nodes, ignore_selfloops=True):
    """
    For each edge appearing in any solution, compute:
    - Its frequency across solutions
    - Whether it is a true positive (present in ground truth)
    Returns lists of (frequency, is_correct) for directed and bidirected edges.
    """
    gt_directed = set(gk.edgelist(gt))
    gt_bidirected_raw = gk.bedgelist(gt)
    gt_bidirected = set(tuple(sorted(e)) for e in gt_bidirected_raw)

    if ignore_selfloops:
        gt_directed = {(u, v) for u, v in gt_directed if u != v}

    directed_freq, bidirected_freq = compute_edge_frequencies(
        solutions, n_nodes, ignore_selfloops=ignore_selfloops)

    directed_calibration = []
    for edge, freq in directed_freq.items():
        is_correct = 1 if edge in gt_directed else 0
        directed_calibration.append((freq, is_correct))

    bidirected_calibration = []
    for edge, freq in bidirected_freq.items():
        is_correct = 1 if edge in gt_bidirected else 0
        bidirected_calibration.append((freq, is_correct))

    all_gt_directed_not_found = gt_directed - set(directed_freq.keys())
    for edge in all_gt_directed_not_found:
        directed_calibration.append((0.0, 1))

    all_gt_bidirected_not_found = gt_bidirected - set(bidirected_freq.keys())
    for edge in all_gt_bidirected_not_found:
        bidirected_calibration.append((0.0, 1))

    return directed_calibration, bidirected_calibration


# =========================================================================
# Single experiment
# =========================================================================

def run_single_experiment(network_num, batch_idx, data_root, concat,
                          tau_max, maxu, timeout_hours, pnum,
                          ignore_selfloops=True):
    """Run a single configuration on Ruben data and return calibration data."""
    GT = simp_nets(network_num, selfloop=True)
    n_nodes = len(GT)
    MAXCOST = 50

    try:
        data = load_ruben_data(data_root, network_num, batch_idx,
                               concat=concat)
    except FileNotFoundError as e:
        print(f"  Skipping net={network_num}, batch={batch_idx}: {e}")
        return None

    dataframe = pp.DataFrame(data.values)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=None)
    g_estimated, A_mat, B_mat = cv.Glag2CG(results)

    DD = (np.abs((np.abs(A_mat / np.abs(A_mat).max()) +
                  (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
    BD = (np.abs((np.abs(B_mat / np.abs(B_mat).max()) +
                  (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

    timeout_sec = 60 * 60 * timeout_hours if timeout_hours > 0 else 0
    priorities = [1, 2, 1, 2, 3]
    r = drasl([g_estimated], weighted=True, capsize=0,
              timeout=timeout_sec,
              urate=min(maxu, 3 * n_nodes + 1),
              dm=[DD], bdm=[BD],
              GT_density=int(100 * gk.density(GT)),
              edge_weights=priorities, pnum=pnum, optim='optN',
              selfloop=True)

    if len(r) == 0:
        print(f"  No solutions for net={network_num}, batch={batch_idx}")
        return None

    r_sorted = sorted(r, key=lambda sol: sol[1])
    n_keep = max(1, len(r_sorted) // 5)
    r = r_sorted[:n_keep]

    dir_cal, bidir_cal = compute_calibration_data(
        r, GT, n_nodes, ignore_selfloops=ignore_selfloops)

    return {
        'directed_calibration': dir_cal,
        'bidirected_calibration': bidir_cal,
        'n_solutions': len(r),
        'network': network_num,
        'batch': batch_idx,
    }


# =========================================================================
# Calibration binning & plotting
# =========================================================================

def bin_calibration_data(all_calibration_points, n_bins=10):
    """Bin calibration points by frequency and compute precision per bin."""
    if not all_calibration_points:
        return [], [], [], []

    freqs = np.array([p[0] for p in all_calibration_points])
    correct = np.array([p[1] for p in all_calibration_points])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_precisions = []
    bin_counts = []
    bin_stderrs = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (freqs >= lo) & (freqs <= hi)
        else:
            mask = (freqs >= lo) & (freqs < hi)

        n_in_bin = mask.sum()
        if n_in_bin > 0:
            precision = correct[mask].mean()
            stderr = (np.sqrt(precision * (1 - precision) / n_in_bin)
                      if n_in_bin > 1 else 0)
            bin_centers.append((lo + hi) / 2)
            bin_precisions.append(precision)
            bin_counts.append(n_in_bin)
            bin_stderrs.append(stderr)

    return bin_centers, bin_precisions, bin_counts, bin_stderrs


def plot_calibration(all_directed_cal, all_bidirected_cal, output_path):
    """Produce calibration plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title in [
        (axes[0], all_directed_cal, 'Directed Edges'),
        (axes[1], all_directed_cal + all_bidirected_cal,
         'All Edges (Directed + Bidirected)')
    ]:
        centers, precisions, counts, stderrs = bin_calibration_data(
            data, n_bins=10)
        if not centers:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        ax.bar(centers, precisions, width=0.08, alpha=0.6,
               color='steelblue', edgecolor='black', linewidth=0.5)
        ax.errorbar(centers, precisions, yerr=stderrs, fmt='o',
                    color='darkblue', markersize=5, capsize=3)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3,
                label='Perfect calibration')
        ax.set_xlabel('Edge Agreement Frequency in Solution Set',
                      fontsize=12)
        ax.set_ylabel('Precision (Fraction Correct)', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=10)

        for c, p, n in zip(centers, precisions, counts):
            ax.annotate(f'n={n}', (c, p), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=7,
                        color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Calibration plot saved to {output_path}")


def print_summary(directed_cal, bidirected_cal):
    """Print a summary table of the calibration results."""
    all_cal = directed_cal + bidirected_cal
    if not all_cal:
        print("No calibration data collected.")
        return

    centers, precisions, counts, _ = bin_calibration_data(all_cal, n_bins=5)
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY (All Edges) — Ruben Data")
    print("=" * 60)
    print(f"{'Frequency Bin':>15} {'Precision':>12} {'N Edges':>10}")
    print("-" * 40)
    for c, p, n in zip(centers, precisions, counts):
        lo = max(0, c - 0.1)
        hi = min(1, c + 0.1)
        print(f"  {lo:.0%} - {hi:.0%}       {p:.3f}        {n}")
    print("=" * 60)


# =========================================================================
# Mode: run (single SLURM array task)
# =========================================================================

def mode_run(args):
    tasks = build_task_list(args.NETWORKS, args.BATCHES)
    total_tasks = len(tasks)

    if args.task_id < 0 or args.task_id >= total_tasks:
        print(f"ERROR: task_id {args.task_id} out of range "
              f"[0, {total_tasks - 1}]")
        sys.exit(1)

    net_num, batch = tasks[args.task_id]
    print(f"[Task {args.task_id}/{total_tasks - 1}] "
          f"Network={net_num}, batch={batch}")

    ignore_sl = not args.include_selfloops
    result = run_single_experiment(
        net_num, batch, args.data_root, args.concat,
        args.tau_max, args.maxu, args.TIMEOUT, args.PNUM,
        ignore_selfloops=ignore_sl)

    ts = args.timestamp or "notimestamp"
    out_dir = os.path.join(args.output_dir, ts, "tasks")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"task_{args.task_id:04d}.zkl")
    save_payload = {
        'task_id': args.task_id,
        'network': net_num,
        'batch': batch,
        'result': result,
    }
    zkl.save(save_payload, out_path)
    print(f"Task result saved to {out_path}")

    if result is not None:
        print(f"  Solutions: {result['n_solutions']}, "
              f"dir edges: {len(result['directed_calibration'])}, "
              f"bidir edges: {len(result['bidirected_calibration'])}")
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
    tasks = build_task_list(args.NETWORKS, args.BATCHES)
    total_expected = len(tasks)

    print(f"Found {len(task_files)} / {total_expected} task result files "
          f"in {task_dir}")

    all_directed_cal = []
    all_bidirected_cal = []
    all_results = []
    n_skipped = 0

    for fpath in task_files:
        payload = zkl.load(fpath)
        result = payload['result']
        if result is not None:
            all_directed_cal.extend(result['directed_calibration'])
            all_bidirected_cal.extend(result['bidirected_calibration'])
            all_results.append(result)
        else:
            n_skipped += 1

    print(f"Loaded {len(all_results)} successful results, "
          f"{n_skipped} skipped/empty")

    agg_dir = os.path.join(args.output_dir, ts)
    save_data = {
        'all_directed_cal': all_directed_cal,
        'all_bidirected_cal': all_bidirected_cal,
        'all_results': all_results,
    }
    save_path = os.path.join(agg_dir, 'edge_calibration_results_ruben.zkl')
    zkl.save(save_data, save_path)
    print(f"Aggregated results saved to {save_path}")

    plot_path = os.path.join(agg_dir, 'edge_frequency_calibration_ruben.pdf')
    plot_calibration(all_directed_cal, all_bidirected_cal, plot_path)
    print_summary(all_directed_cal, all_bidirected_cal)


# =========================================================================
# Mode: local (sequential)
# =========================================================================

def mode_local(args):
    os.makedirs(args.output_dir, exist_ok=True)
    all_directed_cal = []
    all_bidirected_cal = []
    all_results = []

    ignore_sl = not args.include_selfloops
    tasks = build_task_list(args.NETWORKS, args.BATCHES)
    total = len(tasks)

    for idx, (net_num, batch) in enumerate(tasks):
        print(f"[{idx + 1}/{total}] Network={net_num}, batch={batch}")
        result = run_single_experiment(
            net_num, batch, args.data_root, args.concat,
            args.tau_max, args.maxu, args.TIMEOUT, args.PNUM,
            ignore_selfloops=ignore_sl)
        if result is not None:
            all_directed_cal.extend(result['directed_calibration'])
            all_bidirected_cal.extend(result['bidirected_calibration'])
            all_results.append(result)
            print(f"  Solutions: {result['n_solutions']}, "
                  f"dir edges tracked: "
                  f"{len(result['directed_calibration'])}, "
                  f"bidir edges tracked: "
                  f"{len(result['bidirected_calibration'])}")

    save_data = {
        'all_directed_cal': all_directed_cal,
        'all_bidirected_cal': all_bidirected_cal,
        'all_results': all_results,
    }
    save_path = os.path.join(args.output_dir,
                             'edge_calibration_results_ruben.zkl')
    zkl.save(save_data, save_path)
    print(f"\nResults saved to {save_path}")

    plot_path = os.path.join(args.output_dir,
                             'edge_frequency_calibration_ruben.pdf')
    plot_calibration(all_directed_cal, all_bidirected_cal, plot_path)
    print_summary(all_directed_cal, all_bidirected_cal)


# =========================================================================
# Mode: plot_only
# =========================================================================

def mode_plot_only(args):
    print(f"Loading saved results from {args.path}")
    saved = zkl.load(args.path)
    plot_calibration(saved['all_directed_cal'], saved['all_bidirected_cal'],
                     args.path.replace('.zkl', '_calibration.pdf'))
    print_summary(saved['all_directed_cal'], saved['all_bidirected_cal'])


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()

    if args.mode is None:
        print("ERROR: specify a mode: run | aggregate | local | plot_only")
        print("  run        - single SLURM array task (use with --task_id)")
        print("  aggregate  - combine task results and produce plots")
        print("  local      - run all tasks sequentially")
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
