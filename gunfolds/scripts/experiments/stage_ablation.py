"""
Stage-by-Stage Ablation Experiment

Demonstrates that each lexicographic optimization stage performs its intended role
by running the same input through three configurations:

  Stage 1: p=[0,0,0,0,1]  -- Density optimization only
  Stage 2: p=[0,1,0,1,2]  -- Density (priority 2) + bidirected matching (priority 1)
  Stage 3: p=[1,2,1,2,3]  -- Full pipeline: density + bidirected + directed

Uses ringmore graphs (ring + random extra edges) of configurable size and density.

For each stage, records: solution set size, orientation F1, adjacency F1,
density deviation, and omission/commission errors.

This addresses Reviewer jQK7 Q2: "What does the solution set look like after
stage 1 alone, and how does it change through stages 2 and 3?"

Modes:
  run        - Execute a single task identified by --task_id (for SLURM array jobs)
  aggregate  - Combine individual task results and produce summary + plots
  local      - Run all tasks sequentially in a single process (original behavior)
  plot_only  - Re-plot from saved aggregated results
"""
import os
import sys
import glob
import copy
import numpy as np
import argparse
import time
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gunfolds.utils import bfutils
from gunfolds.utils import zickle as zkl
from gunfolds.utils import graphkit as gk
from gunfolds.solvers.clingo_rasl import drasl
from gunfolds.utils.calc_procs import get_process_count
from gunfolds.estimation import linear_model as lm
from gunfolds import conversions as cv

import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import tigramite.data_processing as pp

CLINGO_LIMIT = 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))

STAGES = {
    'Stage 1: Density only':       {'priorities': [0, 0, 0, 0, 1], 'tercile': 'bottom'},
    'Stage 2: Density+Bidirected': {'priorities': [0, 1, 0, 1, 2], 'tercile': 'middle'},
    'Stage 3: Full pipeline':      {'priorities': [1, 2, 1, 2, 3], 'tercile': 'top'},
}


def build_task_list(node_sizes, densities, undersampling_rates, batches):
    """Build an ordered list of (n_nodes, density, u_rate, batch) tuples.

    The SLURM array task ID indexes into this list.
    """
    tasks = []
    for n in node_sizes:
        for d in densities:
            for u in undersampling_rates:
                for batch in range(1, batches + 1):
                    tasks.append((n, d, u, batch))
    return tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description='Stage-by-stage ablation experiment (SLURM array compatible).')

    sub = parser.add_subparsers(dest='mode', help='Execution mode')

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("-n", "--NODE_SIZES", nargs='+', default=[4, 5],
                        type=int, help="graph sizes (number of nodes)")
    shared.add_argument("-d", "--DENSITIES", nargs='+', default=[0.3, 0.2],
                        type=float, help="target directed-edge densities")
    shared.add_argument("-u", "--UNDERSAMPLING", nargs='+', default=[2, 3],
                        type=int, help="undersampling rates")
    shared.add_argument("-b", "--BATCHES", default=10, type=int,
                        help="number of batches per configuration")
    shared.add_argument("--ssize", default=5000, type=int, help="sample size")
    shared.add_argument("--noise", default=0.1, type=float, help="noise level")
    shared.add_argument("--output_dir", default="results_stage_ablation",
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
# Graph generation
# =========================================================================

def generate_ringmore_graph(n_nodes, target_density):
    """Generate a ring graph with extra random directed edges.

    The ring itself contributes n directed edges (density = n / n^2 = 1/n).
    We add enough random edges to approximate ``target_density``.

    Returns a gunfolds graph dictionary.
    """
    ring_edges = n_nodes
    target_edges = int(round(target_density * n_nodes * n_nodes))
    extra = max(0, target_edges - ring_edges)
    g = gk.ringmore(n_nodes, extra)
    return g


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


def Glag2CG(results):
    graph_array = results['graph']
    bidirected_edges = np.where(graph_array == 'o-o', 1, 0).astype(int)
    directed_edges = np.where(graph_array == '-->', 1, 0).astype(int)
    graph_dict = cv.adjs2graph(directed_edges[:, :, 1], bidirected_edges[:, :, 0])
    A_matrix = results['val_matrix'][:, :, 1]
    B_matrix = results['val_matrix'][:, :, 0]
    return graph_dict, A_matrix, B_matrix


# =========================================================================
# Metrics
# =========================================================================

def compute_f1(omission, commission, n_gt_edges, n_possible_edges):
    """Compute F1 from omission/commission counts."""
    tp = n_gt_edges - omission
    fp = commission
    fn = omission
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall


def evaluate_solution_set(solutions, GT, n_nodes, tercile='top'):
    """Compute OCE for all solutions, select a tercile, return averaged metrics.

    Solutions are sorted by total OCE error (ascending = best first).
      tercile='bottom' -> worst 1/3   (Stage 1: density only)
      tercile='middle' -> middle 1/3  (Stage 2: density + bidirected)
      tercile='top'    -> best 1/3    (Stage 3: full pipeline)
    """
    n_gt_dir = len(set(gk.edgelist(GT)))
    n_gt_bidir = len(set(tuple(sorted(e)) for e in gk.bedgelist(GT)))
    n_possible_dir = n_nodes * n_nodes
    n_possible_bidir = n_nodes * (n_nodes - 1) // 2
    density_gt = gk.density(GT)

    scored = []
    for answer in solutions:
        g1 = bfutils.num2CG(answer[0][0], n_nodes)
        oce = gk.OCE(g1, GT, normalized=False)
        total_err = oce['total'][0] + oce['total'][1]
        scored.append((total_err, g1, oce))

    scored.sort(key=lambda x: x[0])
    n = len(scored)
    t1 = max(1, n // 3)
    t2 = max(t1 + 1, 2 * n // 3)

    if tercile == 'top':
        selected = scored[:t1]
    elif tercile == 'middle':
        selected = scored[t1:t2]
    else:
        selected = scored[t2:]

    if not selected:
        selected = scored

    metrics_accum = defaultdict(list)
    for total_err, g1, oce in selected:
        dir_omission = oce['directed'][0]
        dir_commission = oce['directed'][1]
        bidir_omission = oce['bidirected'][0]
        bidir_commission = oce['bidirected'][1]

        orient_f1, orient_p, orient_r = compute_f1(
            dir_omission, dir_commission, n_gt_dir, n_possible_dir)
        adj_f1, adj_p, adj_r = compute_f1(
            oce['total'][0], oce['total'][1],
            n_gt_dir + n_gt_bidir, n_possible_dir + n_possible_bidir)

        density_g1 = gk.density(g1)

        metrics_accum['orientation_f1'].append(orient_f1)
        metrics_accum['orientation_precision'].append(orient_p)
        metrics_accum['orientation_recall'].append(orient_r)
        metrics_accum['adjacency_f1'].append(adj_f1)
        metrics_accum['adjacency_precision'].append(adj_p)
        metrics_accum['adjacency_recall'].append(adj_r)
        metrics_accum['dir_omission'].append(dir_omission)
        metrics_accum['dir_commission'].append(dir_commission)
        metrics_accum['bidir_omission'].append(bidir_omission)
        metrics_accum['bidir_commission'].append(bidir_commission)
        metrics_accum['density_deviation'].append(abs(density_g1 - density_gt))
        metrics_accum['density_solution'].append(density_g1)
        metrics_accum['density_gt'].append(density_gt)
        metrics_accum['total_error'].append(total_err)

    return {key: np.mean(vals) for key, vals in metrics_accum.items()}


def run_single_config(GT, g_estimated, DD, BD, priorities, n_nodes,
                      timeout_sec, pnum, tercile='top'):
    """Run drasl with specific priorities and return metrics."""
    start = time.time()
    r = drasl([g_estimated], weighted=True, capsize=0,
              timeout=timeout_sec,
              urate=min(4, 3 * n_nodes + 1),
              dm=[DD], bdm=[BD],
              GT_density=int(1000 * gk.density(GT)),
              edge_weights=priorities, pnum=pnum, optim='optN', selfloop=False)
    elapsed = time.time() - start

    if len(r) == 0:
        return None

    metrics = evaluate_solution_set(r, GT, n_nodes, tercile=tercile)
    if metrics is None:
        return None

    metrics['n_solutions'] = len(r)
    metrics['runtime_sec'] = elapsed
    return metrics


# =========================================================================
# Single experiment (one task = one n_nodes × density × u_rate × batch)
# =========================================================================

def run_ablation(n_nodes, target_density, u_rate, batch_idx, ssize, noise,
                 timeout_hours, pnum):
    """Run all three stages for one configuration."""
    GT = generate_ringmore_graph(n_nodes, target_density)
    actual_density = gk.density(GT)
    print(f"  Generated ringmore graph: n={n_nodes}, target_d={target_density}, "
          f"actual_d={actual_density:.3f}, edges={len(gk.edgelist(GT))}")

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
    g_estimated, A_mat, B_mat = Glag2CG(results_pcmci)

    DD = (np.abs((np.abs(A_mat / np.abs(A_mat).max()) +
                  (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
    BD = (np.abs((np.abs(B_mat / np.abs(B_mat).max()) +
                  (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

    stage_results = {}
    for stage_name, cfg in STAGES.items():
        priorities = cfg['priorities']
        tercile = cfg['tercile']
        print(f"    Running {stage_name} with priorities={priorities}")
        metrics = run_single_config(GT, g_estimated, DD, BD, priorities,
                                    n_nodes, timeout_sec, pnum, tercile=tercile)
        if metrics is not None:
            stage_results[stage_name] = metrics
            print(f"      Orient F1={metrics['orientation_f1']:.3f}, "
                  f"Adj F1={metrics['adjacency_f1']:.3f}, "
                  f"Solutions={metrics['n_solutions']}, "
                  f"Density dev={metrics['density_deviation']:.3f}, "
                  f"Time={metrics['runtime_sec']:.1f}s")
        else:
            print(f"      No solutions found")

    return stage_results


# =========================================================================
# Aggregation & reporting
# =========================================================================

def aggregate_results(all_results):
    """Aggregate metrics across all runs, grouped by stage."""
    aggregated = {}
    for stage_name in STAGES:
        metrics_list = []
        for run_result in all_results:
            if run_result and stage_name in run_result:
                metrics_list.append(run_result[stage_name])

        if not metrics_list:
            continue

        agg = {}
        for key in metrics_list[0]:
            vals = [m[key] for m in metrics_list]
            agg[key] = {
                'mean': np.mean(vals),
                'std': np.std(vals),
                'median': np.median(vals),
                'n': len(vals),
            }
        aggregated[stage_name] = agg

    return aggregated


def print_summary_table(aggregated):
    """Print a formatted summary table."""
    print("\n" + "=" * 90)
    print("STAGE-BY-STAGE ABLATION RESULTS")
    print("=" * 90)

    header = f"{'Stage':<30} {'Orient F1':>10} {'Adj F1':>10} {'Dens Dev':>10} {'#Solutions':>10} {'Time(s)':>10}"
    print(header)
    print("-" * 90)

    for stage_name in STAGES:
        if stage_name not in aggregated:
            continue
        a = aggregated[stage_name]
        print(f"{stage_name:<30} "
              f"{a['orientation_f1']['mean']:>8.3f}±{a['orientation_f1']['std']:.3f}"
              f"{a['adjacency_f1']['mean']:>8.3f}±{a['adjacency_f1']['std']:.3f}"
              f"{a['density_deviation']['mean']:>8.3f}±{a['density_deviation']['std']:.3f}"
              f"{a['n_solutions']['mean']:>8.0f}±{a['n_solutions']['std']:.0f}"
              f"{a['runtime_sec']['mean']:>8.1f}±{a['runtime_sec']['std']:.1f}")

    print("=" * 90)

    print("\nDetailed breakdown:")
    print("-" * 90)
    header2 = (f"{'Stage':<30} {'Dir Omm':>9} {'Dir Com':>9} "
               f"{'Bid Omm':>9} {'Bid Com':>9} {'Dens Sol':>9}")
    print(header2)
    print("-" * 90)
    for stage_name in STAGES:
        if stage_name not in aggregated:
            continue
        a = aggregated[stage_name]
        print(f"{stage_name:<30} "
              f"{a['dir_omission']['mean']:>7.2f}±{a['dir_omission']['std']:.2f}"
              f"{a['dir_commission']['mean']:>7.2f}±{a['dir_commission']['std']:.2f}"
              f"{a['bidir_omission']['mean']:>7.2f}±{a['bidir_omission']['std']:.2f}"
              f"{a['bidir_commission']['mean']:>7.2f}±{a['bidir_commission']['std']:.2f}"
              f"{a['density_solution']['mean']:>7.3f}±{a['density_solution']['std']:.3f}")
    print("=" * 90)


def _bar_chart(ax, stage_names, short_names, aggregated, metric_key,
               ylabel, title, fmt='.3f', ylim=None):
    """Helper to draw a single bar chart panel."""
    colors = ['#4ECDC4', '#45B7D1', '#2C3E50']
    means = [aggregated[s][metric_key]['mean'] for s in stage_names]
    stds = [aggregated[s][metric_key]['std'] for s in stage_names]
    bars = ax.bar(range(len(stage_names)), means, yerr=stds,
                  capsize=5, color=colors[:len(stage_names)],
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(stage_names)))
    ax.set_xticklabels(short_names[:len(stage_names)], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    if ylim is not None:
        ax.set_ylim(ylim)
    for bar, m in zip(bars, means):
        label = f'{m:{fmt}}'
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02 * ax.get_ylim()[1],
                label, ha='center', va='bottom', fontsize=9)


def plot_ablation(aggregated, output_dir):
    """Produce three PDF figures: metrics, runtime, and solution count."""
    stage_names = [s for s in STAGES if s in aggregated]
    if not stage_names:
        print("No data to plot.")
        return

    short_names = ['Density\nOnly', 'Density+\nBidirected', 'Full\nPipeline']

    # --- Figure 1: F1 metrics + density deviation ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    _bar_chart(axes1[0], stage_names, short_names, aggregated,
               'orientation_f1', 'Orientation F1', 'Orientation F1', ylim=(0, 1.05))
    _bar_chart(axes1[1], stage_names, short_names, aggregated,
               'adjacency_f1', 'Adjacency F1', 'Adjacency F1', ylim=(0, 1.05))
    _bar_chart(axes1[2], stage_names, short_names, aggregated,
               'density_deviation', 'Density Deviation', 'Density Deviation from GT')
    fig1.tight_layout()
    path1 = os.path.join(output_dir, 'stage_ablation_metrics.pdf')
    fig1.savefig(path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Metrics plot saved to {path1}")

    # --- Figure 2: Runtime per stage ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    _bar_chart(ax2, stage_names, short_names, aggregated,
               'runtime_sec', 'Runtime (seconds)', 'Compute Time per Stage', fmt='.1f')
    fig2.tight_layout()
    path2 = os.path.join(output_dir, 'stage_ablation_runtime.pdf')
    fig2.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Runtime plot saved to {path2}")

    # --- Figure 3: Solution set size per stage ---
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    _bar_chart(ax3, stage_names, short_names, aggregated,
               'n_solutions', 'Number of Solutions', 'Solution Set Size per Stage', fmt='.0f')
    fig3.tight_layout()
    path3 = os.path.join(output_dir, 'stage_ablation_solutions.pdf')
    fig3.savefig(path3, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"Solutions plot saved to {path3}")


# =========================================================================
# Mode: run (single SLURM array task)
# =========================================================================

def mode_run(args):
    tasks = build_task_list(args.NODE_SIZES, args.DENSITIES,
                            args.UNDERSAMPLING, args.BATCHES)
    total_tasks = len(tasks)

    if args.task_id < 0 or args.task_id >= total_tasks:
        print(f"ERROR: task_id {args.task_id} out of range [0, {total_tasks - 1}]")
        sys.exit(1)

    n_nodes, density, u_rate, batch = tasks[args.task_id]
    print(f"[Task {args.task_id}/{total_tasks - 1}] "
          f"n={n_nodes}, d={density}, u={u_rate}, batch={batch}")

    result = run_ablation(n_nodes, density, u_rate, batch,
                          args.ssize, args.noise,
                          args.TIMEOUT, args.PNUM)

    ts = args.timestamp or "notimestamp"
    out_dir = os.path.join(args.output_dir, ts, "tasks")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"task_{args.task_id:04d}.zkl")
    save_payload = {
        'task_id': args.task_id,
        'n_nodes': n_nodes,
        'density': density,
        'u_rate': u_rate,
        'batch': batch,
        'result': result,
    }
    zkl.save(save_payload, out_path)
    print(f"Task result saved to {out_path}")

    if result is not None:
        for stage_name, metrics in result.items():
            print(f"  {stage_name}: Orient F1={metrics['orientation_f1']:.3f}, "
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
    tasks = build_task_list(args.NODE_SIZES, args.DENSITIES,
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
        'stages': {k: v for k, v in STAGES.items()},
    }
    save_path = os.path.join(agg_dir, 'stage_ablation_results.zkl')
    zkl.save(save_data, save_path)
    print(f"Aggregated results saved to {save_path}")

    aggregated = aggregate_results(all_results)
    print_summary_table(aggregated)
    plot_ablation(aggregated, agg_dir)


# =========================================================================
# Mode: local (original sequential behavior)
# =========================================================================

def mode_local(args):
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    tasks = build_task_list(args.NODE_SIZES, args.DENSITIES,
                            args.UNDERSAMPLING, args.BATCHES)
    total = len(tasks)

    for idx, (n_nodes, density, u_rate, batch) in enumerate(tasks):
        print(f"\n[{idx + 1}/{total}] n={n_nodes}, d={density}, "
              f"u={u_rate}, batch={batch}")
        result = run_ablation(n_nodes, density, u_rate, batch,
                              args.ssize, args.noise,
                              args.TIMEOUT, args.PNUM)
        all_results.append(result)

    save_data = {
        'all_results': all_results,
        'args': {k: v for k, v in vars(args).items() if k != 'mode'},
        'stages': {k: v for k, v in STAGES.items()},
    }
    save_path = os.path.join(args.output_dir, 'stage_ablation_results.zkl')
    zkl.save(save_data, save_path)
    print(f"\nResults saved to {save_path}")

    aggregated = aggregate_results(all_results)
    print_summary_table(aggregated)
    plot_ablation(aggregated, args.output_dir)


# =========================================================================
# Mode: plot_only
# =========================================================================

def mode_plot_only(args):
    print(f"Loading saved results from {args.path}")
    saved = zkl.load(args.path)
    aggregated = aggregate_results(saved['all_results'])
    print_summary_table(aggregated)
    plot_ablation(aggregated, os.path.dirname(os.path.abspath(args.path)))


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
