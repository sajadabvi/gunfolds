"""
Stage-by-Stage Ablation Experiment

Demonstrates that each lexicographic optimization stage performs its intended role
by running the same input through three configurations:

  Stage 1: p=[0,0,0,0,1]  -- Density optimization only
  Stage 2: p=[0,1,0,1,2]  -- Density (priority 2) + bidirected matching (priority 1)
  Stage 3: p=[1,2,1,2,3]  -- Full pipeline: density + bidirected + directed

For each stage, records: solution set size, orientation F1, adjacency F1,
density deviation, and omission/commission errors.

This addresses Reviewer jQK7 Q2: "What does the solution set look like after
stage 1 alone, and how does it change through stages 2 and 3?"
"""
import os
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
from gunfolds.scripts.datasets.simple_networks import simp_nets

import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import tigramite.data_processing as pp

CLINGO_LIMIT = 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))

STAGES = {
    'Stage 1: Density only':       [0, 0, 0, 0, 1],
    'Stage 2: Density+Bidirected': [0, 1, 0, 1, 2],
    'Stage 3: Full pipeline':      [1, 2, 1, 2, 3],
}

parser = argparse.ArgumentParser(description='Stage-by-stage ablation experiment.')
parser.add_argument("-n", "--NETWORKS", nargs='+', default=[1, 2, 3, 4, 5],
                    type=int, help="Sanchez-Romero network numbers")
parser.add_argument("-u", "--UNDERSAMPLING", nargs='+', default=[2, 3],
                    type=int, help="undersampling rates")
parser.add_argument("-b", "--BATCHES", default=10, type=int,
                    help="number of batches per configuration")
parser.add_argument("-p", "--PNUM", default=PNUM, type=int, help="number of CPUs")
parser.add_argument("-t", "--TIMEOUT", default=2, type=int, help="timeout in hours")
parser.add_argument("--ssize", default=5000, type=int, help="sample size")
parser.add_argument("--noise", default=0.1, type=float, help="noise level")
parser.add_argument("--output_dir", default="results_stage_ablation",
                    help="output directory")
parser.add_argument("--plot_only", default=None,
                    help="path to saved results .zkl to just plot")
args = parser.parse_args()

TIMEOUT_SEC = args.TIMEOUT * 60 * 60


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


def compute_f1(omission, commission, n_gt_edges, n_possible_edges):
    """Compute F1 from omission/commission counts."""
    tp = n_gt_edges - omission
    fp = commission
    fn = omission
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall


def evaluate_solution_set(solutions, GT, n_nodes):
    """Evaluate solution set against ground truth, returning metrics for the best solution."""
    gt_directed = set(gk.edgelist(GT))
    gt_bidirected_raw = gk.bedgelist(GT)
    gt_bidirected = set(tuple(sorted(e)) for e in gt_bidirected_raw)
    n_gt_dir = len(gt_directed)
    n_gt_bidir = len(gt_bidirected)
    n_possible_dir = n_nodes * n_nodes
    n_possible_bidir = n_nodes * (n_nodes - 1) // 2

    best_total_err = float('inf')
    best_metrics = None

    for answer in solutions:
        g1 = bfutils.num2CG(answer[0][0], n_nodes)

        oce = gk.OCE(g1, GT, normalized=False)
        total_err = oce['total'][0] + oce['total'][1]

        if total_err < best_total_err:
            best_total_err = total_err

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
            density_gt = gk.density(GT)
            density_dev = abs(density_g1 - density_gt)

            best_metrics = {
                'orientation_f1': orient_f1,
                'orientation_precision': orient_p,
                'orientation_recall': orient_r,
                'adjacency_f1': adj_f1,
                'adjacency_precision': adj_p,
                'adjacency_recall': adj_r,
                'dir_omission': dir_omission,
                'dir_commission': dir_commission,
                'bidir_omission': bidir_omission,
                'bidir_commission': bidir_commission,
                'density_deviation': density_dev,
                'density_solution': density_g1,
                'density_gt': density_gt,
                'total_error': best_total_err,
            }

    return best_metrics


def run_single_config(GT, g_estimated, DD, BD, priorities, n_nodes):
    """Run drasl with specific priorities and return metrics."""
    start = time.time()
    r = drasl([g_estimated], weighted=True, capsize=0,
              timeout=TIMEOUT_SEC,
              urate=min(15, 3 * n_nodes + 1),
              dm=[DD], bdm=[BD],
              GT_density=int(1000 * gk.density(GT)),
              edge_weights=priorities, pnum=args.PNUM, optim='optN')
    elapsed = time.time() - start

    if len(r) == 0:
        return None

    metrics = evaluate_solution_set(r, GT, n_nodes)
    if metrics is None:
        return None

    metrics['n_solutions'] = len(r)
    metrics['runtime_sec'] = elapsed
    return metrics


def run_ablation(network_num, u_rate, batch_idx):
    """Run all three stages for one configuration."""
    GT = simp_nets(network_num, selfloop=True)
    n_nodes = len(GT)
    A = cv.graph2adj(GT)
    MAXCOST = 10000

    try:
        W = create_stable_weighted_matrix(A, threshold=0.2, powers=[2, 3, 4])
    except ValueError:
        print(f"  Skipping: could not create stable matrix")
        return None

    dd = genData(W, rate=u_rate, ssize=args.ssize, noise=args.noise)

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
    for stage_name, priorities in STAGES.items():
        print(f"    Running {stage_name} with priorities={priorities}")
        metrics = run_single_config(GT, g_estimated, DD, BD, priorities, n_nodes)
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


def plot_ablation(aggregated, output_path):
    """Create a grouped bar chart of stage-wise metrics."""
    stage_names = [s for s in STAGES if s in aggregated]
    if not stage_names:
        print("No data to plot.")
        return

    short_names = ['Density\nOnly', 'Density+\nBidirected', 'Full\nPipeline']
    metrics_to_plot = [
        ('orientation_f1', 'Orientation F1'),
        ('adjacency_f1', 'Adjacency F1'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax_idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        means = [aggregated[s][metric_key]['mean'] for s in stage_names]
        stds = [aggregated[s][metric_key]['std'] for s in stage_names]

        colors = ['#4ECDC4', '#45B7D1', '#2C3E50']
        bars = axes[ax_idx].bar(range(len(stage_names)), means, yerr=stds,
                                capsize=5, color=colors[:len(stage_names)],
                                edgecolor='black', linewidth=0.5)
        axes[ax_idx].set_xticks(range(len(stage_names)))
        axes[ax_idx].set_xticklabels(short_names[:len(stage_names)], fontsize=10)
        axes[ax_idx].set_ylabel(metric_label, fontsize=12)
        axes[ax_idx].set_ylim(0, 1.05)
        axes[ax_idx].set_title(metric_label, fontsize=13)

        for bar, m in zip(bars, means):
            axes[ax_idx].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                              f'{m:.3f}', ha='center', va='bottom', fontsize=9)

    if 'density_deviation' in aggregated.get(stage_names[0], {}):
        means = [aggregated[s]['density_deviation']['mean'] for s in stage_names]
        stds = [aggregated[s]['density_deviation']['std'] for s in stage_names]
        colors = ['#4ECDC4', '#45B7D1', '#2C3E50']
        bars = axes[2].bar(range(len(stage_names)), means, yerr=stds,
                           capsize=5, color=colors[:len(stage_names)],
                           edgecolor='black', linewidth=0.5)
        axes[2].set_xticks(range(len(stage_names)))
        axes[2].set_xticklabels(short_names[:len(stage_names)], fontsize=10)
        axes[2].set_ylabel('Density Deviation', fontsize=12)
        axes[2].set_title('Density Deviation from GT', fontsize=13)
        for bar, m in zip(bars, means):
            axes[2].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                         f'{m:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ablation plot saved to {output_path}")


def main():
    if args.plot_only:
        print(f"Loading saved results from {args.plot_only}")
        saved = zkl.load(args.plot_only)
        aggregated = aggregate_results(saved['all_results'])
        print_summary_table(aggregated)
        plot_ablation(aggregated, args.plot_only.replace('.zkl', '_ablation.pdf'))
        return

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    total = len(args.NETWORKS) * len(args.UNDERSAMPLING) * args.BATCHES
    count = 0

    for net_num in args.NETWORKS:
        for u_rate in args.UNDERSAMPLING:
            for batch in range(1, args.BATCHES + 1):
                count += 1
                print(f"\n[{count}/{total}] Network={net_num}, u={u_rate}, batch={batch}")
                result = run_ablation(net_num, u_rate, batch)
                all_results.append(result)

    save_data = {
        'all_results': all_results,
        'args': vars(args),
        'stages': {k: v for k, v in STAGES.items()},
    }
    save_path = os.path.join(args.output_dir, 'stage_ablation_results.zkl')
    zkl.save(save_data, save_path)
    print(f"\nResults saved to {save_path}")

    aggregated = aggregate_results(all_results)
    print_summary_table(aggregated)
    plot_path = os.path.join(args.output_dir, 'stage_ablation.pdf')
    plot_ablation(aggregated, plot_path)


if __name__ == '__main__':
    main()
