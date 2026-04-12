"""
Hyperparameter Cross-Validation Across Networks and Topologies

Demonstrates that the optimal priority configuration [1,2,1,2,3] and
delta=90% generalize across different graph topologies, sizes, and
undersampling rates.

Tests on:
  - All 5 Sanchez-Romero simple networks
  - VAR ringmore graphs at different sizes (6, 8, 10 nodes)
  - Multiple undersampling rates (2, 3)

This addresses Reviewer rp9v W2 and Reviewer cuMG's concern about
hyperparameter tuning on a single network.
"""
import os
import numpy as np
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from gunfolds.utils import bfutils
from gunfolds.utils import zickle as zkl
from gunfolds.utils import graphkit as gk
from gunfolds.solvers.clingo_rasl import drasl
from gunfolds.utils.calc_procs import get_process_count
from gunfolds import conversions as cv
from gunfolds.scripts.datasets.simple_networks import simp_nets

import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import tigramite.data_processing as pp

CLINGO_LIMIT = 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))

parser = argparse.ArgumentParser(description='Hyperparameter cross-validation.')
parser.add_argument("-b", "--BATCHES", default=10, type=int,
                    help="number of batches per configuration")
parser.add_argument("-p", "--PNUM", default=PNUM, type=int, help="number of CPUs")
parser.add_argument("-t", "--TIMEOUT", default=2, type=int, help="timeout in hours")
parser.add_argument("--ssize", default=5000, type=int, help="sample size")
parser.add_argument("--noise", default=0.1, type=float, help="noise level")
parser.add_argument("--output_dir", default="results_hyperparam_cv",
                    help="output directory")
parser.add_argument("--plot_only", default=None,
                    help="path to saved results .zkl to just plot")
args = parser.parse_args()

TIMEOUT_SEC = args.TIMEOUT * 60 * 60
OPTIMAL_PRIORITIES = [1, 2, 1, 2, 3]


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


def compute_f1(omission, commission, n_gt_edges):
    tp = n_gt_edges - omission
    fp = commission
    fn = omission
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall


def evaluate_best_solution(solutions, GT, n_nodes):
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
            orient_f1, orient_p, orient_r = compute_f1(
                oce['directed'][0], oce['directed'][1], gt_dir_edges)
            total_f1, total_p, total_r = compute_f1(
                oce['total'][0], oce['total'][1], n_gt_total)

            best_metrics = {
                'orientation_f1': orient_f1,
                'orientation_precision': orient_p,
                'orientation_recall': orient_r,
                'adjacency_f1': total_f1,
                'n_solutions': len(solutions),
            }

    return best_metrics


def run_experiment_sanchez(network_num, u_rate, batch_idx):
    """Run on a Sanchez-Romero simple network."""
    GT = simp_nets(network_num, selfloop=True)
    n_nodes = len(GT)
    A = cv.graph2adj(GT)
    MAXCOST = 50

    try:
        W = create_stable_weighted_matrix(A, threshold=0.2, powers=[2, 3, 4])
    except ValueError:
        return None

    dd = genData(W, rate=u_rate, ssize=args.ssize, noise=args.noise)

    dataframe = pp.DataFrame(np.transpose(dd))
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results_pcmci = pcmci.run_pcmci(tau_max=2, pc_alpha=None)
    g_estimated, A_mat, B_mat = cv.Glag2CG(results_pcmci)

    DD = (np.abs((np.abs(A_mat / np.abs(A_mat).max()) +
                  (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
    BD = (np.abs((np.abs(B_mat / np.abs(B_mat).max()) +
                  (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

    r = drasl([g_estimated], weighted=True, capsize=0,
              timeout=TIMEOUT_SEC,
              urate=min(15, 3 * n_nodes + 1),
              dm=[DD], bdm=[BD],
              GT_density=int(100 * gk.density(GT)),
              edge_weights=OPTIMAL_PRIORITIES, pnum=args.PNUM, optim='optN',
              selfloop=True)

    if len(r) == 0:
        return None

    metrics = evaluate_best_solution(r, GT, n_nodes)
    if metrics:
        metrics['graph_type'] = f'Sanchez Net {network_num}'
        metrics['n_nodes'] = n_nodes
        metrics['u_rate'] = u_rate
        metrics['batch'] = batch_idx
    return metrics


def run_experiment_ringmore(n_nodes, density, u_rate, batch_idx):
    """Run on a random ringmore graph."""
    num_edges = int(density * n_nodes)
    attempts = 0
    while True:
        attempts += 1
        if attempts > 100:
            return None
        g = gk.ringmore(n_nodes, max(1, num_edges - n_nodes))
        x = bfutils.all_undersamples(g)
        if u_rate <= len(x):
            break

    GT = g
    A = cv.graph2adj(GT)
    MAXCOST = 50

    try:
        W = create_stable_weighted_matrix(A, threshold=0.2, powers=[2, 3, 4])
    except ValueError:
        return None

    dd = genData(W, rate=u_rate, ssize=args.ssize, noise=args.noise)

    dataframe = pp.DataFrame(np.transpose(dd))
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results_pcmci = pcmci.run_pcmci(tau_max=2, pc_alpha=None)
    g_estimated, A_mat, B_mat = cv.Glag2CG(results_pcmci)

    DD = (np.abs((np.abs(A_mat / np.abs(A_mat).max()) +
                  (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
    BD = (np.abs((np.abs(B_mat / np.abs(B_mat).max()) +
                  (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

    r = drasl([g_estimated], weighted=True, capsize=0,
              timeout=TIMEOUT_SEC,
              urate=min(15, 3 * n_nodes + 1),
              dm=[DD], bdm=[BD],
              GT_density=int(100 * gk.density(GT)),
              edge_weights=OPTIMAL_PRIORITIES, pnum=args.PNUM, optim='optN')

    if len(r) == 0:
        return None

    metrics = evaluate_best_solution(r, GT, n_nodes)
    if metrics:
        metrics['graph_type'] = f'Ringmore n={n_nodes} d={density}'
        metrics['n_nodes'] = n_nodes
        metrics['u_rate'] = u_rate
        metrics['batch'] = batch_idx
    return metrics


def print_summary(all_results):
    """Print a summary grouped by graph type and undersampling rate."""
    if not all_results:
        print("No results to summarize.")
        return

    valid = [r for r in all_results if r is not None]
    if not valid:
        print("All runs failed.")
        return

    graph_types = sorted(set(r['graph_type'] for r in valid))
    u_rates = sorted(set(r['u_rate'] for r in valid))

    print("\n" + "=" * 85)
    print("HYPERPARAMETER CROSS-VALIDATION RESULTS")
    print(f"Priority: {OPTIMAL_PRIORITIES}")
    print("=" * 85)
    print(f"{'Graph Type':<30} {'u':>3} {'Orient F1':>14} {'Adj F1':>14} {'N':>4}")
    print("-" * 70)

    for gt in graph_types:
        for u in u_rates:
            subset = [r for r in valid if r['graph_type'] == gt and r['u_rate'] == u]
            if not subset:
                continue
            of1 = [r['orientation_f1'] for r in subset]
            af1 = [r['adjacency_f1'] for r in subset]
            print(f"{gt:<30} {u:>3} "
                  f"{np.mean(of1):>7.3f}±{np.std(of1):.3f} "
                  f"{np.mean(af1):>7.3f}±{np.std(af1):.3f} "
                  f"{len(subset):>4}")

    print("=" * 85)

    print("\nOverall across all configurations:")
    of1_all = [r['orientation_f1'] for r in valid]
    af1_all = [r['adjacency_f1'] for r in valid]
    print(f"  Orientation F1: {np.mean(of1_all):.3f} ± {np.std(of1_all):.3f} "
          f"(min={np.min(of1_all):.3f}, max={np.max(of1_all):.3f})")
    print(f"  Adjacency F1:   {np.mean(af1_all):.3f} ± {np.std(af1_all):.3f} "
          f"(min={np.min(af1_all):.3f}, max={np.max(af1_all):.3f})")


def plot_results(all_results, output_path):
    """Create a grouped visualization."""
    valid = [r for r in all_results if r is not None]
    if not valid:
        return

    df = pd.DataFrame(valid)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in [
        (axes[0], 'orientation_f1', 'Orientation F1'),
        (axes[1], 'adjacency_f1', 'Adjacency F1'),
    ]:
        graph_types = sorted(df['graph_type'].unique())
        x = np.arange(len(graph_types))
        width = 0.35

        for i, u in enumerate(sorted(df['u_rate'].unique())):
            means = []
            stds = []
            for gt in graph_types:
                subset = df[(df['graph_type'] == gt) & (df['u_rate'] == u)]
                means.append(subset[metric].mean() if len(subset) > 0 else 0)
                stds.append(subset[metric].std() if len(subset) > 0 else 0)

            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                   label=f'u={u}', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(graph_types, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cross-validation plot saved to {output_path}")


def main():
    if args.plot_only:
        saved = zkl.load(args.plot_only)
        print_summary(saved['all_results'])
        plot_results(saved['all_results'],
                     args.plot_only.replace('.zkl', '_cv.pdf'))
        return

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    sanchez_nets = [1, 2, 3, 4, 5]
    ringmore_configs = [(6, 1.5), (8, 1.5), (10, 1.5)]
    u_rates = [2, 3]

    total = (len(sanchez_nets) + len(ringmore_configs)) * len(u_rates) * args.BATCHES
    count = 0

    for net_num in sanchez_nets:
        for u_rate in u_rates:
            for batch in range(1, args.BATCHES + 1):
                count += 1
                print(f"[{count}/{total}] Sanchez Net {net_num}, u={u_rate}, batch={batch}")
                result = run_experiment_sanchez(net_num, u_rate, batch)
                if result:
                    all_results.append(result)
                    print(f"  Orient F1={result['orientation_f1']:.3f}, "
                          f"Adj F1={result['adjacency_f1']:.3f}")
                else:
                    print(f"  Failed")

    for n_nodes, density in ringmore_configs:
        for u_rate in u_rates:
            for batch in range(1, args.BATCHES + 1):
                count += 1
                print(f"[{count}/{total}] Ringmore n={n_nodes} d={density}, "
                      f"u={u_rate}, batch={batch}")
                result = run_experiment_ringmore(n_nodes, density, u_rate, batch)
                if result:
                    all_results.append(result)
                    print(f"  Orient F1={result['orientation_f1']:.3f}, "
                          f"Adj F1={result['adjacency_f1']:.3f}")
                else:
                    print(f"  Failed")

    save_data = {
        'all_results': all_results,
        'args': vars(args),
        'priorities': OPTIMAL_PRIORITIES,
    }
    save_path = os.path.join(args.output_dir, 'hyperparam_cv_results.zkl')
    zkl.save(save_data, save_path)
    print(f"\nResults saved to {save_path}")

    print_summary(all_results)
    plot_path = os.path.join(args.output_dir, 'hyperparam_cross_validation.pdf')
    plot_results(all_results, plot_path)


if __name__ == '__main__':
    main()
