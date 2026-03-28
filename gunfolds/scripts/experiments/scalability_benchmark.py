"""
Scalability / Runtime Benchmark

Measures wall-clock runtime and F1 accuracy as a function of graph size (number
of nodes), from 5 to 20 nodes. Generates random ringmore graphs, undersamples,
adds noise via edge perturbation, and runs the weighted drasl solver.

This addresses Reviewer jQK7 Q3, Reviewer L7F5, Reviewer rp9v Q1, and
Reviewer cuMG's scalability concerns.
"""
import os
import random
import numpy as np
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gunfolds.utils import bfutils
from gunfolds.utils import bfutils as bfu
from gunfolds.utils import zickle as zkl
from gunfolds.utils import graphkit as gk
from gunfolds.solvers.clingo_rasl import drasl
from gunfolds.utils.calc_procs import get_process_count

CLINGO_LIMIT = 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))

parser = argparse.ArgumentParser(description='Scalability benchmark.')
parser.add_argument("--nodes", nargs='+', default=[5, 6, 7, 8, 9, 10, 12, 15, 18, 20],
                    type=int, help="node counts to benchmark")
parser.add_argument("-u", "--UNDERSAMPLING", nargs='+', default=[2, 3],
                    type=int, help="undersampling rates")
parser.add_argument("-b", "--BATCHES", default=5, type=int,
                    help="repetitions per configuration for error bars")
parser.add_argument("-d", "--DEG", default=1.5, type=float,
                    help="average degree of graphs")
parser.add_argument("-p", "--PNUM", default=PNUM, type=int, help="number of CPUs")
parser.add_argument("-t", "--TIMEOUT", default=4, type=int, help="timeout in hours")
parser.add_argument("-x", "--MAXU", default=8, type=int,
                    help="max undersampling rate to search")
parser.add_argument("--output_dir", default="results_scalability",
                    help="output directory")
parser.add_argument("--plot_only", default=None,
                    help="path to saved results .zkl to just plot")
args = parser.parse_args()

TIMEOUT_SEC = args.TIMEOUT * 60 * 60


def run_single_benchmark(n_nodes, u_rate, deg, batch_idx):
    """Run a single benchmark: generate graph, undersample, perturb, solve, time it."""
    num_edges = int(deg * n_nodes)

    attempts = 0
    while True:
        attempts += 1
        if attempts > 100:
            print(f"    Could not generate valid graph after 100 attempts")
            return None
        g = gk.ringmore(n_nodes, max(1, num_edges - n_nodes))
        x = bfu.all_undersamples(g)
        if u_rate <= len(x):
            break

    g_broken = x[u_rate - 1].copy()

    node = random.randint(1, len(g_broken))
    child = random.randint(1, len(g_broken))
    if child in g_broken[node]:
        choices = [1, 2, 3]
        choices.remove(g_broken[node][child])
        g_broken[node][child] = random.choice(choices)
    else:
        g_broken[node][child] = random.randint(1, 4)

    gt_density = int(1000 * gk.density(g))
    max_urate = min(args.MAXU, 3 * len(g_broken) + 1)

    start = time.time()
    try:
        r = drasl([g_broken], capsize=0, weighted=True,
                  urate=max_urate, timeout=TIMEOUT_SEC,
                  pnum=args.PNUM, GT_density=gt_density,
                  edge_weights=[1, 2, 1, 2, 3], optim='optN')
    except Exception as e:
        print(f"    Error: {e}")
        return None
    elapsed = time.time() - start

    timed_out = elapsed >= (TIMEOUT_SEC - 1)

    n_solutions = len(r) if r else 0

    best_err = float('inf')
    best_f1 = 0.0
    if r:
        for answer in r:
            g1 = bfu.num2CG(answer[0][0], n_nodes)
            errors = gk.OCE(g1, g, normalized=False)
            total_err = errors['total'][0] + errors['total'][1]
            if total_err < best_err:
                best_err = total_err
                n_gt = len(gk.edgelist(g)) + len(set(tuple(sorted(e)) for e in gk.bedgelist(g)))
                tp = n_gt - errors['total'][0]
                fp = errors['total'][1]
                fn = errors['total'][0]
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                best_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        'n_nodes': n_nodes,
        'u_rate': u_rate,
        'batch': batch_idx,
        'runtime_sec': elapsed,
        'n_solutions': n_solutions,
        'best_f1': best_f1,
        'timed_out': timed_out,
        'gt_density': gk.density(g),
    }


def plot_scalability(all_results, output_dir):
    """Create runtime and F1 vs. node count plots."""
    u_rates = sorted(set(r['u_rate'] for r in all_results))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for u_rate in u_rates:
        subset = [r for r in all_results if r['u_rate'] == u_rate and not r['timed_out']]
        if not subset:
            continue

        nodes_set = sorted(set(r['n_nodes'] for r in subset))
        mean_times = []
        std_times = []
        mean_f1s = []
        std_f1s = []
        valid_nodes = []

        for n in nodes_set:
            node_results = [r for r in subset if r['n_nodes'] == n]
            if not node_results:
                continue
            valid_nodes.append(n)
            times = [r['runtime_sec'] for r in node_results]
            f1s = [r['best_f1'] for r in node_results]
            mean_times.append(np.mean(times))
            std_times.append(np.std(times))
            mean_f1s.append(np.mean(f1s))
            std_f1s.append(np.std(f1s))

        axes[0].errorbar(valid_nodes, mean_times, yerr=std_times,
                         marker='o', capsize=4, label=f'u={u_rate}')
        axes[1].errorbar(valid_nodes, mean_f1s, yerr=std_f1s,
                         marker='s', capsize=4, label=f'u={u_rate}')

    axes[0].set_xlabel('Number of Nodes', fontsize=12)
    axes[0].set_ylabel('Runtime (seconds)', fontsize=12)
    axes[0].set_title('Runtime vs. Graph Size', fontsize=13)
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Number of Nodes', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('Accuracy vs. Graph Size', fontsize=13)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'scalability_benchmark.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scalability plot saved to {plot_path}")


def print_summary(all_results):
    """Print summary table."""
    u_rates = sorted(set(r['u_rate'] for r in all_results))

    print("\n" + "=" * 80)
    print("SCALABILITY BENCHMARK RESULTS")
    print("=" * 80)

    for u_rate in u_rates:
        print(f"\nUndersampling rate u={u_rate}:")
        print(f"{'Nodes':>6} {'Runtime(s)':>14} {'F1':>10} {'#Solutions':>12} {'Timeouts':>10}")
        print("-" * 55)

        subset = [r for r in all_results if r['u_rate'] == u_rate]
        nodes_set = sorted(set(r['n_nodes'] for r in subset))

        for n in nodes_set:
            node_results = [r for r in subset if r['n_nodes'] == n]
            completed = [r for r in node_results if not r['timed_out']]
            n_timeouts = sum(1 for r in node_results if r['timed_out'])

            if completed:
                times = [r['runtime_sec'] for r in completed]
                f1s = [r['best_f1'] for r in completed]
                sols = [r['n_solutions'] for r in completed]
                print(f"{n:>6} {np.mean(times):>8.1f}±{np.std(times):>4.1f} "
                      f"{np.mean(f1s):>7.3f}±{np.std(f1s):.3f} "
                      f"{np.mean(sols):>8.0f}±{np.std(sols):>3.0f} "
                      f"{n_timeouts:>6}/{len(node_results)}")
            else:
                print(f"{n:>6} {'ALL TIMED OUT':>14} {'N/A':>10} {'N/A':>12} "
                      f"{n_timeouts:>6}/{len(node_results)}")

    print("=" * 80)


def main():
    if args.plot_only:
        print(f"Loading saved results from {args.plot_only}")
        saved = zkl.load(args.plot_only)
        print_summary(saved['all_results'])
        plot_scalability(saved['all_results'], os.path.dirname(args.plot_only))
        return

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    total = len(args.nodes) * len(args.UNDERSAMPLING) * args.BATCHES
    count = 0

    for n_nodes in args.nodes:
        for u_rate in args.UNDERSAMPLING:
            for batch in range(1, args.BATCHES + 1):
                count += 1
                print(f"[{count}/{total}] Nodes={n_nodes}, u={u_rate}, batch={batch}")
                result = run_single_benchmark(n_nodes, u_rate, args.DEG, batch)
                if result is not None:
                    all_results.append(result)
                    status = "TIMEOUT" if result['timed_out'] else "OK"
                    print(f"    {status}: {result['runtime_sec']:.1f}s, "
                          f"F1={result['best_f1']:.3f}, "
                          f"solutions={result['n_solutions']}")

    save_data = {
        'all_results': all_results,
        'args': vars(args),
    }
    save_path = os.path.join(args.output_dir, 'scalability_results.zkl')
    zkl.save(save_data, save_path)
    print(f"\nResults saved to {save_path}")

    print_summary(all_results)
    plot_scalability(all_results, args.output_dir)


if __name__ == '__main__':
    main()
