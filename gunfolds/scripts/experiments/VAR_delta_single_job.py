"""
Run a single delta configuration for hyperparameter tuning
Designed for parallel execution on SLURM cluster

Each job tests one delta value across multiple batches
"""
import os
import numpy as np
import pandas as pd
from gunfolds import conversions as cv
from gunfolds.scripts.datasets.simple_networks import simp_nets
from gunfolds.scripts import my_functions as mf
from gunfolds.solvers.clingo_rasl import drasl
from gunfolds.utils import graphkit as gk
from gunfolds.utils.calc_procs import get_process_count
from gunfolds.utils import bfutils
from gunfolds.utils import zickle as zkl
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import argparse
from datetime import datetime
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run single delta configuration.')
    parser.add_argument("-S", "--JOB_ID", required=True, help="SLURM array task ID (1-N)", type=int)
    parser.add_argument("-n", "--NET", default=3, help="number of simple network", type=int)
    parser.add_argument("-u", "--UNDERSAMPLING", default=3, help="sampling rate in generated data", type=int)
    parser.add_argument("-x", "--MAXU", default=5, help="maximum number of undersampling to look for solution.", type=int)
    parser.add_argument("--num_batches", default=5, help="number of batches to average over", type=int)
    parser.add_argument("-p", "--PNUM", default=1, help="number of CPUs per job", type=int)
    parser.add_argument("--delta_min", default=0.0, help="minimum delta as fraction of min_cost", type=float)
    parser.add_argument("--delta_max", default=2.0, help="maximum delta as fraction of min_cost", type=float)
    parser.add_argument("--delta_step", default=0.1, help="step size for delta as fraction", type=float)
    return parser.parse_args()

def job_id_to_delta(job_id, delta_min, delta_max, delta_step):
    """
    Convert job ID (1-indexed) to delta value (as percentage of min_cost)
    """
    # Generate all delta values as percentages
    delta_values = []
    current = delta_min
    while current <= delta_max + 1e-9:  # Small epsilon for floating point
        delta_values.append(round(current, 2))
        current += delta_step
    
    if job_id < 1 or job_id > len(delta_values):
        raise ValueError(f"Job ID must be between 1 and {len(delta_values)}, got {job_id}")
    
    # Convert 1-indexed job_id to 0-indexed
    return delta_values[job_id - 1]

def RASL_with_delta(data_path, network_GT, delta, priorities, PNUM, include_selfloop=True):
    """
    Run RASL and select solutions based on delta threshold
    
    Args:
        data_path: Path to data file
        network_GT: Ground truth network
        delta: Cost threshold as fraction of min_cost (e.g., 0.1 = 10%)
        priorities: Priority weights for RASL
        PNUM: Number of processes
        include_selfloop: Whether to include self-loops
    
    Returns:
        Dictionary with metrics (orientation only) or None if no solutions found
    """
    try:
        data = pd.read_csv(data_path, delimiter='\t')
        dataframe = pp.DataFrame(data.values)
        cond_ind_test = ParCorr()
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05)
        g_estimated, A, B = cv.Glag2CG(results)
        
        MAXCOST = 50
        DD = (np.abs((np.abs(A / np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
        BD = (np.abs((np.abs(B / np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

        r_estimated = drasl([g_estimated], weighted=True, capsize=0, timeout=0,
                            urate=min(5, (3 * len(g_estimated) + 1)),
                            dm=[DD],
                            bdm=[BD],
                            scc=False,
                            GT_density=int(100 * gk.density(network_GT)),
                            edge_weights=priorities, pnum=PNUM, optim='optN', selfloop=True)

        if not r_estimated:
            return None
        
        # Extract solutions with their costs
        solutions_with_costs = []
        for answer in r_estimated:
            graph_num = answer[0][0]
            undersampling = answer[0][1]
            cost = answer[1]
            solutions_with_costs.append((graph_num, undersampling, cost))
        
        # Sort by cost
        solutions_with_costs.sort(key=lambda x: x[2])
        
        # Find minimum cost
        min_cost = solutions_with_costs[0][2]
        
        # Calculate absolute delta from percentage
        absolute_delta = delta * min_cost
        
        # Select solutions within delta threshold
        selected_solutions = [s for s in solutions_with_costs if s[2] <= min_cost + absolute_delta]
        
        if not selected_solutions:
            return None
        
        # Compute F1 scores for all selected solutions
        f1_scores = []
        for graph_num, undersampling, cost in selected_solutions:
            res_rasl = bfutils.num2CG(graph_num, len(network_GT))
            rasl_sol = mf.precision_recall(res_rasl, network_GT, include_selfloop=include_selfloop)
            f1_scores.append(rasl_sol)
        
        # Average metrics across selected solutions (ORIENTATION ONLY)
        avg_metrics = {
            'orientation': {
                'precision': np.mean([s['orientation']['precision'] for s in f1_scores]),
                'recall': np.mean([s['orientation']['recall'] for s in f1_scores]),
                'F1': np.mean([s['orientation']['F1'] for s in f1_scores])
            },
            'num_solutions_selected': len(selected_solutions),
            'num_solutions_total': len(solutions_with_costs),
            'min_cost': float(min_cost),
            'max_cost_selected': float(selected_solutions[-1][2]),
            'delta_percentage': float(delta),
            'delta_absolute': float(absolute_delta)
        }
        
        return avg_metrics
    except Exception as e:
        print(f"Error in RASL_with_delta: {e}", file=sys.stderr)
        return None

def run_single_delta_config(args, delta, priorities):
    """
    Run a single delta configuration across all batches
    """
    network_GT = simp_nets(args.NET, selfloop=True)
    
    print(f"=" * 80)
    print(f"Job ID: {args.JOB_ID}")
    print(f"Delta: {delta*100:.0f}% of min_cost")
    print(f"Network: {args.NET}, Undersampling: {args.UNDERSAMPLING}")
    print(f"Testing {args.num_batches} batches")
    print(f"Priorities (fixed): {priorities}")
    print(f"Metric: Orientation F1 only")
    print(f"=" * 80)
    
    # Run all batches for this delta value
    batch_metrics = []
    for batch_num in range(1, args.num_batches + 1):
        path = os.path.expanduser(
            f'~/DataSets_Feedbacks/8_VAR_simulation/net{args.NET}/u{args.UNDERSAMPLING}/txtSTD/data{batch_num}.txt')
        
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}", file=sys.stderr)
            continue
        
        print(f"Processing batch {batch_num}/{args.num_batches}...", end=" ", flush=True)
        
        try:
            metrics = RASL_with_delta(path, network_GT, delta, priorities, args.PNUM, include_selfloop=True)
            if metrics:
                batch_metrics.append(metrics)
                print(f"✓ (selected {metrics['num_solutions_selected']}/{metrics['num_solutions_total']} solutions)")
            else:
                print("✗ (no results)")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if not batch_metrics:
        print("ERROR: No successful batches for this configuration", file=sys.stderr)
        return None
    
    # Average metrics across batches (ORIENTATION ONLY)
    avg_metrics = {
        'orientation': {
            'precision': np.mean([m['orientation']['precision'] for m in batch_metrics]),
            'recall': np.mean([m['orientation']['recall'] for m in batch_metrics]),
            'F1': np.mean([m['orientation']['F1'] for m in batch_metrics])
        }
    }
    
    # Average solution counts and delta values
    avg_solutions_selected = np.mean([m['num_solutions_selected'] for m in batch_metrics])
    avg_solutions_total = np.mean([m['num_solutions_total'] for m in batch_metrics])
    avg_min_cost = np.mean([m['min_cost'] for m in batch_metrics])
    avg_max_cost_selected = np.mean([m['max_cost_selected'] for m in batch_metrics])
    avg_delta_absolute = np.mean([m['delta_absolute'] for m in batch_metrics])
    
    # Create result dictionary
    result = {
        'job_id': args.JOB_ID,
        'delta_percentage': delta,
        'delta_absolute': avg_delta_absolute,
        'num_successful_batches': len(batch_metrics),
        'avg_solutions_selected': avg_solutions_selected,
        'avg_solutions_total': avg_solutions_total,
        'avg_min_cost': avg_min_cost,
        'avg_max_cost_selected': avg_max_cost_selected,
        'orientation_precision': avg_metrics['orientation']['precision'],
        'orientation_recall': avg_metrics['orientation']['recall'],
        'orientation_F1': avg_metrics['orientation']['F1']
    }
    
    print(f"\nResults for delta={delta*100:.0f}% (absolute: {avg_delta_absolute:.0f}):")
    print(f"  Orientation F1: {result['orientation_F1']:.4f}")
    print(f"  Orientation Precision: {result['orientation_precision']:.4f}")
    print(f"  Orientation Recall: {result['orientation_recall']:.4f}")
    print(f"  Avg solutions selected: {result['avg_solutions_selected']:.1f}/{result['avg_solutions_total']:.1f}")
    print(f"  Successful batches: {result['num_successful_batches']}/{args.num_batches}")
    
    return result

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set environment
    omp_num_threads = args.PNUM
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    
    # Original priorities from VAR_for_ruben_nets.py
    priorities = [4, 2, 5, 3, 1]
    
    print(f"\nStarting delta hyperparameter job {args.JOB_ID}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Convert job ID to delta value
    try:
        delta = job_id_to_delta(args.JOB_ID, args.delta_min, args.delta_max, args.delta_step)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run the delta configuration
    result = run_single_delta_config(args, delta, priorities)
    
    if result is None:
        print("ERROR: Failed to generate results", file=sys.stderr)
        sys.exit(1)
    
    # Save individual result
    results_folder = 'VAR_ruben/delta_tuning/individual_jobs'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Save as both CSV and ZKL
    csv_filename = f'{results_folder}/job_{args.JOB_ID:04d}_net{args.NET}_u{args.UNDERSAMPLING}.csv'
    zkl_filename = f'{results_folder}/job_{args.JOB_ID:04d}_net{args.NET}_u{args.UNDERSAMPLING}.zkl'
    
    # Save CSV
    df = pd.DataFrame([result])
    df.to_csv(csv_filename, index=False)
    print(f"\n✓ Results saved to: {csv_filename}")
    
    # Save ZKL
    zkl.save(result, zkl_filename)
    print(f"✓ Results saved to: {zkl_filename}")
    
    print(f"\n{'=' * 80}")
    print(f"Job {args.JOB_ID} completed successfully!")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    main()

