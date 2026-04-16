"""
Hyperparameter tuning for delta parameter in DRASL solution selection

Delta determines which solutions to consider based on cost relative to minimum:
- Sort solutions by cost
- Select solutions where: cost <= min_cost + delta
- Compute average F1 score across selected solutions
- Goal: Find delta that maximizes average F1
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
from distutils.util import strtobool
from datetime import datetime
from tqdm import tqdm

def parse_arguments(PNUM):
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for delta parameter.')
    parser.add_argument("-p", "--PNUM", default=PNUM, help="number of CPUs in machine.", type=int)
    parser.add_argument("-n", "--NET", default=3, help="number of simple network", type=int)
    parser.add_argument("-u", "--UNDERSAMPLING", default=3, help="sampling rate in generated data", type=int)
    parser.add_argument("-x", "--MAXU", default=5, help="maximum number of undersampling to look for solution.", type=int)
    parser.add_argument("--num_batches", default=5, help="number of batches to average over", type=int)
    parser.add_argument("--delta_min", default=0.0, help="minimum delta as fraction of min_cost (e.g., 0.0 = 0%%)", type=float)
    parser.add_argument("--delta_max", default=2.0, help="maximum delta as fraction of min_cost (e.g., 2.0 = 200%%)", type=float)
    parser.add_argument("--delta_step", default=0.1, help="step size for delta as fraction (e.g., 0.1 = 10%%)", type=float)
    parser.add_argument("--test_mode", action='store_true', help="test mode with fewer delta values")
    return parser.parse_args()

def RASL_with_delta(data_path, network_GT, delta, priorities, PNUM, include_selfloop=True):
    """
    Run RASL and select solutions based on delta threshold
    
    Args:
        data_path: Path to data file
        network_GT: Ground truth network
        delta: Cost threshold as fraction of min_cost (e.g., 0.1 = 10% of min_cost)
        priorities: Priority weights for RASL
        PNUM: Number of processes
        include_selfloop: Whether to include self-loops
    
    Returns:
        Dictionary with metrics: {orientation, num_solutions_selected}
        Or None if no solutions found
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
        # r_estimated is a set of tuples: ((graph_number, (undersampling,)), cost)
        solutions_with_costs = []
        for answer in r_estimated:
            graph_num = answer[0][0]  # Graph number
            undersampling = answer[0][1]  # Undersampling tuple
            cost = answer[1]  # Cost
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
        print(f"Error in RASL_with_delta: {e}")
        return None

def get_delta_values(args):
    """
    Generate delta values to test based on arguments (as percentages of min_cost)
    """
    if args.test_mode:
        # Test mode: fewer values (0%, 10%, 25%, 50%, 100%)
        return [0.0, 0.1, 0.25, 0.5, 1.0]
    else:
        # Generate range of values as percentages
        delta_values = []
        current = args.delta_min
        while current <= args.delta_max + 1e-9:  # Small epsilon for floating point
            delta_values.append(round(current, 2))
            current += args.delta_step
        return delta_values

def main():
    # Setup
    CLINGO_LIMIT = 64
    PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))
    args = parse_arguments(PNUM)
    
    omp_num_threads = args.PNUM
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    
    network_GT = simp_nets(args.NET, selfloop=True)
    
    # Use original priorities from VAR_for_ruben_nets.py
    priorities = [4, 2, 5, 3, 1]
    
    # Get delta values to test
    delta_values = get_delta_values(args)
    
    print("=" * 80)
    print("DELTA HYPERPARAMETER TUNING (ORIENTATION F1 ONLY)")
    print("=" * 80)
    print(f"Network: {args.NET}, Undersampling: {args.UNDERSAMPLING}")
    print(f"Averaging over {args.num_batches} batches (1-{args.num_batches})")
    print(f"Testing {len(delta_values)} delta values (as % of min_cost)")
    print(f"Delta range: {delta_values[0]*100:.0f}% to {delta_values[-1]*100:.0f}% of min_cost")
    print(f"Priorities (fixed): {priorities}")
    print(f"Metric: Orientation F1 only")
    print("=" * 80)
    print()
    
    # Results storage
    results = []
    
    # Test each delta value
    for delta in tqdm(delta_values, desc="Testing delta values"):
        print(f"\n{'=' * 80}")
        print(f"Testing delta: {delta}")
        print(f"{'=' * 80}")
        
        # Run multiple batches
        batch_metrics = []
        for batch_num in range(1, args.num_batches + 1):
            path = os.path.expanduser(
                f'~/DataSets_Feedbacks/8_VAR_simulation/net{args.NET}/u{args.UNDERSAMPLING}/txtSTD/data{batch_num}.txt')
            
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                continue
            
            print(f"  Batch {batch_num}/{args.num_batches}...", end=" ", flush=True)
            
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
            print(f"Warning: No successful batches for delta={delta}")
            continue
        
        # Average metrics across batches (ORIENTATION ONLY)
        avg_metrics = {
            'orientation': {
                'precision': np.mean([m['orientation']['precision'] for m in batch_metrics]),
                'recall': np.mean([m['orientation']['recall'] for m in batch_metrics]),
                'F1': np.mean([m['orientation']['F1'] for m in batch_metrics])
            }
        }
        
        # Average solution counts
        avg_solutions_selected = np.mean([m['num_solutions_selected'] for m in batch_metrics])
        avg_solutions_total = np.mean([m['num_solutions_total'] for m in batch_metrics])
        avg_min_cost = np.mean([m['min_cost'] for m in batch_metrics])
        avg_max_cost_selected = np.mean([m['max_cost_selected'] for m in batch_metrics])
        
        # Average delta values
        avg_delta_absolute = np.mean([m['delta_absolute'] for m in batch_metrics])
        
        # Store result
        result = {
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
        
        results.append(result)
        
        print(f"\nResults for delta={delta*100:.0f}% (absolute: {avg_delta_absolute:.0f}):")
        print(f"  Orientation F1: {result['orientation_F1']:.4f}")
        print(f"  Orientation Precision: {result['orientation_precision']:.4f}")
        print(f"  Orientation Recall: {result['orientation_recall']:.4f}")
        print(f"  Avg solutions selected: {result['avg_solutions_selected']:.1f}/{result['avg_solutions_total']:.1f}")
    
    # Convert to DataFrame and sort by orientation F1
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('orientation_F1', ascending=False)
    
    # Save results
    if not os.path.exists('VAR_ruben/delta_tuning'):
        os.makedirs('VAR_ruben/delta_tuning')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'VAR_ruben/delta_tuning/delta_tuning_net{args.NET}_u{args.UNDERSAMPLING}_{timestamp}.csv'
    zkl_filename = f'VAR_ruben/delta_tuning/delta_tuning_net{args.NET}_u{args.UNDERSAMPLING}_{timestamp}.zkl'
    
    df_results.to_csv(csv_filename, index=False)
    print(f"\n✓ Results saved to: {csv_filename}")
    
    # Save full data with metadata
    zkl_data = {
        'results': results,
        'metadata': {
            'net': args.NET,
            'undersampling': args.UNDERSAMPLING,
            'num_batches': args.num_batches,
            'priorities': priorities,
            'delta_min': args.delta_min,
            'delta_max': args.delta_max,
            'delta_step': args.delta_step,
            'timestamp': timestamp
        }
    }
    zkl.save(zkl_data, zkl_filename)
    print(f"✓ Full data saved to: {zkl_filename}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TOP 5 DELTA VALUES (by Orientation F1):")
    print("=" * 80)
    print(df_results.head(5)[['delta_percentage', 'delta_absolute', 'orientation_F1', 
                               'orientation_precision', 'orientation_recall',
                               'avg_solutions_selected']].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION:")
    print("=" * 80)
    best = df_results.iloc[0]
    print(f"Delta: {best['delta_percentage']*100:.0f}% of min_cost (absolute: {best['delta_absolute']:.0f})")
    print(f"Orientation F1: {best['orientation_F1']:.4f}")
    print(f"  - Precision: {best['orientation_precision']:.4f}")
    print(f"  - Recall: {best['orientation_recall']:.4f}")
    print(f"Avg solutions selected: {best['avg_solutions_selected']:.1f}/{best['avg_solutions_total']:.1f}")
    print(f"Successful batches: {best['num_successful_batches']}/{args.num_batches}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print(f"\nTo analyze these results, run:")
    print(f"  python VAR_analyze_delta.py -f {csv_filename}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

