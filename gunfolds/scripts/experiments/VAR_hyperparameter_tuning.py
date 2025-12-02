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
import itertools
from tqdm import tqdm

def parse_arguments(PNUM):
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for priorities.')
    parser.add_argument("-p", "--PNUM", default=PNUM, help="number of CPUs in machine.", type=int)
    parser.add_argument("-n", "--NET", default=3, help="number of simple network", type=int)
    parser.add_argument("-u", "--UNDERSAMPLING", default=3, help="sampling rate in generated data", type=int)
    parser.add_argument("-x", "--MAXU", default=5, help="maximum number of undersampling to look for solution.", type=int)
    parser.add_argument("--num_batches", default=5, help="number of batches to average over", type=int)
    parser.add_argument("--test_all", action='store_true', help="test all possible priority combinations (3125 total)")
    parser.add_argument("--test_subset", action='store_true', help="test a curated subset of priority combinations")
    return parser.parse_args()

def RASL_with_priorities(data_path, network_GT, priorities, PNUM, include_selfloop=True):
    """
    Run RASL with specific priorities on given data
    """
    data = pd.read_csv(data_path, delimiter='\t')
    dataframe = pp.DataFrame(data.values)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05)
    g_estimated, A, B = cv.Glag2CG(results)
    
    MAXCOST = 10000
    DD = (np.abs((np.abs(A / np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
    BD = (np.abs((np.abs(B / np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)

    r_estimated = drasl([g_estimated], weighted=True, capsize=0, timeout=0,
                        urate=min(5, (3 * len(g_estimated) + 1)),
                        dm=[DD],
                        bdm=[BD],
                        scc=False,
                        GT_density=int(1000 * gk.density(network_GT)),
                        edge_weights=priorities, pnum=PNUM, optim='optN', selfloop=True)

    # Find best solution by maximizing F1 scores
    max_f1_score = 0
    max_answer = None
    for answer in r_estimated:
        res_rasl = bfutils.num2CG(answer[0][0], len(network_GT))
        rasl_sol = mf.precision_recall(res_rasl, network_GT, include_selfloop=include_selfloop)
        
        curr_f1 = (rasl_sol['orientation']['F1']) + (rasl_sol['adjacency']['F1']) + (rasl_sol['cycle']['F1'])
        
        if curr_f1 > max_f1_score:
            max_f1_score = curr_f1
            max_answer = answer

    if max_answer is None:
        return None
        
    res_rasl = bfutils.num2CG(max_answer[0][0], len(network_GT))
    metrics = mf.precision_recall(res_rasl, network_GT, include_selfloop=include_selfloop)
    
    return metrics

def run_batch(args, network_GT, priorities, batch_num):
    """
    Run a single batch with given priorities
    """
    path = os.path.expanduser(
        f'~/DataSets_Feedbacks/8_VAR_simulation/net{args.NET}/u{args.UNDERSAMPLING}/txtSTD/data{batch_num}.txt')
    
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return None
    
    metrics = RASL_with_priorities(path, network_GT, priorities, args.PNUM, include_selfloop=True)
    return metrics

def average_metrics(metrics_list):
    """
    Average metrics across multiple batches
    """
    if not metrics_list or all(m is None for m in metrics_list):
        return None
    
    # Filter out None values
    valid_metrics = [m for m in metrics_list if m is not None]
    
    if not valid_metrics:
        return None
    
    avg_metrics = {
        'orientation': {
            'precision': np.mean([m['orientation']['precision'] for m in valid_metrics]),
            'recall': np.mean([m['orientation']['recall'] for m in valid_metrics]),
            'F1': np.mean([m['orientation']['F1'] for m in valid_metrics])
        },
        'adjacency': {
            'precision': np.mean([m['adjacency']['precision'] for m in valid_metrics]),
            'recall': np.mean([m['adjacency']['recall'] for m in valid_metrics]),
            'F1': np.mean([m['adjacency']['F1'] for m in valid_metrics])
        },
        'cycle': {
            'precision': np.mean([m['cycle']['precision'] for m in valid_metrics]),
            'recall': np.mean([m['cycle']['recall'] for m in valid_metrics]),
            'F1': np.mean([m['cycle']['F1'] for m in valid_metrics])
        }
    }
    
    return avg_metrics

def get_priority_combinations(test_all=False, test_subset=False):
    """
    Generate priority combinations to test
    """
    if test_all:
        # All possible combinations: 5^5 = 3125
        return list(itertools.product([1, 2, 3, 4, 5], repeat=5))
    elif test_subset:
        # Curated subset of interesting combinations
        priority_sets = [
            # Original
            [4, 2, 5, 3, 1],
            # All equal priority
            [1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3],
            # Extremes
            [5, 5, 5, 5, 5],
            [1, 1, 1, 1, 5],
            [5, 1, 1, 1, 1],
            # Ascending/Descending
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            # High first constraint
            [5, 1, 1, 1, 1],
            [5, 2, 2, 2, 2],
            [5, 3, 3, 3, 3],
            # High second constraint
            [1, 5, 1, 1, 1],
            [2, 5, 2, 2, 2],
            # Various balanced combinations
            [3, 3, 3, 1, 1],
            [1, 1, 3, 3, 3],
            [2, 2, 2, 4, 4],
            [4, 4, 2, 2, 2],
            # Random variations
            [5, 3, 1, 2, 4],
            [1, 3, 5, 2, 4],
            [2, 4, 1, 5, 3],
        ]
        return [tuple(p) for p in priority_sets]
    else:
        # Default: small representative set
        priority_sets = [
            [4, 2, 5, 3, 1],  # Original
            [1, 1, 1, 1, 1],  # All equal
            [5, 4, 3, 2, 1],  # Descending
            [1, 2, 3, 4, 5],  # Ascending
            [5, 5, 5, 5, 5],  # All high
        ]
        return [tuple(p) for p in priority_sets]

def main():
    # Setup
    CLINGO_LIMIT = 64
    PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))
    args = parse_arguments(PNUM)
    
    omp_num_threads = args.PNUM
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    
    network_GT = simp_nets(args.NET, selfloop=True)
    
    # Get priority combinations to test
    priority_combinations = get_priority_combinations(args.test_all, args.test_subset)
    
    print(f"Testing {len(priority_combinations)} priority combinations")
    print(f"Network: {args.NET}, Undersampling: {args.UNDERSAMPLING}")
    print(f"Averaging over {args.num_batches} batches (1-{args.num_batches})")
    print("-" * 80)
    
    # Results storage
    results = []
    
    # Test each priority combination
    for priorities in tqdm(priority_combinations, desc="Testing priorities"):
        print(f"\nTesting priorities: {priorities}")
        
        # Run multiple batches
        batch_metrics = []
        for batch_num in range(1, args.num_batches + 1):
            print(f"  Batch {batch_num}/{args.num_batches}...", end=" ")
            metrics = run_batch(args, network_GT, list(priorities), batch_num)
            if metrics:
                batch_metrics.append(metrics)
                print("✓")
            else:
                print("✗")
        
        # Average across batches
        if batch_metrics:
            avg_metrics = average_metrics(batch_metrics)
            
            if avg_metrics:
                result = {
                    'priorities': str(priorities),
                    'p1': priorities[0], 'p2': priorities[1], 'p3': priorities[2], 
                    'p4': priorities[3], 'p5': priorities[4],
                    'num_successful_batches': len(batch_metrics),
                    'orientation_precision': avg_metrics['orientation']['precision'],
                    'orientation_recall': avg_metrics['orientation']['recall'],
                    'orientation_F1': avg_metrics['orientation']['F1'],
                    'adjacency_precision': avg_metrics['adjacency']['precision'],
                    'adjacency_recall': avg_metrics['adjacency']['recall'],
                    'adjacency_F1': avg_metrics['adjacency']['F1'],
                    'cycle_precision': avg_metrics['cycle']['precision'],
                    'cycle_recall': avg_metrics['cycle']['recall'],
                    'cycle_F1': avg_metrics['cycle']['F1'],
                    'combined_F1': (avg_metrics['orientation']['F1'] + 
                                   avg_metrics['adjacency']['F1'] + 
                                   avg_metrics['cycle']['F1']) / 3.0
                }
                results.append(result)
                print(f"  Combined F1: {result['combined_F1']:.4f}")
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by combined F1 score
    df_results = df_results.sort_values('combined_F1', ascending=False)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_folder = 'VAR_ruben/hyperparameter_tuning'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Save as CSV
    csv_filename = f'{results_folder}/priority_tuning_net{args.NET}_u{args.UNDERSAMPLING}_{timestamp}.csv'
    df_results.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")
    
    # Save as pickle for full data preservation
    zkl_filename = f'{results_folder}/priority_tuning_net{args.NET}_u{args.UNDERSAMPLING}_{timestamp}.zkl'
    zkl.save({'results': results, 'args': vars(args)}, zkl_filename)
    print(f"Full results saved to: {zkl_filename}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TOP 10 PRIORITY COMBINATIONS:")
    print("=" * 80)
    print(df_results[['priorities', 'orientation_F1', 'adjacency_F1', 'cycle_F1', 'combined_F1']].head(10).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("DETAILED TOP 5:")
    print("=" * 80)
    for idx, row in df_results.head(5).iterrows():
        print(f"\nRank {df_results.index.get_loc(idx) + 1}: Priorities {row['priorities']}")
        print(f"  Orientation - P: {row['orientation_precision']:.4f}, R: {row['orientation_recall']:.4f}, F1: {row['orientation_F1']:.4f}")
        print(f"  Adjacency   - P: {row['adjacency_precision']:.4f}, R: {row['adjacency_recall']:.4f}, F1: {row['adjacency_F1']:.4f}")
        print(f"  Cycle       - P: {row['cycle_precision']:.4f}, R: {row['cycle_recall']:.4f}, F1: {row['cycle_F1']:.4f}")
        print(f"  Combined F1: {row['combined_F1']:.4f}")
        print(f"  Successful batches: {row['num_successful_batches']}/{args.num_batches}")

if __name__ == "__main__":
    main()

