"""
Run a single priority configuration for hyperparameter tuning
Designed for parallel execution on SLURM cluster
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
import itertools
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run single priority configuration.')
    parser.add_argument("-S", "--JOB_ID", required=True, help="SLURM array task ID (1-3125)", type=int)
    parser.add_argument("-n", "--NET", default=3, help="number of simple network", type=int)
    parser.add_argument("-u", "--UNDERSAMPLING", default=3, help="sampling rate in generated data", type=int)
    parser.add_argument("-x", "--MAXU", default=5, help="maximum number of undersampling to look for solution.", type=int)
    parser.add_argument("--num_batches", default=5, help="number of batches to average over", type=int)
    parser.add_argument("-p", "--PNUM", default=1, help="number of CPUs per job", type=int)
    return parser.parse_args()

def job_id_to_priority(job_id):
    """
    Convert job ID (1-3125) to priority configuration
    Job IDs are 1-indexed, priorities are from itertools.product([1,2,3,4,5], repeat=5)
    """
    if job_id < 1 or job_id > 3125:
        raise ValueError(f"Job ID must be between 1 and 3125, got {job_id}")
    
    # Generate all combinations (0-indexed internally)
    all_priorities = list(itertools.product([1, 2, 3, 4, 5], repeat=5))
    
    # Convert 1-indexed job_id to 0-indexed
    priority_tuple = all_priorities[job_id - 1]
    
    return list(priority_tuple)

def RASL_with_priorities(data_path, network_GT, priorities, PNUM, include_selfloop=True):
    """
    Run RASL with specific priorities on given data
    """
    try:
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

        if not r_estimated:
            return None
            
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
    except Exception as e:
        print(f"Error in RASL_with_priorities: {e}", file=sys.stderr)
        return None

def run_single_priority_config(args, priorities):
    """
    Run a single priority configuration across all batches
    """
    network_GT = simp_nets(args.NET, selfloop=True)
    
    print(f"=" * 80)
    print(f"Job ID: {args.JOB_ID}")
    print(f"Priorities: {priorities}")
    print(f"Network: {args.NET}, Undersampling: {args.UNDERSAMPLING}")
    print(f"Testing {args.num_batches} batches")
    print(f"=" * 80)
    
    # Run all batches for this priority configuration
    batch_metrics = []
    for batch_num in range(1, args.num_batches + 1):
        path = os.path.expanduser(
            f'~/DataSets_Feedbacks/8_VAR_simulation/net{args.NET}/u{args.UNDERSAMPLING}/txtSTD/data{batch_num}.txt')
        
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}", file=sys.stderr)
            continue
        
        print(f"Processing batch {batch_num}/{args.num_batches}...", end=" ", flush=True)
        
        try:
            metrics = RASL_with_priorities(path, network_GT, priorities, args.PNUM, include_selfloop=True)
            if metrics:
                batch_metrics.append(metrics)
                print("✓")
            else:
                print("✗ (no results)")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if not batch_metrics:
        print("ERROR: No successful batches for this configuration", file=sys.stderr)
        return None
    
    # Average metrics across batches
    avg_metrics = {
        'orientation': {
            'precision': np.mean([m['orientation']['precision'] for m in batch_metrics]),
            'recall': np.mean([m['orientation']['recall'] for m in batch_metrics]),
            'F1': np.mean([m['orientation']['F1'] for m in batch_metrics])
        },
        'adjacency': {
            'precision': np.mean([m['adjacency']['precision'] for m in batch_metrics]),
            'recall': np.mean([m['adjacency']['recall'] for m in batch_metrics]),
            'F1': np.mean([m['adjacency']['F1'] for m in batch_metrics])
        },
        'cycle': {
            'precision': np.mean([m['cycle']['precision'] for m in batch_metrics]),
            'recall': np.mean([m['cycle']['recall'] for m in batch_metrics]),
            'F1': np.mean([m['cycle']['F1'] for m in batch_metrics])
        }
    }
    
    # Create result dictionary
    result = {
        'job_id': args.JOB_ID,
        'priorities': str(tuple(priorities)),
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
    
    print(f"\nResults for priorities {priorities}:")
    print(f"  Orientation F1: {result['orientation_F1']:.4f}")
    print(f"  Adjacency F1: {result['adjacency_F1']:.4f}")
    print(f"  Cycle F1: {result['cycle_F1']:.4f}")
    print(f"  Combined F1: {result['combined_F1']:.4f}")
    print(f"  Successful batches: {result['num_successful_batches']}/{args.num_batches}")
    
    return result

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set environment
    omp_num_threads = args.PNUM
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    
    print(f"\nStarting hyperparameter job {args.JOB_ID} of 3125")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Convert job ID to priority configuration
    try:
        priorities = job_id_to_priority(args.JOB_ID)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run the priority configuration
    result = run_single_priority_config(args, priorities)
    
    if result is None:
        print("ERROR: Failed to generate results", file=sys.stderr)
        sys.exit(1)
    
    # Save individual result
    results_folder = 'VAR_ruben/hyperparameter_tuning/individual_jobs'
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

