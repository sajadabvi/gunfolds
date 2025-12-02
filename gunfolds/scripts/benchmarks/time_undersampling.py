"""
Unified time undersampling experiment runner.

This module consolidates time undersampling scripts for different methods:
- GIMME_time_undersampling_data.py
- MVAR_time_undersampling_data.py
- MVGC_time_undersampling_data.py
- slurm_FASK_time_undersampling_data.py
- slurm_GIMME_time_undersampling_data.py
- slurm_MVAR_time_undersampling_data.py
- slurm_MVGC_time_undersampling_data.py

Usage:
    # Local execution
    python time_undersampling.py --method MVGC --networks 1 2 3
    
    # SLURM execution
    python time_undersampling.py --method GIMME --slurm --job-id 1
"""
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.io import loadmat
import random

# Same conditional imports as benchmark_runner
try:
    from py_tetrad.tools import TetradSearch as ts
    HAS_TETRAD = True
except ImportError:
    HAS_TETRAD = False

try:
    import tigramite.data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    HAS_TIGRAMITE = True
except ImportError:
    HAS_TIGRAMITE = False

import networkx as nx
from gunfolds import conversions as cv
from gunfolds.scripts.datasets.simple_networks import simp_nets
from gunfolds.scripts import my_functions as mf
from gunfolds.solvers.clingo_rasl import drasl
from gunfolds.utils import graphkit as gk
from gunfolds.utils import bfutils
from gunfolds.utils import zickle as zkl
from gunfolds.utils.calc_procs import get_process_count

PNUM = 4


# ============================================================================
# DATA LOADING AND METHOD EXECUTION
# ============================================================================

def run_method_on_data(method, data_path, network_GT, **kwargs):
    """
    Run specified method on data and return estimated graph.
    
    Args:
        method: Method name ('MVGC', 'MVAR', 'FASK', 'GIMME', 'PCMCI')
        data_path: Path to data file (or matrix for MVGC/MVAR)
        network_GT: Ground truth network
        **kwargs: Additional method-specific parameters
        
    Returns:
        Estimated graph
    """
    if method == 'MVGC':
        # data_path is actually the mat file path for MVGC
        mat_data = loadmat(data_path)
        mat = mat_data['sig']
        for i in range(len(network_GT)):
            mat[i, i] = 1
        B = np.zeros((len(network_GT), len(network_GT))).astype(int)
        return cv.adjs2graph(mat, np.zeros((len(network_GT), len(network_GT))))
    
    elif method == 'MVAR':
        mat_data = loadmat(data_path)
        mat = mat_data['sig']
        for i in range(len(network_GT)):
            mat[i, i] = 1
        B = np.zeros((len(network_GT), len(network_GT))).astype(int)
        return cv.adjs2graph(mat, np.zeros((len(network_GT), len(network_GT))))
    
    elif method == 'FASK':
        if not HAS_TETRAD:
            raise ImportError("FASK requires py_tetrad")
        
        data = pd.read_csv(data_path, delimiter='\t')
        search = ts.TetradSearch(data)
        search.set_verbose(False)
        search.use_sem_bic()
        search.use_fisher_z(alpha=0.05)
        search.run_fask(alpha=0.05, left_right_rule=1)
        
        graph_string = str(search.get_string())
        nodes = mf.parse_nodes(graph_string)
        edges = mf.parse_edges(graph_string)
        
        adj_matrix = mf.create_adjacency_matrix(edges, nodes)
        for i in range(len(network_GT)):
            adj_matrix[i, i] = 0
        
        B = np.zeros((len(network_GT), len(network_GT))).astype(int)
        return cv.adjs2graph(adj_matrix, np.zeros((len(network_GT), len(network_GT))))
    
    elif method == 'GIMME':
        # GIMME requires pre-processed beta files
        beta_file = data_path  # Assuming data_path points to beta file
        std_error_file = data_path.replace('BetasStd.csv', 'StdErrors.csv')
        
        graph = mf.read_gimme_to_graph(beta_file, std_error_file)
        numeric_graph = mf.convert_nodes_to_numbers(graph)
        numeric_graph_no_selfloops = numeric_graph.copy()
        numeric_graph_no_selfloops.remove_edges_from(nx.selfloop_edges(numeric_graph_no_selfloops))
        
        return gk.nx2graph(numeric_graph_no_selfloops)
    
    elif method == 'PCMCI':
        if not HAS_TIGRAMITE:
            raise ImportError("PCMCI requires tigramite")
        
        data = pd.read_csv(data_path, delimiter='\t')
        dataframe = pp.DataFrame(data.values)
        cond_ind_test = ParCorr()
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05)
        g_estimated, _, _ = cv.Glag2CG(results)
        return mf.remove_bidir_edges(g_estimated)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def run_undersampling_experiment(method, network_num, file_num, concat=True,
                                 apply_rasl=False, output_dir='results'):
    """
    Run time undersampling experiment for a single network/file combination.
    
    Args:
        method: Causal discovery method
        network_num: Network number (1-6)
        file_num: File number (1-60)
        concat: Use concatenated data
        apply_rasl: Apply RASL post-processing
        output_dir: Output directory
        
    Returns:
        Dictionary with results
    """
    print(f"Processing Network {network_num}, File {file_num}")
    
    network_GT = simp_nets(network_num, selfloop=True)
    include_selfloop = False
    
    # Construct data path based on method
    num = str(file_num) if file_num > 9 else '0' + str(file_num)
    
    if method in ['MVGC', 'MVAR']:
        data_prefix = method
        data_path = f'expo_to_mat/{data_prefix}_expo_to_py_n{network_num}_{"concat" if concat else "individual"}/mat_file_{file_num}.mat'
    elif method in ['FASK', 'PCMCI']:
        data_path = os.path.expanduser(
            f'~/DataSets_Feedbacks/1. Simple_Networks/Network{network_num}_amp/'
            f'data_fslfilter_concat/concat_BOLDfslfilter_{num}.txt')
    elif method == 'GIMME':
        data_path = (f"/Users/mabavisani/GSU Dropbox Dropbox/Mohammadsajad Abavisani/Mac/"
                    f"Documents/PhD/Research/code/GIMME/gimme-master/gimme-master/"
                    f"{network_num}_05VARfalse/individual/concat_BOLDfslfilter_{num}BetasStd.csv")
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Run method
    try:
        estimated_graph = run_method_on_data(method, data_path, network_GT)
    except Exception as e:
        print(f"  Error running {method}: {e}")
        return None
    
    # Evaluate against original GT
    metrics_original = mf.precision_recall(estimated_graph, network_GT, 
                                          include_selfloop=include_selfloop)
    
    results = {
        'network': network_num,
        'file': file_num,
        'method': method,
        'metrics_original_GT': metrics_original
    }
    
    # Optionally evaluate against undersampled GT
    # (Testing different undersampling rates)
    all_undersamples = bfutils.all_undersamples(network_GT)
    
    for u_idx, undersampled_GT in enumerate(all_undersamples):
        undersampled_GT_clean = mf.remove_bidir_edges(undersampled_GT)
        metrics_u = mf.precision_recall(estimated_graph, undersampled_GT_clean,
                                       include_selfloop=include_selfloop)
        results[f'metrics_GT_u{u_idx+1}'] = metrics_u
    
    # Optionally apply RASL post-processing
    if apply_rasl:
        try:
            nx_graph = gk.graph2nx(estimated_graph)
            two_cycles = mf.find_two_cycles(nx_graph)
            
            DD = np.ones((len(network_GT), len(network_GT))) * 5000
            BD = np.ones((len(network_GT), len(network_GT))) * 10000
            
            for cycle in two_cycles:
                DD[cycle[0]-1][cycle[1]-1] = 2500
                DD[cycle[1]-1][cycle[0]-1] = 2500
            
            for i in range(len(network_GT)):
                DD[i][i] = 10000
            
            B = np.zeros((len(network_GT), len(network_GT))).astype(int)
            for cycle in two_cycles:
                B[cycle[0]-1][cycle[1]-1] = 1
                B[cycle[1]-1][cycle[0]-1] = 1
            
            estimated_graph_bidir = cv.adjs2graph(
                estimated_graph[0], B) if isinstance(estimated_graph, tuple) else estimated_graph
            
            edge_weights = [1, 3, 1, 3, 2]
            r_estimated = drasl([estimated_graph_bidir], weighted=True, capsize=0,
                               urate=min(5, (3 * len(network_GT) + 1)),
                               scc=False,
                               dm=[DD],
                               bdm=[BD],
                               GT_density=int(1000 * gk.density(network_GT)),
                               edge_weights=edge_weights, pnum=PNUM, optim='optN')
            
            # Find best RASL solution
            max_f1_score = 0
            best_rasl_graph = None
            
            for answer in r_estimated:
                res_rasl = bfutils.num2CG(answer[0][0], len(network_GT))
                rasl_metrics = mf.precision_recall(res_rasl, network_GT,
                                                   include_selfloop=include_selfloop)
                
                curr_f1 = (rasl_metrics['orientation']['F1'] + 
                          rasl_metrics['adjacency']['F1'] + 
                          rasl_metrics['cycle']['F1'])
                
                if curr_f1 > max_f1_score:
                    max_f1_score = curr_f1
                    best_rasl_graph = res_rasl
                    results['metrics_rasl'] = rasl_metrics
            
            print(f"  RASL improved F1: {max_f1_score:.3f}")
            
        except Exception as e:
            print(f"  Warning: RASL post-processing failed: {e}")
    
    return results


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def run_batch_experiments(method, networks=[1,2,3,4,5,6], num_files=60,
                         concat=True, apply_rasl=False, output_dir='results'):
    """Run experiments across multiple networks and files"""
    
    print(f"\n{'='*60}")
    print(f"Time Undersampling Experiments: {method}")
    print(f"Networks: {networks}, Files: {num_files} per network")
    print(f"{'='*60}\n")
    
    all_results = []
    
    for nn in networks:
        for fl in range(1, num_files + 1):
            result = run_undersampling_experiment(
                method=method,
                network_num=nn,
                file_num=fl,
                concat=concat,
                apply_rasl=apply_rasl,
                output_dir=output_dir
            )
            
            if result:
                all_results.append(result)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_file = f"{output_dir}/{method}_time_undersampling_{timestamp}.zkl"
    zkl.save(all_results, output_file)
    
    print(f"\n✓ Experiments complete!")
    print(f"  Total: {len(all_results)} experiments")
    print(f"  Saved to: {output_file}")
    
    return all_results


# ============================================================================
# SLURM SUPPORT
# ============================================================================

def slurm_job(method, job_id, networks=[1,2,3,4,5,6], files_per_network=60,
             concat=True, apply_rasl=False, output_dir='results'):
    """
    Run a single SLURM job.
    
    Job ID maps to (network_num, file_num) combinations.
    """
    # Calculate which network and file from job_id
    total_per_network = files_per_network
    network_idx = (job_id - 1) // total_per_network
    file_num = ((job_id - 1) % total_per_network) + 1
    
    if network_idx >= len(networks):
        print(f"Job ID {job_id} out of range")
        return
    
    network_num = networks[network_idx]
    
    print(f"SLURM Job {job_id}: Network {network_num}, File {file_num}")
    
    result = run_undersampling_experiment(
        method=method,
        network_num=network_num,
        file_num=file_num,
        concat=concat,
        apply_rasl=apply_rasl,
        output_dir=output_dir
    )
    
    # Save individual result
    os.makedirs(f"{output_dir}/individual_jobs", exist_ok=True)
    output_file = f"{output_dir}/individual_jobs/{method}_job{job_id}_n{network_num}_f{file_num}.zkl"
    zkl.save(result, output_file)
    
    print(f"  Saved to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified time undersampling experiment runner')
    
    parser.add_argument('--method', required=True,
                       choices=['MVGC', 'MVAR', 'FASK', 'GIMME', 'PCMCI'],
                       help='Causal discovery method')
    parser.add_argument('--networks', nargs='+', type=int, default=[1,2,3,4,5,6],
                       help='Network numbers to test')
    parser.add_argument('--num-files', type=int, default=60,
                       help='Number of files per network')
    parser.add_argument('--concat', action='store_true', default=True,
                       help='Use concatenated data')
    parser.add_argument('--apply-rasl', action='store_true',
                       help='Apply RASL post-processing')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory')
    
    # SLURM-specific options
    parser.add_argument('--slurm', action='store_true',
                       help='Run in SLURM mode (single job)')
    parser.add_argument('--job-id', type=int,
                       help='SLURM job ID (required if --slurm)')
    
    args = parser.parse_args()
    
    if args.slurm:
        if args.job_id is None:
            parser.error("--job-id required when using --slurm")
        
        slurm_job(
            method=args.method,
            job_id=args.job_id,
            networks=args.networks,
            files_per_network=args.num_files,
            concat=args.concat,
            apply_rasl=args.apply_rasl,
            output_dir=args.output_dir
        )
    else:
        run_batch_experiments(
            method=args.method,
            networks=args.networks,
            num_files=args.num_files,
            concat=args.concat,
            apply_rasl=args.apply_rasl,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()

