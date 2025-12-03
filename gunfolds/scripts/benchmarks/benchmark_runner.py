"""
Unified benchmark runner for method comparison experiments (Fig 4).

This module consolidates benchmark scripts for different causal discovery methods:
- FASK_fig4.py
- GIMME_fig4.py
- MVAR_fig4.py
- MVGC_fig4.py
- PCMCI_fig4.py
- RASL_fig4.py

Usage:
    python benchmark_runner.py --method MVGC --networks 1 2 3 4 5 6 --concat
    python benchmark_runner.py --method GIMME --networks 1 2 3 4 5 6 --plot-only
    python benchmark_runner.py --method FASK --networks 3 --output results/
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.io import loadmat
from datetime import datetime

# Conditional imports for specific methods
try:
    from py_tetrad.tools import TetradSearch as ts
    HAS_TETRAD = True
except ImportError:
    HAS_TETRAD = False
    print("Warning: py_tetrad not available (needed for FASK)")

try:
    import tigramite.data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    HAS_TIGRAMITE = True
except ImportError:
    HAS_TIGRAMITE = False
    print("Warning: tigramite not available (needed for PCMCI)")

import networkx as nx
from gunfolds import conversions as cv
from gunfolds.scripts.datasets.simple_networks import simp_nets
from gunfolds.scripts import my_functions as mf
from gunfolds.utils import zickle as zkl


# ============================================================================
# DATA LOADERS FOR EACH METHOD
# ============================================================================

def load_data_mvgc(network_num, file_num, concat=True):
    """Load MVGC results from matlab files"""
    folder_read = f'expo_to_mat/MVGC_expo_to_py_n{network_num}_{"concat" if concat else "individual"}'
    mat_data = loadmat(f'{folder_read}/mat_file_{file_num}.mat')
    mat = mat_data['sig']
    network_GT = simp_nets(network_num, selfloop=True)
    
    # Add self-loops
    for i in range(len(network_GT)):
        mat[i, i] = 1
    
    B = np.zeros((len(network_GT), len(network_GT))).astype(int)
    estimated_graph = cv.adjs2graph(mat, np.zeros((len(network_GT), len(network_GT))))
    
    return estimated_graph, network_GT


def load_data_mvar(network_num, file_num, concat=True):
    """Load MVAR results from matlab files"""
    folder_read = f'expo_to_mat/MVAR_expo_to_py_n{network_num}_{"concat" if concat else "individual"}_new1'
    mat_data = loadmat(f'{folder_read}/mat_file_{file_num}.mat')
    mat = mat_data['sig']
    network_GT = simp_nets(network_num, selfloop=True)
    
    # Add self-loops
    for i in range(len(network_GT)):
        mat[i, i] = 1
    
    B = np.zeros((len(network_GT), len(network_GT))).astype(int)
    estimated_graph = cv.adjs2graph(mat, np.zeros((len(network_GT), len(network_GT))))
    
    return estimated_graph, network_GT


def load_data_fask(network_num, file_num, concat=True):
    """Load data and run FASK algorithm"""
    if not HAS_TETRAD:
        raise ImportError("FASK requires py_tetrad package")
    
    num = str(file_num) if file_num > 9 else '0' + str(file_num)
    path = os.path.expanduser(
        f'~/DataSets_Feedbacks/1. Simple_Networks/Network{network_num}_amp/'
        f'data_fslfilter_concat/concat_BOLDfslfilter_{num}.txt')
    
    data = pd.read_csv(path, delimiter='\t')
    network_GT = simp_nets(network_num, True)
    
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
    estimated_graph = cv.adjs2graph(adj_matrix, np.zeros((len(network_GT), len(network_GT))))
    
    return estimated_graph, network_GT


def load_data_gimme(network_num, file_num, concat=True):
    """Load GIMME results from beta and standard error files"""
    num = str(file_num) if file_num > 9 else '0' + str(file_num)
    
    beta_file = (f"/Users/mabavisani/GSU Dropbox Dropbox/Mohammadsajad Abavisani/Mac/"
                f"Documents/PhD/Research/code/GIMME/gimme-master/gimme-master/"
                f"{network_num}_05VARfalse/individual/concat_BOLDfslfilter_{num}BetasStd.csv")
    
    std_error_file = (f"/Users/mabavisani/GSU Dropbox Dropbox/Mohammadsajad Abavisani/Mac/"
                     f"Documents/PhD/Research/code/GIMME/gimme-master/gimme-master/"
                     f"{network_num}_05VARfalse/individual/StdErrors/concat_BOLDfslfilter_{num}StdErrors.csv")
    
    network_GT = simp_nets(network_num, True)
    graph = mf.read_gimme_to_graph(beta_file, std_error_file)
    numeric_graph = mf.convert_nodes_to_numbers(graph)
    numeric_graph_no_selfloops = numeric_graph.copy()
    numeric_graph_no_selfloops.remove_edges_from(nx.selfloop_edges(numeric_graph_no_selfloops))
    
    from gunfolds.utils import graphkit as gk
    estimated_graph = gk.nx2graph(numeric_graph_no_selfloops)
    
    return estimated_graph, network_GT


def load_data_pcmci(network_num, file_num, concat=True):
    """Load data and run PCMCI algorithm"""
    if not HAS_TIGRAMITE:
        raise ImportError("PCMCI requires tigramite package")
    
    num = str(file_num) if file_num > 9 else '0' + str(file_num)
    path = os.path.expanduser(
        f'~/DataSets_Feedbacks/1. Simple_Networks/Network{network_num}_amp/'
        f'data_fslfilter_concat/concat_BOLDfslfilter_{num}.txt')
    
    data = pd.read_csv(path, delimiter='\t')
    network_GT = simp_nets(network_num, True)
    
    dataframe = pp.DataFrame(data.values)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05)
    g_estimated, _, _ = cv.Glag2CG(results)
    estimated_graph = mf.remove_bidir_edges(g_estimated)
    
    return estimated_graph, network_GT


def load_data_rasl(network_num, file_num, concat=True):
    """Load saved RASL results"""
    # RASL typically runs on top of other methods, so this loads pre-computed results
    # This is a placeholder - actual implementation depends on where RASL results are stored
    raise NotImplementedError("RASL loading not yet implemented - run RASL experiments first")


# ============================================================================
# METHOD REGISTRY
# ============================================================================

METHOD_LOADERS = {
    'MVGC': load_data_mvgc,
    'MVAR': load_data_mvar,
    'FASK': load_data_fask,
    'GIMME': load_data_gimme,
    'PCMCI': load_data_pcmci,
    'RASL': load_data_rasl,
}

# Baseline data reported by Ruben et al. for comparison
RUBEN_BASELINES = {
    'GIMME': {'P_O': 0.66, 'R_O': 0.5, 'P_A': 0.91, 'R_A': 0.88, 'P_C': 0.12, 'R_C': 0.09},
    'MVAR': {'P_O': 0.53, 'R_O': 0.78, 'P_A': 0.79, 'R_A': 0.92, 'P_C': 0.22, 'R_C': 0.62},
    # Add other baselines as needed
}


# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================

def run_benchmark(method, networks=[1,2,3,4,5,6], num_files=60, concat=True, 
                 include_selfloop=False, output_dir='results'):
    """
    Run benchmark for specified method across networks.
    
    Args:
        method: Method name ('MVGC', 'MVAR', 'FASK', 'GIMME', 'PCMCI', 'RASL')
        networks: List of network numbers to test
        num_files: Number of data files per network
        concat: Use concatenated data
        include_selfloop: Include self-loops in evaluation
        output_dir: Directory to save results
        
    Returns:
        Dictionary with precision, recall, F1 scores for all metrics
    """
    if method not in METHOD_LOADERS:
        raise ValueError(f"Unknown method: {method}. Choose from {list(METHOD_LOADERS.keys())}")
    
    loader_func = METHOD_LOADERS[method]
    
    print(f"\n{'='*60}")
    print(f"Running benchmark for {method}")
    print(f"Networks: {networks}, Files per network: {num_files}")
    print(f"{'='*60}\n")
    
    # Initialize result storage
    results = {
        'orientation': {'precision': [], 'recall': [], 'F1': []},
        'adjacency': {'precision': [], 'recall': [], 'F1': []},
        'cycle': {'precision': [], 'recall': [], 'F1': []}
    }
    
    # Run experiments
    for nn in networks:
        print(f"Processing Network {nn}...")
        for fl in range(1, num_files + 1):
            try:
                estimated_graph, network_GT = loader_func(nn, fl, concat)
                
                # Compute metrics
                metrics = mf.precision_recall(estimated_graph, network_GT, 
                                             include_selfloop=include_selfloop)
                
                # Store results
                for metric_type in ['orientation', 'adjacency', 'cycle']:
                    results[metric_type]['precision'].append(metrics[metric_type]['precision'])
                    results[metric_type]['recall'].append(metrics[metric_type]['recall'])
                    results[metric_type]['F1'].append(metrics[metric_type]['F1'])
                
                if fl % 10 == 0:
                    print(f"  Processed {fl}/{num_files} files")
                    
            except Exception as e:
                print(f"  Warning: Error processing Network {nn}, File {fl}: {e}")
                continue
    
    print(f"\n✓ Benchmark completed for {method}")
    print(f"  Total samples: {len(results['orientation']['F1'])}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as zkl
    zkl_file = f"{output_dir}/{method}_benchmark_{timestamp}.zkl"
    zkl.save(results, zkl_file)
    print(f"  Saved to: {zkl_file}")
    
    return results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_benchmark_results(results, method, baseline=None, output_path=None):
    """
    Create boxplot visualization of benchmark results.
    
    Args:
        results: Dictionary with precision, recall, F1 scores
        method: Method name for title
        baseline: Optional baseline data for comparison (dict with P_O, R_O, etc.)
        output_path: Path to save figure
    """
    # Prepare data
    data_method = [
        [results['orientation']['precision'], results['orientation']['recall'], 
         results['orientation']['F1']],
        [results['adjacency']['precision'], results['adjacency']['recall'], 
         results['adjacency']['F1']],
        [results['cycle']['precision'], results['cycle']['recall'], 
         results['cycle']['F1']]
    ]
    
    # Prepare baseline if provided
    if baseline:
        n_samples = len(results['orientation']['F1'])
        data_baseline = [
            [[random.uniform(baseline['P_O']-0.06, baseline['P_O']+0.06) for _ in range(n_samples)],
             [random.uniform(baseline['R_O']-0.06, baseline['R_O']+0.06) for _ in range(n_samples)],
             [random.uniform(mf.calculate_f1_score(baseline['P_O'], baseline['R_O'])-0.06,
                            mf.calculate_f1_score(baseline['P_O'], baseline['R_O'])+0.06) for _ in range(n_samples)]],
            [[random.uniform(baseline['P_A']-0.06, baseline['P_A']+0.06) for _ in range(n_samples)],
             [random.uniform(baseline['R_A']-0.06, baseline['R_A']+0.06) for _ in range(n_samples)],
             [random.uniform(mf.calculate_f1_score(baseline['P_A'], baseline['R_A'])-0.06,
                            mf.calculate_f1_score(baseline['P_A'], baseline['R_A'])+0.06) for _ in range(n_samples)]],
            [[random.uniform(baseline['P_C']-0.06, baseline['P_C']+0.06) for _ in range(n_samples)],
             [random.uniform(baseline['R_C']-0.06, baseline['R_C']+0.06) for _ in range(n_samples)],
             [random.uniform(mf.calculate_f1_score(baseline['P_C'], baseline['R_C'])-0.06,
                            mf.calculate_f1_score(baseline['P_C'], baseline['R_C'])+0.06) for _ in range(n_samples)]]
        ]
    else:
        data_baseline = None
    
    # Create figure
    titles = ['Orientation', 'Adjacency', '2 Cycles']
    colors = ['gray', 'blue'] if data_baseline else ['blue']
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    
    for i, (data_m, title) in enumerate(zip(data_method, titles)):
        ax = axes[i]
        
        bplots = []
        
        # Plot baseline if available
        if data_baseline:
            data_b = data_baseline[i]
            bplots.append(
                ax.boxplot(data_b, positions=np.array(range(len(data_b))) * 2.0 - 0.6,
                          patch_artist=True, showmeans=True, widths=0.2))
        
        # Plot method results
        offset = -0.4 if data_baseline else 0
        bplots.append(
            ax.boxplot(data_m, positions=np.array(range(len(data_m))) * 2.0 + offset,
                      patch_artist=True, showmeans=True, widths=0.2))
        
        # Set colors
        for bplot, color in zip(bplots, colors):
            for patch in bplot['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        
        # Plot individual points
        if data_baseline:
            for j in range(len(data_baseline[i])):
                jitter = np.random.uniform(-0.05, 0.05, size=len(data_baseline[i][j]))
                ax.plot(np.ones_like(data_baseline[i][j]) * (j * 2.0 - 0.6) + jitter,
                       data_baseline[i][j], 'o', color='black', alpha=0.5, markersize=3)
        
        for j in range(len(data_m)):
            jitter = np.random.uniform(-0.05, 0.05, size=len(data_m[j]))
            ax.plot(np.ones_like(data_m[j]) * (j * 2.0 + offset) + jitter,
                   data_m[j], 'o', color='black', alpha=0.5, markersize=3)
        
        # Labels and formatting
        ax.set_xticks(range(0, len(data_m) * 2, 2))
        ax.set_xticklabels(['Precision', 'Recall', 'F1-score'])
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title(f'({title})')
        ax.grid(True)
        ax.set_ylim(0, 1)
    
    # Super title
    plt.suptitle(f'{method} - Networks 1-6 (All data)')
    
    # Legend
    if data_baseline:
        gray_patch = mpatches.Patch(color='gray', label='Baseline (Ruben et al.)')
        blue_patch = mpatches.Patch(color='blue', label=f'{method} Results')
        plt.legend(handles=[gray_patch, blue_patch], loc='upper right')
    else:
        blue_patch = mpatches.Patch(color='blue', label=f'{method} Results')
        plt.legend(handles=[blue_patch], loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to: {output_path}")
    
    plt.show()
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified benchmark runner for causal discovery methods')
    parser.add_argument('--method', required=True, 
                       choices=['MVGC', 'MVAR', 'FASK', 'GIMME', 'PCMCI', 'RASL'],
                       help='Causal discovery method to benchmark')
    parser.add_argument('--networks', nargs='+', type=int, default=[1,2,3,4,5,6],
                       help='Network numbers to test (default: 1 2 3 4 5 6)')
    parser.add_argument('--num-files', type=int, default=60,
                       help='Number of data files per network (default: 60)')
    parser.add_argument('--concat', action='store_true', default=True,
                       help='Use concatenated data (default: True)')
    parser.add_argument('--include-selfloop', action='store_true', default=False,
                       help='Include self-loops in evaluation')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for results (default: results/)')
    parser.add_argument('--plot-only', type=str, metavar='ZKL_FILE',
                       help='Skip benchmark, only plot from saved .zkl file')
    parser.add_argument('--baseline', action='store_true',
                       help='Include baseline comparison in plot')
    args = parser.parse_args()
    
    if args.plot_only:
        # Load and plot existing results
        print(f"Loading results from: {args.plot_only}")
        results = zkl.load(args.plot_only)
        
        baseline = RUBEN_BASELINES.get(args.method) if args.baseline else None
        output_plot = args.plot_only.replace('.zkl', '_plot.png')
        
        plot_benchmark_results(results, args.method, baseline=baseline, 
                             output_path=output_plot)
    else:
        # Run benchmark
        results = run_benchmark(
            method=args.method,
            networks=args.networks,
            num_files=args.num_files,
            concat=args.concat,
            include_selfloop=args.include_selfloop,
            output_dir=args.output_dir
        )
        
        # Generate plot
        baseline = RUBEN_BASELINES.get(args.method) if args.baseline else None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_plot = f"{args.output_dir}/{args.method}_benchmark_{timestamp}_plot.png"
        
        plot_benchmark_results(results, args.method, baseline=baseline, 
                             output_path=output_plot)
        
        print(f"\n{'='*60}")
        print(f"✓ Benchmark complete!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

