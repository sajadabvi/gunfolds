"""
Analyze saved GCM results from gcm_on_ICA.py

This script loads and analyzes Granger Causality Mapping results saved by gcm_on_ICA.py
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import glob

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze saved GCM experiment results')
    parser.add_argument("-d", "--directory", default=None,
                        help="Directory containing saved results (e.g., gcm_roebroeck/12262025103045)")
    parser.add_argument("-t", "--timestamp", default=None,
                        help="Timestamp folder to analyze (e.g., 12262025103045). If not specified, uses most recent.")
    parser.add_argument("--plot", action='store_true',
                        help="Generate plots")
    parser.add_argument("--threshold", default=0.0, type=float,
                        help="Edge frequency threshold for display (0.0-1.0)")
    parser.add_argument("--aggregate", action='store_true',
                        help="Aggregate per-subject results (useful for parallel runs)")
    parser.add_argument("--expected-subjects", default=None, type=int,
                        help="Expected number of subjects (for progress tracking in parallel runs)")
    parser.add_argument("--watch", action='store_true',
                        help="Watch mode: continuously update aggregation as new results arrive")
    parser.add_argument("--watch-interval", default=30, type=int,
                        help="Seconds between updates in watch mode (default: 30)")
    return parser.parse_args()

def find_latest_results():
    """Find the most recent timestamped results folder"""
    base_dir = "gcm_roebroeck"
    if not os.path.exists(base_dir):
        return None
    
    # List all subdirectories that look like timestamps (14 digits)
    subdirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.isdigit() and len(item) == 14:
            subdirs.append((item, item_path))
    
    if not subdirs:
        return None
    
    # Sort by timestamp (most recent first)
    subdirs.sort(reverse=True)
    return subdirs[0][1]  # Return path to most recent

def aggregate_subject_results(csv_dir, node_names=None):
    """
    Aggregate results from individual subject CSV files.
    Useful for parallel runs where group files don't exist yet.
    """
    # Find all subject adjacency CSVs
    adj_files = sorted(glob.glob(os.path.join(csv_dir, '*Adj_labeled.csv')))
    fdiff_files = sorted(glob.glob(os.path.join(csv_dir, '*Fdiff_labeled.csv')))
    
    if not adj_files:
        return None
    
    # Load first file to get structure
    first_adj = pd.read_csv(adj_files[0], index_col=0)
    if node_names is None:
        node_names = list(first_adj.columns)
    n_nodes = len(node_names)
    
    # Initialize accumulators
    hits = np.zeros((n_nodes, n_nodes), dtype=int)
    fdiff_sum = np.zeros((n_nodes, n_nodes), dtype=float)
    
    # Accumulate across subjects
    n_processed = 0
    for adj_file in adj_files:
        try:
            adj_df = pd.read_csv(adj_file, index_col=0)
            adj_df = adj_df.loc[node_names, node_names]  # Reorder if needed
            hits += adj_df.to_numpy().astype(int)
            n_processed += 1
        except Exception as e:
            print(f"Warning: Could not load {adj_file}: {e}")
    
    for fdiff_file in fdiff_files:
        try:
            fdiff_df = pd.read_csv(fdiff_file, index_col=0)
            fdiff_df = fdiff_df.loc[node_names, node_names]
            fdiff_sum += fdiff_df.to_numpy()
        except Exception as e:
            print(f"Warning: Could not load {fdiff_file}: {e}")
    
    if n_processed == 0:
        return None
    
    # Calculate rates and means
    rate = hits / float(n_processed)
    fdiff_mean = fdiff_sum / float(n_processed)
    
    # Create DataFrames
    results = {
        'hits': pd.DataFrame(hits, index=node_names, columns=node_names),
        'rate': pd.DataFrame(rate, index=node_names, columns=node_names),
        'fdiff': pd.DataFrame(fdiff_mean, index=node_names, columns=node_names),
        'n_subjects': n_processed,
        'subject_files': adj_files,
        'aggregated': True
    }
    
    return results

def load_results(directory, force_aggregate=False):
    """Load GCM results from specified directory"""
    csv_dir = os.path.join(directory, "csv")
    
    if not os.path.exists(csv_dir):
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    
    # Load group-level results if they exist
    group_hits_file = os.path.join(csv_dir, "group_edge_hits.csv")
    group_rate_file = os.path.join(csv_dir, "group_edge_rate.csv")
    group_fdiff_file = os.path.join(csv_dir, "group_Fdiff_mean.csv")
    
    results = {}
    has_group_files = (os.path.exists(group_hits_file) and 
                       os.path.exists(group_rate_file) and 
                       os.path.exists(group_fdiff_file))
    
    if has_group_files and not force_aggregate:
        # Load pre-computed group results
        results['hits'] = pd.read_csv(group_hits_file, index_col=0)
        results['rate'] = pd.read_csv(group_rate_file, index_col=0)
        results['fdiff'] = pd.read_csv(group_fdiff_file, index_col=0)
        results['aggregated'] = False
    else:
        # Aggregate from individual subject files
        print("Aggregating results from individual subject files...")
        agg_results = aggregate_subject_results(csv_dir)
        if agg_results is None:
            raise ValueError(f"No subject results found in {csv_dir}")
        results.update(agg_results)
    
    # Load run parameters if available
    params_file = os.path.join(directory, "run_params.csv")
    if os.path.exists(params_file):
        results['params'] = pd.read_csv(params_file)
    
    # Count subject files if not already done
    if 'n_subjects' not in results:
        subject_files = sorted(glob.glob(os.path.join(csv_dir, '*Adj_labeled.csv')))
        results['n_subjects'] = len(subject_files)
        results['subject_files'] = subject_files
    
    return results

def plot_group_edge_frequency(edge_rate, node_names, output_path):
    """
    Create group-level edge frequency plot (same style as gcm_on_ICA.py)
    """
    N = len(node_names)
    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i, label=node_names[i])
    
    edge_weight = edge_rate
    W = edge_weight / (edge_weight.max() + 1e-12)
    
    for i in range(N):
        for j in range(N):
            if i != j and edge_weight[i, j] > 0:
                G.add_edge(i, j, weight=float(W[i, j]), raw=float(edge_weight[i, j]))
    
    # Fixed circle positions
    angles = np.linspace(-np.pi/2, 3*np.pi/2, N, endpoint=False)
    pos = {i: (float(np.cos(a)), float(np.sin(a))) for i, a in enumerate(angles)}
    
    plt.figure(figsize=(6, 5))
    NODE_SIZE = 700
    nx.draw_networkx_nodes(G, pos, node_size=NODE_SIZE)
    nx.draw_networkx_labels(G, pos, labels={k: node_names[k] for k in range(N)}, font_size=9)
    
    # Edge widths
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    if weights:
        w_min, w_max = min(weights), max(weights)
        if w_min == w_max:
            widths = [10.0] * len(weights)
        else:
            min_thick, max_thick = 1, 10
            k = 5
            denom = np.exp(k) - 1.0
            widths = []
            for w in weights:
                t = (w - w_min) / (w_max - w_min)
                s = (np.exp(k * t) - 1.0) / denom
                widths.append(min_thick + (max_thick - min_thick) * s)
    else:
        widths = []
    
    max_w = max(widths) if widths else 1.0
    arrowsize = int(max(18, np.ceil(2.5 * max_w)))
    margin = 2.0
    
    widths_map = {edge: w for edge, w in zip(G.edges(), widths)}
    drawn = set()
    for (u, v) in G.edges():
        if (u, v) in drawn:
            continue
        has_back = G.has_edge(v, u)
        if has_back:
            w_uv = widths_map.get((u, v), 1.0)
            w_vu = widths_map.get((v, u), 1.0)
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=w_uv, arrows=True,
                                   arrowstyle='-|>', arrowsize=arrowsize,
                                   min_source_margin=margin, min_target_margin=margin,
                                   connectionstyle='arc3,rad=0.25')
            nx.draw_networkx_edges(G, pos, edgelist=[(v, u)], width=w_vu, arrows=True,
                                   arrowstyle='-|>', arrowsize=arrowsize,
                                   min_source_margin=margin, min_target_margin=margin,
                                   connectionstyle='arc3,rad=0.25')
            drawn.add((u, v))
            drawn.add((v, u))
        else:
            w_uv = widths_map.get((u, v), 1.0)
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=w_uv, arrows=True,
                                   arrowstyle='-|>', arrowsize=arrowsize,
                                   min_source_margin=margin, min_target_margin=margin,
                                   connectionstyle='arc3,rad=0.0')
    
    plt.title("Group GCM: edge frequency")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def save_aggregated_results(results, directory):
    """Save aggregated results to group CSV files"""
    csv_dir = os.path.join(directory, "csv")
    grp_prefix = os.path.join(csv_dir, "group")
    
    if 'hits' in results:
        results['hits'].to_csv(grp_prefix + "_edge_hits.csv")
    if 'rate' in results:
        results['rate'].to_csv(grp_prefix + "_edge_rate.csv")
    if 'fdiff' in results:
        results['fdiff'].to_csv(grp_prefix + "_Fdiff_mean.csv")
    
    print(f"Saved aggregated results to: {csv_dir}/group_*.csv")
    
    # Also create the group plot
    if 'rate' in results:
        fig_dir = os.path.join(directory, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        grp_png = os.path.join(fig_dir, "group_edge_frequency.png")
        
        node_names = list(results['rate'].columns)
        plot_group_edge_frequency(results['rate'].to_numpy(), node_names, grp_png)
        print(f"Saved group plot to: {grp_png}")

def print_statistics(results, threshold=0.0, expected_subjects=None):
    """Print summary statistics"""
    print("=" * 80)
    print("GCM RESULTS ANALYSIS")
    print("=" * 80)
    
    if 'params' in results:
        print("\nRun Parameters:")
        for _, row in results['params'].iterrows():
            print(f"  {row['parameter']}: {row['value']}")
    
    # Show subject progress
    n_subjects = results['n_subjects']
    if expected_subjects is not None:
        progress_pct = (n_subjects / expected_subjects) * 100
        print(f"\nSubject Progress: {n_subjects}/{expected_subjects} ({progress_pct:.1f}%)")
        if n_subjects < expected_subjects:
            print(f"  ⚠️  Warning: Only {n_subjects} of {expected_subjects} subjects processed")
            print(f"  Analysis is based on partial data")
    else:
        print(f"\nNumber of subjects: {n_subjects}")
    
    if results.get('aggregated', False):
        print("  (Aggregated from individual subject files)")
    
    if 'rate' in results:
        rate_df = results['rate']
        nodes = list(rate_df.columns)
        n_nodes = len(nodes)
        
        print(f"Number of nodes: {n_nodes}")
        print(f"Node names: {', '.join(nodes)}")
        
        # Edge statistics
        rate_matrix = rate_df.to_numpy()
        np.fill_diagonal(rate_matrix, 0)  # Exclude self-loops
        
        total_possible = n_nodes * (n_nodes - 1)
        edges_above_threshold = np.sum(rate_matrix > threshold)
        
        print(f"\n--- Edge Frequency Statistics ---")
        print(f"Threshold: {threshold:.2%}")
        print(f"Total possible edges: {total_possible}")
        print(f"Edges above threshold: {edges_above_threshold} ({edges_above_threshold/total_possible:.1%})")
        
        if edges_above_threshold > 0:
            rates_above = rate_matrix[rate_matrix > threshold]
            print(f"Mean frequency (above threshold): {rates_above.mean():.2%}")
            print(f"Max frequency: {rate_matrix.max():.2%}")
            print(f"Median frequency (above threshold): {np.median(rates_above):.2%}")
        
        # Top edges
        print(f"\n--- Top 10 Most Frequent Edges ---")
        edges_list = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and rate_matrix[i, j] > 0:
                    edges_list.append((nodes[i], nodes[j], rate_matrix[i, j]))
        
        edges_list.sort(key=lambda x: x[2], reverse=True)
        for src, dst, freq in edges_list[:10]:
            if 'hits' in results:
                hits = int(results['hits'].loc[src, dst])
                print(f"  {src:8s} → {dst:8s}: {freq:6.1%} ({hits}/{results['n_subjects']} subjects)")
            else:
                print(f"  {src:8s} → {dst:8s}: {freq:6.1%}")
        
        # Node-wise statistics
        print(f"\n--- Per-Node Statistics ---")
        out_degree = np.sum(rate_matrix > threshold, axis=1)
        in_degree = np.sum(rate_matrix > threshold, axis=0)
        
        for i, node in enumerate(nodes):
            out_edges = out_degree[i]
            in_edges = in_degree[i]
            print(f"  {node:8s}: Out={out_edges} In={in_edges} (Total={out_edges+in_edges})")
    
    if 'fdiff' in results:
        fdiff_df = results['fdiff']
        fdiff_matrix = fdiff_df.to_numpy()
        np.fill_diagonal(fdiff_matrix, 0)
        
        print(f"\n--- F-statistic Difference Statistics ---")
        print(f"Mean Fdiff: {fdiff_matrix[fdiff_matrix != 0].mean():.4f}")
        print(f"Max Fdiff: {fdiff_matrix.max():.4f}")
        print(f"Min Fdiff: {fdiff_matrix[fdiff_matrix != 0].min():.4f}")

def plot_edge_frequency_histogram(results, output_dir):
    """Plot histogram of edge frequencies"""
    if 'rate' not in results:
        print("Warning: No rate data available for histogram")
        return
    
    rate_df = results['rate']
    rate_matrix = rate_df.to_numpy()
    np.fill_diagonal(rate_matrix, 0)
    
    # Get non-zero frequencies
    frequencies = rate_matrix[rate_matrix > 0]
    
    if len(frequencies) == 0:
        print("Warning: No edges found for histogram")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(frequencies, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Edge Frequency')
    plt.ylabel('Count')
    plt.title('Distribution of Edge Frequencies Across Subjects')
    
    stats_text = f"Mean: {frequencies.mean():.2%}\nMedian: {np.median(frequencies):.2%}\nMax: {frequencies.max():.2%}"
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'edge_frequency_histogram.pdf'))
    plt.savefig(os.path.join(output_dir, 'edge_frequency_histogram.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/edge_frequency_histogram.pdf")

def plot_heatmap(results, output_dir, threshold=0.0):
    """Plot heatmap of edge frequencies"""
    if 'rate' not in results:
        print("Warning: No rate data available for heatmap")
        return
    
    rate_df = results['rate']
    rate_matrix = rate_df.to_numpy()
    nodes = list(rate_df.columns)
    
    # Apply threshold
    rate_thresholded = rate_matrix.copy()
    rate_thresholded[rate_thresholded < threshold] = 0
    
    plt.figure(figsize=(8, 7))
    im = plt.imshow(rate_thresholded, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    
    plt.colorbar(im, label='Edge Frequency')
    plt.xticks(range(len(nodes)), nodes, rotation=45, ha='right')
    plt.yticks(range(len(nodes)), nodes)
    plt.xlabel('Target')
    plt.ylabel('Source')
    plt.title(f'GCM Edge Frequencies (threshold={threshold:.2%})')
    
    # Add text annotations
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if rate_thresholded[i, j] > 0:
                text_color = 'white' if rate_thresholded[i, j] > 0.5 else 'black'
                plt.text(j, i, f'{rate_thresholded[i, j]:.2f}',
                        ha='center', va='center', color=text_color, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'edge_frequency_heatmap.pdf'))
    plt.savefig(os.path.join(output_dir, 'edge_frequency_heatmap.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/edge_frequency_heatmap.pdf")

def plot_fdiff_heatmap(results, output_dir):
    """Plot heatmap of F-statistic differences"""
    if 'fdiff' not in results:
        print("Warning: No Fdiff data available for heatmap")
        return
    
    fdiff_df = results['fdiff']
    fdiff_matrix = fdiff_df.to_numpy()
    nodes = list(fdiff_df.columns)
    
    plt.figure(figsize=(8, 7))
    vmax = np.percentile(fdiff_matrix[fdiff_matrix > 0], 95) if np.any(fdiff_matrix > 0) else 1
    im = plt.imshow(fdiff_matrix, cmap='viridis', vmin=0, vmax=vmax, aspect='auto')
    
    plt.colorbar(im, label='Mean F-statistic Difference')
    plt.xticks(range(len(nodes)), nodes, rotation=45, ha='right')
    plt.yticks(range(len(nodes)), nodes)
    plt.xlabel('Target')
    plt.ylabel('Source')
    plt.title('GCM Mean F-statistic Differences')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fdiff_heatmap.pdf'))
    plt.savefig(os.path.join(output_dir, 'fdiff_heatmap.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/fdiff_heatmap.pdf")

def compare_with_threshold(results, thresholds, output_dir):
    """Compare network density at different thresholds"""
    if 'rate' not in results:
        print("Warning: No rate data available for threshold comparison")
        return
    
    rate_df = results['rate']
    rate_matrix = rate_df.to_numpy()
    np.fill_diagonal(rate_matrix, 0)
    
    n_nodes = len(rate_df.columns)
    total_possible = n_nodes * (n_nodes - 1)
    
    densities = []
    n_edges = []
    
    for thresh in thresholds:
        n_edge = np.sum(rate_matrix > thresh)
        n_edges.append(n_edge)
        densities.append(n_edge / total_possible)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot number of edges
    ax1.plot(thresholds, n_edges, marker='o', linewidth=2)
    ax1.set_xlabel('Frequency Threshold')
    ax1.set_ylabel('Number of Edges')
    ax1.set_title('Edges vs Threshold')
    ax1.grid(True, alpha=0.3)
    
    # Plot density
    ax2.plot(thresholds, densities, marker='o', linewidth=2, color='orange')
    ax2.set_xlabel('Frequency Threshold')
    ax2.set_ylabel('Network Density')
    ax2.set_title('Density vs Threshold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_comparison.pdf'))
    plt.savefig(os.path.join(output_dir, 'threshold_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/threshold_comparison.pdf")

def main():
    args = parse_arguments()
    
    # Determine directory to analyze
    if args.directory:
        directory = args.directory
    elif args.timestamp:
        directory = os.path.join("gcm_roebroeck", args.timestamp)
    else:
        latest = find_latest_results()
        if latest is None:
            print("Error: No results found in gcm_roebroeck/")
            print("Please specify a directory with -d or timestamp with -t")
            return
        directory = latest
        print(f"Auto-detected most recent results: {directory}")
    
    # Watch mode: continuously update aggregation
    if args.watch:
        import time
        print(f"\n{'='*80}")
        print("WATCH MODE: Monitoring for new results")
        print(f"{'='*80}")
        print(f"Update interval: {args.watch_interval} seconds")
        print(f"Expected subjects: {args.expected_subjects if args.expected_subjects else 'unknown'}")
        print("Press Ctrl+C to stop")
        print(f"{'='*80}\n")
        
        last_n_subjects = 0
        iteration = 0
        
        try:
            while True:
                iteration += 1
                print(f"\n[Update {iteration}] {pd.Timestamp.now()}")
                
                # Load and aggregate results
                results = load_results(directory, force_aggregate=True)
                n_subjects = results['n_subjects']
                
                # Check if new subjects appeared
                if n_subjects > last_n_subjects:
                    print(f"✓ New results detected: {n_subjects} subjects (was {last_n_subjects})")
                    last_n_subjects = n_subjects
                    
                    # Save aggregated results
                    save_aggregated_results(results, directory)
                    
                    # Print brief statistics
                    print_statistics(results, threshold=args.threshold, expected_subjects=args.expected_subjects)
                    
                    # Check if complete
                    if args.expected_subjects and n_subjects >= args.expected_subjects:
                        print(f"\n🎉 All {args.expected_subjects} subjects completed!")
                        print("Final aggregation saved. Exiting watch mode.")
                        break
                else:
                    print(f"No new results ({n_subjects} subjects)")
                
                # Wait before next check
                if args.expected_subjects is None or n_subjects < args.expected_subjects:
                    time.sleep(args.watch_interval)
                else:
                    break
        
        except KeyboardInterrupt:
            print("\n\nWatch mode interrupted by user")
            print(f"Last count: {last_n_subjects} subjects")
        
        # Final analysis with plots if requested
        if args.plot:
            print(f"\n{'='*80}")
            print("Generating final plots...")
            print(f"{'='*80}")
            results = load_results(directory, force_aggregate=True)
        else:
            return
    
    else:
        # Normal mode: single analysis
        print(f"Loading results from: {directory}")
        results = load_results(directory, force_aggregate=args.aggregate)
        
        # Print statistics
        print_statistics(results, threshold=args.threshold, expected_subjects=args.expected_subjects)
        
        # Save aggregated results if we aggregated them
        if results.get('aggregated', False) or args.aggregate:
            save_aggregated_results(results, directory)
    
    # Generate plots
    if args.plot:
        print("\n" + "=" * 80)
        print("GENERATING PLOTS")
        print("=" * 80)
        
        output_dir = os.path.join(directory, "analysis_plots")
        os.makedirs(output_dir, exist_ok=True)
        
        plot_edge_frequency_histogram(results, output_dir)
        plot_heatmap(results, output_dir, threshold=args.threshold)
        plot_fdiff_heatmap(results, output_dir)
        
        # Threshold comparison
        thresholds = np.linspace(0, 1, 21)
        compare_with_threshold(results, thresholds, output_dir)
        
        print(f"\nAll plots saved to: {output_dir}/")
    
    # Save summary
    summary_file = os.path.join(directory, "analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("GCM RESULTS ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Analysis timestamp: {pd.Timestamp.now()}\n")
        f.write(f"Results directory: {directory}\n")
        f.write(f"Number of subjects: {results['n_subjects']}\n\n")
        
        if 'rate' in results:
            rate_df = results['rate']
            rate_matrix = rate_df.to_numpy()
            np.fill_diagonal(rate_matrix, 0)
            
            f.write("Edge Frequency Statistics:\n")
            f.write(f"  Mean: {rate_matrix[rate_matrix > 0].mean():.2%}\n")
            f.write(f"  Max: {rate_matrix.max():.2%}\n")
            f.write(f"  Edges with freq > 0: {np.sum(rate_matrix > 0)}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

