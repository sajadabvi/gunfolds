"""
Analyze saved solutions from fMRI experiment

This script loads and analyzes the solution information saved by fmri_experiment.py
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gunfolds.utils import zickle as zkl
from gunfolds.scripts import my_functions as mf
from gunfolds.utils import graphkit as gk

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze saved fMRI experiment solutions')
    parser.add_argument("-d", "--directory", default=None,
                        help="Directory containing saved solutions (e.g., fbirn_results/12262025103045/combined)")
    parser.add_argument("-t", "--timestamp", default=None,
                        help="Timestamp folder to analyze (e.g., 12262025103045). If not specified, uses most recent.")
    parser.add_argument("-g", "--group", default=None, type=int,
                        help="Analyze specific group (default: all combined)")
    parser.add_argument("--plot", action='store_true',
                        help="Generate plots")
    parser.add_argument("--compare_gt", action='store_true',
                        help="Compare against ground truth (requires GT to be specified)")
    return parser.parse_args()

def find_latest_results():
    """Find the most recent timestamped results folder"""
    base_dir = "fbirn_results"
    if not os.path.exists(base_dir):
        return None
    
    # List all subdirectories that look like timestamps (12 digits)
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

def load_solutions(directory, group=None):
    """Load solutions from specified directory"""
    if group is not None:
        solutions_file = os.path.join(directory, f"../group_{group}/solutions/solutions_info_{group}.zkl")
    else:
        solutions_file = os.path.join(directory, "all_solutions_info.zkl")
    
    params_file = os.path.join(directory, "selection_params.zkl")
    
    if not os.path.exists(solutions_file):
        raise FileNotFoundError(f"Solutions file not found: {solutions_file}")
    
    solutions = zkl.load(solutions_file)
    
    if os.path.exists(params_file):
        params = zkl.load(params_file)
    else:
        params = None
    
    return solutions, params

def analyze_cost_distribution(all_solutions):
    """Analyze cost distribution across all solutions"""
    all_costs = []
    costs_by_subject = {}
    costs_by_group = {}
    
    for subject_info in all_solutions:
        subject_id = subject_info['subject_id']
        group = subject_info['group']
        
        subject_costs = [s['cost'] for s in subject_info['solutions']]
        all_costs.extend(subject_costs)
        costs_by_subject[subject_id] = subject_costs
        
        if group not in costs_by_group:
            costs_by_group[group] = []
        costs_by_group[group].extend(subject_costs)
    
    return {
        'all_costs': all_costs,
        'costs_by_subject': costs_by_subject,
        'costs_by_group': costs_by_group
    }

def print_cost_statistics(cost_analysis, params):
    """Print summary statistics about costs"""
    all_costs = cost_analysis['all_costs']
    
    print("=" * 80)
    print("COST DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    if params:
        print(f"\nSelection method: {params['selection_mode']}")
        if params['selection_mode'] == 'top_k':
            print(f"  Top K: {params['top_k']}")
        elif params['selection_mode'] == 'delta_threshold':
            print(f"  Delta multiplier: {params['delta_multiplier']}")
        print(f"  Run timestamp: {params['timestamp']}")
    
    print(f"\nTotal solutions: {len(all_costs)}")
    print(f"Cost range: [{np.min(all_costs):.2f}, {np.max(all_costs):.2f}]")
    print(f"Mean cost: {np.mean(all_costs):.2f}")
    print(f"Median cost: {np.median(all_costs):.2f}")
    print(f"Std deviation: {np.std(all_costs):.2f}")
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        print(f"  {p}th: {np.percentile(all_costs, p):.2f}")
    
    # Group statistics
    costs_by_group = cost_analysis['costs_by_group']
    if len(costs_by_group) > 1:
        print("\n" + "=" * 80)
        print("GROUP STATISTICS")
        print("=" * 80)
        for group in sorted(costs_by_group.keys()):
            group_costs = costs_by_group[group]
            print(f"\nGroup {group}:")
            print(f"  Solutions: {len(group_costs)}")
            print(f"  Mean cost: {np.mean(group_costs):.2f}")
            print(f"  Median cost: {np.median(group_costs):.2f}")
            print(f"  Cost range: [{np.min(group_costs):.2f}, {np.max(group_costs):.2f}]")

def print_subject_summary(all_solutions):
    """Print summary statistics per subject"""
    print("\n" + "=" * 80)
    print("PER-SUBJECT SUMMARY")
    print("=" * 80)
    
    data_rows = []
    for subject_info in all_solutions:
        subject_id = subject_info['subject_id']
        group = subject_info['group']
        num_sols = subject_info['num_solutions']
        costs = [s['cost'] for s in subject_info['solutions']]
        
        data_rows.append({
            'Subject': subject_id,
            'Group': group,
            'Num_Solutions': num_sols,
            'Min_Cost': np.min(costs),
            'Max_Cost': np.max(costs),
            'Mean_Cost': np.mean(costs),
            'Median_Cost': np.median(costs)
        })
    
    df = pd.DataFrame(data_rows)
    print(df.to_string(index=False))
    
    return df

def plot_cost_distributions(cost_analysis, params, output_dir):
    """Generate plots of cost distributions"""
    os.makedirs(output_dir, exist_ok=True)
    
    all_costs = cost_analysis['all_costs']
    costs_by_group = cost_analysis['costs_by_group']
    
    # Overall cost distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_costs, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.title('Distribution of Solution Costs (All Solutions)')
    
    # Add statistics text
    stats_text = f"Mean: {np.mean(all_costs):.2f}\nMedian: {np.median(all_costs):.2f}\nStd: {np.std(all_costs):.2f}"
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_distribution_all.pdf'))
    plt.savefig(os.path.join(output_dir, 'cost_distribution_all.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/cost_distribution_all.pdf")
    
    # Cost distribution by group
    if len(costs_by_group) > 1:
        plt.figure(figsize=(10, 6))
        for group in sorted(costs_by_group.keys()):
            plt.hist(costs_by_group[group], bins=30, alpha=0.5, label=f'Group {group}', edgecolor='black')
        plt.xlabel('Cost')
        plt.ylabel('Frequency')
        plt.title('Distribution of Solution Costs by Group')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cost_distribution_by_group.pdf'))
        plt.savefig(os.path.join(output_dir, 'cost_distribution_by_group.png'), dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/cost_distribution_by_group.pdf")
    
    # Boxplot by group
    if len(costs_by_group) > 1:
        plt.figure(figsize=(8, 6))
        data_for_box = [costs_by_group[g] for g in sorted(costs_by_group.keys())]
        labels_for_box = [f'Group {g}' for g in sorted(costs_by_group.keys())]
        plt.boxplot(data_for_box, labels=labels_for_box)
        plt.ylabel('Cost')
        plt.title('Cost Distribution by Group (Boxplot)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cost_boxplot_by_group.pdf'))
        plt.savefig(os.path.join(output_dir, 'cost_boxplot_by_group.png'), dpi=150)
        plt.close()
        print(f"Saved: {output_dir}/cost_boxplot_by_group.pdf")

def compare_with_ground_truth(all_solutions, network_GT, include_selfloop=True):
    """Compare selected solutions with ground truth"""
    print("\n" + "=" * 80)
    print("GROUND TRUTH COMPARISON")
    print("=" * 80)
    
    all_metrics = []
    
    for subject_info in all_solutions:
        subject_id = subject_info['subject_id']
        group = subject_info['group']
        
        for sol in subject_info['solutions']:
            graph = sol['graph']
            cost = sol['cost']
            
            # Compute metrics
            metrics = mf.precision_recall(graph, network_GT, include_selfloop=include_selfloop)
            
            all_metrics.append({
                'subject_id': subject_id,
                'group': group,
                'cost': cost,
                'orientation_precision': metrics['orientation']['precision'],
                'orientation_recall': metrics['orientation']['recall'],
                'orientation_F1': metrics['orientation']['F1'],
                'adjacency_precision': metrics['adjacency']['precision'],
                'adjacency_recall': metrics['adjacency']['recall'],
                'adjacency_F1': metrics['adjacency']['F1']
            })
    
    df = pd.DataFrame(all_metrics)
    
    # Summary statistics
    print("\nOverall Metrics (mean ± std):")
    for metric in ['orientation_F1', 'orientation_precision', 'orientation_recall',
                   'adjacency_F1', 'adjacency_precision', 'adjacency_recall']:
        mean = df[metric].mean()
        std = df[metric].std()
        print(f"  {metric}: {mean:.4f} ± {std:.4f}")
    
    # By group
    if df['group'].nunique() > 1:
        print("\nMetrics by Group:")
        for group in sorted(df['group'].unique()):
            group_df = df[df['group'] == group]
            print(f"\n  Group {group}:")
            for metric in ['orientation_F1', 'adjacency_F1']:
                mean = group_df[metric].mean()
                std = group_df[metric].std()
                print(f"    {metric}: {mean:.4f} ± {std:.4f}")
    
    return df

def main():
    args = parse_arguments()
    
    # Determine directory to analyze
    if args.directory:
        # User specified exact directory
        directory = args.directory
    elif args.timestamp:
        # User specified timestamp
        directory = os.path.join("fbirn_results", args.timestamp, "combined")
    else:
        # Auto-detect most recent results
        latest = find_latest_results()
        if latest is None:
            print("Error: No results found in fbirn_results/")
            print("Please specify a directory with -d or timestamp with -t")
            return
        directory = os.path.join(latest, "combined")
        print(f"Auto-detected most recent results: {latest}")
    
    # Load solutions
    print(f"Loading solutions from: {directory}")
    all_solutions, params = load_solutions(directory, args.group)
    print(f"Loaded {len(all_solutions)} subjects")
    
    # Analyze cost distribution
    cost_analysis = analyze_cost_distribution(all_solutions)
    print_cost_statistics(cost_analysis, params)
    
    # Subject summary
    df_summary = print_subject_summary(all_solutions)
    
    # Save summary CSV
    output_dir = directory if args.group is None else os.path.join(directory, f"../group_{args.group}")
    summary_file = os.path.join(output_dir, "analysis_summary.csv")
    df_summary.to_csv(summary_file, index=False)
    print(f"\nSaved summary to: {summary_file}")
    
    # Generate plots
    if args.plot:
        print("\n" + "=" * 80)
        print("GENERATING PLOTS")
        print("=" * 80)
        plot_cost_distributions(cost_analysis, params, output_dir)
    
    # Ground truth comparison
    if args.compare_gt:
        # You need to define the ground truth network here
        network_GT = {1: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1}, 2: {1: 1, 3: 1, 4: 1, 5: 1, 6: 1},
                      3: {1: 1, 2: 1, 4: 1, 5: 1, 6: 1}, 4: {1: 1, 2: 1, 3: 1, 5: 1, 6: 1},
                      5: {1: 1, 2: 1, 3: 1, 4: 1, 6: 1}, 6: {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}}
        
        df_gt = compare_with_ground_truth(all_solutions, network_GT)
        gt_file = os.path.join(output_dir, "ground_truth_comparison.csv")
        df_gt.to_csv(gt_file, index=False)
        print(f"\nSaved GT comparison to: {gt_file}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

