"""
Script to analyze and visualize hyperparameter tuning results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from gunfolds.utils import zickle as zkl

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter tuning results.')
    parser.add_argument("-f", "--file", required=True, help="CSV file with results", type=str)
    parser.add_argument("-o", "--output", default="", help="output folder for plots", type=str)
    return parser.parse_args()

def load_results(filename):
    """Load results from CSV file"""
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.endswith('.zkl'):
        data = zkl.load(filename)
        df = pd.DataFrame(data['results'])
    else:
        raise ValueError("File must be .csv or .zkl")
    return df

def plot_f1_comparison(df, output_folder):
    """Plot F1 scores comparison"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    top_n = min(20, len(df))
    df_top = df.head(top_n)
    
    x = np.arange(len(df_top))
    width = 0.25
    
    ax.bar(x - width, df_top['orientation_F1'], width, label='Orientation', alpha=0.8)
    ax.bar(x, df_top['adjacency_F1'], width, label='Adjacency', alpha=0.8)
    ax.bar(x + width, df_top['cycle_F1'], width, label='Cycle', alpha=0.8)
    
    ax.set_xlabel('Priority Configuration')
    ax.set_ylabel('F1 Score')
    ax.set_title(f'F1 Scores for Top {top_n} Priority Configurations')
    ax.set_xticks(x)
    ax.set_xticklabels(df_top['priorities'], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_folder:
        plt.savefig(f'{output_folder}/f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_combined_f1(df, output_folder):
    """Plot combined F1 scores"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    top_n = min(20, len(df))
    df_top = df.head(top_n)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_top)))
    bars = ax.bar(range(len(df_top)), df_top['combined_F1'], color=colors)
    
    ax.set_xlabel('Priority Configuration (Ranked)')
    ax.set_ylabel('Combined F1 Score')
    ax.set_title(f'Combined F1 Score for Top {top_n} Priority Configurations')
    ax.set_xticks(range(len(df_top)))
    ax.set_xticklabels(df_top['priorities'], rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df_top['combined_F1'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    if output_folder:
        plt.savefig(f'{output_folder}/combined_f1.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_priority_heatmap(df, output_folder):
    """Plot heatmap showing how each priority position affects performance"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['orientation_F1', 'adjacency_F1', 'cycle_F1']
    titles = ['Orientation F1', 'Adjacency F1', 'Cycle F1']
    
    for ax, metric, title in zip(axes, metrics, titles):
        # Create a matrix: rows = priority positions (p1-p5), cols = priority values (1-5)
        heatmap_data = np.zeros((5, 5))
        count_data = np.zeros((5, 5))
        
        for _, row in df.iterrows():
            for i in range(1, 6):  # p1 to p5
                priority_pos = i - 1
                priority_val = int(row[f'p{i}']) - 1
                heatmap_data[priority_pos, priority_val] += row[metric]
                count_data[priority_pos, priority_val] += 1
        
        # Average the values
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap_data = np.divide(heatmap_data, count_data)
            heatmap_data = np.nan_to_num(heatmap_data)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   xticklabels=[1, 2, 3, 4, 5], 
                   yticklabels=['p1', 'p2', 'p3', 'p4', 'p5'],
                   ax=ax, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel('Priority Value')
        ax.set_ylabel('Priority Position')
    
    plt.tight_layout()
    if output_folder:
        plt.savefig(f'{output_folder}/priority_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_precision_recall(df, output_folder):
    """Plot precision-recall scatter plots"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = [('orientation', 'Orientation'), 
               ('adjacency', 'Adjacency'), 
               ('cycle', 'Cycle')]
    
    for ax, (metric, title) in zip(axes, metrics):
        scatter = ax.scatter(df[f'{metric}_recall'], 
                           df[f'{metric}_precision'],
                           c=df['combined_F1'],
                           s=100,
                           alpha=0.6,
                           cmap='viridis')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{title}: Precision vs Recall')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add diagonal line (F1 = 0.5 contour)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.3, label='P=R line')
        
        plt.colorbar(scatter, ax=ax, label='Combined F1')
    
    plt.tight_layout()
    if output_folder:
        plt.savefig(f'{output_folder}/precision_recall.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def generate_summary_table(df, output_folder):
    """Generate and save summary statistics"""
    summary = {
        'Metric': [],
        'Mean': [],
        'Std': [],
        'Min': [],
        'Max': [],
        'Best Priority': []
    }
    
    metrics = ['orientation_F1', 'adjacency_F1', 'cycle_F1', 'combined_F1',
               'orientation_precision', 'orientation_recall',
               'adjacency_precision', 'adjacency_recall',
               'cycle_precision', 'cycle_recall']
    
    for metric in metrics:
        summary['Metric'].append(metric)
        summary['Mean'].append(df[metric].mean())
        summary['Std'].append(df[metric].std())
        summary['Min'].append(df[metric].min())
        summary['Max'].append(df[metric].max())
        best_idx = df[metric].idxmax()
        summary['Best Priority'].append(df.loc[best_idx, 'priorities'])
    
    summary_df = pd.DataFrame(summary)
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    if output_folder:
        summary_df.to_csv(f'{output_folder}/summary_statistics.csv', index=False)
        print(f"\nSummary saved to: {output_folder}/summary_statistics.csv")

def analyze_priority_patterns(df):
    """Analyze patterns in successful priority configurations"""
    print("\n" + "=" * 80)
    print("PRIORITY PATTERN ANALYSIS")
    print("=" * 80)
    
    # Top 10 configurations
    top_10 = df.head(10)
    
    print("\n1. Average priority values in top 10 configurations:")
    for i in range(1, 6):
        avg = top_10[f'p{i}'].mean()
        std = top_10[f'p{i}'].std()
        print(f"   p{i}: {avg:.2f} ± {std:.2f}")
    
    print("\n2. Most common priority values in top 10:")
    for i in range(1, 6):
        mode = top_10[f'p{i}'].mode()
        if len(mode) > 0:
            print(f"   p{i}: {mode.values[0]}")
    
    print("\n3. Priority value distribution in top 10:")
    all_priorities = []
    for i in range(1, 6):
        all_priorities.extend(top_10[f'p{i}'].tolist())
    
    for val in range(1, 6):
        count = all_priorities.count(val)
        pct = (count / len(all_priorities)) * 100
        print(f"   Value {val}: {count} times ({pct:.1f}%)")

def main():
    args = parse_arguments()
    
    # Load results
    print(f"Loading results from: {args.file}")
    df = load_results(args.file)
    
    print(f"Loaded {len(df)} priority configurations")
    
    # Setup output folder
    if args.output:
        output_folder = args.output
    else:
        output_folder = os.path.dirname(args.file)
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Generate all analyses
    print("\nGenerating visualizations...")
    
    plot_f1_comparison(df, output_folder)
    print("✓ F1 comparison plot")
    
    plot_combined_f1(df, output_folder)
    print("✓ Combined F1 plot")
    
    plot_priority_heatmap(df, output_folder)
    print("✓ Priority heatmap")
    
    plot_precision_recall(df, output_folder)
    print("✓ Precision-recall plots")
    
    generate_summary_table(df, output_folder)
    
    analyze_priority_patterns(df)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

