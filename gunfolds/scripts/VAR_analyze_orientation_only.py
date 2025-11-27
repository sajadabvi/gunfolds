"""
Script to analyze hyperparameter tuning results - ORIENTATION F1 ONLY
All rankings and analysis based solely on orientation_F1
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid hanging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from gunfolds.utils import zickle as zkl

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter tuning results (Orientation F1 only).')
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

def plot_orientation_f1_top(df, output_folder, top_n=20):
    """Plot top configurations by orientation F1"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df_top = df.head(top_n)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_top)))
    bars = ax.bar(range(len(df_top)), df_top['orientation_F1'], color=colors)
    
    ax.set_xlabel('Priority Configuration (Ranked by Orientation F1)', fontsize=12)
    ax.set_ylabel('Orientation F1 Score', fontsize=12)
    ax.set_title(f'Top {top_n} Priority Configurations by Orientation F1', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(df_top)))
    ax.set_xticklabels(df_top['priorities'], rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df_top['orientation_F1'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    if output_folder:
        plt.savefig(f'{output_folder}/orientation_f1_top.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_orientation_f1_distribution(df, output_folder):
    """Plot distribution of orientation F1 scores"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    axes[0].hist(df['orientation_F1'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(df['orientation_F1'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["orientation_F1"].mean():.4f}')
    axes[0].axvline(df['orientation_F1'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["orientation_F1"].median():.4f}')
    axes[0].set_xlabel('Orientation F1 Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Orientation F1 Scores')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot([df['orientation_F1']], labels=['Orientation F1'])
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Orientation F1 Score Distribution')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if output_folder:
        plt.savefig(f'{output_folder}/orientation_f1_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_priority_impact_on_orientation(df, output_folder):
    """Plot how each priority position/value affects orientation F1"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a matrix: rows = priority positions (p1-p5), cols = priority values (1-5)
    heatmap_data = np.zeros((5, 5))
    count_data = np.zeros((5, 5))
    
    for _, row in df.iterrows():
        for i in range(1, 6):  # p1 to p5
            priority_pos = i - 1
            priority_val = int(row[f'p{i}']) - 1
            heatmap_data[priority_pos, priority_val] += row['orientation_F1']
            count_data[priority_pos, priority_val] += 1
    
    # Average the values
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_data = np.divide(heatmap_data, count_data)
        heatmap_data = np.nan_to_num(heatmap_data)
    
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', 
               xticklabels=[1, 2, 3, 4, 5], 
               yticklabels=['p1', 'p2', 'p3', 'p4', 'p5'],
               ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Avg Orientation F1'})
    ax.set_title('Priority Impact on Orientation F1', fontsize=14, fontweight='bold')
    ax.set_xlabel('Priority Value')
    ax.set_ylabel('Priority Position')
    
    plt.tight_layout()
    if output_folder:
        plt.savefig(f'{output_folder}/priority_impact_orientation.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_orientation(df, output_folder):
    """Plot precision vs recall for orientation"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(df['orientation_recall'], 
                       df['orientation_precision'],
                       c=df['orientation_F1'],
                       s=100,
                       alpha=0.6,
                       cmap='viridis')
    
    ax.set_xlabel('Orientation Recall', fontsize=12)
    ax.set_ylabel('Orientation Precision', fontsize=12)
    ax.set_title('Orientation: Precision vs Recall', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add diagonal line (P=R line)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.3, label='P=R line')
    
    # Highlight top 10
    top_10 = df.head(10)
    ax.scatter(top_10['orientation_recall'], 
              top_10['orientation_precision'],
              s=200, 
              facecolors='none', 
              edgecolors='red', 
              linewidths=2,
              label='Top 10')
    
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Orientation F1')
    
    plt.tight_layout()
    if output_folder:
        plt.savefig(f'{output_folder}/precision_recall_orientation.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_table(df, output_folder):
    """Generate and save summary statistics"""
    summary = {
        'Metric': ['Orientation F1', 'Orientation Precision', 'Orientation Recall'],
        'Mean': [
            df['orientation_F1'].mean(),
            df['orientation_precision'].mean(),
            df['orientation_recall'].mean()
        ],
        'Std': [
            df['orientation_F1'].std(),
            df['orientation_precision'].std(),
            df['orientation_recall'].std()
        ],
        'Min': [
            df['orientation_F1'].min(),
            df['orientation_precision'].min(),
            df['orientation_recall'].min()
        ],
        'Max': [
            df['orientation_F1'].max(),
            df['orientation_precision'].max(),
            df['orientation_recall'].max()
        ],
        'Best Priority': [
            df.loc[df['orientation_F1'].idxmax(), 'priorities'],
            df.loc[df['orientation_precision'].idxmax(), 'priorities'],
            df.loc[df['orientation_recall'].idxmax(), 'priorities']
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (ORIENTATION ONLY)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    if output_folder:
        summary_df.to_csv(f'{output_folder}/summary_statistics_orientation.csv', index=False)
        print(f"\nSummary saved to: {output_folder}/summary_statistics_orientation.csv")

def analyze_priority_patterns(df):
    """Analyze patterns in successful priority configurations"""
    print("\n" + "=" * 80)
    print("PRIORITY PATTERN ANALYSIS (TOP 10 BY ORIENTATION F1)")
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
    
    # Sort ONLY by orientation F1
    df = df.sort_values('orientation_F1', ascending=False)
    df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} priority configurations")
    print(f"Sorted by: Orientation F1 (descending)")
    
    # Setup output folder
    if args.output:
        output_folder = args.output
    else:
        output_folder = os.path.dirname(args.file)
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Generate all analyses
    print("\nGenerating visualizations focused on Orientation F1...")
    
    plot_orientation_f1_top(df, output_folder)
    print("✓ Top configurations plot")
    
    plot_orientation_f1_distribution(df, output_folder)
    print("✓ Distribution plots")
    
    plot_priority_impact_on_orientation(df, output_folder)
    print("✓ Priority impact heatmap")
    
    plot_precision_recall_orientation(df, output_folder)
    print("✓ Precision-recall plot")
    
    generate_summary_table(df, output_folder)
    
    analyze_priority_patterns(df)
    
    # Show top 10
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS (BY ORIENTATION F1)")
    print("=" * 80)
    top_10_display = df.head(10)[['priorities', 'orientation_F1', 'orientation_precision', 'orientation_recall']]
    print(top_10_display.to_string(index=False))
    
    # Show best
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    best = df.iloc[0]
    print(f"\nPriorities: {best['priorities']}")
    print(f"Orientation F1: {best['orientation_F1']:.4f}")
    print(f"Orientation Precision: {best['orientation_precision']:.4f}")
    print(f"Orientation Recall: {best['orientation_recall']:.4f}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

