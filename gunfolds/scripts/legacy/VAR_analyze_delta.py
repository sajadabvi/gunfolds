"""
Analyze and visualize delta hyperparameter tuning results
Focus: ORIENTATION F1 ONLY
Delta: Percentage of minimum cost
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze delta tuning results.')
    parser.add_argument("-f", "--file", required=True, help="CSV file with delta tuning results")
    parser.add_argument("-o", "--output_dir", default=None, help="output directory for plots (default: same as input)")
    return parser.parse_args()

def plot_delta_vs_f1(df, output_dir):
    """
    Plot how Orientation F1 changes with delta percentage
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Delta vs Orientation F1 (Percentage of Min Cost)', fontsize=16, fontweight='bold')
    
    # Sort by delta percentage for proper line plots
    df_sorted = df.sort_values('delta_percentage')
    
    # Convert to percentage for x-axis
    delta_pct = df_sorted['delta_percentage'] * 100
    
    # Plot 1: Orientation F1
    ax = axes[0, 0]
    ax.plot(delta_pct, df_sorted['orientation_F1'], 'o-', linewidth=2, markersize=8, color='steelblue')
    ax.set_xlabel('Delta (% of Min Cost)', fontsize=12)
    ax.set_ylabel('Orientation F1 Score', fontsize=12)
    ax.set_title('Delta vs Orientation F1', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mark best point
    best_idx = df_sorted['orientation_F1'].idxmax()
    best_delta_pct = df_sorted.loc[best_idx, 'delta_percentage'] * 100
    best_f1 = df_sorted.loc[best_idx, 'orientation_F1']
    ax.plot(best_delta_pct, best_f1, 'r*', markersize=20, label=f'Best: δ={best_delta_pct:.0f}%')
    ax.legend(fontsize=10)
    
    # Auto-scale y-axis with padding to highlight differences
    y_min = df_sorted['orientation_F1'].min()
    y_max = df_sorted['orientation_F1'].max()
    y_padding = (y_max - y_min) * 0.1  # 10% padding
    ax.set_ylim([max(0, y_min - y_padding), min(1.0, y_max + y_padding)])
    
    # Plot 2: Precision and Recall
    ax = axes[0, 1]
    ax.plot(delta_pct, df_sorted['orientation_precision'], 'o-', label='Precision', linewidth=2, color='green')
    ax.plot(delta_pct, df_sorted['orientation_recall'], 's-', label='Recall', linewidth=2, color='orange')
    ax.set_xlabel('Delta (% of Min Cost)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Delta vs Precision & Recall', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Auto-scale y-axis with padding to highlight differences
    y_min = min(df_sorted['orientation_precision'].min(), df_sorted['orientation_recall'].min())
    y_max = max(df_sorted['orientation_precision'].max(), df_sorted['orientation_recall'].max())
    y_padding = (y_max - y_min) * 0.1  # 10% padding
    ax.set_ylim([max(0, y_min - y_padding), min(1.0, y_max + y_padding)])
    
    # Plot 3: Number of solutions selected
    ax = axes[1, 0]
    ax.plot(delta_pct, df_sorted['avg_solutions_selected'], 'o-', 
            color='purple', linewidth=2, markersize=8)
    ax.set_xlabel('Delta (% of Min Cost)', fontsize=12)
    ax.set_ylabel('Avg Solutions Selected', fontsize=12)
    ax.set_title('Delta vs Number of Solutions Selected', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add reference line for total solutions
    if 'avg_solutions_total' in df_sorted.columns:
        avg_total = df_sorted['avg_solutions_total'].mean()
        ax.axhline(y=avg_total, color='r', linestyle='--', alpha=0.5, 
                   label=f'Total solutions: {avg_total:.1f}')
        ax.legend(fontsize=10)
    
    # Plot 4: Precision vs Recall scatter
    ax = axes[1, 1]
    scatter = ax.scatter(df_sorted['orientation_recall'], df_sorted['orientation_precision'],
                        c=delta_pct, cmap='viridis', s=100, alpha=0.7)
    ax.set_xlabel('Orientation Recall', fontsize=12)
    ax.set_ylabel('Orientation Precision', fontsize=12)
    ax.set_title('Precision vs Recall (colored by delta %)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Delta (%)', fontsize=10)
    
    # Mark best point
    best_recall = df_sorted.loc[best_idx, 'orientation_recall']
    best_prec = df_sorted.loc[best_idx, 'orientation_precision']
    ax.plot(best_recall, best_prec, 'r*', markersize=20, label='Best')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'delta_vs_orientation_f1_analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

def plot_f1_bar_chart(df, output_dir):
    """
    Bar chart for top 10 configurations
    """
    top_n = min(10, len(df))
    df_top = df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(top_n)
    width = 0.35
    
    # Convert to percentage for labels
    delta_pct = df_top['delta_percentage'] * 100
    
    bars1 = ax.bar(x - width/2, df_top['orientation_precision'], width, label='Precision', alpha=0.8, color='green')
    bars2 = ax.bar(x + width/2, df_top['orientation_recall'], width, label='Recall', alpha=0.8, color='orange')
    
    ax.set_xlabel('Configuration (ranked by Orientation F1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Delta Configurations - Precision & Recall', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'δ={d:.0f}%' for d in delta_pct], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    # Add F1 score text above bars
    for i, (idx, row) in enumerate(df_top.iterrows()):
        ax.text(i, 0.95, f"F1={row['orientation_F1']:.3f}", 
               ha='center', fontsize=9, fontweight='bold', color='red')
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'top_configurations_bar_chart.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

def plot_f1_top10(df, output_dir):
    """
    Horizontal bar chart of orientation F1 for top 10
    """
    top_n = min(10, len(df))
    df_top = df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    delta_pct = df_top['delta_percentage'] * 100
    
    bars = ax.barh(range(top_n), df_top['orientation_F1'], alpha=0.8, color='steelblue')
    
    # Color the best bar differently
    bars[0].set_color('darkgreen')
    bars[0].set_alpha(0.9)
    
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f'δ={d:.0f}%' for d in delta_pct], fontsize=11)
    ax.set_xlabel('Orientation F1 Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Delta (% of Min Cost)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Delta Values - Orientation F1', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([0, 1.0])
    
    # Add value labels
    for i, (idx, row) in enumerate(df_top.iterrows()):
        ax.text(row['orientation_F1'] + 0.01, i, f"{row['orientation_F1']:.4f}", 
               va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'orientation_f1_top10.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

def plot_solutions_vs_f1(df, output_dir):
    """
    Plot relationship between number of solutions and F1 score
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    delta_pct = df['delta_percentage'] * 100
    
    # Plot 1: Solutions selected vs Orientation F1
    ax = axes[0]
    scatter = ax.scatter(df['avg_solutions_selected'], df['orientation_F1'], 
                        c=delta_pct, cmap='viridis', s=100, alpha=0.7)
    ax.set_xlabel('Avg Solutions Selected', fontsize=12)
    ax.set_ylabel('Orientation F1 Score', fontsize=12)
    ax.set_title('Number of Solutions vs Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Delta (%)', fontsize=10)
    
    # Mark best point
    best_idx = df['orientation_F1'].idxmax()
    best_sols = df.loc[best_idx, 'avg_solutions_selected']
    best_f1 = df.loc[best_idx, 'orientation_F1']
    ax.plot(best_sols, best_f1, 'r*', markersize=20, label='Best Config')
    ax.legend(fontsize=10)
    
    # Plot 2: Percentage of solutions vs F1
    ax = axes[1]
    df['pct_solutions'] = df['avg_solutions_selected'] / df['avg_solutions_total'] * 100
    scatter = ax.scatter(df['pct_solutions'], df['orientation_F1'],
                        c=delta_pct, cmap='viridis', s=100, alpha=0.7)
    ax.set_xlabel('Percentage of Solutions Selected (%)', fontsize=12)
    ax.set_ylabel('Orientation F1 Score', fontsize=12)
    ax.set_title('Solution Percentage vs Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Delta (%)', fontsize=10)
    
    # Mark best point
    best_pct = df.loc[best_idx, 'pct_solutions']
    ax.plot(best_pct, best_f1, 'r*', markersize=20, label='Best Config')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'solutions_vs_f1.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

def generate_statistics_table(df, output_dir):
    """
    Generate summary statistics table
    """
    stats = {
        'Metric': [],
        'Best Delta (%)': [],
        'Best Value': [],
        'Mean': [],
        'Std': [],
        'Min': [],
        'Max': []
    }
    
    metrics = {
        'orientation_F1': 'Orientation F1',
        'orientation_precision': 'Orientation Precision',
        'orientation_recall': 'Orientation Recall',
        'avg_solutions_selected': 'Avg Solutions Selected'
    }
    
    for col, name in metrics.items():
        stats['Metric'].append(name)
        best_idx = df[col].idxmax()
        best_delta_pct = df.loc[best_idx, 'delta_percentage'] * 100
        stats['Best Delta (%)'].append(f"{best_delta_pct:.0f}")
        stats['Best Value'].append(f"{df.loc[best_idx, col]:.4f}")
        stats['Mean'].append(f"{df[col].mean():.4f}")
        stats['Std'].append(f"{df[col].std():.4f}")
        stats['Min'].append(f"{df[col].min():.4f}")
        stats['Max'].append(f"{df[col].max():.4f}")
    
    stats_df = pd.DataFrame(stats)
    
    # Save to CSV
    filename = os.path.join(output_dir, 'summary_statistics.csv')
    stats_df.to_csv(filename, index=False)
    print(f"✓ Saved: {filename}")
    
    # Print to console
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(stats_df.to_string(index=False))
    print("=" * 80)

def main():
    args = parse_arguments()
    
    # Read results
    print(f"Reading results from: {args.file}")
    df = pd.read_csv(args.file)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.file)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n" + "=" * 80)
    print("DELTA HYPERPARAMETER TUNING - ANALYSIS")
    print("FOCUS: ORIENTATION F1 ONLY")
    print("=" * 80)
    print(f"Results file: {args.file}")
    print(f"Number of delta values tested: {len(df)}")
    print(f"Delta range: {df['delta_percentage'].min()*100:.0f}% to {df['delta_percentage'].max()*100:.0f}%")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot_delta_vs_f1(df, output_dir)
    plot_f1_bar_chart(df, output_dir)
    plot_f1_top10(df, output_dir)
    plot_solutions_vs_f1(df, output_dir)
    
    # Generate statistics
    print("\nGenerating statistics...")
    generate_statistics_table(df, output_dir)
    
    # Print best configuration
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    best = df.iloc[0]
    print(f"\nDelta: {best['delta_percentage']*100:.0f}% of min_cost (absolute: {best['delta_absolute']:.0f})")
    print(f"Orientation F1: {best['orientation_F1']:.4f}")
    print(f"  - Precision: {best['orientation_precision']:.4f}")
    print(f"  - Recall: {best['orientation_recall']:.4f}")
    print(f"\nAvg solutions selected: {best['avg_solutions_selected']:.1f}/{best['avg_solutions_total']:.1f}")
    if 'avg_solutions_total' in df.columns:
        pct = best['avg_solutions_selected'] / best['avg_solutions_total'] * 100
        print(f"Percentage: {pct:.1f}%")
    print(f"Successful batches: {best['num_successful_batches']}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print(f"\n1. Best delta value: {best['delta_percentage']*100:.0f}% of min_cost")
    print(f"   Absolute value: {best['delta_absolute']:.0f}")
    print(f"   This maximizes orientation F1 score.")
    
    # Find delta that selects ~50% of solutions
    if 'avg_solutions_total' in df.columns:
        df['pct_solutions'] = df['avg_solutions_selected'] / df['avg_solutions_total'] * 100
        df_50pct = df.iloc[(df['pct_solutions'] - 50).abs().argsort()[:1]]
        if not df_50pct.empty:
            delta_50 = df_50pct.iloc[0]
            print(f"\n2. Delta for ~50% solutions: {delta_50['delta_percentage']*100:.0f}%")
            print(f"   Selects {delta_50['pct_solutions']:.1f}% of solutions")
            print(f"   Orientation F1: {delta_50['orientation_F1']:.4f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll plots saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - delta_vs_orientation_f1_analysis.png")
    print("  - top_configurations_bar_chart.png")
    print("  - orientation_f1_top10.png")
    print("  - solutions_vs_f1.png")
    print("  - summary_statistics.csv")
    print("=" * 80)

if __name__ == "__main__":
    main()
