"""
Unified VAR analysis and visualization script.

This module consolidates VAR analysis scripts:
- VAR_analyze_delta.py
- VAR_analyze_hyperparameters.py
- VAR_analyze_orientation_only.py

Usage:
    # Analyze delta tuning results
    python var_analyzer.py --analysis-type delta -f results.csv
    
    # Analyze hyperparameter tuning results
    python var_analyzer.py --analysis-type hyperparameters -f results.csv
    
    # Analyze orientation-only results
    python var_analyzer.py --analysis-type orientation -f results.csv
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from gunfolds.utils import zickle as zkl


# ============================================================================
# DELTA ANALYSIS
# ============================================================================

def analyze_delta(df, output_dir):
    """
    Analyze delta hyperparameter tuning results.
    Focus: ORIENTATION F1 ONLY
    Delta: Percentage of minimum cost
    """
    print("\n=== Delta Analysis ===")
    
    # Sort by delta percentage for proper line plots
    df_sorted = df.sort_values('delta_percentage')
    
    # Convert to percentage for x-axis
    delta_pct = df_sorted['delta_percentage'] * 100
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Delta vs Orientation F1 (Percentage of Min Cost)', 
                fontsize=16, fontweight='bold')
    
    # Plot 1: Orientation F1
    ax = axes[0, 0]
    ax.plot(delta_pct, df_sorted['orientation_F1'], 'o-', linewidth=2, 
           markersize=8, color='steelblue')
    ax.set_xlabel('Delta (% of Min Cost)', fontsize=12)
    ax.set_ylabel('Orientation F1 Score', fontsize=12)
    ax.set_title('Delta vs Orientation F1', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mark best point
    best_idx = df_sorted['orientation_F1'].idxmax()
    best_delta_pct = df_sorted.loc[best_idx, 'delta_percentage'] * 100
    best_f1 = df_sorted.loc[best_idx, 'orientation_F1']
    ax.plot(best_delta_pct, best_f1, 'r*', markersize=20, 
           label=f'Best: δ={best_delta_pct:.0f}%')
    ax.legend(fontsize=10)
    
    # Auto-scale y-axis with padding
    y_min = df_sorted['orientation_F1'].min()
    y_max = df_sorted['orientation_F1'].max()
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim([max(0, y_min - y_padding), min(1.0, y_max + y_padding)])
    
    # Plot 2: Precision and Recall
    ax = axes[0, 1]
    ax.plot(delta_pct, df_sorted['orientation_precision'], 'o-', 
           label='Precision', linewidth=2, color='green')
    ax.plot(delta_pct, df_sorted['orientation_recall'], 's-', 
           label='Recall', linewidth=2, color='orange')
    ax.set_xlabel('Delta (% of Min Cost)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Delta vs Precision & Recall', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Auto-scale y-axis
    y_min = min(df_sorted['orientation_precision'].min(), 
               df_sorted['orientation_recall'].min())
    y_max = max(df_sorted['orientation_precision'].max(), 
               df_sorted['orientation_recall'].max())
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim([max(0, y_min - y_padding), min(1.0, y_max + y_padding)])
    
    # Plot 3: Number of solutions selected
    ax = axes[1, 0]
    ax.plot(delta_pct, df_sorted['avg_solutions_selected'], 'o-', 
           color='purple', linewidth=2, markersize=8)
    ax.set_xlabel('Delta (% of Min Cost)', fontsize=12)
    ax.set_ylabel('Avg Solutions Selected', fontsize=12)
    ax.set_title('Delta vs Number of Solutions Selected', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add reference line for total solutions
    if 'avg_solutions_total' in df_sorted.columns:
        avg_total = df_sorted['avg_solutions_total'].mean()
        ax.axhline(y=avg_total, color='r', linestyle='--', alpha=0.5, 
                  label=f'Total solutions: {avg_total:.1f}')
        ax.legend(fontsize=10)
    
    # Plot 4: Precision vs Recall scatter
    ax = axes[1, 1]
    scatter = ax.scatter(df_sorted['orientation_recall'], 
                        df_sorted['orientation_precision'],
                        c=delta_pct, cmap='viridis', s=100, alpha=0.7)
    ax.set_xlabel('Orientation Recall', fontsize=12)
    ax.set_ylabel('Orientation Precision', fontsize=12)
    ax.set_title('Precision vs Recall (colored by delta %)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Delta (%)', fontsize=10)
    
    # Mark best point
    best_recall = df_sorted.loc[best_idx, 'orientation_recall']
    best_prec = df_sorted.loc[best_idx, 'orientation_precision']
    ax.plot(best_recall, best_prec, 'r*', markersize=20, 
           label=f'Best F1={best_f1:.3f}')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'delta_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved delta analysis plot: {output_path}")
    plt.close()
    
    # Print summary
    print(f"\n  Best delta: {best_delta_pct:.0f}% (F1={best_f1:.4f})")
    print(f"  Range: F1 from {df_sorted['orientation_F1'].min():.4f} " + 
          f"to {df_sorted['orientation_F1'].max():.4f}")


# ============================================================================
# HYPERPARAMETER ANALYSIS
# ============================================================================

def analyze_hyperparameters(df, output_dir):
    """Analyze hyperparameter tuning results"""
    print("\n=== Hyperparameter Analysis ===")
    
    # 1. F1 scores comparison
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
    output_path = os.path.join(output_dir, 'f1_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved F1 comparison: {output_path}")
    plt.close()
    
    # 2. Combined F1 plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_top)))
    bars = ax.bar(range(len(df_top)), df_top['combined_F1'], color=colors)
    
    ax.set_xlabel('Priority Configuration (Ranked)')
    ax.set_ylabel('Combined F1 Score')
    ax.set_title(f'Combined F1 Score for Top {top_n} Priority Configurations')
    ax.set_xticks(range(len(df_top)))
    ax.set_xticklabels(df_top['priorities'], rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_top['combined_F1'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'combined_f1.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved combined F1: {output_path}")
    plt.close()
    
    # 3. Priority heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['orientation_F1', 'adjacency_F1', 'cycle_F1']
    titles = ['Orientation F1', 'Adjacency F1', 'Cycle F1']
    
    for ax, metric, title in zip(axes, metrics, titles):
        # Create heatmap: rows = priority positions, cols = priority values
        heatmap_data = np.zeros((5, 5))
        count_data = np.zeros((5, 5))
        
        for _, row in df.iterrows():
            for i in range(1, 6):  # p1 to p5
                col_name = f'p{i}'
                if col_name in row:
                    pos = i - 1  # position index (0-4)
                    val = int(row[col_name]) - 1  # priority value index (0-4)
                    heatmap_data[pos, val] += row[metric]
                    count_data[pos, val] += 1
        
        # Average
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap_data = np.where(count_data > 0, heatmap_data / count_data, 0)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   xticklabels=['1','2','3','4','5'],
                   yticklabels=['p1','p2','p3','p4','p5'],
                   vmin=0, vmax=1, ax=ax)
        ax.set_xlabel('Priority Value')
        ax.set_ylabel('Priority Position')
        ax.set_title(title)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'priority_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved priority heatmap: {output_path}")
    plt.close()
    
    # 4. Precision-Recall plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, metric in zip(axes, ['orientation', 'adjacency', 'cycle']):
        prec_col = f'{metric}_precision'
        rec_col = f'{metric}_recall'
        f1_col = f'{metric}_F1'
        
        scatter = ax.scatter(df[rec_col], df[prec_col], 
                           c=df[f1_col], cmap='viridis', s=50, alpha=0.6)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{metric.capitalize()} Precision vs Recall')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('F1 Score')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'precision_recall.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved precision-recall: {output_path}")
    plt.close()
    
    # Summary statistics
    summary_stats = pd.DataFrame({
        'Metric': ['Orientation', 'Adjacency', 'Cycle', 'Combined'],
        'Best_F1': [
            df['orientation_F1'].max(),
            df['adjacency_F1'].max(),
            df['cycle_F1'].max(),
            df['combined_F1'].max()
        ],
        'Mean_F1': [
            df['orientation_F1'].mean(),
            df['adjacency_F1'].mean(),
            df['cycle_F1'].mean(),
            df['combined_F1'].mean()
        ],
        'Std_F1': [
            df['orientation_F1'].std(),
            df['adjacency_F1'].std(),
            df['cycle_F1'].std(),
            df['combined_F1'].std()
        ]
    })
    
    output_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_stats.to_csv(output_path, index=False)
    print(f"  Saved summary statistics: {output_path}")
    
    print(f"\n  Best configuration: {df.iloc[0]['priorities']}")
    print(f"  Combined F1: {df.iloc[0]['combined_F1']:.4f}")


# ============================================================================
# ORIENTATION-ONLY ANALYSIS
# ============================================================================

def analyze_orientation(df, output_dir):
    """Analyze orientation-only results (simplified version)"""
    print("\n=== Orientation-Only Analysis ===")
    
    # Focus only on orientation metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Orientation-Only Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: F1 distribution
    ax = axes[0, 0]
    ax.hist(df['orientation_F1'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Orientation F1 Score')
    ax.set_ylabel('Frequency')
    ax.set_title('F1 Score Distribution')
    ax.axvline(df['orientation_F1'].mean(), color='r', linestyle='--', 
              label=f'Mean: {df["orientation_F1"].mean():.3f}')
    ax.legend()
    
    # Plot 2: Precision vs Recall
    ax = axes[0, 1]
    scatter = ax.scatter(df['orientation_recall'], df['orientation_precision'],
                        c=df['orientation_F1'], cmap='viridis', alpha=0.6)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.colorbar(scatter, ax=ax, label='F1 Score')
    
    # Plot 3: Top configurations
    ax = axes[1, 0]
    top_n = min(15, len(df))
    df_top = df.nlargest(top_n, 'orientation_F1')
    
    if 'priorities' in df_top.columns:
        labels = df_top['priorities'].astype(str)
    else:
        labels = [f'Config {i+1}' for i in range(len(df_top))]
    
    ax.barh(range(len(df_top)), df_top['orientation_F1'], alpha=0.8)
    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Orientation F1 Score')
    ax.set_title(f'Top {top_n} Configurations')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Metrics comparison
    ax = axes[1, 1]
    metrics = ['precision', 'recall', 'F1']
    means = [df[f'orientation_{m}'].mean() for m in metrics]
    stds = [df[f'orientation_{m}'].std() for m in metrics]
    
    x = np.arange(len(metrics))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylabel('Score')
    ax.set_title('Mean Metrics (±1 std)')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'orientation_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved orientation analysis: {output_path}")
    plt.close()
    
    print(f"\n  Mean F1: {df['orientation_F1'].mean():.4f} ± {df['orientation_F1'].std():.4f}")
    print(f"  Best F1: {df['orientation_F1'].max():.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified VAR analysis and visualization')
    parser.add_argument('--analysis-type', required=True,
                       choices=['delta', 'hyperparameters', 'orientation'],
                       help='Type of analysis to perform')
    parser.add_argument('-f', '--file', required=True, type=str,
                       help='CSV or ZKL file with results')
    parser.add_argument('-o', '--output-dir', default=None,
                       help='Output directory for plots (default: same as input file)')
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.file) or '.'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {args.file}")
    if args.file.endswith('.csv'):
        df = pd.read_csv(args.file)
    elif args.file.endswith('.zkl'):
        data = zkl.load(args.file)
        if isinstance(data, dict) and 'results' in data:
            df = pd.DataFrame(data['results'])
        else:
            df = pd.DataFrame(data)
    else:
        raise ValueError("File must be .csv or .zkl")
    
    print(f"Loaded {len(df)} results")
    
    # Run appropriate analysis
    if args.analysis_type == 'delta':
        analyze_delta(df, args.output_dir)
    elif args.analysis_type == 'hyperparameters':
        analyze_hyperparameters(df, args.output_dir)
    elif args.analysis_type == 'orientation':
        analyze_orientation(df, args.output_dir)
    
    print(f"\n✓ Analysis complete! Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

