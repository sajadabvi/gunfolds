"""
Generate a comprehensive report from hyperparameter tuning results
"""
import pandas as pd
import argparse
import os
from datetime import datetime
from gunfolds.utils import zickle as zkl

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate report from hyperparameter tuning results.')
    parser.add_argument("-f", "--file", required=True, help="CSV file with results", type=str)
    parser.add_argument("-o", "--output", default="", help="output file path", type=str)
    parser.add_argument("--top_n", default=10, help="number of top results to include", type=int)
    return parser.parse_args()

def load_results(filename):
    """Load results from CSV or ZKL file"""
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    elif filename.endswith('.zkl'):
        data = zkl.load(filename)
        df = pd.DataFrame(data['results'])
    else:
        raise ValueError("File must be .csv or .zkl")
    return df

def generate_markdown_report(df, output_file, top_n=10):
    """Generate a markdown report"""
    
    report = []
    report.append("# RASL Priority Hyperparameter Tuning Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Total Configurations Tested:** {len(df)}")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    best = df.iloc[0]
    report.append(f"**Best Configuration:** `{best['priorities']}`")
    report.append(f"- Combined F1 Score: **{best['combined_F1']:.4f}**")
    report.append(f"- Orientation F1: {best['orientation_F1']:.4f}")
    report.append(f"- Adjacency F1: {best['adjacency_F1']:.4f}")
    report.append(f"- Cycle F1: {best['cycle_F1']:.4f}")
    report.append("")
    
    # Overall Statistics
    report.append("## Overall Statistics")
    report.append("")
    report.append("| Metric | Mean | Std | Min | Max |")
    report.append("|--------|------|-----|-----|-----|")
    
    metrics = ['combined_F1', 'orientation_F1', 'adjacency_F1', 'cycle_F1']
    metric_names = ['Combined F1', 'Orientation F1', 'Adjacency F1', 'Cycle F1']
    
    for metric, name in zip(metrics, metric_names):
        mean = df[metric].mean()
        std = df[metric].std()
        min_val = df[metric].min()
        max_val = df[metric].max()
        report.append(f"| {name} | {mean:.4f} | {std:.4f} | {min_val:.4f} | {max_val:.4f} |")
    
    report.append("")
    
    # Top N Configurations
    report.append(f"## Top {top_n} Configurations")
    report.append("")
    report.append("### Ranked by Combined F1 Score")
    report.append("")
    
    for i, (idx, row) in enumerate(df.head(top_n).iterrows(), 1):
        report.append(f"### {i}. Priorities: `{row['priorities']}`")
        report.append("")
        report.append(f"**Combined F1:** {row['combined_F1']:.4f}")
        report.append("")
        report.append("| Metric | Precision | Recall | F1 |")
        report.append("|--------|-----------|--------|-----|")
        report.append(f"| Orientation | {row['orientation_precision']:.4f} | {row['orientation_recall']:.4f} | {row['orientation_F1']:.4f} |")
        report.append(f"| Adjacency | {row['adjacency_precision']:.4f} | {row['adjacency_recall']:.4f} | {row['adjacency_F1']:.4f} |")
        report.append(f"| Cycle | {row['cycle_precision']:.4f} | {row['cycle_recall']:.4f} | {row['cycle_F1']:.4f} |")
        report.append("")
        report.append(f"*Successful batches: {row['num_successful_batches']}*")
        report.append("")
    
    # Priority Value Analysis
    report.append("## Priority Value Analysis")
    report.append("")
    report.append("### Average Priority Values in Top 10")
    report.append("")
    report.append("| Position | Average Value | Std Dev |")
    report.append("|----------|---------------|---------|")
    
    top_10 = df.head(10)
    for i in range(1, 6):
        avg = top_10[f'p{i}'].mean()
        std = top_10[f'p{i}'].std()
        report.append(f"| p{i} | {avg:.2f} | {std:.2f} |")
    
    report.append("")
    
    # Most Common Priorities
    report.append("### Most Common Priority Configurations (Top 10)")
    report.append("")
    report.append("| Priority | Count |")
    report.append("|----------|-------|")
    
    priority_counts = df['priorities'].value_counts().head(10)
    for priority, count in priority_counts.items():
        report.append(f"| `{priority}` | {count} |")
    
    report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    # Find best for each metric
    best_orientation = df.loc[df['orientation_F1'].idxmax()]
    best_adjacency = df.loc[df['adjacency_F1'].idxmax()]
    best_cycle = df.loc[df['cycle_F1'].idxmax()]
    best_combined = df.loc[df['combined_F1'].idxmax()]
    
    report.append("### Best Configurations by Metric")
    report.append("")
    report.append(f"1. **Best Overall (Combined F1):** `{best_combined['priorities']}` (F1={best_combined['combined_F1']:.4f})")
    report.append(f"2. **Best for Orientation:** `{best_orientation['priorities']}` (F1={best_orientation['orientation_F1']:.4f})")
    report.append(f"3. **Best for Adjacency:** `{best_adjacency['priorities']}` (F1={best_adjacency['adjacency_F1']:.4f})")
    report.append(f"4. **Best for Cycle Detection:** `{best_cycle['priorities']}` (F1={best_cycle['cycle_F1']:.4f})")
    report.append("")
    
    report.append("### Suggested Priority Range")
    report.append("")
    report.append("Based on top 10 performing configurations:")
    report.append("")
    for i in range(1, 6):
        values = top_10[f'p{i}'].values
        min_val = int(values.min())
        max_val = int(values.max())
        mode_val = int(top_10[f'p{i}'].mode().values[0]) if len(top_10[f'p{i}'].mode()) > 0 else 'N/A'
        report.append(f"- **p{i}:** Range [{min_val}, {max_val}], Most common: {mode_val}")
    
    report.append("")
    
    # Performance Distribution
    report.append("## Performance Distribution")
    report.append("")
    
    # Quartiles
    q25 = df['combined_F1'].quantile(0.25)
    q50 = df['combined_F1'].quantile(0.50)
    q75 = df['combined_F1'].quantile(0.75)
    
    report.append(f"- **Q1 (25th percentile):** {q25:.4f}")
    report.append(f"- **Q2 (50th percentile/Median):** {q50:.4f}")
    report.append(f"- **Q3 (75th percentile):** {q75:.4f}")
    report.append("")
    
    high_performers = len(df[df['combined_F1'] >= q75])
    report.append(f"- **High performers (≥Q3):** {high_performers} configurations ({high_performers/len(df)*100:.1f}%)")
    report.append("")
    
    # Notes
    report.append("## Notes")
    report.append("")
    report.append("- F1 scores range from 0.0 (worst) to 1.0 (perfect)")
    report.append("- Combined F1 is the average of Orientation, Adjacency, and Cycle F1 scores")
    report.append("- Higher priority values (1-5) indicate higher importance in optimization")
    report.append("- Equal priority values mean those constraints are weighted equally")
    report.append("")
    
    # Full Results Table
    report.append("## Appendix: Full Results Table")
    report.append("")
    report.append("| Rank | Priorities | Combined F1 | Orient F1 | Adj F1 | Cycle F1 |")
    report.append("|------|-----------|-------------|-----------|--------|----------|")
    
    for i, (idx, row) in enumerate(df.iterrows(), 1):
        report.append(f"| {i} | `{row['priorities']}` | {row['combined_F1']:.4f} | {row['orientation_F1']:.4f} | {row['adjacency_F1']:.4f} | {row['cycle_F1']:.4f} |")
    
    report.append("")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    return '\n'.join(report)

def main():
    args = parse_arguments()
    
    # Load results
    print(f"Loading results from: {args.file}")
    df = load_results(args.file)
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(args.file)[0]
        output_file = f"{base_name}_report.md"
    
    # Generate report
    print(f"Generating report for {len(df)} configurations...")
    report_text = generate_markdown_report(df, output_file, args.top_n)
    
    print(f"\n✓ Report saved to: {output_file}")
    print(f"\nPreview (first 50 lines):")
    print("=" * 80)
    print('\n'.join(report_text.split('\n')[:50]))
    print("=" * 80)
    print(f"\n(See full report in {output_file})")

if __name__ == "__main__":
    main()

