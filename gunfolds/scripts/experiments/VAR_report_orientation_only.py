"""
Generate report from hyperparameter tuning results - ORIENTATION F1 ONLY
All rankings based solely on orientation_F1
"""
import pandas as pd
import argparse
import os
from datetime import datetime
from gunfolds.utils import zickle as zkl

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate report (Orientation F1 only).')
    parser.add_argument("-f", "--file", required=True, help="CSV file with results", type=str)
    parser.add_argument("-o", "--output", default="", help="output file path", type=str)
    parser.add_argument("--top_n", default=20, help="number of top results to include", type=int)
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

def generate_markdown_report(df, output_file, top_n=20):
    """Generate a markdown report focused on orientation F1"""
    
    report = []
    report.append("# RASL Priority Hyperparameter Tuning Report")
    report.append("# **ORIENTATION F1 ONLY** - All rankings by Orientation F1")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Total Configurations Tested:** {len(df)}")
    report.append(f"**Ranking Criterion:** Orientation F1 (edge direction accuracy)")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    best = df.iloc[0]
    report.append(f"**Best Configuration:** `{best['priorities']}`")
    report.append(f"- **Orientation F1 Score: {best['orientation_F1']:.4f}**")
    report.append(f"- Orientation Precision: {best['orientation_precision']:.4f}")
    report.append(f"- Orientation Recall: {best['orientation_recall']:.4f}")
    report.append("")
    
    # Overall Statistics
    report.append("## Overall Statistics (Orientation Metrics)")
    report.append("")
    report.append("| Metric | Mean | Std | Min | Max |")
    report.append("|--------|------|-----|-----|-----|")
    
    metrics = ['orientation_F1', 'orientation_precision', 'orientation_recall']
    metric_names = ['Orientation F1', 'Orientation Precision', 'Orientation Recall']
    
    for metric, name in zip(metrics, metric_names):
        mean = df[metric].mean()
        std = df[metric].std()
        min_val = df[metric].min()
        max_val = df[metric].max()
        report.append(f"| {name} | {mean:.4f} | {std:.4f} | {min_val:.4f} | {max_val:.4f} |")
    
    report.append("")
    
    # Top N Configurations
    report.append(f"## Top {top_n} Configurations (Ranked by Orientation F1)")
    report.append("")
    
    for i, (idx, row) in enumerate(df.head(top_n).iterrows(), 1):
        report.append(f"### {i}. Priorities: `{row['priorities']}`")
        report.append("")
        report.append(f"**Orientation F1:** {row['orientation_F1']:.4f}")
        report.append("")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| **F1 Score** | **{row['orientation_F1']:.4f}** |")
        report.append(f"| Precision | {row['orientation_precision']:.4f} |")
        report.append(f"| Recall | {row['orientation_recall']:.4f} |")
        report.append("")
        report.append(f"*Successful batches: {row['num_successful_batches']}*")
        report.append("")
    
    # Priority Value Analysis
    report.append("## Priority Value Analysis (Top 10)")
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
    
    # Distribution Analysis
    report.append("## Performance Distribution")
    report.append("")
    
    q25 = df['orientation_F1'].quantile(0.25)
    q50 = df['orientation_F1'].quantile(0.50)
    q75 = df['orientation_F1'].quantile(0.75)
    
    report.append(f"- **Q1 (25th percentile):** {q25:.4f}")
    report.append(f"- **Q2 (50th percentile/Median):** {q50:.4f}")
    report.append(f"- **Q3 (75th percentile):** {q75:.4f}")
    report.append("")
    
    high_performers = len(df[df['orientation_F1'] >= q75])
    report.append(f"- **High performers (≥Q3):** {high_performers} configurations ({high_performers/len(df)*100:.1f}%)")
    report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    best_config = df.iloc[0]
    
    report.append("### Best Configuration")
    report.append("")
    report.append(f"**Use priorities: `{best_config['priorities']}`**")
    report.append("")
    report.append(f"This configuration achieved:")
    report.append(f"- Orientation F1: **{best_config['orientation_F1']:.4f}**")
    report.append(f"- Orientation Precision: {best_config['orientation_precision']:.4f}")
    report.append(f"- Orientation Recall: {best_config['orientation_recall']:.4f}")
    report.append("")
    
    # Priority recommendations
    report.append("### Suggested Priority Range (from top 10)")
    report.append("")
    for i in range(1, 6):
        values = top_10[f'p{i}'].values
        min_val = int(values.min())
        max_val = int(values.max())
        mode_val = int(top_10[f'p{i}'].mode().values[0]) if len(top_10[f'p{i}'].mode()) > 0 else 'N/A'
        report.append(f"- **p{i}:** Range [{min_val}, {max_val}], Most common: {mode_val}")
    
    report.append("")
    
    # How to use
    report.append("## How to Use These Results")
    report.append("")
    report.append("Update your code (e.g., `VAR_for_ruben_nets.py` line 152) with the best priorities:")
    report.append("")
    report.append("```python")
    report.append(f"priorities = {list(eval(best_config['priorities']))}  # Best for Orientation F1")
    report.append("```")
    report.append("")
    
    # Full Results Table
    report.append("## Appendix: Full Results Table (Top 50)")
    report.append("")
    report.append("| Rank | Priorities | Orientation F1 | Precision | Recall |")
    report.append("|------|-----------|----------------|-----------|--------|")
    
    for i, (idx, row) in enumerate(df.head(50).iterrows(), 1):
        report.append(f"| {i} | `{row['priorities']}` | {row['orientation_F1']:.4f} | {row['orientation_precision']:.4f} | {row['orientation_recall']:.4f} |")
    
    report.append("")
    
    # Notes
    report.append("## Notes")
    report.append("")
    report.append("- **Ranking criterion:** Orientation F1 ONLY")
    report.append("- **Orientation F1:** Measures edge direction accuracy (most important for causal inference)")
    report.append("- **Precision:** True positives / (True positives + False positives)")
    report.append("- **Recall:** True positives / (True positives + False negatives)")
    report.append("- **F1 Score:** Harmonic mean of precision and recall")
    report.append("- Higher priority values (1-5) indicate higher importance in optimization")
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
    
    # Sort by orientation F1 ONLY
    df = df.sort_values('orientation_F1', ascending=False)
    df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} configurations")
    print(f"Sorted by: Orientation F1 (descending)")
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(args.file)[0]
        output_file = f"{base_name}_report_ORIENTATION_ONLY.md"
    
    # Generate report
    print(f"\nGenerating report for {len(df)} configurations...")
    report_text = generate_markdown_report(df, output_file, args.top_n)
    
    print(f"\n✓ Report saved to: {output_file}")
    
    # Print summary
    best = df.iloc[0]
    print(f"\n{'=' * 80}")
    print("BEST CONFIGURATION (BY ORIENTATION F1)")
    print(f"{'=' * 80}")
    print(f"Priorities: {best['priorities']}")
    print(f"Orientation F1: {best['orientation_F1']:.4f}")
    print(f"Orientation Precision: {best['orientation_precision']:.4f}")
    print(f"Orientation Recall: {best['orientation_recall']:.4f}")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()

