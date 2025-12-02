#!/usr/bin/env python
"""
Quick script to analyze hyperparameter tuning results and generate plots
Automatically finds the most recent results file or you can specify one
"""
import os
import sys
import glob
import argparse

def find_latest_results():
    """Find the most recent results CSV file"""
    # Search in common locations
    search_paths = [
        'VAR_ruben/hyperparameter_tuning/*.csv',
        'VAR_ruben/hyperparameter_tuning/priority_tuning_*.csv',
        '*.csv',
        'priority_tuning_*.csv',
    ]
    
    all_files = []
    for pattern in search_paths:
        all_files.extend(glob.glob(pattern))
    
    # Filter for priority tuning files
    priority_files = [f for f in all_files if 'priority_tuning' in f]
    
    if priority_files:
        # Sort by modification time, newest first
        priority_files.sort(key=os.path.getmtime, reverse=True)
        return priority_files[0]
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter tuning results')
    parser.add_argument('-f', '--file', help='CSV results file (auto-detects if not specified)')
    parser.add_argument('--skip-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--skip-report', action='store_true', help='Skip report generation')
    args = parser.parse_args()
    
    # Find results file
    if args.file:
        results_file = args.file
    else:
        print("Searching for results file...")
        results_file = find_latest_results()
        
        if not results_file:
            print("\nERROR: No results file found!")
            print("\nPlease either:")
            print("  1. Run with -f option: python quick_analyze_results.py -f /path/to/results.csv")
            print("  2. Or place results.csv in current directory")
            sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(results_file):
        print(f"ERROR: File not found: {results_file}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Found results file: {results_file}")
    print(f"{'='*80}\n")
    
    # Get absolute path
    results_file = os.path.abspath(results_file)
    output_dir = os.path.dirname(results_file)
    
    # Generate plots
    if not args.skip_plots:
        print("Generating plots and analysis...")
        import subprocess
        
        try:
            cmd = ['python', 'VAR_analyze_hyperparameters.py', '-f', results_file, '-o', output_dir]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            print("✓ Plots generated successfully!\n")
        except subprocess.CalledProcessError as e:
            print(f"Error generating plots: {e}")
            print(e.stderr)
    
    # Generate report
    if not args.skip_report:
        print("Generating comprehensive report...")
        import subprocess
        
        try:
            report_file = results_file.replace('.csv', '_report.md')
            cmd = ['python', 'VAR_generate_report.py', '-f', results_file, '-o', report_file]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            print("✓ Report generated successfully!\n")
        except subprocess.CalledProcessError as e:
            print(f"Error generating report: {e}")
            print(e.stderr)
    
    # Summary
    print(f"\n{'='*80}")
    print("COMPLETE! Files generated:")
    print(f"{'='*80}")
    print(f"\nResults: {results_file}")
    
    if not args.skip_plots:
        print(f"\nPlots in: {output_dir}/")
        print("  - f1_comparison.png")
        print("  - combined_f1.png")
        print("  - priority_heatmap.png")
        print("  - precision_recall.png")
        print("  - summary_statistics.csv")
    
    if not args.skip_report:
        report_file = results_file.replace('.csv', '_report.md')
        print(f"\nReport: {report_file}")
    
    print(f"\n{'='*80}")
    print("You can now:")
    print(f"  - View plots: open {output_dir}/*.png")
    if not args.skip_report:
        print(f"  - Read report: open {report_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

