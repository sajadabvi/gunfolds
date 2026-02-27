"""
Collect and combine results from parallel hyperparameter tuning jobs
"""
import os
import pandas as pd
import glob
import argparse
from datetime import datetime
from gunfolds.utils import zickle as zkl
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Collect parallel hyperparameter tuning results.')
    parser.add_argument("-n", "--NET", default=3, help="network number", type=int)
    parser.add_argument("-u", "--UNDERSAMPLING", default=3, help="undersampling rate", type=int)
    parser.add_argument("--input_dir", default="VAR_ruben/hyperparameter_tuning/individual_jobs",
                       help="directory with individual job results")
    parser.add_argument("--output_dir", default="VAR_ruben/hyperparameter_tuning",
                       help="directory for combined results")
    parser.add_argument("--min_jobs", default=3000, type=int,
                       help="minimum number of jobs required (default: 3000 out of 3125)")
    return parser.parse_args()

def collect_results(input_dir, net, undersampling):
    """
    Collect all individual CSV results into a single DataFrame
    """
    print(f"Searching for results in: {input_dir}")
    
    # Find all CSV files for this network and undersampling
    pattern = f"{input_dir}/job_*_net{net}_u{undersampling}.csv"
    csv_files = glob.glob(pattern)
    
    print(f"Found {len(csv_files)} result files")
    
    if len(csv_files) == 0:
        print(f"ERROR: No result files found matching pattern: {pattern}")
        return None
    
    # Read and combine all CSVs
    all_results = []
    failed_jobs = []
    
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            all_results.append(df)
        except Exception as e:
            job_id = os.path.basename(csv_file).split('_')[1]
            failed_jobs.append((job_id, str(e)))
            print(f"Warning: Could not read {csv_file}: {e}", file=sys.stderr)
    
    if not all_results:
        print("ERROR: No valid result files could be read")
        return None
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Sort by combined F1 score (descending)
    combined_df = combined_df.sort_values('combined_F1', ascending=False)
    combined_df = combined_df.reset_index(drop=True)
    
    print(f"\n{'=' * 80}")
    print(f"Collection Summary:")
    print(f"  Total jobs collected: {len(combined_df)}")
    print(f"  Failed jobs: {len(failed_jobs)}")
    print(f"  Success rate: {len(combined_df)/3125*100:.1f}%")
    
    if failed_jobs:
        print(f"\n  Failed job IDs: {', '.join([j[0] for j in failed_jobs[:10]])}", end="")
        if len(failed_jobs) > 10:
            print(f" ... and {len(failed_jobs)-10} more")
        else:
            print()
    
    print(f"{'=' * 80}\n")
    
    return combined_df, failed_jobs

def generate_summary(df):
    """
    Generate summary statistics
    """
    print("=" * 80)
    print("TOP 10 PRIORITY CONFIGURATIONS:")
    print("=" * 80)
    
    top_10 = df.head(10)[['job_id', 'priorities', 'orientation_F1', 'adjacency_F1', 'cycle_F1', 'combined_F1']]
    print(top_10.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS:")
    print("=" * 80)
    
    metrics = ['combined_F1', 'orientation_F1', 'adjacency_F1', 'cycle_F1']
    
    print("\n{:<20} {:>10} {:>10} {:>10} {:>10}".format("Metric", "Mean", "Std", "Min", "Max"))
    print("-" * 80)
    
    for metric in metrics:
        mean = df[metric].mean()
        std = df[metric].std()
        min_val = df[metric].min()
        max_val = df[metric].max()
        print("{:<20} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            metric, mean, std, min_val, max_val))
    
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION:")
    print("=" * 80)
    
    best = df.iloc[0]
    print(f"\nJob ID: {best['job_id']}")
    print(f"Priorities: {best['priorities']}")
    print(f"Combined F1: {best['combined_F1']:.4f}")
    print(f"  - Orientation F1: {best['orientation_F1']:.4f}")
    print(f"  - Adjacency F1: {best['adjacency_F1']:.4f}")
    print(f"  - Cycle F1: {best['cycle_F1']:.4f}")
    print(f"Successful batches: {best['num_successful_batches']}")

def main():
    args = parse_arguments()
    
    print("=" * 80)
    print("PARALLEL HYPERPARAMETER TUNING - RESULTS COLLECTION")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Network: {args.NET}")
    print(f"Undersampling: {args.UNDERSAMPLING}")
    print(f"Input directory: {args.input_dir}")
    print("=" * 80)
    print()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        print("Make sure your SLURM jobs have completed and saved results.")
        sys.exit(1)
    
    # Collect results
    result = collect_results(args.input_dir, args.NET, args.UNDERSAMPLING)
    
    if result is None:
        print("ERROR: Failed to collect results")
        sys.exit(1)
    
    combined_df, failed_jobs = result
    
    # Check if we have enough results
    if len(combined_df) < args.min_jobs:
        print(f"\nWARNING: Only collected {len(combined_df)} jobs out of 3125 total")
        print(f"         This is below the minimum threshold of {args.min_jobs}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(1)
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save combined results
    csv_filename = f'{args.output_dir}/priority_tuning_net{args.NET}_u{args.UNDERSAMPLING}_{timestamp}_parallel.csv'
    combined_df.to_csv(csv_filename, index=False)
    print(f"\n✓ Combined results saved to: {csv_filename}")
    
    # Save as ZKL with metadata
    zkl_filename = f'{args.output_dir}/priority_tuning_net{args.NET}_u{args.UNDERSAMPLING}_{timestamp}_parallel.zkl'
    zkl_data = {
        'results': combined_df.to_dict('records'),
        'metadata': {
            'net': args.NET,
            'undersampling': args.UNDERSAMPLING,
            'total_jobs': 3125,
            'collected_jobs': len(combined_df),
            'failed_jobs': len(failed_jobs),
            'timestamp': timestamp,
            'failed_job_ids': [j[0] for j in failed_jobs]
        }
    }
    zkl.save(zkl_data, zkl_filename)
    print(f"✓ Full data saved to: {zkl_filename}")
    
    # Save failed jobs list
    if failed_jobs:
        failed_filename = f'{args.output_dir}/failed_jobs_net{args.NET}_u{args.UNDERSAMPLING}_{timestamp}.txt'
        with open(failed_filename, 'w') as f:
            f.write(f"Failed Jobs: {len(failed_jobs)} out of 3125\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 80 + "\n")
            for job_id, error in failed_jobs:
                f.write(f"Job {job_id}: {error}\n")
        print(f"✓ Failed jobs list saved to: {failed_filename}")
    
    # Generate summary
    print()
    generate_summary(combined_df)
    
    # Instructions for next steps
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("\n1. Analyze results:")
    print(f"   python VAR_analyze_hyperparameters.py -f {csv_filename}")
    print("\n2. Generate report:")
    print(f"   python VAR_generate_report.py -f {csv_filename}")
    print("\n" + "=" * 80)
    print("Collection complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

