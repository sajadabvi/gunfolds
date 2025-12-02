#!/bin/bash
# submit_gcm_parallel.sh
# Helper script to submit GCM parallel array job with shared timestamp
# Adapted for GSU URSA cluster with qTRDGPU partition

# =============================================================================
# Configuration
# =============================================================================

# Default parameters
ALPHA=50           # Default alpha = 50 (0.05)
MAX_SUBJECTS=100   # Default, adjust based on your data
CONCURRENT=20      # Number of jobs to run concurrently

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -A|--alpha)
            ALPHA="$2"
            shift 2
            ;;
        -n|--num-subjects)
            MAX_SUBJECTS="$2"
            shift 2
            ;;
        -c|--concurrent)
            CONCURRENT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -A, --alpha ALPHA           Alpha value x1000 (default: 50 = 0.05)"
            echo "  -n, --num-subjects N        Number of subjects (default: 100)"
            echo "  -c, --concurrent N          Number of concurrent jobs (default: 20)"
            echo "  -h, --help                  Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -A 50 -n 50 -c 10"
            echo ""
            echo "Cluster Configuration:"
            echo "  Partition: qTRDGPU"
            echo "  Account: psy53c17"
            echo "  Email: mabavisani@gsu.edu"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Directory Setup
# =============================================================================

# Create necessary directories
mkdir -p logs
mkdir -p err
mkdir -p out

echo "=============================================================="
echo "GCM PARALLEL PROCESSING - SUBMISSION SCRIPT"
echo "=============================================================="
echo "Date and Time: $(date)"
echo "Submit Directory: $(pwd)"
echo "User: $(whoami)"
echo "=============================================================="

# =============================================================================
# Generate Shared Timestamp
# =============================================================================

TIMESTAMP=$(date '+%m%d%Y%H%M%S')

# Create the output directory with timestamp
OUT_DIR="gcm_roebroeck/$TIMESTAMP"
mkdir -p "$OUT_DIR/csv"
mkdir -p "$OUT_DIR/figures"

echo "Created output directory: $OUT_DIR"

# =============================================================================
# Save Run Metadata
# =============================================================================

METADATA_FILE="$OUT_DIR/run_metadata.txt"
{
    echo "=============================================================="
    echo "GCM Parallel Run Metadata"
    echo "=============================================================="
    echo "Timestamp: $TIMESTAMP"
    echo "Submission Time: $(date)"
    echo "Submit Directory: $(pwd)"
    echo "User: $(whoami)"
    echo "Hostname: $(hostname)"
    echo ""
    echo "Parameters:"
    echo "  Alpha: $ALPHA ($(echo "scale=4; $ALPHA/1000" | bc))"
    echo "  Number of Subjects: $MAX_SUBJECTS (0 to $((MAX_SUBJECTS-1)))"
    echo "  Concurrent Jobs: $CONCURRENT"
    echo ""
    echo "Cluster Configuration:"
    echo "  Partition: qTRDGPU"
    echo "  Account: psy53c17"
    echo "  Memory per job: 4GB"
    echo "  Time limit: 7200 seconds (2 hours)"
    echo "  CPUs per job: 1"
    echo ""
    echo "Output Locations:"
    echo "  CSV files: $OUT_DIR/csv/"
    echo "  Figures: $OUT_DIR/figures/"
    echo "  SLURM output: ./out/gcm_out*.out"
    echo "  SLURM errors: ./err/gcm_error*.err"
    echo "  Task logs: ./logs/gcm_task_*.log"
    echo "=============================================================="
} > "$METADATA_FILE"

echo "Saved metadata to: $METADATA_FILE"

# =============================================================================
# Submit Array Job
# =============================================================================

echo ""
echo "=============================================================="
echo "Submitting GCM Parallel Array Job"
echo "=============================================================="
echo "Timestamp: $TIMESTAMP"
echo "Output Directory: $OUT_DIR"
echo "Alpha: $ALPHA ($(echo "scale=4; $ALPHA/1000" | bc))"
echo "Subjects: 0-$((MAX_SUBJECTS-1))"
echo "Concurrent: $CONCURRENT"
echo "Array Spec: 0-$((MAX_SUBJECTS-1))%$CONCURRENT"
echo "=============================================================="

# Calculate array specification
ARRAY_SPEC="0-$((MAX_SUBJECTS-1))%$CONCURRENT"

# Submit the job
SUBMIT_OUTPUT=$(sbatch --array=$ARRAY_SPEC slurm_gcm_parallel.sh $TIMESTAMP $ALPHA 2>&1)
SUBMIT_STATUS=$?

if [ $SUBMIT_STATUS -eq 0 ]; then
    JOB_ID=$(echo "$SUBMIT_OUTPUT" | awk '{print $4}')
    
    echo ""
    echo "✓ Job submitted successfully!"
    echo "=============================================================="
    echo "Job ID: $JOB_ID"
    echo "Timestamp: $TIMESTAMP"
    echo "=============================================================="
    
    # Save job info to metadata
    {
        echo ""
        echo "Job Submission:"
        echo "  Job ID: $JOB_ID"
        echo "  Array Spec: $ARRAY_SPEC"
        echo "  Submission Status: SUCCESS"
    } >> "$METADATA_FILE"
    
    # Create convenience files
    echo "$TIMESTAMP" > logs/latest_timestamp.txt
    echo "$JOB_ID" > logs/latest_job_id.txt
    
    # Print monitoring commands
    echo ""
    echo "Monitoring Commands:"
    echo "------------------------------------------------------------"
    echo "Check job status:"
    echo "  squeue -j $JOB_ID"
    echo "  squeue -u $(whoami) | grep gcm"
    echo ""
    echo "Check progress:"
    echo "  ls $OUT_DIR/csv/*Adj_labeled.csv | wc -l"
    echo "  watch -n 30 'ls $OUT_DIR/csv/*Adj_labeled.csv | wc -l'"
    echo ""
    echo "View logs:"
    echo "  tail -f out/gcm_out${JOB_ID}*.out"
    echo "  tail -f err/gcm_error${JOB_ID}*.err"
    echo "  tail -f logs/gcm_task_*.log"
    echo ""
    echo "Watch for completion:"
    echo "  python analyze_gcm_results.py -t $TIMESTAMP --watch --expected-subjects $MAX_SUBJECTS"
    echo ""
    echo "Analyze results (after completion):"
    echo "  python analyze_gcm_results.py -t $TIMESTAMP --aggregate --plot"
    echo ""
    echo "Check failed jobs:"
    echo "  sacct -j $JOB_ID --format=JobID,State,ExitCode | grep -v COMPLETED"
    echo "=============================================================="
    
    # Create a quick check script
    CHECK_SCRIPT="logs/check_gcm_progress_${TIMESTAMP}.sh"
    {
        echo "#!/bin/bash"
        echo "# Quick progress check for GCM run $TIMESTAMP"
        echo ""
        echo "TIMESTAMP=$TIMESTAMP"
        echo "JOB_ID=$JOB_ID"
        echo "EXPECTED=$MAX_SUBJECTS"
        echo ""
        echo "echo '============================================================'"
        echo "echo 'GCM Progress Check'"
        echo "echo '============================================================'"
        echo "echo \"Timestamp: \$TIMESTAMP\""
        echo "echo \"Job ID: \$JOB_ID\""
        echo "echo ''"
        echo ""
        echo "# Check job status"
        echo "echo 'Job Status:'"
        echo "squeue -j \$JOB_ID 2>/dev/null || echo '  No jobs running (may be complete or not started)'"
        echo "echo ''"
        echo ""
        echo "# Count completed subjects"
        echo "COMPLETED=\$(ls gcm_roebroeck/\$TIMESTAMP/csv/*Adj_labeled.csv 2>/dev/null | wc -l)"
        echo "PERCENT=\$(echo \"scale=1; \$COMPLETED * 100 / \$EXPECTED\" | bc 2>/dev/null || echo '0')"
        echo "echo \"Progress: \$COMPLETED/\$EXPECTED (\$PERCENT%)\""
        echo "echo ''"
        echo ""
        echo "# Check for failures"
        echo "echo 'Failed Jobs:'"
        echo "sacct -j \$JOB_ID --format=JobID,State,ExitCode 2>/dev/null | grep -E 'FAILED|TIMEOUT|CANCELLED' || echo '  None'"
        echo "echo '============================================================'"
    } > "$CHECK_SCRIPT"
    chmod +x "$CHECK_SCRIPT"
    
    echo ""
    echo "Created progress check script: $CHECK_SCRIPT"
    echo "Run with: ./$CHECK_SCRIPT"
    echo ""
    
else
    echo ""
    echo "✗ Job submission failed!"
    echo "=============================================================="
    echo "Error output:"
    echo "$SUBMIT_OUTPUT"
    echo "=============================================================="
    
    # Save failure info to metadata
    {
        echo ""
        echo "Job Submission:"
        echo "  Submission Status: FAILED"
        echo "  Error: $SUBMIT_OUTPUT"
    } >> "$METADATA_FILE"
    
    exit 1
fi

