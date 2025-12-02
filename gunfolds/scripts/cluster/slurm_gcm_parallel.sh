#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4g
#SBATCH -p qTRDGPU
#SBATCH -t 7200
#SBATCH -J gcm_parallel
#SBATCH -e ./err/gcm_error%A-%a.err
#SBATCH -o ./out/gcm_out%A-%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mabavisani@gsu.edu
#SBATCH --oversubscribe

# =============================================================================
# GCM Parallel Processing - Array Job Script
# =============================================================================
# This script runs ONE subject (one array task) of the GCM analysis
# Usage:
#   sbatch --array=0-N slurm_gcm_parallel.sh <TIMESTAMP> <ALPHA>
# Example:
#   sbatch --array=0-49 slurm_gcm_parallel.sh 12262025103045 50
# =============================================================================

# Get timestamp from command line argument (shared across all array jobs)
TIMESTAMP=$1
ALPHA=${2:-50}  # Default alpha = 50 (0.05) if not provided

# Validate inputs
if [ -z "$TIMESTAMP" ]; then
    echo "Error: TIMESTAMP not provided"
    echo "Usage: sbatch --array=0-N slurm_gcm_parallel.sh <TIMESTAMP> [ALPHA]"
    exit 1
fi

# Directory setup
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
mkdir -p "./err"
mkdir -p "./out"

# Subject index from array task ID
SUBJECT_IDX=$SLURM_ARRAY_TASK_ID

# =============================================================================
# Logging
# =============================================================================

# Create a master log file for the entire array job (only once)
MASTER_LOG="$LOG_DIR/gcm_master_job_${SLURM_ARRAY_JOB_ID}.log"
if [ ! -f "$MASTER_LOG" ]; then
    {
        echo "=============================================================="
        echo "GCM PARALLEL PROCESSING - MASTER LOG"
        echo "=============================================================="
        echo "Date and Time: $(date)"
        echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
        echo "Timestamp: $TIMESTAMP"
        echo "Alpha: $ALPHA ($(echo "scale=4; $ALPHA/1000" | bc))"
        echo "Output Directory: gcm_roebroeck/$TIMESTAMP"
        echo "=============================================================="
        echo -e "\nContents of this SLURM script:"
        cat "$0"
        echo "=============================================================="
    } > "$MASTER_LOG"
fi

# Individual task log
TASK_LOG="$LOG_DIR/gcm_task_${SLURM_ARRAY_TASK_ID}.log"
{
    echo "Task ID: $SLURM_ARRAY_TASK_ID"
    echo "Subject Index: $SUBJECT_IDX"
    echo "Timestamp: $TIMESTAMP"
    echo "Alpha: $ALPHA"
    echo "Start Time: $(date)"
    echo "Hostname: $HOSTNAME"
    echo "Node: $SLURM_NODELIST"
} > "$TASK_LOG"

# =============================================================================
# Environment Setup
# =============================================================================

# Update job name
job_name="gcm_${SUBJECT_IDX}"
scontrol update jobid=$SLURM_JOB_ID name=$job_name

# Set OpenMP threads
export OMP_NUM_THREADS=1

# Load modules
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

# Activate conda environment
echo "Activating conda environment..." >&2
source /home/users/mabavisani/anaconda3/bin/activate
conda activate multi_v3

echo "==========================================="
echo "GCM Parallel Processing"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Subject Index: $SUBJECT_IDX"
echo "Timestamp: $TIMESTAMP"
echo "Alpha: $ALPHA"
echo "Node: $SLURM_NODELIST"
echo "Hostname: $HOSTNAME"
echo "==========================================="

# Navigate to scripts directory
cd $SLURM_SUBMIT_DIR

# =============================================================================
# Run GCM Analysis
# =============================================================================

echo "Executing: python gcm_on_ICA.py -S $SUBJECT_IDX -t $TIMESTAMP -A $ALPHA" >&2

# Run GCM for this subject with the shared timestamp
python gcm_on_ICA.py -S $SUBJECT_IDX -t $TIMESTAMP -A $ALPHA

EXIT_CODE=$?

# =============================================================================
# Logging Completion
# =============================================================================

{
    echo "End Time: $(date)"
    echo "Exit Code: $EXIT_CODE"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Status: SUCCESS"
        echo "Subject $SUBJECT_IDX completed successfully"
    else
        echo "Status: FAILED"
        echo "Subject $SUBJECT_IDX failed with exit code $EXIT_CODE"
    fi
} >> "$TASK_LOG"

exit $EXIT_CODE

