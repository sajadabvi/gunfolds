#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=2g
#SBATCH -p qTRDGPU
#SBATCH -t 7600
#SBATCH -J delta_tune
#SBATCH -e ./err/error%A-%a.err
#SBATCH -o ./out/out%A-%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mabavisani@gsu.edu
#SBATCH --oversubscribe

# =============================================================================
# Delta Hyperparameter Tuning - Parallel Execution Script
# =============================================================================
# This script runs ONE delta configuration (out of N total)
# Submit with: sbatch --array=1-N slurm_delta_tuning.sh
#
# To calculate N (number of jobs):
#   N = (DELTA_MAX - DELTA_MIN) / DELTA_STEP + 1
#
# Example: delta_min=0, delta_max=100000, delta_step=5000
#   N = (100000 - 0) / 5000 + 1 = 21 jobs
#
# Submit with: sbatch --array=1-21 slurm_delta_tuning.sh
# =============================================================================

# Python script to run
PYTHON_SCRIPT="VAR_delta_single_job.py"

# Configuration parameters (adjust these as needed)
NET=3                    # Network number
UNDERSAMPLING=3          # Undersampling rate
NUM_BATCHES=5           # Number of batches to average over
PNUM=1                  # CPUs per job (typically 1 for parallel array jobs)

# Delta parameters - ADJUST THESE FOR YOUR EXPERIMENT
# Delta is now specified as a PERCENTAGE of the minimum cost
# Example: DELTA_MIN=0.0 means 0% of min_cost
#          DELTA_MAX=2.0 means 200% of min_cost
#          DELTA_STEP=0.1 means 10% increments

DELTA_MIN=0.0           # Minimum delta as fraction (0.0 = 0% of min_cost)
DELTA_MAX=2.0           # Maximum delta as fraction (2.0 = 200% of min_cost)
DELTA_STEP=0.1          # Step size as fraction (0.1 = 10% increments)

# =============================================================================
# IMPORTANT: Calculate number of jobs before submitting!
# N = (DELTA_MAX - DELTA_MIN) / DELTA_STEP + 1
# Example: (2.0 - 0.0) / 0.1 + 1 = 21 jobs
# Then submit with: sbatch --array=1-N slurm_delta_tuning.sh
# =============================================================================

# Directory setup
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
mkdir -p "./err"
mkdir -p "./out"

# Job-specific parameters
S=$SLURM_ARRAY_TASK_ID

# Combine parameters for the Python script
SCRIPT_PARAMS="-S $S -n $NET -u $UNDERSAMPLING --num_batches $NUM_BATCHES -p $PNUM"
SCRIPT_PARAMS="$SCRIPT_PARAMS --delta_min $DELTA_MIN --delta_max $DELTA_MAX --delta_step $DELTA_STEP"

# =============================================================================
# Logging
# =============================================================================

# Create a master log file for the entire array job (only once)
MASTER_LOG="$LOG_DIR/master_job_${SLURM_ARRAY_JOB_ID}.log"
if [ ! -f "$MASTER_LOG" ]; then
    {
        echo "=============================================================="
        echo "DELTA HYPERPARAMETER TUNING - MASTER LOG"
        echo "=============================================================="
        echo "Date and Time: $(date)"
        echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
        echo "Network: $NET"
        echo "Undersampling: $UNDERSAMPLING"
        echo "Batches per config: $NUM_BATCHES"
        echo "CPUs per job: $PNUM"
        echo "Delta range: ${DELTA_MIN} to ${DELTA_MAX} (step: ${DELTA_STEP})"
        echo "Delta interpretation: Percentage of minimum cost"
        echo "Metric: Orientation F1 only"
        echo "=============================================================="
        echo -e "\nContents of $PYTHON_SCRIPT:"
        cat "$PYTHON_SCRIPT"
        echo "=============================================================="
        echo -e "\nContents of this SLURM script:"
        cat "$0"
        echo "=============================================================="
    } > "$MASTER_LOG"
fi

# Individual task log
TASK_LOG="$LOG_DIR/task_${SLURM_ARRAY_TASK_ID}.log"
{
    echo "Task ID: $SLURM_ARRAY_TASK_ID"
    echo "Start Time: $(date)"
    echo "Hostname: $HOSTNAME"
    echo "Script Parameters: $SCRIPT_PARAMS"
} > "$TASK_LOG"

# =============================================================================
# Environment Setup
# =============================================================================

# Update job name
job_name="delta_${S}"
scontrol update jobid=$SLURM_JOB_ID name=$job_name

# Set number of CPUs
scontrol update jobid=$SLURM_JOB_ID NumCPUs=$PNUM

# Set OpenMP threads
export OMP_NUM_THREADS=$PNUM

# Load modules
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

# Activate conda environment
echo "Activating conda environment..." >&2
source /home/users/mabavisani/anaconda3/bin/activate
conda activate multi_v3

echo "Hostname: $HOSTNAME" >&2
echo "Job ID: $SLURM_JOB_ID" >&2
echo "Array Task ID: $SLURM_ARRAY_TASK_ID" >&2
echo "Running delta configuration $S" >&2

# =============================================================================
# Run Python Script
# =============================================================================

echo "Executing: python $PYTHON_SCRIPT $SCRIPT_PARAMS" >&2

# Run the script and capture exit code
python "$PYTHON_SCRIPT" $SCRIPT_PARAMS
EXIT_CODE=$?

# =============================================================================
# Logging Completion
# =============================================================================

{
    echo "End Time: $(date)"
    echo "Exit Code: $EXIT_CODE"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Status: SUCCESS"
    else
        echo "Status: FAILED"
    fi
} >> "$TASK_LOG"

# Exit with the same code as the Python script
exit $EXIT_CODE

