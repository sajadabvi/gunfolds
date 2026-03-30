#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 15
#SBATCH --mem=10g
#SBATCH -p qTRDGPU
#SBATCH -t 1-00:00:00
#SBATCH -J stg_abl
#SBATCH -e ./err/stg_abl_error%A-%a.err
#SBATCH -o ./out/stg_abl_out%A-%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mabavisani@gsu.edu
#SBATCH --oversubscribe

# =============================================================================
# Stage Ablation - SLURM Array Job Script
# =============================================================================
# Runs ONE task (network × undersampling × batch combination) per array element.
# Each task runs all three ablation stages (density-only, density+bidirected,
# full pipeline) and saves results to a per-task file.
#
# Usage:
#   sbatch --array=0-99 slurm_stage_ablation.sh <TIMESTAMP> [EXTRA_ARGS...]
#
# The total number of tasks = len(NETWORKS) × len(UNDERSAMPLING) × BATCHES.
# With defaults (networks=[1,2,3,4,5], u=[2,3], batches=10): 5×2×10 = 100 tasks.
#
# Examples:
#   sbatch --array=0-99%20  slurm_stage_ablation.sh 03302026120000
#   sbatch --array=0-99%20  slurm_stage_ablation.sh 03302026120000 --ssize 10000 --noise 0.05
#   sbatch --array=0-79%20  slurm_stage_ablation.sh 03302026120000 -n 1 2 3 4 -u 2 3
#
# After all tasks complete, aggregate results:
#   python stage_ablation.py aggregate --timestamp 03302026120000
# =============================================================================

TIMESTAMP=$1
shift
EXTRA_ARGS="$@"

if [ -z "$TIMESTAMP" ]; then
    echo "Error: TIMESTAMP not provided"
    echo "Usage: sbatch --array=0-N slurm_stage_ablation.sh <TIMESTAMP> [EXTRA_ARGS...]"
    exit 1
fi

TASK_ID=$SLURM_ARRAY_TASK_ID

# Directory setup
LOG_DIR="./logs"
mkdir -p "$LOG_DIR" "./err" "./out"

# Master log (created once per array job)
MASTER_LOG="$LOG_DIR/stg_abl_master_${SLURM_ARRAY_JOB_ID}.log"
if [ ! -f "$MASTER_LOG" ]; then
    {
        echo "=============================================================="
        echo "STAGE ABLATION - MASTER LOG"
        echo "=============================================================="
        echo "Date:          $(date)"
        echo "Job Array ID:  $SLURM_ARRAY_JOB_ID"
        echo "Timestamp:     $TIMESTAMP"
        echo "Extra Args:    $EXTRA_ARGS"
        echo "=============================================================="
    } > "$MASTER_LOG"
fi

# Per-task log
TASK_LOG="$LOG_DIR/stg_abl_task_${TASK_ID}.log"
{
    echo "Task ID:      $TASK_ID"
    echo "Start Time:   $(date)"
    echo "Hostname:     $HOSTNAME"
    echo "Node:         $SLURM_NODELIST"
} > "$TASK_LOG"

# =============================================================================
# Environment
# =============================================================================
job_name="stg_abl_${TASK_ID}"
scontrol update jobid=$SLURM_JOB_ID name=$job_name

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-15}
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

echo "Activating conda environment..." >&2
source /home/users/mabavisani/anaconda3/etc/profile.d/conda.sh
conda activate multi_v3

echo "==========================================="
echo "Stage Ablation Experiment"
echo "==========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array Task:   $SLURM_ARRAY_TASK_ID"
echo "Task ID:      $TASK_ID"
echo "Timestamp:    $TIMESTAMP"
echo "==========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="${SCRIPT_DIR}/../experiments"
cd "$EXPERIMENT_DIR"

# =============================================================================
# Run experiment
# =============================================================================
PNUM=${SLURM_CPUS_PER_TASK:-15}

CMD="python stage_ablation.py run \
    --task_id $TASK_ID \
    --timestamp $TIMESTAMP \
    --PNUM $PNUM \
    $EXTRA_ARGS"

echo "Executing: $CMD" >&2
eval $CMD
EXIT_CODE=$?

# =============================================================================
# Log completion
# =============================================================================
{
    echo "End Time:   $(date)"
    echo "Exit Code:  $EXIT_CODE"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Status: SUCCESS"
    else
        echo "Status: FAILED (exit code $EXIT_CODE)"
    fi
} >> "$TASK_LOG"

exit $EXIT_CODE
