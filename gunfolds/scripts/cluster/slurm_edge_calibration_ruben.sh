#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 15
#SBATCH --mem=10g
#SBATCH -p qTRDGPU
#SBATCH -t 1-00:00:00
#SBATCH -J ecal_rub
#SBATCH -e ./err/ecal_ruben_error%A-%a.err
#SBATCH -o ./out/ecal_ruben_out%A-%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mabavisani@gsu.edu
#SBATCH --oversubscribe

# =============================================================================
# Edge Frequency Calibration (Ruben Data) - SLURM Array Job Script
# =============================================================================
# Runs ONE task (network × batch combination) per array element.
#
# Usage:
#   sbatch --array=0-239 slurm_edge_calibration_ruben.sh <TIMESTAMP> [EXTRA_ARGS...]
#
# Example:
#   sbatch --array=0-239%60 slurm_edge_calibration_ruben.sh 03302026120000
#   sbatch --array=0-239%60 slurm_edge_calibration_ruben.sh 03302026120000 --tau_max 2
# =============================================================================

TIMESTAMP=$1
shift
EXTRA_ARGS="$@"

if [ -z "$TIMESTAMP" ]; then
    echo "Error: TIMESTAMP not provided"
    echo "Usage: sbatch --array=0-N slurm_edge_calibration_ruben.sh <TIMESTAMP> [EXTRA_ARGS...]"
    exit 1
fi

TASK_ID=$SLURM_ARRAY_TASK_ID

# Directory setup
LOG_DIR="./logs"
mkdir -p "$LOG_DIR" "./err" "./out"

# Master log (created once per array job)
MASTER_LOG="$LOG_DIR/ecal_ruben_master_${SLURM_ARRAY_JOB_ID}.log"
if [ ! -f "$MASTER_LOG" ]; then
    {
        echo "=============================================================="
        echo "EDGE FREQUENCY CALIBRATION (RUBEN DATA) - MASTER LOG"
        echo "=============================================================="
        echo "Date:          $(date)"
        echo "Job Array ID:  $SLURM_ARRAY_JOB_ID"
        echo "Timestamp:     $TIMESTAMP"
        echo "Extra Args:    $EXTRA_ARGS"
        echo "=============================================================="
    } > "$MASTER_LOG"
fi

# Per-task log
TASK_LOG="$LOG_DIR/ecal_ruben_task_${TASK_ID}.log"
{
    echo "Task ID:      $TASK_ID"
    echo "Start Time:   $(date)"
    echo "Hostname:     $HOSTNAME"
    echo "Node:         $SLURM_NODELIST"
} > "$TASK_LOG"

# =============================================================================
# Environment
# =============================================================================
job_name="ecal_rub_${TASK_ID}"
scontrol update jobid=$SLURM_JOB_ID name=$job_name

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-15}
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

echo "Activating conda environment..." >&2
source /home/users/mabavisani/anaconda3/etc/profile.d/conda.sh
conda activate multi_v3

echo "==========================================="
echo "Edge Frequency Calibration (Ruben Data)"
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

CMD="python edge_frequency_calibration_ruben.py run \
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
