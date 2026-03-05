#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8g
#SBATCH -p qTRDGPU
#SBATCH -t 7200
#SBATCH -J fmri_large
#SBATCH -e ./err/fmri_error%A-%a.err
#SBATCH -o ./out/fmri_out%A-%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mabavisani@gsu.edu
#SBATCH --oversubscribe

# =============================================================================
# fMRI Large Experiment - Array Job Script
# =============================================================================
# Runs ONE subject for a given (N_COMPONENTS, SCC_STRATEGY, METHOD) config.
#
# Usage:
#   sbatch --array=0-309 slurm_fmri_large.sh <TIMESTAMP> <N_COMP> <SCC> <METHOD>
#
# Examples:
#   sbatch --array=0-309 slurm_fmri_large.sh 03012026120000 20 domain RASL
#   sbatch --array=0-309 slurm_fmri_large.sh 03012026120000 53 none PCMCI
#   sbatch --array=0-309 slurm_fmri_large.sh 03012026120000 10 none GCM
# =============================================================================

TIMESTAMP=$1
N_COMP=${2:-10}
SCC_STRATEGY=${3:-domain}
METHOD=${4:-RASL}

if [ -z "$TIMESTAMP" ]; then
    echo "Error: TIMESTAMP not provided"
    echo "Usage: sbatch --array=0-N slurm_fmri_large.sh <TIMESTAMP> <N_COMP> <SCC> <METHOD>"
    exit 1
fi

SUBJECT_IDX=$SLURM_ARRAY_TASK_ID
CONFIG_TAG="N${N_COMP}_${SCC_STRATEGY}_${METHOD}"

# Directory setup
LOG_DIR="./logs"
mkdir -p "$LOG_DIR" "./err" "./out"

# Master log (created once per array job)
MASTER_LOG="$LOG_DIR/fmri_master_${SLURM_ARRAY_JOB_ID}_${CONFIG_TAG}.log"
if [ ! -f "$MASTER_LOG" ]; then
    {
        echo "=============================================================="
        echo "FMRI LARGE EXPERIMENT - MASTER LOG"
        echo "=============================================================="
        echo "Date:          $(date)"
        echo "Job Array ID:  $SLURM_ARRAY_JOB_ID"
        echo "Timestamp:     $TIMESTAMP"
        echo "Config:        $CONFIG_TAG"
        echo "N Components:  $N_COMP"
        echo "SCC Strategy:  $SCC_STRATEGY"
        echo "Method:        $METHOD"
        echo "=============================================================="
    } > "$MASTER_LOG"
fi

# Per-task log
TASK_LOG="$LOG_DIR/fmri_task_${CONFIG_TAG}_subj${SUBJECT_IDX}.log"
{
    echo "Task ID:      $SLURM_ARRAY_TASK_ID"
    echo "Subject:      $SUBJECT_IDX"
    echo "Config:       $CONFIG_TAG"
    echo "Start Time:   $(date)"
    echo "Hostname:     $HOSTNAME"
    echo "Node:         $SLURM_NODELIST"
} > "$TASK_LOG"

# =============================================================================
# Environment
# =============================================================================
job_name="fmri_${CONFIG_TAG}_${SUBJECT_IDX}"
scontrol update jobid=$SLURM_JOB_ID name=$job_name

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

echo "Activating conda environment..." >&2
source /home/users/mabavisani/anaconda3/bin/activate
conda activate multi_v3

echo "==========================================="
echo "fMRI Large Experiment"
echo "==========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array Task:   $SLURM_ARRAY_TASK_ID"
echo "Subject:      $SUBJECT_IDX"
echo "Config:       $CONFIG_TAG"
echo "==========================================="

cd $SLURM_SUBMIT_DIR

# =============================================================================
# Run experiment
# =============================================================================

# Build RASL-specific args only when needed
EXTRA_ARGS=""
if [ "$METHOD" = "RASL" ]; then
    EXTRA_ARGS="--MAXU 4 --PRIORITY 11112 --selection_mode top_k --top_k 10"
fi

CMD="python fmri_experiment_large.py \
    --subject_idx $SUBJECT_IDX \
    --n_components $N_COMP \
    --scc_strategy $SCC_STRATEGY \
    --method $METHOD \
    --timestamp $TIMESTAMP \
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
