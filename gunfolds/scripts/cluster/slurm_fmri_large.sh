#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 15
#SBATCH --mem=160g
#SBATCH -p qTRDGPU
#SBATCH -t 2-00:00:00
#SBATCH -J fmri_large
#SBATCH -e ./err/fmri_error%A-%a.err
#SBATCH -o ./out/fmri_out%A-%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mabavisani@gsu.edu

# =============================================================================
# fMRI Large Experiment - Array Job Script
# =============================================================================
# Runs ONE subject per array task (parallel subjects).  Default workflow:
#   N=20, domain SCC, RASL only, MAXU 5, PRIORITY 11112, PCMCI seeding from Exp4
#   best config (pcmci tau_max=1 alpha=0.05 fdr none), GT_density fixed 22.
#
# Resources: qTRDGPU, 15 CPUs, 160G RAM, 2-day wall clock.
#
# Usage:
#   sbatch --array=0-309%50 slurm_fmri_large.sh <TIMESTAMP> [N_COMP] [SCC] [METHOD] [GT_MODE] [GT_VALUE]
#
#   --array=0-309%50  -> 310 subjects (indices 0-309), max 50 concurrent tasks.
#   GT_MODE: omit (N=20 defaults to fixed 22) | none | fixed | fraction
#
# Examples:
#   TIMESTAMP=$(date +%m%d%Y%H%M%S)
#   sbatch --array=0-309%50 slurm_fmri_large.sh "$TIMESTAMP" 20 domain RASL
#   sbatch --array=0-309%50 slurm_fmri_large.sh "$TIMESTAMP" 20 domain RASL fixed 22
#   sbatch --array=0-309%50 slurm_fmri_large.sh "$TIMESTAMP" 20 domain RASL fraction 0.5
# =============================================================================

TIMESTAMP=$1
N_COMP=${2:-20}
SCC_STRATEGY=${3:-domain}
METHOD=${4:-RASL}

if [ -z "$TIMESTAMP" ]; then
    echo "Error: TIMESTAMP not provided"
    echo "Usage: sbatch --array=0-309%50 slurm_fmri_large.sh <TIMESTAMP> [N_COMP] [SCC] [METHOD] [GT_MODE] [GT_VAL]"
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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-15}
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

echo "Activating conda environment..." >&2
source /home/users/mabavisani/anaconda3/etc/profile.d/conda.sh
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
PNUM=${SLURM_CPUS_PER_TASK:-15}
EXTRA_ARGS="--PNUM $PNUM"
if [ "$METHOD" = "RASL" ]; then
    # Exp4 N=20 hyperparam grid best PCMCI seed: run_pcmci tau=1 alpha=0.05 no FDR
    EXTRA_ARGS="$EXTRA_ARGS --MAXU 5 --PRIORITY 11112 --selection_mode top_k --top_k 10"
    EXTRA_ARGS="$EXTRA_ARGS --pcmci_method pcmci --pcmci_tau_max 1 --pcmci_alpha 0.05 --pcmci_fdr none"
    # GT_density: $5/$6 override; if omitted and N=20, use fixed 22 (density×100, literature default)
    if [ -n "${5:-}" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --gt_density_mode $5"
        [ "$5" = "fixed" ] && [ -n "${6:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --gt_density $6"
        [ "$5" = "fraction" ] && EXTRA_ARGS="$EXTRA_ARGS --gt_density_fraction ${6:-0.5}"
    elif [ "$N_COMP" = "20" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --gt_density_mode fixed --gt_density 22"
    fi
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
