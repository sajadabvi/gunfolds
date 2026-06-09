#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 15
#SBATCH --mem=160g
#SBATCH -p qTRDGPU
#SBATCH -t 5-00:00:00
#SBATCH -J rfmri_large
#SBATCH -e ./err/rfmri_error%A-%a.err
#SBATCH -o ./out/rfmri_out%A-%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mabavisani@gsu.edu

# =============================================================================
# REFACTORED fMRI Large Experiment - Array Job Script
# =============================================================================
# Runs ONE subject per array task with the cost-band-retention +
# cost-weighted-posterior pipeline (refactored_fmri_experiment_large.py).
#
# Differences vs slurm_fmri_large.sh (legacy):
#   - retention: --selection_mode cost_band --delta_band --max_keep --tau
#     (keeps the full near-optimal band + a per-subject posterior), instead of
#     --selection_mode top_k --top_k 10.
#   - optional --bootstrap B block-bootstrap stability selection (B*solve cost!).
#   - results land in fbirn_results_refactored/<TIMESTAMP>/<CONFIG_TAG>/.
#
# Env overrides (optional): DELTA_BAND (0.5), MAX_KEEP (200), TAU (1.0),
#   BOOTSTRAP (0), BLOCK_LEN (20).
#
# Usage:
#   sbatch --array=0-309%50 refactored_slurm_fmri_large.sh <TIMESTAMP> [N_COMP] [SCC] [METHOD] [GT_MODE] [GT_VALUE]
#   BOOTSTRAP=50 sbatch --array=0-309%20 refactored_slurm_fmri_large.sh "$TS" 13 domain RASL fixed 32
# =============================================================================

TIMESTAMP=$1
N_COMP=${2:-20}
SCC_STRATEGY=${3:-domain}
METHOD=${4:-RASL}

# refactored-pipeline knobs (env-overridable)
DELTA_BAND=${DELTA_BAND:-0.5}
MAX_KEEP=${MAX_KEEP:-200}
TAU=${TAU:-1.0}
BOOTSTRAP=${BOOTSTRAP:-0}
BLOCK_LEN=${BLOCK_LEN:-20}

if [ -z "$TIMESTAMP" ]; then
    echo "Error: TIMESTAMP not provided"
    echo "Usage: sbatch --array=0-309%50 refactored_slurm_fmri_large.sh <TIMESTAMP> [N_COMP] [SCC] [METHOD] [GT_MODE] [GT_VAL]"
    exit 1
fi

SUBJECT_IDX=$SLURM_ARRAY_TASK_ID
CONFIG_TAG="N${N_COMP}_${SCC_STRATEGY}_${METHOD}"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR" "./err" "./out"

MASTER_LOG="$LOG_DIR/rfmri_master_${SLURM_ARRAY_JOB_ID}_${CONFIG_TAG}.log"
if [ ! -f "$MASTER_LOG" ]; then
    {
        echo "=============================================================="
        echo "REFACTORED FMRI LARGE EXPERIMENT - MASTER LOG"
        echo "=============================================================="
        echo "Date:          $(date)"
        echo "Job Array ID:  $SLURM_ARRAY_JOB_ID"
        echo "Timestamp:     $TIMESTAMP"
        echo "Config:        $CONFIG_TAG"
        echo "Retention:     cost_band delta<=${DELTA_BAND} max_keep=${MAX_KEEP} tau=${TAU}"
        echo "Bootstrap:     ${BOOTSTRAP} (block_len=${BLOCK_LEN})"
        echo "=============================================================="
    } > "$MASTER_LOG"
fi

TASK_LOG="$LOG_DIR/rfmri_task_${CONFIG_TAG}_subj${SUBJECT_IDX}.log"
{
    echo "Task ID:    $SLURM_ARRAY_TASK_ID"
    echo "Subject:    $SUBJECT_IDX"
    echo "Config:     $CONFIG_TAG"
    echo "Start:      $(date)"
    echo "Node:       $SLURM_NODELIST"
} > "$TASK_LOG"

job_name="rfmri_${CONFIG_TAG}_${SUBJECT_IDX}"
scontrol update jobid=$SLURM_JOB_ID name=$job_name

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-15}
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

echo "Activating conda environment..." >&2
source /home/users/mabavisani/anaconda3/etc/profile.d/conda.sh
conda activate multi_v3

echo "==========================================="
echo "REFACTORED fMRI Large Experiment"
echo "  Subject: $SUBJECT_IDX  Config: $CONFIG_TAG"
echo "  cost_band delta<=${DELTA_BAND} max_keep=${MAX_KEEP} tau=${TAU} bootstrap=${BOOTSTRAP}"
echo "==========================================="

cd $SLURM_SUBMIT_DIR

PNUM=${SLURM_CPUS_PER_TASK:-15}
EXTRA_ARGS="--PNUM $PNUM"
if [ "$METHOD" = "RASL" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --MAXU 4 --PRIORITY 11112 --tol_low 5 --tol_high 5"
    EXTRA_ARGS="$EXTRA_ARGS --selection_mode cost_band --delta_band ${DELTA_BAND} --max_keep ${MAX_KEEP} --tau ${TAU}"
    EXTRA_ARGS="$EXTRA_ARGS --bootstrap ${BOOTSTRAP} --block_len ${BLOCK_LEN}"
    EXTRA_ARGS="$EXTRA_ARGS --pcmci_method pcmci --pcmci_tau_max 1 --pcmci_alpha 0.05 --pcmci_fdr none"
    if [ -n "${5:-}" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --gt_density_mode $5"
        [ "$5" = "fixed" ] && [ -n "${6:-}" ] && EXTRA_ARGS="$EXTRA_ARGS --gt_density $6"
        [ "$5" = "fraction" ] && EXTRA_ARGS="$EXTRA_ARGS --gt_density_fraction ${6:-0.5}"
    elif [ "$N_COMP" = "20" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --gt_density_mode fixed --gt_density 22"
    fi
fi

CMD="python refactored_fmri_experiment_large.py \
    --subject_idx $SUBJECT_IDX \
    --n_components $N_COMP \
    --scc_strategy $SCC_STRATEGY \
    --method $METHOD \
    --timestamp $TIMESTAMP \
    $EXTRA_ARGS"

echo "Executing: $CMD" >&2
eval $CMD
EXIT_CODE=$?

{
    echo "End:        $(date)"
    echo "Exit Code:  $EXIT_CODE"
    [ $EXIT_CODE -eq 0 ] && echo "Status: SUCCESS" || echo "Status: FAILED ($EXIT_CODE)"
} >> "$TASK_LOG"

exit $EXIT_CODE
