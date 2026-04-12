#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8g
#SBATCH -p qTRDHM
#SBATCH -t 0-08:00:00
#SBATCH -J exp4_n20
#SBATCH -e ./err/exp4_n20_error%A-%a.err
#SBATCH -o ./out/exp4_n20_out%A-%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mabavisani@gsu.edu
#SBATCH --oversubscribe

# =============================================================================
# Exp 4: PCMCI Hyperparameter Grid Search — N=20 Components
# =============================================================================
# Runs ONE hyperparameter configuration per array task (48 configs total).
# Each task processes all subjects for its config and writes a partial JSON.
# After the array completes, run the merge step to get the summary table.
#
# USAGE
# -----
# 1. Submit the array (from gunfolds/scripts/real_data/):
#
#       TIMESTAMP=$(date +%m%d%Y%H%M%S)
#       sbatch --array=0-47 slurm_exp4_pcmci_n20.sh $TIMESTAMP
#
# 2. After all tasks finish, merge and print the summary:
#
#       python exp4_pcmci_n20_hyperparam_grid.py \
#           --merge --run_dir results_exp4_$TIMESTAMP
#
# CONFIG INDEX MAPPING (printed at submit time for reference):
#   0-31  : run_pcmci   (4 tau_max x 4 alpha x 2 fdr)
#   32-47 : run_pcmciplus (4 tau_max x 4 pc_alpha)
#
# WALL-TIME ESTIMATE (311 subjects, N=20):
#   PCMCI   tau=1: ~12 min | tau=2: ~25 min | tau=3: ~40 min | tau=4: ~55 min
#   PCMCIplus tau=1-4: 5-15 min
#   8-hour limit is very generous; typical finish in 1-3 hours per task.
# =============================================================================

TIMESTAMP=${1:-$(date +%m%d%Y%H%M%S)}
TASK_ID=$SLURM_ARRAY_TASK_ID
N_TOTAL=48
RUN_DIR="results_exp4_${TIMESTAMP}"

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
LOG_DIR="./logs"
mkdir -p "$LOG_DIR" "./err" "./out" "$RUN_DIR"

# Master log (created once per array job using a lockfile approach)
MASTER_LOG="$LOG_DIR/exp4_master_${SLURM_ARRAY_JOB_ID}.log"
LOCK="${MASTER_LOG}.lock"
if mkdir "$LOCK" 2>/dev/null; then
    {
        echo "=============================================================="
        echo "EXP 4: PCMCI N=20 HYPERPARAM GRID — MASTER LOG"
        echo "=============================================================="
        echo "Date:          $(date)"
        echo "Job Array ID:  $SLURM_ARRAY_JOB_ID"
        echo "Timestamp:     $TIMESTAMP"
        echo "Run dir:       $RUN_DIR"
        echo "Total configs: $N_TOTAL"
        echo "=============================================================="
    } > "$MASTER_LOG"
    rmdir "$LOCK"
fi

# Per-task log
TASK_LOG="$LOG_DIR/exp4_task_${TASK_ID}.log"
{
    echo "Task ID:    $TASK_ID / $((N_TOTAL - 1))"
    echo "Run dir:    $RUN_DIR"
    echo "Start:      $(date)"
    echo "Host:       $HOSTNAME"
    echo "Node:       $SLURM_NODELIST"
} > "$TASK_LOG"

# ---------------------------------------------------------------------------
# Validate task range
# ---------------------------------------------------------------------------
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "$N_TOTAL" ]; then
    echo "ERROR: TASK_ID $TASK_ID outside valid range [0, $((N_TOTAL - 1))]" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
scontrol update jobid="$SLURM_JOB_ID" name="exp4_cfg${TASK_ID}"

export OMP_NUM_THREADS=1   # PCMCI/ParCorr is single-threaded; prevent oversubscription
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/

echo "Activating conda environment..." >&2
source /home/users/mabavisani/anaconda3/etc/profile.d/conda.sh
conda activate multi_v3

echo "==================================================" >&2
echo "Exp 4: PCMCI N=20 hyperparameter grid"           >&2
echo "Job ID:      $SLURM_JOB_ID"                       >&2
echo "Array task:  $TASK_ID / $((N_TOTAL - 1))"         >&2
echo "Run dir:     $RUN_DIR"                             >&2
echo "==================================================" >&2

# ---------------------------------------------------------------------------
# Run the experiment (single config)
# ---------------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"

CMD="python exp4_pcmci_n20_hyperparam_grid.py \
    --task_id $TASK_ID \
    --run_dir $RUN_DIR"

echo "Executing: $CMD" >&2
eval $CMD
EXIT_CODE=$?

# ---------------------------------------------------------------------------
# Log completion
# ---------------------------------------------------------------------------
{
    echo "End:        $(date)"
    echo "Exit code:  $EXIT_CODE"
    echo "Status:     $([ $EXIT_CODE -eq 0 ] && echo SUCCESS || echo FAILED)"
} >> "$TASK_LOG"

# If this is the last task to finish, trigger the merge automatically.
# We check by counting completed partial files; the last writer merges.
N_DONE=$(ls "$RUN_DIR"/cfg_*.json 2>/dev/null | wc -l)
if [ "$N_DONE" -eq "$N_TOTAL" ]; then
    echo "All $N_TOTAL configs done — running merge step..." >&2
    python exp4_pcmci_n20_hyperparam_grid.py \
        --merge --run_dir "$RUN_DIR" \
        >> "$LOG_DIR/exp4_merge_${SLURM_ARRAY_JOB_ID}.log" 2>&1
    echo "Merge complete. See $LOG_DIR/exp4_merge_${SLURM_ARRAY_JOB_ID}.log" >&2
fi

exit $EXIT_CODE
