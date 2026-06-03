#!/bin/bash
# =============================================================================
# Submit the drasl runtime-scaling experiment as 100 SLURM jobs:
#   N ∈ {8, 10, 12, 14, 18, 20, 24, 30, 42, 54}  ×  10 instances  =  100 jobs.
#   N >= 24 use uniform 6-node SCCs (so 24=4×6, 30=5×6, 42=7×6, 54=9×6).
#
# Single global walltime: 36 hours.  drasl internal timeout: 35 hours
# (1-hour safety margin so the script can write a "timeout" CSV row before
# being killed).
#
# RAM scales with N (drasl grounding footprint grows with N).
#
# Submission pattern: one sbatch per (n_nodes, instance_id) pair, mirroring
# how submit_fmri_experiment.sh dispatches per-config arrays.  We do NOT
# bundle into a single SLURM array because per-N RAM allocations differ.
#
# Usage:
#   bash submit_runtime_scaling.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/runtime_scaling.py"

OUTPUT_DIR="results/runtime_scaling"
LOG_DIR="logs/runtime_scaling"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ─── SLURM constants (match slurm_fmri_large.sh conventions) ─────────────────
PARTITION="qTRDGPU"
ACCOUNT="psy53c17"
EMAIL="mabavisani@gsu.edu"
WALLTIME="36:00:00"
CPUS=16              # 15 clingo threads + 1 buffer
TIMEOUT_HOURS=35

# Per-N RAM tier — drasl grounding footprint scales with N² × urate, so
# larger graphs need progressively more memory.
declare -A MEM_BY_N=(
    [8]="8g"   [10]="8g"
    [12]="32g" [14]="32g"
    [16]="64g"
    [18]="64g" [20]="64g"
    [24]="128g" [30]="128g"
    [42]="192g"
    [54]="256g"
)

N_VALUES=(16)
INSTANCES_PER_N=10

# Set FORCE=1 to disable both skip guards and resubmit everything.
FORCE="${FORCE:-0}"

# Snapshot of currently-queued/running job names for this user (cheap, one squeue call).
RUNNING_NAMES=""
if [ "$FORCE" != "1" ]; then
    RUNNING_NAMES=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
fi

JOB_IDS=()
SKIPPED_DONE=0
SKIPPED_RUNNING=0

for N in "${N_VALUES[@]}"; do
    MEM="${MEM_BY_N[$N]}"
    for ((I=0; I<INSTANCES_PER_N; I++)); do
        JOB_NAME="rt_n${N}_i${I}"
        OUT_LOG="${LOG_DIR}/n${N}_inst${I}.out"
        ERR_LOG="${LOG_DIR}/n${N}_inst${I}.err"
        CSV="${OUTPUT_DIR}/n${N}_inst${I}.csv"

        # Skip 1: a previous run already wrote a completed CSV.
        if [ "$FORCE" != "1" ] && [ -f "$CSV" ] && tail -n 1 "$CSV" | grep -q ',completed$'; then
            printf "  [skip] N=%-2s  inst=%-2s  (already completed)\n" "$N" "$I"
            SKIPPED_DONE=$((SKIPPED_DONE + 1))
            continue
        fi

        # Skip 2: a job with this name is already queued or running.
        if [ "$FORCE" != "1" ] && echo "$RUNNING_NAMES" | grep -qxF "$JOB_NAME"; then
            printf "  [skip] N=%-2s  inst=%-2s  (already in queue: %s)\n" "$N" "$I" "$JOB_NAME"
            SKIPPED_RUNNING=$((SKIPPED_RUNNING + 1))
            continue
        fi

        JOB_ID=$(sbatch \
            --parsable \
            -J "$JOB_NAME" \
            -p "$PARTITION" \
            -A "$ACCOUNT" \
            --mail-type=FAIL \
            --mail-user="$EMAIL" \
            -N 1 -n 1 \
            --cpus-per-task=${CPUS} \
            --mem=${MEM} \
            -t "$WALLTIME" \
            -o "$OUT_LOG" \
            -e "$ERR_LOG" \
            --wrap "#!/bin/bash
                set -e
                export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-${CPUS}}
                export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
                . /home/users/mabavisani/anaconda3/etc/profile.d/conda.sh
                conda activate multi_v3
                cd \$SLURM_SUBMIT_DIR
                python ${RUNNER} \
                    --n_nodes ${N} \
                    --instance_id ${I} \
                    --output_dir ${OUTPUT_DIR} \
                    --timeout_hours ${TIMEOUT_HOURS} \
                    --pnum \${SLURM_CPUS_PER_TASK:-15}
            ")
        JOB_IDS+=("$JOB_ID")
        printf "  submitted N=%-2s  inst=%-2s  mem=%-4s  JobID=%s\n" \
            "$N" "$I" "$MEM" "$JOB_ID"
    done
done

echo ""
echo "=============================================================="
echo "SUBMISSION COMPLETE"
echo "=============================================================="
echo "Total jobs:    ${#JOB_IDS[@]} submitted  (skipped ${SKIPPED_DONE} completed, ${SKIPPED_RUNNING} already-queued)"
echo "Output dir:    ${OUTPUT_DIR}/"
echo "Log dir:       ${LOG_DIR}/"
echo "Walltime/job:  ${WALLTIME}"
echo "Aggregate when done:"
echo "  python ${SCRIPT_DIR}/aggregate_results.py --input_dir ${OUTPUT_DIR}"
echo ""

# Save submission record
RECORD="${LOG_DIR}/submission_$(date +%m%d%Y%H%M%S).log"
{
    echo "Submitted: $(date)"
    echo "Walltime:  ${WALLTIME}"
    echo "CPUs:      ${CPUS}"
    echo "Job IDs:   ${JOB_IDS[*]}"
} > "$RECORD"
echo "Submission record: $RECORD"
