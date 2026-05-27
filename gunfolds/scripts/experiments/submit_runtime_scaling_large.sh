#!/bin/bash
# =============================================================================
# Submit ONE instance each at N=30, N=42, N=54 — the regime where the previous
# 100-job sweep timed out at the old 36h wall.  This script asks for the
# cluster's maximum walltime (5d 8h = 128h) and memory tuned to ~1.6× the
# worst MaxRSS observed in sacct for that N (so we stay comfortably under
# the per-CPU memory ceiling and don't get silently bumped to extra CPUs).
#
# Memory tier (revised — N=54 maxed out, N=30 & N=42 quadrupled):
#   N=30 → 160 GB  (4× the original 40 GB)
#   N=42 → 256 GB  (4× the original 64 GB)
#   N=54 → 480 GB  (close to qTRDGPU node max of 512 GB, leaving ~32 GB for OS)
# CPUs are sized explicitly per-N to keep mem/cpu ≤ 15 GB (qTRDGPU's
# MaxMemPerCPU); without this, SLURM silently bumps --cpus-per-task to
# satisfy the ratio.  Sizing:
#   N=30 → 11 CPUs (160/15 = 10.7)
#   N=42 → 18 CPUs (256/15 = 17.1)
#   N=54 → 32 CPUs (480/15 = 32.0; also = half of a 64-CPU qTRDGPU node)
# All three fit within a single qTRDGPU node (64 CPUs / 512 GB).
#
# drasl internal timeout: 127h (1h safety margin under the 128h SLURM kill,
# so the runner can write a timeout CSV row before being SIGKILL'd).
#
# Uses the new --w_bias_magnitudes flag so VAR-W draws are scale-aware and
# magnitude-biased (see runtime_scaling.py:create_stable_weighted_matrix).
#
# Usage:
#   bash submit_runtime_scaling_large.sh           # uses instance_id=0
#   INSTANCE_ID=3 bash submit_runtime_scaling_large.sh
#   FORCE=1 bash submit_runtime_scaling_large.sh   # skip "already completed" guard
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/runtime_scaling.py"

OUTPUT_DIR="results/runtime_scaling"
LOG_DIR="logs/runtime_scaling"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ─── SLURM constants ─────────────────────────────────────────────────────────
PARTITION="qTRDGPU"
ACCOUNT="psy53c17"
EMAIL="mabavisani@gsu.edu"
WALLTIME="5-08:00:00"   # 5 days 8 hours — partition maximum (qTRD/qTRDGPU/qTRDHM)
TIMEOUT_HOURS=127        # 1h margin under SLURM wall

INSTANCE_ID="${INSTANCE_ID:-0}"
FORCE="${FORCE:-0}"

# Per-N RAM and CPU tiers — N=54 maxed to ~480GB (qTRDGPU node has 512GB);
# N=30 and N=42 are 4× the original sizing.  CPUs are scaled to keep
# mem/cpu ≤ 15 GB so SLURM doesn't silently bump --cpus-per-task.
declare -A MEM_BY_N=(
    [30]="160g"
    [42]="256g"
    [54]="480g"
)
declare -A CPUS_BY_N=(
    [30]=11
    [42]=18
    [54]=32
)
N_VALUES=(30 42 54)

RUNNING_NAMES=""
if [ "$FORCE" != "1" ]; then
    RUNNING_NAMES=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
fi

JOB_IDS=()
SKIPPED_DONE=0
SKIPPED_RUNNING=0

for N in "${N_VALUES[@]}"; do
    MEM="${MEM_BY_N[$N]}"
    CPUS="${CPUS_BY_N[$N]}"
    I="$INSTANCE_ID"
    JOB_NAME="rt_n${N}_i${I}_long"
    OUT_LOG="${LOG_DIR}/n${N}_inst${I}_long.out"
    ERR_LOG="${LOG_DIR}/n${N}_inst${I}_long.err"
    CSV="${OUTPUT_DIR}/n${N}_inst${I}.csv"

    if [ "$FORCE" != "1" ] && [ -f "$CSV" ] && tail -n 1 "$CSV" | grep -q ',completed$'; then
        printf "  [skip] N=%-2s  inst=%-2s  (already completed)\n" "$N" "$I"
        SKIPPED_DONE=$((SKIPPED_DONE + 1))
        continue
    fi

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
        --mail-type=FAIL,END \
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
                --pnum \${SLURM_CPUS_PER_TASK:-15} \
                --w_bias_magnitudes
        ")
    JOB_IDS+=("$JOB_ID")
    printf "  submitted N=%-2s  inst=%-2s  cpus=%-2s  mem=%-5s  walltime=%s  JobID=%s\n" \
        "$N" "$I" "$CPUS" "$MEM" "$WALLTIME" "$JOB_ID"
done

echo ""
echo "=============================================================="
echo "LARGE-N RESUBMISSION COMPLETE"
echo "=============================================================="
echo "Jobs submitted: ${#JOB_IDS[@]}  (skipped ${SKIPPED_DONE} completed, ${SKIPPED_RUNNING} already-queued)"
echo "Walltime/job:   ${WALLTIME}  (drasl --timeout_hours=${TIMEOUT_HOURS})"
echo "Output dir:     ${OUTPUT_DIR}/"
echo "Log dir:        ${LOG_DIR}/"
echo ""

RECORD="${LOG_DIR}/submission_large_$(date +%m%d%Y%H%M%S).log"
{
    echo "Submitted: $(date)"
    echo "Walltime:  ${WALLTIME}"
    echo "Instance:  ${INSTANCE_ID}"
    for N in "${N_VALUES[@]}"; do
        echo "  N=${N}  cpus=${CPUS_BY_N[$N]}  mem=${MEM_BY_N[$N]}"
    done
    echo "Job IDs:   ${JOB_IDS[*]}"
} > "$RECORD"
echo "Submission record: $RECORD"
