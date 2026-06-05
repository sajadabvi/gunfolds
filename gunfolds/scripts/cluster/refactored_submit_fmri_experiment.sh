#!/bin/bash
# =============================================================================
# Submit the REFACTORED fMRI experiment (cost-band retention + posterior)
# =============================================================================
#
# Mirrors submit_fmri_experiment.sh but drives refactored_slurm_fmri_large.sh
# (-> refactored_fmri_experiment_large.py).  Results land in
# fbirn_results_refactored/<TIMESTAMP>/, analysed by
# analysis/refactored_analyze_fmri_experiment.py.
#
# Usage:
#   bash refactored_submit_fmri_experiment.sh [N_SUBJECTS] [GT_DENSITY_MODE] [VALUE]
#
# Refactored-pipeline knobs via env (forwarded to every task):
#   DELTA_BAND (0.5)  MAX_KEEP (200)  TAU (1.0)  BOOTSTRAP (0)  BLOCK_LEN (20)
#
# Examples:
#   bash refactored_submit_fmri_experiment.sh 310 fixed
#   BOOTSTRAP=50 TAU=1.0 bash refactored_submit_fmri_experiment.sh 310 fixed
# =============================================================================

N_SUBJECTS=${1:-310}
GT_DENSITY_MODE=${2:-}
GT_DENSITY_VALUE=${3:-}
LAST_IDX=$((N_SUBJECTS - 1))
TIMESTAMP=$(date +%m%d%Y%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/refactored_slurm_fmri_large.sh"

# refactored-pipeline knobs (env-overridable, forwarded to tasks)
DELTA_BAND=${DELTA_BAND:-0.5}
MAX_KEEP=${MAX_KEEP:-200}
TAU=${TAU:-1.0}
BOOTSTRAP=${BOOTSTRAP:-0}
BLOCK_LEN=${BLOCK_LEN:-20}

TIME_LIMIT="5-08:00:00"
CPUS=15
MEM="230g"
MAX_PARALLEL=10

CHUNK=$((N_SUBJECTS / 3)); REMAINDER=$((N_SUBJECTS % 3))
END1=$((CHUNK + (REMAINDER > 0 ? 1 : 0) - 1))
START2=$((END1 + 1)); END2=$((START2 + CHUNK + (REMAINDER > 1 ? 1 : 0) - 1))
START3=$((END2 + 1)); END3=$((LAST_IDX))
PARTITIONS=("qTRDGPU" "qTRDHM" "qTRD")
RANGE_STARTS=( 0 $START2 $START3 ); RANGE_ENDS=( $END1 $END2 $END3 )

mkdir -p ./logs ./err ./out

echo "=============================================================="
echo "REFACTORED FMRI EXPERIMENT - BATCH SUBMISSION"
echo "=============================================================="
echo "Timestamp:    $TIMESTAMP"
echo "Subjects:     0-${LAST_IDX} (${N_SUBJECTS})"
echo "Retention:    cost_band delta<=${DELTA_BAND} max_keep=${MAX_KEEP} tau=${TAU}"
echo "Bootstrap:    ${BOOTSTRAP} (block_len=${BLOCK_LEN})"
[ -n "$GT_DENSITY_MODE" ] && echo "GT_density:   mode=$GT_DENSITY_MODE${GT_DENSITY_VALUE:+ value=$GT_DENSITY_VALUE}"
echo "Output:       fbirn_results_refactored/${TIMESTAMP}/"
echo "=============================================================="

EXPORT_VARS="ALL,DELTA_BAND=${DELTA_BAND},MAX_KEEP=${MAX_KEEP},TAU=${TAU},BOOTSTRAP=${BOOTSTRAP},BLOCK_LEN=${BLOCK_LEN}"
JOB_IDS=()

submit_config() {
    local N_COMP=$1 SCC=$2 METHOD=$3 PARTITION=$4 ARRAY_RANGE=$5
    local CONFIG_TAG="N${N_COMP}_${SCC}_${METHOD}"
    local SLURM_ARGS=("$SLURM_SCRIPT" "$TIMESTAMP" "$N_COMP" "$SCC" "$METHOD")
    if [ -n "$GT_DENSITY_MODE" ] && [ "$METHOD" = "RASL" ]; then
        SLURM_ARGS+=("$GT_DENSITY_MODE")
        [ "$GT_DENSITY_MODE" = "fixed" ] && [ -n "$GT_DENSITY_VALUE" ] && SLURM_ARGS+=("$GT_DENSITY_VALUE")
        [ "$GT_DENSITY_MODE" = "fraction" ] && SLURM_ARGS+=("${GT_DENSITY_VALUE:-0.5}")
    fi
    local JOB_ID
    JOB_ID=$(sbatch \
        --array=${ARRAY_RANGE}%${MAX_PARALLEL} \
        --partition=${PARTITION} --time=${TIME_LIMIT} \
        --cpus-per-task=${CPUS} --mem=${MEM} \
        --export="${EXPORT_VARS}" \
        --job-name="rfmri_${CONFIG_TAG}" \
        "${SLURM_ARGS[@]}" | awk '{print $NF}')
    JOB_IDS+=("$JOB_ID")
    printf "  %-25s  %-10s  array=%-12s  JobID: %s\n" "$CONFIG_TAG" "$PARTITION" "$ARRAY_RANGE" "$JOB_ID"
}

echo "Submitting refactored RASL domain configurations..."
for i in 0 1 2; do
    PART="${PARTITIONS[$i]}"; RANGE="${RANGE_STARTS[$i]}-${RANGE_ENDS[$i]}"
    submit_config 20 domain RASL "$PART" "$RANGE"
    submit_config 53 domain RASL "$PART" "$RANGE"
done

echo ""
echo "=============================================================="
echo "SUBMISSION COMPLETE"
echo "=============================================================="
echo "Shared timestamp:  $TIMESTAMP"
echo "Results:           fbirn_results_refactored/$TIMESTAMP/"
echo "Job IDs:           ${JOB_IDS[*]}"
echo ""
echo "After completion, analyse with:"
echo "  python ../analysis/refactored_analyze_fmri_experiment.py \\"
echo "      --timestamp $TIMESTAMP --results_root fbirn_results_refactored \\"
echo "      --tau ${TAU} --correction fdr --plot"
echo "=============================================================="

RECORD="./logs/refactored_submission_${TIMESTAMP}.log"
{
    echo "Timestamp: $TIMESTAMP"
    echo "Submitted: $(date)"
    echo "N_SUBJECTS: $N_SUBJECTS"
    echo "Retention: cost_band delta<=${DELTA_BAND} max_keep=${MAX_KEEP} tau=${TAU}"
    echo "Bootstrap: ${BOOTSTRAP} (block_len=${BLOCK_LEN})"
    echo "Configs: RASL N=20,53 x SCC=domain"
    echo "Job IDs: ${JOB_IDS[*]}"
} > "$RECORD"
echo "Submission record: $RECORD"
