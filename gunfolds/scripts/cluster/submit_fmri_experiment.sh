#!/bin/bash
# =============================================================================
# Submit fMRI RASL experiment configurations as SLURM array jobs
# =============================================================================
#
# Usage:
#   bash submit_fmri_experiment.sh [N_SUBJECTS] [GT_DENSITY_MODE] [VALUE]
#
#   GT_DENSITY_MODE (optional): none (default) | fixed | fraction
#   VALUE (optional): for fixed = 0-1000 (omit for N-specific default: 10→350, 20→215, 53→125);
#                     for fraction = 0-1 (default 0.5)
#
# Example:
#   bash submit_fmri_experiment.sh 310
#   bash submit_fmri_experiment.sh 310 fixed
#   bash submit_fmri_experiment.sh 310 fixed 215
#   bash submit_fmri_experiment.sh 310 fraction 0.5
#
# Total jobs: 620 (310 subjects x 2 configs), no duplication.
# Subjects are split across 3 partitions (qTRDGPU, qTRDHM, qTRD),
# with 10 parallel tasks per partition (30 total parallel).
#
# Configurations:
#   - RASL with N=20 x SCC=domain
#   - RASL with N=53 x SCC=domain
# =============================================================================

N_SUBJECTS=${1:-310}
GT_DENSITY_MODE=${2:-}
GT_DENSITY_VALUE=${3:-}
LAST_IDX=$((N_SUBJECTS - 1))
TIMESTAMP=$(date +%m%d%Y%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_fmri_large.sh"

# Resource limits
TIME_LIMIT="5-08:00:00"
CPUS=15
MEM="230g"
MAX_PARALLEL=10

# Split subjects into 3 roughly equal partitions
CHUNK=$((N_SUBJECTS / 3))
REMAINDER=$((N_SUBJECTS % 3))

# Partition 1 gets the extra subjects if N_SUBJECTS isn't divisible by 3
END1=$((CHUNK + (REMAINDER > 0 ? 1 : 0) - 1))
START2=$((END1 + 1))
END2=$((START2 + CHUNK + (REMAINDER > 1 ? 1 : 0) - 1))
START3=$((END2 + 1))
END3=$((LAST_IDX))

PARTITIONS=("qTRDGPU" "qTRDHM" "qTRD")
RANGE_STARTS=( 0        $START2  $START3 )
RANGE_ENDS=(   $END1    $END2    $END3   )

# Ensure log directories exist
mkdir -p ./logs ./err ./out

echo "=============================================================="
echo "FMRI EXPERIMENT - BATCH SUBMISSION"
echo "=============================================================="
echo "Timestamp:    $TIMESTAMP"
echo "Subjects:     0-${LAST_IDX} (${N_SUBJECTS} total)"
echo "SLURM script: $SLURM_SCRIPT"
echo "Resources:    ${CPUS} CPUs, ${MEM} mem, ${TIME_LIMIT} time"
echo "Parallel:     ${MAX_PARALLEL} per partition (30 total)"
[ -n "$GT_DENSITY_MODE" ] && echo "GT_density:   mode=$GT_DENSITY_MODE${GT_DENSITY_VALUE:+ value=$GT_DENSITY_VALUE}"
echo "=============================================================="
echo "Subject distribution:"
for i in 0 1 2; do
    COUNT=$(( ${RANGE_ENDS[$i]} - ${RANGE_STARTS[$i]} + 1 ))
    echo "  ${PARTITIONS[$i]}:  subjects ${RANGE_STARTS[$i]}-${RANGE_ENDS[$i]}  ($COUNT subjects)"
done
echo "=============================================================="
echo ""

JOB_IDS=()

submit_config() {
    local N_COMP=$1
    local SCC=$2
    local METHOD=$3
    local PARTITION=$4
    local ARRAY_RANGE=$5
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
        --partition=${PARTITION} \
        --time=${TIME_LIMIT} \
        --cpus-per-task=${CPUS} \
        --mem=${MEM} \
        --job-name="fmri_${CONFIG_TAG}" \
        "${SLURM_ARGS[@]}" \
        | awk '{print $NF}')

    JOB_IDS+=("$JOB_ID")
    printf "  %-25s  %-10s  array=%-12s  JobID: %s\n" \
        "$CONFIG_TAG" "$PARTITION" "$ARRAY_RANGE" "$JOB_ID"
}

echo "Submitting RASL domain configurations..."
for i in 0 1 2; do
    PART="${PARTITIONS[$i]}"
    RANGE="${RANGE_STARTS[$i]}-${RANGE_ENDS[$i]}"
    submit_config 20 domain RASL "$PART" "$RANGE"
    submit_config 53 domain RASL "$PART" "$RANGE"
done

TOTAL_JOBS=0
for i in 0 1 2; do
    COUNT=$(( ${RANGE_ENDS[$i]} - ${RANGE_STARTS[$i]} + 1 ))
    TOTAL_JOBS=$(( TOTAL_JOBS + COUNT * 2 ))
done

echo ""
echo "=============================================================="
echo "SUBMISSION COMPLETE"
echo "=============================================================="
echo "Total jobs submitted: ${TOTAL_JOBS} (6 array jobs, no duplicates)"
echo "Shared timestamp:     $TIMESTAMP"
echo "Results will be in:   fbirn_results/$TIMESTAMP/"
echo ""
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "After all jobs complete, run the analysis:"
echo "  python ../analysis/analyze_fmri_experiment.py --timestamp $TIMESTAMP --plot"
echo "=============================================================="

# Save submission record
RECORD="./logs/submission_${TIMESTAMP}.log"
{
    echo "Timestamp: $TIMESTAMP"
    echo "Submitted: $(date)"
    echo "N_SUBJECTS: $N_SUBJECTS"
    echo "Resources: ${CPUS} CPUs, ${MEM} mem, ${TIME_LIMIT} time, ${MAX_PARALLEL} parallel/partition"
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "Subject distribution:"
    for i in 0 1 2; do
        COUNT=$(( ${RANGE_ENDS[$i]} - ${RANGE_STARTS[$i]} + 1 ))
        echo "  ${PARTITIONS[$i]}: subjects ${RANGE_STARTS[$i]}-${RANGE_ENDS[$i]} ($COUNT)"
    done
    echo ""
    echo "Configurations:"
    echo "  RASL:  N=20,53 x SCC=domain"
    [ -n "$GT_DENSITY_MODE" ] && echo "  GT_density: mode=$GT_DENSITY_MODE${GT_DENSITY_VALUE:+ value=$GT_DENSITY_VALUE}"
} > "$RECORD"
echo "Submission record: $RECORD"

# Save experiment config markdown in results directory
RESULTS_DIR="fbirn_results/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
CONFIG_MD="${RESULTS_DIR}/experiment_config.md"
{
    echo "# Experiment Run: ${TIMESTAMP}"
    echo ""
    echo "**Script:** \`submit_fmri_experiment.sh\` (full)"
    echo ""
    echo "## Parameters"
    echo ""
    echo "| Parameter | Value |"
    echo "|-----------|-------|"
    echo "| Timestamp | \`${TIMESTAMP}\` |"
    echo "| Submitted | $(date) |"
    echo "| N Subjects | ${N_SUBJECTS} (indices 0–${LAST_IDX}) |"
    echo "| Configurations | RASL: N=20, N=53 x SCC=domain |"
    if [ -n "$GT_DENSITY_MODE" ]; then
        echo "| GT Density Mode | ${GT_DENSITY_MODE} |"
        [ "$GT_DENSITY_MODE" = "fixed" ] && echo "| GT Density Value | ${GT_DENSITY_VALUE:-(N-specific default)} |"
        [ "$GT_DENSITY_MODE" = "fraction" ] && echo "| GT Density Fraction | ${GT_DENSITY_VALUE:-0.5} |"
    else
        echo "| GT Density Mode | none (default) |"
    fi
    echo "| RASL Params | MAXU=4, PRIORITY=11112, selection_mode=top_k, top_k=10 |"
    echo ""
    echo "## Resources"
    echo ""
    echo "| Resource | Value |"
    echo "|----------|-------|"
    echo "| CPUs per task | ${CPUS} |"
    echo "| Memory | ${MEM} |"
    echo "| Time limit | ${TIME_LIMIT} |"
    echo "| Max parallel per partition | ${MAX_PARALLEL} |"
    echo "| Total jobs | ${TOTAL_JOBS} (6 array jobs) |"
    echo ""
    echo "## Subject Distribution"
    echo ""
    echo "| Partition | Subjects | Count |"
    echo "|-----------|----------|-------|"
    for i in 0 1 2; do
        COUNT=$(( ${RANGE_ENDS[$i]} - ${RANGE_STARTS[$i]} + 1 ))
        echo "| ${PARTITIONS[$i]} | ${RANGE_STARTS[$i]}–${RANGE_ENDS[$i]} | ${COUNT} |"
    done
    echo ""
    echo "## Job IDs"
    echo ""
    echo "\`\`\`"
    echo "${JOB_IDS[*]}"
    echo "\`\`\`"
    echo ""
    echo "## SLURM Script"
    echo ""
    echo "\`${SLURM_SCRIPT}\`"
} > "$CONFIG_MD"
echo "Experiment config:  $CONFIG_MD"
