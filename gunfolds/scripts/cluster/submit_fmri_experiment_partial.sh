#!/bin/bash
# =============================================================================
# Submit a single fMRI experiment config on qTRDGPU only (partial run)
# =============================================================================
#
# Usage:
#   bash submit_fmri_experiment_partial.sh [N_SUBJECTS] [GT_DENSITY_MODE] [VALUE]
#
#   GT_DENSITY_MODE (optional): none (default) | fixed | fraction
#   VALUE (optional): for fixed = 0-1000 (omit for N-specific default: 10→350, 20→215, 53→125);
#                     for fraction = 0-1 (default 0.5)
#
# Example:
#   bash submit_fmri_experiment_partial.sh 310
#   bash submit_fmri_experiment_partial.sh 310 fixed
#   bash submit_fmri_experiment_partial.sh 310 fixed 350
#   bash submit_fmri_experiment_partial.sh 310 fraction 0.5
#
# Runs RASL with N=10, SCC=domain on ALL subjects, using only qTRDGPU.
# Use this for quick partial runs or when only GPU partition is available.
#
# =============================================================================

N_SUBJECTS=${1:-310}
GT_DENSITY_MODE=${2:-}
GT_DENSITY_VALUE=${3:-}
LAST_IDX=$((N_SUBJECTS - 1))
TIMESTAMP=$(date +%m%d%Y%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_fmri_large.sh"

# Resource limits (same as full script)
TIME_LIMIT="5-08:00:00"
CPUS=15
MEM="20g"
MAX_PARALLEL=310

# Single partition: qTRDGPU only
PARTITION="qTRDGPU"
ARRAY_RANGE="0-${LAST_IDX}"

# Config: RASL, N=10, domain
N_COMP=10
SCC=domain
METHOD=RASL
CONFIG_TAG="N${N_COMP}_${SCC}_${METHOD}"

# Ensure log directories exist
mkdir -p ./logs ./err ./out

echo "=============================================================="
echo "FMRI EXPERIMENT - PARTIAL (qTRDGPU only)"
echo "=============================================================="
echo "Timestamp:    $TIMESTAMP"
echo "Subjects:     0-${LAST_IDX} (${N_SUBJECTS} total)"
echo "Partition:    $PARTITION"
echo "Config:       $CONFIG_TAG"
echo "SLURM script: $SLURM_SCRIPT"
echo "Resources:    ${CPUS} CPUs, ${MEM} mem, ${TIME_LIMIT} time"
echo "Parallel:     ${MAX_PARALLEL} tasks"
[ -n "$GT_DENSITY_MODE" ] && echo "GT_density:   mode=$GT_DENSITY_MODE${GT_DENSITY_VALUE:+ value=$GT_DENSITY_VALUE}"
echo "=============================================================="
echo ""

SLURM_ARGS=("$SLURM_SCRIPT" "$TIMESTAMP" "$N_COMP" "$SCC" "$METHOD")
if [ -n "$GT_DENSITY_MODE" ]; then
    SLURM_ARGS+=("$GT_DENSITY_MODE")
    [ "$GT_DENSITY_MODE" = "fixed" ] && [ -n "$GT_DENSITY_VALUE" ] && SLURM_ARGS+=("$GT_DENSITY_VALUE")
    [ "$GT_DENSITY_MODE" = "fraction" ] && SLURM_ARGS+=("${GT_DENSITY_VALUE:-0.5}")
fi

JOB_ID=$(sbatch \
    --array=${ARRAY_RANGE}%${MAX_PARALLEL} \
    --partition=${PARTITION} \
    --time=${TIME_LIMIT} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --job-name="fmri_${CONFIG_TAG}" \
    "${SLURM_ARGS[@]}" \
    | awk '{print $NF}')

echo "Submitted: $CONFIG_TAG on $PARTITION, array=$ARRAY_RANGE, JobID: $JOB_ID"
echo ""
echo "=============================================================="
echo "SUBMISSION COMPLETE"
echo "=============================================================="
echo "Total jobs:     ${N_SUBJECTS} (1 array job)"
echo "Shared timestamp: $TIMESTAMP"
echo "Results will be in: fbirn_results/$TIMESTAMP/"
echo ""
echo "Monitor: squeue -u \$USER"
echo ""
echo "After completion, run analysis:"
echo "  python ../analysis/analyze_fmri_experiment.py --timestamp $TIMESTAMP --plot"
echo "=============================================================="

# Save submission record
RECORD="./logs/submission_partial_${TIMESTAMP}.log"
{
    echo "Timestamp: $TIMESTAMP"
    echo "Submitted: $(date)"
    echo "N_SUBJECTS: $N_SUBJECTS"
    echo "Partition: $PARTITION"
    echo "Config: $CONFIG_TAG (N=$N_COMP, SCC=$SCC, METHOD=$METHOD)"
    [ -n "$GT_DENSITY_MODE" ] && echo "GT_density: mode=$GT_DENSITY_MODE${GT_DENSITY_VALUE:+ value=$GT_DENSITY_VALUE}"
    echo "Job ID: $JOB_ID"
} > "$RECORD"
echo "Submission record: $RECORD"

# Save experiment config markdown in results directory
RESULTS_DIR="fbirn_results/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
CONFIG_MD="${RESULTS_DIR}/experiment_config.md"
{
    echo "# Experiment Run: ${TIMESTAMP}"
    echo ""
    echo "**Script:** \`submit_fmri_experiment_partial.sh\` (partial — qTRDGPU only)"
    echo ""
    echo "## Parameters"
    echo ""
    echo "| Parameter | Value |"
    echo "|-----------|-------|"
    echo "| Timestamp | \`${TIMESTAMP}\` |"
    echo "| Submitted | $(date) |"
    echo "| N Subjects | ${N_SUBJECTS} (indices 0–${LAST_IDX}) |"
    echo "| Configuration | ${CONFIG_TAG} (N=${N_COMP}, SCC=${SCC}, METHOD=${METHOD}) |"
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
    echo "| Partition | ${PARTITION} |"
    echo "| CPUs per task | ${CPUS} |"
    echo "| Memory | ${MEM} |"
    echo "| Time limit | ${TIME_LIMIT} |"
    echo "| Max parallel | ${MAX_PARALLEL} |"
    echo "| Total jobs | ${N_SUBJECTS} (1 array job) |"
    echo ""
    echo "## Job ID"
    echo ""
    echo "\`\`\`"
    echo "${JOB_ID}"
    echo "\`\`\`"
    echo ""
    echo "## SLURM Script"
    echo ""
    echo "\`${SLURM_SCRIPT}\`"
} > "$CONFIG_MD"
echo "Experiment config:  $CONFIG_MD"
