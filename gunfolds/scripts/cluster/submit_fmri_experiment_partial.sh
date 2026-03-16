#!/bin/bash
# =============================================================================
# Submit a single fMRI experiment config on qTRDGPU only (partial run)
# =============================================================================
#
# Usage:
#   bash submit_fmri_experiment_partial.sh [N_SUBJECTS]
#
# Example:
#   bash submit_fmri_experiment_partial.sh 310
#
# Runs RASL with N=10, SCC=domain on ALL subjects, using only qTRDGPU.
# Use this for quick partial runs or when only GPU partition is available.
#
# =============================================================================

N_SUBJECTS=${1:-310}
LAST_IDX=$((N_SUBJECTS - 1))
TIMESTAMP=$(date +%m%d%Y%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_fmri_large.sh"

# Resource limits (same as full script)
TIME_LIMIT="5-08:00:00"
CPUS=15
MEM="230g"
MAX_PARALLEL=10

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
echo "=============================================================="
echo ""

JOB_ID=$(sbatch \
    --array=${ARRAY_RANGE}%${MAX_PARALLEL} \
    --partition=${PARTITION} \
    --time=${TIME_LIMIT} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --job-name="fmri_${CONFIG_TAG}" \
    "$SLURM_SCRIPT" "$TIMESTAMP" "$N_COMP" "$SCC" "$METHOD" \
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
    echo "Job ID: $JOB_ID"
} > "$RECORD"
echo "Submission record: $RECORD"
