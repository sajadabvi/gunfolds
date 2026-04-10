#!/bin/bash
# =============================================================================
# Submit Stage Ablation as SLURM array jobs
# =============================================================================
#
# Runs graphs of size 4 and 5 with density=0.32, 30 batches each.
# 2 sizes × 1 density × 2 undersampling rates × 30 batches = 120 tasks.
#
# Usage:
#   bash submit_stage_ablation.sh
# =============================================================================

NODE_SIZES="4 5"
DENSITIES="0.32"
UNDERSAMPLING="2 3"
BATCHES=30
SSIZE=5000
NOISE=0.1
MAX_PARALLEL=100
PARTITION="qTRDGPU"
TIME_LIMIT="1:00:00"
CPUS=15
MEM="15g"

TIMESTAMP=$(date +%m%d%Y%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_stage_ablation.sh"

# 2 sizes × 1 density × 2 u-rates × 30 batches = 120 tasks
TOTAL_TASKS=120
LAST_IDX=$((TOTAL_TASKS - 1))

mkdir -p ./logs ./err ./out

echo "=============================================================="
echo "STAGE ABLATION - BATCH SUBMISSION"
echo "=============================================================="
echo "Timestamp:       $TIMESTAMP"
echo "Node sizes:      $NODE_SIZES"
echo "Densities:       $DENSITIES"
echo "Undersampling:   $UNDERSAMPLING"
echo "Batches:         $BATCHES"
echo "Total tasks:     $TOTAL_TASKS (array 0-${LAST_IDX})"
echo "Max parallel:    $MAX_PARALLEL"
echo "Partition:       $PARTITION"
echo "Resources:       ${CPUS} CPUs, ${MEM} mem, ${TIME_LIMIT} time"
echo "Sample size:     $SSIZE"
echo "Noise:           $NOISE"
echo "=============================================================="
echo ""

EXTRA_ARGS="-n ${NODE_SIZES} -d ${DENSITIES} -u ${UNDERSAMPLING} -b ${BATCHES} --ssize ${SSIZE} --noise ${NOISE}"

JOB_ID=$(sbatch \
    --array=0-${LAST_IDX}%${MAX_PARALLEL} \
    --partition=${PARTITION} \
    --time=${TIME_LIMIT} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --job-name="stg_abl" \
    "${SLURM_SCRIPT}" "$TIMESTAMP" $EXTRA_ARGS \
    | awk '{print $NF}')

echo "Submitted array job: $JOB_ID"
echo "  Array range: 0-${LAST_IDX} (max ${MAX_PARALLEL} parallel)"
echo ""
echo "=============================================================="
echo "SUBMISSION COMPLETE"
echo "=============================================================="
echo "Total tasks:      $TOTAL_TASKS"
echo "Shared timestamp: $TIMESTAMP"
echo "Results will be:  results_stage_ablation/${TIMESTAMP}/tasks/"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER -j $JOB_ID"
echo ""
echo "After all tasks complete, aggregate results:"
echo "  python stage_ablation.py aggregate \\"
echo "      --timestamp $TIMESTAMP \\"
echo "      -n ${NODE_SIZES} -d ${DENSITIES} -u ${UNDERSAMPLING} -b ${BATCHES}"
echo "=============================================================="

# Save submission record
RECORD="./logs/stg_abl_submission_${TIMESTAMP}.log"
{
    echo "Timestamp: $TIMESTAMP"
    echo "Submitted: $(date)"
    echo "Node sizes: $NODE_SIZES"
    echo "Densities: $DENSITIES"
    echo "Undersampling: $UNDERSAMPLING"
    echo "Batches: $BATCHES"
    echo "Total tasks: $TOTAL_TASKS"
    echo "Max parallel: $MAX_PARALLEL"
    echo "Partition: $PARTITION"
    echo "Resources: ${CPUS} CPUs, ${MEM} mem, ${TIME_LIMIT} time"
    echo "Sample size: $SSIZE"
    echo "Noise: $NOISE"
    echo "Job ID: $JOB_ID"
    echo ""
    echo "Aggregate command:"
    echo "  python stage_ablation.py aggregate --timestamp $TIMESTAMP -n ${NODE_SIZES} -d ${DENSITIES} -u ${UNDERSAMPLING} -b ${BATCHES}"
} > "$RECORD"
echo "Submission record: $RECORD"
