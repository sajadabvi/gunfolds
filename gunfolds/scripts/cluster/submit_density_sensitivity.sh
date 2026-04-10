#!/bin/bash
# =============================================================================
# Submit Density Sensitivity as SLURM array jobs
# =============================================================================
#
# Runs ringmore graphs of 5 and 6 nodes with 1,2,3 extra edges,
# undersampling rates 2,3, and 10 batches each.
# 2 sizes × 3 extra_edges × 2 undersampling rates × 10 batches = 120 tasks.
#
# Usage:
#   bash submit_density_sensitivity.sh
# =============================================================================

NODE_SIZES="5 6"
EXTRA_EDGES="1 2 3"
UNDERSAMPLING="2 3"
BATCHES=10
SSIZE=5000
NOISE=0.1
MAX_PARALLEL=100
PARTITION="qTRDGPU"
TIME_LIMIT="1:00:00"
CPUS=15
MEM="15g"

TIMESTAMP=$(date +%m%d%Y%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_density_sensitivity.sh"

# 2 sizes × 3 extra_edges × 2 u-rates × 10 batches = 120 tasks
TOTAL_TASKS=120
LAST_IDX=$((TOTAL_TASKS - 1))

mkdir -p ./logs ./err ./out

echo "=============================================================="
echo "DENSITY SENSITIVITY - BATCH SUBMISSION"
echo "=============================================================="
echo "Timestamp:       $TIMESTAMP"
echo "Node sizes:      $NODE_SIZES"
echo "Extra edges:     $EXTRA_EDGES"
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

EXTRA_ARGS="-n ${NODE_SIZES} -e ${EXTRA_EDGES} -u ${UNDERSAMPLING} -b ${BATCHES} --ssize ${SSIZE} --noise ${NOISE}"

JOB_ID=$(sbatch \
    --array=0-${LAST_IDX}%${MAX_PARALLEL} \
    --partition=${PARTITION} \
    --time=${TIME_LIMIT} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --job-name="dns_sns" \
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
echo "Results will be:  results_density_sensitivity/${TIMESTAMP}/tasks/"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER -j $JOB_ID"
echo ""
echo "After all tasks complete, aggregate results:"
echo "  python density_sensitivity.py aggregate \\"
echo "      --timestamp $TIMESTAMP \\"
echo "      -n ${NODE_SIZES} -e ${EXTRA_EDGES} -u ${UNDERSAMPLING} -b ${BATCHES}"
echo "=============================================================="

# Save submission record
RECORD="./logs/dns_sns_submission_${TIMESTAMP}.log"
{
    echo "Timestamp: $TIMESTAMP"
    echo "Submitted: $(date)"
    echo "Node sizes: $NODE_SIZES"
    echo "Extra edges: $EXTRA_EDGES"
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
    echo "  python density_sensitivity.py aggregate --timestamp $TIMESTAMP -n ${NODE_SIZES} -e ${EXTRA_EDGES} -u ${UNDERSAMPLING} -b ${BATCHES}"
} > "$RECORD"
echo "Submission record: $RECORD"
