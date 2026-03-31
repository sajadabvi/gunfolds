#!/bin/bash
# =============================================================================
# Submit Stage Ablation as SLURM array jobs
# =============================================================================
#
# Submits two graph configurations with shared timestamp:
#   Config 1: 4-node graphs, density=0.32, 20 batches
#   Config 2: 5-node graphs, density=0.24, 20 batches
#
# Each config × 2 undersampling rates × 20 batches = 40 tasks per config,
# 80 tasks total.
#
# Usage:
#   bash submit_stage_ablation.sh
# =============================================================================

UNDERSAMPLING="2 3"
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

mkdir -p ./logs ./err ./out

echo "=============================================================="
echo "STAGE ABLATION - BATCH SUBMISSION"
echo "=============================================================="
echo "Timestamp:       $TIMESTAMP"
echo "Config 1:        n=4, d=0.32, 20 batches"
echo "Config 2:        n=5, d=0.24, 20 batches"
echo "Undersampling:   $UNDERSAMPLING"
echo "Max parallel:    $MAX_PARALLEL"
echo "Partition:       $PARTITION"
echo "Resources:       ${CPUS} CPUs, ${MEM} mem, ${TIME_LIMIT} time"
echo "Sample size:     $SSIZE"
echo "Noise:           $NOISE"
echo "=============================================================="
echo ""

# --- Config 1: n=4, d=0.32, 20 batches → 1×1×2×20 = 40 tasks (ids 0-39) ---
ARGS1="-n 4 -d 0.32 -u ${UNDERSAMPLING} -b 20 --ssize ${SSIZE} --noise ${NOISE}"
JOB1=$(sbatch \
    --array=0-39%${MAX_PARALLEL} \
    --partition=${PARTITION} \
    --time=${TIME_LIMIT} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --job-name="stg_abl_n4" \
    "${SLURM_SCRIPT}" "$TIMESTAMP" $ARGS1 \
    | awk '{print $NF}')

echo "Config 1 submitted: job $JOB1 (n=4, d=0.32, 40 tasks)"

# --- Config 2: n=5, d=0.24, 20 batches → 1×1×2×20 = 40 tasks (ids 0-39) ---
ARGS2="-n 5 -d 0.24 -u ${UNDERSAMPLING} -b 20 --ssize ${SSIZE} --noise ${NOISE}"
JOB2=$(sbatch \
    --array=0-39%${MAX_PARALLEL} \
    --partition=${PARTITION} \
    --time=${TIME_LIMIT} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --job-name="stg_abl_n5" \
    "${SLURM_SCRIPT}" "$TIMESTAMP" $ARGS2 \
    | awk '{print $NF}')

echo "Config 2 submitted: job $JOB2 (n=5, d=0.24, 40 tasks)"

echo ""
echo "=============================================================="
echo "SUBMISSION COMPLETE"
echo "=============================================================="
echo "Total tasks:      80 (40 per config)"
echo "Shared timestamp: $TIMESTAMP"
echo "Results will be:  results_stage_ablation/${TIMESTAMP}/tasks/"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER -j $JOB1,$JOB2"
echo ""
echo "After all tasks complete, aggregate EACH config separately:"
echo ""
echo "  cd ../experiments"
echo ""
echo "  python stage_ablation.py aggregate \\"
echo "      --timestamp $TIMESTAMP \\"
echo "      -n 4 -d 0.32 -u ${UNDERSAMPLING} -b 20"
echo ""
echo "  python stage_ablation.py aggregate \\"
echo "      --timestamp $TIMESTAMP \\"
echo "      -n 5 -d 0.24 -u ${UNDERSAMPLING} -b 20"
echo "=============================================================="

# Save submission record
RECORD="./logs/stg_abl_submission_${TIMESTAMP}.log"
{
    echo "Timestamp: $TIMESTAMP"
    echo "Submitted: $(date)"
    echo "Config 1: n=4, d=0.32, 20 batches, job=$JOB1"
    echo "Config 2: n=5, d=0.24, 20 batches, job=$JOB2"
    echo "Undersampling: $UNDERSAMPLING"
    echo "Total tasks: 80"
    echo "Partition: $PARTITION"
    echo "Resources: ${CPUS} CPUs, ${MEM} mem, ${TIME_LIMIT} time"
    echo "Sample size: $SSIZE"
    echo "Noise: $NOISE"
    echo ""
    echo "Aggregate commands:"
    echo "  python stage_ablation.py aggregate --timestamp $TIMESTAMP -n 4 -d 0.32 -u ${UNDERSAMPLING} -b 20"
    echo "  python stage_ablation.py aggregate --timestamp $TIMESTAMP -n 5 -d 0.24 -u ${UNDERSAMPLING} -b 20"
} > "$RECORD"
echo "Submission record: $RECORD"
