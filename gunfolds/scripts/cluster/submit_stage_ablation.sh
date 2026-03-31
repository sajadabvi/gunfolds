#!/bin/bash
# =============================================================================
# Submit Stage Ablation as SLURM array jobs
# =============================================================================
#
# Usage:
#   bash submit_stage_ablation.sh [OPTIONS]
#
# Options:
#   -n NODE_SIZES      Space-separated node sizes (default: "5 6")
#   -d DENSITIES       Space-separated densities (default: "0.2 0.25")
#   -u UNDERSAMPLING   Space-separated undersampling rates (default: "2 3")
#   -b BATCHES         Number of batches per config (default: 10)
#   -m MAX_PARALLEL    Max simultaneous array tasks (default: 100)
#   -p PARTITION       SLURM partition (default: qTRDGPU)
#   -t TIME_LIMIT      Time limit per task (default: 1:00:00)
#   -s SSIZE           Sample size (default: 5000)
#   --noise NOISE      Noise level (default: 0.1)
#
# Examples:
#   bash submit_stage_ablation.sh
#   bash submit_stage_ablation.sh -n "5 6 7" -d "0.2 0.25 0.3" -b 20
#   bash submit_stage_ablation.sh -p qTRDGPU -m 30
#
# Default configuration: 2 sizes × 2 densities × 2 rates × 10 batches = 80 tasks
# Each task runs all 3 ablation stages: 15 CPUs, 1GB memory, 1 hour
# =============================================================================

# Defaults
NODE_SIZES="5 6"
DENSITIES="0.3 0.4"
UNDERSAMPLING="2 3"
BATCHES=10
MAX_PARALLEL=100
PARTITION="qTRDGPU"
TIME_LIMIT="1:00:00"
SSIZE=5000
NOISE=0.1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n) NODE_SIZES="$2"; shift 2 ;;
        -d) DENSITIES="$2"; shift 2 ;;
        -u) UNDERSAMPLING="$2"; shift 2 ;;
        -b) BATCHES="$2"; shift 2 ;;
        -m) MAX_PARALLEL="$2"; shift 2 ;;
        -p) PARTITION="$2"; shift 2 ;;
        -t) TIME_LIMIT="$2"; shift 2 ;;
        -s) SSIZE="$2"; shift 2 ;;
        --noise) NOISE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Compute total tasks: |node_sizes| × |densities| × |undersampling| × batches
read -ra N_ARR <<< "$NODE_SIZES"
read -ra D_ARR <<< "$DENSITIES"
read -ra U_ARR <<< "$UNDERSAMPLING"
N_SIZES=${#N_ARR[@]}
N_DENS=${#D_ARR[@]}
N_URATES=${#U_ARR[@]}
TOTAL_TASKS=$((N_SIZES * N_DENS * N_URATES * BATCHES))
LAST_IDX=$((TOTAL_TASKS - 1))

TIMESTAMP=$(date +%m%d%Y%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_stage_ablation.sh"

# Resource limits
CPUS=15
MEM="15g"

# Ensure directories exist
mkdir -p ./logs ./err ./out

echo "=============================================================="
echo "STAGE ABLATION - BATCH SUBMISSION"
echo "=============================================================="
echo "Timestamp:       $TIMESTAMP"
echo "Node sizes:      $NODE_SIZES ($N_SIZES)"
echo "Densities:       $DENSITIES ($N_DENS)"
echo "Undersampling:   $UNDERSAMPLING ($N_URATES)"
echo "Batches:         $BATCHES"
echo "Total tasks:     $TOTAL_TASKS (array 0-${LAST_IDX})"
echo "Max parallel:    $MAX_PARALLEL"
echo "Partition:       $PARTITION"
echo "Resources:       ${CPUS} CPUs, ${MEM} mem, ${TIME_LIMIT} time"
echo "Sample size:     $SSIZE"
echo "Noise:           $NOISE"
echo "SLURM script:    $SLURM_SCRIPT"
echo "=============================================================="
echo ""

# Build extra args string to forward to the Python script
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
