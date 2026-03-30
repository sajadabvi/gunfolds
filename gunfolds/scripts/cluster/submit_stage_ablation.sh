#!/bin/bash
# =============================================================================
# Submit Stage Ablation as SLURM array jobs
# =============================================================================
#
# Usage:
#   bash submit_stage_ablation.sh [OPTIONS]
#
# Options:
#   -n NETWORKS        Space-separated network numbers (default: "1 2 3 4 5")
#   -u UNDERSAMPLING   Space-separated undersampling rates (default: "2 3")
#   -b BATCHES         Number of batches per config (default: 10)
#   -m MAX_PARALLEL    Max simultaneous array tasks (default: 20)
#   -p PARTITION       SLURM partition (default: qTRDGPU)
#   -t TIME_LIMIT      Time limit per task (default: 1-00:00:00)
#   -s SSIZE           Sample size (default: 5000)
#   --noise NOISE      Noise level (default: 0.1)
#
# Examples:
#   bash submit_stage_ablation.sh
#   bash submit_stage_ablation.sh -n "1 2 3" -u "2 3 4" -b 20
#   bash submit_stage_ablation.sh -p qTRDGPU -m 30
#
# Default configuration: 5 networks × 2 rates × 10 batches = 100 tasks
# Each task runs all 3 ablation stages: 15 CPUs, 10GB memory
# =============================================================================

# Defaults
NETWORKS="1 2 3 4 5"
UNDERSAMPLING="2 3"
BATCHES=10
MAX_PARALLEL=100
PARTITION="qTRDGPU"
TIME_LIMIT="1-00:00:00"
SSIZE=5000
NOISE=0.1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n) NETWORKS="$2"; shift 2 ;;
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

# Compute total tasks: |networks| × |undersampling| × batches
read -ra NET_ARR <<< "$NETWORKS"
read -ra U_ARR <<< "$UNDERSAMPLING"
N_NETS=${#NET_ARR[@]}
N_URATES=${#U_ARR[@]}
TOTAL_TASKS=$((N_NETS * N_URATES * BATCHES))
LAST_IDX=$((TOTAL_TASKS - 1))

TIMESTAMP=$(date +%m%d%Y%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_stage_ablation.sh"

# Resource limits
CPUS=15
MEM="10g"

# Ensure directories exist
mkdir -p ./logs ./err ./out

echo "=============================================================="
echo "STAGE ABLATION - BATCH SUBMISSION"
echo "=============================================================="
echo "Timestamp:       $TIMESTAMP"
echo "Networks:        $NETWORKS ($N_NETS)"
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
EXTRA_ARGS="-n ${NETWORKS} -u ${UNDERSAMPLING} -b ${BATCHES} --ssize ${SSIZE} --noise ${NOISE}"

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
echo "      -n ${NETWORKS} -u ${UNDERSAMPLING} -b ${BATCHES}"
echo "=============================================================="

# Save submission record
RECORD="./logs/stg_abl_submission_${TIMESTAMP}.log"
{
    echo "Timestamp: $TIMESTAMP"
    echo "Submitted: $(date)"
    echo "Networks: $NETWORKS"
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
    echo "  python stage_ablation.py aggregate --timestamp $TIMESTAMP -n ${NETWORKS} -u ${UNDERSAMPLING} -b ${BATCHES}"
} > "$RECORD"
echo "Submission record: $RECORD"
