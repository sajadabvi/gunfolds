#!/bin/bash
# =============================================================================
# Submit Edge Frequency Calibration (Ruben Data) as SLURM array jobs
# =============================================================================
#
# Usage:
#   bash submit_edge_calibration_ruben.sh [OPTIONS]
#
# Options:
#   -n NETWORKS        Space-separated network numbers (default: "1 2 3 5")
#   -b BATCHES         Number of data files per network (default: 60)
#   -m MAX_PARALLEL    Max simultaneous array tasks (default: 60)
#   -p PARTITION       SLURM partition (default: qTRDGPU)
#   -t TIME_LIMIT      Time limit per task (default: 1-00:00:00)
#   --tau_max TAU      Max lag for PCMCI (default: 1)
#   --maxu MAXU        Max undersampling rate to search (default: 8)
#   --no-concat        Use individual (non-concatenated) data files
#   --include-selfloops  Include self-loops in precision calculation
#
# Examples:
#   bash submit_edge_calibration_ruben.sh
#   bash submit_edge_calibration_ruben.sh -n "1 2 3" -b 30
#   bash submit_edge_calibration_ruben.sh -p qTRDHM -m 40
#
# Default configuration: 4 networks × 60 batches = 240 tasks
# Each task: 15 CPUs, 10GB memory
# =============================================================================

# Defaults
NETWORKS="1 2 3 5"
BATCHES=60
MAX_PARALLEL=60
PARTITION="qTRDGPU"
TIME_LIMIT="1-00:00:00"
TAU_MAX=1
MAXU=8
CONCAT_FLAG=""
SELFLOOP_FLAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n) NETWORKS="$2"; shift 2 ;;
        -b) BATCHES="$2"; shift 2 ;;
        -m) MAX_PARALLEL="$2"; shift 2 ;;
        -p) PARTITION="$2"; shift 2 ;;
        -t) TIME_LIMIT="$2"; shift 2 ;;
        --tau_max) TAU_MAX="$2"; shift 2 ;;
        --maxu) MAXU="$2"; shift 2 ;;
        --no-concat) CONCAT_FLAG="--no-concat"; shift ;;
        --include-selfloops) SELFLOOP_FLAG="--include-selfloops"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Compute total tasks: |networks| × batches
read -ra NET_ARR <<< "$NETWORKS"
N_NETS=${#NET_ARR[@]}
TOTAL_TASKS=$((N_NETS * BATCHES))
LAST_IDX=$((TOTAL_TASKS - 1))

TIMESTAMP=$(date +%m%d%Y%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_edge_calibration_ruben.sh"

# Resource limits
CPUS=15
MEM="10g"

# Ensure directories exist
mkdir -p ./logs ./err ./out

echo "=============================================================="
echo "EDGE FREQUENCY CALIBRATION (RUBEN DATA) - BATCH SUBMISSION"
echo "=============================================================="
echo "Timestamp:       $TIMESTAMP"
echo "Networks:        $NETWORKS ($N_NETS)"
echo "Batches:         $BATCHES"
echo "Total tasks:     $TOTAL_TASKS (array 0-${LAST_IDX})"
echo "Max parallel:    $MAX_PARALLEL"
echo "Partition:       $PARTITION"
echo "Resources:       ${CPUS} CPUs, ${MEM} mem, ${TIME_LIMIT} time"
echo "tau_max:         $TAU_MAX"
echo "maxu:            $MAXU"
echo "Concat:          ${CONCAT_FLAG:-yes (default)}"
echo "Self-loops:      ${SELFLOOP_FLAG:-excluded (default)}"
echo "SLURM script:    $SLURM_SCRIPT"
echo "=============================================================="
echo ""

# Build extra args string to forward to the Python script
EXTRA_ARGS="-n ${NETWORKS} -b ${BATCHES} --tau_max ${TAU_MAX} --maxu ${MAXU} ${CONCAT_FLAG} ${SELFLOOP_FLAG}"

JOB_ID=$(sbatch \
    --array=0-${LAST_IDX}%${MAX_PARALLEL} \
    --partition=${PARTITION} \
    --time=${TIME_LIMIT} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --job-name="ecal_rub" \
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
echo "Results will be:  results_edge_calibration_ruben/${TIMESTAMP}/tasks/"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER -j $JOB_ID"
echo ""
echo "After all tasks complete, aggregate results:"
echo "  python edge_frequency_calibration_ruben.py aggregate \\"
echo "      --timestamp $TIMESTAMP \\"
echo "      -n ${NETWORKS} -b ${BATCHES}"
echo "=============================================================="

# Save submission record
RECORD="./logs/ecal_ruben_submission_${TIMESTAMP}.log"
{
    echo "Timestamp: $TIMESTAMP"
    echo "Submitted: $(date)"
    echo "Networks: $NETWORKS"
    echo "Batches: $BATCHES"
    echo "Total tasks: $TOTAL_TASKS"
    echo "Max parallel: $MAX_PARALLEL"
    echo "Partition: $PARTITION"
    echo "Resources: ${CPUS} CPUs, ${MEM} mem, ${TIME_LIMIT} time"
    echo "tau_max: $TAU_MAX"
    echo "maxu: $MAXU"
    echo "Concat: ${CONCAT_FLAG:-yes}"
    echo "Self-loops: ${SELFLOOP_FLAG:-excluded}"
    echo "Job ID: $JOB_ID"
    echo ""
    echo "Aggregate command:"
    echo "  python edge_frequency_calibration_ruben.py aggregate --timestamp $TIMESTAMP -n ${NETWORKS} -b ${BATCHES}"
} > "$RECORD"
echo "Submission record: $RECORD"
